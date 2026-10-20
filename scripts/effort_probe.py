"""推理努力档探针：同一个 checkpoint 上测"输出长度 / 命中率"随 b 的曲线.

对应 V4.1 §5.1.4 的部署侧承诺：**一份权重**，靠标量 b 在成本–质量曲线上移动。
本脚本把这条曲线**实测出来**（而不是假设它成立），并给出 τ 的标定参考。

用法：
    # 内置小样本（冒烟用）
    python scripts/effort_probe.py --model data/llm-sft7 --levels 25,50,75,100

    # 自带题面（每行 {"prompt": "...", "answer": "..."}；answer 可省）
    python scripts/effort_probe.py --model runs/xxx --prompts p.jsonl \\
        --levels low,high,max --new-tokens 256

输出：
    effort | 平均输出 token | 相对 50 档倍数 | 命中率 | 说明
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_dspark import load_backbone  # noqa: E402
from src.llm.local.effort import (EFFORT_TIERS, effort_to_tier,  # noqa: E402
                                  parse_effort_levels)
from src.llm.local.infer import LocalChatBackend  # noqa: E402

# 内置题面：短、可判定、覆盖"算/答/解释"三种长度需求
DEFAULT_PROMPTS = [
    {"prompt": "1+1 等于几？只回答数字。", "answer": "2"},
    {"prompt": "用一句话解释什么是摸鱼。"},
    {"prompt": "写下三个质数，用逗号分隔。", "answer": "2"},
]


def parse_args(argv=None) -> argparse.Namespace:
    """命令行参数（模型/档位/题面/生成长度/每题重复次数）."""
    p = argparse.ArgumentParser(description="推理努力档探针")
    p.add_argument("--model", required=True, help="主干 run 目录或 model.pt")
    p.add_argument("--levels", default="25,50,75,100",
                   help="训练过的档位（逗号分隔；也接受 low,high,max）")
    p.add_argument("--prompts", default="", help="题面 jsonl（缺省用内置 3 条）")
    p.add_argument("--new-tokens", type=int, default=128, help="最大输出 token")
    p.add_argument("--runs", type=int, default=1, help="每题每个档位重复次数")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度（0=贪心）")
    p.add_argument("--device", default="", help="cuda / cpu（默认自动）")
    return p.parse_args(argv)


def load_prompts(path: str) -> list[dict]:
    """读题面 jsonl；空路径用内置题面."""
    if not path:
        return [dict(x) for x in DEFAULT_PROMPTS]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"题面文件 {path} 里没有有效行")
    return out


def main(argv=None) -> int:
    """探针入口：按档位逐题生成，打印长度/命中率表."""
    args = parse_args(argv)
    levels = parse_effort_levels(args.levels)
    if not levels:
        raise SystemExit("--levels 为空：至少要有一个档位")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_backbone(args)
    model.to(dev).eval()
    backend = LocalChatBackend(model, tok)
    prompts = load_prompts(args.prompts)
    print(f"模型 {args.model}｜档位 {levels}｜{len(prompts)} 题 × {args.runs} 次"
          f"｜最多 {args.new_tokens} token", flush=True)

    rows = []
    for b in levels:
        n_tok, hits, judged, samples = 0, 0, 0, 0
        for item in prompts:
            text = item.get("prompt") or item.get("question") or ""
            answer = item.get("answer")
            for _ in range(max(1, args.runs)):
                resp = backend.chat([{"role": "user", "content": text}],
                                    max_new_tokens=args.new_tokens,
                                    temperature=args.temperature,
                                    effort=b)
                content = resp.get("content") or ""
                n = len(tok.encode(content))
                n_tok += n
                samples += 1
                if answer:                       # 给了答案才算命中
                    judged += 1
                    hits += int(str(answer).strip() in content)
        avg = n_tok / max(1, samples)
        acc = (hits / judged) if judged else None
        rows.append({"effort": b, "avg_tokens": avg, "acc": acc,
                     "samples": samples})
        tier = effort_to_tier(b)
        print(f"  b={b:>3}{f'({tier})' if tier else '     '}｜平均 {avg:6.1f} token"
              f"｜命中 {('—' if acc is None else f'{acc * 100:5.1f}%')}",
              flush=True)

    base = next((r for r in rows if r["effort"] == EFFORT_TIERS["low"]),
                rows[0])
    print("\n档位 × 相对成本（以最低档为 1.0）")
    for r in rows:
        rel = r["avg_tokens"] / max(1e-9, base["avg_tokens"])
        print(f"  b={r['effort']:>3}｜×{rel:4.2f}")

    # 单调性提示：论文的承诺是"努力越高、推理越长"；没训过就必然不单调
    toks = [r["avg_tokens"] for r in rows]
    monotone = all(a <= b + 1e-9 for a, b in zip(toks, toks[1:]))
    print(f"\n长度单调性：{'符合（越高档越长）' if monotone else '不符合'}")
    if not monotone:
        print("  提示：努力档是**训练出来的**（§5.1.4 的 RL）。没做过努力档条件训练的\n"
              "  checkpoint 不会因为多一行系统提示就变长——先按 docs/details/reasoning-effort.md\n"
              "  的配方训练，再用本表验证。")
    if len(rows) >= 2:
        slope = (toks[-1] - toks[0]) / max(1, rows[-1]["effort"] - rows[0]["effort"])
        print(f"实测长度斜率 ≈ {slope:.2f} token/档（eq.16 的 s_x·τ；"
              f"标定 τ 时除以自己的 s_x）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
