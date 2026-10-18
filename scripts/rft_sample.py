"""RFT 采样过滤 — 用冠军权重采样，留下整句，转 Belle jsonl 供下轮 SFT.

用法：
  python scripts/rft_sample.py --model data/llm-sft-hyper/champion/model \\
      --vocab data/llm-sft-hyper/champion/vocab.json \\
      --prompts data/sft-zh --out data/sft-rft --n-prompts 200 --n-samples 2

过滤器（只留整句）：复读率 rep4<0.25、长度 50~600、`<`占比<0.03。
全量（含分数）存 all.jsonl（以后 DPO 的偏好对直接从这里出，不用重采）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="RFT 采样过滤")
    p.add_argument("--model", required=True, help="权重前缀（model.pt 同目录）")
    p.add_argument("--vocab", required=True, help="词表 vocab.json")
    p.add_argument("--prompts", required=True, help="SFT 语料目录（抽 instruction）")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--n-prompts", type=int, default=200, help="抽多少 prompt")
    p.add_argument("--n-samples", type=int, default=2, help="每 prompt 采几条")
    p.add_argument("--max-new-tokens", type=int, default=256, help="每条最大长度")
    p.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    p.add_argument("--penalty", type=float, default=1.2, help="重复惩罚")
    p.add_argument("--min-len", type=int, default=50, help="保留最短")
    p.add_argument("--max-len", type=int, default=600, help="保留最长")
    p.add_argument("--max-rep4", type=float, default=0.25, help="复读率上限")
    p.add_argument("--max-lt", type=float, default=0.03, help="< 占比上限")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--device", default="", help="设备（空=自动）")
    p.add_argument("--memory-every", type=int, default=4, help="记忆层间隔（与权重一致）")
    p.add_argument("--retro-every", type=int, default=4, help="RETRO 间隔（与权重一致）")
    p.add_argument("--hyper-streams", type=int, default=2, help="超连接流数（与权重一致）")
    p.add_argument("--preset", choices=["tiny", "base"], default="base",
                   help="模型规格（tiny 测试用，base 实跑）")
    return p.parse_args(argv)


def rep4(text: str) -> float:
    """4-gram 复读率（judge 同口径）."""
    grams = [text[i:i + 4] for i in range(max(len(text) - 3, 1))]
    return 1 - len(set(grams)) / max(len(grams), 1)


def main(argv=None) -> int:
    """采样过滤主流程."""
    import random

    import torch

    from src.llm.local.config import SmallLLMConfig
    from src.llm.local.data import iter_local_sft
    from src.llm.local.infer import LocalChatBackend, SimpleTokenizer

    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tok = SimpleTokenizer()
    tok._chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    tok.vocab_size = len(tok._chars) + 4
    from src.llm.local.config import (
        SmallLLMConfig,
        sync_csa_reuse_from_checkpoint,
        tiny_test_config,
    )

    cfg = tiny_test_config() if args.preset == "tiny" else SmallLLMConfig()
    # 按来源权重对齐 CSA2（该开关不改 state_dict 键，装错不报错只会静默变语义）
    sync_csa_reuse_from_checkpoint(cfg, args.model)
    tok.vocab_size = len(tok._chars) + 4  # 文件只存 _chars，容量另算
    # cfg.vocab_size 保持预设容量（8192）：训练扩词复用空行不改形状，
    # 建模必须同宽，否则 embed/lm_head 严格载入必炸（init_memory.py 同规则）
    cfg.memory_every = args.memory_every
    if args.retro_every > 0:
        cfg.retro_enabled = True
        cfg.retro_every = args.retro_every
    cfg.hyper_streams = args.hyper_streams
    be = LocalChatBackend.load(args.model, cfg)
    be.model = be.model.to(device).eval()
    print(f"权重已载入：{args.model}（{device}）", flush=True)

    prompts = []
    for ins, inp, _out in iter_local_sft(args.prompts):
        if len(ins) >= 8:
            prompts.append((ins, inp or ""))
        if len(prompts) >= args.n_prompts:
            break
    print(f"prompt {len(prompts)} 条，每条采 {args.n_samples} 遍", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, total = 0, 0
    done = set()
    if (out / "all.jsonl").exists():
        import json as _json2

        for line in open(out / "all.jsonl", encoding="utf-8"):
            try:
                r = _json2.loads(line)
                done.add((r["instruction"], r.get("input", "")))
            except (ValueError, KeyError):
                continue
        print(f"断点续采：跳过已采 {len(done)} 对 prompt", flush=True)
    fall = open(out / "all.jsonl", "a", buffering=1, encoding="utf-8")
    fkeep = open(out / "kept.jsonl", "a", buffering=1, encoding="utf-8")
    for ins, inp in prompts:
        if (ins, inp) in done:
            continue
        for _ in range(args.n_samples):
            total += 1
            r = be.chat([{"role": "user", "content": ins}],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.penalty)
            text = (r["content"] or "").strip()
            n = len(text)
            m = {"rep4": round(rep4(text), 4) if n else 1.0,
                 "lt": round(text.count("<") / n, 4) if n else 1.0}
            fall.write(json.dumps({"instruction": ins, "input": inp,
                                   "output": text, **m}, ensure_ascii=False) + "\n")
            if args.min_len <= n <= args.max_len and m["rep4"] <= args.max_rep4 \
                    and m["lt"] <= args.max_lt:
                fkeep.write(json.dumps({"instruction": ins, "input": inp,
                                        "output": text}, ensure_ascii=False) + "\n")
                kept += 1
            if total % 50 == 0:
                print(f"已采 {total} 条，保留 {kept} 条…", flush=True)
    fall.close()
    fkeep.close()
    print(f"完成：采样 {total}，保留 {kept}（{kept / max(total, 1):.1%}）-> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
