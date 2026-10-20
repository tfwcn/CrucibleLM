"""DSpark 草稿器基准：接受率 / 端到端速度 / 成本曲线标定.

三件事，正好对应论文 §2.4.3 的三块：

1. **接受率**（草稿器好不好）：`hit@i` = 第 i 位贪心命中率、前缀存活率；
2. **端到端速度**（投机值不值）：贪心 vs DSpark（两种调度口径）的 tokens/s；
3. **成本曲线**（调度器参数）：实测"草稿一步 / 验证一步"的相对耗时，
   `dspark_draft_cost` 就该填这个比值——填错只影响选多长，不影响正确性。

用法：
    python scripts/bench_dspark.py --model data/llm-sft7 --dspark data/llm-sft7/dspark.pt \\
        --prompt "摸鱼是一门艺术" --new-tokens 64
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_dspark import load_backbone  # noqa: E402
from src.llm.local.dspark import attach_dspark  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """命令行参数（模型/草稿器/生成长度/重复次数）."""
    p = argparse.ArgumentParser(description="DSpark 草稿器基准")
    p.add_argument("--model", required=True, help="主干 run 目录或 model.pt")
    p.add_argument("--dspark", default="", help="dspark.pt（不给就只测主干）")
    p.add_argument("--prompt", default="摸鱼是一门艺术", help="prompt 文本")
    p.add_argument("--new-tokens", type=int, default=64, help="生成 token 数")
    p.add_argument("--runs", type=int, default=3, help="每种模式重复次数（取中位）")
    p.add_argument("--window", type=int, default=5, help="每轮起草窗口（含主干那颗）")
    p.add_argument("--device", default="", help="cuda / cpu（默认自动）")
    return p.parse_args(argv)


def timed_generate(model, ids, n: int, runs: int) -> tuple[float, torch.Tensor]:
    """跑 runs 次生成，返回中位耗时（秒）与最后一次结果."""
    times, out = [], None
    with torch.no_grad():
        for _ in range(max(1, runs)):
            t0 = time.time()
            out = model.generate(ids, max_new_tokens=n)
            times.append(time.time() - t0)
    return statistics.median(times), out


def main(argv=None) -> int:
    """基准入口：三段测量 + 打印标定建议."""
    args = parse_args(argv)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_backbone(args)
    if args.dspark:
        meta = attach_dspark(model, args.dspark)
        print(f"草稿器已挂载（step {meta.get('step')}）："
              f"{model.config.dspark_layers} 层 / 窗 {model.config.dspark_window} / "
              f"并行 {model.config.dspark_depth} 位", flush=True)
    model.to(dev).eval()
    ids = torch.tensor([tok.encode(args.prompt)[:512]], device=dev)
    n = int(args.new_tokens)

    # ---- 1) 贪心基线 ----
    model.config.draft_enabled = False
    t_greedy, ref = timed_generate(model, ids, n, args.runs)
    print(f"贪心：{n / t_greedy:.1f} tok/s（{t_greedy:.2f}s / {n} tok）", flush=True)

    if model.dspark is None:
        print("未挂草稿器：只报告主干速度（--dspark 指定 dspark.pt 可测投机）")
        return 0

    # ---- 2) 成本曲线：草稿一步 vs 验证一步 ----
    win = max(2, int(args.window))
    ctx = ids[0].tolist()[-model.config.dspark_window:]
    h_last = model.embed(ids[:, -1:])
    y = torch.tensor([ctx[-win:]], device=dev)
    tgt_times, dr_times = [], []
    with torch.no_grad():
        for _ in range(max(3, args.runs)):
            t0 = time.time()
            model.dspark(h_last, torch.tensor([ctx], device=dev), model.embed)
            dr_times.append(time.time() - t0)
            # 验证一步 = `win` 个位置的一次并行前向；预填充不计时（会被摊销）
            _h, pasts = model._prefill(ids, max_new_tokens=win)
            t0 = time.time()
            model._decode_multi(y, pasts)
            tgt_times.append(time.time() - t0)
    c_draft = statistics.median(dr_times) / statistics.median(tgt_times)
    print(f"成本比：草稿一步 / 验证一步 = {c_draft:.2f}"
          f"（→ --dspark-draft-cost 建议 {c_draft:.2f}）", flush=True)

    # ---- 3) DSpark（两种调度口径）+ 接受率 ----
    for schedule in ("confidence", "throughput"):
        model.config.draft_enabled = True
        model.config.draft_source = "dspark"
        model.config.draft_window = win
        model.config.dspark_enabled = True
        model.config.dspark_schedule = schedule
        model.config.dspark_draft_cost = c_draft
        t_spec, out = timed_generate(model, ids, n, args.runs)
        st = getattr(model, "_spec_stats", {})
        same = torch.equal(out, ref)
        acc = st.get("accepted", 0)
        rounds = st.get("rounds", 1)
        print(f"DSpark/{schedule}：{n / t_spec:.1f} tok/s"
              f"（{t_spec:.2f}s，{n / t_greedy / (n / t_spec):.2f}x 相对贪心）"
              f"｜接受 {acc}/{rounds} 轮（每轮 {acc / max(1, rounds):.2f} 个）"
              f"｜提议 {st.get('proposed', 0)} 验证 {st.get('verified', 0)}"
              f"｜与贪心一致 {same}", flush=True)
        if not same:
            print("  ⚠ 与贪心不一致：这是正确性 bug（验证必须精确），请先修")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
