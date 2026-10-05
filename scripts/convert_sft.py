"""SFT 转格式 — HF 数据集 -> Belle 三元组 jsonl（instruction/input/output）.

用法：
  python scripts/convert_sft.py --dataset BelleGroup/train_0.5M_CN \\
      --out data/sft-belle --dedup-dir data/sft-zh

去重：与 --dedup-dir 下现有 SFT 按 instruction 精确去重（防重复分布），
空 instruction/output、超短 output 按阈值丢弃。输出 data/sft-belle/*.jsonl，
可直接 --local-path 混合训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="SFT 转 Belle jsonl")
    p.add_argument("--dataset", required=True, help="HF 数据集名")
    p.add_argument("--split", default="train", help="切分")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--dedup-dir", default="", help="去重参照 SFT 目录（空=不去重）")
    p.add_argument("--min-out-len", type=int, default=20, help="output 最短字符")
    p.add_argument("--limit", type=int, default=0, help="最多行数（0=全量）")
    p.add_argument("--cache-dir", default="data/_dl/hf", help="HF 缓存目录")
    p.add_argument("--endpoint", default="https://hf-mirror.com",
                   help="HF 镜像（直连超时用）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """转格式主流程."""
    import os

    from datasets import load_dataset

    args = parse_args(argv)
    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    print(f"源数据 {len(ds)} 行，列 {ds.column_names}", flush=True)

    seen: set[str] = set()
    if args.dedup_dir:
        from src.llm.local.data import iter_local_sft

        for instruction, _inp, _out in iter_local_sft(args.dedup_dir):
            seen.add(instruction.strip())
        print(f"参照去重基 {len(seen)} 条 instruction", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_dup = n_short = 0
    fp = open(out / "part-0000.jsonl", "w", encoding="utf-8")
    for i, row in enumerate(ds):
        if args.limit and n_ok >= args.limit:
            break
        ins = (row.get("instruction") or "").strip()
        out_text = (row.get("output") or "").strip()
        if not ins or not out_text:
            n_short += 1
            continue
        if len(out_text) < args.min_out_len:
            n_short += 1
            continue
        if ins in seen:
            n_dup += 1
            continue
        seen.add(ins)
        fp.write(json.dumps({"instruction": ins,
                             "input": (row.get("input") or "").strip(),
                             "output": out_text},
                            ensure_ascii=False) + "\n")
        n_ok += 1
        if n_ok % 100000 == 0:
            print(f"已写 {n_ok} 行…", flush=True)
    fp.close()
    print(f"完成：保留 {n_ok}，去重 {n_dup}，丢弃(空/短) {n_short} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
