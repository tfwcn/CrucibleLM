"""构建 BM25 检索库 — 扫本地语料目录，pickle 落盘（供 RETRO 用）.

用法：python scripts/build_retrieval.py --data data/hq-zh --out data/retro.bm25
语料走 iter_local_texts（txt/md/jsonl/parquet）；长文按段落切块，
块大小 ~600 字符，重叠 100，供粒度检索。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从仓库任意目录都能 python scripts/build_retrieval.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iter_chunks(root: str, chunk: int = 600, overlap: int = 100, sft: bool = False):
    """按段落切块，块大小约 chunk，重叠 overlap（防跨块知识断裂）.

    sft=True 时走 iter_local_sft（instruction/input/output 拼文本），
    否则走 iter_local_texts（纯文 .txt/.md/.jsonl/parquet）。
    """
    from src.llm.local.data import iter_local_sft, iter_local_texts

    if sft:
        for instruction, inp, output in iter_local_sft(root):
            text = f"{instruction}\n{inp}\n{output}" if inp else f"{instruction}\n{output}"
            for ch in _split_text(text, chunk, overlap):
                yield ch
        return
    for text in iter_local_texts(root):
        for ch in _split_text(text, chunk, overlap):
            yield ch


def _split_text(text: str, chunk: int, overlap: int):
    """单文本按段落切块（块约 chunk 字符，重叠 overlap）."""
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    buf = ""
    for p in paras:
        if len(buf) + len(p) > chunk and buf:
            yield buf
            buf = buf[-overlap:] + p
        else:
            buf = buf + ("\n" if buf else "") + p
    if buf:
        yield buf


def main(argv=None) -> int:
    """构建并落盘（打印统计）."""
    from src.llm.local.retrieval import BM25Retriever

    p = argparse.ArgumentParser(description="构建 BM25 检索库")
    p.add_argument("--data", required=True, help="语料目录（遍历 txt/md/jsonl/parquet）")
    p.add_argument("--out", required=True, help="输出路径（.bm25）")
    p.add_argument("--chunk", type=int, default=600, help="块大小（字符）")
    p.add_argument("--overlap", type=int, default=100, help="块间重叠（字符）")
    p.add_argument("--sft", action="store_true",
                   help="SFT 目录（instruction/input/output 拼文本，而非纯文）")
    p.add_argument("--limit", type=int, default=0, help="最多块数（0=全量）")
    args = p.parse_args(argv)

    r = BM25Retriever()
    n = 0
    for i, ch in enumerate(iter_chunks(args.data, args.chunk, args.overlap, args.sft)):
        if args.limit and n >= args.limit:
            break
        r.add(str(n), ch)
        n += 1
        if n % 5000 == 0:
            print(f"已建 {n} 块…", flush=True)
    r.save(args.out)
    print(f"完成：{n} 块，{len(r)} 文档，落盘 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
