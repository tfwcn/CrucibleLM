"""休眠扫描 — 冻结 backbone 跑校准集，逐专家统计中间激活，给复壮提供名单.

用法：
  python scripts/scan_dormancy.py --src data/llm-sft5/best/model.pt \\
      --vocab data/llm-sft5/vocab.json --data data/sft-zh \\
      --out data/dormancy --memory-every 4 --retro-every 4

输出 out/report.json（逐层 dormant 比例）+ masks.pt（masks/scores）。
只读权重，不碰训练；先看数字再决定做不做手术（某层 >10% 才值得）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="休眠扫描")
    p.add_argument("--src", required=True, help="权重 model.pt")
    p.add_argument("--vocab", required=True, help="词表 vocab.json")
    p.add_argument("--data", required=True, help="SFT 校准语料目录")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--samples", type=int, default=512, help="校准段数")
    p.add_argument("--seq-len", type=int, default=512, help="校准序列长度")
    p.add_argument("--batch", type=int, default=4, help="校准 batch")
    p.add_argument("--rel-threshold", type=float, default=1e-3,
                   help="休眠阈值（层均值的相对倍数）")
    p.add_argument("--memory-every", type=int, default=0, help="记忆层间隔（与权重一致）")
    p.add_argument("--retro-every", type=int, default=0, help="RETRO 交错间隔（与权重一致）")
    p.add_argument("--retro-len", type=int, default=64, help="chunk 长度（与权重一致）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--device", default="", help="设备（空=自动）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """扫描主流程."""
    import torch

    from src.llm.local.config import SmallLLMConfig
    from src.llm.local.data import SFT_PROMPT, iter_local_sft
    from src.llm.local.infer import SimpleTokenizer
    from src.llm.local.model import TinyLLM
    from src.llm.local.rejuvenate import collect_mid_activations, dormancy_masks

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tok = SimpleTokenizer()
    tok._chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    tok.vocab_size = len(tok._chars) + 4

    config = SmallLLMConfig()
    config.memory_every = args.memory_every
    if args.retro_every > 0:
        config.retro_enabled = True
        config.retro_every = args.retro_every
        config.retro_chunk_len = args.retro_len
    model = TinyLLM(config).to(device).eval()
    # 严格载入（形状对不上直接炸，不静默：扫描必须基于真权重）
    model.load_state_dict(torch.load(args.src, map_location=device))
    print(f"权重已载入：{args.src}", flush=True)

    texts: list[str] = []
    for instruction, inp, output in iter_local_sft(args.data):
        texts.append(SFT_PROMPT.format(instruction=instruction, input=inp or ""))
        texts.append(output)
        if len(texts) >= args.samples * 2:
            break
    print(f"校准文本 {len(texts)} 段", flush=True)
    batches = []
    for b in range(0, len(texts), args.batch):
        ids = [tok.encode(t, add_bos=False)[:args.seq_len]
               for t in texts[b:b + args.batch]]
        maxlen = max(len(r) for r in ids)
        batches.append(torch.tensor(
            [r + [SimpleTokenizer.PAD] * (maxlen - len(r)) for r in ids],
            dtype=torch.long, device=device))

    acts = collect_mid_activations(model, batches)
    masks, scores, report = dormancy_masks(acts, args.rel_threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({str(k): v for k, v in report.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"masks": {k: v.cpu() for k, v in masks.items()},
                "scores": {k: v.cpu() for k, v in scores.items()},
                "rel_threshold": args.rel_threshold}, out / "masks.pt")
    total_dorm = sum(m.float().mean().item() for m in masks.values()) / len(masks)
    print(f"全局休眠占比：{total_dorm:.4f}（>0.10 值得手术，<0.05 建议结案）",
          flush=True)
    for li in sorted(report):
        print(f"  层 {li}：dormant={report[li]['dormant_frac']:.4f} "
              f"({report[li]['n_experts']} 个专家)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
