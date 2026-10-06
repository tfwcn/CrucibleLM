"""复壮手术 — 按 masks.pt 名单恒等式重开休眠神经元，产出可严格载入的新起点.

用法：
  python scripts/rejuvenate.py --src data/llm-sft5/best/model.pt \\
      --masks data/dormancy/masks.pt --dst data/llm-sft-rejuv \\
      --vocab data/llm-sft5/vocab.json --cap-frac 0.2

流程：同架构建模 → 严格载入 → 手术（输入随机重开 + 输出置零）→
自检（同 batch 前后输出 allclose，阈值内才落盘）→ 存 dst/model.pt。
新开训练直接新开优化器（无 stale 动量）；继续旧优化器需另清对应动量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="复壮手术")
    p.add_argument("--src", required=True, help="术前权重 model.pt")
    p.add_argument("--masks", required=True, help="scan_dormancy 产出的 masks.pt")
    p.add_argument("--dst", required=True, help="输出目录")
    p.add_argument("--vocab", required=True, help="词表 vocab.json（自检编码用）")
    p.add_argument("--cap-frac", type=float, default=0.2, help="每专家最多重开占比")
    p.add_argument("--memory-every", type=int, default=0, help="记忆层间隔（与权重一致）")
    p.add_argument("--retro-every", type=int, default=0, help="RETRO 交错间隔（与权重一致）")
    p.add_argument("--retro-len", type=int, default=64, help="chunk 长度（与权重一致）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--device", default="", help="设备（空=自动）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """手术主流程."""
    import torch

    from src.llm.local.config import SmallLLMConfig
    from src.llm.local.infer import SimpleTokenizer
    from src.llm.local.model import TinyLLM
    from src.llm.local.rejuvenate import apply_rejuvenation

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = SmallLLMConfig()
    config.memory_every = args.memory_every
    if args.retro_every > 0:
        config.retro_enabled = True
        config.retro_every = args.retro_every
        config.retro_chunk_len = args.retro_len
    model = TinyLLM(config).to(device).eval()
    model.load_state_dict(torch.load(args.src, map_location=device))
    print(f"术前权重已载入：{args.src}", flush=True)

    blob = torch.load(args.masks, map_location="cpu")
    masks = {k: v for k, v in blob["masks"].items()}
    scores = {k: v for k, v in blob.get("scores", {}).items()}
    info = apply_rejuvenation(model, masks, scores or None,
                              seed=args.seed, cap_frac=args.cap_frac)
    n_applied = sum(info["applied"].values())
    print(f"重开 {n_applied} 个神经元（{len(info['applied'])} 个专家），"
          f"跳过 {len(info['skipped'])} 个专家", flush=True)
    if n_applied == 0:
        print("无人可开：masks 全空，请回查 report.json（大概率该结案）",
              flush=True)
        return 2

    # 自检：同 batch 前后输出近似一致（休眠单元贡献≈0，阈值内才落盘）
    tok = SimpleTokenizer()
    tok._chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    before = torch.load(args.src, map_location=device)
    ref = TinyLLM(config).to(device).eval()
    ref.load_state_dict(before)
    x = torch.randint(0, len(tok._chars) + 4, (1, 32), device=device)
    with torch.no_grad():
        d = (ref(x)["logits"] - model(x)["logits"]).abs().max().item()
    print(f"手术前后 logits 差：{d:.6f}（休眠越彻底越小）", flush=True)
    if d > 1e-3:
        print("差值过大：名单含活性单元，中止落盘（调高阈值重扫）", flush=True)
        return 3

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dst / "model.pt")
    (dst / "meta.json").write_text(json.dumps({
        "src": args.src, "applied": {str(k): v for k, v in info["applied"].items()},
        "max_diff": d,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已落盘：{dst / 'model.pt'}（可直接 --sft-init 严格载入，新开优化器）",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
