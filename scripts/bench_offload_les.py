"""专家 offload / LES 预取的性能小基准 — 对比分组 bmm / 同步 offload / 异步预取.

用途：量化"跨设备搬运"相对分组快路径的吞吐代价，以及 LES 异步预取能挽回多少。
只读推理，不训练；默认用 tiny 配置（秒级），可 `--preset base` 跑真实 128M。

用法：
    python scripts/bench_offload_les.py                 # tiny，容量受限
    python scripts/bench_offload_les.py --preset base --capacity 8 --new-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import SmallLLMConfig, tiny_test_config
from src.llm.local.model import TinyLLM


def _sync(device: str) -> None:
    """CUDA 计时前同步（否则测到的是下发时间而非完成时间）."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_generate(model: TinyLLM, x: torch.Tensor, n: int, device: str) -> float:
    """跑一次生成，返回墙钟毫秒（含一次同步）."""
    model.eval()
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        model.generate(x, max_new_tokens=n)
        _sync(device)
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="offload/LES 预取性能基准")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "base"])
    ap.add_argument("--capacity", type=int, default=4, help="热专家容量（0=不限）")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prefetch", action="store_true", help="开启 LES 异步预取")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tiny_test_config() if args.preset == "tiny" else SmallLLMConfig()
    torch.manual_seed(0)

    x = torch.randint(0, cfg.vocab_size, (1, 8), device=device)

    def build(offload: bool) -> TinyLLM:
        m = TinyLLM(cfg).to(device)
        if offload:
            m.setup_offload(compute_device=device)
        m.eval()
        return m

    # 基线：分组 bmm（不 offload）
    base = build(False)
    base_ms = min(bench_generate(base, x, args.new_tokens, device)
                  for _ in range(args.repeat))

    # 同步 offload
    cfg.offload_enabled = True
    cfg.offload_target = "cpu"
    cfg.offload_capacity = args.capacity
    cfg.offload_prefetch = False
    cfg.les_enabled = False
    sync_m = build(True)
    sync_ms = min(bench_generate(sync_m, x, args.new_tokens, device)
                  for _ in range(args.repeat))
    sync_moves = sum(m.store.moves for m in _moes(sync_m) if m.store is not None)

    # LES 异步预取（cpu target + CUDA 才真正异步）
    cfg.offload_prefetch = True
    cfg.les_enabled = args.prefetch
    les_m = build(True)
    les_ms = min(bench_generate(les_m, x, args.new_tokens, device)
                 for _ in range(args.repeat))
    les_moves = sum(m.store.moves for m in _moes(les_m) if m.store is not None)

    print(f"== offload/LES 基准（device={device}, capacity={args.capacity}, "
          f"tokens={args.new_tokens}）==")
    print(f"  分组 bmm 基线 : {base_ms:8.1f} ms")
    print(f"  同步 offload  : {sync_ms:8.1f} ms  ({sync_ms / base_ms:.2f}x) moves={sync_moves}")
    if args.prefetch:
        print(f"  LES 异步预取  : {les_ms:8.1f} ms  ({les_ms / base_ms:.2f}x) moves={les_moves}")


def _moes(model: TinyLLM) -> list:
    """取出模型里所有 FineGrainedMoE 层（含 store 的层）."""
    from src.llm.local.moe import FineGrainedMoE

    return [m for m in model.modules() if isinstance(m, FineGrainedMoE)]


if __name__ == "__main__":
    main()
