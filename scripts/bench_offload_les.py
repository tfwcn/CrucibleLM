"""专家 offload / LES 预取的性能基准 — 对比分组 bmm / 同步 offload / 异步预取.

用途：量化"跨设备搬运"相对分组快路径的吞吐代价，以及 LES 异步预取能挽回多少。
只读推理，不训练。默认 tiny（秒级）；`--preset base` 跑真实 ~128M；
`--model data/llm-ckpt/model --run-yaml data/llm-ckpt/run.yaml` 用真实权重
（各变体共享同一份权重，保证路由一致、可比）。

用法：
    python scripts/bench_offload_les.py --preset base --capacity 8 --new-tokens 64
    python scripts/bench_offload_les.py --preset base --model data/llm-ckpt/model \
        --capacity 8 --prefetch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import (
    SmallLLMConfig,
    sync_csa_reuse_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM


def _sync(device: str) -> None:
    """CUDA 计时前同步（否则测到的是下发时间而非完成时间）."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_generate(model: TinyLLM, x: torch.Tensor, n: int, device: str) -> tuple[float, int]:
    """跑一次生成，返回 (墙钟毫秒, 实际新增 token 数)（含一次同步）."""
    model.eval()
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        out = model.generate(x, max_new_tokens=n)
        _sync(device)
    return (time.perf_counter() - t0) * 1000.0, int(out.shape[1] - x.shape[1])


def _moes(model: TinyLLM) -> list:
    """取出模型里所有 FineGrainedMoE 层（含 store 的层）."""
    from src.llm.local.moe import FineGrainedMoE

    return [m for m in model.modules() if isinstance(m, FineGrainedMoE)]


def _moves(model: TinyLLM) -> int:
    """累计各层 offload store 的搬运次数."""
    return sum(m.store.moves for m in _moes(model) if m.store is not None)


def main() -> None:
    ap = argparse.ArgumentParser(description="offload/LES 预取性能基准")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "base"])
    ap.add_argument("--model", default="", help="权重前缀（LocalChatBackend 风格，如 data/llm-ckpt/model）")
    ap.add_argument("--run-yaml", default="", help="训练 run.yaml（复用 memory/retro/hyper 结构开关）")
    ap.add_argument("--target", default="cpu", choices=["cpu", "disk"], help="offload 冷端")
    ap.add_argument("--offload-dir", default="/tmp/crucible-offload-bench")
    ap.add_argument("--capacity", type=int, default=4, help="热专家容量（0=不限）")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3, help="每场景重复取最小值")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prefetch", action="store_true", help="开启 LES 异步预取对照")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tiny_test_config() if args.preset == "tiny" else SmallLLMConfig()
    if args.model:
        # 按真实权重对齐 CSA2（该开关不改 state_dict 键，装错不报错只会静默变语义）
        sync_csa_reuse_from_checkpoint(cfg, args.model)
    if args.run_yaml:
        from scripts.measure_mtp_acceptance import apply_run_yaml

        apply_run_yaml(cfg, args.run_yaml)

    # 参考权重：随机（固定种子）或真实 checkpoint
    torch.manual_seed(0)
    ref = TinyLLM(cfg)
    if args.model:
        ref.load_state_dict(torch.load(args.model + ".pt", map_location="cpu"))
        print(f"已载入真实权重：{args.model}.pt")
    ref_sd = ref.state_dict()
    del ref

    x = torch.randint(0, cfg.vocab_size, (1, 8), device=device)

    def build(offload: bool, prefetch: bool, les: bool) -> TinyLLM:
        """用参考权重建模型；offload=True 时按配置挂 store（共享权重→路由一致）."""
        cfg.offload_enabled = offload
        cfg.offload_target = args.target
        cfg.offload_dir = args.offload_dir if args.target == "disk" else ""
        cfg.offload_capacity = args.capacity
        cfg.offload_prefetch = prefetch
        cfg.les_enabled = les
        m = TinyLLM(cfg)
        m.load_state_dict(ref_sd)
        m = m.to(device)
        if offload:
            m.setup_offload(compute_device=device)
        m.eval()
        # 预热（cuDNN 自选 / 分配器等）
        for _ in range(args.warmup):
            bench_generate(m, x, 8, device)
        return m

    def bench(m: TinyLLM) -> tuple[float, int]:
        best_ms, best_tok = float("inf"), 0
        for _ in range(args.repeat):
            ms, tok = bench_generate(m, x, args.new_tokens, device)
            if ms < best_ms:
                best_ms, best_tok = ms, tok
        return best_ms, best_tok

    print(f"== offload/LES 基准（device={device}, preset={args.preset}, "
          f"target={args.target}, capacity={args.capacity}, tokens={args.new_tokens}）==")
    base = build(False, False, False)
    base_ms, base_tok = bench(base)
    print(f"  分组 bmm 基线 : {base_ms:8.1f} ms  {base_tok / base_ms * 1000:6.1f} tok/s")
    del base

    sync = build(True, False, False)
    sync_ms, sync_tok = bench(sync)
    print(f"  同步 offload  : {sync_ms:8.1f} ms  ({sync_ms / base_ms:.2f}x) "
          f"{sync_tok / sync_ms * 1000:6.1f} tok/s  moves={_moves(sync)}")
    del sync

    if args.prefetch:
        les = build(True, True, True)
        les_ms, les_tok = bench(les)
        print(f"  LES 异步预取  : {les_ms:8.1f} ms  ({les_ms / base_ms:.2f}x) "
              f"{les_tok / les_ms * 1000:6.1f} tok/s  moves={_moves(les)}")


if __name__ == "__main__":
    main()
