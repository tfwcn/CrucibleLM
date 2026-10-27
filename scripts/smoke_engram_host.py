"""宿主 Engram 表在真机上的端到端冒烟 — 逐位一致 / 显存账 / 单步吞吐.

用途：`docs/details/host-resident-memory.md` 的实测来源。回答三个问题：

1. **改了语义没有**：同权重下 `engram_host=True/False` 的前向 logits 是否逐位相同
   （表搬走了但数学没变，否则"分设备"只是在改模型）；
2. **省了多少显存**：GPU 参数字节之差是否正好是表大小（表真的不在卡上了）；
3. **付了多少时间**：单步耗时差（PCIe + CPU gather 的账单 vs Sinkhorn 的全表归一化）。

只跑极小的合成数据，不读语料、不写 checkpoint、不动 `data/`。默认 20 步，秒级。

两个变体**分进程**测显存：同进程里先建的模型会一直占着显存，绝对占用不可比。

用法：
    python scripts/smoke_engram_host.py                     # 默认表 65521 行
    python scripts/smoke_engram_host.py --table-size 262139 --steps 30
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import tiny_test_config  # noqa: E402
from src.llm.local.engram import EngramTable  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.optim import build_hybrid_optimizer  # noqa: E402

_MB = 1 << 20


def _cfg(host: bool, args: argparse.Namespace):
    """极小 Engram 配置（每 2 层插 1 个 Engram；表大小/维度由命令行给）."""
    cfg = tiny_test_config()
    cfg.engram_every = 2
    cfg.engram_table_size = args.table_size
    cfg.engram_d_mem = args.d_mem
    cfg.engram_heads = args.heads
    cfg.engram_orders = "2,3"
    cfg.engram_host = host
    return cfg


def _tables(model: TinyLLM) -> list[EngramTable]:
    return [m for m in model.modules() if isinstance(m, EngramTable)]


def _build(host: bool, args: argparse.Namespace) -> TinyLLM:
    """确定性构建：同 seed 下两个变体的**初始权重逐位相同**（否则测不出语义差异）.

    注意顺序：先在 **CPU** 上建模型并随机初始化，再 `.cuda()`。
    宿主表的主权重永远留 CPU，若先 `.cuda()` 再 `normal_`，两个变体会分别用
    CUDA/CPU 随机数生成器 → 权重不同，逐位比较就失去意义。
    """
    torch.manual_seed(0)
    model = TinyLLM(_cfg(host, args))
    with torch.no_grad():
        for layer in model.layers:
            eng = getattr(layer, "engram", None)
            if eng is None:
                continue
            # 打破 W_V 恒等起点与未初始化表：否则两条路径都输出 ~0，测不出差异
            eng.w_v.weight.normal_(0.0, 0.02)
            for t in eng.tables:
                t.weight.normal_(0.0, 0.3)
    return model.cuda()      # 宿主表的 `_apply` 早退，主权重仍留 CPU


def _ce(model: TinyLLM, ids: torch.Tensor) -> torch.Tensor:
    """自回归移位 CE（与训练脚本同口径；这里只关心可训练性与吞吐）."""
    logits = model(ids)["logits"]
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def _run_variant(args: argparse.Namespace, host: bool) -> dict:
    """单变体测量（子进程入口）：返回参数量/稳态/峰值/单步耗时/loss."""
    model = _build(host, args)
    param_mib = sum(p.numel() * p.element_size()
                    for p in model.parameters() if p.device.type == "cuda") / _MB
    table_mib = sum(t.nbytes() for t in _tables(model)) / _MB

    # 固定同一批数据：两个子进程各自构造（同 seed），loss 才可比
    torch.manual_seed(1)
    batches = [torch.randint(0, model.config.vocab_size, (args.batch, args.seq),
                             device="cuda") for _ in range(args.steps + 4)]

    with torch.no_grad():
        loss0 = _ce(model, batches[0]).item()
    # 快照必须放 CPU：设备变体的表在显存里，克隆到原地会污染显存测量（+64 MiB）
    snap = {i: t.weight.detach().cpu().clone()
            for i, t in enumerate(_tables(model))}

    opt = build_hybrid_optimizer(model, sinkhorn=True, adam_lr=args.lr,
                                 host_lr=args.host_lr)
    opt.zero_grad()
    # 预热（不计时）：首步含 kernel 选择/分配器行为，算进去会系统性高估
    for ids in batches[:3]:
        _ce(model, ids).backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    steady = torch.cuda.memory_allocated()      # 权重 + 优化器状态的稳态占用
    torch.cuda.reset_peak_memory_stats()

    losses = []
    t0 = time.perf_counter()
    for ids in batches[3:3 + args.steps]:
        loss = _ce(model, ids)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.steps * 1000
    peak = torch.cuda.max_memory_allocated()

    changed = 0
    with torch.no_grad():
        for i, t in enumerate(_tables(model)):
            changed += int((t.weight.detach().cpu() != snap[i]).any(dim=1).sum())
    return {
        "host": host,
        "ms_per_step": round(ms, 2),
        "param_MiB": round(param_mib, 1),
        "table_MiB": round(table_mib, 1),
        "steady_MiB": round(steady / _MB, 1),
        "peak_MiB": round(peak / _MB, 1),
        "loss0": round(loss0, 4),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "changed_rows": changed,
        "finite": all(v == v and abs(v) < 1e9 for v in losses),
    }


def _spawn(args: argparse.Namespace, host: bool) -> dict:
    """起一个干净的子进程测一个变体（显存绝对占用才可比）."""
    cmd = [sys.executable, __file__, "--variant", "host" if host else "device",
           "--table-size", str(args.table_size), "--d-mem", str(args.d_mem),
           "--heads", str(args.heads), "--steps", str(args.steps),
           "--batch", str(args.batch), "--seq", str(args.seq),
           "--lr", str(args.lr), "--host-lr", str(args.host_lr)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    tag = "RESULT_JSON:"
    line = next(ln for ln in out.splitlines() if ln.startswith(tag))
    return json.loads(line[len(tag):])


def main() -> None:
    ap = argparse.ArgumentParser(description="宿主 Engram 表真机冒烟")
    ap.add_argument("--table-size", type=int, default=65521,
                    help="每头表行数（须素数；默认 65521 ≈ 8 MB/表）")
    ap.add_argument("--d-mem", type=int, default=64, help="每阶维度")
    ap.add_argument("--heads", type=int, default=4, help="每阶头数")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3, help="主干 AdamW lr")
    ap.add_argument("--host-lr", type=float, default=0.03,
                    help="宿主表行稀疏 Adagrad lr（Adagrad 口径）")
    ap.add_argument("--variant", choices=("device", "host"), default=None,
                    help="内部用：只测一个变体并打印 JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("本冒烟需要 CUDA（要测 H2D 与显存账）")

    if args.variant is not None:                      # 子进程：只测一个变体
        res = _run_variant(args, host=args.variant == "host")
        print("RESULT_JSON:" + json.dumps(res), flush=True)
        return

    # ===== 1. 逐位一致：同权重、同输入，两条路径的 logits =====
    m_dev = _build(False, args)
    m_host = _build(True, args)
    torch.manual_seed(2)
    x = torch.randint(0, m_dev.config.vocab_size, (args.batch, args.seq), device="cuda")
    with torch.no_grad():
        a = m_dev(x)["logits"]
        b = m_host(x)["logits"]
    diff = (a - b).abs().max().item()
    del m_dev, m_host, a, b
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[1/3] 前向逐位一致：max|Δ| = {diff:.3e}"
          f"（{'逐位相同' if diff == 0 else '有差，需查 dtype/量化'}）", flush=True)

    # ===== 2/3. 分进程测显存与吞吐 =====
    dev = _spawn(args, host=False)
    host = _spawn(args, host=True)
    print(f"[2/3] 表 {dev['table_MiB']:.1f} MiB｜GPU 参数 "
          f"{dev['param_MiB']:.1f} → {host['param_MiB']:.1f} MiB"
          f"（省 {dev['param_MiB'] - host['param_MiB']:.1f} MiB）", flush=True)
    for r in (dev, host):
        tag = "host  " if r["host"] else "device"
        print(f"  {tag} 单步 {r['ms_per_step']:7.2f} ms｜稳态 {r['steady_MiB']:7.1f} MiB｜"
              f"峰值 {r['peak_MiB']:7.1f} MiB｜loss0 {r['loss0']:.4f}｜"
              f"{r['loss_first']:.4f} → {r['loss_last']:.4f}"
              + (f"｜被改动行 {r['changed_rows']}" if r["host"] else ""), flush=True)

    d_ms = host["ms_per_step"] - dev["ms_per_step"]
    print(f"\n[3/3] 稳态显存差 {dev['steady_MiB'] - host['steady_MiB']:.1f} MiB"
          f"（峰值差 {dev['peak_MiB'] - host['peak_MiB']:.1f} MiB）｜"
          f"单步代价 {d_ms:+.2f} ms"
          f"（{d_ms / dev['ms_per_step'] * 100:+.1f}%）")
    print(f"      训练前 loss device {dev['loss0']} vs host {host['loss0']}"
          "（同权重同数据，应相同）")
    if not (dev["finite"] and host["finite"]):
        raise SystemExit("loss 出现 NaN/Inf，冒烟失败")


if __name__ == "__main__":
    main()
