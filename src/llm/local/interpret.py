"""MoE 可解释性分析（只读、CPU 可跑）：SwiGLU 符号区间统计 + 专家激活画像/生长追踪.

参考 Goodfire 2026《Neural Networks Are Already Clean》：
- **双负区间**：SwiGLU 的 `silu(gate)*up` 在 `gate<0 且 up<0` 时输出为正，且"看起来像
  负极点"；约 4% 神经元的"最强激活"可能全被这个 echo 占据——必须**分开看四种符号组合**。
- **分化生长**：神经元随训练沿多条轴生长，休眠可能是"未长成"而非永久死亡。

本模块只做统计画像，不训练、不改权重。分组 bmm 下专家子模块 forward 不被调用，
故按 `rejuvenate.py` 的做法在 MoE 输入上逐专家重算 gate/up。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def iter_experts(moe):
    """按 (kind, idx, expert) 遍历路由专家与共享专家."""
    for ei, e in enumerate(moe.experts):
        yield "routed", ei, e
    for ei, e in enumerate(moe.shared):
        yield "shared", ei, e


@torch.no_grad()
def collect_moe_inputs(model, batches, layers=None) -> dict[int, torch.Tensor]:
    """收集各 MoE 层的前向输入，展平成 (N, d) 的 fp32，供逐专家重算激活.

    `batches` 为 token id 张量序列；`layers` 为要保留的层号集合（None=全部）。
    """
    inputs: dict[int, list[torch.Tensor]] = {}
    handles = []
    for li, layer in enumerate(model.layers):
        if layers is not None and li not in layers:
            continue

        def _hook(_m, inp, _o, _li=li):
            inputs.setdefault(_li, []).append(inp[0].detach())

        handles.append(layer.moe.register_forward_hook(_hook))
    try:
        for b in batches:
            model(b)
    finally:
        for hd in handles:
            hd.remove()
    return {li: torch.cat([t.reshape(-1, t.shape[-1]).float() for t in ts], 0)
            for li, ts in inputs.items()}


def regime_summary(mid: torch.Tensor, dn: torch.Tensor,
                   top_frac: float = 0.01) -> dict:
    """单专家 (N, h) 中间激活的正极响应里，双负（−×−）占比.

    `mid = silu(gate)*up`，`dn = (gate<0)&(up<0)`（双负，输出为正）。只看 `mid>0`
    的响应，分别给出"全部正激活"与"最强 top_frac 段"的 −×− 占比，以及"最强单个
    激活即双负"的神经元比例（echo 主导）。无正激活的神经元（死单元）不计入。
    """
    h = mid.shape[1]
    all_sh: list[float] = []
    top_sh: list[float] = []
    argmax_dn: list[bool] = []
    n_pos = 0
    for j in range(h):
        v = mid[:, j]
        pos = v > 0
        np_ = int(pos.sum())
        n_pos += np_
        if np_ == 0:
            continue
        dv = dn[pos, j].float()
        vp = v[pos]
        all_sh.append(float(dv.mean()))
        k = max(1, int(round(np_ * top_frac)))
        top_idx = vp.topk(k).indices
        top_sh.append(float(dv[top_idx].mean()))
        argmax_dn.append(bool(dv[vp.argmax()]))
    n = len(all_sh)
    return {
        "n_pos": n_pos,
        "n_neurons": n,
        "dn_share": (sum(all_sh) / n) if n else 0.0,
        "dn_top": (sum(top_sh) / n) if n else 0.0,
        "frac_echo": (sum(1 for s in top_sh if s > 0.5) / n) if n else 0.0,
        "frac_argmax_dn": (sum(1 for f in argmax_dn if f) / n) if n else 0.0,
    }


@torch.no_grad()
def expert_sign_regimes(model, batches, layers=None, top_frac: float = 0.01) -> dict:
    """逐专家统计 SwiGLU 四符号组合占比，返回 {(li, kind, ei): regime_summary}."""
    out = {}
    for li, x in collect_moe_inputs(model, batches, layers).items():
        moe = model.layers[li].moe
        for kind, ei, e in iter_experts(moe):
            gate = e.w_gate(x)
            up = e.w_up(x)
            s = regime_summary(F.silu(gate) * up, (gate < 0) & (up < 0),
                               top_frac)
            s.update(li=li, kind=kind, ei=ei)
            out[(li, kind, ei)] = s
    return out


@torch.no_grad()
def expert_activation_profile(model, batches, layers=None) -> dict:
    """逐专家统计中间激活 |silu(gate)*up| 的逐神经元均值/峰值/p99.

    返回 {(li, kind, ei): {"mean", "peak", "p99"}}，每个都是长度 h 的向量。
    用于休眠判定的峰值保护与跨 checkpoint 生长追踪。
    """
    out = {}
    for li, x in collect_moe_inputs(model, batches, layers).items():
        moe = model.layers[li].moe
        for kind, ei, e in iter_experts(moe):
            mid = (F.silu(e.w_gate(x)) * e.w_up(x)).abs().float()  # (N, h)
            n = mid.shape[0]
            k = max(1, min(n, int(round(0.99 * (n - 1))) + 1))
            out[(li, kind, ei)] = {
                "mean": mid.mean(0),
                "peak": mid.amax(0),
                "p99": mid.kthvalue(k, dim=0).values,
            }
    return out
