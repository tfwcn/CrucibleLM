"""神经元复壮（ReDo 思想）：找出长期沉睡的 FFN 中间神经元，恒等式重开.

原理：旧权重已拟合 easy token（loss≈0 → 梯度≈0），RHO 只反向 hard token
梯度——重开的新权重第一口梯度天然来自难数据，躺不回去。
安全三件套：①休眠按**激活**判定（magnitude 小可能是被抑制的重要特征），并可选
**峰值保护**（`peak_frac>0`：平均激活低但峰值高者不判休眠，防误杀稀有强特征）；
②手术对休眠单元恒等（输入侧随机重开 + 输出侧置零；激活恒零则逐位不变，无 loss 尖峰）；
③单次 cap 占比 + 术前归档 + A/B 验收。
新训练直接新开优化器（无 stale 动量问题）；继续旧优化器需另清对应动量。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def collect_mid_activations(model, batches: list[torch.Tensor],
                            ) -> dict[tuple, dict[str, torch.Tensor]]:
    """收集各专家中间激活 |silu(gate)*up| 的**逐神经元均值与峰值**.

    返回 {(layer, kind, expert): {"mean": (h,), "peak": (h,)}}。峰值供休眠判定的
    "少见但强烈"保护使用——只看均值会把"偶尔大幅激活的稀有特征"误判为休眠
    （Goodfire 2026 提示：SwiGLU 神经元的强响应可能与弱响应符号区间不同）。

    在 MoE 模块上挂一个 hook（分组 bmm 后专家子模块 forward 不再被调用，
    挂专家上收不到数据），逐专家重算中间激活；流式累加和/计数/取最大，
    内存 O(专家数×h)，与校准量无关。backbone 冻结、no_grad，调用方保证 eval。
    """
    sums: dict[tuple, torch.Tensor] = {}
    peaks: dict[tuple, torch.Tensor] = {}
    counts: dict[tuple, int] = {}
    handles = []
    for li, layer in enumerate(model.layers):
        def _hook(_m, inp, _o, _li=li, _layer=layer):
            x = inp[0].detach()
            with torch.no_grad():
                groups = (("routed", _layer.moe.experts),
                          ("shared", _layer.moe.shared))
                for kind, experts in groups:
                    for ei, e in enumerate(experts):
                        mid = (F.silu(e.w_gate(x)) * e.w_up(x)).detach().float().abs()
                        red = tuple(range(mid.dim() - 1))  # 除最后一维（神经元）外全归约
                        part = mid.sum(dim=red)
                        pk = mid.amax(dim=red)
                        key = (_li, kind, ei)
                        n = mid.numel() // mid.shape[-1]
                        if key in sums:
                            sums[key] = sums[key] + part.cpu()
                            peaks[key] = torch.maximum(peaks[key], pk.cpu())
                            counts[key] += n
                        else:
                            sums[key] = part.cpu()
                            peaks[key] = pk.cpu()
                            counts[key] = n

        handles.append(layer.moe.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch)
    finally:
        for hd in handles:
            hd.remove()
    return {key: {"mean": sums[key] / counts[key], "peak": peaks[key]}
            for key in sums}


def dormancy_masks(acts: dict,
                   rel_threshold: float = 1e-3,
                   peak_frac: float = 0.0,
                   ) -> tuple[dict[tuple, torch.Tensor],
                              dict[tuple, torch.Tensor], dict]:
    """激活休眠判定：score=神经元平均|激活|，阈值=**该专家**神经元均值×rel_threshold.

    （每个 key 是一个专家；阈值在其内部比较——路由专家接收的 token 子集不同、
    激活尺度各异，逐专家比逐层更稳健。实测两种口径给出的最闲比值都≈0.28。）
    `acts` 取值可为 `collect_mid_activations` 的 `{"mean","peak"}` 字典，也可直接是
    逐神经元均值张量（向后兼容；此时无峰值保护）。

    `peak_frac>0` 开启**峰值保护**：平均激活低于阈值、但峰值达到**该专家**最大峰值
    `peak_frac` 倍的神经元不判休眠（少见但强烈 → 误杀代价高，是"被抑制的重要特征"）。
    与休眠阈值同用逐专家口径，避免"整层尺度大就永不保护"。

    返回 ({key: bool mask（True=休眠）}, {key: score}, 报告 {layer: {...}})。
    报告逐层给出 `dormant_frac`/`n_experts`/`rescued`（被峰值保护救回的神经元数）。
    """
    masks: dict[tuple, torch.Tensor] = {}
    scores: dict[tuple, torch.Tensor] = {}
    by_layer: dict[int, list[float]] = {}
    rescued_by_layer: dict[int, int] = {}
    peak_max: dict[tuple, float] = {}
    if peak_frac > 0:
        for key, val in acts.items():
            if isinstance(val, dict) and "peak" in val:
                peak_max[key] = float(val["peak"].max())
    for key, val in acts.items():
        li = key[0]
        h = val["mean"] if isinstance(val, dict) else val
        # collect 已给逐神经元均值（1 维）则直接用，否则按 token 平均
        score = h.float() if h.dim() == 1 else h.float().abs().mean(0)
        expert_mean = score.mean().item()
        thr = expert_mean * rel_threshold
        mask = score < thr
        if peak_frac > 0 and isinstance(val, dict) and "peak" in val:
            protect = val["peak"].float() >= peak_max.get(key, 0.0) * peak_frac
            rescued_by_layer[li] = rescued_by_layer.get(li, 0) + int(
                (mask & protect).sum().item())
            mask = mask & ~protect
        masks[key] = mask
        scores[key] = score
        by_layer.setdefault(li, []).append(float(mask.float().mean()))
    report = {li: {"dormant_frac": sum(v) / len(v), "n_experts": len(v),
                   "rescued": rescued_by_layer.get(li, 0)}
              for li, v in by_layer.items()}
    return masks, scores, report


def apply_rejuvenation(model, masks: dict[tuple, torch.Tensor],
                       scores: dict[tuple, torch.Tensor] | None = None,
                       seed: int = 0, cap_frac: float = 0.2,
                       ) -> dict:
    """恒等式复壮：休眠神经元输入侧随机重开 + 输出侧置零.

    超 cap 的专家只取 score 最低的前 cap_frac（按激活分数排序）。
    返回 {"applied": {(li,kind,ei): n}, "skipped": [...]}。
    休眠单元输出不变（单测锁定）；调用后建议新开优化器（或清对应动量）。
    """
    import math

    g = torch.Generator().manual_seed(seed)
    applied: dict[tuple, int] = {}
    skipped: list[tuple] = []
    for (li, kind, ei), mask in masks.items():
        experts = (model.layers[li].moe.experts if kind == "routed"
                   else model.layers[li].moe.shared)
        expert = experts[ei]
        idx = torch.where(mask)[0]
        if len(idx) == 0:
            skipped.append((li, kind, ei))
            continue
        cap = max(1, int(len(mask) * cap_frac))
        if len(idx) > cap:
            ident = (li, kind, ei)
            if scores is not None and ident in scores:
                order = torch.argsort(scores[ident][idx])
                idx = idx[order[:cap]]
            else:
                idx = idx[:cap]
        d = expert.w_gate.weight.shape[1]
        std = 1.0 / math.sqrt(d)
        with torch.no_grad():
            for j in idx.tolist():
                expert.w_gate.weight[j].normal_(0, std, generator=g)
                expert.w_up.weight[j].normal_(0, std, generator=g)
                expert.w_down.weight[:, j].zero_()
        applied[(li, kind, ei)] = len(idx)
    return {"applied": applied, "skipped": skipped}
