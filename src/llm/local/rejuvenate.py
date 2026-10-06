"""神经元复壮（ReDo 思想）：找出长期沉睡的 FFN 中间神经元，恒等式重开.

原理：旧权重已拟合 easy token（loss≈0 → 梯度≈0），RHO 只反向 hard token
梯度——重开的新权重第一口梯度天然来自难数据，躺不回去。
安全三件套：①休眠按**激活**判定（magnitude 小可能是被抑制的重要特征）；
②手术对休眠单元恒等（输入侧随机重开 + 输出侧置零；激活恒零则逐位不变，无 loss 尖峰）；
③单次 cap 占比 + 术前归档 + A/B 验收。
新训练直接新开优化器（无 stale 动量问题）；继续旧优化器需另清对应动量。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def collect_mid_activations(model, batches: list[torch.Tensor],
                            ) -> dict[tuple, torch.Tensor]:
    """收集各专家中间激活的平均|值|（silu(gate)*up），返回 {(layer, kind, expert): (h,)}.

    在 MoE 模块上挂一个 hook（分组 bmm 后专家子模块 forward 不再被调用，
    挂专家上收不到数据），逐专家重算中间激活；流式累加和/计数，
    内存 O(专家数×h)，与校准量无关。backbone 冻结、no_grad，调用方保证 eval。
    """
    sums: dict[tuple, torch.Tensor] = {}
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
                        mid = F.silu(e.w_gate(x)) * e.w_up(x)
                        part = mid.detach().float().abs().sum(
                            dim=tuple(range(mid.dim() - 1)))
                        key = (_li, kind, ei)
                        if key in sums:
                            sums[key] = sums[key] + part.cpu()
                            counts[key] += mid.numel() // mid.shape[-1]
                        else:
                            sums[key] = part.cpu()
                            counts[key] = mid.numel() // mid.shape[-1]

        handles.append(layer.moe.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch)
    finally:
        for hd in handles:
            hd.remove()
    return {key: sums[key] / counts[key] for key in sums}


def dormancy_masks(acts: dict[tuple, torch.Tensor],
                   rel_threshold: float = 1e-3,
                   ) -> tuple[dict[tuple, torch.Tensor],
                              dict[tuple, torch.Tensor], dict]:
    """激活休眠判定：score=神经元平均|激活|，阈值=层均值×rel_threshold.

    返回 ({key: bool mask（True=休眠）}, {key: score}, 报告 {layer: ...})。
    """
    masks: dict[tuple, torch.Tensor] = {}
    scores: dict[tuple, torch.Tensor] = {}
    by_layer: dict[int, list[float]] = {}
    for key, h in acts.items():
        li = key[0]
        # collect 已给逐神经元均值（1 维）则直接用，否则按 token 平均
        score = h.float() if h.dim() == 1 else h.float().abs().mean(0)
        layer_mean = score.mean().item()
        thr = layer_mean * rel_threshold
        mask = score < thr
        masks[key] = mask
        scores[key] = score
        by_layer.setdefault(li, []).append(float(mask.float().mean()))
    report = {li: {"dormant_frac": sum(v) / len(v), "n_experts": len(v)}
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
