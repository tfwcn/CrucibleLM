"""架构迁移工具箱 — 改结构不重交学费（bert2BERT/Upcycling/SVD 路线）.

精确性分级（必须分清）：
- 精确（数学恒等）：加恒等层、专家加宽（Net2Net），迁移前后输出逐位一致；
- 近似（有损耗）：层子集、SVD 低秩分解，附重构误差上界，用后需短训恢复；
- 跨架构映射（如 GQA->MLA）只给 SVD 工具 + 手册配方，不做"看起来很美"的假精确。

所有操作就地改模型 + 同步改 config（层数/维度），state_dict 键名不变，
旧 checkpoint 仍可按行续接（见 scripts/train_local_llm.py:load_weights_overlap）。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.llm.local.model import TinyLLM, build_block


def add_identity_layers(model: TinyLLM, n_add: int) -> None:
    """追加恒等层（bert2BERT 式加深）：输出投影置零，pre-norm 架构下块恒等.

    新层输出恒为输入，迁移前后模型输出逐位一致，loss 从旧终点起步。
    """
    config = model.config
    base = len(model.layers)
    for i in range(n_add):
        layer = build_block(config, base + i)
        # 置零残差输出：注意力输出投影 + 所有专家（含共享）的下投影
        with torch.no_grad():
            layer.attn.w_o.weight.zero_()
            for expert in list(layer.moe.experts) + list(layer.moe.shared):
                expert.w_down.weight.zero_()
        model.layers.append(layer)
    config.n_layers = len(model.layers)


def widen_experts(model: TinyLLM, new_hidden: int, seed: int = 0) -> None:
    """专家加宽（Net2WiderNet）：复制 hidden 单元并平分下投影行权重.

    每个专家的输出恒等（复制对的权重和不变），迁移前后逐位一致。
    new_hidden 必须大于当前值。
    """
    config = model.config
    old_hidden = config.expert_hidden
    if new_hidden <= old_hidden:
        raise ValueError(f"new_hidden={new_hidden} 必须大于当前 {old_hidden}")
    g = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        for expert in list(layer.moe.experts) + list(layer.moe.shared):
            _widen_swiglu(expert, old_hidden, new_hidden, g)
    config.expert_hidden = new_hidden


def _widen_swiglu(
    expert: nn.Module, old_hidden: int, new_hidden: int, g: torch.Generator
) -> None:
    """单个 SwiGLU 专家加宽：新增单元从旧单元均匀采样复制，下投影行减半."""
    with torch.no_grad():
        extra = new_hidden - old_hidden
        # 被复制的源单元（均匀采样，可重复；重复时权重继续平分，仍精确）
        src = torch.randint(0, old_hidden, (extra,), generator=g)
        w_gate = torch.cat([expert.w_gate.weight.data,
                            expert.w_gate.weight.data[src]], dim=0)
        w_up = torch.cat([expert.w_up.weight.data,
                          expert.w_up.weight.data[src]], dim=0)
        w_down_old = expert.w_down.weight.data  # (d_model, hidden)：hidden 在 dim1
        # 计数每个源被复制几次（含自身 1 次），下投影列按份数平分
        counts = torch.ones(old_hidden, dtype=torch.float32)
        counts.index_add_(0, src, torch.ones(extra))
        w_down_old = w_down_old / counts.unsqueeze(0).to(w_down_old.dtype)
        w_down_new = torch.cat([w_down_old, w_down_old[:, src]], dim=1)
        expert.w_gate.weight = nn.Parameter(w_gate)
        expert.w_up.weight = nn.Parameter(w_up)
        expert.w_down.weight = nn.Parameter(w_down_new)


def subselect_layers(model: TinyLLM, keep: list[int]) -> None:
    """层子集（变浅/变窄结构用，有损耗：删掉的层知识丢失，需短训恢复）."""
    config = model.config
    kept = [copy.deepcopy(model.layers[i]) for i in keep]
    model.layers = nn.ModuleList(kept)
    config.n_layers = len(kept)


def svd_split(
    weight: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """截断 SVD 分解 W ≈ A@B（A: down, B: up），返回 (A, B, 相对重构误差）.

    用途：跨架构映射（如老师 K 投影 -> MLA 的 W_DKV/W_UK），
    或宽矩阵压缩。误差 = 1 - top-rank 奇异值能量占比。
    """
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    rank = min(rank, s.numel())
    energy = float((s[:rank] ** 2).sum() / (s ** 2).sum().clamp_min(1e-12))
    sq = torch.sqrt(s[:rank]).to(weight.dtype)
    down = (u[:, :rank] * sq.unsqueeze(0)).to(weight.dtype)
    up = (sq.unsqueeze(1) * vh[:rank, :]).to(weight.dtype)
    return down, up, round(1 - energy, 4)
