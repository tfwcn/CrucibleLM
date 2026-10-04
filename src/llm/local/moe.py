"""细粒度 MoE — DeepSeek 系多小专家 + 共享专家 + 负载均衡 aux loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """单个 SwiGLU 前馈专家（小中间维度）."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前向."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FineGrainedMoE(nn.Module):
    """细粒度 MoE：top-k 路由 + 共享专家兜底."""

    def __init__(
        self,
        d_model: int,
        n_experts: int = 16,
        top_k: int = 4,
        expert_hidden: int = 192,
        n_shared: int = 1,
        aux_coef: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_coef = aux_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden) for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden) for _ in range(n_shared)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向并返回 (输出, aux_loss).

        aux_loss: 标准负载均衡损失 mean(p) 与 one-hot 负载的点积，
        防止路由坍缩到少数专家。

        专家分组计算：每个专家只算分给它的 token（而非全量前向再 mask），
        总计算量从 n_experts 降到约 top_k，数值与全量版一致。
        """
        orig_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])  # (N, d)
        logits = self.router(flat)  # (N, n_experts)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-6)
        # 加权聚合被选中的专家输出（小模型逐专家 mask，清晰优先）
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            # 该专家在 top-k 各槽位中的权重求和
            pick = top_i == e  # (N, top_k)
            if not pick.any():
                continue
            sel = pick.any(-1)  # (N,) 分给该专家的 token
            # 权重对齐到输入精度（AMP 下 top_w 常为 fp32：避免 index_put 报类型错，
            # 也避免旧式 where 隐式提升把残差流抬成 fp32）
            w_sum = (top_w * pick.to(top_w.dtype)).sum(-1).to(flat.dtype)  # (N,)
            contrib = torch.zeros_like(flat)
            contrib[sel] = w_sum[sel].unsqueeze(-1) * expert(flat[sel])
            out = out + contrib
        out = out.view(orig_shape)
        # 共享专家直通（常驻知识通道）
        for expert in self.shared:
            out = out + expert(x)
        # 负载均衡 aux loss：专家重要性分布与实际负载分布的点积
        flat_b = top_i.numel() // self.top_k
        importance = probs.reshape(-1, self.n_experts).mean(0)
        load_frac = (
            F.one_hot(top_i.reshape(-1), self.n_experts).float().sum(0)
            / max(flat_b * self.top_k, 1)
        )
        aux_loss = (importance * load_frac).sum() * self.n_experts * self.aux_coef
        return out, aux_loss
