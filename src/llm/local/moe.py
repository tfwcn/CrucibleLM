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

        分组 bmm：每个 token 展开成 top-k 行，按专家排序后垫齐 batch，
        3 次 bmm 算完所有专家（替代 16 次逐专家 Python dispatch，
        解码主力开销），再按权重散射加回。数值与逐专家版一致（单测锁定）。
        """
        orig_shape = x.shape
        d = x.shape[-1]
        flat = x.reshape(-1, d)  # (N, d)
        n = flat.shape[0]
        logits = self.router(flat)  # (N, n_experts)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-6)
        # 展开：每 token k 行（token 序号、专家号、归一化权重）
        m = n * self.top_k
        tok_idx = torch.arange(n, device=x.device).repeat_interleave(self.top_k)
        exp_idx = top_i.reshape(-1)
        # 权重对齐到输入精度（AMP 下 top_w 常为 fp32：避免 index_put 报类型错，
        # 也避免旧式 where 隐式提升把残差流抬成 fp32）
        w_row = top_w.reshape(-1).to(flat.dtype)
        order = torch.argsort(exp_idx, stable=True)
        exp_s = exp_idx[order]
        counts = torch.bincount(exp_s, minlength=self.n_experts)
        max_n = int(counts.max().item())
        # 组内槽位：slot[r] = 该行在其专家组内的序号
        cum = counts.cumsum(0)
        slot = torch.arange(m, device=x.device) - (cum[exp_s] - counts[exp_s])
        # 画布 (E, max_n, *)：token / 权重 / 有效掩码
        canvas = torch.zeros(self.n_experts, max_n, d,
                             device=x.device, dtype=flat.dtype)
        canvas[exp_s, slot] = flat.repeat_interleave(self.top_k, dim=0)[order]
        w_canvas = torch.zeros(self.n_experts, max_n,
                               device=x.device, dtype=flat.dtype)
        w_canvas[exp_s, slot] = w_row[order]
        row_mask = torch.zeros(self.n_experts, max_n,
                               device=x.device, dtype=torch.bool)
        row_mask[exp_s, slot] = True
        # 专家权重堆叠（view 级 stack，反向精确回传到各专家参数）
        w_g = torch.stack([e.w_gate.weight for e in self.experts])  # (E, h, d)
        w_u = torch.stack([e.w_up.weight for e in self.experts])
        w_d = torch.stack([e.w_down.weight for e in self.experts])  # (E, d, h)
        gate = F.silu(canvas @ w_g.transpose(1, 2)) * (canvas @ w_u.transpose(1, 2))
        out_e = (gate @ w_d.transpose(1, 2)) * (
            w_canvas * row_mask.to(flat.dtype)).unsqueeze(-1)
        out = torch.zeros_like(flat)
        out.index_add_(0, tok_idx[order], out_e[exp_s, slot])
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
