"""Hybrid Transformer 块 — MLA 全注意力层与线性注意力层交替 + MoE."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.mla import RMSNorm
from src.llm.local.moe import FineGrainedMoE
from src.llm.local.sparse_attn import SparseMLAModule


class HybridBlock(nn.Module):
    """单个 Hybrid 块：注意力（MLA/线性二选一）+ 细粒度 MoE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        full_attn: bool,
        q_lora_rank: int = 192,
        kv_lora_rank: int = 128,
        qk_rope_dim: int = 32,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        yarn_scale: float = 1.0,
        n_experts: int = 16,
        top_k: int = 4,
        expert_hidden: int = 192,
        n_shared: int = 1,
        aux_coef: float = 0.01,
        dropout: float = 0.0,
        sparse_threshold: int = 4096,
        sparse_window: int = 4096,
        sparse_sink: int = 128,
        sparse_stride: int = 512,
        sparse_chunk: int = 2048,
        linear_chunk: int = 2048,
    ):
        super().__init__()
        self.full_attn = full_attn
        self.norm1 = RMSNorm(d_model)
        if full_attn:
            # DeepSeek 系 MLA 全注意力 + DSA 思想稀疏（短序列自动退化为 dense）
            self.attn: nn.Module = SparseMLAModule(
                d_model, n_heads, q_lora_rank, kv_lora_rank, qk_rope_dim,
                max_seq_len, rope_theta, yarn_scale, dropout,
                sparse_threshold, sparse_window, sparse_sink,
                sparse_stride, sparse_chunk,
            )
        else:
            # Qwen3-Next 系线性注意力（超长自动切块递推）
            self.attn = GatedDeltaLite(d_model, n_heads, dropout, linear_chunk)
        self.norm2 = RMSNorm(d_model)
        self.moe = FineGrainedMoE(
            d_model, n_experts, top_k, expert_hidden, n_shared, aux_coef,
        )

    def forward(
        self,
        h: torch.Tensor,
        past: tuple | torch.Tensor | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
        """前向：past 为解码缓存（MLA 传 latent 元组，线性层传状态矩阵）.

        return_state=False 时注意力层不返回缓存（训练路径）。
        """
        a_out, new_past = self.attn(self.norm1(h), past, return_state)  # type: ignore[arg-type]
        h = h + a_out
        m_out, aux = self.moe(self.norm2(h))
        return h + m_out, (new_past, aux)
