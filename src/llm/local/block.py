"""Hybrid Transformer 块 — MLA 全注意力层与线性注意力层交替 + MoE + 可选记忆层."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.hyperconn import HyperConnRes
from src.llm.local.memory import ProductKeyMemory
from src.llm.local.mla import RMSNorm
from src.llm.local.moe import FineGrainedMoE
from src.llm.local.retro import RetroFusion
from src.llm.local.sparse_attn import SparseMLAModule


class HybridBlock(nn.Module):
    """单个 Hybrid 块：注意力（MLA/线性二选一）+ 细粒度 MoE + 可选记忆层 + 可选 RETRO 交错融合."""

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
        use_memory: bool = False,
        memory_slots: int = 4096,
        memory_topk: int = 8,
        use_retro: bool = False,
        retro_heads: int = 8,
        hyper_streams: int = 0,
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
        # 记忆层（默认 None：无参数，state_dict 兼容；开后残差并联在 MoE 之后）
        self.memory: ProductKeyMemory | None = (
            ProductKeyMemory(d_model, memory_slots, memory_topk, dropout)
            if use_memory else None
        )
        # RETRO 交错融合（默认 None；开后在记忆层之后再残差并联，同构零初始化恒等）
        self.retro: RetroFusion | None = (
            RetroFusion(d_model, retro_heads, dropout)
            if use_retro else None
        )
        # 超连接残差（默认 None；开后 block 输入输出为 n 路流，恒等起点）
        assert hyper_streams == 0 or hyper_streams >= 2, \
            "hyper_streams 须为 0（关）或 ≥2（n=1 退化恒等，无意义）"
        self.hyper: HyperConnRes | None = (
            HyperConnRes(d_model, hyper_streams)
            if hyper_streams >= 2 else None
        )

    def forward(
        self,
        h: torch.Tensor,
        past: tuple | torch.Tensor | None = None,
        return_state: bool = True,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        reserve: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
        """前向：past 为解码缓存（MLA 传 latent 元组，线性层传状态矩阵）.

        return_state=False 时注意力层不返回缓存（训练路径）。
        retro_mem=(mem_h, mem_mask)：V2 交错融合的 frozen chunk 编码；
        None 或本层无融合块时跳过（backbone 本体，评测/生成路径）。
        reserve: prefill 预留总长（只用于全注意力静态缓存装箱；线性层忽略）。
        h: 无超连接时 (b, t, d)；有超连接时 (b, t, n, d) n 路流。
        """
        xs = h
        h_in = self.hyper.pre_mix(h) if self.hyper is not None else h
        h = h_in
        if past is None and self.full_attn:
            a_out, new_past = self.attn(  # type: ignore[call-arg]
                self.norm1(h), past, return_state, reserve)
        else:
            a_out, new_past = self.attn(self.norm1(h), past, return_state)  # type: ignore[arg-type]
        h = h + a_out
        m_out, aux = self.moe(self.norm2(h))
        h = h + m_out
        if self.memory is not None:
            # 记忆层自带 norm + 残差，直接叠加
            h = self.memory(h)
        if self.retro is not None and retro_mem is not None:
            mh, mm = retro_mem
            h = self.retro(h, mh, mm)
        if self.hyper is not None:
            # 超连接写回：F 为本块在均值流上的总增量，流混合后按 post 写回各路
            h = self.hyper.combine(xs, h - h_in)
        return h, (new_past, aux)
