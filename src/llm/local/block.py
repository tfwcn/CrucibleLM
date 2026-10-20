"""Hybrid Transformer 块 — MLA 全注意力层与线性注意力层交替 + MoE + 可选记忆层."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.llm.local.engram import EngramMemory
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
        swiglu_limit: float = 10.0,
        router_aux_free: bool = True,
        router_bias_lr: float = 1e-3,
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
        o_groups: int = 1,
        o_lora_rank: int = 0,
        sparse_indexer: bool = False,
        indexer_heads: int = 4,
        indexer_head_dim: int = 64,
        indexer_topk: int = 64,
        indexer_block: int = 8,
        indexer_pool_blocks: int = 32,
        indexer_loss_coef: float = 0.01,
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
        use_engram: bool = False,
        engram_orders: tuple[int, ...] | list[int] = (2, 3, 4),
        engram_heads: int = 8,
        engram_d_mem: int = 256,
        engram_table_size: int = 8191,
        engram_fp8: bool = True,
        vocab_size: int = 8192,
        csa_mode: str = "full",
        compress_ratio: int = 1,
        swa_window: int = 0,
        ced: bool = False,
        kv_quant: str = "",
        block_cache: bool = False,
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
                o_groups=o_groups, o_lora_rank=o_lora_rank,
                sparse_indexer=sparse_indexer, indexer_heads=indexer_heads,
                indexer_head_dim=indexer_head_dim, indexer_topk=indexer_topk,
                indexer_block=indexer_block,
                indexer_pool_blocks=indexer_pool_blocks,
                indexer_loss_coef=indexer_loss_coef,
                compress_ratio=compress_ratio,
                swa_window=swa_window,
                ced=ced,
                kv_fmt=kv_quant,
                block_cache=block_cache,
            )
            # CSA2 三模式（V4.1）：full 自算自存；reindex/reuse 借生产者 main KV
            # （`csa_reuse=True` 时解码不写自己的缓冲），纯超参不进 state_dict
            self.attn.csa_mode = csa_mode
            self.attn.csa_reuse = csa_mode != "full"
        else:
            # Qwen3-Next 系线性注意力（超长自动切块递推）
            self.attn = GatedDeltaLite(d_model, n_heads, dropout, linear_chunk)
        self.norm2 = RMSNorm(d_model)
        self.moe = FineGrainedMoE(
            d_model, n_experts, top_k, expert_hidden, n_shared, aux_coef,
            swiglu_limit, router_aux_free, router_bias_lr,
            routed_scaling_factor, norm_topk_prob,
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
        # Engram 条件记忆（默认 None；开后在注意力之前残差并联，恒等起点）
        self.engram: EngramMemory | None = (
            EngramMemory(d_model, vocab_size, engram_orders, engram_heads,
                         engram_d_mem, engram_table_size, engram_fp8, dropout)
            if use_engram else None
        )

    def forward(
        self,
        h: torch.Tensor,
        past: tuple | torch.Tensor | None = None,
        return_state: bool = True,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        reserve: int = 0,
        engram_ids: torch.Tensor | None = None,
        reuse: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        csa_sel: object | None = None,
        ced_src: torch.Tensor | None = None,
        q_off: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
        """前向：past 为解码缓存（MLA 传 latent 元组，线性层传状态矩阵）.

        return_state=False 时注意力层不返回缓存（训练路径）。
        retro_mem=(mem_h, mem_mask)：V2 交错融合的 frozen chunk 编码；
        None 或本层无融合块时跳过（backbone 本体，评测/生成路径）。
        reserve: prefill 预留总长（只用于全注意力静态缓存装箱；线性层忽略）。
        engram_ids: 本层 Engram 的上下文 token id（(b,L)，右对齐到 h 末位）；
        None 或本层无 Engram 时跳过。
        reuse: CSA2 跨层复用（最近生产者的 latent 缓存），全注意力层 prefill 用。
        csa_sel: CSA2 层间选择通道（`sparse_indexer` 开启时由 TinyLLM 建，逐层共享）。
        ced_src: CED 解码器层的编码器输出 `H_enc`（KV/rope 键由它投影；None=本层自算）。
        q_off: 查询首 token 的绝对位置（CED 跳算预填充时 `h` 只覆盖尾部）。
        h: 无超连接时 (b, t, d)；有超连接时 (b, t, n, d) n 路流。
        """
        xs = h
        h_in = self.hyper.pre_mix(h) if self.hyper is not None else h
        h = h_in
        if self.engram is not None and engram_ids is not None:
            # Engram 在注意力之前：先把静态模式"查"出来（论文：早期注入最优，
            # 替主干省下早期层的特征组装）；h_in 不变，超连接写回仍以 h_in 为基准
            h = self.engram(h, engram_ids)
        if self.full_attn:
            a_out, new_past = self.attn(  # type: ignore[call-arg]
                self.norm1(h), past, return_state, reserve, reuse, csa_sel,
                ced_src=ced_src, q_off=q_off)
        else:
            if ced_src is not None or q_off:
                raise ValueError(
                    "线性注意力层不支持 CED（递推状态无法从编码器输出投影）")
            a_out, new_past = self.attn(self.norm1(h), past, return_state)  # type: ignore[arg-type]
        h = h + a_out
        m_out, aux = self.moe(self.norm2(h))
        # indexer 蒸馏损失（仅有 indexer 的全注意力层提供；进各层 aux 统一汇总）
        idx_aux = getattr(self.attn, "_last_indexer_aux", None)
        if idx_aux is not None:
            aux = aux + idx_aux
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
