"""轻量稀疏注意力 — DSA 思想的诚实简化版（sink + 滑动窗口 + 膨胀跨步）.

DeepSeek V4 的 DSA/CSA 全套需要论文细节与定制 kernel 才能复刻；
这里实现行为明确的子集，专治 200K 上下文的 O(N^2) 灾难：

- 短序列（<= sparse_threshold）走 dense，与 MLAModule 数学完全一致；
- 长序列按块预填充：每块先按静态 pattern 聚集 key（sink + 跨步 + 窗口），
  再在聚集子集上做因果 SDPA，内存 O(chunk * n_selected)，全程可微；
- 解码：从 latent 缓存展开后同样按 pattern 聚集，单步 O(n_selected)。

因果性保证：静态部分全 < 块起点，块内用 tril 掩码（单测锁定）。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import MLAModule


class IndexerSelect(NamedTuple):
    """学习式 indexer 的一次选择结果（供 CSA2 层间复用）.

    topk_mask: (b, n_q, n_static) bool —— token 级最终选择（True=该候选参与注意力）；
    pool_mask: (b, n_q, n_static) bool —— 块级候选池（True=该候选属于入选块），
        供后续 Reindex 层缩小打分范围（复用池、但重选 Top-K）。
    """

    topk_mask: torch.Tensor
    pool_mask: torch.Tensor


def select_keys(
    total: int,
    start: int,
    end: int,
    sink: int,
    window: int,
    stride: int,
) -> tuple[torch.Tensor, int]:
    """为查询区间 [start, end) 选择可见 key 下标（升序）并返回静态部分长度.

    可见集 = sink 区 [0, min(sink,start)) ∪ 跨步区 {j<start: j%stride==0}
           ∪ 窗口 [max(0,start-window), start) ∪ 本块 [start, end)（块内因果另掩）。
    静态部分（前 n_static 个）全 < start，对块内所有查询可见；
    本块部分调用方配 tril 掩码。
    """
    static: set[int] = set(range(0, min(sink, start)))
    static.update(range(0, start, stride))
    static.update(range(max(0, start - window), start))
    static_sorted = sorted(static)
    chunk = list(range(start, end))
    idx = static_sorted + chunk
    return torch.tensor(idx, dtype=torch.long), len(static_sorted)


def chunk_causal_mask(
    n_queries: int, n_static: int, n_chunk: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """块内因果加性掩码：静态列全放行，块列配 tril（0=放行，-inf=屏蔽）."""
    mask = torch.zeros(n_queries, n_static + n_chunk, device=device, dtype=dtype)
    if n_chunk > 0:
        tril = torch.tril(torch.ones(n_queries, n_chunk, device=device, dtype=torch.bool))
        mask[:, n_static:] = torch.where(
            tril, torch.zeros((), device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )
    return mask


class SparseMLAModule(MLAModule):
    """稀疏 MLA：继承 MLA 全部投影与缓存，只替换注意力核心.

    无新增参数，state_dict 与 MLAModule 完全兼容。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_dim: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        yarn_scale: float = 1.0,
        dropout: float = 0.0,
        sparse_threshold: int = 4096,
        sparse_window: int = 4096,
        sparse_sink: int = 128,
        sparse_stride: int = 512,
        sparse_chunk: int = 2048,
        o_groups: int = 1,
        o_lora_rank: int = 0,
        sparse_indexer: bool = False,
        indexer_heads: int = 4,
        indexer_head_dim: int = 64,
        indexer_topk: int = 64,
        indexer_block: int = 8,
        indexer_pool_blocks: int = 32,
    ):
        super().__init__(
            d_model, n_heads, q_lora_rank, kv_lora_rank, qk_rope_dim,
            max_seq_len, rope_theta, yarn_scale, dropout,
            o_groups=o_groups, o_lora_rank=o_lora_rank,
        )
        # 纯超参，不注册为 Parameter/Buffer（state_dict 与 MLA 一致，checkpoint 通用）
        self.sparse_threshold = sparse_threshold
        self.sparse_window = sparse_window
        self.sparse_sink = sparse_sink
        self.sparse_stride = sparse_stride
        self.sparse_chunk = sparse_chunk
        # 分层稀疏 indexer（学习式 top-K；默认关，开启新增 w_iq/w_ik）
        self.sparse_indexer = bool(sparse_indexer)
        self.indexer_heads = int(indexer_heads)
        self.indexer_head_dim = int(indexer_head_dim)
        self.indexer_topk = int(indexer_topk)
        self.indexer_block = int(indexer_block)
        self.indexer_pool_blocks = int(indexer_pool_blocks)
        if self.sparse_indexer:
            # indexer Q 来自 hidden，indexer K 由主 KV latent 投影（V4.1：不另开压缩路径）
            self.w_iq = nn.Linear(
                d_model, self.indexer_heads * self.indexer_head_dim, bias=False)
            self.w_ik = nn.Linear(
                kv_lora_rank, self.indexer_heads * self.indexer_head_dim, bias=False)

    def _indexer_select(
        self,
        h_q: torch.Tensor,
        c_kv: torch.Tensor,
        static_pos: torch.Tensor,
        n_q: int,
        shared: IndexerSelect | None = None,
    ) -> IndexerSelect:
        """学习式 indexer：对静态候选打分 → 块级候选池 → 池内 token 级 top-K.

        h_q: 当块 hidden (b, n_q, d)；c_kv: 全历史 latent (b, L, r)；
        static_pos: (n_static,) 候选全局位置（升序，均 < 块起点，故对块内所有查询可见）；
        shared: 前一层（Full）的选择；给定时只在其候选池内打分（CSA2 Reindex 语义）。

        返回 IndexerSelect（topk_mask/pool_mask 均 (b, n_q, n_static)）。
        打分提 fp32（候选数不大，保数值稳定）。
        """
        b = h_q.shape[0]
        heads, hdim = self.indexer_heads, self.indexer_head_dim
        qi = self.w_iq(h_q).view(b, n_q, heads, hdim).float()
        ki = self.w_ik(c_kv[:, static_pos]).view(b, -1, heads, hdim).float()
        scores = torch.einsum("bqhd,bkhd->bqk", qi, ki) * (hdim ** -0.5)
        n_static = scores.shape[-1]
        flat = scores.reshape(b * n_q, n_static)
        # 块级候选池：候选按其全局位置分块，块分 = 块内最大分
        blk = (static_pos // self.indexer_block).to(scores.device)
        n_blk = int(blk[-1].item()) + 1
        block_scores = flat.new_full((b * n_q, n_blk), float("-inf"))
        block_scores.scatter_reduce_(
            1, blk.view(1, -1).expand(b * n_q, -1), flat,
            reduce="amax", include_self=True)
        top_blk = block_scores.topk(
            min(self.indexer_pool_blocks, n_blk), dim=-1).indices
        sel_blk = block_scores.new_zeros((b * n_q, n_blk), dtype=torch.bool)
        sel_blk.scatter_(1, top_blk, True)
        pool_mask = sel_blk.gather(
            1, blk.view(1, -1).expand(b * n_q, -1)).view(b, n_q, n_static)
        if shared is not None:
            # Reindex：只在前层的候选池内打分；交集为空时退回自己的池（防空掩码→NaN）
            inter = pool_mask & shared.pool_mask
            empty = ~inter.any(-1, keepdim=True)
            pool_mask = torch.where(empty, pool_mask, inter)
        masked = scores.masked_fill(~pool_mask, float("-inf"))
        # 池内全被 mask 的极端情况（不应发生）：退回未 mask 的分数，保证 softmax 有非 -inf
        bad = torch.isinf(masked).all(-1, keepdim=True)
        masked = torch.where(bad, scores, masked)
        topk = masked.topk(min(self.indexer_topk, n_static), dim=-1).indices
        topk_mask = torch.zeros_like(masked, dtype=torch.bool)
        topk_mask.scatter_(2, topk, True)
        return IndexerSelect(topk_mask=topk_mask, pool_mask=pool_mask)

    def _sparse_scores(
        self,
        q_c: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        start: int,
        end: int,
        total: int,
        c_kv: torch.Tensor | None = None,
        h_c: torch.Tensor | None = None,
        shared: "IndexerSelect | None" = None,
    ) -> torch.Tensor:
        """对查询 [start, end) 做聚集 + 掩码 + 注意力（softmax 提 fp32 保精度）.

        q_c 为已切好的查询块 (b, h, C=end-start, d)；k/v 为全历史。
        解码时 q_c 只有当步 1 个位置（下标与全历史不对齐，由 start/end 表达绝对位置）。
        indexer 开启且给了 c_kv/h_c 时：静态候选先经学习式 top-K（块级池 → 池内 top-K），
        未入选的静态列掩 -inf（当前块的本地 SWA 列不受影响）。
        """
        b, h, _, d = k_full.shape
        idx, n_static = select_keys(
            total, start, end, self.sparse_sink, self.sparse_window, self.sparse_stride)
        idx = idx.to(k_full.device)
        # 聚集 key（index_select 可微，反向走 index_add）
        k_sel = k_full[:, :, idx, :].contiguous()  # (b, h, K, d)
        v_sel = v_full[:, :, idx, :].contiguous()
        scale = q_c.shape[-1] ** -0.5
        scores = torch.matmul(q_c, k_sel.transpose(-1, -2)) * scale
        mask = chunk_causal_mask(
            end - start, n_static, end - start, q_c.device, scores.dtype)
        if (self.sparse_indexer and n_static > 0
                and c_kv is not None and h_c is not None):
            sel = self._indexer_select(h_c, c_kv, idx[:n_static], end - start, shared)
            # (b, 1, n_q, n_static)：静态列按 top-K 掩码，块列（本地 SWA）保持因果掩码
            static = mask[:, :n_static].unsqueeze(0).unsqueeze(0).masked_fill(
                ~sel.topk_mask[:, None], float("-inf"))
            chunk = mask[:, n_static:].unsqueeze(0).unsqueeze(0)
            mask = torch.cat(
                [static.expand(b, 1, -1, -1), chunk.expand(b, 1, -1, -1)], dim=-1)
        # softmax 提 fp32（bf16 下长序列 exp 易欠精），再转回
        probs = F.softmax(scores.float() + mask.float(), dim=-1).to(scores.dtype)
        if self.training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        return torch.matmul(probs, v_sel)

    def _full_qkv_for_gather(
        self, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 hidden 算全序列 q/k/v（含 rope 拼接）+ latent 缓存原料.

        返回 (q_full, k_full, v_full, c_kv, k_rope_raw)，一次投影处处复用。
        """
        q, k, v, q_rope, k_rope, c_kv, k_rope_raw = self._qkv(h)
        k_rope_full = k_rope.expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_full], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        return q_full, k_full, v_full, c_kv, k_rope_raw

    def forward(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = True,
        reserve: int = 0,
        reuse: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """稀疏前向：短序列/解码缓存命中时退化为 dense 等价，超长才分块聚集.

        reuse: 跨层复用（CSA2）最近 Full 层的 latent 缓存；None=正常自算。
        """
        b, t, _ = h.shape
        if past is not None:
            return self._decode(h, past)
        if reuse is not None:
            # 跨层复用：Q 自算、K/V 借共享 latent（短序列走 dense，超长走分块聚集）
            if t <= self.sparse_threshold:
                return super().forward(h, None, return_state, reserve, reuse)
            q_full = self._q_only(h)
            k_full, v_full = self._kv_from_latent(reuse[0], reuse[1], total=t)
            outs = []
            for s in range(0, t, self.sparse_chunk):
                e = min(s + self.sparse_chunk, t)
                outs.append(self._sparse_scores(
                    q_full[:, :, s:e, :].contiguous(), k_full, v_full, s, e, t,
                    c_kv=reuse[0], h_c=h[:, s:e]))
            out = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, t, -1)
            return self._out_proj(out), (reuse if return_state else None)
        if t <= self.sparse_threshold:
            # 短序列：与 MLAModule 逐位一致（reserve 透传装箱）
            return super().forward(h, None, return_state, reserve)
        # 长序列分块预填充
        q_full, k_full, v_full, c_kv, k_rope_raw = self._full_qkv_for_gather(h)
        outs = []
        for s in range(0, t, self.sparse_chunk):
            e = min(s + self.sparse_chunk, t)
            outs.append(self._sparse_scores(
                q_full[:, :, s:e, :].contiguous(), k_full, v_full, s, e, t,
                c_kv=c_kv, h_c=h[:, s:e]))
        out = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, t, -1)
        if not return_state:
            return self._out_proj(out), None
        # 缓存仍存 latent（与 MLA 一致，200K 上下文每 token 仅 ~320B/层），
        # 静态装箱（reserve 由 _prefill 按 prompt+max_new 精确给出）
        return self._out_proj(out), self._pack_cache(c_kv, k_rope_raw, max(reserve, t))

    @torch._dynamo.disable
    def _decode(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """增量解码：latent 缓存展开后按 pattern 聚集（短上下文自动退化为全量）.

        静态缓存版：按位写进预分配缓冲（不再 `cat`），pattern 按实际长度算
        （gather 下标全落在有效区，无需额外掩码，与 cat 版逐位一致）。
        """
        b, t, _ = h.shape
        buf_c, buf_kr, pos = past
        if not self.csa_reuse:
            # Full 层：按位写自己的 latent；Reuse 层跳过（共享缓冲已由 Full 写好）
            w_idx = pos.reshape(1) + torch.arange(t, device=h.device)
            buf_c.index_copy_(1, w_idx, self.w_dkv(h))
            buf_kr.index_copy_(1, w_idx, self.w_kr(h))
        pos_next = pos + t
        # 实际长度（pattern 按实际算；一次同步，eager 可忽略；
        # compile 下此处 graph-break 但不重编）
        total = int(pos_next.item())
        # 当步 Q（含绝对位置 RoPE）
        q_c = self.w_dq(h)
        q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        off = total - t
        q_rope = apply_rope_wrap(q_rope, self.rope_cos[off:off + t], self.rope_sin[off:off + t])
        # 全缓冲展开（latent 省的是缓存不是计算，单步展开可接受；
        # pattern 只 gather 有效下标，空位不参与；rope 下标钳制防 L 超缓存）
        k = self.k_norm(self.w_uk(buf_c).view(b, -1, self.n_heads, self.head_dim))
        v = self.w_uv(buf_c).view(b, -1, self.n_heads, self.head_dim)
        _rlen = self.rope_cos.shape[0]
        _ridx = torch.arange(buf_kr.shape[1], device=h.device).clamp(max=_rlen - 1)
        k_rope_hist = apply_rope_wrap(
            buf_kr,
            self.rope_cos[_ridx],
            self.rope_sin[_ridx],
        ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_hist], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        # 查询即最后 t 个位置：select_keys(total, total-t, total)，t==1 时与旧式一致；
        # t>1 为投机解码并行验证服务（块内因果由 chunk_causal_mask 保证）
        out = self._sparse_scores(q_full, k_full, v_full, total - t, total, total,
                                  c_kv=buf_c, h_c=h)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self._out_proj(out), (buf_c, buf_kr, pos_next)


def apply_rope_wrap(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """apply_rope 的局部别名（避免循环导入，保持调用点清晰）."""
    from src.llm.local.rope import apply_rope

    return apply_rope(x, cos, sin)
