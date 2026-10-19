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
        供后续 Reindex 层缩小打分范围（复用池、但重选 Top-K）；
    logits: (b, n_q, n_static) fp32 —— 候选上的**未掩** indexer 打分（蒸馏损失用；
        只有生产者 full/reindex 有，Reuse 层直接复用时为生产者的打分）。
    """

    topk_mask: torch.Tensor
    pool_mask: torch.Tensor
    logits: torch.Tensor | None = None


class CsaChannel:
    """CSA2 层间选择通道：生产者（full/reindex）写、消费者（reindex/reuse）读.

    **按"层序号"键控**（`items[层序号][块序号]`），而不是按"轮到的第几层"：
    每层只写自己的槽，消费者按 `csa_src`（最近生产者的层序号）去读。
    这样在梯度检查点反向重算时，任何层读到的都是它的生产者在**上一次前向**写下的
    同值内容（重算写入幂等），不会读到被"后面的层"覆盖掉的值（Reindex 层会覆写
    "最新选择"，若共用单一槽会让重算数学与首次前向不一致 → 静默错梯度）。

    - 预填充：块序号 = 该层分块循环的下标（各层 `sparse_chunk` 一致故对齐）；
    - 解码：单步单块，块序号恒为 0，生产者每步重写自己的槽；
    - 只有 `sparse_indexer` 开启且真有借用层时才创建（否则 `None`，零开销）。

    V4.1 CSA2 三模式静态分配靠它落地：Full 造池 + 选 Top-K；Reindex 借池
    （复用 main KV 与 indexer K）但用自己的 indexer Q 重选 Top-K；Reuse 直接
    取最近生产者槽里最新的 Top-K。
    """

    def __init__(self) -> None:
        self.items: dict[int, list["IndexerSelect | None"]] = {}

    def get(self, layer: "int | None", chunk: int) -> "IndexerSelect | None":
        """取某层第 chunk 块的选择（未写过/层号为空返回 None）."""
        if layer is None:
            return None
        per = self.items.get(layer)
        if per is None or chunk >= len(per):
            return None
        return per[chunk]

    def put(self, layer: int, chunk: int, sel: "IndexerSelect") -> None:
        """写某层第 chunk 块的选择（同一 forward 内按层序、块序推进）."""
        per = self.items.setdefault(layer, [])
        while len(per) <= chunk:
            per.append(None)
        per[chunk] = sel


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


def select_block_keys(
    n_blk: int,
    sink: int,
    stride: int,
    m: int,
) -> torch.Tensor:
    """压缩路径的**远端块 id**（升序）：开头 ceil(sink/m) 块 ∪ 每 max(1, stride//m) 块取一块.

    `n_blk` 为可用的"完整且因果安全"块数（见 `SparseMLAModule._far_block_ids`）。
    语义与 `select_keys` 的 sink/stride 一致，只是**单位从 token 换成 m 块**——
    同样的远端列数预算下，覆盖的历史跨度 ×m。
    """
    if n_blk <= 0:
        return torch.zeros(0, dtype=torch.long)
    sink_blk = min(n_blk, (sink + m - 1) // m)
    stride_blk = max(1, stride // m)
    ids = sorted(set(range(sink_blk)) | set(range(0, n_blk, stride_blk)))
    return torch.tensor(ids, dtype=torch.long)


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
        indexer_loss_coef: float = 0.01,
        compress_ratio: int = 1,
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
        # indexer 蒸馏损失系数（DSA：学习式打分拟合主注意力分布；0=不训 indexer）
        self.indexer_loss_coef = float(indexer_loss_coef)
        # 主 KV 压缩比 m（V4.1 CSA2(ratio, mode)）：远端 main KV 按 m 个连续 token
        # 不重叠地池化成一个条目；m=1 = 不压缩（逐位等价旧行为，无新增键）。
        self.compress_ratio = int(compress_ratio)
        if self.compress_ratio > 1:
            # 压缩权重：(m, kv_lora_rank) 的逐位置逐通道权重，初值 1 → 块内算术平均。
            # 只对 m>1 的层建参数（改键）→ 默认关（`compress_ratios` 为空）。
            self.z_cmp = nn.Parameter(torch.ones(self.compress_ratio, kv_lora_rank))
        # 最近一次前向的 indexer 辅助损失（已缩放/未缩放；赋值语义，重算幂等）
        self._last_indexer_aux: torch.Tensor | None = None
        self._last_indexer_kl: torch.Tensor | None = None
        if self.sparse_indexer:
            # indexer Q 来自 hidden，indexer K 由主 KV latent 投影（V4.1：不另开压缩路径）
            self.w_iq = nn.Linear(
                d_model, self.indexer_heads * self.indexer_head_dim, bias=False)
            self.w_ik = nn.Linear(
                kv_lora_rank, self.indexer_heads * self.indexer_head_dim, bias=False)
        # CSA2 三模式（由 TinyLLM.build_block / _link_csa_reuse 填，纯超参）：
        # csa_index = 本层层序号（写通道用）；csa_src = 最近生产者的层序号（读通道用）
        self.csa_index: int | None = None
        self.csa_src: int | None = None

    def _indexer_select(
        self,
        h_q: torch.Tensor,
        c_cols: torch.Tensor,
        col_pos: torch.Tensor,
        n_q: int,
        shared: IndexerSelect | None = None,
    ) -> IndexerSelect:
        """学习式 indexer：对静态候选打分 → 块级候选池 → 池内 token 级 top-K.

        h_q: 当块 hidden (b, n_q, d)；c_cols: **候选列自己的 latent** (b, n_static, r)
        （压缩路径下是池化后的块 latent）；col_pos: 候选列的全局位置 (n_static,)
        （压缩路径下取块尾位置，用于块级候选池分块）；
        shared: 前一层（Full）的选择；给定时只在其候选池内打分（CSA2 Reindex 语义）。

        返回 IndexerSelect（topk_mask/pool_mask 均 (b, n_q, n_static)）。
        打分提 fp32（候选数不大，保数值稳定）。
        """
        b = h_q.shape[0]
        heads, hdim = self.indexer_heads, self.indexer_head_dim
        qi = self.w_iq(h_q).view(b, n_q, heads, hdim).float()
        ki = self.w_ik(c_cols).view(b, -1, heads, hdim).float()
        scores = torch.einsum("bqhd,bkhd->bqk", qi, ki) * (hdim ** -0.5)
        n_static = scores.shape[-1]
        flat = scores.reshape(b * n_q, n_static)
        # 块级候选池：候选按其全局位置分块，块分 = 块内最大分
        blk = (col_pos // self.indexer_block).to(scores.device)
        n_blk = int(blk.max().item()) + 1  # 压缩路径下列位置非严格升序，取 max 更稳
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
        return IndexerSelect(topk_mask=topk_mask, pool_mask=pool_mask,
                             logits=scores)

    def _far_block_ids(self, total: int, start: int) -> torch.Tensor:
        """远端（压缩）块 id：只取"完整且在查询块之前"的块（因果安全）.

        - 完整：块尾 `(b+1)*m-1 < total`（缓存/当前段里真有这些行）；
        - 因果：块尾 `< start`（在查询块之前，绝不看未来 token）；
        - 只在 `start - sparse_window` 之前：更近的位置由**未压缩**的窗口/本块列覆盖。
        建议 `m ≤ sparse_window`（否则窗口与压缩列之间会漏掉不足一个块的近端 token）。
        """
        m = self.compress_ratio
        n_valid = min(total // m, start // m)
        n_far = min(n_valid, max(0, (start - self.sparse_window + m - 1) // m))
        return select_block_keys(n_far, self.sparse_sink, self.sparse_stride, m)

    def _compressed_columns(
        self, c_src: torch.Tensor, kr_src: torch.Tensor, blk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """远端压缩块列：latent 按 m 块池化（可学习权重）→ (c_blk, k_blk, v_blk, blk_end).

        块内不重叠、覆盖 `[b*m, (b+1)*m)`；rope 用**块尾位置**（该条目覆盖到块尾为止）。
        池化 = `sum_i z_i ⊙ c_i / m`（z 初值 1 → 算术平均），`index_select` 可微。
        """
        m = self.compress_ratio
        pos = (blk_ids[:, None] * m
               + torch.arange(m, device=c_src.device)[None, :])   # (B, m)
        members = c_src[:, pos]                                    # (b, B, m, r)
        c_blk = (members * self.z_cmp).sum(dim=2) / m
        blk_end = (blk_ids + 1) * m - 1
        kr_blk = apply_rope_wrap(
            kr_src[:, pos[:, -1]], self.rope_cos[blk_end], self.rope_sin[blk_end])
        b = c_src.shape[0]
        k = self.k_norm(self.w_uk(c_blk).view(b, -1, self.n_heads, self.head_dim))
        v = self.w_uv(c_blk).view(b, -1, self.n_heads, self.head_dim)
        k = torch.cat(
            [k, kr_blk.unsqueeze(2).expand(-1, -1, self.n_heads, -1)],
            dim=-1).transpose(1, 2)
        return c_blk, k, v.transpose(1, 2), blk_end

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
        kr_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, "IndexerSelect | None", torch.Tensor | None]:
        """对查询 [start, end) 做聚集 + 掩码 + 注意力（softmax 提 fp32 保精度）.

        q_c 为已切好的查询块 (b, h, C=end-start, d)；k/v 为全历史。
        解码时 q_c 只有当步 1 个位置（下标与全历史不对齐，由 start/end 表达绝对位置）。
        indexer 开启且给了 c_kv/h_c 时按 `self.csa_mode` 选主 KV（块级池 → 池内
        top-K），未入选的静态列掩 -inf（当前块的本地 SWA 列不受影响）：
        - `full`：全候选上造池 + 自选 Top-K；
        - `reindex`：只在 `shared`（前层）的候选池内打分，用**自己的** indexer Q 重选；
        - `reuse`：直接复用 `shared` 的 Top-K（不算 indexer）；`shared` 缺失时退化为
          `full` 自选（安全兜底）。
        `compress_ratio > 1` 时远端列改由 `kr_raw`/`c_kv` 按 m 块池化而来（见
        `_compressed_columns`），本地窗口与当前块保持未压缩。
        返回 (注意力输出, 本层选择, indexer 蒸馏 KL)。
        训练且为生产者（full/reindex）时额外算**蒸馏损失**（DSA 做法：让学习式打分
        去拟合主注意力在候选上的分布）——否则 indexer 只进掩码、拿不到任何梯度。
        """
        b, h, _, d = k_full.shape
        n_q = end - start
        cols_c: torch.Tensor | None = None
        cols_pos: torch.Tensor | None = None
        if self.compress_ratio > 1 and c_kv is not None and kr_raw is not None:
            # 压缩路径（V4.1 CSA2(ratio, mode)）：远端 main KV 压成 m 块列，
            # 本地窗口 + 当前块保持未压缩（SWA KV 不压缩，逐层本地）。
            blk_ids = self._far_block_ids(total, start).to(k_full.device)
            local = torch.arange(
                max(0, start - self.sparse_window), end, device=k_full.device)
            c_blk, k_blk, v_blk, blk_end = self._compressed_columns(
                c_kv, kr_raw, blk_ids)
            k_cols = torch.cat([k_blk, k_full[:, :, local, :]], dim=2).contiguous()
            v_cols = torch.cat([v_blk, v_full[:, :, local, :]], dim=2).contiguous()
            # 静态列 = 压缩块列 + 窗口列（窗口全 < start，对块内所有查询可见）
            n_near = local.numel() - n_q
            cols_c = torch.cat([c_blk, c_kv[:, local[:n_near]]], dim=1)
            cols_pos = torch.cat([blk_end, local[:n_near]])
            n_static = blk_ids.numel() + n_near
        else:
            idx, n_static = select_keys(
                total, start, end, self.sparse_sink, self.sparse_window, self.sparse_stride)
            idx = idx.to(k_full.device)
            # 聚集 key（index_select 可微，反向走 index_add）
            k_cols = k_full[:, :, idx, :].contiguous()  # (b, h, K, d)
            v_cols = v_full[:, :, idx, :].contiguous()
            if c_kv is not None:
                # indexer 的候选 latent 与位置（m=1 时逐位等价旧实现）
                cols_c = c_kv[:, idx[:n_static]]
                cols_pos = idx[:n_static]
        scale = q_c.shape[-1] ** -0.5
        scores = torch.matmul(q_c, k_cols.transpose(-1, -2)) * scale
        mask = chunk_causal_mask(
            end - start, n_static, end - start, q_c.device, scores.dtype)
        sel: IndexerSelect | None = None
        kl: torch.Tensor | None = None
        if (self.sparse_indexer and n_static > 0
                and cols_c is not None and h_c is not None):
            if self.csa_mode == "reuse" and shared is not None:
                # Reuse：复用最新的 Top-K，不算 indexer（V4.1：Reuse 只借 KV + 索引）
                sel = shared
            else:
                # Full：全候选造池自选；Reindex：借前层池、自选（V4.1 三模式核心差异）
                sel = self._indexer_select(
                    h_c, cols_c, cols_pos, end - start,
                    shared if self.csa_mode == "reindex" else None)
                if (self.training and sel.logits is not None
                        and self.csa_mode in ("full", "reindex")):
                    # 蒸馏目标：主注意力在（全可见的）静态候选上的头平均分布；detach 目标
                    with torch.no_grad():
                        target = F.softmax(
                            scores[:, :, :, :n_static].float(), dim=-1).mean(1)
                    # 每查询平均（不能用 batchmean：它只除 batch 维 → 被 n_q 放大）
                    kl = F.kl_div(F.log_softmax(sel.logits, dim=-1), target,
                                  reduction="none").sum(-1).mean()
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
        return torch.matmul(probs, v_cols), sel, kl

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
        csa_sel: "CsaChannel | None" = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """稀疏前向：短序列/解码缓存命中时退化为 dense 等价，超长才分块聚集.

        reuse: 跨层复用（CSA2）最近生产者的 main KV latent 缓存；None=正常自算。
        csa_sel: CSA2 层间选择通道（仅 `sparse_indexer` 开启时非 None）——生产者
            （full/reindex）写、消费者（reindex/reuse）读；短序列走 dense 时不参与。
        """
        b, t, _ = h.shape
        if past is not None:
            return self._decode(h, past, csa_sel)
        self._last_indexer_aux = None  # 每次前向重算（赋值语义，梯度检查点重算幂等）
        self._last_indexer_kl = None
        if reuse is not None:
            # 跨层复用：Q 自算、K/V 借共享 latent（短序列走 dense，超长走分块聚集）
            # 压缩开启时恒走分块路径：dense 用未压缩 latent，与压缩表示不一致。
            if self.compress_ratio == 1 and t <= self.sparse_threshold:
                return super().forward(h, None, return_state, reserve, reuse)
            q_full = self._q_only(h)
            k_full, v_full = self._kv_from_latent(reuse[0], reuse[1], total=t)
            out = self._run_blocks(
                q_full, k_full, v_full, h, t, c_kv=reuse[0], csa_sel=csa_sel,
                kr_raw=reuse[1])
            return self._out_proj(out), (reuse if return_state else None)
        if self.compress_ratio == 1 and t <= self.sparse_threshold:
            # 短序列：与 MLAModule 逐位一致（reserve 透传装箱）
            return super().forward(h, None, return_state, reserve)
        # 长序列分块预填充（压缩开启时短序列也走这里）
        q_full, k_full, v_full, c_kv, k_rope_raw = self._full_qkv_for_gather(h)
        out = self._run_blocks(
            q_full, k_full, v_full, h, t, c_kv=c_kv, csa_sel=csa_sel,
            kr_raw=k_rope_raw)
        if not return_state:
            return self._out_proj(out), None
        # 缓存仍存 latent（与 MLA 一致，200K 上下文每 token 仅 ~320B/层），
        # 静态装箱（reserve 由 _prefill 按 prompt+max_new 精确给出）
        return self._out_proj(out), self._pack_cache(c_kv, k_rope_raw, max(reserve, t))

    def _run_blocks(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        h: torch.Tensor,
        t: int,
        c_kv: torch.Tensor,
        csa_sel: "CsaChannel | None" = None,
        kr_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """分块聚集注意力并返回 `(b, t, n_heads*head_dim)`（头维已并回特征维）.

        生产者（full/reindex）逐块写通道；消费者（reindex/reuse）先按块序从通道
        读前层的选择（生产者先于消费者执行，故读到的一定是同一次 forward 里
        最近生产者写下的值）。
        """
        outs = []
        auxs: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        for i, s in enumerate(range(0, t, self.sparse_chunk)):
            e = min(s + self.sparse_chunk, t)
            # 消费者按 `csa_src` 读最近生产者的同块选择（生产者先执行故已就绪）
            shared = (csa_sel.get(self.csa_src, i) if csa_sel is not None else None)
            o, sel, kl = self._sparse_scores(
                q_full[:, :, s:e, :].contiguous(), k_full, v_full, s, e, t,
                c_kv=c_kv, h_c=h[:, s:e], shared=shared, kr_raw=kr_raw)
            if (sel is not None and csa_sel is not None
                    and self.csa_mode in ("full", "reindex")
                    and self.csa_index is not None):
                csa_sel.put(self.csa_index, i, sel)
            if kl is not None:
                kls.append(kl)
                auxs.append(kl * self.indexer_loss_coef)
            outs.append(o)
        # 蒸馏损失（已按系数缩放，进各层 aux 汇总统一加）；重算幂等（赋值语义）。
        # 层内多块取**均值**（否则随块数线性放大），层间由模型侧累加。
        self._last_indexer_aux = (torch.stack(auxs).mean() if auxs else None)
        self._last_indexer_kl = (torch.stack(kls).mean() if kls else None)
        return torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(
            q_full.shape[0], t, -1)

    @torch._dynamo.disable
    def _decode(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        csa_sel: "CsaChannel | None" = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """增量解码：latent 缓存展开后按 pattern 聚集（短上下文自动退化为全量）.

        静态缓存版：按位写进预分配缓冲（不再 `cat`），pattern 按实际长度算
        （gather 下标全落在有效区，无需额外掩码，与 cat 版逐位一致）。
        """
        b, t, _ = h.shape
        buf_c, buf_kr, pos = past
        self._last_indexer_aux = None  # 解码不算蒸馏损失（推理路径）
        self._last_indexer_kl = None
        if not self.csa_reuse:
            # 生产者（Full 层）：按位写自己的 latent；Reindex/Reuse 跳过
            # （共享缓冲已由生产者写好，两者都借 main KV）
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
        out, sel, _kl = self._sparse_scores(
            q_full, k_full, v_full, total - t, total, total, c_kv=buf_c, h_c=h,
            shared=(csa_sel.get(self.csa_src, 0) if csa_sel is not None else None),
            kr_raw=buf_kr)
        if (sel is not None and csa_sel is not None
                and self.csa_mode in ("full", "reindex")
                and self.csa_index is not None):
            # 解码单块：生产者把本步选择写进自己的槽（后续层同一步内读）
            csa_sel.put(self.csa_index, 0, sel)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self._out_proj(out), (buf_c, buf_kr, pos_next)


def apply_rope_wrap(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """apply_rope 的局部别名（避免循环导入，保持调用点清晰）."""
    from src.llm.local.rope import apply_rope

    return apply_rope(x, cos, sin)
