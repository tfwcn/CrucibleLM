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


class SwaBranch(nn.Module):
    """层内滑动窗口注意力（V4.1：**每层自带**、与全局主 KV 完全独立）.

    论文口径：每层同时有全局注意力与 SWA；SWA KV 由**当前层 hidden** 产生
    （不跨层复用、不压缩、不进 kv_lora_rank 瓶颈），窗口定长 → 存储与序列长度无关。
    CSA2 的"selected entries together with the layer-local SWA KV"就是
    **同一次 softmax** 里既放全局稀疏列、也放 SWA 列。

    结构与 MLA 同构（content + 解耦 rope，二者拼起来才和全局 K 同维）：
    `w_k` 出 content（每头 head_dim）、`w_kr` 出共享 rope 键（qk_rope_dim）、`w_v` 出值。
    缓存为三个**定长环**（k content / v / rope 键），按 `pos % n_win` 覆盖写。
    """

    def __init__(self, d_model: int, n_heads: int, qk_rope_dim: int,
                 window: int, dropout: float = 0.0):
        super().__init__()
        from src.llm.local.mla import RMSNorm

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_rope_dim = int(qk_rope_dim)
        self.window = int(window)
        assert self.window > 0, "SWA 窗口须为正"
        # SWA 的 K/V 走独立投影（论文里全局与本地是两条分支，不共用上投影）
        self.w_k = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_kr = nn.Linear(d_model, self.qk_rope_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def project(self, h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hidden → 本层 SWA 的 (k_content, k_rope_raw, v)，**未加 rope**."""
        b, t, _ = h.shape
        k = self.k_norm(self.w_k(h).view(b, t, self.n_heads, self.head_dim))
        kr = self.w_kr(h)
        v = self.w_v(h).view(b, t, self.n_heads, self.head_dim)
        return k, kr, v

    def band_mask(
        self, q_pos: torch.Tensor, kv_pos: torch.Tensor,
    ) -> torch.Tensor:
        """带状因果掩码（加性）：位置 q 只看 `[q-n_win+1, q]` 内的 KV.

        q_pos: (n_q,) 查询绝对位置；kv_pos: (n_kv,) KV 绝对位置。
        返回 (n_q, n_kv)，0=放行、-inf=屏蔽。位置 < 0 的槽位（环里没写过）
        由调用方裁掉，不进这里。
        """
        vis = ((kv_pos[None, :] <= q_pos[:, None])
               & (kv_pos[None, :] > q_pos[:, None] - self.window))
        return torch.where(
            vis, torch.zeros((), device=q_pos.device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=q_pos.device, dtype=torch.float32))


class SparseMLAModule(MLAModule):
    """稀疏 MLA：继承 MLA 全部投影与缓存，只替换注意力核心.

    `swa_window=0`（默认）时无新增参数，state_dict 与 MLAModule 完全兼容；
    `swa_window>0` 时新增层内 SWA 分支（`swa.*`，改键、opt-in）。
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
        swa_window: int = 0,
        ced: bool = False,
        kv_fmt: str = "",
    ):
        super().__init__(
            d_model, n_heads, q_lora_rank, kv_lora_rank, qk_rope_dim,
            max_seq_len, rope_theta, yarn_scale, dropout,
            o_groups=o_groups, o_lora_rank=o_lora_rank, ced=ced,
            kv_fmt=kv_fmt,
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
        # 层内 SWA（V4.1：每层自带、窗口定长）：0=关（无新增参数/键，逐位兼容旧 ckpt）
        self.swa_window = int(swa_window)
        # 层内 SWA 环的量化格式（V4.1）：主 KV 开量化时环用 **FP8**（论文：SWA 对
        # 量化更敏感，保留 FP8；主 KV 才用 FP4）
        self.swa_fmt = "fp8" if self.kv_fmt else ""
        self.swa: SwaBranch | None = (
            SwaBranch(d_model, n_heads, qk_rope_dim, self.swa_window, dropout)
            if self.swa_window > 0 else None
        )
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

    def _far_block_ids(self, total: int, start: int,
                       window: int | None = None) -> torch.Tensor:
        """远端（压缩）块 id：只取"完整且在查询块之前"的块（因果安全）.

        - 完整：块尾 `(b+1)*m-1 < total`（缓存/当前段里真有这些行）；
        - 因果：块尾 `< start`（在查询块之前，绝不看未来 token）；
        - 只在 `start - sparse_window` 之前：更近的位置由**未压缩**的窗口/本块列覆盖。
        建议 `m ≤ sparse_window`（否则窗口与压缩列之间会漏掉不足一个块的近端 token）。
        """
        m = self.compress_ratio
        win = self.sparse_window if window is None else window
        n_valid = min(total // m, start // m)
        n_far = min(n_valid, max(0, (start - win + m - 1) // m))
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
        swa: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
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
        `swa=(k_swa, kr_swa, v_swa, kv_pos)` 给定时，把**层内 SWA 列**拼进同一次 softmax
        （V4.1：selected entries together with the layer-local SWA KV），
        SWA 列对块内所有查询可见、但受带状窗口掩码限制（`SwaBranch.band_mask`）。
        返回 (注意力输出, 本层选择, indexer 蒸馏 KL)。
        训练且为生产者（full/reindex）时额外算**蒸馏损失**（DSA 做法：让学习式打分
        去拟合主注意力在候选上的分布）——否则 indexer 只进掩码、拿不到任何梯度。
        """
        b, h, _, d = k_full.shape
        n_q = end - start
        # SWA 分支开启时：本地（窗口 + 当前块）**全部**由 SWA 列承担，
        # 全局主 KV 只出"选中条目"（V4.1：selected entries + layer-local SWA KV），
        # 避免近端 token 在两条分支里被重复计权。
        has_swa = swa is not None and self.swa is not None
        win = 0 if has_swa else self.sparse_window
        n_chunk = 0 if has_swa else n_q
        # SWA 开启时全局 pattern 必须按**块对齐起点**取：stride 集合依赖起点，
        # 若用查询位置（解码时 start=total）会与预填充的块起点不一致，
        # 近端 token 就会同时出现在"全局列"和"SWA 列"里被重复计权（解码≠全前向）。
        blk_start = (start // max(1, self.sparse_chunk)) * max(1, self.sparse_chunk) \
            if has_swa else start
        cols_c: torch.Tensor | None = None
        cols_pos: torch.Tensor | None = None
        if self.compress_ratio > 1 and c_kv is not None and kr_raw is not None:
            # 压缩路径（V4.1 CSA2(ratio, mode)）：远端 main KV 压成 m 块列，
            # 本地窗口 + 当前块保持未压缩（SWA KV 不压缩，逐层本地）。
            blk_ids = self._far_block_ids(total, blk_start, win).to(k_full.device)
            lo = max(0, start - win)
            local = (torch.arange(lo, end, device=k_full.device) if n_chunk
                     else torch.zeros(0, dtype=torch.long, device=k_full.device))
            c_blk, k_blk, v_blk, blk_end = self._compressed_columns(
                c_kv, kr_raw, blk_ids)
            k_cols = torch.cat([k_blk, k_full[:, :, local, :]], dim=2).contiguous()
            v_cols = torch.cat([v_blk, v_full[:, :, local, :]], dim=2).contiguous()
            # 静态列 = 压缩块列 + 窗口列（窗口全 < start，对块内所有查询可见）
            n_near = local.numel() - n_chunk
            cols_c = torch.cat([c_blk, c_kv[:, local[:n_near]]], dim=1)
            cols_pos = torch.cat([blk_end, local[:n_near]])
            n_static = blk_ids.numel() + n_near
        else:
            idx, n_static = select_keys(
                total, blk_start, blk_start + n_chunk,
                self.sparse_sink, win, self.sparse_stride)
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
            n_q, n_static, n_chunk, q_c.device, scores.dtype)
        # 层内 SWA 列（同一次 softmax）：拼在最后，掩码换成带状窗口（对块内所有查询）
        n_swa = 0
        if has_swa:
            k_swa, kr_swa, v_swa, swa_pos = swa
            if k_swa.shape[1] > 0:
                swa_pos = swa_pos.to(q_c.device)
                # SWA 的 K 也按**绝对位置**旋转（局部顺序必须可辨）；
                # 环里存未加 rope 的原料，读时现算（rope 只占 qk_rope_dim）
                _rc = self.rope_cos.shape[0]
                _ri = swa_pos.clamp(max=_rc - 1)
                kr_swa = apply_rope_wrap(
                    kr_swa, self.rope_cos[_ri], self.rope_sin[_ri])
                k_swa = torch.cat(
                    [k_swa, kr_swa.unsqueeze(2).expand(
                        -1, -1, self.n_heads, -1)], dim=-1)
                k_swa = k_swa.transpose(1, 2).to(scores.dtype)
                v_swa = v_swa.transpose(1, 2).to(scores.dtype)
                n_swa = k_swa.shape[2]
                s_swa = torch.matmul(q_c, k_swa.transpose(-1, -2)) * scale
                scores = torch.cat([scores, s_swa], dim=-1)
                q_pos = torch.arange(start, end, device=q_c.device)
                band = self.swa.band_mask(q_pos, swa_pos)
                mask = torch.cat([mask, band.to(mask.dtype)], dim=-1)
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
        if n_swa > 0:
            v_cols = torch.cat([v_cols, v_swa], dim=2)
        return torch.matmul(probs, v_cols), sel, kl

    def _full_qkv_for_gather(
        self, h: torch.Tensor, ced_src: torch.Tensor | None = None,
        q_off: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 hidden 算全序列（或查询尾段）q/k/v（含 rope 拼接）+ latent 缓存原料.

        返回 (q_full, k_full, v_full, c_kv, k_rope_raw)，一次投影处处复用。
        `ced_src`/`q_off`：CED 解码器层的 KV 来自编码器输出，且跳算时查询只覆盖
        尾部（KV 仍覆盖全序列）——见 `forward`。
        """
        q, k, v, q_rope, k_rope, c_kv, k_rope_raw = self._qkv(
            h, ced_src=ced_src, q_off=q_off)
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
        write_cache: bool = True,
        swa_floor: int = 0,
        ced_src: torch.Tensor | None = None,
        q_off: int = 0,
    ) -> tuple[torch.Tensor, tuple | None]:
        """稀疏前向：短序列/解码缓存命中时退化为 dense 等价，超长才分块聚集.

        reuse: 跨层复用（CSA2）最近生产者的 main KV latent 缓存；None=正常自算。
        csa_sel: CSA2 层间选择通道（仅 `sparse_indexer` 开启时非 None）——生产者
            （full/reindex）写、消费者（reindex/reuse）读；短序列走 dense 时不参与。
        write_cache=False：**只读**解码（SWA Bounded Replay 用）——全局主 KV 的
            行已由权威前向写好，重放只重建 SWA 环，绝不覆盖权威值（V4.1 §3.2.2
            "reusing the cached global KV without recomputation or overwriting"）。
        swa_floor：SWA 列的下界（重放起点 s），见 `_swa_read`。
        ced_src：CED 解码器层的编码器输出 `H_enc`（全局 KV/rope 键由它投影）；
            `q_off` 非 0 时 `h` 只覆盖查询尾段、`ced_src` 仍覆盖全序列（跳算预填充，
            见 `TinyLLM._prefill`）。CED 恒走分块路径（dense 兜底没有这套语义）。
        """
        b, t, _ = h.shape
        if past is not None:
            return self._decode(h, past, csa_sel, write_cache=write_cache,
                                swa_floor=swa_floor, ced_src=ced_src)
        self._last_indexer_aux = None  # 每次前向重算（赋值语义，梯度检查点重算幂等）
        self._last_indexer_kl = None
        if reuse is not None:
            # 跨层复用：Q 自算、K/V 借共享 latent（短序列走 dense，超长走分块聚集）
            # 压缩/SWA 开启时恒走分块路径：dense 用未压缩 latent、且不走层内
            # SWA（近端走全局窗口），与解码路径分叉。
            if (self.compress_ratio == 1 and self.swa is None
                    and not self.ced and t <= self.sparse_threshold):
                return super().forward(h, None, return_state, reserve, reuse)
            q_full = self._q_only(h)
            k_full, v_full = self._kv_from_latent(reuse[0], reuse[1], total=t)
            # indexer 的候选 latent 需要 float（量化缓存先反量化）
            from src.llm.local.kv_quant import QuantBuffer

            c_cols = (reuse[0].dequant() if isinstance(reuse[0], QuantBuffer)
                      else reuse[0])
            kr_cols = (reuse[1].dequant() if isinstance(reuse[1], QuantBuffer)
                       else reuse[1])
            swa_kv = self._swa_project(h) if self.swa is not None else None
            out = self._run_blocks(
                q_full, k_full, v_full, h, t, c_kv=c_cols, csa_sel=csa_sel,
                kr_raw=kr_cols, swa_kv=swa_kv)
            if not return_state:
                return self._out_proj(out), None
            # 只借生产者的全局主 KV（buf/pos），**SWA 环必须各建各的**：层内
            # SWA 是本层自己的分支，若整份 past 元组照抄，解码时两层会往同一个
            # 环里写、彼此覆盖（pitfall 38）。
            packed = (reuse[0], reuse[1], reuse[2])
            if swa_kv is not None:
                packed = packed + self._pack_swa(
                    *swa_kv, t, self._swa_capacity(1))
            return self._out_proj(out), packed
        if (self.compress_ratio == 1 and self.swa is None and not self.ced
                and t <= self.sparse_threshold):
            # 短序列：与 MLAModule 逐位一致（reserve 透传装箱）
            return super().forward(h, None, return_state, reserve)
        # 长序列分块预填充（压缩/SWA 开启时短序列也走这里：dense 兜底路径不带
        # 层内 SWA，若短 prompt 走 dense、解码走 SWA，两者必然分叉）
        q_full, k_full, v_full, c_kv, k_rope_raw = self._full_qkv_for_gather(
            h, ced_src=ced_src, q_off=q_off)
        # 全序列长（KV/缓存覆盖到 `total`）；`h`/`q_full` 只覆盖查询段
        # `[q_off, total)`——CED 跳算时它们就是尾段
        total = q_off + t
        # SWA 投影：查询段里的**每个**位置都要有自己的 SWA 列（否则前段输出与
        # 全前向分叉）；环另由 `_pack_swa` 按绝对位置装（定长）。
        # 开 KV 量化时 SWA 环按 FP8 存 → 这里也要 fake-quant，否则"预填充用的是
        # 未量化的 SWA 列、解码用的是环里的 FP8 值"，两条路径必然分叉。
        swa_kv = self._swa_project(h) if self.swa is not None else None
        out = self._run_blocks(
            q_full, k_full, v_full, h, total, c_kv=c_kv, csa_sel=csa_sel,
            kr_raw=k_rope_raw, swa_kv=swa_kv, q_off=q_off)
        if not return_state:
            return self._out_proj(out), None
        # 缓存仍存 latent（与 MLA 一致，200K 上下文每 token 仅 ~320B/层），
        # 静态装箱（reserve 由 _prefill 按 prompt+max_new 精确给出）；
        # CED 解码器层的 latent 来自编码器输出 → 长度就是全序列，`pos` 为 `total`
        packed = self._pack_cache(c_kv, k_rope_raw, max(reserve, total))
        if swa_kv is not None:
            # 环只服务**之后**的单 token 解码 → 容量 `window`（多 token 解码时
            # `_grow_swa_rings` 再按需扩），绝不按 prompt 长度建环
            packed = packed + self._pack_swa(
                *swa_kv, total, self._swa_capacity(1))
        return self._out_proj(out), packed

    def empty_swa_rings(
        self, t: int = 1, batch: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """新建三个空环（容量 `window + max(0, t-1)`，行号 = 位置 % 容量）.

        SWA Bounded Replay 的起点：环从空开始，重放过程中按位置逐行写入，
        环里只会有重放段 `[s, total)` 的内容——这正是论文"把 SWA 截断到
        重放段"的实现（更早的 SWA 键不复存在，也就不可能被读到）。
        """
        assert self.swa is not None, "本层未开 SWA（swa_window=0）"
        cap = self._swa_capacity(t)
        k_shape = (batch, cap, self.n_heads, self.swa.head_dim)
        kr_shape = (batch, cap, self.swa.qk_rope_dim)
        if self.swa_fmt:
            from src.llm.local.kv_quant import QuantBuffer

            return (QuantBuffer.zeros(k_shape, self.swa_fmt, device=device),
                    QuantBuffer.zeros(kr_shape, self.swa_fmt, device=device),
                    QuantBuffer.zeros(k_shape, self.swa_fmt, device=device))
        k = torch.zeros(*k_shape, device=device, dtype=dtype)
        kr = torch.zeros(*kr_shape, device=device, dtype=dtype)
        return k, kr, torch.zeros_like(k)

    def _swa_project(self, h: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """层内 SWA 的 K/V 投影 + QAT（`swa_fmt` 非空时按 FP8 fake-quant）.

        SWA 环按 FP8 存（V4.1：SWA 对量化更敏感，保留 FP8），所以**预填充的 SWA 列**
        与**环里的值**必须是同一套量化数值，否则增量解码与全前向分叉（实测 0.3 量级）。
        """
        k, kr, v = self.swa.project(h)
        if self.swa_fmt:
            from src.llm.local.kv_quant import fake_quant

            k = fake_quant(k, self.swa_fmt)
            kr = fake_quant(kr, self.swa_fmt)
            v = fake_quant(v, self.swa_fmt)
        return k, kr, v

    def _swa_capacity(self, t: int = 1) -> int:
        """SWA 环容量：`window + max(0, t-1)`.

        单 token 解码只需 `window` 行；多 token 增量（投机验证一次喂 t 个）需要
        查询并集 `[start-window+1, end)` 共 `window + t - 1` 行，否则最前面几个
        查询会读到被覆盖的槽位（与"环 = 窗口"的直觉冲突，见 pitfalls 34）。
        """
        return self.swa.window + max(0, int(t) - 1)

    def _pack_swa(
        self, k: torch.Tensor, kr: torch.Tensor, v: torch.Tensor,
        pos_end: int, cap: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把本层 SWA 的 k/kr/v 尾巴装进**定长环**（行号 = 绝对位置 % 容量）.

        SWA 存储与序列长度无关（V4.1：窗口定长 → 与 L 无关），这是它相对
        全局主 KV 的根本优势；环只保留最近 `window + max(0, t-1)` 个位置
        （t = 之后可能一次解码的 token 数：预填充给 1、多 token 解码给 t）。
        `pos_end` 是最后写入 token 的绝对位置 +1，尾巴按绝对位置右对齐写。

        注意 `cap` 与 `pos_end` 必须分开传：预填充时 `pos_end=L` 而 `cap=window`
        ——若按 `window+L-1` 建环，SWA 缓存就退化成 O(L)，与"存储有界"矛盾
        （pitfall 34）。
        """
        b = k.shape[0]
        # k/kr/v 可能只覆盖尾部（Bounded Replay）：按**绝对位置**右对齐写入
        t_proj = k.shape[1]
        n = min(t_proj, cap)
        rows = torch.arange(pos_end - n, pos_end, device=k.device) % cap
        k_tail = k[:, t_proj - n:t_proj]
        kr_tail = kr[:, t_proj - n:t_proj]
        v_tail = v[:, t_proj - n:t_proj]
        if self.swa_fmt:
            from src.llm.local.kv_quant import QuantBuffer

            k_ring = QuantBuffer.zeros(
                (b, cap, self.n_heads, self.swa.head_dim), self.swa_fmt,
                device=k.device).write_rows_(rows, k_tail)
            kr_ring = QuantBuffer.zeros(
                (b, cap, self.swa.qk_rope_dim), self.swa_fmt,
                device=k.device).write_rows_(rows, kr_tail)
            v_ring = QuantBuffer.zeros(
                (b, cap, self.n_heads, self.swa.head_dim), self.swa_fmt,
                device=k.device).write_rows_(rows, v_tail)
            return k_ring, kr_ring, v_ring
        k_ring = torch.zeros(b, cap, self.n_heads, self.swa.head_dim,
                             device=k.device, dtype=k.dtype)
        kr_ring = torch.zeros(b, cap, self.swa.qk_rope_dim,
                              device=k.device, dtype=k.dtype)
        v_ring = torch.zeros_like(k_ring)
        k_ring[:, rows] = k_tail
        kr_ring[:, rows] = kr_tail
        v_ring[:, rows] = v_tail
        return k_ring, kr_ring, v_ring

    def _grow_swa_rings(
        self, rings: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pos: int, need: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """扩容环（多 token 解码超过 slack 时）：按位置重映射有效槽位.

        旧环里的有效位置是 `[pos - cap_old, pos)`（行号 = 位置 % cap_old），
        新环容量取 2x 与 `need` 的较大者；重映射后旧槽位仍可按 `位置 % cap_new` 命中。
        """
        cap_old = rings[0].shape[1]
        if cap_old >= need:
            return rings
        cap_new = max(cap_old * 2, need)
        b = rings[0].shape[0]
        n_old = min(cap_old, pos)
        old_pos = torch.arange(pos - n_old, pos, device=rings[0].device)
        src = old_pos % cap_old
        dst = old_pos % cap_new
        from src.llm.local.kv_quant import QuantBuffer

        out = []
        for ring, width in zip(rings, (None, self.swa.qk_rope_dim, None)):
            if isinstance(ring, QuantBuffer):
                # 量化环：先反量化取有效行，再按新模写入新环
                new = QuantBuffer.zeros((b, cap_new, *ring.shape[2:]),
                                        ring.fmt, device=ring.device)
                new.write_rows_(dst, ring.dequant().index_select(1, src))
                out.append(new)
                continue
            if ring.dim() == 4:  # (b, cap, n_heads, head_dim)
                new = torch.zeros(b, cap_new, ring.shape[2], ring.shape[3],
                                  device=ring.device, dtype=ring.dtype)
            else:
                new = torch.zeros(b, cap_new, width,
                                  device=ring.device, dtype=ring.dtype)
            new[:, dst] = ring[:, src]
            out.append(new)
        return tuple(out)

    def _swa_read(
        self, rings: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        start: int, end: int, floor: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从环里取查询 [start, end) 需要的 SWA 列（升序绝对位置）.

        需要的并集 = `[start - n_win + 1, end)`；行号按 `pos % cap` 映射，
        位置 < 0 的部分裁掉（序列开头本无历史）。
        `floor` 给 SWA Bounded Replay 用：论文口径"重放起点 s 之后，查询 i 只看
        `[max(s, i-W+1), i]`"——重放段之前的位置**根本没进环**，必须靠 floor
        显式裁掉，否则会读到环里未初始化的零行（软注意力照付权重）。
        """
        from src.llm.local.kv_quant import QuantBuffer

        n_win = self.swa.window
        k_ring, kr_ring, v_ring = rings
        cap = k_ring.shape[1]
        lo = max(0, int(floor), start - n_win + 1)
        pos = torch.arange(lo, end, device=k_ring.device)
        rows = pos % cap
        if isinstance(k_ring, QuantBuffer):
            # 环按 FP8 存（V4.1）：读时反量化整环再取列（环很小，代价可忽略）
            k_ring = k_ring.dequant()
            kr_ring = kr_ring.dequant()
            v_ring = v_ring.dequant()
        return k_ring[:, rows], kr_ring[:, rows], v_ring[:, rows], pos

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
        swa_kv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        q_off: int = 0,
    ) -> torch.Tensor:
        """分块聚集注意力并返回 `(b, t, n_heads*head_dim)`（头维已并回特征维）.

        `swa_kv=(k_swa, kr_swa, v_swa)`（本层 SWA 全序列投影）给定时，
        每块再切出 `[start-window+1, end)` 的 SWA 列一起参与同一次 softmax。

        生产者（full/reindex）逐块写通道；消费者（reindex/reuse）先按块序从通道
        读前层的选择（生产者先于消费者执行，故读到的一定是同一次 forward 里
        最近生产者写下的值）。
        `q_off > 0` 时 `q_full`/`h` 只覆盖查询尾段 `[q_off, t)`（CED 跳算预填充），
        KV 仍覆盖全序列——块循环从 `q_off` 起，局部下标统一减 `q_off`。
        """
        outs = []
        auxs: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        for i, s in enumerate(range(q_off, t, self.sparse_chunk)):
            e = min(s + self.sparse_chunk, t)
            qs, qe = s - q_off, e - q_off
            # 消费者按 `csa_src` 读最近生产者的同块选择（生产者先执行故已就绪）
            shared = (csa_sel.get(self.csa_src, i) if csa_sel is not None else None)
            swa_cols = None
            if swa_kv is not None:
                off = t - swa_kv[0].shape[1]      # 投影起点（Bounded Replay）
                lo = max(off, s - self.swa.window + 1)
                if e > lo:
                    swa_cols = (
                        swa_kv[0][:, lo - off:e - off],
                        swa_kv[1][:, lo - off:e - off],
                        swa_kv[2][:, lo - off:e - off],
                        torch.arange(lo, e, device=h.device))
            o, sel, kl = self._sparse_scores(
                q_full[:, :, qs:qe, :].contiguous(), k_full, v_full, s, e, t,
                c_kv=c_kv, h_c=h[:, qs:qe], shared=shared, kr_raw=kr_raw,
                swa=swa_cols)
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
            q_full.shape[0], t - q_off, -1)

    @torch._dynamo.disable
    def _decode(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        csa_sel: "CsaChannel | None" = None,
        write_cache: bool = True,
        swa_floor: int = 0,
        ced_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """增量解码：latent 缓存展开后按 pattern 聚集（短上下文自动退化为全量）.

        静态缓存版：按位写进预分配缓冲（不再 `cat`），pattern 按实际长度算
        （gather 下标全落在有效区，无需额外掩码，与 cat 版逐位一致）。

        `write_cache=False` 为 SWA Bounded Replay 的只读模式：全局主 KV 的
        `[pos, pos+t)` 行已由权威前向写好，重放只重建 SWA 环，不重写全局
        （V4.1 §3.2.2）。此时 `pos` 是**回退**后的重放起点。
        """
        b, t, _ = h.shape
        buf_c, buf_kr, pos = past[0], past[1], past[2]
        rings = tuple(past[3:6]) if len(past) > 3 else None
        self._last_indexer_aux = None  # 解码不算蒸馏损失（推理路径）
        self._last_indexer_kl = None
        if not self.csa_reuse and write_cache:
            # 生产者（Full 层）：按位写自己的 latent；Reindex/Reuse 跳过
            # （共享缓冲已由生产者写好，两者都借 main KV）
            from src.llm.local.kv_quant import QuantBuffer

            if isinstance(buf_c, QuantBuffer):
                # 量化缓存：按格式就地量化写（QAT 数值与训练一致）
                buf_c.write_(int(pos.item()), self._kv_latent(h, ced_src))
                buf_kr.write_(int(pos.item()), self._rope_key(h, ced_src))
            else:
                w_idx = pos.reshape(1) + torch.arange(t, device=h.device)
                # 对齐缓冲 dtype：prefill 可能建在 autocast 下（bf16 缓冲），解码若
                # 不带 autocast 会得到 fp32；同 dtype 时 `.to` 是 no-op
                buf_c.index_copy_(1, w_idx,
                                  self._kv_latent(h, ced_src).to(buf_c.dtype))
                buf_kr.index_copy_(1, w_idx,
                                   self._rope_key(h, ced_src).to(buf_kr.dtype))
        pos_next = pos + t
        # 实际长度（pattern 按实际算；一次同步，eager 可忽略；
        # compile 下此处 graph-break 但不重编）
        total = int(pos_next.item())
        # 当步 Q（含绝对位置 RoPE）
        q_c = self.w_dq(h)
        q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        off = total - t
        # 位置下标钳制在 rope 表内：解码到接近 max_seq_len 时（prompt 已占满表）
        # 直接用切片会得到空表 → apply_rope 里 view 崩在无关的地方（MLA 侧同款处理）
        _rlen = self.rope_cos.shape[0]
        _qidx = (off + torch.arange(t, device=h.device)).clamp(max=_rlen - 1)
        q_rope = apply_rope_wrap(
            q_rope, self.rope_cos[_qidx], self.rope_sin[_qidx])
        # 全缓冲展开（latent 省的是缓存不是计算，单步展开可接受；
        # pattern 只 gather 有效下标，空位不参与；rope 下标钳制防 L 超缓存）。
        # 量化缓存先反量化（V4.1：反量化后再算注意力，不需要 FP4 矩阵乘）
        from src.llm.local.kv_quant import QuantBuffer

        c_use = buf_c.dequant() if isinstance(buf_c, QuantBuffer) else buf_c
        kr_use = buf_kr.dequant() if isinstance(buf_kr, QuantBuffer) else buf_kr
        k = self.k_norm(self.w_uk(c_use).view(b, -1, self.n_heads,
                                             self.head_dim))
        v = self.w_uv(c_use).view(b, -1, self.n_heads, self.head_dim)
        _rlen = self.rope_cos.shape[0]
        _ridx = torch.arange(kr_use.shape[1], device=h.device).clamp(
            max=_rlen - 1)
        k_rope_hist = apply_rope_wrap(
            kr_use,
            self.rope_cos[_ridx],
            self.rope_sin[_ridx],
        ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_hist], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        # 查询即最后 t 个位置：select_keys(total, total-t, total)，t==1 时与旧式一致；
        # t>1 为投机解码并行验证服务（块内因果由 chunk_causal_mask 保证）
        swa_cols = None
        if self.swa is not None:
            k_swa, kr_swa, v_swa = self._swa_project(h)
            if rings is None:
                # 兼容旧缓存（无 SWA 环）：本步现场补一个（后续步即走环）
                rings = self._pack_swa(k_swa, kr_swa, v_swa, total,
                                       self._swa_capacity(t))
            else:
                rings = self._grow_swa_rings(
                    rings, int(pos.item()), self._swa_capacity(t))
                cap = rings[0].shape[1]
                rows = ((pos.reshape(1) + torch.arange(t, device=h.device))
                        % cap)
                from src.llm.local.kv_quant import QuantBuffer

                if isinstance(rings[0], QuantBuffer):
                    # 量化环：原地按行写（环内位置 = 位置 % cap）
                    for ring, new in zip(rings, (k_swa, kr_swa, v_swa)):
                        ring.write_rows_(rows, new)
                    rings = tuple(r.clone() for r in rings)
                else:
                    rings = tuple(
                        ring.index_copy(1, rows, new.to(ring.dtype)).clone()
                        for ring, new in zip(rings, (k_swa, kr_swa, v_swa)))
            swa_cols = self._swa_read(rings, total - t, total, floor=swa_floor)
        out, sel, _kl = self._sparse_scores(
            q_full, k_full, v_full, total - t, total, total, c_kv=c_use,
            h_c=h,
            shared=(csa_sel.get(self.csa_src, 0) if csa_sel is not None else None),
            kr_raw=kr_use, swa=swa_cols)
        if (sel is not None and csa_sel is not None
                and self.csa_mode in ("full", "reindex")
                and self.csa_index is not None):
            # 解码单块：生产者把本步选择写进自己的槽（后续层同一步内读）
            csa_sel.put(self.csa_index, 0, sel)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        new_past = (buf_c, buf_kr, pos_next)
        if self.swa is not None:
            new_past = new_past + tuple(rings)
        return self._out_proj(out), new_past


def apply_rope_wrap(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """apply_rope 的局部别名（避免循环导入，保持调用点清晰）."""
    from src.llm.local.rope import apply_rope

    return apply_rope(x, cos, sin)
