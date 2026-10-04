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

import torch
import torch.nn.functional as F

from src.llm.local.mla import MLAModule


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
    ):
        super().__init__(
            d_model, n_heads, q_lora_rank, kv_lora_rank, qk_rope_dim,
            max_seq_len, rope_theta, yarn_scale, dropout,
        )
        # 纯超参，不注册为 Parameter/Buffer（state_dict 与 MLA 一致，checkpoint 通用）
        self.sparse_threshold = sparse_threshold
        self.sparse_window = sparse_window
        self.sparse_sink = sparse_sink
        self.sparse_stride = sparse_stride
        self.sparse_chunk = sparse_chunk

    def _sparse_scores(
        self,
        q_c: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        start: int,
        end: int,
        total: int,
    ) -> torch.Tensor:
        """对查询 [start, end) 做聚集 + 掩码 + 注意力（softmax 提 fp32 保精度）.

        q_c 为已切好的查询块 (b, h, C=end-start, d)；k/v 为全历史。
        解码时 q_c 只有当步 1 个位置（下标与全历史不对齐，由 start/end 表达绝对位置）。
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
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """稀疏前向：短序列/解码缓存命中时退化为 dense 等价，超长才分块聚集."""
        b, t, _ = h.shape
        if past is not None:
            return self._decode(h, past)
        if t <= self.sparse_threshold:
            # 短序列：与 MLAModule 逐位一致
            return super().forward(h, None, return_state)
        # 长序列分块预填充
        q_full, k_full, v_full, c_kv, k_rope_raw = self._full_qkv_for_gather(h)
        outs = []
        for s in range(0, t, self.sparse_chunk):
            e = min(s + self.sparse_chunk, t)
            outs.append(self._sparse_scores(
                q_full[:, :, s:e, :].contiguous(), k_full, v_full, s, e, t))
        out = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, t, -1)
        if not return_state:
            return self.w_o(out), None
        # 缓存仍存 latent（与 MLA 一致，200K 上下文每 token 仅 ~320B/层）
        return self.w_o(out), (c_kv, k_rope_raw)

    def _decode(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """增量解码：latent 缓存展开后按 pattern 聚集（短上下文自动退化为全量）."""
        b, t, _ = h.shape
        c_prev, kr_prev = past
        c_new = torch.cat([c_prev, self.w_dkv(h)], dim=1)
        k_rope_new = torch.cat([kr_prev, self.w_kr(h)], dim=1)
        total = c_new.shape[1]
        # 当步 Q（含绝对位置 RoPE）
        q_c = self.w_dq(h)
        q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        off = total - t
        q_rope = apply_rope_wrap(q_rope, self.rope_cos[off:off + t], self.rope_sin[off:off + t])
        # 全历史展开（latent 省的是缓存不是计算，单步展开可接受）
        k = self.k_norm(self.w_uk(c_new).view(b, total, self.n_heads, self.head_dim))
        v = self.w_uv(c_new).view(b, total, self.n_heads, self.head_dim)
        k_rope_hist = apply_rope_wrap(
            k_rope_new,
            self.rope_cos[:total],
            self.rope_sin[:total],
        ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_hist], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        # 查询即最后位置：select_keys(total, total-1, total)，q 只有当步
        out = self._sparse_scores(q_full, k_full, v_full, total - 1, total, total)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.w_o(out), (c_new, k_rope_new)


def apply_rope_wrap(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """apply_rope 的局部别名（避免循环导入，保持调用点清晰）."""
    from src.llm.local.rope import apply_rope

    return apply_rope(x, cos, sin)
