"""MLA 全注意力 — DeepSeek 系低秩 KV 压缩 + 解耦 RoPE + QK-Norm.

标准 MHA 的 KV 缓存为 2 * n_heads * head_dim；
MLA 只缓存 latent 向量 c_kv（kv_lora_rank 维）+ 共享 RoPE 键，
压缩比约为 (2*768)/(128+32) ≈ 9.6x，这是长上下文省显存的关键。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.rope import apply_rope, build_rope_cache


class RMSNorm(nn.Module):
    """RMSNorm（无偏置，DeepSeek/Qwen 通用选择）."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化并缩放（内部 fp32 求方差，保持输入精度，混合精度安全）."""
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (xf * self.weight.float()).to(dtype)


class MLAModule(nn.Module):
    """MLA 注意力层（含训练前向 + 增量解码缓存）."""

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
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_rope_dim = qk_rope_dim
        self.kv_lora_rank = kv_lora_rank
        # Q：下投影 -> 上投影（内容部分）
        self.w_dq = nn.Linear(d_model, q_lora_rank, bias=False)
        self.w_uq = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
        # Q 解耦 RoPE 部分（每头独立旋转）
        self.w_qr = nn.Linear(q_lora_rank, n_heads * qk_rope_dim, bias=False)
        # KV 联合下投影 -> 每头上投影
        self.w_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.w_uk = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        # 共享 RoPE 键（全头共用一份，省缓存）
        self.w_kr = nn.Linear(d_model, qk_rope_dim, bias=False)
        # 输出投影 + QK-Norm（Qwen3 系做法，稳定训练）
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout
        # 预计算 RoPE 缓存（解耦部分维度小，可一次建好）
        cos, sin = build_rope_cache(max_seq_len, qk_rope_dim, rope_theta, yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _qkv(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 hidden 计算各头 q/k/v、旋转 q_rope/k_rope，并附带 latent 原料.

        后两项是未旋转的 c_kv 与共享 rope 键，供解码缓存复用（省两次投影）。
        """
        b, t, _ = h.shape
        # Q 内容流
        q_c = self.w_dq(h)
        q = self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        # Q 旋转流
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)
        # KV 压缩流（训练时展开；解码时只缓存 c_kv）
        c_kv = self.w_dkv(h)
        k = self.w_uk(c_kv).view(b, t, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.w_uv(c_kv).view(b, t, self.n_heads, self.head_dim)
        # 共享旋转键（先旋转再补头维度，避免广播错位）
        k_rope_raw = self.w_kr(h)
        k_rope = apply_rope(k_rope_raw, self.rope_cos, self.rope_sin).unsqueeze(2)
        return q, k, v, q_rope, k_rope, c_kv, k_rope_raw

    def forward(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """训练/Prefill 前向：全序列并行注意力.

        past: 解码缓存 (c_kv_cache, k_rope_cache)，解码时传入可增量拼接。
        return_state=False 时不返回解码缓存（训练用不上，省两次投影）。
        返回 (输出, 新缓存或 None)。
        """
        b, t, _ = h.shape
        if past is not None:
            # 增量解码：拼接 latent + 共享 rope 键（省约 9x 显存），只算当前步的 Q
            c_prev, kr_prev = past
            c_new = torch.cat([c_prev, self.w_dkv(h)], dim=1)
            k_rope_new = torch.cat([kr_prev, self.w_kr(h)], dim=1)
            q_c = self.w_dq(h)
            q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
            q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
            # 单 token 解码：旋转相位按绝对位置 offset 切片
            off = c_prev.shape[1]
            q_rope = apply_rope(
                q_rope,
                self.rope_cos[off:off + t],
                self.rope_sin[off:off + t],
            )
            k = self.k_norm(self.w_uk(c_new).view(b, -1, self.n_heads, self.head_dim))
            v = self.w_uv(c_new).view(b, -1, self.n_heads, self.head_dim)
            # 全历史旋转键：先在 (b, T, rope_dim) 上旋转，再广播到各头
            k_rope_hist = apply_rope(
                k_rope_new,
                self.rope_cos[:k_rope_new.shape[1]],
                self.rope_sin[:k_rope_new.shape[1]],
            ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
            k_full = torch.cat([k, k_rope_hist], dim=-1).transpose(1, 2)
            v_full = v.transpose(1, 2)
            # 解码时 query 只有 1 步，对全历史做非因果注意力即可
            out = F.scaled_dot_product_attention(
                q_full, k_full, v_full, is_causal=False, dropout_p=0.0,
            )
            out = out.transpose(1, 2).contiguous().view(b, t, -1)
            return self.w_o(out), (c_new, k_rope_new)
        # 训练/Prefill：全序列并行
        q, k, v, q_rope, k_rope, c_kv, k_rope_raw = self._qkv(h)
        k_rope_full = k_rope.expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_full], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_full, k_full, v_full, is_causal=True, dropout_p=self.dropout,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        if not return_state:
            return self.w_o(out), None
        return self.w_o(out), (c_kv, k_rope_raw)

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """每 token KV 缓存字节数（对比 MHA 说明压缩收益）."""
        return (self.kv_lora_rank + self.qk_rope_dim) * dtype_bytes
