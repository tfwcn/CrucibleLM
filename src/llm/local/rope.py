"""RoPE 旋转位置编码 + YaRN 外推缩放."""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    yarn_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构建 RoPE 的 cos/sin 缓存.

    YaRN 简化版：把 inv_freq 除以 yarn_scale，等效拉长有效上下文。
    yarn_scale=1 时退化为标准 RoPE。
    """
    # 角频率：theta^(-2i/d)，YaRN 用 scale 稀释高频外推压力
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    inv_freq = inv_freq / yarn_scale
    # 位置 * 频率 -> 外积得到 (seq_len, half)
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq).to(device=device, dtype=dtype)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    """对 x 做旋转位置编码.

    x: 任意形状，序列轴由 seq_dim 指定（如 (b, t, dim) 用 1，
    (b, t, heads, dim) 也用 1）；
    cos/sin: (seq_len, dim)，只取前 seq 片并广播到其余轴。
    """
    seq = x.shape[seq_dim]
    c = cos[:seq].to(device=x.device, dtype=x.dtype)
    s = sin[:seq].to(device=x.device, dtype=x.dtype)
    # 构造广播形状：除序列轴与最后一轴外全为 1
    shape = [1] * x.dim()
    shape[seq_dim] = seq
    shape[-1] = c.shape[-1]
    c = c.view(*shape)
    s = s.view(*shape)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # cos/sin 前后半相同（构建时 cat 了两份 freqs），直接复用整块
    return torch.cat([x1 * c[..., :half] - x2 * s[..., :half],
                      x1 * s[..., half:] + x2 * c[..., half:]], dim=-1)


def yarn_scaling_factor(train_len: int, target_len: int) -> float:
    """按目标长度估算 YaRN scale（目标/训练长度，截断到 [1, 8]）."""
    if target_len <= train_len:
        return 1.0
    return min(8.0, target_len / train_len)
