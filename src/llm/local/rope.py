"""RoPE 旋转位置编码 + YaRN 外推（NTK-by-parts + 幅度校正）.

## 为什么不能只把频率除以 scale

朴素的"`inv_freq / s`"把**所有**维度一起拉长：高频维度（负责区分相邻 token）也被
压缩，短距离分辨率跟着掉，长文里"位置近似、内容相似"的片段开始互相干扰。
YaRN（Peng et al., 2024）的做法是按**波长**分三段：

| 维度（波长 λ 相对训练长度） | 处理 | 原因 |
|---|---|---|
| λ 短（高频，β_fast=32 圈以内） | **不缩放**（外推） | 短距离分辨率不能动 |
| λ 长（低频，β_slow=1 圈以上） | **除以 s**（插值） | 训练时从没见过的超长距离靠插值 |
| 中间 | 线性斜坡混合 | 避免两段之间出现"频率断点" |

再乘一个幅度校正 `mscale = 0.1·ln(s) + 1`（长上下文注意力熵变高，logits 需要放大
一点）；我们把它折进 `cos/sin`（等价于把 q、k 各放大 √mscale），因此不需要改任何
注意力实现。

`scale=1` 时整条路径**逐位退化为标准 RoPE**（不建斜坡、不乘 mscale），
所以老 checkpoint 不受影响。
"""

from __future__ import annotations

import math

import torch


def _standard_inv_freq(dim: int, theta: float, device=None) -> torch.Tensor:
    """标准 RoPE 角频率：`θ^(-2i/d)`，i ∈ [0, d/2)."""
    half = dim // 2
    return 1.0 / (theta ** (torch.arange(half, dtype=torch.float32,
                                        device=device) / half))


def _correction_dim(num_rotations: float, dim: int, theta: float,
                    original_len: int) -> float:
    """YaRN 分界：跑完 `num_rotations` 圈的长度对应的维度下标（HF 同款公式）."""
    return (dim * math.log(original_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(theta))


def yarn_inv_freq(
    dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    original_len: int = 8192,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device=None,
) -> torch.Tensor:
    """YaRN 的 NTK-by-parts 角频率（长度 d/2）.

    - 高频段（下标小）：保持 `1/θ^(2i/d)`（外推，短距离分辨率不变）；
    - 低频段（下标大）：改为 `1/(s·θ^(2i/d))`（插值，拉长有效上下文）；
    - 中间段：按 `beta_fast`/`beta_slow` 决定的 [low, high] 区间线性斜坡。

    `scale=1` 直接返回标准频率（逐位一致）。
    """
    inv = _standard_inv_freq(dim, theta, device)
    s = float(scale)
    if s <= 1.0:
        return inv
    half = inv.numel()
    low = math.floor(_correction_dim(beta_fast, dim, theta, original_len))
    high = math.ceil(_correction_dim(beta_slow, dim, theta, original_len))
    low = max(low, 0)
    high = min(high, half - 1)
    if high <= low:  # 极端参数下退化为"全插值"
        return inv / s
    ramp = torch.clamp((torch.arange(half, dtype=torch.float32,
                                     device=device) - low) / (high - low), 0, 1)
    # ramp=0（高频，下标小）→ 外推（不动）；ramp=1（低频，下标大）→ 插值（÷s）
    return (inv / s) * ramp + inv * (1 - ramp)


def yarn_mscale(scale: float) -> float:
    """YaRN 幅度校正 `0.1·ln(s)+1`（s=1 时为 1.0，即不校正）."""
    s = float(scale)
    return 1.0 if s <= 1.0 else 0.1 * math.log(s) + 1.0


def build_rope_cache(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    yarn_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    yarn_original_len: int = 8192,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构建 RoPE 的 cos/sin 缓存 `(seq_len, dim)`.

    `yarn_scale>1` 时频率走 NTK-by-parts（见 `yarn_inv_freq`），并乘幅度校正
    `√mscale`；`yarn_scale=1` 时与标准 RoPE 逐位相同。
    """
    if dim % 2:
        raise ValueError(f"RoPE 维度须为偶数，当前 {dim}")
    inv = yarn_inv_freq(dim, theta, yarn_scale, yarn_original_len,
                        yarn_beta_fast, yarn_beta_slow, device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv).to(dtype)
    emb = torch.cat([freqs, freqs], dim=-1)
    gain = math.sqrt(yarn_mscale(yarn_scale))
    return (emb.cos() * gain).to(dtype), (emb.sin() * gain).to(dtype)


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


def yarn_scale_mismatch(scale: float, original_len: int, target_len: int,
                        tol: float = 2.0) -> bool:
    """`yarn_scale` 与 `target/original` 的比例是否"差得离谱"（默认容差 2 倍）.

    用途：启动自检。比如训练长度 8K、目标 1M（比例 128）却写 `yarn_scale=8`，
    等于只外推了 64K，后面全是裸外推——这种配置不会报错，只会悄悄掉点。
    """
    if target_len <= original_len or scale <= 1.0:
        return False
    ideal = float(target_len) / float(original_len)
    hi, lo = max(scale, ideal), max(1e-9, min(scale, ideal))
    return hi / lo > tol


def yarn_scaling_factor(train_len: int, target_len: int,
                        max_scale: float = 512.0) -> float:
    """按目标长度估算 YaRN scale（目标/训练长度，截断到 [1, max_scale]）.

    上界 512 是给"4K 训练 → 1M 推理"这种 256 倍外推留的空间
    （V4.1 的上下文从 4K 扩到 1M，正好 256×）；旧实现截断到 8 会让 1M 直接失效。
    """
    if target_len <= train_len:
        return 1.0
    return min(max_scale, target_len / train_len)
