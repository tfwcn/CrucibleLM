"""KV 缓存量化：MXFP4（E2M1 + 每 16 通道 E4M3 scale）与 FP8（E4M3）.

V4.1 §2.4.4：主 KV 用 **FP4** 存储（反量化后再算注意力，所以不需要硬件支持 FP4
矩阵乘），SWA KV 保留 **FP8**（对量化更敏感）。量化格式取 OCP 标准的 MXFP4：
E2M1（4 bit，值域 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}）+ 每 16 个通道一个 E4M3 scale，
**不带** NVFP4 的第二级全局 scale（论文：主 KV 幅值上界只有 ~22.6，量程足够）。

QAT：训练期用"量化-反量化 + 直通梯度"（STE）让模型适应量化误差；推理期同一套
数值（先反量化再算注意力）——所以训练与推理**逐位一致**，不需要两套 kernel。

本文件只提供"数学 + 打包"，不碰模型结构：
- `quantize_e2m1(x)` / `dequantize_e2m1(idx, scale)`：逐块量化/反量化；
- `QuantBuffer`：缓存布局（packed uint8 + E4M3 scale），提供 `dequant()`、
  `write_()`、`resize()`、`clone()`——存储真省（FP4 约 4.5 bit/通道，FP8 约 8.5），
  读取时一次反量化成 float 再参与注意力（与 V4.1 同口径）。
"""

from __future__ import annotations

import torch

# E2M1 的 15 个可表示值（升序；负零与正零合并）。位模式 → 值：
#   s eem（1 符号 + 2 指数 + 1 尾数），值 = sign * {0, .5, 1, 1.5, 2, 3, 4, 6}
E2M1_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
               0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
# 4-bit 码字：0..7 = 正（0,.5,1,1.5,2,3,4,6），8..15 = 负（同幅值）
E2M1_CODES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32)

FP8_E4M3_MAX = 448.0
# 量化块大小（MXFP4：每 16 个通道一个 E4M3 scale）
KV_QUANT_BLOCK = 16


def _floor_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """把正数张量**向下**舍入到 E4M3 可表示值（保证结果 ≤ 输入）.

    为什么必须向下：量化误差的"幂等性"依赖"反量化后的块最大值恰为
    `max_code × scale`"。若 scale 向上舍入（> amax/k），反量化最大 < 原 amax，
    再量化就会得到更小的 scale——量化两次就漂移一次（实测 0.055，足以让
    prefill 与增量解码分叉）。向下舍入时反量化的最大值正好回到 `scale` 的整数倍，
    再量化得到**同一个** scale。
    """
    q = x.to(torch.float8_e4m3fn).float()
    for _ in range(2):  # 就近舍入可能向上；退一档（1-1/16 超过 E4M3 最大相对步长）
        q = torch.where(q > x, (q * (1 - 1 / 16)).to(torch.float8_e4m3fn).float(),
                        q)
    return q.to(torch.float8_e4m3fn)


def _nearest_code(v: torch.Tensor) -> torch.Tensor:
    """把值映射到最近的 4-bit 码字（0..15），输入已除以 scale.

    在**有序**码字表上二分：E2M1 的正负幅值对称，故按 (|v|, sign) 处理——
    幅值走 7 个正码字的 `searchsorted`，符号决定 +8 偏移（0 归正）。
    """
    a = v.abs()
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                       device=v.device, dtype=torch.float32)
    idx = torch.searchsorted(mag, a.contiguous()).clamp(max=7)
    # 与左侧邻居比距离，取更近的
    left = (idx - 1).clamp(min=0)
    d_right = (mag[idx] - a).abs()
    d_left = (mag[left] - a).abs()
    idx = torch.where(d_left < d_right, left, idx)
    neg = (v < 0).to(torch.uint8) * 8
    return idx.to(torch.uint8) + neg


def quantize_e2m1(
    x: torch.Tensor, block: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿最后一维按 `block` 个通道一组做 MXFP4 量化.

    返回 `(codes, scales)`：`codes` 是 uint8（值 0..15，**未打包**，形状同 x），
    `scales` 是 (..., last//block) 的 E4M3 张量（每块一个 scale，`max_abs/6`）。
    """
    assert x.shape[-1] % block == 0, (
        f"最后一维 {x.shape[-1]} 必须能被量化块 {block} 整除")
    xf = x.float()
    blocks = xf.view(*xf.shape[:-1], xf.shape[-1] // block, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # **先把 scale 舍入到它的存储格式（E4M3），再用它算码字**：否则量化用的是
    # "精确 scale"，而反量化用的是"存储后的 scale"，两者差一个 E4M3 舍入步，
    # 误差会超过半个码字间距（实测 1.145 vs 半步 0.875）。
    # scale **向下**舍入到 E4M3（见 `_floor_to_e4m3`）：保证 amax/scale ≥ 6，
    # 于是最大码字 = 6、反量化最大值 = 6*scale，再量化得到同一个 scale（幂等）。
    scale_store = _floor_to_e4m3(
        (amax / E2M1_MAX).clamp(min=1e-8, max=FP8_E4M3_MAX))
    scale = scale_store.float()
    codes = _nearest_code(blocks / scale)
    codes = codes.view(x.shape)
    return codes, scale_store.squeeze(-1)


def dequantize_e2m1(
    codes: torch.Tensor, scale: torch.Tensor, block: int = 16,
) -> torch.Tensor:
    """反量化：`code → 值 × scale`（返回 fp32）."""
    vals = E2M1_CODES.to(codes.device)[codes.long()]
    s = scale.float().unsqueeze(-1).expand(
        *vals.shape[:-1], vals.shape[-1] // block, block).reshape(vals.shape)
    return vals * s


def quantize_fp8(
    x: torch.Tensor, block: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿最后一维按 `block` 通道一组做 FP8（E4M3）量化，返回 (fp8, scales)."""
    assert x.shape[-1] % block == 0
    xf = x.float()
    blocks = xf.view(*xf.shape[:-1], xf.shape[-1] // block, block)
    # 同 e2m1：scale 向下舍入到 E4M3（幂等），amax/scale ≥ 1 由下面的 clamp 兜住
    scale = _floor_to_e4m3(
        blocks.abs().amax(dim=-1, keepdim=True).clamp(
            min=1e-8, max=FP8_E4M3_MAX))
    q = (blocks / scale.float()).clamp(-1.0, 1.0).to(torch.float8_e4m3fn)
    return q.view(x.shape), scale.squeeze(-1)


def dequantize_fp8(
    q: torch.Tensor, scale: torch.Tensor, block: int = 16,
) -> torch.Tensor:
    """反量化 FP8：`q × scale`（返回 fp32）."""
    vals = q.float()
    s = scale.float().unsqueeze(-1).expand(
        *vals.shape[:-1], vals.shape[-1] // block, block).reshape(vals.shape)
    return vals * s


def fake_quant(
    x: torch.Tensor, fmt: str, block: int = 16,
) -> torch.Tensor:
    """量化-反量化（QAT 用），返回与输入同 dtype；梯度**直通**（STE）.

    训练期每个前向都把缓存写成"会被量化成的值"，反向按恒等传（`x + (q-x).detach()`），
    于是模型学到的是"在量化误差下仍然好用"的表示——这就是 QAT。
    """
    if not fmt:
        return x
    if fmt == "fp4":
        codes, scale = quantize_e2m1(x, block)
        q = dequantize_e2m1(codes, scale, block)
    elif fmt == "fp8":
        q8, scale = quantize_fp8(x, block)
        q = dequantize_fp8(q8, scale, block)
    else:
        raise ValueError(f"未知 KV 量化格式 {fmt!r}（支持 'fp4'/'fp8'/空）")
    return (x.float() + (q - x.float()).detach()).to(x.dtype)


def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    """把 (..., n) 的 4-bit 码字打包成 (..., n/2) uint8（偶通道在高 4 位）."""
    n = codes.shape[-1]
    assert n % 2 == 0, "4-bit 打包要求最后一维为偶数"
    c = codes.to(torch.uint8)
    hi = c[..., 0::2] << 4
    lo = c[..., 1::2]
    return (hi | lo).contiguous()


def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """把 (..., n/2) uint8 解回 (..., n) 的 4-bit 码字."""
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2,
                      device=packed.device, dtype=torch.uint8)
    out[..., 0::2] = hi
    out[..., 1::2] = lo
    return out


class QuantBuffer:
    """量化缓存缓冲：存储 `packed`/`fp8` + E4M3 scale，逻辑上仍是 (b, cap, r).

    对外提供解码所需的最小接口（`dequant()`/`write_()`/`resize()`/`clone()`/
    `shape`），让 MLA 的缓存读写不必关心"存的是 4-bit 还是 8-bit"。
    """

    def __init__(self, data: torch.Tensor, scale: torch.Tensor,
                 logical_shape: tuple[int, ...], fmt: str, block: int = 16):
        # 内部一律用 **uint8** 存（fp8 数据/所有 scale 都是 E4M3 的 bit pattern）：
        # torch 的 `float8_e4m3fn` 在 CPU 上缺 index_copy_/赋值算子，包成 uint8
        # 后读写都能走常规路径；反量化时再 view 回 fp8。
        self.data = data              # fp4: uint8 (…, r/2)；fp8: uint8 (…, r)
        self.scale = scale            # uint8 (…, r/block)，E4M3 bit pattern
        self.logical_shape = tuple(logical_shape)
        self.fmt = fmt
        self.block = block

    # ---- 构造 ----
    @classmethod
    def zeros(cls, shape: tuple[int, ...], fmt: str, block: int = 16,
              device=None, dtype=None) -> "QuantBuffer":
        r = shape[-1]
        if fmt == "fp4":
            data = torch.zeros(*shape[:-1], r // 2, device=device,
                               dtype=torch.uint8)
        elif fmt == "fp8":
            data = torch.zeros(*shape, device=device, dtype=torch.uint8)
        else:
            raise ValueError(f"未知 KV 量化格式 {fmt!r}")
        scale = torch.zeros(*shape[:-1], r // block, device=device,
                            dtype=torch.uint8)
        return cls(data, scale, shape, fmt, block)

    @classmethod
    def from_tensor(cls, x: torch.Tensor, fmt: str,
                    block: int = 16) -> "QuantBuffer":
        if fmt == "fp4":
            codes, scale = quantize_e2m1(x, block)
            data = pack_4bit(codes)
        elif fmt == "fp8":
            data, scale = quantize_fp8(x, block)
            data = data.view(torch.uint8)
        else:
            raise ValueError(f"未知 KV 量化格式 {fmt!r}")
        return cls(data, scale.view(torch.uint8), tuple(x.shape), fmt, block)

    # ---- 读 ----
    def dequant(self) -> torch.Tensor:
        """反量化成 fp32 张量 (b, cap, r)（每次读取一次，V4.1 同口径）."""
        scale = self.scale.view(torch.float8_e4m3fn)
        if self.fmt == "fp4":
            codes = unpack_4bit(self.data)
            return dequantize_e2m1(codes, scale, self.block)
        q = self.data.view(torch.float8_e4m3fn)
        return dequantize_fp8(q, scale, self.block)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.logical_shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        # 反量化后的 dtype（缓存逻辑类型跟着模型精度走，存储才是 4/8 bit）
        return torch.float32

    def nbytes(self) -> int:
        return (self.data.numel() * self.data.element_size()
                + self.scale.numel() * self.scale.element_size())

    # ---- 写（index_copy_ 语义：把 values 量化后写到 [start, start+t)）----
    def write_(self, start: int, values: torch.Tensor) -> "QuantBuffer":
        """把 `values` (b, t, r) 量化后写入第 `start` 行起（返回 self 便于链式）."""
        t = values.shape[1]
        idx = torch.arange(start, start + t, device=values.device)
        if self.fmt == "fp4":
            codes, scale = quantize_e2m1(values, self.block)
            packed = pack_4bit(codes)
            self.data.index_copy_(1, idx, packed)
        else:
            data, scale = quantize_fp8(values, self.block)
            self.data.index_copy_(1, idx, data.view(torch.uint8))
        self.scale.index_copy_(1, idx, scale.view(torch.uint8))
        return self

    def write_rows_(self, rows: torch.Tensor, values: torch.Tensor) -> "QuantBuffer":
        """按**任意行号**写（SWA 环的行号 = 位置 % cap，不是连续区间）."""
        if self.fmt == "fp4":
            codes, scale = quantize_e2m1(values, self.block)
            self.data.index_copy_(1, rows, pack_4bit(codes))
        else:
            data, scale = quantize_fp8(values, self.block)
            self.data.index_copy_(1, rows, data.view(torch.uint8))
        self.scale.index_copy_(1, rows, scale.view(torch.uint8))
        return self

    def gather_rows(self, rows: torch.Tensor) -> "QuantBuffer":
        """按行号取子集（重映射用；行号必须升序唯一）."""
        return QuantBuffer(self.data.index_select(1, rows).clone(),
                           self.scale.index_select(1, rows).clone(),
                           (self.logical_shape[0], int(rows.numel()),
                            *self.logical_shape[2:]),
                           self.fmt, self.block)

    def clone(self) -> "QuantBuffer":
        return QuantBuffer(self.data.clone(), self.scale.clone(),
                           self.logical_shape, self.fmt, self.block)

    def resize(self, cap: int) -> "QuantBuffer":
        """扩容到 `cap` 行（拷贝旧有效区，新行零）——与 Tensor 缓冲的 2x 扩容同语义."""
        old = self.logical_shape[1]
        if cap <= old:
            return self
        head = self.logical_shape[:-1]
        if self.fmt == "fp4":
            data = torch.zeros(*head[:-1], cap, self.data.shape[-1],
                               device=self.data.device, dtype=self.data.dtype)
        else:
            data = torch.zeros(*head[:-1], cap, self.data.shape[-1],
                               device=self.data.device, dtype=self.data.dtype)
        data[:, :old] = self.data
        scale = torch.zeros(*head[:-1], cap, self.scale.shape[-1],
                            device=self.scale.device, dtype=self.scale.dtype)
        scale[:, :old] = self.scale
        return QuantBuffer(data, scale, head[:-1] + (cap, self.logical_shape[-1]),
                           self.fmt, self.block)


def kv_bytes_per_token(rank: int, fmt: str, block: int = 16) -> float:
    """每 token 每份缓存的字节数（fp4: 4.5 bit/通道，fp8: 8.5 bit/通道）."""
    if not fmt:
        return 2.0 * rank
    bits = 4.5 if fmt == "fp4" else 8.5
    return bits / 8.0 * rank
