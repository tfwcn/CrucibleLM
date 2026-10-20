"""Engram 条件记忆 — n-gram 哈希查表（DeepSeek-V4.1 §2.4.2 口径，默认关）.

与 `memory.py` 的 Product-Key 不同：这里用**确定性 n-gram 哈希**查表——键是
"过去 n-1 个 token + 当前 token" 组成的局部上下文，经多头 multiplicative-XOR
哈希映射到素数大小的表，O(1) 取回静态记忆 `e_t`；再用当前 hidden 作 query 门控
（`α=σ(RMSNorm(h)·RMSNorm(W_K e)/√d)`），拿 `W_V e` 后残差加回主干。

V4.1 相对原版 Engram 的两处修改，本模块都对齐：

1. **去掉短因果卷积**（原版 kernel=4）。论文的理由是"收益不值得推理栈多一份复杂度"。
   于是 `lookback = max_order - 1`：解码每步只需 `max_order-1` 个历史 id，
   没有跨步卷积状态，天然因果。
2. **FP8**：哈希表与 K/V 投影都走 FP8。训练用 fake-quant + STE（QAT，模型学的是
   "在 FP8 误差下仍然好用"的表示）；推理可 `pack()` 成**真 FP8 存储**（表从
   `2~4 B/通道` 降到 `1.06 B/通道`）。表更新交给 Sinkhorn（`optim.py` 按参数名
   `...engram.tables.K.weight` 路由，与 token embedding / 预测头同一档）。

规格（论文）：阶 `{2,3,4}`、每阶 8 个哈希头、每阶记忆维 2048（→ 256/头）、
每头约 16M 行的素数表、两个模块（放在少数早期层）。本地按规模缩
（`engram_table_size` 默认 8191），机制不变。

为什么叫"条件记忆"：MoE 用稀疏**计算**处理动态逻辑，Engram 用稀疏**查找**取回
静态知识（实体、公式、模板）。论文发现二者互补（U 型分配律：约 20~25% 稀疏预算
给记忆最优），且只插少数早期层即可"替主干提前完成特征组装"，等效加深网络；
确定性寻址还让它可预取（`e` 只依赖 token id、不依赖 hidden，故可先算）。

恒等起点：`W_V` 零初始化 → 开局输出恒为 0，`h ← h + 0`，不破坏已训权重
（与 memory/hyperconn 同约定，单测锁定）。FP8 下 0 仍是 0（量化不引入偏置）。

默认关（`engram_every=0` 且 `engram_layers` 为空）时模块不存在，
`state_dict` 与旧 checkpoint 逐位兼容。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.kv_quant import dequantize_fp8, fake_quant, quantize_fp8
from src.llm.local.mla import RMSNorm

# 乘法哈希乘子（与 RF 类似取大素数，打散乘性规律）
_PRIME_MUL = 1000003
_MASK31 = 0x7FFF_FFFF
# FP8 块量化的通道组大小（与 kv_quant 一致）
_QBLOCK = 16


def is_prime(n: int) -> bool:
    """朴素素性判定（表大小须为素数，抗碰撞；fail-fast 用）."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def hash_ngrams(gram: torch.Tensor, table_size: int, seed: int) -> torch.Tensor:
    """确定性 multiplicative-XOR 哈希： `(..., n)` → `(...)` 表下标.

    相同 n-gram 恒得同一下标（正是"确定性寻址可预取"的基础）；`seed` 让同一
    n-gram 在不同头/阶落到不同表。
    """
    h = torch.full(gram.shape[:-1], int(seed), dtype=torch.long, device=gram.device)
    for j in range(gram.shape[-1]):
        h = (h * _PRIME_MUL + gram[..., j].long()) & _MASK31
    # xor-shift 低位混合，避免仅低位参与导致聚集
    h = ((h ^ (h >> 13)) * _PRIME_MUL) & _MASK31
    h = (h ^ (h >> 7)) & _MASK31
    return h % int(table_size)


class EngramTable(nn.Module):
    """一张 n-gram 哈希表：FP8 查找（QAT）+ 可打包成真 FP8 存储（推理期）.

    两种状态：
    - **训练/未打包**：主权重是普通 `nn.Parameter`（优化器要它），前向把
      **查出来的行**做一次 fake-quant。按行量化与按表量化逐位等价（块量化的
      scale 只在本行内取），但只碰被查的行 → 不会每步扫全表。
    - **已打包**（`pack()`，仅推理）：权重压成 `uint8` 码字 + E4M3 scale，
      前向从打包存储取行再反量化；数值与 fake-quant 路径**逐位相同**（量化-
      反量化幂等，见 pitfalls 45）。打包是**快照**，训练中禁止（会读到旧表）。
    """

    def __init__(self, rows: int, cols: int, fp8: bool = True):
        super().__init__()
        self.rows = int(rows)
        self.cols = int(cols)
        self.fp8 = bool(fp8)
        if self.fp8:
            assert self.cols % _QBLOCK == 0, (
                f"FP8 表的每头维度须被 {_QBLOCK} 整除，当前 {self.cols}")
        self.weight = nn.Parameter(torch.empty(self.rows, self.cols))
        # 打包后的真 FP8 存储（uint8 视图）；None = 未打包。transient（不进 state_dict）
        self.register_buffer("w8", None, persistent=False)
        self.register_buffer("s8", None, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """查表：`idx (...)` → `(..., cols)`（FP8 精度）."""
        if self.w8 is not None:
            if self.training:
                raise RuntimeError(
                    "Engram 表已打包成 FP8 存储（快照）：训练会读到旧表，"
                    "先 `unpack()` 再训练")
            return self._lookup_packed(idx)
        rows = F.embedding(idx, self.weight)
        return fake_quant(rows, "fp8") if self.fp8 else rows

    def _lookup_packed(self, idx: torch.Tensor) -> torch.Tensor:
        """从打包存储取行并反量化（与 fake-quant 路径逐位相同）."""
        flat = idx.reshape(-1)
        q = self.w8.index_select(0, flat).view(torch.float8_e4m3fn)
        s = self.s8.index_select(0, flat).view(torch.float8_e4m3fn)
        q = q.view(*idx.shape, self.cols)
        s = s.view(*idx.shape, self.cols // _QBLOCK)
        return dequantize_fp8(q, s, _QBLOCK).to(self.weight.dtype)

    @torch.no_grad()
    def pack(self) -> None:
        """把整张表压成真 FP8 存储（`rows × (cols + cols/16)` 字节）."""
        if self.training:
            raise RuntimeError(
                "训练中不要打包 Engram 表：打包是快照，不随优化器步进更新")
        q, s = quantize_fp8(self.weight.detach(), _QBLOCK)
        self.w8 = q.view(torch.uint8).clone()
        self.s8 = s.view(torch.uint8).clone()

    def unpack(self) -> None:
        """丢弃打包存储，回到主权重（训练/继续训练前必做）."""
        self.w8 = None
        self.s8 = None

    def nbytes(self) -> int:
        """当前实际存储字节数（打包后 1.0625 B/通道，否则按主权重 dtype）."""
        if self.w8 is not None:
            return self.w8.numel() + self.s8.numel()
        return self.weight.numel() * self.weight.element_size()


class EngramMemory(nn.Module):
    """n-gram 哈希条件记忆（残差式；`W_V` 零初始化 → 开局恒等）.

    规格对齐 V4.1 §2.4.2：阶 `{2,3,4}`、每阶 8 个哈希头、每阶记忆维 `d_mem`
    （论文 2048）、**无短卷积**、表 + K/V 投影 FP8（QAT）、表更新走 Sinkhorn。
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        orders: tuple[int, ...] | list[int] = (2, 3, 4),
        n_heads: int = 8,
        d_mem: int = 256,
        table_size: int = 8191,
        fp8: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.orders = [int(o) for o in orders]
        assert self.orders and all(o >= 2 for o in self.orders), \
            f"Engram 阶须 ≥2，当前 {self.orders}"
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1, "engram_heads 须 ≥1"
        self.table_size = int(table_size)
        assert is_prime(self.table_size), \
            f"engram_table_size={table_size} 须为素数（抗碰撞，尽量互异）"
        self.max_order = max(self.orders)
        # 左边界占位：仅参与哈希，不查真实词表（只是一个固定的"句首"符号）
        self.boundary = int(vocab_size)
        assert int(d_mem) % self.n_heads == 0, (
            f"engram_d_mem={d_mem}（每阶维度）须被头数={self.n_heads} 整除")
        self.d_mem = int(d_mem)          # **每阶**记忆维（论文 2048）
        self.d_head = self.d_mem // self.n_heads
        self.d_out = self.d_mem * len(self.orders)   # 各阶拼接后的维度
        self.fp8 = bool(fp8)
        if self.fp8:
            assert self.d_out % _QBLOCK == 0, (
                f"FP8 投影要求拼接维 {self.d_out} 被 {_QBLOCK} 整除")
        # 每 (阶, 头) 一张素数大小的表；各头 seed 不同 → 同一 n-gram 多头分散
        # （论文用"互异素数"进一步打散，这里同素数 + 异 seed 已足够分散）
        self.tables = nn.ModuleList()
        self.seeds: list[int] = []
        for o in self.orders:
            for k in range(self.n_heads):
                self.tables.append(
                    EngramTable(self.table_size, self.d_head, self.fp8))
                self.seeds.append((o * 1009 + k * 97 + 13) & _MASK31)
        # 上下文门控：hidden 作 Q，记忆作 K/V（RMSNorm 稳定梯度）
        self.w_k = nn.Linear(self.d_out, d_model, bias=False)
        self.w_v = nn.Linear(self.d_out, d_model, bias=False)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.dropout = dropout
        self._init_weights()

    @property
    def lookback(self) -> int:
        """解码须提供的右对齐上下文长度：`max_order - 1`（无卷积状态）."""
        return self.max_order - 1

    def _init_weights(self) -> None:
        """恒等起点：Value 投影零初始化 → 开局输出恒 0（FP8 下 0 仍是 0）."""
        nn.init.zeros_(self.w_v.weight)
        nn.init.normal_(self.w_k.weight, std=0.02)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    def _proj(self, lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """FP8 权重的线性投影（QAT：fake-quant + STE；推理即论文的 FP8 K/V 投影）."""
        w = fake_quant(lin.weight, "fp8") if self.fp8 else lin.weight
        return F.linear(x, w)

    def _grams_last(self, ctx: torch.Tensor, n: int, t: int) -> torch.Tensor:
        """取 `ctx` 末 `t` 位的 n-gram： `(b,L)` → `(b,t,n)`（左补 boundary）.

        `gram[p, i] = ctx[p + i]`，即位置 p 的 gram 为 `(x_{p-n+1}, …, x_p)`。
        """
        need = t + n - 1  # 末 t 位各需 n 个 token
        if ctx.shape[1] < need:
            pad = ctx.new_full((ctx.shape[0], need - ctx.shape[1]), self.boundary)
            ctx = torch.cat([pad, ctx], dim=1)
        ctx = ctx[:, ctx.shape[1] - need:]
        return torch.stack([ctx[:, i:i + t] for i in range(n)], dim=-1)

    def _embed(self, ids: torch.Tensor, tv: int) -> torch.Tensor:
        """为末 `tv` 位取回多头 n-gram 记忆并拼接： `(b,L)` → `(b,tv,d_out)`."""
        embeds: list[torch.Tensor] = []
        gi = 0
        for o in self.orders:
            gram = self._grams_last(ids, o, tv)  # (b,tv,o)
            for _ in range(self.n_heads):
                idx = hash_ngrams(gram, self.table_size, self.seeds[gi])
                embeds.append(self.tables[gi](idx))
                gi += 1
        return torch.cat(embeds, dim=-1)

    def forward(self, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """记忆残差：`h (b,t,d)` + 由 `ids (b,L≥t)`（右对齐到 h 末位）取回的门控记忆.

        无短卷积：`ids` 只需末 `t` 位算 n-gram（首 token 左补 boundary），
        故解码每步与全前向逐位一致，且**无跨步状态**。
        """
        t = h.shape[1]
        e = self._embed(ids, t)                    # (b,t,d_out)
        v = self._proj(self.w_v, e)                # (b,t,d_model)
        kk = self.norm_k(self._proj(self.w_k, e))
        # 上下文门控：记忆与当前上下文不一致时 α→0，自动抑制噪声
        gate = (self.norm_q(h) * kk).sum(-1, keepdim=True) / math.sqrt(h.shape[-1])
        y = torch.sigmoid(gate) * v
        if self.training and self.dropout > 0:
            y = F.dropout(y, p=self.dropout)
        return h + y

    # ---- FP8 存储：打包 / 解包 / 计费（推理期真省内存，训练必须解包）----
    def pack(self) -> None:
        """所有表打包成真 FP8 存储（推理期；训练前必须 `unpack()`）."""
        for t in self.tables:
            t.pack()

    def unpack(self) -> None:
        """所有表回到主权重."""
        for t in self.tables:
            t.unpack()

    def nbytes(self) -> int:
        """所有表的实际存储字节数（打包后约 1.06 B/通道）."""
        return sum(t.nbytes() for t in self.tables)
