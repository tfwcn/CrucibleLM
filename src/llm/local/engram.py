"""Engram 条件记忆 — n-gram 哈希查表（DeepSeek Engram 思想的可训练子集，默认关）.

与 `memory.py` 的 Product-Key 不同：这里用**确定性 n-gram 哈希**查表——键是
"过去 n-1 个 token + 当前 token" 组成的局部上下文，经多头 multiplicative-XOR
哈希映射到素数大小的表，O(1) 取回静态记忆 `e_t`；再用当前 hidden 作 query 门控
（`α=σ(RMSNorm(h)·RMSNorm(W_K e)/√d)`），拿 `W_V e`，最后短因果卷积平滑后残差
加回主干。

为什么叫"条件记忆"：MoE 用稀疏**计算**处理动态逻辑，Engram 用稀疏**查找**取回
静态知识（实体、公式、模板）。论文发现二者互补（U 型分配律：约 20~25% 稀疏
预算给记忆最优），且只插少数早期层（layer 2 最佳）即可"替主干提前完成特征
组装"，等效加深网络；确定性寻址还让它可预取（`e` 只依赖 token id、不依赖
hidden，故可先算）。

恒等起点：`W_V` 与卷积权重零初始化（conv 无 bias）→ 开局输出恒为 0，
`h ← h + 0`，不破坏已训权重（与 memory/hyperconn 同约定，单测锁定）。

默认关（`engram_every=0`）时模块不存在，`state_dict` 与旧 checkpoint 逐位兼容。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import RMSNorm

# 乘法哈希乘子（与 RF 类似取大素数，打散乘性规律）
_PRIME_MUL = 1000003
_MASK31 = 0x7FFF_FFFF


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


class EngramMemory(nn.Module):
    """n-gram 哈希条件记忆（残差式；W_V/卷积零初始化 → 开局恒等）."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        orders: tuple[int, ...] | list[int] = (2, 3),
        n_heads: int = 2,
        d_mem: int = 64,
        table_size: int = 8191,
        conv_kernel: int = 4,
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
        groups = len(self.orders) * self.n_heads
        assert d_mem % groups == 0, \
            f"engram_d_mem={d_mem} 须被 阶数×头数={groups} 整除"
        self.d_head = d_mem // groups
        # 每 (阶, 头) 一张素数大小的表；各头 seed 不同 → 同一 n-gram 多头分散
        self.tables = nn.ModuleList()
        self.seeds: list[int] = []
        for o in self.orders:
            for k in range(self.n_heads):
                self.tables.append(nn.Embedding(self.table_size, self.d_head))
                self.seeds.append((o * 1009 + k * 97 + 13) & _MASK31)
        # 上下文门控：hidden 作 Q，记忆作 K/V（RMSNorm 稳定梯度）
        self.w_k = nn.Linear(d_mem, d_model, bias=False)
        self.w_v = nn.Linear(d_mem, d_model, bias=False)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.norm_v = RMSNorm(d_model)
        # 短因果卷积：膨胀率取最大阶，扩大感受野（论文 kernel=4）
        self.dilation = self.max_order
        if conv_kernel and conv_kernel > 0:
            self.conv: nn.Conv1d | None = nn.Conv1d(
                d_model, d_model, int(conv_kernel), groups=d_model,
                dilation=self.dilation, bias=False)
        else:
            self.conv = None
        self.dropout = dropout
        self._init_weights()

    @property
    def extra(self) -> int:
        """短卷积的左右感受野（因果卷积左侧需要的前文长度）."""
        if self.conv is None:
            return 0
        return (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

    @property
    def lookback(self) -> int:
        """解码时须提供的右对齐上下文长度（补足卷积感受野 + n-gram 阶）."""
        # 卷积读 `V`（id 决定）；`V[p]` 又依赖 `p-max_order+1..p` 的 id，故共需
        # `extra + max_order - 1` 个历史 id 才能算准末位。
        return self.extra + self.max_order - 1

    def _init_weights(self) -> None:
        """恒等起点：Value 投影与卷积零初始化（conv 无 bias）→ 输出恒 0."""
        nn.init.zeros_(self.w_v.weight)
        if self.conv is not None:
            nn.init.zeros_(self.conv.weight)
        nn.init.normal_(self.w_k.weight, std=0.02)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

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
        """为末 `tv` 位取回多头 n-gram 记忆并拼接： `(b,L)` → `(b,tv,d_mem)`."""
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
        """记忆残差：`h (b,t,d)` + 由 `ids (b,L)`（右对齐到 h 末位）取回的门控记忆.

        短卷积作用在**仅由 id 决定**的 `V=W_V e` 上（而非门控后的 `Ṽ`）：这样
        解码每步只需窗口内 `L ≥ lookback` 个历史 id，**无需跨步卷积状态**，
        单步 O(lookback)，也天然满足因果（改未来 id 不改前部输出）。
        """
        t = h.shape[1]
        tv = t + self.extra  # 多取 extra 位供卷积左侧感受野
        e = self._embed(ids, tv)          # (b,tv,d_mem)
        v = self.w_v(e)                   # (b,tv,d_model)
        if self.conv is not None:
            z = F.pad(self.norm_v(v).transpose(1, 2), (self.extra, 0))
            z = self.conv(z).transpose(1, 2)   # 因果：out[i] 只用 V[i-extra..i]
            v = F.silu(z) + v
        v = v[:, -t:]
        kk = self.norm_k(self.w_k(e[:, -t:]))
        # 上下文门控：记忆与当前上下文不一致时 α→0，自动抑制噪声
        gate = (self.norm_q(h) * kk).sum(-1, keepdim=True) / math.sqrt(h.shape[-1])
        y = torch.sigmoid(gate) * v
        if self.training and self.dropout > 0:
            y = F.dropout(y, p=self.dropout)
        return h + y
