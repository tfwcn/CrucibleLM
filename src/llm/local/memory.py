"""Product-Key 记忆层 — CPU 可放的稀疏知识表（Meta Memory Layers / DeepSeek Engram 思想）.

结构：查询切两半，分别在两个 √M 子码本里 top-k，再笛卡尔组合重排取 top-k，
只把 k 个 value 行搬上计算设备加权求和。查找本身 O(√M)，天然适合放 CPU：
搬运量恒为 k 行，与表大小无关。

- 表放 CPU（`to("cpu")` 的 Parameter，优化器按参维护状态，autograd 经
  index_select/index_add 回传，无需手写）；
- value 全零初始化 = 精确恒等（输出恒为输入），开局不破坏已训权重；
- B 方案初始化（见 `init_memory_from_activations`）：backbone 冻结跑校准集，
  hidden 做 k-means，类中心劈半当子码本，value 保持零（恒等起点最安全）。

config.memory_every=0 时模块不存在，state_dict 与旧版逐位兼容。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import RMSNorm


class ProductKeyMemory(nn.Module):
    """乘积键记忆：(b, T, d) -> 同形，残差式（输出 = 输入 + 查表增量）."""

    def __init__(
        self,
        d_model: int,
        n_slots: int = 4096,
        top_k: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model 须为偶数（查询劈半）"
        side = int(n_slots ** 0.5)
        assert side * side == n_slots, "n_slots 须为完全平方数（√M×√M 子码本）"
        self.side = side
        self.top_k = top_k
        half = d_model // 2
        # 子码本放 CPU（稀疏查找不吃并行，显存只留命中的 k 行）
        self.keys1 = nn.Parameter(torch.randn(side, half) * 0.02)
        self.keys2 = nn.Parameter(torch.randn(side, half) * 0.02)
        self.values = nn.Parameter(torch.zeros(n_slots, d_model))
        self.norm = RMSNorm(d_model)
        self.dropout = dropout
        self.to("cpu")

    def _lookup(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (选中 value 行索引 (N, k), 权重 (N, k))，N = b*T."""
        dev, dt = x.device, x.dtype
        q = x.reshape(-1, x.shape[-1])
        q1, q2 = q[:, :q.shape[-1] // 2], q[:, q.shape[-1] // 2:]
        k1 = self.keys1.to(dev, dt)
        k2 = self.keys2.to(dev, dt)
        # 每半取 top-k（k 取 min，避免槽数不足）
        k = min(self.top_k, self.side)
        s1, i1 = torch.topk(q1 @ k1.T, k, dim=-1)
        s2, i2 = torch.topk(q2 @ k2.T, k, dim=-1)
        # 笛卡尔组合：候选槽 = i1*M + i2，得分相加
        cand_idx = (i1.unsqueeze(-1) * self.side + i2.unsqueeze(-2)).reshape(-1, k * k)
        cand_score = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).reshape(-1, k * k)
        kk = min(self.top_k, k * k)
        top_score, top_pos = torch.topk(cand_score, kk, dim=-1)
        sel = cand_idx.gather(-1, top_pos)  # (N, kk)，CPU/GPU 一致性由调用方保证
        w = F.softmax(top_score, dim=-1)
        return sel, w

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """残差式前向（value 全零时恒等）."""
        x = self.norm(h)
        sel, w = self._lookup(x)
        # 只搬 k 行上设备（稀疏的核心收益）
        v = self.values.to(x.device, x.dtype)[sel]  # (N, k, d)
        delta = (w.unsqueeze(-1) * v).sum(-2).view_as(h)
        if self.training and self.dropout > 0:
            delta = F.dropout(delta, p=self.dropout)
        return h + delta


def _kmeans_pp(
    data: torch.Tensor, n_clusters: int, iters: int = 10, seed: int = 0,
) -> torch.Tensor:
    """极简 k-means（CPU，B 方案初始化用；大数据请换 faiss）。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.shape[0], generator=g)[:n_clusters]
    centers = data[idx].clone()
    for _ in range(iters):
        dist = torch.cdist(data, centers, p=2)
        assign = dist.argmin(-1)
        for c in range(n_clusters):
            members = data[assign == c]
            if len(members):
                centers[c] = members.mean(0)
    return centers


def init_memory_from_activations(
    module: ProductKeyMemory,
    hiddens: torch.Tensor,
    seed: int = 0,
) -> dict:
    """B 方案初始化：hidden 劈半后各做 k-means（标准 PKM 做法），value 保持零.

    hiddens: (N, d) 校准集 hidden（backbone 冻结跑出来的）。
    返回 {"n_samples": N, "inertia": 两半平均类内距离} 供日志记录。
    零 value => 输出恒等，可单测锁定"初始化前后模型输出一致"。
    """
    with torch.no_grad():
        d = hiddens.shape[-1]
        h1, h2 = hiddens.float()[:, :d // 2], hiddens.float()[:, d // 2:]
        c1 = _kmeans_pp(h1, module.side, seed=seed)
        c2 = _kmeans_pp(h2, module.side, seed=seed + 1)
        module.keys1.copy_(c1.to(module.keys1.dtype))
        module.keys2.copy_(c2.to(module.keys2.dtype))
        module.values.zero_()
        d1 = torch.cdist(h1, c1, p=2).min(-1).values.mean()
        d2 = torch.cdist(h2, c2, p=2).min(-1).values.mean()
        inertia = float((d1 + d2) / 2)
    return {"n_samples": hiddens.shape[0], "inertia": round(inertia, 4)}
