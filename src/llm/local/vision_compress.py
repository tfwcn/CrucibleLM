"""视觉 token 压缩 — 把图像 patch 压成少量视觉 token 再送进主干（前瞻，默认关）.

动机：多模态里的图像 patch 数很多（如 224×224 / patch16 → 196 个），远超
文本侧的 token 需求；一次注意力前就要为它们付出 KV 与算力。先把 patch
编码 + 压缩到固定少量 query token（`vision_tokens`），再投影到 `d_model`，
即可显著缩短视觉前缀长度。

职责边界：本模块只管"图像 → 视觉 token 序列"这一段结构，
不含图像编码器预训练配方；默认关（`vision_enabled=False`）时模型里
根本不存在该模块，旧 checkpoint 逐位兼容（与 memory/retro 同风格）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """把图像切成不重叠 patch： (B,C,H,W) → (B, N, C*patch*patch)，N=(H/P)(W/P)."""
    if x.dim() != 4:
        raise ValueError(f"patchify 期望 (B,C,H,W)，实际 {tuple(x.shape)}")
    b, c, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError(f"H/W({h},{w}) 必须是 patch({patch}) 的整数倍")
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nh, nw, C, P, P)
    return x.reshape(b, (h // patch) * (w // patch), c * patch * patch)


def unpatchify(tokens: torch.Tensor, patch: int, channels: int,
               grid: tuple[int, int]) -> torch.Tensor:
    """patchify 的逆（测试用）： (B,N,C*P*P) → (B,C,H,W)."""
    nh, nw = grid
    b, n, _ = tokens.shape
    x = tokens.reshape(b, nh, nw, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B,C,nh,P,nw,P)
    return x.reshape(b, channels, nh * patch, nw * patch)


class PerceiverBlock(nn.Module):
    """单层 Perceiver 压缩块：可学习查询对 patch 做 cross-attention."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """(B,T,d) 查询聚合 (B,N,d) patch，返回 (B,T,d)."""
        q = self.norm_q(queries)
        kv = self.norm_kv(patches)
        attn, _ = self.cross(q, kv, kv, need_weights=False)
        queries = queries + attn
        queries = queries + self.ff(self.norm_ff(queries))
        return queries


class VisionCompressor(nn.Module):
    """图像 → 固定数量视觉 token，并投影到主干 d_model.

    流程：patchify → patch_embed（线性）→ 加位置嵌入 → 压缩到 n_tokens
    （avgpool 无参 / perceiver 可学习查询）→ 输出 (B, n_tokens, d_model)。
    """

    def __init__(
        self,
        d_model: int,
        in_channels: int = 3,
        patch: int = 16,
        n_tokens: int = 64,
        mode: str = "avgpool",
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mode not in ("avgpool", "perceiver"):
            raise ValueError(f"vision_compress 须为 avgpool/perceiver，当前 {mode}")
        self.mode = mode
        self.patch = int(patch)
        self.n_tokens = int(n_tokens)
        self.patch_embed = nn.Linear(in_channels * patch * patch, d_model)
        # 位置嵌入按最大 4096 patch 预留（足够 2048×2048/patch16=16384? 取 4096 覆盖常见尺寸）
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        if mode == "perceiver":
            self.queries = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
            nn.init.normal_(self.queries, std=0.02)
            self.perceiver = PerceiverBlock(d_model, n_heads, dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → (B, n_tokens, d_model)."""
        tokens = patchify(images, self.patch)  # (B,N,C*P*P)
        n = tokens.shape[1]
        if n > self.pos_embed.shape[1]:
            raise ValueError(
                f"patch 数 {n} 超过位置嵌入上限 {self.pos_embed.shape[1]}")
        h = self.patch_embed(tokens) + self.pos_embed[:, :n, :]
        if self.mode == "avgpool":
            # 自适应平均池化到固定 token 数： (B,d,N) → (B,d,T)
            pooled = F.adaptive_avg_pool1d(
                h.transpose(1, 2), self.n_tokens).transpose(1, 2)
            return self.out_norm(pooled)
        # perceiver：可学习查询 cross-attention 聚合
        b = h.shape[0]
        q = self.queries.expand(b, -1, -1)
        return self.out_norm(self.perceiver(q, h))
