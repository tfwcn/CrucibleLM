"""DeepSeek-ViT 视觉入路 — patch14 → ViT（2D-RoPE）→ 3×3 unshuffle → MLP projector.

对应 DeepSeek-V4.1 §2.1.1 的多模态入路，四个设计点各有原因：

1. **原生分辨率 + 2D-RoPE**：不用可学习绝对位置表（那会钉死最大分辨率），
   行/列各占一半维度做旋转 —— 任意 H×W 都能编，位置关系只由相对 (Δrow, Δcol) 决定。
2. **线性 patch embedding**（不是卷积）：卷积核在 Muon 下不好做正交化更新；
   `patchify + Linear` 等价于无重叠卷积，且能进 Muon。
3. **RMSNorm + SwiGLU**：与主干同构，减少"视觉塔 / 语言塔"的实现分叉。
4. **3×3 pixel-unshuffle（作用在**特征网格**上）**：把相邻 3×3 个 patch 特征
   沿通道拼起来，token 数直接 ÷9 —— 论文用它把 1344×1344 的图压到
   96×96/9 = 1024 个视觉 token；随后 2 层 MLP projector 投到主干宽度。

为什么 unshuffle 放在 ViT **之后**而不是像素上：先让 ViT 在细粒度 patch 上
看清局部（14×14 的细节），再按 3×3 邻域合并；在像素上先合并等于把有效 patch
变大到 42×42，细节在进网络前就没了。

职责边界：本模块只管"图像 → 视觉 token 序列"，不含图像编码器预训练配方。
默认关（`vision_enabled=False`）时模型里根本没有该模块，旧 checkpoint 逐位兼容。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import RMSNorm


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


def vision_grid(h: int, w: int, patch: int, unshuffle: int = 1
                ) -> tuple[int, int, int]:
    """返回 (网格行数, 网格列数, 视觉 token 数)：`ntok = nh*nw//r²`.

    单独抽出来是为了**不跑前向就能算 token 数**（显存规划/单测都用得上）。
    H/W 必须能被 `patch × unshuffle` 整除（否则 patchify 或 unshuffle 会缺角）。
    """
    if int(patch) < 1 or int(unshuffle) < 1:
        raise ValueError(f"patch/unshuffle 须 ≥1，当前 {patch}/{unshuffle}")
    step = int(patch) * int(unshuffle)
    if h % step or w % step:
        raise ValueError(
            f"H/W({h},{w}) 必须是 patch×unshuffle({step}) 的整数倍")
    nh, nw = h // int(patch), w // int(patch)
    return nh, nw, (nh // int(unshuffle)) * (nw // int(unshuffle))


# ---------------------------------------------------------------- 2D-RoPE
def rope2d(nh: int, nw: int, head_dim: int, device=None,
           dtype: torch.dtype = torch.float32,
           theta: float = 10000.0) -> tuple[torch.Tensor, ...]:
    """2D-RoPE 的 cos/sin：`(N, head_dim//4)` 四件套（行/列各一对）.

    约定：`head_dim` 的**前一半**维度按行坐标旋转、后一半按列坐标旋转；每一半内部
    按相邻对 `(2i, 2i+1)` 当 (实, 虚) 旋转。故要求 `head_dim % 4 == 0`。

    这样 `q·k` 只依赖相对位移 `(Δrow, Δcol)` —— 分辨率变了位置语义也不变，
    所以"原生分辨率"成立（不用重训位置表）。
    """
    if head_dim % 4:
        raise ValueError(f"2D-RoPE 要求 head_dim 是 4 的倍数，当前 {head_dim}")
    q = head_dim // 4  # 每一半里的“旋转对数”
    inv = 1.0 / (theta ** (torch.arange(q, dtype=torch.float32,
                                        device=device) / max(q, 1)))
    rows = torch.arange(nh, device=device, dtype=torch.float32
                        ).repeat_interleave(nw)
    cols = torch.arange(nw, device=device, dtype=torch.float32).repeat(nh)
    fr = torch.outer(rows, inv)   # (N, q)
    fc = torch.outer(cols, inv)
    # (N, q) → (1, 1, N, q) 便于对 (B,H,N,q) 广播
    return tuple(t.view(1, 1, -1, q) for t in
                 (fr.cos().to(dtype), fr.sin().to(dtype),
                  fc.cos().to(dtype), fc.sin().to(dtype)))


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """把最后一维的相邻对 `(a,b)` 旋转成 `(a·cos − b·sin, a·sin + b·cos)`."""
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return out.flatten(-2)


def apply_rope2d(x: torch.Tensor, cos_r: torch.Tensor, sin_r: torch.Tensor,
                 cos_c: torch.Tensor, sin_c: torch.Tensor) -> torch.Tensor:
    """对 `(B, N, H, head_dim)` 做 2D-RoPE（前一半行、后一半列）."""
    half = x.shape[-1] // 2
    return torch.cat([
        _rotate_pairs(x[..., :half], cos_r, sin_r),
        _rotate_pairs(x[..., half:], cos_c, sin_c),
    ], dim=-1)


# ---------------------------------------------------------------- ViT 层
class ViTBlock(nn.Module):
    """DeepSeek-ViT 一层：pre-RMSNorm + **双向**自注意力（2D-RoPE）+ SwiGLU.

    视觉与语言的关键差别：patch 之间**没有因果**（图是同时可见的），
    所以不建因果掩码。这也是它不能直接复用语言侧 `HybridBlock` 的原因。
    """

    def __init__(self, d_vit: int, n_heads: int, ffn_hidden: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_vit % n_heads == 0, "d_vit 须被头数整除"
        self.n_heads = int(n_heads)
        self.head_dim = d_vit // n_heads
        assert self.head_dim % 4 == 0, "2D-RoPE 要求 head_dim 是 4 的倍数"
        self.norm1 = RMSNorm(d_vit)
        self.q_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.k_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.v_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.o_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.norm2 = RMSNorm(d_vit)
        self.gate = nn.Linear(d_vit, ffn_hidden, bias=False)
        self.up = nn.Linear(d_vit, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_vit, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cos_r, sin_r, cos_c, sin_c
                ) -> torch.Tensor:
        """x: (B, N, d_vit)；返回同形（双向注意力，N 个 patch 互相可见）."""
        b, n, _ = x.shape
        h = self.norm1(x)
        shape = (b, n, self.n_heads, self.head_dim)
        q = self.q_proj(h).view(*shape).transpose(1, 2)   # (B,H,N,hd)
        k = self.k_proj(h).view(*shape).transpose(1, 2)
        v = self.v_proj(h).view(*shape).transpose(1, 2)
        q = apply_rope2d(q, cos_r, sin_r, cos_c, sin_c)
        k = apply_rope2d(k, cos_r, sin_r, cos_c, sin_c)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        ctx = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, -1)
        x = x + self.o_proj(ctx)
        h = self.norm2(x)
        return x + self.down(self.act(self.gate(h)) * self.up(h))


class PerceiverBlock(nn.Module):
    """单层 Perceiver 压缩块：可学习查询对 patch 做 cross-attention（可选压缩用）."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = RMSNorm(d_model)
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


# ---------------------------------------------------------------- 完整编码器
class VisionCompressor(nn.Module):
    """图像 → 视觉 token 序列（DeepSeek-ViT 精简版 + 可选二次压缩）.

    流程（V4.1 §2.1.1）：

    ```
    (B,C,H,W) → patchify(14) → Linear → 2D-RoPE → ViT × L
              → 3×3 特征 unshuffle（token ÷9，通道 ×9）
              → MLP projector（2 层）→ d_model
              → [可选] 压到 n_tokens（avgpool / perceiver）
    ```

    `n_tokens=0`（默认）表示**不再二次压缩**，与论文一致（1344×1344 → 1024 token）；
    设为 >0 时再用池化/查询压到固定长度，供小模型节省前缀开销。
    """

    def __init__(
        self,
        d_model: int,
        in_channels: int = 3,
        patch: int = 14,
        n_tokens: int = 0,
        mode: str = "avgpool",
        n_heads: int = 4,
        dropout: float = 0.0,
        unshuffle: int = 3,
        d_vit: int = 256,
        vit_layers: int = 12,
        vit_heads: int = 4,
        projector_hidden: int = 512,
        vit_ffn_mult: int = 4,
    ):
        super().__init__()
        if mode not in ("avgpool", "perceiver"):
            raise ValueError(f"vision_compress 须为 avgpool/perceiver，当前 {mode}")
        if int(unshuffle) < 1:
            raise ValueError(
                f"vision_unshuffle 须 ≥1（1=不做特征重排），当前 {unshuffle}")
        if int(n_tokens) < 0:
            raise ValueError(f"vision_tokens 须 ≥0（0=不二次压缩），当前 {n_tokens}")
        self.mode = mode
        self.patch = int(patch)
        self.unshuffle = int(unshuffle)
        self.n_tokens = int(n_tokens)
        self.d_vit = int(d_vit)
        self.vit_layers = int(vit_layers)
        # 线性 patch embedding（Muon 兼容；等价于无重叠卷积）
        self.patch_embed = nn.Linear(in_channels * self.patch * self.patch,
                                     self.d_vit, bias=False)
        # 双向 ViT 主干（norm 在残差内，故块外不再加 norm）
        self.blocks = nn.ModuleList([
            ViTBlock(self.d_vit, vit_heads, vit_ffn_mult * self.d_vit, dropout)
            for _ in range(self.vit_layers)
        ])
        # 特征重排后通道 = r²·d_vit → 2 层 MLP projector → 主干宽度
        self.projector = nn.Sequential(
            nn.Linear(self.unshuffle ** 2 * self.d_vit, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, int(d_model)),
        )
        if self.n_tokens > 0 and mode == "perceiver":
            self.queries = nn.Parameter(torch.zeros(1, self.n_tokens, int(d_model)))
            nn.init.normal_(self.queries, std=0.02)
            self.perceiver = PerceiverBlock(int(d_model), n_heads, dropout)
        self.out_norm = RMSNorm(int(d_model))

    # -------------------------------------------------------------- 前向
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → (B, n_tokens_eff, d_model)."""
        if images.dim() != 4:
            raise ValueError(f"期望 (B,C,H,W)，实际 {tuple(images.shape)}")
        _, _, h, w = images.shape
        nh, nw, n_tok = vision_grid(h, w, self.patch, self.unshuffle)
        tokens = patchify(images, self.patch)             # (B, nh*nw, C*P²)
        x = self.patch_embed(tokens)                      # (B, N, d_vit)
        # 2D-RoPE：只用相对位移，故分辨率可变
        rope = rope2d(nh, nw, self.blocks[0].head_dim if self.blocks else 4,
                      device=x.device, dtype=x.dtype) if self.blocks else None
        if rope is not None:
            for blk in self.blocks:
                x = blk(x, *rope)
        # 3×3 特征 unshuffle：相邻 3×3 patch 沿通道拼接 → token 数 ÷9
        x = self._grid_unshuffle(x, nh, nw, self.unshuffle)
        x = self.projector(x)                             # (B, n_tok, d_model)
        if self.n_tokens > 0 and self.n_tokens != n_tok:
            if self.mode == "perceiver":
                q = self.queries.expand(x.shape[0], -1, -1)
                x = self.perceiver(q, x)
            else:
                x = F.adaptive_avg_pool1d(
                    x.transpose(1, 2), self.n_tokens).transpose(1, 2)
        return self.out_norm(x)

    @staticmethod
    def _grid_unshuffle(x: torch.Tensor, nh: int, nw: int,
                        r: int) -> torch.Tensor:
        """特征网格上的 unshuffle：`(B,nh·nw,d)` → `(B,(nh/r)(nw/r), r²·d)`.

        顺序约定：先按"块内 (i,j) 偏移"再按通道拼接（与 `F.pixel_unshuffle`
        的通道顺序一致），保证 patch 内部结构是可学习的连续维度。
        """
        if r == 1:
            return x
        if nh % r or nw % r:
            raise ValueError(f"网格 ({nh},{nw}) 不能按 {r} 整除")
        b, _, d = x.shape
        x = x.view(b, nh // r, r, nw // r, r, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, nh/r, nw/r, r, r, d)
        return x.view(b, (nh // r) * (nw // r), r * r * d)
