"""DeepSeek-ViT 视觉入路单测 — patch14 / 2D-RoPE / 双向 / 3×3 unshuffle / projector.

四条硬契约：
1. **原生分辨率**：位置只由 2D-RoPE 的相对位移决定（无绝对位置表 → 无分辨率上限）；
2. **token 数 ÷9**：1344×1344 / patch14 / 3×3 unshuffle → 96×96 网格 → 1024 token；
3. **双向注意力**：patch 之间互不可见是语言侧的事，视觉侧必须全能看见；
4. 默认关：`vision_enabled=False` 时模型无视觉参数，旧 checkpoint 逐位兼容。
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from src.llm.local.config import tiny_test_config
from src.llm.local.mla import RMSNorm
from src.llm.local.model import TinyLLM
from src.llm.local.vision_compress import (VisionCompressor, apply_rope2d,
                                           patchify, rope2d, unpatchify,
                                           vision_grid)


# ------------------------------- patchify / 网格口径 -------------------------------
def test_patchify_roundtrip():
    """patchify/unpatchify 往返逐位还原."""
    x = torch.randn(2, 3, 8, 12)
    p = patchify(x, 4)
    assert p.shape == (2, 6, 3 * 4 * 4)
    assert torch.equal(unpatchify(p, 4, 3, (2, 3)), x)


def test_patchify_invalid():
    """非整倍数尺寸/维数错误：报错（fail-fast）."""
    with pytest.raises(ValueError, match="整数倍"):
        patchify(torch.randn(1, 3, 10, 10), 4)
    with pytest.raises(ValueError, match="B,C,H,W"):
        patchify(torch.randn(3, 10, 10), 4)


def test_vision_grid_matches_paper():
    """论文口径：1344×1344 / patch14 / 3×3 → 1024 个视觉 token（÷9）."""
    nh, nw, n = vision_grid(1344, 1344, 14, 3)
    assert (nh, nw) == (96, 96)
    assert n == 1024
    # 不做 unshuffle 时就是 patch 数
    assert vision_grid(224, 224, 14, 1)[2] == 256
    # 336/14=24 → 24/3=8 → 8×8=64（整除才允许，避免 unshuffle 缺角）
    assert vision_grid(336, 336, 14, 3)[2] == 64
    # 224/14=16，16 不被 3 整除 → 明确报错而不是悄悄丢边
    with pytest.raises(ValueError, match="整数倍"):
        vision_grid(224, 224, 14, 3)


def test_vision_grid_divisibility_error():
    """H/W 必须被 patch×unshuffle 整除（否则 unshuffle 缺角）."""
    with pytest.raises(ValueError, match="整数倍"):
        vision_grid(1344, 1344, 14, 5)
    with pytest.raises(ValueError, match="patch/unshuffle"):
        vision_grid(64, 64, 14, 0)


# ------------------------------- 2D-RoPE -------------------------------
def test_rope2d_preserves_norm():
    """旋转不改模长（数值稳定的前提）."""
    cr, sr, cc, sc = rope2d(3, 4, 8)
    x = torch.randn(1, 2, 12, 8)
    y = apply_rope2d(x, cr, sr, cc, sc)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope2d_is_relative():
    """`q·k` 只依赖相对位移 (Δrow, Δcol)：同一位移在不同位置分数相同.

    这正是"原生分辨率"成立的原因：分辨率变了也不用重训位置表。
    """
    cr, sr, cc, sc = rope2d(3, 4, 8)
    qv, kv = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)

    def pair(i, j):
        qi = apply_rope2d(qv, cr[:, :, i:i + 1], sr[:, :, i:i + 1],
                          cc[:, :, i:i + 1], sc[:, :, i:i + 1])
        kj = apply_rope2d(kv, cr[:, :, j:j + 1], sr[:, :, j:j + 1],
                          cc[:, :, j:j + 1], sc[:, :, j:j + 1])
        return float((qi * kj).sum())

    # 网格 3×4（idx=r*4+c）：(0,1)→(1,2) 与 (1,0)→(2,1) 都是 Δ=(1,1)
    assert pair(1, 6) == pytest.approx(pair(4, 9), abs=1e-4)
    # 同列跨行 Δ=(0,1) 同样一致
    assert pair(0, 1) == pytest.approx(pair(4, 5), abs=1e-4)
    # 但不同位移必须不同（否则位置编码白给）
    assert abs(pair(0, 0) - pair(1, 6)) > 1e-3


def test_rope2d_requires_dim_multiple_of_4():
    """head_dim 须为 4 的倍数（行/列各占一半、每半按对旋转）."""
    with pytest.raises(ValueError, match="4 的倍数"):
        rope2d(2, 2, 6)


# ------------------------------- 结构 -------------------------------
def test_no_conv_and_uses_rmsnorm_swiglu():
    """线性 patch embedding（Muon 兼容）+ RMSNorm + SwiGLU（与主干同构）."""
    c = VisionCompressor(d_model=32, patch=4, unshuffle=1, d_vit=32,
                         vit_layers=1, vit_heads=4, projector_hidden=64)
    assert isinstance(c.patch_embed, nn.Linear)
    assert not any(isinstance(m, nn.Conv2d) for m in c.modules())
    assert isinstance(c.blocks[0].norm1, RMSNorm)
    assert isinstance(c.out_norm, RMSNorm)
    # SwiGLU：gate/up/down 三支
    assert hasattr(c.blocks[0], "gate") and hasattr(c.blocks[0], "up")


def test_projector_is_two_layers():
    """2 层 MLP projector（论文：hidden 5120 → 主干宽度）."""
    c = VisionCompressor(d_model=48, patch=4, unshuffle=1, d_vit=16,
                         vit_layers=0, projector_hidden=32)
    linears = [m for m in c.projector if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == [32, 48]


@pytest.mark.parametrize("n_tokens,mode", [(0, "avgpool"), (8, "avgpool"),
                                           (8, "perceiver")])
def test_output_shapes(n_tokens, mode):
    """输出固定形状 (B, n_tokens_eff, d_model)，数值有限；0=不二次压缩."""
    c = VisionCompressor(d_model=32, patch=4, unshuffle=2, d_vit=16,
                         vit_layers=2, vit_heads=4, projector_hidden=24,
                         n_tokens=n_tokens, mode=mode)
    c.eval()
    imgs = torch.randn(2, 3, 16, 16)          # 4×4 网格 → unshuffle2 → 2×2
    with torch.no_grad():
        out = c(imgs)
    assert out.shape == (2, n_tokens or 4, 32)
    assert torch.isfinite(out).all()


def test_unshuffle_reduces_tokens_in_forward():
    """前向 token 数确实按 r² 缩小（不是只在公式里）."""
    c = VisionCompressor(d_model=16, patch=4, unshuffle=3, d_vit=16,
                         vit_layers=0, projector_hidden=16)
    with torch.no_grad():
        out = c(torch.randn(1, 3, 24, 24))     # 6×6 → 2×2
    assert out.shape[1] == 4


def test_vision_is_bidirectional():
    """视觉注意力不建因果掩码：最后一个 patch 会影响第一个 patch 的输出."""
    torch.manual_seed(0)
    c = VisionCompressor(d_model=16, patch=8, unshuffle=1, d_vit=16,
                         vit_layers=1, vit_heads=2, projector_hidden=16)
    c.eval()
    x = torch.zeros(1, 3, 16, 16)              # 2×2 patch
    with torch.no_grad():
        a = c(x)
        # 只改右下角那个 patch 的像素
        x2 = x.clone()
        x2[:, :, 8:, 8:] = 3.0
        b = c(x2)
    assert not torch.allclose(a[:, 0], b[:, 0], atol=1e-6), \
        "因果掩码才会让 0 号 patch 不变；视觉必须双向"


def test_invalid_mode_raises():
    """非法压缩方式：构建即报错."""
    with pytest.raises(ValueError, match="avgpool/perceiver"):
        VisionCompressor(d_model=16, mode="magic")
    with pytest.raises(ValueError, match="vision_unshuffle"):
        VisionCompressor(d_model=16, unshuffle=0)
    with pytest.raises(ValueError, match="vision_tokens"):
        VisionCompressor(d_model=16, n_tokens=-1)


def test_forward_validation_and_grads():
    """尺寸不整除报错；反向梯度有限（可训练）."""
    c = VisionCompressor(d_model=16, patch=4, unshuffle=3, d_vit=16,
                         vit_layers=1, vit_heads=4, projector_hidden=16)
    with pytest.raises(ValueError, match="整数倍"):
        c(torch.randn(1, 3, 24, 20))
    with pytest.raises(ValueError, match="B,C,H,W"):
        c(torch.randn(3, 24, 24))
    out = c(torch.randn(1, 3, 24, 24))
    out.sum().backward()
    assert torch.isfinite(c.patch_embed.weight.grad).all()
    assert torch.isfinite(c.projector[0].weight.grad).all()


def test_perceiver_queries_trainable():
    """perceiver 查询可训练：反向梯度有限（无 NaN）."""
    c = VisionCompressor(d_model=32, patch=4, n_tokens=8, mode="perceiver",
                         n_heads=4, unshuffle=1, d_vit=16, vit_layers=1)
    c(torch.randn(2, 3, 16, 16)).sum().backward()
    assert c.queries.grad is not None
    assert torch.isfinite(c.queries.grad).all()


# ------------------------------- 模型接线 -------------------------------
def _vision_cfg(**kw):
    cfg = tiny_test_config()
    cfg.vision_enabled = True
    cfg.vision_tokens = kw.get("n_tokens", 0)
    cfg.vision_patch = 4
    cfg.vision_unshuffle = kw.get("unshuffle", 1)
    cfg.vision_d_vit = 16
    cfg.vision_vit_layers = kw.get("layers", 1)
    cfg.vision_vit_heads = 4
    cfg.vision_projector_hidden = 24
    return cfg


def test_default_off_no_module():
    """默认关：模型无视觉模块，state_dict 不含视觉键（旧 checkpoint 兼容）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    assert m.vision is None
    assert not any("vision" in k for k in m.state_dict())
    with pytest.raises(ValueError, match="vision_enabled=False"):
        m.encode_vision(torch.randn(1, 3, 8, 8))


def test_vision_prefix_attaches_and_loss_alignment():
    """视觉 token 拼在文本前：logits 长度 = 视觉 + 文本；CE 只取文本段."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    cfg = _vision_cfg(unshuffle=2, n_tokens=0)
    m = TinyLLM(cfg).eval()
    x = torch.randint(0, 256, (1, 6))
    imgs = torch.randn(1, 3, 8, 8)          # 2×2 网格 → unshuffle2 → 1 token
    with torch.no_grad():
        ve = m.encode_vision(imgs)
        out = m(x, targets=x, vision_embeds=ve, return_token_losses=True)
    v, t = ve.shape[1], x.shape[1]
    assert ve.shape == (1, 1, cfg.d_model)
    assert out["logits"].shape == (1, v + t, cfg.vocab_size)
    ref = F.cross_entropy(out["logits"][:, v:v + t - 1].reshape(-1, cfg.vocab_size),
                          x[:, 1:].reshape(-1))
    assert torch.allclose(out["main_loss"], ref, atol=1e-5)
    assert out["token_losses"].shape == (1, t - 1)


def test_no_vision_loss_unchanged():
    """无视觉前缀时主 loss 口径与旧式一致."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config()).eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = m(x, targets=x)
    ref = F.cross_entropy(out["logits"][:, :-1].reshape(-1, m.config.vocab_size),
                          x[:, 1:].reshape(-1))
    assert torch.allclose(out["main_loss"], ref, atol=1e-6)


def test_vision_backward_and_train_step():
    """视觉前缀可训练：编码器与投影都拿得到梯度；train_step 能更新."""
    from src.llm.local.train import build_optimizer, train_step

    torch.manual_seed(0)
    m = TinyLLM(_vision_cfg())
    x = torch.randint(0, 256, (1, 4))
    ve = m.vision(torch.randn(1, 3, 8, 8))
    ve.retain_grad()
    m(x, targets=x, vision_embeds=ve)["loss"].backward()
    assert torch.isfinite(m.vision.patch_embed.weight.grad).all()
    assert ve.grad is not None and torch.isfinite(ve.grad).all()

    opt = build_optimizer(m, lr=1e-3)
    before = m.vision.patch_embed.weight.detach().clone()
    info = train_step(m, opt, x, vision_images=torch.randn(1, 3, 8, 8))
    assert torch.isfinite(torch.tensor(info["loss"]))
    assert not torch.equal(before, m.vision.patch_embed.weight.detach())
