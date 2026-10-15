"""视觉 token 压缩单测 — patch 往返 / 压缩形状 / 默认关 / 前缀接入（CPU 可跑）."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.model import TinyLLM
from src.llm.local.vision_compress import (
    VisionCompressor,
    patchify,
    unpatchify,
)


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


@pytest.mark.parametrize("mode", ["avgpool", "perceiver"])
def test_compressor_output_shape(mode):
    """两种压缩方式都输出固定 token 数并投影到 d_model."""
    c = VisionCompressor(d_model=32, in_channels=3, patch=4, n_tokens=8, mode=mode)
    c.eval()
    imgs = torch.randn(2, 3, 16, 16)  # 16 patches
    with torch.no_grad():
        out = c(imgs)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_compressor_compress_ratio():
    """压缩率正确：patch 数 >> 输出 token 数."""
    c = VisionCompressor(d_model=16, patch=8, n_tokens=4, mode="avgpool")
    c.eval()
    imgs = torch.randn(1, 3, 128, 128)  # (128/8)^2 = 256 patch
    with torch.no_grad():
        out = c(imgs)
    assert out.shape == (1, 4, 16)


def test_compressor_too_many_patches_raises():
    """超位置嵌入上限：报错."""
    c = VisionCompressor(d_model=16, patch=1, n_tokens=4, mode="avgpool")
    c.eval()
    with pytest.raises(ValueError, match="位置嵌入上限"):
        c(torch.randn(1, 3, 80, 80))  # 6400 patch > 4096


def test_compressor_invalid_mode():
    """非法压缩方式：构建即报错."""
    with pytest.raises(ValueError, match="avgpool/perceiver"):
        VisionCompressor(d_model=16, mode="magic")


def test_vision_perceiver_grad_finite():
    """perceiver 查询可训练：反向梯度有限（无 NaN）."""
    c = VisionCompressor(d_model=32, patch=4, n_tokens=8, mode="perceiver")
    imgs = torch.randn(2, 3, 16, 16)
    out = c(imgs)
    out.sum().backward()
    assert c.queries.grad is not None
    assert torch.isfinite(c.queries.grad).all()


def test_vision_default_off_no_module():
    """默认关：模型无视觉模块，state_dict 不含视觉键（旧 checkpoint 兼容）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    assert m.vision is None
    assert not any("vision" in k for k in m.state_dict())
    with pytest.raises(ValueError, match="vision_enabled=False"):
        m.encode_vision(torch.randn(1, 3, 8, 8))


def test_vision_forward_prefix_attaches():
    """开视觉：视觉 token 拼在文本前，logits 长度 = 视觉 token 数 + 文本长."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.vision_enabled = True
    cfg.vision_tokens = 8
    cfg.vision_patch = 4
    m = TinyLLM(cfg)
    m.eval()
    imgs = torch.randn(1, 3, 16, 16)
    x = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        ve = m.encode_vision(imgs)
        out = m(x, vision_embeds=ve)
    assert ve.shape == (1, 8, cfg.d_model)
    assert out["logits"].shape == (1, 8 + 6, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_vision_train_loss_alignment():
    """视觉前缀 + targets：loss 只取文本段错位一位（视觉段不进 CE）."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.vision_enabled = True
    cfg.vision_tokens = 4
    cfg.vision_patch = 4
    m = TinyLLM(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 6))
    ve = m.encode_vision(torch.randn(1, 3, 8, 8))
    with torch.no_grad():
        out = m(x, targets=x, vision_embeds=ve, return_token_losses=True)
    v, t = ve.shape[1], x.shape[1]
    logits = out["logits"]
    # 参考：文本段 logits[v : v+T-1] 对齐 targets[:, 1:]
    ref = F.cross_entropy(
        logits[:, v:v + t - 1].reshape(-1, cfg.vocab_size),
        x[:, 1:].reshape(-1))
    assert torch.allclose(out["main_loss"], ref, atol=1e-5)
    assert torch.isfinite(out["loss"]).all()
    assert out["token_losses"].shape == (1, t - 1)


def test_no_vision_train_loss_unchanged():
    """无视觉前缀时主 loss 口径与旧式一致（logits[:, :-1] vs targets[:, 1:]）."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = m(x, targets=x)
    ref = F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, m.config.vocab_size),
        x[:, 1:].reshape(-1))
    assert torch.allclose(out["main_loss"], ref, atol=1e-6)


def test_vision_train_loss_backward_grads():
    """视觉前缀可训练：反向有梯度且有限（视觉前缀 + 编码器参数都拿得到）."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.vision_enabled = True
    cfg.vision_tokens = 4
    cfg.vision_patch = 4
    m = TinyLLM(cfg)
    m.train()
    x = torch.randint(0, 256, (1, 6))
    ve = m.vision(torch.randn(1, 3, 8, 8))  # 训练编码（带梯度）
    ve.retain_grad()
    out = m(x, targets=x, vision_embeds=ve)
    out["loss"].backward()
    assert m.vision.patch_embed.weight.grad is not None
    assert torch.isfinite(m.vision.patch_embed.weight.grad).all()
    assert ve.grad is not None and torch.isfinite(ve.grad).all()


def test_vision_train_step_accepts_vision_embeds():
    """train_step 透传视觉图像：一步训练 loss 有限、编码器参数更新."""
    from src.llm.local.train import build_optimizer, train_step

    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.vision_enabled = True
    cfg.vision_tokens = 4
    cfg.vision_patch = 4
    m = TinyLLM(cfg)
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 256, (1, 6))
    before = m.vision.patch_embed.weight.detach().clone()
    info = train_step(m, opt, x, vision_images=torch.randn(1, 3, 8, 8))
    assert torch.isfinite(torch.tensor(info["loss"]))
    assert not torch.equal(before, m.vision.patch_embed.weight.detach())
