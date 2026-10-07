"""超连接残差单测 — 双随机性 + 恒等起点 + 流分化 + 解码一致 + 兼容."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.hyperconn import HyperConnRes, sinkhorn
from src.llm.local.model import TinyLLM


def _tiny_hyper(n: int = 2):
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.hyper_streams = n
    m = TinyLLM(cfg).eval()
    return m


def test_sinkhorn_doubly_stochastic():
    """Sinkhorn 输出行列和≈1、非负（Birkhoff 多面体）.

    输入用真实量级（动态项经 α 缩放后小幅；randn×2 的极端比值 20 轮收不紧，
    那不是我们的工况——论文 Fig.7 的 1.6 界也是同理的近似界）。
    """
    torch.manual_seed(0)
    h = sinkhorn(torch.randn(2, 5, 4, 4) * 0.5)
    assert torch.allclose(h.sum(-1), torch.ones(2, 5, 4), atol=1e-3)
    assert torch.allclose(h.sum(-2), torch.ones(2, 5, 4), atol=1e-3)
    assert bool((h >= 0).all())


def test_hyperconn_rejects_single_stream():
    """n=1 退化恒等，直接拒（用 0 关，别占坑）。"""
    with pytest.raises(AssertionError):
        HyperConnRes(64, 1)


def test_identity_at_init():
    """恒等起点：超连接模型载入基线权重后输出逐位一致（流全同 + g≈0 + post≈1）."""
    torch.manual_seed(0)
    m0 = TinyLLM(tiny_test_config()).eval()
    m1 = _tiny_hyper(2)
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and missing and all("hyper" in k for k in missing)
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        o0 = m0(x)["logits"]
        o1 = m1(x, )["logits"]
    assert torch.allclose(o0, o1, atol=1e-4)


def test_streams_differentiate():
    """流分化存活：动态 post 打开后各路输出不同（打破对称，否则恒为恒等白开销）."""
    m = _tiny_hyper(2)
    with torch.no_grad():
        for layer in m.layers:
            if layer.hyper is not None:
                layer.hyper.logit.fill_(10.0)
                layer.hyper.alpha_post.fill_(1.0)  # 动态写回激活（否则 post 恒 1）
    x = torch.randint(0, 256, (1, 8))
    layer = m.layers[0]
    h = torch.randn(1, 8, 128)
    xs = h.unsqueeze(2).expand(-1, -1, 2, -1)
    with torch.no_grad():
        h_in = xs.mean(2)
        a_out, _ = layer.attn(layer.norm1(h_in), None, False)
        f = h_in + a_out
        y = layer.hyper.combine(xs, f - h_in)
    assert not torch.allclose(y[:, :, 0], y[:, :, 1], atol=1e-6)


def test_hyper_gradients_finite():
    """超连接参数吃到梯度、无 NaN（小矩阵 fp32 通路）。"""
    m = _tiny_hyper(2)
    m.train()
    x = torch.randint(0, 256, (1, 8))
    loss = m(x, targets=x)["loss"]
    loss.backward()
    n_hyper_grad = 0
    for name, p in m.named_parameters():
        if "hyper" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            n_hyper_grad += 1
    assert n_hyper_grad > 0


def test_hyper_decode_matches_full_forward():
    """解码一致：prefill+单步解码 == 全前向（_decode_step 流路径同步）。"""
    m = _tiny_hyper(2)
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        full = m(x)["logits"]
        h, pasts = m._prefill(x[:, :9], max_new_tokens=4)
        h2, _ = m._decode_step(x[:, 9:10], pasts)
        step_logits = m.lm_head(h2)
    assert torch.allclose(full[:, 9:10], step_logits, atol=1e-5)


def test_hyper_state_dict_compatible():
    """关闭时无 hyper 键；开启后仅 hyper 键新增（旧权重 overlap 可载）."""
    torch.manual_seed(0)
    m0 = TinyLLM(tiny_test_config())
    assert not any("hyper" in k for k in m0.state_dict())
    m1 = _tiny_hyper(2)
    keys1 = set(m1.state_dict())
    assert any("hyper" in k for k in keys1 - set(m0.state_dict()))
    n_hyper_layers = sum(getattr(layer, "hyper", None) is not None
                         for layer in m1.layers)
    assert n_hyper_layers == len(m1.layers)
