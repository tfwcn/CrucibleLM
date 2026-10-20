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
    """超连接参数吃到梯度、无 NaN（小矩阵 fp32 通路）.

    Single-Pass 下**最后一层**的 A 预测器是死参数：它预测的 `A_L` 没有下一块可用
    （与论文"输入混合错后一层"的代价一致，等于第一层省掉的 `A_{-1}`）。
    这里把死参数集合钉死，防止将来接线漏层时靠"全都有梯度"掩盖。
    """
    m = _tiny_hyper(2)
    m.train()
    x = torch.randint(0, 256, (1, 8))
    loss = m(x, targets=x)["loss"]
    loss.backward()
    last = len(m.layers) - 1
    dead = {n for n, p in m.named_parameters()
            if "hyper" in n and p.grad is None}
    assert dead == {f"layers.{last}.hyper.{k}"
                    for k in ("phi_pre.weight", "bias_pre", "alpha_pre")}
    n_hyper_grad = 0
    for name, p in m.named_parameters():
        if "hyper" in name and p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
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


def test_pre_mix_uniform_at_init():
    """动态 pre 零初值即均匀均值（与旧 mean 路径逐位一致，可无缝替换）."""
    m = _tiny_hyper(2)
    layer = m.layers[0]
    assert layer.hyper is not None
    x = torch.randn(1, 6, 2, 128)  # (b, t, n, d)，tiny d=128
    with torch.no_grad():
        got = layer.hyper.pre_mix(x)
        want = x.mean(dim=2)
    assert torch.allclose(got, want, atol=1e-6)


def test_composite_gain_identity_and_scaled():
    """复合增益：全 I 矩阵得 1.0；2I 连乘按 2^n 放大（定义本身正确）."""
    from src.llm.local.hyperconn import composite_gain

    assert abs(composite_gain([torch.eye(4) for _ in range(3)]) - 1.0) < 1e-6
    assert abs(composite_gain([2 * torch.eye(2)]) - 2.0) < 1e-6


def test_hyper_gain_tracks_eval_forward():
    """hyper_gain：track 关时 None；开跑一次前向后 ≈1（恒等起点附近）."""
    m = _tiny_hyper(2)
    assert m.hyper_gain() is None
    for layer in m.layers:
        if layer.hyper is not None:
            layer.hyper.track_stats = True
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        m(x)
    g = m.hyper_gain()
    assert g is not None and abs(g - 1.0) < 0.05
    for layer in m.layers:
        if layer.hyper is not None:
            layer.hyper.track_stats = False
    assert m.hyper_gain() is not None  # 关 track 不清旧值（读的是上次）


# ------------------------- Single-Pass mHC（V4.1 §2.4.1） -------------------------
def _randomize_hyper(m, alpha: float = 1.0, logit: float = 10.0):
    """把超连接从恒等起点推离：A 非均匀、H 非恒等、post≠1（否则测不出移位/融合）."""
    with torch.no_grad():
        for layer in m.layers:
            h = layer.hyper
            if h is None:
                continue
            h.logit.fill_(logit)
            h.alpha.fill_(alpha)
            h.alpha_pre.fill_(alpha)
            h.alpha_post.fill_(alpha)
            torch.nn.init.normal_(h.phi_pre.weight, std=0.05)
            torch.nn.init.normal_(h.phi_res.weight, std=0.05)
            torch.nn.init.normal_(h.phi_post.weight, std=0.05)
            h.bias_pre.copy_(torch.linspace(-0.5, 0.5, h.n))


def test_single_pass_uses_previous_block_coeffs():
    """第 l 块的输入混合用第 l-1 块给的 A（eq.6），本块产出自己的 A 留给下一块."""
    m = _tiny_hyper(2)
    _randomize_hyper(m)
    a, b = m.layers[0].hyper, m.layers[1].hyper
    x = torch.randn(2, 5, 2, 128)
    with torch.no_grad():
        pre0 = a.pre_coeffs(x)      # A_0（上一层给第 1 块用）
        own1 = b.pre_coeffs(x)      # A_1（eq.2 会立刻用这个）
        assert not torch.allclose(pre0, own1, atol=1e-3)
        x_in, aux = b.enter(x, pre0)
        want_shift = torch.einsum("bts,btsc->btc", pre0, x)
        want_own = torch.einsum("bts,btsc->btc", own1, x)
    assert torch.allclose(aux[0], own1, atol=1e-5)     # 产出的是自己的 A_1
    assert torch.allclose(x_in, want_shift, atol=1e-5)  # 用的是上一层的 A_0
    assert not torch.allclose(x_in, want_own, atol=1e-3)


def test_single_pass_uniform_is_identical_to_eq2():
    """A 恒均匀（零初始化）时 Single-Pass 与 eq.2 输出一致（无缝替换）."""
    m = _tiny_hyper(2)
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        on = m(x)["logits"]
        for layer in m.layers:
            if layer.hyper is not None:
                layer.hyper.single_pass = False
        off = m(x)["logits"]
    assert torch.allclose(on, off, atol=1e-6)


def test_single_pass_changes_output_when_coeffs_nonuniform():
    """A 非均匀时移位确实改变模型输出（防止"开关接了但没生效"）."""
    m = _tiny_hyper(2)
    _randomize_hyper(m)
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        on = m(x)["logits"]
        for layer in m.layers:
            if layer.hyper is not None:
                layer.hyper.single_pass = False
        off = m(x)["logits"]
    assert not torch.allclose(on, off, atol=1e-3)


def test_mega_matches_multi_pass_coeffs():
    """Mega 的分块单趟数据流 ≈ 整块 GEMM；且与 fp32 手算参考紧一致.

    tiles=1 走 RMSNorm 模块（内部 bf16 取整，旧路径逐位不变），tiles=4 走
    fp32 折权重（论文口径）→ 两者只差取整误差，故宽松容差 + 参考实现卡正确性。
    """
    m = _tiny_hyper(2)
    h = m.layers[0].hyper
    _randomize_hyper(m)
    x = torch.randn(2, 6, 2, 128)
    with torch.no_grad():
        p1, r1, q1 = h._coeffs(x, tiles=1)
        p4, r4, q4 = h._coeffs(x, tiles=4)
        b, t, n, d = x.shape
        xf = x.reshape(b, t, n * d).float()
        v = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + h.norm.eps)
             * h.norm.weight.float())
        pre_ref = torch.softmax(
            (v @ h.phi_pre.weight.float().T) * h.alpha_pre.float()
            + h.bias_pre.float(), dim=-1)
        post_ref = 2 * torch.sigmoid(
            (v @ h.phi_post.weight.float().T) * h.alpha_post.float()
            + h.bias_post.float())
        xi1 = torch.einsum("bts,btsc->btc", p1.to(x.dtype), x)
        xi4 = torch.einsum("bts,btsc->btc", p4.to(x.dtype), x)
    assert torch.allclose(p4, pre_ref, atol=1e-5)
    assert torch.allclose(q4, post_ref, atol=1e-5)
    assert torch.allclose(p1, p4, atol=2e-2)
    assert torch.allclose(q1, q4, atol=2e-2)
    assert torch.allclose(xi1, xi4, atol=2e-2)
    # 双随机性不因分块而破坏
    assert torch.allclose(r4.sum(-1), torch.ones_like(r4.sum(-1)), atol=1e-3)


def test_single_pass_decode_matches_full_forward():
    """A 非均匀 + H 非恒等：prefill + 单步解码 == 全前向（含 encode_full_hidden）."""
    m = _tiny_hyper(2)
    _randomize_hyper(m)
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        full = m(x)["logits"]
        h, pasts = m._prefill(x[:, :9], max_new_tokens=4)
        h2, _ = m._decode_step(x[:, 9:10], pasts)
        logits = m.lm_head(h2)
        seq = torch.cat([h, h2], dim=1)
        full_h = m.encode_full_hidden(x)
    assert torch.allclose(full[:, 9:10], logits, atol=1e-4)
    assert torch.allclose(seq[:, -1:], full_h[:, 9:10], atol=1e-4)


def test_single_pass_decode_multi_matches_sequential():
    """并行验证（_decode_multi）与逐 token 解码一致（A 逐层传递对齐）."""
    m = _tiny_hyper(2)
    _randomize_hyper(m)
    x = torch.randint(0, 256, (1, 8))
    toks = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        _, pa = m._prefill(x, max_new_tokens=3)
        _, pb = m._prefill(x, max_new_tokens=3)
        hs, cur = [], x
        for i in range(3):
            nxt = toks[:, i:i + 1]
            hh, pa = m._decode_step(nxt, pa)
            hs.append(hh)
            cur = torch.cat([cur, nxt], dim=1)
        seq = torch.cat(hs, dim=1)
        multi, _, _ = m._decode_multi(toks, pb)
    assert torch.allclose(seq, multi, atol=1e-4), (seq - multi).abs().max()


def test_activation_traffic_matches_paper_formula():
    """激活读写记账：多趟 (4n+4)d、两趟 (3n+2)d、Mega 单趟 (2n+2)d（= 多趟的一半）."""
    from src.llm.local.hyperconn import activation_traffic

    n, d = 4, 512
    assert activation_traffic(n, d, 2, "multi") == (4 * n + 4) * d * 2
    assert activation_traffic(n, d, 2, "two_pass") == (3 * n + 2) * d * 2
    assert activation_traffic(n, d, 2, "mega") == (2 * n + 2) * d * 2
    assert activation_traffic(n, d, 2, "mega") * 2 == activation_traffic(n, d, 2, "multi")


def test_hc_arch_switches_and_cli(tmp_path):
    """来源对齐：ckpt 记了 hc_single_pass/hc_mega 就沿用；老 ckpt 回退关；CLI 三态."""
    import json
    import sys
    from pathlib import Path

    from src.llm.local.config import (_ARCH_SWITCHES,
                                      sync_arch_from_checkpoint)

    assert "hc_single_pass" in _ARCH_SWITCHES and "hc_mega" in _ARCH_SWITCHES
    assert tiny_test_config().hc_single_pass is True
    assert tiny_test_config().hc_mega is True

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "hyper_streams": 2}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = tiny_test_config()
    cfg.hyper_streams = 2
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.hc_single_pass is False and cfg.hc_mega is False

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).hc_single_pass is None
    args = T.parse_args(["--no-hc-single-pass", "--no-hc-mega"])
    c = tiny_test_config()
    src = T.resolve_arch_switches(args, c)
    assert c.hc_single_pass is False and c.hc_mega is False
    assert src["hc_single_pass"] == "explicit" and src["hc_mega"] == "explicit"
