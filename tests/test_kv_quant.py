"""KV 缓存量化单测 — `kv_quant`（DeepSeek-V4.1 §2.4.4：FP4 主 KV + FP8 SWA KV）.

核心契约：
- 格式：`fp4` = MXFP4（E2M1 码字 15 个值 + 每 16 通道一个 E4M3 scale，**不带**
  NVFP4 的第二级全局 scale）；`fp8` = E4M3 + 每 16 通道 scale；
- 量化误差有界：最近邻码字 → 每元素误差 ≤ 半个量化步长（同一 block 内）；
- 打包无损：4-bit 码字 ↔ uint8 双通道打包/解包逐位一致；
- `QuantBuffer` 存储真省（fp4 ≈ 4.5 bit/通道、fp8 ≈ 8.5，对照 bf16 的 16）；
- QAT 用 STE：前向是量化值，反向梯度**直通**（恒等）；
- 模型侧：`kv_quant` 非空时 prefill + 增量解码与全前向**逐位一致**（两条路径都走
  同一个 `_kv_latent` QAT 出口，读缓存时反量化）；`kv_cache_bytes` 按量化格式计；
  关闭时逐位等价旧行为且不加 `state_dict` 键；因果 + Finite。
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig, sync_arch_from_checkpoint  # noqa: E402
from src.llm.local.kv_quant import (  # noqa: E402
    E2M1_VALUES,
    QuantBuffer,
    dequantize_e2m1,
    fake_quant,
    kv_bytes_per_token,
    pack_4bit,
    quantize_e2m1,
    unpack_4bit,
)
from src.llm.local.model import TinyLLM  # noqa: E402


def _cfg(**over) -> SmallLLMConfig:
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=1,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=128, yarn_scale=1.0,
        sparse_threshold=4, sparse_window=4, sparse_sink=2,
        sparse_stride=2, sparse_chunk=4,
        sparse_indexer=False, csa_reuse_every=1, grad_ckpt=False,
    )
    base.update(over)
    return SmallLLMConfig(**base)


# ------------------------------- 码字与误差 -------------------------------
def test_e2m1_codebook_is_ocp_mxfp4():
    """E2M1 的 15 个可表示值：±{0.5,1,1.5,2,3,4,6} 与 0（无独立 -0）."""
    vals = sorted(set(abs(float(v)) for v in E2M1_VALUES))
    assert vals == sorted([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def test_fp4_quant_error_within_half_gap_per_code():
    """每个元素的误差 ≤ 它落点码字的**半个局部间距**（E2M1 码字不等距）.

    E2M1 的码字间距随指数变化（0.5 / 0.5 / 0.5 / 0.5 / 1 / 1 / 2），所以"半步长"
    不是常数——用全局 step/2 当界会误判（实测 6.0 附近允许到 1.0 单位）。
    这里按反量化后的码字查表得到该码字的允许误差（= 相邻间距的一半里较大的那个）。
    """
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 3.0
    codes, scale = quantize_e2m1(x, block=16)
    xq = dequantize_e2m1(codes, scale, block=16)
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # 每个（无符号）码字 k 的最大允许误差（单位：scale）
    gap_lo = torch.cat([torch.zeros(1), mag[1:] - mag[:-1]])   # 左邻间距
    gap_hi = torch.cat([mag[1:] - mag[:-1], torch.zeros(1)])   # 右邻间距
    allow = torch.maximum(gap_lo, gap_hi) / 2
    step = scale.float().repeat_interleave(16, dim=-1)
    code_mag = codes.long() % 8                                 # 幅值码字
    err_unit = (x - xq).abs() / step.clamp(min=1e-8)
    allow_e = allow.to(err_unit.device)[code_mag]
    assert float((err_unit - allow_e).max()) <= 1e-4, (
        float((err_unit - allow_e).max()))
    # 绝对误差也有上界：最大码字间隔 2 * scale
    assert float((x - xq).abs().max()) <= float(step.max()) + 1e-5


def test_pack_unpack_is_lossless():
    """4-bit 码字 → uint8 打包 → 解包，逐位一致（偶通道高位、奇通道低位）."""
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (4, 32), dtype=torch.uint8)
    packed = pack_4bit(codes)
    assert packed.shape == (4, 16)
    assert torch.equal(unpack_4bit(packed), codes)


def test_quant_buffer_roundtrip_write_and_resize():
    """QuantBuffer：往返 == quant-dequant；`write_` 只动目标行；`resize` 保全旧内容."""
    torch.manual_seed(0)
    x = torch.randn(2, 20, 32)
    buf = QuantBuffer.from_tensor(x, "fp4")
    codes, scale = quantize_e2m1(x, 16)
    ref = dequantize_e2m1(codes, scale, 16)
    assert torch.allclose(buf.dequant(), ref, atol=0.0)
    # write_ 只改 [5, 7)
    new = torch.randn(2, 2, 32) * 5
    before = buf.dequant().clone()
    buf.write_(5, new)
    after = buf.dequant()
    assert torch.allclose(after[:, :5], before[:, :5], atol=0.0)
    assert torch.allclose(after[:, 7:], before[:, 7:], atol=0.0)
    c2, s2 = quantize_e2m1(new, 16)
    assert torch.allclose(after[:, 5:7], dequantize_e2m1(c2, s2, 16), atol=0.0)
    # resize 保留旧内容
    grown = buf.resize(40)
    assert grown.shape == (2, 40, 32)
    assert torch.allclose(grown.dequant()[:, :20], after, atol=0.0)
    assert float(grown.dequant()[:, 20:].abs().max()) == 0.0


def test_quant_buffer_storage_is_smaller_than_bf16():
    """fp4 每通道 4.5 bit、fp8 8.5 bit；相对 bf16 分别是 ~0.28 / ~0.53 倍."""
    per_fp4 = kv_bytes_per_token(128, "fp4")
    per_fp8 = kv_bytes_per_token(128, "fp8")
    assert abs(per_fp4 - 128 * 4.5 / 8) < 1e-6
    assert abs(per_fp8 - 128 * 8.5 / 8) < 1e-6
    assert per_fp4 / (128 * 2) < 0.29
    x = torch.randn(1, 16, 128)
    buf = QuantBuffer.from_tensor(x, "fp4")
    assert buf.nbytes() < x.numel() * 2 * 0.30


def test_fake_quant_has_straight_through_gradient():
    """STE：`fake_quant` 前向是量化值，反向梯度恒等（全 1）."""
    x = torch.randn(4, 32, requires_grad=True)
    y = fake_quant(x, "fp4")
    assert not torch.equal(y, x)          # 前向确实被量化
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


# ------------------------------- 模型侧 -------------------------------
def test_kv_quant_off_is_bitwise_identical_and_adds_no_keys():
    """`kv_quant=""`（默认）：键集合不变、输出逐位相同."""
    torch.manual_seed(0)
    a = TinyLLM(_cfg())
    torch.manual_seed(0)
    b = TinyLLM(_cfg(kv_quant=""))
    assert set(a.state_dict()) == set(b.state_dict())
    ids = torch.randint(0, 256, (1, 12))
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.equal(a.encode_full_hidden(ids), b.encode_full_hidden(ids))


def test_kv_quant_cache_is_quantized_and_shrinks():
    """开启后缓存是 QuantBuffer、`kv_cache_bytes` 显著下降（同一模型同 ctx）."""
    torch.manual_seed(0)
    base = TinyLLM(_cfg())
    m = TinyLLM(_cfg(kv_quant="fp4"))
    m.eval()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=8)
    assert isinstance(pasts[0][0], QuantBuffer)
    shrink = (m.kv_cache_bytes(8192, 2)
              / base.kv_cache_bytes(8192, 2))
    print(f"[kv_quant] fp4 缓存相对 bf16 = {shrink:.3f}")
    assert shrink < 0.35, shrink


@pytest.mark.parametrize("fmt", ["fp4", "fp8"])
def test_kv_quant_prefill_plus_decode_matches_full_forward(fmt):
    """量化开启后 `prefill + 增量解码 == 一次全前向`（两条路径同一 QAT 出口）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(kv_quant=fmt))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        pre, pasts = m._prefill(ids[:, :16], max_new_tokens=8)
        tail, _ = m._decode_step(ids[:, 16:20], pasts)
    assert torch.allclose(pre, full[:, :16], atol=1e-5), (
        float((pre - full[:, :16]).abs().max()))
    assert torch.allclose(tail, full[:, 16:20], atol=1e-5), (
        float((tail - full[:, 16:20]).abs().max()))


def test_kv_quant_causal_and_finite():
    """量化开启后：改未来 token 不影响前部；梯度 finite（STE 不产 NaN）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(kv_quant="fp4"))
    ids = torch.randint(0, 256, (1, 20))
    ids2 = ids.clone()
    ids2[0, 16] = (ids2[0, 16] + 7) % 256
    m.eval()
    with torch.no_grad():
        a = m.encode_full_hidden(ids)
        b = m.encode_full_hidden(ids2)
    assert (a[0, :16] - b[0, :16]).abs().max() < 1e-6
    m.train()
    m(ids, targets=ids)["loss"].backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{name}"


def test_kv_quant_session_cache_roundtrip(tmp_path):
    """量化缓存的会话落盘/读回（落盘直接存打包张量，读回可继续解码）."""
    from src.llm.local.session_cache import SessionCache

    torch.manual_seed(0)
    m = TinyLLM(_cfg(kv_quant="fp4", swa_window=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    sc = SessionCache(m)
    with torch.no_grad():
        sc.extend(ids)
        path = str(tmp_path / "q.pt")
        sc.save(path, drop_swa=False)
        back = SessionCache.load(path, m)
        h = back.extend(torch.randint(0, 256, (1, 2)))
    assert torch.isfinite(h).all()
    assert len(back) == 26


def test_kv_quant_rejects_bad_format_and_indivisible_dims():
    """非法格式 / 维度不能被 16 整除 → 早失败."""
    with pytest.raises(ValueError, match="未知 kv_quant"):
        TinyLLM(_cfg(kv_quant="fp6"))
    with pytest.raises(ValueError, match="能被块 16 整除"):
        TinyLLM(_cfg(kv_quant="fp4", qk_rope_dim=12))


def test_kv_quant_checkpoint_alignment_and_cli(tmp_path):
    """来源对齐：ckpt 记了 `kv_quant` 就沿用；老 ckpt 回退空；CLI 三态."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "kv_quant": "fp4"}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = _cfg()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.kv_quant == "fp4"

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(json.dumps({"config": {
        "d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = _cfg()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.kv_quant == ""

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).kv_quant is None
    args = T.parse_args(["--kv-quant", "fp8"])
    c = _cfg()
    src = T.resolve_arch_switches(args, c)
    assert c.kv_quant == "fp8" and src["kv_quant"] == "explicit"


def test_zero_rows_do_not_become_nan():
    """全零行（如恒等起点的零权重 / 全零 latent）：scale 不能下溢成 0.

    `amax/6` 或 `amax/448` 为 0 时，scale 的下界若取 1e-8，向下舍入到 E4M3 就是 **0**
    → `0/0 = NaN`。下界必须取 E4M3 的最小正值（2^-9）。见 pitfalls 51。
    """
    z = torch.zeros(2, 3, 32)
    for fmt in ("fp8", "fp4"):
        q = fake_quant(z, fmt)
        assert torch.isfinite(q).all(), fmt
        assert float(q.abs().max()) == 0.0
    # 极小值走同一路径：量化到 0（E4M3 次正规以下），但**不能**是 NaN/inf
    tiny = torch.full((2, 32), 1e-9)
    for fmt in ("fp8", "fp4"):
        assert torch.isfinite(fake_quant(tiny, fmt)).all(), fmt
    # QuantBuffer 的零填充槽读出来也必须是有限值
    buf = QuantBuffer.zeros((1, 4, 32), "fp4")
    assert torch.isfinite(buf.dequant()).all()
    assert float(buf.dequant().abs().max()) == 0.0
