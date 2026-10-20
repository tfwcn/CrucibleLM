"""长上下文（1M）单测 — YaRN NTK-by-parts / 阶段表 / 启动自检 / 1M 预设.

三条硬契约：
1. `yarn_scale=1` 时**逐位等于**标准 RoPE（老 checkpoint 不受影响）；
2. NTK-by-parts：高频维度不动（短距离分辨率）、低频维度 ÷s（长距离插值）、
   中间线性斜坡（无频率断点）；
3. 64K 起训必须有稀疏注意力（V4.1 不做 dense 预热）——配置违规要**启动即报错**。
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (SmallLLMConfig, longctx_1m_config,
                                  tiny_test_config)
from src.llm.local.longctx import (LONG_CTX_MIN, format_report, parse_ctx_stages,
                                   rope_cache_bytes, seq_len_at, sparse_ready,
                                   suggest_yarn, validate_longctx)
from src.llm.local.rope import (build_rope_cache, yarn_inv_freq, yarn_mscale,
                                yarn_scale_mismatch, yarn_scaling_factor)

DIM, HALF, THETA = 16, 8, 10000.0


def _std_inv(dim=DIM, theta=THETA):
    half = dim // 2
    return 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))


# ------------------------------- YaRN 频率 -------------------------------
def test_scale_one_is_bitwise_standard_rope():
    """scale=1：cos/sin 与标准 RoPE **逐位**相同（老 ckpt 兼容的硬契约）."""
    cos, sin = build_rope_cache(64, DIM, THETA, 1.0)
    f = torch.outer(torch.arange(64, dtype=torch.float32), _std_inv())
    emb = torch.cat([f, f], dim=-1)
    assert torch.equal(cos, emb.cos())
    assert torch.equal(sin, emb.sin())


def test_ntk_by_parts_high_freq_untouched_low_freq_interpolated():
    """高频维度不缩放、低频维度 ÷s（这是 YaRN 与"朴素 /s"的唯一区别）."""
    std = _std_inv()
    for s in (2.0, 8.0, 128.0):
        inv = yarn_inv_freq(DIM, THETA, s, 8192)
        assert torch.equal(inv[0], std[0]), "高频维度必须原样外推"
        assert torch.isclose(inv[-1] * s, std[-1], atol=1e-9), "低频维度必须插值"
        # 中间存在"既不是纯外推也不是纯插值"的斜坡维度
        mid = inv[1:-1]
        assert any(not torch.isclose(v / std[i + 1], torch.tensor(1.0), atol=1e-6)
                   and not torch.isclose(v * s / std[i + 1], torch.tensor(1.0),
                                         atol=1e-6)
                   for i, v in enumerate(mid))


def test_ntk_freq_monotone_and_ramp_continuous():
    """频率随维度单调不增（否则位置编码会出现"折返"）."""
    for s in (2.0, 4.0, 128.0):
        inv = yarn_inv_freq(DIM, THETA, s, 8192)
        assert (inv[1:] <= inv[:-1] + 1e-12).all()


def test_mscale():
    """幅度校正：s=1 → 1.0（不校正）；s>1 单调增，约 0.1·ln s + 1."""
    assert yarn_mscale(1.0) == 1.0
    assert yarn_mscale(0.5) == 1.0
    assert yarn_mscale(8.0) == pytest.approx(0.1 * math.log(8.0) + 1.0)
    assert yarn_mscale(128.0) > yarn_mscale(8.0)


def test_mscale_folded_into_cache():
    """mscale 折进 cos/sin（= q、k 各放大 √mscale → 内积放大 mscale）."""
    s = 8.0
    cos, sin = build_rope_cache(8, DIM, THETA, s)
    gain = math.sqrt(yarn_mscale(s))
    f = torch.outer(torch.arange(8, dtype=torch.float32),
                    yarn_inv_freq(DIM, THETA, s, 8192))
    emb = torch.cat([f, f], dim=-1)
    assert torch.allclose(cos, emb.cos() * gain, atol=1e-5)


def test_one_million_positions_stay_distinguishable():
    """1M 位置仍然"可分辨"：相邻位置的 cos 不塌成同一个值（外推没退化）."""
    cos, _ = build_rope_cache(1_048_576, 64, THETA, 128.0,
                              yarn_original_len=8192)
    assert (cos[-1] - cos[-2]).abs().max() > 1e-3
    # 且首尾不相等（不是"所有位置映射到同一点"）
    assert not torch.allclose(cos[0], cos[-1])


def test_rope_cache_shape_and_validation():
    """缓存形状与偶数维度校验."""
    cos, sin = build_rope_cache(32, DIM)
    assert cos.shape == sin.shape == (32, DIM)
    with pytest.raises(ValueError, match="偶数"):
        build_rope_cache(8, 7)


def test_yarn_scaling_factor_cap():
    """比例上界放宽到 512：4K→1M（256×）必须算得出来，旧上限 8 会直接失效."""
    assert yarn_scaling_factor(8192, 1048576) == pytest.approx(128.0)
    assert yarn_scaling_factor(4096, 1048576) == pytest.approx(256.0)
    assert yarn_scaling_factor(8192, 8192) == 1.0
    assert yarn_scaling_factor(8192, 1048576 * 8) == 512.0


def test_yarn_scale_mismatch():
    """比例对不上要能被自检抓到（8 倍 vs 需要的 128 倍）."""
    assert yarn_scale_mismatch(128.0, 8192, 1048576) is False
    assert yarn_scale_mismatch(8.0, 8192, 1048576) is True
    assert yarn_scale_mismatch(1.0, 8192, 65536) is False


# ------------------------------- 阶段表 -------------------------------
def test_parse_ctx_stages():
    """阶段表解析：排序、覆盖、非法项报错."""
    assert parse_ctx_stages("") == []
    assert parse_ctx_stages(" 0:65536 , 34000000000:1048576 ")[1] == (
        34_000_000_000, 1_048_576)
    assert parse_ctx_stages("100:2048,0:8192") == [(0, 8192), (100, 2048)]
    # 同起点后者覆盖
    assert parse_ctx_stages("0:8192,0:4096") == [(0, 4096)]
    with pytest.raises(ValueError, match="token数:序列长度"):
        parse_ctx_stages("65536")
    with pytest.raises(ValueError, match="整数"):
        parse_ctx_stages("0:abc")


def test_seq_len_at():
    """按已消费 token 取阶段长度；无阶段表时用默认值."""
    st = parse_ctx_stages("0:65536,34000000000:1048576")
    assert seq_len_at(0, st, 8192) == 65536
    assert seq_len_at(33_999_999_999, st, 8192) == 65536
    assert seq_len_at(34_000_000_000, st, 8192) == 1048576
    assert seq_len_at(123, [], 8192) == 8192


# ------------------------------- 自检 -------------------------------
def test_validate_ok_for_1m_preset():
    """1M 预设本身不违规（只对"没开 SWA"给一条警告）."""
    cfg = longctx_1m_config()
    warns = validate_longctx(cfg, 1048576)
    assert all("SWA" in w for w in warns)


def test_validate_rejects_dense_long_training():
    """64K 训练但把稀疏全关 = dense 预热，必须启动即报错."""
    cfg = tiny_test_config()
    cfg.max_seq_len = 131072
    cfg.sparse_threshold = 1 << 30      # 长序列也走 dense
    cfg.sparse_indexer = 0
    cfg.csa_modes = ""
    cfg.csa_reuse_every = 1
    cfg.swa_window = 0
    assert sparse_ready(cfg) is False
    with pytest.raises(ValueError, match="64K 稀疏起训|稀疏"):
        validate_longctx(cfg, LONG_CTX_MIN)
    # 短序列不受限
    assert validate_longctx(cfg, 4096) == []


def test_validate_rejects_seq_beyond_max():
    """训练长度超过 max_seq_len：位置缓存装不下."""
    cfg = tiny_test_config()
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_longctx(cfg, cfg.max_seq_len + 1)


def test_validate_warns_without_yarn_and_with_bad_scale():
    """长于训练长度却没开 YaRN / scale 对不上：都要给警告."""
    cfg = SmallLLMConfig(max_seq_len=1048576, yarn_scale=1.0,
                         yarn_original_len=8192)
    warns = "\n".join(validate_longctx(cfg, 65536))
    assert "yarn_scale=1" in warns
    cfg2 = SmallLLMConfig(max_seq_len=1048576, yarn_scale=8.0,
                          yarn_original_len=8192)
    assert "比例对不上" in "\n".join(validate_longctx(cfg2, 1048576))


def test_format_report_and_cache_bytes():
    """报告格式 + RoPE 缓存字节数（1M×64 约 512MB，属警告量级）."""
    cfg = longctx_1m_config()
    assert "长上下文自检" in format_report(cfg, 4096)
    assert "警告" in format_report(cfg, 1048576)
    assert rope_cache_bytes(1024, 64) == 2 * 1024 * 64 * 4
    assert rope_cache_bytes(1_048_576, 64) / 1024 ** 2 == 512.0


def test_suggest_yarn():
    """建议值可直接写入配置."""
    assert suggest_yarn(8192, 1048576) == {"yarn_original_len": 8192,
                                           "yarn_scale": 128.0}
    assert suggest_yarn(65536, 1048576)["yarn_scale"] == pytest.approx(16.0)


# ------------------------------- 模型接线 -------------------------------
def test_model_uses_extended_yarn_cache():
    """模型层：YaRN 参数透传到 MLA 的 cos/sin 缓存（且不随 scale 改变形状）."""
    from src.llm.local.model import TinyLLM

    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.yarn_scale, cfg.yarn_original_len = 8.0, 256
    m = TinyLLM(cfg)
    attn = next(l.attn for l in m.layers if hasattr(l.attn, "rope_cos"))
    assert attn.rope_cos.shape == (cfg.max_seq_len, cfg.qk_rope_dim)
    # mscale 已折进缓存：模长 > 1（标准 RoPE 的 cos/sin 模长为 1）
    assert float(attn.rope_cos.abs().max()) > 1.0


def test_1m_preset_fields():
    """1M 预设：max_seq_len / yarn_scale / original_len 三者自洽."""
    cfg = longctx_1m_config()
    assert cfg.max_seq_len == 1_048_576
    assert cfg.yarn_scale == 128.0
    assert cfg.yarn_original_len == 8192
    assert cfg.yarn_scale == pytest.approx(cfg.max_seq_len / cfg.yarn_original_len)
