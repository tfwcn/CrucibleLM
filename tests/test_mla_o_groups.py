"""MLA 分组输出投影单测 — o_groups / o_lora_rank（DeepSeek-V4.1）.

CPU 可跑。核心契约：
- 默认关（o_groups<=1 或 o_lora_rank<=0）：键与数值与旧实现逐位一致（只认 `w_o.weight`）；
- 开启：按 n_heads 分组，各组头输出降维到 o_lora_rank，拼接后升维回 d_model，
  等价于"逐组降维→各组升维到 d_model→相加"（把求和并进矩阵乘法）；
- 数学：`_out_proj` 与朴素逐组和一致；r = 每组输入维时与稠密 `w_o` 精确等价；
- 因果/Finite：改未来输入不影响前部；无 NaN 梯度；
- 解码：prefill + 增量解码 == 全前向；
- 迁移：`group_output_projection` 近似无损（err 随 r 增大而减小），键由 w_o 变 w_o_d/w_o_u；
- 来源对齐：`sync_arch_from_checkpoint` 读 o_groups/o_lora_rank，旧 ckpt 回退 1/0；
  n_heads 不整除 o_groups 直接报错。
"""

import json

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.migrate import group_output_projection  # noqa: E402
from src.llm.local.mla import MLAModule  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402


# ------------------------------- 默认关/键兼容 -------------------------------
def test_off_is_legacy_keys():
    """默认关：state_dict 只认 w_o.weight（键与旧 checkpoint 完全一致）."""
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256)
    assert m.grouped_o is False
    keys = list(m.state_dict())
    assert "w_o.weight" in keys
    assert not any("w_o_d" in k or "w_o_u" in k for k in keys)


def test_on_changes_keys_loudly():
    """开启：不再有 w_o.weight，改为 w_o_d/w_o_u（旧 ckpt 严格载入会报缺键，不静默）."""
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256, o_groups=2, o_lora_rank=16)
    assert m.grouped_o is True
    keys = list(m.state_dict())
    assert "w_o.weight" not in keys
    assert "w_o_u.weight" in keys
    assert "w_o_d.0.weight" in keys and "w_o_d.1.weight" in keys


def test_config_default_off_and_wiring():
    """config 默认关；开启后逐层接线到 MLAModule."""
    assert SmallLLMConfig().o_groups == 1
    assert SmallLLMConfig().o_lora_rank == 0
    cfg = tiny_test_config()
    cfg.o_groups = 2
    cfg.o_lora_rank = 32
    m = TinyLLM(cfg)
    # tiny：full_attn_every=2 → 第 0/2 层是全注意力（分组输出），其余线性注意力
    assert m.layers[0].attn.grouped_o is True
    assert getattr(m.layers[1].attn, "grouped_o", False) is False
    assert m.layers[0].attn.o_groups == 2
    assert m.layers[0].attn.o_lora_rank == 32


def test_incompatible_groups_raises():
    """n_heads 不能被 o_groups 整除时构造即报错（早失败）."""
    with pytest.raises(AssertionError):
        MLAModule(128, 4, 32, 32, 16, max_seq_len=256, o_groups=3, o_lora_rank=16)
    with pytest.raises(AssertionError):
        MLAModule(128, 4, 32, 32, 16, max_seq_len=256, o_groups=2, o_lora_rank=999)


# ------------------------------- 数学正确 -------------------------------
def test_out_proj_matches_naive_group_sum():
    """`_out_proj` == 朴素"逐组降维 → 各组升维到 d_model → 相加"."""
    torch.manual_seed(0)
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256, o_groups=2, o_lora_rank=16)
    m.eval()
    x = torch.randn(2, 5, 4 * 32)
    with torch.no_grad():
        y = m._out_proj(x)
    in_g = m.heads_per_group * m.head_dim
    g = x.view(2, 5, 2, in_g)
    acc = torch.zeros(2, 5, 128)
    for i in range(2):
        up = m.w_o_u.weight[:, i * 16:(i + 1) * 16]  # (d_model, r)
        acc = acc + m.w_o_d[i](g[:, :, i]) @ up.T
    assert torch.allclose(y, acc, atol=1e-5)


def test_r_equals_input_is_exact_dense():
    """o_lora_rank = 每组输入维（不降维）时，分组输出可精确表示稠密 w_o（err=0）."""
    torch.manual_seed(1)
    cfg = tiny_test_config()
    dense = TinyLLM(cfg)
    dense.eval()
    in_g = (cfg.n_heads // 2) * (cfg.d_model // cfg.n_heads)  # o_groups=2 时每组输入维
    grouped, err = group_output_projection(dense, 2, in_g)
    assert err == 0.0
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        a = dense(x)["logits"]
        b = grouped(x)["logits"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_grouped_model_is_causal_and_finite():
    """分组模型：改未来 token 不影响前部；前向反向无 NaN."""
    torch.manual_seed(2)
    cfg = tiny_test_config()
    cfg.o_groups = 2
    cfg.o_lora_rank = 32
    m = TinyLLM(cfg)
    m.train()
    ids = torch.randint(0, 256, (1, 12))
    ids2 = ids.clone()
    ids2[0, 9] = (ids2[0, 9] + 7) % 256
    m.eval()
    with torch.no_grad():
        a = m(ids)["logits"]
        b = m(ids2)["logits"]
    assert (a[0, :9] - b[0, :9]).abs().max() < 1e-4, "未来 token 污染了前部（因果破坏）"
    assert (a[0, 9:] - b[0, 9:]).abs().max() > 0, "同/后续位置应有响应（测试本身有效）"
    m.train()
    out = m(ids, targets=ids)
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_grouped_decode_matches_full():
    """分组 MLA：prefill + 增量解码 == 一次全前向对应段."""
    torch.manual_seed(3)
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256, o_groups=2, o_lora_rank=16)
    m.eval()
    h = torch.randn(1, 16, 128)
    with torch.no_grad():
        full, _ = m(h)
        _, past = m(h[:, :8], None, True, 16)
        multi, _ = m(h[:, 8:16], past)
    assert torch.allclose(multi, full[:, 8:], atol=1e-5), (multi - full[:, 8:]).abs().max()


# ------------------------------- 迁移 -------------------------------
def test_migration_err_shrinks_with_rank():
    """迁移误差随 o_lora_rank 增大而减小；开启后键变为 w_o_d/w_o_u."""
    torch.manual_seed(4)
    dense = TinyLLM(tiny_test_config())
    dense.eval()
    _, err_small = group_output_projection(dense, 2, 8)
    _, err_big = group_output_projection(dense, 2, 48)
    assert err_small > err_big
    grouped, _ = group_output_projection(dense, 2, 32)
    keys = list(grouped.state_dict())
    assert "layers.0.attn.w_o_u.weight" in keys
    assert "layers.0.attn.w_o.weight" not in keys  # 全注意力层：w_o 已被分组替换
    # 线性注意力层（无分组输出）的 w_o 键保留
    assert "layers.1.attn.w_o.weight" in keys


def test_migration_guards():
    """非法迁移参数直接报错."""
    dense = TinyLLM(tiny_test_config())
    with pytest.raises(ValueError):
        group_output_projection(dense, 1, 32)
    with pytest.raises(ValueError):
        group_output_projection(dense, 2, 0)


# ------------------------------- 来源对齐 -------------------------------
def test_sync_reads_o_switches(tmp_path):
    """装载既有权重时对齐 o_groups/o_lora_rank；旧格式回退 1/0（关）."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(
        json.dumps({"config": {"o_groups": 4, "o_lora_rank": 192}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.o_groups == 4 and cfg.o_lora_rank == 192

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 768}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.o_groups == 1 and cfg2.o_lora_rank == 0
