"""MoE 可解释性分析单测 — SwiGLU 四符号区间统计 + 专家激活画像（手算校验）."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.interpret import (
    expert_activation_profile,
    expert_sign_regimes,
    regime_summary,
)
from src.llm.local.model import TinyLLM


def _tiny_eval():
    torch.manual_seed(0)
    return TinyLLM(tiny_test_config()).eval()


def test_regime_summary_hand_computed():
    """手算校验：2 神经元 × 3 token 覆盖 ++ 与 −− 两种正极来源."""
    gate = torch.tensor([[1.0, -1.0], [2.0, -3.0], [1.0, -1.0]])
    up = torch.tensor([[1.0, -1.0], [-2.0, -1.0], [1.0, -1.0]])
    mid = torch.nn.functional.silu(gate) * up
    dn = (gate < 0) & (up < 0)
    s = regime_summary(mid, dn, top_frac=1.0)
    # 神经元0 正激活全为 +×+（双负占比 0）；神经元1 全为 −×−（占比 1）→ 均值 0.5
    assert s["dn_share"] == pytest.approx(0.5)
    assert s["dn_top"] == pytest.approx(0.5)
    assert s["frac_argmax_dn"] == pytest.approx(0.5)
    assert s["n_neurons"] == 2


def test_regime_summary_skips_dead_units():
    """无非正激活的神经元（死单元）不计入，避免污染均值."""
    mid = torch.full((4, 3), -1.0)
    dn = torch.zeros(4, 3, dtype=torch.bool)
    s = regime_summary(mid, dn, top_frac=0.5)
    assert s["n_neurons"] == 0
    assert s["dn_share"] == 0.0


def test_expert_sign_regimes_keys_and_ranges():
    """逐专家统计覆盖全部专家，且占比都在 [0,1]."""
    m = _tiny_eval()
    batches = [torch.randint(0, 256, (2, 16)) for _ in range(2)]
    stats = expert_sign_regimes(m, batches)
    n_expected = m.config.n_layers * (m.config.n_experts + 1)
    assert len(stats) == n_expected
    for _key, s in stats.items():
        for k in ("dn_share", "dn_top", "frac_echo", "frac_argmax_dn"):
            assert 0.0 <= s[k] <= 1.0


def test_expert_activation_profile_shapes():
    """画像形状对齐专家隐层宽度，且 p99 ≤ peak、均为非负."""
    m = _tiny_eval()
    h = m.layers[0].moe.experts[0].w_gate.weight.shape[0]
    batches = [torch.randint(0, 256, (1, 16)) for _ in range(2)]
    prof = expert_activation_profile(m, batches)
    for v in prof.values():
        assert v["mean"].shape == (h,)
        assert bool((v["mean"] >= 0).all())
        assert bool((v["p99"] <= v["peak"] + 1e-6).all())
