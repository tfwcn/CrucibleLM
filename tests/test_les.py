"""LES 专家调度单测 — 预测确定性 / 窗口 / 排除 / offload 数值不变（CPU 可跑）."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.les import LesPredictor, build_les
from src.llm.local.moe import FineGrainedMoE


def test_les_predict_empty_is_zero_score_order():
    """无历史时按专家号升序预测（确定性兜底）."""
    p = LesPredictor(5, history=4, topn=3)
    assert p.predict() == [0, 1, 2]


def test_les_predict_recent_weighted():
    """越近的命中权重越高：最新步命中的专家排前."""
    p = LesPredictor(5, history=4, topn=2)
    p.observe([3])
    p.observe([4])
    # 最新命中 4（权重 1）> 上一步 3（权重 1/2）
    assert p.predict() == [4, 3]


def test_les_predict_deterministic_and_exclude():
    """同输入同输出；exclude 里的专家不出现在结果中."""
    a, b = LesPredictor(6, 4, 3), LesPredictor(6, 4, 3)
    for obs in ([1, 2], [2, 3], [3, 4]):
        a.observe(obs)
        b.observe(obs)
    assert a.predict() == b.predict()
    out = a.predict(exclude=[a.predict()[0]])
    assert a.predict()[0] not in out


def test_les_history_window_evicts_old():
    """窗口滑出：更早的命中不再计入."""
    p = LesPredictor(4, history=2, topn=1)
    p.observe([0])
    p.observe([1])
    p.observe([2])  # 0 被滑出
    assert p.predict() == [2]


def test_les_observe_out_of_range_ignored():
    """越界专家号被忽略（防御，不崩）."""
    p = LesPredictor(3, 4, 2)
    p.observe([0, 99])
    assert p.predict() == [0, 1]


def test_les_invalid_n_experts():
    """非法专家数构建报错."""
    with pytest.raises(ValueError, match="n_experts"):
        LesPredictor(0)


def test_les_with_offload_does_not_change_output():
    """挂上 LES 预取后 offload 前向数值不变（只影响搬运时序）."""
    torch.manual_seed(0)
    m = FineGrainedMoE(64, n_experts=8, top_k=2, expert_hidden=16, n_shared=1)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        o_g, _ = m(x)
    m2 = FineGrainedMoE(64, n_experts=8, top_k=2, expert_hidden=16, n_shared=1)
    m2.load_state_dict(m.state_dict())
    m2.eval()
    m2.enable_offload("cpu", target="cpu")
    m2.enable_les(history=4, topn=3)
    with torch.no_grad():
        for _ in range(3):
            o_l, _ = m2(x)
    assert torch.allclose(o_g, o_l, atol=1e-5), (o_g - o_l).abs().max()


def test_build_les():
    """工厂函数构建正常."""
    p = build_les(8, history=4, topn=2)
    assert isinstance(p, LesPredictor) and p.n_experts == 8
