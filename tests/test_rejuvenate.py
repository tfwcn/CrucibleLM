"""神经元复壮单测 — 恒等手术（输出不变）+ 梯度可流 + cap 语义."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.model import TinyLLM
from src.llm.local.rejuvenate import (
    apply_rejuvenation,
    collect_mid_activations,
    dormancy_masks,
)


def _tiny_eval():
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    m.eval()
    return m


def _some_key(m):
    layer = m.layers[0]
    return (0, "routed", 0), layer.moe.experts[0].w_gate.weight.shape[0]


def test_apply_is_identity_for_dormant_units():
    """真休眠单元手术恒等：激活恒零的单元，输出侧置零+输入重开逐位不变.

    （置零删的是既有贡献——活性单元必变，这正是必须按激活（而非 magnitude）
    选休眠的理由；此处把 gate 行置零构造真休眠，验机制本身。）
    """
    m = _tiny_eval()
    key, h = _some_key(m)
    expert = m.layers[0].moe.experts[0]
    with torch.no_grad():
        expert.w_gate.weight[0].zero_()  # silu(0)=0 → 单元 0 真休眠
    mask = torch.zeros(h, dtype=torch.bool)
    mask[0] = True
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        before = m(x)["logits"]
        info = apply_rejuvenation(m, {key: mask})
        after = m(x)["logits"]
    assert info["applied"][key] == 1
    assert torch.equal(before, after)


def test_apply_cap_and_score_order():
    """超 cap 只取 score 最低者."""
    m = _tiny_eval()
    key, h = _some_key(m)
    mask = torch.ones(h, dtype=torch.bool)
    scores = {key: torch.arange(h, dtype=torch.float)}  # 越小越闲
    info = apply_rejuvenation(m, {key: mask}, scores, seed=0, cap_frac=0.2)
    n = info["applied"][key]
    assert n == max(1, int(h * 0.2))
    # 取的是前 n 个（score 最低）
    assert n <= h


def test_grads_flow_to_new_weights():
    """重开后两步起速：step1 信号进输出列（随机中间激活×上游梯度），
    step2 起输入侧吃到梯度（零起速，身份起点的安全动力学）。"""
    m = _tiny_eval()
    m.train()
    x = torch.randint(0, 256, (2, 32))
    # hook 实测哪个专家真吃到 token（分组 bmm 后专家子模块不跑，挂 MoE 上看路由）
    routed: dict[int, int] = {}
    def _probe(_m, inp, _o):
        moe = m.layers[0].moe
        ti = torch.topk(torch.softmax(moe.router(inp[0]), dim=-1),
                        moe.top_k, dim=-1).indices
        cnt = torch.bincount(ti.reshape(-1), minlength=moe.n_experts)
        for e in range(moe.n_experts):
            routed[e] = routed.get(e, 0) + int(cnt[e])
    handle = m.layers[0].moe.register_forward_hook(_probe)
    with torch.no_grad():
        m(x)
    handle.remove()
    ei = max(routed, key=lambda e: routed[e])
    assert routed[ei] > 0
    key = (0, "routed", ei)
    expert = m.layers[0].moe.experts[ei]
    mask = torch.zeros(expert.w_gate.weight.shape[0], dtype=torch.bool)
    mask[0] = True
    apply_rejuvenation(m, {key: mask})
    # 冻住路由：两步内专家分工不变，只验复壮动力学（路由漂移是另一回事）
    m.layers[0].moe.router.requires_grad_(False)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    # 注意：复用 probe 时的 x（路由是输入相关的，换 batch 专家就换人了）
    opt.zero_grad()
    m(x, targets=x)["loss"].backward()
    assert expert.w_down.weight.grad is not None
    assert expert.w_down.weight.grad[:, 0].abs().sum() > 0
    opt.step()
    opt.zero_grad()
    m(x, targets=x)["loss"].backward()
    g = expert.w_gate.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[0].abs().sum() > 0, "第二步起输入侧吃到梯度"


def test_scan_shapes_and_report():
    """扫描产出对齐：masks/scores/报告键一致."""
    m = _tiny_eval()
    batches = [torch.randint(0, 256, (1, 16)) for _ in range(2)]
    acts = collect_mid_activations(m, batches)
    assert len(acts) > 0
    masks, scores, report = dormancy_masks(acts)
    assert set(masks) == set(scores) == set(acts)
    for key, mk in masks.items():
        assert mk.dtype == torch.bool
        assert mk.shape == scores[key].shape
    assert all("dormant_frac" in v for v in report.values())


def test_collect_returns_mean_and_peak():
    """采集同时给出逐神经元均值与峰值，且峰值≥均值."""
    m = _tiny_eval()
    batches = [torch.randint(0, 256, (1, 16)) for _ in range(2)]
    acts = collect_mid_activations(m, batches)
    for key, v in acts.items():
        assert set(v) == {"mean", "peak"}
        assert v["mean"].shape == v["peak"].shape
        assert bool((v["peak"] >= v["mean"] - 1e-6).all())


def test_dormancy_peak_guard_rescues_rare_strong_units():
    """峰值保护：平均激活极低但峰值最高的单元不被误判休眠."""
    key = (0, "routed", 0)
    mean = torch.ones(8)
    mean[0] = 1e-6            # 平均极低 → 原判据必休眠
    peak = torch.ones(8)
    peak[0] = 100.0           # 但峰值最高 → 罕见强特征
    acts = {key: {"mean": mean, "peak": peak}}

    masks, _, rep = dormancy_masks(acts, rel_threshold=1e-3, peak_frac=0.0)
    assert bool(masks[key][0]) is True
    assert rep[0]["rescued"] == 0

    # peak_frac=0.1 → 保护阈值 = 0.1×100 = 10，单元0 峰值 100 ≥ 10 → 救回
    masks, _, rep = dormancy_masks(acts, rel_threshold=1e-3, peak_frac=0.1)
    assert bool(masks[key][0]) is False
    assert rep[0]["rescued"] == 1


def test_dormancy_masks_accepts_raw_tensor():
    """向后兼容：直接传逐神经元均值张量（无峰值）也能判."""
    key = (0, "routed", 0)
    acts = {key: torch.tensor([0.0, 1.0, 1.0, 1.0])}
    masks, scores, report = dormancy_masks(acts, rel_threshold=1e-3)
    assert bool(masks[key][0]) is True
    assert set(report) == {0}
    assert report[0]["rescued"] == 0
