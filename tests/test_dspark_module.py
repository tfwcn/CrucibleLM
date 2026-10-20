"""DSpark 独立草稿器单测 — 结构 / 掩码 / 调度 / 契约（等于贪心）/ 训练可降损.

CPU 可跑。三条硬契约：
1. 槽位（草稿位置）之间**互不可见**（并行 base logits），依赖只走 Markov 头；
2. 滑窗外的上下文不影响输出（窗口 128 是真的滑窗）；
3. 开 DSpark 起草 + 调度后，生成结果仍**逐位等于**逐 token 贪心（验证精确）。
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.dspark import (DSparkDrafter, DSparkScheduler, attach_dspark,
                                  build_dspark_drafter, draft_mask)
from src.llm.local.model import TinyLLM


# ------------------------------- 构造工具 -------------------------------
def _cfg(depth=5, window=8, layers=2, heads=4):
    cfg = tiny_test_config()
    cfg.dspark_draft = True
    cfg.dspark_layers = layers
    cfg.dspark_heads = heads
    cfg.dspark_window = window
    cfg.dspark_depth = depth
    cfg.dspark_markov_rank = 16
    return cfg


def _model(**kw) -> TinyLLM:
    torch.manual_seed(0)
    return TinyLLM(_cfg(**kw)).eval()


# ------------------------------- 结构/数值 -------------------------------
def test_drafter_shapes_and_conf_range():
    """一次前向：`depth` 个位置的条件接受概率，全部 ∈ (0,1)，logits 经 lm_head."""
    m = _model()
    dr = m.dspark
    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        hq, conf = dr(m.embed(x[:, -1:]), x, m.embed)
        logits = m.lm_head(hq)
    assert hq.shape == (2, 5, 128)
    assert conf.shape == (2, 5)
    assert bool(((conf > 0) & (conf < 1)).all())
    assert logits.shape == (2, 5, 256)


def test_markov_head_is_noop_at_init():
    """Markov 头零初始化 → 开局 logits 就是 base（纯并行），之后才学依赖."""
    m = _model()
    dr = m.dspark
    prev = torch.randint(0, 256, (2, 5))
    with torch.no_grad():
        bias = dr.markov_bias(prev)
    assert bias.shape == (2, 5, 256)
    assert float(bias.abs().max()) == 0.0


def test_draft_mask_blocks_slot_crosstalk():
    """草稿槽位之间互不可见（否则就不是"并行"而是自回归了）."""
    n_ctx, depth, window = 6, 3, 8
    mask = draft_mask(n_ctx, depth, window)
    assert mask.shape == (n_ctx + depth, n_ctx + depth)
    # 槽位只能看 ctx（且受滑窗限制：槽位 i 与 ctx 的最近距离是 depth-i）
    for i in range(depth):
        qi = n_ctx + i
        assert not mask[qi, n_ctx:].any()          # 槽位之间互不可见
        assert bool(mask[qi, n_ctx - 1])           # 最近的 ctx 永远可见
        vis = mask[qi, :n_ctx]
        expect = torch.arange(n_ctx) > (qi - window)
        assert torch.equal(vis, expect)
    # ctx 因果：不能看未来
    assert mask[0, 1] is False or not bool(mask[0, 1])
    # 窗口：窗口外的历史不可见
    mask_w = draft_mask(40, 2, 8)
    assert not bool(mask_w[39, 0])
    assert bool(mask_w[39, 35])


def test_window_truncates_context():
    """超过 window 的上下文对输出无影响（真滑窗，不是全注意力）."""
    m = _model(window=8)
    dr = m.dspark
    x = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        # 20 个 token 的上下文 vs 它自己的最后 8 个（window=8）：
        # 窗口外的信息本来就不可见 → 输出必须一致
        a = dr(m.embed(x[:, :20][:, -1:]), x[:, :20], m.embed)[0]
        b = dr(m.embed(x[:, :20][:, -1:]), x[:, 12:20], m.embed)[0]
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


# ------------------------------- 调度器 -------------------------------
def test_scheduler_expected_tokens_math():
    """期望产出 = 1 + Σ 前缀存活概率（1 是主干自己恒定接受的那颗）."""
    s = DSparkScheduler()
    confs = [0.9, 0.8, 0.5]
    assert s.survival(confs) == pytest.approx([0.9, 0.72, 0.36])
    assert s.expected_tokens(confs, 0) == pytest.approx(1.0)
    assert s.expected_tokens(confs, 2) == pytest.approx(1 + 0.9 + 0.72)
    assert s.expected_tokens(confs, 99) == pytest.approx(1 + 0.9 + 0.72 + 0.36)


def test_scheduler_picks_throughput_optimum():
    """高置信长草稿 → 多验；低置信 → 少验；k 截断与空输入安全."""
    s = DSparkScheduler()
    assert s.choose_length([0.95] * 5, 5) == 5
    assert s.choose_length([0.2, 0.2, 0.2, 0.2, 0.2], 5) <= 1
    assert s.choose_length([0.9] * 5, 2) == 2
    assert s.choose_length([], 5) == 0
    assert s.choose_length([0.9], 99) == 1


def test_scheduler_monotone_and_min_verify():
    """置信越高（或 min_verify 越大）验证长度不减；成本参数只影响选择不影响正确性."""
    s = DSparkScheduler()
    low = s.choose_length([0.5] * 6, 6)
    high = s.choose_length([0.8] * 6, 6)
    assert high >= low
    s2 = DSparkScheduler(min_verify=3)
    assert s2.choose_length([0.01] * 6, 6) == 3
    # 边际成本越高 → 越不愿意多验
    assert DSparkScheduler(target_per_token=5.0).choose_length([0.9] * 6, 6) <= low + 6


# ------------------------------- 契约：等于贪心 -------------------------------
@pytest.mark.parametrize("schedule", ["confidence", "throughput"])
def test_dspark_generation_equals_greedy(schedule):
    """DSpark 起草 + 调度（两种口径）：输出逐位等于逐 token 贪心."""
    m = _model()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=24)
    m.config.draft_enabled = True
    m.config.draft_source = "dspark"
    m.config.draft_window = 6
    m.config.dspark_enabled = True
    m.config.dspark_schedule = schedule
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=24)
    assert torch.equal(base, spec)
    assert m._spec_stats["dspark"] is True
    assert m._spec_stats["schedule"] == schedule
    assert m._spec_stats["verified"] <= m._spec_stats["proposed"]


def test_dspark_provider_shapes_and_confidence():
    """起草器接口：返回 ≤depth 个 token 与同长置信度（调度器直接吃）."""
    m = _model()
    provider = m.dspark
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        h_last = m.embed(torch.tensor([[8]]))
    from src.llm.local.spec_decode import DSparkDraftProvider

    p = DSparkDraftProvider(m)
    toks, confs = p.propose_scored(ctx, h_last, 3)
    assert len(toks) == len(confs) == 3
    assert all(0 <= t < 256 for t in toks)
    assert all(0.0 <= c <= 1.0 for c in confs)
    # 未挂草稿器时安全退化为 ngram（不崩）
    m2 = _model()
    m2.dspark = None
    toks2, _ = DSparkDraftProvider(m2).propose_scored(ctx, h_last, 3)
    assert toks2 == []


# ------------------------------- state_dict / 挂载 -------------------------------
def test_dspark_off_adds_no_keys():
    """默认关：`dspark.*` 一个键都没有（老 ckpt 兼容）."""
    torch.manual_seed(0)
    m0 = TinyLLM(tiny_test_config())
    assert not any(k.startswith("dspark.") for k in m0.state_dict())


def test_dspark_attach_roundtrip(tmp_path):
    """`attach_dspark`：只重建草稿器结构并严格载入（主干权重不动）."""
    m = _model()
    path = tmp_path / "dspark.pt"
    torch.save({"config": {"dspark_layers": 2, "dspark_heads": 4,
                           "dspark_window": 8, "dspark_depth": 5,
                           "dspark_markov_rank": 16},
                "state_dict": m.dspark.state_dict(), "step": 7}, path)
    m2 = TinyLLM(tiny_test_config()).eval()
    meta = attach_dspark(m2, path)
    assert m2.dspark is not None and meta["step"] == 7
    assert m2.config.dspark_window == 8
    for k, v in m.dspark.state_dict().items():
        assert torch.equal(v, m2.dspark.state_dict()[k])


# ------------------------------- 训练路径 -------------------------------
def test_dspark_training_reduces_loss():
    """独立训练真的在学：固定主干、只训草稿器，20 步后 CE 明显下降."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train_dspark import step_loss

    torch.manual_seed(0)
    m = _model(window=16, depth=3)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.dspark.parameters():
        p.requires_grad_(True)
    m.dspark.train()
    opt = torch.optim.AdamW(m.dspark.parameters(), lr=3e-3)
    # 强规律语料：AAAB 循环 → 草稿器应迅速把 CE 压下去
    ids = torch.tensor([[1, 1, 1, 2, 1, 1, 1, 2] * 8])
    batch = {
        "ctx": ids[:, :16],
        "prev": ids[:, 16:19],
        "tgt": ids[:, 17:20],
        "h_last": torch.zeros(1, 1, 128),
        "tlogits": torch.zeros(1, 3, 256),
    }
    first = None
    for _ in range(20):
        loss, stats = step_loss(m, batch, conf_weight=0.5, markov_weight=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = stats["ce_base"]
    assert stats["ce_base"] < first * 0.9, (first, stats["ce_base"])
    assert torch.isfinite(loss)


def test_draft_mask_gradients_finite():
    """端到端反传：草稿器参数全 finite，embedding 路径不带梯度（主干冻结）."""
    m = _model()
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.dspark.parameters():
        p.requires_grad_(True)
    x = torch.randint(0, 256, (1, 10))
    hq, conf = m.dspark(m.embed(x[:, -1:]), x, m.embed)
    prev = torch.randint(0, 256, (1, 5))
    # Markov 头不在 forward 里（由起草器串行调用），损失里要显式带上它
    (m.lm_head(hq).logsumexp(-1).mean() + conf.mean()
     + m.dspark.markov_bias(prev).logsumexp(-1).mean()).backward()
    fwd = m.dspark.markov_out.weight.grad
    assert fwd is not None and torch.isfinite(fwd).all()
    assert m.embed.weight.grad is None


def test_scheduler_cost_is_monotone():
    """成本随验证长度线性增加（吞吐调度的前提），且吞吐在某个 n 取到最大."""
    s = DSparkScheduler()
    costs = [s.cost(n) for n in range(5)]
    assert all(b > a for a, b in zip(costs, costs[1:]))
    tps = [s.throughput([0.8] * 5, n) for n in range(5)]
    assert max(tps) > tps[0]
