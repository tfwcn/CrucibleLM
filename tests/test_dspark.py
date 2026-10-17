"""DSpark 置信调度投机解码单测 — 调度函数 / 置信度 / 默认关 / 等于贪心 / 截断.

CPU 可跑。契约：DSpark 只改"每轮验多长"，不改最终序列——
开与不开都必须逐位等于逐 token 贪心（验证仍精确）。
"""

import pytest

torch = pytest.importorskip("torch")

import src.llm.local.spec_decode as spec_mod
from src.llm.local.config import tiny_test_config
from src.llm.local.model import TinyLLM
from src.llm.local.spec_decode import (
    MtpDraftProvider,
    NgramDraftProvider,
    schedule_verify_length,
    scored_propose,
)


def _tiny():
    torch.manual_seed(0)
    return TinyLLM(tiny_test_config())


def _spec_cfg(source="hybrid", window=5, dspark=False, threshold=0.5):
    cfg = tiny_test_config()
    cfg.draft_enabled = True
    cfg.draft_source = source
    cfg.draft_window = window
    cfg.dspark_enabled = dspark
    cfg.dspark_conf_threshold = threshold
    return cfg


# --------------------------- 调度函数（纯逻辑） ---------------------------
def test_schedule_threshold_zero_verifies_all():
    """threshold<=0 退化为全验证（= 固定窗口，默认关的等价路径）."""
    assert schedule_verify_length([0.1, 0.1, 0.1], 0.0) == 3
    assert schedule_verify_length([0.1, 0.1, 0.1], -1.0) == 3


def test_schedule_cumulative_product():
    """按累计存活概率截断：0.9*0.9=0.81≥0.8，再乘 0.9=0.729<0.8 → 验 2 个."""
    assert schedule_verify_length([0.9, 0.9, 0.9], 0.8) == 2
    assert schedule_verify_length([0.9, 0.9, 0.9], 0.5) == 3


def test_schedule_low_confidence_truncates_to_min():
    """首个草稿置信就不达标 → 截到 min_verify（默认 1，保底推进）."""
    assert schedule_verify_length([0.1, 0.99, 0.99], 0.5) == 1
    assert schedule_verify_length([0.1, 0.99, 0.99], 0.5, min_verify=2) == 2


def test_schedule_clamps_and_empty():
    """k 截断、空输入、min_verify 超界都安全."""
    assert schedule_verify_length([0.9] * 5, 0.5, k=3) == 3
    assert schedule_verify_length([], 0.5) == 0
    assert schedule_verify_length([0.9], 0.5, min_verify=99) == 1


# --------------------------- 置信度接口 ---------------------------
def test_scored_propose_fallback_for_legacy_provider():
    """老起草器（只有 propose）回退为恒定置信度 1.0（等价不做调度）."""

    class _Legacy:
        def propose(self, context, h_last, k):
            return [1, 2][:max(0, k)]

    toks, confs = scored_propose(_Legacy(), [0], None, 2)
    assert toks == [1, 2] and confs == [1.0, 1.0]


def test_ngram_propose_scored_confidence():
    """ngram 置信度 = m/(m+1)，命中越长越可信，长度与 token 对齐."""
    p = NgramDraftProvider(min_match=2)
    toks, confs = p.propose_scored([1, 2, 3, 1, 2], None, 3)
    assert toks == [3, 1, 2]
    assert len(confs) == len(toks)
    assert all(0.0 < c < 1.0 for c in confs)
    # 无命中 → 空
    assert p.propose_scored([9, 8, 7], None, 3) == ([], [])


def test_mtp_propose_scored_confidence():
    """MTP 起草：1 个候选 + softmax 最大概率作置信度."""
    m = _tiny()
    p = MtpDraftProvider(m)
    toks, confs = p.propose_scored([1, 2, 3], torch.randn(1, 1, m.config.d_model), 4)
    assert len(toks) == 1 and len(confs) == 1
    assert 0.0 <= confs[0] <= 1.0


# --------------------------- 契约：等于贪心 ---------------------------
@pytest.mark.parametrize("source", ["ngram", "mtp", "hybrid"])
def test_dspark_matches_greedy(source):
    """各起草来源开 DSpark：输出仍逐位等于逐 token 贪心."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=32)
    m.config.draft_enabled = True
    m.config.draft_source = source
    m.config.draft_window = 5
    m.config.dspark_enabled = True
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=32)
    assert torch.equal(base, spec)
    assert m._spec_stats["dspark"] is True


def test_dspark_default_off_stats():
    """默认关：`dspark` 标记为 False，且不产生 proposed/verified 字段."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    m.config.draft_enabled = True
    m.config.draft_source = "hybrid"
    with torch.no_grad():
        m.generate(x, max_new_tokens=16)
    assert m._spec_stats["dspark"] is False
    assert "proposed" not in m._spec_stats


def test_dspark_requires_draft_enabled():
    """只开 DSpark 不开 draft：不进投机路径（_spec_eligible 仍要求 draft_enabled）."""
    m = _tiny()
    m.eval()
    m.config.dspark_enabled = True
    x = torch.randint(0, 256, (1, 8))
    assert not m._spec_eligible(x, 0.0)
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=16)
        other = m.generate(x, max_new_tokens=16)
    assert torch.equal(base, other)


# --------------------------- 截断效果 ---------------------------
class _LowConfProvider:
    """长草稿但置信度极低 → DSpark 应把验证长度压到 min_verify."""

    def __init__(self, k=4, conf=0.1):
        self.k = k
        self.conf = conf

    def propose(self, context, h_last, k):
        return [0] * max(0, k)

    def propose_scored(self, context, h_last, k):
        n = max(0, min(k, self.k))
        return [0] * n, [self.conf] * n


def test_dspark_trims_verification(monkeypatch):
    """低置信长草稿：verified < proposed，但输出仍等于贪心."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=24)
    monkeypatch.setattr(
        spec_mod, "build_draft_provider", lambda *a, **k: _LowConfProvider())
    m.config.draft_enabled = True
    m.config.draft_window = 5
    m.config.dspark_enabled = True
    m.config.dspark_conf_threshold = 0.5
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=24)
    assert torch.equal(spec, base)
    st = m._spec_stats
    assert st["verified"] < st["proposed"]  # 后缀被截断
    assert st["verified"] == st["rounds"]   # 每轮只验 min_verify=1 个草稿


def test_dspark_threshold_zero_equals_fixed(monkeypatch):
    """threshold=0：不截断，verified == proposed（退化回固定窗口）."""
    m = _tiny()
    m.eval()
    monkeypatch.setattr(
        spec_mod, "build_draft_provider", lambda *a, **k: _LowConfProvider())
    m.config.draft_enabled = True
    m.config.draft_window = 5
    m.config.dspark_enabled = True
    m.config.dspark_conf_threshold = 0.0
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        m.generate(x, max_new_tokens=16)
    assert m._spec_stats["verified"] == m._spec_stats["proposed"]
