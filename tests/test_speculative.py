"""投机解码单测 — 起草器 / 并行验证 / "等于贪心"契约 / 回滚（CPU 可跑）."""

import pytest

torch = pytest.importorskip("torch")

import src.llm.local.spec_decode as spec_mod
from src.llm.local.config import tiny_test_config
from src.llm.local.model import TinyLLM
from src.llm.local.spec_decode import (
    HybridDraftProvider,
    MtpDraftProvider,
    NgramDraftProvider,
    build_draft_provider,
)


def _tiny():
    """构建极小模型（CPU 秒级）."""
    torch.manual_seed(0)
    return TinyLLM(tiny_test_config())


def _spec_cfg(source="hybrid", window=5):
    """开投机解码的极小配置."""
    cfg = tiny_test_config()
    cfg.draft_enabled = True
    cfg.draft_source = source
    cfg.draft_window = window
    return cfg


# --------------------------- 起草器单测 ---------------------------
def test_ngram_provider_finds_suffix_continuation():
    """前缀查表：尾部后缀此前出现过则抄其后续（取最近一次）."""
    p = NgramDraftProvider(min_match=2)
    ctx = [1, 2, 3, 1, 2]
    assert p.propose(ctx, None, 3) == [3, 1, 2]
    # 无匹配返回空
    assert p.propose([9, 8, 7, 6], None, 3) == []
    # k=0 恒空
    assert p.propose(ctx, None, 0) == []


def test_ngram_provider_min_match_guards():
    """min_match 提高后，短重复不再误命中."""
    p = NgramDraftProvider(min_match=3)
    assert p.propose([1, 2, 1, 2], None, 2) == []  # 只有 2 长重复，被拒


def test_mtp_provider_returns_single():
    """MTP 起草：只给 1 个候选（只多看一步）."""
    m = _tiny()
    p = MtpDraftProvider(m)
    out = p.propose([1, 2, 3], torch.randn(1, 1, m.config.d_model), 4)
    assert len(out) == 1 and 0 <= out[0] < m.config.vocab_size
    # k=0 恒空
    assert p.propose([1], torch.randn(1, 1, m.config.d_model), 0) == []


def test_build_draft_provider_dispatch():
    """来源分派：未知来源回落 ngram（不崩）."""
    m = _tiny()
    assert isinstance(build_draft_provider(m, "ngram"), NgramDraftProvider)
    assert isinstance(build_draft_provider(m, "mtp"), MtpDraftProvider)
    assert isinstance(build_draft_provider(m, "hybrid"), HybridDraftProvider)
    assert isinstance(build_draft_provider(m, "???"), NgramDraftProvider)


# --------------------------- 契约：等于贪心 ---------------------------
@pytest.mark.parametrize("source", ["ngram", "mtp", "hybrid"])
def test_spec_matches_greedy(source):
    """各起草来源：投机输出与逐 token 贪心逐位一致（含重复惩罚）."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=32)
        base_rp = m.generate(x, max_new_tokens=32, repetition_penalty=1.3)
    m.config.draft_enabled = True
    m.config.draft_source = source
    m.config.draft_window = 5
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=32)
        spec_rp = m.generate(x, max_new_tokens=32, repetition_penalty=1.3)
    assert torch.equal(base, spec)
    assert torch.equal(base_rp, spec_rp)


def test_spec_stream_matches_generate():
    """stream_tokens 走投机路径时与 generate 逐位一致."""
    m = _tiny()
    m.eval()
    m.config.draft_enabled = True
    m.config.draft_source = "hybrid"
    m.config.draft_window = 6
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        g = m.generate(x, max_new_tokens=24)
        s = list(m.stream_tokens(x, max_new_tokens=24))
    assert s == g[0, 8:].tolist()


def test_spec_default_off_unchanged():
    """开关关时生成路径逐位不变（默认关的契约）."""
    m = _tiny()
    m.eval()
    assert m.config.draft_enabled is False
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        a = m.generate(x, max_new_tokens=24)
        b = m.generate(x, max_new_tokens=24)
    assert torch.equal(a, b)


def test_spec_temperature_falls_back():
    """非贪心（temperature>0）不走投机：保证采样语义不被篡改."""
    m = _tiny()
    m.eval()
    m.config.draft_enabled = True
    # batch>1 同样回落
    x2 = torch.randint(0, 256, (2, 8))
    assert not m._spec_eligible(x2, 0.0)
    # temperature>0 回落
    x1 = torch.randint(0, 256, (1, 8))
    assert not m._spec_eligible(x1, 0.7)


# --------------------------- 并行验证原语 ---------------------------
def test_decode_multi_matches_sequential():
    """_decode_multi（多 token 并行）与逐 token _decode_step 数值一致."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    toks = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        _, pa = m._prefill(x, max_new_tokens=8)
        _, pb = m._prefill(x, max_new_tokens=8)
        hs = []
        cur = pa
        for i in range(toks.shape[1]):
            hh, cur = m._decode_step(toks[:, i:i + 1], cur)
            hs.append(hh)
        seq = torch.cat(hs, dim=1)
        multi, _ = m._decode_multi(toks, pb)
    assert torch.allclose(seq, multi, atol=1e-5), (seq - multi).abs().max()


# --------------------------- 接受 / 回滚路径 ---------------------------
class _CorrectProvider:
    """返回已知贪心序列的下一段 → 必然全接受（锁定"零回滚"快分支）."""

    def __init__(self, full_ids: list[int]):
        self.full = full_ids

    def propose(self, context, h_last, k):
        return self.full[len(context): len(context) + max(0, k)]


class _WrongProvider:
    """恒返回错误草稿 → 必然拒答（锁定"部分接受 + 回滚重放"分支）."""

    def propose(self, context, h_last, k):
        return [0] * max(0, k)


def test_spec_full_accept_path(monkeypatch):
    """全接受分支：起草=真实贪心续写，输出一致且确有接受发生."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=24)
    full = base[0].tolist()  # 完整贪心序列（prompt + 生成）
    monkeypatch.setattr(
        spec_mod, "build_draft_provider", lambda *a, **k: _CorrectProvider(full))
    m.config.draft_enabled = True
    m.config.draft_window = 5
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=24)
    assert torch.equal(spec, base)
    assert m._spec_stats["accepted"] > 0
    assert m._spec_stats["rounds"] < m._spec_stats["emitted"]


def test_spec_reject_rollback_path(monkeypatch):
    """回滚分支：恒错草稿下输出仍等于贪心（状态经重放精确对齐）."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=24)
    monkeypatch.setattr(
        spec_mod, "build_draft_provider", lambda *a, **k: _WrongProvider())
    m.config.draft_enabled = True
    m.config.draft_window = 5
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=24)
    assert torch.equal(spec, base)
    assert m._spec_stats["accepted"] == 0  # 全部拒答，走了回滚


def test_spec_respects_eos():
    """eos 命中即停：投机路径与贪心路径对 eos 的处理一致（不多生成）."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    m.config.draft_enabled = False
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=32, eos_id=5)
    m.config.draft_enabled = True
    m.config.draft_source = "hybrid"
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=32, eos_id=5)
    assert torch.equal(base, spec)
