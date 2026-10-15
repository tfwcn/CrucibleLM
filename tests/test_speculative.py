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


def test_hybrid_prefers_mtp_unless_ngram_agrees():
    """hybrid 策略：ngram 与 MTP 首猜一致才采纳 ngram 长草稿，否则只用 MTP.

    这样既拿到 ngram 的一致性续写，又避免"把 MTP 猜测并入上下文"污染 ngram
    后缀检索（旧实现的坑：实测 ngram 补位几乎永不命中）。"""
    m = _tiny()
    p = HybridDraftProvider(m)

    class _FakeMtp:
        def __init__(self, out):
            self.out = out

        def propose(self, context, h_last, k):
            return list(self.out[:max(0, k)])

    class _FakeNg:
        def __init__(self, out):
            self.out = out

        def propose(self, context, h_last, k):
            return list(self.out[:max(0, k)])

    # 1) 一致：采纳 ngram（更长）
    p.mtp, p.ngram = _FakeMtp([7]), _FakeNg([7, 8, 9, 10])
    assert p.propose([1, 2], None, 4) == [7, 8, 9, 10]
    # 2) 不一致：只用 MTP 单猜
    p.mtp, p.ngram = _FakeMtp([7]), _FakeNg([5, 8, 9, 10])
    assert p.propose([1, 2], None, 4) == [7]
    # 3) 无 MTP 首猜：退回 ngram
    p.mtp, p.ngram = _FakeMtp([]), _FakeNg([5, 8, 9, 10])
    assert p.propose([1, 2], None, 4) == [5, 8, 9, 10]
    # k<=0 恒空
    assert p.propose([1, 2], None, 0) == []


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


def test_spec_matches_greedy_with_hyper_memory_retro():
    """全特性配置（hyper 多流 + memory + retro）下投机仍逐位等于贪心.

    真实 128M 检查点就是这套配置；step_pasts 跳位必须在 hyper 展开/combine、
    memory、retro 都开时仍成立。"""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.hyper_streams = 2
    cfg.memory_every = 2
    cfg.memory_slots = 64
    cfg.memory_topk = 4
    cfg.retro_enabled = True
    cfg.retro_every = 2
    cfg.retro_heads = 4
    cfg.retro_chunk_len = 8
    m = TinyLLM(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=24)
    m.config.draft_enabled = True
    m.config.draft_source = "hybrid"
    m.config.draft_window = 5
    with torch.no_grad():
        spec = m.generate(x, max_new_tokens=24)
    assert torch.equal(base, spec)


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
        multi, _, step_pasts = m._decode_multi(toks, pb)
    assert torch.allclose(seq, multi, atol=1e-5), (seq - multi).abs().max()


def test_decode_multi_step_pasts_match_sequential():
    """_decode_multi 的 step_pasts[i] 与逐 token 解码到第 i 步的缓存等价.

    投机解码靠它做"部分接受零重放"：从 step_pasts[A] 续跑必须和从头逐 token
    跑到 A 的续跑逐位一致，否则契约破。"""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    toks = torch.randint(0, 256, (1, 4))
    nxt = torch.randint(0, 256, (1, 1))
    with torch.no_grad():
        _, pa = m._prefill(x, max_new_tokens=8)
        _, pb = m._prefill(x, max_new_tokens=8)
        _, _, step_pasts = m._decode_multi(toks, pa)
        seq_pasts = []
        cur = pb
        for i in range(toks.shape[1]):
            _, cur = m._decode_step(toks[:, i:i + 1], cur)
            seq_pasts.append(cur)
        for A in range(toks.shape[1]):
            h_m, _ = m._decode_step(nxt, step_pasts[A])
            h_s, _ = m._decode_step(nxt, seq_pasts[A])
            assert torch.allclose(h_m, h_s, atol=1e-5), (A, (h_m - h_s).abs().max())


# --------------------------- 接受 / 回滚路径 ---------------------------
class _CorrectProvider:
    """返回已知贪心序列的下一段 → 必然全接受（锁定"零回滚"快分支）."""

    def __init__(self, full_ids: list[int]):
        self.full = full_ids

    def propose(self, context, h_last, k):
        return self.full[len(context): len(context) + max(0, k)]


class _WrongProvider:
    """恒返回错误草稿 → 必然拒答（锁定"部分接受 → 按步缓存跳位"分支）."""

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
    """部分接受路径：恒错草稿下输出仍等于贪心（缓存按步对齐，零重放）."""
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
    assert m._spec_stats["accepted"] == 0  # 全部拒答，走了部分接受跳位


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
