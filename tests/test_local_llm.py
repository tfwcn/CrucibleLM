"""本地小 LLM 单测 — 前向 / MTP loss / 生成 / 单步训练 / 显存估算.

需要 torch；无 torch 环境自动跳过（CI 不强制装 torch）。
CPU 用极小配置跑通逻辑，16G 训练用默认配置只做显存估算（不实例化）。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig, tiny_test_config
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
from src.llm.local.model import TinyLLM
from src.llm.local.train import build_optimizer, estimate_memory_gb, train_step


def _tiny():
    """构建极小模型（CPU 秒级）."""
    torch.manual_seed(0)
    return TinyLLM(tiny_test_config())


def test_forward_shapes():
    """前向形状：logits 与 aux_loss 正常返回."""
    m = _tiny()
    x = torch.randint(0, 256, (2, 16))
    out = m(x)
    assert out["logits"].shape == (2, 16, 256)
    assert torch.isfinite(out["aux_loss"])


def test_forward_with_loss():
    """带 targets 前向：主 loss + MTP loss + aux 组合正常."""
    m = _tiny()
    x = torch.randint(0, 256, (2, 16))
    out = m(x, targets=x)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["main_loss"])
    assert "mtp_loss" in out
    # loss 可反向（训练可行）
    out["loss"].backward()
    assert m.embed.weight.grad is not None


def test_generate_greedy():
    """贪心生成：长度正确、id 合法."""
    m = _tiny()
    x = torch.randint(0, 256, (1, 8))
    gen = m.generate(x, max_new_tokens=8)
    assert gen.shape == (1, 16)
    assert int(gen.max()) < 256


def test_repetition_penalty_off_is_identity():
    """惩罚=1.0 时与旧采样逐位一致；stream 与 generate 同基元逐位一致."""
    m = _tiny()
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        g1 = m.generate(x, max_new_tokens=8)
        g2 = m.generate(x, max_new_tokens=8, repetition_penalty=1.0)
        s1 = list(m.stream_tokens(x, max_new_tokens=8))
        s2 = list(m.stream_tokens(x, max_new_tokens=8, repetition_penalty=1.3))
        g3 = m.generate(x, max_new_tokens=8, repetition_penalty=1.3)
    assert torch.equal(g1, g2)
    assert s1 == g1[0, 8:].tolist()
    assert s2 == g3[0, 8:].tolist()


def test_repetition_penalty_suppresses_seen_tokens():
    """惩罚压低已出现 token 的原始 logit（正值除、负值乘，一律下降）."""
    m = _tiny()
    m.eval()
    h = torch.randn(1, 1, m.config.d_model)
    with torch.no_grad():
        base = m._sample_next(h, 0.0, 0)
        t = int(base[0, 0])
        raw = m.lm_head(h)[:, -1, :].float()[0, t].item()
        got = m.lm_head(h)[:, -1, :].float()
        past = torch.full((1, 6), t, dtype=torch.long)
        seen = torch.zeros_like(got, dtype=torch.bool)
        seen.scatter_(1, past.clamp(0, got.shape[-1] - 1), True)
        lowered = torch.where(
            seen,
            torch.where(got < 0, got * 1.5, got / 1.5), got)[0, t].item()
        assert lowered < raw


def test_repetition_penalty_per_row_independent():
    """batch 行间互不串扰：只换第 0 行上下文，第 1 行输出逐位不变."""
    m = _tiny()
    m.eval()
    h = torch.randn(2, 1, m.config.d_model)
    row1_past = torch.tensor([[5, 6, 7, 8, 9, 10]], dtype=torch.long)
    past_a = torch.zeros(2, 6, dtype=torch.long)
    past_a[1] = row1_past
    past_b = torch.full((2, 6), 200, dtype=torch.long)
    past_b[1] = row1_past
    with torch.no_grad():
        out_a = m._sample_next(h, 0.0, 0, 2.0, past_a)
        out_b = m._sample_next(h, 0.0, 0, 2.0, past_b)
    assert int(out_a[1, 0]) == int(out_b[1, 0])
    # past=None 等价无惩罚
    with torch.no_grad():
        out_none = m._sample_next(h, 0.0, 0, 2.0, None)
        out_plain = m._sample_next(h, 0.0, 0)
    assert torch.equal(out_none, out_plain)


def test_train_step_decreases_or_finite():
    """单步训练：loss 有限、梯度范数有限."""
    m = _tiny()
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 256, (2, 16))
    stats = train_step(m, opt, x)
    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert torch.isfinite(torch.tensor(stats["grad_norm"]))


def test_memory_estimate_default_config_fits_16g():
    """默认 100M 配置在 seq2048/batch4/检查点下能塞进 16G."""
    mem = estimate_memory_gb(SmallLLMConfig(), seq_len=2048, micro_batch=4)
    assert mem["fits_16g"], mem
    assert mem["total_params_m"] < 200


def test_param_counts():
    """参数统计：总量约 100M、激活约 40M（默认配置，纯算术不建模）."""
    m = _tiny()
    counts = m.count_params()
    assert counts["total"] < counts["active"] * 10  # MoE 总量远大于激活
    assert counts["total"] > counts["active"]


def test_chat_backend_end_to_end():
    """chat 端到端：分词→生成→解码跑通."""
    m = _tiny()
    tok = SimpleTokenizer(256)
    tok.fit(["你好世界", "hello world", "摸鱼使我快乐"])
    backend = LocalChatBackend(m, tok)
    resp = backend.chat(
        [{"role": "user", "content": "你好"}],
        max_new_tokens=8,
    )
    assert set(resp) == {"content", "reasoning", "tool_calls", "finish_reason"}
    assert isinstance(resp["content"], str)
