"""SwiGLU 限幅单测 — 默认开 / 数学正确 / 分组=逐专家 / 分析口径一致 / 架构沿用.

CPU 可跑。核心契约：
- 生产默认 `swiglu_limit=10.0`（V4.1），`0` = 关闭且逐位等价旧实现；
- 限幅只作用于 gate/up 预激活（`clamp(z,-L,L)`），符号不变、不改 state_dict 键；
- 分组 bmm 与逐专家路径在限幅下仍数值一致；
- 限幅是"改前向语义不改键"的开关：装载既有权重须按来源对齐（旧 run 回退 0）。
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.moe import FineGrainedMoE, SwiGLUExpert  # noqa: E402


# ------------------------------- 默认值 -------------------------------
def test_default_is_on_tiny_is_off():
    """生产默认限幅 10.0；CPU 极小配置固定不限幅（结构契约不随生产默认漂移）."""
    assert SmallLLMConfig().swiglu_limit == 10.0
    assert tiny_test_config().swiglu_limit == 0.0


def test_model_wires_limit_to_all_experts():
    """模型把 config.swiglu_limit 传到每个（路由+共享）专家；限幅不出现在 state_dict."""
    from src.llm.local.model import TinyLLM

    cfg = tiny_test_config()
    cfg.swiglu_limit = 7.5
    m = TinyLLM(cfg)
    for layer in m.layers:
        assert layer.moe.swiglu_limit == 7.5
        for e in list(layer.moe.experts) + list(layer.moe.shared):
            assert e.limit == 7.5
    # 限幅是纯超参：不引入任何 state_dict 键（装载兼容）
    assert not any("limit" in k for k in m.state_dict())


# ------------------------------- 数学正确 -------------------------------
def _manual_expert(expert, x, limit):
    """与实现无关的参考 SwiGLU（可带对称限幅）."""
    g = expert.w_gate(x)
    u = expert.w_up(x)
    if limit > 0:
        g = g.clamp(-limit, limit)
        u = u.clamp(-limit, limit)
    return expert.w_down(F.silu(g) * u)


def test_forward_matches_manual_with_clamp():
    """限幅生效时前向等于手工参考（紧公差）；且确实改变了输出."""
    torch.manual_seed(0)
    e = SwiGLUExpert(8, 6, limit=1.0)
    x = 50.0 * torch.randn(4, 8)  # 预激活远超阈值，确保限幅触发
    ref = _manual_expert(e, x, 1.0)
    assert torch.allclose(e(x), ref, atol=1e-6)
    # 与不限幅的输出应明显不同（证明限幅确实在起作用）
    e0 = SwiGLUExpert(8, 6, limit=0.0)
    e0.load_state_dict(e.state_dict())
    assert not torch.allclose(e(x), e0(x), atol=1e-3)


def test_limit_zero_is_exact_old_behavior():
    """limit=0 → 逐位等价无限幅旧实现."""
    torch.manual_seed(1)
    e = SwiGLUExpert(8, 6, limit=0.0)
    x = torch.randn(5, 8)
    assert torch.equal(e(x), e.w_down(F.silu(e.w_gate(x)) * e.w_up(x)))


def test_clamp_keeps_sign():
    """限幅只压幅值、不改符号（双负区间分析口径不变）."""
    g = torch.tensor([-100.0, -3.0, 0.5, 4.0, 100.0])
    c = g.clamp(-10.0, 10.0)
    assert torch.equal(torch.sign(g), torch.sign(c))


# --------------------------- 分组 == 逐专家 ---------------------------
def _naive_moe(m, x):
    """全量朴素 MoE 前向（参考实现，含限幅走 expert 自身）."""
    probs = torch.softmax(m.router(x), -1)
    tw, ti = torch.topk(probs, m.top_k, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True).clamp_min(1e-6)
    out = torch.zeros_like(x)
    for e, expert in enumerate(m.experts):
        w = torch.zeros_like(tw[..., 0])
        for k in range(m.top_k):
            w = w + torch.where(ti[..., k] == e, tw[..., k], torch.zeros(()))
        out = out + w.unsqueeze(-1) * expert(x)
    for expert in m.shared:
        out = out + expert(x)
    return out


def test_grouped_matches_naive_under_clamp():
    """分组 bmm 与朴素全量在限幅下一致；且梯度有限."""
    torch.manual_seed(2)
    m = FineGrainedMoE(32, n_experts=6, top_k=2, expert_hidden=12,
                       n_shared=2, aux_coef=0.01, swiglu_limit=1.0)
    m.eval()
    x = 20.0 * torch.randn(2, 10, 32)  # 大输入确保限幅触发
    out, aux = m(x)
    assert torch.allclose(out, _naive_moe(m, x), atol=1e-5)
    assert torch.isfinite(aux)
    (out.sum() + aux).backward()
    for n, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_offload_path_matches_grouped_under_clamp():
    """offload 逐专家路径与分组路径在限幅下数值一致（含共享专家）."""
    torch.manual_seed(3)
    m = FineGrainedMoE(32, n_experts=4, top_k=2, expert_hidden=12,
                       n_shared=1, swiglu_limit=1.0)
    m.eval()
    x = 20.0 * torch.randn(2, 6, 32)
    grouped, _ = m(x)
    # 逐专家路径：临时塞一个"全部驻留"的桩 store，强制走 _forward_offload
    class _Store:
        def all_resident(self):
            return False

        def ensure(self, ids, prefetch=False):
            return None

        def hot_ids(self):
            return []

    m.store = _Store()
    offload, _ = m._forward_offload(x)
    assert torch.allclose(grouped, offload, atol=1e-5)


# --------------------------- 架构沿用/来源对齐 ---------------------------
def test_sync_arch_reads_swiglu_from_checkpoint(tmp_path):
    """装载既有权重时从 hparams 对齐 swiglu_limit；旧格式回退 0（不限幅）."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(
        json.dumps({"config": {"swiglu_limit": 20.0}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    assert "swiglu_limit" in sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.swiglu_limit == 20.0

    # 旧 checkpoint：有配置但无该字段 → 0（早于限幅特性）
    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"csa_reuse_every": 2}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.swiglu_limit == 0.0


def test_resolve_swiglu_limit_branches(tmp_path):
    """决策优先级：显式 > config-json > checkpoint 沿用 > 续训保守 0 / 新训默认 10."""
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    def ns(**kw):
        d = dict(swiglu_limit=None, resume=False, ckpt_dir="",
                 init_checkpoint="", sft_init="", config_json="")
        d.update(kw)
        return argparse.Namespace(**d)

    cfg = SmallLLMConfig()  # 架构默认 = 10.0
    assert T.resolve_swiglu_limit(ns(swiglu_limit=3.0), cfg) == (3.0, "explicit")
    assert T.resolve_swiglu_limit(ns(swiglu_limit=0.0), cfg) == (0.0, "explicit")
    assert T.resolve_swiglu_limit(ns(), cfg) == (10.0, "default")
    assert T.resolve_swiglu_limit(
        ns(config_json="x.json"), _cfg(swiglu_limit=4.0)) == (4.0, "config_json")
    # 续训能读到 hparams → 沿用
    (tmp_path / "hparams.json").write_text(
        json.dumps({"config": {"swiglu_limit": 6.0}}), encoding="utf-8")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    assert T.resolve_swiglu_limit(
        ns(init_checkpoint=str(ckpt)), cfg) == (6.0, "checkpoint")
    # 续训但无架构信息 → 保守按 0（不静默改前向）
    empty = tmp_path / "noarch"
    empty.mkdir()
    (empty / "model.pt").write_bytes(b"")
    assert T.resolve_swiglu_limit(
        ns(init_checkpoint=str(empty / "model.pt")), cfg) == (0.0, "conservative")
    assert T.resolve_swiglu_limit(
        ns(sft_init=str(empty / "model.pt")), cfg) == (0.0, "conservative")


def _cfg(**over):
    """构造生产默认结构的小配置（仅覆盖指定字段）."""
    base = dict(swiglu_limit=10.0)
    base.update(over)
    return SmallLLMConfig(**base)
