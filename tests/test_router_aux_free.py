"""免辅助损失路由单测 — sqrtsoftplus + noaux_tc + routed scaling（V4.1）.

CPU 可跑。核心契约：
- 生产默认 `router_aux_free=True`（V4.1）；关闭时逐位等价旧口径（softmax+greedy+aux）；
- aux-free 下 aux loss 恒 0，权重 = 原始打分归一化 × `routed_scaling_factor`；
- 选择用 `打分 + 偏置`；偏置按负载 sign 式更新（update_router_bias），能把负载摊平；
- 偏置是非持久化 buffer：不进 state_dict，装载键兼容；
- 属"改前向语义不改键"的开关：装载既有权重须按来源对齐（旧 run 回退旧口径）。
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.moe import FineGrainedMoE  # noqa: E402


# ------------------------------- 默认值 -------------------------------
def test_defaults_on_tiny_is_legacy():
    """生产默认走 aux-free；CPU 极小配置固定旧口径（结构契约不随生产默认漂移）."""
    cfg = SmallLLMConfig()
    assert cfg.router_aux_free is True
    assert cfg.routed_scaling_factor == 1.5
    assert cfg.norm_topk_prob is True
    assert tiny_test_config().router_aux_free is False


def test_model_wires_router_fields_and_no_bias_key():
    """模型把路由字段传到每层 MoE；偏置 buffer 不出现在 state_dict（键兼容）."""
    from src.llm.local.model import TinyLLM

    cfg = tiny_test_config()
    cfg.router_aux_free = True
    cfg.routed_scaling_factor = 2.0
    m = TinyLLM(cfg)
    for layer in m.layers:
        assert layer.moe.router_aux_free is True
        assert layer.moe.routed_scaling_factor == 2.0
    keys = list(m.state_dict().keys())
    assert not any("expert_bias" in k for k in keys), "偏置不应进 state_dict"


# --------------------------- 数学正确（aux-free） ---------------------------
def _naive_moe(m, x):
    """与实现无关的参考 MoE（按模块自身开关走对应口径）."""
    logits = m.router(x)
    if m.router_aux_free:
        score = torch.sqrt(F.softplus(logits))
        sel = score + m.expert_bias.to(score.dtype)
    else:
        score = F.softmax(logits, dim=-1)
        sel = score
    _, ti = torch.topk(sel, m.top_k, dim=-1)
    w = torch.gather(score, -1, ti)
    if m.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
    w = (w * (m.routed_scaling_factor if m.router_aux_free else 1.0)).to(x.dtype)
    out = torch.zeros_like(x)
    for e, expert in enumerate(m.experts):
        we = torch.zeros_like(w[..., 0])
        for k in range(m.top_k):
            we = we + torch.where(ti[..., k] == e, w[..., k], torch.zeros(()))
        out = out + we.unsqueeze(-1) * expert(x)
    for expert in m.shared:
        out = out + expert(x)
    return out


def test_aux_free_grouped_matches_naive_and_zero_aux():
    """aux-free：分组前向等于朴素参考，aux loss 恒 0，梯度有限."""
    torch.manual_seed(0)
    m = FineGrainedMoE(32, n_experts=6, top_k=2, expert_hidden=12,
                       n_shared=2, swiglu_limit=1.0, router_aux_free=True,
                       routed_scaling_factor=1.5)
    m.eval()
    x = torch.randn(2, 10, 32)
    out, aux = m(x)
    assert torch.allclose(out, _naive_moe(m, x), atol=1e-5)
    assert float(aux) == 0.0
    out.sum().backward()
    for n, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_aux_free_offload_matches_grouped():
    """aux-free 下 offload 逐专家路径与分组路径一致."""
    torch.manual_seed(1)
    m = FineGrainedMoE(32, n_experts=4, top_k=2, expert_hidden=12, n_shared=1,
                       router_aux_free=True)
    m.eval()
    x = torch.randn(2, 6, 32)
    grouped, _ = m(x)

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


def test_legacy_path_reproduces_softmax_aux():
    """关闭 aux-free → 旧口径：aux>0 且等于标准负载均衡式（回归保护）."""
    torch.manual_seed(2)
    m = FineGrainedMoE(32, n_experts=5, top_k=2, expert_hidden=10, n_shared=1,
                       aux_coef=0.01, router_aux_free=False)
    m.eval()
    x = torch.randn(2, 7, 32)
    out, aux = m(x)
    assert torch.allclose(out, _naive_moe(m, x), atol=1e-5)
    probs = F.softmax(m.router(x), -1)
    _, ti = torch.topk(probs, 2, dim=-1)
    imp = probs.reshape(-1, 5).mean(0)
    lf = F.one_hot(ti.reshape(-1), 5).float().sum(0) / ti.numel()
    assert torch.allclose(aux, (imp * lf).sum() * 5 * 0.01, atol=1e-6)


def test_sqrtsoftplus_monotone_and_positive():
    """打分 = sqrt(softplus)：正值、随 logits 单调增（保 top-k 直觉）."""
    z = torch.linspace(-20, 20, 41)
    s = torch.sqrt(F.softplus(z))
    assert (s > 0).all()
    assert (s[1:] >= s[:-1]).all()


# --------------------------- 偏置机制（noaux_tc） ---------------------------
def test_bias_sign_update_exact():
    """update_router_bias 按 sign(目标−负载) 更新（精确一步）."""
    m = FineGrainedMoE(16, n_experts=4, top_k=1, expert_hidden=8,
                       router_aux_free=True, router_bias_lr=0.1)
    m.train()
    # 手工注入：全部 token 命中专家 0
    m._last_load = torch.tensor([1.0, 0.0, 0.0, 0.0])
    m.update_router_bias()
    # 目标 0.25：专家 0 过载 → 偏置降；其余欠载 → 升高
    assert m.expert_bias[0].item() == pytest.approx(-0.1)
    for i in (1, 2, 3):
        assert m.expert_bias[i].item() == pytest.approx(0.1)


def test_bias_balances_load_over_steps():
    """零 router（等分打分）下，偏置更新能在多步内把负载摊到多个专家."""
    torch.manual_seed(0)
    m = FineGrainedMoE(16, n_experts=4, top_k=1, expert_hidden=8,
                       router_aux_free=True, router_bias_lr=0.1)
    with torch.no_grad():
        m.router.weight.zero_()  # 打分全相等 → 完全由偏置决定选择（制造极端不均）
    m.train()
    chosen = set()
    x = torch.randn(4, 16)
    for _ in range(20):
        m(x)
        load = m._last_load
        chosen.update((load > 0).nonzero().flatten().tolist())
        m.update_router_bias()
    assert len(chosen) >= 2, f"负载未摊开：只用到 {chosen}"


def test_update_bias_noop_when_disabled():
    """aux-free 关闭时 update_router_bias 不动偏置."""
    m = FineGrainedMoE(16, n_experts=4, top_k=2, expert_hidden=8,
                       router_aux_free=False)
    m.train()
    m._last_load = torch.tensor([1.0, 0.0, 0.0, 0.0])
    m.update_router_bias()
    assert torch.equal(m.expert_bias, torch.zeros(4))


def test_eval_does_not_accumulate_load():
    """eval 前向不累计负载（不污染训练的偏置更新）."""
    m = FineGrainedMoE(16, n_experts=4, top_k=1, expert_hidden=8,
                       router_aux_free=True)
    m.eval()
    m(torch.randn(4, 16))
    assert m._last_load is None


# --------------------------- 来源对齐 ---------------------------
def test_sync_reads_router_fields(tmp_path):
    """装载既有权重时对齐路由开关；旧格式回退旧口径（aux loss + 无缩放）."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "router_aux_free": False, "routed_scaling_factor": 2.0}}),
        encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.router_aux_free is False
    assert cfg.routed_scaling_factor == 2.0

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 768}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.router_aux_free is False       # 旧 run 不被静默换路由
    assert cfg2.routed_scaling_factor == 1.0


def test_restore_arch_switches_priority(tmp_path):
    """通用解析：显式 CLI > config-json > checkpoint 沿用 > 续训保守 / 新训默认."""
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    def ns(**kw):
        d = dict(csa_reuse_every=None, swiglu_limit=None, router_aux_free=None,
                 routed_scaling_factor=None, norm_topk_prob=None,
                 resume=False, ckpt_dir="", init_checkpoint="",
                 sft_init="", config_json="")
        d.update(kw)
        return argparse.Namespace(**d)

    # 新训练：架构默认（aux-free 开）
    cfg = SmallLLMConfig()
    src = T.resolve_arch_switches(ns(), cfg)
    assert cfg.router_aux_free is True
    assert src["router_aux_free"] == "default"

    # 显式关
    cfg = SmallLLMConfig()
    T.resolve_arch_switches(ns(router_aux_free=False), cfg)
    assert cfg.router_aux_free is False

    # config-json 视为显式来源
    cfg = SmallLLMConfig(router_aux_free=False)
    src = T.resolve_arch_switches(ns(config_json="x.json"), cfg)
    assert cfg.router_aux_free is False
    assert src["router_aux_free"] == "config_json"

    # checkpoint 沿用
    (tmp_path / "hparams.json").write_text(
        json.dumps({"config": {"router_aux_free": False}}), encoding="utf-8")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    cfg = SmallLLMConfig()
    src = T.resolve_arch_switches(ns(init_checkpoint=str(ckpt)), cfg)
    assert cfg.router_aux_free is False
    assert src["router_aux_free"] == "checkpoint"

    # 续训但无架构信息 → 保守旧口径
    empty = tmp_path / "noarch"
    empty.mkdir()
    (empty / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    src = T.resolve_arch_switches(ns(init_checkpoint=str(empty / "model.pt")), cfg)
    assert cfg.router_aux_free is False
    assert src["router_aux_free"] == "conservative"


def test_cli_router_tristate():
    """CLI 三态：不传=None（走沿用/默认），--router-aux-free=开，--no-...=关."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).router_aux_free is None
    assert T.parse_args(["--router-aux-free"]).router_aux_free is True
    assert T.parse_args(["--no-router-aux-free"]).router_aux_free is False
    assert T.parse_args([]).routed_scaling_factor is None
    assert T.parse_args(
        ["--routed-scaling-factor", "1.2"]).routed_scaling_factor == 1.2
