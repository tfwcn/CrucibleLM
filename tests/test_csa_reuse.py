"""CSA2 跨层 KV 复用单测 — 默认开/显式关 / 按序分配 / 参数共享 / 解码一致 / 因果.

CPU 可跑。核心契约：
- 生产默认 `csa_reuse_every=2` → 有 Reuse 层（`_csa_active=True`）；
  显式 `=1` → 全 Full，与旧 checkpoint 逐位兼容；
- Reuse 层复用最近 Full 层的 latent，只自算 Q/上投影，**不重复存 KV 缓存**；
- Reuse 层与 Full 层共享 `w_dkv`/`w_kr` 权重对象（避免死参数）；
- `prefill + 逐 token 解码 == 全前向`、`_decode_multi == 逐 token`、投机 == 贪心。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (
    SmallLLMConfig,
    sync_csa_reuse_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM


def _csa_cfg(every=2):
    """开 CSA2 的极小配置（tiny: full_attn_every=2 → 层 0/2 为全注意力）."""
    cfg = tiny_test_config()
    cfg.csa_reuse_every = every
    return cfg


def _full_layers(m):
    return [i for i, L in enumerate(m.layers) if getattr(L, "full_attn", False)]


def _reuse_layers(m):
    return [i for i, L in enumerate(m.layers)
            if getattr(L.attn, "csa_reuse", False)]


# --------------------------- 默认开 / 显式关 / 分配 ---------------------------
def _prod_cfg(**over):
    """生产默认结构的小模型（默认 csa_reuse_every=2）."""
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=2,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=256, yarn_scale=1.0,
    )
    base.update(over)
    return SmallLLMConfig(**base)


def test_csa_default_is_on():
    """生产默认 `csa_reuse_every=2` → 默认就有 Reuse 层（无需传参）."""
    assert SmallLLMConfig().csa_reuse_every == 2
    m = TinyLLM(_prod_cfg())
    assert m._csa_active is True
    # full_attn_every=2：层 0/2 为全注意力，ordinal 0=Full、1=Reuse
    assert _reuse_layers(m) == [2]


def test_tiny_preset_stays_baseline():
    """CPU 极小配置固定全 Full：单测结构契约不随生产默认漂移."""
    assert tiny_test_config().csa_reuse_every == 1


def test_sync_csa_reuse_from_checkpoint(tmp_path):
    """加载既有权重时按来源对齐 CSA2；无来源则不动（返回 False）."""
    import json

    cfg = SmallLLMConfig()  # 默认 2
    (tmp_path / "hparams.json").write_text(
        json.dumps({"config": {"csa_reuse_every": 1}}), encoding="utf-8")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    assert sync_csa_reuse_from_checkpoint(cfg, str(ckpt)) is True
    assert cfg.csa_reuse_every == 1
    # config.json（migrate 输出的扁平 dict）优先于 hparams
    (tmp_path / "config.json").write_text(
        json.dumps({"csa_reuse_every": 3}), encoding="utf-8")
    assert sync_csa_reuse_from_checkpoint(cfg, str(ckpt)) is True
    assert cfg.csa_reuse_every == 3
    # 无来源 → False，配置保持不变
    empty = tmp_path / "nope"
    empty.mkdir()
    cfg2 = SmallLLMConfig()
    assert sync_csa_reuse_from_checkpoint(cfg2, str(empty)) is False
    assert cfg2.csa_reuse_every == 2


def test_restore_arch_switch_reads_hparams(tmp_path):
    """续训/迁移时从 hparams.json 读回架构开关；读不到返回 None."""
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    ns = lambda **kw: argparse.Namespace(  # noqa: E731
        **{"resume": False, "ckpt_dir": "", "init_checkpoint": "", **kw})
    assert T._restore_arch_switch(ns(init_checkpoint=str(ckpt)),
                                  "csa_reuse_every") is None
    (tmp_path / "hparams.json").write_text(
        json.dumps({"config": {"csa_reuse_every": 2}}), encoding="utf-8")
    assert T._restore_arch_switch(ns(init_checkpoint=str(ckpt)),
                                  "csa_reuse_every") == 2
    assert T._restore_arch_switch(ns(resume=True, ckpt_dir=str(tmp_path)),
                                  "csa_reuse_every") == 2


def test_resolve_csa_reuse_every_branches(tmp_path):
    """决策优先级：命令行显式 > checkpoint 架构 > 续训保守 1 / 新训默认开."""
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    def ns(**kw):
        d = dict(csa_reuse_every=None, resume=False, ckpt_dir="",
                 init_checkpoint="", sft_init="", config_json="")
        d.update(kw)
        return argparse.Namespace(**d)

    cfg = _prod_cfg()  # 架构默认 = 2
    # 1) 命令行显式优先（给出即用，含显式关=1）
    assert T.resolve_csa_reuse_every(ns(csa_reuse_every=3), cfg) == (3, "explicit")
    assert T.resolve_csa_reuse_every(ns(csa_reuse_every=1), cfg) == (1, "explicit")
    # 2) 全新训练 → 架构默认（开）
    assert T.resolve_csa_reuse_every(ns(), cfg) == (2, "default")
    # 2b) --config-json 视为显式来源（其值优先于 checkpoint 沿用）
    assert T.resolve_csa_reuse_every(
        ns(config_json="x.json"), _prod_cfg(csa_reuse_every=1)) == (1, "config_json")
    # 3) 续训能读到 hparams → 沿用 checkpoint 架构（这里记为关）
    (tmp_path / "hparams.json").write_text(
        json.dumps({"config": {"csa_reuse_every": 1}}), encoding="utf-8")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    assert T.resolve_csa_reuse_every(
        ns(init_checkpoint=str(ckpt)), cfg) == (1, "checkpoint")
    # 4) 续训但 checkpoint 无架构信息 → 保守按 1（不静默换架构）
    empty = tmp_path / "noarch"
    empty.mkdir()
    (empty / "model.pt").write_bytes(b"")
    assert T.resolve_csa_reuse_every(
        ns(init_checkpoint=str(empty / "model.pt")), cfg) == (1, "conservative")
    # 5) --sft-init 严格载入同样走保守策略（无架构信息时按 1）
    assert T.resolve_csa_reuse_every(
        ns(sft_init=str(empty / "model.pt")), cfg) == (1, "conservative")


def test_csa_reuse_every_one_is_off():
    """显式 `csa_reuse_every=1` 也等价于关（1=全 Full）."""
    m = TinyLLM(_csa_cfg(every=1))
    assert m._csa_active is False
    assert _reuse_layers(m) == []


def test_csa_assigns_reuse_by_full_ordinal():
    """按全注意力层的序号分配：ordinal 0=Full，1=Reuse（full_attn_every=2）."""
    m = TinyLLM(_csa_cfg(every=2))
    fulls = _full_layers(m)
    assert fulls == [0, 2]
    # ordinal = layer_idx // full_attn_every → 0(Full), 1(Reuse)
    assert not getattr(m.layers[0].attn, "csa_reuse", False)
    assert getattr(m.layers[2].attn, "csa_reuse", False)
    assert m._csa_active is True


def test_csa_first_full_layer_never_reuse():
    """任何配置下第一个全注意力层必须是 Full（Reuse 需要有共享源）."""
    for every in (2, 3, 4):
        m = TinyLLM(_csa_cfg(every=every))
        first = _full_layers(m)[0]
        assert not getattr(m.layers[first].attn, "csa_reuse", False)


# --------------------------- 参数共享 ---------------------------
def test_csa_shares_kv_downproj_only():
    """Reuse 层与最近 Full 层共享 KV 下投影，但不共享上投影."""
    m = TinyLLM(_csa_cfg(every=2))
    full, reuse = m.layers[0].attn, m.layers[2].attn
    assert reuse.w_dkv is full.w_dkv
    assert reuse.w_kr is full.w_kr
    # 上投影/Q/输出仍各层独立（Reuse 只借 latent，不借上投影）
    assert reuse.w_uk is not full.w_uk
    assert reuse.w_uv is not full.w_uv
    assert reuse.w_dq is not full.w_dq
    assert reuse.w_o is not full.w_o


def test_csa_sharing_reduces_param_count():
    """共享下投影 → 总参数比同形状非共享配置少（无死参数重复计数）."""
    torch.manual_seed(0)
    base = TinyLLM(tiny_test_config())
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    total_base = sum(p.numel() for p in base.parameters())
    total_csa = sum(p.numel() for p in m.parameters())
    full = m.layers[0].attn
    per = (sum(p.numel() for p in full.w_dkv.parameters())
           + sum(p.numel() for p in full.w_kr.parameters()))
    assert total_base - total_csa == len(_reuse_layers(m)) * per


def test_csa_state_dict_round_trip():
    """state_dict 存读一致（共享权重对象可正常序列化/载入）."""
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    m.eval()
    m2 = TinyLLM(_csa_cfg(every=2))
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        assert torch.equal(m(x)["logits"], m2(x)["logits"])


# --------------------------- 解码 == 全前向 ---------------------------
def _decode_matches_full(cfg):
    torch.manual_seed(0)
    m = TinyLLM(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 6))
    toks = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        _, pasts = m._prefill(x, max_new_tokens=toks.shape[1])
        hs = []
        for i in range(toks.shape[1]):
            hh, pasts = m._decode_step(toks[:, i:i + 1], pasts)
            hs.append(hh)
        seq = torch.cat(hs, dim=1)
        full = m.encode_full_hidden(torch.cat([x, toks], dim=1))
    return seq, full[:, -toks.shape[1]:]


def test_csa_decode_matches_full_forward():
    """逐 token 解码（Reuse 层只读共享缓冲）与全前向末段一组紧公差内一致."""
    seq, ref = _decode_matches_full(_csa_cfg(every=2))
    assert torch.allclose(seq, ref, atol=1e-5), (seq - ref).abs().max()


def test_csa_decode_matches_full_forward_with_hyper():
    """超连接 + CSA2：多流展开/combine 下解码仍与全前向一致."""
    cfg = _csa_cfg(every=2)
    cfg.hyper_streams = 2
    seq, ref = _decode_matches_full(cfg)
    assert torch.allclose(seq, ref, atol=1e-5), (seq - ref).abs().max()


def test_csa_decode_multi_matches_sequential():
    """并行验证（_decode_multi）与逐 token 解码一致（Reuse 层按位读共享缓冲）."""
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    m.eval()
    x = torch.randint(0, 256, (1, 6))
    toks = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        _, pa = m._prefill(x, max_new_tokens=toks.shape[1])
        _, pb = m._prefill(x, max_new_tokens=toks.shape[1])
        hs = []
        for i in range(toks.shape[1]):
            hh, pa = m._decode_step(toks[:, i:i + 1], pa)
            hs.append(hh)
        seq = torch.cat(hs, dim=1)
        multi, _, _ = m._decode_multi(toks, pb)
    assert torch.allclose(seq, multi, atol=1e-5), (seq - multi).abs().max()


def test_csa_generate_runs_and_is_causal():
    """生成可跑；且改末位 token 不改前部 hidden（Reuse 借 latent 仍严格因果）."""
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        assert m.generate(x, max_new_tokens=6).shape == (1, 14)
        h1 = m.encode_full_hidden(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 256  # 只改末位
        h2 = m.encode_full_hidden(x2)
    # 因果：末位之前的所有位置输出不变
    assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-6)


def test_csa_reuse_actually_changes_math():
    """Reuse 层确实借共享 latent（输出与纯 Full 不同，证明接线非恒等）."""
    torch.manual_seed(0)
    base = TinyLLM(tiny_test_config())
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    m.load_state_dict(base.state_dict())  # 非共享上投影权重对齐
    base.eval(); m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        a = base(x)["logits"]
        b = m(x)["logits"]
    assert not torch.allclose(a, b, atol=1e-4)


# --------------------------- 投机 / 训练 ---------------------------
def test_csa_spec_matches_greedy():
    """开 CSA2 + 投机解码：投机输出与逐 token 贪心逐位一致."""
    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
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


def test_csa_train_step_finite():
    """一步训练：loss 有限、共享下投影拿到梯度（Reuse 借的正是它）."""
    from src.llm.local.train import build_optimizer, train_step

    torch.manual_seed(0)
    m = TinyLLM(_csa_cfg(every=2))
    m.train()
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    info = train_step(m, opt, x)
    assert torch.isfinite(torch.tensor(info["loss"]))
    shared = m.layers[0].attn.w_dkv.weight
    assert shared.grad is not None
    assert torch.isfinite(shared.grad).all()


def test_csa_train_step_finite_with_grad_ckpt():
    """梯度检查点 + CSA2：Full 层返回缓存供 Reuse 借，检查点路径仍有限可训."""
    from src.llm.local.train import build_optimizer, train_step

    torch.manual_seed(0)
    cfg = _csa_cfg(every=2)
    cfg.grad_ckpt = True
    m = TinyLLM(cfg)
    m.train()
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    info = train_step(m, opt, x)
    assert torch.isfinite(torch.tensor(info["loss"]))


# --------------------------- 缓存收益 ---------------------------
def test_csa_kv_cache_reduced():
    """Reuse 层不新增 KV 缓存 → 满上下文缓存字节显著下降."""
    base = TinyLLM(tiny_test_config())
    m = TinyLLM(_csa_cfg(every=2))
    seq = 4096
    assert m.kv_cache_bytes(seq) < base.kv_cache_bytes(seq)
    # tiny: 2 个全注意力层里 1 个 Reuse → 全注意力缓存减半
    mla = base.config
    per_layer = (mla.kv_lora_rank + mla.qk_rope_dim) * seq * 2
    assert base.kv_cache_bytes(seq) - m.kv_cache_bytes(seq) == per_layer
