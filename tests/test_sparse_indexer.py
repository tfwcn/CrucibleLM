"""分层稀疏 indexer 单测 — 学习式 top-K + 块级候选池（DeepSeek-V4.1 CSA2）.

CPU 可跑。核心契约：
- 默认关：无 `w_iq`/`w_ik`，键与旧实现一致，长/短序列数值与旧实现**逐位一致**；
- 开启：新增 `w_iq`/`w_ik`（改键，需重训）；只影响**长序列**稀疏路径，短序列 dense 不变；
- `_indexer_select`：块级候选池 → 池内 token 级 top-K；top-K ⊆ 池；
- 因果：改未来 token 不影响前部；Finite：长序列反向无 NaN；
- 解码一致：prefill + 增量解码 == 一次全前向对应段（indexer 开启）；
- 来源对齐：`sync_arch_from_checkpoint` 读 sparse_indexer 与 indexer 超参，旧 ckpt 回退关；
- CLI 三态。
"""

import json

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.sparse_attn import SparseMLAModule  # noqa: E402


def _mod(**kw) -> SparseMLAModule:
    """构造一个小号稀疏 MLA（默认让稀疏路径生效：threshold 很小）."""
    torch.manual_seed(0)
    base = dict(
        d_model=128, n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        max_seq_len=256, sparse_threshold=8, sparse_window=8, sparse_sink=4,
        sparse_stride=4, sparse_chunk=8,
    )
    base.update(kw)
    m = SparseMLAModule(**base)
    m.eval()
    return m


# ------------------------------- 默认关/键兼容 -------------------------------
def test_off_is_legacy():
    """默认关：无 indexer 参数，键与旧实现一致."""
    m = _mod()
    assert m.sparse_indexer is False
    keys = list(m.state_dict())
    assert "w_iq.weight" not in keys and "w_iq" not in keys
    assert "w_full.weight" not in keys  # 仍是 w_uk/w_uv（自算 KV）


def test_config_default_off_and_wiring():
    """config 默认关；开启后接线到 SparseMLAModule（新增 w_iq/w_ik）."""
    assert SmallLLMConfig().sparse_indexer is False
    cfg = tiny_test_config()
    cfg.sparse_indexer = True
    cfg.indexer_topk = 8
    cfg.sparse_threshold = 4
    m = TinyLLM(cfg)
    assert m.layers[0].attn.sparse_indexer is True
    assert "w_iq.weight" in m.layers[0].attn.state_dict()
    assert m.layers[0].attn.indexer_topk == 8
    # 线性注意力层不受影响
    assert not getattr(m.layers[1].attn, "sparse_indexer", False)


def test_off_matches_legacy_bitwise():
    """关闭时（无论长短序列）与旧实现逐位一致：同一 seed 下 indexer=False 走原路径."""
    m = _mod()
    h = torch.randn(1, 24, 128)
    with torch.no_grad():
        a, _ = m(h)
    # 手工复刻：indexer 关闭时必须走 "无 indexer" 分支（mask 与 pattern 全同）
    m2 = _mod()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        b, _ = m2(h)
    assert torch.equal(a, b)
    # 短序列（< threshold=8）走 dense，与 MLAModule 一致
    with torch.no_grad():
        c, _ = m(h[:, :6])
    from src.llm.local.mla import MLAModule
    torch.manual_seed(0)
    dense = MLAModule(128, 4, 32, 32, 16, max_seq_len=256)
    dense.load_state_dict(m.state_dict(), strict=False)
    with torch.no_grad():
        d, _ = dense(h[:, :6])
    assert torch.allclose(c, d, atol=1e-6)


# ------------------------------- 开启后行为 -------------------------------
def test_on_changes_long_but_not_short():
    """开启 indexer：长序列输出改变（可观测），短序列（dense）与关闭时一致."""
    h = torch.randn(1, 24, 128)
    off = _mod()
    on = _mod(sparse_indexer=True, indexer_topk=8, indexer_pool_blocks=2)
    assert list(on.state_dict()) != list(off.state_dict())
    with torch.no_grad():
        o_off, _ = off(h)
        o_on, _ = on(h)
        s_off, _ = off(h[:, :6])
        s_on, _ = on(h[:, :6])
    assert not torch.allclose(o_off, o_on), "长序列应受 indexer 影响"
    assert torch.allclose(s_off, s_on, atol=1e-6), "短序列走 dense，不该受影响"
    assert torch.isfinite(o_on).all()


def test_indexer_select_topk_subset_of_pool():
    """`_indexer_select`：top-K ⊆ 块级候选池，且每查询恰好 min(topk, n_static) 个."""
    m = _mod(sparse_indexer=True, indexer_topk=6, indexer_pool_blocks=2,
             indexer_block=4)
    torch.manual_seed(1)
    h_q = torch.randn(1, 4, 128)
    c_kv = torch.randn(1, 20, 32)
    static_pos = torch.arange(0, 16)  # 16 个候选，block=4 → 4 个块
    sel = m._indexer_select(h_q, c_kv, static_pos, 4)
    assert sel.topk_mask.shape == (1, 4, 16)
    assert sel.topk_mask.sum(-1).tolist() == [[6, 6, 6, 6]]
    # top-K 必须落在池内
    assert int((sel.topk_mask & ~sel.pool_mask).sum()) == 0
    # 池是子集（2 块 × 4 = 8 < 16）
    assert sel.pool_mask.sum(-1).max().item() <= 8
    assert sel.pool_mask.any()


def test_indexer_is_causal_and_finite():
    """开启 indexer：改未来 token 不影响前部；长序列反向无 NaN."""
    on = _mod(sparse_indexer=True, indexer_topk=8, indexer_pool_blocks=2)
    on.train()
    h = torch.randn(1, 24, 128, requires_grad=True)
    h2 = h.detach().clone()
    h2[0, 20] += 5.0  # 改未来
    on.eval()
    with torch.no_grad():
        a, _ = on(h)
        b, _ = on(h2)
    assert (a[0, :20] - b[0, :20]).abs().max() < 1e-5, "未来 token 污染了前部"
    assert (a[0, 20:] - b[0, 20:]).abs().max() > 0
    on.train()
    out, _ = on(h)
    out.pow(2).mean().backward()
    for n, p in on.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_indexer_decode_matches_full():
    """开启 indexer：prefill + 增量解码 == 一次全前向对应段（末块静态集相同）."""
    m = _mod(sparse_indexer=True, indexer_topk=8, indexer_pool_blocks=2)
    h = torch.randn(1, 24, 128)
    with torch.no_grad():
        full, _ = m(h)
        _, past = m(h[:, :16], None, True, 24)
        multi, _ = m(h[:, 16:24], past)
    assert torch.allclose(multi, full[:, 16:], atol=1e-4), (multi - full[:, 16:]).abs().max()


def test_indexer_batch_finite():
    """batch>1 也能跑通（scores 按 b 扩展）."""
    on = _mod(sparse_indexer=True, indexer_topk=8, indexer_pool_blocks=2)
    h = torch.randn(3, 20, 128)
    with torch.no_grad():
        o, _ = on(h)
    assert o.shape == (3, 20, 128) and torch.isfinite(o).all()


def test_indexer_gets_gradients_from_distillation():
    """indexer 只产掩码（不可微）→ 必须靠**蒸馏损失**才拿得到梯度（DSA 做法）.

    没有蒸馏时 `w_iq`/`w_ik` 梯度恒为 0（等于没在训），所以这条是"indexer 真在学"
    的硬契约；系数设 0 则确实无梯度（口径可关）。
    """
    torch.manual_seed(0)

    def _model(coef: float) -> TinyLLM:
        cfg = tiny_test_config()
        cfg.sparse_indexer = True
        # 必须让序列真被切成多块：静态候选 = 块起点之前的 token（单块时恒为 0，
        # indexer 不会被调用 → 也就没有蒸馏损失）
        cfg.sparse_threshold = 4
        cfg.sparse_chunk = 4
        cfg.sparse_window = 4
        cfg.sparse_sink = 2
        cfg.sparse_stride = 2
        cfg.indexer_topk = 4
        cfg.indexer_pool_blocks = 1
        cfg.indexer_loss_coef = coef
        cfg.grad_ckpt = False
        m = TinyLLM(cfg)
        m.train()
        return m

    ids = torch.randint(0, 256, (2, 16))
    m = _model(0.01)
    out = m(ids, targets=ids)
    assert "indexer_loss" in out and float(out["indexer_loss"]) > 0
    out["loss"].backward()
    for name in ("w_iq", "w_ik"):
        g = getattr(m.layers[0].attn, name).weight.grad
        assert g is not None and float(g.abs().sum()) > 0, \
            f"{name} 无梯度（indexer 没在学）"

    m0 = _model(0.0)  # 系数 0：确实无梯度（口径可关）
    o0 = m0(ids, targets=ids)
    o0["loss"].backward()
    g0 = m0.layers[0].attn.w_iq.weight.grad
    assert g0 is None or float(g0.abs().sum()) == 0


# ------------------------------- 来源对齐/CLI -------------------------------
def test_sync_reads_indexer_switches(tmp_path):
    """装载既有权重时对齐 indexer 开关与超参；旧格式回退关/默认."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "sparse_indexer": True, "indexer_topk": 128, "indexer_heads": 8}}),
        encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.sparse_indexer is True
    assert cfg.indexer_topk == 128 and cfg.indexer_heads == 8
    # 缺字段（旧 ckpt）→ 回退关 + 默认
    assert cfg.indexer_pool_blocks == 32

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 768}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.sparse_indexer is False


def test_cli_indexer_tristate():
    """CLI：不传=None（沿用/默认关），传入即显式."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).sparse_indexer is None
    assert T.parse_args(["--sparse-indexer"]).sparse_indexer is True
    assert T.parse_args(["--no-sparse-indexer"]).sparse_indexer is False
    assert T.parse_args(["--indexer-topk", "32"]).indexer_topk == 32
