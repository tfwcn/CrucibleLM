"""CSA2 三模式单测 — Full / Reindex / Reuse（DeepSeek-V4.1）.

CPU 可跑。核心契约：
- 模式分配：`csa_modes` 显式指定（别名、循环）；空则由 `csa_reuse_every` 推导旧行为；
- **不改 `state_dict` 键**：同一 `csa_reuse_every` 下 Full/Reuse 与 Full/Reindex 键集合相同；
- 借用层（reindex/reuse）与最近生产者共享 `w_dkv`/`w_kr`（indexer 开启时 reindex 还共享 `w_ik`）；
- Reindex ≠ Reuse（`sparse_indexer` 开启时：Reindex 借池自选 Top-K，Reuse 直接用最新 Top-K）；
- 无 indexer 时 Reindex **退化为 Reuse**（选择与 query 无关、逐层相同）→ 逐位一致；
- 因果/Finite；`prefill + 增量解码 == 全前向`；梯度检查点开关不改梯度（通道按层号键控）。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    _norm_csa_modes,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.sparse_attn import (  # noqa: E402
    CsaChannel,
    IndexerSelect,
    SparseMLAModule,
)


def _cfg(**over) -> SmallLLMConfig:
    """小模型：4 层、每 2 层 1 个全注意力层（层 0/2）→ 2 个全注意力层可放两档模式."""
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=2,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=256, yarn_scale=1.0,
        sparse_threshold=8, sparse_window=8, sparse_sink=4,
        sparse_stride=4, sparse_chunk=8,
        sparse_indexer=True, indexer_heads=4, indexer_head_dim=32,
        indexer_topk=2, indexer_block=4, indexer_pool_blocks=1,
    )
    base.update(over)
    return SmallLLMConfig(**base)


def _mod(**over) -> SparseMLAModule:
    torch.manual_seed(0)
    base = dict(
        d_model=128, n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        max_seq_len=256, sparse_threshold=8, sparse_window=8, sparse_sink=4,
        sparse_stride=4, sparse_chunk=8,
    )
    base.update(over)
    m = SparseMLAModule(**base)
    m.eval()
    return m


# ------------------------------- 模式分配 -------------------------------
def test_csa_mode_list_legacy_derivation():
    """空 `csa_modes` → 由 csa_reuse_every 推导旧行为（逐位兼容旧 checkpoint）."""
    assert SmallLLMConfig().csa_modes == ""
    assert SmallLLMConfig(csa_reuse_every=1).csa_mode_list() == ("full",)
    assert SmallLLMConfig(csa_reuse_every=2).csa_mode_list() == ("full", "reuse")
    assert SmallLLMConfig(csa_reuse_every=3).csa_mode_list() == (
        "full", "reuse", "reuse")


def test_csa_modes_explicit_and_aliases():
    """显式模式串：别名归一化 + 非法值报错（配置错误要早失败）."""
    assert _norm_csa_modes("f, ri ,re") == "full,reindex,reuse"
    assert _norm_csa_modes("Full;reuse") == "full,reuse"
    assert _norm_csa_modes(["full", "REINDEX"]) == "full,reindex"
    assert _norm_csa_modes("") == ""
    assert _norm_csa_modes(None) == ""
    with pytest.raises(ValueError, match="非法模式"):
        _norm_csa_modes("full,half")
    cfg = SmallLLMConfig(n_layers=6, full_attn_every=2, csa_reuse_every=1,
                         csa_modes="full,reindex,reuse")
    assert cfg.csa_mode_list() == ("full", "reindex", "reuse")
    # 非全注意力层恒为 full（不受影响）；全注意力层按序号循环
    assert [cfg.csa_mode_of(i) for i in range(6)] == [
        "full", "full", "reindex", "full", "reuse", "full"]


def test_modes_do_not_change_state_dict_keys():
    """关键契约：模式分配**不改键**——同 csa_reuse_every 下两种分配键集合相同."""
    legacy = TinyLLM(_cfg(csa_reuse_every=2, csa_modes=""))
    reindex = TinyLLM(_cfg(csa_reuse_every=2, csa_modes="full,reindex"))
    reuse = TinyLLM(_cfg(csa_reuse_every=2, csa_modes="full,reuse"))
    k0 = set(legacy.state_dict())
    assert k0 == set(reindex.state_dict()) == set(reuse.state_dict())
    # 分配正确：层 0 = full（生产者），层 2 = 指定模式
    assert legacy.layers[0].attn.csa_mode == "full"
    assert legacy.layers[2].attn.csa_mode == "reuse"
    assert reindex.layers[2].attn.csa_mode == "reindex"
    # 借用者 `csa_reuse=True`（解码不写自己的缓冲 → 不计 KV 缓存）
    assert reuse.layers[2].attn.csa_reuse is True
    assert reindex.layers[2].attn.csa_reuse is True
    assert reuse.layers[0].attn.csa_reuse is False


def test_borrowers_share_projections_from_producer():
    """借用层与最近生产者共享 `w_dkv`/`w_kr`；Reindex 连 `w_ik` 也借（只自带 `w_iq`）."""
    m = TinyLLM(_cfg(csa_modes="full,reindex"))
    prod, cons = m.layers[0].attn, m.layers[2].attn
    assert cons.w_dkv is prod.w_dkv and cons.w_kr is prod.w_kr
    assert cons.w_ik is prod.w_ik          # V4.1：Reindex 复用 main KV + indexer K
    assert cons.w_iq is not prod.w_iq      # …但自带自己的 indexer Q
    assert cons.csa_src == 0 and cons.csa_index == 2
    m2 = TinyLLM(_cfg(csa_modes="full,reuse"))
    u = m2.layers[2].attn
    assert u.w_dkv is m2.layers[0].attn.w_dkv and u.csa_mode == "reuse"


def test_first_layer_degrades_to_full():
    """首层被指定为借用者时退化为 Full（保证有共享源，不能没生产者）."""
    m = TinyLLM(_cfg(csa_modes="reuse,reuse"))
    assert m.layers[0].attn.csa_mode == "full"
    assert m.layers[2].attn.csa_mode == "reuse"
    assert m._csa_active is True
    assert m.layers[2].attn.csa_src == 0


def test_modes_off_means_all_full():
    """`csa_reuse_every=1` 且空 `csa_modes` → 全 Full（无共享，逐位兼容旧实现）."""
    m = TinyLLM(_cfg(csa_reuse_every=1, csa_modes=""))
    assert m._csa_active is False
    assert all(getattr(L.attn, "csa_mode", "full") == "full" for L in m.layers)


# ------------------------------- 选择通道语义 -------------------------------
def test_channel_reuse_takes_shared_topk_reindex_picks_its_own():
    """通道：Reuse 原样取共享 Top-K（同对象）；Reindex 借池自选（收窄到池交集）."""
    torch.manual_seed(1)
    # 单块（block=8 ≥ 候选数）→ 自身池 = 全部候选；再 ∩ 共享池 = 前 2 个候选
    shared = IndexerSelect(
        topk_mask=torch.zeros(1, 3, 8, dtype=torch.bool),
        pool_mask=torch.zeros(1, 3, 8, dtype=torch.bool))
    shared.topk_mask[:, :, :2] = True
    shared.pool_mask[:, :, :2] = True
    h_q = torch.randn(1, 3, 128)
    c_kv = torch.randn(1, 20, 32)
    pos = torch.arange(0, 8)
    m = _mod(sparse_indexer=True, indexer_topk=2, indexer_block=8,
             indexer_pool_blocks=1)
    m.csa_mode = "reindex"
    sel = m._indexer_select(h_q, c_kv, pos, 3, shared)
    assert sel is not shared, "Reindex 必须重算，不能照抄"
    # 打分域被收窄到共享池 → 池与 Top-K 都落在前 2 个候选内
    assert sel.pool_mask.sum(-1).tolist() == [[2, 2, 2]]
    assert sel.topk_mask.sum(-1).tolist() == [[2, 2, 2]]
    assert int((sel.topk_mask & ~shared.pool_mask).sum()) == 0


def test_channel_reuse_mode_passes_shared_object():
    """Reuse 模式：`_sparse_scores` 直接返回共享选择（不算 indexer）."""
    torch.manual_seed(2)
    m = _mod(sparse_indexer=True, indexer_topk=2)
    m.csa_mode = "reuse"
    shared = IndexerSelect(torch.zeros(1, 8, 8, dtype=torch.bool),
                           torch.ones(1, 8, 8, dtype=torch.bool))
    shared.topk_mask[:, :, 0] = True
    out = m._sparse_scores(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 24, 16),
                           torch.randn(1, 4, 24, 16), 8, 16, 24,
                           c_kv=torch.randn(1, 24, 32),
                           h_c=torch.randn(1, 8, 128), shared=shared)
    assert out[1] is shared and out[2] is None  # 复用同一对象、无蒸馏损失


def test_channel_keyed_by_layer_index():
    """通道按层号键控：互不覆盖（梯度检查点重算安全的根因）."""
    ch = CsaChannel()
    a = IndexerSelect(torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
    b = IndexerSelect(torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    ch.put(2, 0, a)
    ch.put(0, 0, b)
    assert ch.get(2, 0) is a and ch.get(0, 0) is b
    assert ch.get(4, 0) is None and ch.get(None, 0) is None
    assert ch.get(2, 5) is None  # 未写的块号


def test_sparse_scores_reuse_uses_shared_mask():
    """`_sparse_scores`：reuse 模式下静态列只留共享 top-K（非入选列改了也不影响输出）."""
    m = _mod(sparse_indexer=True, indexer_topk=2)
    m.csa_mode = "reuse"
    h = torch.randn(1, 24, 128)
    c_kv = torch.randn(1, 24, 32)
    k_full = torch.randn(1, 4, 24, 16)
    v_full = torch.randn(1, 4, 24, 16)
    q_full = torch.randn(1, 4, 8, 16)
    from src.llm.local.sparse_attn import select_keys

    idx, n_static = select_keys(24, 8, 16, 4, 8, 4)
    shared = IndexerSelect(
        topk_mask=torch.zeros(1, 8, n_static, dtype=torch.bool),
        pool_mask=torch.ones(1, 8, n_static, dtype=torch.bool))
    sel_col = 3
    shared.topk_mask[:, :, sel_col] = True
    out, sel, kl = m._sparse_scores(q_full, k_full, v_full, 8, 16, 24,
                                    c_kv=c_kv, h_c=h[:, 8:16], shared=shared)
    assert sel is shared and kl is None and torch.isfinite(out).all()
    # 非入选静态列（第 0 号候选）随意改值 → 输出不变（"只读 top-K"）
    other = int(idx[0])  # 静态第 0 号候选（未被选中）
    assert not bool(shared.topk_mask[0, 0, 0])
    k2, v2 = k_full.clone(), v_full.clone()
    k2[:, :, other, :] = 99.0
    v2[:, :, other, :] = -99.0
    out2, _, _ = m._sparse_scores(q_full, k2, v2, 8, 16, 24,
                                  c_kv=c_kv, h_c=h[:, 8:16], shared=shared)
    assert torch.allclose(out, out2, atol=1e-6)
    # 入选列（第 3 号）改了必须影响输出（否则测试是空转）
    chosen = int(idx[sel_col])
    assert chosen != other
    k3 = k_full.clone()
    k3[:, :, chosen, :] += 1.0
    out3, _, _ = m._sparse_scores(q_full, k3, v_full, 8, 16, 24,
                                  c_kv=c_kv, h_c=h[:, 8:16], shared=shared)
    assert (out3 - out).abs().max() > 1e-4


# ------------------------------- Reindex vs Reuse -------------------------------
def _same_weights(a: TinyLLM, b: TinyLLM) -> None:
    b.load_state_dict(a.state_dict())


def test_reindex_differs_from_reuse_with_indexer():
    """有 indexer：Reindex（借池自选）与 Reuse（照抄最新 Top-K）输出不同."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 32))
    re = TinyLLM(_cfg(csa_modes="full,reindex"))
    ru = TinyLLM(_cfg(csa_modes="full,reuse"))
    _same_weights(re, ru)
    re.eval(); ru.eval()
    with torch.no_grad():
        h_re = re.encode_full_hidden(ids)
        h_ru = ru.encode_full_hidden(ids)
    assert torch.isfinite(h_re).all()
    assert (h_re - h_ru).abs().max() > 1e-6, "Reindex 与 Reuse 不应完全相同"


def test_reindex_equals_reuse_without_indexer():
    """无 indexer：Reindex 退化为 Reuse（选择与 query 无关）→ 逐位一致."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 32))
    re = TinyLLM(_cfg(csa_modes="full,reindex", sparse_indexer=False))
    ru = TinyLLM(_cfg(csa_modes="full,reuse", sparse_indexer=False))
    _same_weights(re, ru)
    re.eval(); ru.eval()
    with torch.no_grad():
        assert torch.equal(re.encode_full_hidden(ids), ru.encode_full_hidden(ids))


def test_modes_are_causal_and_finite():
    """因果（改未来 token 不影响前部）+ Finite（无 NaN 梯度）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(csa_modes="full,reindex"))
    m.train()
    ids = torch.randint(0, 256, (1, 32))
    ids2 = ids.clone()
    m.eval()
    with torch.no_grad():
        a = m.encode_full_hidden(ids)
        ids2[0, 28] = (ids2[0, 28] + 7) % 256
        b = m.encode_full_hidden(ids2)
    assert (a[0, :28] - b[0, :28]).abs().max() < 1e-6, "未来 token 污染前部"
    assert (a[0, 28:] - b[0, 28:]).abs().max() > 0
    m.train()
    out = m(ids, targets=ids)
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_modes_decode_matches_full():
    """三模式 + indexer：prefill + 增量解码 == 一次全前向（末段）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(csa_modes="full,reindex"))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        pre, pasts = m._prefill(ids[:, :16], max_new_tokens=8)
        assert pre.shape == (1, 16, 128)
        tail, _ = m._decode_step(ids[:, 16:24], pasts)
    assert torch.allclose(tail, full[:, 16:], atol=1e-4), \
        (tail - full[:, 16:]).abs().max()


def test_grad_ckpt_matches_plain_grads():
    """梯度检查点开关不改梯度（选择通道按层号键控，反向重算幂等）."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 32))
    a = TinyLLM(_cfg(csa_modes="full,reindex"))
    b = TinyLLM(_cfg(csa_modes="full,reindex", grad_ckpt=True))
    _same_weights(a, b)
    a.train(); b.train()
    a(ids, targets=ids)["loss"].backward()
    b(ids, targets=ids)["loss"].backward()
    ga = a.layers[0].attn.w_iq.weight.grad
    gb = b.layers[0].attn.w_iq.weight.grad
    assert ga is not None and gb is not None
    assert torch.allclose(ga, gb, atol=1e-6), (ga - gb).abs().max()


def test_kv_cache_counts_producers_only():
    """KV 缓存只算生产者（full）；reindex/reuse 借用不新增缓存."""
    m = TinyLLM(_cfg(csa_modes="full,reuse"))
    c = m.config
    n_lin = c.n_layers - c.n_full_layers
    mla = (c.kv_lora_rank + c.qk_rope_dim) * 1024 * 2
    lin = c.n_heads * (c.d_model // c.n_heads) ** 2 * 2
    assert m.kv_cache_bytes(1024) == 1 * mla + n_lin * lin


# ------------------------------- 来源对齐 / CLI -------------------------------
def test_sync_reads_csa_modes(tmp_path):
    """装载既有权重：对齐 csa_modes；旧格式回退空（旧行为）."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "csa_modes": "full,reindex,reuse"}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.csa_modes == "full,reindex,reuse"
    assert cfg.csa_mode_list() == ("full", "reindex", "reuse")

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 768}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.csa_modes == ""


def test_cli_csa_modes_and_resolver(tmp_path):
    """CLI `--csa-modes` 三态（不传=None）+ 通用来源对齐取 explicit."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).csa_modes is None
    assert T.parse_args(["--csa-modes", "f,ri,r"]).csa_modes == "f,ri,r"
    cfg = tiny_test_config()
    src = T.resolve_arch_switches(T.parse_args(["--csa-modes", "full,reuse"]), cfg)
    assert cfg.csa_modes == "full,reuse" and src["csa_modes"] == "explicit"
    # indexer 超参 CLI 必须真的落进 config（曾经定义了却没接线）
    cfg2 = tiny_test_config()
    changed = T.apply_indexer_overrides(
        T.parse_args(["--indexer-heads", "8", "--indexer-loss-coef", "0.05"]), cfg2)
    assert changed == ["indexer_heads", "indexer_loss_coef"]
    assert cfg2.indexer_heads == 8 and cfg2.indexer_loss_coef == 0.05
    cfg3 = tiny_test_config()
    assert T.apply_indexer_overrides(T.parse_args([]), cfg3) == []
    assert (cfg3.indexer_heads, cfg3.indexer_topk, cfg3.indexer_loss_coef) == (
        tiny_test_config().indexer_heads, tiny_test_config().indexer_topk,
        tiny_test_config().indexer_loss_coef)

    # checkpoint 沿用
    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "csa_modes": "full,reindex"}}), encoding="utf-8")
    cfg2 = tiny_test_config()
    src2 = T.resolve_arch_switches(T.parse_args(["--resume", "--ckpt-dir", str(d)]), cfg2)
    assert cfg2.csa_modes == "full,reindex" and src2["csa_modes"] == "checkpoint"
