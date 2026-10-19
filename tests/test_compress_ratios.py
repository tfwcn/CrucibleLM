"""主 KV 逐层压缩比单测 — `compress_ratios`（DeepSeek-V4.1 `CSA2(ratio, mode)`）.

CPU 可跑。核心契约：
- m=1（默认/空表）**逐位等价**旧行为，且不新增 `state_dict` 键；
- m = 每条 main KV 条目聚合的连续 token 数（**不重叠**），远端按 m 块池化、
  本地窗口与当前块保持未压缩；块尾位置作 rope 代表位置；
- 压缩比只按全注意力层循环分配，非全注意力层恒为 1；
- 借用层（reuse/reindex）沿用**生产者**的压缩比与池化权重（同一对象，无死参数）；
- 压缩开启后恒走分块路径（`sparse_threshold` 不再影响结果）；
- 因果（不看未来 token）+ Finite 梯度；`prefill + 增量解码 == 全前向`（chunk 对齐）。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    _norm_compress_ratios,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.sparse_attn import (  # noqa: E402
    SparseMLAModule,
    select_block_keys,
)


def _cfg(**over) -> SmallLLMConfig:
    """小模型：4 层、每 2 层 1 个全注意力层（层 0/2）+ 小 chunk（便于 chunk 对齐解码）."""
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=2,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=256, yarn_scale=1.0,
        sparse_threshold=4, sparse_window=4, sparse_sink=2,
        sparse_stride=2, sparse_chunk=4,
        sparse_indexer=False, csa_reuse_every=1, grad_ckpt=False,
    )
    base.update(over)
    return SmallLLMConfig(**base)


# ------------------------------- m=1：逐位等价 -------------------------------
def test_ratio_one_is_bitwise_identical_and_adds_no_keys():
    """空表 / "1" / "1,1" 三种写法与不开压缩：键相同、输出逐位相同."""
    torch.manual_seed(0)
    base = TinyLLM(_cfg())
    ids = torch.randint(0, 256, (2, 16))
    base.eval()
    with torch.no_grad():
        ref = base.encode_full_hidden(ids)
    for txt in ("", "1", "1,1"):
        torch.manual_seed(0)
        m = TinyLLM(_cfg(compress_ratios=txt))
        assert set(m.state_dict()) == set(base.state_dict())
        assert not any("z_cmp" in k for k in m.state_dict())
        m.eval()
        with torch.no_grad():
            out = m.encode_full_hidden(ids)
        assert torch.equal(out, ref)


def test_ratio_assignment_by_full_attn_layer_and_keys():
    """压缩比按全注意力层循环；非全注意力层不受影响；m>1 新增 z_cmp 键."""
    cfg = _cfg(compress_ratios="2")
    assert [cfg.compress_ratio_of(i) for i in range(4)] == [2, 1, 2, 1]
    assert cfg.compress_ratio_list() == (2,)
    cfg2 = _cfg(compress_ratios="2,4")
    assert [cfg2.compress_ratio_of(i) for i in range(4)] == [2, 1, 4, 1]
    m = TinyLLM(cfg)
    assert m.layers[0].attn.compress_ratio == 2
    assert m.layers[2].attn.compress_ratio == 2
    assert m.layers[0].attn.z_cmp.shape == (2, cfg.kv_lora_rank)
    assert not hasattr(m.layers[1].attn, "compress_ratio")   # 线性层不受影响
    assert any("z_cmp" in k for k in m.state_dict())
    m1 = TinyLLM(_cfg())
    assert not (set(m.state_dict()) - set(m1.state_dict())) == set()


def test_validation_errors():
    """非法压缩比早失败（0/负数/非整数）."""
    assert _norm_compress_ratios("") == ""
    assert _norm_compress_ratios("2, 1") == "2,1"
    assert _norm_compress_ratios([2, 1, 2]) == "2,1,2"      # 保序/保重复（循环语义）
    with pytest.raises(ValueError, match="非法压缩比"):
        _norm_compress_ratios("0")
    with pytest.raises(ValueError, match="非法压缩比"):
        _norm_compress_ratios("-2")
    with pytest.raises(ValueError, match="非整数"):
        _norm_compress_ratios("a")


# --------------------------- 池化数学与块选择 ---------------------------
def _mod(ratio: int = 2, **over) -> SparseMLAModule:
    cfg = _cfg(**over)
    mod = SparseMLAModule(
        cfg.d_model, cfg.n_heads, cfg.q_lora_rank, cfg.kv_lora_rank,
        cfg.qk_rope_dim, cfg.max_seq_len, 10000.0, 1.0, 0.0,
        cfg.sparse_threshold, cfg.sparse_window, cfg.sparse_sink,
        cfg.sparse_stride, cfg.sparse_chunk, compress_ratio=ratio)
    mod.eval()
    return mod


def test_pooled_block_is_mean_at_init_then_learned():
    """z=1 时块值 = 块内算术平均；改 z 后按权重加权；rope 取块尾位置."""
    mod = _mod(2)
    b, t, r, rope = 1, 8, mod.kv_lora_rank, mod.qk_rope_dim
    torch.manual_seed(0)
    c = torch.randn(b, t, r)
    kr = torch.randn(b, t, rope)
    blk = torch.tensor([0, 2])          # 块 b 覆盖 [b*m, (b+1)*m)
    c_blk, k, v, blk_end = mod._compressed_columns(c, kr, blk)
    expect = torch.stack([c[:, 0:2].mean(1), c[:, 4:6].mean(1)], dim=1)
    assert torch.allclose(c_blk, expect, atol=1e-6)
    assert blk_end.tolist() == [1, 5]
    # 代表 rope = 块尾位置的 rope（经 rope 旋转）
    from src.llm.local.rope import apply_rope
    want_kr = apply_rope(kr[:, [1, 5]], mod.rope_cos[blk_end], mod.rope_sin[blk_end])
    assert k.shape == (b, mod.n_heads, 2, mod.qk_rope_dim + mod.head_dim)
    assert torch.allclose(k[:, 0, :, mod.head_dim:], want_kr, atol=1e-6)
    # 手工设置 z：块值 = Σ z_i ⊙ c_i / m
    with torch.no_grad():
        mod.z_cmp.copy_(torch.tensor([[2.0] * r, [0.0] * r]))
    c_blk2, _, _, _ = mod._compressed_columns(c, kr, blk)
    assert torch.allclose(c_blk2[:, 0], c[:, 0], atol=1e-6)
    assert torch.allclose(c_blk2[:, 1], c[:, 4], atol=1e-6)


def test_far_block_ids_are_complete_and_causal():
    """远端块：整块在窗口之前、且整块在查询之前（绝不看未来）."""
    mod = _mod(4)
    blk = mod._far_block_ids(total=64, start=40)
    assert blk.numel() > 0
    ends = (blk + 1) * 4 - 1
    assert int(ends.max()) < 40                     # 因果：块尾 < 查询块起点
    assert int(ends.max()) < 40 - mod.sparse_window + 4   # 只在窗口之前
    assert blk.tolist() == sorted(blk.tolist())
    # 窗口覆盖全部历史（start-window=0）→ 没有远端块
    assert mod._far_block_ids(total=8, start=4).numel() == 0
    # 窗口只覆盖 [start-window, start)，更早的整块仍要被压缩列覆盖（无缝）
    assert mod._far_block_ids(total=8, start=8).tolist() == [0]
    # 块完整：块尾 < total
    assert int(ends.max()) < 64


def test_block_pattern_covers_more_history_per_budget():
    """同样列数预算下，远端列覆盖的 token 数 = 列数 × m（压缩比的核心收益）."""
    ids1 = select_block_keys(64, sink=8, stride=16, m=1)
    ids4 = select_block_keys(64, sink=8, stride=16, m=4)
    assert ids1.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 16, 32, 48]
    # m=4：sink 块 = ceil(8/4)=2；stride 块 = 16//4 = 4
    assert ids4.tolist() == [0, 1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # 每个块 token 覆盖：m=4 覆盖 4 个 token，故"代表的历史"约 m×
    assert len(ids4) * 4 >= len(ids1) * 1
    the_end_1 = int(ids1.max()) * 1 + 1
    the_end_4 = int(ids4.max()) * 4 + 4
    assert the_end_4 > the_end_1 * 2


# ----------------------------- 前向语义/因果 -----------------------------
def test_far_columns_are_actually_read():
    """压缩列真参与注意力：改未入选块不影响输出，改入选块必须影响输出."""
    # stride=6（→ stride 块 = 3）使开头几块里存在"未被 pattern 选中"的远端块
    mod = _mod(2, sparse_stride=6)
    b, total, start, end = 1, 32, 24, 28
    torch.manual_seed(0)
    c = torch.randn(b, total, mod.kv_lora_rank)
    kr = torch.randn(b, total, mod.qk_rope_dim)
    k_full, v_full = mod._kv_from_latent(c, kr, total=total)
    q = torch.randn(b, mod.n_heads, end - start, mod.head_dim + mod.qk_rope_dim)
    h = torch.randn(b, end - start, 128)
    out = mod._sparse_scores(q, k_full, v_full, start, end, total,
                             c_kv=c, h_c=h, kr_raw=kr)[0]
    far = mod._far_block_ids(total, start)
    assert far.numel() > 0
    # 未入选块的成员（找一个不在 far 里的块，且不在窗口 [start-window, end) 内）
    chosen = [bl for bl in range(total // 2)
              if bl not in far.tolist() and (bl + 1) * 2 <= start - mod.sparse_window]
    assert chosen, "需要至少一个未被 pattern 选中的远端块"
    c2 = c.clone()
    c2[:, chosen[0] * 2:chosen[0] * 2 + 2] += 5.0
    k2, v2 = mod._kv_from_latent(c2, kr, total=total)
    out2 = mod._sparse_scores(q, k2, v2, start, end, total,
                              c_kv=c2, h_c=h, kr_raw=kr)[0]
    assert torch.allclose(out, out2, atol=1e-6)
    # 入选块改值 → 输出必须变
    c3 = c.clone()
    c3[:, far[0] * 2:far[0] * 2 + 2] += 5.0
    k3, v3 = mod._kv_from_latent(c3, kr, total=total)
    out3 = mod._sparse_scores(q, k3, v3, start, end, total,
                              c_kv=c3, h_c=h, kr_raw=kr)[0]
    assert not torch.allclose(out, out3, atol=1e-6)


def test_causality_and_finite_grads():
    """改未来 token 前部输出不变；压缩 + indexer 梯度全 finite."""
    for idx_on in (False, True):
        torch.manual_seed(0)
        m = TinyLLM(_cfg(compress_ratios="2", sparse_indexer=idx_on,
                         indexer_heads=4, indexer_head_dim=32, indexer_topk=2,
                         indexer_block=4, indexer_pool_blocks=1))
        m.eval()
        ids = torch.randint(0, 256, (1, 32))
        ids2 = ids.clone()
        ids2[:, 28:] = (ids2[:, 28:] + 7) % 256
        with torch.no_grad():
            a = m.encode_full_hidden(ids)
            b = m.encode_full_hidden(ids2)
        assert torch.equal(a[:, :24], b[:, :24])
        assert not torch.allclose(a[:, 24:], b[:, 24:], atol=1e-6)
        m.train()
        o = m(ids, targets=ids)
        assert torch.isfinite(o["loss"])
        o["loss"].backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        z = m.layers[0].attn.z_cmp.grad
        assert z is not None and float(z.abs().sum()) > 0   # 池化权重真在学


def test_decode_matches_full_when_chunk_aligned():
    """prefill + 顺序解码（按 chunk 对齐）== 全前向；压缩开启时仍成立."""
    for over in (dict(), dict(csa_modes="full,reuse"),
                 dict(csa_modes="full,reindex,reuse", sparse_indexer=True,
                      indexer_heads=4, indexer_head_dim=32, indexer_topk=2,
                      indexer_block=4, indexer_pool_blocks=1)):
        torch.manual_seed(0)
        m = TinyLLM(_cfg(compress_ratios="2", **over))
        m.eval()
        ids = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            full = m.encode_full_hidden(ids)
            pre, pasts = m._prefill(ids[:, :16], max_new_tokens=16)
            outs = []
            for s in range(16, 32, 4):
                h, pasts = m._decode_step(ids[:, s:s + 4], pasts)
                outs.append(h)
        tail = torch.cat(outs, dim=1)
        assert torch.allclose(pre, full[:, :16], atol=1e-4)
        assert torch.allclose(tail, full[:, 16:32], atol=2e-3), (
            float((tail - full[:, 16:32]).abs().max()))


def test_compression_forces_block_path():
    """压缩开启后 `sparse_threshold` 不再影响结果（短序列也走分块路径）."""
    torch.manual_seed(0)
    a = TinyLLM(_cfg(compress_ratios="2", sparse_threshold=1))
    torch.manual_seed(0)
    b = TinyLLM(_cfg(compress_ratios="2", sparse_threshold=4096))
    ids = torch.randint(0, 256, (2, 16))
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.equal(a.encode_full_hidden(ids), b.encode_full_hidden(ids))


# ----------------------------- 借用层与来源对齐 -----------------------------
def test_borrow_layer_inherits_producer_ratio_and_weights():
    """reuse/reindex 层沿用生产者的压缩比与 z_cmp（同一对象）；多余的不留死参数."""
    m = TinyLLM(_cfg(compress_ratios="2", csa_modes="full,reuse"))
    assert m.layers[0].attn.compress_ratio == 2
    assert m.layers[2].attn.compress_ratio == 2
    assert m.layers[2].attn.z_cmp is m.layers[0].attn.z_cmp
    # 生产者不压缩（m=1）而消费层配了 m=4 → 跟随生产者，且删掉自己的 z_cmp（无死参数）
    m2 = TinyLLM(_cfg(compress_ratios="1,4", csa_modes="full,reuse"))
    assert m2.layers[2].attn.compress_ratio == 1
    assert not hasattr(m2.layers[2].attn, "z_cmp")
    assert not any(k.endswith("z_cmp") and k.startswith("layers.2.")
                   for k in m2.state_dict())


def test_checkpoint_alignment_and_cli(tmp_path):
    """来源对齐：checkpoint 记录压缩比则沿用；读不到保守回退空表（=不压缩）."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "compress_ratios": "4"}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg2 = _cfg()
    sync_arch_from_checkpoint(cfg2, str(d / "model.pt"))
    assert cfg2.compress_ratios == "4" and cfg2.compress_ratio_of(0) == 4

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg3 = _cfg()
    sync_arch_from_checkpoint(cfg3, str(d2 / "model.pt"))
    assert cfg3.compress_ratios == "" and cfg3.compress_ratio_of(0) == 1
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).compress_ratios is None
    args = T.parse_args(["--compress-ratios", "2,1"])
    assert args.compress_ratios == "2,1"
    c = tiny_test_config()
    src = T.resolve_arch_switches(args, c)
    assert c.compress_ratios == "2,1" and src["compress_ratios"] == "explicit"
