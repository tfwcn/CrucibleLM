"""块主序主 KV 缓存单测 — `kv_block_cache`（DeepSeek-V4.1 §3.2 CSA2 的显存半边）.

核心契约：
- **只改缓存布局，不改前向数学**：`compress_ratio=m` 的层把每 m 个 token 折成
  一行（块行数 `L -> ceil(L/m)`，外加定长 m 行暂存前缀），同权重下预填充/解码
  的输出与"逐 token 缓存"逐位一致（0.0）；
- 行值 = `Σ_{i<m} z_i ⊙ c_i / m`（与 `_compressed_columns` 池化**同一算式**），
  rope 行存块尾那条未旋转键；
- **残块不可读**：块没凑满 m 个 token 前只进暂存前缀，对任何查询都不可见
  （近端由层内 SWA 承担）；
- 预填充（一次长 prompt）与解码（逐步）走**同一条写路径**（`_block_add`），
  量化缓存下块均值**只被量化一次** → 跨残块的逐 token 解码仍与全前向一致；
- 暂存前缀在缓存**头部**，`ensure_room` 尾部扩容不搬动它（残块不丢）；
- 构建期拒绝缺前提的组合（无 SWA / 无 m>1 / `swa_window < m`）；
- 不改 `state_dict` 键；`kv_cache_bytes` 按 `m + ceil(L/m)` 行计费。
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    _ARCH_SWITCHES,
    SmallLLMConfig,
    sync_arch_from_checkpoint,
)
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.session_cache import SessionCache  # noqa: E402


def _cfg(**over) -> SmallLLMConfig:
    """小模型：4 层全注意力、压缩比 2、SWA 窗口 8、chunk=4（块边界 8 对齐 4）."""
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=1,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=512, yarn_scale=1.0,
        sparse_threshold=4, sparse_window=4, sparse_sink=2,
        sparse_stride=2, sparse_chunk=4,
        sparse_indexer=False, csa_reuse_every=1, grad_ckpt=False,
        swa_window=8, compress_ratios="2",
    )
    base.update(over)
    return SmallLLMConfig(**base)


def _rows(past) -> int:
    """某层主 KV 缓存的**行数**（量化缓存走 logical_shape）."""
    c = past[0]
    return c.shape[1] if hasattr(c, "shape") else c.logical_shape[1]


def _pair(**over):
    """同权重的一对模型：block 关 / 开（用于"只改布局不改数学"的逐位对照）."""
    torch.manual_seed(1)
    a = TinyLLM(_cfg(**over))
    b = TinyLLM(_cfg(kv_block_cache=True, **over))
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    return a, b


def _ids(n: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 256, (1, n), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- 构建期前提


def test_rejects_invalid_combinations():
    """缺前提的组合构建期就失败（否则静默丢上下文/无收益）."""
    with pytest.raises(ValueError, match="swa_window"):
        TinyLLM(_cfg(kv_block_cache=True, swa_window=0))
    with pytest.raises(ValueError, match="m>1"):
        TinyLLM(_cfg(kv_block_cache=True, compress_ratios=""))
    with pytest.raises(ValueError, match="swa_window >= max"):
        TinyLLM(_cfg(kv_block_cache=True, compress_ratios="8", swa_window=4))


def test_no_new_state_dict_keys():
    """只改缓存布局：`state_dict` 键与不开块主序时完全相同."""
    a = TinyLLM(_cfg())
    b = TinyLLM(_cfg(kv_block_cache=True))
    assert list(a.state_dict()) == list(b.state_dict())


def test_block_cache_only_on_compressed_layers():
    """m=1 的层没有块可折 → 该层仍是逐 token 行（混合压缩比时逐层不同）."""
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="1,2"))
    rows = [_rows(p) for p in m._prefill(_ids(32), max_new_tokens=8)[1]]
    # 全注意力层 0/2: m=1 → 40 行（=reserve）；层 1/3: m=2 → 2 + ceil(40/2)=22
    assert rows == [40, 22, 40, 22]


# ---------------------------------------------------------------- 布局与行值


def test_row_count_is_l_over_m():
    """行数 = `m`（暂存前缀）+ `ceil(reserve/m)`：这是"省显存"的唯一依据."""
    for m_ratio in (2, 4):
        m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios=str(m_ratio),
                         swa_window=8))
        pasts = m._prefill(_ids(32), max_new_tokens=8)[1]
        # reserve = 32 + 8 = 40；全注意力层 4 个，线性层为 3 元组 (S, k, v)
        full = [p for p in pasts if len(p) == 6 or len(p) == 3]
        n_rows = {_rows(p) for p in full if len(p) == 6}
        assert n_rows == {m_ratio + (40 + m_ratio - 1) // m_ratio}


def test_block_row_equals_weighted_mean():
    """块行 = `Σ z_i⊙c_i/m`，rope 行 = 块尾 token 的未旋转键；残块进暂存前缀."""
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8))
    attn = m.layers[0].attn
    torch.manual_seed(7)
    c = torch.randn(1, 8, m.config.kv_lora_rank)
    kr = torch.randn(1, 8, m.config.qk_rope_dim)
    with torch.no_grad():
        buf_c, buf_kr, pos = attn._pack_cache_block(c, kr, reserve=16)
    assert int(pos) == 8
    pooled = (c.view(1, 2, 4, -1) * attn.z_cmp).sum(2) / 4
    # 暂存前缀在 [0, m)，块行从第 m 行起
    assert torch.allclose(buf_c[:, 4:6], pooled, atol=1e-6)
    assert torch.allclose(buf_kr[:, 4:6], kr[:, 3::4], atol=1e-6)
    # 暂存前缀里是**走过暂存的那一块**的原始 token（这里是最先的 4 个；中间的完整块
    # 走"直接池化"分支不过暂存）。它不参与读取，残留值无害——残块按槽位覆盖，
    # 且凑满 m 个才会 flush。
    assert torch.allclose(buf_c[:, :4], c[:, :4], atol=1e-6)

    # 再把 2 个残块 token 加进来：只进暂存，不出块行
    tail_c = torch.randn(1, 2, m.config.kv_lora_rank)
    tail_kr = torch.randn(1, 2, m.config.qk_rope_dim)
    with torch.no_grad():
        attn._block_add(buf_c, buf_kr, 8, tail_c, tail_kr)
    assert torch.allclose(buf_c[:, :2], tail_c, atol=1e-6)
    assert torch.allclose(buf_kr[:, :2], tail_kr, atol=1e-6)
    # 块行 2 仍未出现（要等凑满 4 个 token）
    assert float(buf_c[:, 6].abs().max()) == 0.0


def test_partial_block_invisible_until_complete():
    """残块没凑满 m 个 → 只进暂存、不可读；凑满才成为块行."""
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8,
                     sparse_chunk=8))
    m.eval()
    with torch.no_grad():
        _, pasts = m._prefill(_ids(6), max_new_tokens=8)
        buf_c, pos = pasts[0][0], pasts[0][2]
        attn = m.layers[0].attn
        assert int(pos) == 6
        # 6 个 token → 1 个完整块（行 m=4）、残块 2 个 token（暂存行 0、1）
        assert int(pos) == 6
        assert attn._far_block_ids(6, 6, 0).tolist() == [0]
        # 残块 token 必须在暂存前缀里（否则上下文就丢了）
        assert float(attn._block_scratch(buf_c)[:, :2].abs().max()) > 0.0
        # 残块对应的块行还没出现（块号 1 → 行 5）
        assert float(buf_c[:, 5:6].abs().max()) == 0.0


# ---------------------------------------------------------------- 等价性


def test_prefill_matches_token_layout():
    """同权重：预填充输出与逐 token 缓存**逐位**相同（0.0，非近似）."""
    a, b = _pair()
    ids = _ids(32)
    with torch.no_grad():
        diff = float((a.encode_full_hidden(ids) - b.encode_full_hidden(ids)).abs().max())
    assert diff == 0.0


def test_decode_matches_token_layout():
    """同权重：逐 token 解码的输出也逐位相同（读的是同一组数）."""
    a, b = _pair()
    ids, nxt = _ids(32), _ids(4, seed=1)
    with torch.no_grad():
        _, pa = a._prefill(ids, max_new_tokens=8)
        _, pb = b._prefill(ids, max_new_tokens=8)
        ha = torch.cat([a._decode_step(nxt[:, i:i + 1], pa)[0] for i in range(4)], 1)
        hb = torch.cat([b._decode_step(nxt[:, i:i + 1], pb)[0] for i in range(4)], 1)
    assert float((ha - hb).abs().max()) == 0.0


def test_single_token_decode_matches_full_forward_across_partial_block():
    """prompt 长度不是 m 的倍数：逐 token 解码跨过残块边界仍与全前向一致.

    这是最容易错的一条：残块补齐前后，可见列集合会从"无"变成"有该块"，
    写路径若不统一（例如解码时逐步累加、预填充时一次池化），两者必然分叉。
    """
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8,
                     sparse_chunk=8))
    m.eval()
    ids = _ids(30)                       # 30 % 4 = 2 → 残块 2 个 token
    toks = [_ids(1, seed=10 + i) for i in range(6)]
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=8)
        hs = []
        for tk in toks:
            h, pasts = m._decode_step(tk, pasts)
            hs.append(h)
        full = m.encode_full_hidden(torch.cat([ids] + toks, dim=1))[:, -6:]
    assert float((torch.cat(hs, dim=1) - full).abs().max()) < 1e-5


def test_aligned_multi_token_decode_matches_full_forward():
    """块对齐的多 token 解码（一次 4 个，起点 32）仍与全前向一致.

    块**不对齐**的多 token 步会与全前向有差（列集合按步首位置确定，压缩块轴
    同样如此），单 token 解码永远对齐——见 pitfalls。
    """
    a, b = _pair()
    ids, nxt = _ids(32), _ids(4, seed=3)
    with torch.no_grad():
        _, pa = a._prefill(ids, max_new_tokens=8)
        _, pb = b._prefill(ids, max_new_tokens=8)
        ha, _ = a._decode_step(nxt, pa)
        hb, _ = b._decode_step(nxt, pb)
        full = b.encode_full_hidden(torch.cat([ids, nxt], dim=1))[:, -4:]
    assert float((ha - hb).abs().max()) == 0.0
    assert float((hb - full).abs().max()) < 1e-5


def test_indexer_selects_block_rows():
    """分层 indexer + 块主序：候选 latent 就是块行，预填充/解码都对得上全前向."""
    a, b = _pair(sparse_indexer=True, indexer_topk=4, indexer_pool_blocks=2,
                 compress_ratios="2", swa_window=8)
    ids = _ids(32)
    with torch.no_grad():
        diff = float((a.encode_full_hidden(ids) - b.encode_full_hidden(ids)).abs().max())
        _, pb = b._prefill(ids, max_new_tokens=8)
        h, pb2 = b._decode_step(_ids(1, seed=4), pb)
    assert diff == 0.0
    assert torch.isfinite(h).all()
    # indexer 的候选就是块行；缓冲按预留量（32+8）装，不随解码步数增长
    assert _rows(pb2[0]) == 2 + (40 + 1) // 2


def test_reuse_layer_shares_block_buffer():
    """CSA2 借用层共享生产者的**块主序**缓冲（同一对象、同一行号语义）."""
    a, b = _pair(csa_reuse_every=2, compress_ratios="2", swa_window=8)
    ids = _ids(32)
    with torch.no_grad():
        diff = float((a.encode_full_hidden(ids) - b.encode_full_hidden(ids)).abs().max())
        _, pb = b._prefill(ids, max_new_tokens=8)
        # 层 1/3 是 Reuse（借层 0/2 的 buf），同一 Python 对象
        assert pb[1][0] is pb[0][0] and pb[3][0] is pb[2][0]
        h, _ = b._decode_step(_ids(1, seed=6), pb)
    assert diff == 0.0
    assert torch.isfinite(h).all()


def test_quant_block_mean_quantized_once():
    """`kv_quant` + 块主序：块均值只量化一次 → 跨残块解码仍与全前向一致.

    若在解码里"边累加边重新舍入"，m 步内会漂移到数倍量化步长（实测 fp8 0.03 /
    fp4 0.08），故写路径必须先把残块攒进暂存、凑满后一次池化+量化。
    """
    for fmt in ("fp8", "fp4"):
        m = TinyLLM(_cfg(kv_block_cache=True, kv_quant=fmt, compress_ratios="4",
                         swa_window=8, sparse_chunk=8))
        m.eval()
        ids = _ids(30)
        toks = [_ids(1, seed=20 + i) for i in range(6)]
        with torch.no_grad():
            _, pasts = m._prefill(ids, max_new_tokens=8)
            hs = []
            for tk in toks:
                h, pasts = m._decode_step(tk, pasts)
                hs.append(h)
            full = m.encode_full_hidden(torch.cat([ids] + toks, dim=1))[:, -6:]
        assert float((torch.cat(hs, dim=1) - full).abs().max()) < 1e-4, fmt


# ---------------------------------------------------------------- 会话缓存


def test_scratch_survives_growth():
    """`ensure_room` 只扩尾部：暂存前缀里的残块 token 不会被搬走/清零."""
    m = TinyLLM(_cfg(compress_ratios="4", swa_window=8, sparse_chunk=8,
                     kv_block_cache=True))
    m.eval()
    ids = _ids(31)                       # 31 % 4 = 3 → 残块有 3 个 token
    toks = [_ids(1, seed=30 + i) for i in range(5)]
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=1)   # 预留极小 → 解码必扩
        sc = SessionCache(m)
        sc.pasts, sc.ids = pasts, ids.clone()
        hs = [sc.extend(tk) for tk in toks]
        full = m.encode_full_hidden(sc.ids)[:, -5:]
    assert float((torch.stack(hs, dim=1) - full).abs().max()) < 1e-5
    # 扩容后行数 ≥ m + ceil(L/m)（2x 扩容，不是按 token 数精确值）
    import math

    need = 4 + math.ceil(sc.ids.shape[1] / 4)
    assert _rows(sc.pasts[0]) >= need


def test_session_save_load_keeps_block_rows(tmp_path):
    """落盘（drop_swa=False）载入后块行/暂存前缀原样，下一步输出逐位一致."""
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8,
                     sparse_chunk=8))
    m.eval()
    ids = _ids(31)
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=8)
        sc = SessionCache(m)
        sc.pasts, sc.ids = pasts, ids.clone()
        sc.extend(_ids(2, seed=40))
        path = str(tmp_path / "sess.pt")
        sc.save(path, drop_swa=False)
        sc2 = SessionCache.load(path, m)
        nxt = _ids(1, seed=41)
        h1 = sc.extend(nxt)
        h2 = sc2.extend(nxt)
    assert int(sc2.pasts[0][2]) == int(sc.pasts[0][2])
    assert float((h1 - h2).abs().max()) < 1e-5


def test_session_save_load_replays_swa_over_block_rows(tmp_path):
    """`drop_swa=True`（默认）：只存全局主 KV（含块行/暂存），环靠只读重放重建.

    块主序下"权威全局 KV"就是那 `ceil(L/m)` 行；重放只重建环、绝不写块行，
    所以载入后块行必须与存盘前逐位相同，而重建的环比"无环冷启动"更接近真值。
    """
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8,
                     sparse_chunk=8))
    m.eval()
    ids = _ids(29)                        # 29 % 4 = 1 → 残块 1 个 token
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=8)
        sc = SessionCache(m)
        sc.pasts, sc.ids = pasts, ids.clone()
        sc.extend(_ids(2, seed=50))
        path = str(tmp_path / "sess.pt")
        sc.save(path)                      # 默认 drop_swa=True
        sc2 = SessionCache.load(path, m)
        # 全局主 KV（块行 + 暂存前缀）原样：同一 pos、同一行值
        assert int(sc2.pasts[0][2]) == int(sc.pasts[0][2])
        assert torch.allclose(sc2.pasts[0][0], sc.pasts[0][0])
        nxt = _ids(1, seed=51)
        h_ref = sc.extend(nxt)
        h_replay = sc2.extend(nxt)
        # 对照：环清零 = 冷启动
        cold = SessionCache(m)
        cold.pasts = [tuple(p[:3]) + tuple(torch.zeros_like(x) for x in p[3:])
                      for p in sc.pasts]
        cold.ids = sc.ids.clone()
        h_cold = cold.extend(nxt)
    assert torch.isfinite(h_replay).all()
    err_replay = float((h_replay - h_ref).abs().max())
    err_cold = float((h_cold - h_ref).abs().max())
    assert err_replay < err_cold


# ---------------------------------------------------------------- 计费/训练/来源


def test_kv_cache_bytes_counts_rows():
    """`kv_cache_bytes` 按 `m + ceil(L/m)` 行计费：差额正好是省下的行."""
    a = TinyLLM(_cfg())
    b = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="4", swa_window=8))
    for seq_len in (1024, 32768):
        sa = a.kv_cache_bytes(seq_len)
        sb = b.kv_cache_bytes(seq_len)
        assert sb < sa
        row_bytes = (b.config.kv_lora_rank + b.config.qk_rope_dim) * 2
        rows_saved = seq_len - (4 + (seq_len + 3) // 4)
        # 4 个全注意力层都存 KV（无 Reuse）
        assert sa - sb == 4 * rows_saved * row_bytes


def test_training_gradients_finite_and_z_cmp_used():
    """训练（return_state=False）不需要缓存，但池化权重要拿到梯度且输出有限."""
    m = TinyLLM(_cfg(kv_block_cache=True, compress_ratios="2", swa_window=8))
    m.train()
    ids = _ids(24)
    out = m(ids, targets=ids)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    n_grad, n_pool = 0, 0
    for layer in m.layers:
        z = getattr(layer.attn, "z_cmp", None)
        if z is None:
            continue
        n_pool += 1
        assert z.grad is not None and torch.isfinite(z.grad).all()
        if float(z.grad.abs().sum()) > 0:
            n_grad += 1
    # 4 层都是 m=2（各自独立池化权重），至少得有一层真的用到（块长 ≥ 2 才有池化梯度）
    assert n_pool == 4 and n_grad >= 1


def test_checkpoint_alignment_and_cli(tmp_path):
    """来源对齐：ckpt 记了 `kv_block_cache` 就沿用；老 ckpt 回退关；CLI 三态."""
    import json
    import sys
    from pathlib import Path

    assert "kv_block_cache" in _ARCH_SWITCHES
    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "kv_block_cache": True, "swa_window": 8, "compress_ratios": "2",
    }}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = _cfg()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.kv_block_cache is True and cfg.swa_window == 8

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(json.dumps({"config": {
        "d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = _cfg()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.kv_block_cache is False

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).kv_block_cache is None
    args = T.parse_args(["--kv-block-cache", "--swa-window", "8",
                         "--compress-ratios", "2"])
    c = _cfg()
    src = T.resolve_arch_switches(args, c)
    assert c.kv_block_cache is True and src["kv_block_cache"] == "explicit"
