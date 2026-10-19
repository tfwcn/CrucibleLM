"""层内 SWA 单测 — `swa_window`（DeepSeek-V4.1 §3.2.1「each layer retains its own SWA KV」）.

核心契约：
- 每层**自带** SWA 分支（独立投影 `swa.*`，不改结构/不新增键时不可见），
  与全局主 KV 放**同一次 softmax**（论文：selected entries together with the
  layer-local SWA KV）；
- 窗口语义：查询 `i` 只看 `[i-W+1, i]`（带状 + 因果），窗口外与未来一律不可见；
- 环**定长**：容量 `window + max(0, t-1)`（t = 一次解码 token 数），
  **与序列长度无关**（这是 SWA 相对全局主 KV 的根本优势，也是 `kv_cache_bytes`
  里它是加法项的原因）；
- SWA 开启时全局主 KV 只出"选中条目"、本地近端完全交给 SWA（不重复计权），
  且恒走分块路径（dense 兜底不带层内 SWA，否则短 prompt 预填充与解码分叉）；
- `swa_window=0` 逐位等价旧行为且不新增 `state_dict` 键；
- 因果（改未来不影响前部）+ Finite 梯度；`prefill + 增量解码 == 全前向`；
- SWA Bounded Replay（V4.1 §3.2.2）：**只读**权威全局主 KV、只重建 SWA 环；
  `n_win ≥ 已缓长度` 时逐位精确，否则优于冷启动（无环）。
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig, sync_arch_from_checkpoint  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.rope import apply_rope  # noqa: E402
from src.llm.local.session_cache import SessionCache  # noqa: E402


def _cfg(**over) -> SmallLLMConfig:
    """小模型：4 层全注意力、chunk=4（便于 chunk 对齐解码）."""
    base = dict(
        vocab_size=256, d_model=128, n_layers=4, full_attn_every=1,
        n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=8, top_k=2, expert_hidden=64, n_shared=1,
        mtp_depth=1, max_seq_len=256, yarn_scale=1.0,
        sparse_threshold=4, sparse_window=4, sparse_sink=2,
        sparse_stride=2, sparse_chunk=4,
        sparse_indexer=False, csa_reuse_every=1, grad_ckpt=False,
    )
    base.update(over)
    return SmallLLMConfig(**base)


def _swa_layers(pasts):
    """取带 SWA 环（六元组）的层缓存."""
    return [p for p in pasts if isinstance(p, tuple) and len(p) == 6]


def _ring_caps(pasts):
    return {int(p[3].shape[1]) for p in _swa_layers(pasts)}


def _ring_diff(a, b) -> float:
    """两组 SWA 环的最大逐元素差（同层数、同容量时）."""
    ra, rb = _swa_layers(a), _swa_layers(b)
    assert len(ra) == len(rb) and len(ra) > 0
    worst = 0.0
    for pa, pb in zip(ra, rb):
        for x, y in zip(pa[3:], pb[3:]):
            worst = max(worst, float((x - y).abs().max()))
    return worst


# ------------------------------- 关：逐位等价 -------------------------------
def test_swa_off_is_bitwise_identical_and_adds_no_keys():
    """`swa_window=0`（默认）：键集合不变、无 `swa.*`、输出逐位相同."""
    torch.manual_seed(0)
    a = TinyLLM(_cfg())
    torch.manual_seed(0)
    b = TinyLLM(_cfg(swa_window=0))
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any(".swa." in k for k in a.state_dict()), "关时不得新增参数"
    ids = torch.randint(0, 256, (1, 12))
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.equal(a.encode_full_hidden(ids), b.encode_full_hidden(ids))


def test_swa_on_adds_only_swa_keys():
    """开启后只多出 `swa.*` 参数（其它层结构不变）."""
    torch.manual_seed(0)
    base = TinyLLM(_cfg())
    torch.manual_seed(0)
    on = TinyLLM(_cfg(swa_window=8))
    extra = set(on.state_dict()) - set(base.state_dict())
    assert extra, "应新增层内 SWA 参数"
    assert all(".swa." in k for k in extra), sorted(extra)


# ------------------------------- 窗口数学 -------------------------------
def test_swa_is_dense_sliding_window_attention():
    """单层、静态集为空时：层输出 == 手写滑动窗口注意力（带状因果、同一次 softmax）.

    配置成"全局列全空"（sink=0、stride 超长、窗口=0）后，注意力只剩 SWA 列，
    于是可与手写参考逐位对齐——这是窗口语义（含 rope、qk 拼接、scale）的硬契约。
    """
    W = 3
    torch.manual_seed(0)
    m = TinyLLM(_cfg(n_layers=1, swa_window=W, sparse_sink=0,
                     sparse_stride=10 ** 6, sparse_threshold=1,
                     sparse_chunk=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 6))
    b, t = ids.shape
    layer = m.layers[0]
    attn = layer.attn
    with torch.no_grad():
        hn = layer.norm1(m.embed(ids))
        q_c = attn.w_dq(hn)
        hd, rd = attn.head_dim, attn.qk_rope_dim
        q = attn.q_norm(attn.w_uq(q_c).view(b, t, attn.n_heads, hd))
        qr = apply_rope(attn.w_qr(q_c).view(b, t, attn.n_heads, rd),
                        attn.rope_cos, attn.rope_sin)
        k, kr, v = attn.swa.project(hn)
        kr = apply_rope(kr, attn.rope_cos,
                        attn.rope_sin).unsqueeze(2).expand(-1, -1, attn.n_heads, -1)
        qf = torch.cat([q, qr], -1).transpose(1, 2)
        kf = torch.cat([k, kr], -1).transpose(1, 2)
        vf = v.transpose(1, 2)
        s = torch.matmul(qf, kf.transpose(-1, -2)) * (qf.shape[-1] ** -0.5)
        pos = torch.arange(t)
        vis = ((pos[None, :] <= pos[:, None])
               & (pos[None, :] > pos[:, None] - W))
        s = s.masked_fill(~vis, float("-inf"))
        o = torch.matmul(torch.softmax(s.float(), -1).to(vf.dtype), vf)
        ref = attn._out_proj(o.transpose(1, 2).reshape(b, t, -1))
        got, _ = attn(hn, None, False)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()


def test_swa_window_masks_out_of_window_and_future():
    """带状掩码：距离 > W-1 或未来位置一律 -inf，窗口内为 0."""
    W = 4
    torch.manual_seed(0)
    m = TinyLLM(_cfg(n_layers=1, swa_window=W))
    band = m.layers[0].attn.swa.band_mask(
        torch.arange(3, 8), torch.arange(0, 10))
    for i in range(3, 8):
        for j in range(10):
            allowed = (j <= i) and (j > i - W)
            if not allowed:
                assert band[i - 3, j] == float("-inf"), (i, j)
            else:
                assert band[i - 3, j] == 0.0, (i, j)


# ------------------------------- 缓存布局 -------------------------------
def test_ring_is_bounded_and_grows_only_for_multi_token():
    """环容量 = `window + max(0, t-1)`：预填充=窗口、单步=窗口、4 步解码扩容.

    关键：**与序列长度无关**——40 个 token 的预填充后环仍是 8 行。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=4)
        assert _ring_caps(pasts) == {8}, _ring_caps(pasts)
        nxt = torch.randint(0, 256, (1, 4))
        _, p4, _ = m._decode_multi(nxt, pasts)
        assert _ring_caps(p4) == {16}, _ring_caps(p4)  # 8 -> max(16, 8+3)
        _, p1 = m._decode_step(nxt[:, :1], pasts)
        assert _ring_caps(p1) == {8}, _ring_caps(p1)


def test_kv_cache_swa_term_is_seq_independent():
    """SWA 缓存是**定长环**：缓存增量只随序列长度线性涨全局主 KV 部分.

    `kv_cache_bytes(L2) - kv_cache_bytes(L1)` 必须等于全局主 KV 的增量，
    SWA 项完全不含 `L`（V4.1：SWA KV storage is bounded independently of
    sequence length）。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    off = TinyLLM(_cfg(swa_window=0))
    c = m.config
    per_tok = (c.kv_lora_rank + c.qk_rope_dim) * 2 * c.n_full_layers
    for on_m in (m,):
        d = on_m.kv_cache_bytes(1000, 2) - on_m.kv_cache_bytes(500, 2)
        assert d == per_tok * 500, (d, per_tok * 500)
    # 开启后固定多出的部分 = 每全注意力层
    # (2*n_heads*head_dim + qk_rope_dim) * n_win * 2 字节（k/v 环 + 共享 rope 环）
    head_dim = c.d_model // c.n_heads
    per_ring = (2 * c.n_heads * head_dim + c.qk_rope_dim) * c.swa_window * 2
    expect = per_ring * c.n_full_layers
    assert m.kv_cache_bytes(1000, 2) - off.kv_cache_bytes(1000, 2) == expect


# ------------------------------- 因果 / 一致性 -------------------------------
def test_swa_causal_and_finite():
    """改未来 token 不影响前部；SWA 分支参数梯度 finite."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    ids = torch.randint(0, 256, (1, 24))
    ids2 = ids.clone()
    ids2[0, 20] = (ids2[0, 20] + 7) % 256
    m.eval()
    with torch.no_grad():
        a = m.encode_full_hidden(ids)
        b = m.encode_full_hidden(ids2)
    assert (a[0, :20] - b[0, :20]).abs().max() < 1e-6, "未来 token 污染前部"
    assert (a[0, 20:] - b[0, 20:]).abs().max() > 0
    m.train()
    m(ids, targets=ids)["loss"].backward()
    for n, p in m.named_parameters():
        if ".swa." in n and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_prefill_plus_decode_matches_full_forward():
    """`prefill + 增量解码 == 一次全前向`（SWA 开启，chunk 对齐 16->24）.

    增量一次喂 `sparse_chunk` 个 token（多 token 一步解码只在**块对齐**时与
    全前向等价：全局 pattern 的静态集依赖块起点，跨块的一步解码必然有差）。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    chunk = m.config.sparse_chunk
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        pre, pasts = m._prefill(ids[:, :16], max_new_tokens=8)
        assert torch.allclose(pre, full[:, :16], atol=1e-5)
        h, pasts = m._decode_step(ids[:, 16:16 + chunk], pasts)
        tail = h
        for s in range(16 + chunk, 24, chunk):
            h, pasts = m._decode_step(ids[:, s:s + chunk], pasts)
            tail = torch.cat([tail, h], dim=1)
    assert torch.allclose(tail, full[:, 16:], atol=1e-4), \
        (tail - full[:, 16:]).abs().max()


def test_short_prompt_also_uses_swa_path():
    """短 prompt（t ≤ sparse_threshold）也走分块路径：预填充与解码不分叉.

    dense 兜底路径不带层内 SWA；若短 prompt 走 dense、解码走 SWA，两者必然
    分叉——所以 SWA 开启时 `sparse_threshold` 不再决定走哪条路。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=4, sparse_threshold=8, sparse_chunk=4))
    m.eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        pre, pasts = m._prefill(ids[:, :4], max_new_tokens=4)
        assert _ring_caps(pasts) == {4}, _ring_caps(pasts)
        tail, _ = m._decode_step(ids[:, 4:8], pasts)
    assert torch.allclose(pre, full[:, :4], atol=1e-5)
    assert torch.allclose(tail, full[:, 4:], atol=1e-4), \
        (tail - full[:, 4:]).abs().max()


def test_grad_ckpt_matches_plain_grads():
    """梯度检查点开关不改 SWA 分支梯度（重算幂等）."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 32))
    a = TinyLLM(_cfg(swa_window=8))
    b = TinyLLM(_cfg(swa_window=8, grad_ckpt=True))
    b.load_state_dict(a.state_dict())
    a.train(); b.train()
    a(ids, targets=ids)["loss"].backward()
    b(ids, targets=ids)["loss"].backward()
    for name in ("swa.w_k.weight", "swa.w_kr.weight", "swa.w_v.weight"):
        ga = dict(a.named_parameters())[f"layers.0.attn.{name}"].grad
        gb = dict(b.named_parameters())[f"layers.0.attn.{name}"].grad
        assert ga is not None and gb is not None
        assert torch.allclose(ga, gb, atol=1e-6), (name, (ga - gb).abs().max())


def test_reuse_layers_carry_their_own_swa():
    """CSA2 借用层（reindex/reuse）**各有自己的** SWA 环，不借生产者的.

    论文口径：SWA 是每层自带、与全局主 KV 独立的分支——所以 `csa_reuse` 只
    影响全局主 KV 的来源，不改变 SWA 的归属。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8, csa_reuse_every=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=4)
    full_layers = [i for i, l in enumerate(m.layers)
                   if getattr(l, "full_attn", False)]
    assert len(full_layers) >= 2
    assert len(_swa_layers(pasts)) == len(full_layers)
    reuse_idx = [i for i in full_layers
                 if getattr(m.layers[i].attn, "csa_reuse", False)]
    assert reuse_idx, "该配置应存在借用层"
    for i in reuse_idx:
        assert m.layers[i].attn.swa is not None
        assert _swa_layers(pasts)[i][3].abs().sum() > 0, "借用层环不得为空"
    # 借用层与生产者的环是不同对象
    prod = full_layers[0]
    assert _swa_layers(pasts)[prod][3] is not _swa_layers(pasts)[reuse_idx[0]][3]


def test_reuse_swa_decode_matches_full_forward():
    """CSA2 借用层 + 层内 SWA：`prefill + 解码 == 全前向`（环各建各的）.

    这条是"借用层照抄生产者整份 past"那个 bug 的回归：环若共享，解码时两层
    往同一个环里写、互相覆盖，增量解码立刻与全前向分叉。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8, csa_reuse_every=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    chunk = m.config.sparse_chunk
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        _, pasts = m._prefill(ids[:, :16], max_new_tokens=8)
        h, pasts = m._decode_step(ids[:, 16:16 + chunk], pasts)
        tail = h
        for s in range(16 + chunk, 24, chunk):
            h, pasts = m._decode_step(ids[:, s:s + chunk], pasts)
            tail = torch.cat([tail, h], dim=1)
    assert torch.allclose(tail, full[:, 16:], atol=1e-4), \
        (tail - full[:, 16:]).abs().max()


def test_speculative_decoding_disabled_under_swa():
    """SWA 开启时禁用投机解码（部分接受会把未接受 token 的环行留在缓存里）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8, draft_enabled=True))
    ids = torch.randint(0, 256, (1, 4))
    assert m._spec_eligible(ids, 0.0) is False
    off = TinyLLM(_cfg(swa_window=0, draft_enabled=True))
    assert off._spec_eligible(ids, 0.0) is True


# ------------------------------- Bounded Replay -------------------------------
def test_bounded_replay_is_exact_when_window_covers_sequence():
    """`n_win ≥ 已缓长度`：重放逐位精确（环 == 原环），且全局主 KV 只读未改."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=32))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=4)
        before = [p[0].clone() for p in _swa_layers(pasts)]
        rebuilt = m.replay_swa_from_ids(ids, [p[:3] for p in pasts])
        after = [p[0] for p in _swa_layers(rebuilt)]
        assert _ring_caps(rebuilt) == _ring_caps(pasts)
        assert _ring_diff(pasts, rebuilt) < 1e-5, _ring_diff(pasts, rebuilt)
        # 只读：全局主 KV 一个字节都没动（论文 "without ... overwriting"）
        assert all(float((x - y).abs().max()) == 0.0
                   for x, y in zip(before, after))


def test_bounded_replay_rebuilds_hybrid_state_from_segment():
    """hybrid 也要"整段重放=精确"：线性层状态必须**清零重建**，不能沿用末态.

    末态已经含过重放段（以及整段）的贡献，再喂一遍是重复计入——比"近似"更糟。
    这里同时锁两件事：① `n_win ≥ 已缓长度` 时（含 hybrid）按索引逐步重放精确；
    ② 沿用末态那种写法误差**显著更大**（防有人"图省事"改回去）。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=32, full_attn_every=2, csa_reuse_every=1))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=4)
        slim = [p[:3] for p in pasts]
        rebuilt = m.replay_swa_from_ids(ids, slim)
        assert _ring_diff(pasts, rebuilt) < 1e-5, _ring_diff(pasts, rebuilt)
        x = ids[:, :1]
        h_exact, _ = m._decode_step(x, pasts)
        h_rep, _ = m._decode_step(x, rebuilt)
        assert float((h_exact - h_rep).abs().max()) < 1e-5
        # 对照：沿用线性末态的重放（错误做法）——误差必须明显更大
        total = int(pasts[0][2].item())
        naive: list = []
        for layer, p in zip(m.layers, pasts):
            if not (isinstance(p, tuple) and len(p) >= 3
                    and isinstance(p[2], torch.Tensor) and p[2].numel() == 1):
                naive.append(p)                  # 线性层：沿用末态（错）
                continue
            rings = ()
            if (getattr(layer, "full_attn", False)
                    and getattr(layer.attn, "swa", None) is not None):
                rings = layer.attn.empty_swa_rings(
                    1, batch=p[0].shape[0], device=p[0].device, dtype=p[0].dtype)
            naive.append((p[0], p[1], p[2] - total) + rings)
        for i in range(total):
            _, naive = m._decode_step(ids[:, i:i + 1], naive,
                                      write_cache=False, swa_floor=0)
        naive_pasts = [tuple(p[:3]) + tuple(r[3:]) if len(r) > 3 else p
                       for p, r in zip(pasts, naive)]
        h_naive, _ = m._decode_step(x, naive_pasts)
    e_rep = float((h_exact - h_rep).abs().max())
    e_naive = float((h_exact - h_naive).abs().max())
    assert e_naive > 100 * e_rep, (e_rep, e_naive)


def test_bounded_replay_beats_cold_start():
    """窗口短于已缓长度时：重放是**近似**，但明显优于冷启动（无环）.

    V4.1 §3.2.2 承认"重放出的前缀状态是近似的、与位置相关"；本仓在未训练的
    随机小模型上量化这条：重放误差 < 冷启动误差（论文靠训练后的有效感受野，
    随机权重下层间无收缩性，故只锁"严格更优"这条弱契约）。
    """
    for over in ({}, {"full_attn_every": 2}):  # 全注意力 / hybrid 各测一次
        torch.manual_seed(0)
        m = TinyLLM(_cfg(swa_window=8, **over))
        m.eval()
        ids = torch.randint(0, 256, (1, 24))
        with torch.no_grad():
            _, pasts = m._prefill(ids, max_new_tokens=4)
            slim = [p[:3] for p in pasts]
            rebuilt = m.replay_swa_from_ids(ids, slim)
            x = ids[:, :1]
            exact, _ = m._decode_step(x, pasts)
            rep, _ = m._decode_step(x, rebuilt)
            cold, _ = m._decode_step(x, slim)
        e_rep = float((exact - rep).abs().max())
        e_cold = float((exact - cold).abs().max())
        assert e_rep < e_cold, (over, e_rep, e_cold)


def test_bounded_replay_keeps_decode_running():
    """重放后 `pasts` 结构与正常解码同构（六元组）且可继续增量解码."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=4)
        rebuilt = m.replay_swa_from_ids(ids, [p[:3] for p in pasts])
        assert all(len(p) == 6 for p in _swa_layers(rebuilt))
        h, p2 = m._decode_step(torch.randint(0, 256, (1, 1)), rebuilt)
        assert torch.isfinite(h).all()
        assert all(len(p) == 6 for p in _swa_layers(p2))


# ------------------------------- 会话缓存 -------------------------------
def test_session_cache_roundtrip_drop_swa(tmp_path):
    """`save(drop_swa=True)` 只存全局主 KV；`load` 重放补齐环，可继续解码."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8))
    m.eval()
    ids = torch.randint(0, 256, (1, 32))
    sc = SessionCache(m)
    with torch.no_grad():
        sc.extend(ids)
        assert all(len(p) == 6 for p in _swa_layers(sc.pasts))
        path = str(tmp_path / "sc.pt")
        sc.save(path)
        blob = torch.load(path, weights_only=False)
        assert blob["swa_dropped"] is True
        assert all(len(p) == 3 for p in blob["pasts"]), "落盘不得含 SWA 环"
        back = SessionCache.load(path, m)
        assert all(len(p) == 6 for p in _swa_layers(back.pasts))
        assert _ring_caps(back.pasts) == {8}
        # ids 只是记账，读回后必须留在 CPU（否则 GPU 上 extend 的 cat 会设备不一致）
        assert back.ids.device.type == "cpu"
        h = back.extend(torch.randint(0, 256, (1, 2)))
        assert torch.isfinite(h).all()
        assert len(back) == 34


def test_session_cache_roundtrip_keeps_swa_exactly(tmp_path):
    """`save(drop_swa=False)`：环也落盘，`load` 逐位精确（不经重放）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=32))
    m.eval()
    ids = torch.randint(0, 256, (1, 16))
    sc = SessionCache(m)
    with torch.no_grad():
        sc.extend(ids)
        path = str(tmp_path / "sc.pt")
        sc.save(path, drop_swa=False)
        blob = torch.load(path, weights_only=False)
        assert blob["swa_dropped"] is False
        assert all(len(p) == 6 for p in _swa_layers(blob["pasts"]))
        back = SessionCache.load(path, m)
    assert _ring_diff(sc.pasts, back.pasts) == 0.0


def test_ensure_room_keeps_shared_buffer_identity():
    """扩容后 CSA2 借用层与生产者仍**共享同一缓冲**（pitfall 36 回归）.

    逐层各建一份新缓冲会让生产者写进自己那份、借用层读到旧副本。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(swa_window=8, csa_reuse_every=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        _, pasts = m._prefill(ids, max_new_tokens=1)
    sc = SessionCache(m)
    sc.pasts = pasts
    sc.ids = ids
    reuse_idx = [i for i, l in enumerate(m.layers)
                 if getattr(l, "full_attn", False)
                 and getattr(l.attn, "csa_reuse", False)]
    prod_idx = [i for i, l in enumerate(m.layers)
                if getattr(l, "full_attn", False)
                and not getattr(l.attn, "csa_reuse", False)]
    assert reuse_idx and prod_idx
    buf_id = {i: id(pasts[i][0]) for i in prod_idx + reuse_idx}
    sc.ensure_room(64)          # 触发 2x 扩容
    after = {i: id(sc.pasts[i][0]) for i in prod_idx + reuse_idx}
    caps = {int(sc.pasts[i][0].shape[1]) for i in prod_idx}
    assert all(c > 16 for c in caps), caps
    # 每个借用层与它前面的生产者必须仍是同一个对象
    for i in reuse_idx:
        prod = max(j for j in prod_idx if j < i)
        assert buf_id[i] == buf_id[prod]
        assert after[i] == after[prod], f"层 {i} 与生产者 {prod} 扩容后不再共享缓冲"
    # SWA 环定长：扩容原样带上
    assert _ring_caps(sc.pasts) == {8}, _ring_caps(sc.pasts)


# ----------------------------- 来源对齐 / CLI -----------------------------
def test_checkpoint_alignment_and_cli(tmp_path):
    """来源对齐：checkpoint 记了 `swa_window` 就沿用；老 ckpt 读不到保守回退 0（关）.

    层内 SWA 改 `state_dict` 键（新增 `swa.*`），所以"默认值升级"绝不能把老 run
    静默换成新架构——续训/迁移必须显式开或从 hparams 沿用。
    """
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "swa_window": 256}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = _cfg()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.swa_window == 256

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(json.dumps({"config": {
        "d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = _cfg()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.swa_window == 0, "老 ckpt 必须回退关（否则装载即缺 swa.* 键）"
    assert not any(".swa." in k for k in TinyLLM(cfg2).state_dict())

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).swa_window is None       # 未指定 = 三态里的 None
    args = T.parse_args(["--swa-window", "512"])
    assert args.swa_window == 512
    c = _cfg()
    src = T.resolve_arch_switches(args, c)
    assert c.swa_window == 512 and src["swa_window"] == "explicit"
