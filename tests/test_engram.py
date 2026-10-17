"""Engram 条件记忆单测 — 哈希确定性 / n-gram 对齐 / 恒等起点 / 解码一致 / 默认关.

CPU 可跑。数学正确性用"解码 == 全前向"与参考实现卡死；恒等起点锁定
"开开关不破坏已训权重"；默认关锁定 state_dict 兼容。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.engram import EngramMemory, hash_ngrams, is_prime
from src.llm.local.model import TinyLLM


def _engram_cfg(every=2, table=127, d_mem=32):
    """开 Engram 的极小配置（小素数表，CPU 秒级）."""
    cfg = tiny_test_config()
    cfg.engram_every = every
    cfg.engram_table_size = table
    cfg.engram_d_mem = d_mem
    cfg.engram_orders = "2,3"
    cfg.engram_heads = 2
    return cfg


def _randomize_engram(m):
    """把恒等起点的 Engram 权重打乱（否则输出与基线相同，测不出接线）."""
    with torch.no_grad():
        for layer in m.layers:
            eng = getattr(layer, "engram", None)
            if eng is None:
                continue
            for t in eng.tables:
                torch.nn.init.normal_(t.weight, std=0.05)
            torch.nn.init.normal_(eng.w_k.weight, std=0.05)
            torch.nn.init.normal_(eng.w_v.weight, std=0.05)
            if eng.conv is not None:
                torch.nn.init.normal_(eng.conv.weight, std=0.05)


# --------------------------- 哈希与 n-gram ---------------------------
def test_is_prime():
    assert is_prime(2) and is_prime(127) and is_prime(8191)
    assert not is_prime(1) and not is_prime(4) and not is_prime(100)


def test_hash_deterministic_and_seed_sensitive():
    """同 n-gram 恒同下标（确定性寻址）；不同 seed 打散；下界在 [0, table)."""
    g = torch.tensor([[1, 2], [1, 2], [3, 4]])
    a = hash_ngrams(g, 127, 5)
    assert torch.equal(a, hash_ngrams(g, 127, 5))
    assert a[0] == a[1]                     # 相同 gram → 相同下标
    assert not torch.equal(a, hash_ngrams(g, 127, 6))  # seed 不同 → 不同
    assert int(a.min()) >= 0 and int(a.max()) < 127


def test_grams_last_alignment():
    """末 t 位的 n-gram 对齐：gram[p] = (x_{p-n+1}, …, x_p)."""
    e = EngramMemory(d_model=8, vocab_size=100, orders=(2, 3), n_heads=1,
                     d_mem=8, table_size=127, conv_kernel=0)
    ctx = torch.tensor([[10, 11, 12, 13]])
    assert e._grams_last(ctx, 2, 2).tolist() == [[[11, 12], [12, 13]]]
    assert e._grams_last(ctx, 3, 2).tolist() == [[[10, 11, 12], [11, 12, 13]]]


def test_grams_last_pads_with_boundary():
    """上下文不足时左补 boundary（=vocab_size，词表外哨兵）."""
    e = EngramMemory(d_model=8, vocab_size=100, orders=(2,), n_heads=1,
                     d_mem=8, table_size=127, conv_kernel=0)
    assert e.boundary == 100
    assert e._grams_last(torch.tensor([[10]]), 2, 1).tolist() == [[[100, 10]]]


# --------------------------- 构造校验 ---------------------------
def test_engram_table_must_be_prime():
    with pytest.raises(AssertionError, match="素数"):
        EngramMemory(d_model=8, vocab_size=100, table_size=100)


def test_engram_d_mem_must_divide():
    with pytest.raises(AssertionError, match="整除"):
        EngramMemory(d_model=8, vocab_size=100, orders=(2, 3), n_heads=2, d_mem=10)


def test_engram_order_min_two():
    with pytest.raises(AssertionError, match="≥2"):
        EngramMemory(d_model=8, vocab_size=100, orders=(1,))


# --------------------------- 恒等起点 / 默认关 ---------------------------
def test_engram_identity_at_init():
    """开 Engram 不改变已训权重输出（W_V/卷积零初始化 → 开局 Y=0）."""
    torch.manual_seed(0)
    base = TinyLLM(tiny_test_config())
    torch.manual_seed(1)
    m = TinyLLM(_engram_cfg())
    m.load_state_dict(base.state_dict(), strict=False)  # 只缺 engram 新键
    base.eval(); m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        a = base(x)["logits"]
        b = m(x)["logits"]
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_engram_default_off_no_module():
    """默认关：模型无 Engram 模块，state_dict 不含 engram 键（旧 checkpoint 兼容）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    assert all(getattr(layer, "engram", None) is None for layer in m.layers)
    assert not any("engram" in k for k in m.state_dict())
    assert not m._has_engram()


# --------------------------- 解码 == 全前向 ---------------------------
def _decode_matches_full(cfg):
    torch.manual_seed(0)
    m = TinyLLM(cfg)
    _randomize_engram(m)
    m.eval()
    x = torch.randint(0, 256, (1, 6))
    toks = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        _, pasts = m._prefill(x, max_new_tokens=toks.shape[1])
        hs = []
        cur = x
        for i in range(toks.shape[1]):
            nxt = toks[:, i:i + 1]
            eids = m._engram_block_ids(cur, nxt)
            hh, pasts = m._decode_step(nxt, pasts, engram_ids=eids)
            hs.append(hh)
            cur = torch.cat([cur, nxt], dim=1)
        seq = torch.cat(hs, dim=1)
        full = m.encode_full_hidden(torch.cat([x, toks], dim=1))
    return seq, full[:, -toks.shape[1]:]


def test_engram_decode_matches_full_forward():
    """逐 token 解码（带 Engram 上下文）与全前向末段逐位一致."""
    seq, ref = _decode_matches_full(_engram_cfg())
    assert torch.allclose(seq, ref, atol=1e-5), (seq - ref).abs().max()


def test_engram_decode_matches_full_forward_with_hyper():
    """超连接 + Engram：解码跳位仍与全前向一致（多流展开/combine 都对）."""
    cfg = _engram_cfg()
    cfg.hyper_streams = 2
    seq, ref = _decode_matches_full(cfg)
    assert torch.allclose(seq, ref, atol=1e-5), (seq - ref).abs().max()


def test_engram_decode_multi_matches_sequential():
    """并行验证（_decode_multi）与逐 token 解码一致（含 Engram 上下文）."""
    torch.manual_seed(0)
    m = TinyLLM(_engram_cfg())
    _randomize_engram(m)
    m.eval()
    x = torch.randint(0, 256, (1, 6))
    toks = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        _, pa = m._prefill(x, max_new_tokens=toks.shape[1])
        _, pb = m._prefill(x, max_new_tokens=toks.shape[1])
        hs, cur = [], x
        for i in range(toks.shape[1]):
            nxt = toks[:, i:i + 1]
            hh, pa = m._decode_step(
                nxt, pa, engram_ids=m._engram_block_ids(cur, nxt))
            hs.append(hh)
            cur = torch.cat([cur, nxt], dim=1)
        seq = torch.cat(hs, dim=1)
        multi, _, _ = m._decode_multi(
            toks, pb, engram_ids=m._engram_block_ids(x, toks))
    assert torch.allclose(seq, multi, atol=1e-5), (seq - multi).abs().max()


def test_engram_spec_matches_greedy():
    """开 Engram + 投机解码：投机输出与逐 token 贪心逐位一致."""
    torch.manual_seed(0)
    m = TinyLLM(_engram_cfg())
    _randomize_engram(m)
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


# --------------------------- 可训练性 ---------------------------
def test_engram_train_step_updates_params():
    """一步训练：loss 有限、Engram 参数更新（梯度非空）."""
    from src.llm.local.train import build_optimizer, train_step

    torch.manual_seed(0)
    m = TinyLLM(_engram_cfg())
    _randomize_engram(m)
    m.train()
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    eng = m.layers[0].engram
    before = eng.w_v.weight.detach().clone()
    info = train_step(m, opt, x)
    assert torch.isfinite(torch.tensor(info["loss"]))
    assert not torch.equal(before, eng.w_v.weight.detach())
    assert m.layers[0].engram.w_v.weight.grad is not None
    assert torch.isfinite(m.layers[0].engram.w_v.weight.grad).all()
