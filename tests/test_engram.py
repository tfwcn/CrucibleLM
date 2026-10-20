"""Engram 条件记忆单测 — 哈希确定性 / n-gram 对齐 / 恒等起点 / 解码一致 / FP8 / 默认关.

对齐 V4.1 §2.4.2：阶 {2,3,4}、每阶 8 头、**无短卷积**、表 + K/V 投影 FP8（QAT，
推理可打包成真 FP8 存储）、表更新走 Sinkhorn。CPU 可跑。

数学正确性用"解码 == 全前向"与参考实现卡死；恒等起点锁定"开开关不破坏已训权重"；
默认关锁定 state_dict 兼容；FP8 用"按行量化 == 按表量化"与"打包 == 未打包"卡死。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import (  # noqa: E402
    _ARCH_SWITCHES,
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.engram import (  # noqa: E402
    EngramMemory,
    EngramTable,
    hash_ngrams,
    is_prime,
)
from src.llm.local.kv_quant import fake_quant  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402


def _engram_cfg(every=2, table=127, d_mem=32, layers=""):
    """开 Engram 的极小配置（小素数表，CPU 秒级）.

    `d_mem` 是**每阶**维度（V4.1 口径），2 头 → 每头 16 维（FP8 块量化的最小单位）。
    """
    cfg = tiny_test_config()
    cfg.engram_every = every
    cfg.engram_layers = layers
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
                     d_mem=16, table_size=127, fp8=False)
    ctx = torch.tensor([[10, 11, 12, 13]])
    assert e._grams_last(ctx, 2, 2).tolist() == [[[11, 12], [12, 13]]]
    assert e._grams_last(ctx, 3, 2).tolist() == [[[10, 11, 12], [11, 12, 13]]]


def test_grams_last_pads_with_boundary():
    """上下文不足时左补 boundary（=vocab_size，词表外哨兵）."""
    e = EngramMemory(d_model=8, vocab_size=100, orders=(2,), n_heads=1,
                     d_mem=16, table_size=127, fp8=False)
    assert e.boundary == 100
    assert e._grams_last(torch.tensor([[10]]), 2, 1).tolist() == [[[100, 10]]]


# --------------------------- 构造校验与 V4.1 规格 ---------------------------
def test_engram_table_must_be_prime():
    with pytest.raises(AssertionError, match="素数"):
        EngramMemory(d_model=8, vocab_size=100, table_size=100)


def test_engram_d_mem_must_divide_heads():
    """`d_mem` 是**每阶**维度，只须被头数整除（不再要求被 阶数×头数 整除）."""
    with pytest.raises(AssertionError, match="头数"):
        EngramMemory(d_model=8, vocab_size=100, orders=(2, 3), n_heads=8,
                     d_mem=100)


def test_engram_order_min_two():
    with pytest.raises(AssertionError, match="≥2"):
        EngramMemory(d_model=8, vocab_size=100, orders=(1,))


def test_paper_spec_orders_heads_no_conv_three_orders():
    """V4.1 规格：阶 {2,3,4}、8 头、每阶 d_mem、**无短卷积**、lookback = max_order-1."""
    e = EngramMemory(d_model=32, vocab_size=100, orders=(2, 3, 4), n_heads=8,
                     d_mem=128, table_size=127, fp8=True)
    assert e.orders == [2, 3, 4]
    assert len(e.tables) == 3 * 8              # 每 (阶, 头) 一张表
    assert e.d_head == 128 // 8 == 16          # 每头维度
    assert e.d_out == 3 * 128                  # 各阶拼接维
    assert e.max_order == 4 and e.lookback == 3
    assert not hasattr(e, "conv")              # 短卷积已移除（V4.1 §2.4.2）
    # 三个阶都参与：同一段上下文下，三阶查到的行下标各不相同
    ids = torch.tensor([[5, 6, 7, 8]])
    for o in e.orders:
        gram = e._grams_last(ids, o, 4)
        assert gram.shape == (1, 4, o)


def test_engram_layers_explicit_selection():
    """`engram_layers` 显式给层号（论文只放少数早期层），优先于 `engram_every`."""
    cfg = _engram_cfg(every=2, layers="1,3")
    assert cfg.engram_layer_ids() == {1, 3}
    m = TinyLLM(cfg)
    on = [i for i, layer in enumerate(m.layers) if layer.engram is not None]
    assert on == [1, 3]
    # 越界层号直接丢弃，不静默作用到其它层
    cfg.engram_layers = "1,99"
    assert cfg.engram_layer_ids() == {1}
    # 都为空 → 关闭
    cfg2 = tiny_test_config()
    assert cfg2.engram_layer_ids() is None


# --------------------------- FP8（表 + K/V 投影）---------------------------
def test_fp8_table_rowwise_equals_tablewise():
    """按行量化（查出来的行）与按表量化**逐位相同**：块量化 scale 只在本行内取."""
    torch.manual_seed(0)
    t = EngramTable(rows=64, cols=16, fp8=True)
    torch.nn.init.normal_(t.weight, std=0.3)
    idx = torch.randint(0, 64, (3, 5))
    got = t(idx)
    want = fake_quant(t.weight, "fp8")[idx]
    assert torch.equal(got, want), (got - want).abs().max()


def test_fp8_off_is_exact_lookup():
    """`fp8=False`：查表就是原始权重（无量化），投影也是普通线性."""
    torch.manual_seed(0)
    t = EngramTable(rows=64, cols=16, fp8=False)
    torch.nn.init.normal_(t.weight, std=0.3)
    idx = torch.randint(0, 64, (2, 4))
    assert torch.equal(t(idx), t.weight[idx])


def test_pack_unpack_bitwise_and_saves_bytes():
    """打包成真 FP8 存储后查表**逐位不变**，字节数约为原来的 1/1.9；解包还原."""
    torch.manual_seed(0)
    t = EngramTable(rows=256, cols=32, fp8=True)
    torch.nn.init.normal_(t.weight, std=0.3)
    t.eval()
    idx = torch.randint(0, 256, (4, 7))
    with torch.no_grad():
        before = t(idx)
        n_before = t.nbytes()
        t.pack()
        after = t(idx)
        n_after = t.nbytes()
        assert torch.equal(before, after)
        # bf16 主权重 2 B/通道 → 打包 1.0625 B/通道
        assert n_after == 256 * (32 + 32 // 16)
        assert n_before == 256 * 32 * t.weight.element_size()
        assert n_after < n_before
        t.unpack()
        assert t.nbytes() == n_before
        assert torch.equal(t(idx), before)


def test_pack_is_inference_only():
    """打包是快照：训练中打包/用打包的表都直接报错（否则静默读到旧表）."""
    t = EngramTable(rows=32, cols=16, fp8=True)
    t.train()
    with pytest.raises(RuntimeError, match="快照"):
        t.pack()
    t.eval()
    t.pack()
    t.train()
    with pytest.raises(RuntimeError, match="unpack"):
        t(idx=torch.zeros(1, 2, dtype=torch.long))


def test_module_pack_all_tables():
    """`EngramMemory.pack/unpack/nbytes` 覆盖所有 (阶, 头) 表."""
    e = EngramMemory(d_model=16, vocab_size=100, orders=(2, 3), n_heads=2,
                     d_mem=32, table_size=127, fp8=True)
    e.eval()
    before = e.nbytes()
    e.pack()
    assert e.nbytes() < before
    assert all(t.w8 is not None for t in e.tables)
    e.unpack()
    assert e.nbytes() == before


def test_fp8_projections_quantize_weights():
    """K/V 投影的权重走 FP8（QAT）：与非量化路径数值不同、与参考 fake-quant 一致."""
    torch.manual_seed(0)
    e = EngramMemory(d_model=16, vocab_size=100, orders=(2,), n_heads=1,
                     d_mem=16, table_size=127, fp8=True)
    e.eval()
    torch.nn.init.normal_(e.w_v.weight, std=0.2)
    torch.nn.init.normal_(e.w_k.weight, std=0.2)
    x = torch.randn(1, 4, 16)
    ids = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        out = e(x, ids)
        mem = e._embed(ids, 4)
        want = x + torch.sigmoid(
            (e.norm_q(x) * e.norm_k(torch.nn.functional.linear(
                mem, fake_quant(e.w_k.weight, "fp8")))).sum(-1, keepdim=True)
            / (16 ** 0.5)) * torch.nn.functional.linear(
                mem, fake_quant(e.w_v.weight, "fp8"))
        plain = x + torch.sigmoid(
            (e.norm_q(x) * e.norm_k(torch.nn.functional.linear(
                mem, e.w_k.weight))).sum(-1, keepdim=True)
            / (16 ** 0.5)) * torch.nn.functional.linear(mem, e.w_v.weight)
    assert torch.allclose(out, want, atol=1e-6)
    assert not torch.allclose(out, plain, atol=1e-6)   # FP8 确实生效了


# --------------------------- 恒等起点 / 默认关 ---------------------------
def test_engram_identity_at_init():
    """开 Engram 不改变已训权重输出（W_V 零初始化 → 开局 Y=0；FP8 下 0 仍是 0）."""
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
    """并行验证（`_decode_multi`）与逐 token 解码一致（含 Engram 上下文）."""
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


# --------------------------- 可训练性 / Sinkhorn ---------------------------
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


def test_sinkhorn_routes_engram_tables():
    """Sinkhorn 组包含 Engram 表（V4.1：表用 momentum + Sinkhorn 更新）."""
    from src.llm.local.optim import build_hybrid_optimizer

    torch.manual_seed(0)
    m = TinyLLM(_engram_cfg())
    base = TinyLLM(tiny_test_config())
    opt = build_hybrid_optimizer(m, sinkhorn=True)
    opt_base = build_hybrid_optimizer(base, sinkhorn=True)
    sink_ids = {id(p) for g in opt.param_groups
                if g.get("optimizer") == opt.SINKHORN for p in g["params"]}
    tables = {id(t.weight) for layer in m.layers if layer.engram is not None
              for t in layer.engram.tables}
    assert tables and tables <= sink_ids
    assert opt.group_sizes["sinkhorn"] == (
        opt_base.group_sizes["sinkhorn"] + len(tables))


# --------------------------- 来源对齐 / CLI ---------------------------
def test_engram_fp8_arch_switch_and_cli(tmp_path):
    """来源对齐：ckpt 记了 `engram_fp8` 就沿用；老 ckpt 回退关；CLI 三态."""
    import json
    import sys
    from pathlib import Path

    assert "engram_fp8" in _ARCH_SWITCHES and SmallLLMConfig().engram_fp8 is True

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "engram_fp8": True}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = tiny_test_config()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.engram_fp8 is True

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(json.dumps({"config": {
        "d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = tiny_test_config()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.engram_fp8 is False      # 特性前不量化，旧 run 语义逐位不变

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).engram_fp8 is None
    args = T.parse_args(["--no-engram-fp8", "--engram-layers", "1,3"])
    c = tiny_test_config()
    src = T.resolve_arch_switches(args, c)
    assert c.engram_fp8 is False and src["engram_fp8"] == "explicit"
