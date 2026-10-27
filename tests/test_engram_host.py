"""宿主 Engram 表单测 — 取行逐位一致 / 反传只回命中行 / 行稀疏 Adagrad / 不随 to() 搬.

对齐 `docs/details/host-resident-memory.md`：记忆表留 CPU、只把命中行搬上算力设备。
四条硬约束：
1. **前向逐位一致**：宿主表与设备表在相同权重下输出必须相同（否则"分设备"改的是语义）；
2. **反向不物化全表**：`weight.grad` 保持 None，梯度只进旁路队列；
3. **只动命中行**：一次 step 后未命中行必须逐字节不变（行稀疏的判据）；
4. **不随 `model.to()` 搬**：宿主表主权重永远在 CPU（这才是"不占显存"）。
"""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config  # noqa: E402
from src.llm.local.engram import EngramMemory, EngramTable  # noqa: E402
from src.llm.local.kv_quant import fake_quant  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.optim import SparseRowAdagrad, build_hybrid_optimizer  # noqa: E402


def _dev() -> str:
    """有卡用卡（覆盖真 H2D），无卡退化到 CPU（CI 环境）."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ===== 1. 前向：宿主取行 == 稠密取行 =====


def test_lookup_host_matches_dense():
    """宿主表取行 + FP8 量化，必须与"整表 index_select + fake_quant"逐位一致."""
    torch.manual_seed(0)
    t = EngramTable(rows=257, cols=32, fp8=True, host=True)
    torch.nn.init.normal_(t.weight, std=0.3)   # 空表是未初始化内存，NaN 会让比较失效
    idx = torch.randint(0, 257, (4, 16))
    got = t.lookup(idx, _dev(), torch.float32)

    ref_rows = t.weight.detach().index_select(0, idx.reshape(-1))
    ref = fake_quant(ref_rows.reshape(4, 16, 32), "fp8")
    assert torch.equal(got.cpu(), ref)


def test_lookup_host_rejects_gpu_index():
    """宿主表的索引必须先在 CPU 上算好（省掉逐表 D2H 同步）."""
    t = EngramTable(rows=31, cols=32, host=True)
    bad = torch.zeros(2, 3, dtype=torch.long)
    if torch.cuda.is_available():
        bad = bad.cuda()
        with pytest.raises(ValueError, match="CPU"):
            t.lookup(bad, "cuda", torch.float32)
    else:
        # 无卡时该分支不可达，明确跳过而不是假装通过
        pytest.skip("需要 CUDA 才能构造「非 CPU 索引」的用例")


def test_memory_module_host_matches_device_path():
    """整模块：宿主表与设备表在同权重下输出一致（先打破 W_V 零初始化）."""
    torch.manual_seed(1)
    kw = dict(d_model=64, vocab_size=512, orders=(2, 3), n_heads=2,
              d_mem=32, table_size=127, fp8=True, dropout=0.0)
    mem_dev = EngramMemory(**kw, host=False)
    mem_host = EngramMemory(**kw, host=True)
    with torch.no_grad():
        # 全零 W_V 会让两条路径都输出 0，测不出差异 —— 必须打破恒等起点
        mem_dev.w_v.weight.normal_(0.0, 0.02)
    mem_host.load_state_dict(mem_dev.state_dict())

    h = torch.randn(2, 8, 64)
    ids = torch.randint(0, 500, (2, 10))
    with torch.no_grad():
        a = mem_dev(h, ids)
        b = mem_host(h, ids)
    assert torch.allclose(a, b, atol=1e-6), "宿主表路径改了语义"


# ===== 2. 反向：只回命中行，不物化全表梯度 =====


def test_backward_only_returns_hit_rows():
    """反传只把命中行塞进旁路队列，`weight.grad` 必须保持 None."""
    t = EngramTable(rows=100, cols=32, fp8=False, host=True)
    idx = torch.tensor([[1, 2], [2, 3]])
    out = t.lookup(idx, "cpu", torch.float32)
    out.sum().backward()

    assert t.weight.grad is None, "反传物化了全表梯度（行稀疏失效）"
    assert len(t._grad_sink) == 1
    got_idx, got_g = t._grad_sink[0]
    assert got_idx.tolist() == [1, 2, 2, 3], "命中行索引丢失/乱序"
    assert torch.equal(got_g, torch.ones(4, 32))


def test_backward_accumulates_across_micro_batches():
    """梯度累积：多次 backward 的条目都留在队列里，step 时一次消费."""
    t = EngramTable(rows=50, cols=32, fp8=False, host=True)
    for _ in range(3):
        t.lookup(torch.tensor([5, 6]), "cpu", torch.float32).sum().backward()
    assert len(t._grad_sink) == 3
    opt = SparseRowAdagrad([t], lr=0.1)
    assert opt.pending_rows() == 6
    opt.step()
    assert opt.pending_rows() == 0, "step 后旁路队列未清空"
    assert len(t._grad_sink) == 0


# ===== 3. 行稀疏 Adagrad：口径正确 + 未命中行不动 =====


def test_sparse_adagrad_matches_dense_reference():
    """与稠密 Adagrad 同口径（同 lr/eps），且未命中行逐字节不变."""
    torch.manual_seed(2)
    t = EngramTable(rows=50, cols=32, fp8=False, host=True)
    w0 = t.weight.detach().clone()

    idx = torch.tensor([3, 7, 3])          # 行 3 被命中两次 → 梯度应相加
    w = torch.tensor([[1.0], [2.0], [3.0]])  # 每次命中的上游梯度不同
    out = t.lookup(idx, "cpu", torch.float32)
    (out * w).sum().backward()

    lr, eps = 0.1, 1e-10
    opt = SparseRowAdagrad([t], lr=lr, eps=eps)
    opt.step()

    # 参考：稠密口径（全表累加器 + 全表更新），只比较被命中行与未命中行
    ref = w0.clone()
    g = torch.zeros_like(w0)
    g.index_add_(0, idx, torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1).expand(3, 32))
    g2 = g * g
    upd = g / (g2.sqrt() + eps)
    ref = ref - lr * upd

    assert torch.equal(t.weight[~torch.isin(torch.arange(50), torch.tensor([3, 7]))],
                       w0[~torch.isin(torch.arange(50), torch.tensor([3, 7]))]), \
        "未命中行被改动了（不是行稀疏更新）"
    assert torch.allclose(t.weight[[3, 7]], ref[[3, 7]], atol=1e-5)


def test_sparse_adagrad_state_bytes_is_one_table():
    """累加器是惰性分配的一整张表（= 1× 表大小），不随步数增长."""
    t = EngramTable(rows=100, cols=32, fp8=False, host=True)
    opt = SparseRowAdagrad([t], lr=0.1)
    assert opt.state_bytes() == 0, "未 step 就分配了累加器"
    for _ in range(3):
        t.lookup(torch.tensor([1, 2]), "cpu", torch.float32).sum().backward()
        opt.step()
    assert opt.state_bytes() == 100 * 32 * 4


def test_zero_grad_clears_pending_sink():
    """zero_grad 应丢弃未消费的行梯度，防跨步残留."""
    t = EngramTable(rows=20, cols=32, fp8=False, host=True)
    t.lookup(torch.tensor([1]), "cpu", torch.float32).sum().backward()
    opt = SparseRowAdagrad([t], lr=0.1)
    assert opt.pending_rows() == 1
    opt.zero_grad()
    assert opt.pending_rows() == 0


# ===== 4. 打包（推理）：宿主 FP8 存储与未打包逐位一致 =====


def test_pack_host_roundtrip_bitwise():
    """宿主表打包成真 FP8 后取行，与未打包路径逐位一致（1.06 B/通道）."""
    torch.manual_seed(3)
    t = EngramTable(rows=64, cols=32, fp8=True, host=True)
    torch.nn.init.normal_(t.weight, std=0.3)   # 空表全是未初始化内存，量化会出怪值
    idx = torch.randint(0, 64, (2, 8))
    t.eval()
    before = t.lookup(idx, "cpu", torch.float32).clone()
    t.pack()
    after = t.lookup(idx, "cpu", torch.float32)
    assert t.w8 is not None and t.w8.device.type == "cpu", "打包存储应留在 host"
    assert torch.equal(before, after)
    t.unpack()
    assert torch.equal(t.lookup(idx, "cpu", torch.float32), before)


# ===== 5. 设备策略：宿主表不随 model.to() 搬家 =====


def _tiny_engram_model(host: bool) -> TinyLLM:
    cfg = tiny_test_config()
    cfg.engram_every = 2
    cfg.engram_table_size = 127
    cfg.engram_d_mem = 32
    cfg.engram_orders = "2,3"
    cfg.engram_heads = 2
    cfg.engram_host = host
    return TinyLLM(cfg)


def test_host_flag_defaults_off():
    """默认关：既有 run 的结构与优化器分组不受影响."""
    from src.llm.local.config import _ARCH_SWITCHES

    cfg = tiny_test_config()
    assert cfg.engram_host is False
    # 登记进架构开关清单 → 续训从 hparams 沿用，不会静默把表搬回显存
    assert "engram_host" in _ARCH_SWITCHES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能验证 to(cuda)")
def test_host_tables_stay_on_cpu_after_to_cuda():
    """`model.to('cuda')` 之后宿主表主权重仍在 CPU（= 不占显存），且前向可跑."""
    model = _tiny_engram_model(host=True).to("cuda")
    tables = [m for m in model.modules()
              if isinstance(m, EngramTable) and m.host]
    assert tables, "配置未生效：没有宿主表"
    assert all(t.weight.device.type == "cpu" for t in tables)

    x = torch.randint(0, 64, (2, 8), device="cuda")
    with torch.no_grad():
        out = model(x)
    logits = out["logits"] if isinstance(out, dict) else (
        out[0] if isinstance(out, tuple) else out)
    assert logits.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能验证 to(cuda)")
def test_host_tables_not_in_dense_optimizer_groups():
    """宿主表不能进 Muon/Adam 组（否则又会物化全表梯度）."""
    model = _tiny_engram_model(host=True).to("cuda")
    opt = build_hybrid_optimizer(model, sinkhorn=True)
    assert opt.group_sizes["host"] > 0
    host_ids = {id(t.weight) for t in opt._host_tables(model)}
    dense_ids = {id(p) for g in opt.muon.param_groups + opt.adam.param_groups
                 for p in g["params"]}
    if opt.sinkhorn_opt is not None:
        dense_ids |= {id(p) for g in opt.sinkhorn_opt.param_groups
                      for p in g["params"]}
    assert not (host_ids & dense_ids), "宿主表混进了稠密优化器组"


# ===== 6. 优化器存盘/续跑：宿主组必须能被识别与恢复 =====

def test_combined_optimizer_roundtrip_with_host_group():
    """存盘带 `host` 组；恢复后累加器逐位一致、lr 以存盘为准."""
    torch.manual_seed(4)
    m = _tiny_engram_model(host=True)
    torch.nn.init.normal_(m.layers[0].engram.w_v.weight, std=0.02)
    opt = build_hybrid_optimizer(m, sinkhorn=True, host_lr=0.05)
    ids = torch.randint(0, m.config.vocab_size, (2, 8))
    m(ids)["logits"].sum().backward()
    opt.step()
    opt.zero_grad()

    sd = opt.state_dict()
    assert "host" in sd and "host" in sd["kind"], "宿主组没进存盘"

    m2 = _tiny_engram_model(host=True)
    opt2 = build_hybrid_optimizer(m2, sinkhorn=True, host_lr=0.01)
    opt2.load_state_dict(sd)
    assert opt2.host_opt is not None and opt2.host_opt.lr == 0.05
    a1 = [opt.host_opt.state[id(t.weight)] for t in opt.host_opt.tables]
    a2 = [opt2.host_opt.state[id(t.weight)] for t in opt2.host_opt.tables]
    assert len(a1) == len(a2)
    for x, y in zip(a1, a2):
        assert torch.equal(x, y), "宿主表累加器没有逐位恢复"


def test_optimizer_rejects_group_mismatch():
    """换了宿主/设备表必须**大声报错**（否则静默用错优化器口径）."""
    torch.manual_seed(5)
    m = _tiny_engram_model(host=True)
    opt = build_hybrid_optimizer(m, sinkhorn=True)
    ids = torch.randint(0, m.config.vocab_size, (2, 8))
    m(ids)["logits"].sum().backward()
    opt.step()
    sd = opt.state_dict()

    m_dev = _tiny_engram_model(host=False)      # 表回到显存 → 分组不同
    opt_dev = build_hybrid_optimizer(m_dev, sinkhorn=True)
    with pytest.raises(RuntimeError, match="分组"):
        opt_dev.load_state_dict(sd)
