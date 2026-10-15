"""跨设备专家 offload 单测 — 数值等价 / LRU / 磁盘往返 / 默认关（CPU 可跑）."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.moe import FineGrainedMoE
from src.llm.local.model import TinyLLM
from src.llm.local.offload import ExpertStore


def _moe(seed=0):
    """构造小 MoE（CPU 秒级）."""
    torch.manual_seed(seed)
    m = FineGrainedMoE(64, n_experts=8, top_k=2, expert_hidden=16, n_shared=1)
    m.eval()
    return m


def test_offload_cpu_matches_grouped():
    """offload(cpu) 与分组 bmm 数值一致（含 aux）."""
    m = _moe()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        o_g, aux_g = m(x)
    m2 = _moe()
    m2.load_state_dict(m.state_dict())
    m2.enable_offload("cpu", target="cpu")
    with torch.no_grad():
        o_o, aux_o = m2(x)
    assert torch.allclose(o_g, o_o, atol=1e-5), (o_g - o_o).abs().max()
    assert torch.allclose(aux_g, aux_o)


@pytest.mark.parametrize("capacity", [0, 2, 3])
def test_offload_disk_matches_grouped(tmp_path, capacity):
    """offload(disk) 各容量下与分组一致（含 capacity<命中数的淘汰场景）."""
    m = _moe()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        o_g, _ = m(x)
    m2 = _moe()
    m2.load_state_dict(m.state_dict())
    m2.enable_offload("cpu", target="disk", offload_dir=str(tmp_path),
                      capacity=capacity)
    with torch.no_grad():
        o_o, _ = m2(x)
    assert torch.allclose(o_g, o_o, atol=1e-5), (o_g - o_o).abs().max()


def test_offload_default_off_uses_grouped():
    """开关关时不建 store，仍走分组快路径（默认关契约）."""
    m = _moe()
    assert m.store is None
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        m(x)
    assert m.store is None


def test_expert_store_lru_eviction(tmp_path):
    """LRU：超容量淘汰最久未用；命中即提序."""
    m = _moe()
    store = ExpertStore(m.experts, "cpu", target="disk",
                        offload_dir=str(tmp_path), capacity=2)
    store.ensure([0, 1, 2])
    assert store.hot_ids() == [1, 2]
    store.ensure([1])          # 命中 1，提序
    assert store.hot_ids() == [2, 1]
    store.ensure([0])          # 新入 0，淘汰最旧的 2
    assert store.hot_ids() == [1, 0]


def test_expert_store_disk_preserves_weights(tmp_path):
    """磁盘往返后专家权重逐位不变."""
    m = _moe()
    orig = [e.w_gate.weight.detach().clone() for e in m.experts]
    store = ExpertStore(m.experts, "cpu", target="disk", offload_dir=str(tmp_path))
    for i in range(len(m.experts)):
        store.ensure([i])  # 逐个读回
        assert torch.equal(m.experts[i].w_gate.weight, orig[i])


def test_expert_store_invalid_target():
    """非法目标/磁盘缺目录：构建即报错（fail-fast）."""
    m = _moe()
    with pytest.raises(ValueError, match="cpu/disk"):
        ExpertStore(m.experts, "cpu", target="tpu")
    with pytest.raises(ValueError, match="offload_dir"):
        ExpertStore(m.experts, "cpu", target="disk", offload_dir="")


def test_expert_store_disk_paths_namespaced(tmp_path):
    """多层共用一个 offload_dir 时，不同 store 的落盘文件不能互相覆盖.

    回归：早期 `_disk_path` 恒为 `expert_{i}.pt`，后建层静默顶掉先建层权重。"""
    m1 = _moe(seed=1)
    m2 = _moe(seed=2)
    ref1 = [e.w_gate.weight.detach().clone() for e in m1.experts]
    ref2 = [e.w_gate.weight.detach().clone() for e in m2.experts]
    s1 = ExpertStore(m1.experts, "cpu", target="disk",
                     offload_dir=str(tmp_path), name="layer0")
    s2 = ExpertStore(m2.experts, "cpu", target="disk",
                     offload_dir=str(tmp_path), name="layer1")
    for i in range(len(m1.experts)):
        s1.ensure([i])
        s2.ensure([i])
        assert torch.equal(m1.experts[i].w_gate.weight, ref1[i])
        assert torch.equal(m2.experts[i].w_gate.weight, ref2[i])


def test_expert_store_prefetch_fills_but_never_evicts(tmp_path):
    """预取只在有空位时搬入，绝不淘汰（避免挤掉马上要用的专家）."""
    m = _moe()
    store = ExpertStore(m.experts, "cpu", target="disk",
                        offload_dir=str(tmp_path), capacity=2)
    store.ensure([0, 1])
    assert store.hot_ids() == [0, 1]
    moves = store.moves
    # 容量已满：预取应整体跳过，热集不变、不再搬运
    store.ensure([5, 6], prefetch=True)
    assert store.hot_ids() == [0, 1]
    assert store.moves == moves
    # 正常 ensure 仍会淘汰最旧的（淘汰权留给真正要用的那步）
    store.ensure([7])
    assert store.hot_ids() == [1, 7]
    # 腾出空位后，预取可填空位但不替换现有热专家
    m2 = _moe()
    s2 = ExpertStore(m2.experts, "cpu", target="disk",
                     offload_dir=str(tmp_path), name="s2", capacity=3)
    s2.ensure([0])
    s2.ensure([1, 2], prefetch=True)
    assert s2.hot_ids() == [0, 1, 2]


def test_model_setup_offload_disk_end_to_end(tmp_path):
    """整模型磁盘 offload：开/关 logits 与生成逐位一致（含多层 + 淘汰）."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    m = TinyLLM(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base_logits = m(x)["logits"]
        base = m.generate(x, max_new_tokens=16)
    cfg.offload_enabled = True
    cfg.offload_target = "disk"
    cfg.offload_dir = str(tmp_path)
    cfg.offload_capacity = 2  # < top_k，强制淘汰
    m.setup_offload()
    m.eval()
    with torch.no_grad():
        off_logits = m(x)["logits"]
        off = m.generate(x, max_new_tokens=16)
    assert torch.allclose(base_logits, off_logits, atol=1e-5)
    assert torch.equal(base, off)


def test_model_setup_offload_end_to_end(tmp_path):
    """整模型接入：offload 开/关 logits 一致、生成一致."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    m = TinyLLM(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base_logits = m(x)["logits"]
        base = m.generate(x, max_new_tokens=16)
    cfg.offload_enabled = True
    cfg.offload_target = "cpu"
    m.setup_offload()
    m.eval()
    with torch.no_grad():
        off_logits = m(x)["logits"]
        off = m.generate(x, max_new_tokens=16)
    assert torch.allclose(base_logits, off_logits, atol=1e-5)
    assert torch.equal(base, off)
