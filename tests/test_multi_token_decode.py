"""多 token 增量解码回归 — 投机验证依赖的并行解码与逐 token / 全前向一致."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.mla import MLAModule
from src.llm.local.sparse_attn import SparseMLAModule


def test_mla_multi_token_decode_matches_full():
    """MLA：prefill 后一次解码多 token == 一次全前向对应段."""
    torch.manual_seed(0)
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256)
    m.eval()
    h = torch.randn(1, 16, 128)
    with torch.no_grad():
        full, _ = m(h)
        _, past = m(h[:, :8], None, True, 16)
        multi, _ = m(h[:, 8:16], past)  # 一次解码 8 个
    assert torch.allclose(multi, full[:, 8:], atol=1e-5), (multi - full[:, 8:]).abs().max()


def test_mla_multi_token_decode_is_causal():
    """MLA 多 token 解码块内因果：改未来解码 token，前面输出不变（曾漏掩码）."""
    torch.manual_seed(0)
    m = MLAModule(128, 4, 32, 32, 16, max_seq_len=256)
    m.eval()
    h = torch.randn(1, 16, 128)
    h2 = h.clone()
    h2[0, 11] += 50.0  # 改解码块的第 4 个 token（未来）
    with torch.no_grad():
        _, p1 = m(h[:, :8], None, True, 16)
        o1, _ = m(h[:, 8:12], p1)
        _, p2 = m(h[:, :8], None, True, 16)
        o2, _ = m(h2[:, 8:12], p2)
    diff = (o1 - o2).abs()
    assert diff[0, :3].max() < 1e-4, "块内未来 token 污染了前面位置（因果破坏）"
    assert diff[0, 3].max() > 0, "同位置应有响应（测试本身有效）"


def test_sparse_multi_token_decode_matches_full():
    """稀疏路径：多 token 解码与全前向一致（窗口盖住全序列场景）."""
    torch.manual_seed(0)
    m = SparseMLAModule(128, 4, 32, 32, 16, max_seq_len=256,
                        sparse_threshold=100000, sparse_window=256,
                        sparse_sink=128, sparse_stride=64, sparse_chunk=32)
    m.eval()
    h = torch.randn(1, 16, 128)
    with torch.no_grad():
        full, _ = m(h)
        _, past = m(h[:, :8], None, True, 16)
        multi, _ = m(h[:, 8:16], past)
    assert torch.allclose(multi, full[:, 8:], atol=1e-4), (multi - full[:, 8:]).abs().max()


def test_linear_multi_token_decode_matches_sequential():
    """线性层：一次解码多 token == 逐个 t=1 解码（前向精确，紧公差）."""
    torch.manual_seed(0)
    g = GatedDeltaLite(128, 4)
    g.eval()
    h = torch.randn(1, 16, 128)
    with torch.no_grad():
        _, pa = g(h[:, :8])
        _, pb = g(h[:, :8])
        multi, _ = g(h[:, 8:16], pa)  # 一次 8 个
        outs = []
        cur = pb
        for i in range(8, 16):
            o, cur = g(h[:, i:i + 1], cur)  # 逐个
            outs.append(o)
        seq = torch.cat(outs, dim=1)
    assert torch.allclose(multi, seq, atol=1e-5), (multi - seq).abs().max()
