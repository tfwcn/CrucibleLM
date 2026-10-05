"""RETRO V2 交错融合单测 — 默认关闭时旧行为逐位不变，开后恒等起点 + 解码一致."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.model import TinyLLM
from src.llm.local.retro import build_batch_chunk_ids


def _tiny_interleaved(every: int = 2, chunk_len: int = 16):
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.retro_enabled = True
    cfg.retro_every = every
    cfg.retro_chunk_len = chunk_len
    m = TinyLLM(cfg).eval()
    return m, cfg


def _fake_chunks(vocab: int = 256, b: int = 1, k: int = 2, length: int = 16):
    ids = torch.randint(4, vocab, (b, k, length))
    mask = torch.ones(b, k, length, dtype=torch.bool)
    return ids, mask


def test_interleaved_identity_at_init():
    """交错融合零初始化：接 chunk 与不接 chunk 数值等价（恒等起点）."""
    m, _ = _tiny_interleaved()
    x = torch.randint(0, 256, (1, 12))
    ids, mask = _fake_chunks()
    with torch.no_grad():
        o_no = m(x)["logits"]
        o_with = m(x, chunk_ids=ids, chunk_mask=mask)["logits"]
    assert torch.allclose(o_no, o_with, atol=1e-5)


def test_chunk_builder_shapes_and_masks():
    """chunk 构造器：形状 (b,K,L)，短文截断、缺文补空行（全 False）."""

    class Tok:
        def encode(self, text, add_bos=True):
            return [ord(c) % 250 + 4 for c in text]

    hits = [["abcdefghij", "xy"], []]
    ids, mask = build_batch_chunk_ids(Tok(), hits, k=2, chunk_len=6)
    assert ids.shape == (2, 2, 6) and mask.shape == (2, 2, 6)
    # 第一行第二篇 "xy"：前 2 True 后 4 False；第二行全空：全 False
    assert mask[0, 1].tolist() == [True, True, False, False, False, False]
    assert not mask[1].any()
    assert (ids[1] == 0).all()


def test_interleaved_decode_matches_full_forward_when_trained():
    """回归：交错融合 w_o 非零后，prefill+解码必须与全前向一致（含 _decode_step 同步）."""
    m, _ = _tiny_interleaved()
    with torch.no_grad():
        for layer in m.layers:
            if layer.retro is not None:
                layer.retro.w_o.weight.normal_(std=0.02)
    x = torch.randint(0, 256, (1, 10))
    ids, mask = _fake_chunks()
    with torch.no_grad():
        full = m(x, chunk_ids=ids, chunk_mask=mask)["logits"]
        # 手工复刻 forward 的 frozen 编码（与模型内逻辑同值）
        b, kk, ll = ids.shape
        mem_h = m.embed(ids.reshape(b, kk * ll)).reshape(b, kk * ll, -1)
        rmem = (mem_h, mask.reshape(b, kk * ll))
        h, pasts = m._prefill(x[:, :9], rmem)
        h2, _ = m._decode_step(x[:, 9:10], pasts, rmem)
        step_logits = m.lm_head(h2)
    assert torch.allclose(full[:, 9:10], step_logits, atol=1e-5)


def test_interleaved_gradients_finite():
    """交错融合梯度有限（融合参数有 grad，无 NaN；frozen 编码不吃梯度）."""
    m, _ = _tiny_interleaved()
    m.train()
    x = torch.randint(0, 256, (1, 8))
    ids, mask = _fake_chunks()
    loss = m(x, targets=x, chunk_ids=ids, chunk_mask=mask)["loss"]
    loss.backward()
    n_retro_grad = 0
    for name, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
        if "retro" in name and p.grad is not None:
            n_retro_grad += 1
    assert n_retro_grad > 0


def test_interleaved_state_dict_keys():
    """关闭时无 retro 键；交错开后仅层级 retro 键新增（无单点 model.retro）."""
    torch.manual_seed(0)
    m0 = TinyLLM(tiny_test_config())
    assert not any("retro" in k for k in m0.state_dict())
    m1, _ = _tiny_interleaved()
    keys1 = set(m1.state_dict())
    assert any("retro" in k for k in keys1)
    assert m1.retro is None
    assert sum(getattr(layer, "retro", None) is not None
               for layer in m1.layers) > 0
    # 旧权重可 overlap 载入（缺的仅 retro 键）
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and all("retro" in k for k in missing)


def test_chunk_without_interleaved_raises():
    """retro_every=0 的模型传 chunk_ids 直接报错（防 v1/v2 接错线）."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.retro_enabled = True
    m = TinyLLM(cfg).eval()
    x = torch.randint(0, 256, (1, 8))
    ids, mask = _fake_chunks()
    with pytest.raises(ValueError):
        m(x, chunk_ids=ids, chunk_mask=mask)
