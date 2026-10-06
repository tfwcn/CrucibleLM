"""会话缓存/RETRO/记忆层单测 — 全部默认路径（flag 关）下旧行为逐位不变."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig, tiny_test_config
from src.llm.local.memory import ProductKeyMemory, init_memory_from_activations
from src.llm.local.model import TinyLLM
from src.llm.local.session_cache import SessionCache


def _tiny_eval():
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    m.eval()
    return m


def test_session_cache_equals_full_forward():
    """SessionCache：首段 prefill + 逐 token 续跑 == 一次全前向末位 hidden."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (1, 23))
    with torch.no_grad():
        full = m(x)["logits"]
    sc = SessionCache(m)
    h1 = sc.extend(x[:, :10])  # prefill 10
    h2 = None
    for t in range(10, 23):
        h2 = sc.extend(x[:, t:t + 1])  # decode 13 步
    assert h2 is not None
    # extend 返回末位置向量 (b,d)；与全前向末位 logits 应一致（同缓存数学）
    with torch.no_grad():
        logits = m.lm_head(h2.unsqueeze(1))
    assert torch.allclose(logits, full[:, -1:], atol=1e-5)


def test_session_cache_save_load(tmp_path):
    """存盘读回后继续续跑 == 不间断续跑（长度差 1 位的等价性弱化，查接口与状态）."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (1, 12))
    sc = SessionCache(m)
    sc.extend(x[:, :6])
    assert sc.ids.shape == (1, 6) and len(sc.pasts) == len(m.layers)
    p = str(tmp_path / "sc.pt")
    sc.save(p)
    sc2 = SessionCache.load(p, m)
    assert sc2.ids.shape == (1, 6) and len(sc2.pasts) == len(m.layers)
    h = sc2.extend(x[:, 6:7])
    assert torch.isfinite(h).all()


def test_retro_disabled_path_unchanged():
    """retro_enabled=False：forward 不接 mem 时等同旧路径（config 默认就是 False）."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (2, 16))
    out = m(x)
    assert out["logits"].shape == (2, 16, 256)
    assert m.retro is None


def test_retro_enabled_identity_at_init():
    """RetroFusion 零初始化输出投影时：接 mem 与不接 mem 在数值上等价（恒等融合）."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.retro_enabled = True
    m = TinyLLM(cfg).eval()
    x = torch.randint(0, 256, (1, 12))
    mem = torch.randn(1, 3, cfg.d_model)
    with torch.no_grad():
        o_no = m(x)["logits"]
        o_with = m(x, mem=mem)["logits"]
    assert torch.allclose(o_no, o_with, atol=1e-5)


def test_memory_zero_values_identity():
    """ProductKeyMemory：value 全零时残差恒等（初始化安全）+ 形状保持."""
    torch.manual_seed(0)
    pk = ProductKeyMemory(64, n_slots=64, top_k=4)
    x = torch.randn(2, 10, 64)
    assert torch.allclose(pk(x), x, atol=1e-6)
    assert pk(x).shape == x.shape


def test_memory_init_from_activations_identity():
    """B 方案初始化：value 保持零 => 初始化后模块输出仍恒等（可锁定）."""
    torch.manual_seed(0)
    pk = ProductKeyMemory(64, n_slots=64, top_k=4)
    hiddens = torch.randn(64, 64)
    info = init_memory_from_activations(pk, hiddens)
    assert info["n_samples"] == 64 and info["inertia"] >= 0
    x = torch.randn(2, 5, 64)
    assert torch.allclose(pk(x), x, atol=1e-6)


def test_memory_gradients_finite():
    """记忆层梯度有限（key/value 都有 grad_fn，无 NaN）."""
    torch.manual_seed(0)
    pk = ProductKeyMemory(32, n_slots=16, top_k=2)
    x = torch.randn(1, 4, 32)
    loss = pk(x).pow(2).mean()
    loss.backward()
    for p in pk.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_memory_enabled_state_dict_compatible():
    """memory_every=0 时无 memory 模块，state_dict 与旧版一致；开启后多出对应键."""
    torch.manual_seed(0)
    m0 = TinyLLM(tiny_test_config())
    cfg = tiny_test_config()
    cfg.memory_every = 2
    m1 = TinyLLM(cfg)
    keys0 = set(m0.state_dict())
    keys1 = set(m1.state_dict())
    # 旧键都在，新键仅 memory 相关
    assert keys0 <= keys1
    assert any("memory" in k for k in keys1 - keys0)
    # 旧版权重可载入开启记忆层的新模型（行续接路径已是 .to(device) 严格模式，
    # 此处用 load 的非严格语义对比差异键仅 memory）
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and all("memory" in k for k in missing)


def test_memory_decode_matches_full_forward_when_trained():
    """回归：_decode_step 必须同步走记忆层.

    零初始化时 memory 恒等，漏掉也看不出来；value 训出非零后，
    prefill+单步解码必须仍与全前向逐位一致（曾差 ~0.69）。
    """
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.memory_every = 2
    m = TinyLLM(cfg).eval()
    with torch.no_grad():
        for layer in m.layers:
            if layer.memory is not None:
                layer.memory.values.normal_(std=0.1)
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        full = m(x)["logits"]
        h, pasts = m._prefill(x[:, :9])
        h2, _ = m._decode_step(x[:, 9:10], pasts)
        step_logits = m.lm_head(h2)
    assert torch.allclose(full[:, 9:10], step_logits, atol=1e-5)


def test_retro_all_masked_rows_stay_finite_identity():
    """全行被 mask（无有效记忆）时不許出 NaN，退化为零增量恒等."""
    from src.llm.local.retro import RetroFusion

    torch.manual_seed(0)
    rf = RetroFusion(64, 4).eval()
    h = torch.randn(1, 5, 64)
    mem = torch.randn(1, 3, 64)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with torch.no_grad():
        o = rf(h, mem, mask)
    assert torch.isfinite(o).all()
    assert torch.allclose(o, h, atol=1e-6)


def test_memory_init_with_fewer_samples_than_slots():
    """校准样本比 side 还少时不崩（有放回采样），且仍保持恒等起点."""
    torch.manual_seed(0)
    pk = ProductKeyMemory(32, n_slots=64, top_k=4)  # side=8, 只给 5 个样本
    info = init_memory_from_activations(pk, torch.randn(5, 32))
    assert info["n_samples"] == 5 and info["inertia"] >= 0
    x = torch.randn(1, 3, 32)
    assert torch.allclose(pk(x), x, atol=1e-6)


def test_session_cache_grows_and_stays_correct():
    """ensure_room：预留打满后自动 2x 扩，续跑仍与全前向一致."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        full = m(x)["logits"]
    sc = SessionCache(m)
    sc.extend(x[:, :4], max_new_tokens=2)  # 只留 2 步余量
    h = sc.extend(x[:, 4:8])  # 4 步 > 余量，触发扩容
    with torch.no_grad():
        logits = m.lm_head(h.unsqueeze(1))
    assert torch.allclose(logits, full[:, 7:8], atol=1e-5)
