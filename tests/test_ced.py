"""CED 单测 — `ced_split`（DeepSeek-V4.1 §2.2 因果编码器-解码器）.

核心契约：
- 前 `ced_split` 层是**因果编码器**；其后各层（解码器）的全局 KV **不再**由自己的
  hidden 产生，而是由第 `ced_split` 层的输出 `H_enc` 用**逐层投影权重**投影
  （`C_l = H_enc W_l^{KV}`，rope 键同理）。Q 与（可选）层内 SWA 仍来自自己。
- `ced_split=0`（默认）逐位等价旧行为、不新增键；`>0` 时解码器层的
  `w_dkv`/`w_kr` 换成 `w_ced_kv`/`w_ced_kr`（改键、opt-in）。
- 解码器区间必须**全是全注意力层**（线性层的递推状态无法投影）——构建期早失败。
- 解码器层恒为 CSA2 Full（layer-dependent 投影，不能借别的层）。
- 预填充跳算（`ced_prefill_skip`，默认开）：解码器层只跑最后 `swa_window` 个 token；
  `swa_window=0` 时**与全量 prefill 逐位等价**（解码器层没有局部依赖），
  `swa_window>0` 时是论文承认的近似（但明显优于"只跑末位置"）。
- 因果性 / Finite 梯度（含 `w_ced_kv` 拿到梯度、梯度检查点下梯度仍回流到编码器）。
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig, sync_arch_from_checkpoint  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402


def _cfg(**over) -> SmallLLMConfig:
    """全注意力小模型（CED 要求解码器区间全是全注意力层）."""
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


def _enc_hidden(m: TinyLLM, ids: torch.Tensor) -> torch.Tensor:
    """手算 `H_enc`：前 `ced_split` 层串行后的残差流 hidden（全序列）."""
    h = m.embed(ids)
    for layer in m.layers[:m.ced_split]:
        h, _ = layer(h, None, False)
    return h


# ------------------------------- 关：逐位等价 -------------------------------
def test_ced_off_is_bitwise_identical_and_adds_no_keys():
    """`ced_split=0`（默认）：键集合不变、无 `w_ced_*`、输出逐位相同."""
    torch.manual_seed(0)
    a = TinyLLM(_cfg())
    torch.manual_seed(0)
    b = TinyLLM(_cfg(ced_split=0))
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any("w_ced_" in k for k in a.state_dict())
    ids = torch.randint(0, 256, (1, 16))
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.equal(a.encode_full_hidden(ids), b.encode_full_hidden(ids))


def test_ced_on_swaps_kv_projection_keys():
    """开启后解码器层**只有** `w_ced_kv`/`w_ced_kr`（没有自己的 `w_dkv`/`w_kr`）."""
    m = TinyLLM(_cfg(ced_split=2))
    for i, layer in enumerate(m.layers):
        attn = layer.attn
        if i >= 2:
            assert attn.ced is True
            assert attn.w_dkv is None and attn.w_kr is None
            assert isinstance(attn.w_ced_kv, torch.nn.Linear)
            assert isinstance(attn.w_ced_kr, torch.nn.Linear)
        else:
            assert attn.ced is False
            assert not hasattr(attn, "w_ced_kv")
            assert isinstance(attn.w_dkv, torch.nn.Linear)
    keys = set(m.state_dict())
    assert "layers.2.attn.w_ced_kv.weight" in keys
    assert "layers.2.attn.w_dkv.weight" not in keys


# ------------------------------- 语义：KV 真的来自编码器 -------------------------------
def test_ced_decoder_latent_is_projected_from_encoder_output():
    """解码器层缓存里的 latent == `w_ced_kv(H_enc)`（H_enc = 编码器最后一层输出）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        h_enc = _enc_hidden(m, ids)
        _, pasts = m._prefill(ids, max_new_tokens=4)
        for i in (2, 3):
            want = m.layers[i].attn.w_ced_kv(h_enc)
            got = pasts[i][0][:, :12]
            assert torch.allclose(got, want, atol=1e-6), (
                i, float((got - want).abs().max()))
            # rope 键同理
            want_kr = m.layers[i].attn.w_ced_kr(h_enc)
            assert torch.allclose(pasts[i][1][:, :12], want_kr, atol=1e-6)


def test_ced_decoder_output_does_not_depend_on_own_kv_projection():
    """去掉解码器层的 `w_ced_kv`（置零）会改输出；改别的层不会（隔离性）.

    更强的契约：解码器层的全局 KV 只经 `H_enc` 一条路进来——把 `w_ced_kv` 置零，
    输出必须变；把**编码器**的 `w_dkv` 置零，解码器层的 KV 仍由 `H_enc` 决定
    （H_enc 自己会变，所以这条不测隔离性，只测"确实用了 w_ced_kv"）。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        base = m.encode_full_hidden(ids)
        with torch.no_grad():
            m.layers[2].attn.w_ced_kv.weight.zero_()
        changed = m.encode_full_hidden(ids)
    assert float((base - changed).abs().max()) > 1e-4


# ------------------------------- 因果 / 一致性 -------------------------------
def test_ced_causal_and_finite():
    """改未来 token 不影响前部；CED 投影参数梯度 finite 且非 None."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2))
    ids = torch.randint(0, 256, (1, 20))
    ids2 = ids.clone()
    ids2[0, 16] = (ids2[0, 16] + 7) % 256
    m.eval()
    with torch.no_grad():
        a = m.encode_full_hidden(ids)
        b = m.encode_full_hidden(ids2)
    assert (a[0, :16] - b[0, :16]).abs().max() < 1e-6, "未来 token 污染前部"
    assert (a[0, 16:] - b[0, 16:]).abs().max() > 0
    m.train()
    m(ids, targets=ids)["loss"].backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{name}"
    assert m.layers[2].attn.w_ced_kv.weight.grad is not None
    assert float(m.layers[2].attn.w_ced_kv.weight.grad.abs().max()) > 0


def test_ced_grad_ckpt_still_reaches_encoder():
    """梯度检查点下 `ced_src` 必须仍把梯度带回编码器层（闭包捕获会静默断梯度）.

    做法：同一权重、开/关梯度检查点各跑一步，比较编码器层权重的梯度。
    """
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 16))
    a = TinyLLM(_cfg(ced_split=2))
    b = TinyLLM(_cfg(ced_split=2, grad_ckpt=True))
    b.load_state_dict(a.state_dict())
    a.train(); b.train()
    a(ids, targets=ids)["loss"].backward()
    b(ids, targets=ids)["loss"].backward()
    ga = dict(a.named_parameters())
    gb = dict(b.named_parameters())
    worst = 0.0
    for name, pa in ga.items():
        if pa.grad is None:
            continue
        assert gb[name].grad is not None, f"梯度检查点下缺梯度：{name}"
        worst = max(worst, float((pa.grad - gb[name].grad).abs().max()))
    assert worst < 1e-5, worst
    # 编码器层（含 w_dkv）必须拿到非零梯度——断掉的话这里会是 0
    enc_kv = ga["layers.0.attn.w_dkv.weight"].grad
    assert enc_kv is not None and float(enc_kv.abs().max()) > 0


# ------------------------------- 预填充跳算 -------------------------------
def test_ced_prefill_skip_is_exact_without_swa():
    """SWA 关 + 跳算：`prefill` 的末位置 hidden 与全量前向**逐位等价**.

    解码器层没有局部依赖（全局 KV 来自编码器、无 SWA），所以只跑最后 1 个 token
    就够——其余位置的输出没有任何消费者。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    assert m._ced_prefill_q_off(24) == 20
    with torch.no_grad():
        pre, pasts = m._prefill(ids, max_new_tokens=4)
        full = m.encode_full_hidden(ids)
    assert torch.allclose(pre[:, -1:], full[:, -1:], atol=1e-6), (
        float((pre[:, -1:] - full[:, -1:]).abs().max()))
    # 缓存仍是全序列：pos == 序列长，解码器层 latent 覆盖全部位置
    for i in (2, 3):
        assert int(pasts[i][2].item()) == ids.shape[1]


def test_ced_prefill_skip_off_matches_full_reference():
    """`ced_prefill_skip=False`：把跳算关掉后，prefill 的末位置与全量完全一致."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2, ced_prefill_skip=False))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    assert m._ced_prefill_q_off(24) == 0
    with torch.no_grad():
        pre, _ = m._prefill(ids, max_new_tokens=4)
        full = m.encode_full_hidden(ids)
    assert torch.allclose(pre, full, atol=1e-6)


def test_ced_prefill_skip_with_swa_replays_two_windows():
    """SWA 开 + 跳算：尾段取 **2 个窗口**，末位置与被读到的环行都与全量一致.

    为什么是 2W 而不是 W：窗口列的 K/V 由窗口内各位置**自己的 hidden** 生成，
    "将来会被解码读到"的那 W 个环行自身也要自洽，于是再往前留一个 W（尾段首部
    那段窗口不全会偏，但偏差沿残差快速衰减）。实测 tail=2W 时末位置逐位一致，
    tail=W 时差 1e-1 量级——这条单测把口径锁死。
    """
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2, swa_window=4))
    m.eval()
    ids = torch.randint(0, 256, (1, 24))
    assert m._ced_prefill_q_off(24) == 16          # 尾段 8 = 2*swa_window
    with torch.no_grad():
        pre, pasts = m._prefill(ids, max_new_tokens=4)
        full = m.encode_full_hidden(ids)
        err = float((pre[:, -1:] - full[:, -1:]).abs().max())
        h, _ = m._decode_step(torch.randint(0, 256, (1, 1)), pasts)
    assert err < 1e-6, err
    assert torch.isfinite(h).all()
    # 对比：只留 1 个窗口（W）时末位置有明显偏差——证明 2W 不是随便定的
    m._ced_prefill_q_off = lambda n: 20
    with torch.no_grad():
        pre_w, _ = m._prefill(ids, max_new_tokens=4)
    err_w = float((pre_w[:, -1:] - full[:, -1:]).abs().max())
    assert err_w > 1e-2, err_w


def test_ced_prefill_plus_decode_matches_full_forward():
    """跳算 prefill + 增量解码 == 全前向（SWA 关时严格）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg(ced_split=2))
    m.eval()
    ids = torch.randint(0, 256, (1, 25))
    with torch.no_grad():
        full = m.encode_full_hidden(ids)
        _, pasts = m._prefill(ids[:, :24], max_new_tokens=4)
        step, _ = m._decode_step(ids[:, 24:], pasts)
    assert torch.allclose(step, full[:, 24:], atol=1e-6), (
        float((step - full[:, 24:]).abs().max()))


# ------------------------------- 约束与来源 -------------------------------
def test_ced_requires_full_attention_decoder():
    """解码器区间含线性注意力层 → 构建期早失败（否则跳算静默算错）."""
    cfg = _cfg(full_attn_every=2, ced_split=2)  # 层 2 全注意力、层 3 线性
    with pytest.raises(ValueError, match="必须全是全注意力层"):
        TinyLLM(cfg)
    with pytest.raises(ValueError, match="必须小于层数"):
        TinyLLM(_cfg(ced_split=4))


def test_ced_decoder_layers_are_full_mode():
    """解码器层恒为 Full（不借前层缓存）：layer-dependent 投影不能被借用合并."""
    m = TinyLLM(_cfg(ced_split=2, csa_reuse_every=2))
    for i in (2, 3):
        assert m.layers[i].attn.csa_mode == "full"
        assert m.layers[i].attn.csa_reuse is False
    # 编码器层仍按 CSA2 规则（层 1 是 reuse）
    assert m.layers[1].attn.csa_reuse is True


def test_ced_checkpoint_alignment_and_cli(tmp_path):
    """来源对齐：ckpt 记了 `ced_split` 就沿用；老 ckpt 回退 0=关；CLI 三态."""
    import json

    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(json.dumps({"config": {
        "ced_split": 2}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = _cfg()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.ced_split == 2

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(json.dumps({"config": {
        "d_model": 128}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = _cfg()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.ced_split == 0
    assert not any("w_ced_" in k for k in TinyLLM(cfg2).state_dict())

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).ced_split is None
    args = T.parse_args(["--ced-split", "2", "--no-ced-prefill-skip"])
    assert args.ced_split == 2 and args.ced_prefill_skip is False
    c = _cfg()
    src = T.resolve_arch_switches(args, c)
    assert c.ced_split == 2 and src["ced_split"] == "explicit"
    assert c.ced_prefill_skip is False and src["ced_prefill_skip"] == "explicit"
