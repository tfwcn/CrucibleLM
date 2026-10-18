"""MTP 深度单测 — 级联多 token 预测（DeepSeek-V4.1 num_nextn_predict_layers=3）.

CPU 可跑。核心契约：
- `mtp_depth=1` 与旧实现**逐位一致**（键名 `mtp.norm`/`mtp.proj` 不变，可续训旧 run）；
- `mtp_depth=D` 增加 D-1 个浅层头（`mtp.extra.*`），第 d 级预测 t+1+d，逐级链式；
- 目标对齐：第 d 级只用 `targets[:, d+1 : d+1+n]`（改别的 target 位不该影响它）；
- 链式：第 d 级读第 d-1 级的输出；第 d（≥2）级还读"下一 token"嵌入；
- Finite：各级反向无 NaN；
- 来源对齐：`sync_arch_from_checkpoint` 读 mtp_depth，旧 ckpt 回退 1；CLI 三态。
"""

import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.llm.local.config import (  # noqa: E402
    SmallLLMConfig,
    sync_arch_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.model import TinyLLM  # noqa: E402


def _tiny(depth: int = 1, seed: int = 0) -> TinyLLM:
    torch.manual_seed(seed)
    cfg = tiny_test_config()
    cfg.mtp_depth = depth
    m = TinyLLM(cfg)
    m.eval()
    return m


# ------------------------------- 默认关/键兼容 -------------------------------
def test_depth1_keys_and_no_extra():
    """深度 1：MTP 键与旧实现一致（无 mtp.extra.*）."""
    m = _tiny(1)
    keys = list(m.state_dict())
    assert "mtp.norm.weight" in keys
    assert "mtp.proj.weight" in keys
    assert not any("mtp.extra" in k for k in keys)


def test_depth3_adds_extra_keys():
    """深度 3：多出 2 级浅层头（每级 norm_h/norm_e/proj），且不破坏前向."""
    m = _tiny(3)
    keys = list(m.state_dict())
    for i in (0, 1):
        for w in ("norm_h", "norm_e", "proj"):
            assert f"mtp.extra.{i}.{w}.weight" in keys
    # 每组拼接投影：2*d_model 输入
    assert m.mtp.extra[0].proj.weight.shape[1] == 2 * m.config.d_model


def test_default_config_depth1():
    """config 默认深度 1（键兼容旧 run）."""
    assert SmallLLMConfig().mtp_depth == 1


# ------------------------------- 深度 1 逐位回归 -------------------------------
def test_depth1_matches_legacy_formula():
    """深度 1 的 mtp_loss == 旧口径：CE(lm_head(mtp(h_{:T-2})), targets[:,2:]) 逐位一致."""
    m = _tiny(1)
    ids = torch.randint(0, 256, (1, 9))
    captured = {}
    m.final_norm.register_forward_hook(
        lambda mod, inp, outp: captured.__setitem__("h", outp.detach()))
    with torch.no_grad():
        out = m(ids, targets=ids)
    h = captured["h"]
    t = ids.shape[1]
    ref = F.cross_entropy(
        m.lm_head(m.mtp(h[:, :t - 2])).reshape(-1, m.config.vocab_size),
        ids[:, 2:].reshape(-1),
    )
    assert torch.allclose(out["mtp_loss"], ref, atol=1e-6)
    assert len(out["mtp_level_losses"]) == 1


# ------------------------------- 深度 3 正确性 -------------------------------
def test_depth3_levels_and_mean():
    """深度 3：三级 loss 全 finite，mtp_loss == 三级均值，末级预测 t+4."""
    m = _tiny(3, seed=1)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        out = m(ids, targets=ids)
    lv = out["mtp_level_losses"]
    assert len(lv) == 3
    assert all(torch.isfinite(x) for x in lv)
    assert torch.allclose(out["mtp_loss"], torch.stack(lv).mean(), atol=1e-6)
    # n = T-1-D = 8；第 d 级目标为 targets[:, d+1 : d+1+8]，末级（d=3）到 t+4
    # （用不同 target 与 ids 分离验证：下面 test_target_alignment 覆盖对齐）


def test_target_alignment_per_level():
    """目标对齐：第 d 级只被 `targets[:, d+1 : d+1+n]` 影响（改独占位只动该级）."""
    m = _tiny(3, seed=2)
    ids = torch.randint(0, 256, (1, 10))
    lab = torch.randint(0, 256, (1, 10))
    n = 10 - 1 - 3  # 6
    with torch.no_grad():
        base = m(ids, targets=lab)["mtp_level_losses"]
        # 索引 2 只被第 1 级用（[2,8)），索引 9 只被第 3 级用（[4,10)）
        lab2 = lab.clone()
        lab2[0, 2] = (lab2[0, 2] + 1) % 256
        only_l1 = m(ids, targets=lab2)["mtp_level_losses"]
        lab3 = lab.clone()
        lab3[0, 9] = (lab3[0, 9] + 1) % 256
        only_l3 = m(ids, targets=lab3)["mtp_level_losses"]
    assert not torch.allclose(base[0], only_l1[0]), "第 1 级应对 target 位 2 敏感"
    assert torch.allclose(base[1], only_l1[1]) and torch.allclose(base[2], only_l1[2])
    assert not torch.allclose(base[2], only_l3[2]), "第 3 级应对 target 位 9 敏感"
    assert torch.allclose(base[0], only_l3[0]) and torch.allclose(base[1], only_l3[1])
    assert n == 6


def test_chaining_and_token_embedding_used():
    """链式：改第 1 级投影会动更深级；改第 d 级输入嵌入会动该级（且梯度非零）."""
    m = _tiny(3, seed=3)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = m(ids, targets=ids)["mtp_level_losses"]
        # 打乱第 1 级投影 → 第 2/3 级（链式下游）也随之改变
        m.mtp.proj.weight.copy_(torch.randn_like(m.mtp.proj.weight) * 0.05)
        after = m(ids, targets=ids)["mtp_level_losses"]
    assert not torch.allclose(base[0], after[0])
    assert not torch.allclose(base[1], after[1]), "第 2 级应链式依赖第 1 级输出"
    assert not torch.allclose(base[2], after[2]), "第 3 级应链式依赖第 2 级输出"

    # 第 ≥2 级确实读了"下一 token"嵌入（梯度非零）
    m.train()
    out = m(ids, targets=ids)
    out["loss"].backward()
    assert float(m.mtp.extra[0].norm_e.weight.grad.abs().sum()) > 0, "未用到 token 嵌入"


def test_depth3_grads_finite():
    """深度 3：前向反向各处无 NaN 梯度."""
    m = _tiny(3, seed=4)
    m.train()
    ids = torch.randint(0, 256, (2, 12))
    out = m(ids, targets=ids)
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_depth3_train_step_finite():
    """深度 3 跑通 train_step（CPU），loss finite 且 mtp_loss 存在."""
    import math

    from src.llm.local.train import train_step

    m = _tiny(3, seed=5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = torch.randint(0, 256, (2, 16))
    res = train_step(m, opt, ids)
    assert math.isfinite(res["loss"])
    assert res.get("mtp_loss") is not None and math.isfinite(res["mtp_loss"])


def test_too_short_sequence_skips_mtp():
    """序列太短（T ≤ D+1）时跳过 MTP，不报错（无 mtp_loss 键）."""
    m = _tiny(3, seed=6)
    ids = torch.randint(0, 256, (1, 4))  # n = 4-1-3 = 0
    out = m(ids, targets=ids)
    assert "mtp_loss" not in out
    assert torch.isfinite(out["loss"])


# ------------------------------- 来源对齐 -------------------------------
def test_sync_reads_mtp_depth(tmp_path):
    """装载既有权重时对齐 mtp_depth；旧格式回退 1."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "hparams.json").write_text(
        json.dumps({"config": {"mtp_depth": 3}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"")
    cfg = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg, str(d / "model.pt"))
    assert cfg.mtp_depth == 3

    d2 = tmp_path / "old"
    d2.mkdir()
    (d2 / "hparams.json").write_text(
        json.dumps({"config": {"d_model": 768}}), encoding="utf-8")
    (d2 / "model.pt").write_bytes(b"")
    cfg2 = SmallLLMConfig()
    sync_arch_from_checkpoint(cfg2, str(d2 / "model.pt"))
    assert cfg2.mtp_depth == 1


def test_cli_mtp_depth_tristate():
    """CLI：不传=None（沿用/默认），传入即显式深度."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_local_llm as T

    assert T.parse_args([]).mtp_depth is None
    assert T.parse_args(["--mtp-depth", "3"]).mtp_depth == 3
