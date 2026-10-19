"""优化器新件单测：head-wise Muon（#14）+ Sinkhorn 更新（#15）.

V4.1 §2.5：
- **head-wise Muon**：Q/K 的上投影按**头**切块后各自做 Newton-Schulz 正交化——
  等价于"每个头一个预条件器"，能处理注意力头之间的异质性；
- **Sinkhorn 更新**（Algorithm 1）：给 embedding / 预测头 / Engram 表这类
  "token×特征"大矩阵，用"Nesterov 动量 + 行/列交替 L2 归一化"（Sinkhorn 平衡）
  替代 Adam。只存一份动量缓冲；`Δ = √n·U^(K)` 使行/列 RMS ≈ 1；
  有效学习率 `η̃ = γη`（γ=0.18，对齐 Adam 的更新幅度），且**不做权重衰减**。

本文件锁：批量 NS5 与逐头循环逐位一致 / Sinkhorn 的数学（行/列 RMS、√n 因子、
近零行屏蔽、K 次交替的奇偶语义）/ 两个优化器的单步更新方向与幅度 / 分组路由
（Q/K 进 head-wise、embed/head/Engram 表进 Sinkhorn）/ 存盘往返与开关不匹配早失败 /
端到端可训。
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import SmallLLMConfig  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.optim import (  # noqa: E402
    CombinedOptimizer,
    Muon,
    SinkhornAdamLike,
    build_hybrid_optimizer,
    head_split_columns,
    sinkhorn_balance,
    zeropower_batched_newtonschulz5,
    zeropower_via_newtonschulz5,
)


def _cfg(**over) -> SmallLLMConfig:
    base = dict(
        vocab_size=256, d_model=64, n_layers=2, full_attn_every=1,
        n_heads=4, q_lora_rank=16, kv_lora_rank=16, qk_rope_dim=8,
        n_experts=4, top_k=1, expert_hidden=32, n_shared=1,
        mtp_depth=0, max_seq_len=64, sparse_threshold=4, sparse_window=4,
        sparse_sink=2, sparse_stride=2, sparse_chunk=4,
        engram_every=1, engram_orders="2,3",
    )
    base.update(over)
    return SmallLLMConfig(**base)


# ------------------------------- head-wise Muon -------------------------------
def test_batched_newtonschulz_matches_per_head_loop():
    """批量 NS5 与"逐头循环调普通 NS5"逐位一致（同一 a/b/c、同 fp32 内算）."""
    torch.manual_seed(0)
    g = torch.randn(4, 8, 6)
    batched = zeropower_batched_newtonschulz5(g, steps=5)
    loop = torch.stack([zeropower_via_newtonschulz5(g[i], steps=5)
                        for i in range(g.shape[0])])
    assert torch.allclose(batched, loop, atol=1e-6), (
        float((batched - loop).abs().max()))


def test_head_split_requires_divisible_rows():
    """按头切分要求行数能被头数整除（否则说明不是按头堆叠的权重）."""
    w = torch.randn(32, 8)
    view, per = head_split_columns(w, 4)
    assert view.shape == (4, 8, 8) and per == 8
    with pytest.raises(ValueError, match="整除"):
        head_split_columns(torch.randn(30, 8), 4)


def test_muon_head_wise_equals_per_head_muon():
    """head-wise 组的一步更新 == 把每个头当成独立参数各跑一次标准 Muon."""
    torch.manual_seed(0)
    w = torch.randn(8, 4)                 # 4 头 × 每头 2 行
    g = torch.randn(8, 4)
    lr, mu, wd = 0.05, 0.9, 0.01
    # 参照：逐头独立 Muon
    ref = w.clone()
    heads = [torch.nn.Parameter(w[i * 2:(i + 1) * 2].clone()) for i in range(4)]
    for h in heads:
        h.grad = g[i * 2:(i + 1) * 2].clone() if False else None
    for i, h in enumerate(heads):
        h.grad = g[i * 2:(i + 1) * 2].clone()
    opt_ref = Muon(heads, lr=lr, momentum=mu, weight_decay=wd)
    opt_ref.step()
    for i, h in enumerate(heads):
        ref[i * 2:(i + 1) * 2] = h.detach()
    # 被测：head-wise 组
    p = torch.nn.Parameter(w.clone())
    p.grad = g.clone()
    opt = Muon([p], lr=lr, momentum=mu, weight_decay=wd)
    opt.param_groups[0]["head_wise"] = True
    opt.param_groups[0]["n_heads"] = 4
    opt.step()
    assert torch.allclose(p.detach(), ref, atol=1e-6), (
        float((p.detach() - ref).abs().max()))


# ------------------------------- Sinkhorn -------------------------------
def test_sinkhorn_balance_equalizes_row_and_column_rms():
    """K=11 次交替后：未被屏蔽的行/列 RMS 都 ≈1，且整体缩放 = √n."""
    torch.manual_seed(0)
    g = torch.randn(64, 16) * 3.0
    u = sinkhorn_balance(g, steps=11, tau=1e-12)
    row_rms = u.pow(2).mean(dim=1).sqrt()
    col_rms = u.pow(2).mean(dim=0).sqrt()
    assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=5e-2), (
        float((row_rms - 1).abs().max()))
    assert torch.allclose(col_rms, torch.ones_like(col_rms), atol=5e-2), (
        float((col_rms - 1).abs().max()))


def test_sinkhorn_balance_masks_near_zero_rows():
    """`ρ_i ≤ τ·ρ̄` 的行被置 0（论文第 5 行）；τ=0 时不屏蔽任何行."""
    g = torch.randn(32, 8)
    g[5] = 0.0                     # 一行全零
    g[7] = 1e-9                    # 数值上近零
    u = sinkhorn_balance(g, steps=11, tau=1e-3)
    assert float(u[5].abs().max()) == 0.0
    assert float(u[7].abs().max()) == 0.0
    # 对照：τ=0（不屏蔽）时那行会被"归一化"放大到单位范数——这正是论文要
    # 用 τ 屏蔽近零行的原因（否则噪声被抬成信号）；纯零行仍是 0。
    u0 = sinkhorn_balance(g, steps=11, tau=0.0)
    assert float(u0[5].abs().max()) == 0.0
    assert float(u0[7].abs().max()) > 0.1, float(u0[7].abs().max())


def test_sinkhorn_balance_matches_reference_loop():
    """与逐行/逐列的朴素参考实现逐位一致（奇数步行、偶数步列的语义）."""
    torch.manual_seed(0)
    g = torch.randn(8, 5)
    steps, eps, tau = 5, 1e-20, 1e-9
    x = g.float().clone()
    rho = x.norm(dim=1)
    x = x * (rho > tau * rho.mean()).to(x.dtype).unsqueeze(1)
    for k in range(1, steps + 1):
        if k % 2 == 1:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(eps)
        else:
            x = x / x.norm(dim=0, keepdim=True).clamp_min(eps)
    ref = x * (g.shape[1] ** 0.5)
    got = sinkhorn_balance(g, steps=steps, eps=eps, tau=tau)
    assert torch.allclose(got, ref, atol=1e-7)


def test_sinkhorn_opt_step_direction_and_magnitude():
    """单步 = `-η·γ·Δ`（Δ 为平衡后的动量），且不做权重衰减、只存一份状态."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 4))
    p.grad = torch.randn(16, 4)
    lr, gamma, mu = 1e-3, 0.18, 0.95
    opt = SinkhornAdamLike([p], lr=lr, momentum=mu, lr_scale=gamma, steps=11)
    before = p.detach().clone()
    opt.step()
    # 期望：动量 = 0（初始）→ buf = (1-μ)·g；Nesterov = (1-μ)·g
    nes = (1 - mu) * p.grad
    want = before - lr * gamma * sinkhorn_balance(nes, steps=11)
    assert torch.allclose(p.detach(), want, atol=1e-7)
    st = opt.state[p]
    assert set(st) == {"momentum_buffer"}, st.keys()  # 只有一份动量（无二阶矩）


def test_sinkhorn_opt_handles_zero_grad_and_finite():
    """全零梯度不产生 NaN/Inf（行/列归一化的 eps 保护）."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    p.grad = torch.zeros(8, 4)
    opt = SinkhornAdamLike([p], lr=1e-3)
    opt.step()
    assert torch.isfinite(p.detach()).all()


# ------------------------------- 分组装配 -------------------------------
def test_combined_optimizer_routes_params_to_groups():
    """Q/K 上投影进 head-wise；embed/lm_head/Engram 表进 Sinkhorn；其余按 2D/非 2D 分."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg())
    opt = build_hybrid_optimizer(
        m, muon_lr=0.02, adam_lr=3e-4, head_wise=True,
        n_heads=m.config.n_heads, sinkhorn=True)
    names = {id(p): n for n, p in m.named_parameters()}
    head_names = {names[id(p)] for g in opt.head_wise_muon.param_groups
                  for p in g["params"]}
    sink_names = {names[id(p)] for g in opt.sinkhorn_opt.param_groups
                  for p in g["params"]}
    adam_names = {names[id(p)] for g in opt.adam.param_groups
                  for p in g["params"]}
    assert head_names and all(any(k in n for k in ("w_uq", "w_qr", "w_uk"))
                              for n in head_names), sorted(head_names)[:3]
    assert sink_names and all(("embed" in n or "lm_head" in n
                               or ".tables." in n) for n in sink_names)
    assert not (head_names & sink_names)
    # norm/bias/1D 在 Adam 组
    assert any("norm" in n for n in adam_names)
    assert "layers.0.attn.w_dq.weight" in {names[id(p)]
                                          for p in opt.muon.param_groups[0]["params"]}


def test_combined_optimizer_state_roundtrip_and_mismatch():
    """存盘往返；开关组合变了必须早失败（旧动量与新分组对不上）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg())
    torch.manual_seed(0)
    a = build_hybrid_optimizer(m, head_wise=True, n_heads=4, sinkhorn=True)
    ids = torch.randint(0, 256, (1, 8))
    out = m(ids, targets=ids)
    out["loss"].backward()
    a.step()
    sd = a.state_dict()
    assert sd["kind"] == "muon+headwise+sinkhorn+adamw"
    b = build_hybrid_optimizer(m, head_wise=True, n_heads=4, sinkhorn=True)
    b.load_state_dict(sd)
    c = build_hybrid_optimizer(m, head_wise=False, sinkhorn=True)
    with pytest.raises(RuntimeError, match="开关变了"):
        c.load_state_dict(sd)
    d = build_hybrid_optimizer(m, head_wise=True, n_heads=4, sinkhorn=False)
    with pytest.raises(RuntimeError, match="开关变了"):
        d.load_state_dict(sd)
    with pytest.raises(RuntimeError, match="不是 Muon 存盘"):
        d.load_state_dict({"kind": "adamw"})


def test_combined_optimizer_trains_end_to_end():
    """三组都开时连续 5 步 loss 下降（数值稳定、无 NaN）."""
    torch.manual_seed(0)
    m = TinyLLM(_cfg())
    opt = build_hybrid_optimizer(
        m, muon_lr=0.02, adam_lr=1e-3, head_wise=True,
        n_heads=m.config.n_heads, sinkhorn=True)
    ids = torch.randint(0, 256, (1, 16))
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = m(ids, targets=ids)
        loss = out["loss"]
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], losses
