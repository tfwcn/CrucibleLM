"""Muon 优化器 — 隐藏层 2D 权重的正交化动量 SGD（Moonshot/Keller Jordan 路线）.

标准混合配方：2D 隐藏权重走 Muon，其余（embedding/norm/偏置/卷积）走 AdamW。
Muon 单份动量状态（Adam 两份），优化器显存减半；收敛约 2x 计算效率。

- 正交化用 Newton-Schulz 迭代（fp32 内算，防 bf16 病态），高瘦矩阵自动转置；
- **head-wise Muon**（V4.1 §2.5）：Q/K 的上投影按**头**切块后各自正交化——
  等价于"每头一个预条件器"，能处理注意力头之间的异质性（GLM5/Kimi-K3 同样验证）；
- 解耦权重衰减 + 动量（默认 momentum=0.95, wd=0.01, lr=0.02，按需调）；
- **Sinkhorn 更新**（V4.1 §2.5 Algorithm 1）：embedding / lm_head / Engram 表这类
  "token×特征"大矩阵用"动量 + Sinkhorn 平衡"替代 Adam——同样只存一份动量缓冲，
  行/列交替归一化把更新的行/列 RMS 拉平，实测优于 Adam（省优化器状态）；
- CombinedOptimizer 包装两者，state_dict 兼容训练脚本的存盘/续跑；
  注意：AdamW checkpoint 的 optim.pt 与 Muon 不互通，切换优化器需新开
  动量（权重 model.pt 照常复用）。
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Newton-Schulz 正交化（fp32 内算，返回与输入同形状同设备）."""
    assert g.dim() == 2, "Muon 只处理 2D 参数"
    orig_dtype = g.dtype
    x = g.float()
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x = x / (x.norm() + eps)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(orig_dtype)


def zeropower_batched_newtonschulz5(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """批量 Newton-Schulz：`g` 形状 (H, r, c)，对每个"头"独立正交化.

    head-wise Muon 用它：把 (n_heads*rows_per_head, cols) 的行切成 H 块，
    逐块正交化等价于"每头一个预条件器"（V4.1 §2.5）。数学与
    `zeropower_via_newtonschulz5` 逐位一致（同 a/b/c 系数、同 fp32 内算）。
    """
    assert g.dim() == 3, "批量版只处理 (H, r, c)"
    orig_dtype = g.dtype
    x = g.float()
    # 高瘦矩阵转置（在"头内"维度上转置）
    transposed = False
    if x.shape[1] > x.shape[2]:
        x = x.transpose(1, 2)
        transposed = True
    x = x / (x.norm(dim=(1, 2), keepdim=True) + eps)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        a_mat = torch.bmm(x, x.transpose(1, 2))
        b_mat = b * a_mat + c * torch.bmm(a_mat, a_mat)
        x = a * x + torch.bmm(b_mat, x)
    if transposed:
        x = x.transpose(1, 2)
    return x.to(orig_dtype)


def head_split_columns(
    w: torch.Tensor, n_heads: int,
) -> tuple[torch.Tensor, int]:
    """把 (n_heads*per, in) 的权重按行切成 (H, per, in)（返回视图与每头行数）.

    Q/K 的上投影都是"每头一段连续行"，所以按行均分即可；不是整数倍时早失败
    （说明该权重不是按头堆叠的，用普通 Muon 更安全）。
    """
    rows, cols = w.shape
    if n_heads <= 1 or rows % n_heads != 0:
        raise ValueError(
            f"head-wise Muon 要求行数 {rows} 能被头数 {n_heads} 整除")
    per = rows // n_heads
    return w.view(n_heads, per, cols), per


class Muon(torch.optim.Optimizer):
    """纯 Muon：2D 参数正交化动量更新（配合 AdamW 管 1D/embedding 用）."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, {"lr": lr, "momentum": momentum,
                                  "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        """单步更新（无闭包需求，closure 透传占位）.

        `group["head_wise"]` 为真时按 `group["n_heads"]` 把 2D 权重按行切块、
        逐头正交化（head-wise Muon）。动量缓冲与普通 Muon 同形状/同精度。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            head_wise = bool(group.get("head_wise", False))
            n_heads = int(group.get("n_heads", 1))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if head_wise:
                    batch, _ = head_split_columns(g, n_heads)
                    orth = zeropower_batched_newtonschulz5(batch).reshape_as(g)
                else:
                    orth = zeropower_via_newtonschulz5(g)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # 正交化当前梯度后做重球动量（fp32 状态，防 bf16 累积误差）
                buf_fp32 = buf.float() if buf.is_floating_point() else buf
                buf_fp32.mul_(mu).add_(orth.float())
                state["momentum_buffer"] = buf_fp32.to(p.dtype) if p.is_floating_point() else buf_fp32
                if wd > 0:
                    p.data.mul_(1 - lr * wd)  # 解耦权重衰减
                p.data.add_(state["momentum_buffer"], alpha=-lr)
        return loss


@torch.no_grad()
def sinkhorn_balance(
    g: torch.Tensor, steps: int = 11, eps: float = 1e-20,
    tau: float = 1e-3,
) -> torch.Tensor:
    """Sinkhorn 平衡（V4.1 §2.5 Algorithm 1 的第 3~17 行）.

    输入动量（Nesterov）矩阵 `g` (m, n)：先按行把近零行置 0（`ρ_i ≤ τ·ρ̄`），
    再**行/列交替**做 L2 归一化（`K` 次，奇数步行、偶数步列），最后乘 `sqrt(n)`
    把"单位行 L2 范数"换算成"单位行 RMS"。返回与输入同形状同 dtype。

    语义：把更新的行/列 RMS 都拉平成 ≈1 —— 对 embedding / 预测头这种
    "一行一个 token、一列一个特征"的大矩阵，等价于给每个 token 一个公平的步长，
    不会让高频 token 的更新把低频 token 淹没（论文：实测优于 Adam）。
    """
    orig_dtype = g.dtype
    x = g.float()
    m, n = x.shape
    # 近零行屏蔽（论文第 3~5 行）：用行 L2 范数与均值比
    rho = x.norm(dim=1)
    rho_bar = rho.mean()
    if float(rho_bar) > 0:
        x = x * (rho > tau * rho_bar).to(x.dtype).unsqueeze(1)
    for k in range(1, int(steps) + 1):
        if k % 2 == 1:   # 行归一化
            norm = x.norm(dim=1, keepdim=True)
            x = x / norm.clamp_min(eps)
        else:            # 列归一化
            norm = x.norm(dim=0, keepdim=True)
            x = x / norm.clamp_min(eps)
    x = x * (n ** 0.5)   # 单位行 L2 → 单位行 RMS
    return x.to(orig_dtype)


class SinkhornAdamLike(torch.optim.Optimizer):
    """动量 + Sinkhorn 平衡的更新器（替代 Adam，给 embedding/lm_head/Engram 表）.

    与 Muon 同构：只存**一份**动量缓冲（Adam 两份），Nesterov 动量 + Sinkhorn
    平衡取代 Newton-Schulz。**不做权重衰减**（论文：Sinkhorn 更新不施加 wd），
    有效学习率乘 `lr_scale`（论文 γ=0.18，用于对齐 Adam 的更新幅度）。
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        lr_scale: float = 0.18,
        steps: int = 11,
        eps: float = 1e-20,
        tau: float = 1e-3,
    ):
        super().__init__(params, {"lr": lr, "momentum": momentum,
                                  "lr_scale": lr_scale, "steps": steps,
                                  "eps": eps, "tau": tau})

    @torch.no_grad()
    def step(self, closure=None):
        """单步：动量 → Nesterov → Sinkhorn 平衡 → 按 η·γ 更新."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            eta = group["lr"] * group["lr_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() != 2:
                    raise ValueError(
                        "Sinkhorn 更新只处理 2D 矩阵（embedding/lm_head/Engram 表）")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"].float()
                # Algorithm 1 第 1~2 行：动量 + Nesterov 组合梯度
                buf.mul_(mu).add_(g.float(), alpha=1 - mu)
                nesterov = buf * mu + g.float() * (1 - mu)
                delta = sinkhorn_balance(
                    nesterov, steps=group["steps"], eps=group["eps"],
                    tau=group["tau"])
                state["momentum_buffer"] = buf.to(p.dtype) if p.is_floating_point() else buf
                p.data.add_((delta.to(p.dtype) * eta).neg())
        return loss


class CombinedOptimizer:
    """Muon（2D 隐藏权重）+ AdamW（其余）的联合优化器.

    对外与单个优化器同接口：zero_grad/step/state_dict/load_state_dict，
    param_groups 汇总两边（lr 调度按组名区分：muon 组与 adam 组）。
    """

    MUON = "muon"
    ADAM = "adam"
    SINKHORN = "sinkhorn"

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_wd: float = 0.01,
        adam_lr: float = 3e-4,
        head_wise: bool = False,
        n_heads: int = 1,
        sinkhorn: bool = False,
        sinkhorn_lr_scale: float = 0.18,
        sinkhorn_steps: int = 11,
    ):
        muon_params: list = []      # 普通 Muon（2D 隐藏权重）
        head_params: list = []      # head-wise Muon（Q/K 上投影，按头分块）
        sink_params: list = []      # Sinkhorn（embedding / lm_head / Engram 表）
        adam_params: list = []      # 其余（norm/bias/1D）
        # Q/K 上投影的名字（每头一段连续行）；论文：head-wise Muon for Q and K
        head_names = ("w_uq", "w_qr", "w_uk")
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # token×特征大矩阵：token embedding / 预测头 / Engram 表
            # （Engram 表的参数名形如 `layers.N.engram.tables.K.weight`）
            is_emb = ("embed" in name or "lm_head" in name
                      or ("engram" in name and ".tables." in name))
            if is_emb and sinkhorn and p.dim() == 2:
                # token×特征 大矩阵：动量 + Sinkhorn 平衡（省 Adam 的两份状态）
                sink_params.append(p)
            elif p.dim() == 2 and not is_emb:
                if head_wise and any(hn in name for hn in head_names):
                    rows = p.shape[0]
                    if n_heads > 1 and rows % n_heads == 0:
                        head_params.append(p)
                        continue
                muon_params.append(p)
            else:
                adam_params.append(p)
        try:
            is_cuda = next(model.parameters()).is_cuda
        except StopIteration:
            is_cuda = False
        self.muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=muon_wd)
        self.head_wise_muon = None
        if head_params:
            self.head_wise_muon = Muon(
                head_params, lr=muon_lr, momentum=muon_momentum,
                weight_decay=muon_wd)
            for g in self.head_wise_muon.param_groups:
                g["head_wise"] = True
                g["n_heads"] = int(n_heads)
        self.sinkhorn_opt = None
        if sink_params:
            self.sinkhorn_opt = SinkhornAdamLike(
                sink_params, lr=adam_lr, lr_scale=sinkhorn_lr_scale,
                steps=sinkhorn_steps)
        self.adam = torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.1,
            fused=is_cuda and torch.cuda.is_available(),
        )
        # 直接引用内部组字典（lr 调度就地写，浅拷贝会断开关联）
        self.param_groups: list = []
        for g in self.muon.param_groups:
            g["optimizer"] = self.MUON
            self.param_groups.append(g)
        if self.head_wise_muon is not None:
            for g in self.head_wise_muon.param_groups:
                g["optimizer"] = self.MUON   # lr 调度与普通 Muon 同组
                self.param_groups.append(g)
        if self.sinkhorn_opt is not None:
            for g in self.sinkhorn_opt.param_groups:
                g["optimizer"] = self.SINKHORN
                self.param_groups.append(g)
        for g in self.adam.param_groups:
            g["optimizer"] = self.ADAM
            self.param_groups.append(g)
        # 便于日志/单测：各组参数个数
        self.group_sizes = {
            "muon": len(muon_params), "muon_head_wise": len(head_params),
            "sinkhorn": len(sink_params), "adam": len(adam_params),
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零所有组梯度."""
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.head_wise_muon is not None:
            self.head_wise_muon.zero_grad(set_to_none=set_to_none)
        if self.sinkhorn_opt is not None:
            self.sinkhorn_opt.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """各组合走一步."""
        self.muon.step()
        if self.head_wise_muon is not None:
            self.head_wise_muon.step()
        if self.sinkhorn_opt is not None:
            self.sinkhorn_opt.step()
        self.adam.step()

    def state_dict(self) -> dict:
        """存盘格式（含 optimizer 标记，跨版本可辨）."""
        out = {"kind": "muon+adamw", "muon": self.muon.state_dict(),
               "adam": self.adam.state_dict()}
        if self.head_wise_muon is not None:
            out["head_wise_muon"] = self.head_wise_muon.state_dict()
            out["kind"] = "muon+headwise+adamw"
        if self.sinkhorn_opt is not None:
            out["sinkhorn"] = self.sinkhorn_opt.state_dict()
            out["kind"] = out["kind"].replace("+adamw", "+sinkhorn+adamw")
        return out

    def load_state_dict(self, state: dict) -> None:
        """恢复（旧 checkpoint 的组不匹配会报错，属预期：切换优化器需新开动量）."""
        kind = state.get("kind")
        want_sink = self.sinkhorn_opt is not None
        want_head = self.head_wise_muon is not None
        has_sink = "sinkhorn" in state
        has_head = "head_wise_muon" in state
        if kind not in ("muon+adamw", "muon+headwise+adamw",
                        "muon+sinkhorn+adamw", "muon+headwise+sinkhorn+adamw"):
            raise RuntimeError(
                "该 optim.pt 不是 Muon 存盘，切换优化器请删掉 optim.pt 后重跑（权重不受影响）")
        if want_sink != has_sink or want_head != has_head:
            raise RuntimeError(
                "优化器分组与存盘不一致（head_wise/sinkhorn 开关变了）："
                "请删掉 optim.pt 后重跑（权重不受影响）")
        self.muon.load_state_dict(state["muon"])
        if has_head:
            self.head_wise_muon.load_state_dict(state["head_wise_muon"])
        if has_sink:
            self.sinkhorn_opt.load_state_dict(state["sinkhorn"])
        self.adam.load_state_dict(state["adam"])


def build_hybrid_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    head_wise: bool = False,
    n_heads: int = 1,
    sinkhorn: bool = False,
    sinkhorn_lr_scale: float = 0.18,
    sinkhorn_steps: int = 11,
) -> CombinedOptimizer:
    """构建 Muon(+head-wise)+Sinkhorn+AdamW 联合优化器.

    `head_wise`：Q/K 上投影按头分块正交化（V4.1）；`sinkhorn`：embedding /
    lm_head / Engram 表用动量 + Sinkhorn 平衡（V4.1，省 Adam 的状态）。
    """
    return CombinedOptimizer(
        model, muon_lr=muon_lr, adam_lr=adam_lr, head_wise=head_wise,
        n_heads=n_heads, sinkhorn=sinkhorn,
        sinkhorn_lr_scale=sinkhorn_lr_scale, sinkhorn_steps=sinkhorn_steps)
