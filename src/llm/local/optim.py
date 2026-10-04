"""Muon 优化器 — 隐藏层 2D 权重的正交化动量 SGD（Moonshot/Keller Jordan 路线）.

标准混合配方：2D 隐藏权重走 Muon，其余（embedding/norm/偏置/卷积）走 AdamW。
Muon 单份动量状态（Adam 两份），优化器显存减半；收敛约 2x 计算效率。

- 正交化用 Newton-Schulz 迭代（fp32 内算，防 bf16 病态），高瘦矩阵自动转置；
- 解耦权重衰减 + 动量（默认 momentum=0.95, wd=0.01, lr=0.02，按需调）；
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
        """单步更新（无闭包需求，closure 透传占位）."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # 正交化当前梯度后做重球动量（fp32 状态，防 bf16 累积误差）
                buf_fp32 = buf.float() if buf.is_floating_point() else buf
                buf_fp32.mul_(mu).add_(zeropower_via_newtonschulz5(g).float())
                state["momentum_buffer"] = buf_fp32.to(p.dtype) if p.is_floating_point() else buf_fp32
                if wd > 0:
                    p.data.mul_(1 - lr * wd)  # 解耦权重衰减
                p.data.add_(state["momentum_buffer"], alpha=-lr)
        return loss


class CombinedOptimizer:
    """Muon（2D 隐藏权重）+ AdamW（其余）的联合优化器.

    对外与单个优化器同接口：zero_grad/step/state_dict/load_state_dict，
    param_groups 汇总两边（lr 调度按组名区分：muon 组与 adam 组）。
    """

    MUON = "muon"
    ADAM = "adam"

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_wd: float = 0.01,
        adam_lr: float = 3e-4,
    ):
        muon_params: list = []
        adam_params: list = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # embedding 与 lm_head 走 AdamW（标准 Muon 配方排除项）；
            # tied 权重在 parameters() 只出现一次，按名判断即可
            if p.dim() == 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        try:
            is_cuda = next(model.parameters()).is_cuda
        except StopIteration:
            is_cuda = False
        self.muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=muon_wd)
        self.adam = torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.1,
            fused=is_cuda and torch.cuda.is_available(),
        )
        # 直接引用内部组字典（lr 调度就地写，浅拷贝会断开关联）
        self.param_groups: list = []
        for g in self.muon.param_groups:
            g["optimizer"] = self.MUON
            self.param_groups.append(g)
        for g in self.adam.param_groups:
            g["optimizer"] = self.ADAM
            self.param_groups.append(g)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零两边梯度."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """两边各走一步."""
        self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        """存盘格式（含 optimizer 标记，跨版本可辨）."""
        return {"kind": "muon+adamw", "muon": self.muon.state_dict(),
                "adam": self.adam.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """恢复（AdamW 旧 checkpoint 会 KeyError，属预期：切换优化器需新开动量）."""
        if state.get("kind") != "muon+adamw":
            raise RuntimeError("该 optim.pt 不是 Muon 存盘，切换优化器请删掉 optim.pt 后重跑（权重不受影响）")
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def build_hybrid_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
) -> CombinedOptimizer:
    """构建 Muon+AdamW 联合优化器."""
    return CombinedOptimizer(model, muon_lr=muon_lr, adam_lr=adam_lr)
