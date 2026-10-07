"""mHC-lite 超连接残差 — DeepSeek mHC 思想的最小可用移植.

完整 mHC = H_res（流混合）+ H_pre（聚合）+ H_post（写回）；
本移植只做 H_res（八成收益，论文消融 -0.022/-0.027），pre 用均值聚合，
post 用动态逐流写回（对称性破缺的关键，见下）。

对称性警告：n 路输入流初始完全相同（embedding 展开），若写回也相同，
流永远分不开、H_res 恒为恒等白开销。破缺靠随机初始化的固定参数：
φ_post 各流独立 → 写回 scales 从 step 0 就不同 → 流立即分化；
φ_res/bias 同理。w_post 若用全 1 静态值则永不对称——故不用静态，
直接动态 post（参数量多 nC×n/层，可忽略）。

恒等起点：g=sigmoid(logit)≈0（logit=-6）→ H≈I；
post=2σ(0)=1 → y_s = x_s + F。全流与基线逐位一致（单测锁定）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.llm.local.mla import RMSNorm


def sinkhorn(tilde: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """熵投影到 Birkhoff 多面体（双随机矩阵）：exp 后交替行列归一.

    tilde: (..., n, n)；n≤4 小矩阵，朴素 autograd 直通即可
    （论文用定制反向 + TileLang，我们用不上）。
    """
    m = torch.exp(tilde)
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
        m = m / m.sum(-2, keepdim=True).clamp_min(1e-12)
    return m


class HyperConnRes(nn.Module):
    """超连接残差混合：H_res 流混合 + 动态 post 写回."""

    def __init__(
        self,
        d_model: int,
        n_streams: int = 2,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        assert n_streams >= 2, "n=1 退化为恒等（无意义），用 0 表示关闭"
        self.n = n_streams
        self.iters = sinkhorn_iters
        self.norm = RMSNorm(d_model * n_streams)
        self.phi_res = nn.Linear(d_model * n_streams, n_streams * n_streams,
                                 bias=False)
        self.bias_res = nn.Parameter(torch.zeros(n_streams * n_streams))
        # α=0 精确恒等起点（动态项开局全零；梯度非零，零速启动，复壮同款动力学）
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.logit = nn.Parameter(torch.tensor(-6.0))  # g≈0.0025，近恒等起点
        self.phi_post = nn.Linear(d_model * n_streams, n_streams, bias=False)
        self.bias_post = nn.Parameter(torch.zeros(n_streams))
        self.alpha_post = nn.Parameter(torch.tensor(0.0))

    def mappings(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """算 H_res (b,t,n,n) 双随机 + post (b,t,n) 写回系数（小矩阵 fp32 求稳）."""
        b, t, n, c = x.shape
        v = self.norm(x.reshape(b, t, n * c)).float()
        w_res = self.phi_res.weight.float()
        # 注意：alpha 保持张量运算（float(0维张量) 会 detach 断梯度）
        tilde = (v @ w_res.T) * self.alpha.float() + self.bias_res.float()
        h_res = sinkhorn(tilde.view(b, t, n, n), self.iters)
        g = torch.sigmoid(self.logit).float()
        h = (1 - g) * torch.eye(n, device=x.device, dtype=torch.float32
                                ).view(1, 1, n, n) + g * h_res
        w_post = self.phi_post.weight.float()
        post = 2 * torch.sigmoid(
            (v @ w_post.T) * self.alpha_post.float() + self.bias_post.float())
        dt = x.dtype
        return h.to(dt), post.to(dt)

    def combine(self, x: torch.Tensor, f_out: torch.Tensor) -> torch.Tensor:
        """y_s = Σ_r H[s,r]·x_r + post_s·F（流混合 + 写回）."""
        h, post = self.mappings(x)
        mixed = torch.einsum("btsr,btrc->btsc", h, x)
        return mixed + post.unsqueeze(-1) * f_out.unsqueeze(2)

    def extra_repr(self) -> str:
        """模块摘要（打印模型结构用）."""
        return f"n_streams={self.n}"
