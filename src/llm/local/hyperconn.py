"""mHC 超连接残差 — Single-Pass mHC（DeepSeek-V4.1 §2.4.1）+ Mega-mHC 数据流.

完整 mHC = 流混合 `H_res`（n×n，双随机）+ 输入聚合 `A`（1×n，凸组合）+
写回 `C`（n×1 = 本实现的动态 post）。本移植三条都做：

```
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)          # 原始 mHC（eq.2）
X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)      # Single-Pass mHC（eq.6，默认）
```

**为什么移位**：eq.2 里 `A_l` 依赖 `X_l` 的**全维归约**（RMSNorm 的平方和 + 投影），
所以"算出 A_l → 再用 A_l 混合 X_l"必须读两遍 `X_l`。把输入混合错后一层（用上一层
的 `A_{l-1}`）后，混合与系数预测可以**在同一趟 tile 里**完成——论文的 Mega-mHC
kernel 因此把激活读写从 `(4n+4)d`（三 kernel）降到 `(2n+2)d`（理想下界）。
单测锁 `activation_traffic()` 的比值；PyTorch 侧用 tile 循环实现同一数据流
（真 kernel 级节省要靠自定义算子，这里只对齐**数据流与数值**）。

对称性警告：n 路输入流初始完全相同（embedding 展开），若写回也相同，
流永远分不开、`H_res` 恒为恒等白开销。破缺靠随机初始化的固定参数：
`phi_post` 各流独立 → 写回 scales 从 step 0 就不同 → 流立即分化；
`phi_res`/bias 同理。`w_post` 若用全 1 静态值则永不对称——故不用静态，
直接动态 post（参数量多 nC×n/层，可忽略）。

恒等起点：`g=sigmoid(logit)≈0`（logit=-6）→ H≈I；`A` 零初值 → 均匀均值；
`post=2σ(0)=1` → `y_s = x_s + F`。全流与基线逐位一致（单测锁定）。
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


def activation_traffic(n_streams: int, d_model: int, bytes_per: int = 2,
                       path: str = "mega") -> int:
    """每 token 的激活读写字节数（V4.1 §2.4.1 的记账口径，单位：字节）.

    - `multi`：三个 kernel 的多趟实现 `(4n+4)d`（残差更新/系数/混合各读一遍）；
    - `two_pass`：单趟语义但分两趟 `(3n+2)d`（混合要等 A 的全维归约）；
    - `mega`：Single-Pass + Mega-mHC 单趟 `(2n+2)d`（残差与混合各读写一次）。
    """
    n = int(n_streams)
    k = {"multi": 4 * n + 4, "two_pass": 3 * n + 2, "mega": 2 * n + 2}[path]
    return k * int(d_model) * int(bytes_per)


class HyperConnRes(nn.Module):
    """超连接残差：Single-Pass 输入混合 + H_res 流混合 + 动态 post 写回."""

    def __init__(
        self,
        d_model: int,
        n_streams: int = 2,
        sinkhorn_iters: int = 20,
        single_pass: bool = True,
        mega: bool = True,
        mega_tiles: int = 4,
    ):
        super().__init__()
        assert n_streams >= 2, "n=1 退化为恒等（无意义），用 0 表示关闭"
        self.n = n_streams
        self.iters = sinkhorn_iters
        # Single-Pass：输入混合用上一层给的 A_{l-1}（V4.1 eq.6）
        self.single_pass = bool(single_pass)
        # Mega-mHC：沿 hidden 维分块的单趟数据流（数值等价，少读一遍 X_l）
        self.mega = bool(mega)
        self.mega_tiles = max(1, int(mega_tiles))
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
        # 动态 pre 聚合（A_l 的预测器）：零初值即均匀均值（softmax 保守恒）
        self.phi_pre = nn.Linear(d_model * n_streams, n_streams, bias=False)
        self.bias_pre = nn.Parameter(torch.zeros(n_streams))
        self.alpha_pre = nn.Parameter(torch.tensor(0.0))
        # Amax 监控缓存（论文 Fig.3 的 instability 预警器）：track_stats 开时
        # 记录 H 在 batch/位置上的均值（n×n，detach）；默认关，compiled 解码
        # 路径不受影响（属性写入会触发 dynamo 重编，故用 flag 门控）
        self.track_stats = False
        self._last_H: torch.Tensor | None = None

    # ---------------------------------------------------------------- 系数预测
    def _coeffs(
        self, x: torch.Tensor, tiles: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一趟算出 `(A_l, S_l, C_l)`（全 fp32）：`(b,t,n,d)` → 三种系数.

        `S_l` 是投影到 Birkhoff 多面体后的**双随机矩阵**（未与恒等混合，
        混合在 `combine` 里以 delta 形式做，保证 g=0 时逐位恒等）。

        `tiles=1`：整块 GEMM（多趟写法，X 在 norm 里读一遍、混合时再读一遍）；
        `tiles>1`：Mega-mHC 的**单趟数据流**——沿 hidden 维切块，同一趟里
        ① 累加 RMSNorm 的平方和（不乘权重）② 累加三个投影（乘权重后）；
        循环结束再乘 `1/rms`。RMSNorm 的权重折进投影正是论文 §2.4.1 的写法
        （"normalization weights are folded into the projection weights offline,
        and the RMS division is applied after the projection"）。

        数学上两种走法等价（只差浮点求和顺序）：`_coeffs(tiles=4) ≈ _coeffs(1)`。
        """
        b, t, n, c = x.shape
        w_pre = self.phi_pre.weight.float()
        w_res = self.phi_res.weight.float()
        w_post = self.phi_post.weight.float()
        d_tot = n * c
        if not tiles or tiles <= 1 or c <= 1:
            # 与 RMSNorm 同一算式，但**保持 fp32**（不做 bf16 回程）：
            # 系数精度只影响混合系数（~1e-6），却让 Mega 与整块 GEMM 两条实现
            # 逐位可比（否则 bf16 取整会让两条路径的系数差 1e-3 并被层数放大）
            xf = x.reshape(b, t, d_tot).float()
            v = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.norm.eps)
                 * self.norm.weight.float())
            acc_pre = v @ w_pre.T
            acc_res = v @ w_res.T
            acc_post = v @ w_post.T
        else:
            # Mega-mHC 单趟数据流：X_l 只读一遍
            nt = max(1, min(int(tiles), c))
            step = (c + nt - 1) // nt
            w = self.norm.weight.float()
            ss = torch.zeros((b, t), dtype=torch.float32, device=x.device)
            acc_pre = torch.zeros((b, t, n), dtype=torch.float32,
                                  device=x.device)
            acc_res = torch.zeros((b, t, n * n), dtype=torch.float32,
                                  device=x.device)
            acc_post = torch.zeros((b, t, n), dtype=torch.float32,
                                   device=x.device)
            for s in range(0, c, step):
                e = min(s + step, c)
                xt = x[:, :, :, s:e].float()                 # (b,t,n,step)
                ss = ss + xt.pow(2).sum(dim=(2, 3))          # 未乘权重的平方和
                # 与 flatten(n*d) 对齐的下标：第 j 路流占 [j*d, (j+1)*d)
                idx = torch.cat([torch.arange(j * c + s, j * c + e,
                                              device=x.device)
                                 for j in range(n)])
                vt = xt.reshape(b, t, -1) * w[idx]          # x⊙w（权重折进投影）
                acc_pre = acc_pre + vt @ w_pre[:, idx].T
                acc_res = acc_res + vt @ w_res[:, idx].T
                acc_post = acc_post + vt @ w_post[:, idx].T
            # RMS 在归约后统一乘（论文：权重离线折进投影，最后再做 RMS 除法）
            rms = torch.rsqrt(ss / d_tot + self.norm.eps).unsqueeze(-1)
            acc_pre = acc_pre * rms
            acc_res = acc_res * rms
            acc_post = acc_post * rms
        pre = torch.softmax(
            acc_pre * self.alpha_pre.float() + self.bias_pre.float(), dim=-1)
        tilde = acc_res * self.alpha.float() + self.bias_res.float()
        # 返回**未混合**的双随机矩阵：混合在 combine 里写成
        # `x + g·(S x − x)`，g≈0 时与恒等**逐位**一致（直接算 `(1−g)I+gS`
        # 会引入 bf16 取整，恒等起点就不精确了）
        s_mat = sinkhorn(tilde.view(b, t, n, n), self.iters)
        post = 2 * torch.sigmoid(
            acc_post * self.alpha_post.float() + self.bias_post.float())
        return pre, s_mat, post

    # ---------------------------------------------------------------- 入口/出口
    def pre_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """A_l = softmax(φ_pre(RMSNorm(X_l)))：`(b,t,n)` 凸组合系数（fp32）."""
        return self._coeffs(x, tiles=None)[0]

    def pre_mix(self, x: torch.Tensor,
                pre: torch.Tensor | None = None) -> torch.Tensor:
        """输入混合：`A·X` → `(b,t,d)`. 给了 `pre` 就用它（Single-Pass）. """
        n = self.n
        if pre is None:
            pre = self.pre_coeffs(x)
        return torch.einsum("bts,btsc->btc", pre.to(x.dtype), x)

    def enter(
        self, x: torch.Tensor, pre_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """块入口：返回 `(块输入, (A_next, B_l, C_l))`.

        - `single_pass=True`（V4.1 eq.6）：用**上一层**的 `A_{l-1}` 混合；
          本层只负责产出 `A_l` 留给下一块。第 0 块没有上一层 → 均匀系数
          （= 零初始化 α/bias 时的 softmax，与旧 mean 聚合逐位一致）。
        - `single_pass=False`（eq.2）：本层预测 `A_l` 并立即使用（多一趟）。

        `(B_l, C_l)` 一并返回，供 `combine` 复用同一次归一化（省一趟 H 投影）。
        """
        n = self.n
        tiles = self.mega_tiles if self.mega else 1
        pre, h_res, post = self._coeffs(x, tiles=tiles)
        if self.single_pass:
            if pre_prev is None:
                pre_use = torch.full((x.shape[0], x.shape[1], n), 1.0 / n,
                                     dtype=torch.float32, device=x.device)
            else:
                pre_use = pre_prev.float()
        else:
            pre_use = pre
        x_in = torch.einsum("bts,btsc->btc", pre_use.to(x.dtype), x)
        return x_in, (pre, h_res, post)

    def combine(
        self, x: torch.Tensor, f_out: torch.Tensor,
        aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """`y_s = Σ_r B[s,r]·x_r + C_s·F`（流混合 + 写回）.

        `aux` 由 `enter` 返回（同一 `x` 上算的 `(A_next, B, C)`），复用可省一次
        H 投影；不给就现算（旧调用方式，逐位一致）。
        """
        if aux is None:
            _, s_mat, post = self._coeffs(x, tiles=None)
        else:
            _, s_mat, post = aux
        s = s_mat.to(x.dtype)
        post = post.to(x.dtype)
        g = torch.sigmoid(self.logit).to(x.dtype)
        if self.track_stats:
            with torch.no_grad():
                n = self.n
                eye = torch.eye(n, device=x.device, dtype=torch.float32
                                ).view(1, 1, n, n)
                self._last_H = ((1 - g.float()) * eye
                                + g.float() * s_mat.detach().float()
                                ).mean(dim=(0, 1))
        # H = (1−g)I + gS ⇒ Hx = x + g(Sx − x)：g=0 时**逐位**恒等
        mixed = x + g * (torch.einsum("btsr,btrc->btsc", s, x) - x)
        return mixed + post.unsqueeze(-1) * f_out.unsqueeze(2)

    # 兼容旧接口：外部（单测/工具）用 `mappings` 只看系数
    def mappings(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """算 H_res (b,t,n,n)（已按 g 混合进恒等）+ post (b,t,n) 写回系数."""
        _, s_mat, post = self._coeffs(x, tiles=None)
        n = self.n
        g = torch.sigmoid(self.logit).float()
        eye = torch.eye(n, device=x.device, dtype=torch.float32).view(1, 1, n, n)
        h_res = (1 - g) * eye + g * s_mat
        dt = x.dtype
        return h_res.to(dt), post.to(dt)

    def extra_repr(self) -> str:
        """模块摘要（打印模型结构用）."""
        return (f"n_streams={self.n}, single_pass={self.single_pass}, "
                f"mega={self.mega}(tiles={self.mega_tiles}), "
                f"sinkhorn_iters={self.iters}")


def composite_gain(mats: list[torch.Tensor]) -> float:
    """复合映射 Amax 增益（论文 Fig.3/7 指标）：各层 H 均值按层序连乘后，
    行/列和绝对值的最大值。≈1 健康，飙升=残差流爆炸预警。"""
    comp = mats[-1]
    for m in reversed(mats[:-1]):
        comp = comp @ m
    return max(comp.sum(-1).abs().max().item(),
               comp.sum(-2).abs().max().item())
