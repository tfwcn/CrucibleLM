"""Gated-DeltaNet-lite 线性注意力 — Qwen3-Next 系 Hybrid 架构的线性侧.

完整 Gated DeltaNet 需要 Triton 内核才高效；这里用可并行训练的门控线性注意力
等价形式（衰减加权的 QK^T @ V，O(N^2) 并行训练 / O(N) 常数缓存推理），
保留三处关键思想：内容相关遗忘门、QK-Norm、K/V 短卷积。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import RMSNorm


class GatedDeltaLite(nn.Module):
    """门控线性注意力层."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 linear_chunk: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.linear_chunk = linear_chunk
        # Q/K/V/门控/输出门投影
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, n_heads, bias=True)  # 每头一个遗忘门
        self.w_u = nn.Linear(d_model, d_model, bias=False)  # 输出门（SiLU）
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        # K/V 短卷积（Qwen3-Next 式局部建模，depthwise 1D）
        # 注意：必须用左补零的因果卷积！padding=1 的双边补零会让 K/V 看到
        # 未来 1 个位置，而训练目标恰是下一 token——模型会学到复制通道，
        # loss 虚假塌到 ~0（留存集同样沦陷，生成变复读机），教训见单测
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def _step_conv(
        self,
        conv: nn.Conv1d,
        proj: torch.Tensor,
        buf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单步/全序列通用因果卷积.

        proj: 本次投影后输入 (b, t, d)；buf: 上次尾部 (b, d, kernel-1) 或 None。
        返回 (卷积输出 (b, t, d), 新尾部 (b, d, kernel-1))，尾部恒取真实输入
        （不含补零），供下次解码还原左感受野。
        解码时靠 buf 还原左感受野，否则每步 k/v 与全前向对不上（Mamba 系同理）。
        """
        k = conv.kernel_size[0]
        xt = proj.transpose(1, 2)  # (b, d, t)
        if k <= 1:
            return conv(xt).transpose(1, 2), xt[:, :, :0]
        # 新尾部恒 (k-1) 长：输入不足时左补零（序列开头本就无历史）；
        # 恒定长度保证下次拼接后必满足 kernel 长度
        tail = xt[:, :, -(k - 1):]
        if tail.shape[-1] < k - 1:
            tail = F.pad(tail, (k - 1 - tail.shape[-1], 0))
        new_buf = tail.detach() if torch.is_grad_enabled() else tail
        if buf is not None:
            xt = torch.cat([buf.to(xt.device, xt.dtype), xt], dim=-1)
        else:
            xt = F.pad(xt, (k - 1, 0))
        return conv(xt).transpose(1, 2), new_buf

    def _project(
        self, h: torch.Tensor, bufs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               tuple[torch.Tensor, torch.Tensor] | None]:
        """投影 + 短卷积 + QK-Norm，返回 (q, k, v, gate, out_gate, 新卷积尾部).

        bufs: 解码缓存的 (k_tail, v_tail)，None 表示从头（左补零）。
        """
        b, t, _ = h.shape
        # 短卷积需要 (b, d, t) 排布；左补 (kernel-1) 保证严格因果（输出 i 只见输入 <=i）
        kb, vb = (bufs if bufs is not None else (None, None))
        k, new_kb = self._step_conv(self.k_conv, self.w_k(h), kb)
        v, new_vb = self._step_conv(self.v_conv, self.w_v(h), vb)
        q = self.w_q(h).view(b, t, self.n_heads, self.head_dim)
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        gate = self.w_g(h)  # (b, t, heads)，内容相关遗忘门
        u = self.w_u(h)
        return q, k, v, gate, u, (new_kb, new_vb)

    def forward(
        self,
        h: torch.Tensor,
        past: tuple | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, tuple | None]:
        """并行前向（训练/Prefill）或带状态解码.

        past: 三元缓存 (S, k_buf, v_buf)——循环状态 + 卷积左文（Mamba 系同理），
        推理时 O(1) 更新；k/v 解码步靠 buf 还原感受野，与全前向逐位一致。
        return_state=False 跳过解码状态循环（训练用不上，省约 t 步 Python 开销）。
        返回 (输出, 新缓存或 None)。
        """
        b, t, _ = h.shape
        if past is not None:
            s_prev, kb, vb = past
            q, k, v, gate, u, bufs = self._project(h, (kb, vb))
            new_bufs = bufs
        else:
            q, k, v, gate, u, new_bufs = self._project(h)
            s_prev = None
        decay = torch.sigmoid(gate)  # (b, t, heads)，接近 1=记住，接近 0=遗忘
        if past is not None and t == 1:
            # 增量解码：S_t = decay * S_{t-1} + k^T v（外积更新）
            kv = k[:, 0].unsqueeze(-1) @ v[:, 0].unsqueeze(-2)  # (b, h, d, d)
            new_state = s_prev * decay[:, 0].view(b, self.n_heads, 1, 1) + kv
            # o = scale·q^T·S（与并行分支 scores 里的 scale 对齐，漏了则训推不一致）
            o = (q[:, 0].unsqueeze(-2) @ new_state).squeeze(-2) * self.scale
            o = (o.reshape(b, t, -1) * F.silu(u)).contiguous()
            return self.w_o(o), (new_state, new_bufs[0], new_bufs[1])
        if t > self.linear_chunk:
            # 长序列分块递推（前向精确，块间截断 BPTT，见 _forward_chunked）；
            # 缓存拼三元组（S + 卷积尾），与短序列分支一致
            out, s_lin = self._forward_chunked(q, k, v, decay, u)
            return out, (s_lin, new_bufs[0], new_bufs[1])
        # 并行训练形式：衰减累积矩阵 D[i,j] = prod_{s=j+1..i} decay_s（j<=i）
        # 注意：必须先填 -inf 再 exp（若先 exp 再 mask，上三角 exp(+350)=inf，
        # 前向虽被 mask 掩盖，反向 0×inf 会产生 NaN 污染 w_g 梯度）
        log_decay = torch.log(decay.clamp_min(1e-6)).transpose(1, 2)  # (b, h, t)
        cum = log_decay.cumsum(-1)
        causal = torch.tril(torch.ones(t, t, device=h.device, dtype=torch.bool))
        d_mat = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(
            ~causal, float("-inf")).exp()  # (b, h, t, t)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        scores = scores * d_mat
        o = torch.einsum("bhts,bshd->bthd", scores, v)
        o = (o.reshape(b, t, -1) * F.silu(u)).contiguous()
        if self.training and self.dropout > 0:
            o = F.dropout(o, p=self.dropout)
        # 用显式循环累积解码状态（验证序列短时可接受；长序列训练走上式并行）
        # 状态不参与梯度（训练用不上，解码在 no_grad 下），避免展开 t 步大计算图；
        # return_state=False 时整个循环跳过（训练每步省约 t×layers 个 Python 小算子）
        if not return_state:
            return self.w_o(o), None
        with torch.no_grad():
            # 防御：past 非空但 t>1（正常只发生在 prefill，past 应为 None），
            # 此时从零重算状态（旧语义），不信任传入缓存
            s = torch.zeros(
                b, self.n_heads, self.head_dim, self.head_dim,
                device=h.device, dtype=h.dtype,
            )
            for i in range(t):
                kv_i = k[:, i].detach().unsqueeze(-1) @ v[:, i].detach().unsqueeze(-2)
                s = s * decay[:, i].detach().view(b, self.n_heads, 1, 1) + kv_i
        return self.w_o(o), (s, new_bufs[0], new_bufs[1])

    def _forward_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """分块递推前向：块内并行 + 块间状态传递（前向数学精确）.

        反向在块边界截断（Transformer-XL 式，超长训练的标准做法）：
        状态 S 每块 detach，梯度只在块内流动，内存恒为 O(chunk^2)；
        短序列走并行分支时梯度是全序列精确的。
        """
        b, t, h, d = q.shape
        chunk = self.linear_chunk
        outs: list[torch.Tensor] = []
        state = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
        for s in range(0, t, chunk):
            e = min(s + chunk, t)
            state = state.detach()  # 截断点：前向精确，反向块内
            qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
            dc = decay[:, s:e]  # (b, C, h)
            logd = torch.log(dc.clamp_min(1e-6)).transpose(1, 2)  # (b, h, C)
            cum = logd.cumsum(-1)
            c_len = e - s
            diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)
            causal = torch.tril(torch.ones(c_len, c_len, device=q.device, dtype=torch.bool))
            dmat = diff.masked_fill(~causal, float("-inf")).exp()  # (b, h, C, C)
            # heads-first 排布与 dmat 对齐
            qh, kh, vh = (x.transpose(1, 2) for x in (qc, kc, vc))  # (b, h, C, d)
            scores = torch.einsum("bhcd,bhsd->bhcs", qh, kh) * self.scale
            o_intra = torch.einsum("bhcs,bhsd->bhcd", scores * dmat, vh)
            # 跨块项：q_i^T (Dcum_i · S_in) / sqrt(d)——scale 别漏，
            # 并行分支的 scale 在 scores 里，这里必须显式补上（曾漏过，教训）
            cross = (torch.einsum("bhcd,bhde->bhce", qh, state)
                     * cum.exp().unsqueeze(-1) * self.scale)
            outs.append((o_intra + cross).transpose(1, 2))  # 回 (b, C, h, d)
            # S_out = Dtot·S_in + Σ_j w_j k_j^T v_j
            w = (cum[..., -1:] - cum).exp().view(b, h, c_len, 1, 1)
            kv = kh.unsqueeze(-1) * vh.unsqueeze(-2)  # (b, h, C, d, d)
            state = cum[..., -1:].exp().view(b, h, 1, 1) * state + (w * kv).sum(2)
        o = torch.cat(outs, dim=1)
        o = (o.reshape(b, t, -1) * F.silu(u)).contiguous()
        if self.training and self.dropout > 0:
            o = F.dropout(o, p=self.dropout)
        return self.w_o(o), state
