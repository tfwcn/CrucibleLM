"""MLA 全注意力 — DeepSeek 系低秩 KV 压缩 + 解耦 RoPE + QK-Norm.

标准 MHA 的 KV 缓存为 2 * n_heads * head_dim；
MLA 只缓存 latent 向量 c_kv（kv_lora_rank 维）+ 共享 RoPE 键，
压缩比约为 (2*768)/(128+32) ≈ 9.6x，这是长上下文省显存的关键。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.rope import apply_rope, build_rope_cache


class RMSNorm(nn.Module):
    """RMSNorm（无偏置，DeepSeek/Qwen 通用选择）."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化并缩放（内部 fp32 求方差，保持输入精度，混合精度安全）."""
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (xf * self.weight.float()).to(dtype)


class MLAModule(nn.Module):
    """MLA 注意力层（含训练前向 + 增量解码缓存）."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_dim: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        yarn_scale: float = 1.0,
        dropout: float = 0.0,
        o_groups: int = 1,
        o_lora_rank: int = 0,
        ced: bool = False,
        kv_fmt: str = "",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_rope_dim = qk_rope_dim
        self.kv_lora_rank = kv_lora_rank
        # Q：下投影 -> 上投影（内容部分）
        self.w_dq = nn.Linear(d_model, q_lora_rank, bias=False)
        self.w_uq = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
        # Q 解耦 RoPE 部分（每头独立旋转）
        self.w_qr = nn.Linear(q_lora_rank, n_heads * qk_rope_dim, bias=False)
        # KV 联合下投影 -> 每头上投影
        # CED（V4.1 §2.2）：解码器层的全局 KV 由**编码器输出** `H_enc` 投影（逐层
        # 投影权重），不再看自己的 hidden；因此这层的 `w_dkv`/`w_kr` 换成
        # `w_ced_kv`/`w_ced_kr`（键集不同 → 改键、opt-in）。Q 与 SWA 仍来自自己。
        self.ced = bool(ced)
        # KV 缓存量化格式（V4.1 §2.4.4）：""=关；"fp4"=MXFP4（E2M1 + 每 16 通道
        # E4M3 scale）；"fp8"=E4M3。训练期 fake-quant（STE）做 QAT、推理期同一数值，
        # 差别只在**存储**（QuantBuffer 打包，反量化后再算注意力）。
        from src.llm.local.config import _norm_kv_quant

        self.kv_fmt = _norm_kv_quant(kv_fmt)
        if self.kv_fmt:
            from src.llm.local.kv_quant import KV_QUANT_BLOCK

            for _name, _dim in (("kv_lora_rank", kv_lora_rank),
                                ("qk_rope_dim", qk_rope_dim)):
                if _dim % KV_QUANT_BLOCK:
                    raise ValueError(
                        f"KV 量化要求 {_name}={_dim} 能被块 {KV_QUANT_BLOCK} 整除")
        if not self.ced:
            self.w_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.w_uk = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        # 共享 RoPE 键（全头共用一份，省缓存）
        if self.ced:
            # 占位 None（不进 state_dict）：借用层链接/迁移工具可据此判断键集
            self.w_dkv = None
            self.w_ced_kv = nn.Linear(d_model, kv_lora_rank, bias=False)
            self.w_ced_kr = nn.Linear(d_model, qk_rope_dim, bias=False)
            self.w_kr = None
        else:
            self.w_kr = nn.Linear(d_model, qk_rope_dim, bias=False)
        # 输出投影 + QK-Norm（Qwen3 系做法，稳定训练）
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        # 输出投影：默认单个稠密 w_o（键名 w_o.weight，与旧 checkpoint 兼容）。
        # o_groups>1 且 o_lora_rank>0 时改为**分组低秩**（DeepSeek-V4.1：o_groups=8 +
        # 每组中间维 o_lora_rank）：把 n_heads 分成 o_groups 组，每组头输出先降维到
        # o_lora_rank，拼接后再升维回 d_model。省参数、多一组非线性前没有（纯线性重组），
        # 但改变了 state_dict 键（旧 checkpoint 需迁移或重训，见 group_output_projection）。
        self.o_groups = int(o_groups)
        self.o_lora_rank = int(o_lora_rank)
        self.grouped_o = self.o_groups > 1 and self.o_lora_rank > 0
        if self.grouped_o:
            assert n_heads % self.o_groups == 0, \
                f"n_heads({n_heads}) 须能被 o_groups({self.o_groups}) 整除"
            self.heads_per_group = n_heads // self.o_groups
            in_per_group = self.heads_per_group * self.head_dim
            assert 0 < self.o_lora_rank <= in_per_group, (
                f"o_lora_rank({self.o_lora_rank}) 须在 (0, 每组输入维 {in_per_group}] 内")
            self.w_o_d = nn.ModuleList([
                nn.Linear(in_per_group, self.o_lora_rank, bias=False)
                for _ in range(self.o_groups)
            ])
            self.w_o_u = nn.Linear(self.o_groups * self.o_lora_rank, d_model,
                                   bias=False)
            # 占位：分组时不注册 w_o（键集不同，旧 checkpoint 严格载入会报缺键）
            self.w_o = None
        else:
            self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout
        # CSA2 跨层复用（DeepSeek-V4.1 三模式，由 build_block 按 config.csa_modes /
        # csa_reuse_every 设，均纯超参、不进 state_dict）：
        #   "full"    —— 自算 main KV（下投影 latent + rope）+ 自写缓存 + 自选 Top-K；
        #   "reindex" —— 复用最近生产者的 main KV（借 latent），但用**自己的**
        #                indexer Q 重新选 Top-K；
        #   "reuse"   —— 复用 main KV 且复用最近的 Top-K，不算 indexer。
        # `csa_reuse` 为"借用者"布尔（reindex/reuse 均为 True），供解码判断
        # 是否写自己的缓冲（只有 full 写）。
        self.csa_mode = "full"
        self.csa_reuse = False
        # 预计算 RoPE 缓存（解耦部分维度小，可一次建好）
        cos, sin = build_rope_cache(max_seq_len, qk_rope_dim, rope_theta, yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def out_proj_weight(self) -> torch.Tensor:
        """残差输出投影的权重（按深度缩放初始化 / 迁移置零用）.

        分组低秩时返回升维投影 `w_o_u`（降维 `w_o_d` 是内部瓶颈，置零它没意义）；
        否则返回稠密 `w_o`。
        """
        return self.w_o_u.weight if self.grouped_o else self.w_o.weight

    def _out_proj(self, out: torch.Tensor) -> torch.Tensor:
        """输出投影：(b, t, n_heads*head_dim) → (b, t, d_model).

        分组低秩：按头分组各自降维到 o_lora_rank，拼接后一次升维（等价于每组各有一个
        o_lora_rank→d_model 升维再求和，只是把求和并进了矩阵乘法）。关闭时退化为稠密
        `w_o`，与旧实现逐位一致。
        """
        if not self.grouped_o:
            return self.w_o(out)
        b, t, _ = out.shape
        in_per_group = self.heads_per_group * self.head_dim
        grouped = out.view(b, t, self.o_groups, in_per_group)
        latents = torch.stack(
            [self.w_o_d[i](grouped[:, :, i]) for i in range(self.o_groups)],
            dim=2,
        )  # (b, t, o_groups, o_lora_rank)
        return self.w_o_u(latents.reshape(b, t, self.o_groups * self.o_lora_rank))

    def _pack_cache(
        self, c_kv: torch.Tensor, k_rope_raw: torch.Tensor, reserve: int,
    ) -> tuple:
        """装箱解码缓存：`self.kv_fmt` 非空时返回 QuantBuffer（真省存储），
        否则返回 fp 张量（逐位兼容旧行为）。"""
        if self.kv_fmt:
            assert reserve >= c_kv.shape[1], "缓存预留不足序列"
            return (self._quant_cache(c_kv, reserve),
                    self._quant_cache(k_rope_raw, reserve),
                    torch.tensor(c_kv.shape[1], dtype=torch.long,
                                 device=c_kv.device))
        return self._pack_cache_fp(c_kv, k_rope_raw, reserve)

    def _quant_cache(self, x: torch.Tensor, reserve: int):
        """把 (b, t, r) 的缓存写成量化缓冲（尾部补零到 reserve，逻辑长仍是 reserve）."""
        from src.llm.local.kv_quant import QuantBuffer

        b, t, r = x.shape
        tail = max(0, reserve - t)
        if tail:
            x = torch.cat([x, torch.zeros(b, tail, r, device=x.device,
                                          dtype=x.dtype)], dim=1)
        return QuantBuffer.from_tensor(x, self.kv_fmt)

    def _pack_cache_fp(
        self, c_kv: torch.Tensor, k_rope_raw: torch.Tensor, reserve: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """静态缓存装箱：latent 尾部补零到 reserve 长，并附位置计数.

        返回 (buf_c, buf_kr, pos)，pos 为 0 维 long 张量（值动态、形状恒定，
        dynamo 不会按值特化）。解码按位写、掩码读，不再 `cat` 增长。
        """
        b, t, _ = c_kv.shape
        assert reserve >= t, f"缓存预留 {reserve} 不足序列 {t}"
        if reserve == t:
            buf_c, buf_kr = c_kv, k_rope_raw
        else:
            pad = torch.zeros(b, reserve - t, c_kv.shape[-1],
                              device=c_kv.device, dtype=c_kv.dtype)
            buf_c = torch.cat([c_kv, pad], dim=1)
            pad_r = torch.zeros(b, reserve - t, k_rope_raw.shape[-1],
                                device=c_kv.device, dtype=k_rope_raw.dtype)
            buf_kr = torch.cat([k_rope_raw, pad_r], dim=1)
        pos = torch.tensor(t, device=c_kv.device, dtype=torch.long)
        return buf_c, buf_kr, pos

    def _valid_mask(self, length: int, pos: torch.Tensor, t: int,
                    device: torch.device) -> torch.Tensor:
        """有效位掩码 (1, 1, t, L)：已写位置放行 + 预留空位屏蔽 + 块内因果.

        t==1（逐 token 解码）时退化为旧式 (1,1,1,L)，逐位不变；
        t>1（投机解码并行验证）时必须补块内因果掩码：新查询只许看到
        自身及之前，否则会偷看同批未来草稿（未开因果的经典 bug）。
        """
        keys = torch.arange(length, device=device).view(1, 1, 1, length)
        mask = keys < (pos + t).view(1, 1, 1, 1)
        if t > 1:
            q_pos = pos.reshape(1) + torch.arange(t, device=device)
            mask = mask & (keys <= q_pos.view(1, 1, t, 1))
        return mask

    def _kv_latent(self, h: torch.Tensor,
                   ced_src: torch.Tensor | None = None) -> torch.Tensor:
        """主 KV 的下投影 latent：CED 解码器层来自编码器输出，其余来自本层 hidden.

        `kv_fmt` 非空时在这里做 QAT（fake-quant + STE）：训练与推理都按"会被量化成
        的值"计算，区别只在存储（`QuantBuffer` 打包）。放这里是因为它是**唯一**的
        latent 出口（prefill / 解码 / CED 全走它），避免漏掉某条路径导致训推不一致。
        """
        if not self.ced:
            return self.w_dkv(h)
        if ced_src is None:
            raise ValueError("CED 解码器层必须传 ced_src（编码器输出 H_enc）")
        return self.w_ced_kv(ced_src)

    def _rope_key(self, h: torch.Tensor,
                  ced_src: torch.Tensor | None = None) -> torch.Tensor:
        """共享 rope 键（未旋转）：CED 解码器层同样来自编码器输出.

        必须与 `_kv_latent` 同源：否则"跳算前缀"时还需要在前缀上跑解码器层
        才能算出 rope 键，预填充就省不下来了。
        """
        if not self.ced:
            return self.w_kr(h)
        if ced_src is None:
            raise ValueError("CED 解码器层必须传 ced_src（编码器输出 H_enc）")
        return self.w_ced_kr(ced_src)

    def _qkv(
        self, h: torch.Tensor, ced_src: torch.Tensor | None = None,
        q_off: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 hidden 计算各头 q/k/v、旋转 q_rope/k_rope，并附带 latent 原料.

        后两项是未旋转的 c_kv 与共享 rope 键，供解码缓存复用（省两次投影）。
        `ced_src`：CED 解码器层的编码器输出（KV 与 rope 键由它投影）。
        `q_off`：查询首 token 的绝对位置（CED 跳算预填充时只喂尾部，位置从
        `q_off` 起；0 = 从头开始，逐位等价旧行为）。
        """
        b, t, _ = h.shape
        # Q 内容流
        q_c = self.w_dq(h)
        q = self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        # Q 旋转流（相位按绝对位置取：跳算时查询不从 0 起）
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(
            q_rope, self.rope_cos[q_off:], self.rope_sin[q_off:])
        # KV 压缩流（训练时展开；解码时只缓存 c_kv）。
        # CED 跳算时 `h` 只覆盖查询尾段、而 KV 覆盖**全序列**（来自 `ced_src`），
        # 故 KV 侧的长度要单独取；rope 相位也从 0 起（KV 的绝对位置从 0 起）。
        c_kv = self._kv_latent(h, ced_src)
        t_kv = c_kv.shape[1]
        k = self.w_uk(c_kv).view(b, t_kv, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.w_uv(c_kv).view(b, t_kv, self.n_heads, self.head_dim)
        # 共享旋转键（先旋转再补头维度，避免广播错位）
        k_rope_raw = self._rope_key(h, ced_src)
        # QAT（V4.1 §2.4.4）：训练与"不走缓存的前向"都按**会被量化成的值**计算，
        # 这样它和缓存里存的值一致（缓存写入处对同一个张量做同一量化）。
        if self.kv_fmt:
            from src.llm.local.kv_quant import fake_quant

            c_kv = fake_quant(c_kv, self.kv_fmt)
            k = self.w_uk(c_kv).view(b, t_kv, self.n_heads, self.head_dim)
            k = self.k_norm(k)
            v = self.w_uv(c_kv).view(b, t_kv, self.n_heads, self.head_dim)
            k_rope_raw = fake_quant(k_rope_raw, self.kv_fmt)
        k_rope = apply_rope(
            k_rope_raw, self.rope_cos, self.rope_sin
        ).unsqueeze(2)
        return q, k, v, q_rope, k_rope, c_kv, k_rope_raw

    def _q_only(self, h: torch.Tensor, q_off: int = 0) -> torch.Tensor:
        """只算 Q（含 RoPE），供跨层复用层（K/V 借共享 latent）.

        `q_off` 为查询首 token 的绝对位置（CED 跳算时非 0）。
        """
        b, t, _ = h.shape
        q_c = self.w_dq(h)
        q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(
            q_rope, self.rope_cos[q_off:], self.rope_sin[q_off:])
        return torch.cat([q, q_rope], dim=-1).transpose(1, 2)

    def _kv_from_latent(
        self, c_kv: torch.Tensor, k_rope_raw: torch.Tensor, total: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由共享 latent 展开全头 K/V（跨层复用层只借 latent，不借上投影）.

        `total` 非空时只取前 `total` 个位置（prefill 缓冲可能补零到 reserve）。
        """
        from src.llm.local.kv_quant import QuantBuffer, fake_quant

        if isinstance(c_kv, QuantBuffer):
            c_kv = c_kv.dequant()
        if isinstance(k_rope_raw, QuantBuffer):
            k_rope_raw = k_rope_raw.dequant()
        if total is not None:
            c_kv = c_kv[:, :total]
            k_rope_raw = k_rope_raw[:, :total]
        b, T, _ = c_kv.shape
        k = self.k_norm(self.w_uk(c_kv).view(b, T, self.n_heads, self.head_dim))
        v = self.w_uv(c_kv).view(b, T, self.n_heads, self.head_dim)
        k_rope = apply_rope(
            k_rope_raw, self.rope_cos[:T], self.rope_sin[:T],
        ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        k_full = torch.cat([k, k_rope], dim=-1).transpose(1, 2)
        return k_full, v.transpose(1, 2)

    def _forward_reuse(
        self, h: torch.Tensor,
        reuse: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        return_state: bool,
    ) -> tuple[torch.Tensor, tuple | None]:
        """跨层复用（prefill/训练）：Q 自算，K/V 借共享 latent，因果掩码同 prefill.

        CED 解码器层**恒为 Full**（`build_block` 保证）：它的 KV 是
        `H_enc @ W_l`，逐层权重不同，"借用"会把 layer-dependent 投影合并回一份。
        """
        if self.ced:
            raise RuntimeError(
                "CED 解码器层不支持 CSA2 借用（全局 KV 由编码器输出逐层投影）")
        b, t, _ = h.shape
        buf_c, buf_kr, pos = reuse
        total = buf_c.shape[1]
        q_full = self._q_only(h)
        k_full, v_full = self._kv_from_latent(buf_c, buf_kr)
        # prefill 语义：查询即 prompt 全段（pos 从 0 起），掩码屏蔽补零空位 + 块内因果
        zero = torch.zeros((), dtype=torch.long, device=h.device)
        out = F.scaled_dot_product_attention(
            q_full, k_full, v_full,
            attn_mask=self._valid_mask(total, zero, t, h.device),
            dropout_p=self.dropout,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self._out_proj(out), (reuse if return_state else None)

    def _decode_attend(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        write: bool = True,
        ced_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """从静态缓冲读 K/V 做掩码注意力；`write=False` 用于跨层复用层（不重复写）.

        `ced_src`：CED 解码器层的编码器输出（当步），KV/rope 键由它投影后写入缓存。
        """
        from src.llm.local.kv_quant import QuantBuffer, fake_quant

        b, t, _ = h.shape
        buf_c, buf_kr, pos = past
        total = buf_c.shape[1]
        if write:
            p0 = int(pos.item())
            if isinstance(buf_c, QuantBuffer):
                # 量化缓存：写到哪就把哪一段按格式量化（QAT 数值与训练一致）
                buf_c.write_(p0, self._kv_latent(h, ced_src))
                buf_kr.write_(p0, self._rope_key(h, ced_src))
            else:
                # 范围写（t==1 即单槽；dynamo 安全，形状恒定）
                w_idx = pos.reshape(1) + torch.arange(t, device=h.device)
                # 对齐缓冲 dtype（prefill 可能建于 autocast 下；同 dtype 时是 no-op）
                buf_c.index_copy_(1, w_idx,
                                  self._kv_latent(h, ced_src).to(buf_c.dtype))
                buf_kr.index_copy_(1, w_idx,
                                   self._rope_key(h, ced_src).to(buf_kr.dtype))
        # 读之前先反量化（V4.1：反量化后再算注意力，不需要 FP4 矩阵乘）；
        # 注意别覆盖 `buf_c/buf_kr`——返回的 past 必须是**原始（量化）缓冲**
        if isinstance(buf_c, QuantBuffer):
            c_use, kr_use = buf_c.dequant(), buf_kr.dequant()
        else:
            c_use, kr_use = buf_c, buf_kr
        pos_next = pos + t
        q_c = self.w_dq(h)
        q = self.q_norm(self.w_uq(q_c).view(b, t, self.n_heads, self.head_dim))
        q_rope = self.w_qr(q_c).view(b, t, self.n_heads, self.qk_rope_dim)
        # 旋转相位按绝对位置 gather（dynamo 安全，不按值特化）；
        # 下标钳制在 rope 缓存内（有效位本就 ≤ rope_len，旧约束不变）
        rlen = self.rope_cos.shape[0]
        q_idx = (pos.reshape(1) + torch.arange(t, device=h.device)).clamp(
            max=rlen - 1)
        q_rope = apply_rope(
            q_rope,
            self.rope_cos[q_idx],
            self.rope_sin[q_idx],
        )
        k = self.k_norm(self.w_uk(c_use).view(b, total, self.n_heads,
                                              self.head_dim))
        v = self.w_uv(c_use).view(b, total, self.n_heads, self.head_dim)
        rlen = self.rope_cos.shape[0]
        ridx = torch.arange(total, device=h.device).clamp(max=rlen - 1)
        k_rope_hist = apply_rope(
            kr_use,
            self.rope_cos[ridx],
            self.rope_sin[ridx],
        ).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_hist], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        # 解码时 query 只有 t 步，对全缓冲做掩码注意力即可
        # （SDPA bool 掩码 True=放行；预留空位 False 屏蔽）
        out = F.scaled_dot_product_attention(
            q_full, k_full, v_full,
            attn_mask=self._valid_mask(total, pos, t, h.device),
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self._out_proj(out), (buf_c, buf_kr, pos_next)

    def forward(
        self,
        h: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = True,
        reserve: int = 0,
        reuse: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        ced_src: torch.Tensor | None = None,
        q_off: int = 0,
    ) -> tuple[torch.Tensor, tuple | None]:
        """训练/Prefill 前向：全序列并行注意力.

        past: 静态解码缓存 (buf_c, buf_kr, pos)，pos 为已缓存长度（0 维张量）；
        传入即增量解码（按位写、掩码读，不再 `cat` 增长）。
        reserve: prefill 预留总长（prompt + 后续解码，generate 由 max_new_tokens
        精确给出；0 表示恰好当前长度，解码前由调用方扩到够用）。
        reuse: 跨层复用（CSA2）：`(buf_c, buf_kr, pos)` 为最近 Full 层的 latent 缓存，
        本层借它出 K/V（只自算 Q）；None=正常自算。仅 prefill/训练（past=None）用。
        return_state=False 时不返回解码缓存（训练用不上，省两次投影）。
        返回 (输出, 新缓存或 None)。
        """
        b, t, _ = h.shape
        if past is not None:
            # 增量解码；Reuse 层跳过写入、只读共享缓冲（由最近的 Full 层写好）
            return self._decode_attend(h, past, write=not self.csa_reuse,
                                       ced_src=ced_src)
        if reuse is not None:
            return self._forward_reuse(h, reuse, return_state)
        # 训练/Prefill：全序列并行
        q, k, v, q_rope, k_rope, c_kv, k_rope_raw = self._qkv(
            h, ced_src=ced_src, q_off=q_off)
        k_rope_full = k_rope.expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k, k_rope_full], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_full, k_full, v_full, is_causal=True, dropout_p=self.dropout,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        if not return_state:
            return self._out_proj(out), None
        # 静态缓存装箱（reserve=0 即恰好当前长度，解码前由调用方扩到够用）
        return self._out_proj(out), self._pack_cache(c_kv, k_rope_raw, max(reserve, t))

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """每 token KV 缓存字节数（对比 MHA 说明压缩收益）."""
        return (self.kv_lora_rank + self.qk_rope_dim) * dtype_bytes
