"""完整迷你模型 — embedding + Hybrid 层叠 + MTP 头 + 生成循环."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.block import HybridBlock
from src.llm.local.config import SmallLLMConfig
from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.mla import RMSNorm
from src.llm.local.sparse_attn import CsaChannel


def build_block(config: SmallLLMConfig, layer_idx: int) -> HybridBlock:
    """按配置构建单个 Hybrid 块（模型与迁移工具共用，避免参数漂移）."""
    use_memory = (config.memory_every > 0
                  and layer_idx % config.memory_every == 0)
    use_retro = (config.retro_enabled and config.retro_every > 0
                 and layer_idx % config.retro_every == 0)
    use_hyper = config.hyper_streams >= 2
    # Engram 层：`engram_layers` 显式优先（论文只放第 1、14 层），否则按 every 均匀插
    engram_ids = config.engram_layer_ids()
    use_engram = bool(engram_ids) and layer_idx in engram_ids
    full = config.is_full_attn_layer(layer_idx)
    # CED（V4.1 §2.2）：解码器层的全局 KV 由编码器输出逐层投影，**恒为 Full**
    # （layer-dependent 投影本就逐层不同，借用会把它们合并回一份缓存，失去意义）
    is_ced_dec = config.is_ced_decoder(layer_idx)
    # CSA2 三模式静态分配（V4.1）：`csa_modes` 显式指定，空则由 csa_reuse_every
    # 推导旧行为（Full/Reuse 交替）。full=自算自存；reindex/reuse=借生产者 main KV。
    csa_mode = "full" if is_ced_dec else config.csa_mode_of(layer_idx)
    blk = HybridBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        full_attn=full,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_dim=config.qk_rope_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        yarn_scale=config.yarn_scale,
        n_experts=config.n_experts,
        top_k=config.top_k,
        expert_hidden=config.expert_hidden,
        n_shared=config.n_shared,
        aux_coef=config.aux_loss_coef,
        swiglu_limit=config.swiglu_limit,
        router_aux_free=config.router_aux_free,
        router_bias_lr=config.router_bias_lr,
        routed_scaling_factor=config.routed_scaling_factor,
        norm_topk_prob=config.norm_topk_prob,
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        dropout=config.dropout,
        sparse_threshold=config.sparse_threshold,
        sparse_window=config.sparse_window,
        sparse_sink=config.sparse_sink,
        sparse_stride=config.sparse_stride,
        sparse_chunk=config.sparse_chunk,
        sparse_indexer=config.sparse_indexer,
        indexer_heads=config.indexer_heads,
        indexer_head_dim=config.indexer_head_dim,
        indexer_topk=config.indexer_topk,
        indexer_block=config.indexer_block,
        indexer_pool_blocks=config.indexer_pool_blocks,
        indexer_loss_coef=config.indexer_loss_coef,
        linear_chunk=config.linear_chunk,
        use_memory=use_memory,
        memory_slots=config.memory_slots,
        memory_topk=config.memory_topk,
        use_retro=use_retro,
        retro_heads=config.retro_heads,
        hyper_streams=config.hyper_streams if use_hyper else 0,
        hc_single_pass=bool(getattr(config, "hc_single_pass", False)),
        hc_mega=bool(getattr(config, "hc_mega", False)),
        hc_sinkhorn_iters=int(getattr(config, "hc_sinkhorn_iters", 20)),
        use_engram=use_engram,
        engram_orders=config.engram_order_list,
        engram_heads=config.engram_heads,
        engram_d_mem=config.engram_d_mem,
        engram_table_size=config.engram_table_size,
        engram_fp8=bool(getattr(config, "engram_fp8", True)),
        vocab_size=config.vocab_size,
        csa_mode=csa_mode,
        compress_ratio=config.compress_ratio_of(layer_idx),
        swa_window=config.swa_window,
        ced=is_ced_dec,
        kv_quant=config.kv_quant,
        block_cache=bool(getattr(config, "kv_block_cache", False)),
    )
    if full:
        # 层序号：CSA2 选择通道按层号键控，生产者写、借用者读（见 CsaChannel）
        blk.attn.csa_index = layer_idx
    return blk


class _MTPLevel(nn.Module):
    """MTP 第 ≥2 级：上一级 hidden ⊕ 下一 token 嵌入 → 变换（DeepSeek 式值输入）.

    比第 1 级多喂一个"下一 token 的嵌入"——让更深一级的预测能挂上真实 token 信息，
    而不是只靠上一级的 hidden 硬猜。
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.norm_h = RMSNorm(config.d_model)
        self.norm_e = RMSNorm(config.d_model)
        self.proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)

    def forward(self, h_prev: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
        """拼接（归一化后的上一级 hidden、下一 token 嵌入）再做线性变换."""
        return self.proj(torch.cat(
            [self.norm_h(h_prev), self.norm_e(tok_emb)], dim=-1))


class MTPHead(nn.Module):
    """MTP 多 token 预测头（DeepSeek 系训练加速技巧，推理时关闭）.

    第 1 级（键名 `norm`/`proj`，与旧实现逐位一致）：主干 hidden 预测 t+2；
    更深级（`extra.{i}`，仅 depth>1 时存在）：上一级 hidden ⊕ 下一 token 嵌入，
    预测更远一步（DeepSeek-V4.1 `num_nextn_predict_layers`；本项目用线性变换替代
    整层 Transformer 块）。
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.depth = max(int(config.mtp_depth), 0)
        self.norm = RMSNorm(config.d_model)
        # 浅层变换：让 MTP 头有独立于主 head 的表达空间
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # 更深级（depth>1）；depth=1 时为空 ModuleList，state_dict 键与旧实现一致
        self.extra = nn.ModuleList([
            _MTPLevel(config) for _ in range(max(self.depth - 1, 0))
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """第 1 级前向（推理/起草用）：主干 hidden → 预测 t+2 的 hidden."""
        return self.proj(self.norm(h))

    def forward_all(
        self, h0: torch.Tensor, tok_embeds: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """逐级前向，返回各级 hidden（长度 = depth），供训练多目标 loss.

        h0：第 1 级输入（主干 hidden）；
        tok_embeds：第 d 级（d≥2）所需的"下一 token"嵌入，长度 depth-1（按级递增）。
        """
        outs = [self.forward(h0)]
        for level, emb in zip(self.extra, tok_embeds):
            outs.append(level(outs[-1], emb))
        return outs


class TinyLLM(nn.Module):
    """可训练+可推理的迷你 LLM."""

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.config = config
        # 块主序缓存的前提校验（压缩比>1 + 层内 SWA 且窗口 ≥ m）：早失败
        self.block_cache = config.block_cache_validated()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            build_block(config, i)
            for i in range(config.n_layers)
        ])
        # CSA2 跨层复用：链接共享 KV 下投影并判定是否启用（供各路径统一判断）
        # CED 编码器层数（0=关）；构建期校验解码器区间全是全注意力层（早失败）
        self.ced_split = config.ced_split_validated()
        self._link_csa_reuse()
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # 权重绑定：省一块词表矩阵（约 6M 参数）
            self.lm_head.weight = self.embed.weight
        self.mtp = MTPHead(config) if config.mtp_depth > 0 else None
        # RETRO 融合（默认 None：无参数，state_dict 兼容；开后处理 mem 输入）
        self.retro: nn.Module | None = None
        if config.retro_enabled and config.retro_every <= 0:
            from src.llm.local.retro import RetroFusion

            self.retro = RetroFusion(
                config.d_model, config.retro_heads, config.dropout)
        # 视觉 token 压缩（默认 None：无参数，state_dict 兼容；开后多一段视觉前缀）
        self.vision: nn.Module | None = None
        if config.vision_enabled:
            from src.llm.local.vision_compress import VisionCompressor

            self.vision = VisionCompressor(
                config.d_model, config.vision_d_in, config.vision_patch,
                config.vision_tokens, config.vision_compress, config.n_heads,
                config.dropout, config.vision_unshuffle)
        self._init_weights()

    def _csa_channel(self) -> CsaChannel | None:
        """CSA2 选择通道：仅 `sparse_indexer` 开启且真有借用层时创建（否则 None）."""
        if not (self.config.sparse_indexer and self._csa_active):
            return None
        return CsaChannel()

    def _link_csa_reuse(self) -> None:
        """CSA2：借用层与最近生产者的 KV 下投影共享（`w_dkv`/`w_kr` 同权重对象）.

        生产者 = `full`（自算 main KV + 索引 + 写缓存）/ `reindex`（借 main KV，
        但用自己的 indexer Q 重选 Top-K，并把结果作为新的"最新选择"给后续 Reuse）。
        借用者 = `reindex`/`reuse`（复用最近生产者的 main KV）。共享下投影既避免
        "永不更新的死参数"，也让"复用"名实相符：各层仍用自己的上投影 `w_uk`/`w_uv`
        展开 K/V，也有自己的 Q 与输出投影。
        `sparse_indexer` 开启时，Reindex 连 **indexer K**（`w_ik`）也借生产者的
        （V4.1：Reindex 复用 main KV + indexer K，只自带 indexer Q）。
        借用层记下 `csa_src`（最近生产者的层序号），供选择通道读取。
        """
        last_full = None
        for idx, layer in enumerate(self.layers):
            if not getattr(layer, "full_attn", False):
                continue
            mode = getattr(layer.attn, "csa_mode", "full")
            if mode != "full" and last_full is not None:
                layer.attn.w_dkv = last_full.attn.w_dkv
                layer.attn.w_kr = last_full.attn.w_kr
                if (getattr(layer.attn, "sparse_indexer", False)
                        and getattr(last_full.attn, "sparse_indexer", False)):
                    layer.attn.w_ik = last_full.attn.w_ik
                layer.attn.csa_src = last_full.attn.csa_index
                # 主 KV 的**压缩表示**也必须名实相符：借用层沿用生产者的压缩比与池化
                # 权重（同一对象），否则同一条 main KV 在两层被池化成不同条目。
                layer.attn.compress_ratio = getattr(last_full.attn, "compress_ratio", 1)
                _z = getattr(last_full.attn, "z_cmp", None)
                if _z is None:
                    if getattr(layer.attn, "z_cmp", None) is not None:
                        del layer.attn.z_cmp  # 生产者不压缩 → 本层也不留死参数
                else:
                    layer.attn.z_cmp = _z
                if mode == "reindex":
                    last_full = layer  # Reindex 也是生产者（更新"最新选择"）
            else:
                # 首个全注意力层（或借用者无前置生产者）退化为 Full，保证有共享源
                layer.attn.csa_mode = "full"
                layer.attn.csa_reuse = False
                layer.attn.csa_src = idx
                last_full = layer
        self._csa_active = any(
            getattr(layer.attn, "csa_mode", "full") != "full"
            for layer in self.layers if getattr(layer, "full_attn", False))

    def _init_weights(self) -> None:
        """GPT-2 式初始化：embedding 小方差 + 残差输出投影按深度缩放.

        否则 tied 词表头（std=1 的 embedding）会把 logits 放大约 sqrt(d) 倍，
        初始 CE 高达上百，训练起步困难。
        """
        import math

        n = self.config.n_layers
        torch.nn.init.normal_(self.embed.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * n)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attn.out_proj_weight(), std=residual_std)
            # 分层稀疏 indexer（若开）：小方差初始化，避免初始选择偏置过强
            if getattr(layer.attn, "sparse_indexer", False):
                torch.nn.init.normal_(layer.attn.w_iq.weight, std=0.02)
                torch.nn.init.normal_(layer.attn.w_ik.weight, std=0.02)
            for expert in list(layer.moe.experts) + list(layer.moe.shared):
                torch.nn.init.normal_(expert.w_down.weight, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_token_losses: bool = False,
        mem: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
        chunk_ids: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
    ) -> dict:
        """前向：训练时传 targets 返回组合 loss，推理时只返回 logits.

        loss = 主 CE(t+1) + mtp_weight * MTP 各级 CE(t+2..t+1+depth) 均值 + 各层 aux_loss 之和。
        return_token_losses=True 时附带逐 token 主 loss（RHO 选择用，
        ignore 位为 0 且反向无梯度，自然不会被选中）。
        mem/mem_mask: v1 单点检索记忆 (b, M, d)/(b, M)，仅 retro_enabled
        且 retro_every=0 时生效；retro 开启但 mem 为空（评测/生成路径）
        则跳过融合，测 backbone 本体。
        chunk_ids/chunk_mask: v2 交错融合的 (b, K, L)/(b, K, L) 文档 token，
        仅 retro_every>0 的层生效；模型侧 frozen embedding 查表编码
        （no_grad，不吃梯度；autocast 下自动同精度）。
        """
        v_len = 0 if vision_embeds is None else int(vision_embeds.shape[1])
        if input_ids.shape[1] + v_len > self.config.max_seq_len:
            raise ValueError(
                f"输入长度 {input_ids.shape[1]}(+视觉 {v_len}) 超过 "
                f"max_seq_len={self.config.max_seq_len}"
                "（RoPE 缓存盖不住）：请调大 max_seq_len 或直接用 longctx_config()（200K 预设）"
            )
        h = self.embed(input_ids)
        # Engram 上下文 id（右对齐到 h 末位）；有视觉前缀时视觉位置用词表外哨兵占位，
        # 文本 token 仍与 h 的文本段对齐（首个文本 token 的 n-gram 以哨兵打底）
        engram_ids = input_ids
        # 视觉前缀（结构预留，默认 None）：视觉 token 已投影到 d_model，拼在文本前。
        # 训练时（targets 非空）视觉段不参与 loss：logits 只取文本段错位一位对齐
        if v_len:
            h = torch.cat([vision_embeds.to(h.dtype), h], dim=1)
            engram_ids = torch.cat([
                input_ids.new_full((input_ids.shape[0], v_len),
                                   self.config.vocab_size),
                input_ids], dim=1)
        # 超连接：embedding 展开成 n 路流（各路初始相同，对称性由 H 参数打破）；
        # 模块（注意/MoE/记忆/retro）只看均值流，FLOP 不变
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            h = h.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        aux_total = torch.zeros((), device=h.device, dtype=h.dtype)
        # V2 交错 mem：frozen 查表一次编码，供各融合层共享（无梯度）
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None
        if chunk_ids is not None:
            if not self._has_interleaved_retro():
                raise ValueError("传了 chunk_ids 但交错融合未开（retro_every=0）")
            with torch.no_grad():
                b, kk, ll = chunk_ids.shape
                mem_h = self.embed(chunk_ids.reshape(b, kk * ll)).reshape(
                    b, kk * ll, -1)
            retro_mem = (mem_h, chunk_mask.reshape(b, kk * ll)
                         if chunk_mask is not None else None)
        elif self._has_interleaved_retro():
            pass  # 评测/生成无检索：各层跳过，测 backbone 本体
        shared = None  # CSA2：最近生产者的 main KV latent 缓存（供借用层复用）
        csa_sel = self._csa_channel()  # CSA2 选择通道（indexer 关时为 None）
        idx_kl: torch.Tensor | None = None  # indexer 蒸馏损失（未缩放，供日志）
        idx_n = 0  # 贡献蒸馏损失的层数（日志取层间均值）
        ced_split = self.ced_split
        h_ced: torch.Tensor | None = None  # 编码器输出 H_enc（CED 解码器层的 KV 源）
        hc_pre: torch.Tensor | None = None  # Single-Pass mHC：上一层给的 A_{l-1}
        for li, layer in enumerate(self.layers):
            if ced_split and li == ced_split:
                # 第 ced_split 层**之前**的 hidden（即编码器最后一层输出）就是 H_enc；
                # 超连接时取均值流（模块只看均值流，与 FLOP 约定一致）
                h_ced = h.mean(dim=2) if hyper_n else h
            ced_src = h_ced if (ced_split and li >= ced_split) else None
            # 梯度检查点只在训练时生效（推理/生成走正常路径，保留解码缓存）；
            # 训练不返回解码缓存（return_state=False），省状态循环与两次投影
            is_full = bool(getattr(layer, "full_attn", False))
            is_reuse = is_full and bool(getattr(layer.attn, "csa_reuse", False))
            reuse = shared if is_reuse else None
            # CSA2 开启时全注意力层需返回缓存（reserve=0，不补零、数值不变）供借用
            rs = bool(self._csa_active and is_full)
            if self.config.grad_ckpt and self.training:
                h, aux_total, p, hc_pre = self._layer_ckpt(
                    layer, h, aux_total, retro_mem, engram_ids, reuse, rs,
                    csa_sel, ced_src=ced_src, hc_pre=hc_pre)
            else:
                h, (p, aux), hc_pre = layer(
                    h, None, rs, retro_mem,
                    engram_ids=engram_ids, reuse=reuse,
                    csa_sel=csa_sel, ced_src=ced_src, hc_pre=hc_pre)
                aux_total = aux_total + aux
            kl = getattr(layer.attn, "_last_indexer_kl", None)
            if kl is not None:
                idx_kl = kl if idx_kl is None else idx_kl + kl
                idx_n += 1
            if self._csa_active and is_full and not is_reuse:
                shared = p
        if hyper_n:
            h = h.mean(dim=2)  # 流聚合（均值即 H_pre，论文消融默认）
        h = self.final_norm(h)
        if self.retro is not None:
            if mem is None:
                # 评测/生成路径无检索：跳过融合（测的是 backbone 本体；
                # retro 增益由专用评测度量，不在此）
                pass
            else:
                h = self.retro(h, mem, mem_mask)
        elif mem is not None:
            raise ValueError("传了 v1 mem 但单点融合未开（retro_enabled=False 或 retro_every>0 走交错）")
        logits = self.lm_head(h)
        out: dict = {"logits": logits, "aux_loss": aux_total.detach()}
        if idx_kl is not None:
            # indexer 蒸馏损失（层间均值、未缩放，便于和主 loss 比大小；已按系数计入 aux_loss）
            out["indexer_loss"] = (idx_kl / max(idx_n, 1)).detach()
        if targets is None:
            return out
        # 主 loss：预测下一 token（视觉前缀存在时只取文本段，视觉段不参与）
        text_logits = logits[:, v_len:v_len + input_ids.shape[1] - 1]
        ce_tokens = F.cross_entropy(
            text_logits.reshape(-1, self.config.vocab_size),
            targets[:, 1:].reshape(-1),
            reduction="none",
        )
        main_loss = ce_tokens.mean()
        loss = main_loss
        out["main_loss"] = main_loss.detach()
        # 附带引用（不 detach，供 RHO 替换时精确扣除主 loss 的梯度贡献）
        out["_main_mean"] = main_loss
        if return_token_losses:
            out["token_losses"] = ce_tokens.view(input_ids.shape[0], -1)
        # MTP loss：第 d 级用位置 i 的 hidden 预测 i+d+1 的 token（只取文本段）。
        # 第 1 级复现旧口径（逐位一致）；更深级额外喂入下一 token 嵌入。
        if self.mtp is not None:
            depth = self.mtp.depth
            n = input_ids.shape[1] - 1 - depth  # 可用位置数（各级共用）
            if n > 0:
                # 第 d（≥2）级所需的"下一 token"嵌入 = input_ids[:, i+d]（用输入序列，
                # 避开 SFT 的 -100 掩码无法查表的问题）
                tok_embeds = [self.embed(input_ids[:, d:d + n])
                              for d in range(2, depth + 1)]
                level_h = self.mtp.forward_all(
                    h[:, v_len:v_len + n], tok_embeds)
                level_losses = [
                    F.cross_entropy(
                        self.lm_head(hd).reshape(-1, self.config.vocab_size),
                        targets[:, d + 1:d + 1 + n].reshape(-1),
                    )
                    for d, hd in enumerate(level_h, start=1)
                ]
                mtp_loss = torch.stack(level_losses).mean()
                loss = loss + self.config.mtp_loss_weight * mtp_loss
                out["mtp_loss"] = mtp_loss.detach()
                out["mtp_level_losses"] = [x.detach() for x in level_losses]
        loss = loss + aux_total
        out["loss"] = loss
        return out

    def _has_interleaved_retro(self) -> bool:
        """是否有 V2 交错融合层（retro_enabled 且 retro_every>0）."""
        return bool(self.config.retro_enabled and self.config.retro_every > 0)

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """按最近一次训练前向的负载更新各层路由偏置（免辅助损失负载均衡）.

        每个优化器步调用一次；`router_aux_free=False` 时各层为无操作。
        """
        for layer in self.layers:
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.update_router_bias()

    def _has_engram(self) -> bool:
        """是否有 Engram 条件记忆层（`engram_layers` 显式或 `engram_every>0`）."""
        return any(getattr(layer, "engram", None) is not None for layer in self.layers)

    def _engram_lookback(self) -> int:
        """Engram 解码所需的最小右对齐上下文长度（最大 N-gram 阶 - 1；无卷积）."""
        best = 0
        for layer in self.layers:
            eng = getattr(layer, "engram", None)
            if eng is not None:
                best = max(best, eng.lookback)
        return best

    def _engram_block_ids(
        self, ctx_ids: torch.Tensor, tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """拼 Engram 右对齐上下文 `ctx_ids` + `tokens`（无 Engram 层返回 None）.

        `_decode_step`/`_decode_multi` 只对末 `len(tokens)` 位算 n-gram；这里补足
        前 `lookback` 个 token，避免每步重扫全史（卷积+哈希都只依赖 id，无状态）。
        """
        if not self._has_engram():
            return None
        n = self._engram_lookback()
        if n <= 0:
            return tokens
        return torch.cat([ctx_ids[:, -n:], tokens], dim=1)

    def hyper_gain(self) -> float | None:
        """超连接复合 Amax 增益（论文 Fig.3/7 指标），无超连接返回 None.

        读各层 `_last_H`（需先开 track_stats 跑一次前向）；≈1 健康，
        飙升即残差流爆炸预警。"""
        mats = []
        for layer in self.layers:
            hyp = getattr(layer, "hyper", None)
            if hyp is not None and getattr(hyp, "_last_H", None) is not None:
                mats.append(hyp._last_H)
        if not mats:
            return None
        from src.llm.local.hyperconn import composite_gain

        return composite_gain(mats)

    def _layer_ckpt(
        self, layer: HybridBlock, h: torch.Tensor, aux_total: torch.Tensor,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        engram_ids: torch.Tensor | None = None,
        reuse: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
        csa_sel: CsaChannel | None = None,
        ced_src: torch.Tensor | None = None,
        hc_pre: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None,
               torch.Tensor | None]:
        """单层梯度检查点：重算换显存（训练 past=None，不用解码缓存）.

        `ced_src`（CED 解码器层的编码器输出）**必须作为 checkpoint 的显式输入**：
        闭包里捕获的张量在重算时被当作常量，梯度不会回流到编码器层
        （retro_mem/engram_ids 是 frozen 的，不需要梯度，所以能留在闭包）。
        """
        from torch.utils.checkpoint import checkpoint

        def fn(hh: torch.Tensor, src: torch.Tensor | None,
               pre: torch.Tensor | None):
            out, (p, aux), nxt = layer(hh, None, return_state, retro_mem,
                                       engram_ids=engram_ids, reuse=reuse,
                                       csa_sel=csa_sel, ced_src=src,
                                       hc_pre=pre)
            return out, aux, p, nxt

        # `hc_pre` 与 `ced_src` 同理必须显式传入：它来自上一层、需要梯度，
        # 被闭包捕获会在重算时当成常量，梯度不回传到上一层的系数预测器
        h_new, aux, p, nxt = checkpoint(fn, h, ced_src, hc_pre,
                                        use_reentrant=False)
        return h_new, aux_total + aux, p, nxt

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成（增量 KV/状态缓存，线性层 O(1)/步）.

        temperature=0 为贪心；>0 时按温度采样（可配 top-k 截断）。
        生成后恢复之前的 train/eval 状态：训练循环中穿插采样若把模型
        永久留在 eval 模式，梯度检查点会被静默关闭，显存随即爆炸。
        """
        was_training = self.training
        self.eval()
        try:
            return self._generate_inner(
                input_ids, max_new_tokens, temperature, top_k, eos_id,
                repetition_penalty)
        finally:
            if was_training:
                self.train()

    def _prefill(
        self, input_ids: torch.Tensor,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        max_new_tokens: int = 64,
    ) -> tuple[torch.Tensor, list]:
        """Prefill：整段 prompt 一次前向，返回 (末位置 hidden 前的 h, 每层缓存）.

        h 为全序列 hidden（调用方取 h[:, -1:] 算 logits）；pasts 供单步解码续跑。
        retro_mem：V2 交错融合的 frozen chunk 编码（生成带检索时传，与 decode 同值）。
        max_new_tokens: 静态缓存预留（prompt 长 + 该值；generate 精确传入，
        不够写时解码报错，SessionCache 用 ensure_room 提前扩）。

        CED（V4.1 §2.2）：编码器层照常在**全序列**上跑；解码器层的全局 KV 由编码器
        输出投影（与它在哪些位置跑过无关），其输出只服务自己的 SWA 环——所以解码器
        段只喂最后 `swa_window` 个 token（SWA 关时 1 个，只要末位置算 logits）。
        这就是论文的 Decoder SWA Bounded Replay：预填充从 O(NL) 降到 O(NL/2)。
        返回的 h 因此只覆盖尾段（调用方只用末位置），pasts 的 pos 仍是全序列长度。
        """
        h = self.embed(input_ids)
        n_tok = input_ids.shape[1]
        reserve = n_tok + max(0, max_new_tokens)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            h = h.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        shared = None  # CSA2：最近生产者的 main KV latent 缓存（供借用层复用）
        csa_sel = self._csa_channel()  # CSA2 选择通道（indexer 关时为 None）
        ced_split = self.ced_split
        # 解码器段从哪个绝对位置开始（块对齐；关掉跳算时 = 0，即全序列）
        ced_q_off = self._ced_prefill_q_off(n_tok)
        h_ced: torch.Tensor | None = None
        hc_pre: torch.Tensor | None = None  # Single-Pass mHC：上一层给的 A_{l-1}
        pasts: list = []
        for li, layer in enumerate(self.layers):
            is_full = bool(getattr(layer, "full_attn", False))
            is_reuse = is_full and bool(getattr(layer.attn, "csa_reuse", False))
            reuse = shared if is_reuse else None
            if ced_split and li == ced_split:
                h_ced = h.mean(dim=2) if hyper_n else h
            if ced_split and li >= ced_split:
                if li == ced_split:
                    h = h_ced[:, ced_q_off:]  # 只保留查询尾段
                h, (p, _), hc_pre = layer(
                    h, retro_mem=retro_mem, reserve=reserve,
                    engram_ids=input_ids, reuse=reuse, csa_sel=csa_sel,
                    ced_src=h_ced, q_off=ced_q_off, hc_pre=hc_pre)
            else:
                h, (p, _), hc_pre = layer(
                    h, retro_mem=retro_mem, reserve=reserve,
                    engram_ids=input_ids, reuse=reuse, csa_sel=csa_sel,
                    hc_pre=hc_pre)
            pasts.append(p)
            if self._csa_active and is_full and not is_reuse:
                shared = p
        if hyper_n:
            h = h.mean(dim=2)
        return self.final_norm(h), pasts

    def _sample_next(
        self,
        h_last: torch.Tensor,
        temperature: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        past_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按温度/top-k 采样下一步（贪心 temperature=0），返回 (b, 1) id.

        repetition_penalty: HF 语义，已出现 token 的 logit 正值除以惩罚、
        负值乘以惩罚（=1.0 关闭，逐位等价旧行为）；past_ids 为 (b, T) 上下文
        （含已生成），None 时不惩罚。专治贪心在高频 token（模板 `<`、
        markdown `####`）上的循环吸引子。
        """
        next_logits = self.lm_head(h_last)[:, -1, :].float()
        if repetition_penalty != 1.0 and past_ids is not None:
            v = next_logits.shape[-1]
            seen = torch.zeros_like(next_logits, dtype=torch.bool)
            seen.scatter_(1, past_ids.clamp(0, v - 1), True)
            next_logits = torch.where(
                seen,
                torch.where(next_logits < 0,
                            next_logits * repetition_penalty,
                            next_logits / repetition_penalty),
                next_logits,
            )
        if temperature > 0:
            next_logits = next_logits / max(temperature, 1e-6)
            if top_k > 0:
                kth, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits = torch.where(
                    next_logits < kth[..., -1:],
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )
            return torch.multinomial(F.softmax(next_logits, dim=-1), 1)
        return next_logits.argmax(-1, keepdim=True)

    def _ced_prefill_q_off(self, n_tok: int) -> int:
        """CED 预填充里解码器段的**起始绝对位置**（跳算边界）.

        - `ced_prefill_skip=False` → 0（全序列，严格对齐训练口径，不省算）；
        - 默认（跳算）：对齐到 `sparse_chunk` 的块边界（最后一个 token 所在块的
          起点）。**必须块对齐**：全局稀疏 pattern（sink/stride/窗口）依赖查询块
          起点，块内对齐后跳算与全量的分块完全一致——`swa_window=0` 时输出
          **逐位等价**（pitfall 35 同一类坑）。
        - `swa_window>0` 时再把起点按块往前推，直到尾段覆盖 **2 个窗口**：
          窗口列的 K/V 来自窗口内各位置自己的 hidden，要让"将来会被读到的那
          W 个环行"本身也自洽，就得再往前留一个 W（尾段首部那段窗口不全会偏，
          但它的偏差沿残差快速衰减）。实测 tail=2W 时末位置与环都与全量**逐位
          一致**；tail=W 时末位置差 1e-1 量级。
        """
        if not self.ced_split or not bool(self.config.ced_prefill_skip):
            return 0
        chunk = max(1, int(self.config.sparse_chunk))
        q_off = ((n_tok - 1) // chunk) * chunk
        need = max(1, 2 * int(self.config.swa_window))
        while n_tok - q_off < need and q_off > 0:
            q_off = max(0, q_off - chunk)
        return q_off

    def _decode_step(
        self, nxt: torch.Tensor, pasts: list,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        engram_ids: torch.Tensor | None = None,
        write_cache: bool = True,
        swa_floor: int = 0,
    ) -> tuple[torch.Tensor, list]:
        """单 token 步进各层：返回 (新 hidden, 新缓存），供流式/批量生成共用.

        engram_ids: Engram 上下文 id（(b, L≥max_order)，右对齐到本步 token）。
        write_cache=False / swa_floor：只读重放（SWA Bounded Replay），见
        `SparseMLAModule.forward` 与 `TinyLLM.replay_swa_from_ids`。
        """
        hh = self.embed(nxt)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            hh = hh.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        new_pasts: list = []
        csa_sel = self._csa_channel()  # 每步一个：生产者本步写、借用者同步读
        ced_split = self.ced_split
        h_ced: torch.Tensor | None = None  # 当步编码器输出（CED 解码器层的 KV 源）
        hc_pre: torch.Tensor | None = None  # Single-Pass mHC：上一层给的 A_{l-1}
        for li, (layer, p) in enumerate(zip(self.layers, pasts)):
            if ced_split and li == ced_split:
                h_ced = hh.mean(dim=2) if hyper_n else hh
            ced_src = h_ced if (ced_split and li >= ced_split) else None
            # 每层 norm 由 block 内部处理，这里直接走 attn+moe 等价路径：
            # 为复用逻辑，重新拼 block 前向（单 token 开销可忽略）
            xxs = hh
            hc_aux = None
            if hyper_n and layer.hyper is not None:
                hh_in, hc_aux = layer.hyper.enter(hh, hc_pre)
                hc_pre = hc_aux[0]
            else:
                hh_in = hh
            hh = hh_in
            if layer.engram is not None and engram_ids is not None:
                # Engram 与 block.forward 同序（注意力之前残差并联）
                hh = layer.engram(hh, engram_ids)
            hh_norm = layer.norm1(hh)
            if layer.full_attn:
                a_out, p2 = layer.attn(hh_norm, p, csa_sel=csa_sel,
                                       write_cache=write_cache,
                                       swa_floor=swa_floor,
                                       ced_src=ced_src)  # type: ignore[call-arg]
            else:
                a_out, p2 = layer.attn(hh_norm, p)  # type: ignore[arg-type]
            hh = hh + a_out
            m_out, _ = layer.moe(layer.norm2(hh))
            hh = hh + m_out
            if layer.memory is not None:
                # 记忆层也在 block 前向里（MoE 之后残差并联）：解码必须同步走，
                # 否则 value 训出非零后增量解码与全前向分叉（单测锁定）
                hh = layer.memory(hh)
            if layer.retro is not None and retro_mem is not None:
                # V2 交错融合同样同步（block 前向末节；w_o 非零后漏掉即分叉）
                mh, mm = retro_mem
                hh = layer.retro(hh, mh, mm)
            if hyper_n and layer.hyper is not None:
                # 超连接写回与 block.forward 同式（均值流增量 + 流混合）
                hh = layer.hyper.combine(xxs, hh - hh_in, hc_aux)
            new_pasts.append(p2)
        if hyper_n:
            hh = hh.mean(dim=2)
        return self.final_norm(hh), new_pasts

    @torch.no_grad()
    def _decode_multi(
        self, tokens: torch.Tensor, pasts: list,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        engram_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list, list]:
        """并行验证：一次前向喂 t 个 token，返回 (全段 hidden (b,t,d), 新缓存).

        投机解码专用：与逐 token `_decode_step` **同一算术**（MLA 多查询 SDPA
        逐行独立、稀疏路径按 start=total-t 聚集、线性层走多 token 增量递推），
        因此验证出的主干贪心序列即真实贪心序列（契约见 test_speculative.py）。
        普通生成仍走 `_decode_step`，此路径不参与编译。

        额外返回 `step_pasts[i]`：提交前 i+1 个 token 后每层的完整缓存。部分
        接受时据此直接跳到接受位（零重放）。MLA 复用同一缓冲、只调 pos；线性层
        取 `multi_steps[i]`（该分支已按步记录）。
        """
        hh = self.embed(tokens)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            hh = hh.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        new_pasts: list = []
        lin_steps: dict[int, list] = {}
        csa_sel = self._csa_channel()  # 并行验证同一批 token：单块选择通道
        ced_split = self.ced_split
        h_ced: torch.Tensor | None = None
        hc_pre: torch.Tensor | None = None  # Single-Pass mHC：上一层给的 A_{l-1}
        for li, (layer, p) in enumerate(zip(self.layers, pasts)):
            if ced_split and li == ced_split:
                h_ced = hh.mean(dim=2) if hyper_n else hh
            ced_src = h_ced if (ced_split and li >= ced_split) else None
            xxs = hh
            hc_aux = None
            if hyper_n and layer.hyper is not None:
                hh_in, hc_aux = layer.hyper.enter(hh, hc_pre)
                hc_pre = hc_aux[0]
            else:
                hh_in = hh
            hh = hh_in
            if layer.engram is not None and engram_ids is not None:
                # Engram 与 block.forward 同序（并行验证的算术也必须一致）
                hh = layer.engram(hh, engram_ids)
            hh_norm = layer.norm1(hh)
            if layer.full_attn:
                a_out, p2 = layer.attn(hh_norm, p, csa_sel=csa_sel,
                                       ced_src=ced_src)  # type: ignore[call-arg]
            else:
                a_out, p2 = layer.attn(hh_norm, p)  # type: ignore[arg-type]
                lin_steps[li] = layer.attn.multi_steps
            hh = hh + a_out
            m_out, _ = layer.moe(layer.norm2(hh))
            hh = hh + m_out
            if layer.memory is not None:
                hh = layer.memory(hh)
            if layer.retro is not None and retro_mem is not None:
                mh, mm = retro_mem
                hh = layer.retro(hh, mh, mm)
            if hyper_n and layer.hyper is not None:
                hh = layer.hyper.combine(xxs, hh - hh_in, hc_aux)
            new_pasts.append(p2)
        if hyper_n:
            hh = hh.mean(dim=2)
        # 每步 pasts：线性层取记录的第 i 步；MLA/稀疏复用原缓冲（已写全部位置），
        # 只把 pos 调回 "输入 i+1 个 token 后" 的位置（越界写由掩码屏蔽）
        t = tokens.shape[1]
        if t == 1:
            step_pasts = [new_pasts]  # 单 token：并行结果即该步缓存
        else:
            step_pasts = []
            for i in range(t):
                plist: list = []
                for li, (layer, old_p) in enumerate(zip(self.layers, pasts)):
                    if li in lin_steps:
                        plist.append(lin_steps[li][i])
                    else:
                        # 兼容三元组（无 SWA）与五元组（层内 SWA 环，定长不随步长变）
                        buf_c, buf_kr, pos = old_p[0], old_p[1], old_p[2]
                        plist.append((buf_c, buf_kr, pos + i + 1)
                                     + tuple(old_p[3:]))
                step_pasts.append(plist)
        return self.final_norm(hh), new_pasts, step_pasts

    def _spec_eligible(self, input_ids: torch.Tensor, temperature: float) -> bool:
        """是否走投机解码：需开关开、batch=1、纯贪心（temperature=0）.

        层内 SWA 开启时**禁用**：并行验证一次喂 `t` 个 token，部分接受时
        缓存里已写进了未接受 token 的 SWA 环行（位置却落在窗口内），后续
        解码会把"影子 token"的 K/V 当真实历史读走（pitfall 37）。
        先保证正确性，再谈这条组合的收益（正常生成仍走 `_decode_step`）。
        """
        if int(getattr(self.config, "swa_window", 0)) > 0:
            return False
        return bool(self.config.draft_enabled) and input_ids.shape[0] == 1 \
            and temperature == 0.0

    @torch.no_grad()
    def _spec_stream(
        self, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None,
        repetition_penalty: float,
    ):
        """投机解码内循环（贪心；逐 token yield 新 id int）.

        每轮：主头出权威下一位 → 起草器补足后续候选 → `_decode_multi` 一次并行
        验证 → 接受最长草稿前缀。部分接受时直接取 `step_pasts[accepted]`（并行
        前向已按步记录缓存）跳位，**零重放**。输出与 `_generate_inner` 的贪心
        逐位一致（单测锁定）。
        """
        from src.llm.local.spec_decode import (
            build_draft_provider,
            schedule_verify_length,
            scored_propose,
        )

        cfg = self.config
        provider = build_draft_provider(self, cfg.draft_source, cfg.draft_ngram_min)
        window = max(1, int(cfg.draft_window))
        # DSpark 置信调度（默认关）：按草稿存活置信度动态定本轮验证长度
        dspark = bool(getattr(cfg, "dspark_enabled", False))
        thr = float(getattr(cfg, "dspark_conf_threshold", 0.5))
        min_v = int(getattr(cfg, "dspark_min_verify", 1))
        n_proposed = 0
        n_verified = 0
        dev = input_ids.device
        # 预留：每轮最多并行写 window 个位置，多留 window 防越界
        h, pasts = self._prefill(
            input_ids, max_new_tokens=max_new_tokens + window)
        ctx = input_ids[0].tolist()
        n_new = 0
        rounds = 0
        n_accepted = 0
        while n_new < max_new_tokens:
            rounds += 1
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=dev)
            t0 = int(self._sample_next(
                h[:, -1:], 0.0, 0, repetition_penalty, ctx_t)[0, 0])
            k = window - 1
            if k > 0 and dspark:
                # DSpark：取带置信度的草稿，按累计存活概率截断验证长度
                drafts, confs = scored_propose(provider, ctx, h[:, -1:], k)
                n_proposed += len(drafts)
                drafts = list(drafts[:schedule_verify_length(confs, thr, min_v, k)])
                n_verified += len(drafts)
            else:
                drafts = provider.propose(ctx, h[:, -1:], k) if k > 0 else []
            feed = [t0] + list(drafts)
            y = torch.tensor([feed], dtype=torch.long, device=dev)
            eids = self._engram_block_ids(ctx_t, y)
            h_multi, _, step_pasts = self._decode_multi(
                y, pasts, engram_ids=eids)
            # 验证：逐位比草稿与主干贪心（seen 上下文逐步累积，惩罚口径一致）
            accepted = 0
            seen = list(ctx)
            for i in range(1, len(feed)):
                seen_t = torch.tensor([seen], dtype=torch.long, device=dev)
                tg = int(self._sample_next(
                    h_multi[:, i - 1:i], 0.0, 0, repetition_penalty, seen_t)[0, 0])
                if tg == feed[i]:
                    accepted += 1
                    seen.append(feed[i])
                else:
                    break
            n_accepted += accepted
            committed = feed[:1 + accepted]
            # 无论全接受还是部分接受，都直接跳到"已提交前缀末位"那一步的缓存：
            # h/缓存取自同一次并行验证，无需回滚重放（step_pasts 已按步对齐）。
            h = h_multi[:, accepted:accepted + 1]
            pasts = step_pasts[accepted]
            stop = False
            for tok in committed:
                yield tok
                n_new += 1
                ctx.append(tok)
                if (eos_id is not None and tok == eos_id) or n_new >= max_new_tokens:
                    stop = True
                    break
            if stop:
                break
        # 统计（日志/单测用：accepted 越接近 window×rounds 越省）
        self._spec_stats = {"rounds": rounds, "accepted": n_accepted,
                            "emitted": n_new, "window": window,
                            "dspark": dspark}
        if dspark:
            self._spec_stats["proposed"] = n_proposed
            self._spec_stats["verified"] = n_verified

    def compile_decode(self, mode: str = "default") -> "TinyLLM":
        """编译单步解码（推理专用）.

        前提：静态缓存（形状恒定，无增长 `cat`），否则逐周 Rehab 编反而更慢
        （教训见 pitfalls 第 9 条）。sparse 层的 pattern 选择会 graph-break
        但不重编。约束：编译后改权重/切精度/切模式必须重调；
        训练路径不用；与 eager 逐位一致，单测锁定。返回 self。
        """
        self._decode_step = torch.compile(  # type: ignore[method-assign]
            self._decode_step, mode=mode, dynamic=True)
        return self

    @torch.no_grad()
    def _generate_inner(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成内循环（调用方 generate 已处理 eval 模式切换）.

        temperature=0 为贪心；>0 时按温度采样（可配 top-k 截断）。
        与 stream_tokens 同一基元（数学一致，单测锁定）。
        """
        if self._spec_eligible(input_ids, temperature):
            ids = list(self._spec_stream(
                input_ids, max_new_tokens, eos_id, repetition_penalty))
            if not ids:
                return input_ids.clone()
            add = torch.tensor([ids], dtype=input_ids.dtype,
                               device=input_ids.device)
            return torch.cat([input_ids, add], dim=1)
        h, pasts = self._prefill(input_ids, max_new_tokens=max_new_tokens)
        cur = input_ids
        for _ in range(max_new_tokens):
            nxt = self._sample_next(h[:, -1:], temperature, top_k,
                                    repetition_penalty, cur)
            eids = self._engram_block_ids(cur, nxt)
            cur = torch.cat([cur, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            h, pasts = self._decode_step(nxt, pasts, engram_ids=eids)
        return cur

    @torch.no_grad()
    def stream_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
    ):
        """逐 token 生成器（SSE 流式/OpenAI stream 用），逐个 yield 新 id（int）.

        与 generate() 同一 prefill+解码路径（数学一致，单测锁定），
        生成后同样恢复 train/eval 状态。
        """
        was_training = self.training
        self.eval()
        try:
            if self._spec_eligible(input_ids, temperature):
                yield from self._spec_stream(
                    input_ids, max_new_tokens, eos_id, repetition_penalty)
                return
            h, pasts = self._prefill(input_ids, max_new_tokens=max_new_tokens)
            seen: list[torch.Tensor] = []
            for _ in range(max_new_tokens):
                cur = (torch.cat([input_ids, *seen], dim=1) if seen
                       else input_ids)
                nxt = self._sample_next(h[:, -1:], temperature, top_k,
                                        repetition_penalty, cur)
                eids = self._engram_block_ids(cur, nxt)
                yield int(nxt[0, 0])
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
                seen.append(nxt)
                h, pasts = self._decode_step(nxt, pasts, engram_ids=eids)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def encode_full_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取全序列 hidden（调用方按真实长度 gather，padding 安全）.

        不碰 train/eval 状态（只关梯度；dropout>0 时调用方自行 eval）。
        训练长序列走分块路径（linear_chunk/sparse），与 generate 的 prefill 一致。
        """
        h = self.embed(input_ids)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            h = h.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        shared = None  # CSA2：最近生产者的 main KV latent 缓存
        csa_sel = self._csa_channel()  # CSA2 选择通道（indexer 关时为 None）
        ced_split = self.ced_split
        h_ced: torch.Tensor | None = None
        hc_pre: torch.Tensor | None = None  # Single-Pass mHC：上一层给的 A_{l-1}
        for li, layer in enumerate(self.layers):
            if ced_split and li == ced_split:
                h_ced = h.mean(dim=2) if hyper_n else h
            ced_src = h_ced if (ced_split and li >= ced_split) else None
            is_full = bool(getattr(layer, "full_attn", False))
            is_reuse = is_full and bool(getattr(layer.attn, "csa_reuse", False))
            reuse = shared if is_reuse else None
            rs = bool(self._csa_active and is_full)
            h, (p, _), hc_pre = layer(
                h, None, rs, engram_ids=input_ids, reuse=reuse,
                csa_sel=csa_sel, ced_src=ced_src, hc_pre=hc_pre)
            if self._csa_active and is_full and not is_reuse:
                shared = p
        if hyper_n:
            h = h.mean(dim=2)
        return self.final_norm(h)

    @torch.no_grad()
    def replay_swa_from_ids(
        self, ids: torch.Tensor, pasts: list,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> list:
        """SWA Bounded Replay：只重放最近 `n_win` 个 token 重建各层 SWA 环.

        V4.1 §3.2.2 的部署口径：持久化缓存只存**全局主 KV**（无法重建），
        SWA 环定长、可重建，故不必落盘；恢复时重放最近 `n_win` 个 token
        近似重建（精确重建要 `L × n_win`，有效感受野 << `n_win`）。

        实现严格对着论文那段话：
        - "replay the last n_win tokens ... **reusing the cached global KV
          without recomputation or overwriting**"——重放走只读解码
          （`write_cache=False`），全局主 KV 只读不写，权威值一个字节都不动；
        - "for a replay starting at position s, a query at position i attends to
          SWA keys in `[max(s, i-W+1), i]`"——环从空开始（`empty_swa_rings`）
          且读时带 `floor=s`，重放段之前的 SWA 键根本不存在，也就读不到；
        - 逐层各自重放：SWA 的 K/V 来自**本层输入 hidden**，所以必须真跑前向。

        近似来源（写进文档，不粉饰）：
        1. 重放段各层的输入 hidden 由"全局读权威 + SWA 截断到 s"共同决定，
           与真值只在 SWA 局部键上有偏差；
        2. **线性注意力层的递推状态不回退，而是清零点**（`start=0` 整段重放时它
           本来就从零重建，故逐位精确；`start>0` 时等价于"在重放起点放弃更早的
           递推记忆"，与 SWA 截断同一语义）。**不能沿用缓存末态**：末态已经含
           重放段的贡献，再喂一遍就是重复计入，实测下一步 hidden 误差 3e-1
           （清零 4e-6，无环冷启动 3.5）——比"近似"更糟，是矛盾状态（pitfall 39）。

        返回新的 `pasts`：全局主 KV 与 pos 用**权威原值**，SWA 环换成重放结果。
        """
        n_win = self.config.swa_window
        if n_win <= 0 or not pasts:
            return pasts
        p0 = pasts[0]
        if not (isinstance(p0, tuple) and len(p0) >= 3
                and isinstance(p0[2], torch.Tensor) and p0[2].numel() == 1):
            return pasts
        total = int(p0[2].item())
        start = max(0, total - n_win)
        # ids 只是记账用、可常驻 CPU（长会话省显存）；这里按模型设备临时搬一份
        ids = ids.to(p0[0].device)
        ctx = ids[:, :total] if ids.shape[1] >= total else ids
        tail = ctx[:, start:]
        t_tail = int(tail.shape[1])
        # 重放用缓存：pos 回退到 start；SWA 环清空；线性层状态清零（见 docstring 第 2 条）
        replay: list = []
        for layer, p in zip(self.layers, pasts):
            if not (isinstance(p, tuple) and len(p) >= 3
                    and isinstance(p[2], torch.Tensor) and p[2].numel() == 1):
                replay.append(tuple(
                    torch.zeros_like(x) if isinstance(x, torch.Tensor) else x
                    for x in p))
                continue
            rings: tuple = ()
            if (bool(getattr(layer, "full_attn", False))
                    and getattr(layer.attn, "swa", None) is not None):
                cap0 = p[0]
                rings = layer.attn.empty_swa_rings(
                    1, batch=cap0.shape[0], device=cap0.device,
                    dtype=cap0.dtype)
            replay.append((p[0], p[1], p[2] - t_tail) + rings)
        for i in range(t_tail):
            _, replay = self._decode_step(
                tail[:, i:i + 1], replay, retro_mem=retro_mem,
                engram_ids=ctx[:, :start + i + 1], write_cache=False,
                swa_floor=start)
        # 取重放出的 SWA 环 + 权威的全局主 KV / pos
        out: list = []
        for p, r in zip(pasts, replay):
            out.append(tuple(p[:3]) + tuple(r[3:]) if len(r) > 3 else p)
        return out

    @torch.no_grad()
    def encode_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取末位置 hidden（输入定长/单条时用；变长 batch 请用 full+gather）.

        不碰 train/eval 状态（只关梯度；dropout>0 时调用方自行 eval）。
        """
        return self.encode_full_hidden(input_ids)[:, -1, :]

    @torch.no_grad()
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """图像 → 视觉 token 序列 (B, n_tokens, d_model)（需 vision_enabled=True）."""
        if self.vision is None:
            raise ValueError("vision_enabled=False，模型无视觉模块")
        return self.vision(images)

    def setup_offload(self, compute_device=None) -> "TinyLLM":
        """按配置为每层 MoE 开启专家 offload / LES 调度（默认关时无操作）.

        与 `compile_decode` 同风格：改权重/切设备后需重新调用。仅推理路径用。
        返回 self。
        """
        cfg = self.config
        if not (cfg.offload_enabled or cfg.les_enabled):
            return self
        if compute_device is None:
            compute_device = next(self.parameters()).device
        for li, layer in enumerate(self.layers):
            if cfg.offload_enabled:
                layer.moe.enable_offload(
                    compute_device,
                    target=cfg.offload_target,
                    offload_dir=cfg.offload_dir,
                    capacity=cfg.offload_capacity,
                    prefetch=cfg.offload_prefetch,
                    name=f"layer{li}",
                )
            if cfg.les_enabled:
                layer.moe.enable_les(cfg.les_history, cfg.les_topn)
        return self

    def count_params(self) -> dict:
        """统计总参数与每 token 激活参数（MoE 只计 top-k + 共享专家）."""
        total = sum(p.numel() for p in self.parameters())
        c = self.config
        # 每专家参数：SwiGLU 三矩阵
        per_expert = 3 * c.d_model * c.expert_hidden
        per_layer_active = (
            # 注意力取全量（上界估计，线性层与 MLA 相近量级）
            sum(p.numel() for p in self.layers[0].attn.parameters())
            + (c.top_k + c.n_shared) * per_expert
            + c.d_model * c.n_experts  # 路由矩阵全量参与
        )
        # 记忆层激活：每 token top-k 个槽（keys 查表 + values 加权）
        n_mem_layers = sum(1 for layer in self.layers
                           if getattr(layer, "memory", None) is not None)
        mem_active = n_mem_layers * c.memory_topk * c.d_model * 2
        # RETRO 融合：整块参与（单点开后计入；交错按层数累加）
        retro_active = (sum(p.numel() for p in self.retro.parameters())
                        if self.retro is not None else 0)
        retro_active += sum(
            sum(p.numel() for p in layer.retro.parameters())
            for layer in self.layers
            if getattr(layer, "retro", None) is not None)
        # 超连接：整块参与（n 小量，可忽略但计入）
        hyper_active = sum(
            sum(p.numel() for p in layer.hyper.parameters())
            for layer in self.layers
            if getattr(layer, "hyper", None) is not None)
        # Engram：只有投影/归一参与计算，查表每 token 仅 gather 一行（各阶拼接 = d_out），
        # 巨大的哈希表不进"激活参数"
        engram_active = 0
        for layer in self.layers:
            eng = getattr(layer, "engram", None)
            if eng is None:
                continue
            table = sum(t.weight.numel() for t in eng.tables)
            engram_active += (sum(p.numel() for p in eng.parameters())
                              - table + eng.d_out)
        active = (
            self.embed.weight.numel()
            + per_layer_active * c.n_layers
            + mem_active + retro_active + hyper_active + engram_active
            + self.final_norm.weight.numel()
            + (0 if c.tie_embeddings else self.lm_head.weight.numel())
        )
        return {"total": total, "active": active,
                "total_m": round(total / 1e6, 1), "active_m": round(active / 1e6, 1)}

    def kv_cache_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """满上下文 KV/状态缓存字节数（MLA 层 latent + 线性层状态）.

        CSA2 跨层复用时，Reuse 层不新增 KV 缓存（借最近 Full 层），故只计 Full 层。
        """
        c = self.config
        # 主 KV：逐 token 增长；块主序缓存（#4）下每层只存 `ceil(L/m)` 行（行值 =
        # m 个 token 的块均值），`kv_quant` 非空时按量化格式计（fp4 4.5 bit/通道、
        # fp8 8.5 bit/通道）。**逐层算**：混合压缩比时各层行数不同。
        if c.kv_quant:
            from src.llm.local.kv_quant import kv_bytes_per_token

            row_bytes = (
                kv_bytes_per_token(c.kv_lora_rank, c.kv_quant)
                + kv_bytes_per_token(c.qk_rope_dim, c.kv_quant, block=16)
            )
        else:
            row_bytes = (c.kv_lora_rank + c.qk_rope_dim) * dtype_bytes
        lin_per_layer = c.n_heads * (c.d_model // c.n_heads) ** 2 * dtype_bytes
        # 层内 SWA：定长环（k/v 各 n_heads*head_dim + 共享 rope 键 n_win*qk_rope_dim）
        # ——**与序列长度无关**，故这里是加法项而不是乘 seq_len 项
        # （V4.1 的"SWA 存储有界，与 L 无关"）；SWA 是**每层自带**，不随 Reuse 减少
        swa_per_layer = 0
        if c.swa_window > 0:
            head_dim = c.d_model // c.n_heads
            if c.kv_quant:
                # SWA 环按 FP8 存（V4.1：SWA 对量化更敏感，保留 FP8）
                from src.llm.local.kv_quant import kv_bytes_per_token

                swa_per_layer = int(round(kv_bytes_per_token(
                    2 * c.n_heads * head_dim + c.qk_rope_dim, "fp8")
                    * c.swa_window))
            else:
                swa_per_layer = (2 * c.n_heads * head_dim + c.qk_rope_dim) \
                    * c.swa_window * dtype_bytes
        n_full = c.n_full_layers
        n_lin = c.n_layers - n_full
        # 统计真正存 KV 的全注意力层（非 Reuse），并按该层自己的行数计费
        n_store = 0
        mla_bytes = 0
        for layer in self.layers:
            if not getattr(layer, "full_attn", False):
                continue
            if getattr(layer.attn, "csa_reuse", False):
                continue
            n_store += 1
            if getattr(layer.attn, "block_cache", False):
                m = max(1, int(getattr(layer.attn, "compress_ratio", 1)))
                # 块行 ceil(L/m) + 定长 m 行暂存前缀（残块里逐 token 暂存）
                rows = m + (seq_len + m - 1) // m
            else:
                rows = seq_len
            mla_bytes += row_bytes * rows
        if n_store == 0:
            n_store = n_full
            mla_bytes = row_bytes * seq_len
        return (mla_bytes + n_lin * lin_per_layer + n_full * swa_per_layer)
