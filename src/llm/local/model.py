"""完整迷你模型 — embedding + Hybrid 层叠 + MTP 头 + 生成循环."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.block import HybridBlock
from src.llm.local.config import SmallLLMConfig
from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.mla import RMSNorm


def build_block(config: SmallLLMConfig, layer_idx: int) -> HybridBlock:
    """按配置构建单个 Hybrid 块（模型与迁移工具共用，避免参数漂移）."""
    use_memory = (config.memory_every > 0
                  and layer_idx % config.memory_every == 0)
    use_retro = (config.retro_enabled and config.retro_every > 0
                 and layer_idx % config.retro_every == 0)
    use_hyper = config.hyper_streams >= 2
    return HybridBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        full_attn=config.is_full_attn_layer(layer_idx),
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
        dropout=config.dropout,
        sparse_threshold=config.sparse_threshold,
        sparse_window=config.sparse_window,
        sparse_sink=config.sparse_sink,
        sparse_stride=config.sparse_stride,
        sparse_chunk=config.sparse_chunk,
        linear_chunk=config.linear_chunk,
        use_memory=use_memory,
        memory_slots=config.memory_slots,
        memory_topk=config.memory_topk,
        use_retro=use_retro,
        retro_heads=config.retro_heads,
        hyper_streams=config.hyper_streams if use_hyper else 0,
    )


class MTPHead(nn.Module):
    """MTP 多 token 预测头（DeepSeek 系训练加速技巧，推理时关闭）.

    用主干末 hidden 预测 t+2 的 token：一个浅层块 + 复用词表投影。
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.depth = config.mtp_depth
        self.norm = RMSNorm(config.d_model)
        # 浅层变换：让 MTP 头有独立于主 head 的表达空间
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """输入主干 hidden，输出 MTP 用的 hidden（logits 由主模型词表头复用）."""
        return self.proj(self.norm(h))


class TinyLLM(nn.Module):
    """可训练+可推理的迷你 LLM."""

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            build_block(config, i)
            for i in range(config.n_layers)
        ])
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
                config.dropout)
        self._init_weights()

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
            torch.nn.init.normal_(layer.attn.w_o.weight, std=residual_std)
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

        loss = 主 CE(t+1) + mtp_weight * MTP CE(t+2) + 各层 aux_loss 之和。
        return_token_losses=True 时附带逐 token 主 loss（RHO 选择用，
        ignore 位为 0 且反向无梯度，自然不会被选中）。
        mem/mem_mask: v1 单点检索记忆 (b, M, d)/(b, M)，仅 retro_enabled
        且 retro_every=0 时生效；retro 开启但 mem 为空（评测/生成路径）
        则跳过融合，测 backbone 本体。
        chunk_ids/chunk_mask: v2 交错融合的 (b, K, L)/(b, K, L) 文档 token，
        仅 retro_every>0 的层生效；模型侧 frozen embedding 查表编码
        （no_grad，不吃梯度；autocast 下自动同精度）。
        """
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"输入长度 {input_ids.shape[1]} 超过 max_seq_len={self.config.max_seq_len}"
                "（RoPE 缓存盖不住）：请调大 max_seq_len 或直接用 longctx_config()（200K 预设）"
            )
        h = self.embed(input_ids)
        # 视觉前缀（结构预留，默认 None）：视觉 token 已投影到 d_model，拼在文本前
        if vision_embeds is not None:
            if targets is not None:
                raise NotImplementedError("视觉前缀暂只支持推理路径（loss 对齐待接线）")
            h = torch.cat([vision_embeds.to(h.dtype), h], dim=1)
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
        for layer in self.layers:
            # 梯度检查点只在训练时生效（推理/生成走正常路径，保留解码缓存）；
            # 训练不返回解码缓存（return_state=False），省状态循环与两次投影
            if self.config.grad_ckpt and self.training:
                h, aux_total = self._layer_ckpt(layer, h, aux_total, retro_mem)
            else:
                h, (_, aux) = layer(h, None, False, retro_mem)
                aux_total = aux_total + aux
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
        if targets is None:
            return out
        # 主 loss：预测下一 token
        ce_tokens = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.config.vocab_size),
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
        # MTP loss：用位置 i 的 hidden 预测 i+2 的 token
        if self.mtp is not None and targets.shape[1] > 2:
            mtp_h = self.mtp(h[:, :-2])
            mtp_logits = self.lm_head(mtp_h)
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, self.config.vocab_size),
                targets[:, 2:].reshape(-1),
            )
            loss = loss + self.config.mtp_loss_weight * mtp_loss
            out["mtp_loss"] = mtp_loss.detach()
        loss = loss + aux_total
        out["loss"] = loss
        return out

    def _has_interleaved_retro(self) -> bool:
        """是否有 V2 交错融合层（retro_enabled 且 retro_every>0）."""
        return bool(self.config.retro_enabled and self.config.retro_every > 0)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单层梯度检查点：重算换显存（训练 past=None，不用解码缓存）."""
        from torch.utils.checkpoint import checkpoint

        def fn(hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out, (_, aux) = layer(hh, None, False, retro_mem)
            return out, aux

        h_new, aux = checkpoint(fn, h, use_reentrant=False)
        return h_new, aux_total + aux

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
        """
        h = self.embed(input_ids)
        reserve = input_ids.shape[1] + max(0, max_new_tokens)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            h = h.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        pasts: list = []
        for layer in self.layers:
            h, (p, _) = layer(h, retro_mem=retro_mem, reserve=reserve)
            pasts.append(p)
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

    def _decode_step(
        self, nxt: torch.Tensor, pasts: list,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list]:
        """单 token 步进各层：返回 (新 hidden, 新缓存），供流式/批量生成共用."""
        hh = self.embed(nxt)
        hyper_n = self.config.hyper_streams if self.config.hyper_streams >= 2 else 0
        if hyper_n:
            hh = hh.unsqueeze(2).expand(-1, -1, hyper_n, -1)
        new_pasts: list = []
        for layer, p in zip(self.layers, pasts):
            # 每层 norm 由 block 内部处理，这里直接走 attn+moe 等价路径：
            # 为复用逻辑，重新拼 block 前向（单 token 开销可忽略）
            xxs = hh
            hh_in = layer.hyper.pre_mix(hh) if hyper_n and layer.hyper is not None else hh
            hh_norm = layer.norm1(hh_in)
            a_out, p2 = layer.attn(hh_norm, p)  # type: ignore[arg-type]
            hh = hh_in + a_out
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
                hh = layer.hyper.combine(xxs, hh - hh_in)
            new_pasts.append(p2)
        if hyper_n:
            hh = hh.mean(dim=2)
        return self.final_norm(hh), new_pasts

    @torch.no_grad()
    def _decode_multi(
        self, tokens: torch.Tensor, pasts: list,
        retro_mem: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list]:
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
        for li, (layer, p) in enumerate(zip(self.layers, pasts)):
            xxs = hh
            hh_in = layer.hyper.pre_mix(hh) if hyper_n and layer.hyper is not None else hh
            hh_norm = layer.norm1(hh_in)
            a_out, p2 = layer.attn(hh_norm, p)  # type: ignore[arg-type]
            if isinstance(layer.attn, GatedDeltaLite):
                lin_steps[li] = layer.attn.multi_steps
            hh = hh_in + a_out
            m_out, _ = layer.moe(layer.norm2(hh))
            hh = hh + m_out
            if layer.memory is not None:
                hh = layer.memory(hh)
            if layer.retro is not None and retro_mem is not None:
                mh, mm = retro_mem
                hh = layer.retro(hh, mh, mm)
            if hyper_n and layer.hyper is not None:
                hh = layer.hyper.combine(xxs, hh - hh_in)
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
                        buf_c, buf_kr, pos = old_p
                        plist.append((buf_c, buf_kr, pos + i + 1))
                step_pasts.append(plist)
        return self.final_norm(hh), new_pasts, step_pasts

    def _spec_eligible(self, input_ids: torch.Tensor, temperature: float) -> bool:
        """是否走投机解码：需开关开、batch=1、纯贪心（temperature=0）."""
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
        from src.llm.local.spec_decode import build_draft_provider

        cfg = self.config
        provider = build_draft_provider(self, cfg.draft_source, cfg.draft_ngram_min)
        window = max(1, int(cfg.draft_window))
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
            drafts = provider.propose(ctx, h[:, -1:], k) if k > 0 else []
            feed = [t0] + list(drafts)
            y = torch.tensor([feed], dtype=torch.long, device=dev)
            h_multi, _, step_pasts = self._decode_multi(y, pasts)
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
                            "emitted": n_new, "window": window}

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
            cur = torch.cat([cur, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            h, pasts = self._decode_step(nxt, pasts)
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
                yield int(nxt[0, 0])
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
                seen.append(nxt)
                h, pasts = self._decode_step(nxt, pasts)
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
        for layer in self.layers:
            h, _ = layer(h, None, False)
        if hyper_n:
            h = h.mean(dim=2)
        return self.final_norm(h)

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
        active = (
            self.embed.weight.numel()
            + per_layer_active * c.n_layers
            + mem_active + retro_active + hyper_active
            + self.final_norm.weight.numel()
            + (0 if c.tie_embeddings else self.lm_head.weight.numel())
        )
        return {"total": total, "active": active,
                "total_m": round(total / 1e6, 1), "active_m": round(active / 1e6, 1)}

    def kv_cache_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """满上下文 KV/状态缓存字节数（MLA 层 latent + 线性层状态）."""
        c = self.config
        mla_per_layer = (c.kv_lora_rank + c.qk_rope_dim) * seq_len * dtype_bytes
        lin_per_layer = c.n_heads * (c.d_model // c.n_heads) ** 2 * dtype_bytes
        n_full = c.n_full_layers
        n_lin = c.n_layers - n_full
        return n_full * mla_per_layer + n_lin * lin_per_layer
