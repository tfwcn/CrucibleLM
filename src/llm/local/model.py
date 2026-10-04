"""完整迷你模型 — embedding + Hybrid 层叠 + MTP 头 + 生成循环."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.block import HybridBlock
from src.llm.local.config import SmallLLMConfig
from src.llm.local.mla import RMSNorm


def build_block(config: SmallLLMConfig, layer_idx: int) -> HybridBlock:
    """按配置构建单个 Hybrid 块（模型与迁移工具共用，避免参数漂移）."""
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
    ) -> dict:
        """前向：训练时传 targets 返回组合 loss，推理时只返回 logits.

        loss = 主 CE(t+1) + mtp_weight * MTP CE(t+2) + 各层 aux_loss 之和。
        return_token_losses=True 时附带逐 token 主 loss（RHO 选择用，
        ignore 位为 0 且反向无梯度，自然不会被选中）。
        """
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"输入长度 {input_ids.shape[1]} 超过 max_seq_len={self.config.max_seq_len}"
                "（RoPE 缓存盖不住）：请调大 max_seq_len 或直接用 longctx_config()（200K 预设）"
            )
        h = self.embed(input_ids)
        aux_total = torch.zeros((), device=h.device, dtype=h.dtype)
        for layer in self.layers:
            # 梯度检查点只在训练时生效（推理/生成走正常路径，保留解码缓存）；
            # 训练不返回解码缓存（return_state=False），省状态循环与两次投影
            if self.config.grad_ckpt and self.training:
                h, aux_total = self._layer_ckpt(layer, h, aux_total)
            else:
                h, (_, aux) = layer(h, None, False)
                aux_total = aux_total + aux
        h = self.final_norm(h)
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

    def _layer_ckpt(
        self, layer: HybridBlock, h: torch.Tensor, aux_total: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单层梯度检查点：重算换显存（训练 past=None，不用解码缓存）."""
        from torch.utils.checkpoint import checkpoint

        def fn(hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out, (_, aux) = layer(hh, None, False)
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
                input_ids, max_new_tokens, temperature, top_k, eos_id)
        finally:
            if was_training:
                self.train()

    def _prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Prefill：整段 prompt 一次前向，返回 (末位置 hidden 前的 h, 每层缓存）.

        h 为全序列 hidden（调用方取 h[:, -1:] 算 logits）；pasts 供单步解码续跑。
        """
        h = self.embed(input_ids)
        pasts: list = []
        for layer in self.layers:
            h, (p, _) = layer(h)
            pasts.append(p)
        return self.final_norm(h), pasts

    def _sample_next(
        self,
        h_last: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """按温度/top-k 采样下一步（贪心 temperature=0），返回 (b, 1) id."""
        next_logits = self.lm_head(h_last)[:, -1, :]
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
        self, nxt: torch.Tensor, pasts: list
    ) -> tuple[torch.Tensor, list]:
        """单 token 步进各层：返回 (新 hidden, 新缓存），供流式/批量生成共用."""
        hh = self.embed(nxt)
        new_pasts: list = []
        for layer, p in zip(self.layers, pasts):
            # 每层 norm 由 block 内部处理，这里直接走 attn+moe 等价路径：
            # 为复用逻辑，重新拼 block 前向（单 token 开销可忽略）
            hh_norm = layer.norm1(hh)
            a_out, p2 = layer.attn(hh_norm, p)  # type: ignore[arg-type]
            hh = hh + a_out
            m_out, _ = layer.moe(layer.norm2(hh))
            hh = hh + m_out
            new_pasts.append(p2)
        return self.final_norm(hh), new_pasts

    @torch.no_grad()
    def _generate_inner(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """自回归生成内循环（调用方 generate 已处理 eval 模式切换）.

        temperature=0 为贪心；>0 时按温度采样（可配 top-k 截断）。
        与 stream_tokens 同一基元（数学一致，单测锁定）。
        """
        h, pasts = self._prefill(input_ids)
        cur = input_ids
        for _ in range(max_new_tokens):
            nxt = self._sample_next(h[:, -1:], temperature, top_k)
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
    ):
        """逐 token 生成器（SSE 流式/OpenAI stream 用），逐个 yield 新 id（int）.

        与 generate() 同一 prefill+解码路径（数学一致，单测锁定），
        生成后同样恢复 train/eval 状态。
        """
        was_training = self.training
        self.eval()
        try:
            h, pasts = self._prefill(input_ids)
            for _ in range(max_new_tokens):
                nxt = self._sample_next(h[:, -1:], temperature, top_k)
                yield int(nxt[0, 0])
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
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
        for layer in self.layers:
            h, _ = layer(h, None, False)
        return self.final_norm(h)

    @torch.no_grad()
    def encode_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取末位置 hidden（输入定长/单条时用；变长 batch 请用 full+gather）.

        不碰 train/eval 状态（只关梯度；dropout>0 时调用方自行 eval）。
        """
        return self.encode_full_hidden(input_ids)[:, -1, :]

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
        active = (
            self.embed.weight.numel()
            + per_layer_active * c.n_layers
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
