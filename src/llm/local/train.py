"""训练工具 — 单步训练、显存估算、16G 配方."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.llm.local.config import RECIPE_16G, SmallLLMConfig
from src.llm.local.model import TinyLLM


def estimate_memory_gb(
    config: SmallLLMConfig,
    seq_len: int = 2048,
    micro_batch: int = 4,
    dtype_bytes: int = 2,
    grad_checkpointing: bool = True,
    optimizer: str = "adamw",
) -> dict:
    """估算训练显存（bf16，便于配 16G 预算）.

    组成：权重 + 梯度 + Adam 双状态 + 激活。
    激活逐项累加：线性层 O(N^2) 并行矩阵（scores+衰减矩阵）、
    MoE 逐专家物化输出、其余残差/qkv/norm。
    开梯度检查点后只保留每层输入 + 反向时一层的瞬时重算峰值。
    """
    # 用解析式重算总参数（不实例化模型也能估）
    per_expert = 3 * config.d_model * config.expert_hidden
    attn_per_layer = (
        config.d_model * config.q_lora_rank  # w_dq
        + config.q_lora_rank * config.n_heads * (config.d_model // config.n_heads)  # w_uq
        + config.q_lora_rank * config.n_heads * config.qk_rope_dim  # w_qr
        + config.d_model * config.kv_lora_rank  # w_dkv
        + config.kv_lora_rank * config.n_heads * (config.d_model // config.n_heads) * 2  # uk+uv
        + config.d_model * config.qk_rope_dim  # w_kr
        + config.d_model * config.d_model  # w_o
    )
    moe_per_layer = (
        config.d_model * config.n_experts  # router
        + (config.n_experts + config.n_shared) * per_expert
    )
    total = (
        config.vocab_size * config.d_model  # embedding（绑定时 head 不另计）
        + config.n_layers * (attn_per_layer + moe_per_layer)
    )
    weights_gb = total * dtype_bytes / 1e9
    grads_gb = weights_gb  # 梯度与权重同尺寸
    if optimizer == "muon":
        # Muon 单份 fp32 动量（2D 参数约 95%）+ AdamW 双份（其余约 5%）
        adam_gb = (total * 0.95 * 4 + total * 0.05 * 8) / 1e9
    else:
        adam_gb = total * 4 * 2 / 1e9  # m+v 两份 fp32 状态
    # 激活（逐项累加，单位 GB）
    b, t, d = micro_batch, seq_len, config.d_model
    n_full = config.n_full_layers
    n_lin = config.n_layers - n_full
    # 线性层 O(N^2) 并行项：超 linear_chunk 切块递推，峰值按块算
    eff_t = min(t, config.linear_chunk)
    quad_gb = 2 * b * config.n_heads * eff_t * eff_t * dtype_bytes / 1e9 * n_lin
    # MLA 层：短序列走 flash dense（不物化），超阈值走分块聚集，
    # 瞬时峰 = chunk 查询 × 聚集 key（sink+窗口+T/stride+chunk）
    if t > config.sparse_threshold:
        sel = (config.sparse_sink + config.sparse_window
               + t // config.sparse_stride + config.sparse_chunk)
        mla_gb = (b * config.n_heads * config.sparse_chunk * sel
                  * dtype_bytes / 1e9 * n_full)
    else:
        mla_gb = 0.0
    # MoE 逐专家物化：(n_experts + n_shared) 块 (b, t, d)
    moe_gb = ((config.n_experts + config.n_shared) * b * t * d
              * dtype_bytes / 1e9 * config.n_layers)
    # 其余残差/qkv/norm：每层约 6 块 (b, t, d)
    base_gb = 6 * b * t * d * dtype_bytes / 1e9 * config.n_layers
    if grad_checkpointing:
        # 检查点：每层只保留输入 + 反向时一层的瞬时重算峰值
        # （线性层已按 linear_chunk 切块，MLA 超阈值按 mla_gb 聚集峰值）
        acts_gb = (
            config.n_layers * b * t * d * dtype_bytes / 1e9
            + 2 * b * config.n_heads * eff_t * eff_t * dtype_bytes / 1e9
            + (config.n_experts + config.n_shared) * b * t * d * dtype_bytes / 1e9
            + 6 * b * t * d * dtype_bytes / 1e9
            + mla_gb
        )
    else:
        acts_gb = quad_gb + moe_gb + base_gb + mla_gb
    total_gb = weights_gb + grads_gb + adam_gb + acts_gb
    return {
        "total_params_m": round(total / 1e6, 1),
        "weights_gb": round(weights_gb, 2),
        "grads_gb": round(grads_gb, 2),
        "adam_gb": round(adam_gb, 2),
        "activations_gb": round(acts_gb, 2),
        "total_gb": round(total_gb, 2),
        "fits_16g": total_gb < 15.0,
    }


def select_topk_loss(
    token_losses: torch.Tensor,
    keep_ratio: float,
    min_keep: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RHO 选择：只保留 loss 最高的 keep_ratio 部分（ignore 位 loss 为 0，自然落选）.

    返回 (选中均值, bool 掩码)；掩码基于 detach 阈值，不可微（标准做法）；
    选中数保底 min_keep，防空选 NaN。
    """
    flat = token_losses.reshape(-1)
    n = flat.numel()
    k = max(min(int(n * keep_ratio), n), min(min_keep, n))
    if k >= n:
        return flat.mean(), torch.ones_like(flat, dtype=torch.bool)
    with torch.no_grad():
        thr = torch.topk(flat.detach(), k).values.min()
    mask = flat >= thr
    # topk 边界并列可能多选，无妨（均值口径一致即可）
    return (flat * mask).sum() / mask.sum().clamp_min(1), mask


def train_step(
    model: TinyLLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    labels: torch.Tensor | None = None,
) -> dict:
    """单步训练：前向（含 MTP+aux）→ 反向 → 裁剪 → 更新.

    input_ids 即 targets（自回归，内部自动错位一位）；
    SFT 时传 labels（含 -100 掩码），input_ids 仅作输入。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = labels if labels is not None else input_ids
    if use_amp and input_ids.is_cuda:
        # bf16 混合精度（16G 训练推荐，可省近半激活显存）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, targets=targets)
            loss = out["loss"]
        loss.backward()
    else:
        out = model(input_ids, targets=targets)
        loss = out["loss"]
        loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "main_loss": float(out.get("main_loss", loss).detach()) if torch.is_tensor(out.get("main_loss", None)) else out.get("main_loss"),
        "mtp_loss": float(out["mtp_loss"].detach()) if torch.is_tensor(out.get("mtp_loss", None)) else out.get("mtp_loss"),
        "aux_loss": float(out["aux_loss"].detach()),
        "grad_norm": float(grad_norm),
    }


def build_optimizer(model: TinyLLM, lr: float = 3e-4) -> torch.optim.Optimizer:
    """AdamW 优化器（DeepSeek/Qwen 系常用超参），CUDA 上用 fused 内核加速."""
    try:
        is_cuda = next(model.parameters()).is_cuda
    except StopIteration:
        is_cuda = False
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
        fused=is_cuda and torch.cuda.is_available(),
    )


def get_16g_recipe() -> dict:
    """返回 16G 训练配方（展示用，训练脚本直接抄这些值）."""
    return dict(RECIPE_16G)
