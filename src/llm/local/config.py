"""超参数 — 默认配置约 128M 总参数 / 约 48M 激活参数，16G 显存可从零训练."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmallLLMConfig:
    """迷你模型超参数."""

    # 词表大小（小词表是 100M 量级可行的关键）
    vocab_size: int = 8192
    # 模型 hidden 维度
    d_model: int = 768
    # Transformer 层数
    n_layers: int = 12
    # 每隔几层放一个 MLA 全注意力层，其余为线性注意力层（Qwen3-Next 式混合）
    full_attn_every: int = 4
    # 注意力头数（head_dim = d_model // n_heads = 64）
    n_heads: int = 12
    # MLA：Q 下投影秩
    q_lora_rank: int = 192
    # MLA：KV 联合压缩秩（远小于 n_heads*head_dim，省 KV 缓存的关键）
    kv_lora_rank: int = 128
    # MLA：解耦 RoPE 维度（Q/K 共享同一份旋转）
    qk_rope_dim: int = 32
    # MoE 专家总数（细粒度小专家）
    n_experts: int = 16
    # 每 token 激活专家数
    top_k: int = 4
    # 单个专家中间维度（小专家，控制总参数量）
    expert_hidden: int = 192
    # 共享专家个数（常驻知识通道）
    n_shared: int = 1
    # 负载均衡 aux loss 系数
    aux_loss_coef: float = 0.01
    # MTP 额外预测步数（1 表示除 t+1 外再预测 t+2）
    mtp_depth: int = 1
    # MTP loss 权重
    mtp_loss_weight: float = 0.3
    # 最大上下文长度
    max_seq_len: int = 8192
    # RoPE 基频
    rope_theta: float = 10000.0
    # YaRN 外推缩放因子（>1 时扩展有效上下文）
    yarn_scale: float = 2.0
    # dropout
    dropout: float = 0.0
    # embedding 与输出头是否共享权重（省约 6M 参数）
    tie_embeddings: bool = True
    # 梯度检查点：逐层重算激活，seq2048+ 上下文跑进 16G 的关键（训练时开启）
    grad_ckpt: bool = False
    # 轻量稀疏注意力（DSA 思想简化版）：超阈值自动切 sink+窗口+跨步，200K 上下文用
    sparse_threshold: int = 4096  # 短于此走 dense，与 MLA 数学一致
    sparse_window: int = 4096  # 滑动窗口（每个查询回看这么多）
    sparse_sink: int = 128  # 注意力汇点（打头保留）
    sparse_stride: int = 512  # 膨胀跨步（窗口外每隔这么多保留一个）
    sparse_chunk: int = 2048  # 预填充分块（显存/速度折中）
    # 线性注意力分块阈值：超长时切块递推（前向精确，反向块内截断），线性层 O(N^2) 的解药
    linear_chunk: int = 2048

    @property
    def head_dim(self) -> int:
        """每个头的维度."""
        return self.d_model // self.n_heads

    def is_full_attn_layer(self, layer_idx: int) -> bool:
        """判断某层是否为 MLA 全注意力层（其余为线性注意力层）."""
        return layer_idx % self.full_attn_every == 0

    @property
    def n_full_layers(self) -> int:
        """全注意力层数量（决定 KV 缓存大小）."""
        return sum(1 for i in range(self.n_layers) if self.is_full_attn_layer(i))


# 16G 训练配方建议（bf16 + AdamW + 梯度检查点）
RECIPE_16G: dict = {
    "precision": "bf16",
    "optimizer": "AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)",
    "seq_len": 2048,
    "micro_batch": 4,
    "grad_accum": 8,
    "effective_tokens_per_step": 2048 * 4 * 8,
    "grad_checkpointing": True,
    "notes": "权重 bf16 约 0.26GB；Adam 状态约 1GB；开梯度检查点后 "
    "seq2048/batch4 实测峰值约 5GB（RTX 3080 Laptop 16G），余量充足；"
    "不开检查点线性层 O(N^2) 矩阵会爆显存。",
}


# CPU 验证用的极小配置（单测/无卡环境跑通前向+生成+单步训练）
def tiny_test_config() -> "SmallLLMConfig":
    """返回 CPU 可跑的极小配置."""
    return SmallLLMConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        full_attn_every=2,
        n_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_dim=16,
        n_experts=8,
        top_k=2,
        expert_hidden=64,
        n_shared=1,
        mtp_depth=1,
        max_seq_len=256,
        yarn_scale=1.0,
    )


# 长上下文预设（200K 推理）：短训 + YaRN 外推的标准路线，
# 真要 200K 质量还需阶段性长文微调（32K→128K），此处只给结构就绪
def longctx_config() -> "SmallLLMConfig":
    """返回 200K 上下文配置（推理就绪；KV 缓存约 200MB，线性层常数状态）."""
    return SmallLLMConfig(
        max_seq_len=204800,
        yarn_scale=8.0,
        sparse_threshold=4096,
        sparse_window=4096,
        sparse_sink=128,
        sparse_stride=512,
        sparse_chunk=2048,
    )
