"""本地小 LLM — 参考 DeepSeek V4 系 + Qwen3-Next 系技术的 16G 显存迷你模型

技术来源（思想借鉴，自研简化实现）：
- DeepSeek 系：MLA 低秩 KV 压缩 + 解耦 RoPE、细粒度 MoE（多小专家 + 共享专家）、MTP 多 token 预测
- Qwen3-Next 系：Hybrid 注意力（全注意力层与 Gated-DeltaNet-lite 线性层交替）、QK-Norm、短卷积、YaRN 长上下文外推

组成：
- config.py      超参数（默认 ~110M 总参数 / ~40M 激活，16G 可从零训练）
- rope.py        RoPE + YaRN 缩放
- mla.py         MLA 全注意力（含 latent KV 缓存，省显存的关键）
- linear_attn.py Gated-DeltaNet-lite 线性注意力（O(N) 推理，Qwen3-Next 式混合）
- moe.py         细粒度 MoE + 共享专家 + 负载均衡 aux loss
- block.py       Hybrid Transformer 块
- model.py       完整模型（含 MTP 头、前向/生成/参数统计）
- train.py       训练单步、16G 配方、显存估算
- infer.py       简单分词器 + chat 封装（可对接现有 Runner 调试）
"""

from src.llm.local.config import SmallLLMConfig
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
from src.llm.local.model import TinyLLM

__all__ = ["SmallLLMConfig", "TinyLLM", "LocalChatBackend", "SimpleTokenizer"]
