"""RETRO-lite — 检索融合：小逻辑模型 + 外部知识库（BM25 现成）.

完整 RETRO 需要双向编码器 + 交错 cross-attention，成本高；
v1 只做最薄的一层：final_norm 之后、lm_head 之前，加一个
cross-attention 融合块，memory = 检索文本的 embedding 均值。
信号弱但管线全通（检索→编码→融合→loss），质量验证靠 eval harness；
v2 升级项：frozen 编码器 / kNN-LM 式插值（见 README）。

默认关闭（config.retro_enabled=False 时模块不存在，state_dict 兼容）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.local.mla import RMSNorm


class RetroFusion(nn.Module):
    """单点 cross-attention 融合：h  attend 检索记忆，残差加回.

    输出投影零初始化时恒等（h 进 h 出），新开模块不破坏已训权重。
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 须整除 n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = RMSNorm(d_model)
        self.norm_m = RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        with torch.no_grad():
            self.w_o.weight.zero_()  # 恒等起点：开即无感，训练再长

    def forward(self, h: torch.Tensor, mem: torch.Tensor,
                mem_mask: torch.Tensor | None = None) -> torch.Tensor:
        """h: (b, T, d) 主干 hidden；mem: (b, M, d) 检索记忆（已编码）.

        mem_mask: (b, M) bool，True=有效（padding 补齐时用）。
        """
        b, t, d = h.shape
        m = mem.shape[1]
        if m == 0:
            return h  # 无检索命中时恒等（空库/全过滤）
        q = self.w_q(self.norm_q(h)).view(b, t, self.n_heads, self.head_dim)
        k = self.w_k(self.norm_m(mem)).view(b, m, self.n_heads, self.head_dim)
        v = self.w_v(self.norm_m(mem)).view(b, m, self.n_heads, self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        if mem_mask is not None:
            scores = scores.masked_fill(
                ~mem_mask.view(b, 1, 1, m), float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        if self.training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        o = torch.einsum("bhts,bshd->bthd", probs, v).reshape(b, t, d)
        return h + self.w_o(o)


def encode_memories_mean(
    embed: nn.Embedding, tokenizer, texts: list[str], max_len: int = 256,
) -> torch.Tensor:
    """检索文本编码 v1：embedding 均值（零成本；v2 换 frozen 编码器）.

    返回 (len(texts), d) CPU 张量；空列表返回 (0, d)。
    """
    vecs: list[torch.Tensor] = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_bos=False)[:max_len]
            if not ids:
                continue
            e = embed(torch.tensor([ids], dtype=torch.long))
            vecs.append(e.mean(dim=1).squeeze(0).cpu())
    if not vecs:
        d = embed.weight.shape[1]
        return torch.zeros(0, d)
    return torch.stack(vecs, dim=0)


def retrieve_for_texts(
    retriever, texts: list[str], k: int = 2, max_chars: int = 2000,
) -> list[list[str]]:
    """每段输入捞 top-K 文档文本（截断防爆），返回与 texts 对齐的列表."""
    out: list[list[str]] = []
    for text in texts:
        hits = retriever.query(text[-max_chars:], k=k)
        out.append([doc for doc, _ in hits])
    return out


def build_batch_mem(
    embed: nn.Embedding,
    tokenizer,
    hits: list[list[str]],
    k: int,
    max_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(b, M, d) 的记忆张量 + (b, M) 有效掩码（不足 k 个命中时补零向量）.

    补零向量在掩码里标 True，避免 softmax 全 -inf 产生 NaN（零向量经
    RMSNorm 后输出零，softmax 权重均匀但 value 全零，融合增量恒为零）。
    """
    d = embed.weight.shape[1]
    rows: list[torch.Tensor] = []
    out_mask: list[list[bool]] = []
    for doc_list in hits:
        docs = doc_list[:k]
        vecs = encode_memories_mean(embed, tokenizer, docs, max_len) if docs \
            else torch.zeros(0, d)
        m = vecs.shape[0]
        if m < k:
            vecs = torch.cat([vecs, torch.zeros(k - m, d)], dim=0)
        rows.append(vecs)
        out_mask.append([True] * k)
    return torch.stack(rows, dim=0), torch.tensor(out_mask, dtype=torch.bool)
