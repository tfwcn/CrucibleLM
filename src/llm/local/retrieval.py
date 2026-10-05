"""BM25 检索 — 零依赖中文检索（字符级，免分词）.

用途：决策时从 200K 会话历史里捞 top-K 相关片段（2~4K），
直觉头只看"最近 + 捞回"，不啃全量。无外部依赖，CPU 微秒级。
"""

from __future__ import annotations

import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """字符级分词（中文免切词，空白丢弃）."""
    return [ch for ch in text if not ch.isspace()]


class BM25Retriever:
    """BM25（k1/b 标准参数），add 建库，query 取 top-K."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: dict[str, Counter] = {}
        self.lens: dict[str, int] = {}
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, doc_id: str, text: str) -> None:
        """入库（同 id 覆盖）."""
        if doc_id in self.docs:
            for tok in self.docs[doc_id]:
                self.df[tok] -= 1
        toks = tokenize(text)
        tf = Counter(toks)
        self.docs[doc_id] = tf
        self.lens[doc_id] = len(toks)
        for tok in tf:
            self.df[tok] += 1
        self.avgdl = sum(self.lens.values()) / max(len(self.lens), 1)

    def _idf(self, tok: str) -> float:
        """平滑 IDF（未见词给下限分，不直接归零）."""
        n = len(self.docs)
        df = self.df.get(tok, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """取 top-K (doc_id, 分数)，空库/空查询返回空."""
        qtf = Counter(tokenize(text))
        if not qtf or not self.docs:
            return []
        scores: dict[str, float] = {}
        for tok, qf in qtf.items():
            idf = self._idf(tok)
            for doc_id, tf in self.docs.items():
                f = tf.get(tok, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lens[doc_id] / max(self.avgdl, 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def save(self, path: str) -> None:
        """整体 pickle 落盘（文档原文 + 索引，可直接载入）."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """从 pickle 载回（与 save 配套）."""
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 不是 BM25Retriever")
        return obj
