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
        self.postings: dict[str, list[str]] = {}  # 倒排：token -> 含它的 doc 列表
        self.avgdl = 0.0

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, doc_id: str, text: str) -> None:
        """入库（同 id 覆盖，倒排同步更新）."""
        if doc_id in self.docs:
            for tok in self.docs[doc_id]:
                self.df[tok] -= 1
                lst = self.postings.get(tok)
                if lst is not None and doc_id in lst:
                    lst.remove(doc_id)
        toks = tokenize(text)
        tf = Counter(toks)
        self.docs[doc_id] = tf
        self.lens[doc_id] = len(toks)
        for tok in tf:
            self.df[tok] += 1
            self.postings.setdefault(tok, []).append(doc_id)
        self.avgdl = sum(self.lens.values()) / max(len(self.lens), 1)

    def _idf(self, tok: str) -> float:
        """平滑 IDF（未见词给下限分，不直接归零）."""
        n = len(self.docs)
        df = self.df.get(tok, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, text: str, k: int = 5, max_terms: int = 64) -> list[tuple[str, float]]:
        """取 top-K (doc_id, 分数)，空库/空查询返回空.

        两处剪枝（30 万文档下单次 5 秒起，训练每步查 batch 次不可接受）：
        1. 只取 idf 最高的 max_terms 个查询词——中文虚词（的/是/在）出现在
           几乎所有文档，带着它们扫全库等于没剪；
        2. 倒排取候选后单遍打分（每文档一次遍历累加各词贡献），
           不再"每词扫一遍候选"。
        老索引无倒排时回落全扫描（慢但对）。
        """
        import heapq

        qtf = Counter(tokenize(text))
        if not qtf or not self.docs:
            return []
        idf = {tok: self._idf(tok) for tok in qtf}
        terms = sorted(qtf, key=lambda t: idf[t], reverse=True)[:max_terms]
        postings = getattr(self, "postings", None)
        if postings:
            cand: set[str] = set()
            for tok in terms:
                lst = postings.get(tok)
                if lst:
                    cand.update(lst)
            if not cand:
                return []
            doc_ids = cand
        else:
            doc_ids = self.docs.keys()  # type: ignore[assignment]
        scored: list[tuple[float, str]] = []
        for doc_id in doc_ids:
            tf = self.docs[doc_id]
            dl = self.lens[doc_id] / max(self.avgdl, 1)
            s = 0.0
            for tok in terms:
                f = tf.get(tok, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl)
                s += idf[tok] * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, doc_id))
        return [(doc_id, s) for s, doc_id in heapq.nlargest(k, scored)]

    def save(self, path: str) -> None:
        """整体 pickle 落盘（文档原文 + 索引，可直接载入）."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """从 pickle 载回（与 save 配套；老索引无倒排时就地重建）."""
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 不是 BM25Retriever")
        if not getattr(obj, "postings", None) and obj.docs:
            obj.postings = {}
            for doc_id, tf in obj.docs.items():
                for tok in tf:
                    obj.postings.setdefault(tok, []).append(doc_id)
        return obj
