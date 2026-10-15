"""投机解码起草器 — 前缀 n-gram 查表 + MTP 头，零训练、默认关.

职责边界：起草器**只负责猜**接下来的 token，正确性由主干的并行验证兜底
（猜错即退回，最终序列与逐 token 贪心一致，契约由
`tests/test_speculative.py` 锁定）。起草本身不读不写模型权重（MTP 头只读），
因此对训练完全无副作用。

三种来源：
- `ngram`：前缀查表（prompt-lookup），重复文本命中率极高且零成本；
- `mtp`：复用已训好的 MTP 头猜 t+2（只多看一步，故只给 1 个候选）；
- `hybrid`：MTP 起头 + ngram 补足（默认）。
"""

from __future__ import annotations


class DraftProvider:
    """起草器接口：给已提交上下文与末位 hidden，返回最多 k 个候选 token."""

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """返回候选 token 列表（长度 ≤ k；无把握就返回空，主干退化为普通贪心）."""
        raise NotImplementedError


class NgramDraftProvider(DraftProvider):
    """前缀 n-gram 查表：找上下文末段此前出现过的地方，抄它后面的 token.

    经典 prompt-lookup 思路：模板、代码、复读等重复文本命中率极高且零成本。
    min_match 太短会误命中（"的的的…"），太长覆盖不足，默认 2。
    scan_limit 限定回看窗口，避免长上下文里每轮 O(N) 全扫。
    """

    def __init__(self, min_match: int = 2, scan_limit: int = 1024):
        self.min_match = max(1, int(min_match))
        self.scan_limit = max(16, int(scan_limit))

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """从最长可能后缀往短找，命中即抄后续 k 个（不足则截断）."""
        if k <= 0:
            return []
        ctx = context[-(self.scan_limit + k):]
        n = len(ctx)
        if n < self.min_match + 1:
            return []
        # 后缀最长可能长度：受限于"后面还得有 k 个可抄"和上下文总长
        hi = min(n - 1, self.min_match + k - 1)
        for m in range(hi, self.min_match - 1, -1):
            suffix = ctx[n - m:]
            # 从后往前找上一次出现（取最近的一次，续写更连贯）
            for j in range(n - m - 1, -1, -1):
                if ctx[j:j + m] == suffix:
                    out = ctx[j + m:j + m + k]
                    if out:
                        return list(out[:k])
        return []


class MtpDraftProvider(DraftProvider):
    """MTP 头起草：末位 hidden 经 MTP 头 + 词表头，猜 t+2 的 token.

    MTP 只多看一步（`mtp(h_T)` 预测 T+2），无法链式预测后续位，故只给 1 个候选。
    """

    def __init__(self, model):
        self.model = model

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """返回 1 个候选（模型无 MTP 头或 k<=0 时返回空）."""
        if k <= 0 or getattr(self.model, "mtp", None) is None:
            return []
        import torch

        with torch.no_grad():
            mtp_h = self.model.mtp(h_last)
            logits = self.model.lm_head(mtp_h)[:, -1, :]
        return [int(logits.argmax(-1)[0])]


class HybridDraftProvider(DraftProvider):
    """MTP 起头 + ngram 补足：MTP 给第 1 位，ngram 顺延补足剩余位.

    第 1 位用 MTP（它天然就是 t+2 的预测）；之后把 MTP 猜测并入上下文，
    用 ngram 继续抄，命中则一次凑满窗口。
    """

    def __init__(self, model, min_match: int = 2, scan_limit: int = 1024):
        self.mtp = MtpDraftProvider(model)
        self.ngram = NgramDraftProvider(min_match, scan_limit)

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """MTP 首猜，随后用 ngram 在"上下文 + 已猜"上续抄."""
        if k <= 0:
            return []
        drafts = self.mtp.propose(context, h_last, 1)
        if len(drafts) < k:
            merged = context + drafts
            drafts += self.ngram.propose(merged, h_last, k - len(drafts))
        return drafts[:k]


def build_draft_provider(model, source: str, min_match: int = 2) -> DraftProvider:
    """按 `draft_source` 配置构建起草器（未知来源回落 ngram，不报错崩生成）."""
    src = (source or "ngram").lower()
    if src == "mtp":
        return MtpDraftProvider(model)
    if src == "hybrid":
        return HybridDraftProvider(model, min_match)
    return NgramDraftProvider(min_match)
