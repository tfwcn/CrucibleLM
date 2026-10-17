"""投机解码起草器 — 前缀 n-gram 查表 + MTP 头，零训练、默认关.

职责边界：起草器**只负责猜**接下来的 token，正确性由主干的并行验证兜底
（猜错即退回，最终序列与逐 token 贪心一致，契约由
`tests/test_speculative.py` 锁定）。起草本身不读不写模型权重（MTP 头只读），
因此对训练完全无副作用。

三种来源：
- `ngram`：前缀查表（prompt-lookup），重复文本命中率极高且零成本；
- `mtp`：复用已训好的 MTP 头猜 t+2（只多看一步，故只给 1 个候选）；
- `hybrid`：MTP 起头 + ngram 补足（默认）。

**DSpark 置信调度**（`dspark_enabled`）：在起草之上，按每个草稿 token 的存活置信度
动态决定"这轮验多长"（`schedule_verify_length`），提前截断看起来必被拒的后缀，
省目标模型算力。仅 `draft_enabled` 时生效，且不改变最终序列（验证仍精确）。
"""

from __future__ import annotations


def schedule_verify_length(
    confs: list[float],
    threshold: float,
    min_verify: int = 1,
    k: int | None = None,
) -> int:
    """DSpark 置信调度：返回该轮可验证的草稿数（最长"累计存活概率 ≥ 阈值"前缀）.

    DSpark（*DSpark: Confidence-Scheduled Speculative Decoding with
    Semi-Autoregressive Generation*，arXiv 2607.05147）的核心之一：不再对每个
    请求都验证同一固定块，而是按每个草稿 token 的**存活置信度**决定验多长——
    后缀一旦"看起来必被拒"就提前截断，省下目标模型验证废草稿的算力
    （verification waste）。

    这里实现其**置信调度**部分（单请求、无引擎负载感知）：累计存活概率
    `∏ confs[i] ≥ threshold` 的草稿才送去验证。`threshold<=0` 退化为"全验证"
    （= 固定窗口，即默认关时的等价路径）；无论置信多低都至少验
    `min_verify` 个草稿，保证每轮有推进。
    """
    n = len(confs) if k is None else max(0, min(len(confs), k))
    if n <= 0:
        return 0
    if threshold <= 0.0:
        return n
    cum = 1.0
    verify = 0
    for i in range(n):
        c = confs[i]
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        cum *= c
        if cum < threshold:
            break
        verify = i + 1
    return max(verify, min(max(0, int(min_verify)), n))


def scored_propose(provider, context: list[int], h_last, k: int):
    """取 (tokens, confs)：起草器有 `propose_scored` 就用它，否则回退 `propose`.

    回退时给恒定置信度 1.0（等价于"不做调度"，老起草器/测试桩仍可用）。
    """
    fn = getattr(provider, "propose_scored", None)
    if callable(fn):
        toks, confs = fn(context, h_last, k)
        return list(toks), [float(c) for c in confs]
    toks = list(provider.propose(context, h_last, k))
    return toks, [1.0] * len(toks)


class DraftProvider:
    """起草器接口：给已提交上下文与末位 hidden，返回最多 k 个候选 token."""

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """返回候选 token 列表（长度 ≤ k；无把握就返回空，主干退化为普通贪心）."""
        raise NotImplementedError

    def propose_scored(self, context: list[int], h_last, k: int):
        """返回 (候选 token, 各自存活置信度)；基类回退为恒定置信度 1.0."""
        toks = self.propose(context, h_last, k)
        return list(toks), [1.0] * len(toks)


class NgramDraftProvider(DraftProvider):
    """前缀 n-gram 查表：找上下文末段此前出现过的地方，抄它后面的 token.

    经典 prompt-lookup 思路：模板、代码、复读等重复文本命中率极高且零成本。
    min_match 太短会误命中（"的的的…"），太长覆盖不足，默认 2。
    scan_limit 限定回看窗口，避免长上下文里每轮 O(N) 全扫。
    """

    def __init__(self, min_match: int = 2, scan_limit: int = 1024):
        self.min_match = max(1, int(min_match))
        self.scan_limit = max(16, int(scan_limit))

    def _find(self, context: list[int], k: int) -> tuple[list[int], int]:
        """返回 (抄到的 token 列表, 命中的后缀长度 m)；无命中返回 ([], 0)."""
        if k <= 0:
            return [], 0
        ctx = context[-(self.scan_limit + k):]
        n = len(ctx)
        if n < self.min_match + 1:
            return [], 0
        # 后缀最长可能长度：受限于"后面还得有 k 个可抄"和上下文总长
        hi = min(n - 1, self.min_match + k - 1)
        for m in range(hi, self.min_match - 1, -1):
            suffix = ctx[n - m:]
            # 从后往前找上一次出现（取最近的一次，续写更连贯）
            for j in range(n - m - 1, -1, -1):
                if ctx[j:j + m] == suffix:
                    out = ctx[j + m:j + m + k]
                    if out:
                        return list(out[:k]), m
        return [], 0

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """从最长可能后缀往短找，命中即抄后续 k 个（不足则截断）."""
        return self._find(context, k)[0]

    def propose_scored(self, context: list[int], h_last, k: int):
        """带置信度的起草：置信度 = `m/(m+1)`（后缀越长越可信，单调 ∈ (0,1)）."""
        out, m = self._find(context, k)
        if not out:
            return [], []
        conf = m / (m + 1.0)
        return out, [conf] * len(out)


class MtpDraftProvider(DraftProvider):
    """MTP 头起草：末位 hidden 经 MTP 头 + 词表头，猜 t+2 的 token.

    MTP 只多看一步（`mtp(h_T)` 预测 T+2），无法链式预测后续位，故只给 1 个候选。
    """

    def __init__(self, model):
        self.model = model

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """返回 1 个候选（模型无 MTP 头或 k<=0 时返回空）."""
        return self.propose_scored(context, h_last, k)[0]

    def propose_scored(self, context: list[int], h_last, k: int):
        """返回 1 个候选及其置信度（softmax 最大概率，供 DSpark 置信调度）."""
        if k <= 0 or getattr(self.model, "mtp", None) is None:
            return [], []
        import torch

        with torch.no_grad():
            mtp_h = self.model.mtp(h_last)
            logits = self.model.lm_head(mtp_h)[:, -1, :]
            conf = float(torch.softmax(logits.float(), dim=-1).max())
            tok = int(logits.argmax(-1)[0])
        return [tok], [conf]


class HybridDraftProvider(DraftProvider):
    """MTP 起头 + ngram 补足（仅在 ngram 与 MTP 首猜一致时采纳其续写）.

    实测：MTP 对 t+2 的首猜贪心命中率 ~0.6~0.8，而 ngram 首猜只有 ~0.04。
    若把 MTP 猜测直接并入上下文取 ngram 续写，ngram 的后缀检索会被"没见过的
    草稿"污染而几乎永不命中（旧实现的坑）。因此只在两者**首 token 一致**时，
    才认为 ngram 找到的是一段可信续写，采纳它更长的自洽草稿；否则只用 MTP。
    """

    def __init__(self, model, min_match: int = 2, scan_limit: int = 1024):
        self.mtp = MtpDraftProvider(model)
        self.ngram = NgramDraftProvider(min_match, scan_limit)

    def propose(self, context: list[int], h_last, k: int) -> list[int]:
        """MTP 首猜与 ngram 续写一致则采纳 ngram（更长），否则退回 MTP 单猜."""
        return self.propose_scored(context, h_last, k)[0]

    def propose_scored(self, context: list[int], h_last, k: int):
        """带置信度的 hybrid：一致时用 ngram 长草稿（首 token 置信取两路较大者）."""
        if k <= 0:
            return [], []
        mt, mc = scored_propose(self.mtp, context, h_last, 1)      # t+2 首猜
        ng, nc = scored_propose(self.ngram, context, h_last, k)    # 自洽续写
        if ng and mt and ng[0] == mt[0]:
            # 一致：采纳更长的自洽续写；首 token 两路都认，置信取较大者
            confs = [max(mc[0], nc[0])] + list(nc[1:])
            return ng[:k], confs[:k]
        if mt:
            return mt, mc         # 不一致：MTP 首猜更准，只用它
        return ng[:k], nc[:k]     # 无 MTP 头或无首猜：退回 ngram


def build_draft_provider(model, source: str, min_match: int = 2) -> DraftProvider:
    """按 `draft_source` 配置构建起草器（未知来源回落 ngram，不报错崩生成）."""
    src = (source or "ngram").lower()
    if src == "mtp":
        return MtpDraftProvider(model)
    if src == "hybrid":
        return HybridDraftProvider(model, min_match)
    return NgramDraftProvider(min_match)
