"""LES 式专家调度 — 用历史命中预测下一步热点专家，供 offload 预取.

思想（借鉴 LES：Lookahead Expert Scheduling）：MoE 的专家命中在相邻步之间
高度相关（同一段文本反复激活同一组专家）。维护一个近期命中窗口，用"越近越重"
的打分预测下一步最可能命中的专家集合，提前把它们搬上算力设备，把搬运延迟
藏到计算后面。

边界：
- 纯统计、零可训练参数，不污染 state_dict（与 memory/retro 的"轻量外挂"同风格）；
- 只影响"何时把哪些专家搬上设备"（时序），**不改变数值结果**（单测锁定）；
- 默认关；无 offload 时即便开了也无副作用（predict 结果没人消费）。
"""

from __future__ import annotations

from collections import deque


class LesPredictor:
    """专家命中预测器：近期命中窗口 + 越近越重的加权打分."""

    def __init__(self, n_experts: int, history: int = 8, topn: int = 4):
        if n_experts <= 0:
            raise ValueError("n_experts 须为正")
        self.n_experts = int(n_experts)
        self.history = max(1, int(history))
        self.topn = max(1, int(topn))
        self._recent: deque[set[int]] = deque(maxlen=self.history)

    def observe(self, hit_ids) -> None:
        """记录本步真实命中的专家集合（滑出窗口最旧者）."""
        self._recent.append({int(i) for i in hit_ids})

    def predict(self, exclude=()) -> list[int]:
        """预测下一步最可能命中的 topn 个专家（排除已在热端的）.

        打分 = 各历史步命中该专家的 1/(age+1) 之和（最新步权重 1，越旧越轻）；
        并列时按专家号升序，保证确定性（同输入同输出）。
        """
        ex = {int(i) for i in exclude}
        scores = [0.0] * self.n_experts
        for age, hits in enumerate(reversed(self._recent)):
            w = 1.0 / (age + 1)
            for i in hits:
                if 0 <= i < self.n_experts:
                    scores[i] += w
        order = sorted(range(self.n_experts), key=lambda i: (-scores[i], i))
        return [i for i in order if i not in ex][:self.topn]


def build_les(n_experts: int, history: int = 8, topn: int = 4) -> LesPredictor:
    """按配置构建 LES 预测器."""
    return LesPredictor(n_experts, history, topn)
