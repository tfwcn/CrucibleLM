"""可控推理努力（Reasoning Effort，b ∈ [1,100]）— V4.1 §5.1.4 + 附录 C.

三块职责，对应"训练 → 部署"的完整链路：

| 环节 | 做法 | 本模块函数 |
|---|---|---|
| 条件化 | 系统提示前置一行 `Reasoning Effort: {b} (range 1–100; ...)` | `effort_instruction` / `apply_effort_messages` |
| 训练 | 同 `(prompt, b)` 子组内均值中心化；长度惩罚 `k(b)` 随 b 指数衰减 | `group_advantages` / `length_penalty` / `penalty_coefficient` |
| 部署 | `low/high/max = 50/75/100`；中间值由训练过的有限档插值 | `tier_to_effort` / `resolve_effort` |

## 为什么必须"组内比较"而不是"跨档比较"

高努力档天生更长、更容易做对。若把 b=25 和 b=100 的回答放进同一个 GRPO 组，
组均值会被长回答抬高——**"努力"这个被优化的量恰好是组内方差最大的那个量**，
优势函数于是把"答得更长"当成"答得更好"，努力档位迅速坍缩到最高档（或最低档）。

所以分组键是 `(prompt, b)`：只在同档内比谁更好；跨档的"长度差异"不由优势函数
表达，而由 `k(b)` 指数衰减的惩罚项塑形（eq.12）。这也是论文 Eq.8 那段
"responses from different effort levels are not directly compared" 的落地方式。

## 为什么惩罚系数要指数衰减

附录 C 的一阶条件：`p'(ℓ) = k(b)/L_norm`，而边际收益 `p'(ℓ) ≈ a·exp(-ℓ/s)`
本身是指数的；要让**最优长度对 b 近似线性**（eq.16），`k(b)` 也必须指数衰减：

```
ℓ*(b) ≈ C - s·log k0 + s·τ·(b − b_min)
```

`k0` 控制整体"往短里压"的力度，`τ` 控制"档位之间拉开多远"。
"""

from __future__ import annotations

import math

# 努力档的取值范围（论文：1–100 的整数）
EFFORT_MIN = 1
EFFORT_MAX = 100
# 公开 API 的三档（Table 2）：low=50 / high=75 / max=100
EFFORT_TIERS = {"low": 50, "high": 75, "max": 100}
# 指令原文（论文 §5.1.4，逐字对齐；换措辞等于换了一个没训过的条件）
_INSTRUCTION = ("Reasoning Effort: {b} (range 1–100; higher values request more "
                "thorough reasoning)")


# ------------------------------------------------------------------ 校验/解析
def validate_effort(b) -> int:
    """校验努力档并归一成 `int`（1..100）.

    越界**直接报错**而不是静默截断：静默截断会让"配置写错"变成"效果没变化"，
    排查成本极高（`clamp_effort` 是给部署侧 API 的宽容版本）。
    """
    if isinstance(b, bool):
        raise ValueError(f"努力档须为整数 1–100，实际 {b!r}（bool 不是档位）")
    if isinstance(b, float):
        if not b.is_integer():
            raise ValueError(f"努力档须为整数 1–100，实际 {b!r}")
        b = int(b)
    if not isinstance(b, int):
        raise ValueError(f"努力档须为整数 1–100，实际 {type(b).__name__}")
    if not (EFFORT_MIN <= b <= EFFORT_MAX):
        raise ValueError(
            f"努力档须在 [{EFFORT_MIN}, {EFFORT_MAX}] 内，实际 {b}")
    return b


def clamp_effort(b) -> int:
    """部署侧宽容版：四舍五入并截断到 [1,100]（非法输入退回默认 50）."""
    try:
        v = float(b)
    except (TypeError, ValueError):
        return EFFORT_TIERS["low"]
    if math.isnan(v):
        return EFFORT_TIERS["low"]
    return max(EFFORT_MIN, min(EFFORT_MAX, int(round(v))))


def tier_to_effort(name) -> int:
    """档名 → 数值：`low/high/max`（大小写不敏感），也接受数字字符串."""
    if isinstance(name, (int, float)) and not isinstance(name, bool):
        return validate_effort(name)
    key = str(name).strip().lower()
    if key in EFFORT_TIERS:
        return EFFORT_TIERS[key]
    try:
        return validate_effort(int(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"未知努力档 {name!r}：须为 {sorted(EFFORT_TIERS)} 或 1–100 的整数") from exc


def effort_to_tier(b) -> str | None:
    """数值 → 档名（只在**恰好等于**三档之一时返回，否则 None = 插值档）."""
    v = validate_effort(b)
    for name, val in EFFORT_TIERS.items():
        if v == val:
            return name
    return None


def resolve_effort(value) -> int | None:
    """统一入口：`None`/空串 → None（不改提示）；档名/数字 → 1..100 的整数."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return tier_to_effort(value)


def parse_effort_levels(spec) -> list[int]:
    """解析训练档位列表：`"25,50,75,100"` 或 `"low,high,max"`（去重升序）."""
    if spec is None:
        return []
    items = spec if isinstance(spec, (list, tuple, set)) else str(spec).split(",")
    out = {tier_to_effort(x) for x in items if str(x).strip()}
    return sorted(out)


def average_spacing(levels: list[int]) -> float:
    """档位平均间隔 `Δb`（eq.10 里 `τ = λ·Δb` 的 Δb）；少于 2 档返回 0."""
    if len(levels) < 2:
        return 0.0
    return (levels[-1] - levels[0]) / (len(levels) - 1)


def tau_from_levels(levels: list[int], lam: float = 1.0) -> float:
    """`τ = λ·Δb`：b 每增加 τ，惩罚系数乘 e⁻¹（eq.10）.

    `Δb=0`（只有一档）时退回 25（= 100/4），避免除零把惩罚变成阶跃。
    """
    if lam <= 0:
        raise ValueError(f"λ 须 > 0，实际 {lam}")
    d = average_spacing(levels)
    return lam * d if d > 0 else 25.0


# ------------------------------------------------------------------ 条件化
def effort_instruction(b) -> str:
    """生成前置指令行（论文原文；数字用阿拉伯写法、破折号用 en dash）."""
    return _INSTRUCTION.format(b=validate_effort(b))


def _strip_effort_lines(text: str) -> str:
    """去掉已有的 `Reasoning Effort:` 行（保证幂等，不叠加多行）."""
    keep = [ln for ln in text.splitlines()
            if not ln.lstrip().lower().startswith("reasoning effort:")]
    return "\n".join(keep).strip()


def apply_effort(system: str | None, b) -> str:
    """给系统提示前置努力指令；**幂等**（已有同名字段会被替换）."""
    body = _strip_effort_lines(system or "")
    line = effort_instruction(b)
    return f"{line}\n\n{body}" if body else line


def apply_effort_messages(messages: list[dict], b) -> list[dict]:
    """对 OpenAI 风格 messages 应用努力档：有 system 就改它，没有就在最前面插一条.

    返回**新列表**（不改调用方的 messages，避免同一请求重试时叠加两行）。
    """
    level = validate_effort(b)
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = apply_effort(m.get("content") or "", level)
            return out
    out.insert(0, {"role": "system", "content": apply_effort("", level)})
    return out


# ------------------------------------------------------------------ 奖励塑形
def penalty_coefficient(b, k0: float = 1.0, b_min: int = EFFORT_MIN,
                        tau: float = 25.0) -> float:
    """长度惩罚系数 `k(b) = k0·exp(−(b−b_min)/τ)`（eq.12）.

    低档惩罚重（往短里压），高档惩罚轻（允许长推理）；随 b **单调不增**。
    """
    level = validate_effort(b)
    if k0 < 0:
        raise ValueError(f"k0 须 ≥ 0，实际 {k0}")
    if tau <= 0:
        raise ValueError(f"τ 须 > 0，实际 {tau}")
    return float(k0) * math.exp(-(level - int(b_min)) / float(tau))


def length_penalty(tokens: int, b, k0: float = 1.0,
                   b_min: int = EFFORT_MIN, tau: float = 25.0,
                   l_norm: float = 1000.0, cap: float = 1.0) -> float:
    """长度惩罚 `r_len = −min(C_max, k(b)·ℓ/L_norm)`（eq.11）.

    返回**非正数**；`cap` 是最大扣分（达到上限后追加 token 不再惩罚，
    对应论文"capped region 上边际惩罚为零"）。`tokens<=0` 不罚。
    """
    if tokens < 0:
        raise ValueError(f"token 数须 ≥0，实际 {tokens}")
    if l_norm <= 0:
        raise ValueError(f"L_norm 须 > 0，实际 {l_norm}")
    if cap < 0:
        raise ValueError(f"cap 须 ≥ 0，实际 {cap}")
    if tokens == 0 or cap == 0 or k0 == 0:
        return 0.0
    raw = penalty_coefficient(b, k0, b_min, tau) * float(tokens) / float(l_norm)
    return -min(float(cap), raw)


def predicted_length(b, s_x: float = 1000.0, c_x: float = 0.0,
                     k0: float = 1.0, b_min: int = EFFORT_MIN,
                     tau: float = 25.0) -> float:
    """附录 C 的最优长度近似 `ℓ*(b) ≈ C_x − s_x·log k0 + s_x·τ·(b − b_min)`（eq.16）.

    只在**未触顶**区间成立；用途是做单测的"线性趋势"基准与标定 τ 的对照曲线。
    """
    level = validate_effort(b)
    if k0 <= 0:
        return math.inf
    return (float(c_x) - float(s_x) * math.log(float(k0))
            + float(s_x) * float(tau) * (level - int(b_min)))


def group_advantages(rewards: list[float], groups: list,
                     normalize_std: bool = False) -> list[float]:
    """组相对优势：**同组内**均值中心化（论文 Eq.8 的 group-relative advantage）.

    - 分组键通常是 `(prompt_id, b)`：跨档不比较（见模块文档）；
    - 组内只有 1 个样本 → 优势 0（没有比较对象，强行给 0 比给"绝对值"安全）；
    - `normalize_std=True` 时再除以组内标准差（GRPO 常见做法；论文正文只提
      mean-centered，故默认关闭，保持与论文一致）。
    """
    if len(rewards) != len(groups):
        raise ValueError(f"rewards({len(rewards)}) 与 groups({len(groups)}) 长度不一致")
    if not rewards:
        return []
    buckets: dict = {}
    for i, g in enumerate(groups):
        buckets.setdefault(g, []).append(i)
    out = [0.0] * len(rewards)
    for idxs in buckets.values():
        vals = [float(rewards[i]) for i in idxs]
        mean = sum(vals) / len(vals)
        if len(idxs) < 2:
            continue
        if normalize_std:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
            if std <= 0:
                continue
            for i, v in zip(idxs, vals):
                out[i] = (v - mean) / std
        else:
            for i, v in zip(idxs, vals):
                out[i] = v - mean
    return out


def group_key(prompt_id, b) -> tuple:
    """标准分组键 `(prompt_id, b)`（写成函数是为了让调用方别再各自拼）."""
    return (prompt_id, validate_effort(b))
