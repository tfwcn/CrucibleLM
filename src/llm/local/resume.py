"""续训一致性守卫 —— 重启前比对「上次 run 的配方」，防静默改坏曲线.

由来（见 docs/pitfalls/local-llm-training.md 第 78 条）：
base 预训练 `train.log` 里那段「4.18 → 4.65」的所谓退化，实际是 3210 步重启时
`max_steps` 从 10000 改成 14000 造成的三连击，一条都不报错：

1. **cosine 曲线被重算**：`lr_schedule(3210, 14000, 50, 3e-4)` 给出 2.67e-4，
   而按原计划（10000）此时应是 2.39e-4；更要紧的是此后不再继续衰减，
   等于把已经走了一半的退火拉回高位重新走一遍。
2. **Adam 动量若丢失**：`load_ckpt` 遇到不兼容的 `optim.pt` 只打印一行警告，
   动量归零后 lr=2.67e-4 的更新等效退化成 sign-SGD 大步长，一步就能把
   loss 从 3.92 打到 6.03。
3. **5310 步换 `local_path`**：holdout 取的是「当前混合流的前 256 篇」
   （`collect_holdout`），换语料 = 换考卷，「4.60 → 7.19」不是模型变差，
   是新考卷更难。

三者共同点：**只在重启时发生，且永远静默**。本模块只负责「检测 + 报错 +
给修复建议」，不替调用方决定策略；确实要改配方时显式加 `--force-resume`。

设计取舍：比对源用 `ckpt_dir/hparams.json`（`dump_hparams` 每次启动都会覆盖），
所以必须在覆盖**之前**读旧值。旧 ckpt 缺字段时只跳过该键，不误报。
"""

from __future__ import annotations

import json
from pathlib import Path

# 改动其中任一项，都会让「同一条训练曲线」这个前提失效：
# - max_steps/lr/warmup/muon_lr/optimizer → LR 曲线与更新尺度
# - seq_len/batch/accum → tokens_per_step，tokens_seen 不再可比
# - data/local_path/pretrain_mix/sft_mix → 语料与 holdout（评测集换人）
# - phase → 损失口径（pretrain 全序列 vs sft 掩码）
CRITICAL_KEYS: tuple[str, ...] = (
    "phase",
    "max_steps",
    "lr",
    "warmup",
    "muon_lr",
    "optimizer",
    "seq_len",
    "batch",
    "accum",
    "data",
    "local_path",
    # 语料重复遍数改了 = 同一批样本见到的次数变了，曲线不可比
    "sft_epochs",
    "pretrain_mix",
    "sft_mix",
    # 分词器换了 = 同一段文本切出不同 id 序列，等价换语言；必须显式放行
    "vocab",
    "subword_vocab",
)

# 浮点比较的相对容差（3e-4 与 0.0003 必须视为相等）
_REL_TOL = 1e-9


def load_prev_hparams(ckpt_dir: Path | str) -> dict | None:
    """读上次 run 落盘的 hparams（不存在/坏文件返回 None，不抛）."""
    path = Path(ckpt_dir) / "hparams.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _same(a: object, b: object) -> bool:
    """宽松相等：数值按相对容差比，其余按 == 比（None 与缺省同样对待）."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        # 布尔只与布尔相等：`true` 与 `1` 判为不同（守卫宁可多报，不可漏报）
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num):
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
        scale = max(abs(fa), abs(fb), 1e-12)
        return abs(fa - fb) <= _REL_TOL * scale
    return a == b


def diff_resume(prev: dict, args: object) -> list[tuple[str, object, object]]:
    """比对上次配方与本次参数，返回 [(键, 旧值, 新值)]（按 CRITICAL_KEYS 顺序）.

    `prev` 缺某键 → 该项跳过（旧 ckpt 无记录，不误报）。
    """
    out: list[tuple[str, object, object]] = []
    for key in CRITICAL_KEYS:
        if key not in prev:
            continue
        old = prev.get(key)
        new = getattr(args, key, None)
        if not _same(old, new):
            out.append((key, old, new))
    return out


def optim_state_missing(ckpt_dir: Path | str) -> bool:
    """`optim.pt` 是否存在（缺失 = 动量归零，续训等效换优化器）."""
    return not (Path(ckpt_dir) / "optim.pt").exists()


def format_block(diffs: list[tuple[str, object, object]],
                 optim_lost: bool) -> str:
    """把差异渲染成可直接粘贴到终端的报错文本（含每条的影响与修法）."""
    lines = ["续训配方与上次 run 不一致，已中止（避免静默改坏训练曲线）：", ""]
    advice = {
        "max_steps": "cosine 曲线会被重算，LR 不再沿原计划衰减",
        "lr": "峰值学习率变了，与已走过的退火段不可比",
        "warmup": "warmup 段位置变了",
        "muon_lr": "Muon 组峰值变了",
        "optimizer": "优化器换了，动量语义不同",
        "seq_len": "每步 token 数变了，tokens_seen 不再可比",
        "batch": "每步 token 数变了，tokens_seen 不再可比",
        "accum": "每步 token 数变了，tokens_seen 不再可比",
        "data": "语料来源变了",
        "local_path": "语料变了 → holdout 也跟着换，val_loss 不可比",
        "pretrain_mix": "语料配比变了 → holdout 也跟着换",
        "sft_mix": "语料配比变了 → holdout 也跟着换",
        "phase": "损失口径变了（pretrain 全序列 vs sft 掩码）",
        "vocab": "分词器换了 = 同一段文本切出不同 id 序列",
        "subword_vocab": "追加了子词 = 切法变了，新 token 的 embedding 需重新适配",
    }
    for key, old, new in diffs:
        lines.append(f"  · {key}: {old!r} → {new!r}")
        if key in advice:
            lines.append(f"      影响：{advice[key]}")
    if optim_lost:
        lines.append("  · optim.pt 缺失：Adam/Muon 动量将归零，"
                     "等效于用大学习率重踢一脚")
    lines += [
        "",
        "三种正当做法：",
        "  1) 只想接着跑完原曲线 → 把参数改回上面的旧值；",
        "  2) 想换配方/换语料 → 用新的 --ckpt-dir 起一轮新 run（推荐，曲线干净）；",
        "  3) 确实要在原地改 → 加 --force-resume（会照常继续，但这段曲线要单独标注）。",
    ]
    return "\n".join(lines)
