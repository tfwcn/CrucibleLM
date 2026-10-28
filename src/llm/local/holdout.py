"""固定评测集 —— 让跨 run 的 val_loss 真正可比.

由来（见 docs/pitfalls/local-llm-training.md 第 79 条）：
`collect_holdout` 的 holdout 是「当前混合语料流的前 256 篇」，所以
`--local-path` 一改、`holdout` 就换人。base run 在 5310 步把
`data/hq-zh` 换成 `data/hq-zh:3,data/ultra-zh:1`，val_loss 从 4.599
跳到 7.198，再花 4600 步降回 5.01 —— 这个 7.19→5.01 的"进步"和
4.18→4.60 的"退化"其实都不在同一个评测集上，两条曲线无法拼在一起看。

本模块把 holdout 落盘成一份带 phase 标记的 JSON：首次运行自动写入，
之后的 run 一律复用。**只做存取与校验，不改训练流的跳过逻辑** ——
只要保存的就是"当前语料流的前 N 篇"，后续同源 run 的跳过位置与它一致，
不会把评测文档漏进训练集。
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = 1


def save_holdout(path: Path | str, items: list, phase: str) -> None:
    """落盘评测集（pretrain 为 str 列表，sft 为 (prompt, response) 列表）."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "phase": phase,
        "items": [list(it) if isinstance(it, tuple) else it for it in items],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_holdout(path: Path | str, phase: str) -> list | None:
    """读回评测集；文件缺失 / 版本或 phase 不符 / 内容坏 → 返回 None（不抛）.

    phase 不符（pretrain 与 sft 的样本形状不同）时返回 None，调用方据此
    重新生成并覆盖，避免把 sft 对喂给 pretrain 的打包函数。
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return None
    if data.get("phase") != phase:
        return None
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None
    if phase == "sft":
        # (prompt, response) 对：两元且都是 str，否则整份作废
        out = []
        for it in items:
            if (not isinstance(it, list) or len(it) != 2
                    or not all(isinstance(s, str) for s in it)):
                return None
            out.append((it[0], it[1]))
        return out
    return [it for it in items if isinstance(it, str)]
