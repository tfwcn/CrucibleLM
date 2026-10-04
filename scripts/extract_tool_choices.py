"""抽取工具选择样本 — 遍历会话库，产出 (上下文, 工具名) 训练对.

来源：data/sandbox/*/session/*/conversation.db 的 messages 表
（assistant.tool_calls JSON 取 function.name）。
上下文取调用前的消息拼接（按 id 排序，截断尾部保留最近，默认 1500 字）；
同 (上下文, 工具) 精确去重；输出 jsonl：{"context": ..., "tool": ...}。

用法：python scripts/extract_tool_choices.py [--base data/sandbox] [--out data/tool_choices.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 上下文截断（保留尾部最近内容，工具选择依赖近期消息）
CONTEXT_CHARS = 1500


def iter_session_dbs(base: Path) -> list[Path]:
    """找出所有会话库（路径不存在返回空，不报错）."""
    if not base.exists():
        return []
    return sorted(base.rglob("conversation.db"))


def read_messages(db: Path) -> list[dict]:
    """读消息（按 id 排序，坏库跳过返回空）."""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM messages ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for role, content, tool_calls in rows:
        item: dict = {"role": role or "", "content": content or ""}
        if tool_calls:
            try:
                item["tool_calls"] = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(item)
    return out


def extract_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """消息流转 (上下文, 工具名) 对：每个带 tool_calls 的 assistant 取首个工具名."""
    role_tag = {"system": "系统", "user": "用户", "assistant": "助手",
                "tool": "工具", "observation": "观察"}
    pairs: list[tuple[str, str]] = []
    context: list[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        calls = m.get("tool_calls") or []
        if role == "assistant" and calls:
            name = ((calls[0].get("function") or {}).get("name") or "").strip()
            if name and context:
                ctx = "\n".join(context)[-CONTEXT_CHARS:]
                pairs.append((ctx, name))
        if content:
            context.append(f"<{role_tag.get(role, role)}>\n{content}")
    return pairs


def main(argv=None) -> int:
    """抽取主流程（打印工具分布统计）."""
    p = argparse.ArgumentParser(description="抽取工具选择样本")
    p.add_argument("--base", default="data/sandbox", help="会话沙盒根目录")
    p.add_argument("--out", default="data/tool_choices.jsonl", help="输出 jsonl")
    args = p.parse_args(argv)
    from collections import Counter

    dbs = iter_session_dbs(Path(args.base))
    print(f"会话库：{len(dbs)} 个", flush=True)
    seen: set[tuple[str, str]] = set()
    dist: Counter = Counter()
    n = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for db in dbs:
            for ctx, tool in extract_pairs(read_messages(db)):
                if (ctx, tool) in seen:
                    continue
                seen.add((ctx, tool))
                dist[tool] += 1
                f.write(json.dumps({"context": ctx, "tool": tool},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"样本：{n} 对，工具 {len(dist)} 种", flush=True)
    for tool, c in dist.most_common(10):
        print(f"  {tool}: {c}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
