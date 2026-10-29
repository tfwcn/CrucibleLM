"""从 moyume-agents 注册表导出真实工具 schema（给合成 SFT 数据用）.

## 为什么需要

`data/` 下的 SFT 语料里工具调用样本是 **0 条**（grep `tool_call|shell_exec|fs_manage`
只命中 1 处误匹配）。模型不会凭空学会调用工具，必须有监督信号；而最好的
schema 源不是手写的，是运行时真正注册的那 51 个工具 —— 手写迟早会和代码漂移。

本脚本把 `create_default_registry()` 扫出来的每个 `FunctionTool` 连同
`BaseTool` 自声明元数据（分组、意图标签、可见性）一起落盘，供
`scripts/synth_tool_sft.py` 消费。

## 怎么跑（必须用 agents 自己的 venv）

工具模块在 import 时会拉 apscheduler / dotenv / matplotlib，训练用的
`llm-train` venv 里没有这些包，直接 import 会全线失败（工具数 0）。所以：

    cd moyume-agents
    .venv/bin/python ../CrucibleLM/scripts/dump_tool_schemas.py \\
        --out ../CrucibleLM/data/tool_schemas.json

脚本自己把 agents root 插进 `sys.path`，`--agents-root` 默认取
「本脚本所在目录的 ../../moyume-agents」，也可显式指定。

## 产物结构

```json
{
  "version": 1,
  "agents_root": "/…/moyume-agents",
  "tools": [
    {
      "path": "workspace/fs_manage",      // 工具箱路径（模型实际看到的形式）
      "name": "fs_manage",
      "group": "workspace",
      "group_title": "工作区",
      "group_description": "读写文件、执行命令",
      "description": "读写工作区文件…",
      "search_tags": ["读文件", "写文件", "查看代码"],
      "visible_to": ["main", "worker"],
      "parameters": {"type": "object", "properties": {…}, "required": […]},
      "example_call": "toolbox_call(path=\"workspace/fs_manage\", arguments=…)"
    }
  ]
}
```

`example_call` 复刻 `ToolboxManager._render_tool` 里那份「可复制的最小正确示例」
（string 全填 `"..."`、integer 填 5），所以它**不含任何答案**，可以原样放进
训练 prompt 当格式样板，不会被模型抄走具体值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION = 1


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="导出 moyume-agents 真实工具 schema")
    p.add_argument("--agents-root", default="",
                   help="moyume-agents 仓库根（默认按脚本位置推 ../../moyume-agents）")
    p.add_argument("--out", default="", help="输出 JSON（默认 <本仓库>/data/tool_schemas.json）")
    p.add_argument("--only-visible", default="main",
                   help="只导出对谁可见的工具（main / worker / all；默认 main）")
    return p.parse_args(argv)


def default_agents_root() -> Path:
    """按脚本位置推 moyume-agents 根目录（CrucibleLM/scripts/ → ../../moyume-agents）."""
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "moyume-agents"


def example_call(path: str, params: dict) -> str:
    """复刻 `ToolboxManager._call_example`：可复制的最小正确示例（值为占位符）."""
    props = (params or {}).get("properties") or {}
    required = set((params or {}).get("required") or [])
    example: dict = {}
    for pname, pmeta in props.items():
        if pname not in required and len(example) >= 2:
            continue
        pmeta = pmeta or {}
        default = pmeta.get("default")
        ptype = pmeta.get("type", "string")
        if default is not None:
            example[pname] = default
        elif ptype == "integer":
            example[pname] = 5
        elif ptype == "boolean":
            example[pname] = True
        elif ptype == "array":
            example[pname] = []
        elif ptype == "object":
            example[pname] = {}
        else:
            example[pname] = "..."
        if len(example) >= 3:
            break
    args = json.dumps(example, ensure_ascii=False)
    return f'toolbox_call(path="{path}", arguments={args})'


def collect(agents_root: Path, only_visible: str) -> dict:
    """建注册表并整理成可落盘结构（不写文件，便于单测）."""
    sys.path.insert(0, str(agents_root))
    from src.tools.base import get_tool_meta  # noqa: E402
    from src.tools.registry import create_default_registry  # noqa: E402

    registry = create_default_registry()
    names = registry.all_names
    if not names:
        raise SystemExit(
            "注册表为空：十有八九是依赖没装（apscheduler / dotenv / matplotlib）。\n"
            "请用 moyume-agents 自己的 venv 跑：cd moyume-agents && "
            ".venv/bin/python ../CrucibleLM/scripts/dump_tool_schemas.py"
        )

    want = None if only_visible == "all" else only_visible
    tools: list[dict] = []
    skipped: list[str] = []
    for name in names:
        tool = registry.get(name)
        if tool is None:
            continue
        meta = get_tool_meta(name) or {}
        visible = sorted(meta.get("visible_to") or {"main", "worker"})
        if want and want not in visible:
            skipped.append(name)
            continue
        group = meta.get("group") or ""
        path = f"{group}/{name}" if group else name
        params = tool.openai_schema()["function"]["parameters"]
        tools.append({
            "path": path,
            "name": name,
            "group": group,
            "group_title": meta.get("group_title") or "",
            "group_description": meta.get("group_description") or "",
            "description": tool.description or "",
            "search_tags": list(meta.get("search_tags") or ()),
            "visible_to": visible,
            "parameters": params,
            "example_call": example_call(path, params),
        })

    tools.sort(key=lambda t: t["path"])
    return {
        "version": VERSION,
        "agents_root": str(agents_root),
        "skipped": skipped,
        "tools": tools,
    }


def main(argv=None) -> int:
    """主流程：建表 → 校验 → 落盘 → 打报告."""
    args = parse_args(argv)
    agents_root = Path(args.agents_root) if args.agents_root else default_agents_root()
    if not (agents_root / "src" / "tools" / "registry.py").exists():
        raise SystemExit(f"--agents-root 不像 moyume-agents：{agents_root}")

    payload = collect(agents_root, args.only_visible)
    tools = payload["tools"]

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "data" / "tool_schemas.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    groups: dict[str, int] = {}
    for t in tools:
        groups[t["group"] or "(无分组)"] = groups.get(t["group"] or "(无分组)", 0) + 1
    total_chars = sum(len(json.dumps(t["parameters"], ensure_ascii=False)) for t in tools)
    print(f"导出工具 {len(tools)} 个（跳过不可见 {len(payload['skipped'])} 个）")
    print("分组分布：" + "，".join(f"{g}×{n}" for g, n in sorted(groups.items())))
    print(f"参数 schema 合计 {total_chars} 字符，平均 {total_chars // max(len(tools), 1)} 字符/工具")
    no_desc = [t["path"] for t in tools if not t["description"]]
    if no_desc:
        print(f"警告：{len(no_desc)} 个工具无描述：{no_desc[:5]}")
    print(f"已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
