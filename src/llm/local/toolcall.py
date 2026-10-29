"""工具调用的**单一渲染与解析实现** —— 训练 prompt 与推理 prompt 共用.

## 为什么必须只有一个实现

工具调用是一个「输入格式即契约」的任务：模型在 SFT 时看到的系统提示长得像
`可用工具：\n- path：说明\n\n🔧 工具「path」\n  参数（N 个）：…`，推理时也必须
**逐字节相同**。这类漂移最难查——训练照常收敛、loss 照常下降，只是模型看到的
东西变了，线下评测掉点却找不到原因（历史上 SFT 模板 `<用户>/<助手>` 与采样
模板 `<user>/<assistant>` 不一致就是这么埋下的）。

所以渲染函数只放这里一份，`scripts/synth_tool_sft.py`（造数据）和
`src/llm/local/infer.py`（跑推理）都 import 它。

## 协议

模型输出一行紧凑 JSON：

    {"path":"workspace/fs_manage","arguments":{"action":"read","path":"docs/x.md"}}

- `path` 是**工具箱路径**（`分组/工具名`，与 moyume-agents 的 `toolbox_call`
  的 path 参数同义）；没有分组时就是工具名。
- `arguments` 是目标工具自己的参数对象，**不是** OpenAI 那层
  `{"path":…,"arguments":…}` 再包一层。
- `path` 为空 = 「所列工具都做不了，别调」（不硬套工具）。
- `arguments` 为空对象 = **SELECT**：只要工具、参数表还没给（渐进式发现的第一步），
  调用方拿到后应补上该工具的参数表再来一轮。这个语义让「选工具」和「填参数」
  可以分成两个小任务学，而不是逼 128M 一步到位。

## 三种输入形态

`normalize_tools()` 把下面三种都归一成内部结构（`path/description/parameters`）：

1. 本仓库 `dump_tool_schemas.py` 的产物条目（自带 `path`）；
2. OpenAI `{"type":"function","function":{…}}`（`path` 取 `function.name`）；
3. 裸 `{"name":…,"parameters":…}`。
"""

from __future__ import annotations

import json

#: 目录里单行说明的截断长度（与 ToolboxManager._MAX_DESC_LEN 同量级）
DESC_LEN = 60

#: 空调用的 path（负样本 / 无合适工具时的约定输出）
NO_TOOL = ""

#: 输出格式规则（训练/推理共用，改一个字都必须两边同时生效）
OUTPUT_RULE = (
    '输出规则：需要调用工具时，只输出一行 JSON，格式 '
    '{"path":"工具箱路径","arguments":{...}}；'
    '所列工具都无法完成时输出 {"path":"","arguments":{}}。'
)

#: 角色设定变体。训练时随机取（避免模型只认一种开场白），推理用 `DEFAULT_PERSONA`
PERSONAS = [
    "你是摸鱼助手，可以调用工具帮用户完成任务。",
    "你是一个能操作工作区的助手，按用户请求调用合适的工具。",
    "你在帮用户处理本地项目，需要时调用工具而不是凭空回答。",
]
DEFAULT_PERSONA = PERSONAS[0]

#: 约束解码用的宽松 schema：只保证「是两个字段的 JSON」，
#: 不校验 path 在不在候选里（那个交给解析阶段，报错更好定位）
CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["path", "arguments"],
    "additionalProperties": False,
}


def truncate(text: str, length: int = DESC_LEN) -> str:
    """压成单行并截断（渲染目录用）."""
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= length else text[: length - 1] + "…"


def normalize_tools(tools: list[dict] | None) -> list[dict]:
    """把三种工具声明形态归一成 `{path, name, description, parameters}`."""
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else None
        src = fn if isinstance(fn, dict) else tool
        name = (src.get("name") or "").strip()
        path = (tool.get("path") or src.get("path") or name).strip()
        if not path:
            continue
        params = src.get("parameters") or {"type": "object", "properties": {}}
        out.append({
            "path": path,
            "name": name or path.split("/")[-1],
            "description": (src.get("description") or "").strip(),
            "parameters": params,
        })
    return out


def render_catalog(tools: list[dict], exclude: str = "") -> str:
    """工具目录：一行一个 `- 路径：说明`.

    `exclude` 命中的工具不列 —— FILL 形态下它的参数表已经在下方单独给出，
    再在目录里重复一遍只是白烧 token。
    """
    lines = []
    for tool in tools:
        if exclude and tool["path"] == exclude:
            continue
        lines.append(f"- {tool['path']}：{truncate(tool['description'])}")
    return "\n".join(lines)


def render_params(tool: dict) -> str:
    """目标工具的完整参数表（中文口径，与 moyume-agents 的 browse 渲染一致）."""
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    lines = [f"🔧 工具「{tool['path']}」",
             f"  描述：{tool['description'] or '无'}"]
    if not props:
        lines.append("  参数：无")
        return "\n".join(lines)
    lines.append(f"  参数（{len(props)} 个）：")
    for name, meta in props.items():
        meta = meta or {}
        mark = "必填" if name in required else "可选"
        ptype = meta.get("type", "string")
        default = meta.get("default")
        default_s = ""
        if default is not None and default != "":
            default_s = f"，默认 {default}"
        desc = truncate(meta.get("description", ""), 70)
        lines.append(f"    - {name}: {ptype}（{mark}{default_s}）{desc}")
    return "\n".join(lines)


def render_system(persona: str, catalog: str, params: str = "") -> str:
    """拼系统提示：有 `params` 是 FILL 形态，没有是 SELECT 形态."""
    parts = [persona, "可用工具：", catalog]
    if params:
        parts += ["", params]
    parts += ["", OUTPUT_RULE]
    return "\n".join(parts)


def build_tool_system(tools: list[dict] | None, target: str = "",
                      persona: str = DEFAULT_PERSONA) -> str:
    """高层封装：把工具列表渲染成系统提示（训练与推理都走这里）.

    `target` 非空时渲染成 FILL 形态（目标工具的完整参数表附在下方）；
    为空则渲染成 SELECT 形态（只有目录）。
    """
    normalized = normalize_tools(tools)
    params = ""
    if target:
        hit = next((t for t in normalized if t["path"] == target), None)
        if hit is not None:
            params = render_params(hit)
    return render_system(persona, render_catalog(normalized, exclude=target), params)


# ---------------------------------------------------------------- 解析
def parse_tool_call(text: str) -> dict | None:
    """把模型输出解析成 `{"path":…,"arguments":…}`；不是这个形状返回 None.

    容忍三种常见偏差（小模型一定会犯）：
    1. 前后有多余文字/空白（截取第一个 `{` 到最后一个 `}`）；
    2. `arguments` 是 **JSON 字符串**而不是对象（OpenAI 风格，字段叫 `arguments`）；
    3. 只有一个 `path` 字符串（退化形态），按空参数处理。

    不猜、不纠错：`path` 不在调用方的候选里由调用方判断（它才知道候选集）。
    `arguments` 解析不出来就返回 None，避免把坏参数喂给工具。
    """
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    body = text[start:end + 1]
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    path = obj.get("path")
    if isinstance(path, (int, float)):
        path = str(path)
    if not isinstance(path, str):
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"path": path.strip(), "arguments": args}


def to_openai_tool_calls(path: str, arguments: dict,
                         call_id: str | None = None) -> list[dict]:
    """内部形态 → OpenAI `tool_calls`（服务端/前端都认这个）."""
    if not path:
        return []
    return [{
        "id": call_id or f"call_{_short_id(path, arguments)}",
        "type": "function",
        "function": {
            "name": path,
            "arguments": json.dumps(arguments or {}, ensure_ascii=False,
                                    separators=(",", ":")),
        },
    }]


def _short_id(path: str, arguments: dict) -> str:
    """给调用生成稳定短 id（同一调用重复出现时 id 一致，便于日志对齐）."""
    import hashlib

    raw = path + json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def format_call(path: str, arguments: dict) -> str:
    """内部形态 → 训练用的紧凑单行 JSON（与 `parse_tool_call` 互逆）."""
    return json.dumps({"path": path, "arguments": arguments},
                      ensure_ascii=False, separators=(",", ":"))
