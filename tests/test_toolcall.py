"""工具调用协议单测 —— 渲染/解析/路径对齐/推理端接线（离线）.

守两件事：
1. `toolcall.py` 的渲染与解析是**训练和推理共用**的，任何一边单独改都会让
   模型看到的分布漂移（loss 照降、线下掉点，最难查的那种 bug）；
2. `LocalChatBackend.chat(tools=...)` 真的能吐出 OpenAI 形状的 tool_calls，
   幻觉路径 / 空 path / 坏 JSON 都要被正确地「当作没调用」。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.toolcall import (  # noqa: E402
    CALL_SCHEMA,
    DEFAULT_PERSONA,
    NO_TOOL,
    OUTPUT_RULE,
    build_tool_system,
    format_call,
    normalize_tools,
    parse_tool_call,
    render_catalog,
    render_params,
    render_system,
    to_openai_tool_calls,
    truncate,
)

TOOLS = [
    {"path": "workspace/fs_manage", "name": "fs_manage",
     "description": "对用户工作区内的文件进行读写操作。",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string", "description": "list/read/write"},
         "path": {"type": "string"},
     }, "required": ["action"]}},
    {"path": "web/web_search", "name": "web_search",
     "description": "通用网页搜索，返回标题与链接。",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
     }, "required": ["query"]}},
]


# ---------------------------------------------------------------- 归一化
def test_normalize_from_repo_schema():
    """本仓库 schema 条目：path 直接用."""
    got = normalize_tools(TOOLS)
    assert [t["path"] for t in got] == ["workspace/fs_manage", "web/web_search"]


def test_normalize_from_openai_shape():
    """OpenAI 形态：path 缺失时回落到 function.name（不丢工具）."""
    got = normalize_tools([{
        "type": "function",
        "function": {"name": "shell_exec", "description": "执行命令",
                     "parameters": {"type": "object", "properties": {}}},
    }])
    assert got[0]["path"] == "shell_exec"
    assert got[0]["description"] == "执行命令"


def test_normalize_skips_broken_entries():
    """缺 name/path 的条目跳过而不是抛（一个坏工具不该让整轮掉线）."""
    assert normalize_tools([{}, {"type": "function", "function": {}}]) == []


def test_normalize_empty():
    """None / 空列表都安全（默认不挂工具时走原路径）."""
    assert normalize_tools(None) == []
    assert normalize_tools([]) == []


# ---------------------------------------------------------------- 渲染
def test_truncate_single_line_and_ellipsis():
    """目录说明压成单行并截断（换行会把目录撑散）."""
    assert truncate("第一行\n第二行") == "第一行 第二行"
    assert truncate("啊" * 100, 10).endswith("…")
    assert len(truncate("啊" * 100, 10)) == 10


def test_render_catalog_one_line_per_tool():
    """目录一行一个工具，格式固定 `- 路径：说明`."""
    text = render_catalog(TOOLS)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("- workspace/fs_manage：")


def test_render_params_marks_required():
    """参数表必须标必填/可选，且带参数语义（模型靠它决定填什么）."""
    text = render_params(TOOLS[0])
    assert "🔧 工具「workspace/fs_manage」" in text
    assert "action: string（必填）" in text
    assert "list/read/write" in text


def test_render_system_select_vs_fill():
    """SELECT 形态没有参数表，FILL 形态有；两者都以输出规则收尾."""
    select = render_system(DEFAULT_PERSONA, render_catalog(TOOLS))
    fill = render_system(DEFAULT_PERSONA, render_catalog(TOOLS),
                         render_params(TOOLS[0]))
    assert "🔧" not in select
    assert "🔧" in fill
    assert select.endswith(OUTPUT_RULE)
    assert fill.endswith(OUTPUT_RULE)


def test_build_tool_system_target_excludes_from_catalog():
    """FILL 形态下目标工具不在目录里重复出现（省 token，也避免模型看两遍）."""
    text = build_tool_system(TOOLS, target="workspace/fs_manage")
    head, _, tail = text.partition("🔧")
    assert "workspace/fs_manage" not in head
    assert "workspace/fs_manage" in tail


def test_build_tool_system_unknown_target_falls_back_to_select():
    """目标不在候选里 → 退回 SELECT 形态（不抛，调用方可能是猜的路径）."""
    text = build_tool_system(TOOLS, target="nope/x")
    assert "🔧" not in text


# ---------------------------------------------------------------- 解析
def test_parse_exact_call():
    """标准输出能解析."""
    call = parse_tool_call('{"path":"web/web_search","arguments":{"query":"x"}}')
    assert call == {"path": "web/web_search", "arguments": {"query": "x"}}


def test_parse_tolerates_surrounding_text():
    """前后多余文字要容忍（小模型常在 JSON 前说一句话）."""
    call = parse_tool_call('好的，我来查：\n{"path":"a/b","arguments":{}}\n以上。')
    assert call["path"] == "a/b"


def test_parse_arguments_as_json_string():
    """`arguments` 是 JSON 字符串（OpenAI 风格）时也要能解."""
    call = parse_tool_call('{"path":"a/b","arguments":"{\\"x\\":1}"}')
    assert call["arguments"] == {"x": 1}


def test_parse_empty_arguments_is_select():
    """空 arguments = SELECT（不是错误）."""
    call = parse_tool_call('{"path":"a/b","arguments":{}}')
    assert call["arguments"] == {}


def test_parse_empty_path():
    """空 path = 「都做不了」，解析层保留（由调用方决定不调）."""
    assert parse_tool_call(format_call(NO_TOOL, {}))["path"] == ""


def test_parse_rejects_non_json():
    """不是 JSON → None（不能让坏输出进工具执行）."""
    for text in ["我做不到", "", "{不是 json", "[1,2]" , "null"]:
        assert parse_tool_call(text) is None


def test_parse_rejects_bad_arguments():
    """arguments 是坏 JSON 字符串 / 非对象 → None."""
    assert parse_tool_call('{"path":"a/b","arguments":"{坏的"}') is None
    assert parse_tool_call('{"path":"a/b","arguments":[1]}') is None


def test_parse_missing_path():
    """没有 path 字段 → None（旧形态 {"name":…} 不在协议内）."""
    assert parse_tool_call('{"name":"a","arguments":{}}') is None


def test_parse_missing_arguments_defaults_empty():
    """只有 path 时按空参数处理（退化但不该崩）."""
    assert parse_tool_call('{"path":"a/b"}')["arguments"] == {}


def test_format_parse_roundtrip():
    """`format_call` 与 `parse_tool_call` 互逆（包括中文与嵌套）."""
    cases = [("workspace/fs_manage", {"action": "读", "path": "a b/c.md"}),
             ("x/y", {}),
             ("pcr/pcr_homework", {"limit": 5, "tail": True})]
    for path, args in cases:
        assert parse_tool_call(format_call(path, args)) == {"path": path,
                                                           "arguments": args}


# ---------------------------------------------------------------- OpenAI 形态
def test_to_openai_tool_calls_shape():
    """转出来的结构必须与 OpenAI 一致（arguments 是 JSON 字符串）."""
    calls = to_openai_tool_calls("web/web_search", {"query": "摸鱼"})
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "web/web_search"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"query": "摸鱼"}


def test_to_openai_tool_calls_empty_path_is_no_call():
    """空 path 不产生 tool_calls（这才是负样本的正确下游行为）."""
    assert to_openai_tool_calls("", {}) == []


def test_to_openai_call_id_is_stable():
    """同一次调用的 id 稳定（日志/重放能对齐）."""
    a = to_openai_tool_calls("a/b", {"x": 1})[0]["id"]
    b = to_openai_tool_calls("a/b", {"x": 1})[0]["id"]
    c = to_openai_tool_calls("a/b", {"x": 2})[0]["id"]
    assert a == b and a != c


def test_call_schema_is_valid_json_schema_subset():
    """约束解码用的 schema 必须只含它支持的关键字（多余关键字会报错）."""
    assert CALL_SCHEMA["type"] == "object"
    assert set(CALL_SCHEMA["required"]) == {"path", "arguments"}
    assert CALL_SCHEMA["properties"]["arguments"]["type"] == "object"


# ---------------------------------------------------------------- 推理端接线
torch = pytest.importorskip("torch")


def _backend(vocab_size: int = 512):
    """极小模型（随机权重，只验证接线不验证质量）."""
    from src.llm.local.config import tiny_test_config
    from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
    from src.llm.local.model import TinyLLM

    cfg = tiny_test_config()
    cfg.vocab_size = vocab_size
    model = TinyLLM(cfg).eval()
    tok = SimpleTokenizer(vocab_size)
    chars = ["的", "是", "{", "}", '"', ":", ",", "a", "b", "path", "arguments",
             " ", "\n"] + list("abcdefghijklmnopqrstuvwxyz/_-")
    tok._chars = chars
    tok._ids = {c: i + 4 for i, c in enumerate(tok._chars)}
    tok._sync_meta()
    return LocalChatBackend(model, tok)


def test_inject_tools_replaces_system_message():
    """注入后只有一条 system，且内容是渲染结果（不是拼在用户消息里）."""
    backend = _backend()
    messages = [{"role": "system", "content": "你是摸鱼助手"},
                {"role": "user", "content": "读一下 a.md"}]
    out = backend._inject_tools(messages, TOOLS)
    systems = [m for m in out if m["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith("你是摸鱼助手")
    assert "workspace/fs_manage" in systems[0]["content"]
    # 原 messages 不能被就地改动（调用方可能复用）
    assert messages[0]["content"] == "你是摸鱼助手"


def test_inject_tools_inserts_when_missing_system():
    """没有 system 时插到最前，且用默认 persona."""
    backend = _backend()
    out = backend._inject_tools([{"role": "user", "content": "hi"}], TOOLS)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith(DEFAULT_PERSONA)


def test_tool_response_format_defaults_to_constraint():
    """带 tools 且没显式给 response_format 时默认上约束（否则可能吐非 JSON）."""
    backend = _backend()
    got = backend._tool_response_format(TOOLS, "", None)
    assert got["type"] == "json_schema"
    assert got["json_schema"]["schema"] is CALL_SCHEMA


def test_tool_response_format_respects_caller():
    """调用方显式给了 response_format 就不覆盖（可能是 json_object）."""
    backend = _backend()
    assert backend._tool_response_format(TOOLS, "", {"type": "json_object"}) == {
        "type": "json_object"}
    assert backend._tool_response_format([], "", None) is None


def test_parse_tool_output_rejects_hallucinated_path():
    """模型编了一个不在候选里的路径 → 当作没调用."""
    backend = _backend()
    text = format_call("workspace/not_exist", {})
    assert backend._parse_tool_output(text, TOOLS) is None


def test_parse_tool_output_maps_short_name():
    """`shell_exec` ↔ `workspace/shell_exec` 要能对上，且返回调用方的那份路径."""
    backend = _backend()
    tools = [{"type": "function",
              "function": {"name": "shell_exec", "description": "跑命令",
                           "parameters": {"type": "object", "properties": {}}}}]
    call = backend._parse_tool_output("{\"path\":\"shell_exec\",\"arguments\":{}}",
                                      tools)
    assert call["path"] == "shell_exec"


def test_parse_tool_output_ambiguous_tail_is_rejected():
    """两个分组都有同名工具时不做猜测（宁可报没匹配）."""
    backend = _backend()
    tools = [
        {"path": "a/run", "parameters": {"type": "object", "properties": {}}},
        {"path": "b/run", "parameters": {"type": "object", "properties": {}}},
    ]
    text = '{"path":"run","arguments":{}}'
    assert backend._parse_tool_output(text, tools) is None


def test_parse_tool_output_empty_path_is_no_call():
    """空 path（负样本的约定输出）→ None，不产生 tool_calls."""
    backend = _backend()
    assert backend._parse_tool_output(format_call(NO_TOOL, {}), TOOLS) is None


def test_chat_with_tools_returns_openai_shape():
    """端到端接线：chat(tools=...) 返回带 finish_reason=tool_calls 的形状.

    随机权重模型不保证输出合法路径，所以只断言「结构自洽」：
    有 tool_calls 时 content 必须为空；没有时 content 必须保留原文。
    """
    backend = _backend()
    out = backend.chat([{"role": "user", "content": "搜一下摸鱼"}],
                       max_new_tokens=24, tools=TOOLS, seed=0)
    assert set(out) == {"content", "reasoning", "tool_calls", "finish_reason"}
    if out["tool_calls"]:
        assert out["content"] == ""
        assert out["finish_reason"] == "tool_calls"
        call = out["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] in {t["path"] for t in TOOLS}
    else:
        assert out["finish_reason"] in {"stop", "length"}


def test_chat_without_tools_unchanged():
    """不传 tools 时行为与改造前完全一致（tool_calls 恒 None）."""
    backend = _backend()
    out = backend.chat([{"role": "user", "content": "hi"}], max_new_tokens=8,
                       seed=0)
    assert out["tool_calls"] is None
    assert out["content"] != "" or out["finish_reason"] == "length"
