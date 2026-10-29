"""工具调用合成数据单测 —— schema 合法率、候选集、模板一致性（离线，无 torch）.

这组测试守的是「数据本身对不对」，不是模型好不好：合成脚本一旦静默产出
坏样本（参数缺必填、path 不在 schema 里、prompt 与推理端不一致），训练会
照常跑完、loss 照常下降，只是模型学的是废东西。所以必须在这里卡死。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from synth_tool_sft import (  # noqa: E402
    NEGATIVE_REQUESTS,
    NO_TOOL,
    SCENARIOS,
    build_fill,
    build_negative,
    build_select,
    check_scenarios,
    dump_call,
    pick_distractors,
    render_catalog,
    render_params,
    sample_args,
    validate_args,
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "tool_schemas.json"


def _tools() -> list[dict]:
    """最小 schema 表（不依赖 data/ 下那份真实产物，测试要能独立跑）."""
    return [
        {"path": "workspace/fs_manage", "name": "fs_manage", "group": "workspace",
         "description": "对用户工作区内的文件进行读写操作。",
         "parameters": {"type": "object", "properties": {
             "action": {"type": "string", "description": "list/read/write"},
             "path": {"type": "string"},
             "limit": {"type": "integer"},
         }, "required": ["action"]}},
        {"path": "workspace/shell_exec", "name": "shell_exec", "group": "workspace",
         "description": "在工作区执行终端命令。",
         "parameters": {"type": "object", "properties": {
             "command": {"type": "string"}, "timeout": {"type": "integer"},
         }, "required": ["command"]}},
        {"path": "web/web_search", "name": "web_search", "group": "web",
         "description": "通用网页搜索。",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string"}, "count": {"type": "integer"},
         }, "required": ["query"]}},
        {"path": "task/taskplan_write", "name": "taskplan_write", "group": "task",
         "description": "创建或整体更新任务规划。",
         "parameters": {"type": "object", "properties": {
             "steps": {"type": "string"}, "title": {"type": "string"},
         }, "required": ["steps"]}},
    ]


def _by_path(tools: list[dict]) -> dict[str, dict]:
    return {t["path"]: t for t in tools}


# ---------------------------------------------------------------- 参数校验
def test_validate_args_accepts_valid():
    """必填齐全、类型正确 → 放行."""
    tool = _by_path(_tools())["workspace/fs_manage"]
    validate_args(tool, {"action": "read", "path": "docs/x.md"})


def test_validate_args_rejects_missing_required():
    """缺必填 → 报错（不能让缺参样本进训练集）."""
    tool = _by_path(_tools())["workspace/fs_manage"]
    with pytest.raises(ValueError, match="必填"):
        validate_args(tool, {"path": "docs/x.md"})


def test_validate_args_rejects_unknown_key():
    """幻觉参数 → 报错（运行时会把它们白名单过滤掉，训练不该教）."""
    tool = _by_path(_tools())["workspace/fs_manage"]
    with pytest.raises(ValueError, match="未声明"):
        validate_args(tool, {"action": "read", "tool_path": "x"})


def test_validate_args_bool_not_integer():
    """bool 是 int 的子类，必须单独挡掉（否则 True 会被当成 timeout=1）."""
    tool = _by_path(_tools())["workspace/shell_exec"]
    with pytest.raises(ValueError, match="bool"):
        validate_args(tool, {"command": "ls", "timeout": True})


def test_validate_args_type_mismatch():
    """类型不符 → 报错."""
    tool = _by_path(_tools())["workspace/shell_exec"]
    with pytest.raises(ValueError, match="integer"):
        validate_args(tool, {"command": "ls", "timeout": "5"})


def test_sample_args_raises_on_missing_var():
    """模板变量没给值必须抛（静默留着 `$file` 会污染训练集）."""
    with pytest.raises(KeyError):
        sample_args({"path": "x"}, {"path": "$file"}, {})


# ---------------------------------------------------------------- 渲染
def test_dump_call_is_compact_single_line():
    """输出形态：紧凑单行 JSON（换行/空格都是要模型浪费 token 学的噪声）."""
    text = dump_call("workspace/fs_manage", {"action": "read", "path": "a.md"})
    assert "\n" not in text
    assert ": " not in text and ", " not in text
    assert json.loads(text) == {"path": "workspace/fs_manage",
                                "arguments": {"action": "read", "path": "a.md"}}


def test_dump_call_keeps_chinese_readable():
    """中文不转义（字符级词表里 \\uXXXX 要烧 6 个 token）."""
    text = dump_call("pcr/pcr_homework", {"stage": "D面"})
    assert "D面" in text


def test_render_params_marks_required_and_default():
    """参数表必须标出必填与默认值 —— 模型靠这个判断能不能省略."""
    tool = _by_path(_tools())["workspace/fs_manage"]
    text = render_params(tool)
    assert "action: string（必填）" in text
    assert "limit: integer（可选）" in text
    assert "list/read/write" in text          # 参数语义（枚举含义）必须在


def test_render_params_no_args():
    """零参数工具也要能渲染（不能出现空列表/报错）."""
    text = render_params({"path": "task/taskplan_get", "description": "看计划",
                          "parameters": {"type": "object", "properties": {}}})
    assert "参数：无" in text


def test_render_catalog_excludes_target():
    """FILL 形态下目标工具不进目录（它的详情已在下方，重复只是烧 token）."""
    tools = _tools()
    text = render_catalog(tools, exclude="web/web_search")
    assert "web/web_search" not in text
    assert "workspace/fs_manage" in text


# ---------------------------------------------------------------- 候选集
def test_pick_distractors_contains_target_and_size():
    """候选集必须含目标，且规模在区间内（太小无干扰、太长吃满上下文）."""
    tools = _tools()
    target = _by_path(tools)["web/web_search"]
    import random

    rng = random.Random(0)
    for _ in range(50):
        picked = pick_distractors(tools, target, rng, 1, 2)
        paths = [t["path"] for t in picked]
        assert "web/web_search" in paths
        assert len(set(paths)) == len(paths)       # 不重复
        assert 2 <= len(picked) <= 3               # 至少 1 个干扰


def test_pick_distractors_position_varies():
    """目标位置必须随机（固定第一位会让模型只学"取第一条"）."""
    tools = _tools()
    target = _by_path(tools)["web/web_search"]
    import random

    rng = random.Random(1)
    spots = set()
    for _ in range(100):
        picked = pick_distractors(tools, target, rng, 1, 2)
        spots.add([t["path"] for t in picked].index("web/web_search"))
    assert len(spots) > 1


# ---------------------------------------------------------------- 样本结构
def _sample_text(sample: dict, idx: int) -> str:
    return sample["messages"][idx]["content"]


def test_build_select_has_empty_arguments():
    """SELECT 只选工具，不填参数（参数表还没给，填了就是幻觉）."""
    tools = _tools()
    target = _by_path(tools)["web/web_search"]
    import random

    s = build_select(target, "搜一下 LLM", tools, random.Random(0), 1, 2)
    body = json.loads(_sample_text(s, 2))
    assert body["path"] == "web/web_search"
    assert body["arguments"] == {}
    assert "🔧" not in _sample_text(s, 0)        # 没有参数表


def test_build_fill_has_params_block_and_valid_args():
    """FILL 必须带目标工具参数表，输出参数要过 schema 校验."""
    tools = _tools()
    target = _by_path(tools)["workspace/fs_manage"]
    import random

    s = build_fill(target, "读一下 a.md", {"action": "read", "path": "a.md"},
                   tools, random.Random(0), 1, 2)
    system = _sample_text(s, 0)
    assert "🔧 工具「workspace/fs_manage」" in system
    body = json.loads(_sample_text(s, 2))
    validate_args(target, body["arguments"])


def test_build_negative_emits_empty_path():
    """负样本：空 path（明确"不调用"），不是硬套一个工具."""
    tools = _tools()
    import random

    s = build_negative("帮我翻译成法语", tools, random.Random(0), 1, 2)
    assert json.loads(_sample_text(s, 2))["path"] == NO_TOOL


def test_sample_messages_shape():
    """样本是 messages 三件套：system/user/assistant，顺序固定."""
    tools = _tools()
    import random

    s = build_select(_by_path(tools)["task/taskplan_write"], "列个计划", tools,
                     random.Random(0), 1, 2)
    assert [m["role"] for m in s["messages"]] == ["system", "user", "assistant"]
    assert s["version"] == 1


# ---------------------------------------------------------------- 场景自检
def test_check_scenarios_catches_undeclared_var():
    """场景里写了 `$nope` 但没声明 → 生成前就报错（不是跑到那一轮才炸）."""
    tools = _by_path(_tools())
    bad = [{"tool": "web/web_search", "req": ["搜 $nope"], "vars": {},
            "args": {"query": "x"}}]
    with pytest.raises(SystemExit, match="未声明的变量"):
        check_scenarios(bad, tools)


def test_check_scenarios_catches_missing_required():
    """场景漏填必填参数 → 报错（否则每条样本都会被 validate_args 拦下）."""
    tools = _by_path(_tools())
    bad = [{"tool": "web/web_search", "req": ["搜东西"], "vars": {}, "args": {}}]
    with pytest.raises(SystemExit, match="缺必填参数"):
        check_scenarios(bad, tools)


def test_check_scenarios_catches_type_mismatch():
    """模板值只能填 string 型参数（`$var` 渲染出来永远是 str）."""
    tools = _by_path(_tools())
    bad = [{"tool": "workspace/shell_exec", "req": ["跑命令"],
            "vars": {"n": ["5"]}, "args": {"command": "ls", "timeout": "$n"}}]
    with pytest.raises(SystemExit, match="不能用 \\$模板"):
        check_scenarios(bad, tools)


def test_check_scenarios_catches_unknown_param():
    """参数名拼错 → 报错（运行时会被白名单弹掉，训练不该教）."""
    tools = _by_path(_tools())
    bad = [{"tool": "workspace/shell_exec", "req": ["跑"], "vars": {},
            "args": {"cmd": "ls"}}]
    with pytest.raises(SystemExit):
        check_scenarios(bad, tools)


# ---------------------------------------------------------------- 真实场景库
def test_builtin_scenarios_are_consistent():
    """内置场景库必须过自检（防有人改了模板忘了补 vars）."""
    if not SCHEMA_PATH.exists():
        pytest.skip("data/tool_schemas.json 不在（data/ 不入库）")
    from synth_tool_sft import load_schemas

    tools = load_schemas(SCHEMA_PATH)
    check_scenarios(SCENARIOS, _by_path(tools))


def test_builtin_scenarios_cover_main_groups():
    """场景库要覆盖主要分组（只训 workspace 的话，news/trigger 一样不会调）."""
    groups = {s["tool"].split("/")[0] for s in SCENARIOS}
    for need in ("workspace", "web", "task", "news", "trigger", "memory",
                 "agent", "pcr"):
        assert need in groups, f"场景库缺少 {need} 分组"


def test_negative_requests_are_universally_impossible():
    """负样本请求必须是"任何工具都做不到"的事（翻译/写诗/算术/闲聊）."""
    forbidden = ("搜索", "查", "读", "文件", "下载", "抓")
    for req in NEGATIVE_REQUESTS:
        assert not any(w in req for w in forbidden), f"负样本可能被工具完成：{req}"


# ---------------------------------------------------------------- 端到端
def test_generated_dataset_roundtrip_via_train_loader(tmp_path):
    """端到端：生成的 jsonl 能被训练侧读回，且 prompt 与推理模板逐字节一致.

    这条最关键 —— 它同时守住了「合成脚本的产物格式」和「训练/推理模板对齐」
    两件事，任何一边漂移都会红。
    """
    if not SCHEMA_PATH.exists():
        pytest.skip("data/tool_schemas.json 不在（data/ 不入库）")
    from synth_tool_sft import main as synth_main

    out = tmp_path / "tool_call.jsonl"
    synth_main(["--schemas", str(SCHEMA_PATH), "--out", str(out),
                "--n", "60", "--report", "0", "--seed", "7"])
    rows = [json.loads(line) for line in
            out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 60

    tools = {t["path"]: t for t in json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8"))["tools"]}
    from src.llm.local.data import iter_local_sft_pairs

    pairs = list(iter_local_sft_pairs(tmp_path))
    assert len(pairs) == len(rows)

    role_cn = {"system": "系统", "user": "用户", "assistant": "助手"}
    for row, (prompt, response) in zip(rows, pairs):
        msgs = row["messages"]
        expect = "\n".join(f"<{role_cn[m['role']]}>\n{m['content']}"
                           for m in msgs[:-1]) + "\n<助手>\n"
        # 与推理端 _prepare_ids 的拼法逐字节一致
        assert prompt == expect
        assert response == msgs[-1]["content"]
        # 输出必须是合法 JSON，且 path 要么空、要么在 schema 里
        body = json.loads(response)
        assert set(body) == {"path", "arguments"}
        assert body["path"] == NO_TOOL or body["path"] in tools
        # arguments 为空 = SELECT 样本（只选工具，参数表还没给），不算缺参
        if body["path"] and body["arguments"]:
            validate_args(tools[body["path"]], body["arguments"])


def test_generated_all_arguments_legal(tmp_path):
    """全部样本的参数都要过 schema 校验（不是抽查）."""
    if not SCHEMA_PATH.exists():
        pytest.skip("data/tool_schemas.json 不在（data/ 不入库）")
    from types import SimpleNamespace

    from synth_tool_sft import generate

    args = SimpleNamespace(
        schemas=str(SCHEMA_PATH), out="", n=300, fill_ratio=0.6, neg_ratio=0.12,
        real="", real_repeat=1, min_distract=3, max_distract=6,
        qq_prefix_ratio=0.1, seed=3, report=0,
    )
    samples, _stats = generate(args)
    tools = {t["path"]: t for t in json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8"))["tools"]}
    bad = 0
    for s in samples:
        body = json.loads(s["messages"][2]["content"])
        if not body["path"] or not body["arguments"]:
            continue        # 负样本 / SELECT 样本，不查参数
        try:
            validate_args(tools[body["path"]], body["arguments"])
        except ValueError:
            bad += 1
    assert bad == 0


def test_fill_args_come_from_request_context(tmp_path):
    """参数值必须来自上下文（请求里提到的路径要出现在 arguments.path 里）.

    这是「抄答案 vs 真读上下文」的分界线：如果参数值与请求无关，模型学到的
    只是"输出一个 JSON"，运行时就会编路径。
    """
    if not SCHEMA_PATH.exists():
        pytest.skip("data/tool_schemas.json 不在（data/ 不入库）")
    from types import SimpleNamespace

    from synth_tool_sft import generate

    args = SimpleNamespace(
        schemas=str(SCHEMA_PATH), out="", n=400, fill_ratio=1.0, neg_ratio=0.0,
        real="", real_repeat=1, min_distract=2, max_distract=3,
        qq_prefix_ratio=0.0, seed=5, report=0,
    )
    samples, _ = generate(args)
    checked = 0
    for s in samples:
        body = json.loads(s["messages"][2]["content"])
        user = s["messages"][1]["content"]
        tool_args = body["arguments"]
        path = tool_args.get("path")
        # 只查"请求里显式点名了路径"的样本
        if isinstance(path, str) and path and path in user:
            checked += 1
    assert checked > 30, f"能校验的样本太少（{checked}），随机化可能出了问题"
