"""可控推理努力单测 — 指令注入 / 惩罚数学 / 同档分组优势 / 服务接线.

四条硬契约：
1. 指令文本与论文**逐字**一致，且注入**幂等**（重试不会叠两行）；
2. `effort=None` 时 prompt 与老路径逐位一致（老客户端零影响）；
3. 惩罚系数随 b **指数衰减**、长度惩罚**有上限**（eq.11/12）；
4. 优势只在 `(prompt, b)` 同档内中心化——跨档比较会把"更长"当成"更好"。
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.effort import (EFFORT_MAX, EFFORT_MIN, EFFORT_TIERS,
                                  apply_effort, apply_effort_messages,
                                  average_spacing, clamp_effort, effort_instruction,
                                  effort_to_tier, group_advantages, group_key,
                                  length_penalty, parse_effort_levels,
                                  penalty_coefficient, predicted_length,
                                  resolve_effort, tau_from_levels, tier_to_effort,
                                  validate_effort)
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
from src.llm.local.model import TinyLLM

INSTRUCTION = ("Reasoning Effort: {b} (range 1–100; higher values request more "
               "thorough reasoning)")


# ------------------------------------------------------------ 校验 / 档位
def test_validate_effort_bounds():
    """1/100 合法；0/101/负数越界报错（不静默截断）."""
    assert validate_effort(1) == EFFORT_MIN
    assert validate_effort(100) == EFFORT_MAX
    assert validate_effort(50.0) == 50
    for bad in (0, 101, -1, 50.5, "50", None, True, [50]):
        with pytest.raises(ValueError):
            validate_effort(bad)


def test_clamp_effort_is_lenient_for_serving():
    """部署侧宽容：截断 + 四舍五入；无法解析退回 low."""
    assert clamp_effort(-5) == 1
    assert clamp_effort(1000) == 100
    assert clamp_effort(50.6) == 51
    assert clamp_effort("abc") == EFFORT_TIERS["low"]
    assert clamp_effort(float("nan")) == EFFORT_TIERS["low"]


def test_tier_mapping_matches_table2():
    """Table 2：low=50 / high=75 / max=100；非档位值返回 None（插值档）."""
    assert (EFFORT_TIERS["low"], EFFORT_TIERS["high"], EFFORT_TIERS["max"]) == (
        50, 75, 100)
    assert tier_to_effort("low") == 50
    assert tier_to_effort("HIGH") == 75
    assert tier_to_effort("max") == 100
    assert tier_to_effort(60) == 60
    assert tier_to_effort("60") == 60
    assert effort_to_tier(50) == "low"
    assert effort_to_tier(75) == "high"
    assert effort_to_tier(100) == "max"
    assert effort_to_tier(60) is None
    with pytest.raises(ValueError, match="未知努力档"):
        tier_to_effort("ultra")


def test_resolve_effort_none_and_blank():
    """None/空串 = 不改提示（老路径）."""
    assert resolve_effort(None) is None
    assert resolve_effort("") is None
    assert resolve_effort("  ") is None
    assert resolve_effort("low") == 50


def test_parse_effort_levels():
    """档位列表：数字/档名混用、去重升序；空 → 空列表."""
    assert parse_effort_levels("25,50,75,100") == [25, 50, 75, 100]
    assert parse_effort_levels("low,high,max") == [50, 75, 100]
    assert parse_effort_levels("100, 25,50,25") == [25, 50, 100]
    assert parse_effort_levels("") == []
    assert parse_effort_levels(None) == []
    assert parse_effort_levels([50, 100]) == [50, 100]
    with pytest.raises(ValueError):
        parse_effort_levels("25,abc")


# ------------------------------------------------------------ 条件化
def test_instruction_is_verbatim():
    """指令文本与论文一致（换措辞 = 换了个没训过的条件）."""
    assert effort_instruction(75) == INSTRUCTION.format(b=75)
    assert "range 1–100" in effort_instruction(1)


def test_apply_effort_is_idempotent():
    """注入幂等：连注两次 == 注一次（重试不会叠两行）."""
    once = apply_effort("你是助手。", 75)
    twice = apply_effort(once, 75)
    assert once == twice
    # 换档位 = 替换而不是追加
    other = apply_effort(once, 25)
    assert other.count("Reasoning Effort:") == 1
    assert "Reasoning Effort: 25" in other
    assert "你是助手。" in other


def test_apply_effort_empty_system():
    """空系统提示：只留指令行（不留空行）."""
    assert apply_effort("", 50) == INSTRUCTION.format(b=50)
    assert apply_effort(None, 50) == INSTRUCTION.format(b=50)


def test_apply_effort_messages_no_mutation_and_insert():
    """messages：不改入参；无 system 时插一条；有 system 时替换其内容."""
    msgs = [{"role": "user", "content": "你好"}]
    out = apply_effort_messages(msgs, 75)
    assert msgs == [{"role": "user", "content": "你好"}]          # 未被修改
    assert out[0]["role"] == "system"
    assert "Reasoning Effort: 75" in out[0]["content"]
    assert out[1] == msgs[0]

    msgs2 = [{"role": "system", "content": "老系统提示"},
             {"role": "user", "content": "hi"}]
    out2 = apply_effort_messages(msgs2, 50)
    assert out2[0]["content"].startswith("Reasoning Effort: 50")
    assert "老系统提示" in out2[0]["content"]
    assert len(out2) == 2
    assert msgs2[0]["content"] == "老系统提示"


# ------------------------------------------------------------ 惩罚数学
def test_penalty_coefficient_decays_exponentially():
    """k(b) = k0·exp(-(b-bmin)/τ)：bmin 处 = k0，bmin+τ 处 = k0/e，单调不增."""
    assert penalty_coefficient(1, k0=2.0, b_min=1, tau=25) == pytest.approx(2.0)
    assert penalty_coefficient(26, k0=2.0, b_min=1, tau=25) == pytest.approx(
        2.0 / math.e)
    vals = [penalty_coefficient(b, k0=1.0, b_min=1, tau=25)
            for b in range(1, 101, 5)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    with pytest.raises(ValueError):
        penalty_coefficient(50, tau=0)
    with pytest.raises(ValueError):
        penalty_coefficient(50, k0=-1)


def test_length_penalty_is_negative_capped_and_monotone():
    """长度惩罚：非正、随长度单调变差、触顶后不再增长（eq.11 的 min(C_max, ·)）."""
    kw = dict(k0=1.0, b_min=1, tau=25, l_norm=100.0, cap=0.8)
    assert length_penalty(0, 50, **kw) == 0.0
    p1 = length_penalty(50, 50, **kw)
    p2 = length_penalty(1000, 50, **kw)
    assert p1 < 0 and p2 <= p1
    assert p2 == pytest.approx(-0.8)              # 触顶（k(50)·1000/100 已远超 cap）
    # 触顶后再加 token 不改变惩罚（边际惩罚为零）
    assert length_penalty(1000, 50, **kw) == pytest.approx(-0.8)
    # 低档惩罚更重（同等长度）
    assert length_penalty(100, 25, **kw) < length_penalty(100, 100, **kw)
    with pytest.raises(ValueError):
        length_penalty(-1, 50, **kw)
    with pytest.raises(ValueError):
        length_penalty(10, 50, l_norm=0)


def test_predicted_length_is_affine_in_effort():
    """eq.16/17：最优长度对 b 仿射，斜率 = s_x·τ，二阶差分 ≈ 0."""
    s, tau = 30.0, 4.0
    pts = [predicted_length(b, s_x=s, k0=1.0, b_min=1, tau=tau)
           for b in (5, 30, 55, 80)]
    d1 = [b - a for a, b in zip(pts, pts[1:])]
    assert all(abs(x - s * tau * 25) < 1e-9 for x in d1)
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    assert all(abs(x) < 1e-9 for x in d2)


def test_tau_from_levels():
    """τ = λ·平均间隔；只有一档时退回 25（不除零）."""
    assert average_spacing([25, 50, 75, 100]) == pytest.approx(25.0)
    assert average_spacing([50]) == 0.0
    assert tau_from_levels([25, 50, 75, 100], lam=1.0) == pytest.approx(25.0)
    assert tau_from_levels([25, 50, 75, 100], lam=0.5) == pytest.approx(12.5)
    assert tau_from_levels([75]) == 25.0
    with pytest.raises(ValueError):
        tau_from_levels([25, 50], lam=0)


# ------------------------------------------------------------ 分组优势
def test_group_advantages_same_group_mean_centered():
    """同组内优势均值为 0；数值 = 奖励减组均值."""
    rewards = [1.0, 0.0, 0.5]
    adv = group_advantages(rewards, ["a", "a", "a"])
    assert sum(adv) == pytest.approx(0.0)
    assert adv == pytest.approx([0.5, -0.5, 0.0])


def test_group_advantages_do_not_compare_across_efforts():
    """跨档不比较：把整组奖励同时 +100，优势不变（组均值一起平移）."""
    base = group_advantages([1.0, 0.0, 0.5], [(0, 25)] * 3)
    shifted = group_advantages([101.0, 100.0, 100.5], [(0, 25)] * 3)
    assert base == pytest.approx(shifted)
    # 不同档（同 prompt）各自中心化，互不影响
    rewards = [1.0, 0.0, 0.9, 0.2]
    groups = [group_key(7, 25), group_key(7, 25), group_key(7, 100), group_key(7, 100)]
    adv = group_advantages(rewards, groups)
    assert adv[0:2] == pytest.approx([0.5, -0.5])
    assert adv[2:4] == pytest.approx([0.35, -0.35])


def test_group_advantages_singleton_is_zero():
    """组内只有一个样本：没有比较对象，优势 0（给绝对值会引入跨档尺度）."""
    adv = group_advantages([3.0, 1.0, 0.0], ["solo", "g", "g"])
    assert adv[0] == 0.0
    assert adv[1:3] == pytest.approx([0.5, -0.5])


def test_group_advantages_normalize_std_option():
    """可选除以组内标准差（GRPO 常见写法；默认关闭以对齐论文的 mean-centered）."""
    adv = group_advantages([1.0, -1.0], ["g", "g"], normalize_std=True)
    assert adv == pytest.approx([1.0, -1.0])
    # 全相同 → 标准差 0 → 优势 0（不产生 NaN）
    assert group_advantages([2.0, 2.0], ["g", "g"], normalize_std=True) == [0.0, 0.0]


def test_group_advantages_validation():
    """长度不一致 / 空输入."""
    with pytest.raises(ValueError, match="长度不一致"):
        group_advantages([1.0], ["a", "b"])
    assert group_advantages([], []) == []


# ------------------------------------------------------------ 接线
def _backend():
    """tiny 模型 + 字符分词器（CPU 秒级）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config()).eval()
    tok = SimpleTokenizer(256)
    # 字符级分词器要见过这些字符，否则指令里的 ASCII 会被当 OOV 丢掉
    tok.fit(["你好世界", "摸鱼使我快乐", "用户助手系统",
             "".join(chr(c) for c in range(32, 127)),
             f"Reasoning Effort: 75 (range 1–100; higher values request more "
             f"thorough reasoning)"])
    return LocalChatBackend(m, tok), tok


def test_prepare_ids_injects_instruction_only_when_asked():
    """`effort=None` 与老路径逐位一致；给了档位才多出指令行."""
    backend, tok = _backend()
    msgs = [{"role": "user", "content": "你好"}]
    ids_off, _ = backend._prepare_ids(msgs, 8)
    ids_off2, _ = backend._prepare_ids(msgs, 8, None)
    assert ids_off.tolist() == ids_off2.tolist()
    assert "Reasoning Effort" not in tok.decode(ids_off[0].tolist())

    ids_on, _ = backend._prepare_ids(msgs, 8, 75)
    text = tok.decode(ids_on[0].tolist())
    assert "Reasoning Effort: 75" in text
    assert "thorough reasoning" in text
    assert "<用户>" in text                    # 原消息仍在


def test_chat_accepts_tier_string():
    """chat 端到端接受档名（low/high/max），返回结构不变."""
    backend, _ = _backend()
    resp = backend.chat([{"role": "user", "content": "你好"}],
                        max_new_tokens=4, effort="high")
    assert set(resp) == {"content", "reasoning", "tool_calls", "finish_reason"}
    with pytest.raises(ValueError):
        backend.chat([{"role": "user", "content": "你好"}],
                     max_new_tokens=4, effort="ultra")


def test_server_params_parse_effort(tmp_path):
    """服务层：`reasoning_effort` 与 `effort` 等价，缺省 None."""
    from src.llm.local.server import OpenAIServer

    backend, _ = _backend()
    srv = OpenAIServer(backend)
    out = srv._chat_params({"messages": [], "reasoning_effort": "max"})
    assert out[-1] == "max"
    assert srv._chat_params({"messages": [], "effort": 60})[-1] == 60
    assert srv._chat_params({"messages": []})[-1] is None


def test_server_rejects_bad_effort_with_400(tmp_path):
    """非法档位走 400（参数错误不该冒成 500）；合法档位正常出流."""
    import json
    import threading
    import time
    import urllib.error
    import urllib.request

    from src.llm.local.server import OpenAIServer

    backend, _ = _backend()
    server = OpenAIServer(backend)
    threading.Thread(target=server.serve_forever,
                     kwargs={"host": "127.0.0.1", "port": 0},
                     daemon=True).start()
    for _ in range(100):
        if server._httpd is not None:
            break
        time.sleep(0.05)
    port = server._httpd.server_address[1]
    try:
        def post(obj):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(obj).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read())

        status, _ = post({"messages": [{"role": "user", "content": "你好"}],
                          "max_tokens": 4, "reasoning_effort": 75})
        assert status == 200
        try:
            post({"messages": [{"role": "user", "content": "你好"}],
                  "max_tokens": 4, "reasoning_effort": "ultra"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            assert "未知努力档" in json.loads(exc.read())["error"]["message"]
        else:                                     # pragma: no cover
            raise AssertionError("非法档位应返回 400")
    finally:
        server.shutdown()
