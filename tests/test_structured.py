"""结构化输出（JSON Schema 约束解码）单测."""

import json

import pytest
import torch

from src.llm.local.structured import (
    JsonGrammar,
    JsonGrammarConstraint,
    JsonSchemaError,
    parse_response_format,
)


def feed(g: JsonGrammar, text: str):
    """逐字符推进；返回终态（非法则断言失败）."""
    st = g.initial()
    for ch in text:
        st = g.step(st, ch)
        assert st is not None, f"{ch!r} 在 {st} 下应合法"
    return st


def accepts(g: JsonGrammar, text: str) -> bool:
    """整串是否是**完整** JSON（accepting=True）."""
    st = g.initial()
    for ch in text:
        st = g.step(st, ch)
        if st is None:
            return False
    return g.accepting(st)


def ascii_tokens() -> list[str]:
    """id 0–3 为特殊 token，4 起映射到可打印 ASCII."""
    return ["", "", "", ""] + [chr(i) for i in range(32, 127)]


EOS = 2


def greedy(grammar: JsonGrammar, max_steps: int = 200) -> str:
    """确定性的"最省事"生成：结构性字符（引号/括号/逗号/冒号）优先.

    不加这个偏好，任意字符串会一直吐空格（空格 id 比结束引号小），
    永远等不到 EOS——这正说明 mask 真的在按语法走，而不是自由生成。
    """
    tokens = ascii_tokens()
    c = JsonGrammarConstraint(grammar, tokens, EOS)
    prefer = torch.zeros(len(tokens))
    for ch in '":,':
        prefer[tokens.index(ch)] = 100.0
    for ch in "}]":
        prefer[tokens.index(ch)] = 200.0   # 能关就先关，避免 ",," 死循环
    quote = tokens.index('"')
    out = []
    for _ in range(max_steps):
        logits = prefer
        if c.state[1] == "str":
            # 字符串内容：测试里强制立刻结束（否则 ']' 之类会被当普通字符）
            logits = torch.full_like(prefer, -1e9)
            logits[quote] = 10.0
        masked = c.mask_logits(logits)
        tid = int(masked.argmax())
        c.accept(tid)              # EOS 也要推进（finished=True）
        if tid == EOS:
            break
        out.append(tokens[tid])
    assert c.finished, "生成未在步数内自然结束"
    return "".join(out)


class TestBasic:
    def test_object_required(self):
        g = JsonGrammar({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        })
        assert accepts(g, '{"a":1}')
        assert accepts(g, '{"a":-0}')
        assert not accepts(g, '{"a":1')        # 未闭合
        assert not accepts(g, '{"a":"x"}')     # 类型错
        assert not accepts(g, '{"b":1}')       # 未知键
        assert not accepts(g, '{"a":1,}')      # 多余逗号

    def test_key_prefix(self):
        """一个键是另一个键的前缀（"a" / "ab"）时都能正常结束."""
        g = JsonGrammar({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "ab": {"type": "integer"}},
            "required": ["a", "ab"],
        })
        assert accepts(g, '{"a":1,"ab":2}')
        assert accepts(g, '{"ab":2,"a":1}')

    def test_optional_key(self):
        g = JsonGrammar({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": [],
        })
        assert accepts(g, "{}")
        assert accepts(g, '{"a":1}')

    def test_enum(self):
        g = JsonGrammar({"type": "string", "enum": ["cat", "car"]})
        assert accepts(g, '"cat"')
        assert accepts(g, '"car"')
        assert not accepts(g, '"ca"')          # 未闭合
        assert not accepts(g, '"cb"')          # 不在枚举

    def test_enum_prefix_siblings(self):
        """一个枚举值是另一个的前缀（"a" / "ab"）."""
        g = JsonGrammar({"type": "string", "enum": ["a", "ab"]})
        assert accepts(g, '"a"')
        assert accepts(g, '"ab"')

    def test_const(self):
        g = JsonGrammar({"const": "ok"})
        assert accepts(g, '"ok"')
        assert not accepts(g, '"no"')
        g2 = JsonGrammar({"const": 42})
        assert accepts(g2, "42")

    def test_array(self):
        g = JsonGrammar({"type": "array", "items": {"type": "integer"}})
        assert accepts(g, "[]")
        assert accepts(g, "[1,2,3]")
        assert not accepts(g, "[1,]")
        assert not accepts(g, "[1")

    def test_nested(self):
        g = JsonGrammar({
            "type": "object",
            "properties": {
                "xs": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"v": {"type": "boolean"}},
                    "required": ["v"],
                }},
            },
            "required": ["xs"],
        })
        assert accepts(g, '{"xs":[{"v":true},{"v":false}]}')
        assert not accepts(g, '{"xs":[{"v":1}]}')

    def test_literals(self):
        for text in ("true", "false", "null"):
            g = JsonGrammar({"type": "null" if text == "null" else "boolean"})
            assert accepts(g, text)
        g = JsonGrammar({"type": "boolean"})
        assert not accepts(g, "tru")
        assert not accepts(g, "truex")

    def test_numbers(self):
        g = JsonGrammar({"type": "number"})
        for text in ("0", "1", "-1", "1.5", "-0.25", "1e10", "1E+10", "2.5e-3"):
            assert accepts(g, text), text
        for text in ("01", "1.", "-", "1e", ".5", "+1", "1e+"):
            assert not accepts(g, text), text

    def test_integer_rejects_fraction(self):
        g = JsonGrammar({"type": "integer"})
        assert accepts(g, "-12")
        assert not accepts(g, "1.5")
        assert not accepts(g, "1e3")

    def test_any_json(self):
        g = JsonGrammar({})
        assert accepts(g, '{"x":[1,true,null,"s"]}')
        assert accepts(g, "[]")
        assert accepts(g, '"hi"')
        assert accepts(g, "-3.5e2")

    def test_accepting_timing(self):
        g = JsonGrammar({"type": "object", "properties": {}, "required": []})
        st = g.initial()
        assert not g.accepting(st)
        for ch in "{}":
            st = g.step(st, ch)
        assert st is not None and not g.accepting(g.initial())
        assert g.accepting(st)

    def test_whitespace_rejected(self):
        """紧凑 JSON：不做空白跳过（约束解码的常规取舍）."""
        g = JsonGrammar({"type": "object", "properties": {}, "required": []})
        assert not accepts(g, "{ }")


class TestSchemaValidation:
    @pytest.mark.parametrize("schema", [
        {"anyOf": [{"type": "string"}]},
        {"$ref": "#/definitions/x"},
        {"type": "object", "additionalProperties": True,
         "properties": {"a": {"type": "integer"}}},
        {"type": "array"},
        {"type": "weird"},
        {"enum": []},
        {"enum": [1, 2]},
        {"type": "object", "required": ["missing"], "properties": {}},
    ])
    def test_rejects_unsupported(self, schema):
        with pytest.raises(JsonSchemaError):
            JsonGrammar(schema)

    def test_rejects_non_dict(self):
        with pytest.raises(JsonSchemaError):
            JsonGrammar([1, 2])


class TestMask:
    def test_eos_only_when_accepting(self):
        g = JsonGrammar({"type": "integer"})
        tokens = ascii_tokens()
        c = JsonGrammarConstraint(g, tokens, EOS)
        # 初始：EOS 不允许
        m = c.mask_logits(torch.zeros(len(tokens)))
        assert not torch.isfinite(m[EOS])
        # 读完一个具体数字后：EOS 允许
        c.accept(tokens.index("7"))
        m = c.mask_logits(torch.zeros(len(tokens)))
        assert torch.isfinite(m[EOS])

    def test_illegal_chars_masked(self):
        g = JsonGrammar({"type": "integer"})
        tokens = ascii_tokens()
        c = JsonGrammarConstraint(g, tokens, EOS)
        m = c.mask_logits(torch.zeros(len(tokens)))
        assert torch.isfinite(m[tokens.index("1")])
        assert not torch.isfinite(m[tokens.index("a")])
        assert not torch.isfinite(m[tokens.index("[")])

    def test_multi_char_token_rejected_if_breaks_grammar(self):
        """多字符 token 必须整串合法（"12" 可以，"1a" 不行）."""
        g = JsonGrammar({"type": "integer"})
        tokens = ["", "", "", "", "12", "1a", "1", "2"]
        c = JsonGrammarConstraint(g, tokens, 2)
        m = c.mask_logits(torch.zeros(len(tokens)))
        assert torch.isfinite(m[4])       # "12"
        assert not torch.isfinite(m[5])   # "1a"
        assert torch.isfinite(m[6])       # "1"

    def test_mask_does_not_mutate_input(self):
        g = JsonGrammar({"type": "integer"})
        c = JsonGrammarConstraint(g, ascii_tokens(), EOS)
        logits = torch.zeros(len(c.tokens))
        before = logits.clone()
        c.mask_logits(logits)
        assert torch.equal(logits, before)


class TestEndToEnd:
    def test_greedy_generates_valid_json(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age"],
        }
        text = greedy(JsonGrammar(schema))
        data = json.loads(text)               # 语法必须合法
        assert isinstance(data["name"], str)
        assert isinstance(data["age"], int)
        assert set(data) >= {"name", "age"}

    def test_greedy_enum(self):
        text = greedy(JsonGrammar({"type": "string", "enum": ["yes", "no"]}))
        assert text in ('"yes"', '"no"')

    def test_greedy_nested(self):
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {
                "ok": {"type": "boolean"}}, "required": ["ok"]},
        }
        text = greedy(JsonGrammar(schema))
        data = json.loads(text)
        assert isinstance(data, list)


class TestBackendIntegration:
    """接进 LocalChatBackend：约束必须让真实生成循环产出合法 JSON."""

    @staticmethod
    def _backend():
        from src.llm.local.config import tiny_test_config
        from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
        from src.llm.local.model import TinyLLM

        torch.manual_seed(0)
        tok = SimpleTokenizer(256)
        # 字符级分词器必须覆盖 JSON 结构字符，否则约束退化为"无法生成"
        tok.fit(["你好世界", "hello world", "摸鱼使我快乐",
                 '{}[]":,.-+0123456789truefalsn ok go '])
        return LocalChatBackend(TinyLLM(tiny_test_config()), tok)

    def test_chat_with_schema(self):
        backend = self._backend()
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        resp = backend.chat(
            [{"role": "user", "content": "你好"}],
            max_new_tokens=16,
            response_format={"type": "json_schema",
                             "json_schema": {"schema": schema}},
        )
        data = json.loads(resp["content"])      # 约束保证可解析
        assert isinstance(data["ok"], bool)
        assert resp["finish_reason"] == "stop"

    def test_stream_with_enum(self):
        backend = self._backend()
        pieces = list(backend.stream(
            [{"role": "user", "content": "你好"}],
            max_new_tokens=16,
            response_format={"type": "json_schema",
                             "json_schema": {"schema": {
                                 "type": "string", "enum": ["go"]}}},
        ))
        assert json.loads("".join(pieces)) == "go"

    def test_chat_without_constraint_unchanged(self):
        """response_format=None 时与老路径一致（不构造自动机）."""
        backend = self._backend()
        resp = backend.chat([{"role": "user", "content": "你好"}],
                            max_new_tokens=4)
        assert resp["finish_reason"] in ("stop", "length")

    def test_bad_schema_raises(self):
        backend = self._backend()
        with pytest.raises(JsonSchemaError):
            backend.chat([{"role": "user", "content": "你好"}],
                         max_new_tokens=4,
                         response_format={"type": "anyOf"})


class TestParseResponseFormat:
    def test_none_and_text(self):
        assert parse_response_format(None) is None
        assert parse_response_format({"type": "text"}) is None

    def test_json_object(self):
        assert parse_response_format({"type": "json_object"}) == {}

    def test_json_schema(self):
        sch = {"type": "object", "properties": {}, "required": []}
        assert parse_response_format(
            {"type": "json_schema", "json_schema": {"schema": sch}}) == sch
        assert parse_response_format(
            {"type": "json_schema", "json_schema": sch}) == sch

    def test_rejects_unknown(self):
        with pytest.raises(JsonSchemaError):
            parse_response_format({"type": "xml"})
        with pytest.raises(JsonSchemaError):
            parse_response_format({"type": "json_schema", "json_schema": 3})
