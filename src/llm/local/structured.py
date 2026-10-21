"""结构化输出（JSON Schema 约束解码）— 对标 DeepInfer `structure_output`.

DeepInfer 的做法是"离线条目编译成自动机 + 解码每步求合法 token 掩码"，依赖
`xgrammar`（C++ 绑定）。本模块用纯 Python 复刻同一思路的**最小可用子集**，
不引入新依赖，适配本仓库的字符级分词器：

```
JSON Schema ──编译──> 字符级增量自动机（不可变状态）──每步──> 合法字符集
                                                              │
                                      vocab 里首字符命中的 token → logits mask
```

支持的模式（未列出的直接报错，不做静默降级）：

- `{"type":"object","properties":{...},"required":[...],"additionalProperties":false}`
- `{"type":"array","items":{...}}`
- `{"type":"string"}` / `{"type":"string","enum":[...]}` / `{"const":...}`
- `{"type":"number"}` / `{"type":"integer"}`
- `{"type":"boolean"}` / `{"type":"null"}`
- `{}`（任意 JSON 值）

生成**紧凑 JSON**（不允许空白），这是约束解码的常规取舍：省一个状态维度，
且对下游 `json.loads` 无影响。

为什么必须用自动机而不是"先自由生成再解析"：后者在长输出上重试成本高，且
无法保证语法；约束解码把"合法"变成每一步的 mask，**贪心/采样都只能落在语法内**。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "JsonSchemaError",
    "JsonGrammar",
    "JsonGrammarConstraint",
    "parse_response_format",
]

# 数字的增量 DFA 状态：'-' 之后、整数、小数点、小数、指数……
_NUM_EXPAND = {
    "neg": "0123456789",
    "zero": ".eE",
    "int": "0123456789.eE",
    "dot": "0123456789",
    "frac": "0123456789eE",
    "exp": "+-0123456789",
    "exp_sign": "0123456789",
    "exp_digits": "0123456789",
}
# 哪些数字状态已经构成合法 JSON number（可以结束）
_NUM_ACCEPT = {"zero", "int", "frac", "exp_digits"}

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}
_HEX = "0123456789abcdefABCDEF"
# 通配哨兵：allowed() 里出现它 = "任意非控制字符都行"（词表可能含中文，
# 自动机不该替分词器假设字符集）。mask 侧看到它就逐字符用 step() 校验。
ANY = "\x00"
_PRINTABLE = frozenset(chr(i) for i in range(32, 127))


class JsonSchemaError(ValueError):
    """Schema 不被本模块支持（显式报错，不静默降级）."""


class JsonGrammar:
    """JSON Schema → 字符级增量自动机（状态不可变，可安全复制）.

    状态用 `(stack, mode, data)` 元组表示：

    - `stack`：嵌套帧，`("obj", schema, done_keys, cur_key)` 或 `("arr", schema)`；
    - `mode`：当前期望，`value` / `obj_key` / `obj_colon` / `obj_comma` /
      `arr_value` / `arr_comma` / `str` / `num` / `lit` / `done`；
    - `data`：该模式的附加信息（schema、已读前缀、转义状态等）。

    所有方法都是纯函数式：`step()` 返回新状态，失败返回 `None`，不修改入参。
    """

    def __init__(self, schema: dict | None):
        """编译 schema（结构校验在编译期做完，运行期不再抛错）."""
        if schema is None:
            schema = {}
        if not isinstance(schema, dict):
            raise JsonSchemaError(f"schema 须为对象，实际 {type(schema).__name__}")
        self.schema = schema
        self._check(schema)
        self._root = schema

    # ------------------------------------------------------------ 编译期校验
    def _check(self, node: dict, path: str = "$") -> None:
        """递归校验支持的模式（提前失败，避免解码到一半才发现）."""
        if not isinstance(node, dict):
            raise JsonSchemaError(f"{path} 须为对象")
        if "anyOf" in node or "oneOf" in node or "allOf" in node \
                or "not" in node or "$ref" in node:
            raise JsonSchemaError(f"{path} 不支持组合 schema（anyOf/oneOf/allOf/$ref）")
        if "const" in node:
            if not isinstance(node["const"], (str, int, float, bool)) \
                    and node["const"] is not None:
                raise JsonSchemaError(f"{path}.const 只支持标量")
            return
        if "enum" in node:
            vals = node["enum"]
            if not isinstance(vals, list) or not vals:
                raise JsonSchemaError(f"{path}.enum 须为非空数组")
            if not all(isinstance(v, str) for v in vals):
                raise JsonSchemaError(f"{path}.enum 只支持字符串枚举")
            return
        typ = node.get("type")
        if typ is None:
            return  # {} = 任意 JSON 值
        if typ == "object":
            props = node.get("properties", {})
            if not isinstance(props, dict):
                raise JsonSchemaError(f"{path}.properties 须为对象")
            for k, v in props.items():
                if not isinstance(k, str):
                    raise JsonSchemaError(f"{path}.properties 的键须为字符串")
                self._check(v, f"{path}.{k}")
            req = node.get("required", [])
            if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
                raise JsonSchemaError(f"{path}.required 须为字符串数组")
            for r in req:
                if r not in props:
                    raise JsonSchemaError(f"{path}.required 含未定义属性 {r!r}")
            if node.get("additionalProperties", False):
                raise JsonSchemaError(
                    f"{path}.additionalProperties=true 未实现（会破坏键的自动机）")
            return
        if typ == "array":
            items = node.get("items")
            if items is None:
                raise JsonSchemaError(f"{path}.items 必填（未实现无约束数组）")
            self._check(items, f"{path}[]")
            return
        if typ in ("string", "number", "integer", "boolean", "null"):
            return
        raise JsonSchemaError(f"{path}.type 不支持 {typ!r}")

    # ------------------------------------------------------------ 状态构造
    def initial(self) -> tuple:
        """初始状态：等待顶层值."""
        return ((), "value", self._root)

    # ------------------------------------------------------------ 允许字符
    def allowed(self, state: tuple) -> frozenset[str]:
        """当前状态允许的下一字符集合（用于构造 token mask）."""
        stack, mode, data = state
        if mode == "done":
            return frozenset()          # 只能出 EOS
        if mode == "value":
            return self._value_start_chars(data)
        if mode == "obj_key":
            return self._key_chars(data)
        if mode == "obj_colon":
            return frozenset(":")
        if mode == "obj_comma":
            schema, done = data
            out = {","} if self._has_new_key(schema, done) else set()
            if self._required_met(schema, done):
                out.add("}")
            return frozenset(out)
        if mode == "arr_value":
            schema, first = data
            out = set(self._value_start_chars(schema))
            if first:
                out.add("]")
            return frozenset(out)
        if mode == "arr_comma":
            return frozenset(",]")
        if mode == "str":
            return self._str_chars(data)
        if mode == "num":
            cur, is_int = data
            out = set(_NUM_EXPAND[cur])
            if is_int:
                out -= set(".eE")
            if cur in _NUM_ACCEPT:
                out |= set(self.allowed(self._finish_value(state)))
            return frozenset(out)
        if mode == "lit":
            rest, _ = data
            return frozenset(rest[0]) if rest else frozenset()
        raise JsonSchemaError(f"未知状态 {mode}")

    def _value_start_chars(self, schema: dict) -> frozenset[str]:
        """按 schema 给出一个值的首字符集合."""
        if "const" in schema:
            c = schema["const"]
            return frozenset(self._literal_chars(c))
        if "enum" in schema:
            return frozenset('"')
        typ = schema.get("type")
        if typ == "object":
            return frozenset("{")
        if typ == "array":
            return frozenset("[")
        if typ == "string":
            return frozenset('"')
        if typ in ("number", "integer"):
            return frozenset("-0123456789")
        if typ == "boolean":
            return frozenset("tf")
        if typ == "null":
            return frozenset("n")
        # {}：任意值
        return frozenset('{["-0123456789tfn')

    def _key_chars(self, data: tuple) -> frozenset[str]:
        """对象键位置：已知属性名的前缀集合；required 齐了才允许 '}'."""
        done, prefix, in_key, schema, allow_close = data
        props = schema.get("properties", {})
        free = not props                # 未声明 properties = 任意键
        out: set[str] = set()
        if not in_key:
            out.add('"')
            if allow_close and self._required_met(schema, done):
                out.add("}")
        else:
            if free:
                out |= set(_PRINTABLE) | {ANY}
            if prefix in props and prefix not in done:
                out.add('"')          # 恰好是完整键：允许结束（"a" 与 "ab" 并存）
            for k in props:
                if k not in done and k.startswith(prefix) and len(k) > len(prefix):
                    out.add(k[len(prefix)])
        return frozenset(out)

    def _str_chars(self, data: tuple) -> frozenset[str]:
        """字符串值：普通字符 + 结束引号 + 转义序列."""
        mode, prefix, esc, hexbuf, schema = data
        if mode == "escape":
            return frozenset(_ESCAPES) | frozenset("u")
        if mode == "unicode":
            return frozenset(_HEX) if len(hexbuf) < 4 else frozenset()
        out: set[str] = set(_PRINTABLE) | {ANY}
        # 字符串枚举/常量：只放行能让前缀继续匹配的字符
        choices = None
        if "const" in schema:
            choices = [str(schema["const"])]
        elif "enum" in schema:
            choices = list(schema["enum"])
        if choices is not None:
            out = {v[len(prefix)] for v in choices
                   if v.startswith(prefix) and len(v) > len(prefix)}
            if any(v == prefix for v in choices):
                out.add('"')          # 完整枚举值允许结束；更长的继续匹配
        return frozenset(out)

    # ------------------------------------------------------------ 状态推进
    def step(self, state: tuple, ch: str) -> tuple | None:
        """吃一个字符；非法返回 None（调用方据此拒绝该 token）."""
        stack, mode, data = state
        if mode == "done":
            return None
        if mode == "value":
            return self._start_value(stack, data, ch)
        if mode == "obj_key":
            return self._step_key(stack, data, ch)
        if mode == "obj_colon":
            if ch != ":":
                return None
            key_schema, done, key = data
            frame = stack[-1]
            stack2 = stack[:-1] + (("obj", frame[1], frame[2], key),)
            return (stack2, "value", key_schema)
        if mode == "obj_comma":
            schema, done = data
            if ch == ",":
                if not self._has_new_key(schema, done):
                    return None
                return (stack, "obj_key", (done, "", False, schema, False))
            if ch == "}":
                if not self._required_met(schema, done):
                    return None
                return self._close_container(stack)
            return None
        if mode == "arr_value":
            schema, first = data
            if ch == "]" and first:
                return self._close_container(stack)
            return self._start_value(stack, schema, ch, in_array=True)
        if mode == "arr_comma":
            schema = data
            if ch == ",":
                return (stack, "arr_value", (schema, False))
            if ch == "]":
                return self._close_container(stack)
            return None
        if mode == "str":
            return self._step_str(stack, data, ch)
        if mode == "num":
            return self._step_num(stack, data, ch)
        if mode == "lit":
            return self._step_lit(stack, data, ch)
        return None

    def _start_value(self, stack: tuple, schema: dict, ch: str,
                     in_array: bool = False) -> tuple | None:
        """开始一个值：按首字符分流到对象/数组/字符串/数字/字面量."""
        if "const" in schema:
            text = self._literal_text(schema["const"])
            if not text.startswith(ch):
                return None
            return self._lit_state(stack, text[1:], in_array)
        if "enum" in schema:
            if ch != '"':
                return None
            return (stack, "str", ("text", "", "", "", schema))
        typ = schema.get("type")
        if ch == "{":
            if typ not in (None, "object"):
                return None
            frame = ("obj", schema, frozenset(), None)
            return (stack + (frame,), "obj_key",
                    (frozenset(), "", False, schema, True))
        if ch == "[":
            if typ not in (None, "array"):
                return None
            items = schema.get("items") if typ == "array" else {}
            frame = ("arr", items)
            return (stack + (frame,), "arr_value", (items, True))
        if ch == '"':
            if typ not in (None, "string"):
                return None
            return (stack, "str", ("text", "", "", "", schema))
        if ch == "t":
            return self._lit_state(stack, "rue", in_array) if typ in (None, "boolean") else None
        if ch == "f":
            return self._lit_state(stack, "alse", in_array) if typ in (None, "boolean") else None
        if ch == "n":
            return self._lit_state(stack, "ull", in_array) if typ in (None, "null") else None
        if ch == "-" or ch.isdigit():
            if typ not in (None, "number", "integer"):
                return None
            nxt = "neg" if ch == "-" else ("zero" if ch == "0" else "int")
            return (stack, "num", (nxt, typ == "integer"))
        return None

    def _lit_state(self, stack: tuple, rest: str, in_array: bool) -> tuple:
        """字面量（true/false/null）或 const 剩余部分的中间状态."""
        return (stack, "lit", (rest, in_array))

    def _step_lit(self, stack: tuple, data: tuple, ch: str) -> tuple | None:
        """推进字面量；走完后结束该值."""
        rest, in_array = data
        if not rest.startswith(ch):
            return None
        rest = rest[1:]
        if rest:
            return (stack, "lit", (rest, in_array))
        return self._finish_value((stack, "lit", (rest, in_array)))

    def _step_key(self, stack: tuple, data: tuple, ch: str) -> tuple | None:
        """对象键：前缀匹配已知属性名；结束后进入冒号."""
        done, prefix, in_key, schema, allow_close = data
        props = schema.get("properties", {})
        free = not props
        if not in_key:
            if ch == "}" and allow_close and self._required_met(schema, done):
                return self._close_container(stack)   # 空对象
            if ch != '"':
                return None
            return (stack, "obj_key", (done, "", True, schema, allow_close))
        if ch == '"':
            if not free and (prefix not in props or prefix in done):
                return None
            return (stack, "obj_colon", (props.get(prefix, {}), done, prefix))
        if ch == "\\" or ch == "\n" or ch == "\r" or ch == "\t":
            return None  # 键名不支持转义/控制字符（属性名本来就是普通字符）
        prefix2 = prefix + ch
        if not free and not any(k.startswith(prefix2) for k in props):
            return None
        return (stack, "obj_key", (done, prefix2, True, schema, allow_close))

    def _step_str(self, stack: tuple, data: tuple, ch: str) -> tuple | None:
        """字符串值：普通字符 / 转义 / 结束引号."""
        _, prefix, esc, hexbuf, schema = data
        if esc == "escape":
            if ch == "u":
                return (stack, "str", ("text", prefix, "unicode", "", schema))
            if ch not in _ESCAPES:
                return None
            return (stack, "str", ("text", prefix + _ESCAPES[ch], "", "", schema))
        if esc == "unicode":
            if ch not in _HEX:
                return None
            hexbuf += ch
            if len(hexbuf) < 4:
                return (stack, "str", ("text", prefix, "unicode", hexbuf, schema))
            return (stack, "str", ("text", prefix + "?", "", "", schema))
        if ch == "\\":
            return (stack, "str", ("text", prefix, "escape", "", schema))
        if ch == '"':
            if "enum" in schema and prefix not in schema["enum"]:
                return None
            if "const" in schema and prefix != str(schema["const"]):
                return None
            return self._finish_value((stack, "str", data))
        if ch == "\n" or ch == "\r" or ch == "\t":
            return None  # 未转义的控制字符非法
        prefix2 = prefix + ch
        if "enum" in schema:
            if not any(v.startswith(prefix2) for v in schema["enum"]):
                return None
        elif "const" in schema:
            if not str(schema["const"]).startswith(prefix2):
                return None
        return (stack, "str", ("text", prefix2, "", "", schema))

    def _step_num(self, stack: tuple, data: tuple, ch: str) -> tuple | None:
        """数字 DFA；遇到非数字字符时先尝试结束当前数字."""
        cur, is_int = data
        expand = _NUM_EXPAND[cur]
        if is_int:
            expand = "".join(c for c in expand if c not in ".eE")
        if ch in expand:
            if cur == "neg":
                nxt = "zero" if ch == "0" else "int"
            elif cur == "zero":
                nxt = "dot" if ch == "." else "exp"
            elif cur == "int":
                nxt = "dot" if ch == "." else ("exp" if ch in "eE" else "int")
            elif cur == "dot":
                nxt = "frac"
            elif cur == "frac":
                nxt = "exp" if ch in "eE" else "frac"
            elif cur == "exp":
                nxt = "exp_sign" if ch in "+-" else "exp_digits"
            elif cur == "exp_sign":
                nxt = "exp_digits"
            else:
                nxt = "exp_digits"
            return (stack, "num", (nxt, is_int))
        # 当前数字已合法 → 先结束，再让下一状态处理该字符（如 ',' '}'）
        if cur in _NUM_ACCEPT:
            return self.step(self._finish_value((stack, "num", data)), ch)
        return None

    # ------------------------------------------------------------ 值完成/回父级
    def _finish_value(self, state: tuple) -> tuple:
        """一个值结束：回到父级的逗号位；栈空则进入 done."""
        stack, _, _ = state
        if not stack:
            return ((), "done", None)
        frame = stack[-1]
        if frame[0] == "obj":
            _, schema, done, key = frame
            done2 = done | {key}
            return (stack[:-1] + (("obj", schema, done2, None),),
                    "obj_comma", (schema, done2))
        _, items = frame
        return (stack[:-1] + (("arr", items),), "arr_comma", items)

    def _close_container(self, stack: tuple) -> tuple:
        """关闭当前对象/数组：pop 自己，回到父级的逗号位（栈空则 done）."""
        parent = stack[:-1]
        if not parent:
            return ((), "done", None)
        frame = parent[-1]
        if frame[0] == "obj":
            _, schema, done, key = frame
            done2 = done | {key}
            return (parent[:-1] + (("obj", schema, done2, None),),
                    "obj_comma", (schema, done2))
        _, items = frame
        return (parent[:-1] + (("arr", items),), "arr_comma", items)

    def _has_new_key(self, schema: dict, done: frozenset) -> bool:
        """是否还有未写过的属性（无 properties 时视为任意键，永远有）."""
        props = schema.get("properties", {})
        if not props:
            return True
        return any(k not in done for k in props)

    def _required_met(self, schema: dict, done: frozenset) -> bool:
        """required 是否已全部出现."""
        return all(r in done for r in schema.get("required", []))

    @staticmethod
    def _literal_text(v: Any) -> str:
        """const 值 → JSON 字面量文本（字符串带引号）."""
        import json

        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

    def _literal_chars(self, v: Any) -> str:
        """const 的首字符."""
        return self._literal_text(v)[:1]

    # ------------------------------------------------------------ 终态判断
    def accepting(self, state: tuple) -> bool:
        """是否已构成一个完整 JSON（可以出 EOS）.

        数字是唯一会"停在中途但仍合法"的原子：`1` 之后允许直接 EOS，
        不必等一个分隔符把状态推到 done。
        """
        _, mode, data = state
        if mode == "done":
            return True
        if mode == "num":
            if data[0] not in _NUM_ACCEPT:
                return False
            # 数字后可能还差 ',' / '}' / ']'，不能只看数字本身
            return self.accepting(self._finish_value(state))
        return False


class JsonGrammarConstraint:
    """把 `JsonGrammar` 接到生成循环：每步 mask logits，采样后推进状态.

    只支持单序列（`B=1`）；批量约束解码需要每序列一份自动机，本仓库暂不需要。
    """

    def __init__(self, grammar: JsonGrammar, tokens: list[str], eos_id: int,
                 vocab_size: int | None = None):
        """tokens: id → 字符串（特殊 token 留空串）；eos_id 单独传入."""
        self.grammar = grammar
        self.tokens = tokens
        self.eos_id = int(eos_id)
        self.vocab_size = int(vocab_size or len(tokens))
        self.state = grammar.initial()
        self.finished = False
        self.text = ""                    # 已接受的文本（调试/校验用）

    # ------------------------------------------------------------ 掩码
    def _build_mask(self, device) -> "object":
        """构造当前状态的合法 token 掩码（True=允许）."""
        import torch

        allowed = self.grammar.allowed(self.state)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        any_ok = ANY in allowed
        for tid, s in enumerate(self.tokens[:self.vocab_size]):
            if not s or tid == self.eos_id:
                continue
            if (any_ok or s[0] in allowed) and self._can_take(s):
                mask[tid] = True
        if self.grammar.accepting(self.state):
            mask[self.eos_id] = True
        return mask

    def check_available(self) -> None:
        """校验词表能继续生成（词表缺 JSON 结构字符时 schema 不可满足）."""
        if not bool(self._build_mask("cpu").any()):
            raise JsonSchemaError(
                "词表缺少当前 schema 所需的字符，无法做约束解码"
                "（字符级分词器需先 fit 出 JSON 语法字符）")

    def mask_logits(self, logits):
        """把非法 token 置 -inf；返回新张量（不原地改，避免污染调用方）."""
        vocab = int(logits.shape[-1])
        if vocab != self.vocab_size:
            # 防御：模型宽度与构造时不一致时按 logits 为准（多出的 id 一律非法）
            self.vocab_size = vocab
        mask = self._build_mask(logits.device)
        if not bool(mask.any()):
            raise RuntimeError(
                "约束解码无法继续：当前状态没有任何合法 token"
                "（词表缺少所需字符，或 schema 不可满足）")
        out = logits.clone()
        return out.masked_fill(~mask, float("-inf"))

    def _can_take(self, text: str) -> bool:
        """从当前状态能否完整吃下这个 token（多字符 token 必须整串合法）."""
        st = self.state
        for ch in text:
            st = self.grammar.step(st, ch)
            if st is None:
                return False
        return True

    # ------------------------------------------------------------ 推进
    def accept(self, token_id: int) -> None:
        """采样到 token 后推进自动机（与 mask 用同一套 step，保证一致）."""
        tid = int(token_id)
        if tid == self.eos_id:
            self.finished = True
            return
        s = self.tokens[tid] if 0 <= tid < len(self.tokens) else ""
        for ch in s:
            nxt = self.grammar.step(self.state, ch)
            if nxt is None:
                raise RuntimeError(
                    f"约束解码内部不一致：token {tid}({s!r}) 在状态 "
                    f"{self.state[1]} 下非法")
            self.state = nxt
        self.text += s


def parse_response_format(response_format: dict | None) -> dict | None:
    """解析 OpenAI 风格 `response_format`，返回 schema（None=不约束）.

    支持：
    - `{"type":"json_object"}` → 任意 JSON；
    - `{"type":"json_schema","json_schema":{"schema":{...}}}` → 指定 schema；
    - `{"type":"json_schema","json_schema":{...}}` → 省略 schema 包裹的简写。

    其他类型报 `JsonSchemaError`（服务端转 400）。
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise JsonSchemaError("response_format 须为对象")
    typ = response_format.get("type")
    if typ == "text":
        return None
    if typ == "json_object":
        return {}
    if typ == "json_schema":
        body = response_format.get("json_schema")
        if not isinstance(body, dict):
            raise JsonSchemaError("json_schema 须为对象")
        schema = body.get("schema", body)
        if not isinstance(schema, dict):
            raise JsonSchemaError("json_schema.schema 须为对象")
        return schema
    raise JsonSchemaError(
        f"response_format.type 须为 text/json_object/json_schema，实际 {typ!r}")
