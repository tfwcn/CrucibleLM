# 结构化输出：把 JSON Schema 编译成"每步只允许合法 token"的自动机

> 一句话：不训练、不重试，解码时按语法把非法 token 的 logits 置 `-inf`，
> 模型再想跑偏也跑不出合法 JSON。对标 DeepInfer `structure_output`（xgrammar）
> 的工程思路，用纯 Python 复刻最小子集，零新依赖。

## 痛点

让模型"输出 JSON"的常规做法是提示词 + 解析失败重试：

- 提示词只能**提高概率**，不能保证；小模型尤其容易少引号、多逗号；
- 重试在长输出上成本翻倍，且可能连续失败；
- 线上解析失败要么 500，要么把半截 JSON 当文本返回，下游难处理。

约束解码换一个位置解决：把"合法"从**事后检查**挪到**每步采样前**——
非法 token 的 logits 是 `-inf`，贪心和采样都只能落在语法内。

## 直觉

把 schema 看成一门只有有限状态的文法，解码就是**在自动机上走**：

```
词表 token ──逐字符──> 自动机状态
                        │
        当前状态允许的下一字符集合 → 只放行"整串合法"的 token
```

每一步只做两件事：**给当前状态求合法 token 掩码**、**把采样到的 token 吃进状态**。
状态是纯数据（不可变元组），所以"试吃一个 token"可以复制状态跑一遍，失败就丢弃。

## 原理

### 1) 为什么是字符级自动机，不是 token 级

token 化方式会变（字符级 / BPE / SentencePiece），但 **JSON 语法是字符级**的。
把 schema 编译到字符级，再在 mask 时把"token 字符串"喂进自动机验证，就能同时适配：

- 字符级词表（本仓库）：每个 token 一个字符，mask 就是集合查表；
- 多字符 token：`_can_take()` 逐字符 `step()`，整串合法才放行
  （例如 `"12"` 可以、`"1a"` 不行）。

### 2) 状态设计

状态是不可变元组 `(stack, mode, data)`：

| 字段 | 含义 |
|---|---|
| `stack` | 嵌套容器帧：`("obj", schema, done_keys, cur_key)` / `("arr", items_schema)` |
| `mode` | 当前期望：`value` / `obj_key` / `obj_colon` / `obj_comma` / `arr_value` / `arr_comma` / `str` / `num` / `lit` / `done` |
| `data` | 该模式的附加信息（schema、已读前缀、数字 DFA 状态、转义状态……） |

`step(state, ch)` 返回新状态或 `None`（非法），**从不修改入参**——
这是"试吃 token"和"批量求 mask"能安全共享同一套逻辑的前提。

### 3) 三个容易写错的地方

**(a) 值的结束与容器的关闭不是一回事。**
标量（字符串/数字/字面量）读完要"回到父级的逗号位"，用 `_finish_value()`
（父帧记录着 `cur_key`，此刻把 key 记进 `done_keys`）；而 `}`/`]` 是**关闭当前容器**，
要先 pop 自己再回到父级，用 `_close_container()`。两者混用会让 `}` 把对象
"还给它自己"，进入死循环。

**(b) 数字可以停在中途但仍然合法。**
`1` 读完后状态还停在 `num`（因为下一个字符可能是 `.`/`e`），但此时
`json.loads("1")` 已经合法。所以 `accepting()` 不能只看 `mode=="done"`：

```
accepting(num) = 数字可结束 and accepting(_finish_value(state))
```

后一半很关键——对象里的 `{"a":1` 数字合法，但对象没闭合，**不能**出 EOS。

**(c) 逗号后的位置不能直接 `}`。**
`{"a":1,}` 必须拒绝。`obj_key` 状态带 `allow_close` 标志：对象开头/读完值后为
`True`，刚吃过逗号为 `False`。重复键同理：`done_keys` 里的键不再允许出现，
否则模型会一直追加 `,"a":...` 直到 `max_tokens`。

### 4) `ANY` 哨兵：不假设词表字符集

本仓库词表含中文，而 schema 的"任意字符串"应该允许词表里的任何字符。
`allowed()` 不可能枚举未知字符，于是用一个不可能出现在正常文本里的哨兵：

```python
ANY = "\x00"   # allowed() 里出现它 = "任意非控制字符都行"
```

mask 侧看到 `ANY` 就放行所有 token，再用 `step()` 逐字符确认（控制字符、
未转义引号等仍会被自动机拒绝）。

### 5) 紧凑 JSON 与封闭对象（有意的取舍）

- **不允许空白**：JSON 允许 `{ "a" : 1 }`，但空白会让状态多一维（`skip_ws`）。
  约束解码统一输出紧凑 JSON，`json.loads` 结果完全等价。
- **对象视为封闭结构**：有 `properties` 时拒绝额外键；**没有** `properties`
  时才允许任意键（对应 JSON Schema 默认 `additionalProperties=true` 的语义）。

## 支持的 schema 子集

| 模式 | 示例 |
|---|---|
| 对象 | `{"type":"object","properties":{...},"required":[...]}` |
| 数组 | `{"type":"array","items":{...}}` |
| 任意字符串 | `{"type":"string"}` |
| 字符串枚举 | `{"type":"string","enum":["yes","no"]}` |
| 常量 | `{"const":"ok"}` / `{"const":42}` |
| 数字 / 整数 | `{"type":"number"}` / `{"type":"integer"}`（整数字禁止 `.`/`e`） |
| 布尔 / 空 | `{"type":"boolean"}` / `{"type":"null"}` |
| 任意 JSON | `{}` |

**显式拒绝**（编译期抛 `JsonSchemaError`，不静默降级）：
`anyOf`/`oneOf`/`allOf`/`not`/`$ref`、`additionalProperties: true`、
无 `items` 的数组、字符串以外的 `enum`。

## 从想法到代码

```
JsonGrammar(schema)
  ├─ _check()                     编译期递归校验，不支持的模式立刻报错
  ├─ initial()                    ((), "value", root_schema)
  ├─ allowed(state)               → frozenset[字符]（可能含 ANY 哨兵）
  └─ step(state, ch)              → 新状态 | None

JsonGrammarConstraint(grammar, tokens, eos_id)
  ├─ mask_logits(logits)          合法 token 留原值，其余 -inf；accepting 时放开 EOS
  ├─ accept(token_id)             逐字符 step；EOS → finished=True
  └─ finished                     生成循环据此 break

TinyLLM.generate(..., constraint=)
  └─ _sample_next(): 重复惩罚 → constraint.mask_logits() → 温度/top-k
     采样后 constraint.accept()；finished 即停
```

接入点只有两处：`_sample_next` 里一次 mask、循环里一次 `accept`。
有约束时**禁用投机解码**（草稿器不感知语法，验证会被大量拒绝）。

## 用法

```python
# Python API
backend.chat(messages, response_format={
    "type": "json_schema",
    "json_schema": {"schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }},
})

# HTTP（OpenAI 兼容）
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 128,
  "response_format": {"type": "json_object"}      # 任意 JSON
}
```

- `{"type":"text"}` / 缺省 = 不约束（与老路径逐位一致）；
- schema 非法或词表缺所需字符 → 服务端在写 SSE 头之前返回 **400**；
- 模型在 `max_tokens` 内没闭合 → 内容仍是合法 JSON 前缀，`finish_reason="length"`。

## 代价与限制

- **每步 O(V)** 构造掩码（字符级词表按首字符查表）；8192 词表在 CPU 上也是微秒级，
  相对一次前向可忽略。多字符 BPE 词表会退化为 O(V·平均 token 长度)，需加状态签名缓存。
- **只支持 `B=1`**：批量约束解码需要每序列一份自动机 + 变长状态，当前服务是单请求。
- **只支持 JSON Schema 子集**：覆盖工具调用/表单抽取的常见需求；
  复杂组合（`anyOf`）需显式扩展，不允许"尽力而为"。
- **不处理 `format`/`pattern`/数值范围**：这些是"值域约束"，不是语法约束，
  与自动机正交；需要时在业务层校验。

## 测试

`tests/test_structured.py`（38 项）：

- 语法接受/拒绝：对象必填、键前缀（`"a"` vs `"ab"`）、可选键、尾逗号、重复键、
  枚举兄弟前缀、嵌套、数字 DFA（`01`/`1.`/`1e` 拒绝）、字面量、任意 JSON；
- schema 编译期校验：组合 schema / `$ref` / 无 items 数组 / 非法枚举全部报错；
- mask：EOS 仅在 accepting 时允许、非法字符被 mask、多字符 token 整串校验、
  不修改输入张量；
- 端到端：贪心生成后 `json.loads` 成功；`LocalChatBackend.chat/stream`
  的 `response_format` 真正产出合法 JSON；
- 服务端：合法 schema 200 + 可解析，非法 schema 400。

## 坑

见 `docs/pitfalls/local-llm-training.md` 第 67–71 条（EOS 未 accept、尾逗号、
重复键死循环、词表缺字符、数字中途合法）。
