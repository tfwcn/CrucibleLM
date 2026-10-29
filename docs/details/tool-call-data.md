# 工具调用数据：让 128M 学会「先选工具，再填参数」

> 一句话：工具调用不做分类头，做成**两段式合成 SFT**——第一步只给工具目录让模型选
> `path`（SELECT），第二步把目标工具的参数表摆上让它填 `arguments`（FILL）；
> 输出统一为一行紧凑 JSON `{"path":"分组/工具","arguments":{...}}`，
> 渲染与解析只有一份实现（`src/llm/local/toolcall.py`），训练和推理共用。

## 痛点

`moyume-agents` 的真实对话里，主智能体调度 50 个工具，靠的是「渐进式发现」：
先 `toolbox_search` 找工具，再 `toolbox_browse` 看参数，最后 `toolbox_call` 执行。
这套流程对 7B 以上的模型没问题，对 128M 是灾难：

- **一次输出太长**：完整 `toolbox_call` 参数（path + 目标工具全部参数 + 嵌套 JSON 字符串）
  动辄 300+ 字符，还要在同一个 token 里既选对工具又填对所有字段；
- **候选集太大**：50 个工具的参数 schema 合计 34.5k 字符，全塞进 8192 的上下文
  只剩不到四分之一给对话；
- **没有分类头可退**：加一个「工具选择分类头」看似省事，但参数怎么来？
  分类头只能给 path，`arguments` 仍然要生成，等于只解决了 1/3 的问题，
  还要维护一份和 schema 同步的标签表。

真实轨迹也不够用：`moyume-agents/data/sandbox` 下 9 个会话库共 281 条消息，
其中 **user 只有 11 条**、带 `tool_calls` 的 assistant 104 条、`toolbox_call`
共 115 次——但每个请求只取第一次调用（多轮里后续调用依赖前一步结果，脱离上下文
就是错样本），最终只有 **9 条**可用，其中 2 条参数名已过时（`ref` vs `subtask_id`）
被 schema 校验丢掉。**靠 9 条学不会工具调用**，必须合成。

## 直觉：把「会调工具」拆成两个 128M 学得会的动作

模仿人类用工具的过程：先在目录里找到「该用哪个」，再看这个工具「要填什么」。
两件事的难度完全不同，混在一起学才是 128M 学不会的原因。

| 阶段 | 输入 | 输出 | 训练样本 |
|---|---|---|---|
| **SELECT** | 系统提示里只有工具目录（`- path：说明`）+ 用户请求 | `{"path":"web/web_search","arguments":{}}` | 2 794 条 |
| **FILL** | 目录（去掉目标工具）+ 目标工具完整参数表 + 用户请求 | 完整参数 | 2 507 条 |
| **负样本** | 目录 + 任何工具都做不到的请求 | `{"path":"","arguments":{}}` | 720 条 |

- `arguments` **为空对象就是 SELECT**（协议约定，不是缺字段）：调用方拿到 path 后
  把该工具的参数表再喂一轮，同一模型接着做 FILL。这正好对上 moyume-agents 的
  `toolbox_browse → toolbox_call`；
- FILL 形态里目标工具**从目录里剔除**（参数表已在下方单独渲染），省 token 也避免
  模型看两遍同样的说明；
- 输出不走 `toolbox_call` 包装，直接是「工具箱路径 + 目标工具参数」——包装层是
  运行时实现细节，学它等于白白多学一层嵌套 JSON 字符串转义；
- `path` 为空 = 「所列工具都做不了」，让模型学会**不硬套工具**（真实场景里
  「写首诗」「算个数」占比很高，全都硬调 `web_search` 比不调更糟）。

## 实现：一份渲染，两个消费方

`src/llm/local/toolcall.py` 是唯一实现：

```
normalize_tools()   三种工具声明（本仓库 schema / OpenAI function / 裸 name）→ 内部结构
render_catalog()    工具目录，一行一个 `- path：说明`
render_params()     目标工具参数表（必填/可选/默认值/类型/说明）
render_system()     persona + 目录 + [参数表] + 输出规则
parse_tool_call()   模型输出 → {"path","arguments"}（容错 JSON 串/多余文字）
to_openai_tool_calls()  内部形态 → OpenAI tool_calls
```

`scripts/synth_tool_sft.py`（造数据）和 `src/llm/local/infer.py`（跑推理）都 import 它。
**这不是洁癖**：工具调用是「输入格式即契约」的任务，训练和推理的系统提示必须逐字节
相同。历史上 SFT 模板 `<用户>/<助手>` 与采样模板不一致，就埋过一次「loss 正常下降、
线下掉点」的坑（pitfall 81）。

合成侧的结构：

- 44 条手写场景覆盖 **45 个工具路径**（不含 worker-only 的 `ask_parent`）+
  8 类「分组级」请求（如「你有什么记忆」「看看任务进度」）→ 这些只该选到
  `memory/memory_list`、`task/taskplan_get` 这类入口工具，考验目录理解；
- 值池按字段语义分开：24 个文件路径 / 10 个目录 / 18 条命令 / 15 个查询 /
  8 个 URL / 8 个 glob / 9 个模式 / 5 个子智能体 / 5 个任务 / 8 个记忆键 /
  5 个触发器 / 3 篇博客 / 4 个群 / 5 个用户；
- 请求模板用 `$var` 引用值池，`check_scenarios()` 在**生成前**静态校验：
  变量出处、参数名是否在该工具 schema 里、类型是否匹配、必填是否漏，
  错一条直接抛（宁可报错也不静默造坏样本）；
- 每次采样随机抽 3~8 个干扰工具进目录（从 50 个里取），保证目录组合不固定，
  模型不能靠「目录里只有一个工具」蒙答案；
- 真实轨迹以 `--real ../moyume-agents/data/sandbox --real-repeat 3` 掺入，
  只保留 path 存在且参数过 schema 的调用（9 条 → 21 条，含 2 条被丢弃的旧参数名）。

样本形态（`messages`，直接喂 `iter_local_sft_pairs`）：

```json
{"messages": [
  {"role": "system", "content": "你是一个能操作工作区的助手…\n可用工具：\n- web/web_search：…\n\n输出规则：…"},
  {"role": "user",   "content": "帮我搜一下今天的 AI 新闻"},
  {"role": "assistant","content": "{\"path\":\"web/web_search\",\"arguments\":{\"query\":\"今天的 AI 新闻\"}}"}
]}
```

## 实测（本机，6021 条）

| 指标 | 值 |
|---|---|
| 总量 | 6 021 条（fill 2 507 / select 2 794 / negative 720）+ 真实轨迹重复后计入 |
| 唯一率 | 100%（按 messages 全串去重） |
| 覆盖工具路径 | 45 / 45（含 `toolbox_search` / `toolbox_browse` 两个入口） |
| 单条长度（字符） | 中位 561 / P90 829 / P99 1 250 / 最长 1 467 |
| prompt 长度（字符） | 中位 515 / P90 745 / P99 1 117 / 最长 1 332 |

**`seq_len` 下限是被样本长度硬约束的**（`pack_pairs` 在 prompt 超长时静默 `continue`）：

| `seq_len` | 被丢弃样本 | 输出被截断 |
|---|---|---|
| 512 | **51.3%** | 15.2% |
| 1 024 | 1.7% | 1.5% |
| **1 536** | 0% | 0% |
| 2 048 | 0% | 0% |

冒烟实测：`--seq-len 256` 时**全部样本被丢弃**，日志只有一行「数据流耗尽，提前结束」，
看起来像数据路径错，其实是长度不够。真实 SFT 用 **2 048**（留出子词表压缩后的余量
与更长对话的空间）。

## 从想法到代码

| 想法 | 代码 |
|---|---|
| 「选工具」和「填参数」分成两个小任务 | `build_select()` / `build_fill()`：FILL 把目标工具参数表附在目录下方 |
| 空参数 = 只选了工具 | `arguments: {}` 约定；`synth` 统计里 `not b["arguments"]` 即 SELECT |
| 工具从真实注册表来，不手抄 | `scripts/dump_tool_schemas.py` → `data/tool_schemas.json` |
| 训练/推理 prompt 不许漂移 | `toolcall.py` 单一实现，两侧 import |
| 输出必须能直接执行 | `parse_tool_call()` + 候选校验；幻觉 path 当「没调用」 |
| 模型不能靠单一目录蒙对 | 每次采样随机抽 3~8 个干扰工具 |
| 不许静默造坏样本 | `check_scenarios()` 生成前静态校验变量/参数名/类型/必填 |

推理端接线（`LocalChatBackend.chat(tools=, tool_target=)`）：

1. `_inject_tools()`：把工具渲染进 system（原 system 内容当 persona 槽）；
2. `_tool_response_format()`：有 tools 且调用方没显式给 `response_format` 时，
   默认上 `CALL_SCHEMA` 约束解码——**只保证「是两个字段的 JSON」**，path 合法性后判；
3. `_parse_tool_output()`：解析 + 候选校验，路径按「精确 → 唯一末段名」两级匹配
   （`workspace/shell_exec` ↔ OpenAI 声明里的 `shell_exec`），返回**调用方那侧**的路径；
4. 有调用 → `finish_reason="tool_calls"`、`content` 为空；无调用 → 保留原文。

## 代价与坑

- **格式由约束保证，内容只能靠 SFT**：未 SFT 的 `data/llm-sft7` 在 `CALL_SCHEMA`
  约束下输出的是 `{"path":"","arguments":{"arguments":"",",":",",…` 这种垃圾循环
  （合法 JSON、无意义内容）。所以「加约束」不等于「会调工具」，两者都要做。
- **混训配比（`--sft-epochs` + 权重）**：`data/sft-zh` 38 万行、`data/sft-belle`
  42 万行，工具数据只有 6 021 行。按 1:1 轮询工具只占 0.8%，等于没学。

  注意 `interleave_weighted` 的语义是「**一遍**流按权重轮询，耗尽的源移出」：
  工具流 6 021 条一轮就没了，之后全是被稀释的语料——所以**光调权重没用**，
  必须配合遍数：

  ```
  --local-path data/sft-tool:5,data/sft-zh:1 --sft-epochs 3
  ```

  一遍的长度由「工具流耗尽」决定：工具 6 021 条 + zh 约 1 204 条。3 遍 =
  工具 3×2.43M + zh 3×1.27M ≈ **11.1M token**（子词表口径），
  `batch 4 × seq 2048 × accum 2` ≈ **680 步**、约 25 分钟。

  推荐起始配方（先跑这个，再动别的）：

  | 项 | 值 | 理由 |
  |---|---|---|
  | `--sft-init` | `data/llm-sft7/best/model.pt` | 已有中文对话能力，别从零学格式 |
  | `--subword-vocab` | `data/subwords-sft-tool.json` | 工具文本 2.20x 压缩，等价上下文翻倍 |
  | `--local-path` | `data/sft-tool:5,data/sft-zh:1` | 工具占 66% token，zh 3.8M 做防遗忘 |
  | `--sft-epochs` | 3 | 2.43M × 3 ≈ 7.3M 工具 token |
  | `--seq-len` | 2048 | 1 536 起零丢弃，2048 留余量 |
  | `--batch/--accum` | 4 / 2 | 16.4k token/步，约 680 步（步数太少学不动新行为） |
  | `--lr` | 5e-5（`--warmup 30`） | 比 sft7 的 1e-5 高：要学**新行为**，不是微调风格 |
  | `--holdout-file` | `data/holdout-sft-tool.json` | 固定评测集，跨 run 才能比 |

  对照组：`--local-path data/sft-tool --sft-epochs 3`（纯工具，7.3M token），
  用来量「混 zh 到底救回了多少对话能力」——**先跑纯工具组**，如果中文对话
  明显退化，再上 5:1；不退化就别浪费算力。
- 真实轨迹只有 9 条可用，是**数据本身的限制**（11 条 user 消息），不是抽取 bug。
  提高真实占比只能靠后续积累会话，不要重复采样假装多样性。
- 子词表必须用工具样本来挖：`--boost-file` 掺 12% 工具文本才有 2.20x，
  纯中文语料挖出来对工具 schema 只有 1.42x（pitfall 80）。

## 文件

| 文件 | 作用 |
|---|---|
| `src/llm/local/toolcall.py` | 渲染 + 解析的唯一实现 |
| `scripts/dump_tool_schemas.py` | 从 moyume-agents 注册表导出 50 个工具的 schema |
| `scripts/synth_tool_sft.py` | 合成 SFT（场景库 + 值池 + 静态自检 + 真实轨迹掺入） |
| `src/llm/local/data.py` | `iter_local_sft_pairs` / `iter_local_sft_objects` / `convo_to_pairs` |
| `src/llm/local/infer.py` | `chat(tools=, tool_target=)` 接线 |
| `tests/test_toolcall.py` / `tests/test_tool_sft.py` | 渲染/解析/路径对齐/端到端模板一致性 |
