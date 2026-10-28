# 追加式子词表：不动旧权重，把上下文翻一倍

> 一句话：往词表**末尾**追加 2000 个高频子词，`encode` 改成最长匹配，
> 旧的 6110 个字符 id 一位不移 —— 同一份权重可以直接续训，而工具调用类文本的
> token 数实测降到 **0.45 倍**（2.20x 压缩），上下文有效长度翻倍。

## 痛点

本仓库的分词器是**纯字符级**（`SimpleTokenizer`：一个字符 = 一个 id）。
这在中文上是合理取舍（中文 1 字 ≈ 1 个语义单元），但在**英文与 JSON** 上是纯亏：

| 内容 | 字符数 | 字符级 token | 典型 BPE 约 |
|---|---|---|---|
| 工具 schema | 210 | 210 | ~95 |
| OpenAI `tool_calls` | 222 | 222 | ~108 |
| 结果信封 | 44 | 44 | ~12 |

`2048` 的上下文因此只装得下 2048 个字符：一张三工具 schema 就吃掉 15%，
`"arguments"` 要烧 9 个 token，`"name"` 4 个。模型参数没变，但**每个 token
携带的信息量**只有同龄 BPE 模型的三分之一左右。

直接换一个 32k BPE 词表（重训分词器）是最干净的做法，代价是**已有的
655M token 预训练全部作废**——embedding 行整体重排，旧权重一行都对不上。
这台机器的吞吐（8.2k tok/s）意味着重训一次 base 要 22 小时，这不是首选。

## 直觉：词表可以「只加不改」

`SimpleTokenizer` 原本就有一条硬约定（`_chars` 注释里写着）：

> 扩词规则：只允许末尾追加，旧 id 永不移位（已训权重逐行兼容的前提）。

`extend_vocab_and_model` 一直在用这条约定（新增字符复用/扩展 embedding 的空行）。
把它从「追加字符」放宽到「追加**任意表面串**」，就得到子词能力，而 id 语义不变：

```
id 4 .. 6113      旧字符（原样）
id 6114 .. 8113   新子词（embedding 随机初始化，需要重新适配）
```

编码时对每个位置先在 `_max_len` 窗口里**最长匹配**，命中就用子词 id，
否则退回单字符。解码仍然是「把表面串拼起来」，完全不用改。

这就是「warm restart」而不是「无损续训」：旧字符行还在、还在用，
但新 token 的 embedding 是空的，模型要先学会它们，同时要适应同一段文本
**切法变了**（`"name"` 从一个 4 个 id 的序列变成 1 个 id）。实测这个适配期
比从零短得多（旧知识全在），但 sft 阶段的前 100 步 loss 会有一个可见的台阶。

## 实测（本机，真实语料）

挖掘来源：`data/sft-zh:3,data/sft-belle:1`，8.1M 字符，`nmax=5`，`min_count=8`，
另外掺 12% 的工具调用样例（`--boost-file`）把 JSON 类 n-gram 抬进榜内。
词表容量 `8192 - 4 = 8188`，sft7 已用 6110 → 只能追加 **2078**，取了 2000。

| 内容 | 字符 | 字符级 | 追加子词后 | 压缩 |
|---|---|---|---|---|
| 工具 schema | 210 | 210 | 95 | **2.21x** |
| OpenAI `tool_calls` | 222 | 222 | 108 | **2.06x** |
| 结果信封（`ok/error/retryable`） | 44 | 44 | 12 | **3.67x** |
| 中文对话 | 56 | 56 | 29 | 1.93x |
| **合计** | 535 | 535 | **243** | **2.20x** |

对照（不掺工具样例，纯中文挖掘）：中文语料 1.44x，工具探针只有 1.42x ——
**挖掘语料决定子词去向**，要让模型会用工具，就得把工具文本喂进挖掘器。

正确性验证（`tests/test_subword_vocab.py` + 真机冒烟）：

- 1000 条真实 SFT 文本 `decode(encode(x)) == x`，**0 失败**；
- 追加 2000 子词后 `embed` **行数不变**（`resized=False`，走的是复用空行那条路）
  → 优化器状态仍然有效，可以 `--resume`；
- 旧 6110+4 行 embedding **逐位不变**（`torch.equal` 断言）；
- `structured.py` 的 JSON 约束解码在混合词表上照常工作（它本来就按
  「整串逐字符吃」实现，多字符 token 是天然支持的）。

## 为什么 `nmax=5` 是甜点

`nmax` 是「最多几个字符拼成一个 token」。实测三档：

| nmax | 候选数 | 工具探针压缩 | 备注 |
|---|---|---|---|
| 4 | 865k | 2.04x | 抓不住 `arguments` / `toolbox`（都 >4 字符） |
| **5** | **1.12M** | **2.20x** | 覆盖 JSON 键名与常用英文词根 |
| 8 | 400k（被剪枝） | 2.01x | 长 n-gram 频次低，统计时被 `keep` 榜挤掉 |

`count_ngrams` 每攒够 `keep×3` 个候选就按频次剪枝（`PRUNE_MULT`），
所以更长的 n-gram 反而更容易被高频短串挤出去。要用 `nmax>5` 得同步放大 `--keep`，
内存也跟着涨，收益不明显，**默认 5**。

## 从想法到代码

| 位置 | 改动 |
|---|---|
| `src/llm/local/infer.py::SimpleTokenizer.__init__` | 新增 `_meta_key/_max_len/_multi` 三个缓存字段 |
| `…::_sync_meta` | 按 `(len(_chars), len(_ids))` 指纹重建「是否含多字符 / 最长几字符」 |
| `…::encode` | `_multi` 为真时走最长匹配（先试长、退到单字符、最后 UNK）；否则原逐字符路径 |
| `…::fit` | 清空 `_meta_key` 强制重建（拟合后是新表） |
| `…::extend_vocab_and_model` | 语义从「追加字符」放宽到「追加表面串」，其余逻辑不变 |
| `src/llm/local/structured.py` | **无需改动**：`_can_take`/`accept` 本来就逐字符推进 |
| `scripts/build_subwords.py` | 新脚本：统计 n-gram → 打分排序 → 落盘 JSON + 压缩比报告 |
| `scripts/train_local_llm.py --subword-vocab` | 优化器构建**之前**追加，落盘 `vocab.json` 自然带上 |
| `src/llm/local/resume.py` | `vocab`/`subword_vocab` 进 `CRITICAL_KEYS`：换分词器必须 `--force-resume` |

缓存用「长度指纹」而不是显式 `_refresh()` 调用，是为了 **8 个老脚本**：
`rft_sample.py` / `init_memory.py` / `analyze_experts.py` 等都是
`tok._chars = json.load(...)` + 手搓 `_ids`，没有多字符项时 `_multi=False`
→ 走原路径，逐位不变；词表里有多字符项时自动开启最长匹配，不用改任何脚本。

## 用法

```bash
# 1) 挖子词（8M 字符约 40 秒；--boost-file 决定多少预算给工具 JSON）
python scripts/build_subwords.py \
    --corpus "data/sft-zh:3,data/sft-belle:1" \
    --vocab data/llm-sft7/vocab.json \
    --out data/subwords-sft-tool.json \
    --limit 2000 --nmax 5 \
    --boost-file data/tool_boost.txt --boost-frac 0.12

# 2) 续训（必须同时固定评测集，否则 val_loss 不可比）
python scripts/train_local_llm.py --phase sft --preset base \
    --resume --force-resume \
    --ckpt-dir data/llm-sft8 \
    --vocab data/llm-sft7/vocab.json \
    --subword-vocab data/subwords-sft-tool.json \
    --holdout-file data/holdout-sft.json \
    --max-steps 1500 --lr 2e-5
```

注意 `--force-resume`：换分词器会被 `resume.py` 拦下，这是**故意的**——
它等价于换了一门语言，曲线的台阶必须被记录在案。

## 代价与注意

- **容量是硬约束**：`vocab_size - 4` 是 embedding 行数上限（base = 8188）。
  装满 8188 后，`extend_vocab` 里的「追加缺字」就会**扩行**，那时优化器必须新开。
  挖矿脚本会先算余量并提示。
- **纯空白串要过滤**：缩进/换行 n-gram（`"\n  "`）频次极高但零语义，
  `select_subwords` 用 `isspace()` 剔掉。
- **UNK 会在解码时消失**：表外字符（emoji、生僻字）仍然是 UNK，`decode` 直接跳过，
  round-trip 对它不成立 —— 这是**改造前就有的行为**，不是本次引入的。
- **挖掘语料 = 投资方向**：只从中文百科挖，工具文本的压缩比只有 1.42x。
  下个模型要会用工具，`--boost-file` 必须喂工具调用样例（真实轨迹 > 模板）。
