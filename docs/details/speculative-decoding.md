# 投机解码：先猜后验，一次吐多个

> 一句话：复用已训好的 MTP 头 + 前缀 n-gram 起草后续 token，主干一次并行验证，
> 接受最长草稿前缀就一次提交多个 token；**输出与逐 token 贪心逐位一致**。

## 痛点

batch=1 解码是 launch-bound：单步约 1800 个小算子，每个付一次 Python 分发与
kernel 启动（见 `inference-acceleration.md`）。每个 token 都要完整走一遍主干，
GPU 算得飞快却大部分时间在等启动。

## 直觉

阅读理解时你往往能"猜"到下一句，猜对了整段都不用逐字推。投机解码就是先让一个
便宜的猜手（draft）押几个 token，再让权威的主干一次性核对——猜对了就白拿。

## 原理

```
每轮：主头出权威下一位 t0
      起草器补 k-1 个后续候选 d2..dk
      喂 [t0, d2..dk] 一次并行前向（_decode_multi）
      得到主干在每个位置的贪心 → 与草稿逐位比
      接受最长草稿前缀 + 主头的下一个 token → 提交，状态对齐（零重放）
```

关键：**验证必须与逐 token 解码同算术**，否则主干"验证出的贪心"和"真贪心"
会在近并列 logit 上分叉，"等于贪心"的契约就守不住。因此本模型的多 token 增量
解码三条路径都做了对齐：MLA 多查询 SDPA（逐行独立）、稀疏路径按
`start=total-t` 聚集、线性层按单步同款递推。

## 我们的实现

- 文件：`spec_decode.py`（起草器：`ngram` / `mtp` / `hybrid`）、
  `model.py`（`_spec_stream` 主循环 + `_decode_multi` 并行验证）；
- 配置：`draft_enabled`（默认关）、`draft_window`、`draft_source`、`draft_ngram_min`；
  以及 DSpark 置信调度：`dspark_enabled`（默认关）、`dspark_conf_threshold`、
  `dspark_min_verify`；
- 为支撑并行验证，三处解码路径补了多 token 能力（均带回归单测）：
  - `mla.py:_valid_mask`：`t>1` 时补块内因果掩码（原来只掩空位，会偷看同批未来）；
  - `sparse_attn.py:_decode`：查询改用最后 t 个位置（`total-t`）；
  - `linear_attn.py`：新增 `past≠None, t>1` 分支，按单步同款递推（状态初值 `s_prev`），
    并把每步缓存 `(s_i, k 尾, v 尾)` 记进 `multi_steps`。
- **零重放回退**：`_decode_multi` 额外返回 `step_pasts[i]`——"提交前 i+1 个 token 后"
  每层的完整缓存（线性层取 `multi_steps[i]`；MLA 复用同一缓冲、只把 `pos` 调回
  `pos+i+1`，越界写由掩码屏蔽）。部分接受时直接 `h=h_multi[:, A]`、
  `pasts=step_pasts[A]` 跳位，**不再逐 token 重放**。实测这一步把拒答代价从
  "一整步前向"降到零，吞吐 1.29x→1.45x（见下）。
- 门槛：先跑 `scripts/measure_mtp_acceptance.py`，接受率 > 50% 才值得开。

### P0 接受率门槛（真实 checkpoint 实测）

MTP 对 t+2 的接受率（`measure_mtp_acceptance.py`，字符级分词，序列截到 8192）：

| 权重 | 多轮对话 | 长文 | 代码 |
|---|---|---|---|
| 预训练 base（step 5300） | 30.8% | 30.7% | 48.3% |
| SFT-7 best | 39.2% | 38.7% | **66.3%** |

（head>mid>tail 一致：越靠前 MTP 首猜越准；"代码"从 `data/sft-coig-code` 取样本。）

**结论：默认关；按负载决定。** 自然语言（对话/长文）30–39% < 50% 门槛，不值得开；
**代码 66% > 门槛，值得开**（投机解码专吃"高可预测"文本）。用内置样例句会因文本
高度重复把接受率虚高到 68%——**必须用代表性语料测，别拿模板句自欺**。

带 memory/retro/hyper 的 SFT checkpoint 需复用其结构配置才能载权重：

```bash
python scripts/measure_mtp_acceptance.py --model data/llm-sft7/best/model \
    --run-yaml data/llm-sft7/run.yaml --text data/val-code.txt
```

### 实测加速（约 128M、RTX 3080 Laptop、batch=1 贪心）

| 场景 | 起草来源 | 加速 | 备注 |
|---|---|---|---|
| 普通中文对话 | mtp / hybrid | ~1.45–1.5x | MTP 首猜命中 ~0.6–0.8，输出逐位等于贪心 |
| 代码 / 列表 | hybrid | ~1.6–1.7x | ngram 偶与 MTP 一致时采纳更长续写，多拿几位 |
| 普通中文对话 | ngram | ~0.9x | ngram 首猜几乎全错（~0.04），净亏——别单独用 |
| 任意 | off | 1.0x（基线） | 默认关，生成路径逐位不变 |

注：上表加速取自旧版训练权重（自然语言首猜命中较高）；上上表（新权重）显示自然语言
接受率已降到 30–39%，**加速只在代码等高可预测场景成立**。两份数据一致指向：投机解码
的收益强依赖"文本可预测性"，不是普适开关。

结论：**MTP 是唯一高命中来源**（它天然预测 t+2）；ngram 仅对重复/模板文本有用，
且必须"与 MTP 首猜一致才采纳"，否则会污染后缀检索（见 pitfalls）。

### DSpark 置信调度验证（借鉴 DeepSeek-V4.1）

论文 *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation*（arXiv 2607.05147）指出投机解码的隐性浪费：**固定窗口下，后缀里
"看起来必被拒"的草稿也会被主干完整验证一遍**（verification waste）。它的核心之一
是**置信调度验证**——给每个草稿 token 一个"存活置信度"，据此动态决定每轮验多长，
提前把必被拒的后缀截掉。

我们实现其**置信调度**部分（单请求、无引擎负载感知）：

```
每轮：起草器返回 (草稿 tokens, 各自置信度 conf∈[0,1])
      cum = 1；只验满足 ∏ conf ≥ dspark_conf_threshold 的最长前缀
      （无论多低至少验 dspark_min_verify 个，保底每轮推进）
      喂 [t0] + 该前缀 一次并行验证 → 接受最长前缀（与固定窗口完全一样的接受逻辑）
```

置信度来源：MTP 用其 softmax 最大概率（天然校准）；ngram 用匹配后缀长度 `m/(m+1)`
（命中越长越可信）。`dspark_conf_threshold<=0` 退化为"全验证"（= 固定窗口）。
**关键：这只减少验证长度，不改变最终序列**——验证仍是精确的贪心核对，被截断的草稿
只是下一轮再猜，输出始终逐位等于逐 token 贪心（单测锁定）。

与论文的差距（诚实说明）：论文还有**半自回归起草**（并行 backbone + 轻量顺序
"Markov"头，缓解后缀衰减）与**引擎负载感知调度**。前者需要独立的草稿模型（本项目
没有，用 MTP + ngram 的 hybrid 代偿），后者是多请求服务端才有的问题（本模型单请求）。
故我们只取"置信调度"，并在 `_spec_stats` 里记 `proposed`/`verified` 供对照。

| 场景（合成低置信长草稿） | proposed | verified | 说明 |
|---|---|---|---|
| 关 DSpark（固定窗口） | 每轮 k | 每轮 k | 基线 |
| 开 DSpark（阈值 0.5） | 每轮 k | 每轮 1 | 后缀被截，输出仍等于贪心 |
| `dspark_conf_threshold=0` | 每轮 k | 每轮 k | 退化回固定窗口 |

## 从想法到代码

```python
# 起草 → 并行验证 → 接受最长前缀 → 零重放跳位
if dspark:
    drafts, confs = scored_propose(provider, ctx, h[:, -1:], window - 1)
    v = schedule_verify_length(confs, threshold, min_verify, window - 1)
    drafts = drafts[:v]                                  # 置信调度：截断必被拒后缀
else:
    drafts = provider.propose(ctx, h[:, -1:], window - 1)
feed = [t0] + drafts                                     # t0 为主头权威下一位
h_multi, _, step_pasts = self._decode_multi(feed, pasts) # 一次前向喂完整窗口
accepted = 最长前缀(草稿 == 主干在对应位置的贪心)
h = h_multi[:, accepted]          # 已提交前缀末位的 hidden
pasts = step_pasts[accepted]      # 同步缓存（并行前向按步记录，零重放）
```

## 代价与坑

- 只有 batch=1 且 `temperature=0`（纯贪心）走此路径；采样/批量自动回落普通生成；
- **起草源选错会净亏**：ngram 首猜命中率极低，单独用会 ~0.9x（每次验证一轮全拒）；
  默认 `hybrid` 在 ngram 与 MTP 首猜不一致时只回退 MTP 单猜，故不亏；
- 每步缓存（`multi_steps`）显存随窗口线性增长：`window` 别开太大（默认 5 足够）；
- `repetition_penalty` 已按"seen 上下文逐步累积"对齐，验证时逐位应用，契约不破；
- 多 token 解码的三处改动均默认不影响 `t==1`：`_valid_mask` 在 `t==1` 时与旧式
  逐位相同，稀疏/线性分支只在新代码路径生效（单测锁定）；
- **DSpark 截断可能"误伤"本会被接受的草稿**：阈值越高越激进，省了算力也可能少拿
  接受位、多跑几轮。它不改输出（仍等于贪心），只改"每轮验多长"，故是一个纯粹的
  吞吐权衡；单请求小模型上收益有限，开关默认关。`dspark_conf_threshold<=0` 即回退
  固定窗口（等价关闭）。

## 一句话总结

起草猜、主干验、接受前缀一次吐——复用 MTP 头与 n-gram，零训练拿到解码加速，
且严格不改变输出。
