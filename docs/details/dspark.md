# DSpark：独立训练的投机解码草稿器

> 一句话：投机解码的瓶颈是"草稿猜不准就白验"。DSpark 用一个**独立训练**的 3 层小模型
> 一次并行猜 5 个位置（半自回归），配 Markov 头补位置间依赖、置信头估接受概率，
> 再用吞吐曲线决定"这轮验多长"。

## 痛点

投机解码省钱的前提是草稿的接受率。两条老路都不够：

- **MTP 头**（DeepSeek-V3）：与主干联合训练、共享前向，只能多看**一步**（t+2）；
  要多猜几位就得加深 MTP，等于让主干多背几个头，且与主干更新强耦合。
- **n-gram 查表**：零成本，但只在复读/模板文本上有效，开放文本命中率个位数。

还有第二个浪费：**固定验证窗口**。草稿置信度低时仍然整窗验证，验的多数会被拒——
这部分算力纯亏。

## 直觉

派一个"实习生"（3 层小模型）先把下面 5 个词写好，正式员工（主干）一次并行批改：
批到第一个错就停，后面的草稿直接扔。实习生写得好 → 一轮推进 5 个 token；
写得烂 → 白花一点批改钱，但**答案永远由正式员工签字**（验证精确）。

## 原理（V4.1 §2.4.3）

```
base   = lm_head( Drafter( [ctx..., h_last] , slot_1..slot_m ) )   # 一次前向、并行
logits_i = base_i + U · E(t_{i-1})                                 # Markov 头（一阶低秩）
c_i    = σ(w · h_i)                                                # 条件接受概率
S_i    = ∏_{j≤i} c_j                                               # 前缀存活概率
n*     = argmax_n  (1 + Σ_{i≤n} S_i) / (draft_cost + base + per_token·n)
```

三块设计各自解决一个问题：

| 组件 | 解决的问题 | 我们的实现 |
|---|---|---|
| 3 层滑窗（128） | 读多少历史才够猜 | 只喂最近 128 个 token 的 embedding，末位换成主干 hidden |
| **并行** `m=5` 槽位 | 一次前向出 5 位 | 追加 5 个可学习 query 槽位；槽位之间**互不可见**（否则退化成串行） |
| **Markov 头** | 并行的代价：位置间无依赖 | `base_i + U·E(t_{i-1})`（秩 32，零初始化 = 开局退化为纯并行） |
| **置信头** | 调度器要知道"该验多长" | 逐位置 `σ(w·h_i)`，训练目标是主干在该草稿 token 上的概率质量 |
| **吞吐调度** | 固定窗口浪费算力 | 期望产出 / 期望成本 取最大（下面详述） |

**为什么"半"自回归**：位置之间并行（便宜），依赖只走一阶 Markov（够用且极便宜：
`2·V·r` 参数，V=8192、r=32 时 0.5M）。真要全自回归就退化成小号主干，一次只能出一位。

**为什么独立训练**：MTP 是与主干**联合**训练的，主干一更新，MTP 的输入分布就漂移；
DSpark 放在预训练之后单独开班（主干冻结、只训草稿器），后训练再一起训但**梯度不回传
主干**——它永远对齐当前策略，线上服务和 RL rollout 都能加速。

**调度器的成本模型**（`dspark.DSparkScheduler`）：

```
tokens(n) = 1 + Σ_{i=1..n} S_i          # 含主干自己那颗恒定接受的首 token
cost(n)   = draft_cost + base_cost + per_token · n
```

`s_i` 由置信头给出，成本系数要**实测标定**（本仓库给了 `scripts/bench_dspark.py`，
它直接输出"草稿一步 / 验证一步"的比值）。系数填错只影响"选多长"，不影响正确性。

## 从想法到代码

```python
# 一次前向（草稿器）
x = embed(ctx[-window:]);  x[-1] = h_last        # 末位用主干 hidden
x = cat([x, q_slots])                             # 追加 m 个查询槽位
mask = 因果 & 滑窗 & ~(槽位×槽位)                 # 槽位互不可见
hq = blocks(x)[-m:]                               # 3 层滑窗注意力
conf = sigmoid(conf_head(hq))                     # 逐位置接受概率

# 起草（部署）
for i in range(m):
    logits = lm_head(hq[:, i]) + markov_bias(prev)  # prev = 上一颗草稿（首位=上下文末位）
    tok = argmax(logits)                            # 存成草稿，同时喂给下一位
```

## 我们的实现

| 位置 | 内容 |
|---|---|
| `src/llm/local/dspark.py` | `DSparkDrafter`（3 层滑窗 + 槽位 + Markov/置信头）、`DSparkScheduler`、`attach_dspark` |
| `src/llm/local/spec_decode.py` | `DSparkDraftProvider`（`draft_source="dspark"`）、`build_draft_provider` |
| `src/llm/local/model.py` | `dspark_draft=True` 时挂 `model.dspark`；`_spec_stream` 支持 `confidence`/`throughput` 两种调度 |
| `scripts/train_dspark.py` | 独立训练：主干冻结、只训草稿器，落 `dspark.pt` |
| `scripts/bench_dspark.py` | 接受率 / 端到端 tok/s / 成本曲线标定 |

开关（全部 opt-in，默认关，不污染老 ckpt）：

| 配置 | 默认 | 作用 |
|---|---|---|
| `dspark_draft` | False | **构建草稿器**（结构性，state_dict 多 `dspark.*` 键） |
| `dspark_layers` / `dspark_window` / `dspark_depth` | 3 / 128 / 5 | 论文口径 |
| `dspark_heads` / `dspark_markov_rank` | 4 / 32 | 小模型够用 |
| `dspark_enabled` | False | 开调度（只影响"验多长"） |
| `dspark_schedule` | `confidence` | `confidence`=累计存活阈值；`throughput`=期望吞吐最优（论文口径） |
| `dspark_target_base` / `dspark_target_per_token` / `dspark_draft_cost` | 1.0 / 0.15 / 0.6 | 成本曲线（**实测标定**） |
| `draft_source="dspark"` | `hybrid` | 用训练好的草稿器起草 |

## 实测

真实 checkpoint（162.7M、d=768、12 层、vocab 8192）+ 随机初始化草稿器（18.2M）：

| 检查 | 结果 |
|---|---|
| 端到端一致 | DSpark 起草（两种调度）与逐 token 贪心**逐位相同** ✅ |
| 成本比（草稿一步 / 验证一步） | **0.04**（3 层 18M vs 12 层 162M；`--dspark-draft-cost` 建议填这个） |
| 未训练草稿器的接受率 | 0/32 轮（随机权重本该如此；同时也说明退化路径不崩） |
| 端到端速度 | 0.99x（接受率 0 时基本等于贪心开销，投机不亏太多） |

训练机制冒烟（tiny 主干 4 层、草稿器 0.5M、3 步）：loss 12.78 → 12.72，
`s`/`k` 三路损失都下降方向正确，`dspark.pt` 只含草稿器 + 结构快照（主干一字节没动）。

单测 15 项（`tests/test_dspark_module.py`）：形状/置信区间、Markov 零初始化、
槽位互不可见、滑窗截断、调度数学（存活概率/期望产出/成本单调/k 截断）、
两种调度都等于贪心、`attach_dspark` 严格载入、训练 20 步 CE 降 10%+、梯度有限。

## 代价与坑

- **草稿器要单独训**：不训也能用（不崩、仍正确），但接受率≈0，等于白算。先跑
  `scripts/train_dspark.py`；它要求分词表与主干同源（`<run>/vocab.json`）。
- **有空转窗口**：`depth=5` 但每轮只验 `n*` 个时，后面的槽位白算了——这是"并行出 5
  位"的固定成本；调度器已经在 `cost(n)` 里把它摊掉（起草成本与 n 无关）。
- **窗口占位**：槽位本身占窗口宽度（`slot i` 与最老 ctx 的距离是 `depth-i`），所以
  `window` 要 ≥ 想看的上下文 + depth，否则最老的几个 token 会被槽位挤出窗口。
- **Markov 头零初始化**：第一步梯度只进 `markov_out`，`markov_emb` 第二步才开始动
  （不是死参数，别急着删）；训练日志里 `ce_base == ce_markov` 是正常的开局。
- **调度器系数别抄默认值**：0.6 只是"单请求小模型"的量级；实测本机是 0.04（草稿器
  比主干小 9 倍），填大了会永远只验 1 个。
- 与 SWA 缓存组合时投机解码本就禁用（pitfall 37）；`dspark_draft` 不影响 KV 布局。

## 一句话总结

把"多猜几步"从主干的负担里拆出来：3 层小模型并行猜 5 位、Markov 补依赖、置信头
估存活、吞吐曲线定窗口——猜错不要紧，正式员工的验证永远是精确的。
