# 主 KV 逐层压缩比：一个条目代表 m 个 token

> 一句话：远端的历史不必**逐 token** 留档——把相邻 m 个 token 的主 KV 压成**一个**（学习式加权平均），
> 同样的远端列数预算下模型就能"看到" m 倍的历史；近处（本地窗口 + 当前块）保持未压缩，因为近处要精确。
> 这就是 V4.1 `CSA2(ratio, mode)` 里的 `ratio`。

## 痛点

稀疏 pattern（sink + stride + window）本身已经在丢信息：它每隔 `stride` 个 token 才留一个远端
候选，**丢掉的是中间那些 token 的全部内容**。序列越长、`stride` 越大，丢得越狠。
于是"远端只有几个孤立快照"，模型的远记忆既稀又薄。

## 直觉

读书时我们不做"每 256 页摘一句"，而是做**每 256 页摘一段**——一页摘要是 4 页的概括，比 4 个孤立
句子信息量大得多。压缩比 `m` 就是这个"一段几页"：

```
token:   0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
m=4 条目:  [块 0]   [块 1]     [块 2]      [块 3]
           ↑ 块内 m 个 token 的 latent 加权平均（权重可学）
```

- `m=1`：一个条目 = 一个 token（**旧行为，逐位不变**）；
- `m>1`：一个条目 = m 个连续 token 的摘要，块与块**不重叠**（CSA2 相对 CSA 的关键改动：
  CSA 的相邻块重叠，一半算力在做重复摘要；CSA2 去掉重叠，同样列数覆盖更多 token）。

## 原理（DeepSeek-V4.1 的口径）

V4.1 的每个注意力层有一个二元组 **`CSA2(ratio, mode)`** 静态分配：

| 分量 | 含义 | 本仓库 |
|---|---|---|
| `mode` | Full / Reindex / Reuse（谁生产 main KV、谁选 Top-K） | `csa_modes`（见 `csa-modes.md`） |
| `ratio` | 主 KV 的**时间轴压缩比** m（每条目聚合的连续 token 数） | `compress_ratios`（本篇） |

两条正交的轴：`mode` 决定**层次结构上** KV 复用到什么程度；`ratio` 决定**条目粒度**。
V4.1 编码半用 `m=2`、解码半用 `m=1`——预填充时省列数（便宜），逐 token 解码时保持精确。

论文里 **SWA KV 层内不压缩**：本地窗口（近处）必须逐 token 精确，压缩只作用于远端主 KV。
所以压缩列与未压缩的本地列**同时**参与同一次注意力。

## 我们的实现

### 1. 压缩比分配：`config.compress_ratios` + `compress_ratio_of`

- `compress_ratios`（默认 `""`）逗号分隔的整数表，**按全注意力层序号循环**（保序、保重复，
  因为它是循环表不是集合）：`"2"` → 全注意力层全用 m=2；`"2,1"` → 交替；
- `compress_ratio_of(layer_idx)`：**非全注意力层恒为 1**（线性/SWA 层不受影响）；
- `m < 1` / 非整数直接 `ValueError`（早失败好过静默按不压缩跑）；
- m>1 的层新增 `z_cmp` 参数 → **改 `state_dict` 键**，故默认关、登记进 `_ARCH_SWITCHES`，
  续训/迁移必须 `sync_arch_from_checkpoint` 沿用来源（读不到回退空表 = 全 1）。

### 2. 池化：块内逐位置逐通道可学习权重

```python
# src/llm/local/sparse_attn.py:_compressed_columns
m = self.compress_ratio                      # 每条目聚合的 token 数
pos     = blk_ids[:, None] * m + arange(m)   # 块 b 覆盖 [b*m, (b+1)*m)
members = c_src[:, pos]                      # (b, B, m, r)，index_select 可微
c_blk   = (members * self.z_cmp).sum(2) / m  # z 初值 1 → 块内算术平均
blk_end = (blk_ids + 1) * m - 1              # 条目代表位置 = 块尾
kr_blk  = apply_rope(kr_src[:, pos[:, -1]], rope_cos[blk_end], ...)
```

- `z_cmp` 形状 `(m, kv_lora_rank)`，**只对 m>1 的层建**（m=1 不建 → 逐位不变、不加键）；
- 初值全 1 → 池化 = 算术平均（"压缩"的最中性起点），梯度可把它学成"块内更看重某几个位置/通道"；
- 代表位置取**块尾**：该条目覆盖到块尾为止，与本地窗口 `[start-window, start)` 无缝衔接
  （否则块内后半个 token 会在窗口和压缩条目之间"漏掉"）。

### 3. 远端选谁：pattern 从 token 轴搬到**块轴**

```python
# _far_block_ids：只取"完整且在查询块之前"的块
n_valid = min(total // m, start // m)                      # 完整 + 因果（绝不看未来）
n_far   = min(n_valid, (start - sparse_window + m - 1) // m)
blk_ids = select_block_keys(n_far, sink, stride, m)        # 同 sink/stride 语义，单位=块
```

- **因果**：`start // m` 夹取保证块尾 `< start`，绝不看未来 token；
- **完整**：`total // m` 保证条目落在真实存在的数据区（解码时缓存是预分配缓冲，
  越界行是未写过的填充）；
- `sink` 换算成 `ceil(sink/m)` 块（覆盖开头约 sink 个 token），`stride` 换算成
  `max(1, stride//m)` 块——**保持同样的列数预算**；
- 本地列（窗口 `[start-window, start)` + 当前块 `[start, end)`）一律**未压缩**，
  与压缩列拼在同一张列表里：静态列 = 压缩块列 + 窗口列，块内因果 = 当前块列。

好处是远端条目数不变（甚至更少），而每个条目代表 m 个 token：
**列数预算不变，远端"看得见"的 token 数 ×m**。实测（12 层 / 2048 上下文 / sink=64 / stride=256 /
窗口 512）：

| m | 远端列数 | 覆盖到位置 | 表示 token 数 |
|---|---|---|---|
| 1 | 69 | 1281 | 69 |
| 2 | 37 | 1282 | 74 |
| 4 | 21 | 1284 | 84 |

覆盖跨度不变（同样的 stride 网格），但每条目携带的信息量不同：m=1 时那些位置**只有单 token 快照**，
m=4 时是 4 token 摘要——这正是论文里"压缩比换 KV 容量/质量"的那根旋钮。

### 4. 传参：池化源要跟着"谁生产 main KV"走

- `mode=Full`：池化源 = 本层自己的 latent（预填充取当前段 `w_dkv(h)`，解码取自己的缓存缓冲）；
- `mode=Reindex/Reuse`：池化源 = **生产者的** latent 缓冲（借用层连缓存都是共享对象）；
- **借用层沿用生产者的 `compress_ratio` 与 `z_cmp`（同一对象）**：否则同一条 main KV
  在两层里被池化成不同条目，"复用 main KV"就名不副实。生产者不压缩（m=1）时，
  借用层把自己多余的 `z_cmp` **删掉**（`del`），不留"永不更新"的死参数。

### 5. 压缩开启 → 恒走分块路径

`compress_ratio > 1` 时**不再**对短序列走 dense（`sparse_threshold` 失效）：
dense 用未压缩 latent，与压缩表示不一致，同一模型在短/长序列会看到两种 KV 表示。
单测锁定"`sparse_threshold=1` 与 `4096` 结果逐位相同"。

## 从想法到代码

```
痛点：远端 stride 采样只留孤立单 token 快照 → 远记忆又稀又薄
  ↓
直觉：一段摘要 > 几个孤立句子；m = 一段几页
  ↓
原理：CSA2(ratio, mode)；m 条目不重叠；本地 SWA 不压缩；代表位置 = 块尾
  ↓
代码：
  config.compress_ratios → compress_ratio_of(layer_idx)   # 逐全注意力层，非全注意力层恒 1
  HybridBlock(compress_ratio=...) → SparseMLAModule.z_cmp  # 只 m>1 建参数
  _far_block_ids()      → 块轴的 sink/stride，且完整 + 因果
  _compressed_columns() → Σ z⊙c / m + 块尾 rope → 拼在本地未压缩列之前
  _link_csa_reuse()     → 借用层共享生产者的 m 与 z_cmp（生产者 m=1 则删掉自己的）
```

## 代价与边界

- **本层注意力列数下降**（远端列从 sink+span/stride 变成 sink/m+span/stride），
  故同样的远记忆质量下更便宜；反过来同预算下远记忆更厚。
- **KV 显存尚未下降**：缓存仍按 token 逐条存原始 latent（池化在读取时做）。
  真正的显存收益要求"冻结块落盘成单条目 + 未压缩窗口用定长环"的**缓存布局重构**
  （块主序 + 环形窗口），与 #7「每层 SWA 环形缓存」天然同工——届时 `kv_cache_bytes`
  才能按 `L/m + window` 计。本篇先把**表示语义**（模型看到什么）对齐。
- 建议 `m ≤ sparse_window`：`m > 窗口` 时窗口与压缩列之间会漏掉不足一个块的近端 token
  （`_far_block_ids` 的因果夹取不补这一段）。
- 池化权重 `z_cmp` 与主注意力一起端到端学（首块 `n_static=0` 时不被调用；
  真实规模冒烟里每层都能拿到非零梯度）。
- 单测：`tests/test_compress_ratios.py`（m=1 逐位等价 + 不加键 / 块池化数学 / 块尾 rope /
  块轴 pattern 覆盖 / 压缩列真参与（改未入选块不影响输出）/ 因果 + Finite /
  `prefill+解码 == 全前向`（chunk 对齐）/ threshold 失效 / 借用层沿用生产者 /
  来源对齐 + CLI 三态）。
