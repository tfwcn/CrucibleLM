# 1M 上下文：64K 稀疏起训 + 阶段延展 + 启动自检

> 一句话：长上下文不是"把 `max_seq_len` 调大"就完事——它是一条**训练路线**：
> 从 64K 起就带稀疏注意力（不做 dense 预热），训练到某个 token 预算再拉长，
> 位置编码用 YaRN 配套外推，最后靠 SWA/CSA2 把解码成本压成近似常数。

## 痛点

三条独立的坑，任何一条踩了都会"跑得动但白烧算力"：

1. **先 dense 短训、再切稀疏长训**：注意力可见集合在训练中途突变，路由和索引器
   都来不及适配，loss 会抖出一段不可恢复的台阶；
2. **训练 8K、推理 1M 却不调 YaRN**：位置编码裸外推，长文里位置互相混叠；
3. **`max_seq_len` 调大了但 `yarn_original_len` 还是旧值**：分界算错，等于
   高频维度被误插值（短距离分辨率崩）。

## 直觉

把"上下文"当成**体育训练**：先在 64K 的场地上把稀疏注意力这套动作练熟
（论文明确"no dense attention warmup"），体能上来（token 预算够）再换更大的场地；
地图（位置编码）要按新场地重新标定，否则跑出去就迷路。

## 原理（V4.1 的口径）

| 阶段 | 序列长度 | 关键约束 |
|---|---|---|
| 预训练起手 | **64K** | 稀疏注意力**从零一起训**，不先跑 dense |
| 延展 | 34T token 时 → **1M** | 位置编码换成大 scale 的 YaRN |
| 部署 | 1M | 单 token Decode FLOPs 近似常数（SWA/CED/CSA2） |

我们把它落成三块**纯逻辑**（不动结构、不影响老 ckpt）：

```
parse_ctx_stages("0:65536,34000000000:1048576")   # 阶段表
seq_len_at(tokens_seen, stages, default)          # 当前该用多长
validate_longctx(config, seq_len)                 # 启动自检（致命→抛错，其余→警告）
```

自检规则：

| 情况 | 处理 |
|---|---|
| `seq_len > max_seq_len` | **报错**（位置缓存装不下） |
| `seq_len ≥ 64K` 且没有任何稀疏机制 | **报错**（违反"64K 稀疏起训、无 dense 预热"） |
| 长于 `yarn_original_len` 但 `yarn_scale=1` | 警告（裸外推） |
| `yarn_scale` 与 `目标/训练` 差 2 倍以上 | 警告（分界白算） |
| `seq_len ≥ 256K` 但层内 SWA 关 | 警告（解码要扫全量 KV） |
| RoPE 缓存 > 256MB | 警告（固定开销） |

## 从想法到代码

```python
# 训练脚本启动时
print(format_report(config, seq_len))     # 自检 + 阶段提示
# 违规直接抛 ValueError，不会"先跑两小时再说"
```

## 我们的实现

| 位置 | 内容 |
|---|---|
| `src/llm/local/longctx.py` | 阶段表、`seq_len_at`、`sparse_ready`、`validate_longctx`、`suggest_yarn`、`rope_cache_bytes` |
| `src/llm/local/rope.py` | YaRN NTK-by-parts（见 `rope-yarn.md`），`yarn_scaling_factor` 上界 512 |
| `src/llm/local/config.py` | `longctx_config()`（200K）、`longctx_1m_config()`（1M：`max_seq_len=1048576`、`yarn_scale=128`、`yarn_original_len=8192`） |
| `scripts/train_local_llm.py` | 启动自检 + `--ctx-ramp`（阶段表；真正切段用续训改 `--seq-len` 重打包） |

## 实测

| 检查 | 结果 |
|---|---|
| 1M 预设自洽 | `max_seq_len=1048576`、`yarn_scale=128`、`yarn_original_len=8192`（128 = 1M/8K） |
| 1M RoPE 缓存 | 64 维时 **512MB**（自检会警告） |
| 64K dense 配置 | 启动**报错**（`sparse_threshold` 调大 + 稀疏全关的构造） |
| 阶段表 | `0:65536` → 34e9:1048576；`seq_len_at(34e9)` 正确切换 |
| 长序列单 token 解码（tiny 4 层、32K 表） | 512→16384 长度：**6.4ms → 20.4ms**（约 3 倍，不是常数） |
| 单测 | `tests/test_longctx.py` 19 项全绿 |

**关于最后一行**：解码耗时没有变成常数，因为当前 `SparseMLAModule._decode` 会把
**整个 latent 缓冲**上投影成 K/V（"先投影再聚集"），成本 O(total)。`kv_block_cache=True`
的块主序路径已经绕开它（远端列按块行取，不再逐步展开全缓冲），层内 SWA 也会把
近端交给环。所以：**要"Decode FLOPs 近常数"，就得开 `kv_block_cache` 或 `swa_window`**；
只靠稀疏掩码省的是注意力里的乘加，省不掉投影。这一点已记进 pitfalls 60。

## 代价与坑

- **64K 起训是硬约束**：`sparse_threshold` 设得比训练长度还大 = 实际在跑 dense，
  自检会直接拦住（这是**训练配方错误**，不是性能问题）。
- **`yarn_original_len` 必须等于真正训过的长度**，改了训练长度要同步改它。
- **`--ctx-ramp` 只做提示与自检**：数据是按 `--seq-len` 一次性打包的，真换阶段要
  续训时改 `--seq-len` 重新打包（`data_cursor` 会接着跳）。
- 1M 的显存账不在 RoPE 缓存（512MB）而在 KV：`kv_lora_rank=512` 时每 token 约 1KB，
  1M 就是 1GB——这才是 `kv_quant` / `compress_ratios` / `swa_window` 要解决的问题。

## 一句话总结

64K 就带稀疏、到预算再拉长、位置编码同步换 scale、解码靠块缓存与 SWA——四条一起，
才是"1M 上下文"，单改 `max_seq_len` 只是把缓存占满。
