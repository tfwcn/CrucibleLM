# 块主序主 KV 缓存：一个条目就是一行缓存

> 一句话：`compress_ratios` 只让模型**看到**更长的远端（同样的列数覆盖 m 倍历史），
> 但缓存里仍然逐 token 存着每一个 latent——**压缩发生在读取时，存储一点没省**。
> 块主序缓存把压缩提前到**写入时**：每 m 个 token 折成一行存下来，
> 主 KV 行数从 `L` 降到 `ceil(L/m)`。

## 痛点

`compress_ratios=m` 之后，注意力只读 `ceil(L/m)` 个远端列，但缓存仍然存 `L` 行 latent：

```
token 轴:  0 1 2 3 4 5 6 7 ...
缓存:      c0 c1 c2 c3 c4 c5 c6 c7 ...   ← L 行（每行 kv_lora_rank + qk_rope_dim）
读取:      [c0+c1+c2+c3]/4  [c4+..+c7]/4   ← 每步现算（O(m) 次加法/列）
```

于是"压缩"只省了注意力的列数（FLOPs 侧），**显存/带宽一行没省**，而且每次读远端都要
把 m 个成员再捞出来池化一遍。序列越长，这个"存了不用"的浪费越明显。

## 直觉

给读书笔记做摘要：与其把每页笔记都塞进抽屉、需要时再临时拼一段摘要，不如**每读完 4 页
就做一张卡片放进去**。抽屉里只有卡片（`L/4` 张），翻起来也快。

那"正在读、还没凑满 4 页"的那几页怎么办？夹在书里（**暂存前缀**），不进抽屉——
因为一张不完整的卡片对以后的翻查没有意义。

## 原理（DeepSeek-V4.1 的口径）

V4.1 的 CSA2 是 `CSA2(ratio, mode)`：压缩条目**本身**就是缓存单元，近端由层内 SWA 承担
（`each layer retains its own SWA KV`）。也就是说：

- **远端主 KV**：只存压缩条目（m 个 token 一行）；
- **近端**：SWA 环（窗口）逐 token 精确，当前块也在 SWA 里；
- 两者拼进**同一次 softmax**（见 `per-layer-swa.md`）。

本项目 `compress_ratios` 实现的是"条目表示"（读时池化），本篇实现"条目存储"
（写时折行）——两者数学同源：行值就是 `Σ_{i<m} z_i ⊙ c_i / m`，`z_cmp` 可学习、初值 1。

## 布局

一层主 KV 缓存（`compress_ratio=m>1` 的层）是**一个**张量，行号有明确语义：

```
行:  [ 0 .. m-1 ]        [ m .. m+n_blk-1 ]
     ↑ 暂存前缀           ↑ 块行（块号 b 对应行 m+b）
     一行一个 token        一行 = 块内 m 个 token 的加权均值
     （当前残块）           rope 行 = 块尾 token 的**未旋转**键
```

- 块行数 `n_blk = ceil(L/m)`；总行数 `m + ceil(L/m)`（m 行暂存是定长开销，可忽略）。
- **暂存前缀放在头部**：`ensure_room` 只在尾部扩容（2x），前缀天然不被搬走 →
  残块里的 token 不会因为扩容丢上下文（pitfall 50）。
- 只读块行 `[m + b]`，其中 `b < start // m`：块没凑满 m 个 token 前**任何查询都读不到它**。

## 写路径（唯一入口 `_block_add`）

预填充（一次长 prompt）与解码（逐步）走**同一段代码**，分三段：

| 段 | 覆盖 | 做什么 |
|---|---|---|
| ① 当前残块 | `pos % m` 起的 `n_a = min(t, 块尾 - pos)` 个 | 逐 token 写暂存前缀；凑满 m 个立刻 `_flush_block` |
| ② 中间的完整块 | 之后整块整块的 `n_full = rest // m` 个 | 直接对本步 latent 池化（不过暂存，省一次读写） |
| ③ 尾部残块 | 剩下的 `rest % m` 个 | 写暂存前缀第 `0..rem` 行，等后续步补齐 |

为什么非要统一：预填充时"一次池化"和解码时"逐步累加"在浮点/量化下**不是同一个数**。
早期实现让解码沿块行累加、预填充一次池化，结果 prompt 长度不是 m 的倍数时解码与全前向
差 0.03（fp8）/0.08（fp4）（pitfall 48）。现在两条路径共用 `_block_add`，量化也只在
`_flush_block` 里发生一次。

## 读路径（`_block_columns`）

- 取块行 → `c_src[:, m + blk_ids]`，`kr_src[:, m + blk_ids]`；
- 投影 `k = k_norm(w_uk(c_blk))`、`v = w_uv(c_blk)`（与逐 token 路径**共用**
  `_project_block_kv`）；
- rope 用**块尾位置** `(b+1)*m - 1`（这条目覆盖到块尾为止）；
- `c_blk` 送 indexer 当候选 latent（候选就是块行，不是 token）。

块主序下 `n_chunk` 恒为 0、`local` 恒为空：近端（窗口 + 当前块）全由层内 SWA 提供，
全局主 KV 只出"完整块"的远端列。因此解码时连 `k_full`/`v_full` 的逐 token 展开都省了
（每步少一次整缓冲上投影，而缓冲本来就只有 `L/m` 行）。

## 前提（构建期早失败）

| 前提 | 原因 |
|---|---|
| `compress_ratios` 里有 `m>1` | 全 m=1 时块行 = token 行，纯赔本 |
| `swa_window > 0` | 当前块那不足 m 个 token 不在主 KV 里，必须由 SWA 承担 |
| `swa_window >= max(m)` | 块边界左侧剩的 `pos % m` 个 token 只能从 SWA 读，窗口不够就静默丢上下文 |

`TinyLLM.__init__` 调 `config.block_cache_validated()`，三条都不满足时直接 `ValueError`。
m=1 的层（混合压缩比时）自动保持逐 token 布局。

## 代价与收益

- 收益：主 KV 行数 `L → m + ceil(L/m)`。`kv_cache_bytes` 逐层按各自 m 计费。
  实测（12 层 / 107.7M / 混合 `compress_ratios="4,2"` / `swa_window=128` / fp8）：
  8192 ctx 主 KV 3.24 → 1.74 MiB（0.54x）、32768 ctx 10.41 → 4.43 MiB（0.43x）
  ——没到 1/m 是因为还叠加了 SWA 环与线性层状态这些"不随 m 缩"的项。
- 代价 1：**残块不可读**——块没凑满 m 个 token 前，它的信息只存在于 SWA 窗口里。
  若 `m` 大于窗口，块边界附近的上下文就会缺（故强制 `swa_window >= m`）。
- 代价 2：量化粒度变粗。块均值是"新的缓存单元"，量化发生在它上面（论文口径一致）：
  与"逐 token 量化再池化"相比误差略大，但 QAT 训练与推理**自洽**（预填充/解码逐位同值）。
- 代价 3：m 行的定长暂存前缀（可忽略）。

## 契约（单测锁定，`tests/test_block_cache.py`）

1. 缺前提的三种组合构建期抛错；
2. **不新增 `state_dict` 键**；
3. 行数 = `m + ceil(reserve/m)`（混合压缩比时逐层不同）；
4. 块行 = `Σ z_i⊙c_i/m`；rope 行 = 块尾未旋转键；残块只进暂存；
5. 同权重下"开/关块主序"的**预填充输出逐位相同（0.0）**、逐 token 解码也逐位相同；
6. 单 token 解码（含跨残块边界）== 全前向（<1e-5）；块对齐的 4-token 解码同样 ==；
7. `kv_quant` 叠加：块均值只量化一次 → 跨残块解码仍 == 全前向（fp8/fp4）；
8. indexer / CSA2 Reuse 借用共享块缓冲时同样逐位一致；
9. `ensure_room` 扩容后残块不丢（下一步仍 == 全前向）；
10. `SessionCache` 落盘/载入：`drop_swa=False` 逐位一致；`drop_swa=True` 只存全局 KV，
    载入的重放结果**优于无环冷启动**（块行是权威值，不参与重放写入）。

## 不做的事

- **块稀疏 kernel**：行数省了、读得也少，但注意力仍按 dense 列张量算（真实加速要
  block-sparse kernel，与 indexer 的注释同一口径）。
- **跨层共享块行**：CSA2 的 Reuse/Reindex 已经共享缓冲，块主序沿用生产者布局，不额外引入。

## 打通方式

```bash
# 前提：压缩比 + 层内 SWA（窗口不小于 m）
python scripts/train_local_llm.py --compress-ratios 4 --swa-window 32 \
    --kv-block-cache            # 关：--no-kv-block-cache
```

启动日志会打印"层→m"的映射与行数变化。旧 checkpoint 从 `hparams.json` 沿用
（`_ARCH_SWITCHES` 里的 `kv_block_cache`，不读则回退关）。
