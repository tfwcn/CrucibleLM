# 稀疏注意力：200K 的"跳读"策略

> 一句话：query 不看全文，只看"开头 + 附近 + 等距抽样"，计算量从 O(N²) 降到 O(N·K)。

## 痛点

全注意力哪怕配了 MLA，200K 上下文的 `QK^T` 也有 200K×200K = 400 亿个
score——算不动、存不下。稀疏的思路：大多数 score 本来就接近零，
干脆不算。

## 直觉

人读超长文档是跳读的：开头（标题/orientation）必看，当前段落附近细看，
中间大段只抽几个锚点。稀疏 pattern 就是把这套行为写死：
sink（开头 128）+ window（附近 4096）+ stride（每 512 抽 1 个）。

## 原理

对查询区间 `[start, end)`，可见 key 下标 = 三部分并集：

1. **sink 区** `[0, min(128, start))`：注意力汇点（attention sink），
   大量"不知道看哪"时的默认落点，必须保留，否则分布漂；
2. **跨步区** `{j < start : j % 512 == 0}`：长距离锚点；
3. **窗口** `[start-4096, start)` + **本块** `[start, end)`（块内配 tril 因果掩码）。

`select_keys()` 返回排好序的下标 + 静态部分长度，gather 出来做小 attention。
阈值以下（`sparse_threshold=4096`）直接退化成 dense——短序列不值得折腾，
数学逐位一致（单测锁定）。

## 我们的实现

- 文件：`src/llm/local/sparse_attn.py`（`SparseMLAModule`，继承 MLA 只换注意力核心，
  无新增参数，state_dict 与 MLA 通用）
- 缓存仍是 latent（c + rope 键），200K 下每 token 约 320B/层；
- 解码：pattern 按**实际长度**算（`select_keys(total, ...)`），gather 下标全落在
  有效区，空位天然不参与——静态缓存下无需额外掩码；
- softmax 提 fp32（bf16 下长序列 exp 易欠精），再转回；
- 长序列 prefill 按 `sparse_chunk=2048` 分块（显存/速度折中）。

## 代价与坑

- pattern 是手写的启发式，不是学出来的——如果关键信息恰好落在"跳过"的缝里，
  模型永远看不见。200K 的评测要做"大海捞针"（needle-in-haystack）专项，
  不能只看困惑度。
- `select_keys` 是纯 Python 集合逻辑，`torch.compile` 按 `start` 的值特化、
  每步重编——整段 `_decode` 标了 `@torch._dynamo.disable`（eager 跑不重编，
  见 pitfalls 第 9 条后传）。

## 一句话总结

跳读三件套（开头+附近+抽样），200K 可算的代价是"缝里的针可能看不见"。
