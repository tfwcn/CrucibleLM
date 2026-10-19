# 分层稀疏 indexer：让"跳读"学会自己找重点

> 一句话：固定图案（sink+窗口+跨步）是**拍脑袋的跳读**；换成一个小学习式打分器
> 先粗选块、再在块里挑 token，让每层自己决定读哪些主 KV。DeepSeek-V4.1 CSA2 的
> Hierarchical Sparse Indexer。

## 痛点

[稀疏注意力](./sparse-attention.md) 用固定 pattern 决定"看哪"：
开头必看、附近细看、中间等距抽点。问题是——
**哪些历史位置真正重要，本该由内容决定，而不是由"位置除 512 余 0"决定**。
一句关键定义可能落在跨步的空隙里，永远读不到。

V4.1 的做法：训练一个轻量 **indexer**（一组 Q/K 打分器）为主 KV 的每个位置打分，
每个查询选 Top-K；再把它**分层**——先按块粗选一批候选（块分=块内最大分），
只在候选池里做 token 级 Top-K。粗选把搜索域从"全文"压到"常数个块"，
于是越深的层 indexer 越便宜。

## 直觉

图书馆找书：固定 pattern 是"每排书架抽一本"；indexer 是**先看目录**（块级粗选，
一本书一个分），在可能是对的那几本书里**逐页翻**（token 级 Top-K）。
目录让搜索域从"整个图书馆"缩到"几本书"，翻页才不贵。

## 原理

```
打分：  s(i,j) = <W_iq·h_i , W_ik·c_kv_j> / sqrt(d_idx)     # indexer Q 来自 hidden，
                                                            # indexer K 由主 KV latent 投影
块级池：blk(j) = pos_j // block；块分 = max_{j∈blk} s(i,j)
        选出分数最高的 P 个块 → 候选池（P×block ≈ 常数）
token 级：只在池内对每个查询取 Top-K → 这些主 KV 才是这一层真正读的
```

三个要点：

1. **indexer K 由主 KV latent 投影**（`w_ik`），不另开一条压缩路径——
   复用已经算出来的 `c_kv`，不新增状态（CSA2 相对 CSA 的简化之一）；
2. **块级池 → token 级 Top-K** 是"分层"的由来：粗选常数化搜索域，
   细选决定最终读谁；
3. **本地 SWA 列不受 indexer 影响**：当前块（含自身）永远可见，
   因果遮罩照旧——indexer 只管"历史主 KV"这一路（V4.1：selected main KV + layer-local SWA KV）。

## 我们的实现

- 文件：`src/llm/local/sparse_attn.py`（`SparseMLAModule._indexer_select` / `_sparse_scores`）
- 参数（`config.py`）：`sparse_indexer`（默认 **关**）、`indexer_heads=4`、
  `indexer_head_dim=64`、`indexer_topk=64`、`indexer_block=8`、`indexer_pool_blocks=32`
  （V4.1：32 头 / 128 维 / top-k 512 / 2048 块 × 8 位置）
- `w_iq`（hidden→indexer Q）、`w_ik`（主 KV latent→indexer K）；打分提 fp32 保稳定
- 落点：`_sparse_scores` 在静态候选列上叠加"未入选掩 `-inf`"的掩码，
  当前块列保持原因果掩码 → softmax 后等价于"只读 top-K 主 KV + 本地 SWA"
- 返回 `IndexerSelect(topk_mask, pool_mask, logits)`：`pool_mask` 即块级候选池，
  供后续 CSA2 **Reindex** 层复用；`logits` 是未掩的 fp32 打分（**蒸馏损失**用，见下节）
- 来源对齐：登记进 `config._ARCH_SWITCHES`（旧 ckpt 回退关）；CLI `--sparse-indexer` 等

### 打分器怎么训：蒸馏损失（否则恒零梯度）

indexer 的产出是**布尔掩码**，只用来填 `-inf`——`masked_fill` 不可微，
所以 `w_iq`/`w_ik` **拿不到任何梯度**（不加信号就是随机初始化在那儿选 Top-K，
"假装在学"）。按 DSA 的做法，用**主注意力自己的分布**当老师：

```
target = softmax(主注意力打分[:, :, 静态候选列]).mean(头)   # detach：只训 indexer
kl     = KL(target ‖ softmax(indexer 打分))                 # 每查询平均（nats）
aux   += indexer_loss_coef * kl                             # 进该层 aux 汇总
```

- 目标**必须 detach**：否则这项会把主注意力也往 indexer 的分布上拽，污染主任务；
- 系数 `indexer_loss_coef`（默认 0.01）只缩放 indexer 自身的等效学习步长，设 0 = 不训；
- 归约别用 `F.kl_div(..., reduction="batchmean")`：它只除 batch 维，`(b, n_q, n_cand)`
  会被 `n_q` 放大成几百 nats（看着吓人）；用 `reduction="none"` 后 `.sum(-1).mean()`；
- 层内多块取均值、层间由模型累加进 `aux_loss`；另以 `indexer_loss`（层间均值、未缩放）
  单独打日志；
- 初期 KL 很小（主注意力与 indexer 都接近均匀），随主注意力变尖而增大——
  这正是 indexer 需要开始追的时段，属正常现象。

## 从想法到代码

"先看目录再翻页"就是**两次 topk**：一次在块上，一次在池内 token 上：

```python
qi = w_iq(h_q)                      # (b, n_q, H, D)
ki = w_ik(c_kv[static_pos])         # (b, n_static, H, D)
s  = einsum("bqhd,bkhd->bqk", qi, ki) / D**0.5
blk = static_pos // block           # 每块一个 id
blk_s = scatter_max(s, blk)         # 块分 = 块内最大
top_blk = blk_s.topk(P).indices     # 目录：选 P 个块
pool = is_block_selected(blk)       # 候选池掩码
topk = s.masked_fill(~pool, -inf).topk(K).indices   # 翻页：池内 Top-K
# 注意力：静态列只留 topk，当前块列照旧（本地 SWA）
# 训练信号：让 s 去拟合主注意力在候选上的分布（target detach）
kl = KL(softmax(主注意力打分[:, :, 静态列]).mean(头) ‖ softmax(s))
```

## 代价与坑

- **不打分器就没梯度**：掩码不可微，务必带蒸馏损失（见上节）；单测
  `test_indexer_gets_gradients_from_distillation` 是硬契约。
- **`sparse_chunk` 太大 = indexer 不参与**：整段序列只有一块时，静态候选（块起点之前的
  token）数为 0，indexer 根本不会被调用——测 indexer 必须把 `sparse_chunk` 调小。
- **改 `state_dict` 键**（新增 `w_iq`/`w_ik`）：属"改键"的 opt-in → **默认关**，
  开启需重训；已登记 `_ARCH_SWITCHES` 让续训自动沿用。
- **只影响长序列**：`t ≤ sparse_threshold` 走 dense（与 MLA 逐位一致），
  indexer 完全不参与——短序列不受影响（单测锁定）。
- **改的是"读哪些"，不是张量形状**：我们的候选集本就是常数规模，
  所以 indexer 的收益在这里是"**学习式选择**"（内容决定重点）而非"把 O(N²) 降成 O(N·K)"。
  真实加速需要 block-sparse kernel（未实现）——这条要诚实写清，别假装省了算力。
- **池交集为空会 NaN**：Reindex 层用自己的池 ∩ 前层池，若不相交则该查询全 `-inf`。
  实现里对"空交集"回退到自己的池，并对"全 `-inf` 行"回退到未掩分数，双保险。
- **块按候选的全局位置分**（不是按候选序号）：跨步候选多落在各自独立的块里，
  窗口段才是连续块——这是"块=连续位置"的忠实近似。
- 打分和池统计是 `(b·n_q, n_blk)` 的布尔/浮点张量；`n_blk ≤ n_static` 有界，
  不会随上下文线性膨胀（这是我们这套简化能省显存的原因）。

## 一句话总结

固定图案是拍脑袋跳读，indexer 是"先看目录再翻页"的学习式跳读；
分层（块级池 → token 级 Top-K）让搜索域常数化，代价是改键（默认关）且真实加速要 kernel 配合。
