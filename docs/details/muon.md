# Muon：2D 权重的正交化动量优化器

> 一句话：隐藏层矩阵用"正交化梯度"更新，又快又省显存；**Q/K 上投影按头分块**
> （head-wise，每头一个预条件器）；embedding/norm 还是 AdamW（或 Sinkhorn，见
> [sinkhorn-update](./sinkhorn-update.md)），各干各的。

## 痛点

AdamW 给每个参数存 2 份状态（动量 + 方差），128M 倒是不大，但 2D 大矩阵上
AdamW 的逐元素自适应其实是"杀鸡用牛刀"——真正有用的是更新方向的整体形状。
Moonshot（Kimi）验证过：矩阵参数用 Muon，同等步数效果更好，状态还减半。

## 直觉

AdamW 像给每个士兵单独配给养（逐参数自适应）；Muon 像整队列操（Newton-Schulz
迭代把动量矩阵"掰正"成交叉正交的方阵再走）。队列整齐，步子就大；
而且只存 1 份动量，显存减半。

## 原理

```
Muon: M = βM + G；M = NewtonSchulz(M)（正交化）；W -= lr·M
```

Newton-Schulz 是不用 SVD 求近似正交因子的迭代（5 步左右收敛，全是矩阵乘，
GPU 友好）。直觉：把梯度矩阵的"方向"留下，"大小"抹平——大矩阵更新最怕
某些方向一家独大。

只用于 **2D 隐藏权重**（attention/MLP 的矩阵）；embedding、norm、
router 这些向量/特殊参数继续 AdamW——不同形状不同脾气，硬套反而坏事。
`--muon-lr` 和 `--lr` 分开调度（Muon 步子天然大，默认 0.02 vs AdamW 的 3e-4
量级，不能共用）。

## 我们的实现

- 文件：`src/llm/local/optim.py`（`build_hybrid_optimizer`）
- 开关：`--optimizer muon`（默认关，默认 AdamW）
- 注意：Muon 存盘和 AdamW **不互通**——切换优化器要删 `optim.pt` 重开动量
  （权重不受影响，只丢动量，warmup 几天就回来）

### head-wise Muon（V4.1 §2.5，`--muon-head-wise`）

把 Muon 看成**预条件梯度下降**：原版给所有头共用**一个**预条件器，而注意力头
之间是异质的（不同头学不同模式），共用一个预条件器等于强行把它们的更新形状
拉齐。head-wise 的做法是：Q/K 的上投影按**头**切块（每头一段连续行
`head_dim × in`），**每个块各自**做 Newton-Schulz——每头一个预条件器。
论文实测优于原版 Muon（GLM5 / Kimi-K3 同样验证）。

- 实现：`head_split_columns()` 把 `(n_heads*per, in)` 切成 `(H, per, in)`，
  `zeropower_batched_newtonschulz5()` 用 `bmm` 一次处理所有头（与逐头循环的普通
  NS5 **逐位一致**，单测锁定）；
- 作用于 `w_uq`（Q 内容上投影）、`w_qr`（Q 的 rope 部分）、`w_uk`（K 上投影）；
  `w_uv`/下投影**不切**（论文口径是"Q and K weights"）；
- 行数必须能被头数整除，否则早失败（说明不是按头堆叠的权重）。

## 从想法到代码

队列操就是"正交化动量"，2D 才用（向量继续 AdamW）；Q/K 再按头切一层：

```python
# 2D 隐藏权重走 Muon，其余走 AdamW（build_hybrid_optimizer 按形状分流）
M = newton_schulz(momentum)   # 正交化：方向留下，大小抹平
W -= muon_lr * M              # muon_lr 与 AdamW 的 lr 分开调度

# head-wise（V4.1）：Q/K 上投影按行切成 H 块，逐头正交化
heads = W.view(H, rows // H, cols)          # 每头一段连续行
W -= muon_lr * batched_ns5(heads).reshape_as(W)
```

## 代价与坑

- Muon 偏好大 batch：小 batch 下噪声把正交化带偏，反而不如 AdamW 稳。
  batch×accum 太小就别开。
- 混合优化器意味着两套 lr 调度，看曲线时确认看的是哪条（日志都记了）。
- **head-wise 的批量 NS5 要按头 reshape**：`(H, per, in)` 的 `bmm` 与逐头循环等价，
  但**不能**把整行视作一个矩阵（那就退化成原版 Muon 了）。
- 优化器分组变了（head_wise/sinkhorn 开关切换）→ 旧 `optim.pt` 的分组对不上，
  装载时**早失败**并提示删 `optim.pt`（权重照常复用，只丢动量）。

## 一句话总结

大矩阵走队列，小参数走单兵——分开优化，各自最优；切过去记得删动量。
