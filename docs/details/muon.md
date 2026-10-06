# Muon：2D 权重的正交化动量优化器

> 一句话：隐藏层矩阵用"正交化梯度"更新，又快又省显存；embedding/norm 还是 AdamW，各干各的。

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

## 从想法到代码

队列操就是"正交化动量"，2D 才用（向量继续 AdamW）：

```python
# 2D 隐藏权重走 Muon，其余走 AdamW（build_hybrid_optimizer 按形状分流）
M = newton_schulz(momentum)   # 正交化：方向留下，大小抹平
W -= muon_lr * M              # muon_lr 与 AdamW 的 lr 分开调度
```

## 代价与坑

- Muon 偏好大 batch：小 batch 下噪声把正交化带偏，反而不如 AdamW 稳。
  batch×accum 太小就别开。
- 混合优化器意味着两套 lr 调度，看曲线时确认看的是哪条（日志都记了）。

## 一句话总结

大矩阵走队列，小参数走单兵——分开优化，各自最优；切过去记得删动量。
