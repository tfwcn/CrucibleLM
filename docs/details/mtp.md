# MTP：让模型多看一步

> 一句话：除了预测下一个 token，再加个头预测下下个——训练信号翻倍，推理时关掉不花钱。

## 痛点

标准 LM 训练每个位置只学一个信号（t+1）。语料就那么多，信号密度决定
样本效率。MTP（Multi-Token Prediction，DeepSeek 系）白嫖一份：用同一个
hidden 再押一注 t+2。

## 直觉

做阅读理解时，好学生不只想到下一句，还想到下下句。MTP 就是逼模型
"想远一步"——被迫学更长程的规划，而不是只学局部搭配。考试（推理）时
不加试，平时（训练）多练。

## 原理

```
h         → lm_head → 预测 t+1（主 loss）
MTP(h[:-2]) → lm_head → 预测 t+2（mtp_loss × 0.3）
```

MTP 头是个浅层块（`norm + proj`，`mtp_depth=1`），复用主词表头，
不新增大矩阵。loss 权重 0.3——MTP 是"辅修"，不能喧宾夺主（t+2 本来就更难，
权重太大主 loss 反而学不好）。

## 我们的实现

- 文件：`src/llm/local/model.py`（`MTPHead`）
- 参数：`mtp_depth=1`、`mtp_loss_weight=0.3`
- RHO 选择时 MTP 全量（aux 和 MTP 都必须看全路由/全序列，不能只挑难的）；
- 推理/评测：MTP 头直接不用，零成本。

## 从想法到代码

多押一注就是多一个浅头，复用主词表头，不新增大矩阵：

```python
mtp_h = self.mtp(h[:, :-2])          # norm + proj，独立表达空间
mtp_loss = CE(lm_head(mtp_h), targets[:, 2:])  # 拿 i 处 hidden 押 i+2
loss = main + 0.3 * mtp_loss + aux
```

## 代价与坑

- MTP loss 天然比主 loss 高（t+2 更难猜，实测 3.8~4.7 vs 主 1.6~2.0），
  看曲线别慌，这是口径问题（AGENTS.md 也记了：loss 口径变化要同步解读）。
- 浅层头表达能力弱，t+2 猜不对是常态——它的价值是给主干提供"远视"梯度，
  不是它自己猜多准。

## 一句话总结

训练时多押一注，推理时不花钱——样本效率的免费午餐（打三折吃）。
