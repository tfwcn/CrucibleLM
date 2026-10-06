# RHO：只学不会的 token

> 一句话：每个 batch 只反向 loss 最高的 50% token——会了的不浪费梯度，省的是达标步数。

## 痛点

SFT 数据刷第二遍时，大部分 token 模型早会了（loss≈0），梯度≈0，
反向它们等于空转。RHO（Stanford 的" окружающих data selection"思想的在线版）
只挑难的学，同样步数多学一倍，号称省的不是单步时间，是"达标步数"。

## 直觉

刷题：会了的题跳过，只刷错题本。RHO 就是自动错题本——每 batch 按 loss
排名，只反向最难的前 50%。MTP/aux 全量（路由和远视不能只看局部）。

## 原理

```
token_losses = CE per token（ignore 位恒 0，自然落选）
mask = top50%(token_losses)          # 自参照版：batch 内百分位
mask = top50%(student - teacher)      # 老师版：超额 loss，只在锚点配对位排名
loss = loss - main_mean + (loss·mask).mean()   # 精确扣除：同张量相减再加回
```

两个版本：

1. **自参照**（`--rho-keep 0.5`）：batch 内百分位，零额外开销。
   缺点：batch 全是简单题时，"最难的 50%" 也不难——矮子里拔将军。
2. **老师参照**（`--rho-ref teacher` + `--kd-teacher`）：超额 loss
   （学生−老师）排名，区分"真不会"和"噪声"。老师前向和 KD 共用一次
   （`do_kd` 的 micro 才算，其余 micro 回落自参照），零额外开销。

关键细节：**精确扣除**——`loss - main_mean + selected`，同张量相减，
保证没选中的 token 梯度贡献精确为零（而不是"约等于零"），
保底防空选（自参照 64 个，老师版 8 个——配对位少，阈值跟着小）。

## 我们的实现

- 文件：`src/llm/local/train.py`（`select_topk_loss`）、
  `src/llm/local/distill.py`（`select_by_excess`）
- 开关：`--rho-keep 0.5`、`--rho-ref teacher/none`
- RHO 的 loss 口径天然偏高（只平均难的），**绝不能和旧曲线比大小**
  （AGENTS.md 明文规定；看 val，不看 train loss）

## 从想法到代码

错题本就是 mask，同张量相减保证没选中的梯度精确为零：

```python
mask = top50%(token_losses)                    # 自参照：batch 内百分位
mask = top50%(student - teacher)               # 老师版：超额 loss，锚点位排名
loss = loss - main_mean + (loss * mask).mean() # 精确扣除
```

## 代价与坑

- 省的是步数不是时间：计算图一样大，单步时间不变——别指望 step 更快，
  指望更少 step 达标。
- `select_by_excess` 曾返回展平 mask，batch>1 直接广播炸——修过，
  回归单测锁了形状（pitfalls 风格的老坑）。
- 保底 64 个：如果 batch 全简单，64 个"相对难"的照样反向，无妨。

## 一句话总结

错题本自动化：难的才配吃梯度；看 val 别看 train loss，不然自己吓自己。
