# 免辅助损失路由：给专家调"价格"，而不是罚款

> 一句话：不再用 aux loss 惩罚不均衡（那会给主任务塞干扰梯度），而是给每个专家一个
> 可上下浮动的偏置，按负载**符号式**调价——DeepSeek-V4.1 的 `noaux_tc` + `sqrtsoftplus`。

## 痛点

细粒度 MoE 最经典的做法是**辅助损失**：让 router 的"选择概率均值"和"实际命中占比"
尽量一致，不一致就罚。问题是这笔惩罚是**加在主 loss 上的额外梯度**，它会为了均衡去
拉偏 router，和"哪个专家最适合这个 token"的正事互相打架——系数调大掉点、调小失衡，
很难两头兼顾。

## 直觉

aux loss 像"罚款"：均衡了，但顺手把别的也罚了。免辅助损失路由换成**调价**：
每个专家挂一个价格（偏置 `b_i`），选择时按 `打分 + 价格` 排序。

- 谁**太闲**（命中低于平均）→ 抬价 → 更容易被选中；
- 谁**太挤**（命中高于平均）→ 压价 → 少被选中。

价格按负载**只更新一个符号**（升/降），而且**不进反向、不产生干扰梯度**。
均衡由"价格机制"自己收敛，主任务的梯度干干净净。

## 原理

**打分**（V4.1 `scoring_func=sqrtsoftplus`）：

```
score = sqrt(softplus(z))        # z = router(x)，正值、随 z 单调、比 softmax 更宽
```

**选择**（`noaux_tc`：top-k 看"打分+价格"）：

```
sel = score + b                  # b 是逐专家偏置（可正可负，初始 0）
top_i = topk(sel, k)             # 选中哪些专家
```

**权重**（`norm_topk_prob=true` + `routed_scaling_factor`）：

```
w = gather(score, top_i)         # 权重用"原始打分"，不含价格（价格只影响选择）
w = w / Σw                       # 归一化
w = w * routed_scaling_factor    # V4.1 = 1.5
```

**调价**（每个优化器步一次，`sign` 式，不进梯度）：

```
f_i = 本轮命中专家 i 的占比        # Σ_i f_i = 1
b_i += γ · sign(1/E − f_i)       # 欠载(f<1/E)→抬价；过载→压价
```

与旧口径的对照：

| | 旧口径（本项目原实现） | 免辅助损失（V4.1） |
|---|---|---|
| 打分 | `softmax(z)` | `sqrt(softplus(z))` |
| 选择 | 按打分 top-k | 按 `打分 + 偏置` top-k |
| 权重 | 打分 top-k 归一化 | 原始打分归一化 × `routed_scaling_factor` |
| 均衡 | aux loss（加在主 loss 上） | 逐专家偏置（sign 式，不进梯度） |

## 我们的实现

- 文件：`src/llm/local/moe.py`
  - `_score`：按 `router_aux_free` 选 sqrtsoftplus / softmax；
  - `_route`：**唯一**路由入口，分组 bmm 与逐专家/offload 两条路径共用（避免两份拷贝）；
  - `_aux_loss`：仅旧口径使用的负载均衡损失；
  - `_accumulate_load`：记录最近一次训练前向的逐专家命中占比（**赋值**语义，见坑）；
  - `update_router_bias`：按 `sign(1/E − f)` 更新逐专家偏置。
- 配置：`router_aux_free`（默认 True）、`router_bias_lr`（γ，默认 1e-3）、
  `routed_scaling_factor`（默认 1.5）、`norm_topk_prob`（默认 True）、`aux_loss_coef`（旧口径用）。
- 接线：`model.update_router_bias()` 逐层转发；`train.py:train_step` 与训练脚本主循环在
  `optimizer.step()` 之后各调用一次。
- **来源对齐**：属"改前向语义不改键"的开关，登记在 `config._ARCH_SWITCHES`
  （`router_aux_free`/`routed_scaling_factor`/`norm_topk_prob`），旧 run 缺字段回退旧口径。

## 从想法到代码

"调价"= 偏置，键在"选择用带价打分、权重用原始打分"：

```python
score = sqrt(softplus(router(x)))          # 打分
top_i = topk(score + b, k)                 # 选（带价）
w = normalize(gather(score, top_i)) * 1.5  # 权重（不带价）
# 每步：b += γ * sign(1/E - 命中占比)       # 调价，不进反向
```

## 代价与坑

- **偏置不进 `state_dict`**（非持久化 buffer）：好处是装载键兼容（旧 checkpoint 照常 strict
  载入）；代价是**续训时偏置归零**、需数步重新自适应。这是"键兼容 vs 续跑精确"的取舍，
  本项目选键兼容（与全仓库一致）。
- **调价用的是"最近一次训练前向"的负载**，不是整个梯度累积批的负载：`update_router_bias`
  每优化器步调用一次，取最后一个 micro-batch 的统计。`sign` 式对噪声鲁棒，但这是近似，
  写在这里备查。前向里用**赋值**（而非累加）保存负载，是为了让梯度检查点的**重算幂等**——
  累加会在反向重算里被计两次。
- **日志口径变化**（约定 6）：开启后 `aux_loss` 恒为 0，别再拿它当"路由健康度"看；要看
  均衡改看偏置漂移或逐专家命中占比。
- **`routed_scaling_factor` 改残差量级**（专家输出 ×1.5），属前向数学变更——旧 run 必须靠
  hparams 回退到 1.0，否则静默改数字。
- **阈值/均衡需长训验证**：本项完成"结构就绪 + 单测锁定机制"，是否比 aux loss 更优需要
  同起点 A/B（本项目 128M 单卡长训）；短测只能证明机制正确（负载能被摊平），不能证明收敛更好。

## 一句话总结

把"罚款式均衡"换成"价格式均衡"：选择按 `打分+价格`、权重按原始打分、价格按负载符号式
升降且不进反向；主任务梯度干净，均衡自己收敛——代价是价格不进存档、续训重来。
