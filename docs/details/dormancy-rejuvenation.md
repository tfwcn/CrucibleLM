# 神经元复壮：找死掉的神经元，恒等式重开

> 一句话：扫描长期零激活的 FFN 神经元，输入侧随机重开 + 输出侧置零（输出不变），让难数据去教它们——实测零休眠，手术封存。

## 痛点

小模型训到后期，一批权重可能在"摸鱼"（接近零、无贡献）。想法：
把摸鱼的找出来重开，难数据会教会它们——容量回收，不加参数加 adequate。

## 直觉

公司末位淘汰：先考勤（激活扫描）找出长期缺勤的，再招新人顶位置，
但交接期业务不能停（恒等式：输出不变），新人只接最难的单（RHO 的难 token
天然流向它们，因为旧人把 easy 的分都吃完了，梯度≈0）。

## 原理

1. **休眠判定看激活不看权重**（ReDo 原文结论）：magnitude 小可能是被抑制的
   重要特征；中间激活（`silu(gate)×up`）长期≈0 才是真死。
   `score = mean(|act|)`，阈值 = 层均值 × 1e-3。
2. **恒等手术**：`w_gate[j]`、`w_up[j]` 随机重开，`w_down[:, j]` 置零——
   新激活再大，乘零也是零，输出逐位不变，无 loss 尖峰
   （和记忆层 value 零初始化、retro `w_o` 零初始化同一哲学）。
   但注意：恒等只对**真休眠**成立——置零删的是既有贡献，
   活性单元必变（单测证伪过，这正是必须按激活选的理由）。
3. **两步起速**：step1 梯度进输出列（随机中间激活×上游梯度），
   step2 起输入侧吃到梯度——零起速，安全动力学（单测锁）。
4. **配额与时机**：单次 ≤20%/专家（防 superposition 连根拔），
   卡 stage 边界 + `--consolidate-steps` 冷却，新开优化器（无 stale 动量）。

## 我们的实现与实测

- 文件：`src/llm/local/rejuvenate.py`
  （`collect_mid_activations` 流式累加 O(1) 内存 + `dormancy_masks` +
  `apply_rejuvenation`）,`scripts/scan_dormancy.py`，`scripts/rejuvenate.py`
  （自检阈值 1e-3，超了自动中止落盘）
- **实测结论（sft5-best，12 层×17 专家，1024 段文本）：全局休眠 0.0000，
  最闲神经元仍有均值的 26%**——无对象可开，手术封存，A/B 免了。
  交叉验证：aux 0.12（路由均衡）+ 零休眠 = 模型已被榨干，
  "效果一般"不是容量闲置，杠杆只剩数据与规模（见 ARCHITECTURE 容量节）。

## 从想法到代码

考勤即 hook 均值，交接即"输入重开 + 输出置零"：

```python
mid = silu(expert.w_gate(x)) * expert.w_up(x)  # hook 在 MoE 上重算（专家 forward 已不跑）
score = mean(|mid|); dormant = score < 1e-3 * 层均值
expert.w_gate[j].normal_(); expert.w_up[j].normal_()  # 输入重开
expert.w_down[:, j].zero_()                          # 输出置零 → 恒等
```

## 代价与坑

- 分组 bmm 后专家子模块 forward 不再被调用——hook 必须挂在 MoE 上重算
  中间激活，挂专家上收不到数据（实测空 buffer，调试半小时）。
- RHO-teacher 这类"按位置选"的单测，注意 batch 内路由是输入相关的：
  probe 和训练必须用**同一个 x**，重画一个路由就换人，梯度结构性为零。

## 一句话总结

考勤（激活扫描）→ 恒等交接 → 难单培养——机制全对，但本模型全员在岗，无人可裁。
