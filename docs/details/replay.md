# 回放与课程：防忘的三件套

> 一句话：难样本回放池 + pretrain 掺混 + 新分布巩固期，专治"学了新的忘了旧的"。

## 痛点

SFT 第二遍刷同样数据，val 走 U 型——新分布把旧能力覆盖了（灾难性遗忘的
轻量版）。三件套从三个方向拽住它。

## 直觉

- **回放池**：随身带个错题本，每步掺 15% 旧难题重做（海马体式巩固）；
- **SFT 掺 pretrain**：学说话（SFT）时别忘了认字（pretrain），掺 15% 通用语料；
- **巩固期**：刚换分布的前 N 步，lr 钳住不升、回放加码——搬家先站稳再跑。

## 原理与实现

1. **回放池**（`--replay-ratio 0.15 --replay-capacity 8192`）：
   `src/llm/local/data.py` 的 `ReplayBuffer`（FIFO），每步最难 1 块入池
   （SFT 连 label 一起存），按比例采样重放。日志记 `replay`（本步是否命中）
   与 `replay_size`。
2. **SFT 掺 pretrain**（`--sft-replay-dir data/hq-zh --sft-replay-ratio 0.15`）：
   batch 级混合（`interleave_batches`），**主流耗尽即停**（回放无限循环
   不决定 epoch 长度），回放不计 `data_cursor`（刻意重复）。
   注意：两种 block 同 batcher 但批次同构——混的是 batch 不是 block，
   否则 PackedBatcher 的拼 batch 会炸。
3. **课程**（`--curriculum "0.7:0.0:3000"`）：min-score 阈值线性放开，
   仅有 score 字段的源生效（ultrafineweb），阈值实时读，日志记 `cur_min_score`。
4. **巩固期**（`--consolidate-steps N --consolidate-replay 0.4`）：
   新分布开头 N 步回放加码 + lr 钳制在起始值（warmup 不升）。

## 代价与坑

- 三者默认全关：500 步 A/B（同数据同种子，eval 低 ≥0.1 且 aux 不飘）赢了再开——
  别一上来全开，归因会死。
- 回放池的"难"是用逐 token loss 判的（`token_losses` 前向恒开，供评分用），
  RHO 开了之后"难"的定义和 RHO 是一套，不打架。

## 一句话总结

错题本 + 掺着学 + 搬家先站稳——遗忘是分布的病，药是分布的混。
