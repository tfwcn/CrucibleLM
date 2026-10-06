# 采样与生成：prefill 一次、单步续写

> 一句话：prefill 建缓存，单步采样续写；贪心必配重复惩罚，不然掉进 `<` 循环。

## 痛点

自回归生成 = 同一个前向跑几百次。朴素实现每步重算全部历史（O(N²) 总量），
200K 直接去世。必须：prefill 一次建缓存 + 每步只算 1 个 token。

## 直觉

盖楼：prefill 是打地基（一次），解码是一层层砌砖（每步只砌一块，
但脚手架=缓存一直在）。采样器是"选哪块砖"：贪心拿最像的，
温度采样掷骰子，top-k 先扔掉差生再掷，重复惩罚专治"老拿同一块砖"。

## 原理与实现

- 文件：`src/llm/local/model.py`（`generate`/`_generate_inner`/`stream_tokens`/
  `_sample_next`/`_prefill`/`_decode_step`）
- `_prefill(input_ids, max_new_tokens)`：整段 prompt 一次前向，静态缓存精确预留；
  `_decode_step(nxt, pasts)`：单步进各层（attn+MoE+记忆+retro 全同步，
  漏一个就分叉，pitfalls 第 7 条）；
- `generate` 与 `stream_tokens` 同一基元（单测锁逐位一致），SSE 流式直接调后者；
- **train/eval 保护**：生成前后恢复模式（训练中穿插采样若把模型留在 eval，
  梯度检查点被静默关闭，显存爆炸——pitfalls 级教训，单测锁）；
- **重复惩罚**（`repetition_penalty`，HF 语义：已出现 token 正 logit 除、
  负 logit 乘，默认 1.0 关闭）：贪心配 1.2~1.4，实测 4-gram 复读 0.58→0.04。
  SFT 模板标签（`<用户>`）和 markdown（`####`）是超高频 token，
  不惩罚必掉进循环吸引子——且与训练步数无关（三档权重 `<` 占比一模一样，
  pitfalls 第 8 条）；
- 服务端 `repetition_penalty` + `top_k` 全透传（`top_k` 曾解析了又丢掉，
  README 却写透传，顺手修实了）。

## 从想法到代码

采样即"改 logits 再掷骰子"，惩罚即"见过的降权"：

```python
logits = lm_head(h_last)[:, -1, :]
seen = scatter(past_ids)                       # 上下文出现过的位置
logits = where(seen, where(logits < 0, logits*p, logits/p), logits)  # HF 语义
nxt = argmax(logits) if temp == 0 else multinomial(softmax(logits/temp))
```

## 代价与坑

- 贪心 + 无惩罚 = 循环 Fuji：sample 评测必须固定 prompt 集看复读率数字，
  别只看自我介绍那一题（幸存者偏差，pitfalls 第 8 条）。
- `stream_tokens` 只 yield batch 第 0 条——batch 推理走 `generate`，别混。

## 一句话总结

地基一次，砖一块块砌，选砖用骰子+惩罚——生成质量一半在权重，一半在采样器。
