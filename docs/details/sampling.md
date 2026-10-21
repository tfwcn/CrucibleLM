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

## 确定性采样（`seed`，对标 DeepInfer deterministic sampling）

默认路径用全局 RNG：`torch.manual_seed` 一变、batch 里多一条请求，同一条 prompt
的采样结果就可能不同。线上评测/回放/缓存都讨厌这个不确定性。

`generate(..., seed=N)` 改成**每行独立 RNG**：

```python
gens = [Generator(device).manual_seed(N + i) for i in range(batch)]
probs = softmax(logits / temp)
nxt = cat([multinomial(probs[i:i+1], 1, generator=gens[i]) for i in range(batch)])
```

- 行 `i` 只用自己的随机流 → 同一请求放进任何 batch、任何位置，输出逐位一致；
- 不读全局 RNG → 调用前后 `manual_seed` 怎么变都不影响；
- `seed=None` 走原路径（全局 RNG），与老行为逐位一致；
- 贪心（`temperature=0`）本来就确定，seed 只影响采样路径。

服务端对应 OpenAI 的 `seed` 字段；`stream` 与 `generate` 用同一 seed 时输出一致。
真正的"批不变"还要求算子层归约顺序与 batch 无关（DeepInfer 改的是 kernel），
本仓库在 PyTorch 上只保证**采样层**的批不变；batch 大小变化可能带来末位浮点差异，
需要逐位复现时固定 batch 形状。

## 代价与坑

- 贪心 + 无惩罚 = 循环 Fuji：sample 评测必须固定 prompt 集看复读率数字，
  别只看自我介绍那一题（幸存者偏差，pitfalls 第 8 条）。
- `stream_tokens` 只 yield batch 第 0 条——batch 推理走 `generate`，别混。

## 一句话总结

地基一次，砖一块块砌，选砖用骰子+惩罚——生成质量一半在权重，一半在采样器。
