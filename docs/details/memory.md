# Product-Key 记忆层：CPU 上放得下的知识表

> 一句话：每 N 层插一个"查表"模块，4096 个槽只激活 8 个，表放 CPU，搬运量恒定——开局恒等，不破坏已训权重。

## 痛点

128M 的脑容量背不下多少事实，加参数又贵。外挂知识表是第三条路：
知识不进权重，进一张可查的表——Meta Memory Layers / DeepSeek Engram 的思想。

## 直觉

字典查字：先翻部首目录（快），再翻到那一页（准）。Product-Key 把 M 个槽
拆成 √M×√M 两个子码本：查询切两半，各找各的 top-k，再笛卡尔组合重排——
查找 O(√M)，4096 个槽只比 64 次，天然适合放 CPU（搬运量恒为 k 行，
与表大小无关）。

## 原理

```
q 切两半 → q1 在 keys1 里 top-k，q2 在 keys2 里 top-k
候选槽 = i1×side + i2（k² 个），得分相加再取 top-k
delta = Σ softmax(score)·values[slot]；输出 = h + delta（残差式）
```

- **value 全零初始化 = 精确恒等**：开局输出就是输入，开开关不破坏已训权重
  （单测锁死"初始化前后输出一致"）；
- **B 方案 key 初始化**：backbone 冻结跑校准集，hidden 劈半各做 k-means，
  簇心当子码本（`scripts/init_memory.py`，标准 PKM 做法），value 保持零；
- `n_slots` 须为完全平方数（√M×√M），否则启动直接报错（别传 3000 这种）。

## 我们的实现

- 文件：`src/llm/local/memory.py`（`ProductKeyMemory` + `init_memory_from_activations`）
- 开关：`--enable-memory --memory-every 4 --memory-slots 4096 --memory-topk 8`
  （12 层里 0/4/8 层有记忆，和 Hybrid 钉子户同层，吃最准的 hidden）
- 流程：`init_memory.py --src 旧权重 --out 新目录` 产校准权重 →
  `--sft-init 新目录/model.pt --enable-memory` 开训；
  `vocab_size` 保持预设容量 8192（SFT 扩词复用空行不改形状，缩表会导致
  embedding 静默覆写不上——实测抓到过，见 ARCHITECTURE）；
- 表参数随模型设备走（构造时先放 CPU，`model.to(device)` 会整体搬运）；
  稀疏收益在于每步只 gather 命中的 k 行；
- `_decode_step` 必须同步走记忆层（value 非零后漏掉即分叉 0.69，pitfalls 第 7 条）。

## 从想法到代码

查字典即"劈半查目录 + 笛卡尔对页码"，恒等即 value 全零：

```python
i1, i2 = topk(q1 @ K1.T), topk(q2 @ K2.T)   # 两半各查各的
slot = i1 * side + i2                        # 笛卡尔组合，k² 候选再取 top-k
delta = softmax(score) @ values[slot]        # 只搬 k 行
return h + delta                              # values 全零时恒等
# B 初始化：init_memory.py 跑校准集 → hidden 劈半 k-means → 簇心当 keys
```

## 代价与坑

- 记忆是"外挂"，不是"学会"：keys 定了之后靠训练微调，学新事实不如 RAG 快，
  但比 RAG 稳（不依赖检索质量）。
- 不做 logit 蒸馏——那是另一套配方，别混。

## 一句话总结

查字典式记忆：目录放 CPU、开局恒等、B 方案给目录——知识外挂的最小可用形态。
