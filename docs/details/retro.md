# RETRO：开卷考试系统

> 一句话：训练时给模型"开卷"——每段文本配 BM25 捞回的邻居，v1 看均值（话题方向），v2 看原文（逐字可抄）。

## 痛点

见《记忆层》：128M 背不下。RETRO（DeepMind 原版）是另一条外挂路：
知识不背，考试时现查。完整版要双向编码器 + 全层交错 cross-attention，
太贵——我们分两版做。

## 直觉

v1 像考前看"知识点摘要"：知道这题大概讲什么，但细节还得自己编。
v2 像把参考书摊在桌上：原文字句都能抄。摘要便宜但弱，抄书贵但强。

## 原理

**v1 单点**（`retro_every=0`）：`final_norm` 之后、`lm_head` 之前，
一次 cross-attention：`h + w_o(attend(h, mem))`，mem = 检索文本的
embedding 均值。`w_o` 零初始化，恒等起点。

**V2 交错**（`--retro-every N --retro-len L`）：每 N 层一个同构融合块，
吃的 mem 是 token 级 chunk（`build_batch_chunk_ids` 产 id + mask，
模型侧 frozen embedding 查表编码，no_grad 不吃梯度）。
attention 天然能做"指针式复制"，这是 v1 到 v2 的质变。

**检索工程**（`retrieval.py`）：BM25 字符级零依赖 + 倒排（token→文档表）+
只取 idf 最高的词做候选、单遍打分 + 堆取 top-K。33 万文档：全扫描 5.4s
→ 64 词 2.4s → 训练用 12 词 0.76s（top-1 与全量一致）。老索引无倒排时
`load` 就地重建。训练时检索与 GPU 计算重叠（后台单线程前瞻下个 micro，
0.64s/micro 被掩盖）。

## 我们的实现

- 文件：`src/llm/local/retro.py`、`retrieval.py`、`scripts/build_retrieval.py`
  （`--sft` 拼 SFT 三元组建库）
- 开关：`--enable-retro --retro-db <pkl> --retro-k 2`（+ V2 的 every/len）
- 全 mask 行退化为零增量（防 NaN）；评测/生成不传 mem，量的永远是 backbone；
- **开卷语义警告**：chunk 不做因果 mask、同源未过滤——自检索的 loss 虚低
  只能由专用评测度量，勿与 backbone val 比大小；
- 用法推荐 B（mid-training 增广 + 尾段关掉冷却），A（真 RAG 上线）等
  检索升级 + mem 消融诊断（真 mem vs 随机 mem 有 loss 差）后再做。

## 从想法到代码

v1 是均值拼前缀，v2 是原文按 token 融；检索是"倒排 + 掐头去尾"：

```python
# v1: mem = mean(embed(doc))            # (b, K, d)，话题方向
# v2: ids, mask = build_batch_chunk_ids(...)  # (b, K, L)，frozen embed 查表
h = h + w_o(attend(h, mem))             # w_o 零初始化，恒等起点
hits = query(text, k=2, max_terms=12)   # 只取 idf 最高的词，0.76s/查
```

## 代价与坑

- v1 的增益期望≈0（均值向量只能给方向），它的任务是证明管线不炸——
  别拿 v1 的 val 说事。
- 字符级 BM25 天花板低（同义不同字无解）；要上 A，先换 embedding 召回。

## 一句话总结

v1 证明管线，v2 实现抄书，检索是瓶颈——开卷系统的三级跳。
