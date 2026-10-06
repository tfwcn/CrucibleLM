# Hybrid 交替：全注意力与线性注意力的排班表

> 一句话：1 层全注意力（记准）配 3 层线性注意力（算快），质量和效率都要。

## 痛点

全 attention 准但贵（平方复杂度），线性 attention 便宜但糊（定长状态）。
全用前者，200K 跑不动；全用后者，长依赖质量掉。Qwen3-Next 的答案是：
不掺在一起算，而是**分层排班**——有些层记准，有些层跑量。

## 直觉

像接力赛：3 个跑量选手（线性层）飞快推进，每隔一段换 1 个记忆选手
（全注意力层）把前面的细节对一遍、钉死。记忆选手不用多，钉子户太多
队伍就慢了。

## 原理

没有新公式，就是排班：`full_attn_every=4` 表示每 4 层中有 1 层是全注意力
（第 0/4/8 层），剩下 9 层全是线性。12 层一共 3 个"钉子户"。

为什么不是 1:1？实验（Qwen3-Next 论文 + 社区复现）表明质量拐点在
"每 3~4 个线性配 1 个全"附近：再密，速度收益被吃光；再疏，
长依赖（比如 8K 外的指代）开始掉点。128M 上我们直接沿用 1:3，
没烧钱重扫——这是抄作业，不是调参。

## 我们的实现

- 文件：`src/llm/local/block.py`（`HybridBlock`，`full_attn` 二选一）、
  `src/llm/local/config.py`（`is_full_attn_layer()`）
- 全注意力层 = `SparseMLAModule`（MLA + 稀疏，短序列自动退化 dense）；
  线性层 = `GatedDeltaLite`
- 记忆层和 RETRO 交错块可以和"钉子户"同层（sft5 配 `memory_every=4` 落在 0/4/8）：
  记忆需要最准的 hidden，钉子户给的就是最准的（注意 `retro_every` 默认 0
  是单点融合，只在显式设为 4 时才交错，别抄错）

## 从想法到代码

排班就是一行取模，构造时决定，之后全模型通用（迁移工具也调它，防漂移）：

```python
def is_full_attn_layer(i): return i % full_attn_every == 0  # 0/4/8 是钉子户
def build_block(cfg, i):  # 模型/迁移/训练共用的唯一入口
    attn = SparseMLAModule(...) if is_full_attn_layer(i) else GatedDeltaLite(...)
```

## 代价与坑

- 排班是写死的（`layer_idx % 4`），改比例要重训——架构级决策，试错成本高，
  所以迁移工具（`migrate_model.py`）只动层数/宽度，不动排班。
- 推理加速（compile）时两类层的缓存格式不同（latent 三元组 vs 状态三元组），
  `_decode_step` 里各走各的，别混。

## 一句话总结

排班表就是架构：3 个跑量的配 1 个记准的，比例是抄来的，位置决定了记忆放哪。
