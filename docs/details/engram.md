# Engram 条件记忆：n-gram 哈希查表

> 一句话：MoE 用稀疏**计算**处理动态逻辑，Engram 用稀疏**查找**取回静态知识——
> 把"过去 n-1 个 token + 当前 token"哈希进一张素数大小的表，O(1) 取回记忆，
> 再用当前 hidden 门控后残差加回主干。只插少数早期层，默认关。

## 痛点

128M 的脑容量背不下多少"死知识"（实体、模板、公式、成语搭配）。这类信息是**静态**的，
让注意力与 MoE 每步重新组装很浪费——早期层大量算力花在"把常见模式拼出来"，
留不出手处理真正需要推理的长程依赖。

`memory.py` 已经用 Product-Key 做了一层"可查的知识表"，但 PKM 的键是**学出来的**隐向量，
查表要先算 hidden 再 top-k 比较；而 Engram 指出：**键可以就是 token 序列本身**——
用局部 n-gram 做确定性哈希，O(1) 直接寻址，不需要学、也不需要比较。

## 直觉

看到"城市是"就条件反射想到"北京/上海"——这是**语言模型里的条件反射**。
n-gram 记忆就是把这种条件反射外挂成一张查表：给定局部上下文，直接翻到那一页。
因为键由 id 唯一决定（确定性寻址），整张表可以放 CPU/磁盘、按需预取，就像字典
不必整本背下来。

## 原理

```
对每个阶 n∈{2,3}、每个哈希头 k：
    gram = (x_{t-n+1}, …, x_t)      # 局部 n-gram（句首左补 boundary）
    idx  = hash(gram, seed_{n,k}) % table_size   # multiplicative-XOR，确定性
    e_{n,k} = Table_{n,k}[idx]                    # 查一行
e_t = concat_{n,k} e_{n,k}          # (d_mem)
V   = W_V e_t ；  K = W_K e_t
α   = σ( RMSNorm(h)·RMSNorm(K) / √d )          # 上下文门控
Y   = α ⊙ ( SiLU(Conv(RMSNorm(V))) + V )       # 短因果卷积 + 残差
h  ← h + Y                                       # 注意力之前残差并联
```

三个要点：

1. **多头、多阶哈希**：同一 n-gram 在不同头/阶落到不同表（seed 不同、表大小取素数），
   把碰撞压力摊开；`e_t` 由各头拼接，维数 `d_mem` 按 `阶数×头数` 均分。
2. **上下文门控**：记忆是"死的"，α 让它只在和当前上下文吻合时才放行——避免每次都把
   静态知识硬塞进去。
3. **早期注入**：论文消融显示只插少数早期层最优（约 layer 2），"替主干提前完成特征
   组装"，等效加深网络；插满所有层反而变差（与 MoE 抢活）。

## 我们的实现

- 文件：`engram.py`（`EngramMemory` + `hash_ngrams`/`is_prime`）、`block.py`
  （`HybridBlock.engram`，注意力之前残差并联）、`model.py`（`build_block` 按
  `engram_every` 选层 + `_engram_block_ids` 解码补上下文）；
- 配置：`engram_every`(默认 0=关)、`engram_orders`("2,3")、`engram_heads`(2)、
  `engram_d_mem`(64)、`engram_table_size`(8191，须素数)、`engram_conv`(4)；
- **恒等起点**：`W_V` 与卷积权重零初始化（conv 无 bias）→ 开局 `Y=0`，`h ← h + 0`，
  开启开关不破坏已训权重（单测锁死"初始化前后输出一致"）；
- **默认关**（`engram_every=0`）时模型里根本不存在该模块，`state_dict` 与旧
  checkpoint 逐位兼容（与 memory/retro/vision 同风格）；
- **解码一致性**：`_decode_step`/`_decode_multi` 手拼了 attn+moe 路径，Engram 已同步
  接入；每步只需窗口内 `lookback` 个历史 id（见下），`prefill + 逐 token 解码 == 全前向`
  由单测锁定；
- **容量**：`count_params` 把巨大的哈希表计入总参数，但"激活参数"只算投影/卷积/归一 +
  每 token 查回的 `d_mem` 行（与 MoE 的激活口径一致）。

## 从想法到代码

```
# 1) 取末 tv 位多头 n-gram 记忆（tv = t + extra，extra 为卷积感受野）
gram = _grams_last(ids, n, tv)               # (b,tv,n)，gram[p,i]=ctx[p+i]
idx  = hash_ngrams(gram, table_size, seed)   # 确定性 multiplicative-XOR
e    = cat([table[(o,k)](idx) for o,k])      # (b,tv,d_mem)
# 2) 短因果卷积作用在 id 决定的 V=W_V·e 上（无跨步状态；左补 extra 后 conv）
v = w_v(e); z = conv(pad(norm_v(v))); v = silu(z) + v
# 3) 上下文门控（唯一依赖 hidden 的部分，逐位置）
alpha = sigmoid( (norm_q(h) · norm_k(w_k(e[:,-t:]))) / sqrt(d) )
return h + alpha * v[:, -t:]
```

## 代价与坑

- **卷积若有状态，解码必翻车**（本项目第 7 条教训的同类坑）。论文的卷积作用在门控后的
  `Ṽ=α⊙V` 上，而 α 依赖各位置 hidden——逐 token 解码时没有历史 hidden，卷积感受野会
  退化。**我们的取舍**：把卷积放在门控**之前**、作用在仅由 id 决定的 `V=W_V e` 上，
  于是感受野内全部只依赖 id，解码每步只需 `lookback = extra + max_order - 1` 个历史 id，
  **无跨步状态**。这与论文不等价（少了一层"对门控信号做卷积"），但保住了 n-gram 寻址、
  多头、门控、短卷积四个要素，且数学上严格因果、可复现。
- **表大小必须素数**：`engram_table_size` 非素数直接 assert 失败（抗碰撞）。
  `d_mem` 必须被 `阶数×头数` 整除；`engram_orders` 各项须 ≥2，否则构建报错。
- **边界哨兵**：句首用词表外常量 `vocab_size` 左补（只参与哈希、不查真实 embedding），
  保证首 token 的 n-gram 也有确定键，不与任何真实 token 混淆。
- **接入手拼的解码路径**：`model._decode_step`/`_decode_multi` 不走 `block.forward`，
  Engram 必须在两处同步（同 memory 第 7 条坑）。零初始化阶段恒等看不出来，权重量
  非零后才分叉——"解码 == 全前向"单测就是守这条。
- **有视觉前缀时**：视觉位置用哨兵 id 占位，文本 token 仍与 h 的文本段对齐，
  首个文本 token 的 n-gram 以哨兵打底。

## 参数速查（`config.py`）

| 键 | 默认 | 说明 |
|---|---|---|
| `engram_every` | 0 | 每 N 层插一个 Engram；0=关（无参数） |
| `engram_orders` | "2,3" | N-gram 阶（逗号分隔，须 ≥2） |
| `engram_heads` | 2 | 每阶哈希头数 |
| `engram_d_mem` | 64 | 拼接后记忆维（须被 阶数×头数 整除） |
| `engram_table_size` | 8191 | 每头哈希表大小（须素数） |
| `engram_conv` | 4 | 短因果卷积核；0=不做卷积 |
