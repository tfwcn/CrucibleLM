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

## 原理（V4.1 §2.4.2 口径）

```
对每个阶 n∈{2,3,4}、每个哈希头 k（共 8 头）：
    gram = (x_{t-n+1}, …, x_t)      # 局部 n-gram（句首左补 boundary）
    idx  = hash(gram, seed_{n,k}) % table_size   # multiplicative-XOR，确定性
    e_{n,k} = Table_{n,k}[idx]                    # 查一行（**FP8**）
e_t = concat_{n,k} e_{n,k}          # (阶数 × d_mem)
K   = W_K e_t ；  V = W_V e_t        # 两个投影都走 FP8
α   = σ( RMSNorm(h)·RMSNorm(K) / √d )          # 上下文门控
Y   = α ⊙ V                                       # **无短卷积**
h  ← h + Y                                        # 注意力之前残差并联
```

V4.1 相对原版 Engram 的两处修改，本项目都对齐：

1. **去掉短因果卷积**。原版用 kernel=4 的深度可分离因果卷积平滑 `V`，V4.1 认为
   "收益不值得推理栈多一份复杂度"而删掉。副作用是**解码变简单**：`lookback = max_order-1`
   （阶 4 → 需要 3 个历史 id），没有跨步卷积状态，`_decode_step` 不用存尾部。
2. **FP8**。表与 K/V 投影都是 FP8。训练用 fake-quant + STE（QAT，模型学的是
   "在 FP8 误差下仍然好用"的表示）；推理可 `pack()` 成**真 FP8 存储**
   （表约 1.06 B/通道，对照 bf16 的 2 B/通道）。表更新交给 **Sinkhorn**
   （`--sinkhorn`，与 token embedding / 预测头同一档，省掉 Adam 的一阶/二阶状态）。

三个要点：

1. **多头、多阶哈希**：同一 n-gram 在不同头/阶落到不同表（seed 不同、表大小取素数），
   把碰撞压力摊开；`e_t` 由各头拼接，每阶贡献 `d_mem` 维（论文每阶 2048、8 头 × 256）。
2. **上下文门控**：记忆是"死的"，α 让它只在和当前上下文吻合时才放行——避免每次都把
   静态知识硬塞进去。
3. **早期注入**：论文消融显示只插少数早期层最优（第 1、14 层），"替主干提前完成特征
   组装"，等效加深网络；插满所有层反而变差（与 MoE 抢活）。

## 我们的实现

- 文件：`engram.py`（`EngramMemory` + `EngramTable` + `hash_ngrams`/`is_prime`）、
  `block.py`（`HybridBlock.engram`，注意力之前残差并联）、`model.py`
  （`build_block` 按 `engram_layers`/`engram_every` 选层 + `_engram_block_ids` 解码补上下文）、
  `optim.py`（Sinkhorn 按参数名 `...engram.tables.K.weight` 路由）；
- 配置：`engram_every`(0=关)、`engram_layers`(""=按 every 均匀插；显式如 "1,13")、
  `engram_orders`("2,3,4")、`engram_heads`(8)、`engram_d_mem`(256，**每阶**维度)、
  `engram_table_size`(8191，须素数)、`engram_fp8`(True)；
- **恒等起点**：`W_V` 零初始化 → 开局 `Y=0`，`h ← h + 0`，开启开关不破坏已训权重
  （单测锁死"开 Engram 前后 logits 相同"；FP8 下 0 仍是 0）；
- **默认关**（`engram_every=0` 且 `engram_layers` 空）时模型里根本不存在该模块，
  `state_dict` 与旧 checkpoint 逐位兼容（与 memory/retro/vision 同风格）；
- **解码一致性**：`_decode_step`/`_decode_multi` 手拼了 attn+moe 路径，Engram 已同步
  接入；每步只需 `lookback = max_order-1` 个历史 id，`prefill + 逐 token 解码 == 全前向`
  由单测锁定（fp8 开/关都测）；
- **容量**：`count_params` 把巨大的哈希表计入总参数，但"激活参数"只算投影/归一 +
  每 token 查回的 `d_out` 行（与 MoE 的激活口径一致）。

## 从想法到代码

```python
# 1) 取末 t 位多头 n-gram 记忆（无卷积 → 不需要 extra 位）
e = cat([table[(n,k)](hash_ngrams(grams_last(ids, n, t), P, seed[n,k]))
         for n in (2,3,4) for k in range(8)])      # (b,t,3*d_mem)
# 2) FP8 投影（QAT：权重 fake-quant，梯度直通）
v  = F.linear(e, fake_quant(w_v.weight, "fp8"))
kk = norm_k(F.linear(e, fake_quant(w_k.weight, "fp8")))
# 3) 上下文门控（唯一依赖 hidden 的部分，逐位置）
alpha = sigmoid((norm_q(h) * kk).sum(-1, keepdim=True) / sqrt(d))
return h + alpha * v
```

`EngramTable` 的两种状态：

```python
# 训练/未打包：主权重是 nn.Parameter，前向只量化**查出来的行**（按行量化与按表量化
# 逐位等价，但不会每步扫全表）
rows = F.embedding(idx, self.weight); return fake_quant(rows, "fp8")
# 推理已打包：从 uint8 码字 + E4M3 scale 取行再反量化（与上面逐位相同）
q, s = quantize_fp8(self.weight, 16); self.w8 = q.view(uint8); self.s8 = s.view(uint8)
```

## FP8 与 Sinkhorn 的边界

- **打包是快照**：`pack()` 之后训练会读到旧表，故 `pack()` 在 `train()` 下直接报错，
  已打包的 `forward` 在 `train()` 下也报错（提示先 `unpack()`）。推理流水线：
  `load → pack() → generate`；继续训练：`unpack()`。
- **主权重保留**：训练必须有 fp32/bf16 主权重（优化器要它），所以"打包省内存"只在
  推理/部署阶段成立；这也是 QAT 的标准形态（master weight + quantized copy）。
- **投影的 QAT 成本**：每步对 `w_k`/`w_v` 做一次 fake-quant（一次全权重 pass）。
  真实 FP8 推理会把权重 resident 成 fp8、省掉这次 cast；本项目保持"训练/推理同一套
  数值"，代价记在这里。
- **Sinkhorn**：表是 2D 大矩阵（`table_size × d_head`），正好是 Algorithm 1 的形状；
  按参数名自动进 `sinkhorn` 组，不需要额外配置。切换 `--sinkhorn` 会改变优化器分组，
  续训须删 `optim.pt`（既有约定，pitfall 44）。

## 实测记录（本机 128M 续训）

单测只保证"接得对、恒等起点、解码一致、FP8 数值自洽"，**不保证有收益**。

**设置**：真实 128M 中途权重续训；`seq_len=512`、`batch=8`（4096 token/步）、
`lr=1e-4`（warmup 20 + cosine）、`grad_ckpt`；基线 vs `engram_every=4`（层 0/4/8，
表 8191，阶 2/3，头 2），同一 seed、逐位相同的数据顺序。**评测固定前 20 批**
（`eval_batches=20`；注意口径——见 pitfalls 第 22 条，改 `eval_batches` 会平移曲线）。

**结果（600 步 ≈ 2.46M token）**：两条 `val_loss` 曲线几乎重合，差值全程在 ±0.001 内，
末步 4.3133 vs 4.3133——**统计上无效果**。第 1 步 `loss` 逐位相同，锁定"恒等起点在
真实训练里成立"。

**迭代 2（预算 4 倍，2500 步 ≈ 10.2M token）**：三臂：基线、Engram（表 8191）、
Engram（表 65521，验证"减少哈希碰撞能否让记忆行收敛"）。25 个评测点上两臂与基线的差
**全程在 ±0.002 内**，末步 4.3904 / 4.3905 / 4.3908——**仍无效果，且没有随时间放大的
趋势**。表放大 8 倍也没改变任何东西，否掉了"碰撞导致行不收敛"的假设：瓶颈在**单遍语料
没有任何 n-gram 会重现**。

**V4.1 口径的结构冒烟（12 层 / 120.8M / 层 1、7 两模块 / 阶 2,3,4 / 8 头 / 表 8191）**：

| 检查 | 结果 |
|---|---|
| 表参数 | 6.29M/模块（激活参数 44.4M，表不进激活） |
| 恒等起点 | 开 Engram 前后 `max|Δlogits| = 0` |
| 训练 | loss 10.90、梯度全 finite、表梯度非零（打乱 `W_V` 后） |
| Sinkhorn 路由 | `sinkhorn` 组 49 个参数张量，48 个 Engram 表全在其中 |
| 解码一致 | 单 token 解码 vs 全前向 `1.0e-5`（fp8，块主序 + 量化同开） |
| FP8 打包 | 48.0 MiB → 12.7 MiB（0.27x，主权重 fp32；对照 bf16 约 0.53x），查表**逐位一致** |

**判读**：结构上该有的都有、且不伤基线；**是否涨点仍取决于语料里 n-gram 是否重现**
（多 epoch / 高频模板 / 代码），这是后续实验方向。

## 代价与坑

- **全零行会让 FP8 scale 下溢**：`W_V` 零初始化 → 投影权重全零 → `amax=0` →
  "scale 下界 1e-8 向下舍入到 E4M3 就是 0" → `0/0 = NaN`。修复：下界取 E4M3 最小正值
  `2^-9`（pitfall 51）。这条对 KV 量化同样成立（全零 latent 行）。
- **删掉短卷积的兼容性**：旧 Engram checkpoint 含 `engram.conv.weight`，装载会大声报
  unexpected key（不是静默错）——Engram 默认关、无历史正式 run，重训即可。
  `engram_d_mem` 的语义也从"拼接总维"改成"每阶维"（论文口径），沿用旧值会让参数量差
  `阶数` 倍；`_ARCH_SWITCHES` 只登记了 `engram_fp8`（同键改数值），其余是结构性开关，
  续训须显式给同样的 `--engram-*`（否则缺键/多键会当场报错）。
- **表大小必须素数**：`engram_table_size` 非素数直接 assert 失败（抗碰撞）。
  `d_mem` 须被头数整除；`engram_orders` 各项须 ≥2，否则构建报错。
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
| `engram_layers` | "" | 显式层号（如 "1,13"）；非空时忽略 `engram_every` |
| `engram_orders` | "2,3,4" | N-gram 阶（逗号分隔，须 ≥2；V4.1 用 2/3/4） |
| `engram_heads` | 8 | 每阶哈希头数（V4.1 用 8） |
| `engram_d_mem` | 256 | **每阶**记忆维（须被头数整除；V4.1 每阶 2048） |
| `engram_table_size` | 8191 | 每头哈希表大小（须素数；V4.1 约 16M） |
| `engram_fp8` | True | 表 + K/V 投影 FP8（QAT；`--no-engram-fp8` 关） |
