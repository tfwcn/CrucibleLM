# CSA2 三模式（Full / Reindex / Reuse）：把"复用"拆成两档

> 一句话：`csa_reuse_every` 只能做"每 N 层一个 Full、其余全按 Reuse"的**均匀两态**；
> V4.1 的 CSA2 用的是**静态三模式**——中间那档 **Reindex** 复用前层的 main KV
> 却**自己重选 Top-K**，既不付"造 KV"的钱，也不完全交出"选择权"。

## 痛点

两级太少。Reuse 层把两样东西一起借走了：**main KV latent**（贵的、重复的）和
**Top-K 选择**（便宜的、逐层各异的）。后者其实没必要借——同一份 latent，
下一层看到的是**不同的 hidden**，它想看的重点本来就不一样。全借走等于让三层
共用一套"重点"，白白损失表达力；而让每层都当 Full 又会把 KV 缓存重新乘回层数。

## 直觉

一排读者共用一本书（main KV）：

- **Full**：认真做笔记（写缓存），并自己划重点（选 Top-K）；
- **Reindex**：不重做笔记，翻前一个人的笔记，但**按自己的问题重新划重点**；
- **Reuse**：连重点也照着前一个人的看。

Reindex 的关键在于"重划重点"很便宜：它只需要自己的一份 indexer **Q**（`w_iq`），
indexer **K** 可以直接从别人已经算好的 main KV latent 投影（`w_ik`），
不必重造 KV 下投影（`w_dkv`/`w_kr`）。

## 原理（DeepSeek-V4.1 CSA2 的静态分配）

V4.1 把全注意力层**静态**分成三档（不是按数据动态路由），逐层轮转：

| 模式 | main KV | indexer K（`w_ik`） | indexer Q（`w_iq`） | Top-K 选择 | 写 KV 缓存 |
|---|---|---|---|---|---|
| **Full** | 自算 | 自算 | 自算 | 自选（全候选造池） | ✅ |
| **Reindex** | 借生产者 | 借生产者 | **自算** | **自选（限前层候选池内）** | ❌ |
| **Reuse** | 借生产者 | — | — | 借（直接用最新 Top-K） | ❌ |

"生产者"= 最近一个 **Full 或 Reindex** 层（Reindex 会更新"最新选择"，所以它也是生产者）。

我们的分配方式与 V4.1 一致：`csa_modes` 是**逐全注意力层**的模式串，按全注意力层序号循环：

```
csa_modes = "full,reindex,reuse"   →  全注意力层序号 0=Full, 1=Reindex, 2=Reuse, 3=Full, …
```

**没有 indexer 时 Reindex 会退化成 Reuse**（此时选择只由静态 pattern 决定、与 query 无关、
逐层相同，重选 = 照抄）。这不是近似，是数学等价——所以三模式**必须建立在
分层稀疏 indexer（`sparse_indexer`）之上**，这也是 `AGENTS.md` 把顺序改成
**5（indexer）→ 3（三模式）→ 4（压缩比/KV 源层）** 的原因。

## 我们的实现

### 1. 模式分配：`config.csa_modes` + `csa_mode_of`

- `csa_modes`（默认 `""`）逗号分隔，别名 `f`/`ri`/`re`；**空 = 旧行为**——由
  `csa_reuse_every` 推导（`2 → ("full","reuse")`，长度 = N）；
- `csa_mode_list()` 给出循环表，`csa_mode_of(layer_idx)` 给出某层模式
  （**非全注意力层恒为 `"full"`**，线性/SWA 层不受影响）；
- 非法模式直接 `ValueError`（早失败好过静默按 Full 跑）；
- `csa_reuse = (mode != "full")`：借用者解码时不写自己的缓冲，`kv_cache_bytes`
  因此自动只计 Full/…（生产者的缓存被借用者复用，不新增）。

### 2. 权重共享：`_link_csa_reuse` 按模式接管

借用层与最近生产者共享 `w_dkv`/`w_kr`（避免"永不更新的死参数"），
`sparse_indexer` 开启时 **Reindex 连 `w_ik` 也借**——只留自己的一份 `w_iq`，
完全对上 V4.1 的口径。每个借用层记下 `csa_src = 生产者层序号`，供选择通道读取；
首个全注意力层（或借用者找不到前置生产者）**退化为 Full**，保证总有共享源。

### 3. 层间选择通道：`CsaChannel`（按**层号**键控）

Reuse 要拿到"最新 Top-K"，Reindex 要拿到"前层候选池"，而这两样都产生于**同一次前向**
的**更早的层**——不能塞进 `past`（那是跨步状态）。所以引入一个前向内的临时通道：

- **键是"层序号"**，不是"执行顺序号"：每层只写自己的槽，借用者按 `csa_src` 读。
  这是必须的——梯度检查点反向重算时，若通道按执行序覆盖，重算的 Reindex 层会读到
  **自己被覆盖后**的值（池 ≠ 原池），前向数学与首次不一致却**不报错**（静默错梯度）。
  见 pitfalls 第 30 条。
- 预填充：`items[层号][块号]`（各层 `sparse_chunk` 一致，块号自然对齐）；
  解码：单步单块，块号恒为 0，生产者每步重写自己的槽；
- 只有 `sparse_indexer` 开启**且真有借用层**（`_csa_active`）时才创建，否则 `None`、
  零开销。

### 4. indexer 蒸馏损失（否则三模式是空转）

indexer 的产出是**掩码**（不可微），所以 `w_iq`/`w_ik` 本身**拿不到任何梯度**——
不加训练信号就是"假装在学"。按 DSA 的做法补一个**蒸馏损失**：让 indexer 的打分分布
去拟合主注意力在同一批静态候选上的分布（**目标 detach**，只训 indexer 不反向拉主注意力）：

```
target = softmax(主注意力分数[:, :, static 列]).mean(头)   # detach
kl     = KL(target ‖ softmax(indexer 分数))               # 每查询平均（nats）
aux   += indexer_loss_coef * kl                            # 进各层 aux 汇总
```

系数 `indexer_loss_coef`（默认 0.01）**只缩放 indexer 自己的学习步长**（目标已 detach，
不会污染主损失的口径）；设为 0 = 不训 indexer（口径可关）。层内多块取均值、层间由模型
累加进 `aux_loss`，另以 `indexer_loss`（层间均值、未缩放）单独打日志便于读。

## 从想法到代码

```python
# 1) 分配（config.build_block）：显式模式串优先，空则回退旧的均匀两态
csa_mode = config.csa_mode_of(layer_idx)          # 非全注意力层 → "full"
blk = HybridBlock(..., csa_mode=csa_mode)         # attn.csa_mode / attn.csa_reuse = mode != "full"

# 2) 共享（TinyLLM._link_csa_reuse）：借用者共享下投影 + indexer K，并记生产者
layer.attn.w_dkv = last.attn.w_dkv
layer.attn.w_kr  = last.attn.w_kr
if mode == "reindex" and indexer_on: layer.attn.w_ik = last.attn.w_ik
layer.attn.csa_src = last.attn.csa_index          # 读通道用

# 3) 通道（SparseMLAModule._run_blocks / _decode）：生产者写、借用者读
shared = ch.get(self.csa_src, chunk) if ch else None
o, sel, kl = self._sparse_scores(..., shared=shared)   # reuse: sel = shared；reindex: 限池自选
if sel and self.csa_mode in ("full", "reindex"): ch.put(self.csa_index, chunk, sel)
```

## 参数速查（`config.py`）

| 键 | 默认 | 说明 |
|---|---|---|
| `csa_modes` | `""` | 逐全注意力层模式串（`full`/`reindex`/`reuse`，别名 `f`/`ri`/`re`）；空 = 由 `csa_reuse_every` 推导旧行为。**改语义但不改 `state_dict` 键**，续训/迁移自动沿用 checkpoint |
| `csa_reuse_every` | 2 | 旧的均匀两态（未设 `csa_modes` 时生效）：每 N 个全注意力层留 1 个 Full |
| `indexer_loss_coef` | 0.01 | indexer 蒸馏损失系数；只影响 indexer 自身学习步长，0 = 不训 |

## 代价与口径

- **不改键**：模式分配（含 Reindex 的 `w_ik` 共享）**不新增/不删除 `state_dict` 键**，
  所以「旧 Full/Reuse 权重 ↔ 新三模式分配」可以直接互载而不报错——**这是坑**（pitfalls 23），
  装载脚本必须 `sync_arch_from_checkpoint` 对齐来源。
- **省的是选择，不是算力**：Reindex 复用 main KV，省的是"重造 KV 下投影"和"全候选重选"；
  indexer 的打分与注意力本身仍按候选集算。真实算力收益要等 block-sparse kernel
  （同 pitfalls 29）。
- **KV 缓存**：只有 Full 层计缓存；Reindex/Reuse 共用生产者的缓冲，等于把"每 N 层 1 个 Full"
  推广成"任意非均匀分配"（这正是 #4 压缩比/KV 源层的分配基础）。
- **收益尚未实测**：三模式的效果依赖 indexer 训得动，A/B 需在 indexer 有效后按 500 步口径做；
  在 indexer 未收敛前，Reindex 与 Reuse 的差距会被"选择质量差"掩盖。

## 单测锁定（`tests/test_csa_modes.py`）

- 模式分配：`csa_mode_list` 旧行为推导、别名归一、非法值报错、逐层 `csa_mode_of`；
- **不改键**：`csa_reuse_every=2` 下 `""`/`full,reindex`/`full,reuse` 三者键集合相同；
- 共享关系：借用层与生产者的 `w_dkv`/`w_kr`/`w_ik` **同对象**，`w_iq` 独立，`csa_src` 正确；
- 首层退化：`csa_modes="reuse,reuse"` 时层 0 仍退化为 Full；
- 通道语义：Reuse 原样取共享对象（且不算 indexer）、Reindex 借池自选并收窄到池交集、
  通道按层号键控互不覆盖；
- 掩码语义：非入选静态列改值**不影响输出**、入选列改值**必须影响输出**（非空转）；
- Reindex ≠ Reuse（indexer 开）、Reindex **逐位等于** Reuse（indexer 关）；
- 因果（改未来 token 前部不变）+ 梯度 Finite；
- `prefill + 增量解码 == 全前向`、**梯度检查点开关不改梯度**（通道键控的回归锁）；
- `kv_cache_bytes` 只计生产者；来源对齐（`sync_arch_from_checkpoint`）+ CLI `--csa-modes` 三态。
