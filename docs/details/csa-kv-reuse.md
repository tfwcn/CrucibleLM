# CSA2 跨层 KV 复用：相邻层共享一份 latent

> 一句话：全注意力层的 KV 缓存随层数线性膨胀，而相邻层的 KV 高度相似——
> 干脆让一部分全注意力层（Reuse）**不自己算 KV**，直接借最近的 Full 层的 latent，
> 只自算 Q 和输出投影。层数越深、全注意力层越多，省下的 KV 缓存越多。

## 痛点

MLA 已经把每层 KV 压到 `kv_lora_rank + qk_rope_dim`，但**层数**这一维还在乘。
12 层里 3 个全注意力层、200K 上下文时，KV 缓存仍是显存大头；层数翻倍就翻倍。
DeepSeek-V4.1 的 CSA2 指出：**跨层复用**是继"低秩压缩"之后的第二条压缩轴——
同一位置的信息，相邻层需要的 KV 其实很像，何必各存一份。

## 直觉

想象一排读者共用一本书：第一个人（Full 层）认真做了笔记（latent 缓存），
后面几个人（Reuse 层）不重做笔记，直接翻前一个人的本子——但**各自用自己的问题
（Q）去问这本笔记**，得出的答案仍各不相同。笔记（KV）只存一份，问题（Q）各问各的。

## 原理（DeepSeek-V4.1 CSA2，我们的诚实子集）

论文把全注意力层分三种模式轮转：

- **Full**：自己算 latent（`c_kv`）+ 共享 RoPE 键，写进缓存；
- **Reindex**：自己算 Q，但**重建** key 索引（跳过写缓存）；
- **Reuse**：连索引都复用最近 Full 层的，只算 Q 和输出。

我们实现 **Full / Reuse 两态**（Reindex 依赖分层稀疏索引器，属于定制 kernel 范畴，
暂不移植）：

```
full_attn_every=2 时全注意力层序号 ordinal=0,1,2,…
    ordinal % csa_reuse_every == 0  → Full（自算自存）
    否则                              → Reuse（借最近 Full 层的 latent）
```

Reuse 层**没有自己的 KV 缓存**：解码时它读到的是 Full 层刚写好的同一块缓冲，
只跳过写入、照常出 Q 并做掩码注意力。KV 缓存因此只由 Full 层承担。

## 我们的实现

- 配置：`csa_reuse_every`（默认 **2=开**；1=全 Full=关）；2=Full/Reuse 交替，KV 减半；
  3=每 3 个全注意力层留 1 个 Full；
- `build_block` 按 `layer_idx // full_attn_every % csa_reuse_every` 判模式，写进
  `attn.csa_reuse`（纯超参，不进 state_dict）；
- `TinyLLM._link_csa_reuse`：把每个 Reuse 层的 `w_dkv`/`w_kr` **替换成最近 Full 层的
  同一权重对象**（共享下投影）。这样既没有"永不更新的死参数"，也让"复用"名实相符；
  各层仍保留自己的上投影 `w_uk`/`w_uv`、Q 投影、输出投影（Reuse 只借 latent，不借上投影）；
- **prefill/训练**：Full 层返回缓存，Reuse 层经 `reuse=` 参数借它出 K/V；
- **解码**：Reuse 层的 `past` 本身就是共享缓冲（`_prefill` 里与 Full 层同对象），
  它只需 `write=False`（跳过 `index_copy_`）读缓冲——不必另传参数，`_decode_step`/
  `_decode_multi` 无需改动；
- `kv_cache_bytes` 只计非 Reuse 的全注意力层，如实反映收益。

## 从想法到代码

```python
# build_block：按全注意力层序号分配模式
ordinal = layer_idx // config.full_attn_every
csa_reuse = config.csa_reuse_every > 1 and (ordinal % config.csa_reuse_every != 0)

# TinyLLM._link_csa_reuse：Reuse 层共享最近 Full 层的 KV 下投影
layer.attn.w_dkv = last_full.attn.w_dkv    # 同一权重对象，无死参数
layer.attn.w_kr  = last_full.attn.w_kr
last_full = layer                           # 只有 Full 层更新共享源

# MLAModule.forward：Reuse 层两种入口
if past is not None:                        # 解码：只读共享缓冲，不写
    return self._decode_attend(h, past, write=not self.csa_reuse)
if reuse is not None:                       # prefill/训练：K/V 借共享 latent
    return self._forward_reuse(h, reuse, return_state)   # Q 自算，K/V 用 reuse
```

## 实测记录（本机 128M 续训）

与 Engram 同一套对照设置（真实 128M 中途权重续训，`seq_len=512`、`batch=8`
（4096 token/步）、`lr=1e-4`（warmup 20 + cosine）、同一 seed 与逐位相同的数据顺序），
基线 vs `csa_reuse_every=2`。两次实验**评测口径不同**（见 pitfalls 第 22 条），
绝对 loss 不可跨次比较：

- **迭代 1（600 步，前 20 批）**：末步 +0.021（4.3347 vs 4.3133），只看趋势；
- **迭代 3（2500 步 ≈ 10.2M token，全 holdout 38 批）**：与迭代 2 的基线同口径对照。

**迭代 3 结果**：差距**从 +0.037 收敛到 ~+0.011 后平台化**——step 100 +0.037 →
500 +0.024 → 900 +0.009 → 之后 1000~2500 稳定在 +0.009~+0.013，末步 +0.0126
（4.4030 vs 4.3904）。

**迭代 4/5（判定"容量损失 vs 适配滞后"）**：从各自 2500 步末态、同 seed 同数据继续——
迭代 4 用 lr 2e-5/cosine 2000 步，迭代 5 用更小的 lr 1e-5/cosine 800 步，两轮独立复现同一现象：

- 续训必经"重新加热"暂态（两臂 loss 从 4.39 一起被抬到 4.59 附近），**暂态中差距一度缩到
  0 甚至为负**（看上去像"再训就补上了"）；
- 随 lr 衰减、loss 回落，差距**重新出现并稳定**：i4 末步 +0.0052、i5 末步 +0.0047
  （两轮不同 lr 却几乎同一残差）；
- 把 i3/i4/i5 全部 28 个评测点按 `base_loss` 排序：**loss 越低 gap 越大**
  （相关系数 r = **−0.45**；最低 loss 的 5 个点全为正，最高 loss 的点多为负），
  外推到 i3 的收敛点 4.39 即 +0.0126。

**判读（修正）**：暂态里的"差距消失"是**假象**——基线收敛得更深、极小点更尖，"重新加热"
对它的扰动更大，于是暂时显得 CSA2 更好；一旦两者都回落，差距就回来了。**证据不支持
"再训就能补上"**：差距随收敛深度增长、且在两个独立 lr 下复现同一残差，与"Reuse 层没有
自己的 KV latent（容量真的少了一块）"一致——是**小而稳的固有容量差额**，不是单纯的适配
滞后。（原计划"平 lr 长跑"已做，结论不变。）

**折算**：本机 8192 上下文的 KV 从 8.75MB 降到 6.13MB（**-30%**，3 个全注意力层减到 2；
32768 ctx −32%，参数 −0.1%），换来收敛点 val_loss **+0.012 ≈ 相对 +0.27%**。
小模型短预算下这是个**可辩护的取舍**：长上下文省三成 KV、质量代价千分之三——
故已设为**默认开**（显式传 `1` 关闭）。**但不构成"免费午餐"**，别宣传成无损。

## 代价与坑

- **必须紧跟一个 Full 源**：Reuse 层借用"最近的 Full 层"，所以第一个全注意力层永远
  是 Full（`_link_csa_reuse` 里兜底：无前置 Full 就退化为 Full），否则无缓冲可借。
- **decoding 与 prefill 的取用方式不同**：prefill/训练要显式把 Full 层的 latent 传进来
  （那时 Reuse 层自己 `past=None`）；解码则因为缓存对象已被 `_prefill` 共享成同一块，
  只靠 `write` 开关即可。两处都要接，漏一处就会"训练对、解码错"（第 7 条坑的同类）。
- **reuse 只借 latent，不借上投影**：Reuse 层用自己的 `w_uk`/`w_uv` 展开 K/V。
  这与论文"连上层投影一起共享"不完全等价，但保留了每层表达力，且缓存对象的共享
  口径清晰（省的就是 latent 缓存）。已在文档写明，勿当成论文复刻。
- **与旧 checkpoint 的兼容**：**默认已开**（`csa_reuse_every=2`）。关（显式传 1）时无任何共享、
  `_csa_active=False`，与旧 checkpoint 逐位兼容。开启后参数量**变小**（去掉了重复的下投影）
  且**不改 state_dict 键**，因此 Full↔Reuse 的权重不能直接互载却也不报错——模式属于结构开关
  （同记忆层）。加载既有权重的脚本必须先 `sync_csa_reuse_from_checkpoint` 对齐来源
  （见 pitfalls 第 23 条）。
- **Reuse 层不再向后传播"自己的 KV 投影"**：共享的 `w_dkv` 只从 Full 层拿梯度，
  但 Reuse 层的 K/V 用共享 latent，梯度会经由共享 latent 回流，训练仍是联合的。

## 参数速查（`config.py`）

| 键 | 默认 | 说明 |
|---|---|---|
| `csa_reuse_every` | 2 | 每 N 个全注意力层留 1 个 Full；1=全 Full（关），2=KV 减半。**默认开**，续训/迁移自动沿用 checkpoint 架构 |
