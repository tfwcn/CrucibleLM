# 层内 SWA + 有界重放：每层自己的"近处记忆"

> 一句话：让**每一层**自带一条滑动窗口注意力（窗口 K/V 由本层 hidden 产生、定长环存储），
> 与"全局稀疏主 KV"放进**同一次 softmax**——局部看精确、远端看稀疏；
> 落盘时只存全局主 KV，SWA 环用**最近 `n_win` 个 token 重放**近似重建。

## 痛点

稀疏注意力（sink + 窗口 + 跨步）用**一套**模式同时管远端和近端，于是近端被"跨步"糊掉：
窗口 `sparse_window=4096` 意味着最近 4096 个 token **全量参与**——编码长 prompt 时这
一块既是计算大头，又挤占了远端列数预算。想压窗口，近处就看不准。

另一层痛点是**缓存**：只要近端靠主 KV 提供，缓存就必须逐 token 存到"窗口那么远"，
而且窗口一放大缓存跟着涨。想"近处精确、缓存又小"，只能把近处单独做成**定长**结构。

## 直觉

人读一页纸：眼睛盯着的那一行看得一字不差（**滑动窗口**），页首标题和几个加粗小标题
一眼扫过（**全局稀疏**）。两者不是两种阅读模式，而是**同时**进行的。

论文的说法是 `selected entries together with the layer-local SWA KV`——同一次注意力里，
既放稀疏选出来的全局条目，也放本层自己的窗口条目。窗口里最新的那些位置**只**出现在
SWA 分支（不重复从主 KV 取），所以两条分支不冲突、也不重复计权。

## 原理（DeepSeek-V4.1 的口径）

- **每层都同时有全局注意力与 SWA**（前两层甚至只有 SWA）。全局 Q 与 SWA KV **各层自带**，
  只有全局 KV / indexer K 会跨层复用（那是 CSA2 的事）。
- SWA KV 由**本层 hidden** 产生 → 它天然不可能跨层复用（这也是它和 CSA2 正交的原因）。
- 窗口定长 `n_win`，故 **SWA 存储与序列长度无关**：环里永远只有 `n_win` 个位置。
- **SWA Bounded Replay**：SWA 依赖跨层累积，精确重建 `L` 层的 SWA 需要重放 `L × n_win`
  个 token（层数倍）。V4.1 只重放**最近 `n_win` 个** token，并把 SWA **截断到重放段**：
  "对从位置 `s` 开始的重放，位置 `i` 的查询只看 `[max(s, i-W+1), i]` 的 SWA 键"。
  由此换来的近似正是他们把 SWA KV 从持久化缓存里删掉的依据。
- 持久化缓存只留**全局 KV**；SWA 环丢了就重放 `n_win` 个 token 重建。

## 我们的实现

### 1. 开关与参数：`swa_window`（默认 0=关）

```python
# config.py
swa_window: int = 0   # >0 时每层多一条 SWA 分支（新增 swa.w_k/swa.w_kr/swa.w_v + swa.k_norm）
```

- **默认关**：它改 `state_dict` 键（新增 `swa.*`），旧 ckpt 无法直接载入；
  且论文口径下窗口该取多少要靠 A/B（见"代价"里的显存账）。已登记 `_ARCH_SWITCHES`
  （fallback `0`），续训/迁移从 checkpoint 的 `hparams.json` 沿用，读不到保守按关。
- 只对**全注意力层**生效（线性注意力层没有 KV，谈不上窗口）。
- 开启后 `sparse_window`（全局分支的本地窗口）与"当前块列"**都不再用**：
  近端全部交给 SWA 分支，全局分支只出 sink/stride 远端条目（`win=0, n_chunk=0`）。
- CLI：`--swa-window N`（`scripts/train_local_llm.py`）。

### 2. 分支本体：`SwaBranch`（独立投影，与全局 K 同维以便拼同一次 softmax）

```python
# sparse_attn.py:SwaBranch
w_k  : d_model → n_heads*head_dim     # content key（每头）
w_kr : d_model → qk_rope_dim          # 解耦 rope key（全头共享）
w_v  : d_model → n_heads*head_dim
k_norm: RMSNorm(head_dim)             # 与 MLA 的 k_norm 同理
```

K = `[k_content ‖ rope(k_rope)]`，与全局主 KV 的 K 维数相同，所以**可以拼进同一个
`QK^T`**：`scores = [全局静态列 ‖ SWA 列]`，一次 softmax、一次 `P·V`。

### 3. 环：三个定长环，行号 = `位置 % 容量`

```
环内容：k_ring (b, cap, n_heads, head_dim)
        kr_ring(b, cap, qk_rope_dim)      ← rope 键**存原料**，读时按绝对位置现算
        v_ring (b, cap, n_heads, head_dim)
容量  ：window + max(0, t-1)              # t = 一次解码的 token 数（预填充给 1）
行号  ：位置 % cap                        # 扩容后旧槽位仍按新模命中
```

- `rope` 存原料、读时旋转：环里的位置是**环形**的，只有绝对位置能决定旋转角
  （局部序号会错位）。旋转角下标 `pos.clamp(max=rope_table_len-1)` 防超表。
- 预填充只装**最后 `window` 个位置**（`_pack_swa(..., pos_end=L, cap=window)`）——
  若按 `L` 建环，SWA 缓存就退化成 O(L)，"存储有界"当场失效（pitfall 34）。
- 多 token 一步解码（投机验证）需要 `window+t-1` 行，先 `_grow_swa_rings` 按位置
  重映射扩容再写；把环搬到持久化时不需要它变长。

### 4. 读：窗口列 + 块内因果

```python
# _swa_read：查询 [start, end) 需要的并集 = [start - n_win + 1, end)
lo  = max(0, floor, start - n_win + 1)     # floor = 有界重放的起点 s
pos = arange(lo, end)
k, kr, v = ring[:, pos % cap], ...
```

- 带宽掩码 `SwaBranch.band_mask(q_pos, kv_pos)`：`kv_pos ∈ (q-W, q]` 放行，其余 `-inf`
  ——因果 + 带状一次搞定，块内所有查询共用一张掩码。
- `floor` 只服务于有界重放：重放段之前的位置**根本没进环**，但环里的零行若不被显式裁掉，
  软注意力会照付权重（零行不是"不存在"）。

### 5. 三条路径必须一致（开启即恒走分块路径）

| 路径 | SWA 列怎么来 |
|---|---|
| 预填充（`_run_blocks`） | 本层全序列投影 `swa.project(h)`，按块切 `[start-W+1, end)` |
| 增量解码（`_decode`） | 环里 `[pos-W+1, pos)`（含当步），驱动环的写入 |
| 有界重放（`replay_swa_from_ids`） | 只读解码 + 空环 + `floor=s`，重建环、不动全局 KV |

- `swa_window>0` 时**不再**对短序列走 dense 兜底（`sparse_threshold` 失效）：
  dense 走的是 `MLAModule` 那条完全独立的实现，没有"层内 SWA"这个概念，走了就分叉
  （pitfall 35）。
- 全局 pattern 的 stride 集合依赖起点，所以取 `blk_start = (start//sparse_chunk)*sparse_chunk`：
  解码与预填充必须按同一块起点取，否则近端 token 会同时出现在全局列和 SWA 列里**计权两次**
  （pitfall 35 后半）。

### 6. 有界重放：`TinyLLM.replay_swa_from_ids`

```python
start = max(0, total - n_win)                 # 只重放最近 n_win 个 token
replay = [清空 SWA 环 + pos 回退到 start 的 pasts]   # 全局 KV 用权威原值
for i in range(total - start):               # 逐 token 只读解码
    _decode_step(tail[:, i:i+1], replay, write_cache=False, swa_floor=start)
pasts = [权威 (buf_c, buf_kr, pos) + 重放出的环]      # 全局 KV 一个字节都没动
```

- `write_cache=False`：全局主 KV 的行早已由权威前向写好，重放**只重建环**
  （论文：`reusing the cached global KV without recomputation or overwriting`）。
- `swa_floor=start`：即使环里还有更早的行，也用不到——语义上等价于"截断到重放段"。
- **线性层的递推状态：清零重建，不沿用末态**。末态已经含过重放段（乃至整段）的贡献，
  重放时再喂一遍就是**重复计入**；更糟的是 `start=0` 的整段重放里末态还含"未来"信息。
  实测（base 规模 / 64 token 会话 / `n_win=128`）下一步 hidden 误差：沿用末态 **2.99e-1**、
  清零 **4.2e-6**、无环冷启动 **3.47**——"沿用末态"几乎和不重建一样糟（pitfall 39）。
  清零 = 论文的"截断到重放段"：从段首以零状态重新累积。
- **精确性**：`n_win ≥ 已缓长度` 时是（数值上）逐位精确——连 hybrid 也成立，因为线性状态
  本来就从零重建（单测锁定 1e-5）。`n_win < 已缓长度` 时是**有意的截断**：丢掉段首之前的
  长程递推记忆，与论文"重放前缀状态是近似的、与位置相关"的口径一致。
- 截断方向是"更省、略差"，且明确优于"不重建"（单测锁"重放误差 < 冷启动误差"这条弱契约，
  随机权重下层间无收缩性，不敢锁更细的）。

### 7. 会话落盘：只存全局 KV，读回重放

```python
sc.save(path)                    # 默认 drop_swa=True：落盘只有 (buf_c, buf_kr, pos)
sc.save(path, drop_swa=False)    # 环也存 → load 逐位精确（调试/对照）
SessionCache.load(path, model)   # 见 swa_dropped 标记 → 自动 replay_swa()
```

- 默认丢环：环的体积是 `(2*n_heads*head_dim + qk_rope_dim)*n_win*dtype` **每层**，
  与全局主 KV 同量级甚至更大；它定长且能重放，存它纯属浪费带宽（V4.1 的做法）。
- 丢环的代价必须说清：读回是**近似**恢复（`n_win` 覆盖不到整段时）。

## 从想法到代码

```
痛点：一套 sink+窗口+跨步 同时管远端近端 → 近端要么糊（窗口小）要么贵（窗口大）
  ↓
直觉：近处用"眼睛盯的那一行"，远端用"扫到的标题"，两者同时进行
  ↓
原理：每层自带 SWA（本层 hidden 产生）；与全局条目同一次 softmax；环定长；
      落盘只存全局 KV，SWA 用最近 n_win 个 token 重放重建（截断到重放段）
  ↓
代码：
  config.swa_window → build_block → HybridBlock(swa_window) → SparseMLAModule.swa = SwaBranch
  _sparse_scores(swa=...)  → 全局静态列 ‖ SWA 列，一次 softmax（band_mask 管因果+窗口）
  _pack_swa(pos_end, cap)  → 三个定长环（行号 = 位置 % cap）
  _swa_read(..., floor)    → 环 → 升序绝对位置列（重放起点之后才可见）
  replay_swa_from_ids      → 只读解码重建环（write_cache=False, swa_floor=start）
  SessionCache.save/load   → drop_swa 默认丢环；load 时 replay_swa() 补齐
```

## 代价与边界

- **显存**：每全注意力层多一条环，`(2*n_heads*head_dim + qk_rope_dim) * n_win * dtype`。
  本仓 base（12 层 / 12 头 / head_dim 64 / rope 32 / bf16，`full_attn_every=4` → 3 个全注意力层）：

  | `n_win` | 每层环 | 3 层合计 | 对照：全局主 KV @32K（2 个存储层）|
  |---|---|---|---|
  | 128（论文取值） | 0.40 MB | 1.2 MB | 21 MB |
  | 512 | 1.60 MB | 4.8 MB | 21 MB |
  | 4096（= 默认 `sparse_window`） | 12.85 MB | 38.6 MB | 21 MB |

  **别把 `sparse_window` 直接搬过来**：SWA 环是逐 token 的满维 K/V（不像主 KV 是 latent），
  窗口开到 4096 时环比它省下来的还贵。建议 `n_win ≤ 512`，按 A/B 定。
- **算力**：近端窗口从"主 KV latent 上投影 + 稀疏 gather"变成"本层独立投影 + 带状掩码"，
  prefill 的窗口部分不再省；换来的是带宽掩码结构更规整（将来可上 flash 式 kernel）。
- **近似有两处**：① 丢环落盘的读回（Bounded Replay）；② 多轮对话里 SWA 只在重放窗口内精确。
  两者都不影响训练，只影响"从盘里恢复的会话"。
- **投机解码与 SWA 互斥**（`_spec_eligible` 恒 False）：并行验证一次写 `t` 行环，
  部分接受会把未确认 token 的行留在环里被后续读到（pitfall 37）。
- **SWA 与 CSA2 正交**：借用层（Reindex/Reuse）只借**全局**主 KV，SWA 环必须各建各的
  （`_pack_swa` 现建，不照抄生产者的 past——pitfall 38）。
- **`kv_cache_bytes` 的 SWA 项不含 `seq_len`**：这是"存储有界"的可检验形式，
  单测直接断言"`L` 的增量里 SWA 一字节都没有"。
- **压缩比正交但不叠加收益**：`compress_ratios` 目前只压**表示**（池化在读取时做），
  全局缓存仍逐 token 存 latent；真正的显存收益要等"压缩条目落盘成块 + 本环替代未压缩窗口"
  的缓存布局重构。本环已经把"未压缩窗口"这一半做完了（详见 `kv-compression-ratio.md`）。
- 单测：`tests/test_layer_swa.py`（关时逐位等价 + 不加键 / 窗口数学对手写滑窗 /
  环定长与多 token 扩容 / `kv_cache_bytes` 与 L 无关 / 因果 + Finite / 梯度检查点幂等 /
  `prefill+解码 == 全前向`（含借用层）/ 短 prompt 也走 SWA 路径 / 环是每层自己的 /
  投机解码关闭 / 重放精确（`n_win ≥ L`，含 hybrid 且**对照"沿用末态"必须差 100 倍**）/
  优于冷启动 + 只读 / 会话落盘两种模式 / 共享缓冲扩容保身份 / 来源对齐 + CLI 三态）。
- 真机冒烟（base 12 层 / d_model 768 / bf16 / RTX 3080 Laptop）：loss 11.89、SWA 分支
  12 个参数梯度 finite（max 4.7e-3）、梯度检查点开/关梯度差 2.4e-6、fp32 下
  `prefill + 解码 1 token` 与全前向差 3.2e-4 / 1.1e-4（跨长度形状差带来的浮点噪声；
  bf16 下这个量级会被线性注意力状态放大，**关掉 SWA 也一样大**，属既有口径）、
  丢环落盘→重放重建后继续解码正常。`kv_cache_bytes`：`swa_window=256` 时
  8192 ctx 5.84→8.14 MiB、32768 ctx 20.84→23.14 MiB——**SWA 项恒 2.30 MiB 不随 L 变**。
