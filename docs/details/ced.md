# CED 因果编码器-解码器：上半层不再自己造 KV

> 一句话：把前一半层当**因果编码器**，后一半层的全局 KV 不再由自己的 hidden 产生，
> 而是由编码器输出 `H_enc` 用**逐层投影权重**投影出来——于是预填充只需在前缀上算完
> 编码器层，解码器层只跑最后 `2×窗口` 个 token，预填充算力近半。

## 痛点

agent 工作流里最贵的是**预填充**：每次工具调用都带来一段新 prompt，KV 缓存一 miss
就得把这段从头算一遍——而算力花在"造 KV"上（`N` 个 token × `L` 层）。长 prompt
配长上下文，这部分是分钟级开销。

可注意一件事：**前缀的 hidden 是没人要的**。除了最后一个位置要出 logits，前置 token
的输出只在"生成后续 token 的 KV"时才被用到。如果 KV 不依赖该层自己的 hidden，
前缀的层计算就是纯粹的浪费。

## 直觉

两段式流水线：前半段（编码器）读原文，产出一份"摘要流"；后半段（解码器）每层都从
这份摘要流**各自投影**一套 KV，自己只补一份"近处窗口"。

于是对一段已知前缀，后半段只需要处理**尾巴**——因为：
- 它的全局 KV 由摘要流投影而来，与"自己在哪些位置跑过"无关；
- 它的局部窗口 KV 只服务最近 `W` 个位置，更早的没人读。

这就是 YoCo 的思路，CED 在它上面加了"layer-dependent 投影"和"逐层 SWA"。

## 原理（DeepSeek-V4.1 §2.2）

对 `l > L/2` 的层（解码器）：

```
C_l = H_{L/2} · W_l^{KV}        # KV 条目（latent）
Z_l = H_{L/2} · W_l^{Z}         # 对应的压缩权重（CSA2 ratio 用）
```

- `H_{L/2}` 是第 `L/2` 层的输出（编码器最后一层的 hidden）；
- `W_l^{KV}`/`W_l^{Z}` 是**逐层**投影权重（每层一套，所以每层 KV 不同）；
- 全局 Q 与 SWA KV **仍是自己的**（论文：SWA 在所有层维持逐层计算，因为局部
  细节必须来自本层 hidden）；
- 预填充复杂度 `O(NL) → O(NL/2 + n_win·L/2) ≈ O(NL/2)`；
- **Decoder SWA Bounded Replay**：因为解码器 SWA 只能逐层算，预填充时只重放
  prompt 最后 `n_win` 个 token（足够供第一步解码），不是 `L/2 × n_win`。

## 我们的实现

### 1. 开关与切分：`ced_split`

```python
# config.py
ced_split: int = 0        # 0=关；>0 = 前 ced_split 层为编码器，其余为解码器
ced_prefill_skip: bool = True   # 推理预填充跳算（默认开）
```

- 解码器层的 `w_dkv`/`w_kr` 被 `w_ced_kv`/`w_ced_kr` 取代 → **改键**，故默认关、
  登记 `_ARCH_SWITCHES`，续训/迁移从 `hparams.json` 沿用（读不到回退 0）。
- **早失败约束**：`0 < ced_split < n_layers`，且解码器区间必须**全是全注意力层**。
  线性注意力层的递推状态没法从 `H_enc` 投影，跳算会静默算错（pitfall 42 记了笔）。
  所以本仓 1:3 hybrid 的 base 配置**不满足**——CED 需要 `full_attn_every=1`
  （全注意力）这类配置。这是诚实的边界，不是偷懒。

### 2. KV 投影：两个 helper 收口

```python
# mla.py
def _kv_latent(self, h, ced_src=None):    # 主 KV 的 latent
    return self.w_dkv(h) if not self.ced else self.w_ced_kv(ced_src)
def _rope_key(self, h, ced_src=None):     # 共享 rope 键（同样来自 H_enc）
    return self.w_kr(h) if not self.ced else self.w_ced_kr(ced_src)
```

**rope 键也必须从 `H_enc` 来**：否则跳算前缀时还得在前缀上跑解码器层才能算 rope 键，
省算就落空了。`_qkv` 里 Q 侧长度 = 查询段、KV 侧长度 = 全序列（跳算时两者不同）。

### 3. 前向接线：`H_enc` 就是"第 ced_split 层的输入"

```python
# model.py:forward / _prefill / _decode_step / encode_full_hidden / _decode_multi
h_ced = None
for li, layer in enumerate(self.layers):
    if ced_split and li == ced_split:
        h_ced = h            # 第 ced_split 层**之前**的残差流 = 编码器最后一层输出
    ced_src = h_ced if (ced_split and li >= ced_split) else None
    layer(..., ced_src=ced_src)
```

超连接（n 路流）时取均值流（`h.mean(dim=2)`），与"模块只看均值流、FLOP 不变"一致。

### 4. 推理预填充跳算：只跑尾巴

```python
# model.py:_prefill（ced_split>0 时）
q_off = self._ced_prefill_q_off(n_tok)      # 块对齐的起点
h_enc = 编码器段的输出（全序列）
h = h_enc[:, q_off:]                        # 解码器段只喂尾段
for layer in 解码器段:
    h, p = layer(h, ..., ced_src=h_enc, q_off=q_off)
```

- **必须块对齐**：全局稀疏 pattern（sink/stride/窗口）依赖查询块起点；不对齐时末位置
  与全量差 `1e-1`，对齐后**逐位一致**（pitfall 40）。
- **尾段取 `2 × swa_window`**：窗口列的 K/V 来自窗口内各位置自己的 hidden，
  "将来会被读到的 `W` 个环行"自身也要自洽 → 再往前留一个 `W`。实测 tail=2W 时末位置
  与环都逐位一致；tail=W 时差 `1e-1` 量级。
- 缓存依然**全序列**：解码器层的 `buf_c` 长度 = `N`（来自 `H_enc` 的全序列投影），
  `pos = N`——后续解码读历史全局 KV 不受跳算影响。
- `ced_prefill_skip=False` 时跑全序列（严格对齐训练口径、不省算），用于做 A/B 与排障。

### 5. 与 CSA2 / 层内 SWA 的关系

- **压缩权重 `Z_l`（论文式(1) 的另一半）**：论文里 `Z_l = H_enc W_l^Z` 是"每个条目的
  压缩权重"，也是从 `H_enc` 投影的。本仓把它简化成 `compress_ratios` 的**共享**权重
  `z_cmp`（块内位置 × 通道，不依赖 hidden）——简化已记在 `kv-compression-ratio.md`；
  CED 下 `z_cmp` 仍然作用在 `C_l` 上（池化源变成编码器投影的 latent），语义不变。
- **CSA2 复用**：CED 解码器层**恒为 Full**（各层自己的投影），不借前层缓存。
  代价是存储：`kv_cache_bytes` 里解码器层各算一份（实测 base-full 配置 8192 ctx
  下 19.6 → 27.1 MiB，**+38%**），换来的是预填充省 44% 的层计算。这是"用存储换算力"
  的显式取舍，不是 bug。
- **层内 SWA**（#7）：正交且互补。SWA 提供"近处精确"，CED 提供"上半层不造 KV"；
  CED 的跳算长度正是由 SWA 窗口决定的（`swa_window=0` 时甚至只跑最后 1 个 token）。
- **梯度检查点**：`ced_src` 必须作为 `checkpoint` 的**显式输入**传入，否则闭包里
  被当常量、梯度静默断在编码器（pitfall 41）。

## 从想法到代码

```
痛点：预填充把算力花在"造 KV"上；前缀 hidden 没人要，却要跟着算完 L 层
  ↓
直觉：两段流水线——前半段产摘要流，后半段各层从摘要流各自投影 KV，只补局部窗口
  ↓
原理：C_l = H_{L/2} W_l^KV（逐层投影）；SWA 仍逐层；Decoder SWA Bounded Replay 只重放 n_win
  ↓
代码：
  config.ced_split → build_block(ced=config.is_ced_decoder(i)) → MLAModule.ced
  MLA._kv_latent/_rope_key(h, ced_src)  → 解码器层 KV/rope 键来自 H_enc
  TinyLLM.forward/_decode_step          → 逐层记录 h_ced 并下传（含 eng/grad_ckpt 路径）
  TinyLLM._prefill + _ced_prefill_q_off → 尾段 = 2×窗口、起点块对齐，缓存仍全序列
  TinyLLM._prefill 的 q_off            → SparseMLAModule._run_blocks 按绝对位置分块
```

## 代价与边界

- **hybrid 上不可用**：解码器区间必须全是全注意力层（线性层递推状态不能投影）。
  本仓 base（`full_attn_every=4`）不满足，要用 CED 得换 `full_attn_every=1` 之类的配置。
- **存储不降反升**：解码器层恒 Full（不借前层），`kv_cache_bytes` 会变大（见上）。
  CED 的卖点是**算力**，不是显存。
- **跳算的精确性**：`swa_window=0` 时严格逐位等价（fp32 实测差 1e-6 量级）；
  `swa_window>0` 时尾段首部那 `W` 个位置内部的窗口不全，理论上是论文承认的近似。
  我们按 `2×窗口` 留足余量后实测 fp32 差 `1.3e-6`、bf16 差 `1.4e-2`（bf16 的形状差
  本仓本来就是这个量级，关掉 CED 也一样）。
- **训练不跳算**：训练要每个位置的 loss，所以解码器层在全序列上跑；这与论文
  "post-training 里模拟同样的 replay 做 train-aware adaptation"不完全一致——
  我们没做那一步，跳算的近似只出现在推理侧。
- **真机冒烟**（12 层全注意力 / `ced_split=6` / `swa_window=128` / 8K prompt）：
  解码器层处理 `49152 → 6144` token·层（**省 87.5%**），总 token·层
  `98304 → 55296`（**省 43.75%**，对上论文的"近半"）；loss 11.88、`w_ced_*`
  12 个参数梯度 finite（1.9e-3）、编码器 `w_dkv` 梯度 finite、跳算缓存可继续解码、
  贪心生成正常。
- 单测：`tests/test_ced.py`（关时逐位等价 + 不加键 / 键集切换 / **解码器 latent ==
  `w_ced_kv(H_enc)`** / 置零 `w_ced_kv` 改输出 / 因果 + Finite / 梯度检查点下梯度
  仍回流编码器 / 跳算无 SWA 时逐位等价 / `ced_prefill_skip=False` 全量对齐 /
  SWA 开时 2W 口径（含 tail=W 必错的对照）/ `prefill+解码 == 全前向` /
  解码器含线性层早失败 / 解码器恒 Full / 来源对齐 + CLI 三态）。
