# RoPE 与 YaRN：从 8K 训练到 1M 外推

> 一句话：位置藏在旋转里；训练只见 8K，靠 **NTK-by-parts** 把低频维度"调慢"、
> 高频维度"不动"，再把幅度补一点，撑到 1M（256 倍）。

## 痛点

Transformer 天生不认识顺序。绝对位置表（学 8192 个向量）外推即死——第 8193 个
位置没见过。RoPE 好得多（相对位置 + 可外推的形式），但训练 8K、推理 1M 时仍然崩：

- **高频维度**（管相邻 token 的区分）转了太多圈 → 混叠，"隔壁"和"很远"看起来一样；
- **低频维度**（管长距离）训练时就没转够 → 1M 处几乎分辨不出先后。

朴素修法 `inv_freq / s` 把所有维度一起拉慢：低频是对了，**高频的分辨率跟着陪葬**，
短文本反而掉点。

## 直觉

把每个维度想成一块转速不同的表盘。YaRN 说：**高频表盘别动**（短距离还要靠它），
**低频表盘调慢 s 倍**（长距离本来就没见过，插值更安全），中间用斜坡平滑过渡——
避免两个区段之间出现"频率断点"（断点会让位置编码在某个距离上突然失去区分度）。

## 原理（NTK-by-parts）

```
inv_freq_i = 1 / θ^(2i/d)                          # 标准 RoPE
λ_i        = 2π / inv_freq_i                       # 该维度的波长
r_i        = clamp((i − low) / (high − low), 0, 1)  # 斜坡
inv_freq_i' = (inv_freq_i / s)·r_i + inv_freq_i·(1 − r_i)
mscale      = 0.1·ln(s) + 1                        # 幅度校正
```

- `low = ⌊correction_dim(β_fast=32)⌋`、`high = ⌈correction_dim(β_slow=1)⌉`，
  `correction_dim(n) = d·ln(L/(n·2π)) / (2·ln θ)`（L = 训练长度）；
- 下标小（高频）→ `r=0` → **外推，频率不动**；
- 下标大（低频）→ `r=1` → **插值，频率 ÷s**；
- 中间 → 线性混合。

`mscale` 我们**折进 cos/sin**（等价于把 q、k 各放大 `√mscale`，内积放大 `mscale`），
所以不用改任何注意力实现。

**注意方向**：`r=0` 必须是"外推"（不动），`r=1` 才是"插值"（÷s）。写反了会
"高频被拉慢、低频不动"——短距离分辨率直接崩（我们踩过，见 pitfalls 59）。

## 从想法到代码

```python
inv  = 1 / theta ** (arange(half) / half)              # 标准频率
low, high = correction_dim(beta_fast), correction_dim(beta_slow)
ramp = clamp((arange(half) - low) / (high - low), 0, 1)
inv_yarn = (inv / s) * ramp + inv * (1 - ramp)         # ramp=1 → 插值
cos, sin = cos(pos ⊗ inv_yarn) * sqrt(mscale), sin(...) * sqrt(mscale)
```

## 我们的实现

| 位置 | 内容 |
|---|---|
| `src/llm/local/rope.py` | `yarn_inv_freq`（NTK-by-parts）、`yarn_mscale`、`build_rope_cache`、`apply_rope`、`yarn_scaling_factor`、`yarn_scale_mismatch` |
| `src/llm/local/mla.py` / `sparse_attn.py` / `block.py` | YaRN 四参（`yarn_scale`/`yarn_beta_fast`/`yarn_beta_slow`/`yarn_original_len`）逐层透传 |
| `src/llm/local/config.py` | `longctx_config()`（200K）、**`longctx_1m_config()`（1M）** |
| `src/llm/local/longctx.py` | 阶段表、启动自检（64K 必须稀疏）、YaRN 参数建议 |

| 配置 | 默认 | 说明 |
|---|---|---|
| `yarn_scale` | 2.0 | = 目标长度 / 训练长度（8K→1M 是 **128**） |
| `yarn_original_len` | 8192 | **真正训过的长度**（分界靠它算，填错等于分界错） |
| `yarn_beta_fast` / `yarn_beta_slow` | 32 / 1 | 高频/低频分界（YaRN 论文默认） |

> `yarn_scaling_factor()` 的上界从 8 放宽到 **512**：4K→1M 是 256 倍，旧上限会让
> 1M 直接算成 8 倍（等于没外推）。

## 实测

| 检查 | 结果 |
|---|---|
| `yarn_scale=1` | cos/sin 与标准 RoPE **逐位相同**（老 ckpt 零影响） |
| NTK-by-parts | 高频维度（下标 0）恒等于标准值；低频维度精确 `÷s`；频率随维度单调不增 |
| 1M 可分辨性 | 1M 处相邻位置的 cos 仍有 1e-3 量级差异（没有塌成同一点） |
| RoPE 缓存 | `seq×dim` 固定开销：64K×16 ≈ 8MB；**1M×64 = 512MB**（自检会给警告） |
| 单测 | `tests/test_longctx.py` **19 项**全绿 |

## 代价与坑

- **`yarn_original_len` 填错比不开 YaRN 更糟**：分界按它算，填成 1M 会让"高频不动"
  的区段覆盖到不合适的维度。它必须等于骨干**真正训过的**长度。
- **1M 的 RoPE 缓存是 512MB**（`2×seq×dim×4B`）：fp32 建的，属固定开销；
  `longctx.validate_longctx` 超 256MB 就警告。
- **外推 ≠ 有质量**：YaRN 只解决"位置编码不崩"，1M 的**能力**仍要在长文上训
  （V4.1 是 34T token 时从 64K 拉到 1M）。别指望 8K 短训 + YaRN 直接得到 1M。
- 解码成本另算：见 `docs/details/long-context.md`（`kv_block_cache`/SWA 才让
  单 token 解码 FLOPs 近似常数）。

## 一句话总结

高频不动、低频调慢、中间斜坡、幅度补一点——`scale=1` 逐位不变，`scale=128`
把 8K 的模型推到 1M 的位置语义。
