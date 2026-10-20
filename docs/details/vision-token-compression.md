# 视觉入路：DeepSeek-ViT（patch14 + 2D-RoPE + 3×3 unshuffle + MLP projector）

> 一句话：图像先过一层**原生分辨率**的小 ViT 看清局部，再按 3×3 邻域合并把 token
> 数打九折（÷9），最后用 2 层 MLP 投影到主干宽度，拼在文本 embedding 前面。

## 痛点

多模态的账全在**视觉 token 数**上：1344×1344、patch 16 → 7056 个 token，一次前向
就要为它们付 KV 缓存和 O(N²) 注意力，比正文长几十倍。但直接把 patch 数砍掉
（比如把 patch 放大到 42）又会让网络在进第一层之前就丢掉细节。

## 直觉

先让每块 14×14 的小 patch 在 ViT 里**看清自己的邻居**（细粒度），再按 3×3 邻域
把特征拼起来降分辨率——信息已经"记"在特征里了，合并时不会丢细节。

## 原理（V4.1 §2.1.1）

```
(B,C,H,W) → patchify(14)  → N = H·W/14²
          → 线性 patch_embed            （不是卷积）
          → + 2D-RoPE                   （行/列各占一半维度）
          → ViT × L                     （双向注意力，RMSNorm + SwiGLU）
          → 3×3 特征 unshuffle          （token ÷9，通道 ×9）
          → MLP projector（2 层）        → d_model
          → [可选] 压到 n_tokens         （avgpool / perceiver）
          → 拼在文本 embedding 前
```

四个设计点都有明确的"为什么"：

| 设计 | 替代方案的问题 | 好处 |
|---|---|---|
| **2D-RoPE**（无绝对位置表） | 可学习绝对位置表把最大分辨率钉死，换分辨率要重训 | `q·k` 只依赖 `(Δrow, Δcol)`，任意 H×W 都能编 |
| **线性 patch embedding** | 卷积核在 Muon 下难以做正交化更新 | 等价于无重叠卷积，但能进 Muon |
| **RMSNorm + SwiGLU** | 视觉塔与主干两套实现 | 与主干同构，少一条分叉 |
| **3×3 unshuffle 在特征网格上** | 在像素上先 unshuffle（= 有效 patch 变 42） | 细节在 ViT 里看过再合并，不吃亏 |

**token 数账**（论文口径，`vision_grid()` 可直接算）：

| 输入 | 网格 | unshuffle 后 |
|---|---|---|
| 336×336 | 24×24 = 576 | 64 |
| 672×672 | 48×48 = 2304 | 256 |
| 1344×1344 | 96×96 = 9216 | **1024** |

（1344 = 14×96，96 被 3 整除 → 1024 token，正好是论文的"约 1344×1344 上限"。）

## 从想法到代码

```python
# 2D-RoPE：行/列各半，半内按 (实,虚) 相邻对旋转
cos_r, sin_r, cos_c, sin_c = rope2d(nh, nw, head_dim)   # (1,1,N,hd/4)
q = cat([rot(q[..., :hd//2], cos_r, sin_r),             # 前一半 = 行
         rot(q[..., hd//2:], cos_c, sin_c)], -1)        # 后一半 = 列

# 3×3 特征 unshuffle（B, nh·nw, d) → (B, (nh/3)(nw/3), 9d)
x = x.view(b, nh//3, 3, nw//3, 3, d).permute(0, 1, 3, 2, 4, 5).reshape(...)

# 2 层 projector → 主干宽度
x = Linear(hidden→d_model)(GELU(Linear(9d→hidden)(x)))
```

## 我们的实现

| 位置 | 内容 |
|---|---|
| `src/llm/local/vision_compress.py` | `patchify`/`unpatchify`、`vision_grid`、`rope2d`/`apply_rope2d`、`ViTBlock`、`VisionCompressor`、可选 `PerceiverBlock` |
| `src/llm/local/model.py` | `vision_enabled=True` 时挂 `model.vision`；`encode_vision()`（推理）/ `forward(vision_embeds=...)`（训练，视觉段不进 CE） |
| `src/llm/local/train.py` | `train_step(vision_images=...)` 走**带梯度**的编码器 |

开关（默认关；`vision_enabled=False` 时模型里没有该模块，`state_dict` 逐位兼容）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `vision_enabled` | False | 总开关（结构性，多 `vision.*` 键） |
| `vision_patch` | **14** | 论文口径（旧默认 16 已改；无历史视觉 ckpt，见 pitfalls 58） |
| `vision_unshuffle` | **3** | 3×3 特征 unshuffle → token ÷9；`1` = 不做 |
| `vision_d_vit` / `vision_vit_layers` / `vision_vit_heads` | 256 / 12 / 4 | 论文 1024 / 32 / 16（本仓库按小模型缩） |
| `vision_projector_hidden` | 512 | 2 层 MLP projector 隐层（论文 5120） |
| `vision_tokens` | **0** | 0 = 不二次压缩（保留 unshuffle 后的网格）；>0 再加池化/查询压缩 |
| `vision_compress` | `avgpool` | 二次压缩方式（`perceiver` 用可学习查询） |

## 实测

| 检查 | 结果 |
|---|---|
| token 数 | 336²/672²/1344² → 64/256/**1024**（÷9 与论文一致） |
| 分辨率可变 | 同一个塔吃 168²/336²/672² 都前进（无位置表上限） |
| 视觉塔参数 | 论文口径（32 层 / 1024 / 16）**588.7M**；本仓库默认（12/256/4）**14.3M** |
| 前向耗时 | 336²（64 token）2 层塔、CPU 约 **27ms** |
| 与主干串联 | 64 视觉 + 16 文本 token，logits 长 80；CE 只算文本段；视觉塔梯度全 finite |
| 双向性 | 改右下 patch 会改变左上 patch 的输出（因果掩码才会不变） |
| 单测 | `tests/test_vision_compress.py` **21 项**全绿 |

## 代价与坑

- **H/W 必须被 `patch × unshuffle` 整除**：不整除直接报错，而不是悄悄丢边
  （悄悄丢边会让同一张图在不同尺寸下语义漂移）。224×224 + patch14 + r=3 就是
  这种情形（16 不被 3 整除）——要么补边到 336，要么 `vision_unshuffle=1`。
- **2D-RoPE 要求 `head_dim % 4 == 0`**（行/列各一半、每半按对旋转）。
- **视觉注意力是双向的**：不能复用语言侧 `HybridBlock`（那个带因果掩码），
  否则 0 号 patch 看不到后面的 patch，"看整张图"这件事就废了。
- **视觉 token 进 MoE 时**：本仓库**未**实现论文的"文本/图像各自一套路由偏置"
  （§2.1.1 的多模态 aux-free 负载均衡）。当前视觉 token 与文本共享同一套
  `expert_bias`；接多模态训练前建议补上，否则两种分布的负载会互相干扰。
- 二次压缩（`vision_tokens>0`）是本仓库为小模型加的**额外**手段，论文不压
  （1024 token 直接进 LLM）；它会让位置语义变模糊，别在需要细粒度定位的任务上开。

## 本地训练冒烟（链路是否真的可训）

结构对不代表梯度通。`scripts/smoke_vision_train.py` 用一个**视觉是唯一变量**的
合成任务验证全链路（RTX 3080 Laptop，600 步约 30 秒）：

| 项 | 设置 |
|---|---|
| 任务 | 42×42 灰度图，整体亮度 = 数字/9，要求输出数字 0–9 |
| 视觉侧 | patch14 → 3×3 网格 → 2 层双向 ViT → `unshuffle=3` → **1 个 token** |
| 模型 | 0.92M（其中视觉塔 0.26M） |
| 监督 | 提示 `"\n答"` + 答案 `数字 + EOS`，全序列监督 |
| 优化 | AdamW lr=1e-3 + cosine，batch 32 |

结果：100 步 70.5%，**200 步起 100% 并保持到 600 步**（1024 样本复测 100%）。

```bash
python scripts/smoke_vision_train.py --steps 600 --batch 32
```

**边界**：这证明的是"ViT + unshuffle + projector + 前缀对齐 + CE 位置"这条链路的
梯度是通的、能学会把图像信息用于输出；它**不**证明视觉能力（任务只有一个标量）。
另外两个容易踩的坑（答案位下标、token id vs 数值）见 pitfalls 63/64——它们会让
"loss 正常下降但准确率恒 0%"。


## 一句话总结

先在 14×14 的粒度上"看清"，再按 3×3 合并"瘦身"，位置用相对编码所以分辨率随便改——
视觉前缀的成本从 9216 token 降到 1024，且没有丢掉进入网络前的细节。
