# 视觉 token 压缩：图像先瘦身再进主干

> 一句话：把图像 patch 先编码并压缩到固定少量视觉 token，再投影进主干——
> 缩短视觉前缀、省 KV 与算力。**结构预留，默认关**。

## 痛点

多模态里图像 patch 数很多（如 224×224、patch16 → 196 个 token），一次前向就要
为它们付出 KV 缓存与注意力算力，常常超过文本本身。直接把 patch 全塞进主干很浪费。

## 直觉

看一张图不需要逐像素复述——先扫一眼提炼几个"要点"（少量 query token）就够了。

## 原理

```
图像 (B,C,H,W) → patchify 成 N 个 patch
              → 线性 patch_embed + 位置嵌入
              → 压缩到 n_tokens（avgpool 无参 / perceiver 可学习查询 cross-attn）
              → 投影到 d_model → 拼在文本 embedding 前
```

`avgpool`：自适应平均池化到固定 token 数，零额外查询参数；
`perceiver`：一组可学习 query 对 patch 做 cross-attention 聚合，表达更强。

## 我们的实现

- 文件：`vision_compress.py`（`patchify`/`unpatchify` + `VisionCompressor`）、
  `model.py`（`TinyLLM.vision` 可选模块 + `encode_vision` + `forward(vision_embeds=...)`）；
- 配置：`vision_enabled`（默认关）、`vision_tokens`、`vision_patch`、
  `vision_compress`、`vision_d_in`；
- 默认关时模型里根本不存在该模块，`state_dict` 与旧 checkpoint 逐位兼容
  （与 memory/retro 的"默认 None"同风格）；
- **推理**：`encode_vision`（`no_grad`）+ `forward(vision_embeds=...)`，视觉 token
  拼在文本前；
- **训练 loss 已接线**：`forward(targets=..., vision_embeds=...)` 只对**文本段**算 loss
  —— logits 取 `[:, V : V+T-1]` 对齐 `targets[:, 1:]`（V 为视觉 token 数），视觉段
  不参与 CE；MTP 同样只取文本段（`h[:, V : -2]` 预测 `targets[:, 2:]`）。
  `V=0` 时该式退化为旧式，逐位不变（单测锁定）。
- 训练侧入口 `train.py:train_step(..., vision_images=...)`：经 `model.vision`（**带梯度**）
  编码，编码器参数一起吃梯度（`encode_vision` 是 no_grad 的推理版，勿用于训练）。

## 从想法到代码

```python
tokens = patchify(images, self.patch)              # (B,N,C*P*P)
h = self.patch_embed(tokens) + self.pos_embed[:, :N]
pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), self.n_tokens)   # avgpool 路径
return self.out_norm(pooled.transpose(1, 2))       # (B, n_tokens, d_model)
# 主干：h = torch.cat([vision_embeds, self.embed(input_ids)], dim=1)
# 训练：只算文本段错位一位（视觉段不进 CE）
text_logits = logits[:, V : V + T - 1]             # V=视觉 token 数
loss = CE(text_logits, targets[:, 1:])
```

## 代价与坑

- 位置嵌入按上限 4096 patch 预留，超限直接报错（fail-fast，别静默截断）；
- 仅结构预留：不含图像编码器预训练配方，也不含视觉-文本对齐训练；
- 默认关是硬约束：开它会给 `state_dict` 加新键，旧权重需从零/迁移加载这部分；
- **loss 位移易错**：视觉前缀把位置整体后移 V，CE 必须取 `logits[:, V:V+T-1]`
  才对齐 `targets[:, 1:]`；错当成 `logits[:, :-1]` 会把视觉 token 的预测算进 loss
  且整体错位（单测 `test_vision_train_loss_alignment` 用参考实现卡死）。

## 一句话总结

图像先压到少量视觉 token 再进主干——省 KV、省算力的结构预留，默认关。
