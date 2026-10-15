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
- 当前只接线到**推理前缀**：视觉 token 拼在文本前送主干；训练 loss 对齐
  （label 位移）尚未接线，传 `targets` 会显式报错而非静默出错。

## 从想法到代码

```python
tokens = patchify(images, self.patch)              # (B,N,C*P*P)
h = self.patch_embed(tokens) + self.pos_embed[:, :N]
pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), self.n_tokens)   # avgpool 路径
return self.out_norm(pooled.transpose(1, 2))       # (B, n_tokens, d_model)
# 主干：h = torch.cat([vision_embeds, self.embed(input_ids)], dim=1)
```

## 代价与坑

- 位置嵌入按上限 4096 patch 预留，超限直接报错（fail-fast，别静默截断）；
- 仅结构预留：不含图像编码器预训练配方，也不含视觉-文本对齐训练；
- 默认关是硬约束：开它会给 `state_dict` 加新键，旧权重需从零/迁移加载这部分；
- 训练 loss 路径未接线，`vision_embeds` + `targets` 同时给会抛 `NotImplementedError`。

## 一句话总结

图像先压到少量视觉 token 再进主干——省 KV、省算力的结构预留，默认关。
