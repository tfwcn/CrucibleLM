# MTP：让模型多看一步

> 一句话：除了预测下一个 token，再加个头预测下下个——训练信号翻倍，推理时关掉不花钱。

## 痛点

标准 LM 训练每个位置只学一个信号（t+1）。语料就那么多，信号密度决定
样本效率。MTP（Multi-Token Prediction，DeepSeek 系）白嫖一份：用同一个
hidden 再押一注 t+2。

## 直觉

做阅读理解时，好学生不只想到下一句，还想到下下句。MTP 就是逼模型
"想远一步"——被迫学更长程的规划，而不是只学局部搭配。考试（推理）时
不加试，平时（训练）多练。

## 原理

```
h            → lm_head → 预测 t+1（主 loss）
第 1 级 MTP(h) → lm_head → 预测 t+2
第 2 级 MTP_2(第1级 ⊕ Emb(t_{i+2})) → lm_head → 预测 t+3
第 3 级 MTP_3(第2级 ⊕ Emb(t_{i+3})) → lm_head → 预测 t+4
```

MTP 头是浅层块，复用主词表头，不新增大矩阵。**第 1 级**（`norm + proj`）只用主干
hidden；**更深级**（`extra.{i}`）在上一级 hidden 之外再喂"下一 token 的嵌入"
（DeepSeek 式值输入：让远视预测挂上真实 token，而不是硬猜），逐级链式。深度由
`mtp_depth` 控制：`1`=旧口径（只押 t+2），`3`=押到 t+4（DeepSeek-V4.1
`num_nextn_predict_layers=3`）。

多级 loss 取**均值**再乘权重（不是求和）：保持 `mtp_loss_weight` 的量纲与深度无关，
不然深度一变、辅助压力跟着翻倍。loss 权重 0.3——MTP 是"辅修"，不能喧宾夺主
（t+2 比 t+1 更难，权重太大主 loss 反而学不好）。

## 我们的实现

- 文件：`src/llm/local/model.py`（`MTPHead` + `_MTPLevel`）
- 参数：`mtp_depth`（默认 1）、`mtp_loss_weight=0.3`
- 第 1 级键名 `mtp.norm`/`mtp.proj`（与旧实现**逐位一致**，可续训旧 run，用于起草）；
  更深级键名 `mtp.extra.{i}.{norm_h,norm_e,proj}`（`depth>1` 才有）
- `MTPHead.forward(h)`：只算第 1 级（`spec_decode`/`measure_mtp_acceptance` 起草用）；
  `MTPHead.forward_all(h0, tok_embeds)`：逐级前向返回各级 hidden（训练用）
- 输入嵌入取**输入序列** `input_ids[:, i+d]`（SFT 的 `-100` 掩码查表会越界，且那些位置
  CE 本就 ignore，故用输入序列更稳）
- RHO 选择时 MTP 全量（aux 和 MTP 都必须看全路由/全序列，不能只挑难的）；
- 推理/评测：MTP 头直接不用（起草只吃第 1 级），零成本。

## 从想法到代码

多押几注就是多几个浅头 + 链式喂入"下一 token"，全部复用主词表头：

```python
n = T - 1 - depth                      # 各级共用位置数
h0 = h[:, :n]                          # 第 1 级输入
tok = [embed(input_ids[:, d:d+n]) for d in range(2, depth+1)]  # 更深级的 token 嵌入
outs = [proj_1(norm_1(h0))]            # 第 1 级
for d, (level, e) in enumerate(zip(extra, tok), start=2):
    outs.append(level(outs[-1], e))    # 第 d 级：上一级 ⊕ Emb(t_{i+d})
losses = [CE(lm_head(o), targets[:, d+1:d+1+n]) for d, o in enumerate(outs, 1)]
loss = main + 0.3 * mean(losses) + aux
```

## 代价与坑

- MTP loss 天然比主 loss 高（t+2 更难猜，实测 3.8~4.7 vs 主 1.6~2.0），
  看曲线别慌，这是口径问题（AGENTS.md 也记了：loss 口径变化要同步解读）。
  深度越大，末级越难（t+4），`mtp_level_losses` 逐级递增是常态。
- 浅层头表达能力弱，t+2 猜不对是常态——它的价值是给主干提供"远视"梯度，
  不是它自己猜多准。
- **`mtp_depth>1` 改 `state_dict` 键**（新增 `mtp.extra.*`）：与 1/2 号开关不同，
  属"改键"的 opt-in——**默认 1**，旧 run 才能续；开启需重训。
  已登记 `config._ARCH_SWITCHES`（回退 1），续训自动沿用。
- **深度 3 的语义**：各级位置数 `n = T-1-depth` 比深度 1 少 2/3，短序列
  （`T ≤ depth+1`）直接跳过 MTP（不报错，只是没有 `mtp_loss`）。
- **输入嵌入取 `input_ids` 而非 `targets`**：SFT 的 `-100` 掩码直接查表会越界，
  且那些位置 CE 本就 ignore——用输入序列既安全又不丢信息。
- 深度只影响**训练辅助目标**，主干前向（推理）不变；起草仍只吃第 1 级，
  所以 `serve_openai` 等推理路径零成本、键兼容（前提是 checkpoint 深度一致）。
- 级联比"多个独立头各自从主干押注"更贴近 DeepSeek：第 2 级能看到第 1 级的中间结论，
  但也因此**串行**（第 3 级必须等第 2 级），多一点点训练步耗时，换来更长的规划链。

## 一句话总结

训练时多押几注（链式、逐级喂真实 token），推理时不花钱——样本效率的免费午餐（打三折吃）。
