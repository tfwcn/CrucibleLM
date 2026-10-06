# 推理加速：从 13 到 30.9 tok/s 的三级跳

> 一句话：MoE 分组（砍 dispatch）→ 静态缓存（形状恒定）→ compile（融算子），batch=1 在 RTX 3080 Laptop 上 13→30.9 tok/s。

## 痛点

128M 的模型理论上早该几十 tok/s，实测 13——batch=1 单 token 解码下，
每个张量都极小，GPU 几微秒算完就闲着；但每步约 1800 个小算子，
每个付一次 Python 分发（~10μs）+ kernel 启动（~10–20μs）：
**不是算不动，是启动太多**（launch-bound）。

## 三级跳

**第 1 级：MoE 分组 bmm（13→22.5，1.7x）**
16 次逐专家 Python dispatch → 每个 token 展开 top-k 行 → 按专家排序垫齐 →
3 次 bmm 全算完 → 散射加回。数学逐位一致（单测锁），训练前向同路顺手加速。

**第 2 级：静态缓存（形状恒定，不直接提速，是第 3 级的门票）**
`cat` 增长改预分配 + 按位 `index_copy_` 写 + 全缓冲掩码读，
past 从二元组变 `(buf, buf, pos)` 三元组（`pos` 是 0 维 long 张量，
值动态、形状恒定）。`_prefill` 按 `prompt+max_new` 精确预留，
超了 fail-fast（IndexError），长会话走 `ensure_room` 2x 扩。
附带修了两个真 bug：SDPA bool 掩码 `True`=放行（写反了会全零输出），
rope 下标钳制（预留 L 可超 rope 缓存，空位反正被掩）。

**第 3 级：compile（22.5→30.9，1.37x）**
`model.compile_decode()`（`mode="default"` + `dynamic=True`）。
两道墙都是实测撞出来的：
1. cudagraphs（reduce-overhead）要静态内存，`cat` 时代直接炸——
   静态缓存就是为它铺的路，但 cudagraphs 依然不用（默认模式已够）；
2. sparse 的 `select_keys` 是纯 Python 集合逻辑，dynamo 按 `start` 的值特化、
   每步重编——整段 `_decode` 标 `@torch._dynamo.disable`（eager 跑不重编，
   输出形状恒定，下游不断）。
约束：编译后改权重/切精度/切模式必须重调；训练路径不用。

**第 0 级（白送的）：bf16 推理**
`load(..., dtype="bf16")`，同速（launch-bound 下带宽不是瓶颈）但显存减半，
200K 长上下文有用。RMSNorm 内部 fp32，half 安全。

## 我们的实现与实测

- 文件：`moe.py`（分组）、`mla.py`/`sparse_attn.py`（静态缓存）、
  `model.py`（`compile_decode`）、`infer.py`（`dtype`）
- batch=1、base 权重、bf16：eager 22.5 → compile 30.9；编译/原生真权重逐位一致
- 反面教材：静态缓存之前 compile 是 5.1（越编越慢）——"本地无效不进库"，
  当时直接删了方法只留教训，缓存落地后才请回来

## 代价与坑

- prefill 缓存从 T 长变 L 长（多预留区），200K 会话按需 reservation，别瞎扩；
- `ensure_room` 的 doubling 在编译图里会触发重编一次——均摊可忽略，
  但别在循环里反复横跳；
- 完整血泪见 pitfalls 第 9 条。

## 一句话总结

砍 dispatch、定形状、融算子——推理优化的通用三板斧，缺一不可，顺序不能反。
