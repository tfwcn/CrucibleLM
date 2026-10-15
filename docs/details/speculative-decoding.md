# 投机解码：先猜后验，一次吐多个

> 一句话：复用已训好的 MTP 头 + 前缀 n-gram 起草后续 token，主干一次并行验证，
> 接受最长草稿前缀就一次提交多个 token；**输出与逐 token 贪心逐位一致**。

## 痛点

batch=1 解码是 launch-bound：单步约 1800 个小算子，每个付一次 Python 分发与
kernel 启动（见 `inference-acceleration.md`）。每个 token 都要完整走一遍主干，
GPU 算得飞快却大部分时间在等启动。

## 直觉

阅读理解时你往往能"猜"到下一句，猜对了整段都不用逐字推。投机解码就是先让一个
便宜的猜手（draft）押几个 token，再让权威的主干一次性核对——猜对了就白拿。

## 原理

```
每轮：主头出权威下一位 t0
      起草器补 k-1 个后续候选 d2..dk
      喂 [t0, d2..dk] 一次并行前向（_decode_multi）
      得到主干在每个位置的贪心 → 与草稿逐位比
      接受最长草稿前缀 + 主头的下一个 token → 提交，状态对齐
```

关键：**验证必须与逐 token 解码同算术**，否则主干"验证出的贪心"和"真贪心"
会在近并列 logit 上分叉，"等于贪心"的契约就守不住。因此本模型的多 token 增量
解码三条路径都做了对齐：MLA 多查询 SDPA（逐行独立）、稀疏路径按
`start=total-t` 聚集、线性层按单步同款递推。

## 我们的实现

- 文件：`spec_decode.py`（起草器：`ngram` / `mtp` / `hybrid`）、
  `model.py`（`_spec_stream` 主循环 + `_decode_multi` 并行验证）；
- 配置：`draft_enabled`（默认关）、`draft_window`、`draft_source`、`draft_ngram_min`；
- 为支撑并行验证，三处解码路径补了多 token 能力（均带回归单测）：
  - `mla.py:_valid_mask`：`t>1` 时补块内因果掩码（原来只掩空位，会偷看同批未来）；
  - `sparse_attn.py:_decode`：查询改用最后 t 个位置（`total-t`）；
  - `linear_attn.py`：新增 `past≠None, t>1` 分支，按单步同款递推（状态初值 `s_prev`）。
- 回滚：部分接受时缓存回退到本轮起点，逐 token 重放已提交前缀。MLA 缓冲按
  `pos` 回退即可（空位被掩码），线性层状态是新建张量（非原地改），故回退零拷贝。
- 门槛：先跑 `scripts/measure_mtp_acceptance.py`，接受率 > 50% 才值得开。

## 从想法到代码

```python
# 起草 → 并行验证 → 接受最长前缀
drafts = provider.propose(ctx, h[:, -1:], window - 1)   # ngram/mtp/hybrid
feed = [t0] + drafts                                     # t0 为主头权威下一位
h_multi, pasts = self._decode_multi(feed, pasts)         # 一次前向喂完整窗口
accepted = 最长前缀(草稿 == 主干在对应位置的贪心)
# 全接受：直接用并行缓存；部分接受：回退起点逐 token 重放前缀
```

## 代价与坑

- 只有 batch=1 且 `temperature=0`（纯贪心）走此路径；采样/批量自动回落普通生成；
- 起草命中率低时会有"验证一轮又重放"的额外开销——务先用接受率脚本卡门槛；
- `repetition_penalty` 已按"seen 上下文逐步累积"对齐，验证时逐位应用，契约不破；
- 多 token 解码的三处改动均默认不影响 `t==1`：`_valid_mask` 在 `t==1` 时与旧式
  逐位相同，稀疏/线性分支只在新代码路径生效（单测锁定）。

## 一句话总结

起草猜、主干验、接受前缀一次吐——复用 MTP 头与 n-gram，零训练拿到解码加速，
且严格不改变输出。
