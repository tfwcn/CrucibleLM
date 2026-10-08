# 超连接残差：mHC-lite（残差流从 1 路扩成 n 路）

> 一句话：把 `h = h + F` 改成 n 路流：均值进层算 F，各路按双随机矩阵拌回去——DeepSeek mHC（arXiv:2512.24880）只取 H_res 的最小移植。

## 痛点

残差连接十年没变过（`x + F(x)`），信息只能顺着一条道走。Hyper-Connections
把残差流扩成 n 路并可学习混合（-0.027 loss，27B 实测），但无约束混合连乘
爆炸（Amax 增益飙到 3000，12k 步 loss 起飞）。mHC 用 Sinkhorn 锁成双随机
压到 ≤1.6——我们移植它的"流混合"思想，不要它的 TileLang/DualPipe。

## 直觉

凉拌菜：n 碗相同的底（embedding 展开），每碗按不同比例（post 写回）
拌入同一锅炒菜（F 输出），再互相匀一勺（H_res 凸组合）。碗从第一层开始
味道就不同（随机 φ 打破对称），越拌越融合，但总量守恒（双随机行列和=1）。

## 原理

```
H_res = Sinkhorn(exp(α·(RMSNorm(x_vec) @ φ_res) + b))  # 双随机，谱范数≤1
post  = 2σ(α·(x_vec @ φ_post) + b)                     # 逐流写回系数
y_s   = Σ_r H[s,r]·x_r + post_s·F                      # 混合 + 写回
```

三条数学性质（论文 §4.1）：谱范数 ≤1（不膨胀）、乘法封闭（连乘仍双随机）、
Birkhoff 几何（=置换的凸组合，单调拌匀）。n=1 退化为恒等——我们现状就是 n=1。

**对称性警告**（差点做错）：n 路输入初始完全相同，若写回也相同，
流永远分不开、H_res 恒为恒等白开销。破缺靠随机初始化的 φ_post
（各流独立）——w_post 若用全 1 静态值则永不对称，故直接动态 post。
恒等起点：g=sigmoid(-6)≈0 → H≈I；α=0 → post=2σ(0)=1，逐位一致（单测锁 1e-4）。

## 从想法到代码

```python
h_in = x.mean(2)                    # pre=均值聚合（论文消融默认）
...attn/moe/memory/retro 照旧...    # 模块只看均值流，FLOP 不变
H, post = hyper.mappings(x_stream)  # fp32 小矩阵，20 轮 Sinkhorn
y = einsum(H, x) + post * F_out     # 混合 + 写回
```

## 我们的实现

- 文件：`src/llm/local/hyperconn.py`（`HyperConnRes` + `sinkhorn`）
- 开关：`--hyper-streams 2`（0=关；1 直接拒；先 2，赢了再 4）
- 每层约 52k 参数（n=4, d=768），12 层共 0.6M（+0.5%）；
  模块 FLOP 不变（只看均值流），多的是 n×n 拌合（可忽略）；
- pre 用动态凸组合（softmax，零初值即均匀，与旧 mean 路径逐位一致），
  H_res 全动态 + Sinkhorn，post 动态写回——论文 Table 1 的最小集再加 pre，
  pre 开销 12k/层可忽略；
- Amax 监控：`track_stats` 开时记录各层 H 均值，eval 附 `hyper_gain`
 （复合映射行列和，≈1 健康；默认关，compiled 路径不受影响）；
- 缓存格式不变（attn past 照旧），SessionCache/`_decode_step` 只需展开/聚合；
  旧权重 overlap 载入（缺的仅 hyper 键）；

## 代价与坑

- 激活显存 ×n（残差流变宽，12 层约多 3 份 hidden；grad-ckpt 兜着，16G 够）；
- A/B 已结案（sft-hyper 500 步：ema -0.023，hyper_gain 全程 1.0，后 500 步平台期）：
  赢面定格在前 500 步的结构红利，已合入 `sft-full.yaml`（`hyper_streams: 2`）；
- 128M 上 -0.021 未必复现（论文 27B 的数）：A/B 赢的是"-0.023 vs sft6 同段"，
  不是论文数字，继续保持诚实；
- `float(0维张量)` 会 detach 断梯度——α 必须保持张量运算（修过，单测锁梯度）。

## 一句话总结

残差流扩 n 路，双随机锁稳定，恒等起点——拓扑层面的下一代，先 A/B 再说话。
