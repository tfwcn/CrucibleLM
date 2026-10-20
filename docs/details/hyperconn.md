# 超连接残差 mHC / Single-Pass mHC

> 一句话：把 `h = h + F` 改成 n 路残差流——均值进层算 F、各路按**双随机矩阵**拌回去；
> V4.1 再把输入混合的系数**错后一层**（Single-Pass），让"混合 + 系数预测"能在一趟里
> 完成，激活读写从 `(4n+4)d` 降到 `(2n+2)d`（Mega-mHC）。

## 痛点

残差连接十年没变过（`x + F(x)`），信息只能顺着一条道走。Hyper-Connections 把残差流扩成
n 路并可学习混合（27B 实测 −0.027 loss），但**无约束**混合连乘会爆炸（Amax 增益飙到
3000，12k 步 loss 起飞）。mHC 用 Sinkhorn 把混合矩阵锁成双随机，压到 ≤1.6。

## 直觉

凉拌菜：n 碗相同的底（embedding 展开），每碗按不同比例（post 写回）拌入同一锅炒菜
（F 输出），再互相匀一勺（混合矩阵）。碗从第一层开始味道就不同（随机 φ 打破对称），
越拌越融合，但总量守恒（双随机：行列和都 =1）。

## 原理（V4.1 §2.4.1）

```
X_l ∈ R^{n×d}                                   # 相邻块之间的 n 路流
原始 mHC   ：X_{l+1} = B_l X_l + C_l F_l(A_l X_l)      (A_l, B_l, C_l) = H(X_l)   eq.2
Single-Pass：X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)  (A_l, B_l, C_l) = H(X_l)   eq.6
```

- `A_l ∈ R^{1×n}`：输入聚合（凸组合，softmax）；
- `B_l ∈ R^{n×n}`：流混合（**双随机**，Sinkhorn 投影；20 轮，论文取值）；
- `C_l ∈ R^{n×1}`：写回（本实现的动态 post，`2σ(·)`）；
- `H`：归一化 + 投影（论文把归一化权重离线折进投影，RMS 除法**放在投影之后**）。

**为什么要移位**：eq.2 里 `A_l` 依赖 `X_l` 的**全维归约**（RMSNorm 平方和 + 投影），
所以"算 A_l → 用 A_l 混合 X_l"非读两遍 `X_l` 不可：

| 实现 | 激活读写 | 说明 |
|---|---|---|
| 三个 kernel（eq.2） | `(4n+4)d` | 残差更新 / 系数 / 混合各读一遍 |
| 单趟语义但两趟 | `(3n+2)d` | 混合要等 A 的归约完成 |
| **Single-Pass + Mega** | `(2n+2)d` | 理想下界：残差读写各一次 |

移位后 `A_{l-1}` 在进入本块时**已知**，于是每个 hidden tile 可以同时喂给"输入混合"
和"系数预测的累加"，一次读完 `X_l`（`activation_traffic()` 把这三种记账写成函数，
单测锁比值）。

三条数学性质（论文 §4.1）：谱范数 ≤1（不膨胀）、乘法封闭（连乘仍双随机）、
Birkhoff 几何（= 置换的凸组合，单调拌匀）。

**对称性警告**（差点做错）：n 路输入初始完全相同，若写回也相同，流永远分不开、
混合矩阵恒为恒等（纯白开销）。破缺靠随机初始化的 φ_post（各流独立）——静态全 1
的 w_post 永远不对称，故直接动态 post。

**恒等起点**：`g=sigmoid(-6)≈0`、`α=0` → `A` 均匀、`H≈I`、`post=1`，输出与单流基线
一致（单测锁 1e-4；实测无量化 12 层 2e-5、开 fp8 KV 时 0.4——量化不连续会放大 1e-7
级差异，不是接线问题）。

## 从想法到代码

```python
# 一趟 tile 循环（Mega-mHC 的 PyTorch 版数据流）：X_l 只读一遍
ss = acc_pre = acc_res = acc_post = 0
for tile in split(d, 4):
    xt = X[:, :, :, tile].float()                 # (b,t,n,tile)
    ss      += xt.pow(2).sum((2, 3))              # RMS 的分子（不乘权重）
    vt       = xt.reshape(b, t, -1) * w[tile]     # 权重折进投影
    acc_pre += vt @ W_pre[:, tile].T              # 三个系数投影一起累加
    acc_res += vt @ W_res[:, tile].T
    acc_post+= vt @ W_post[:, tile].T
rms = rsqrt(ss / (n*d) + eps)                     # 归约完再乘（论文口径）
A_next = softmax(acc_pre * rms * α_pre + b_pre)
S      = Sinkhorn(exp(acc_res * rms * α + b_res)) # 双随机
C      = 2σ(acc_post * rms * α_post + b_post)
x_in   = Σ_s A_prev[s] · X[:, :, s, :]            # ← 上一层给的 A（eq.6）
y      = X + g·(S X − X) + C·F(x_in)              # delta 形式，g=0 时逐位恒等
```

`A_next` 一路穿到下一层：`HybridBlock.forward(..., hc_pre=A_prev)` 返回
`(h, (past, aux), A_next)`，模型 loop 里 `hc_pre` 局部变量逐层传递（`_decode_step` /
`_decode_multi` / `encode_full_hidden` / CED 预填充 / 梯度检查点五条路径都同步）。
第 0 层没有上一层 → A 取均匀（= 零初始化 α 时的 softmax，与旧 mean 聚合逐位一致）。

## 我们的实现

- 文件：`src/llm/local/hyperconn.py`（`HyperConnRes` + `sinkhorn` +
  `activation_traffic`）、`block.py`（`enter`/`combine` 与 `hc_pre` 透传）、
  `model.py`（五条前向路径的 `hc_pre` 串接）；
- 开关：`--hyper-streams n`（0=关，≥2 开；1 直接拒）、
  `--hc-single-pass/--no-hc-single-pass`（默认开，V4.1 eq.6）、
  `--hc-mega/--no-hc-mega`（默认开，分块单趟；关 = 整块 GEMM 的多趟写法）、
  `--hc-sinkhorn-iters`（默认 20）；
- 参数开销：n=4、d=384 时约 4×10k/层，12 层 +0.5%；模块 FLOP 不变（只看均值流）；
- `hc_single_pass` / `hc_mega` 都登记进 `config._ARCH_SWITCHES`：老 checkpoint
  （无这两个键）回退 `False` = 特性前语义，续训不会被静默换架构；
- Amax 监控：`track_stats` 开时记录各层 H 均值，eval 附 `hyper_gain`（复合映射行列和，
  ≈1 健康；默认关，compiled 路径不受影响）；
- 缓存格式不变（attn past 照旧）；旧权重 overlap 载入（缺的仅 hyper 键）。

## 实测（12 层 / 107.5M / n=4 / d=384 / bf16）

| 检查 | 结果 |
|---|---|
| 恒等起点（不含量化） | `max|Δlogits| = 2.2e-05`（12 层） |
| 恒等起点（fp8 KV + 块主序） | `4.4e-01`：量化码字不连续放大 1e-7 级差异（非 bug） |
| 训练（普通 / grad_ckpt） | loss 10.90 / 10.88，梯度全 finite，首层 A 预测器有梯度 |
| 解码 vs 全前向 | `7.8e-06`（A 非均匀、H 非恒等） |
| 投机解码 == 贪心 | ✅（并行验证路径的 `hc_pre` 同样逐层串） |
| Mega vs 多趟（无量化） | `2.9e-05`（同一数学，只差求和顺序） |
| 激活读写（d=384, bf16） | 多趟 15.0 KiB、两趟 10.5 KiB、Mega **7.5 KiB（0.50x）** |
| 前反向耗时（Mega / 多趟） | 1.45s / 1.39s（Python 层 tile 循环无收益；收益在真实 kernel 的显存流量） |

## 代价与坑

- **最后一层的 A 预测器是死参数**：Single-Pass 中它预测的 `A_L` 没有下一块可用
  （等价于第一层省掉的 `A_{-1}`），只有 `phi_pre/bias_pre/alpha_pre` 无梯度。
  单测把死参数集合钉死，避免将来接线漏层靠"全都有梯度"掩盖。
- **恒等起点必须用 delta 形式**：`Hx = x + g(Sx − x)`。若直接算 `(1−g)I + gS`，
  bf16 里 `0.9975·x` 的取整会让"恒等"变成 1e-3 级差异，12 层后放大到 4e-1。
- **Mega 与多趟的等价性要在无量化配置下比**：两条实现差 1e-5，但 fp8/fp4 KV 的
  码字边界是断续的，1e-5 能翻转量化码字 → 差异跳到 1e-1（与 #4 的结论同源）。
- **`hc_pre` 必须作为梯度检查点的显式输入**（同 `ced_src`）：它来自上一层、需要梯度，
  被闭包捕获会在重算时当常量，梯度不回传到上一层的系数预测器。
- 激活显存 ×n（残差流变宽；grad-ckpt 兜着）；`h - h_in` 的相减会损失一点精度
  （既有设计，实测恒等起点 2e-5，可接受）。
- A/B 已结案（128M SFT 500 步、n=2）：ema −0.023，`hyper_gain` 全程 1.0，
  后段进入平台期——赢面集中在早期结构红利；128M 上 −0.021 未必复现（论文是 27B 的数）。
- `float(0维张量)` 会 detach 断梯度——α 必须保持张量运算（修过，单测锁梯度）。

## 一句话总结

残差流扩 n 路、双随机锁稳定、恒等起点；V4.1 再把 A 错后一层换单趟数据流——
数学不变、激活读写减半，代价是最后一层多一个没人用的预测器。
