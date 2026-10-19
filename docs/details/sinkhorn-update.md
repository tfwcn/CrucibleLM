# Sinkhorn 更新：给"token×特征"大矩阵一个公平步长

> 一句话：embedding / 预测头 / Engram 表这类大矩阵不再用 Adam，而是"动量 + 行/列交替
> 归一化"（Sinkhorn 平衡）——只存**一份**动量缓冲，还把每个 token 的更新幅度拉平。

## 痛点

embedding 与预测头是**矩阵轴不对称**的参数：一行一个 token、一列一个特征。Adam 给
每个**元素**各存一份动量和方差，于是：

- 优化器状态翻倍：词表 8192 × d_model 768 的 embedding 就是 6.3M 参数，Adam 要
  两份 fp32 状态（≈48MB）；词表再大（128K）就是 GB 级；
- 高频 token 的梯度又密又大，低频 token 的梯度小而稀——逐元素自适应**救不了**
  "整行几乎不更新"这个结构性问题。

V4.1 的观察：新引入的 Engram 表让 Adam 状态进一步膨胀，于是把这类矩阵换成
"动量 + Sinkhorn 平衡"，**状态减半**且实测优于 Adam。

## 直觉

Sinkhorn 平衡是在两张"排行榜"之间来回校准：
- 按**行**归一化 → 每一行（每个 token）的更新幅度拉平（不让高频词淹没低频词）；
- 按**列**归一化 → 每一列（每个特征）的更新幅度拉平（不让强势特征抢走整个步长）；
- 交替做 K 次（奇数行、偶数列），两边都趋于"单位 RMS"。

它和 Muon 是一家人：**Muon 用 Newton-Schulz 正交化，Sinkhorn 用交替归一化**，
两者都只需要一份动量，都作用于矩阵的**整体形状**而不是逐元素。

## 原理（DeepSeek-V4.1 §2.5 Algorithm 1）

记 `W ∈ R^{m×n}`（embedding：`m`=词表、`n`=hidden），梯度 `G`，动量 `M`：

```
M  ← β·M + (1-β)·G                    # 动量
Ĝ  ← β·M + (1-β)·G                    # Nesterov 组合梯度
ρ_i = ‖Ĝ[i,:]‖₂,  ρ̄ = mean(ρ)          # 行范数
U⁰ = Ĝ；若 ρ_i ≤ τ·ρ̄ 则 U⁰[i,:] = 0     # 屏蔽近零行（防噪声被放大）
for k = 1..K:                          # K 次交替（K=11，奇数）
    k 奇 → U[i,:] /= (‖U[i,:]‖₂ + ε)    # 行归一化
    k 偶 → U[:,j] /= (‖U[:,j]‖₂ + ε)    # 列归一化
Δ  = √n · Uᴷ                           # 单位行 L2 → 单位行 RMS
η̃  = γ·η                               # γ=0.18，对齐 Adam 的更新幅度
W  ← W - η̃·Δ                          # **不施加权重衰减**
```

- `√n` 那一步是把"单位 L2 范数的行"换算成"单位 RMS 的行"（`‖x‖₂=1` 的行，
  其 RMS 是 `1/√n`，所以乘回 `√n`）；
- **τ 不是可有可无的**：`τ=0`（不屏蔽）时，一行 `1e-9` 的噪声会被归一化放大到
  ~O(1)，等于把噪声抬成信号（单测里有这条对照）；
- 论文超参：`K=11`、`τ=1e-3`、`ε=1e-20`、`β=0.95`、`γ=0.18`、不做 weight decay。

## 我们的实现

- 文件：`src/llm/local/optim.py`
  - `sinkhorn_balance(g, steps=11, eps=1e-20, tau=1e-3)`：纯函数，输入 Nesterov
    动量矩阵，返回平衡后的更新（fp32 内算，行/列 `clamp_min(eps)` 防 0 除）；
  - `SinkhornAdamLike`：`torch.optim.Optimizer` 子类，只存 `momentum_buffer`
    一份状态；`eta = lr * lr_scale`。
- 分组（`build_hybrid_optimizer` / `CombinedOptimizer`）：
  - **Muon**：2D 隐藏权重（含 head-wise 子组，见 [muon](./muon.md)）；
  - **Sinkhorn**：token embedding、`lm_head`、Engram 表（参数名形如
    `layers.N.engram.tables.K.weight`）；
  - **AdamW**：norm / bias / 1D 参数。
- 开关：`--sinkhorn`（默认关；改训练动态，属"要 A/B 才开"）。`--sinkhorn-lr-scale`
  （γ=0.18）、`--sinkhorn-steps`（K=11）。
- 存盘：`state_dict()` 的 `kind` 会变成 `muon+headwise+sinkhorn+adamw` 之类；
  装载时分组与存盘不一致（开关改了）会**早失败**并提示删 `optim.pt`——权重照常
  复用，只丢动量（与既有 Muon/AdamW 不互通的策略一致）。

## 从想法到代码

```
痛点：embedding/头/Engram 表是 token×特征大矩阵——Adam 状态翻倍，且低频 token 整行不更新
  ↓
直觉：行/列交替归一化 = 两张排行榜互相校准，把每个 token/特征的更新幅度拉平
  ↓
原理：动量 + Nesterov → 屏蔽近零行 → K 次行/列交替 L2 归一化 → Δ=√n·Uᴷ → η̃=γη
  ↓
代码：
  build_hybrid_optimizer(sinkhorn=True) → 把 embed/lm_head/.tables. 分到 SinkhornAdamLike
  SinkhornAdamLike.step()  → buf=μbuf+(1-μ)g; nes=μbuf+(1-μ)g; Δ=sinkhorn_balance(nes)
                              p -= lr*γ*Δ      （不做 weight decay）
  state_dict()/load_state_dict() → kind 里带分组标记，开关不匹配早失败
```

## 代价与边界

- **只适用于 2D 矩阵**（`g.dim() != 2` 直接报错）：norm/bias 这类 1D 参数仍在 Adam。
- **不改权重衰减**：Sinkhorn 组不做 wd（论文口径），原本靠 wd 抑制的过拟合在这里
  要靠别的手段（dropout / 数据）。
- **γ 是经验系数**：它把 Sinkhorn 的更新幅度对齐到 Adam 量级，换模型规模要重调
  （论文 0.18；Moonlight 用 0.2）。
- **K 必须是奇数**：算法以行归一化开始、以行归一化结束（`K=11`）。偶数次会让
  行/列 RMS 的平衡停在半路（不是致命错，但和论文不一致）。
- 真机冒烟（base 12 层 / 词表 8192）：三组状态合计 487 MiB vs 全 AdamW 974 MiB
  （**省 50%**）；5 步 loss 11.86→4.24；分组 `muon 692 / head-wise 9 /
  sinkhorn 1 / adam 95`（tied embedding 只有一个矩阵进 Sinkhorn）。
- 单测：`tests/test_sinkhorn_muon.py`（平衡后行/列 RMS≈1、近零行屏蔽、
  `τ=0` 会放大噪声的对照、与朴素参考逐位一致、单步更新 = `-ηγΔ`、只有一份状态、
  全零梯度不产 NaN、分组路由、存盘往返与开关不匹配早失败、端到端 5 步 loss 下降）。
