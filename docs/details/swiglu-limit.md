# SwiGLU 限幅：给激活套个笼头

> 一句话：在 `silu(gate) * up` 之前把 gate/up 对称截断到 `±swiglu_limit`，
> 压住偶发的激活离群值——DeepSeek-V4.1 的 `swiglu_limit=10.0`。

## 痛点

SwiGLU 是 `silu(gate) * up`，两个分支都是无界线性投影。训练久了，
个别神经元的预激活会飙到几十上百（outlier），一旦乘在一起就是平方级放大，
把整条残差流带偏：既让 bf16 动态范围吃紧（精度损失、甚至 NaN），
又让少数维度主导上层读取，压制其余特征。

## 直觉

像给水管装减压阀：正常水压（|z| ≤ 10）完全不动，只在**爆表时**削顶。
大部分时候这个阀是"死"的——不改变日常水流，只在异常时刻兜底。

## 原理

对 gate/up 的**预激活**做对称限幅，再进激活与相乘：

```
gate = clamp(W_gate · x, -L, L)
up   = clamp(W_up   · x, -L, L)
out  = W_down · (silu(gate) * up)
```

- `L = swiglu_limit`（V4.1 取 `10.0`）；`L <= 0` 表示关闭，退回 `silu(W_gate·x) * (W_up·x)`；
- **只压幅值、不改符号**（`clamp` 保号）——所以对"双负区间"这类按符号划分的
  分析（见《MoE 可解释性》）没有影响，只影响极端幅值；
- 只作用于**专家内部**，不碰 router、不碰残差流，也不引入任何参数。

> 说明：V4.1 技术报告只给了 `swiglu_limit` 这一个字段，没有展开它 clamp 的是 gate、
> up 还是二者、是否对称。本实现取**对 gate 与 up 都做对称限幅**这一最保守口径；
> 若日后拿到更精确语义，只改 `SwiGLUExpert.gate_up` 一处即可。

## 我们的实现

- 文件：`src/llm/local/moe.py`
  - `SwiGLUExpert(d_model, hidden, limit)`：存 `self.limit`（纯超参，**不入 state_dict**）；
  - `SwiGLUExpert.gate_up(x)`：返回限幅后的 `(gate, up)`，前向与离线分析共用，
    保证"模型算的"和"分析看到的"是同一口径；
  - `FineGrainedMoE(..., swiglu_limit)`：分组 bmm 快路径与逐专家/offload 路径都按同一阈值截断。
- 配置：`SmallLLMConfig.swiglu_limit`（默认 `10.0`=开，`0`=关）；CLI `--swiglu-limit`。
- 与模型/分析接线：`model.py:build_block` 透传；`interpret.py`、`rejuvenate.py`
  改走 `expert.gate_up()`，统计与前向一致。
- **来源对齐**：这是"改前向语义、不改 state_dict 键"的开关，装载既有权重时必须
  按来源对齐（同 CSA2）。`config.sync_arch_from_checkpoint` 会把旧 run（hparams 无此
  字段）回退到 `0`（不限幅），逐位保持旧行为；训练脚本 `resolve_swiglu_limit`
  在续训/迁移读不到架构记录时同样保守按 `0`。

## 从想法到代码

"减压阀"= 预激活限幅；键在"预激活 vs 激活之后"这个位置：

```python
def gate_up(self, x):                      # 与模型前向同口径
    g, u = self.w_gate(x), self.w_up(x)
    if self.limit > 0:                     # 0 = 关闭，逐位等价旧实现
        g, u = g.clamp(-L, L), u.clamp(-L, L)
    return g, u

def forward(self, x):
    g, u = self.gate_up(x)
    return self.w_down(F.silu(g) * u)      # 截断在前，silu/相乘在后
```

分组快路径在 `canvas @ w_g` 之后以同一阈值 `clamp`，与逐专家版数值一致（单测锁定）。

## 代价与坑

- **几乎零代价**：`L=10` 时正常量级的预激活（|z| ≲ 几）根本不触发，等于不放；
  只有离群值被削。所以它不拖慢、不改变可复现性（同种子同结果）。
- **它确实改前向数字**（当且仅当 |z|>L）：因此**默认值升级不能静默套到旧 run 上**——
  必须靠 `hparams.json` 来源对齐（见上）。这是本开关唯一需要小心的点。
- 别把限幅错加到**激活之后**（`silu` 输出本就非负有界，限幅无意义）或 router 上
  （那会改变路由，属于另一回事）。
- 阈值太小（如 1）会真的削掉正常信号、明显掉点——它是安全阀，不是正则化主力。

## 一句话总结

在 SwiGLU 进门处放一个 ±10 的减压阀：平常不动、爆表兜底，只压幅值不改符号，
零参数、零额外开销，唯一纪律是"旧 run 靠 hparams 认清自己有没有这个阀"。
