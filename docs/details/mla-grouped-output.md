# MLA 分组输出投影：把"最后一哆嗦"也压成低秩

> 一句话：注意力头算完拼成一大条，别用一个巨型 `w_o` 直接拍回 `d_model`——先按头分组、
> 每组压到 `o_lora_rank` 的小瓶颈，再统一升维。DeepSeek-V4.1：`o_groups=8`、
> 每组中间维 `o_lora_rank=1024`。

## 痛点

MLA 已经把 KV 压得很狠，但**输出投影 `w_o` 一直是全尺寸**：输入是
`n_heads × head_dim`（V4.1 是 64×512 = 32768），输出 `d_model`。这一个矩阵就占了
注意力层参数的大头，且各头之间没有任何结构约束——纯靠训练学出 64 个头的组合关系。

## 直觉

64 个头各说各话，最后让一个 `32768 × d_model` 的巨矩阵决定怎么汇总，又贵又难训。
改成"先分组、组内汇总、再跨组汇总"：

- 把 64 个头分成 8 组（每组 8 个头），**组内**先压到一个 1024 维的中间表示；
- 8 个组的 1024 维拼成 8192 维，再升维回 `d_model`。

等于给"头 → 输出"之间插了一层按组共享的瓶颈，参数量大幅下降（32768×d_model →
32768×1024 + 8192×d_model），而且组内/组间的分工更清晰。

## 原理

设每组头数 `h_g = n_heads / o_groups`，每组输入维 `in_g = h_g × head_dim`：

```
组内降维：z_i = W_down_i · (组 i 的头输出)      # in_g → o_lora_rank
拼接升维：y   = W_up · [z_1; z_2; ...; z_G]     # G*o_lora_rank → d_model
```

注意 `W_up · [z_1;...;z_G] = Σ_i W_up[:, i 段] · z_i`，所以**数学上等价于
"每组各有一个 `o_lora_rank → d_model` 的升维，再把 G 个结果相加"**——把求和并进了
一次矩阵乘法，实现更省。整条路径仍是纯线性（无非线性），所以它不是"增加表达力"，
而是**同表达力下的参数压缩 + 分组归纳偏置**。

`o_lora_rank = in_g`（不降维）时，任何稠密 `w_o` 都能被精确表示——这是迁移无损的上界。

## 我们的实现

- 文件：`src/llm/local/mla.py`
  - `MLAModule.__init__`：`o_groups>1 且 o_lora_rank>0` 时建 `w_o_d`（`ModuleList`，每组
    一个 `in_g→o_lora_rank`）+ `w_o_u`（`G*o_lora_rank→d_model`），**不建 `w_o`**；
    否则只建稠密 `w_o`，键与数值与旧实现逐位一致。
  - `_out_proj(out)`：唯一输出投影入口（训练/prefill/跨层复用/增量解码四条路径共用），
    分组时 reshape → 逐组降维 → 拼接升维，关闭时退化为 `self.w_o(out)`。
  - `out_proj_weight()`：返回残差输出投影权重（分组时是 `w_o_u`），给按深度初始化
    （`model._init_weights`）与迁移置零（`migrate.add_identity_layers`）统一调用。
  - 稀疏路径 `sparse_attn.SparseMLAModule` 继承 `_out_proj`，四条 `self.w_o(out)` 全部改走它。
- 配置：`config.o_groups`（默认 1）、`config.o_lora_rank`（默认 0）——**默认关**。
- 迁移：`migrate.group_output_projection(model, G, r)`：对每层 `w_o` 按头分成 G 个列块，
  逐块做秩 r 的截断 SVD，写入 `w_o_d`/`w_o_u`；返回 (新模型, 平均相对重构误差)。
  只处理全注意力层，线性层原样保留。CLI：`scripts/migrate_model.py --o-groups G --o-lora-rank r`。
- 来源对齐：登记在 `config._ARCH_SWITCHES`（回退 1/0），续训自动沿用，旧 ckpt 回退到关。

## 从想法到代码

"分组降维再拼接升维"，关键是分组是**按头切**、升维是**一次矩阵乘法**：

```python
in_g = (n_heads // o_groups) * head_dim
g = out.view(b, t, o_groups, in_g)          # 按头分组
z = stack([w_o_d[i](g[:, :, i]) for i in range(o_groups)])  # 组内降维
y = w_o_u(z.flatten(2))                     # 拼接升维回 d_model
# 关闭：y = w_o(out)  （逐位等价旧实现）
```

迁移（把已有 `w_o` 拆成两组权重，近似但可控）：

```python
for 每组列块 block in w_o.split(in_g, dim=1):
    down, up, err = svd_split(block, r)     # block ≈ down @ up
    w_o_u[:, 该组段] = down;  w_o_d_g.weight = up
```

## 代价与坑

- **改 state_dict 键**（`w_o` → `w_o_d`/`w_o_u`）：这是本项与 1/2 号最大的不同——它不是
  "改语义不改键"，所以**默认关**（按约定 5，会破坏旧 run 的装载 = 有副作用）。开启需重训，
  或走 `migrate_model.py` 的 SVD 迁移（近似，附重构误差，迁移后建议短训恢复）。
- **纯线性，不增表达力**：`r < in_g` 时是**近似**（截断能量 = 重构误差）；想无损就取
  `r = in_g`（但那样就没省参数的意义）。本机极小配置（hidden 128）实测 r=32 时迁移重构误差
  约 0.21（各组平均），说明 `w_o` 的能量并不集中在少数奇异值上，压缩有代价——这也是
  "迁移后要短训"的原因。
- **整除约束**：`n_heads` 必须能被 `o_groups` 整除，`0 < o_lora_rank ≤ in_g`，构造时断言，
  早失败早发现。
- **续训自动沿用**：虽然改键，但仍登记进 `_ARCH_SWITCHES`——好处是续训/装载脚本能按
  hparams 自动重建正确结构；若 checkpoint 与之不符，strict 载入会**大声报缺键**（不会静默），
  这也是"改键比改语义安全"的地方。
- **参数**：`estimate_memory_gb` 里 `w_o` 仍按 `d_model²` 估算（该函数是粗略预算，不做分组区分）。

## 一句话总结

输出投影从"一个巨型矩阵"改成"按头分组→组内瓶颈→拼接升维"，同表达力省参数、给路由式
分组归纳偏置；代价是改键（默认关）且低秩是近似——开启要么重训，要么用 SVD 迁移再短训。
