# 跨词表蒸馏：向 0.5B 老师要答案

> 一句话：学生（8192 字符表）和老师（10 万级 BPE）词表对不上，就按"字符"对齐，只在双方都切得开的地方学。

## 痛点

标准 logit 蒸馏要求师生同词表。MiniCPM 是 BPE（" transformer" 一个 token），
我们是字符级（一字一 token），位置根本对不齐——蒸馏无从下手。
但 0.5B 老师的中文知识对 128M 是降维打击，不要白不要。

## 直觉

两个{度量衡}不同的人合伙砌墙：按"砖缝"（字符边界）对齐。老师一个 token
覆盖"AB"两个字，学生恰好在 A、B 处各有一个位置——那这两个位置就能学；
老师 token 横跨半个字的（BPE 经常这样），跳过不学。只学"对得上"的，
剩下的自己悟。

## 原理

```
老师 token k 覆盖字符区间 [a, b) → 学生位置 (b-1) 预测字符 b
配对处：KL(老师分布 || 学生分布)，双方只取"锚点"字符子集截断重归一
Loss = CE + kd_alpha·KL/T² + MTP（Hinton T² 还原梯度量级，按配对数平均）
```

- **锚点**：表面字符串一致的字符（单汉字大概率在老师侧独立成 token），
  建表一次（423/6110，启动日志里有数）；
- **温度 T**：软化分布，KL 乘 T² 补回梯度（Hinton 原配方）；
- **KD 降频**（`--kd-every 8`）：老师前向贵（0.5B），每 8 个 micro 跑 1 次，
  kd 项放大 8 倍保期望——期望一致，方差稍大，实测无妨。
  （离线缓存版被否了：省的是时间不是质量，降频更简单。）

## 我们的实现

- 文件：`src/llm/local/distill.py`（`AnchorDistiller`：`teacher_token_losses` +
  `batch_kl` + `select_by_excess`）
- 开关：`--kd-teacher data/teacher-0.5b --kd-alpha 0.5 --kd-every 8`
- MiniCPM 自带 modeling 引用了新版 transformers 已删的 FX 特性，
  loader 里打了兼容垫片 + `use_cache=False`（否则报格式错）——
  transformers 锁 4.54.1，别升级；
- RHO-teacher 复用同一路老师前向（`do_kd` 的 micro 才算，其余回落自参照）

## 从想法到代码

对齐即查表：老师 token 的字符区间 → 学生位置，锚点处截断做 KL：

```python
s_pos = pos_of_char[bnd - 1]          # 老师 token k 覆盖 [a,b) → 学生位置
kl = KL(softmax(t[t_anchor]/T) || log_softmax(s[s_anchor]/T)) * T² / n_pairs
# kd-every=8：每 8 micro 跑 1 次老师，kd 项 ×8 保期望
```

## 代价与坑

- 锚点只有 423 个字符（7%），93% 的位置老师帮不上——KD 是"家教"不是"代考"，
  主力还得是 CE。
- 老师前向哪怕降频也是额外开销；`kd-every` 太大（>16）方差会抖，
  8 是实测甜点。

## 一句话总结

词表不同按砖缝对齐，对得上就学、对不上跳过——0.5B 的家教费（1/8 前向）花得值。
