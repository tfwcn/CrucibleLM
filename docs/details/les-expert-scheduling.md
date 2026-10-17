# LES 专家调度：预判下一步该搬谁

> 一句话：MoE 相邻步的专家命中高度相关，用近期命中窗口预测下一步热点专家，
> 提前搬上算力设备，把 offload 的搬运延迟藏到计算后面。

## 痛点

专家 offload（`cross-device-offload.md`）是"命中后才搬"，搬运延迟暴露在关键
路径上、命中率低时甚至净变慢。若能在算之前就知道"下一步大概用哪几个专家"，
就能把它们提前备好。

## 直觉

你读同一段文字时，大脑调用的"专家"是连续的——上一句用的知识，下一句大概率还用。
命中序列因此高度可预测。

## 原理

```
维护近期命中窗口（最近 history 步）
打分：某专家的分 = 各历史步命中它的 1/(age+1) 之和（越近越重）
predict(topn)：取分数最高的若干，排除已在热端的
observe(hits)：把本步真实命中滑入窗口
```

预测结果喂给 `ExpertStore.ensure(prefetch=True)` 做**预取**（提前搬运），只影响
"何时搬"，不改变数值。纯统计、零可训练参数，不污染 `state_dict`。

**关键约束一：预取只填空槽、绝不淘汰任何专家。** 两条理由：
- 淘汰工作集：预取会按 LRU 把"马上要用的专家"挤回冷端，凭空多出搬运（实测
  16 专家、`capacity=8` 时搬运数翻 3 倍、吞吐减半）；
- 回收"预取未用"：看似合理，实则形成"预测→搬入→没用→回收→再预测"的 churn，
  容量越宽裕越烈（真实 128M、`capacity=12` 实测搬运 173→9382、吞吐 1.64x→2.28x）。

故容量满时预取**直接跳过**，淘汰权只留给"真正要用专家"的那一步。

**关键约束二：预取不更新 `_hot`。** 预取命中的专家只进 `_resident`（物理驻留集），
不进 `_hot`（工作集 LRU 序），因此不改变工作集的"最近使用"时序。
容量按 `_resident` 计（防止预取把显存撑爆）；`_resident` 命中即跳过重复搬运。

## 异步搬运（cpu target + CUDA）

预取要真正省时间，必须让搬运与当前步计算**重叠**。本版实现：

- 冷专家主副本保留为 **pinned CPU 张量**（`_cold`），专家模块置 `meta`；
- 预取时在**独立 CUDA stream** 上从 pinned 源做 H2D 拷贝（真异步），并记录
  `torch.cuda.Event`；
- 正常 `ensure` 命中同一专家时 `current_stream().wait_event(ev)` 等拷贝完成再用
  ——保证数值正确（不会读到半截权重）。
- `disk` 目标仍同步（磁盘 I/O 异步化需额外线程，本版不做）。

**全驻留快路径**：capacity≤0（声明"不淘汰"）时初始化即全量搬入，forward 稳定退回
分组 bmm 快路径（避免"部分专家从未命中、永远 meta"导致逐专家 dispatch）。

## 我们的实现

- 文件：`les.py`（`LesPredictor`）、`moe.py`（`enable_les` + 前向里的
  `predict`/`observe` 挂点）、`model.py`（`setup_offload` 里按配置挂载）；
- 配置：`les_enabled`（默认关）、`les_history`、`les_topn`；
- 无 offload 时开了也无副作用（预测没人消费）；预取与真实搬运都幂等，输出不变。

### 实测（真实 128M、预训练 base、12 层 16 专家 top_k=4、RTX 3080 Laptop、64 token）

`scripts/bench_offload_les.py --preset base --model data/llm-ckpt/model`（repeat=3，取最小值）：

| 容量 | 分组基线 | 同步 offload | +LES 异步预取 | moves（同步→LES） |
|---|---|---|---|---|
| 0（不限） | 22.7 tok/s | 0.98x | 0.94x | 192→192（全驻留→快路径） |
| 4（=top_k） | 23.9 tok/s | 1.66x | 1.77x | 1540→1596 |
| 8 | 24.8 tok/s | 1.77x | 1.75x | 688→744 |
| 12 | 24.6 tok/s | 1.58x | 1.67x | 173→218 |

（disk target、capacity=8：1.95x；数值上所有 offload 变体生成均与基线**逐位一致**。）

> 早期版本允许预取"回收预取未用空位继续预取"，在容量=12 时形成
> "预测→搬入→没用→回收→再预测"的 churn，搬运 173→9382、吞吐恶化到 2.28x。
> 改为**预取只填空槽、绝不淘汰任何专家**后，搬运 173→218，churn 消失。

**结论：异步预取在真实 128M 上中性偏负（略慢 0.05–0.11x）。** 原因：
1. **搬运本就很小**：单专家 ~1.8MB，H2D 拷贝在 3080 上极快；offload 的真正代价是
   "逐专家 Python dispatch"（分组 bmm 用 3 次 bmm 替代 16 次逐专家），异步重叠救不了；
2. **预取引入额外搬运**：容量有空槽时预取会搬入"预测但没用"的专家，占位后被真实
   命中回收 → moves 略增 → 略慢。

要变净收益需要：**(a) 路由可预测且稳定**（训练后命中集中）+ **(b) offload 是
"省显存才能跑"的刚需**（专家池大到分组路径放不下），而非"能跑却搬到 CPU"。
当前 128M 全量 512MB 显存绰绰有余，offload 是能力演示而非性能手段。

### 实测（tiny 对照）

tiny 配置（8 专家 top_k=2）同样中性偏负：容量 2/6 时预取让 1.18x→1.35x、1.11x→1.25x。
两套配置结论一致：**默认关，异步预取不解决根本问题**。

> 旧结论（"预取只填空位、永不劣化"）只对**同步**预取成立（预取跳过即零开销）；
> 一旦允许预取回收"预取未用"空位并异步搬运，多搬的专家会带来开销，故必须结合
> 预测质量评估，不能默认开。

## 从想法到代码

```python
# moe.py 命中路径开头：先按预测预取（只填空位/回收未用），再算真实命中
if self.prefetch and self.les is not None:
    self.store.ensure(self.les.predict(self.store.hot_ids()), prefetch=True)
...
if self.les is not None:
    self.les.observe(hit.tolist())                              # 记录本步命中

# offload.py 预取（async_=True 时在 side stream 拷贝并记录 event）
if prefetch:
    for i in todo:
        if i in self._resident:                                 # 已在设备上：跳过
            continue
        if self.capacity > 0 and len(self._resident) >= self.capacity \
                and not self._evict_victim({i}, unused_only=True):
            break                                               # 无未用可回收：不动工作集
        self._load(i, async_=True)                              # pinned→GPU，记 event

# 正常 ensure 命中预取来的专家：等 event 再算（保证不读半截权重）
if i in self._resident:
    self._await_ready(i)
    self._hot[i] = None                                         # 进工作集
```

## 代价与坑

- 预测错了只是白占一个 `_resident` 名额（会在下次腾位时被优先回收），数值正确性
  与 offload 解耦；
- **预取绝不淘汰工作集**是硬约束：否则会挤掉工作集，搬运数暴涨、吞吐倒退
  （见 pitfalls 第 16 条）；
- **异步预取要用 pinned 源**：非 pinned 的 H2D 拷贝会阻塞 host，退化成同步；
  `_cold` 主副本一律 `.pin_memory()`；
- **event 等待不能省**：side stream 上的拷贝与默认流无序，必须
  `current_stream().wait_event(ev)` 后再用模块参数，否则读到未写完的权重；
- **容量按 `_resident` 计**（不是 `_hot`）：否则预取会绕过容量上限，显存爆掉；
- **不重复搬运**：`_resident` 命中即跳过 `_load`（否则预取来的、又正常 ensure 的
  专家会被搬两次）；
- 目前只按"近期频次"打分，未用专家转移矩阵；若命中序列更结构化可再升级，
  但**保持零参数**是底线（否则 checkpoint 不再通用）；
- 随机（未训练）模型上预取中性偏负（预测不准 + 搬运太小），**默认关**；要用需先
  确认训练后路由可预测（见上表）。

## 一句话总结

用历史命中预测下一步热点，提前在独立 stream 上把专家搬好——预测只影响时序，
不影响结果；是否净赚取决于路由可预测性与专家规模，默认关。
