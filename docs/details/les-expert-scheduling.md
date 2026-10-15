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

预测结果喂给 `ExpertStore.ensure` 做**预取**（提前搬运），只影响"何时搬"，
不改变数值。纯统计、零可训练参数，不污染 `state_dict`。

## 我们的实现

- 文件：`les.py`（`LesPredictor`）、`moe.py`（`enable_les` + 前向里的
  `predict`/`observe` 挂点）、`model.py`（`setup_offload` 里按配置挂载）；
- 配置：`les_enabled`（默认关）、`les_history`、`les_topn`；
- 无 offload 时开了也无副作用（预测没人消费）；预取与真实搬运都幂等，输出不变。

## 从想法到代码

```python
# moe.py 命中路径开头：先按预测预取，再算真实命中
if self.prefetch and self.les is not None:
    self.store.ensure(self.les.predict(self.store.hot_ids()))   # 提前搬
...
if self.les is not None:
    self.les.observe(hit.tolist())                              # 记录本步命中
```

## 代价与坑

- 预测错了只是白搬一次（LRU 会自然换出），不会算错——数值正确性与 offload 解耦；
- 目前只按"近期频次"打分，未用专家转移矩阵；若命中序列更结构化可再升级，
  但**保持零参数**是底线（否则 checkpoint 不再通用）；
- `les_topn` 与 offload 容量要配合：topn 超过容量会互相挤兑，均摊反而变差。

## 一句话总结

用历史命中预测下一步热点，提前把专家搬好——预测只影响时序，不影响结果。
