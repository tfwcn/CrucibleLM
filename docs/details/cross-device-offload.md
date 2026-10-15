# 跨设备推理：专家 offload 到 CPU/磁盘

> 一句话：把未被本批路由命中的 MoE 专家权重留在 CPU 内存或磁盘，只把命中的
> 少数专家搬上算力设备（LRU），显存受限时也能跑大专家池。

## 痛点

细粒度 MoE 的总参数远大于每 token 激活参数（默认 16 专家只激活 top-4）。
显存按"总参数"而非"激活参数"占用，小卡上专家池一大就塞不下；但每一步真正
用到的只有少数专家。

## 直觉

图书馆不必把每本书都摆在你桌上——只把你正要看的那几本取过来，看完放回架上。
专家命中随文本高度聚集，所以"按需搬运"能省下大量常驻显存。

## 原理

```
冷端（CPU 内存 / 磁盘）：未被命中的专家权重
热端（算力设备）：当前命中专家，容量上限按 LRU 淘汰
命中 → 搬上热端 → 逐专家算完散射加回 → 换出最久未用的
```

`cpu` 目标：冷专家物理驻 CPU；`disk` 目标：冷专家 state_dict 落盘、参数置 meta
（真释放内存），命中时 `to_empty` 分配再 `load_state_dict` 读回。

## 我们的实现

- 文件：`offload.py`（`ExpertStore`：搬运 + LRU）、`moe.py`（`enable_offload`
  与 `_forward_offload` 命中路径）、`model.py`（`setup_offload`）；
- 配置：`offload_enabled`（默认关）、`offload_target`、`offload_dir`、
  `offload_capacity`、`offload_prefetch`；
- 默认关时 MoE 仍走分组 bmm 快路径（逐位不变）；开启时才切换到"只算命中专家"
  的逐专家路径，数值与朴素全量实现一致（单测锁定）；
- `LocalChatBackend.load` 会按配置自动 `setup_offload()`。

## 从想法到代码

```python
# moe.py：开启后走命中路径（逐个 ensure，不能用容量<命中数的批量 ensure）
if self.store is not None:
    return self._forward_offload(x)
...
for e in torch.unique(top_i).tolist():
    self.store.ensure([e])                 # 命中专家搬上算力设备（LRU）
    out = out.index_add(0, idx, w_e.unsqueeze(-1) * self.experts[e](flat[sel]))
```

## 代价与坑

- **必须逐个 ensure**：若先整批 ensure 再统一计算，`capacity < 命中数` 时后搬入
  的专家会把先搬入的挤回冷端，随后用到已置 meta 的专家 → 结果错（真踩过，
  单测 `capacity=2/3` 覆盖）；
- 搬运本身有开销，命中率低时反而更慢——配合 LES 预取（见 `les-expert-scheduling.md`）；
- `disk` 目标要给定目录；读回用 `to_empty` + `load_state_dict`（meta 参数不能
  直接 `copy_`）；
- 仅推理路径；训练不要开（会破坏优化器对参数对象的持有）。

## 一句话总结

权重按"是否被本批命中"在 CPU/磁盘与算力设备间搬运，用 LRU 控制常驻上限——
显存换时间，默认关。
