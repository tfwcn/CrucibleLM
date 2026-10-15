"""专家权重跨设备搬运 — 冷专家驻 CPU/磁盘，热点专家按需上算力设备（LRU）.

动机：细粒度 MoE 的总参数远大于每 token 激活参数（默认 16 专家只激活 top-4）。
把**未被本批路由命中**的专家权重留在 CPU 内存或磁盘，只把命中的少数专家
搬到算力设备，即可在显存受限时仍跑大专家池。

约束（与 AGENTS 一致）：
- 默认关；关时 MoE 走原有分组 bmm 快路径，逐位不变（单测锁定）。
- offload 路径只在"命中专家"上做逐专家前向，数值与朴素全量实现一致。
- 仅推理路径使用；不做可微搬运（专家权重 to(device) 是原地替换 param.data）。

异步预取（LES 加速）：
- 当 target="cpu" 且 compute_device 为 CUDA 时，启用 side stream 异步搬运。
- 冷专家权重保留 pinned CPU 副本（_cold），_load 在 side stream 上从 pinned
  源拷贝到 GPU（真正异步）；ensure 时通过 event 等待拷贝完成。
- target="disk" 保持同步（磁盘 I/O 异步化需额外线程，暂不做）。
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch


class ExpertStore:
    """按专家号管理"冷副本"（CPU/磁盘）与"热副本"（算力设备）的搬运与淘汰.

    - target="cpu"：冷专家物理驻留 CPU 内存（pinned，支持异步 H2D）；
    - target="disk"：冷专家 state_dict 落盘，参数置 meta（真正释放内存），
      命中时 `to_empty` 分配再 `load_state_dict` 读回（同步）；
    - capacity>0：热副本数上限，超出按 LRU 淘汰回冷端；<=0 不淘汰。
    - 异步预取：cpu target + CUDA 时，_load 在 side stream 执行，ensure 通过 event 等待。
    """

    def __init__(
        self,
        experts,
        compute_device,
        target: str = "cpu",
        offload_dir: str = "",
        capacity: int = 0,
        name: str = "expert",
    ):
        if target not in ("cpu", "disk"):
            raise ValueError(f"offload_target 须为 cpu/disk，当前 {target}")
        if target == "disk" and not offload_dir:
            raise ValueError("offload_target=disk 时必须给 offload_dir")
        self.experts = list(experts)
        self.compute_device = torch.device(compute_device)
        self.cpu_device = torch.device("cpu")
        self.target = target
        self.offload_dir = offload_dir
        self.capacity = int(capacity)
        self.name = name

        # 异步预取支持：side stream + event 追踪
        self._stream = None
        self._ready_events: dict[int, torch.cuda.Event] = {}  # 专家号 -> 拷贝完成事件
        self._cold: dict[int, dict[str, torch.Tensor]] = {}  # 专家号 -> pinned CPU 主副本（仅 cpu target）
        if target == "cpu" and self.compute_device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.compute_device)

        # LRU：键为专家号，值占位（顺序即最近使用顺序）
        self._hot: "OrderedDict[int, None]" = OrderedDict()
        self.moves = 0  # 搬运计数（日志/测试用）

        if target == "disk":
            os.makedirs(offload_dir, exist_ok=True)
            for i, e in enumerate(self.experts):
                torch.save(e.state_dict(), self._disk_path(i))
                self._to_meta(e)
        else:
            # cpu target：保留 pinned CPU 主副本，专家模块初始化为 meta（不占 GPU 内存）
            for i, e in enumerate(self.experts):
                # 保存 pinned CPU 主副本
                state = e.state_dict()
                pinned_state = {k: v.pin_memory() if v.is_cuda else v.cpu().pin_memory()
                                for k, v in state.items()}
                self._cold[i] = pinned_state
                self._to_meta(e)

    def _disk_path(self, i: int) -> str:
        """单个专家的落盘路径（前缀含 store 名，避免多层共用目录互相覆盖）."""
        return os.path.join(self.offload_dir, f"{self.name}_{i}.pt")

    def _to_meta(self, module) -> None:
        """清空专家参数到 meta（不占内存；读取前须 to_empty 分配）."""
        module.to(torch.device("meta"))

    def _load(self, i: int, async_: bool = False) -> None:
        """把专家 i 从冷端搬到算力设备（不更新 LRU，由 ensure 负责）.

        async_=True 时：在 side stream 上异步拷贝，记录 event；
        async_=False 时：同步拷贝（ensure 正常路径）。
        """
        e = self.experts[i]
        if self.target == "disk":
            # disk 目标保持同步（磁盘 I/O 异步化需额外线程，暂不做）
            e.to_empty(device=self.compute_device)
            e.load_state_dict(
                torch.load(self._disk_path(i), map_location=self.compute_device))
            self.moves += 1
            return

        # cpu 目标：从 pinned CPU 主副本拷贝到 GPU
        if async_ and self._stream is not None:
            # 异步拷贝：在 side stream 上执行
            with torch.cuda.stream(self._stream):
                e.to_empty(device=self.compute_device)
                # 从 pinned 主副本 load_state_dict（H2D 拷贝在 side stream 上）
                e.load_state_dict(self._cold[i])
                # 记录事件：拷贝完成
                ev = torch.cuda.Event()
                ev.record(self._stream)
                self._ready_events[i] = ev
            self.moves += 1
        else:
            # 同步拷贝
            e.to_empty(device=self.compute_device)
            e.load_state_dict(self._cold[i])
            self.moves += 1

    def _evict(self, i: int) -> None:
        """把专家 i 从算力设备退回冷端."""
        e = self.experts[i]
        if self.target == "disk":
            torch.save(e.state_dict(), self._disk_path(i))
            self._to_meta(e)
        else:
            # cpu target：拷贝回 pinned 主副本（推理阶段权重不变，可 skip；但为通用性保留）
            # 实际上推理时权重不变，evict 只需 meta 即可，无需拷贝回 _cold
            self._to_meta(e)
        self.moves += 1

    def ensure(self, ids, prefetch: bool = False) -> None:
        """保证列出的专家当前在算力设备上（命中即搬到热端，必要时 LRU 淘汰）.

        `prefetch=True` 用于 LES 预取：**只填空闲容量、绝不淘汰**。预取是为了
        把搬运延迟藏到计算前，若它反而把"马上要用的专家"挤回冷端，就会凭空
        多出搬运（实测预取+淘汰使 moves 翻 3 倍、吞吐减半）。故预取在容量已满时
        直接跳过，把淘汰权完全留给真正要用专家的那一步。

        异步预取（cpu+GPU）：prefetch 时在 side stream 启动拷贝并记录 event；
        正常 ensure 时等待 event 完成（若存在）。

        注意：prefetch 只加载专家到算力设备，**不更新 LRU 顺序**（不加入 `_hot`），
        避免 prefetch 本身改变命中专家的"最近使用"时序。
        """
        for i in ids:
            i = int(i)
            if i in self._hot:
                self._hot.move_to_end(i)
                continue
            if prefetch and self.capacity > 0 and len(self._hot) >= self.capacity:
                continue  # 预取不淘汰：无空位就不搬
            # 若 prefetched 且有 event，等待拷贝完成
            if i in self._ready_events:
                if self._stream is not None:
                    torch.cuda.current_stream().wait_event(self._ready_events[i])
                del self._ready_events[i]
            self._load(i, async_=prefetch)
            # prefetch 不更新 _hot（不改变 LRU 时序）；正常 ensure 才更新
            if not prefetch:
                self._hot[i] = None
            if self.capacity > 0:
                while len(self._hot) > self.capacity:
                    old, _ = self._hot.popitem(last=False)
                    self._evict(old)

    def hot_ids(self) -> list[int]:
        """当前驻留算力设备的专家号（最近使用序）."""
        return list(self._hot.keys())

    def stats(self) -> dict:
        """搬运统计（日志用）."""
        return {"hot": len(self._hot), "moves": self.moves,
                "target": self.target, "capacity": self.capacity}
