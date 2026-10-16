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
        self._ready_events: dict[int, torch.cuda.Event] = {}  # 预取中：拷贝完成事件
        self._cold: dict[int, dict[str, torch.Tensor]] = {}  # 专家号 -> pinned CPU 主副本（仅 cpu target）
        if target == "cpu" and self.compute_device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.compute_device)

        # _hot：被 ensure 真正命中的专家（LRU 序，队首最旧最易淘汰）；
        # _resident：物理驻留算力设备的全部专家（含预取但未用的），容量按它计
        self._hot: "OrderedDict[int, None]" = OrderedDict()
        self._resident: set[int] = set()
        self.moves = 0  # 搬运计数（日志/测试用）

        if target == "disk":
            os.makedirs(offload_dir, exist_ok=True)
            for i, e in enumerate(self.experts):
                torch.save(e.state_dict(), self._disk_path(i))
                self._to_meta(e)
        else:
            # cpu target：保留 pinned CPU 主副本，专家模块初始化为 meta（不占显存）；
            # 主副本一律落在 CPU（即便传入的专家当时在 GPU），否则 _cold 不构成"冷端"
            for i, e in enumerate(self.experts):
                state = e.state_dict()
                self._cold[i] = {
                    k: v.detach().to(self.cpu_device).pin_memory()
                    for k, v in state.items()
                }
                self._to_meta(e)

        # capacity<=0 = 声明"不淘汰"：反正迟早全驻留，直接在初始化时全量搬入，
        # 让 forward 稳定走分组 bmm 快路径（避免"部分专家从未命中、永远 meta"）
        if self.capacity <= 0:
            self.ensure(range(len(self.experts)))

    def _disk_path(self, i: int) -> str:
        """单个专家的落盘路径（前缀含 store 名，避免多层共用目录互相覆盖）."""
        return os.path.join(self.offload_dir, f"{self.name}_{i}.pt")

    def _to_meta(self, module) -> None:
        """清空专家参数到 meta（不占内存；读取前须 to_empty 分配）."""
        module.to(torch.device("meta"))

    def _await_ready(self, i: int) -> None:
        """若专家 i 正在 side stream 上异步搬运，等待其完成并清除事件."""
        ev = self._ready_events.pop(i, None)
        if ev is not None and self._stream is not None:
            torch.cuda.current_stream().wait_event(ev)

    def _load(self, i: int, async_: bool = False) -> None:
        """把专家 i 从冷端搬到算力设备（不更新 LRU，由 ensure 负责）.

        async_=True 时：在 side stream 上从 pinned CPU 主副本异步拷贝、记录事件；
        async_=False 时：同步拷贝（ensure 正常路径）。
        """
        e = self.experts[i]
        if self.target == "disk":
            # disk 目标保持同步（磁盘 I/O 异步化需额外线程，暂不做）
            e.to_empty(device=self.compute_device)
            e.load_state_dict(
                torch.load(self._disk_path(i), map_location=self.compute_device))
        elif async_ and self._stream is not None:
            # 异步拷贝：在 side stream 上执行（pinned 源 → 真正异步 H2D）
            with torch.cuda.stream(self._stream):
                e.to_empty(device=self.compute_device)
                e.load_state_dict(self._cold[i])
                ev = torch.cuda.Event()
                ev.record(self._stream)
                self._ready_events[i] = ev
        else:
            # 同步拷贝
            e.to_empty(device=self.compute_device)
            e.load_state_dict(self._cold[i])
        self._resident.add(i)
        self.moves += 1

    def _evict(self, i: int) -> None:
        """把专家 i 从算力设备退回冷端（若还在异步搬运，先等它落地再退）."""
        self._await_ready(i)
        e = self.experts[i]
        if self.target == "disk":
            torch.save(e.state_dict(), self._disk_path(i))
        # cpu/disk 均置 meta 释放显存；推理阶段权重不变，无需拷回 _cold
        self._to_meta(e)
        self._resident.discard(i)
        self._hot.pop(i, None)
        self.moves += 1

    def _evict_victim(self, protect: set[int], unused_only: bool = False) -> bool:
        """挑一个专家腾位，返回是否成功淘汰.

        - 优先"预取但未用"（`_resident−_hot`）——它们是投机搬运，可随时回收；
        - `unused_only=False`（正常 ensure）时若无可回收的，再退 LRU 工作集；
        - `unused_only=True`（预取）时**绝不退工作集**，返回 False 让调用方跳过。
        protect（当前正要加载的专家号）绝不淘汰。
        """
        for i in [r for r in self._resident if r not in self._hot and r not in protect]:
            self._evict(i)
            return True
        if unused_only:
            return False
        for i in list(self._hot.keys()):  # 队首 = 最久未用
            if i not in protect:
                self._evict(i)
                return True
        return False

    def ensure(self, ids, prefetch: bool = False) -> None:
        """保证列出的专家当前在算力设备上（命中即搬到热端，必要时 LRU 淘汰）.

        容量按 `_resident`（含预取）计，避免预取把显存撑爆。

        `prefetch=True`（LES 预取）：**只填空闲容量，绝不淘汰工作集**。预取是为
        把搬运延迟藏到计算前，若它反而把"马上要用的专家"挤回冷端，就会凭空多出
        搬运（实测预取+淘汰使 moves 翻 3 倍、吞吐减半）。故：
          - 容量满时有"预取未用"的可回收，回收之；
          - 没有可回收的（工作集占满），直接跳过，淘汰权留给真正要用专家的那步；
          - 预取**不更新 `_hot`**，不改变工作集的"最近使用"时序。

        异步预取（cpu+GPU）：prefetch 在 side stream 启动拷贝并记录 event；
        正常 ensure 命中同一专家时等待该 event 完成再使用（保证数值正确）。
        """
        todo = [int(i) for i in ids]
        if prefetch:
            for i in todo:
                if i in self._resident:
                    continue
                if self.capacity > 0:
                    if len(self._resident) >= self.capacity and \
                            not self._evict_victim({i}, unused_only=True):
                        break  # 无可回收空位：不淘汰工作集
                self._load(i, async_=True)
            return
        for i in todo:
            if i in self._resident:
                self._await_ready(i)  # 可能是预取来的，等拷贝完成
                if i in self._hot:
                    self._hot.move_to_end(i)
                else:
                    self._hot[i] = None
                continue
            if self.capacity > 0:
                while len(self._resident) >= self.capacity:
                    if not self._evict_victim({i}):
                        break
            self._load(i, async_=False)
            self._hot[i] = None

    def hot_ids(self) -> list[int]:
        """当前被 ensure 命中（工作集）的专家号，最近使用序."""
        return list(self._hot.keys())

    def all_resident(self) -> bool:
        """所有专家是否都已驻留算力设备（全驻留时可退回分组 bmm 快路径）."""
        return len(self._resident) >= len(self.experts)

    def stats(self) -> dict:
        """搬运统计（日志用）."""
        return {"hot": len(self._hot), "resident": len(self._resident),
                "moves": self.moves, "target": self.target,
                "capacity": self.capacity}
