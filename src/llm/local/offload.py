"""专家权重跨设备搬运 — 冷专家驻 CPU/磁盘，热点专家按需上算力设备（LRU）.

动机：细粒度 MoE 的总参数远大于每 token 激活参数（默认 16 专家只激活 top-4）。
把**未被本批路由命中**的专家权重留在 CPU 内存或磁盘，只把命中的少数专家
搬到算力设备，即可在显存受限时仍跑大专家池。

约束（与 AGENTS 一致）：
- 默认关；关时 MoE 走原有分组 bmm 快路径，逐位不变（单测锁定）。
- offload 路径只在"命中专家"上做逐专家前向，数值与朴素全量实现一致。
- 仅推理路径使用；不做可微搬运（专家权重 to(device) 是原地替换 param.data）。
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch


class ExpertStore:
    """按专家号管理"冷副本"（CPU/磁盘）与"热副本"（算力设备）的搬运与淘汰.

    - target="cpu"：冷专家物理驻留 CPU 内存；
    - target="disk"：冷专家 state_dict 落盘，参数置 meta（真正释放内存），
      命中时 `to_empty` 分配再 `load_state_dict` 读回；
    - capacity>0：热副本数上限，超出按 LRU 淘汰回冷端；<=0 不淘汰。
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
        # 落盘文件前缀：**必须每个 store 唯一**——多层共用一个 offload_dir 时，
        # 若都用 expert_{i}.pt 会互相覆盖，后建层静默顶掉先建层的专家权重
        # （实测整模偏差 0.15，教训见 docs/pitfalls/local-llm-training.md 15）
        self.name = name
        # LRU：键为专家号，值占位（顺序即最近使用顺序）
        self._hot: "OrderedDict[int, None]" = OrderedDict()
        self.moves = 0  # 搬运计数（日志/测试用）
        if target == "disk":
            os.makedirs(offload_dir, exist_ok=True)
            for i, e in enumerate(self.experts):
                torch.save(e.state_dict(), self._disk_path(i))
                self._to_meta(e)
        else:
            for e in self.experts:
                e.to(self.cpu_device)

    def _disk_path(self, i: int) -> str:
        """单个专家的落盘路径（前缀含 store 名，避免多层共用目录互相覆盖）."""
        return os.path.join(self.offload_dir, f"{self.name}_{i}.pt")

    def _to_meta(self, module) -> None:
        """清空专家参数到 meta（不占内存；读取前须 to_empty 分配）."""
        module.to(torch.device("meta"))

    def _load(self, i: int) -> None:
        """把专家 i 从冷端搬到算力设备（不更新 LRU，由 ensure 负责）."""
        e = self.experts[i]
        if self.target == "disk":
            e.to_empty(device=self.compute_device)
            e.load_state_dict(
                torch.load(self._disk_path(i), map_location=self.compute_device))
        else:
            e.to(self.compute_device)
        self.moves += 1

    def _evict(self, i: int) -> None:
        """把专家 i 从算力设备退回冷端."""
        e = self.experts[i]
        if self.target == "disk":
            torch.save(e.state_dict(), self._disk_path(i))
            self._to_meta(e)
        else:
            e.to(self.cpu_device)
        self.moves += 1

    def ensure(self, ids) -> None:
        """保证列出的专家当前在算力设备上（命中即搬到热端，必要时 LRU 淘汰）."""
        for i in ids:
            i = int(i)
            if i in self._hot:
                self._hot.move_to_end(i)
                continue
            self._load(i)
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
