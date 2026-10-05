"""会话增量状态 — turn 间隙落盘，下 turn 读回续跑，200K 会话内存归零.

原理：MLA 的 latent 缓存 + 线性层的常数状态（含卷积尾）就是会话的
"长期记忆"，本来只活在生成函数的局部变量里。这里把它显式化：
extend() 喂一段、past 常驻，需要时 torch.save 落盘（几百 MB），
下 turn 读回继续增量解码，不再全量 prefill。

只动推理路径（no_grad，不碰 train/eval 状态）；训练循环不用它。
"""

from __future__ import annotations

import torch


class SessionCache:
    """会话缓存：past 列表 + 已见 id，下 turn 增量续跑."""

    def __init__(self, model):
        self.model = model
        self.pasts: list | None = None
        self.ids: torch.Tensor | None = None

    def __len__(self) -> int:
        """已缓存 token 数."""
        return int(self.ids.shape[1]) if self.ids is not None else 0

    @torch.no_grad()
    def extend(self, input_ids: torch.Tensor) -> torch.Tensor:
        """喂一段、更新缓存，返回末位置 hidden（供直觉头/采样用）.

        首段走快速并行 prefill；后续按单 token 步进（复用模型 _decode_step，
        与 generate 解码路径数学一致，单测锁定）。不碰 train/eval 状态。
        """
        dev = self._device()
        ids = input_ids.to(dev)
        if self.pasts is None:
            h, self.pasts = self.model._prefill(ids)
        else:
            h = None
            for t in range(ids.shape[1]):
                h, self.pasts = self.model._decode_step(ids[:, t:t + 1], self.pasts)
        cur = ids.to("cpu")
        self.ids = cur if self.ids is None else torch.cat([self.ids, cur], dim=1)
        assert h is not None
        # 统一返回末位置向量 (b, d)：prefill 分支 h 为全序列，decode 分支为单步
        return h[:, -1, :]

    def _device(self) -> torch.device:
        """模型所在设备."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def save(self, path: str) -> None:
        """落盘（past 全是小张量：latent/状态/卷积尾，无 O(N^2) 大物）."""
        assert self.pasts is not None, "空缓存无需存盘"
        torch.save({"pasts": self.pasts,
                    "ids": self.ids}, path)

    @classmethod
    def load(cls, path: str, model) -> "SessionCache":
        """读回（map 到模型所在设备）."""
        sc = cls(model)
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        blob = torch.load(path, map_location=dev)
        sc.pasts = blob["pasts"]
        sc.ids = blob["ids"]
        return sc

    def clear(self) -> None:
        """清空（会话结束/超长截断时调用，调用方另行处理）."""
        self.pasts = None
        self.ids = None
