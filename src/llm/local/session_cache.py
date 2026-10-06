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
    def extend(self, input_ids: torch.Tensor, max_new_tokens: int = 2048,
               ) -> torch.Tensor:
        """喂一段、更新缓存，返回末位置 hidden（供直觉头/采样用）.

        首段走快速并行 prefill（顺带预留 max_new_tokens 静态缓存）；
        后续按单 token 步进（复用模型 _decode_step，与 generate 解码路径
        数学一致，单测锁定）。不碰 train/eval 状态。
        容量不够时自动 ensure_room 扩（2x，均摊可忽略）。
        """
        dev = self._device()
        ids = input_ids.to(dev)
        if self.pasts is None:
            h, self.pasts = self.model._prefill(ids, max_new_tokens=max_new_tokens)
        else:
            self.ensure_room(ids.shape[1])
            h = None
            for t in range(ids.shape[1]):
                h, self.pasts = self.model._decode_step(ids[:, t:t + 1], self.pasts)
        cur = ids.to("cpu")
        self.ids = cur if self.ids is None else torch.cat([self.ids, cur], dim=1)
        assert h is not None
        # 统一返回末位置向量 (b, d)：prefill 分支 h 为全序列，decode 分支为单步
        return h[:, -1, :]

    @staticmethod
    def _is_static_triple(p) -> bool:
        """MLA 系静态缓存三元组（buf_c, buf_kr, pos）判定.

        线性层 past 也是三元组 (S, k_buf, v_buf)，但第 3 元是 float 卷积尾；
        静态三元组的第 3 元恒为 long 标量 pos，以此区分。
        """
        return (isinstance(p, tuple) and len(p) == 3
                and isinstance(p[2], torch.Tensor)
                and p[2].dtype == torch.long and p[2].numel() == 1)

    @torch.no_grad()
    def ensure_room(self, need: int) -> None:
        """保证各层静态缓存剩余容量 ≥ need，不够按 2x 扩.

        eager 重分配（拷贝旧有效区）；编译过的 decode 重编一次，
        指数扩容下均摊可忽略。线性层状态定长，无需处理。
        """
        if self.pasts is None:
            return
        for li, p in enumerate(self.pasts):
            if not self._is_static_triple(p):
                continue
            buf_c, buf_kr, pos = p
            cur = int(pos.item())
            if cur + need <= buf_c.shape[1]:
                continue
            nlen = max(buf_c.shape[1] * 2, cur + need)
            new_c = torch.zeros(buf_c.shape[0], nlen, buf_c.shape[2],
                                device=buf_c.device, dtype=buf_c.dtype)
            new_c[:, :cur, :] = buf_c[:, :cur, :]
            new_kr = torch.zeros(buf_kr.shape[0], nlen, buf_kr.shape[2],
                                 device=buf_kr.device, dtype=buf_kr.dtype)
            new_kr[:, :cur, :] = buf_kr[:, :cur, :]
            self.pasts[li] = (new_c, new_kr, pos)

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
