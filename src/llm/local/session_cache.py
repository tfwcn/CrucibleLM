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
        """MLA 系静态缓存判定：三元组 `(buf_c, buf_kr, pos)` 或
        六元组 `(buf_c, buf_kr, pos, k_ring, kr_ring, v_ring)`（层内 SWA 开启）.

        线性层 past 也是三元组 (S, k_buf, v_buf)，但第 3 元是 float 卷积尾；
        静态缓存的第 3 元恒为 long 标量 pos，以此区分。
        """
        return (isinstance(p, tuple) and len(p) in (3, 6)
                and isinstance(p[2], torch.Tensor)
                and p[2].dtype == torch.long and p[2].numel() == 1)

    @torch.no_grad()
    def ensure_room(self, need: int) -> None:
        """保证各层静态缓存剩余容量 ≥ need，不够按 2x 扩.

        eager 重分配（拷贝旧有效区）；编译过的 decode 重编一次，
        指数扩容下均摊可忽略。线性层状态定长，无需处理。

        CSA2 借用层（reindex/reuse）与生产者**共享同一个 latent 缓冲**，
        扩容必须按张量身份去重：否则逐层各建一份新缓冲，生产者写进自己那份、
        借用层读到的还是旧副本（pitfall 36）。SWA 环定长，原样带上。
        """
        if self.pasts is None:
            return
        # 第 1 遍：按旧张量身份汇总目标容量（共享缓冲取各请求的最大值）。
        # **块主序缓存**下一行装 m 个 token → 目标行数用 `ceil((cur+need)/m)`。
        plans: dict[int, tuple[torch.Tensor, int]] = {}
        for li, p in enumerate(self.pasts):
            if not self._is_static_triple(p):
                continue
            cur = int(p[2].item())
            need_rows = self._rows_needed(li, cur + need)
            for t in (p[0], p[1]):
                if need_rows <= t.shape[1]:
                    continue
                nlen = max(t.shape[1] * 2, need_rows)
                old = plans.get(id(t))
                if old is None or nlen > old[1]:
                    plans[id(t)] = (t, nlen)
        if not plans:
            return
        # 第 2 遍：生成新缓冲（同一旧张量只生成一次）。
        # 量化缓存（QuantBuffer）走自己的 resize（打包存储 + 块 scale）；
        # 注意 dtype 可能是 float8/int8，不能统一 torch.zeros 再拷贝。
        fresh: dict[int, object] = {}
        for key, (t, nlen) in plans.items():
            if hasattr(t, "resize") and not isinstance(t, torch.Tensor):
                fresh[key] = t.resize(nlen)
            else:
                new = torch.zeros(t.shape[0], nlen, t.shape[2],
                                  device=t.device, dtype=t.dtype)
                new[:, :t.shape[1]] = t
                fresh[key] = new
        # 第 3 遍：就地替换（未扩容的 past 保持原对象，SWA 环元组原样续接）
        for li, p in enumerate(self.pasts):
            if not self._is_static_triple(p):
                continue
            nc, nkr = fresh.get(id(p[0])), fresh.get(id(p[1]))
            if nc is None and nkr is None:
                continue
            self.pasts[li] = (nc if nc is not None else p[0],
                              nkr if nkr is not None else p[1], p[2]) \
                + tuple(p[3:])

    def _block_m(self, li: int) -> int:
        """该层主 KV 缓存的压缩比 m（块主序），非块主序层返回 0.

        扩容按**行**算（块主序下 `m + ceil((cur+need)/m)` 行），拿 token 数直接比
        行数会多扩 m 倍——不致命但白占内存，且会掩盖"块主序真的省了 L/m"。
        暂存前缀在**头部**（行 `[0, m)`），尾部扩容天然不搬动它（残块不会丢）。
        """
        try:
            attn = self.model.layers[li].attn
        except (AttributeError, IndexError, TypeError):
            return 0
        if not getattr(attn, "block_cache", False):
            return 0
        return max(0, int(getattr(attn, "compress_ratio", 1)))

    def _rows_needed(self, li: int, tokens: int) -> int:
        """该层在 `tokens` 个 token 下需要的行数（块主序：m 行暂存 + ceil(tokens/m)）."""
        m = self._block_m(li)
        return (m + (tokens + m - 1) // m) if m else tokens

    def _device(self) -> torch.device:
        """模型所在设备."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def save(self, path: str, drop_swa: bool = True) -> None:
        """落盘（past 全是小张量：latent/状态/卷积尾，无 O(N^2) 大物）.

        `drop_swa=True`（默认，对齐 V4.1 §3.2.2）：**SWA 环不落盘**，持久化
        只存全局主 KV；环定长且可由最近 `n_win` 个 token 重放重建，存它纯属
        浪费带宽——它的体积是 `2*n_heads*head_dim*n_win` 每层（n_win=4096、
        n_heads=64、head_dim=128 时约 67MB/层，与 200K 会话的全局主 KV 同量级）。
        代价：`load` 需一次 Bounded Replay（近似，见 `replay_swa`）。
        `drop_swa=False`：环也存 → `load` 逐位精确（调试/对照用）。
        """
        assert self.pasts is not None, "空缓存无需存盘"
        slim = ([p[:3] if isinstance(p, tuple) and len(p) == 6 else p
                 for p in self.pasts] if drop_swa else self.pasts)
        torch.save({"pasts": slim, "ids": self.ids,
                    "swa_dropped": bool(drop_swa)}, path)

    @classmethod
    def load(cls, path: str, model) -> "SessionCache":
        """读回（map 到模型所在设备）；缺 SWA 环时用 Bounded Replay 近似重建.

        V4.1 §3.2.2：落盘只存全局主 KV，SWA 环由**最近 `n_win` 个 token** 重放
        重建（精确重建需 `L × n_win`）。重放走
        `model.replay_swa_from_ids(ids)`——它只读权威全局主 KV、只重建 SWA 环，
        见 `TinyLLM.replay_swa_from_ids`（近似来源写在那边的 docstring）。
        """
        sc = cls(model)
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        # weights_only=False：落盘里有 QuantBuffer/meta 等自定义对象（本文件由
        # 自己 save 产生，可信）；torch>=2.6 默认 True 会拒载
        blob = torch.load(path, map_location=dev, weights_only=False)
        sc.pasts = blob["pasts"]
        # ids 只是记账（`extend` 往 CPU 上拼），**别跟着 map 到设备**：否则 GPU 上
        # 读回后第一次 `extend` 会在 `torch.cat` 处设备不一致而炸（本仓踩过）。
        sc.ids = blob["ids"].to("cpu") if blob["ids"] is not None else None
        if blob.get("swa_dropped"):
            sc.replay_swa()
        return sc

    @torch.no_grad()
    def replay_swa(self) -> None:
        """用最近 `n_win` 个 token 重建各层 SWA 环（Bounded Replay）.

        重建后 `pasts` 从三元组补齐为六元组
        `(buf_c, buf_kr, pos, k_ring, kr_ring, v_ring)`，
        与未落盘直接续跑的解码路径完全同构。
        """
        if self.pasts is None or self.ids is None:
            return
        rebuild = getattr(self.model, "replay_swa_from_ids", None)
        if rebuild is None:
            return
        self.pasts = rebuild(self.ids, self.pasts)

    def clear(self) -> None:
        """清空（会话结束/超长截断时调用，调用方另行处理）."""
        self.pasts = None
        self.ids = None
