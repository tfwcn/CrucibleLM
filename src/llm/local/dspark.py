"""DSpark：独立训练的投机解码草稿器（DeepSeek-V4.1 §2.4.3）.

三件事，缺一不成"半自回归"：

1. **3 层 Transformer**（滑窗注意力 128）读最近 `window` 个 token 的 embedding
   （末位换成主干 hidden），**一次前向**并行产出 `depth` 个位置的 base logits。
   并行意味着位与位之间**互不可见**——这是"半自回归"的"半"。
2. **Markov 头**补上依赖：`logits_i = base_i + U·E(t_{i-1})`（低秩一阶转移，
   零初始化 = 开局退化为纯并行）。这是"半自回归"的"自回归"。
3. **置信头**给每个位置一个**条件接受概率** `c_i = σ(w·h_i)`，调度器用它算前缀
   存活概率 `S_i = ∏_{j≤i} c_j`，再结合引擎吞吐曲线选"这轮验多长"。

为什么草稿器要**独立训练**（V4.1 的做法）：主干预训练时只带 MTP（浅层、联合训练）；
DSpark 在预训练后单独开班——主干冻结、只训草稿器；后训练阶段继续一起训但**梯度
不回传主干**，让它始终对齐当前策略。所以本模块不带主干参数，也不参与主损失。

与 MTP 的区别：MTP 只多看一步（t+2）且与主干共享前向；DSpark 一次给
`depth` 个位置、有自己的 3 层、自己的 Markov 依赖，验证时由主干并行兜底。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.llm.local.mla import RMSNorm


def _rope_for(pos_start: int, n: int, dim: int, device, dtype=torch.float32,
              theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """按**绝对位置**构造一段 cos/sin（(n, dim)）.

    不用全局缓存：草稿器只需要 [pos_start, pos_start+n) 这一小段（≤ window+depth），
    长上下文下按需生成反而省内存，也避免和主干的 YaRN 缩放耦合。
    """
    half = dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32,
                                        device=device) / half))
    pos = torch.arange(pos_start, pos_start + n, dtype=torch.float32,
                       device=device)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
    """对 `(b, h, L, hd)` 的最后一维做 RoPE（半块拆分，与 `rope.apply_rope` 同构）."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c[..., :half] - x2 * s[..., :half],
                      x1 * s[..., half:] + x2 * c[..., half:]], dim=-1)


def draft_mask(n_ctx: int, depth: int, window: int, device=None) -> torch.Tensor:
    """草稿器的注意力掩码 `(n, n)` bool（True=可见）.

    - ctx 内部：因果 + 滑窗（只看最近 `window` 个，含自己）；
    - 槽位（草稿位置）：只看 ctx（滑窗内），**互不可见** —— 这是"并行 base
      logits"的定义；位置之间的依赖交给 Markov 头，而不是注意力。
    """
    n = n_ctx + depth
    idx = torch.arange(n, device=device)
    causal = idx[:, None] >= idx[None, :]
    within = (idx[:, None] - idx[None, :]) < window
    slot = idx >= n_ctx
    crosstalk = slot[:, None] & slot[None, :]
    return causal & within & ~crosstalk


class _DraftBlock(nn.Module):
    """草稿器的一层：滑窗因果自注意力 + SwiGLU（pre-norm，极小）."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 须被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        """x: (b, L, d)；attn_mask: (L, L) bool，True=可见."""
        b, n, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        x = x + self.o_proj(ctx)
        h = self.norm2(x)
        return x + self.down(self.act(self.gate(h)) * self.up(h))


class DSparkDrafter(nn.Module):
    """半自回归草稿器：一次前向给 `depth` 个 base logits + 条件接受概率.

    参数量 ≈ `layers × (4d² + 6d²)`（d=384 时约 5M），不含词表头与 embedding
    （两者都**借用冻结的主干**，草稿器与主干共享语义空间）。
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 3,
        n_heads: int = 4,
        window: int = 128,
        depth: int = 5,
        markov_rank: int = 32,
        ffn_mult: int = 2,
    ):
        super().__init__()
        assert n_layers >= 1 and depth >= 1 and markov_rank >= 1
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.window = max(2, int(window))
        self.depth = int(depth)
        self.layers = int(n_layers)
        self.blocks = nn.ModuleList([
            _DraftBlock(d_model, n_heads, ffn_mult * d_model)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        # 并行查询槽位：每个槽位对应一个草稿位置（互不可见 → 必须靠 Markov 头串起来）
        self.q_slots = nn.Parameter(torch.randn(depth, d_model) * 0.02)
        # Markov 头（一阶、低秩）：U·E(t_{i-1}) 加到该位置的 base logits 上
        self.markov_emb = nn.Embedding(vocab_size, markov_rank)
        self.markov_out = nn.Linear(markov_rank, vocab_size, bias=False)
        # 零初始化 → 开局 Markov 项为 0（纯并行、可训）
        nn.init.zeros_(self.markov_emb.weight)
        nn.init.zeros_(self.markov_out.weight)
        # 置信头：逐位置预测"给定前缀已接受时，本位置被接受"的概率
        self.conf_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.zeros_(self.conf_head.bias)

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        h_last: torch.Tensor,
        ctx_ids: torch.Tensor,
        embed_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """一次前向：`(base_logits (b,m,V), conf (b,m))`.

        h_last: (b,1,d) 主干末位 hidden（调用方负责 no_grad）；
        ctx_ids: (b,L) 最近 L 个已提交 token（L ≤ window），末位会被 h_last 替换；
        embed_fn: token ids → embedding（冻结的主干 embedding）。
        """
        b, l = ctx_ids.shape
        l = min(l, self.window)
        ids = ctx_ids[:, -l:]
        x = embed_fn(ids).to(h_last.dtype)
        # 末位换成主干 hidden：草稿器从"主干看到的状态"接着往下猜
        x = torch.cat([x[:, :-1], h_last.to(x.dtype)], dim=1) if l > 1 else h_last
        m = self.depth
        slots = self.q_slots.view(1, m, self.d_model).expand(b, -1, -1)
        x = torch.cat([x, slots.to(x.dtype)], dim=1)
        n = x.shape[1]
        # 只关心窗口内部的**相对**位置：q·k 的 RoPE 内积只依赖相对偏移，
        # 掩码也只按相对距离算，故绝对起点取 0 即可（草稿器无需知道上下文长度）
        cos, sin = _rope_for(0, n, self.blocks[0].head_dim, x.device, x.dtype)
        mask = draft_mask(l, m, self.window, device=x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        hq = self.norm(x[:, l:])
        conf = torch.sigmoid(self.conf_head(hq).squeeze(-1))   # (b,m)
        return hq, conf

    def markov_bias(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """`U·E(t_prev)`：`(b,)` 或 `(b,k)` long → `(b,k,V)`（或 (b,V)）."""
        return self.markov_out(self.markov_emb(prev_ids))


class DSparkScheduler:
    """DSpark 吞吐调度器：按存活概率 + 引擎成本曲线选验证长度.

    期望产出（每轮，含主干自己那颗必然接受的首 token）：

        tokens(n) = 1 + Σ_{i=1..n} S_i,   S_i = ∏_{j=1..i} c_j

    每轮成本（一次起草 + 一次 n 位置并行验证）：

        cost(n) = draft_cost + base_cost + per_token · n

    吞吐 = tokens(n) / cost(n)。`c_i` 是**条件**接受概率（置信头输出），
    `S_i` 即前缀存活概率——这就是论文说的 "estimate prefix survival
    probabilities, combine with profiled engine throughput curves"。

    成本系数是**可测常量**（`dspark_target_base`/`dspark_target_per_token`/
    `dspark_draft_cost`），与模型规模、批次、KV 布局都有关：本仓库给的是
    单请求小模型的量级（一次验证前向 ≈ 1 + 0.15n 个 token 的算力，
    一次草稿前向 ≈ 0.6）——它只影响"选多长"，不影响正确性。
    """

    def __init__(
        self,
        target_base: float = 1.0,
        target_per_token: float = 0.15,
        draft_cost: float = 0.6,
        min_verify: int = 0,
        max_verify: int | None = None,
    ):
        self.target_base = max(1e-6, float(target_base))
        self.target_per_token = max(0.0, float(target_per_token))
        self.draft_cost = max(0.0, float(draft_cost))
        self.min_verify = max(0, int(min_verify))
        self.max_verify = max_verify

    # -------------------------------------------------------------- 记账
    def survival(self, confs: list[float]) -> list[float]:
        """前缀存活概率 `S_i`（含第 i 个草稿被接受的概率）."""
        out: list[float] = []
        cum = 1.0
        for c in confs:
            cum *= min(1.0, max(0.0, float(c)))
            out.append(cum)
        return out

    def expected_tokens(self, confs: list[float], n: int) -> float:
        """验 n 个草稿的期望产出 token 数（1 个首 token + 期望接受数）."""
        n = max(0, min(int(n), len(confs)))
        return 1.0 + sum(self.survival(confs)[:n])

    def cost(self, n: int) -> float:
        """验 n 个草稿的相对成本（起草一次 + 验证 n+1 个位置）."""
        return self.draft_cost + self.target_base + self.target_per_token * max(0, int(n))

    def throughput(self, confs: list[float], n: int) -> float:
        """期望吞吐（token / 单位算力）."""
        return self.expected_tokens(confs, n) / self.cost(n)

    def choose_length(self, confs: list[float], k: int | None = None) -> int:
        """选吞吐最大的验证长度（并列时取更长，减少轮数/首 token 开销）."""
        if not confs:
            return 0
        hi = len(confs) if k is None else max(0, min(len(confs), int(k)))
        if self.max_verify is not None:
            hi = min(hi, int(self.max_verify))
        lo = min(self.min_verify, hi)
        best_n, best_tp = lo, self.throughput(confs, lo)
        for n in range(lo, hi + 1):
            tp = self.throughput(confs, n)
            if tp >= best_tp:          # >=：并列取更长（省一轮首 token 的固定成本）
                best_n, best_tp = n, tp
        return best_n


def attach_dspark(model, path) -> dict:
    """给已建好的模型挂上训练好的草稿器（结构按 dspark.pt 里的 config 重建）.

    推理/评测脚本用它：主干权重不动，只多一个 `model.dspark` 与
    `config.dspark_*`（结构参数必须与训练时一致，否则 `load_state_dict` 报错）。
    """
    payload = torch.load(path, map_location="cpu")
    for k, v in (payload.get("config") or {}).items():
        setattr(model.config, k, v)
    model.config.dspark_draft = True
    model.dspark = build_dspark_drafter(model.config)
    model.dspark.load_state_dict(payload["state_dict"], strict=True)
    dev = next(model.parameters()).device
    model.dspark.to(dev).eval()
    for p in model.dspark.parameters():
        p.requires_grad_(False)
    return {k: v for k, v in payload.items() if k != "state_dict"}


def build_dspark_drafter(config) -> DSparkDrafter:
    """按训练配置构建草稿器（与 `TinyLLM` 的字段一致，供脚本/单测复用）."""
    return DSparkDrafter(
        d_model=int(config.d_model),
        vocab_size=int(config.vocab_size),
        n_layers=int(getattr(config, "dspark_layers", 3)),
        n_heads=int(getattr(config, "dspark_heads", 4)),
        window=int(getattr(config, "dspark_window", 128)),
        depth=int(getattr(config, "dspark_depth", 5)),
        markov_rank=int(getattr(config, "dspark_markov_rank", 32)),
    )
