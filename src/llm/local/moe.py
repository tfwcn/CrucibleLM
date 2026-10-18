"""细粒度 MoE — DeepSeek 系多小专家 + 共享专家 + 负载均衡（aux loss / 免辅助损失路由）
+ SwiGLU 限幅."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """单个 SwiGLU 前馈专家（小中间维度），可选预激活限幅.

    `limit > 0` 时对 gate/up 预激活做**对称限幅**（DeepSeek-V4.1 `swiglu_limit`）：
    `clamp(z, -limit, limit)` 后再进 silu/相乘，压住离群值（个别维度爆炸主导残差流）。
    `limit <= 0` 时逐位等价旧实现（无限幅）。
    """

    def __init__(self, d_model: int, hidden: int, limit: float = 0.0):
        super().__init__()
        # 限幅阈值（纯超参，不入 state_dict，键名与形状不变）
        self.limit = float(limit)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def gate_up(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回（已限幅的）gate/up 预激活，供前向与离线分析共用，保证口径一致."""
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.limit > 0:
            gate = gate.clamp(-self.limit, self.limit)
            up = up.clamp(-self.limit, self.limit)
        return gate, up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前向；开启限幅时先截断 gate/up 预激活."""
        gate, up = self.gate_up(x)
        return self.w_down(F.silu(gate) * up)


class FineGrainedMoE(nn.Module):
    """细粒度 MoE：top-k 路由 + 共享专家兜底."""

    def __init__(
        self,
        d_model: int,
        n_experts: int = 16,
        top_k: int = 4,
        expert_hidden: int = 192,
        n_shared: int = 1,
        aux_coef: float = 0.01,
        swiglu_limit: float = 10.0,
        router_aux_free: bool = True,
        router_bias_lr: float = 1e-3,
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_coef = aux_coef
        # SwiGLU 预激活限幅（DeepSeek-V4.1；0=关）。默认 10.0，与 config 一致
        self.swiglu_limit = float(swiglu_limit)
        # 路由（DeepSeek-V4.1 免辅助损失）；关闭时逐位等价旧口径（softmax+greedy+aux）
        self.router_aux_free = bool(router_aux_free)
        self.router_bias_lr = float(router_bias_lr)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden, self.swiglu_limit)
             for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden, self.swiglu_limit)
             for _ in range(n_shared)]
        )
        # 免辅助损失路由偏置（非持久化：不进 state_dict，保持装载键兼容；续训时归零后
        # 数步内自适应）。仅 router_aux_free 时被 update_router_bias 更新。
        self.register_buffer("expert_bias", torch.zeros(n_experts),
                             persistent=False)
        # 最近一次训练前向的逐专家命中占比（用于偏置更新；赋值语义=重算幂等）
        self._last_load: torch.Tensor | None = None
        # 可选：专家跨设备搬运（默认 None = 走分组 bmm 快路径，逐位不变）
        self.store = None
        # 可选：LES 专家预测器（默认 None；仅 offload 时才有意义）
        self.les = None
        self.prefetch = False

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        """路由打分：sqrtsoftplus（V4.1）或 softmax（旧口径）."""
        if self.router_aux_free:
            return torch.sqrt(F.softplus(logits))
        return F.softmax(logits, dim=-1)

    def _aux_loss(self, score: torch.Tensor, top_i: torch.Tensor) -> torch.Tensor:
        """旧口径负载均衡 aux loss（专家重要性分布与实际负载分布的加权点积）."""
        flat_b = top_i.numel() // self.top_k
        importance = score.reshape(-1, self.n_experts).mean(0)
        load_frac = (
            F.one_hot(top_i.reshape(-1), self.n_experts).float().sum(0)
            / max(flat_b * self.top_k, 1)
        )
        return (importance * load_frac).sum() * self.n_experts * self.aux_coef

    def _route(
        self, flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一路由：返回 (权重, top-k 专家号, aux_loss).

        aux_free（V4.1 noaux_tc）：按 `score + 偏置` 选 top-k；权重取**原始 score**，
        归一化后 × `routed_scaling_factor`；不产生 aux loss，改累计本轮负载供
        `update_router_bias`。关闭时（greedy+softmax）与旧实现逐位一致（单测锁定）。
        """
        logits = self.router(flat)
        score = self._score(logits)
        sel = (score + self.expert_bias.to(score.dtype)
               if self.router_aux_free else score)
        _, top_i = torch.topk(sel, self.top_k, dim=-1)
        weights = torch.gather(score, -1, top_i)  # 权重用原始打分（不含偏置）
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        scale = self.routed_scaling_factor if self.router_aux_free else 1.0
        top_w = (weights * scale).to(flat.dtype)
        if self.router_aux_free:
            aux = torch.zeros((), device=flat.device, dtype=flat.dtype)
            self._accumulate_load(top_i)
        else:
            aux = self._aux_loss(score, top_i)
        return top_w, top_i, aux

    def _accumulate_load(self, top_i: torch.Tensor) -> None:
        """记录本次训练前向的逐专家命中占比（赋值语义：checkpoint 重算幂等）."""
        if not self.training:
            return
        with torch.no_grad():
            counts = torch.bincount(
                top_i.detach().reshape(-1).contiguous(),
                minlength=self.n_experts).to(torch.float32)
            self._last_load = counts / max(top_i.numel(), 1)

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """按最近一次负载更新路由偏置（DeepSeek 免辅助损失负载均衡，sign 式）.

        欠载专家 → 提高偏置（更易被选），过载专家 → 降低。仅在 `router_aux_free` 且
        `router_bias_lr>0` 时生效；每个优化器步调用一次（训练循环负责）。
        """
        if not self.router_aux_free or self.router_bias_lr <= 0:
            return
        f = self._last_load
        if f is None:
            return
        target = 1.0 / self.n_experts
        self.expert_bias += self.router_bias_lr * torch.sign(
            target - f.to(device=self.expert_bias.device,
                          dtype=self.expert_bias.dtype))

    def enable_offload(
        self,
        compute_device,
        target: str = "cpu",
        offload_dir: str = "",
        capacity: int = 0,
        prefetch: bool = False,
        name: str = "expert",
    ):
        """开启专家 offload（构建 ExpertStore；由模型 setup_offload 调用）."""
        from src.llm.local.offload import ExpertStore

        self.store = ExpertStore(
            self.experts, compute_device, target, offload_dir, capacity, name)
        self.prefetch = prefetch
        return self

    def enable_les(self, history: int = 8, topn: int = 4):
        """开启 LES 专家预测（隐含预取；无 offload 时不会被消费）."""
        from src.llm.local.les import build_les

        self.les = build_les(self.n_experts, history, topn)
        self.prefetch = True
        return self

    def _forward_offload(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """offload 前向：只把命中专家搬到算力设备，逐专家算后散射加回.

        与朴素全量实现数学一致（tests/test_offload.py 对比分组版逐位/紧公差）。
        共享专家常驻（不参与 offload），aux loss 与分组版同式。
        """
        orig_shape = x.shape
        d = x.shape[-1]
        flat = x.reshape(-1, d)
        top_w, top_i, aux_loss = self._route(flat)
        hit = torch.unique(top_i)
        # LES：按历史命中预测下一步热点，先搬上算力设备（预取；只影响时序）
        if self.prefetch and self.les is not None:
            # 预取：只填空闲容量、不淘汰（避免挤掉马上要用的专家）
            self.store.ensure(self.les.predict(self.store.hot_ids()), prefetch=True)
        out = torch.zeros_like(flat)
        for e in hit.tolist():
            sel = (top_i == e).any(dim=-1)
            if not bool(sel.any()):
                continue
            # 逐个 ensure（不能整批先 ensure 再算：capacity < 命中数时，
            # 后搬入的会把先搬入的挤回冷端，随后用到已置 meta 的专家 → 错）
            self.store.ensure([e])
            idx = sel.nonzero(as_tuple=True)[0]
            w_e = (top_w * (top_i == e).to(top_w.dtype)).sum(-1)[sel]
            y_e = self.experts[e](flat[sel])
            out = out.index_add(0, idx, w_e.unsqueeze(-1) * y_e)
        out = out.view(orig_shape)
        for expert in self.shared:
            out = out + expert(x)
        # LES：记录本步真实命中，供下一步预测
        if self.les is not None:
            self.les.observe(hit.tolist())
        return out, aux_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向并返回 (输出, aux_loss).

        aux_loss: 标准负载均衡损失 mean(p) 与 one-hot 负载的点积，
        防止路由坍缩到少数专家。

        分组 bmm：每个 token 展开成 top-k 行，按专家排序后垫齐 batch，
        3 次 bmm 算完所有专家（替代 16 次逐专家 Python dispatch，
        解码主力开销），再按权重散射加回。数值与逐专家版一致（单测锁定）。

        offload 开启时：默认走逐专家命中路径；若**全部专家都已驻留**算力设备
        （容量足够、预热完成），则退回分组快路径（数值一致，省掉逐专家 dispatch）。
        """
        if self.store is not None:
            if not self.store.all_resident():
                # 还有冷专家：逐专家命中路径（只搬命中专家上算力设备）
                return self._forward_offload(x)
            # 全驻留：分组快路径
        return self._forward_grouped(x)

    def _forward_grouped(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """分组 bmm 前向（全部专家在算力设备时用；数值与逐专家版一致）."""
        orig_shape = x.shape
        d = x.shape[-1]
        flat = x.reshape(-1, d)  # (N, d)
        n = flat.shape[0]
        top_w, top_i, aux_loss = self._route(flat)
        # 展开：每 token k 行（token 序号、专家号、归一化权重）
        m = n * self.top_k
        tok_idx = torch.arange(n, device=x.device).repeat_interleave(self.top_k)
        exp_idx = top_i.reshape(-1)
        # 权重对齐到输入精度（AMP 下 top_w 常为 fp32：避免 index_put 报类型错，
        # 也避免旧式 where 隐式提升把残差流抬成 fp32）
        w_row = top_w.reshape(-1).to(flat.dtype)
        order = torch.argsort(exp_idx, stable=True)
        exp_s = exp_idx[order]
        counts = torch.bincount(exp_s, minlength=self.n_experts)
        max_n = int(counts.max().item())
        # 组内槽位：slot[r] = 该行在其专家组内的序号
        cum = counts.cumsum(0)
        slot = torch.arange(m, device=x.device) - (cum[exp_s] - counts[exp_s])
        # 画布 (E, max_n, *)：token / 权重 / 有效掩码
        canvas = torch.zeros(self.n_experts, max_n, d,
                             device=x.device, dtype=flat.dtype)
        canvas[exp_s, slot] = flat.repeat_interleave(self.top_k, dim=0)[order]
        w_canvas = torch.zeros(self.n_experts, max_n,
                               device=x.device, dtype=flat.dtype)
        w_canvas[exp_s, slot] = w_row[order]
        row_mask = torch.zeros(self.n_experts, max_n,
                               device=x.device, dtype=torch.bool)
        row_mask[exp_s, slot] = True
        # 专家权重堆叠（view 级 stack，反向精确回传到各专家参数）
        w_g = torch.stack([e.w_gate.weight for e in self.experts])  # (E, h, d)
        w_u = torch.stack([e.w_up.weight for e in self.experts])
        w_d = torch.stack([e.w_down.weight for e in self.experts])  # (E, d, h)
        gate_raw = canvas @ w_g.transpose(1, 2)
        up_raw = canvas @ w_u.transpose(1, 2)
        if self.swiglu_limit > 0:
            # 与 SwiGLUExpert.forward 同一限幅口径（分组/逐专家数值须一致，单测锁定）
            lim = self.swiglu_limit
            gate_raw = gate_raw.clamp(-lim, lim)
            up_raw = up_raw.clamp(-lim, lim)
        gate = F.silu(gate_raw) * up_raw
        out_e = (gate @ w_d.transpose(1, 2)) * (
            w_canvas * row_mask.to(flat.dtype)).unsqueeze(-1)
        out = torch.zeros_like(flat)
        out.index_add_(0, tok_idx[order], out_e[exp_s, slot])
        out = out.view(orig_shape)
        # 共享专家直通（常驻知识通道）
        for expert in self.shared:
            out = out + expert(x)
        return out, aux_loss
