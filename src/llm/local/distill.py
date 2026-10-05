"""跨词表 logit 蒸馏 — ULD 思想的字符级简化版（MiniCPM 系 OPD 的轻量替代）.

背景：标准 logit 蒸馏要求师生同词表；MiniCPM（10 万级 BPE）与本模型
（8192 字符表）对不上。通用 ULD 用动态规划对齐两种切分——而本模型是
一字一 token，学生位置 i 恒等于第 i 个字符，对齐退化成查表。

做法（anchor 法）：
- 锚点：双方表面字符串完全相同的 token（单汉字在 BPE 里大概率独立成 token）。
  两个分布都截断到锚点子集、重归一化，再做 KL（teacher||student，前向保覆盖）。
- 对齐位：老师 token t_k 覆盖字符 [a,b)，取"学生位置 b-1 预测字符 b"
  与"老师位置 k 预测 token k+1"配对（同一未来）。
- UNK/特殊位无字符，自然被排除在配对之外。

Loss：CE(真值) + kd_alpha * KL/T^2（T=温度，KL 按 T^2 缩放是 Hinton 标准做法）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_anchor_mapping(
    student_chars: list[str],
    teacher_tokenizer,
    skip_ids: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[int, int]:
    """建锚点映射 student_id -> teacher_id（表面字符串完全一致）.

    逐个学生字符调老师分词器：恰好切成 1 个 token 且解码回来一致才算锚点。
    skip_ids: 跳过特殊位（默认 PAD/BOS/EOS/UNK）。
    """
    mapping: dict[int, int] = {}
    for sid, ch in enumerate(student_chars):
        student_id = sid + 4  # 学生 id = 字符序号 + 4（0-3 为特殊位）
        if student_id in skip_ids or not ch:
            continue
        ids = teacher_tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) != 1:
            continue
        if teacher_tokenizer.decode(ids) == ch:
            mapping[student_id] = ids[0]
    return mapping


def anchor_coverage(
    student_chars: list[str],
    teacher_tokenizer,
    sample_texts: list[str],
) -> dict:
    """Phase 1 度量：锚点覆盖率（可映射率 + 老师概率质量占比估计）."""
    mapping = build_anchor_mapping(student_chars, teacher_tokenizer)
    index = {ch: i for i, ch in enumerate(student_chars)}
    total = sum(len(t) for t in sample_texts)
    # 采样文本中有多少字符落在锚点上（学生侧可蒸馏比例）
    anchored = sum(
        1 for t in sample_texts for ch in t
        if ch in index and (index[ch] + 4 in mapping)
    )
    return {
        "n_anchors": len(mapping),
        "student_vocab": len(student_chars) + 4,
        "anchor_rate": round(len(mapping) / max(len(student_chars), 1), 4),
        "text_coverage": round(anchored / max(total, 1), 4),
    }


def load_teacher(model_dir: str, device: str = "cuda"):
    """加载老师模型 + 分词器（frozen bf16 eval）.

    MiniCPM 自带 modeling 引用了新版 transformers 已删除的 FX 特性检测，
    这里打兼容垫片（仅影响可选特性分支，不影响前向）。
    """
    try:
        import transformers.utils.import_utils as _iu

        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda *a, **k: False
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


class AnchorDistiller:
    """锚点蒸馏器：老师 frozen，只出分布；学生拟合锚点上的 KL."""

    def __init__(
        self,
        teacher_model,
        teacher_tokenizer,
        anchor_map: dict[int, int],
        temperature: float = 2.0,
    ):
        self.teacher = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.temperature = temperature
        # 张量化映射（anchor 在学生/老师词表中的 id 序列，一一对应）
        s_ids = sorted(anchor_map)
        self.student_anchor_ids = torch.tensor(s_ids, dtype=torch.long)
        self.teacher_anchor_ids = torch.tensor(
            [anchor_map[s] for s in s_ids], dtype=torch.long)

    def _student_positions(
        self, ids: list[int], id_to_char: dict[int, str]
    ) -> tuple[str, list[int | None]]:
        """学生 id 序列解码为字符 + 每字符对应的学生位置（特殊位记 None）."""
        chars: list[str] = []
        pos_of_char: list[int | None] = []
        for p, i in enumerate(ids):
            ch = id_to_char.get(i)
            if ch is None:
                continue
            chars.append(ch)
            pos_of_char.append(p)
        return "".join(chars), pos_of_char

    def _teacher_pass(
        self, text: str, device: torch.device,
    ) -> tuple[list[int], list[tuple[int, int]], torch.Tensor] | None:
        """单文本的老师前向：返回 (t_ids, offsets, logits)，太短返回 None."""
        enc = self.teacher_tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False)
        t_ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]
        if len(t_ids) < 2 or len(text) < 4:
            return None
        with torch.no_grad():
            t_input = torch.tensor([t_ids], device=device)
            # 老师可能在 CPU（小卡场景）；尽量放同卡，不行就 CPU 算完搬回；
            # MiniCPM 自带 modeling 必须显式关 cache（否则报格式错）
            try:
                t_dev = next(self.teacher.parameters()).device
            except StopIteration:
                t_dev = torch.device("cpu")
            t_logits = self.teacher(
                t_input.to(t_dev), use_cache=False).logits.float().to(device)
        return t_ids, offsets, t_logits

    def aligned_pairs(
        self,
        student_ids: torch.Tensor,
        id_to_char: dict[int, str],
    ) -> list[list[tuple[int, int, int]]]:
        """每序列的对齐位：[(学生位置, 老师位置, 老师目标 id)].

        老师 token k 覆盖字符 [a,b) -> 学生位置 (b-1) 预测字符 b，
        与老师位置 k 预测 token k+1 配对（同一未来）。
        不跑老师前向（纯分词对齐），供 excess 选择与 KL 共用。
        """
        device = student_ids.device
        all_pairs: list[list[tuple[int, int, int]]] = []
        for b in range(student_ids.shape[0]):
            ids = student_ids[b].tolist()
            text, pos_of_char = self._student_positions(ids, id_to_char)
            pairs: list[tuple[int, int, int]] = []
            tp = self._teacher_pass(text, device)
            if tp is None:
                all_pairs.append(pairs)
                continue
            t_ids, offsets, _ = tp
            for k in range(len(t_ids) - 1):
                a, bnd = offsets[k]
                if bnd <= 0 or bnd >= len(pos_of_char):
                    continue
                s_pos = pos_of_char[bnd - 1]
                if s_pos is None or s_pos + 1 >= student_ids.shape[1]:
                    continue
                pairs.append((s_pos, k, t_ids[k + 1]))
            all_pairs.append(pairs)
        return all_pairs

    def teacher_token_losses(
        self,
        student_ids: torch.Tensor,
        id_to_char: dict[int, str],
    ) -> list[list[tuple[int, float]]]:
        """每序列 [(学生位置, 老师 CE)]：T=1 原始分布，不做温度（参考用）."""
        device = student_ids.device
        out: list[list[tuple[int, float]]] = []
        for b in range(student_ids.shape[0]):
            ids = student_ids[b].tolist()
            text, pos_of_char = self._student_positions(ids, id_to_char)
            tp = self._teacher_pass(text, device)
            if tp is None:
                out.append([])
                continue
            t_ids, offsets, t_logits = tp
            seq: list[tuple[int, float]] = []
            for k in range(len(t_ids) - 1):
                a, bnd = offsets[k]
                if bnd <= 0 or bnd >= len(pos_of_char):
                    continue
                s_pos = pos_of_char[bnd - 1]
                if s_pos is None or s_pos + 1 >= student_ids.shape[1]:
                    continue
                t_loss = F.cross_entropy(
                    t_logits[0, k].unsqueeze(0),
                    torch.tensor([t_ids[k + 1]], device=device))
                seq.append((s_pos, float(t_loss)))
            out.append(seq)
        return out

    def batch_kl(
        self,
        student_ids: torch.Tensor,
        student_logits: torch.Tensor,
        id_to_char: dict[int, str],
    ) -> torch.Tensor:
        """整 batch 的锚点 KL（teacher||student），无配对时返回 0.

        student_ids: (b, T) 输入 id；student_logits: (b, T, Vs) 全精度要求不高。
        老师前向 no_grad，只让学生拿梯度（与 teacher_token_losses 共用 _teacher_pass，
        同一 micro-step 内调两次会跑两次老师前向，生产可用缓存合并，此处清晰优先）。
        """
        device = student_logits.device
        s_anchor = self.student_anchor_ids.to(device)
        t_anchor = self.teacher_anchor_ids.to(device)
        t = self.temperature
        total_kl = torch.zeros((), device=device, dtype=torch.float32)
        n_pairs = 0
        for b in range(student_ids.shape[0]):
            ids = student_ids[b].tolist()
            text, pos_of_char = self._student_positions(ids, id_to_char)
            tp = self._teacher_pass(text, device)
            if tp is None:
                continue
            t_ids, offsets, t_logits = tp
            # 配对：老师 token k 覆盖 [a,b) -> 学生位置 (b-1) 预测字符 b
            for k in range(len(t_ids) - 1):
                a, bnd = offsets[k]
                if bnd <= 0 or bnd >= len(pos_of_char):
                    continue
                s_pos = pos_of_char[bnd - 1]  # 预测字符 b 的学生位置
                if s_pos is None or s_pos + 1 >= student_logits.shape[1]:
                    continue
                s_dist = student_logits[b, s_pos]
                t_dist = t_logits[0, k]
                s_a = s_dist[s_anchor] / t
                t_a = t_dist[t_anchor] / t
                s_log = F.log_softmax(s_a.float(), dim=-1)
                t_prob = F.softmax(t_a.float(), dim=-1)
                total_kl = total_kl + F.kl_div(s_log, t_prob, reduction="sum")
                n_pairs += 1
        if n_pairs == 0:
            return total_kl.to(student_logits.dtype)
        # Hinton 缩放：KL 按 T^2 还原梯度量级，再按配对数平均
        return (total_kl * (t * t) / n_pairs).to(student_logits.dtype)


def select_by_excess(
    student_losses: torch.Tensor,
    pair_losses: list[list[tuple[int, float]]],
    keep_ratio: float,
    min_keep: int = 8,
) -> torch.Tensor:
    """老师参照 RHO 选择：excess = 学生 − 老师，只在配对位上排名.

    pair_losses: 每序列 [(学生位置, 老师 CE)]（teacher_token_losses 产出）。
    配对位按 excess 取 top；非配对位恒训练（无老师信号，保持原 CE 不动）。
    返回与 student_losses 同形 bool 掩码。
    """
    flat = student_losses.reshape(-1)
    stride = student_losses.shape[-1]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    # 非配对位恒开（注意：含 ignore 位，其 loss 为 0、无梯度，开着也无妨）
    paired_pos: set[int] = set()
    scored: list[tuple[float, int]] = []  # (excess, flat_idx)
    for b, seq in enumerate(pair_losses):
        for s_pos, t_loss in seq:
            idx = b * stride + s_pos
            paired_pos.add(idx)
            s_loss = float(flat[idx].detach())
            scored.append((s_loss - t_loss, idx))
    mask[:] = True
    for idx in paired_pos:
        mask[idx] = False  # 先关配对位，再按 excess 开
    if not scored:
        return mask
    k = max(min(int(len(scored) * keep_ratio), len(scored)),
            min(min_keep, len(scored)))
    if k >= len(scored):
        for _, idx in scored:
            mask[idx] = True
        return mask.view(student_losses.shape)
    with torch.no_grad():
        vals = torch.tensor([s for s, _ in scored])
        thr = torch.topk(vals, k).values.min().item()
    for s, idx in scored:
        if s >= thr:
            mask[idx] = True
    return mask.view(student_losses.shape)
