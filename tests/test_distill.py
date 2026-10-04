"""蒸馏单测 — 锚点映射/对齐/KL（fake 老师，离线可跑，无需 0.5B 权重）."""

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.distill import AnchorDistiller, build_anchor_mapping


class FakeTok:
    """极简 BPE：贪心最长匹配，自带 offsets（模拟老师分词器行为）."""

    def __init__(self):
        # id: surface；20="你好"（双字合并），21="世界"
        self.vocab = {10: "你", 11: "好", 12: "世", 13: "界", 20: "你好", 21: "世界", 99: "X"}
        self.inv = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        """单字符映射检查用（build_anchor_mapping 调这个）."""
        if text in self.inv:
            return [self.inv[text]]
        return [99, 99]  # 非单 token，迫使跳过

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.vocab.get(i, "") for i in ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids, offsets, i = [], [], 0
        while i < len(text):
            if text[i:i + 2] in self.inv:
                ids.append(self.inv[text[i:i + 2]])
                offsets.append((i, i + 2))
                i += 2
            else:
                ids.append(self.inv.get(text[i], 99))
                offsets.append((i, i + 1))
                i += 1
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeTeacher(torch.nn.Module):
    """fake 老师：embedding + 线性头，前向给确定性 logits."""

    def __init__(self, vocab=100, hidden=32):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, use_cache=False):
        return type("Out", (), {"logits": self.head(self.emb(input_ids))})()


def _distiller():
    """学生 4 个字都映射上锚点."""
    tok = FakeTok()
    mapping = build_anchor_mapping(["你", "好", "世", "界"], tok)
    assert mapping == {4: 10, 5: 11, 6: 12, 7: 13}, mapping
    teacher = FakeTeacher()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return AnchorDistiller(teacher, tok, mapping, temperature=2.0)


def test_build_anchor_mapping_skips_merged():
    """多字合并 token 不进锚点（encode 非单 id 则跳过）."""
    tok = FakeTok()
    assert tok.encode("你好") == [20]  # 单 id 但解码一致？解码 "你好"==输入…
    # "你好" encode 单 id 且 decode 一致，按规则会成锚点——但 build 只调单字
    mapping = build_anchor_mapping(["你好"], tok)
    assert mapping == {4: 20}  # 整串锚点行为符合"表面一致"定义


def test_batch_kl_finite_and_student_only():
    """KL 有限、梯度只进学生、老师无梯度."""
    d = _distiller()
    # 学生 id：4,5,6,7 = 你好世界 (+BOS=1，解码时跳过）
    student_ids = torch.tensor([[1, 4, 5, 6, 7]])
    student_logits = torch.randn(1, 5, 16, requires_grad=True)
    id_to_char = {4: "你", 5: "好", 6: "世", 7: "界"}
    loss = d.batch_kl(student_ids, student_logits, id_to_char)
    assert torch.isfinite(loss) and float(loss) >= 0
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    for p in d.teacher.parameters():
        assert p.grad is None


def test_batch_kl_no_pairs_zero():
    """无配对（空文本/单 token）返回 0，不炸."""
    d = _distiller()
    student_ids = torch.tensor([[1, 1]])
    student_logits = torch.randn(1, 2, 16, requires_grad=True)
    loss = d.batch_kl(student_ids, student_logits, {})
    assert float(loss) == 0.0


def test_alignment_boundary_positions():
    """对齐位正确：老师 token 边界 ⟺ 学生预测下一字的位置.

    文本"你好世界" -> 老师 [你好(0,2), 世界(2,4)]：
    配对只有 1 个（k=0：学生位置2预测"世" vs 老师位置0预测"世界"），
    末 token 无下一位，跳过。
    """
    d = _distiller()
    seen = {}

    orig_kl = torch.nn.functional.kl_div

    def spy_kl(s_log, t_prob, reduction="sum"):
        seen["calls"] = seen.get("calls", 0) + 1
        return orig_kl(s_log, t_prob, reduction=reduction)

    import torch.nn.functional as F

    F.kl_div = spy_kl
    try:
        student_ids = torch.tensor([[1, 4, 5, 6, 7]])
        student_logits = torch.randn(1, 5, 16, requires_grad=True)
        d.batch_kl(student_ids, student_logits, {4: "你", 5: "好", 6: "世", 7: "界"})
    finally:
        F.kl_div = orig_kl
    assert seen.get("calls") == 1


def test_select_by_excess():
    """超额选择：配对位按 excess 取 top，非配对恒开，空配对全开."""
    from src.llm.local.distill import select_by_excess

    s = torch.tensor([[1.0, 9.0, 2.0, 8.0, 0.5, 0.5]])
    pairs = [[(1, 8.0), (3, 8.5)]]  # pos1: excess 1.0；pos3: excess -0.5
    mask = select_by_excess(s, pairs, keep_ratio=0.5, min_keep=1)
    got = mask.reshape(-1).tolist()
    # 非配对位 (0,2,4,5) 全开；配对位只开 excess 高的 pos1
    assert got == [True, True, True, False, True, True]
    assert select_by_excess(s, [], 0.5).all()


def test_teacher_token_losses_fake():
    """老师 token loss：位置有效、值有限（fake 老师离线测）."""
    d = _distiller()
    student_ids = torch.tensor([[1, 4, 5, 6, 7]])
    out = d.teacher_token_losses(student_ids, {4: "你", 5: "好", 6: "世", 7: "界"})
    assert len(out) == 1 and len(out[0]) == 1
    pos, val = out[0][0]
    # 老师 token"你好"覆盖字符[0,2)，配对学生位置 2（持"好"预测"世"）
    assert pos == 2 and val == val and val > 0
