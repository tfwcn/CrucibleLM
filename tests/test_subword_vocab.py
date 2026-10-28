"""追加式子词表单测 — 最长匹配、旧 id 稳定、round-trip、约束解码兼容（离线）."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config  # noqa: E402
from src.llm.local.infer import SimpleTokenizer, extend_vocab_and_model  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402

CHARS = (list("的是一不了我在有和人这中大为上个国摸鱼資{tool_cal\",:}[]")
         + list("abcdefghijklmnopqrstuvwxyz0123456789") + [" "])


def _tok(subwords=()):
    """字符表 + 可选子词的分词器（id 从 4 起）."""
    tok = SimpleTokenizer(8192)
    tok._chars = list(CHARS) + list(subwords)
    tok._ids = {s: i + 4 for i, s in enumerate(tok._chars)}
    return tok


def _char_encode(tok: SimpleTokenizer, text: str) -> list[int]:
    """参照实现：逐字符编码（老口径，用于证明无子词时逐位一致）."""
    return [tok._ids.get(ch, tok.UNK) for ch in text]


# ---------------------------------------------------------------- 分词器
def test_no_subwords_bitwise_same_as_chars():
    """没有多字符项时，encode 与逐字符实现逐位一致（默认路径不改行为）."""
    tok = _tok()
    for text in ["摸鱼是打工人的续命良药", '{"name":"shell_exec"}', "abc 123"]:
        assert tok.encode(text, add_bos=False) == _char_encode(tok, text)


def test_longest_match_prefers_longest():
    """最长优先：同时命中 "ab"/"abc" 时选 "abc"."""
    tok = _tok(["ab", "abc"])
    ids = tok.encode("xabc", add_bos=False)
    assert ids[1:] == [tok._ids["abc"]]


def test_subword_fallback_to_char():
    """子词没覆盖到的位置退回单字符（不会吞字符也不会错位）."""
    tok = _tok(["摸鱼"])
    ids = tok.encode("摸鱼是", add_bos=False)
    assert ids == [tok._ids["摸鱼"], tok._ids["是"]]


def test_unknown_char_is_unk():
    """表外字符 → UNK（与老行为一致，不越界）."""
    tok = _tok(["摸鱼"])
    ids = tok.encode("摸魚", add_bos=False)     # 第二个字是繁体，不在表内
    assert ids[0] == tok._ids["摸"]
    assert ids[1] == SimpleTokenizer.UNK


def test_roundtrip_text():
    """decode(encode(x)) == x（中文 / 英文 / JSON 混排）."""
    tok = _tok(['{"', '":"', '"}', "tool", "call"])
    for text in ["摸鱼是一不了我在有和人这中大为上个国",
                 '{"name":"tool","args":{}}',
                 "call tool 123"]:
        assert tok.decode(tok.encode(text, add_bos=False)) == text


def test_bos_prepended():
    """add_bos 语义不变（默认加 BOS，add_bos=False 不加）."""
    tok = _tok(["摸鱼"])
    assert tok.encode("摸鱼")[0] == SimpleTokenizer.BOS
    assert tok.encode("摸鱼", add_bos=False)[0] != SimpleTokenizer.BOS


def test_empty_surface_does_not_hang():
    """_chars 里混入空串时不能死循环（空串会匹配长度 0）."""
    tok = _tok()
    tok._chars = list(CHARS) + ["", "摸鱼"]
    tok._ids = {s: i + 4 for i, s in enumerate(tok._chars)}
    tok._meta_key = None
    ids = tok.encode("摸鱼", add_bos=False)
    assert len(ids) == 1


def test_external_chars_rewrite_takes_effect():
    """旧脚本直接改 _chars/_ids 后自动生效（无需手动调任何方法）."""
    tok = SimpleTokenizer(8192)
    tok._chars = list(CHARS)
    tok._ids = {s: i + 4 for i, s in enumerate(tok._chars)}
    assert tok._multi is False
    tok._chars = list(CHARS) + ["摸鱼"]
    tok._ids = {s: i + 4 for i, s in enumerate(tok._chars)}
    assert len(tok.encode("摸鱼", add_bos=False)) == 1     # 自动切到最长匹配


def test_max_len_after_sync():
    """最长匹配长度按最长表面串计算."""
    tok = _tok(['{"', '":"', "toolbox_call"])
    tok._sync_meta()
    assert tok._max_len == len("toolbox_call")
    assert tok._multi is True


# ---------------------------------------------------------------- 扩词落位
def _model(vocab_size=128):
    """极小模型（CPU）用于扩词落位测试."""
    cfg = tiny_test_config()
    cfg.vocab_size = vocab_size
    return TinyLLM(cfg)


def test_extend_keeps_old_rows_bitwise():
    """追加子词：旧 embedding 行逐位不变，新行随机非零，id 从末尾接."""
    tok = _tok()
    model = _model()
    model.eval()
    old_n = model.embed.weight.shape[0]
    old_chars = list(tok._chars)
    used = len(old_chars) + 4                 # 已用行 = 特殊 token + 字符表
    before = model.embed.weight.detach().clone()

    n, resized = extend_vocab_and_model(tok, model, ["摸鱼", '{"'])
    assert n == 2 and resized is False
    assert model.embed.weight.shape[0] == old_n            # 复用空行，形状不变
    # 已用行逐位不变（预留容量的空行会被重初始化，这是设计如此）
    assert torch.equal(model.embed.weight[:used], before[:used])
    # 新 id 接在旧的后面，且顺序与输入一致
    assert tok._ids["摸鱼"] == len(old_chars) + 4
    assert tok._ids['{"'] == len(old_chars) + 5
    assert torch.count_nonzero(model.embed.weight[len(old_chars) + 4]) > 0


def test_extend_dedupes_existing():
    """已在表里的项不重复追加（重复调用安全）."""
    tok = _tok(["摸鱼"])
    model = _model()
    n, resized = extend_vocab_and_model(tok, model, ["摸鱼", "摸"])
    assert n == 0 and resized is False


def test_extend_tied_head_stays_tied():
    """追加时若 lm_head 与 embed 共享权重，扩行后必须仍然共享（不能断链）."""
    tok = _tok()
    model = _model()
    assert model.lm_head.weight is model.embed.weight
    old_n = model.embed.weight.shape[0]
    extra = [f"sub{i}" for i in range(old_n)]        # 故意超出容量 → 扩行
    n, resized = extend_vocab_and_model(tok, model, extra)
    assert n == len(extra) and resized is True
    assert model.lm_head.weight is model.embed.weight


def test_extend_grows_rows_when_full():
    """容量不足时扩行，返回 resized=True（调用方需新开动量）."""
    tok = _tok()
    model = _model(vocab_size=len(CHARS) + 4 + 1)   # 只剩 1 行余量，装不下 2 个
    old_n = model.embed.weight.shape[0]
    n, resized = extend_vocab_and_model(tok, model, ["摸鱼", '{"'])
    assert n == 2 and resized is True
    # 扩行 = 旧行数 + 新增数（保证新 id 全部落在范围内）
    assert model.embed.weight.shape[0] == old_n + n


# ---------------------------------------------------------------- 与约束解码
def test_constraint_accepts_multi_char_tokens():
    """约束解码必须能吃掉多字符 token（structured.py 按整串校验）."""
    from src.llm.local.structured import JsonGrammar, JsonGrammarConstraint

    vocab = ["", "", "", ""] + ascii_chars() + ['{"', '":"', '"}', '"name"']
    vocab = vocab[:4] + vocab[4:]                     # id 0-3 特殊
    tokens = [""] * (4 + len(vocab) - 4)
    for i, s in enumerate(vocab):
        tokens[i] = s
    g = JsonGrammar({"type": "object", "properties": {"name": {"type": "string"}},
                     "required": ["name"], "additionalProperties": False})
    c = JsonGrammarConstraint(g, tokens, eos_id=2, vocab_size=len(tokens))
    tid = tokens.index('{"')
    assert c._can_take('{"') is True
    c.accept(tid)
    assert c.text == '{"'
    assert c.state is not None


def ascii_chars() -> list[str]:
    """可打印 ASCII（给约束解码测试当字符级词表）."""
    return [""] + [chr(i) for i in range(32, 127)]


# ---------------------------------------------------------------- 压缩效果
def test_subword_reduces_tokens_on_json():
    """回归：固定子词表必须让工具调用串变短（否则这套机制没意义）."""
    probe = '{"name":"tool","arguments":{"path":"data/x"}}'
    plain = _tok()
    withsub = _tok(['{"', '":"', '"}', '",', 'name', 'arguments', 'tool', 'data'])
    assert len(withsub.encode(probe, add_bos=False)) < len(plain.encode(probe, add_bos=False))
