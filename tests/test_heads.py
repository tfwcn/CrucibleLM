"""直觉头/RAG/抽取单测 — BM25、头训练、样本抽取（CPU 可跑）."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import extract_tool_choices as ext
from src.llm.local.config import tiny_test_config
from src.llm.local.heads import (
    IntuitionHead,
    calibrate_temperature,
    evaluate_head,
    extract_features,
    train_head,
)
from src.llm.local.infer import SimpleTokenizer
from src.llm.local.model import TinyLLM
from src.llm.local.retrieval import BM25Retriever


def test_bm25_ranking():
    """BM25：相关文档排第一，空查询/空库返回空，覆盖更新."""
    r = BM25Retriever()
    assert r.query("你好", k=3) == []
    r.add("a", "今天天气不错适合散步")
    r.add("b", "深度学习模型训练技巧")
    r.add("c", "今天天气很好有太阳")
    top = r.query("今天天气怎么样", k=2)
    assert [d for d, _ in top] == ["a", "c"] or top[0][0] in ("a", "c")
    assert top[0][1] >= top[1][1]
    r.add("a", "完全不相关的内容xyz")
    assert r.query("今天天气怎么样", k=1)[0][0] in ("b", "c")


def _tiny_backbone():
    """tiny backbone + 字符表（特征提取共用）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config()).eval()
    tok = SimpleTokenizer(256)
    tok.fit(["你好世界", "今天天气不错", "深度学习训练", "周末去爬山"])
    return m, tok


def test_extract_features_shapes():
    """特征提取：变长 batch 按真实长度 gather，不取 PAD 位."""
    m, tok = _tiny_backbone()
    feats = extract_features(m, tok, ["你好", "今天天气不错适合散步"], max_len=64)
    assert feats.shape == (2, 128)
    assert torch.isfinite(feats).all()


def test_train_head_binary():
    """二分类头：可分数据上 loss 下降、准确率上去."""
    torch.manual_seed(0)
    head = IntuitionHead(16, 2)
    feats = torch.cat([torch.randn(20, 16) - 2, torch.randn(20, 16) + 2])
    labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    curve = train_head(head, feats, labels, epochs=15, lr=1e-2)
    assert curve[-1] < curve[0]
    rep = evaluate_head(head, feats, labels)
    assert rep["acc"] > 0.9 and rep["ece"] < 0.3


def test_train_head_multiclass_and_calibration():
    """多分类头 + 温度校准返回网格内值."""
    torch.manual_seed(0)
    head = IntuitionHead(8, 3)
    feats = torch.randn(30, 8)
    labels = torch.randint(0, 3, (30,))
    train_head(head, feats, labels, epochs=3)
    t = calibrate_temperature(head, feats, labels)
    assert t in (0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0)
    rep = evaluate_head(head, feats, labels, temperature=t)
    assert set(rep) == {"acc", "nll", "ece", "latency_ms"}


def _make_session_db(path: Path) -> None:
    """造会话库：user→assistant(tool)→tool结果→assistant 回复."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, role TEXT, content TEXT,"
        " reasoning TEXT, tool_calls TEXT, tool_call_id TEXT, name TEXT,"
        " subtask_id TEXT, depth INTEGER, ts INTEGER)")
    calls = json.dumps([{"function": {"name": "shell_exec", "arguments": "{}"}}])
    rows = [
        ("user", "查一下北京天气", None),
        ("assistant", "", calls),
        ("tool", "晴转多云", None),
        ("assistant", "北京今天晴", None),
    ]
    for i, (role, content, tc) in enumerate(rows):
        conn.execute(
            "INSERT INTO messages (role, content, tool_calls, ts) VALUES (?,?,?,?)",
            (role, content, tc, i))
    conn.commit()
    conn.close()


def test_extract_pairs_from_db(tmp_path):
    """抽取：tool 调用转样本对，上下文含调用前消息."""
    db = tmp_path / "conversation.db"
    _make_session_db(db)
    pairs = ext.extract_pairs(ext.read_messages(db))
    assert len(pairs) == 1
    ctx, tool = pairs[0]
    assert tool == "shell_exec"
    assert "查一下北京天气" in ctx
    assert ext.read_messages(tmp_path / "nope.db") == []
    assert ext.iter_session_dbs(tmp_path / "nodir") == []
