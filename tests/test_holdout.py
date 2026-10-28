"""固定评测集单测 — 落盘/复用/拒绝坏文件，以及 collect_holdout 真的会复用（离线）."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.llm.local.holdout import load_holdout, save_holdout  # noqa: E402

FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "llm-corpus")


# ---------------------------------------------------------------- 存取
def test_roundtrip_pretrain(tmp_path):
    """预训练评测集（str 列表）原样取回."""
    p = tmp_path / "h.json"
    save_holdout(p, ["甲", "乙", "丙"], "pretrain")
    assert load_holdout(p, "pretrain") == ["甲", "乙", "丙"]


def test_roundtrip_sft_keeps_tuples(tmp_path):
    """SFT 评测集存 list、取回必须是 tuple（下游按 (prompt, response) 解包）."""
    p = tmp_path / "h.json"
    save_holdout(p, [("<用户>\n问\n<助手>\n", "答"), ("p2", "a2")], "sft")
    got = load_holdout(p, "sft")
    assert got == [("<用户>\n问\n<助手>\n", "答"), ("p2", "a2")]
    assert all(isinstance(it, tuple) for it in got)


def test_missing_file_returns_none(tmp_path):
    """文件不存在 → None（调用方据此落盘）."""
    assert load_holdout(tmp_path / "nope.json", "pretrain") is None


def test_corrupt_file_returns_none(tmp_path):
    """坏 JSON → None，不抛（不能因为评测集坏了就起不了训练）."""
    p = tmp_path / "h.json"
    p.write_text("{不是 json", encoding="utf-8")
    assert load_holdout(p, "pretrain") is None


def test_phase_mismatch_returns_none(tmp_path):
    """pretrain 的评测集不能喂给 sft（样本形状不同）."""
    p = tmp_path / "h.json"
    save_holdout(p, ["纯文本"], "pretrain")
    assert load_holdout(p, "sft") is None


def test_version_mismatch_returns_none(tmp_path):
    """格式版本不符 → 当作没有，重新生成."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"version": 999, "phase": "pretrain",
                             "items": ["x"]}), encoding="utf-8")
    assert load_holdout(p, "pretrain") is None


def test_empty_items_returns_none(tmp_path):
    """空列表不能当有效评测集（否则 eval 拿不到任何 batch）."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"version": 1, "phase": "pretrain", "items": []}),
                 encoding="utf-8")
    assert load_holdout(p, "pretrain") is None


def test_bad_sft_item_invalidates_whole_file(tmp_path):
    """SFT 里混进形状不对的项 → 整份作废（宁可重切，不要半份错对）."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"version": 1, "phase": "sft",
                             "items": [["只有一项"]]}), encoding="utf-8")
    assert load_holdout(p, "sft") is None


def test_save_creates_parent_dirs(tmp_path):
    """父目录不存在时自动创建（首次跑就指定 ckpt 子路径的常见用法）."""
    p = tmp_path / "deep" / "dir" / "h.json"
    save_holdout(p, ["x"], "pretrain")
    assert p.exists()


# ---------------------------------------------------------------- 训练流程接线
def _args(**kw):
    """collect_holdout 需要的最小参数集."""
    base = {
        "data": "local",
        "phase": "pretrain",
        "local_path": FIXTURE,
        "pretrain_mix": "hq:1",
        "pretrain_min_score": 0.0,
        "holdout_file": "",
        "seq_len": 32,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_collect_holdout_reuses_saved_file(tmp_path):
    """首次调用落盘，二次调用必须复用（而不是按语料重新切）."""
    flow = pytest.importorskip("train_local_llm")
    p = tmp_path / "h.json"

    first, _ = flow.collect_holdout(_args(holdout_file=str(p)), 0)
    assert p.exists()

    # 篡改首项做标记：复用成功则标记会被读回来
    data = json.loads(p.read_text(encoding="utf-8"))
    data["items"][0] = "★复用标记★"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    second, _ = flow.collect_holdout(_args(holdout_file=str(p)), 0)
    assert second[0] == "★复用标记★"
    assert len(second) == len(first)


def test_collect_holdout_unchanged_without_flag():
    """不传 --holdout-file 时行为完全不变（默认关，逐位兼容旧流程）."""
    flow = pytest.importorskip("train_local_llm")
    holdout, _ = flow.collect_holdout(_args(), 0)
    assert holdout and all(isinstance(t, str) for t in holdout)


def test_collect_holdout_sft_roundtrip(tmp_path):
    """SFT 途径：落盘后复用应拿到 (prompt, response) 元组."""
    flow = pytest.importorskip("train_local_llm")
    p = tmp_path / "sft.json"
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    # Belle 风格：instruction / input / output
    rows = [{"instruction": f"问题{i}", "input": "", "output": f"答案{i}"}
            for i in range(4)]
    (sft_dir / "part.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8")

    a = _args(phase="sft", local_path=str(sft_dir), holdout_file=str(p))
    first, _ = flow.collect_holdout(a, 0)
    second, _ = flow.collect_holdout(a, 0)
    assert first == second
    assert isinstance(second[0], tuple) and len(second[0]) == 2
