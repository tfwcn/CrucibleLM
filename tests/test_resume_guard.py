"""续训一致性守卫单测 — 比对口径、浮点容差、旧 ckpt 兼容（离线，无 torch 依赖）."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.resume import (  # noqa: E402
    CRITICAL_KEYS,
    diff_resume,
    format_block,
    load_prev_hparams,
    optim_state_missing,
)


def _args(**kw):
    """构造最小 args 命名空间（缺失的键按 None 处理，模拟旧 ckpt 无记录）."""
    base = {
        "phase": "pretrain",
        "max_steps": 10000,
        "lr": 3e-4,
        "warmup": 50,
        "muon_lr": 0.02,
        "optimizer": "adamw",
        "seq_len": 2048,
        "batch": 4,
        "accum": 8,
        "data": "local",
        "local_path": "data/hq-zh:3,data/ultra-zh:1",
        "pretrain_mix": "hq:1",
        "sft_mix": "belle:1",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_same_recipe_no_diff():
    """配方完全一致（含浮点等价写法）→ 零差异."""
    prev = dict(vars(_args()))
    prev["lr"] = 0.0003          # JSON 落盘后的等价值
    assert diff_resume(prev, _args(lr=3e-4)) == []


def test_max_steps_change_detected():
    """延长 max_steps = 重算 cosine，必须报出来（base run 3210 步的原坑）."""
    diffs = diff_resume({"max_steps": 10000}, _args(max_steps=14000))
    assert diffs == [("max_steps", 10000, 14000)]


def test_local_path_change_detected():
    """换语料 = 换 holdout，val_loss 不可比（base run 5310 步的原坑）."""
    diffs = diff_resume(
        {"local_path": "data/hq-zh"}, _args(local_path="data/hq-zh:3,data/ultra-zh:1"))
    assert diffs == [("local_path", "data/hq-zh",
                      "data/hq-zh:3,data/ultra-zh:1")]


def test_missing_keys_skipped():
    """旧 ckpt 的 hparams 缺字段时跳过，不误报（向后兼容）."""
    assert diff_resume({"phase": "pretrain"}, _args()) == []


def test_empty_prev_no_diff():
    """空字典 → 无差异（无记录即无从比对）."""
    assert diff_resume({}, _args()) == []


def test_diff_order_follows_critical_keys():
    """多项差异按 CRITICAL_KEYS 顺序输出（便于对照报错文本）."""
    prev = {"phase": "pretrain", "max_steps": 10000, "lr": 3e-4}
    diffs = diff_resume(prev, _args(phase="sft", max_steps=2000, lr=1e-4))
    assert [d[0] for d in diffs] == ["phase", "max_steps", "lr"]
    assert [d[0] for d in diffs] == [k for k in CRITICAL_KEYS if k in
                                     {"phase", "max_steps", "lr"}]


def test_bool_not_confused_with_int():
    """布尔与整数不混同（True 不等于 1，避免开关误判相等）."""
    diffs = diff_resume({"phase": True}, _args(phase=1))
    assert [d[0] for d in diffs] == ["phase"]


def test_none_and_missing_equivalent():
    """None 与缺省同样对待：旧 run 没记 muon_lr 时不该报差异."""
    assert diff_resume({"muon_lr": None}, _args(muon_lr=None)) == []


def test_format_block_mentions_impact_and_fix():
    """报错文本必须给出影响与三种修法，否则用户只会照旧加参数硬闯."""
    text = format_block([("max_steps", 10000, 14000)], optim_lost=True)
    assert "max_steps" in text
    assert "cosine" in text
    assert "optim.pt" in text
    assert "--force-resume" in text
    assert "--ckpt-dir" in text


def test_format_block_only_optim_lost():
    """只有动量丢失（配方没变）时也要能渲染."""
    text = format_block([], optim_lost=True)
    assert "optim.pt" in text
    assert "max_steps" not in text


def test_load_prev_hparams_roundtrip(tmp_path):
    """能读回上次落盘的 hparams；缺失/坏文件返回 None 不抛."""
    (tmp_path / "hparams.json").write_text(
        json.dumps({"max_steps": 10000}), encoding="utf-8")
    assert load_prev_hparams(tmp_path) == {"max_steps": 10000}
    assert load_prev_hparams(tmp_path / "nope") is None
    (tmp_path / "hparams.json").write_text("{不是 json", encoding="utf-8")
    assert load_prev_hparams(tmp_path) is None


def test_optim_state_missing(tmp_path):
    """optim.pt 存在性判定（缺失 = 动量归零）."""
    assert optim_state_missing(tmp_path) is True
    (tmp_path / "optim.pt").write_bytes(b"")
    assert optim_state_missing(tmp_path) is False


def test_real_base_log_recipe_would_be_caught():
    """回归：用真实 base run 前后两次启动的配方，守卫必须命中.

    3210 步那次启动（max_steps 10000→14000），与 5310 步那次
    （max_steps 14000→10000 + 换 local_path）都必须被拦下。
    """
    prev = {"phase": "pretrain", "max_steps": 10000, "lr": 3e-4,
            "seq_len": 2048, "batch": 4, "accum": 8, "data": "local",
            "local_path": "data/hq-zh", "pretrain_mix": "hq:1"}
    at_3210 = _args(max_steps=14000, local_path="data/hq-zh")
    assert [d[0] for d in diff_resume(prev, at_3210)] == ["max_steps"]

    prev["max_steps"] = 14000
    at_5310 = _args(max_steps=10000,
                    local_path="data/hq-zh:3,data/ultra-zh:1")
    assert [d[0] for d in diff_resume(prev, at_5310)] == ["max_steps",
                                                         "local_path"]
