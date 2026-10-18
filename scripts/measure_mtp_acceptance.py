"""离线测量 MTP 头接受率 — 决定是否值得开投机解码（P0 前置门槛）.

原理：MTP 头 `mtp(h_i)` 训练目标是预测 t+2。投机解码把它的 argmax 当作草稿，
与主干自身在 t+2 处的贪心（`argmax(main(h_{i+1}))`）比对：命中即"草稿正确可接受"。
本脚本在验证文本上统计这个命中率，接受率越高，投机解码越值得开。

用法：
    python scripts/measure_mtp_acceptance.py --model data/llm-ckpt --text data/val.txt
    python scripts/measure_mtp_acceptance.py --preset tiny   # 无权重，仅冒烟（随机初始化）
    # 带 memory/retro/hyper 的 SFT checkpoint：用 --run-yaml 复用其训练配置
    python scripts/measure_mtp_acceptance.py --model data/llm-sft7/best/model \
        --run-yaml data/llm-sft7/run.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import (
    SmallLLMConfig,
    sync_csa_reuse_from_checkpoint,
    tiny_test_config,
)
from src.llm.local.infer import LocalChatBackend
from src.llm.local.model import TinyLLM


def apply_run_yaml(cfg: "SmallLLMConfig", path: str) -> None:
    """把训练 run.yaml 里的结构开关套到配置上（简单 key: value，够用）.

    只认与 state_dict 结构相关的键（memory/retro/hyper），其余忽略——载权重
    只需结构一致，训练超参无关。
    """
    kv: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    # enable_* 是布尔；memory_every 缺省取训练脚本默认 4
    if kv.get("enable_memory", "false").lower() == "true":
        cfg.memory_every = int(kv.get("memory_every", 4))
        if kv.get("memory_slots"):
            cfg.memory_slots = int(kv["memory_slots"])
        if kv.get("memory_topk"):
            cfg.memory_topk = int(kv["memory_topk"])
    if kv.get("enable_retro", "false").lower() == "true":
        cfg.retro_enabled = True
        cfg.retro_every = int(kv.get("retro_every", 0))
        if kv.get("retro_len"):
            cfg.retro_chunk_len = int(kv["retro_len"])
    if kv.get("hyper_streams"):
        cfg.hyper_streams = int(kv["hyper_streams"])


def measure(model: TinyLLM, ids: torch.Tensor) -> dict:
    """在 id 序列上统计 MTP 对 t+2 的接受率（整体 + 三分位分布）."""
    model.eval()
    if model.mtp is None:
        raise ValueError("模型无 MTP 头（mtp_depth<=0），无法测量")
    with torch.no_grad():
        h = model.encode_full_hidden(ids)          # (1, T, d)，final_norm 后
        main_logits = model.lm_head(h)             # 位置 i 预测 t+1
        mtp_logits = model.lm_head(model.mtp(h))   # 位置 i 预测 t+2（训练口径）
    T = h.shape[1]
    if T < 3:
        raise ValueError("序列太短（<3），无法评估 t+2")
    # 主干在 t+2 的权威贪心：由 h_{i+1} 的 argmax 给出（i 从 0 到 T-3）
    main_next2 = main_logits[0, 1:T - 1].argmax(-1)   # 预测 i+2（i=0..T-3）
    mtp_draft = mtp_logits[0, 0:T - 2].argmax(-1)     # 草稿 i+2
    hit = (main_next2 == mtp_draft)
    n = hit.numel()
    acc = float(hit.float().mean())
    third = max(1, n // 3)
    head = float(hit[:third].float().mean())
    mid = float(hit[third:2 * third].float().mean())
    tail = float(hit[2 * third:].float().mean())
    # 建议窗口：接受率^k 仍 > 0.1 的最大 k（越靠后预期收益越低）
    best_k = 1
    for k in range(1, 9):
        if acc ** k > 0.1:
            best_k = k
    return {"n": n, "accept_rate": round(acc, 4),
            "head": round(head, 4), "mid": round(mid, 4), "tail": round(tail, 4),
            "suggested_window": best_k}


def main() -> None:
    """解析参数 → 载模型/文本 → 打印接受率报告."""
    ap = argparse.ArgumentParser(description="测量 MTP 头对 t+2 的接受率")
    ap.add_argument("--model", type=str, default="",
                    help="模型前缀（LocalChatBackend.load 的 path，如 data/llm-ckpt）")
    ap.add_argument("--text", type=str, default="",
                    help="验证文本文件（UTF-8）；缺省用内置样例句")
    ap.add_argument("--preset", type=str, default="tiny",
                    choices=["tiny", "base"], help="无 --model 时的随机配置")
    ap.add_argument("--max-chars", type=int, default=8000, help="最多评估多少字符")
    ap.add_argument("--dtype", type=str, default="", help="fp16/bf16（可选，省显存）")
    ap.add_argument("--run-yaml", type=str, default="",
                    help="训练 run.yaml；用于复用 memory/retro/hyper 结构开关载权重")
    args = ap.parse_args()

    if args.model:
        cfg = SmallLLMConfig()
        # 按权重目录对齐 CSA2（该开关不改 state_dict 键，装错不报错只会静默变语义）
        sync_csa_reuse_from_checkpoint(cfg, args.model)
        if args.run_yaml:
            apply_run_yaml(cfg, args.run_yaml)
        backend = LocalChatBackend.load(
            args.model, cfg, dtype=args.dtype or None)
        model, tok = backend.model, backend.tokenizer
    else:
        cfg = tiny_test_config() if args.preset == "tiny" else SmallLLMConfig()
        torch.manual_seed(0)
        model = TinyLLM(cfg)
        from src.llm.local.infer import SimpleTokenizer

        tok = SimpleTokenizer(cfg.vocab_size)
        tok.fit(["摸鱼使我快乐，上班时间合理休息放松。", "你好世界 hello world"])

    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")[: args.max_chars]
    else:
        text = ("摸鱼是指在上班时间合理休息放松，保持工作与生活的平衡。"
                "现代职场压力大，适度摸鱼有助于恢复精力，提高效率。") * 20
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    ids = ids[:, : cfg.max_seq_len]

    report = measure(model, ids)
    print("== MTP 接受率报告 ==")
    for k, v in report.items():
        print(f"  {k}: {v}")
    thr = 0.5
    verdict = "建议开启投机解码（接受率达标）" if report["accept_rate"] > thr \
        else f"暂不建议：接受率 {report['accept_rate']} ≤ 阈值 {thr}，先调 mtp_loss_weight"
    print(f"  结论: {verdict}")


if __name__ == "__main__":
    main()
