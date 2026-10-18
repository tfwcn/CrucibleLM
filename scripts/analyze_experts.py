"""MoE 专家可解释性分析（只读，CPU 可跑）：SwiGLU 双负区间 + 跨 checkpoint 生长追踪.

用法：
  # 1) 单 checkpoint：统计 SwiGLU 四种符号组合（双负 echo 占比）
  python scripts/analyze_experts.py --mode regime \\
      --src data/llm-ckpt/model.pt --vocab data/llm-ckpt/vocab.json --data data/hq-zh

  # 2) 多 checkpoint：追踪专家神经元的生长（均值/峰值随训练变化）
  python scripts/analyze_experts.py --mode growth \\
      --ckpts data/llm-ckpt/ckpt-004800/model.pt,data/llm-ckpt/ckpt-005300/model.pt \\
      --vocab data/llm-ckpt/vocab.json --data data/hq-zh

只读权重，不碰训练。背景与结论见 docs/details/moe-interpretability.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="MoE 专家可解释性分析（SwiGLU 双负 / 生长追踪）")
    p.add_argument("--mode", choices=["regime", "growth", "both"], default="both")
    p.add_argument("--src", default="", help="单 checkpoint 权重 model.pt（regime/both）")
    p.add_argument("--ckpts", default="", help="逗号分隔的多个 model.pt（growth；缺省用 --src）")
    p.add_argument("--vocab", required=True, help="词表 vocab.json")
    p.add_argument("--data", required=True, help="语料目录（取前若干窗口）")
    p.add_argument("--seq-len", type=int, default=512, help="每窗口 token 数")
    p.add_argument("--batch", type=int, default=4, help="一次前向多少窗口")
    p.add_argument("--windows", type=int, default=64, help="总窗口数")
    p.add_argument("--layers", default="", help="只分析这些层（逗号分隔；空=全部）")
    p.add_argument("--top-frac", type=float, default=0.01,
                   help="最强激活段占比（双负在这段里的份额）")
    p.add_argument("--rel-threshold", type=float, default=1e-3,
                   help="休眠阈值（层均值相对倍数，生长追踪用）")
    p.add_argument("--out", default="", help="结果 JSON 输出路径（可选）")
    p.add_argument("--device", default="cpu", help="设备（默认 cpu）")
    return p.parse_args(argv)


def build_windows(tok, data_dir: str, seq_len: int, want: int) -> list[list[int]]:
    """从语料目录顺序取 token，切成 want 个定长窗口."""
    from src.llm.local.data import iter_local_texts

    need = want * seq_len
    toks: list[int] = []
    for text in iter_local_texts(data_dir):
        toks.extend(tok.encode(text, add_bos=False))
        if len(toks) >= need:
            break
    if len(toks) < seq_len:
        raise RuntimeError(f"语料不足：仅 {len(toks)} token，至少需 {seq_len}")
    n = min(want, len(toks) // seq_len)
    return [toks[i * seq_len:(i + 1) * seq_len] for i in range(n)]


def to_batches(windows: list[list[int]], batch: int, device):
    """窗口按 batch 分组为张量序列."""
    import torch

    out = []
    for i in range(0, len(windows) - batch + 1, batch):
        out.append(torch.tensor(windows[i:i + batch], dtype=torch.long,
                                device=device))
    return out


def load_model(src: str, grad_ckpt: bool = False):
    """按 checkpoint 的保存配置建模型并严格载入权重（CPU）."""
    import torch

    from src.llm.local.config import (SmallLLMConfig, load_saved_config,
                                      sync_csa_reuse_from_checkpoint)
    from src.llm.local.model import TinyLLM

    config = load_saved_config(src) or SmallLLMConfig()
    # 旧 checkpoint 的 hparams 可能没有 csa 字段：按来源对齐，避免静默换架构
    sync_csa_reuse_from_checkpoint(config, src)
    config.grad_ckpt = grad_ckpt
    model = TinyLLM(config)
    model.load_state_dict(torch.load(src, map_location="cpu"))
    return model.eval(), config


def parse_layers(spec: str, n_layers: int) -> set[int] | None:
    """解析 --layers；空返回 None（全部）."""
    if not spec.strip():
        return None
    return {int(x) for x in spec.split(",") if x.strip()}


def print_regime_report(stats: dict, layers: set[int] | None) -> None:
    """打印逐层 SwiGLU 符号区间汇总 + 双负占比最高的专家."""
    by_layer: dict[int, list[dict]] = {}
    for (li, _kind, _ei), s in stats.items():
        if layers is not None and li not in layers:
            continue
        by_layer.setdefault(li, []).append(s)

    def _avg(rows, key):
        vals = [r[key] for r in rows if r["n_neurons"] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n== SwiGLU 双负区间：逐层汇总 ==")
    print(f"{'层':>3} {'专家':>4} {'双负占正激活':>12} {'双负占最强段':>12}"
          f" {'echo主导神经元':>13} {'最强即双负':>10}")
    tot = []
    for li in sorted(by_layer):
        rows = by_layer[li]
        tot.extend(rows)
        print(f"{li:>3} {len(rows):>4} {_avg(rows, 'dn_share'):>12.4f}"
              f" {_avg(rows, 'dn_top'):>12.4f} {_avg(rows, 'frac_echo'):>13.4f}"
              f" {_avg(rows, 'frac_argmax_dn'):>10.4f}")
    print(f"{'全':>3} {len(tot):>4} {_avg(tot, 'dn_share'):>12.4f}"
          f" {_avg(tot, 'dn_top'):>12.4f} {_avg(tot, 'frac_echo'):>13.4f}"
          f" {_avg(tot, 'frac_argmax_dn'):>10.4f}")

    ranked = sorted((s for s in tot if s["n_neurons"] > 0),
                    key=lambda s: s["dn_top"], reverse=True)[:5]
    print("\n双负 echo 最重的 5 个专家（按最强段双负占比）:")
    for s in ranked:
        print(f"  层{s['li']:>2} {s['kind']:<6}#{s['ei']:<3}"
              f" 双负占最强段 {s['dn_top']:.3f}  双负占全体 {s['dn_share']:.3f}"
              f"  echo神经元 {s['frac_echo']:.3f}  正激活数 {s['n_pos']}")


def print_growth_report(profiles: dict[str, dict], names: list[str],
                        rel_threshold: float) -> None:
    """打印各 checkpoint 的激活画像规模 + 首末配对（休眠/生长转移与相关性）."""
    import torch

    print("\n== 专家神经元生长追踪 ==")
    print(f"{'checkpoint':<44} {'神经元数':>8} {'激活均值':>10} {'激活峰值':>10}")
    for name in names:
        prof = profiles[name]
        means = torch.cat([v["mean"] for v in prof.values()])
        peaks = torch.cat([v["peak"] for v in prof.values()])
        print(f"{name:<44} {means.numel():>8} {float(means.mean()):>10.4f}"
              f" {float(peaks.mean()):>10.4f}")

    if len(names) < 2:
        print("（只有 1 个 checkpoint，跳过配对比较）")
        return

    a, b = profiles[names[0]], profiles[names[-1]]

    def _dormant_set(prof):
        """按层均值相对阈值判休眠，返回整层归一化后的 (key,j) 集合."""
        out = set()
        for key, v in prof.items():
            mean = v["mean"]
            lm = float(mean.mean())
            thr = lm * rel_threshold
            idx = torch.where(mean < thr)[0]
            for j in idx.tolist():
                out.add((key, j))
        return out

    da, db = _dormant_set(a), _dormant_set(b)
    print(f"\n休眠神经元数：{names[0]} {len(da)} → {names[-1]} {len(db)}")
    print(f"  末态休眠里、起点活跃（新变休眠）: {len(db - da)}")
    print(f"  起点休眠里、末态活跃（自己长起来）: {len(da - db)}")

    # 逐层归一化后，首末的神经元激活强度相关性（1=稳定，0=完全换人）
    xs, ys = [], []
    for key in a:
        if key not in b:
            continue
        ma, mb = a[key]["mean"], b[key]["mean"]
        sa, sb = float(ma.mean()), float(mb.mean())
        if sa <= 0 or sb <= 0:
            continue
        xs.extend((ma / sa).tolist())
        ys.extend((mb / sb).tolist())
    if len(xs) > 1:
        import statistics as st
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / len(xs)
        sx = st.pstdev(xs)
        sy = st.pstdev(ys)
        r = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
        print(f"\n逐层归一化激活强度、首末相关 r = {r:+.3f}"
              f"（1=神经元职责稳定，越低=训练中换得越多）")


def main(argv=None) -> int:
    """主流程."""
    import torch

    from src.llm.local.infer import SimpleTokenizer
    from src.llm.local.interpret import expert_activation_profile, expert_sign_regimes

    args = parse_args(argv)
    torch.manual_seed(0)
    device = torch.device(args.device)

    tok = SimpleTokenizer(8192)
    with open(args.vocab, encoding="utf-8") as f:
        tok._chars = json.load(f)
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    tok.vocab_size = len(tok._chars) + 4

    windows = build_windows(tok, args.data, args.seq_len, args.windows)
    batches = to_batches(windows, args.batch, device)
    print(f"语料窗口：{len(windows)} × {args.seq_len} token（{args.data}）", flush=True)

    srcs = [args.src] if args.src else []
    ckpts = [c for c in args.ckpts.split(",") if c.strip()]
    result: dict = {"mode": args.mode}

    if args.mode in ("regime", "both") and srcs:
        model, config = load_model(srcs[0])
        model = model.to(device)
        layers = parse_layers(args.layers, config.n_layers)
        print(f"分析权重：{srcs[0]}（{config.n_layers} 层，"
              f"{config.n_experts} 专家，csa_reuse_every={config.csa_reuse_every}）",
              flush=True)
        stats = expert_sign_regimes(model, batches, layers, args.top_frac)
        print_regime_report(stats, layers)
        result["regime"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in stats.items()}

    if args.mode in ("growth", "both"):
        ckpt_list = ckpts or srcs
        if ckpt_list:
            profiles: dict[str, dict] = {}
            for src in ckpt_list:
                model, config = load_model(src)
                model = model.to(device)
                layers = parse_layers(args.layers, config.n_layers)
                profiles[src] = expert_activation_profile(model, batches, layers)
                del model
            print_growth_report(profiles, ckpt_list, args.rel_threshold)
            result["growth"] = {
                src: {f"{k[0]}|{k[1]}|{k[2]}": {
                    "mean": v["mean"].tolist(), "peak": v["peak"].tolist(),
                    "p99": v["p99"].tolist()} for k, v in prof.items()}
                for src, prof in profiles.items()}

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已写：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
