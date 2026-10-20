"""DSpark 草稿器独立训练（DeepSeek-V4.1 §2.4.3 的"独立开班"）.

主干**冻结**、只训草稿器（3 层滑窗 + Markov 头 + 置信头）——与论文一致：
预训练后单独训 DSpark，后训练一起训但**梯度不回传主干**，让它始终对齐当前策略。

训练目标（三路，都是逐位置并行）：

```
base   = lm_head(drafter(ctx, h_last))                 # 半自回归的"半"（并行）
markov = base + U·E(t_{i-1})                           # 半自回归的"自回归"（一阶）
conf_i = σ(w·h_i)  ≈  P(主干接受第 i 个草稿 | 前 i-1 个已接受)
```

- CE(base) 保证"没有 Markov 也能猜个大概"（并行 base 是骨架）；
- CE(markov) 才是部署时真正用的 logits（依赖上一颗草稿 token）；
- BCE(conf, softmax(主干 logits)[草稿 token]) 教置信头预测**接受概率**，
  调度器靠它算前缀存活 `S_i = ∏ c_j`。用主干概率做软标签（而不是硬 0/1），
  既等于投机采样的接受概率，也让梯度更稳。

分词表必须与主干**同一张**（`<ckpt>/vocab.json`）；数据用同源本地文本目录。

用法：
    python scripts/train_dspark.py --model data/llm-exp/xxx/best --local-path data/corpus \\
        --steps 2000 --out data/llm-exp/xxx/dspark.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_local_llm import load_weights_overlap  # noqa: E402
from src.llm.local.config import SmallLLMConfig  # noqa: E402
from src.llm.local.data import (PackedBatcher, interleave_weighted,  # noqa: E402
                                iter_local_texts, pack_pretrain,
                                parse_local_mix)
from src.llm.local.infer import SimpleTokenizer  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402

DSPARK_KEYS = ("dspark_layers", "dspark_heads", "dspark_window", "dspark_depth",
               "dspark_markov_rank")


def parse_args(argv=None) -> argparse.Namespace:
    """命令行参数（数据/分词/草稿器结构/优化器）."""
    p = argparse.ArgumentParser(description="DSpark 草稿器独立训练")
    p.add_argument("--model", required=True,
                   help="主干 checkpoint 目录（含 model.pt + hparams.json + vocab.json）"
                        "或 model.pt 路径")
    p.add_argument("--vocab", default="", help="分词表（默认用 <ckpt>/vocab.json）")
    p.add_argument("--out", required=True, help="输出 dspark.pt 路径")
    p.add_argument("--local-path", default="",
                   help="训练文本目录（逗号分隔，支持 路径:权重）")
    p.add_argument("--seq-len", type=int, default=512, help="训练序列长度")
    p.add_argument("--batch", type=int, default=8, help="batch size")
    p.add_argument("--steps", type=int, default=2000, help="训练步数")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--warmup", type=int, default=100, help="线性预热步数")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW 权重衰减")
    p.add_argument("--conf-weight", type=float, default=0.5, help="置信头损失权重")
    p.add_argument("--markov-weight", type=float, default=1.0,
                   help="Markov 修正后 CE 的权重（base CE 权重恒 1）")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数")
    p.add_argument("--window", type=int, default=128, help="草稿器滑窗（论文 128）")
    p.add_argument("--depth", type=int, default=5, help="一次并行草稿位置数（论文 5）")
    p.add_argument("--layers", type=int, default=3, help="草稿器层数（论文 3）")
    p.add_argument("--heads", type=int, default=4, help="草稿器注意力头数")
    p.add_argument("--markov-rank", type=int, default=32, help="Markov 头低秩维")
    p.add_argument("--device", default="", help="cuda / cpu（默认自动）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--log-every", type=int, default=20, help="日志间隔（步）")
    p.add_argument("--save-every", type=int, default=500, help="存档间隔（步）")
    return p.parse_args(argv)


def load_backbone(args: argparse.Namespace) -> tuple[TinyLLM, SimpleTokenizer, Path]:
    """载入冻结主干 + 分词表（分词表必须与主干同源，否则训练信号全错）."""
    ckpt = Path(args.model)
    if ckpt.is_file():
        weight = ckpt
        roots = [ckpt.parent]
    else:
        roots = [ckpt] + ([ckpt.parent] if ckpt.parent != ckpt else [])
        weight = next((r / "model.pt" for r in roots if (r / "model.pt").exists()),
                      ckpt / "model.pt")
        if not weight.exists() and (ckpt / "best" / "model.pt").exists():
            weight = ckpt / "best" / "model.pt"
    # 训练脚本把 hparams.json/vocab.json 放在 run 根，权重放在 best/ 或根：
    # 两种布局都认（`--model <run>`、`--model <run>/best` 都能跑）
    def _find(name: str) -> Path | None:
        for r in roots:
            if (r / name).exists():
                return r / name
        return None

    ckpt_dir = weight.parent
    hp = _find("hparams.json")
    if hp is None:
        raise SystemExit(f"缺少 hparams.json（在 {roots} 下都没找到）")
    cfg_dict = json.loads(hp.read_text(encoding="utf-8")).get("config", {})
    cfg = SmallLLMConfig(**cfg_dict)
    # 草稿器结构由 CLI 决定（与主干结构无关），并挂到模型上
    cfg.dspark_draft = True
    # 草稿器结构默认取配置里的值（bench 等脚本不带这些 CLI 参数）
    for field, arg in (("dspark_layers", "layers"), ("dspark_heads", "heads"),
                       ("dspark_window", "window"), ("dspark_depth", "depth"),
                       ("dspark_markov_rank", "markov_rank")):
        if hasattr(args, arg):
            setattr(cfg, field, int(getattr(args, arg)))
    model = TinyLLM(cfg)
    # 草稿器是新训的，主干 ckpt 里本来就没有 `dspark.*`：先摘下来再叠权重，
    # 免得刷几十行"缺失，保留初始化"的假警告
    dr = model.dspark
    model.dspark = None
    load_weights_overlap(model, weight)
    model.dspark = dr
    vocab = Path(args.vocab) if getattr(args, "vocab", "") else _find("vocab.json")
    if vocab is None or not vocab.exists():
        raise SystemExit(f"缺少 vocab.json（在 {roots} 下都没找到，用 --vocab 指定）")
    tok = SimpleTokenizer(cfg.vocab_size)
    tok._chars = json.loads(vocab.read_text(encoding="utf-8"))
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    return model, tok, ckpt_dir


def build_stream(args: argparse.Namespace, tok: SimpleTokenizer):
    """本地文本 → packed 训练块流（与主干训练同一个打包口径）."""
    if not args.local_path:
        raise SystemExit("--local-path 必填（与主干同源的本地文本目录）")
    dirs = parse_local_mix(args.local_path)
    # 多源按权重交错（与主干训练的混料口径一致；单源时退化为直读）
    stream = interleave_weighted([iter_local_texts(p) for p, _ in dirs],
                                 [w for _, w in dirs])
    return pack_pretrain(stream, tok.encode, int(args.seq_len),
                         SimpleTokenizer.EOS)


def sample_positions(n: int, low: int, high: int, dev) -> list[int]:
    """每条样本一个随机预测位置 p（保证 left: window，right: depth）."""
    if high <= low:
        raise SystemExit(f"seq_len 太短：需要 > window+depth（low={low}, high={high}）")
    return torch.randint(low, high, (n,), device=dev).tolist()


def build_batch(model: TinyLLM, x: torch.Tensor, p: list[int],
                window: int, depth: int) -> dict:
    """按位置切出 (h_last, ctx_ids, tgt, prev_ids) 与主干的软接受标签.

    `h_last`/`tlogits` 全部 no_grad：草稿器的训练信号**不回流**主干（论文口径）。
    """
    b = x.shape[0]
    ctx = torch.stack([x[i, p[i] - window + 1: p[i] + 1] for i in range(b)])
    prev = torch.stack([x[i, p[i]: p[i] + depth] for i in range(b)])
    tgt = torch.stack([x[i, p[i] + 1: p[i] + 1 + depth] for i in range(b)])
    with torch.no_grad():
        h = model.encode_full_hidden(x)
        h_last = torch.stack([h[i, p[i]] for i in range(b)]).unsqueeze(1)
        h_tgt = torch.stack([h[i, p[i] + 1: p[i] + 1 + depth] for i in range(b)])
        tlogits = model.lm_head(h_tgt)          # (b, m, V) 只算 m 个位置
    return {"ctx": ctx, "prev": prev, "tgt": tgt, "h_last": h_last,
            "tlogits": tlogits}


def step_loss(model: TinyLLM, batch: dict, conf_weight: float,
              markov_weight: float) -> tuple[torch.Tensor, dict]:
    """一步损失：base CE + Markov CE + 置信头 BCE（三路都逐位置并行）."""
    dr = model.dspark
    hq, conf = dr(batch["h_last"], batch["ctx"], model.embed)
    base = model.lm_head(hq)                              # (b,m,V)
    markov = base + dr.markov_bias(batch["prev"]).to(base.dtype)
    tgt = batch["tgt"]
    v = base.shape[-1]
    ce_base = F.cross_entropy(base.reshape(-1, v), tgt.reshape(-1))
    ce_markov = F.cross_entropy(markov.reshape(-1, v), tgt.reshape(-1))
    # 软接受标签：主干在草稿 token 上的概率质量（= 投机采样的接受概率）
    with torch.no_grad():
        draft = markov.argmax(-1)                          # (b,m) 与部署一致
        soft = torch.softmax(batch["tlogits"].float(), dim=-1)
        accept = soft.gather(-1, draft.unsqueeze(-1)).squeeze(-1)
        hit = (draft == tgt).float()
    bce = F.binary_cross_entropy(conf.float().clamp(1e-6, 1 - 1e-6), accept)
    loss = ce_base + markov_weight * ce_markov + conf_weight * bce
    stats = {"loss": float(loss.detach()), "ce_base": float(ce_base.detach()),
             "ce_markov": float(ce_markov.detach()), "bce": float(bce.detach()),
             "hit1": float(hit[:, 0].mean()), "hit5": float(hit.mean()),
             "conf": float(conf.detach().float().mean()),
             "accept": float(accept.mean())}
    return loss, stats


def main(argv=None) -> int:
    """训练入口：冻结主干 → 训草稿器 → 落 dspark.pt."""
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ckpt_dir = load_backbone(args)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dr = model.dspark
    assert dr is not None
    # 冻结只针对主干：草稿器必须**重新打开**梯度（否则优化器拿到空参数表）
    for p in dr.parameters():
        p.requires_grad_(True)
    dr.train()
    n_par = sum(p.numel() for p in dr.parameters())
    print(f"主干已冻结（{ckpt_dir.name}）｜草稿器 {n_par/1e6:.1f}M："
          f"{args.layers} 层 / 窗 {args.window} / 并行 {args.depth} 位 / "
          f"Markov 秩 {args.markov_rank}", flush=True)

    stream = build_stream(args, tok)
    batches = PackedBatcher(stream, int(args.batch))
    opt = torch.optim.AdamW([p for p in dr.parameters() if p.requires_grad],
                            lr=float(args.lr), weight_decay=float(args.weight_decay))
    low = int(args.window) - 1
    high = int(args.seq_len) - int(args.depth) - 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ema: dict[str, float] = {}
    t0 = time.time()
    step = 0
    for batch in batches:
        step += 1
        x = batch[0] if isinstance(batch, tuple) else batch
        x = x.to(dev)
        if x.shape[1] < int(args.seq_len):
            continue
        ps = sample_positions(x.shape[0], low, high, x.device)
        data = build_batch(model, x, ps, int(args.window), int(args.depth))
        loss, stats = step_loss(model, data, float(args.conf_weight),
                                float(args.markov_weight))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dr.parameters(), float(args.grad_clip))
        # 线性预热
        lr = float(args.lr) * min(1.0, step / max(1, int(args.warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        for k, v in stats.items():
            ema[k] = v if k not in ema else 0.9 * ema[k] + 0.1 * v
        if step % max(1, int(args.log_every)) == 0:
            print(f"step {step}｜loss {ema.get('loss', 0):.4f}"
                  f"（base {ema.get('ce_base', 0):.4f} / markov {ema.get('ce_markov', 0):.4f}"
                  f" / bce {ema.get('bce', 0):.4f}）"
                  f"｜贪心命中 1位 {ema.get('hit1', 0):.3f} 全 {ema.get('hit5', 0):.3f}"
                  f"｜置信 {ema.get('conf', 0):.3f} vs 接受 {ema.get('accept', 0):.3f}"
                  f"｜{time.time() - t0:.0f}s", flush=True)
        if step % max(1, int(args.save_every)) == 0:
            save_dspark(out, cfg=model.config, dr=dr, step=step, stats=ema)
            print(f"已存 {out}（step {step}）", flush=True)
        if step >= int(args.steps):
            break
    save_dspark(out, cfg=model.config, dr=dr, step=step, stats=ema)
    print(f"训练结束：step={step}，草稿器权重 {out}"
          f"（主干保持冻结，未改动 {ckpt_dir}）", flush=True)
    return 0


def save_dspark(path: Path, cfg, dr, step: int, stats: dict) -> None:
    """只落草稿器（+ 结构快照），主干权重一个字节都不碰."""
    payload = {
        "config": {k: getattr(cfg, k) for k in DSPARK_KEYS if hasattr(cfg, k)},
        "state_dict": {k: v.detach().cpu() for k, v in dr.state_dict().items()},
        "step": int(step),
        "stats": {k: float(v) for k, v in stats.items()},
    }
    torch.save(payload, path)


def load_dspark(path, dr) -> dict:
    """把 dspark.pt 载回草稿器（返回元信息；结构不符会大声报错）."""
    payload = torch.load(path, map_location="cpu")
    dr.load_state_dict(payload["state_dict"], strict=True)
    return {k: v for k, v in payload.items() if k != "state_dict"}


if __name__ == "__main__":
    raise SystemExit(main())
