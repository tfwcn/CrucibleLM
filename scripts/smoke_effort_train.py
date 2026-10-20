"""努力档（#18）训练冒烟 — 合成"档位 → 回答长度"任务，验证条件化链路可训.

**这个冒烟验证什么、不验证什么**：

- 验证：努力档文本经 `apply_effort()`（生产同款注入函数）进入系统提示后，
  同一份权重能对不同档位产生**不同长度**的输出，且训练能收敛（loss 下降、
  各档位输出稳定）。链路断在任一环（注入没进 tokenizer、档位没进上下文、
  EOS 学不会）都会让"长度-档位"曲线变成平的。
- 不验证：模型是否真的"想得更久更对"。那是 RL 阶段（奖励塑形 + 分组优势，
  见 `docs/details/reasoning-effort.md`）才有的行为，SFT 只能学条件化。

任务：固定问题，四个档位各对应一段长度递增的固定回答；模型必须
"看到档位 → 复述对应长度的回答 → 在正确位置停"。

用法（约 40 秒 / 800 步，RTX 3080）：
    python scripts/smoke_effort_train.py --steps 800
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import SmallLLMConfig  # noqa: E402
from src.llm.local.effort import apply_effort  # noqa: E402
from src.llm.local.infer import SimpleTokenizer  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.train import build_optimizer, train_step  # noqa: E402

# 档位 → 回答（长度单调递增；内容不同，便于模型"记住在哪停"）
LEVELS: dict[int, str] = {
    25: "短。",
    50: "略长一点，说两句。",
    75: "这是一个比较长的回答，需要解释一下原因。",
    100: "这是一个很长的回答，需要把前因后果都讲清楚，"
         "再补充一些例子，最后给出结论。",
}
QUESTION = "介绍一下摸鱼。"


def parse_args(argv=None) -> argparse.Namespace:
    """命令行参数."""
    p = argparse.ArgumentParser(description="努力档训练冒烟")
    p.add_argument("--steps", type=int, default=800, help="训练步数")
    p.add_argument("--batch", type=int, default=16, help="每步样本数")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW 峰值学习率")
    p.add_argument("--eval-every", type=int, default=200, help="评测间隔")
    p.add_argument("--device", default="", help="cuda / cpu（默认自动）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args(argv)


def build(device: torch.device, seed: int) -> tuple[TinyLLM, SimpleTokenizer]:
    """构建 tiny 模型 + 覆盖全部模板/回答字符的分词器."""
    cfg = SmallLLMConfig(
        vocab_size=128, d_model=128, n_layers=2, full_attn_every=2, n_heads=4,
        q_lora_rank=64, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=4, top_k=2, expert_hidden=128, n_shared=1,
        max_seq_len=256, dropout=0.0,
    )
    torch.manual_seed(seed)
    model = TinyLLM(cfg).to(device)
    tok = SimpleTokenizer(128)
    texts = [_prompt(b) + a + "\n" for b, a in LEVELS.items()]
    tok.fit(texts)
    return model, tok


def _prompt(b: int | None) -> str:
    """构造系统提示（含努力档）+ 用户问题 + 助手起手式."""
    system = apply_effort("", b) if b is not None else ""
    return f"<系统>\n{system}\n<用户>\n{QUESTION}\n<助手>\n"


def make_batch(tok: SimpleTokenizer, level: int, batch: int,
               device: torch.device,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """同一档位一批样本：input_ids + labels（只监督回答段 + EOS）."""
    p_ids = tok.encode(_prompt(level), add_bos=False)
    ans = tok.encode(LEVELS[level], add_bos=False) + [tok.EOS]
    ids = torch.tensor([p_ids + ans] * batch, device=device)
    labels = torch.tensor(
        [[-100] * len(p_ids) + ans] * batch, device=device)
    return ids, labels


@torch.no_grad()
def measure(model: TinyLLM, tok: SimpleTokenizer, device: torch.device,
            n: int = 8) -> dict[int, float]:
    """各档位贪心生成的回答长度（token 数，不含 EOS）."""
    model.eval()
    out: dict[int, float] = {}
    for level in LEVELS:
        p_ids = torch.tensor(
            [tok.encode(_prompt(level), add_bos=False)] * n, device=device)
        gen = model.generate(p_ids, max_new_tokens=48, temperature=0.0,
                             eos_id=tok.EOS)
        # 去掉提示部分；遇到 EOS 截断
        new = gen[:, p_ids.shape[1]:].tolist()
        lens = []
        for row in new:
            cut = row.index(tok.EOS) if tok.EOS in row else len(row)
            lens.append(cut)
        out[level] = sum(lens) / len(lens)
    return out


def main(argv=None) -> int:
    """训练入口：按档位轮流采样，定期打印 loss 与各档位长度."""
    args = parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok = build(device, args.seed)
    opt = build_optimizer(model, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    levels = list(LEVELS)
    print(f"设备 {device}｜档位 {levels}｜各档目标长度 "
          f"{ {b: len(tok.encode(LEVELS[b], add_bos=False)) + 1 for b in levels} }"
          f"｜{args.steps} 步 × batch {args.batch}", flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        level = levels[(step - 1) % len(levels)]       # 轮转档位，保证每档等量
        ids, labels = make_batch(tok, level, args.batch, device)
        info = train_step(model, opt, ids, labels=labels,
                          use_amp=(device.type == "cuda"))
        sched.step()
        if step % max(1, args.eval_every) == 0 or step == args.steps:
            lens = measure(model, tok, device)
            print(f"step={step:>4} loss={float(info['loss']):.4f} "
                  f"长度={ {b: round(v, 1) for b, v in lens.items()} } "
                  f"用时 {time.time() - t0:.1f}s", flush=True)

    lens = measure(model, tok, device, n=16)
    vals = [lens[b] for b in levels]
    monotone = all(x < y for x, y in zip(vals, vals[1:]))
    print(f"\n各档平均长度 {[round(v, 1) for v in vals]}："
          f"{'严格单调递增，通过' if monotone else '不单调，未通过'}", flush=True)
    return 0 if monotone else 1


if __name__ == "__main__":
    raise SystemExit(main())
