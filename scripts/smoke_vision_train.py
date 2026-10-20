"""视觉入路训练冒烟 — 合成"亮度 → 数字"任务，验证 DeepSeek-ViT 全链路可训.

为什么用合成任务：仓库里没有图文数据，但"视觉塔能不能训"是结构问题，与数据规模无关。
本任务把图像信息压到**一个标量**（整幅图亮度 = 数字/9），要求模型：

```
图像 (42×42) → patchify(14) → ViT(2 层, 双向 + 2D-RoPE)
             → 3×3 特征 unshuffle（3×3 网格 → 1 个 token）
             → 2 层 MLP projector → 拼在文本前 → 输出那个数字
```

只要视觉塔 ↔ 主干的梯度链路有一处断了（detach、没接线、只训 LM），
准确率就停在 10%（随机）。跑通即证明：编码器带梯度、前缀对齐、loss 位置正确。

**评测下标**：视觉前缀占 `v` 个位置，文本 token j 由绝对位置 `v + j - 1` 预测；
本脚本在"最后一个提示 token"处取 argmax，与该样本的答案 token id 比较
（不是与数字 0–9 比较——那会把 token id 和数值混淆，得到恒 0% 的假失败）。

用法（约 15 秒 / 600 步，RTX 3080）：
    python scripts/smoke_vision_train.py --steps 600 --batch 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.config import SmallLLMConfig  # noqa: E402
from src.llm.local.infer import SimpleTokenizer  # noqa: E402
from src.llm.local.model import TinyLLM  # noqa: E402
from src.llm.local.train import build_optimizer, train_step  # noqa: E402

# 图像边长 42 = patch14 × 3；unshuffle3 后网格 1×1 → 1 个视觉 token
SIDE = 42
# 提示只留 2 个 token：模板越长，模型越容易"忽略图像只背模板"
PROMPT = "\n答"


def parse_args(argv=None) -> argparse.Namespace:
    """命令行参数（步数/批大小/学习率/设备）."""
    p = argparse.ArgumentParser(description="视觉入路训练冒烟")
    p.add_argument("--steps", type=int, default=600, help="训练步数")
    p.add_argument("--batch", type=int, default=32, help="每步样本数")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW 峰值学习率")
    p.add_argument("--eval-every", type=int, default=100, help="评测间隔")
    p.add_argument("--device", default="", help="cuda / cpu（默认自动）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="判定通过的准确率（默认 95%%）")
    return p.parse_args(argv)


def build(device: torch.device, seed: int) -> tuple[TinyLLM, SimpleTokenizer]:
    """构建带视觉塔的 tiny 模型 + 只含模板字符的分词器."""
    cfg = SmallLLMConfig(
        vocab_size=128, d_model=128, n_layers=2, full_attn_every=2, n_heads=4,
        q_lora_rank=64, kv_lora_rank=32, qk_rope_dim=16,
        n_experts=4, top_k=2, expert_hidden=128, n_shared=1,
        max_seq_len=64, dropout=0.0,
        vision_enabled=True, vision_patch=14, vision_unshuffle=3,
        vision_d_vit=64, vision_vit_layers=2, vision_vit_heads=4,
        vision_projector_hidden=128, vision_tokens=0,
    )
    torch.manual_seed(seed)
    model = TinyLLM(cfg).to(device)
    tok = SimpleTokenizer(128)
    tok.fit([PROMPT + str(d) for d in range(10)])
    return model, tok


def make_batch(tok: SimpleTokenizer, batch: int, device: torch.device,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造一批样本：随机数字 d → 亮度 d/9 的图 + 提示 + 答案 d + EOS.

    返回 (input_ids, answer_token_ids, images)。
    """
    digits = torch.randint(0, 10, (batch,), device=device)
    prompt_ids = tok.encode(PROMPT, add_bos=False)
    ans_ids = [tok.encode(str(int(d)), add_bos=False)[0] for d in digits]
    input_ids = torch.tensor(
        [prompt_ids + [a, tok.EOS] for a in ans_ids], device=device)
    # 亮度编码数字：整幅图填充 d/9；加 3% 噪声，避免学到"绝对常数"捷径
    fill = (digits.to(torch.float32) / 9.0).view(-1, 1, 1, 1)
    images = (fill.expand(batch, 3, SIDE, SIDE)
              + torch.randn(batch, 3, SIDE, SIDE, device=device) * 0.03)
    return input_ids, torch.tensor(ans_ids, device=device), images.clamp(0, 1)


@torch.no_grad()
def evaluate(model: TinyLLM, tok: SimpleTokenizer, device: torch.device,
             n: int = 512) -> float:
    """贪心评测：最后一个提示 token 处的 argmax 是否等于答案 token id."""
    model.eval()
    input_ids, ans, images = make_batch(tok, n, device)
    p_len = len(tok.encode(PROMPT, add_bos=False))
    ve = model.encode_vision(images)                  # 推理路径（no_grad）
    logits = model(input_ids[:, :p_len], vision_embeds=ve)["logits"]
    pos = ve.shape[1] + p_len - 1                     # 文本 j 由 v+j-1 预测
    pred = logits[:, pos].argmax(-1)
    return float((pred == ans).float().mean())


def main(argv=None) -> int:
    """训练入口：前向/反向 + 定期打印 loss 与准确率."""
    args = parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok = build(device, args.seed)
    opt = build_optimizer(model, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    vis = sum(p.numel() for p in model.vision.parameters())
    total = sum(p.numel() for p in model.parameters())
    n_vis = model.encode_vision(
        torch.zeros(1, 3, SIDE, SIDE, device=device)).shape[1]
    print(f"设备 {device}｜模型 {total / 1e6:.2f}M（视觉塔 {vis / 1e6:.2f}M）"
          f"｜{args.steps} 步 × batch {args.batch}｜视觉 token 数 {n_vis}",
          flush=True)
    print(f"随机基线 10%（10 分类）；判定线 {args.threshold * 100:.0f}%",
          flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        ids, _, imgs = make_batch(tok, args.batch, device)
        info = train_step(model, opt, ids, vision_images=imgs,
                          use_amp=(device.type == "cuda"))
        sched.step()
        if step % max(1, args.eval_every) == 0 or step == args.steps:
            acc = evaluate(model, tok, device)
            print(f"step={step:>4} loss={float(info['loss']):.4f} "
                  f"acc={acc * 100:5.1f}% 用时 {time.time() - t0:.1f}s",
                  flush=True)

    acc = evaluate(model, tok, device, n=1024)
    ok = acc >= args.threshold
    print(f"\n最终准确率 {acc * 100:.1f}%（随机 10%）："
          f"{'通过' if ok else '未通过，视觉链路可能断了'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
