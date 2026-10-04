"""直觉头 — frozen backbone 上的毫瓦级决策器（Jev 式 System1）.

用法：backbone 定型后冻结，一次前向提 hidden（encode_last_hidden），
小 MLP 拟合决策任务（工具选择/升级二分类/意图路由）。高置信走快道，
低置信回落生成式慢道，最差退化成现状。
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntuitionHead(nn.Module):
    """两层 MLP 头（d -> hidden -> n_classes，二分类时 n_classes=1 用 BCE）."""

    def __init__(self, d_model: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_classes if n_classes > 2 else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 logits（多类 (…, n)，二类 (…, 1)）."""
        return self.net(x)


def extract_features(
    model,
    tokenizer,
    texts: list[str],
    max_len: int = 512,
    device: str = "cpu",
    batch_size: int = 8,
) -> torch.Tensor:
    """批量提末位置 hidden（CPU 张量，backbone 不动）."""
    feats: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        ids = [tokenizer.encode(t)[:max_len] for t in chunk]
        lengths = torch.tensor([len(s) for s in ids])
        width = max(len(s) for s in ids)
        batch = torch.tensor(
            [s + [0] * (width - len(s)) for s in ids], dtype=torch.long)
        h = model.encode_full_hidden(batch.to(device)).cpu()
        # 按真实长度 gather（padding 位不能用）
        feats.append(h[torch.arange(len(ids)), lengths - 1])
    return torch.cat(feats, dim=0)


def train_head(
    head: IntuitionHead,
    feats: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> list[float]:
    """训头（backbone 已 frozen，只动头参数；返回每轮 loss）。"""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = feats.shape[0]
    curve: list[float] = []
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = feats[idx], labels[idx]
            opt.zero_grad(set_to_none=True)
            logits = head(x)
            if head.n_classes <= 2 and logits.shape[-1] == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y.float())
            else:
                loss = F.cross_entropy(logits, y.long())
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            count += len(idx)
        curve.append(total / max(count, 1))
    return curve


def calibrate_temperature(
    head: IntuitionHead,
    feats: torch.Tensor,
    labels: torch.Tensor,
    grid: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
) -> float:
    """温度校准：在验证集上最小化 NLL 选 T（置信度可审计的前提）."""
    head.eval()
    with torch.no_grad():
        logits = head(feats)
    if head.n_classes <= 2 and logits.shape[-1] == 1:
        probs_fn = lambda t: torch.sigmoid(t)  # noqa: E731
        nll_fn = lambda p: F.binary_cross_entropy(p, labels.float())  # noqa: E731
    else:
        probs_fn = lambda t: F.softmax(t, dim=-1)  # noqa: E731
        nll_fn = lambda p: F.cross_entropy(  # noqa: E731
            torch.log(p.clamp_min(1e-9)), labels.long())
    best_t, best_nll = 1.0, float("inf")
    with torch.no_grad():
        for t in grid:
            nll = float(nll_fn(probs_fn(logits / t)))
            if nll < best_nll:
                best_nll, best_t = nll, t
    return best_t


def evaluate_head(
    head: IntuitionHead,
    feats: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 10,
) -> dict:
    """评测：准确率 + NLL + ECE（期望校准误差）+ 单条延迟."""
    head.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = head(feats) / temperature
        if head.n_classes <= 2 and logits.shape[-1] == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5).long()
            nll = float(F.binary_cross_entropy(probs, labels.float()))
            conf = torch.where(preds == 1, probs, 1 - probs)
        else:
            probs = F.softmax(logits, dim=-1)
            conf, preds = probs.max(-1)
            nll = float(F.cross_entropy(logits, labels.long()))
    dt = (time.time() - t0) / max(len(feats), 1) * 1000
    acc = float((preds == labels.long()).float().mean())
    # ECE：按置信度分桶，|准确率−置信度| 加权平均
    ece, total = 0.0, 0
    correct = (preds == labels.long()).float()
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if int(m.sum()) == 0:
            continue
        ece += abs(float(correct[m].mean()) - float(conf[m].mean())) * int(m.sum())
        total += int(m.sum())
    ece = ece / max(total, 1)
    return {"acc": round(acc, 4), "nll": round(nll, 4),
            "ece": round(ece, 4), "latency_ms": round(dt, 3)}
