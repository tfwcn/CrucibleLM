"""架构迁移 — 改结构不重交学费（加深/专家加宽/层裁剪 + SVD 配方）.

从已有 checkpoint 出发做结构手术，输出新目录（model.pt + config.json + vocab.json），
再用训练脚本的 --init-checkpoint 接上继续训（按行续接，见 load_weights_overlap）。

精确操作（输出逐位一致，可直接续训）：
  --add-layers N        追加 N 个恒等层（bert2BERT 式加深）
  --expert-hidden M     专家 hidden 扩到 M（Net2WiderNet，需大于当前）

有损操作（需短训恢复，--max-steps 另给小值验证）：
  --keep-layers 0,1,2  只保留指定层（变浅/抽层）
  --svd-report         打印各投影矩阵的低秩能量占比（跨架构映射选 rank 用）

用法：
  python scripts/migrate_model.py --src data/llm-ckpt --out data/llm-16L \\
      --add-layers 4
  python scripts/train_local_llm.py ... --init-checkpoint data/llm-16L/model.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.llm.local.config import SmallLLMConfig
from src.llm.local.migrate import (
    add_identity_layers,
    subselect_layers,
    svd_split,
    widen_experts,
)
from src.llm.local.model import TinyLLM


def load_source_config(src: Path) -> SmallLLMConfig:
    """从 hparams.json 恢复配置（训练脚本每次启动都落盘）."""
    hparams = json.loads((src / "hparams.json").read_text(encoding="utf-8"))
    return SmallLLMConfig(**hparams["config"])


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="架构迁移工具箱")
    p.add_argument("--src", required=True, help="源 ckpt 目录（含 model.pt + hparams.json）")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--add-layers", type=int, default=0, help="追加恒等层数")
    p.add_argument("--expert-hidden", type=int, default=0, help="专家 hidden 目标值")
    p.add_argument("--keep-layers", default="",
                   help="只保留指定层，如 0,1,2（逗号分隔，空=全保留）")
    p.add_argument("--svd-report", action="store_true", help="打印低秩能量占比")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """迁移主流程."""
    args = parse_args(argv)
    src = Path(args.src)
    out = Path(args.out)
    config = load_source_config(src)
    print(f"源配置：{config.n_layers} 层，hidden {config.d_model}，"
          f"专家 {config.n_experts}x{config.expert_hidden}", flush=True)
    model = TinyLLM(config)
    model.load_state_dict(torch.load(src / "model.pt", map_location="cpu"))
    model.eval()
    ops: list[str] = []
    if args.keep_layers:
        keep = [int(x) for x in args.keep_layers.split(",") if x.strip() != ""]
        subselect_layers(model, keep)
        ops.append(f"keep-layers={keep}")
    if args.add_layers > 0:
        add_identity_layers(model, args.add_layers)
        ops.append(f"add-layers={args.add_layers}")
    if args.expert_hidden > 0:
        widen_experts(model, args.expert_hidden, seed=args.seed)
        ops.append(f"expert-hidden={args.expert_hidden}")
    if args.svd_report:
        from collections import defaultdict

        energy: dict[str, list[float]] = defaultdict(list)
        with torch.no_grad():
            for layer in model.layers:
                for proj_name in ("w_dq", "w_dkv"):
                    proj = getattr(layer.attn, proj_name, None)
                    if proj is None:  # 线性层无此投影，跳过
                        continue
                    r = min(64, min(proj.weight.shape))
                    _, _, err = svd_split(proj.weight, r)
                    energy[proj_name].append(err)
        for proj_name, errs in energy.items():
            avg = sum(errs) / max(len(errs), 1)
            print(f"SVD rank64 平均重构误差 {proj_name}: {avg:.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    from dataclasses import asdict

    (out / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    if (src / "vocab.json").exists():
        shutil.copy(src / "vocab.json", out / "vocab.json")
    print(f"迁移完成 [{'; '.join(ops) or '无操作'}]：{config.n_layers} 层，"
          f"专家 hidden {config.expert_hidden} -> {out}/model.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
