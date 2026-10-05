"""记忆层 B 方案初始化 — 冻结 backbone 跑校准集，hidden 聚类当 key.

用法：
  python scripts/init_memory.py --src data/llm-sft2/ckpt-000200/model.pt \\
      --vocab data/llm-sft/vocab.json --data data/sft-zh \\
      --out data/llm-sft-mem-init --memory-every 4

流程：按 base 预设建含记忆层的模型 -> 形状匹配覆写旧权重（缺的全零恒等）
-> SFT 模板拼校准文本 -> 逐层收集记忆层输入 hidden -> k-means 设 key
-> 落盘 out/model.pt（可直接 --sft-init 严格载入续训）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="记忆层 B 方案初始化")
    p.add_argument("--src", required=True, help="旧权重 model.pt（无记忆层的版本）")
    p.add_argument("--vocab", required=True, help="词表 vocab.json")
    p.add_argument("--data", required=True, help="SFT 语料目录（校准集来源）")
    p.add_argument("--out", required=True, help="输出目录（model.pt + meta.json）")
    p.add_argument("--samples", type=int, default=256, help="校准序列数")
    p.add_argument("--seq-len", type=int, default=512, help="校准序列长度")
    p.add_argument("--batch", type=int, default=4, help="校准 batch")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="每记忆层最多收集 token 数（超了按步长抽稀）")
    p.add_argument("--memory-every", type=int, default=4, help="记忆层插入间隔（与训练一致）")
    p.add_argument("--memory-slots", type=int, default=4096, help="记忆槽位数（须完全平方）")
    p.add_argument("--memory-topk", type=int, default=8, help="每 token 激活槽数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--device", default="", help="设备（空=自动，cuda 优先）")
    return p.parse_args(argv)


def overlap_load(model, src: str, device) -> None:
    """形状匹配覆写（旧权重无 memory 键，留零恒等；形状差只警告）."""
    import torch

    state = torch.load(src, map_location=device)
    own = model.state_dict()
    n_hit, n_miss, n_skip = 0, [], []
    with torch.no_grad():
        for name, param in own.items():
            if name not in state:
                n_miss.append(name)
                continue
            if state[name].shape != param.shape:
                n_skip.append(f"{name}{tuple(state[name].shape)}")
                continue
            param.copy_(state[name])
            n_hit += 1
    print(f"覆写 {n_hit} 个参数；新增（恒等）{len(n_miss)} 个；形状跳过 {len(n_skip)} 个",
          flush=True)
    for name in n_miss[:10]:
        print(f"  新增：{name}", flush=True)


def main(argv=None) -> int:
    """校准主流程."""
    import torch

    from src.llm.local.config import SmallLLMConfig
    from src.llm.local.data import SFT_PROMPT, iter_local_sft
    from src.llm.local.infer import SimpleTokenizer
    from src.llm.local.memory import init_memory_from_activations
    from src.llm.local.model import TinyLLM

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 词表（与训练脚本同口径：直接读 _chars）
    tok = SimpleTokenizer()
    tok._chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    tok.vocab_size = len(tok._chars) + 4

    # 含记忆层的模型 + 旧权重覆写（vocab_size 保持预设容量 8192，
    # 与训练脚本同口径：SFT 扩词复用空行不改形状，覆写要求形状逐位一致）
    config = SmallLLMConfig()
    config.memory_every = args.memory_every
    config.memory_slots = args.memory_slots
    config.memory_topk = args.memory_topk
    model = TinyLLM(config).to(device).eval()
    mem_layers = [(i, layer.memory) for i, layer in enumerate(model.layers)
                  if layer.memory is not None]
    assert mem_layers, "--memory-every 下无记忆层，请检查"
    print(f"记忆层位置：{[i for i, _ in mem_layers]}（共 {len(mem_layers)} 个）",
          flush=True)
    overlap_load(model, args.src, device)

    # 校准文本（SFT 模板拼，与训练分布一致）
    texts: list[str] = []
    for instruction, inp, output in iter_local_sft(args.data):
        texts.append(SFT_PROMPT.format(instruction=instruction, input=inp or ""))
        texts.append(output)
        if len(texts) >= args.samples * 2:
            break
    print(f"校准文本 {len(texts)} 段", flush=True)

    # 逐层 hook 收集记忆层输入 hidden（backbone 冻结，no_grad）
    buffers: dict[int, list[torch.Tensor]] = {i: [] for i, _ in mem_layers}
    handles = []
    for i, mem in mem_layers:
        # hook 里直接展平成 token 行（各 batch 长度不一，cat 前必须先 reshape）
        handles.append(mem.register_forward_hook(
            lambda _m, inp, _o, li=i:
            buffers[li].append(inp[0].detach().cpu().reshape(-1))))
    try:
        with torch.no_grad():
            for b in range(0, len(texts), args.batch):
                ids = [tok.encode(t, add_bos=False)[:args.seq_len]
                       for t in texts[b:b + args.batch]]
                maxlen = max(len(r) for r in ids)
                batch = torch.tensor(
                    [r + [SimpleTokenizer.PAD] * (maxlen - len(r)) for r in ids],
                    dtype=torch.long, device=device)
                model(batch)
    finally:
        for hd in handles:
            hd.remove()

    # 每层抽稀 + k-means 设 key（value 保持零，恒等起点）
    report = {}
    for i, mem in mem_layers:
        h = torch.cat(buffers[i], dim=0).reshape(-1, config.d_model)
        step = max(1, (h.shape[0] + args.max_tokens - 1) // args.max_tokens)
        h = h[::step]
        info = init_memory_from_activations(mem, h, seed=args.seed)
        report[f"layer_{i}"] = info
        print(f"层 {i}：{info['n_samples']} token，inertia={info['inertia']}",
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(json.dumps({
        "src": args.src, "memory_every": args.memory_every,
        "memory_slots": args.memory_slots, "memory_topk": args.memory_topk,
        "layers": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已落盘：{out / 'model.pt'}（可直接 --sft-init 严格载入）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
