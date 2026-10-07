"""本地小 LLM 训练流程 — 中文预训练 + 指令微调一站式脚本.

语料（公开中文语料，脚本内自动选用）：
- 预训练：epfml/FineWeb2-HQ:cmn_Hani（首选，质量过滤中文），
  失败自动回退 HuggingFaceFW/fineweb-2:cmn_Hani（全量中文），均 HF 流式、无需下载。
- SFT：BelleGroup/train_0.5M_CN（50 万中文指令）。
- 无网时用 --data local --local-path <txt目录> 回退。

用法示例（16G 单卡预训练）：
  python scripts/train_local_llm.py --phase pretrain --preset base \\
      --seq-len 2048 --batch 4 --accum 8 --max-steps 10000 --ckpt-dir data/llm-ckpt

CPU 冒烟测试（几十秒跑通全流程）：
  python scripts/train_local_llm.py --phase pretrain --preset tiny \\
      --data local --local-path tests/fixtures/llm-corpus --max-steps 5

SFT（接预训练权重，只学 output，prompt 掩掉）：
  python scripts/train_local_llm.py --phase sft --preset base \\
      --sft-init data/llm-ckpt/model.pt --max-steps 2000 --ckpt-dir data/llm-sft
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

# 允许从仓库任意位置以 scripts/ 相对路径运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.llm.local.config import SmallLLMConfig, tiny_test_config
from src.llm.local.data import (
    PackedBatcher,
    PrefetchIterator,
    ShuffleBuffer,
    fit_tokenizer_on_stream,
    interleave_batches,
    iter_hf_pretrain,
    iter_hf_pretrain_mix,
    iter_hf_sft,
    iter_hf_sft_mix,
    iter_local_sft,
    iter_local_texts,
    pack_pairs,
    pack_pretrain,
    pack_sft,
)
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
from src.llm.local.model import TinyLLM
from src.llm.local.train import build_optimizer

# 预留评测的文档数（同时复用做分词器拟合 + 训练前跳过）
HOLDOUT_DOCS = 256


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="本地小 LLM 中文训练流程")
    p.add_argument("--phase", choices=["pretrain", "sft"], default="pretrain",
                   help="预训练或指令微调")
    p.add_argument("--preset", choices=["tiny", "base"], default="base",
                   help="tiny=CPU 验证配置，base=100M 默认配置")
    p.add_argument("--seq-len", type=int, default=512, help="训练序列长度")
    p.add_argument("--batch", type=int, default=4, help="每步 batch（序列数）")
    p.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--max-steps", type=int, default=1000, help="训练步数上限")
    p.add_argument("--lr", type=float, default=3e-4, help="峰值学习率")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="优化器：adamw（默认，稳妥）或 muon（约 2x 效率，2D 隐藏权重走 Muon 其余走 AdamW）")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Muon 部分峰值学习率")
    p.add_argument("--kd-teacher", default="",
                   help="蒸馏老师权重目录（空=关闭；如 data/teacher-0.5b，跨词表锚点 KL）")
    p.add_argument("--kd-alpha", type=float, default=0.5, help="蒸馏 loss 权重")
    p.add_argument("--kd-temp", type=float, default=2.0, help="蒸馏温度")
    p.add_argument("--kd-every", type=int, default=1,
                   help="每 N 个 micro-step 算一次老师（降频省时间；>1 时 kd 项自动放大 N 倍保期望；1=每个都算）")
    p.add_argument("--extend-vocab", action="store_true",
                   help="扫描语料缺字追加进词表（旧 id 不动；容量内复用空行，超限扩行并新开动量）")
    p.add_argument("--extend-scan-docs", type=int, default=2000,
                   help="扩词扫描文档数")
    p.add_argument("--extend-max-new", type=int, default=512,
                   help="最多新增字数（按频次取）")
    p.add_argument("--rho-keep", type=float, default=0.0,
                   help="RHO 选择：只反向 loss 最高的该比例 token（0=关闭，建议 0.4~0.6；MTP/aux 不受影响）")
    p.add_argument("--rho-ref", choices=["none", "teacher"], default="none",
                   help="RHO 参照：none=自参照 batch 百分位；teacher=超额 loss（需 --kd-teacher，学生减老师）")
    p.add_argument("--replay-ratio", type=float, default=0.0,
                   help="回放比例：每步该概率从难样本池取 batch（0=关闭，建议 0.1~0.2）")
    p.add_argument("--replay-capacity", type=int, default=8192,
                   help="回放池上限（块数，FIFO 淘汰）")
    p.add_argument("--consolidate-steps", type=int, default=0,
                   help="巩固期步数（0=关闭；新分布开头回放加码 + lr 不升）")
    p.add_argument("--consolidate-replay", type=float, default=0.4,
                   help="巩固期回放比例")
    p.add_argument("--curriculum", default="",
                   help="课程：min-score 起止与步数，如 0.7:0.0:3000（仅有 score 字段的源生效）")
    p.add_argument("--ema-every", type=int, default=0,
                   help="EMA 影子更新间隔（0=关闭；影子只做一致性参照，不占梯度）")
    p.add_argument("--ema-weight", type=float, default=0.05,
                   help="EMA 一致性 loss 权重")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA 动量")
    # 检索增强（RETRO-lite）与记忆层：默认关，四开足
    p.add_argument("--retro-db", default="",
                   help="RETRO 检索库路径（BM25 索引落盘前请先跑 build_retrieval；空=关闭）")
    p.add_argument("--retro-k", type=int, default=2, help="每段检索取 top-K 文档")
    p.add_argument("--retro-len", type=int, default=64,
                   help="V2 交错：每文档取多少 token（仅 retro_every>0 时用）")
    p.add_argument("--retro-every", type=int, default=0,
                   help="V2 交错：每 N 层一个融合块；0=v1 单点（final_norm 后融合均值）")
    p.add_argument("--enable-retro", action="store_true",
                   help="模型侧开启 RETRO 融合（需 --retro-db 提供，仅在 mid-training 起用）")
    p.add_argument("--enable-memory", action="store_true",
                   help="开启 Product-Key 记忆层（每 memory_every 层一个）")
    p.add_argument("--memory-every", type=int, default=4, help="记忆层插入间隔（N 层一个）")
    p.add_argument("--memory-slots", type=int, default=4096, help="记忆槽位数")
    p.add_argument("--memory-topk", type=int, default=8, help="每 token 激活槽数")
    p.add_argument("--warmup", type=int, default=50, help="warmup 步数")
    p.add_argument("--ckpt-dir", default="data/llm-ckpt", help=" checkpoint 目录")
    p.add_argument("--resume", action="store_true", help="从 ckpt-dir/latest 继续")
    p.add_argument("--eval-every", type=int, default=100, help="评测间隔（0=关闭）")
    p.add_argument("--eval-batches", type=int, default=10, help="每次评测批数")
    p.add_argument("--sample-every", type=int, default=200, help="生成采样间隔（0=关闭）")
    p.add_argument("--save-every", type=int, default=100, help="存盘间隔")
    p.add_argument("--keep-last", type=int, default=20,
                   help="只保留最近 N 个版本快照（<=0 全保留；每个约 1.3GB）")
    p.add_argument("--data", choices=["hf", "local"], default="hf", help="语料来源")
    p.add_argument("--local-path", default="", help="本地语料目录（--data local 时必填；多目录混合逗号分隔，可带权重如 a:3,b:1）")
    p.add_argument("--pretrain-mix", default="hq:1",
                   help="HF 预训练混合配比，如 hq:1,ultrafineweb:2（可选 hq/fineweb2/ultrafineweb）")
    p.add_argument("--pretrain-min-score", type=float, default=0.0,
                   help="质量分下限（仅有 score 字段的源生效，如 ultrafineweb）")
    p.add_argument("--sft-mix", default="belle:1",
                   help="SFT 混合配比，如 belle:2,agent-general:1（可选 belle/agent-general/agent-code/agent-search/agent-tool）")
    p.add_argument("--sft-replay-dir", default="",
                   help="SFT 阶段掺 pretrain 回放的语料目录（空=关闭；防第二遍刷 SFT 过拟合）")
    p.add_argument("--sft-replay-ratio", type=float, default=0.0,
                   help="回放 batch 占比（0~1，如 0.15；主流耗尽即停，回放不计 cursor）")
    p.add_argument("--sft-replay-pool", type=int, default=8192,
                   help="回放池文档数（取目录前 N 篇打乱后循环，防 20GB 全载入内存）")
    p.add_argument("--hf-endpoint", default="",
                   help="HF 镜像源（默认读 HF_ENDPOINT 环境变量；"
                   "国内直连超时请传 https://hf-mirror.com，代码内生效不依赖 shell 传递）")
    p.add_argument("--vocab", default="", help="复用已有 vocab.json（默认拟合新表）")
    p.add_argument("--sft-init", default="", help="SFT 起始权重（model.pt，默认随机初始化）")
    p.add_argument("--init-checkpoint", default="",
                   help="迁移/换架构后的起始权重（model.pt，按行续接；与 --resume 同用时只取 step 计数，权重与动量都从此处新开）")
    p.add_argument("--config-json", default="",
                   help="模型配置 JSON（migrate 输出的 config.json；默认按 preset 构建）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--no-amp", action="store_true", help="禁用 bf16 混合精度")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="开梯度检查点（seq2048+ 进 16G 必需，重算换显存）")
    p.add_argument("--fit-chars", type=int, default=20_000_000,
                   help="分词器拟合采样字符数")
    p.add_argument("--shuffle-buffer", type=int, default=512,
                   help="block 级 shuffle 蓄水池（0=关闭；平滑站点聚集，评测流不受影响）")
    p.add_argument("--prefetch", type=int, default=4,
                   help="后台预取 batch 数（0=关闭；打包挪到后台线程，GPU 不等 CPU）")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    """固定随机种子."""
    random.seed(seed)
    torch.manual_seed(seed)


def lr_schedule(step: int, max_steps: int, warmup: int, peak: float) -> float:
    """warmup 线性升温 + cosine 衰减到 10%."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def build_config(args: argparse.Namespace) -> SmallLLMConfig:
    """按 preset 构建配置（训练序列不得超过 RoPE 缓存则自动扩容）."""
    config = tiny_test_config() if args.preset == "tiny" else SmallLLMConfig()
    config.max_seq_len = max(config.max_seq_len, args.seq_len + 8)
    return config


def load_tokenizer(args: argparse.Namespace, config: SmallLLMConfig,
                   sample_texts: list[str]) -> SimpleTokenizer:
    """加载或拟合分词器（拟合采样截断到 --fit-chars 防爆内存）."""
    if args.vocab:
        tok = SimpleTokenizer(config.vocab_size)
        with open(args.vocab, encoding="utf-8") as f:
            tok._chars = json.load(f)
        tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
        print(f"分词器已加载：{args.vocab}（{len(tok._chars)} 字符）", flush=True)
        return tok
    tok = fit_tokenizer_on_stream(iter(sample_texts), config.vocab_size,
                                  max_chars=args.fit_chars)
    print(f"分词器拟合完成（{len(tok._chars)} 字符）", flush=True)
    return tok


def collect_holdout(args: argparse.Namespace, data_cursor: int = 0) -> tuple[list, object]:
    """收集评测文档并返回 (评测文本, 训练文本流）.

    HF 流式：先取前 N 个文档做评测+分词拟合，训练流跳过它们；
    本地文件：按文件名切分前后段。
    data_cursor>0 时训练流再快进跳过这么多（断点续流，见 README）；
    返回的训练流恒为 CountedIterator（.n 供存盘记录消费数）。
    """
    from src.llm.local.data import CountedIterator as _Counted
    from src.llm.local.data import interleave_weighted as _ilw
    from src.llm.local.data import parse_local_mix as _plm
    from src.llm.local.data import skip_items as _skip

    if args.data == "local" and args.phase == "sft":
        # 本地 SFT（Belle 格式 jsonl/csv，多目录逗号分隔，可带权重）；
        # triple 在此套模板转对，与 HF 路径的 pairs 口径一致
        from src.llm.local.data import SFT_PROMPT as _TPL
        from src.llm.local.data import parse_local_mix as _plm2

        sdirs = _plm2(args.local_path)

        def _local_pairs():
            if len(sdirs) == 1:
                raw = iter_local_sft(sdirs[0][0])
            else:
                from src.llm.local.data import interleave_weighted as _ilw2

                raw = _ilw2([iter_local_sft(p) for p, _ in sdirs],
                            [w for _, w in sdirs])
            for instruction, inp, output in raw:
                yield (_TPL.format(instruction=instruction, input=inp or ""),
                       output)

        sft_pairs = _local_pairs()
        pairs = list(_take(sft_pairs, HOLDOUT_DOCS))
        _skip(sft_pairs, data_cursor)
        print(f"本地 SFT 语料 {args.local_path}：评测 {len(pairs)} 对", flush=True)
        return pairs, _Counted(sft_pairs)
    if args.data == "local":
        dirs = _plm(args.local_path)
        if len(dirs) == 1 and dirs[0][1] == 1:
            # 单目录：前半评测、后半训练，外加断点快进（文件顺序稳定）
            holdout_all = _take(iter_local_texts(dirs[0][0]), HOLDOUT_DOCS)
            n_hold = max(2, len(holdout_all) // 2)
            holdout = holdout_all[:n_hold]

            def train_gen_single():
                skipped = iter_local_texts(dirs[0][0])
                _skip(skipped, n_hold + data_cursor)
                yield from skipped

            if data_cursor:
                print(f"本地语料：评测 {len(holdout)} 文档，断点快进 {data_cursor}，训练流延迟遍历",
                      flush=True)
            else:
                print(f"本地语料：评测 {len(holdout)} 文档，训练流延迟遍历（不预加载）", flush=True)
            return holdout, _Counted(train_gen_single())
        # 多目录：混合流取评测，各目录按消费计数精确跳过（不 whole-list 物化）
        counted = [_Counted(iter_local_texts(p)) for p, _ in dirs]
        weights = [w for _, w in dirs]
        holdout = _take(_ilw(counted, weights), HOLDOUT_DOCS)
        counts = [c.n for c in counted]

        def train_gen_mixed():
            streams = []
            for (p, _), skip in zip(dirs, counts):
                it = iter_local_texts(p)
                _skip(it, skip)
                streams.append(it)
            mixed = _ilw(streams, weights)
            _skip(mixed, data_cursor)
            yield from mixed

        print(f"本地混合语料 {args.local_path}：评测 {len(holdout)} 文档"
              f"（各目录已消费 {counts}，断点快进 {data_cursor}），训练流延迟遍历",
              flush=True)
        return holdout, _Counted(train_gen_mixed())
    # HF 流式：顺序消费（同一流上先攒评测，再继续做训练，避免两次建流）
    if args.phase == "sft":
        sft_stream = iter_hf_sft_mix(args.sft_mix, seed=args.seed)
        pairs = list(_take(sft_stream, HOLDOUT_DOCS))
        print(f"SFT 语料（{args.sft_mix}）：评测 {len(pairs)} 对", flush=True)
        _skip(sft_stream, data_cursor)
        return pairs, _Counted(sft_stream)
    stream = iter_hf_pretrain_mix(args.pretrain_mix, seed=args.seed,
                                  min_score=args.pretrain_min_score)
    holdout = list(_take(stream, HOLDOUT_DOCS))
    print(f"HF 语料（{args.pretrain_mix}）：评测 {len(holdout)} 文档（同步拟合分词器）",
          flush=True)
    _skip(stream, data_cursor)
    return holdout, _Counted(stream)


def collect_missing(args, tok) -> list[str]:
    """扫描语料找词表缺字（按频次排序，截断到 --extend-max-new）.

    各源用独立新流扫描（不 consum 训练流）；HF 多走几千文档，多一次网络开销。
    """
    from collections import Counter

    cnt: Counter = Counter()

    def feed(text_iter, budget: int) -> None:
        for i, text in enumerate(text_iter):
            if i >= budget:
                break
            for ch in text:
                if ch not in tok._ids:
                    cnt[ch] += 1
            if len(cnt) >= args.extend_max_new * 4:
                break  # 候选够多早停

    if args.phase == "sft":
        if args.data == "local":
            from src.llm.local.data import parse_local_mix as _plm

            def local_sft_texts():
                for p, _ in _plm(args.local_path):
                    for instruction, inp, output in iter_local_sft(p):
                        yield instruction + (inp or "") + output

            feed(local_sft_texts(), args.extend_scan_docs)
        else:
            def sft_texts():
                for prompt, response in iter_hf_sft_mix(args.sft_mix, seed=args.seed + 999):
                    yield prompt + response

            feed(sft_texts(), args.extend_scan_docs)
    elif args.data == "local":
        from src.llm.local.data import parse_local_mix as _plm

        def local_texts_multi():
            for p, _ in _plm(args.local_path):
                yield from iter_local_texts(p)

        feed(local_texts_multi(), args.extend_scan_docs)
    else:
        feed(iter_hf_pretrain_mix(args.pretrain_mix, seed=args.seed + 999,
                                  min_score=args.pretrain_min_score),
             args.extend_scan_docs)
    return [ch for ch, _ in cnt.most_common(args.extend_max_new)]


def _take(it, n: int) -> list:
    """从迭代器取前 n 个."""
    out = []
    for _, x in zip(range(n), it):
        out.append(x)
    return out


def parse_curriculum(spec: str) -> tuple[float, float, int] | None:
    """解析课程 "start:end:steps"（空串=关闭）."""
    if not spec:
        return None
    try:
        s, e, n = spec.split(":")
        return float(s), float(e), max(int(n), 1)
    except (ValueError, AttributeError) as ex:
        raise SystemExit(f"--curriculum 格式应为 start:end:steps，收到 {spec!r}") from ex


def curriculum_value(start: float, end: float, steps: int, step: int) -> float:
    """课程阈值：线性过渡，steps 步后保持 end."""
    if step >= steps:
        return end
    return start + (end - start) * (step / steps)


def dump_hparams(args: argparse.Namespace, config, ckpt_dir: Path) -> dict:
    """落盘超参与配置（复核训练历史用）：全部 CLI 参数 + 关键派生量."""
    from dataclasses import asdict, is_dataclass

    hparams = dict(vars(args))
    hparams["config"] = asdict(config) if is_dataclass(config) else str(config)
    # 派生量：每步 token、模型规模（启动时快照）
    try:
        hparams["tokens_per_step"] = args.batch * (args.seq_len + 1) * args.accum
    except Exception:
        pass
    hparams["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (ckpt_dir / "hparams.json").write_text(
        json.dumps(hparams, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return hparams


def save_ckpt(ckpt_dir: Path, model: TinyLLM, optim, step: int, tokens: int,
              extra: dict | None = None, keep_last: int = 20,
              data_cursor: int = 0) -> Path:
    """存盘：latest（续跑用）+ 版本快照 ckpt-{step}（只留最近 keep_last 个）.

    latest 含优化器状态（精确续跑）；快照只存权重 + 元信息（省 2/3 写盘，
    NFS 上一次全量存盘要 several 分钟），从快照恢复需新开动量。
    data_cursor 为训练文本流已消费数（断点续流用，存进 latest.json）。
    20 个快照约 10GB 磁盘；keep_last<=0 表示全保留。返回本次快照目录。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"step": step, "tokens_seen": tokens, "data_cursor": data_cursor,
            **(extra or {})}
    meta_text = json.dumps(meta)
    # latest：续跑入口（权重+优化器+元信息）
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optim.state_dict(), ckpt_dir / "optim.pt")
    (ckpt_dir / "latest.json").write_text(meta_text, encoding="utf-8")
    # 版本快照：权重+元信息（无 optim.pt，独立目录，prune 时整个删）
    snap = ckpt_dir / f"ckpt-{step:06d}"
    snap.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), snap / "model.pt")
    (snap / "meta.json").write_text(meta_text, encoding="utf-8")
    if keep_last > 0:
        snaps = sorted(p for p in ckpt_dir.glob("ckpt-*") if p.is_dir())
        for old in snaps[:-keep_last] if len(snaps) > keep_last else []:
            import shutil

            shutil.rmtree(old, ignore_errors=True)
    return snap


def save_best(ckpt_dir: Path, model, step: int, val: float) -> Path:
    """冠军快照：val 新低时另存 best/（权重 + 元信息，永不轮转）.

    存的是调用方给的模型（EMA 开时传影子，冠军即 EMA 权重）；
    只存权重（无 optim.pt），供 --sft-init / --init-checkpoint 起新轮用。
    """
    best = ckpt_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best / "model.pt")
    (best / "meta.json").write_text(
        json.dumps({"step": step, "val_loss": val}, ensure_ascii=False),
        encoding="utf-8")
    return best


def load_weights_overlap(model: TinyLLM, path, map_location=None) -> None:
    """按行续接旧权重：形状一致直接拷；embed/lm_head 允许行数变多（旧行拷贝）."""
    state = torch.load(path, map_location=map_location)
    for name, param in model.named_parameters():
        if name not in state:
            print(f"  警告：{name} 在 checkpoint 中缺失，保留初始化", flush=True)
            continue
        old = state[name]
        if old.shape == param.shape:
            param.data.copy_(old)
        elif (old.dim() == param.dim() and old.shape[1:] == param.shape[1:]
                and old.shape[0] < param.shape[0]
                and ("embed" in name or "lm_head" in name)):
            param.data[:old.shape[0]].copy_(old)
            print(f"  {name}：续接 {old.shape[0]} 行，新增 {param.shape[0] - old.shape[0]} 行随机初始化",
                  flush=True)
        else:
            print(f"  警告：{name} 形状 {tuple(old.shape)}->{tuple(param.shape)} 对不上，保留初始化",
                  flush=True)
    for name, buf in model.named_buffers():
        if name in state and state[name].shape == buf.shape:
            buf.copy_(state[name])


def load_ckpt(ckpt_dir: Path, model: TinyLLM, optim=None,
              map_location=None) -> dict:
    """加载 checkpoint，返回元信息（优化器不匹配则警告并新开动量，权重不受影响）."""
    try:
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=map_location))
    except RuntimeError as e:
        if "size mismatch" not in str(e):
            raise
        # 扩词后的形状差：走按行续接（常见于 embedding 行数变多）
        print(f"权重形状变化，按行续接旧权重：{e}", flush=True)
        load_weights_overlap(model, ckpt_dir / "model.pt", map_location)
    meta = json.loads((ckpt_dir / "latest.json").read_text(encoding="utf-8"))
    if optim is not None and (ckpt_dir / "optim.pt").exists():
        try:
            optim.load_state_dict(
                torch.load(ckpt_dir / "optim.pt", map_location=map_location))
        except (RuntimeError, KeyError, ValueError) as e:
            print(f"警告：优化器状态不兼容（{e}），动量新开，权重照常恢复", flush=True)
    return meta


@torch.no_grad()
def evaluate(model: TinyLLM, make_eval, n_batches: int, device,
               restore_train: bool = True) -> float:
    """评测集平均 loss（SFT 掩码同样生效；评测流每次重建避免耗尽）.

    restore_train=False 时保持 eval 模式（EMA 影子评测用；影子无 dropout
    需求，翻成 train 反而引入噪声）。
    """
    model.eval()
    total, count = 0.0, 0
    try:
        for _, batch in zip(range(n_batches), make_eval()):
            if isinstance(batch, tuple):
                x, y = (t.to(device) for t in batch)
                out = model(x, targets=y)
            else:
                x = batch.to(device)
                out = model(x, targets=x)
            total += float(out["loss"])
            count += 1
    except StopIteration:
        pass  # 评测样本不足时按已有批次平均
    if restore_train:
        model.train()
    else:
        model.eval()
    return total / max(count, 1)


def _forward_batch(model: TinyLLM, batch, device, distiller=None,
                   kd_alpha: float = 0.0, id_to_char: dict | None = None,
                   rho_keep: float = 0.0, ema_model=None,
                   ema_weight: float = 0.0, rho_ref: str = "none",
                   do_kd: bool = True, mem=None, mem_mask=None,
                   chunk_ids=None, chunk_mask=None) -> dict:
    """单个 micro-batch 前向（预训练与 SFT 统一入口，SFT 带 -100 掩码）.

    distiller 非空时加锚点 KL（跨词表蒸馏）：老师内部 no_grad 只出分布，
    学生侧经 logits 直连主干，与 CE 共图累加，无需 detach。
    do_kd=False 跳过老师前向（降频省时间，调用方按 --kd-every 控制）。
    rho_keep>0 时主 loss 只取 top 部分（MTP/aux 全量，aux 必须看全路由）；
    rho_ref=teacher 时按超额 loss（学生−老师）选，需 distiller，否则回落自参照。
    ema_model 非空时加 logits-MSE 一致性（影子 no_grad，不占梯度）。
    """
    from src.llm.local.train import select_topk_loss

    if isinstance(batch, tuple):
        x, y = (t.to(device) for t in batch)
        out = model(x, targets=y, return_token_losses=True, mem=mem, mem_mask=mem_mask,
                    chunk_ids=chunk_ids, chunk_mask=chunk_mask)
    else:
        x = batch.to(device)
        out = model(x, targets=x, return_token_losses=True, mem=mem, mem_mask=mem_mask,
                    chunk_ids=chunk_ids, chunk_mask=chunk_mask)
    if rho_keep > 0:
        if rho_ref == "teacher" and do_kd and distiller is not None and id_to_char is not None:
            from src.llm.local.distill import select_by_excess

            pair_losses = distiller.teacher_token_losses(x, id_to_char)
            mask = select_by_excess(
                out["token_losses"].detach(), pair_losses, rho_keep)
            sel = ((out["token_losses"] * mask).sum()
                   / mask.sum().clamp_min(1))
        else:
            # 无老师或 KD 降频跳过的 micro：回落自参照（零老师开销）
            sel, _ = select_topk_loss(out["token_losses"], rho_keep)
        # 精确扣除主 loss 梯度贡献（同张量相减），再加选中部分
        out["loss"] = out["loss"] - out["_main_mean"] + sel
        out["main_loss"] = sel.detach()
    if distiller is not None and kd_alpha > 0 and id_to_char is not None and do_kd:
        kd = distiller.batch_kl(x, out["logits"], id_to_char)
        out["loss"] = out["loss"] + kd_alpha * kd
        out["kd_loss"] = kd.detach()
    if ema_model is not None and ema_weight > 0:
        with torch.no_grad():
            ema_logits = ema_model(x)["logits"]
        ema_loss = torch.nn.functional.mse_loss(
            out["logits"].float(), ema_logits.float().to(out["logits"].device))
        out["loss"] = out["loss"] + ema_weight * ema_loss
        out["ema_loss"] = ema_loss.detach()
    return out


def main(argv=None) -> int:
    """训练主流程."""
    args = parse_args(argv)
    set_seed(args.seed)
    # CUDA 加速开关（须在首次建卡操作前设置）：可扩展显存段防碎片 + cudnn autotune（短卷积固定形状受益）
    import os as _os

    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # 镜像源尽早生效（import datasets 之后、建流之前打补丁）

    from src.llm.local.data import configure_hf_endpoint

    endpoint = args.hf_endpoint or _os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        configure_hf_endpoint(endpoint)
        print(f"HF 镜像源：{endpoint}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f"设备：{device}，混合精度：{use_amp}，阶段：{args.phase}", flush=True)

    config = build_config(args)
    if args.config_json:
        # 迁移后的结构：以 config.json 为准（仍保证 RoPE 缓存盖住训练序列）
        config = SmallLLMConfig(**json.loads(
            Path(args.config_json).read_text(encoding="utf-8")))
        config.max_seq_len = max(config.max_seq_len, args.seq_len + 8)
        print(f"模型配置已载入：{args.config_json}"
              f"（{config.n_layers} 层，专家 hidden {config.expert_hidden}）",
              flush=True)
    config.grad_ckpt = args.grad_ckpt
    # 检索/记忆：显式 flag 才覆盖 config（默认零变化）
    if args.enable_retro:
        config.retro_enabled = True
        config.retro_every = max(args.retro_every, 0)
        config.retro_chunk_len = args.retro_len
        if config.retro_every > 0:
            if config.retro_chunk_len < 1:
                raise SystemExit("--retro-len 至少为 1")
            print(f"RETRO V2 交错已开：每 {config.retro_every} 层一个融合块，"
                  f"每文档 {config.retro_chunk_len} token（frozen 编码）",
                  flush=True)
    if args.enable_memory:
        config.memory_every = max(args.memory_every, 1)
        config.memory_slots = args.memory_slots
        config.memory_topk = args.memory_topk
        # 槽数要拆两个 √M 子码本，非完全平方数提前拦下（否则构造时 assert 崩栈）
        side = int(config.memory_slots ** 0.5)
        if side * side != config.memory_slots:
            raise SystemExit(
                f"--memory-slots 须为完全平方数（√M×√M 子码本），"
                f"当前 {config.memory_slots}；可取 1024/4096/16384")
        if config.memory_topk < 1:
            raise SystemExit("--memory-topk 至少为 1")
        print(f"记忆层已开：每 {config.memory_every} 层一个，"
              f"槽数 {config.memory_slots}（子码本 {side}×{side}），top-{config.memory_topk}",
              flush=True)
    # 断点续流：先读上次消费数（无文件/无键则从头，兼容旧 checkpoint）
    resume_cursor = 0
    if args.resume:
        _latest = Path(args.ckpt_dir) / "latest.json"
        if _latest.exists():
            try:
                resume_cursor = int(json.loads(
                    _latest.read_text(encoding="utf-8")).get("data_cursor", 0))
                if resume_cursor:
                    print(f"断点续流：跳过已消费 {resume_cursor} 文档", flush=True)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
    # 课程阈值（可调用对象进数据流，每行实时读；无课程时退化为固定值）
    curriculum = parse_curriculum(args.curriculum)
    score_state = {"min": curriculum[0] if curriculum else args.pretrain_min_score}
    if curriculum:
        args.pretrain_min_score = lambda: score_state["min"]  # noqa: E731
    holdout, train_stream = collect_holdout(args, resume_cursor)

    # 分词器：评测文档复用做拟合（省一次采样；SFT 对用 t[-1] 取 response）
    fit_sample = holdout if args.phase == "pretrain" else [t[0] + t[-1] for t in holdout]
    tok = load_tokenizer(args, config, fit_sample)
    eos = SimpleTokenizer.EOS

    # 数据打包流（评测流每次重建， holdout 是 list 可反复迭代）
    encode = tok.encode
    if args.phase == "pretrain":
        train_blocks = pack_pretrain(train_stream, encode, args.seq_len, eos)

        def make_eval() -> PackedBatcher:
            return PackedBatcher(
                pack_pretrain(iter(holdout), encode, args.seq_len, eos), args.batch)
    else:
        sft_stream = train_stream  # collect_holdout 已按阶段返回对应流
        sft_blocks = pack_pairs(sft_stream, encode, args.seq_len, eos)

        def make_eval() -> PackedBatcher:
            return PackedBatcher(
                pack_pairs(iter(holdout), encode, args.seq_len, eos), args.batch)

        if args.shuffle_buffer > 0:
            sft_blocks = ShuffleBuffer(sft_blocks, args.shuffle_buffer, args.seed)
        sft_trains = PackedBatcher(sft_blocks, args.batch)
        trains = sft_trains
        # SFT 掺 pretrain 回放（batch 级混，主流耗尽即停；回放无限循环不计 cursor）
        if args.sft_replay_dir and args.sft_replay_ratio > 0:
            if not 0.0 <= args.sft_replay_ratio < 1.0:
                raise SystemExit("--sft-replay-ratio 须在 [0,1) 内")
            if args.sft_replay_pool < 1:
                raise SystemExit("--sft-replay-pool 至少为 1")
            pool = list(_take(iter_local_texts(args.sft_replay_dir),
                              args.sft_replay_pool))
            if not pool:
                raise SystemExit(f"--sft-replay-dir 无可用文档：{args.sft_replay_dir}")
            random.Random(args.seed).shuffle(pool)
            rep_blocks = pack_pretrain(itertools.cycle(pool), encode,
                                       args.seq_len, eos)
            if args.shuffle_buffer > 0:
                rep_blocks = ShuffleBuffer(rep_blocks, args.shuffle_buffer,
                                           args.seed + 1)
            rep_trains = PackedBatcher(rep_blocks, args.batch)
            denom, num = 20, min(19, max(1, round(args.sft_replay_ratio * 20)))
            trains = interleave_batches(sft_trains, rep_trains, denom - num, num)
            print(f"SFT 回放已开：{len(pool)} 篇 pretrain 循环，"
                  f"batch 占比约 {num}/{denom}", flush=True)
    # block 级 shuffle（只打乱 pretrain 训练流；SFT 分支上面自理；
    # 评测流 make_eval 保持确定性）
    if args.phase == "pretrain" and args.shuffle_buffer > 0:
        train_blocks = ShuffleBuffer(train_blocks, args.shuffle_buffer, args.seed)
    if args.phase == "pretrain":
        trains = PackedBatcher(train_blocks, args.batch)
    # 后台预取（打包在后台线程做，主循环只消费；device 传输仍在主线程）
    if args.prefetch > 0:
        trains = PrefetchIterator(trains, args.prefetch)

    # 模型与优化器
    model = TinyLLM(config).to(device)
    # 扩词（可选）：模型建成后、优化器建成前做，优化器直接看到最终参数
    if args.extend_vocab:
        from src.llm.local.infer import extend_vocab_and_model

        missing = collect_missing(args, tok)
        if missing:
            n_new, resized = extend_vocab_and_model(tok, model, missing)
            print(f"扩词：新增 {n_new} 字（{''.join(missing[:20])}…），"
                  f"{'超限扩行，动量新开' if resized else '复用空行，优化器照常续'}",
                  flush=True)
        else:
            print("扩词：未发现缺字，词表不变", flush=True)
    if args.optimizer == "muon":
        from src.llm.local.optim import build_hybrid_optimizer

        optim = build_hybrid_optimizer(model, muon_lr=args.muon_lr, adam_lr=args.lr)
        print(f"优化器：Muon+AdamW（muon_lr={args.muon_lr}，adam_lr={args.lr}）", flush=True)
    else:
        optim = build_optimizer(model, lr=args.lr)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_step, tokens_seen = 0, 0
    if args.phase == "sft" and args.sft_init:
        model.load_state_dict(torch.load(args.sft_init, map_location=device))
        print(f"SFT 起始权重已加载：{args.sft_init}", flush=True)
    if args.init_checkpoint:
        # 迁移权重：按行续接（新结构），动量新开；与 --resume 同用时只取计数
        load_weights_overlap(model, args.init_checkpoint, map_location=device)
        print(f"迁移权重已加载：{args.init_checkpoint}", flush=True)
    if args.resume:
        if (ckpt_dir / "latest.json").exists() and not args.init_checkpoint:
            meta = load_ckpt(ckpt_dir, model, optim, map_location=device)
            start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
            print(f"已恢复：step={start_step}，tokens={tokens_seen}", flush=True)
        elif (ckpt_dir / "latest.json").exists():
            meta = json.loads((ckpt_dir / "latest.json").read_text(encoding="utf-8"))
            start_step, tokens_seen = meta["step"], meta.get("tokens_seen", 0)
            print(f"已恢复计数 step={start_step}（权重来自 --init-checkpoint，动量新开）",
                  flush=True)
        else:
            # 防呆：无 checkpoint 时 --resume 等价于从零开始，明确告知避免误解
            print(f"警告：{ckpt_dir} 下无 latest.json，--resume 无效，将从零开始训练",
                  flush=True)
    (ckpt_dir / "vocab.json").write_text(
        json.dumps(tok._chars, ensure_ascii=False), encoding="utf-8")
    # 蒸馏老师（可选）：frozen 锚点 KL，id->字映射供解码对齐
    distiller = None
    id_to_char: dict = {i + 4: ch for i, ch in enumerate(tok._chars)}
    if args.kd_teacher:
        from src.llm.local.distill import (
            AnchorDistiller, build_anchor_mapping, load_teacher)

        teacher_model, teacher_tok = load_teacher(args.kd_teacher, str(device))
        anchor_map = build_anchor_mapping(tok._chars, teacher_tok)
        id_to_char = {i + 4: ch for i, ch in enumerate(tok._chars)}
        distiller = AnchorDistiller(teacher_model, teacher_tok, anchor_map,
                                    temperature=args.kd_temp)
        print(f"蒸馏老师已加载：{args.kd_teacher}（锚点 {len(anchor_map)}/{len(tok._chars)}）",
              flush=True)
    # EMA 影子（可选）：bf16 深拷贝，no grad，只做一致性参照
    ema_model = None
    if args.ema_every > 0:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"EMA 影子已建（每 {args.ema_every} 步动量 {args.ema_decay} 同步）",
              flush=True)

    # RETRO 检索（可选）：预构建 BM25 索引，训练时按 batch 检出 top-K 融合
    retro_index = None
    if args.retro_db:
        import pickle

        from src.llm.local.retrieval import BM25Retriever

        try:
            retro_index = BM25Retriever.load(args.retro_db)
        except (OSError, pickle.UnpicklingError, EOFError,
                TypeError, AttributeError) as e:
            print(f"警告：RETRO 库加载失败（{e}），已禁用", flush=True)
            retro_index = None
        if isinstance(retro_index, BM25Retriever):
            print(f"RETRO 检索库已载：{len(retro_index)} 文档", flush=True)
        else:
            print("警告：retro_db 不是 BM25Retriever，已禁用", flush=True)
            retro_index = None

    log_path = ckpt_dir / "train.log"
    logf = open(log_path, "a", encoding="utf-8")
    # 参数落盘 + 启动标记（同一文件多次续跑时，用 run_start 区分段落；
    # 之前日志里出现两个 step=1 就是这么来的）
    hparams = dump_hparams(args, config, ckpt_dir)
    logf.write(json.dumps({"event": "run_start", "step": start_step,
                           "tokens": tokens_seen,
                           "hparams": {k: hparams[k] for k in
                                       ("phase", "preset", "seq_len", "batch",
                                        "accum", "max_steps", "lr", "optimizer",
                                        "grad_ckpt", "pretrain_mix", "sft_mix",
                                        "rho_keep", "rho_ref", "kd_teacher",
                                        "kd_alpha", "replay_ratio",
                                        "consolidate_steps", "extend_vocab",
                                        "sft_replay_ratio", "ema_every")
                                       if k in hparams}},
                          ensure_ascii=False) + "\n")
    logf.flush()
    backend = LocalChatBackend(model, tok)
    # 回放池（难 block 重放；逐 token loss 前向恒开，供评分用）
    from src.llm.local.data import ReplayBuffer

    replay_buf = (ReplayBuffer(args.replay_capacity, args.seed)
                  if args.replay_ratio > 0 or args.consolidate_steps > 0 else None)
    rng = random.Random(args.seed)
    # RETRO 前瞻线程池（循环外建、finally 关；retro 关闭时为 None）
    retr_pool = None
    if retro_index is not None and args.enable_retro:
        from concurrent.futures import ThreadPoolExecutor

        retr_pool = ThreadPoolExecutor(max_workers=1)
    t0 = time.time()
    step = start_step
    # 冠军追踪：同目录续跑时继承历史最佳（best/meta.json），避免更差的覆盖冠军
    best_val = float("inf")
    _best_meta = ckpt_dir / "best" / "meta.json"
    if _best_meta.exists():
        try:
            best_val = float(json.loads(
                _best_meta.read_text(encoding="utf-8")).get("val_loss", float("inf")))
            print(f"历史最佳 val={best_val:.4f}（best/ 保留）", flush=True)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # 巩固期起始 lr（lr 不升：钳制在起始值内）
    lr0 = [g["lr"] for g in optim.param_groups]
    try:
        while step < args.max_steps:
            # 巩固期判定（新分布开头）：回放加码 + lr 钳制
            consolidating = (args.consolidate_steps > 0
                             and step < start_step + args.consolidate_steps)
            replay_ratio = (max(args.replay_ratio, args.consolidate_replay)
                            if consolidating else args.replay_ratio)
            # 学习率调度（warmup+cosine；Muon 组按 muon 峰值同形状缩放；巩固期不升）
            for gi, g in enumerate(optim.param_groups):
                peak = args.muon_lr if g.get("optimizer") == "muon" else args.lr
                scheduled = lr_schedule(step, args.max_steps, args.warmup, peak)
                g["lr"] = min(scheduled, lr0[gi]) if consolidating else scheduled
            # 梯度累积：每个 micro-step 取新 batch（回放命中时从池子取）；
            # KD 降频：只在 micro 下标整除 kd_every 时跑老师（省 7/8 老师前向），
            # kd 项放大 kd_every 倍保期望（--kd-every 1 即旧行为）。
            kd_every = max(args.kd_every, 1)
            optim.zero_grad(set_to_none=True)
            accum_stats: dict = {}
            used = 0
            from_replay = False
            kd_seen: list[float] = []  # KD 只在部分 micro 跑，跨 micro 收集以便日志
            # RETRO 前瞻：后台线程查下个 micro 的 hits，主线程只做 tensor 组装；
            # GPU 算当前 micro 时 CPU 同时检索，0.64s/micro 的检索被掩盖。
            # retro 关闭时 pool 为 None，走直路零开销。
            # （pool 在循环外建、finally 关，见下）

            def _take_batch():
                """取一个 micro batch（回放命中优先），耗尽返回 None."""
                replay_hit = None
                if replay_buf is not None and len(replay_buf) >= args.batch:
                    if rng.random() < replay_ratio:
                        replay_hit = replay_buf.sample(args.batch)
                try:
                    batch = (replay_hit if replay_hit is not None
                             else next(trains))
                except StopIteration:
                    return None
                return batch, replay_hit is not None

            def _retrieve_hits(batch):
                """纯 CPU 检索（线程安全）：batch -> hits（tensor 组装留主线程）."""
                from src.llm.local.retro import retrieve_for_texts

                texts = []
                x_rows = batch[0] if isinstance(batch, tuple) else batch
                for row in x_rows:
                    chars = "".join(
                        id_to_char.get(int(i), '') for i in row.tolist() if int(i) > 3)
                    texts.append(chars[-1000:])
                return retrieve_for_texts(retro_index, texts, k=args.retro_k,
                                          max_terms=12)

            pending = _take_batch()
            pending_fut = (retr_pool.submit(_retrieve_hits, pending[0])
                           if retr_pool is not None and pending is not None
                           else None)
            micro_i = 0
            # 双终止：accum 个数到了停（主条件，与旧 for 语义一致），
            # 数据耗尽也停（pending 为 None；单目录小数据走这条）
            while pending is not None and micro_i < args.accum:
                batch, is_replay = pending
                from_replay = from_replay or is_replay
                # 下一个先取好、检索先交出去，再算当前（重叠窗口）
                pending = _take_batch()
                if retr_pool is not None:
                    hits = (pending_fut.result() if pending_fut is not None
                            else None)
                    pending_fut = (retr_pool.submit(_retrieve_hits, pending[0])
                                   if pending is not None else None)
                else:
                    hits = None
                do_kd = (micro_i % kd_every == 0)
                micro_i += 1
                kd_w = args.kd_alpha * kd_every if do_kd else 0.0
                # RETRO：按 batch 检出 top-K 融合（hits 已就绪，只做 tensor 组装）；
                # retro_every>0 走 V2 交错（chunk token id，模型侧 frozen 编码），
                # =0 走 v1 单点（均值向量 mem）
                mem, mem_mask, chunk_ids, chunk_mask = None, None, None, None
                if hits is not None:
                    from src.llm.local.retro import (
                        build_batch_chunk_ids, build_batch_mem)

                    if args.retro_every > 0:
                        chunk_ids, chunk_mask = build_batch_chunk_ids(
                            tok, hits, args.retro_k, args.retro_len)
                        chunk_ids = chunk_ids.to(device)
                        chunk_mask = chunk_mask.to(device)
                    else:
                        mem, mem_mask = build_batch_mem(
                            model.embed, tok, hits, args.retro_k)
                        mem = mem.to(device)
                        mem_mask = mem_mask.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = _forward_batch(model, batch, device,
                                             distiller, kd_w, id_to_char,
                                             args.rho_keep, ema_model, args.ema_weight,
                                             args.rho_ref, do_kd, mem, mem_mask,
                                             chunk_ids, chunk_mask)
                else:
                    out = _forward_batch(model, batch, device,
                                         distiller, kd_w, id_to_char,
                                         args.rho_keep, ema_model, args.ema_weight,
                                         args.rho_ref, do_kd, mem, mem_mask,
                                         chunk_ids, chunk_mask)
                tokens_seen += batch[0].numel() if isinstance(batch, tuple) else batch.numel()
                (out["loss"] / args.accum).backward()
                accum_stats = {k: float(v.detach()) if torch.is_tensor(v) else v
                               for k, v in out.items() if k.endswith("loss")}
                if "kd_loss" in out:
                    kd_seen.append(float(out["kd_loss"]))
                used += 1
            if used == 0:
                print("数据流耗尽，提前结束", flush=True)
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            # 回放池投喂：取最后 micro 的逐 token loss，最难 1 块入池
            if replay_buf is not None and "token_losses" in out:
                with torch.no_grad():
                    scores = out["token_losses"].detach().float().mean(-1)
                    # batch 可能是 (x, y) 元组（SFT）或单张量
                    base = batch[0] if isinstance(batch, tuple) else batch
                    if scores.numel() == base.shape[0] and scores.numel() > 0:
                        hardest = int(scores.argmax())
                        if isinstance(batch, tuple):
                            replay_buf.push([(batch[0][hardest], batch[1][hardest])])
                        else:
                            replay_buf.push([batch[hardest]])
            # 课程推进 + EMA 同步（都按优化器步数走，与 micro-step 无关）
            if curriculum:
                score_state["min"] = curriculum_value(*curriculum, step)
            if ema_model is not None and step % args.ema_every == 0:
                # decay 按同步间隔换算（每步等效 decay^every；0.999 每 10 步同步
                # ≈ 每步 0.9999，记忆上万步影子冻住——必须换算，否则 best/ 存过期权重）
                eff = args.ema_decay ** max(args.ema_every, 1)
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(eff).add_(
                            p.detach().to(ema_p.dtype), alpha=1 - eff)
            record = {"step": step, "tokens": tokens_seen,
                      "lr": optim.param_groups[0]["lr"],
                      "secs": round(time.time() - t0, 1), **accumuloss(accum_stats)}
            if curriculum:
                record["cur_min_score"] = round(score_state["min"], 3)
            if replay_buf is not None:
                record["replay"] = from_replay
                record["replay_size"] = len(replay_buf)
            if kd_seen:
                # KD 降频时只有部分 micro 有值，取均值（accumuloss 只保留了最后 micro 的键）
                record["kd_loss"] = sum(kd_seen) / len(kd_seen)
            if step % 10 == 0 or step == 1:
                print(f"step={step} loss={record.get('loss', float('nan')):.4f} "
                      f"tokens={tokens_seen} lr={record['lr']:.2e}", flush=True)
                logf.write(json.dumps(record, ensure_ascii=False) + "\n")
                logf.flush()
            if args.eval_every and step % args.eval_every == 0:
                val = evaluate(model, make_eval, args.eval_batches, device)
                record_val: dict = {"step": step, "val_loss": val}
                champ_model, champ_val = model, val
                if ema_model is not None:
                    # EMA 影子同步评（restore_train=False 保 eval 模式），
                    # 冠军按影子值选、存影子权重（平滑冠军）
                    ema_val = evaluate(ema_model, make_eval, args.eval_batches,
                                       device, restore_train=False)
                    record_val["ema_val_loss"] = ema_val
                    champ_model, champ_val = ema_model, ema_val
                msg = f"[eval] step={step} val_loss={val:.4f}"
                if "ema_val_loss" in record_val:
                    msg += f" ema_val={record_val['ema_val_loss']:.4f}"
                print(msg, flush=True)
                logf.write(json.dumps(record_val, ensure_ascii=False) + "\n")
                logf.flush()
                if champ_val < best_val:
                    best_val = champ_val
                    save_best(ckpt_dir, champ_model, step, champ_val)
                    print(f"[best] step={step} val={champ_val:.4f} -> best/",
                          flush=True)
            if args.sample_every and step % args.sample_every == 0:
                resp = backend.chat([{"role": "user", "content": "介绍一下你自己。"}],
                                    max_new_tokens=64)
                print(f"[sample] step={step} {resp['content'][:200]}", flush=True)
                logf.write(json.dumps({"step": step, "sample": resp["content"]},
                                      ensure_ascii=False) + "\n")
                logf.flush()
            if args.save_every and step % args.save_every == 0:
                snap = save_ckpt(ckpt_dir, model, optim, step, tokens_seen,
                                 {"phase": args.phase, "preset": args.preset},
                                 keep_last=args.keep_last,
                                 data_cursor=_data_cursor(train_stream))
                print(f"[ckpt] step={step} -> {snap.name}", flush=True)
    finally:
        if retr_pool is not None:
            retr_pool.shutdown(wait=True)
        logf.close()
    save_ckpt(ckpt_dir, model, optim, step, tokens_seen,
              {"phase": args.phase, "preset": args.preset},
              keep_last=args.keep_last,
              data_cursor=_data_cursor(train_stream))
    print(f"训练结束：step={step}，权重 {ckpt_dir}/model.pt，分词表 {ckpt_dir}/vocab.json",
          flush=True)
    print("试用：LocalChatBackend.load("
          f'"{ckpt_dir}/model", config) 后 backend.chat([...])', flush=True)
    return 0


# 断点续流余量：在途数据（shuffle 蓄水池 + 预取队列）已计数但未训练，
# 存盘时回退这部分，宁可少量重见、不丢数据（单遍训练丢数据不可挽回）
RESUME_SLACK_DOCS = 1024


def _data_cursor(train_stream) -> int:
    """当前消费游标（扣掉在途余量，向下取整到 0）."""
    n = getattr(train_stream, "n", 0) or 0
    return max(n - RESUME_SLACK_DOCS, 0)


def accumuloss(stats: dict) -> dict:
    """重命名累积统计的 loss 键（与 train_step 返回对齐）."""
    out = {}
    for k, v in stats.items():
        if k == "loss":
            out["loss"] = v
        elif k == "main_loss":
            out["main_loss"] = v
        elif k == "mtp_loss":
            out["mtp_loss"] = v
        elif k == "aux_loss":
            out["aux_loss"] = v
        elif k == "kd_loss":
            out["kd_loss"] = v
        elif k == "ema_loss":
            out["ema_loss"] = v
    return out


if __name__ == "__main__":
    raise SystemExit(main())
