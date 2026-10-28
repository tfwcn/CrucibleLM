"""挖高频子词 — 产出 `--subword-vocab` 用的追加表（只追加，旧 id 不移位）.

## 为什么需要

本仓库的分词器是**纯字符级**（`SimpleTokenizer`，base run 实测 4839 字），
一个字 = 一个 token。代价在非中文内容上最明显，实测（`data/llm-ckpt/vocab.json`）：

    工具 schema（96 字符）  →  96 token     典型 BPE 约 30 token
    工具调用（76 字符）     →  76 token     典型 BPE 约 26 token
    英文句子（65 字符）     →  65 token     典型 BPE 约 15 token

也就是说 2048 的上下文实际只装得下 2048 个字符，一张三工具 schema 就吃掉 15%。
把 3000 个高频子词追加到词表末尾，可以在**不动任何旧 embedding 行**的前提下
把英文/JSON/常用词组的 token 数压下来（本机实测见脚本末尾报告）。

## 做法

1. 按 `--corpus`（与训练同一套 `目录[:权重]` 语法）流式读 `--total-chars` 个字符；
2. 统计 n=2..`--nmax` 的 n-gram，分块合并 + 定期剪枝控制内存；
3. 打分 `count × (n-1)`（每命中一次省 n-1 个 token），排序取前 `--limit`；
4. 过滤：仅保留字符集已在词表内的串、纯空白串，以及会跟旧单字符重复的项；
5. 落盘 JSON 字符串列表（`--out`），并打印压缩比与工具调用探针的前后 token 数。

产物直接喂给训练脚本：

    python scripts/build_subwords.py \\
        --corpus "data/hq-zh:3,data/ultra-zh:1" \\
        --vocab data/llm-ckpt/vocab.json \\
        --out data/subwords-zh.json

    python scripts/train_local_llm.py --resume ... \\
        --subword-vocab data/subwords-zh.json --holdout-file data/holdout.json

注意：追加子词会**改变 token 序列本身**（同一段文本切法变了），所以它是
「warm restart」而不是「无损续训」——旧权重能直接用（id 稳定），但新 token 的
embedding 是随机初始化的，需要一段重新适配期。详见 `docs/details/subword-vocab.md`。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHUNK_CHARS = 500_000          # 单块字符数（控制局部 Counter 峰值内存）
PRUNE_MULT = 3                 # glob 体积超过 keep*PRUNE_MULT 就剪枝


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="挖高频子词（追加式扩词表）")
    p.add_argument("--corpus", required=True,
                   help="语料目录，支持 目录[:权重],目录[:权重] 混合（与 --local-path 同语法）")
    p.add_argument("--vocab", default="",
                   help="现有 vocab.json（用于算容量余量、限定字符集；建议必填）")
    p.add_argument("--out", required=True, help="输出 JSON（字符串列表，交给 --subword-vocab）")
    p.add_argument("--limit", type=int, default=3000, help="最多选多少个子词")
    p.add_argument("--nmax", type=int, default=5,
                   help="最长子词字符数（2..nmax）。实测 5 最优：4 抓不住 arguments/toolbox，"
                        "8 的候选会被高频短串挤出 keep 榜")
    p.add_argument("--total-chars", type=int, default=8_000_000,
                   help="统计多少字符（越大越准，8M 约 30 秒）")
    p.add_argument("--min-count", type=int, default=8, help="最低出现次数")
    p.add_argument("--keep", type=int, default=400_000, help="合并期保留的候选数")
    p.add_argument("--vocab-size", type=int, default=8192,
                   help="模型词表容量（embedding 行数；用于容量校验）")
    p.add_argument("--boost-file", default="",
                   help="额外加权文本（如工具调用样例），提升其 n-gram 优先级")
    p.add_argument("--boost-frac", type=float, default=0.05,
                   help="加权文本在统计字符里的目标占比（0=不加权）")
    p.add_argument("--probe", default="",
                   help="探针文本文件（默认用内置工具调用样例，打印前后 token 数）")
    p.add_argument("--report-chars", type=int, default=200_000,
                   help="压缩比实测用的字符数（0=跳过）")
    return p.parse_args(argv)


def _looks_like_sft(root: Path) -> bool:
    """目录像不像 SFT 语料（有 .csv，或首行 jsonl 带 instruction 字段）."""
    if any(p.suffix.lower() == ".csv" for p in Path(root).rglob("*")):
        return True
    for path in sorted(Path(root).rglob("*.jsonl")):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and "instruction" in obj:
                        return True
                    if isinstance(obj, dict) and "text" in obj:
                        return False
        except OSError:
            continue
    return False


def iter_corpus_texts(spec: str) -> Iterator[str]:
    """按 `目录[:权重]` 读本地语料文本（复用训练侧同一套解析器）.

    自动区分两种目录：SFT（instruction/input/output）与预训练（text/parquet）。
    子词挖掘要的是「真实会被编码的文本」，所以 SFT 目录必须把三元组拼回
    一句完整的话，否则异步的问答字段会被当成独立文档统计。
    """
    from src.llm.local.data import (
        interleave_weighted,
        iter_local_sft,
        iter_local_texts,
        parse_local_mix,
    )

    def _stream(root: str) -> Iterator[str]:
        """单目录 → 文本流（SFT 拼提示+回复，pretrain 直读）."""
        if _looks_like_sft(Path(root)):
            for instruction, inp, output in iter_local_sft(root):
                yield f"{instruction}{inp or ''}\n{output}"
        else:
            yield from iter_local_texts(root)

    dirs = parse_local_mix(spec)
    streams = [_stream(path) for path, _ in dirs]
    if len(streams) == 1:
        yield from streams[0]
        return
    yield from interleave_weighted(streams, [w for _, w in dirs])


def with_boost(base: Iterator[str], boosted: str, frac: float) -> Iterator[str]:
    """按目标占比把加权文本插进基础流（不是追加到末尾 —— 末尾会被截断掉）.

    每消费约 `len(boosted) × (1-frac)/frac` 个基础字符，就插一次加权文本，
    使加权文本在总字符里约占 `frac`。`frac<=0` 时原样透传。
    """
    if frac <= 0 or not boosted:
        yield from base
        return
    interval = max(len(boosted), int(len(boosted) * (1.0 - frac) / frac))
    acc = 0
    for text in base:
        yield text
        acc += len(text)
        if acc >= interval:
            yield boosted
            acc = 0


def iter_chunks(texts: Iterator[str], chunk_chars: int = CHUNK_CHARS) -> Iterator[str]:
    """把文本流攒成定长字符块（统计 n-gram 用，避免跨块截断太频繁）."""
    buf: list[str] = []
    size = 0
    for text in texts:
        buf.append(text)
        size += len(text)
        if size >= chunk_chars:
            yield "".join(buf)
            buf, size = [], 0
    if buf:
        yield "".join(buf)


def count_ngrams(texts: Iterator[str], nmax: int, total_chars: int,
                 keep: int) -> tuple[Counter, int]:
    """统计 n=2..nmax 的 n-gram 频次（分块合并 + 定期剪枝，返回 (计数, 消费字符数)）."""
    total: Counter = Counter()
    consumed = 0
    for chunk in iter_chunks(texts):
        local: Counter = Counter()
        n = len(chunk)
        for k in range(2, max(2, nmax) + 1):
            for i in range(n - k + 1):
                local[chunk[i:i + k]] += 1
        total.update(local)
        consumed += n
        if len(total) > keep * PRUNE_MULT:
            total = Counter(dict(total.most_common(keep)))
        if consumed >= total_chars:
            break
    return total, consumed


def select_subwords(counts: Counter, charset: set[str], limit: int,
                    min_count: int) -> list[str]:
    """按 `count × (n-1)` 排序选子词（只留字符集内的、非纯空白的）."""
    scored: list[tuple[float, str]] = []
    for text, cnt in counts.items():
        if cnt < min_count or len(text) < 2:
            continue
        if text.isspace():          # 纯空白（缩进/换行串）不值得占词表
            continue
        if charset and any(ch not in charset for ch in text):
            continue
        scored.append((cnt * (len(text) - 1), text))
    # 同分时长子词优先（一次覆盖更多字符）
    scored.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
    out: list[str] = []
    seen: set[str] = set()
    for _, text in scored:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_tokenizer(chars: list[str], subwords: list[str], vocab_size: int):
    """装配「字符表 + 追加子词」的分词器（只追加，旧 id 不变）."""
    from src.llm.local.infer import SimpleTokenizer

    tok = SimpleTokenizer(vocab_size)
    tok._chars = list(chars) + [s for s in subwords if s not in set(chars)]
    tok._ids = {s: i + 4 for i, s in enumerate(tok._chars)}
    return tok


def token_count(text: str, chars: list[str], subwords: list[str],
                vocab_size: int) -> tuple[int, int]:
    """返回 (字符级 token 数, 追加子词后 token 数)，均不含 BOS."""
    before = len(build_tokenizer(chars, [], vocab_size).encode(text, add_bos=False))
    after = len(build_tokenizer(chars, subwords, vocab_size).encode(text, add_bos=False))
    return before, after


PROBE_PARTS: list[tuple[str, str]] = [
    ("工具 schema",
     '{"type":"function","function":{"name":"shell_exec","description":'
     '"在工作区执行 shell 命令","parameters":{"type":"object","properties":'
     '{"command":{"type":"string"}},"required":["command"],'
     '"additionalProperties":false}}}'),
    ("OpenAI tool_calls",
     '{\n  "id": "call_b68ec5374e8845fc94d1c581",\n  "type": "function",\n'
     '  "function": {"name": "toolbox_call", "arguments": "{\\"path\\": '
     '\\"agent/subagent_run\\", \\"arguments\\": {\\"agent\\": '
     '\\"公主连结小助手\\", \\"task\\": \\"查询当前公会战排名\\"}}"}\n}'),
    ("结果信封", '{"ok":false,"error":"网络超时","retryable":true}'),
    ("中文对话",
     '<用户>\n请读取 docs/ARCHITECTURE.md 并报告其中的层数\n<助手>\n'
     '好的喵～本小姐这就去查。'),
]
DEFAULT_PROBE = "\n".join(text for _, text in PROBE_PARTS)


def main(argv=None) -> int:
    """主流程：统计 → 选择 → 校验 → 落盘 → 报告."""
    args = parse_args(argv)

    chars: list[str] = []
    if args.vocab:
        chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    charset = set(chars)
    budget = args.vocab_size - 4 - len(chars)
    if args.vocab and budget <= 0:
        print(f"词表已满（{len(chars)}+4 >= {args.vocab_size}），无法追加", file=sys.stderr)
        return 2
    limit = args.limit if not args.vocab else min(args.limit, budget)
    if limit < args.limit:
        print(f"容量限制：--limit {args.limit} → {limit}"
              f"（余量 {budget} 行，超出会扩行导致动量新开）")

    texts = iter_corpus_texts(args.corpus)
    if args.boost_file and args.boost_frac > 0:
        boosted = Path(args.boost_file).read_text(encoding="utf-8")
        texts = with_boost(texts, boosted, args.boost_frac)
        print(f"加权文本：{args.boost_file}"
              f"（占统计字符约 {args.boost_frac:.0%}）")

    counts, consumed = count_ngrams(texts, args.nmax, args.total_chars, args.keep)
    print(f"统计 {consumed / 1e6:.1f}M 字符，n-gram 候选 {len(counts)}")

    subwords = select_subwords(counts, charset, limit, args.min_count)
    print(f"选中子词 {len(subwords)} 个（min_count={args.min_count}，nmax={args.nmax}）")

    capacity = args.vocab_size - 4
    used = len(chars) + len(subwords)
    print(f"词表占用：{len(chars)} 字符 + {len(subwords)} 子词 = {used}/{capacity}"
          f"（余 {capacity - used} 行）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subwords, ensure_ascii=False), encoding="utf-8")
    print(f"已写出：{out}")

    # 报告：语料样本压缩比 + 工具调用探针（分段，便于看是哪类内容受益）
    if args.report_chars > 0:
        sample_texts = iter_corpus_texts(args.corpus)
        sample = "".join(list(iter_chunks(sample_texts, args.report_chars))[:1])[:args.report_chars]
        if sample:
            b, a = token_count(sample, chars, subwords, args.vocab_size)
            ratio = b / max(a, 1)
            print(f"语料样本 {len(sample)} 字符：{b} → {a} token"
                  f"（压缩 {ratio:.2f}x，{b / max(len(sample), 1):.2f} → "
                  f"{a / max(len(sample), 1):.2f} token/字符）")
    probe = DEFAULT_PROBE
    if args.probe:
        probe = Path(args.probe).read_text(encoding="utf-8")
    b, a = token_count(probe, chars, subwords, args.vocab_size)
    print(f"工具调用探针 {len(probe)} 字符：{b} → {a} token（{b / max(a, 1):.2f}x）")
    if not args.probe:
        # 分段报告：看清是哪类内容受益（schema 与 tool_calls 才是关键）
        for label, part in PROBE_PARTS:
            pb, pa = token_count(part, chars, subwords, args.vocab_size)
            print(f"  · {label:18s} {len(part):3d} 字符  {pb:3d} → {pa:3d} token"
                  f"  （{pb / max(pa, 1):.2f}x）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
