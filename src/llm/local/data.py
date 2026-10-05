"""语料与数据流 — 中文预训练 + 指令微调的数据供给.

语料选择（2026 年公开可用的中文语料）：
- 预训练首选 ``epfml/FineWeb2-HQ:cmn_Hani``：FineWeb2 中文 top-10% 质量过滤子集，
  论文称达到同等效果约需 6x 更少 token，非常适合 100M 小模型。
- 预训练回退 ``HuggingFaceFW/fineweb-2:cmn_Hani``：全量中文（543B 词），流式无需下载。
- SFT 用 ``BelleGroup/train_0.5M_CN``：50 万条中文指令（instruction/input/output）。
- 无网/复现实验可用本地 ``.txt/.md/.jsonl`` 目录回退。

全部走流式 + 打包（packing）：多文档拼接后切成定长块，零磁盘占用。
"""

from __future__ import annotations

import json
import queue
import random
import threading
from collections import deque
from pathlib import Path
from typing import Iterator

import torch

from src.llm.local.infer import SimpleTokenizer

# 预训练语料注册表（按 key 引用，mix 时按权重交错）
# MiniCPM 系 Ultra-FineWeb 为中英双 split，取 zh；字段名 content，附质量分 score
PRETRAIN_SOURCES: dict[str, dict] = {
    "hq": {"dataset": "epfml/FineWeb2-HQ", "split": "train",
            "data_files": "cmn_Hani/*.parquet", "text_field": "text"},
    "fineweb2": {"dataset": "HuggingFaceFW/fineweb-2", "split": "train",
                 "data_files": "cmn_Hani/*.parquet", "text_field": "text"},
    "ultrafineweb": {"dataset": "openbmb/Ultra-FineWeb", "split": "train",
                     "data_files": "data/ultrafineweb_zh/*",
                     "text_field": "content", "score_field": "score"},
    # 注：仓库元数据虽标 en/zh 逻辑 split，datasets 只认 train，
    # 靠 data_files 把文件限定在中文目录，split 固定 train
}
DEFAULT_PRETRAIN_MIX = "hq:1"
# SFT 语料注册表：triple（Belle 式 instruction/input/output）与
# messages（OpenAI 式多轮对话，需转成 (prompt, response) 对）两种格式
SFT_SOURCES: dict[str, dict] = {
    "belle": {"dataset": "BelleGroup/train_0.5M_CN", "split": "train",
              "format": "triple"},
    "agent-general": {"dataset": "openbmb/UltraData-SFT-Agent-2609",
                      "config": "General-Agent", "split": "train",
                      "format": "messages"},
    "agent-code": {"dataset": "openbmb/UltraData-SFT-Agent-2609",
                   "config": "Code-Agent", "split": "train",
                   "format": "messages"},
    "agent-search": {"dataset": "openbmb/UltraData-SFT-Agent-2609",
                     "config": "Search-Agent", "split": "train",
                     "format": "messages"},
    "agent-tool": {"dataset": "openbmb/UltraData-SFT-Agent-2609",
                   "config": "Tool-Use", "split": "train",
                   "format": "messages"},
}
DEFAULT_SFT_MIX = "belle:1"

# SFT 模板：prompt 部分 loss 掩掉，只学 output
SFT_PROMPT = "<用户>\n{instruction}{input}\n<助手>\n"
# 被掩掉位置的 label（cross_entropy 默认 ignore_index）
IGNORE = -100


def require_datasets():
    """惰性导入 datasets（HF 流式必需，未装时给安装提示）."""
    try:
        from datasets import load_dataset

        return load_dataset
    except ImportError as e:
        raise ImportError(
            "训练语料流式需要 datasets 库：pip install datasets"
        ) from e


def configure_hf_endpoint(url: str | None) -> str | None:
    """设定 HF 镜像源（代码内生效，不依赖 shell 变量传递）.

    datasets/huggingface_hub 部分版本只在特定路径读环境变量，
    这里把环境变量 + 运行期常量一起打补丁，确保流式真正走镜像。
    返回最终生效的 endpoint。
    """
    if not url:
        return None
    import os

    os.environ["HF_ENDPOINT"] = url
    try:
        from datasets import config as ds_config

        ds_config.HF_ENDPOINT = url
        ds_config.HUB_DATASETS_URL = url + "/datasets/{repo_id}/resolve/{revision}/{path}"
    except ImportError:
        pass
    try:
        from huggingface_hub import constants as hf_constants

        if hasattr(hf_constants, "ENDPOINT"):
            hf_constants.ENDPOINT = url
    except ImportError:
        pass
    return url


def iter_hf_pretrain(
    split: str = "train",
    buffer_size: int = 10000,
    seed: int = 0,
    sources: list | None = None,
    min_score: float = 0.0,
) -> Iterator[str]:
    """流式产生预训练中文文本（单源兼容口，内部走 mix 单源）.

    data_files 限定在中文子目录：否则 datasets 会全仓库列文件
    （几千个 parquet 跨几十个语言目录，慢且在弱网下超时），
    限定后只翻中文前缀。
    sources 兼容旧元组 [(dataset, config_dir)] 与新 spec 字典混用。
    """
    if sources is None:
        specs = [dict(PRETRAIN_SOURCES["hq"])]
    else:
        specs = []
        for item in sources:
            if isinstance(item, dict):
                specs.append(item)
            else:
                name, config = item
                spec = {"dataset": name, "split": split, "text_field": "text"}
                if config is not None:
                    # 只传 data_files 不传 config 名：避免与仓库自带配置冲突
                    spec["data_files"] = f"{config}/*.parquet"
                specs.append(spec)
    errors: list[str] = []
    for spec in specs:
        try:
            yield from _iter_hf_source(spec, buffer_size, seed, min_score)
            return
        except Exception as e:  # 当前源失败则试下一个
            errors.append(f"{spec.get('dataset')}: {e}")
    raise RuntimeError("所有预训练语料源均不可用：" + "；".join(errors))


def _iter_hf_source(
    spec: dict,
    buffer_size: int = 10000,
    seed: int = 0,
    min_score: float = 0.0,
) -> Iterator[str]:
    """单个 HF 源的流式文本（失败抛异常，调用方决定回退/跳过）.

    min_score 可为浮点或零参可调用（课程 schedule 每行实时读阈值）。
    """
    load_dataset = require_datasets()
    kwargs: dict = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    ds = load_dataset(spec["dataset"], **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    text_field = spec.get("text_field", "text")
    score_field = spec.get("score_field")
    get_thr = min_score if callable(min_score) else lambda: min_score
    for row in ds:
        text = (row.get(text_field) or "").strip()
        if len(text) < 32:  # 过滤太短的噪声文档
            continue
        if score_field:
            thr = get_thr()
            if thr > 0:
                try:
                    if float(row.get(score_field) or 0) < thr:
                        continue
                except (TypeError, ValueError):
                    pass
        yield text


def parse_pretrain_mix(mix: str) -> list[tuple[dict, int]]:
    """解析预训练混合配比（注册表固定为 PRETRAIN_SOURCES）."""
    return parse_mix(mix, PRETRAIN_SOURCES)


def parse_mix(mix: str, registry: dict[str, dict]) -> list[tuple[dict, int]]:
    """解析混合配比 "a:1,b:2" -> [(spec, weight)]（未知 key 报错）."""
    out: list[tuple[dict, int]] = []
    for part in mix.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, w = part.rsplit(":", 1)
            weight = max(int(w), 1)
        else:
            key, weight = part, 1
        if key not in registry:
            raise ValueError(
                f"未知语料 {key!r}，可选：{sorted(registry)}")
        out.append((registry[key], weight))
    if not out:
        raise ValueError("混合配比为空")
    return out


def parse_local_mix(value: str) -> list[tuple[str, int]]:
    """解析本地多目录 "dirA:3,dirB:1" -> [(目录, 权重)]（无权重默认 1）."""
    out: list[tuple[str, int]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, w = part.rpartition(":")
        # "dir:3" 形式（rpartition 只切最后一个冒号）；后缀非整数则冒号属目录名
        if path:
            try:
                out.append((path.strip(), max(int(w), 1)))
                continue
            except ValueError:
                pass
        out.append((part, 1))
    if not out:
        raise ValueError("本地语料目录为空")
    return out


class CountedIterator:
    """计数包装：记录迭代器被消费了多少.

    用途有二：训练流跳过评测部分；续跑断点续流（latest.json 存消费数，
    续跑时跳过。prefetch/shuffle 在计数点下游，kill 时在途的少量数据会
    重见，误差有界——见 README）。
    """

    def __init__(self, it):
        self.it = iter(it)
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        v = next(self.it)
        self.n += 1
        return v


# 旧名兼容
_Counted = CountedIterator


def skip_items(stream, n: int) -> None:
    """快进跳过 n 个元素（续跑断点续流，不分词只遍历，耗尽即停）."""
    it = iter(stream)
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            break


def interleave_weighted(
    streams: list[Iterator[str]], weights: list[int]
) -> Iterator[str]:
    """加权轮询交错多源（权重按比例展开一轮模式，耗尽的源移出轮转）."""
    assert len(streams) == len(weights) and streams
    # 展开一轮模式，如权重 [1,2] -> [0,1,1]
    pattern: list[int] = []
    for idx, w in enumerate(weights):
        pattern.extend([idx] * max(w, 1))
    alive = [True] * len(streams)
    iters = [iter(s) for s in streams]
    pos = 0
    while any(alive):
        idx = pattern[pos % len(pattern)]
        pos += 1
        if not alive[idx]:
            continue
        try:
            yield next(iters[idx])
        except StopIteration:
            alive[idx] = False


def interleave_batches(
    main: Iterator, replay: Iterator, w_main: int = 17, w_replay: int = 3,
) -> Iterator:
    """按权重轮询取 batch，主流耗尽即停（replay 侧可无限循环，不决定终止）.

    SFT 掺 pretrain 回放用：主 SFT 流决定 epoch 长度，回放只做正则防过拟合；
    回放流不计 data_cursor（刻意重复，见训练脚本）。
    """
    assert w_main >= 1 and w_replay >= 0
    if w_replay == 0:
        yield from main
        return
    pattern = [0] * w_main + [1] * w_replay
    iters = [iter(main), iter(replay)]
    pos = 0
    while True:
        idx = pattern[pos % len(pattern)]
        pos += 1
        try:
            yield next(iters[idx])
        except StopIteration:
            if idx == 0:
                return
            iters[1] = iter(replay)


def iter_hf_pretrain_mix(
    mix: str = DEFAULT_PRETRAIN_MIX,
    buffer_size: int = 10000,
    seed: int = 0,
    min_score: float = 0.0,
) -> Iterator[str]:
    """按配比混合多源（建流失败的源警告跳过，全失败才报错）."""
    parsed = parse_pretrain_mix(mix)
    streams: list[Iterator[str]] = []
    weights: list[int] = []
    for i, (spec, w) in enumerate(parsed):
        try:
            # 预检：取 1 个文档验证源可用（流式建流本身不触发网络）
            probe = _iter_hf_source(spec, buffer_size, seed + i, min_score)
            first = next(probe)

            def chained(first_doc=first, rest=probe):
                yield first_doc
                yield from rest

            streams.append(chained())
            weights.append(w)
        except Exception as e:
            print(f"警告：语料源 {spec.get('dataset')} 不可用，已跳过（{e}）", flush=True)
    if not streams:
        raise RuntimeError(f"混合配比 {mix!r} 的所有源均不可用")
    if len(streams) == 1:
        yield from streams[0]
        return
    yield from interleave_weighted(streams, weights)


def convo_to_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """多轮对话转 (prompt, response) 对：每个 assistant 消息配之前全部上下文.

    system/user/tool/observation 进上下文（打中文 role 标签），空 content 跳过；
    prompt 为空的不出对（无条件续写不适合 SFT）。
    """
    role_tag = {"system": "系统", "user": "用户", "tool": "工具",
                "observation": "观察", "assistant": "助手"}
    pairs: list[tuple[str, str]] = []
    context: list[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        # content 偶为结构化 tool_call，转字符串保留信息
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        content = content.strip()
        if not content:
            continue
        if role == "assistant":
            prompt = "\n".join(context)
            if prompt:
                pairs.append((prompt, content))
            context.append(f"<助手>\n{content}")
        else:
            context.append(f"<{role_tag.get(role, role)}>\n{content}")
    return pairs


def iter_hf_sft(
    split: str = "train",
    buffer_size: int = 10000,
    seed: int = 0,
) -> Iterator[tuple[str, str]]:
    """SFT 对流（默认 Belle，兼容口；triple 已在此套好模板）."""
    yield from _iter_sft_source(dict(SFT_SOURCES["belle"]), buffer_size, seed)


def _iter_sft_source(spec: dict, buffer_size: int, seed: int) -> Iterator[tuple[str, str]]:
    """单个 SFT 源的 (prompt, response) 对流.

    triple 源在此套模板，messages 源的上下文自带 role 标签、直接可用，
    下游 pack_pairs 不再二次包装。
    """
    load_dataset = require_datasets()
    kwargs: dict = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    ds = load_dataset(spec["dataset"], **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    if spec.get("format", "triple") == "messages":
        for row in ds:
            yield from convo_to_pairs(row.get("messages") or [])
    else:
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            output = (row.get("output") or "").strip()
            if instruction and output:
                prompt = SFT_PROMPT.format(
                    instruction=instruction, input=row.get("input") or "")
                yield prompt, output


def iter_hf_sft_mix(
    mix: str = DEFAULT_SFT_MIX,
    buffer_size: int = 10000,
    seed: int = 0,
) -> Iterator[tuple[str, str]]:
    """按配比混合多 SFT 源的 (prompt, response) 对（建流失败的源警告跳过）."""
    parsed = parse_mix(mix, SFT_SOURCES)
    streams: list[Iterator] = []
    weights: list[int] = []
    for i, (spec, w) in enumerate(parsed):
        try:
            gen = _iter_sft_source(spec, buffer_size, seed + i)
            first = next(gen)

            def chained(first_item=first, rest=gen):
                yield first_item
                yield from rest

            streams.append(chained())
            weights.append(w)
        except Exception as e:
            print(f"警告：SFT 源 {spec.get('dataset')} 不可用，已跳过（{e}）", flush=True)
    if not streams:
        raise RuntimeError(f"SFT 混合配比 {mix!r} 的所有源均不可用")
    if len(streams) == 1:
        yield from streams[0]
        return
    yield from interleave_weighted(streams, weights)


def iter_local_sft(root: str | Path) -> Iterator[tuple[str, str, str]]:
    """遍历本地 Belle 格式 SFT 文件（.jsonl/.csv，instruction/input/output 列）.

    无网回退 / 魔搭落盘用；字段缺失的行跳过。
    """
    import csv as _csv

    root = Path(root)
    files = sorted(
        [p for p in root.rglob("*") if p.suffix.lower() in {".jsonl", ".csv"}]
    )
    if not files:
        raise RuntimeError(f"本地 SFT 目录无可用文件：{root}")
    for path in files:
        if path.suffix.lower() == ".csv":
            with open(path, encoding="utf-8", errors="ignore", newline="") as f:
                for row in _csv.DictReader(f):
                    instruction = (row.get("instruction") or "").strip()
                    output = (row.get("output") or "").strip()
                    if instruction and output:
                        yield instruction, row.get("input") or "", output
        else:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    instruction = (obj.get("instruction") or "").strip()
                    output = (obj.get("output") or "").strip()
                    if instruction and output:
                        yield instruction, obj.get("input") or "", output


def iter_local_texts(root: str | Path) -> Iterator[str]:
    """遍历本地目录的 .txt/.md/.jsonl/.parquet 文本（无网回退/复现实验用）.

    jsonl 每行取 text/content/body 字段，取不到则整行当文本；
    parquet 取 text 列（FineWeb2 系分片直读，pyarrow 惰性导入）。
    """
    root = Path(root)
    files = sorted(
        [p for p in root.rglob("*")
         if p.suffix.lower() in {".txt", ".md", ".jsonl", ".parquet"}]
    )
    if not files:
        raise RuntimeError(f"本地语料目录无可用文件：{root}")
    for path in files:
        if path.suffix.lower() == ".parquet":
            yield from _iter_parquet_texts(path)
        elif path.suffix.lower() == ".jsonl":
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = (
                            obj.get("text") or obj.get("content") or obj.get("body") or ""
                        )
                    except json.JSONDecodeError:
                        text = line
                    if len(text.strip()) >= 32:
                        yield text.strip()
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            # 按空行切文档，避免超长单文档
            for chunk in text.split("\n\n"):
                chunk = chunk.strip()
                if len(chunk) >= 32:
                    yield chunk


def _iter_parquet_texts(path: Path) -> Iterator[str]:
    """分批读 parquet 的 text 列（只取所需列，embeddings 等大列不进内存）."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("读 parquet 需要 pyarrow：pip install pyarrow") from e
    pf = pq.ParquetFile(path)
    col = next((c for c in ("text", "content", "body") if c in pf.schema.names), None)
    if col is None:
        raise RuntimeError(f"{path} 无 text/content/body 列，可用列：{pf.schema.names[:8]}")
    for batch in pf.iter_batches(batch_size=1024, columns=[col]):
        for text in batch.column(col).to_pylist():
            if text and len(text.strip()) >= 32:
                yield text.strip()


def fit_tokenizer_on_stream(
    texts: Iterator[str],
    vocab_size: int,
    max_chars: int = 20_000_000,
) -> SimpleTokenizer:
    """在文本流上采样拟合字符分词器（截断到 max_chars 防爆内存）."""
    buf: list[str] = []
    total = 0
    for text in texts:
        buf.append(text)
        total += len(text)
        if total >= max_chars:
            break
    tok = SimpleTokenizer(vocab_size)
    tok.fit(buf)
    return tok


def pack_pretrain(
    texts: Iterator[str],
    encode,
    seq_len: int,
    eos_id: int,
) -> Iterator[torch.Tensor]:
    """预训练打包：拼接文档 + EOS，切成 (seq_len+1) 块（+1 留给 label 错位）."""
    target = seq_len + 1
    buf: list[int] = []
    for text in texts:
        buf.extend(encode(text, add_bos=False))
        buf.append(eos_id)
        while len(buf) >= target:
            yield torch.tensor(buf[:target], dtype=torch.long)
            buf = buf[target:]


def pack_sft(
    triples: Iterator[tuple[str, str, str]],
    encode,
    seq_len: int,
    eos_id: int,
    bos_id: int = SimpleTokenizer.BOS,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """SFT 打包（Belle 三元组兼容口）：套模板后走 pack_pairs，行为与旧版一致."""
    def as_pairs():
        for instruction, inp, output in triples:
            yield (SFT_PROMPT.format(instruction=instruction, input=inp or ""),
                   output)

    yield from pack_pairs(as_pairs(), encode, seq_len, eos_id, bos_id)


def pack_pairs(
    pairs: Iterator[tuple[str, str]],
    encode,
    seq_len: int,
    eos_id: int,
    bos_id: int = SimpleTokenizer.BOS,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """SFT 打包：prompt 掩 IGNORE，只在 output 上算 loss，多样本拼满一块.

    pairs 为已成型的 (prompt, response)，模板/对话转换由上游完成，此处不二次包装。
    """
    target = seq_len + 1
    ids: list[int] = [bos_id]
    labs: list[int] = [IGNORE]

    def emit() -> tuple[torch.Tensor, torch.Tensor] | None:
        """发出当前块（pad 到定长）."""
        if len(ids) < 8:  # 太短的块丢弃
            return None
        pad = target - len(ids)
        full_ids = ids + [eos_id] * pad
        full_labs = labs + [IGNORE] * pad
        return torch.tensor(full_ids[:target], dtype=torch.long), torch.tensor(
            full_labs[:target], dtype=torch.long
        )

    for prompt, output in pairs:
        p_ids = encode(prompt, add_bos=False)
        o_ids = encode(output, add_bos=False) + [eos_id]
        # 超长单样本直接截断，保证单块内只放完整样本
        if len(p_ids) + len(o_ids) + 1 > target:
            keep = target - len(p_ids) - 2
            if keep <= 0:
                continue
            o_ids = o_ids[:keep] + [eos_id]
        # 放不下则先发出当前块，另起一块
        if len(ids) + len(p_ids) + len(o_ids) > target:
            packed = emit()
            if packed is not None:
                yield packed
            ids, labs = [bos_id], [IGNORE]
        ids.extend(p_ids)
        labs.extend([IGNORE] * len(p_ids))
        ids.extend(o_ids)
        labs.extend(o_ids)
    packed = emit()
    if packed is not None:
        yield packed


class ShuffleBuffer:
    """block 级 shuffle：蓄水池随机吐块，平滑爬取顺序的站点聚集.

    本地 parquet 按爬取顺序存（相邻文档同站点同模板），不打乱则 loss 呈台阶跳变；
    HF 流式虽有文档级 shuffle，打包拼接仍保留局部顺序，同样受益。
    评测流不要包这一层（保持确定性）。
    """

    def __init__(self, blocks, capacity: int = 512, seed: int = 0):
        self.blocks = blocks
        self.capacity = capacity
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator:
        """蓄满后随机吐出并补充；流尽后把池子洗牌倒空（每个块恰好产出一次）."""
        pool: list = []
        for b in self.blocks:
            pool.append(b)
            if len(pool) >= self.capacity:
                idx = self.rng.randrange(len(pool))
                pool[idx], pool[-1] = pool[-1], pool[idx]
                yield pool.pop()
        self.rng.shuffle(pool)
        yield from pool


class ReplayBuffer:
    """难样本回放池：存 RHO 选出的难 block，按比例重放（海马体式巩固）.

    存 CPU 张量（SFT 连 label 一起存元组），上限 FIFO 淘汰；
    sample(n) 随机拼 batch，不足返回 None（调用方回落主数据流）。
    """

    def __init__(self, capacity: int = 8192, seed: int = 0):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, batch) -> None:
        """推入并按行存：(b,S) 批量 / (x,y) 批量元组 / 单 block / 单对 / 它们的列表."""
        if isinstance(batch, list):
            for b in batch:
                self.push(b)
            return
        if isinstance(batch, tuple):
            xs, ys = batch
            if xs.dim() == 1:
                self.buf.append((xs.detach().cpu(), ys.detach().cpu()))
            else:
                for i in range(xs.shape[0]):
                    self.buf.append((xs[i].detach().cpu(), ys[i].detach().cpu()))
            return
        if batch.dim() == 1:
            self.buf.append(batch.detach().cpu())
        else:
            for i in range(batch.shape[0]):
                self.buf.append(batch[i].detach().cpu())

    def sample(self, n: int):
        """随机取 n 个拼 batch；不足返回 None."""
        if len(self.buf) < n:
            return None
        picks = self.rng.sample(range(len(self.buf)), n)
        items = [self.buf[i] for i in picks]
        if isinstance(items[0], tuple):
            return (torch.stack([x for x, _ in items]),
                    torch.stack([y for _, y in items]))
        return torch.stack(items)


class PrefetchIterator:
    """后台预取：单独线程提前打包后几个 batch，主线程只管消费，GPU 不等 CPU.

    batch 保留在 CPU，主循环转 `.to(device)` 时才上卡（CUDA 操作留在主线程）。
    顺序与原流一致（FIFO），异常会抛给消费端，流尽抛 StopIteration。
    """

    def __init__(self, source, capacity: int = 4):
        self._source = iter(source)
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._error: list = []
        self._done = False
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self) -> None:
        """后台生产：正常结束或异常都放哨兵，避免消费端永久阻塞."""
        try:
            for item in self._source:
                self._queue.put(item)
        except Exception as e:  # noqa: BLE001 - 透传给消费端
            self._error.append(e)
        finally:
            self._queue.put(None)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self):
        """取一批；哨兵后标记结束（再次调用直接 StopIteration，不卡死）."""
        if self._done:
            raise StopIteration
        item = self._queue.get()
        if item is None:
            self._done = True
            if self._error:
                raise self._error[0]
            raise StopIteration
        return item


class PackedBatcher:
    """把定长块攒成 (batch, seq_len+1) 批次；模型内部自动错位算 loss."""
    def __init__(self, blocks: Iterator[torch.Tensor] | Iterator[tuple], batch_size: int):
        self.blocks = iter(blocks)  # 统一转迭代器（ShuffleBuffer 等只实现 __iter__ 的也可接入）
        self.batch_size = batch_size
        self.batches = 0  # 已产出批次数
        self.tokens = 0  # 已产出 token 数（含 SFT 的掩位，统计口径一致）

    def __iter__(self) -> "PackedBatcher":
        return self

    def __next__(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """取一批；数据耗尽抛 StopIteration."""
        buf = []
        for _ in range(self.batch_size):
            buf.append(next(self.blocks))  # 耗尽时自然抛出 StopIteration
        self.batches += 1
        if isinstance(buf[0], tuple):
            x = torch.stack([b[0] for b in buf])
            y = torch.stack([b[1] for b in buf])
            self.tokens += x.numel()
            return x, y
        x = torch.stack(buf)
        self.tokens += x.numel()
        return x
