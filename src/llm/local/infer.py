"""本地推理封装 — 字符级分词器 + chat 接口（对接现有 Runner 调试用）.

生产级分词（BPE/SentencePiece）可后续替换 SimpleTokenizer；
chat 消息格式与 src/llm/client.py 的规范化响应保持一致。
"""

from __future__ import annotations

import torch

from src.llm.local.config import SmallLLMConfig
from src.llm.local.model import TinyLLM


class SimpleTokenizer:
    """字符级分词器（零依赖，演示端到端 chat 用）.

    保留 4 个特殊 token：<pad>=0, <bos>=1, <eos>=2, <unk>=3，
    其余按字符序映射到 4..vocab_size-1。
    扩词规则：只允许末尾追加，旧 id 永不移位（已训权重逐行兼容的前提）。

    **追加式子词**（`docs/details/subword-vocab.md`）：`_chars` 存的是「表面串」，
    既可以是单个字符，也可以是 `"the"` / `"\":"` 这样的多字符子词。存在多字符项时
    `encode` 自动改用**最长匹配**；不存在时走原来的逐字符路径（逐位一致）。
    检测靠 `_sync_meta()` 的长度指纹，所以外部直接改 `_chars` / `_ids` 的旧脚本
    仍然正确：它们没有多字符项 → `_multi=False` → 老路径。
    """

    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._chars: list[str] = []
        self._ids: dict[str, int] = {}
        # 多字符 token 缓存：`(len(_chars), len(_ids))` 变了就重建
        # （脚本里 `tok._chars = json.load(...)` 后自动生效，无需再调任何方法）
        self._meta_key: tuple[int, int] | None = None
        self._max_len: int = 1
        self._multi: bool = False

    def _sync_meta(self) -> None:
        """按长度指纹重建「是否含多字符 token / 最长几字符」."""
        key = (len(self._chars), len(self._ids))
        if key == self._meta_key:
            return
        self._meta_key = key
        self._max_len = 1
        self._multi = False
        for s in self._chars:
            n = len(s)
            if n > 1:            # 空串/单字符不算；空串绝不能进匹配循环
                self._multi = True
                if n > self._max_len:
                    self._max_len = n

    def fit(self, texts: list[str]) -> None:
        """从语料收集字符表（按频次排序，截断到 vocab 上限）."""
        from collections import Counter

        cnt = Counter("".join(texts))
        budget = self.vocab_size - 4
        self._chars = [ch for ch, _ in cnt.most_common(budget)]
        self._ids = {ch: i + 4 for i, ch in enumerate(self._chars)}
        self._meta_key = None

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """编码为 id 序列（有多字符 token 时按最长匹配切）."""
        self._sync_meta()
        if self._multi:
            ids: list[int] = []
            i, n = 0, len(text)
            max_len = self._max_len
            while i < n:
                take = 0
                tid = -1
                for length in range(min(max_len, n - i), 1, -1):
                    hit = self._ids.get(text[i:i + length])
                    if hit is not None:
                        tid, take = hit, length
                        break
                if take == 0:
                    # 退到单字符；不在表里则 UNK（与老行为一致）
                    ids.append(self._ids.get(text[i], self.UNK))
                    i += 1
                else:
                    ids.append(tid)
                    i += take
        else:
            ids = [self._ids.get(ch, self.UNK) for ch in text]
        return [self.BOS] + ids if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        """解码为文本（跳过特殊 token；多字符 token 直接拼表面串）."""
        inv = {i + 4: ch for i, ch in enumerate(self._chars)}
        return "".join(inv[i] for i in ids if i in inv)



def extend_vocab_and_model(
    tokenizer: SimpleTokenizer,
    model: TinyLLM,
    new_chars: list[str],
    init_std: float = 0.02,
) -> tuple[int, bool]:
    """追加表面串并落位 embedding（旧 id 稳定，旧行原样保留）.

    `new_chars` 可以是单字符，也可以是 `"the"` / `"\\":"` 这样的多字符子词
    （见 `docs/details/subword-vocab.md`）；分词器按 `_chars` 自动切到最长匹配。
    两条路：容量内（常见）复用空行并重初始化（旧 random 值经 decay 已漂移），
    形状全不变，优化器照常续；超限才扩行（tied 重绑），优化器需新开。
    返回 (新增数, 是否扩行）。
    """
    fresh = [ch for ch in new_chars if ch not in tokenizer._ids]
    if not fresh:
        return 0, False
    base = len(tokenizer._chars)
    cap = model.embed.num_embeddings - 4
    if base + len(fresh) <= cap:
        # 落在现有容量内：只扩词表 + 重初始化空行
        tokenizer._chars.extend(fresh)
        tokenizer._ids = {ch: i + 4 for i, ch in enumerate(tokenizer._chars)}
        tokenizer.vocab_size += len(fresh)
        with torch.no_grad():
            torch.nn.init.normal_(
                model.embed.weight[base + 4:base + 4 + len(fresh)], std=init_std)
            if model.lm_head.weight is not model.embed.weight:
                torch.nn.init.normal_(
                    model.lm_head.weight[base + 4:base + 4 + len(fresh)], std=init_std)
        return len(fresh), False
    # 超限扩行
    tokenizer._chars.extend(fresh)
    tokenizer._ids = {ch: i + 4 for i, ch in enumerate(tokenizer._chars)}
    tokenizer.vocab_size += len(fresh)

    def _grow_embed(mod: torch.nn.Embedding) -> None:
        """embedding 扩行（设备/精度保持，旧行拷贝）."""
        old_w = mod.weight.data
        old_n, d = old_w.shape
        w = torch.empty((old_n + len(fresh), d),
                        device=old_w.device, dtype=old_w.dtype)
        w[:old_n].copy_(old_w)
        torch.nn.init.normal_(w[old_n:], std=init_std)
        mod.weight = torch.nn.Parameter(w)
        if hasattr(mod, "num_embeddings"):
            mod.num_embeddings = old_n + len(fresh)

    old_n = model.embed.num_embeddings
    was_tied = model.lm_head.weight is model.embed.weight
    _grow_embed(model.embed)
    if was_tied:
        model.lm_head.weight = model.embed.weight  # 重绑：换 Parameter 后旧引用失效
    elif model.lm_head.weight.shape[0] == old_n:
        _grow_embed(model.lm_head)
    model.config.vocab_size = len(tokenizer._chars) + 4
    return len(fresh), True


_ROLE_MARKERS = ("<系统>", "<用户>", "<助手>")


def sanitize_role_markers(text: str) -> str:
    """转义文本里的角色标记，防止用户内容伪造对话轮次.

    模板用 `<用户>` / `<助手>` 分隔轮次；若用户输入原样携带 `<助手>`，
    模型看到的就是一个"提前开始的助手轮"，等于提示注入。
    字符级分词器没有"特殊 token id"可拦，只能在拼模板前把标记的尖括号
    换成全角（`＜助手＞`），语义仍是同一句话，但不再匹配模板标记。
    """
    for marker in _ROLE_MARKERS:
        text = text.replace(marker, marker.replace("<", "＜").replace(">", "＞"))
    return text


class LocalChatBackend:
    """本地模型 chat 封装：messages -> 生成回复（OpenAI 兼容的返回形状）."""

    def __init__(self, model: TinyLLM, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _prepare_ids(self, messages: list[dict], max_new_tokens: int,
                     effort: object = None,
                     ) -> tuple["torch.Tensor", "torch.device"]:
        """拼 prompt + 编码 + 截断 + 上设备（chat/stream 共用）.

        `effort`（1–100 或 low/high/max）非空时，先把
        `Reasoning Effort: {b} ...` 前置进系统提示（V4.1 §5.1.4）。
        None = 不改提示（与老行为逐位一致）。
        """
        from src.llm.local.effort import apply_effort_messages, resolve_effort

        level = resolve_effort(effort)
        if level is not None:
            messages = apply_effort_messages(messages, level)
        # role 标签与 SFT 模板统一用中文（此前采样用英文 <user>/<assistant>，
        # 与 SFT_PROMPT 的 <用户>/<助手> 不一致，已统一）
        role_cn = {"system": "系统", "user": "用户", "assistant": "助手"}
        parts = []
        for m in messages:
            role = role_cn.get(m.get("role", "user"), m.get("role", "user"))
            content = sanitize_role_markers(m.get("content") or "")
            parts.append(f"<{role}>\n{content}")
        prompt = "\n".join(parts) + "\n<助手>\n"
        ids = self.tokenizer.encode(prompt)
        # 截断到模型上下文（保尾部，演示策略）
        max_ctx = self.model.config.max_seq_len - max_new_tokens
        ids = ids[-max_ctx:]
        input_ids = torch.tensor([ids], dtype=torch.long)
        # 切到模型所在设备
        device = next(self.model.parameters()).device
        return input_ids.to(device), device

    def _make_constraint(self, response_format: object):
        """`response_format` → 约束对象（None=不约束）.

        只支持单序列；schema 不合法时抛 `JsonSchemaError`（服务端转 400）。
        """
        from src.llm.local.structured import (
            JsonGrammar,
            JsonGrammarConstraint,
            parse_response_format,
        )

        schema = parse_response_format(response_format)
        if schema is None:
            return None
        # 词表以模型 logits 宽度为准：分词器字符表可能短于模型容量
        # （未训练的空行不可解码，被空串挡在 mask 外）
        vocab = int(self.model.config.vocab_size)
        tokens = [""] * vocab
        for i, ch in enumerate(self.tokenizer._chars[:max(0, vocab - 4)]):
            tokens[i + 4] = ch
        constraint = JsonGrammarConstraint(
            JsonGrammar(schema), tokens, SimpleTokenizer.EOS,
            vocab_size=vocab)
        constraint.check_available()
        return constraint

    def validate_response_format(self, response_format: object) -> None:
        """提前校验 `response_format`（服务端在写 SSE 头之前调，好回 400）."""
        self._make_constraint(response_format)

    def _inject_tools(self, messages: list[dict], tools: list[dict],
                      target: str = "") -> list[dict]:
        """把工具声明渲染成系统提示，替换（或插入）system 消息.

        渲染走 `toolcall.render_system` —— 与 `scripts/synth_tool_sft.py`
        造数据时**同一份实现**，保证训练/推理逐字节一致（这是工具调用最容易
        静默坏掉的地方，详见 `src/llm/local/toolcall.py` 模块说明）。

        原 system 内容直接当 `persona` 槽用：训练数据的 system 段就是这个形状
        （persona + 目录 + 规则），所以用户自定义的开场白能自然融进去；
        没传 system 时用 `DEFAULT_PERSONA`。
        """
        from src.llm.local.toolcall import (
            DEFAULT_PERSONA,
            normalize_tools,
            render_catalog,
            render_params,
            render_system,
        )

        normalized = normalize_tools(tools)
        idx = next((i for i, m in enumerate(messages)
                    if (m.get("role") or "") == "system"), None)
        persona = ((messages[idx].get("content") if idx is not None else "")
                   or DEFAULT_PERSONA).strip()
        params = ""
        if target:
            hit = next((t for t in normalized if t["path"] == target), None)
            if hit is not None:
                params = render_params(hit)
        text = render_system(persona, render_catalog(normalized, exclude=target),
                            params)
        out = [dict(m) for m in messages]
        if idx is None:
            out.insert(0, {"role": "system", "content": text})
        else:
            out[idx] = {"role": "system", "content": text}
        return out

    def _tool_response_format(self, tools: list[dict], target: str,
                              response_format: object) -> object:
        """工具调用默认走约束解码（只保证「两个字段的 JSON」，path 合法性后判）."""
        if not tools or response_format is not None:
            return response_format
        from src.llm.local.toolcall import CALL_SCHEMA

        return {"type": "json_schema",
                "json_schema": {"name": "tool_call", "schema": CALL_SCHEMA}}

    def _parse_tool_output(self, text: str, tools: list[dict]) -> dict | None:
        """解析模型输出；路径不在候选里时当作没调用（返回 None）.

        候选校验必须在这里做：模型完全可能拼出一个 schema 里不存在的路径，
        那是幻觉，不能交给工具执行（真实运行时会直接报「工具不存在」）。

        路径匹配按「精确 → 唯一末段名」两级：训练数据的 path 是
        `分组/工具名`（如 `workspace/shell_exec`），而 OpenAI `tools` 声明里
        只有 `name`（`shell_exec`）。返回的 path 统一用**调用方提供的那份**，
        这样上层不用再做一次映射。
        """
        from src.llm.local.toolcall import normalize_tools, parse_tool_call

        call = parse_tool_call(text)
        if call is None or not call["path"]:
            return None
        normalized = normalize_tools(tools)
        if not normalized:
            return call
        want = call["path"]
        hit = next((t for t in normalized if t["path"] == want), None)
        if hit is None:
            # 末段名唯一时放行（`workspace/shell_exec` ↔ `shell_exec`）
            tail = want.split("/")[-1]
            same = [t for t in normalized if t["path"].split("/")[-1] == tail]
            if len(same) == 1:
                hit = same[0]
        if hit is None:
            return None
        return {"path": hit["path"], "arguments": call["arguments"]}

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        effort: object = None,
        response_format: object = None,
        seed: int | None = None,
        tools: list[dict] | None = None,
        tool_target: str = "",
    ) -> dict:
        """对话生成，返回 {"content","reasoning","tool_calls","finish_reason"}.

        `effort`：推理努力档（1–100 / low / high / max）；None = 不注入指令。
        `response_format`：结构化输出（`{"type":"json_object"}` 或
        `{"type":"json_schema",...}`）；约束解码保证输出是合法 JSON，
        未在 max_new_tokens 内闭合时 `finish_reason="length"`。
        `seed`：确定性采样（同一 prompt + seed 逐位可复现，与 batch 组成无关）。

        `tools`：工具声明（OpenAI `{"type":"function",...}` 或本仓库 schema
        条目都行）。非空时把工具目录渲染进系统提示，默认加 `CALL_SCHEMA`
        约束解码，输出解析成 OpenAI `tool_calls`：
        - 模型给了非空 path 且在候选里 → `tool_calls` 有值、`content` 为空；
        - path 为空（「都做不了」）或不在候选里（幻觉）→ `tool_calls=None`，
          `content` 保留原文（前端能看出模型说了什么）；
        - `arguments` 为空对象 → SELECT，调用方补 `tool_target` 再来一轮拿参数。
        `tool_target`：非空时渲染 FILL 形态（把该工具的完整参数表附进提示），
        用来实现「先选工具、再填参数」两步（见 `toolcall.py`）。
        """
        if tools:
            messages = self._inject_tools(messages, tools, tool_target)
            response_format = self._tool_response_format(tools, tool_target,
                                                         response_format)
        input_ids, _ = self._prepare_ids(messages, max_new_tokens, effort)
        constraint = self._make_constraint(response_format)
        gen = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_id=SimpleTokenizer.EOS,
            constraint=constraint,
            seed=seed,
        )
        new_ids = gen[0].tolist()[len(input_ids[0]):]
        done = constraint is None or bool(constraint.finished)
        text = self.tokenizer.decode(new_ids)
        tool_calls = None
        if tools:
            from src.llm.local.toolcall import to_openai_tool_calls

            call = self._parse_tool_output(text, tools)
            if call is not None:
                tool_calls = to_openai_tool_calls(call["path"], call["arguments"])
        return {
            "content": "" if tool_calls else text,
            "reasoning": None,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else (
                "stop" if done else "length"),
        }

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        effort: object = None,
        response_format: object = None,
        seed: int | None = None,
        tools: list[dict] | None = None,
        tool_target: str = "",
    ):
        """流式生成，逐块 yield 文本（SSE 用；字符级分词保证增量解码精确）.

        `effort` 语义同 `chat`；`response_format` 同样生效（流式也走约束，
        下游拿到的是逐字符的合法 JSON 前缀）；`seed` 语义同 `chat`；
        `tools`/`tool_target` 只做提示注入与约束（同 `chat`），但**不解析**
        `tool_calls` —— 流式没法在吐字节的同时给出结构化结果，调用方攒完
        全文用 `toolcall.parse_tool_call` 自己解析。
        """
        if tools:
            messages = self._inject_tools(messages, tools, tool_target)
            response_format = self._tool_response_format(tools, tool_target,
                                                         response_format)
        input_ids, _ = self._prepare_ids(messages, max_new_tokens, effort)
        constraint = self._make_constraint(response_format)
        inv = {i + 4: ch for i, ch in enumerate(self.tokenizer._chars)}
        for tid in self.model.stream_tokens(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_id=SimpleTokenizer.EOS,
                constraint=constraint,
                seed=seed):
            ch = inv.get(tid)
            if ch:  # 特殊 token（BOS/EOS/UNK）跳过不吐
                yield ch

    def save(self, path: str) -> None:
        """保存权重 + 字符表."""
        import json

        torch.save(self.model.state_dict(), path + ".pt")
        with open(path + ".vocab.json", "w", encoding="utf-8") as f:
            json.dump(self._chars_safe(), f, ensure_ascii=False)

    def _chars_safe(self) -> list[str]:
        """字符表导出."""
        return self.tokenizer._chars

    @classmethod
    def load(cls, path: str, config: SmallLLMConfig,
             dtype: str | None = None) -> "LocalChatBackend":
        """加载权重 + 字符表.

        词表按顺序找：path.vocab.json（save 默认）→ 同目录 vocab.json（训练落盘布局）。
        dtype: None=fp32 原样；"fp16"/"bf16" 半精度推理（RMSNorm 内部 fp32，
        half 安全；logits 精度降一档，贪心 argmax 基本不变，采样温度建议重验证）。
        """
        import json
        from pathlib import Path as _Path

        model = TinyLLM(config)
        model.load_state_dict(torch.load(path + ".pt", map_location="cpu"))
        if dtype is not None:
            dmap = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp32": torch.float32}
            if dtype not in dmap:
                raise ValueError(f"dtype 须为 {sorted(dmap)}，当前 {dtype}")
            model = model.to(dmap[dtype])
        model.eval()
        # 专家 offload / LES：按配置为各层 MoE 挂搬运器（默认关时无操作）
        model.setup_offload()
        tok = SimpleTokenizer(config.vocab_size)
        candidates = [path + ".vocab.json",
                      str(_Path(path).parent / "vocab.json")]
        for vocab_path in candidates:
            try:
                with open(vocab_path, encoding="utf-8") as f:
                    tok._chars = json.load(f)
                break
            except (OSError, json.JSONDecodeError):
                continue
        else:
            raise FileNotFoundError(f"找不到分词表（试过 {candidates}）")
        tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
        return cls(model, tok)
