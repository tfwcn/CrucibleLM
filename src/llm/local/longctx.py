"""超长上下文（到 1M）的阶段管理与自检 — V4.1 的"64K 稀疏起训 + YaRN 外推".

三件事，都是**训练配方**层面的约束（不是新结构）：

1. **64K 稀疏起训、不做 dense 预热**：论文明确"train from scratch at a sequence
   length of 64K, without any dense attention warmup stages"。反过来做（先 dense 短
   序列、再切稀疏长序列）会让注意力的可见集合在训练中途突变，路由/索引器都来不及
   适配——所以这里在**启动时**就把这种配置拦下来。
2. **阶段延展**：64K 起训，到某个 token 预算再拉长（论文 34T 时拉到 1M）。
   `seq_len_at()` 给纯函数，训练脚本可以按已消费 token 决定下一阶段长度。
3. **YaRN 参数得配套**：`yarn_original_len` 要等于**真正训过的长度**，
   `yarn_scale = 目标/训练长度`（4K→1M 就是 256 倍）；对不上就会"外推但没校正"。

本模块只做**纯逻辑 + 自检**，不改模型结构、不碰权重，因此对老 checkpoint 零影响。
"""

from __future__ import annotations

# 长上下文的起训门槛（V4.1 用 64K 起训；到这条线就必须有稀疏注意力）
LONG_CTX_MIN = 65536
# 1M 上下文的"超长"门槛（超过它，解码成本必须靠 SWA/CSA 压住）
XL_CTX = 262144


def parse_ctx_stages(spec: str) -> list[tuple[int, int]]:
    """解析阶段配置 `"0:65536,34000000000:1048576"` → `[(0,65536),(34e9,1048576)]`.

    每项是 `已消费token数:序列长度`；按起点排序，起点重复时后者覆盖前者。
    空串返回空列表（= 不启用阶段延展）。
    """
    out: dict[int, int] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"阶段配置项须为 'token数:序列长度'，当前 {item!r}")
        tok, seq = item.split(":", 1)
        try:
            t, s = int(tok), int(seq)
        except ValueError as exc:
            raise ValueError(f"阶段配置项须为整数，当前 {item!r}") from exc
        if t < 0 or s < 1:
            raise ValueError(f"阶段配置须 token≥0、seq≥1，当前 {item!r}")
        out[t] = s
    return sorted(out.items())


def seq_len_at(tokens_seen: int, stages: list[tuple[int, int]],
               default: int) -> int:
    """按"已消费 token 数"返回当前阶段的序列长度（无阶段配置时返回 default）."""
    if not stages:
        return int(default)
    cur = int(default)
    for start, seq in stages:
        if tokens_seen >= start:
            cur = int(seq)
        else:
            break
    return cur


def sparse_ready(config) -> bool:
    """是否具备稀疏注意力（任一条都算：索引器 / CSA 借用 / 层内 SWA / 稀疏阈值）."""
    if int(getattr(config, "sparse_indexer", 0) or 0) > 0:
        return True
    if str(getattr(config, "csa_modes", "") or "").strip():
        return True
    if int(getattr(config, "csa_reuse_every", 0) or 0) > 1:
        return True
    if int(getattr(config, "swa_window", 0) or 0) > 0:
        return True
    # 稀疏阈值 < 训练长度：等价于"长序列走 sink+窗口+跨步"
    return int(getattr(config, "sparse_threshold", 0) or 0) < LONG_CTX_MIN


def rope_cache_bytes(seq_len: int, dim: int, bytes_per: int = 4) -> int:
    """RoPE cos/sin 缓存的字节数（`2 × seq × dim × 字节`）——长上下文的第一笔固定开销."""
    return 2 * int(seq_len) * int(dim) * int(bytes_per)


def suggest_yarn(train_len: int, target_len: int) -> dict:
    """按训练/目标长度给出 YaRN 参数（直接塞进配置即可）."""
    from src.llm.local.rope import yarn_scaling_factor

    return {
        "yarn_original_len": int(train_len),
        "yarn_scale": float(yarn_scaling_factor(train_len, target_len)),
    }


def validate_longctx(config, seq_len: int) -> list[str]:
    """启动自检：返回警告列表；**致命配置**直接抛 `ValueError`.

    致命（会训坏而不是训慢）：
    - `seq_len > max_seq_len`：位置编码/缓存根本装不下；
    - `seq_len ≥ 64K` 且完全没有稀疏注意力：违反"64K 稀疏起训、无 dense 预热"。
    警告（能跑但大概率白烧算力）：
    - 长于训练长度却没开 YaRN（`yarn_scale<=1`）；
    - `yarn_original_len` 与 `seq_len` 差 8 倍以上（分界算错）；
    - 超长（≥256K）却没开层内 SWA；
    - RoPE 缓存 > 512MB。
    """
    from src.llm.local.rope import yarn_scale_mismatch

    cfg = config
    seq_len = int(seq_len)
    warns: list[str] = []
    max_len = int(getattr(cfg, "max_seq_len", 0) or 0)
    if seq_len > max_len:
        raise ValueError(
            f"训练长度 {seq_len} 超过 max_seq_len={max_len}："
            f"位置缓存装不下（把 max_seq_len 一起调大）")
    if seq_len >= LONG_CTX_MIN and not sparse_ready(cfg):
        raise ValueError(
            f"训练长度 {seq_len} ≥ {LONG_CTX_MIN}，但没有任何稀疏注意力"
            f"（sparse_indexer/csa_modes/csa_reuse_every/swa_window/sparse_threshold）。"
            f"V4.1 口径是\"64K 稀疏起训、不做 dense 预热\"："
            f"先开稀疏再上长序列，否则训练中途可见集合突变，路由与索引器都来不及适配")
    original = int(getattr(cfg, "yarn_original_len", 0) or 0) or seq_len
    scale = float(getattr(cfg, "yarn_scale", 1.0) or 1.0)
    if seq_len > original and scale <= 1.0:
        warns.append(
            f"训练长度 {seq_len} 超过 yarn_original_len={original}，但 yarn_scale=1："
            f"等于裸外推（建议 {suggest_yarn(original, seq_len)}）")
    if yarn_scale_mismatch(scale, original, seq_len):
        warns.append(
            f"yarn_scale={scale} 与 {original}→{seq_len} 的比例对不上："
            f"建议 {suggest_yarn(original, seq_len)}")
    if seq_len >= XL_CTX and int(getattr(cfg, "swa_window", 0) or 0) <= 0:
        warns.append(
            f"序列长度 {seq_len} ≥ {XL_CTX} 但层内 SWA 关闭："
            f"解码每步仍要扫全量 KV（V4.1 的\"context 长 256 倍、Decode FLOPs 只涨 1/4\""
            f"靠的就是 SWA/CED/CSA2）")
    dim = int(getattr(cfg, "qk_rope_dim", 0) or 0)
    if dim > 0:
        nbytes = rope_cache_bytes(seq_len, dim)
        if nbytes > 256 * 1024 ** 2:
            warns.append(f"RoPE 缓存约 {nbytes / 1024 ** 2:.0f}MB（seq×dim 太大）")
    return warns


def format_report(config, seq_len: int) -> str:
    """把自检结果格式化成一行行中文（训练脚本启动日志用）."""
    warns = validate_longctx(config, seq_len)
    if not warns:
        return f"长上下文自检通过（seq_len={seq_len}）"
    return "\n".join(f"  警告：{w}" for w in warns)
