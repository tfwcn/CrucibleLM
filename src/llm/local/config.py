"""超参数 — 默认配置约 128M 总参数 / 约 48M 激活参数，16G 显存可从零训练."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmallLLMConfig:
    """迷你模型超参数."""

    # 词表大小（小词表是 100M 量级可行的关键）
    vocab_size: int = 8192
    # 模型 hidden 维度
    d_model: int = 768
    # Transformer 层数
    n_layers: int = 12
    # 每隔几层放一个 MLA 全注意力层，其余为线性注意力层（Qwen3-Next 式混合）
    full_attn_every: int = 4
    # 注意力头数（head_dim = d_model // n_heads = 64）
    n_heads: int = 12
    # MLA：Q 下投影秩
    q_lora_rank: int = 192
    # MLA：KV 联合压缩秩（远小于 n_heads*head_dim，省 KV 缓存的关键）
    kv_lora_rank: int = 128
    # MLA：解耦 RoPE 维度（Q/K 共享同一份旋转）
    qk_rope_dim: int = 32
    # MLA 分组输出投影（DeepSeek-V4.1：o_groups=8 + 每组中间维 o_lora_rank=1024）：
    # 把 n_heads 分成 o_groups 组，每组头输出先降维到 o_lora_rank，拼接后一次升维回
    # d_model，省输出投影参数（本机 12 头/head 64 为例，o_groups=4+o_lora_rank=192）。
    # o_groups<=1 或 o_lora_rank<=0 = 关（稠密 w_o，键/数值与旧实现逐位一致）。
    # **默认关**（区别于 1/2）：本项**改变 state_dict 键**（w_o → w_o_d/w_o_u），旧
    # checkpoint 严格载入会报缺键，故按约定 5 归为需迁移/重训的 opt-in——开启需重训，
    # 或走 `migrate.group_output_projection`（对 w_o 分组 SVD，近似无损，附重构误差）。
    o_groups: int = 1
    o_lora_rank: int = 0
    # MoE 专家总数（细粒度小专家）
    n_experts: int = 16
    # 每 token 激活专家数
    top_k: int = 4
    # 单个专家中间维度（小专家，控制总参数量）
    expert_hidden: int = 192
    # 共享专家个数（常驻知识通道）
    n_shared: int = 1
    # 负载均衡 aux loss 系数
    aux_loss_coef: float = 0.01
    # SwiGLU 限幅（DeepSeek-V4.1 `swiglu_limit=10.0`）：gate/up 预激活对称截断到
    # [-limit, limit] 再进 silu/相乘，压住激活离群值（个别维度爆炸主导残差流）。
    # 0=关闭（逐位等价旧实现）。只压极值尾部，正常量级下几乎不触发；但仍按"改前向语义"
    # 处理：续训/装载从 checkpoint 的 hparams.json 沿用，读不到按 0（旧行为），见
    # sync_arch_from_checkpoint。docs/details/swiglu-limit.md
    swiglu_limit: float = 10.0

    # ===== 路由（DeepSeek-V4.1：sqrtsoftplus 打分 + noaux_tc 免辅助损失 + routed scaling）=====
    # True=免辅助损失路由（V4.1）：score=sqrt(softplus(logits))，按 score+逐专家偏置 选
    # top-k，权重取原始 score 归一化后 ×routed_scaling_factor；不产生 aux loss，改由偏置
    # 按负载 sign 式均衡（update_router_bias）。False=旧口径（softmax + greedy + aux loss）。
    # 默认开；旧 run 靠 hparams 对齐（fallback False=旧语义）。
    router_aux_free: bool = True
    # 免辅助损失偏置更新率 γ（sign 式；0=冻结偏置）
    router_bias_lr: float = 1e-3
    # 路由权重缩放（V4.1 routed_scaling_factor=1.5；仅 router_aux_free 生效）
    routed_scaling_factor: float = 1.5
    # top-k 权重是否归一化（V4.1=true；旧口径恒归一化）
    norm_topk_prob: bool = True
    # MTP 额外预测步数：1=只预测 t+2（旧口径，键兼容）；D>1=级联预测 t+2…t+1+D
    # （DeepSeek-V4.1 num_nextn_predict_layers=3）。**默认 1**：D>1 每级新增浅层头
    # `mtp.extra.*`，**改变 state_dict 键**，旧 checkpoint 需重训，故属 opt-in；
    # 推理/起草只用第 1 级，主干前向不变（MTP 是训练期辅助目标）。
    mtp_depth: int = 1
    # MTP loss 权重
    mtp_loss_weight: float = 0.3
    # 最大上下文长度
    max_seq_len: int = 8192
    # RoPE 基频
    rope_theta: float = 10000.0
    # YaRN 外推缩放因子（>1 时扩展有效上下文）
    yarn_scale: float = 2.0
    # dropout
    dropout: float = 0.0
    # embedding 与输出头是否共享权重（省约 6M 参数）
    tie_embeddings: bool = True
    # 梯度检查点：逐层重算激活，seq2048+ 上下文跑进 16G 的关键（训练时开启）
    grad_ckpt: bool = False
    # 轻量稀疏注意力（DSA 思想简化版）：超阈值自动切 sink+窗口+跨步，200K 上下文用
    sparse_threshold: int = 4096  # 短于此走 dense，与 MLA 数学一致
    sparse_window: int = 4096  # 滑动窗口（每个查询回看这么多）
    sparse_sink: int = 128  # 注意力汇点（打头保留）
    sparse_stride: int = 512  # 膨胀跨步（窗口外每隔这么多保留一个）
    sparse_chunk: int = 2048  # 预填充分块（显存/速度折中）
    # ===== 分层稀疏 indexer（DeepSeek-V4.1 CSA2：学习式 top-K + 块级候选池）=====
    # False=沿用固定图案（sink+跨步+窗口全聚集，见 sparse_attn.select_keys）；
    # True=用轻量学习式 indexer 从候选集中选 Top-K 主 KV（块级候选池 → 池内 token 级
    # top-K，即"分层"），并在后续层间复用候选池/选择（见 csa_modes）。
    # **新增参数 w_iq/w_ik → 改 state_dict 键**，故默认关（opt-in，需重训）。
    # 口径说明：我们的候选集本就是常数规模，indexer 改的是"读哪些主 KV"（学习式选择），
    # 而非张量形状；真实加速需 block-sparse kernel（未实现）。
    sparse_indexer: bool = False
    # indexer 头数与每头维度（V4.1：32 头 / 128 维）
    indexer_heads: int = 4
    indexer_head_dim: int = 64
    # 每个查询最终读的 token 级主 KV 条数（V4.1：attention top-k = 512）
    indexer_topk: int = 64
    # indexer 蒸馏损失系数（DSA 做法：让学习式打分拟合主注意力在候选上的分布；
    # **必须 >0 否则 indexer 只进掩码、拿不到任何梯度**；0=不训 indexer）
    indexer_loss_coef: float = 0.01
    # 块级候选池：块大小（按候选的全局位置分块）与选中的块数（V4.1：2048 块 × 8 位置）
    indexer_block: int = 8
    indexer_pool_blocks: int = 32
    # CSA2 跨层 KV 复用（DeepSeek-V4.1）：每 N 个全注意力层里 1 个 Full（自算 latent），
    # 其余 Reuse（复用最近 Full 层的 latent，自算 Q/上投影，不重复存 KV 缓存）。
    # 实测（本机 128M 续训 2500 步）：8192 上下文 KV −30%，val_loss +0.012（相对 +0.27%），
    # 收敛后差距不归零——是"有代价的收益"，故默认开（2），要无损兼容或对照时传 1。
    # 1=全 Full（逐位兼容旧 checkpoint，无共享）；2=Full/Reuse 交替。
    # 注意：改默认值只影响"新建模型"；续训/迁移时训练脚本会从 checkpoint 的
    # hparams.json 沿用架构，读不到则保守按 1 处理，避免静默换架构。
    csa_reuse_every: int = 2
    # CSA2 三模式静态分配（V4.1）：逐"全注意力层"指定 Full/Reindex/Reuse。
    # 逗号分隔，按全注意力层序号循环；别名 full/f、reindex/ri、reuse/re。
    #   例 "full,reindex,reuse"：第 1 个全注意力层自算全局 KV + 索引（Full）；
    #   之后 Reindex 层复用 main KV + indexer K、但用自己的 indexer Q 重选 Top-K；
    #   Reuse 层直接复用最新的 Top-K（V4.1 CSA2 decode 的静态分配思路）。
    # 空 = 由 `csa_reuse_every` 推导旧行为（Full/Reuse 交替，Reuse 复用最新选择），
    # 逐位兼容旧 checkpoint；非空时**覆盖** `csa_reuse_every`。
    # 语义变更但**不改 state_dict 键**（`csa_reuse_every` 同族）；Reindex 只在
    # `sparse_indexer` 开启时才与 Reuse 有区别（无 indexer 时退化为 Reuse，
    # 因为选择与 query 内容无关、逐层相同）——见 docs/details/csa-modes.md。
    csa_modes: str = ""
    # 显式"KV 源层"表（V4.1 非均匀分配，如 "2,8,14,20"）：列出的**全注意力层**为
    # 生产者（Full，自算自存 main KV + 自选 Top-K），其余全注意力层复用最近生产者的
    # main KV（Reuse）——间距可以任意非均匀（这正是与"每 N 层统一"的区别）。
    # 为空 = 按 `csa_modes`（空则 `csa_reuse_every`）均匀分配。
    # 合法性：只能列全注意力层（`is_full_attn_layer`），否则报错（早失败）；
    # 首个全注意力层若不在表内会退化为 Full 兜底（否则无共享源）。
    # 与 `csa_modes` 的关系：本表只决定"谁生产"，模式仍取 `csa_modes` 该位置的值
    # （消费层若被指定为 reindex 则仍是 reindex）；未指定则生产者 Full、消费者 Reuse。
    # 同族语义开关：不改 state_dict 键，续训/迁移自动沿用 checkpoint。
    kv_source_layer_ids: str = ""
    # 逐层主 KV 压缩比（V4.1 `CSA2(ratio, mode)`）：按**全注意力层**序号循环的整数表
    # （如 "2" 或 "1,2"）。m = 每个 main KV 条目聚合的连续 token 数（**不重叠**，
    # CSA2 去掉 CSA 的相邻块重叠）：m=1 = 不压缩（默认，逐位等价旧行为）。
    # m>1 时远端 main KV 按 m 块池化（块内逐位置逐通道可学习权重 `z_cmp`，初值 1
    # =算术平均），本地窗口与当前块保持**未压缩**（V4.1：SWA KV 不压缩、逐层本地）；
    # 块的代表位置取块尾。效果：同样的远端列数预算下，覆盖的历史跨度 ×m。
    # 改键（m>1 的层新增 `z_cmp`）→ **默认关**（空串），续训/迁移从 hparams 沿用。
    # 注意：压缩开启后该层恒走分块路径（短序列也不再走 dense），否则同一模型在
    # 短/长序列会看到两种 KV 表示；且借用层（reuse/reindex）沿用生产者的压缩比。
    compress_ratios: str = ""
    # 层内 SWA 窗口（V4.1：**每层**自带滑动窗口注意力，与全局主 KV 完全独立）：
    # 0 = 关（默认，无新增参数/键，逐位兼容旧 ckpt）；n>0 时每层多一条
    # `swa.w_k`/`swa.w_kr`/`swa.w_v` 分支（**改键** → opt-in），窗口 n 个 token，
    # 缓存为定长环（`(2*n_heads*head_dim + qk_rope_dim)*n` 个元素/层，**与序列长度
    # 无关**；落盘默认丢环，读回用最近 n 个 token 重放重建）。
    # 全局主 KV 只出"选中条目"（论文：selected entries + layer-local SWA KV），
    # 本地近端完全由 SWA 承担；`sparse_window` 仅在 SWA 关闭时才有意义。
    # 窗口别照搬 `sparse_window`：环是逐 token 的满维 K/V，n 大时环比主 KV 还贵
    # （base 规模 n=4096 时 3 层共 38.6MB，比 32K 全局主 KV 的 21MB 还多），建议 ≤512。
    swa_window: int = 0
    # CED 编码器-解码器（V4.1 §2.2，YoCo 路线的结构改进）：前 `ced_split` 层当
    # **因果编码器**，其余层当**解码器**——解码器的全局 KV 不再由自己的 hidden
    # 产生，而是由第 `ced_split` 层的输出 `H_enc` 用**逐层投影权重**投影
    # （`C_l = H_enc W_l^{KV}`、`Z_l = H_enc W_l^{Z}`，rope 键同理），于是推理
    # 预填充只需在前缀上算完编码器层，解码器层只跑最后 `swa_window` 个 token
    # （其输出仅服务于自己的 SWA 环）→ 预填充计算量近半（论文 O(NL)→O(NL/2)）。
    # 0 = 关（默认，无新增参数/键，逐位兼容旧 ckpt）；>0 时解码器层的
    # `w_dkv`/`w_kr` 换成 `w_ced_kv`/`w_ced_kr`（**改键** → opt-in）。
    # 约束（早失败）：`0 < ced_split < n_layers`，且解码器区间必须**全是全注意力层**
    # （线性层的递推状态没法从 `H_enc` 投影，必须逐位置跑，跳算就不成立——
    # 本仓 1:3 hybrid 的 base 配置因此不满足，需要 `full_attn_every=1` 之类的配置）。
    # 与 CSA2 复用互斥：CED 解码器层恒为 Full（各层投影权重本就逐层不同，
    # 借用会把 layer-dependent 投影又合并回一份缓存）。
    ced_split: int = 0
    # CED 预填充跳算（只在 `ced_split>0` 时有意义，默认**开**）：解码器层只在前
    # `ced_split` 层之后**最后 n_win 个 token** 上跑（V4.1 §3.2.2 Decoder SWA
    # Bounded Replay）。`swa_window=0` 时退化为"只跑最后 1 个 token"，与全量
    # prefill **逐位等价**（解码器层没有局部依赖）；`swa_window>0` 时是论文承认的
    # 近似（重建出的解码器 SWA 与全量前向不严格相等）。关掉即"全量 prefill"，
    # 严格对齐训练口径，代价是失去 CED 的省算。
    ced_prefill_skip: bool = True
    # KV 缓存量化（V4.1 §2.4.4）："" = 关（默认）；"fp4" = 主 KV 用 MXFP4
    # （E2M1 + 每 16 通道 E4M3 scale，4.5 bit/通道）；"fp8" = 主 KV 用 E4M3（8.5 bit）。
    # 训练期在 `_kv_latent`/`_rope_key` 出口做 fake-quant（STE）→ 即 QAT；推理期同一
    # 数值，区别只在存储（`QuantBuffer` 打包，读时反量化再算注意力——不需要 FP4 矩阵乘）。
    # **默认关**：改缓存数值（精度损失）且改缓存布局（扩容器走 QuantBuffer）。
    # 真省存储：FP4 相对 bf16 约 3.4x（latent+rope 一起算），相对 FP8 约 1.9x。
    kv_quant: str = ""
    # 块主序缓存（V4.1 #4 的显存半边）：`compress_ratio>1` 时**缓存只存池化后的块行**
    # （每 m 个 token 折成一行，凑满即一次池化+量化），于是主 KV 行数 `L → m+ceil(L/m)`。
    # 代价与前提：近端不再由主 KV 的逐 token 窗口提供——**必须开层内 SWA**
    # （`swa_window>=m`，论文里近端正是由 SWA 环承担的），否则当前块的 token 无处可读。
    # 默认关（改缓存布局 + 依赖 SWA 组合；不改 state_dict 键，续训从 hparams 沿用）。
    kv_block_cache: bool = False
    # 线性注意力分块阈值：超长时切块递推（前向精确，反向块内截断），线性层 O(N^2) 的解药
    linear_chunk: int = 2048
    # RETRO 融合：False=模块不存在（state_dict 兼容）；True=末层后加 cross-attention
    retro_enabled: bool = False
    retro_heads: int = 8
    # V2 交错：0=v1 单点（final_norm 后融合均值 mem）；>0=每 N 层一个融合块，
    # 吃 token 级 chunk mem（frozen embedding 编码，可逐字抄）
    retro_every: int = 0
    retro_chunk_len: int = 64  # 每检索文档取多少 token（padding 补齐，mask 标 False）
    # 记忆层：0=关闭（无参数，checkpoint 通用）；>0=每 N 个 block 加一个 ProductKeyMemory
    memory_every: int = 0
    memory_slots: int = 4096  # 槽位数（64×64 子码本）
    memory_topk: int = 8  # 每 token 激活槽数
    # 超连接残差：0=关闭（标准单流残差）；≥2=n 路流混合（mHC-lite，见 hyperconn.py）
    hyper_streams: int = 0

    # ===== Engram 条件记忆（n-gram 哈希查表，默认关；见 engram.py / docs/details/engram.md）=====
    # 每 N 层插一个 Engram（论文最优在少数早期层）；0=关闭（无参数，旧 checkpoint 兼容）
    engram_every: int = 0
    # N-gram 阶（逗号分隔，须 ≥2）：n 表示用"过去 n-1 个 token + 当前"查表
    engram_orders: str = "2,3"
    # 每阶哈希头数（各头不同 seed → 分散碰撞）
    engram_heads: int = 2
    # 拼接后记忆向量维度 d_mem（须被 阶数×头数 整除）
    engram_d_mem: int = 64
    # 每头哈希表大小（须为素数，抗碰撞）
    engram_table_size: int = 8191
    # 短因果卷积核（0=不做卷积；论文 kernel=4）
    engram_conv: int = 4

    # ===== 投机解码（推理加速，默认关；见 spec_decode.py / docs/details/speculative-decoding.md）=====
    # 关闭时生成路径逐位不变（单测锁定）
    draft_enabled: bool = False
    # 每轮最多起草的 token 数（喂给主干并行验证的窗口）。实测 MTP 只给 1 个候选，
    # 仅当 ngram 与 MTP 首猜一致时才会凑满窗口；窗口偏大对拒答轮几乎无额外代价
    draft_window: int = 5
    # 起草来源：ngram（前缀查表，重复/模板文本命中率高）/ mtp（复用 MTP 头猜 t+2，首猜最准）/
    # hybrid（MTP 起头，仅当 ngram 首猜一致时采纳其更长续写；默认）
    draft_source: str = "hybrid"
    # ngram 起草的最短匹配后缀长度（太短会误命中，太长覆盖不足）
    draft_ngram_min: int = 2
    # DSpark 置信调度验证（DeepSeek-V4.1 / arXiv 2607.05147）：按草稿存活置信度
    # 动态定"这轮验多长"，提前截断看起来必被拒的后缀，省目标模型算力。
    # 仅 draft_enabled 时生效；不改变最终序列（验证仍精确，单测锁定等于贪心）。
    dspark_enabled: bool = False
    dspark_conf_threshold: float = 0.5  # 累计存活概率下限（越小越激进；<=0=全验证）
    dspark_min_verify: int = 1  # 每轮至少验证的草稿数（保底每轮有推进）

    # ===== 跨设备专家 offload（默认关；见 offload.py / docs/details/cross-device-offload.md）=====
    offload_enabled: bool = False
    # offload 目标设备的专家冷副本落点：cpu（内存）或 disk（磁盘，更省内存）
    offload_target: str = "cpu"
    # disk 目标的落盘目录（offload_target=disk 时必填）
    offload_dir: str = ""
    # GPU 常驻专家数上限（LRU 淘汰）；<=0 表示不限
    offload_capacity: int = 0
    # 是否根据上一步路由结果预取下一位可能命中的专家
    offload_prefetch: bool = False

    # ===== LES 专家调度（默认关；见 les.py / docs/details/les-expert-scheduling.md）=====
    les_enabled: bool = False
    # 预测器历史窗口（最近多少步路由命中参与打分）
    les_history: int = 8
    # 每步预测搬上算力设备的专家数
    les_topn: int = 4

    # ===== 视觉 token 压缩（默认关；见 vision_compress.py / docs/details/vision-token-compression.md）=====
    vision_enabled: bool = False
    # 压缩后送进主干的视觉 token 数
    vision_tokens: int = 64
    # patch 边长（图像 H/W 需为其整数倍）
    vision_patch: int = 16
    # 压缩方式：avgpool（自适应平均池化，无参数查询）/ perceiver（可学习查询 cross-attention）
    vision_compress: str = "avgpool"
    vision_d_in: int = 3  # 输入通道（RGB=3）
    # 3×3 pixel-unshuffle（DeepSeek-V4.1）：r>1 时先按 r×r 重排降采样，patch 数除以 r²
    # （r=3 → 视觉 token 省 9x）；1 表示不做（默认，逐位兼容）
    vision_unshuffle: int = 1

    @property
    def head_dim(self) -> int:
        """每个头的维度."""
        return self.d_model // self.n_heads

    @property
    def engram_order_list(self) -> list[int]:
        """解析 `engram_orders`（"2,3" → [2, 3]），忽略空项."""
        out: list[int] = []
        for part in str(self.engram_orders).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out

    def is_full_attn_layer(self, layer_idx: int) -> bool:
        """判断某层是否为 MLA 全注意力层（其余为线性注意力层）."""
        return layer_idx % self.full_attn_every == 0

    @property
    def n_full_layers(self) -> int:
        """全注意力层数量（决定 KV 缓存大小）."""
        return sum(1 for i in range(self.n_layers) if self.is_full_attn_layer(i))

    def csa_mode_list(self) -> tuple[str, ...]:
        """CSA2 模式表（按全注意力层序号循环）.

        非空 `csa_modes` 直接展开；空则由 `csa_reuse_every` 推导旧行为：
        `("full", "reuse", ...)`（长度 = csa_reuse_every；1 时全 Full）。
        """
        if self.csa_modes:
            return tuple(_norm_csa_modes(self.csa_modes).split(","))
        if self.csa_reuse_every <= 1:
            return ("full",)
        return tuple(
            "full" if k % self.csa_reuse_every == 0 else "reuse"
            for k in range(self.csa_reuse_every))

    def csa_mode_of(self, layer_idx: int) -> str:
        """某层的 CSA2 模式（非全注意力层恒为 "full"，不受影响）.

        显式 `kv_source_layer_ids` 非空时，它只决定"谁生产"：表内层是生产者
        （Full，除非 `csa_modes` 该位置显式写成 reindex），表外全注意力层是消费者
        （Reuse，除非该位置显式写成 reindex）。为空时按 `csa_modes`/`csa_reuse_every`
        的循环均匀分配。
        """
        if not self.is_full_attn_layer(layer_idx):
            return "full"
        modes = self.csa_mode_list()
        k = layer_idx // max(1, self.full_attn_every)
        base = modes[k % len(modes)]
        srcs = self.kv_source_set()
        if not srcs:
            return base
        if layer_idx in srcs:
            return "reindex" if base == "reindex" else "full"
        return "reuse" if base == "full" else base

    def kv_source_set(self) -> frozenset[int]:
        """KV 源层（生产者)集合；空串 → 空集（由 `csa_modes` 推导）.

        合法性早失败：层号必须落在 `[0, n_layers)` 且必须是全注意力层。
        """
        if not self.kv_source_layer_ids:
            return frozenset()
        ids = _parse_layer_ids(_norm_layer_ids(self.kv_source_layer_ids))
        for i in ids:
            if not 0 <= i < self.n_layers:
                raise ValueError(
                    f"kv_source_layer_ids 含越界层号 {i}（共 {self.n_layers} 层）")
            if not self.is_full_attn_layer(i):
                raise ValueError(
                    f"kv_source_layer_ids 含非全注意力层 {i}"
                    f"（full_attn_every={self.full_attn_every}，"
                    f"全注意力层序号需能被其整除）")
        return frozenset(ids)

    def compress_ratio_list(self) -> tuple[int, ...]:
        """主 KV 压缩比表（按全注意力层序号循环）；空串 = `(1,)`（不压缩）."""
        vals = _parse_layer_ids(_norm_compress_ratios(self.compress_ratios))
        return tuple(vals) if vals else (1,)

    def compress_ratio_of(self, layer_idx: int) -> int:
        """某层的主 KV 压缩比 m（非全注意力层恒为 1，不受影响）."""
        if not self.is_full_attn_layer(layer_idx):
            return 1
        vals = self.compress_ratio_list()
        k = layer_idx // max(1, self.full_attn_every)
        return int(vals[k % len(vals)])

    def ced_split_validated(self) -> int:
        """校验并返回 CED 的编码器层数（0=关）；非法配置早失败.

        早失败的理由：解码头区一旦混进线性注意力层，"只跑最后 n_win 个 token"
        就缺了前缀的递推状态（静默算错比报错贵得多），所以宁可构建期就拒绝。
        """
        split = int(self.ced_split)
        if split <= 0:
            return 0
        if split >= self.n_layers:
            raise ValueError(
                f"ced_split={split} 必须小于层数 {self.n_layers}（留至少一层解码器）")
        bad = [i for i in range(split, self.n_layers)
               if not self.is_full_attn_layer(i)]
        if bad:
            raise ValueError(
                f"CED 解码器层必须全是全注意力层，但 {bad} 是线性注意力层"
                f"（full_attn_every={self.full_attn_every}）：线性层的递推状态"
                "无法从编码器输出投影，预填充跳算会静默算错")
        return split

    def block_cache_validated(self) -> bool:
        """校验块主序缓存的前提（压缩比>1 且开了层内 SWA），返回是否启用.

        为什么必须开 SWA：块主序把"每 m 个 token"折成一行后，当前块里那不足 m 个
        token 的原始 K/V 就不在缓存里了——论文的解法是让**层内 SWA** 承担近端。
        没有 SWA 时这些 token 无处可读（会静默丢掉最近 m-1 个 token），故构建期拒绝。
        还要求 `swa_window >= max(m)`：否则块边界左侧那 `start % m` 个 token
        落在 SWA 窗口之外，查询根本看不到它们（静默丢上下文）。
        """
        if not bool(self.kv_block_cache):
            return False
        ratios = self.compress_ratio_list()
        if max(ratios) <= 1:
            raise ValueError(
                "kv_block_cache=True 需要 compress_ratios 里有 m>1（无压缩时块行=token 行，没意义）")
        if int(self.swa_window) <= 0:
            raise ValueError(
                "kv_block_cache=True 需要 swa_window>0：块主序缓存不再逐 token 保存近端，"
                "近端必须由层内 SWA 环承担（否则当前块的 token 读不到）")
        if int(self.swa_window) < max(ratios):
            raise ValueError(
                f"kv_block_cache=True 需要 swa_window >= max(compress_ratios)"
                f"（当前 {self.swa_window} < {max(ratios)}）：块边界左侧剩余的"
                f" `位置 % m` 个 token 只能从 SWA 读，窗口不够就会静默丢上下文")
        return True

    def is_ced_decoder(self, layer_idx: int) -> bool:
        """该层是否为 CED 解码器层（全局 KV 由编码器输出投影）."""
        return 0 < int(self.ced_split) <= layer_idx < self.n_layers

    def ced_decoder_layers(self) -> tuple[int, ...]:
        """CED 解码器层序号（空 = 关）."""
        split = self.ced_split_validated()
        return tuple(range(split, self.n_layers)) if split else ()


# 16G 训练配方建议（bf16 + AdamW + 梯度检查点）
RECIPE_16G: dict = {
    "precision": "bf16",
    "optimizer": "AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)",
    "seq_len": 2048,
    "micro_batch": 4,
    "grad_accum": 8,
    "effective_tokens_per_step": 2048 * 4 * 8,
    "grad_checkpointing": True,
    "notes": "权重 bf16 约 0.26GB；Adam 状态约 1GB；开梯度检查点后 "
    "seq2048/batch4 实测峰值约 5GB（RTX 3080 Laptop 16G），余量充足；"
    "不开检查点线性层 O(N^2) 矩阵会爆显存。",
}


# CPU 验证用的极小配置（单测/无卡环境跑通前向+生成+单步训练）
def tiny_test_config() -> "SmallLLMConfig":
    """返回 CPU 可跑的极小配置."""
    return SmallLLMConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        full_attn_every=2,
        n_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_dim=16,
        n_experts=8,
        top_k=2,
        expert_hidden=64,
        n_shared=1,
        mtp_depth=1,
        max_seq_len=256,
        yarn_scale=1.0,
        # CPU 极小配置保持"全 Full"基线：单测断言的是结构契约，不该随生产默认值漂移
        csa_reuse_every=1,
        # 同理：极小配置不做 SwiGLU 限幅，避免随机权重下前向数值随生产默认漂移
        swiglu_limit=0.0,
        # 同理：极小配置走旧口径路由（softmax+aux），契约不随生产默认（aux-free）漂移
        router_aux_free=False,
    )


# 长上下文预设（200K 推理）：短训 + YaRN 外推的标准路线，
# 真要 200K 质量还需阶段性长文微调（32K→128K），此处只给结构就绪
def longctx_config() -> "SmallLLMConfig":
    """返回 200K 上下文配置（推理就绪；KV 缓存约 200MB，线性层常数状态）."""
    return SmallLLMConfig(
        max_seq_len=204800,
        yarn_scale=8.0,
        sparse_threshold=4096,
        sparse_window=4096,
        sparse_sink=128,
        sparse_stride=512,
        sparse_chunk=2048,
    )


def _read_saved_config(path) -> dict | None:
    """读 checkpoint 附近保存的配置字典（`config.json` 优先，其次 `hparams.json` 的 `config`）.

    `path` 可为权重文件、权重前缀或所在目录；会从该目录**向上找最多 2 层**，以兼容
    `ckpt-XXXXX/model.pt` 这类快照布局（hparams 常放在上一级目录）。读不到返回 None。
    """
    import json
    from pathlib import Path

    p = Path(path)
    start = p if p.is_dir() else p.parent
    for d in (start, start.parent, start.parent.parent):
        for name, wrap in (("config.json", None), ("hparams.json", "config")):
            f = d / name
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if wrap:
                data = data.get(wrap) if isinstance(data, dict) else None
            if isinstance(data, dict) and data:
                return data
    return None


def load_saved_config(path) -> "SmallLLMConfig | None":
    """从 checkpoint 附近读回完整模型配置（`config.json` 优先，其次 `hparams.json` 的 `config`）.

    只取 `SmallLLMConfig` 认得的字段（多余键忽略），旧格式缺的字段用当前默认。读不到返回 None。
    """
    data = _read_saved_config(path)
    if not data:
        return None
    fields = set(SmallLLMConfig.__dataclass_fields__)
    try:
        return SmallLLMConfig(**{k: v for k, v in data.items() if k in fields})
    except (TypeError, ValueError):
        return None


# 架构开关（登记表）：改默认值只影响"新建模型"；续训/迁移/装载必须从 checkpoint 的
# hparams.json 沿用来源。找到配置但缺某字段 → 回退到"早于该特性"的值（旧 checkpoint
# 语义），避免默认值升级把旧 run 静默换架构/改前向数学。
# 分两类：①只改前向语义/共享关系、不改 state_dict 键（csa_reuse_every / swiglu_limit /
# 路由三件套）——装错不报错，必须靠来源对齐；②改结构、改键（o_groups/o_lora_rank）——
# 装错会严格载入报错，但同样登记在此以便续训自动沿用（旧 ckpt 回退关）。
# 值 = (缺失时的回退值, 归一化函数)。
def _norm_at_least_one_int(v) -> int:
    """至少 1 的整数归一化（如 CSA2 档位、o_groups）."""
    return max(int(v), 1)


def _norm_nonneg_int(v) -> int:
    """非负整数归一化：<=0 视为关闭（0）."""
    i = int(v)
    return i if i > 0 else 0


def _norm_nonneg_float(v) -> float:
    """非负浮点归一化：<=0 视为关闭（0）."""
    f = float(v)
    return f if f > 0 else 0.0


def _norm_bool(v) -> bool:
    """布尔归一化：兼容 JSON 的 true/false 与字符串 "1/true/yes/on"."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return bool(v)


_CSA_MODE_ALIASES: dict[str, str] = {
    "full": "full", "f": "full",
    "reindex": "reindex", "ri": "reindex",
    "reuse": "reuse", "re": "reuse",
}


def _norm_kv_quant(v) -> str:
    """KV 量化格式："" / "fp4" / "fp8"（非法值早失败）."""
    s = str(v or "").strip().lower()
    if s in ("", "none", "off", "0", "false"):
        return ""
    if s in ("fp4", "mxfp4"):
        return "fp4"
    if s in ("fp8", "e4m3"):
        return "fp8"
    raise ValueError(f"未知 kv_quant 格式 {v!r}（支持 ''/'fp4'/'fp8'）")


def _norm_csa_modes(v) -> str:
    """CSA2 模式串归一化："full,reindex,reuse" 式逗号列表（别名 f/ri/re）.

    非法模式直接报错（配置错误是用户可见的，早失败好过静默按 Full 跑）。
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        items = [str(x) for x in v]
    else:
        items = str(v).replace(";", ",").split(",")
    out = []
    for raw in items:
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok not in _CSA_MODE_ALIASES:
            raise ValueError(
                f"csa_modes 非法模式 {raw!r}；仅支持 full/reindex/reuse（别名 f/ri/re）")
        out.append(_CSA_MODE_ALIASES[tok])
    return ",".join(out)


def _norm_layer_ids(v) -> str:
    """层号表归一化："2,8,14" / [2, 8, 14] → "2,8,14"（去重、保序；空 → ""）.

    非整数或负数直接报错（配置错误用户可见，早失败）。
    """
    if v is None or v == "":
        return ""
    items = v if isinstance(v, (list, tuple)) else str(v).replace(";", ",").split(",")
    out: list[int] = []
    for raw in items:
        tok = str(raw).strip()
        if not tok:
            continue
        try:
            x = int(tok)
        except ValueError:
            raise ValueError(
                f"层号表含非整数 {raw!r}（形如 \"2,8,14\"）") from None
        if x < 0:
            raise ValueError(f"层号表含负层号 {x}")
        if x not in out:
            out.append(x)
    return ",".join(str(x) for x in out)


def _parse_layer_ids(s: str) -> tuple[int, ...]:
    """规范化层号串 → 元组（空串 → ()）."""
    return tuple(int(x) for x in str(s).split(",") if x.strip())


def _norm_compress_ratios(v) -> str:
    """主 KV 压缩比表归一化：\"2,1\" / [2, 1] → \"2,1\"（**保序/保重复**，循环语义）.

    每个值必须 ≥ 1（1=不压缩）；非法值早失败（配置错误用户可见）。
    """
    if v is None or v == "":
        return ""
    items = v if isinstance(v, (list, tuple)) else str(v).replace(";", ",").split(",")
    out: list[int] = []
    for raw in items:
        tok = str(raw).strip()
        if not tok:
            continue
        try:
            x = int(tok)
        except ValueError:
            raise ValueError(
                f"compress_ratios 含非整数 {raw!r}（形如 \"2\" 或 \"1,2\"）") from None
        if x < 1:
            raise ValueError(f"compress_ratios 含非法压缩比 {x}（须 ≥ 1，1=不压缩）")
        out.append(x)
    return ",".join(str(x) for x in out)


_ARCH_SWITCHES: dict[str, tuple[object, object]] = {
    "csa_reuse_every": (1, _norm_at_least_one_int),
    "swiglu_limit": (0.0, _norm_nonneg_float),
    # 路由开关（V4.1）；旧 checkpoint 缺字段 → 回退旧口径（aux loss + 无缩放 + 归一化）
    "router_aux_free": (False, _norm_bool),
    "routed_scaling_factor": (1.0, _norm_nonneg_float),
    "norm_topk_prob": (True, _norm_bool),
    # MLA 分组输出投影（V4.1）；改 state_dict 键（w_o → w_o_d/w_o_u），旧 ckpt 回退 1/0=关
    "o_groups": (1, _norm_at_least_one_int),
    "o_lora_rank": (0, _norm_nonneg_int),
    # MTP 深度（V4.1=3）；改 state_dict 键（depth>1 增 mtp.extra.*），旧 ckpt 回退 1
    "mtp_depth": (1, _norm_nonneg_int),
    # 分层稀疏 indexer（V4.1）；改 state_dict 键（新增 w_iq/w_ik），旧 ckpt 回退关
    "sparse_indexer": (False, _norm_bool),
    "indexer_heads": (4, _norm_at_least_one_int),
    "indexer_head_dim": (64, _norm_at_least_one_int),
    "indexer_topk": (64, _norm_at_least_one_int),
    "indexer_block": (8, _norm_at_least_one_int),
    "indexer_pool_blocks": (32, _norm_at_least_one_int),
    # CSA2 三模式（V4.1）；语义变更但不改键，空=旧行为（由 csa_reuse_every 推导）
    "csa_modes": ("", _norm_csa_modes),
    # 显式非均匀 KV 源层（V4.1）；（语义变更但不改键，空=均匀分配）
    "kv_source_layer_ids": ("", _norm_layer_ids),
    # 逐层主 KV 压缩比（V4.1）；改 state_dict 键（m>1 的层新增 z_cmp），旧 ckpt 回退空=全 1
    "compress_ratios": ("", _norm_compress_ratios),
    # 层内 SWA（V4.1）；改 state_dict 键（新增 swa.*），旧 ckpt 回退 0=关
    "swa_window": (0, _norm_nonneg_int),
    # CED 编码器-解码器（V4.1）；改 state_dict 键（解码器层换成 w_ced_*），旧 ckpt 回退 0=关
    "ced_split": (0, _norm_nonneg_int),
    # CED 预填充跳算（推理近似性开关，不改键）；旧 ckpt 沿用默认开
    "ced_prefill_skip": (True, _norm_bool),
    # KV 缓存量化格式（V4.1）；不改 state_dict 键（只改缓存数值/布局），旧 ckpt 回退空=不量化
    "kv_quant": ("", _norm_kv_quant),
    # 块主序 KV 缓存（#4 显存半边，依赖层内 SWA）；不改键，旧 ckpt 回退关
    "kv_block_cache": (False, _norm_bool),
}


def sync_arch_from_checkpoint(config: "SmallLLMConfig", path) -> list[str]:
    """按 checkpoint 对齐所有"改语义不改键"的架构开关，返回被对齐的字段名（无来源=空）.

    这些开关只改变权重共享/前向数学，不改变 state_dict 键名与形状——装错架构
    **不会报错**，只会静默改变语义（如把 Reuse 层的 KV 投影换成相邻 Full 层的，
    或对旧 run 强加 SwiGLU 限幅）。故加载既有权重的脚本应先调用本函数对齐来源。

    读不到任何配置返回 []，调用方用当前默认值；**找到配置但缺某字段**时，说明该
    checkpoint 早于该特性，回退到"特性前"的值（CSA2→1、SwiGLU 限幅→0、免辅助损失
    路由→关 + 缩放 1.0），保证旧 run 语义逐位不变。
    """
    data = _read_saved_config(path)
    if data is None:
        return []
    synced: list[str] = []
    for field, (fallback, norm) in _ARCH_SWITCHES.items():
        try:
            setattr(config, field, norm(data.get(field, fallback)))
        except (TypeError, ValueError):
            continue
        synced.append(field)
    return synced

