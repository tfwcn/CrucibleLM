# CrucibleLM 项目指南

> 本文件供 AI 智能体阅读，按重要性递减排列。优先读取前几节即可掌握核心约束。

## 一句话定位

CrucibleLM（坩埚）——**技术优先**的 LLM 架构试验场。对标 **DeepSeek-V4.1-Flash**，逐项独立实现并验证它的整套技术（CED / CSA2 三模式 / 分层稀疏 indexer / FP4 KV / SwiGLU-limit / 免辅助损失路由 / Single-Pass mHC / Engram / MTP-3 / DSpark / head-wise Muon / Sinkhorn 更新 / 视觉压缩），把原理、实现、代价、单测沉淀成可复现的技术档案。规模（现 ~128M、单卡 16G）只是"能跑起来"的参考，**不是目标**——为跑通技术允许超预算。

| 层 | 技术栈 |
|---|---|
| 模型 | PyTorch（bf16），自研结构无特殊算子依赖 |
| 数据 | datasets（HF 流式）/ modelscope（魔搭落盘）/ pyarrow |
| 老师 | transformers 4.5x（MiniCPM 自带 modeling 与 v5 互斥，勿升级） |
| 单测 | pytest（需 torch，无 torch 自动跳过） |

## 项目目标（技术优先：V4.1 全特性对齐）

**目标**：以 `DeepSeek-V4.1-Flash` 技术报告《Pushing the Limits of KV Cache Compression》及其 HF `config.json` 为对标，把下表每一项在本仓库**独立实现 + 单测锁定 + 写清原理与代价**。规模不设上限，为跑通技术允许超预算。

**清单（✅ 已有 / 🔶 部分或简化 / ❌ 待做）**

| # | 技术点 | 现状 | 落点 |
|---|---|---|---|
| 1 | SwiGLU 限幅 `swiglu_limit` | ✅ | `moe.py` |
| 2 | 免辅助损失路由（sqrtsoftplus + noaux_tc + routed_scaling） | ✅ | `moe.py` |
| 3 | CSA2 三模式 Full/Reindex/Reuse | ✅ `csa_modes`（空=旧 Full/Reuse 交替；Reindex 无 indexer 时数学上=Reuse） | `config.py`/`mla.py`/`sparse_attn.py`/`model.py` |
| 4 | 逐层压缩比 `compress_ratios` + KV 源层（非均匀）+ 块主序缓存 | ✅ ①压缩比 `compress_ratios`（远端 m 块池化 + 学习权重 `z_cmp`、本地窗口不压缩）；②KV 源层非均匀 `kv_source_layer_ids`（任意间距）；③**块主序缓存** `kv_block_cache`（写时折行，主 KV 行数 `L→m+ceil(L/m)`，近端由层内 SWA 承担，与逐 token 布局逐位一致；量化叠加时块均值只量化一次） | `config.py`/`block.py`/`sparse_attn.py`/`model.py`/`session_cache.py` |
| 5 | 分层稀疏 indexer（学习式 top-K + 块级候选池） | ✅ 开关 `sparse_indexer`（默认关，改键）+ 蒸馏损失 `indexer_loss_coef` | `sparse_attn.py` |
| 6 | CED 编码器-解码器（上半层全局 KV 由 L/2 层投影，省近半预填充） | ✅ 开关 `ced_split`（解码器层全局 KV/rope 键由编码器输出逐层投影；预填充只跑编码器全序列 + 解码器尾段 2×窗口，8K 实测总算力 −43.75%；约束：解码器区间须全为全注意力层，base 的 1:3 hybrid 不满足；解码器恒 Full → KV 存储 +38%） | `config.py`/`mla.py`/`sparse_attn.py`/`model.py` |
| 7 | 每层 SWA + SWA Bounded Replay | ✅ 开关 `swa_window`（每层自带独立投影分支，与全局条目同一次 softmax；定长环缓存与序列长度无关；落盘只存全局 KV，读回有界重放重建；SWA 下禁用投机解码） | `config.py`/`sparse_attn.py`/`model.py`/`session_cache.py` |
| 8 | FP4 主 KV（MXFP4 + QAT）+ FP8 SWA KV | ✅ 开关 `kv_quant`（MXFP4：E2M1 + 每 16 通道 E4M3 scale，4.5 bit/通道；SWA 环联动 FP8。QAT 走 fake-quant/STE，读时反量化再算注意力；缓存打包成 `QuantBuffer`，实测 8192 ctx 0.41x、32768 ctx 0.32x） | `config.py`/`kv_quant.py`/`mla.py`/`sparse_attn.py`/`model.py` |
| 9 | MLA 分组输出投影 `o_lora_rank` + `o_groups` | ✅ | `mla.py` |
| 10 | Single-Pass mHC（n 流 + Sinkhorn + Mega-mHC） | ✅ 开关 `hyper_streams`（A/B/C 全动态：输入凸组合 + 双随机流混合（Sinkhorn 20 轮）+ 动态写回）+ `hc_single_pass`（默认开，A 错后一层 = eq.6，混合与系数预测同趟完成）+ `hc_mega`（默认开，分块单趟数据流，激活读写 (4n+4)d→(2n+2)d，与整块 GEMM 等价 3e-5）；老 ckpt 两项回退关 = eq.2 多趟；恒等起点 delta 形式逐位；最后一层 A 空转已钉死 | `config.py`/`hyperconn.py`/`block.py`/`model.py` |
| 11 | Engram（n=2/3/4、8 头、去短卷积、FP8、Sinkhorn 更新） | ✅ 开关 `engram_layers`/`engram_every`（阶 2/3/4 × 8 头、每阶 `d_mem` 维；**无短卷积** → `lookback = max_order-1`；表 + K/V 投影 FP8（QAT，`engram_fp8` 默认开，老 ckpt 回退关），`pack()` 打包成真 FP8 存储 0.53x/bf16 且查表逐位一致；表自动进 Sinkhorn 组；显式层号优先；解码 vs 全前向 1e-5） | `config.py`/`engram.py`/`block.py`/`model.py`/`optim.py` |
| 12 | MTP 深度 3 | ✅ | `model.py` |
| 13 | DSpark 独立训练草稿模块（3 层 + Markov 头 + 置信头 + 吞吐调度） | ✅ 开关 `dspark_draft`（3 层滑窗 128 + 并行 5 槽位（互不可见）+ Markov 头 `U·E(t_{i-1})` 秩 32 + 置信头（条件接受率→前缀存活 `S_i`）；`DSparkScheduler` 按 `1+ΣS_i` / `draft+base+per_token·n` 取吞吐最优（`dspark_schedule=throughput`），另有 `confidence` 阈值口径；独立训练脚本主干冻结、梯度不回传；未训练也不破坏正确性（验证精确 = 贪心）；实测 162.7M 主干 + 18.2M 草稿器，草稿一步 = 验证一步的 0.04） | `config.py`/`dspark.py`/`spec_decode.py`/`model.py`/`scripts/train_dspark.py` |
| 14 | head-wise Muon（Q/K 按头分块） | ✅ 开关 `--muon-head-wise`（w_uq/w_qr/w_uk 按头切块逐头 NS5，bmm 批量、与逐头循环逐位一致） | `optim.py` |
| 15 | Sinkhorn 更新（embedding/head 用 momentum+Sinkhorn 替 Adam） | ✅ 开关 `--sinkhorn`（Algorithm 1：Nesterov + 近零行屏蔽 + K=11 行/列交替归一化 + γ=0.18；只存一份动量，实测优化器状态 −50%） | `optim.py` |
| 16 | 视觉：patch14 + 3×3 downsample | ✅ 开关 `vision_enabled`（DeepSeek-ViT：线性 patch embed（Muon 兼容）+ **2D-RoPE**（无位置表 → 原生分辨率）+ 双向 ViT（RMSNorm+SwiGLU）+ **3×3 特征 unshuffle**（token ÷9）+ 2 层 MLP projector；1344²/patch14/r3 → 96×96 → **1024** token；`vision_grid()` 离线算 token 数；视觉塔 588.7M（论文口径）/14.3M（默认 12L·256）；**未做**多模态 aux-free 路由偏置；训练冒烟 `scripts/smoke_vision_train.py`：600 步到 100%） | `vision_compress.py` |
| 17 | 1M 上下文（64K 稀疏起训 + YaRN 外推） | ✅ **真实 YaRN NTK-by-parts**（高频不动/低频 ÷s/中间斜坡 + mscale 折进 cos/sin；`scale=1` 逐位等于标准 RoPE，旧上限 8→512）+ `longctx_1m_config()`（8K→1M = scale 128）+ 启动自检（64K 必须稀疏、违规报错；YaRN 比例/缓存/无 SWA 警告）+ `--ctx-ramp` 阶段表；实测 1M 处位置仍可分辨、RoPE 缓存 512MB；解码 FLOPs 近常数需开 `kv_block_cache`/SWA（pitfalls 60）；训练冒烟：162.7M 续训 @6144（跨 4096 稀疏阈值）5 步 loss 有限 | `rope.py`/`longctx.py` |
| 18 | 可控推理努力（1–100 整数档） | ✅ 开关 `effort`（系统提示前置 `Reasoning Effort: {b} (range 1–100; ...)`，**幂等**注入；`effort=None` 与原 prompt 逐位一致）；训练侧 **同 `(prompt,b)` 子组**均值中心化 + 长度惩罚 `−min(C_max, k(b)·ℓ/L_norm)`、`k(b)=k0·exp(−(b−b_min)/τ)`（指数惩罚 ⇔ 长度对 b 仿射）；部署三档 `low/high/max=50/75/100`；`scripts/effort_probe.py` 实测曲线（未训练 checkpoint 实测**不单调**，符合预期）；训练冒烟 `scripts/smoke_effort_train.py`：SFT 后长度 2/9/20/37 严格单调（只证条件化） | `effort.py` |

**每项交付模板**：`docs/details/<技术>.md`（含"从想法到代码"）+ 单测（数学/因果/Finite）+ `docs/ARCHITECTURE.md` 开关行；踩坑进 `docs/pitfalls/`。默认开按约定 5；**前向数学变更**（1/3/4/5/6/7/8/9/10/11）必须重跑冒烟，并给出"旧 checkpoint 来源对齐或显式关闭"路径（沿用 `hparams.json`）。

**依赖修正（原顺序不合理）**：3/4/5 是同一个 **CSA2 家族**，咬合在一起——
**5（indexer）→ 3（三模式）→ 4（逐层压缩比 + KV 源层）**，必须当一个块做。
原因：3 的 **Reindex** 语义是"复用前层 main KV + indexer K，但用自己的 indexer Q 重新选 Top-K"——
没有 5 的 indexer（选择与 query 内容无关、逐层相同）时，Reindex 会**退化成与 Reuse 完全一样**，
无法单独验证；4 的"非均匀 KV 源层 / 逐层压缩比"又直接建立在这套 Full/Reindex/Reuse 分配之上。
所以原"骨架级（6/7/3/4/5）"里把 6/7 排在 3/4/5 之前是错的。

**建议顺序**：先隔离性强、可回退的（1/2/9/12，已完成），再接 **CSA2 家族（5 → 3 → 4）**，
然后骨架级（6/7），最后记忆与量化（11/13/8/14/15/16/17/18）。

## 操作约束

- **全程用中文思考和回答**：推理过程与最终答复一律使用中文，代码命名保持英文。
- **禁止擅自启停用户的训练进程**：`pkill`/重启只在用户明确指示时执行；只读检查（日志、`nvidia-smi`、`ps`）随意。
- **默认不碰正在跑的训练**：改代码只改文件，生效靠用户下次重启；`--resume` 语义保持（权重/优化器/step/数据游标全续）。

## 核心约定（必须遵守）

1. **注释、日志等用户可见文本用中文**，代码命名保持英文。
2. **新功能必须带单测**：数学正确性（与朴素实现逐位/紧公差对比）、因果性（改未来输入，前部输出不变）、 Finite 性（无 NaN 梯度）。
3. **修改代码前先看 `docs/pitfalls/`**，发现新坑立即更新（已有：因果卷积泄漏、generate 吞 train 模式、解码 scale/conv-state 对齐等）。
4. **前向数学变更 vs 工程变更要分清**：改前向数学（注意力/掩码/初始化）必须重跑冒烟 + 写回归单测；改数据流/存盘/调度默认关闭，不影响在跑进程。
5. **开关默认按"有用且无副作用就默认开"**：新开关只要判断有净收益、且不产生副作用，就默认开启
   （默认关才是例外，须说明理由）。**"无副作用"三分**：不改既有 run 的权重语义（续训/迁移/装载
   必须从 checkpoint 的 `hparams.json` 沿用架构，读不到保守按旧默认，避免默认值升级把旧 run
   静默换架构）、有显式关闭方式、不显著拖慢或破坏可复现性。反之，**有代价或依赖外部条件**的
   开关仍默认关（opt-in）——需老师模型的 `--kd-*`、增算力/显存的 `--rho-*`/`--ema-*`/`--muon`、
   改训练分布的 `--replay-*` 等，这些要 500 步 A/B 赢了再开。默认开的开关同样要满足：
   (a) 在 `config.py` 注释与 `docs/ARCHITECTURE.md` 写清收益与代价，(b) 提供显式关闭方式，
   (c) 续训/迁移从 checkpoint 的 `hparams.json` 沿用来源架构。
6. **loss 口径变化必须同步更新日志解读**：如 RHO 只算难 token（数值天然偏高），避免和旧曲线直接比大小。
7. **文档三处分家**：新技术同步起一篇 `docs/details/<技术>.md`（痛点→直觉→原理→实现→坑，一技术一文件，参数对着代码写）；
   `docs/ARCHITECTURE.md` 只留参数表与开关；教训进 `docs/pitfalls/`。"为什么"和"是什么"分家，文档才不会越长越没法看。
   details 每篇必填"从想法到代码"一节（概念→关键代码映射，5~10 行伪代码）。
   **文档里只放通用路径**（`data/sft-zh`、`data/llm-ckpt` 这类流程产物），
   本机历史产物路径（`ckpt-*/best/`、试错权重）永不进文档——和"本地有效才进库"同一条规矩。

## Git 规则（必须遵守）

1. **`data/` 永不进库**（语料/权重/老师/log，`.gitignore` 已覆盖）。`git add -A` 之前必跑 `git status --short` 目检 + `git check-ignore <大文件>` 抽查。
2. **push 前扫敏感信息**：全库 grep `api_key/password/secret/hf_*/sk-` 等模式 + 文件名扫 `.env/.pem/credentials`，零命中才推。
3. **commit 信息格式**：`feat/fix/docs/test:` 前缀 + 中文一句话，复杂改动正文列"改了什么/为什么/验证结果"。
4. **push 网络抖动重试**：`GIT_TERMINAL_PROMPT=0 git push`，超时等一会儿再推，commit 在本地是安全的。
   补充：**不要 push，一律只 commit 到本地**，push 由用户手动执行。
5. **单仓纪律**：模型开发只在本仓库；历史项目中的旧模型代码已删除归档，禁止两边同时改同一逻辑。
6. **大改先单测后提交**：`python -m pytest tests/ -q` 全绿才 commit；GPU 相关改动另加真机冒烟（CPU 单测覆盖不到精度/显存问题）。
7. **本地试过有效才进库**：引用本机历史产物（`ckpt-*/best/`、试错权重）的配置一律放
   `configs-local/`（已 gitignore），不 commit 也不写进 README；`configs/` 与 README
   只收录"别人从零能跑通"的（通用路径 + 已验证结论）。防 commit 堆积无用试错记录。
