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
| 2 | 免辅助损失路由（sqrtsoftplus + noaux_tc + routed_scaling） | ❌ | `moe.py` |
| 3 | CSA2 三模式 Full/Reindex/Reuse | 🔶 仅 Full/Reuse | `mla.py` |
| 4 | 逐层压缩比 `compress_ratios` + KV 源层（非均匀） | 🔶 每 N 层统一 | `config.py`/`model.py` |
| 5 | 分层稀疏 indexer（学习式 top-K + 块级候选池） | 🔶 固定 sink/window/stride | `sparse_attn.py` |
| 6 | CED 编码器-解码器（上半层全局 KV 由 L/2 层投影，省近半预填充） | ❌ | `model.py`/`block.py` |
| 7 | 每层 SWA + SWA Bounded Replay | 🔶 hybrid 骨架 | `sparse_attn.py` |
| 8 | FP4 主 KV（MXFP4 + QAT）+ FP8 SWA KV | ❌ | `mla.py`/`optim.py` |
| 9 | MLA 分组输出投影 `o_lora_rank` + `o_groups` | ❌ | `mla.py` |
| 10 | Single-Pass mHC（n 流 + Sinkhorn + Mega-mHC） | 🔶 mHC-lite | `hyperconn.py` |
| 11 | Engram（n=2/3/4、8 头、去短卷积、FP8、Sinkhorn 更新） | 🔶 小表 + 短卷积 | `engram.py` |
| 12 | MTP 深度 3 | 🔶 深度 1 | `model.py` |
| 13 | DSpark 独立训练草稿模块（3 层 + Markov 头 + 置信头 + 吞吐调度） | 🔶 仅解码启发式 | `spec_decode.py` |
| 14 | head-wise Muon（Q/K 按头分块） | 🔶 整块 Muon | `optim.py` |
| 15 | Sinkhorn 更新（embedding/head 用 momentum+Sinkhorn 替 Adam） | ❌ | `optim.py` |
| 16 | 视觉：patch14 + 3×3 downsample | 🔶 downsample 对齐 | `vision_compress.py` |
| 17 | 1M 上下文（64K 稀疏起训 + YaRN 外推） | 🔶 默认 8K / 长上下文 200K | `config.py` + 训练 |
| 18 | 可控推理努力（1–100 整数档） | ❌ | 推理 / 后训练 |

**每项交付模板**：`docs/details/<技术>.md`（含"从想法到代码"）+ 单测（数学/因果/Finite）+ `docs/ARCHITECTURE.md` 开关行；踩坑进 `docs/pitfalls/`。默认开按约定 5；**前向数学变更**（1/3/4/6/7/8/9/10/11）必须重跑冒烟，并给出"旧 checkpoint 来源对齐或显式关闭"路径（沿用 `hparams.json`）。

**建议顺序**：先隔离性强、可回退的（1/2/9/12），再骨架级（6/7/3/4/5），最后记忆与量化（11/13/8/14/15/16/17/18）。

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
