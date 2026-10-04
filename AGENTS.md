# CrucibleLM 项目指南

> 本文件供 AI 智能体阅读，按重要性递减排列。优先读取前几节即可掌握核心约束。

## 一句话定位

CrucibleLM（坩埚）——融合最新 LLM 架构的**实验性**小模型：MLA + Hybrid 线性注意力 + 稀疏 + 细粒度 MoE + MTP，约 128M 参数，单卡 16G 可从零训练，200K 上下文推理就绪。实验品，不是稳定版。

| 层 | 技术栈 |
|---|---|
| 模型 | PyTorch（bf16），自研结构无特殊算子依赖 |
| 数据 | datasets（HF 流式）/ modelscope（魔搭落盘）/ pyarrow |
| 老师 | transformers 4.5x（MiniCPM 自带 modeling 与 v5 互斥，勿升级） |
| 单测 | pytest（需 torch，无 torch 自动跳过） |

## 操作约束

- **禁止擅自启停用户的训练进程**：`pkill`/重启只在用户明确指示时执行；只读检查（日志、`nvidia-smi`、`ps`）随意。
- **默认不碰正在跑的训练**：改代码只改文件，生效靠用户下次重启；`--resume` 语义保持（权重/优化器/step/数据游标全续）。

## 核心约定（必须遵守）

1. **注释、日志等用户可见文本用中文**，代码命名保持英文。
2. **新功能必须带单测**：数学正确性（与朴素实现逐位/紧公差对比）、因果性（改未来输入，前部输出不变）、 Finite 性（无 NaN 梯度）。
3. **修改代码前先看 `docs/pitfalls/`**，发现新坑立即更新（已有：因果卷积泄漏、generate 吞 train 模式、解码 scale/conv-state 对齐等）。
4. **前向数学变更 vs 工程变更要分清**：改前向数学（注意力/掩码/初始化）必须重跑冒烟 + 写回归单测；改数据流/存盘/调度默认关闭，不影响在跑进程。
5. **新训练开关默认关闭**：`--rho-*`、`--kd-*`、`--ema-*`、`--muon` 等一律 opt-in，现有命令行为不变。
6. **loss 口径变化必须同步更新日志解读**：如 RHO 只算难 token（数值天然偏高），避免和旧曲线直接比大小。

## Git 规则（必须遵守）

1. **`data/` 永不进库**（语料/权重/老师/log，`.gitignore` 已覆盖）。`git add -A` 之前必跑 `git status --short` 目检 + `git check-ignore <大文件>` 抽查。
2. **push 前扫敏感信息**：全库 grep `api_key/password/secret/hf_*/sk-` 等模式 + 文件名扫 `.env/.pem/credentials`，零命中才推。
3. **commit 信息格式**：`feat/fix/docs/test:` 前缀 + 中文一句话，复杂改动正文列"改了什么/为什么/验证结果"。
4. **push 网络抖动重试**：`GIT_TERMINAL_PROMPT=0 git push`，超时等一会儿再推，commit 在本地是安全的。
5. **单仓纪律**：模型开发只在本仓库；历史项目中的旧模型代码已删除归档，禁止两边同时改同一逻辑。
6. **大改先单测后提交**：`python -m pytest tests/ -q` 全绿才 commit；GPU 相关改动另加真机冒烟（CPU 单测覆盖不到精度/显存问题）。
