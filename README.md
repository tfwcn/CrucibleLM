# CrucibleLM

> An experimental testbed fusing state-of-the-art LLM architectures (MLA, hybrid linear attention, MoE) into a 128M model trainable on a single 16GB GPU.

[中文说明](./README-cn.md)（待补充） | [架构详解](./docs/ARCHITECTURE.md) | [训练血泪史](./docs/pitfalls/local-llm-training.md)

## What is this

CrucibleLM（坩埚）是一个**融合新架构的实验性小模型**：把 DeepSeek 系（MLA 低秩压缩、细粒度 MoE、MTP）与 Qwen-Next 系（Hybrid 全/线性注意力交替、门控线性层）熔于一炉，在 ~128M 总参数 / ~48M 激活下验证这些技术在小规模上的行为。单卡 16G 可从零训练，200K 上下文推理就绪。

This is a research testbed, not a production model. Expect rough edges; see pitfalls docs for lessons learned the hard way.

## Quickstart

```bash
pip install torch && pip install -e ".[hf]"  # datasets 用于 HF 流式语料
python scripts/train_local_llm.py --phase pretrain --preset tiny \
    --data local --local-path tests/fixtures/llm-corpus --max-steps 5
```

16G 单卡中文预训练（详见 `docs/ARCHITECTURE.md` 训练章节）：

```bash
python scripts/train_local_llm.py --phase pretrain --preset base \
    --seq-len 2048 --batch 4 --accum 8 --max-steps 10000 \
    --grad-ckpt --ckpt-dir data/llm-ckpt \
    --data local --local-path data/corpus \
    --eval-every 200 --sample-every 200
```

## Layout

```
CrucibleLM/
├── src/llm/local/      # 模型本体：MLA/线性/稀疏注意力、MoE、MTP、Muon、蒸馏、直觉头
├── scripts/            # train_local_llm.py（训练）/ migrate_model.py（架构迁移）/ extract_tool_choices.py
├── tests/              # 单测（需 torch，无 torch 自动跳过）
├── docs/               # ARCHITECTURE.md（架构与训练手册）/ pitfalls/（踩坑记录）
└── data/               # 语料与 checkpoint（gitignored，需自备）
```

## Reproduce training（复现训练全流程）

环境：单卡 16G（如 RTX 3080 Laptop），Python 3.12，CUDA 版 torch。

```bash
# 0. 环境（独立 venv，别污染系统）
uv venv ~/.venvs/llm-train --python 3.12
uv pip install --python ~/.venvs/llm-train/bin/python torch datasets modelscope pyarrow
export PATH="$HOME/.local/bin:$PATH"
LLMPY=~/.venvs/llm-train/bin/python

# 1. 数据（魔搭 CN CDN，训练全程零网络；约 23GB）
$LLMPY -c "
from modelscope import snapshot_download
snapshot_download('epfml/FineWeb2-HQ', repo_type='dataset',
    allow_patterns=['cmn_Hani/000_0000[0-9].parquet',
                    'cmn_Hani/000_0001[0-9].parquet'], local_dir='data/hq-zh')
snapshot_download('openbmb/Ultra-FineWeb', repo_type='dataset',
    allow_patterns=['data/ultrafineweb_zh/ultrafineweb-zh-part-00[1-4]-of-256.parquet'],
    local_dir='data/ultra-zh')"
# SFT 数据（173MB）：见 docs/ARCHITECTURE.md 语料表
# 老师（0.5B，可选，SFT 蒸馏用）：
$LLMPY -c "
from modelscope import snapshot_download
snapshot_download('OpenBMB/MiniCPM4-0.5B', local_dir='data/teacher-0.5b')"

# 2. 预训练 base（约 65k tokens/步，16 秒/步，10k 步约 46 小时）
tmux new -s llm
$LLMPY scripts/train_local_llm.py \
  --phase pretrain --preset base \
  --seq-len 2048 --batch 4 --accum 8 \
  --max-steps 10000 --lr 3e-4 \
  --grad-ckpt --ckpt-dir data/llm-ckpt \
  --data local --local-path "data/hq-zh:3,data/ultra-zh:1" \
  --eval-every 200 --sample-every 200 \
  --extend-vocab --rho-keep 0.5 \
  --replay-ratio 0.15 --consolidate-steps 300 --resume
# Ctrl-b d 脱离；tmux attach -t llm 回来看

# 3. 监控（另开终端）
tail -f data/llm-ckpt/train.log   # loss 曲线（JSONL）
nvidia-smi -l 2                   # 显存（应稳定 ~8GB）
# 健康标准：eval 稳步降、aux 恒 0.12、单进程；详见 docs/pitfalls/

# 4. SFT（base 收敛后，另起 ckpt 目录，base 原封不动留作回滚）
$LLMPY scripts/train_local_llm.py \
  --phase sft --preset base \
  --seq-len 2048 --batch 4 --accum 8 \
  --sft-init data/llm-ckpt/model.pt \
  --max-steps 2000 --lr 1e-4 --warmup 20 \
  --grad-ckpt --ckpt-dir data/llm-sft \
  --vocab data/llm-ckpt/vocab.json \
  --data local --local-path data/sft-zh \
  --extend-vocab --eval-every 100 --sample-every 100 \
  --kd-teacher data/teacher-0.5b --kd-alpha 0.5
```

关键默认值（不写即生效）：`--save-every 100` 存盘 + `--keep-last 20` 只留 20 个快照、
`--shuffle-buffer 512`、`--prefetch 4`、AdamW 优化器。续跑直接加 `--resume`
（权重/优化器/step/数据游标全续）。直连 HF 超时请用 `--hf-endpoint https://hf-mirror.com`。

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](./LICENSE).
