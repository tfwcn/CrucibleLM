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

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](./LICENSE).
