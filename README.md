# CrucibleLM

> An experimental testbed fusing state-of-the-art LLM architectures (MLA, hybrid linear attention, MoE) into a 128M model trainable on a single 16GB GPU.

[中文说明](./README-cn.md) | [Architecture](./docs/ARCHITECTURE.md) | [Training pitfalls](./docs/pitfalls/local-llm-training.md)

## What is this

CrucibleLM ("crucible") is an **experimental small model fusing new architectures**: DeepSeek-style ideas (low-rank MLA compression, fine-grained MoE, MTP) melted together with Qwen-Next-style ideas (hybrid full/linear attention, gated linear layers), at ~128M total params / ~48M active, to study how these techniques behave at small scale. Trainable from scratch on a single 16GB GPU, 200K-context inference ready.

This is a research testbed, not a production model. Expect rough edges; see pitfalls docs for lessons learned the hard way.

## Quickstart

```bash
pip install torch && pip install -e ".[hf]"  # datasets for HF streaming corpora
python scripts/train_local_llm.py --phase pretrain --preset tiny \
    --data local --local-path tests/fixtures/llm-corpus --max-steps 5
```

## Layout

```
CrucibleLM/
├── src/llm/local/      # model: MLA/linear/sparse attention, MoE, MTP, Muon, distillation, intuition heads
├── scripts/            # train_local_llm.py / migrate_model.py / extract_tool_choices.py
├── tests/              # unit tests (need torch, auto-skip without it)
├── docs/               # ARCHITECTURE.md (manual) / pitfalls/ (lessons learned)
└── data/               # corpora & checkpoints (gitignored, bring your own)
```

## Reproduce training

Hardware: single 16GB GPU (e.g. RTX 3080 Laptop), Python 3.12, CUDA torch.

```bash
# 0. Env (isolated venv)
uv venv ~/.venvs/llm-train --python 3.12
uv pip install --python ~/.venvs/llm-train/bin/python torch datasets modelscope pyarrow
export PATH="$HOME/.local/bin:$PATH"

# 1. Data (ModelScope CN CDN, zero network during training; ~23GB)
~/.venvs/llm-train/bin/python -c "
from modelscope import snapshot_download
snapshot_download('epfml/FineWeb2-HQ', repo_type='dataset',
    allow_patterns=['cmn_Hani/000_0000[0-9].parquet',
                    'cmn_Hani/000_0001[0-9].parquet'], local_dir='data/hq-zh')
snapshot_download('openbmb/Ultra-FineWeb', repo_type='dataset',
    allow_patterns=['data/ultrafineweb_zh/ultrafineweb-zh-part-00[1-4]-of-256.parquet'],
    local_dir='data/ultra-zh')"
# SFT data (173MB): see corpus table in docs/ARCHITECTURE.md
# Teacher (0.5B, optional, for SFT distillation):
~/.venvs/llm-train/bin/python -c "
from modelscope import snapshot_download
snapshot_download('OpenBMB/MiniCPM4-0.5B', local_dir='data/teacher-0.5b')"

# 2. Pretrain base (~65k tokens/step, ~16s/step, ~46h for 10k steps; foreground)
~/.venvs/llm-train/bin/python scripts/train_local_llm.py \
  --phase pretrain --preset base \
  --seq-len 2048 --batch 4 --accum 8 \
  --max-steps 10000 --lr 3e-4 \
  --grad-ckpt --ckpt-dir data/llm-ckpt \
  --data local --local-path "data/hq-zh:3,data/ultra-zh:1" \
  --eval-every 200 --sample-every 200 \
  --extend-vocab --rho-keep 0.5 \
  --replay-ratio 0.15 --consolidate-steps 300 --resume

# 3. Monitor (separate terminal)
tail -f data/llm-ckpt/train.log   # loss curves (JSONL)
nvidia-smi -l 2                   # VRAM (stable ~8GB expected)
# Healthy signs: eval declining, aux pinned at 0.12, single process; see docs/pitfalls/

# 4. SFT (after base converges; separate ckpt dir, base kept as rollback)
~/.venvs/llm-train/bin/python scripts/train_local_llm.py \
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

Key defaults (active when not specified): `--save-every 100` checkpoints + `--keep-last 20`
snapshots, `--shuffle-buffer 512`, `--prefetch 4`, AdamW optimizer. Resume anytime with
`--resume` (weights/optimizer/steps/data cursor all restored). If direct HF access times
out, use `--hf-endpoint https://hf-mirror.com`.

## Advanced switches (all default-off, see docs/ARCHITECTURE.md)

- Long context: `longctx_config()` for 200K inference (YaRN×8 + sparse MLA); train short,
  extrapolate, then staged long-context finetuning;
- Optimizer: `--optimizer muon` (~2x efficiency, half optimizer memory);
- Learning efficiency: `--rho-keep` (RHO selection), `--curriculum`, `--ema-*` (EMA shadow),
  `--replay-*` (replay buffer);
- Cross-tokenizer distillation: `--kd-teacher` (MiniCPM 0.5B, anchor KL);
- Architecture migration: `scripts/migrate_model.py` (deepen/widen/prune + `--init-checkpoint`);
- Inference side: intuition heads (`heads.py`, milliwatt decision heads on a frozen backbone)
  + BM25 retrieval (`retrieval.py`).

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

## Inference

```python
from src.llm.local import TinyLLM, LocalChatBackend, SmallLLMConfig

# Local weights + vocab (training outputs work directly, no conversion)
backend = LocalChatBackend.load("data/llm-ckpt/model", SmallLLMConfig())
resp = backend.chat([{"role": "user", "content": "你好"}], max_new_tokens=128)
print(resp["content"])  # resp also has reasoning/tool_calls/finish_reason (OpenAI-shaped)

# Sampling: temperature=0 greedy; >0 temperature sampling with top_k cutoff
# (no top-p/repetition penalty by default; for looping use temperature 0.7-1.0
# plus short max_new_tokens)
resp = backend.chat(messages, max_new_tokens=256, temperature=0.7, top_k=50)
```

- **Incremental decoding**: `generate()` carries KV/state caches (MLA latent, linear O(1)
  states), one prefill then single-step decoding;
- **200K context**: `longctx_config()` + same `generate()` (sparse gather decoding,
  ~0.2GB KV; one 200K prefill takes minutes, then normal per-step speed);
- **Fast-lane decisions** (no generation): `heads.py` intuition heads — one forward pass
  on a frozen backbone, milliseconds on CPU, see inference chapter in docs/ARCHITECTURE.md.

## OpenAI-compatible API (`scripts/serve_openai.py`, zero third-party deps)

```bash
python scripts/serve_openai.py --model data/llm-ckpt/model --port 8000
# --config data/llm-16L/config.json (migrated arch) --api-key xxx (optional Bearer auth)
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true}'
```

- `POST /v1/chat/completions` (true-incremental SSE streaming + non-streaming, `stop` supported),
  `POST /v1/completions` (legacy), `GET /v1/models`, `/health`;
- Point the OpenAI Python SDK at `base_url` and go (`top_k` passed through; `top_p`/`logprobs`
  not supported yet and silently ignored — `usage` is zeros pending token counting).

## License

MIT — see [LICENSE](./LICENSE).
