# 本地小 LLM（`src/llm/local`）

参考 DeepSeek V4 系 + Qwen3-Next 系思想的迷你模型：**约 128M 总参数 / 约 48M 激活**
（`count_params()` 实测），16G 显存（bf16 + AdamW + 梯度检查点，
seq2048/micro_batch4）可从零训练——RTX 3080 Laptop 16G 实测峰值 **约 5.2GB**。

## 技术对照

| 来源 | 借鉴点 | 本实现 |
|---|---|---|
| DeepSeek V3/V4 | MLA 低秩 KV 压缩 + 解耦 RoPE | `mla.py`：KV 压缩到 `kv_lora_rank=128` + 共享 rope 键，缓存省约 9x |
| DeepSeek V3/V4 | 细粒度 MoE + 共享专家 | `moe.py`：16 小专家 top-4 + 1 共享专家 + aux 负载均衡 |
| DeepSeek V3 | MTP 多 token 预测 | `model.py:MTPHead`：训练时额外预测 t+2，推理关闭 |
| DeepSeek Engram | 条件记忆：n-gram 哈希查表（与 MoE 并列的稀疏轴） | `engram.py`：多头 n-gram 哈希 + 上下文门控 + 短因果卷积，早期层残差并联（默认关） |
| DeepSeek-V4.1 | CSA2 跨层 KV 复用 | `mla.py`/`sparse_attn.py`：Full/Reuse 交替，Reuse 借最近 Full 层 latent（`csa_reuse_every`，**默认 2=开**，1=关） |
| DeepSeek-V4.1 | 3×3 pixel-unshuffle 视觉压缩 | `vision_compress.py`：`vision_unshuffle=3` 先按 3×3 重排降采样，视觉 token /9（默认 1=关） |
| DeepSeek DSpark | 置信调度验证（spec decoding） | `spec_decode.py`：按草稿存活置信度动态定验证长度，截断必被拒后缀（`dspark_enabled`，默认关） |
| Qwen3-Next | Hybrid 注意力（全/线性交替） | `block.py`：每 4 层 1 层 MLA，其余 `GatedDeltaLite` 线性层 |
| Qwen3 系 | QK-Norm、短卷积 | MLA/线性层均有 RMSNorm；线性层 K/V 加 depthwise 短卷积 |
| Qwen/DeepSeek 通用 | RoPE + YaRN 外推 | `rope.py`：`yarn_scale` 稀释高频，8k 上下文 |

## 默认配置（`config.py:SmallLLMConfig`）

- 词表 8192（小词表是 100M 可行的关键），`d_model=768`，12 层，12 头
- MLA：`q_lora_rank=192`，`kv_lora_rank=128`，`qk_rope_dim=32`
- MoE：16 专家 / top-4 / 专家 hidden 192 / 1 共享专家
- 上下文 8192（YaRN scale=2），embedding 与输出头绑定省约 6M

## 用法

```python
from src.llm.local import SmallLLMConfig, TinyLLM
from src.llm.local.train import build_optimizer, estimate_memory_gb, train_step

# 1. 显存预算（不建模，纯算术）
print(estimate_memory_gb(SmallLLMConfig(), seq_len=2048, micro_batch=4))

# 2. 训练
model = TinyLLM(SmallLLMConfig()).cuda().bfloat16()
opt = build_optimizer(model)
stats = train_step(model, opt, batch_ids)  # batch_ids 即 targets（自回归）
# SFT（prompt 已掩 -100）：train_step(model, opt, x, labels=y)

# 3. 推理 / 对接 Runner 调试
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
tok = SimpleTokenizer(8192)
tok.fit(corpus)
backend = LocalChatBackend(model.cpu(), tok)
print(backend.chat([{"role": "user", "content": "你好"}])["content"])
```

16G 配方见 `config.py:RECIPE_16G`（bf16 / seq2048 / micro_batch4 / grad_accum8 /
梯度检查点开）。生产长序列训练建议把线性层的并行前向换成 chunk-wise 递推
（当前为清晰起见用 O(N²) 并行 + 循环累积状态）。

## 性能优化（RTX 3080 Laptop 16G 实测，seq512×batch2）

| 优化 | 做法 | 效果 |
|---|---|---|
| MoE 分组计算 | 专家只算路由命中的 token（`moe.py`），附带修了 AMP 下 `where` 隐式升 fp32 | 专家计算量 16→约 5 |
| 跳过解码状态 | 训练传 `return_state=False`，省每步约 1.8 万个 Python 小算子 + MLA 两次投影 | 前向 Python 开销大减 |
| 后台预取 | `PrefetchIterator`（`--prefetch 4`），打包挪出主线程 | GPU 不等 CPU（CPU 单核 100% 时最明显） |
| block shuffle | `ShuffleBuffer`（`--shuffle-buffer 512`），平滑爬取顺序的站点聚集 | loss 台阶跳变消失 |
| 一行流 | fused AdamW + cudnn benchmark + `expandable_segments` | 稳定小加速 + 防碎片 |

合计单步 1.87s→0.8s（约 2.3x），峰值显存 2.1GB→1.2GB（同配置）。
`torch.compile` 没上：MoE 动态路由形状会导致频繁重编译，收益不稳定。

## 中文语料与训练流程（`scripts/train_local_llm.py` + `data.py`）

语料均为公开中文语料（已联网验证可用）：

| 阶段 | 语料 | 规模/特点 |
|---|---|---|
| 预训练首选 | `epfml/FineWeb2-HQ:cmn_Hani`（key `hq`） | FineWeb2 中文 top-10% 质量过滤，论文称同等效果约 6x 更少 token |
| 预训练回退 | `HuggingFaceFW/fineweb-2:cmn_Hani`（key `fineweb2`） | 全量中文（543B 词），源失败自动切换 |
| 预训练混合 | `openbmb/Ultra-FineWeb` 中文（key `ultrafineweb`） | MiniCPM5 底料，中英双 split 取中文，字段 `content`+质量分 `score` |
| SFT | `BelleGroup/train_0.5M_CN`（key `belle`） | 50 万中文指令（instruction/input/output），prompt 掩 loss 只学 output |
| SFT 本地 | 魔搭落盘 Belle 系数据 | `data/sft-zh/`：COIG（4.4 万条人类校验）+ alpaca-gpt4-zh；`--data local --local-path data/sft-zh`，训练零网络 |
| SFT 扩充 | HF 直下 + 转 Belle jsonl + 去重 | `scripts/convert_sft.py --dataset BelleGroup/train_0.5M_CN --out data/sft-belle --dedup-dir data/sft-zh`（51.9 万→42 万，去重/短丢弃）；`--local-path "data/sft-zh:3,data/sft-belle:1"` 混合；换数据=换评测尺子，跨轮 val 不可直接比 |
| SFT 扩充2 | COIG 考试+代码（魔搭 AI-ModelScope/COIG） | `--local-jsonl … --schema coig-exam` 拼卷（模板 instruction 必须连 input 联合去重，否则万题共用一句模板会被误杀）；考试 3.8 万 + 代码 1.1 万，与 sft-zh/belle 双去重；`--local-path` 再加 `data/sft-coig-exam:1,data/sft-coig-code:1` |
| 数据分工 | 准备与训练分离 | 数据准备是一次性离线步骤（每个新数据集跑一次 convert，落盘 `data/` 即完事，HF 缓存 `data/_dl/hf/` 复用）；训练只引用本地目录，零网络。`data/` 永不进库 |
| SFT 混合 | `openbmb/UltraData-SFT-Agent-2609`（key `agent-general/code/search/tool`） | 4 个 Agent 子集（多轮 messages，对话转样本对）；英文为主，作工具能力补充，中文主力仍是 Belle |
| 无网回退 | 本地 `.txt/.md/.jsonl/.parquet` 目录 | `--data local`，jsonl 取 text/content/body 字段，parquet 直读 text/content 列 |
| 本地混合 | 多目录逗号分隔，可带权重 | `--local-path "data/hq-zh:3,data/ultra-zh:1"`，评测从混合头取，各目录精确跳过 |

HF 多源混合（`--pretrain-mix "hq:1,ultrafineweb:2"`，按权重轮询；`--pretrain-min-score`
过滤低分，仅有 score 字段的源生效）。**推荐节奏**：HQ 跑完 base（当前 10k），
`--resume` 切 ultrafineweb 混合做 mid-training（MiniCPM5 同款分阶段换分布），
权重/优化器/step 照常续，数据流从头走。

全部 HF 语料走**流式**（`datasets` 库），无需下载；
训练用 packing（多文档拼接切定长块），评测留前 256 文档（同时复用拟合分词器）。

国内网络弱（parquet 分片下载超时）时改走**魔搭落盘**（同一数据集有镜像，
CN CDN 一次下几个分片，训练全程零网络；1 个分片约 2 亿字中文，10k 步约需 3~5 个）：

```bash
uv pip install --python ~/.venvs/llm-train/bin/python modelscope pyarrow
~/.venvs/llm-train/bin/python -c "
from modelscope import snapshot_download
snapshot_download('epfml/FineWeb2-HQ', repo_type='dataset',
    allow_patterns=['cmn_Hani/000_0000[0-4].parquet'], local_dir='data/hq-zh')"
# 训练时换成本地：--data local --local-path data/hq-zh（parquet 直读 text 列）
```

```bash
# 依赖（训练机上装，项目本体不强制依赖；用独立 venv，别污染项目 .venv）
uv venv ~/.venvs/llm-train --python 3.12
uv pip install --python ~/.venvs/llm-train/bin/python torch datasets
LLMPY=~/.venvs/llm-train/bin/python

# CPU 冒烟（本地 fixture，几十秒跑通训练/评测/采样/存盘/续跑）
$LLMPY scripts/train_local_llm.py --phase pretrain --preset tiny \
    --data local --local-path tests/fixtures/llm-corpus --max-steps 5

# 国内网络直连 huggingface.co 超时：用 --hf-endpoint 走镜像（代码内生效，
# 不依赖 shell 变量传递；同时 data_files 只列中文子目录，不再全仓库翻文件）
#   --hf-endpoint https://hf-mirror.com

# 16G 单卡中文预训练（FineWeb2-HQ 流式，约 65k tokens/步 @seq2048/batch4/accum8，
# --grad-ckpt 必开：线性层 O(N^2) 并行矩阵 + MoE 物化不开检查点会爆显存；
# 想试 Muon 加 --optimizer muon --muon-lr 0.02，约 2x 效率且优化器显存减半）
# tmux 跑后台：会话在=进程在，天然防重复（勿同时起两个写同一 ckpt 目录，会写坏权重）
tmux new -s llm
$LLMPY scripts/train_local_llm.py --phase pretrain --preset base \
    --seq-len 2048 --batch 4 --accum 8 --max-steps 10000 --lr 3e-4 \
    --grad-ckpt --ckpt-dir data/llm-ckpt \
    --hf-endpoint https://hf-mirror.com \
    --eval-every 200 --sample-every 200 --save-every 500
# Ctrl-b d 脱离；tmux attach -t llm 回来看；tmux ls 查会话
```

防重复（不用 tmux 时更需要注意）：开新训练前先查旧进程：

```bash
ps aux | grep train_local_llm | grep -v grep   # 有输出=还在跑，先 pkill -f train_local_llm.py
```

```bash
# SFT（接预训练权重；Belle 中文打底，agent 英文工具能力按需混合 + 0.5B 老师蒸馏）
$LLMPY scripts/train_local_llm.py --phase sft --preset base \
    --sft-init data/llm-ckpt/model.pt --seq-len 2048 --batch 4 --accum 8 \
    --grad-ckpt --max-steps 2000 --ckpt-dir data/llm-sft
    # 加混合：--sft-mix "belle:2,agent-general:1"
    # 加老师：--kd-teacher data/teacher-0.5b --kd-alpha 0.5

# 续跑（权重+优化器+step 恢复；流式数据重建流继续，换 seed 洗牌；
# 先确认旧进程已死，并 --vocab 复用分词表避免重新拟合）
$LLMPY scripts/train_local_llm.py --phase pretrain --preset base \
    --seq-len 2048 --batch 4 --accum 8 --grad-ckpt \
    --resume --vocab data/llm-ckpt/vocab.json \
    --ckpt-dir data/llm-ckpt --max-steps 20000
```

产出：`model.pt`（权重）+ `optim.pt` + `latest.json`（step/tokens）+
`vocab.json`（分词表）+ `train.log`（JSONL 训练曲线），可直接
`LocalChatBackend.load("<ckpt-dir>/model", config)` 对话试用。
`hparams.json` 存全量超参 + 模型配置 + 派生量（复核历史用），
`train.log` 每次启动首行 `run_start` 标记段落（续跑多次时按此切分）。
断点续流：`latest.json` 记 `data_cursor`（训练文本流已消费数），
`--resume` 自动快进跳过（prefetch/shuffle 在计数点下游，kill 时在途的
约千级文档会重见，已扣余量，宁可重见不丢数据）。
`--eval-every/--sample-every/--save-every` 控制评测/中文生成采样/存盘间隔
（默认每 100 步存一次 `ckpt-{step}/` 版本快照，`--keep-last` 默认只留 20 个，
约 10GB 磁盘（快照只存权重无优化器状态，精确续跑走 latest）；
`--shuffle-buffer`（默认 512）平滑站点聚集，`--prefetch`（默认 4）后台打包，
GPU 利用率低时先看 CPU 是否被 parquet 解码/分词占满。
`--extend-vocab` 扫描语料缺字追加进词表（旧 id 不动；容量内复用空行优化器照常续，
超限扩行则动量新开；`--extend-scan-docs/--extend-max-new` 控制扫描量与上限）。

## 长上下文（200K 推理就绪，32K 可训）

- 稀疏 MLA（`sparse_attn.py`，DSA 思想简化版）：sink 128 + 窗口 4096 + 跨步 512，
  超 `sparse_threshold`（默认 4096）自动切分块聚集，短序列数学上退化为 dense；
  解码从 latent 缓存展开后同样聚集。无新增参数，checkpoint 与 dense 通用。
- 线性层超 `linear_chunk`（默认 2048）切块递推：前向精确，反向块间截断
  （Transformer-XL 式）；另修了两处训推不一致：跨块项补 `×scale`、
  解码补卷积状态缓存（否则每步 k/v 与全前向对不上，Mamba 系同理）。
- `longctx_config()`：200K 预设（YaRN×8）。200K 推理 KV 仅约 0.2GB（latent 缓存），
  线性层常数状态；**训练按"短训 + YaRN 外推 + 阶段性长文微调"走**，16G 实测
  32K+ckpt 约 4GB，200K 直接训练又慢又贵，不推荐。

```python
from src.llm.local.config import longctx_config
from src.llm.local.model import TinyLLM
model = TinyLLM(longctx_config())  # 推理：generate() 支持 200K（含稀疏解码）
```

## 优化器：Muon（`--optimizer muon`）

2D 隐藏权重走 Muon（Newton-Schulz 正交化动量，`optim.py`），embedding/norm 等
走 AdamW。约 2x 计算效率（Moonshot/Kimi 路线），优化器状态减半。
`--muon-lr` 默认 0.02（与 `--lr` 分开调度，同形状余弦）。注意：
Muon 存盘与 AdamW 不互通，切换优化器删 `optim.pt` 重开动量即可（权重不受影响）；
Muon 偏好大 batch，小 batch 下不如 AdamW 稳。

## 记忆层与 RETRO（`engram.py` / `memory.py` / `retro.py`，可选，默认关）

多条外挂能力，默认全关，逐个 flag 放开，互不依赖：

- **Engram 条件记忆**（`--engram-every N --engram-orders 2,3 --engram-heads H
  --engram-d-mem D --engram-table-size P --engram-conv K`）：与 MoE 并列的第二条稀疏轴。
  每 N 层插一个 Engram，把局部 n-gram 经多头哈希查素数表取回记忆，用当前 hidden 门控后
  残差加回（注意力之前）。`W_V`/卷积零初始化 → **恒等起点**，开开关不破坏已有权重。
  只有投影/卷积/归一参与计算、表每 token 只 gather 一行（`d_mem`），故表再大也不涨激活
  参数。注意卷积**放在门控之前**、作用在仅由 id 决定的 `V=W_V e` 上，换取解码**无跨步
  状态**（否则要像线性层一样存卷积尾，`_decode_step` 得同步）；`_decode_step`/
  `_decode_multi` 已同步接入，`prefill+逐 token 解码 == 全前向` 单测锁定。
  `engram_table_size` 须素数、`engram_d_mem` 须被 阶数×头数 整除，否则启动报错。
  **实测（本机 128M 续训，600/2500 步、表 8191/65521）：中性**——与基线 val_loss 差
  全程 ±0.002，既不涨点也不掉点（单遍 Web 文本无 n-gram 重现 → 表无可学目标）；
  详见 `docs/details/engram.md` 与 pitfalls 第 21 条。

- **Product-Key 记忆层**（`--enable-memory --memory-every N --memory-slots M
  --memory-topk K`）：每 N 层插一个记忆层，把 hidden 投到 M 个 slot 上取 top-K 读出。
  `value` 零初始化 → **恒等起点**，开开关不破坏已有权重（单测锁死"初始化前后输出一致"）。
  key 用 B 方案初始化：`scripts/init_memory.py` 冻结 backbone 跑校准集
  （SFT 模板拼文本，默认 512 段），逐层收集记忆层输入 hidden 做 k-means，
  簇心当 key（`init_memory_from_activations()`，value 保持零）。
  用法：先 `init_memory.py --src <旧model.pt> --out <新目录>` 产出可严格载入的
  起始权重，再 `--sft-init <新目录/model.pt> --enable-memory` 开训；
  `--memory-slots` 须为完全平方数（√M×√M 子码本），否则启动时直接报错。
  注意两点：一是 `vocab_size` 保持预设容量（SFT 扩词复用空行不改形状，
  校准脚本绝不能按词表实际字符数缩表，否则 embedding 覆写不上还静默成功，
  曾实测覆写丢失）；二是 `model._decode_step` 手拼了 attn+moe，记忆层必须同步跟进
  （零初始化阶段恒等看不出来，value 训出非零后才分叉，教训见 pitfalls 第 7 条）；
  表参数随模型设备走（构造时先放 CPU，`model.to(device)` 会整体搬运）。
  不做 logit 蒸馏，那是另一套配方。
  双开组合起点：mem 校准权重 + retro 全零键 overlap 拼成 both-init
  （缺键必须全是 retro，`--sft-init` 严格载入且与单记忆版 logits 差 0.0 才开训）。
- **RETRO-lite 检索融合**（`--enable-retro --retro-db <pickle> --retro-k K
  --retro-heads H`）：每个 chunk 用 BM25 检索 top-K 邻居，拼成前缀 mem 段送进模型。
  `RetroFusion` 的 `w_o` 零初始化，同样恒等起点；索引由 `scripts/build_retrieval.py`
  切块构建，`BM25Retriever.save/load` 走 pickle。
  无有效记忆的行退化为零增量（全 mask 防 NaN）；评测/生成路径不传 mem，
  量的是 backbone 本体（retro 增益需专用评测，见下）。
- **RETRO V2 交错**（`--retro-every N --retro-len L`，默认 0=保持 v1 单点）：
  每 N 层交错一个同构融合块，吃 token 级 chunk mem，可逐字抄（v1 均值向量
  只能给话题方向）。chunk 由 `build_batch_chunk_ids` 产出 id + mask，
  模型侧 frozen embedding 查表编码（no_grad，不吃梯度；autocast 下自动同精度）。
  同样零初始化恒等，`_decode_step` 已同步（w_o 非零后 prefill+解码 vs 全前向
  单测锁定）。注意 chunk mem 不做因果 mask、同源未过滤，是"开卷"语义：
  自检索的 loss 虚低由专用评测度量，勿与 backbone val 比大小；
  当前推荐用法是 B（mid-training 增广 + 尾段关掉冷却），A（真 RAG 上线）
  等检索升级 + mem 消融诊断（真 mem vs 随机 mem 有 loss 差）后再做。
- **检索工程**：`BM25Retriever` 带倒排（token→文档表），`query` 只取 idf
  最高的 64 个词做候选、单遍打分 + 堆取 top-K；33 万文档下全扫描 5.4s/查
  → 64 词 2.4s → 训练用 12 词 0.76s（top-1 与全量一致，增广够用）。
  老索引无倒排时 `load` 就地重建。SFT 目录（instruction 列）建库加 `--sft`。
  训练时检索与 GPU 计算重叠：后台单线程前瞻下个 micro 的 hits（纯 CPU 部分），
  主线程只做 tensor 组装，0.64s/micro 的检索开销被掩盖；retro 关闭时零开销。
- **会话增量状态落盘**（`session_cache.py` 的 `SessionCache`）：存 MLA latent +
  线性层状态 + 短卷积尾，`save()/load()` 跨进程恢复，turn 之间不丢长上下文。
  库侧组件（推理路径专用，不进训练循环）；服务进程按会话 id 复用待接线。
  与上面两条正交，可叠加。

开关全关时旧 checkpoint 逐位兼容，`build_block` 是模型/迁移/训练共用的唯一构造入口。

## 学习效率：RHO 选择 + 课程 + EMA（`--rho-keep/--curriculum/--ema-*`）

- **RHO**（`--rho-keep 0.5`）：主 loss 只反向最高的 50% token（ignore 位恒 0 自然落选，
  保底 64 个）；MTP/aux 全量（aux 必须看全路由）。省的不是单步时间（图一样大），
  是达标步数；先用自参照版（batch 内百分位），老师版等 KD 上线复用老师前向。
- **老师参照 RHO**（`--rho-ref teacher` + `--kd-teacher`）：超额 loss（学生−老师）
  只在锚点配对位排名，非配对位恒训练；区分"真不会"与"噪声"，老师前向与 KD 共用。
- **课程**（`--curriculum "0.7:0.0:3000"`）：min-score 阈值线性放开，仅有 score
  字段的源生效（ultrafineweb），HQ 自动跳过；阈值实时读，日志记 `cur_min_score`。
- **EMA**（`--ema-every 100 --ema-weight 0.05`）：fp32 影子 + logits-MSE 一致性，
  稳定器（专治站点切换鼓包），多约 0.5GB 显存和一次前向。
  decay 按同步间隔换算（等效每步 `decay^every`，否则影子冻住、冠军过期——实测抓到过）。
  开 EMA 后评测同步测影子（`ema_val_loss`，影子保持 eval 模式），
  冠军按影子值选、存影子权重（平滑冠军，过拟合抖动期白捡精度）。
- **冠军快照**（自动）：每次 eval 出现新低就另存 `best/`（权重 + meta，
  永不轮转，供下轮 `--sft-init best/model.pt`）；同目录续跑继承历史最佳，
  更差的不覆盖冠军。`--keep-last` 只管步数快照，冠军独立。
- **SFT 掺 pretrain 回放**（`--sft-replay-dir data/hq-zh --sft-replay-ratio 0.15
  --sft-replay-pool 8192`）：batch 级混（主流耗尽即停，回放无限循环；
  两种 block 同 batcher 但批次同构，混的是 batch 不是 block）。
  回放只做正则防过拟合，不计 data_cursor（刻意重复）；评测仍是纯 SFT。
  池子只取前 N 篇打乱（防 20GB 全载入内存）。第二遍刷 SFT 必开。
- **回放**（`--replay-ratio 0.15 --replay-capacity 8192`）：每步最难 1 块入池
  （FIFO），按比例重放旧难样本（海马体式巩固）；SFT 连 label 一起存；
  日志记 `replay`（本步是否命中）与 `replay_size`。
- **巩固期**（`--consolidate-steps N --consolidate-replay 0.4`）：新分布开头 N 步
  回放加码 + lr 钳制在起始值（warmup 不升），换分布/切 Ultra 时开。
三者默认全关，500 步 A/B（同数据同种子，eval 低 ≥0.1 且 aux 不飘）赢了再开。

## 容量利用率：零休眠结论（`scripts/scan_dormancy.py`）

12 层 × 17 专家逐神经元扫描（中间激活均值）：
最闲神经元仍有**该专家均值**的 26%（sft5-best，1024 段 SFT 文本；base 预训练权重复核
为 0.29，逐层口径 0.28，结论对口径不敏感），全局休眠占比 0.0000。
结论：SwiGLU + dense 梯度下没有 ReDo 式的死神经元，复壮手术无对象，
`scripts/rejuvenate.py` 封存（自检阈值 1e-3 会正确 abort）。"效果一般"
不是容量闲置——aux 0.12（路由均衡）+ 零休眠双交叉验证模型已被榨干，
杠杆只剩数据量级与参数量级。
配套：`dormancy_masks` 新增**峰值保护**（`--peak-frac`，默认关），高阈值下可救回约 84%
被均值误杀的"少见但强烈"单元；真实阈值下空转（无对象）。跨 6 个周期 checkpoint
（每 100 步）追踪：末段 500 步激活幅度持平、职责稳定（首末相关 r≈0.9）、两端零休眠。
可解释性工具（SwiGLU 双负区间 / 激活画像）见 `docs/details/moe-interpretability.md`。

## 跨词表蒸馏（`--kd-teacher`）MiniCPM（10 万级 BPE）与本模型（8192 字符表）词表不对齐，标准 logit 蒸馏做不了。
`distill.py` 做 ULD 思想的字符级简化版：学生一字一 token ⟺ 字符坐标，
老师 BPE 自带字符区间，对齐退化成查表；双方只在表面字符串一致的锚点
（单汉字大概率独立成 token）上截断重归一做 KL（teacher||student）。
Loss：`CE + kd_alpha·KL/T² + MTP`（MTP 正交保留）。

```bash
# 老师（0.5B，Apache-2.0，魔搭/HF 都有；transformers 4.5x，自带 modeling 需 trust_remote_code）
$LLMPY scripts/train_local_llm.py --phase sft --preset base \
    --sft-init data/llm-ckpt/model.pt \
    --kd-teacher data/teacher-0.5b --kd-alpha 0.5 --kd-temp 2.0 \
    --max-steps 2000 --ckpt-dir data/llm-sft
```

注意：锚点覆盖约 50% 字符（高频字基本命中），CE 主 loss 一直保留全覆盖；
老师每 micro-step 前向一次（约 +30% 步时，0.5B bf16 约 1.3GB 常驻）。
transformers 版本锁 4.5x（老师 modeling 与 v5 互斥，`load_teacher` 内有垫片）；
老师只在 SFT/mid-training 开，base 预训练继续吃大锅饭。

## 推理/显存加速扩展（`spec_decode.py`/`offload.py`/`les.py`/`vision_compress.py`/CSA2，默认全关）

多项 opt-in 能力，默认全关；关闭时旧命令与 checkpoint 逐位兼容（单测锁定）。
参数对着 `config.py`，细节见 `docs/details/` 对应各篇：

| 开关（默认） | 作用 | 文件 |
|---|---|---|
| `draft_enabled`(False) / `draft_window`(5) / `draft_source`(hybrid) / `draft_ngram_min`(2) / `dspark_enabled`(False) / `dspark_conf_threshold`(0.5) / `dspark_min_verify`(1) | 投机解码：MTP+n-gram 起草 + 主干并行验证（DSpark 置信调度验证），输出=逐 token 贪心 | `spec_decode.py` |
| `offload_enabled`(False) / `offload_target`(cpu) / `offload_dir`("") / `offload_capacity`(0) / `offload_prefetch`(False) | 专家 offload 到 CPU/磁盘 + LRU 淘汰（capacity≤0 时全驻留→分组快路径） | `offload.py` |
| `les_enabled`(False) / `les_history`(8) / `les_topn`(4) | LES 专家命中预测，供 offload 异步预取（cpu target 走 side stream） | `les.py` |
| `vision_enabled`(False) / `vision_tokens`(64) / `vision_patch`(16) / `vision_compress`(avgpool) / `vision_d_in`(3) / `vision_unshuffle`(1) | 视觉 token 压缩（结构预留，推理前缀 + 训练 loss 已接线；`vision_unshuffle=3` 走 3×3 pixel-unshuffle，token /9） | `vision_compress.py` |
| `csa_reuse_every`(2) | CSA2 跨层 KV 复用：每 N 个全注意力层留 1 个 Full，其余 Reuse 借共享 latent；缓存降幅依全注意力层数而定（12 层/full_attn_every=4 → 3 层里 2 Full 1 Reuse，实测 8192 ctx KV −30%、32768 −32%）。**默认 2=开**（实测有代价的净收益），1=关 | `mla.py`/`sparse_attn.py`/`model.py` |

- **投机解码**：先用 `scripts/measure_mtp_acceptance.py` 卡接受率门槛（>50%）；
  实测自然语言 30–39%（不值得开）、代码 66%（值得开），**默认关、按负载决定**。
  仅 batch=1、`temperature=0` 生效，输出与逐 token 贪心逐位一致。为支撑并行
  验证，MLA（`_valid_mask` 块内因果）、稀疏（`_decode` 取最后 t 个查询）、线性
  （`past≠None, t>1` 分支）三条解码路径补齐多 token 增量，`t==1` 逐位不变。
  部分接受靠 `_decode_multi` 返回的 `step_pasts[i]` 直接跳位（零重放）。
  实测加速：普通对话 ~1.45–1.5x、代码 ~1.6–1.7x（MTP 首猜命中 ~0.6–0.8）；
  ngram 单用净亏（首猜命中 ~0.04），故 `hybrid` 仅在 ngram 与 MTP 首猜一致时采纳其更长续写。
  **DSpark 置信调度**（`dspark_enabled`，默认关）：给每个草稿 token 一个存活置信度
  （MTP=softmax 最大概率、ngram=`m/(m+1)`），只验证累计存活概率 ≥ `dspark_conf_threshold`
  的最长前缀，提前截断"必被拒"后缀以省验证算力；不改输出（仍等于贪心），
  `threshold<=0` 即退化回固定窗口。细节见 `docs/details/speculative-decoding.md`。
- **专家 offload**：开启后 MoE 从分组 bmm 切到"只算命中专家"的逐专家路径，数值一致；
  逐个 `ensure`（`capacity<命中数` 也不会误用已置 meta 的专家）；磁盘目标落盘文件按
  层命名空间隔离（`layer{li}_{i}.pt`），多层共用目录不互相覆盖；`capacity≤0`（声明
  不淘汰）时初始化即全量搬入、forward 退回分组快路径（避免"部分专家永远 meta"）。
- **LES**：纯统计、零可训练参数；预测只影响搬运时序，不改变数值。cpu target + CUDA
  时走**独立 CUDA stream 异步搬运**（pinned 主副本 + event 等待，normal ensure 命中
  时 `wait_event` 再用）；预取**只填空槽、绝不淘汰任何专家**（含"预取未用"，否则
  会 churn），容量按驻留集 `_resident` 计。实测真实 128M 上中性偏负（0.05–0.11x 慢；
  瓶颈是逐专家 dispatch 而非搬运），默认关；要变净收益需 offload 是"省显存才能跑"
  的刚需 + 路由可预测（见 `docs/details/les-expert-scheduling.md`）。
- **视觉压缩**：默认关时模型无该模块、`state_dict` 兼容；**推理前缀 + 训练 loss 均已接线**
  （`forward(targets=..., vision_embeds=...)` 只算文本段错位一位；`train_step(vision_images=...)`
  经带梯度的 `model.vision` 编码），视觉段不进 CE。
- **CSA2 跨层 KV 复用**：`csa_reuse_every=N`（默认 **2=开**；1=全 Full=关）时，每 N 个全注意力层
  留 1 个 Full，其余为 Reuse——借最近 Full 层的 latent 缓存出 K/V（只自算 Q）。Reuse 层
  与 Full 层**共享 `w_dkv`/`w_kr` 权重对象**（无死参数），仍保留各自的上投影/Q/输出；
  解码时 Reuse 层的缓存就是共享缓冲，只跳过写入。`kv_cache_bytes` 只计非 Reuse 层。
  关（=1）时逐位兼容；开启后参数量变小、缓存下降。
  架构开关随 checkpoint 走：训练脚本未显式传 `--csa-reuse-every` 时，续训/迁移会从
  checkpoint 的 `hparams.json` 沿用其架构；读不到则保守按 1，避免"默认值升级"把旧 run
  静默换架构。`preset tiny`（CPU 单测）固定全 Full，不受生产默认影响。
  **实测（本机 128M 续训 2500 步 + 续训判定 2 轮）**：8192 上下文 KV 8.75MB → 6.13MB
  （**−30%**；32768 −32%，参数 −0.1%），代价 val_loss **+0.012（相对 +0.27%）**。
  续训判定（lr 2e-5/1e-5 各一轮，同 seed 同数据）：差距随收敛深度增长（r=−0.45），
  两个独立 lr 复现同一残差——是**小而稳的固有容量差额，非适配滞后**；故是**可辩护的
  取舍，不是无损**。细节见 `docs/details/csa-kv-reuse.md`。

## 直觉头/RAG（`heads.py` + `retrieval.py`，SFT 后启用）

- **样本抽取**：`scripts/extract_tool_choices.py` 扫会话库
  （`data/sandbox/*/session/*/conversation.db`），产出 (上下文, 工具名) jsonl，
  精确去重；真库 9 会话 → 99 对 7 种工具（分布偏斜，训头时加权）。
- **头**：`IntuitionHead`（d→256→n 类，二分类自动 BCE）+ `train_head`（AdamW，
  backbone frozen）+ `calibrate_temperature`（网格选 T，置信度可审计）
  + `evaluate_head`（acc/NLL/ECE/单条延迟）。
- **RAG**：`BM25Retriever`（字符级零依赖），决策取"最近 + top-K 捞回"，
  不啃全量；200K 会话靠"RAG 切片先行、增量状态随后"。
- **特征**：`TinyLLM.encode_full_hidden` + 按真实长度 gather（padding 安全）；
  `encode_last_hidden` 仅定长/单条用。CPU 实测：512 上下文 0.6 秒/1.1GB，
  无 GPU 服务器常驻 1 worker 约 1~2GB。
- **会话缓存**：`SessionCache.extend()` 存/读 MLA latent + 线性状态 + 卷积尾，
  `save()/load()` 可跨进程恢复，短 turn 增量续跑。

## 架构迁移工具箱（`migrate.py` + `scripts/migrate_model.py`，改结构不重交学费）

精确操作（输出逐位一致，可直接 `--init-checkpoint` 续训）：
- `--add-layers N`：追加恒等层（bert2BERT 式加深，输出投影置零，pre-norm 下恒等）；
- `--expert-hidden M`：专家加宽（Net2WiderNet，复制单元 + 下投影列平分，ulp 级误差内一致）。

有损操作（需短训恢复）：
- `--keep-layers 0,1,2`：层裁剪；
- `--svd-report`：各投影低秩能量占比（跨架构映射如 GQA→MLA 选 rank 用，
  配 `svd_split` 手册配方，不做假精确的自动映射）。

```bash
python scripts/migrate_model.py --src data/llm-ckpt --out data/llm-16L --add-layers 4
~/.venvs/llm-train/bin/python scripts/train_local_llm.py ... \
    --config-json data/llm-16L/config.json \
    --init-checkpoint data/llm-16L/model.pt --ckpt-dir data/llm-16L-train ...
```

规则：日常迭代（数据/loss/长度/优化器）永远 `--resume`，不动结构；
精确迁移直接续，近似迁移先小步验证（loss 回到旧终点附近再全速）；
`build_block` 为模型与迁移共用的唯一构造入口，防参数漂移。

    ## 单测

```bash
python -m pytest tests/ -q  # 需要 torch，无 torch 自动跳过
```

新增能力各有独立单测：`test_speculative.py`（起草器 + "等于贪心"契约 + 部分接受跳位）、
`test_offload.py`（CPU/磁盘数值等价 + LRU）、`test_les.py`（预测确定性 + 数值不变）、
`test_vision_compress.py`（patch 往返 + 压缩形状 + 默认关兼容）、
`test_multi_token_decode.py`（多 token 增量解码回归）。
