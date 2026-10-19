# FP4 主 KV + FP8 SWA KV：缓存存 4 bit，算的时候再抬回 float

> 一句话：主 KV 缓存按 **MXFP4**（E2M1 + 每 16 通道一个 E4M3 scale，4.5 bit/通道）
> 存，SWA 环按 **FP8**（E4M3，8.5 bit/通道）存；训练期用 fake-quant（STE）做 QAT，
> 推理期同一套数值，**读时反量化再算注意力**——所以不需要硬件支持 FP4 矩阵乘。

## 痛点

长上下文 agent 的 KV 缓存是纯粹的内存账：`latent + rope` 每 token 每层 320B（本仓
`kv_lora_rank=128` + `qk_rope_dim=32`），32K 就 10MB/层。CSA2 已经把"层"这一维压了，
#7 把"窗口"这一维压成定长，剩下的就是**每个条目本身的位宽**。

## 直觉

缓存里的数字不需要 16 位都留着：注意力是"加权平均"，量化噪声会被 softmax 权重
摊平。真正要留精度的是 **scale**（每个小块的幅度尺度），值本身 4 bit 就够——这就是
块量化（block-wise quantization）的标准套路。

配合 QAT：训练时就把它当"已经被压成 4 bit"来用，模型会自己学会在噪声下工作，推理时
就不掉点。

## 原理（DeepSeek-V4.1 §2.4.4）

- 格式：OCP 标准 **MXFP4** = E2M1（4 bit，值域 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}）+
  每 **16 个通道**一个 **E4M3** scale（NVFP4 的做法，但**省略**第二级全局 scale）。
  量程够用：RMSNorm 后 latent L2 范数 ≤ √512，逐通道最大幅值 ≈ 22.6（实测训练中 ~10），
  而 E2M1 × E4M3 最大可到 448×6 = 2688。
- **FP4 只用来省存储，不加速矩阵乘**：读的时候先反量化，再走普通注意力（论文原话：
  dequantizing cached values before attention）。
- 非 RoPE 与 RoPE 分量用**同一格式**；**在 RoPE 之后量化**（RoPE 前量化收益甚微，还会
  给解码加开销）——本仓的 rope 键是读时现转的，所以量化点在"未旋转原料"上（见"边界"）。
- **SWA KV 保留 FP8**：SWA 对量化更敏感，压到 FP4 掉点明显。
- 训练期 QAT（post-training 引入），推理期同一数值。

## 我们的实现

### 1. 数值层：`kv_quant.py`

```python
quantize_e2m1(x, block=16)   # → (codes uint8 0..15, scale E4M3，每 16 通道一个)
dequantize_e2m1(codes, scale)
quantize_fp8(x, block=16)    # → (e4m3 值, scale E4M3)
fake_quant(x, fmt)           # 量化→反量化 + STE（x + (q - x).detach()）
```

两个关键细节：

1. **scale 先舍入到 E4M3，再用它算码字**。否则"量化用精确 scale、反量化用存储 scale"，
   误差会超过半个码字间距（实测 1.145 vs 允许 0.875）。
2. **scale 必须向下舍入**（`_floor_to_e4m3`）。这样才能保证 `amax/scale ≥ max_code`，
   于是反量化后的块最大值恰是 `max_code × scale`，再量化得到**同一个** scale——量化
   **幂等**。若向上舍入，prefill 存的值与解码时再量化的值会漂移（实测 0.055，足以让
   两条路径分叉）。

### 2. 存储层：`QuantBuffer`

```python
buf = QuantBuffer.from_tensor(x, "fp4")   # 打包：uint8（两通道/字节）+ E4M3 scale
buf.dequant()                             # (b, cap, r) 的 float
buf.write_(pos, values)                   # 量化后按行写
buf.write_rows_(rows, values)             # SWA 环：行号 = 位置 % cap
buf.resize(cap)                           # 扩容（保持旧内容）
```

内部**一律用 uint8 存**（FP8 数据与所有 scale 都是 E4M3 的 bit pattern）：torch 的
`float8_e4m3fn` 在 CPU 上缺 `index_copy_`/赋值算子，包成 uint8 后读写都走常规路径。

### 3. 模型接线：量化点只有两处

- **`_qkv` 出口**（`c_kv` / `k_rope_raw`）：`fake_quant` —— 训练与"不走缓存的前向"都按
  会被量化成的值算；
- **缓存写入**（`_pack_cache` / `QuantBuffer.write_`）：对**同一个**张量做同一量化 →
  与上面逐位相等（不需要依赖幂等性，但幂等性保证了跨轮一致）。

读取时（`_decode` / `_kv_from_latent`）先 `dequant()` 再算注意力。

**SWA 分支也要 QAT**（`_swa_project`）：SWA 环按 FP8 存，若预填充的 SWA 列用未量化投影、
解码读环里的 FP8 值，两条路径会分叉（实测差 0.3）。所以 `_swa_project` 在 `swa_fmt`
非空时同样 fake-quant。

### 4. 开关与存储账

```python
# config.py
kv_quant: str = ""     # ""=关 / "fp4"（主 KV MXFP4）/ "fp8"（主 KV E4M3）
swa_fmt = "fp8" if kv_quant else ""     # SWA 环跟着用 FP8（论文口径）
```

- 默认关：改缓存数值（精度损失）+ 改缓存布局（`QuantBuffer`），属 opt-in；
  不改 `state_dict` 键，登记 `_ARCH_SWITCHES`（fallback ""）。
- `kv_cache_bytes` 按格式计：fp4 主 KV 4.5 bit/通道、fp8 SWA 环 8.5 bit/通道。
  实测（12 层 / `swa_window=128`）：

  | 上下文 | fp16 缓存 | fp4 主 KV + fp8 环 | 倍率 |
  |---|---|---|---|
  | 8192 | 6.99 MiB | 2.86 MiB | 0.41x |
  | 32768 | 21.99 MiB | 7.08 MiB | 0.32x |

### 5. 会话落盘

`SessionCache.save/load` 直接 pickle `QuantBuffer`（本来就是打包张量，落盘天然省）；
`load` 用 `weights_only=False`（torch ≥2.6 默认 `True` 会拒载自定义类）。

## 从想法到代码

```
痛点：长上下文 KV 缓存是内存账；CSA2 压了"层"、SWA 压了"窗口"，每个条目的位宽还没压
  ↓
直觉：注意力是加权平均，值 4 bit 够，scale 才要精度；块量化 + QAT 让模型适应噪声
  ↓
原理：MXFP4 = E2M1 + 每 16 通道 E4M3 scale（无全局 scale）；FP4 只省存储，读时反量化；
       SWA 环保留 FP8；训练期 fake-quant（STE）
  ↓
代码：
  kv_quant.quantize_e2m1/dequantize_e2m1  → 码字表 + 向下舍入的 E4M3 scale（幂等）
  kv_quant.QuantBuffer                    → uint8 打包 + write_/write_rows_/resize/dequant
  MLA._qkv 出口 fake_quant                → QAT（训练/无缓存前向）
  SparseMLAModule._swa_project             → SWA 环的 QAT（同 FP8 数值）
  _pack_cache / _decode / _kv_from_latent  → 存打包、读反量化
  kv_cache_bytes                          → 4.5/8.5 bit 口径
```

## 代价与边界

- **精度损失是真实的**：主 KV 4 bit、SWA 8 bit；随机权重下不体现，训过的模型需要
  A/B（论文：QAT 后掉点"可忽略"，但那是 552B 规模 + 完整 post-training）。
- **只省存储、不加速**：反量化发生在读路径，算力反而略增（每步把整缓冲抬回 float）。
- **量化不连续 → 跨形状比较不适用**：同长度/同分块的路径**逐位一致**（实测 0.0），
  但"2048 长度的前向"与"2050 长度的前向取前 2048 位"会因浮点形状差跨过码字边界，
  末端差 0.3 量级。所以一致性契约一律按"同长度 + 同 `sparse_chunk` 对齐"锁。
- **论文在 RoPE 之后量化，我们在原料上量化**：本仓 rope 键读时现转，量化点落在未旋转
  原料上（旋转是正交变换、保范数，误差量级相近）；代价是读时要多一次 rope，省了
  "缓存里存旋转后键"的语义分叉。
- **多 token 一步解码**（投机验证路径）：SWA 下投机已被禁用（#7），量化时该路径不作
  契约；单 token 解码实测差 1.5e-3。
- 单测：`tests/test_kv_quant.py`（E2M1 码字表 / 误差 ≤ 半个局部码字间距 / 打包无损 /
  `QuantBuffer` 往返+按行写+扩容 / 存储倍率 / STE 梯度恒等 / 关时逐位等价不加键 /
  开启后缓存是 `QuantBuffer` 且字节数 <0.35x / `prefill+解码 == 全前向`（fp4 与 fp8）/
  因果 + Finite / 会话落盘往返 / 非法格式与不整除维度早失败 / 来源对齐 + CLI 三态）。
