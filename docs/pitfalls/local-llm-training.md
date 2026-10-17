# 本地小 LLM 训练三坑（`src/llm/local` 实战教训，RTX 3080 Laptop 16G）

## 1. 短卷积必须严格因果（loss 虚假塌到 ~0 的元凶）

`GatedDeltaLite` 的 K/V 短卷积若用 `padding=1` 双边补零，输出位置 `i` 会看到
输入 `i+1`——而自回归训练的目标恰是下一 token，等于把答案抄给模型。
模型会逐渐把全部容量转到复制通道：

- loss 加速塌到 ~0.1（复制通道越学越依赖，正反馈）
- 留存集同样沦陷（复制对任何数据有效，不是过拟合）
- 生成变复读机（推理时没有标准答案可抄）
- 随机 token 对照组短期纹丝不动（通道隐蔽，几十步内发现不了）

修法：`padding=0` + 左补 `(kernel-1)`（见 `_causal_conv`）。
回归单测：`tests/test_train_flow.py::test_linear_attn_causal_no_future_leak`
（改输入位置 `i`，断言输出 `<i` 纹丝不动）。

教训：任何"近似因果"（ asymmetric pad、单边截断验证）都不允许进训练路径；
凡是新加的序列混合操作，先写因果性单测再训。

## 2. `generate()` 必须恢复 train/eval 状态（OOM 真凶）

`generate()` 开头 `self.eval()` 后若不恢复，训练循环中穿插一次采样，
之后所有训练步都被静默关闭梯度检查点（开关条件是 `self.training`），
9 个线性层的 O(N²) 注意力矩阵同时驻留，16G 直接 OOM。
崩溃点恰是采样后的第一个训练步，极具迷惑性。

修法：`generate()` 包 `try/finally` 恢复之前状态（见 `_generate_inner` 拆分）。
回归单测：`test_generate_restores_train_mode`。

## 3. 本地爬取语料必须 block 级 shuffle（loss 台阶跳变）

parquet 按爬取顺序存（相邻文档同站点同模板），本地流若不打乱，
模型一个站点一个站点地学，换站点 loss 跳变。
`ShuffleBuffer`（默认 512 块蓄水池，`--shuffle-buffer` 可调），只包训练流，
评测流保持确定性。HF 流式虽有文档级 shuffle，打包拼接仍保留局部顺序，
同样走这一层。

## 4. 短卷积解码必须缓存左文（训推 k/v 对不上）

K/V 短卷积（kernel=3）让位置 `i` 的 k/v 依赖输入 `i-2..i`。
解码每步只喂 1 个 token，若直接卷积，左感受野全是零，与全前向算出的 k/v
不一致——生成质量系统性受损，且写法上极易漏（单测必须覆盖
prefill+单步解码 vs 全前向逐位一致）。
修法：past 里加 `(k_buf, v_buf)`（Mamba 系 `conv_state` 同理），
`_step_conv` 统一处理全序列/单步两种形状，尾部恒 `(k-1)` 长。

**尾巴要取 `buf + current` 的末段，不是只取 `current`。** 若在拼接 `buf` 之前
截尾，单步解码时 buf 每步都被"左补零"覆盖成 `[0, x_t]`，从第 3 个 token 起
左感受野丢一个真实 token：S 状态偏差可达 1.9（量级同 S 本身），逐位置输出偏差
0.1~0.4，看起来"像随机噪声"而不是 bug，很容易误判成数值精度问题。
判据：prefill 一次拿到的 conv 尾部，必须等于逐 token 解码同样步数后的尾部。

## 5. 线性注意力所有分支的 scale 必须一致

并行分支 scores 自带 `×scale`，串行/解码/分块的 `q^T·S` 形式必须显式补上，
漏掉就是全局 √d 倍的输出偏移，经 norm/router 非线性放大后训推分叉。
凡新增注意力计算路径，一律跑"与并行分支逐位一致"单测。

## 6. SSE 无长度响应必须显式关连接

`text/event-stream` 响应既无 Content-Length 又无 chunked 时，
`Connection: keep-alive` 会让"读到 EOF 为止"的客户端（urllib 的 `read()`、
部分 SDK）永远等待。教训：SSE 写完 `[DONE]` 后置
`handler.close_connection = True`；单测必须按真 SSE 客户端写法
（增量读到 `[DONE]`），而不是 `read()` 一把梭——后者测不出 framing bug，
还会把 framing 问题伪装成"偶发 hang"（时好时坏极具迷惑性）。

## 7. 解码路径手拼 block 前向时，新加的子层必须同步跟进

`model._decode_step` 为复用逻辑手拼了 `attn+moe`（省一次 block 调度），
记忆层加进 `HybridBlock`（MoE 之后）时解码侧漏跟：零初始化阶段恒等看不出，
value 训出非零后增量解码与全前向分叉约 0.69（logits 量级）。
教训：凡在 block 里加子层，同步改 `_decode_step`，并加"value 非零后
prefill+解码 vs 全前向逐位一致"单测——恒等起点的模块，bug 只在训出
非零后现形，默认的恒等单测盖不住。

## 8. 贪心解码必配重复惩罚（高频 token 循环吸引子）

SFT 模板标签（`<用户>`）和 markdown（`####`）里的 `<`、`#` 是超高频 token，
贪心解码稍一犹豫就掉进循环吸引子（`<<<<…`、`高糖×30`），且与训练步数无关
（三档权重 `<` 占比一模一样 0.125）——"300 步后变瓢"第一眼像过拟合，
8 个 prompt 一铺开才发现 200 步也一样瓢，只是 sample prompt 恰好是最会背的一题
（幸存者偏差）。修法：`_sample_next` 加 HF 语义 repetition_penalty
（已出现 token 正 logit 除、负 logit 乘），贪心配 1.2~1.4，
实测 4-gram 复读 0.58→0.04；服务侧 `repetition_penalty` 参数透传。
判据：固定 prompt 集 × 多档权重，`<` 占比 + 4-gram 复读率双降才算修好，
别只看 sample 那一题。

## 10. micro 循环必须以 accum 个数为终止主条件（数据耗尽只做辅条件）

把 `for micro_i in range(accum)` 重构成 `while pending is not None`
（为塞检索前瞻）后，单目录小数据冒烟全过——因为 1 个文档秒耗尽、
`used==0` 正常退出；一上 50 万对的混合数据直接转几千个 micro 不停
（CPU 700% 烧 15 分钟，faulthandler 抓栈显示在正常 forward 里空转，
极具迷惑性）。教训两条：一是终止条件重构必须保持原语义
（个数主条件 + 耗尽辅条件，双条件缺一不可）；二是 CPU 冒烟必须含
"耗不尽"的大数据用例，只测小数据等于没测（见 test 里的 mix 冒烟备注）。

## 9. torch.compile 在增长缓存上越编越慢（删掉的教训）

`_decode_step` 每步约 1800 个小算子，看似 compile 的天菜，实测 eager 22.5
→ compile 5.1 tok/s（更慢）。两道墙：
1. cudagraphs（reduce-overhead）要求静态内存，MLA latent/卷积尾每步 `cat`
   增长，抓图即炸；
2. dynamic=True 也救不了：dynamo 把每层 attn 拆成子帧，符号形状没传进去，
   `past[i] size mismatch` 逐层每步重编（recompiles 日志刷屏）。
结论：先静态缓存（预分配、按位写，不再 `cat` 增长，涉及 mla/sparse 两处
decode），再谈 compile；顺序反了就是负优化。这条是"本地无效不进库"活例子：
方法写完、单测全绿、真机一测变慢——删了，只留教训。
后续：静态缓存落地后 compile 复活（`model.compile_decode()`），sparse 的
`select_keys` 纯 Python 集合逻辑标 `@torch._dynamo.disable`（eager 跑、
不重编；输出形状恒定，下游不断）——30.9 tok/s。
附带实测：MoE 分组 bmm（16 次 dispatch→3 次）13→22.5 tok/s，真提速，已留；
bf16 推理同速（launch-bound 下带宽不是瓶颈）但显存减半，200K 长上下文有用，
`LocalChatBackend.load(..., dtype="bf16")`，已留。

## 11. 多 token 增量解码：补块内因果 + 与单步同算术（投机验证的隐性依赖）

投机解码的并行验证会一次喂 t 个 token，触发各注意力层此前只在 `t==1` 下用过的
"带 past 解码"路径。两处会静默出错：

1. **块内因果缺失**：MLA 解码掩码原来只屏蔽"预留空位"（`keys < pos+t`），没屏蔽
   同批未来的 token——`t>1` 时新查询会偷看同批里更靠后的草稿位置。修法：
   `_valid_mask` 在 `t>1` 时补 `keys <= q_pos` 的因果项；`t==1` 时该式退化为旧式，
   逐位不变（单测 `test_mla_multi_token_decode_is_causal`）。
2. **算术不一致**：验证路径若与逐 token 解码的浮点顺序不同，主干"验证出的贪心"
   会在近并列 logit 上与真贪心分叉，破坏"输出=贪心"的契约。修法：线性层 `t>1`
   分支按单步同款递推（初值 `s_prev`，不图并行）；稀疏解码取最后 t 个查询
   （`start=total-t`，块内由 `chunk_causal_mask` 保证因果）。
   凡新增"能一次解码多 token"的路径，一律加"与逐 token/全前向一致"回归单测。

## 12. 专家 offload 必须逐个 ensure，不能整批

`_forward_offload` 最初一次性 `ensure(命中的全部专家)` 再统一算。当
`offload_capacity < 命中专家数` 时，后搬入的专家按 LRU 把先搬入的挤回冷端
（disk 目标下参数被置 meta），随后用到这些"已置 meta"的专家 → 输出错乱
（实测偏差 0.33，量级同正常输出，极像精度问题）。修法：在逐专家循环里
**用前逐个 ensure**；单测用 `capacity=2/3`（小于命中数）覆盖。
教训：LRU 淘汰与"正在使用的对象"必须解耦——先 ensure 再用的前提是容量够，
容量不够就必须"取一个用一个"。

## 13. 投机解码的部分接受：用"按步缓存"跳位，别逐 token 重放

部分接受草稿后要把状态挪到"已提交前缀末位"处。最初实现是"回退到本轮起点 +
逐 token 重放前缀"，每次拒答要多付**一整步前向**（实测占轮时 ~20%，吞吐
1.29x→1.45x 就是省掉它）。改法：让 `_decode_multi` 顺手返回 `step_pasts[i]`
（"提交前 i+1 个 token 后"每层的完整缓存），转入下一轮时直接
`h = h_multi[:, A]`、`pasts = step_pasts[A]`，**零重放**：

- 线性层的多 token 分支本来就在按步递推，顺带把每步 `(s_i, k 尾, v 尾)` 记进
  `multi_steps`（卷积尾就是原始投影序列的定长切片，见 `linear_attn.py`）；
- MLA/稀疏复用同一缓冲（`index_copy_` 原地写，已含全部 t 个位置），只需把
  `pos` 指回 `pos+A+1`，越界处由掩码屏蔽；
- **坑**：`t==1`（本轮无草稿）不会走多 token 分支，`multi_steps` 不更新——必须
  单独把 `step_pasts` 特判为 `[new_pasts]`，否则会读到 `None`。
- 契约由 `test_decode_multi_step_pasts_match_sequential` 锁定：从 `step_pasts[A]`
  续跑必须和从头逐 token 跑到 A 的续跑逐位一致。

## 14. hybrid 起草别把 MTP 猜测并进上下文喂 ngram（后缀检索被污染）

`hybrid` 最初是"MTP 给第 1 位，再把 `上下文+MTP猜测` 喂 ngram 补后续位"。结果
ngram **几乎永不命中**：ngram 从最长后缀往短找，而最长的那个后缀现在以"没见过的
MTP 草稿"结尾，任何长度都匹配不上历史。实测 hybrid 退化成纯 MTP（每轮恒 1 个候选）。
修法：**只在 ngram 与 MTP 首猜一致时**才采纳 ngram 的更长续写，否则只用 MTP 单猜。
教训：多来源起草要各猜各的、再按"一致性"融合，别把一家的猜测当上下文去喂另一家。

## 15. 磁盘 offload 的落盘文件名必须每个 store 唯一（多层共用目录会互相覆盖）

`ExpertStore` 落盘路径最初恒为 `{offload_dir}/expert_{i}.pt`，**不含层标识**。而
`setup_offload` 为每一层各建一个 store，却共用同一个 `offload_dir`——于是后建层
把先建层的专家文件逐个覆盖，且只在"磁盘 prefill 前"读回时才暴露：`prefill` 过后
权重已被换成别层的专家，输出偏差 0.15（量级同正常输出，极像精度问题）。CPU 目标
不受影响（专家存内存、各 store 各自持有引用），所以只测 CPU 的端到端用例照不出这个坑。
修法：`ExpertStore` 加 `name` 前缀，`setup_offload` 传 `layer{li}`。
教训：**凡是"多个实例写同一目录"的持久化，文件名/命名空间必须区分实例**；
端到端回归要覆盖每一种 target，别只测最省事的那种。

## 16. 预取只填空位、绝不淘汰（否则挤掉工作集，搬运数暴涨）

LES 预取最初走的是普通 `ensure`，会和真实使用一样做 LRU 淘汰。结果：预取把
"下一步马上要用的专家"按 LRU 挤回冷端，紧接着真实使用又要搬回来——实测
（16 专家、top_k=4、`les_topn=4`、`capacity=8`）搬运数翻 3 倍（~1.2 万→~3.7 万）、
吞吐减半（~6.7→~4.7 tok/s）。修法：预取在容量满时**不淘汰工作集**——只回收
"预取但从未被真正命中"的专家（投机占位，可随时丢弃）；没有可回收的就直接跳过，
淘汰权只留给"真正要用专家"的那一步。
教训：**预取是"锦上添花"，必须保证它不会破坏当前工作集**；任何"提前做"的优化
都要论证它不会和"正在做的事"争资源。

## 17. 异步预取必须配 pinned 源 + event 等待，且容量按"驻留集"计

把预取搬到独立 CUDA stream 上有三个坑，缺一个就出错或无效：

1. **源必须 pinned**：非 pinned 的 H2D 拷贝会阻塞 host 线程，`with stream` 也
   退化成同步——异步化为零。故 `_cold` 主副本一律 `.pin_memory()`。
2. **必须 event 等待**：side stream 上的拷贝与默认流无序。正常 `ensure` 命中
   预取来的专家时，要 `torch.cuda.current_stream().wait_event(ev)` 再用模块参数，
   否则读到未写完的权重（数值随机错，不报错）。
3. **容量按 `_resident` 计，不是 `_hot`**：预取进的专家不进 `_hot`（不扰动工作集
   LRU 时序），若容量只数 `_hot`，预取会绕过上限把显存撑爆。
   另需 `_resident` 命中即跳过 `_load`，否则"预取来的 + 正常 ensure 的"同一专家
   被搬两次。

教训：**"异步"不是加个 `with stream` 就完事**——源内存、同步原语、容量口径
三件套要对齐，否则要么没异步、要么静默错误。

## 18. 异步预取在真实模型上中性偏负；且"回收预取未用"会引发 churn

实现异步预取后实测（真实 128M、RTX 3080 Laptop、`bench_offload_les.py`）：预取在
各容量下比同步 offload 略慢 0.05–0.11x；根因是 offload 的代价主要来自"逐专家
dispatch"（分组 bmm 用 3 次 bmm 替 16 次逐专家），而单专家 ~1.8MB、H2D 拷贝极快，
异步重叠省不到什么，反倒多搬了"预测但没用"的专家。

更要命的是**允许预取"回收预取未用空位继续预取"**：容量=12（宽裕）时形成
"预测→搬入→没用→回收→再预测"的 churn，搬运 173→9382、吞吐恶化到 2.28x。
修法：**预取只填空槽、绝不淘汰任何专家**——既不掉工作集，也不回收预取未用。
改后搬运 173→218，churn 消失。

教训：
- **预测类优化必须先验证"预测质量"，并认清瓶颈在哪**——瓶颈是 dispatch 而非搬运时，
  异步预取只是添乱；
- **"回收投机占位"看似无害，实则会自激**：任何"提前做 + 满了就回收再做"的循环都要
  先论证不会形成 churn，否则越勤快越亏；
- 异步/预取类开关默认关，给出量化门槛（offload 是"省显存才能跑"的刚需 + 路由可预测）
  再谈收益。


## 19. 记忆/门控层里的短卷积同样是"有状态"的（Engram 解码对不上全前向）

第 4 条讲过短卷积解码要缓存左文。Engram 条件记忆又踩了同一个坑的变体：论文把
短卷积作用在**门控后**的 `Ṽ=α⊙V` 上，而 α 依赖各位置的 hidden。逐 token 解码时
手上只有当前 hidden，卷积感受野里的历史值拿不到——于是"prefill+逐 token 解码"
与"整段一次前向"数值对不上（实测 last-position diff ~0.1，纯贪心输出分叉）。

修法不是给 Engram 也接一套跨步卷积状态（那要动 `_decode_step`/`_decode_multi`
的按步缓存切片，面很大），而是**换位置**：把短卷积放在门控**之前**，作用在仅由
token id 决定的 `V=W_V e` 上。这样感受野内全部只依赖 id，解码每步只需窗口内
`lookback = (kernel-1)*dilation + max_order - 1` 个历史 id，**无跨步状态**，
`_decode_step`/`_decode_multi` 只要把这段上下文 id 传进去即可。代价是少了一层
"对门控信号做卷积"（与论文不完全等价，已在 details 文档写明）。

教训：
- **凡是时序卷积，先问"它的输入是不是只依赖 id"**：只依赖 id → 用窗口 id 现算，
  无状态；依赖 hidden → 必须缓存卷积尾（第 4 条）。
- 取巧换位置时要把"与原文不等价"写进 details，别让后人以为复刻了论文。
