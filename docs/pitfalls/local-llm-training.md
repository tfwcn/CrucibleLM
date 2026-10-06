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
