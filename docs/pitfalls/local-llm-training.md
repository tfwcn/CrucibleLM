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
