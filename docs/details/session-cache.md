# SessionCache：跨 turn 的长上下文存折

> 一句话：把解码状态（MLA latent + 线性状态 + 卷积尾）落盘，下 turn 读回接着算，不再全量 prefill——200K 会话内存归零。

## 痛点

多轮对话每 turn 都把全部历史重算一遍（prefill），200K 上下文 prefill 一次
分钟级，turn 一多全花在重复计算上。而生成函数的局部变量里本来就躺着
全部状态——把它显式化、落盘，就是 SessionCache。

## 直觉

银行存折：钱（状态）存银行（磁盘），办事（turn）只带存折，
不用每次把全部家当背来。`extend()` 是存钱，`save()/load()` 是异地取款。

## 原理

存的是三样东西（恰好是两类注意力的全部家当）：

1. MLA latent 缓存（静态三元组 `(buf_c, buf_kr, pos)`，见《推理加速》）；
2. 线性层状态（`(S, k_buf, v_buf)` 常数大小）；
3. 已见 token id（`ids`，续跑对账用）。

`extend()` 首段走并行 prefill（顺带按 `max_new_tokens` 预留），
后续按单 token 步进，复用 `model._decode_step`（与 generate 同数学，
单测锁 `extend == 全前向末位`）。容量不够 `ensure_room` 按 2x 扩
（eager 重分配；编译图重编一次，均摊忽略）。

## 我们的实现

- 文件：`src/llm/local/session_cache.py`
- `save()/load()` 走 `torch.save`（past 全是小张量，无 O(N²) 大物，
  几百 MB）；`load` 按模型所在设备 map；
- 只动推理路径（`no_grad`，不碰 train/eval 状态），训练循环不用它；
- 服务进程按会话 id 复用待接线（现状）；和 RETRO 正交可叠加
  （一个管"说过的话"，一个管"库里的知识"）。

## 代价与坑

- 预留是"prompt + max_new"精确制，超了 fail-fast（IndexError）——
  长会话调用方负责 `ensure_room`，别指望静默扩。
- 落盘文件含完整会话状态，server 侧按会话隔离存，别串（隐私 + 串味双风险）。

## 一句话总结

状态落盘、turn 间续跑——长会话的内存归零术。
