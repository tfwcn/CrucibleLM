# 技术详解目录

一技术一文件，尽量通俗：先讲痛点和直觉，再讲原理和我们的实现，最后是代价与坑。
数学结论看 ARCHITECTURE.md，血泪看 pitfalls/，这里看"为什么长这样"。

## 注意力与位置

- [MLA：低秩压缩的 KV 缓存](./mla.md)
- [线性注意力：O(1) 状态的金鱼记忆](./linear-attention.md)
- [Hybrid 交替：排班表](./hybrid.md)
- [稀疏注意力：200K 跳读](./sparse-attention.md)
- [RoPE 与 YaRN：位置和外推](./rope-yarn.md)

## 前馈与目标

- [细粒度 MoE](./moe.md)
- [MTP：多看一步](./mtp.md)
- [归一与激活](./norms.md)

## 优化器与学习效率

- [Muon](./muon.md)
- [RHO：只学不会的](./rho.md)
- [EMA：影子冠军](./ema.md)
- [回放与课程](./replay.md)
- [跨词表蒸馏](./kd-distillation.md)

## 外挂记忆与检索

- [Product-Key 记忆层](./memory.md)
- [RETRO：开卷考试](./retro.md)
- [SessionCache：跨 turn 存折](./session-cache.md)

## 推理与数据

- [采样与生成](./sampling.md)
- [推理加速：13→30.9 tok/s](./inference-acceleration.md)
- [数据管线](./data-pipeline.md)
- [存盘与续跑](./checkpoint.md)

## 决策与迁移

- [直觉头：毫瓦级决策器](./intuition-heads.md)
- [架构迁移：不重交学费](./migrate.md)
- [神经元复壮：恒等式重开](./dormancy-rejuvenation.md)
- [超连接残差：mHC-lite](./hyperconn.md)
