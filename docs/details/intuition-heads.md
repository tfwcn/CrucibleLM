# 直觉头：毫瓦级决策器

> 一句话：backbone 冻结，hidden 上接个小分类头做快速决策——不生成、一次前向、CPU 毫秒级。

## 痛点

有些决策不需要生成：这段上下文该调哪个工具？这条回复有没有毒？
上 128M 生成又慢又贵。直觉头是"快思考"：backbone 当特征提取器，
小头直接分类。

## 直觉

老中医把脉：不问诊（不生成），摸一下（一次前向）就有数。
脉象（hidden）是 backbone 练好的，头只需要学"脉象→结论"的映射，
数据要得少（真库 9 会话 → 99 对样本就能起步）。

## 原理与实现

- 文件：`src/llm/local/heads.py`（`IntuitionHead`：d→256→n 类，
  二分类自动 BCE）+ `train_head`（AdamW，backbone frozen，只训头）+
  `calibrate_temperature`（网格选 T，置信度可审计）+
  `evaluate_head`（acc/NLL/ECE/单条延迟一条龙）
- 特征：`encode_full_hidden` + 按真实长度 gather（padding 安全），
  `encode_last_hidden` 仅定长/单条用；
- 样本：`scripts/extract_tool_choices.py` 扫会话库
  （`data/sandbox/*/session/*/conversation.db`），产出 (上下文, 工具名)，
  精确去重；分布偏斜训头时加权；
- RAG 切片：决策取"最近 + top-K 捞回"（`retrieval.py`），不啃全量；
  200K 会话靠"RAG 切片先行、增量状态随后"（+ `SessionCache` 联动）。

## 从想法到代码

把脉即"冻结 backbone，只训头"，置信度要校准：

```python
feat = backbone.encode_full_hidden(ids)  # 冻结，不花梯度
logits = IntuitionHead(feat)              # d→256→n类，二分类自动 BCE
T = calibrate_temperature(...)            # 网格选 T，不然置信度全是 0.99
```

## 代价与坑

- 头的上限是 backbone 的 hidden 质量——backbone 不行，头再调也没用，
  别在头上浪费时间，先看 backbone 的 val。
- 温度校准（temperature scaling）是必须的：不校准的置信度全是 0.99，
  下游不敢用。ECE 进评测报告，不是可选项。
- CPU 实测：512 上下文 0.6 秒/1.1GB，无 GPU 服务器常驻 1 worker 约 1~2GB——
  这是它存在的理由（生成做不到）。

## 一句话总结

 frozen backbone，毫瓦级头，毫秒级决策——生成是深思，直觉头是条件反射。
