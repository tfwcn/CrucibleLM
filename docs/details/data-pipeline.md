# 数据管线：语料进来，batch 出去

> 一句话：HF 流式 / 魔搭落盘 / 本地 SFT 三路进，打包→shuffle→预取→batch，断点续流精确到文档。

## 痛点

训练最怕两种死法：数据断了（流耗尽静默停）和数据乱了（resume 位置对不上，
同一批数据刷两遍还不知道）。管线的要求：快（GPU 不等 CPU）、准
（cursor 精确）、杂（多源按权重混）。

## 数据源（三路）

| 路 | 预训练 | SFT |
|---|---|---|
| HF 流式 | hq / fineweb2 / ultrafineweb（`--pretrain-mix` 配比 + `--pretrain-min-score` 过滤） | belle / agent-*（`--sft-mix`） |
| 魔搭落盘 | `data/hq-zh`（20GB）/ `data/ultra-zh`（4.8GB）parquet | `data/sft-zh`（COIG+alpaca） |
| 本地扩充 | — | `data/sft-belle`（42 万，`convert_sft.py` 转格式+去重）、`data/sft-coig-*`（考试+代码） |

`convert_sft.py`：HF/本地 jsonl → Belle 三元组，instruction+input 联合去重
（考试题 instruction 是模板，单键去重会误杀 3.7 万真题——实测教训），
空/短 output 丢弃。`build_retrieval.py`：同一批语料切块建 BM25 索引
（`--sft` 拼三元组）。

## 打包与混合

- `pack_pretrain`：文档拼接 + EOS，切 `seq_len+1` 块（+1 留 label 错位）；
- `pack_pairs`：prompt 掩 IGNORE，只在 output 上算 loss，多样本拼满一块，
  超长单样本截断（保证块内完整样本）；
- `ShuffleBuffer(512)`：block 级 shuffle，防爬取顺序的站点聚集
  （loss 台阶跳变就是这么没的）；
- `PrefetchIterator(4)`：打包挪后台线程，GPU 不等 CPU；
- **多目录混合**：`--local-path "a:3,b:1"` 按权重轮询；SFT 掺 pretrain 回放
  是 batch 级混（`interleave_batches`，主流耗尽即停，回放不计 cursor）——
  混 block 会脏 batch（PackedBatcher 按首元素定形状），这是血泪分界线；
- **断点续流**：`data_cursor` 精确到文档（多目录各记各的消费数），
  存盘扣 `RESUME_SLACK_DOCS=1024` 在途余量（宁可少量重见、不丢数据）。

## 扩词

`--extend-vocab`：新字符只许末尾追加（旧 id 永不移位，已训权重逐行兼容）；
容量内复用空行重初始化，超限才扩行（tied 重绑）+ 动量新开。
校准脚本注意：`vocab_size` 保持预设 8192，缩表会导致 embedding 静默覆写不上
（实测抓到过）。

## 从想法到代码

打包即"拼满切块"，混合即"按权轮询"，续流即"记数快进"：

```python
ids += prompt_ids + output_ids          # SFT：prompt 掩 IGNORE，只学 output
blocks = interleave([a]*3 + [b]*1)     # 多目录按权重轮询
_skip(stream, data_cursor - 1024)      # 断点续流，扣在途余量宁重勿丢
```

## 代价与坑

- 换数据 = 换评测尺子：holdout 取自流的前 256 个（多目录交错取），
  加新目录后 val 和历史比不了——要比就固化评测集。
- `convert_sft.py` 的 `--dedup-dir` 支持逗号分隔多目录，新数据对**所有**
  旧数据去重，别只对一个。

## 一句话总结

三路进货、打包掩码、按权混合、精确续流——管线是训练的生命线，慢不得、错不得。
