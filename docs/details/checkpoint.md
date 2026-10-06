# 存盘与续跑：latest、快照、冠军

> 一句话：latest 精确续跑（权重+优化器+游标），版本快照只留 20 个轮转，冠军 `best/` 永存——三个东西三种命。

## 痛点

长训练最怕断：断电、被 kill、手滑 Ctrl+C。续跑要恢复三样——权重、
优化器动量、数据游标，少一样就分叉。以及：冠军只活在某一步，
轮转快照可能把它转掉。

## 三种存盘

1. **`latest`**（`model.pt` + `optim.pt` + `latest.json`，每 100 步）：
   精确续跑入口。`--resume` 读它，step/tokens/data_cursor 全续。
   optim 1GB（模型 0.5GB 的两倍），NFS 上写一次几分钟——这就是
   快照只存权重的原因。
2. **版本快照**（`ckpt-000100/`…，权重 + meta，无 optim）：
   只留最近 20 个（`--keep-last`），旧的整目录删。从快照恢复要新开动量
   （`--sft-init`/`--init-checkpoint` 走这条，动量新开是 feature 不是 bug）。
3. **冠军 `best/`**（权重 + meta，永不轮转）：每次 eval 新低自动存
   （EMA 开时存影子权重），同目录续跑继承历史最佳，更差的不覆盖。
   下轮 `--sft-init best/model.pt`，冠军永动机。

## 续跑语义

- `--resume`：有 `latest.json` 精确续，无则警告后从零开始（防呆，
  别把"没续上"当"续上了"）；
- `--init-checkpoint` + `--resume` 混用：权重取前者，只取后者的计数
  （动量新开，日志明示）；
- `data_cursor` 精确到文档 + 扣 1024 在途余量（见《数据管线》）；
- 配置存档：启动器把 YAML 拷进 `ckpt-dir/run.yaml`——复现认这个文件，
  不认你当时敲了什么。

## 从想法到代码

三种命三种写法：latest 全量（含 optim），快照轮转，冠军看 eval：

```python
save_ckpt(...)          # 每 100 步：model.pt + optim.pt + latest.json，快照只留 20
if ema_val < best_val:  # 每次 eval：新低才存
    save_best(ema_or_model)  # best/ 永不轮转
```

## 代价与坑

- `optim.pt` 和权重是分开的两个文件，拷 checkpoint 务必成对拷，
  单拷 model.pt 就只能新开动量（能跑，但是 warmup 要重走）。
- `keep-last` 只管步数快照：冠军在 `best/` 里是独立的，别 `rm -rf` 整个
  ckpt 目录——要清只清 `ckpt-*`。

## 一句话总结

latest 保命，快照回滚，best 封神——三种存盘，三种命，别混。
