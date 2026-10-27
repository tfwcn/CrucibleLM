"""记忆分层硬件基准 — H2D 小块延迟 / 宿主表 gather / 磁盘随机读 IOPS / mmap 页缓存放大.

用途：回答"记忆权重能不能放 CPU/内存/磁盘"这个前置问题。模型侧一行不改，纯只读测量，
拿到的三个数决定后续方案怎么走：

1. **H2D 小块延迟**：Engram 每 token 只搬 24 行（3 阶 × 8 头）≈ 768 B，
   大块带宽好看没用，小块延迟 + kernel launch 才是关键路径；
2. **宿主表 gather 开销**：表留在 CPU 时，随机 `index_select` 是**纯串行 CPU 工作**，
   prefill 4096 token 要 gather ~10 万行，这一项可能比 H2D 还贵；
3. **磁盘随机读**：`/dev/sdd` 是虚拟盘，4 KB QD=1 的 p50 决定"表落盘"到底存不存在。

输出：`--out` 目录下同名 `.json` + `.md`（默认 `data/bench/`，已 gitignore）。
临时大文件写 `--scratch-dir`（默认本地盘 `/root/.cache/crucible-bench`），
**不要**用 `data/`（NFS，随机读走网络）或 `/tmp`（tmpfs，等于测内存）。

用法：
    python scripts/bench_mem_tier.py --quick          # 秒级冒烟
    python scripts/bench_mem_tier.py                  # 完整口径（2 GB 临时文件）
    python scripts/bench_mem_tier.py --file-gb 4 --qd 16
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

# 4 KB 对齐（O_DIRECT 硬要求；同时是绝大多数盘的物理页大小）
_ALIGN = 4096
_PAGE = 4096


def _ms(sec: float) -> float:
    """秒 → 毫秒（报告统一用 ms/µs）."""
    return sec * 1000.0


def _pct(vals: list[float], q: float) -> float:
    """分位数（q∈[0,1]，样本量小用线性插值即可，不必上 numpy）."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _stats(vals_ms: list[float]) -> dict:
    """一组毫秒样本 → p50/p95/mean/max."""
    return {
        "n": len(vals_ms),
        "p50_ms": round(_pct(vals_ms, 0.50), 4),
        "p95_ms": round(_pct(vals_ms, 0.95), 4),
        "mean_ms": round(statistics.fmean(vals_ms), 4) if vals_ms else float("nan"),
        "max_ms": round(max(vals_ms), 4) if vals_ms else float("nan"),
    }


# ===== 1. H2D：小块延迟（pinned vs pageable）=====


def bench_h2d(sizes_kb: list[int], iters: int, pinned: bool) -> list[dict]:
    """测 CPU→GPU 拷贝：逐次 CUDA event 计时（设备侧）+ 整轮墙钟（端到端，含下发）.

    `pinned=True` 走锁页内存（可 `non_blocking` 真异步）；`False` 走普通内存
    （H2D 会阻塞 host，是很多"异步不生效"的根源）。
    """
    dev = "cuda"
    out: list[dict] = []
    for kb in sizes_kb:
        nbytes = kb * 1024
        n = max(1, nbytes // 2)  # fp16 元素数
        src = torch.empty(n, dtype=torch.float16, pin_memory=pinned)
        dst = torch.empty(n, dtype=torch.float16, device=dev)
        # 预热：首次触碰会有页错误/上下文建立，不能计入
        for _ in range(5):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

        # (a) 设备侧：每次拷贝两端各记一个 event，量真实占用时长
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for _ in range(iters)]
        for s, e in evs:
            s.record()
            dst.copy_(src, non_blocking=True)
            e.record()
        torch.cuda.synchronize()
        dev_ms = [s.elapsed_time(e) for s, e in evs]

        # (b) 端到端：整轮墙钟 / 次数，含 Python 侧下发与同步开销
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        e2e_ms = _ms(time.perf_counter() - t0) / iters

        # (c) 批量摊销：一次 event 包住 K 次拷贝，扣掉 event 自身开销
        #     （单次 event 计时的分辨率对本机 4 KB 拷贝来说不够，会虚高）
        batch = max(2, min(50, iters))
        torch.cuda.synchronize()
        s0 = torch.cuda.Event(enable_timing=True)
        s1 = torch.cuda.Event(enable_timing=True)
        s0.record()
        for _ in range(batch):
            dst.copy_(src, non_blocking=True)
        s1.record()
        torch.cuda.synchronize()
        batch_us = s0.elapsed_time(s1) * 1000.0 / batch

        out.append({
            "size_kb": kb,
            "pinned": pinned,
            "device": _stats(dev_ms),
            "batch_us": round(batch_us, 2),
            "e2e_avg_ms": round(e2e_ms, 4),
            # 按 3 KB/token（3 阶 × 8 头 × 32 维 FP8）折算的每秒 token 上限
            "e2e_GBps": round(nbytes / (e2e_ms / 1000.0) / 1e9, 3) if e2e_ms > 0 else float("nan"),
        })
    return out


# ===== 2. 宿主表 gather：表在 CPU 时每步要付的串行代价 =====


def bench_host_table(slots: int, cols: int, n_rows: int, iters: int) -> dict:
    """模拟"表留 CPU、只把命中行搬上 GPU"的一次前向.

    三步分别计时：CPU 随机 gather → 汇聚到 pinned 缓冲 → H2D。
    `n_rows` 取 `token 数 × 阶数 × 头数`（预填 4096 token / 3 阶 / 8 头 = 98304 行）。
    """
    table = torch.randint(0, 256, (slots, cols), dtype=torch.uint8)  # FP8 主副本形态
    idx = torch.randint(0, slots, (n_rows,), dtype=torch.long)
    pinned = torch.empty(n_rows, cols, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(n_rows, cols, dtype=torch.uint8, device="cuda")
    nbytes = n_rows * cols

    def _one() -> tuple[float, float, float]:
        # (a) CPU 随机 gather：每个索引基本都是一次 cache miss，纯串行
        t0 = time.perf_counter()
        rows = table.index_select(0, idx)
        t1 = time.perf_counter()
        # (b) 汇聚到锁页缓冲（H2D 要走 pinned 才能异步）
        pinned.copy_(rows)
        t2 = time.perf_counter()
        # (c) 上卡
        gpu.copy_(pinned, non_blocking=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return _ms(t1 - t0), _ms(t2 - t1), _ms(t3 - t2)

    for _ in range(3):  # 预热
        _one()
    g, p, h = zip(*(_one() for _ in range(iters)))  # type: ignore[misc]
    return {
        "slots": slots,
        "cols": cols,
        "n_rows": n_rows,
        "bytes": nbytes,
        "gather_ms": _stats(list(g)),
        "pin_copy_ms": _stats(list(p)),
        "h2d_ms": _stats(list(h)),
        "total_p50_ms": round(_pct(list(g), 0.5) + _pct(list(p), 0.5) + _pct(list(h), 0.5), 4),
    }


# ===== 3. 磁盘随机读 IOPS（O_DIRECT，绕页缓存）=====


def _make_file(path: Path, size_gb: float) -> None:
    """造一个**真占块**的随机内容文件（稀疏文件的洞读起来不走盘，会假快）."""
    total = int(size_gb * (1 << 30))
    blk = os.urandom(1 << 20)  # 1 MiB 随机块，重复写：内容不重复也无所谓
    with open(path, "wb") as f:
        left = total
        while left > 0:
            n = min(left, len(blk))
            f.write(blk[:n])
            left -= n
        f.flush()
        os.fsync(f.fileno())


def _pread_one(fd: int, buf: memoryview, off: int) -> None:
    """单次对齐读（O_DIRECT 要求 offset/长度/缓冲区都 4 KB 对齐）."""
    got = os.preadv(fd, [buf], off)
    if got != len(buf):
        raise OSError(f"短读：{got} != {len(buf)}")


def bench_disk(path: Path, size_gb: float, block_kb: int, iters: int, qd: int) -> dict:
    """随机读：先测 QD=1（单次延迟），再测 QD=N（并发吞吐）."""
    nblk = int(size_gb * (1 << 30)) // _ALIGN
    blen = block_kb * 1024
    flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
    if not getattr(os, "O_DIRECT", 0):
        raise RuntimeError("本平台无 O_DIRECT，随机读会被页缓存污染，基准无意义")

    # 随机偏移（4 KB 对齐，且保证 block 不越界）
    import random

    rng = random.Random(20261025)
    offs = [rng.randrange(0, max(1, nblk - blen // _ALIGN)) * _ALIGN for _ in range(iters)]

    fd = os.open(path, flags)
    mm = mmap.mmap(-1, blen)
    buf = memoryview(mm)[:blen]
    try:
        _pread_one(fd, buf, offs[0])  # 预热（建 inode/块映射）
        lats = []
        t_all = time.perf_counter()
        for off in offs:
            t0 = time.perf_counter()
            _pread_one(fd, buf, off)
            lats.append(_ms(time.perf_counter() - t0))
        wall = time.perf_counter() - t_all
    finally:
        os.close(fd)

    seq = {
        "block_kb": block_kb,
        "qd": 1,
        "iops": round(len(offs) / wall, 1),
        "MBps": round(len(offs) * blen / wall / 1e6, 1),
        "lat": _stats(lats),
    }
    if qd <= 1:
        return seq

    # QD=N：每线程自己一个 fd + 缓冲，轮转取偏移
    def _worker(seed: int) -> list[float]:
        f = os.open(path, flags)
        m = mmap.mmap(-1, blen)
        b = memoryview(m)[:blen]
        loc = []
        r = random.Random(seed)
        try:
            for _ in range(max(1, iters // qd)):
                off = r.randrange(0, max(1, nblk - blen // _ALIGN)) * _ALIGN
                t0 = time.perf_counter()
                _pread_one(f, b, off)
                loc.append(_ms(time.perf_counter() - t0))
        finally:
            os.close(f)
        return loc

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=qd) as ex:
        parts = list(ex.map(_worker, range(qd)))
    wall_qd = time.perf_counter() - t0
    flat = [x for p in parts for x in p]
    return {
        "block_kb": block_kb,
        "qd": qd,
        "iops": round(len(flat) / wall_qd, 1),
        "MBps": round(len(flat) * blen / wall_qd / 1e6, 1),
        "lat": _stats(flat),
    }


# ===== 4. mmap 页缓存放大：热集能否把冷盘兜住 =====


def _touch_pages(mm: mmap.mmap, offs: list[int]) -> None:
    """逐页触碰（每页取 1 字节），触发缺页读盘."""
    for off in offs:
        _ = mm[off]


def _mount_source(path: Path) -> str:
    """最长前缀匹配 /proc/self/mounts → 块设备名（如 /dev/sdd）；找不到返回空串."""
    best, src = -1, ""
    try:
        with open("/proc/self/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                dev, mp = parts[0], parts[1].replace("\\040", " ")
                if str(path).startswith(mp) and len(mp) > best:
                    best, src = len(mp), dev
    except OSError:
        pass
    return src


def _sectors_read(dev: str) -> int | None:
    """块设备累计读扇区数（/sys/block/<name>/stat 第 3 列，512 B/扇区）.

    容器里常常没有 `/proc/self/io`，改用设备级计数（本机空载时足够准）。
    """
    name = os.path.basename(dev) if dev else ""
    if not name:
        return None
    try:
        with open(f"/sys/block/{name}/stat") as f:
            return int(f.read().split()[2])
    except (OSError, IndexError, ValueError):
        return None


def bench_mmap(path: Path, npages: int, hot_frac: float) -> dict:
    """两遍 + 热集：冷遍测随机缺页，热遍验页缓存，热集重复访问验命中率.

    只对本文件 `POSIX_FADV_DONTNEED`（不动全局 drop_caches，不影响别人）。
    """
    size = path.stat().st_size
    total_pages = size // _PAGE
    import random

    rng = random.Random(7)
    npages = min(npages, total_pages)
    cold_offs = [rng.randrange(total_pages) * _PAGE for _ in range(npages)]
    hot_n = max(1, int(npages * hot_frac))
    hot_offs = rng.sample(cold_offs, min(hot_n, len(cold_offs)))

    # 先把这个文件的页缓存丢掉（只影响本文件）
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

    mfd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(mfd, 0, access=mmap.ACCESS_READ)  # mmap 自己 dup fd，原文可关
    finally:
        os.close(mfd)
    dev = _mount_source(path)
    mb = 512.0 / 1e6  # 扇区 → MB
    try:
        b0 = _sectors_read(dev)
        t0 = time.perf_counter()
        _touch_pages(mm, cold_offs)
        cold_ms = _ms(time.perf_counter() - t0)
        b1 = _sectors_read(dev)

        t0 = time.perf_counter()
        _touch_pages(mm, cold_offs)
        warm_ms = _ms(time.perf_counter() - t0)
        b2 = _sectors_read(dev)

        t0 = time.perf_counter()
        for _ in range(10):  # 热集 10 轮：命中率上界的直观量
            _touch_pages(mm, hot_offs)
        hot_ms = _ms(time.perf_counter() - t0)
        b3 = _sectors_read(dev)
    finally:
        mm.close()

    def _delta(a, b) -> float | None:
        return round((b - a) * mb, 1) if (a is not None and b is not None) else None

    return {
        "pages": npages,
        "hot_pages": len(hot_offs),
        "cold_per_page_us": round(cold_ms * 1000 / npages, 2),
        "warm_per_page_us": round(warm_ms * 1000 / npages, 2),
        "hot_per_page_us": round(hot_ms * 1000 / (len(hot_offs) * 10), 2),
        "cold_read_MB": _delta(b0, b1),
        "warm_read_MB": _delta(b1, b2),
        "hot_read_MB": _delta(b2, b3),
        "cold_ms": round(cold_ms, 1),
        "warm_ms": round(warm_ms, 1),
        "dev": dev,
    }


# ===== 5. 判定与报告 =====


def verdict(res: dict) -> dict:
    """按预设阈值给结论（先定死阈值，避免事后找理由）.

    阈值来源：
    - H2D 4 KB pinned 端到端 < 30 µs：与一次 kernel launch 同量级，可藏进计算；
    - 磁盘 4 KB QD=1 p50 < 200 µs：一次随机读能藏在 1 ms 级算子后面；
    - 宿主表 gather（4096 token/步）< 5 ms：相对 128M 模型单步百 ms 级可接受。
    """
    h2d = {(r["size_kb"], r["pinned"]): r for r in res["h2d"]}
    h2d_small = h2d.get((4, True), {})
    gather = res["host_table"]
    disk4 = next((r for r in res["disk"] if r["block_kb"] == 4 and r["qd"] == 1), {})

    h2d_ok = h2d_small.get("e2e_avg_ms", 1e9) < 0.030
    gather_ok = gather.get("total_p50_ms", 1e9) < 5.0
    disk_ok = disk4.get("lat", {}).get("p50_ms", 1e9) < 0.200
    return {
        "h2d_4k_pinned_us": round(h2d_small.get("e2e_avg_ms", float("nan")) * 1000, 1),
        "host_gather_4096tok_ms": gather.get("total_p50_ms"),
        "disk_4k_qd1_p50_us": round(disk4.get("lat", {}).get("p50_ms", float("nan")) * 1000, 1),
        "host_table_ok": bool(h2d_ok and gather_ok),
        "disk_table_ok": bool(disk_ok),
        "建议": (
            ("A 档（表留 host + 按需搬行）可行；" if h2d_ok and gather_ok else
             "A 档不划算（H2D 或 CPU gather 太贵）；") +
            ("C 档（表落盘 + 页缓存兜热行）可行" if disk_ok else
             "C 档不可行（4 KB 随机读太慢），表只能留在内存且受容量限制")
        ),
    }


def _md_table(rows: list[dict], cols: list[str]) -> str:
    """把若干 dict 渲染成 markdown 表（缺列按空串）."""
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def write_report(res: dict, out_dir: Path, tag: str) -> tuple[Path, Path]:
    """落 JSON（机器读）+ Markdown（人读，方便直接贴进 docs）."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{tag}.json"
    mp = out_dir / f"{tag}.md"
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    v = res["verdict"]
    h2d_rows = [{
        "size": f'{r["size_kb"]} KB',
        "pinned": "是" if r["pinned"] else "否",
        "设备侧 p50": f'{r["device"]["p50_ms"] * 1000:.1f} µs',
        "批量摊销": f'{r["batch_us"]:.1f} µs',
        "端到端": f'{r["e2e_avg_ms"] * 1000:.1f} µs',
        "等效带宽": f'{r["e2e_GBps"]:.2f} GB/s',
    } for r in res["h2d"]]
    disk_rows = [{
        "块大小": f'{r["block_kb"]} KB',
        "QD": r["qd"],
        "IOPS": r["iops"],
        "MB/s": r["MBps"],
        "p50": f'{r["lat"]["p50_ms"] * 1000:.0f} µs',
        "p95": f'{r["lat"]["p95_ms"] * 1000:.0f} µs',
    } for r in res["disk"]]
    g = res["host_table"]
    mmr = res["mmap"]

    md = f"""# 记忆分层硬件基准（{res["时间"]}）

机器：{res["环境"]["gpu"]} / {res["环境"]["cpu"]} / 内存 {res["环境"]["ram_gb"]} GB
临时文件：`{res["环境"]["scratch"]}`（{res["环境"]["file_gb"]} GB，O_DIRECT 绕页缓存）

## 结论

- **H2D 4 KB locked 端到端**：{v["h2d_4k_pinned_us"]} µs
- **宿主表 gather + 上卡**（4096 token/步）：{v["host_gather_4096tok_ms"]} ms
- **磁盘 4 KB 随机读 p50（QD=1）**：{v["disk_4k_qd1_p50_us"]} µs
- **A 档（表留 host）**：{"✅ 可行" if v["host_table_ok"] else "❌ 不划算"}
- **C 档（表落盘）**：{"✅ 可行" if v["disk_table_ok"] else "❌ 不可行"}

> {v["建议"]}

## 1. H2D 拷贝（pinned vs pageable）

{_md_table(h2d_rows, ["size", "pinned", "设备侧 p50", "批量摊销", "端到端", "等效带宽"])}

判读：小块看的不是带宽而是**固定开销**（launch + 同步）。端到端与设备侧的差就是 host 侧下发成本；
pageable 行的端到端明显高于 pinned 说明拷贝在阻塞 host，异步预取必须用 pinned 源。

## 2. 宿主表 gather（表在 CPU，只搬命中行）

| 项 | 值 |
|---|---|
| 表规模 | {g["slots"]:,} 行 × {g["cols"]} 列 |
| 每步行数 | {g["n_rows"]:,}（= token × 阶 × 头） |
| 搬运量 | {g["bytes"] / 1e6:.2f} MB |
| CPU gather p50 | {g["gather_ms"]["p50_ms"]:.2f} ms |
| 汇聚 pinned p50 | {g["pin_copy_ms"]["p50_ms"]:.2f} ms |
| H2D p50 | {g["h2d_ms"]["p50_ms"]:.2f} ms |
| **合计 p50** | **{g["total_p50_ms"]:.2f} ms** |

判读：CPU 侧随机 gather 是**串行**的，索引每行基本一次 cache miss；它不随 GPU 变快，
是"表留 host"最容易被低估的一项。

## 3. 磁盘随机读（O_DIRECT）

{_md_table(disk_rows, ["块大小", "QD", "IOPS", "MB/s", "p50", "p95"])}

## 4. mmap 页缓存

| 项 | 值 |
|---|---|
| 抽样页数 | {mmr["pages"]:,} |
| 块设备 | `{mmr.get("dev", "?")}` |
| 冷遍每页 | {mmr["cold_per_page_us"]} µs |
| 热遍每页 | {mmr["warm_per_page_us"]} µs |
| 热集每页（10 轮） | {mmr["hot_per_page_us"]} µs |
| 冷遍块设备读量 | {mmr["cold_read_MB"]} MB |
| 热遍块设备读量 | {mmr["warm_read_MB"]} MB |
| 热集块设备读量 | {mmr["hot_read_MB"]} MB |

判读：冷/热每页耗时之比就是"落盘 + 页缓存"能拿到的收益上限（本机约
{round(mmr["cold_per_page_us"] / max(mmr["warm_per_page_us"], 1e-9))}×）；热集读盘接近 0 说明
只要工作集能被页缓存兜住，落盘表的稳态代价由**命中率**而非盘速决定。
mmap 冷缺页会自动带上预读，所以每页均摊常比 O_DIRECT 单块读便宜——但 I/O 总量被放大，
表更大、访问更散时这个便宜会消失。
"""
    mp.write_text(md, encoding="utf-8")
    return jp, mp


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆分层硬件基准（H2D / 宿主表 / 磁盘 / mmap）")
    ap.add_argument("--scratch-dir", default="/root/.cache/crucible-bench",
                    help="临时大文件目录（必须本地盘；NFS/tmpfs 测不出真实盘速）")
    ap.add_argument("--out", default="data/bench", help="报告输出目录（已 gitignore）")
    ap.add_argument("--file-gb", type=float, default=2.0, help="临时文件大小")
    ap.add_argument("--block-kb", default="4,64,256", help="磁盘随机读块大小（逗号分隔）")
    ap.add_argument("--qd", type=int, default=8, help="并发深度（额外测一组）")
    ap.add_argument("--iters", type=int, default=200, help="每场景采样次数")
    ap.add_argument("--h2d-iters", type=int, default=200)
    ap.add_argument("--slots", type=int, default=1_000_000, help="宿主表行数")
    ap.add_argument("--cols", type=int, default=32, help="每行字节数（FP8 表每头维度）")
    ap.add_argument("--tokens", type=int, default=4096, help="按多少 token 折算每步行数")
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mmap-pages", type=int, default=20000)
    ap.add_argument("--hot-frac", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true", help="缩小规模，秒级冒烟")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA（本基准的第一段就是 H2D）")

    if args.quick:
        args.file_gb = 0.25
        args.iters = 40
        args.h2d_iters = 40
        args.mmap_pages = 2000

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / "rand.bin"
    t0 = time.time()
    if not tmp.exists() or tmp.stat().st_size < int(args.file_gb * (1 << 30)):
        print(f"[1/4] 造临时文件 {tmp}（{args.file_gb} GB）…", flush=True)
        _make_file(tmp, args.file_gb)
    else:
        print(f"[1/4] 复用临时文件 {tmp}", flush=True)

    print("[2/4] H2D 小块延迟…", flush=True)
    h2d = []
    for pinned in (True, False):
        h2d += bench_h2d([4, 64, 1024, 8192], args.h2d_iters, pinned)

    print("[3/4] 宿主表 gather…", flush=True)
    n_rows = args.tokens * args.orders * args.heads
    host_table = bench_host_table(args.slots, args.cols, n_rows, max(10, args.iters // 4))

    print("[4/4] 磁盘随机读 + mmap…", flush=True)
    disk = []
    for kb in [int(x) for x in args.block_kb.split(",") if x]:
        disk.append(bench_disk(tmp, args.file_gb, kb, args.iters, qd=1))
        disk.append(bench_disk(tmp, args.file_gb, kb, args.iters, qd=args.qd))
    mmr = bench_mmap(tmp, args.mmap_pages, args.hot_frac)

    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((ln.split(":", 1)[1].strip()
                        for ln in f if ln.startswith("model name")), "?")
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        cpu, ram_gb = "?", 0.0

    res = {
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "耗时秒": round(time.time() - t0, 1),
        "环境": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "cpu": cpu,
            "ram_gb": ram_gb,
            "scratch": str(tmp),
            "file_gb": args.file_gb,
            "torch": torch.__version__,
        },
        "params": vars(args),
        "h2d": h2d,
        "host_table": host_table,
        "disk": disk,
        "mmap": mmr,
    }
    res["verdict"] = verdict(res)

    tag = time.strftime("mem_tier_%Y%m%d_%H%M%S")
    jp, mp = write_report(res, Path(args.out), tag)
    print(f"\n完成：{jp}\n      {mp}")
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
