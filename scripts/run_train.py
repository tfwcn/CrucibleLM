"""训练启动器 — 读 YAML 配置，转 CLI，前台起训练.

用法：
  python scripts/run_train.py configs/sft5.yaml            # 前台跑（Ctrl+C 即停）
  python scripts/run_train.py configs/sft5.yaml --dry-run  # 只打印命令不跑
  python scripts/run_train.py configs/sft5.yaml lr=1e-5 max_steps=500  # 临时覆盖

YAML 键即 train_local_llm.py 的参数名（下划线式，如 max_steps）；
布尔开关写 true/false，为 null 时用训练脚本默认值。
未知键直接报错并列出合法键（防手滑写错）。启动前把本次配置拷进
<ckpt-dir>/run.yaml，复现只认这个文件。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    """解析启动器参数并起训练子进程."""
    p = argparse.ArgumentParser(description="YAML 训练启动器")
    p.add_argument("config", help="YAML 配置（如 configs/sft5.yaml）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不跑")
    p.add_argument("overrides", nargs="*", help="临时覆盖（key=value）")
    args = p.parse_args(argv)

    import yaml  # 延迟导入，保持训练脚本零新增依赖

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{args.config} 不是键值映射")
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"覆盖格式须为 key=value：{item}")
        key, raw = item.split("=", 1)
        try:
            cfg[key] = yaml.safe_load(raw)
        except yaml.YAMLError:
            cfg[key] = raw

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import train_local_llm

    argv = _argv_from_cfg(cfg)
    # 真解析一遍：未知键/非法值在这里直接报错（行为与手写 CLI 一致）
    train_local_llm.parse_args(argv)
    train_argv = [sys.executable, "scripts/train_local_llm.py", *argv]
    print(" ".join(train_argv), flush=True)

    ckpt_dir = Path(cfg.get("ckpt_dir") or "data/llm-ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "run.yaml")
    print(f"配置已存档：{ckpt_dir / 'run.yaml'}", flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(train_argv, cwd=str(Path(__file__).resolve().parent.parent))


def _argv_from_cfg(cfg: dict) -> list[str]:
    """不依赖 parser 对象的直转（键名下划线转连字符，布尔按开关处理）.

    类型校验交给 train 的 parse_args（真解析一遍，未知键/非法值即报错）。
    开关语义：True 即加 flag（全是 store_true 肯定式，--no-amp 本意即禁用，
    同样 True 加）。
    """
    import train_local_llm

    parser_actions = _get_parser_actions(train_local_llm)
    argv: list[str] = []
    for key, val in cfg.items():
        if key == "description":
            continue
        act = parser_actions.get(key)
        if act is None:
            raise SystemExit(f"未知配置键：{key}")
        flag = act
        if val is True:
            argv.append(flag)
        elif val is False or val is None:
            continue
        else:
            argv += [flag, str(val)]
    return argv


def _get_parser_actions(mod) -> dict[str, str]:
    """从 parse_args 源码静态抽取 dest->长 flag（防手写映射漂移）."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mod.parse_args))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        strs = [a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        longs = [s for s in strs if s.startswith("--")]
        if not longs:
            continue
        dest = None
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                dest = kw.value.value
        if dest is None:
            dest = longs[0].lstrip("-").replace("-", "_")
        out[dest] = longs[0]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
