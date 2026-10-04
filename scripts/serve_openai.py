"""OpenAI 兼容服务启动 — 本地权重秒变 /v1/chat/completions.

用法：
  python scripts/serve_openai.py --model data/llm-ckpt/model --port 8000
  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true}'

说明：--model 指向权重前缀（model.pt + vocab.json 同目录）；
--config 可选（默认 base 预设；迁移结构用 migrate 输出的 config.json）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="OpenAI 兼容服务")
    p.add_argument("--model", required=True, help="权重前缀（如 data/llm-ckpt/model）")
    p.add_argument("--config", default="", help="模型配置 JSON（默认 base 预设）")
    p.add_argument("--name", default="cruciblelm", help="对外模型名（/v1/models 返回）")
    p.add_argument("--host", default="0.0.0.0", help="监听地址")  # noqa: S104 - 服务绑定按需配置
    p.add_argument("--port", type=int, default=8000, help="监听端口")
    p.add_argument("--api-key", default="", help="Bearer 鉴权（空=不鉴权）")
    p.add_argument("--device", default="", help="设备（空=自动，cuda 优先）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """服务主流程."""
    import torch

    from src.llm.local.config import SmallLLMConfig
    from src.llm.local.infer import LocalChatBackend
    from src.llm.local.server import OpenAIServer

    args = parse_args(argv)
    if args.config:
        import json

        config = SmallLLMConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        config = SmallLLMConfig()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend = LocalChatBackend.load(args.model, config)
    backend.model = backend.model.to(device)
    print(f"权重已加载：{args.model}（{device}）", flush=True)
    OpenAIServer(backend, model_name=args.name, api_key=args.api_key).serve_forever(
        args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
