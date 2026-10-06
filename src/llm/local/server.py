"""OpenAI 兼容服务 — /v1/chat/completions（流式 SSE + 非流式）与 /v1/models.

零第三方依赖（stdlib http.server + 线程池），OpenAI SDK / 任意 chat UI 直连。
chat 消息拼法与 LocalChatBackend 一致（中文 role 标签）。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.llm.local.infer import LocalChatBackend


class OpenAIServer:
    """OpenAI 兼容服务：backend 注入 + 可选 Bearer 鉴权."""

    def __init__(self, backend: LocalChatBackend, model_name: str = "cruciblelm",
                 api_key: str = ""):
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self._lock = threading.Lock()  # 串行生成（单卡 KV 状态不可并发）
        self._httpd: ThreadingHTTPServer | None = None

    def serve_forever(self, host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104 - 服务绑定按需配置
        """阻塞服务（Ctrl-C 退出）."""
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((host, port), handler)
        print(f"OpenAI-compatible API: http://{host}:{port}/v1", flush=True)
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        """停止服务（测试用）."""
        if self._httpd is not None:
            self._httpd.shutdown()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """构造请求处理器（闭包绑定 self）."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # 静默访问日志
                pass

            def _send_json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _unauthorized(self) -> bool:
                """鉴权检查（未配 api_key 则放行），返回 True 表示已拒绝."""
                if not server.api_key:
                    return False
                auth = self.headers.get("Authorization", "")
                if auth != f"Bearer {server.api_key}":
                    self._send_json(401, {"error": {"message": "无效的 API Key",
                                                    "type": "invalid_request_error"}})
                    return True
                return False

            def do_GET(self) -> None:  # noqa: N802 - http.server 命名约定
                if self._unauthorized():
                    return
                path = urllib.parse.urlparse(self.path).path
                if path == "/v1/models":
                    self._send_json(200, {"object": "list", "data": [{
                        "id": server.model_name, "object": "model",
                        "created": int(time.time()), "owned_by": "cruciblelm"}]})
                elif path in ("/health", "/v1/health"):
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": {"message": "未知路径"}})

            def do_POST(self) -> None:  # noqa: N802 - http.server 命名约定
                if self._unauthorized():
                    return
                path = urllib.parse.urlparse(self.path).path
                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    req = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send_json(400, {"error": {"message": "非法 JSON"}})
                    return
                if path == "/v1/chat/completions":
                    server.handle_chat(self, req)
                elif path == "/v1/completions":
                    server.handle_completions(self, req)
                else:
                    self._send_json(404, {"error": {"message": "未知路径"}})

        return Handler

    def _chat_params(self, req: dict) -> tuple[list[dict], int, float, int, float, list[str]]:
        """解析通用参数（messages/max_tokens/temperature/top_k/repetition_penalty/stop）."""
        messages = req.get("messages") or []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        max_tokens = int(req.get("max_tokens") or 128)
        temperature = float(req.get("temperature") or 0.0)
        top_k = int(req.get("top_k") or req.get("top_logprobs") or 0)
        repetition_penalty = float(req.get("repetition_penalty") or 1.0)
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        return messages, max_tokens, temperature, top_k, repetition_penalty, stop

    @staticmethod
    def _apply_stop(text: str, stop: list[str]) -> tuple[str, str]:
        """截断 stop 序列之后的内容，返回 (文本, finish_reason)."""
        for s in stop:
            if s and s in text:
                return text.split(s)[0], "stop"
        return text, "length"

    def handle_chat(self, handler: BaseHTTPRequestHandler, req: dict) -> None:
        """处理 chat 请求（stream=true 走 SSE 真增量）。"""
        (messages, max_tokens, temperature, top_k,
         repetition_penalty, stop) = self._chat_params(req)
        stream = bool(req.get("stream", False))
        model_id = req.get("model") or self.model_name
        if not stream:
            with self._lock:
                resp = self.backend.chat(messages, max_new_tokens=max_tokens,
                                         temperature=temperature, top_k=top_k,
                                         repetition_penalty=repetition_penalty)
            content, finish = self._apply_stop(resp["content"] or "", stop)
            handler._send_json(200, {  # noqa: SLF001 - 同类内部调用
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion", "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": content},
                             "finish_reason": finish or resp["finish_reason"]}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0}})
            return
        # SSE 流式：prefill 一次，逐 token 吐 data 块；
        # 显式 Connection: close（无 Content-Length/chunked 时 keep-alive 会让
        # 读到 EOF 的客户端永远等待；本服务吞吐下短连接足够）
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        created = int(time.time())
        prefix = {"id": f"chatcmpl-{created}000", "object": "chat.completion.chunk",
                  "created": created, "model": model_id}
        try:
            with self._lock:
                for piece in self.backend.stream(messages, max_new_tokens=max_tokens,
                                                 temperature=temperature, top_k=top_k,
                                                 repetition_penalty=repetition_penalty):
                    chunk = dict(prefix)
                    chunk["choices"] = [{"index": 0,
                                         "delta": {"content": piece},
                                         "finish_reason": None}]
                    handler.wfile.write(
                        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    handler.wfile.flush()
            done = dict(prefix)
            done["choices"] = [{"index": 0, "delta": {},
                                "finish_reason": "stop"}]
            handler.wfile.write(
                f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode())
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
            handler.close_connection = True  # SSE 收尾即关连接：
            # 无 Content-Length 时 keep-alive 会让读到 EOF 的客户端永等
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端提前断开，正常结束

    def handle_completions(self, handler: BaseHTTPRequestHandler, req: dict) -> None:
        """兼容旧 /v1/completions（prompt 当 user 消息，非流式）."""
        prompt = req.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)
        model_id = req.get("model") or self.model_name
        max_tokens = int(req.get("max_tokens") or 128)
        temperature = float(req.get("temperature") or 0.0)
        with self._lock:
            resp = self.backend.chat([{"role": "user", "content": prompt}],
                                     max_new_tokens=max_tokens,
                                     temperature=temperature)
        handler._send_json(200, {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion", "created": int(time.time()),
            "model": model_id,
            "choices": [{"text": resp["content"] or "", "index": 0,
                         "finish_reason": resp["finish_reason"]}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
