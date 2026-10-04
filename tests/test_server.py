"""服务单测 — stream 与 generate 一致性 + OpenAI 端到端（本机回环）."""

import json
import threading
import time
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from src.llm.local.config import tiny_test_config
from src.llm.local.infer import LocalChatBackend, SimpleTokenizer
from src.llm.local.model import TinyLLM
from src.llm.local.server import OpenAIServer


def _tiny_backend(tmp_path):
    """tiny 权重按训练落盘布局存盘（model.pt + vocab.json），走 load 回读."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    tok = SimpleTokenizer(256)
    tok.fit(["你好世界", "今天天气不错", "hello world test"])
    m.eval()
    prefix = str(tmp_path / "model")
    torch.save(m.state_dict(), prefix + ".pt")
    with open(tmp_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(tok._chars, f, ensure_ascii=False)
    return LocalChatBackend.load(prefix, tiny_test_config())


def test_stream_matches_generate():
    """stream 逐 token 拼接 == generate 贪心全量（同一种子一路数学）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    m.eval()
    x = torch.randint(0, 256, (1, 8))
    full = m.generate(x, max_new_tokens=8)
    expect = full[0].tolist()[8:]
    got = list(m.stream_tokens(x, max_new_tokens=8))
    assert got == expect
    # train 模式调用后恢复（generate 的 eval 保护同样覆盖 stream）
    m.train()
    list(m.stream_tokens(x, max_new_tokens=2))
    assert m.training


def _serve_in_thread(backend, **kwargs):
    """后台起服务，返回 (server, port)."""
    server = OpenAIServer(backend, **kwargs)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"host": "127.0.0.1", "port": 0},
        daemon=True)
    thread.start()
    for _ in range(100):
        if server._httpd is not None:
            break
        time.sleep(0.05)
    port = server._httpd.server_address[1]
    return server, port


def _get(port, path, key=""):
    """GET（/v1/models 用）."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers={**({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def _post(port, path, obj, key=""):
    """POST JSON（可带 Bearer）."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def test_openai_endpoints(tmp_path):
    """端到端：models/chat/completions 鉴权/非流式/流式."""
    backend = _tiny_backend(tmp_path)
    server, port = _serve_in_thread(backend, model_name="tiny-test",
                                    api_key="secret")
    try:
        # 未授权（GET /v1/models 错 key 应 401）
        try:
            _get(port, "/v1/models", key="wrong")
            authed = True
        except Exception as e:
            authed = "401" not in str(e)
        assert not authed
        # models
        code, body = _get(port, "/v1/models", key="secret")
        assert code == 200 and json.loads(body)["data"][0]["id"] == "tiny-test"
        # chat 非流式
        code, body = _post(port, "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "你好"}],
                            "max_tokens": 8, "stop": ["zzz"]}, key="secret")
        data = json.loads(body)
        assert code == 200
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)
        # chat 流式：增量读到 [DONE]（真 SSE 客户端行为，顺带验证逐块到达）
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "你好"}],
                             "max_tokens": 8, "stream": True}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer secret"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            chunks, done = [], False
            buf = ""
            while not done:
                piece = resp.read(64).decode()
                if not piece:
                    break
                buf += piece
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                        else:
                            chunks.append(json.loads(payload))
        assert done, "流式缺少 [DONE] 收尾"
        assert chunks and all("delta" in c["choices"][0] for c in chunks)
        # 旧 completions 接口
        code, body = _post(port, "/v1/completions",
                           {"prompt": "你好", "max_tokens": 8}, key="secret")
        assert code == 200 and "text" in json.loads(body)["choices"][0]
    finally:
        server.shutdown()


def test_apply_stop():
    """stop 截断：命中截断并改 finish，miss 保持 length."""
    text, finish = OpenAIServer._apply_stop("abcSTOPdef", ["STOP"])
    assert (text, finish) == ("abc", "stop")
    text, finish = OpenAIServer._apply_stop("abcdef", ["STOP"])
    assert (text, finish) == ("abcdef", "length")
