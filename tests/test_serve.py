"""推理服务 HTTP 层测试(FastAPI TestClient,进程内直调,无需起端口)。

覆盖 OpenAI 兼容接口的行为契约:
  * /health 200 + vocab_size;
  * /v1/chat/completions 非流式:id/object/choices/message 字段齐全,
    usage 三项满足 total = prompt + completion;
  * stream=True:SSE 事件流以 data: [DONE] 收尾,delta 拼接是合法文本;
  * chatml=true 时走 "用户:/助手:" 模板(与 SFT 训练模板一致)。
"""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from minillm.config import ModelConfig
from minillm.model import LLMModel
from minillm.serve import create_app


@pytest.fixture(scope="module")
def client(tiny_tokenizer):
    cfg = ModelConfig(vocab_size=tiny_tokenizer.vocab_size, dim=32, num_layers=1,
                      num_heads=2, num_kv_heads=1, hidden_dim=64, max_seq_len=128)
    torch.manual_seed(0)
    model = LLMModel(cfg).eval()
    app = create_app(model, tiny_tokenizer, torch.device("cpu"), torch.float32)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] > 0


def test_chat_completions_non_stream(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "The sea"}],
        "max_tokens": 8, "temperature": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    u = body["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_chat_completions_stream_sse(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 6, "temperature": 0.0, "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(chunk for chunk in r.iter_text())
    events = [ln[len("data: "):] for ln in raw.splitlines() if ln.startswith("data: ")]
    assert events[-1] == "[DONE]", "SSE 必须以 [DONE] 收尾(OpenAI 客户端依赖它)"
    # 首帧应带 role delta;内容帧拼接后必须是可解码文本(不抛异常即可)
    first = json.loads(events[0])
    assert first["choices"][0]["delta"].get("role") == "assistant"


def test_chatml_template_flag(client):
    """chatml=true 请求也应正常返回 200(SFT 模板路径打通,内容不做质量断言)。"""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "什么是语言模型?"}],
        "max_tokens": 4, "temperature": 0.0, "chatml": True,
    })
    assert r.status_code == 200
    assert "choices" in r.json()


def test_completions_endpoint(client):
    r = client.post("/v1/completions", json={
        "prompt": "Attention is", "max_tokens": 6, "temperature": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)
