"""推理服务:FastAPI 提供 OpenAI Chat Completions 兼容接口。

启动:
    python scripts/serve.py --ckpt out/ckpt --host 0.0.0.0 --port 8000
调用:
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .generate import Generator
from .model import LLMModel
from .tokenizer import BPETokenizer


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    model: str = "minillm"
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float | None = 0.9
    top_k: int | None = 50
    stream: bool = False
    # 该模型是基座(未对齐),默认不做模板拼接,直接续写最后一条 user 消息


def format_prompt(messages: list[ChatMessage]) -> str:
    """把对话拼成续写 prompt。基座模型无指令微调,简单取最后一条用户输入。"""
    last_user = [m for m in messages if m.role == "user"]
    return last_user[-1].content if last_user else ""


def create_app(model: LLMModel, tokenizer: BPETokenizer, device, dtype) -> FastAPI:
    app = FastAPI(title="minillm", version="0.1.0")
    gen = Generator(model, tokenizer, device, dtype)

    def _chunk(cid: str, created: int, text: str | None, finish: str | None,
               first_role: bool = False) -> str:
        delta = {}
        if first_role:
            delta["role"] = "assistant"
        if text is not None:
            delta["content"] = text
        obj = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": "minillm",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    @app.get("/health")
    def health():
        return {"status": "ok", "vocab_size": tokenizer.vocab_size}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        prompt = format_prompt(req.messages)

        if req.stream:
            def sse() -> Iterator[str]:
                yield _chunk(cid, created, None, None, first_role=True)
                first = True
                for piece in gen.stream(prompt, max_new_tokens=req.max_tokens,
                                        temperature=req.temperature,
                                        top_k=req.top_k, top_p=req.top_p):
                    if first:  # 跳过 prompt 回显由 echo_prompt=False 保证,这里 piece 均为新内容
                        first = False
                    yield _chunk(cid, created, piece, None)
                yield _chunk(cid, created, None, "stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text = gen.generate(prompt, max_new_tokens=req.max_tokens,
                            temperature=req.temperature, top_k=req.top_k,
                            top_p=req.top_p)
        # generate 会带 prompt 前缀,返回时去掉
        completion = text[len(prompt):] if text.startswith(prompt) else text
        return {
            "id": cid, "object": "chat.completion", "created": created,
            "model": "minillm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(tokenizer.encode(prompt)),
                      "completion_tokens": len(tokenizer.encode(completion)),
                      "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(completion))},
        }

    @app.post("/v1/completions")
    def completions(payload: dict):
        prompt = payload.get("prompt", "")
        text = gen.generate(prompt,
                            max_new_tokens=int(payload.get("max_tokens", 128)),
                            temperature=float(payload.get("temperature", 0.8)),
                            top_k=payload.get("top_k", 50),
                            top_p=payload.get("top_p", 0.9))
        return {"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion",
                "choices": [{"index": 0, "text": text[len(prompt):], "finish_reason": "stop"}]}

    return app
