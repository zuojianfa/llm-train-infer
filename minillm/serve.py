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
    # 基座模型(未对齐)默认 False,直接续写最后一条 user 消息;
    # SFT 微调后的模型传 chatml=True,套用与训练一致的 "用户:/助手:" 模板
    chatml: bool = False


def format_prompt(messages: list[ChatMessage], chatml: bool = False) -> str:
    """把对话拼成续写 prompt。

    两种模式(对应模型的两种形态):
      * 基座模型(chatml=False,默认):没有对齐过任何模板,直接取最后一条
        用户输入做"续写",最不容易触发分布外行为;
      * SFT 模型(chatml=True):使用与 minillm/sft.py::format_example 完全相同
        的模板 —— "用户:{q}\\n助手:"。训练/推理模板必须逐字符一致,否则
        模型看到的上下文分布和训练时不同,回答质量会明显劣化
        (这是新手做微调最常见的 bug 之一)。多轮对话把所有历史轮都拼进去。
    """
    if not chatml:
        last_user = [m for m in messages if m.role == "user"]
        return last_user[-1].content if last_user else ""
    parts = []
    for m in messages:
        if m.role == "user":
            parts.append(f"用户:{m.content}\n")
        elif m.role == "assistant":
            # 历史 assistant 轮作为上下文保留(去掉其末尾换行差异,统一由模板控制)
            parts.append(f"助手:{m.content}\n")
    parts.append("助手:")                    # 引导模型从"助手:"后开始续写
    return "".join(parts)


def create_app(model: LLMModel, tokenizer: BPETokenizer, device, dtype,
               chatml: bool = False) -> FastAPI:
    """构建推理服务。chatml=True 表示挂载 SFT 模型,请求默认套用对话模板。"""
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
        # 请求级 chatml 优先;未显式指定时跟随服务挂载模式(create_app 的 chatml)
        prompt = format_prompt(req.messages, chatml=req.chatml or chatml)

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
