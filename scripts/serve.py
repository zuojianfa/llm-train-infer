"""启动推理服务(FastAPI + Uvicorn,OpenAI 兼容接口)。

示例:
    python scripts/serve.py --ckpt out/best --host 0.0.0.0 --port 8000

调用:
    curl http://localhost:8000/health
    curl http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"机器学习是"}],"max_tokens":64}'
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import uvicorn

from minillm.serve import create_app
from minillm.tokenizer import BPETokenizer
from minillm.train import get_device, get_dtype, load_model


def main():
    ap = argparse.ArgumentParser(description="minillm serving")
    ap.add_argument("--ckpt", default="out/ckpt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32", "float16"],
                    help="CPU 服务建议 float32")
    ap.add_argument("--chatml", action="store_true",
                    help="挂载的是 SFT 模型:请求按 \"用户:/助手:\" 模板拼 prompt(与训练一致)")
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    tok = BPETokenizer.load(args.tokenizer)
    model, step = load_model(args.ckpt, device, dtype)
    print(f"[serve] ckpt={args.ckpt} step={step} device={device} dtype={dtype} chatml={args.chatml}")

    app = create_app(model, tok, device, dtype, chatml=args.chatml)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
