"""交互式 / 一次性文本生成(推理测试入口)。

示例:
    # 一次性续写
    python scripts/generate.py --ckpt out/ckpt --prompt "机器学习是" --max-new-tokens 128

    # greedy 解码 + 交互模式(/exit 退出)
    python scripts/generate.py --ckpt out/best --interactive --temperature 0
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from minillm.generate import Generator
from minillm.tokenizer import BPETokenizer
from minillm.train import get_device, get_dtype, load_model


def main():
    ap = argparse.ArgumentParser(description="minillm inference")
    ap.add_argument("--ckpt", default="out/ckpt", help="checkpoint 目录(model.pt+meta.json)")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="<=0 表示 greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32", "float16"],
                    help="CPU 推理建议 float32")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    tok = BPETokenizer.load(args.tokenizer)
    model, step = load_model(args.ckpt, device, dtype)
    print(f"[gen] loaded {args.ckpt} (step={step}), device={device}, dtype={dtype}")

    gen = Generator(model, tok, device, dtype)

    def run(prompt: str):
        out = gen.generate(prompt, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           top_k=args.top_k or None, top_p=args.top_p or None)
        print(out)
        print("-" * 60)

    if args.interactive:
        while True:
            try:
                p = input("prompt> ").strip()
            except EOFError:
                break
            if p in ("/exit", "/quit", "exit", "quit", ""):
                break
            run(p)
    elif args.prompt:
        run(args.prompt)
    else:
        ap.error("请提供 --prompt 或使用 --interactive")


if __name__ == "__main__":
    main()
