"""验证集困惑度评估 CLI。

示例:
    python scripts/eval.py --ckpt out/best
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math

import torch

from minillm.data import TextDataset
from minillm.evaluate import evaluate_loss
from minillm.tokenizer import BPETokenizer
from minillm.train import get_device, get_dtype, load_model


def main():
    ap = argparse.ArgumentParser(description="minillm evaluation")
    ap.add_argument("--ckpt", default="out/ckpt")
    ap.add_argument("--val", default="data/val.txt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    tok = BPETokenizer.load(args.tokenizer)
    model, step = load_model(args.ckpt, device, dtype)
    ds = TextDataset.from_file(args.val, tok)

    class _Cfg:
        batch_size = args.batch_size
        seq_len = args.seq_len
        eval_iters = args.iters

    loss = evaluate_loss(model, ds, _Cfg(), device, dtype)
    print(f"[eval] ckpt={args.ckpt} step={step}")
    print(f"[eval] val loss {loss:.4f} | perplexity {math.exp(min(loss, 20)):.2f} "
          f"({args.iters} batches x {args.batch_size}x{args.seq_len})")


if __name__ == "__main__":
    main()
