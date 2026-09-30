"""预训练入口(命令行)。

示例:
    # 默认 ~70M 模型,5000 步
    python scripts/train.py

    # 冒烟测试:小模型、少量步数
    python scripts/train.py --dim 256 --num-layers 4 --max-steps 20 \
        --batch-size 8 --seq-len 128 --warmup-steps 5 --eval-interval 10 \
        --save-interval 10 --out-dir out/smoke

    # 断点续训(自动读取 out/ckpt)
    python scripts/train.py --resume
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from minillm.config import TrainConfig
from minillm.train import train


def main():
    ap = argparse.ArgumentParser(description="minillm pretraining")
    # 数据
    ap.add_argument("--data", default="data/train.txt")
    ap.add_argument("--val", default="data/val.txt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    # 模型规模
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--num-kv-heads", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    # 优化
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    # 运行
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--save-interval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="从头训练,不加载已有 checkpoint")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        val_path=args.val,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        out_dir=args.out_dir,
    )
    overrides = {
        "dim": args.dim, "num_layers": args.num_layers, "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads, "hidden_dim": args.hidden_dim,
        "max_seq_len": args.max_seq_len,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg.model, k, v)
    if args.hidden_dim is None and args.dim is not None:
        # SwiGLU 中间层约为 8/3 * dim,并对齐到 32 的倍数
        cfg.model.hidden_dim = (round(args.dim * 8 / 3) // 32) * 32

    for k in ("lr", "weight_decay", "warmup_steps", "max_steps", "batch_size",
              "seq_len", "log_interval", "eval_interval", "save_interval", "seed"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum

    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()
