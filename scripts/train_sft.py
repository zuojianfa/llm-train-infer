"""SFT 监督微调入口(命令行)。

在预训练 checkpoint 的基础上,用指令-回答配对数据继续训练。
模型只学习续写 "助手:" 之后的答案(prompt 段 label=-100 被屏蔽)。

示例:
    # 1) 先生成演示指令数据(离线内置)
    python scripts/prepare_sft_data.py

    # 2) 从冒烟基座热启动做 SFT
    python scripts/train_sft.py --init-ckpt out/smoke/best \
        --train data/sft_train.jsonl --val data/sft_val.jsonl \
        --dim 256 --num-layers 4 --epochs 30 --batch-size 4 \
        --lr 6e-5 --warmup-steps 5 --eval-interval 10 --save-interval 20 \
        --out-dir out/sft --device cpu --dtype float32

    # 3) 用微调后的模型生成对比效果
    python scripts/generate.py --ckpt out/sft/best --tokenizer out/tokenizer.json \
        --prompt "用户:什么是注意力机制?\n助手:" --temperature 0.8
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from minillm.config import TrainConfig
from minillm.sft import run_sft


def main():
    ap = argparse.ArgumentParser(description="minillm supervised fine-tuning (SFT)")
    # 数据与初始化
    ap.add_argument("--init-ckpt", default="out/ckpt",
                    help="预训练 checkpoint 目录(SFT 从这里热启动)")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft_val.jsonl")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    # 模型规模:必须与 init-ckpt 的 meta.json 一致(load_model 按 ckpt 配置重建,
    # 这里的参数只是冗余校验用途时可省略;默认不传即沿用 ckpt 结构)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    # 优化:SFT 默认小学习率(预训练的 ~1/5),防灾难性遗忘
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3, help="数据轮数(SFT 通常 1~3)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    # 运行
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu", "mps"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--out-dir", default="out-sft")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = TrainConfig(
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.epochs,          # run_sft 内部把"步数"重新解释为 epoch 数
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.grad_accum,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    for k in ("dim", "num_layers"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg.model, k, v)

    run_sft(cfg, init_ckpt=args.init_ckpt, data_path=args.train, val_path=args.val)


if __name__ == "__main__":
    main()
