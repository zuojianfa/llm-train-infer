"""在语料上训练 BPE 分词器 -> out/tokenizer.json"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path

from minillm.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/train.txt")
    ap.add_argument("--out", default="out/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=8192)
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"[tokenizer] training BPE on {len(text):,} chars -> vocab={args.vocab_size}")
    t0 = time.time()
    tok = BPETokenizer.train(text, vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"[tokenizer] done in {time.time()-t0:.1f}s, actual vocab={tok.vocab_size}, saved to {args.out}")

    # 自检:编解码往返
    sample = "Hello world! 机器学习很有趣."
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"[tokenizer] roundtrip check:")
    print(f"  input : {sample!r}")
    print(f"  ids   : {ids[:20]}{'...' if len(ids) > 20 else ''} (len={len(ids)})")
    print(f"  decode: {back!r}")


if __name__ == "__main__":
    main()
