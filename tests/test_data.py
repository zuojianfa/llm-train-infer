"""数据管道单元测试:token 流构造、(x,y) 右移对齐、batch 形状、可复现采样。

TextDataset 是训练循环的地基,"y == x 右移一位"这条不变量一旦写错,
模型会学成"预测当前词",loss 看似下降但生成完全错位——所以单独锁死。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minillm.data import TextDataset, make_token_stream


def _toy_dataset() -> TextDataset:
    # 手工构造已知 token 流:10..199
    return TextDataset(list(range(10, 210)))


def test_from_file_inserts_eos_per_line(tiny_tokenizer, corpus_file):
    ds = TextDataset.from_file(str(corpus_file), tiny_tokenizer)
    n_lines = len([l for l in corpus_file.read_text(encoding="utf-8").split("\n") if l.strip()])
    # 每行末尾恰好追加一个 <eos>(id=0)
    eos_count = int((ds.data == 0).sum())
    assert eos_count == n_lines


def test_x_is_shifted_y():
    """核心不变量:对任意窗口,y[i] == x[i+1](next-token 目标)。"""
    ds = _toy_dataset()
    torch.manual_seed(0)
    x, y = ds.random_batch(batch_size=3, seq_len=16, device=torch.device("cpu"))
    assert x.shape == (3, 16) and y.shape == (3, 16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "x/y 没有严格右移对齐!"


def test_random_batch_shapes_and_dtype():
    ds = _toy_dataset()
    x, y = ds.random_batch(4, 8, torch.device("cpu"))
    assert x.dtype == torch.long and y.dtype == torch.long   # embedding 要求 int64
    assert int(x.min()) >= 10 and int(y.max()) <= 209         # 全部落在流内


def test_fixed_batch_reproducible():
    """同 seed 两次 fixed_batch 必须给出完全相同的窗口(评估可比的前提)。"""
    ds = _toy_dataset()
    x1, y1 = ds.fixed_batch(3, 16, seed=7, device=torch.device("cpu"))
    x2, y2 = ds.fixed_batch(3, 16, seed=7, device=torch.device("cpu"))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    x3, _ = ds.fixed_batch(3, 16, seed=8, device=torch.device("cpu"))
    assert not torch.equal(x1, x3)                # 不同 seed 应给不同窗口


def test_corpus_too_short_raises():
    ds = TextDataset([1, 2, 3])
    with pytest.raises(ValueError):
        ds.random_batch(2, 10, torch.device("cpu"))


def test_make_token_stream_matches_dataset(tiny_tokenizer, corpus_file):
    arr = make_token_stream(str(corpus_file), tiny_tokenizer)
    ds = TextDataset.from_file(str(corpus_file), tiny_tokenizer)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, ds.data)


def test_empty_lines_skipped(tiny_tokenizer, tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("hello\n\n\nworld\n", encoding="utf-8")
    ds = TextDataset.from_file(str(p), tiny_tokenizer)
    # 两个非空行 => 两个 <eos>;空行不贡献任何 token
    assert int((ds.data == 0).sum()) == 2
