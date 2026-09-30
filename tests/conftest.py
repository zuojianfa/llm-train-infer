"""pytest 公共 fixture:小词表分词器 / 迷你模型配置 / 临时数据文件。

设计原则:
* 全部离线、秒级完成 —— CI 与本地 `pytest` 一条命令跑通;
* 会话级(session scope)复用分词器,避免每个测试重复训练 BPE(最贵的一步);
* 模型用极小配置(dim=64, 2 层),CPU float32,数值行为与大模型一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 让 `pytest` 在仓库根目录直接可用(无需 pip install -e .)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minillm.config import ModelConfig          # noqa: E402
from minillm.tokenizer import SPACE, BPETokenizer  # noqa: E402


# ------------------------------------------------------------------ 语料
CORPUS_LINES = [
    "The teacher explained the lesson patiently until every student understood.",
    "Language models predict the next token given a context of previous tokens.",
    "Attention lets the model weigh all positions of a sequence in parallel.",
    "分词器把原始文本切分成一个个子词单元。音乐能够表达语言难以描述的情感。",
    "大海在阳光的照耀下闪烁着金色的光芒。北京的秋天天空高远而澄澈。",
    "机器学习的核心是从数据中学习规律,并用学到的规律对新样本做出预测。",
    "Transformer uses self-attention and feed-forward layers to model text.",
    "In 2020, GPT-3 had 175 billion parameters; 12 * 8 = 96 is trivial though!",
]


@pytest.fixture(scope="session")
def corpus_text() -> str:
    return "\n".join(CORPUS_LINES)


@pytest.fixture(scope="session")
def tiny_tokenizer(corpus_text) -> BPETokenizer:
    """会话级小 BPE:vocab_size=1024 => ~1k 次合并,训练耗时约 1~2 秒。

    注意词表本身含 BMP 兜底原子集(~6 万 id),这是无损往返的根基;
    vocab_size 参数只控制"额外学习多少条 merge 规则"(GPT-2 计数方式)。
    """
    return BPETokenizer.train(corpus_text, vocab_size=1024, verbose=False)


@pytest.fixture()
def tiny_model_cfg() -> ModelConfig:
    """极小但结构完整的模型配置:GQA(4 头/2KV头)、RoPE、SwiGLU 全都在。"""
    return ModelConfig(vocab_size=300, dim=64, num_layers=2, num_heads=4,
                       num_kv_heads=2, hidden_dim=128, max_seq_len=128)


@pytest.fixture()
def corpus_file(tmp_path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(CORPUS_LINES), encoding="utf-8")
    return p


@pytest.fixture()
def sft_jsonl(tmp_path) -> Path:
    """两条指令样本的 JSONL,供 SFTDataset 测试使用。"""
    rows = [
        {"instruction": "什么是语言模型?", "output": "根据上下文预测下一个词的程序。"},
        {"instruction": "2+2 等于几?", "output": "2+2 等于 4。"},
    ]
    p = tmp_path / "sft.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p
