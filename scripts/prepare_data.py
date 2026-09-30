"""下载预训练语料 -> data/train.txt / data/val.txt。

优先尝试 HuggingFace datasets(需网络);失败则回退到内置的离线样例语料,
保证全流程在任何环境下都能跑通。
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import random
from pathlib import Path

DATA_DIR = Path("data")


def try_hf_datasets(name: str, split: str, max_rows: int) -> list[str] | None:
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split=split, streaming=True)
        texts = []
        for i, row in enumerate(ds):
            t = row.get("text")
            if not t or len(t.strip()) < 40:
                continue
            texts.append(t.strip().replace("\n", " "))
            if len(texts) >= max_rows:
                break
        return texts or None
    except Exception as e:  # 无网络/库异常均回退
        print(f"[prepare] HF dataset '{name}' unavailable: {type(e).__name__}: {e}")
        return None


def offline_corpus() -> list[str]:
    """内置多主题英文+中文句子,作为离线兜底语料(重复扩充以覆盖词表)。"""
    en = [
        "The cat sat on the mat and watched the birds outside the window.",
        "Machine learning models learn patterns from data to make predictions.",
        "The transformer architecture uses self attention to process sequences.",
        "Natural language processing helps computers understand human text.",
        "A journey of a thousand miles begins with a single step.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Deep learning is a subset of machine learning based on neural networks.",
        "Training a language model requires large amounts of text data.",
        "The sun rises in the east and sets in the west every single day.",
        "Water boils at one hundred degrees Celsius at sea level pressure.",
        "She wrote a letter to her friend about their summer vacation plans.",
        "The old library stood quietly at the corner of the busy street.",
        "Programming is the art of telling a computer what exactly to do.",
        "He played the guitar by the fire while the rain fell softly outside.",
        "The stock market rose sharply after the central bank cut interest rates.",
        "Scientists discovered a new species of frog deep in the rain forest.",
        "Reading books expands your vocabulary and improves your writing skills.",
        "The train arrived late because of heavy snow across the northern region.",
        "Artificial intelligence is changing how we work and how we live.",
        "The child asked endless questions about why the sky is blue at sunset.",
        "Coffee tastes better when it is freshly ground and properly brewed.",
        "The company announced a new product line at the technology conference.",
        "Exercise regularly and sleep well to keep your body and mind healthy.",
        "Photosynthesis converts sunlight into chemical energy inside plant cells.",
        "The museum displayed ancient paintings from the seventeenth century.",
        "Language models predict the next word given the previous context.",
        "Attention mechanisms allow the model to focus on relevant tokens.",
        "Gradient descent updates the weights to minimize the training loss.",
        "The tokenizer splits raw text into subword units called tokens.",
        "Embedding layers map discrete tokens into dense vector representations.",
        "The rocket launched successfully and reached orbit nine minutes later.",
        "Music can express emotions that words alone cannot describe fully.",
        "Farmers grow wheat and corn in the wide fertile plains of the Midwest.",
        "The doctor advised him to drink more water and get plenty of rest.",
        "Quantum computing exploits superposition to solve certain hard problems.",
        "They walked along the beach collecting shells as the tide went out.",
        "The recipe calls for two eggs, flour, sugar, and a pinch of salt.",
        "Historical empires rose and fell along the banks of great rivers.",
        "Solar panels convert light directly into electricity using semiconductors.",
        "The teacher explained the lesson patiently until every student understood.",
    ]
    zh = [
        "今天天气很好,我们一起去公园散步吧。",
        "机器学习是人工智能的一个重要分支领域。",
        "Transformer模型通过自注意力机制处理序列数据。",
        "自然语言处理让计算机能够理解人类的文字。",
        "读书可以增长知识,也能让人心情平静。",
        "春天的时候,花园里开满了各种各样的花。",
        "他每天早上六点起床,然后跑步半个小时。",
        "这个城市的地铁线路非常发达,出行很方便。",
        "深度学习需要大量的数据和强大的计算资源。",
        "语言模型的任务是根据上文预测下一个字。",
        "大海在阳光的照耀下闪烁着金色的光芒。",
        "老师耐心地讲解了一道又一道数学题。",
        "人工智能正在深刻地改变我们的生活和工作方式。",
        "冬天下雪的时候,孩子们喜欢在院子里堆雪人。",
        "这本书讲述了一个关于勇气和友谊的故事。",
        "分词器把原始文本切分成一个个子词单元。",
        "训练损失随着训练步数的增加而逐渐下降。",
        "北京的秋天天空高远,空气清爽宜人。",
        "科学家用望远镜观察遥远的星系和行星。",
        "音乐能够表达语言难以描述的情感。",
    ]
    rng = random.Random(0)
    lines: list[str] = []
    # 组合扩充:随机拼接 2~4 个句子成段落,制造多样上下文
    pool = en + zh
    for _ in range(6000):
        k = rng.randint(2, 4)
        para = " ".join(rng.sample(en, k)) if rng.random() < 0.7 else "".join(rng.sample(zh, k))
        lines.append(para)
    # 加入原始句子本身
    lines.extend(en)
    lines.extend(zh)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=120000, help="最多下载多少条文本")
    ap.add_argument("--val-rows", type=int, default=2000)
    ap.add_argument("--offline", action="store_true", help="强制使用内置离线语料")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.txt"
    val_path = DATA_DIR / "val.txt"

    texts = None
    if not args.offline:
        for name, key in [("wiki40b", None), ("wikipedia", "20230501"), ("oscar", None),
                          ("smog", None)]:
            try:
                split = "train"
                texts = try_hf_datasets(name, split, args.max_rows)
                if texts:
                    print(f"[prepare] downloaded {len(texts)} docs from '{name}'")
                    break
            except Exception:
                continue

    if not texts:
        print("[prepare] using built-in offline corpus (network failed or --offline)")
        texts = offline_corpus()

    rng = random.Random(42)
    rng.shuffle(texts)
    val, train = texts[: args.val_rows], texts[args.val_rows:] or texts
    if not train:
        train, val = texts, texts[: args.val_rows]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train))
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val))
    print(f"[prepare] wrote {train_path} ({len(train)} docs) and {val_path} ({len(val)} docs)")


if __name__ == "__main__":
    main()
