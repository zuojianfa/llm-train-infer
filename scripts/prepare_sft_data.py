"""生成一份内置的中文指令-回答 SFT 演示数据集(JSONL),完全离线。

真实场景下你应从 Alpaca-zh / BELLE / Firefly 等开源指令集转换;
本脚本内置几十条手写样例,保证教学流程"clone 即可跑",不依赖网络。

数据格式约定(minillm/sft.py 的 SFTDataset 读取):
    data/sft_train.jsonl   每行一个 JSON:{"instruction": "...", "output": "..."}
    data/sft_val.jsonl     同格式,用于监控 SFT 是否过拟合

用法:
    python scripts/prepare_sft_data.py            # 写到 data/ 目录
    python scripts/prepare_sft_data.py --seed 7   # 打乱时换随机种子
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 内置演示指令集:覆盖问答/续写/翻译/常识几类最常见的 SFT 任务形态。
# 内容刻意与预训练语料(wiki 风格陈述句)不同,让模型学到"被提问 -> 作答"的行为切换。
EXAMPLES = [
    {"instruction": "用一句话解释什么是语言模型。",
     "output": "语言模型是一个根据已有文字预测下一个词概率的程序,连起来就能生成连贯文本。"},
    {"instruction": "把下面句子翻译成英文:大海在阳光的照耀下闪烁着金色的光芒。",
     "output": "The sea sparkles with golden light under the shine of the sun."},
    {"instruction": "请写一句关于秋天的描述。",
     "output": "北京的秋天天空高远,银杏叶铺满街道,像一层薄薄的金子。"},
    {"instruction": "什么是注意力机制?",
     "output": "注意力机制让模型在处理每个词时,可以按相关程度加权查看句中其他词,从而捕捉长距离依赖。"},
    {"instruction": "请列举三种常见的编程语言。",
     "output": "常见的编程语言有 Python、Java 和 C++。"},
    {"instruction": "补全句子:书是人类进步的",
     "output": "书是人类进步的阶梯。"},
    {"instruction": "为什么天空是蓝色的?",
     "output": "因为阳光进入大气层后,波长较短的蓝光比红光更容易被空气分子散射,四面八方都充满散射的蓝光,所以天空看起来是蓝色的。"},
    {"instruction": "请把这句话改写得更礼貌:把门关上。",
     "output": "麻烦您帮忙把门关上,谢谢。"},
    {"instruction": "简单介绍一下 Transformer。",
     "output": "Transformer 是一种完全基于注意力结构的神经网络,抛弃了循环单元,能够并行处理整个序列,是当前大语言模型的基础架构。"},
    {"instruction": "12 乘以 8 等于多少?",
     "output": "12 乘以 8 等于 96。"},
    {"instruction": "写一首关于月亮的两句小诗。",
     "output": "月光落在窗台上,像一封没有寄出的信。"},
    {"instruction": "什么是梯度下降?",
     "output": "梯度下降是沿着损失函数下降最快的方向逐步调整参数的优化方法,可以想象成蒙着眼睛沿山坡一步步走向谷底。"},
    {"instruction": "请用中文回答:apple 是什么意思?",
     "output": "apple 的意思是苹果。"},
    {"instruction": "给初学者一条学习编程的建议。",
     "output": "先选一门语言把基础语法过一遍,然后立刻动手写小项目,在报错中学习是最快的路径。"},
    {"instruction": "解释一下 KV Cache 的作用。",
     "output": "KV Cache 在生成每个新词时缓存历史 token 的键和值,避免重复计算整个前缀,把自回归生成的复杂度从平方级降到线性级。"},
    {"instruction": "把数字 2026 用汉字写出来。",
     "output": "二千零二十六,通常读作二零二六。"},
    {"instruction": "请说一个关于时间的比喻。",
     "output": "时间像一条单行道,车辆川流不息,却永远不能掉头。"},
    {"instruction": "水在标准大气压下多少度沸腾?",
     "output": "水在标准大气压下的沸点是 100 摄氏度。"},
]


def main():
    ap = argparse.ArgumentParser(description="build demo SFT dataset (JSONL)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = EXAMPLES[:]
    rng.shuffle(data)                       # 固定种子打乱 => train/val 划分可复现
    n_val = max(1, int(len(data) * args.val_ratio))
    val, train = data[:n_val], data[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for row in rows:
                # ensure_ascii=False:中文原样写出,人可以直接打开检查数据质量
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sft-data] {p}: {len(rows)} samples")


if __name__ == "__main__":
    main()
