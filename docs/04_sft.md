# 教程 04：SFT 监督微调 —— 让模型学会"对话"

对应代码：`minillm/sft.py`、`scripts/train_sft.py`、`scripts/prepare_sft_data.py`。

## 1. SFT 与预训练的区别

同一个模型、同一种 next-token 损失，只有两点不同：

1. **数据格式**：无标注原始文本 → `(instruction, response)` 配对（JSONL，每行一条）；
2. **损失掩码**：prompt 部分（用户说的话）标签置 `-100`（PyTorch CrossEntropyLoss 的忽略位），
   模型只对 response 计算 loss——学的是"如何回答"，而不是"背诵问题"。

## 2. chat template

训练与服务必须使用**逐字符一致**的模板，否则推理时分布漂移。本工程采用 ChatML 风格：

```text
{每个 turn: <|im_start|>角色\n内容<|im_end|>\n}{末尾追加 <|im_start|>assistant\n 作为生成引导}
```

- `sft.py::format_example`（训练侧）与 `serve.py::format_prompt(chatml=True)`（服务侧）
  输出完全相同，由 `tests/test_sft.py` 断言锁定；
- 标签构造：prompt 段全部 `-100`，response 段为右移一位的 token id——这就是"只学回答"。

## 3. 训练策略差异

| 项 | 预训练 | SFT |
|---|---|---|
| 学习率 | 3e-4 量级 | **6e-5 量级（约 1/5）**，防灾难遗忘 |
| 采样单位 | step 数 | epoch 数（指令集通常很小） |
| 起点 | 随机初始化 | **热启动**：`--init-ckpt out/smoke/best` |
| padding | 无 | batch 内右侧 pad，pad 位同样置 -100 |

## 4. 运行

```bash
python scripts/prepare_sft_data.py                     # 生成 data/sft_train.jsonl / sft_val.jsonl
python scripts/train_sft.py --init-ckpt out/smoke/best \
    --out-dir out/sft --epochs 2 --lr 6e-5
```

评估用 `sft.py::evaluate_sft_loss`：loss 按**有效监督 token 数**归一（忽略 -100 位），
不同长度样本混 batch 时统计才公平。测试覆盖见 `tests/test_sft.py`（掩码位置、模板一致性、端到端更新）。
