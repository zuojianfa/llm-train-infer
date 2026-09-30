# 教程 03：预训练 —— next-token prediction

对应代码：`minillm/data.py`、`minillm/train.py`，入口 `scripts/train.py`。

## 1. 训练目标就一句话
给定前文，预测下一个 token，交叉熵损失：

```
loss = CE(logits[:, :-1], input_ids[:, 1:])   # target = input 右移一位
```

`data.py` 把整本语料 tokenize 后用 `<eos>` 分隔拼接成一条长河，再切成固定 seq_len 的块；
不变量 `y == x 右移` 由 `tests/test_data.py` 锁定。

## 2. 优化器与调度
- **AdamW**：自适应动量 + 解耦权重衰减；embedding 和 norm 参数不做 weight decay（分组见 `train.py::build_optimizer`）。
- **lr 调度**：warmup 线性升温（防初期梯度噪声带偏）+ cosine 衰减（后期精修）。
- **梯度累积**：显存不够时用 accum_steps 个小 batch 模拟大 batch，攒满才 step 一次。
- **bf16 autocast**：半精度省显存提速，loss 用 fp32 计算保数值稳定。

## 3. checkpoint 与断点续训
保存 model/optim/RNG/step；恢复后继续训练与不间断训练**参数完全一致**
（等价性测试：`tests/test_train.py`）。这是长跑训练的保险丝。

## 4. 冒烟命令（CPU 低内存可跑通）
```bash
python scripts/train.py --hidden 256 --layers 4 --heads 4 --kv-heads 2 \
    --steps 300 --batch-size 8 --seq-len 256 --lr 3e-4 --out-dir out/smoke
python scripts/eval.py --ckpt out/smoke/best      # 验证集 loss 与 PPL
```
参考结果：loss 5.83 → 4.31，val PPL ≈ 74（小模型小语料的正常水平）。
