"""验证集困惑度(perplexity)评估。

困惑度 ppl = exp(平均交叉熵 loss),直觉解释:模型在"平均多少选一中"地
猜下一个 token。ppl=1 表示完美预测;ppl=20 表示平均在 20 个候选里犹豫。
预训练阶段它是最可靠的单一指标(loss 越低,自回归概率分配越好)。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import TextDataset


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, dataset: TextDataset, cfg,
                  device: torch.device, dtype: torch.dtype) -> float:
    """在固定随机 batch 上平均交叉熵(近似验证 loss)。

    为什么用"随机窗口取 eval_iters 次"而不是遍历整个验证集:验证语料可能
    很大且被拼成一条长流,遍历代价高;随机窗口的方差随 iters 增大而减小,
    对趋势判断足够(与 nanoGPT 的 estimate_loss 同一思路)。
    注意 model.eval()/model.train() 成对出现——dropout 等训练专属行为必须
    在评估时关闭、评估后恢复,否则日志 loss 与训练 loss 不可比。
    """
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = dataset.random_batch(cfg.batch_size, cfg.seq_len, device)
        # 与训练完全一致的 autocast + fp32 交叉熵,保证 train/val loss 同口径可比
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(device.type in ("cuda", "mps") or dtype != torch.float32)):
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def perplexity(loss: float) -> float:
    """loss -> ppl。min(loss,20) 防止未训好的模型 exp 溢出为 inf。"""
    return math.exp(min(loss, 20.0))
