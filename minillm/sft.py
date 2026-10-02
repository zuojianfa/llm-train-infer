"""监督微调(SFT):在预训练基座上,用 (指令, 回答) 配对数据做有监督学习。

====================================================================
一、SFT 与预训练到底差在哪?(理解本模块的关键)
====================================================================
预训练(minillm/train.py)的目标是 next-token prediction:
    语料 = 一整条 token 流 -> 随机截窗口 -> 每个位置的 token 都要预测。
SFT(Supervised Fine-Tuning)的模型结构、损失函数**完全一样**,
变的只有两点:
  1. 数据组织方式:从"连续文本流"变成 "(prompt, response) 配对列表",
     每条样本是一段人工写的指令-回答,而不是自然爬取的文档;
  2. 损失掩码(loss mask):只对 response 部分的 token 计算 loss,
     prompt(用户输入)部分被屏蔽(label = -100)。

为什么要屏蔽 prompt?SFT 数据里的"问题"是人写的,不属于模型要学会
生成的分布;模型要学的是"看到问题之后如何接出好答案"。如果对 prompt
也算 loss,模型会浪费容量去模仿问句写法,稀释真正重要的答案信号。
业界所有 SFT 实现(LLaMA-Factory / Axolotl / TRL)都这么做,只是叫法
不同(prompt masking / loss on completion only)。

====================================================================
二、Chat Template:为什么要把对话拼成一个字符串?
====================================================================
Transformer 只吃 token 序列,不吃"消息列表",所以必须先把
    [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
按固定模板拼成一段纯文本。本工程采用极简风格(基座模型没有专门的
角色控制 token,用普通文本行作为角色边界):

    用户:{question}
    助手:{answer}<eos>

编码一条样本时,分别编码 "prompt 前缀" 和 "answer + <eos>",
拼接成 input_ids;再把前 len(prompt_ids) 个 label 置为 -100(忽略),
就是全部的秘密。PyTorch 的 F.cross_entropy(ignore_index=-100) 与
HuggingFace 约定一致(-100 是事实标准,源于 sklearn)。

====================================================================
三、训练策略上的差异(超参层面)
====================================================================
* 学习率:通常是预训练的 1/5 ~ 1/10(3e-4 -> 3e-5~6e-5)。SFT 数据量小
  (几百~几万条),大 lr 会把预训练学到的通用知识"冲掉"(灾难性遗忘);
* 轮数:1~3 个 epoch 即可,多了立刻过拟合背诵答案;
* warmup 比例短、cosine 衰减照旧;
* 权重从预训练 checkpoint 热启动(load_model),而不是随机初始化。

本模块提供:
  format_example   -- 单条样本 -> (prompt_str, target_str) 文本
  SFTDataset       -- JSONL 数据集,产出带 -100 掩码的 (x, y) batch
  run_sft          -- 完整训练循环(热启动 + loss 掩码 + checkpoint)
配套 CLI:scripts/train_sft.py;数据准备:scripts/prepare_sft_data.py。
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .evaluate import evaluate_loss
from .model import LLMModel, count_params
from .tokenizer import BPETokenizer
from .train import build_optimizer, get_device, get_dtype, get_lr, load_model, save_checkpoint

# ---------------------------------------------------------------- 常量约定
IGNORE_INDEX = -100        # PyTorch/HF 交叉熵的 ignore_index 惯例值


def format_example(example: dict) -> tuple[str, str]:
    """把一条 {"instruction": "...", "output": "..."} 样本拼成 (prompt, target)。

    prompt 是"模型看到的上下文"(不参与 loss),target 是"模型要学会续写的内容"
    (参与 loss)。注意 prompt 末尾自带换行——它既是角色边界的自然延续,
    也保证解码时助手回答从新行开始,与推理服务拼 prompt 的方式一致。
    """
    instruction = example["instruction"].strip()
    output = example["output"].strip()
    prompt = f"用户:{instruction}\n助手:"
    target = f"{output}\n"
    return prompt, target


class SFTDataset:
    """指令-回答配对数据集:预先把每条样本编码成 (input_ids, labels) 对。

    与预训练 TextDataset 的本质区别:
      * TextDataset 是一条无限长的 token 流上随机截窗(无配对、全位置算 loss);
      * SFTDataset 是有限条样本,每条 = prompt_ids + answer_ids,
        其中 prompt 段的 label 全部是 IGNORE_INDEX(-100),只有 answer 段
        保留真实 token id 作为标签。

    预编码(list 存内存)对几千条的小指令集完全够用;大规模 SFT(几十万条)
    应改为惰性编码或离线 tokenize 成 .npy,思路同 nanoGPT->LLaMA-Factory 的演进。
    """

    def __init__(self, path: str, tokenizer: BPETokenizer):
        self.tok = tokenizer
        self.samples: list[tuple[list[int], list[int]]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)                      # JSONL:一行一个样本
                prompt, target = format_example(ex)
                p_ids = tokenizer.encode(prompt)           # 条件部分:不算 loss
                t_ids = tokenizer.encode(target) + [tokenizer.EOS_ID]  # 目标 + <eos>
                # labels 与 input_ids 等长:prompt 段填 -100,answer 段填真实 id。
                # 训练时 x = ids[:-1], y = labels[1:],右移一位后第 i 个 label
                # 仍是"位置 i+1 的真实 token",与预训练的 next-token 目标对齐。
                labels = [IGNORE_INDEX] * len(p_ids) + list(t_ids)
                self.samples.append((p_ids + t_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def random_batch(self, batch_size: int, seq_len: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """随机抽 batch_size 条样本,pad 到统一长度后返回 (x, y)。

        padding 细节(新手最容易踩坑的地方):
          * input_ids 用 0(<eos>)补齐 —— 补什么其实无所谓,因为对应 label 是 -100;
          * labels 尾部补 IGNORE_INDEX —— 保证 pad 位置不产生任何梯度;
          * 截断:超过 seq_len 的样本直接砍掉尾部(教学数据极短,不会触发;
            生产做法是丢弃或分块,绝不能静默截掉 answer 结尾的 <eos>,
            否则模型学不会"何时停止生成")。
        """
        idxs = np.random.randint(0, len(self.samples), size=batch_size)
        xs, ys = [], []
        for i in idxs:
            ids, labels = self.samples[i]
            if len(ids) > seq_len + 1:                 # 预留 1 个右移位
                ids, labels = ids[:seq_len + 1], labels[:seq_len + 1]
            xs.append(ids)
            ys.append(labels)
        max_len = max(len(t) for t in xs)
        x = torch.full((batch_size, max_len), 0, dtype=torch.long)
        y = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        for b, (ids, labels) in enumerate(zip(xs, ys)):
            # x[:, :-1] 喂模型,y[:, 1:] 当标签 => 这里存的是"对齐前的原始序列"
            x[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[b, :len(labels)] = torch.tensor(labels, dtype=torch.long)
        return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_sft_loss(model: torch.nn.Module, dataset: SFTDataset, cfg,
                      device: torch.device, dtype: torch.dtype) -> float:
    """SFT 验证 loss:遍历整个验证集(样本少,不必像预训练那样随机采样)。

    与 evaluate_loss(预训练版)唯一的区别是 cross_entropy 传 ignore_index,
    让 prompt/pad 位置不计入平均——这样 train/val 数字口径一致,可直接比较。
    """
    model.eval()
    losses, n_tok = [], 0
    for ids, labels in [(s[0], s[1]) for s in dataset.samples]:
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(device.type in ("cuda", "mps", "xpu") or dtype != torch.float32)):
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               y.view(-1), ignore_index=IGNORE_INDEX, reduction="sum")
        cnt = int((y != IGNORE_INDEX).sum())
        losses.append(loss.item())
        n_tok += cnt
    model.train()
    return sum(losses) / max(1, n_tok)         # 按"有效监督 token"归一


def run_sft(cfg: TrainConfig, init_ckpt: str, data_path: str,
            val_path: str | None = None, resume: bool = False) -> None:
    """SFT 主循环。结构与 train.train() 平行,方便对照学习两者差异。

    流程:种子 -> 加载分词器 -> **热启动预训练权重** -> 构造配对数据集
          -> 步循环(采样 batch -> 前向 -> ignore_index 交叉熵 -> 反向)。
    与预训练循环相比刻意简化了 loss 尖峰回退(SFT 数据干净、几乎不炸),
    保留了梯度累积/checkpoint/最优保存等同样重要的工程件。
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device(cfg.device)
    dtype = get_dtype(cfg.dtype)
    print(f"[sft] device={device} dtype={dtype}")

    tok = BPETokenizer.load(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size      # 同预训练:embedding 行数以实际词表为准

    # ---- 热启动:load_model 读 meta.json 重建同构模型再灌预训练权重。
    # 这是"SFT"区别于"从头训指令模型"的根本:站在基座的通用语言分布上,
    # 只用少量数据修正输出行为(cf. Ouyang et al. 2022, InstructGPT §3.2)。
    model, pre_step = load_model(init_ckpt, device, dtype)
    print(f"[sft] hot-start from {init_ckpt} (pretrain step={pre_step}), "
          f"params={count_params(model)/1e6:.2f}M")

    train_ds = SFTDataset(data_path, tok)
    val_ds = SFTDataset(val_path, tok) if val_path and os.path.exists(val_path) else None
    print(f"[sft] data: {len(train_ds)} train samples"
          + (f", {len(val_ds)} val samples" if val_ds else ""))

    optimizer = build_optimizer(model, cfg)
    ckpt_dir = os.path.join(cfg.out_dir, "ckpt")

    # epoch 制采样:SFT 数据有限,"跑完一遍数据"才是自然的进度单位
    # (预训练的 token 流是无限的,所以那边用 max_steps)。
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (cfg.batch_size * cfg.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * cfg.max_steps      # 此处 max_steps 复用为 epoch 数
    cfg.max_steps = total_steps                        # get_lr 的 cosine 调度需要总步数
    print(f"[sft] epochs={steps_per_epoch and total_steps // steps_per_epoch} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    running_ema = None
    best_val = float("inf")
    t0 = time.time()
    step = 0
    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = train_ds.random_batch(cfg.batch_size, cfg.seq_len, device)
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(device.type in ("cuda", "mps", "xpu") or dtype != torch.float32)):
                logits = model(x[:, :-1])              # 去掉最后一个 token 作为输入
                # 关键一行:ignore_index=-100 => prompt 段与 pad 段零贡献。
                # .float():bf16 logits 直接进 CE 精度不足(同预训练的理由)。
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    y[:, 1:].contiguous().view(-1),
                    ignore_index=IGNORE_INDEX)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running_ema = accum_loss if running_ema is None else 0.99 * running_ema + 0.01 * accum_loss
        step += 1

        if step % cfg.log_interval == 0 or step == cfg.max_steps:
            print(f"[sft] step {step}/{cfg.max_steps} (epoch {step // steps_per_epoch}) "
                  f"| loss {accum_loss:.4f} (ema {running_ema:.4f}) | lr {lr:.2e} "
                  f"| grad {grad_norm:.2f}")

        if val_ds is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            vl = evaluate_sft_loss(model, val_ds, cfg, device, dtype)
            print(f"[sft ] step {step} | val loss(answer-only) {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}")
            if vl < best_val:
                best_val = vl
                save_checkpoint(model, os.path.join(cfg.out_dir, "best"), step)

        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_checkpoint(model, ckpt_dir, step, optimizer, is_latest=True)
            print(f"[sft] saved checkpoint @ step {step} -> {ckpt_dir}")

    save_checkpoint(model, ckpt_dir, step, optimizer, is_latest=True)
    print(f"[sft] done in {time.time()-t0:.1f}s, final ckpt @ {ckpt_dir}")
