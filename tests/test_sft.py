"""SFT 监督微调单元测试:模板拼接、loss 掩码位置、padding、端到端一步训练。

test_labels_mask_prompt_prefix 是 SFT 正确性的核心断言:
    labels 的前 len(prompt_ids) 位必须全是 -100,其余必须是真实 token id,
    且 labels[i] == input_ids[i](同位置自监督,右移在 batch 层做)。
掩码错位(差一个 token)会让模型"背问题、不答问题",肉眼难查,全靠测试兜底。
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from minillm.config import ModelConfig, TrainConfig
from minillm.model import LLMModel
from minillm.sft import IGNORE_INDEX, SFTDataset, evaluate_sft_loss, format_example, run_sft
from minillm.tokenizer import BPETokenizer


# ------------------------------------------------------------------ 模板
def test_format_example_template():
    prompt, target = format_example({"instruction": "你好吗?", "output": "我很好。"})
    assert prompt == "用户:你好吗?\n助手:"          # 与 serve.py chatml 模板一致
    assert target == "我很好。\n"


def test_chat_template_matches_serve():
    """训练模板(sft.format_example)与推理模板(serve.format_prompt chatml)必须逐字符一致。"""
    from minillm.serve import ChatMessage, format_prompt
    q = "什么是语言模型?"
    train_prompt, _ = format_example({"instruction": q, "output": "x"})
    infer_prompt = format_prompt([ChatMessage(role="user", content=q)], chatml=True)
    assert train_prompt == infer_prompt


# ------------------------------------------------------------------ 数据集
def test_labels_mask_prompt_prefix(tiny_tokenizer, sft_jsonl):
    ds = SFTDataset(str(sft_jsonl), tiny_tokenizer)
    assert len(ds) == 2
    for ids, labels in ds.samples:
        assert len(ids) == len(labels)             # 等长,一一对应
        prompt, _ = format_example(json.loads(open(sft_jsonl, encoding="utf-8").readlines()[ds.samples.index((ids, labels))]))
        p_len = len(tiny_tokenizer.encode(prompt))
        # prompt 段全部屏蔽;answer 段(含末尾 <eos>)保留真实标签
        assert all(l == IGNORE_INDEX for l in labels[:p_len]), "prompt 未被完全屏蔽"
        assert all(l != IGNORE_INDEX for l in labels[p_len:]), "answer 段被误屏蔽"
        assert labels[-1] == tiny_tokenizer.EOS_ID, "样本必须以 <eos> 收尾"
        # 同位置对齐:x=ids[:-1], y=labels[1:] 后 y[i] 就是 ids[i+1]
        assert all(labels[i] == ids[i] for i in range(p_len, len(ids)))


def test_batch_padding_uses_ignore_index(tiny_tokenizer, sft_jsonl):
    ds = SFTDataset(str(sft_jsonl), tiny_tokenizer)
    torch.manual_seed(0)
    x, y = ds.random_batch(batch_size=4, seq_len=64, device=torch.device("cpu"))
    assert x.shape == y.shape and x.shape[0] == 4
    # 短样本尾部 pad:label 必为 IGNORE,input 补 0(<eos>)
    lengths = [len(s[0]) for s in ds.samples]
    max_len = x.shape[1]
    if max_len > max(lengths):                     # 存在 pad 时检查 pad 区
        row = lengths.index(min(lengths))
        pad_start = min(lengths)
        assert (y[row, pad_start:] == IGNORE_INDEX).all()


def test_cross_entropy_with_ignore_index_math(tiny_tokenizer, sft_jsonl):
    """手工验证 ignore_index 语义:全 -100 的 label 给 CE 返回 0/nan 均不应崩。"""
    logits = torch.randn(1, 5, 30)
    y_all_ignore = torch.full((1, 5), IGNORE_INDEX)
    loss = F.cross_entropy(logits.view(-1, 30), y_all_ignore.view(-1),
                           ignore_index=IGNORE_INDEX)
    assert not torch.isfinite(loss) or loss.item() == 0.0   # PyTorch 对空有效集返回 0


# ------------------------------------------------------------------ 端到端
def test_run_sft_one_step_and_checkpoint(tiny_tokenizer, sft_jsonl, tmp_path):
    """完整跑一遍 run_sft(微型配置):热启动 -> 更新 -> best/ckpt 落盘 -> 可加载生成。"""
    cfg_model = ModelConfig(vocab_size=tiny_tokenizer.vocab_size, dim=32, num_layers=1,
                             num_heads=2, num_kv_heads=1, hidden_dim=64, max_seq_len=128)
    init_dir = str(tmp_path / "pre")
    torch.manual_seed(0)
    base = LLMModel(cfg_model)
    from minillm.train import save_checkpoint
    save_checkpoint(base, init_dir, step=0)        # 伪造一个"预训练"基座

    tcfg = TrainConfig(tokenizer_path=None, device="cpu", dtype="float32",
                       out_dir=str(tmp_path / "sft"), lr=1e-3, warmup_steps=1,
                       max_steps=2,                 # = 2 epochs
                       batch_size=2, seq_len=64, gradient_accumulation_steps=1,
                       log_interval=1, eval_interval=1, save_interval=1, seed=0)
    tcfg.model = cfg_model
    tok_path = str(tmp_path / "tok.json")
    tiny_tokenizer.save(tok_path)
    tcfg.tokenizer_path = tok_path

    before = {n: p.detach().clone() for n, p in base.named_parameters()}
    run_sft(tcfg, init_ckpt=init_dir, data_path=str(sft_jsonl), val_path=str(sft_jsonl))

    # checkpoint 产出且能被推理侧 load_model 读回
    from minillm.train import load_model
    model, step = load_model(str(tmp_path / "sft" / "ckpt"), torch.device("cpu"), torch.float32)
    assert step >= 1
    changed = sum(1 for n, p in model.named_parameters()
                  if not torch.equal(before[n], p.detach()))
    assert changed > 0, "SFT 后权重没有任何变化 => 训练循环没生效"

    # 验证 loss 函数口径正常(有限值)
    ds = SFTDataset(str(sft_jsonl), tiny_tokenizer)
    vl = evaluate_sft_loss(model, ds, tcfg, torch.device("cpu"), torch.float32)
    assert vl == vl and vl < 50.0                  # 非 nan、量级合理
