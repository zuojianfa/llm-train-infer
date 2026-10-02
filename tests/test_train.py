"""训练循环单元测试:LR 调度、优化器分组、checkpoint 存取与续训等价性。

最有价值的是 test_resume_equivalence:中断->保存->恢复后继续一步,
结果应与"从未中断"完全一致 —— 这验证了 optim.pt 里 Adam 动量 + RNG
状态序列化路径的正确性(断点续训最容易错的地方)。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minillm.config import ModelConfig, TrainConfig
from minillm.model import LLMModel
from minillm.train import (build_optimizer, get_device, get_dtype, get_lr,
                           load_model, save_checkpoint, try_resume)


# ------------------------------------------------------------------ LR 调度
def test_lr_warmup_linear_then_cosine():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    # warmup 段:线性从 ~0 升到 lr(step+1 约定保证第 0 步就有非零 lr)
    assert get_lr(0, cfg) == pytest.approx(0.1)
    assert get_lr(9, cfg) == pytest.approx(1.0)
    # 中段单调递减
    lrs = [get_lr(s, cfg) for s in range(10, 110)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine 段必须单调不增"
    # 末端收敛到 lr * min_lr_ratio
    assert get_lr(110, cfg) == pytest.approx(0.1)
    assert get_lr(500, cfg) == pytest.approx(0.1)   # 超出不越界继续衰减


def test_get_dtype_map():
    assert get_dtype("bfloat16") == torch.bfloat16
    assert get_dtype("float32") == torch.float32
    with pytest.raises(KeyError):
        get_dtype("int8")


def test_get_device_prefers_xpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", type("XPU", (), {"is_available": staticmethod(lambda: True)}), raising=False)
    monkeypatch.setattr(torch.backends, "mps", type("MPS", (), {"is_available": staticmethod(lambda: True)}), raising=False)
    assert get_device("auto").type == "xpu"
    assert get_device("xpu").type == "xpu"


# ------------------------------------------------------------------ 优化器分组
def test_weight_decay_grouping():
    """norm γ / embedding(一维参数或名字含 embeddings)进 no_decay 组。"""
    cfg = ModelConfig(vocab_size=50, dim=16, num_layers=1, num_heads=2,
                      num_kv_heads=1, hidden_dim=32)
    model = LLMModel(cfg)
    tcfg = TrainConfig(weight_decay=0.1)
    opt = build_optimizer(model, tcfg)
    assert len(opt.param_groups) == 2
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    n_decayed = sum(p.numel() for p in decay["params"])
    n_total = sum(p.numel() for p in model.parameters())
    assert 0 < n_decayed < n_total                 # 两组都非空且互补
    emb = model.tok_embeddings.weight
    assert any(p is emb for p in no_decay["params"]), "embedding 不该被 weight decay"


# ------------------------------------------------------------------ checkpoint
@pytest.fixture()
def tiny_pair():
    cfg = ModelConfig(vocab_size=60, dim=16, num_layers=1, num_heads=2,
                      num_kv_heads=1, hidden_dim=32)
    torch.manual_seed(42)
    m1 = LLMModel(cfg).eval()
    return cfg, m1


def test_save_load_weights_identical(tiny_pair, tmp_path):
    cfg, m1 = tiny_pair
    d = str(tmp_path / "ckpt")
    save_checkpoint(m1, d, step=7)
    m2, step = load_model(d, torch.device("cpu"), torch.float32)
    assert step == 7
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    assert torch.equal(m1(x), m2(x))               # 权重逐位一致


def test_resume_equivalence(tiny_pair, tmp_path):
    """A: 连续两步;B: 一步->存盘->读回->再一步。两者最终权重必须一致。"""
    cfg, _ = tiny_pair
    tcfg = TrainConfig(lr=1e-3, weight_decay=0.0, warmup_steps=1, max_steps=10)

    def make_step(model, opt):
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        logits = model(x)
        loss = logits.mean()          # 任意可微目标
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # ---- A:不间断跑两步
    torch.manual_seed(0); np.random.seed(0)
    mA = LLMModel(cfg)
    optA = build_optimizer(mA, tcfg)
    make_step(mA, optA)
    make_step(mA, optA)

    # ---- B:第一步后保存(含 optim/RNG),重建模型+优化器并 resume,再第二步
    torch.manual_seed(0); np.random.seed(0)
    mB = LLMModel(cfg)
    optB = build_optimizer(mB, tcfg)
    make_step(mB, optB)
    d = str(tmp_path / "resume")
    save_checkpoint(mB, d, step=1, optimizer=optB, is_latest=True)

    mC = LLMModel(cfg)                # 全新实例(权重稍后由 ckpt 覆盖)
    optC = build_optimizer(mC, tcfg)
    start = try_resume(d, mC, optC, torch.device("cpu"))
    assert start == 1
    # 关键:RNG 恢复到中断时刻 => 下一步的 randint 序列与 A 完全相同
    make_step(mC, optC)

    for (n, pa), (_, pc) in zip(mA.named_parameters(), mC.named_parameters()):
        assert torch.allclose(pa, pc, atol=1e-7), f"param mismatch: {n}"


def test_try_resume_without_optim_returns_zero(tiny_pair, tmp_path):
    cfg, m = tiny_pair
    d = str(tmp_path / "latest_only")
    save_checkpoint(m, d, step=3)                  # 不传 optimizer => 无 optim.pt
    opt = build_optimizer(m, TrainConfig())
    assert try_resume(d, m, opt, torch.device("cpu")) == 0


def test_try_resume_skips_incompatible_vocab(tmp_path):
    old_cfg = ModelConfig(vocab_size=40, dim=16, num_layers=1, num_heads=2,
                          num_kv_heads=1, hidden_dim=32)
    new_cfg = ModelConfig(vocab_size=80, dim=16, num_layers=1, num_heads=2,
                          num_kv_heads=1, hidden_dim=32)

    old_model = LLMModel(old_cfg)
    d = str(tmp_path / "incompatible")
    old_opt = build_optimizer(old_model, TrainConfig(lr=1e-3))
    save_checkpoint(old_model, d, step=3, optimizer=old_opt, is_latest=True)

    new_model = LLMModel(new_cfg)
    opt = build_optimizer(new_model, TrainConfig())
    assert try_resume(d, new_model, opt, torch.device("cpu")) == 0
