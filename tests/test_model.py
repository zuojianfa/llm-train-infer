"""模型层单元测试:RMSNorm / RoPE / GQA Attention / KV Cache 一致性 / 形状与因果性。

其中最关键的是 test_kv_cache_matches_full_forward:
    "增量解码"与"整段重算"必须给出逐位一致的 logits —— 这条性质是
    所有推理引擎(vLLM/TensorRT-LLM)正确性的第一道防线,一旦 KV Cache
    的位置切片或掩码构造写错(本工程曾真实踩过 start_pos Bug),
    这个测试会立刻失败。
"""

from __future__ import annotations

import math

import pytest
import torch

from minillm.config import ModelConfig
from minillm.model import LLMModel, RMSNorm, apply_rotary_emb, precompute_freqs_cis


@pytest.fixture(scope="module")
def model() -> LLMModel:
    cfg = ModelConfig(vocab_size=300, dim=64, num_layers=2, num_heads=4,
                      num_kv_heads=2, hidden_dim=128, max_seq_len=128)
    torch.manual_seed(0)
    m = LLMModel(cfg).eval()
    return m


# ------------------------------------------------------------------ 基本形状
def test_forward_shape_and_logits_dtype(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 7))
    logits = model(x)
    assert logits.shape == (2, 7, model.cfg.vocab_size)
    assert logits.dtype == torch.float32          # forward 末尾强制 fp32
    # logit soft-cap:|z| < 50(Gemma 风格 tanh 压缩)
    assert logits.abs().max().item() < 50.0 + 1e-3


def test_gradient_flows_to_all_params(model):
    x = torch.randint(0, model.cfg.vocab_size, (1, 5))
    loss = model(x).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad: {name}"


def test_weight_tying(model):
    assert model.output.weight is model.tok_embeddings.weight


def test_param_count_positive(model):
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


# ------------------------------------------------------------------ RMSNorm
def test_rmsnorm_unit_rms():
    """归一化后每个 token 向量的均方根应≈1(weight=1 时)。"""
    norm = RMSNorm(32)
    x = torch.randn(4, 32) * 7.0                  # 任意尺度
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ------------------------------------------------------------------ RoPE
def test_rope_relative_position_property():
    """RoPE 核心性质:q·k 内积只依赖相对位移(m-n),不依赖绝对位置。"""
    hd = 16
    cos, sin = precompute_freqs_cis(hd, 64, 10000.0)
    freqs = (cos, sin)
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, hd)                  # (B,H,S,hd)
    k = torch.randn(1, 2, 1, hd)
    def dot(shift_q: int, shift_k: int) -> float:
        cq = torch.zeros(1, 2, 1, hd); cq[:, :, 0] = q[0, 0, 0]
        ck = torch.zeros(1, 2, 1, hd); ck[:, :, 0] = k[0, 0, 0]
        fq = (freqs[0][shift_q:shift_q+1], freqs[1][shift_q:shift_q+1])
        fk = (freqs[0][shift_k:shift_k+1], freqs[1][shift_k:shift_k+1])
        rq = apply_rotary_emb(cq, fq)
        rk = apply_rotary_emb(ck, fk)
        return float((rq[0, 0, 0] * rk[0, 0, 0]).sum())
    # (m=10,n=13) 与 (m=30,n=33) 相对距离都是 3,内积应相同
    assert abs(dot(10, 13) - dot(30, 33)) < 1e-4
    # 相对距离不同则内积不同(否则位置编码形同虚设)
    assert abs(dot(10, 13) - dot(10, 14)) > 1e-6


def test_rope_preserves_norm():
    """旋转是正交变换:向量模长不变。"""
    hd = 16
    cos, sin = precompute_freqs_cis(hd, 32, 10000.0)
    x = torch.randn(1, 2, 1, hd)
    y = apply_rotary_emb(x, (cos[:1], sin[:1]))
    assert torch.allclose(x.norm(), y.norm(), atol=1e-5)


# ------------------------------------------------------------------ 因果性
def test_causal_mask_no_future_leak(model):
    """改最后一个 token 之后的世界不该影响之前位置的输出?反过来验证:
    把前缀延长一位,原前缀各位置的 logits 必须保持不变(因果性定义)。"""
    tok = torch.randint(0, model.cfg.vocab_size, (1, 6))
    out_full = model(tok)
    out_prefix = model(tok[:, :5])
    assert torch.allclose(out_full[:, :5], out_prefix, atol=1e-5)


# ------------------------------------------------------------------ KV Cache
@torch.no_grad()
def test_kv_cache_matches_full_forward(model):
    """逐步增量生成(prefill+decode with cache)== 一次性全量前向。"""
    B, S = 1, 9
    x = torch.randint(0, model.cfg.vocab_size, (B, S))
    full = model(x)                                # (B,S,V) 训练/整段路径

    # 先 prefill 前 4 个 token,再逐 token 增量喂入剩余 5 个
    caches = model.init_cache(B, torch.device("cpu"), torch.float32)
    logits_first = model(x[:, :4], start_pos=0, caches=caches)
    if isinstance(logits_first, tuple):
        logits_first, caches = logits_first
    collected = [logits_first]
    pos = 4
    for i in range(4, S):
        out = model(x[:, i:i+1], start_pos=pos, caches=caches)
        logits_step, caches = out
        collected.append(logits_step)
        pos += 1
    inc = torch.cat([c if c.shape[1] == 1 else c[:, -1:] for c in
                     [collected[0]] + [t for t in collected[1:]]], dim=1)
    # 对齐比较:增量路径第 j 段的最后一行 == 全量路径对应位置
    assert inc.shape[1] == S
    err = (inc - full).abs().max().item()
    assert err < 1e-4, f"KV cache diverged from full forward: max err {err}"


@torch.no_grad()
def test_start_pos_shifts_rope(model):
    """同一 token 在不同 start_pos 下 logits 必须不同(位置真的进了 RoPE)。"""
    x = torch.randint(0, model.cfg.vocab_size, (1, 5))
    caches_a = model.init_cache(1, torch.device("cpu"), torch.float32)
    _, caches_a = model(x, start_pos=0, caches=caches_a)
    nxt = torch.randint(0, model.cfg.vocab_size, (1, 1))
    la, _ = model(nxt, start_pos=5, caches=model.init_cache(1, torch.device("cpu"), torch.float32))
    # 用两个独立空缓存、不同 start_pos 对比:位置窗口不同 -> 输出不同
    lb, _ = model(nxt, start_pos=9, caches=model.init_cache(1, torch.device("cpu"), torch.float32))
    assert not torch.allclose(la, lb, atol=1e-6)


def test_gqa_head_shapes(model):
    """GQA:KV 头数少于 Q 头数,缓存张量的头维必须是 num_kv_heads。"""
    caches = model.init_cache(2, torch.device("cpu"), torch.float32)
    k, v = caches[0]
    assert k.shape == (2, model.cfg.num_kv_heads, 0, model.cfg.head_dim)
    assert v.shape == k.shape


def test_max_seq_len_boundary(model):
    with pytest.raises(Exception):
        model(torch.randint(0, 10, (1, model.cfg.max_seq_len + 1)))
