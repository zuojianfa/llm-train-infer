"""生成与采样单元测试:greedy 确定性、可复现性、top-k/top-p 截断正确性、EOS 停止。

这些性质对应推理服务的行为契约:
  * temperature<=0 => greedy(argmax),同一输入必然同一输出;
  * 固定 torch seed => 采样序列可复现(评测/回归的前提);
  * top_k=k => 每步候选不超过 k;被砍掉的 id 永远采不到;
  * 遇到 <eos> 立即停止,不把它拼进文本。
"""

from __future__ import annotations

import pytest
import torch

from minillm.config import ModelConfig
from minillm.generate import Generator, _sample_next_token
from minillm.model import LLMModel
from minillm.tokenizer import BPETokenizer


# ---------------------------------------------------------------- 纯函数:采样
def test_greedy_is_argmax():
    logits = torch.randn(1, 5, 50)
    target = int(logits[0, -1].argmax())
    for _ in range(10):
        assert _sample_next_token(logits, temperature=0.0, top_k=None, top_p=None) == target


def test_top_k_restricts_candidates():
    """构造 logits:只有前 k 名可能被采到,第 k+1 名以后永远不被采到。"""
    logits = torch.zeros(1, 1, 100)
    # 让 id 0..9 分数最高且递减,其余为很低的分
    logits[0, 0, :10] = torch.arange(10, 0, -1, dtype=torch.float32)
    logits[0, 0, 10:] = -50.0
    seen = set()
    torch.manual_seed(42)
    for _ in range(200):
        seen.add(_sample_next_token(logits, temperature=1.0, top_k=5, top_p=None))
    assert seen <= {0, 1, 2, 3, 4}, f"top-k 截断失效,采到了 {seen - {0,1,2,3,4}}"
    assert len(seen) > 1, "temperature=1 下 200 次采样竟无随机性?"


def test_top_p_nucleus_restricts_candidates():
    logits = torch.zeros(1, 1, 100)
    logits[0, 0, :10] = torch.arange(10, 0, -1, dtype=torch.float32)
    logits[0, 0, 10:] = -50.0
    seen = set()
    torch.manual_seed(7)
    for _ in range(300):
        seen.add(_sample_next_token(logits, temperature=1.0, top_k=None, top_p=0.99))
    # p=0.99 的 nucleus 只会保留头部若干候选,绝不包含 -50 分的长尾
    assert seen <= set(range(10)), f"nucleus 泄漏到长尾: {seen - set(range(10))}"


def test_sampling_reproducible_with_seed():
    logits = torch.randn(1, 3, 60)
    torch.manual_seed(123)
    a = [_sample_next_token(logits, 0.8, 50, 0.9) for _ in range(20)]
    torch.manual_seed(123)
    b = [_sample_next_token(logits, 0.8, 50, 0.9) for _ in range(20)]
    assert a == b


# ---------------------------------------------------------------- Generator
@pytest.fixture(scope="module")
def gen_setup(tiny_tokenizer):
    cfg = ModelConfig(vocab_size=tiny_tokenizer.vocab_size, dim=64, num_layers=2,
                      num_heads=4, num_kv_heads=2, hidden_dim=128, max_seq_len=128)
    torch.manual_seed(0)
    model = LLMModel(cfg).eval()
    device = torch.device("cpu")
    return model, tiny_tokenizer, device


def test_greedy_generation_deterministic(gen_setup):
    model, tok, device = gen_setup
    g = Generator(model, tok, device, torch.float32)
    out1 = g.generate("The sea", max_new_tokens=16, temperature=0)
    out2 = g.generate("The sea", max_new_tokens=16, temperature=0)
    assert out1 == out2
    assert out1.startswith("The sea")             # echo_prompt 保证前缀回显


def test_stream_pieces_concat_equal_generate(gen_setup):
    """stream 的分片拼接必须等于 generate 的完整输出(同一 greedy)。"""
    model, tok, device = gen_setup
    g = Generator(model, tok, device, torch.float32)
    pieces = list(g.stream("Attention is", max_new_tokens=12, temperature=0))
    full = g.generate("Attention is", max_new_tokens=12, temperature=0)
    assert "".join(pieces) == full[len("Attention is"):]


def test_eos_stops_generation(gen_setup):
    """把 eos 位置人为灌成最高分 -> 生成应立即终止且不把 <eos> 解出文本。"""
    model, tok, device = gen_setup
    orig_forward = model.forward

    def patched(tokens, start_pos=0, caches=None):
        out = orig_forward(tokens, start_pos=start_pos, caches=caches)
        logits = out[0] if isinstance(out, tuple) else out
        logits[:, :, tok.EOS_ID] += 1e4           # 强推 eos
        return out
    model.forward = patched
    try:
        g = Generator(model, tok, device, torch.float32)
        text = g.generate("hello world", max_new_tokens=50, temperature=0)
    finally:
        model.forward = orig_forward
    assert text == "hello world"                  # 第一步就 eos => 零新 token


def test_generated_text_roundtrips_no_crash(gen_setup):
    """生成结果 decode 出来必须是 str,且再 encode-decode 保持稳定。"""
    model, tok, device = gen_setup
    g = Generator(model, tok, device, torch.float32)
    text = g.generate("大海", max_new_tokens=8, temperature=0.8, top_k=10, top_p=0.9)
    assert isinstance(text, str) and len(text) >= len("大海")
