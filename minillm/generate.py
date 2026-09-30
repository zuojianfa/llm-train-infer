"""推理与采样:KV Cache 增量生成 + temperature / top-k / top-p。"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F

from .model import LLMModel
from .tokenizer import BPETokenizer


@torch.no_grad()
def _sample_next_token(logits: torch.Tensor, temperature: float,
                       top_k: int | None, top_p: float | None) -> int:
    """取最后一个位置的分布做采样 -> token id。支持 greedy / top-k / nucleus 采样。

    三种解码策略层层叠加(参考 Holtzman et al. 2019, The Curious Case...):
    - temperature=T:p ∝ exp(z/T)。T->0 退化为 argmax(greedy),T 越大分布越平、
      文本越发散;T<=0 时直接走 greedy 分支。
    - top-k:只保留概率最高的 k 个候选再归一化——砍掉长尾"垃圾词",
      但固定 k 在分布尖锐时保留过多、平坦时保留过少;
    - top-p(nucleus):动态保留累计概率达到 p 的最小前缀,候选数随上下文
      自适应变化,是对 top-k 缺陷的修正(Holtzman 2019)。两者常同时开。
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]            # (B,V) 只关心下一个 token
    logits = logits.float()                  # bf16 logits 精度不足,先升 fp32
    if temperature <= 0:  # greedy
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        # v[:, [-1]] 是第 k 大的值;比它小的全部置 -inf -> softmax 后概率为 0
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        # nucleus:保留累计概率达到 p 的最小前缀(至少一个 token)。
        # (cumsum - sorted_probs) 是"不含当前项"的前缀和 >= p,说明从这项起全砍掉;
        # 减自身保证排在第一位的 token 永远保留(否则 p 极小时会全被移除)
        remove = (cumsum - sorted_probs) >= top_p
        keep_val = torch.where(remove, torch.tensor(float("-inf"), device=logits.device),
                               torch.tensor(0.0, device=logits.device))
        # scatter_ 把排序空间的 0/-inf 映射回原始 vocab 空间,再加到 logits 上
        bias = torch.zeros_like(logits).scatter_(-1, sorted_idx, keep_val)
        logits = logits + bias

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class Generator:
    """带 KV Cache 的自回归生成器(prefill 一次 + 每步单 token 增量)。

    为什么必须用 KV Cache:不用缓存的话,生成第 t 个 token 要把长度 t 的
    整个序列重新前向一遍,总代价 O(S²) 次前向;缓存历史 K/V 后每步只算
    新 token 一层注意力。这正是 LLM 推理引擎(vLLM/TensorRT-LLM)围绕
    显存管理做文章的原因——KV Cache 大小 = 层数×2×seq×H_kv×hd×batch。

    用法:
        gen = Generator(model, tokenizer, device, dtype)
        text = gen.generate("prompt", max_new_tokens=128, temperature=0.8, top_p=0.9)
        for piece in gen.stream("prompt"): ...   # 流式输出
    """

    def __init__(self, model: LLMModel, tokenizer: BPETokenizer,
                 device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.model = model.eval()            # eval:关闭 dropout 等训练专属行为
        self.tok = tokenizer
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int = 128,
               temperature: float = 0.8, top_k: int | None = 50,
               top_p: float | None = 0.9, eos_id: int | None = None,
               echo_prompt: bool = False) -> Iterator[str]:
        """流式生成:逐段产出解码后的新文本片段(适合打字机效果/HTTP SSE)。"""
        assert not self.model.training, "推理前请 model.eval()"
        eos_id = self.tok.EOS_ID if eos_id is None else eos_id
        ids = self.tok.encode(prompt) or [self.tok.EOS_ID]   # 空 prompt 兜底
        generated: list[int] = []

        # ---- prefill:整个 prompt 一次前向,填充 KV cache
        # 这一步并行处理了 prompt 的所有位置,是"prefill 算力密集"的来源
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, caches = self.model(x, start_pos=0,
                                    caches=self.model.init_cache(1, self.device, self.dtype))
        pos = len(ids)                         # 下一个 token 的绝对位置

        if echo_prompt:
            yield prompt
        decoded_len = 0                        # 已解码过的 token 数(增量解码避免重复)
        for _ in range(max_new_tokens):
            nxt = _sample_next_token(logits, temperature, top_k, top_p)
            if nxt == eos_id:                  # 模型主动结束
                break
            generated.append(nxt)
            # 只解码"新增部分"再拼接成流;子词边界可能让单 token 解出空串
            # (如纯哨兵空格),所以 piece 为空时跳过 yield
            piece = self.tok.decode(generated[decoded_len:])
            decoded_len = len(generated)
            if piece:
                yield piece
            # ---- decode:单 token 增量前向(start_pos=pos,S=1,复用 caches)
            x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
            logits, caches = self.model(x, start_pos=pos, caches=caches)
            pos += 1
            if pos >= self.model.cfg.max_seq_len:   # RoPE 表/位置预算上限
                break

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = 0.9, eos_id: int | None = None) -> str:
        """非流式生成,返回完整文本(prompt + 续写)。stream 的便捷封装。"""
        pieces = list(self.stream(prompt, max_new_tokens, temperature, top_k,
                                  top_p, eos_id, echo_prompt=True))
        return "".join(pieces)
