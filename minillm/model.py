"""GPT 风格 Decoder-only Transformer,现代 LLM 标配组件:

- RMSNorm(前置归一化)
- RoPE 旋转位置编码(无学习式位置 embedding)
- Grouped Query Attention(GQA,Q 头多于 KV 头)+ KV Cache
- SwiGLU 前馈网络
- 输入/输出 embedding 权重共享
- Logit soft-capping(Gemma 风格,稳定训练)

训练用并行前向;推理自带增量 KV Cache。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm:LayerNorm 的简化版——只除以均方根,不减均值、不加 bias。

    为什么可行:Transformer 里归一化的主要作用是"控制激活值的尺度",
    减均值带来的好处很小,Llama/TII 实验证明去掉后训练质量不变,但省掉
    一次求均值 + 一次减法,更快更简单(见 Llama2 论文 §3.2)。
    公式:y = x / sqrt(mean(x^2) + eps) * weight,weight 初始化为全 1。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps                          # 防止除零的最小值
        self.weight = nn.Parameter(torch.ones(dim))  # 逐维可学习缩放 γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim),最后一维是特征维
        # pow(2).mean(-1): 逐 token 求特征平方均值 E[x^2];keepdim 便于广播
        # .add(eps).rsqrt(): 加 eps 防零后再开 reciprocal 平方根,即 1/rms
        # 先在 float32 下计算统计量再转回原 dtype:混合精度训练数值稳定的关键
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float,
                         device=None, dtype=torch.float32):
    """RoPE 的 cos/sin 预计算表,形状均为 (max_seq_len, head_dim/2)。

    RoPE 原理(旋转位置编码,论文: RoFormer Su et al. 2021):
      把 q/k 向量的相邻两维 (x1,x2) 看作二维平面上的向量,按"位置 m"旋转
      角度 m*θ_i,其中第 i 组维度的角频率 θ_i = theta^(-2i/head_dim)。
      两个位置 m、n 的 q·k 内积只依赖相对距离 n-m —— 这就是"相对位置感知"
      的来源,同时保留绝对位置信息,且可外推到比训练更长的序列。
    实现选型:返回实数 (cos, sin) 两张表而非复数张量 cis——复数在 CPU bf16
    autocast、checkpoint 序列化等场景支持不佳,实数实现对所有设备/精度都安全。
    """
    # freqs[i] = θ_i:i 从 0 到 head_dim/2-1,每 2 维共享一个频率(指数取 2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()   # m = 0,1,...,S-1
    angles = torch.outer(positions, freqs)                # (S, hd/2):角度矩阵 m*θ_i
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)  # (cos, sin)


def apply_rotary_emb(x: torch.Tensor, freqs_cis) -> torch.Tensor:
    """对 q 或 k 施加旋转位置编码。

    x: (B, H, S, hd) 注意力的 query/key;freqs_cis: (cos, sin),各 (S', hd/2)。
    核心公式:[x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos],即二维旋转矩阵。
    """
    cos, sin = freqs_cis
    B, H, S, hd = x.shape
    s_eff = min(S, cos.shape[0])   # 按 rope 表自身长度切片,兼容不同头数/形状
    # 把最后一维 hd 拆成 (hd/2, 2):偶数位是 x1,奇数位是 x2
    x_r = x.reshape(B, H, S, hd // 2, 2).float()
    x1, x2 = x_r[..., 0], x_r[..., 1]                    # 各 (B,H,S,hd/2)
    # cos/sin 广播到 (B,H,S,hd/2);用 zeros_like 填充再赋值,是为了容忍
    # 推理时 S'=1 增量 chunk 与缓存窗口 start_pos 错位的情况
    c = torch.zeros_like(x1)
    s = torch.zeros_like(x1)
    c[:, :, :s_eff] = cos[:s_eff].float()[None, None]     # (1,1,s_eff,hd/2)
    s[:, :, :s_eff] = sin[:s_eff].float()[None, None]
    # 复数乘法的实数展开:(x1+i x2)(c+i s) = (x1 c - x2 s) + i(x1 s + x2 c)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)  # (B,H,S,hd/2,2)
    return out.reshape(B, H, S, hd).to(x.dtype)


class Attention(nn.Module):
    """多头自注意力(MHA)+ 分组查询注意力(GQA)+ KV Cache。

    三个关键概念,给初学者:
    1) 自注意力(Attention Is All You Need, Vaswani 2017):
       每个 token 生成 query(我要找什么)、key(我能被什么匹配)、
       value(匹配成功后我提供的内容)。score = softmax(q·k / sqrt(hd)),
       输出 = Σ score·v。除以 sqrt(head_dim) 是因为高维随机向量内积的方差
       随维度线性增长,不缩放会让 softmax 进入饱和区、梯度消失。
    2) GQA(Llama2/MQA 折中,Ainslie 2023):Q 有 H 个头,K/V 只有 H_kv 个
       头(n_rep = H/H_kv 个 Q 头共享一组 KV)。推理时 KV Cache 显存直接
       缩小 n_rep 倍,质量几乎不掉——这是现代 LLM 的标配。
    3) KV Cache:自回归生成时,前面 token 的 K/V 只依赖输入本身,不会变;
       缓存下来,每步只需为新 token 计算一次 K/V 并拼接,把 O(S²) 重算
       降为 O(S)(详见 forward 里 cache 分支与 LLMModel.forward 的掩码)。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads            # Q 头数 H
        self.n_kv_heads = cfg.num_kv_heads      # KV 头数 H_kv(GQA)
        self.head_dim = cfg.head_dim            # 每头维度 hd = dim / H
        self.n_rep = self.n_heads // self.n_kv_heads      # GQA:每个 KV 头重复次数
        # 四个投影均为无 bias 线性层(Llama 惯例:bias 对质量无益且拖慢训练)
        # wq 输出 H*hd = dim;xk/xv 输出 H_kv*hd ≤ dim(GQA 省参数)
        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: torch.Tensor | None,
                cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """x: (B,S,dim);返回 (输出 (B,S,dim), 新 KV 缓存)。"""
        B, S, _ = x.shape
        # 1) 投影并拆头:(B,S,dim) -> (B,S,H,hd) -> transpose 成 (B,H,S,hd)
        #    SDPA 要求头维在 batch 后、序列维前,所以必须 transpose(1,2)
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2) RoPE 只作用于 q/k(位置信息体现在"谁看谁"的匹配上,v 不需要)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # 3) 推理增量模式:把历史 K/V 缓存在时间维(dim=2)拼到当前 chunk 前面
        if cache is not None:
            ck, cv = cache                       # (B,H_kv,T_prev,hd)
            xk = torch.cat([ck, xk], dim=2)      # (B,H_kv,T_prev+S,hd)
            xv = torch.cat([cv, xv], dim=2)
        new_cache = (xk, xv)                     # 交回上层保存,供下一步使用

        # 4) GQA 展开:把每组 KV 复制 n_rep 次,凑成 H 头与 Q 对齐
        #    (B,H_kv,1,T,hd) --expand--> (B,H_kv,n_rep,T,hd) --reshape--> (B,H,T,hd)
        #    expand 是零拷贝广播视图,reshape 时才真正物化,比 repeat_interleave 省内存
        if self.n_rep > 1:  # GQA -> MQA:复制 KV
            T = xk.shape[2]
            xk = xk[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T,
                                             self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
            xv = xv[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T,
                                             self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

        # 5) 融合注意力:SDPA 内部按后端(flash/mem-efficient/math)自动选最优实现
        #    is_causal=True 让 PyTorch 自己生成下三角掩码(训练全量前向最快路径);
        #    带缓存推理时必须传显式 mask(见 LLMModel.forward),此时 is_causal=False
        y = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(mask is None),
        )                                        # (B,H,S,hd)
        # 6) 合头:(B,H,S,hd) -> (B,S,H*hd) -> (B,S,dim),再过输出投影
        y = y.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(y), new_cache


class FeedForward(nn.Module):
    """SwiGLU 前馈网络:Llama 论文用实验替换了 GPT 的两层 ReLU MLP。

    结构: down( silu(gate(x)) * up(x) )
    直觉:up 分支提供"内容",gate 分支经 SiLU 后充当可学习的"门控",
    逐维相乘让网络学会"哪些特征在该 token 上放行"。因为多了一个 gate
    矩阵,为保持参数量与传统 FFN 相当,中间维取约 8/3*dim(而非 4*dim)。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三个矩阵乘法 + 一个逐元素激活/相乘,全部是 (B,S,dim)x(dim,hidden) 级别
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 层:Pre-Norm 残差结构。

    x = x + Attention(RMSNorm(x))     # 先归一化再进子层,残差主干保持"干净"
    x = x + FFN(RMSNorm(x))           # Pre-LN(Post-LN 对比见 Xiong 2020)
    Pre-Norm 让深层网络梯度直通残差主干,训练稳定,是现代 LLM 的统一选择。
    """

    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)

    def forward(self, x, freqs_cis, mask, cache=None):
        h, cache = self.attention(self.attention_norm(x), freqs_cis, mask, cache)
        x = x + h                                # 第一个残差:主干 + 注意力输出
        h = self.feed_forward(self.ffn_norm(x))
        return x + h, cache                      # 第二个残差:主干 + FFN 输出


class LLMModel(nn.Module):
    """完整 Decoder-only LLM:Embedding -> N x TransformerBlock -> Norm -> LM Head。

    参数量估算(权重共享时,近似公式):
      P ≈ V*dim(嵌入) + L*(4*dim^2 + 3*dim*hidden)(每层 attn+ffn) + 2*dim(归一化)
      默认配置 V≈8k、L=8、dim=512、hidden=1408 => ~70M 参数。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # 词嵌入:token id -> dim 维向量;输出头复用同一矩阵(weight tying),
        # 省一半 V*dim 参数且通常还涨点(GPT-2/Llama 均如此,Press & Wolf 2017)
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(i, cfg) for i in range(cfg.num_layers)
        )
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)   # 最终归一化(Pre-Norm 结构的收尾)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  # LM head:dim -> logits
        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight  # 关键:两个模块共享同一 Parameter

        # RoPE 表:注册为 buffer(随 .to(device) 自动搬运,但不进 optimizer)。
        # 用两个实数 cos/sin 而非复数张量,避免 bf16/CPU autocast 兼容性问题
        cos, sin = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)  # persistent=False:不写入 state_dict
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # 残差投影按深度缩放:Llama2/GPT-NeoX 做法。
        # 直觉:每个 block 输出是"主干 + 子层",子层方差若不缩小,经过 L 层累加后
        # 残差主干方差会线性膨胀(L 倍),深层训练不稳。把 wo/down 的初始 std
        # 除以 sqrt(2*L)(2 来自每层有两个子层),使主干方差在初始化时≈恒定。
        std = 0.02
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, module: nn.Module):
        """统一初始化:Linear/Embedding 用 N(0, 0.02),bias 置零。

        0.02 是小标准差(Karpathy nanoGPT 经验值):太大则前向激活爆炸,
        太小则梯度信号弱;Transformer 对 init scale 敏感,值得单独调。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _freqs_on(self, device) -> tuple:
        """按需把 RoPE 表搬到目标设备(model.to() 已处理,这里兜底 CPU->GPU)。"""
        cos, sin = self.rope_cos, self.rope_sin
        if cos.device != device:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin

    def forward(self, tokens: torch.Tensor,
                start_pos: int = 0,
                caches: list | None = None,
                ) -> torch.Tensor:
        """tokens: (B,S)。训练时 caches=None 全量前向;推理传增量 token 与缓存列表。

        两种模式共用同一份代码,差别只在:
          训练:start_pos=0, caches=None -> SDPA is_causal,无掩码构造开销
          推理:start_pos=已有长度, caches=各层 (K,V) -> 显式掩码 + KV 拼接
        """
        B, S = tokens.shape
        # 位置上界检查:RoPE 表只预计算了 max_seq_len 个位置,start_pos + S 超过
        # 它会让 cos/sin 切片变短而"静默"用错位置编码(输出错但不会报错)。显式
        # 抛出,让超序列使用尽早失败而非产生不可靠的 logits。
        if start_pos + S > self.cfg.max_seq_len:
            raise ValueError(
                f"序列长度 {start_pos + S} 超过 RoPE 表上限 max_seq_len={self.cfg.max_seq_len}"
            )
        h = self.tok_embeddings(tokens)             # (B,S,dim):查表,可微
        cos_all, sin_all = self._freqs_on(h.device)
        # apply_rotary_emb 内部对传入的 cos/sin 再做 [:S] 切片,
        # 因此这里必须传"绝对位置"对应的窗口,而不是从 0 开始的前缀。
        # 例:prompt 长 10,生成第 11 个 token 时 start_pos=10,S=1,
        # 应取 cos[10:11] —— 位置编码必须反映真实序列位置,GQA/KV cache 才正确。
        freqs_cis = (cos_all[start_pos:start_pos + S], sin_all[start_pos:start_pos + S])

        # 因果掩码:训练时 SDPA 的 is_causal 已足够;带缓存推理需要显式掩码
        mask = None
        if caches is not None:
            # 掩码列数必须等于注意力里真实的 K 总数(历史缓存长度 + 当前 chunk),
            # 而不是 start_pos + S。两者在"从位置 0 连续生成"时恰好相等,但一旦
            # 用空缓存从某 start_pos 跳跃续写就不同——按真实缓存长度构造才稳健。
            cached_len = caches[0][0].shape[2]
            total = cached_len + S
            # 掩码形状 (S, total):行=当前 chunk 的每个 query,列=全部 K 位置。
            # 用 float32 构造 -inf,SDPA 会广播到 q/k 的 dtype(bf16 下 -inf 同样有效)
            causal = torch.full((S, total), float("-inf"), device=h.device, dtype=torch.float32)
            # 规则:query i(绝对位置 start_pos+i)可以看见所有 j <= start_pos+i 的 key。
            # 前 cached_len 列(历史缓存)对全部 query 可见 -> 置 0;
            # 后 S 列(chunk 内部)保持下三角因果 -> triu(diagonal=1) 上三角为 -inf
            # ⚠️ 历史列必须显式置 0:只写 causal[:, cached_len:] 会让前 cached_len 列
            #    保持 -inf,增量解码就看不到缓存里的历史 K/V——这是经典的 KV cache bug。
            causal[:, :cached_len] = 0.0
            causal[:, cached_len:] = torch.triu(
                torch.full((S, S), float("-inf"), device=h.device, dtype=torch.float32),
                diagonal=1,
            )
            mask = causal[None, None]  # (1,1,S,total):广播到 (B,H,...) 的头/batch 维

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            h, c = layer(h, freqs_cis, mask, cache)
            if caches is not None:
                new_caches.append(c)     # 更新后的 KV 缓存,供下一个 token 使用

        h = self.norm(h)                       # 最后一层输出的 Pre-Norm 收尾归一化
        logits = self.output(h).float()        # (B,S,V);强制 fp32:softmax/CE 数值稳定
        # logit soft cap(Gemma 风格):z' = cap*tanh(z/cap),把 logits 平滑压到 ±cap。
        # 相比硬 clamp,tanh 处处可导;防止个别维度发散主导 softmax、破坏 bf16 训练。
        logits = 50.0 * torch.tanh(logits / 50.0)
        if caches is not None:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def init_cache(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype) -> list:
        """预分配"空"KV 缓存:每层一对 (K,V),时间维长度为 0,靠 cat 逐步增长。

        为什么不用预分配满长度定长缓冲:实现简单、无索引簿记;代价是每步
        cat 拷贝 O(T),对教学模型完全可接受。生产系统(vLLM)用 PagedAttention
        解决该问题,见 docs/tutorial.md 延伸阅读。
        """
        caches = []
        for _ in range(self.cfg.num_layers):
            k = torch.zeros(batch_size, self.cfg.num_kv_heads, 0, self.cfg.head_dim,
                            device=device, dtype=dtype)
            v = torch.zeros_like(k)
            caches.append((k, v))
        return caches

    def param_stats(self) -> dict:
        """参数量统计:total 含冻结参数,trainable 只算 requires_grad=True。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
