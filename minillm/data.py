"""数据管道(教学版逐行注释):文本文件 -> token 流 -> 随机 batch。

====================================================================
一、为什么 LLM 预训练的数据组织方式是"拼接成一条流"?
====================================================================
传统 NLP 任务(如翻译)会把数据组织成 (输入句, 输出句) 的配对。
但 GPT 类自回归语言模型的训练目标极其简单:
    给定前 t 个 token,预测第 t+1 个 token(next-token prediction)。
因此我们根本不需要"配对",只需要一段连续文本,把它左移一位就自动
得到了无数组 (x, y):
    tokens = [t0, t1, t2, t3]
    x      = [t0, t1, t2, t3]   <- 模型看到的上下文
    y      = [t1, t2, t3, ??]   <- 模型要预测的"下一个 token"
这就是 nanoGPT / Llama 等所有主流预训练框架的做法:
    1. 把整个语料库编码后首尾拼接成一条超长 token 流(numpy 数组);
    2. 每个训练 step,在这条流上随机截取 batch_size 个长度为
       seq_len+1 的窗口;
    3. 窗口去掉最后一个元素作为输入 x,去掉第一个元素作为标签 y。

这样做的好处:
  * 实现极简:没有 DataLoader、没有 collate_fn,一次 numpy 切片即可;
  * 随机截取保证每个 batch 都是语料的无偏采样;
  * 文档之间用 <eos>(id=0)分隔,模型会自然学到"句子/文档结束"信号。

代价:偶尔会截到跨文档的窗口(一半是文章A结尾、一半是文章B开头),
但由于 <eos> 的存在,模型很快学会在 <eos> 后"重置"注意力,实践中
对收敛几乎没有影响——这也是 nanoGPT 的原样做法。

====================================================================
二、本模块的两个角色
====================================================================
  TextDataset : 持有一条 token 流,提供 random_batch(训练用)
                和 iterate_eval(评估用,顺序、不重叠、可复现)。
  fixed_batch : 与 random_batch 相同的切法,但用固定种子,保证评估可复现。
  make_token_stream : 把"读文件 + 分词 + 加 <eos>"封装成便捷函数,
                      供 scripts/prepare_data.py 等复用。
"""

from __future__ import annotations  # 允许 `list[int] | None` 这类新式类型注解在旧 Python 上运行

import numpy as np                 # 数值数组:token 流本质是一个巨大的 int 数组
import torch                       # batch 最终要变成 torch.Tensor 才能喂给模型


class TextDataset:
    """把一条 token 流包装成可以按需采样 mini-batch 的"数据集"对象。

    参数
    ----
    text_ids : list[int] 或 np.ndarray
        已经分好词的整条语料。内部统一转成 np.uint32 存储:
        - uint32 足够容纳任何实际词表(vocab 一般 <= 65536),
          比 int64 省一半内存(千万级 token 的语料差异很大);
        - 用 numpy 而非 python list,是为了后面能直接用切片
          `self.data[i:i+L]` 拿到连续窗口(list 切片会构造新对象,
          numpy 切片是零拷贝视图,更快)。
    """

    def __init__(self, text_ids: list[int] | np.ndarray):
        # 如果传进来的是 python list,先转成 uint32 数组;
        # 如果已经是 ndarray(比如从 .npy 缓存加载),就直接引用,避免复制。
        self.data = np.asarray(text_ids, dtype=np.uint32) if isinstance(text_ids, list) else text_ids

    @classmethod
    def from_file(cls, path: str, tokenizer, add_eos: bool = True) -> "TextDataset":
        """读取 UTF-8 纯文本文件并构造成 token 流。

        文件格式约定:每行是一段独立文本(空行跳过)。
        我们逐行 encode,而不是把整个文件一次性 encode,原因有两个:
          1. BPE 编码复杂度近似 O(n),但对超长字符串仍可能很慢,
             按行切分可以把单次编码长度控制在合理范围;
          2. 保留"行/段"边界——行与行之间插入 <eos>,让模型学到
             文档结构信号(换行处即语义单元结束)。

        add_eos=True 时,每行末尾追加一个 id=0 的 <eos> token,
        充当"文档分隔符"(见模块 docstring 第一节)。
        """
        with open(path, encoding="utf-8") as f:
            text = f.read()  # 一次性读入小语料;真实大规模场景应 mmap 或分片

        ids: list[int] = []  # 最终的 token 流,先攒在 python list 里
        for para in text.split("\n"):      # 按换行切成"段"
            if not para.strip():           # 跳过空行/纯空白行,浪费 token 位
                continue
            ids.extend(tokenizer.encode(para))  # 该段 -> token id 列表,接在流尾部
            if add_eos:
                ids.append(0)              # <eos>:本词典中 id=0 固定为序列结束符
        return cls(ids)                    # 交给 __init__ 转成 uint32 数组

    def __len__(self) -> int:
        """返回 token 流的总长度(单位是 token,不是样本数)。"""
        return len(self.data)

    # ------------------------------------------------------------------
    # 训练采样:随机有放回地截取定长窗口
    # ------------------------------------------------------------------
    def random_batch(self, batch_size: int, seq_len: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """采一个训练 batch,返回 (x, y),形状均为 [batch_size, seq_len]。

        核心思想(再强调一次):y 就是 x 整体右移一位的结果。
        我们实际截取的是长度为 seq_len+1 的原始窗口 w,然后:
            x = w[:-1]   前 seq_len 个 token,作为模型输入
            y = w[1:]    后 seq_len 个 token,作为预测目标
        这样第 i 个位置上的 (x[:, i], y[:, i]) 恰好构成
        "用 w[..i] 预测 w[i+1]" 的监督对,一次前向同时算出
        seq_len 个位置的损失(next-token loss),效率极高。
        """
        # 窗口起点最大只能取到 len(data) - (seq_len+1),否则切片会越界变短。
        max_i = len(self.data) - seq_len - 1
        if max_i <= 0:
            # 语料比一个窗口还短,物理上无法采样——直接报错提醒用户
            # 要么换更大的语料,要么调小 seq_len。
            raise ValueError(f"语料太短({len(self.data)} tokens),无法切出 seq_len={seq_len}")

        # 均匀随机采样 batch_size 个起点。np.random.randint 的上界是开区间,
        # 所以写 size=batch_size、范围 [0, max_i) 保证每个窗口都完整合法。
        idxs = np.random.randint(0, max_i, size=batch_size)

        # 对每个起点做零拷贝切片,stack 成 [B, seq_len+1] 的矩阵。
        # astype(np.int64):torch 的 embedding/交叉熵要求索引是 int64(long),
        # 而我们的流是 uint32,必须显式转换,否则 torch.from_numpy 会报错。
        x = np.stack([self.data[i:i + seq_len] for i in idxs]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + seq_len] for i in idxs]).astype(np.int64)

        # 送入 GPU/CPU。.to(device) 内部会做一次 H2D 拷贝;
        # 生产级实现会用 pinned memory + 多进程预取来隐藏这段延迟,
        # 教学代码保持最朴素的同步拷贝。
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y

    # ------------------------------------------------------------------
    # 固定种子采样:供评估使用,保证可复现
    # ------------------------------------------------------------------
    def fixed_batch(self, batch_size: int, seq_len: int, seed: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """与 random_batch 完全相同的窗口切法,但起点由固定 seed 决定。

        为什么评估不能直接用 random_batch?
        因为评估指标(val loss / perplexity)必须可复现:如果每次评估
        随机采到不同的窗口,PPL 数字会抖动,训练曲线上就分不清是模型
        变好了还是"这次抽到的题简单"。用 np.random.default_rng(seed)
        这个独立的局部随机源(不污染全局 np.random 状态,训练采样
        序列不受影响),同一 seed 永远得到同一批窗口 → 指标可比。
        """
        max_i = len(self.data) - seq_len - 1
        if max_i <= 0:
            raise ValueError(f"语料太短({len(self.data)} tokens),无法切出 seq_len={seq_len}")
        rng = np.random.default_rng(seed)              # 局部 RNG,可复现且无副作用
        idxs = rng.integers(0, max_i, size=batch_size)  # 固定的一组窗口起点
        x = np.stack([self.data[i:i + seq_len] for i in idxs]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + seq_len] for i in idxs]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def make_token_stream(path: str, tokenizer, add_eos: bool = True) -> np.ndarray:
    """便捷函数:文本文件 -> uint32 token 流(numpy 数组)。

    只是 TextDataset.from_file 的薄封装,给 scripts/prepare_data.py
    这类"只想看一眼语料被编码成多少 token"的场景复用,避免到处
    import 整个类。返回 ndarray 方便直接 .tobytes() 存缓存或统计长度。
    """
    return TextDataset.from_file(path, tokenizer, add_eos).data
