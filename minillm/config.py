"""模型与训练配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 8192          # 训练分词器后回填
    dim: int = 512                  # 隐藏层宽度
    num_layers: int = 8             # Transformer 层数
    num_heads: int = 8              # Q 注意力头数
    num_kv_heads: int = 2           # KV 头数(GQA),需能整除 num_heads
    hidden_dim: int = 1408          # SwiGLU 中间层(约为 8/3 * dim)
    max_seq_len: int = 1024         # RoPE 支持的上下文长度
    rope_theta: float = 10000.0     # RoPE 基频
    dropout: float = 0.0            # 小模型可不加
    tie_embeddings: bool = True     # 输入/输出 embedding 权重共享
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainConfig:
    # 数据
    data_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    tokenizer_path: str = "out/tokenizer.json"

    # 模型规模(默认对应 ~70M 参数的小模型)
    model: ModelConfig = field(default_factory=ModelConfig)

    # 优化
    lr: float = 3e-4
    min_lr_ratio: float = 0.1       # cosine 衰减下限 = lr * min_lr_ratio
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 5000
    batch_size: int = 32            # 每步的序列条数(每个累积微步)
    seq_len: int = 512              # 每条序列长度
    gradient_accumulation_steps: int = 4
    loss_spike_threshold: float = 5.0  # loss 尖峰回退阈值(相对 EMA 的倍数)

    # 运行
    device: str = "auto"            # auto / cpu / cuda / xpu / mps
    dtype: str = "bfloat16"         # 计算精度(bf16 在 CPU/GPU/XPU 上均可用)
    out_dir: str = "out"
    log_interval: int = 20
    eval_interval: int = 500
    eval_iters: int = 20
    save_interval: int = 1000
    seed: int = 1337
    num_workers: int = 0
