"""预训练循环:AdamW + Cosine(warmup) + 梯度裁剪 + 梯度累积 + loss 尖峰回退。

Checkpoint 目录结构(out/ckpt/):
    model.pt     -- 模型权重(state_dict,含 buffers 之外的可学习参数)
    meta.json    -- ModelConfig dict,便于独立恢复模型
    optim.pt     -- 优化器状态 + RNG 状态(仅 latest 保留)
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
from .data import TextDataset
from .evaluate import evaluate_loss
from .model import LLMModel, count_params


# --------------------------------------------------------------------- utils
def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _get_device_rng_states():
    if torch.cuda.is_available():
        return torch.cuda.get_rng_state_all()
    if hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "get_rng_state_all"):
        return torch.xpu.get_rng_state_all()
    return None


def _restore_device_rng_states(rng_states):
    if rng_states is None:
        return
    if hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "set_rng_state_all"):
        states = [s.to("xpu") if isinstance(s, torch.Tensor) and s.device.type != "xpu" else s for s in rng_states]
        torch.xpu.set_rng_state_all(states)
    elif torch.cuda.is_available():
        states = [s.to("cuda") if isinstance(s, torch.Tensor) and s.device.type != "cuda" else s for s in rng_states]
        torch.cuda.set_rng_state_all(states)


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW + 参数分组:weight decay 不作用于 norm/bias/embedding(Llama 惯例)。

    为什么:L2 正则(weight decay)的目的是抑制大矩阵权重过拟合;而 RMSNorm 的
    γ、bias、词嵌入这些"低维/查表"参数被衰减会直接损害表达能力(嵌入向量
    本身就该有大范数差异),所以把它们单独分到 weight_decay=0 组。
    判据 p.ndim <= 1 覆盖所有一维参数(norm γ / bias),再按名字排除 embedding。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in name or "embeddings" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=tuple(cfg.betas), eps=1e-8,
    )


def get_lr(step: int, cfg: TrainConfig) -> float:
    """学习率调度:线性 warmup + cosine 衰减到 min_lr。

    - warmup:训练初期 Adam 的二阶矩估计(梯度平方的滑动平均)还没建立,
      步长过大容易发散,故从 0 线性升到 lr,通常占前 1~5% 步数;
    - cosine:中后期按余弦平滑降到 lr*min_lr_ratio,比阶梯衰减更稳,
      末期小学习率帮助 loss 收敛到更平坦的极小值(Llama/GPT 均用此调度)。
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0 平滑下降
    decayed = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos
    return cfg.lr * decayed


# ------------------------------------------------------------------ ckpt I/O
def save_checkpoint(model: LLMModel, ckpt_dir: str, step: int,
                    optimizer=None, is_latest: bool = False) -> None:
    """保存权重 + 配置;is_latest=True 时额外保存优化器/RNG 状态以支持续训。

    分文件的原因:model.pt(权重)与 optim.pt(Adam 动量等)用途不同——
    推理只需要前者,best/ 目录因此不写 optim.pt,省磁盘也避免误覆盖续训状态。
    """
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step}, d / "model.pt")
    model.cfg.save(d / "meta.json")             # ModelConfig 单独存 JSON,人可读
    if optimizer is not None and is_latest:
        # RNG 状态一并保存:断点续训后数据采样序列与中断前完全一致(可复现)
        torch.save({"optimizer": optimizer.state_dict(), "step": step,
                    "rng": {
                        "python": None,
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "torch_xpu": _get_device_rng_states() if hasattr(torch, "xpu") and torch.xpu.is_available() else None,
                    }},
                   d / "optim.pt")


def load_model(ckpt_dir: str, device: torch.device,
               dtype: torch.dtype | None = None) -> tuple[LLMModel, int]:
    """推理/断点续训用的模型加载。返回 (model, step)。

    先按 meta.json 重建同构模型再灌权重;state_dict 用 cpu 加载后再 .to(device),
    保证 GPU 上保存的 checkpoint 能在纯 CPU 机器上恢复(反之亦然)。
    """
    d = Path(ckpt_dir)
    cfg = ModelConfig.load(str(d / "meta.json"))
    model = LLMModel(cfg)
    # 本地 checkpoint 含非 tensor 对象(step 等),且 optim.pt 还带 numpy RNG,
    # torch>=2.6 默认 weights_only=True 会反序列化失败,故显式置 False(可信本地文件)。
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    step = state.get("step", 0)
    model.to(device=device, dtype=dtype)
    # RoPE buffer 随模型一起 .to(device) 即可(实数 cos/sin buffer)
    return model.eval(), step


def _is_compatible_checkpoint(model: LLMModel, ckpt_dir: str) -> bool:
    """判断 checkpoint 的模型配置是否与当前训练配置兼容。

    如果词表或架构尺寸不同,则直接拒绝恢复,避免在大 vocab 切换后用旧权重
    覆盖新模型并产生 size mismatch。此类场景通常意味着用户更换了 tokenizer/
    训练配置,需要清理旧 out/ckpt 或从新目录开始训练。
    """
    meta_path = Path(ckpt_dir) / "meta.json"
    if not meta_path.exists():
        return True

    try:
        ckpt_cfg = ModelConfig.load(str(meta_path))
    except Exception:
        return False

    current = model.cfg
    for key in (
        "vocab_size", "dim", "num_layers", "num_heads", "num_kv_heads",
        "hidden_dim", "max_seq_len", "rope_theta", "rms_norm_eps",
        "tie_embeddings", "dropout"
    ):
        if getattr(ckpt_cfg, key) != getattr(current, key):
            print(f"[train] checkpoint mismatch on {key}: ckpt={getattr(ckpt_cfg, key)} current={getattr(current, key)}")
            return False
    return True


def try_resume(ckpt_dir: str, model: LLMModel, optimizer: torch.optim.Optimizer,
               device: torch.device) -> int:
    """存在 optim.pt 则恢复训练状态,返回已完成步数;否则返回 0(从头训练)。

    恢复内容:模型权重(model.pt)、Adam 的一阶/二阶矩、当前 step、
    numpy/torch RNG——缺任何一样,"续训"都不等价于"没中断过"。
    """
    p = Path(ckpt_dir) / "optim.pt"
    if not p.exists():
        return 0
    if not _is_compatible_checkpoint(model, ckpt_dir):
        print(f"[train] skip resume from {p}: checkpoint config does not match current model")
        return 0
    # optim.pt 含 numpy/torch RNG 状态,torch>=2.6 默认 weights_only=True 会因
    # numpy global 不在白名单而报错;本地可信 checkpoint 显式置 False。
    st = torch.load(p, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(st["optimizer"])
    # 关键:同时恢复模型权重(model.pt),否则续训带着全新随机权重继续,
    # "断点续训"形同虚设。load 用 cpu 再 .to(device) 以跨设备兼容(与 load_model 一致)。
    ckpt_path = Path(ckpt_dir) / "model.pt"
    if ckpt_path.exists():
        mw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(mw["model"])
        except RuntimeError as e:
            print(f"[train] skip resume from {ckpt_path}: incompatible checkpoint weights ({e})")
            return 0
        model.to(device=device)
    rng = st.get("rng", {})
    if isinstance(rng.get("numpy"), tuple) or rng.get("numpy") is not None:
        try:
            np.random.set_state(rng["numpy"])
        except Exception:
            pass                                # numpy 版本差异导致格式不符:跳过而非崩溃
    cpu_rng = rng.get("torch_cpu")
    if isinstance(cpu_rng, torch.Tensor) and cpu_rng.dtype == torch.uint8:
        torch.set_rng_state(cpu_rng)
    if rng.get("torch_cuda") and device.type == "cuda" and torch.cuda.is_available():
        _restore_device_rng_states(rng["torch_cuda"])
    if rng.get("torch_xpu") and device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        _restore_device_rng_states(rng["torch_xpu"])
    print(f"[train] resumed from {p} at step {st['step']}")
    return st["step"]


# ---------------------------------------------------------------------- train
def train(cfg: TrainConfig, resume: bool = True) -> None:
    """预训练主循环。流程:种子 -> 数据 -> 模型 -> (续训) -> 步循环。"""
    # 固定所有随机源:同一 seed + 同一配置 => 可完全复现的训练过程
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device(cfg.device)
    dtype = get_dtype(cfg.dtype)
    print(f"[train] device={device} dtype={dtype}")

    from .tokenizer import BPETokenizer
    tok = BPETokenizer.load(cfg.tokenizer_path)
    # 关键回填:词表大小以实际训练出的 tokenizer 为准,而非配置里的默认值,
    # 否则 embedding 行数与 token id 上限不匹配会越界崩溃
    cfg.model.vocab_size = tok.vocab_size
    print(f"[train] tokenizer loaded: {tok}")

    train_ds = TextDataset.from_file(cfg.data_path, tok)
    val_ds = TextDataset.from_file(cfg.val_path, tok) if os.path.exists(cfg.val_path) else None
    print(f"[train] data: train={len(train_ds)} tokens, val={len(val_ds) if val_ds else 0} tokens")

    model = LLMModel(cfg.model).to(device=device, dtype=dtype)
    n_params = count_params(model)
    print(f"[train] model params: {n_params/1e6:.2f}M "
          f"(layers={cfg.model.num_layers}, dim={cfg.model.dim}, "
          f"heads={cfg.model.num_heads}/{cfg.model.num_kv_heads})")

    optimizer = build_optimizer(model, cfg)
    ckpt_dir = os.path.join(cfg.out_dir, "ckpt")
    start_step = try_resume(ckpt_dir, model, optimizer, device) if resume else 0

    # tokens/step = batch*seq*累积步数;"看到多少 token"是比步数更公平的算力度量
    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.gradient_accumulation_steps
    total_tokens = tokens_per_step * cfg.max_steps
    print(f"[train] steps={cfg.max_steps}, tokens/step={tokens_per_step}, "
          f"total tokens≈{total_tokens/1e6:.1f}M")

    running_loss_ema = None      # loss 指数滑动平均:尖峰检测的基线
    best_val = float("inf")      # 历史最优验证 loss -> out/best
    t0 = time.time()
    step = start_step

    while step < cfg.max_steps:
        # ---- 每步按调度器更新 lr(warmup/cosine 都体现在这里)
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- 梯度累积:显存不够时的标准技巧。
        # 连续做 grad_accum 个 mini-batch 的前反向、梯度在 .grad 里累加,
        # 攒够后一次性 clip+step,等效 batch = batch_size * grad_accum,
        # 而峰值显存只相当于单个 mini-batch(nanoGPT/Llama 均如此)。
        optimizer.zero_grad(set_to_none=True)   # set_to_none:省一次 memset 且省显存
        accum_loss = 0.0
        spike_skip = False
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = train_ds.random_batch(cfg.batch_size, cfg.seq_len, device)
            # autocast:矩阵乘走 bf16(快、省显存),softmax/norm 等自动回退 fp32
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(device.type in ("cuda", "mps", "xpu") or dtype != torch.float32)):
                logits = model(x)               # (B,S,V)
                # 交叉熵:预测 x 的下一 token(y = x 右移一位);ignore_index=-1
                # 预留给将来可能的 padding(当前语料无 padding,纯防御)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                       y.view(-1), ignore_index=-1)
            loss = loss / cfg.gradient_accumulation_steps   # 除以累积数=对梯度取平均
            loss.backward()
            accum_loss += loss.item()

        # ---- loss 尖峰检测:大 loss 常由坏 batch 触发,若照常更新会把模型
        # "炸"出好区域。策略:loss > EMA 的 threshold 倍时跳过本次参数更新
        # (梯度直接丢弃),EMA 仍缓慢吸收该值以免基线僵化。生产系统一般还会
        # 回滚到上一 checkpoint,此处简化为只跳过,见 docs/tutorial.md。
        if running_loss_ema is None:
            running_loss_ema = accum_loss
        if accum_loss > cfg.loss_spike_threshold * running_loss_ema and step > cfg.warmup_steps:
            print(f"[train] step {step}: loss spike {accum_loss:.3f} (ema {running_loss_ema:.3f}), skip update")
            optimizer.zero_grad(set_to_none=True)
            running_loss_ema = 0.99 * running_loss_ema + 0.01 * accum_loss
            step += 1
            continue

        # ---- 全局梯度范数裁剪:超过 grad_clip 则整体等比缩小方向不变,
        # 防止个别大梯度把 Adam 动量带偏(与尖峰检测互补)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running_loss_ema = 0.99 * running_loss_ema + 0.01 * accum_loss

        step += 1

        # ---- 日志:ppl = exp(loss) 是语言模型的标准可读指标
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            toks_s = (step - start_step) * tokens_per_step / max(dt, 1e-6)
            ppl = math.exp(min(running_loss_ema, 20))     # min 防溢出
            print(f"[train] step {step}/{cfg.max_steps} | loss {accum_loss:.4f} "
                  f"(ema {running_loss_ema:.4f}, ppl {ppl:.1f}) | lr {lr:.2e} "
                  f"| grad {grad_norm:.2f} | {toks_s/1e3:.1f}k tok/s")

        # ---- 定期评估:val loss 创新低就把"纯权重"另存 out/best(不含 optim)
        if val_ds is not None and step % cfg.eval_interval == 0:
            vl = evaluate_loss(model, val_ds, cfg, device, dtype)
            print(f"[eval ] step {step} | val loss {vl:.4f} | val ppl {math.exp(min(vl, 20)):.1f}")
            if vl < best_val:
                best_val = vl
                save_checkpoint(model, os.path.join(cfg.out_dir, "best"), step)

        # ---- 定期保存 latest(含优化器状态),供断电续训
        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_checkpoint(model, ckpt_dir, step, optimizer, is_latest=True)
            print(f"[train] saved checkpoint @ step {step} -> {ckpt_dir}")

    save_checkpoint(model, ckpt_dir, step, optimizer, is_latest=True)
    print(f"[train] done in {time.time()-t0:.1f}s, final ckpt @ {ckpt_dir}")
