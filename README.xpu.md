# MiniLLM XPU 运行说明

本文档是 [README.md](README.md) 的 XPU 专用补充，适用于 Intel XPU 环境。

## 1. 先确认环境

在你自己的 venv 中执行：

```bash
source /home/xrsh/ai/qwen/.venv/bin/activate
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('xpu_available:', hasattr(torch, 'xpu') and torch.xpu.is_available())
print('cuda_available:', torch.cuda.is_available())
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    x = torch.randn(2, 3, device='xpu')
    print('xpu_tensor_device:', x.device)
PY
```

期望输出：

```text
torch: 2.14.1+xpu
xpu_available: True
xpu_tensor_device: xpu:0
```

如果 `xpu_available` 是 `True`，说明 XPU 运行时已经可用。

## 2. 安装依赖

在项目根目录执行：

```bash
cd /home/xrsh/ai/llm-train-infer
source /home/xrsh/ai/qwen/.venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

项目的通用依赖在 [requirements.txt](requirements.txt) 中；核心依赖是 PyTorch，并且你当前环境已经使用的是 Intel XPU 版 PyTorch。

## 3. 先训练 tokenizer

```bash
cd /home/xrsh/ai/llm-train-infer
source /home/xrsh/ai/qwen/.venv/bin/activate
python scripts/train_tokenizer.py --vocab-size 8192
```

这一步会生成 `out/tokenizer.json`。

## 4. XPU 训练示例

### 4.1 冒烟/快速验证

```bash
python scripts/train.py --device xpu \
  --hidden 256 --layers 4 --heads 4 --kv-heads 2 \
  --steps 300 --batch-size 8 --seq-len 256 \
  --lr 3e-4 --out-dir out/smoke_xpu
```

这个版本适合快速验证代码是否能在 XPU 上跑起来。

### 4.2 更接近真实训练的 XPU 配置

```bash
python scripts/train.py --device xpu \
  --hidden 512 --layers 8 --heads 8 --kv-heads 4 \
  --steps 5000 --batch-size 32 --seq-len 512 \
  --lr 3e-4 --warmup-steps 200 \
  --eval-interval 500 --save-interval 500 \
  --out-dir out/train_xpu
```

### 4.3 更保守的 XPU 配置

```bash
python scripts/train.py --device xpu \
  --hidden 256 --layers 4 --heads 4 --kv-heads 2 \
  --steps 1000 --batch-size 16 --seq-len 256 \
  --lr 3e-4 --warmup-steps 100 \
  --eval-interval 100 --save-interval 200 \
  --out-dir out/train_xpu_small
```

## 5. 评估与生成

训练完成后，可以评估或直接生成：

```bash
python scripts/eval.py --ckpt out/train_xpu/ckpt
python scripts/generate.py --ckpt out/train_xpu/ckpt --prompt "你好" --max-new-tokens 64
```

## 6. SFT 微调

```bash
python scripts/prepare_sft_data.py
python scripts/train_sft.py --device xpu \
  --init-ckpt out/train_xpu/ckpt \
  --train data/sft_train.jsonl \
  --val data/sft_val.jsonl \
  --out-dir out/sft_xpu \
  --epochs 2 --batch-size 8 --seq-len 256
```

## 7. 设备自动选择

在 [minillm/train.py](minillm/train.py) 里，`auto` 的选择顺序是：

```text
CUDA -> XPU -> MPS -> CPU
```

也就是说，如果没有显式传 `--device xpu`，程序会优先尝试 CUDA，再尝试 XPU，再尝试 MPS，最后回退到 CPU。

## 8. 常见问题

### 8.1 旧 checkpoint 恢复失败

如果你之前训练过不同词表或不同配置的模型，可能会遇到 `size mismatch`。此时可以：

```bash
rm -rf out/ckpt out/smoke_xpu out/train_xpu
```

或换一个新的 `--out-dir`。

### 8.2 速度看起来和 CPU 差不多

这是小模型 / 小 batch 的正常现象。XPU 的优势通常在更大 workload（更大模型、更大 batch、更长 seq_len）下更明显。

### 8.3 监控不到 XPU 负载

可以尝试：

```bash
xpu-smi
intel_gpu_top
sycl-ls
```

如果这些命令不存在，通常说明本机没有安装对应的 Intel XPU 监控组件。

## 9. 结论

这个工程已接入 Intel XPU 运行支持；在正常安装了 Intel XPU 版 PyTorch 的环境中，可以使用 `--device xpu` 运行预训练、评估、生成和 SFT。
