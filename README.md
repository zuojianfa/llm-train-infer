# MiniLLM —— 从零实现 Transformer LLM：训练 → 微调 → 推理 全流程

一个**教学优先**的迷你 LLM 工程：不依赖任何大模型库，用纯 PyTorch 手写
BPE 分词器、Llama2 风格 Decoder-only Transformer（RoPE / RMSNorm / SwiGLU / GQA / KV Cache），
并串起 **预训练 → 验证评估 → SFT 监督微调 → 采样生成 → OpenAI 兼容 HTTP 服务** 的完整链路。
所有核心代码均配有逐行中文教学注释，目标是让零基础读者也能"事无巨细"地理解 LLM 原理。

## 1. 项目结构

```
minillm/                 # 核心库（每个模块都有教学级注释）
  config.py              # ModelConfig：超参数集中定义与参数量估算
  tokenizer.py           # BPE 分词器：字节对编码训练/编码/解码，无损往返
  data.py                # 数据管线：tokenize、拼接、input/target 右移、batch 采样
  model.py               # Transformer 主干：RMSNorm/RoPE/Attention(GQA)/SwiGLU FFN/KV Cache
  train.py               # 预训练循环：AdamW、warmup+cosine、梯度累积、bf16、checkpoint
  evaluate.py            # 验证集 loss / 困惑度(PPL) 评估
  generate.py            # 推理：greedy / top-k / top-p 采样，流式输出，KV Cache
  sft.py                 # SFT 监督微调：chat template + prompt 段 loss mask(-100)
  serve.py               # FastAPI 服务：OpenAI Chat Completions 兼容接口
scripts/                 # CLI 入口（薄封装，逻辑都在 minillm/ 内）
  prepare_data.py        # 下载语料（失败自动回退内置离线双语语料）
  train_tokenizer.py     # 训练 BPE 词表 -> out/tokenizer.json
  train.py               # 预训练
  eval.py                # 评估 PPL
  train_sft.py           # SFT 微调（从预训练 ckpt 热启动）
  prepare_sft_data.py    # 生成内置中文指令样例 data/sft_*.jsonl
  generate.py            # 交互式/单次文本生成
  serve.py               # 启动 HTTP 服务
tests/                   # pytest 单元测试：tokenizer 往返 / KV Cache 一致性 / 采样可复现 / SFT 掩码 ...
docs/                    # 教程文档系列（见下）
data/ out/               # 运行产物（语料、词表、checkpoint），已被 .gitignore 排除
```

## 2. 快速开始

```bash
pip install -r requirements.txt          # torch/fastapi/pytest 等
```

全流程六步走（每步产物供下一步使用）：

```bash
# ① 准备语料（默认离线内置语料；--source wiki40b 可尝试在线下载）
python scripts/prepare_data.py

# ② 训练 BPE 分词器（生成 out/tokenizer.json）
python scripts/train_tokenizer.py --vocab-size 8192

# ③ 预训练（冒烟配置示例，CPU 低内存可跑通）
python scripts/train.py --hidden 256 --layers 4 --heads 4 --kv-heads 2 \
    --steps 300 --batch-size 8 --seq-len 256 --lr 3e-4 --out-dir out/smoke

# ④ 评估验证集困惑度
python scripts/eval.py --ckpt out/smoke/best

# ⑤ 文本生成（可加 --stream 流式输出）
python scripts/generate.py --ckpt out/smoke/best --prompt "人工智能" --max-new-tokens 64

# ⑥ SFT 监督微调 + 部署
python scripts/prepare_sft_data.py
python scripts/train_sft.py --init-ckpt out/smoke/best --out-dir out/sft --epochs 2
python scripts/serve.py --ckpt out/sft/best --chatml &     # http://127.0.0.1:8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minillm","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

运行测试：

```bash
pytest tests/ -x          # 全部离线、秒级完成
```

## 3. 学习路径（docs/）

| 文档 | 内容 |
|---|---|
| [docs/01_tokenizer.md](docs/01_tokenizer.md) | BPE 分词原理：字节对合并、无损往返、OOV 兜底 |
| [docs/02_model.md](docs/02_model.md) | Transformer 逐个组件拆解：RMSNorm→RoPE→GQA→SwiGLU |
| [docs/03_training.md](docs/03_training.md) | 预训练全流程：next-token 目标、优化器、调度、显存技巧 |
| [docs/04_sft.md](docs/04_sft.md) | 监督微调：chat template、loss mask、防灾难遗忘 |
| [docs/05_inference.md](docs/05_inference.md) | 推理与采样：KV Cache、top-k/top-p、流式输出 |
| [docs/06_service.md](docs/06_service.md) | 把模型变成 OpenAI 兼容 API 服务 |

## 4. 设计要点速览

- **词表与 OOV**：基础词表收录全部 BMP 原子字符 + 语料字符 + `<unk>`；编码时仅当合并结果
  是词表成员才应用 merge 规则，保证 encode→decode **逐字符无损往返**（有回归测试锁定）。
- **训练目标**：`target = input 右移一位`，padding 位置不参与 loss（详见 docs/03）。
- **SFT**：与预训练共用同一模型，差异只在数据格式（prompt/response 配对）与
  loss mask（prompt 段置 `-100`，只学回答）；模板字符串与 `serve.py --chatml` 严格一致。
- **KV Cache**：增量解码与全量前向数值等价（`tests/test_model.py` 中有断言），这是
  所有生产级推理加速的地基。
- **网络形态**：单机单进程，无分布式；设备自动选择 CUDA/MPS/CPU；支持 bf16 autocast。

## 5. License

MIT，见 [LICENSE](LICENSE)。
