# MiniLLM 完整训练与推理架构流程图

## 🎯 整体流程概览

```mermaid
flowchart TD
    %% ============ 阶段 0: 环境准备 ============
    subgraph Phase0[阶段 0: 环境准备]
        P0A[安装依赖\npip install -r requirements.txt]
        P0B[检查设备\nCUDA/XPU/MPS/CPU]
    end

    %% ============ 阶段 1: 数据准备 ============
    subgraph Phase1[阶段 1: 语料准备]
        P1A[原始语料文件\ndata/train.txt, data/val.txt]
        P1B[scripts/prepare_data.py]
        P1C{在线下载?}
        P1D[HuggingFace wiki40b]
        P1E[内置离线双语语料]
        P1F[生成训练/验证文本文件]
        
        P1A --> P1B
        P1B --> P1C
        P1C -->|--source wiki40b| P1D
        P1C -->|默认/失败回退| P1E
        P1D --> P1F
        P1E --> P1F
    end

    %% ============ 阶段 2: 分词器训练 ============
    subgraph Phase2[阶段 2: BPE 分词器训练]
        P2A[scripts/train_tokenizer.py]
        P2B[读取训练语料]
        P2C[BPE 训练算法\nminillm/tokenizer.py::BPETokenizer.train()]
        P2D[词表构建:\n- 原子符号(BMP全集+代理对)\n- 迭代合并高频符号对\n- 合并规则按优先级排序]
        P2E[保存词表\nout/tokenizer.json\n{vocab, merges}]
        P2F[自检:编解码往返测试]
        
        P2A --> P2B --> P2C --> P2D --> P2E --> P2F
    end

    %% ============ 阶段 3: 预训练 ============
    subgraph Phase3[阶段 3: 预训练]
        P3A[scripts/train.py]
        P3B[加载配置\nTrainConfig + ModelConfig]
        P3C[加载分词器\nBPETokenizer.load()]
        P3D[构建数据集\nTextDataset.from_file()]
        P3E[构建模型\nLLMModel(ModelConfig)]
        P3F{是否续训?}
        P3G[加载 checkpoint\nmodel.pt + optim.pt + RNG状态]
        P3H[训练主循环\nminillm/train.py::train()]
        
        %% 训练循环内部
        subgraph TrainLoop[训练步循环]
            TL1[更新学习率\nWarmup + Cosine]
            TL2[梯度累积循环\ngradient_accumulation_steps]
            TL3[随机采样 batch\nTextDataset.random_batch()]
            TL4[前向传播\nmodel(x) -> logits]
            TL5[计算交叉熵损失\nNext-token Prediction]
            TL6[反向传播 + 梯度累积]
            TL7[Loss 尖峰检测\n跳过异常更新]
            TL8[梯度裁剪\nclip_grad_norm_]
            TL9[优化器步进\nAdamW.step()]
            TL10[EMA 更新]
            TL11[日志记录]
            TL12{评估间隔?}
            TL13[验证集评估\nminillm/evaluate.py]
            TL14{保存间隔?}
            TL15[保存 checkpoint\nmodel.pt + meta.json + optim.pt]
            TL16{最佳模型?}
            TL17[保存 best/\n仅模型权重]
        end
        
        P3A --> P3B --> P3C --> P3D --> P3E --> P3F
        P3F -->|是| P3G --> P3H
        P3F -->|否| P3H
        P3H --> TrainLoop
        TrainLoop --> TL1 --> TL2 --> TL3 --> TL4 --> TL5 --> TL6 --> TL7 --> TL8 --> TL9 --> TL10 --> TL11 --> TL12
        TL12 -->|是| TL13 --> TL14
        TL12 -->|否| TL14
        TL14 -->|是| TL15 --> TL16
        TL14 -->|否| TL16
        TL16 -->|是| TL17
    end

    %% ============ 阶段 4: 评估 ============
    subgraph Phase4[阶段 4: 评估验证集]
        P4A[scripts/eval.py]
        P4B[加载模型权重\nload_model(ckpt_dir)]
        P4C[加载验证集]
        P4D[计算困惑度 PPL\nminillm/evaluate.py]
        P4E[输出 val_loss & val_ppl]
        
        P4A --> P4B --> P4C --> P4D --> P4E
    end

    %% ============ 阶段 5: 文本生成 ============
    subgraph Phase5[阶段 5: 文本生成/推理]
        P5A[scripts/generate.py]
        P5B[加载模型 + 分词器]
        P5C[构建 Generator\nminillm/generate.py]
        P5D{流式输出?}
        P5E[Generator.stream()\n逐token产出]
        P5F[Generator.generate()\n一次性返回]
        P5G[采样策略:\n- Greedy (T=0)\n- Temperature\n- Top-k\n- Top-p (Nucleus)]
        P5H[KV Cache 增量解码\nprefill + decode 循环]
        
        P5A --> P5B --> P5C --> P5D
        P5D -->|--stream| P5E
        P5D -->|默认| P5F
        P5E --> P5G --> P5H
        P5F --> P5G --> P5H
    end

    %% ============ 阶段 6: SFT 微调 ============
    subgraph Phase6[阶段 6: SFT 监督微调]
        P6A[scripts/prepare_sft_data.py]
        P6B[生成指令数据\ndata/sft_train.jsonl\ndata/sft_val.jsonl]
        P6C[scripts/train_sft.py]
        P6D[加载预训练权重\n热启动 load_model()]
        P6E[构建 SFTDataset\n带 -100 loss mask]
        P6F[SFT 训练循环\nminillm/sft.py::run_sft()]
        
        %% SFT 循环内部
        subgraph SFTLoop[SFT 步循环]
            SL1[更新学习率]
            SL2[梯度累积]
            SL3[随机采样 batch\nSFTDataset.random_batch()]
            SL4[前向 + 带 mask 交叉熵\nignore_index=-100]
            SL5[反向 + 梯度裁剪 + 步进]
            SL6[日志 + 评估 + 保存]
        end
        
        P6A --> P6B --> P6C --> P6D --> P6E --> P6F --> SFTLoop
        SFTLoop --> SL1 --> SL2 --> SL3 --> SL4 --> SL5 --> SL6
    end

    %% ============ 阶段 7: 服务部署 ============
    subgraph Phase7[阶段 7: HTTP 推理服务]
        P7A[scripts/serve.py]
        P7B[加载 SFT checkpoint]
        P7C[创建 FastAPI 应用\nminillm/serve.py::create_app()]
        P7D[启动服务\nuvicorn --host 0.0.0.0 --port 8000]
        P7E[OpenAI 兼容接口:\n- POST /v1/chat/completions\n- POST /v1/completions\n- GET /health]
        P7F[流式响应 SSE\nchat.completion.chunk]
        P7G[ChatML 模板\n用户:...\n助手:...]
        
        P7A --> P7B --> P7C --> P7D --> P7E
        P7E --> P7F
        P7E --> P7G
    end

    %% ============ 核心模块依赖关系 ============
    subgraph Core[核心库模块依赖图]
        M1[minillm/config.py\nModelConfig / TrainConfig]
        M2[minillm/tokenizer.py\nBPETokenizer]
        M3[minillm/data.py\nTextDataset / SFTDataset]
        M4[minillm/model.py\nLLMModel / Attention / RMSNorm / RoPE / SwiGLU]
        M5[minillm/train.py\n训练循环 / 优化器 / 调度器 / Checkpoint]
        M6[minillm/evaluate.py\nPPL 评估]
        M7[minillm/generate.py\nGenerator / 采样策略 / KV Cache]
        M8[minillm/sft.py\nSFT 微调 / Loss Mask / Chat Template]
        M9[minillm/serve.py\nFastAPI / OpenAI API 兼容]
        
        M1 --> M2
        M1 --> M3
        M1 --> M4
        M1 --> M5
        M1 --> M6
        M1 --> M7
        M1 --> M8
        M1 --> M9
        M2 --> M3
        M2 --> M7
        M2 --> M8
        M2 --> M9
        M3 --> M5
        M3 --> M6
        M3 --> M8
        M4 --> M5
        M4 --> M6
        M4 --> M7
        M4 --> M8
        M4 --> M9
        M5 --> M8
        M7 --> M9
    end

    %% ============ 数据流向 ============
    subgraph DataFlow[关键数据流向]
        DF1[原始文本\n*.txt] -->|分词编码| DF2[Token ID 流\nnp.uint32 数组]
        DF2 -->|随机截窗| DF3[训练 Batch\nx: [B, S], y: [B, S]]
        DF3 -->|前向| DF4[Logits\n[B, S, V]]
        DF4 -->|交叉熵| DF5[Loss 标量]
        DF5 -->|反向| DF6[梯度更新]
        
        DF7[指令配对 JSONL\n*.jsonl] -->|格式化+编码| DF8[SFT 样本\n(input_ids, labels=-100/真实)]
        DF8 -->|Pad + Batch| DF9[SFT Batch\nx: [B, S], y: [B, S]]
        DF9 -->|前向+Mask CE| DF10[SFT Loss]
        
        DF11[用户 Prompt\n字符串] -->|编码| DF12[Token IDs\nList[int]]
        DF12 -->|Prefill| DF13[KV Cache 初始化\n全层 (K,V)]
        DF13 -->|Decode 循环| DF14[单 Token 前向\n+ 采样 + KV 更新]
        DF14 -->|解码| DF15[生成文本\n字符串]
    end

    %% ============ 模型架构内部 ============
    subgraph ModelArch[LLMModel 内部架构]
        MA1[Input Tokens\n[B, S]]
        MA2[Token Embedding\n[B, S, D]\n权重共享]
        MA3[N x TransformerBlock\nPre-Norm 残差]
        MA4[RMSNorm\n最终归一化]
        MA5[LM Head\n[B, S, V]\nLogit Soft Cap]
        MA6[Output Logits\n[B, S, V]]
        
        MA1 --> MA2 --> MA3 --> MA4 --> MA5 --> MA6
        
        subgraph Block[TransformerBlock 结构]
            B1[RMSNorm]
            B2[Attention\nGQA + RoPE + KV Cache]
            B3[残差连接 x + h]
            B4[RMSNorm]
            B5[FeedForward\nSwiGLU]
            B6[残差连接 x + h]
            B1 --> B2 --> B3 --> B4 --> B5 --> B6
        end
        
        MA3 --> Block
    end

    %% ============ 设备与精度 ============
    subgraph DeviceDtype[设备与精度自动选择]
        DD1[TrainConfig.device=auto]
        DD2{设备检测优先级}
        DD3[CUDA GPU]
        DD4[Intel XPU]
        DD5[Apple MPS]
        DD6[CPU]
        DD7[TrainConfig.dtype=bfloat16]
        DD8[autocast 上下文\n矩阵乘走 BF16\nSoftmax/Norm 回退 FP32]
        
        DD1 --> DD2
        DD2 --> DD3
        DD2 --> DD4
        DD2 --> DD5
        DD2 --> DD6
        DD7 --> DD8
    end

    %% ============ 连接各阶段 ============
    Phase0 --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase3 --> Phase5
    Phase3 --> Phase6
    Phase6 --> Phase5
    Phase6 --> Phase7
    
    %% 样式
    classDef phase fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef core fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef device fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    
    class Phase0,Phase1,Phase2,Phase3,Phase4,Phase5,Phase6,Phase7 phase;
    class Core core;
    class DataFlow data;
    class ModelArch model;
    class DeviceDtype device;
```

---

## 🔄 核心数据流向详解

### 1. 预训练数据流
```
原始文本 (data/train.txt)
    │
    ▼
BPETokenizer.encode() ──► Token ID 列表 [t0, t1, t2, ...]
    │
    ▼
TextDataset.from_file() ──► 拼接成单条流 + 插入 <eos>(id=0)
    │
    ▼
np.uint32 数组 (零拷贝视图)
    │
    ▼
random_batch(batch_size, seq_len)
    │
    ├──► x = data[i : i+seq_len]      ──► [B, S] 模型输入
    └──► y = data[i+1 : i+1+seq_len]  ──► [B, S] 标签(右移一位)
    │
    ▼
model.forward(x) ──► logits [B, S, V]
    │
    ▼
F.cross_entropy(logits.view(-1,V), y.view(-1)) ──► loss
    │
    ▼
loss.backward() → optimizer.step() → 参数更新
```

### 2. SFT 数据流 (关键差异: Loss Mask)
```
JSONL 样本 {"instruction": "...", "output": "..."}
    │
    ▼
format_example() ──► prompt="用户:...\n助手:", target="回答\n"
    │
    ▼
prompt_ids = tokenizer.encode(prompt)        # 不算 loss
target_ids = tokenizer.encode(target) + [EOS] # 算 loss
    │
    ▼
labels = [-100]*len(prompt_ids) + target_ids  # -100 = ignore_index
input_ids = prompt_ids + target_ids
    │
    ▼
SFTDataset.random_batch() ──► Pad 到统一长度
    │
    ▼
model(x[:, :-1]) ──► logits
    │
    ▼
F.cross_entropy(..., ignore_index=-100) ──► 仅 answer 部分产生梯度
```

### 3. 推理数据流 (KV Cache)
```
用户 Prompt "人工智能"
    │
    ▼
tokenizer.encode() ──► [t0, t1, t2]
    │
    ▼
=== Prefill 阶段 (并行) ===
model(input_ids, start_pos=0, caches=empty)
    │
    ├──► 计算全 prompt 的 K, V
    ├──► 填充各层 KV Cache: (B, H_kv, T_prompt, hd)
    └──► 输出最后位置的 logits
    │
    ▼
=== Decode 循环 (增量, 串行) ===
for step in range(max_new_tokens):
    │
    ├──► 采样下一个 token: _sample_next_token(logits, T, top_k, top_p)
    │
    ├──► 加入 generated 列表
    │
    ├──► 增量前向: model([[next_token]], start_pos=pos, caches=prev_caches)
    │       │
    │       ├──► 只计算新 token 的 Q, K, V
    │       ├──► K, V 拼接到缓存: cat([cache_K, new_K], dim=2)
    │       ├──► Attention: Q 看完整 K/V (含历史)
    │       └──► 返回新 logits + 更新后的 caches
    │
    └──► pos += 1
    │
    ▼
tokenizer.decode(generated) ──► 输出文本
```

---

## 🏗️ 模型架构详细结构

```
LLMModel (Decoder-only Transformer)
│
├── tok_embeddings: Embedding(V, D) ──────┐
│                                         │ 权重共享
├── layers: ModuleList[TransformerBlock]  │
│   │                                     ▼
│   │                                 output: Linear(D, V, bias=False)
│   │                                     ▲
│   │                                     │ weight = tok_embeddings.weight
│   ▼                                     │
│ ┌─────────────────────────────────────┐ │
│ │ TransformerBlock (× N 层)           │ │
│ │                                     │ │
│ │ ┌─────────────────────────────────┐ │ │
│ │ │ Attention (GQA + RoPE + KV Cache)│ │ │
│ │ │                                 │ │ │
│ │ │ xq = wq(x)  → (B, S, H, hd)    │ │ │
│ │ │ xk = wk(x)  → (B, S, H_kv, hd) │ │ │
│ │ │ xv = wv(x)  → (B, S, H_kv, hd) │ │ │
│ │ │                                 │ │ │
│ │ │ RoPE(xq), RoPE(xk)             │ │ │
│ │ │                                 │ │ │
│ │ │ if cache: cat([cache_K, xk])   │ │ │
│ │ │ GQA: expand KV to H heads      │ │ │
│ │ │ SDPA(q, k, v, mask)            │ │ │
│ │ │ wo(out)                        │ │ │
│ │ └─────────────────────────────────┘ │ │
│ │                 │                   │ │
│ │                 ▼                   │ │
│ │         x = x + attention_out       │ │ (残差 1)
│ │                 │                   │ │
│ │                 ▼                   │ │
│ │ ┌─────────────────────────────────┐ │ │
│ │ │ FeedForward (SwiGLU)            │ │ │
│ │ │                                 │ │ │
│ │ │ gate = silu(w_gate(x))          │ │ │
│ │ │ up   = w_up(x)                  │ │ │
│ │ │ down = w_down(gate * up)        │ │ │
│ │ └─────────────────────────────────┘ │ │
│ │                 │                   │ │
│ │                 ▼                   │ │
│ │         x = x + ffn_out             │ │ (残差 2)
│ └─────────────────────────────────────┘ │
│
├── norm: RMSNorm(D) ──────────────────────┤
│
└── output: Linear(D, V) ──────────────────┘ (共享嵌入权重)
```

---

## ⚙️ 关键超参数对照表

| 组件 | 预训练默认值 | SFT 典型值 | 说明 |
|------|-------------|-----------|------|
| **学习率** | 3e-4 | 3e-5 ~ 6e-5 | SFT 需更小学习率防灾难性遗忘 |
| **最大步数/轮数** | 5000 steps | 1-3 epochs | SFT 数据少,按 epoch 计算 |
| **Warmup** | 200 steps | 较短 | SFT 收敛快 |
| **Batch Size** | 32 | 16-32 | 视显存调整 |
| **Seq Len** | 512 | 512-1024 | 取决于对话长度 |
| **Loss Mask** | 无 (全位置) | prompt=-100 | SFT 核心差异 |
| **权重初始化** | 随机 | 预训练权重 | 热启动 |

---

## 📁 产物文件结构

```
out/
├── tokenizer.json           # BPE 词表 + 合并规则
├── smoke/                   # 冒烟测试产物
│   ├── ckpt/
│   │   ├── model.pt         # 模型权重
│   │   ├── meta.json        # ModelConfig
│   │   └── optim.pt         # 优化器状态 + RNG
│   └── best/                # 最佳验证 loss 模型
│       ├── model.pt
│       └── meta.json
├── sft/                     # SFT 微调产物
│   ├── ckpt/
│   │   ├── model.pt
│   │   ├── meta.json
│   │   └── optim.pt
│   └── best/
│       ├── model.pt
│       └── meta.json
```

---

## 🔗 脚本调用链路

```
用户命令
    │
    ├─► python scripts/prepare_data.py
    │       └─► minillm/tokenizer.basic_symbols() 等工具函数
    │
    ├─► python scripts/train_tokenizer.py
    │       └─► minillm.tokenizer.BPETokenizer.train() → .save()
    │
    ├─► python scripts/train.py
    │       └─► minillm.train.train() 
    │           ├─► minillm.data.TextDataset
    │           ├─► minillm.model.LLMModel
    │           ├─► minillm.train.build_optimizer / get_lr / save_checkpoint
    │           └─► minillm.evaluate.evaluate_loss
    │
    ├─► python scripts/eval.py
    │       └─► minillm.evaluate.evaluate_loss()
    │
    ├─► python scripts/generate.py
    │       └─► minillm.generate.Generator.stream() / generate()
    │           └─► minillm.model.LLMModel.forward() (with KV Cache)
    │
    ├─► python scripts/prepare_sft_data.py
    │       └─► 生成 data/sft_train.jsonl, data/sft_val.jsonl
    │
    ├─► python scripts/train_sft.py
    │       └─► minillm.sft.run_sft()
    │           ├─► minillm.train.load_model() (热启动)
    │           ├─► minillm.sft.SFTDataset
    │           └─► minillm.sft.evaluate_sft_loss()
    │
    └─► python scripts/serve.py
            └─► minillm.serve.create_app()
                ├─► minillm.generate.Generator
                └─► FastAPI + uvicorn
```

---

## ✅ 验收测试点

| 测试文件 | 验证内容 |
|----------|----------|
| `tests/test_tokenizer.py` | BPE 编解码往返、特殊字符处理、emoji 代理对 |
| `tests/test_data.py` | 数据集采样形状、固定种子可复现 |
| `tests/test_model.py` | KV Cache 数值等价性、RoPE 位置、参数量统计 |
| `tests/test_train.py` | 训练步数、checkpoint 保存/恢复、RNG 状态 |
| `tests/test_sft.py` | Loss mask 正确性、SFT 训练收敛 |
| `tests/test_generate.py` | 采样确定性、流式输出、KV Cache 一致性 |
| `tests/test_serve.py` | API 格式、ChatML 模板、流式 SSE |

运行: `pytest tests/ -x` (全部离线、秒级完成)