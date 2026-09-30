"""minillm: 从零实现的小型 Transformer LLM 训练-推理全流程工程。

模块结构:
    config      -- 模型/训练配置
    tokenizer   -- BPE 分词器(训练/编解码/持久化)
    model       -- GPT 风格 Decoder-only Transformer(RoPE/RMSNorm/SwiGLU/GQA/KV Cache)
    data        -- 数据加载与采样
    train       -- 预训练循环(Cosine+Warmup / AdamW / 梯度裁剪 / AMP / checkpoint)
    generate    -- 推理采样(greedy / top-k / top-p / temperature)+ KV Cache
    evaluate    -- 验证集困惑度评估
    serve       -- FastAPI OpenAI 兼容推理服务
"""

__version__ = "0.1.0"
