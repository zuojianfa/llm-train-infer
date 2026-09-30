# 教程 02：Transformer 主干 —— 逐个零件拆解

对应代码：`minillm/model.py`、`minillm/config.py`。架构为 Llama2 风格 Decoder-only。

数据流（batch=B, 序列长=T, 隐藏维=d）：

```
token ids (B,T) → Embedding (B,T,d)
  → [N × (RMSNorm → GQA注意力 → +残差 → RMSNorm → SwiGLU FFN → +残差)]
  → RMSNorm → Linear 投影到词表 (B,T,V)
```

## 1. RMSNorm
LayerNorm 的简化版：只除均方根、不减均值，`y = x/rms(x) * w`。更快且效果相当。

## 2. RoPE 旋转位置编码
不用加性位置向量，而是把 q/k 每两维看作平面里的向量，**按位置角度 θ·pos 旋转**。
两个向量的点积只依赖**相对距离**——外推性好。工程上用实数实现（sin/cos 查表），
避免复数张量在 autocast 下的兼容问题。

## 3. 因果自注意力 + GQA
`softmax(QKᵀ/√d + causal_mask) · V`。causal mask 保证位置 t 只能看到 ≤t 的过去
（否则训练会"作弊"看到答案）。GQA：多个 Q 头共享一组 K/V 头，
KV Cache 体积缩小 heads/kv_heads 倍。

## 4. SwiGLU FFN
`down( silu(gate(x)) * up(x) )`：门控 MLP，比 ReLU 两层强；hidden_dim≈8/3·d 使参数量对齐传统 4d。

## 5. KV Cache
解码时每步只算新 token 的 q/k/v，历史 k/v 缓存在张量里拼接复用，
把每步 O(T²) 重算降为 O(T)。**正确性锚点**：增量前向必须与全量前向逐位一致
（`tests/test_model.py::test_kv_cache_matches_full_forward`）。

## 6. 其他细节
- 输入输出 embedding 权重绑定（weight tying）省一半词表参数；
- logit soft-cap：`cap*tanh(logit/cap)` 防 bf16 下 logits 爆炸；
- 残差缩放 `1/√(2N)`：深层堆叠时稳住方差。
