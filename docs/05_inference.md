# 教程 05：推理与采样 —— 从 logits 到文本

对应代码：`minillm/generate.py`，入口 `scripts/generate.py`。

## 1. 自回归解码循环

```
prompt tokens → [模型前向(带 KV Cache) → 最后位置 logits → 采样一个 token] × N
```

每一步只有**新 token** 参与前向计算，历史 k/v 从缓存读取（见 docs/02 第 5 节）。

## 2. 采样策略（temperature → top-k → top-p）

1. **temperature**：`logits / T` 后再 softmax。T→0 趋于 argmax（贪心），T=1 原分布，T>1 更随机；
2. **top-k**：只保留概率最大的 k 个候选，其余置 -inf——砍掉长尾垃圾词；
3. **top-p（nucleus）**：按概率降序累加到刚超过 p 为止截断——自适应候选数量；
4. **multinomial 抽样**：在截断后的分布上按概率抽签；固定 seed 结果可复现（有测试锁定）。

greedy 模式 = temperature≈0 + top-k/top-p 关闭，直接取 argmax。

## 3. 停止条件

生成遇到 `<eos>` 立即停止（多余算力不浪费）；或达到 max_new_tokens。
流式接口 `stream_generate` 逐 token yield 解码片段，分片拼接与一次性 `generate` 完全一致
（`tests/test_generate.py` 锁定）。

## 4. 运行

```bash
python scripts/generate.py --ckpt out/smoke/best --prompt "人工智能" \
    --max-new-tokens 64 --temperature 0.8 --top-k 50 --top-p 0.9 [--stream]
```

