# 教程 06：HTTP 服务 —— OpenAI 兼容 API

对应代码：`minillm/serve.py`、`scripts/serve.py`。技术栈 FastAPI + Uvicorn。

## 1. 路由

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查，返回 200 与模型状态 |
| `POST /v1/chat/completions` | Chat Completions 兼容：messages 数组、`stream: true` 走 SSE |
| `POST /v1/completions` | 传统补全接口：裸 prompt |

响应字段（`id/object/created/choices/usage`）与 OpenAI 规范对齐，
`usage.prompt_tokens/completion_tokens` 由真实 tokenizer 计数得出，恒等式有测试校验。

## 2. chatml 开关

`--chatml` 挂载后，服务端用与 SFT 训练**同一模板**把 messages 拼成 prompt（docs/04 第 2 节），
保证微调后的模型在推理侧看到一致的对话格式；请求体也可用 `chatml` 字段单独覆盖。

## 3. 启动与验证

```bash
python scripts/serve.py --ckpt out/sft/best --chatml &      # 默认 127.0.0.1:8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minillm","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

SSE 流式以 `data: [DONE]` 收尾；以上行为均有 `tests/test_serve.py`（TestClient，无需真实端口）覆盖。
