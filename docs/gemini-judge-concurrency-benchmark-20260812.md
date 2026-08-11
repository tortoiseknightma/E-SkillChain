# Gemini 3.6 Flash Judge 并发容量实测（2026-08-12）

结论：active Final Judge 使用 8 个任务、`max_workers=8` 同时启动时，8/8 均在首次 provider 调用通过完整输出契约，实际峰值达到 `8 inflight`。任务错误率、首调错误率和服务错误率均为 0%，因此 8 是本轮已验证的安全并发下界。本次未测试 8 以上，不能解释为 AIFast 或 Gemini 的服务上限。

请求保持 production wire：AIFast OpenAI-compatible `gemini-3.6-flash`、`response_format={"type":"json_object"}`、`max_tokens=2048`、600 秒 timeout，省略 thinking、temperature、top-p 和 seed。输出经过 `parse_final_judge_output_v4` 严格校验；生产外层允许对空或无效 JSON 重试一次，但本轮 8 个任务均首调成功，因此没有验证“并发 8 下同时发生 repair retry”的容量。

## 实测结果

| 指标 | 结果 |
|---|---:|
| 任务数 | 8 |
| provider 调用数 | 8 |
| 首调成功 | 8 |
| 最终成功 | 8 |
| 重试 | 0 |
| 总错误率 | 0% |
| 服务错误（429/5xx/连接/超时） | 0 |
| 实际峰值 inflight | 8 |
| 墙钟时间 | 7.68 秒 |
| 单任务 median / p95 / max | 5.48 / 7.68 / 7.68 秒 |
| 输入 / 输出 tokens | 12,959 / 4,023 |
| 总吞吐 | 约 132.6k tokens/min |
| reasoning metadata present | 8/8 |
| 人民币费用 | 未报告：仓库未冻结 AIFast 单价 |

机器可读 receipt：`runs/portfolio/gemini-final-judge-concurrency-8-20260812-v1.json`。receipt 仅保存响应与 request ID 的 SHA-256、usage、延迟和错误分类，不保存模型正文、图片编码或凭据。

## 运行建议与边界

- 新的 Portfolio 运行配置可以把 Final Judge 并发 8 作为已实测起点，同时继续记录首次尝试与最终任务错误率。
- production 的单次 HTTP 调用仍保持 SDK retry=0；只有外层固定规则可以对空或无效 JSON 重试一次，不能扩大为无界重试或挑选最好结果。
- 本次是 exploratory capacity probe，没有消费正式实验授权，也不关闭价格/BOM、远程处理授权或 Judge—human calibration 缺口。
- AIFast 返回的精确模型名仅证明第三方网关响应契约，不能描述成 Google provider-attested 身份。
