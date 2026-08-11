# Qwen Assistant 并发容量实测（2026-08-12）

结论：active Assistant `qwen3-vl-flash-2026-01-22` 的安全运行配置是 `assistant_concurrency=2`、平滑启动 `0.5 requests/s`（30 RPM）。60-call 有效轮全部成功，实际峰值并发为 2。继续提高 worker 数不会增加吞吐，因为该日期快照的 100,000 TPM 已经成为主要瓶颈。

探针交替执行 30 次图像工具选择和 30 次带工具证据的最终回答，使用生产调用参数：non-thinking、图像输入、function calling、`temperature=0`、`top_p=1`、`max_tokens=4096`、禁用 parallel tool calls、单次 provider 尝试和 180 秒 timeout。结果只保存响应/request ID 哈希、usage、延迟和错误分类，不保存模型正文或凭据。

## 有效轮：0.5 requests/s

| 指标 | 结果 |
|---|---:|
| 调用数 | 60 |
| 成功 | 60 |
| 总错误率 | 0% |
| 服务错误（429/5xx/连接/超时） | 0 |
| 工具选择 / 最终回答成功 | 30/30 / 30/30 |
| 实际峰值 inflight | 2 |
| 墙钟时间 | 119.7 秒 |
| 单请求 median / p95 / max | 1.06 / 1.98 / 2.29 秒 |
| 输入 / 输出 tokens | 186,810 / 4,112 |
| 实际吞吐 | 约 95.7k tokens/min |
| 费用 | CNY 0.03418950 |

机器可读 receipt：`runs/portfolio/qwen-assistant-concurrency-60-20260812-v2.json`。

## 诊断轮：1 request/s

最初按 60 RPM 启动，但捕获到的响应已经达到约 155.5k tokens/min，超过公开的 100k TPM。结果出现 11 个 429，证明只按 RPM 配速不足。该轮另外有 25 个本地 contract 失败，根因是初版探针对合成最终回答施加了不属于容量判定的语义标题检查；这些不是 provider 错误，已从有效轮判定中移除。诊断轮费用 CNY 0.02809095，receipt 为 `runs/portfolio/qwen-assistant-concurrency-60-20260812-v1.json`。

两轮共调用 120 次，合计费用 CNY 0.06228045。

## 运行建议

- 新 Assistant 运行采用并发 2、30 RPM（每 2 秒启动一次真实 HTTP 调用）。
- 必须按每次实际 provider call 限速，而不是按外层 query 限速；单个 query 可能包含 route 和多个 action calls。
- 现有历史 profile、receipt 和已经冻结的运行配置不回写。新建 execution overlay 时再绑定本次容量值。
- 总错误率门保持不超过 2%，服务错误率门保持 0%；429 不应通过无界重试隐藏。
- 阿里云当前文档给出的北京地域快照额度是 60 RPM、100,000 TPM，并提示还可能实施 RPS/TPS 和流量增速保护。有效轮的 95.7k tokens/min 只剩约 4.3% 余量，因此不建议高于 0.5 requests/s。
