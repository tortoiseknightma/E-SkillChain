# Qwen thinking 并发容量实测（2026-08-12）

结论：对固定 240-call Portfolio Feedback 批次，使用
`max_workers=240`、平滑启动 `8 requests/s`。本次实测的实际峰值为
`221 inflight`；这是当前完整工作量需要的最大有用并发，不建议无节制继续提高线程数。

生产请求保持 `qwen3.7-plus-2026-05-26`、`enable_thinking=true`、
`thinking_budget=2048`、`max_completion_tokens=4096`、严格 JSON Schema、
单次尝试和 600 秒 timeout。历史 selected48 运行的冻结并发 2 不回写；本结论用于新的
240-row Portfolio 工作。

## 实测结果

| 指标 | 结果 |
|---|---:|
| 调用数 | 240 |
| 成功 | 236 |
| 总错误率 | 1.67% |
| 服务错误（429/5xx/连接/超时） | 0 |
| 严格 parse error | 3 |
| `finish_reason=length` | 1 |
| 实际峰值 inflight | 221 |
| 墙钟时间 | 83.9 秒 |
| 单请求 median / p95 / max | 29.7 / 46.4 / 54.0 秒 |
| 输入 / 输出 tokens | 860,880 / 376,177 |
| 费用 | CNY 4.731176 |

本轮 1 次 length 与 3 次 parse failure 都是在 provider 已返回响应后发生的输出契约问题，
不是并发基础设施错误。包括预检、失败探针与阶梯确认在内，本阶段已记录费用为
CNY 7.206136；早期 2 个 parse-only 探针因旧版探针未保留 usage，实际总费用略高，
按相邻调用估计仍低于 CNY 7.3。

按本批次总 token 与墙钟时间计算，实际处理速率约 884k tokens/min，低于该模型快照
公开的 1M TPM；8 requests/s 也低于 600 RPM 折算的 10 requests/s。线程数 240 并不
意味着 240 个请求同时起飞：平滑启动与提前完成使实际峰值停在 221，并避免启动突发。

机器可读原始 receipt：
`runs/portfolio/qwen37-feedback-concurrency-probe-240-20260812-v1.json`。

## 运行建议

- 240 条固定批次：`max_workers=240`，启动速率 `8 requests/s`。
- 接受门：总错误率不超过 2%，服务错误率必须为 0；parse/length 作为真实负结果落盘，
  不通过重试挑选更好输出。
- 预计完整批次约 1.5 分钟，而不是在并发 2 下等待约一小时；真实生产 prompt 或图片若
  显著增大 token，应按 1M TPM 下调启动速率，不能只提高并发。
- 若批次数超过 240，保持 240 inflight 上限与 8 requests/s 启动速率，先观察 TPM、
  p95 与服务错误；不要把本次结果外推为账号的无限并发能力。
