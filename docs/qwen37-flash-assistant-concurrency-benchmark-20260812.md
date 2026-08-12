# Qwen3.7 Flash Assistant 并发容量实测（2026-08-12）

结论：新 Assistant `qwen3.7-flash-2026-07-15` 在北京 DashScope 端点的
60-task 生产调用线压测中达到实际峰值并发 60，60/60 成功，服务错误率和总错误率均为
0%。60 已是 60-task 测试能够证明的并发上限；增加 worker 不能产生超过任务数的实际并发。

## 限流边界与运行配置

附件给出的北京区快照额度是 30,000 RPM / 5,000,000 TPM，且账号下 RAM 子账号、业务
空间和 API Key 合并计算；服务还可能按 RPS/TPS 和请求增速保护。不同模型额度相互独立。

- 极限验证：60 workers、60 requests/s 平滑启动，仅用于 60-task 容量探针。
- 主实验配置：60 workers、20 requests/s 平滑启动。
- 保守余量：沿用旧生产线约 3.18k tokens/call 的高估，20 requests/s 约消耗
  3.816M TPM，低于额度的 80%，保留约 1.184M TPM（23.7%）。
- 判定门：总错误率不超过 2%，服务错误率必须为 0%；本轮均为 0%。

## 60-task 极限轮

| 指标 | 结果 |
|---|---:|
| 调用数 | 60 |
| 成功 / 失败 | 60 / 0 |
| 429 / 5xx / 连接 / 超时 | 0 |
| 实际峰值 inflight | 60 |
| 启动速率 | 60 requests/s |
| median / p95 / max 延迟 | 11.128 / 14.613 / 17.360 秒 |
| 输入 / 输出 tokens | 72,330 / 6,561 |
| 费用 | CNY 0.0197148 |

调用交替覆盖 30 次图像工具选择和 30 次带 typed tool evidence 的最终回答，使用
non-thinking、图像输入、function calling、`temperature=0`、`top_p=1`、
`max_tokens=4096`、禁用 parallel tool calls、单次 provider 尝试和 180 秒 timeout。
receipt 仅保存哈希、usage、延迟和错误分类，不保存模型正文、图像内容或凭据。

机器可读 receipt：
`runs/portfolio/qwen37-flash-assistant-concurrency-60-20260812-v1.json`。

## 解释限制

这证明的是 60-task 范围内的最大并发和一次无错误突发，不代表可以无限期维持
60 requests/s。主实验必须使用 20 requests/s 的 provider-call 级平滑启动；如果同账号
还有该模型的其他调用，应共享同一个节流器或相应降低速率。历史实验身份和 receipt 不回写。
