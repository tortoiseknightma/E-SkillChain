# Qwen3.8-Max thinking 并发容量实测（2026-08-12）

结论：在 active Feedback 生产 wire 上，60 个任务使用 `max_workers=60`、平滑启动 `8 requests/s` 时，实际峰值达到 `60 inflight`。59/60 请求通过完整输出契约，总错误率 1.67%，低于 2% 门限；服务错误率为 0%。因此 60 是本轮已验证的安全并发下界，但本次没有测试 60 以上，不能把它解释为服务上限。

生产请求保持 `qwen3.8-max`、`enable_thinking=true`、`thinking_budget=2048`、`max_completion_tokens=6144`、严格 JSON Schema、单次 provider 尝试和 600 秒 timeout。prompt 使用 GCS-aware v6 契约，`skill_suggestions` 还经过 policy-label 校验。

## 实测结果

| 指标 | 结果 |
|---|---:|
| 调用数 | 60 |
| 成功 | 59 |
| 总错误率 | 1.67% |
| 服务错误（429/5xx/连接/超时） | 0 |
| 严格 parse error | 1 |
| 实际峰值 inflight | 60 |
| 墙钟时间 | 65.1 秒 |
| 单请求 median / p95 / max | 54.6 / 59.2 / 62.6 秒 |
| 输入 / 输出 tokens | 237,300 / 168,824 |
| 处理吞吐 | 约 374k tokens/min |
| 费用 | CNY 8.925264 |

唯一失败在 provider 返回 `finish_reason=stop` 后产生：回答正文为空，严格解析失败；该调用有 814 个输出 token，说明模型只产生了 reasoning 而没有可解析正文。它属于真实负结果，不通过内部重试隐藏。

机器可读原始 receipt：`runs/portfolio/qwen38-feedback-concurrency-60-20260812-v1.json`。receipt 仅保存响应与 request ID 的 SHA-256、usage、延迟和错误分类，不保存模型正文或凭据。

## 运行建议

- 60 条固定批次可使用 `max_workers=60`，并以 `8 requests/s` 平滑启动。
- 接受门保持总错误率不超过 2%、服务错误率为 0；parse/length 算作真实失败。
- 本轮错误率只有一个样本余量，生产运行仍应保留错误统计与失败落盘；若追求更严格的成功率，需要单独设计受控重试策略并报告首次尝试错误率。
- 当前已授权实验 artifact 中冻结的 concurrency=2 不应被静默改写；采用 60 并发启动新的正式阶段时，应生成新的运行配置/授权绑定。
- 阿里云当前文档给出的北京地域 `qwen3.8-max` 额度是 30,000 RPM、5,000,000 TPM，并说明该 alias 使用动态软限流；本次实际 8 requests/s 与约 374k tokens/min 均明显低于这两个公开值。服务仍可能实施按秒限流，因此平滑启动必须保留。
