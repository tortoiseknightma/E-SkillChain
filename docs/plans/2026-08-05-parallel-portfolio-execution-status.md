# 200×5 并行执行状态（2026-08-05）

## 已冻结输入

- runtime：`runs/portfolio/portfolio-public-data-runtime-v26`
- launch：`runs/portfolio/portfolio-dev-mini-200x5-launch-v33`
- treatment、query、rubric、model、prompt 和评分逻辑均未因并行化修改。

## 实际执行

- v6 安全 canary 使用 Qwen=2、Kimi=1，验证了 shared-route 依赖、聚合 gate 和 append-only budget ledger；由于 Kimi 等待时间过长，保留账本后停止。
- 同一 execution root `runs/portfolio/portfolio-dev-mini-200x5-execution-v22` 使用 v7 profile 恢复，已提升为 worker=8、Assistant=8、Final Judge=4、Qwen=40 RPM；已结算调用只复用 checkpoint，不重放。
- `smooth_start_v1` 将上述 40 RPM 作为平滑启动速率执行，相邻 Qwen 请求至少间隔 1.5 秒；历史 profile 无需改写，加载时采用该安全默认值，Assistant 并发槽位不再产生冷启动突发。
- v7 当前状态以 execution root 的 ledger、attempt receipts 和 `tmp/v22-v7-live.out` 为准。未结算 reservation 按保守 accountable cost 计入，不自动释放或隐藏。

## 验证

- `tests/evaluation/test_portfolio_parallel.py`、`tests/evaluation/test_run_portfolio_matrix.py`、`tests/evaluation/test_portfolio_launch.py`：36 passed。
- parallel 相关源码通过 Ruff 和 compileall。
- Full 只在同 batch 的 S1+S2 stage 完成后进入；shard finalization 仍为串行，避免 run-state / ledger tail 竞争。

## 恢复规则

进程中断后的 reservation 不重放；恢复时生成 `orphaned_provider_call` receipt，并将保守 reserve 纳入 phase cap。只有完整 batch barrier 和 audit 通过后，才继续后续 batch。

2026-08-05 的 burst 修复后，`portfolio-dev-mini-200x5-execution-v23-recovery/parallel-profile-v10.json` 已绑定 `smooth_start_v1`、40 RPM 和 1.5 秒最小启动间隔。零调用校验通过；实际恢复在新 provider start 之前被历史 `dm-026` attempt receipts 的 `query_retryable_attempt_limit_reached` 拦截，monitor 已按 no-progress 规则退出，未通过删除 receipt 或修改历史结果绕过。

## 持续监控

`scripts/monitor_portfolio_execution.py` 已在同一 execution root 后台运行。它等待当前恢复进程结束，然后以同一 v7 profile 继续执行剩余 batch；达到 40 个 shard 完成、phase cap 或 8 次恢复上限时自动退出。状态记录在 `tmp/v22-monitor-status.jsonl`，matrix 输出记录在 `tmp/v22-monitor-matrix.out`。
