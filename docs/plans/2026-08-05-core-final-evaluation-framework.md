# Core 最终评测配置框架（v1）

状态：`draft_waiting_for_dev_mini_diagnostics`
轨道：Portfolio Track
用途：在不启动 core、不调用模型的前提下，先固定最终评测中不能随结果变化的部分，并为当前 `dev_mini=200 × 5` 诊断后的一次配置细化保留受限入口。

## 1. 最终目标与边界

本框架面向“尽可能复现论文机制目标”的最终 core 评测，但仍遵守公开数据复现的诚实边界：它评价 Skill 创建、路由和正文优化带来的端到端变化，不声称复现企业生产流量、内部工具、线上 A/B 或论文 Table 2 的同名绝对分数。

框架本身只有配置权，没有执行权：

- `execution_authorized=false`；
- `model_calls_performed=0`；
- 不包含启动 core 的命令或开关；
- 即便所有 blocker 被关闭，也只会进入 `ready_to_freeze`，之后仍需单独生成并审批 launch/execution package。

## 2. 已固定、不可由 mini 结果改写的部分

| 项目 | 固定值 |
|---|---|
| Profile | `core` |
| 查询数 | 1,500 |
| 主配置 | `NoSkill / LLMStaticSkill / S1 / S1+S2 / Full` |
| Canonical 实例数 | 7,500 |
| Capability | 现有全部 6 个 |
| 分片几何 | 25 query/batch，60 batches，300 config shards |
| 质量主指标 | `J_project` 0–100，按固定 full denominator 计算 |
| 路由主指标 | canonical capability macro-F1；NoSkill 为 N/A |
| 不确定性 | leakage component 级 10,000 次 paired cluster bootstrap，95% CI |
| 路由配对检验 | McNemar |
| Canonical 运行 | 每个配置一次；同一 query、输入顺序、工具面和 action budget 配对 |
| 解封 | 单一 sealed test schedule，不暴露中间分数，不按结果重试或挑选 |
| 工具评测 | 全部 7 个 canonical tool；每工具 30–50 条真实 gold；Multi-Product 另含端到端 gold |
| Judge 可靠性 | 独立 50–100 条 grouped Judge-human audit，解封前冻结阈值 |
| 人评 | core 100–150 条 blind SBS，暂定 125；`Full vs LLMStatic`、`Full vs S1+S2` |

固定主比较为：`S1−LLMStatic`、`S1+S2−S1`、`Full−S1+S2`、`Full−LLMStatic`。`challenge_final` 如存在，必须与 core 主结果分开报告。

## 3. mini 后只允许细化的槽位

框架预留六个 decision slot，每个 slot 必须引用一个内容寻址的决策文件；不能直接修改固定契约。

1. `diagnostic_runtime_patch_set`：对称的 Assistant 输出/action 限制、Judge 契约与 retry、evidence projection；依据完整 200×5 诊断和独立 Judge-human audit。
2. `parallel_execution_profile`：Assistant/Judge 并发、RPM、active waves、circuit breaker；依据 mini、7,500-instance 零 provider stress 和小规模真 provider smoke。
3. `phase_budget_cap`：core 费用与 retry ceiling；必须有 owner budget authority。
4. `judge_and_tool_gate_thresholds`：Judge-human、工具 gold set、parser failure 阈值。
5. `optional_repetitions`：是否对预注册的 `NoSkill / S1+S2 / Full` 做两次重复；必须在解封前决定，不能看结果后追加。
6. `analysis_seeds_and_min_n`：bootstrap、人评抽样 seed、capability eligibility 和最小可报告样本量。

当前仅作为容量规划起点的 provisional baseline 是：Assistant concurrency 2、final Judge concurrency 8、Assistant 40 RPM、最多 10 active waves、core phase budget 候选上限 CNY 1,100。Assistant 的 40 RPM 必须由 `smooth_start_v1` gate 平滑执行，相邻 Qwen 请求启动至少间隔 1.5 秒，不能只检查滚动分钟总量后突发放行。它们没有执行效力，应由 mini 诊断后的 decision artifact 明确接受或替换。

## 4. 数据与运行时绑定

2026-08-05 合入 `master` 的 core acquisition/synthesis pipeline 及 `E:\skillchain-data` 外置数据根已经完成只读核验。tracked candidate 使用 portable root id `skillchain-data`，不把机器相关的盘符写入绑定身份；每个 binding 保存 root id、root-relative path、文件 SHA-256 和 byte count。

以下 8 项已经绑定并逐文件复核：

- r2 `plan/core.json` 与 `pre-generation-manifest.json`；
- r3 materialized `queries.jsonl`；
- r3 200-query owner audit 终审通过 receipt；
- capability assignments v9；
- asset catalog v9 的 manifest/assets/components。

数据核验结果为：1,500 条 plan、1,500 条唯一 query、60 个 25-query batch；split 为 `dev_mini/opt_pool/val/test_frozen = 200/800/200/300`；六 capability 为 `375/225/300/300/90/210`；867 个被引用资产全部存在于 1,022-row catalog 且本地文件无缺失。r3 queries 文件 SHA-256 为 `e2899c5b0813d796692ab175cabfc722e3f3a59f61b658adb939c2e283895e87`。

r3 的 200 条 owner audit 已于 2026-08-06 由 owner 明确终审通过。终审 decisions 为 200 `pass`、0 `needs_revision`、0 `reject`；decisions 文件 SHA-256=`967183c6f483b766ce825e97677e9dd425232022ca57ac4bae7f257b943510c4`，终审 receipt 文件/self SHA-256=`47a345b87cc00b31397a6766c526fee1f1bddb41de97db31f431e0624462b47c` / `94c0860e0c5117c38a006243452a6f24e19e30f677730f787617d089a553af91`。审核分钟数未提供，receipt 明确记录为 `not_reported`。该 corpus audit approval 与实验配置的 `owner_freeze_approval` 仍是两个不同角色。

以下输入仍保持 `pending`：

- remote-processing authorization/receipt、leakage/exclusion lock、test seal；
- 最终 runtime lock、treatment chain、四个 Skill Bank、Judge rubric、role selection、tool/environment lock；
- dev_mini completion/diagnostic decision、Judge-human audit、tool gate、预算审批、stress/smoke、人评计划和 owner freeze approval。

`challenge_final_seal` 是可选绑定，不阻塞 core 主结果冻结。

## 5. 使用方式

当前 tracked candidate 已包含上述 7 个 core data binding。若需要创建一个全 pending 的新候选用于测试（零调用、create-only）：

```powershell
uv run python scripts/prepare_core_final_evaluation.py `
  --output tmp/core-final-unbound.json
```

mini 结束后，从已有 artifact 增量绑定证据：

```powershell
uv run python scripts/prepare_core_final_evaluation.py `
  --base specs/evaluation/core-final-evaluation-framework-v1.json `
  --artifact-root D:\athena\ECommerceSkillChain `
  --artifact-root-id repository `
  --bind dev_mini_200x5_completion=path/to/completion.json@<sha256> `
  --output path/to/next-framework.json
```

slot 只能在其依赖的 evidence roles 已绑定后解析：

```powershell
uv run python scripts/prepare_core_final_evaluation.py `
  --base path/to/bound-framework.json `
  --resolve parallel_execution_profile=path/to/parallel-decision.json@<sha256> `
  --output path/to/resolved-framework.json
```

最后使用 `--require-ready-to-freeze` 做 fail-closed 检查。若仍有任何 required binding 或 refinement pending，命令返回 3 且不写文件。

## 6. dev_mini 完成后的细化顺序

```text
200×5 完整运行与诊断
→ 仅形成一次有记录的 runtime/算法修正决定
→ Judge-human 与 tool gold gate
→ 7,500-instance fake-provider stress
→ 小规模真 provider 并发 smoke
→ 冻结并发、预算、阈值、seed、可选重复和人评计划
→ 绑定 core 数据、Bank、runtime、代码与环境哈希
→ ready_to_freeze
→ 单独审批并生成 core launch/execution package
```

实现入口：`src/skillchain/evaluation/core_final_config.py`；编译入口：`scripts/prepare_core_final_evaluation.py`；聚焦测试：`tests/evaluation/test_core_final_config.py`。
