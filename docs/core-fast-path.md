# Core 1,500 Fast Path

`Core Fast Path` 是 Core r3 `1,500` 条数据上的默认 Portfolio 实验入口。它与历史
Formal Core 执行链并存，但不读取或创建 authorization、control、launch、reservation、
claim、bound、receipt、目录树哈希或多级 gate 产物。

## 固定实验几何

```text
dev_mini 200：固定 smoke24
opt_pool 800：Static 失败归因、canary12、body48 和候选输入
val 200：S1 / S2 / S3 各一次接受或回滚
test_frozen 300：Bank 冻结后的五配置最终评测
```

最终输出包含 `val 200×5=1,000` 和 `test 300×5=1,500` 条逻辑结果。回滚配置是
parent 的精确 alias，不产生新的 Assistant、工具或 Judge 调用。全 Core 只额外补齐
`LLMStaticSkill / S1 / S1+S2` 的 route-only 结果，不执行 `1,500×5` 端到端矩阵。

冻结配置位于 [`specs/core-experiment-fast-v1.json`](../specs/core-experiment-fast-v1.json)。
其中包含字面 canary12、smoke24、body48 ID，固定模型角色、Gate、并发度和 CNY 250
总费用上限。`CORE_FAST_OPT_STATIC_RESULTS` 必须指向已完成的 opt800 Static 归一化结果；
Fast Path 不会重新选样或读取旧 Creator240。

## 命令

PowerShell：

```powershell
$env:CORE_FAST_OPT_STATIC_RESULTS = 'E:\path\to\opt800-static-observations.jsonl'
uv run python scripts/run_core_experiment.py validate
uv run python scripts/run_core_experiment.py run --through s1
uv run python scripts/run_core_experiment.py run --through s2
uv run python scripts/run_core_experiment.py run --through full
uv run python scripts/run_core_experiment.py run --through test
uv run python scripts/run_core_experiment.py report
```

`run --through full` 在冻结 S1/S2/S3 后立即补齐 NoSkill val200，并写出五配置
`val-results.jsonl` 的 1,000 条逻辑结果；`run --through test` 再生成 test300、Final
Judge、全 1,500 route-only 补齐和最终报告。Static 的 opt800 路由直接复用
`CORE_FAST_OPT_STATIC_RESULTS`，不会重复付费调用。

使用同一 `--output-root` 重复执行会自动续跑。完整 `result` 直接复用；只有 `intent`
而没有 `result` 的调用会一次性终结为 `interrupted_unknown`，不会自动重放。`report`
只读取已完成调用并重建逻辑矩阵、指标和案例，缺少物理结果时直接失败，不会产生费用。

`validate --inputs-only` 跳过凭据和命令可用性检查，但仍严格验证 Core 1,500、固定
split、样本 ID、Static Bank、opt800 结果和 route attribution。正式 `validate` 还检查
每个 provider 凭据、执行 bridge 和最坏调用量估算不超过 CNY 250。

## 执行适配边界

默认 spec 使用
`skillchain.evaluation.core_fast.live_adapter:create_adapter`：Assistant 直接使用
`CoreFastAssistantRunner` 与现有真实工具/GCS，Feedback 使用现有严格 Schema/parser，
Creator 启动一次临时 Codex CLI session，Judge 使用现有 packet、prompt 和 parser；所有
SDK 自动重试关闭。若要替换 provider，可改为
[`scripts/core_fast_call.py`](../scripts/core_fast_call.py) 这个极薄的 JSON stdin/stdout
bridge，并设置 `CORE_FAST_CALL_FACTORY=module:function`。工厂返回对象必须实现：

```python
invoke(CallIntent) -> CallResult
```

每次 `invoke` 只能进行一次 provider/execution attempt，必须关闭 SDK 自动重试，并把
requested/returned model、原始输出、Schema 结果、token、费用、延迟和失败原因写入
`CallResult`。五种 role 为 `feedback / creator / assistant / judge / route_only`。调用
payload 已包含固定 Query、Bank、Schema、runtime paths，以及 S3 的
`reuse_parent_route_and_tool`。适配器必须复用现有 Assistant + 真实工具 + GCS、Feedback
Schema 和 Final Judge parser；不得重新接回旧 Portfolio launch/ledger 或 Core final
blocker。

Assistant 输出使用 `AssistantObservation`：`gcs_score` 是五个 GCS component 的逻辑与，
只能是 `0` 或 `1`，不是 component 平均分。Judge 输出的 `j_project` 使用项目既有
`0–100` 标度。S3 返回的 route decision、tool input/hash 和 tool trace 必须与 parent
一致，否则候选回滚。

不访问网络的完整编排测试使用内置 fake provider：

```powershell
uv run pytest -q tests/evaluation/test_core_fast.py
```

## 产物

```text
runs/core-fast/<experiment-id>/
├── manifest.json
├── calls/<role>/<call-id>.intent.json
├── calls/<role>/<call-id>.result.json
├── banks/{llm_static,s1-*,s2-*,full-*}.json
├── decisions/{s1,s2,full}.json
└── reports/
    ├── val-results.jsonl
    ├── test-results.jsonl
    ├── route-all1500.jsonl
    ├── route-all1500-summary.json
    ├── cases.jsonl
    └── summary.json
```

`summary.json` 报告 capability-macro/query-micro GCS、`J_project` 与四段差值、route
macro-F1/confusion、hard error、tool/card/evidence compliance、六类切片、调用/token/
费用/延迟，以及只用于报告的 paired leakage-group bootstrap。

## 结论边界

- Qwen3.8-Max 是 moving alias；必须同时记录 requested/returned model。
- `test300` 是冻结的 response-level holdout，不是真正 blind test。
- Core 图片只用于私有模型推理，不公开展示。
- 输出属于 Core Portfolio 实验，不声明论文正式复现或生产流量效果。

历史 Formal 入口和产物保持原样、只读兼容；它们不再是本入口的 Quickstart 前置。
