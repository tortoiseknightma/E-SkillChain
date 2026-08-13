# Core 1,500 Fast Path

`Core Fast Path` 是 Core r3 `1,500` 条数据上的默认 Portfolio 实验入口。它与历史
Formal Core 执行链并存，但不读取或创建 authorization、control、launch、reservation、
claim、bound、receipt、目录树哈希或多级 gate 产物。

## 固定实验几何

```text
dev_mini 200：固定 smoke24
opt_pool 800：discovery600 供失败归因、48 条 Feedback 选择和候选输入；replay200 只作 development
val 200：body_gate75 接受或回滚 S1；其余冻结分区供 S2 / S3 与组合检查
test_frozen 300：Bank 冻结后的五配置最终评测
```

最终输出包含 `val 200×5=1,000` 和 `test 300×5=1,500` 条逻辑结果。回滚配置是
parent 的精确 alias，不产生新的 Assistant、工具或 Judge 调用。全 Core 只额外补齐
`LLMStaticSkill / S1 / S1+S2` 的 route-only 结果，不执行 `1,500×5` 端到端矩阵。

冻结配置位于 [`specs/core-experiment-fast-v1.json`](../specs/core-experiment-fast-v1.json)。
其中包含字面 canary12、smoke24、body48 ID，固定模型角色、Gate、并发度和 CNY 250
总费用上限。默认 spec 已绑定 deterministic action-response v4 下 fresh Qwen3.7 Static 的 800 条
`AssistantObservation` 及其 SHA；旧 `export_core_fast_opt_static.py` 只用于迁移历史 Qwen-VL
artifact，不能生成或替代当前 Qwen3.7 基线。重新生成新 Static 必须使用独立 bootstrap spec、
fresh output root 与 `static-opt800` 命令，且会产生 provider 费用。

### S1 的两段边界

```text
Static opt800
→ discovery600 冻结 48 条 Feedback（canary6 + remaining42）
→ canary6 通过后执行 remaining42
→ 完整 48 条在 Creator 前通过 completeness/service/parse-schema 终止门
→ parent-bound sparse Creator
→ dev smoke24
→ replay200 development 筛查
→ 仅在通过后访问一次 body_gate75 接受门
→ 接受候选或 byte-exact 回滚到 Static parent
```

### Deterministic runtime 与 S1 treatment surface

`core-fast-deterministic-action-response-v4` 在路由后由 runner 固定工具序列和参数，并由纯函数
从 public DTO 编译 cards、evidence、section 与 fallback。机械 closure 不再属于 S1。
Creator 必须逐字复制 parent objective、tool steps、fallback instruction 与 citations；唯一可变且
被 runtime 消费的字段是 `core-fast-semantic-policy-v2`。已接受 R1 只开放 Document 的
`ocr_extraction_plan=literal-material-spans`，用于从公开 OCR 行选择 literal material span；
Description 与其余五项 Skill 均冻结。空 policy 或对 Multi/Exact 等无消费面的 patch 会 fail closed。

默认 Static SHA 为
`fb88a2145b6671de054229117f3b0bd2121d675fa26a6233d8c31471b5987502`，来自独立 create-only
root `D:\athena\experiment-runs\portfolio-core-qwen37-deterministic-v4-20260813\static-opt-run`。该 qualification
为 800/800 outer success、0 hard error、676/800 GCS success。

### Accepted typed semantic-policy R1（2026-08-13）

唯一变量是 Document 的 `ocr_extraction_plan=literal-material-spans`。完整 48 Feedback 在 Creator
前通过 terminal Gate；Creator 产出 parent-bound 单能力 patch。replay200 中 Document 从 `7/8`
升至 `8/8`，六能力 macro `+2.0833pp`；body_gate75 中 Document 从 `3/4` 升至 `4/4`，
六能力 macro `+4.1667pp`，hard-error delta `0pp`，component bootstrap 95% CI 下界 `0pp`。
decision 为 `accepted=true`、`alias_of=null`，selected Bank 为
`efc7cbb3a25daf22aee45b943b0ec4cdca42f3e49644bd53fafaad2a85c3aaef`。

S1 Gate 对 sparse treatment 采用能力局部的因果口径：实际 patch capability 且 parent/candidate
都路由到该 capability 的行使用 candidate observation；byte-exact inherit 的能力和目标路由不一致
行使用 frozen parent observation。数值阈值没有降低；此调整消除了冻结 Description 的独立路由采样
噪声，目标能力的真实回退仍全部保留。accepted run root 为
`D:\athena\experiment-runs\portfolio-core-qwen37-deterministic-v4-20260813\s1-r1e-document-literal-span`。
继续 S2 时必须搭配原 canonical spec
`D:\athena\experiment-runs\portfolio-core-qwen37-deterministic-v4-20260813\specs\s1-r1-document-literal-span.json`；
tracked default spec 内容虽已同步，但重新格式化后的文件 SHA 不同，会按预期拒绝旧 root resume。

`replay200` 的结果已经在历史 R0 中被观察，因此它只承担 development/过拟合筛查，
不能再被称为独立验证。其门为 capability-macro GCS delta `≥0pp`、hard-error delta
`≤+1pp`、每 capability delta `≥−3pp`。`body_gate75` 才是 S1 接受门：macro delta
`≥+2pp`、leakage-component bootstrap 95% CI 下界 `≥0pp`、hard-error delta `≤+1pp`、
每 capability delta `≥−3pp`。R0 未通过 replay，所以历史上没有访问 body gate。

### R0 诊断与 R1 保护

历史 R0 的整体 replay macro 从 `21.7729%` 上升到 `25.2749%`，但 Encyclopedia
从 `3/40` 降到 `1/40`。两条 Static-success 回退有不同机制：

- `r2-core-0695` 的候选仍走 fallback，但漏掉精确 `not enough evidence` marker，得到
  `fallback_contract_failed`；同时跳过了 `object_detect`，并在无来源时陈述了事实。
- `r2-core-0796` 保留正确工具序列和 fallback marker，但正文开头多出字面
  `<|begin|>`，得到 `output_section_invalid`；这是 Assistant 格式执行 lapse，不足以证明
  Encyclopedia Skill 内容本身有错。

旧 Feedback 还包含“无工具证据时补常识、物种或外部研究”等与 cited-evidence 合同冲突
的建议。R1 因此不再整 Bank 重写，而采用以下 fail-closed 边界：

- 六个 capability 都必须显式选择 `inherit|patch`，默认 `inherit`；每项都必须绑定
  parent Skill SHA，继承项保持 byte-exact。
- 只把 `[policy_compatible]` suggestion 交给 Creator；`requires_new_evidence` 和
  `rejected` 只留作诊断，不能触发 patch。
- 所有 Description/objective 冻结，R1 只允许受信编译器规定的 sparse authored fields；
  最多 patch 3 个 capability。
- Encyclopedia 是 R1 protected capability，必须继承 parent，不能因其他能力的高频失败
  再次被顺带改写。
- Creator 输出、Feedback bundle、parent Bank 和 AuthoringInput 都以 SHA 绑定；非法字段、
  工具序列漂移、缺少安全证据或 resume artifact 漂移均拒绝候选。

R0 的候选与 replay/rollback receipt 永久保留为历史诊断，R1 不修改或重解释这些字节。

### Qwen3.7 R1–R5 execution ledger 与最终 S1 disposition

Static opt800 已用 `qwen3.7-flash-2026-07-15` fresh 重跑；成功产物 800/800、SHA
`ced36fb3050612be9a45a9fcb0d0d835b0f7a40ef422009066b6680df7368fc0`。R1 使用
12 条 fresh Qwen3.8 Feedback，R2–R5 复用 SHA 绑定的相同 call results，新增 Feedback
provider call 均为 0。每轮只允许一个 capability patch，其他五项 byte-exact inherit。

| Round | 唯一候选边界 | 最深阶段 | Development 结果 | Disposition |
|---|---|---|---|---|
| R1 | Multi positive closure | smoke24 + replay200 | `8→9`，4 条 paired regression、7 个新 occurrence | 回滚；body 0-call |
| R2 | Multi tool-first + DTO closure | smoke24 + replay200 | `8→20`，但仍有 1 条 paired regression、3 个新 occurrence | 回滚；body 0-call |
| R3 | Multi DTO template | Creator | authored-content guard 拒绝 sparse proposal | 无 candidate；Assistant/body 0-call |
| R4 | Document literal line copy | smoke24 + replay200 | `0→0`，新增 3 个 contract occurrence | 回滚；body 0-call |
| R5 | Style evidence copy | smoke24 + replay200 | `7→26`，但有 2 条 paired regression、4 个新 occurrence | 回滚；body 0-call；S1 停止 |

R2 是最强 Multi 信号：净增 12 条且 tool-contract failure 总数 `15→5`，但唯一回退样本
跳过工具并编造 item/card handle。R5 是最强 Style 信号：净增 19 条，但分别出现漏掉精确
fallback marker、跳过工具并编造候选的回退。两者均不能用净增益覆盖新引入风险。R3 的
DTO 假设没有真正得到 Assistant 实验；Creator 外层成功，但指令被词法 guard 拒绝，不能
写成算法负结果或无授权重试。

五个 decision 最终均 `accepted=false`、`alias_of=llm_static`，selected Bank 为
`da5cfe1f93f2cb57b389c97738034cf10cab931dcc30e145acc6d8873ff1348a`。本次可追踪
provider 记录为 7,823 calls、CNY `3.9202578`，另有废弃 Static v1 的一条未知 orphan；
可用 v2 lineage 为 5,326 calls、CNY `2.6998944`。5 次 Codex Creator session 的人民币
cost basis 不可得。`body_gate75`、Final Judge、S2、val 与 `test300` 均未访问。

R1–R5 五次 Creator 额度已经耗尽，不追加第六轮，不放宽门槛，也不启动 S2。证据根为
`E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v2`。

### 授权后的 R6–R10 execution ledger

用户随后明确授权第二批最多五轮。运行前修复了 R3 的词法 guard 假阳性：路径禁词与路径
marker 现在必须局部关联，安全的业务文本不再被跨句拼接误杀；真实 `results/...` 等路径和
`eval/gold/judge` 等强禁词仍 fail closed。S1 decision 同时记录脱敏的稳定 rejection reason。

R6–R10 继续绑定同一个 Static v2、同一份 Feedback manifest
`4e5d7013665d5ba70ce7c92fbe71ba5c44a363477c1d19d2337552382fb88043` 和同一个
Static parent；每轮使用独立 output root、单 capability patch，其他五项 byte-exact inherit。

| Round | 唯一候选边界 | Target replay success | Paired 回退 / 新 occurrence | Disposition |
|---|---|---:|---:|---|
| R6 | Multi tool-first + exact public DTO copier | `8→18` | `2 / 4` | 回滚；body 0-call |
| R7 | Style tool-first + evidence/fallback serializer | `7→22` | `2 / 4` | 回滚；body 0-call |
| R8 | Recipe detect/lookup + source-only closure | `3→10` | `0 / 7` | 回滚；body 0-call |
| R9 | Exact image/text input latch | `17→17` | `2 / 2` | 回滚；body 0-call |
| R10 | Document literal OCR lines | `0→0` | `0 / 1` | 回滚；body 0-call；S1 停止 |

第二批 Assistant 共 1,120 个 outer、3,588 个真实 Qwen3.7 model calls、
`6,818,346 / 552,635` input/output tokens，新增可计费用 CNY `1.8057772`。所有 1,120
个 outer 结果均 success/schema-valid，3,588 个 provider request ID 全局唯一且模型身份一致，
没有 provider/capacity error；receipt 中的本地 response-contract/runtime failures保留在分母。
5 次 Creator 共 `167,532 / 5,584` tokens，人民币 cost basis 不可得。复制的 Feedback receipt
不重复计费。`body_gate75`、Final Judge、S2、val 与 `test300` 仍均未访问。

R8 是最接近保留的候选：没有 Static-success→candidate-failure，但仍在原失败样本上引入 7
个新 contract occurrence，因此零新增规则必须拒绝。第二批五次 Creator 额度已耗尽，不追加
R11。证据根为 `E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v3`。

## 实测并发与配速

| Role | 新 Fast Path 配置 | 当前阶段的实际并发 | 配速作用点 |
|---|---:|---:|---|
| Assistant | 60 workers；20 requests/s | ≤60 | 每次真实 route/action/body provider call |
| Qwen3.8 Feedback | 60 workers；8 requests/s | `min(60, 48)=48` | 每条冻结 Feedback provider call |

新 Assistant 的 60-call 极限轮为 60/60、峰值 inflight 60，未出现 429、5xx、连接或超时
错误。极限轮以 60 requests/s 启动；主实验使用 20 requests/s，以免账号级合并限流影响
其他调用。详见
[`qwen37-flash-assistant-concurrency-benchmark-20260812.md`](qwen37-flash-assistant-concurrency-benchmark-20260812.md)。

Qwen3.8 Feedback 的 60-call 实测使用 60 workers、8 requests/s，59/60 通过、服务错误为
0；60 是本次验证值，不是服务上限。当前 S1 只有固定 canary12，所以不会人为制造 60 个
调用来占满 worker。详见
[`qwen38-feedback-concurrency-benchmark-20260812.md`](qwen38-feedback-concurrency-benchmark-20260812.md)。

以上数值只绑定新建的 Core Fast execution overlay。历史 Formal Core 的 2/40/1.5s、
1/20/3s 等 profile、已有配置、授权、receipt 和实验结果均保持原样，不回写；并发调整也
不得改变模型、样本、重试或中间结果判定。

## 命令

PowerShell：

```powershell
uv run python scripts/run_core_experiment.py validate
```

当前只推荐上述零调用验证命令。以下付费/下游命令是编排接口示例，不代表任一已执行 lineage 仍有
执行授权；只有另行批准新 S1 lineage，且 S1 真实接受后才能按顺序使用：

```powershell
uv run python scripts/run_core_experiment.py run --through s1
uv run python scripts/run_core_experiment.py run --through s2
uv run python scripts/run_core_experiment.py run --through full
uv run python scripts/run_core_experiment.py run --through test
uv run python scripts/run_core_experiment.py report
```

默认 spec 已绑定新 Qwen3.7 Static v2，并明确标成 fresh R1 模板；`validate` 不产生调用，
但 `run --through s1` 会创建新的 Feedback/Creator/Assistant 调用。第二批五轮额度也已耗尽，
因此除非另行授权新的 lineage，不应执行该付费命令，更不能对任一失败 round 执行 S2。

`run --through full` 在冻结 S1/S2/S3 后立即补齐 NoSkill val200，并写出五配置
`val-results.jsonl` 的 1,000 条逻辑结果；`run --through test` 再生成 test300、Final
Judge、全 1,500 route-only 补齐和最终报告。Static 的 opt800 路由直接复用
spec 中 SHA 绑定的离线观察，不会重复付费调用。

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
- 五轮负结果只证明这些 sparse candidates 未通过 development screen；它不证明 S1
  方法普遍无效，也不把 development replay 写成 test 结论。R3 仅是 proposal guard 拒绝，
  没有产生可供算法判定的 Assistant 结果。
- Assistant fixed repair 目前只收紧 section 结构，非空 product candidate 与 card closure
  仍由 GCS fail-closed 检出；这个已记录的 P2 不影响本次拒绝结论，但不得借此追认或重跑 R3。

历史 Formal 入口和产物保持原样、只读兼容；它们不再是本入口的 Quickstart 前置。
