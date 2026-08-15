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
其中包含字面 canary6、smoke24、body48 ID，固定模型角色、Gate、并发度和 CNY 250
总费用上限。默认 spec 已绑定 model-generated action-response 的 fresh Qwen3.7 Static opt800、
bootstrap 生成的 fixed samples 与 SHA `a9949cd673fff547…2c0805a7`；`validate --inputs-only` 与完整
`validate` 均通过。旧
`export_core_fast_opt_static.py` 只用于迁移历史 artifact，不能生成或替代当前 runtime 基线。
重新生成 Static 必须使用独立 bootstrap spec、fresh output root 与 `static-opt800` 命令，且会产生
provider 费用。

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

### Model-generated runtime 与 S1 treatment surface

当前 `core-fast-model-generated-action-response-v1` 删除了 deterministic response compiler。冻结
Description 完成路由后，runner 将所选 Skill Body 和允许的函数工具交给 action model；模型负责
工具选择、参数以及最终回答文本，runner 负责真实工具执行、可见证据记录、格式预检与最多一次固定
修复。DTO/card/evidence/fallback 不再由纯函数直接生成，也不再存在 `core-fast-semantic-policy-v2`
的解析或渲染路径。

S1 保留六个独立 Creator 会话，但每个 capability 的结构化提案拆成两个互斥 surface：

- `action-policy`：只描述 tool-first、公共参数构造、继续/重试和停止条件。screen 复用 Static 的
  selected route，不复用任何 tool/answer，重新执行完整 action/tool loop；成功指标是
  `route_acceptable ∧ tool_contract_pass`，protected-success 也按这一谓词定义。回答格式错误不会污染
  action 归因；tool error、action runtime/timeout 或固定 route 漂移仍属于严重度升级并硬拒绝。
- `response-policy`：只描述 visible evidence、cards、answer sections、uncertainty 与 fallback。
  screen 复用 Static route/tool/scorer evidence（包括固定的 tool-error 分支），只发起 answer model call；成功指标是
  `no_hard_error ∧ evidence_grounded ∧ output_contract_pass`，protected-success 按该谓词定义。

两类 surface 各自执行有界风险筛查并可独立回滚；只有通过的 surface 才在本 capability 内组合，随后
六能力 fan-in。Description、工具集合、compiler-owned Body headings 和另外五个 Skill 保持冻结。完整
fan-in replay 再恢复模型自主 route/action/tool/answer，因此最终接受仍覆盖两类 policy 的交互风险。

default spec 不复用任何 deterministic v4/v5/v6 Static。新的 model-generated Static 已 800/800
完成；no-op qualification 在 137 条 treatment-reached replay 上为 1 gain / 0 regression 并通过。
旧 accepted Bank 仍不能跨 runtime 当作新 parent。

### Model-generated fresh lineage 与 30 轮 S1 终态（2026-08-14）

证据根为
`D:\athena\experiment-runs\portfolio-core-qwen37-model-generated-v1-20260813`。fresh Static 为
168/800 GCS success、135/800 hard error，成本 ¥1.2151480；authoritative no-op root 为
`noop-qualification-v4`，成本 ¥0.2627260。新的 R1–R10 合计 456 Feedback result、42 个独立
Codex Creator result 和 2,781 Assistant result，可追踪 DashScope 成本 ¥5.2082646；Creator
人民币成本不可得。十轮都 `accepted=false`，selected Bank 均回滚为 Static
`da5cfe1f…1348a`，没有访问 S2、Judge、val 其余分区或 test。

| 轮次 | 机制增量 | 最深结果 | 结论 |
|---|---|---|---|
| R1 | contrastive 48 | canary 5/6 | parse gate 停止；0 Creator/Assistant |
| R2 | adaptive confirmation v1 | Style 局部 +1/0；组合 replay −0.4065pp | 回滚 |
| R3 | stable fan-in | full48 47/48，含 1 service error | full-batch gate 停止 |
| R4 | role-separated packet | Multi + Style 局部保留；replay +0.9812pp；body 0 | 覆盖最广，仍回滚 |
| R5 | failure-rich selection | canary 5/6 | JSON parse gate 停止 |
| R6 | supported-cluster selection | 无保留分支 | 回滚 |
| R7 | attributed Feedback | Multi replay +3.4483pp；body +10pp；总体 body +1.6667pp | 最接近接受，低于 +2pp，回滚 |
| R8 | single-step replace | Multi replay +3.4483pp；body 0 | 回滚 |
| R9 | rejected-edit memory | Multi +1/−1，净 0 | 回滚 |
| R10 | attributed + single-step + memory | Multi +1/−2；无保留分支 | 回滚并停止 |

上述 R1–R10 是 dual-policy campaign 之前的 model-generated 开发历史。随后新的 30 轮 campaign
使用 48-row balanced Feedback、六个独立 Creator、action/response 两类 screen 与 capability fan-in，
全部从 Static `da5cfe1f…1348a` 独立派生。R12 的 Style action-policy 是唯一 accepted 候选：
replay macro +0.4065pp，body75 macro +2.0833pp，Style +12.5pp，hard error −1.3333pp，selected
Bank `e70ed907…096cd`。其余 29 轮全部回滚或在 Feedback/local gate 停止。

Balanced top 10 冻结为 R12/R17/R26/R29/R30/R4/R9/R13/R14/R21，覆盖 Encyclopedia、Exact、
Multi、Style、Recipe 五项能力；Document 没有任何候选进入 body gate，不能靠纳入 invalid 轮次虚增覆盖。
R15 虽访问 body，但 oracle coverage 不完整，因此保留在全量 ledger、排除出 eligible top 10。
排名不是 acceptance override：只有 R12 deployable。R17/R26 的 Multi+Recipe body 增益更高，但
hard-error 超限；R29 的 Multi+Style hard-error delta=0，body macro 仅 +1.6667pp。canonical 排名
位于 `D:\athena\experiment-runs\portfolio-core-dual-policy-campaign-20260814-v2\campaign-top10.json`，
完整逐轮事实、成本与边界见 [S1 实验日志](s1-experiment-log.html)。

### R12 自适应 30 轮与一次性 test300 终态（2026-08-15）

R34–R63 共 30 个前向 round 已全部终结，所有独立候选都只从 R12 Bank
`e70ed907…096cd` 派生；R17/R26/R29 和其他 rejected Bank 从未成为 parent、未复制局部 Skill、
也未进入 fan-in。R52 是唯一新增 `AcceptedBranchArtifact`：Encyclopedia response-policy 在
replay200 为 macro `+0.8333pp`，在 body75 为 macro `+3.125pp`、CI95 lower `+2.7778pp`、
hard-error `−4pp`。其余 round 均在 Feedback、局部有界风险、replay 或 body 正式门回滚。

最后的 B09 在任何调用前同时冻结 R61–R63，并以零调用 preflight 证明 3/3/3 evidence、surface
separability、结构与行为 treatment sensitivity。R61 因 live Feedback projection 未识别新 v9 attribution
label 而在本地 canary3 停止，0 provider/Creator/Assistant，不能作为算法结论。前向修复后，R62 的
Recipe stop-after-first-success action rule 局部为 9 gains / 1 regression，replay Recipe `+6.6667pp`、
macro `+1.1111pp`，但 body75 macro `0pp`、hard-error `+1.3333pp`，因此回滚。R63 的
Encyclopedia stop rule 局部为 5 gains / 8 regressions、net `−3`，未进入正式 replay。B09 计费
DashScope 为 `¥1.24871935`，2 个 Creator 会话，Creator 人民币成本不可得。

accepted-only finalization 没有可与 R52 组合的 Recipe 分支，因此直接冻结 R52 为唯一 finalist。
唯一一次 paired test300 共运行 R12 与 R52 600 个 outer query，成本 `¥0.9376664`：R52 的
Encyclopedia `+9.8361pp`，capability macro `+1.6393pp`，CI95 lower `0pp`，hard-error `+1pp`。
它只因未达到冻结的 macro `+2pp` 门而失败，最终 `cycle-selected` 回滚为 R12。该 test300 已永久
消费，不能再描述为 untouched 五配置 test；本次没有 Judge、S2、S3 或五配置矩阵调用。

前向阶段决定不改写上述事实：R52 现作为进入 S2 前的正式预备分支。后续 S2 的 working parent
绑定 R52 Bank `67b92b61…55bc8`，且只能修改 Description；R52 的 Body 与 accepted S1 evidence
必须保持 byte-exact。该身份表示“允许作为 S2 优化输入”，不表示 R52 已通过 test300 或已部署。
在新的 S2 route gate 正式接受候选前，Portfolio selected Bank 仍是 R12；S2 失败也回退 R12，
不把 R52 自动提升为最终 Bank。机器可读边界见
[`specs/s2-r52-preparatory-branch-v1.json`](../specs/s2-r52-preparatory-branch-v1.json)。

### Adaptive S2 runtime（离线 ready，真实调用尚未开始）

S2 不再复用旧的一次性 `s2-route-optimizer-once`/hybrid attribution 路径。新的 forward-only
round 绑定 R52 或上一轮 accepted S2 Bank、source decision、source manifest、机器可读的 R52
preparatory authorization，以及该 working parent 的 fresh route-only opt800。每轮只允许一个
target capability；Creator 不能自由重写 Description，只能输出一条
`when / route_to / must_preserve_query_ids` typed rule。runtime 将它规范化追加到目标
Description，并强制其余五个 Description 以及六个 Skill 的 Body/operators/static refs byte-exact。

无 provider 的 `prepare-s2-round` 从 discovery600 选择一个 dominant confusion cluster：3 个失败、
3 个 parent-success、3 个历史 regression/额外 parent-success；不足 9 条时在 Creator 前停止。
候选先在 route-only replay200 上接受有界风险筛查：`gains >= 1`、`net >= 1`、
`regressions <= 2`、`gains >= 4 × regressions`，且 6 个显式保护例必须 0 regression。通过后才执行
smoke24 与冻结 `route_gate75` 的 parent/candidate 各 75 条完整 Assistant；不再误用 val200。
accepted S2 Bank 可以成为下一轮 parent；rejected round 只回滚到 working parent，不能作为新
source。无论 working branch 如何，Portfolio selected 在新的最终选择前仍为 R12。

初始 R52 parent 的准备顺序如下；`<profile-root>` 与 `<round-root>` 必须是不同的新目录：

```powershell
uv run python scripts/prepare_s2_adaptive_round.py bootstrap-spec `
  --source-spec D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b07-20260815\specs\r52.json `
  --source-root D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b07-20260815\runs\r52 `
  --source-stage s1 --source-round-id r52 `
  --preparatory-binding specs\s2-r52-preparatory-branch-v1.json `
  --route-results-path <route800.jsonl> --output-spec <bootstrap-spec.json> `
  --experiment-id <experiment-id> --cycle-id <cycle-id> --round-id s2r1 `
  --target-capability <capability>

uv run python scripts/run_core_experiment.py --spec <bootstrap-spec.json> `
  --output-root <profile-root> s2-parent-route800

uv run python scripts/prepare_s2_adaptive_round.py freeze-spec `
  --bootstrap-spec <bootstrap-spec.json> `
  --route-bootstrap <profile-root>\s2-parent-route800-bootstrap.json `
  --output-spec <round-spec.json>

uv run python scripts/run_core_experiment.py --spec <round-spec.json> validate
uv run python scripts/run_core_experiment.py --spec <round-spec.json> `
  --output-root <round-root> prepare-s2-round
uv run python scripts/run_core_experiment.py --spec <round-spec.json> `
  --output-root <round-root> s2-readiness
uv run python scripts/run_core_experiment.py --spec <round-spec.json> `
  --output-root <round-root> run --through s2
```

parent route800 对同一 accepted parent 可跨 rejected rounds 复用；只有 accepted S2 改变了 Bank 时才
需要新的 route800 profile。按当前保守单价，profile 最坏约 CNY 4；完成 profile 后单轮 ceiling 为
206 route-only + 174 Assistant + 1 Creator，DashScope 预计 CNY 6.25。两段必须分阶段汇报，避免把
合计 CNY 10.25 放入同一个自主预算窗口。`s2-readiness` 本身 0-call，并保持 S3、Judge、已消费的
test300 与五配置矩阵 sealed。

以下 v4 accepted lineage 是前向开发历史；v5/v6 fan-out 又为 Multi/Exact 增加 typed selector，并分别
使用 fresh Static lineage 评估。三条 lineage 彼此只读，不能互相 resume 或追溯重判。

以下 v4 Static SHA 只属于历史 qualification：
`fb88a2145b6671de054229117f3b0bd2121d675fa26a6233d8c31471b5987502`，来自独立 create-only
root `D:\athena\experiment-runs\portfolio-core-qwen37-deterministic-v4-20260813\static-opt-run`。该 qualification
为 800/800 outer success、0 hard error、676/800 GCS success。

### 历史 accepted typed semantic-policy R1（v4，2026-08-13；只读）

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

### 六能力 fan-out / fan-in：10 轮结果（2026-08-13）

accepted v4 Document Bank 保持只读；本阶段没有继续 S2。新的 v5/v6 lineage 把 typed
semantic-policy treatment surface 扩到全部六个 capability，并真实运行了以下 fan-out/fan-in：

```text
balanced discovery Feedback
→ 6 个 capability branch（各自 parent-bound Creator）
→ 各分支 smoke24（相同固定集合；本能力候选 + 其余五项 byte-exact inherit）
→ 各分支只在本 capability 的 replay 子集上复用 Static route/tool trace
→ 独立 screen：gains≥1；gains−regressions≥1；regressions≤2；gains≥4×regressions
→ failure reason 迁移只记诊断；普通失败→hard/runtime failure 或 trace drift 仍硬拒绝
→ 只组合通过分支，另外能力 byte-exact inherit
→ 组合 Bank 重跑完整 replay200
→ 通过后才访问一次 body_gate75
→ accepted fan-in Bank 或整体回滚到同一 parent
```

这里的“并行”是独立的算法分支与证据账本；调度器仍可按共享 provider 并发/限速串并行执行，
不能让六个分支分别绕过全局容量。Creator 不再输出 whole-bank 自由重写：每次调用只能 patch
其分支的 model-generated Body authoring fields，另外五项必须绑定同一 parent 且保持 byte-exact。
fan-in 会复核 branch
Bank、编译 receipt、screen hash 与 parent identity，拒绝携带其他能力变化的分支。

以下结果全部属于已删除的 deterministic v5/v6 runtime，只作历史证据。v5 对 Exact/Multi 新开放的是公开 DTO 上的 evidence-term selector：它可以把不满足语义条件的
candidate 降为 unresolved/剔除，但不能改变 item coverage、card 字段、handle、工具顺序或
fallback。空 selector 与 v4 行为等价。由于 runtime contract identity 已变化，新一轮必须使用
`prepare_core_fast_qwen37_lineage.py bootstrap-spec` 生成 create-only fresh Static opt800，再用
`freeze-fanout-r1` 绑定 SHA/fixed samples。实际 fresh Static 如下：v5 为 800/800、GCS
675/800、SHA `d335c076…d345`；v6 为 800/800、GCS 675/800、SHA `f2cd96ad…3c97`。
v6 进一步只暴露 compiler 最终引用的 cards，并在 common-trace replay 中把空检测 fallback
保留为普通 GCS failure，而不是伪造 provider/oracle gap。

10 轮终态：

| Round | 机制变量 | 最深阶段 | 终态 |
|---|---|---|---|
| R1 | 48-row stratified Feedback | full Feedback gate：46/48，4.17% parse | 停止，Creator 0-call |
| R2 | 60-row stratified Feedback | full gate：55/60，8.33% parse、1.67% service | 停止，Creator 0-call |
| R3 | 60-row contrastive Feedback | full gate：58/60，3.33% parse | 停止，Creator 0-call |
| R4 | contrastive + one fan-out Creator | Document replay primitive 误产生 8/8 oracle gap | 机制失败，回滚 |
| R5 | 六个独立 Creator + common-trace replay | Document replay +2.0833pp；body +4.1667pp | accepted；备份 |
| R6 | v6 card closure + narrow selectors | replay +2.0833pp；body 0pp | 整 Bank 回滚 |
| R7 | selector regularization | full gate：58/60，3.33% parse | 停止，Creator 0-call |
| R8 | 空 selector + Document literal spans | replay +2.0833pp；body +4.1667pp | accepted |
| R9 | failure-heavy Feedback | canary：5/6，16.67% parse | 停止，Creator 0-call |
| R10 | compact 48-row contrastive packet | 48/48；replay +2.0833pp；body +4.1667pp | accepted；最终选择 |

R8 与 R10 的有效 Document Skill 字节相同，Skill SHA 为 `ec6ae462…6fdf`。R10 的 selected
Bank 为 `51ae438e9ef9b4bad5d2809a7d0333e2627ff5ebd6ddd4aad6daeffbd53ac6cf`；相较 R8，
它用 48 而不是 60 条 Feedback，且 48/48 schema-valid，因此作为后续 S2 的备选起点。
R10 其余五个 Creator 返回 no-op typed policy，由 sparse compiler 在 replay 前拒绝；这不是
五能力的负效果，也不构成虚假的覆盖。R10 的 replay/body 只有 Document 改变，其他能力 delta
均为 0，hard-error delta 为 0，body component-bootstrap 95% CI 下界为 0。

`replay200` 的结果已经在历史 R0 中被观察，因此它只承担 development/过拟合筛查，
不能再被称为独立验证。新 lineage 的门为 capability-macro GCS delta `≥0pp`、hard-error delta
`≤+1pp`、每 capability delta `≥−5pp`。`body_gate75` 才是 S1 接受门：macro delta
`≥+2pp`、leakage-component bootstrap 95% CI 下界 `≥0pp`、hard-error delta `≤+1pp`、
每 capability delta `≥−5pp`。已冻结的旧 prompt-only R0–R10 与 accepted v4 decision 仍按当时的
`−3pp`/零回退策略解释，不追溯重判；R0 未通过 replay，所以历史上没有访问 body gate。

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
- 在该历史 R1 中所有 Description/objective 冻结，受信编译器最多允许 3 个 sparse patch；
  新 fan-out 每个 branch 仍只允许 1 个 patch。
- Encyclopedia 在该历史 R1 中受保护；新 fan-out 取消 whole-bank 联动，改为 Encyclopedia
  独立分支、独立筛查，失败只回滚本能力。
- Creator 输出、Feedback bundle、parent Bank 和 AuthoringInput 都以 SHA 绑定；非法字段、
  工具序列漂移、缺少安全证据或 resume artifact 漂移均拒绝候选。

R0 的候选与 replay/rollback receipt 永久保留为历史诊断，R1 不修改或重解释这些字节。

### 历史 prompt-only Qwen3.7 R1–R5 execution ledger

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

### 历史 prompt-only R6–R10 execution ledger

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

## R12 parent 条件化单 Surface 周期

30 轮 model-generated dual-policy campaign 已冻结；其中只有 R12 通过完整 replay200/body75
正式门。新的 `s1-counterfactual-v1` 不恢复 whole-bank 自由重写，也不把 rejected Bank 拼回
parent，而是显式绑定：

- R12 Bank `e70ed907825833a0bbb97ccde068fd38d4a6342824cd9dcdfb543723feb096cd`；
- Bank file SHA `edf83d288fcbb67b78b53a6390b713f845238836b7fc4f3378aca34c23bf9489`；
- accepted decision SHA `1edc7fba7f05ee3deffc8a41907931a76e660fba5058262e3251156957923bc4`；
- 来源 manifest、fresh R12 parent opt800 与各自 SHA；
- R12 Style Skill SHA 与八个已改善的 parent-protection query。

R31 与 R32 必须在任何 Feedback/Creator 调用前同时冻结。R31 只允许 Recipe action-policy，
R32 只允许 Multi response-policy；每轮仍生成六个 BranchRecord，但只有目标 capability 调用一次
Creator，其余五项均为 `protected_inherit`。输入 packet 固定为同一失败簇的 3 个 failure、状态
最接近的 3 个 R12 parent-success，以及 3 个历史 regression。证据不足时直接以
`insufficient_counterfactual_evidence` 终止，不允许缩小 packet 或改选规则重试。

Creator 只能输出一条 provider-visible 条件规则：`when`、单 surface 的 `then`，以及恰好三个
`must_preserve` 成功状态。compiler 只追加规范化文本：

```text
If and only if <when>, <then>. Otherwise preserve the parent behavior, including <must_preserve>.
```

局部 screen 沿用有界风险门，并额外要求三个 packet parent-success 0 regression、R12 Style 八个
保护样本与 Style Skill byte-exact、failure→hard/runtime 0 次、action/response 固定边界不漂移。
随后必须依次通过 replay200（macro ≥0pp）与 body75（macro ≥+2pp、CI95 lower ≥0）；两级
hard-error delta 均 ≤+1pp、各能力 floor 均为 −5pp。普通 failure reason 迁移只记录，不自动拒绝。

R33 不调用 Creator。零个 accepted branch 时停止；一个时直接成为 finalist；两个时才从 R12
byte-exact fan-in Recipe/Multi，并重新执行 replay200/body75。组合失败时，只能在两个已经独立
正式 accepted 的 Bank 中按预注册排序选一个 finalist。局部通过但 body75 未接受的分支永远不得
组合。最后的 `s1-finalist-test` 仅允许一个已冻结 finalist 与 R12 在 test300 上成对执行一次，
不调用 Judge/S2/S3；测试失败则 cycle-selected 仍为 R12。一旦执行，该 test300 不再是 untouched
五配置测试，后续若需要无偏五配置比较必须建立新 holdout。

运行顺序：

```powershell
uv run python scripts/prepare_s1_counterfactual_cycle.py bootstrap --cycle-root <cycle-root>
uv run python scripts/run_core_experiment.py --spec <cycle-root>\specs\s1-parent-opt800-bootstrap.json --output-root <cycle-root>\runs\parent-opt800 s1-parent-opt800
uv run python scripts/prepare_s1_counterfactual_cycle.py freeze-cycle --cycle-root <cycle-root> --bootstrap-receipt <cycle-root>\runs\parent-opt800\s1-parent-opt800-bootstrap.json
uv run python scripts/run_core_experiment.py --spec <cycle-root>\specs\r31.json --output-root <cycle-root>\runs\r31 run --through s1
uv run python scripts/run_core_experiment.py --spec <cycle-root>\specs\r32.json --output-root <cycle-root>\runs\r32 run --through s1
uv run python scripts/run_s1_counterfactual_cycle.py finalize --cycle-definition <cycle-root>\cycle-definition.json
uv run python scripts/run_s1_counterfactual_cycle.py s1-finalist-test --cycle-definition <cycle-root>\cycle-definition.json --finalist-receipt <cycle-root>\cycle-finalist.json --output-root <cycle-root>\runs\test300
```

前四类产物均为 create-only；同一 round 只允许 CallStore resume，不允许重采或重试挑结果。预算
在 bootstrap/freeze/finalize/test 各边界重新核算，整个周期 DashScope 预计费用超过 CNY 10 即
停止并请求授权。

本周期现已终结。canonical parent baseline 位于
`D:\athena\experiment-runs\portfolio-core-r12-counterfactual-v3-20260814`，800/800、GCS
success 164、hard error 146、SHA `4c6bafa65e320f46…aa84fcf`、成本 ¥1.2439722。更早的 v1
在 provider 前因错误 config 全部失败（¥0）；v2 为 799/800，单个 non-retryable provider failure，
成本 ¥1.2357178 加一个未知 orphan，未进入 selection。终态 v4 中，R31 9/9 Feedback 通过，但
Creator 的 action `then` 要求生成 fallback/answer/evidence，跨 response surface，故
`counterfactual_contract_rejected`、Assistant replay/body 0-call；R32 因不足三个 matched
parent-success，以 `insufficient_counterfactual_evidence`、Creator 0-call 终止。R33 冻结 0 个
accepted branch、`finalist=null`、`test300_allowed=false`。周期可追踪 DashScope 合计
¥2.52357745；唯一 Creator 成本不可得。test300、S2/S3/Judge 均未访问，selected Bank 保持 R12。

这个终态是**安全性成功、实验吞吐失败**：两个计划 treatment 进入 replay 的数量为 0/2，所以不报告
S1 增益或负增益，也不调整局部有界风险门、replay/body gate 或 R33 规则。后续机制版本
`single-surface-counterfactual-fanout-v5` 做三项离线收口：action Creator 只能输出 typed tool-loop
condition/transition，不能表达 answer/cards/evidence/fallback 文本；每条 Feedback 必须精确归属目标
surface，跨 surface 或无标签 suggestion 在 Creator 前剔除；response 样本必须先满足可固定 route/tool、
成功 tool trace、无 hard error，再按公开 evidence 类型与 empty/nonempty 分支聚类，而不要求相同的精确
cardinality。

`freeze-cycle` 在生成可运行 spec 前，会对所有预注册轮次一次性执行零 provider 的
evidence-feasibility 与 treatment-sensitivity preflight。每轮必须同时提供 3 个同簇 failure、3 个
合格 parent-success、3 个合格 historical regression，并证明 probe 只改变目标 Skill 的目标 surface；
任一轮失败时只落 preflight 诊断，不生成 runnable round specs。Core Fast 在 `validate` 以及第一次
Feedback 前都会复核 preflight 文件 SHA、R12 parent/opt 身份、所有轮次通过状态和当前 selection
manifest SHA。真实 R12 opt800 的只读审计显示：R31 已通过这两类 preflight；R32 的旧证据中
`r2-core-0784` 没有成功 tool trace且为 hard/action failure，因此仍被阻断。修复方式是下一周期事前
冻结新的、response-qualified regression evidence，不是把该样本强行降格或放宽 gate。R12 仍是唯一
合法 parent；R33 仍只组合正式 accepted branch，test300 仍只允许一个 frozen finalist 使用一次。

### 自适应首批 R34–R38

`s1-r12-adaptive-v1-b01` 在同一 R12 parent 上预注册 5 个顺序 slot，并在每轮终态后落一份
create-only retrospective 与下一轮 memory。首批终态不是新的 S1 增益结论：R34 因 Feedback
disposition/surface 标签冲突停在 canary；R35 因描述性 `s1-*` config 未映射到 S1 runtime 而在
provider 前失败；R36 的 response `then` 是多条件 checklist，被单句 compiler 拒绝；R38 同时把
Style 声明为 target 和 byte-exact protected Skill，因 parent invariant 在 0 Assistant 时停止。只有 R37
获得有效局部 action 比较：Encyclopedia 4 gains / 10 regressions，且 2/3 显式 parent-success 回退，
因此按既有有界风险门拒绝。五轮均未进入正式 replay200/body75，也未访问 test300/S2/S3/Judge。

本批对应的前向机制修正为：Feedback suggestion 同时携带 disposition 与唯一 surface；动态
`s1-*` trace label 统一映射到 S1 runtime treatment；response `when/then/must_preserve` 在 JSON
Schema 层限制为单句且禁止分号/换行；实例化 `must_preserve` 只保留在 verifier 与 receipt，不渲染进
live Skill；all-round preflight 除 evidence-feasibility 和 treatment-sensitivity 外，新增
protected-target compatibility。任何 target 命中 `s1_parent.protected_skill_sha256` 时，整个 batch
必须在 0 Feedback 阶段拒绝。局部/正式门不变，R12 仍是唯一 selected parent。

准备脚本为：

```powershell
uv run python scripts/prepare_s1_adaptive_batch.py freeze-batch --plan specs/s1-r12-adaptive-batch01.json --bootstrap-receipt <r12-bootstrap-receipt>
uv run python scripts/prepare_s1_adaptive_batch.py freeze-round --plan specs/s1-r12-adaptive-batch01.json --batch-definition <batch-definition.json> --bootstrap-receipt <r12-bootstrap-receipt> --round-id <round-id> --memory <prior-round-memory.json>
```

首批证据根为
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b01-20260815`；可追踪 DashScope 成本
¥0.3496057，4 个 Creator 会话的人民币 cost basis 不可得。

### 自适应第二阶段 R39–R43

R39 最初在 B02 独立 root 执行。其 Creator 产生了一个 schema-valid 但非法的 Recipe
`before-first-tool → recipe_lookup` transition；局部 action screen 为 2 gains / 5 regressions，包含
两个显式 parent-success 回退。随后 action IR 增加 capability-specific 顺序约束：Recipe 与
Encyclopedia 必须先 `object_detect`，lookup 只能在成功且非空检测后消费
`last-visible-tool-output`；retry/stop 也绑定一致的 prior state。与此同时，all-round treatment probe
从 Bank 存储顺序改为 canonical action 顺序。active-code preflight 因而不再与 B02 frozen bytes
一致，B02 在任何 R40 provider call 前 create-only 停止，R40–R43 转入独立 B03；旧 root 不 resume。

B03 四轮及 R39 的终态如下：

| Round | Target surface | Parent→candidate | Gains / regressions | 终态 |
|---|---|---:|---:|---|
| R39 | Recipe action | 8→5 | 2 / 5 | explicit protection regressions；回滚 |
| R40 | Exact action | 17→17 | 4 / 4 | net 0；explicit protection regression；回滚 |
| R41 | Encyclopedia response | 10→9 | 1 / 2 | fixed trace/evidence；net −1；回滚 |
| R42 | Multi response | 13→13 | 0 / 0 | no treatment gain；回滚 |
| R43 | Encyclopedia action | 24→23 | 8 / 9 | canonical detect→lookup；net −1；回滚 |

五轮的 Feedback full-batch gate 与 typed proposal 均有效，且没有 severity escalation 或固定边界
trace mismatch；但没有分支满足既有 gains/net/ratio/parent-success 门，所以 formal replay200、body75、
fan-in、test300、S2/S3/Judge 仍全部为 0-call。R12 Bank `e70ed907…096cd` 继续是唯一 selected
parent，不能把这些局部数字报告为系统级 S1 正增益或负增益。阶段共 50 个 Feedback provider calls、
340 个 Assistant outer / 629 个 Assistant inner calls、5 个 Creator，会话新增可追踪 DashScope
¥0.67932075；Creator 人民币 cost basis 不可得。

下一预算阶段的机制优先级已经由本阶段证据限定：response parent-success 必须在当轮 live
parent-control 中仍成功；与 parent Body 语义等价的 response clause 应在 Assistant 前标记为 no-op；
action condition 必须具有能区分 failure 与 protected-success 的 provider-visible predicate。上述是
待验证的前向改造，不改变任何已有 gate，也不使 R39–R43 candidate 合法化。canonical evidence
roots 为 `D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b02-20260815` 与
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b03-20260815`。

### 自适应第三阶段 R44–R48

B04 先以 0 provider 调用完成了一次旧 schema preflight；随后 action condition 与 response failure
family 的 active contract 收紧，因此它被 create-only 标记为 superseded。独立 B05 在任何新
Feedback/Creator 调用前一次性冻结并复核 R44–R48：五轮均为 3 failure + 3 live parent-success +
3 historical regression，且全部通过 `evidence_feasible`、`treatment_separable` 与
`treatment_sensitive`。action failure 只允许 provider-visible 的 post-tool error / invalid-arguments
状态；response failure 按 item association、card closure、unsupported claim、citation closure 等
语义 family 聚合，不再以精确 card 数量制造簇身份。R12 与八个 Style 保护样本始终冻结。

| Round | Target surface | 局部 screen | 最深 gate | 终态 |
|---|---|---:|---|---|
| R44 | Recipe action：invalid-arguments retry | 7 gains / 1 regression | replay200：macro +1.6667pp；Recipe +10pp；hard error +2pp | 正式 hard-error 门回滚 |
| R45 | Multi response：item association | 0 / 0 | local response screen | 最小 gain/net 门回滚 |
| R46 | Multi response：card closure | 0 / 0；1 条普通 reason 迁移 | local response screen | 最小 gain/net 门回滚 |
| R47 | Encyclopedia response：unsupported claim | 未执行 | Creator contract | 禁止评价式内容；candidate null |
| R48 | Encyclopedia response：citation closure | 0 / 1；net −1 | local response screen | 有界风险门回滚 |

R44 证明枚举化 action IR 与 surface-filtered Feedback 可以把可分离 treatment 推进正式 replay；
但 tool completion 改善后暴露出额外 response hard error，因此不能放宽 `hard-error delta ≤ +1pp`，
也不能把被拒的 R44 Bank 与未来 response patch 拼接。R45/R46 表明两个 Multi closure 文本 family
虽然安全，却没有改变评分行为；R47 不能记为算法负结果，因为 Assistant 为 0-call；R48 则真实产生
1 条 parent-success regression。普通失败到另一普通 reason 的迁移均只进入诊断，没有触发硬拒绝。

B05 合计 47 个 Feedback provider calls（45 个 terminal success、2 个格式失败 attempt）、5 个
Creator、674 个 Assistant outer calls；Feedback 为 `218,949 / 138,219` input/output tokens，Assistant
为 `3,652,448 / 331,082`，新增可追踪 DashScope 成本 ¥1.23552605。Creator 为
`187,533 / 4,378` tokens，人民币 cost basis 不可得。仅 R44 访问正式 replay200；body75、fan-in、
test300、S2/S3/Judge 均未访问。R12 `e70ed907…096cd` 仍是唯一 parent 与 selected Bank。

下一阶段不重复 Multi DTO/card closure 文案。优先把 response `then` 从自由文本收紧为 typed
semantic operation，使评价标签和跨 family 内容不可表达；同时让 treatment-sensitivity probe 绑定
一个可预测的评分行为变化，而不只验证 policy text/hash 不同。所有轮次仍须在新 Feedback 前一次性
通过 evidence feasibility、treatment separability、treatment sensitivity 与 parent protection；
局部、replay、body 与 R33/test300 规则全部保持不变。canonical root 为
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b05-20260815`。

### B06/B07 surface-closed response 阶段（R49–R54）

`single-surface-counterfactual-fanout-v6` 关闭了 B05 暴露的两个 treatment 通道问题。action
proposal 继续使用 capability-bound tool-loop state/transition，类型上不能携带 answer、cards、
evidence 或 fallback 内容；response proposal 改为封闭枚举 IR，condition 只允许 terminal tool、
empty/nonempty evidence outcome 与公开 evidence kind，directive 只允许当前 failure family 对应的一项
规范化 response operation。Creator 不再拥有自由文本 `when/then`，也不能从 response proposal 写回
action surface。Feedback evidence 在进入 Creator 前按 surface 投影：action 看不到回答/card/evidence
内容，response 看不到 action violation/transition 内容。

response selector 的资格也改为回答端语义，而不是整条工具链的粗粒度相等：只有实际承载回答证据的
terminal tool 参与 fingerprint，上游 `object_detect` 不再把 Recipe/Encyclopedia 的 response cluster
误分裂；每个 failure family 同时绑定预期 reason code 与 scorer component。preflight 不只证明 policy
bytes/hash 会变化，还要求 3 个 failure 在相同 terminal evidence 上具有目标 reason/component failure，
3 个 parent-success 在同一边界上保持该 component success。

tracked `specs/s1-r12-adaptive-batch06.json` 预注册五个独立 response round：R49 Recipe fallback、
R50 Exact unsupported claim、R51 Encyclopedia unsupported claim、R52 Encyclopedia citation closure、
R53 Multi item association。前两个 create-only root 在 0 provider call 时分别发现 scorer component
绑定错误和不可形成 3/3/3 对照的 failure family；它们保持为失败诊断，没有生成可运行实验。最终
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b06-v3-20260815\batch-preflight.json`
对 R49–R53 均得到 `evidence_feasible=true`、`treatment_separable=true`、
`structural_treatment_sensitive=true`、`behavior_treatment_sensitive=true`，且五个实际 selection
manifest SHA 与冻结值一致。

资格证明后没有复用旧 root。R49 在 provider request 前暴露 live adapter 仍把 surface-projected
冻结输入与 full baseline projection 比较；三条 canary 都是本地 identity error、实际 provider 0-call，
因此不构成 Recipe 算法结果。修复 live/local projection parity 并加集成测试后，R50 的 9/9 Feedback
通过，但 free-form `must_preserve` 句子触发 authored-content guard，Assistant 仍是 0-call。前向 v7
把 preservation 收窄为 query-ID-only typed refs，condition 和 operation 继续使用封闭枚举；没有重试
R49/R50，也没有改写其终态。

独立 B07 root 随后执行 R51–R54：

| Round | Target | 局部 screen | 正式门 | 终态 |
|---|---|---:|---|---|
| R51 | Encyclopedia response · unsupported claim | 0 gain / 1 regression | 未进入 replay200 | 回滚 |
| R52 | Encyclopedia response · citation closure | 2 / 0 | replay macro `+0.8333pp`；body macro `+3.125pp`、CI95 lower `+2.7778pp`、hard-error `−4pp` | AcceptedBranchArtifact |
| R53 | Multi response · item association | 0 / 0；1 条普通 reason 迁移 | 未进入正式 replay200 | 安全无效，回滚 |
| R54 | Recipe response · empty-source fallback | 0 / 0；1 条普通 reason 迁移 | 未进入正式 replay200 | 安全无效，回滚 |

R52 candidate Bank 为 `67b92b61…55bc8`，Encyclopedia Skill 为 `76622c04…d2916`；它通过了
局部有界风险、replay200 和 body75 全部门，因此被 create-only 冻结为 accepted branch。它尚未
成为 cycle selected Bank，也没有被用作 R53/R54 parent。R12 `e70ed907…096cd` 仍是所有独立
round 的唯一 parent 和当前 selected Bank；R52 只能进入后续 accepted-only fan-in/finalist。

R49–R54 新增可追踪 DashScope ¥1.3534802、5 个 Creator 会话；Creator 人民币 cost basis 不可得。
自适应进度为 21/30。test300、S2/S3/Judge 均未访问，R55 尚未冻结/运行；局部门、正式
hard-error 门、R33 accepted-only fan-in 与 one-finalist test300 规则未改变。canonical roots 为
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b06-v3-20260815`、
`...-b06-v4-20260815` 与 `...-b07-20260815`。

### B08 capability-specific response IR 阶段（R55–R60）

R31/R32 的 treatment throughput 0/2 已被前向机制关闭，而不是通过放宽 gate 追认：action Creator
继续使用不能表达 answer/cards/evidence/fallback 的 typed tool-loop IR；Feedback 在进入 Creator 前按
action/response surface 投影；response selector 只有在 route 正确、无 hard error、terminal tool 成功且
目标 reason/component 可与 matched parent-success 对照时才允许成簇。B08 又把 response operation 从
跨能力通用动词收窄为 capability + failure-family 枚举，并用 `parent-counterfactual-v11` 绑定实际评分行为。

首次 B08 零调用 preflight 因两个 Recipe source family 找不到 3 个 matched parent-success 而 fail closed；
没有调用 provider。前向 B08 v2 在任何新 Feedback 前一次性冻结 R56–R60，并对五轮全部得到
`evidence_feasible=true`、`treatment_separable=true`、`structural_treatment_sensitive=true`、
`behavior_treatment_sensitive=true`。R55 是 B07 预注册的唯一 Exact operational replacement：完整九条
Feedback 为 8 success + 1 provider 500，terminal gate 在 Creator 前停止。

| Round | Target | 局部 screen | 正式门 | 终态 |
|---|---|---:|---|---|
| R55 | Exact response · unsupported claim | 未执行；Feedback 8/9 | 完整 Feedback gate | operational failure；无算法结论 |
| R56 | Multi response · complete item mapping | 1 gain / 0 regression | replay macro `−1.7241pp`；Multi `−10.3448pp`；hard 0pp | 正式门回滚 |
| R57 | Multi response · referenced cards | 0 / 1 | 未进入 replay200 | 局部门回滚 |
| R58 | Encyclopedia response · supported cited claims | 0 / 2；5 条普通 reason 迁移 | 未进入 replay200 | 局部门回滚 |
| R59 | Encyclopedia response · unknown citation | 1 / 0 | replay macro 0pp；Encyclopedia 0pp；hard `+1.5pp` | 正式 hard-error 门回滚 |
| R60 | Exact response · evidenced claims/cards | 0 / 0 | 未进入 replay200 | 最小 gain/net 门回滚 |

普通失败到另一普通 reason 的五条 R58 迁移只进入诊断；真正否决的是两条 parent-success regression。
R56 与 R59 证明 treatment 已经可达且局部可产生 gain，但 formal replay 分别暴露 capability 大幅下跌与
hard-error 超限，所以不能 fan-in。五个候选全部从 R12 独立派生，R52 没有成为 parent，R55–R60 也
没有相互继承。B08 阶段新增可追踪 DashScope ¥1.85708305、5 个 Creator 会话；body75、test300、
S2/S3/Judge 均未访问。自适应进度为 27/30；R12 仍 selected，R52 仍是唯一新增 accepted branch。
canonical root 为
`D:\athena\experiment-runs\portfolio-core-r12-adaptive-v1-b08-v2-20260815`。

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
0；60 是本次验证值，不是服务上限。当前 S1 先执行固定 canary6，再按 selection policy 补齐
48 或 60 条完整 Feedback 集合，所以不会人为制造 60 个
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

当前 tracked spec 已冻结 fresh Static，且该命令返回 `runtime_ready=true`。以下通用付费/下游命令仍是
编排接口；新的 R12 parent 周期必须使用上文独立 spec/root，不能把 rejected candidate 带入 S2：

```powershell
uv run python scripts/run_core_experiment.py run --through s1
uv run python scripts/run_core_experiment.py run --through s2
uv run python scripts/run_core_experiment.py run --through full
uv run python scripts/run_core_experiment.py run --through test
uv run python scripts/run_core_experiment.py report
```

默认 spec 指向当前 model-generated fresh Static，不再指向最终 R10 的 deterministic v6 Static。
`validate` 不产生调用且当前通过；它不等于授权重跑 S1。历史 deterministic roots 与当前 30 轮
canonical roots 都保持只读，不能跨 runtime 继续 S2，也不能把任何 rejected candidate 当作 parent。
后续阶段唯一合法的 S1 selected Bank 是 R12 `e70ed907…096cd`。

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
Creator 为 fan-out 的每个 capability 启动一个相互隔离的 Codex CLI session，Judge 使用现有 packet、prompt 和 parser；所有
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
