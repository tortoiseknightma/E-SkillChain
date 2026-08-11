# SkillChain 公开数据复现评价协议

> **协议 ID：** `skillchain-eval-v0`
> **状态：** `v0-draft`；只允许 mini 工程纵切，不授权 formal/core 结果
> **制定日期：** 2026-07-20
> **实现基线：** `master@a80af22` + `codex/authoring-codex-high-approval@1ea786a`
> **复现契约：** [`reproduction-contract.md`](reproduction-contract.md)
> **taxonomy：** `ecommerce-mvp-taxonomy-v0`，semantic SHA-256 `af23dbe76e8c0cffdbb056b44c6595af6c508eaa0158f96227cbc92e244ca315`（`specs/taxonomy/ecommerce-mvp-taxonomy-v0.json`）
> **前瞻 Task Specification：** `ecommerce-task-spec-v1`，semantic SHA-256 `f8d5596de5d0ba98235f82c7c176a5b774b33d7bdd7e84fb00a07b5b0b7a7f0d`（`specs/task_specs/ecommerce-task-spec-v1.json`）；v0 仅保留用于历史工件重放
> **规范化内容 SHA-256：** `7436dd9fde6f83939599003fd98b03dc9bdb15e4baf6c52310b19e48ae5b42d8`

## 1. 目标、分析单位与生效条件

本协议预先规定五个主配置的比较、数据可见性、指标、统计、排除、错误处理、停止条件和冻结过程。它用于阻止候选 Skill、pilot 输出、Judge 结果或 final holdout 反向定义任务与评价标准。

默认分析单位是一个冻结 `query_id`；因同 product、asset、近重复、boundary、template 或 generator family 产生相关性时，统计重采样单位提升为权威 split manifest 中的联合 leakage component。所有配置必须在同一 `query_id` 集合上配对运行，错误也保留在配对集合和分母内。

本协议仍是 draft。以下字段在从 mini 升级为唯一一次 `v1-frozen` 修订前必须填入 lock manifest，否则 formal runner 必须拒绝启动：

- `ecommerce-mvp-taxonomy-v0` 与前瞻 `ecommerce-task-spec-v1` 的 canonical semantic
  hash，以及后者对前者的绑定；TaskSpec v0 只允许用于已冻结历史工件重放；
- 选择 `core` 或 `full`、最终 capability 集和每组实际样本量；
- Stage 2/3 的最小效果或非回退界值、最小推断 n；
- 每项工具的 formal go/no-go 阈值；
- Judge—human calib/audit 数值门与最大重试数；
- 作者/Assistant/feedback/final 模型、prompt/rubric、预算和人工分钟上限；
- dataset/query/split/index/tool/gold/bank/environment manifest hash；
- 协议/契约 hash 与 clean code commit。

未填字段可以用于测试协议实现，但不能产生“正式”“达到论文效果”或 core/full 机制结论。

### 1.1 前瞻模型角色锁（2026-07-24）

当前目标角色为 Assistant=`DashScope/qwen3-vl-flash-2026-01-22`（non-thinking）、Author=`Codex CLI/gpt-5.6-sol/high`。当前 owner-selected Portfolio v6 将 visual Feedback 设为百炼/DashScope `kimi-k2.6`（non-thinking，temperature=`0.6`、top-p=`0.95`），final evaluator 设为 AIFast OpenAI-compatible `gemini-3.6-flash`（不发送 temperature/top-p/top-k，thinking 为中转站默认且不可验证）。两者使用独立 provider/model family/endpoint、packet、prompt、cache、input 和 artifact namespace；可以声明运行时跨 provider 隔离，但不得把第三方中转站返回的模型名描述为 Google provider-attested 身份。Phase 3 label reviewer 不得兼作 feedback 或 final evaluator。

Codex Author 使用独立的 `codex_mediated_static_author_v1` 证据口径；当前事务实现代际为 v4。LLMStaticSkill 与 S1 必须逐字节共享同一语义输入，并共享同一 Codex surface、模型、`high` 和 session 预算。“一次调用”表示一个 create-only `codex exec` process/session；它不等价于可证明的单 provider attempt。每次授权只允许 1 个 session，且 0 retry、follow-up、repair、fallback 和 tool activity；进程失败、无效输出或 schema 拒绝也会消费授权。receipt 必须将 served revision、provider request ID、平台内部 backend attempt 和费用记为 unavailable/unobservable，并固定 `formal_provider_call_eligible=false`；token 只有在 CLI `turn.completed` 提供时才能保存为 `platform-reported_non-provider-attested`，不得冒充 provider usage。输入通过 frozen stdin、仓库外空 scratch、read-only、ephemeral、ignore-user-config/rules、项目 `AGENTS.md` negative probe 和事件审计约束；由于 CLI 仍带平台基础指令且 read-only 不证明全局不可读，只能声明 `behaviorally_constrained_not_mechanically_proven`。因此 RQ1b 的最终报告必须披露这一限制，不能把它描述为与旧 DashScope container 相同强度的隔离。

Kimi Feedback 与 Gemini final 都通过 OpenAI-compatible `image_url` 接受 Base64 Data URL。FeedbackPacket/FinalEvaluationPacket v2 必须先用 verified catalog、文件身份和 SHA-256 复验本地图片，只保存 MIME+SHA；runner 在对应 remote-processing preflight 后瞬时读取字节并生成多模态 content part。严禁把 Base64 写入文本、packet、prompt snapshot 或结果回执后声称模型看图。active selection 是 [`model-role-selection-v7.json`](../specs/authoring/model-role-selection-v7.json)，其 file/self SHA-256=`fbfbe9437731052743b3025962a22e4d3cd6432c2212b7cc418b2118e9fd2ba4` / `fc3e8ad7a1b2ace95ffe3df4c775a8c38425274459853d555dabb2c836097633`；v2-v6 历史不改写。v7 将 Kimi Feedback 前向身份锁定为 cache-v8、response-schema-v1 prompt-v5 与 plain-JSON transport-v3，parser-v3/output contract-v4 不变。现有 Core v1 权限覆盖 Kimi Feedback；AIFast Gemini Judge 在新的 owner authorization、价格/BOM 与 runtime/launch relock 完成前 fail closed。历史 AIFast Gemini Feedback smoke 不是新 Judge 的 production 证据，也不能转授权。production 五配置矩阵回执和 Judge—human audit 未完成前，所有 final evaluation 仍 `formal_eligible=false`。

## 2. 研究问题与预注册比较

| RQ | 问题 | 预注册主比较 | 主评价量 |
|---|---|---|---|
| RQ1a | Creation 相对无 Skill 是否有增益？ | `S1 − NoSkill` | 配对 `J_project` 差值；四个维度必须同时报告 |
| RQ1b | 冻结训练轨迹是否比资源匹配的一次性静态 authoring 更有效？ | `S1 − LLMStaticSkill` | 配对 `J_project` 差值；公共输入、预算和人工分钟审计为有效性前提 |
| RQ1c（可选） | S1 是否优于独立专家人工 Skill？ | `S1 − ExternalExpertManual` | 与 RQ1b 相同；只有 provenance gate 通过才建立该比较 |
| RQ2 | Stage 2 是否提升 capability 路由并保持跨来源/措辞表现？ | `S1+S2 − S1` | `canonical_capability` macro-F1 配对差值 |
| RQ3 | 固定同一 Stage 2 路由后，Stage 3 是否提升响应质量？ | `Full − S1+S2` | 复用 route trace 的配对 `J_project` 差值 |
| RQ4 | 四条 Stage 3 归因路径各自贡献多少？ | `Full − 每个冻结消融 Bank` | 预注册 test 子集上的配对 `J_project` 差值，四比较成族校正 |
| RQ5（扩展） | T1–T3 shift 下的适应、遗忘、成本和 Bank 增长如何？ | 不属于 v1.0 | 另建协议；不得从 T0 结果回答 |

RQ1a–RQ4 的结果方向不构成项目成败标准。置信区间跨 0、负增益或 capability 间方向相反均如实报告。`ExternalExpertManual` 缺席时 RQ1c 标为“未复现/不可回答”，而不是删除后假装论文 Manual 基线已覆盖。

## 3. 配置、诊断与公平性不变量

### 3.1 五个主配置

| 配置 | treatment | 明确不包含 |
|---|---|---|
| `NoSkill` | 移除 Skill route/inject；保留同一 Assistant、用户输入、工具与等价推理预算 | 隐藏专家规则、额外 prompt、额外工具调用 |
| `LLMStaticSkill` | 同一作者模型只读冻结 AuthoringPacket，一次性生成并确定性编译；不按结果迭代 | 任何轨迹、标签、响应、Bank、rubric/gold、gate/test、Judge/指标 |
| `S1` | 与 LLMStaticSkill 完全相同的公共 AuthoringPacket，只额外加入预注册训练轨迹，生成 Stage 1 Bank | S2/S3、route/body gate、final holdout，以及 S1 独占的 reference Skill |
| `S1+S2` | 在 S1 上最多四轮只改路由层，按 `route_gate` 接受/回滚 | Body/Cs/Od 暗改、body_gate 或 final holdout |
| `Full` | 在冻结 S1+S2 上最多三轮只改 Body，按 `body_gate` 接受/回滚 | 重新路由、修改 Description/Cs/Od 或 final holdout |

所有主配置共享：

- 同一 Query 内容与顺序、Assistant backbone、system/task 约束和解码参数；
- 同一 ToolSpec、模型/索引/KB artifact、工具调用上限、timeout 与错误策略；
- 可比的 inference token budget；若 Skill 注入占用上下文，必须报告注入 token，不得给 NoSkill 额外隐藏推理；
- 同一 frozen rubric、final evaluator、重试策略和 metric implementation；
- 同一排除清单、分母和 paired group bootstrap 样本；
- S1+S2 与 Full 的逐 query route decision 完全相同。任何不一致都使 RQ3 无效。

#### Portfolio Track 首批执行口径（2026-07-31 至 2026-08-04 历史快照）

以下段落保留首批执行时的历史状态。当前 Portfolio V1 收口见本小节末尾的 2026-08-11 更新；这些规则不把本 draft 升级为 formal 协议，也不授权论文效果结论：

- shared-route 公平性规则保持不变：每个 query 只生成一份 create-only `shared-stage2-route-v1` artifact；router 只选择冻结 capability，S1+S2 与 Full 分别按各自 Bank 做确定性 capability→slug 映射，并引用同一 route identity。两配置保留相同 action turn/output/tool budget，且都显式预留同一 route turn/usage，不能让 Full 因复用路由获得额外 action 预算。runtime v10 只实现了该执行语义，其 treatment Bank 后续被确认是 deterministic scaffold，不能据此形成阶段结论。
- 可重试的 pre-response provider failure 只写 create-only attempt receipt，不立即生成 Assistant 终态或 fixed-zero；runner 停在最早未完成 query，恢复时仍按冻结顺序继续。`portfolio-shard-attempt-v1` 的 circuit threshold 和每 query 最大 retryable attempt 均为 2；已经捕获有效模型响应后的 schema、路由、工具或任务失败仍按行内结果处理，不能借熔断选择性重跑。
- NoSkill 仍可引用 execution v6 的冻结 shard，但旧 execution v7 的四个 skilled shard 只能作为 immutable diagnostic。当前边界重审证明 S1→S1+S2 在六个 capability 上都同时修改了 Body，违反 S2 Description-only 约束；其 `LLMStatic/S1/S1+S2/Full` 名称来自 scaffold，不能升级为真实 treatment。
- runtime v11 / launch v14 及后续 stale-parent 中间链均已 superseded。当前有效链为 runtime v22 / launch v25 / execution v16：S2 绑定上一 gate 的精确 S1 parent 并只改 Description，S3 绑定 gate-selected S2 parent 并只改 Body；finalizer 和 loader 重验 current-parent attribution、typed gate result-set、模型/effort、实现与 executable SHA、thread/scratch identity。最终决策为 `S1 accepted / S2 accepted / S3 rolled_back`，因此 S1+S2 与 Full 的 output Bank 都是 `9502ef4b…`。launch v25 仍包含完整 1,000 instances / 40 shards，但当前只完成 optimization25 的五个逻辑 shard；evaluation175 尚未执行。
- S2/S3 attribution 的 `source_bank_sha256` 必须等于上一 gate 选出的精确 current parent Bank，并与该 gate-parent smoke 的 summary/results/逐行 result SHA 同源；任一 lineage 或 session 绑定不一致时，即使存在真实模型 receipt，整条 chain 仍须 fail-close。

旧 execution v7 的 125 行和均值 NoSkill/LLMStaticSkill/S1/S1+S2/Full=`62.88/75.30/67.94/74.56/74.26` 只描述 scaffold 行为；重审状态为 `failed`，核心 blocker 是六项 S2 Body 越界。当前真实 treatment gate 为：S1 `J 68.90 → 77.12`、route `0.84 → 0.88`、hard `1 → 0`（接受）；S2 `J 77.12 → 77.44`、route `0.88 → 0.92`、hard 保持 `0`（接受）；S3 的独立 stage sample 为 parent/candidate `75.56 → 73.80`、route 都为 `0.92`、hard 都为 `0`（回滚）。这些 gate 只决定 development 接受/回滚。

v16 clean composite 已完成 optimization25 的 25×5：均值为 `73.34/73.04/76.14/77.92/76.64`，S1+S2 相对 Static 的配对均值 `+4.88`、route macro-F1 `0.8913 → 0.9277`，但 cluster CI95 `[-0.4220, 9.7826]` 跨 0。S1+S2 / Full 的 25/25 shared route 和 output Bank 都相同；no-op 配对差 `-1.28`，不得归因于 S3。集中审计为 125 行完整、`blockers=[]`、`review_required`：唯一 non-scored 行是 S1 `dm-020` 的 Judge 连续格式失败，另有 19 条 card contract violation，均保留在冻结全分母并单独分类。在该历史快照时，disjoint evaluation175 尚未执行。

#### Portfolio V1 收口更新（2026-08-11）

`dev_mini 200×5` 的真实模型 / 工具矩阵已经完成。未参与 development gate 的 evaluation175 上，NoSkill / LLMStaticSkill / S1 / S1+S2 / Full 的 Mean Legacy J 为 `68.706 / 69.937 / 71.131 / 72.080 / 70.940`；`S1+S2 − Static=+2.143`，按 20,000 次 leakage-group 配对 Bootstrap 得到 95% CI `[-1.111, 5.422]`，跨 0，因此只作为方向性开发信号。

随后完成一次真实 `Qwen Feedback → Codex Creator → Assistant replay200 → GCS Gate` 闭环。Static→S1 candidate 的 GCS capability-macro 为 `21.7729% → 25.2749%`（`+3.5021pp`），query-micro 为 `26% → 31%`（`+5pp`），hard-error delta 为 0；但 Visual Encyclopedia 为 `7.5% → 2.5%`（`−5pp`），越过预冻结的 `−3pp` 单能力底线，因此整 Bank 被拒绝并 byte-exact 回滚。前置 Gate 失败后，Final Judge、body gate、val 与 `test300` 均未启动。

### 3.2 诊断配置

- `SpecBaseline`：不调用作者 LLM，从冻结 Task Specification/ToolSpec 确定性编译。只用于区分规范注入和 LLM authoring，不进入五配置主表。
- `OracleRoute+S1`：读取 gold/acceptable capability 强制路由至冻结 S1 Bank。只估计路由误差被移除后的上界，不用于候选选择、部署结论或主表。

诊断配置可使用同一 test 的前提是它们在解封前已预注册、Bank 已冻结，并与主配置在同一个无中间结果暴露的 sealed job 中执行。否则只能使用开发集或新的 holdout。

### 3.3 ExternalExpertManual 补充列

该列必须通过 `reproduction-contract.md` 第 8 节全部 gate，并在 test 解封前冻结。资格、独立性、可见输入、固定工时/报酬、无结果访问声明、结构化表单、编译器、人工 diff 与最终 Bank hash 都进入 manifest。内容阶段使用生成式 AI 时必须改名 `ExpertCuratedLLMSkill`，不得用于 RQ1c。缺证据时 fail closed，不影响五配置主实验。

## 4. AuthoringPacket 与信息边界

### 4.1 公共 AuthoringPacket

`LLMStaticSkill` 和 S1 共享逐字节相同的公共 AuthoringPacket。Qwen v3/Flash 的正式响应是不可改写的历史负结果：provider、usage、费用和隔离合格，但 payload 有 40 个 schema 错误，故 `draft_formal_eligible=false`；该响应不得人工补字段、自动修复、重试或切换 Plus。项目所有者随后批准的 Qwen 接口修订已由 [`authoring-structured-submission-deviation-v1.json`](../specs/authoring/authoring-structured-submission-deviation-v1.json) 和 [`authoring-freeze-lock-v4.json`](../specs/authoring/authoring-freeze-lock-v4.json) 前瞻冻结。DashScope 的 JSON mode 只提供 `json_object`，不提供本任务需要的 response-format JSON Schema；因此 Qwen v4 改用 provider 支持的强制单函数提交：模型必须恰好一次调用不可执行的 `submit_authoring_payload`，其 parameters 是展开后的精确无哈希 `AuthoringDraftPayload` schema，assistant text 与并行 tool calls 均禁止。provider framing 仍不被当作 schema 正确性证明；受信 Runner 严格解析 arguments、执行 Pydantic 与规则/工具权限校验，再负责 canonicalization、输入绑定和摘要。

Qwen v4 保持 taxonomy、Task Specification、ToolSpec、空公开资料 allowlist、空 Reference Skill、默认模型 `qwen3-vl-flash-2026-01-22`、temperature 0、top-p 1、seed 20260722、non-thinking、32,000/8,000/40,000 token、US$0.05 和 30 分钟人工审查上限不变，并显式传递 `enable_thinking=false`。由于旧响应已经存在，本 deviation 不提供 Plus 变体或任何事后 fallback。预算账本诚实区分 1 次已知可审计 provider response、1 次 provider generation/billing outcome 未知的历史 incident，以及最多 1 次新增 provider attempt；新增授权在容器启动前通过 create-only claim 原子消耗，失败、崩溃或 schema 拒绝均不得删除 claim 后重试。LLMStatic authoring sandbox runtime 为 `formal-v3`，由 `deploy/authoring/locks-v4` 的 image、profile、network policy、deployment receipt 和四项 egress 探针共同闭环；它不等于仍待 C3 签发的 authority-issued tool registry runtime。

唯一 Qwen v4 attempt `llm-static-primary-20260724-structured-v1` 已执行并永久消费 claim。Flash 身份、`finish_reason=tool_calls`、空 assistant text、单个正确 submission、usage、费用与隔离合格：18,839 input、1,762 output、20,601 total token、821 microUSD。provider 将 `drafts` 数组二次 JSON 序列化为字符串，严格外层 schema 因此报 1 个类型错误并拒绝 draft。只读诊断即使临时解析该字符串，内部 6 个 draft 仍有 16 个校验错误；这说明不能把 deterministic unwrap 当作无损充分修复，但也不能把全部内部错误都归因于 provider：函数 parameters 没有表达自定义词法排序/唯一性 validator，提示词要求复制 Task Specification coverage 却没有要求这些数组词法排序，而冻结 Task Specification 的 6 个源数组顺序又与 Runner 的排序要求冲突。其余错误包括错误键名以及把 precondition/safety rule 映射为 step success rule，仍属于明确的结构或语义失败。正式 [`adjudication receipt`](../specs/authoring/llm-static-primary-20260724-structured-v1-adjudication-receipt.json) 固化这项归因，invocation receipt 记录 `formal_provider_call_eligible=true`、`draft_formal_eligible=false` 和 `retry/fallback/repair=false`。不生成 pre-review draft，不调用 Plus，也不在原授权下重试；C1 仍未通过。

Codex v4 的独立 run `llm-static-codex-primary-20260724-high-v4` 已执行并消费 claim。进程、事件、结构化 final 和工具/规则覆盖均正常，但 raw output 两次使用 trusted compiler 保留的独立词 `label`，因此状态为 `codex_session_completed_draft_rejected`。TaskSpec 与 ToolSpec 本身要求模型理解 detector 的同名字段，而 prompt 未披露输出侧词法禁令；只读 A/B 仅替换这两个词即可让同一六能力 payload 通过。该结果被归类为 prompt/validator 可见性冲突，receipt file SHA-256=`155a398b1133a917d278ef05743835c2ce68f035ff5ff90f72f7d32550fd644e`，不得修补或重分类。

owner 随后授权以新 run/freeze 继续。v5 保持 taxonomy、TaskSpec、ToolSpec、`gpt-5.6-sol/high`、空公开资料、空 Reference Skill、session budget、runtime 和零 retry/follow-up/repair/fallback/tool 不变，只在 Author 可见 prompt 中规定对象检测类别预测的允许措辞。freeze file/payload SHA-256=`fcc3b86957fe74b725b837cabb8c6b782d9be713799c1f5380fa0fff7c6d4060` / `defd21b77b883f7db1a487de799f44f321482701086d7ece6325ab55c0fa976a`，并绑定完整 v4 历史。

v5 run `llm-static-codex-primary-20260724-high-v5` 已生成 `formal_codex_session_eligible=true` 的 pre-review draft：exit 0，1 final，0 visible tool/retry/follow-up/repair/fallback，34,455 input / 1,515 output token。raw/pre-review/bundle SHA-256=`90fad5c27e6bb8a9122b470cfdef5a427bb4c0eb793d52ce4ffb52ed89b73a6b` / `e0c26c1c054514bfc4629fc4e84557da63f6b05335ca6c3b8709efe0a0e08908` / `41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3`。canonical bundle 已从磁盘独立重放完整 binding chain、事件、raw final、trusted compiler、进程终态和 receipt 自哈希。token 仍只是 platform-reported、non-provider-attested；served revision、provider request ID、backend attempt 和费用不可得。AI 复核不能替代项目所有者人工 checklist，真实 C3 runtime 前也不能编译 Bank，因此完整 C1 仍未通过。

公共 AuthoringPacket 的组成原则为：

- 冻结 Task Specification、ToolSpec；
- 许可允许、revision/hash 固定的公开资料；
- 固定模板、输出 schema、字段顺序和 compiler 版本；
- 若 Creator 需要 reference Skill，则包含两种 authoring 共同可见、内容固定且单独哈希的 `ReferenceSkillBundle`；
- 作者模型 identity、解码参数、call/token/cost budget；
- schema、工具权限、引用和安全 checklist；
- 每个输入的 path-independent logical ID 与内容 hash。

禁止字段包括任何 pilot/opt/val/test/challenge/tool-gold/judge 样本或摘要、轨迹、标签、响应、split、除共同冻结 `ReferenceSkillBundle` 外的 Bank/Skill、rubric/gold、Judge prompt/output、gate 结果、指标、错误分析，以及可通过路径/缓存/日志间接恢复这些内容的信息。

S1 的唯一额外 treatment 必须封装为独立 `TrajectoryBundle`，与公共 packet 分开 hash。任何 reference Skill 必须进入两种 authoring 共用的 AuthoringPacket；若 S1 独占 reference Skill，RQ1b 只能降级为复合 pipeline 比较，不能声称隔离了轨迹价值。静态 authoring API 不得有 corpus/eval 参数；allowlist serializer 和污染攻击测试必须证明禁用字段无法进入 prompt。

### 4.2 预算与人工审查

- `LLMStaticSkill` 与 S1 使用同一已选模型、surface、reasoning effort、单 session/单合格 draft 上限和用户审查分钟上限。前瞻 Codex v4 候选计划硬锁 1 session、0 follow-up/repository retry/repair/fallback/visible tool activity、600 秒、64 KiB final、16 MiB event log 和 30 分钟人工审查；30k/6k/36k token 与 3000 microUSD 仅作披露用比较目标，不能冒充 CLI 已强制或 provider 已回执的预算。若 `turn.completed` 提供 usage，只能按 platform-reported、non-provider-attested 披露。失败或拒绝同样消费该 session；历史 Qwen v3/v4 已拒绝响应、superseded Codex v1/v3 和 preclaim-rejected Codex v2 都不进入 Bank、人工编辑或结果选择，也不能被解释为“从多个候选中选更好者”。
- 作者可见 ToolSpec 必须能由 production runtime 原样执行。历史 TaskSpec v0 的 `product.multi_search` 曾允许 `object_detect` + `image_product_search`，但后者不能消费 detector crop；该缺口已经在候选 TaskSpec v1/registry v2 中以前瞻方式修复：唯一 operator 是 authority-issued `multi_product_search@1.0.0` composite ToolSpec，只接 authoritative `asset_id`，由 Runner 内部持有 detect→private crop→retrieval。这个状态只关闭代码/fixture 机制；新的 authoring freeze 必须绑定 exact TaskSpec v1/registry v2，且真实 C3 artifact/runtime 双锁、RPC gold、production 端到端 evidence 与 authority receipt 仍须通过，不能把候选 composite 名称当成已签发工具。
- 二者使用相同 checklist。用户只校验 schema、工具权限、公开来源引用与安全，不凭效果添加领域技巧。
- 任何实质性人工内容改动必须保存 before/after、理由和分钟数；超时或越界使该 Bank 失去主比较资格。
- S1 的轨迹输入 token、Engineer Loop token、总费用和机器时长另行报告，不因共同上限而隐藏 treatment 成本。
- 每个正式 authoring run 只允许一个 create-only claim，claim 在模型进程启动前即消耗；网络失败、provider outcome 未知、schema 拒绝或进程崩溃都不得在同一 run 自动重试、fallback 或 repair。历史 Qwen v4 与 Codex v4/v5 都使用独立的前瞻 freeze/authority；v5 成功后不得重新抽样挑选版本。

## 5. 数据 profile 与 capability 资格

| Profile | disjoint 主划分 | 用途与最低声明 |
|---|---|---|
| `mini` | `dev_mini=200` | 离线纵切、接口/泄漏/成本/Judge 精度验证；不发布总体效果结论 |
| `core` | `dev_mini=200 / opt_pool=800 / val=200 / test_frozen=300`，总计 1,500 | 个人项目主目标；至少 5 个合格 capability；只称小规模机制复现 |
| `full` | `dev_mini=200 / opt_pool=2,800 / val=500 / test_frozen=1,000`，总计 4,500 | core 全部门通过后的扩展目标 |

challenge 与主划分独立：core 100–150，full 100–300，并预拆 `challenge_dev/challenge_final`。正式 profile 必须在查看 test 主结果前，根据 mini 的人工工时、费用、工具质量、Judge audit 和 bootstrap 精度模拟选择。

2026-07-24 的 RAW 获取检查点只关闭传输层缺口：MVP profile 的 34 个活动项已全部完成，其中 31 个本地字节项合计 82,247,477,174 字节，另有 3 个 command 管理项。固定提交的 DuRecDial 2.0/CrossWOZ、SROIE、CORD v2 和 RPC Kaggle v5 包均已完成本地验证；RPC 的本地 SHA-256/结构收据见 [`rpc-kaggle-acquisition-v1.json`](../specs/data_sources/rpc-kaggle-acquisition-v1.json)。FashionIQ 以 75,267 accepted + 2,416 explicitly excluded 完整覆盖固定 77,683 条清单，被排除身份不得进入 gallery、query、正负例或指标分母。JDDC 2.0 已永久退出所有 profile，不再计作缺失或阻断。MVP 对话轨迹改用 DuRecDial/CrossWOZ/MUGE 的抽象交互模式与 Codex 受控领域重写，所有输出必须在 batch manifest 和质量报告中标记 `data_origin=synthetic_derived`，并保留逐条 synthesis provenance。它们不能提供 capability、商品、图片或事实 gold，也不能支持“复现 JDDC 2.0 真实流量规模或原始多模态用户分布”的外部有效性结论。该检查点不改变任何 capability 资格、split、样本配额或冻结门；只有后续 source lock、许可/PII、selection/disposition、Asset/KB catalog、人工 gold、Mock 人审和 leakage gate 全部通过后，对应来源才可进入本协议的正式数据 profile。2026-08-01 起 U-NEED 也因无法获得而永久退出；其候选机制只可由上述 `synthetic_derived` 替代方案承接。Full 对话实现待其余候选授权包及条款实际到手后另行冻结，不从当前候选清单自动推导。

当前 v0 taxonomy 恰好包含以下 6 个 MVP capability；正式 evaluator 从冻结 artifact 读取该集合，不能从候选 Bank 推断：

| `canonical_capability` | 顶层 intent | `requires_card` | Task Specification 允许的工具 |
|---|---|---:|---|
| `knowledge.visual_encyclopedia` | `encyclopedia` | false | `encyclopedia_lookup`, `object_detect` |
| `product.exact_match` | `exact_match` | true | `image_product_search`, `text_product_search` |
| `product.multi_search` | `multi_product` | true | `multi_product_search` |
| `product.style_recommendation` | `divergent_rec` | true | `style_similar_search` |
| `utility.document_reading` | `utility` | false | `document_ocr` |
| `utility.recipe_guidance` | `utility` | false | `object_detect`, `recipe_lookup` |

该快照绑定 taxonomy semantic hash `af23dbe76e8c0cffdbb056b44c6595af6c508eaa0158f96227cbc92e244ca315`
与前瞻 Task Specification v1 semantic hash
`f8d5596de5d0ba98235f82c7c176a5b774b33d7bdd7e84fb00a07b5b0b7a7f0d`，且后者内部绑定
前者。新 Phase 3 语料、S1 Creator 与前瞻 authoring 统一使用 v1；v0 只保留为历史 replay
兼容面。若 Exact Match 在系统结果可见前因真实多视图资格失败而移出主表，剩余 5 个
capability 仍须分别满足样本、工具和质量门，才可达到 core“至少 5 个”的最低资格；
Exact Match 本身只作 exploratory/移除，不能因总数仍够而放宽其 eligibility。

正式主表 capability 在扩量前冻结：

- core 每个 capability 的 opt/val/test 至少 `60/30/30` 条；full 至少 `100/40/50` 条；
- 尽量覆盖至少两个 source/template family；不足时明确标记单源限制；
- route_gate/body_gate 按 capability 分层，core 每个 gate 目标 ≥10，full ≥15；用于分组推断的硬最小 n 在 v1 由 mini 精度模拟冻结；
- 不达资格的 capability 只作描述性/exploratory 报告，不进入 macro 主结论，也不能通过复制或近重复样本补足；
- Exact Match 只有在存在同 product、不同真实 asset、非近重复、非 derivation 的 query/gallery 正例并通过 eligibility 时进入主表；否则在读取系统结果前降为 exploratory 或移除。

所有数据先在联合 leakage component 上分组，再满足 intent/capability/boundary 分层。跨 split 的 asset、product、near-duplicate、source-record、derivation、boundary、template 或 generator component 交集必须为 0。

数据质量与 shortcut gate 同样在系统结果前执行：schema pass 和需要事实样本的 evidence coverage 都必须为 100%；mini 人工抽样事实支持率目标 ≥95%、capability label agreement 目标 ≥90%，并完整报告模板近重复、repair/reject 和图文不一致。val/test 由用户 100% 人工确认；全部 boundary 和随机 10–20% 非 boundary 样本在至少 7 天洗脱期后盲重标。natural ambiguity 必须保留至少两个合理解释或 acceptable capability 的具体依据。

source/template/generator/图片尺寸等非语义 shortcut 的 macro-F1 达到 `max(2×chance, majority+0.20)` 时，数据集阻断；必须在看系统主结果前重平衡/重建，或将受污染 capability 降为 exploratory。image-only/text-only 与完整输入差距的阈值 `epsilon_cross_modal` 在 v1 冻结；差距低于阈值时加入跨模态冲突 challenge，并将其作为对应 capability 的主要稳健性结果。

## 6. 数据可见性与允许决策

| 数据/产物 | 何时可见 | 可用于 | 严格禁止 |
|---|---|---|---|
| 公共 Task Spec/ToolSpec/资料 | authoring 前冻结 | LLMStaticSkill、S1 的共同公共输入 | 候选结果反向改写 v0 语义 |
| `dev_mini` 200 | v0 实现期 | 工程纵切、成本/人时、CI 精度模拟、一次协议修订 | 论文效果结论；伪装成 formal test |
| `opt_pool` | formal evolution 期 | S1 预注册轨迹、S2 失败挖掘、S3 attribution | final evaluation；回写标签以迎合结果 |
| `route_gate` | 每轮 S2 候选完成后 | 只决定 Stage 2 接受/回滚 | Stage 3、Task Spec/rubric 修改、最终报告替代 test |
| `body_gate` | 每轮 S3 候选完成后 | 只决定 Stage 3 接受/回滚 | Stage 2、路由修改、Task Spec/rubric 修改 |
| `shadow_val` | val 冻结后可自动运行 | 只生成监控报告 | 选候选、调 prompt/阈值、提前停止或回滚 |
| `test_frozen` | 所有系统/Bank/evaluator 冻结后的 sealed job | 一次 canonical paired final evaluation；预注册 repetitions | 任何创建、改写、门控、排除、补样或再冻结 |
| `challenge_dev` | 开发期 | 跨源/人写诊断；正式冻结后只读 | 与主 test 池化；final evolution 的候选选择 |
| `challenge_final` | 与 test 同时解封 | 一次独立 final robustness 报告 | 调参、改 Bank、回写 Task Spec；与 challenge_dev 混合 |
| `tool_test_frozen` | 工具 artifact 冻结后 | 独立 formal 工具质量和 go/no-go | Assistant 训练/val/test；与其资产 component 重叠 |
| `judge_calib` | final prompt 冻结前 | 调 Judge prompt/rubric、确定数值校准门 | 报正式效果或替代 audit |
| `judge_audit` | 五配置 mini 输出全部冻结后 | 一次 evaluator 审计、决定能否进入 formal | 在解封前调 prompt；解封后在同 audit 上反复优化 |

正式 val 在首次使用前按 group 固定拆分：core 为 `route_gate=80 / body_gate=80 / shadow_val=40`；full 为 `200/200/100`。三个子集互斥，不因某轮样本不足重新分配。

### 6.1 test/challenge 解封次数

- `test_frozen` 对研究者的逻辑解封次数为 **1**；五主配置、已预注册诊断配置和已冻结消融在一个 sealed evaluation schedule 中运行，所有计划运行结束前不暴露中间分数。
- canonical run 每个主配置执行 1 次。预算允许时，可在解封前预注册 NoSkill、S1+S2、Full 各追加恰好 2 次 repetition；固定输入顺序，不能先看 canonical 结果再决定是否重复。
- `challenge_final` 的逻辑解封次数同样为 **1**，与 test 同阶段但单独 hash、单独报告，不与 test 合并。
- 只有在没有生成可用 model output 的基础设施故障时才允许重跑；必须保存 aborted run、故障证据，并对受影响的完整 paired group/全部配置一致重跑。模型拒答、timeout、tool error、parse error 或低分不是基础设施故障。
- 解封后新增配置、专家基线或修订系统必须使用新的密封 holdout，或明确标为 post hoc；不得复用旧 test 伪装预注册结果。

## 7. 指标

### 7.1 响应质量主指标

冻结 rubric 对所有配置相同，逐样本输出：

- Task Completion Rate (`TCR`，0–10)；
- Card Content Correctness (`CCC`，0–10，仅在 `requires_card=True` 时适用)；
- Content Quality (`CQ`，0–20)；
- Content Authenticity (`CA`，0–10)。

本项目标量指标为：

```text
J_project = 100 × (TCR + CQ + CA + applicable_CCC)
                  / (40 + 10 × requires_card)
```

`requires_card=True` 且未出卡时 `CCC=0`，不能记 N/A；`requires_card=False` 时 CCC=N/A 且不进入分母。四维均值、适用样本数和 `J_project` 必须一起报告，不得把 `J_project` 称为论文 Table 2 Avg。主响应比较使用逐 query 的配对 `J_project` 差；安全/真实性维度出现实质回退时，即使聚合分数上升也必须单列，不能只报总分胜利。

因此无卡样本的可用维度满分为 `10+20+10=40`，需要卡片的样本再加 `CCC=10`，满分为 50；实现和手算 fixture 必须同时覆盖两条分母路径。

### 7.2 路由主指标

- 主指标：`canonical_capability` macro-F1；不具备正式资格的 capability 不进入 macro 集合，集合在结果前冻结。
- macro-F1 始终使用冻结 capability universe；某个 replicate 中无预测或无真阳性的类别按 `zero_division=0` 处理，不能动态缩小类别集合抬高分数。
- 必报辅助指标：capability per-class P/R/F1、混淆矩阵，顶层 intent macro/micro/weighted F1。
- boundary 同时报 strict hit 与 acceptable-set hit；strict 为主，acceptable-set 为辅助。
- `NoSkill` 无路由，Routing F1 为 N/A，不得补造默认 route。
- T3/OOD（后续协议）报告 no-skill precision/recall、coverage-risk 和错误接收率。

### 7.3 辅助指标

- 盲 SBS：Full vs LLMStaticSkill、Full vs S1+S2 的 win/tie/lose；core 100–150 条，full 100–300 条，随机左右顺序。单人只报告带洗脱期的 intra-rater；有第二位真人才报告 inter-rater。
- 运行可靠性：成功率、拒答、timeout、tool error、Judge parse/error 和重试率。
- 资源：作者/Assistant/Judge tokens、图片数、缓存命中、机器时长、人工分钟、货币费用、p50/p95 latency、Bank 条目数与 tokens。
- 分组诊断：intent、capability、source、boundary、requires_card、tool-use、route-correct/deviation；未达到冻结最小 n 时只描述原始 n 和样本，不做非回退推断。
- run-to-run variance：预注册 repetitions 与样本 bootstrap 分开报告；Judge serving variance 在独立校准样本上测量。

### 7.4 工具独立指标

工具结果只来自验证过的 typed formal run：

- image/text product search：Recall@K、MRR、nDCG；
- style search：池化分级相关性的 nDCG 与多样性指标；
- KB lookup：hit、citation span 正确率、evidence support/coverage；
- detection：label+bbox AP50，并报告失败率；
- OCR：CER、字段准确率、grounded-line evidence coverage；
- Multi-Product：`detect → canonical crop → per-object retrieval` 端到端成功率，不能由 AP50 或单次 retrieval 代替；
- 全部工具：p50/p95 latency、错误码分布和资源成本。

mini 每个 canonical 工具 10–20 条 gold 只验证纵切；core 每工具 30–50 条，full 每工具 50–100 条真实人工/公开 gold。全部七个工具都必须覆盖，ranking、detection、OCR 基础任务进入 `tool_test_frozen`；Multi-Product 另有端到端 gold。每项 formal go/no-go 阈值必须由真实 mini gold 的可行性与误差分析在 v1 一次性冻结；`TBD`、fake fixture 或 diagnostic benchmark 不能通过 core gate。

## 8. 统计方法

### 8.1 主估计

- 所有主比较是同 query、同输入顺序的 paired estimand；先报告各配置绝对值，再报告预注册方向的配对差值。
- 95% CI 使用 **10,000 次 paired cluster bootstrap**：以权威联合 leakage component 为 cluster，有放回抽取 cluster，保留 cluster 内全部 query 和配置配对。随机种子写入 RunManifest。
- 路由 macro-F1 在每个 bootstrap replicate 上从预测重新计算；缺失预测按错误类别处理，不从样本删除。响应质量对 `J_project` 差取 percentile 2.5%/97.5% CI。
- test 与 challenge_final 分开估计，不合并；core 与 full 也不跨 profile 合并。

对 RQ1a、RQ1b、RQ2、RQ3，只有预注册主差值的双侧 95% CI 下界大于 0，才表述为“观察到改善证据”；CI 跨 0 表述为“未建立改善”，上界小于 0 表述为“观察到回退证据”。这不是项目交付成败门。RQ2 在 test 上判断主差值，source-held-out/challenge_final 只作单独的稳健性结论；不得通过池化掩盖某一集合的回退。响应聚合改善但冻结硬安全维度越界时，结论必须写成 mixed/unsafe，不能只声称总体改善。

### 8.2 检验与多重比较

- RQ2 除 macro-F1 配对 bootstrap 外，报告逐 query strict-correct 的配对列联表和双侧 McNemar 检验；混淆矩阵始终给出。
- RQ4 的四个消融比较构成一个 family，p 值用 Holm 方法校正；同时报告未校正效应和 95% CI，不能只给显著性。
- capability/source/boundary 拆解默认是描述性分析。只有达到 v1 冻结最小 n 的 capability 才执行预注册非回退门；其余不作显著性声明。
- SBS 报 win/tie/lose 与有效 n；双侧 sign test 排除 tie 但必须保留 tie 数，并给 cluster bootstrap 的净胜率 CI。
- 不用多个 seed、多个 Judge 或多个指标中“最好看的一个”替代预注册主结果。探索性分析显式标为 exploratory。

### 8.3 接受/回滚门

Stage 2 只使用 `route_gate` 的 capability macro-F1 配对差；Stage 3 只使用 `body_gate` 的 `J_project` 配对差及硬安全规则。v1 必须通过 mini 精度模拟预先填写：

- `delta_route_min` 与允许的非回退界；
- `delta_body_min` 与每个硬安全维度的最大允许回退；
- 分组门的最小 cluster/query n；
- Judge/工具/cost go/no-go 数值。

门槛未填、样本不足或 CI 不满足时，候选回滚或只标 exploratory；不得从 `shadow_val`、test 或 challenge_final 借证据通过。

## 9. 排除、缺失与错误规则

### 9.1 唯一允许的样本排除

排除必须在任何配置输出可见前，由统一 validator 对所有配置一次性决定并写入含原因/hash 的 exclusion manifest。只允许：

1. source revision、license、用途权限或 PII 审批不满足；
2. 原始 asset 字节损坏/缺失，或 query 无法从冻结输入重建；
3. 权威 group audit 发现跨 split leakage、同 asset/near-duplicate、错误 derivation 或 eligibility 违规；
4. 人工在盲于配置输出时确认 Task Specification 本身不可判定，且没有可冻结的 acceptable set；
5. 预注册 capability 资格在结果前未达到。

所有排除同时报告原始 n、排除 n、原因、group 和时间。不得因为模型失败、Judge 不同意、工具错误、响应低质或结果不利而排除；不得只对某配置排除。

### 9.2 错误保留

- Assistant timeout、provider error、拒答、tool failure、空响应和 schema error 都保留在分母；使用实际 fallback/可见响应评分，无可见响应时各适用质量维度按 0 计并单列错误类型。
- `requires_card=True` 的缺卡固定 `CCC=0`。
- route 缺失/非法/不可映射按 strict 与 acceptable-set 均错误；不计为 no-skill 成功。
- final Judge 按冻结次数和完全相同的重试策略处理；重试后仍 parse/error 时，主分析对各适用维度固定按 0 计，同时报告 Judge-error rate 和盲人工敏感性分析。最大尝试次数在 v1 冻结，不能按配置或结果变化。
- 工具 benchmark 的失败作为失败结果，不能从 latency 或准确率分母消失；formal evaluator 只接受 verifier 重新校验的 typed run。
- 缓存命中与未命中共享同一内容语义；可写 cache、自算摘要、手工 hits 或协调篡改 manifest 不能生成 formal 结果。

### 9.3 基础设施故障

只有独立于配置、且在任何可用模型响应产生前发生的损坏（例如 runner 崩溃、不可恢复磁盘 I/O）可标为 infrastructure invalidation。必须保留原 run 和证据，对受影响的完整 paired group 及全部配置重跑。API timeout、内容拒答、工具失败、模型格式错误或分数差是系统结果，不是可重跑借口。

## 10. Judge—human 校准与盲化

- 从独立 grouped 样本预拆 50–100 条 `judge_calib` 与密封 `judge_audit`；二者 group 零交集，并且都与 formal `test_frozen/challenge_final` 的联合 leakage component 零交集。
- calib 可用于调整 rubric/prompt；audit 在五配置 mini 输出冻结前不可读。解封 audit 后不得在原 audit 上反复调参；未达 v1 数值门时，补建新的密封 audit 样本后才可再次校准。
- final evaluator 只接收 allowlist `FinalEvaluationPacket`：用户实际可见图文、response/cards、完成任务必需且净化的 tool evidence、冻结 rubric。
- packet/prompt 明确禁止 config、Bank/skill slug、Description、Body、GT capability、source、split、stage/lineage、轨迹、分数和能恢复身份的路径/cache key。
- feedback 使用独立 `FeedbackPacket`、prompt 和 cache namespace。当前 `evaluator-isolation-v3` 将 DashScope `kimi-k2.6` Feedback 与 AIFast `gemini-3.6-flash` final 锁为不同 provider runtime 和模型家族，同时保留输入投影、prompt、cache 与 artifact 隔离。该声明不消除第三方网关身份风险，也不能替代盲 SBS 人评和 Judge—human 校准。
- v1 数值门至少覆盖：维度级一致性、Spearman/Kendall、系统性偏差、重测 variance 与 Judge-error rate；门值由 mini 校准精度和人工资源在 audit 解封前冻结。

Judge audit 未通过时，不得进入正式 benchmark。可继续改实现和收集新的独立校准/审计集，但旧 audit 不再密封，不能被重复用作通过证据。

## 11. 停止条件与 go/no-go

### 11.1 立即停止并禁止扩量

出现任一条件即 fail closed：

- 任一 P0 没有真实 artifact/manifest/测试/人工审批证据；
- 任一 P1 既未有验证实现，也未在 Reproduction Contract 中明确降级并从对应主结论排除；
- taxonomy/Task Specification/AuthoringPacket 可被候选 Skill、pilot 响应、Judge 或结果字段污染；
- 任一联合 leakage component 跨 split，或 tool gold 与 Assistant val/test 资产 component 非零交集；
- Exact Match 无合格多视图正例却仍被放入主结论；
- formal tool artifact、gold、外部 digest、Judge audit 或盲化测试未通过；
- LLMStatic authoring 未在只暴露冻结 packet 的隔离 job 中执行，或 Assistant 的模型/请求/usage/latency/tool trace 仍由 backend response 自报；
- profile 的 capability 样本资格、bootstrap 精度、人工工时或预算超过 v1 上限；
- test/challenge_final 被提前读取，或结果被用于修改数据、系统、Bank、rubric、prompt、排除规则或协议；
- README/报告把 LLMStaticSkill、SpecBaseline 或用户格式审查冒称为 ManualSkill。

### 11.2 阶段停止

- Stage 2 最多 4 轮，Stage 3 最多 3 轮；达到轮数、无合格更新样本、预算/人工上限或所有候选 gate 失败时停止并保留最后一个通过版本。
- Stage 2 每 Skill <30 个真实失败或 Stage 3 每 Skill <50 个真实 attributed responses 时，不执行相应正式更新；可做明确标注的 exploratory 纵切。
- `shadow_val` 不参与提前停止。test/challenge_final 运行后不再继续演化。
- mini C1–C5 未通过时不构建 core/full；core 未通过时不启动 full。预算紧张时缩 profile/capability 或诚实停止，不降低泄漏、许可、gold 或 evaluator 门。

### 11.3 formal go/no-go 记录

启动 core 前必须生成 tracked 决策记录，包含所选 profile/capability、实际 n、人工工时、成本上限、工具阈值与结果、Judge hash/门、CI 精度模拟、全部 P0/P1 状态和保留局限。任何 `TBD` 都导致 no-go。

## 12. 一次修订、冻结与偏离报告

1. `v0` 在候选 Skill、pilot Assistant 响应或 Judge 结果参与决策前定义固定部分。taxonomy/Task Specification v0 只能来自冻结公共输入，不能从候选 Body/结果反推。
2. mini 纵切后允许一次批量修订为 `v1`：填写本协议第 1 节预先声明的实测参数，并发布 diff、过程性理由、影响、新 hash 和批准时间。
3. 不允许修订的固定项包括：RQ、五个主配置、主比较、主指标、authoring 禁区、group split、错误保留、test/challenge 密封和 ExternalExpertManual 命名/provenance gate。若这些项必须改变，建立新协议 ID 和新 holdout。
4. `v1` 必须在正式 benchmark 生成和任何 test/challenge_final 解封前冻结。dataset、rubric、prompt、model、tool、Bank、code 和 environment hash 一并写入 lock manifest。
5. 冻结后的实现 bug 不做静默热修：保存 incident、影响和原 run；若影响评分或 treatment，原实验失效并在新协议/未见 holdout 上重跑。
6. 所有偏离都进入 deviation log，至少包含发现时间、是否已见结果、影响配置/样本/指标、处理和声称降级。未登记偏离使正式结果不可发布。

本文的 `规范化内容 SHA-256` 使用与 `reproduction-contract.md` 第 11.1 节相同的唯一算法：UTF-8 文本先将 CRLF 和单独 CR 统一为 LF；在首个二级标题之前严格匹配 ``^> \*\*规范化内容 SHA-256：\*\* `[0-9a-f]{64}`$``，且全文必须恰好一行；删除该行及其行尾 LF（若有），保留其他全部内容，不做 Unicode 归一化，再以 UTF-8（无 BOM）编码并计算 SHA-256。独立 protocol lock manifest 还要再次绑定该值，缺失、重复或不一致均 fail closed。当前状态是 `v0-draft`；capability/Task Specification、formal SpecBaseline、污染哨兵和隔离 Runner 已实现。Qwen 与 Codex v4 负结果保持不可改写；Codex v5 已取得可独立重放的合格 pre-review draft。C1 仍等待 owner 人工 checklist 与 C3 后 Bank compile，C3 authority-issued tool registry runtime 也仍待签发；协议本身还缺最终 hash、批准时间和 mini 后唯一一次冻结证据。

## 13. 2026-07-20 实施状态与禁止解释

当前代码以四个外部摘要共同验证 query/split/rubric/audit selection，并能生成五配置 diagnostic Assistant bundle、只从 verified input/plan/run 组合 final packet、盲化 evaluation ID、错误分母、Judge score schema 和 sealed-audit commitment。FeedbackPacket/FinalEvaluationPacket 使用隔离的 cache namespace；两者的 image snapshot 只保存 MIME+SHA，Base64 仅在 verified runtime 的瞬时 `image_url` wire 中存在，provider 若回显 input image Data URL 或原始编码则内容不落盘并以 `input_image_echo` fail closed。当前 v6 visual Feedback/final runtime 分别强制消费 `dashscope-kimi-feedback` / `aifast-gemini-judge` preflight；Kimi Feedback 使用冻结 plain-JSON transport 和 free-text-trim parser，Gemini final 请求 `json_object` transport、只接受原始整数维度分数并在本地确定性计算 tier/`J_project`。两类结果回执绑定 catalog/authorization/receipt/provider/request/usage/latency，加载时从未删节 raw response 重建 parsed output/final scores，并支持 create-only 落盘重验。历史 v5 Gemini-Feedback/Kimi-final 回执只按其原身份诊断，不重签或解释为 v6。formal tool evaluator仍会深度重验外部 digest、人工 gold-review ledger、tool↔Assistant component 隔离、runtime authority、证据绑定和完整工具覆盖。fixture 只用 synthetic query 验证工程接口与同环境重建；其 Python 进程阻断 socket/DNS，但不是 OS 沙箱，也不估计任何论文效果。Linux workflow 尚无外部 run receipt，checkout/setup/依赖安装不受该 guard 约束。

production Assistant runner-owned 采集机制已实现；Kimi visual Feedback 与 Gemini final 的离线 wire、严格 parser、fail-closed runner 和单结果 create-only receipt 已实现到本地代码层。既有 Gemini smoke 只证明当时 AIFast Feedback transport 连通，不证明新 final Judge 已获资产权限、价格冻结或完成校准；角色交换验收尚未进行 provider 调用。五配置 1,000 行批量编排与矩阵 manifest、真实 200-query mini、真实 tool gold、approved audit sealer 与 50–100 条 Judge—human 校准仍缺。在这些完成前，本协议中的阈值、成本、人工分钟和 Judge hash 保持未冻结。任何 synthetic 分数、backend 自报执行信息、测试 double 成功率或 fixture tree digest 都不能填入 formal 结果表。当前决定为 **core NO-GO**，证据与启动条件记录在 [`go-no-go/2026-07-20-core-no-go.md`](go-no-go/2026-07-20-core-no-go.md)。
