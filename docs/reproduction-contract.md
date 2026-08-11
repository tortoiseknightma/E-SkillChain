# SkillChain 公开数据机制级复现契约

> **契约 ID：** `skillchain-open-reproduction-v0`
> **状态：** `v0-draft`；尚未授权 formal/core 实验
> **制定日期：** 2026-07-20
> **实现基线：** `master@a80af22` + `codex/authoring-codex-high-approval@1ea786a`
> **上位实施计划：** [`plans/2026-07-09-skillchain-reproduction.md`](plans/2026-07-09-skillchain-reproduction.md)
> **收口计划：** [`plans/2026-07-20-p0-p1-closure.md`](plans/2026-07-20-p0-p1-closure.md)
> **评价协议：** [`evaluation-protocol.md`](evaluation-protocol.md)
> **taxonomy：** `ecommerce-mvp-taxonomy-v0`，semantic SHA-256 `af23dbe76e8c0cffdbb056b44c6595af6c508eaa0158f96227cbc92e244ca315`（`specs/taxonomy/ecommerce-mvp-taxonomy-v0.json`）
> **前瞻 Task Specification：** `ecommerce-task-spec-v1`，semantic SHA-256 `f8d5596de5d0ba98235f82c7c176a5b774b33d7bdd7e84fb00a07b5b0b7a7f0d`（`specs/task_specs/ecommerce-task-spec-v1.json`）；v0 仅保留用于历史工件重放
> **规范化内容 SHA-256：** `9ef5a9d6e61842c85880b3857cb41e2b91c7f1388c702069b78c8ecd80be176d`

## 1. 契约目的与权威边界

本项目在无法访问论文生产数据、内部工具、原始模型组合和在线流量的条件下，使用公开数据、合规采集、LLM 合成与人工质检，复现 SkillChain 的三阶段**机制**。项目的准确名称是：

> **SkillChain 的公开数据机制级复现与受控扩展。**

本契约是“可以声称什么”的权威文件；它不是项目进度表，也不是实验结果。负结果、无显著增益以及与论文方向不一致的结果都属于有效结果。不得以获得论文走势或漂亮分数作为验收条件。

文件之间的职责和冲突处理如下：

1. 本契约规定复现范围、替代项、不可复现项和命名边界。
2. `evaluation-protocol.md` 规定研究问题、比较、数据可见性、指标和统计决策。
3. `specs/taxonomy/ecommerce-mvp-taxonomy-v0.json` 与前瞻
   `specs/task_specs/ecommerce-task-spec-v1.json` 规定新语料、Stage 1 和后续实验的任务语义；
   Task Specification 必须绑定 taxonomy 的 canonical semantic hash，二者均独立于候选 Skill
   和实验结果。TaskSpec v0 只服务于已冻结历史工件的兼容重放，不得再成为新 Phase 3
   语料或 Stage 1 输入的默认版本。
4. tracked manifest 绑定一次实际运行的精确数据、模型、工具、prompt、Bank、代码和环境。
5. canonical plan 只规定实施顺序。若计划文字与本契约冲突，执行更严格的约束并阻断 formal run，直至通过版本化修订消除冲突。

`v0-draft` 不等于冻结。只有第 11 节的冻结条件全部满足、外部 lock manifest 记录本文规范化摘要且状态变为 `v1-frozen` 后，本契约才能授权正式 benchmark。

## 2. 论文机制对齐

以下内容是本项目承诺复现的机制，不表示数据、模型或绝对数值等价：

| ID | 对齐项 | 本项目的可验证实现约束 |
|---|---|---|
| A1 | Skill 表示 | 每个 Skill 保留 `Description / Body / Cs / Od` 四元组、稳定 slug、版本、父版本与内容哈希。 |
| A2 | Stage 1 — Creator | 从预注册训练轨迹和公共 authoring 输入创建初始 Skill；保存输入、prompt、模型、原始输出、编译结果、Human Gate 与 lineage。 |
| A3 | Stage 2 — Route Optimizer | 只允许改变路由层：更新 `Description`，或执行有审计记录的 Update/Discard/Merge。未合并 Skill 的 `Body/Cs/Od` 必须逐字节不变。不能借 Stage 2 暗改执行规则。 |
| A4 | Stage 3 — Body Refiner | 只允许改变 `Body`；`Description/Cs/Od`、capability mapping 和已冻结的 Stage 2 route decision 不变。S1+S2 与 Full 必须复用同一份路由结果。 |
| A5 | 反馈闭环 | 失败归因、规则路径、定性归纳和统计聚合生成候选修改；候选只能经预注册 gate 接受或回滚。feedback 不能兼作 final evaluation。 |
| A6 | 累计配置 | 按 `NoSkill → S1 → S1+S2 → Full` 分离 Creation、Routing 和 Body 的增量贡献，并加入本项目的可比静态基线 `LLMStaticSkill`。 |
| A7 | 轮数与更新资格 | 正式主配置最多 4 轮 Stage 2、3 轮 Stage 3；论文阈值“每 Skill 至少 30 个路由失败”和“至少 50 个 attributed responses”作为更新资格，不得复制样本凑数。 |
| A8 | Stage 3 主阈值 | TCR `.20`、CCC `.10`、CQ `.05`、CA `.10` 作为预注册 Poor 阈值主配置，同时报告阈值敏感性。 |
| A9 | 核心消融 | 在读取 final holdout 前构建并冻结 `−LLM Judge / −Rule Path / −Qualitative Induction / −Statistical Aggregation` 四个 Bank。 |
| A10 | 收敛与成本 | 逐轮报告质量、路由、最差组、Bank 大小、token、费用和延迟；不只报告最终均值。 |

其中 A1–A10 是“机制对齐声明”。任何实现若放宽 edit boundary、让 final test 参与演化或让最终评价器读取 Skill/配置身份，都不能计为本契约下的机制复现。

## 3. 明确替代

| 论文要素 | 本项目替代 | 必须披露的影响 |
|---|---|---|
| 生产流量与内部电商数据 | 固定 revision 的公开数据、合规采集、grounded LLM 合成和人工确认 | 分布、难度、语言与长尾覆盖不同，不能比较绝对分数。 |
| 淘宝内部工具 | 本地、可审计的七工具 registry 与固定 artifact | 工具质量和失败模式不同；工具必须先独立 benchmark。 |
| 论文模型组合 | 可访问且在 manifest 中锁定的作者、Assistant、feedback 与 final evaluator 模型 | 模型能力是复现差异，不得隐去 provider/model/revision。 |
| 专家团队 Human Gate | AI 预审加用户 checklist 终审 | 只能声称“单人项目 Gate”；记录人工分钟和所有实质性 diff。 |
| 专家 ManualSkill 主基线 | 一次性、无轨迹的 `LLMStaticSkill` 主基线 | 回答的是轨迹价值，不是“自动方法是否超过专家手写 Skill”。详见第 7 节。 |
| 原始评价细节 | 独立冻结 rubric、盲化 final evaluator 与 Judge—human 校准 | `J_project` 是本项目指标，不声称等同论文 Table 2 Avg。 |
| 生产索引、检测、OCR、KB | 外部摘要锁定的本地/合法预计算 artifact | 正式报告同时给出独立工具质量；diagnostic 运行不能冒充 formal。 |

所有替代项都必须在最终报告的方法和局限中逐项出现。仅在 README 提及而在结果表中省略，不满足披露要求。

### 3.1 当前模型角色替代（Portfolio v5）

当前前瞻角色锁为：

| 角色 | 选择 | 必须披露的不可等价性 |
|---|---|---|
| Assistant | DashScope `qwen3-vl-flash-2026-01-22`，non-thinking | 与论文原 Assistant 不同；正式 run 仍需精确 request/usage/latency/tool receipt |
| Author | Codex CLI `gpt-5.6-sol`，reasoning effort=`high` | “一次调用”只能审计为一个 Codex session；served revision、provider request ID、平台内部 backend attempt、费用和 packet-only OS 隔离均不可证明；CLI token 仅可能作为 platform-reported、non-provider-attested usage |
| Final Judge | AIFast OpenAI-compatible `gemini-3.6-flash`，请求 `json_object` response format，不发送 temperature/top-p/top-k | 第三方网关不提供 Google provider-attested 身份；新的 owner authorization、价格/ceiling、production 回执与 Judge—human 校准尚未完成，live execution fail closed |
| Feedback evaluator | 百炼/DashScope `kimi-k2.6`；non-thinking，temperature=`0.6`、top-p=`0.95` | 必须读取原图、用户对话、可见响应/cards/evidence 与内部 tool trace；与 final Judge 使用不同 provider/runtime、模型家族、packet、prompt、cache、输入投影和产物 namespace |

2026-07-24 的 `kimi/kimi-k3` + DeepSeek、2026-07-29 的共享 `kimi-k2.6`、v5 Gemini-Feedback/Kimi-final 与 v6 Kimi-Feedback/Gemini-final 选择均保留在历史 `model-role-selection-v2` 至 `v6` 和冻结工件中，不做改写。当前 owner-selected Portfolio selection 是 [`model-role-selection-v7.json`](../specs/authoring/model-role-selection-v7.json)，file/self SHA-256=`fbfbe9437731052743b3025962a22e4d3cd6432c2212b7cc418b2118e9fd2ba4` / `fc3e8ad7a1b2ace95ffe3df4c775a8c38425274459853d555dabb2c836097633`：Feedback=`DashScope kimi-k2.6`，final=`AIFast gemini-3.6-flash`。v7 只前向强化 Feedback 响应 `schema_version=1` 与每项 1–4 条 evidence 的提示身份；parser-v3 和 output contract-v4 不变。它保持跨 provider/模型家族隔离，但新的 AIFast Judge owner authorization、价格/BOM、runtime relock、production 回执和人评校准仍未完成，因此不具备 Formal Research eligibility。

Codex Author 的正式命名必须是 `LLMStaticSkill-Codex` 或明确声明 `codex_mediated_static_author_v1`，证据等级为 `platform-mediated_non-provider-attested`；不得把它写成 OpenAI API/provider-attested 单次调用。LLMStaticSkill 与 S1 必须共享同一 Codex surface/model/high/session envelope，否则 RQ1b 不再隔离“是否额外看到训练轨迹”。一次授权只允许一个 session，失败、无效输出或 schema 拒绝也会消费授权；retry、follow-up、repair、fallback 和 tool activity 均为 0。Qwen v3/v4 Author 负结果继续作为不可改写历史；任何 Qwen candidate 或未调用 runtime 都不能迁移为 Codex authority。

Codex v4 的独立真实 session 已执行并因 Author prompt 未披露的 trusted-compiler 保留词冲突被拒绝；receipt file SHA-256=`155a398b1133a917d278ef05743835c2ce68f035ff5ff90f72f7d32550fd644e`，结果保持原样。v5 只前瞻新增允许措辞，其他公共输入与预算不变；freeze file/payload SHA-256=`fcc3b86957fe74b725b837cabb8c6b782d9be713799c1f5380fa0fff7c6d4060` / `defd21b77b883f7db1a487de799f44f321482701086d7ece6325ab55c0fa976a`，并绑定完整 v4 历史。

v5 run `llm-static-codex-primary-20260724-high-v5` 已消费独立 claim，生成 `formal_codex_session_eligible=true` 的六能力 pre-review draft。raw/pre-review/bundle SHA-256=`90fad5c27e6bb8a9122b470cfdef5a427bb4c0eb793d52ce4ffb52ed89b73a6b` / `e0c26c1c054514bfc4629fc4e84557da63f6b05335ca6c3b8709efe0a0e08908` / `41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3`。完整 canonical bundle 已从磁盘独立重放；仓库侧记录 0 retry/follow-up/repair/fallback/tool。

该成功不改变证据命名：served revision、provider request ID、backend attempt 和费用仍不可得，输入隔离仍不是 OS 级证明，所以只能称 `LLMStaticSkill-Codex` / `platform-mediated_non-provider-attested`。项目所有者人工 checklist 与真实 C3 runtime 后 Bank compile 尚未完成；pre-review draft 不得直接当作可执行 Bank，完整 C1 仍未通过。

FeedbackPacket/FinalEvaluationPacket v2 只保存 byte-verified、远程处理已授权 catalog 图片的 MIME+SHA；visual Feedback/final runtime 分别消费 `dashscope-kimi-feedback` / `aifast-gemini-judge` fresh preflight，再从同一 verified catalog 瞬时读取图片字节并生成 OpenAI-compatible `image_url` Data URL。Base64 不得进入文本、packet、prompt snapshot 或结果回执；若 provider 回显 input image Data URL 或完整编码，runner 只保留原响应 SHA/字节数并以 `input_image_echo` fail closed。Feedback 输出使用严格的 rule-violation/ideal-gap/image-evidence/suggestion JSON 契约；final 模型只提交适用维度的原始整数分数，本地实现确定性计算 tier 与 `J_project`。两类 create-only 结果回执必须绑定 packet/prompt/wire/image/catalog/authorization/provider/request/usage/latency；未删节响应在加载时必须重新解析，final 分数必须重新编译并与回执一致。历史 AIFast Feedback 与 Kimi Judge smoke/receipt 不改写，也不能转授权或冒充新角色证据。现有 Core v1 owner authorization 覆盖 Kimi Feedback；Gemini Judge 在新 Core v2 owner authorization、价格/ceiling 与 source/runtime/launch relock前必须 fail closed。五配置批量矩阵回执、approved audit sealer 与 Judge—human 校准完成前，C4 仍不关闭，也不授权报告正式 Judge 分数。

## 4. 明确不复现

v1.0 主项目不声称复现以下内容：

- 生产数据分布、生产流量频率、内部商品库、内部工具或真实业务约束；
- 论文的原始模型组合、内部 prompt、未公开规则路径或专有标注流程；
- 线上一周 A/B、业务转化指标或任何线上因果结论；
- 论文绝对分数、Table 2 Avg 的数值等价或跨数据集的直接优劣；
- 论文中的领域专家 `ManualSkill`，除非未来样本通过第 8 节的 provenance gate；
- T1–T3 漂移下的长期自进化结论；该部分属于后续平台扩展，v1.0 正式 benchmark 仅称为 T0；
- 未达到预注册样本资格的 capability、缺少真实多视图正例的 Exact Match，或没有正式 gold 的工具质量声明。

不复现项不得通过重命名掩盖。例如，LLM 生成后由用户做格式检查的 Skill 不能称为 ManualSkill；仅有单图商品或裁剪衍生图的数据不能称为真实多视图 Exact Match。

## 5. 新增受控扩展

以下内容是本项目为了提高可审计性或支持后续研究而新增，不应归因于原论文：

- asset/product/pHash/boundary/template/generator 联合 group-aware split 与传递闭包防泄漏；
- source-held-out、human-written `challenge_dev/challenge_final`；
- `SpecBaseline` 规范下界和 `OracleRoute` 路由诊断上界；
- feedback/final packet 隔离、Judge—human calib/audit、盲化 SBS；
- paired group bootstrap、置信区间、最差 capability、错误保留和统计 gate；
- 工具独立 gold、typed formal run、外部 trust root 与逐资产云上传许可；
- token、人工分钟、费用、p50/p95 延迟、失败率和 Bank 复杂度的 Pareto 报告；
- T0/T1/T2/T3 episode、open-set/new-skill、遗忘和 transfer 指标；后四项在 v1.0 后单独注册实验。

## 6. 五个主配置

五个主配置必须运行在同一冻结 Query 集、输入顺序、Assistant 骨干、解码策略、工具版本、tool budget 和可比 inference token budget 上。差异只能来自下表列出的 treatment：

| 配置 | 可用 Bank/路由 | 允许的创建或更新信息 | 禁止项 |
|---|---|---|---|
| `NoSkill` | 无 Skill 路由与注入；仍可使用与其他配置相同的工具 | 用户可见输入、同一 system/task 约束 | 不得用隐藏规则、额外 prompt 或额外工具预算补偿无 Skill。Routing F1 为 N/A。 |
| `LLMStaticSkill` | 一次性静态 Bank | 只读冻结 `AuthoringPacket`；与 S1 使用同一作者模型、固定模板及预注册 calls/tokens；用户只做相同 checklist 的有限审查 | 任何数据轨迹、标签、Assistant 响应、候选/既有演化 Bank、rubric/gold、gate/test、Judge 输出、分数或结果导向迭代。 |
| `S1` | Stage 1 Bank | 与 LLMStaticSkill 完全相同的 `AuthoringPacket`，只额外加入预注册训练轨迹 bundle | route/body gate、test/challenge_final、Judge 分数与后续 S2/S3 产物；不得独占 reference Skill。 |
| `S1+S2` | S1 经 Stage 2 后的 Bank | 只使用允许的 opt failure/correct-route 轨迹与 `route_gate`；只能执行第 2 节 A3 的变更 | 修改未合并 Skill 的 Body/Cs/Od，使用 body_gate、test 或 challenge_final。 |
| `Full` | S1+S2 经 Stage 3 后的 Bank | 只使用允许的 attributed responses 与 `body_gate`；只能执行第 2 节 A4 的变更 | 重新路由、修改 Description/Cs/Od，或使用 test/challenge_final。 |

`LLMStaticSkill` 与 S1 的作者模型调用上限、用户审查分钟上限，以及 execution surface 能实际强制的 token/费用硬上限，必须在看到候选输出前冻结。若 execution surface 无法强制或提供 provider-attested 的逐调用 token/费用上限（当前 Codex CLI 即属此类），则必须改为在候选输出前冻结相同的 token/费用比较目标、证据等级和 `enforcement=unavailable` 限制；这些目标只能披露，不能写成硬上限或 provider 回执，session 后若平台给出 usage 也只能按其实际证据等级记录。二者的公共输入逐字节相同；S1 唯一的实验性额外信息是预注册轨迹。若 Creator 需要 reference Skill，它必须作为内容固定的 `ReferenceSkillBundle` 同时进入两者的公共 AuthoringPacket；S1 独占 reference Skill 会使 RQ1b 失效。轨迹 token、额外模型成本、机器时长和人工分钟分项报告，不能用一个“总成本”掩盖处理差异。

## 7. 诊断配置及其边界

### 7.1 SpecBaseline

`SpecBaseline` 不调用作者 LLM，只从同一冻结 Task Specification 和 ToolSpec 经确定性编译器生成 Bank。其用途是区分：

- 仅把任务规范注入 Assistant 的增益；
- 一次性 LLM authoring 在无轨迹条件下的额外增益。

它是诊断下界，不进入五配置主表，不可改名为 ManualSkill，也不能用于替代 RQ1b 的 `LLMStaticSkill`。正式身份 `bank-spec-v0` 必须仅由冻结 taxonomy、Task Specification、ToolSpec、模板和 compiler 重建；相同公共输入必须得到逐字节相同的 Bank 与 hash。

### 7.2 OracleRoute

`OracleRoute` 从冻结 gold/acceptable capability 强制选择正确 capability，默认与冻结 S1 Bank 配对为 `OracleRoute+S1` sanity check。它估计路由错误被移除后的诊断上界，但会读取部署时不可见的标签，因此：

- 不能进入五配置主表或 deployment claim；
- 不能向 Creator、Stage 2、Stage 3 或 Assistant 的非 Oracle 运行回流信息；
- 不能用来选择 capability、修改 Bank、调 prompt 或修改 test；
- 必须在结果中单列为 diagnostic，并清楚标注使用了 gold route。

## 8. ExternalExpertManual：可选且 fail-closed

`ExternalExpertManual` 不属于 v1.0 必达项，不得阻塞主实验。只有同时满足下列条件，才可作为**预注册补充列**：

1. **资格：** 作者是独立于项目的合格电商/对应 capability 领域专家；保存可核验的资格依据、能力范围和经隐私处理的公开摘要。
2. **独立性：** 作者在产物冻结前未接触本项目的任何训练/评测轨迹、标签、Assistant 响应、候选 Bank、rubric/gold、Judge 输出、分数或论文复现结果；保存带日期的声明和实际可见输入清单。
3. **预注册：** 专家、capability、固定工时、报酬、允许资料、结构化表单、编译器版本和人工审查规则在主结果解封前登记。
4. **专业输入：** 专家填写 capability 范围、工具触发与顺序、回答步骤、输出约束、事实来源、拒答和 fallback；不要求专家理解 SKILL.md。
5. **确定性产物：** 版本化编译器把表单转为 Skill；保存表单、编译日志、人工分钟、所有 diff 和最终 Bank hash。项目用户只能做 schema、权限、引用和安全检查，不能代写领域内容。
6. **无生成式代写：** 若内容阶段使用生成式 AI，必须改标为 `ExpertCuratedLLMSkill`；若 AI 使用情况或来源无法证明，则排除该补充列。
7. **评价隔离：** 补充列必须在 test/challenge_final 解封前冻结。若主 test 已经可见，后加入的专家产物只能使用新的密封 holdout 或标为 post hoc，不能补写成预注册比较。

任何一项证据缺失都 fail closed：可以保留工程演示，但不能以 `ExternalExpertManual` 名称进入正式表格，也不能回答 RQ1c。

## 9. Authoring 信息边界

`AuthoringPacket` 只包含所有无轨迹作者都可公平获得的公共输入：

- 冻结 Task Specification 与 ToolSpec 的版本、内容和 hash；
- 许可允许且 revision/hash 固定的公开参考资料；模型可见内容必须是严格 UTF-8 可读摘录，并与同一份锁定字节、acquisition record 和 license evidence 双向校验，不把 base64 当作模型需要自行解码的正文；
- 固定输出 schema、字段顺序、模板和确定性 compiler 版本；
- 若 Creator 需要 reference Skill，则包含两种 authoring 共同可见、内容固定且单独哈希的 `ReferenceSkillBundle`；
- 作者模型的 provider/model/endpoint/revision、解码参数、call/token/cost 上限；
- 通用 schema、工具权限、引用和安全 checklist；
- 上述每项的 canonical hash。

Qwen v3 的默认 Flash 正式响应保留为不可改写的历史负结果：虽然身份、finish、usage、费用与隔离合格，但 payload 有 40 个 schema 错误，没有形成可审查 draft。项目所有者随后以前瞻方式批准 [`authoring-structured-submission-deviation-v1.json`](../specs/authoring/authoring-structured-submission-deviation-v1.json)，当时的公共输入、输出通道、额外预算与修复后的 sandbox runtime 由 [`authoring-freeze-lock-v4.json`](../specs/authoring/authoring-freeze-lock-v4.json) 共同绑定。Qwen v4 保持 taxonomy、Task Specification、ToolSpec、空公开资料 allowlist、空 Reference Skill、默认 `qwen3-vl-flash-2026-01-22`、固定 temperature 0、top-p 1、seed 20260722、non-thinking、32k/8k/40k token、US$0.05 和 30 分钟人工审查上限不变；旧输出已经存在，因此本 deviation 不提供 Plus 变体、自动 fallback 或事后人工换模。

DashScope JSON mode 只约束 JSON 对象，不提供 response-format JSON Schema。本次接口修订改用 provider 支持的强制单函数提交：模型只能恰好一次调用不可执行的 `submit_authoring_payload`，其 parameters 是精确展开的无哈希 `AuthoringDraftPayload` schema，assistant text 和 parallel tool calls 均禁止，并显式发送 `enable_thinking=false`。受信 Runner 不信任 provider framing，仍执行 strict JSON、Pydantic、规则覆盖和工具权限校验，再做 canonicalization、输入绑定与摘要。调用授权记录 1 次已知可审计正式响应和 1 次 provider outcome 未知的历史 incident，只允许最多 1 次新增 attempt；create-only claim 在容器启动前原子消耗授权，任何失败都不能删除 claim 重试。`deploy/authoring/locks-v4` 绑定 LLMStatic authoring sandbox `formal-v3` image、准确 CLI profile、network policy、deployment receipt 与四项 egress 探针。

唯一 Qwen v4 attempt 已执行并永久消费 claim。provider 完成一次正确命名的单 tool call，assistant text 为空，模型、usage、费用与隔离合格（18,839 input、1,762 output、20,601 total token、821 microUSD），但 `drafts` 被二次序列化为字符串，严格外层 schema 报 1 个类型错误。只读诊断即使解析该字符串，内部 6 个 draft 仍有 16 个校验错误；因此不能将 unwrap 视为足以保持 treatment 的机械修复。审计同时确认内部契约也有责任：函数 JSON Schema 和提示词没有表达 Runner 的词法排序要求，Task Specification 又要求按源顺序复制其中 6 个与 Runner 顺序冲突的数组；另外仍存在错误键名和错误 success-rule 映射。该归因只解释负结果，不改变其正式拒绝状态。receipt 明确记录 `formal_provider_call_eligible=true`、`draft_formal_eligible=false`、无 retry/fallback/repair。原授权不得重跑，也不调用 Plus。该 sandbox runtime 不等于工具 runtime；当前没有合格 draft，最终 Bank compile 更仍须等待 C3 authority-issued tool registry runtime。

AuthoringPacket 中公开的工具图还必须与 production runtime 同构。历史 TaskSpec v0 曾让 `product.multi_search` 先 `object_detect`、再把 detector crop 交给只接受 authoritative query asset 的 `image_product_search`；内部 `MultiProductSearchService` 又没有成为作者可见 authority，形成不可执行闭环。当前候选 TaskSpec v1/registry v2 已把唯一 operator 改为 authority-issued `multi_product_search@1.0.0` composite ToolSpec：模型只提交 authoritative `asset_id`，Runner 私下完成 detect→canonical crop→per-object retrieval，formal evaluator 不再接收第二套 `multi_product_chain` executor。该改动已经通过候选代码和负向 fixture，但不等于真实工具已签发；新的 authoring freeze 必须绑定 exact v1/v2，C3 还需真实 artifact/runtime 双锁、RPC gold、production 端到端 evidence 和 `authority_issued=true` receipt。

Task Specification 和公共资料中的每条事实性/安全规则必须标为 `public_source`、`tool_contract` 或 `project_choice`，并绑定可复核 source reference。LLM 可以协助抽取候选表述，但 LLM 输出本身不能成为规则正确性的证据；无法回到固定来源的内容只能明确降级为项目选择或删除。关键词污染扫描只是 defense-in-depth 词法哨兵，不能证明语义隔离；正式边界必须由来源 allowlist、独立外部摘要和隔离 authoring 进程共同提供。

Codex v4/v5 工作不改写上述 Qwen 事实，也不迁移旧批准。每个真实 run 都有独立 freeze、approval、guard、claim、output 和 receipt；v4 负结果与 v5 合格 pre-review draft 同时保留。可以写成“真实 Author 调用子阶段已完成并得到可复核 draft”，但在 owner 人工 checklist 与 C3 后 Bank compile 之前不得写成完整 C1 通过。

`AuthoringPacket` 明确禁止包含：

- pilot/opt/val/test/challenge/tool-gold/judge 样本、轨迹、标签、响应、ID、split 或统计摘要；
- 除共同冻结 `ReferenceSkillBundle` 外的任何 Bank/Skill 内容，以及全部 S1/S2/S3 候选或演化 lineage；
- benchmark rubric、gold、Judge prompt/output、gate 结果、错误分析或任何效果指标；
- 能间接恢复禁用字段的文件路径、缓存键、日志、检索索引或自然语言摘要。

S1 的 creator 输入必须另命名为 `S1CreatorPacket`，其组成只能是冻结 `AuthoringPacket` 加预注册 `TrajectoryBundle`。任何 reference Skill 若被使用，必须已经位于两种 authoring 共享的公共 packet 中；不得把轨迹或 S1 专属示例偷偷写回公共 AuthoringPacket。每次 authoring 保存可见文件 allowlist、字节摘要、完整 prompt、raw output、usage/cost、人工 diff 与最终 Bank hash；相同 raw draft 必须由 compiler 生成逐字节相同的 Bank。

2026-07-24 的数据来源修订永久移除 JDDC 2.0。MVP `TrajectoryBundle` 若来自
DuRecDial 2.0/CrossWOZ/MUGE 模式与 Codex 受控重写，必须把每个 batch 声明为
`data_origin=synthetic_derived`，并绑定 seed、plan、verified Asset/KB、生成器和
人工 review 证据。源对话只允许提供抽象交互模式，不能提供 capability、商品、图片
或事实 gold，也不能把派生轨迹描述为真实用户日志。该降级只保留机制复现资格，
不能支持 JDDC 2.0 真实流量规模或原始多模态用户分布的外部有效性结论。Full 的
对话来源待授权包和条款实际取得后另行冻结，不能由当前候选清单自动继承。

## 10. 可回答与不可回答的研究问题

| 结论 | 本项目在何时可以回答 | 明确不能推出 |
|---|---|---|
| S1 相对 NoSkill 的机制增益 | RQ1a 的同样本、同骨干/工具比较通过正式协议 | 论文生产环境的绝对增益或线上业务影响 |
| 训练轨迹相对一次性静态 authoring 的价值 | RQ1b 的 S1 vs LLMStaticSkill，公共输入和预算审计通过 | 自动方法超过人类专家；LLMStaticSkill 不是 ManualSkill |
| Stage 2 的路由贡献 | S1+S2 vs S1，capability macro-F1 与同组 CI；未见来源另报 | 所有未来分布上的理论单调性 |
| Stage 3 的 Body 贡献 | Full vs S1+S2 复用同一 route trace，冻结 rubric 与盲评通过 | 把路由变化误归因于 Body；论文绝对分数等价 |
| 四条归因路径的贡献 | 四个消融 Bank 在 final 解封前冻结并一次性评估 | 解封后按结果重建消融所得到的因果结论 |
| 专家人工基线 | 仅在第 8 节全部通过后回答可选 RQ1c | 当前用户手写、LLM 草稿或格式审查不能支持该结论 |
| 长期自进化能力 | 另行注册并完成 T1–T3 episode 后 | v1.0 T0 benchmark 不能证明持续适应、抗遗忘或 new-skill 能力 |

如果正式 profile 最终只能达到 mini，本项目只能声称“可审计工程纵切/系统原型”，不能声称完成实证机制复现。

## 11. 版本、内容哈希与冻结规则

### 11.1 内容哈希

本文的 `规范化内容 SHA-256` 按以下唯一算法计算：以 UTF-8 读取全文，将 CRLF 和单独 CR 都统一为 LF；在首个二级标题之前，按逻辑行严格匹配正则 ``^> \*\*规范化内容 SHA-256：\*\* `[0-9a-f]{64}`$``，全文匹配必须恰好一行，否则立即失败；删除该匹配行及其行尾 LF（若该行有行尾），其他字节和正文中对摘要字段的文字说明一律保留，不做 Unicode 归一化；最后以 UTF-8（无 BOM）编码并计算 SHA-256。冻结时，摘要同时写入本文和独立 protocol lock manifest；声明值、重算值或 lock manifest 任一不一致都 fail closed。该规则避免自引用，也避免宽松的“包含字段名”匹配误删正文。

### 11.2 一次修订

- `v0` 必须在候选 Skill、pilot Assistant 响应和 Judge 结果进入决策视野前完成；纯排版修正也必须留在 Git 历史中。
- mini 工程纵切结束后，允许**一次**批量修订为 `v1`。修订必须发布逐项 diff、过程性理由、影响分析、新 hash 和批准时间。
- 该次修订只能填写 v0 预先声明依赖 mini 实测的 profile、样本量/最小 n、cost/CI、工具门、Judge 校准门、prompt/rubric 和实现可行性参数，或修复已记录的规范歧义；不得因效果方向改变 RQ、五个主配置、主指标、authoring 禁区、group split 或 test/challenge 密封规则。
- 正式 benchmark 生成开始后，`v1` 冻结。任何语义变更都必须建立新的协议 ID、重新构建受影响的 benchmark/Bank，并使用未解封 holdout；不得在原实验上就地改写。

### 11.3 冻结门

将本文状态改为 `v1-frozen` 前必须同时具备：

- capability registry、Task Specification、ToolSpec、评价协议各自的版本和外部记录 hash；
- 已选 core/full profile、MVP capability、所有待填数值门槛、预算和人工工时上限；
- 数据、query、split、tool gold、模型、prompt、rubric、Judge 和环境 manifest；
- `LLMStaticSkill`/S1 公共 authoring 输入相等性与禁用字段测试；
- 已真实部署并有外部回执的可验证隔离 LLMStatic authoring runner，以及由 production Runner 自行记录的 Assistant provider/model/request/usage/latency/tool-trace provenance；
- README、canonical plan、本契约和实际代码基线对项目名称、五配置及限制的相同表述；
- P0 gate 全部有运行证据，而非只有计划文字。

当前本文是 `v0-draft`，因此只允许协议实现、mini 工程验证和不产生正式效果声明的 diagnostic run。C0 也只有在 README 等文件完成一致性检查后才能关闭；本文创建本身不代表 C0 已关闭。

## 12. 2026-07-20 工程收口状态

本轮已实现：verified AuthoringPacket、严格 UTF-8 可读摘录、确定性静态编译与 formal SpecBaseline；要求外部 digest/人工 review 的公开数据 gate；七工具/Multi-Product typed formal run、runtime authority、逐 case GoldReviewLedger、tool↔Assistant component 隔离与 evaluator；Phase4 query/split/rubric/audit-selection 共同绑定、五配置 diagnostic bundle、FeedbackPacket/FinalEvaluationPacket v2、不同 feedback/final cache 与 prompt namespace，以及 Kimi visual Feedback/Gemini final 的本地 fail-closed runner、严格结构化 parser、本地派生评分、错误保守处理和 create-only 单结果 receipt；另有支持从空目录同环境重建的 synthetic fixture。角色交换后的 provider 权限、价格和真实运行尚未完成。Base64 只存在于 preflight 后的瞬时视觉 wire。非 integration pytest 和 fixture Python 进程会阻断 socket/DNS，但这不是 OS 沙箱；Linux workflow 尚无外部 run receipt，且 checkout/setup/依赖安装不受该 guard 约束。第三方网关身份风险、fixture 与所有当前 Phase4 评价产物均强制 `formal_eligible=false`。

这不把本文升级为 `v1-frozen`。历史 Qwen LLMStatic container runner、自行采集执行 provenance 的 `ProductionAssistantRunner`、Gemini visual Feedback 与 Kimi final 单结果 runner/receipt 均保持原样；新的 Kimi Feedback/Gemini final 只完成代码与前向身份迁移，尚未获得完整调用资格或真实运行。Qwen v3/v4 与 Codex v4 负结果保持原样；Codex v5 已生成 strict-validated 六能力 pre-review draft、通过 canonical replay并完成 owner checklist，但 C3 后 Bank compile 尚缺。Portfolio 专用 verified loader 已能以外部摘要和稳定重读复验 8×25 accepted corpus、历史 plan/catalog v1、capability assignments、权限 overlay、图片字节及 processor scope；已有 create-only 清单不因角色交换自动升级，必须用 v6 role selection 与新权限链重锁。C3 authority-issued tool registry runtime 仍需真实 mini catalog、模型/index、隔离 gold 与人工审批后另行签发；此外仍缺四个 treatment Bank、Assistant runtime lock 与可见 card/evidence 投影、五配置 1,000 行 Assistant/Judge 矩阵回执、approved sealer/key receipt、50–100 条 Judge—human 校准和 200-query S1→S3 纵切。C1 与当前 core 决策因此仍为 **NO-GO**；详见 [`go-no-go/2026-07-20-core-no-go.md`](go-no-go/2026-07-20-core-no-go.md)。
