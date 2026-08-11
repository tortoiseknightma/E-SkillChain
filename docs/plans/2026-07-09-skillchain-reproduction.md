# SkillChain 公开数据机制级复现与受控扩展实施计划（修订版）

> **状态：** canonical implementation plan；历史 Qwen 与 Codex v4 负结果保持不可变。Codex v5 已通过独立 freeze/approval/claim 完成一个 `gpt-5.6-sol/high` 真实 session，生成 strict-validated 六能力 pre-review draft并通过 canonical replay。owner 人工 checklist、C3 后 Bank、真实 Assistant run 与 Linux CI 外部回执待补，C1/core **NO-GO**
> **首次制定：** 2026-07-09
> **本次修订：** 2026-07-30
> **论文：** Hu et al. (2026), *SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants*，本地 PDF 位于 `docs/`
> **剩余 P0/P1 收口计划：** [`2026-07-20-p0-p1-closure.md`](2026-07-20-p0-p1-closure.md)
> **当前 go/no-go：** [`../go-no-go/2026-07-20-core-no-go.md`](../go-no-go/2026-07-20-core-no-go.md)
> **当前数据来源决定：** [`../data-source-adjustment.md`](../data-source-adjustment.md)；机器可读口径为 [`../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`](../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json)

### 2026-07-24 模型角色修订

后续 mini 使用的目标角色改为：Assistant=`DashScope/qwen3-vl-flash-2026-01-22`；Author=`Codex CLI/gpt-5.6-sol/high`；final Offline Judge=`DashScope/kimi/kimi-k3/max`。feedback evaluator 暂保留不同于 Kimi 的 DeepSeek 家族；Phase 3 label synthesis 使用独立配置，不能随 Assistant 选型一起漂移。

Author 已建立独立的 `codex_mediated_static_author_v1` 证据口径和 `platform-mediated_non-provider-attested` 证据等级，不再套用 Qwen price/seed/endpoint/container receipt；当前事务实现代际为 v4。Codex/high v1 在 owner approval 前被审查取代；v2 获得 exact owner approval 后，Runner 在创建 claim/Popen 前因继承的完整 `PATH` 含会变化的 Codex Desktop `.codex/tmp/arg0` 条目而拒绝，因此没有 claim、output、receipt 或 inference，旧 approval 不可复用。v3 以绝对 CLI binary 和确定性环境修复 PATH，却在 preapproval P0 审计中暴露 claim post-commit 异常分类和 post-claim 持久终态缺口，故在 approval/claim/inference 前废弃。

Codex v4 的真实 session 因 prompt 未披露的 trusted-compiler 保留词冲突被拒绝并原样留证。v5 只前瞻新增对象检测预测的允许措辞，其他 treatment 不变；candidate/run=`authoring-codex-high-20260724-v5` / `llm-static-codex-primary-20260724-high-v5`，freeze file/payload SHA-256=`fcc3b86957fe74b725b837cabb8c6b782d9be713799c1f5380fa0fff7c6d4060` / `defd21b77b883f7db1a487de799f44f321482701086d7ece6325ab55c0fa976a`。v5 status=`codex_session_completed_draft_ready_for_review`、`formal_codex_session_eligible=true`；raw/pre-review/bundle SHA-256=`90fad5c27e6bb8a9122b470cfdef5a427bb4c0eb793d52ce4ffb52ed89b73a6b` / `e0c26c1c054514bfc4629fc4e84557da63f6b05335ca6c3b8709efe0a0e08908` / `41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3`，canonical bundle 已独立重放。

该协议可审计一个 Codex CLI session 和仓库侧零 retry/follow-up/repair/fallback；token 最多是 CLI platform-reported/non-provider-attested，served revision、provider request ID、平台内部 attempt/retry、费用与 OS 级 packet-only 输入隔离不能证明。self-hash 只证明内容完整性，不是创建者身份认证或 authority；同用户恶意重写、特权篡改、全盘故障、文件系统原子性失效和平台内部不可观测 retry 不在保证范围。候选冻结并由 owner 确认 exact 摘要和路径前，不得执行真实调用。

Kimi K3 的 final Judge 选择尚不能执行：官方直供接口的视觉输入要求公网 URL，不接受现有 Base64/data URL；必须先实现由 exact asset SHA、上传/托管/过期/删除回执和 treatment-blind 复用规则约束的 URL transport，再实现正式 Judge Runner。此前所有 NO-GO 门保持不变。

### Portfolio evaluator 前向替代（当前 v5）

上述 K3/DeepSeek 选择与共享 `kimi-k2.6` 选择只作为历史保留。当前 Portfolio Track 使用 AIFast `gemini-3.6-flash` 承担 visual Feedback，百炼/DashScope `kimi-k2.6` 承担 final Judge：Gemini 不发送 temperature/top-p/top-k，thinking 使用中转站默认且不可验证；final 使用 `enable_thinking=true,temperature=1.0,top_p=0.95`、无 seed。`model-role-selection-v5.json` 的 file/self SHA-256=`7fc6b0ba20ab542478ce5e3d5d53974ff619333a657c3b806eca88584ac62e8c` / `b0d65f8ffa5b6ce237cd9907ba9fe4c64129706596cf611b62829c396e8458f3`。

FeedbackPacket/FinalEvaluationPacket v2 只保存 verified image MIME+SHA；visual Feedback/final runner 分别消费 `aifast-gemini-feedback` / `dashscope-kimi-judge` remote-processing preflight，再瞬时生成 OpenAI-compatible `image_url` Base64 Data URL，Base64 不进入文本或持久化 evaluator artifact。`evaluator-isolation-v3` 强制不同 provider runtime、模型家族、packet、prompt、cache、输入与 artifact；第三方 AIFast 返回的模型名仍不能冒充 Google provider-attested 身份。Feedback 已严格解析视觉归因字段，final 只接受原始整数分数并本地计算 tier/`J_project`，解析/provider 错误不修复、不重试且保守归零；两类单结果 receipt 可 create-only 落盘重验。迁移后 AIFast 文本握手与一次 Base64 图片严格 JSON smoke 均成功，精确返回 `gemini-3.6-flash` 和 `finish_reason=stop`、零仓库重试；这不是 production evaluator run。五配置完整矩阵 receipt/manifest 与 Judge—human calibration 仍未完成，C4/core 继续 **NO-GO**。

## 0. 项目目标与定位

本项目的目标是在无法访问论文内部生产数据、生产工具和在线流量的条件下，使用公开数据、合规采集、LLM 合成与人工质检，完成 SkillChain 三阶段机制的可审计复现，并将其建设为可持续接入后续自进化研究的实验平台，最终作为 Agent 算法工程师求职作品集。

项目正式定位为：

> **SkillChain 的公开数据机制级复现与受控扩展。**

不得把本项目描述为论文生产数据、在线 A/B、原始模型组合或绝对数值的严格复现。负结果、无显著增益和与论文方向不一致的结果均是有效实验结果，不以“复现论文预期走势”为验收条件。

### 0.1 Reproduction Contract

| 层级 | 本项目承诺 |
|---|---|
| 论文机制对齐 | Skill 四元组；Stage 1/2/3 职责；Stage 2 只修改路由层、Stage 3 只修改 Body；论文轮数、样本下限和 Poor 阈值作为预注册主配置；五个累计系统配置；收敛曲线和 Stage 3 核心消融 |
| 机制替代 | 公开/自建数据替代生产流量；可访问模型替代论文模型；本地工具替代生产工具；个人人工 Gate 替代领域专家团队；明确记录每项差异 |
| 不复现 | 生产数据分布、淘宝内部商品/工具、线上一周 A/B、论文绝对分数和业务指标 |
| 新增扩展 | 分组防泄漏 benchmark、跨源 challenge、OracleRoute、Judge—human 校准、统计置信门、成本/延迟指标、T0/T1/T2/T3 漂移 episode、open-set/new-skill 评测 |

### 0.2 预注册研究问题

- **RQ1a — Creation vs NoSkill：** 在相同骨干与工具权限下，S1 是否优于 NoSkill？
- **RQ1b — Trajectory value：** 在相同 Task Specification、作者模型、工具权限和预注册 authoring budget 下，使用冻结轨迹创建的 S1 是否优于不读取任何轨迹/标签/评测输出的一次性 `LLMStaticSkill`？
- **RQ1c — Expert Manual（可选）：** 若后续能获得未接触项目轨迹与结果的独立合格领域专家，S1 是否优于 `ExternalExpertManual`？当前不把 RQ1c 纳入必达范围。
- **RQ2 — Routing：** Stage 2 是否提升 capability-level 路由质量，并在未见来源/措辞上保持增益？
- **RQ3 — Body：** 在复用同一 Stage 2 路由决策时，Stage 3 是否提升冻结 rubric 下的响应质量？
- **RQ4 — Attribution：** Rule path、LLM Judge、Qualitative Induction、Statistical Aggregation 各自贡献多少？
- **RQ5 — Evolution（M6 扩展，不属于 v1.0 主结论）：** 面对 query-style、source、category 和 new-capability shift，系统的适应速度、遗忘、成本和 Bank 增长如何？v1.0 主 benchmark 全部标为 T0；只有 Phase 12 完成 T1–T3 后才回答该问题。

每个 RQ 在正式实验前冻结主指标、统计方法、样本排除规则和失败条件。报告同时呈现总体均值、置信区间、最差 capability、成本、延迟和负结果。

---

## 1. 当前仓库状态与恢复原则

截至 2026-07-20：

- `master` 已 fast-forward 到 `f9ae37a`（`feat: complete auditable tool layer`），其中包含 `d851eb7` 的工具/语料集成、`8aca5b5` 的过期文档清理以及完整工具层补丁；它现在是后续实现的唯一基线。
- 严格 KB catalog/BM25 bundle、检测与 OCR 的锁模运行契约、文档安全审批 catalog、7 工具 ToolSpec/runtime registry、create-only retrieval-run、工具独立 diagnostic benchmark 基础设施与相应 CLI 均已进入 `master`。正式 artifact loader 必须匹配外部固定摘要；builder 自算的自哈希只用于生成待登记 identity，不能自证为 formal。所有正式入口默认拒绝 provisional 数据、未锁模型、未审计图片路径、可写 embedding cache 和裸 evaluator 输入。
- `codex/complete-tool-layer` 当前与 `master` 同指向 `f9ae37a`；它和 `codex/phase1-data-cleaning`、`codex/phase2-tool-layer`、`codex/phase2-phase3-integration` 一样只保留为历史开发分支，不再代表独立的当前可交付状态。
- 集成主线已经实现 schema v2、显式 v1→v2 migration、不可变 asset catalog、catalog-bound planning/batch/review/split、group-constrained exact split、完整 assignment 冻结、query/gallery eligibility gate，以及 eligibility-bound product index/search provenance。
- 实现基线 `f9ae37a` 的离线全仓回归为 **642 passed、16 skipped**；skip 来自显式外部 LLM 条件、大小写敏感性差异和当前 Windows 主机不允许创建 symlink 的攻击测试，不把 skip 计作相应平台路径已验证。
- 当前 `data/clean/` 与 `data/kb/` 无可用正式产物；历史 Phase 1 数字视为已验证运行记录，不视为当前可加载 artifact。
- 默认测试仍主要验证 fake/tmp 契约；真实 KB provenance、模型权重、OCR 隐私审批、正式 gold 和端到端实验必须另过 integration gate，不能因代码基础设施完成而宣称真实工具质量达标。

恢复原则：保留已有高质量下载、原子发布、哈希、缓存、审计和测试实现；先修订研究协议与 schema，再集成分支，不在旧标签/切分契约上继续扩大数据或调用费用。

---

## 2. 全局决定

| 维度 | 修订后决定 |
|---|---|
| 实验范围 | 五个主配置：NoSkill / LLMStaticSkill / S1 / S1+S2 / Full；另加 SpecBaseline 与 OracleRoute 诊断配置；恢复论文 Stage 3 四项消融；在线 A/B 明确不复现。ExternalExpertManual 仅在获得独立合格领域专家时作为补充配置 |
| 静态基线 | `LLMStaticSkill` 由与 S1 相同的作者模型在预注册 budget 内只读取冻结 AuthoringPacket（Task Specification、ToolSpec、允许的公开资料、固定模板，以及若使用则双方共享的 Reference Skill）一次性生成，再由确定性编译器转为 SKILL.md；不得读取 pilot/opt/route/body/test 轨迹、标签、响应、S1 产物或评测结果。用户只做 schema、工具权限、引用和安全 checklist 审批，不负责凭空发明领域规则。它明确替代当前无法诚实实施的论文 Expert ManualSkill，报告中不得简称为 ManualSkill。另保留不调用 LLM、直接编译 Task Specification 的 `SpecBaseline` 作为诊断下界 |
| 语言 | 数据、Skill、响应与主报告使用中文；README 提供中文主文和英文摘要/完整英文版 |
| Skill 粒度 | 五个顶层 intent 仅用于汇总；路由 GT 使用 Bank-independent 的 `canonical_capability`。MVP 先覆盖 5–7 个 capability，正式 Bank 目标 8–12 个，前提是每个 capability 有足够样本和工具支持 |
| 数据规模 | 分为 mini / core / full 三个预注册 profile；4,500 条是 full 目标而非在纵切完成前必须投入的门槛。正式结果必须声明 profile 与样本量，不能把 core 结果包装成 full |
| 数据切分 | 先按 asset/product/pHash/boundary/template/generator group 切分，再满足意图与边界分层；同资产及近重复不得跨 split |
| 路由 GT | `canonical_intent + canonical_capability + acceptable_capabilities`；val/test 100% 人工确认；opt_pool 可弱监督但保留全部 label provenance |
| 边界样本 | 15–20% 仅作为目标区间；必须经构造规则或人工确认是真实边界。跨意图同图组覆盖全量 corpus，不能只存在于 dev_mini；随机普通图不得直接标为 natural ambiguity |
| 商品查询 | Exact Match 的正例必须是同一 product 的另一真实视图；query 与 gallery 禁止同 asset/近重复。product group 不得跨数据 split。Style Search 则排除 query 的同 product，避免把精确匹配当作风格检索 |
| 模型 | provider/model/endpoint/revision/prompt/decoding 参数进入 RunManifest。反馈模型与最终评价器尽量分离；模型替换属于实验变量，不写死到核心 schema |
| 工具 | 检索和确定性执行在本地；embedding backend 可插拔。远程 embedding 仅在许可允许且 manifest 明示时使用，不能再声称“运行时零外部 API” |
| Utility 范围 | MVP 保留 recipe 与 document；document 必须有 OCR。健康场景在无可靠知识、安全 rubric 和适格人工审核前不进入主实验 |
| Human Gate | AI 可生成预审报告，但最终批准由用户执行并记录 checklist、耗时、reject 原因；不把单人 Gate 描述为论文的领域专家团队 |
| 统计 | 主结果使用 paired bootstrap 95% CI；路由另报配对检验/混淆矩阵；单调门是“验证集经验选择约束”，不是理论泛化保证 |
| 预算 | 先用 dev_mini 实测每阶段调用量、tokens、图片数、缓存命中、时长和费用，再冻结全量预算。未经 cost gate 不启动 4,500 条全流程 |
| 可发布性 | 研究集、云上传许可和公开 Demo 许可分开审计；公开 Demo 使用独立的 permissive subset，不展示 MEP-3M/ISIA 等限制数据 |

正式 profile 在看到主结果前，依据 dev_mini 的标签耗时、调用成本和 paired-bootstrap 精度模拟选择并写入协议：

| Profile | disjoint 主数据划分 | 用途与最低声明 |
|---|---|---|
| mini | dev_mini 200 | 仅用于离线纵切、接口验收和成本测量，不发布总体效果结论 |
| core | dev_mini 200 / opt_pool 800 / val 200 / test_frozen 300，共 1,500 | 个人可交付的主目标；至少覆盖 5 个 capability，并明确标为小规模机制复现 |
| full | dev_mini 200 / opt_pool 2,800 / val 500 / test_frozen 1,000，共 4,500 | 扩展目标；仅在 core 纵切、人工工时与费用 gate 通过后执行 |

challenge set 独立于上述划分，core 为 100–150 条、full 为 100–300 条。若 pilot 显示 test 的 95% CI 精度或最小分组样本不足，应在冻结 test 前调整 profile/样本量；不得看过 test 结果后补样。若资源只允许 mini，交付物应称为系统原型，不声称完成实证复现。

### 2.1 数据源使用约束

source portfolio 冻结为“按监督信号选源”，而不是“按来源分配意图”。正式计划必须消费 [`ecommerce-mvp-source-portfolio-v1.json`](../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json)，不得因某个压缩包已经下载就把它提升为主 gold：

| Capability | MVP 数量 | 主监督来源 | 补充/挑战来源 |
|---|---:|---|---|
| Exact Match | 35 | ABO 的 product/listing identity 与真实多视图 | Products-10K 留到 core；MUGE/MEP-3M 只供 hard-negative 候选 |
| Multi-Product | 35 | RPC 的场景、SKU reference、bbox 与逐商品 identity | COCO/Open Images 只作 open-world detector challenge；SKU-110K 只作高密度检测 |
| Style Recommendation | 35 | FashionIQ 的参考图 + 相对自然语言修改 | Polyvore 留到 core；DeepFashion 只作候选 gallery/同款排除；ABO 扩展非服饰风格 |
| Visual Encyclopedia | 35 | Wikimedia 图片 + zhwiki/Wikidata 固定 revision 证据 | iNaturalist 作细粒度/拒答 challenge；COCO/Open Images 作常见物体 challenge |
| Document Reading | 30 | WildReceipt 中文收据 | CORD/SROIE 补字段 gold；Wikimedia Documents 只作 wild challenge |
| Recipe Guidance | 30 | ISIA Food-500 菜品识别 + 下厨房 recipe KB | Recipe1M+ 配对子集留到实体映射完成后的 core 扩展 |

独立语言层不再依赖 JDDC 2.0；该数据集已于 2026-07-24 永久退出所有 profile。MVP 改用 DuRecDial 2.0 的偏好追问/推荐模式、CrossWOZ 的约束冲突/无结果/放宽条件/目标修改结构与 MUGE 的短查询措辞。它们只改变表达和交互形态，不能提供 capability、asset、事实或 split gold。正确组合是 `TaskSpec 语义 + 真实资产/KB 事实 + DuRecDial/CrossWOZ/MUGE 模式 + Codex 受控领域重写 + synthetic_derived 标记 + 人工审核`。这只能支持机制级复现，不能声称复现 JDDC 2.0 的真实流量规模或原始多模态用户分布。Full 对话实现待 U-NEED 等授权包和条款取得后另行冻结。

所有来源继续遵守以下共同约束：

- Exact Match 正例必须是同一 product 的另一真实、非 near-duplicate asset；不得用裁剪、加噪、换背景或单图来源伪造。
- Style Recommendation 必须排除同 product；类目相同或关键词相近不构成偏好真值。
- Recipe Guidance 必须将菜品识别与食谱证据拆开，并通过经审核的 dish alias/entity mapping 连接。
- Visual Encyclopedia 只引用与图片实体绑定的固定 revision 证据，不把 LLM 合成事实当 gold。
- 同一资产/实体跨意图复用时，全部 query 进入同一联合 leakage component 和 split。
- 自建爬虫仅访问无需登录的公开页面，遵守站点条款、robots、速率限制和删除请求；禁止绕过鉴权/反爬、采集个人敏感信息或把“网页可访问”等同于“可训练/可再分发”。每个采集器需保存 acquisition decision、抓取时间、URL、响应许可信息和删除/重建路径。

---

## 3. 核心数据与实验契约

`schemas.py` 不再采用“以后只增不改名”的永久承诺，改为 `schema_version + migration`。核心 schema 只存稳定语义；provider、数据源和 capability 通过 registry/version 管理。

```python
class DatasetAsset(BaseModel):
    schema_version: int
    asset_id: str
    source_dataset: str
    source_revision: str
    source_record_id: str
    transform_policy_version: str
    local_path: str
    sha256: str
    phash: str | None
    near_duplicate_cluster_id: str | None
    product_id: str | None
    derivation_parent_asset_ids: list[str]
    license_id: str
    source_url: str | None
    attribution: str | None
    cloud_upload_allowed: bool | None
    public_demo_allowed: bool

class Query(BaseModel):
    schema_version: int
    taxonomy_version: str
    task_spec_version: str
    query_id: str
    asset_id: str
    leakage_group_id: str
    boundary_group_id: str | None
    template_family: str
    generator_batch_id: str
    turns: list[ConversationTurn]
    canonical_intent: str
    canonical_capability: str
    acceptable_capabilities: list[str]
    is_boundary: bool
    boundary_strategy: str | None
    requires_card: bool
    episode: str = "t0"           # v1.0 为 t0；Phase 12 扩展 t1/t2/t3
    split: str
    label_provenance: list[LabelDecision]

class BankCapabilityMap(BaseModel):
    bank_hash: str
    capability_to_skill: dict[str, list[str]]
    aliases: dict[str, str]

class AssistantResult(BaseModel):
    run_id: str
    query_id: str
    bank_hash: str | None
    routed_skill: str | None
    routing_raw: str | None
    tool_calls: list[ToolCall]
    response_text: str
    cards: list[Card]
    usage: Usage
    latency_ms: int

class FinalEvaluationPacket(BaseModel):
    evaluation_id: str
    blinded_query: EvaluationQueryView
    task_rubric_version: str
    response_text: str
    cards: list[Card]
    tool_evidence: list[Evidence]

class FeedbackPacket(BaseModel):
    query_id: str
    turns: list[ConversationTurn]
    image: EvaluationImage
    canonical_capability: str
    acceptable_capabilities: list[str]
    response_text: str
    cards: list[Card]
    tool_evidence: list[Evidence]
    tool_trace: list[AssistantToolTrace]
    rubric: RubricSnapshot

class JudgeScores(BaseModel):
    tcr: int                       # 0..10
    ccc: int | None                # requires_card=False 时 N/A
    cq: int                        # 0..20
    ca: int                        # 0..10

class RunManifest(BaseModel):
    run_id: str
    git_commit: str
    schema_version: int
    dataset_hash: str
    split_hashes: dict[str, str]
    query_manifest_hash: str
    taxonomy_hash: str
    index_hashes: dict[str, str]
    tool_manifest_hash: str
    bank_hash: str | None
    rubric_hash: str
    prompt_hashes: dict[str, str]
    model_configs: dict[str, ModelConfig]
    environment_lock_hash: str
    working_tree_state: str       # clean / dirty
    patch_hash: str | None
    seeds: dict[str, int]
    repetition_id: str
    parent_run_id: str | None
```

`BankCapabilityMap` 必须通过不变量检查：每个可路由 skill slug 只映射到一个 canonical capability；alias 不得形成环；同 capability 可有多个 Skill 时，必须保留 skill-level 辅助指标。`FinalEvaluationPacket` 在送入最终 Judge 前移除 system/config/bank/skill slug、Description、Body、`canonical_*` 标签、source 和 split 元数据；只保留用户实际可见输入、响应/cards、完成任务所需的工具证据与独立冻结 rubric。只有 `FeedbackPacket` 可以包含当前 Skill 内容，二者不得共用 prompt 或缓存命名空间。

### 3.1 路由评价口径

- 主指标：`canonical_capability` macro-F1。
- 辅助指标：顶层 intent macro/micro/weighted F1、per-capability P/R/F1、混淆矩阵。
- 边界样本同时报告 strict hit 与 acceptable-set hit。
- T3/OOD 报告 no-skill precision/recall、coverage-risk 和错误接收率。
- NoSkill 没有路由结果，Routing F1 为 N/A。

正式 `val` 在首次使用前按 group 固定拆成互斥子集：

| Profile | route_gate | body_gate | shadow_val |
|---|---:|---:|---:|
| core（val=200） | 80 | 80 | 40 |
| full（val=500） | 200 | 200 | 100 |

`route_gate` 只决定 Stage 2 接受/回滚，`body_gate` 只决定 Stage 3 接受/回滚；`shadow_val` 可生成监控报告，但不得用于选候选、改 prompt、调阈值或提前停止。challenge 也预先拆成可反复诊断的 `challenge_dev` 与像 test 一样密封的 `challenge_final`，两者 hash 分开保存。

capability 是否进入正式主表在扩量前冻结：core 中每个 capability 至少有 opt/val/test = 60/30/30 条，full 中至少为 100/40/50 条，并尽量覆盖两个 source/template family；route_gate/body_gate 按 capability 分层，core 每个 gate 目标 ≥10 条、full 目标 ≥15 条。分组推断门的最小 n 由 mini 的功效/精度模拟预注册；未达到时该 capability 只作描述性报告，不能用不稳定的“最差组非回退”决定接受。Stage 2 的“≥30 个失败”和 Stage 3 的“≥50 个 attributed responses”是更新资格，不是通过复制/合成失败样本强行满足的数据配额。

### 3.2 响应质量口径

- `requires_card=True` 且未出卡时，CCC=0，而不是 N/A。
- `requires_card=False` 时 CCC=N/A，单样本项目指标定义为：

```text
J_project = 100 × (TCR + CQ + CA + applicable_CCC)
                  / (40 + 10 × requires_card)
```

- 同时独立报告四个维度均值及各自样本数，不声称 `J_project` 等同论文未充分披露的 Table 2 Avg。
- 冻结 `benchmark_rubric` 用于跨 Bank 比较；从当前 Body 派生的规则只作为诊断信号。
- Judge parse/error 不从分母静默删除；重试后仍失败需单独报告并按预注册保守策略处理。

### 3.3 Artifact 与可审计性

大文件继续忽略，但以下内容必须进入 Git：

```text
artifacts/manifests/dataset-*.json
artifacts/manifests/query-*.json
artifacts/manifests/index-*.json
artifacts/manifests/tool-*.json
artifacts/manifests/bank-*.json
artifacts/manifests/run-*.json
results/summary/*.json
results/samples/*.jsonl
tests/fixtures/mini/
```

Git tag 只标记里程碑；Bank 身份由 canonical manifest 的内容哈希决定。`active.json` 等指针不得保存不可迁移的绝对 worktree 路径。

---

## 4. 目标文件结构

```text
ECommerceSkillChain/
├── docs/
│   ├── reproduction-contract.md
│   ├── evaluation-protocol.md
│   ├── data-card.md
│   ├── model-and-prompt-card.md
│   └── plans/2026-07-09-skillchain-reproduction.md
├── artifacts/manifests/             # 进入 Git 的内容寻址 manifest
├── results/{summary,samples}/        # 进入 Git 的聚合结果和小型审计样本
├── tests/fixtures/mini/              # Python socket-guarded synthetic 纵切
├── data/{raw,clean,index,queries,kb} # 大文件，gitignore
├── runs/local/                       # 原始运行产物，gitignore
├── skills_bank/
│   ├── _specs/
│   └── <skill_slug>/SKILL.md
├── src/skillchain/
│   ├── config.py
│   ├── schemas.py
│   ├── migrations/
│   ├── llm/                          # structured response / tools / batch / cache
│   ├── benchmark/                    # taxonomy / grouping / split / episodes
│   ├── data/
│   ├── tools/
│   ├── synthesis/
│   ├── assistant/
│   ├── evaluation/
│   ├── stages/
│   └── experiments/                  # strategy registry + experiment runner
├── scripts/
├── report/
└── demo/
```

---

## 5. 实施阶段

## Phase 0：仓库脚手架（历史完成，需补契约）

已完成 Python/uv、基础 schema/config/LLM wrapper 和 API 冒烟。历史测试记录保留，但修订后的补充任务为：

- [x] 建立 Python 3.12 + uv 项目、基础 pydantic schema 和模型调用封装。
- [x] 建立 Phase 0 冒烟测试与 usage 日志。
- [x] 将真实 API 测试标记为 `integration`；非 integration pytest 由 autouse Python socket guard 阻断 DNS/connect，不调用外部 API、不产生费用。该 guard 不是 OS 沙箱，也不约束任意外部子进程。
- [x] 将 LLM 返回值升级为结构化 `LLMResponse`，包含文本、tool calls、usage、finish reason、request/model identity 和 latency。
- [x] 统一当前 provider/model 配置，禁止出现 `provider=deepseek`、实际 `model=glm-*` 一类身份错配。
- [x] 更新 `.env.example`，只保留实际使用的变量和正确端点说明。

**验收：** 默认测试的 Python socket 路径被阻断；integration test 明确 opt-in；manifest 中 provider/model 与实际返回一致。CI 的 checkout/setup/依赖安装可联网，不能称为整项 workflow OS 级零网络。

## Phase 1：数据获取与清洗（旧 adapter 保留，MVP 来源待重建）

2026-07-22 的历史盘点发现 `raw/` 只有 21,504 个文件、8,147,316,535 字节，且 `clean/` 与 `kb/` 为空。2026-07-24 的统一下载器运行已完成并验证 MVP profile 的全部 34 个活动项；31 个本地字节项合计 82,247,477,174 字节，另有 3 个 command 管理项。ABO compact、WildReceipt、固定提交的 DuRecDial 2.0/CrossWOZ、SROIE、CORD v2 和 RPC Kaggle v5 均已取得并验证。FashionIQ 固定清单收口为 75,267 张有效图片与 2,416 条显式排除；JDDC 2.0 已永久退出。各来源的明确许可条款、PII/use review 与外部 source lock 仍需逐源完成。`data/raw` 总量包含既有 cache/recovery，不等于 formal dataset，selection/review/catalog manifest、许可/PII、source lock 与 Mock 人审仍未完成。详见 [`../data-source-adjustment.md`](../data-source-adjustment.md)、[`../data-download-runbook.md`](../data-download-runbook.md) 与 [`../mock-trajectory-runbook.md`](../mock-trajectory-runbook.md)。

旧 MUGE、MEP-3M、COCO、iNaturalist、Wikimedia、ISIA、zhwiki 和 recipe adapter 及安全/原子发布测试继续保留，但其历史运行数字和本地 raw 文件都不代表调整后角色已经合格：MUGE 降为语言/hard-negative 来源，MEP-3M 移出 MVP，COCO 与 Wikimedia Documents 降为 challenge，ISIA 只负责菜品识别。

- [x] 完成现有公开源的历史下载、格式探查、清洗、损坏/尺寸过滤和基础源内 pHash 去重；这些历史运行和 pHash 只作开发参考，不能替代最终落盘字节上的正式跨源 catalog。
- [x] 记录 2026-07-22 本地缓存盘点，并冻结机器可读 [`ecommerce-mvp-source-portfolio-v1.json`](../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json)；该完成项只证明“来源角色决定已记录”，不证明数据 ready。
- [x] 用 [`raw-download-profiles-v1.json`](../../specs/data_sources/raw-download-profiles-v1.json) 与本地 overrides 完成 MVP RAW 的全部 34 个活动项；下载完成不替代正式 source lock、许可/PII、selection 或 catalog。
- [x] MEP-3M 历史 adapter 固定 revision、文件大小和 SHA-256；其 tier 已改为 core，不再是 MVP 重建依赖。
- [ ] 为已取得的 ABO/WildReceipt/FashionIQ/DuRecDial 2.0/CrossWOZ 和仍门控的 RPC 补齐严格 adapter、固定 revision、外部 expected digest、acquisition receipt、selection/disposition 和人工 review；DuRecDial/CrossWOZ/MUGE 只能进入语言/交互模式层，Mock 输出必须是 `synthetic_derived`。JDDC 2.0 已永久退出。
- [ ] 为 MUGE、Wikimedia/zhwiki、iNaturalist、ISIA、下厨房 recipe、CORD/SROIE 及 challenge 来源补齐相应 revision、摘要与 source-specific gold 资格；不得用 challenge/语言来源替代主监督信号。
- [x] 实现不可变 DatasetAsset catalog：从同一次最终字节快照重算 SHA-256 与统一 64-bit pHash；按内容、dataset-scoped source record/product、近重复传递闭包和 derivation lineage 生成 typed leakage component；catalog artifact/manifest/self hash、非空 coverage root、Unicode/casefold 路径冲突、未登记/不支持图片和真实文件均可重验。批次 stage/accept 只复核本批引用资产，避免每批重复解码整个 catalog。
- [ ] 让所有 Phase 1 adapter 输出严格 `DatasetAssetDraft`，补齐各源 revision/license/transform/lineage，并基于重建后的真实 mini/core 数据发布首个正式 catalog；当前只有通用 builder/verify CLI 和 fixture，不能把基础设施完成误写成数据 artifact 已完成。
- [ ] 为 KB 补 source URI、revision、license、citation span；构建 Wikimedia image↔entity↔fixed-revision evidence 与 dish↔reviewed alias/entity↔recipe evidence 两条显式映射，并验证 coverage。
- [ ] 生成“下载/变换/云上传/embedding/再分发/公开 Demo”五维许可矩阵和 PII 审计。
- [ ] 一套命令通过参数重建 mini/core/full profile，并输出 tracked `dataset-*.json` manifest；三种 profile 共用下载、变换和校验实现。

当前通用 catalog 工具入口为 `uv run python scripts/build_asset_catalog.py build ...`、`verify ...` 和 `audit-gallery ...`。`build` 强制提供至少一个 coverage root；`audit-gallery` 直接读取 canonical schema-v2 Query 与带路径的 gallery 引用，成功时 create-only 发布绑定 catalog/query/gallery 哈希的 eligibility manifest，失败时返回非零状态。它们是 adapter 收口与 CI gate，不替代尚未完成的真实数据重建。

**验收：** 从固定来源能重建按监督层分离的 mini；200-query MVP 严格按 35/35/35/35/30/30 分配，约 40 条 boundary、至少 40 条人工直接编写或重写；主源、语言层和 challenge 轨分别有独立 manifest。所选正式 profile 的文件数、哈希、许可、entity/product mapping 和 near-duplicate 统计完整；不以本地目录、历史数字或 adapter 存在替代当前 artifact 验证。

## Phase 2：研究协议冻结与分支集成（新增，当前最高优先级）

- [x] 创建 `docs/reproduction-contract.md`，从本计划提取并冻结复现/替代/不复现范围。当前文件为 `v0-draft`；只有其自身冻结门满足后才升级为 formal 权威版本。
- [x] 创建 `docs/evaluation-protocol.md`，定义 RQ、标签本体、profile、主指标、统计方法、排除规则、val 子集用途、challenge 可见性和 test 使用次数；当前文件为 `v0-draft`，mini 纵切后允许一次有版本记录的修订，正式 benchmark 生成前冻结。
- [x] 定义 5 个 intent 与每个 intent 10 个候选 capability 的覆盖矩阵；MVP 选择 6 个有 TaskSpec 和工具设计支撑的 capability。真实数据支撑仍受 source portfolio 与 C2 gate 阻断，不能再把旧来源下载状态作为完成依据。
- [x] 为 6 个 MVP capability 编写 Task Specification v0：成功/失败行为、工具权限、输出格式、安全约束、card eligibility、fallback 和独立规则来源；它是 LLMStaticSkill/S1 的共同输入，不从候选 Skill Body 反向生成。事实与规则均回到固定来源或明确标为项目选择，不能让 LLM 输出自证正确。
- [x] 实现 schema v2 与显式 v1→v2 migration：v2 拒绝 legacy 字段和缺失版本，`gt_skill` 只允许作为 migration provenance；新计划使用版本化的 Bank-independent coarse capability taxonomy。正式 8–12 capability taxonomy 与 Task Specification 仍由本 Phase 的独立任务冻结。
- [x] 将 Phase 2 与 Phase 3 已提交代码集成到 `codex/phase2-phase3-integration`；保留各自测试，并完成 schema/group split 与防泄漏升级。
- [x] 修复 Phase 3 的 schema/group split 链路：planning → staged → accepted → labeled 全程保留 `leakage_group_id/boundary_group_id/boundary_strategy/template_family/generator_batch_id`；按四类 namespaced group 的传递闭包建立原子组，以精确 MILP 分配整个组，冻结完整 assignment 并审计零跨 split group。template family 已改为独立于 batch ID、跨两个隔离生成批次的真实 prompt family block，不再是被 generator batch 完全覆盖的空约束。
- [x] 将 asset catalog 接入 planner、active pointer、batch manifest、label/review/split loader 和 frozen split manifest；正式 plan/review/split 默认必须提供匹配 catalog，逐条复核 `asset_id + image_path + leakage_group_id`，同一 catalog component 只能进入一个 generator batch。path identity 仅保留为显式 provisional 兼容路径。
- [x] 闭合生成与 accepted provenance：BatchDraftManifest v2 绑定 plan/catalog/policy/seed、完整 25 条 generation input 和 draft bytes；split 前逐批验证 canonical ledger/manifest/results，并证明派生 `queries.jsonl` 可按 ledger 顺序逐字节重建。review/仲裁在实际送审前复核图片字节，labels manifest 绑定 full plan/catalog；审核、pending、仲裁与 split 入口都从 accepted source + reviews 确定性重建分歧项、review-error 项和 5% spot-check 队列，协调改写 queue、计数与哈希也不能删除强制抽检项。
- [x] 正式 full/core freeze 必须加载真实 full plan；full assignment 还必须是已审核 labels 的纯 split 派生。冻结与重验均核对 plan 静态字段、完整 assignment、catalog 引用和联合 group 零泄漏，不再接受调用方自报 plan hash。
- [x] 实现 query/gallery eligibility gate 与 CLI：对 exact content、近重复 cluster、同 source record 与 derivation 的混合关系做统一传递闭包；Exact Match 必须存在同 product 的另一非近重复真实视图；Divergent Recommendation 对 query 或 gallery 缺 product identity 均 fail closed，并排除同 product gallery 项。CLI 绑定实际 Query/gallery artifact hash 并发布不可变 eligibility manifest。
- [x] 将 eligibility 接入商品 index 与搜索结果：正式 index build 必须用实际 Query/gallery/verified catalog 重跑 gate，将完整 `products.parquet` 与 gallery asset 做 path/source/product identity 双射，在 embedding 前后复核输入 artifact 与引用图片字节，并把 eligibility/gallery 作为 index bundle artifact；manifest v2 默认拒绝未审计 index，`--allow-incomplete` 与仅限 `--limit` smoke 的 provisional 开关正交。每条商品搜索结果携带统一的 index/eligibility/query/gallery/catalog/source hash binding。
- [x] 实现 create-only product retrieval-run：从权威 schema-v2 Query 与 verified AssetCatalog 推导完整 query ID/顺序和实际图片，正式路径使用无 SQLite cache 的 `FormalEmbeddingBackend`、已校验图片 bytes 和 live `ProductSearchService` 派生 runtime identity；每条 call 保存输入 identity、实际 query-vector SHA、完整 `ProductSearchTrace`、ToolSpec/runtime/index binding 与固定错误码。任意 callable 只能生成 `ProvisionalRetrievalRun`，不能通过 evaluator gate。
- [x] 已实现由外部摘要共同锁定 Query JSONL、FrozenSplitManifest、rubric 与 judge-audit selection 的 `VerifiedPhase4Inputs`，五配置 `AssistantMatrixPlan` / `AssistantRunBundle`、create-only manifest、深度重载，以及只从 verified input/plan/run 组合 final packet 的隔离契约。新增 exact `ProductionAssistantRunner`：模型只返回严格 semantic action，Runner 自行调用统一模型入口和正式 registry，逐行保存 provider request、usage、Runner latency、实际 tool trace 与自哈希 receipt；任意 backend 仍只能走 `backend-reported-v1` diagnostic 入口。所有产物继续固定 `formal_eligible=false`，尚需真实五配置 Assistant/Judge 和人工校准。
- [ ] 修复 full corpus 规划：跨意图 triplet 覆盖全量；natural ambiguity 必须来自显式候选筛选并经人工/模型验证，不能随机标注。
- [ ] 移除 artifact 指针中的绝对 worktree 路径，改为仓库相对路径 + content hash。
- [x] 建立可从空目录确定性重建的 synthetic diagnostic fixture 与 Linux CI workflow；fixture Python 进程阻断 socket/DNS，当前只有 6 条 query，所有评价产物均不具备 formal 身份。workflow 尚无外部 run receipt，checkout/setup/依赖安装需要网络，不能把它描述成 OS 级完全离线或真实 200 条 mini。
- [ ] 用 50 条样本做标注计时试验，冻结单人总工时上限、10–20% 洗脱期盲重标比例和第二标注者（如有）预算；LLM 预审不得计作独立人评。

**阻断门：** group split、stable capability labels、tracked manifest 和集成主线未完成前，不启动全量合成、Stage 1 正式 Bank 或任何最终分数实验。

## Phase 3：工具层（代码基础设施已补齐，真实 artifact 与质量验收未完成）

### 3.1 Embedding 与商品索引

- [x] Phase 2 分支已实现 `qwen3-vl-embedding` 客户端、SQLite cache、canary 漂移检查、商品 image/text 索引和三个商品查询接口。
- [x] 正式 retrieval 与日常缓存路径已经分离：正式路径强制 freshly verified canary、无缓存查询、immutable image bytes、实际 query-vector hash 和由 live index/backend 派生的 runtime binding；SQLite cache 只用于非正式构建/开发，合法形状的投毒向量不能进入 formal run。
- [ ] 把 embedding backend 抽象为 local/remote 可切换；正式 run 固定 backend、模型版本、维度、canary 和源数据 hash。
- [ ] 许可不允许云上传的数据不得进入远程 embedding；公开可复现轨至少提供 local backend 或完整预计算许可说明。
- [ ] Exact Match 以 ABO 为 MVP product-identity 主干，保留同 product 的非近重复真实视图作为正例，仅排除同 asset/近重复；product/cluster 不跨 query split，返回中记录 inclusion/exclusion 证据。eligibility 已被 index、结果 binding 与 create-only retrieval-run 强制消费，但最终 eval manifest 尚未实现；ABO compact RAW 已在本地验证，仍须从 spins/listings 实测并冻结合格多视图覆盖。在资格裁决与最终 eval 接线完成前不得发布正式 Exact Match 分数；Products-10K 只在 core 扩展。
- [x] Multi-Product 链路已实现确定性 `detect → floor/ceil crop → per-object trace search`：不持久化 crop，本级记录 detector/input、bbox、crop bytes/hash/尺寸、查询向量 hash、retrieval binding 和 hits，并在每阶段前后复核父图字节。该完成项只证明链路机制，不证明 COCO fixture 具备 SKU gold。
- [x] 离线关闭 Multi-Product authoring/production 工具图缺口：TaskSpec v1/registry v2 将唯一 operator 改为 `multi_product_search@1.0.0` composite ToolSpec，只接收 authoritative `asset_id` 并由 Runner 内部持有 detect→private crop→retrieval；formal evaluator 不再接受第二套 executor，任意 crop/path/bbox 和 generation 混配测试 fail closed。真实 C3 artifact/runtime 双锁、RPC gold、production 端到端 evidence 和 `authority_issued=true` receipt 仍待完成，故 P0-07/C3 不因本项打勾而关闭。
- [ ] 以 RPC 的单品 reference、结账场景、bbox 和 SKU identity 构建正式 Multi-Product gold，报告 detect/crop/per-object retrieval/SKU 端到端成功率；COCO/Open Images 只保留为 detector challenge，不进入 SKU 主指标。
- [ ] 将现有依赖 MEP-3M 类目 anchor 的 style 逻辑降为历史 diagnostic，正式 MVP 改为消费 FashionIQ 的参考图 + 相对自然语言修改；Style Search 排除同 product。Polyvore 留到 core，DeepFashion 只作候选 gallery/同款排除，MUGE 只供中文措辞。

### 3.2 KB、检测与文档工具

- [x] `encyclopedia_lookup` / `recipe_lookup` 代码：严格 KB schema v2、verified/provisional catalog、双库联合原子发布、固定 jieba/BM25 参数、全量稳定排序、OOV 空结果、citation/provenance 和每 hit binding。正式链路依次要求外部 source、catalog、bundle SHA-256，`allow_provisional` 不能降低 verified artifact 的锁要求，正式 bundle 也不得脱离已外锁 catalog 加载。现有 wiki/recipe adapter 只会诚实地产生 `unverified` v2 条目；补齐真实 source revision/license 后才能构建正式 catalog。
- [x] `object_detect` 代码：create-only 模型 manifest 固定权重、类别映射、配置与 backend 版本；正式 loader 必须匹配外部登记的 manifest SHA，publisher 返回的自算 identity 不直接获得 formal 标记；推理前后复验模型和输入字节，输出稳定 detection ID、bbox 与 runtime binding。真实权重和可选 runtime 尚未纳入正式 artifact。
- [x] `document_ocr` 代码：本地可注入 OCR backend、行级 evidence、结构化字段引用和 untrusted-content 标记；document-safety catalog 以外部固定 review-ledger SHA 为信任根，逐条绑定精确 query/asset、图片 bytes、AssetCatalog 和脱敏直接父血缘，并整包 create-only 发布。正式 MVP gold 必须来自 WildReceipt，CORD/SROIE 作字段补充；Wikimedia Documents 只作布局/拍摄 challenge。真实 OCR 模型、人工审批 ledger/catalog 与 gold 仍需单独发布和验收。
- [x] `registry`：恰好注册 7 个 canonical MVP 名称，递归严格 JSON Schema、版本化/自哈希 ToolSpec、canonical I/O、query asset/text context gate、live handler/runtime binding 和 create-only registry manifest；三个商品工具的正式输出为含输入/向量/binding 的 trace，正式运行显式注入 service，不使用模块级 singleton。

### 3.3 工具独立 benchmark

- [ ] 优先复用许可兼容的公开 gold；mini 每工具 10–20 条仅验接口，core 每工具 30–50 条，full 每工具 50–100 条人工/公开金标，不再以 3 条挑选样本作为质量结论。
- [x] formal gold loader 强制匹配外部 gold 与逐 case 人工 `GoldReviewLedger` 摘要，绑定 reviewer、时间、依据/evidence revision、盲于系统结果声明和 gold-case hash；精确覆盖权威 assignment，拒绝 group 跨 `tool_dev` / `tool_test_frozen`，且 frozen split 覆盖 ranking/detection/OCR。
- [x] tool gold 与 Assistant `val/test_frozen` 的隔离 verifier 同时深验 Assistant 四份外锁 artifact、同一 AssetCatalog/policy 与两侧 assignment，并要求 catalog leakage component 零交集；协调重算 self-hash 不能绕过外部 digest。真实 gold/Assistant artifact 仍需运行并产生零交集回执。
- [x] diagnostic benchmark 基础设施报告 ranking Recall@K/MRR/nDCG、label-aware IoU=0.5 detection precision/recall/AP50，以及 Unicode codepoint CER、字段准确率和 grounded evidence coverage；字段只有“值正确且引用的真实 OCR line 文本支持该值”才计 evidence。AP50 明确不是 COCO mAP，错误样本保留在分母。
- [x] typed run/verifier/evaluator 的代码覆盖已扩到全部 7 个 canonical 工具及 Multi-Product 组合链：image/text product ranking、style、KB hit/citation/evidence、detection、OCR grounded fields，以及 detect→per-object retrieval；真实 gold、冻结阈值和各工具质量报告仍未产生，因此不能把机制覆盖写成七工具已正式验收。
- [x] 实现只消费 verifier handle 的独立 formal benchmark evaluator 与 CLI；它会重新验证 artifact bytes、外部 digest、人工 gold-review ledger、tool↔Assistant component 隔离、runtime authority/evidence binding、工具覆盖和错误分母。当前仍没有绑定真实模型/gold 的 mini formal report，executor、裸 predictions 和 diagnostic fixture 不能由此获得 formal 身份。
- [ ] 工具结果跨系统配置固定或按 cache hash 复用，避免把工具漂移误归因为 Skill 演化。
- [ ] 报告 p50/p95 latency、吞吐、费用、失败率和索引构建资源。

**验收：** mini fixture 的工具调用在 Python socket guard 下不访问外部 API；所选正式 profile（core/full）的工具均通过预注册 tool_test 门槛；资产隔离检查为 0 交集；正式 Assistant run 引用固定 index/tool manifest。该 guard 不等于 OS 级断网。

## Phase 4：200 条 grouped dev_mini 合成查询与轨迹

- [x] 实现 `synthesis/planning.py` 与 active-plan store：从 portfolio mini AssetCatalog 生成 200 条 grouped `dev_mini` plan，在 asset/image/leakage component 层分配且保持跨意图/boundary group 完整；最终 plan SHA=`5e3b0d67545c3d9a311df6c133c427564e4064174158eddf603b3b075e9fa6be`。
- [ ] Formal planner 仍须按 source portfolio 复验监督角色与 gold 资格：ABO/RPC/FashionIQ/WildReceipt 分别承担 Exact/Multi-Product/Style/Document 主 gold；DuRecDial 2.0/CrossWOZ/MUGE 只进入语言/交互模式层，Codex Mock 只作 `synthetic_derived` 组合；challenge 来源不得占用主 capability 配额。当前 query corpus 的完成不能替代这些 formal gold 门。
- [x] 通过 `generate-phase3-corpus` 工作流按 capability 生成单轮或澄清 trajectory；八个 accepted 批次均绑定 turns、Codex 模型声明、prompt/seed、template family、asset catalog、generation input 和 batch hash，生成只进入 staging 并逐批由 owner 接受。
- [x] 实现生成草稿的确定性收口契约：v2 draft manifest 对当前 plan/catalog/seed、完整 batch input 和草稿字节做 canonical hash 绑定；旧计划、旧 seed、换图或换 draft 均不得被 stage 后回填成新 provenance。
- [x] 实现确定性机械 validator：精确 25 条、plan ID 全覆盖、turn shape、图片/catalog/seed/TaskSpec/plan binding、唯一 final-user text、revision、reject/accept 与 append-only ledger 均 fail closed；失败修订保留在 rejected。
- [ ] 事实证据、card requirement、禁用词、工具可达性与 formal gold 的内容质量门仍须随真实 tool/catalog pipeline 完成；人工 query 接受不能替代这些 evaluator 门。
- [ ] 实现模板指纹 + 文本 embedding 去重；重复簇、repair/reject 原因和阈值进入 query manifest。
- [x] 五 intent 与六个 MVP capability 完成预定覆盖：intent=`35/35/35/35/60`，Exact/Multi-Product/Style/Encyclopedia/Document/Recipe=`35/35/35/35/30/30`；200 个 query ID 与 final-user text 唯一。
- [ ] `dev_mini` 尚未按 group 划成 pilot_seed / pilot_opt / pilot_route_gate / pilot_body_gate / pilot_shadow；当前全部记录仍是 `split=dev_mini`，尤其不能冒充 Stage 1 `opt_pool`。
- [x] 计划包含恰好 40 条 boundary，八批全部由 owner 人工审阅并正式接受；188 条为 `[user]`，12 条为 `[user,assistant,user]`。这里只能声称人审 query 可接受性，不能把 plan-owned auto label 说成 200 条人工标注。
- [ ] Formal natural ambiguity 仍须逐组保存两个以上合理解释或 acceptable capabilities 依据；“至少 40 条由人工直接编写或重写”也没有因人工点击接受而自动满足。
- [x] 生成批次全部经 owner 人工审阅；拒绝修订 `001-r1/001-r2/005-r1/006-r1` 原样保留，accepted 为 `001-r3/002-r1/003-r1/004-r1/005-r2/006-r2/007-r1/008-r1`。没有第二位人工标注者，不能报告 inter-rater；页面 elapsed timer 也不等于 active review effort。
- [x] 每批输出机械 quality report：count=25、duplicate=0、intent/boundary/turn 分布和 results SHA 可重验；按 ledger 顺序拼接得到 315,195-byte `queries.jsonl`，SHA=`6f8eda4fe663733708d6e797c954e58f098d3938857c6f2b143e24e202437f03`。
- [ ] evidence coverage、人工事实支持率、capability label agreement、模板语义近重复率、source shortcut 与图文不一致率仍待 formal evaluator/gold；不由上述机械报告替代。

**当前验收：** 本地 `dev_mini` query corpus 已收口为 8 批/200 条，staging 为空且 `next_batch_id=null`；plan/query/ledger SHA 已记录，accepted 结果与聚合文件逐字节一致。`data/` 被 Git 忽略，语料本体不随代码提交；所有 query 仍为 `synthetic_derived`。Formal 主源/gold、pilot 子划分、自然歧义依据与内容质量门仍未关闭，此阶段不冻结最终 test，也不把 `dev_mini` 重命名为 `opt_pool`。

## Phase 5：助手主循环与可比基线执行器

- [x] 静态 authoring 模块已实现严格 UTF-8 可读公开资料与原始字节双绑定、SKILL.md roundtrip、canonical hash、Bank manifest、parent lineage 和 capability mapping；SpecBaseline 可走 verified formal input，LLMStatic Python gateway 仅限 diagnostic。
- [x] 实现正式隔离的 LLMStatic authoring runner：由外部摘要锁定 Docker sandbox profile、engine binary、不可变镜像与 provider egress policy commitment；容器 rootfs 只读、drop all capabilities/no-new-privileges，只挂载 canonical AuthoringRequest（只读）与新输出目录，临时凭据只按 provider 环境变量名注入；parent 保存 profile/image/command/request/response/stdio/latency 回执。v3 默认 Flash request/response/isolation 已完整保存但 draft 有 40 个 schema 错误，作为历史负结果保留。v4 又用最终源码 create-only 锁定 LLMStatic authoring sandbox `formal-v3`/`locks-v4`，四项 egress/DNS 探针通过，并在任何新输出前冻结 forced submission、唯一 run/claim 与额外一次 attempt。该 attempt 已执行：provider framing、模型、usage、费用与隔离合格，但 `drafts` 被序列化为字符串，严格拒绝；内部只读诊断仍有 16 项错误。claim 已耗尽，无 retry/fallback/repair。事后审计区分了 provider wire/key 违规、作者规则映射错误和项目内部契约矛盾：function schema/提示词没有表达 lexical sort，而 TaskSpec-copy 的 6 个源数组顺序与 Runner 冲突。新的前瞻协议必须把规范复制字段交给可信 compiler、让 compiler 负责表示层排序并继续严格校验语义集合；不能修补旧响应。实时网络复验当前只支持 Docker，不虚构 Podman 等价证据。
- [x] 已实现可注入 backend 的五配置统一 diagnostic 执行/产物契约，固定 query 顺序、backbone、registry runtime 和 inference budget，并保存逐请求快照；真实模型上的 `route → inject → generate → tool_loop → compose` 纵切仍待 200 条 mini 验证。
- [x] production Assistant runner 自行采集 provider/model/request ID、usage、Runner 总 latency 与实际 registry tool trace；正式入口只接受 exact `ProductionAssistantRunner` 与 verified-file AssetCatalog，逐请求复核 asset/path/bytes 和 cloud-upload permission，Skilled 配置执行 Description-only route → selected Body injection，逐行 receipt 绑定 catalog/asset/model/tool。任意 backend response 自报字段只能进入 diagnostic bundle，不能获得 runner-owned 身份。真实五配置运行回执仍待发布，且 Phase4 在 Judge 校准完成前继续 `formal_eligible=false`。
- [ ] router 输出 skill slug/no-skill，并通过 BankCapabilityMap 评价 capability；保留原始候选和置信信息。
- [x] Assistant request/response 契约同时支持明确隔离的 diagnostic `backend-reported-v1` 与 production `runner-owned-v1`：后者由 Runner 调 registry 后构造 tool trace，并把逐模型调用与总时延 receipt 嵌入结果行；真实工具子集、实际调用和外部运行回执仍须随 mini run 复验。
- [x] `AssistantRunResultRow` 与五配置 run bundle 可 create-only 落盘、深度重载和恢复校验；query ID 唯一，异常转为保留在分母中的固定错误结果。
- [x] 五配置 matrix contract 强制共享同一骨干、工具 runtime、输入顺序和 inference budget；NoSkill 只移除 Skill treatment。尚需真实 Assistant run 证明调用服务也遵守契约。
- [x] S1+S2 与 Full 的计划/结果契约可绑定同一 Stage 2 route decision；真实 200 条纵切仍须逐 query 证明 route trace 相同。

**验收：** dev_mini 上 NoSkill 与两个 seed Skill 可运行；相同输入/manifest 可恢复且不重复计费；NoSkill Routing F1 输出 N/A。

## Phase 6：冻结尺度的双路评测框架

- [ ] 从 Phase 2 的 Task Specification v0 生成并人工冻结 mini `benchmark_rubric`；所有配置和 Bank 版本使用同一 rubric，mini 结束后只能在正式预注册前做一次显式版本修订。
- [ ] rule path 分为 frozen benchmark rules 与 evolving diagnostic rules，二者不可混用。
- [x] final evaluator 的 allowlist serializer 只接受盲化 `FinalEvaluationPacket`，并拒绝配置、Bank/skill、Description、Body、GT capability、source、split 与演化诊断字段；独立 `FeedbackPacket` 只用于反馈路径。真实 prompt/rubric 尚未冻结。
- [x] feedback/final evaluator identity、prompt snapshot 和 v2 cache namespace 已分别绑定并 fail closed；当前使用 AIFast `gemini-3.6-flash` Feedback 与 DashScope `kimi-k2.6` final。`evaluator-isolation-v3` 锁定不同 provider runtime、模型家族及 packet/prompt/input/artifact 隔离，同时披露第三方网关身份风险。两类 runner 均从权限 preflight 后的 verified catalog 瞬时生成 Base64 `image_url`；Feedback 严格解析视觉归因 JSON，final 严格解析原始整数分数并本地编译 tier/`J_project`，错误保守留在分母，单结果 receipt 可 create-only 落盘重验；Gemini Base64 strict-JSON live smoke 已通过。production 五配置 1,000 行矩阵 receipt/manifest 与 Judge—human 校准仍未完成，故 C4 不因单结果连通而关闭。
- [x] 已实现强类型 JudgeScores、tier 一致性、值域、raw/error identity 和保守错误处理；真实 Judge 输出与人工校准仍缺。
- [x] 已实现 `requires_card` 与 `J_project` 公式及 missing-card、no-card、error fixture；synthetic 手算只验证实现，不构成效果分数。
- [ ] 选 50–100 条人工评分，按 group 预拆 `judge_calib` 与密封 `judge_audit`：只允许用 calib 调 rubric/prompt；audit 在 Phase 7 五配置输出产生后解封，覆盖不同质量层并报告一致率、Spearman/Kendall、分维偏差和重测 variance。
- [ ] temp=0 只作为解码参数，不把“重跑逐字节相同”作为云模型正确性条件。

**验收：** 同一 rubric 候选能公平评估 NoSkill 与固定 seed Skills，且盲化单测证明 final prompt 不含 Bank/Skill/GT 字段；judge_calib 达标、audit 仍密封；所有错误保留在分母和报告中。

## Phase 7：mini 三阶段纵切与正式协议冻结

### 7.1 LLMStaticSkill 与 Stage 1 pilot

- [x] 冻结并保留历史 `AuthoringPacket`/事件链：schema v3 主/备 packet、Qwen v3/v4 正式负结果和已消费 Qwen v4 claim 均为不可变历史；Qwen v5 只提供 author-content v2/compiler v4 语义来源。Codex/high v1 在 approval 前被取代；v2 exact approval 后在 claim/process 前因 PATH gate 拒绝；v3 在 preapproval P0 审计后废弃；v4 candidate 已冻结但未批准或调用。Codex 四代累计 inference=0，旧 approval/claim budget 不得迁移。
- [x] 完成 Codex/high v4 的离线代码与独立 P0/P1 审计：保留绝对 CLI binary、确定性 env、仓库外 scratch、source/runtime closure、I/O/process supervision 与 strict event validation；新增 classified create-once、nonce claim、pre-existing conditional terminal guard 和严格 canonical-bundle override。该项只证明候选代码边界，不是 freeze 或 authority。
- [x] 生成并复验 `authoring-codex-high-20260724-v4` 的 exact packet/schema/request/stdin/runtime/freeze，记录 file/payload、runtime/source、planned guard 与 v2-retirement 摘要；定向回归为 v4 `24 passed`、v2/v3/v4 `48 passed`、相关 core `130 passed, 1 skipped, 2 deselected`。扩大回归明确排除当前环境缺少 `huggingface_hub`、无法收集的 OpenCLIP artifact 单测后，首轮为 `1069 passed, 25 skipped, 2 deselected, 15 failed, 8 errors`；23 项均为 Windows 目录原子 rename 的间歇性 `WinError 5`，定向复跑先 `22 passed, 1 failed`，最后一项再单独通过。因此当前没有可复现的逻辑失败，但不声称一次全仓运行全绿，也不声称 OpenCLIP 可选路径已验证。该完成项仍不创建 authority，也不执行 inference。
- [x] owner 已授权并执行独立 Codex v4/v5 run；v4 负结果保留，v5 生成合格 pre-review draft，0 retry/follow-up/repair/fallback/tool。
- [ ] owner 完成 v5 draft 人工 checklist；真实 C3 authority runtime 后确定性编译并重建 LLMStatic Bank。
- [x] 实现版本化 diagnostic 静态 authoring gateway、正式隔离 container runner 与确定性编译器：`LLMStaticSkill` 只允许读取 AuthoringPacket 并通过不可执行的 `submit_authoring_payload` 提交一个无哈希结构 payload，可信 Runner 严格校验后绑定输入并生成全部 SHA-256，再确定性生成 Description/Body/Cs/Od；保存 prompt/raw output/model/tokens/cost、用户 checklist/diff、输入输出 hash，并拒绝污染输入。ToolSpec 在 packet 中冻结，实际工具 runtime 到 Bank 编译时独立锁定。另可从 verified input 生成不调用 LLM 的 formal SpecBaseline；当前主机历史 Qwen LLMStatic authoring sandbox `formal-v3`/`locks-v4` 和 Qwen v4 调用回执均已冻结，但 draft 被拒绝，因而没有冻结 Bank。任何新 authoring 方案必须建立新的 owner-approved deviation，不能复用 Qwen v4 claim 或 Codex v2 approval。
- [ ] 对同一 Task Specification、作者模型、骨干模型和工具权限执行 LLMStaticSkill 与 S1 Creator pilot；用户执行同一 checklist 和相同审查分钟上限，authoring 调用、tokens、费用和机器时长分项核算。
- [ ] 若能获得未接触本项目轨迹与结果的独立合格领域专家，可按预注册资格和固定人时另建 `ExternalExpertManual`；否则明确记录“未复现论文专家人工基线”，不得把 LLMStaticSkill、SpecBaseline、LLM 草稿或用户格式审查改名为 ManualSkill。
- [ ] S1 Creator 从 pilot_seed/pilot_opt 抽轨迹 + Bank reference Skill 输出 SKILL.md；保存完整 prompt/job/result hash。
- [ ] Engineer Loop 限定 operators、static refs、schema、工具权限和 5–10 条试跑；≤3 轮，所有人工修改与分钟数留档。
- [ ] Human Gate 使用 AI 预审 + 用户 checklist 终审；approve/reject、原因和耗时全部留档。

### 7.2 Stage 2 pilot

- [ ] 在 pilot_opt 上挖掘误路由/正确路由案例，完成至少一次只改 Description 的 Update/Discard/Merge 纵切。
- [ ] 候选 Bank 断言未合并 Skill 的 Body/Cs/Od hash 不变；pilot_route_gate 决定接受/回滚，pilot_shadow 只报告。
- [ ] pilot 不强行满足论文正式样本下限；实际失败数不足时只验证机制并标为 exploratory。

### 7.3 Stage 3 pilot

- [ ] 对 route-correct 与 route-deviation response 做 attribution、四维聚合、Body patch 和一次接受/回滚。
- [ ] 断言只改 Body，并复用 Stage 2 route cache；pilot_body_gate 决定接受/回滚。
- [ ] 跑 `correct-route-only`、盲化 final Judge 和 frozen-vs-diagnostic rule 的最小消融，验证贡献可隔离。

- [ ] 根据 mini 的质量、标注耗时、Judge 校准、调用费用和 CI 精度模拟，冻结正式 `evaluation-protocol`、Task Specification、rubric、prompt、预算上限与 core/full profile；冻结后不得因结果方向修改。
- [ ] 在五个主配置 pilot 输出上解封 `judge_audit`；若未达预注册一致性门，Evaluator 不得进入正式实验。修订后必须补充新的密封 audit 样本，不能在原 audit 上反复调参；通过后冻结 evaluator hash。

**验收：** NoSkill / LLMStaticSkill / S1 / S1+S2 / Full 在 200 条内部子集上端到端可运行，SpecBaseline/OracleRoute 只作诊断；每阶段 edit 边界、回滚、manifest、缓存和错误路径经测试；只声称工程纵切，不发布 mini 效果结论。未通过不得扩量。

## Phase 8：正式 core/full benchmark 构建与冻结

- [ ] 根据 Phase 7 冻结的选择执行 core（主目标）或 full（扩展目标）；从 dev plan 运行同一 synthesis/validation pipeline，禁止另写一次性全量脚本。
- [ ] 在 `asset/product/near-duplicate/boundary/template/generator` 联合 group 上切分；跨 split 每类 group 交集必须为 0，Exact Match 的同 product 多视图正例整体留在同一 split。
- [ ] 按第 3.1 节固定拆分 route_gate/body_gate/shadow_val；建立 source-held-out、human-written `challenge_dev/challenge_final`，test/challenge_final 含未见模板与生成批次。
- [ ] opt_pool 允许弱监督并保留 label provenance；val/test 100% 用户人工确认。全部 boundary 与随机 10–20% 非 boundary 在至少 7 天洗脱期后盲重标；只有第二位真人参与时才报告 inter-rater。
- [ ] 标注前先用 50 条实测每条分钟数并计算总人时；超出冻结上限时在看结果前选择 core、缩小 capability 或延长周期，不得用 LLM “一致”冒充人工确认。
- [ ] 模糊样本保存 primary + acceptable set，不强迫天然歧义只有一个合法标签；capability 按第 3.1 节 eligibility 表审计。
- [ ] 冻结 test/challenge_final 内容、rubric、asset groups 和 manifest hash；Bank 映射独立保存，禁止回写 Query。
- [ ] 统计 split × intent × capability × boundary × source × episode，并运行 source-classifier/image-only/text-only、模板 ID 和图片尺寸等 shortcut baselines。
- [ ] 在正式生成前预注册 shortcut 阻断线：source/template/generator/尺寸等非语义特征若达到 `max(2×chance, majority+0.20)` macro-F1，benchmark 不得通过；必须重平衡/重生成，或将污染 capability 降为 exploratory。若 image-only/text-only 与完整输入差距小于预注册 ε，则加入跨模态冲突集，并把该 challenge 结果作为对应能力的主结论。
- [ ] 重跑 Phase 4 的质量门并发布 data card：事实支持、标签一致、模板近重复、repair/reject、许可、PII 与人时。

**验收：** 所有 group 泄漏交集为 0；所选 profile 满足预注册 capability 样本资格、gate 分层与 CI 精度目标；shortcut 未触发阻断线或受污染能力已降级；challenge_final 与训练/开发模板及生成批次隔离；test 和 challenge_final 密封后才进入正式演化。

## Phase 9：正式 Stage 1/2/3 演化

### 9.1 LLMStaticSkill 与 Stage 1

- [ ] **LLMStaticSkill：** 用冻结 AuthoringPacket、作者模型和 budget 一次性生成静态 Skill；不读取任何 corpus 轨迹、标签、响应、route/body gate、S1 产物或结果，生成后不得按效果迭代。LLM 参与必须如实标记为 `llm_static`。
- [ ] **S1 Creator：** 只在完整共享 AuthoringPacket 之外增加从 opt_pool 预注册抽取的 TrajectoryBundle；任何 Reference Skill 都必须已在 AuthoringPacket 中同样提供给 LLMStaticSkill。Engineer Loop ≤3 轮、Human Gate 与 LLMStaticSkill 使用相同 checklist 和人工审查分钟上限，模型 tokens/费用另计。
- [ ] 生成 `bank-llm-static-v0`、诊断 `bank-spec-v0` 与 `bank-s1-v0` manifest、capability mapping 和完整 authoring/compiler/creator/人工审计记录；Git tag 只作为内容哈希的里程碑别名。
- [ ] 在 route/body gate 之外的预注册诊断子集跑 NoSkill / SpecBaseline / LLMStaticSkill / S1 / OracleRoute+S1 sanity check；只修实现 bug，不按效果方向改协议。

### 9.2 Stage 2 Route Optimizer

- [ ] 在 opt_pool 上对照 stable capability GT 挖掘失败；按 strict 与 acceptable-set 两种口径记录。
- [ ] 根因分类：boundary ambiguity / missing skill / visual parsing error；另保存 classifier 置信和人工审计子集。
- [ ] Case A 使用误路由样本；Case B 使用正确样本，二者均看图文和全部 Description。
- [ ] Update 仅改 Description；Discard 保留 alias/lineage；Merge 只允许 Body/Cs/Od 可无损继承的重复 Skill，否则进入 Stage 1/3 待办，禁止暗改 Body。
- [ ] 候选 Bank 上断言所有未合并 Skill 的 Body/Cs/Od hash 不变。
- [ ] route_gate 使用 paired bootstrap：满足预注册非劣界/最小效应；仅对达到第 3.1 节预注册最小 n 的 capability 应用分组非回退门，其他组只描述；shadow_val 只监控，不参与每轮选择。
- [ ] 最多 4 轮；每 Skill 目标 ≥30 失败样本，不足时报告实际数且不伪造补样。
- [ ] 输出 P/R/F1、混淆矩阵、根因分布、操作清单、Bank 大小、成本和延迟。

**Stage 2 验收：** 仅声明 route_gate 上的经验单调；test/challenge_final 保持未读；source-held-out/challenge_dev 结果只作开发诊断。

### 9.3 Stage 3 Body Refiner

- [ ] 主实现纳入所有 attributed responses，包括 route-correct 与 route-deviation；分别分层报告。
- [ ] 另做 `correct-route-only` 消融，以隔离 Body 能力但不替代主实现。
- [ ] 每 Skill ≥50 条；按 Skill × 维度聚合 tier，并记录有效样本数。
- [ ] 论文阈值 TCR .20 / CCC .10 / CQ .05 / CA .10 作为预注册主配置，同时做阈值敏感性分析。
- [x] visual feedback Judge 已结合原图、完整用户对话、可见 response/cards/tool evidence 与内部 tool trace，并严格解析 `rule_violations`、`ideal_response_gaps`、image-grounded evidence 与 `skill_suggestions`；parse/provider 错误显式留档且不得进入 Refiner。跨样本聚合、verified Assistant-row join 与到 Refiner 的 batch contract 仍待 production orchestrator 接线。
- [ ] 代码断言只允许 Body 改动；Description、Cs、Od、capability mapping 和 Stage 2 route cache 不变。
- [ ] frozen rubric 下比较旧/新响应；Body-relative 指标只作诊断，禁止通过删除约束刷分。
- [ ] Human Gate + body_gate paired bootstrap J_project gate；分组非回退仅用于样本量达标的 capability，安全硬规则无论样本量都不得回退；≤3 轮。
- [ ] S2 与 Full 复用路由结果，Routing F1 应完全一致；不把云模型随机差异解释为论文现象。
- [ ] 在 opt_pool/body_gate 上分别构建 −LLM Judge / −Rule Path / −Qualitative Induction / −Statistical Aggregation 四个消融 Bank；与主 Bank 一同冻结 manifest，构建过程不得读取 test/challenge_final。

**Phase 9 验收：** 逐轮报告四维、J_project、置信区间、规则覆盖、Body token 数、费用和 latency；LLMStaticSkill/S1 的公共信息、authoring budget、审查人时和额外模型成本分项可比较；test/challenge_final 保持未读，正式 Bank manifest 冻结。

## Phase 10：最终实验、消融与人评

- [ ] 冻结 dataset/query/split/index/tool/rubric/prompt/model/bank/environment manifest 和 clean 代码 commit；若必须用 dirty tree，保存 patch hash 并把 run 标为非发布候选；执行 dry-run cost gate。
- [ ] 五个主配置在同一 `test_frozen` 上执行一个 canonical paired run：NoSkill / LLMStaticSkill / S1 / S1+S2 / Full；SpecBaseline 与 OracleRoute 只作诊断；若存在合格 ExternalExpertManual，则只作为预注册补充列。
- [ ] OracleRoute 只作诊断上界，不混入五个主配置表。
- [ ] 主结果：四维、J_project、capability/intent 路由、paired delta+95% CI、分 intent/capability/source/boundary 拆解。
- [ ] 收敛曲线：Stage 2 最多 4 轮、Stage 3 最多 3 轮；同时标费用、Bank 大小和最差组。
- [ ] 对 Phase 9 已预先构建并冻结的四个 Stage 3 消融 Bank，在预注册 test 子集/challenge_final 上一次性评估；解封后禁止构建新 Bank 或据此继续修改系统。
- [ ] core 100–150 条、full 100–300 条 Full vs LLMStaticSkill、Full vs S1+S2 随机顺序盲 SBS；报告 win/tie/lose。单人项目报告洗脱期 intra-rater；有第二位真人时才报告 inter-rater。
- [ ] 除 canonical run 外，对 NoSkill、S1+S2、Full 在预注册 test 子集追加 2 次 repetition（预算允许时做全量），固定输入顺序并分别报告样本 bootstrap CI 与 run-to-run variance；Judge serving variance 在独立校准子集重复测。
- [ ] 做 error analysis：source shortcut、tool failure、route deviation、judge disagreement、missing skill、visual parsing。

**验收：** test 未被用于任何改写或门控；所有配置共享任务尺度、工具版本和样本；统计、排除、错误和费用可从 tracked summary 重建。

## Phase 11：报告、README 与 Demo

- [ ] `report/skillchain-open-reproduction.md`：Reproduction Contract、论文未披露项、数据/模型差异、方法、主结果、消融、人评、成本、误差、负结果和局限。
- [ ] `README.md` 与 `README_en.md`：架构、三行 mini quickstart、结果摘要、manifest、许可和复现实验命令。
- [ ] 每个里程碑更新 README；不再等 Phase 11 才首次提供可展示物。
- [ ] Gradio Demo：展示路由候选、capability mapping、Skill、tool trace、证据、cards 和 Bank 前后对比。
- [ ] Demo 使用独立 permissive dataset/index；限制数据不得进入公开界面或第三方 API。
- [ ] CI：Python socket-guarded unit + synthetic mini E2E；真实数据/API integration 单独手动触发，Linux workflow 取得外部 run receipt 后才记录平台覆盖。
- [ ] 发布 v1.0 manifest bundle；最终 Git tag 只指向已通过复现检查的 commit。

## Phase 12：持续自进化研究平台

- [ ] 抽象 `Creator / RouterOptimizer / BodyRefiner / FailureAttributor / Evaluator / AcceptanceGate / DataEpisode` 接口。
- [ ] 每篇后续论文通过 `ExperimentSpec` 注册策略和参数，不直接修改全局常量形成隐式实验。
- [ ] 构建 T0/T1/T2/T3 episodes：初始流量、可适应 shift、未来 holdout、new-capability/OOD。
- [ ] missing-skill 进入 discovery queue 并可触发新 Stage 1；visual-parsing error 进入 upstream backlog。
- [ ] 保留 replay buffer 和旧任务回归集，评估稳定性—可塑性、遗忘、forward/backward transfer。
- [ ] 增加 Bank 规模实验（约 10/50/100+ Skills）、层级路由/检索路由和 token/latency scaling。
- [ ] 统一比较质量、成本、延迟、Bank 复杂度和人工投入的 Pareto。

---

## 6. 里程碑与停止条件

| 里程碑 | 必须交付 | 未满足时 |
|---|---|---|
| M0 研究协议与集成 | Reproduction Contract、Task Spec v0、schema v2、group split、Phase 2/3 集成分支、socket-guarded CI workflow 与外部 run receipt | 不得启动付费全量合成 |
| M1 工程纵切 | 200 条、MVP 工具、盲化 Judge、NoSkill/LLMStaticSkill/S1/一轮 S2/一轮 S3、回滚与 manifest | 不得构建 core/full |
| M2 正式预注册 | Judge—human 校准、标注计时、cost/CI precision plan、冻结 profile/rubric/prompt | 不得启动正式 benchmark |
| M3 正式 benchmark | 选定的 core 或 full grouped corpus、人工 val/test、dev/final challenge、冻结 hashes | 不得启动正式演化；final test 继续密封 |
| M4 正式演化 | 可比 LLMStaticSkill/S1、最多 4 轮 S2、最多 3 轮 S3、冻结 Bank | 不得读取 final test |
| M5 最终实验与发布 | 五配置、Oracle 诊断、CI、消融、人评、运行方差、成本、报告与可复现 bundle | 方可声称完成相应 profile 的机制复现 |
| M6 平台化（可选） | episodes、策略接口、回归集 | 不阻塞 v1.0；作为后续研究独立迭代 |

预算或时间紧张时，优先级为：实验协议与防泄漏 > LLMStaticSkill/SpecBaseline/OracleRoute 基线 > 独立评价与人评 > 完整三阶段纵切 > 数据规模 > Demo 装饰。`ExternalExpertManual` 仅在取得合格独立专家时作为可选补充，不得为了按期得到“上涨曲线”跳过标签、切分和 evaluator gate。

---

## 7. 论文未披露项与本项目选择

| 论文歧义 | 本项目处理 |
|---|---|
| Table 2 Avg 无法由四个已报告维度直接复算；附录归一化区间表述不一致 | 报四维及各自样本数；使用明确定义的 J_project，不声称数值等价 |
| Routing F1 未说明 macro/micro/weighted，正文 intent 与 Table 8 Skill 标签粒度不完全一致 | capability macro-F1 为主，其他口径全部同时报告 |
| Rule path 完整规则未公开 | 冻结公开 rubric/rules，并把本项目新增规则列为实现选择 |
| Merge 后 Body/Cs/Od 继承未说明 | 仅允许无损继承；其他 Merge 进入跨阶段待办 |
| Stage 3 声称只改 Body，但论文 F1 明显下降 | 复用 Stage 2 route cache；本项目不把 F1 下滑视为预期现象 |
| NoSkill 无 Body 时 Judge 尺度未公开 | 所有配置使用相同 frozen task rubric，Body 只作诊断上下文 |

---

## 8. 主要风险与预案

| 风险 | 预案 |
|---|---|
| 下载可得性被误当成监督适配性 | 机器可读 source portfolio；主 gold、语言层和 challenge 分离；本地 raw cache 不提供 ready 身份 |
| 公开数据与意图强相关 | 多意图复用资产、跨源 holdout、source/image/text shortcut baselines |
| 同图/近重复泄漏 | asset/product/pHash/boundary/template/generator group split；跨 split 交集验收 |
| Exact Match 自检索或无正例 | 正例保留同 product 的另一真实视图；排除同 asset/near-duplicate；product group 不跨 split；单图商品不进正式 Exact Match |
| capability 样本不足 | 缩小正式 Bank；不足的 capability 不进入主实验，不伪造样本下限 |
| Judge 自洽偏差 | frozen rubric、反馈/最终 evaluator 分离、人评校准、盲 SBS |
| val 适应性过拟合 | gate/shadow 分离、paired CI、final test 一次性使用 |
| 云模型非确定或别名漂移 | raw output/cache/model identity/canary；报告 variance，不要求逐字节复现 |
| LLM 调用预算超支 | dev_mini 实测、内容哈希缓存、cost gate、硬上限和中止条件 |
| 单人人工标注超支 | 50 条计时、core 优先、总人时硬上限、洗脱期重标；无第二真人不声称 inter-rater |
| 爬虫合规或内容删除 | 条款/robots/速率/PII acquisition review；不绕过鉴权；保留删除、重建和 provenance 路径 |
| 数据不可云上传或公开演示 | 五维许可矩阵；本地 backend；独立 permissive Demo 数据 |
| 文档 PII/prompt injection | EXIF/PII 审计、打码、OCR 安全测试、拒绝不合规样本 |
| 分支继续漂移 | M0 先集成；schema/manifest 先冻结；后续实验从同一主线分支化 |
| 负结果影响作品集 | 如实报告并做归因；强调 benchmark、实验隔离、回滚和工程严谨性 |

---

## 9. 求职交付叙事

最终项目重点展示：

1. 在无法访问生产数据时，如何构建带 provenance、人工金标和 group leakage control 的公开视觉意图 benchmark。
2. 如何用资源匹配但不读取轨迹的 LLMStaticSkill、确定性 SpecBaseline 下界、OracleRoute、冻结 rubric、盲化 final Judge（与 Feedback 共享 K2.6，但角色与工件隔离）和盲 SBS 分离 Creator、Router、Body 与 evaluator 的真实贡献，并诚实披露无法复现论文专家 ManualSkill 的边界。
3. 如何实现内容寻址 Bank、受约束 edit、失败归因、统计门控、回滚、成本追踪和漂移适应。
4. 如何报告质量—成本—延迟—Bank 复杂度的 Pareto，而不是只展示一个 LLM Judge 均分。
5. 如何诚实记录无法复现项、论文歧义、负结果和后续研究假设。

这比追求与论文相似的漂亮数值更能证明 Agent 算法工程、实验设计和研究迭代能力。
