# Portfolio 25-query × 5 configs 诊断报告（2026-08-04）

## 结论摘要

本轮已经确认：旧的确定性 post-smoke scaffold 不再具有正式矩阵资格，当前 Bank 来自真实的
Creator / Route Optimizer / Body Refiner 调用、字段级变更约束和可重算的接受/回滚 gate。真实
development gate 接受了 S1 和 S2，拒绝并回滚了 S3；因此正式部署的 `Full` Bank 与
`S1+S2` Bank 内容相同。

干净的 v16 composite batch 已完成 25 query × 5 configs：NoSkill `73.34`、LLMStaticSkill
`73.04`、S1 `76.14`、S1+S2 `77.92`、Full `76.64`。S1+S2 相对 Static 的配对均值为
`+4.88`，route accuracy / 六类 macro-F1 从 `0.88 / 0.8913` 提升到 `0.92 / 0.9277`；但这是
完整的 optimization25，不是独立 evaluation175，聚类 bootstrap 95% 区间 `[-0.4220, 9.7826]`
仍跨 0，不能当作 headline effect。Full 与 S1+S2 使用相同 Bank 和 25/25 相同 route，配对差
`-1.28`、区间 `[-6.8701, 4.0005]`，明确是 no-op 采样差，不能归因于 S3。

v13、v14 和 v15 的 skilled 部分都不是可发布结果：v13 已实际观测到三种非语义契约异常并在
扩展到 287 个 Final 后主动停止；v14 在 17 个 Final 时发现可能把第二轮 Judge 基础设施失败
写成固定零分的 P0；v15 的并发窗口又触发 provider pre-response failure。三次运行均保留为
不可改写的诊断产物。v15 唯一完整、经独立 comparison contract 冻结的 NoSkill shard 被 v16
按 52 个工件逐一复验后外部引用，其余四个配置全部在 v16 fresh root 重跑。

v16 模型行本身没有发现新的指标无效、比较不公平或 treatment 冒名缺陷：125/125 行完整，
Assistant / tool / execution hard error 均为 0，route contract terminal failure 为 0，集中审计
`blockers=[]`。唯一非计分行是 S1 `dm-020`：Judge 首次 empty、第二次返回缺少冻结 wrapper 的
JSON，正确耗尽两次上限并保留 `J=0`；这是 evaluator 格式噪声，不是 runner/parser 缺陷，也不能
选择性补跑。19 条 card violation 集中在 5 个 query，均可归因于路由、公开检索零召回或模型
输出选择；双向 card score guard 已按冻结规则生效，不是 evidence DTO 泄漏或计分失效。

本版的 Bank 候选生成和算法优化轮次至此停止。后续只允许验证程序修复、完成干净矩阵和如实
报告结果；本报告列出的算法空间属于下一版，不再据此生成新 Bank 候选。

## 1. 范围与证据边界

- v13 batch 1 是同一组 25 query 在五个配置上的 125 个 Final，只是 optimization / development
  子集，不是独立的 175-query evaluation。
- v13 batch 1 的原始 `J_project` 均值为：NoSkill `69.02`、LLMStaticSkill `68.50`、S1
  `68.90`、S1+S2 `73.54`、Full `75.54`。这些数值包含下述程序性固定零分，只能用于诊断，
  不能作为五配置最终结论。
- 本报告读取 v13--v16、treatment gate、v3 batch audit 与 v3 analysis 的不可变产物；v16 执行
  发生真实模型调用，后续审计、分析和本报告更新本身均为零模型调用。
- v13/v14 历史产物不得由新 parser、重试策略或人工重建结果覆盖。诊断性重算可以帮助归因，
  但不能回写原始分数。

主要协议入口是 `docs/evaluation-protocol.md` 与 `docs/reproduction-contract.md`；运行边界实现见
`scripts/run_portfolio_shard.py`、`src/skillchain/evaluation/final_runtime.py` 和
`scripts/finalize_portfolio_shard.py`。

## 2. v13 batch 1 的三类已验证程序异常

| 类别 | 已验证事实 | 对指标的影响 | 处置 |
| --- | --- | --- | --- |
| Judge fenced JSON | NoSkill `dm-025` 的 Judge 返回 fenced JSON；内容中可见 `CA=8`、`CQ=12`、`TCR=6`，但不满足冻结的“单个裸 JSON 对象”契约 | 被记为 `invalid_judge_json` 和 `J_project=0`；这是 evaluator contract failure，不是已证明的 Assistant 语义失败 | 历史零分保留；新运行只对 empty 或 non-empty invalid JSON 做一次同线重试，第二次仍失败则继续 fail-close |
| Judge flat schema | LLMStaticSkill `dm-020` 返回扁平 scores 与 dimension names，而不是冻结结果 schema；其可见分数同样为 `CA=8`、`CQ=12`、`TCR=6` | 被记为 `invalid_judge_json` 和 `J_project=0`；不能选择性人工修成有效结果 | 与上一项使用同一有界重试规则；严格 parser 不扩展为“猜测并修复任意 JSON” |
| Route unexpected keys | S1 `dm-011` 的 router 返回了契约外字段，触发 `route_contract_error` | 路由在本地严格校验边界失败并固定零分；该行不能用来证明 S1 语义路由错误 | 新运行仅对 `invalid_json`、`non_object`、`unexpected_keys`、`schema_invalid` 允许一次有界重试；越界 capability、空响应、length、tool-call 和预算失败仍 fail-close |

两条 Judge 原始响应可以在诊断中重建为各自 `J_project=65`，但这种重建不改变 v13 的历史结果。
若只做这一诊断替换，NoSkill / LLMStaticSkill 均值分别为 `71.62 / 71.10`；该数字不能进入正式
结果表，因为它绕过了当时冻结的 parser 和计分路径。

v13 继续运行后又发现同类 route contract failure：LLMStaticSkill 的 `dm-038`、`dm-044`，S1
的 `dm-044`、`dm-047`，S1+S2 的 `dm-044`，以及 Full 的 `dm-044`。这说明 batch 1 的 S1
`dm-011` 不是孤立偶发点，也不能把受影响固定零分解释成算法能力差异。

对应的回归覆盖位于 `tests/evaluation/test_final_runtime.py`、
`tests/evaluation/test_run_portfolio_matrix.py`、`tests/evaluation/test_portfolio_batch_audit.py` 和
`tests/evaluation/test_prepare_portfolio_evolution_inputs.py`。

## 3. v13 与 v14 为何中止

### 3.1 v13：观测到污染后主动停止

v13 在 287 个 Final 时停止。停止原因是已经实际出现上述 Judge 输出契约失败和重复的 route
contract failure；继续运行会把已知的非语义固定零分扩散到剩余矩阵。v13 是部分诊断矩阵，
不是可补齐后发布的正式矩阵。

停止时的账本事实为：settled `CNY 38.473958550000`，unresolved reserve
`CNY 6.392217600000`，accountable `CNY 44.866176150000`。保留 unresolved reserve 是保守记账，
不表示这些调用产生了可用 Final。

### 3.2 v14：未观测到污染，因潜在 P0 预防性停止

v14 在 17 个 Final 时停止。这里必须与 v13 区分：当时并未观测到同类污染；静态审查发现，
Judge 首轮 empty / invalid 已进入一次重试后，若内部第二次调用发生 `provider_error` 或 timeout
且没有 response，旧 runner 可能直接写入 `J_project=0`。这会把基础设施无响应误当作模型质量
失败，因此属于足以停止矩阵的潜在 P0。

v14 停止时 settled `CNY 2.162600000000`，unresolved reserve `CNY 6.848512000000`，accountable
`CNY 9.011112000000`。补丁后的 shard runner identity 以 SHA-256 前缀 `e0a17b…` 锁定；由于运行
语义已经改变，v15 必须使用 fresh root，而不是恢复 v14 checkpoint。

### 3.3 v15：冻结完整 NoSkill，废弃不完整 skilled 分支

v15 在 provider pre-response failure 和熔断诊断后停止。NoSkill 已完整产生 25 Assistant / 25
Final，并以 audit SHA-256 `c08f38a…` 冻结；Static / S1 / S1+S2 只有部分行，Full 未开始，均未
进入正式比较。停止时 settled `CNY 11.723577450000`、unresolved reserve
`CNY 3.962470400000`、accountable `CNY 15.686047850000`。后续没有把 reserve 当作可再次消费的
授权，也没有选择性续跑这些不完整 skilled shard。

v16 使用严格 external-frozen-reference 复验 v15 NoSkill 的 52 个工件与 corpus、catalog、图片、
模型、采样、预算、registry、rubric、parser 和 NoSkill tool contract；四个 skilled 配置从新 root
运行。这样既避免为同一 NoSkill 重复付费，也没有把 v15 的部分 skilled 结果混入比较。

## 4. 真实 treatment chain 与 gate 结论

| Stage | `J_project` | route accuracy | adherence | hard error | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| S1 Creator | `68.90 → 77.12` | `0.84 → 0.88` | `0.585000 → 0.621667` | `1 → 0` | 接受 |
| S2 Route Optimizer | `77.12 → 77.44` | `0.88 → 0.92` | `0.621667 → 0.656667` | `0 → 0` | 接受 |
| S3 Body Refiner | `75.56 → 73.80` | `0.92 → 0.92` | `0.683300 → 0.698300` | `0 → 0` | 回滚 |

S3 使用独立的 stage gate 采样，因此它的 parent `J_project=75.56` 不能与 S2 gate 的
`J_project=77.44` 直接相减。三个 gate 都是 development / optimization 决策，不是独立 evaluation
上的效果估计。

S3 回滚后，最终发布的 S1+S2 与 Full Bank 内容身份相同。这是 gate 的正确结果，不是再次出现
scaffold 冒充正式 treatment。v16 中两者复用了 25/25 相同的 Stage-2 route artifact；paired
`Full - S1+S2` 为胜 8、平 8、负 9，均值 `-1.28`，聚类 bootstrap 95% 区间
`[-6.8701, +4.0005]`。在 Bank 和 route 都相同的前提下，这个差异来自后续 Assistant / Judge
采样，不是 S3 贡献。v13 旧诊断中同一 no-op contrast 曾为 `+2.00`、范围
`[-22.50, +37.50]`，方向反转进一步说明单次小样本阶段差不能归因。正式报告必须写成“尝试 S3
并成功回滚，Full 部署等价于 S1+S2”，不得写成“Full 证明了 Body Refiner 增益”。

## 5. 与论文 Table 2 / 3 / 4 / 5 / 7 及 Figure 3 的差异

论文来源为 [SkillChain 原论文（arXiv）](https://arxiv.org/abs/2606.12984)。

### 5.1 Table 2：离线结果方向不同，但绝对分数不可比

论文 Table 2 报告的 Avg / routing F1 为：NoSkill `59.1 / —`、Manual Skill
`64.9 / 61.5`、S1 `62.5 / 65.5`、S1+S2 `67.2 / 78.0`、Full `72.2 / 73.5`。论文中 S2
主要提升路由，Full 的 Avg 继续上升，但 routing F1 相比 S1+S2 下降。

本项目真实 development gate 的方向是 S1、S2 接受，S3 回滚；v16 optimization25 也以 S1+S2
取得最高描述性均值。这个方向与论文的“Full 取得最高 Avg”不同，但不能据此判定复现失败或算法
相反，因为：

- 本项目 `J_project` 有独立冻结定义，不等于论文 Table 2 Avg；本项目 route accuracy 也不等于
  routing F1。
- 本项目 LLMStaticSkill 不是论文的 Manual Skill baseline。
- 本项目使用公开数据、六个 capability 和公开工具，论文使用生产语义、五类意图和不同工具环境。
- 本项目 Assistant 为 Qwen3-VL-Flash；论文为 Qwen3-VL-235B-A22B-Instruct。Judge / feedback / authoring 模型、
  temperature 和推理预算也不同：本项目包含 Kimi K2.6（temperature 1）与 GPT-5.6-Sol high，
  论文对应使用 Gemini-3.1-Pro-Preview（temperature 0）与 Claude Sonnet 4.6。
- v13 batch 1 只有 25 个 optimization query；论文离线评测为 1,000 条，并要求每 intent 至少
  150 条。论文 S2 每 Skill 至少聚合 30 个失败、最多四轮，S3 每 Skill 至少 50 个样本、最多
  三轮；本项目第一版 gate 不满足这一统计设置。

因此只能比较机制和误差方向，不能比较绝对分数、百分点或显著性。

### 5.2 Table 3：论文逐 intent 增益不能由本项目小样本 gate 复现

论文 Table 3 把 S2 的 routing F1 增益和 S3 的五项质量增益按五个 intent 拆开：S2 的 F1 增益从
Exact Match 的 `+3.1` 到 Encyclopedia 的 `+18.5`；S3 的 Avg 增益从 Utility Assistance 的
`+0.3` 到 Encyclopedia 的 `+8.4`。本项目是六个 capability，且 development gate 没有达到论文
每 intent / 每 Skill 的样本量，因此目前不能形成同口径的逐 intent 增益表。

### 5.3 Figure 3：本项目没有等价的人工盲评

论文 Figure 3 的 300 条 blind side-by-side 中，Full 相对 Manual Skill 的胜 / 平 / 负为
`51.0% / 16.8% / 32.3%`。本项目目前使用自动 Judge，没有与该设计等价的盲化人工 SBS，也没有
论文的 Manual Skill 对手；自动 `J_project` 不能替代该人类偏好结论。

### 5.4 Table 5：支持继续控制 Judge 和 aggregation 噪声，不提供本项目增益换算

论文 Table 5 报告：移除 Judge 时 Avg `-4.2`；移除规则反馈时 Avg `-1.9`、CCC `-3.8`；移除
定性反馈时 Avg `-0.9`；移除 aggregation 时 Avg `-3.8`、CCC `-12.2`。这说明论文体系中的
Judge、规则反馈和跨样本聚合都有贡献。

本项目 v13 的两个 invalid Judge 固定零、相同 Bank 的 S1+S2 / Full 大幅 paired 波动，以及小样本
S3 gate，恰好说明 Judge 噪声和 aggregation 不足仍会主导局部结论；但 Table 5 的消融幅度不能
换算为本项目预期收益。

### 5.5 Table 4：本项目没有线上 A/B 对照

论文 Table 4 的一周线上 A/B 报告 Interactive UV `+1.92 pp`、Full-read `+4.98 pp`、dwell time
`+2.85 s`、7-day return `+1.15 pp`。本项目没有生产流量、线上用户分布或留存指标，不能声称
复现或反驳这些业务结论；Portfolio 结果只能描述公开数据上的离线机制验证。

### 5.6 Table 7：实现规模与统计前提不同

论文 Table 7 固定了 Qwen3-VL-235B-A22B-Instruct、temperature 0 的 Gemini-3.1-Pro-Preview Judge、Claude Sonnet 4.6，并规定
S2 每 Skill 至少 30 个失败样本、最多四轮，S3 每 Skill 至少 50 个样本、最多三轮，离线评测每
intent 至少 150 条。本项目第一版的模型、六 capability、公开工具和 25-query development gate
均不同；这正是只能比较机制方向、不能换算论文百分点的主要原因。

## 6. 根因：事实与推断分开

### 6.1 已验证事实

1. **运行契约把可重试的格式失败写入了算法分母。** v13 的三项异常发生在 route / Judge
   contract 边界，而不是经审计确认的任务语义失败；旧策略缺少严格、单次、有账本约束的重试。
2. **第二次 Judge 调用的基础设施失败分类不完整。** v14 静态审查确认 no-response provider
   failure 可能落成固定零分；修复后它必须保留为未完成 attempt，而不能伪装成 Final。
3. **同一 deployed treatment 的重复采样有显著方差。** v13 的 S1+S2 / Full 使用相同 Bank 和
   相同 route，却出现 `[-22.50, +37.50]` 的 paired 差值范围。因此单次差异不足以做阶段归因。
4. **旧 scaffold 的根因是 provenance gate 缺失。** readiness 曾只校验文件、schema 与 hash，
   没有强制验证 Creator / Optimizer / Refiner 输入、模型 receipt、字段级 diff、parent lineage 和
   可重算 gate。当前 finalizer / loader 已补上这些约束；旧 scaffold 仍只具有 diagnostic 身份。
5. **本项目 gate 的样本量低于论文算法的聚合前提。** 当前 gate 可以验证候选能否执行和回滚，
   不能稳定区分 Skill 缺陷、Assistant 采样与 Judge 采样。
6. **v16 的核心执行链已干净闭合。** 125 行 paired input 零违规，四个 skilled 配置均为 25/25
   Assistant success，所有配置 tool error 为 0，S1+S2 / Full 的 shared route 25/25 一致；没有
   再出现 v13 的 route contract fixed-zero 或 v14 的 pre-response failure 落成任务零分。
7. **S1 `dm-020` 是冻结策略下的 evaluator anomaly，不是程序错误。** 首次 empty 后的第二次响应
   是语法合法但 schema 非法的 JSON；Static / Full 的四次同类 invalid 首轮均在第二次成功，证明
   retry 路径正常。S1 官方均值必须保留 `76.14`；仅诊断性补 wrapper 会得到 `79.54`，该值没有
   正式资格，也说明 25-query 排序对单个 Judge 格式失败敏感。
8. **19 条 card violation 是五个 query 的可解释行为失败。** `dm-002` 的公开 style search 对五
   配置都返回零候选；`dm-003` 的 forbidden-card 违规伴随 encyclopedia → exact-match 误路由；
   `dm-007` 的 required-card 缺失伴随 exact-match → multi-search 误路由；`dm-015` 在 Static / S1
   仍误路由，而 S2 修复后 S1+S2 / Full 合规；`dm-023` 只在 NoSkill 违规。统一 evidence DTO 与
   双向 score guard 均重放一致：9 行由 guard 将 CCC/CA 覆写为 0，另 10 行 Judge 原本已给 0。
   对 125 个 Assistant 工件扫描也未发现内部路径、URL、SHA 或 catalog source 泄漏。因此这些行
   应进入算法、检索或遵循度分析，而不是再改 parser。
9. **收口阶段发现的三项审计集成缺陷已修复且未改写模型结果。** execution importer 已区分 source
   execution-root 累计成本与冻结 shard-local 成本；analyzer 已支持严格 external frozen shard；
   comparison contract 的 Judge identity 已补齐 `thinking_budget` 与两个 billable-token 上限。
   旧 v2 失败审计保持不动；v3 审计 125 行完整、`blockers=[]`。

### 6.2 当前解释，尚非独立因果证明

- 被拒绝的 S3 candidate 强化了一对一检测、重复标签分离和“不得增加 / 遗漏”约束。它提高了
  adherence，却降低 `J_project`；当前解释是 Body 变得过长、过硬，增加了生成负担，并可能违背
  用户希望合并同类商品的语义。这与 gate 方向一致，但还没有通过受控消融证明。
- 当前 S1 更接近一次 bootstrap Creator：输入对初始失败轨迹和 rationale 的覆盖仍弱于论文的
  failure-driven creator。它能形成真实闭环，但不能据此声称等价复现论文 S1。
- v13 的 no-op 方差提示 Judge temperature、单样本输出和 Assistant 随机性都可能影响 gate；各自
  方差占比尚未被独立估计。

## 7. 下一版算法空间（本版不执行）

1. **S1 改为真正 failure-driven。** 把 response、工具轨迹、Judge 维度失败和归因 rationale 作为
   冻结输入，区分“缺 Skill”“Skill 描述错误”“Body 执行错误”，再创建或局部修改 Skill。
2. **S2 使用 confusion-pair 驱动的最小 Description 修改。** 优先处理 exact-match ↔ encyclopedia、
   exact-match ↔ multi-product 等已观测混淆对；用 macro-F1、质量和 hard error 的 Pareto gate，
   避免为了修一类路由破坏其他能力。
3. **S3 使用多样本 tier aggregation 和稀疏 Body patch。** 每 Skill 先按 poor-rate / 置信区间或
   bootstrap 聚合；将确定性规则缺口与 Judge 语义缺口分开；限制修改 span、Body 长度和新增约束数，
   并保护已经通过的正样本。
4. **把用户语义放入质量 gate。** 除 route、adherence 和 hard error 外，单列 CQ、CCC、分组意图、
   输出完整性与多商品回答长度，避免 adherence 单指标 Goodhart。
5. **先估计 no-op 噪声再决定 gate margin。** 对相同 Bank / route 做少量预注册重复，冻结 Judge
   rubric，降低或固定 Judge temperature，并以 no-op 方差设定最小可接受增益；不得重复采样挑最好值。
6. **继续保持公开 evidence DTO 与紧凑多商品输出契约。** 对模型可见 evidence / card 使用统一、
   脱敏、可引用的投影，并给多商品任务明确的紧凑结果结构和回答长度预算；这是程序与接口约束，
   不应通过生成更多 Bank candidate 来掩盖。

这些项目记录为下一版方向；本版不再修改 rubric、标签、evaluation split 或生成新的 S1/S2/S3
候选来追逐更好分数。

## 8. v16 clean composite 结果

### 8.1 完整性与主指标

- 运行范围：optimization25 的 25 query × 5 configs，125/125 Final；六个 capability 均覆盖。
- 物理来源：NoSkill 为 v15 的完整冻结 shard；Static / S1 / S1+S2 / Full 为 v16 fresh execution。
- `evaluation175=0/175`、`all200=25/200`；完整 200×5 尚未执行。
- launch state 仍为 `running`，因为 launch plan 包含 40 个 shard；已授权的 batch 1 五个逻辑 shard
  均为 completed，这不是当前批次残留 worker 或缺行。

| Config | Mean J | route acc. | route macro-F1 | adherence | hard error | Judge anomaly | card violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoSkill | 73.34 | N/A | N/A | N/A | 0 | 0 | 5 |
| LLMStaticSkill | 73.04 | 0.88 | 0.8913 | 0.6500 | 0 | 0 | 4 |
| S1 | 76.14 | 0.88 | 0.8913 | 0.6183 | 0 | 1 | 4 |
| S1+S2 | 77.92 | 0.92 | 0.9277 | 0.6583 | 0 | 0 | 3 |
| Full | 76.64 | 0.92 | 0.9277 | 0.6567 | 0 | 0 | 3 |

相对 Static，S1 / S1+S2 / Full 的配对均值分别为 `+3.10 / +4.88 / +3.60`；相对 NoSkill
分别为 `+2.80 / +4.58 / +3.30`。所有 25-query 区间都跨 0，不能作显著性或泛化声明。S1+S2
相对 Static 的 capability 变化主要来自 style recommendation `+15.0`、document reading
`+13.125` 和 recipe guidance `+3.333`；exact match `-1.6`、multi search `0`，encyclopedia
`+1.875`。其中 document reading 差异受到 S1 `dm-020` evaluator anomaly 强烈影响，不能全部归因
于 Skill。

### 8.2 重试、共同路由与 no-op

- NoSkill：1 次 Judge initial empty，第二次成功；Static：2 次 initial invalid，均第二次成功；
  S1：1 次 initial empty，第二次仍 schema invalid，终态 `parse_error/J=0`；S1+S2：0 次；Full：
  2 次 initial invalid，均第二次成功。
- 所有 route 均产生合法选择，没有 terminal route contract failure；S1+S2 / Full 的 25 个共同
  route artifact 全部 identity matched，0 mismatch。
- S1+S2 / Full 的 Bank SHA 均为 `9502ef4b…`；`Full - S1+S2=-1.28`，W/T/L=`8/8/9`，
  CI95=`[-6.8701, 4.0005]`，`treatment_attribution_allowed=false`。

### 8.3 成本与审计

- v16 新运行四个 skilled shard：402 次模型调用，settled `CNY 13.706014250000`，unresolved
  reserve `0`。冻结 NoSkill shard 原始局部成本 `CNY 2.959384900000`、86 次调用；本批五配置
  对应的已观测成本合计 `CNY 16.665399150000`、488 次调用。
- 当前 phase carry-in 为 `CNY 11.723577450000`，因此账本累计 accountable
  `CNY 25.429591700000`，在 cap `CNY 41.867467850000` 下剩余 `CNY 16.437876150000`。
  unresolved 已为 0；旧 v15 reserve 没有被重新授权。
- v3 batch audit SHA-256 为 `27a0cf22…`：25 query、125 rows、paired-input violation 0、
  `blockers=[]`、`status=review_required`。review 仅来自 1 条 non-scored Judge 和 19 条 card
  violation；这两类均已在上文分类。
- v3 analysis SHA-256 为 `909612fd…`；审计在 row projection 前后 byte-for-value 一致，treatment
  runtime 独立重放通过，分析阶段 `model_calls_performed=0`。

按首批观测成本线性估计，剩余 7 个 25×5 batch 约需 `CNY 116.65779405`；相对当前剩余授权的
点估计缺口为 `CNY 100.21991790`，尚未包含安全余量。因此当前不得启动 evaluation175；需要新的
预算授权和新的 execution scope，不能把 25-query optimization 结果包装成 200-query 结论。

## 9. 仓库证据入口

- 协议与指标：`docs/evaluation-protocol.md`、`docs/reproduction-contract.md`
- 论文：[SkillChain 原论文（arXiv）](https://arxiv.org/abs/2606.12984)
- route 执行与有界重试：`scripts/run_portfolio_shard.py`
- Final Judge 严格解析与有界重试：`src/skillchain/evaluation/final_runtime.py`
- shard finalization：`scripts/finalize_portfolio_shard.py`
- batch 审计：`scripts/audit_portfolio_batch.py`
- treatment gate 与字段边界：`src/skillchain/evaluation/portfolio_treatments.py`
- treatment lineage loader：`src/skillchain/evaluation/portfolio_treatment_io.py`
- treatment runtime 独立重放：`scripts/finalize_portfolio_treatment_runtime.py`
- 聚焦回归：`tests/evaluation/test_final_runtime.py`、
  `tests/evaluation/test_run_portfolio_matrix.py`、
  `tests/evaluation/test_portfolio_batch_audit.py`、
  `tests/evaluation/test_prepare_portfolio_evolution_inputs.py`
- 历史诊断根目录（只读归因，不改写原始结果）：
  `runs/portfolio/portfolio-dev-mini-200x5-execution-v13/`、
  `runs/portfolio/portfolio-dev-mini-200x5-execution-v14/`
- 冻结 NoSkill 来源：`runs/portfolio/portfolio-dev-mini-200x5-execution-v15/`
- clean skilled execution：`runs/portfolio/portfolio-dev-mini-200x5-execution-v16/`
- v3 batch audit：
  `runs/portfolio/portfolio-dev-mini-200x5-audits-v3/dev-mini-001-r3-integrity-audit.json`
- v3 analysis：`runs/portfolio/portfolio-dev-mini-200x5-analysis-v3/`
