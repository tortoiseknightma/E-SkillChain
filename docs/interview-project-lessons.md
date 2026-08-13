# 项目踩坑与解决记录（面试复盘版）

这份文档持续沉淀 SkillChain 复现与后续自进化实验中，真正体现工程判断的“困难—诊断—取舍—解决—验证”经历。它不是开发流水账，也不为了面试效果虚构困难；只有能够被代码、测试、实验产物或评审记录支持的内容才进入正式条目。

## 怎么读、怎么维护

- `已验证`：解决方案已经由测试或可审计产物验证，可以作为面试中的事实陈述。
- `部分解决`：关键机制已经落地，但仍有明确的外部证据或生产化工作未完成。
- `待验证`：目前是合理判断或设计，不能在面试中表述为已经完成。
- 一次常规重命名、小修小补或单纯照方案编码，不单独记录。
- 新条目优先写清“为什么原先看似可行的做法其实不成立”，再写具体实现。

面试时优先使用每个条目的“30 秒回答”；被追问设计细节、取舍或验证方式时，再展开“2 分钟回答”。

## 条目索引

| 编号 | 主题 | 状态 | 能体现的能力 |
| --- | --- | --- | --- |
| 01 | 随机切分会让近重复商品跨集合泄漏 | 已验证 | 数据治理、评测设计、可复现性 |
| 02 | “对象声称自己可信”不等于运行时真的可信 | 部分解决 | 安全边界、系统设计、对抗性测试 |
| 03 | 个人无法专业编写 ManualSkill 时，如何保持基线诚实 | 部分解决 | 实验设计、研究复现、边界意识 |
| 04 | policy hash 不等于网络策略真的生效 | 已验证 | 容器安全、供应链锁定、故障诊断 |
| 05 | 下载了数据，不代表它拥有任务需要的监督信号 | 部分解决 | 数据选型、问题建模、证据边界 |
| 06 | 不要让 LLM 计算哈希，也不要让协议冻结依赖尚未就绪的机器 runtime | 已验证 | 信任边界、协议设计、可复现性 |
| 07 | 一次性付费调用必须先落审计证据，再做本地解释 | 部分解决 | 故障恢复、可审计运行、预算治理 |
| 08 | authority-issued 不等于外部工件已获批准 | 部分解决 | 能力安全、供应链、可复现运行 |
| 09 | “本地 embedding”只有进入 authority 路径并适配中文才成立 | 部分解决 | 架构一致性、模型选型、供应链 |
| 10 | 数据集名称和二手描述不能替代官方任务/语言/许可元数据 | 部分解决 | 数据尽调、评测有效性、证据边界 |
| 11 | JSON mode 不等于结构化输出受 schema 约束 | 部分解决 | LLM 工程、实验治理、负结果处理 |
| 12 | Schema、TaskSpec 与运行时工具图必须组成同一可执行契约 | 部分解决 | 契约设计、根因归因、端到端可执行性 |
| 13 | 模型可访问不等于 provider 级可审计；多模态支持也不等于现有图片传输可用 | 部分解决 | 模型治理、证据分级、适配器设计 |
| 14 | “只允许一次”的模型调用需要预先存在的持久终态 | 代码/审计/候选冻结已收口，待 owner 确认与调用 | 并发控制、故障原子性、事务性审计、信任边界 |
| 15 | 隐藏的 validator 词汇约束会把语义正确的 LLM 输出变成系统性 bad case | 已验证 | 契约设计、LLM 可靠性、负结果归因 |
| 16 | 断点续传不只是追加字节：必须锁住远端身份和授权边界 | 已验证 | 大数据工程、故障恢复、安全设计 |
| 17 | 数万张小图的“断点续传”应以已验证对象为单位 | 部分解决 | 对象级恢复、数据处置、审计 |
| 18 | 浏览器下载完成后仍要区分“官方包”“导出副本”和无关隐私文件 | 部分解决 | 数据边界、隐私治理、离线验证 |
| 19 | 缺失的真实语料不能靠未标记的 Mock 悄悄替代 | 部分解决 | 外部有效性、合成数据治理、降级设计 |
| 20 | 全量测试失败不一定是功能回归：先把失败按共同系统调用归因 | 已验证 | 故障归因、跨平台测试、原子发布 |
| 21 | 数据策略变了，授权 policy 也必须作为版本化依赖一起迁移 | 已验证 | 数据治理、授权边界、不可变审计 |
| 22 | 真实 source lock 不能替许可证，证据准备者也不能冒充 owner | 部分解决 | 数据供应链、许可治理、PII、信任边界 |
| 23 | 两份都“正确”的 source lock 仍可能覆盖不同字节集合 | 部分解决 | 数据供应链、纵深校验、TOCTOU、诚实阻断 |
| 24 | 历史 runtime 在新工作树重放失败时，不能为测试变绿而改写证据 | 部分解决 | 可复现性、故障归因、不可变审计、环境治理 |
| 25 | “审批过 FashionIQ”不等于任意 adapter 输出都自动合规 | 部分解决 | 授权执行、数据闭包、TOCTOU、防泄漏 |
| 26 | “用户确认当前 Codex 模型”不等于可复核的 Codex CLI Mock 生成会话 | 部分解决 | LLM 数据治理、执行面证据、输入闭包、可复现性 |
| 27 | 冗余运行状态也属于 adapter 的输入信任边界 | 已验证并修复 | 最小信任根、fail-closed 性能、真实数据验证、TOCTOU |
| 28 | 小型 fixture 的标识符和文件类型不能代表官方全集 | 已验证并修复 | Schema 推断、真实数据契约、确定性抽样、人工审核边界 |
| 29 | “原始归档成员”不等于“原始分辨率图片” | 已验证并修复 | 证据语义、人工审核、最小化采集、哈希绑定 |
| 30 | 次级问法标签不能替换论文的顶层实验意图 | 已验证并修复 | 论文复现、契约治理、本体建模、集成测试 |
| 31 | 人工批准数量不等于防泄漏后的可用容量 | 已验证；35-pair 容量子门已关闭，formal C2 仍未关闭 | 图建模、数据泄漏、确定性优化、诚实停止 |
| 32 | 前向契约升级不能改写冻结历史 | 已验证并修复 | 版本治理、兼容迁移、可复现性、回归设计 |
| 33 | Stage 1 要保留完整交互，但不能偷看后阶段反馈 | 输入边界已验证，真实运行待完成 | 论文复现、信息隔离、不可变输入、实验公平性 |
| 34 | 全局 source-lock plan 与单分片 receipt 会把局部 exact-scope 修复放大成十源重审 | 已验证阻断、修复待实现 | 数据供应链、契约演化、变更隔离、授权边界 |
| 35 | 不可变 canonical bundle 不能在人工审核后原地追加文件 | 人工审核路径已验证并修复，Bank finalizer 待实现 | 不可变审计、生命周期设计、信任根、可复现性 |
| 36 | 数量够、排序固定、带 self-hash 仍不等于 200-query 计划可复现 | 已验证并修复；首批图像权限仍待确认 | 数据去重、canonical 序列化、计划约束、状态机 |
| 37 | “写入拒绝理由”与“发布 rejected 目录”之间也存在可恢复的部分提交 | 已验证并修复 | 状态机、原子发布、Windows 文件锁、幂等恢复 |
| 38 | 审阅辅助分层不能反向升级成偏离论文目标的硬契约 | 已验证并修复；r3 已按非强制分层正式接受 | 论文复现、数据契约、人工审核、最小权威 |
| 39 | 把“全通过后的三步操作”合并，不能把两个人工接受边界也合并 | 已验证并落地；后继批次仍待人工审阅 | 工作流设计、状态机、审计边界、故障恢复 |
| 40 | 视觉 Feedback 模型迁移：从工程解阻到异源评价恢复 | v5 迁移与视觉链路已验证；production 五配置评价待运行 | 多模态工程、实验设计、契约迁移、诚实披露 |
| 41 | 1×5 smoke 清单不能直接放大成 200×5 正式运行 | NoSkill v6 已封存；v10 首批 1×5 已重跑，剩余 35 shard 未授权 | 批处理调度、实验公平性、成本治理、故障恢复 |
| 42 | JSON action envelope 不是工具协议：不能把 Runner 失败误当成 NoSkill 能力 | 已验证、修复并以冻结 v6 重跑 | 原生工具调用、基线公平性、错误归因、Evaluator 版本化 |
| 43 | 单独重跑 Full 会把路由随机性和服务故障误归因给 Body Refiner | 已修复并完成首批 1×5 重跑；Judge parse noise 待分类 | 因果归因、共享实验变量、故障熔断、预算治理 |
| 44 | treatment 名称、Bank 文件和哈希齐全，仍不能证明 Creator/Optimizer/Refiner 真正执行 | 已修复并以真实链完成 clean 25×5；evaluation175 待预算 | Treatment provenance、阶段边界、接受/回滚、实验归因 |
| 45 | 调用方 timeout 不等于 materialize worker 已停止 | 已验证并修复 | 并发恢复、原子写入、故障隔离 |
| 46 | 同一 query_id 只锁成员，不授予旧正文复用权 | 已验证并落地 | 数据 lineage、反泄漏、语料重生成、证据边界 |
| 47 | accepted 必须晚于 durable ledger；编码错误应在此前 fail-closed | 已验证并落地 | 事务顺序、编码可靠性、可恢复执行、不可变审计 |
| 48 | 验证幂等不等于内存幂等：Core catalog 深验要复用证明并控制图像副本 | 已验证主修复；跨进程残留已进 backlog | 内存诊断、信任边界、多进程资源治理 |
| 49 | 跨品类搭配不能伪装成同类视觉相似：需要 query-independent evidence graph | 工具分支与真实 smoke 完成；Core 重锁/模型增益待验证 | 检索建模、Codex 辅助数据闭环、证据边界、实验隔离 |
| 50 | JSON Schema 能收紧结构契约，但不能让 reasoning-only 空正文变得不可能 | canary 已验证；phase60/全量未执行 | LLM 结构化输出、故障恢复、不可变审计、成本治理 |
| 51 | Sparse patch 不等于因果隔离：需要 capability-local paired screen 与 byte-exact 回滚 | 已验证；Qwen3.7 S1 两批十轮负结果已冻结，未进入 S2 | 算法实验、因果归因、负迁移防护、诚实停止 |

---

## 01. 随机切分会让近重复商品跨集合泄漏

**状态：已验证**

### 一句话问题

如果按样本行随机切分，同一商品或近重复图片可能同时进入训练、验证和测试集，使指标看起来提升，实际只是模型见过了高度相似的对象。

### 背景与影响

项目数据来自多个公开来源，并会经过清洗和 LLM 合成。同一商品可能有多张图、多个来源记录或被不同合成查询引用。传统的随机 split 只保证“行不同”，不能保证“商品实体或近重复组不同”。这会直接破坏后续 Skill 演化增益的可信度，是评测有效性的高风险问题。

### 观察到的证据

- 原始 schema 缺少足以稳定表达商品组、来源和派生关系的字段。
- 仅按 query/sample ID 切分，无法证明同一商品、同源图片或派生样本没有跨集合。
- 防泄漏检查需要在构建期和正式评测入口重复验证，而不能只依赖一次离线脚本。

### 根因

数据集的独立性单位选错了：真正需要隔离的是“商品/近重复组及其派生链”，而不是单条 JSONL 记录。

### 考虑过的方案与取舍

1. **普通随机切分**：实现最简单，但无法防止实体级泄漏，淘汰。
2. **按来源切分**：能降低部分同源泄漏，但相同商品仍可能出现在不同来源，覆盖不完整。
3. **group-aware split**：用稳定 group 标识约束同组样本只进入一个集合；实现和验证成本更高，但符合实际独立性要求。
4. **再叠加图像哈希与派生链检查**：可以覆盖 group 标注错误或近重复图片，是正式评测需要的纵深防线。

### 最终方案

升级到 schema v2，显式保存稳定的分组、来源和派生关系；切分时以 group 为原子单位，并生成冻结的 split manifest。正式评测入口不只读取切分结果，还重新校验输入摘要、组间零重叠和相关绑定，避免切分文件在后续阶段被替换或静默修改。

### 如何验证

- 单元测试覆盖同组样本不得跨 split、顺序变化不影响确定性结果，以及非法 manifest 必须失败。
- 正式评测输入通过类型化验证对象进入，原始对象混用、摘要不一致和 TOCTOU 修改均应 fail closed。
- 当前相关全量测试已通过；最终对真实数据的零重叠证明仍应保留为独立评测产物。

dev/opt 的 198 条 Style 还做了全量只读复核：81 条进入 cross、117 条保留 same-category，新增命中均为真实跨品类表达；81 条的 runtime 与 ceiling 候选及顺序差异为 0，其中 34 条有 exact graph evidence、47 条无覆盖。公开 DTO 抽查没有泄漏内部 ID、路径、hash 或 edge identity，也没有把内部 graph/source 名称或规则分数公开成“相关度”。

### 剩余限制

group-aware split 的效果依赖 group 构建质量。真实数据仍需结合商品标识、图片 SHA/pHash、来源关系和人工抽查，不能把一个分组字段当成绝对真相。

### 30 秒回答

我最早发现按行随机切分会让同一商品或近重复图片跨训练和测试集，导致 Skill 演化的提升其实来自数据泄漏。我把 schema 升级为能表达商品组和派生关系的 v2，以 group 为单位确定性切分，并冻结 manifest；在正式评测入口又做摘要和零重叠复验。这样不仅防住显式重复，也能通过图片哈希和派生链继续发现隐性泄漏。

### 2 分钟回答

这个问题难点在于，电商多模态数据的独立单位不是一行样本。一个商品会有多图、多来源记录，LLM 还可能基于同一对象生成多个 query。如果只随机分行，即使 sample ID 完全不同，模型仍可能在训练时见过测试商品。我的处理分三层：第一层升级 schema，保留稳定 group、来源和派生链；第二层以 group 为原子做确定性 split，并冻结输入和输出摘要；第三层在正式评测入口重新验证 manifest、组间零重叠和文件绑定，防止后续替换或 TOCTOU。取舍是数据准备更复杂，也可能减少可用样本，但换来的是可解释、可复验的增益。剩余风险是分组本身可能漏标，所以真实数据还要叠加 SHA/pHash、来源映射和人工抽查。

### 证据入口

- `docs/reproduction-contract.md`
- `docs/evaluation-protocol.md`
- schema、split、正式评测输入及防泄漏相关测试

---

## 02. “对象声称自己可信”不等于运行时真的可信

**状态：已验证**

### 一句话问题

早期设计允许工具或后端通过可写字段报告“我是正式实现”，但这种自报身份可以被伪造，无法作为正式实验的可信依据。

### 背景与影响

论文复现需要区分诊断性 stub/mock 与真正调用检测、OCR、检索或远程模型的实现。如果正式评测只检查一个布尔字段或普通属性，测试替身甚至被 monkeypatch 的对象都可能冒充生产实现，最终生成看似完整但不具备研究效力的报告。

### 根因

把“被验证对象提供的声明”误当成“验证者掌握的证据”，混淆了数据平面与信任平面。

### 考虑过的方案与取舍

1. **布尔标记或类名白名单**：简单，但字段、继承和 monkeypatch 都容易绕过。
2. **只在 CLI 层检查配置**：能挡住部分误用，但库调用仍可绕过，而且不能证明实际请求路径。
3. **由运行时验证器签发不可伪造的 authority/receipt**：实现更复杂，需要绑定对象状态和调用前后检查，但信任边界清晰。

### 最终方案

将 diagnostic 与 formal registry 分开；正式能力不再由对象的可写字段决定，而由受控验证过程签发并保存不透明 authority。对网络会话、模型/引擎状态、索引构建与查询绑定进行更深检查，并补充 monkeypatch、自我提权和状态替换的对抗性测试。无法满足条件时直接降级为 diagnostic 或拒绝正式运行。

### 如何验证

- 测试覆盖伪造 formal 标记、替换类方法、注入模型/引擎、修改网络 Session 配置等路径。
- 正式 registry 会重新检查结构、服务和绑定状态，而不只相信构造时声明。
- 当前工具层的本地信任机制与 production Assistant 的 runner-owned provenance 已通过机制测试；真实五配置模型运行和外部回执尚未产生，因此仍不能把项目标记为实验 GO。

### 剩余限制

diagnostic backend 仍可自报模型和调用信息，但只能走 `backend-reported-v1` 入口；production 入口只接受 exact Runner 并逐行绑定模型与工具回执。Phase4 仍因真实模型/Judge 校准和外部运行证据缺失而保持 `formal_eligible=false`。

### 30 秒回答

我遇到过一个容易被忽略的可信性问题：工具对象或 backend 用可写字段自报“formal”，测试替身也能伪装成正式实现。我把诊断和正式 registry 分离，正式权限改成由验证器签发的不透明 authority；Assistant production 入口再收回模型调用、usage、时延和 tool trace 的采集权，逐行生成 Runner receipt。现在本地工具链不能靠自我声明提权，diagnostic backend 也不能进入 runner-owned 结果。

### 2 分钟回答

最初我们关注的是“接口能不能调用”，后来发现正式复现还要回答“这次调用到底是不是那个真实实现”。如果让对象自己提供 `formal=True`，攻击者或误配置都能改字段，mock 也会进入正式报告。本质上是把被验证方的声明当成验证证据。我重新设计了信任边界：diagnostic 和 formal registry 使用不同类型；formal authority 只能由受控校验流程生成，并在外部弱引用状态中维护；登记和调用时还会检查类方法、服务、模型或引擎状态、网络 Session 以及构建/查询绑定。Assistant 侧进一步由 production Runner 独占统一模型入口和正式 registry，backend 只能提供 diagnostic 自报路径；每个结果保存模型 request/usage、Runner 总时延和实际 tool trace receipt。我还专门写对抗性测试尝试 monkeypatch、任意 backend 注入和协调重哈希。代价是实现更严格、适配新后端更麻烦，但它能防止“代码跑通”等同于“实验有效”。真实五配置和 Judge 校准仍未完成，所以项目继续保持 NO-GO。

### 证据入口

- `docs/go-no-go/2026-07-20-core-no-go.md`
- formal/diagnostic registry、authority、网络与后端对抗性测试

---

## 03. 个人无法专业编写 ManualSkill 时，如何保持基线诚实

**状态：离线前瞻机制已修，真实 C1 未验证**

### 一句话问题

论文中的 ManualSkill 需要领域专家编写，而个人复现者不具备相同专业条件；如果自己写或让 LLM 代写却仍称为 ManualSkill，会让基线定义失真。

### 背景与影响

该项目既要适合个人完成，也要能在求职面试和复现实验中经得起追问。ManualSkill 不是普通提示词：作者身份和知识来源本身就是实验条件。缺少专家时强行复刻标签，会把资源限制掩盖成方法结果。

### 根因

“复现相同的基线名称”与“复现相同的基线生成条件”并不是一回事。个人可调用强 LLM，但 LLM 生成内容不等价于独立领域专家手工编写。

### 考虑过的方案与取舍

1. **由项目作者自行编写 ManualSkill**：成本低，但不具备独立专家条件，不能诚实对标论文。
2. **让 LLM 生成后命名为 ManualSkill**：内容可能更好，但基线标签错误，会污染实验解释。
3. **取消全部静态基线**：最保守，但失去有价值的对照。
4. **重新命名并记录来源的可审计静态基线**：既保留对照，也不伪装作者身份。

### 最终方案

主静态基线采用 `LLMStaticSkill`：通过冻结的 AuthoringPacket 约束模型、参数、公开来源、请求与响应绑定；规则化的 `SpecBaseline` 只作为 diagnostic 对照。如果未来找到合格且独立的领域专家，可通过结构化表单采集内容，再编译为 Skill，并命名为 `ExternalExpertManual`。只要生成式 AI 参与了内容创作，就应命名为 `ExpertCuratedLLMSkill`，不能继续称为 ManualSkill。

### 如何验证

- 类型和产物元数据明确记录 authoring method、模型参数、来源摘要及 formal eligibility。
- 严格的 LLM 响应与公开来源字节绑定，避免事后替换作者输入。
- schema v3 的 v3/Flash 正式调用有 40 个 schema 错误；v4 forced-function 调用又因 stringified `drafts` 和内部 16 项错误被拒绝。两份负结果均保持不可变，没有可审查 Bank，因此 C1 仍未关闭。

### 剩余限制

缺少真正独立的专家就无法声称复现了论文原定义的 ManualSkill。Codex v5 已取得合格
LLMStatic draft，owner 也已在 5 分钟内完成六项 checklist 并接受 unchanged draft；但
最终 Bank compile 仍须绑定 C3 authority-issued tool registry runtime。若获得专家，还需
保存其资格、独立性和评审记录，且不能把现有 LLMStatic acceptance 改名为 ManualSkill。

### 30 秒回答

我没有领域专家资源，直接自己写 ManualSkill 或让 LLM 写完仍叫 ManualSkill，都会让基线失真。我的解决办法是保留实验对照但诚实重命名：主基线叫 LLMStaticSkill，用冻结的 AuthoringPacket 绑定模型、参数、来源和响应；规则基线只用于诊断。未来若有独立专家就单列 ExternalExpertManual，AI 参与则叫 ExpertCuratedLLMSkill。这样资源受限，但结论边界是可信的。

### 2 分钟回答

这个困难不是单纯“不会写提示词”，而是实验条件不可得。论文的 ManualSkill 隐含了领域专家身份和人工知识来源；个人复现者用强 LLM 可以产出内容，却不能把它包装成相同条件。我比较了自行编写、LLM 冒充、完全取消对照和重新定义可审计基线几种方案，最后选择诚实分层：LLMStaticSkill 是主要静态对照，AuthoringPacket 冻结模型、参数、公开来源、请求和响应；SpecBaseline 只是规则化诊断；只有真正独立专家参与时才增加 ExternalExpertManual，而且 AI 参与就另命名为 ExpertCuratedLLMSkill。这个取舍牺牲了与论文某一标签的表面一致性，但保住了实验可解释性，也更符合个人开源复现的现实。v3 与 v4 真实 run 均证明隔离/审计链路有效，却都产生 schema 负结果。因此我会说协议与执行边界已验证、当前 LLMStatic 基线不可用，而不是夸大成完整复现。

### 证据入口

- `docs/plans/2026-07-20-p0-p1-closure.md`
- `docs/go-no-go/2026-07-20-core-no-go.md`
- AuthoringPacket、LLMStaticSkill、SpecBaseline 及相关测试

---

## 04. policy hash 不等于网络策略真的生效

**状态：已验证**

### 一句话问题

最初的 formal profile 只记录一个 `network_policy_sha256`，却没有证明运行时网络、proxy 和镜像仍与该摘要对应；这会让“有策略文件”被误当成“策略已执行”。

### 背景与影响

LLMStaticSkill 必须只读取冻结的 AuthoringPacket，同时仍要访问指定模型 provider。如果直接给 author 容器普通外网，它可以访问任意站点；如果只用 Docker internal network，它又无法调用 provider。更隐蔽的问题是：即使 manifest 保存了 policy hash，容器网络或 proxy 在运行前被替换，hash 本身也不会阻止流量越界。这个边界一旦失真，静态基线可能读取未授权信息，RQ1b 的公平比较就失效。

### 观察到的证据

- 原 profile 只有 network 名称和 policy 摘要，formal runner 没有检查 Docker network ID、`Internal` 标记、已连接 peer 或 proxy 的实际镜像/启动参数。
- 第一次真实镜像自检在 `python -I` 下报 `ModuleNotFoundError`：源码仅靠 `PYTHONPATH` 可见，而 isolated mode 会忽略它。随后又暴露 `tools/__init__.py` 的 eager imports，使 authoring worker 实际需要图像与索引依赖。
- 第一版生成的 profile 被仓库 loader 拒绝为非 canonical JSON。根因是部署脚本漏掉了仓库规范要求的末尾 LF，导致“脚本自算摘要”与正式 verifier 使用的字节算法不同。
- 最终 deployment receipt 记录四项退出码为 0：allowlisted `dashscope.aliyuncs.com:443` CONNECT 成功、`example.com:443` 返回 403、author 容器直接连接公网 IP 失败、外部 DNS 解析失败。

### 根因

设计把三件不同的事混在了一起：配置意图、运行时强制执行和事后审计。摘要只能绑定某些字节，不能创建防火墙；Docker network 名称也不是稳定身份；构建成功不代表 `python -I` 下的正式入口能导入。另一个根因是重复实现 canonical JSON 规则，而没有一开始逐字节对齐仓库 serializer。

### 考虑过的方案与取舍

1. **只保留 policy 文件和摘要：** 简单，但只能审计意图，不能阻止 drift 或绕过。
2. **author 容器直接接普通 bridge，再在 Python 中检查 URL：** SDK、子进程或直接 socket 都可绕过，不能称为机械隔离。
3. **完全断网并由宿主代发请求：** 网络边界最简单，但宿主代理将看到凭据和消息，需要新增一套受控 RPC、认证与收据协议。
4. **internal network + 唯一双宿主 CONNECT proxy：** author 没有默认外网路由，proxy 只允许一个 hostname/443；实现复杂度适中，且 HTTPS 隧道让 proxy 不持有 API key、不读取请求正文。代价是 hostname 级 allowlist 无法约束 provider 域名内的具体 URL path，并且当前证据是单主机 Docker Desktop 证据。

### 最终方案

使用同一个不可变 image digest 运行 author worker 和 CONNECT proxy。author job 只加入 `--internal` network；proxy 同时加入 internal 与专用 egress bridge，精确允许 `dashscope.aliyuncs.com:443`，拒绝 IP literal、其他 hostname 和其他端口。proxy 使用只读 rootfs、drop-all capabilities、no-new-privileges、资源上限，并明确禁止注入 provider credential。

构建侧用根目录 `.dockerignore` 做 allowlist，只把 `src/`、Dockerfile 和 hash-locked requirements 发送给 build daemon；`.env`、数据、运行产物和 Git 元数据不进入 build context。

formal sandbox profile 升级到 schema v2，绑定 Docker CLI 绝对路径及二进制摘要、本地不可变 image ID、network ID、proxy image、双网名称、proxy URL 和 policy 文件摘要。每次正式调用前，runner 重新 inspect Docker：内部网络 ID/`Internal` 必须一致，network 中只能有锁定 proxy，proxy 必须正在运行、使用锁定镜像和固定命令、恰好连接两张指定网络，安全参数和 allowlist 环境也必须一致。任何 drift 都在注入 API key 和启动 author job 前 fail closed。

### 如何验证

- 镜像基础层使用 repository digest，Python requirements 使用 `--require-hashes`，最终运行使用本地不可变 `sha256:<image-id>`。
- 镜像内用 `python -I` 导入正式 `authoring_worker`，不依赖 `PYTHONPATH`。
- 单测覆盖 CONNECT 解析中的 HTTP method/version、IP literal、非法端口、尾点 hostname 等绕过形态。
- 真实网络探针同时验证 allow、deny 和 direct bypass；profile 再经正式 loader 按外部文件摘要重载，并通过 runner 的实时 Docker inspection。
- 历史 Qwen v4 调用绑定的 LLMStatic authoring sandbox 锁位于 `deploy/authoring/locks-v4/`；runtime 为 `formal-v3`，镜像为 `sha256:6e86773a4039cb0232f361547aa80ea8312da91c6947216f7770dfe4825d0f6e`，lock manifest 文件 SHA-256 为 `c9afe6d94e9b54f81cc08c18fbb69263a1d457799dbd9fdc4b9c911c5dc2bfb3`。Runner 还会证明 CLI profile 正是 manifest 中冻结的文件，而不是调用方提供的另一套自洽 profile；这些是历史 Qwen v4 值，不属于另有独立 runtime/freeze 的 Codex v4。

### 剩余限制

这次只证明当前 Windows Docker Desktop 主机上的镜像与 provider-only egress 边界，尚未得到独立 Linux CI 回执。CONNECT proxy 只能限制 hostname/port，不能观察加密隧道内的 path；这是避免 proxy 接触 prompt/response 的有意取舍。v3 与 v4 付费 Flash request/response/isolation receipt 均已生成，但两份 draft 都不合格，所以 C1 与 core 仍是 NO-GO。临时 key 只在运行时注入，绝不能作为冻结工件保存。Docker 状态检查和启动 job 之间仍存在本机管理员可利用的 TOCTOU 窗口；当前威胁模型信任主机 operator，不声称抵抗本机 root/管理员。该 sandbox 锁也不能替代 C3 authority-issued tool registry runtime。

### 30 秒回答

我发现只在 manifest 里写 `network_policy_sha256` 并不能证明策略真的生效。于是我把 author 容器放进 Docker internal network，只保留一个双宿主 CONNECT proxy，并把 proxy 精确限制到模型 provider 的 443 端口。正式 runner 每次调用前都会 inspect 真实 network ID、peer 集合、proxy 镜像、命令和安全参数；我还跑了允许目标成功、非 allowlist 403、直接 IP 失败、外部 DNS 失败四类真实探针。部署过程中 `python -I` 导入失败和 canonical JSON 少一个换行也都被自检抓到，最后镜像、engine、policy、profile 和回执全部按外部摘要锁定。

### 2 分钟回答

原来的设计在 profile 里记录 network 名和 policy hash，看起来可审计，但本质只绑定了配置意图：同名 network 可以重建，proxy 可以换镜像，甚至 author 容器仍可能直接出网。我把强制执行拆成两层。第一层是网络拓扑：author job 只有 Docker internal network，没有默认公网路由；唯一能出网的是双宿主 proxy。第二层是应用 allowlist：proxy 只接受标准 HTTPS CONNECT，目标必须精确等于 `dashscope.aliyuncs.com:443`，IP literal、其他域名和端口都拒绝，而且 proxy 不拿 API key、看不到 TLS 内的正文。

为了让摘要对应真实状态，我把 sandbox profile 升到 v2，除了 image/engine hash 还绑定 network ID、proxy image 和双网身份。runner 在每次模型调用前执行 Docker inspect，核对 internal 标志、peer 集合、只读 rootfs、cap-drop、no-new-privileges、固定 command 和 allowlist env，drift 就 fail closed。真实部署还暴露了两个很有价值的问题：`python -I` 会忽略 `PYTHONPATH`，所以镜像必须把包放进系统 site-packages；部署脚本生成的 JSON 少了仓库 canonical 规则要求的末尾换行，正式 loader 因而拒绝。修复后我用四类网络探针验证允许、拒绝、直连和 DNS 绕过，并把镜像、Docker binary、network、policy、profile、receipt 全部锁定。边界是这仍是单主机部署证据，不是作者模型质量或论文增益证据，所以项目继续 NO-GO。

### 证据入口

- `deploy/authoring/Dockerfile`、`requirements.lock`、`deploy.ps1`
- `deploy/authoring/locks-v4/network-policy.json`、`sandbox-profile.json`、`deployment-receipt.json`、`lock-manifest.json`
- `src/skillchain/runners/egress_proxy.py`
- `src/skillchain/static_authoring.py` 中的 `AuthoringSandboxProfile` 与 `FormalContainerAuthoringGateway`
- `tests/runners/test_egress_proxy.py`、`tests/test_static_authoring.py`

---

## 05. 下载了数据，不代表它拥有任务需要的监督信号

**状态：离线机械闭环完成，真实 C3 未通过**

### 一句话问题

早期方案按“手头有哪些公开数据”给 capability 分配来源，结果把类目、bbox、单图或语言标签误当成 SKU identity、风格偏好、字段级 OCR 和食谱事实等更强监督信号。

### 背景与影响

项目需要评价 Exact Match、Multi-Product、Style Recommendation、Visual Encyclopedia、Document Reading 和 Recipe Guidance。它们看起来都能从图像数据开始，但需要的 gold 不同：例如 Exact Match 要同一 SKU 的真实多视图，Multi-Product 要场景 bbox 与逐对象 SKU 身份，Style 要相对语言或兼容性偏好，Recipe 要把菜品实体连接到可引用食谱。若只因某个来源已经下载就让它承担这些任务，流水线即使可运行，指标也不再测量声称的 capability。

### 观察到的证据

- 2026-07-22 只读盘点发现，本地主工作副本 `raw/` 有 21,504 个文件、8,147,316,535 字节，但 `clean/` 和 `kb/` 为空，正式 selection/review/catalog manifest 也不存在。
- MUGE 有中文 query-image pair，却没有足以支持 MVP Exact Match 的稳定 SKU 多视图关系；MEP-3M 本地文件较多，但规模和标签更适合 core 长尾/hard negative。
- COCO 有 bbox，却没有 RPC 所提供的场景商品到 SKU reference 的端到端身份关系。
- ISIA Food-500 能监督菜品识别，但不能单独证明某份食谱的食材和步骤。
- 本地缺少 ABO、RPC、FashionIQ、WildReceipt、JDDC 2.0，说明原计划的主要监督缺口不能由旧缓存自动补齐。
- 原 planner 仍按五 intent 固定为 50/30/40/40/40，utility 内也没有 30 条 Document + 30 条 Recipe 的独立 quota；若只改文档，实际 200 条计划仍会违反新分配。

以上文件数、字节数与目录为空是本地盘点事实；各来源角色是基于其标签结构和 2026-07-22 数据来源修订做出的设计决定，真正进入 formal run 仍待 source lock、人工 review 与 artifact 验证。

### 根因

把“数据可获得性”当成“标签对目标任务的可识别性”。来源名称被当作意图代理，缺少一层明确的 capability→监督信号→来源角色契约，也就无法在 planner 或评测前阻止来源越权。

### 考虑过的方案与取舍

1. **继续复用已下载来源**：最快、无需新增 adapter，但会让 gold 语义失真，淘汰。
2. **删除旧数据后全量重下**：目录更整洁，但破坏可恢复缓存，而且 source selection 尚未完成，不必要也有风险。
3. **只在文档中改描述**：成本低，但后续代码或计划容易再次漂移。
4. **保留 raw cache，冻结机器可读 source portfolio**：需要维护新契约和测试，但能把主 gold、语言层、KB、candidate gallery 与 challenge 分开，并允许旧 adapter 继续服务 diagnostic/core。

### 最终方案

保留现有 raw cache，不把它升级为 `clean/` 或正式 catalog；新增 `ecommerce-mvp-source-portfolio-v1.json`，按可靠监督信号规定来源角色和禁用用途。MVP 的 200 条分配固定为 35/35/35/35/30/30；ABO、RPC、FashionIQ、WildReceipt 分别提供 Exact、Multi-Product、Style、Document 主监督；JDDC 2.0/MUGE 只提供语言形态；COCO、iNaturalist、Wikimedia Documents 进入 challenge；MEP-3M 等延后到 core。确定性 planner 同步改为六 capability quota，并把 utility 拆为 Document/Recipe 各 30 条、boundary 调整为约定的 40 条。canonical 计划、收口计划、NO-GO 和历史工具计划都链接到同一 portfolio。

### 如何验证

- `tests/test_data_source_portfolio.py` 验证 200 条总量与 capability 配额、四个关键主源、语言/challenge 禁用用途和 core 延后来源，并把 JSON 配额绑定到 planner 常量。
- `tests/synthesis/test_planning.py` 验证实际计划的六 capability 数量、8×25 批次和 40 条 boundary；formal catalog 测试验证缺 capability assignment 时 fail closed。
- 两份当前计划和 NO-GO 记录必须链接机器可读 portfolio，避免计划文本再次脱节。
- `docs/data-source-adjustment.md` 保留带日期的本地盘点和来源角色说明。
- 真正的数据闭环尚未验证：后续还要从固定来源重建、生成外部 expected digest、完成人工 review，并由 formal loader 重验。

### 剩余限制

source portfolio 目前是计划与验证契约，不是数据下载或质量证明。ABO/RPC/FashionIQ/WildReceipt/JDDC 2.0 的 adapter、实际字节、source lock 和 gold review 尚未完成；本地盘点只统计文件与大小，没有替大文件做内容哈希或标签质量抽样。license/PII/use gate 也仍然有效。

### 30 秒回答

我发现项目虽然下载了约 7.6 GiB 数据，但关键问题不是容量，而是标签能不能识别目标能力：COCO 的 bbox 不能评价逐商品 SKU 检索，MUGE 的图文对不能自动变成同 SKU 多视图，食品分类也不是食谱证据。我没有删除缓存，而是冻结了机器可读 source portfolio，把主 gold、语言层、KB 和 challenge 分开，并用测试锁定 200 条配额与禁用用途。这样“文件存在”不再等于“正式可用”。

### 2 分钟回答

最初的计划是按手头公开数据分 intent，看上去覆盖很全，但我逐项追问 gold 从哪里来后发现监督错位：Exact Match 需要同一商品的真实多视图，Multi-Product 需要 bbox 后还能回到 SKU，Style 需要相对语言或兼容性偏好，Document 需要字段级 OCR/KIE，Recipe 则要菜品实体到可引用食谱的映射。对本地目录审计后，虽然 raw cache 有 21,504 个文件、约 7.6 GiB，clean 和 KB 却为空，而且关键新主源都缺失。我的取舍是保留 raw cache，避免破坏性重排；同时建立机器可读 portfolio，明确 ABO/RPC/FashionIQ/WildReceipt 的主监督角色，JDDC/MUGE 只管语言，COCO/iNaturalist/Wikimedia Documents 只做 challenge，MEP-3M 等留到 core。再用测试锁定配额和 forbidden roles，并同步所有当前计划。现在设计漂移被控制住了，但我会明确说明数据 acquisition、人工 review 和 formal artifact 还没有完成，所以状态只是部分解决。

### 证据入口

- `docs/data-source-adjustment.md`
- `specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`
- `src/skillchain/synthesis/planning.py`
- `tests/test_data_source_portfolio.py`
- `tests/synthesis/test_planning.py`
- `docs/plans/2026-07-09-skillchain-reproduction.md`
- `docs/go-no-go/2026-07-20-core-no-go.md`

---

## 06. 不要让 LLM 计算哈希，也不要让协议冻结依赖尚未就绪的机器 runtime

**状态：已验证**

### 一句话问题

原 schema 要求作者模型输出输入摘要、draft 摘要和 bundle 摘要，同时把 AuthoringPacket 绑定到当前工具 runtime；前者不适合生成模型，后者让 C1 协议冻结反向依赖 C3 部署状态。

### 背景与影响

LLMStaticSkill 只有一次成功调用预算。若 prompt 要求模型准确计算 SHA-256，最可能的首个真实 bad case 不是能力差异，而是不可合理完成的序列计算，浪费唯一调用并污染基线。另一方面，作者只需要知道允许的 ToolSpec，不需要知道索引、OCR 或 detector 在某台机器上的 runtime hash；过早绑定会导致工具工件更新时 AuthoringPacket 也被迫变化，使 LLMStaticSkill 与 S1 的公共输入难以长期保持逐字节一致。

### 观察到的证据

- 旧 `AuthoringDraftBundle` 要求 `authoring_input_sha256`、每个 `draft_sha256` 和 `bundle_sha256` 全部由模型响应提供，Runner 直接严格解析。
- 旧 AuthoringInput 强制包含 `tool_registry_runtime_sha256`；而生产工具 runtime 属于后续 C3 工件，当前只有 ToolSpec 能真实冻结。
- schema v3 定向测试证明模型响应中没有任何 digest 字段，Runner 仍能生成并复验完整内部 bundle；冻结 packet 的 tool runtime 值为 `null`，正式 SpecBaseline 在拿到 authority-issued registry 后把 Bank 绑定到真实 tool runtime。

### 根因

设计混淆了三种身份：不可信内容的语义字段、可信边界生成的内容摘要、以及部署时执行工件的身份。把它们全部塞进模型输出和作者公共输入，看起来“绑定更完整”，实际上跨错了信任边界和生命周期。

### 考虑过的方案与取舍

1. 保留 prompt 要求模型算哈希：改动最少，但真实成功率不可接受，且哈希并不因此可信。
2. 让 Runner 忽略模型哈希：能运行，但保留了误导性字段，难以审计究竟哪些字节有权威性。
3. 冻结一个 synthetic runtime hash：可以立即生成 packet，却把诊断环境伪装成生产依赖。
4. 拆分 hash-free payload、可信 normalization、ToolSpec identity 和 compile-time runtime binding：需要 schema/编译器升级与容器重建，但边界最清晰。

### 最终方案

schema v3 的作者响应只包含 `schema_version=1` 与六个 capability payload，不包含任何输入或内容摘要。Runner 严格解析、检查 capability/rule/tool/source 覆盖，再确定性生成 draft 和 bundle SHA-256。AuthoringPacket 冻结 ToolSpec identity，并以 `deferred_until_bank_compile` 明示 tool runtime 尚未绑定；正式编译必须提供独立验证的 C3 registry runtime，否则 fail closed。历史 v3 freeze 曾分别生成默认 Flash 与备选 Plus packet并禁止自动 fallback；v4 deviation 因 v3 输出已存在而只授权同一 Flash primary，Plus 不再可用。

### 如何验证

- `test_model_returns_hash_free_payload_and_runner_adds_identities` 验证 raw response 无摘要、normalized bundle 有完整摘要。
- `test_frozen_primary_packet_has_no_public_material_and_defers_runtime` 证明真实冻结 packet 可用独立正式 registry 加载，且最终 Bank 绑定该 registry runtime。
- Qwen v3 freeze 测试逐字节重建并检查历史主/备角色、空 sources、预算和外部文件摘要；Qwen v4 freeze 测试则检查 primary-only、forced submission、deviation、claim 和 runtime bundle 绑定。
- LLMStatic authoring sandbox `formal-v3`/`locks-v4` 已用最终作者源码重新部署，四项真实 network probe 再次通过。

### 剩余限制

v3 已执行真实作者调用并证明 hash-free 设计不会丢失调用审计，但 payload 有 40 个 schema 错误；v4 forced-function 调用也已执行，framing 合格但 `drafts` 被二次序列化，内部仍有 16 项错误。它验证了接口边界，却没有得到可用 draft。作者调用本身不需要 C3 tool registry runtime，因为 packet 只公开 ToolSpec；只有合格 draft 的最终 Bank compile 必须等待 C3 真实工件与 authority-issued registry。官方价格未来可能变化，但本次 packet 使用锁定价格文件和固定保守汇率，变化只能通过新 freeze ID 处理。

### 30 秒回答

我在准备真实作者调用时发现，旧协议让 LLM 自己计算多个 SHA-256，这会把唯一一次调用浪费在模型不擅长且不可信的计算上；同时 AuthoringPacket 还依赖尚未完成的工具 runtime。我的修复是 schema v3：模型只输出无哈希结构 payload，Runner 负责规范化和摘要；packet 只冻结 ToolSpec，真实 tool runtime 到 Bank 编译时独立绑定。测试证明 raw response 不含摘要、冻结 packet 能在正式 registry 到位后绑定 Bank，authoring 容器也用新代码重新锁定。

### 2 分钟回答

这个问题本质上是信任边界与生命周期混淆。作者模型负责提出内容，不应该成为密码学身份的权威；AuthoringPacket 负责公平公共信息，也不应该绑定后续 C3 才能生成的机器工件。旧版却要求模型返回 input/draft/bundle 三层摘要，并把当前 registry runtime 写进 packet。这样即使模型偶然生成了 64 位字符串，它也没有可信意义；工具索引一更新，公共 authoring 输入还会跟着漂移。我比较了保留字段后忽略、用 synthetic runtime 占位和彻底分层，最终选择升级 schema/编译器：模型输出严格的 hash-free payload；受信 Runner 校验 capability、规则、工具和引用后生成 canonical bundle 与摘要；packet 冻结 ToolSpec，compile-time 必须由 authority-issued registry 提供真实 tool runtime。历史 v3 主/备 packet 验证了这种分层；在 v3 输出失败后，v4 只修提交接口且保持同一 Flash，不允许 Plus fallback。代价是需要重建容器和更新审计格式，但换来的是作者调用不再依赖 C3、C1/C3 不再循环依赖，以及更清楚的可复现证据链。

### 证据入口

- `src/skillchain/static_authoring.py`
- `scripts/freeze_authoring_packet.py`
- `specs/authoring/authoring-freeze-lock-v2.json`
- `specs/authoring/static-author-v3.txt`
- `specs/authoring/authoring-freeze-lock-v4.json`
- `specs/authoring/static-author-v4.txt`
- `tests/test_static_authoring.py`
- `tests/test_authoring_freeze.py`
- `deploy/authoring/locks-v4/`

---

## 07. 一次性付费调用必须先落审计证据，再做本地解释

**状态：部分解决**

### 一句话问题

正式 authoring 只有一次调用预算：既要在解释内容前持久化 provider 事实，也必须在核心调用层消费不可复用的 owner authority；只在 CLI 检查 candidate 或只让单个 Gateway one-shot，都挡不住直接 API/重新实例化造成的重复调用。

### 背景与影响

LLMStatic 基线要求固定 AuthoringPacket、单次调用、固定费用上限和完整 request/response/usage 证据。这里的关键不是“程序报错后再试一次”这么简单：如果供应商已经成功生成，盲目重试可能违反调用预算并产生第二笔费用；如果响应没有落盘，就无法审核内容，也不能把这次尝试算作 C1 证据。后续 v5 审查又发现，CLI 拒绝 `candidate_not_authorized` 也不等于核心 API 获得了 authority：只要 `invoke_llm_static`/formal gateway 不持有同一不可复用 capability，其他调用者就可能绕过 CLI。故障因此暴露了写路径、审计顺序和授权 enforcement 层三个问题。

### 观察到的证据

- `llm-static-primary-20260723-v1` 的 egress proxy 允许了到 `dashscope.aliyuncs.com:443` 的 CONNECT，说明请求至少到达了供应商网络边界。
- 父进程只得到 `AuthoringContractError: isolated authoring job returned failure`；旧实现丢弃 worker stderr，且没有发布 response、usage、request ID 或 failure receipt。
- 代码审计证明 `llm.chat()` 在返回成功 response 后无条件调用 `_log_usage()`；容器内 `USAGE_LOG` 位于安装包根目录，而正式 job 以 uid 65532 运行在只读 root filesystem 上。因而只要供应商成功返回，旧 worker 就必然在发布 `authoring-response.json` 前写盘失败。
- 没有证据能反向证明供应商一定成功或一定计费，因此该结果必须标为 unknown，而不是按成功或未调用处理。
- **已验证的修复前事实：** v5 candidate 最初只在 `scripts/run_formal_authoring.py` 被拒绝；核心 `load_verified_authoring_invocation_input()` → `invoke_llm_static()` 与 `FormalContainerAuthoringGateway.complete_once()` 不要求 owner-issued call capability，`max_successful_calls` 也没有执行侧消费点。重新实例化 Gateway 可以绕过对象级 one-shot。
- **已验证的离线修复：** `VerifiedAuthoringCallAuthorization` 只能从外部摘要锁定且 `status=frozen` 的 owner freeze 加载，并同时绑定 deviation/call budget、packet/input/output contract、model/variant、run/output/claim/receipt 和 sandbox runtime bundle；candidate 无法签发。核心 invoke 与 Gateway 双边要求 capability，claim 在任何 runtime/provider 动作前 atomic create-only，预加载第二 handle、重新 loader 或新 Gateway 都不能复用。
- **已验证的第二轮修复前事实：** capability 初版仍允许 `require_formal_eligibility=false` 的 `ControlledAuthoringGateway(UnifiedChatTransport())`，包装器也能间接调用同一真实 provider transport；这虽然只能产生 non-formal 产物，却仍会未经 owner 预算批准发生付费调用。另一方面，正式 CLI 对所有 v5 无条件拒绝，导致即使未来 owner 签发 authority，也没有受支持的 claim→provider→终态 receipt 路径。
- **已验证的第二轮离线修复：** 非正式 v5 现在只接受 exact、未篡改的 `OfflineAuthoringReplayGateway`。它只持有已物化且严格校验、与 canonical request 绑定的 `LLMResponse`，不持有或调用 transport；任意 Controlled transport/包装器都会在 `.complete()` 与 `llm.chat` 前失败。正式 CLI 则加载同一个 `VerifiedAuthoringCallAuthorization`，只读 core 创建的 claim，并在成功、response 已落盘后 draft 拒绝、claim 后结果未知三类终态写 create-only receipt。
- **已验证的路径边界：** output、attempt claim 与 invocation receipt 现在不仅必须不同，还必须两两禁止任一方向的 ancestor/descendant 关系；这避免把 claim 文件配置成 output 的祖先后，先消费预算再在发布阶段失败。
- **待验证事实：** 当前没有真实 v5 owner freeze/authorization，也没有 v5 provider call；新机制只证明未授权路径会 fail closed。

### 根因

系统混淆了三种责任：隔离 worker 负责与供应商通信，父 Runner 负责正式审计，owner authority 决定某次调用是否可发生。旧通用 LLM 客户端却让 worker 写全局 usage，父 Runner 又把“内容可解释成功”放在“供应商事实持久化”之前；v5 初版还把 authority 留在 CLI，误把进程入口当作完整信任边界。对象级 `_used` 只能约束一个实例，不能约束重新实例化、另一个进程或直接核心 API。对一次性付费调用，授权与审计都必须由所有正式路径无法绕过的核心 capability 承担。

### 考虑过的方案与取舍

1. **直接重试旧任务**：最快，但可能重复计费，也会再次触发同一确定性故障，淘汰。
2. **给容器挂载可写全局 `runs/`**：改动小，但扩大 worker 写权限，产生两个 usage 账本和新的污染面。
3. **只修 usage 写入，不保留失败工件**：能消除当前错误，却仍会在输出 schema 不合规或未来基础设施故障时丢失付费证据。
4. **父进程独占审计，先持久化调用结果再解释内容**：需要升级 gateway、Runner、receipt 和冻结镜像，但能把供应商事实、内容验收与最终 Bank compile 分层。
5. **只在 CLI 检查 freeze/candidate**：改动小，但库调用者和 direct Gateway 可绕过，且新实例会重置 one-shot，淘汰。
6. **核心 capability + create-only claim 双重消费**：loader 复验 owner freeze 的全部协议/预算/运行绑定，核心 invoke 与 Gateway 双边要求该 capability；内存锁阻止同 handle 重用，文件 claim 阻止新 loader/进程重用。实现更复杂，但才能把“一次”变成跨入口事实，采用。

### 最终方案

`UnifiedChatTransport` 调用通用客户端时显式关闭 worker-side usage 写入，由父 Runner 从规范化 `LLMResponse` 生成唯一 usage/cost receipt。正式 gateway 在子进程失败时携带 exit code、stdout、stderr、耗时和 command hash；Runner 原子发布 request 与 failure artifacts 后再抛错。供应商成功响应则先原子发布 request、response 和 isolation attestation，再做模型身份、预算和 draft schema/覆盖校验，因此“内容不合格”也不会抹去已发生的调用。v3 修复后的镜像与 freeze 证明了这条顺序；v4 又把预算语义从“成功次数”收紧为容器启动前原子消耗的 attempt claim，并用新 authoring sandbox namespace 重建，旧锁与失败尝试都保留为历史证据。

v5 进一步把“谁能消费 claim”收进核心 trust boundary。`VerifiedAuthoringCallAuthorization` 使用模块私有 verification token；loader 只接受外部文件 SHA 锁定、owner-approved、`status=frozen` 且明确 `authority_issued/provider_call_authorized/budget_authorized/runtime_lock_frozen=true` 的 freeze。它交叉绑定 deviation/call budget、AuthoringInput/packet/output contract、taxonomy/TaskSpec/registry、model/variant/run、create-only output/claim/receipt，以及 sandbox profile、deployment receipt、lock manifest 和 network policy；三条发布路径必须两两不同且双向不嵌套。runtime loader 还复验固定同目录文件集、manifest 回指/自哈希、四项成功探针，以及 profile↔policy↔receipt 的 engine/image/network/proxy/endpoint/runtime 语义一致性。`invoke_llm_static` 与 formal Gateway 双边复验并在容器检查/启动前消费内存状态与 atomic claim；正式 CLI 不自行造 claim，只读并验证 core claim 后写成功或失败的 create-only receipt。candidate、direct API、协调重哈希失败探针、第二 handle、新 loader 或新 Gateway 都不能复用。

审查随后把“非正式”与“无网络”拆开：non-formal 标签不能阻止真实计费，所以 v5 diagnostic 不再接受任何 `ControlledAuthoringGateway` transport。专用 sealed replay 只接收调用前已物化的严格响应，绑定预期 request bytes，且 exact class/method identity 受检；它可以验证 parser/compiler，但没有网络能力。真实 `UnifiedChatTransport` 的 JSON-mode 行为单独做 monkeypatched unit test，不能经 diagnostic invocation 触发。这样 owner authority 管住仓库提供的所有真实 provider 路径，而不是只管最终产物能否标 formal。当前没有真实 v5 freeze，因此能力机制完成不等于授权已签发。

### 如何验证

- `tests/test_llm.py` 证明正式 worker 可把 usage 持久化委托给父 Runner，而不触碰只读文件系统。
- `tests/test_static_authoring.py` 证明 invocation-only verified handle 不虚构 C3 runtime，正式容器成功路径仍可生成回执，worker 失败路径会保留 request、stdout、stderr 和 `job-failure.json`。
- authoring、freeze、egress 与 LLM 针对性测试共 68 项通过。
- v4 authoring sandbox 镜像 `6e8677…0f6e` 与 `locks-v4` 已重新部署；allowlisted CONNECT、非 allowlist 拒绝、direct IP egress 拒绝和 external DNS 拒绝四项真实探针均通过。
- `specs/authoring/incidents/llm-static-primary-20260723-v1.json` 明确把旧尝试标为 `invalid_not_c1_evidence`。
- owner 批准的替代调用真实取得并保留了 provider response、usage、费用和 isolation attestation；随后 draft schema 校验失败，证明“先落证据再解释”的顺序在真实坏输出上有效。
- v4 运行器测试证明 claim 后任意异常都会尽力写终态 receipt，且 claim 永不删除；真实 attempt 随后验证了这条路径：draft 被拒绝后 claim 仍保留，终态 receipt 准确记录 provider 完成、usage/cost 和 `retry/fallback/repair=false`。
- `tests/test_static_authoring_call_authority.py` 覆盖：核心 invoke 与 direct Gateway 无 authority 时在 runtime 前拒绝；candidate freeze 不能签发 handle；同一 claim 在预加载/重新 loader/新 Gateway 间 create-only，不能复用。
- 同一测试文件还证明：即使攻击者协调重算失败 probe 的 deployment receipt、lock manifest、runtime bundle 和 freeze 摘要，语义 cross-check 仍会拒绝；claim/output/receipt 的反向与交叉嵌套也在加载时失败。
- `tests/test_static_authoring.py` 的包装器回归证明 v5 diagnostic 经 `ControlledAuthoringGateway` 时，wrapper `.complete()` 与 `llm.chat` 调用数均为 0；sealed replay 只校验已物化响应。
- `tests/test_formal_authoring_runner.py` 覆盖 candidate/缺 authority 在 invoke 前失败，以及受权 v5 的成功、draft-rejected completed 与 post-claim unknown 三类 receipt；receipt 绑定 claim file/self SHA、request、deviation、budget、output contract 和 runtime bundle/files。
- sealed replay/wrapper、完整正式 Runner、call authority、v1/v2 canonical factory 与 v2 CLI E2E 的最终安全边界定向重跑为 `21 passed in 29.38s`。这是离线 authority 机制证据，不是新 provider 调用或 C1 成功证据。
- 加入 v5 authority、author-content、正式终态 receipt 与八工具契约后，全仓最终回归为 `1030 passed, 25 skipped, 2 deselected in 445.69s (0:07:25)`；skip/deselection 不被包装成真实 provider、模型质量或 C1/C3 证据。

### 剩余限制

旧尝试没有响应、usage 或供应商 request ID，是否成功生成和计费永远无法从仓库证据恢复，因此 v4 账本明确把它列为 provider outcome unknown，而不是零调用。修复后的 v3 替代调用证明审计链有效但 payload 不合格；v4 create-only claim 也已实际消费并留下完整的第二份 draft-rejected receipt。v5 capability 仍信任仓库 owner/operator、本机管理员与外部审批摘要；它不阻止持有凭据的人绕开整个仓库自行调用 provider，只能保证本项目 formal 路径不把那种调用认作授权证据。当前没有真实 v5 freeze/authorization/pre-review draft，不能进入 checklist 或最终 Bank compile；C1 未通过。

### 30 秒回答

我把一次性付费调用拆成“先授权、再落 provider 事实、最后解释内容”。第一次真实调用因只读 usage 路径丢失结果，我没有盲目重试，而是让父 Runner 先原子保存 request/response/隔离证据。后来又发现只在 CLI 拒绝 candidate 可被核心 API 或新 Gateway 绕过，于是增加 owner-issued capability，并在任何 runtime/provider 动作前 create-only claim；核心 invoke 与 Gateway 双边要求，reload 也不能复用。当前 v5 尚未签发，所以只证明 fail closed，不声称调用成功。

### 2 分钟回答

这个任务预注册了“最多一次成功作者调用”，所以普通的异常重试策略并不适用。首个容器请求的代理日志证明它连接到了 DashScope，但父进程只收到模糊的 worker failure，response、usage、request ID 和 stderr 都没留下。代码审计后我发现一个确定性 bug：通用 `llm.chat()` 收到成功响应后会先写 `runs/usage.jsonl`，而镜像中的项目根位于只读文件系统且进程是非 root，因此成功响应反而会在输出文件发布前失败。我们无法证明供应商当时是否成功或计费，所以我把 provider outcome 标成 unknown，没有粉饰成未调用，也没有直接重试。

我比较了给 worker 增加可写挂载、只修当前写盘点和重构审计所有权。最终选择让 worker 只做一次网络调用，父 Runner 独占 usage/cost 记账；gateway 把 exit code、stdio、耗时和 command hash 作为结构化错误返回；Runner 对失败原子保存 failure bundle，对成功则在任何语义校验前保存完整 response 和 isolation attestation。模型输出即使 schema 不合格，付费调用事实仍可审计。v3 替代调用的 provider 与隔离合格，但 payload 有 40 个 schema 错误，响应和 764 microUSD 费用仍被完整留存。后续 v4 授权进一步使用 create-only claim：在启动容器前就永久消费 attempt，失败或崩溃也不能删除后重跑，并尽力生成终态 receipt。旧账无法恢复，v3 负结果不能被修补，v4 也只能产生一个新结果。

v5 审查时又出现了更隐蔽的 trust-boundary 问题：CLI 会拒绝 `candidate_not_authorized`，但核心 invoke 和 formal Gateway 不持有 owner authority；`max_successful_calls=1` 只是数据，Gateway `_used` 也只能约束单实例。于是调用者可以直接进核心 API或新建 Gateway。修复不是再加一层布尔判断，而是建立不可伪造、一次性的 `VerifiedAuthoringCallAuthorization`：loader 从外部摘要锁定的 frozen owner freeze 复验 deviation、预算、packet/output contract、run/output/claim/receipt 和 runtime bundle；核心 invoke 与 Gateway 都要求同一 capability，容器前原子创建 claim；第二 handle、第二 loader 和新 Gateway 都因同一 claim 失败。这样“一次”由跨入口证据实现，而不是依赖大家都走 CLI。真实 v5 authority 尚未签发，所以 C1 仍保持未通过。

### 证据入口

- `specs/authoring/incidents/llm-static-primary-20260723-v1.json`
- `specs/authoring/llm-static-primary-20260724-replacement-v1-adjudication-receipt.json`
- `src/skillchain/llm.py`
- `src/skillchain/static_authoring.py`
- `scripts/run_formal_authoring.py`
- `deploy/authoring/locks-v4/`
- `specs/authoring/authoring-freeze-lock-v3.json`
- `specs/authoring/authoring-freeze-lock-v4.json`
- `tests/test_llm.py`
- `tests/test_static_authoring.py`
- `tests/test_static_authoring_call_authority.py`
- `tests/test_formal_authoring_runner.py`

---

## 08. authority-issued 不等于外部工件已获批准

**状态：部分解决**

### 一句话问题

进程内 authority 能证明 registry 来自审核过的 concrete service graph，却不能证明调用方给它的模型、索引和 catalog 是项目外部批准的那一版；只保存一个刚算出的 runtime hash 仍然是自证。

### 背景与影响

canonical builder 已经检查 exact service/backend 类型、显式 runtime binding、evidence validator 和运行中漂移，并为合格的七工具 v1 或八工具 v2 service graph 签发不可伪造的 handle。但正式实验还需要回答另一个问题：这些 service 加载的 product index、KB、detector/OCR 模型和文档审批究竟来自哪份已审核工件。若把两层信任混为一谈，测试 fixture、调用方临时路径或被协调改写的 manifest 也可能得到进程内 handle，随后被误报为“生产 registry 已冻结”。

### 观察到的证据

- `build_mvp_registry` 的 authority 审核 service graph 与 live binding，但它没有外部 artifact-lock 文件这一输入。
- 旧 `formal_evaluation_cli` 要求调用方提供 `module:callable` runtime factory，仓库却没有生产 factory；实际只能靠测试 fixture 或项目外自写胶水。
- 2026-07-23 本地 `data/clean/`、正式 product/KB index、detector/OCR manifest、AssetCatalog 和 document-safety catalog 均未就绪，因此当时不可能诚实生成真实 C3 tool registry runtime receipt；截至本次 v4 authoring 收口，该 C3 前置事实仍未改变。
- 新测试证明 canonical registry 可以由生产入口重建和签发，同时协调重哈希、runtime 摘要篡改、非 canonical lock、路径逃逸和覆盖发布均会失败。

### 根因

“authority”一词掩盖了两个不同问题：代码权威负责控制谁能签发 handle；项目治理负责批准哪些外部字节可以进入正式运行。前者是进程内能力安全，后者是跨进程、跨时间的供应链信任根。一个 SHA-256 只有在期望值来自被验证对象之外时才有锁定意义。

### 考虑过的方案与取舍

1. 直接提供一个默认 runtime factory：使用方便，但路径和摘要来自环境自报，无法区分 fixture 与正式工件。
2. 只保存 aggregate `registry_runtime_sha256`：文件较小，但不能审查单工具漂移，也无法证明它绑定哪份 artifact lock。
3. 把所有字段放进一份自哈希 manifest：结构简单，但生成过程既选择工件又批准 runtime，仍可协调重哈希。
4. 使用 artifact lock 与 runtime lock 两阶段冻结：步骤更多，还会触发一次 live canary 和本地模型加载，但能拆开输入批准、运行装配与最终 runtime 批准。

### 最终方案

新增 `FormalRegistryArtifactLock`，逐项锁定 embedding track、product query/index、KB catalog/bundle、detector/OCR manifest、统一 AssetCatalog、document-safety catalog/review ledger 和 ToolSpec registry。生产装配器只接受 artifact root 下无符号链接的 canonical 相对路径，复验全部摘要，执行无 cache 的所选 embedding live canary，并真实加载 detector 与 OCR engine；exact concrete services 通过后才调用 canonical builder 取得 authority handle。

随后生成深度不可变、排序的 `FormalRegistryRuntimeLock`。schema/policy v1 保留历史七工具字节；v2 要求恰好八个 tool/evidence binding，并把 `multi_product_search` composite 的 detector/retrieval 传递 runtime 与 evidence 纳入同一 registry snapshot。两代锁都记录被审核 registry/service/backend 方法的源码指纹、aggregate runtime 和精确 artifact-lock 文件 SHA。candidate 明确不是正式批准；正式 loader 必须再取得两份锁文件的外部 SHA，重新装配并逐字段比较 snapshot。formal evaluator v2 只通过该 registry 调用 composite，不再接收 registry 外的 MultiProduct executor。

### 如何验证

- `tests/tools/test_production_registry.py` 覆盖外部摘要、canonical 路径、真实 fixture artifact 全量重载、authority handle、双锁重建、协调重哈希、runtime 篡改、create-only 和环境缺失。
- v2 定向用例覆盖八工具 lock/evidence 全集、generation 混配拒绝、composite runtime 重建，以及 evaluator 拒绝额外 MultiProduct executor；v1 仍按历史七工具路径兼容。
- `scripts/issue_formal_registry.py candidate` 只创建待批准 runtime lock，并输出 `formal_eligible=false`。
- `scripts/issue_formal_registry.py verify` 只有在双锁和 live snapshot 全部一致时才输出 `authority_issued=true` 与 `formal_eligible=true`。
- `docs/formal-tool-registry-runbook.md` 明确每种摘要的语义和外部冻结顺序。

### 剩余限制

当前验证使用的是严格 synthetic fixture，证明机制但不证明真实模型质量、速度或数据许可。真实 clean/query/index、模型 manifest、安全审批和 gold 尚未发布，所以没有提交占位 artifact lock，也没有真实 authority receipt；C3 继续未通过。环境变量中的“expected SHA”仍需要操作者从独立审批记录取得，不能由同一命令临时计算。模型成功加载也不等于 benchmark 指标达标，仍需每工具 gold 和 formal report。

### 30 秒回答

我发现“registry 拿到内部 authority handle”不等于“它用的是获批准工件”。内部 authority 只能审核 concrete Python service graph；如果模型和索引路径由调用方临时提供，fixture 也可能看起来正式。我把它拆成双锁：artifact lock 外部固定所有输入，装配器复验后跑所选 embedding 的 live canary 并真实加载 detector/OCR；再生成逐工具 runtime lock，正式 loader 必须持有两份锁的外部摘要并重建比对。测试覆盖协调重哈希和 runtime 篡改。真实工件尚未齐，所以我明确保持 C3 未通过。

### 2 分钟回答

这个问题的关键是把进程内能力安全和跨时间供应链批准分开。项目原先的 registry authority 设计其实很强：只有 canonical builder、exact concrete types、完整 runtime resolver 和 evidence validator 才能签发 handle，实例或类被改写后也会失效。但它审核的是“对象怎么连起来”，不知道“对象加载的外部字节是否是审批过的版本”。而 evaluator 的 runtime factory 又留给调用方，仓库缺正式实现，于是最容易出现的假闭环是：用 fixture 或临时路径构造一个确实拿到 handle 的 registry，再把当场计算的 hash 当外部锁。

我比较过默认 factory、只锁 aggregate hash 和单一自哈希 manifest，最终选择两阶段双锁。artifact lock 逐项绑定 query/index、KB、模型、AssetCatalog 和 PII review ledger；每份锁文件还必须由调用方提供独立登记的文件 SHA。装配时禁止路径逃逸和 symlink，复验所有工件，运行无 cache canary，并实际加载本地模型，之后 canonical builder 才签发。runtime candidate 保存排序且不可原地修改的 generation-specific binding：v1 恰好七工具，v2 恰好八工具并把 Multi-Product composite 收进同一 authority graph；candidate 不能批准自己。正式加载再次重建整个 graph，与外部固定 runtime lock 逐字段比较，v2 evaluator 也不再接收额外组合 executor。这样能清楚回答“谁签发、签发了什么、外部字节是谁批准的”。目前真实工件仍缺，所以成果是生产机制完成、证据阻断未解除，而不是宣称 C3 已关闭。

### 证据入口

- `src/skillchain/tools/production_registry.py`
- `scripts/issue_formal_registry.py`
- `tests/tools/test_production_registry.py`
- `docs/formal-tool-registry-runbook.md`
- `src/skillchain/tools/registry.py`
- `src/skillchain/tools/formal_evaluation_cli.py`
- `docs/plans/2026-07-20-p0-p1-closure.md`

---

## 09. “本地 embedding”只有进入 authority 路径并适配中文才成立

**状态：部分解决**

### 一句话问题

计划声称受限资产可以走本地 embedding，但生产 registry 原先只允许 DashScope；后来加入的 RN50/OpenAI CLIP 虽然本地，却不适合作为中文查询的正式候选。

### 背景与影响

ABO 等来源的保守许可边界可能禁止把图片上传到远程 embedding 服务。若正式 registry 只能构造远程 backend，所谓“本地替代”只是文档承诺；若为了维持 1024 维索引随手选英文 CLIP，系统虽然能跑，中文文本—图片检索的有效性却没有根据。两者都会让 C3 在安全或任务有效性上形成假闭环。

### 观察到的证据

- `FormalRegistryArtifactLock` v1 和 production loader 只接受 `qwen3-vl-embedding`。
- 三个 retrieval ToolSpec 又固定写成 `remote_embedding`；即使 handler 已按 backend location 选择本地路径，formal receipt 仍会把本地运行错误描述为远程策略。
- Product index 全链路固定 1024 维，最容易接入的 RN50/OpenAI 也是 1024 维，但不是中文/多语种训练目标。
- 已安装 OpenCLIP 3.3.0 的官方配置包含 `xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k`，输出同为 1024 维。
- 多语种 OpenCLIP 还依赖 Hugging Face tokenizer；只锁权重、运行时再联网取 tokenizer 仍然破坏可复现和本地边界。

### 根因

设计只检查了“向量维度兼容”和“有没有一个 local 类”，没有把任务语言、许可边界、authority concrete-type 审核、全部模型依赖和离线加载作为同一条验收链。

### 考虑过的方案与取舍

1. 所有来源继续用 DashScope：中文能力较合适，但受限图片没有上传授权时不可用。
2. 使用 RN50/OpenAI：接入便宜、维度兼容，但中文文本风险太高，只适合作为 diagnostic。
3. 把索引改为动态 512 维并用较小多语种模型：资源更友好，但会扩大当前索引协议迁移；可作为模型 gold 后的后续优化。
4. 先支持 1024 维 XLM-R OpenCLIP：模型较大，但不改现有 index wire contract，并能把全部调用保持本地。

### 最终方案

artifact lock 升级到 v2，正式本地轨只接受多语种 XLM-R OpenCLIP identity。模型 manifest 锁定权重、runtime config、图像预处理参数和 tokenizer 全部文件；backend 把 HF model/tokenizer 路径改为 artifact 内本地目录，设置 `local_files_only` 并核对 OpenCLIP、Torch、Transformers 精确版本。ToolSpec 改用稳定的 `runtime_bound_embedding`，实际 local/remote location 由 authority runtime lock 证明，远程 handler 仍逐资产要求上传许可。registry authority 同时审核新的 concrete backend 及方法身份。RN50 实现只保留测试/诊断兼容，不是 canonical formal 选择。

### 如何验证

- `tests/tools/test_embedding.py` 覆盖中文文本输入、权重/config/tokenizer manifest、精确包版本、本地 tokenizer 路径与 `local_files_only=true`。
- `tests/tools/test_production_registry.py` 证明本地轨缺 manifest 或 model identity 不符时 artifact lock 失败。
- `tests/tools/test_model_artifacts.py` 要求 multimodal embedding manifest 至少包含 weights/config/tokenizer。

### 剩余限制

多语种候选权重约 4.77 GB，尚未下载或在真实 ranking gold 上测量质量/延迟；因此这里只能声明正式装配机制已具备，不能声明 embedding 已锁定通过。若资源成本不可接受，应在看正式系统结果前做动态维度迁移并比较较小的 512 维多语种候选。

### 30 秒回答

我发现计划里的“本地 embedding”实际上进不了 production registry，而且最容易接上的 RN50 虽然维度匹配，却不适合中文查询。我把 artifact lock 和 authority 扩展为真正的本地多语种 OpenCLIP 路径，连 tokenizer 和 Transformers 版本一起锁定，强制 `local_files_only`，避免运行时偷偷联网。现在机制和负向测试通过，但大模型的真实 ranking 质量和资源成本还没测，所以 C3 仍未通过。

### 2 分钟回答

这个问题同时跨许可、模型选择和供应链。ABO 的保守许可策略可能不允许上传图片，因此不能只有 DashScope；但简单加入一个 RN50 类也不够，因为中文文本和图片必须在同一可靠语义空间里，且 registry authority 必须承认这个 concrete backend。进一步看 OpenCLIP 的多语种模型后，我找到一个仍输出 1024 维的 XLM-R/ViT-H 候选，可以不立即改索引协议。真正棘手的是 tokenizer：如果只锁 checkpoint，HF tokenizer 仍可能在运行时联网下载，所谓本地与可复现都不成立。最终我把 artifact lock 升为 v2，锁权重、config 和全部 tokenizer 文件，运行时把 text tower 与 tokenizer 指向同一 artifact 目录，禁止联网，并把 backend/type/method 纳入 authority 审核。代价是候选约 4.77 GB，必须先用 mini ranking gold 测资源和质量；如果太重，再做动态维度迁移，而不是先看主系统结果再换模型。

### 证据入口

- `src/skillchain/tools/embedding.py`
- `src/skillchain/tools/production_registry.py`
- `src/skillchain/tools/registry.py`
- `tests/tools/test_embedding.py`
- `docs/formal-tool-registry-runbook.md`

---

## 10. 数据集名称和二手描述不能替代官方任务/语言/许可元数据

**状态：部分解决**

### 一句话问题

计划把 WildReceipt 写成“中文收据主源”，但官方 MMOCR 元数据将其语言标为 English、license 标为 N/A；继续沿用旧描述会同时夸大中文覆盖并绕过许可阻断。

### 背景与影响

Document Reading 要评价 OCR 文本、结构化字段和 grounded line evidence。WildReceipt 的标注结构很适合这类机制测试，但“结构适合”不等于“语言就是中文”，可公开下载也不等于允许再分发或公开演示。错误描述会让简历项目宣称并未被数据支持的中文文档能力，还可能把未知许可误写为获批。

### 观察到的证据

- 项目原 portfolio role 为 `chinese_receipt_ocr_kie_gold`，计划正文多处写“中文收据”。
- 官方 MMOCR dataset metadata 将 WildReceipt language 标为 English、license 标为 N/A。
- 官方样例格式确实提供 polygon、text、label，证明它适合 OCR/KIE gold；这不证明中文语言覆盖或再分发权限。

### 根因

数据选型时把论文/博客中的用途摘要当成了数据集事实，没有分别审计任务监督、语言、许可、PII 和目标 capability 的映射。

### 考虑过的方案与取舍

1. 完全删除 WildReceipt：避免误述，但会失去很有价值的 receipt KIE 标注。
2. 保留“中文”说法并靠中文 LLM 改写：中文问题不改变图片中文字语言，淘汰。
3. 保留为 receipt OCR/KIE gold，把中文限定为用户问题语言，并另设中文文档 challenge：监督信号诚实，许可与 PII 仍独立 fail closed。

### 最终方案

portfolio role 改为 `receipt_ocr_kie_gold`，计划明确中文问题与数据集语言分离。新增 source-review policy/ledger：每条人工决定必须绑定 exact source lock 和许可证据字节，未知权限默认为 false；WildReceipt 还必须完成 PII review。adapter 在正式清洗前按 revision、source-lock、license ID、evidence SHA、用途和权限再次校验审批，不能只看 source name。由于 RAW 下载由独立会话管理，收口计划又定义了跨会话 handoff contract：必须交付 revision、source-lock、文件 inventory、许可证据原始字节和缺失项；一个可变目录或“下载成功”不能成为 formal trust root。

### 如何验证

- portfolio 与 source-review policy 具有重新计算的外部摘要，并通过 `scripts/review_data_sources.py policy-status`。
- `tests/data/test_source_review.py` 覆盖缺审批、伪 PII 状态、source-lock 漂移和未授权 public demo。
- `docs/data-source-adjustment.md` 已去除“中文收据数据集”的事实声明。

### 剩余限制

真实 source lock、许可页面快照和项目所有者审批尚未由 RAW 会话按 handoff contract 交付，因此 WildReceipt 仍不能进入正式 clean/catalog；另外仍需选择真正的中文文档 challenge，避免把“中文提问英文票据”包装成完整中文 OCR 能力。

### 30 秒回答

我在数据尽调里发现 WildReceipt 被计划误写成中文收据集；官方元数据其实标 English，license 还是 N/A。它的 bbox/text/field 标注仍很适合 OCR/KIE，所以我没有简单删除，而是把 role 改成语言中立的 receipt gold，把中文只放在用户问题层，并新增精确绑定 source lock、许可证据和 PII 决定的人工 gate。真实审批没完成前它不会进入 formal clean。

### 2 分钟回答

这个案例让我把“数据能不能做任务”拆成五个问题：标签是否提供所需监督、语言是否覆盖目标、身份是否可追踪、许可是否允许具体用途、是否含 PII。WildReceipt 在第一个问题上很好：官方格式有 polygon、text 和 field label；但项目从二手描述直接推成“中文收据主源”，而官方 metadata 是 English、license N/A。中文 LLM 改写用户问题并不能改变票据中文字的分布，也不能创造再分发权限。我的处理是保留它的 OCR/KIE 价值，移除中文数据集声明；中文问题与额外中文 challenge 分开报告。同时建立独立 source-review ledger，人的批准绑定 exact revision、source-lock SHA 和许可证据 SHA，并逐项列 local/remote embedding、公开演示、再分发和 PII。这样 adapter 换了字节或权限不足都会失败。当前机制已验证，但真实许可和 PII 决定还没签，所以状态是部分解决。

### 证据入口

- `specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`
- `specs/data_sources/mvp-source-review-policy-v1.json`
- `src/skillchain/data/source_review.py`
- `scripts/review_data_sources.py`
- `tests/data/test_source_review.py`
- `docs/data-source-review-runbook.md`
- `docs/plans/2026-07-20-p0-p1-closure.md`

---

## 11. JSON mode 不等于结构化输出受 schema 约束

**状态：部分解决**

### 一句话问题

`response_format={"type":"json_object"}` 只能保证模型返回某个 JSON 对象，不能保证它符合业务 schema；JSON Schema 可以作为 prompt/audit 合同，但不能被误写成 provider 或通用本地 validator 已强制执行。

### 背景与影响

LLMStatic 是主实验基线的一部分，作者模型必须为 6 个 capability 返回严格步骤、工具绑定、成功规则覆盖和输出契约。该调用预注册为一次成功 provider response，不能依赖“错了就重试”。如果 JSON mode 被误当成 schema constrained decoding，测试 fixture 可以一直返回完美对象，但真实模型第一次暴露字段类型和嵌套结构错误时，基线会在付费调用后才失败。

### 观察到的证据

- `llm-static-primary-20260724-replacement-v1` 的 provider、endpoint、requested/response model、`finish_reason=stop`、usage、费用和隔离回执全部合格。
- 响应文本可被标准 JSON parser 读取，顶层也恰好是 `schema_version` 与 `drafts`，并包含全部 6 个 capability。
- 它不是项目 canonical JSON；更重要的是，即使先做无损 canonicalization，严格 Pydantic 校验仍报告 40 个错误：9 个整数 `step_id`、9 个缺失 `tool_name`、9 个缺失 `success_rule_ids`、6 个 list 形式的 `rule_coverage`，以及 7 个级联 too-short 错误。
- 通用 Qwen/OpenAI-compatible transport 的 `json_mode=True` 实际只发送 `{"type":"json_object"}`，没有发送 `AuthoringDraftPayload` JSON Schema。
- v4 forced-function 响应满足 `tool_calls` finish、空 assistant text、单个正确 tool name 和完整 provider/usage/isolation 约束，但 arguments 把 `drafts` 数组放进 JSON 字符串；外层严格 schema 报 1 个类型错误。诊断性解析该字符串可见全部 6 个 capability，但仍有 16 个排序、规则覆盖和键名错误。
- **已验证的离线事实：** 2026-07-24 复核 DashScope 官方 structured-output/API 文档，公开接口只描述 `json_object` 语法模式，没有可提交本项目 schema 的 response-format `json_schema` constrained output。当前 v5 候选因此显式记录 `provider_guarantee=json_syntax_only` 与 `json_schema_enforcement=prompt_and_audit_only`。
- **已验证的离线事实：** v5 `AuthoringContentPayload` 将 wire `schema_version=2` 设为必填，模型只提交 capability/objective、步骤 instruction/tool/success-rule 映射、fallback 文本和引用；packet-bound schema 不包含 compiler-owned coverage/output/fallback flags/step IDs。Runner 的实际接受门是 strict JSON、Pydantic shape 和 packet-bound trusted compiler，不声称执行通用 JSON Schema validator。
- **待验证事实：** v5 尚未获批 provider call，也没有合格 pre-review draft；所以当前测试只能证明新边界会 fail closed，不能证明 Flash 会一次生成合格 content。

### 根因

系统把四种不同保证混为一谈：JSON 语法有效、JSON Schema 被展示/冻结、对象逐字节 canonical、对象满足业务语义合同。prompt 能展示 schema，却不是传输层 enforcement；Pydantic/可信 compiler 可以在本地严格拒绝，却也不等于 provider constrained decoding。fixture 又精确按内部模型构造响应，因而没有模拟真实模型对嵌套字段的偏差。之前已经避免让 LLM 计算哈希，但仍让它承担了不必要的 canonical serialization 和 TaskSpec 规范复制责任。若为了“解决”这个问题而假设 DashScope 存在 response-format `json_schema`，只会把实验失败换成接口兼容与审计表述错误。

### 考虑过的方案与取舍

1. **人工补齐当前响应**：可以快速得到 draft，但 9 个工具绑定和 9 组规则覆盖属于实质性内容，不再是同一 LLMStatic 基线，淘汰。
2. **Runner 根据 Task Specification 自动推断缺失字段**：确定性较强，但这是看到结果后新增的 compiler 行为，会把 SpecBaseline 信息静默注入模型输出，当前 run 不得采用。
3. **自动调用 Plus 或重试 Flash**：违反禁止自动 fallback 和一次成功 provider response 的预算，也会形成结果驱动选模，淘汰。
4. **保留负结果，使用 provider 支持的 forced function submission**：这是 v4 的前瞻选择；它改善 framing，但真实结果仍把 `drafts` 二次序列化，证明 function parameters 也不能当作业务 schema guarantee。
5. **前瞻 v5 使用最小语义 payload + `json_object` + 可信 compiler**：承认 provider 只保证 JSON 语法；把 packet-bound JSON Schema 嵌入 prompt/request 并冻结用于审计，真正接受门由 strict JSON、Pydantic shape、TaskSpec/registry 同代约束和可信 compiler 组成。它减少模型承担的机械字段，又不虚构 provider 能力；需要全新的 deviation、预算、run ID 和 runtime lock，采用为离线候选。

### 最终方案

v3 响应原样保留，标记为 `formal_provider_call_eligible=true`、`draft_formal_eligible=false`；不生成 pre-review draft，不人工修补，不重试，不调用 Plus。项目所有者随后前瞻批准独立 v4 deviation：保持 taxonomy、Task Specification、ToolSpec、Flash revision、公开资料和数值预算不变，改用一个不可执行的 `submit_authoring_payload` 作为结构化输出信封。请求把展开后的精确 `AuthoringDraftPayload` schema 放入 parameters，强制 named tool choice，禁止 assistant text 和 parallel tool calls，并对 Qwen 显式发送 `enable_thinking=false`；Runner 对 arguments 做 strict JSON、Pydantic、规则覆盖与工具权限校验，然后自己 canonicalize、绑定输入和计算摘要。

审查中还发现两个与“只调用一次”同等级的信任边界问题。第一，检查 output/receipt 不存在不是互斥锁，两个进程可能同时通过；因此 v4 在启动容器前以 create-only 原子 claim 消耗授权，claim 永不删除，任何失败都需要新 owner-approved deviation 才能再次调用。第二，只冻结 runtime manifest 摘要却允许 CLI 传任意 profile，会使实际网络边界脱离 freeze；正式 Runner 现在从 manifest 加载并交叉验证准确 profile、policy、deployment receipt、image/network/proxy/engine 和四项探针。冻结账本也不把 2026-07-23 的 provider outcome unknown incident 当作零调用，而是分别记录 1 次已知可审计响应、1 次未知 incident 和最多 1 次新增 attempt。

真实 v4 结果证明 forced function 改善了 transport framing，却没有把 provider guidance 变成 schema guarantee。Runner 按预注册规则拒绝 stringified `drafts`；没有自动 unwrap，因为即使只读解析内部字符串也仍有 16 项内容/结构错误，继续排序、改键或补覆盖会成为看过结果后的实质 repair。后续契约审计又证明，16 项并非都应归因于模型：function schema/提示词没有表达 lexical sort，TaskSpec-copy 的部分源顺序还与 Runner 冲突；这需要新协议前瞻修复，而不是给 v4 翻案。claim 和终态 receipt 保留，未重试、未调用 Plus。当前结论不是“v4 成功”，而是“接口修订及失败关闭机制成功执行，LLMStatic draft 仍失败”。

当时位于 `codex/authoring-contract-v5` 分支的 v5 离线候选进一步重划字段责任。wire payload v2 的 `schema_version` 必填，作者模型只给出需要判断的语义字段；compiler v4 从已验证 TaskSpec 注入 step ID、规则/输出 coverage、fallback flags 和 canonical 顺序，并对缺失/重复/非法 rule ID、越权 tool、未知引用继续 fail closed。packet-bound JSON Schema 的角色被精确限定为 prompt guidance 与审计快照；候选 manifest 记录 `provider_response_format=json_object`、`provider_guarantee=json_syntax_only`、`json_schema_enforcement=prompt_and_audit_only`。没有 owner authority 时，candidate 不能启动正式调用，也不能落入 legacy 路径。

### 如何验证

- adjudication receipt 绑定 request、response、isolation attestation、provider request ID hash、18,222 token、764 microUSD 和 40 项错误统计。
- 离线复核证明标准 JSON parse 成功，而 `AuthoringDraftPayload.model_validate(..., strict=True)` 失败；因此拒绝原因不是网络或响应截断。
- `test_formal_script_recovers_provider_facts_from_rejected_draft` 验证 Runner 能从已保存的坏 draft response 恢复 provider-call 事实并正确核算费用。
- 原有 `test_noncanonical_draft_text_is_rejected` 保持不变，防止在看到本次输出后放宽当前协议。
- v4 单元测试覆盖 exact forced tool、request-v2 schema hash、assistant text/多 tool/错 tool/duplicate key 拒绝、Qwen 显式 non-thinking、runtime bundle 交叉绑定、failed probe、claim 重入、claim 后异常终态 receipt 和成功 receipt bindings。
- `authoring-freeze-lock-v4.json` 以前瞻、create-only 方式绑定唯一 run/output/receipt/claim、deviation、历史 receipt/incident、submission schema 和 `formal-v3`/`locks-v4`；四项真实 egress 探针均退出 0。
- 真实 v4 receipt 记录 18,839 input、1,762 output、20,601 total token、821 microUSD、一个正确 tool call、空 assistant text、claim consumed 和 `retry/fallback/repair=false`；strict schema 拒绝结果与保存的 raw response 一致。
- v5 定向测试验证：packet-bound schema 不含 compiler-owned 字段；wire `schema_version` 缺失、duplicate/extra key、capability/rule/tool 覆盖错误、TaskSpec/registry generation 混配和协调重哈希都会被拒绝；compiler 对同一 content 产生确定性 canonical Bank。
- `build_authoring_packet_v5_candidate.py --check` 验证 freeze candidate、packet、prompt 与 schema snapshot 可重建，并输出 `candidate_not_authorized`/`provider_call_authorized=false`；CLI、核心 `invoke_llm_static` 和 direct formal gateway 都会在 runtime/provider 前拒绝无 owner-issued capability 的 v5，候选 freeze 不能加载该 handle。该验证只覆盖离线 authority 机制，不是实际调用证据。

### 剩余限制

JSON mode、prompt 中的 JSON Schema 和 function parameters 都不能保证业务 payload 一次
合格；v5 只是减少无谓自由度并加强本地拒绝。后续 v5 真实 session 已得到合格 content，
owner 也完成了限时 checklist，但 CLI 仍不能提供 provider tokenizer、served revision、
底层 attempt、provider request ID 或费用证明。原子 claim 只防同一共享工作区的正常并发，
不能阻止拥有独立 clone 和凭据的恶意 operator；本机管理员也仍在威胁模型之外。最终 Bank
compile 仍等待真实 C3 authority runtime，因此 C1 仍未通过。

### 30 秒回答

真实 authoring 调用让我验证了两层误区：JSON mode 只保证“是 JSON”，forced function 也只是
schema guidance。v3 有 40 个错误，v4 把 `drafts` 变成字符串，内部仍有 16 个错误。我保留
两份负结果，并在 v5 把模型 payload 缩成语义字段，由可信 compiler 注入规范字段；JSON
Schema 只作 prompt/audit，strict JSON + Pydantic + compiler 才是接受门。v5 后续一次生成
合格 draft 并通过 owner checklist，但 C3/Bank 仍使 C1 保持未通过。

### 2 分钟回答

我们的 fixture 一直返回严格的 `AuthoringDraftPayload`，真实 transport 却只把 `json_mode=True` 映射成 OpenAI-compatible 的 `json_object`。这只能约束 JSON 语法。正式 Flash 调用的 provider 身份、模型 revision、finish reason、18,222 token、764 microUSD 和容器隔离全部合格，文本也确实能 parse，并覆盖 6 个 capability；但它把 `step_id` 写成整数，所有步骤缺少 `tool_name` 和 `success_rule_ids`，`rule_coverage` 还是扁平 list。即使无损 canonicalize，仍有 40 个严格 schema 错误。

关键判断是不能因为问题看起来“容易修”就修改当前结果。自动从 Task Spec 补工具和规则会把基线变成 SpecBaseline+LLM prose；人工补齐属于实质性编辑；自动 Plus fallback 或再次 Flash 会形成结果驱动选模。所以我把 provider-call 事实和 draft 可用性拆开记录：响应、usage、费用、request ID hash、隔离和错误统计不可变留存，不生成 pre-review draft。

之后我先查清 provider 的真实能力边界：JSON mode 没有 response-format JSON Schema，而 Qwen function calling 的 parameters 也只是引导。v4 因此前瞻改成 forced named tool，并用原子 claim、冻结 manifest 和实际 CLI profile/policy/receipt/image/network/proxy/engine/四项探针约束唯一额外调用；未知 provider outcome 也单独计数，不能伪装成零成本。

历史 Qwen v4 真实调用满足所有 transport 和隔离约束，却把 `drafts` 数组二次序列化成字符串。Runner 在第一个严格类型错误处拒绝。为了判断是不是纯包装问题，我只做了不落盘的诊断性解析：六个 capability 都在，但仍有 16 项排序、覆盖和键名错误，因此自动 unwrap 后再排序/补字段会实质改变 treatment。我没有这样做，也没有调用 Plus。

前瞻 v5 没有继续赌另一种 provider envelope，而是减少作者模型的责任：wire `schema_version`、capability、objective、步骤语义、tool/success-rule 映射、fallback 文本和引用由模型提交；step ID、coverage、output、fallback flags 和排序由 packet-bound compiler 注入。DashScope 仍使用 `json_object`，JSON Schema 只冻结为 prompt/audit 证据，Runner 以 Pydantic shape 和 compiler 做等价或更强的业务检查。候选还显式拒绝未授权调用。这样能诚实解释“provider 保证到哪里、本地保证从哪里开始”；但在 owner 批准新 deviation/预算/run/runtime 并得到真实合格响应之前，只能称为离线修复，不能称 C1 成功。

### 证据入口

- `specs/authoring/llm-static-primary-20260724-replacement-v1-adjudication-receipt.json`
- `specs/authoring/llm-static-primary-20260724-replacement-v1-invocation-receipt.json`
- `specs/authoring/authoring-structured-submission-deviation-v1.json`
- `specs/authoring/authoring-freeze-lock-v4.json`
- `specs/authoring/authoring-submission-tool-v1.json`
- `specs/authoring/llm-static-primary-20260724-structured-v1-attempt-claim.json`
- `specs/authoring/llm-static-primary-20260724-structured-v1-invocation-receipt.json`
- `specs/authoring/llm-static-primary-20260724-structured-v1-adjudication-receipt.json`
- `specs/authoring/authoring-freeze-candidate-v5.json`
- `specs/authoring/authoring-packet-primary-v5-candidate.json`
- `specs/authoring/authoring-content-schema-v2-candidate.json`
- `specs/authoring/static-author-v5.txt`
- `deploy/authoring/locks-v4/lock-manifest.json`
- `scripts/build_authoring_packet_v5_candidate.py`
- `scripts/run_formal_authoring.py`
- `scripts/freeze_authoring_packet_v4.py`
- `src/skillchain/llm.py`
- `src/skillchain/static_authoring.py`
- `tests/test_static_authoring.py`
- `tests/test_formal_authoring_runner.py`
- `docs/plans/2026-07-20-p0-p1-closure.md`
- DashScope 官方 structured output 文档（2026-07-24 复核）：<https://help.aliyun.com/en/model-studio/qwen-structured-output>

---

## 12. Schema、TaskSpec 与运行时工具图必须组成同一可执行契约

**状态：部分解决**

### 一句话问题

即使 provider 严格遵循可见 JSON Schema，输出仍可能被 Runner 拒绝；即使 draft 通过 schema，Skill 仍可能调用一条 production runtime 根本执行不了的工具链。

### 背景与影响

v4 的目标是用 forced function 把 LLMStatic 作者输出限制为严格结构。真实响应被拒绝后，如果把 16 个内部错误都归因于“模型没有遵守 schema”，就会漏掉项目自身的两类契约缺口：一类是 wire schema、提示词、Task Specification 与 Runner validator 对同一字段承担了互相矛盾的责任；另一类是 TaskSpec 中允许的工具名称组合并不等于 runtime 存在可执行的数据流。这会导致继续换模型或重试仍可能失败，甚至生成 schema 合格但永远不能运行的 Bank。

### 观察到的证据

已验证事实：

- submission function 的 JSON Schema 对 `success_rule_ids` 和多个 coverage 数组没有编码 `uniqueItems`，也无法表达项目自定义的词法排序 validator；提示词要求“从 Task Specification 复制 coverage”，却没有要求这些数组重新词法排序。
- 冻结 Task Specification 至少有 6 个 coverage 数组使用非词法顺序；真实响应按源顺序复制后，被 Runner 的 `sorted(set(...))` 要求拒绝。这部分不是 provider 单方面忽略了“精确 schema”。
- 响应仍有明确的模型/transport 错误：`drafts` 被放进字符串，style 的 `failure_rule_ids` 键带前导空格；knowledge/recipe 的步骤还把 precondition 或 safety rule 当成 success rule，并漏掉真正 success rule。
- `product.multi_search` 的草案要求先检测，再对每个 crop 调用 `image_product_search`；但 canonical registry 将该工具绑定到 authoritative query asset，`ToolExecutionContext.asset_path()` 会拒绝其他 asset。现有 `MultiProductSearchService` 能安全完成 detect→crop→search，却只由 formal evaluator 作为 `multi_product_chain` 使用，不是 AuthoringPacket 中的 ToolSpec。
- **已验证的离线事实：** TaskSpec v1 只改 `product.multi_search` 的 operator/provenance，把不可执行的原子链替换为 `multi_product_search@1.0.0`；registry manifest v2 在保持 v1 七工具兼容的同时加入第八个 composite ToolSpec。它只接受 authoritative `asset_id`，私有 crop/path/bbox 不进入模型可提交参数。
- **已验证的离线事实：** composite 的 ToolSpec、组合实现及 detector/retrieval 传递依赖共同进入 runtime identity；production runtime lock v2 和 Assistant runtime lock v2 要求恰好八个同代工具。formal evaluator v2 通过 `registry.invoke("multi_product_search", ...)` 执行语义上仍叫 `multi_product_chain` 的 claim，并拒绝额外 executor/runtime digest，避免双 authority。
- **已验证的修复前事实：** canonical environment factory 初版即使加载 registry v2，仍无条件返回 legacy `multi_product_executor`；formal evaluator 正确把它识别为未使用的第二 authority 并拒绝，因此低层 registry-only 测试能通过，正式 factory/CLI v2 却无法运行。
- **已验证的离线修复：** environment factory 现在按 manifest generation 分派：v1 保留 legacy executor，v2 固定 `None` 并只走 composite ToolSpec，未知代际 fail closed；正式 CLI 的 v2 create→verify→evaluate 已从 canonical factory/context 贯通。
- **已验证的离线事实：** Production Assistant 在 registry 调用前核对 selected Skill 的 operators；NoSkill 不伪造 Skill 权限。内部 tool trace 不自动进入 final Judge，只有显式 `VisibleToolEvidence` 投影代表实际呈现给用户的证据。
- **待验证事实：** 当前没有真实 C2 catalog/index/model/gold、八工具 artifact/runtime lock 或 authority receipt；fixture 证明机制，不证明 detector/retrieval 质量、资源成本或 production C3。

### 根因

系统缺少一张明确的“字段责任表”和一张端到端“工具数据流图”。规范常量复制、表示层 canonicalization、作者语义判断和 runtime 授权混在同一个模型 payload 中；Pydantic 自定义 validator 又被误认为会自动等价地出现在 provider JSON Schema。工具层按七个原子工具描述能力，评价层却持有额外组合 executor，Assistant 也只验证“工具存在”而未验证“选中 Skill 声明了该 operator”，最终导致 authoring、execution、evaluation 和 user-visible evidence 四条合同分叉。

### 考虑过的方案与取舍

1. **把 v4 响应自动 unwrap、排序并补字段：** 能快速得到输出，但会事后改变已观察 treatment，并掩盖语义映射错误，淘汰。
2. **只在 prompt 中加“请排序”和 crop 调用说明：** 成本最低，但 prompt 不能创建 schema 约束，也不能给 registry 增加不存在的授权数据流，淘汰。
3. **放宽 Runner，接受任意顺序和任意 crop path：** 可以减少失败，却会弱化确定性与资产边界，并给路径/资产越权留下入口，淘汰。
4. **前瞻重划 model/compiler 责任并正式暴露组合工具：** 由可信 compiler 注入 TaskSpec 的 coverage/output/fallback 等规范字段和表示层顺序；模型只负责 objective、步骤语义、工具选择和 success-rule 映射，Runner 继续校验集合完整性与权限。Multi-Product 优先把已有组合服务版本化为 authority-issued composite ToolSpec。改动较大，但能使作者看到的合同与生产执行一致，采用。

### 最终方案

第一，v4 invocation/adjudication receipt 保持原始拒绝状态，没有修补、重试或调用 Plus；历史七工具 specification/function schema 也被显式钉住，新增 ToolName 不能静默改写已消费协议。

第二，前瞻 authoring payload v2 明确字段所有权。TaskSpec 可确定的 coverage、output contract、fallback flags、step ID 和表示顺序由 compiler v4 从已验证 input 注入；模型只提交 capability/objective、步骤语义、工具/成功规则映射、fallback 文本和引用。Pydantic shape 与 packet-bound compiler 对重复、非法、缺失和越权 fail closed。

第三，TaskSpec/registry/Runner/evaluator 按 generation 一起升级。TaskSpec v1 让 `product.multi_search` 只调用 `multi_product_search@1.0.0`；registry v2 的 canonical authority graph 持有父图、detector result 和私有 canonical crop，只暴露 authoritative `asset_id`，返回 typed `MultiProductResult` 并把传递依赖纳入 runtime/evidence identity。生产 runtime lock、Assistant matrix lock 和 formal evaluator 都按八工具 v2 绑定；formal evaluator 不再接受 registry 外的组合 executor。canonical environment factory 同样按 generation 分派，避免 v2 在真实 CLI 中重新注入 legacy executor。

第四，Assistant 把 Skill policy 与运行权限对齐：Skilled 配置只能调用选中 Skill 声明的 operators，NoSkill 仍按其 treatment 运行；内部 trace 与实际展示给用户的 evidence 分开建模，防止 Judge 看见用户没看见的原始工具输出。

这些是当时位于 `codex/authoring-contract-v5` 分支的离线候选机制。它们不授权 provider call，也没有签发真实 C3 runtime；新的模型输出仍需项目所有者批准独立 deviation、预算、runtime namespace 和 run ID，真实八工具 registry 在该阶段仍需 C2 工件、独立双锁与 authority receipt。

### 如何验证

- v4 adjudication receipt 绑定 request/response/isolation/claim 摘要，并把 outer 1 个类型错误、diagnostic inner 16 个错误、6 个 source-order 冲突字段和 Multi-Product 执行风险结构化记录；它不改变 formal result。
- authoring 测试证明 TaskSpec 非词法顺序由 compiler canonicalize，模型 payload 不能提交 compiler-owned 字段；重复/非法/缺失 rule、越权 tool、generation 混配和协调重哈希都失败。v4 frozen function schema 的七工具枚举保持原字节。
- TaskSpec 测试证明 v1 相对 v0 只替换 Multi-Product operator/provenance；registry 测试证明 v2 恰含八工具，composite 可完成 detect→private crop→retrieval，并拒绝任意 asset、`crop_path`、bbox 和未批准远程上传。
- production-registry 测试证明 runtime lock v2 恰绑定八个 tool/evidence identities，generation 字段不能混配，loader 会重建并比较 authority snapshot；canonical environment factory 对 v1 返回 legacy executor、对 v2 返回 `None`。
- formal-evaluation 测试证明 v2 claim 通过 composite ToolSpec/spec SHA/runtime 执行，拒绝 legacy assignment 和额外组合 executor；legacy v1 仍可按旧伪工具路径重建，v2 正式 CLI 可完成 create→verify→evaluate。
- Phase4 测试证明 Production Assistant 的 selected-operator gate 对 Skilled fail closed、NoSkill 不受虚构 Skill policy 影响；八工具 matrix/create/load trace 与 explicit visible-evidence projection 可重建。
- 包含上述 authoring、registry、evaluator 和 Assistant 改动的全仓最终回归为 `1030 passed, 25 skipped, 2 deselected in 445.69s (0:07:25)`；这确认离线兼容性，不替代真实八工具 authority receipt、RPC gold 或正式运行。

### 剩余限制

截至该轮，`codex/authoring-contract-v5` 分支只完成代码与 fixture 的机械闭环：canonical builder 可以构造八工具 authority graph，测试也能生成/复验 v2 candidate lock，但当时真实 catalog/index/model/gold 不存在，所以没有外部批准的 artifact lock、runtime lock、registry receipt 或 formal report，也没有新 LLMStatic draft。因此该轮不能声称 P0-01、P0-07 或 C1/C3 已关闭。TaskSpec v1、registry v2 和 authoring v5 必须作为新 generation 重新做公平性/预算/runtime 冻结，不能回填 v4；真实 Multi-Product 质量仍需 RPC gold 验证。

### 30 秒回答

一次真实 forced-function 调用失败后，我没有简单归因于模型。我发现 schema、TaskSpec 和 Runner 对排序/规范字段责任冲突，Multi-Product 文本还描述了一条 registry 无法授权的 crop 数据流。我的处理是保留负结果，前瞻把机械字段交给 compiler，并把安全组合服务升级为 registry v2 的第八个 composite ToolSpec；Assistant、formal evaluator 和 runtime lock 同代升级。fixture 已闭环，但真实 authority/质量证据未就绪，所以 C1/C3 仍未通过。

### 2 分钟回答

v4 表面上像一个模型格式错误：Qwen 正确调用了 submission function，却把 `drafts` 数组二次序列化成字符串。Runner 拒绝完全正确。但只读诊断显示，里面 16 个错误不能全部算到模型头上。Pydantic 的“词法排序且唯一”自定义规则没有自动进入 provider schema，prompt 只说复制 TaskSpec；而 TaskSpec 有 6 个数组恰好不是词法顺序。模型按一份合同做，Runner 按另一份合同验。与此同时，Multi-Product draft 想 detect 后逐 crop image search，可 registry 的 `query_asset` binding 会拒绝 crop；真正安全的组合服务只藏在 evaluator，作者看不到，runtime 还有两套 authority。

我没有对已观察响应做 unwrap、排序或补字段，而是把错误分成 provider wire/key、作者语义映射和内部合同三类写入不可变 receipt。前瞻 v5 建立字段责任表：模型只输出 objective、步骤语义、工具/成功规则映射、fallback 文本和引用；compiler 从 TaskSpec 注入 coverage/output/fallback flags、step ID 与 canonical 顺序，继续检查全集和权限。

工具侧我把 `MultiProductSearchService` 放进 registry v2，成为只接收 authoritative asset 的第八个 composite ToolSpec；父图、detector box、私有 crop 和逐物 retrieval 都由 Runner 持有，传递 runtime/evidence 进入同一 snapshot。TaskSpec v1、Assistant lock/operator gate、production runtime lock 和 formal evaluator v2 一起升级；formal evaluator 不再接受额外 executor，内部 trace 也不会自动冒充用户可见 evidence。定向回归证明 generation 混配、任意 asset/path/bbox、未审批上传和 runtime drift会失败。当前仍缺真实 C2 工件、双锁 authority receipt、RPC gold 和新 authoring 调用，所以这是工程机制完成而非 formal gate 关闭。

### 证据入口

- `specs/authoring/llm-static-primary-20260724-structured-v1-adjudication-receipt.json`
- `specs/authoring/authoring-submission-tool-v1.json`
- `specs/task_specs/ecommerce-task-spec-v0.json`
- `specs/task_specs/ecommerce-task-spec-v1.json`
- `specs/authoring/authoring-freeze-candidate-v5.json`
- `src/skillchain/static_authoring.py`
- `src/skillchain/task_spec.py`
- `src/skillchain/tools/registry.py`
- `src/skillchain/tools/production_registry.py`
- `src/skillchain/tools/multi_product.py`
- `src/skillchain/tools/formal_evaluation.py`
- `src/skillchain/runners/assistant.py`
- `src/skillchain/evaluation/assistant_runs.py`
- `src/skillchain/evaluation/packets.py`
- `docs/plans/2026-07-20-p0-p1-closure.md`
- `tests/test_static_authoring.py`
- `tests/test_task_spec.py`
- `tests/tools/test_registry.py`
- `tests/tools/test_production_registry.py`
- `tests/tools/test_formal_evaluation.py`
- `tests/evaluation/test_phase4_assurance.py`

---

## 13. 模型可访问不等于 provider 级可审计；多模态支持也不等于现有图片传输可用

**状态：部分解决**

### 一句话问题

重新登录后 CLI 能列出目标模型，并不自动满足“单次 provider 调用、精确 revision、token/费用和输入隔离”这些实验要求；同理，Judge 模型标称支持视觉，也不表示仓库当前的 Base64 图片传输与该 provider 兼容。

### 背景与影响

项目所有者将 Assistant、Author 和 final Judge 分别改为 DashScope Qwen3-VL-Flash、Codex 内部 GPT-5.6-Sol/high 和 DashScope Kimi K3。若只改三个字符串，会产生三种隐蔽错误：历史 Qwen Author freeze 会被当前模型常量破坏；Codex CLI 会被包装成它无法提供的 provider-attested 回执；Kimi 会收到 Base64/data URL 或 Base64 文本，却被误认为看到了图片。这会直接破坏 RQ1b 的公平性和 final Judge 的有效性。

### 可观察症状与证据

已验证事实：

- Codex CLI `0.145.0` 已以 ChatGPT 模式重新登录，catalog 精确包含 `gpt-5.6-sol`，且原生支持 `high`；二进制 SHA-256 为 `83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c`。
- catalog/doctor 都不发生成请求；它们不能证明真实 inference entitlement。重新登录后的 doctor 整体为 OK：认证已配置、required HTTP endpoint 可达并返回 403、Responses WebSocket 握手成功。正式生成仍是唯一最终可用性验证。
- Codex CLI 不提供可独立核验的 served revision、provider request ID、底层 attempt/retry 或本项目可强制的 per-call token/费用上限；read-only/空 scratch 也不能证明全局只读到 packet。
- Kimi K3 是 only-thinking 模型，只允许 `reasoning_effort=max`；默认 temperature=`1.0`、top-p=`0.95`。DashScope 官方直供文档明确图片/视频只接受公网 URL，不接受 Base64。
- 仓库原 label synthesis 共用 `BACKBONE_*`；若直接把 Assistant 改为 Flash，会静默改变 Phase 3 数据生成模型。

### 根因

早期配置把“角色选择”“API provider 能力”“历史 freeze 可复现常量”和“数据合成默认值”混在一组全局常量中，并假设所有 OpenAI-compatible 接口共享相同的模型身份、解码、视觉编码和回执能力。实际上 execution surface 才决定可以证明什么：API、容器、Codex CLI 和不同多模态 provider 的证据面并不相同。

### 考虑过的方案与取舍

1. **只替换全局模型名：** 改动最小，但会破坏历史 Qwen 构建器，并伪造 Codex/Kimi 能力，淘汰。
2. **用额外 “hello” 调用验证 Sol：** 能证明推理大致可达，但会绕过或消耗“一次 Author session”预算，淘汰。
3. **继续把 Codex 塞进旧 API/container Author：** 可以复用代码，却必须虚构 endpoint、price、revision、usage 和 provider receipt，淘汰。
4. **按 execution surface 分协议：** 保留历史 Qwen 常量，新增 Codex session 协议和 Kimi provider 约束；证据较弱但诚实，采用。

### 最终方案

- Assistant、Author、final Judge、feedback evaluator 和 label synthesis 使用独立配置 namespace；历史 Qwen freeze/builders 固定读取 `LEGACY_AUTHOR_*`。
- Codex Author 单列 `codex_mediated_static_author_v1`、`platform-mediated_non-provider-attested`。freeze 绑定 CLI 二进制、模型/effort、stdin/schema、空 scratch、环境 allowlist、命令、Runner、单 session 预算和 create-only run/claim/receipt 路径；claim 必须在 process launch 前创建，任何失败永久消费。
- Codex 硬预算为 1 session、0 follow-up/repository retry/repair/fallback/visible tool activity、600 秒、64 KiB final、16 MiB event log、30 分钟人工审核。30k/6k/36k token 与 3000 microUSD 只作披露目标；不可观测字段明确写 `null/unavailable`，`formal_provider_call_eligible=false`。
- Kimi adapter 强制 max-thinking、1.0/0.95、无 seed、SDK `max_retries=0`，并在当前 local/Base64 图像路径上 fail closed。正式 Judge 必须等待公网 URL 资产 SHA/上传/下载复验/生命周期回执和专用 Runner。
- 首版 Codex/high v1 在 owner approval 前的绑定审计中被取代。v2 后来取得 exact owner approval，但在 claim/process 前因 frozen-PATH gate 拒绝；v3 在 preapproval P0 审计后废弃。截至本条最初记录时，v4 尚待批准；后续 v4/v5 真实结果与更新后的 lineage 见第 15 条。

### 如何验证

- `tests/test_codex_authoring.py` 验证角色锁、CLI/runtime/request/schema/路径双摘要、候选未授权、历史 Qwen freeze 可重建、进程监督以及 tool-event fail closed。
- `tests/test_llm.py` 验证 Kimi exact family、only-thinking 参数、非法 temperature/top-p/seed/local image 拒绝和 SDK 零重试。
- `tests/synthesis/test_labeling.py` 验证改变 Assistant 不会改变 label synthesis 调用。
- 历史 v2 freeze 的完整文件、payload 与 runtime 摘要均已固定，且 exact owner approval 仍作为历史记录保留；随后 preclaim incident 证明该批准没有进入 claim/Popen。v3 又固定绝对 binary/确定性 env，但独立审计在 approval 前阻断。v4 已在真实外部环境生成并复验：freeze file/payload SHA-256=`fb638df577a9dfc27174c56aab7ac9ad32d40ce8119f49ad923d83b8195e55a8` / `121cffbe2e8d055dd8ff49a47fd14f08138f261f5af80b80a06c7e43160be31d`，runtime file/payload=`1175921697491696c0ed365ed956e278206dd8d1d6152d78a00ebd63daf6b697` / `2140b23aa7d8167f96ef083aebd992c9abb9dbd7fcc2eb828b53264be97b0129`，source manifest file=`7a827309f7583a625af41aebb46eb3fbcba3c596da5e3af7569cc5535f48b8a3`。
- `codex-cli-model-access-evidence-v2.json` 与仓库外 prompt negative probe 分别固定重新登录后的连通性和项目指令未注入证据；v2/v3/v4 定向测试与独立审计只证明离线边界，不证明真实 Sol inference 或 Kimi visual Judge 已成功。

### 剩余限制

正式 Codex session 尚未执行；v2 的旧批准不能用于 v4。v4 candidate 已冻结并复验，但仍待新的 exact owner confirmation，且其输入隔离仍是行为约束，不是容器级证明。guard、v2 retirement、v4 approval、claim、output 和 receipt 当前均不存在。Kimi Judge 的公网 URL 资产协议、正式 Runner、隔离 lock、calib/audit 和真实回执均未完成。Assistant 仍需生成精确 production backbone lock。C1/C4 和 core NO-GO 均未关闭。

### 30 秒回答

我遇到的不是单纯“模型名改不了怎么办”，而是不同执行面的证据不等价。Codex CLI 能列出 Sol/high，但它不提供 provider request ID、served revision、底层重试和可强制费用；Kimi 支持视觉，却只接公网 URL，不接现有 Base64。我把角色、历史 freeze、标签合成和 provider transport 解耦，给 Codex 单建一次性 session 协议，所有不可观测字段明确降级；Kimi 当前图像路径直接 fail closed。这样牺牲了一部分结论强度，但没有伪造可复现性。

### 2 分钟回答

模型选型从 Qwen Author 改成 Codex Sol/high、Judge 改成 Kimi K3 后，最危险的做法是只改全局常量。旧 AuthoringInput 强制 HTTPS provider、日期 revision、seed、CNY 价格和 usage receipt；Codex CLI 不具备这些字段。与此同时，Assistant 和 label synthesis 原先共用 `BACKBONE_MODEL`，会让一次评测模型调整静默改变数据生成。Kimi 更隐蔽：官方说支持图像，但直供接口只接受公网 URL，仓库发的是 data URL 或把 Base64 塞进文本。

我先保留 `LEGACY_AUTHOR_*` 让 Qwen v3/v4/v5 历史逐字节可复验，再拆分 Assistant、label、feedback 和 final Judge。Codex v2 协议冻结 CLI 版本和二进制摘要、`gpt-5.6-sol/high`、exact stdin/schema、仓库外空 scratch、Runner、运行依赖闭包和一 session 预算；但 exact approval 后，完整 PATH 的可变 Desktop shim 仍让 preclaim 复验失败。v3 用绝对 binary/确定性 env 修复后，审计又发现 post-commit claim 和 post-claim terminal evidence 缺口。v4 因此再引入 conditional terminal guard 和严格 canonical-bundle override，仍不伪造 provider 证据。Kimi adapter 则强制 max/1.0/.95/no-seed 和 SDK 零重试，local/Base64 image 直接拒绝，等正式 URL 资产协议和 Judge Runner。

验证上，v4 代码、独立 P0/P1 审计和 candidate freeze 已经收口，但 owner approval 与调用尚未发生；Codex inference 总数仍为 0。我仍把 C1/C4 保持 NO-GO，因为模型 catalog、doctor、冻结候选和离线事务测试都不是一次真实推理，Kimi model selection 也不是有效视觉评价。这一处理展示的是实验系统中“可调用”“可审计”和“可用于结论”三个层级必须分开。

### 证据入口

- `specs/authoring/model-role-selection-v3.json`
- `specs/authoring/authoring-freeze-lock-codex-high-v2.json`
- `specs/authoring/authoring-codex-mediation-protocol-v2.json`
- `specs/authoring/codex-cli-model-access-evidence-v2.json`
- `specs/authoring/codex-cli-prompt-isolation-evidence-v1.json`
- `deploy/authoring/locks-codex-v2/runtime-lock.json`
- `deploy/authoring/locks-codex-v2/source-manifest.json`
- `scripts/build_codex_authoring_approval_package.py`
- `scripts/run_codex_authoring.py`
- `src/skillchain/codex_authoring.py`
- `src/skillchain/config.py`
- `src/skillchain/llm.py`
- `tests/test_codex_authoring.py`
- `tests/test_llm.py`
- `tests/synthesis/test_labeling.py`
- <https://help.aliyun.com/zh/model-studio/kimi-api-by-moonshot-ai>

---

## 14. “只允许一次”的模型调用需要预先存在的持久终态

**状态：治理机制已由 v4/v5 两次真实 session 验证；最新结果见第 15 条**

### 一句话问题

一次性调用不能只保证“claim 在进程前创建”：如果 claim 的原子提交被误判，或进程启动后在 receipt 发布前崩溃，授权可能已消费却没有可信终态。

### 背景与影响

Codex Author 预注册为 1 个 session、0 retry/follow-up/repair/fallback/tool，失败也消费授权。这让 Runner 本身成为实验 treatment。普通服务可以靠重试掩盖的 PATH 漂移、进程监督异常或半发布，在这里会把唯一一次机会变成无法区分“模型 bad case”和“执行器 bug”的黑洞。对求职项目而言，最重要的不是让调用勉强跑起来，而是让失败也能被第三方复核，且不能在看到失败后重新解释预算。

### 可观察症状与证据

已验证事实：

- Codex v2 已创建 exact owner approval，但 preclaim 环境复验发现完整 inherited `PATH` 含会变化的 Codex Desktop `.codex/tmp/arg0` 条目。Runner 在 claim 和 `Popen` 前拒绝，因此 v2 没有 output/receipt 或 inference；这证明“已批准”与“已调用”必须分开记录。
- v3 改为冻结绝对 CLI binary 和确定性环境，解决 PATH 漂移；但独立 preapproval 审计发现，hard-link 已提交后如果临时文件清理抛出 `BaseException`，调用方可能把已提交 claim 误判为未提交。审计还发现 post-claim 初始化、部分 process-result copy 和 canonical receipt 发布失败时没有独立持久终态。
- v3 因此在 approval、claim 和 inference 前废弃。v4 新增的测试覆盖 post-link 异常分类、两个 nonce claimant 只有一个能进入 fake launch、`Popen`/线程构造异常后的 kill/wait/reap、未回收/部分 process result 不得 canonicalize、guard 激活、完整 bundle、缺失/额外/被改写 receipt、非法事件、矛盾 status 和审批时 ambiguous create。
- v4 builder、approver、Runner 与 contract 已完成独立审计，当前没有未解决 P0/P1。candidate=`authoring-codex-high-20260724-v4`、run ID=`llm-static-codex-primary-20260724-high-v4`；packet/schema/request/runtime/freeze 已在真实外部环境 create-only 冻结并复验为 `frozen_candidate_awaiting_owner_confirmation`，但 `invocation_authorized=false`。guard、v2 retirement、approval、claim、output 和 receipt 尚未生成，Codex lineage 累计 inference 仍为 0。

待执行：

- 由 owner 对已冻结 candidate 的 exact model/effort、预算、全部摘要与路径、失败消费语义和信任边界重新确认。
- approver 在批准瞬间复验同一冻结闭包，再按 guard→v2 retirement→v4 approval 顺序创建 authority；随后才执行唯一 session。

### 根因

早期设计把“原子创建文件”和“最终一定有审计结论”混成一件事。文件系统原语可能在副作用已经发生后抛异常；进程生命周期又跨越 claim、spawn、pipe thread、kill/wait、解析、编译和目录发布多个故障点。只有成功路径的 receipt 不能证明失败路径，output 目录的 self-hash 也不能替代外部 authority。根因不是某个 `try/except` 少写一行，而是缺少一个在 claim 之前已经存在、能被 claim 激活的持久终态。

### 考虑过的方案与取舍

1. **复用 v2 approval，修正 PATH 后直接重跑：** 最省操作，但会把新执行环境塞进旧 authority，并在失败已知后改变 treatment，淘汰。
2. **只在 v3 修复 claim 清理异常：** 能解决 ambiguous commit，却不能覆盖 `Popen` 后初始化、监督、回收和发布故障，淘汰。
3. **发生异常后再尽力写 failure receipt：** 实现简单，但磁盘/代码异常可能正好让补写也失败，仍会留下无终态窗口，只能作为辅助而非根保证。
4. **approval 时预创建 conditional terminal guard：** 增加工件和验证复杂度，但在 claim 前就有可持久化的 fail-closed 结论；只有完整重验的 canonical bundle 能覆盖，采用。

### 最终方案

- v4 保留 v3 的绝对 binary、确定性环境、仓库外 scratch、递归 runtime/source closure、受限并发 I/O、严格 JSONL 状态机和可信 compiler replay。
- `classified_atomic_create` 在任何 `BaseException` 后稳定重读 destination：只有 exact bytes 已提交才返回 recovered-commit；不存在或出现第三方 bytes 都 fail closed。每次 claim 携带唯一 nonce，竞争者即使字节模板相同也不能同时获胜；recovered/cleanup 状态禁止 launch。
- owner approval 事务先创建 exact conditional terminal guard，再创建 v2 non-inference retirement，最后创建 v4 approval。旧 v2 approval 由此明确终止，不能与 v4 并存为两份可执行 authority。
- v4 claim 必须绑定 guard 的 exact file SHA；claim 一旦存在，guard 自动成为权威失败终态。未知或未回收进程、存活 pipe thread、未完整提交的 process result、异常、日志不完整或发布失败都不能覆盖 guard。
- 只有从磁盘独立验证固定 mandatory/optional 文件集、文件名和摘要、freeze→approval→guard→claim 链、source/runtime/request/schema/compiler binding、事件与 evidence、raw final snapshot、trusted compiler replay、进程终态和 receipt self-hash 的 canonical bundle，才能把终态从 guard 提升为具体成功或失败 receipt。
- raw final 在证据构造前只快照一次；编译、归档和 staging 复验都使用同一字节，避免“验证一份、发布另一份”的 TOCTOU。

### 如何验证

- `tests/test_codex_authoring_v4.py` 覆盖上述 claim、并发、进程回收、guard 和 canonical-bundle 正负路径；v2/v3 测试继续证明旧工件可重建且历史状态不被改写。
- `scripts/build_codex_authoring_approval_package_v4.py` 明确是 create-only、preapproval-only builder，不创建 guard/approval/claim/output/retirement，也不执行 inference；`scripts/approve_codex_authoring_v4.py` 才按固定顺序签发三份 authority 工件。
- 对 Runner/contract、authority builder/approver 和测试覆盖的独立只读审计已经完成，最终没有未解决 P0/P1。审计和测试均未调用模型。
- v4 freeze file/payload SHA-256=`fb638df577a9dfc27174c56aab7ac9ad32d40ce8119f49ad923d83b8195e55a8` / `121cffbe2e8d055dd8ff49a47fd14f08138f261f5af80b80a06c7e43160be31d`；runtime file/payload=`1175921697491696c0ed365ed956e278206dd8d1d6152d78a00ebd63daf6b697` / `2140b23aa7d8167f96ef083aebd992c9abb9dbd7fcc2eb828b53264be97b0129`；source manifest file=`7a827309f7583a625af41aebb46eb3fbcba3c596da5e3af7569cc5535f48b8a3`。
- candidate 前瞻绑定 planned guard/v2 retirement file SHA-256=`47d1f697e43d313deb70d06916364699c1626c138b7499114d887b973a432210` / `1424cca5de3ae8477fc35fab7e6fe06d99923c49697fa9b9d6aaf1e5291b68b3`；两份文件尚不存在，不能把 expected hash 当作已签发 authority。
- 当前 v4 定向回归为 `24 passed`，v2/v3/v4 联合回归为 `48 passed`，相关 core 回归为 `130 passed, 1 skipped, 2 deselected`；测试没有调用模型。扩大回归在排除当前环境缺少 `huggingface_hub`、无法收集的 OpenCLIP artifact 单测后首轮通过 1,069 项，另外 23 项均因 Windows 目录原子 rename 的间歇性 `WinError 5` 失败或报错；定向复跑通过 22 项，最后一项再次单独通过。这证明没有留下可复现的逻辑失败，但不是一次全仓全绿证据，也没有覆盖 OpenCLIP 可选路径。

### 剩余限制

- self-hash 证明的是“这组字节内部自洽”，不是“由谁创建”或“谁有权批准”。当前威胁模型覆盖意外异常和遵守协议的并发 Runner，不覆盖同一用户恶意重写工件、特权账户篡改或外部身份冒充。
- conditional guard 不能抵御全盘/存储同时丢失、文件系统不满足预期原子性、断电破坏持久性或磁盘控制器撒谎；这些需要独立存储、签名/透明日志或更强基础设施。
- Codex CLI 的 base/global instructions、served revision、provider request ID、平台内部 attempt/retry、精确 token 与费用仍不可由仓库强制或证明。即使 v4 成功，也只能报告 `platform-mediated_non-provider-attested`。
- v5 后续已取得 pre-review draft；owner `wenxi_0726` 也已用 5 分钟完成六项 checklist 并接受
  unchanged draft。仍没有可用 LLMStatic Bank；最终 compile 依赖真实 C3 authority-issued
  registry runtime，因此完整 C1 和 core 继续 NO-GO。

### 30 秒回答

我在一次性 Codex Author 调用里发现，claim-before-spawn 仍不够：原子 link 可能已提交却在清理时报错，进程也可能启动后在 receipt 发布前崩溃。v2 实际在 preclaim PATH gate 停下，v3 又被审计出这两个 P0，所以都没有消耗 inference。我在 v4 里让 owner approval 先创建 conditional terminal guard，claim 绑定并激活它；只有从磁盘完整重验的 canonical bundle 才能覆盖。后续 v4 负结果与 v5 成功 bundle 都验证了这套机制；owner review 已完成，完整 C1 仍等待 C3 后 Bank compile。

### 2 分钟回答

这个问题的核心是，一次性预算把基础设施故障变成实验语义。v2 已经获得 owner approval，但 Runner 在 claim 和进程启动前发现 inherited PATH 里有动态 Codex Desktop shim，于是 fail closed；这很好地证明 approval 不等于 inference。v3 改用绝对 binary 和确定性环境，但 preapproval 审计继续发现：hard-link 已提交后清理异常可能让调用方误判 claim 未提交，而且 claim 后的初始化、进程监督和发布仍可能在 receipt 前中断。

我没有靠“异常后尽力补 receipt”解决，因为出错的正是写工件这条路径。v4 在 approval 阶段先放一份 exact conditional terminal guard，再关闭 v2 旧 authority并创建 v4 approval。claim 带唯一 nonce并绑定 guard SHA；它一出现，guard 就是 fail-closed 终态。Runner 只有在进程已回收、pipe thread 全部结束、完整 process result 已提交，且事件、raw final、compiler replay、固定文件集和整个摘要链都从磁盘独立验证后，才允许 canonical bundle 覆盖 guard。self-hash 只当完整性，不当身份认证；同用户恶意篡改、全盘故障和平台内部 retry 明确排除。当前 v4 代码、审计和 candidate freeze 已完成，但 owner approval 和真实调用尚未发生，所以我只声称“候选与治理边界准备好”，不声称 C1 已通过。

### 证据入口

- `specs/authoring/codex-high-v2-preclaim-environment-rejection-v1.json`
- `specs/authoring/authoring-freeze-lock-codex-high-v4.json`
- `deploy/authoring/locks-codex-v4/runtime-lock.json`
- `deploy/authoring/locks-codex-v4/source-manifest.json`
- `scripts/build_codex_authoring_approval_package_v4.py`
- `scripts/approve_codex_authoring_v4.py`
- `scripts/run_codex_authoring_v4.py`
- `src/skillchain/codex_authoring_v4.py`
- `tests/test_codex_authoring_v4.py`
- `docs/go-no-go/2026-07-20-core-no-go.md`

---

## 15. 隐藏的 validator 词汇约束会把语义正确的 LLM 输出变成系统性 bad case

**状态：已验证；真实 v4 负结果与 v5 成功结果均已留证**

### 一句话问题

TaskSpec 和 ToolSpec 要求 Author 理解对象检测的 `label` 字段，但 trusted compiler 又把独立词 `label` 当作评测污染词拒绝；prompt 没有披露这条输出约束，导致语义正确的真实响应被系统性判错。

### 背景与影响

Codex v4 是第一个实际进入 `gpt-5.6-sol/high` Author session 的候选。它覆盖六个 capability、使用正确工具和 success rule，并正常通过 CLI Structured Outputs 外层约束；但本地 Pydantic 在两条步骤文本上拒绝整个 draft。若把这类失败简单归因为“模型不听话”，后续会通过盲目重试、换模型或放宽 compiler 污染实验。若事后改写原响应，又会破坏一次性基线的可审计性。

### 可观察症状与证据

- v4 receipt 状态为 `codex_session_completed_draft_rejected`，`failure_stage=trusted_compiler`；进程 exit 0，未发生 tool/retry/follow-up/repair/fallback。
- raw output 只有两处独立词 `label`：视觉百科与食谱步骤都在提醒“检测类别只是预测”，语义上反而遵守安全约束。
- 只读 A/B 不改能力、工具、规则或结构，只把独立 `label(s)` 改写为 `predicted class name`；同一 payload 随即通过当前 Pydantic 和 trusted compiler，生成六能力 bundle。
- v5 用新的 freeze/approval/claim 只前瞻增加允许措辞，真实 session 状态变为 `codex_session_completed_draft_ready_for_review`、`formal_codex_session_eligible=true`。raw/pre-review/bundle SHA-256 分别为 `90fad5c27e6bb8a9122b470cfdef5a427bb4c0eb793d52ce4ffb52ed89b73a6b`、`e0c26c1c054514bfc4629fc4e84557da63f6b05335ca6c3b8709efe0a0e08908`、`41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3`。

### 根因

污染防护把“保留词扫描”实现为输出 schema 之外的隐藏语义 validator。Author 的可见输入却同时包含使用该词的 TaskSpec 和 ToolSpec。于是输入契约允许、甚至需要理解一种概念，而输出契约暗中禁止最自然的表达。JSON Schema 只能保证结构，无法提示这条自定义词法规则；prompt 又没有提供等价安全措辞，形成不可满足或高度脆弱的跨层契约。

### 考虑过的方案与取舍

1. **直接改写 v4 raw output：** 最快，但会把已消费授权的负结果伪装成成功，淘汰。
2. **放宽全局污染扫描，允许 `label`：** 能消除当前误报，但扩大真实评测信息泄漏面，并改变所有 authoring 路径，淘汰。
3. **自动 repair 或同 run 重试：** 会让静态基线变成结果驱动迭代，破坏预算与 treatment，淘汰。
4. **新 ID/freeze 前瞻披露允许措辞：** 保留 scanner、防止改写历史，只消除 Author 看不见的契约冲突；代价是多一代审计工件，采用。

### 最终方案

- v4 的 freeze、approval、guard、claim、raw output 和 canonical receipt 原样保留。
- v5 freeze 逐字节绑定完整 v4 历史，并声明 `v4_result_rewritten_or_reclassified=false`。
- v5 只在 Author 可见 prompt 增加：对象检测类别预测一律写作 `predicted class name`，不要复制 schema 字段名；模型、预算、TaskSpec、ToolSpec、公开资料、Reference Skill 和 runtime 均不变。
- 每个 run 仍只有一个 session，0 repository retry/follow-up/repair/fallback/tool；若失败必须换新 namespace 和 freeze。

### 如何验证

- `tests/test_codex_authoring_v5.py` 验证 v5 除 prompt 外的关键 packet 字段与 v4 相同，v4 结果仍被绑定，词汇 A/B 能解释原失败。
- v4/v5 定向回归为 `24 passed, 2 skipped`；v5 发布后的独立 canonical replay 为 `3 passed`。
- v5 canonical validator 从磁盘重新验证固定文件集、authority chain、runtime/source/request/schema、事件、raw final、trusted compiler、进程回收与 receipt self-hash。
- v5 receipt 记录 34,455 input、1,515 output token，1 个 final，0 visible tool/retry/follow-up/repair/fallback。
- owner `wenxi_0726` 用 5 分钟完成六项 checklist并接受 unchanged draft；外置 review
  receipt file SHA=`938c38fe5ffd90dedabe09a53f49a6b54c2e5bb2b7d4cd7ab0f22d8f2000d738`，
  写入并独立复载后原 canonical bundle 重放仍为 true。

### 剩余限制

- Codex CLI 仍不能提供 served revision、provider request ID、平台内部 attempt 或费用；证据等级只是 `platform-mediated_non-provider-attested`。
- 项目所有者人工 checklist 已完成；`reviewer_id` 是可问责记录，但不是外部身份签名证明。
- pre-review draft 还不是可执行 Bank；必须等待真实 C3 authority-issued registry runtime 后确定性编译。
- 词法扫描仍只是 defense-in-depth，不能证明语义上绝无评测污染。

### 30 秒回答

真实 Codex Author 第一次输出其实结构、工具和规则都正确，却被 compiler 拒绝，因为 TaskSpec 里允许的 `label` 在输出侧被隐藏 scanner 当成评测污染词。我的关键判断不是修响应或重试，而是先做只读 A/B：只换两个词，同一 payload 就通过，从而证明是跨层契约冲突。我保留 v4 负结果，用新 v5 freeze 只把允许措辞写进 prompt，其他 treatment 不变。v5 随后一次生成合格六能力 draft，完整 receipt 可独立重放。

### 2 分钟回答

这个问题展示了 Structured Outputs 的边界：JSON Schema 能约束对象结构，却看不到 Pydantic 自定义 validator。项目的污染防护把 `label` 视为实验元数据，但 TaskSpec 和 detector ToolSpec 又合法使用这个字段；模型自然写出“不要把 detector label 当成真实身份”，反而因为安全表述被拒绝。v4 进程和事件都成功，只有两处词法命中。

我没有把失败归咎于模型，也没有自动 repair。先在离线诊断里保留原字节，只做单变量替换；六个 draft 随即全部通过。这给了足够强的因果证据。然后建立 v5：新 candidate、run ID、freeze、approval、guard 和 claim，绑定 v4 receipt，明确禁止改写或重分类历史。v5 prompt 只提供 `predicted class name` 这个安全等价措辞，模型、预算、TaskSpec、ToolSpec、资料与 runtime 全不变。

真实 v5 session exit 0，生成 strict-validated pre-review draft；canonical validator 又从磁盘
重放完整证据链。owner 随后在 5 分钟内完成六项 checklist并接受 unchanged draft，外置回执
没有改变原 run。这个修复同时守住两个边界：不为通过测试而放宽防泄漏，也不让模型为看
不见的规则背锅。剩余工作仍要诚实区分：人工门已关闭，但 C3 runtime 后 Bank compile 尚未
完成，所以不是完整 C1 或实验效果通过。

### 证据入口

- `specs/authoring/authoring-freeze-lock-codex-high-v4.json`
- `runs/formal-authoring/llm-static-codex-primary-20260724-high-v4/invocation-receipt.json`
- `specs/authoring/authoring-freeze-lock-codex-high-v5.json`
- `specs/authoring/llm-static-codex-primary-20260724-high-v5-owner-approval.json`
- `runs/formal-authoring/llm-static-codex-primary-20260724-high-v5/invocation-receipt.json`
- `runs/formal-authoring/llm-static-codex-primary-20260724-high-v5/pre-review-draft.json`
- `src/skillchain/codex_authoring_v5.py`
- `scripts/run_codex_authoring_v5.py`
- `tests/test_codex_authoring_v5.py`

---

## 16. 断点续传不只是追加字节：必须锁住远端身份和授权边界

**状态：已验证**

### 一句话问题

只按本地文件长度发送 Range，会在上游对象变化、服务器忽略 Range 或两个会话并发时产生“大小正确但内容拼错”的 RAW 文件，还可能把签名 URL 写进长期状态。

### 背景与影响

MVP/full 数据下载会跨多个 Codex 会话，部分公开对象超过 100 GB，full-upstream 甚至达到数 TB。网络中断和主动停止是正常路径，不能靠一次会话跑完；但 RAW 字节又是后续 source lock、清洗和正式 catalog 的根信任。如果恢复协议只看“已经下载 N 字节”，旧版本前 N 字节与新版本剩余字节可能被静默拼接。授权源的 URL query 还可能包含临时 token，若直接写入状态或日志会扩大凭据暴露面。

### 观察到的证据

- 项目旧下载函数多数只使用 `Range: bytes=N-`，仅按最终大小判断；有的能在服务器返回 `200` 时重写文件，但没有跨会话远端身份状态。
- source portfolio 同时包含稳定公开对象、Hugging Face mutable `main`、实时 API 快照和必须点击授权的数据源，不能用同一种“发现 URL 就下载”的信任假设。
- 现有约 7.6 GiB raw cache 可复用，但关键 MVP 主源仍缺失；删除重来浪费已取得字节，把任意现有文件直接标完成又缺少可审计依据。
- 2026-07-23 至 24 日实际恢复 42,446,704,640 字节的 ABO spins 时，aria2 的 `.part` 逻辑长度曾接近或达到最终大小，但 piece bitmap 仍显示缺片，且项目状态还停留在旧的 3,917,479,936 字节。最终以单一 aria2 进程、控制位图和 `.part.aria2` 消失为准，才交给项目下载器转正。

### 根因

早期实现把断点续传当作传输优化，而不是数据身份协议。它没有区分“目标路径”“远端对象身份”“授权获取方式”和“下载完成后的内容证明”，也没有定义两个进程同时恢复、服务器不支持 Range、签名 URL 过期时的失败语义。

### 考虑过的方案与取舍

1. 直接复用各 adapter 的下载函数：改动最少，但状态格式、校验强度和授权处理不一致，无法统一回答 full 的完成度。
2. 使用通用下载器或云盘客户端：成熟但难以把 profile、source role、人工阻断和项目状态统一绑定，也未必能避免凭据进入日志。
3. 每次从头下载并最后算哈希：最简单可靠，但对百 GB/数 TB 文件不可接受。
4. 建立项目级 manifest、远端身份状态和原子 `.part` 协议：代码更多，但能复用旧字节、显式失败并支持独立会话继续。

### 最终方案

新增统一 RAW 下载编排器和三层 profile：`mvp`、项目有界 `full`、字面上游镜像 `full-upstream`。每个 HTTP 文件写相邻 `.part`，记录去敏后的 URL、长度、ETag 与 Last-Modified；恢复时用 `Range + If-Range`，严格验证 `206 Content-Range`，服务器返回 `200` 或没有可用的远端校验器就从头覆盖 partial。远端身份在 partial 存在时变化则 fail closed，最终通过冻结大小和可用 SHA-256 后原子发布。状态定期刷盘并用 profile 独占锁防并发。若外部 aria2 的 `.part.aria2` 仍存在，项目下载器也 fail closed，因为 aria2 的 piece bitmap 允许非连续写入，不能把逻辑文件长度当作连续断点。点击授权、许可不明或需要人工选择 revision 的来源明确标记 `BLOCKED`，只能由 Git 忽略的本地 overrides 接力；状态不保存 URL query、fragment 或 userinfo。

### 如何验证

- `tests/data/test_raw_download.py` 模拟正常下载、`206` 恢复、Range 被忽略、远端身份变化、摘要不一致和锁冲突，全程不访问网络。
- `tests/test_raw_download_profiles.py` 验证 `mvp`/`full` 对 source portfolio 的层级覆盖、公开 artifact 字段和人工来源的明确阻断。
- CLI 的 `plan/status/verify` 可分别在不下载、跨会话盘点和内容复验阶段使用；真实受限源仍需外部授权，不能由单元测试宣称完成。
- 初始 MVP 运行得到 14 个公开 artifact、51,498,470,482 字节和 2 个 command 快照全部完成；当时 `verify` 全部通过，RPC、FashionIQ、JDDC 2.0、CORD v2、SROIE 保持 5 个 `BLOCKED`，未伪造完成。随后在 2026-07-24 人工取得并本地验证 SROIE/CORD/FashionIQ，新增 DuRecDial 2.0/CrossWOZ 固定快照，并完成 RPC Kaggle v5 的 27,205,167,166 字节本地包。最终 MVP profile 的 34 个活动项全部完成；JDDC 2.0 按 owner 决策永久退出。这些变化都只关闭 acquisition，许可/PII/use review、外部 source lock 与正式数据门仍保持阻断。

### 剩余限制

MVP RAW 的 34 个活动项已全部完成，但若干来源的 formal gate 仍未通过；这不等于 selection、许可/PII、正式 source lock 或 Asset/KB catalog 完成。RPC 的旧 Kaggle URL 在完整 partial 已落盘后返回 404，因此下载器新增了严格的本地 partial 收尾路径：只有 `.aria2` 控制文件不存在、冻结字节数匹配且 SHA-256 通过时才原子转正；收据还记录 ZIP central directory、条目数和不安全路径检查。该摘要是本地 acquisition receipt，不是发布方摘要，正式 source lock 仍需独立证据。实时 API 的候选快照由已有 adapter 管理，自定义 `--root` 不会自动重定向这些 adapter。Recipe1M+、Open Images、Products-10K 等 full/core 来源的大小和许可仍要在批准后冻结，不能由估算替代。

### 30 秒回答

我把跨会话下载视为数据身份协议，而不只是 Range 追加。ABO 的 42.4 GB 实际恢复还暴露了 aria2 预分配陷阱：文件长度可看似完整，但 piece bitmap 仍缺片。新编排器用 `.part`、远端身份锁、原子改名和 profile 独占锁恢复，并在 `.part.aria2` 存在时拒绝转正。初始 14 个公开 artifact 和 2 个 command 快照完成；随后固定 DuRecDial/CrossWOZ 后增至 16 个。授权源始终明确阻断或按 owner disposition 退出，没有用镜像猜测填平状态。

### 2 分钟回答

这个问题的关键是“可继续”不能牺牲“下载的还是同一个对象”。项目的数据从几百 MB 到数 TB，跨会话中断一定会发生；如果只保存 N 字节并在下次发送 Range，源站更新对象后仍可能返回后半段，最终大小甚至完全正确。另一方面 RPC、U-NEED、Recipe1M+ 这类来源要授权，签名 URL 可能带 token，不能被普通状态文件长期保存。我比较过沿用各 adapter、每次从头下和统一编排器，最终选择后者：清单明确 mvp、项目 full 和 full-upstream，避免把 4,500 条评测误解为必须镜像数 TB；文件先写 `.part`，状态记录长度、ETag、Last-Modified 和去敏 URL；恢复发 If-Range 并核对 Content-Range，身份漂移就停，服务器不支持 Range 就安全重写；完成后校验大小/摘要再原子改名。profile 锁阻止两个会话同时写。公开字节可以自动跑，授权源必须通过 Git 忽略的 overrides 显式接力；永久放弃的来源则保留 retired disposition 并从 profile 移除。这样旧缓存可复用，失败可解释，但下载完成仍不等于许可证、PII、source lock 和正式 catalog 已通过。

### 证据入口

- `src/skillchain/data/raw_download.py`
- `scripts/download_raw_datasets.py`
- `specs/data_sources/raw-download-profiles-v1.json`
- `docs/data-download-runbook.md`
- `tests/data/test_raw_download.py`
- `tests/test_raw_download_profiles.py`

---

## 17. 数万张小图的“断点续传”应以已验证对象为单位
**状态：部分解决**

### 一句话问题

FashionIQ 不是一个大压缩包，而是 77,683 个外部图片 URL；把清单直接交给通用
批量下载工具，会同时丢失类别/ASIN 身份、失效链接替代关系、图片有效性和可审计
失败集合。

### 背景与影响

Style Recommendation 的主要监督来自 FashionIQ 的参考图、相对语言和目标图。
官方标注与 URL 清单分属两个仓库，图片又位于外部 Amazon 域名。下载可能跨机器
重启和多个 Codex 会话，单个对象失败很正常；如果用“进程退出码为零”或“文件
存在”作为完成判断，HTML 错误页、零字节文件和遗漏对象都可能悄悄进入后续
selection，最终让 style gold 无法复现。

### 观察到的证据

- 固定提交的三份清单分别有 19,087、31,728、26,868 条，共 77,683 条；
- 所有 URL 都使用 HTTP，分布在 `ecx.images-amazon.com` 和
  `g-ecx.images-amazon.com` 两个域名；
- 官方 metadata 仓库另附 13 张 `broken_links` 替代图片；
- 官方示例下载脚本引用另一个数据集的 TSV，关闭 TLS 校验，也没有原子发布、
  重试、失败清单或跨会话状态，因此不能直接用于这份清单。
- 两次完整重试都得到同一组 2,416 个 `(category, asin, source_url)` 失败身份，
  两组集合差异为 0；这些目标均没有可解码的现存正式图片。
- 最终本地覆盖分区为 75,267 张接受图片加 2,416 条显式排除，二者互斥且总和恰好
  等于固定清单的 77,683 条。

### 根因

这是“许多独立、可变、可能失效的远端对象”问题，不是单一连续字节流问题。
传输层续传、数据集身份和正式用途许可也是三件不同的事：图片全部落盘仍不能替代
license review 或 source lock。

### 考虑过的方案与取舍

1. 直接使用官方示例：代码少，但输入不匹配且关闭 TLS 校验，不能审计失败。
2. 把全部 URL 交给 aria2：吞吐高，但需要额外维护命名、类别、替代图、图片解码
   校验和失败 manifest，且容易把稀疏/临时状态误当正式文件。
3. 每次从头重下：实现简单，但浪费数万次成功请求，重启成本不可接受。
4. 以单图为恢复单元：多做一层校验和状态管理，但已有成功对象天然形成稳定断点，
   与外部服务器是否支持 Range 无关。

### 最终方案

新增 `skillchain.data.fashioniq` 专用下载器。它严格解析固定提交的三份清单，将 HTTP
仅在请求时升级为 HTTPS 并限制域名；每张图片写唯一 `.part`，用 Pillow 验证可解码
后原子改名。官方 13 张替代图优先复用。每次启动先跳过已经验证的目标，损坏对象
重新下载；单进程锁防止两个会话写同一目标。运行快照与最终失败集合分别原子写入
`summary.json` 和 `failures.jsonl`。真实首轮以 6 并发启动，避免与 RPC、CORD 和
浏览器内 SROIE 下载争抢全部带宽。两次完整重试仍得到同一 2,416 条失败后，按 owner
指令将这些身份写入独立的 `exclusions.jsonl` 处置台账；下载器以后跳过它们，并把
状态记为 `complete_with_exclusions`，而不是伪装成成功或继续无限重试。

### 如何验证

`tests/data/test_fashioniq.py` 在无网络条件下验证清单约束、HTTPS 传输地址、官方
替代图、原子发布、再次运行跳过、失败后重试、排除台账生成和完整覆盖校验。
2026-07-24 的真实收口验证得到：`accepted_images=75267`、`excluded=2416`、
`inventory_total=77683`、`status=complete_with_exclusions`，且排除台账 SHA-256 为
`db2a762605994887b68f24179b33137df87182f2e5ef8ddd6f8d161ae6ff5d77`。
统一 MVP 下载协议也通过本地 command verifier 接管 FashionIQ 状态。

### 剩余限制

这 2,416 条是明确的数据处置，不是被找回的图片；若未来要提高覆盖率，应建立新的
受审查 acquisition 版本，不能静默改写现有台账。上游许可文字目前没有在仓库中固定
到明确版本，必须继续作为 formal-use blocker。还需生成外部 source lock、做许可
审批、selection/disposition、泄漏审计与 AssetCatalog，才能用于正式评测。因此
`formal_ready=false`，RAW acquisition 收口不能被误报为正式数据可用。

### 30 秒回答

FashionIQ 是 7.7 万个小对象，不适合把文件存在当完成。我做了按单图恢复的下载器：
固定清单身份、HTTPS 域名约束、`.part` 加图片解码校验后原子发布、官方失效链接
替代、失败 manifest 和单进程锁。两次完整重试后 2,416 条失败集合完全相同，所以
将其写入显式排除台账；75,267 个接受对象加排除项完整覆盖 77,683 条，但许可和
source lock 仍是独立 gate。

### 2 分钟回答

官方 FashionIQ 标注不带图片，另一个固定提交给出 77,683 个 ASIN 到 Amazon URL 的
映射，还有 13 张失效链接替代图。官方示例实际上引用别的数据清单、禁用 TLS 校验，
也没有失败恢复。如果直接批量抓取，重启后很难判断哪些文件完整，HTML 错误页甚至
可能带 `.jpg` 后缀进入训练。我把恢复粒度改成单图：严格解析类别与 ASIN，只允许
清单里的两个域名，将 HTTP 传输升级为 HTTPS；响应先写线程唯一的临时文件，限制
大小并用 Pillow 验证，再用原子改名发布。重复运行会验证已有文件并跳过，坏图和
失败项重试；官方替代图复用；运行快照与失败集合都原子落盘，独占锁阻止并发会话
污染目标。两次完整运行后，同一 2,416 个身份仍失败且没有有效落盘图片。我没有删除
记录、把失败计成成功或无限重启，而是按 owner 决策生成带摘要的排除台账，让验证器
强制检查接受集与排除集互斥且并集等于全部 77,683 条。最终是 75,267 张接受图片和
2,416 条排除，状态明确为 `complete_with_exclusions`。关键的工程判断仍是把“RAW
处置完成”和“可正式使用”分开：许可版本、source lock、selection 和 catalog 尚未
通过，所以 formal-ready 必须 fail closed。

### 证据入口

- `src/skillchain/data/fashioniq.py`
- `scripts/download_fashioniq_images.py`
- `scripts/finalize_fashioniq_exclusions.py`
- `scripts/verify_fashioniq_acquisition.py`
- `tests/data/test_fashioniq.py`
- `data/raw/fashioniq/download-state/summary.json`（Git 忽略的本地运行证据）
- `data/raw/fashioniq/download-state/exclusions.jsonl`（Git 忽略的本地处置台账）
- `docs/data-download-runbook.md`

---

## 18. 浏览器下载完成后仍要区分“官方包”“导出副本”和无关隐私文件
**状态：部分解决**

### 一句话问题

一个“已下载”目录里可以同时出现官方数据、Google Drive 重复副本和完全无关的用户
数据导出；若只按 ZIP 扩展名登记，会把重复样本和隐私文件一起送入正式数据链。

### 背景与影响

SROIE 通过登录后的浏览器取得，无法依赖普通 HTTP 下载器冻结远端 ETag。用户把下载
结果放入指定目录后，共出现 7 个 ZIP。后续 Document Reading gold 需要可靠的收据
图、OCR 框和 company/date/address/total 字段；错误地计入重复文件会改变样本权重，
而误收用户会话导出则会扩大严重的隐私与信任边界问题。

### 观察到的证据

- 5 个语义唯一官方包总计 1,205,755,849 字节，ZIP CRC 与路径安全检查均通过；
- 两个训练 ZIP 分别有 1,547 和 1,611 个物理成员，但去掉 Drive 生成的 `(1)` 至
  `(5)` 后缀后，均为 626 张图与 626 份标注；所有重复逻辑组的内容摘要一致；
- 两个文本测试 ZIP 的成员名称与逐成员 SHA-256 完全一致，只有 ZIP 内部顺序不同；
- 测试图包有 360 张图，文本包有 361 份标注，唯一无图文本是
  `X51006619570.txt`；
- `data-*-batch-0000.zip` 的成员是 `users.json`、`projects/...json`、
  `memories.json` 和 `conversations.json`，来源是 Claude 用户数据导出，不是 SROIE。

### 根因

浏览器和 Google Drive 的导出层按文件名打包，不保证归档字节、成员顺序或副本命名
稳定，也不会替项目做数据集边界判断。原始下载编排器又只支持远端 HTTP artifact
和人工 `BLOCKED` 两种状态，无法诚实表达“文件已由用户取得，只能离线验证”。

### 考虑过的方案与取舍

1. 登记目录里全部 ZIP：最快，但会重复文本测试包并纳入无关隐私导出。
2. 只看文件名中的 626/361：简单，却会掩盖物理成员重复和 360/361 配对缺口。
3. 解压后手工删副本：目录看起来干净，但破坏 RAW、丢失上游异常证据且不可重验。
4. 保留原包并建立 acquisition manifest：多一层协议和校验器，但能把上游原貌、
   逻辑样本口径、排除规则与 formal gate 分开。

### 最终方案

新增 `sroie-browser-acquisition-v1.json`，只列 5 个语义唯一包，固定本地字节数、
SHA-256、物理/逻辑成员数、重复组和测试配对异常，并显式列出重复 ZIP与 Claude
导出的排除原因。`verify_sroie_raw.py` 验证摘要、CRC、ZIP 路径、Drive 副本内容和
跨包配对。通用下载协议增加 `local_artifacts`：它不联网、不声称可重下，只验证
本地已取得文件并将历史 manual 状态替换为逐 artifact `complete`。RAW 不解压、不
删除，排除文件也不进入 manifest 或 Git。

### 如何验证

- `uv run python scripts/verify_sroie_raw.py` 对真实 5 包返回
  `locally_verified_raw_not_formal_source_lock`；
- 带本地 overrides 的统一 `verify` 对 5 个包全部返回 `[ OK ]`，随后统一状态由
  `sroie:__manual__ blocked` 变为 5 个 complete artifact；
- `tests/data/test_sroie.py` 覆盖同内容 Drive 副本与冲突副本 fail-closed；
- `tests/data/test_raw_download.py` 覆盖 `local_artifacts` 离线接管和陈旧 manual
  状态清理与失败时保留人工阻断；相关测试全部通过。

### 剩余限制

这些 SHA-256 是取得文件后的本地收据，不是来自发布方或独立审批记录的外部信任根。
RRC 页面与浏览器下载仍未固定明确许可证版本和允许用途；收据中虽称敏感字段经过
模糊处理，项目仍需逐源 PII/use review。当前还没有 SROIE adapter、
selection/disposition、AssetCatalog、人工 gold 或 leakage 审计，因此不能进入正式
评测。

### 30 秒回答

我没有把浏览器目录里的 ZIP 全部算作 SROIE。核验发现训练包有 Google Drive 同内容
副本、两个测试文本包语义重复，还有一个完全无关的 Claude 会话导出。我保留 RAW
原貌，用 acquisition manifest 固定 5 个唯一包的 SHA、CRC、逻辑样本数和配对异常，
再新增只做离线验证的 `local_artifacts` 接口。这样 RAW 状态可以诚实转为已取得，
但许可、PII 和外部 source lock 仍保持阻断。

### 2 分钟回答

SROIE 是登录后通过浏览器和 Google Drive 导出的，目录里 7 个 ZIP 不能直接视为
7 个数据包。我先做逐包 SHA、CRC、路径和成员分析。两个训练包物理成员明显超过
文件名中的 626，但把 Drive 的 `(1)` 到 `(5)` 副本归一后，恰好各有 626 张图与
626 份标注，而且每个副本组内容一致；两个文本测试 ZIP 逐成员完全相同，只是归档
顺序不同；测试图实际 360 张而文本 361 份，有一个明确孤立 ID。更重要的是，另一个
ZIP 是 Claude 的用户、记忆和会话导出，属于隐私文件，绝不能靠扩展名混入数据集。
我没有清洗或删除 RAW，而是建立机器可读 acquisition manifest，锁住 5 个唯一包、
逻辑口径和排除规则，并写 verifier 重跑 CRC、摘要、重复组与跨包配对。下载协议补充
`local_artifacts`，用于“用户已经取得、不能假装有稳定远端 URL”的对象，只离线校验
并更新统一状态。这个方案关闭了 SROIE 的字节缺口，同时明确保留许可证、PII、
外部 source lock、selection 和 catalog 等 formal gate。

### 证据入口

- `specs/data_sources/sroie-browser-acquisition-v1.json`
- `src/skillchain/data/sroie.py`
- `scripts/verify_sroie_raw.py`
- `src/skillchain/data/raw_download.py`
- `tests/data/test_sroie.py`
- `tests/data/test_raw_download.py`
- `docs/data-download-runbook.md`

---

## 19. 缺失的真实语料不能靠未标记的 Mock 悄悄替代

**状态：部分解决**

### 一句话问题

JDDC 2.0 无法取得时，直接用 LLM 生成“像电商客服”的对话虽然能让流程继续，
却会把机制复现、真实流量外部有效性和 synthetic provenance 三件事混为一谈。

### 背景与影响

JDDC 2.0 原本被规划为中文真实电商、多轮、图像与商品知识结合的语言轨迹来源，
论文规模约为 24.6 万会话、300 万条话语和 50.7 万张图片。当前项目无法取得该包，
owner 决定永久放弃它。若只删掉下载阻断却继续沿用“真实中文电商轨迹”表述，
MVP 会在不知情的情况下把合成样本当真实用户日志，报告也可能错误声称保留了
JDDC 的流量规模和多模态用户分布。

### 观察到的证据

- 原 source portfolio 把 `jddc_2_0` 列为 MVP language source，RAW profile 也一直
  将它保留为人工 `BLOCKED`。
- 仓库已有受控 Phase 3 轨迹工作流，但生成 manifest 原先只有模型 provenance，
  没有直接声明 `synthetic_derived`。
- DuRecDial 2.0 官方固定提交包含中英平行推荐对话，数据条款明确为
  CC BY-NC-SA 4.0；CrossWOZ 固定提交包含 6,012 个中文多轮会话及动态 goal、
  no-offer、goal-change 和 multi-query 结构。
- 2026-07-24 已下载并验证 DuRecDial 2.0 commit
  `1309cc072afd0b832e899e49c62906e700ff6acf`（10,112,093 字节）和 CrossWOZ
  commit `df82c9fdff91b9b130f2d6b89110d3870ba6260e`（20,684,500 字节）的
  SHA-256 固定快照。
- Phase 3 当前只读状态为 seed missing、0 accepted batch、0 accepted query、
  无 active plan；模型门在没有用户确认 `5.6 Sol Ultra` 时正确拒绝生成。

前四项是仓库/官方工件验证事实；“这些组合能否达到足够自然度”仍是待人工评审的
推断，不能先写成已验证质量结论。

### 根因

原计划把一个数据集同时当作“交互机制样板”和“真实分布证据”。前者可以由多个
公开语料的抽象模式与受控重写组合近似，后者必须依赖真实来源本身，无法由 Mock
替代。另一个根因是 provenance 只记录“谁生成”，没有把“这不是实流量”作为
强制机器字段。

### 考虑过的方案与取舍

1. 继续等待 JDDC 2.0：最接近原计划，但获取期限不可控，会永久阻断 MVP。
2. 找一个数据集一对一替代：简单，但没有公开来源同时满足中文、真实电商、多轮、
   图像和商品 KB，容易产生错误等价。
3. 完全用模板/LLM 生成：最快，却缺少真实人写交互现象锚点，也最容易伪装成实流量。
4. 组合公开模式源并显式降级：DuRecDial 提供偏好追问，CrossWOZ 提供约束变化，
   MUGE 提供短查询措辞；事实仍来自 verified Asset/KB，Codex 只受控重写。代价是
   必须承认外部有效性降低，并增加 source lock、pattern inventory 和人工审核。

### 最终方案

将 JDDC 2.0 标记为 `permanently_abandoned`，从 MVP、Full 和 full-upstream 下载
profile 全部移除，同时保留 retired disposition，避免未来被无意加回。新增
`dialogue-trajectory-source-policy-v1.json`：MVP 固定为 DuRecDial 2.0 +
CrossWOZ + MUGE + Codex Mock，TaskSpec 是语义权威，verified Asset/KB 是事实权威，
源对话只提供抽象 pattern，禁止复制源实体或整句。生成 batch draft、staging
manifest 和质量报告强制声明 `data_origin=synthetic_derived`；每批 25 条，只进
staging，仍需独立人工接受。

Full 不预先冻结 adapter 或配额。U-NEED、SIMMC 2.1 和 CSDS 只登记为条件候选，
待包、许可、PII 和 source lock 实际到手后重做设计。owner 提及的 `JDDC 2.1`
单独记录为身份/条款待核验标签，不能与 SIMMC 2.1 混称。

### 如何验证

- raw profile 测试证明 JDDC 2.0 不会从任何 profile resolve，retired disposition
  保留；DuRecDial/CrossWOZ 覆盖 MVP language tier。
- source portfolio 测试证明 Mock 只能是 `synthetic_derived`，并禁止提供
  capability/product/visual facts 或 real-user-log claim。
- 两个公开 commit archive 通过固定字节数和 SHA-256 的项目 downloader `verify`；
  archive 内存在预期 data、README 和 LICENSE。
- synthesis 模型和 batch 测试证明缺失或错误的 `data_origin` 无法通过，staging
  manifest 与 quality report 都回显该字段。
- `guard-model` 在本轮没有明确模型确认时返回非零，且没有写入任何 seed 或轨迹。

### 剩余限制

尚未建立 DuRecDial/CrossWOZ 的正式 pattern inventory、数据用途审批和外部
source lock；MUGE 也仍需明确适用许可。当前没有 accepted seed、active plan 或
Mock batch，因此只完成“可诚实开始”的基础设施，尚未证明自然度、覆盖度或人工
接受率。Full 候选均不能在授权/许可评审前进入正式实现。真实中文电商多模态流量
外部有效性的缺口是明确保留的限制，不是后续 Mock 数量增加就能关闭的指标。

### 30 秒回答

JDDC 2.0 拿不到时，我没有把 LLM 生成对话偷偷当作等价替代，而是拆开了机制复现
和真实分布证据。MVP 用 DuRecDial 的偏好追问、CrossWOZ 的约束变化和 MUGE 的中文
短查询做模式锚点，事实只来自验证过的资产与 KB，Codex 输出强制标记
`synthetic_derived` 并逐批人审。JDDC 2.0 从所有 profile 永久退出，报告也明确不再
声称复现它的真实流量规模或多模态用户分布。

### 2 分钟回答

原方案把 JDDC 2.0 同时当作中文交互样板和真实电商分布证据，但它是授权源，当前
无法取得。如果只把下载状态改成完成，再让 LLM 生成相似对话，实验会产生严重的
provenance 漂移：代码能跑，却无法区分真实用户日志和 synthetic 数据。我比较了
继续等待、寻找一对一替代、纯模板生成和组合降级。公开数据中没有任何单一来源能
同时覆盖中文、真实电商、多轮、图像和商品 KB，所以选择组合降级：DuRecDial 提供
偏好逐步暴露和推荐转移，CrossWOZ 提供 no-offer、条件放宽、多次查询和目标变化，
MUGE 提供中文电商短表达。它们只生成 pattern inventory；TaskSpec 决定任务语义，
verified Asset/KB 决定事实，Codex 负责领域重写。系统层面把 JDDC 2.0 记为 retired
并从所有 profile 删除，生成 manifest 和质量报告强制写
`data_origin=synthetic_derived`，每批 25 条且不能自动接受。这样 MVP 可以继续验证
机制，但我会明确报告真实流量外部有效性降低。Full 则等 U-NEED 等实际授权包与条款
到手后再设计，不提前承诺角色或配额。

### 证据入口

- `specs/data_sources/dialogue-trajectory-source-policy-v1.json`
- `specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`
- `specs/data_sources/raw-download-profiles-v1.json`
- `docs/mock-trajectory-runbook.md`
- `src/skillchain/synthesis/models.py`
- `src/skillchain/synthesis/batches.py`
- `tests/test_data_source_portfolio.py`
- `tests/test_raw_download_profiles.py`
- `tests/synthesis/test_models.py`
- `tests/synthesis/test_batches.py`

---

## 20. 全量测试失败不一定是功能回归：先把失败按共同系统调用归因

**状态：已验证**

### 一句话问题

MVP RAW 交接前的全量测试出现 29 个失败和 10 个 setup error，但共同断点不是数据下载逻辑，而是 Windows 将目录级 `os.replace(staging_dir, output_dir)` 拒绝为 `WinError 5`。

### 背景与影响

如果只看失败数量，会误判本轮 RPC 收尾造成了大范围回归，进而扩大修改范围；如果直接忽略全量测试，又会掩盖真实问题。交接需要同时证明本轮变更正确，并准确记录未通过的环境边界。

### 观察到的证据

- 本轮直接相关的下载、FashionIQ、SROIE、profile、portfolio、协议哈希和计划渲染共 69 项测试全部通过。
- 全量结果为 `944 passed, 24 skipped, 2 deselected, 29 failed, 10 errors`。
- 39 个异常的堆栈都落到既有 asset/catalog/index 发布路径的目录级 `os.replace(...)`，错误为 Windows `PermissionError: [WinError 5]`。
- 从失败列表中隔离重跑三个代表用例，两个依赖目录原子发布的用例仍以同一错误失败，一个不走该路径的用例通过。

以上是本机可复验事实；“防病毒扫描、索引器或 Python/Windows 版本组合导致句柄竞争”目前只是候选解释，尚未得到进程级句柄证据。

### 根因

已定位的直接原因是当前 Windows 环境不能稳定完成现有目录级原子重命名；尚未证明具体是哪个外部进程或平台语义触发拒绝访问。本轮 RPC 只对单个文件执行 `os.replace(partial, final)`，真实收尾和对应测试均已通过，因此不能把目录发布故障归因到 RPC 改动。

### 考虑过的方案与取舍

1. **为交接顺手改写全仓库目录发布协议：** 可能让测试变绿，但会把 RAW 收尾扩张成独立的跨平台原子发布设计，风险过大。
2. **只报告定向测试：** 范围清晰，但会隐藏已执行且失败的全量验证。
3. **保留失败证据并隔离重跑：** 能区分本轮回归和环境阻断，同时把目录发布问题留给独立修复。

选择第三种。

### 最终方案

本轮不改动既有 asset/catalog/index 的目录发布实现；交接记录同时报告 69 项相关测试通过和全量测试的精确失败统计、共同堆栈与隔离复现结果。后续应在单独变更中收集句柄/安全软件证据，并为 Windows 设计经验证的目录发布与恢复协议。

### 如何验证

- `uv run pytest tests/data/test_raw_download.py tests/data/test_fashioniq.py tests/data/test_sroie.py tests/test_raw_download_profiles.py tests/test_data_source_portfolio.py tests/test_protocol_hashes.py tests/test_render_plan.py -q`
- 2026-07-25 合并复验：跨分支关键集为 `94 passed, 2 skipped`；其余 LLM/authoring/synthesis 集为 `205 passed, 2 deselected, 1 error`，唯一 error 仍落在同一个目录级 `os.replace` / `WinError 5` 边界。
- `uv run pytest -q`
- 隔离重跑 `test_clean_cross_source_global_catalog_can_finalize`、`test_failed_build_never_replaces_the_existing_index` 与 `test_publisher_rejects_uncataloged_approval_asset`

### 剩余限制

当前没有 ProcMon/句柄跟踪、另一台 Windows 主机或 Linux CI 的对照回执，因此只能确认共同失败点，不能宣称已确定最终平台根因。全量测试仍不是绿色，交接必须明确这一点。

### 30 秒回答

“全量测试出现 39 个异常时，我没有按数量判断为本轮回归，而是聚类堆栈。所有异常都落在既有目录级 `os.replace` 的 Windows `WinError 5`；本轮相关 69 项全过，代表用例隔离重跑也复现同一平台错误。我保留了完整失败证据，没有扩大 RAW 变更去重写发布协议，并把跨平台目录原子发布作为独立后续问题。”

### 2 分钟回答

“交接前我跑了全量测试，结果是 944 通过、29 失败、10 error。表面上像大回归，但堆栈聚类显示 39 个异常全部来自 asset catalog、index 或 formal fixture 的 staging 目录发布，直接系统调用统一是 `os.replace(staging_dir, output_dir)`，Windows 返回 `WinError 5`。本轮改动涉及 RPC 的单文件 partial 校验与原子转正，不经过这些目录发布路径；相关 69 项测试全部通过。为了避免把相关性当因果，我又隔离跑了三个代表测试，依赖目录发布的两个仍失败，不依赖的一个通过。我的取舍是如实报告全量非绿，不为了追求绿色而在同一提交中重写跨平台发布协议。剩余工作需要用 ProcMon 或句柄工具确认是安全软件、索引器还是平台/运行时语义，并在独立变更中设计恢复协议和 CI 对照。”

### 证据入口

- `src/skillchain/data/asset_catalog.py`
- `tests/data/test_abo.py`
- `tests/tools/test_product_index.py`
- `tests/tools/test_document_safety.py`
- `src/skillchain/data/raw_download.py`
- `tests/data/test_raw_download.py`

---

## 21. 数据策略变了，授权 policy 也必须作为版本化依赖一起迁移

**状态：已验证**

### 一句话问题

数据 portfolio 已永久移除 JDDC 2.0 并改用 DuRecDial/CrossWOZ/MUGE 模式加
`synthetic_derived` Mock，但 C2 source-review v1 仍要求批准 JDDC，且绑定旧 portfolio
摘要；如果不迁移 policy，合法的新数据策略会被旧 gate 阻断，也可能诱使人把派生 Mock
错误包装成外部数据许可。

### 背景与影响

source-review policy 是 adapter 进入正式 `clean/` 前的授权根。它不只是说明文档：
loader 会校验外部摘要、portfolio 摘要、必需来源、用途、许可权限与 PII 状态。数据策略
变化后若只更新下载清单和计划，不更新这个授权根，C2 会出现“事实层已经换源、权限层仍
要求旧源”的分裂。

### 观察到的证据

- 当前 portfolio 文件 SHA-256 为
  `6f2a0c98a2889aa6980e3f2308cfd6a076e5bdd60e0e7e99acc77f8db74b501a`，
  v1 policy 仍绑定旧摘要
  `af27acc01e60d65434484dd7c61429be0b73c8ec8061609ff3efe0e4aef4ea6d`。
- v1 把 `jddc_2_0` 标为 required；portfolio 与 dialogue policy 已将其标为
  `permanently_abandoned`。
- 新策略中的 DuRecDial 2.0 与 CrossWOZ 不在 v1 requirements 中。
- `codex_mock_trajectories` 是派生输出；现有 source record 的 approved 语义却要求
  `download_allowed` 和 `local_research_allowed`，不适合拿来伪装 Mock 的人工接受。

### 根因

此前把 source-review policy 当成了“实现一次即可长期复用”的静态清单，而没有把它视为
source portfolio 的版本化下游依赖。与此同时，旧 purpose 只有宽泛的
`language_style`，无法区分“原始语言风格”与“抽象多轮交互模式”，容易扩大授权用途。

### 考虑过的方案与取舍

1. **直接修改 v1：** 改动最少，但会让历史 SHA 和过去的审批说明失真。
2. **把 Codex Mock 加入外部 source ledger：** 表面上集合完整，实际会把派生输出误建模成
   有下载许可的外部数据，并混淆 source approval 与逐条合成质量审核。
3. **发布前瞻 v2：** 保留 v1 历史，绑定当前 portfolio，重新裁决 required 集合，并让派生
   输出继续走独立 provenance/human-review gate。

### 最终方案

选择 v2。新增 `interaction_pattern` purpose；DuRecDial 2.0 和 CrossWOZ 都是 required，
只批准该用途并要求 PII review；MUGE 保持 `language_style`。retired JDDC 与
`codex_mock_trajectories` 不进入 ledger。下厨房仍保持 required 且要求 PII review；
若许可证据不足，必须在看系统结果前通过 portfolio/policy deviation 替换或降级，不能绕过
审批。

### 如何验证

- v2 policy 使用 canonical JSON，并通过真实 portfolio SHA 加载。
- 测试断言 JDDC/Mock 不在 requirements、DuRecDial/CrossWOZ 的 purpose 与 PII gate
  正确、下厨房要求 PII review。
- 负向测试证明包含 retired JDDC record 的 ledger 会以
  `outside the policy` 失败关闭。
- `uv run --frozen pytest tests/data/test_source_review.py -q`：
  `5 passed`。

### 剩余限制

v2 只完成“应审批哪些来源、可批准什么用途”的机制迁移，不代表任何来源已经获得 owner
批准。真实 source lock、许可证据、PII review、adapter 输出和 Mock 人审仍缺失。
source-review gate 也不能单独阻止下游把 pattern 当 gold；该语义边界仍需 portfolio、
dialogue policy、adapter 和 Phase 3 verifier 共同执行。

### 30 秒回答

“数据策略从 JDDC 改为 DuRecDial/CrossWOZ/MUGE 加受控 Mock 后，我发现权限 policy 仍绑定
旧 portfolio 并要求 JDDC，导致事实层和授权层分裂。我没有覆盖 v1，而是发布不可变 v2：
新增 interaction-pattern 用途，要求两个对话源做 PII review，排除 retired JDDC，并把
Codex Mock 留在独立的 synthetic provenance 与人工审核门。真实 policy 加载和负向 ledger
测试都通过。”

### 2 分钟回答

“这个问题不是简单换一个 source ID。source-review policy 是正式 adapter 的授权根，会
校验外部 hash、用途、权限和 PII。portfolio 已永久移除 JDDC，但 v1 仍把它设为 required，
而且绑定旧摘要；同时新的 Codex Mock 是派生输出，不能为了让集合看起来完整就给它伪造
download permission。我比较了覆盖 v1、把 Mock 塞进 ledger 和前瞻 v2 三种方案，最终保留
v1 历史，发布绑定当前 portfolio 的 v2。DuRecDial/CrossWOZ 只能贡献抽象交互模式并要求
PII review，MUGE 只贡献语言风格，Mock 单独保存 synthetic provenance 并人工审核。测试还
验证 retired JDDC record 会失败关闭。剩余工作是 owner 逐源批准和 adapter/Phase 3 的
端到端语义 enforcement，因此我不会把 policy 更新写成 C2 已通过。”

### 证据入口

- `specs/data_sources/mvp-source-review-policy-v1.json`
- `specs/data_sources/mvp-source-review-policy-v2.json`
- `specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`
- `specs/data_sources/dialogue-trajectory-source-policy-v1.json`
- `src/skillchain/data/source_review.py`
- `tests/data/test_source_review.py`
- `docs/data-source-review-runbook.md`

---

## 22. 真实 source lock 不能替许可证，证据准备者也不能冒充 owner

**状态：部分解决**

### 一句话问题

即使 10 个数据源的 78.69 GB 本地字节都能被精确复验，内容身份也不会自动产生使用权限；
由 Codex 准备的保守建议也不能被写成项目所有者已经签署的 ledger。

### 背景与影响

C2 要求 adapter 在进入正式 `clean/` 前同时绑定 source revision、source-lock SHA、
许可证据 SHA、用途权限和 PII 决定。此前 RAW 下载已经完成，但下载状态里的大小、ETag
和“公开可访问”只能说明取得了某些字节；它们不能回答这些字节是否允许本地 embedding、
是否可再分发、是否能进入公开 demo，也不能替 PII 决定。

### 观察到的证据

- 10 个 required source 的本地范围合计 75,368 个文件、78,692,489,139 字节。
- ABO/RPC 有明确的非商业 CC 许可；FashionIQ 的固定 annotation commit 没有 LICENSE，
  但 CodaLab 条款明确限定非商业学术研究；DuRecDial README 给出数据集
  CC BY-NC-SA 4.0；CrossWOZ 固定 commit 根目录有 Apache-2.0。
- WildReceipt 的官方 MMOCR 元数据明确写 `License: N/A`；ISIA 只有论文/下载页；
  MUGE 本地分卷来自第三方镜像且缺 exact agreement；本地 zhwiki 是第三方过滤包，
  缺官方 dump/变换/attribution 链；下厨房用户授权平台使用，不等于向公众开放数据集复用。
- 收据、对话和用户生成菜谱不能因为正则扫描“没看到某个号码”就声明无 PII，所以采用
  `restricted` 而不是伪造 `reviewed_no_pii`。
- 项目所有者审阅保守 v1 后，明确决定对 10 个 required source 全部给出限定用途
  `approved`。这是风险接受决定，不会改变四个 `NOASSERTION` license ID，也不会解除
  restricted PII、remote embedding、再分发或公开展示限制。

### 根因

原流程混合了三个不同 trust root：下载器提供本地字节事实，许可证据说明上游授权边界，
项目所有者负责接受具体用途风险。如果让一个 builder 同时生成 hash、判断许可并写
`reviewer_id=owner`，整套工件可以内部自洽，却没有独立的人类授权。

### 考虑过的方案与取舍

1. **提交逐文件清单：** 最直观，但 FashionIQ 75,267 张图会制造巨型 Git 工件，审阅和
   diff 都很差。
2. **只保存目录 hash 或 ETag：** 文件小，但 ETag 可能是 multipart 标识，目录 hash
   算法也不可重算，无法证明本地内容。
3. **对未知许可按“公开研究数据”批准：** 能让 C2 更快变绿，但把可下载错误等同于可用，
   法律与研究诚信风险最高。
4. **compact tree commitment + 独立 proposal：** 对排序后的
   `(relative path, bytes, SHA-256)` canonical rows 做摘要，完整覆盖字节又不提交巨型
   inventory；许可未知时权限全 false，并把 AI/preparer proposal 与 owner record 分开。
5. **允许 owner 明示风险接受，但保留证据事实与最小权限：** 尊重项目所有者的最终
   authority，同时不把内部批准写成上游授权；未知许可来源只获本地研究边界，所有来源
   均禁止 remote embedding、再分发和公开 demo。采用该方案。

### 最终方案

新增通用 source-lock schema 和稳定大文件哈希器。显式文件与递归树都拒绝 symlink、
路径逃逸、大小写冲突和读取期间变更；FashionIQ 同时绑定 accepted images、固定
annotation/URL inventory commit 和 exclusion ledger。逐源许可证据生成独立 canonical
文件并由 proposal 精确绑定。v1 保留“批准五个、defer 五个”的保守建议。owner 明确要求
全部批准后，发布新的 v2 proposal 而不覆盖 v1；`SourceReviewProposal` 仍没有
`reviewer_id`，只在 exact v2 proposal SHA 被确认后，才以 `project-owner` create-only
materialize 成 ledger。独立 signature receipt 同时绑定 proposal/ledger SHA，并写明
内部风险接受不替代缺失的上游许可证据。严格 `verify` 现对 v2 返回 10 approved、零
blocker；adapter 仍必须绑定 exact lock、用途和权限。

### 如何验证

- source-lock 构建实际完成，manifest SHA-256 为
  `8b9da4b6bff5e57c5c23a52330ca4dc5e87d4b693e27c3b1d124b7e49259958c`。
- owner proposal SHA-256 为
  `943e16dba66814217b085f78a286738dc32efc03272159d3839619d815d0f5dd`，
  bundle manifest SHA-256 为
  `e06cf8041a378856176c4552d219fb41d55ed41eb33acab56f5a557d13c65f9d`。
- owner-signed v2 proposal/ledger/signature receipt SHA-256 分别为
  `a29e261df3183cf670e31d9e55a5952b8a7cee1ea0321588148b8c541b855ed1`、
  `8bf9ad9d2dd786c84b1849c562557cac327ab1cef4d653e7588e7dbc7af215c9`、
  `028da86082524161ec4f5727d5196cb938ad7efecfbddb3358544b447fc2a642`。
- 单元测试覆盖 lock 重算、文件增删改、外部 lock hash、deferred ledger 状态、
  proposal→owner record 边界、签署回执双哈希和 10 个 v2 approval 的严格加载。
- fixture owner ledger 的 `status` 如实返回五个 approved 与五个 deferred blocker；
  同一历史 ledger 的严格 `verify` 以 exit 2 失败关闭。真实 v2 ledger 的 `status` 和
  `verify` 都返回 10 个 approved、零 blocker。

### 剩余限制

owner 签名只关闭 source-review authorization 子门，不等于取得缺失的上游权利。
ISIA、MUGE、WildReceipt 和下厨房仍是 `NOASSERTION`，zhwiki 的 provenance/attribution
链仍不完整；对外发布前必须披露，且不能扩大到 remote processing、再分发或公开 demo。
source lock 也不证明标签质量、selection 合法、Exact Match eligibility 或无泄漏；这些
仍需 adapter/catalog/gold 阶段的独立证据。因此 C2 整体仍未通过。

### 30 秒回答

“我把 10 个来源约 78.69 GB 的本地字节做成可重算 source lock，并把 AI 建议与 owner
授权分成不同 schema。最初的保守提案是五批五缓；owner 后来明确接受全部 10 个来源的
限定用途风险，我没有覆盖历史，而是发布 v2 并生成绑定 proposal/ledger 双哈希的签署
回执。严格 gate 现在通过，但 `NOASSERTION` 和 PII 限制仍原样保留，所以这只关闭授权
子门，不代表许可缺口或完整 C2 被解决。”

### 2 分钟回答

“这个问题有三层事实：下载器只能证明拿到了字节，许可证据描述上游权利边界，owner
负责接受项目用途和 PII 风险。我先解决字节身份：对每个稳定读取的文件记录相对路径、
大小和 SHA-256，再对排序行做 canonical tree commitment；symlink、路径逃逸、大小写
冲突和读取时变更都会拒绝。接着按固定证据生成保守 v1，五个证据不足来源 deferred，
收据、对话和 UGC 按 restricted。owner 后来明确要求全部批准，我没有把 v1 或证据改写
掉，而是发布 v2：四个来源仍写 `NOASSERTION`，所有来源的 remote embedding、再分发和
公开 demo 都是 false。只有 owner 确认 exact v2 hash 后才生成 ledger，签署回执再绑定
proposal 和 ledger 两个摘要。严格 gate 因此能通过，但这只代表内部授权有效，不代表
未知上游许可突然存在；adapter、catalog、gold、Exact Match 和 leakage 仍必须独立收口。”

### 证据入口

- `src/skillchain/data/source_lock.py`
- `src/skillchain/data/source_review.py`
- `scripts/build_required_source_locks.py`
- `scripts/prepare_source_review_bundle.py`
- `specs/data_sources/c2/source-locks/required-source-lock-manifest.json`
- `specs/data_sources/c2/source-review/source-review-bundle-manifest.json`
- `specs/data_sources/c2/source-review/owner-review-proposals.jsonl`
- `specs/data_sources/c2/source-review-v2/source-review-bundle-manifest.json`
- `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl`
- `specs/data_sources/c2/source-review-v2/signed/owner-ledger-signature-receipt.json`
- `tests/data/test_source_lock.py`
- `tests/data/test_required_source_review_bundle.py`
- `docs/data-source-review-runbook.md`

---

## 23. 两份都“正确”的 source lock 仍可能覆盖不同字节集合

**状态：部分解决**

### 一句话问题

owner 批准的 required lock 与 adapter 自己的 normalized lock 都能独立通过哈希校验，
仍不代表前者覆盖了 adapter 实际读取的每一个 RAW 文件。

### 背景与影响

C2 的 owner ledger 按 exact required-lock SHA 批准来源、用途和权限。ABO Exact Match
adapter 此前又有一套更细的 normalized lock，绑定 acquisition receipt、listing shard、
image metadata、许可证据和定向 original images。若 formal build 只检查“source_id=abo”
或只把两个摘要都写进 manifest，它可能使用 owner 从未按 exact scope 审批的字节，最终
candidate bundle 虽然内部自洽，却不具备正式授权身份。

### 观察到的证据

- 已签署的 ABO required lock 只覆盖
  `abo-images-small.tar`、`abo-listings.tar` 和 `abo-spins.tar` 三个 compact archive。
- 现有 Exact Match adapter 会读取解包后的 listing shard、`images.csv.gz` 和逐个定向取得的
  `images/original` 文件；这些路径不在上述 required-lock scope 中。
- owner ledger 的记录绑定 required-lock SHA；ABO normalized lock 则绑定另一份 source-lock
  SHA，二者不是同一个 trust root。
- 新负向测试使用只覆盖 metadata、不覆盖 original image 的有效 required lock；policy、
  ledger、license evidence 和 normalized lock 都保持自洽时，formal build 仍在发布前以
  `does not cover consumed RAW file` 失败。

以上均为代码、已跟踪 lock 和测试验证的事实。当前真实数据尚未生成新的覆盖性 lock 或
owner reapproval，因此“真实 ABO formal run 可执行”仍是未完成项。

### 根因

原设计把“来源级授权”和“字节集合级授权”混在一起。required lock 面向下载 portfolio，
normalized lock 面向 adapter 的细粒度 chain of custody；两者各自解决的问题不同。如果
没有显式的 scope inclusion 检查，系统只能证明两个摘要分别存在，不能证明 approved
集合包含 consumed 集合。

### 考虑过的方案与取舍

1. **只按 source ID/revision 复用 owner approval：** 接线最少，但 revision 不能证明
   consumed bytes，旧审批可能被套到新下载或新派生物上，淘汰。
2. **让 normalized lock 取代 required lock：** 能覆盖 adapter 输入，但会绕过已签署
   required-lock SHA，并把 adapter 自己生成的信任记录冒充 owner trust root，淘汰。
3. **只把两个 SHA 都写进 manifest：** 提高可追踪性，却仍不证明集合包含关系，不能单独
   解决授权缺口。
4. **双锁 + consumed-path coverage gate：** required lock 继续代表 owner-approved
   RAW 集合，normalized lock 继续证明 raw→normalized chain；formal adapter 逐个证明
   listing、metadata 和 selected original image 都落在 approved scope，并在发布前重验
   lock/policy/ledger/evidence 与 RAW 内容。采用该方案。

### 最终方案

新增不可由普通 dataclass 构造冒充的 `VerifiedABOSourceApproval`。loader 独立加载并复验
required lock、RAW scope、license evidence、source-review policy 和 owner ledger，再按
固定的 Exact Match purposes 与 local research/embedding 权限调用 `require_approval`。
formal review-packet、audit、build 和 verified candidate loader 都要求该 handle。

ABO candidate bundle policy 升级为 v2，manifest 分别绑定 required-lock、policy、ledger、
owner-record、license-evidence 和 normalized-lock SHA。构建前后都会刷新 owner approval；
发布前还逐个检查 raw listing、image metadata 和实际 selected original image 是否被
approved scope 覆盖。任一 identity、权限、scope、文件身份或内容在处理中变化，staging
会被清理且不发布 destination。

### 如何验证

- `tests/data/test_abo.py` 覆盖 revision/source-lock/license/evidence 错配、remote/public
  权限升级、少报 local embedding 权限、missing ledger、伪造 verified handle、
  required scope 漏掉 original image 和 manifest 生成后的 ledger TOCTOU。
- 成功路径断言 candidate bundle policy 为
  `abo-exact-candidate-bundle-v2`，并回显五个 owner-approval binding SHA。
- `uv run ruff check src/skillchain/data/abo.py tests/data/test_abo.py`：通过。
- `uv run pytest -q tests/data/test_source_review.py
  tests/data/test_required_source_review_bundle.py tests/data/test_source_lock.py
  tests/data/test_abo.py`：`51 passed, 1 skipped`。

### 剩余限制

当前真实 ABO compact lock 并不覆盖 existing adapter 的 metadata/original paths，所以新门
会诚实阻断真实 run。下一步必须在任何系统结果可见前二选一：发布覆盖实际 consumed RAW
的新 required lock 并让 owner 对 exact SHA 重新审批；或改造 adapter 只消费当前已批准的
archive scope，同时重新证明这些资产仍满足“真实多视图、非 derivation、非近重复”的
Exact Match 资格。机制测试不能替代这项真实工件与人工授权工作，其余数据源 adapter 也
尚未迁移。

### 30 秒回答

“我接 owner 数据审批时发现一个容易漏掉的供应链问题：ABO 的审批 lock 和 adapter 的
normalized lock 都能独立验真，但前者只锁了三个 archive，后者实际还读 metadata 和定向
original images。只记录两个 SHA 仍不能证明 approved 集合包含 consumed 集合。我做了
不可伪造的 approval handle、双锁 manifest 和逐文件 scope coverage/TOCTOU 门。测试全部
通过，但真实 compact lock 仍不覆盖 adapter 输入，所以系统会诚实阻断，等待新 lock 和
owner reapproval，而不是把机制完成写成数据 ready。”

### 2 分钟回答

“这个问题本质上是两个粒度不同的 trust root。required lock 服务下载 portfolio 和 owner
授权，normalized lock 服务 adapter 的 raw-to-clean chain of custody。它们各自 hash 正确，
并不自动产生集合包含关系。ABO 的真实 required lock 只覆盖 images-small、listings 和
spins 三个 archive；Exact adapter 却读取解包 listing、image metadata 和逐个 original
image。如果只按 source ID 复用 approval，会把来源级授权错误扩大成任意字节授权；如果让
normalized lock 取代 required lock，又等于让被审核的 adapter 自己签发 owner trust root。
最终我保留双锁：loader 先验证 required lock、license evidence、policy 和 owner ledger，
生成带私有 token 的 approval handle；formal build 再逐个证明实际 consumed RAW path 位于
approved scope，manifest v2 同时绑定两套锁和全部审批摘要，发布前重验 TOCTOU。负向测试
覆盖协调错配、权限升级、scope 漏图和 ledger 漂移。机制已验证，但真实 scope mismatch
仍保留为 C2 blocker，必须新 lock + owner reapproval 或调整 adapter 后才能运行。”

### 证据入口

- `src/skillchain/data/abo.py`
- `src/skillchain/data/source_lock.py`
- `src/skillchain/data/source_review.py`
- `tests/data/test_abo.py`
- `specs/data_sources/c2/source-locks/abo.source-lock.json`
- `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl`
- `docs/data-source-review-runbook.md`
- `docs/plans/2026-07-20-p0-p1-closure.md`
- `docs/go-no-go/2026-07-20-core-no-go.md`

---

## 24. 历史 runtime 在新工作树重放失败时，不能为测试变绿而改写证据

**状态：部分解决**

### 一句话问题

已消费的 authoring receipt 把 Python executable 锁到原主工作副本；在 Codex worktree
运行全仓测试时，路径身份不同导致历史 replay 失败，但正确处理不是重算并覆盖历史 lock。

### 背景与影响

Codex v3/v4/v5 authoring 工件用于证明一次性模型调用的 exact runtime 和终态。如果每次换
工作树都更新其 source manifest、freeze 或 receipt，测试可以恢复绿色，却会把“历史上实际
运行了什么”改写成“当前机器上有什么”，破坏一次性授权、负结果和 canonical replay 的
证据价值。

### 观察到的证据

- C2/ABO 相关集合为 `86 passed, 1 skipped`，Ruff 与 `git diff --check` 通过。
- 全仓为 `1137 passed, 27 skipped, 2 deselected, 5 failed`；失败全部位于
  `test_codex_authoring_v3.py`、v4 和 v5 的 package/history/canonical replay。
- tracked v3/v4/v5 source manifest 都绑定
  `D:\athena\ECommerceSkillChain\.venv\Scripts\python.exe`。
- 当前 Codex worktree 实际执行的是
  `C:\Users\torto\.codex\worktrees\6daa\ECommerceSkillChain\.venv\Scripts\python.exe`。
- 本次 diff 没有修改 `deploy/authoring`、`specs/authoring`、`runs/formal-authoring` 或
  authoring runner/compiler 文件。

前四项与未修改清单是命令输出验证的事实。由路径绑定解释五个 replay 失败是基于 manifest
内容和失败集中位置的诊断；尚未在原 `D:\athena` runtime 重新执行同一全仓命令，因此原
runtime 仍可重放属于待复验项。

### 根因

authoring runtime identity 有意包含 absolute executable path，以防同名解释器或依赖环境
被替换；但普通全仓测试同时把这种 machine-bound 历史验证当成任意 worktree 都应通过的
测试。安全目标本身合理，测试可移植性假设却不成立。

### 考虑过的方案与取舍

1. **重建并覆盖 v3/v4/v5 manifest/receipt：** 能立即变绿，但会改写已消费授权和历史
   事实，禁止。
2. **在 worktree 中伪造相同绝对路径：** 可能绕过表面路径检查，也会混淆真实文件身份，
   且无法诚实证明原 runtime，禁止。
3. **忽略全部全仓失败：** 会掩盖真正的功能回归，不可接受。
4. **按证据域归因并保留历史：** 先证明本次相关定向集合通过，记录全仓失败的精确测试、
   runtime 路径和未修改边界；后续在原锁定 runtime 重放，或新增读取历史 artifact 的
   hermetic replay harness，但不改历史字节。采用该方案。

### 最终方案

本批次不修改任何历史 authoring lock、freeze、claim、receipt 或 output。NO-GO 记录同时
保留相关定向回归和全仓结果，明确五个失败属于 machine-bound replay 未在当前 worktree
满足，而不是宣称全仓绿色。修复若需要代码，应是前瞻的新 replay harness/测试分类：历史
validator 仍验证 original identity，普通 worktree CI 则不会通过重签历史来获得成功。

### 如何验证

- 对比 `uv run python -c "import sys; print(sys.executable)"` 与三个 tracked source
  manifest 的 `python.executable.path`，确认路径不同。
- `git diff --name-only -- deploy/authoring specs/authoring runs/formal-authoring ...`
  无输出，确认本批次没有触碰历史 authoring 工件和实现。
- 全仓失败清单只含五个 authoring v3/v4/v5 replay 测试；其余 1,137 项通过。
- 尚需在原锁定 runtime 或经独立批准的 hermetic replay 环境再次验证历史 bundle。

### 剩余限制

当前工作树不能声称全仓绿色，也不能声称原 runtime 一定仍可用。absolute path 只是 runtime
identity 的一部分，真正的 hermetic replay 还应处理解释器字节、依赖树、OS 行为和外部
平台不可见性。是否将 machine-bound 测试从普通 worktree suite 分类出去，需要单独设计和
审查，不能在本 C2 任务里顺手弱化 validator。

### 30 秒回答

“我跑全仓时有五个 authoring 历史重放失败，但相关 C2 测试全绿。我没有直接改历史 hash，
而是比较 manifest 和当前解释器，发现历史锁在 D 盘主工作副本，当前从 Codex worktree
执行。已消费 receipt 的意义就是记录当时 runtime；为新路径重签会篡改事实。我保留五个
失败，证明本次 diff 没碰 authoring 工件，并把后续动作限定为原 runtime 重放或前瞻的
hermetic replay harness。”

### 2 分钟回答

“这类失败容易诱导工程师运行 builder 更新 golden files，但这里的 golden 是一次性模型
调用的审计证据，不是普通快照。全仓 1,137 项通过、五项失败全部集中在 Codex authoring
v3/v4/v5。三个 source manifest 都把 executable 锁到 D 盘主工作副本，而当前测试解释器
位于 C 盘 Codex worktree；本次 diff 又完全没有触碰 authoring 目录。路径绑定本身是为了
防止替换 runtime，问题在于普通 worktree 测试假设它可移植。我比较了覆盖历史、伪造路径、
忽略失败和按证据域归因四种方案，最终保留不可变工件：相关 C2 回归单独证明，NO-GO 中
诚实记录全仓失败，后续只允许在原锁定环境重放或新增不改写历史的 hermetic harness。
这样既不把无关环境失败误判为功能回归，也不为了绿色测试破坏审计链。”

### 证据入口

- `deploy/authoring/locks-codex-v3/source-manifest.json`
- `deploy/authoring/locks-codex-v4/source-manifest.json`
- `deploy/authoring/locks-codex-v5/source-manifest.json`
- `tests/test_codex_authoring_v3.py`
- `tests/test_codex_authoring_v4.py`
- `tests/test_codex_authoring_v5.py`
- `docs/go-no-go/2026-07-20-core-no-go.md`

---

## 25. “审批过 FashionIQ”不等于任意 adapter 输出都自动合规

**状态：部分解决**

owner ledger 已批准 FashionIQ 的限定本地用途，但如果 adapter 不在产出前绑定 exact
revision、source-lock、许可证据、用途和权限，旧审批仍可能被套到新字节上，冻结的
2,416 条下载排除项也可能悄悄回流。

### 背景与影响

FashionIQ 是 Style 主监督来源。它的本地快照由 annotation、URL inventory、75,267 张
accepted image 和 2,416 条 exclusion ledger 共同定义。source-review gate 通过只说明
owner 接受了这一个快照的本地研究与 embedding 用途；它没有自动约束后续 adapter，
也没有证明 adapter 输出完整覆盖 disposition。如果 adapter 只遍历现存图片，排除项会
从审计分母中消失；如果只记录 source name，revision 漂移后仍可能复用旧授权。

### 观察到的证据

- source lock 明确分成 `accepted_images`、`annotations`、`exclusion_ledger` 和
  `url_inventory` 四个 scope。
- owner record 允许 `capability_gold`、`product_gallery`、`tool_gold` 以及本地
  embedding，但明确禁止 remote embedding、再分发和公开 demo。
- 原 `fashioniq.py` 只完成对象级下载、重试、排除冻结和 acquisition verification，
  返回的 `formal_ready` 仍是 false；它没有生成 `DatasetAssetDraft`，也没有调用
  `require_approval(...)`。
- 新定向回归为 11 条，覆盖 happy path、旧 revision、缺用途、缺本地 embedding 权限、
  越权 outbound permission、RAW 漂移和发布后 payload 篡改。

### 根因

授权事实和授权执行原先位于两个不同层：ledger 能回答“owner 批准了哪一个快照做什么”，
downloader 能回答“哪些对象下载成功或被排除”，但 adapter 缺少把两者强制连接起来的
formal entry point。仅让 manifest 自报 source hash 也不够，因为同一个 builder 可以
协调重写输入和输出摘要。

### 考虑过的方案与取舍

1. **只把 ledger SHA 写进 manifest：** 文件很小，但不能证明该 ledger 的 FashionIQ
   record 与当前 lock、license evidence 和用途一致。
2. **只遍历 accepted image：** 能快速生成 catalog 输入，但 2,416 条排除会从分母消失，
   无法证明没有回流。
3. **复制全部图片到 `clean/`：** 隔离直观，但复制约 1 GB 字节增加成本，而且仍不能
   自行解决授权和 TOCTOU。
4. **exact approval + 完整 disposition + create-only metadata bundle：** 先重算四个
   RAW scope，再调用 `require_approval(...)`，accepted 生成引用原始字节的 draft，
   excluded 只进入 disposition；发布前再重算输入并自验输出。采用该方案。

### 最终方案

新增 `build_fashioniq_image_adapter`。它要求外部 source-lock SHA、许可证据 SHA 和
`VerifiedSourceReviewLedger`，在任何 draft 产生前核对 source revision、lock SHA、
license ID/evidence SHA、三个用途和三项本地权限；即使 owner record 声称允许，也硬拒绝
remote embedding、再分发或公开 demo。adapter 对冻结 inventory 逐项生成唯一
disposition，只有 `accepted_asset` 能生成 `DatasetAssetDraft`，`excluded_locked`
永远没有 draft。bundle 以 create-only 方式发布 `dataset-assets.jsonl`、
`asset-dispositions.jsonl` 和绑定上述 trust root 的 canonical manifest；发布前复验
RAW/evidence，loader 还检查外部 manifest digest、精确文件集、payload hash、行数、
accepted/excluded 集合闭包以及 ASIN、路径、URL、product identity 的交叉一致性。

### 如何验证

- `uv run pytest tests/data/test_fashioniq.py -q`：`11 passed`。
- `uv run ruff check src/skillchain/data/fashioniq.py
  tests/data/test_fashioniq.py scripts/build_fashioniq_adapter.py` 通过。
- 测试证明不同 revision、缺少 purpose、缺少 local embedding permission、尝试启用
  outbound 权限以及 RAW tree hash 漂移都会在 output 发布前失败。
- 测试还证明给已发布的 `dataset-assets.jsonl` 追加一个字节后，formal loader 会因
  manifest payload digest 不匹配而拒绝。

### 剩余限制

上述事实由小型真实图片字节 fixture 验证，尚未在本工作树持有的 75,267 张真实 RAW 上
发布正式 adapter manifest，因此不能宣称 FashionIQ 真实 catalog 已完成。当前子 adapter
只关闭 image asset 与 exclusion disposition；caption triplet/style query、人工池化
Style gold、统一 AssetCatalog 和跨来源 leakage 审计仍是 C2 后续任务。C2 总门保持未通过。

### 30 秒回答

“FashionIQ 的 owner approval 不是一个 source-name 白名单。我让 adapter 在产出前重算
四个 RAW scope，并把 revision、source-lock、许可证据、用途和权限一起传给
`require_approval`。同时对 75,267 个 accepted 和 2,416 个 excluded 都生成 disposition，
只有 accepted 能成为 asset draft。发布采用 create-only canonical bundle，发布前后都
做输入和输出复验。11 条测试证明旧审批、少授权、越权 outbound、RAW 漂移和 payload
篡改都会失败；真实全量发布和 Style gold 仍待完成。”

### 2 分钟回答

“这个问题容易被误解成 ledger gate 已绿，所以 adapter 只要写 `source=fashioniq` 就行。
实际上 ledger 批准的是一个由 annotation、URL inventory、accepted images 和 exclusion
ledger 共同定义的 exact snapshot。原 downloader 能保证下载与排除覆盖，却没有正式
adapter；如果直接遍历磁盘图片，2,416 条排除会无声地从审计分母消失。我实现了独立
formal builder：先用外部 SHA 重算四个 scope，再把 revision、lock SHA、license ID、
evidence SHA、用途和本地权限一起交给 owner ledger 校验；remote embedding、redistribution
和 public demo 额外硬拒绝。然后为每个 inventory row 生成 disposition，accepted 才生成
引用原始字节的 `DatasetAssetDraft`，excluded 绝不回流。输出 manifest 绑定 policy、
ledger 和具体 review record，formal loader 复验精确文件集、hash、行数和
ASIN—路径—URL—product 关系。发布前再重算 source lock，防止长流程中的 TOCTOU。
目前 11 条定向测试通过，但尚未在真实 75,267 张图上发布，caption/style gold 和统一
catalog 也未完成，所以我只把它记为 C2 的一个已实现子任务，而不是宣称 C2 已关闭。”

### 证据入口

- `src/skillchain/data/fashioniq.py`
- `scripts/build_fashioniq_adapter.py`
- `tests/data/test_fashioniq.py`
- `specs/data_sources/c2/source-locks/fashioniq.source-lock.json`
- `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl`
- `docs/plans/2026-07-20-p0-p1-closure.md`

---

## 26. “用户确认当前 Codex 模型”不等于可复核的 Codex CLI Mock 生成会话

**状态：部分解决**

### 一句话问题

Phase 3 技能已经正确控制 staging、人工接受、计划字段和
`synthetic_derived` provenance，但它仍把一句 `5.6 Sol Ultra` 用户确认当作生成模型门；
若当前计划改为独立 Codex CLI 会话，这不足以证明实际启动了哪个模型、reasoning effort、
输入闭包、工具读取范围、重试次数或最终输出来源。

### 背景与影响

MVP 用 DuRecDial 2.0、CrossWOZ、MUGE 的抽象模式加 Codex 受控重写来替代不可取得的
JDDC 2.0。Mock 会成为 S1 的训练轨迹输入，因此生成执行面既影响数据 provenance，也影响
后续 `S1 − LLMStaticSkill` 的可解释性。交互式 Codex 主会话中的人工确认可以作为低等级
操作门，但不能自动升级为 Codex CLI session receipt。若 manifest 仍只写
`provider=codex`、`model_display_name=5.6 Sol Ultra` 和
`model_claim_source=user_confirmation`，读者可能把“用户自证”误读为“CLI 运行证据”。

### 观察到的证据

已验证事实：

- `generate-phase3-corpus` 在提交 `a80af22` 中随受控 Mock 方案更新，已经强制每批 25 条、
  只写 staging、禁止自动接受，并将 `data_origin=synthetic_derived` 写入 batch manifest；
  因此它不是整体遗留的旧流程。
- 当前 `status/stats` 为 `seed_status=missing`、0 accepted batch、0 accepted query、
  无 active plan；尚无正式 Mock 产物需要迁移。
- `guard-model` 的实现只是比较 CLI 参数是否等于字符串 `5.6 Sol Ultra`，不启动或检查
  Codex CLI，也不生成 session receipt。当前本机 `codex-cli 0.145.0`；项目已有独立证据
  表明其 catalog 模型标识为 `gpt-5.6-sol`，reasoning effort 需另行指定为 `high`。
- `BatchDraftManifest` 只保存 `provider`、显示名、用户确认、时间和数据输入摘要；没有
  Codex binary/version、requested model、reasoning effort、命令、session/process/event、
  retry/follow-up/tool activity 或 evidence tier 字段。
- `generation_input_sha256` 当前覆盖 plan items、plan、AssetCatalog、leakage policy 和
  accepted seed；`PlannedQuery` 不含 interaction-pattern inventory ID/hash。运行手册要求先
  建立只含结构标签的 pattern inventory，但技能的 Generate 步骤没有读取或绑定该工件，
  也没有绑定 dialogue policy、source-review ledger 或具体 TaskSpec 字节摘要。
- 定向回归
  `uv run --frozen pytest tests/skills/test_generate_phase3_corpus_skill.py
  tests/synthesis/test_cli.py tests/synthesis/test_seeds.py tests/synthesis/test_batches.py -q`
  为 `76 passed`。这证明现有契约内部一致，不证明它满足新的 CLI-session 证据目标。
- PowerShell `Get-Content` 曾显示中文乱码，但 Python 以 UTF-8 读取技能和 contract 得到正常
  中文；这是终端解码显示问题，不是技能文件损坏。

关于“当前 CLI Mock 计划要求多强证据”是设计判断；仓库尚未冻结专用 Phase 3 CLI mediation
protocol，因此不能把下面的修订方案描述为已实现。

### 根因

原技能的信任边界是“已确认模型的主 Codex 交互会话”，而新方向的执行边界是“由仓库启动、
约束并留证的独立 Codex CLI session”。两者都叫 Codex，但前者依赖人机上下文和用户声明，
后者需要冻结命令、输入、运行身份和输出证据。与此同时，JDDC 退出后新增的 source/pattern
治理只进入了 policy 与 runbook，没有完整进入 Phase 3 的机器可读 generation-input closure。

### 考虑过的方案与取舍

1. **原样保留技能并把用户确认称为 CLI 证据：** 改动最少，但会夸大模型身份、effort、
   session 和重试可见性，淘汰。
2. **完全复用一次性静态 Author v5 协议：** 已有成熟的 claim-before-spawn、terminal
   guard 和 canonical receipt，但其空 scratch、零 visible tool activity、单 final 等约束
   是为 LLMStatic authoring 设计，不能直接覆盖逐图查看和每批轨迹合成。
3. **继续使用交互式主会话，但显式降级证据：** 可用于探索性 seed/Mock，manifest 应明确
   `user_confirmed_interactive_session`，不能声称 CLI-mediated 或 provider-attested。
4. **新增 Phase 3 专用 CLI mediation protocol：** 冻结每批输入包、模型/effort、CLI
   binary、命令、允许的只读图像/工件范围、输出 schema 与 session receipt；成本较高，
   但与“用 Codex CLI 会话生成 Mock”的当前方向一致，推荐。

### 最终方案

本轮只完成审查，没有修改正式生成器。结论是：保留技能的 seed → plan → 25 条 staging →
人工 accept 骨架，但在第一次正式生成前把它升级为 v2：

1. 用专用 CLI runner 启动明确的 `gpt-5.6-sol` 和 reasoning effort，记录 binary/version、
   frozen command/env/input hashes、process/event/final-output 摘要及零 repository retry 的
   可观察事实；不可观察的 provider request ID、served revision、内部 retry 和费用继续写
   `unavailable`，证据等级保持 `platform-mediated_non-provider-attested`。
2. 生成输入包绑定 accepted seed、plan slice、TaskSpec、AssetCatalog/KB、dialogue policy、
   owner source-review ledger、interaction-pattern inventory 和允许读取的图像集合摘要。
3. 为每个 planned item 明确 `pattern_id`/组合策略，并把 inventory/policy/review 摘要纳入
   `generation_input_sha256`；禁止模型直接读取 restricted raw utterances。
4. 保留现有 immutable staging、revision、quality report、显式 accept/reject 和
   `synthetic_derived` 约束；不要把 Author v5 的一次性授权或 receipt 复用于 Phase 3。
5. 扩展技能 action taxonomy，允许 `audit/review`，避免把“是否过时”这类只读审查勉强归为
   corpus `status`。

### 如何验证

- 现有 76 项技能/CLI/seed/batch 测试继续通过。
- 新增负向测试：只给 `5.6 Sol Ultra` 字符串但没有 CLI receipt 时不得 formal stage；
  requested model、effort、binary、输入包、pattern inventory、policy 或 source-review hash
  任一漂移都必须 fail closed。
- 新增隔离测试：runner 只能读取冻结输入闭包；restricted 对话原文、inbox/rejected/accepted
  结果、评测标签和 Judge 工件不得进入 prompt 或可见路径。
- 新增 canonical replay：从磁盘重算 draft/result/manifest/session receipt 自哈希，并区分
  repository-observed facts 与平台不可观察字段。
- 在真实生成前再次运行 `status/stats`，预期在 seed 人工接受和 formal active plan 就绪前
  仍拒绝 batch composition。

### 剩余限制

专用 Phase 3 CLI runner、schema v3、pattern inventory 及其 adapter 尚未实现；当前也没有
accepted seed 或 active plan。因此现有技能可以继续承担只读状态和工作流设计参考，但不应
直接用于声称“已按当前 Codex CLI 会话计划可复核地生成正式 Mock”。此外，Codex CLI 的
served revision、provider request ID 和平台内部 retry 仍不可由仓库证明；升级协议只能诚实
提高本地执行证据，不能制造 provider attestation。

### 30 秒回答

“我审查了 Phase 3 技能，发现它的治理骨架并不过时：25 条一批、只进 staging、人工接受和
`synthetic_derived` 都已经跟上 JDDC 退出后的方案。但模型门仍只是用户确认
`5.6 Sol Ultra`，并不证明真正的 `gpt-5.6-sol` CLI session、effort、输入闭包或重试。
更关键的是 runbook 要求 interaction-pattern inventory，generation hash 却没有绑定它。
所以我的结论是‘骨架保留、执行层过时’，正式生成前应加专用 CLI receipt 和完整输入哈希；
现有 76 项测试通过只能证明旧契约自洽。”

### 2 分钟回答

“这个问题容易被版本日期误导。技能刚随受控 Mock 方案更新，已经正确标记
`synthetic_derived`，也有 seed、active plan、25 条 batch、revision、quality report 和人工
accept，所以不能说它整体废弃。真正的漂移发生在信任边界：技能假设人在当前 Codex 主会话
确认显示名，然后 `guard-model` 做字符串比较；新的计划却是独立 Codex CLI session。字符串
门不记录 binary、requested model、reasoning effort、命令、事件、工具读取、retry 或证据
等级。仓库对静态 Author 已经证明 CLI 可用，但那套一次性、空 scratch、零 tool 协议又不能
原封不动套到逐图 Mock。第二个缺口是数据输入闭包：runbook 要求从 DuRecDial/CrossWOZ/MUGE
提炼 restricted-safe pattern inventory，但当前 plan 和 generation hash 只绑定 plan、
AssetCatalog、leakage policy 与 seed，没有绑定 pattern inventory、dialogue policy、
source-review ledger 或 TaskSpec 字节。我的建议是保留治理骨架，新增 Phase 3 专用 CLI
runner/receipt，并把这些工件全部纳入每批 frozen input。不可观察的 provider 字段继续明确写
unavailable。这样既不夸大可复现性，也不会把受限原对话或评测信息泄漏进训练轨迹。”

### 证据入口

- `.agents/skills/generate-phase3-corpus/SKILL.md`
- `.agents/skills/generate-phase3-corpus/references/corpus-contract.md`
- `docs/mock-trajectory-runbook.md`
- `src/skillchain/synthesis/cli.py`
- `src/skillchain/synthesis/models.py`
- `src/skillchain/synthesis/batches.py`
- `tests/skills/test_generate_phase3_corpus_skill.py`
- `tests/synthesis/test_cli.py`
- `tests/synthesis/test_seeds.py`
- `tests/synthesis/test_batches.py`
- `specs/authoring/authoring-codex-mediation-protocol-v5.json`

---

## 27. 冗余运行状态也属于 adapter 的输入信任边界

**状态：已验证并修复**

### 一句话问题

FashionIQ adapter 的 fixture 全绿，但第一次真实发布在重哈希 75,267 张图片后才发现：
代码读取并要求 `summary.json`，owner-signed source lock 却没有批准该文件；一个冗余运行
摘要既越出了信任根，又让本可立即发现的静态错误浪费了一次昂贵全量扫描。

### 背景与影响

C2 要求 formal adapter 只消费 exact source lock 与 owner ledger 批准的字节。FashionIQ 的
冻结事实已经由 URL inventory、75,267 张 accepted image、2,416 条 `exclusions.jsonl`、
`exclusions-summary.json`、失败集合和 annotations 锁定；`summary.json` 只是下载器的可变
运行状态，包含绝对路径、PID、耗时和更新时间。若正式 adapter 读取它，就产生两种风险：
一是未审批字节影响 formal 输出，二是机器相关字段让重建失去可移植性。

### 观察到的症状与证据

已验证事实：

- 第一次真实命令没有发布任何 `data/clean` 半成品，终态为
  `FashionIQProvenanceError: FashionIQ source lock does not bind finalized exclusion state`。
- owner-signed lock 的 `exclusion_ledger` 精确覆盖五个文件，其中包含冻结 exclusions 和两次
  完整失败集合，但不包含 `summary.json`；adapter 的静态布局检查却把 `summary.json` 列为
  required。
- 失败发生前 `load_and_verify_required_source_lock` 已完成 75,267 张图所在 tree 的全量
  重哈希。静态 scope 不兼容本可在读取 canonical lock bytes 后立即拒绝。
- 修复后定向测试为 `18 passed`，Ruff 与 `git diff --check` 通过。
- 同一 owner lock、ledger 和 license evidence 上的真实重跑成功发布 75,267 条
  `DatasetAssetDraft`、77,683 条 disposition，2,416 条冻结排除没有 draft；独立 loader
  复载成功。manifest SHA-256 为
  `f96ed92185bb193e39654bea987e597d19d34c012428b6940a85a92c3208108a`。

### 根因

fixture 在构造 source lock 时顺手把 `summary.json` 也纳入 scope，因此只证明“代码与 fixture
自洽”，没有证明“代码与真实 owner-approved scope 一致”。同时，source-lock loader 把
canonical file/digest 检查与 RAW tree 重算捆在一个入口里，adapter 无法先做便宜的
adapter-specific layout preflight。

### 考虑过的方案与取舍

1. **把 `summary.json` 加进现有 lock 并协调重算摘要：** 会让 owner ledger 的 exact SHA
   失效；没有新 owner approval 时属于伪造授权，淘汰。
2. **新增包含 `summary.json` 的 source lock 并重新取得 owner approval：** 合规，但该文件
   主要是机器相关运行状态，不应扩大正式信任根。
3. **跳过 finalized 状态检查：** 能发布，但会失去 accepted/excluded 完整闭包，淘汰。
4. **只消费已批准的冻结事实，并前移静态 preflight：** 保留完整覆盖验证，又移除冗余、
   未批准、不可移植的输入；采用。

### 选择的方案

新增轻量 `load_required_source_lock`：只验证外部 lock SHA、canonical JSON 和 schema，不走
RAW tree。FashionIQ adapter 先用它检查 exact scope layout，再执行完整 RAW 重验；发布前仍
进行第二次完整重验和 approval TOCTOU 复核。acquisition coverage 被拆成共享实现：
下载器验收仍检查 `summary.json`，formal adapter 则只检查 owner-approved
`exclusions-summary.json`、exclusions、inventory 和 image coverage。正式路径仍要求
`complete_with_exclusions`、计数一致、accepted/excluded 互斥且并集覆盖全部 inventory。

### 如何验证

- `uv run pytest tests/data/test_source_lock.py tests/data/test_fashioniq.py -q`：
  `18 passed`。
- 新测试证明轻量 preflight 不会因 RAW 漂移而假装完成 full verification；完整 loader 仍会
  因相同漂移拒绝。
- 新测试把未获批准的 `summary.json` 改成无效文本，formal adapter 仍能从锁定事实正确发布；
  downloader 的原验收入口仍保留对 summary 的检查。
- 真实发布回执：
  `specs/data_sources/c2/adapters/fashioniq-image-adapter-v1.receipt.json`。
- 独立复载核对 manifest、两个 payload SHA、行数、accepted/excluded 闭包和外部
  manifest digest。

### 剩余限制

这只关闭 FashionIQ image adapter 的真实发布子门，不是 C2 总门。caption/style query
adapter、Style gold、统一 AssetCatalog、query/gallery leakage、其余来源 adapter、Mock
人审和 200-query mini 仍未完成。`data/clean` 是 Git 忽略的本地产物，跨主机重建仍必须持有
相同 RAW cache，并用仓库回执中的外部摘要复验。

### 30 秒回答

“FashionIQ 的 fixture 测试都绿，但真实发布发现 adapter 读取了 owner lock 没批准的
`summary.json`。它只是带 PID、绝对路径和时间的运行摘要，正式事实已经由锁定的 exclusions
summary、失败集合、inventory 和图片覆盖证明。我没有重算 lock 冒充 owner 审批，而是让
formal adapter 只消费现有批准范围，并把静态 scope 检查前移到 75k 文件全量哈希之前。修复
后 18 项测试通过，真实发布和独立复载得到 75,267 accepted、2,416 excluded 的闭包回执。”

### 2 分钟回答

“这次问题同时涉及授权边界和失败成本。第一次真实 FashionIQ 发布先重哈希了 75,267 张图，
然后才因 source lock 不含 `summary.json` 失败。fixture 之所以没暴露，是因为 fixture 自己
把 summary 放进了 lock；它证明了内部自洽，却没有覆盖真实 owner-approved scope。
`summary.json` 又恰好包含 PID、绝对路径和更新时间，是冗余且不可移植的运行状态。可选方案
包括重签新 lock、弱化 finalized 验证或缩小消费面。我选择缩小消费面：formal adapter 继续
严格验证锁定的 exclusions summary、完整排除集合、inventory 和实际 image coverage，
但不读取未批准 summary；下载器自己的验收仍检查它。与此同时我拆出 canonical lock 的轻量
loader，让 adapter-specific layout 在昂贵 RAW walk 前 fail fast，完整 loader 和发布前二次
重验仍保留。最终定向测试 18 项全绿，真实 bundle 的 manifest SHA 固定，独立 loader 复核
75,267 条 asset、77,683 条 disposition 和 2,416 条排除。这个结果只关闭 image 子门，不把
Style gold 或 C2 总门误报为完成。”

### 证据入口

- `src/skillchain/data/source_lock.py`
- `src/skillchain/data/fashioniq.py`
- `tests/data/test_source_lock.py`
- `tests/data/test_fashioniq.py`
- `specs/data_sources/c2/source-locks/fashioniq.source-lock.json`
- `specs/data_sources/c2/adapters/fashioniq-image-adapter-v1.receipt.json`

---

## 28. 小型 fixture 的标识符和文件类型不能代表官方全集

**状态：已验证并修复**

### 上下文、影响与可观察证据

ABO Exact Match 必须从 owner-approved compact archives 得到同 product 的真实多视图候选，
但不能让模型代替人类判断 catalog photo。第一次真实 packet 运行得到 0/50：listing 的长
image ID 需要 tar 内 `images/metadata/images.csv.gz` 映射到短 member path。补上映射后，
全集又暴露 image ID 合法包含 `+`、398,212 张图中有 2 张 PNG，以及 147,702 条 listing
存在重复 `item_id`。过窄 fixture 没覆盖这些形状；一次运行还因重复 item 的相同 hash score
让 heap 尝试比较 dict 而失败。每次失败均发生在 create-only 发布前，没有半成品。

### 根因

实现把“测试样例里见过的字符、JPEG 扩展名和唯一 item”误当成官方 schema。更深层原因是
把 tar member 的短文件名误当成业务 image ID，没有先读取同一 approved archive 内的官方
metadata 映射。

### 方案与取舍

- 直接按短 member ID 连接 listing：得到 0 候选，语义错误。
- 跳过不符合正则的行或重复 item：会静默改变全集和抽样分母，淘汰。
- 使用第三方 metadata mirror：会扩大 trust root 和许可审查，当前不需要。
- 读取 approved tar 内官方 CSV，支持实际 `+` 与 `.jpg/.png`，按 canonical shard/line
  首次出现去重，再用固定 identity hash 选样：采用。

### 选择、验证与剩余限制

新增 `abo_archive_review.py` 和 CLI。它完整映射 398,212 条 image metadata，扫描 147,702
条 listing，只基于 identity 选 50 个唯一 item；随后原样提取 main/other 共 100 张图片，
逐张解码并绑定 archive/member/listing/image SHA。独立 loader 复核 exact file set、每个
payload digest 和每 item 恰好 main+other。小型 tar 回归为 `2 passed`，真实 manifest
SHA=`719e91dd4293f7d7a36af6153ad7533e624e6fc0f5d79a50bb6472ebb06fba87`。

剩余限制是 packet 状态仍为 `pending_human_catalog_photo_review`。必须由项目所有者逐图
审核，再做 near-duplicate/global leakage；AI 或 identity metadata 不能把候选升级为 gold。

### 30 秒回答

“ABO fixture 只用了普通 ID、JPEG 和唯一 item，真实全集却有 `+`、两张 PNG、重复 item，
而 listing image ID 还必须经 tar 内官方 CSV 映射到短 member path。我的前三次真实运行都
fail closed，没有发布半成品。我最终用 archive 内映射、canonical 首次去重和 identity-only
固定 hash 生成 50 对、100 张待人审候选，并独立复载全部 SHA。关键是把候选准备与人工资格
决定分开，不能因 pipeline 成功就宣称 Exact gold 已就绪。”

### 2 分钟回答

“这次问题展示了 fixture 自洽不等于真实 schema。最初我把 image tar 的短文件名当成 listing
image ID，真实运行自然得到 0 对；官方 `images.csv.gz` 其实就在已批准 tar 内。接上映射后，
过窄正则又拒绝带加号的合法 image ID，接着发现全集有两张 PNG；再往后，重复 item 让相同
hash score 的 heap entry 落到 dict 比较。任何一个问题若用‘跳过坏行’解决，都会静默改变
抽样分母。我改成完整验证官方 CSV、支持观测到的官方类型、按 canonical 首次出现对 item
去重，并只用 identity hash 选样，不看像素。最终扫描 147,702 listing、映射 398,212 图，
发布 50 个唯一 pair。它仍只是人审 packet：catalog-photo、近重复和跨源 leakage 都必须独立
通过，才能进入 Exact Match 主结论。”

### 证据入口

- `src/skillchain/data/abo_archive_review.py`
- `scripts/prepare_abo_archive_review.py`
- `tests/data/test_abo_archive_review.py`
- `specs/data_sources/c2/adapters/abo-archive-review-v1.receipt.json`

---

## 29. “原始归档成员”不等于“原始分辨率图片”
**状态：已验证并修复**

### 一句话问题

ABO 的首版人工审核包逐字节复制了 `abo-images-small.tar` 的成员，provenance 没有造假，但审核界面把这种“archive member 原字节”误解成了“官方 original 图片”；真实首对只有
`125×256` 和 `195×256`，不足以承担最终 catalog-photo 审核。

### 背景与影响

Exact Match 需要人判断两张图是否都是真实商品照片。身份抽样可以在 small archive 上完成，但视觉裁决必须看到足够分辨率的官方对象，而且导出的决定必须绑定被审核的 original SHA，而不能绑定 256px derivative。
如果继续用小图，审核者可能把模糊、局部或文字图误判；如果只在 UI 中偷偷替换 URL，决定记录又会继续绑定小图哈希，产生“看的是 A、签的是 B”的审计断裂。

### 观察到的证据

- `src/skillchain/data/abo_archive_review.py` 明确消费 `abo-images-small.tar`；首对 packet 元数据为
  `125×256`、`195×256`。
- 官方逐图 `images/original/...` 对象可定向获取，无需下载整个 118,271,436,800 字节的
  `abo-images-original.tar`。
- 定向获取后的首对为 `1246×2560`、`1949×2560`；100 张 original 合计
  23,204,736 字节。
- 获取前的 formal model 校验还发现真实 image ID `71OkUYGf+eL` 被
  `SafeIdentifier` 错误拒绝；compact selector 已允许 `+`，两个边界不一致。
- 浏览器截图先复现了错误的内部滚动，再验证 `object-fit: contain` 后两张整图完整落入容器。

### 根因

“original”同时被用于两个不同概念：一是“没有被本仓库重新编码的 archive member”，二是
ABO 官方路径中的 `images/original` 分辨率资产。首版实现满足前者，却在面向审核者的语义上暗示了后者。fixture 又只使用不含 `+` 的 formal ID，掩盖了真实 schema 与 compact parser 的差异。

### 考虑过的方案与取舍

1. 继续使用 small 图，只增加“未裁剪”文案：不能解决视觉信息不足，淘汰。
2. 在浏览器直接嵌入官方 S3 URL：会突破当前禁止 remote embedding 的权限边界，也不能冻结响应身份，淘汰。
3. 下载完整 118 GB original tar：最完整，但对 50 对审核明显过重。
4. 按已冻结的 100 个 image path 定向获取官方 original，记录 URL、ETag、SHA、尺寸并发布新 packet：
   最小化下载且能把人审决定绑定到实际看到的字节，采用。

### 最终方案

新增 `abo_original_review.py` 和 CLI，从 compact packet 只继承 identity-only selection；对 100 个确定路径获取官方 original，逐图验证可解码性并记录 HTTP/content identity，自包含发布 create-only packet，同时复制到本地 RAW original cache。新 packet 用 formal
`ABOListingRecord`/`ABOImageManifestRecord` 重新计算 record SHA，状态严格保持
`pending_owner_scope_and_human_catalog_photo_review` 和 `formal_use_allowed=false`。
审核 UI 显示真实像素尺寸，默认 `object-fit: contain`，并提供单独的原尺寸链接。`SafeIdentifier`
同步允许官方 image ID 中已经实测存在的 `+`。

### 如何验证

- 新 original packet：50 对、100 张，manifest
  `ec741fb25b2cecb78c6f0b72f2ccda44c548b3696a32f571a09f01a0c81cb168`。
- download receipt 与 review packet SHA 固定在
  `specs/data_sources/c2/adapters/abo-original-review-v1.receipt.json`。
- `uv run pytest tests/data/test_abo_original_review.py tests/data/test_abo_archive_review.py tests/data/test_abo.py -q`
  返回 `41 passed, 1 skipped`；Ruff 通过。
- 浏览器 DOM 显示首对为 `1246×2560`、`1949×2560`，实际截图显示完整耳环与佩戴示意。

### 剩余限制

定向 original 文件扩大了真实 RAW 消费集合。当前 owner-approved compact lock 没有绑定这些新文件，因此本次只修复审核输入，不能自动升级为 formal source approval。项目所有者随后已完成
100 张人审，exact packet 对账和 canonical loader 复载通过，36/50 对保留；但仍需为精确 original
scope 生成新 lock 并由 owner 对新 SHA 明确批准，之后才能运行 near-duplicate/global leakage 和 Exact eligibility。

### 30 秒回答

“首版 ABO 包没有改写图片，但它复制的是 256px small archive。provenance 上的‘原始 member’被误当成了产品语义上的‘original image’，导致审核信息不足。我没有在 UI 里直接换远程 URL，因为那会让人看到的字节与决定绑定的 SHA 不一致，还违反 remote embedding 边界。我按冻结的 100 个路径定向下载官方 original，记录 ETag、SHA 和尺寸，重新计算 formal record hash，并让 UI 用 contain 完整展示。新包 23 MB、测试 41 通过，但 exact source lock 仍要 owner 对新 SHA 重新批准。”

### 2 分钟回答

“这个问题的关键不是 CSS，而是同一个‘original’词被用于两个信任层。第一版从 owner-approved
`abo-images-small.tar` 按 identity-only hash 选 50 对，并逐字节复制 tar member，所以没有篡改；但首对只有 125×256 和 195×256，不能冒充 ABO 的
`images/original`。如果只把 `<img>` 换成 S3 URL，review JSON 仍绑定 small SHA，等于审核者看 A、系统签 B，而且当前权限不允许 remote embedding。完整 original tar 又有 118 GB。
我选择按已冻结的 100 个 metadata path 定向下载官方对象，逐图记录 URL、ETag、SHA、真实尺寸和原字节；用 formal listing/image 模型重新计算 record digest，再发布自包含、create-only、明确
`formal_use_allowed=false` 的审核包。实现时真实 image ID 中的 `+` 又暴露 formal
SafeIdentifier 与 compact parser 不一致，我统一了约束并加回归测试。最后浏览器用
`object-fit: contain` 验证整图，无内部滚动，并提供原尺寸打开。这样修复了人审证据链，但没有伪造授权：新 RAW exact scope 仍需 owner 对新的 source-lock SHA 明确批准后才能进入 formal C2。”

### 证据入口

- `src/skillchain/data/abo_original_review.py`
- `src/skillchain/data/abo.py`
- `scripts/prepare_abo_original_review.py`
- `scripts/render_abo_catalog_review.py`
- `tests/data/test_abo_original_review.py`
- `tests/data/test_abo.py`
- `specs/data_sources/c2/adapters/abo-original-review-v1.receipt.json`

---

## 30. 次级问法标签不能替换论文的顶层实验意图

**状态：已验证并修复**

### 一句话问题

Phase 3 Skill 把 `exact_match/substitute/complement/attribute_query/scenario_query`
称为五个 canonical intent，但论文、冻结 taxonomy、运行时 schema、planner 和 validator
使用的是 `exact_match/multi_product/divergent_rec/encyclopedia/utility`；前一组不是后一组的
改名，而是把推荐子型和跨意图问法误升格成了论文实验分层。

### 背景与影响

项目初衷是 SkillChain 的公开数据机制级复现与受控扩展。论文 Table 1 用 Exact Match、
Multi-Product、Divergent Recommendation、Encyclopedia、Utility Assistance 定义五种
scene-specific Skill；离线 1,000 条 query、逐 intent 增量和 300 条 SBS 人评都按这五类分层。
公开数据和 capability 可以替代或细化，但若顶层 intent 被换掉，就无法重建论文的路由
ground truth、per-intent 指标和同类比较，因而不再只是“数据替代”，而是改变了实验问题。

本次按 Skill 生成 seed 后，`stage-seeds` 被核心 validator 拒绝。失败关闭避免了错误 seed
进入 staging/accepted，但也暴露出文档、数据政策和执行代码对“canonical”的定义不一致。

### 观察到的证据

已验证事实：

- 论文 PDF 第 3 页 Table 1 明确列出 Exact Match、Multi-Product、Divergent Rec.、
  Encyclopedia、Utility Assistance；第 4 页说明离线集对这五类做 intent-stratified sampling；
  第 5 页 Table 3 按同五类报告增量；第 6 页 SBS 为每类 60 条。
- 提交 `df86e50` 于 2026-07-11 08:32:57 定义 Phase 3 代码契约，
  `src/skillchain/synthesis/models.py` 从一开始就校验论文对齐的五类。
- 提交 `6cfa96a` 于同日 08:47:40 单独新增 Phase 3 Skill，没有修改 synthesis 代码，却在
  `corpus-contract.md` 首次引入另一组五类；该提交之前全仓没有
  `attribute_query/scenario_query` 或字面量 `substitute/complement` 的对应 taxonomy 记录。
- 提交 `aebc7b1` 后冻结的 `ecommerce-mvp-taxonomy-v0.json` 仍保留论文五类，并在其下定义
  六个 MVP capability；其中两个 utility capability 共同归入顶层 `utility`。
- 提交 `a80af22` 新增 dialogue policy 时，一边声明
  `semantic_authority=task_spec`，一边把 Skill 中的错误五类手工复制为
  `canonical_intents`，把局部 Skill 漂移传播到了第二份机器可读规范。
- `tests/skills/test_generate_phase3_corpus_skill.py` 检查模型门、staging、人工接受和占位文本，
  但没有检查 Skill 中的 intent 名称；`tests/test_data_source_portfolio.py` 检查 Mock 来源和
  `synthetic_derived`，也没有把 policy intents 与 taxonomy 比较。synthesis 测试则只验证
  正确的代码侧五类，因此两边都能各自通过。
- 本次错误草稿只存在于 ignored inbox，SHA-256 为
  `a41405dc755ead4c200e3abf5ecd9e7c30af4901c242c92699b8b72d80d3522b`；对应 staging 和
  accepted seed 从未存在。修复时已删除该无效草稿，没有做静默映射或重分类。
- Phase 3 Skill、corpus contract、dialogue policy 和 Mock runbook 现已明确列出论文五类；
  policy 还绑定 exact taxonomy/TaskSpec version 与 SHA-256。

关于最初作者为何选择新五类，git 只能证明它们首次出现的位置，不能证明人的主观来源。
根据标签语义和后续传播路径，以下属于证据支持的推断：Skill 编写时把“电商问法/推荐关系”
这一横切维度误当成“论文路由意图”，而缺少单一权威与跨工件测试使错误长期未被发现。

### 根因

1. **概念层级混淆。** `substitute` 和 `complement` 都可以是
   `divergent_rec` 的子型；`attribute_query` 可能落入 exact match、encyclopedia 或 utility；
   `scenario_query` 描述问法情境，也可能跨 recommendation 和 utility。它们既不互斥，也
   不能覆盖论文五类。
2. **权威被手工复制。** taxonomy/TaskSpec 已是任务语义权威，但 Skill Markdown 和
   dialogue policy 又各写一份 `canonical_intents` 常量，没有绑定 taxonomy version/hash。
3. **测试只验证局部性质。** Skill 测试关注安全工作流，代码测试关注 validator，policy 测试
   关注来源；没有一项 contract/integration test 让“按 Skill 写出的最小 seed”实际通过
   `stage-seeds`。
4. **后续变更放大旧错。** JDDC 退出后的 Mock 政策引用了旧 Skill 标签，却没有重新对照论文
   和冻结 taxonomy；runbook 只写“五个 canonical intent”而未列名，文本审阅也难以发现。

### 考虑过的方案与取舍

1. **放宽 validator，改用 Skill 的五类：** 改动看似最小，但会删除论文的
   Multi-Product/Encyclopedia/Utility 分层，并把 Divergent Rec. 拆成两个非等价桶；无法支持
   论文逐 intent 比较，淘汰。
2. **在 staging 前做一次性名称映射：** 不存在可靠一一映射，例如
   `attribute_query/scenario_query` 都是多对多；静默映射会制造错误 ground truth，淘汰。
3. **保留论文五类为唯一顶层 intent，把新标签降为正交 metadata：** 能保持论文对齐，也允许
   受控研究 substitute、complement、属性追问、场景化表达的覆盖；采用为修复方向。

### 最终方案

本轮按以下顺序完成修复：

1. 以论文 Table 1 和冻结 taxonomy 为顶层权威，继续使用
   `exact_match/multi_product/divergent_rec/encyclopedia/utility`；不要修改核心 validator
   去迁就错误 Skill。
2. Phase 3 Skill 的操作步骤、质量边界和 seed contract 示例全部改为论文五类；Mock runbook
   同时列名，并明确 pattern inventory 是正交维度，不得替换 `canonical_intent`。
3. dialogue policy 的 `canonical_intents` 改为论文五类，并增加 taxonomy/TaskSpec 的
   exact version 与 SHA-256 绑定，避免“声称 authority 却手工漂移”。
4. 保留六 capability 的受控扩展：四个 intent 各一个 capability，`utility` 下分
   document reading 与 recipe guidance；报告中同时给论文五类和项目六 capability，不能把
   capability 扩展冒充论文原 taxonomy。
5. 新增三层回归门：Skill seed 示例必须等于 runtime `INTENT_IDS` 并能真实 staging；旧漂移
   标签必须被 validator 拒绝且不创建 staging；dialogue policy 必须与冻结 taxonomy/TaskSpec
   的集合、版本和摘要一致。
6. 删除未进入 staging 的错误 inbox 草稿，不重写、不映射，也不生成替代 seed。

### 如何验证

- `tests/skills/test_generate_phase3_corpus_skill.py` 从 Markdown seed 示例提取真实 JSON，
  比较 runtime `INTENT_IDS`，并调用 `stage_seed_batch` 完成端到端机械 staging。
- `tests/synthesis/test_seeds.py` 明确回归旧五类，验证 `substitute` 等被 Pydantic intent
  `Literal` 拒绝且不产生 staging 目录。
- `tests/test_data_source_portfolio.py` 逐字段比较 dialogue policy 与冻结 taxonomy/TaskSpec
  的 intent 集合、version 和 SHA-256。
- 三组定向测试返回 `21 passed`；Ruff 和 `git diff --check` 通过。
- 修复后的论文五类 seed 草稿已按正式 Skill 流程进入
  `data/queries/seeds/staging/phase3-paper-five-seeds-20260725-r1`，其
  `seed_set_sha256=767a10caa954890088efe85a3b5ba50628e0ab772bafdf06b6b43dc7159c3250`。
  只读 `status` 返回 `seed_status=staging`、0 accepted batch、0 accepted query；
  这证明修复后的合同可以真实落入待审区，同时没有越权接受正式语料。
- 2026-07-26，owner 在独立后续回合逐字给出上述 batch ID 与 SHA 并明确接受；
  `accept-seeds --confirmation ACCEPT` 将其提升到 `data/queries/seeds/accepted`。随后
  `status/stats` 返回 `seed_status=accepted`、空 staging、0 active plan、0 accepted
  batch、0 accepted query，证明人工门已关闭且接受操作没有隐式生成或接受 query。

### 剩余限制

论文使用生产流量按 intent 分层，而本项目使用公开数据和受控合成；恢复五类顶层口径后仍
只能声称机制级复现，不能声称复现论文流量分布或绝对分数。精确 seed 已由 owner 接受，
但仍没有 active plan、query batch 或 accepted query；生成和逐批接受必须作为后续独立
操作，不能从 seed acceptance 推断。若未来重新引入
`substitute/complement/attribute_query/scenario_query`，必须先另行预注册正交字段、
多标签规则和覆盖配额，不能在看过实验结果后补定义。

### 30 秒回答

“第一次按 Phase 3 Skill 生成 seed 时，真实 validator 拒绝了它。我追 git 发现代码从
Phase 3 初始提交起一直使用论文五类，但 15 分钟后新增的 Skill 单独写了另一套电商问法，
后来 policy 又复制了它。根因是把推荐子型和横切问法误当成论文实验轴，同时缺少跨工件测试。
我没有放宽 validator，而是把 Skill、policy 和 runbook 恢复到论文五类，绑定 taxonomy/
TaskSpec 摘要，并加入真实 staging 与旧标签拒绝测试；21 项通过。错误草稿从未进入 staging，
现已删除。”

### 2 分钟回答

“这次冲突是一个典型的本体层级和单一权威问题。论文 Table 1 的五类不仅是展示名称，也是
Stage 2 路由标签、1,000 条离线集的分层单位、Table 3 的逐意图指标和 300 条 SBS 的平衡
单位。项目冻结 taxonomy 与运行代码都保留 exact match、multi-product、divergent
recommendation、encyclopedia、utility，并在它们下面做六 capability 的公开数据扩展。

但 Phase 3 Skill 的独立 Markdown contract 从首次提交就写成 exact match、substitute、
complement、attribute query、scenario query。它们不是一一改名：substitute 和 complement
属于 divergent recommendation 的子型，attribute/scenario 又跨多个顶层 intent。Skill 测试
只查模型门和 staging，synthesis 测试只查代码 validator，二者都绿；JDDC 退出时，新 policy
又复制了错误列表，即使同一对象里还写着 TaskSpec 才是语义权威。

真实 `stage-seeds` 首次把两套合同接起来，因此 fail closed。我的判断是不能为了让草稿通过
而改 validator，否则项目会从论文机制复现滑向另一项任务。我让论文五类继续作为唯一顶层
实验轴，六 capability 保持受控扩展；Skill 示例经真实 `stage_seed_batch` 回归，policy 与
taxonomy/TaskSpec 的集合、version、SHA 全绑定，旧标签有专门负向测试。错误 inbox 草稿按
原 SHA 核对后删除，没有映射成新的 ground truth。21 项定向测试、Ruff 和 diff check 通过；
修复交付时状态停在 seed staging 且 accepted 仍为零，因此没有越权接受正式语料。后续
owner 在独立回合按精确 ID/SHA 接受 seed 后，系统仍保持 0 active plan/batch/query，
证明 acceptance gate 没有被折叠成生成或 query 批准。次级标签以后若要保留，必须作为
另行预注册的多对多 interaction metadata。”

### 证据入口

- [SkillChain 原论文（arXiv）](https://arxiv.org/abs/2606.12984)
- `docs/reproduction-contract.md`
- `specs/taxonomy/ecommerce-mvp-taxonomy-v0.json`
- `specs/data_sources/dialogue-trajectory-source-policy-v1.json`
- `.agents/skills/generate-phase3-corpus/references/corpus-contract.md`
- `src/skillchain/schemas.py`
- `src/skillchain/synthesis/models.py`
- `src/skillchain/synthesis/planning.py`
- `tests/skills/test_generate_phase3_corpus_skill.py`
- `tests/synthesis/test_seeds.py`
- `tests/test_data_source_portfolio.py`
- commits `df86e50`, `6cfa96a`, `aebc7b1`, `a80af22`

---

## 31. 人工批准数量不等于防泄漏后的可用容量

**状态：已验证；35-pair 容量子门已关闭，formal C2 仍未关闭**

### 一句话问题

ABO v1 的 36 个逐对批准项经全局冲突审计只剩 33 对；扩展到 60 对并完成增量人审后，44 对
两图均获批，确定性最大独立集保留 41 对，关闭了 `dev_mini` Exact Match 的 35-pair 容量
子门，但没有据此冒充 formal 数据已经就绪。

### 背景与影响

Exact Match 的一个样本由同商品的两张图组成。人工审核回答的是“这两张图是否都是可辨认的
目标商品图”，并不自动回答“这个 pair 是否与其他 pair 独立”。v1 如果只按 36 个获批
pair 计数，同一张图片会以不同商品记录重复进入实验，既夸大有效样本量，也会让后续
grouped split 或检索评测发生直接泄漏。另一方面，扩容 packet 发布也不等于扩容成功：
新增图片仍须真实人工审核，最终还要在完整批准集合上重跑同一冲突图。

### 观察到的证据

已验证事实：

- v1 原图 packet 绑定 50 对、100 个图片引用，manifest SHA-256 为
  `ec741fb25b2cecb78c6f0b72f2ccda44c548b3696a32f571a09f01a0c81cb168`。
- owner 人审留下 36 对，human-review manifest SHA-256 为
  `a78b7c07f7ef7b08c115eb378a600183a0574f83192106856d97e9fd618b55b7`。
- 对批准集合重新计算文件 SHA-256 和 EXIF-transpose RGB 64-bit pHash 后，72 个图片引用只
  对应 69 张唯一图片；pair 内冲突为零，跨 pair 冲突为三条。
- `B07TC4PLKR/B07TGZZMCB` 复用 `61oI69Yt4GL`，
  `B08541R61V/B0856BQCS9` 复用 `61SE4RTPjdL`，
  `B08541S9JK/B0856BBVCB` 复用 `61+woWTqkwL`；三组均为 content SHA 和 pHash 距离
  零的直接重复，不是阈值边缘判断。
- 精确最大独立集留下 33 对，确定性排除 `B07TGZZMCB`、`B0856BBVCB`、
  `B0856BQCS9`。审计 receipt 文件 SHA-256 为
  `8b0693aa3eaebddbda113a2ea0dbc1de662d1cc26ec32f69387eff2a5a316e9a`，
  并明确标记 `formal_use_allowed=false`。
- 为吸收拒绝和再次冲突的损耗，v2 packet 确定性扩到 60 pair。original v2 精确复用 95 个
  v1 唯一原图对象、官方下载 18 个缺失对象；carry-forward 重新验证 v1 packet 与
  human-review 两个外部 SHA，把 100 条旧决定作为权威子集只读携带。compact/original/carry
  manifest SHA 分别为
  `717451160e71a5c7ff212197f758f2cc016718076609db6d379473e389d180ad`、
  `c168612971c36922368e6e6c6edda1adec699deaf4ee2b42a0e9a0ff00e6b69a`、
  `3dfb89156ae274a58e2ba651d1856842cf4e40dd36e1a36e623a7d425cdd7794`。
- owner 已完成 v2 的 20 张新增图审核：18 张
  `approve_catalog_product_photo`、零张 `reject_auxiliary_graphic`、2 张
  `reject_non_product`。完整导出为 120 行，文件 SHA-256 为
  `ba66b97b2ba310ce0bb001ea0ab8138305fdefacbccc30eaf13a4ebd0299a6af`。
- v2 human-review manifest SHA-256 为
  `50f9f04c55a4ad096e50e4fb993e3a008a83208fd546208280390bdc08d10c6e`；
  60 个候选 pair 中有 44 对的两张图均获批。
- v2 全局 pair audit 仍只发现原来的三条跨 pair 直接重复，pair 内冲突为零；确定性排除项
  仍为 `B07TGZZMCB`、`B0856BBVCB`、`B0856BQCS9`，最终保留 41 对，高于冻结目标 35。
  audit receipt 文件 SHA-256 / `receipt_self_sha256` 分别为
  `eed94599fc72af2090ab7368fd8574a87cb2eccde4c1b6203a873bc259d152e1` /
  `6fd4398b4ec3874816d92582743da95b968885a70a7fb52bc0b67a83c8559a0e`。
- finalization receipt 文件 SHA-256 / `receipt_self_sha256` 分别为
  `10796ef09b6911c29b88863fea07bb19af7b3632d1019c2c5b6f66c5ebdcb06c` /
  `84d089ba9c4b800884b0206540f7efbb7c82b5f1b254ec5db3a391ed269110bf`；
  它明确记录 `formal_use_allowed=false`，状态为
  `exact_match_capacity_gate_closed_pending_exact_scope_approval`。

### 根因

候选 pair 是按商品记录和主图/其他图关系构造的，但 ABO 元数据中的不同商品记录可以引用
同一个 image ID。人审 UI 以 pair 的视觉质量为中心，没有把全局图片复用图展示为资格冲突。
因此“每个 pair 单独合格”与“所有 pair 组成无泄漏集合”是两个不同约束。

### 考虑过的方案与取舍

1. **直接使用 36 对：** 最快，但把明确重复计入容量并破坏独立性，淘汰。
2. **遇到冲突就按扫描顺序删除：** 可以得到一个无冲突集合，但不保证保留数量最大，顺序
   变化还可能改变结果。
3. **只按 image ID 去重：** 能处理本次三条冲突，却漏掉不同 ID/不同编码的近重复图。
4. **建立冲突图并求精确最大独立集：** pair 为节点，任意图片 content SHA 相同或 pHash
   距离不超过冻结阈值即连边；分连通分量精确求解，并用字典序打破并列。实现稍复杂，但能
   同时给出最大容量、稳定选择和可审计排除原因，因此采用。
5. **降低 35-pair 门槛或复用冲突样本：** 能快速制造“通过”，但会改变预注册比较并引入
   直接泄漏，因此明确拒绝；选择补充真实候选、人工审核，再按原规则审计。

### 最终方案

新增 create-only 的 ABO pair leakage audit。它不信任已有摘要，而是重新验证 packet、人审
manifest、ledger、完整文件集合和每张图的内容；拒绝 symlink、junction 和路径逃逸；在
冻结的 pHash policy 下构建冲突图，求确定性精确最大独立集，并输出自哈希 receipt。资格
不足时通过新增候选和独立人工补审扩容，而不是降低 35 对目标或复用冲突样本。

增量补审也不把旧决定复制成一个自称可信的新文件：original v2 只复用由旧 packet 文件描述、
image ID/path、ETag、尺寸和内容 SHA 共同验证的字节；carry verifier 每次同时复验 target、
旧 packet、旧 human ledger 和所有外部 SHA，并要求 carried ledger 等于权威旧决定在 target
上的完整精确子集。新身份不继承决定；浏览器状态键同时绑定 target 与 carry manifest，避免
同一 target 更换继承包时复用旧 localStorage。

owner 完成新增 20 张图后，finalizer 要求导出决定精确覆盖 v2 的全部 120 个身份，并逐行核对
listing/image/source SHA；随后 pair audit 在新的完整 human ledger 上重新计算内容指纹和
冲突图。结果是 44 对逐对批准、41 对无冲突保留，因而只把 `minimum_retained_pair_count=35`
这个容量子门标为通过；所有 formal 资格继续保持 false。

### 如何验证

- `tests/data/test_abo_pair_audit.py` 覆盖直接重复、近重复、确定性最大独立集、输入摘要漂移、
  不安全路径和 create-only 输出。
- replenishment 测试覆盖伪造 carry/self-rehash、遗漏可携带决定、错误 source SHA、同身份
  内容漂移、链接/junction、create-only、浏览器状态隔离和 tracked receipt self-hash；ABO
  增量聚焦集合为 `16 passed, 2 skipped`，两项跳过均是本机无 symlink 创建权限。
- finalize 重新验证 120 行决定的完整覆盖和身份摘要；新增 20 行的 18/0/2 分布及完整 export
  SHA 均写入自哈希 finalization receipt。
- v2 audit 由 original packet SHA 和 v2 human-review manifest SHA 外部绑定后重算，稳定得到
  44→41、三条跨 pair 冲突和相同的三个排除 ID；audit receipt 与 finalization receipt 的
  文件 SHA、自哈希和交叉字段一致。

### 剩余限制

当前只关闭了 Exact Match 的 35-pair 容量子门。扩展后的 original bytes 尚未进入精确
source scope 并取得 owner approval，formal normalized adapter、全局 AssetCatalog、
Exact Match eligibility 和跨 split/global leakage 审计也尚未完成。所有 v2 receipt 继续
标记 `formal_use_allowed=false`；41 对不能直接写成 formal gold 或 C2 已通过。

### 30 秒回答

“ABO v1 人审显示 36 对合格，但全局内容 SHA 和 pHash 冲突图发现三组共享 gallery 图，
精确最大独立集只剩 33，低于 35 的目标。我把候选确定性扩到 60 对，严格携带 100 条旧决定，
再由 owner 审核新增 20 张图。最终 44 对逐对批准、41 对通过同一冲突审计。这个结果只关闭
容量子门；source scope、adapter、AssetCatalog 和全局 leakage 仍保持未完成。”

### 2 分钟回答

“这里有两个容易混淆的资格问题：人工审核确认一个 pair 的两张图是否都是目标商品图，但
实验独立性要求所有 pair 之间也不能共享或近重复图片。ABO 的不同商品记录会复用 image ID，
所以 36 个逐对批准项只有 69 张唯一图片。审计重新读取原始字节，对 EXIF 校正后的 RGB 图
计算 64-bit pHash，并以 pair 为节点建立冲突图；content SHA 相同或 pHash 距离在预注册
阈值内就连边。三条真实边都是同 image ID、同 SHA、pHash 距离零。为了不让遍历顺序决定
结果，我按连通分量求精确最大独立集，再用 pair ID 字典序处理并列，最终稳定保留 33 对。
receipt 绑定 packet、人审 ledger、每张图指纹和算法版本，并明确禁止 formal 使用。

为了补容量，我把候选确定性扩到 60 对，但没有让用户重审全部 120 张图。original 层只复用
95 个经旧 packet 文件描述、ETag、尺寸和 SHA 共同验证的唯一对象，并下载 18 个缺失对象；
decision 层每次复验旧 packet 与旧 human ledger，把 100 条旧决定作为完整精确子集只读携带，
只留下 20 张新图。owner 的新增决定是 18 张批准、2 张非商品拒绝；finalizer 对 120 行完整
导出逐行核对身份摘要，得到 44 个两图均批准的 pair。随后用同一冻结 policy 重跑冲突图，
仍只有原来的三条边和三个排除项，最终稳定保留 41 对，高于 35。

我把 finalization 与 audit 的文件 SHA、自哈希、human manifest 和导出 SHA 交叉绑定，同时
继续写明 `formal_use_allowed=false`。因为容量够了只代表样本数量子门关闭；exact source
scope 的 owner approval、normalized adapter、AssetCatalog/eligibility 和跨 split/global
leakage 仍是独立门。这件事说明质量审核计数不能替代集合级独立性证明，增量复用也必须重新
建立权威链，而且一个子门通过不能被包装成整条数据链通过。”

### 证据入口

- `src/skillchain/data/abo_pair_audit.py`
- `src/skillchain/data/abo_review_replenishment.py`
- `scripts/audit_abo_original_pairs.py`
- `scripts/prepare_abo_review_carry_forward.py`
- `scripts/render_abo_catalog_review.py`
- `tests/data/test_abo_pair_audit.py`
- `tests/data/test_abo_review_replenishment.py`
- `specs/data_sources/c2/adapters/abo-original-pair-leakage-audit-v1.receipt.json`
- `specs/data_sources/c2/adapters/abo-original-pair-leakage-audit-v2.receipt.json`
- `specs/data_sources/c2/adapters/abo-original-review-replenishment-v2.receipt.json`
- `specs/data_sources/c2/adapters/abo-original-review-v2-finalization.receipt.json`
- `data/clean/abo-original-review-v1/manifest.json`
- `data/clean/abo-original-human-review-v1/manifest.json`
- `data/clean/abo-original-human-review-v2/manifest.json`

---

## 32. 前向契约升级不能改写冻结历史

**状态：已验证并修复**

### 一句话问题

论文五类意图修复要求 Phase 3 新语料统一使用 TaskSpec v1，但把全项目默认 loader 直接改成
v1 会让已经冻结并带哈希的 v0 authoring/replay 工件失去原语义，造成“新合同修好了、历史
证据却被重解释”的可复现性回归。

### 背景与影响

TaskSpec v1 修正了 Multi-Product 的执行语义：唯一 operator 是
`multi_product_search` composite tool，而不是把私有 detect→crop→retrieval 子步骤暴露给
模型。Phase 3 planner、labeling 和 splitting 必须使用这个前向合同；与此同时，仓库保留
若干按 v0 序列化、签名或已消费的历史 authoring 工件。历史文件的意义应由其显式版本决定，
不能随当前默认值变化。

### 观察到的证据

- Phase 3 新默认现在绑定 `ecommerce-task-spec-v1`，摘要为
  `f8d5596de5d0ba98235f82c7c176a5b774b33d7bdd7e84fb00a07b5b0b7a7f0d`。
- taxonomy 仍为 `ecommerce-mvp-taxonomy-v0`；顶层论文五类没有随 TaskSpec 版本升级而改名。
- planner、labeler、splitter 和 dialogue source policy 均显式绑定 v1；Multi-Product 只使用
  `multi_product_search`。
- 共享 `load_default_task_specification()` 继续返回 v0，供冻结历史和显式 legacy replay；
  `CapabilityAssignment` 仍能解析带 v0 identity 的历史值。
- Phase 3/TaskSpec/policy 定向回归为 `181 passed, 1 skipped`，冻结 authoring/replay 额外
  `6 passed`；协议摘要测试 `5 passed`。

### 根因

“默认版本”同时承担了两个不兼容职责：一是决定未来新工件使用什么合同，二是帮助没有显式
路由的旧代码加载历史工件。若只改一个全局函数，调用点不会区分“创建新数据”和“重放旧
证据”，版本升级就会变成静默迁移。

### 考虑过的方案与取舍

1. **全局默认直接切 v1：** 修改少，但会改变旧调用的解释，淘汰。
2. **永久停留 v0：** 保护历史，却让新 Phase 3 继续产生已知错误的 Multi-Product 合同，
   淘汰。
3. **复制整套 synthesis 模块为 v1：** 隔离最强，但产生长期双轨和修复漂移。
4. **按生命周期分流 loader：** 新 Phase 3 入口显式使用 v1 常量/loader，历史通用 loader
   保持 v0；持久工件继续携带 version + SHA。改动范围可控且语义清晰，因此采用。

### 最终方案

把 TaskSpec v1 定义为 Phase 3 的前向默认，而不是仓库所有历史的追溯默认。planning 暴露
Phase 3 专用 version/SHA 常量，labeling 和 splitting 从同一绑定解析 capability；policy
同时记录 taxonomy 与 TaskSpec 的 exact identity。所有新 query 写入 v1，旧 replay 必须由
工件中的显式 v0 identity 加载，不做就地重写或哈希更新。

### 如何验证

- 测试检查新 plan/query 的 TaskSpec v1 版本与摘要、五 intent capability 映射，以及
  Multi-Product 不得再出现旧 operator。
- 单独重放冻结 v0 authoring 路径，确认 scoped migration 没有污染历史。
- `docs/reproduction-contract.md` 与 `docs/evaluation-protocol.md` 的 normalized SHA 已重算，
  `tests/test_protocol_hashes.py` 返回 `5 passed`。

### 剩余限制

双版本在过渡期要求新增调用点明确说明是在创建前向工件还是重放历史。未来若删除 v0 loader，
必须先迁移所有未冻结的消费者，并保留可执行的历史重放环境；不能只因主干已用 v1 就改写
已消费 receipt。真实 C2/C3 尚未完成，所以 v1 的端到端工具效果仍待真实数据验证。

### 30 秒回答

“修复 Multi-Product 合同时，我没有把全局默认从 v0 粗暴改成 v1，因为历史 authoring
工件已经按 v0 冻结。我的做法是把 v1 设为 Phase 3 前向默认，planner、labeler、splitter 和
policy 都绑定同一个 version/SHA；通用 legacy loader 保持 v0，按工件显式 identity 重放。
这样 181 项新路径测试和 6 项历史回放都通过，升级没有改写证据。”

### 2 分钟回答

“这是典型的版本生命周期问题。TaskSpec v1 必须上线，因为论文对齐的 Multi-Product 应只
暴露 composite `multi_product_search`，不能让模型提供任意 bbox/crop。但仓库也有已签名、
已消费的 v0 authoring 和 replay 工件；如果修改一个共享 default loader，旧 JSON 即使字节
没变，也会在新代码里被解释成另一份合同，破坏哈希之外的语义可复现性。

我把调用点按生命周期拆分：Phase 3 生成路径使用专用 v1 loader 和固定 SHA，taxonomy 继续
v0，因为顶层论文意图没有变；labeling/splitting 从同一 TaskSpec 解析，policy 也绑定 exact
identity。历史共享 loader 不变，显式 v0 工件继续原样重放。相比复制整套模块，这种 scoped
migration 避免双轨代码；相比全局切换，它不会静默重解释证据。验证同时覆盖 181 项新路径、
6 项冻结历史和协议摘要。剩余工作是保持每个新调用点都显式选择生命周期，并在真实 C2/C3
完成后再验证 composite tool 的端到端效果。”

### 证据入口

- `src/skillchain/synthesis/planning.py`
- `src/skillchain/synthesis/labeling.py`
- `src/skillchain/synthesis/splitting.py`
- `specs/data_sources/dialogue-trajectory-source-policy-v1.json`
- `docs/reproduction-contract.md`
- `docs/evaluation-protocol.md`
- `tests/test_task_spec.py`
- `tests/test_data_source_portfolio.py`
- `tests/synthesis/`

---

## 33. Stage 1 要保留完整交互，但不能偷看后阶段反馈

**状态：输入边界已验证，真实运行待完成**

### 一句话问题

把 Stage 1 的 “User Trajectories” 简化成最后一条用户文本会丢失真实澄清交互；反过来把
Judge、失败归因或工具执行结果也塞进去又会让 S1 提前获得 S2/S3 才产生的信息，二者都会
偏离论文。

### 背景与影响

论文 Stage 1 的输入是 Task Specification、用户交互轨迹和当前 Skill Bank/参考资料，随后
才进入 Creator、Engineer Loop 和 Human Reflection Gate。项目的合法轨迹既可能是单轮
`[user]`，也可能是 `[user, assistant, user]`。assistant 的澄清问题属于用户交互本身；
Judge 分数、failure attribution、tool trace 和冻结测试结果则是执行/评价阶段信息。两类
assistant 内容不能因为角色名相同而混为一谈。

### 观察到的证据

- 对论文 PDF 第 3 页的文字与流程图复核确认，Stage 1 先消费 TaskSpec 和 User
  Trajectories，再进行 Creator/Engineer/Human Reflection；S2/S3 反馈不是 S1 的前置输入。
- 初版实现曾只投影 user turns，并误嵌 Qwen `AuthoringInput` 语义包；代码审查在生成任何
  accepted S1 工件前发现并纠正。
- 正确公共输入是
  `specs/authoring/authoring-packet-codex-high-v5.json`，文件 SHA-256 为
  `b389da568575e7b1502923db0a4667e70954233924623856b4f96d6a4fdf9cf1`；
  它绑定 `gpt-5.6-sol/high`、TaskSpec v1、同一预算和参考资料边界。
- 新 `TrajectoryBundle` 逐字保留合法 `Query.turns`，只允许 `opt_pool`，要求五类论文意图
  全覆盖，并在 bundle 层绑定 taxonomy/TaskSpec version。
- 严格 schema 不含 `label_provenance`、Judge、failure attribution、tool trace、gate 或
  test 字段；`S1CreatorPacket` 还要求 trajectory version 与公共 Codex 输入完全一致。
- 输出状态固定为 `prepared_not_invoked`；CLI 明确报告 `model_invoked=false`、
  `bank_generated=false`。

### 根因

一是把“trajectory”误理解成对用户文本的扁平语料，而不是有角色和顺序的交互序列；二是
按 `assistant` 角色做粗粒度信息隔离，没有按信息产生阶段区分“澄清问题”和“运行反馈”；
三是仅凭文件名选择 common packet，没有用类型、模型/预算语义和外部 SHA 验证真正共享的
静态作者输入。

### 考虑过的方案与取舍

1. **只保留最后一条 user message：** 输入最小，但删除澄清上下文，可能让 Creator 学到
   不真实的单轮策略，淘汰。
2. **保留整个 Query 或执行记录：** 方便，但把 label provenance 和未来阶段信号带入 S1，
   破坏阶段隔离，淘汰。
3. **按白名单投影交互序列：** 保留 query/asset/image、完整合法 turns、顶层 intent 和
   capability；其他字段默认不存在。再把公共 Codex 输入逐字节嵌入并做版本交叉校验。该
   方案兼顾论文语义和最小披露，因此采用。

### 最终方案

新增 create-only 的 Stage 1 preparation contract：

1. 只从通过 authoritative accepted-corpus verifier 的 `opt_pool` 显式选择 query；
2. 对每条轨迹原样保存 `[user]` 或 `[user, assistant, user]`，不重写内容和顺序；
3. 要求选中集合覆盖且只覆盖论文五类顶层意图，并统一绑定 taxonomy/TaskSpec；
4. 以外部文件 SHA 加载真实 Codex v5 common input，验证 canonical bytes，并逐字节嵌入
   `S1CreatorPacket`；
5. 输出目录和文件 create-only、自哈希，拒绝 symlink、Windows junction/reparse point；
6. preparation 不调用模型、不编译 Bank，也不声称 Stage 1 已运行。

### 如何验证

- 单元测试使用显式 marker 确认 assistant clarification 的角色、顺序和内容完全保留。
- 负向测试覆盖非 `opt_pool`、五意图缺失、混合/漂移版本、额外反馈字段、common input
  digest 漂移、非 canonical bytes、symlink/junction 和重复输出。
- Stage 1、accepted-corpus batch 与 Codex v5 联合回归为 `69 passed, 1 skipped`，Ruff、
  format 和 `git diff --check` 通过。

### 剩余限制

这只关闭了 S1 输入准备合同，不代表 Creator 已调用或 Skill Bank 已生成。论文五类 seed
现已按精确 ID/SHA 接受，但正式语料仍为 0 accepted query，且没有 active plan 或 query
batch；LLMStatic v5 的 owner 人工 checklist 已在任何 S1 输出出现前完成并冻结 unchanged
acceptance。真实 S1 还需要 accepted opt-pool、明确 query selection、一次性 Codex author
authority、Engineer/Human Reflection 工件以及 C3 后的确定性 Bank 编译。

### 30 秒回答

“Stage 1 的隔离不是简单删除所有 assistant 消息。真实轨迹里的澄清问题必须保留，但
Judge、失败归因和工具结果必须排除。我做了严格白名单投影，只接受 opt-pool 的
`[user]` 或 `[user,assistant,user]`，覆盖论文五类，并与同一份 Codex v5 TaskSpec/预算输入
做逐字节和版本绑定。69 项联合测试通过；产物仍明确标成 prepared-not-invoked，所以不会把
准备完成冒充 Stage 1 已运行。”

### 2 分钟回答

“论文把 User Trajectories 作为 Stage 1 输入，但项目 schema 中既有正常的 assistant 澄清，
后续运行又会产生 assistant/tool/Judge 信息。如果按角色一刀切，只留下最后一条 user 文本，
Creator 看不到用户如何在澄清后完成意图；如果整条 Query 或执行 trace 都给它，又会泄露
S2/S3 信号。我的边界是按信息生命周期做白名单投影：保留 query/asset/image、完整合法
turns 和论文五类标签，label provenance、Judge、failure、tool trace、gate/test 字段从
schema 层就不存在。

另一个审查发现初版选错了 common packet：文件名相近，但它是 Qwen AuthoringInput，不是
真实的 Codex v5 静态作者输入。修正后，packet 用外部 SHA 加载
`authoring-packet-codex-high-v5.json`，校验模型、预算、TaskSpec，再逐字节嵌入；trajectory
的 taxonomy/TaskSpec 必须与它一致。输出 create-only、自哈希，并显式声明模型未调用、Bank
未生成。69 项联合回归覆盖完整 turns、污染字段、版本漂移和路径攻击。当前限制也写进合同：
accepted corpus 仍为零，静态基线人审和真实 Creator/Engineer/Human Reflection 尚未发生，
所以只能说 Stage 1 输入边界已经就绪，不能说论文第一阶段已经完成。”

### 证据入口

- [SkillChain 原论文（arXiv）](https://arxiv.org/abs/2606.12984)
- `src/skillchain/stage1.py`
- `scripts/prepare_stage1.py`
- `tests/test_stage1.py`
- `specs/authoring/authoring-packet-codex-high-v5.json`
- `src/skillchain/codex_authoring.py`
- `src/skillchain/codex_authoring_v5.py`

---

## 34. 全局 source-lock plan 与单分片 receipt 会把局部 exact-scope 修复放大成十源重审

**状态：已验证阻断、修复待实现**

### 一句话问题

ABO 的正式 adapter 必须授权它实际读取的字节，但现有 source lock 只覆盖三个 tar；补齐
exact scope 时，单分片 acquisition receipt 又无法表达 16 个 listing shard，而全局 plan
摘要还会改变十个数据源的锁，使本来只属于 ABO 的修复被放大成十源重审。

### 背景与影响

ABO 的图片人工审核已经得到 41 个通过冲突审计的 retained pair，但“样本审核通过”和
“正式 adapter 的全部输入字节已获 owner 授权”是两个独立条件。正式 ABO 路径会直接读取
16 个 `listings/metadata/listings_[0-9a-f].json.gz`、一个
`images/metadata/images.csv.gz`，以及这 41 个 retained pair 所需的 original 图片。
required source lock 必须覆盖这些 listing、image metadata 和 original 的实际消费路径与
哈希，不能只证明它们可能来自某个更大的归档。

当前 ABO lock 只列出 `abo-images-small.tar`、`abo-listings.tar` 和 `abo-spins.tar` 三个
tar。若保持现状，formal approval binder 会因消费字节不在获批 scope 内而 fail closed；
若直接修改现有全局 lock plan，九个未变化的数据源也会得到新锁 SHA，扩大 owner 的复核
范围。因此该问题阻断的是 formal C2 exact-source closure，不能被 41-pair 容量结论替代。

### 观察到的证据

- `specs/data_sources/c2/source-locks/abo.source-lock.json` 仅覆盖上述三个 tar，当前文件
  SHA-256 为
  `337a0b2dbf47701420fbf4d2b6b9ae4a5ca23b1990748c1bbb61c642234cc48e`。
- `src/skillchain/data/abo.py` 的正式绑定会逐项要求 required source lock 覆盖 listing raw
  snapshot、image metadata raw snapshot 和每个 consumed original，而不是仅检查上游 tar。
- retained pair 与原始 locator 的只读核对表明，当前正式 ABO 输入需要覆盖全部 16 个
  listing gzip shard、`images.csv.gz`，以及 41 个 retained pair 引用的 original 文件。
- `ABOAcquisitionReceipt` 只有单数 `listing_artifact`；item locator 还必须绑定该单一
  artifact URI，因此现有 schema 无法忠实表达跨 16 个官方 shard 的 retained 集合。
- `scripts/build_required_source_locks.py` 要求 plan 包含全部十个 required source，并把同一个
  `lock_plan_sha256` 写入每一份 source lock。只要为 ABO 修改 plan 字节，十份锁的内容和
  SHA 都会变化，现有 CLI 不能只替换 ABO。
- `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl` 的 ABO
  决定精确绑定旧 lock SHA；新 exact-scope lock 必然有新 SHA，旧 v2 owner ledger 不能
  自动继承或被解释成对新 scope 的批准。

### 根因

三个不同粒度的契约被耦合在了一起：

1. source lock 最初按可下载归档包建模，而正式 adapter 的授权边界按实际消费文件建模；
2. acquisition receipt 从单 shard fixture 推导成单数 `listing_artifact`，没有覆盖真实
   retained 集合跨多 shard 的形态；
3. 全局 lock plan 的摘要被嵌入每一个 per-source lock，使一个来源的 scope 变化传播到所有
   来源。

owner ledger 精确绑定 lock SHA 本身是正确的安全边界；真正的问题是上游锁的变更粒度过大，
导致一个局部事实修复产生不必要的全局授权 churn。

### 考虑过的方案与取舍

1. **继续用三个 tar 代表内部成员：** 不增加工件，但正式 binder 检查的是 adapter 直接
   消费路径，tar 的存在不能证明已锁定提取后的字节，淘汰。
2. **把 16 个 shard 拼接或伪装成一个 `listing_artifact`：** 可以绕过单数字段，却会丢失
   官方 artifact identity、locator 和逐文件哈希，破坏可追溯性，淘汰。
3. **沿用现有 CLI 重建十份锁并让 owner 全量重审：** 无需改代码，但九个未变来源也被换
   SHA，审核范围和出错面显著放大，只适合作为保守兜底。
4. **先修 receipt 粒度，再引入 ABO 局部 amendment：** 以多 listing artifact 表达真实
   输入，生成 create-only exact pack，并让 ABO scope 更新不改变九份无关锁。实现工作略多，
   但变化与授权范围一致，因此选为下一步设计。

### 最终方案

以下方案已经选定，但尚未实现：

1. 从已冻结的 pair-audit receipt 显式固定 41 个 retained pair；
2. create-only 生成正式 ABO input pack，包含并哈希 16 个 listing gzip shard、
   `images.csv.gz` 和这些 pair 所需的 exact original 集合，同时生成 canonical
   `listings.jsonl`、`image-manifest.jsonl` 与 acquisition receipt；
3. 将 `ABOAcquisitionReceipt` 从单一 `listing_artifact` 升级为多 listing artifacts，
   每个 item locator 绑定其真实 shard，禁止拼接或模糊来源；
4. 增加版本化的 ABO-only source-lock amendment/manifest，或等价地把 plan 绑定改为
   per-source，使 ABO scope 更新时九份无关 source lock 保持逐字节不变；
5. 基于新 ABO SHA create-only 生成新版 proposal、dossier 和 review bundle；旧 v2 ledger
   保持不可变。只有 owner 明确批准新 proposal SHA 与 exact scope 后，才能在新的版本化
   ledger/receipt 中落签并进入 formal adapter。

### 如何验证

目前完成的是阻断诊断，而不是修复验证：

- 已逐项比较现有 ABO lock 与正式 binder 的消费闭包，确认三 tar scope 不满足直接文件
  覆盖要求；
- 已核对 retained locator，确认 listing 输入跨 16 个 gzip shard，并检查 receipt schema
  只能表达单一 `listing_artifact`；
- 已检查 source-lock builder 的 plan 校验和序列化路径，确认全局 `lock_plan_sha256` 会进入
  十份锁；
- 已核对 v2 owner ledger 的精确 lock-SHA 绑定，确认它不能授权尚不存在的新 SHA。

待实现后的验收必须补充：多 shard locator 正反向测试；41-pair 消费文件与 exact lock 的
零遗漏/零多余闭包检查；只改变 ABO scope 时其余九份 lock 字节和 SHA 不变的回归；proposal
准备阶段不创建 ledger，以及无 owner 明确指令时签署路径 fail closed 的测试。

### 剩余限制

多分片 receipt、正式 normalized input pack、ABO-only amendment 和新版 proposal 都尚未
生成，因此不能声称 formal C2 已关闭。41 个 retained pair 所需 original 的最终文件清单、
计数、总字节和 manifest SHA 也必须由 create-only 构建过程冻结，不能从人工审核数量推断。
旧 v2 owner ledger 仍只对旧三-tar lock 有效；新 SHA 的批准必须由 owner 重新作出。这个
阻断不否定已完成的图片质量审核，也不应被表述成整个 Portfolio Track 无法继续，但任何
formal ABO adapter 运行都必须在 exact-scope approval 完成前保持禁止。

### 30 秒回答

“ABO 人审留下了 41 个可用 pair，但正式授权还没闭合。adapter 实际读取 16 个 listing
gzip、一个 `images.csv.gz` 和这些 pair 的 originals，现有锁却只有三个 tar；receipt 又只
支持一个 listing artifact。更麻烦的是，全局 plan SHA 被写进十份锁，局部修改 ABO 会让
十源一起换 SHA。这个阻断已经验证，但还没修复。下一步是多分片 receipt、create-only exact
pack 和 ABO-only lock amendment，再让 owner 针对新 proposal SHA 单独批准；旧 v2 批准
不能继承。”

### 2 分钟回答

“这个问题暴露的是三层粒度不一致。第一层是消费粒度：formal ABO binder 授权的是 adapter
直接读取的 listing shard、image metadata 和 original，不是上游 tar 的抽象存在。第二层是
receipt 粒度：真实 retained 集合的 listing locator 分布在 16 个官方 gzip shard，但当前
`ABOAcquisitionReceipt` 只有一个 `listing_artifact`，把 shard 拼起来会丢失官方身份和
逐文件哈希。第三层是变更粒度：现有 builder 把全局 `lock_plan_sha256` 写进全部十份
per-source lock，所以只改 ABO 的 scope，也会让九个无关来源换 SHA。

owner ledger 精确绑定 SHA 是正确的，不能为了省审核让旧 v2 批准漂移到新字节。我的下一步
设计是先从现有 audit receipt 冻结 41 个 pair，create-only 构建包含 16 个 listing gzip、
`images.csv.gz` 和 exact originals 的 normalized pack；receipt 改成多 artifact，并让每个
locator 绑定真实 shard。然后用 ABO-only amendment 或 per-source plan binding 隔离变更，
基于新 ABO SHA 生成新版 proposal。只有 owner 明确批准该 proposal SHA 和 scope，才在新
ledger 中落签。当前我只能陈述阻断和设计已确认，不能说修复已落地或 formal C2 已通过。”

### 证据入口

- `src/skillchain/data/abo.py`
- `scripts/build_required_source_locks.py`
- `scripts/prepare_source_review_bundle.py`
- `specs/data_sources/required-source-lock-plan-v1.json`
- `specs/data_sources/c2/source-locks/abo.source-lock.json`
- `specs/data_sources/c2/source-locks/required-source-lock-manifest.json`
- `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl`
- `data/clean/abo-original-review-v2/review-packet.jsonl`
- `specs/data_sources/c2/adapters/abo-original-review-v2-finalization.receipt.json`
- `specs/data_sources/c2/adapters/abo-original-pair-leakage-audit-v2.receipt.json`

---

## 35. 不可变 canonical bundle 不能在人工审核后原地追加文件

**状态：人工审核路径已验证并修复，Bank finalizer 待实现**

### 一句话问题

Codex v5 的 canonical run 用精确文件名集合承诺一次模型会话；如果按通用 finalizer 的习惯
把 `human-review.json` 追加到同一目录，反而会让已经通过的 session replay 立即失效。

### 背景与影响

v5 已生成 strict-validated pre-review draft，owner 需要在看到任何 S1 输出前完成限时人工
checklist。审核又必须成为可复核的持久工件，不能只留在聊天记录。与此同时，原 run 目录是
一次模型调用的不可变证据包，后续人类决策和 Bank 编译属于不同生命周期。若把二者混在同一
目录，关闭人工门会破坏 authoring 调用证据；若只保留聊天文本，则未来 compile 无法可靠
绑定“谁、审了哪份 draft、用了几分钟、是否修改”。

### 观察到的证据

- `scripts/run_codex_authoring_v5.py` 的 canonical loader 要求 run root 的实际 children names
  严格等于 receipt 中的 bundle files 加 `invocation-receipt.json`；任何额外文件都会返回
  false。
- v5 canonical invocation receipt file SHA 为
  `c39037076f0f5eaf6a5fd36ea97093945d10a176313f98305e056d7248b1e162`；
  pre-review file/bundle SHA 分别为
  `e0c26c1c054514bfc4629fc4e84557da63f6b05335ca6c3b8709efe0a0e08908` /
  `41fd1e636c060db028a298189b0db98847c06d5f0bd651c1074e8a0d3f4078b3`。
- 通用 `finalize_llm_static` 会向传入 root 创建 `human-review.json`、`static-bank.json`、
  `skills/` 和 `authoring-manifest.json`，并要求 Codex run 不具备的 Qwen-style
  `AuthoringInvocation` 文件形状，不能安全复用。
- owner `wenxi_0726` 明确提交 `review_minutes=5`、unchanged acceptance、非私有输入、
  safety、schema、citation、tool permission、edit scope 六项 true。用户给出的
  `change_reason` 是“未修改，接受原稿”。

以上是代码和工件验证过的事实。关于未来 finalizer 的具体输出目录和 manifest schema，
目前仍是待实现设计，不能表述成已经完成。

### 根因

1. **把不可变调用证据目录误当成可持续追加的工作目录。** 模型 session、人工审核和 Bank
   编译是三个时间不同、authority 不同的阶段，不应共享可变目录。
2. **通用 finalizer 隐含了旧 transport 的共址假设。** 它针对
   `AuthoringInput + AuthoringInvocation` 设计，而 Codex-mediated run 有自己的 terminal
   receipt、事件和输入绑定；形状相似不代表可以复用写路径。
3. **只看 self-hash 会遗漏外部身份绑定。** 一份人审 JSON 可以内部自洽，但若不绑定 exact
   invocation receipt、draft file、bundle 和 authoring input，攻击者可以把另一份自洽
   review 套到当前 run。
4. **词法路径和宽松 schema 都不是信任边界。** 绝对 `RUN_ROOT` 与相对或含 `..` 的
   `output_path` 做 `parents` 比较会漏判；Pydantic 默认宽松转换还会把字符串 `"30"` 转成
   整数后再验证旧 self-hash。两者都可能让原始字节表达与验证器看到的对象身份不一致。

### 考虑过的方案与取舍

1. **向原 run root 追加 `human-review.json`：** 路径直观，但会破坏 exact file-set replay，
   淘汰。
2. **修改旧 invocation receipt，把 review 加进 optional files：** 这等于在人审后重写一次
   已完成模型调用的终态证据，破坏不可变历史，淘汰。
3. **只在文档或聊天中记录接受：** 不污染 run，但没有可由 finalizer 机械加载的 canonical
   artifact，也无法做摘要和预算交叉验证，淘汰。
4. **外置 create-only wrapper：** 保持原 run 完全不变，同时把完整
   `HumanReviewArtifact` 和所有上游摘要绑定到一个小型、可跟踪回执；采用。

### 最终方案

1. 新增 `CodexAuthoringHumanReviewReceipt`，内部嵌入既有 `HumanReviewArtifact`，强制
   `decision=accepted_unchanged`、`changed=false`、pre/post SHA 相同、六项 checklist
   全为 true，并交叉验证 draft bundle、authoring input、5 分钟实际工时和 30 分钟上限。
2. wrapper 同时绑定 canonical invocation receipt 的路径、文件 SHA、payload SHA，以及
   pre-review 的路径、文件 SHA、bundle SHA 和 authoring input SHA，再计算 wrapper 自哈希。
3. `record_codex_authoring_review.py` 写入前先重放完整 canonical bundle，核对 run/candidate、
   receipt self-hash、draft、request 与 budget；目标采用 create-only 或 exact-byte
   idempotent 验证。
4. 审核回执外置到
   `specs/authoring/llm-static-codex-primary-20260724-high-v5-human-review.json`；
   不向原 run root 添加任何文件，也不编译 Bank。
5. 写入路径先解析为规范化绝对路径，再与真实 canonical run identity 做 containment；
   receipt model 默认 `strict=True`，避免调用者遗漏显式 strict 参数时发生类型强制转换。

### 如何验证

- 新 review schema/CLI 的定向测试为 `5 passed`，覆盖外置 create-only、exact-byte 幂等、
  不同字节占位拒绝、draft SHA 漂移、超预算、相对/`..` 路径别名和非严格类型转换拒绝；
  相关联合回归为 `12 passed, 71 deselected`，Ruff 通过。
- 实际 create-only 写入结果为 `changed=false`、`reviewer_id=wenxi_0726`、
  `review_minutes=5`。human-review/self/file SHA 分别为
  `5ca3ef751fbfd556d4fb24733ba2f249055297804c8c2e814d541fb2b03de0ec` /
  `da5a80e06af1b6251dc0eb2c79a524b2bf5f4e02daad949c6aaf84a1ed2f8274` /
  `938c38fe5ffd90dedabe09a53f49a6b54c2e5bb2b7d4cd7ab0f22d8f2000d738`。
- 回执从磁盘按 strict schema 独立复载后完全相等；写入完成后再次运行原 v5 canonical
  validator，结果仍为 true，证明人工门没有破坏模型调用证据。

### 剩余限制

真实 C3 authority runtime 尚未就绪，Codex 专用 finalizer 也尚未实现，因此当前只能声称
“pre-review draft 已由 owner 接受”，不能声称 LLMStatic Bank 已编译或 C1 已关闭。未来
finalizer 必须把外置 review receipt 的文件 SHA 纳入最终 manifest，并把 Bank、skills 和
manifest 发布到独立目录；不能回退到通用 finalizer 的原地写入方式。`reviewer_id` 提供项目内
问责，但不是外部密码学身份证明；更高信任等级仍需签名或外部审计系统。

### 30 秒回答

“Codex v5 的 run 目录不是普通工作目录，它的 receipt 承诺了精确文件集合。我最初检查人工
审核落盘路径时发现，往里面加一个 `human-review.json` 都会让 canonical replay 失败。
所以我没有改旧 receipt，而是做了外置 create-only review wrapper，绑定原 receipt、
pre-review file/bundle、authoring input 和完整六项 checklist。owner 用 5 分钟接受原稿后，
回执独立复载通过，原 session replay 仍为 true。人工门关闭了，但 C3 和 Bank compile 仍
明确未完成。”

### 2 分钟回答

“这里的核心不是文件放哪儿，而是生命周期和 authority 要分层。模型调用终态由 freeze、
approval、guard、claim、事件、raw final 和 trusted compiler 组成，v5 validator 还严格要求
run root 的文件名集合与 receipt 完全一致。人工 review 则发生在调用之后；Bank compile
又要等 C3 runtime。把后两阶段的文件追加到第一阶段目录，会用一个看似合理的审核动作破坏
不可变调用证据。

我考虑过重写 receipt 或只在聊天里留结论。前者改写历史，后者没有机械 trust root。最终做法
是外置一个 create-only wrapper：先完整重放旧 bundle，再加载 exact draft，构造已有的
`HumanReviewArtifact`；wrapper 绑定 invocation receipt file/payload SHA、draft file/bundle
SHA、authoring input、工时、六项 checklist 和自身摘要。真实 owner 决定写入后，外置文件的
SHA 固定，原 run 一个字节和一个文件名都没变，canonical replay 仍通过。下一步不是把它叫
finalized，而是在 C3 就绪后实现专用 finalizer，把这个外部回执作为不可变输入，在独立目录
生成 Bank 和 manifest。”

### 证据入口

- `src/skillchain/codex_authoring_review.py`
- `scripts/record_codex_authoring_review.py`
- `tests/test_codex_authoring_review.py`
- `scripts/run_codex_authoring_v5.py`
- `src/skillchain/static_authoring.py`
- `runs/formal-authoring/llm-static-codex-primary-20260724-high-v5/invocation-receipt.json`
- `runs/formal-authoring/llm-static-codex-primary-20260724-high-v5/pre-review-draft.json`
- `specs/authoring/llm-static-codex-primary-20260724-high-v5-human-review.json`

---

## 36. 数量够、排序固定、带 self-hash 仍不等于 200-query 计划可复现

**状态：已验证并修复；首批图像权限仍待确认**

### 一句话问题

从五个已准备的数据池生成 `dev_mini=200` 时，真实数据连续暴露了三个小 fixture
没有覆盖的确定性缺口：first-N 里有相同图片、manifest 在写回后 self-hash 漂移，以及
跨意图 triplet 候选会被普通 Exact 槽提前消耗；任何一个都足以让“数量看起来够”的计划失败
或产生不可复验状态。

### 背景与影响

Portfolio 第一阶段需要 184 个独立图片 component 支撑 200 条 query：35 Exact、35
Multi-Product、27 独立 Style、27 独立 Encyclopedia、60 Utility，并让 8 张 Exact 图片在
同批复用为论文五意图边界中的 Exact/Style/Encyclopedia triplet。计划还必须按 8 个
25 条批次顺序接受。这里的有效容量单位不是文件行数，而是最终图片字节、AssetCatalog
component、能力 assignment 和 active-plan 状态的共同交集。

### 观察到的证据

- 第一次真实组合在发布前失败，报错
  `Portfolio mini contains exact duplicate image bytes`；FashionIQ 的
  `dress:B0007WFGWS` 与 `dress:B0007WIZYE` 字节 SHA-256 同为
  `7145cd2f...7a8443e`。184 个候选实际只有 183 份唯一字节。
- 新组合器测试构造出大小为 10 的 AssetCatalog component 后，刚发布的 manifest 无法
  自己复载。根因证据是构建时 `component_size_histogram` 使用整数 key，`json.dumps`
  按数值顺序哈希；回载后的 JSON key 为字符串，`mode="json"` 再按字典序排序，
  `1,2,...,10` 与 `1,10,2,...` 产生不同摘要。
- 原 planner 从 Exact pool 顺序取项；若 triplet-compatible 图片较晚出现，前面的普通
  Exact 分配可能先消耗它们，或扫描不适配候选后永久丢弃本可用于普通 Exact 的图片。
- 原 batch 状态以“首个未接受 ID”为主，未证明 accepted ledger 是 active plan 的严格
  前缀，因此历史或人工构造的乱序状态可能先处理 `002` 再处理 `001`。
- 修复后真实 AssetCatalog 为 184 assets / 184 components，catalog SHA-256
  `df17da7d...15c28c96`；254 条 assignment 的 SHA-256 为
  `b579c616...8df2ca2f`。激活计划为 200 条、8 批、40 条 boundary，plan SHA-256
  `5e3b0d67...5e9fa6be`。

### 根因

1. **把记录数当成独立容量。** 稳定排序只能保证重复运行选择同一批记录，不能保证这些记录
   的最终图片字节不同。
2. **对内存对象而非持久化正规形哈希。** Python 的 key 类型与 JSON 的 key 类型不同；
   只要 canonical 排序依赖 key 类型，写前和读后就可能不是同一字节表示。
3. **计划约束没有先于资源消费。** triplet 是跨 capability 的同图约束，不能等普通槽分配
   后再碰运气寻找。
4. **状态机只检查成员关系，没有检查历史顺序。** “批次属于计划”不足以证明它是当前唯一
   可推进的下一批。

### 考虑过的方案与取舍

1. **手工删掉重复图或补一张：** 能让本次运行继续，但选择无法机械重放，淘汰。
2. **放宽全局重复/component 检查：** 保住表面配额，却会夸大独立样本量并破坏 split，
   淘汰。
3. **只把 histogram key 改成字符串：** 能修当前字段，但以后其他 Pydantic/JSON 类型转换
   仍可能复发；不作为最终方案。
4. **计划失败时加大素材池：** 可能掩盖 triplet 消费顺序错误，无法证明固定 184 component
   的理论容量确实可用；淘汰。
5. **内容去重补位、序列化正规化后哈希、预留 triplet、强制 ledger 前缀：** 改动分别位于
   正确的信任边界，且不修改已接受 seed 或 query 文本；采用。

### 最终方案

FashionIQ 先按冻结的 `(source_record_id, local_path)` 排序，再读取最终字节 SHA；遇到重复
就确定性跳过并继续补位到 27，最终 184 池仍做跨源全局重复拒绝。AssetCatalog 先用 dummy
hash 构造严格 model，再对 `model_dump(mode="json", exclude={"catalog_sha256"})` 的持久化
正规形计算 self-hash。planner 在普通 Exact 分配前预留每批一张 triplet 图片，不适配
triplet 的候选延期回普通池而不是丢弃。`stage/accept/status` 共用同一 active-plan 前缀
检查，只允许推进下一批，同时保留已接受批次的幂等重试。

### 如何验证

- Portfolio 组合器测试覆盖真实 FashionIQ 重复记录形态、确定性补位和跨源重复拒绝。
- AssetCatalog 核心测试覆盖两位数 component size 的发布—复载 self-hash round trip。
- synthesis 回归覆盖稀疏 triplet assignments 仍得到 184 个 component 的 200 条计划，
  以及 stage/accept 拒绝跳过 `dev-mini-001`。
- 相关组合器与 catalog 测试为 `21 passed, 1 skipped`；规划与批次测试为 `86 passed`；
  Ruff 与 `git diff --check` 通过。
- 真实 dry-run 和激活均返回 200 条、8 批、40 条 boundary；状态明确指向
  `dev-mini-001`，accepted query 仍为 0，证明生成和接受门没有被折叠。

### 剩余限制

这些验证关闭的是 Portfolio mini 的数据独立性、哈希和计划状态问题，不把非正式来源升级成
论文原始生产流量，也不把 `formal_eligible=false` 改成 true。ABO、RPC、FashionIQ 的
`cloud_upload_allowed=false` 以及其他来源的未声明值仍被原样保留；在 owner 明确授权当前
Codex 会话检查精确选中的首批图片前，不能调用 `view_image`、生成 query draft 或写入
staging。首批 25 条仍需独立人工审阅和明确接受。

### 30 秒回答

“我把五类素材组装成 200-query 计划时，真实数据揭示了三个 fixture 漏洞：固定 first-N
里有两条 FashionIQ 记录其实是同一图片；AssetCatalog 对整数 histogram key 的写前哈希与
JSON 回载后的字符串 key 排序不同；planner 又可能先消耗跨意图 triplet 图片。我没有手工
补数据，而是在选择层按最终字节去重补位、在 catalog 层对序列化正规形哈希、在计划层先
预留 8 个 triplet，并让批次 ledger 必须是严格前缀。真实结果是 184 个独立 component、
254 条能力绑定、200 条计划和 40 条边界样本。”

### 2 分钟回答

“这个收口让我重新确认，‘数量够、排序固定、文件有 SHA’还不是可复现数据计划。第一层是
容量：FashionIQ 前 27 条中两条记录的最终字节完全相同，所以 184 行只有 183 个独立图片。
我让选择器按稳定元数据排序，但以最终字节 SHA 去重并补位，最后仍做跨源重复拒绝。第二层
是表示：component size 到 10 后，构建端的整数 key 按数值排序，JSON 回载端的字符串 key
按字典序排序，self-hash 因而自相矛盾。我改为先经过严格 model 的 JSON 正规形，再计算
摘要。第三层是约束消费：每批需要同一图支持 Exact、Style 和 Encyclopedia，planner 必须
先预留 8 张 triplet 图片，并把不适配候选放回普通池。最后，批次推进也改成 accepted
ledger 必须严格等于 active plan 前缀，不能越过 001。

这些修复分别放在选择、序列化、规划和状态机边界，没有手工改 staging。组合器/catalog
测试 21 通过、1 跳过，synthesis 相关测试 86 通过；真实 catalog 得到 184 个独立
component，assignment 为 254 条，计划为 200 条、8 批、40 条 boundary。当前仍诚实标为
Portfolio 非正式轨道，且图片的云端检查权限没有被静默改写，所以我停在首批逐图生成之前
等待 owner 的精确授权。”

### 证据入口

- `src/skillchain/data/portfolio_mini.py`
- `src/skillchain/data/asset_catalog.py`
- `src/skillchain/synthesis/planning.py`
- `src/skillchain/synthesis/batches.py`
- `tests/data/test_portfolio_mini.py`
- `tests/data/test_asset_catalog.py`
- `tests/synthesis/test_catalog_planning.py`
- `tests/synthesis/test_batches.py`
- `data/clean/query_images/selection-manifest.json`
- `data/clean/portfolio-mini-asset-catalog-v1/manifest.json`
- `data/clean/portfolio-mini-capability-assignments-v1.jsonl`
- `data/queries/plans/dev_mini.manifest.json`

---

## 37. “写入拒绝理由”与“发布 rejected 目录”之间也存在可恢复的部分提交

**状态：已验证并修复**

### 一句话问题

Phase 3 的拒绝操作先在 staging 写 `reason.json`，再把整个目录原子重命名到
`rejected/`；Windows 目录锁若让第二步失败，旧实现会留下正确理由却无法用同一命令重试。

### 背景与影响

`dev-mini-001-r1` 经人工判断“表达过度具体”，owner 明确要求保留为
high-specificity reference 并生成后续修订。拒绝操作必须同时满足三项要求：原稿不可丢失、
拒绝理由不可被改写、目标目录不得覆盖并发创建的结果。若失败后只能手工删
`reason.json` 或搬目录，就会绕过正式语料状态机，也无法证明新修订确实从合法 rejected
状态继续。

### 观察到的证据

已验证事实：

- 官方 `reject-batch` 首次调用返回
  `[WinError 5] 拒绝访问: staging/dev-mini-001-r1 -> rejected/dev-mini-001-r1`。
- 失败后 `staging/dev-mini-001-r1/reason.json` 已存在，内容仍是 owner 的原始理由：
  `表达过度具体；保留为 high-specificity reference`；目标 rejected 目录尚不存在。
- staging/rejected 父目录间的独立空目录 rename probe 成功，source/destination ACL 也允许
  owner 完全控制；因此可以排除父目录整体不可写。具体占用该批次的进程未被证明。
- 旧实现无条件再次调用 `atomic_create_file(reason.json, ...)`，所以即使文件字节完全一致，
  重试也会先因 create-only 规则失败，永远到不了目录发布。
- 修复后同一个官方 CLI 成功，staging 路径消失、rejected 路径存在、reason 字节未改变，
  `status` 显示 `next_batch_id=dev-mini-001`、`next_revision=2`。

### 根因

目录 rename 本身保持了 create-only 原子发布，但整个业务操作并不是单步事务：
`reason.json` 的不可覆盖创建和目录 rename 是两个独立提交点。旧代码只考虑了“目标目录
竞争创建”，没有建模“第一个提交点成功、第二个提交点因外部文件锁失败”的恢复路径。
Windows 对被占用目录返回 access/sharing error，使这个缺口在真实本地预览环境中暴露。

### 考虑过的方案与取舍

1. **手工删除 reason 后重跑：** 简单，但会修改审计证据并绕过状态机，淘汰。
2. **手工把 staging 搬到 rejected：** 能完成表面状态变化，却跳过同理由验证和
   create-only 发布逻辑，淘汰。
3. **把 rename 改成可覆盖 move/copy：** 可能覆盖并发 winner，也削弱原子性，淘汰。
4. **让拒绝操作识别完全相同的已写 reason，并对 Windows 短暂共享锁做有界重试：**
   保留原有不可覆盖语义，同时为已知部分提交提供确定性恢复，采用。

### 最终方案

- `reject_generated_batch` 仍先 create-only 写 reason；若文件已存在，只在它是普通文件且
  字节与本次 canonical reason 完全相同时继续。理由不同、文件类型异常或字节漂移仍失败。
- Windows create-only directory rename 对 `EACCES`、`EBUSY`、WinError 5/32 做 6 次指数退避，
  总等待上限约 1.55 秒；每次失败都重新检查目标是否已被并发创建，一旦存在立即转成
  `FileExistsError`，绝不覆盖。
- 持久占用在有界重试后仍 fail closed，不自动终止进程、不删除文件，也不降级为 copy。

### 如何验证

- 新增回归覆盖：前次发布失败后，同理由可恢复；不同理由不可改写；Windows 前两次共享锁
  失败、第三次成功；目标竞争创建的既有测试继续通过。
- 定向测试：
  `uv run pytest tests/synthesis/test_store.py tests/synthesis/test_batches.py -q`
  返回 `61 passed, 1 skipped`。
- 真实 CLI 重试成功并输出
  `data/queries/rejected/dev-mini-001-r1`；随后检查确认 `staging_exists=false`、
  `rejected_exists=true`，reason 内容与 owner 指令一致，下一 revision 为 2。

### 剩余限制

本次没有取得具体锁持有者的可核验证据；`openfiles /query` 在当前权限下被系统拒绝，因此
只能确认这是 source-specific rename failure，不能归因给浏览器、HTTP server 或杀毒软件。
有界重试只覆盖短暂锁，持久占用仍需要外部关闭句柄后再次执行。该修复也不把两个文件系统
操作变成真正的跨崩溃事务；它提供的是严格同理由的安全恢复，而不是任意幂等覆盖。

### 30 秒回答

“一次正式语料拒绝在 Windows 上暴露了部分提交：理由文件已经 create-only 落盘，但目录
rename 因共享锁失败。旧实现重跑会被自己写下的 reason 挡住。我没有删证据或手工搬目录，
而是让状态机只接受字节完全相同的 reason 续跑，并给 Windows rename 加有界退避，同时保留
目标存在即失败。61 项测试通过后，原 CLI 成功把批次移到 rejected，理由未变，下一修订为
r2。”

### 2 分钟回答

“表面上拒绝批次只是一次目录移动，实际上有两个提交点：先写不可覆盖的 reason，再做
create-only rename。真实运行中第一步成功、第二步被 Windows 文件锁拒绝，于是产生了一个
合法但旧代码无法恢复的中间态。最危险的修法是删 reason 或直接 Move-Item，因为那会破坏
审计链；另一个错误方向是允许覆盖目标，会把并发安全一起丢掉。

我把恢复条件压到最小：已有 reason 必须是普通文件，而且 canonical 字节与本次 owner 理由
完全一致；否则继续失败。rename 只对 Windows 明确的 access/sharing 错误做 6 次指数退避，
每次仍检查目标是否出现，绝不 replace。测试同时覆盖同理由恢复、异理由拒绝、短暂锁和竞争
winner。真实命令随后成功，staging 消失、rejected 存在、reason 未改，状态机分配 r2。锁
持有者没有被可靠识别，所以我把它写成未决限制，而没有把猜测当根因。”

### 证据入口

- `src/skillchain/synthesis/store.py`
- `src/skillchain/synthesis/batches.py`
- `tests/synthesis/test_store.py`
- `tests/synthesis/test_batches.py`
- `data/queries/rejected/dev-mini-001-r1/reason.json`

---

## 38. 审阅辅助分层不能反向升级成偏离论文目标的硬契约

**状态：已验证并修复；图像条件化软分层已贯穿 200 条 dev_mini**

### 一句话问题

S0–S3 原本只是帮助人工覆盖“从模糊到清晰”的写作尺度；如果为了维持固定配额而把它升级成
正式数据权威，反而会让辅助指标主导生成，并偏离以论文五类意图为根本基准的复现目标。

### 背景与影响

项目正式复现的对象是论文五类顶层意图。S0–S3 是为了让合成 query 更像真实用户而引入的
审阅辅助，不是论文标签，也不参与第一阶段的正式分层评测。`dev-mini-001-r2` 最初按
S0/S1/S2/S3=`5/9/8/3` 编写；人工审核通过 24 条，并指出 `dm-017` 的表达比 S1 更明确。

最初的收口方案倾向于先把 specificity 做成权威字段，再生成 r3。这会扩大契约和 schema，
延迟第一阶段，而且错误地假设每个 25 条小批次都必须精确维持一个人为配额。项目 owner
随后明确：每批改动很少，不要求严格保持 `5/9/8/3`，应直接基于 r2 做局部修正。

后续生成又暴露了第二层问题：即使不强制配额，如果只用句长或“约束数量”机械区分 S0–S3，
模型仍会把从图片中看见的颜色、材质、部件和输出格式全部写进用户话术，重新产生“表达过度
具体”。因此本次把分层从简单的长短梯度细化为**图像条件化的用户主动表达量**：先完整理解
图片和计划意图，再决定真实用户在这一档会主动说出多少，而不是把模型掌握的事实都塞给用户。

### 观察到的证据

已验证事实：

- 正式 draft 每条只允许 `plan_id` 和 `turns`；active plan 与 staging 结果冻结论文意图、
  图片、boundary、split 和对话内容，但没有 specificity 字段。
- `dev-mini-001-r2` 已整体进入 rejected；拒绝证据保存了 `24/1/0`、`dm-017`、reviewer
  和耗时，因此没有把局部通过误写成 corpus accept。
- `dev-mini-001-r3` 已由官方 `stage-batch` 生成 25 条。逐条比较 r2 与 r3，排除唯一应变化
  的 `synthesis_batch_id` 后内容差异为 0。
- r3 HTML 将 `dm-017` 从 S1 调整为 S2，因此审阅辅助分布自然变为 `5/8/9/3`；没有为了
  “补齐配额”改写另一条已通过 query。
- r3 quality report 验证：25 条、五意图计数 `5/5/4/4/7`、5 条 boundary、23 条单轮、
  2 条两轮、0 重复；这些才是当前正式批次门。
- 用户随后以精确批次 ID 明确接受 `dev-mini-001-r3`；官方 `accept-batch` 使用 staging
  绑定的 asset catalog 重新校验后完成 promotion，ledger 记录的结果 SHA 与 quality
  report 一致。
- 本地浏览器实测 r3 页面显示 `24/25`、直接定位 `dm-017`、标记 S2，图片以
  `object-fit: contain` 和原始 `500×333` 比例显示，且没有控制台错误。
- 下一批 `dev-mini-002-r1` 继续执行这一边界：正式 plan 仍保持五意图
  `5/5/4/4/7`，HTML 的清晰度软覆盖自然形成 `4/12/8/1`，没有为复刻上一批比例而增加
  限制条件或补偿性改写。
- `dev-mini-003-r1` 在生成前逐项查看 25 个 planned image；同一张格纹手机壳图被构造成
  `dm-051/052/053` 三联边界，分别表达“找同款”“找相似风格”“询问格纹知识”，证明
  顶层意图由用户动作决定，而不是由图片对象决定。
- 本批把四档操作化为：S0 只保留最小任务线索；S1 是可路由的日常短句；S2 自然加入
  1–2 个关注点或通过一次澄清消解歧义；S3 才保留多个可核验细节或输出要求。分层判断看
  整段 trajectory，不简单按最终一句长度计数。
- 只有 `dm-054` 和 `dm-056` 使用 `[user, assistant, user]` 澄清轨迹；其余 23 条保持
  单轮，避免为了展示“多轮能力”人为把清晰请求改成长对话。
- 本批审阅页的软覆盖自然形成 `4/10/8/3`。这不是目标配额，而是逐图生成后的描述性统计；
  正式 quality report 仍只验证论文五意图 `5/5/4/4/7`、5 条 boundary、23/2 轮数和
  0 重复，`results_sha256=6aaebd0cc040cb5da30209063dd2af2bb9ab28b5603ea3fd3335638cd95756c7`。
- 外部公开电商 query 数据只用于校准“用户通常会用短句、关键词式表达”的语言表面，不被
  当成五意图或 S0–S3 的真实流量配额；当前配比仍是本项目 pilot 的人工可审阅覆盖。
- 2026-07-28，八批 `dev_mini` 全部收口为 accepted：200 条 query 的正式五意图分布为
  `35/35/35/35/60`，六 capability 为 `35/35/35/35/30/30`，40 条 boundary，
  188 条单轮、12 条澄清轨迹、0 重复。HTML 软分层聚合为 `38/79/60/23`，仍未进入
  query schema 或正式评测轴。

### 根因

根因不是“schema 少了一个必须字段”，而是没有先区分两类约束：

- **复现权威：** 来自论文、必须进正式契约并由流水线验证，例如五类意图。
- **生成与审阅启发式：** 用于改善自然度、允许人工局部调整，例如 S0–S3。

把启发式误当成硬配额会产生新的契约漂移：为了满足整齐的统计数字而改写本来已通过的样本，
让合成数据服务于自定义标签，而不是服务于论文任务和真实用户表达。

另一个根因是把三件事混成了一个“specificity”：

- 句子有多长；
- 用户主动说出了多少任务条件；
- 结合图片后是否仍需澄清。

短句可以因为型号而非常精确，长句也可能仍然含糊；图片还能消解“这个”的指代。因此分层
必须基于完整图文上下文和 trajectory，而不能只数词、数属性或数标点。

### 考虑过的方案与取舍

1. **把 S0–S3 塞进 `canonical_intent`：** 最省字段，但会破坏论文五意图基准，淘汰。
2. **立即增加 authority-owned specificity schema 和硬分布门：** 可机验，但会把非论文
   轴升为第一阶段正式约束，并要求 planner、schema、哈希和迁移测试同步升级；当前不采用。
3. **每次改一条就补偿性改写另一条以维持配额：** 数字整齐，但增加无价值改动和复审负担，
   淘汰。
4. **将 S 保持为可见但非强制的审阅辅助：** 不污染论文契约，可支持自然度检查，人工改一条
   就只改一条；采用。
5. **按字数或槽位数自动分层：** 便于统计，但会把短型号误判为模糊，也会奖励模型堆砌
   图片属性；淘汰。
6. **图像条件化软分层：** 先掌握完整视觉事实，再按档位有意隐藏用户没有主动表达的槽位；
   只有确实存在多个合理任务解释时才写澄清对话。更依赖人工判断，但最接近本项目要模拟的
   真实使用方式；采用。

### 最终方案

保持论文五类意图为正式根基，并把 S0–S3 限定为生成与审阅辅助：

- **S0：** 只说最小动作或对象线索，允许真实的省略和指代，但结合图片后仍应有合理任务方向。
- **S1：** 意图和对象已可路由，使用用户随手输入的日常短句，不主动追加偏好。
- **S2：** 自然表达 1–2 个关注点，或用一次澄清把原本模糊的动作收敛到计划意图。
- **S3：** 少量保留，包含多个图中可核验细节、排除条件或明确输出要求，但仍避免说明书口吻。

生成时先逐图理解完整事实，再从完整信息中“向下采样”用户会主动表达的部分；不根据文件名
猜图，也不为了升级档位虚构约束。同图跨意图样本通过改变用户动作形成 boundary；多轮只在
自然歧义需要时使用。每批只检查四档是否都有合理覆盖，不要求复刻上一批数字。

历史处理仍保持不变：`dev-mini-001-r2` 作为 rejected 证据保留，r3 只把 `dm-017` 从 S1
调整到 S2，没有补偿性改写。最终 accepted 顺序为 `001-r3/002-r1/003-r1/004-r1/005-r2/
006-r2/007-r1/008-r1`；`001-r1/001-r2/005-r1/006-r1` 继续保留为 rejected。每个后继批次
都经过独立人工审阅；计划耗尽后没有为了维持批次节奏虚构第九批。

### 如何验证

- `data/queries/rejected/dev-mini-001-r2/reason.json`：保存 r2 的人工拒绝证据。
- `data/queries/accepted/dev-mini-001-r3/quality_report.json`：正式批次门全部通过，
  `results_sha256=c1ce5b9473f254f9f660c2f41dd709c358baccdc2e542dcced5989e31e0a9c1b`。
- `data/queries/accepted-ledger.jsonl`：记录 `accepted_at=2026-07-27T11:57:16.737538Z`、
  `count=25` 及同一 results SHA；`status` 显示累计 1 批、25 条，下一批为
  `dev-mini-002-r1`。
- `data/reviews/dev-mini-001-r3/index.html`：S0–S3 仅作为审阅辅助，24 条决定被显式沿用，
  `dm-017` 为 S2；接受后页面数据源已切到不可变 accepted 归档。
- 本地 HTTP 浏览器检查：页面载入成功、`24/25`、dm-017、原图尺寸和 contain 布局均符合
  预期，控制台无 error/warning。
- `data/queries/accepted/dev-mini-002-r1/quality_report.json` 验证第二批 25 条、5 条
  boundary、23/2 轮数分布及 0 重复；HTML 实测初始为 `0/25`，四个软层级筛选分别返回
  `4/12/8/1`，源图仍使用 `object-fit: contain`。
- `data/queries/accepted/dev-mini-003-r1/quality_report.json` 验证第三批 25 条、五意图
  `5/5/4/4/7`、5 条 boundary、23/2 轮数及 0 重复；没有把 S 档写入正式 query schema。
- `data/reviews/dev-mini-003-r1/index.html` 将本次软层级显示为 `4/10/8/3`；本地浏览器逐档
  筛选数量一致，初始为 `0/25`，图片 `object-fit: contain`，控制台无 warning/error。
- 对 `dm-051/052/053` 的人工可见对照确认：相同图片能由“同款、相似推荐、百科询问”形成
  三个不同论文意图，说明分层没有覆盖或改写顶层标签。
- `status` 最终为 8 批/200 条、空 staging、`next_batch_id=null`；聚合 query
  SHA=`6f8eda4fe663733708d6e797c954e58f098d3938857c6f2b143e24e202437f03`。这些记录的
  split 全部是 `dev_mini`，不能据此声称 Stage 1 `opt_pool` 已形成。

### 剩余限制

S0–S3 不是正式 schema 字段，因此不能用于需要机器保证的分层抽样或论文指标报告。若后续
实验设计明确要求按清晰度做独立统计，应另开契约升级，并先说明它是项目新增分析轴而非论文
五意图。当前 dev_mini 扩展只把它用作批次内的软覆盖提示。

这些 query 仍是 `synthetic_derived`，`4/10/8/3` 不能声称代表企业生产流量。分层边界依赖
人工判断，尤其 S0 与“信息不足到不可路由”之间仍需在后续批次抽查；公开真实 query 数据
只能校准语言风格，不能替本项目的图像五意图给出生产分布。

### 30 秒回答

“我们最初用 S0–S3 解决合成 query 过度具体，但我发现只按句长和固定配额仍会诱导模型把
图片里所有属性塞进用户话术。我把它改成图像条件化的软分层：先看懂图片和论文意图，再按档位
控制用户主动表达多少；S2 才加入 1–2 个自然约束，S3 只占少量，多轮只用于真实歧义。
第三批自然得到 `4/10/8/3`，但正式流水线仍只验证论文五意图、boundary、轮数和重复。
这样既提高了真实感，也没有把自创分层变成偏离论文的硬标签。”

### 2 分钟回答

“这个问题先是权威边界，后来又变成生成方法问题。论文五类意图决定正式标签；S0–S3 只是
我们为了模拟真实用户新增的表达尺度。第一批人工审核把 `dm-017` 从 S1 改判为 S2。如果
立即扩 schema、强制维持 `5/9/8/3`，就必须再改一条已通过 query 来凑数，让自创指标反过来
主导论文复现。所以我保留 r2 的拒绝证据，r3 只调整审阅映射，不做补偿性改写。

继续生成时我又发现，“不强制配额”还不够。模型知道图片的全部属性，很容易把颜色、材质、
部件和格式要求全写成用户限制。于是我把分层定义成图像条件化的用户主动表达量：S0 只留
最小线索，S1 是可路由短句，S2 加 1–2 个自然关注点或一次澄清，S3 才保留多个可核验要求。
先逐图理解完整事实，再向下采样用户会说出的槽位；同图边界样本通过改变动作而不是改变对象
来区分意图，多轮只用于确有歧义的样本。

`dev-mini-003-r1` 因此形成 `4/10/8/3` 的描述性覆盖、23 个单轮和 2 个澄清对话；同一手机
壳图还得到同款、相似推荐和百科三个边界 query。正式 quality report 仍只验证五类意图
`5/5/4/4/7`、5 个 boundary 和 0 重复。取舍是 S 档不能用于正式论文统计，但换来了更真实
的用户表达，也避免为好看的比例继续扩建契约。”

### 证据入口

- `data/queries/rejected/dev-mini-001-r2/reason.json`
- `data/queries/rejected/dev-mini-001-r2/results.jsonl`
- `data/queries/accepted/dev-mini-001-r3/manifest.json`
- `data/queries/accepted/dev-mini-001-r3/quality_report.json`
- `data/queries/accepted/dev-mini-001-r3/results.jsonl`
- `data/queries/accepted-ledger.jsonl`
- `data/reviews/dev-mini-001-r3/index.html`
- `data/queries/accepted/dev-mini-002-r1/quality_report.json`
- `data/queries/accepted/dev-mini-002-r1/results.jsonl`
- `data/reviews/dev-mini-002-r1/index.html`
- `data/queries/inbox/batches/dev-mini-003.jsonl`
- `data/queries/accepted/dev-mini-003-r1/manifest.json`
- `data/queries/accepted/dev-mini-003-r1/quality_report.json`
- `data/queries/accepted/dev-mini-003-r1/results.jsonl`
- `data/reviews/dev-mini-003-r1/index.html`
- `data/queries/plans/dev_mini.json`

---

## 39. 把“全通过后的三步操作”合并，不能把两个人工接受边界也合并

**状态：已验证并落地；八批完成后按计划耗尽正常终止**

### 一句话问题

全批通过时可以把“处理摘要、正式接受当前批、生成下一批”压缩成一次 owner 指令，但不能
因此自动接受下一批，也不能把顺序执行伪装成跨两个批次的原子事务。

### 背景与影响

Phase 3 corpus 采用 25 条一批的人工审阅。原流程要求 owner 先粘贴审阅摘要，再单独发送
正式接受，最后再授权生成下一批；三个往返都围绕同一份 `25/0/0` 结果，操作成本高，也容易
在批次 ID 之间发生上下文错配。另一方面，接受会写入不可变目录和 append-only ledger，
而下一批生成还要经过模型门禁、逐图核验、hash 绑定和 staging；如果为了“省一步”把后继
批次也自动接受，就会直接绕过独立人工审核。

### 观察到的证据

已验证事实：

- `dev-mini-002-r1` 的人工摘要明确给出 `25/0/0`、`reviewer_id=wenxi_0726`、
  正数 `review_minutes`，随后 owner 在同一指令中精确写出“接受 dev-mini-002-r1”并要求
  合并生成下一批。
- 官方 `accept-batch` 成功后，`data/queries/accepted-ledger.jsonl` 记录该批 25 条和
  `results_sha256=e06f18d75f0888aa74e82ea7b08209a08d4f59336e65afe6fe8f3648d0a0d9e4`；
  accepted 总量变为 50，下一计划 ID 为 `dev-mini-003`。
- 同一轮中重新通过 `5.6 Sol Ultra` 模型门禁，读取 active plan，逐项查看 25 个计划图片，
  生成并由 `stage-batch` 发布 `dev-mini-003-r1`。
- 后继批次 quality report 为 25 条、5 个 boundary、五意图 `5/5/4/4/7`、23/2 轮数、
  0 重复，`results_sha256=6aaebd0cc040cb5da30209063dd2af2bb9ab28b5603ea3fd3335638cd95756c7`。
- 浏览器实测后继审阅页初始为 `0/25`，S0–S3 软筛选为 `4/10/8/3`，图片
  `object-fit: contain`，全通过摘要包含滚动授权，非全通过分支仍声明“不等于 corpus
  accept”，控制台无 warning/error。
- 该流程继续用于后续批次；`dev-mini-008-r1` 接受后 ledger 达到 8 批/200 条，
  `status` 返回 `next_batch_id=null` 和 `next_revision=null`。系统在模型门、图片查看和
  inbox 写入前正常停止，没有把 owner 的“生成下一批”授权解释成制造计划外第九批。

### 根因

原流程把两个不同概念混在了一起：

- **交互步骤数：** 同一份完整人工证据可以在一个 owner 指令中授权多个顺序动作。
- **状态边界数：** 当前批接受和后继批接受是两次独立的不可变状态转换，必须分别有人审。

如果机械坚持“一条消息只能做一个动作”，会产生不必要的往返；如果反过来把“合并一步”
理解为“连续自动接受”，又会破坏人工审核边界。真正需要约束的是授权证据和状态转换，而
不是聊天消息的数量。

### 考虑过的方案与取舍

1. **保持三次消息：** 最保守，但重复确认同一证据，owner 成本高，淘汰。
2. **全自动接受后续所有批次：** 速度最快，但未审模型输出会直接进入 corpus，淘汰。
3. **把接受与生成实现成一个可回滚事务：** 表面原子，但 accepted ledger 与目录本来就是
   不可变事实；生成失败后回滚已审批次反而篡改审计历史，淘汰。
4. **受控 roll-forward：** 当前指令满足严格条件时，先接受精确当前批，再生成唯一下一批
   到 staging；任何后继批都停在人工门前。采用。

### 最终方案

在 Phase 3 skill 和 corpus contract 中增加 `roll-forward`：

- 只在当前 owner 指令同时绑定精确 staged ID、`25/0/0`、非空 reviewer、正数耗时、
  明确接受当前批和明确生成下一批授权时启用。
- 先用 staging 绑定的 asset catalog 正式接受并复验 ledger，再从 active plan 获取唯一
  next batch，完整执行模型门禁、逐图查看、生成、manifest hash 和 staging。
- 后继批次绝不自动接受；HTML 在全通过时复制含滚动授权的摘要，非全通过仍只输出处理请求。
- 两阶段不是原子事务：若接受成功而生成失败，保留已接受事实，明确报告部分成功并从下一批
  重试，不回滚 ledger。
- 接受后若权威 ledger 表明 plan 已耗尽，roll-forward 以“当前批已接受、无合法后继批”
  成功终止；这不是生成失败，也不触发模型门。

### 如何验证

- `dev-mini-002-r1` 已从 staging 消失并存在于 accepted；ledger 与 quality report 的
  count、plan SHA、seed SHA 和 results SHA 一致。
- 首次 roll-forward 完成时，`dev-mini-003-r1` 只存在于 staging，accepted 总量仍为 50，
  因此当次没有越过第二个人工门；它在后续独立人审后才进入 accepted。
- 后继 draft manifest 绑定 accepted seed、active plan、asset catalog、leakage policy、
  draft bytes 和完整 25-item generation input。
- 新审阅页载入 25 条与原图，初始没有任何决定；本地浏览器验证筛选、完整图比例和摘要分支。
- 最终 ledger 的 8 个 results SHA 均与 accepted 字节一致；按 ledger 顺序拼接与
  315,195-byte `queries.jsonl` 逐字节相同。终态为 staging 空、next ID/revision 均为 null。

### 剩余限制

roll-forward 只减少 owner 往返，不保证接受与生成的事务原子性。浏览器摘要中的
`review_minutes` 仍来自页面本地计时，owner 需要对其真实性负责；混合决定、缺少 reviewer、
耗时无效或只粘贴“本摘要不等于接受”的旧摘要时，仍必须退回普通处理流程。后继批次必须由
人独立审阅并再次明确接受。query 审阅决定和 elapsed timer 当前没有导出为 tracked review
ledger，因此不能把会话摘要描述成仓库内可重建的人时证据。

### 30 秒回答

“我把全批通过后的三次往返压成了一个受控 roll-forward：同一条 owner 指令必须绑定精确
批次、25/0/0、reviewer、正数耗时，并同时明确接受当前批和生成下一批。系统先不可变地接受
当前批并复验 ledger，再完整生成唯一下一批，但只放进 staging。它不是可回滚事务；如果
生成失败，已接受事实保留并报告部分成功。这样省掉重复确认，又没有绕过下一批的人工门。”

### 2 分钟回答

“原来每个 25 条批次审完后，要分别发送摘要、接受和生成下一批，三次消息实际复用了同一份
人工证据。直接全自动化又不成立，因为下一批是新的模型输出，必须独立审核。我把问题拆成
交互边界和状态边界：消息可以合并，两个 corpus accept 不能合并。

实现上新增受控 roll-forward，只有当前指令含精确 staged ID、25/0/0、reviewer、正数耗时
以及两项显式授权才触发。执行顺序固定为：accept 当前批、复验 append-only ledger、通过
模型门禁、读取唯一 next plan、逐图生成并 stage 后继批。接受和生成不是跨目录事务；后一段
失败时绝不回滚已经人工批准的历史，而是报告部分成功。第一次真实运行把
dev-mini-002-r1 正式接受到累计 50 条，并把 dev-mini-003-r1 以 25 条、0 重复的状态留在
staging。相同流程最终推进到 8 批/200 条；最后一次接受后 ledger 返回 next=null，系统
在生成前正常停止，没有虚构第九批。HTML 也只在 25 条全通过时生成滚动授权，其他情况继续
明确不等于接受。这个设计既降低了人工操作成本，也保持了每批独立的人审与不可变审计。”

### 证据入口

- `.agents/skills/generate-phase3-corpus/SKILL.md`
- `.agents/skills/generate-phase3-corpus/references/corpus-contract.md`
- `data/queries/accepted-ledger.jsonl`
- `data/queries/accepted/dev-mini-002-r1/quality_report.json`
- `data/queries/accepted/dev-mini-003-r1/manifest.json`
- `data/queries/accepted/dev-mini-003-r1/quality_report.json`
- `data/queries/accepted/dev-mini-008-r1/quality_report.json`
- `data/reviews/dev-mini-003-r1/index.html`

---

## 40. 视觉 Feedback 模型迁移：从工程解阻到异源评价恢复

**状态：v5 迁移与视觉链路已验证；production 五配置评价待运行**

### 一句话问题

先让 Feedback/final 共用 `kimi-k2.6` 解决了本地图片传输，随后把高频视觉 Feedback 迁到
AIFast `gemini-3.6-flash`，在不动 final 评分路径的前提下恢复异源模型隔离；工程可运行、
评价独立性和第三方网关身份可信度必须分别说明。

### 背景与影响

历史选择是 Kimi K3 final Judge + DeepSeek Feedback。实际接入时发现 K3 路径要求公网 URL，
而项目的 184 张运行图片是本地、逐字节锁定的资产；为了托管 URL 再建设上传、生命周期和下载
复验协议，会延迟第一版 200-query 纵切。项目所有者因此前向决定：Feedback 与 final 都改用百炼
业务空间 `kimi-k2.6`，并要求 Feedback 使用视觉输入。

这项决定同时改变两件事：传输层从“不可用的公网 URL 方案”变成 Base64 Data URL；实验设计从
“不同模型家族”变成“同一模型、不同角色”。前者是工程解阻，后者是必须披露的相关评价风险，
不能用不同 prompt 掩盖。

### 观察到的证据

已验证事实：

- 百炼部署模型的精确 endpoint 名是 `kimi-k2.6`，OpenAI-compatible 消息可用
  `image_url.url=data:<mime>;base64,...` 输入本地图片；K2.6 使用 `enable_thinking`，不使用
  K3 的 `reasoning_effort=max`，也不支持 JSON mode。
- FeedbackPacket/FinalEvaluationPacket 已升级为 v2。持久化 packet/prompt/result 只保存图片
  MIME+SHA；runner 在权限 preflight 后从 verified catalog 瞬时读取字节并生成多模态
  `image_url`。若 provider 回显 input image Data URL 或完整编码，内容会被删节，只保留
  原响应 SHA/字节数并以 `input_image_echo` fail closed，因此 Base64 不落入 evaluator
  artifact。
- Feedback v2 同时携带原图、完整 turns、response/cards、可见 tool evidence、内部 tool trace
  和 rubric，满足“结合视觉与执行轨迹做失败归因”的输入要求。
- visual Feedback runner 固定 non-thinking `0.6/0.95`，严格解析
  `rule_violations/ideal_response_gaps/image-grounded evidence/skill_suggestions`；final runner
  固定 thinking `1.0/0.95`，只接收原始整数维度分数，再由本地计算 tier 与 `J_project`。
  两者都固定一次仓库调用，格式/provider 错误不修复、不重试并 fail closed；未删节
  response 在回执加载时会重新解析，final scores 也会本地重算，防止只协调修改 parsed
  字段后重签自哈希。
- 远程处理 v2 authorization/receipt 在不修改 catalog 字节的前提下新增
  `dashscope-kimi-feedback`。Qwen Assistant、Kimi Judge、Kimi Feedback 三个 processor
  preflight 均重新核对 184 个资产与 200 条 accepted query；visual Feedback runner 会把
  catalog、authorization 和 receipt 摘要写入结果，并拒绝授权 catalog 外图片。
- K2.6/packet/isolation/Feedback/final 的聚焦离线回归已通过。4 次有界 live smoke 先覆盖
  non-thinking/thinking transport，再覆盖严格 Feedback JSON 与 final score JSON；全部精确
  返回 `kimi-k2.6`、`finish_reason=stop`、零仓库重试。首轮两次合计 153 token；结构化两次
  分别为 103 与 971 token，累计 1,227 token。供应商未返回可得费用，不能自行估算成已知
  实际费用。

推断：

- Feedback 能看图后，涉及主体识别、局部属性、图文不一致和工具证据不足的失败归因应比纯文本
  Feedback 更有效；该效果尚未用真实运行数据证明。
- Feedback/final 共享模型可能放大相同视觉偏差和语言偏好，thinking mode 的不同不能消除这种
  相关性。

### 根因

根因不是单纯“模型名填错”，而是把三个契约混成了一个：

1. provider/model 身份；
2. 本地图片如何进入真实多模态 wire；
3. Feedback 与 final 的统计独立性。

历史代码只改模型常量会继续发送不兼容图片，或者把 Base64 放进文本；只让两个角色使用不同
prompt 又会虚构模型独立性；只改 `cloud_upload_allowed` 也不足以证明实际 Feedback processor
在 owner scope 内。

### 考虑过的方案与取舍

1. **继续 K3 + 公网 URL：** 保持历史选择，但需新增托管、过期、删除和下载复验基础设施，
   不利于第一版收口。
2. **Feedback 保留 DeepSeek：** 模型家族独立，但纯文本模型无法直接看原图，视觉失败归因不足。
3. **两个角色统一 K2.6 并声称不同 prompt 即独立：** 工程最省事，但实验声明不诚实，淘汰。
4. **统一 K2.6，前向登记偏离并强化角色/工件隔离：** 解决图片输入，保留可审计边界，同时明确
   相关偏差和 formal 降级；采用。

### 最终方案

- 新增 `model-role-selection-v4.json`，不改写 v2/v3 和历史 authoring freeze。
- Kimi provider 迁移到百炼/DashScope compatible endpoint，密钥仍走 `DASHSCOPE_API_KEY`；
  通用 endpoint 已通过 live smoke，`KIMI_DASHSCOPE_BASE_URL` 只在独立业务空间时显式覆盖。
- Feedback non-thinking、final thinking；二者严格分离 packet、prompt、cache、输入投影和产物，
  但 isolation lock 必须写明 shared-model risk。
- visual Feedback/final 在调用前分别强制消费 owner-authorized remote-processing preflight，
  图片 SHA 必须出现在同一 verified catalog；图片字节只存在于瞬时 wire。两类结果均绑定
  权限链、packet/prompt/image/wire、provider request、usage 和 latency，并可 create-only
  落盘重验。
- K2.6 不支持 JSON mode，因此模型只被要求返回一个无 fence JSON 对象；Runner 拒绝 duplicate
  key、额外字段、字符串/浮点分数、非法维度/范围、非 stop 和多 choice。final 的 tier 与
  `J_project` 始终由本地可信代码计算，不信任模型派生值。
- Portfolio Track 可以继续工程纵切；Formal Research Track 仍 `formal_eligible=false`，且后续
  必须增加盲 SBS、人评校准和共享模型限制披露。

### 如何验证

- K2.6 adapter 单测覆盖 exact model、Base64 Data URL、`enable_thinking`、温度/top-p、禁止
  reasoning effort/seed/JSON mode 和 SDK 零重试。
- packet/isolation 测试验证 Base64 不出现在 packet/prompt/result、wire 解码后与 catalog bytes
  相同、final 不含 treatment 秘密、feedback/final cache namespace 不同。
- Feedback/final runtime 测试验证真实 `image_url` 数组、两种 thinking 参数、权限 preflight、
  严格 JSON 成功/失败、一次调用、错误保守处理、派生分数和 create-only receipt 重验。
- 真实 v2 authorization/receipt 对三个 processor 完成 fresh runtime preflight，catalog SHA
  保持 `d7f443...284a8`，旧 v1 authorization/receipt 未改写。

### 剩余限制

当前已有 4 次 K2.6 live smoke、严格 Feedback/final parser、fail-closed 单次 runner 和可落盘的
单结果 receipt，但没有 production 五配置 1,000 行矩阵 receipt/manifest。Portfolio dev-mini
verified loader 已复验 accepted corpus、plan/catalog/权限链与图片字节，并生成
`dm-019 × 5 configs` 的 create-only 零调用清单；这只证明输入和覆盖面已准备好。Feedback 尚未
完成 verified Assistant-row batch join、跨样本聚合和到 Refiner 的批量合同；四个真实
treatment Bank、Assistant runtime lock、可见 card/evidence 投影、approved audit sealer 与
50–100 条 Judge—human calibration/audit 也未完成。因此只能说“视觉 evaluator 的单结果执行
与解析闭环、以及五配置 smoke 的输入清单已实现”，不能说“五配置实验已启动”、“评价模型独立”、
“Judge 已校准”或“已复现论文原模型组合”。相关提交在本轮工作提交后补记。

Portfolio `dev_mini → S1` 的输入交接也已独立实现：没有把正式 `opt_pool` loader 改成接受
`dev_mini`，而是新增只用于 Portfolio 的严格类型，从 200 条中选择五类意图各一条、排除 smoke
query 的 leakage group，并把 query/image SHA、权限链、零调用 checklist 与 LLMStatic/S1 共用的
Codex v5 authoring input 写入 create-only 三文件 manifest。verified loader 末尾再次深验所有
source，修复了“只重读 authoring/output、未重验 corpus/checklist”的 TOCTOU 缺口。真实重建
返回 `prepared_not_invoked`、零调用、无 Creator、无 Bank、未授权执行；因此它关闭的是输入
交接 P1，不是 S1 闭环。启用 Creator 前还必须通过 cross-fit 或排除对应 leakage groups，避免
用这 5 条改 Bank 后又在同一 200 条上报告主指标。

### 30 秒回答（v4 历史方案）

“原来的 Kimi K3 路径只接受公网 URL，本地锁定图片进不了 Judge。我把 Feedback 和 final 前向
迁到百炼 `kimi-k2.6`，图片先按 catalog 和 SHA 复验，再用真实 `image_url` Base64 Data URL
发送；Feedback 还会读取完整对话、可见回答和工具轨迹，并强制消费远程处理权限 preflight。
K2.6 没有 JSON mode，所以我用严格单对象契约、一次调用和本地派生评分，解析失败保守留在
分母。代价是两个 evaluator 共享模型，所以我没有把不同 prompt 冒充独立性，而是显式记录
相关风险、隔离 packet/cache/artifact，并保留盲人评和校准门。”

### 2 分钟回答（v4 历史方案）

“这个问题有两个层面。工程上，K3 的视觉接口要求公网 URL，而我的 184 张图片来自本地、
逐字节锁定的 Portfolio catalog。为此建设上传托管和生命周期协议会拖慢第一版。实验上，如果
简单把 Feedback 和 final 都改成同一个模型，又会违反原先不同模型家族的独立性契约。

我选择前向迁移到百炼 `kimi-k2.6`，不改写旧 selection 和冻结回执。传输层只在权限 preflight
之后把 verified 图片做瞬时 Base64 Data URL；packet、prompt 和结果只保存 MIME+SHA。Feedback
packet 扩展到原图、完整 turns、response/cards、可见 evidence 和内部 tool trace，non-thinking
运行并严格解析视觉归因字段；final 用同一模型的 thinking 模式，只提交原始整数分数，tier 和
`J_project` 由本地计算。权限层新增 v2 authorization/receipt，三个 processor 的 200-query
preflight 都通过；两类单结果 receipt 绑定 catalog、权限、wire、request、usage 和 latency。

我同时把评价声明降级：isolation v2 只证明角色、prompt、cache、输入和产物隔离，明确写
shared-model correlated risk，不能再说模型独立。4 次有界 live smoke 已证明默认 DashScope
endpoint 能收 Base64，并在 non-thinking/thinking 下返回可被严格 parser 接受的 JSON；但
1,000 行 production 矩阵回执、Assistant 上游接线和 Judge—human calibration 仍未完成，所以
core 继续 NO-GO。这个取舍让第一版工程闭环能推进，同时没有牺牲实验声明的诚实性。”

### 证据入口

- `specs/authoring/model-role-selection-v4.json`
- `specs/data_sources/c2/portfolio-mini-remote-processing-v1/owner-authorization-v2.json`
- `specs/data_sources/c2/portfolio-mini-remote-processing-v1/catalog-v2-receipt-v2.json`
- `src/skillchain/config.py`
- `src/skillchain/llm.py`
- `src/skillchain/evaluation/packets.py`
- `src/skillchain/evaluation/evaluator_isolation.py`
- `src/skillchain/evaluation/evaluator_outputs.py`
- `src/skillchain/evaluation/feedback_runtime.py`
- `src/skillchain/evaluation/final_runtime.py`
- `src/skillchain/evaluation/portfolio_stage1.py`
- `src/skillchain/evaluation/visual_runtime.py`
- `scripts/prepare_portfolio_stage1_handoff.py`
- `runs/portfolio/portfolio-dev-mini-smoke-dm-019-v1/stage1-input-v1/handoff-manifest.json`
- `tests/test_llm.py`
- `tests/evaluation/test_packets.py`
- `tests/evaluation/test_phase4_assurance.py`
- `tests/evaluation/test_feedback_runtime.py`
- `tests/evaluation/test_portfolio_stage1.py`
- `tests/data/test_portfolio_remote_processing.py`
- <https://help.aliyun.com/zh/model-studio/kimi-api>

### 2026-07-30 v5 后续修订

背景与影响：用户取得一个支持 OpenAI-compatible 多模态输入的 AIFast Gemini 凭据，Feedback
与 final 不再必须共用 Kimi。二者中更适合迁移的是 Feedback：它调用频率更高，任务是结合图片、
对话和 tool trace 做归因/建议，Flash 的定位与成本/延迟特征更契合；final Judge 直接决定五配置
主指标，保留已经跑通过 strict parser 和 thinking 参数的 Kimi 路径，可把迁移变量控制在一侧。

已验证事实：

- `/v1/models` 当时没有列出 Gemini，但对精确模型 ID `gemini-3.6-flash` 的真实文本请求成功；
  因此模型列表与实际路由不一致被记录为中转站限制，不能据此声称 Google provider 身份。
- Gemini 3.6 不再发送 temperature/top-p/top-k；AIFast OpenAI-compatible route 没有可验证的
  thinking 控制，因此配置显式记为 `gateway-default-unverified`，而不是伪造 non-thinking。
- `evaluator-isolation-v3` 要求 Feedback/Final 的 provider runtime 和模型家族均不同；Feedback
  processor 从 `dashscope-kimi-feedback` 前向换成 `aifast-gemini-feedback`，历史 v2 权限与
  v4 selection 不改写。
- v3 authorization/receipt 精确绑定同一 184 张图片；fresh runtime preflight 对 200/200 query
  通过。一次真实 Base64 图片调用返回精确模型名、`finish_reason=stop` 和可被严格 Feedback
  parser 接受的 JSON，仓库侧零重试；80 个聚焦离线测试通过。
- 本次 AIFast 共发出 3 个 HTTP 尝试：首次因 Windows 命令行 JSON 引号错误被网关在推理前拒绝；
  随后文本握手成功（4 input / 84 output token），Base64 图片 smoke 成功（1,148 input /
  309 output token）。两次有效模型响应合计 1,545 token；供应商费用不可得，不能自行填报。

根因与取舍：真正的问题不是“换一个模型常量”，而是 provider identity、采样参数、视觉 wire、
资产处理权限和 evaluator isolation 五个约束联动。把 final 改成 Gemini 会同时改变主评分模型、
thinking 行为和 parser 生产路径，迁移风险更大；把 Feedback 改成 Gemini 则恢复异源隔离，并保留
final 基线稳定性。因此最终选择 Feedback=Gemini、Final=Kimi，同时把第三方网关身份风险写进
selection、isolation lock 和契约。

剩余限制：AIFast 返回的精确模型名只证明网关响应契约，不是 Google provider-attested 证据；
网关默认 thinking、实际费用和后台路由不可观测。当前只完成单图 transport/strict-JSON smoke，
尚未产生 200-query × 5 configs 的 production Feedback/Final 矩阵，也未完成 Judge—human 校准。

### 当前 30 秒回答

“我没有直接把决定主指标的 final Judge 换掉，而是把高频、需要看图做失败归因的 Feedback 从
Kimi 迁到 Gemini 3.6 Flash。这样既保留了已验证的 Kimi final 评分路径，又恢复了两个 evaluator
的跨 provider 和模型家族隔离。我同步升级了 provider guard、Base64 wire、资产权限 preflight
和 isolation lock；真实图片 strict-JSON smoke 已通过。第三方中转站不是 Google 身份证明，
所以这项限制被明确写入契约，production 矩阵和人评校准仍待执行。”

### 当前 2 分钟回答

“一开始为了让本地图片能进入视觉 Feedback 和 Judge，我把两个角色都迁到 DashScope Kimi
K2.6，工程链路跑通了，但两个 evaluator 共享模型会产生相关偏差。后来有了支持 Base64 图片的
AIFast Gemini 路由，我没有机械地替换 final：final 直接决定五配置主指标，既有 Kimi thinking
参数、严格整数评分 parser 和本地 `J_project` 编译都已验证；Feedback 则是高频视觉归因和改进
建议，更适合 Flash。因此我只迁移 Feedback，把变量控制在评价链的一侧。

实现上，Gemini provider 只允许精确 `gemini-3.6-flash`，不发送 Gemini 3.6 已弃用的 sampling
参数，也不虚构中转站未证明的 thinking 控制。新的 isolation v3 强制 Feedback/Final 的 provider
runtime 与模型家族都不同；新的资产授权 processor 精确覆盖原 184 张图，调用前对 200 条 query、
plan、catalog、图片字节和权限回执做 fresh preflight。离线 80 个聚焦测试和一次真实 Base64 图片
strict-JSON smoke 均通过。与此同时，我把 `/models` 与实际路由不一致、第三方网关身份非
provider-attested、费用和默认 thinking 不可观测写进限制。这个方案恢复了实验设计的异源隔离，
但不把连通性 smoke 冒充 production 五配置结果。”

新增证据入口：

- `specs/authoring/model-role-selection-v5.json`
- `specs/data_sources/c2/portfolio-mini-remote-processing-v1/owner-authorization-v3.json`
- `specs/data_sources/c2/portfolio-mini-remote-processing-v1/catalog-v3-receipt-v3.json`
- `src/skillchain/evaluation/evaluator_isolation.py`
- `tests/test_llm.py`
- `tests/evaluation/test_feedback_runtime.py`
- `tests/evaluation/test_phase4_assurance.py`

---

## 41. 1×5 smoke 清单不能直接放大成 200×5 正式运行

**状态：NoSkill v6 已封存；runtime v10 首批 1×5 已重跑，剩余 35 个 shard 未授权**

### 一句话问题

单条 query 的五配置清单只能证明配置顺序和输入 loader 可用；若直接循环 1,000 次，会遗漏权限
版本漂移、时间顺序偏差、预算失控、部分失败恢复及 treatment Bank/runtime 绑定。

### 背景、证据与根因

启动盘点发现旧 checklist 仍绑定 v2 的 Kimi Feedback processor，而当前选择已是 v3 的 Gemini
Feedback；同时四份 treatment Bank 和 Assistant 真实工具 runtime lock 并不存在。主矩阵至少
包含 1,000 次 Assistant 与 1,000 次 final Judge，skilled 配置的 route/tool loop 又可能把
Assistant 调用放大到 5,000 次。Kimi thinking 输出计费且 AIFast 价格不可锁定，因此这不是简单
的双层 `for` 循环，而是一个需要调度、预算与恢复协议的批处理系统。

### 最终方案与验证

新增 Portfolio 专用的 active-v3 loader、八工具 runtime、四份 post-smoke Bank 和可恢复分片执行器：
200 条 query × 5 configs 形成 1,000 条自哈希实例，按 8 个 accepted batch × 5 configs 拆成
40 个 25-query create-only 分片；每个 batch 循环轮换首个配置，降低配置与服务时间的系统性耦合。
每条 Assistant 与 final Judge 结果独立 checkpoint，恢复时重新验证自哈希和强类型 schema，失败样本
保留为固定零分。预检还发现“工具确实执行”不等于“用户看见了结果”：若检索结果不投影为 cards，
大量 `requires_card=true` 的样本会被 Judge 系统性低估。因此 Runner 新增确定性的 DTO→visible
cards/evidence 投影，过滤内部 product ID、路径与 runtime binding。后续诊断又删除了 Runner
硬编码、位于 Skill Bank 之外的 evidence 充分性门；证据是否充分由 Skill body/operator allowlist
与 Judge 判定，避免在 NoSkill 路径偷偷增加另一套策略。

历史 launch v9 证明 1,000 个唯一 `(query, config)`、40 个完整分片和零 blocker 的准备链可行；
当前冻结链已经前向升级为 runtime v9 / launch v12 / execution v6，matrix run ID 为
`portfolio-dev-mini-200x5-v4`，并把原生 function-calling Runner、LLM adapter、Judge v3 parser、
registry、rubric 与盲化密钥一并绑定。NoSkill 与 S1+S2 两个 25-query 分片均完成全链路 dry-run，
模型调用为 0。此前真实 LLMStatic 预检达到 25/25 成功，全部可计量预检共 262 次 Qwen 调用、
约 CNY 0.1825272。

首个正式 NoSkill 分片又验证了恢复协议确实会在真实故障下工作：前台进程的输出管道被工具超时关闭后，
已有 checkpoint 没有重发；随后发现严格 Python-dict 重载不能把 JSON array 还原成 tuple，改为严格
JSON bytes 重载后原两条结果被成功复用。运行到 `dm-016` 时，隔离扫描器又把公开工具名
`encyclopedia_lookup` 中的 `encyclopedia` 误判成隐藏标签泄漏；修复为只扫描真正显示给用户的
文本/cards/citations/detections 后，原 Assistant checkpoint 直接进入 Judge，同样没有重跑。

历史 `execution-v3` 的 25/25 行可复载：Assistant 成功/失败=`16/9`，16 个 Judge 均为 `scored`，
9 个失败行固定零分；全分母 `J_project=44.86`，judged-only 均值 `70.09375` 只作诊断；
Assistant/Judge 调用分别为 42/16，费用 CNY 2.3493879。该结果现已明确降级为 immutable
protocol-polluted diagnostic，不能作为 NoSkill 基线。根因与修复见第 42 条。

最终冻结的 `execution-v6` 重新完成同一 25-query 分片：Assistant 成功/失败=`22/3`；其中两条在
三次工具调用后仍请求第 4 个工具，另一条在三次工具调用后的最终生成中以 `finish_reason=length`
截断，三者都作为真实 NoSkill 预算/终止表现固定零分；22 条 Judge 为 `21 scored + 1 parse_error`，
唯一 parse error 是 provider 返回 `finish_reason=stop` 但正文为空。固定 25 条主分母
`J_project=62.88`，scored-only 均值 `74.857142...` 仅作诊断；v6 新增费用 CNY 2.76214425，
包含 v3–v6 诊断链的累计费用 CNY 11.3847877。该分片已封存，不为获得更好数值继续追 parser 或重试。

后续集中审计发现旧四个 skilled shard 存在共享路由和服务故障耦合，故没有继续运行剩余分片。
修复后的 runtime v10 / launch v13 / execution v7 重跑同 batch 四个 skilled 配置，并外部引用上述
NoSkill v6；125 行比较的 shared route 25/25 一致、skilled runtime error=0。完整证据、指标与边界
见第 43 条。

### 剩余限制

首批已能做 25-query 描述性五配置比较，但不能替代完整 200-query 推断。按实测重算后的完整累计
point estimate 为 CNY `144.80049262`，只对未发生费用加 25% buffer 后建议累计 cap CNY 173；
launch v13 原 CNY 182 ceiling 偏保守 9 元。operator-approved 总 cap 仍为 CNY 51，尚需额外批准
CNY 122，剩余 35 个 shard 没有 execution 授权。v6 的 3 个 Assistant 零分是冻结基线行为，
Judge parse error 是 evaluator noise，二者必须分别统计。S2/S3 Bank 仍不能描述为已证明增益。

### 30 秒回答

“我没有把 1×5 smoke 直接套循环，而是把 200×5 变成 40 个可校验分片。每条 Assistant/Judge
结果 create-only、可断点恢复并持续核算预算。预检抓到工具结果没有投影成用户可见卡片的指标 bug；
正式首分片又抓到 JSON action envelope 被误当工具协议的基线公平性 bug。修复后同批 Assistant
从 16/25 提升到一次诊断中的 25/25；最终冻结 v6 诚实保留 3 个 NoSkill 过度调用零分和 1 个 Judge
空响应零分，没有用成功行均值掩盖失败。随后 runtime v10 补齐同批四个 skilled 配置；共享路由、
故障隔离和五配置审计结果见第 43 条。”

### 2 分钟回答

“从单 query 的 1×5 smoke 放大到 200×5，风险不只在调用次数。五个配置若按固定大块顺序运行，
服务时间漂移会和 treatment 耦合；中途失败若整批重跑会重复计费；只保存最终文本又无法判断模型、
工具或 Judge 在哪里失败。因此我先把 1,000 个实例锁成 40 个按 accepted batch 组织、轮换配置
起点的 25-query 分片，每条 Assistant 和 Judge 都 create-only 落盘，恢复时重验自哈希和 schema，
预算从真实 usage 回执累计，Assistant 失败仍以零分留在分母。

真实预检进一步暴露了一个容易漏掉的评价 bug：内部 tool trace 显示检索成功，但 `visible_cards`
仍为空。因为很多商品 query 要求 card，这会让 Judge 把呈现层缺失误判成模型能力差。我没有把完整
工具 DTO 直接塞给 Judge，而是做了确定性最小投影：商品只保留标题、类别、相关度，OCR/知识/检测
只保留用户应看到的证据，内部 ID、路径和 runtime binding 全部过滤。后续又删除了 Runner 中独立于
Skill Bank 的硬编码 evidence gate，避免 NoSkill 获得隐藏策略。最后把 Runner/LLM/Judge parser
源码哈希、八工具 registry、四个 Bank、rubric、权限回执和盲化密钥一起绑定。正式首分片以 v6
完成并保留所有失败；当时我只声称‘首个纵切可可信运行’，没有把单个 NoSkill 分片当作五配置效果。
后续 runtime v10 在不改写该基线的前提下补齐四个 skilled 配置，当前结论与限制见第 43 条。”

### 证据入口

- `src/skillchain/evaluation/portfolio_inputs.py`
- `src/skillchain/evaluation/portfolio_launch.py`
- `src/skillchain/evaluation/evaluator_outputs.py`
- `src/skillchain/evaluation/final_runtime.py`
- `src/skillchain/llm.py`
- `src/skillchain/tools/portfolio_runtime.py`
- `src/skillchain/runners/assistant.py`
- `scripts/prepare_portfolio_launch.py`
- `scripts/prepare_portfolio_execution.py`
- `scripts/run_portfolio_shard.py`
- `runs/portfolio/portfolio-public-data-runtime-v9/runtime-lock.json`
- `runs/portfolio/portfolio-dev-mini-200x5-launch-v12/launch-plan.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/execution-control.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v3/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v4/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v5/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `tests/evaluation/test_portfolio_launch.py`
- `tests/evaluation/test_evaluator_outputs.py`
- `tests/evaluation/test_final_runtime.py`
- `tests/runners/test_assistant_action_contract.py`
- `tests/runners/test_portfolio_assistant_projection.py`

---

## 42. JSON action envelope 不是工具协议：不能把 Runner 失败误当成 NoSkill 能力

**状态：已验证、修复并以冻结 v6 重跑**

### 一句话问题

论文语义中的 NoSkill 是“同一基础模型、同一工具、同一预算，但不提供 Skill Bank/路由策略”，不是
“让模型手写一套脆弱 JSON 控制协议”；若 Runner 本身制造失败，就不能把它解释成留给后续 Skill
优化的合理下界。

### 背景与影响

历史 `execution-v3` 首分片有 9/25 Assistant fail-closed。若直接接受它，会同时破坏两个结论：
一是 NoSkill 被人为削弱，五配置比较不公平；二是 S1/S2/Full 后续即使提升，也无法区分来自 Skill
机制还是来自更容易遵守的控制协议。这是指标有效性的 P0，而不是“基线本来就该差”。

### 可观察证据与根因

- v3 要求模型把 `tool_name/arguments`、最终文本、`selected_capability/skill_slug` 都写进 JSON
  action envelope，让控制面、业务数据面和 runner-owned identity 混在一个易截断对象中。
- 当 output budget 已到 0 时，旧逻辑仍通过 `max(1, remaining)` 发出额外请求；timeout 也只在远端
  调用返回后判断，因此预算与 deadline 都不是真正的前置约束。
- regex 恢复会执行自然语言或嵌套文本里的伪工具 JSON；结束阶段还要求模型重复本应由 runner
  决定的 identity，造成“任务答案正确但包装不合格”的零分。
- Runner 另有一个不来自 Skill Bank 的硬编码 evidence gate，使 NoSkill 仍隐含携带人工策略。
- 修复为原生 function calling 后，完全相同的 25 条 query 在 `execution-v4` 的 Assistant 达到
  25/25 成功；这个反事实证据说明 v3 的 36% 失败主要是协议缺陷，而非 NoSkill 能力。

### 方案取舍与最终选择

没有提高工具/turn/token 预算，没有靠 regex/截断修 JSON，也没有重试失败样本。Assistant 改为
OpenAI-compatible native function calling：从 live registry 生成 ToolSpec schema，单次只接受一个
原生 tool call，执行真实工具后回填 assistant `tool_calls` 与 `role=tool` history；普通非空自然语言
即为 final。NoSkill 可见全部工具但不读取 Bank，skilled 配置仍由 operator allowlist 过滤。route
identity、Skill identity 与终止权归 runner；每次远端调用前检查剩余 token、turn、tool 和总墙钟
deadline；删除 regex 伪工具恢复与 bank-external evidence gate。

Judge 的问题独立治理。v4 的 Assistant 已全部成功，但 Kimi 会返回语义等价的 assessment array、
score mapping 或 dimension-score pairs，且可能省略 envelope。于是 parser 采用前向版本化，而不是
改写历史 receipt：v1/v2 旧 run 仍按原策略逐字节重建；冻结 v3 接受三类 wrapped 与三类 bare 结构，
bare 的 `requires_card` 来自绑定 packet，同时继续严格检查单一 JSON、维度集合/顺序、排除 bool 的
整数与范围，不去 fence、不截取、不修复、不按结果重试。该行为符合百炼
[Function Calling 文档](https://help.aliyun.com/zh/model-studio/qwen-function-calling)所描述的原生
工具调用接口，而不是自造文本协议。

### 验证结果

- `execution-v3`（污染诊断）：Assistant `16/25`，主分母 `J_project=44.86`。
- `execution-v4`（Runner 修复证明）：Assistant `25/25`；Judge 的 4 条等价 shape 被旧 parser
  fail closed，证明 Assistant P0 已关闭而 evaluator P1 尚未关闭。
- `execution-v5`（parser v2 前向诊断）：Assistant `24/25`，唯一失败是三工具预算耗尽后继续调用；
  Judge `23 scored + 1` 裸 mapping parse error。
- `execution-v6`（最终冻结）：Assistant `22/25`，其中 2 条超过三工具预算、1 条最终生成以
  `finish_reason=length` 截断；Judge 对其余 22 条为
  `21 scored + 1` 空正文 parse error。三类 wrapped shape 与 bare score mapping 均已单次计分，
  因而该 parse error 是 evaluator/provider 空响应，不是 parser shape drift。固定主分母
  `J_project=62.88`，scored-only `74.857142...` 只作诊断。
- 最终审计记录 63 次 Assistant、22 次 Judge、38 次真实工具调用；v6 新增费用 CNY 2.76214425，
  全诊断链累计 CNY 11.3847877。聚焦 Runner/Judge 回归在封锁前为 209 passed、4 deselected，
  且 v4/v5 历史 audit 可由新版 finalizer 原样重建。

### 剩余限制

NoSkill 在相同 temperature/budget 下仍有随机过度探索，这正是基线行为，不能挑最好一次；Kimi
thinking 的 latency/token 和偶发空响应也是 evaluator 噪声，必须与 Assistant 能力失败分栏统计。
当前只有一个 25-query NoSkill 分片，不能推断五配置效果；Judge—human calibration 和其余 39 个
分片仍待完成。v6 是本轮最终前向策略，不再启动第四轮 parser 追逐。

### 30 秒回答

“首个 NoSkill 分片最初有 36% 失败，但我没有把它包装成‘弱基线’。论文里的 NoSkill 只是移除
Skill，不是移除可靠工具协议。我定位到模型手写 JSON 同时承担工具调用、终止和 identity，另有
token/timeout 越界与 regex 伪恢复。改成原生 function calling 后，同批 Assistant 一次达到 25/25。
随后我把 Judge shape drift 单独版本化治理；最终冻结 v6 仍诚实保留 2 个真实 NoSkill 过度调用、
1 个最终生成截断和
1 个 Judge 空响应零分，主分母是 62.88。”

### 2 分钟回答

“这个案例最关键的是错误归因。旧 Runner 要模型在 JSON 里同时写工具名、参数、最终答案和本应由
Runner 持有的路由身份；输出一截断，哪怕任务本身做对也会零分。它还会在剩余 token 为 0 时多发
一次调用，timeout 是事后判断，regex 甚至可能执行自然语言里的伪工具 JSON。若把这种 16/25 接受为
NoSkill，就会人为压低基线，使任何 Skill 增益都不可信。

我保持模型、查询、工具和预算不变，只修控制面：使用 provider 原生 function calling，从冻结 registry
生成 schema，真实执行工具并把 `role=tool` 结果回填；身份和终止由 Runner 决定，预算与 deadline
在调用前检查，也移除了 Bank 之外的 evidence gate。相同批次随后出现过 25/25 Assistant 成功，证明
原来的主要失败不是能力下界。

之后我没有把 Judge 问题混回 Assistant。Kimi 的分数内容正确，但会换 assessment array、mapping、
pairs 或省略 envelope。我用 v1/v2/v3 前向 parser 保留旧 receipt 的字节兼容，v3 只接受六种严格
等价结构，仍不修复、不重试。最终 v6 中 2 条 Assistant 因真正超过三工具预算而零分，另 1 条
Assistant 因最终生成截断而零分，1 条 Judge 因空正文零分；这些类别分别报告。这样得到的 62.88
未必最好看，但它才是后续五配置比较可解释的
NoSkill 起点。”

### 证据入口

- `src/skillchain/runners/assistant.py`
- `src/skillchain/llm.py`
- `src/skillchain/evaluation/evaluator_outputs.py`
- `src/skillchain/evaluation/final_runtime.py`
- `scripts/finalize_portfolio_shard.py`
- `tests/runners/test_assistant_action_contract.py`
- `tests/runners/test_portfolio_assistant_projection.py`
- `tests/evaluation/test_evaluator_outputs.py`
- `tests/evaluation/test_final_runtime.py`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v3/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v4/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v5/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/shards/00-dev-mini-001-r3-00-noskill/shard-audit.json`

---

## 43. 单独重跑 Full 会把路由随机性和服务故障误归因给 Body Refiner

**状态：已修复并完成首批 1×5 重跑；比较无 P0/P1 blocker，Judge parse noise 待分类**

### 一句话问题

五配置全部“跑完”不等于比较有效：如果 S1+S2 与 Full 分别调用 router，或一次服务故障连续把
Full 后半段写成零分，那么 `Full − S1+S2` 就同时混入路由差异和基础设施噪声，不能解释为
Stage 3 Body Refiner 的贡献。

### 背景与影响

launch-v12 把 200×5 拆成 8 个 batch、40 个 create-only shard，并用循环配置顺序降低服务时间与
treatment 的长期耦合。用户要求先冻结 NoSkill v6，再按 ordinal 01–04 完成同一 batch 的
LLMStaticSkill、S1、S1+S2、Full，最后集中检查 skilled-only 错误和比较公平性。

这次审计直接影响第一版的两个完成条件：五配置量化比较是否可信，以及 S3 原型是否能进行合法的
接受/回滚判断。若忽略该问题，项目会得到一张看似完整、实则把实现和服务故障当算法效果的结果表。

### 已验证事实

- NoSkill v6 未重跑；旧 shard audit SHA-256
  `1bae36e1bab96e13645e1e5f78b296a15002d0fb91c050ef3c4a9f96a2cd3047`
  与集中审计观测值一致。
- 五个 shard 均有 25 个 Assistant、25 个 Final、complete summary 和通过绑定重验的 shard audit。
- 125/125 paired 行的 query/image、Assistant backbone/system prompt/预算、registry、rubric、
  Judge/parser、Bank capability/operator 和 Assistant→Final identity 未见漂移。
- 评价协议明确要求 S1+S2 与 Full 逐 query 复用同一 Stage-2 route decision；当前 runner
  对每个 skilled config 分别调用 router，也没有持久化共享 decision identity。
- 前 15 个 Full 行中，有 15 对独立 route response；14 对 semantic capability 相同，
  `dm-003` 则是 S1+S2 路由响应后失败、Full 独立路由成功。其余 10 对因 Full 错误无法比较。
- Full 的 `dm-015` Judge 为 provider error；紧接着 `dm-016`～`dm-025` 连续十条 Assistant
  均为 0 model response、0 token、0 tool、约 2.9–3.9 秒的 pre-response runtime error，
  并在约 32 秒内写完。
- S1 有 11 个 Assistant error；其中 6 条已经得到 `finish_reason=stop` 的 route response，
  随后在本地 route parse/schema/Bank binding 边界被拒。现有产物不能继续区分叶级原因。
- Assistant 180 秒预算曾留下 606 秒和 1655 秒的 timeout receipt，说明远程调用的端到端
  wall-clock deadline 没有被严格执行。
- 集中审计 v2 file/audit SHA-256 分别为
  `0ac81f36e4e9650af1c366a6e156f184ca80df14937b14ca9abdfa297883a5eb` /
  `eec577b5869b02577e88d5e10fb35aab43f8ea63b132652d90e07125c2873e24`，状态为 `failed`。

### 推断与证据边界

Full 的十条连续失败与同窗口 Judge provider error、Codex 连接中断高度一致，因此外部网络或
provider 服务窗口故障是强推断；但 runner 的 generic catch-all 没有保留底层异常，不能进一步
声称是 DNS、TCP、网关、鉴权还是 response-contract。

S1 route 错误集中发生在它的 Description 仍偏任务说明、S2 尚未加入明确意图边界的 capability，
而 S1+S2 大多成功，这与 Route Optimizer 生效方向一致。但 route 协议还要求模型同时复制
semantic capability 和配置特定 slug；在 raw response/subtype 缺失时，不能把全部零分直接解释为
S1 路由能力差，也不能为了得到好看结果选择性重跑。

### 根因

根因不是单个模型“表现不好”，而是运行时边界没有把三类状态分开：

1. **共享实验变量缺失：** Full 没有消费 S1+S2 已冻结的 route artifact，而是重新采样 router。
2. **冗余接口字段：** 模型同时输出 capability 与 slug，runner 再检查两者成对；格式/复制失败会
   混入 semantic routing 指标。
3. **失败证据过度折叠：** provider、route parse、Bank binding、action failure 最终大多只剩
   `runtime_error`，且错误时清空已完成的 route identity。
4. **没有故障熔断：** 连续 pre-response provider failure 后仍推进 query，导致服务窗口与 Full
   后 40% 完全耦合。
5. **SDK timeout 不是整体 deadline：** 单次 transport 参数不足以保证 runner 的总 wall-clock
   上限。

### 方案取舍与最终选择

考虑过三个方向：

- **接受固定零分并继续 7 批：** 保持历史不变，但会把已知 P0 扩散到 875 个后续实例；拒绝。
- **只重跑 Full 的失败十条：** 成本最低，却会选择性修补最差配置，并继续违反共享路由契约；
  拒绝。
- **停止、封存诊断、升级运行边界：** 保留真实失败且不改写历史；新版本先产生每 query 唯一的
  Stage-2 route artifact，再让 S1+S2 与 Full 共同消费；采用此方案。

最终没有启动 ordinal 05 之后的 35 个 shard。NoSkill v6 保持冻结，四个 skilled shard 作为
immutable diagnostic；后续修复不得覆盖 execution v6。

### 如何验证

- 集中审计重建 125 个 Assistant→Final packet，逐项验证 paired input、模型/预算、registry、
  Bank/operator、rubric/Judge 和 fixed-zero。
- 对五个 shard summary/audit 做 canonical self-hash 及 runtime/launch/summary 绑定重验。
- 用预先观测的 NoSkill audit SHA 做 digest-only freeze 证明，并明确它不是 wall-clock 顺序证明。
- 新增 S1+S2/Full route reuse 硬门，逐 query 输出 semantic matched/mismatched/unverifiable、
  route response hash 和独立调用计数。
- 修正审计器最初按 asset ID 唯一索引的错误：正式语料允许同图跨意图，现使用
  `(asset_id, image_path, canonical_intent)`；23 个聚焦测试和 Ruff 通过。

### 修复后的已验证事实（2026-07-31）

- 新链为 runtime v10 / launch v13 / execution v7；runtime lock、launch plan、execution control
  SHA-256 分别为 `041f3b92ede5ffff3ed50bb1b0888511694651c837e0d42e73b8341545d36c5a`、
  `c44f35b0c5d3477a698a4e338caf9d8281e5a6a1bb2786a28e5288a2e9ce22fd`、
  `1e5fee6dda705bdc6754de7852dd9e93e3e8eaec9094e28aa917e285ea05dc9c`。
- NoSkill v6 没有重跑。execution v7 以 external frozen reference 深验其 25 Assistant、25 Final、
  summary、audit 共 52 个文件；source audit 仍为
  `1bae36e1bab96e13645e1e5f78b296a15002d0fb91c050ef3c4a9f96a2cd3047`，并通过独立
  comparison contract 绑定可比的 corpus/catalog/image、模型端点、采样/预算、registry、rubric、
  parser 与 NoSkill native-tool contract。
- runtime v10 将 route 从 config 私有模型调用改为每 query 唯一、create-only 的
  `shared-stage2-route-v1` artifact。模型只选 capability；S1+S2 与 Full 按各自 Bank 确定性映射
  slug，引用同一 route identity，并显式预留相同 route turn/usage，action budget 不因复用而变化。
- `portfolio-shard-attempt-v1` 把 pre-response provider failure 与终态任务失败分离：失败只写 attempt
  receipt，停止在最早未完成 query，不写 fixed-zero；连续 2 次打开 circuit，每 query 最多 2 次
  retryable attempt。该首批没有触发 attempt receipt 或熔断，这验证了正常路径，故障路径由聚焦测试
  覆盖，尚不能声称经过第二次真实 provider outage 验证。
- 新集中审计 SHA-256 为
  `e7dbc36077d1dac149aabcf38b24ca36c423bb8e861ba8bcf9b88f9cc0e690d4`：125 行完整、
  paired-input violation=0、skilled runtime error=0；S1+S2/Full route 25/25 matched、0 mismatch，
  shared decision identity 全部持久化。
- 五配置 `J_project` 均值为 NoSkill `62.88`、LLMStaticSkill `75.30`、S1 `67.94`、S1+S2
  `74.56`、Full `74.26`。NoSkill 保留 3 个 Assistant fixed-zero；四个 skilled 配置均为 25/25
  Assistant success。NoSkill `dm-005` 与 S1 `dm-012` 各有一个 Judge parse error，按既定规则保留
  零分并单独标记。因此审计是 `review_required`、blocker 为空，而不是已建立任何算法增益。
- execution v7 本地新增成本 CNY `12.70921185`，加历史 carry-in 后累计 CNY `33.90707085`，低于
  本阶段累计 cap CNY 38 和 operator-approved 总 cap CNY 51。按首批实测重算，剩余 7 批 point
  estimate 为 CNY `110.89342177`，完整累计 point estimate 为 CNY `144.80049262`；只对未发生费用
  加 25% buffer 后为 CNY `172.52384806`，故建议累计 cap 向上取整为 CNY 173。launch v13 原 CNY
  182 ceiling 偏保守 9 元；相对现有批准仍需额外批准 CNY 122，`dashscope_budget_approval_required`
  继续阻止剩余批次。

### 剩余限制与待完成工作

- 两个 Judge parse error 需作为 evaluator noise 分类并在最终报告单列；不能选择性补判或把它们改写成
  Assistant 错误。NoSkill 的 v6 结果属于冻结历史，继续保持不变。
- 首批 25 query 太小，且 LLMStaticSkill 暂时最高、Full 比 S1+S2 低 `0.30`；这些是已验证的描述性
  数字，不是改善或回退结论。待完成 200 query 和配对不确定性分析后再判断算法方向。
- 熔断机制的真实 outage 行为仍待后续自然故障验证；不得主动制造付费故障，也不得通过重复调用挑选
  最佳结果。
- 启动剩余 7 个 batch 前需要新的 execution scope 与预算批准；当前建议累计 cap 为 CNY 173，
  相对现有批准仍差 CNY 122。launch v13 原 CNY 182 只是偏保守的旧 planning ceiling，不自动授权。

### 30 秒回答

“首批旧运行暴露了两个会破坏归因的问题：S1+S2 和 Full 各自重采样路由，且一次 provider 故障
连续污染 Full 后十条。我没有选择性补跑，而是封存 execution v6 和 NoSkill 基线，停止后续批次。
runtime v10 把每条 query 的 Stage-2 路由做成 S1+S2/Full 共用的不可变 artifact，并加入 attempt
receipt 与两次 pre-response failure 熔断。新首批审计验证 25/25 路由一致、四个 skilled 配置零
runtime error；但两条 Judge parse error 仍单列，所以现在只能说比较接口修好了，不能说算法已提升。”

### 2 分钟回答

“这个问题的关键是区分‘系统真的做差了’和‘实验接口制造了差异’。旧首批虽然凑齐五配置，但
S1+S2 和 Full 分别调用 router，Full 后十条又在同一服务窗口内全部零响应失败。前者让
`Full − S1+S2` 混入路由随机性，后者让 treatment 与基础设施时间窗口耦合。只补 Full 最差十条
会形成选择性重试，所以我封存 execution v6、保留全部失败证据和 NoSkill v6，不再改写历史。

新 runtime v10 做了两项针对性修复。第一，每个 query 只创建一份 Stage-2 route artifact；模型只
选择 capability，S1+S2 和 Full 再按各自 Bank 确定性映射 slug，并为两边预留相同 route usage 和
action budget。第二，把 pre-response provider failure 从任务终态拆开：只写 create-only attempt
receipt、停在最早未完成 query，连续两次才开 circuit，基础设施失败不会立刻变成算法零分。

我没有重跑 NoSkill，而是在 execution v7 中逐文件复验旧 shard 的 52 个工件，并用 comparison
contract 锁住数据、模型、预算、registry、rubric 和 parser。新首批 125 行审计显示 paired 输入零
违规，四个 skilled 配置都是 25/25 Assistant success，S1+S2/Full 的 25 条 route identity 全匹配。
均值是 62.88、75.30、67.94、74.56、74.26；其中仍有两条 Judge parse error，因此审计状态是
`review_required` 而非算法通过。我的结论是因果比较接口已经修复，但 25 条描述性结果没有证明
Skill 增益；要等预算授权后的 200-query 配对分析再下结论。”

### 证据入口

- `docs/evaluation-protocol.md`
- `docs/reproduction-contract.md`
- `src/skillchain/runners/assistant.py`
- `scripts/audit_portfolio_batch.py`
- `tests/evaluation/test_portfolio_batch_audit.py`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/audits/dev-mini-001-r3-cross-config-audit-v2.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/audits/dev-mini-001-r3-audit-decision-v1.md`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v6/shards/04-dev-mini-001-r3-04-full/shard-audit.json`
- `runs/portfolio/portfolio-public-data-runtime-v10/runtime-lock.json`
- `runs/portfolio/portfolio-dev-mini-200x5-launch-v13/launch-plan.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v7/execution-control.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v7/audits/dev-mini-001-r3-cross-config-audit.json`

---

## 44. treatment 名称、Bank 文件和哈希齐全，仍不能证明 Creator/Optimizer/Refiner 真正执行

**状态：已验证；v7 为 scaffold diagnostic，stale-parent 中间链均 superseded，当前有效链为 runtime v22 → launch v25 → execution v16**

### 一句话问题

为了尽快补齐五配置 Bank，开发期生成器在 post-smoke 后用确定性 scaffold 拼出了可运行文件；但启动链
只检查文件、名称和哈希，没有证明 S1 Creator、S2 Route Optimizer、S3 Body Refiner 真正按各自
语义执行，导致 scaffold 被错误地带着正式 treatment 名称进入首批矩阵。

### 背景、影响与证据

- 对 execution v7 使用当前审计器重验后状态为 `failed`；S1→S1+S2 在六个 capability 上都修改了
  Body，直接违反“S2 只能修改 Description”的阶段边界。
- 因此 v7 的五配置均值 `62.88 / 75.30 / 67.94 / 74.56 / 74.26` 只能描述旧 scaffold 的运行
  行为，不能用于判断 Creator、Optimizer、Refiner 是否复现论文结论。
- 第一轮真实替换链 v11 仍有 lineage 缺陷：S2/S3 attribution 的来源 Bank 分别为旧的
  `259e4895…` / `89179f16…`，而 gate-selected current parent 是 `9811a647…`；真实模型调用本身
  不能把 stale-parent 输入升级成闭环证据。
- 问题会同时破坏 treatment 真实性、阶段增益归因和 200×5 比较公平性，属于启动正式矩阵前必须
  关闭的 P0。

### 根因

根因是把“工件完整性”误当成“treatment 语义真实性”：

1. readiness gate 只验证 Bank 存在、schema 合法、hash 匹配，没有验证生成来源和模型调用证据。
2. S1/S2/S3 没有独立的输入 packet、模型 receipt、候选 Bank、阶段 diff 和 lineage。
3. 接受/回滚结论可以由汇总文件声明，不能从逐 query smoke 结果重新计算。
4. runtime finalizer 不会从原始输入重放编译和 mutation，重命名的 scaffold 因而可以混入正式链。
5. 临时 scaffold 与正式 treatment 共用命名空间，没有显式 `official_matrix_eligible=false`。
6. 第一轮修复只证明“模型调用存在”，没有强制 S2/S3 attribution 绑定上一 gate 的精确父 Bank 和
   同一 parent smoke；gate report 也未独立解析 result-set，留下 stale evidence 被重新包装的空间。

### 最终选择

- 所有确定性补 Bank 脚本改为 diagnostic scaffold，名称和 manifest 明确标记
  `official_matrix_eligible=false`。
- S1 使用冻结的 25-query Creator packet；S2/S3 使用从各自 current-parent 六 query smoke 重建的
  attribution packet。S2 在 new persistent session 的 turn 1 执行，S3 绑定 S2 detailed receipt 与
  gate 后 resume 同一 thread 的 turn 2；保存 prompt、response、token、程序/可执行文件身份和全部
  输入/输出哈希，不允许 retry 或 fallback。
- 编译器强制阶段语义：S1 创建完整 Bank；S2 只能修改 Description；S3 只能修改 Body，并绑定父
  Bank 和 mutation lineage。
- 每阶段在相同六 capability query 上运行真实 Assistant+Judge smoke，由逐行结果重新计算
  route accuracy、`J_project` 和 hard error，再按预先固定的 Pareto 规则接受或回滚。
- finalizer 从保存的输入和模型输出重新执行 LLMStatic/S1 编译及 S2/S3 mutation，逐项重验 typed
  gate result-set、attribution↔gate-parent 同源关系、session/implementation identity 与 rollback；
  发布后再由独立 loader 重放 implementation snapshot，launch 对没有真实 chain manifest 的 Bank
  fail-close。

### 修复后的结果

- 当前 gate 接受 S1：`J 68.90 → 77.12`、route `0.84 → 0.88`、adherence
  `0.585000 → 0.621667`、hard error `1 → 0`。
- 当前 gate 接受 S2：`J 77.12 → 77.44`、route `0.88 → 0.92`、adherence
  `0.621667 → 0.656667`、hard error 保持 `0`。S2 只修改 Description。
- 当前 gate 回滚 S3：同一 stage sample 上 parent `J 75.56`、candidate `73.80`；route 都是
  `0.92`，adherence 从 `0.683300` 升到 `0.698300`，hard error 都是 `0`。S3 只修改 Body，
  被拒候选和回滚证据完整保留。
- 最终 Bank 为 Static `40a16d…`、S1 `9811a6…`、S1+S2 / Full `9502ef…`。Full 与 S1+S2
  字节相同是 S3 rollback 的部署结果，不是不同名字包装 scaffold。
- clean optimization25 已完成：五配置 Mean J 为 `73.34 / 73.04 / 76.14 / 77.92 / 76.64`；
  S1+S2 相对 Static `+4.88`，route macro-F1 `0.8913 → 0.9277`。但 CI95
  `[-0.4220, 9.7826]` 跨 0，不能当作独立评测结论。
- S1+S2 / Full 使用 25/25 同一 route 和同一 Bank；no-op 配对差为 `-1.28`、CI95
  `[-6.8701, 4.0005]`，审计明确 `treatment_attribution_allowed=false`。
- v16 集中审计 125/125 行完整、paired input 零违规、Assistant/tool/hard error 均为 0、
  `blockers=[]`。唯一非计分行为 S1 `dm-020` 的 Judge 连续格式失败，按冻结两次上限保留零分。
- comparison importer、external-frozen analyzer 和 Judge comparison projection 的三项收口缺陷已修复；
  最新相关回归 137 项及 Ruff 通过。完整 evaluation175 因当前剩余预算不足而未启动。

### 剩余限制

- development gate 与 optimization25 都不能替代独立 175-query evaluation；论文的 S2/S3 还分别
  要求每 Skill 至少 30/50 个样本与多轮聚合。
- S1 的一条 Judge anomaly 会使 25 行官方均值相对仅诊断补 wrapper 低 3.40 pp；相同 Bank/route
  的 no-op 差异也说明当前阶段排序受采样噪声影响。
- 下一版算法空间是 failure-driven S1、confusion-pair 最小 S2 patch、多样本 aggregation、规则与
  Judge 分流、稀疏 S3 Body patch 和包含 CQ/CCC/用户语义的 Pareto gate。本版已停止生成候选，
  不能原地修改 runtime v22 或挑选性补跑。

### 30 秒回答

“我发现五个 Bank 虽然名称、schema 和哈希都齐全，S1/S2/S3 实际却沿用了开发 scaffold；第一轮
替换又用了 stale parent。于是我把 scaffold 永久降级为 diagnostic-only，让每阶段强制绑定父 Bank、
模型 receipt、字段级 diff 和可重算 gate。当前真实链接受 S1、S2，回滚 S3；clean 25×5 里
S1+S2 比 Static 高 4.88 分、路由 macro-F1 从 0.891 升到 0.928，但这只是 optimization25，
evaluation175 尚未跑，所以我不把它包装成论文复现结论。”

### 2 分钟回答

“这个缺陷说明实验工件完整不等于 treatment 真实。之前为了尽快补齐五个 Bank，开发脚本在 smoke
后确定性生成了占位版本；后来启动器只验证 schema、文件和 hash，正式名称就掩盖了来源问题。用新
审计器重放旧 execution v7 后，我发现 S1→S1+S2 的六个 capability 全部发生 Body 变化，而 S2
按定义只能优化 Description，因此旧矩阵不能解释论文里的 Creator、Router Optimizer 或 Body
Refiner。

修复分四层。第一，scaffold 永久标成 diagnostic-only，不能进入矩阵。第二，S1 从冻结 25-query
Creator packet 出发，S2/S3 则从 current-parent 六 query smoke 重建 attribution；S2 创建 persistent
session，S3 绑定 S2 receipt/gate 后 resume 同一 thread。第三，编译器对 S2/S3 做字段级 diff 和
lineage 约束。第四，gate 从逐 query Assistant+Judge 结果重算为 typed result-set，finalizer 与 loader
再独立重放 attribution 同源关系、session/implementation identity 和整个链。

当前真实 gate 里，S1 把 J 从 68.90 提到 77.12、hard error 从 1 降到 0；S2 只改 Description，
把 route 从 0.88 提到 0.92，并保持质量 Pareto 可接受；S3 只改 Body，虽然 adherence 上升，J 却
从 75.56 降到 73.80，所以回滚。最终 Full 与 S1+S2 使用同一 Bank。clean optimization25 的均值
是 73.34、73.04、76.14、77.92、76.64；S1+S2/Full 还复用 25/25 同一路由，它们的 -1.28 分差
被审计标为 no-op sampling noise。另有一条 S1 Judge 连续格式失败，严格保留零分，说明小样本排序
仍敏感。工程闭环已经真实，但效果结论必须等独立 evaluation175；当前剩余授权不足，所以我停在
可复核结果和预算边界，而没有选择性补判或继续调 Bank。”

### 证据入口

- `runs/portfolio/portfolio-real-treatment-development-v1/diagnostics/legacy-v7-batch-audit.json`
- `runs/portfolio/portfolio-real-treatment-development-v1/gates/s1/gate-report.json`
- `runs/portfolio/portfolio-real-treatment-development-v2/inputs/s2/packet.json`
- `runs/portfolio/portfolio-real-treatment-development-v2/inputs/s3/packet.json`
- `runs/portfolio/portfolio-real-treatment-development-v2/gates/s2/gate-report.json`
- `runs/portfolio/portfolio-real-treatment-development-v2/gates/s3/gate-report.json`
- `runs/portfolio/portfolio-public-data-runtime-v22/runtime-lock.json`
- `runs/portfolio/portfolio-public-data-runtime-v22/treatment-chain-manifest.json`
- `runs/portfolio/portfolio-dev-mini-200x5-launch-v25/launch-plan.json`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v16/execution-control.json`
- `runs/portfolio/portfolio-dev-mini-200x5-audits-v3/dev-mini-001-r3-integrity-audit.json`
- `runs/portfolio/portfolio-dev-mini-200x5-analysis-v3/report.md`
- `src/skillchain/evaluation/portfolio_treatments.py`
- `src/skillchain/evaluation/portfolio_treatment_io.py`
- `scripts/finalize_portfolio_treatment_runtime.py`

---

## 45. 调用方 timeout 不等于 materialize worker 已停止

**状态：已验证并修复**

### 一句话问题

监督工具返回 timeout 后，子进程可能仍在写入；立即“恢复”同一输出根会制造第二个 writer，而不是安全续跑。

### 事实与证据

- v9 RPC Multi 物化时，工具超时后进程命令行仍显示原 worker 存活；一次过早续跑短暂造成两组 writer。
- 停止较新的 writer、让原 worker 正常完成后，发现一个临时 checkpoint 副本；它与 canonical checkpoint 字节和 SHA-256 完全相同，已可恢复隔离并留有 receipt。
- 对 1,022 个 checkpoint、目标文件、drafts、run manifest 和 selection manifest 做全量只读复验，结果为 0 drift；同参单 writer 回归为 `copied_this_call=0`。

### 根因与处置

根因是把调用方 timeout 误当作 worker 退出。处置顺序是先查 PID/命令行、停止新 writer、保留原 writer，完成后隔离临时副本并复验全部已发布对象；没有删除原始来源或改写 canonical 产物。

### 最终方案与验证

materializer 现在以原子 create-only writer lock 串行化同一 output root。锁存在即 fail closed，并提示操作员先确认锁中 PID 对应的命令行已退出，再显式处置保留锁；绝不自动猜测 stale lock。聚焦锁测试、完整 `test_portfolio_core_assets.py`（14 passed）和真实 v9 单 writer 回归均已通过。

### 剩余限制

硬杀进程会保留锁，需要人工检查与恢复；这是为了避免以自动清锁换取第二个 writer 的风险。该机制保证本项目 materializer 的单 writer，不替代跨机器的分布式锁。

### 30 秒回答

“我遇到过一次看似普通的 timeout：界面已返回，但数据物化 worker 仍在运行。若马上重试，就会有两个进程同时写同一 checkpoint 树。我先通过命令行确认旧 worker 还活着，停止新 writer，保留旧 writer 完成；随后把唯一临时副本可恢复隔离，并对 1,022 个对象做 0-drift 复验。最后我加了原子 writer lock：timeout 后默认拒绝恢复，必须先确认旧 PID 退出。这比自动清理锁更保守，但能保护可复现数据产物。”

### 证据入口

- `src/skillchain/data/portfolio_core_assets.py`
- `tests/data/test_portfolio_core_assets.py`
- `portfolio-core-v9-multi-closure` 的 selection manifest 与 incident-retention receipt

---

## 46. 同一 query_id 只锁成员，不授予旧正文复用权

**状态：已验证并落地**

### 一句话问题

跨 revision 保留相同 query_id、顺序和任务标签，只能说明计划 lineage 连续；它不能证明旧正文仍引用同一图片、满足同一 turn-shape，或可以安全复用。

### 背景与影响

r2 的目标是在保留 r1 core 前缀成员与实验可比性的同时，修复路径重绑定、Document 扩容和 realism 结构。若把“ID 一样”误当成“内容也一样”，最省事的做法是把历史 dev 正文直接抄入 r2。这会让新的 asset binding、对话结构和边界分布在表面不变的 ID 下悄悄失真，既污染 r2，也使后续训练比较无法解释。

### 观察到的证据

已验证事实：

- r1 core plan 与 r2 前 200 条的 plan_id、顺序、asset_id、leakage_group_id、capability 和 intent 均为 200/200 对齐；这只证明计划成员 lineage。
- 旧 accepted dev 数据相对 r2 前 200 槽位的 image_path 对齐为 0/200。
- 旧正文的 turn shape 为 single=188、three=12；r2 realism 需要 single=140、three=60，只有 142/200 形状相符。
- r2 的 body-free pre-generation manifest 为 bb4a563cfc2124484348039ad014cfefde73fb2763967965bb69f58ce46002a7，绑定 r2 plan SHA b1e637a6801eefa4a66eab8c05ec5a03ded31e1693e0296deb46aee82602dc11，却不含正文。这个 receipt 证明的是“先冻结了什么”，不是“旧文本可以移植”。

### 根因

系统原先容易把三层 identity 混成一个：

- 计划 identity：query_id、顺序、意图、capability 和 split；
- 资产 identity：image_path、asset bytes、component 与 author alias；
- 正文 identity：用户表达、turn shape、约束演化和最终文本 hash。

前两层有可控的 lineage 迁移，第三层必须重新创作并重新验收。相同 query_id 并不提供正文层的等价证明。

### 考虑过的方案与取舍

1. 复用全部旧 dev 正文：最快，但与 0/200 image_path 和 58/200 turn-shape 不匹配冲突，淘汰。
2. 只在 142 条形状相符的样本上复用：仍没有资产绑定或正文语义等价证明，且会破坏 batch 内的新结构，淘汰。
3. 为 r2 改写历史 accepted 记录：会篡改 r1 证据，淘汰。
4. 冻结计划 lineage、让 r2 全部正文从 packet 新写：保持可比性与不可变历史，采用。

### 最终方案

r2 的 legacy_prefix 只承载计划级 lineage；正文不继承。每个 r2 batch 必须从当前 author packet 的 opaque asset alias 与 realism 卡出发，产生全新 plan_id + turns，再用本 revision 的 receipt、schema 校验和 execution ledger 绑定。旧 dev 保留为历史证据，而不是 r2 的隐式 seed。

### 如何验证

- E:/skillchain-data/runs/portfolio-core-20260804-r1/plans/core.json 的 SHA-256 为 90c8d4bfad250f486d5111045914f70abb638fe61ef20ba23749f26afd451625。
- E:/skillchain-data/runs/portfolio-core-20260804-r2/plan/core.json 与其 body-free pre-generation manifest 分别绑定上述 r2 plan SHA 和 manifest SHA。
- 只读比较输出了 0/200 image_path 与 142/200 turn-shape 证据；这些数字被记录在 r2 pre-generation handoff，而没有把旧正文写入 r2。
- 首批 r2 execution 的接受记录使用新 revision 的 draft receipt 与 results SHA，而不是历史 dev 正文 hash。

### 剩余限制

这些比较只说明复用不成立，不能自动证明新正文自然或任务语义正确。1,500 条完成后仍需执行 cluster-aware 的 200 条 owner 抽检，并以五配置真实运行检验训练价值。

### 30 秒回答

“r2 保留了 r1 的 query ID，是为了让计划成员和实验 lineage 可比，但我没有因此复用旧正文。实测旧 dev 和 r2 的图片路径是 0/200 相同，turn-shape 也只对上 142/200。于是我把 identity 拆成计划、资产和正文三层：ID 只锁计划，正文必须按新 packet 重新写、重新 hash、重新进入执行账本。这样既保留版本可比性，也不会把旧语料伪装成新 revision。”

### 2 分钟回答

“这是一个很容易被忽视的数据版本问题。为了保证 r2 和 r1 可比，我们锁住了前 200 条的 query_id、顺序、能力和意图；如果只看表格，会很像可以直接迁移旧 dev 文本。但我把旧 accepted 数据和 r2 plan 做了只读对照：图片路径 0/200 对齐，旧的 188 个单轮、12 个三轮也不符合 r2 的 140/60 realism 要求，只有 142 条 shape 碰巧一致。

我因此把身份边界明确拆开：计划 ID 负责 lineage，asset binding 负责实际图像，正文 identity 由当前 revision 的表达、轮次和 hash 组成。r2 的 pre-generation manifest 是 body-free receipt，只能证明计划与侧车被冻结，不能授权正文复用。最终实现是 legacy_prefix 仅保留元数据；每批都从最新 packet 重新创作，再经过新的 receipt 和 ledger。这样没有改写 r1 历史，也避免同一 ID 在新 revision 中携带旧图片或旧表达。”

### 证据入口

- docs/portfolio-core-r2-pregeneration-handoff.md
- E:/skillchain-data/runs/portfolio-core-20260804-r1/plans/core.json
- E:/skillchain-data/runs/portfolio-core-20260804-r2/plan/core.json
- E:/skillchain-data/runs/portfolio-core-20260804-r2/pre-generation-manifest.json
- src/skillchain/synthesis/portfolio_core_authoring.py

---

## 47. accepted 必须晚于 durable ledger；编码错误应在此前 fail-closed

**状态：已验证并落地**

### 一句话问题

语料的 accepted 状态不是“草稿看起来写完了”就能设置：必须先完成非空规范化、持久 ledger 写入和哈希绑定；任何编码损坏都应在这之前 fail-closed。

### 背景与影响

r2 使用 create-only 的可恢复批处理。若先把 batch 标为 accepted、再尝试持久化 ledger，一次编码或写入故障就会留下“看起来已接受、实际上无法复核”的幽灵状态。这个风险在 PowerShell stdin 发布草稿时成为真实 incident：非 ASCII 在旧 root 中被降成 ?。

### 观察到的证据

已验证事实：

- 旧 create-only root E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts/dev-mini-001 保留原样；其非 ASCII 编码事故导致执行器得到 empty normalized final text。
- 执行器在 durable ledger 之前 fail-closed，没有把该批标为 accepted，也没有通过覆盖旧 root 来“修复”证据。
- 正确 UTF-8 输入发布在 sibling E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts-utf8-v2。真实 dev-mini-001 随后被 accepted 25 条，checkpoint 为 running/25。
- 该 accepted 批的 results SHA-256 为 e0a9a699286c95713516a5ec6f501e17f3d1f0055ae0a663abdb88aeeab80244，draft receipt 为 6e1121c2575ac33d1fbab5690f0c059c9ba197a50ac1932dbb91508b1187ba6a，checkpoint SHA-256 为 bc16a23d4238734ae12a7aabd3d149b267020e334d0217c5ae4aa504bba87492。

### 根因

根因有两个层次：

- PowerShell stdin 的编码边界没有被当作语料写入契约的一部分，非 ASCII 可在进程间变成 ?；
- 若 accepted 状态允许领先于 durable ledger，编码检测即使失败也会留下不可审计的终态。

编码正确性和状态顺序不是展示层问题，而是语料可信执行的共同前置条件。

### 考虑过的方案与取舍

1. 把旧 root 原地重写为 UTF-8：隐藏 incident 并破坏 create-only 证据，淘汰。
2. 忽略规范化为空，仍接受 batch：最快，但会把损坏文本写入训练输入，淘汰。
3. 先设置 accepted、后台补 ledger：容易产生幽灵 accepted 状态，淘汰。
4. 保留失败 root，在 sibling 以显式 UTF-8 重新发布；先校验、落 durable ledger、复验 hash，最后 accepted：可恢复且可审计，采用。

### 最终方案

execution path 现在把 accepted 放在 durable ledger 之后：先读取受绑定输入，验证每条最终文本的规范化结果非空，再 create-only 写入并复验 ledger、receipt 与 checkpoint，最后推进 accepted 状态。编码 incident 的旧 root 永久保留；修复只能在新 sibling root 中进行。这样中断时最多留下未接受证据，不会留下无法证明的 accepted batch。

### 如何验证

- tests/synthesis/test_portfolio_core_execution.py：13 passed，覆盖机械 execution、恢复、输入绑定和 fail-closed 路径。
- execution 与相关 r2 回归联合：28 passed、1 skipped；skip 是 Windows 无创建 symlink 权限时的预期分支。
- 真实 dev-mini-001 的 durable 结果具有 25 条、running/25 checkpoint、draft receipt、results SHA 和 checkpoint SHA 的互相绑定；没有把旧编码失败 root 重新标记为 accepted。

### 剩余限制

该机制已覆盖首批，但尚余 59 个 batch 和最终 200 条 owner 抽检。它保证的是本项目的本地持久顺序与可恢复证据，不替代跨机器的分布式事务，也不替代最终的语义质量审核。

### 30 秒回答

“我遇到过一次真实的 PowerShell stdin 编码问题：非 ASCII 草稿被降成问号。关键不是事后把文件改回来，而是系统必须保证 accepted 晚于 durable ledger。执行器先把规范化后的最终文本校验为空，立即 fail-closed，旧 create-only root 保留；我在新 UTF-8 sibling 重新发布，只有 ledger、receipt 和 checkpoint 都落盘复验后才接受 25 条。这样编码故障不会变成不可审计的训练数据。”

### 2 分钟回答

“这个问题把编码和事务顺序连在了一起。我们原本有 create-only root，但 PowerShell stdin 没有把编码作为严格契约处理，非 ASCII 在旧草稿树里被写成问号。如果系统在输出完成时就标 accepted，哪怕后面的 ledger 写入或规范化验证失败，也会出现一个无法复核的 accepted 事实。

我把状态机改成了 fail-closed 的顺序：先验证每条最终文本规范化后非空；然后以 create-only 方式持久化 ledger、receipt 和 checkpoint，并复验绑定哈希；最后才允许 accepted。事故发生时，执行器在 ledger 前拒绝了空文本，旧 root 原样保留。修复不是覆盖它，而是在 UTF-8 sibling 发布新输入；真实 dev-mini-001 随后以 25 条、running/25 和三组 SHA 被接受。聚焦 execution 测试 13 条全过，连同 r2 回归为 28 passed、1 个 Windows 权限 skip。这个顺序的价值是：失败可以恢复，但不会把损坏数据提前变成训练真相。”

### 证据入口

- docs/portfolio-core-r2-pregeneration-handoff.md
- src/skillchain/synthesis/portfolio_core_execution.py
- tests/synthesis/test_portfolio_core_execution.py
- E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts/dev-mini-001
- E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts-utf8-v2

---

## 48. 验证幂等不等于内存幂等：Core catalog 深验要复用证明并控制图像副本

**状态：已验证主修复；跨进程残留已进 backlog**

### 一句话问题

同一份 catalog 可以每次都通过字节复验，却仍因重复解码、全帧颜色转换与 Pillow native allocator 保留页而在同一长生命进程中累积到 `MemoryError`。

### 背景与影响

Core Gate 0 需要对 1,022 个图像资产、base/runtime 两份 catalog 以及三个远程 processor 权限链做 fail-closed 验证。首次 v3 launch 在重复装载这些对象时出现 `MemoryError`，如果只通过减少验证或全局缓存解决，会破坏权限与资产字节的信任边界。

### 观察到的证据

- Core loader 一次执行链曾重复扫描同一 base/runtime catalog 约 8 次，并在 fingerprint 路径无条件执行 `exif_transpose().convert("RGB")`。
- Python 引用离开作用域不代表 Pillow 的 native allocator 会立即将页还给操作系统；因此逻辑上“重读不改状态”不等于物理内存不增长。
- 修复后的 runtime catalog SHA-256 仍为 `fe01913e8d80f11d3c75c7abee4ae5644498c23d496134d96fbc6919322e9e01`，完成的 clean v3 canary 未再出现 `MemoryError`。

### 根因

验证层把“每个消费者都重新证明同一事实”当成了更强的安全性，而没有区分信任证明与解码对象的生命周期。同时，EXIF 纠正后再无条件转 RGB 制造了可避免的全帧副本。

### 考虑过的方案与取舍

1. 弱化 catalog SHA/pHash 或权限验证：能降低开销，但破坏 fail-closed 边界，淘汰。
2. 无界全局 cache：避免重读，但容易把跨 root 或已变更文件的 TOCTOU 状态混在一起，淘汰。
3. 在一次已验证调用内复用 exact typed object，同时减少图像副本：保留证明强度且寿命周期可控，采用。

### 最终方案

Core input loader 现在对 base/runtime catalog 各完整装载一次，复验 path/root/type 和 verified marker 后，将这对 exact verified objects 传给三个 processor 与 query binding；消费者不再重新解码整个 catalog。图像 fingerprint 路径改为 in-place EXIF transpose，只有非 RGB 图像才转换颜色模式。

### 如何验证

- asset catalog 与 remote-processing 聚焦回归通过，覆盖 exact preverified object 复用与篡改拒绝。
- catalog SHA/pHash 承诺未变，权限链仍由类型化 verified handle 传递。
- clean v3 25×5 canary 完成，5 个授权 shard 全部闭合，零调用 rerun 证明已完成字节可幂等复用。

### 剩余限制

monitor 和 matrix 仍是独立长生命进程，两者会各做一次完整复验并可保留较高 private bytes。若 core1500 或更高并发再触发内存水位问题，下一步是短生命 preflight 子进程：产生绑定 receipt 后退出回收 native heap，而不是放弃深验。

### 30 秒回答

“Core 的 1,022 张资产每次都能通过验证，但首次 launch 仍然出现 MemoryError。我发现逻辑幂等不等于内存幂等：同一 catalog 被重复解码约 8 次，Pillow 还会保留 native heap 页。我没有削弱 SHA/pHash 和权限验证，而是每个 catalog 只深验一次、向下游传 exact verified object，并消除多余的全帧 RGB 副本。后续 clean canary 完整通过。”

### 2 分钟回答

“这个问题很像泄漏，但根因是验证生命周期。Core loader 为了 fail-closed，在三个 processor 和 query binding 中多次重建同一份 1,022 资产 catalog；每次还要解码图像、做 EXIF 纠正和 RGB 转换。Python 对象释放后，Pillow native allocator 不一定立即还页，所以每次都字节正确，内存却不断抬高。

我把信任证明和对象复用分开：base/runtime 各深验一次，然后检查 root、type 和 verified marker，将 exact objects 传给所有消费者。同时 EXIF 改为 in-place，仅非 RGB 时转换。这样 catalog SHA/pHash 和权限边界都没变，但去掉了重复证明和像素副本。clean 25×5 canary 最终闭合，零调用 rerun 也保持全部字节不变。剩下的跨进程 native heap 问题记入 backlog，只在 core1500 再触发时改短生命 preflight 子进程。”

### 证据入口

- `src/skillchain/data/asset_catalog.py`
- `src/skillchain/data/portfolio_remote_processing.py`
- `src/skillchain/evaluation/portfolio_core_inputs.py`
- `tests/data/test_portfolio_remote_processing.py`
- `tests/evaluation/test_portfolio_core_launch.py`
- `runs/portfolio/core-gate0/canary-evidence-v3-clean/final-receipt.json`

---

## 49. 跨品类搭配不能伪装成同类视觉相似：需要 query-independent evidence graph

**状态：工具分支与真实 smoke 完成；Core 重锁/模型增益待验证**

### 一句话问题

同类视觉相似度不能回答“给这条裙子配什么鞋或包”；若直接从任意 catalog 候选中返回结果，card 看似完整，证据语义却是假的。

### 背景与影响

Core Gate 0 的 Style card 违规主要不是 renderer 漏字段，而是跨品类请求没有 grounded candidate。原工具只有 same-category alternative：它适合找相似连衣裙，却没有证据支持从裙子跳到鞋、包或首饰。用 prompt 或 S3 Body 强迫 Assistant 生成搭配卡，只会把缺失检索证据掩盖成自然语言。

### 观察到的证据

- Core selection 中有 130 个 FashionIQ `dress` anchor；当前候选池从同一受绑定 Core catalog 的 ABO 商品中产生。
- Codex 用 ABO listing metadata 和精确图片的可见属性复核并保留了 9 个女性候选：3 双鞋、2 个包、4 件首饰。两双男鞋被明确排除；另有 1 条女性项链因与 `test_frozen` query asset 重叠而排除，防止候选角色与冻结评测角色污染。
- 冻结的 deterministic palette rule 产生 872 条 exact anchor→candidate edge，其中 `footwear=290 / bag=260 / jewelry=322`；graph SHA-256 为 `1cecea70ba7f2c5a992d1fb773bba8d04043ee84e8dec8c2e362d70996b8258a`。图不携带 query、split、treatment 或 score 字段。
- edge 的证据是商品类别、颜色与可见款式 facet 以及 Portfolio curated rule，不是 outfit 共现、共购、点击或搭配金标。
- `dev_mini`-only 的真实 tool-runtime smoke 在零 provider、零 network 下通过：generic coordination 覆盖鞋/包/首饰三个 family；黑色低跟鞋命中 exact 候选，白色低跟鞋和明确否定包均返回 0；same-category mode 保持原路径；“鞋 + 外套”因存在未支持 family 而拒绝 partial DTO、整体返回 0。

### 根因

原设计把“Style”视为单一检索问题，但 same-category similarity 与 cross-category coordination 的关系类型不同：前者可由视觉近邻支持，后者至少需要类别兼容与搭配依据。共用一个无 mode 的候选路径，会让 source、evidence 和回答措辞同时发生语义漂移。

### 考虑过的方案与取舍

1. 沿用视觉相似度，跨类别取 top-k：实现最短，但距离不可比较，也没有搭配证据，淘汰。
2. 只改 Skill 文本，让模型自行“推荐”：能生成流畅答案，却会制造无 provenance 的 card，淘汰。
3. 立即接入 Polyvore 原生 outfit graph：证据更强，但需要新的 source lock、跨数据集实体映射、泄漏审计与更大实现范围，作为 P2 触发式后续。
4. 用受绑定 Core 商品构建 query-independent exact graph：先关闭鞋/包/首饰的真实候选缺口，同时把 curated rule 与原生共现的声明边界写进 contract，采用。

### 最终方案

工具保留两个显式分支。same-category 请求继续走原视觉相似路径；cross-category 请求只读取 exact anchor 的 coordination edges，并对用户请求的 `footwear / bag / jewelry` 做硬过滤。图由 Core selection、runtime catalog、ABO metadata、精确图片和 Codex 复核 seed 确定性构建；neutral candidate 可用于 dress anchor，颜色候选只有在 anchor 本地图像 palette 相容时才建边，男性候选始终排除。运行时对颜色、feature、否定约束以及请求中的全部 family 做 fail-closed 检查；任一明确请求 family 不受支持或无法满足时整体返回 0，不发布看似成功的 partial DTO，也不退化为任意视觉近邻。

公共结果只表达可公开的 category/facet/evidence handle；内部 product ID、路径、图片 SHA 与 graph identity 不进入 provider 可见 DTO。cross-category 结果还移除了没有标定语义的伪“相关度”，避免把 deterministic rule 的排序包装成相似度概率。因为 graph 是新的 runtime source，下一次 Core 执行必须生成新 source receipt 并重锁 TaskSpec/ToolSpec、Bank、runtime 与 launch，历史 lock 不原地修改。

### 如何验证

已验证的数据事实是 9 个女性候选、2 个男性排除、1 个 `test_frozen` 重叠女性项链排除、130 个 dress anchor 与 872 条 query-independent edge；graph SHA 和三类 edge 计数如上。真实 `dev_mini` tool-runtime smoke 已在零 provider、零 network 下验证：同类路径不变、generic 三 family 覆盖、exact 颜色/feature 命中、颜色不匹配、否定约束，以及含 unsupported family 时全请求 fail closed。`val/test_frozen` 没有用于挑图、调规则或判断增益。新 Core source receipt/relock 和 provider 端 Assistant/Judge 实验尚未完成，因此本条目不声称端到端模型增益。

### 剩余限制

当前候选类别只有鞋、包和首饰，且搭配关系来自 Codex 复核属性与确定性 palette rule；它能诚实支持“有依据的 curated recommendation”，不能支持“来自真实 outfit 行为的 compatibility”声明。Polyvore 原生 outfit co-occurrence 只有在当前覆盖或偏好指标成为瓶颈、并能先冻结跨数据集映射与防泄漏协议时再启动，不阻塞本轮 Core。

### 30 秒回答

“我发现 Style 的 card 缺失不是提示词问题：同类视觉相似度不能证明一条裙子该配哪双鞋。于是我把工具拆成不互相污染的两条路径。同类检索保持原逻辑；跨品类只读 query-independent evidence graph。我用 ABO metadata 和精确图片保留了 9 个女性鞋包首饰候选，排除两双男鞋和 1 个与冻结测试资产重叠的项链，再用冻结配色规则为 130 个 dress anchor 生成 872 条 exact edge。dev-only 的零 provider、零 network smoke 已通过；它明确叫 curated coordination，不冒充 outfit 共现，也不声称端到端模型增益。”

### 2 分钟回答

“最初 Style 工具只有 same-category alternative。面对‘给这条裙子配鞋’时，如果继续取视觉 top-k，距离跨类别不可比较；如果只改 S3 prompt，模型又会生成没有真实候选和 provenance 的漂亮答案。因此我把关系类型建模为两个分支：same-category 保持原视觉路径，cross-category 必须命中 exact anchor 的 coordination graph，并按用户请求类别硬过滤。

数据上我没有把 query 或 split 字段写进图。构建器只接受受绑定的 selection、catalog、ABO listing metadata、精确图片和 Codex 复核 seed，还显式禁止 query_id、split、config 和 score。我保留了 3 双女鞋、2 个包和 4 件首饰，排除两双男鞋；为防止跨角色污染，还剔除了 1 条与 `test_frozen` query asset 重叠的女性项链。neutral candidate 通用，颜色候选只有与 anchor 的确定性本地图像 palette 相容才建边，最终是 130 个 dress anchor、872 条 edge，SHA-256 为 `1cecea70ba7f2c5a992d1fb773bba8d04043ee84e8dec8c2e362d70996b8258a`。证据名称明确为 Portfolio curated coordination rule，所以我不声称它来自真实穿搭共现。

工程上，cross 分支对颜色、feature、否定和请求中的全部 family 做 fail-closed；没有 exact edge 或混入外套等不支持类别时整体返回 0，不给 partial DTO。公共 DTO 去掉内部 ID、路径、SHA 和跨品类伪“相关度”。dev-only 的真实 tool-runtime smoke 在零 provider、零 network 下覆盖了 generic 三 family、黑色低跟鞋命中、白色低跟鞋/否定包不命中、same mode 不回归和混合 family 全拒绝。这个 source set 改动意味着旧 runtime lock 不能复用；正式 Core 前仍必须新建 source receipt 并重锁 Bank/runtime/launch，再运行 provider 实验。Polyvore outfit graph 证据更强，但需要跨数据集实体映射和新的泄漏审计，我把它放成指标遇到瓶颈后才触发的 P2，而不是阻塞当前闭环。”

### 证据入口

- `scripts/build_portfolio_style_coordination_graph.py`
- `specs/evaluation/portfolio-style-coordination-candidates-v1.json`
- `src/skillchain/tools/portfolio_runtime.py`
- `src/skillchain/tools/contracts.py`
- `src/skillchain/runners/assistant.py`
- `src/skillchain/evaluation/portfolio_core_runtime_sources.py`
- `tests/evaluation/test_build_portfolio_style_coordination_graph.py`
- `tests/tools/test_portfolio_runtime_style.py`
- `tests/runners/test_portfolio_assistant_projection.py`

---

## 50. JSON Schema 能收紧结构契约，但不能让 reasoning-only 空正文变得不可能

**状态：canary 已验证；phase60/全量未执行**

### 一句话问题

Qwen3.8-Max 的 JSON Object 运行出现不可解析终态；改用 strict JSON Schema 后固定 12 条 canary 最终全部解析，但仍有三次首试失败，其中两次只有 reasoning、最终正文为空，说明结构契约与单次调用可靠性是两个问题。

### 背景与影响

Feedback 模型永久切换为 Qwen3.8-Max 后，原 Round 3 JSON Object root 在 14 次调用后只有 10/12 条解析成功，状态为 `stopped_nonparsed`。直接覆盖旧 root、复用其中输出或无限重试，都会破坏实验身份、失败证据和成本边界；同时，把 JSON Schema 当成“每次一定有正文”的保证也会掩盖真实 provider 故障。

### 观察到的证据

- 旧 JSON Object root 保持只读；新 Schema root 导入旧输出数为 0，并使用独立的 authorization/control/launch/run 身份。
- 新请求使用 `response_format.type=json_schema`、`strict=true` 和冻结的视觉 Feedback schema；固定 canary 为 12 条，provider 调用上限为 15 次，即 12 次首试加全局最多 3 次 retry。
- 实际共调用 15 次，三次首试均记为 `invalid_feedback_json`。其中两次 `finish_reason=stop`，reasoning 分别有 3,672/3,036 bytes，但 final content 为 0 bytes；另一次有非空正文但未通过本地严格解析。三次 retry 后最终 `parsed_count=12`、`error_count=0`、`retry_count=3`。
- create-only reservation、provider-attempt、retry-claim 与 bound artifact 均先落盘再推进状态；本轮新增实际费用为 CNY 1.944600。

### 根因

已验证的实现问题是旧 Round 3 专用路径仍发送 JSON Object，没有使用项目已有的严格 JSON Schema 契约。已验证的运行现象是：即使请求已切成 Schema，Qwen3.8-Max 在开启 thinking 时仍可能返回 reasoning 而没有 final content，或返回未通过本地契约的正文。其 provider 内部原因没有外部证据，因此只能推断为模型/服务的瞬态输出行为，不能归因成“Schema 无效”或本地解析器吞掉了正文。

### 考虑过的方案与取舍

1. 原地修改 JSON Object root 或导入 10 条成功输出：成本更低，但会混合传输契约和历史身份，拒绝。
2. 关闭 thinking 或放宽本地 parser：可能减少失败，却同时改变已冻结模型契约或接受不合规内容，未采用。
3. 改为 strict JSON Schema，并在新 identity 下做固定 12 条 canary：能单独验证传输修复；代价是必须重建治理绑定并支付新调用成本，采用。
4. 无界重试：能提高表面成功率，但不可审计且会挑选幸运结果；改为 create-only 全局 retry 额度 3。

### 最终方案

新增 forward-only Schema transport、result/cache 版本和独立 Schema Round 3 CLI；旧 JSON Object artifacts 字节不变。新 authorization 绑定模型/source/pricing/role 治理版本，选择集固定 12 条，先预留 15 个调用槽和预算；每次调用、失败与 retry claim 都使用 create-only ledger，且 retry 是全局上限 3，不是每条各重试 3 次。canary 完成即停止，phase60 必须获得新的 owner 批准。

### 如何验证

聚焦测试覆盖 SDK 实际 wire 中 `json_schema.strict/schema` 的位置、旧结果反序列化、零调用 preflight、预算预留、并发与全局 retry 竞争、崩溃恢复和 create-only 冲突。真实 canary 的 run receipt 进一步证明：12 条最终全部 parsed、历史输出导入为 0、三张 retry claim 与 15 份 provider attempt 对应、费用为 CNY 1.944600。

### 剩余限制

这只是 12 条 canary，不证明 60/120/240 阶段也会保持同样成功率，更不证明 Feedback 质量或 S1 指标提升。三次 retry 额度已全部用完；若扩大阶段仍出现 reasoning-only 或 schema-invalid 输出，必须在新批准和新预算下诚实停止或另行诊断，不能把重试成功写成零失败。

### 30 秒回答

“Qwen3.8-Max 的旧 JSON Object canary 最终只有 10/12 可解析。我没有覆盖旧产物，而是用 forward-only identity 切到 strict JSON Schema，固定跑 12 条，并把 retry 做成 create-only ledger 上的全局 3 次额度。结果最终 12/12 parsed，新增费用 CNY 1.9446；但中间确实用了三次 retry，其中两次只有 reasoning、final content 为空。这说明 Schema 修好了结构契约，却不能保证 provider 每次都给有效正文，所以我保留失败证据，也没有把 canary 外推成全量结论。”

### 2 分钟回答

“这次问题有两层。第一层是确定的工程缺陷：通用适配器已经支持 JSON Schema，但旧 Round 3 专用执行路径仍发送 JSON Object，因此 Qwen3.8-Max 跑到 14 次调用后只得到 10/12 条可解析结果。第二层是 provider 可靠性：把 wire 改成 strict JSON Schema 后，也不能假设单次调用必然成功。

我没有修补或复用旧 root，因为那会把两个传输契约混成同一个实验身份。我建立了新的 authorization/control/launch/run 链，冻结 schema、模型、source、price 和 role；历史输出导入数必须为 0。执行范围固定为 12 条 canary，最多预留 15 次调用。每个 reservation、provider attempt 和 retry claim 都先 create-only 落盘，retry 是所有样本共享的 3 次上限，因此并发或崩溃都不能偷偷超预算。

真实运行最终是 12/12 parsed、15 次调用、三次 retry、新增 CNY 1.9446。两次失败的证据很关键：`finish_reason=stop`，reasoning 非空，但 final content 是 0 bytes；另一次正文非空却没通过严格 parser。三次重试都成功了，但我不会说‘Schema 消除了失败’，准确结论是它关闭了旧结构契约缺口，而受控重试吸收了少量瞬态故障。由于样本只有 12 条、retry 额度也已用完，phase60 和全量仍需新的批准，且不能从这次 canary 推断 Feedback 质量或最终算法增益。”

### 证据入口

- `src/skillchain/evaluation/portfolio_s1_feedback_recovery_v1.py`
- `src/skillchain/evaluation/portfolio_s1_feedback_round3_schema_v1.py`
- `src/skillchain/evaluation/portfolio_s1_qwen_governance.py`
- `scripts/run_portfolio_s1_feedback_round3_schema_v1.py`
- `tests/evaluation/test_portfolio_s1_feedback_round3_schema_v1.py`
- `tests/evaluation/test_portfolio_s1_qwen_round3_schema_governance.py`
- `runs/portfolio/core-s1/s1-feedback-round3-qwen38-v1/run-round3-v1.json`
- `runs/portfolio/core-s1/s1-feedback-round3-qwen38-schema-v1/run-round3-schema-v2.json`

---

## 51. Sparse patch 不等于因果隔离：需要 capability-local paired screen 与 byte-exact 回滚

**状态：已验证；Qwen3.7 S1 两批十轮负结果已冻结，未进入 S2**

### 一句话问题

把 Creator 限制成 sparse patch 只能减少改动面，不能保证模型执行只受目标 Skill 影响；
必须用配对 development screen 找出 Static-success 回退，并把失败 capability 逐字节恢复。

### 背景与影响

历史 R0 对六能力整 Bank 重写，replay200 的总体 macro 增加 `3.5021pp`，却让 Encyclopedia
下降 `5pp`，超过冻结的单能力底线。随后新 Qwen3.7 Static lineage 的 R1–R10 采用
parent-bound `inherit|patch`、policy-labeled Feedback 和 byte-exact inherit。这个设计关闭了“改一个能力顺手改其他能力”的
文本污染，但没有消除路由采样、工具选择、回答 repair 和高显著度 fallback 指令之间的
运行时交互。

### 观察到的证据

- R2 只 patch Multi，target replay 从 `8→20`，tool-contract failure 总数从 `15→5`；但
  `r2-core-0783` 跳过工具并编造 item/card handle，形成 1 条 Static-success 回退和 3 个
  新 contract occurrence，仍必须回滚。
- R3 的 DTO-copy 假设没有被 Assistant 检验。Creator 外层成功，但 authored-content guard
  因同一句中的 `result` 与斜杠组合拒绝 sparse proposal，candidate 为 null、Assistant 0-call。
- R4 的 Document literal-line copy 为 `0→0` 且引入 3 个新 contract occurrence，没有收益。
- R5 的 Style evidence copy 从 `7→26`，但有 2 条回退：一条漏掉精确 fallback marker，
  另一条跳过工具并编造候选；另有 4 个新 contract occurrence，因此同样回滚。
- 新授权的 R6/R7 分别让 Multi `8→18`、Style `7→22`，但仍各有 2 条 Static-success
  回退和 4 个新增 tool-contract occurrence，复现了“多数 grounding 改善、少数跳过工具”的模式。
- R8 的 Recipe 是第二批最可信的方向性信号：`3→10`、0 条 Static-success 回退，tool/fallback
  总错误也净下降；但 6 条原本已失败的 query 出现 7 个新 reason×query 事件，按预注册的
  零新增 contract screen 仍须回滚。R9 Exact `17→17`，R10 Document `0→0`，均无可接受收益。
- 九个实际 replay round 的 outer Assistant 调用均完整，无容量或服务错误；容量不是这些
  负结果的解释。

### 根因

已验证事实是：R2/R5 的目标能力都有明显净改善，但模型仍偶发跳过必需工具、编造 handle，
或在 empty-result 分支漏掉精确 fallback marker。自然语言 sparse instruction 能提高大多数
DTO/evidence closure，却没有把正向分支、工具先行和 fallback 变成确定性互斥状态机。
R3 又暴露了第二个工程问题：guard 能 fail closed，但没有把精确拒绝原因写入 stage decision，
使一个词法假阳性看起来像算法候选失败。第二批开始前已修复该局部 guard 并落盘脱敏
reason code；R6 真正执行了 DTO-copy 假设，仍复现少量跳过工具，说明剩余主因在运行时
行为约束，而不再是候选校验器。

### 考虑过的方案与取舍

1. 看净分选择 R2 Exact 的 `+3`：会隐藏 paired regression，拒绝。
2. 冻结或重放 R2 的赢家 route 后再评 R3：会在看到结果后改变比较口径，拒绝。
3. 降低 non-regression screen 或重跑直到路由更幸运：属于挑结果，拒绝。
4. capability-local screen + byte-exact compose：保留可归因 patch，失败能力精确恢复；采用。
5. 第一批用完后无授权追加第六轮：拒绝；获得独立的新五轮授权后，预注册 R6–R10、使用
   新 round identity 和独立 root 执行，R10 后不再追加 R11。

### 最终方案

S1 smoke 只检查 provider/schema/oracle 等运行覆盖；候选选择使用固定 replay200 的配对
capability screen。任一 patch 若成功数下降、出现 Static-success 回退或新增 contract reason，
只回滚对应 capability；组合 Bank 重新过 replay 后才有资格访问一次 body gate。两批十轮
最终均为 `retained_patch_capabilities=[]` 或无合法 candidate，selected Bank 是 Static
`da5cfe1f93f2cb57b389c97738034cf10cab931dcc30e145acc6d8873ff1348a`。
因为用户条件是“S1 不回滚才进入 S2”，本轮没有启动 S2，也不携带任何被拒绝的 Body。

### 如何验证

- 回归覆盖 policy projection、sparse compile、oracle coverage、screened Bank resume/hash、
  局部回滚、冻结 bootstrap parity、模型 lineage 和并发 exact wire。
- R1–R10 的 screen/decision 均为 canonical、SHA 绑定产物；`body_gate75`、S2 与 `test300`
  保持 0-call。
- 第二批新增 3,588 次 Qwen3.7 provider call、CNY `1.8057772`，0 次新 Feedback call；
  累计可追踪 provider 记录为 11,411 calls、CNY `5.7260350`。废弃 Static v1 另有一条
  无法恢复用量的 orphan；10 次 Creator 的人民币 cost basis 不可得。

### 剩余限制

Static 与候选是独立模型采样，受保护 capability 的波动不能归因给目标 patch。R2/R5、
R6/R7 的目标净增益很强，R8 也没有 Static-success 回退，但预注册的零回归/零新增 contract
保护真实触发，不能事后放宽。下一次若另行立项，优先把工具调用、DTO/card/evidence 输出
做成版本化的确定性 compiler/runtime contract，而不是继续堆自然语言 prompt。Core test
仍未访问，因此本条只报告 development 负结果。

### 30 秒回答

“我把 S1 收紧成单能力 sparse patch 后，两批十轮里多次看到明显净增益：Multi 和 Style
都能提升十几条，Recipe 也做到 `3→10` 且没有把 Static 成功样本打坏。但候选仍会偶发
跳过工具或迁移 contract 错误。我没有用净增益覆盖风险，十轮都由 paired screen
byte-exact 回滚；body gate 和 S2 都没碰，这证明增益门和停止规则是真实工作的。”

### 2 分钟回答

“历史 whole-bank S1 总体上升却伤了 Encyclopedia，所以我把 Creator 改成 parent-bound
单能力 patch：Description 冻结，其他五项字节继承，只有 policy-compatible Feedback 能进
Creator。第一批里 R2 Multi 从 8 到 20、R5 Style 从 7 到 26，但都有少量成功样本回退。
第二批先修掉 DTO guard 假阳性，再预注册五个独立 round；R6/R7 又复现了大幅净改善和
少量跳工具，R8 Recipe 达到 3 到 10 且零成功回退，但把 6 条既有失败迁移成 7 个新 contract
事件。我的门要求任何 Static-success 回退或新 contract occurrence 都回滚，所以没有用净分
掩盖风险，也没有在结果出来后降门。

最终 R1–R10 全部 selected Static，body gate、S2、val/test 全部 0-call。工程层面，guard
现在会给出稳定、脱敏的拒绝原因；算法层面，证据已经指向自然语言 Skill 能改善 grounding，
却不能稳定保证工具调用和结构化闭合。下一步若另行立项，应转向确定性 compiler/runtime
contract，而不是继续追加 prompt 轮次。这个案例体现的是如何区分方向性增益、发布安全、
执行随机性和工程故障，并在预算与停止边界内诚实交付负结果。”

### 证据入口

- `src/skillchain/evaluation/core_fast/engine.py`
- `src/skillchain/evolution/s1_gcs_gate.py`
- `src/skillchain/evolution/s1_sparse_patch.py`
- `specs/core-experiment-fast-v1.json`
- `tests/evaluation/test_core_fast.py`
- `E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v2`
- `E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v3`

---

## 52. 确定性 runtime 之后，S1 必须拥有真实 treatment surface，Gate 也必须按 treatment reach 归因

**状态：已验证；首个 typed semantic-policy S1 已通过 body gate，停在 S2 前**

### 一句话问题

把 tool-first、DTO/card/evidence closure 和 fallback 全部编译化以后，普通 Skill Body 不再影响运行；
S1 必须编辑 runtime 实际消费的 typed policy，且 sparse Gate 不能让未处理能力的独立路由采样替目标 patch 背锅。

### 背景与影响

历史 R1–R10 表明自然语言 Body 能改善多数 grounding，却会偶发跳工具或破坏结构闭合。项目因此把
机械规则移入 deterministic runtime。但第一版实现直接在工具成功后编译 final，S1 又冻结全部
Description，导致候选 Body 成为无效变量；同时默认 spec 仍绑定旧 runtime Static，完整 48 条
Feedback 也缺少 Creator 前终止门。直接运行会产生无法解释的 S1 比较。

### 观察到的证据

- v2 fresh Static 完成 800/800，但 Recipe evidence selector 使用抽象词过滤 literal source，目标
  replay 从 `26→0`，被 capability screen 全量回滚。
- v3 把长 source 拆成 material statements，并为每句重复 exact handle；fresh Static 为
  `674/800`、0 hard error，Recipe 在新基线中已无可优化空间。
- v4 新增 Document typed `ocr_extraction_plan`。parent 使用 `all-lines`，候选只能切到
  `literal-material-spans`；同一公开 OCR DTO 下，typed policy 改变编译输出，而普通 Body prose 不会。
- v4 fresh Static 为 800/800 outer success、`676/800` GCS、0 hard error。有效 R1 的 Document
  replay 为 `7/8→8/8`，目标能力无回退、无新增 contract error。
- 第一次 whole replay 中未修改 Exact 因独立路由采样从 `25→21`，曾错误触发 capability floor；
  这些能力的 Bank 字节完全相同，故不是 Document treatment 的结果。
- 修正后的 Gate 只在 parent/candidate 都路由到目标能力的 treatment-reached 行采用 candidate；
  replay macro `+2.0833pp`，body_gate75 的 Document `3/4→4/4`、macro `+4.1667pp`、
  hard-error delta `0pp`、bootstrap 95% CI `[0, 8.3333]pp`，最终 `accepted=true`。

### 根因

第一个根因是 treatment surface 消失：compiler 独占最终回答后，Creator 修改的 prose 不再被读取。
第二个根因是证据闭合粒度错误：一个长 source/OCR 行只带一次 handle，但 scorer 按标点拆成多条
material statement。第三个根因是 Gate 把冻结 Description 的两次独立 route sampling 当作 Body
差异，违反 sparse S1 的因果边界。

### 考虑过的方案与取舍

1. 恢复 action LLM 读取整段 Body：重新引入 tool/DTO/fallback 随机失效，拒绝。
2. 继续让 Creator 改 runtime-owned prose：变量不会影响输出，拒绝。
3. 放宽 `−3pp` 或忽略目标真实回退：会降低安全门，拒绝。
4. 让 S1 只编辑 typed semantic policy，并由 compiler 消费；采用。
5. 对 byte-exact inherit 能力和目标错路由行 alias parent observation，目标 treatment-reached 行仍
   保留真实 candidate；采用，所有数值门槛保持不变。

### 最终方案

runtime v4 固定工具、参数、结构、handle 与 fallback，只开放版本化 typed semantic policy。
Document R1 把 `ocr_extraction_plan` 从 `all-lines` 改为 `literal-material-spans`，去除会制造无 handle
statement 的终止标点并跳过 noise-only glyph。Feedback 使用 `canary6 + remaining42`；完整 48 条
必须满足 completeness、service-error 与 parse/schema-error 门才允许 Creator。accepted Bank 为
`efc7cbb3a25daf22aee45b943b0ec4cdca42f3e49644bd53fafaad2a85c3aaef`，按规则未启动 S2/test。

### 如何验证

- treatment-sensitivity 测试证明 typed policy 改变输出、普通 Body prose 不改变输出。
- 完整 48 Feedback terminal Gate 测试证明 remaining42 出错时 Creator 为 0-call。
- sparse Gate 测试证明 untreated capability 与目标 route mismatch 使用 parent，目标真实 candidate
  回退仍会触发门。
- 相关回归覆盖 deterministic runner、sparse compiler、Feedback parser/selector、Core Fast 与 live
  capacity；默认 `validate --inputs-only` 和完整 `validate` 均通过。
- accepted root 用原 canonical spec resume 时 call 文件数保持 `1246→1246`；未发现 S2、Judge 或
  test300 artifact。

### 剩余限制

body_gate75 只有 4 条 Document 样本，CI 下界恰为 `0`，因此这是 Portfolio Track 的首个真实正向
闭环，不是大样本泛化或论文正式结论。Gate 的 causal alias 依赖 S1 Description 冻结；若 S2 修改
Description，必须使用 S2 自己的 route gate，不能套用该口径。Creator 人民币成本不可得；首次
48 Feedback 中两条空 `summary_note` 需要定向重跑，另有一次目录复制错误保留为执行异常。

### 30 秒回答

“我先把工具调用和结构闭合做成确定性 runtime，但马上发现这会让原 S1 Body 变成无效变量。
我把 S1 改成 typed semantic policy，并用 fresh Static 对称重跑。Document 的 literal-span policy
在 replay 从 7/8 到 8/8，在冻结 body gate 从 3/4 到 4/4，macro 提升 4.17pp，最终被接受。
同时我修正了 Gate：未修改能力的独立路由噪声不再归因给 Body patch，但目标真实回退仍 fail closed。”

### 2 分钟回答

“历史十轮表明 prompt 能改善 grounding，但 tool-first、DTO 和 fallback 偶发失效，所以我把这些
机械规则迁到 deterministic compiler。合并后我没有直接跑实验，因为发现 S1 仍只能改 Body，
而 runtime 已不读 Body，候选实际上无法生效。我定义了 compiler 消费的 typed semantic policy，
并在新 contract 下重跑 800 条 Static。

第一版 Recipe selector 过度过滤 source，26 条成功全部丢失，screen 正确回滚。分析后发现 source
handle 只放在长行开头，而 scorer 会按标点拆句，所以我先修 runtime-owned evidence closure。
新基线里 Recipe 已饱和，剩余可归因簇是 Document OCR 行的终止标点和噪声 glyph。我只开放一个
枚举字段，把 all-lines 改成 literal-material-spans；工具、DTO、heading、handle、fallback 全冻结。

候选在 Document replay 修复唯一失败，但第一次 whole gate 被未修改 Exact 的随机路由波动否决。
因为 Description 和 Exact Bank 都字节相同，那不是 S1 treatment。我将 Gate 改成只在 treatment
真正到达的目标行使用 candidate，其余使用 parent；没有降低 +2pp、CI、hard-error 或 capability
floor。最终 replay macro +2.08pp，body gate macro +4.17pp，CI 下界为 0，S1 正式接受。整个过程
没有访问 S2、Judge 或 test，所以现在得到的是一个可复核的 Portfolio 正向闭环和清晰的因果边界。”

### 证据入口

- `src/skillchain/runners/assistant_deterministic_contract.py`
- `src/skillchain/evolution/s1_sparse_patch.py`
- `src/skillchain/evaluation/core_fast/engine.py`
- `specs/core-experiment-fast-v1.json`
- `tests/runners/test_assistant_deterministic_contract.py`
- `tests/evaluation/test_core_fast.py`
- `docs/s1-experiment-log.html`
- `D:\athena\experiment-runs\portfolio-core-qwen37-deterministic-v4-20260813\s1-r1e-document-literal-span`

---

## 新条目模板

复制下面的模板，编号后放到索引和正文中。结论未被验证时必须标为“待验证”或“部分解决”。

```markdown
## NN. 能概括冲突或意外的标题

**状态：待验证 / 部分解决 / 已验证**

### 一句话问题

### 背景与影响

### 观察到的证据

### 根因

### 考虑过的方案与取舍

### 最终方案

### 如何验证

### 剩余限制

### 30 秒回答

### 2 分钟回答

### 证据入口
```
