# Portfolio Core r2 正文生成前交接

本文记录 r2 的生成前快照、可复核诊断以及截至本次更新的执行元数据。它不是语料正文，也不表示 1,500 条 core 的 plan_id + turns 已全部生成、已全部验收或已可作为完整训练集使用。

## 已验证事实

### 1. 不可变 lineage 与静态/动态边界

| 对象 | 实际路径 | 摘要 / SHA-256 |
| --- | --- | --- |
| r1 父计划 | E:/skillchain-data/runs/portfolio-core-20260804-r1/plans/core.json | 1,507,440 bytes；90c8d4bfad250f486d5111045914f70abb638fe61ef20ba23749f26afd451625 |
| r1 运行说明 | E:/skillchain-data/runs/portfolio-core-20260804-r1/portfolio-core-run.json | f67912cabce2dc3cd3aa8abed3e1b6f301357796a9710b802336b4a3bc8c355f |
| r2 静态 run root | E:/skillchain-data/runs/portfolio-core-20260804-r2 | 只含 plan、sidecar、authoring、audit、stage1 与根 manifest；没有 queries、results、turns 或已验收正文。 |
| r2 根 pre-generation manifest | E:/skillchain-data/runs/portfolio-core-20260804-r2/pre-generation-manifest.json | 2,382 bytes；bb4a563cfc2124484348039ad014cfefde73fb2763967965bb69f58ce46002a7；声明 supersedes_run=portfolio-core-20260804-r1、trusted_parent_r1_sha256=90c8…51625，且覆盖 15 个非根产物。 |
| r2 core plan | E:/skillchain-data/runs/portfolio-core-20260804-r2/plan/core.json | 1,516,024 bytes；b1e637a6801eefa4a66eab8c05ec5a03ded31e1693e0296deb46aee82602dc11；1,500 个文本空白的计划行。 |

设计上，静态 r2 root 是 create-only、继承绑定的冻结快照，作者/执行者不得把正文、checkpoint 或动态结果写入其中。动态作者任务必须位于其 sibling：

E:/skillchain-data/runs/portfolio-core-20260804-r2-author-jobs

该 sibling 目前有两个 opaque job 根，而不是 r2 静态 root 的子目录：

| job | 基础 batch | 当前状态 | 25 条 author packet SHA | work order SHA |
| --- | --- | --- | --- | --- |
| r2-author-8d843ca7e0037545b641a828 | dev-mini-001 | author-job checkpoint 仍为 issued；execution 已独立 accepted dev-mini-001..008 共 200 条，见下节 | 8e36289b151df6a290b6ea758bb15f2477aad05fc4e31f198980a669b2b5f50c | bd4d9df31583f9b8ffdb35500ab92db783324b86eab6b608b58ad7f394767750 |
| r2-author-1992a072e26a61eeb4d12b7b | core-009 | issued；无 staged/accepted batch | 8c9ff1a6971ec29ff4e6d42fb40da77edabf5c4005fc82f0afa33835bd4a9796 | ac3fcf70d13cca2f05b9d7145832219fda1b7828d9554276e8801b99818f017c |

它们分别有 23 与 13 个 opaque author asset；author-job 的 checkpoint 仍只是输入发放状态，不应与独立 execution runtime 的 accepted checkpoint 混为一谈。它们是机械流程的已发布输入，不等价于全部 60 批的对话文本均已生成。

### 1.1 dev-mini 200 条里程碑、25% execution 快照与编码事故

已验证的 dev-mini 200 条里程碑为：`dev-mini-001..008` 共 8 个 batch、200/1,500 条已由 r2 execution accepted。该里程碑的 checkpoint SHA-256 为 `237a16541a67536794af937d9c97a112708337fc50134668a85539b54070613a`；aggregate results SHA-256 为 `1d01c160f380fdfd911d725e4684e5885fb433e6cec94796d2dcfdef0a6ece91`。

这 200 条的聚合配额为 `Exact/Multi/Style/Ency/Doc/Recipe=35/35/35/35/30/30`，`boundary=40`；turn shape 为 `single/three=140/60`，interaction 为 `direct/clarification/correction/no-result/goal-change=140/30/10/10/10`。normalized 文本唯一数为 200，legacy overlap 为 0，UTF-8 校验有效。

25% 的当前 execution 快照为：`dev-mini-001..008` 加 `core-009..015` 共 15 个连续 batch、375/1,500 条已 accepted。checkpoint SHA-256 为 `8b64a16319fe569c87229bf74c6abe990393cec748233cec72fc5d70cb9cd7b5`；accepted ledger SHA-256 为 `23b238ca11e36d16bf937990a0e8cbfde63461a3096c5e81c3f601f4e2347427`。只读元数据复核确认 checkpoint 自哈希、canonical ledger、15 个 manifest/author-draft-receipt 的自哈希及 ledger 绑定均一致，accepted 目录数为 15，staging 为空。剩余 45 个 batch、1,125 条；执行从 `core-016` 继续。

一次 PowerShell stdin 编码事故已被 fail-closed 地截住：旧 create-only root E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts/dev-mini-001 将非 ASCII 内容降成 ?；执行器在 durable ledger 之前检测到 empty normalized final text 并拒绝接受。旧证据保留，不删除、不覆盖、也不把它视作 accepted。正确的 UTF-8 草稿发布在 sibling E:/skillchain-data/runs/portfolio-core-20260804-r2-author-drafts-utf8-v2，随后才进入上述 accepted 记录。

### 2. r2 输入绑定

| 输入 | 实际路径 | 摘要 / SHA-256 |
| --- | --- | --- |
| catalog v9 manifest | E:/skillchain-data/clean/portfolio-core-asset-catalog-v9/manifest.json | manifest 文件：b376dd4feb1ea5348937da156596c875a246f600df72ca7bd4b6aae233c90e57；逻辑 catalog_sha256：539dcc5885c09cb1412b0fe334f6dcf8334e5a0ff7dca8ee122c78309856adb6 |
| catalog v9 assets | E:/skillchain-data/clean/portfolio-core-asset-catalog-v9/assets.jsonl | 1,022 assets；a65a2b95ade62efcd52ae53b0b8111bc7f2da880a081734e2f0e213b492f14a3 |
| catalog v9 components | E:/skillchain-data/clean/portfolio-core-asset-catalog-v9/components.jsonl | 872 components；9e050001ee49fd36279b8e3a812e1c3e23e3e60c8a53083bcd6894062e89bf61 |
| capability assignments v9 | E:/skillchain-data/clean/portfolio-core-capability-assignments-v9.jsonl | 1,120 bindings；4dbb18e551ab9194416b7d966d57d8003d41f1dc8a98a363a6357e92e0986a81 |
| r2 selection record | E:/skillchain-data/clean/portfolio-core-r2-selection-manifest-v6.json | 25e8708e413cd4589449ffe6fa8f379b6db4268448b27eac4e9940012c996333；997 include、27 reserve。 |

r2 plan、realism、audit、S1 和 gate receipt 都绑定上述 catalog/assignment SHA，不能在生成期间改指向其它 catalog 或 assignments。

### 3. 计划分布、原子切分与零泄漏检查

r2 core 共有 1,500 行，分成 60 个恰好 25 行的 generator batch。最终 split、六 capability 与 boundary 数量如下；均来自 plan 加 final-splits sidecar 的只读复核。

| final split | 总数 | Exact | Multi | Style | Encyclopedia | Document | Recipe | boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_mini | 200 | 35 | 35 | 35 | 35 | 30 | 30 | 40 |
| opt_pool | 800 | 209 | 117 | 163 | 163 | 32 | 116 | 137 |
| val | 200 | 52 | 29 | 41 | 41 | 10 | 27 | 34 |
| test_frozen | 300 | 79 | 44 | 61 | 61 | 18 | 37 | 52 |
| 合计 | 1,500 | 375 | 225 | 300 | 300 | 90 | 210 | 263 |

其中 Exact=product.exact_match、Multi=product.multi_search、Style=product.style_recommendation、Encyclopedia=knowledge.visual_encyclopedia、Document=utility.document_reading、Recipe=utility.recipe_guidance。

实际切分 sidecar：

- E:/skillchain-data/runs/portfolio-core-20260804-r2/sidecars/final-splits.jsonl：34496b6a588b1dfe012b8ddbe82eed15f1c2f1ecdf916f603740b3af2b1f7615
- E:/skillchain-data/runs/portfolio-core-20260804-r2/sidecars/reuse.jsonl：1a200f8f0db89105e371df53c4b0f3f9cd4b6bca4ece480921889d498be9c94b
- E:/skillchain-data/runs/portfolio-core-20260804-r2/sidecars/split-constraints.json：21c64c2153cd2f57d50a9603ed833f25f3ed8bc1c143f77741fbf75b1d70431b

零泄漏在这里有明确而有限的含义：对 1,500 个计划行，以 query-connected-components-v1 聚合后，leakage_group_id、非空 boundary_group_id、template_family、generator_batch_id 的每个取值均只落在一个 final split；只读复核的跨 split 违规数为 0。计划实际引用 867 个 leakage_group_id，最大复用数为 3。这是 split/资产关联约束已满足，不等价于对尚不存在正文的语义质量或未来训练效果作出保证。

### 4. Realism、审计、S1 与 validation gate

| 产物 | 实际路径 | 摘要 / SHA-256 |
| --- | --- | --- |
| realism assignments | E:/skillchain-data/runs/portfolio-core-20260804-r2/authoring/realism.jsonl | 1,500 行；e5135ab2234936601f9f40b6f3c6d2319c6460c145467b2508fe3325916ad23d |
| realism manifest | E:/skillchain-data/runs/portfolio-core-20260804-r2/authoring/realism-manifest.json | 013a5a65bd7945727aa5e33da92ba78b5398579a9a25f67a234f7ca2cf122028 |
| prompt recipe inventory | E:/skillchain-data/runs/portfolio-core-20260804-r2/authoring/prompt-recipes.jsonl | 12 个 recipe；63618046f74ff5553b75f81fc07d82ce8bffe3d11cf8ed767a63c3a1850b255d |
| audit index / audit / manifest | E:/skillchain-data/runs/portfolio-core-20260804-r2/audit/index.jsonl；audit.json；manifest.json | 1733b23ebeaf5a0ca5169f3d6b99ef0c367a8ddb0c881b84992536a39f7d3adb；db1642811818d257d19f8de25e6a8e56a8f7351e00c80c5f02dc0cb14c7ca35e；7a96cce5e87efc373dfdd7ea5d3564afbb244cb0176bc98b8b1fc24819899774 |
| S1 index / audit / manifest | E:/skillchain-data/runs/portfolio-core-20260804-r2/stage1/index.jsonl；audit.json；manifest.json | eeb4fc8e57788ded6393d3575c442e429c2aa03c8dd083ab2fd850da78b5c593；3cade2c1fbf4858b27144e335771cf4d5a49d2aef0c75bfdff7a815395a8f299；9ec589bc7dd2ab5872cc83881a10ed37bed267cca2347b4192bde6a5f17505bf |
| text-free validation gate receipt | E:/skillchain-data/runs/portfolio-core-20260804-r2-validation-gates.json | 文件 SHA：2f099f5a4efa9589f6948a1fec1419fd9024acdd4069f01195b6c613d2513f11；receipt 内 bundle_payload_sha256：c02af360d9530771b581fee498a58e11616d43622f719a1094c62138fc1b5571。 |

realism assignment 的全量配额为：single_turn=1,050、three_turn=450；direct_request=1,050、underspecified_clarification=225、constraint_correction=75、goal_change_or_multi_query=75、no_result_relaxation=75；S0/S1/S2/S3=225/675/450/150；六类用户风格为 75/525/375/75/375/75；ambiguity 为 none=1,237、clarification_required=105、resolved_near_boundary=158。

200 条 cluster-aware audit 覆盖全部 60 个 generator batch 和 12 个 recipe。实际样本的 interaction 为 128/20/8/23/21（依次为 direct、underspecified、correction、goal-change、no-result），expression 为 S0/S1/S2/S3=29/88/58/25，mandatory 选中 61 条；稀有 strata 中 document boundary=5、counterfactual=19、goal-change=23、no-result=21。它是生成后的 owner 抽检候选清单，不是已生成正文。

S1 静态选择来自 800 条 opt_pool 候选，已选 240 个唯一 component；六 capability 配额为 Exact/Multi/Style/Encyclopedia/Document/Recipe=40/40/40/40/32/48。它冻结的是 S1 的平衡选择输入，不代表 S1 已经消费真实 runner trace。

gate receipt 的 seed 为 20260805，且只使用 val 中八个完整的 25-query 原子组。其实际划分为：

| gate | 行数 | generator batch | Exact / Multi / Style / Encyclopedia / Document / Recipe | boundary |
| --- | ---: | --- | --- | ---: |
| route_gate | 75 | core-044、core-046、core-048 | 20 / 11 / 15 / 15 / 4 / 10 | 8 |
| body_gate | 75 | core-041、core-045、core-047 | 19 / 10 / 16 / 16 / 4 / 10 | 23 |
| shadow_val | 50 | core-042、core-043 | 13 / 8 / 10 / 10 / 2 / 7 | 3 |

receipt 也绑定 r2 plan、catalog、assignments、realism 与 val-interactions 的 SHA；门内五类 interaction 都满足冻结的最小值。这里的原子性来自同一 query-connected-components-v1 规则，不能被后续正文作者重新分割。

### 5. r1 dev 前缀与旧 dev 正文不可复用

下列结论来自只读对比，而不是重新采样：

- r1 core plan 的前 200 条与 r2 的前 200 条，在 plan_id、顺序、asset_id、leakage_group_id、canonical capability、intent 上均为 200/200 对齐；image_path 为 170/200 对齐，改变的 30 条均为 Document Reading 的合法 carry-forward rebind。
- 已接受的旧 dev 数据 D:/athena/ECommerceSkillChain/data/queries/queries.jsonl 与 r2 前 200 个计划槽位的 image_path 对齐为 0/200。
- 旧 dev 正文的 turn shape 为 single=188、three=12；r2 realism 对前 200 条要求 single=140、three=60，只有 142/200 的 turn shape 与新 realism 卡一致。

因此 legacy_prefix 只保留 r1 的 plan ID、顺序和元数据 lineage；不得把旧 dev 的 text 或 turns 作为 r2 的正文复用。八个 dev-mini batch 已按 r2 packet 从头创作并独立 accepted；其余 52 个 batch 仍必须由主会话按 r2 packet 从头创作。

### 6. Source-majority 诊断

把 catalog 中的 source_dataset 与计划行连接，并以每个 source 的多数 canonical capability 作为一个离线猜测器，得到：

| 指标 | r1 | r2 |
| --- | ---: | ---: |
| source-majority accuracy | 98.9333% | 95.6000% |
| 六 capability source-majority macro-F1 | 0.992014 | 0.965116 |

数字下降说明 r2 比 r1 弱化了“仅凭 source 猜 capability”的捷径，但该诊断仍然偏高。source_dataset 不是 author packet 或训练运行时的模型可见输入；这两个指标不能替代 final-split 零泄漏检查，也不能单独证明真实用户分布、正文自然度或最终训练增益。

### 7. 当前测试与调用账目

以下是本阶段已有的聚焦测试记录：

- tests/synthesis/test_portfolio_core_authoring.py：3 passed。
- audit 与 publication 相关测试：7 passed、1 skipped；跳过项是 Windows 无创建 symlink 权限时的预期分支。
- tests/synthesis/test_portfolio_core_gates.py：4 passed。
- r2 split/bridge/gate 的早期相关回归集合：18 passed。
- tests/synthesis/test_portfolio_core_execution.py：13 passed。
- execution 加相关 r2 回归的联合集合：28 passed、1 skipped；skip 仍是 Windows 无创建 symlink 权限时的预期分支。

r2 pre-generation、静态发布、author-job 准备与本文交接累计为 0 次外部模型调用、0 外部模型费用。该计数不覆盖项目历史上其它实验会话，也不是上述 execution accepted 记录的调用或费用汇总。

## 推断与限制

1. 已完成的是可恢复的机械输入准备、隔离和配额检查，而不是对话语义验收。realism 卡控制结构与分布，不能替代对 200 条最终正文的 owner 抽检。
2. 预发布的两个 author job 是接口和恢复路径的样本；它们不代表全部 60 个 batch 已发放。`dev-mini-001..008` 与 `core-009..015` 的 375 条已在独立 execution runtime 中 accepted，但这不推出 `core-016` 或其余 45 批已经生成。
3. source-majority 指标是有用的捷径风险诊断，而非训练或模型的主要结果；应与 post-generation audit、五配置真实运行结果一起报告。
4. static r2 root 的不可变性、dynamic author-jobs sibling 的隔离，以及 future execution ledger 的 create-only 行为，目的是让中断恢复与事后复核可行；它们不是把 core 扩展成 formal research gate 的声明。

## 执行进展与剩余工作

执行账本的测试 blocker 已关闭：tests/synthesis/test_portfolio_core_execution.py 已 13 passed，联合回归为 28 passed、1 skipped；`dev-mini-001..008` 与 `core-009..015` 的 375 条已通过 durable ledger 后 accepted。该闭环证明的是可恢复的多 batch 运行与接受顺序，不是 1,500 条完成证明。

剩余工作是从 `core-016` 开始的其余 45 个 batch、1,125 条的同一恢复式单 writer 执行，以及全部 1,500 条完成后生成 cluster-aware 的 200 条 owner 抽检包。每个新 batch 仍须先经过 corpus model guard，只读对应 model-visible author packet 与 opaque asset alias，随后以恰好 25 条全新 plan_id + turns 提交给 create-only execution ledger；禁止复用旧 dev 正文，禁止并行作者写同一 batch。

用户已批准的 core 机械 auto-approval 只在 ledger、计划/realism/packet SHA 与 schema 校验均通过后生效。200 条抽检通过后才冻结并进入五配置真实运行；若抽检不合格，保留 r2 证据并创建后续 revision/run，禁止原地覆盖已接受产物。

截至本文更新：已 accepted 375/1,500（25%），尚余 45 个 batch、1,125 条与最终 200 条抽检；不得声称 core 语料、训练集或五配置实验已经完成。
