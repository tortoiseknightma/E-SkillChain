# Portfolio Core r2 执行计划

> 状态：执行中；本文件是确定性工程交接包，不包含正式用户话术。
>
> r1：`E:\skillchain-data\runs\portfolio-core-20260804-r1`，保留且不可改写。
>
> r2 目标：在任何正式正文生成前，关闭路径标签捷径、六能力覆盖断层、用户行为未建模、样本相关性过高和抽检覆盖不足问题。

> 2026-08-05 execution 快照（仅元数据）：`dev-mini-001..008` 加 `core-009..015` 已 accepted 375/1,500（25%）；尚余 45 个 batch、1,125 条，执行从 `core-016` 继续。core 尚未完成。

## 0. 当前与历史 execution 里程碑元数据

- 当前 25% checkpoint SHA-256：`8b64a16319fe569c87229bf74c6abe990393cec748233cec72fc5d70cb9cd7b5`；accepted ledger SHA-256：`23b238ca11e36d16bf937990a0e8cbfde63461a3096c5e81c3f601f4e2347427`。15 个 accepted 目录与 ledger 一致，staging 为空；checkpoint/ledger、manifest 与 author-draft-receipt 的元数据绑定已只读复核。
- 历史 200 条 dev-mini 里程碑 checkpoint SHA-256：`237a16541a67536794af937d9c97a112708337fc50134668a85539b54070613a`；aggregate results SHA-256：`1d01c160f380fdfd911d725e4684e5885fb433e6cec94796d2dcfdef0a6ece91`。
- 该 200 条里程碑的 capability：`Exact/Multi/Style/Ency/Doc/Recipe=35/35/35/35/30/30`；`boundary=40`。
- 该 200 条里程碑的 turn shape：`single/three=140/60`；interaction：`direct/clarification/correction/no-result/goal-change=140/30/10/10/10`；normalized 文本唯一数为 200，legacy overlap 为 0，UTF-8 校验有效。

## 1. 冻结范围与非目标

### 1.1 必须保持不变

- 总量 `1,500`，批次 `60 x 25`。
- 顶层五 intent：`exact_match / multi_product / divergent_rec / encyclopedia / utility`。
- 六 capability：
  `product.exact_match / product.multi_search / product.style_recommendation /
  knowledge.visual_encyclopedia / utility.document_reading /
  utility.recipe_guidance`。
- 五个主配置：`NoSkill / LLMStaticSkill / S1 / S1+S2 / Full`。
- 最终 split：`dev_mini=200 / opt_pool=800 / val=200 / test_frozen=300`。
- 五配置共享完全相同的冻结 query、图像字节、预算和工具注册表。
- RAW、r1、已接受 dev_mini、历史 staging/rejected/ledger 均不可删除、覆盖或重写。

### 1.2 本轮明确不做

- 不把合成 query 描述为真实用户日志或生产流量。
- 不由文本作者臆造工具成功、检索失败、OCR 失败或 reward。
- 不改五 intent taxonomy，不增加第六 intent。
- 不扩建许可、PII、外部 trust-root 或发表级治理体系。
- 不调用外部模型，不运行五配置正式实验，不消耗模型预算。
- 子代理不得查看、创作、改写或审核正式用户话术或图片内容。

### 1.3 子代理与主会话的职责边界

- 子代理负责：代码、schema、deterministic planner、sidecar、opaque alias、
  split、audit sampler、测试、create-only r2 计划和 work order。
- 经用户确认的 **5.6 Sol Ultra 主会话**负责：查看每张正式图片并创作
  `plan_id + turns` 正文。
- 子代理必须在第一份 model-visible author packet 发出且所有 pre-generation gate
  通过后停止；不得代写第一批 25 条。

## 2. r2 冻结数据分布

### 2.1 capability x split 精确矩阵

| capability | dev_mini | opt_pool | val | test_frozen | total |
|---|---:|---:|---:|---:|---:|
| Exact Match | 35 | 209 | 52 | 79 | 375 |
| Multi-Product | 35 | 117 | 29 | 44 | 225 |
| Style Recommendation | 35 | 163 | 41 | 61 | 300 |
| Visual Encyclopedia | 35 | 163 | 41 | 61 | 300 |
| Document Reading | 30 | 32 | 10 | 18 | 90 |
| Recipe Guidance | 30 | 116 | 27 | 37 | 210 |
| **合计** | **200** | **800** | **200** | **300** | **1,500** |

解释：Document Reading 从 r1 的 30 条提高到 90 条，30 条 dev 前缀保留，
新增 60 个 fresh、互不泄漏的文档 component，确保 opt/val/test 都可训练和评测。
Utility intent 总量仍为 300，不改变五 intent 主比较。

2026-08-04 的 split-atomic 可行性修订：原表的 `opt_pool Document=30` 与
`60 x 25` 的 generator-batch 原子 split 约束矛盾。`opt_pool=800` 必须由 32 个完整
batch 构成，而每个 batch 都要求至少 1 个 Document，因此下限是 32 而不是 30。
为保持四个 split 总量、全局六 capability 总量、boundary 矩阵和复用上限不变，采用
最小 L1 的四格交换：opt `Document +2 / Recipe -2`，test `Document -2 / Recipe +2`。
因此最终使用上表的 opt `32/116` 与 test `18/37`；不得以混合一个 generator batch
内的 final split 来绕过该约束。

### 2.2 capability x split 的 boundary 精确矩阵

| capability | dev_mini | opt_pool | val | test_frozen | total |
|---|---:|---:|---:|---:|---:|
| Exact Match | 11 | 34 | 8 | 13 | 66 |
| Multi-Product | 2 | 23 | 6 | 8 | 39 |
| Style Recommendation | 10 | 26 | 7 | 10 | 53 |
| Visual Encyclopedia | 11 | 26 | 6 | 10 | 53 |
| Document Reading | 3 | 8 | 2 | 5 | 18 |
| Recipe Guidance | 3 | 20 | 5 | 6 | 34 |
| **合计** | **40** | **137** | **34** | **52** | **263** |

这保持 r1 的五 intent boundary 总量，同时使稀有 Document boundary 在 val/test
中非零。`boundary=17.53%` 是预注册的鲁棒性 benchmark 分布，不得声称是真实流量先验。

### 2.3 用户行为设计分布

以下是 designed mixture，不是真实流量估计；每个 split 均按同一比例冻结。

| interaction_pattern | 比例 | 1500 数量 | turn shape |
|---|---:|---:|---|
| `direct_request` | 70% | 1,050 | `[user]` |
| `underspecified_clarification` | 15% | 225 | `[user, assistant, user]` |
| `constraint_correction` | 5% | 75 | `[user, assistant, user]` |
| `no_result_relaxation` | 5% | 75 | `[user, assistant, user]` |
| `goal_change_or_multi_query` | 5% | 75 | `[user, assistant, user]` |

每 split 精确数量分别为：

- dev/val：`140 / 30 / 10 / 10 / 10`；
- opt：`560 / 120 / 40 / 40 / 40`；
- test：`210 / 45 / 15 / 15 / 15`。

表达复杂度：

| expression level | 比例 | 1500 数量 |
|---|---:|---:|
| S0：最小动作/对象线索、允许省略或指代 | 15% | 225 |
| S1：可直接路由的日常短句 | 45% | 675 |
| S2：1–2 个自然约束或一次澄清 | 30% | 450 |
| S3：多个可验证细节、排除项或输出要求 | 10% | 150 |

用户 register：

- `terse_fragment=25%`
- `colloquial=35%`
- `neutral_complete=25%`
- `polite=5%`
- `code_mixed_or_numeric=5%`
- `typo_or_asr_like=5%`

约束数量：`0=30% / 1=40% / 2=25% / 3_plus=5%`。

boundary 的 ambiguity subtype：

- `resolved_near_boundary=158`
- `clarification_required=105`

`clarification_required` 必须是三 turn；canonical label 解释完整轨迹最终已解析的
目标，而不是把初始歧义硬标为唯一正确路由。

## 3. r2 新产物

所有产物 create-only；失败时保留证据，不覆盖 r1。

- `E:\skillchain-data\clean\portfolio-core-asset-catalog-v9`
- `E:\skillchain-data\clean\portfolio-core-capability-assignments-v9.jsonl`
- `E:\skillchain-data\clean\portfolio-core-v8-capability-binding-plan.json`
- `E:\skillchain-data\clean\portfolio-core-v9-rpc-multi-closure-catalog-receipt.json`

当前冻结 SHA-256：catalog v9 为
`539dcc5885c09cb1412b0fe334f6dcf8334e5a0ff7dca8ee122c78309856adb6`，assignments
v9 为 `4dbb18e551ab9194416b7d966d57d8003d41f1dc8a98a363a6357e92e0986a81`，
v9 receipt 为 `da6e1536bd5dafce2d07ced1bda622112a1543964b8f639254c42a2bc9b943ae`。
父级 v8 binding plan SHA
`10f658c2918bcea32a70e1a1cad913429407fcd0b70b1e1d114d6b8ef9bce69d` 与 assignments
v8 SHA `03d11f63e999450d65faa83227d2705bc3893559841ef2e4fb73cbf00ae32ec0`
继续作为不可变 lineage 输入。
- `E:\skillchain-data\runs\portfolio-core-20260804-r2`
- `plans/core-r2.json` 与 manifest
- `plans/core-r2-dev-prefix.json` 与 manifest
- `authoring/realism-assignments.jsonl` 与 manifest
- `authoring/prompt-family-specs.json` 与 manifest
- 每个 job 的 internal `work-order.json`
- 每个 job 的 model-visible `author-packet.json`
- 每个 job 的 `author-assets/` opaque 普通文件/硬链接
- `audit-samples/<sample-id>/audit-index.jsonl`
- split freeze 后的 `route_gate / body_gate / shadow_val` 选择清单
- component-balanced S1 selection 清单

新 run manifest 必须声明 `supersedes_run=portfolio-core-20260804-r1`，但不得切换或
删除 r1 active pointer，直到 r2 所有 pre-generation gate 通过。

## 4. Work Package 0：基线与工作树保护

### 修改

无代码修改。先记录：

- `git status --short`
- r1 plan/manifest/work-order/checkpoint SHA
- asset catalog v4 与 assignments v4 SHA
- 当前聚焦测试基线

### 约束

- 当前工作树包含大量既有未提交修改；均视为用户工作。
- 不运行 reset/checkout/clean，不批量格式化，不修改无关文件。
- 不提交、不推送、不删除临时或历史数据，除非用户另行明确授权。

### 验收

- r1 所有已存在文件在实施前后字节 SHA 不变。
- 变更清单只包含 r2 计划声明的文件。

## 5. Work Package 1：去除模型可见路径捷径（P0）

### 修改文件

- `src/skillchain/evaluation/assistant_runs.py`
- `src/skillchain/runners/assistant.py`
- `tests/evaluation/test_phase4_assurance.py`
- 必要时相关 typed request schema；不得放宽 tool authority。

### 实现

1. 将内部 verified asset binding 与模型 public payload 分离。
2. public payload 仅保留：
   - opaque `asset_id`/`asset_token`；
   - `text`；
   - `turns`。
3. `image_path`、source dataset、split、intent、capability、真实绝对路径不得进入
   routing/action 模型可见字符串。
4. runner 仍从隐藏的 authoritative binding 读取本地图、验证 bytes/hash，并向模型
   附加图像；工具只能消费 opaque token 对应的 verified asset。
5. forbidden-value 检查增加已知 source/path 词和路径分隔符的回归测试，但不要用
   脆弱的生产字符串黑名单代替结构隔离。

### 验收测试

- 捕获的 routing/action 请求不含 `image_path/query_images/abo/rpc/fashioniq/
  inaturalist/isia/wikimedia`。
- 同一 asset bytes 换本地目录，public payload SHA 完全不变。
- 隐藏 binding 被篡改时 fail closed。
- 图像附件与工具调用仍绑定相同 authoritative bytes。
- 五配置 public input identity 测试保持通过。

## 6. Work Package 2：补足 Document Reading、跨意图绑定与 Multi catalog closure

### 修改文件

- `src/skillchain/data/portfolio_core_inventory.py`
- `src/skillchain/data/portfolio_core_assets.py`
- 对应 tests；仅当现有 Wikimedia adapter 无法表达 fresh selection 时修改 adapter。

### 实现

1. 不做网络下载。保留 dev 前缀 30 个已绑定的 Wikimedia document component。
2. 本地只读盘点确认：Wikimedia 的 282 条目前只有 metadata、没有 fresh image cache；
   不得把 metadata 当作已取得图像。改用项目已验证存在的本地 CORD v2 固定提交与
   SROIE official archives 作为 Document 补充源，这与 source portfolio 中
   `CORD/SROIE 作字段补充`的既定角色一致。
3. 使用显式多源 `carry-forward + fresh` adapter：
   - `carry_forward`：旧 Wikimedia 30；
   - `fresh_cord`：CORD 30；
   - `fresh_sroie`：SROIE 30；
   - 每个 fresh source 额外准备少量 reserve。
   所有 fresh 必须排除旧 record/content/component，并在各源内部及跨源去重。
4. 发布新的 inventory/selection/catalog/assignments；v4 不改。
5. preflight 必须区分“query quota”和“unique candidate floor”。第 2.1 节仍是 query
   quota；最终 candidate floor 冻结为 Exact 340、Multi 125、Style 150、Encyclopedia 152、
   Document 90、Recipe 210。Style/Encyclopedia floor 可包含显式 cross-intent bindings。
   这些 floor 与 max reuse=3、cross-intent 设计共同保证可行，不能误设为每条 query
   都要求一张独立图。
6. materializer 保持断点恢复、RAW 只读、目标 create-only。

### 验收

- 90 个 Document plan slots 可由 90 个独立 component 支撑；CORD/SROIE 各建议额外
  保留 5 个 create-only reserve，但 reserve 不进入 plan quota。
- 30 dev 与 60 fresh 的 asset/content/leakage component 无交集。
- 任一 capability 容量不足时在 r2 plan 发布前明确失败。
- catalog v7 全文件复验通过；v4 SHA 不变。第一次 v5 candidate merge 因 retained
  FashionIQ/iNaturalist 的 source path 未 rebase 而在 preflight fail closed；v5 作为失败
  证据保留且不得物化/规划。v6 修复 rebase 后得到 997 assets/847 components，仍低于
  冻结的 850-component floor；v7 通过签名 sidecar 激活 5 个 CORD reserve，实测
  1002 assets/852 components。v6/v7 必须绑定父 SHA、修复/激活理由和预测/实测计数。
  v8 不改 catalog 或 v7 bytes，只以 create-only binding plan 为 non-dev tail 增加
  14 个 ABO Exact/Style/Encyclopedia triplet 与 10 个 ISIA
  Encyclopedia/Recipe pair；实际 assignments 为 1100，ABO triplet 为
  global/dev/non-dev `28/8/20`，Food pair 为 `10/0/10`。
  capability-aware dry-run 随后证明 Multi tail 在“component 不跨 batch、每 component
  最多三条”下至少需要 86 个 non-dev component，而 v8 只有 70 个。v9 因此只读使用
  已锁定 RPC RAW，提升 4 个旧 reserve 并增加 16 个 fresh；post-catalog 实测为
  1022 assets/872 components、Multi `global/dev/non-dev=125/35/90`，assignments
  append-only 增至 1120。不得以跨 batch 复用或降低 unique-component gate 替代该闭合。

## 7. Work Package 3：capability-aware planner、批次平衡与反事实组

### 修改文件

- `src/skillchain/synthesis/planning.py`
- `tests/synthesis/test_planning.py`
- `tests/synthesis/test_catalog_planning.py`

### 实现

1. 从 r1 core plan 锁定前 200 条的成员、顺序和语义，创建 in-memory r2 dev
   prefix rebind：逐条验证 r1 `core.json` 顶层实际绑定的 catalog
   v4（`b01d4689…`）与 assignments v4（`03d91ee6…`）中相同
   `asset_id + path + intent` 的 component、capability、tools、card 语义均未漂移；仅更新
   assignment provenance 与顶层 SHA 到 catalog v9/assignments v9。v7/v8 是历史 lineage
   输入，不能替代 r1 plan 的直接 parent binding；不得重采样、folder discovery 或改写 r1。
2. 新增 `CORE_R2_CAPABILITY_COUNTS` 与第 2 节的 split/boundary targets；不要改 r1
   constants 的语义。
3. planner 由仅按 intent 分配改为按 `(intent, capability)` 分配。
4. 52 个 tail batch 使用确定性 largest-deficit schedule。每批必须含六 capability；
   每批数量只能是全局均值的 floor/ceil：
   - Exact `6–7`
   - Multi `3–4`
   - Style `5–6`
   - Encyclopedia `5–6`
   - Document `1–2`
   - Recipe `3–4`
5. 每个 batch 使用独立 template-family instance；真实 recipe ID 另存 sidecar，防止
   同 recipe 把多个 batch 连接成一个 split 原子簇。
6. 非 dev 至少创建 30 个 cross-intent counterfactual components：
   - 至少 20 个 Exact/Style/Encyclopedia triplet；
   - 至少 10 个 Food Encyclopedia/Recipe pair；
   - 按 `opt/val/test = 18/5/7` 预冻结 component 归属；同 component 绝不跨 split。
7. 目标独立 component 数 `>=850`；单 component 最多三条。
8. 三重复允许两类：显式 cross-intent counterfactual，或由精确容量/批次隔离证明
   必需的同-capability reuse。后者必须取整数模型给出的最小三重复数量、同 batch/
   同 split、写出 `capacity_required` 理由，并分配三个不同 `reuse_variant`；后续
   surface-near-duplicate audit 必须拒绝纯改写。任何 component 仍最多三条。
9. 每个 generator batch 内 component 不跨其他 generator batch。

### 验收

- 第 2.1、2.2 节所有计数逐项精确。
- 60 个 batch 均为 25 条，且均覆盖六 capability。
- tail Document 非零并精确为 60。
- cross-intent 不再只存在于 dev。
- unique component `>=850`，max reuse `<=3`。
- source ID majority baseline 的 capability 可猜中率必须低于 r1；同时保留该数值作为
  诊断指标，不把它当唯一去捷径证明。

## 8. Work Package 4：realism sidecar 与真实 prompt-family 规格

### 新增/修改文件

- 新增 `src/skillchain/synthesis/portfolio_core_authoring.py`
- 修改 `src/skillchain/synthesis/portfolio_core.py`
- 修改 `src/skillchain/synthesis/batches.py`
- 新增 `tests/synthesis/test_portfolio_core_authoring.py`
- 修改 `tests/synthesis/test_batches.py`

### sidecar schema

`RealismAssignment` 一行对应一个 plan_id：

- `plan_id`
- `trajectory_shape`
- `interaction_pattern`
- `expression_level`
- `user_style`
- `constraint_level`
- `ambiguity_subtype`
- `reuse_variant`
- `prompt_recipe_id`
- `prompt_recipe_sha256`

不含 source/path，不改变 canonical intent/capability。manifest 绑定 plan SHA、catalog
SHA、assignments SHA、seed、row count 和 sidecar SHA。

### 分配算法

- 按 split x capability x boundary 分层的 deterministic largest-remainder 分配。
- 固定 seed 相同则 canonical bytes 完全相同。
- 先满足 interaction/ambiguity 的组合约束，再填 register、expression、constraint。
- 非 `direct_request` 必须三 turn；`clarification_required` 必须映射
  `underspecified_clarification`。
- 不机械声称自然语言满足某种风格；机械 gate 只验证配额、组合和 turn shape，语义
  合规由 200 条抽检验证。

### prompt family

- 定义约 12 个结构化 recipe，不写正式用户例句。
- `template_family` 使用 batch-unique instance ID；`prompt_recipe_id` 可跨 batch 复用。
- recipe spec 和 SHA 必须进入 work order/generation input，而不是只有 `block-xxx` 名称。

### 验收

- sidecar 恰好 1,500 行、plan_id 一一覆盖、配额精确。
- 任一 sidecar/spec bit 改变均改变 generation input SHA，旧 draft 被拒。
- 非法 turn shape 在 staging/auto-approval 前失败。
- legacy dev/full workflow 不提供 sidecar 时保持兼容。

## 9. Work Package 5：opaque author packet 与可恢复 work order

### 修改文件

- `src/skillchain/synthesis/portfolio_core_authoring.py`
- `src/skillchain/synthesis/portfolio_core.py`
- `tests/synthesis/test_portfolio_core_authoring.py`
- `tests/synthesis/test_portfolio_core.py`

### 实现

1. internal `work-order.json` 可保留真实 binding，仅供 orchestration 验证。
2. model-visible `author-packet.json` 不含 canonical path、source ID、assignments
   provenance，只含：plan ID、intent/capability、boundary、realism card、opaque alias。
3. 在 job 内 `author-assets/a0001.<ext>` 创建普通硬链接或经哈希验证的 create-only
   copy；禁止 symlink/junction。
4. alias manifest 内部绑定 alias -> asset_id -> catalog bytes/hash；同一 job 相同
   asset 复用同 alias。
5. work order schema v2 绑定 realism、recipe、alias、author packet 的 SHA；schema v1
   r1 仍可加载且不可改写。
6. CLI 默认把 author packet 作为作者入口，不回显真实 source path。

### 验收

- author packet 字符串扫描不含 source 名、真实目录和 E:/D: 绝对路径。
- alias 是 job root 下普通文件，bytes/hash 与 catalog 一致。
- 缺失、篡改、错映射、symlink 均 fail closed。
- 重复调用返回同 job ID、同 packet bytes，不覆盖已有文件。
- checkpoint `issued -> staged -> accepted` 单调、可恢复。

## 10. Work Package 6：capability-aware split 与 gate 冻结

### 修改文件

- `src/skillchain/synthesis/splitting.py`
- `tests/synthesis/test_r2_splitting.py`
- `tests/synthesis/test_group_splitting_v2.py`
- `tests/synthesis/test_splitting.py`

### 实现

1. `SplitSpec` 增加 backward-compatible capability target/minimum。
2. AtomicGroup 记录 capability 与 boundary-capability counts。
3. MILP 同时满足：split 总量、locked dev、capability 精确矩阵、boundary 精确矩阵、
   GROUP_FIELDS 零泄漏。
4. 不可行时在任何正文生成前失败；不得在看到实验结果后调 quota。
5. val=200 恰为 8 个不可拆分的 25-query generator batch，因此在 split freeze 时
   进一步 group-safe、create-only 冻结为：
   - `route_gate=75`（3 batch）
   - `body_gate=75`（3 batch）
   - `shadow_val=50`（2 batch）
   三个 gate 每 capability 的硬下限依次为 3/3/2，且每个 gate 至少覆盖一条
   boundary；需要的 interaction pattern 下限由完整 realism sidecar 提供。旧的
   80/80/40 会拆开 25-query batch，属于数学不可行配置，必须显式拒绝。

### 验收

- 第 2 节完整矩阵精确满足。
- dev prefix 完整保留，任何 component 零跨 split/gate。
- 不可行 fixture 返回清晰的 capability/boundary infeasible 错误。
- frozen manifest 写出 capability、boundary、component counts 并可重放。

## 11. Work Package 7：cluster-aware 200 条抽检

### 修改文件

- `src/skillchain/synthesis/portfolio_core.py`
- `tests/synthesis/test_portfolio_core.py`

### 算法

使用 deterministic greedy coverage + largest remainder：

1. 先保证 60/60 generator batch 各至少一条。
2. 保证每个非空 capability x boundary x turn-shape cell 至少两条。
3. 保证每个 interaction pattern、expression level、user style、ambiguity subtype、
   prompt recipe、source 至少一条。
4. 对 Document boundary、cross-intent、S3、no-result、goal-change 等稀有 cell
   设置显式 minimum。
5. 剩余名额按 split x intent x capability x boundary 比例填满到恰好 200。
6. `audit-index.jsonl` 记录每条 selection reason、batch、component、strata；manifest
   绑定 index 和 sample query SHA。

### 验收

- 精确 200 条；同 seed 同 bytes。
- 60 batch、全部 recipe、全部非空稀有 strata 被覆盖。
- Document boundary 不再可能抽到 0。
- sample/index/manifest 任一篡改均拒绝 review。
- 若 mandatory coverage 超过 200，提前失败而不是静默丢 strata。

## 12. Work Package 8：S1 选择与真实 trace readiness

### 修改文件

- `src/skillchain/stage1.py`
- `scripts/prepare_stage1.py` 或一个最小选择 CLI
- `tests/test_stage1.py`
- `tests/evaluation/test_phase4_assurance.py`

### S1 selection

- 完整 opt_pool 800 条仍作为冻结 artifact。
- Creator 输入使用 create-only 的 240 条 component-unique 代表集。先按 capability
  取 `min(40, 可用唯一 component 数)`；若某 capability（当前主要是 Document）
  少于 40，则按“已选数/可用唯一 component 数”最小优先、capability 固定顺序
  打破平局，将缺口逐条确定性分配给仍有余量的 capability，直到恰好 240。
- 在每个 capability 的最终额度内，以 capability x interaction x boundary 的均衡
  目标做整数优化；所有非空全局 interaction x boundary cell 必须覆盖，细粒度 cell
  以 L1 偏差最小化。由于少数 cross-intent rare cell 共享其唯一 component，不能同时
  把“每个细粒度 cell 至少一条”和“全局 component 唯一”都设为硬约束。每 component
  恰好最多一条。
- 固定 seed、固定序列化字节和最大 packet bytes。
- 严禁 dev/val/test_frozen 进入 S1。

### runner trace readiness

证明每条真实运行可重建以下 typed record：

- query/component/config/run IDs
- selected/canonical/acceptable capability
- route correctness/hash
- tool status/error（错误保留分母）
- 用户可见 evidence、response、usage、latency
- Judge/rule scores（只在允许阶段可见）

不得依赖解析自由文本日志，也不得把内部 trace 塞给 Final Judge。

### 验收

- S1 selection 恰好 240，六 capability 均非零，配额由上述容量感知算法唯一决定，
  component 不重复；任何 capability 可用唯一 component 总量变化都会使绑定 SHA
  变化并要求重新冻结，而不是静默回退到重复 component。
- 不同 split 泄漏测试 fail closed。
- runner 成功与失败 fixture 都能生成同一 typed trace schema。

## 13. Work Package 9：最小 S2/S3 原型（正式运行前完成）

这部分不阻塞 r2 计划与正文生成，但阻塞五配置完整 core 训练实验。

### 新增最小模块

- `src/skillchain/evolution/models.py`
- `src/skillchain/evolution/attribution.py`
- `src/skillchain/evolution/route_optimizer.py`
- `src/skillchain/evolution/body_refiner.py`
- 对应聚焦 tests

### attribution

`EvolutionExample` 绑定 run/split/config/component hashes，并将真实失败互斥归因为：

- `routing`
- `tool`
- `body`
- `insufficient_evidence`
- `infra_error`

确定性优先级：route 不可接受 -> routing；route 对但工具失败 -> tool；route/工具
正确但回答规则失败 -> body；证据不足或基础设施错误不进入训练。

### S2 gate

- 只读 opt_pool 的 routing attribution。
- 只允许修改 route/description 层。
- 用 route_gate 上 capability + component 宏平均的 route macro-F1 接受/回滚。
- Body/Cs/Od hashes 必须不变；test_frozen 不可读。

### S3 gate

- 只读 route-correct 的 body attribution。
- 冻结并复用 S2 route hash，只允许修改 body。
- body_gate 上配对 J_project 改善为软目标；grounding/safety 不回退为硬门。
- route/description/Cs/Od 漂移或安全回退即 rollback。

### 验收

- S2、S3 各完成一次 fixture smoke 的 candidate -> accept/rollback。
- S1+S2 与 Full 的 route trace 对同 query 完全一致。
- infra/tool failures 不被误当作 body 训练样本。
- 无提升时可重放 rollback；不要求本阶段证明显著正增益。

## 14. Work Package 10：create-only 发布 r2（子代理终点）

按顺序执行，任一 gate 失败即停止：

1. 发布 fresh Document inventory/selection。
2. 只读复核 r1 的 catalog v4/assignments v4 顶层 binding，并完成 create-only RPC v9
   closure；planner 只绑定 catalog v9 与 assignments v9。
3. dry-run capability-aware planner 和 split MILP。
4. 发布 core-r2/dev-prefix plan 与 manifest。
5. 发布 realism sidecar 与 prompt recipe manifest。
6. `prepare` 新 run，声明 supersedes r1。
7. 重新验证全部 SHA、容量、分布、零泄漏、public path isolation。
8. 发行第一批 model-visible author packet 和 opaque aliases。
9. 停止，不查看图片，不写 25 条正文，不调用模型。

必须输出一份 handoff report：

- 改动文件及未触碰的 r1 SHA
- r2 artifact 路径和 SHA
- capability/boundary/interaction/component 分布
- split dry-run 可行性
- public payload/author packet path-leak scanner 结果
- 聚焦测试与相关回归结果
- 第一份 author packet 路径
- 剩余 blocker

## 15. 主会话正式生成与后续执行

子代理完成后，由主会话执行：

1. 用户在当前主会话确认显示模型为 `5.6 Sol Ultra`。
2. 运行 corpus model guard；未通过即停止。
3. 只读取 model-visible author packet；逐个查看 opaque alias 对应图片。
4. `dev-mini-001..008` 加 `core-009..015` 已独立 accepted 375 条。对从 `core-016` 开始的其余 45 个 batch，
   每批恰好生成 25 条 `plan_id + turns`，写入 job inbox。`legacy_prefix` 只表示 r1 core
   plan 的 ID/order/metadata lineage，不表示可复用历史 dev_mini 正文：旧 accepted 200 与
   r2 前缀的 image_path 为 0/200 一致，禁止按相同 query_id 导入旧 turns。
5. core wrapper 验证 schema、plan、catalog、sidecar、turn shape、duplicate、hash 后按
   用户已批准的 core 机械 auto-approval policy 晋级。
6. 可中断恢复，禁止并行作者写同一 batch。
7. 1,500 完成后创建 cluster-aware 200 条样本给 owner 抽检。
8. 抽检通过后冻结 splits/gates；不通过则保留 r2 并新建后继 revision/run，禁止
   原地改写 accepted corpus。
9. 运行五配置，收集真实 runner trace，再执行 S1/S2/S3。

## 16. 测试顺序

### 16.1 聚焦测试

```powershell
uv run pytest `
  tests/data/test_portfolio_core_inventory.py `
  tests/data/test_portfolio_core_assets.py `
  tests/synthesis/test_planning.py `
  tests/synthesis/test_catalog_planning.py `
  tests/synthesis/test_group_splitting_v2.py `
  tests/synthesis/test_splitting.py `
  tests/synthesis/test_batches.py `
  tests/synthesis/test_portfolio_core_authoring.py `
  tests/synthesis/test_portfolio_core.py `
  tests/evaluation/test_phase4_assurance.py `
  tests/test_stage1.py -q
```

### 16.2 相关回归

```powershell
uv run pytest tests/synthesis tests/evaluation tests/test_stage1.py -q
```

只记录无关历史失败到 backlog；不得为与 r2 无关的环境漂移扩大任务。

## 17. Go/No-Go

### G0：允许修改代码

- r1 SHA 已记录；dirty worktree 已识别；无覆盖风险。

### G1：允许发布 r2 plan

- public path leak 修复及测试通过；Document fresh 容量通过；第 2 节矩阵和 split
  MILP 可行；unique component >=850。

### G2：允许主会话生成正文

- realism 1,500/1,500；author packet 无 source/path；第一 work order 所有 SHA 可重建；
  聚焦测试与相关回归无 r2 逻辑失败。

### G3：允许冻结 split

- 1,500 mechanically valid；owner 200 条抽检通过；capability/boundary/interaction
  coverage 与计划一致。

### G4：允许 core 训练实验

- splits/gates/S1 selection 冻结；runner trace schema smoke 通过；S2/S3 最小
  accept/rollback 原型通过；test_frozen 隔离。

## 18. 回滚与停止条件

- 所有新产物 create-only；失败不删除，记录 blocker。
- r2 未过 G2 时不得切换 active pointer，不得生成正式话术。
- split 不可行、Document 容量不足、alias/path 泄漏、SHA 漂移、r1 被改动、测试出现
  可复现 P0/P1 时立即停止。
- 不以降低 capability、boundary、audit 或 isolation 要求来“让测试通过”。
- 如果需要从六 capability 降为五 capability，属于改变项目主张，必须重新请求用户批准。

## 19. 完成定义

r2 工程实施完成需同时满足：

- r1 完整保留；r2 create-only 发布。
- 模型可见 payload 和 author packet 均无 source/path 标签捷径。
- 六 capability 在 opt/val/test 非零且满足精确矩阵。
- 1,500 条 realism assignments 与 plan 一一绑定。
- 60 批近似均衡、cross-intent 扩展到 tail、unique component >=850。
- 200 条 audit sampler 覆盖所有 batch 和稀有 strata。
- S1 component-balanced selection 与 val gates 可确定性冻结。
- 第一份 author packet 已就绪，但没有任何子代理生成的正式话术。
- 聚焦测试及相关回归结果已报告，已知限制进入 backlog。
