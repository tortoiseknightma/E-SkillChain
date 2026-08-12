# Core 1,500 面试导向增益合并执行计划

日期：2026-08-06
轨道：Portfolio Track
状态：Gate 0 与 Static opt800 GCS v2（800/800）已完成；Qwen Feedback→整 Bank S1 的首轮真实闭环已于 2026-08-10 完成。Replay200 因 Encyclopedia `−5pp` 违反单能力 `≥−3pp` 门而失败，候选已 byte-exact 回滚至 Static；未进入 body_gate75、val 或 test。

> 2026-08-12 起的前向执行口径见本文件“2026-08-12 forward-only disposition”。此前冻结的计划、配置、receipt 和 R0 结果均不回写。

## 0. 唯一执行口径

本文件是 Core 1,500 后续开发、候选选择、validation gate、test 解封和结果交付的唯一执行计划。

- `2026-08-06-dev-mini-200x5-core-gain-plan.md` 保留为 200×5 事实、程序缺陷和根因诊断证据，不再单独维护执行状态。
- `D:\athena\ECommerceSkillChain\docs\Core 1500 面试导向增益改造与实验计划.md` 作为设计输入保留，不作为运行入口。
- 既有 Formal Research 文件不改写；本计划只约束 Portfolio Track，也不把 Portfolio 结果包装成论文正式复现。

项目目标不是继续追求论文绝对分数，而是在真实电商语义、公开数据、真实模型和真实工具上，做出能向面试官解释的增益链：

```text
失败归因
→ S1 Creator 的 failure-driven 整 Bank 改进
→ S2 的风险敏感路由优化
→ S3 的 common-route 正文精炼
→ validation 接受或回滚
→ 冻结 test 的一次性最终比较
```

不改变六个 capability 和五个主配置：`NoSkill / LLMStaticSkill / S1 / S1+S2 / Full`。最多进行三轮有记录的算法优化；三轮后停止，不为好看结果改标签、改 rubric 或继续试第四轮。

## 1. 合并后采用、修正和推迟的建议

| 议题 | 合并决策 | 原因与披露 |
|---|---|---|
| S1 failure clustering | 采用，并以整 Bank 候选实现第一轮 | 旧 S1 不是从基线失败和 parent Bank 学习；Core 必须改成真实 failure-driven Creator。当前 Creator 输出完整六 Skill Bank，不包装成 capability-local patch；稀疏局部 S1 compiler 留到后续版本。 |
| S2 confusion-pair Description patch | 第一优先 | 最接近 Skill Route Optimizer，且便于把增益归因到 Description。 |
| TF-IDF + Logistic Regression Hybrid | 条件采用 | 仅当 Description-only 的质量或成本目标不足时进入同一轮比较；若胜出，明确称为“监督式混合路由与成本优化”，不包装成纯 Skill 优化。最终只保留一种 S2。 |
| S2 最少 20 个 route flip | 删除 | flip 数量不是目标；错误改写的代价约为正确修复收益的 2.5 倍，应优化净效用。 |
| Core 外层 `200/800/200/300` | 保持不变 | 当前 grouping 隔离可靠，opt 已用于 surrogate 且 launch/hash 已冻结；诊断后重切 holdout 会破坏连续性。现有来源、repair 和 capability 不均通过 component-aware replay 与分层报告处理。 |
| Val 内层 Gate | 修正计划口径并复用既有 `75/75/50` assignment | `80/80/40` 无法由完整 25-query 原子 batch 构成，且代码已 fail closed 拒绝；既有 create-only receipt 已是 `route_gate=75 / body_gate=75 / shadow_val=50`，首个 val response 前只复核绑定，不重新分配。 |
| SkillOpt 式细粒度 Body patch | 条件采用为 S3 proposal engine | 只迁移原子 patch、edit budget、rejected-edit buffer 和 Skill defect / execution lapse 分类；不引入完整 SkillOpt 训练框架，也不替换现有 Bank compiler、paired Gate、lineage 和 rollback。 |
| S3 Semantic Slots + constrained composer | 条件采用 | 先验证 Body-only 稀疏 patch；只有证据完整且 Body-only 不足时才加入 composer，并单独做 Body-only / Body+Composer 诊断消融。 |
| `interview_challenge=120` | 推迟为可选压力集 | 先使用 Core 已有 38 个 same-image cross-intent group、104 条 query；新增 challenge 只解释边界，不代表自然总体、不参与 gate。 |
| test300 是“未见测试集” | 修正 | test 的标签、来源和文本 shortcut 已被只读审计；它只能称为“冻结的 response-level execution holdout”。 |
| 旧运行向模型暴露 `image_path` | 修正事实 | 200×5 的 provider DTO 将路径映射成常量，模型没有看到真实路径；source/path shortcut 仍是数据偏差诊断，但不能据此声称旧 provider 输入泄漏路径。Core 仍应以 opaque `asset_id/text/turns` fail closed。 |
| 单一 J 分数作为主指标 | 降为历史桥接 | 旧绝对 Judge 方差和成本过高；主指标改为确定性的 Grounded Contract Success，pairwise、人评和 legacy J 为次要证据。 |
| Core 图片公开展示 | 禁止 | 本次批准只允许私有云端模型处理，不允许公开 demo、再分发或推导上游版权许可。 |

## 2. 200×5 基线与统一口径

200×5 的可信回顾主口径是未参与 treatment gate 的 175 条 evaluation 样本：

| Config | n | Mean J | Route accuracy | Card compliance |
|---|---:|---:|---:|---:|
| NoSkill | 175 | 68.706 | N/A | 73.7% |
| LLMStaticSkill | 175 | 69.937 | 80.0% | 81.7% |
| S1 | 175 | 71.131 | 83.4% | 85.1% |
| S1+S2 | 175 | 72.080 | 92.0% | 90.3% |
| Full | 175 | 70.940 | 92.0% | 90.3% |

关键结论：

- `S1+S2 - LLMStaticSkill = +2.143 J`，paired leakage-group bootstrap 95% CI 为 `[-1.111, 5.422]`；它是方向性信号，不是稳定总体增益。
- 清除 hard/evaluator/tool anomaly 后，`product.multi_search` 的 complete-case 增益约 `+13.79 J`，是当前最强机制信号。
- `Full` 和 `S1+S2` 使用同一 Bank；表面差异来自 Assistant/Judge 重采样，不能归因为 S3。
- 旧 S1 只读取带 gold capability/intent 的 25 条轨迹，没有读取基线失败、parent Bank 或 failure attribution；准确名称是 trajectory-conditioned supervised Creator，而不是真正的失败驱动自进化。
- 200×5 settled 成本约 CNY 142.18，含未决 reserve 的 accountable 上界约 CNY 151.10；旧计划中的 CNY 133.85 不再作为预算依据。

后续所有报告同时保留三种互不替代的视图：

1. `raw`：冻结输入上所有可评单元；
2. `complete-case`：按预先冻结的程序/评测异常规则配对剔除；
3. `sensitivity`：展示 parser 修复、hard error 和 tool anomaly 对结论的影响。

不允许用 clean/sensitivity 结果替换 raw headline，也不允许把 optimization 样本混入冻结 holdout 的因果口径。

## 3. 数据协议与 test 边界

Core r3 使用代码中的实际 split 名称，固定为 `dev_mini=200 / opt_pool=800 / val=200 / test_frozen=300`。外层 capability 矩阵不重切：

| Split | 用途 | 总数 | 百科 | 精确 | 多商品 | 风格 | 文档 | 菜谱 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_mini` | 工程 smoke、回归、少量人工检查 | 200 | 35 | 35 | 35 | 35 | 30 | 30 |
| `opt_pool` | 失败发现、候选生成、内部 replay | 800 | 163 | 209 | 117 | 163 | 32 | 116 |
| `val` | 候选接受/回滚 | 200 | 41 | 52 | 29 | 41 | 10 | 27 |
| `test_frozen` | 最终冻结比较，只执行一次 | 300 | 61 | 79 | 44 | 61 | 18 | 37 |

整体 capability 比例为 Exact 25%、Multi 15%、Style 20%、Encyclopedia 20%、Document 6%、Recipe 14%。这是从注册 full-profile mixture 缩放得到的 Portfolio benchmark 分布，不是生产流量估计。

当前 1,500 条包含 867 个 leakage component、60 个完整 25-query generator batch 和 56 个 template family；leakage component、generator batch、template family 三种 grouping 均无跨 split 违规。25-query batch 和 connected grouping 必须原子分配，同图多问、cross-intent、同模板或同生成批次不得拆到不同 split。已有 263 条 boundary（17.53%）和 38 个 same-image cross-intent group（104 条 query）。

外层不重切的原因是：`opt_pool` 已用于 routing surrogate，既有 launch/hash 也已经冻结；现在移动 query 会破坏实验连续性，并可能被理解为看过诊断后调整 holdout。现有不均衡通过内层抽样、分层 Gate 和报告缓解，不通过事后换样本解决。

四个 split 的职责固定为：

- dev200：只用于既有诊断、pairwise rubric 校准和极小 smoke；不再生成新候选。
- opt800：唯一的 failure 收集、局部模型训练和 S1/S2/S3 候选生成集。
- val200：唯一的候选选择、阈值选择、交互检查和接受/回滚依据。
- test300：只在算法、Bank、runtime、rubric、seed、并发、重试和预算全部冻结后解封一次。

### 3.1 Val 内层 Gate：统一为 `75/75/50`

统一并沿用 `route_gate=75 / body_gate=75 / shadow_val=50`。旧计划中的 `80/80/40` 与 25-query generator-batch 原子性冲突，代码会显式拒绝；75、75、50 分别对应 3、3、2 个完整 batch。三个 Gate 的 leakage component、boundary group、template family 和 generator batch 均不得互相拆分或重叠。

既有 assignment 已落盘于 `E:\skillchain-data\runs\portfolio-core-20260804-r2-validation-gates.json`：200 条、seed `20260805`，文件 SHA-256 为 `2f099f5a4efa9589f6948a1fec1419fd9024acdd4069f01195b6c613d2513f11`，并绑定 plan、catalog、capability assignments 和 realism manifest SHA。首个 val response 前必须复核这些 bytes 与 manifest 绑定并直接复用；不得因新增的分层诊断重新生成或优化 Gate assignment。

S2 只能使用 `route_gate`；S1/S3 共用 `body_gate`，但必须在任何候选评分前冻结两阶段合计的候选数量、比较顺序和一次接受/回滚规则；`shadow_val` 只做最终组合 go/no-go，不再据此生成或改写候选。五配置全 val 报告只能在最终 Bank 冻结后生成；产生 response 后不得重排 Gate。

### 3.2 Opt 内 component-aware discovery/replay

S1 候选生成使用预选的 240 条 component-unique 代表集；其余未参与生成的 component 才可用于 opt replay。S2/S3 按 failure cluster 分组时，同一个 leakage component、generator batch 或 template family 必须整体进入 discovery 或 replay 的同一侧，不能让同图、同模板或同批次信息跨两侧。内部 fold assignment、seed、component 清单和 SHA 在任何候选评分前冻结。

Opt replay 只用于淘汰明显过拟合候选；正式 accept/rollback 仍只由对应的 `route_gate` 或 `body_gate` 决定。不得根据 replay 或 Gate 结果重新抽样 discovery、改 failure cluster 或移动 component。

### 3.3 分布偏差与分层评估

现有外层 split 不是完全同分布：Document 在 val/test 仅 10/18 条，dev 的 30 条全部为 Wikimedia 普通文档，而 opt/val/test 的 60 条来自 CORD/SROIE 收据；Document 因此继续标记为 exploratory，不作强 per-capability claim。语言 repair 比例为 dev 12.5%、opt 17.2%、val 27.0%、test 11.0%，其中 val 的 Multi repair 为 14/29、Document 为 6/10。文本模板可预测四个 split 的 balanced accuracy 约 0.475，dev-vs-val 约 0.814，Document dev-vs-tail 约 0.967；这些是分布指纹，不是直接模型泄漏，因为 split/source/template/repair 元数据不得进入 provider DTO。

Gate、val/test 主表和失败分析至少同时报告 capability、boundary、source、repair/non-repair、style submode、same-image cross-intent 六类切片。总体 headline 使用 capability-macro Grounded Contract Success，同时给出 query-micro；置信区间继续按 leakage connected component bootstrap，不能让 Exact 的样本量或 Document 的小样本直接支配总体结论。

只有未来新建 Core r4 时才重新设计外层 split：提高 Document 在 val/test 的样本量并平衡 Wikimedia/CORD/SROIE，使 repair 比例更接近，并预先按 `capability × source × repair × boundary × style_submode` 分层，同时保留 leakage component、template family 和 generator batch 原子性。r4 必须重新生成全部 split SHA、launch 和冻结声明，不能覆盖或改写 r3。

test 的准确披露为：

> test300 的标签、split/source 分布和文本 shortcut 已经接受过只读审计；尚未运行五配置 response-level evaluation，也没有使用 test response、得分或逐题成败生成候选、选择阈值或执行 accept/rollback。

因此，test 是冻结的 response-level execution holdout，不称为真正 blind 或完全 unseen。立即停止进一步的 test shortcut probing、切片调参和候选比较。

主结果按五个 eligible capability 做 capability macro；`utility.document_reading` 在 val/test 只有 10/18 条，只作 exploratory，不作强 per-capability claim。统计按 leakage connected component 聚类，不把同图多问或重复运行当成独立样本。

已有 38 个 same-image cross-intent group、104 条 query 作为主要机制挑战切片。只有在最终算法和 rubric 冻结后，才可另建 100 或 125 条 curated stress test；它不参与候选生成和 gate，且只运行 Static 与 Final，单独报告描述性结果与 cluster CI。

## 4. Core 云端处理授权

项目所有者于 2026-08-06 明确批准 Core 可上传云端用于本次私有 Portfolio 模型推理。授权记录见：

`specs/data_sources/c2/portfolio-core-remote-processing-v1/owner-authorization-v1.json`

授权范围是 Core v9 catalog 的全部 1,022 个资产，而不是只覆盖 1,500 条 query 直接引用的 867 个资产。原因是检索 gallery、候选证据和 embedding 可能读取其余 155 个 catalog 资产；运行时必须继续证明实际访问没有超出这 1,022 个资产。

允许的逻辑处理器严格限定为：

- `dashscope-qwen-assistant`
- `dashscope-kimi-feedback`
- `dashscope-kimi-judge`

这项 owner decision：

- 允许上述处理器接收 Core 图片并执行远端模型推理；
- 不修改基础 catalog，基础 catalog 当前仍为 `true=0 / false=595 / unknown=427`；
- 不授予公开展示、GitHub 图片发布、再分发、上游许可替代或 Formal Research eligibility；
- 不自动批准 Core 的模型调用费用或正式启动；
- 更换 provider/逻辑处理器、扩大资产集合或公开展示时必须重新授权。

Gate 0 必须用 create-only 方式生成 permission-only overlay、receipt，并分别对三个 processor 完成 runtime preflight。只有 overlay 的 1,022 个资产全部为 `cloud_upload_allowed=true`、`public_demo_allowed=true` 仍为 0，且资产 ID、图片 SHA、路径绑定和 leakage components 与基础 catalog 完全一致，才视为运行时授权链关闭。

公开面试材料只使用脱敏文字案例、指标图、架构图和示意图。如需真实图片 demo，另建小规模、明确 `public_demo_allowed=true` 的独立资产集；不得让该展示集参与 headline metric，也不得按结果挑图制造增益。

## 5. 主指标和证据层级

### 5.1 Primary：Grounded Contract Success

每条 query 的主指标为二元成功，必须同时满足：

```text
acceptable route
+ no hard error
+ capability-specific tool correctness / coverage
+ evidence grounding
+ required card / output contract
```

主报告使用 capability-macro Grounded Contract Success，并同时给出 query-micro 值和六 capability 明细。Contract Success 的字段、tool oracle、容错和缺失处理必须在任何 opt/val response 评分前冻结。

### 5.2 Secondary

- route macro-F1、accuracy、per-class recall、confusion matrix 和 McNemar；
- route-correct-only Body quality、tool coverage、evidence/card compliance、hard error；
- boundary、same-image cross-intent、source、repair 和 style submode 切片；
- Assistant/Judge token、调用率、fallback、成本和延迟；
- Static vs Final 的 blind pairwise 与 100 条人评。

旧 `J_project` 只在最多 50 条预冻结样本×5 配置上作为 200×5 历史桥接，不再作为 candidate gate 或 headline。Pairwise Judge 必须先通过顺序交换和人类一致性校准，否则降为 exploratory。

### 5.3 结论分级

- **机制增益**：某个冻结 capability/slice 明确改善且失败链可解释；可以作为案例，不外推为总体增益。
- **系统增益**：Final 相对 Static 的 capability-macro Contract Success 至少 `+2pp`、cluster bootstrap 下界不低于 0、hard-error 差值不超过 `+1pp`，且没有 capability 明显退化。
- **稳健语义质量增益**：除系统增益外，人类 tie-adjusted preference 点估计至少 `0.55`，且 95% CI 下界高于 `0.50`。未满足时只报告方向性或工程/成本收益。

`+5pp` Contract Success 和 `+3pp` route macro-F1 是面试导向的目标值，不是允许事后移动的成功定义。

## 6. Gate 0：在算法轮次前关闭 P0

Gate 0 直接关闭比较无效、Core 无法启动或面试可信度受损的问题：

1. **统一运行和恢复契约**：shard 永远按 `output_relpath` 落盘；历史 fallback 只显式读取；同一 run/shard 幂等恢复且不重复调用。
2. **修复可恢复 Judge JSON**：只剥离完整包裹单一 JSON object 的 Markdown fence，再执行原 strict schema；保留历史 fixture。
3. **修复 hidden-evaluation 假阳性**：保留 `find. EYE MASK FULL TREATMENT` 回归，不改写历史分数。
4. **修复 no-op/rollback 因果语义**：S3 回滚时 Full alias 已接受的 S1+S2 artifacts，主表增量为 0；被拒 candidate 仅作诊断。
5. **硬化 route contract**：最小 JSON schema，只对格式失败做一次等价重试，保存原响应、原因和 receipt。
6. **实现 split-aware Core 入口**：loader、runtime、launch、shard assignment、monitor 和 analyzer 不再硬编码 dev_mini；provider 可见输入只含 opaque DTO，不含真实路径、source、template 或 label。
7. **完成云端 overlay 链**：适配 Core selection schema，生成 create-only catalog/receipt，并对三个 processor 做逐一 preflight。
8. **统一公共 contract renderer**：所有配置共享 card/evidence 的最低结构保证；Skill 只决定语义与策略，避免把格式 bug 当作 S3 增益。
9. **验证工具上限**：
   - Style 先做 30–50 条 tool-ceiling，区分 same-category alternative 与 cross-category coordination，返回有 provenance 的真实相似度/相对属性；硬编码前三条和固定分数不能进入 Core。
   - Multi 使用 typed item mapping，逐物体保留 `matched/unresolved`、去重、数量和 card 对应；formatter 确定性保证映射与长度预算。
10. **冻结成本 BOM 和评测契约**：确定模型、token 上限、并发、重试、Contract Success、pairwise rubric、random seed 和 test 解封事件。

Gate 0 出口：聚焦测试通过；零 provider 的两 shard dry-run 可写入/恢复/重跑；25-query provider canary 无重复调用；Assistant/Judge hard error 总体 `<0.5%`、任一 capability `<1%`；工具 ceiling 达到预冻结覆盖门槛；三个 processor 的授权 preflight 全通过。

未满足 Gate 0 不进入 opt800 算法轮次。这里不新增 formal gate、许可证工程或额外 test probing。

### 6.1 Gate 0 完成记录（2026-08-08）

- v5 canary 完成 25 queries × 5 configs，共 125 个逻辑结果；四个物理 shard 全部 25/25，Full 为 S1+S2 的零调用 alias。
- Assistant hard error、Judge hard error和工具错误均为 0；OCR 从 v3 的 16/16 物理失败修复为 v5 的 16/16 成功。
- provider ledger 为 415 reservations = 415 settlements，0 forfeits、0 unresolved；本 execution root 实际结算 CNY 14.13020855，含历史累计 accountable CNY 57.99552440。
- 冻结复跑的 provider 调用和 ledger event 增量均为 0，checkpoint/terminal bytes 不变；最终 receipt 为 `runs/portfolio/core-gate0/canary-evidence-v5-clean/final-receipt.json`，文件 SHA-256 为 `1c2dd6796eabdba52d9b91f117080a903d4aec317072da629fb6e0d7281fa0de`。
- 诊断结果中 S1 的 `J_project=71.36`，相对 Static 的配对均值为 `+5.94`；S1+S2 为 `67.92`，相对 S1 为 `-3.44`。25-query bootstrap 区间均跨 0，只用于算法定位，不作为 Core headline。
- 唯一 audit review flag 是 26 个逻辑 card 违规（去掉 Full alias 后为 21 个物理违规）；根因是 card-required 的 Style/Exact 工具没有返回 grounded candidate，而不是 renderer 丢卡。该项进入 opt800 前的数据/工具闭合，不回写 Gate 0 结果。
- post-fix runtime 为 `runs/portfolio/core-gate0/treatment-runtime-rebind-v7`；`opt_pool=800` 的零调用 launch 为 `runs/portfolio/core-gate0/opt-launch-v1-ready`。下一付费阶段必须先增加 Static-only 精确授权范围并冻结新 BOM，不直接执行现有五配置 4,000-row launch。

### 6.2 Style 跨品类搭配分支：真实证据、能力边界与重锁要求

针对 Style 中“给这条裙子配鞋、包或首饰”的跨品类请求，当前已实现独立的 `cross_category_coordination` 检索分支；原有 same-category alternative 继续走既有视觉相似路径，不能把跨品类候选伪装成“相似商品”，也不能用跨品类规则改变同类检索结果。

当前图资产是 query-independent 的：构建器只读取 Core selection、runtime catalog、ABO listing metadata、精确图片字节和 Codex 复核 seed，显式拒绝 `query_id/split/user_text/query_text/config/score` 等字段。Codex 结合 ABO metadata 与精确 Core 图片的可见属性，复核并保留了 9 个女性候选，其中鞋 3 个、包 2 个、首饰 4 个；两双男鞋被明确排除，另有 1 条女性项链因与 `test_frozen` query asset 重叠而在建图前排除，防止候选角色与冻结评测角色污染。130 个 FashionIQ `dress` anchor 通过冻结的 deterministic palette rule 与这些候选生成 872 条 exact anchor→candidate edge，其中 `footwear=290 / bag=260 / jewelry=322`；graph SHA-256 为 `1cecea70ba7f2c5a992d1fb773bba8d04043ee84e8dec8c2e362d70996b8258a`，边上保留类别、颜色、款式等可追溯 facet 与置信度。

这批 edge 的准确表述是“ABO 元数据 + Codex 复核可见属性 + 确定性配色规则产生的 Portfolio curated coordination evidence”。它不来自 Polyvore、FashionIQ 或 ABO 的原生 outfit 共现、共购或搭配金标，因此不得声称学习到了真实 outfit compatibility。当前只支持 `footwear / bag / jewelry`；用户明确请求未覆盖类别时必须 fail closed，不能返回其他类别凑数。

该分支改变了工具语义与 Core source set，旧 runtime-sources receipt、treatment runtime lock 和 launch 只能作为历史证据，不能直接绑定下一轮 Core 执行。正式启动前必须 create-only 生成包含 coordination graph 的新 source receipt，复验 selection/catalog/ABO archive/seed/graph self-hash，并据此重新锁定 TaskSpec/ToolSpec、Bank、runtime 和 launch；不得原地改写旧 lock。

工具契约与真实数据 smoke 已验证：实现对颜色、feature、否定约束和请求中全部 family 做 fail-closed 过滤；cross-category 公共 DTO 不再输出没有校准依据的伪“相关度”。一次 `dev_mini`-only、零 provider、零 network 的真实 tool-runtime smoke 覆盖了 generic 请求返回鞋/包/首饰三个 family、黑色低跟鞋命中 exact 候选、白色低跟鞋返回 0、明确否定包返回 0、same-category mode 不回归，以及“鞋 + 外套”混合请求因外套不受支持而拒绝 partial DTO、整体返回 0。`val` 与 `test_frozen` 未用于调图、挑候选、调阈值或生成响应。

剩余边界是 Core fresh source receipt/relock 与 provider 实验：在包含 coordination graph 的新 receipt、TaskSpec/ToolSpec、Bank、runtime 和 launch 全部重新锁定前，不创建付费 Core control。本节证明的是数据图、工具分支和 dev-only 本地 smoke，不是 Assistant 端到端模型增益，也不把尚未运行的 Core 指标写成已验证结果。

在仅将 dev/opt 文本送入 matcher 的全量复核中，198 条 Style 被稳定分为 81 条 cross 与 117 条 same-category；81 条 cross 的 runtime 与 ceiling 候选及顺序逐条一致，差异为 0，其中 34 条有 exact graph evidence、47 条诚实返回无覆盖。这是工具层覆盖与契约一致性结果，仍不等价于 Assistant/Judge 增益。

## 7. 三轮算法优化

### Round 1：S1 Failure-driven Creator

输入只来自 Static 在 opt800 的冻结 Query、深验 schema-v2 checkpoint/sidecar、GCS v2 失败类型、同 capability 成功或近通过 anchor 和 parent Static Bank。gold capability 只用于 attribution/evaluation，不作为“答案提示”；val/test response、test score 和私有 scorer payload 不进入 Feedback 或 Creator 的模型投影。

当前冻结 Static 基线为 execution-v6：800/800 schema-v2 checkpoint 与 sidecar，GCS headline available，capability-macro `22.25%`、query-micro `28.13%`、hard error `1/800`。优先失败簇为 material citation 250、card contract 148、Style evidence 109、Multi mapping 83；这些只用于冻结 discovery 选择和诊断，不改 GCS policy、response contract、split 或标签。

```text
Static opt800 rollout
→ Creator240 所在 discovery600 中确定性选 36 个失败 + 12 个 anchor
→ Qwen Feedback：6-call canary + remaining42，exactly once
→ typed Feedback bundle
→ Codex S1 Creator 一次生成完整六 Skill Bank
→ opt replay200 过拟合筛查
→ body_gate75 Static/S1 配对 GCS system gate
→ 整 Bank 接受或 byte-exact 回滚
```

本轮明确采用现有 Creator 的整 Bank 输出语义：六能力统一接受或统一回滚，不声称 capability-local patch、局部组合或独立 capability 接受。Feedback 只对冻结 48 条代表样本运行真实 DashScope `qwen3.7-plus-2026-05-26`（思考模式、strict JSON Schema），并把结构化 `rule_violations / ideal_response_gaps / skill_suggestions` 聚合成 typed bundle；Creator 只调用一次 Codex CLI `gpt-5.6-sol/high`，无工具、无修复、无重试。用户、Assistant 和 Feedback 文本均为不可信诊断证据，当前 AuthoringPacket 是唯一规范来源。

48 条选择只来自既有 Creator240：每能力 6 个失败与 2 个 anchor，同时保持 `query_id`、raw `leakage_group_id` 和 `asset_id` 唯一；明确区分 32 个 GCS `atomic_component_id` 与 411 个 raw leakage group。选样、失败优先级、source/repair/boundary/style-submode 轮转和 hash 决胜在任何 Feedback 调用前冻结，排除 execution lapse `r2-core-0567`。DashScope Qwen 授权只覆盖这 48 个精确 query/asset/image 绑定，不修改历史授权。

opt replay 使用已冻结的 8 个 batch、200 条未参与 Feedback/Creator 的 query。Static baseline 复用已深验的 execution-v6 对应行，只新跑 S1 候选；通过条件为 capability-macro delta `≥0`、hard-error delta `≤+1pp`、任一 capability delta `≥−3pp`、oracle coverage 完整且无程序或完整性错误。Replay 只淘汰明显过拟合，不构成正式接受。

Replay 通过后，在既有 assignment SHA `2f099f…3f11` 的 `body_gate=75` 上重新运行 Static 与 S1，共 150 条 Assistant rows。正式整 Bank 接受门沿用 GCS system gate：macro delta `≥+2pp`、leakage-component bootstrap 95% CI 下界 `≥0`、hard-error delta `≤+1pp`、每个 capability delta `≥−3pp`。未通过即发布 byte-exact rollback receipt，后续共同父基线仍为 Static。Gate 结果不得反哺选样、Feedback、prompt 或候选。

角色冻结为 Qwen Feedback、AIFast `gemini-3.6-flash` Final Judge；但本轮 S1 接受门只使用确定性 GCS v2，因此 Gemini Judge、Pairwise、legacy Final Judge、shadow_val、test_frozen、S2 和 S3 均为 0-call。Pairwise 若后续需要，只能在进入其他 val 比较前另行校准；不能作为本轮 S1 接受门的隐藏前置。

#### 2026-08-09 实施与 canary 终态

- 真实 execution-v6 与 Creator240 的确定性选择已冻结为 48 条：36 个 failure、9 个 success anchor、3 个 partial anchor；六能力均为 `6 failure + 2 anchor`，`query_id / leakage_group_id / asset_id` 三重唯一均为 48/48，selection SHA 为 `b9fa97bdc7e43cfe4f99c9ee00f3103c95c22be3608b219a59c0d9fc2aa8c21a`。
- 新增的生产链包括：深验 Static GCS corpus、selected48-only AIFast authorization、调用前 create-only reservation、orphan fail-closed、typed Feedback bundle、当前 Style 2.3 Creator 输入、一次性整 Bank Creator、Creator receipt→two-Bank runtime lineage、replay200/body_gate75 executor、manifest-bound GCS export 和 byte-exact accept/rollback Gate。Pairwise、Final、test_frozen 仍为 0-call 禁止项。
- 更新后的零 provider 双跑在真实 opt artifact 上通过：两轮 selection、Feedback model projection 与 candidate Bank 字节一致；外部 Feedback/Codex/Assistant/网络调用均为 0。权威报告为 `runs/portfolio/core-s1/s1-zero-provider-smoke-v4/smoke-report.json`，文件 SHA-256 为 `a5f755e50d78063ab8308e7703e71ba1587571781c876c59157f67a02e276af3`。
- Feedback canary v1（独立 create-only run）调用 4 条后停止：3 parsed，`r2-core-0232` 返回单个完整 JSON fence，旧 exact-JSON parser 记为 terminal `invalid_feedback_json`。旧 artifact 未重解析或改写。随后 parser 升为 `visual-feedback-complete-fence-wrapper-v2`，policy SHA 为 `09196a532b6369a24bebabfcc1916c6714a9066e127037c97ef9b022343fa9ee`，只增加一个完整外层 fence 的支持。
- 新授权、新 identity 的 canary v2 完成 6 条后再次停止：5 parsed，`r2-core-0497` 的四个必填字符串各含一个前导 ASCII 空格，违反冻结 schema 的 trimmed-string 约束。两次 canary 合计 10 次 AIFast 调用、输入 28,547 tokens、输出 12,791 tokens；没有替换样本、同 identity 重试或 Creator 调用。
- 用户随后批准 parser v3 与累计 58 次调用上限。v3 仅对 `summary / description / evidence / skill_suggestions` 自由文本叶做确定性 trim，parser policy 为 `visual-feedback-free-text-trim-v3`，SHA 为 `d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016`；trim 后空值、重复和其他 schema 漂移仍严格拒绝。v3 selected48 授权与零调用 dry-run 均通过，但 canary 第一波 2 条均为 terminal parse error 后按约停止：`r2-core-0202` 把 `skill_suggestions` 生成为 finding 对象数组，`r2-core-0945` 额外回显顶层 `output_contract`。两条均是合法 JSON 但不符合冻结 schema，不属于 trim 问题，也未被修复、替换或重试。
- v1/v2/v3 合计 12 次 AIFast 调用，输入 33,050 tokens、输出 15,931 tokens；12/12 usage 已知，reservation 与 settlement 分别精确为 4/4、6/6、2/2，无 orphan。仍没有生产 Feedback bundle、Creator、candidate Bank、replay 或接受/回滚产物。
- 用户批准 prompt-only v4 与累计 60 次调用上限后，独立 selected48 canary v4 已真实执行并按冻结条件停止：6/6 reservation 与 settlement 闭合，5 parsed；`r2-core-0424` 返回一个本可通过 schema 的完整 JSON object，但其后追加了 `\n[`，因此 parser v3 正确记为 terminal `invalid_feedback_json`。v4 共输入 17,817 tokens、输出 7,829 tokens；没有第 7 次调用、remaining42、bundle 或 Creator。v1-v4 现合计 18 次 AIFast 调用、输入 50,867 tokens、输出 23,760 tokens，全部 usage 已知且无 orphan。Output contract v4（SHA `da9c639f0e7e179642ec32471c29bc38238d8e6d150c92e5a67f9ef38cd9d5c6`）已修复此前两类 schema 形状错误，但 prompt-only 仍不能保证传输层只产生一个 JSON value；v4 artifact 永久保留为 terminal evidence，不能重解析、重试或与历史 run 拼接。继续真实 Feedback 需要另一个显式授权的新 run identity；在此之前不得启动余下样本、bundle 或 Creator。
- 用户批准 JSON-object transport v1 与累计 66 次上限后，selected48 v5 的固定6条 canary 为 6/6 parsed，证明 AIFast 接受并兑现 `response_format={"type":"json_object"}`，且历史失败样本 `r2-core-0424` 不再产生尾随语法垃圾。随后同一 identity 逐字节复用 canary 并进入 remaining42，但在第8个新增样本 `r2-core-0467` 处按冻结条件停止：该响应是语法合法的单一 JSON object，却再次回显禁止的顶层 `output_contract` 并把 `schema_version` 写成2，因此 parser v3 正确记为 terminal `invalid_feedback_json`。v5 精确为 14 calls / 13 parsed / 1 parse error，输入 40,267 tokens、输出 16,790 tokens；没有第15次调用、bundle 或 Creator。v1-v5 累计32 calls、91,134 input、40,550 output；JSON-object mode 只解决语法 framing，不能保证业务 schema。v5 不得重试、重解析或拼接；下一步必须在“新增严格 JSON Schema transport 的新授权 run”与“明确放宽48/48、使用部分 Feedback+GCS 聚合”的 Portfolio 取舍中择一，不能静默改变本轮合同。
- 2026-08-09 用户在 partial13 bundle 尚未发布、Creator 与 S1 replay 均未启动时改变 evaluator 方案：后续 Feedback 改用百炼/DashScope `kimi-k2.6`，final Judge 改用 AIFast `gemini-3.6-flash`。`model-role-selection-v6` 是前向新身份，v1-v5 Feedback run、v5 role selection、旧 Kimi Judge receipt/ledger 均保持历史字节与原语义。现有 Core v1 owner authorization 已明确覆盖 `dashscope-kimi-feedback`，但 selected48 AIFast Feedback authorization 不能转授权给任何 Judge；`aifast-gemini-judge` 必须在新的 Core schema-v2 owner authorization、create-only overlay/receipt、BOM/价格与 provider token ceiling、runtime source/lock/launch/control 全部重锁后才允许真实调用。当前仓内没有可信 AIFast Gemini Judge 价格，因此预算预留 fail closed；本次角色交换实现与验收阶段 provider 调用为 0。
- 随后的零调用失败审计表明，前向 Kimi Feedback 的两条失败响应系统性复制了输入 packet 的 `schema_version=2`，其中一条还为 finding 生成了 5 条 evidence。未重解析、修复或放宽 parser-v3；`model-role-selection-v7` 在任何新 provider call 前取代 v6，绑定 cache-v8 / response-schema-v1 prompt-v5 / Kimi plain-JSON transport-v3，并明确忽略输入 schema 2/3、响应必须使用整数1、每项 evidence 严格为1–4条。output contract-v4、parser-v3、v1-v6 run/result/control 字节与终态保持不变。

#### 2026-08-10 Qwen Feedback 与整 Bank S1 终态

- `model-role-selection-v8` 最终冻结 Qwen Feedback / Gemini Judge。Qwen `qwen3.7-plus-2026-05-26` 的新 identity 按 `6+42` exactly-once 完成 `48/48 parsed`，48 次均启用思考模式（budget 2,048）和 strict JSON Schema；输入/输出为 `161,326 / 74,197` tokens，按冻结价格复算 CNY `0.916228`。typed bundle 文件/自哈希为 `15adcfee664f14d0bca215ce1f22351168a95d287c69cee2094a4b77a426fdb8 / fe1572aed7d7260a4af9c5afde906780616e8895fe887e8a30939bee517a860b`。
- Codex Creator v5 恰好一次调用 `gpt-5.6-sol/high`，输入/输出 `95,857 / 1,896` tokens，0 retry、0 repair、0 follow-up、0 tool activity；生成六能力整 Bank 候选，文件/Bank 自哈希为 `a86494fe55120ce05bc2e5a50a396a45983aca4e784d9c98d8330676c2abd5a9 / a6d8613c730297be96d7e2734b93e333ed65660fa2e53cb4ca650abcef1d9d9f`。
- Replay200 完成 200/200 Assistant checkpoint、667 次已结算 DashScope 模型调用、0 forfeit，费用 CNY `0.282965400000`；本轮 Feedback+replay 的 DashScope 合计 CNY `1.199193400000`。Static→S1 的 GCS macro 为 `21.7729% → 25.2749%`（`+3.5021pp`），micro 为 `26% → 31%`（`+5pp`），hard-error delta `0pp`，coverage/integrity 完整；但 Encyclopedia 为 `7.5% → 2.5%`（`−5pp`），低于预冻结的 `−3pp` 下限，故 replay report 为 `failed`（文件/自哈希 `f3dca1509b4af8615687349d84b84ae87215fad3646c4a20188afb3642a8afa7 / d73029a19947576385a9721d4df134a21f0b59c83ff190706917e37f46f506e7`）。
- 单一 disposition 为 `rolled_back`，basis=`replay200`（文件/自哈希 `c194b5413692678d490f04d0ed9033be67f8430f927dcaa2f168a2ac9843a419 / 8a39d58bf8ece75cb0f587e7637393e418b9a61293bfaaebb7cb6b749b04af9d`）。`output-bank.json` 与 parent Static Bank 逐字节相同，文件 SHA 均为 `64942d068519246eac9d9e6f49ac1d0734196d1aead6aba294aeb00d635cfa6d`；候选仅保留作诊断。因为 replay 未通过，body/val 目录未创建，Gemini Judge、Pairwise 与 Final 调用均为 0。
- 本轮关闭了三个 P0：Creator v4 暴露了 authored-prose scanner 的词法契约缺口，v5 前向绑定精确 forbidden whole-word guard；two-Bank runtime 仅在派生边界前向绑定当前预算/价格合同，同时把 immutable parent 限制为 evidence-only；离线分析仅允许精确 runtime-v5/control 身份进入 immutable-evidence 分支，不能把历史 source lock 当成新的执行能力。历史失败 artifact 均未重写。

#### 2026-08-12 forward-only disposition：Core Fast R1

本节只约束 2026-08-12 之后的新 Portfolio 运行，不重命名、重算或覆盖 2026-08-10 的历史闭环。为便于叙述，已完成并回滚的 whole-bank 首轮记为 `R0`；下一次新身份的 sparse S1 候选记为 `R1`。仓内 Core r3 的默认实验入口前向切换为 `scripts/run_core_experiment.py` 与 `specs/core-experiment-fast-v1.json`，历史 Formal Core 入口继续只读兼容，不再作为 Portfolio Quickstart 前置。

**R0 的样本级诊断。** Encyclopedia `3/40 → 1/40` 的两条 Static-success 回退已经定位：`r2-core-0695` 因候选 fallback 漏掉精确 `not enough evidence` marker 而得到 `fallback_contract_failed`，并伴随跳过 `object_detect` 和无来源事实陈述；`r2-core-0796` 的工具序列与 fallback marker 均保留，但输出前多出 `<|begin|>`，得到 `output_section_invalid`。前者与候选改写后的条件式 object-detect/fallback prose 相符，后者是独立的 Assistant 格式 lapse；现有证据不能把两者都归因于同一个 Skill 根因。旧 Feedback 中另有要求在无检索证据时补充栖息地、危险近似种或 studies 的建议，与 cited-evidence 合同冲突。这些事实支持 sparse 隔离与 policy filter，不支持降低 `−3pp` 门或事后拼接 R0 的赢家。

**R1 的唯一允许变化。** R1 保持模型、GCS、样本标签、fold、Gate 和 Static parent 不变，只把 S1 Creator 从 whole-bank rewrite 改为 parent-bound `inherit|patch`：

- 六个 capability 必须逐项绑定 parent Skill SHA；默认 `inherit`，继承项 byte-exact，不重新渲染；
- 只允许 `[policy_compatible]` Feedback suggestion 触发 patch，`requires_new_evidence` 与 `rejected` 不进入 Creator 的可执行建议；
- 全部 Description/objective 冻结，最多 patch 3 个 capability；未经有效 Feedback 支持的 capability 必须 inherit；
- Encyclopedia 在 R1 中强制 inherit，既不改 Body 也不允许因其他能力的改进发生跨 Skill 文本漂移；
- sparse draft、Feedback bundle、AuthoringInput、parent/candidate Bank 与编译 receipt 全部 SHA 绑定；字段越界、工具序列漂移、非法来源或 resume artifact 漂移均 fail closed。

**Development 与接受必须分开。** 固定 `discovery600` 只用于 canary12 Feedback 和候选生成；已观察的 `replay200` 只作 development 筛查，条件仍为 macro delta `≥0pp`、hard-error delta `≤+1pp`、每 capability delta `≥−3pp`。只有通过 replay 的冻结候选才能访问一次 `body_gate75`；接受条件为 macro delta `≥+2pp`、leakage-component bootstrap 95% CI 下界 `≥0pp`、hard-error delta `≤+1pp`、每 capability delta `≥−3pp`。body gate 的结果不得反哺 Creator、样本或阈值；通过则接受，失败则 byte-exact 回滚并停止该候选线。

用户批准在历史 R0 之后最多再做五轮 S1 优化，而不是把它们当作默认调用额度：最多允许 R1–R5 五个有记录的 development 候选。第一位通过 replay 的候选立即冻结并进入唯一一次 body gate，之后无论接受或回滚都停止 S1；没有通过 replay 才能在新 round identity 下继续下一轮。本段仅对前向 S1 development 取代本文件开头的历史“三轮”总述；它不扩张 S2/S3，也不授权换样、provider 重试挑优、重复访问接受门或访问 test。下方 R1–R3 是同日较早的旧 Assistant lineage；随后用户明确要求切换 Qwen3.7、重跑 Static，并完整使用 R1–R5。当前终态以再下一节为准。

**新运行的容量绑定。** Assistant `qwen3.7-flash-2026-07-15` 使用 worker cap 60、`20 requests/s`，并在每次真实 route/action/body HTTP 调用前配速，而不是按 query 配速。60-task 极限探针曾以 `60 requests/s` 达到实际峰值并发 60、60/60 成功；生产速率降为 20 requests/s，为账号级 5M TPM 限额保留至少约 20% 的保守余量。Qwen3.8 Feedback 的验证配置为 worker cap 60、`8 requests/s`；R1 固定 batch 仅 12 条，所以有效 worker 数是 `min(60,12)=12`。两组数值只写入新 Core Fast execution overlay；后文 8.4 及历史 freeze profile、authorization、receipt 中的旧并发/速率仍按原语义保留，不回写。容量证据分别为 `docs/qwen37-flash-assistant-concurrency-benchmark-20260812.md` 与 `docs/qwen38-feedback-concurrency-benchmark-20260812.md`。

#### 2026-08-12 R1–R3 execution ledger / final disposition

本节是前向追加的实际执行记录，不改写 R0。`R1-v1/v2` 是同一 R1 的执行修订，
`R2-v1/v2/v3` 是同一 R2 的执行修订；版本号不能计作新增算法 round。R2、R3 均逐字节
复用 source manifest SHA
`115ff659b732bf3cb2e12e68585650f5c5ac4af3e2f8bef8be6dbcda3c533768`
绑定的 Feedback 结果，本轮新增 Feedback provider call 为 0。

| Round | 唯一变化 | 新 Feedback | 最深 gate | Candidate / selected Bank | Disposition | 新增可计量费用 |
|---|---|---:|---|---|---|---:|
| R1 | policy label 投影 + parent-bound sparse schema | 12（11 success） | Creator boundary | 无合法 candidate / Static | 未进 smoke、replay、body | CNY `0.05534820` |
| R2 | Exact / Multi / Document sparse patch；其他能力 inherit | 0 | replay200 capability screen | raw `ed595889dc1f…` / Static `da5cfe1f93f…` | 三项全部回滚；body 0-call | Assistant CNY `0.37477635` |
| R3 | Exact-only；五项 inherit；强制 `no supported match` | 0 | replay200 capability screen | raw `c32223e29eec…` / Static `da5cfe1f93f…` | Exact 回滚；body 0-call；S1 停止 | Assistant CNY `0.38035905` |

R2 的 Exact raw success 为 `24→27`，但出现 2 条 Static-success→candidate-failure 和新增
fallback/tool reason，因此不能把净 `+3` 当作通过结论；Multi 为 `7→2`，Document 为
`0→0` 且新增 tool reason，`retained_capability_ids=[]`。R2 的 screened Bank 为
`968e05985f02…`，最终 selected 仍是 Static。

R3 的 Exact 配对分解为 `27 个 0→0 / 1 个 0→1 / 20 个 1→0 / 4 个 1→1`，success
`24→5`。20 条回退中，18 条仍正确路由 Exact、使用与对照相同的 image-search
arguments/result 且各返回 1 个 eligible candidate，却输出 `no supported match` 与
`product_cards:none`，统一触发 `card_contract_failed`；另 2 条是 Description 冻结下的
Exact→Multi 路由采样噪声。已验证事实支持“R3 Body/response-contract 交互导致非空命中
也过度 fallback”，不支持归因给工具、数据或容量。R3 smoke24 的 oracle coverage 完整；
replay200 的 oracle coverage 也完整。224 个 Assistant query 产生 782 个 Qwen model call，
request ID 全部唯一，无 429、timeout 或 provider/service error。

Fast R1–R3 新增 DashScope 费用合计 CNY `0.81048360`。5 次 Codex Creator session 已全部
用于 R1/R2 的执行修订与最终 R3；其 cost basis 不可得，不虚构人民币费用。R3 的
`decisions/s1.json` 冻结 `accepted=false`、`alias_of=llm_static`、selected Bank
`da5cfe1f93f2cb57b389c97738034cf10cab931dcc30e145acc6d8873ff1348a`。`body_gate75`、
Final Judge、val 和 `test300` 从未访问；body 结果没有反哺 Creator。R4/R5 未执行，S1
在 R3 后无条件停止。S2 必须以上述 Static/S1 alias 为 parent，仅修改 Description，不能
带入任何被拒绝的 R2/R3 Body。

#### 2026-08-12 Qwen3.7 fresh Static + R1–R5 最终终态

本节取代上一节作为当前 Core Fast 的唯一前向执行结论，但不改写旧 Qwen-VL lineage。
Assistant 与 route-only 均切换为 `qwen3.7-flash-2026-07-15`，主调用共享 provider-call
级 `60 workers / 20 requests/s`；Qwen3.8 Feedback 保持 `60 / 8 requests/s`。Static
opt800 以新 execution identity 真实重跑，成功产物为 800/800 outer success、2,489 次
Assistant model call、GCS `167/800`、hard error `118/800`，文件 SHA-256 为
`ced36fb3050612be9a45a9fcb0d0d835b0f7a40ef422009066b6680df7368fc0`。所有 inner
requested/returned model 均为新 Qwen3.7，request ID 非空且全局唯一；Static 费用为
CNY `1.2222646`。第一次 Static identity 因合法的重复 multi tool call 被 adapter 错误抛成
provider_error 而废弃；修复后重复调用由 GCS fail closed 计分，不再伪装为 provider 故障。

固定 canary/body 样本按新 Static observation 在 discovery600 中确定性重选。R1 新调用
Qwen3.8 Feedback 12/12 success，source manifest SHA-256 为
`4e5d7013665d5ba70ce7c92fbe71ba5c44a363477c1d19d2337552382fb88043`；R2–R5
逐字节复用该 bundle，各轮新增 Feedback call 为 0。每轮只允许一个 capability patch，其他
五项 byte-exact inherit；Description、Static parent、fold、GCS 和门槛均不变。

| Round | 唯一目标 | Target replay success | Paired 回退 / 新 contract occurrence | 终态 | 新 Assistant 费用 |
|---|---|---:|---:|---|---:|
| R1 | Multi positive closure | `8→9` | `4 / 7` | patch 回滚 | CNY `0.3461880` |
| R2 | Multi tool-first + DTO closure | `8→20` | `1 / 3` | patch 回滚 | CNY `0.3708244` |
| R3 | Multi DTO template | 未运行 | Creator sparse proposal 被 authored-content guard 拒绝 | 无 candidate | CNY `0` Assistant |
| R4 | Document literal line copy | `0→0` | `0 / 3` | patch 回滚 | CNY `0.3511126` |
| R5 | Style evidence copy | `7→26` | `2 / 4` | patch 回滚 | CNY `0.3509304` |

R2 是最强 Multi 信号：净增 12 条且 tool-contract failure 总数 `15→5`，但
`r2-core-0783` 跳过 `multi_product_search` 并编造 item/card handle，构成 1 条明确
Static-success→candidate-failure 和 3 个新 contract occurrence，所以零回归筛查必须拒绝。
R5 是最强 Style 信号：净增 19 条，但一条空结果漏掉冻结 fallback marker，另一条跳过工具
并编造候选，同时出现 4 个新 contract occurrence，也不能发布。R3 的 DTO 假设没有得到
Assistant 实验：Creator 外层成功，但指令中的 `result` 与斜杠组合触发词法 guard；不得把它
写成算法负结果或无授权重试。

五个 `decisions/s1.json` 均为 `accepted=false`、`alias_of=llm_static`，selected Bank 均为
Static `da5cfe1f93f2cb57b389c97738034cf10cab931dcc30e145acc6d8873ff1348a`。没有候选通过
replay screen，所以 `body_gate75`、Final Judge、S2、val 和 test 均为 0-call/未访问。
本授权的 5 次 Codex Creator session 已全部使用，S1 至此按负结果停止；不能启动 S2，不能
追加第六轮，也不能放宽门槛或重跑挑样。

本次可追踪的全部新 provider 记录（含废弃 Static v1、成功 Static v2、唯一 Feedback 和
四轮 Assistant）合计 7,823 calls、`14,383,351 / 1,276,416` input/output tokens、CNY
`3.9202578`。废弃 v1 的 provider_error 行发生在 adapter 后处理，可能已有不可恢复的实际
provider 调用，因此该金额是可追踪下界；可用 v2 lineage 单独为 5,326 calls、CNY
`2.6998944`。5 次 Creator 合计 `171,674 / 4,791` tokens，但人民币 cost basis 不可得，
不得把 receipt 中的 0 解释为免费。证据根为
`E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v2`。

若未来另行授权新的诊断周期，首先应让 sparse validator 输出精确拒绝原因，并以词法安全
表述真正测试 R3 的 DTO-copy 假设；不得读取 body/val/test 反向调参。若 prompt 仍偶发跳过
工具，下一版应转向确定性 DTO 编译，而不是继续堆叠自然语言指令。

#### 2026-08-12 授权后的 R6–R10 第二批终态

用户在 R1–R5 停止后明确授权新的最多五轮。第二批继续绑定 Static v2 SHA
`ced36fb3050612be9a45a9fcb0d0d835b0f7a40ef422009066b6680df7368fc0`、Static Bank
`da5cfe1f93f2cb57b389c97738034cf10cab931dcc30e145acc6d8873ff1348a` 和同一份 Feedback
manifest `4e5d7013665d5ba70ce7c92fbe71ba5c44a363477c1d19d2337552382fb88043`。R6–R10
使用独立 root；任何被拒 candidate 都没有成为下一轮 parent。

运行前只修复已验证的 guard P1：把旧的“整段文本任意路径 marker + 远处 result”组合判断
收窄到局部 path token，并增加脱敏 stable rejection reason。真实实验路径泄露和强禁词仍拒绝。
五轮假设在启动 R6 前冻结，未读取 body/val/test，也未根据中间结果改写后续题目。

| Round | 唯一目标 | Target replay success | Paired 回退 / 新 occurrence | 终态 |
|---|---|---:|---:|---|
| R6 | Multi tool-first + exact public DTO copier | `8→18` | `2 / 4` | patch 回滚 |
| R7 | Style tool-first + evidence/fallback serializer | `7→22` | `2 / 4` | patch 回滚 |
| R8 | Recipe detect/lookup + source-only closure | `3→10` | `0 / 7` | patch 回滚 |
| R9 | Exact image/text input latch | `17→17` | `2 / 2` | patch 回滚 |
| R10 | Document literal OCR lines | `0→0` | `0 / 1` | patch 回滚 |

五个 decision 均为 `accepted=false`、`alias_of=llm_static`、selected Static。R8 虽然没有任何
Static-success 回退，但新增 7 个 contract occurrence，仍不满足预注册的 capability-local
screen。所有轮次止于 replay200；`body_gate75`、S2、Final Judge、val/test 均 0-call。

第二批 Assistant 为 1,120 outer / 3,588 inner Qwen3.7 calls，token 为
`6,818,346 / 552,635`，新增可计费用 CNY `1.8057772`；0 provider/capacity error。五次
Creator token 为 `167,532 / 5,584`，人民币 cost basis 不可得。Feedback 全部 byte-exact
复用，新增 provider call 为 0。第二批额度至 R10 用尽，不追加 R11，不启动 S2。证据根为
`E:\skillchain-data\runs\portfolio-core-qwen37-20260812-v3`。

算法结论是：自然语言 sparse instruction 能稳定产生显著净增益信号，但仍无法把 tool-first、
DTO/card/evidence closure 与 fallback 变成无偶发违约的确定性状态机。下一次若另行授权，应
版本化确定性 compiler/runtime contract，并使用新的对称 parent/candidate lineage；不应继续
追加 prompt 轮次或放宽 screen。

### Round 2：S2 Risk-aware Route Optimizer

先做 Description-only：由 opt800 的 confusion pair 生成最小 Description 修改，Body、工具预算和回答格式全部冻结。重点是 Exact/Multi、Exact/Style 和 Style/Encyclopedia 等已观察边界，不做全局放宽。

风险目标为：

```text
route_utility = corrected_count - 2.5 × broken_count
```

接受条件：route macro-F1 目标 `+3pp`；任一 capability recall 不下降超过 `3pp`；`route_utility > 0`；changed-route rows 的 Contract Success 与 pairwise 净效用非负；hard error 不增加。没有最少 flip 数量要求。

若 Description-only 未达到质量目标，或无法达到路由成本目标，可在同一轮评估 Hybrid：字符 2–5 gram TF-IDF + balanced Logistic Regression，只读取 user turns，在 opt800 训练、`route_gate` 校准阈值；低置信或 top-2 margin 不足时回退视觉 LLM。目标 local coverage 与 route LLM 调用降幅均至少 70%。Description-only 与 Hybrid 最终只能选择一个；选择发生在 `route_gate`，不能再看 `body_gate`、`shadow_val` 或 test。

### Round 3：S3 Body Refiner

冻结 S2 选定的 route artifact、工具输入和 tool trace，只针对 route-correct 且证据充分的 failure cluster 做 Body micro-patch。每次最多修改 1–3 条指令，不整段重写。

#### 采用结论与适用边界

对本地 `D:\athena\SkillOpt`（审计 revision `9969a8f`）的结论是：其细粒度文本优化思想适合用来替换当前 S3 的完整 Body replacement，但没有必要把四 epoch、slow/meta update、通用训练器和标量 Gate 整体接入本项目。

既有 S3 反例证明了这一改造的必要性：被回滚的 `product.multi_search` 候选把 Body 从 3,288 增至 4,272 字符，改动 7 条带 ID 规则以及 Objective、fallback；Gate 中 `J_project` 从 75.56 降至 73.80，而 adherence 从 0.6833 升至 0.6983，6 个直接受影响样本平均约 `-7.33 J`。这说明完整重写容易增加互相冲突的约束，产生“更遵循文本、但业务质量更差”的过优化。

因此，细粒度 patch 对“让 S3 形成可归因的独立增益”基本必要，但不是 Core headline 增益的强制前置。当前 21 个物理 card 违规中有 19 个来自 Style/Exact 没有 grounded candidate，Body 文本无法制造缺失证据；必须先完成数据、检索和 card candidate 闭合。若上游闭合后 opt800 仍没有稳定的 Body 可归因 failure cluster，则跳过 S3，Full 直接引用 S1+S2 artifact。

#### 轻量 `S3-TextOpt` 实现

不把 SkillOpt 增加为运行依赖，只在现有 S3 model proposal 与 `PortfolioSkillMutation` 之间增加一个原生、确定性的 patch proposal/compiler 层。候选 schema 至少包含：

```text
capability_id
failure_cluster_id
edits[]:
  op = replace_rule | insert_after_rule | delete_rule
  rule_id
  expected_text_sha256
  replacement
  evidence_ids
  support_count
  addressed_dimensions
  rationale
```

执行约束固定为：

- 一个候选只处理一个 capability、一个 failure cluster；首个诊断候选固定 `L=1`，确认有效后同一冻结输入下最多 3 个 edit；
- 优先使用 Body 已有的稳定 rule ID（例如 `[multi.success.decomposition]`）定位，`rule_id` 和原文本 SHA 必须唯一匹配；目标缺失、重复或漂移一律 fail closed，不允许退化为尾部 append；
- `Output contract`、Description、operators、references、route 和工具预算继续冻结；首轮 Body 长度增幅不超过 10%，任何候选都不得超过全局 20% 上限；
- trusted compiler 顺序应用 patch、重验 canonical Markdown 和 frozen output contract，再生成现有完整 Body carrier；下游 Bank compiler、mutation hash、runtime binding、lineage、Gate 和 rollback 保持不变；
- attribution 先分类 `SKILL_DEFECT` 与 `EXECUTION_LAPSE`。只有缺失、错误或欠明确的规则进入 Body patch；已有正确规则但 Assistant 偶发未遵循，以及 provider/tool failure、候选为空或证据缺失，均记录为“不修改 Body”；
- 保存 rejected-edit buffer：patch 指纹、证据、受影响规则、确定性指标与 Judge 变化、拒绝原因。后续候选不得生成语义等价 patch，除非有新证据并显式说明为何推翻旧拒绝。

不采用 SkillOpt 的 whole-document rewrite、自动 learning-rate、默认 slow/meta update、ungated appendix、LLM merge 失败后直接拼接、ranking 失败后直接截取，以及 `candidate_score > current_score` 的单标量 Gate。现有 common-route/common-tool 因果复用与多指标 Gate 继续作为唯一接受依据。

#### 触发条件、最小消融与停止规则

S3 只有在以下条件同时满足后才启动：Style/Exact 的 candidate/card 闭合已处理；目标样本 route correct、tool success、grounded evidence 充分；opt800 中形成同 capability、同 failure signature 的稳定 Body cluster。建议预注册触发阈值为不少于 20 个失败样本，并加入不少于 10 个当前通过的 positive anchors；达不到阈值则本轮不生成 S3 候选。

在同一 parent、同一 attribution 输入、同一候选预算下，预注册一次诊断消融：

1. `coarse`：当前完整 Body replacement，只作诊断基线；
2. `fine`：单 rule、`L=1` 的细粒度 patch；
3. 若 `fine` 在 opt 内部交叉验证通过，才允许在相同冻结输入上合并至最多 3 个 edit；只有一个最终候选进入一次 `body_gate`。

Primary 使用确定性的 Grounded Contract Success，并单列 requested-item coverage、重复/遗漏、unsupported claim、card/evidence 对应和长度预算；pairwise AB/BA 作为次要语义证据，legacy J 只作诊断。受影响 cluster 的接受门槛为 Contract Success 至少 `+5pp`，小样本时至少净修复 2 条；positive anchors 不新增失败；无新增 hard/card/evidence error；pairwise tie-adjusted preference 不低于 `0.50`；增益必须超过预注册 no-op 波动，不能只表现为单次 J 或 adherence 上升。

`body_gate` 结果不得反哺候选；Gate 未通过立即回滚。若 `coarse` 与 `fine` 均失败、只提升 adherence/Judge 而不提升确定性 contract，或失败仍主要归因于工具/数据，则停止 S3，不开启第四轮，Full 继续 alias S1+S2。

先评估 Body-only。只有 Body-only 无法满足证据/完成度 contract、且 Style/Multi provenance 已完整时，才允许加入 `SemanticAnswerSlotsV1 + constrained composer`。一旦采用，Full 必须命名为 “Body + constrained generation package”，并在受影响 val 子集报告 Body-only 与 Body+Composer；不能把 package 的全部增益归给 Body Refiner。

S3 使用 `body_gate` 中预先留给本阶段的候选预算。接受条件：common-route/common-tool rows 的 Contract Success 明确正向；pairwise 不低于 `0.50`；hard/evidence error 不恶化。回滚时 Full 引用 S1+S2 artifact，不重新采样伪造差异。

### 组合交互 gate 与停止

三轮各自通过后，组合 Bank 只在 `shadow_val` 做一次最终检查：capability-macro Contract Success 至少 `+2pp`，leakage-group bootstrap 下界不低于 0，hard error 差值不超过 `+1pp`，任何 capability 不得明显退化。失败时只执行预定义的交互回滚，不生成新候选、不改变阈值，也不开启第四轮。

## 8. 执行矩阵、并行化和预算

### 8.1 低成本开发

- 本地 deterministic 指标、route surrogate、tool ceiling 和单元/集成测试优先。
- 当前最多 CNY 100 的 DashScope 授权仅用于 Gate 0 smoke、pairwise 校准和三轮小实验，建议目标消费不超过 CNY 30；所有调用记录角色、模型、次数、用途和可得成本。
- 不用 test 做 pilot，不重复调用挑最好结果。

### 8.2 Validation

- `route_gate=75 / body_gate=75 / shadow_val=50` 的 rollout 严格按各自用途和已冻结的 3/3/2 batch assignment 执行；最终 Bank 冻结后，val200 的五配置各运行一次，共 1,000 Assistant rows，作为完整消融报告。
- 第二次只重跑预先冻结的受影响诊断样本；重复按 query 聚合，不当作额外 n，也不能触发新候选。
- 每轮 gate 保存 parent/candidate/common-trace、接受或回滚决定和 Bank hash。

### 8.3 Test 一次性解封

- 五配置各运行一次：`300 × 5 = 1,500` rows；这是 canonical 五配置消融。
- 预注册的 confirmatory schedule 只对 `LLMStaticSkill` 与最终配置各重复一次：额外 600 rows，总计 2,100 rows。该重复单独报告并按 query 聚合，只用于 Static-vs-Final 稳健性，不替代 canonical 五配置结果。
- 若 Full 是 S3 rollback/no-op，直接复用 S1+S2 artifact，不制造独立采样差异。
- 旧绝对 Judge 最多 50×5；确定性 Contract Success 覆盖全部 rows。
- Pairwise：先 40-call 成本/解析 smoke，再做 dev60 的 AB/BA 交换共 120 calls；正式 test Static vs Final 300 对。只有预算和校准均通过时才预先决定是否再加第二批 300，对 test 结果不可见。
- 人工盲评 100 条，加入 20 个隐藏重复检查 reviewer 一致性。

### 8.4 并行与恢复

沿用 25-query shard/coordinator。先做单 wave canary；通过后冻结 `Qwen inflight=2 / Kimi inflight=8`，若限流或稳定性不足则冻结为 `2/4`。正式运行中不得按中间得分改变并发、模型、重试、候选或样本；只按预定义 circuit breaker 暂停和幂等恢复。

Core 正式执行预算不由本次“允许云端上传”自动授权。Gate 0 smoke 后依据真实单价、token 和 fallback 率提交独立 BOM；建议目标 CNY 120–180、建议硬上限 CNY 200。若 pairwise 均价高于 CNY 0.10，先取消可选第二批 300 pairwise并缩减 legacy J，不削弱确定性 primary 与人工 100 条。

## 9. Test 解封清单

只有以下条件全部满足才允许执行 test300：

- [ ] Core create-only permission overlay、receipt 和三个 processor preflight 完成；
- [ ] split-aware loader/runtime/launch 与 shard 恢复测试通过；
- [ ] 既有 val `75/75/50` Gate receipt 的文件 SHA 与 plan/catalog/assignments 绑定已复核，8 个完整 batch 与全部 grouping atom 无拆分、无重叠；
- [ ] opt discovery/replay component assignment、seed 和 SHA 已冻结，240 条 S1 Creator 输入均为 component-unique；
- [x] S1 Feedback 48 条选择、selected-only DashScope Qwen authorization、6+42 exactly-once control 与 typed bundle SHA 已冻结，48/48 均为 parsed；
- [x] Creator v4 的单次调用因未披露的 lexical guard 契约被拒并永久保留；前向修复后，唯一产出有效候选的 Creator v5 也只调用一次，绑定当前 Style 2.3 AuthoringInput、parent Static Bank 和 Feedback bundle，候选明确为六能力整 Bank；
- [x] replay200 已按预冻结门完成并产生单一 byte-exact rollback receipt；因 Encyclopedia `−5pp` 未通过筛查，未访问 `body_gate75` 或 val，Static 仍为共同父基线；
- [ ] provider DTO 不含路径、source、template、canonical label 或 test metadata；
- [ ] Style/Multi tool ceiling 和公共 evidence/card contract 达标；
- [ ] 三轮算法已停止，唯一 S2 路线和 S3 package 身份已冻结；
- [ ] val 组合 gate 通过或已如实冻结负结果配置；
- [ ] Assistant/Judge 模型、prompt、token、seed、并发、重试和 parser 已冻结；
- [ ] Grounded Contract Success、pairwise rubric、人评抽样、bootstrap seed 已冻结；
- [ ] test bytes/hash、五配置 Bank/算法 hash 和 2,100-row schedule 已冻结；
- [ ] Core 正式调用预算另行批准。

## 10. 交付与面试叙事

最终交付：五配置 val/test 表、route/confusion/cost 图、系统闭环架构图、2–3 个 failure→patch→gate 案例、失败和限制、可复现命令、30 秒/2 分钟介绍与面试追问材料。

固定归因：

- `Static → S1`：Failure-driven Creator；
- `S1 → S1+S2`：被 val 选中的 Description Route Optimizer，或明确披露的 supervised Hybrid Router；
- `S1+S2 → Full`：Body-only Refiner，或明确披露的 Body+Composer package；
- rollback/no-op：算法增量为 0，不用重新采样差异冒充效果。

可以诚实讲述机制增益、系统增益、成本优化和安全回滚；不能声称企业生产流量、论文绝对分数复现、真正 blind test、Core 图片可公开展示，或未通过 gate 的总体质量提升。
