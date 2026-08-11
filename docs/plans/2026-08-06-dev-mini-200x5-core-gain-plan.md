# 200×5 深度诊断与 1,500-query core 增益方案

日期：2026-08-06
轨道：Portfolio Track
状态：诊断完成；作为证据保留，不再单独维护执行状态

> 执行入口已合并到 `2026-08-06-core-1500-interview-gain-execution-plan.md`。本文件只保留 200×5 的事实、程序缺陷和根因诊断；Core 云端处理授权、算法轮次、预算和 test 解封均以合并计划为准。

## 0. 结论与执行决策

200×5 已经提供了可讲的机制信号，但还没有提供可稳定外推到 core 的总体增益结论。

可信结论是：

1. `S1+S2` 相对 `LLMStaticSkill` 在 175 条留出样本上方向性提升 `+2.143 J`，路由准确率从 `80.0%` 提升到 `92.0%`，card compliance 从 `81.7%` 提升到 `90.3%`。
2. 该质量增益主要由 `product.multi_search` 驱动；剔除 hard error、Judge 异常和工具异常后，该 capability 的 `S1+S2 - Static` 仍约为 `+13.79 J`。
3. S2 的路由修正有价值，但错误改写的下行风险明显不对称：修正一次平均约 `+8.64 J`，破坏一次平均约 `-21.14 J`。
4. S3 候选被正确回滚，这是闭环安全性的正面案例；当前 `Full` 与 `S1+S2` 使用同一个 Bank，因此二者的分数差不是 S3 效果。

不能声称的结论是：

1. 目前所有主要 paired bootstrap 95% CI 都跨过 0，不能声称总体增益已经统计稳定。
2. 不能把 `Full - S1+S2 = -1.14 J` 解释为 S3 伤害；两者 Bank 完全相同，差异来自独立 Assistant/Judge 采样。
3. 不能把现有 S1 描述为“从基线失败轨迹中自我改进”。它使用 25 条带 gold capability/intent 的用户轨迹一次性创建六个 Skill，但没有读取基线失败、父 Bank 或 failure attribution。更准确的名称是“trajectory-conditioned supervised Creator”。
4. 不能直接把这 200 条的数值外推到 core。core r3 的 200 条 dev_mini 与本次 200×5 语料即使复用了 ID，文本也 0 条完全相同，而且 capability 内部子分布显著变化。

因此，建议先完成一个 core-readiness sprint，再执行 1,500×5。目标不是继续追论文式形式复现，而是把已发现的真实增益点扩展到 style / exact boundary，并把不可解释的噪声从比较中移除。core 冻结测试的启动门槛应是：validation 上相对 Static 至少 `+4 J`、route macro-F1 不下降、无 capability 明显退化、hard error 低于 0.5%。

## 1. 证据范围与分析口径

全量矩阵由以下权威运行重建：

- execution：`D:\athena\ECommerceSkillChain\runs\portfolio\portfolio-dev-mini-200x5-execution-v33-recovery`
- launch：`D:\athena\ECommerceSkillChain\runs\portfolio\portfolio-dev-mini-200x5-launch-v39-recovery`
- runtime：`D:\athena\ECommerceSkillChain\runs\portfolio\portfolio-public-data-runtime-v29-recovery`
- matrix id：`portfolio-dev-mini-200x5-parallel-v3`

`tmp/core-readiness-analysis/analyze_full_matrix.py` 从 40 个 shard 中还原出 1,000 个 `(query, config)` 单元。现有 `analysis-v33d` 只覆盖最后一批 25 条，不能作为 200×5 总结。

主要口径使用后 175 条 evaluation 样本；前 25 条曾参与 treatment gate，仅作为 optimization/smoke 观察。置信区间按 leakage group 做 paired cluster bootstrap，20,000 次重采样。

## 2. 200×5 全量结果

### 2.1 五配置总体结果

| Config | evaluation n | Mean J | Route accuracy | Card compliance | hard/eval anomaly |
|---|---:|---:|---:|---:|---:|
| NoSkill | 175 | 68.706 | N/A | 73.7% | 3 |
| LLMStaticSkill | 175 | 69.937 | 80.0% | 81.7% | 2 |
| S1 | 175 | 71.131 | 83.4% | 85.1% | 3 |
| S1+S2 | 175 | 72.080 | 92.0% | 90.3% | 1 |
| Full | 175 | 70.940 | 92.0% | 90.3% | 0 |

全 200 条均值依次为 `68.960 / 70.373 / 71.753 / 72.230 / 71.563`。前 25 条 optimization 均值依次为 `70.74 / 73.42 / 76.10 / 73.28 / 75.92`。最后一个 batch 曾出现 `S1+S2=62.08` 的异常低点，但它不是全矩阵趋势；只看该 batch 得出的“S2 有害”结论应撤回。

### 2.2 关键 paired contrast

| Contrast（右减左） | Mean ΔJ | 95% cluster bootstrap CI | 解释 |
|---|---:|---:|---|
| NoSkill → Static | +1.231 | [-2.318, 4.797] | 方向性、未稳定 |
| Static → S1 | +1.194 | [-1.592, 4.049] | 方向性、未稳定 |
| Static → S1+S2 | +2.143 | [-1.111, 5.422] | 当前最佳 treatment 信号 |
| S1 → S1+S2 | +0.949 | [-2.503, 4.372] | 路由明显改善，但 J 增益被噪声稀释 |
| S1+S2 → Full | -1.140 | [-3.901, 1.646] | 同 Bank 重复采样，不是 S3 因果效应 |

剔除 hard/evaluator anomaly 后，`Static → S1` 为 `+1.351 J (n=171)`，`Static → S1+S2` 为 `+1.953 J (n=172)`。进一步剔除工具错误和重试异常，后者仍为 `+1.736 J (n=163)`。因此增益并不是固定零分制造出来的，但合理的保守估计应是 `+1.7～2.0 J`，而非更大的 batch 局部值。

### 2.3 capability 归因

在剔除 hard/evaluator/tool anomaly 的 pairwise complete-case 上，`S1+S2 - Static` 约为：

| Capability | ΔJ | 判断 |
|---|---:|---|
| product.multi_search | +13.79 | 唯一强且稳健的质量增益 |
| utility.document_reading | +2.10 | 小幅正向，且 200 与 core 文档域不同 |
| product.exact_match | -0.28 | 基本持平；原始负值主要受 dm-181 假阳性影响 |
| utility.recipe_guidance | -0.93 | 小幅负向 |
| product.style_recommendation | -1.16 | 工具证据不足，Skill 无法补救 |
| knowledge.visual_encyclopedia | -1.53 | 小幅负向 |

这意味着当前总体增益的根因不是“六类 Skill 普遍变好”，而是 multi_search 的映射、card 和完成度显著改善。core test 中 multi_search 只有 44/300 条，即使保留 `+13.79 J`，对全局均值的贡献也只有约 `+2.0 J`；要得到稳定、可讲的全局增益，必须再从 style、exact boundary 或跨 capability 的 evidence/formatting 中获得约 2～3 J。

## 3. 程序性缺陷与测量缺陷

### P0-1：shard 00 的目录契约不一致

launch plan 期望 shard 00 位于 `execution/shards/00-dev-mini-001-r3-00-noskill/`，实际 `assistant/` 和 `final/` 直接落在 `execution/shards/`。数据有效，但正式 analyzer/auditor 无法无特殊 fallback 地重放完整矩阵。

修复要求：runner 永远按 `output_relpath` 落盘；analyzer 对历史恢复包只允许显式、带 matrix/shard identity 校验的兼容读取。core 前需要用零 provider-call 的两 shard dry-run 验证写入、恢复、重复运行幂等性。

### P0-2：严格 Judge JSON parser 会拒绝可恢复的合法答案

`dm-109/S1` 的 Kimi Judge 两次返回完整合法 JSON，但外层包了单个 Markdown JSON code fence，严格 parser 将其记为 0。最后一次原始维度为 `CA=10, CCC=0, CQ=12, TCR=8`，对应 `J=60`，不是模型没有评分。

修复要求：仅允许去除一个完整包裹 JSON object 的 Markdown fence，再执行原有严格 schema 校验；不得猜字段、补分数或从自然语言抽取。用历史原始响应做回归 fixture。

### P0-3：hidden-evaluation identity 规则曾发生内容假阳性

`dm-181` 的商品标题包含 `find. EYE MASK FULL TREATMENT`，被历史 regex 误判为泄露隐藏评测身份，导致 NoSkill 与 S1+S2 固定零分。当前 fail-closed 修复方向正确，但必须保留这个标题作为回归用例；历史分数不应静默改写。

### P0-4：S3 回滚后的 Full 比较不具因果可解释性

`bank-full.json == bank-s1s2.json`，内部 Bank SHA 均为 `9502ef4b...`。但是两套配置重新独立调用 Assistant 和 Judge，造成伪造的 config 差异。对完全相同 `wire_sha256` 的离线重复 Judge 对照中，19 对里 15 对分数不同，平均绝对差 `8.37 J`，最大 `27.5 J`。这足以解释 `Full - S1+S2` 的表面变化。

修复要求：

- 若 S3 回滚，`Full (S3 rollback)` 直接引用已接受的 S1+S2 assistant/final artifact，主表因果增量记为 0；
- 被拒绝的 S3 candidate 仍可作为单独的 diagnostic ablation 报告；
- S3 gate 必须冻结 route 和 tool trace，只比较 Body；否则 route/工具/采样噪声会淹没正文改动。

### P1：低频运行错误在 core 会放大

全矩阵共有 8 个 Assistant hard error 和 1 个 Judge parse anomaly，包括 `route_contract_error`、`route_length`、runtime error 和 hidden identity false positive。按当前比例线性外推，7,500 个 core 单元可能出现约 60 个 hard-error 单元。

修复要求：route 输出采用最小 JSON schema；只针对格式失败允许一次等价重试；记录原始 response 和修复原因；hard error gate 设为 `<0.5%`，任何 capability 超过 `1%` 均阻止 test unseal。

## 4. treatment 与算法根因

### 4.1 原 scaffold 问题已修，但 S1 的学习信号仍不对

Creator / Optimizer / Refiner 现在确实发生了真实模型调用，已经不是 deterministic post-smoke scaffold。剩余问题更深：S1 输入中的 `source_config`、`source_bank_sha256` 和 failure 信息为空，只含 25 条用户轨迹及 gold capability/intent。它学习的是任务分类描述，不是从失败中归纳可复用 Skill。

core 的 S1 应改为：

```text
Static 执行结果
→ route/tool/body/evaluator attribution
→ 失败 cluster + 同 capability 成功 anchor + parent Skill
→ 每个 cluster 生成 2–3 个最小候选 patch
→ paired/common-trace gate
→ 接受、合并或回滚
```

gold capability 只能用于 attribution/evaluation，不能直接作为 Creator 的“答案提示”。Creator 应看到可公开的任务、动作轨迹、工具证据、失败类型和 parent Skill；不看到 test label 或 test score。

### 4.2 S2 有正向路由能力，但 gate 对破坏风险不敏感

在 175 条 evaluation 上，S1→S1+S2 的路由迁移为：

- 139 条保持正确，平均 `ΔJ=-0.16`；
- 22 条由错改对，平均 `ΔJ=+8.64`；
- 7 条由对改错，平均 `ΔJ=-21.14`；
- 6 条保持错误，但平均 J 仍可能变化；
- 1 条从一种错路由变成另一种错路由。

因此一次 broken route 的期望损失约为一次 corrected route 收益的 2.45 倍。现有 S2 gate 在 25 条中只有 1 条路由行为真正变化，却凭 `+0.32 J` 和 route `0.88→0.92` 接受，没有 CI、per-capability floor 或 broken-route penalty，无法发现旧 mini 中 multi recall 的下降。

新 gate 应使用：

```text
route_utility = corrected_count - 2.5 * broken_count
```

并同时要求：route macro-F1 上升、任一 capability recall 不下降超过 3pp、hard error 不增加、paired ΔJ 不为明显负值。S2 只改 Description；Body、工具预算和回答格式全部冻结。

### 4.3 S3 的变更粒度过大

被拒绝的 S3 candidate 重写了整个 multi_search Body，并把“一项一候选”的约束收得过紧，与“合并重复商品”等真实请求冲突。6 个直接受影响样本平均约 `-7.33 J`，回滚正确。

S3 应从整段重写改为稀疏 micro-patch：一次最多修改一个 failure cluster 对应的 1～3 条指令。multi_search 的主体不应再依赖自由文本维持序号关系，而应由 typed intermediate representation 和确定性 formatter 保证：

```json
{
  "group_repeated_items": true,
  "items": [
    {"item_index": 1, "label": "...", "status": "matched", "candidate_ordinal": 2},
    {"item_index": 2, "label": "...", "status": "unresolved"}
  ]
}
```

LLM 负责解释和简洁总结；item→candidate/card 的对应、未解析项和输出长度由 runner 确定性实现。

## 5. 工具与 evidence 是当前最大上限

### 5.1 style tool 目前并没有执行真实相似度检索

`src/skillchain/tools/portfolio_runtime.py` 的 `trace_similar_styles` 只是从固定顺序中取同 category 的前三个条目，并硬编码 `0.95/0.90/0.85` 与 MMR `0.9/0.8/0.7`。`src/skillchain/runners/assistant.py` 给模型的公开 candidate 只有 title、category、score 和脱敏 handle，没有候选图片、可验证属性或用户 modifier 匹配证据。

因此 style 的正确路由并不等于正确工具执行。200×5 中 route 正确的 24 条纯 style 留出样本，平均仅 `56.09 J`；归一化后 conversation quality 约 `35.8%`，模型会为通用候选编造肩型、颜色、袖型或腰线。公开 handle 脱敏和 card/evidence 契约的方向是正确的，但“语义证据不足”仍未解决。

core 前应把 style 拆成两个可观测 submode：

1. `same_category_alternative`：相似/替代商品；
2. `cross_category_coordination`：鞋、包、外套、卧室物件等搭配。

同一个 style capability 可以保留，但工具必须读取用户文本并选择 submode。前者用真实图像/属性相似度；后者需要目标类别识别和跨类别候选检索。返回 DTO 至少应含可验证的相对属性或 caption evidence、真实相似度来源和 candidate image/evidence handle，禁止硬编码分数。无法支持的样本保留为 challenge slice，不静默删除，也不让它主导工具可达性结论。

core 数据中 272 条 style 来自 FashionIQ，28 条来自 ABO；其中约 166/272 条 FashionIQ style query 的 anchor 在原始 caption graph 中有邻接证据，可直接用于构建“anchor→target + relative caption”的 gold retrieval 集。先做 30～50 条 tool ceiling：若 oracle/gold retrieval 本身无法达到足够覆盖，不应通过改 Skill prompt 继续调 style。

### 5.2 multi_search 应保留现有增益，并补齐结构化完成度

core 有 225 条 multi_search，单图 4～20 个对象、中位数 12；3～10 个类别、中位数 6。数据本身适合展示复杂工具链，但必须拆开测：

- detector gold coverage；
- oracle box → product retrieval；
- end-to-end item/card coverage；
- grouping、数量、未解析项和输出截断。

之前为 dm-005 增加紧凑结构和回答长度预算的方向正确；下一步不是继续加自然语言规则，而是把 item mapping 移到 typed formatter，并让 Skill 只决定 grouping/排序/解释策略。

## 6. core 数据审计：哪些结论不会自然迁移

core r3 共 1,500 条，split 为 `200 / 800 / 200 / 300`。数据 join、工具 binding、asset/product/SHA/pHash/leakage-component 跨 split 隔离均通过，未发现规范化文本重复。主要风险不是泄漏，而是 domain 与 task-tool mismatch。

### 6.1 capability 分布

| Split | encyclopedia | exact | multi | style | document | recipe |
|---|---:|---:|---:|---:|---:|---:|
| dev_mini | 35 | 35 | 35 | 35 | 30 | 30 |
| optimization_pool | 163 | 209 | 117 | 163 | 32 | 116 |
| validation | 41 | 52 | 29 | 41 | 10 | 27 |
| test | 61 | 79 | 44 | 61 | 18 | 37 |

document 在 val/test 只有 10/18 条，不足以支撑强 per-capability claim，应单列 exploratory。不要为了补 n 在看过结果后重分 test；如果要新建更均衡的 Portfolio split，必须在任何 core label/score 被用于优化前完成并重新冻结 leakage component。

### 6.2 style 的数据—工具错配

core dev_mini 的 35/35 style 都是跨类别搭配，而现有 `style_similar_search` 只能返回同类别替代品。全 style corpus 中启发式检查约有 116/300 条明显 coordination 请求，另有约 30/300 条包含预算、配送或尺码条件，而产品 schema 不具备这些字段；validation 中该类属性约为 11/41。

这不是 prompt 能解决的问题。应实现 cross-category/constraint-aware 工具路径，或把 unsupported 部分作为 challenge metric 单独披露。

### 6.3 document 的域迁移

dev 的 30 条 document 全是 Wikimedia 一般文档，tail 的 60 条全是 CORD/SROIE 收据。字符 n-gram 已可用 `96.7%` 准确率区分 dev 与 tail。200×5 中 document 的小幅增益不能外推到收据 OCR；需要字段级 gold evaluation，而非继续在 Wikimedia 文本上改 Body。

### 6.4 split 模板指纹与 repair stratum

仅用文本模板特征预测四个 split 的 balanced accuracy 约 `0.475`（随机为 0.25）；dev-vs-validation 可到 `0.814`，document dev-vs-tail 可到 `0.967`。此外 r3 语言 repair 占 250/1,500，validation 的 repair 比例达 27%，其中 multi 约 48.3%、document 约 60%。

建议按 `template_family / source / repair / boundary / style_submode` 报告 slice，bootstrap 继续以 leakage/generator component 为组；这些元数据只用于审计与分层，不暴露给模型。

## 7. 已完成的小实验：S2 core 文本路由 surrogate

为了验证“扩大 multi、对称扩大 style”的 S2 候选是否值得进入昂贵的图像实验，使用 `qwen3-vl-flash-2026-01-22` 对 optimization_pool 800 条 query 做了文本-only、三描述版本的 batched routing surrogate。未上传图像、未调用 Judge、未接触 validation/test。

| Bank description | Accuracy | Macro-F1 | exact recall | multi recall | style recall |
|---|---:|---:|---:|---:|---:|
| S1 | 91.00% | 92.56% | 72.73% | 100.00% | 93.25% |
| 当前 S2 | **96.63%** | **97.31%** | **90.91%** | **100.00%** | 96.93% |
| 新候选 S2 | 95.13% | 96.08% | 84.21% | 99.15% | **98.16%** |

当前 S2 相对 S1：修正 46 条、破坏 1 条；新候选相对当前 S2：修正 4 条、破坏 16 条。因此候选已回滚。它同时说明：旧 200 mini 中观察到的 multi 路由问题不能直接迁移到 core；core 当前的主要路由缺口是 exact→style（12 条）和 exact→multi（7 条），不是 multi recall。

边界样本上当前 S2 accuracy 为 `92.70%`，非边界为 `97.44%`。下一轮 S2 应只针对 exact 的放宽语句、多对象外观但单商品意图、以及 same-image cross-intent challenge 做最小 confusion-pair patch，不应全局扩张 multi 定义。

实验产物：

- `tmp/core-readiness-analysis/run_core_text_route_surrogate.py`
- `tmp/core-readiness-analysis/core-text-route-surrogate-result.json`
- result SHA-256：`199ee2b208b83ea9fbb58ff19def4d9e748addc01a2101be051d20074fb8df79`
- 本轮共 117 次 Qwen provider call（包含前序 schema 解析失败的已付费响应），271,666 input tokens、86,164 output tokens；按实际模型费率估算共 `CNY 0.16999590`。成功 checkpoint 文件自身记录 `CNY 0.144627`。费用远低于本轮 CNY 100 授权。

## 8. 从综述提炼的可落地闭环

综述中的 SkillOpt、SkillForge、EvoSkill、SkillsVote、SkillRouter/SkillsWild、SkillTester、SkillEvolver、XSkill、CoEvoSkills、AutoRefine、ExpWeaver、SAGE/Agent0/Tool-R0 等工作的共同启发，不是增加更多角色名称，而是把演化过程变成可归因、可搜索、可回滚的局部优化：

```text
结构化 attribution
→ 按 failure cluster 生成多候选
→ 最小字段 patch
→ paired / counterfactual validation
→ risk-aware accept-or-rollback
→ 保存成功与失败 patch 的 utility memory
```

对本项目最有价值的移植是：

1. 多候选但有界：每个 cluster 2～3 个候选，先用确定性指标淘汰，再调用昂贵 Judge；不做无界 best-of-N。
2. 分层搜索：S1 只处理新 Skill/策略缺口，S2 只处理 route confusion，S3 只处理 route-correct 的 Body failure。
3. 反事实 gate：route patch 在同一批 query 上比较 corrected/broken；Body patch 复用相同 route 和 tool trace。
4. utility memory：记录 patch 对哪些 cluster 有益/有害，避免以后重复生成已知坏 mutation。
5. Skill adherence 独立报告：区分“Skill 写得更好但模型没遵循”“模型遵循但工具没有证据”“Judge 发生漂移”。

这比复刻论文中的角色数量或完整框架更适合 Portfolio Track，也更容易向面试官解释算法与工程取舍。

## 9. 建议的 core-readiness 三轮优化

根据项目停止规则，最多做三轮有记录的算法优化。修 parser、shard 和 measurement contract 属于 P0 修复，不计入为追分而进行的算法轮次。

### 前置 P0（零/低 provider call）

1. 修 shard 目录契约和历史 fallback；做 2-shard dry-run/recovery test。
2. 修 fenced Judge JSON；加入 dm-109 fixture。
3. 固化 dm-181 hidden-identity regression。
4. 回滚 alias：同 Bank 的 Full 直接复用 S1+S2 artifacts。
5. 为 route format failure 增加一次严格 schema retry；压测 hard-error rate。
6. 用 30～50 条/工具 gold set 测 detector、retrieval、OCR、style 和 multi formatter ceiling。

### Round 1：工具证据 + typed formatter

目标：解除 style/multi 的工具上限，并保证所有五配置使用完全相同的新 runtime。

- style 实现真实相似度与 coordination submode；把用户 modifier 传入工具；返回可验证证据，不再硬编码 score；
- multi 引入 typed item mapping、grouping flag、deterministic card/output formatter；
- receipt document 增加字段级 OCR gold evaluator；
- 只在 dev_mini/core gold tool set 上调试，不看 val/test 分数。

验收：style top-k/evidence coverage 和 multi end-to-end item coverage 达到预先冻结阈值；无 evidence 时显式 abstain；所有配置 runtime 对称。

### Round 2：真正 failure-driven 的 S1

目标：让 Creator 从 Static 的 opt failures 形成可复核的 Skill 改进，而非从 gold trajectory 一次性写六个 Skill。

- 在 optimization_pool 上先缓存 Static 轨迹；
- attribution 分 route/tool/body/evaluator，只有 route/tool/body 进入 Skill 学习；
- 每个 failure cluster 加 1～2 条成功 anchor 与 parent Skill；
- 每个 capability 单独生成、单独 gate，再组合 Bank；
- accepted patch 必须带 before/after cases 和失败边界。

验收：opt paired `ΔJ ≥ +3` 或 capability-macro 明显改善；至少两个 capability 正向；任一 capability 不下降超过 2 J；hard error 不增加。

### Round 3：风险敏感 S2 + 稀疏 S3

目标：在不破坏已正确 route 的前提下补 exact boundary，并只修 route-correct Body failures。

- 保留当前 S2 为 parent；只生成 exact↔style、exact↔multi 的 confusion-pair 描述 patch；
- 对 text router 与 multimodal router 不一致的少数样本触发 top-2 pairwise rerank，避免所有样本增加调用；
- S2 gate 使用 `corrected - 2.5*broken`、macro-F1、per-cap recall floor、hard error 和成本的 Pareto 条件；
- S3 每次只改一个 failure cluster，冻结 route/tool trace，以 deterministic metrics + 小 Judge sample gate；
- 被拒绝候选进入 utility memory，Full 自动 alias 到已接受 parent。

验收：validation 相对 Static `ΔJ ≥ +4`，route macro-F1 不低于 parent，所有 capability 无明显退化，parser/hard error `<0.5%`。不满足则按轮次停止并如实报告负结果，不打开 test。

## 10. core 实验设计

### 10.1 split 用法

- 旧 200×5：只做回归与架构诊断，不再用于选择候选。
- core dev_mini 200：工具/格式 regression 和少量可视检查。
- optimization_pool 800：failure discovery、候选生成和内部交叉验证；可以切成 discovery/replay folds，但所有 leakage component 必须同 fold。
- validation 200：冻结的 accept/rollback gate；建议保持 route 80、Body 80、shadow 40，禁止把 validation 失败样本反复喂回 Creator。
- test 300：只在所有 Bank、runtime、rubric 和预算冻结后执行一次。

### 10.2 headline 与诊断指标

主结果保持简单：

1. `Static → best evolved config` 的 paired mean J 与 leakage-component bootstrap CI；
2. capability macro-F1、confusion matrix；
3. 六 capability 的 ΔJ；
4. hard error、card/evidence compliance、单 query 成本与延迟。

必须同时报告 route-correct-only Body score、style submode、boundary、source、repair 和 same-image cross-intent challenge，避免把路由、工具和正文效果混成一个数字。

按本次 paired SD `21.68` 粗估，test n=300 在双侧 5%、80% power 下的最小可检测均值约为 `3.5 J`；cluster 相关和 domain shift 只会让要求更高。因此以 `+4～5 J` 为 test-unseal 目标是合理的。仅追求 `+1～2 J` 很可能得到无法讲清的宽 CI。

### 10.3 Judge 设计与成本

现有 200×5 settled 成本约 CNY 142.18，含未决 reserve 的 accountable 上界约 CNY 151.10，约 `CNY 0.151/单元`。Judge persisted-token 成本约 CNY 132.44，占约 98.9%，平均 Judge latency 约 118 秒。按原配置线性执行 7,500 单元，accountable 成本约 CNY 1,133，且主要花在高方差 Judge 上。

Portfolio Track 更合适的方案是：

1. development/opt 全量使用确定性 route/tool/card metrics，昂贵 Judge 只评 failure cluster、候选差异样本和固定随机样本；
2. 在 50～100 条人工盲评样本上校准 Judge 模型和 thinking/output budget；选满足 agreement 阈值的最低成本配置；
3. test 300×5 保留一次统一 Judge，并做 100～150 个 blind side-by-side 人工样本；
4. 不对全量做 Judge self-consistency；完全相同 output 只评分一次并复用；
5. 在 core 执行前另行批准预算。当前 CNY 100 授权只覆盖本轮小实验，不足以启动 1,500×5。

## 11. 对之前优化方向的确认与修正

1. 公开 evidence handle 脱敏、统一 card/evidence 契约：方向正确，历史固定零问题已显著收敛；当前剩余问题是 style DTO 缺少真实语义证据。
2. multi 紧凑结构和长度预算：方向正确；下一步升级为 typed intermediate representation + deterministic formatter，而不是继续堆 Body 文字。
3. S2 confusion-pair + Pareto gate：全量分析和 core 小实验都支持；但当前 S2 应保留，禁止全局放宽 multi/style，只做 exact boundary 的局部候选。
4. S3 多样本聚合、规则路径 + Judge、稀疏 Body 修改：支持；还应增加 common route/tool trace 和回滚 artifact alias。
5. paired/common-route、冻结 rubric、Skill adherence 独立指标：必须纳入 core，否则同 Bank 的采样差会继续冒充算法效果。

## 12. 面试叙事

现在可以诚实讲：

> 我实现了五配置、六 capability、真实模型和真实工具的 1,000 单元实验。完整审计发现 S1+S2 相对 Static 有约 +2.1 J 的方向性增益，路由准确率从 80% 到 92%，并在 multi-product 上获得约 +13.8 J 的稳健提升；同时 S3 的自动 gate 正确回滚了一个有害 patch。进一步的因果审计发现 Full 与 parent 同 Bank，表面差异来自 Judge 方差，于是我把闭环改成 failure attribution、多候选局部 patch、common-trace paired gate 和 risk-aware rollback。

完成 core 优化后，目标叙事是：

> 系统不是靠多写 prompt 获得分数，而是先测工具上限，把失败归因到 route/tool/body，再分别用 Creator、风险敏感 Router Optimizer 和稀疏 Refiner 做局部演化；候选在冻结 validation 上接受或回滚，最终在未见 300-query test 上报告质量、路由、证据覆盖、成本和失败案例。

不应声称论文绝对分数复现、企业生产流量效果、现有 S1 已经 failure-driven，或 Full 已经优于 S1+S2。

## 13. 产物

- 全矩阵重建结果：`tmp/core-readiness-analysis/full-matrix-diagnostics.json`
- 全矩阵重建脚本：`tmp/core-readiness-analysis/analyze_full_matrix.py`
- core 路由 surrogate：`tmp/core-readiness-analysis/core-text-route-surrogate-result.json`
- core 路由实验脚本：`tmp/core-readiness-analysis/run_core_text_route_surrogate.py`
- 未执行的图像 A/B pilot：`tmp/core-readiness-analysis/run_route_candidate_pilot.py`
- 参考综述：`docs/Self_Improving_Agents.pdf`
