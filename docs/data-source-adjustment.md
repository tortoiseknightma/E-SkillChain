忽略 licensing 后，当前组合仍不是技术最优，主要问题是**数据集提供的标签与目标 capability 不完全匹配**。建议从“按来源分配意图”改为“按任务所需监督信号选择来源”。

## 核心调整结论

| 当前来源 | 调整建议 | 原因 |
|---|---|---|
| ABO | 保留，作为 Exact Match 主干和商品 gallery | product/listing 身份与真实多视图最可靠 |
| MUGE | 从商品 gold 降为中文查询语言源 | 有中文 query-image pair，但缺稳定 SKU 多视图关系 |
| MEP-3M | 移出 MVP，core 阶段再考虑 | 规模过大、图片获取不稳定；更适合长尾类别和 hard-negative |
| COCO/Open Images | 降为 detector/open-world challenge | 有 bbox，但没有 SKU identity，无法评价逐商品检索 |
| DeepFashion | 只作服饰候选 gallery | 擅长同款检索，不直接提供“风格推荐”真值 |
| iNaturalist | 保留为细粒度百科 challenge | 物种 taxonomy 和真实复杂图像价值高 |
| Wikimedia/zhwiki | 保留为百科主源 | 图像、实体和知识证据可以自然绑定 |
| Wikimedia Documents | 降为 OCR wild challenge | 图像多样，但字段级 OCR/KIE gold 不稳定 |
| ISIA Food-500 | 保留为中文菜品识别源 | 适合识别菜品，不适合直接作为食谱证据 |
| 下厨房语料 | 保留为中文 recipe KB | 适合食谱检索，但需要与菜品图建立实体映射 |

必须新增或替换的来源是：

1. RPC：Multi-Product主源；
2. WildReceipt：中文Document Reading主源；
3. DuRecDial 2.0 + CrossWOZ：MVP 多轮交互模式来源；
4. Codex Mock：在 TaskSpec 与已验证资产/KB 上做受控领域重写，输出必须标记
   `synthetic_derived`。

JDDC 2.0 已于 2026-07-24 永久退出，不再是 MVP 或 Full 的 RAW/实现依赖。

C2 source review 使用
[`mvp-source-review-policy-v2.json`](../specs/data_sources/mvp-source-review-policy-v2.json)：
它绑定当前 portfolio 摘要，保留 v1 作为历史工件，移除 retired JDDC；DuRecDial 2.0
与 CrossWOZ 只可批准为 `interaction_pattern`，MUGE 只可批准为
`language_style`。Codex Mock 是 `synthetic_derived` 派生输出，不伪装成需要
`download_allowed` 的外部 source approval，而由独立 provenance 与人工接受门治理。

## 各 capability 的最优配置

### 1. Exact Match

推荐：

- MVP主源：ABO；
- core补充：Products-10K；
- hard negative：MEP-3M/MUGE同类商品。

ABO 有147,702个listing、398,212张catalog图，并包含8,222个带24或72帧turntable序列的listing，非常适合构建真实同商品多视图。[ABO数据页](https://registry.opendata.aws/amazon-berkeley-objects/)

Products-10K包含10,000个SKU和约190,000张人工确认图，平均每个SKU有更多真实外观变化，比MUGE更适合作为Exact Match扩展集。[Products-10K官方页](https://products-10k.github.io/challenge.html)

构造规则：

```text
正例：同product/SKU，不同真实asset，排除近重复
难负例：同细类目、外观相似、不同SKU
普通负例：跨类目
```

MUGE和MEP-3M不再提供Exact Match正例，只负责中文措辞与难负例候选。

### 2. Multi-Product

将 RPC 替换为主源。

RPC同时提供：

- 单商品参考图；
-真实多商品结账场景；
-200个细粒度SKU；
-多层次bbox与商品标注。

它天然支持：

```text
detect
→ crop
→ per-object retrieval
→ SKU级端到端成功率
```

这比COCO更贴合当前Multi-Product工具链。[RPC项目页](https://rpc-dataset.github.io/)

建议：

- RPC：主评测；
-自有多商品照片：跨域challenge；
-SKU-110K：只测高密度检测；
-COCO/Open Images：只测open-world detector，不进入SKU检索主指标。

### 3. Style Recommendation

当前用MUGE/DeepFashion关键词推导Style是不够的。应拆成两个任务信号：

- FashionIQ：参考图片 + 相对自然语言修改，例如“更长、更深色、更正式”；
- Polyvore：商品间搭配兼容性；
- DeepFashion：服饰候选库与同款排除；
- ABO：家居、材质和非服饰风格扩展。

FashionIQ就是为“图像 + 自然语言反馈 → 目标图像检索”设计的，比简单同类目检索更接近Style Search。[FashionIQ官方仓库](https://github.com/XiaoxiaoGuo/fashion-iq)

Polyvore提供真实用户构建的outfit、兼容/不兼容标签和fill-in-the-blank任务，适合定义搭配推荐。[Polyvore数据仓库](https://github.com/xthan/polyvore-dataset)

MVP可只用FashionIQ，经中文改写和人工审核；Polyvore留到core。

### 4. Visual Encyclopedia

建议：

- Wikimedia图片 + zhwiki/Wikidata：主源；
- iNaturalist：细粒度物种challenge；
- COCO/Open Images：常见物品open-world challenge。

关键改进是将：

```text
图片实体ID
→ zhwiki/Wikidata实体
→ 固定revision证据
```

直接绑定，而不是让LLM根据图像自由生成百科事实。

iNaturalist应重点用于相似物种、属种混淆和低置信度拒答，不要承担整个百科主数据。

### 5. Document Reading

主源改为：

- WildReceipt：中文收据；
- CORD：复杂收据OCR、行级bbox和结构化字段；
- SROIE：商户、日期、总金额等核心字段；
- Wikimedia Documents：只作布局与拍摄条件challenge。

WildReceipt提供文本框、文本和字段类别，直接适配当前“文本 + 结构化字段 + grounded line evidence”评价。[WildReceipt文档](https://mmocr.readthedocs.io/en/dev-1.x/api/generated/mmocr.datasets.WildReceiptDataset.html)

CORD有图片、OCR框、文本、行组和多级字段标签，非常适合检查“字段正确且引用真实OCR行”的要求。[CORD官方仓库](https://github.com/clovaai/cord)

建议MVP混合：

- 60% WildReceipt；
- 30% CORD/SROIE；
- 10% Wikimedia wild document。

### 6. Recipe Guidance

需要把“菜品识别”和“食谱检索”分开：

```text
食物图片
→ dish/entity identification
→ recipe KB lookup
→ 引用食材与步骤
```

推荐：

- ISIA Food-500：中文菜品识别；
-下厨房语料：中文 recipe KB；
-Recipe1M+：图像与食谱天然配对的补充；
-用户人工维护 dish alias/entity mapping。

Recipe1M+包含超过100万份食谱和约1300万张关联食物图，监督信号比“食物分类数据 + 随机食谱”可靠得多。[Recipe1M+项目页](https://pic2recipe.csail.mit.edu/)

MVP无需下载全量Recipe1M+。只抽取与当前中文菜名可映射的有限子集。

## 增加独立的语言与轨迹层

JDDC 2.0 原本能提供大规模真实中文电商多模态轨迹，但当前无法取得。项目 owner
决定永久放弃该依赖：MVP 不再等待授权，改为机制级 Mock；Full 也不预设 JDDC 2.0
回归。原数据约含 24.6 万会话、300 万条话语和 50.7 万张图片，因此替换后不能再
声称复现了其流量规模或原始多模态用户分布。[JDDC 2.0论文](https://arxiv.org/abs/2109.12913)

MVP 组合为：

- DuRecDial 2.0：偏好追问、推荐和话题转移模式；
- CrossWOZ：约束冲突、无结果、放宽条件、多次查询与动态目标修改模式；
- MUGE：短查询和中文商品表达；
- Codex：在 grounded plan 上组合并重写为项目任务；
- 人工：审核边界、事实、自然度与 synthetic provenance。

这些来源只提供语言和交互结构，不直接提供 capability 或商品事实 gold。正确流程是：

```text
TaskSpec决定语义和标签
+ 真实资产/KB决定事实
+ DuRecDial/CrossWOZ/MUGE提供交互与表达模式
+ Codex负责受控领域重写并标记synthetic_derived
+ 人工单独审核后才能接受
```

Full 的具体实现计划暂缓到授权数据实际到手后再冻结。U-NEED 已于 2026-08-01 因无法获得
而永久退出：其原本的售前需求澄清角色改由 DuRecDial 2.0、CrossWOZ、MUGE 的抽象模式和
受控 Codex 重写承接，所有产物保持 `synthetic_derived`，不得声称真实中文电商售前分布。
SIMMC 2.1 是多模态指代结构候选；项目 owner 已确认 CSDS 已取得覆盖使用、派生和再发布的书面许可，可作为 Full 的条件候选。
用户提及的 `JDDC 2.1` 目前只登记为身份/条款待核验标签，
不能与 SIMMC 2.1 混称或提前成为实现依赖。详见
[`dialogue-trajectory-source-policy-v1.json`](../specs/data_sources/dialogue-trajectory-source-policy-v1.json)。

## 调整后的200条MVP

建议分配：

| Capability | 数量 | 主源 |
|---|---:|---|
| Exact Match | 35 | ABO |
| Multi-Product | 35 | RPC |
| Style Recommendation | 35 | FashionIQ |
| Visual Encyclopedia | 35 | Wikimedia + iNaturalist |
| Document Reading | 30 | WildReceipt + CORD |
| Recipe Guidance | 30 | ISIA Food-500 + 下厨房/Recipe1M+ |
| 合计 | 200 | |

同时满足：

- 40条左右boundary样本；
-至少40条人工直接编写或重写；
-同一商品图同时构造Exact/Style等不同意图；
-同一食物图同时构造Encyclopedia/Recipe；
-同一多物体图构造整体检索、单对象查询和百科类问题；
-所有跨意图复用样本作为一个原子group进入同一split。

## MVP与core分层

为了控制第一轮复杂度，MVP只新增：

- RPC；
- FashionIQ；
- WildReceipt；
- DuRecDial 2.0；
- CrossWOZ；
- Codex `synthetic_derived` Mock（派生层，不是 RAW）。

沿用已有：

- ABO；
- MUGE；
- iNaturalist/Wikimedia；
- ISIA Food-500；
-下厨房KB。

暂不进入MVP：

- MEP-3M；
- Products-10K；
- Polyvore；
- Recipe1M+全量；
-SKU-110K全量。

这些在core阶段分别补充长尾商品、SKU级Exact、搭配推荐、图像食谱配对和高密度检测。

最终推荐的数据架构是：

```text
商品身份层：ABO / Products-10K / RPC
语言轨迹层：DuRecDial 2.0 / CrossWOZ / MUGE / Codex synthetic_derived Mock
风格监督层：FashionIQ / Polyvore / DeepFashion
知识层：zhwiki / Wikimedia / 下厨房 / Recipe1M+
OCR层：WildReceipt / CORD / SROIE
挑战层：COCO / Open Images / iNaturalist / SKU-110K
```

这样每个数据集只承担它真正拥有可靠监督信号的部分，技术上明显优于当前“一套公开数据顺便承担多个不匹配任务”的方案。

---

## 仓库落实记录（2026-07-22）

本次只读盘点 `D:\athena\ECommerceSkillChain\data` 后确认：

- `raw/` 共 21,504 个文件、8,147,316,535 字节（约 7.588 GiB）；
- `clean/` 与 `kb/` 为空，`index/` 只有 12,288 字节的 SQLite embedding cache；
- 本地没有 ABO、RPC、FashionIQ、WildReceipt、JDDC 2.0；
- 现有 raw cache 不能证明 selection/review/catalog、标签适配性或 formal readiness。

因此没有删除或搬移既有 raw cache，而是通过 [`../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`](../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json) 冻结来源角色和禁用用途：MUGE 降为语言/hard-negative 来源，MEP-3M 延后到 core，COCO/Open Images、iNaturalist、Wikimedia Documents 进入 challenge；ISIA Food-500 只负责菜品识别，并通过经审核的实体映射连接 recipe KB。

确定性 planner 已按六 capability 而不是五 intent 分配 200 条 MVP：Exact/Multi-Product/Style/Encyclopedia/Document/Recipe = 35/35/35/35/30/30；八批各 25 条，boundary 为 40 条。正式语料仍须等待主源、source lock、人工 review 和 catalog 闭环，当前 core 决策继续保持 NO-GO。

后续 Portfolio Track 检查点（2026-07-28）：上述 200 条 `dev_mini` 已在本地 portfolio mini
AssetCatalog 上生成，8 批均经人工审阅并正式接受，当前 staging 为空、计划耗尽。该完成项
是 `synthetic_derived` query corpus，不追溯改写本节的历史盘点，也不关闭 Formal Track
仍要求的主源 exact scope、catalog/gold、`opt_pool` 或 core NO-GO。

---

## MVP RAW 获取检查点（2026-07-24）

在保留 2026-07-22 历史盘点的前提下，统一下载器已完成本轮可自动获取的 MVP RAW：

- 16 个公开 artifact 共 51,529,267,075 字节，`verify --profile mvp` 全部返回 `[ OK ]`；其中 ABO listings、small images、42,446,704,640 字节的 spins archive，以及固定提交的 DuRecDial 2.0/CrossWOZ 均已转正；
- WildReceipt、MUGE 五卷、zhwiki、ISIA Food-500、下厨房语料以及 COCO 有界包均为 `complete`；
- `inaturalist_bounded` 与 `wikimedia_documents_bounded` 两个 command 来源完成，分别冻结 1,200 和 282 条候选记录；
- SROIE 的 5 个语义唯一官方浏览器包已在 2026-07-24 本地取得，并通过大小、SHA-256、ZIP CRC、路径安全、逻辑样本数和图文配对验证；它不再属于“RAW 缺失”，但明确许可条款、PII/use review、外部 source lock、selection/disposition 和 catalog 仍阻断 formal 使用。
- FashionIQ 固定清单已收口为 75,267 张有效图片与 2,416 条 owner-directed 排除；两轮完整失败集合逐条一致，接受集与排除集互斥且完整覆盖 77,683 条。CORD v2 固定提交也已全部下载并校验。
- RPC Kaggle v5 包（27,205,167,166 字节）已通过本地 SHA-256 与 ZIP 结构校验并原子转正，收据见 [`rpc-kaggle-acquisition-v1.json`](../specs/data_sources/rpc-kaggle-acquisition-v1.json)；JDDC 2.0 已永久退出而不是继续保持授权阻断。MVP 下载 profile 的 34 个活动项现已全部完成，并已改为固定提交的 DuRecDial 2.0 与 CrossWOZ；Mock 输出必须显式标记
  `synthetic_derived`；没有使用猜测镜像或把 synthetic 数据伪装成真实用户日志。

ABO spins 经 aria2 piece bitmap 断点恢复。aria2 可能预分配 `.part` 的逻辑长度，因此文件大小接近甚至等于清单值时仍不能推断下载完成；只有单一 aria2 进程正常退出、相邻 `.part.aria2` 控制文件消失且最终长度精确匹配后，项目下载器才可验证并原子转正。项目下载器现在会在控制文件存在时 fail closed，避免把有缺片的预分配文件误判为完整对象。

本地 `data/raw` 现有 96,960 个文件、89,842,305,694 字节；该数字包含既有 cache、状态和 recovery 文件，不是 formal dataset 大小。`clean/`、`kb/`、许可/PII 审批、source lock、selection/disposition、正式 Asset/KB catalog 与 Exact Match eligibility 仍未因此自动完成，core 继续 **NO-GO**。
