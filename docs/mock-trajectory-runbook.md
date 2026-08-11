# MVP Mock 轨迹准备与生成手册

## 决策边界

自 2026-07-24 起，JDDC 2.0 永久退出 MVP、Full 和 full-upstream，不再等待授权，
也不再作为下载或实现阻断。MVP 改用以下组合做机制级轨迹复现：

- DuRecDial 2.0：偏好逐步暴露、追问、推荐与话题转移模式；
- CrossWOZ：约束冲突、无结果、条件放宽、多次查询和动态目标修改模式；
- MUGE：中文电商短查询与商品表达方式；
- Codex 受控生成：把上述模式重写到项目权威 TaskSpec，并只绑定已经验证的商品、
  图片和 KB 事实。

DuRecDial/CrossWOZ/MUGE 只提供语言或交互结构，不提供 capability、图片、商品或事实
gold。生成结果一律为 `synthetic_derived`，不得称为真实用户日志。MVP 只能报告机制
级复现，不能报告 JDDC 2.0 的真实流量规模、原始多模态用户分布或商品知识分布已经
复现。机器可读规则见
[`dialogue-trajectory-source-policy-v1.json`](../specs/data_sources/dialogue-trajectory-source-policy-v1.json)。

## 获取并验证模式来源

固定提交的 DuRecDial 2.0 与 CrossWOZ 已进入 MVP RAW profile，运行：

```powershell
uv run python scripts/download_raw_datasets.py download --profile mvp --source durecdial_2_0 --source crosswoz
uv run python scripts/download_raw_datasets.py verify --profile mvp --source durecdial_2_0 --source crosswoz
```

正式选取对话现象前还必须生成 source lock、记录数据许可证和用途限制，并建立只包含
结构标签的 pattern inventory。不得把原始旅游实体、人物、商品名或整句对话复制到
Mock；建议的 pattern 标签至少包括：

- `preference_elicitation`
- `topic_transition`
- `constraint_conflict`
- `no_offer`
- `constraint_relaxation`
- `goal_change`
- `multi_query`

pattern inventory 只描述“发生了什么交互变化”，不携带对话数据中的实体或事实。
它是与论文意图正交的交互维度，不得写入或替换 `canonical_intent`。顶层意图始终以冻结
taxonomy 和论文 Table 1 为准：
`exact_match`、`multi_product`、`divergent_rec`、`encyclopedia`、`utility`。

## Phase 3 生成门

本项目的正式 seed/trajectory 只能通过仓库内 `generate-phase3-corpus` 工作流创作。
真正生成前，用户必须在当前 Codex 会话明确确认模型显示名为 `5.6 Sol Ultra`：

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries guard-model --confirmed-model "5.6 Sol Ultra"
```

没有这项当轮确认时只能运行只读状态命令，不能查看计划图片、创作 seed、写 inbox
或 staging：

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries status --json
uv run python scripts/synth_queries.py --queries-root data/queries stats --json
```

2026-07-28 的最终检查点为：

- `seed_status=accepted`
- `accepted_seed_batch_id=phase3-paper-five-seeds-20260725-r1`
- `seed_set_sha256=767a10caa954890088efe85a3b5ba50628e0ab772bafdf06b6b43dc7159c3250`
- `accepted_batches=8`
- `accepted_queries=200`
- `active_plan_sha256=5e3b0d67545c3d9a311df6c133c427564e4064174158eddf603b3b075e9fa6be`
- `asset_catalog_sha256=df17da7dd7ad7e1077c2e6d084a5e1e79516021979915c9881314b7815c28c96`
- `next_batch_id=null`
- `next_revision=null`
- query staging 与 seed staging 均为空

八个 accepted 批次依次为 `dev-mini-001-r3`、`dev-mini-002-r1`、
`dev-mini-003-r1`、`dev-mini-004-r1`、`dev-mini-005-r2`、
`dev-mini-006-r2`、`dev-mini-007-r1`、`dev-mini-008-r1`。拒绝修订
`dev-mini-001-r1`、`dev-mini-001-r2`、`dev-mini-005-r1` 和
`dev-mini-006-r1` 继续作为历史证据保留。聚合 `queries.jsonl` 为 315,195 bytes，
SHA-256=`6f8eda4fe663733708d6e797c954e58f098d3938857c6f2b143e24e202437f03`；
accepted ledger SHA-256=`79835fa73bfde60a005ea16480c1efc04635b3fee64f8a01bae893fe5cd4e69a`。

200 条 query 均经人工审阅并正式接受；不能把它描述成“200 条人工标注”，因为 intent、
capability 和 boundary 仍是 plan-owned auto label。最终五意图分布为
Exact/Multi-Product/Divergent/Encyclopedia/Utility=`35/35/35/35/60`，六 capability
分布为 Exact/Multi-Product/Style/Encyclopedia/Document/Recipe=`35/35/35/35/30/30`，
含 40 条 boundary、188 条单轮和 12 条澄清轨迹。所有结果仍标记为
`synthetic_derived`；该检查点只关闭本地 `dev_mini` query corpus 构建，不表示五配置实验、
Stage 1 Creator、工具 gold 或 formal/core 结论已完成。

`data/` 整体由 Git 忽略，因此 accepted corpus、ledger 和审阅页是本地工件，不随代码提交；
上述摘要用于绑定当前本地快照，不把 ignored 数据冒充 tracked artifact。审阅决定与页面计时
也没有导出为仓库内 query-review ledger；已知 `review_minutes` 是页面 elapsed timer，不应
当作主动人工工时。

## Mock 轨迹生成规则

种子接受且 active plan 就绪后，按每批恰好 25 条生成。计划拥有图片、intent、
capability、boundary、split、顺序与 ID；模型只能填写 `plan_id` 对应的 `turns`。

每个 draft manifest 和 staging manifest 必须包含：

```json
{
  "data_origin": "synthetic_derived"
}
```

同时逐条 Query 保留 `synth_provider`、`synth_model`、`synthesis_batch_id`、
`synthesis_prompt_id` 和 `seed_set_sha256`。质量报告也必须回显
`data_origin=synthetic_derived`。

生成时遵守以下事实边界：

1. TaskSpec 决定语义和 capability；
2. verified AssetCatalog/KB 决定商品、图片和事实；
3. DuRecDial/CrossWOZ/MUGE 只影响交互模式和中文表达；
4. Codex 不复制源语句，只做领域重写和组合；
5. 每批只写 staging，必须单独人工 review；不得自动 accept；
6. 任何来源、哈希、计划、重复、图片或模型门失败都应 fail closed。

全批通过时可以按 `roll-forward` 顺序执行“接受当前批 → 生成唯一后继批到 staging”；
后继批仍需独立人工接受。如果接受后 `status` 返回 `next_batch_id=null` 和
`next_revision=null`，说明计划已经耗尽，应在模型门、图片查看和写入前正常停止，不能
虚构第九批。

## Stage 1 交接边界

只有形成 authoritative accepted `opt_pool` 后，才能运行 create-only 的
`scripts/prepare_stage1.py`，而不是直接调用 Creator。它生成：

- `trajectory-bundle.json`：保留完整合法 `[user]` 或
  `[user, assistant, user]` 交互，只含论文五类 intent；
- `s1-creator-packet.json`：逐字节嵌入共同的
  `specs/authoring/authoring-packet-codex-high-v5.json`，并交叉校验 taxonomy/TaskSpec
  version。

该投影不得包含 label provenance、Judge、failure attribution、tool trace、gate 或
test 结果。CLI 输出 `prepared_not_invoked`、`model_invoked=false`、
`bank_generated=false`；在一次性 Creator authority、Engineer Check 和 Human
Reflection 规则另行冻结前，准备完成不能表述为 Stage 1 已运行。

当前 200 条 accepted query 的 split 全部是 `dev_mini`，不是 `opt_pool`；
`scripts/prepare_stage1.py` 会明确拒绝它们。不得通过手改 split、改写 accepted 目录或把
`dev_mini` 口头重命名为 `opt_pool` 来绕过此边界。Portfolio Track 若要用这 200 条启动
第一版 S1 闭环，必须先明确并实现与当前五配置 mini 目标一致的输入交接；Formal Research
Track 的 `opt_pool` 契约继续保持不变。

## Full 暂缓项

Full 的具体对话实现计划暂不冻结。U-NEED 已于 2026-08-01 因 owner 确认无法获取而永久退出；其原本候选的售前需求澄清机制继续由 DuRecDial 2.0、CrossWOZ、MUGE 的抽象模式和受控 Codex 重写承接，输出始终为 `synthetic_derived`，不得作为真实电商对话或用户分布证据。取得其余候选数据后重新评审：

- SIMMC 2.1：官方 Git LFS inventory、CC BY-NC-SA 条款、英语虚拟场景边界，以及
  中文改写继续标记为 `synthetic_derived`；
- CSDS：只有取得覆盖使用、派生和再发布的书面许可后才考虑；
- 用户提及的 `JDDC 2.1`：在拿到官方包和条款前仅登记为身份待核验标签，不能与
  SIMMC 2.1 混称，也不能提前写进 Full 实现。

授权到手后必须重新提交 Full source decision；本文件不预先承诺各来源配额、
adapter 或实验角色。
