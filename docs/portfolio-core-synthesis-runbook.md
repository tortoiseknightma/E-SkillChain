# Portfolio core 语料合成运行手册

## 范围与状态

本手册仅适用于 Portfolio Track 的 `core=1,500` 条合成语料：

- `dev_mini=200`
- `opt_pool=800`
- `val=200`
- `test_frozen=300`

它不改写现有 `dev_mini` accepted ledger，也不把机器校验称为 Formal Research 的人工
接受。每个 core run 都位于独立目录，并由
`portfolio-core-run.json` 明示
`mechanical_validation_auto_approved_usable_pending_sample_review_v1` 策略。

每批仍须通过既有的 schema、plan、AssetCatalog、seed、重复文本、25 条数量和 provenance
校验；通过后会被自动标记为“可用、待抽检”。完成 1,500 条后才会生成一个确定性、分层的
**恰好 200 条** owner audit 包。用户通过该样本时，run 变为 `approved`；拒绝时，原 run
保持为 `rejected` 证据，必须以新的 run id/root 重新生成，绝不覆写旧语料。

## 启动前输入

建议把大体积输入与运行工件留在外接盘：

```text
E:\skillchain-data\clean\portfolio-core-v1\query_images\
E:\skillchain-data\clean\portfolio-core-asset-catalog-v1\
E:\skillchain-data\clean\portfolio-core-capability-assignments-v1.jsonl
E:\skillchain-data\runs\portfolio-core-20260804-r1\
```

`query_images` 必须含有五个 planner 文件夹：`exact_match`、`multi_product`、
`divergent_rec`、`encyclopedia`、`utility`。所有候选图片必须已从获准的 RAW 来源以
确定性、可复核的方式 materialize 到该 clean root；原始 archive 和 RAW 不得被该流程删除
或原地改写。

开始实际话术生成前，以下输入必须已就绪：

1. DeepFashion 当前解压完成并通过后处理检查；
2. core clean selection、其 create-only AssetCatalog，以及 capability-assignment JSONL
   已发布并逐文件校验；
3. 已接受的 dev_mini seed 位于
   `D:\athena\ECommerceSkillChain\data\queries\seeds\accepted`；
4. 当前 Codex 会话由用户明确确认模型显示名为 `5.6 Sol Ultra`。

第 4 项是创作门，不影响规划、运行目录准备和只读状态检查。

## 创建可恢复 run

先计算并固定 capability-assignment 文件的 SHA-256，然后运行：

```powershell
uv run python scripts/portfolio_core_synthesis.py prepare `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --run-id portfolio-core-20260804-r1 `
  --data-root E:\skillchain-data\clean\portfolio-core-v1\query_images `
  --seed-source-root D:\athena\ECommerceSkillChain\data\queries `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1 `
  --capability-assignments E:\skillchain-data\clean\portfolio-core-capability-assignments-v1.jsonl `
  --expected-capability-assignments-sha256 <64-lowercase-hex>
```

该命令只写 plan、seed 副本、active pointer 和运行 manifest；不生成任何自然语言 query。
准备过程在临时 sibling 目录完成后才 create-only 发布到 `--run-root`，因此中断不会暴露
半完成的 run。以完全相同输入重跑是幂等的；不同输入必须使用新的 run root。

## 连续生成与恢复

每轮只处理 active plan 的唯一 next batch。先发布或读取该批的可恢复 Codex 工作单；它不调用
外部 LLM API，也不生成 query 文本，而是 create-only 固定 25 个 plan item、图片绝对引用、seed、
catalog/plan provenance、manifest 要求及专属 inbox：

```powershell
uv run python scripts/portfolio_core_synthesis.py work-order `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1
```

返回的 `job_id`、`draft_path` 和 `draft_manifest_path` 是断点。Codex author 在满足当前模型
门禁后只向该 inbox 写该工作单的 draft 与 manifest；重跑同一命令会返回同一工作单和 checkpoint，
不会创建第二个 job。随后绑定该 job 进行机械校验/接受：

```powershell
uv run python scripts/portfolio_core_synthesis.py approve-batch `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --base-batch-id <status 返回的 next_batch_id> `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1 `
  --job-id <work-order 返回的 job_id>
```

状态与断点检查入口：

```powershell
uv run python scripts/portfolio_core_synthesis.py status `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1
```

如果进程恰好在 staging/accepted 发布后、checkpoint 或外层状态更新前中断，重试同一
`work-order` 或携带同一 `--job-id` 的 `approve-batch` 会从不可变批次状态恢复，不会重复生成或
覆盖。若存在当前 batch 的工作单，`approve-batch` 必须带对应 `--job-id` 且只能使用其 inbox。
运行中的 seed、plan、catalog 和 accepted ledger 必须保持同一 hash binding，否则 fail closed；
在报告 `auto_approved_usable_pending_sample_review` 前还会完整验证全部 accepted corpus。

## 200 条 owner audit 与冻结切分

当状态显示 1,500 条已自动批准可用后，发布审检包：

```powershell
uv run python scripts/portfolio_core_synthesis.py sample `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --sample-id owner-audit-200 `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1
```

样本按 `split × intent × capability × boundary` 分层，且 manifest 绑定完整
`queries.jsonl` SHA-256。审核结果不可改写：

```powershell
uv run python scripts/portfolio_core_synthesis.py review-sample `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --sample-id owner-audit-200 `
  --decision pass `
  --reviewer-id owner `
  --review-minutes <positive-integer>
```

若抽检拒绝，使用 `--decision reject --reason "..."`。随后创建新的
`portfolio-core-...-r2` run，并用 `--supersedes-run portfolio-core-20260804-r1`
建立可追溯关系；不得修改 r1。

通过后才能冻结 core split：

```powershell
uv run python scripts/portfolio_core_synthesis.py freeze-splits `
  --run-root E:\skillchain-data\runs\portfolio-core-20260804-r1 `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1
```

该步骤沿用 group-constrained split solver，发布 create-only 的
`split_assignment.jsonl`、`test_frozen.jsonl` 与 manifest，并锁定为 200/800/200/300。
