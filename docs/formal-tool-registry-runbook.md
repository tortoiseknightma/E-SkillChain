# 正式工具 Registry 签发手册

## 当前结论

生产装配入口已经实现，但仓库当前还不能生成真实 runtime lock，也不能宣称 C3 已关闭。2026-07-23 的本地盘点结果是：

- `data/clean/` 没有正式清洗产物；
- `data/queries/split_assignment.jsonl` 不存在；
- product index、KB catalog/bundle、detector/OCR 模型 manifest 均不存在；
- AssetCatalog 与 document-safety catalog 尚未发布；
- `data/index/tool-registry.json` 不存在。

`data/raw/` 中已经下载或正在下载的文件不能直接替代这些正式工件。测试 fixture 也只能证明机制，不能生成正式实验身份。

## 为什么需要两份锁

`build_mvp_registry` 内部的 authority 只能证明“这个 Python 进程中的 registry 来自审核过的七工具 concrete service graph”。它不知道调用方提供的模型、索引和 catalog 是否经过项目外部批准。

因此生产入口使用两份相互绑定、且各自需要外部文件 SHA-256 的锁：

1. `FormalRegistryArtifactLock` v2：锁定 embedding provider（远程或本地多语种 OpenCLIP）、product index/query、KB catalog/index、detector/OCR manifest、AssetCatalog、document-safety catalog，以及 canonical ToolSpec registry。
2. `FormalRegistryRuntimeLock`：锁定 authority 签发后得到的 registry/runtime 摘要、七个 tool runtime binding 和 detector/OCR evidence binding，并反向绑定 artifact-lock 文件字节。

仅在本进程中重新计算一个摘要不构成外部信任根。正式加载时必须同时传入两份锁文件的预先登记 SHA-256。

`registry_runtime_sha256` 还包含被 authority 审核的 registry/service/backend 方法源码指纹。这样另一个进程即使保留相同版本字符串和 artifact 摘要，只要修改了正式实现，也无法重放旧 runtime lock。

三个 product retrieval ToolSpec 使用稳定的 `runtime_bound_embedding` 网络策略：具体 runtime 可以是完全本地，也可以是逐资产许可门控的远程 embedding。ToolSpec 不随机器选择漂移；authority/runtime lock 另外证明本次究竟是 `execution_location=local` 还是 `remote`。因此本地轨不会被错误记录成远程上传，远程轨也不能借这个抽象绕过 `cloud_upload_allowed=true`。

## Artifact lock 字段语义

所有路径均相对于单独传入的 `--artifact-root`，必须使用 POSIX 相对路径，且路径链中不能有符号链接。

| 字段 | `sha256` 的含义 |
|---|---|
| `product_index` | `manifest.json` 的文件 SHA-256 |
| `product_query_artifact` | query artifact 的文件 SHA-256 |
| `embedding_manifest` | 本地 OpenCLIP `ModelArtifactManifest.manifest_sha256`；DashScope 轨必须为空 |
| `kb_catalog` | `KBCatalogManifest.catalog_sha256` |
| `kb_index` | `KBIndexBundleManifest.bundle_sha256` |
| `detector_manifest` | `ModelArtifactManifest.manifest_sha256` |
| `ocr_manifest` | `ModelArtifactManifest.manifest_sha256` |
| `asset_catalog_sha256` | `AssetCatalogManifest.catalog_sha256` |
| `document_safety_catalog_sha256` | `DocumentSafetyCatalogManifest.catalog_sha256` |
| `document_safety_review_ledger_sha256` | 经人工审批的精确 `approvals.jsonl` 摘要 |
| `tool_spec_registry_sha256` | canonical 七工具 ToolSpec registry 摘要；正式来源 schema 与 runtime-bound embedding policy 修正后的当前代码为 `91752c8c766138dddfd79b382af9d6d298d93a9471d22133aac866b565752795` |

真实 artifact lock 不应在这些工件发布前用零值或 fixture 值占位。当前仓库因此刻意没有提交一份冒充可用的正式锁。

## 签发流程

### 1. 独立登记 artifact lock

从已审批、create-only 发布的工件填写 canonical JSON，保存为新的 artifact lock。由独立步骤计算并登记该文件的 SHA-256；不要让后续运行脚本一边修改锁一边接受自己刚算出的摘要。

### 2. 生成 runtime-lock candidate

```powershell
uv run python scripts/issue_formal_registry.py candidate `
  --artifact-root D:\path\to\formal-artifacts `
  --artifact-lock D:\path\to\formal-registry-artifacts-v2.json `
  --expected-artifact-lock-sha256 <externally-pinned-sha256> `
  --destination D:\path\to\formal-registry-runtime-v1.candidate.json
```

该命令会：

- 逐层复验所有外部摘要和真实文件；
- 使用无 cache 的已选 embedding backend 执行一次 live canary；DashScope 会产生远程请求，本地 OpenCLIP 不上传资产；
- 实际加载锁定版本的 Ultralytics detector 和 RapidOCR engine；
- 构造 exact concrete services；
- 由内部 authority 签发 canonical 七工具 registry；
- create-only 写出 runtime-lock candidate。

DashScope 轨可能调用一次付费 API，并需要 `DASHSCOPE_API_KEY`。本地轨必须锁定 `xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k` 权重、配置和全部本地 tokenizer 文件，且运行时禁止联网补文件。candidate 输出固定标记为 `formal_eligible=false`；它还没有权力批准自己。

### 3. 独立冻结 runtime lock

审核 candidate 中每个工具和 evidence runtime binding，把 candidate 作为正式 runtime lock 保存，并在运行脚本之外登记它的文件 SHA-256。任何 artifact、包版本、endpoint policy 或 ToolSpec 变化都应生成新锁，不能覆盖旧锁。

### 4. 重建并验证

```powershell
uv run python scripts/issue_formal_registry.py verify `
  --artifact-root D:\path\to\formal-artifacts `
  --artifact-lock D:\path\to\formal-registry-artifacts-v2.json `
  --expected-artifact-lock-sha256 <externally-pinned-artifact-lock-sha256> `
  --runtime-lock D:\path\to\formal-registry-runtime-v1.json `
  --expected-runtime-lock-sha256 <externally-pinned-runtime-lock-sha256>
```

只有该命令输出 `authority_issued=true` 与 `formal_eligible=true`，且退出码为 0，才得到本进程可用的正式 registry。

## 接入 formal evaluator

`formal_evaluation_cli` 可直接使用环境工厂：

```powershell
$env:SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_ROOT = 'D:\path\to\formal-artifacts'
$env:SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK = 'D:\path\to\formal-registry-artifacts-v2.json'
$env:SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK_SHA256 = '<externally-pinned-sha256>'
$env:SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK = 'D:\path\to\formal-registry-runtime-v1.json'
$env:SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK_SHA256 = '<externally-pinned-sha256>'

uv run python -m skillchain.tools.formal_evaluation_cli `
  --runtime-factory skillchain.tools.production_registry:build_formal_runtime_context_from_env `
  <create-run|verify-run|evaluate-run 参数>
```

环境工厂同时返回相同 concrete services 构造的 `MultiProductSearchService`，避免组合链绕开已锁定的 product/detector runtime。

## 失败语义

下列任一情况必须退出非零，且不能降级到 diagnostic registry：

- 锁文件不是 canonical JSON、包含未知字段，或文件摘要不匹配外部值；
- 路径逃逸 artifact root、包含符号链接或工件缺失；
- artifact lock 与 runtime lock 不互相绑定；
- product query/index、KB、模型或安全审批摘要漂移；
- embedding canary 失败、模型包版本不符、detector/OCR 无法真实加载；
- canonical ToolSpec 或任一 runtime/evidence binding 与 runtime lock 不同；
- 被审核的 registry/service/backend 实现源码与冻结 runtime 不同；
- 任一 service 不是 registry authority 审核的 exact concrete type。

## 当前下一步

先完成 source review、清洗、统一 AssetCatalog 与 query/gallery eligibility，再构建 product/KB index。仓库已经能真实加载 RapidOCR candidate，但只有通过独立 document gold 后才能把候选升级为正式 OCR；detector 必须在 RPC SKU gold 上验证，不能用通用 COCO 权重冒充。多语种 OpenCLIP 轨也必须先完成资源与 ranking gold 验证。只有这些输入都存在，才填写第一份 artifact lock 并运行 candidate。不要从单元测试 fixture 或 candidate spec 复制摘要。
