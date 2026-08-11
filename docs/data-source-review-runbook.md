# 数据源许可与 PII 审批手册

## 目的与边界

RAW 获取由独立会话负责。本流程不下载、不改写 RAW；它只在下载流程已经发布不可变 `source lock` 和许可证据后，由项目所有者作出可审计的人工作用域决定。

下载成功、公开可访问、存在 license 文本和“研究用途”是四个不同事实。任何 adapter 进入正式 `clean/` 前，都必须同时满足：

1. source lock 文件的 SHA-256 由下载/登记步骤之外保存；
2. 许可证据字节与 `license_evidence_sha256` 一致；
3. 人工 review record 精确绑定 source revision、source-lock SHA、license ID 和许可证据 SHA；
4. record 明确列出允许的用途和权限；
5. 含收据、对话或其他潜在个人信息的来源完成 PII 决定；
6. adapter 调用 `VerifiedSourceReviewLedger.require_approval(...)`，证明自己消费的就是被审核字节。

只按 `source_id` 查到一条“approved”记录不够；更换 revision、source lock 或许可页面后必须重新审批。

## 当前 policy

- portfolio：`specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`
- portfolio SHA-256：`6f2a0c98a2889aa6980e3f2308cfd6a076e5bdd60e0e7e99acc77f8db74b501a`
- policy：`specs/data_sources/mvp-source-review-policy-v2.json`
- policy 文件 SHA-256：`8a85cd4657ab24f5b78c6c0a1988c1f33cfe1180b9d9119f7693b7d53838513c`

v1 policy 只作为历史工件保留：它绑定旧 portfolio SHA-256，并把已经永久退出的
`jddc_2_0` 设为必需来源，不能用于当前 C2 审批。v2 的必需集合为
ABO、RPC、FashionIQ、WildReceipt、ISIA Food-500、Wikimedia/zhwiki、下厨房、
MUGE、DuRecDial 2.0 和 CrossWOZ；CORD、SROIE、iNaturalist 与 Wikimedia
Documents 保持可选。`codex_mock_trajectories` 是派生输出，不是需要下载许可的外部
source，因此不进入 source-review ledger；它由独立的 `synthetic_derived`
provenance、逐条事实绑定和人工接受门约束。

验证 policy：

```powershell
uv run python scripts/review_data_sources.py policy-status `
  --policy specs/data_sources/mvp-source-review-policy-v2.json `
  --policy-sha256 8a85cd4657ab24f5b78c6c0a1988c1f33cfe1180b9d9119f7693b7d53838513c `
  --portfolio specs/data_sources/ecommerce-mvp-source-portfolio-v1.json
```

## 所有者如何填写 ledger

ledger 是按 `source_id` 排序、每行一个 canonical JSON object 的 JSONL。不要让下载器或 adapter 根据 URL 自动生成 `decision=approved`；它们只能准备证据。

2026-07-25 先完成了一套不冒充 owner 的 v1 保守审批建议：

- 10 个 source lock：`specs/data_sources/c2/source-locks/`；
- source-lock manifest SHA-256：
  `8b9da4b6bff5e57c5c23a52330ca4dc5e87d4b693e27c3b1d124b7e49259958c`；
- 10 份逐源许可证据：`specs/data_sources/c2/source-review/license-evidence/`；
- owner proposals：
  `specs/data_sources/c2/source-review/owner-review-proposals.jsonl`；
- proposal SHA-256：
  `943e16dba66814217b085f78a286738dc32efc03272159d3839619d815d0f5dd`；
- bundle manifest SHA-256：
  `e06cf8041a378856176c4552d219fb41d55ed41eb33acab56f5a557d13c65f9d`。

项目所有者随后明确要求 10 个 required source 全部批准并签署。为保留审计历史，没有
覆盖 v1，而是发布并签署了 v2：

- proposal plan：`specs/data_sources/owner-review-proposal-plan-v2.json`；
- proposal：`specs/data_sources/c2/source-review-v2/owner-review-proposals.jsonl`；
- proposal SHA-256：
  `a29e261df3183cf670e31d9e55a5952b8a7cee1ea0321588148b8c541b855ed1`；
- bundle manifest SHA-256：
  `a316d616007838f1d3bb6bb662cd1190d0819d82309aa01dba78ca5318e2eb1b`；
- owner-signed ledger：
  `specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl`；
- ledger SHA-256：
  `8bf9ad9d2dd786c84b1849c562557cac327ab1cef4d653e7588e7dbc7af215c9`；
- signature receipt：
  `specs/data_sources/c2/source-review-v2/signed/owner-ledger-signature-receipt.json`；
- receipt SHA-256：
  `028da86082524161ec4f5727d5196cb938ad7efecfbddb3358544b447fc2a642`。

v2 的 `approved` 是项目所有者对限定用途的风险接受，不是新增的上游授权。ISIA、
MUGE、WildReceipt 和下厨房仍保留 `license_id=NOASSERTION`；zhwiki 仍保留第三方过滤包
的 provenance/attribution 缺口。10 个来源全部禁止 remote embedding、redistribution 和
public demo；CrossWOZ、DuRecDial、WildReceipt 和下厨房继续为 `pii_status=restricted`。

source locks 实际读取并提交了 75,368 个文件、78,692,489,139 字节。普通大归档按
排序后的 `(relative path, bytes, SHA-256)` 生成 canonical manifest digest；
FashionIQ 还把 75,267 张 accepted image、固定 annotation/URL inventory commit 和
2,416 条 exclusion ledger 一起纳入树承诺。锁构建过程没有下载或改写 RAW。

可按来源重新读取 RAW 并复验；省略 `--source` 时会复验全部 78.69 GB：

```powershell
uv run python scripts/verify_required_source_locks.py `
  --raw-root D:\athena\ECommerceSkillChain\data\raw `
  --source wildreceipt
```

构建和 proposal 输出目录均为 create-only。需要新 revision 时必须换新的 plan/output
路径并重新审批，不能覆盖旧锁。

历史 v1 proposals 的保守建议如下：

| 来源 | 建议决定 | PII | 允许边界/阻断原因 |
|---|---|---|---|
| ABO | approved | not applicable | CC BY-NC 4.0；仅非商业本地研究与本地 embedding |
| RPC | approved | not applicable | CC BY-NC-SA 4.0；仅非商业本地研究与本地 embedding |
| FashionIQ | approved | not applicable | 固定 commit 无 LICENSE，按 CodaLab academic-research-only 的更窄边界 |
| DuRecDial 2.0 | approved | restricted | 仅本地 `interaction_pattern`；raw utterance 不复制进 Mock |
| CrossWOZ | approved | restricted | 仅本地 `interaction_pattern`；对话/地点记录不离开本地边界 |
| WildReceipt | deferred | restricted | 官方元数据为 `License: N/A`，收据还有识别/交易信息风险 |
| ISIA Food-500 | deferred | not applicable | 公开论文和下载页没有给出图像数据集复用许可 |
| MUGE | deferred | not applicable | 缺 exact MUGE agreement/官方 revision，本地包来自第三方镜像 |
| Wikimedia/zhwiki | deferred | not applicable | 本地第三方过滤包缺官方 dump、变换与 attribution inventory |
| 下厨房 | deferred | restricted | 第三方 archive 无开放复用许可；平台用户授权不等于公众授权 |

`restricted` 是“按潜在 PII 留在本地隔离边界”的决定，不是“已证明没有 PII”。所有
proposal 均禁止 remote embedding、redistribution 和 public demo。

每条记录至少回答：

- `decision`：`approved`、`deferred` 或 `rejected`；
- `purposes`：只批准实际用途，如 `product_gallery`、`capability_gold`、`tool_gold`、`language_style`、`interaction_pattern`；
- `permissions`：download/local research/local embedding/remote embedding/redistribution/public demo 分别判断；
- `pii_status`：`not_applicable`、`reviewed_no_pii`、`restricted` 或 `redacted`；
- `notes`：记录许可冲突、署名要求、地域/展示限制或仍有疑问的地方。

保守规则：

- 没有明确证据的权限写 `false`；
- `restricted` PII 自动禁止 remote embedding、redistribution 和 public demo；
- `redacted` 必须绑定独立 redaction policy SHA；
- ABO 的许可冲突在未解决前按更严格的非商业/本地边界处理；
- WildReceipt 只作为收据 OCR/KIE 来源。官方元数据不能支持“中文收据数据集”表述，且 license 显示为未给出时不得推断 redistribution/public demo；
- JDDC 2.0 已永久退出，不得出现在 v2 ledger；旧 review record 不能迁移成 DuRecDial/CrossWOZ 的批准；
- DuRecDial 2.0 与 CrossWOZ 只批准 `interaction_pattern`，MUGE 只批准 `language_style`；三者都不能提供 capability、商品、图片或事实 gold，raw utterance 也不能直接复制进 Mock；
- DuRecDial/CrossWOZ 在进入 pattern inventory 前必须完成 PII review；其原始内容不获得 remote embedding、redistribution 或 public demo 的默认许可；
- Codex Mock 不使用 `download_allowed` 伪装成外部 source approval；必须保持 `data_origin=synthetic_derived` 并通过单独人工审核；
- 对许可证据不足的来源，即使 owner 批准，也不得把内部风险接受表述成上游许可，
  不得扩大到 remote embedding、再分发或公开展示。

验证完成的 ledger：

```powershell
uv run python scripts/review_data_sources.py verify `
  --policy specs/data_sources/mvp-source-review-policy-v2.json `
  --policy-sha256 8a85cd4657ab24f5b78c6c0a1988c1f33cfe1180b9d9119f7693b7d53838513c `
  --portfolio specs/data_sources/ecommerce-mvp-source-portfolio-v1.json `
  --ledger specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl `
  --ledger-sha256 8bf9ad9d2dd786c84b1849c562557cac327ab1cef4d653e7588e7dbc7af215c9
```

只有命令输出 `status=approved` 才说明 policy 要求的来源均有合格决定；它不替代 adapter 对具体 source lock 的再次绑定。

### 显式 owner 确认

proposal 不是 owner ledger。项目所有者已否决 v1 的五批五缓建议，明确要求 v2 的
10 个来源全部批准，并授权以 `project-owner` 身份签署。实际执行的 create-only
materialization 为：

```powershell
uv run python scripts/prepare_source_review_bundle.py materialize-owner-ledger `
  --proposals specs/data_sources/c2/source-review-v2/owner-review-proposals.jsonl `
  --proposal-sha256 a29e261df3183cf670e31d9e55a5952b8a7cee1ea0321588148b8c541b855ed1 `
  --reviewer-id project-owner `
  --reviewed-at 2026-07-25T02:48:25.8435178Z `
  --owner-instruction "Approve all ten required sources and sign the resulting owner ledger." `
  --output specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl `
  --receipt-output specs/data_sources/c2/source-review-v2/signed/owner-ledger-signature-receipt.json
```

命令使用 create-only 输出；不会覆盖已有 owner ledger。签署回执同时绑定 proposal 与
ledger 的外部 SHA，并明确声明 owner approval 不替代缺失的上游许可证据。可用
`review_data_sources.py status` 验证 canonical bytes、外部 SHA 和 blocker：

```powershell
uv run python scripts/review_data_sources.py status `
  --policy specs/data_sources/mvp-source-review-policy-v2.json `
  --policy-sha256 8a85cd4657ab24f5b78c6c0a1988c1f33cfe1180b9d9119f7693b7d53838513c `
  --portfolio specs/data_sources/ecommerce-mvp-source-portfolio-v1.json `
  --ledger specs/data_sources/c2/source-review-v2/signed/owner-source-review-ledger.jsonl `
  --ledger-sha256 8bf9ad9d2dd786c84b1849c562557cac327ab1cef4d653e7588e7dbc7af215c9
```

当前 `status` 和严格 `verify` 均输出 10 个 approved、零 blocker 和
`status=approved`。若以后改变任何决定，必须发布新的 proposal、ledger 和 receipt，
不能在签署后静默改写 v2。

## 当前仍需人工完成

10 个 required source 的 owner review ledger 已签署并通过严格 gate，不再需要 source
replacement 才能启动 adapter。ABO formal review-packet/audit/build/verified-loader 已接入
`VerifiedABOSourceApproval`：候选 manifest v2 分别绑定 owner-approved required lock、
source-review policy/ledger/record/license evidence 与 adapter normalized lock；任何摘要、
用途、权限、scope 或发布前字节漂移都会失败关闭。

这项机制检查同时确认当前真实 ABO compact lock 只覆盖 `abo-images-small.tar`、
`abo-listings.tar` 和 `abo-spins.tar`，并不覆盖现有 Exact Match adapter 消费的解包
listing/image metadata 与定向 `images/original` 字节。因此当前 ABO formal run 会按设计
阻断。下一步必须选择并冻结一种真实数据路径：扩展 required lock 覆盖全部实际 consumed
RAW 后重新取得 owner exact approval，或改造 adapter 只消费当前已批准 scope；不能把
compact lock 与 normalized lock 冒充同一 trust root。

C2 整体仍未通过：还需把同一 gate 接入其余 adapter，并完成 Mock 人审、
`clean/`/`kb/`、selection/disposition、AssetCatalog/KB catalog、Exact Match eligibility、
隔离 gold 与 leakage 审计。`NOASSERTION` 与 restricted PII 限制仍是必须在最终报告披露的
风险边界。
