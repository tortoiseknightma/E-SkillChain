# RAW 数据下载运行手册

本手册用于在独立 Codex 会话、普通 PowerShell 或终端中下载 MVP / full 所需 RAW 数据。下载器只负责取得和验证原始字节，不会解压、清洗、批准许可证、去除 PII，也不会把文件自动升级成正式 Asset/KB catalog。

机器可读下载清单是 [`raw-download-profiles-v1.json`](../specs/data_sources/raw-download-profiles-v1.json)，来源的监督角色仍以 [`ecommerce-mvp-source-portfolio-v1.json`](../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json) 为准；对话模式与 Mock 的专门边界见 [`dialogue-trajectory-source-policy-v1.json`](../specs/data_sources/dialogue-trajectory-source-policy-v1.json)。

## 当前 MVP 检查点

2026-07-24 的本地运行结果为：MVP profile 的 34 个活动项全部为 `complete`，其中 31 个本地字节项合计 82,247,477,174 字节，另有 3 个 command 管理项；`verify --profile mvp` 对所有 artifact 返回 `[ OK ]`。固定提交并带 SHA-256 的 DuRecDial 2.0（10,112,093 字节）与 CrossWOZ（20,684,500 字节）取代 JDDC 2.0 作为 MVP Mock 的对话模式来源。人工取得的 SROIE 5 个语义唯一官方包（1,205,755,849 字节）通过内容与结构校验，CORD v2 固定提交完成逐 SHA 校验。FashionIQ 固定清单收口为 75,267 张有效图片和 2,416 条显式排除。RPC Kaggle v5 包（27,205,167,166 字节）在上游旧 URL 返回 404 后，使用 Git 忽略的 `local_artifacts` override 对完整 partial 执行大小和 SHA-256 校验并原子转正；机器可读 acquisition 收据见 [`rpc-kaggle-acquisition-v1.json`](../specs/data_sources/rpc-kaggle-acquisition-v1.json)。JDDC 2.0 已永久退出所有 profile。这个检查点只证明 RAW acquisition/候选快照完成，不证明许可、PII、外部 source lock、selection、catalog 或 formal readiness；Mock 也必须另经 synthetic provenance 与人工审核。

## 三种 profile

| Profile | 含义 | 适用场景 |
| --- | --- | --- |
| `mvp` | MVP 主源、语言源、KB、secondary 与有界 challenge 来源 | 先关闭 200-query mini 的数据缺口 |
| `full` | `mvp` 加项目 full 计划的 core/candidate/challenge 扩展 | 4,500-query full 的项目级数据准备 |
| `full-upstream` | `full` 再加上游完整发布镜像 | 仅在确实需要数 TB 原始上游字节时使用 |

`full` 指项目计划所需的有界数据组合，不等于镜像所有上游 bucket。U-NEED 已于 2026-08-01 因无法获得而永久退出所有 profile；SIMMC 2.1 固定提交及七个 LFS 对象、CSDS 官方 `train/val/test` JSON 已在 2026-08-01 完成 acquisition，二者仍只作为 Full 条件候选以 `manual` 方式登记。SIMMC 2.1 仍需完成 CC-BY-NC-SA 激活评审；项目 owner 已确认 CSDS 具有覆盖使用、派生和再发布的书面许可。包已落盘或获许可不表示实现角色、配额或 adapter 已冻结。`full-upstream` 会额外包含 ABO 原图、iNaturalist 2021 全发布、MEP-3M 599 卷、Open Images 和 Recipe1M+ 全发布等；它不是 4,500 条评测的默认前置条件。

## 在新会话中开始

先只看计划和当前已知的空间下界，不会联网下载：

```powershell
uv run python scripts/download_raw_datasets.py plan --profile mvp
uv run python scripts/download_raw_datasets.py plan --profile full
```

开始下载：

```powershell
uv run python scripts/download_raw_datasets.py download --profile mvp
```

下载过程中可以正常中断 Codex 会话或按 `Ctrl+C`。下次进入仓库后运行同一条命令即可继续。full 同理：

```powershell
uv run python scripts/download_raw_datasets.py download --profile full
```

只处理某一个或几个来源时，重复使用 `--source`：

```powershell
uv run python scripts/download_raw_datasets.py download --profile mvp `
  --source abo_compact `
  --source wildreceipt
```

查看跨会话状态和重新验证已完成文件：

```powershell
uv run python scripts/download_raw_datasets.py status --profile mvp
uv run python scripts/download_raw_datasets.py verify --profile mvp
```

默认 RAW 根目录是 `data/raw`，状态位于 `data/raw/.download-state/<profile>/state.json`。若要放到其他磁盘，之后每次都必须传入相同的 `--root`；`status` 还要传对应的 `--state-root`：

```powershell
uv run python scripts/download_raw_datasets.py download --profile full `
  --root E:\skillchain-data\raw `
  --state-root E:\skillchain-data\raw\.download-state
```

注意：清单中的 `command` 来源调用现有项目 adapter，当前仍写入仓库的 `data/`。因此使用自定义 `--root` 时，应先用 `--source` 只选择 `artifacts` 来源；有界 iNaturalist、Wikimedia Documents 和 MEP-3M selected 仍应在仓库默认数据目录单独执行。

## 断点续传语义

- 每个未完成文件写入相邻的 `<文件名>.part`；完成大小与可用 SHA-256 校验后才原子改名。
- 状态每 64 MiB 刷盘一次。意外终止后以 `.part` 的真实长度为准，不依赖最后一次日志。
- 恢复请求同时使用 `Range` 与 `If-Range`。返回 `206` 时严格检查 `Content-Range` 起点。
- 远端不提供强 ETag 或 Last-Modified 时，无法证明 partial 仍属于同一对象，因此会从头覆盖 `.part`，而不是冒险续接。
- 服务器忽略 Range 并返回 `200` 时，从头覆盖 `.part`，不会把整文件追加到半文件后面。
- 部分文件存在时，如果远端 `Content-Length`、`ETag` 或 `Last-Modified` 改变，默认停止并要求人工复核，防止跨版本拼接。
- 同一 profile 有独占锁，避免两个会话同时写同一组状态。只有确认原进程已结束后，才能使用 `--break-lock` 清除陈旧锁。
- 状态文件中的 URL 会移除 query、fragment 和 userinfo；不要把 token、cookie 或密码写进清单。

退出码为 `0` 表示选定来源全部完成，`1` 表示至少一个公开下载/命令失败，`2` 表示公开部分已处理但仍存在授权或人工来源。退出码 `2` 是预期的待办状态，不代表已经取得那些受限数据。

### aria2 位图与项目 partial 的兼容边界

外部 aria2 可用于恢复特别大的公开对象，但它的 `.part` 可能是预分配或稀疏写入：逻辑长度不是连续已下载字节数，真实完成度由相邻 `.part.aria2` piece bitmap 和 aria2 控制台决定。不要同时运行 aria2 和项目下载器，也不要删除或覆盖 `.aria2`、partial、recovery 文件。

项目下载器检测到 `.part.aria2` 时会 fail closed。只有确认 aria2 单进程正常结束、控制文件消失、partial 精确达到 manifest 大小后，才运行同一个 profile 命令，让项目下载器重新验证并原子转正。2026-07-24 的 ABO spins 实际恢复遵循此路径：42,446,704,640 字节 partial 在控制文件消失后被转为 `abo-spins.tar`，旧 Python `downloaded_bytes` 没有覆盖 live bitmap 的判断。

## 授权来源接力

RPC、FashionIQ、CORD/SROIE、DeepFashion、SIMMC 2.1、CSDS、Products-10K、Polyvore 与 Recipe1M+ 等来源不会使用猜测的第三方镜像。JDDC 2.0 与 U-NEED 均已永久退出，不应再为它们创建 override。得到授权 URL 后，在 Git 忽略的 `data/raw/.download-state/manual-overrides.json` 创建本地清单：

```json
{
  "schema_version": 1,
  "sources": {
    "rpc": {
      "kind": "artifacts",
      "revision_policy": "owner-approved-release-identity",
      "artifacts": [
        {
          "path": "rpc/official-package.zip",
          "url": "https://authorized.example/download/signed-url",
          "bytes": 123456789,
          "sha256": "替换为64位小写十六进制摘要"
        }
      ]
    }
  }
}
```

然后把 overrides 作为全局参数放在子命令之前：

```powershell
uv run python scripts/download_raw_datasets.py `
  --overrides data/raw/.download-state/manual-overrides.json `
  download --profile mvp
```

### Recipe1M+ 有界图文配对选择

Recipe1M+ 的授权元数据完成后，不能直接恢复任一整包图像归档。先由项目 owner
对中文菜品实体和英文食谱标题作精确审核，并在 Git 忽略的外接盘状态目录创建映射；
不接受模糊匹配或自动翻译猜测。例如：

```json
{
  "schema_version": 1,
  "reviewed_by": "owner",
  "reviewed_at": "YYYY-MM-DD",
  "max_total_images": 120,
  "entities": [
    {
      "entity_id": "reviewed_dish_id",
      "recipe_title_aliases": ["Exact Recipe1M+ title"],
      "max_recipes": 3,
      "max_images_per_recipe": 1
    }
  ]
}
```

当前 core 选择已根据 owner 授权的同名人工审核，固定为
[`recipe1m-plus-reviewed-entity-map-v1.json`](../specs/data_sources/recipe1m-plus-reviewed-entity-map-v1.json)。
它只包含能与当前 ISIA Food-500 本地卷严格同名对齐的五个实体；没有精确同名食谱的
类别明确不进入该选择。

生成选择清单时，工具只接受通过 `det_ingrs.json` 有效性标记且在 `layer2+.json` 有图像
关联的食谱；旧 `layer2.json` 会被读取并在 inventory 中记录其覆盖情况，但扩展集独有的
食谱不会因未出现于旧层而被排除。`inventory.json` 只含 recipe/image ID，绝不写入图像
URL。URL 只会进入同样位于 Git 忽略目录的 aria2 输入文件，随后才可以启动单连接、
可断点续传的有界图像下载：

```powershell
uv run python scripts/plan_recipe1m_plus_selection.py `
  --layers-archive E:\skillchain-data\raw\recipe1m_plus\authorized_core\recipe1M_layers.tar.gz `
  --det-ingrs E:\skillchain-data\raw\recipe1m_plus\authorized_core\det_ingrs.json `
  --layer2-plus E:\skillchain-data\raw\recipe1m_plus\authorized_core\layer2+.json `
  --reviewed-map E:\skillchain-data\raw\.download-state\recipe1m_plus\reviewed-entity-map.json `
  --inventory E:\skillchain-data\raw\recipe1m_plus\selected\inventory.json `
  --aria2-input E:\skillchain-data\raw\.download-state\recipe1m_plus\selected-images.txt
```

输出文件一旦存在且内容不同，工具会拒绝覆盖；修改审核映射必须另建一次选择目录，保留原
selection 的可复核性。该步骤的边界是下载经审核的少量图像，不是镜像 Recipe1M+ 的图像
归档。aria2 将图像暂存为 `selected_images/<image_id>.jpg.part`；仅在 JPEG 完整可解码后，
才可原子移动到 `selected/images/<image_id>.jpg`。重新生成输入时会跳过已经验证并发布的
图像，不会覆写它们。

### Products-10K 京东鲸盘分卷接管

若已登录的分享页只能逐卷下载，不要改名、移动或删除浏览器落在 `D:\Downloads` 的
`train_part.z01` … `train_part.zNN` 与 `train_part.zip`。全部分卷落盘后，下面命令会按
`z01 → zNN → zip` 顺序流式写入外接盘 `train.zip.part`，仅在官方 `md5.txt` 匹配时原子
转正为 `train.zip`；浏览器原始分卷始终保留：

```powershell
uv run python scripts/assemble_products10k_jd_parts.py `
  --parts-dir D:\Downloads `
  --stem train_part `
  --destination E:\skillchain-data\raw\products_10k\archives\train.zip `
  --md5-manifest E:\skillchain-data\raw\products_10k\archives\md5.txt `
  --archive-name train.zip
```

若分享中同样提供测试集分卷，替换 `train_part`/`train.zip` 为对应的 stem 和归档名即可。
分卷缺失、空文件、顺序不连续或 MD5 不匹配时，工具保留 `.part` 并拒绝把它当成数据集成品。

### MVP 对话模式快照

DuRecDial 2.0 与 CrossWOZ 使用官方仓库 commit archive，而不是移动的默认分支：

```powershell
uv run python scripts/download_raw_datasets.py download --profile mvp `
  --source durecdial_2_0 --source crosswoz
uv run python scripts/download_raw_datasets.py verify --profile mvp `
  --source durecdial_2_0 --source crosswoz
```

当前冻结身份分别为：

- DuRecDial 2.0 commit `1309cc072afd0b832e899e49c62906e700ff6acf`，
  10,112,093 字节，SHA-256
  `5ebe6e7052b5c01fbdfb257b1c502163607cee7a770f6dba1b42515e5cfe6905`；
- CrossWOZ commit `df82c9fdff91b9b130f2d6b89110d3870ba6260e`，
  20,684,500 字节，SHA-256
  `f710a480f88452ff0a2bc664555434caa029004dfe7a70edd91b88a5f00a37da`。

DuRecDial 数据受 CC BY-NC-SA 4.0 非商业限制，CrossWOZ 仓库为 Apache-2.0；
正式选取 pattern 前仍需 source lock 和用途审查。两者只允许抽象交互模式，不能提供
商品/图片事实或 capability gold。后续 Mock 步骤见
[`mock-trajectory-runbook.md`](mock-trajectory-runbook.md)。

下载器不会把 URL query 写入状态，但本地 overrides 自身可能包含签名 URL，必须留在 `data/` 下并禁止提交。若供应方只提供浏览器下载，把最终文件放到清单约定的 `data/raw/<path>`，先用独立 acquisition manifest 固定本地大小/SHA 和结构检查，再把来源覆盖为 `local_artifacts`。该类型不会伪造远端可重下能力，只校验已经取得的本地文件并将结果写入统一状态；`artifacts` 仍用于可通过 HTTP(S) 获取的对象。许可证证据、PII 审批、用途权限与外部 source lock 仍是后续独立 gate。

### SROIE 浏览器包接管

本地机器可读收据为 [`sroie-browser-acquisition-v1.json`](../specs/data_sources/sroie-browser-acquisition-v1.json)。复验内容身份与 ZIP 结构：

```powershell
uv run python scripts/verify_sroie_raw.py
```

通过 Git 忽略的 `manual-overrides.json` 写入统一下载状态：

```powershell
uv run python scripts/download_raw_datasets.py `
  --overrides data/raw/.download-state/manual-overrides.json `
  download --profile mvp --source sroie
```

当前选择 5 个语义唯一包；两份 `text.task1&2-test` 的成员名称与内容完全相同，只登记较早取得的一份。训练包存在 Google Drive 导出产生的 `(1)` 至 `(5)` 后缀副本，但按逻辑 ID 归一后，Task 1/Task 2 各为 626 张图和 626 份标注，所有副本内容一致。测试图包实际为 360 张而非文件名中的 361，文本包有 361 份，其中 `X51006619570.txt` 没有对应测试图；这些上游事实由 verifier fail-closed 锁定，后续 adapter 不能静默改变口径。目录中的 `data-*-batch-0000.zip` 是无关的 Claude 用户数据导出，包含用户/会话 JSON，明确不属于 SROIE，也不得进入清单、测试或 Git。

### FashionIQ 官方图片清单

FashionIQ 的标注仓库不包含图片字节；官方 README 指向另一个固定提交的
ASIN→Amazon 图片 URL 清单。当前本地冻结的清单提交是
`3bb39d7d9a024d92f1b0fd0929b38d73dff24a0a`，共 77,683 条记录。使用专用下载器：

```powershell
uv run python scripts/download_fashioniq_images.py --workers 6
```

图片按 `data/raw/fashioniq/images/<category>/<asin>.jpg` 保存。下载器将清单里的
HTTP 地址仅在传输时升级为 HTTPS，限制为清单中出现的两个 Amazon 域名，校验
图片可解码后才原子发布；官方提供的 13 张 `broken_links` 替代图会优先复用。
`data/raw/fashioniq/download-state/summary.json` 是跨会话进度快照，
`failures.jsonl` 保存本轮仍失败的对象。中断或重启后重复运行同一命令即可：
已验证图片会跳过，损坏文件和失败对象会重新获取。不要同时启动第二个
FashionIQ 下载器；独占锁会拒绝并发写入。

2026-07-24 两轮完整重试均留下逐条相同的 2,416 个失败身份，且这些身份均没有
有效图片。项目 owner 随后明确决定排除它们；本地
`exclusions.jsonl` 保存逐条 disposition，`exclusions-summary.json` 绑定两轮失败
文件摘要和剩余 formal gate。运行：

```powershell
uv run python scripts/verify_fashioniq_acquisition.py
```

必须得到 75,267 accepted + 2,416 excluded = 77,683，且两集合互斥、无
partial，才可在 RAW acquisition 层报告 `complete_with_exclusions`。排除不等于
下载成功，也不能绕过后续 license、source lock、selection review 和 catalog。

取得全部图片字节不代表该来源已进入正式实验。上游只笼统指向
Community Data License Agreement，尚需固定适用版本、用途范围和审批证据；
在完成许可审查、source lock、selection/disposition 与 catalog 前，状态仍不是
formal-ready。

## 空间与完成判定

`plan` 的 `known_download_bytes` 只统计清单中已经冻结大小的文件；`unknown_size_artifacts`、授权来源、解压副本、清洗产物、索引和临时工作空间不包含在内。`download` 会在开始时对已知剩余字节做磁盘预检，但无法替未知/授权来源承诺容量。

一个 profile 的 RAW 下载完成应同时满足：

1. `verify` 对全部 `artifacts` 返回 `[ OK ]`；
2. `status` 中没有 `running` 或 `failed`；
3. 所有 `BLOCKED` 来源已由审核后的 overrides 替换并完成；
4. 对 `command` 来源检查其 adapter 生成的候选快照或固定 revision 工件；
5. 后续另行完成许可证/PII/use review、source lock、selection/disposition 和正式 catalog。
