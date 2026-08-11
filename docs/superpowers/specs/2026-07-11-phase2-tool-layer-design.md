# Phase 2 本地确定性工具层设计规格

> **历史规格，已于 2026-07-20 被取代。** 本文的“6 工具、无 OCR、历史数据已验收、每工具 3 条金标”均不得作为当前验收口径；canonical 设计与剩余阻断项以 `docs/plans/2026-07-09-skillchain-reproduction.md` 为准。

> **2026-07-22 数据来源补充：** 本文中的 MUGE Exact、MEP-3M Style 与 COCO Multi-Product 绑定只描述历史实现，不是当前 MVP gold 设计。当前来源角色以 [`../../data-source-adjustment.md`](../../data-source-adjustment.md) 和 [`../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`](../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json) 为准。

## 目标与范围

完成主计划 Phase 2 的六个工具：三个商品检索工具、两个知识库检索工具和一个目标检测工具，并通过统一 registry 暴露稳定的 JSON Schema。商品向量由 DashScope `qwen3-vl-embedding` 生成；FAISS 检索、MMR、BM25 和 YOLO 推理在本地执行。

本阶段使用 Phase 1 已验收的 41,409 条商品、51,436 条百科和 100,000 条菜谱，不修改原始/清洗数据。实现必须支持中断续跑、固定排序、重复调用字节一致、产物完整性校验和明确失败，且不覆盖当前工作区中用户对 `config.py`、`llm.py`、`.agents/`、`AGENTS.md` 的修改。

## 对主计划的修订

原计划写有“工具层全本地、运行时零外部 API”。采用 DashScope 后，准确口径改为：

- 商品库批量建索引时调用 DashScope 多模态 Embedding API；
- 新图片或新文本查询首次向量化时调用同一 API；
- 查询向量按模型、维度、模态和输入 SHA-256 持久缓存，重复查询不再调用 API；
- FAISS、MMR、BM25、YOLO 和 registry 调度始终本地运行；
- API 不可用且查询未命中缓存时明确失败，不用其他模型静默降级。

本地图片会以 Base64 Data URI 发送给 DashScope。MEP-3M 数据仍限学术研究，图片、向量缓存和索引均位于被 Git 忽略的 `data/`，不进入公开 Demo 或可再分发交付物。

## 文件边界

```text
src/skillchain/tools/
├── __init__.py             # 六工具公共导出
├── settings.py             # Phase 2 模型、维度、k、阈值和产物路径
├── embedding.py            # DashScope HTTP 客户端、缓存、漂移检测
├── product_index.py        # 商品索引构建、加载、manifest 与原子发布
├── product_search.py       # 三个商品检索工具
├── kb_lookup.py            # 两个 jieba + bm25s 检索器及构建逻辑
├── object_detect.py        # YOLO COCO 推理与 80 类中文映射
└── registry.py             # ToolSpec、JSON Schema、调度与规范化序列化
scripts/build_index.py      # products/kb/all/status/prewarm-gold CLI
tests/tools/                # 按组件拆分的离线单元测试
tests/test_tools.py         # 六工具真实金标验收，每工具三条
data/index/                 # 索引、缓存、manifest，全部 gitignore
runs/phase2/                # 金标结果与性能报告
```

工具常量放在 `tools/settings.py`，避免触碰当前有用户未提交改动的全局 `config.py`。它只依赖 `config.DATA_DIR` 和 `config.RUNS_DIR` 解析根路径。

## DashScope Embedding 契约

使用官方 HTTP 端点：

```text
POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/
     multimodal-embedding/multimodal-embedding
```

请求固定为：

- `model="qwen3-vl-embedding"`；
- `parameters.dimension=1024`；
- 不设置 `enable_fusion`，每个 content 返回一个独立向量；
- 图片每批最多 5 张，文本每批最多 20 条；
- 图片使用最终 JPEG/PNG 文件的 Base64 Data URI，单张必须不超过官方 10 MB 上限；
- 文本保持原字符串语义，仅拒绝空白输入，不做会改变检索语义的 Unicode 重写。

客户端使用现有 `requests`，不复用 `llm.py`：Embedding API 不是 OpenAI 兼容接口，且 `llm.py` 当前包含用户未提交改动。客户端不得记录 API key 或完整 Base64，请求日志只保留 request-id、批次大小、模态、耗时和重试次数。

仅对连接错误、超时、HTTP 429 和 5xx 重试；尊重 `Retry-After`，否则使用有上限的指数退避。4xx 参数/鉴权错误立即失败。每个响应必须验证：

1. request-id 存在；
2. embedding 数量与输入数量相同；
3. 响应 index 可无歧义恢复输入顺序；
4. 每个向量恰为 1024 维且全部为有限数；
5. L2 范数非零，随后转为 `float32` 并归一化。

## 缓存、续跑与模型漂移

`data/index/embedding_cache.sqlite3` 使用 WAL 模式，主键包含：

```text
model | dimension | modality | input_sha256
```

值保存归一化 `float32` 向量、输入字节数和响应 request-id。每个成功响应在一个事务内提交，进程中止后只重试尚未提交的输入。缓存命中仍复验维度、有限值和单位范数；损坏记录被拒绝，不静默重算或混用。

`qwen3-vl-embedding` 是服务端别名而非日期快照。首次构建时额外生成固定 canary 文本的向量并把其 SHA-256 与完整向量写入 manifest。每个进程第一次需要产生未缓存查询向量时绕过缓存重新请求 canary；只有维度一致且余弦相似度不低于 `0.999999` 才允许继续。漂移时拒绝向旧索引写入新缓存，并提示全量重建。缓存命中查询不需要联网检查。

## 商品索引

输入为 `data/clean/products.parquet`。构建器按 Parquet 行顺序为每个商品生成：

- 图片独立向量；
- 标题独立向量。

最终产物原子发布到 `data/index/products/`：

```text
image.faiss
text.faiss
image_vectors.npy
text_vectors.npy
products.parquet
manifest.json
```

两个 FAISS 索引均为 `IndexFlatIP`，输入已 L2 归一化，因此分数是余弦相似度。`.npy` 保留同序向量供 MMR、覆盖审计和确定性重建使用；`products.parquet` 是查询所需元数据快照，不依赖后续 Phase 1 文件变动。

manifest 固定记录模型、维度、归一化方式、源 Parquet SHA-256、商品数、向量数、FAISS 类型、canary、构建批次参数和各产物 SHA-256。不记录当前时间或机器路径。工具加载时逐项复验 manifest、文件哈希、FAISS `ntotal/d`、矩阵 shape、商品 ID 唯一性和行对齐；任何不一致均拒绝服务。

构建 CLI 支持 `--limit` 冒烟，但 manifest 会标记 `complete=false`，正式工具默认拒绝加载不完整索引。全量构建完成后打印图片/文本各自的缓存命中数、API 调用数、吞吐和总耗时。

## 三个商品工具

固定返回 10 条 `ProductHit`：

```json
{
  "rank": 1,
  "score": 0.12345678,
  "product": {"product_id": "...", "title": "...", "category_l1": "..."}
}
```

分数统一四舍五入到 8 位小数。最终排序键为 `(-score, product_id)`，消除 FAISS 同分顺序差异。

### `image_product_search(image)`

验证本地图片存在、可解码且不超过 10 MB，生成/读取图片查询向量，在 image 索引中返回前 10 条。

### `text_product_search(query)`

拒绝空白文本，生成/读取文本查询向量，在 text 索引中返回前 10 条。标题与查询使用同一模型、维度和独立向量模式。

### `style_similar_search(image)`

1. 对查询图生成图片向量；
2. 只在 `category_l1 != "unknown"` 的 MEP-3M 商品中找最近锚点，确定可靠一级类目；
3. 在该类目的归一化 image vectors 上计算精确余弦相关度，排除锚点商品后取前 100 个候选；
4. 以 `lambda=0.5` 执行 MMR：`0.5 * relevance - 0.5 * max_similarity_to_selected`；
5. 每轮同分按 `product_id`，返回 10 条。

这样不会把 MUGE 的 `unknown` 当成真实类目，也不把 MEP-3M 标签套到 MUGE 商品上。返回中额外包含 `anchor_category_l1` 和 MMR 分数。

## 两个知识库工具

`encyclopedia_lookup(entity)` 和 `recipe_lookup(dish)` 各固定返回 5 条。构建器流式读取对应 JSONL，经 `jieba.lcut` 精确分词后交给 `bm25s`；文档顺序保持源文件顺序，重复 `entry_id` 或 schema 无效时失败。

索引分别发布到 `data/index/kb/encyclopedia/` 和 `data/index/kb/recipes/`，manifest 记录源 SHA-256、文档数、jieba/bm25s 版本和产物哈希。查询排序键为 `(-bm25_score, entry_id)`，分数保留 8 位；返回 `entry_id/title/text/kind/origin/synth_provider/synth_model`，不丢失来源字段。

## 目标检测工具

`object_detect(image)` 使用 `ultralytics` 的固定 `yolo11n.pt` COCO 权重，CPU 推理参数固定为：

```text
imgsz=640, conf=0.25, iou=0.7, max_det=100, device="cpu", verbose=false
```

权重下载后固定 SHA-256，加载前复验。COCO 80 个英文标签与中文标签在模块常量中逐项对应，测试要求键集合恰为 `0..79`。

每个检测结果为：

```json
{"label":"person","label_zh":"人","bbox":[1.0,2.0,3.0,4.0],"confidence":0.987654}
```

bbox 和 confidence 固定舍入到 6 位，排序键为 `(-confidence, class_id, x1, y1, x2, y2)`。无检测返回空列表。模块延迟导入 ultralytics，安装或权重缺失时给出明确恢复命令，单元测试不触发联网下载。

## Registry 与序列化

registry 只注册以下六个名称：

```text
image_product_search
text_product_search
style_similar_search
encyclopedia_lookup
recipe_lookup
object_detect
```

每个 `ToolSpec` 包含名称、中文说明、严格 input JSON Schema 和 handler。Schema 禁止额外属性；固定 k 不暴露给调用方。`call_tool(name, arguments)` 负责名称校验、Pydantic 输入校验和 JSON 值规范化。canonical serializer 使用 UTF-8、`ensure_ascii=false`、`sort_keys=true`、固定分隔符且禁止 NaN/Infinity，用于“重复调用逐字节一致”的验收。

## CLI 与执行顺序

```powershell
uv run python scripts/build_index.py products --limit 50   # API 冒烟与恢复验证
uv run python scripts/build_index.py products              # 41,409 商品全量
uv run python scripts/build_index.py kb                     # 两个 BM25 索引
uv run python scripts/build_index.py prewarm-gold           # 缓存 18 条金标查询
uv run python scripts/build_index.py status                 # 哈希/数量/完整性报告
```

`products` 和 `kb` 均在 staging 完整写入并自验后原子替换最终目录。冒烟索引与正式索引使用不同目录，不能覆盖已完成全量产物。

## 测试与验收

所有生产行为遵循红—绿—重构。离线单元测试使用协议化 fake embedding backend 和 fake detector，覆盖：

- DashScope 批量边界、响应乱序恢复、429/5xx 重试、4xx 失败、维度/NaN/零范数拒绝；
- SQLite 缓存命中、中断续跑、损坏拒绝和 canary 漂移；
- FAISS 行对齐、manifest 篡改、同分排序和不完整索引拒绝；
- 三个商品工具固定 10 条、可靠类目锚定和 `lambda=0.5` MMR；
- jieba + bm25s 两库隔离、固定 5 条、来源字段保留和同分排序；
- COCO 80 类映射、检测结果规范化、空结果和权重哈希失败；
- registry 恰有六项、Schema 严格、未知工具/额外参数拒绝和 canonical JSON 字节一致。

真实金标测试每工具三条，共 18 条。商品与检测金标从 Phase 1 已固定图片中选取，KB 金标使用稳定实体/菜名；金标输入的查询 embedding 先由 `prewarm-gold` 写入缓存，pytest 验收阶段不产生新 API 费用。结果同时写入 `runs/phase2/gold_results.json`，包括期望、实际、通过条件、索引 manifest SHA-256 和工具输出。

最终验收命令：

```powershell
uv run pytest tests/tools -v
uv run pytest tests/test_tools.py -v
uv run python scripts/build_index.py status
uv run ruff check .
git diff --check
```

验收还要求：同一调用连续两次 canonical JSON 完全相同；全量商品索引 `ntotal=41,409`；百科/菜谱索引分别为 51,436/100,000；无 API key、Base64、原始 MEP-3M 图片或索引进入 Git。

## 非目标

- 不使用 `qwen3-vl-plus` 聊天模型代替 embedding。
- 不启用融合向量，不把图片和标题压成单一商品向量。
- 不引入远端向量数据库、DashScope rerank 或 Elasticsearch。
- 不训练/微调 embedding 或 YOLO。
- 不为现有 MUGE 补类目；style 工具只使用 MEP-3M 自带的可靠类目。
- 不修改 Phase 3 之后的 Assistant、Skill 或优化循环。
