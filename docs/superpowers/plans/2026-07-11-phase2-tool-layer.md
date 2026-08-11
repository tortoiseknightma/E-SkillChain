# Phase 2 本地确定性工具层实现计划

> **历史文档，已于 2026-07-20 被取代。** 不再按本文执行，也不使用本文提到的外部 Skill。当前 canonical 范围见 `docs/plans/2026-07-09-skillchain-reproduction.md`：工具数已从 6 增至 7（加入 OCR），3 条挑选样本仅算 smoke，正式数据必须经过 provenance、group split、model/safety manifest、create-only run 和独立 benchmark gate。

> **2026-07-22 数据来源补充：** 本文中的 MUGE Exact、MEP-3M Style 与 COCO Multi-Product 样例只保留为历史 diagnostic fixture，不再定义 MVP gold。当前 formal 来源角色以 [`../../data-source-adjustment.md`](../../data-source-adjustment.md) 和 [`../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`](../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json) 为准：ABO/RPC/FashionIQ 分别承担 Exact/Multi-Product/Style 主监督，COCO 只作 detector challenge，MEP-3M 延后到 core。

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用 DashScope `qwen3-vl-embedding`、FAISS、jieba/bm25s 和本地 YOLO 构建六个可恢复、可审计、返回确定性 JSON 的 Phase 2 工具，并完成真实索引与每工具三条金标验收。

**架构：** DashScope 客户端与 SQLite cache 负责独立图文向量和断点续跑；构建器把 Phase 1 商品快照、归一化向量、FAISS 索引与 manifest 原子发布。商品/KB/检测工具通过严格 registry 暴露，首次新商品查询允许调用 DashScope，重复查询由内容哈希缓存保证字节一致。

**技术栈：** Python 3.12、requests、SQLite、NumPy、PyArrow、FAISS、jieba、bm25s、Pydantic、Ultralytics YOLO、pytest、uv。

**批准规格：** `docs/superpowers/specs/2026-07-11-phase2-tool-layer-design.md`

---

## 文件结构与职责

### 新建

- `src/skillchain/tools/__init__.py`：公共工具导出。
- `src/skillchain/tools/settings.py`：Phase 2 固定模型、维度、k、阈值和路径。
- `src/skillchain/tools/contracts.py`：`ProductHit`、`Detection`、JSON 值类型和公共校验。
- `src/skillchain/tools/embedding.py`：DashScope HTTP、SQLite cache、canary 漂移检查。
- `src/skillchain/tools/product_index.py`：商品双索引构建、manifest、自验与加载。
- `src/skillchain/tools/product_search.py`：图片/文本检索和同类 MMR。
- `src/skillchain/tools/kb_lookup.py`：百科/菜谱 BM25 构建与查询。
- `src/skillchain/tools/object_detect.py`：YOLO 加载、COCO 中文映射和结果规范化。
- `src/skillchain/tools/registry.py`：六工具 Schema、调度、canonical JSON。
- `scripts/build_index.py`：`products/kb/prewarm-gold/status/all` CLI。
- `tests/tools/conftest.py`：小型商品、KB、fake embedding/detector fixtures。
- `tests/tools/test_embedding.py`：API/cache/canary 测试。
- `tests/tools/test_product_index.py`：索引产物与完整性测试。
- `tests/tools/test_product_search.py`：三商品工具与 MMR 测试。
- `tests/tools/test_kb_lookup.py`：BM25 构建/查询测试。
- `tests/tools/test_object_detect.py`：COCO 映射/检测规范化测试。
- `tests/tools/test_registry.py`：六工具与序列化测试。
- `tests/test_tools.py`：真实 18 条金标验收。

### 修改

- `pyproject.toml`、`uv.lock`：加入锁定的 Ultralytics/PyTorch CPU 依赖。
- `docs/plans/2026-07-09-skillchain-reproduction.md`：修订首次查询 API 口径并记录 Phase 2 实绩。
- `docs/plans/2026-07-09-skillchain-reproduction.html`：由现有 renderer 同步生成。

### 本地生成且不入 Git

- `data/index/embedding_cache.sqlite3`
- `data/index/products/{image,text}.faiss`
- `data/index/products/{image,text}_vectors.npy`
- `data/index/products/{products.parquet,manifest.json}`
- `data/index/kb/{encyclopedia,recipes}/`
- `data/models/yolo11n.pt`
- `runs/phase2/gold_results.json`

---

## 任务 1：工具契约与固定配置

**文件：**
- 创建：`src/skillchain/tools/__init__.py`
- 创建：`src/skillchain/tools/settings.py`
- 创建：`src/skillchain/tools/contracts.py`
- 创建：`tests/tools/__init__.py`
- 创建：`tests/tools/conftest.py`
- 创建：`tests/tools/test_contracts.py`

- [ ] **步骤 1：编写失败的配置与序列化模型测试**

```python
def test_phase2_settings_are_pinned():
    assert EMBEDDING_MODEL == "qwen3-vl-embedding"
    assert EMBEDDING_DIMENSION == 1024
    assert IMAGE_BATCH_SIZE == 5
    assert TEXT_BATCH_SIZE == 20
    assert PRODUCT_K == 10
    assert KB_K == 5
    assert MMR_LAMBDA == 0.5


def test_product_hit_rejects_nan_score():
    with pytest.raises(ValidationError):
        ProductHit(rank=1, score=float("nan"), product=_product())
```

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_contracts.py -v`

预期：collection 失败，`skillchain.tools` 尚不存在。

- [ ] **步骤 3：实现最小配置与契约**

`settings.py` 只从 `skillchain.config` 读取 `DATA_DIR/RUNS_DIR`；其余 Phase 2 常量在本模块集中定义。`contracts.py` 使用 Pydantic finite-number 校验，并提供：

```python
class ProductHit(BaseModel):
    rank: int
    score: float
    product: Product


class StyleHit(ProductHit):
    mmr_score: float
    anchor_category_l1: str


class Detection(BaseModel):
    label: str
    label_zh: str
    bbox: tuple[float, float, float, float]
    confidence: float
```

- [ ] **步骤 4：运行绿灯与现有 schema 回归**

运行：`uv run pytest tests/tools/test_contracts.py tests/test_schemas.py -v`

预期：全部 PASS。

- [ ] **步骤 5：提交任务 1**

```powershell
git add src/skillchain/tools tests/tools
git commit -m "feat: define Phase 2 tool contracts"
```

---

## 任务 2：DashScope Embedding 与可恢复缓存

**文件：**
- 创建：`src/skillchain/tools/embedding.py`
- 创建：`tests/tools/test_embedding.py`
- 修改：`tests/tools/conftest.py`

- [ ] **步骤 1：为请求与响应验证编写红灯测试**

测试用 fake session 返回真实响应形状：

```python
{
    "output": {"embeddings": [
        {"index": 1, "embedding": [0.0, 2.0, ...], "type": "vl"},
        {"index": 0, "embedding": [3.0, 0.0, ...], "type": "vl"},
    ]},
    "usage": {"input_tokens": 3, "image_tokens": 0, "total_tokens": 3},
    "request_id": "req-1",
}
```

覆盖：按 index 恢复顺序、1024 维、NaN/Infinity/零范数、数量缺失、重复 index、4xx 立即失败、429/5xx 重试、`Retry-After` 和 payload 中没有 `enable_fusion`。

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_embedding.py -v`

预期：导入或行为断言 FAIL。

- [ ] **步骤 3：实现 `DashScopeEmbeddingClient`**

核心接口固定为：

```python
class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...


class DashScopeEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        ...
```

HTTP 请求固定 endpoint/model/dimension。图片在读入前检查存在、Pillow 解码、大小 ≤10 MB，再按真实格式生成 Data URI。日志不得包含 Authorization 或 Base64。

- [ ] **步骤 4：为 SQLite cache 写红灯测试**

覆盖：同输入只调用 backend 一次、每成功批次事务提交、中断后只补缺项、主键包含 model/dimension/modality/hash、损坏 BLOB 拒绝、WAL、单位范数复验。

- [ ] **步骤 5：实现 `EmbeddingCache` 与 `CachedEmbeddingBackend`**

SQLite schema 精确为：

```sql
CREATE TABLE embeddings (
  model TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  modality TEXT NOT NULL CHECK(modality IN ('text','image')),
  input_sha256 TEXT NOT NULL,
  input_bytes INTEGER NOT NULL,
  vector BLOB NOT NULL,
  request_id TEXT NOT NULL,
  PRIMARY KEY(model, dimension, modality, input_sha256)
);
```

- [ ] **步骤 6：为 canary 漂移编写红灯并实现**

manifest canary 与绕缓存的新 canary 余弦相似度 `<0.999999` 时抛出 `ModelDriftError`，不得写查询缓存；缓存全命中时不得联网。

- [ ] **步骤 7：运行任务 2 绿灯与真实能力探针**

```powershell
uv run pytest tests/tools/test_embedding.py -v
uv run python -c "from pathlib import Path; from skillchain.tools.embedding import DashScopeEmbeddingClient; c=DashScopeEmbeddingClient(); print(c.embed_texts(['连衣裙']).shape); print(c.embed_images([Path('data/clean/preview_mep3m.jpg')]).shape)"
```

预期：测试全部 PASS；真实输出两次均为 `(1, 1024)`，不打印向量或 key。

- [ ] **步骤 8：提交任务 2**

```powershell
git add src/skillchain/tools/embedding.py tests/tools/test_embedding.py tests/tools/conftest.py
git commit -m "feat: add recoverable DashScope embeddings"
```

---

## 任务 3：商品双索引构建与原子产物

**文件：**
- 创建：`src/skillchain/tools/product_index.py`
- 创建：`tests/tools/test_product_index.py`
- 创建：`scripts/build_index.py`

- [ ] **步骤 1：编写索引构建红灯测试**

用 4 条商品和固定 4 维 fake vectors，要求：

```python
report = build_product_index(products_path, output_dir, backend, limit=None)
assert report.products == 4
assert faiss.read_index(str(output_dir / "image.faiss")).ntotal == 4
assert faiss.read_index(str(output_dir / "text.faiss")).ntotal == 4
assert json.loads((output_dir / "manifest.json").read_text())["complete"] is True
```

另覆盖 `--limit 2` 写入独立 smoke 目录且 `complete=false`、重复 product ID、缺图、backend 行数错误和构建异常不替换旧索引。

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_product_index.py -v`

预期：`build_product_index` 不存在或断言 FAIL。

- [ ] **步骤 3：实现批量构建与产物自验**

按 Parquet 行顺序分批调用 cache backend，写：

```text
image.faiss / text.faiss
image_vectors.npy / text_vectors.npy
products.parquet / manifest.json
```

manifest 中每个产物记录 `{path, bytes, sha256}`；源 Parquet 记录 SHA-256 和行数。写入 staging 后重新加载所有文件，验证 shape、dtype、finite、单位范数、FAISS d/ntotal 和 row alignment，再使用现有 `publish_staged_directory` 原子发布。

- [ ] **步骤 4：实现 `ProductIndex.load()` 的篡改拒绝测试与代码**

逐一篡改 manifest、FAISS、`.npy` 和 metadata，加载必须给出包含文件名的错误；不完整 smoke manifest 默认拒绝，显式 `allow_incomplete=True` 才允许单元测试加载。

- [ ] **步骤 5：实现 CLI products/status 骨架**

```powershell
uv run python scripts/build_index.py products --limit 4
uv run python scripts/build_index.py status --allow-incomplete
```

CLI 返回码、stdout JSON keys 和错误 stderr 写测试；执行期间输出批次、缓存命中、API 调用、items/s 和 elapsed seconds。

- [ ] **步骤 6：运行绿灯**

运行：`uv run pytest tests/tools/test_product_index.py -v`

预期：全部 PASS。

- [ ] **步骤 7：提交任务 3**

```powershell
git add src/skillchain/tools/product_index.py tests/tools/test_product_index.py scripts/build_index.py
git commit -m "feat: build auditable product indices"
```

---

## 任务 4：三个商品检索工具与 MMR

**文件：**
- 创建：`src/skillchain/tools/product_search.py`
- 创建：`tests/tools/test_product_search.py`
- 修改：`src/skillchain/tools/__init__.py`

- [ ] **步骤 1：编写 image/text 固定排序红灯测试**

构造同分向量并断言返回固定 10 条或数据集不足时全部返回，排序为 `(-score, product_id)`，score 为 8 位，重复调用 `model_dump_json()` 一致。

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_product_search.py -v`

预期：商品工具尚不存在。

- [ ] **步骤 3：实现 image/text search**

```python
def image_product_search(image: str | Path) -> list[dict]: ...
def text_product_search(query: str) -> list[dict]: ...
```

生产默认使用 lazy singleton `ProductIndex` 与 `CachedEmbeddingBackend`；测试通过显式 service factory 注入 fake backend，禁止仅测试全局 monkeypatch。

- [ ] **步骤 4：编写 MMR 红灯测试**

fixture 同时包含 MUGE `unknown` 与两个 MEP 类目。断言：

- 锚点只能来自 MEP known category；
- 结果全部与 anchor `category_l1` 相同；
- 结果不包含 anchor product_id；
- `lambda=0.5`；
- 高度相似候选被多样性候选替代；
- 同分按 product_id；
- 返回 10 条并包含 `anchor_category_l1/mmr_score`。

- [ ] **步骤 5：实现 known-category anchor 与精确 MMR**

候选相关度直接使用 `image_vectors.npy` 点积；先取同类 top 100，再按以下公式逐项选择：

```python
mmr = 0.5 * float(query @ candidate) - 0.5 * max_selected_similarity
```

- [ ] **步骤 6：运行绿灯与产品索引回归**

运行：`uv run pytest tests/tools/test_product_search.py tests/tools/test_product_index.py -v`

预期：全部 PASS。

- [ ] **步骤 7：提交任务 4**

```powershell
git add src/skillchain/tools/product_search.py src/skillchain/tools/__init__.py tests/tools/test_product_search.py
git commit -m "feat: add deterministic product search tools"
```

---

## 任务 5：百科与菜谱 BM25 工具

**文件：**
- 创建：`src/skillchain/tools/kb_lookup.py`
- 创建：`tests/tools/test_kb_lookup.py`
- 修改：`scripts/build_index.py`
- 修改：`src/skillchain/tools/__init__.py`

- [ ] **步骤 1：编写两库隔离和 schema 红灯测试**

fixture 百科含“熊猫/竹子”，菜谱含“宫保鸡丁/花生”。断言查询不跨库、固定 5 条或全量、保留 `origin/synth_provider/synth_model`、拒绝重复 entry_id 和无效 provenance。

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_kb_lookup.py -v`

预期：`build_kb_indices`/lookup 尚不存在。

- [ ] **步骤 3：实现流式加载、jieba 分词和 bm25s 持久化**

`build_kb_index(kind, source, output)` 先 Pydantic 验证每行，再对 `title + " " + text` 使用 `jieba.lcut`。manifest 固定记录源 SHA、文档数、依赖版本和产物 SHA；staging 自验后原子发布。

- [ ] **步骤 4：实现确定性查询**

```python
def encyclopedia_lookup(entity: str) -> list[dict]: ...
def recipe_lookup(dish: str) -> list[dict]: ...
```

拒绝空白输入；bm25s 候选扩大到 `min(total, KB_K * 4)`，最终按 `(-score, entry_id)` 取 5 条，score 8 位。

- [ ] **步骤 5：增加 CLI kb/status 并运行绿灯**

运行：`uv run pytest tests/tools/test_kb_lookup.py -v`

预期：全部 PASS。

- [ ] **步骤 6：提交任务 5**

```powershell
git add src/skillchain/tools/kb_lookup.py src/skillchain/tools/__init__.py tests/tools/test_kb_lookup.py scripts/build_index.py
git commit -m "feat: add deterministic Chinese KB lookup"
```

---

## 任务 6：本地 YOLO 目标检测

**文件：**
- 修改：`pyproject.toml`
- 修改：`uv.lock`
- 创建：`src/skillchain/tools/object_detect.py`
- 创建：`tests/tools/test_object_detect.py`
- 修改：`src/skillchain/tools/__init__.py`

- [ ] **步骤 1：编写 COCO 映射与检测规范化红灯测试**

```python
assert set(COCO_LABELS_ZH) == set(range(80))
assert COCO_LABELS_ZH[0] == "人"
```

fake Results 返回乱序 boxes，断言 confidence/bbox 6 位，排序 `(-confidence,class_id,x1,y1,x2,y2)`，无框为空列表，未知 class_id 失败。

- [ ] **步骤 2：运行红灯测试**

运行：`uv run pytest tests/tools/test_object_detect.py -v`

预期：模块尚不存在。

- [ ] **步骤 3：添加 Ultralytics 锁定依赖**

运行：`uv add "ultralytics>=8.3,<9"`

检查：`uv run python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"`

- [ ] **步骤 4：实现 lazy detector 和权重校验**

`object_detect(image)` 固定 `imgsz=640/conf=0.25/iou=0.7/max_det=100/device="cpu"/verbose=False`。权重路径、文件大小和 SHA-256 写入 sidecar manifest；若缺失，明确提示运行 `build_index.py detector`，测试中不自动联网。

- [ ] **步骤 5：增加 `detector` CLI 与本地权重探针**

CLI 下载 `yolo11n.pt` 到 staging，加载模型完成一次 `preview_coco.jpg` 推理，固定实际 SHA-256 后发布权重与 manifest。

- [ ] **步骤 6：运行绿灯**

运行：`uv run pytest tests/tools/test_object_detect.py -v`

预期：全部 PASS。

- [ ] **步骤 7：提交任务 6**

```powershell
git add pyproject.toml uv.lock src/skillchain/tools/object_detect.py src/skillchain/tools/__init__.py tests/tools/test_object_detect.py scripts/build_index.py
git commit -m "feat: add local COCO object detection"
```

---

## 任务 7：六工具 Registry 与 canonical JSON

**文件：**
- 创建：`src/skillchain/tools/registry.py`
- 创建：`tests/tools/test_registry.py`
- 修改：`src/skillchain/tools/__init__.py`

- [ ] **步骤 1：编写 registry 红灯测试**

断言名称集合精确等于六项；每项 input schema 含 `additionalProperties=false`；固定 k 不在 schema；未知工具、缺字段、额外字段、错误类型均拒绝。

- [ ] **步骤 2：编写 canonical JSON 红灯测试**

```python
first = call_tool_json("text_product_search", {"query": "连衣裙"})
second = call_tool_json("text_product_search", {"query": "连衣裙"})
assert first == second
assert json.loads(first)
```

另断言 `ensure_ascii=False`、keys 排序、紧凑 separators、NaN/Infinity 失败。

- [ ] **步骤 3：运行红灯测试**

运行：`uv run pytest tests/tools/test_registry.py -v`

预期：registry 尚不存在。

- [ ] **步骤 4：实现严格 ToolSpec 与调度**

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., JSONValue]
```

`call_tool` 只接受 registry 名称和 dict，使用 Pydantic `extra="forbid"` 验证，再递归拒绝非 JSON 值；`call_tool_json` 使用 `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`。

- [ ] **步骤 5：运行六工具单元测试**

运行：`uv run pytest tests/tools -v`

预期：全部 PASS。

- [ ] **步骤 6：提交任务 7**

```powershell
git add src/skillchain/tools/registry.py src/skillchain/tools/__init__.py tests/tools/test_registry.py
git commit -m "feat: register deterministic local tools"
```

---

## 任务 8：真实索引、18 条金标与 Phase 2 验收

**文件：**
- 创建：`tests/test_tools.py`
- 修改：`scripts/build_index.py`
- 修改：`docs/plans/2026-07-09-skillchain-reproduction.md`
- 生成：`docs/plans/2026-07-09-skillchain-reproduction.html`
- 本地生成：`data/index/*`、`runs/phase2/gold_results.json`

- [ ] **步骤 1：编写 18 条真实金标测试并确认缺索引红灯**

每工具三条，使用 Phase 1 固定输入：

- image search：`exact_match/muge-1000250.jpg`、`muge-1000288.jpg`、`muge-1000338.jpg`，对应 product_id 必须进入 top10；
- text search：`连衣裙` top10 至少 6 条标题含“裙”；`平板电脑` top10 至少 6 条标题或二级类目含“平板电脑”；`进口休闲食品` top10 至少 6 条二级类目为“进口休闲食品”；
- style search：`mep3m-367-1.jpg`、`mep3m-414-1.jpg`、`mep3m-431-103.jpg`，anchor 分别为“运动/户外”“母婴/玩具/童装”“家居/家具/家装/家纺/厨具”，10 条全部同 anchor 类目、product_id 唯一且不含 anchor；
- encyclopedia：`大熊猫`、`竹子`、`花岗岩`，top5 至少一条标题或正文包含原查询词；
- recipe：`宫保鸡丁`、`红烧肉`、`蛋炒饭`，top5 至少一条标题包含原查询词；
- object detect：`coco-139.jpg` 至少检出“电视/椅子”之一，`coco-1503.jpg` 至少检出“笔记本电脑/键盘/鼠标”之一，`coco-2157.jpg` 至少检出“蛋糕/酒杯”之一。

运行：`uv run pytest tests/test_tools.py -v`

预期：因正式索引/权重尚未构建而 FAIL，错误给出对应 build 命令。

- [ ] **步骤 2：运行小批产品 API 冒烟并验证恢复**

```powershell
uv run python scripts/build_index.py products --limit 50
uv run python scripts/build_index.py products --limit 50
uv run python scripts/build_index.py status --allow-incomplete
```

第二次 API 新调用数必须为 0；smoke `ntotal=50` 且不覆盖正式目录。

- [ ] **步骤 3：构建两个真实 KB 索引与 YOLO 权重**

```powershell
uv run python scripts/build_index.py kb
uv run python scripts/build_index.py detector
```

预期：百科 51,436、菜谱 100,000；YOLO preview 探针成功，所有 manifest 哈希通过。

- [ ] **步骤 4：构建 41,409 商品全量索引**

运行：`uv run python scripts/build_index.py products`

允许长时间运行但必须每批 checkpoint；任何中断后重跑同一命令继续。完成条件：

```text
products=41409
image_vectors=(41409,1024)
text_vectors=(41409,1024)
image.faiss ntotal=41409,d=1024
text.faiss ntotal=41409,d=1024
complete=true
```

- [ ] **步骤 5：预热金标查询缓存并运行金标**

```powershell
uv run python scripts/build_index.py prewarm-gold
uv run python scripts/build_index.py all
uv run pytest tests/test_tools.py -v
```

`all` 按 `products → kb → detector → prewarm-gold → status` 调度；已完成且哈希匹配的阶段只验证不重建。

连续运行第二次 pytest，确认 API 新调用为 0 且 canonical outputs 逐字节一致。把期望、实际、manifest SHA 和通过条件原子写到 `runs/phase2/gold_results.json`。

- [ ] **步骤 6：运行完整状态和测试验证**

```powershell
uv run python scripts/build_index.py status
uv run pytest -q
uv run ruff check .
uv run ruff format --check src/skillchain/tools tests/tools tests/test_tools.py scripts/build_index.py
git diff --check
```

预期：所有命令 exit 0；状态报告的商品/KB/权重数量与哈希全部匹配。

- [ ] **步骤 7：更新主计划与 HTML**

把 Phase 2 七个 checkbox 标为完成，写入实际模型、API 调用/缓存命中、吞吐、索引大小、YOLO 权重 SHA、18 条金标结果和“首次查询 API、重复查询缓存”口径。然后运行：

```powershell
uv run python scripts/render_plan.py
uv run python scripts/render_plan.py --check
uv run pytest tests/test_render_plan.py -v
```

- [ ] **步骤 8：提交任务 8**

只提交代码、测试、计划和 HTML；确认 `data/index`、模型权重、cache、MEP 图片和 API key 未暂存。

```powershell
git status --short
git add tests/test_tools.py scripts/build_index.py docs/plans/2026-07-09-skillchain-reproduction.md docs/plans/2026-07-09-skillchain-reproduction.html
git diff --cached --check
git commit -m "feat: complete Phase 2 tool layer and indices"
```

---

## 最终审查清单

- [ ] 规格审查：逐项对照批准规格，无遗漏或额外远端依赖。
- [ ] 代码质量审查：安全、缓存事务、索引行对齐、资源释放和错误语义无 Important/Critical。
- [ ] `DASHSCOPE_API_KEY`、Authorization、Base64 和原始数据没有进入日志、测试快照或 Git。
- [ ] 六工具名称与 Schema 恰好匹配 Phase 2；固定 k 不对调用方开放。
- [ ] 重复调用 canonical JSON 逐字节一致。
- [ ] 全量索引/KB/YOLO/金标状态由新鲜命令输出证明。
- [ ] 当前主工作区用户未提交的 `config.py`、`llm.py`、`.agents/`、`AGENTS.md` 未被修改或提交。
