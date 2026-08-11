# Deterministic product-search tools implementation plan

> **Historical implementation plan; source semantics superseded on 2026-07-22.** The MEP-3M category-anchor behavior below remains a diagnostic compatibility path, not MVP Style gold. Formal Exact/Style evaluation must follow [`../../data-source-adjustment.md`](../../data-source-adjustment.md) and [`../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json`](../../../specs/data_sources/ecommerce-mvp-source-portfolio-v1.json): ABO supplies product identity, FashionIQ supplies relative-language style supervision, and MUGE/MEP-3M cannot supply MVP positives.

> **For AI implementers:** Required workflow: test-first development, with each new test observed failing before its production implementation is added.

**Goal:** Add deterministic image, text, and style-diversification product-search tools that can use the production index lazily and accept explicitly injected test dependencies.

**Architecture:** `ProductSearchService` owns an already validated `ProductIndex` plus an embedding backend. Its modality-specific methods validate inputs, produce one query vector, query the corresponding FAISS index for bounded candidates, then recompute every score from persisted NumPy vectors before sorting by `(-score, product_id)`. Module functions resolve one lazy singleton made by an explicit configurable factory; tests only instantiate the service with a small fake index/backend.

**Technology:** Python 3.12, NumPy, Pillow, Pydantic contracts, existing `ProductIndex`, `CachedEmbeddingBackend`, and pytest.

---

### Task 1: Define failure-first search-service tests

**Files:**
- Create: `tests/tools/test_product_search.py`
- Create later: `src/skillchain/tools/product_search.py`

- [ ] Add a fake backend recording separate image/text calls and producing controlled unit vectors.
- [ ] Add a tiny `ProductIndex` fixture with intentionally tied vectors, product IDs out of row order, MUGE/Mep3m sources, categories, and more than ten matching products.
- [ ] Add failing tests for image/text modal selection, deterministic tie breaking, fixed ten-result limit, JSON serializability/stability, and rejected blank/missing/invalid/oversize inputs.
- [ ] Add failing style-search tests for reliable Mep3m anchors, unknown-anchor rejection, category and anchor exclusion, diversity selected by precise MMR, and MMR tie breaking by product ID.
- [ ] Run `uv run pytest tests/tools/test_product_search.py -q` and confirm collection fails because the public module/service do not exist yet.

### Task 2: Implement the injected service and public functions

**Files:**
- Create: `src/skillchain/tools/product_search.py`
- Modify: `src/skillchain/tools/__init__.py`
- Test: `tests/tools/test_product_search.py`

- [ ] Implement `ProductSearchService(index, backend)` with image/text search methods and a style search method.
- [ ] Validate local image inputs (file, at most 10 MiB, Pillow `verify`) before calling the image backend; reject blank text before calling the text backend.
- [ ] Ask FAISS only for candidate indices, recompute candidate dot products from the row-aligned NumPy vectors, round all output scores to eight decimal places, and construct validated JSON-only `ProductHit`/`StyleHit` dictionaries in stable order.
- [ ] For style search choose the first highest-scoring Mep3m non-empty/non-`unknown` category anchor; limit same-category, non-anchor candidates to image top 100, then select with exact `0.5 * relevance - 0.5 * max_similarity` MMR and stable product-ID tie breaking.
- [ ] Implement an explicit default-service factory configuration/reset hook and a lazy cached singleton factory that loads `PRODUCT_INDEX_DIR`, configures `CachedEmbeddingBackend(DashScopeEmbeddingClient(), EmbeddingCache(...))`, and uses the index canary. Do not make remote calls at import time.
- [ ] Export the requested public functions and service through `skillchain.tools`.

### Task 3: Verify and commit

**Files:**
- Verify: `tests/tools/test_product_search.py`, `tests/tools`, full test suite, compilation

- [ ] Run the focused tests after implementation and confirm they pass without networking.
- [ ] Run `uv run pytest tests/tools -q`, then `uv run pytest -q`; record any pre-existing external failures separately.
- [ ] Run `uv run python -m compileall -q src`.
- [ ] Inspect `git diff --check`, `git diff --stat`, and `git status --short` to verify scope.
- [ ] Commit the implementation, tests, exports, and this plan with `feat: add deterministic product search tools`.
