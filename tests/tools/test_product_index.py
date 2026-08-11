from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    GalleryAssetReference,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.gallery_eligibility import build_gallery_eligibility_manifest
from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.store import canonical_json_bytes, canonical_jsonl_bytes
from skillchain.tools.embedding import CachedEmbeddingBackend, EmbeddingCache
from skillchain.tools.product_index import (
    CANARY_TEXT,
    ProductIndex,
    build_product_index,
)
from skillchain.tools.settings import (
    EMBEDDING_DIMENSION,
    IMAGE_BATCH_SIZE,
    TEXT_BATCH_SIZE,
)


def _unit_vectors(size: int) -> np.ndarray:
    vectors = np.zeros((size, EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(size):
        vectors[index, index] = 1.0
    return vectors


class FakeBackend:
    model = "test-embedding-model"
    execution_location = "local"

    def __init__(self) -> None:
        self.image_calls: list[list[Path]] = []
        self.text_calls: list[list[str]] = []

    def embed_images(self, paths):
        self.image_calls.append(list(paths))
        return _unit_vectors(len(paths))

    def embed_texts(self, texts):
        self.text_calls.append(list(texts))
        return _unit_vectors(len(texts))


def _write_products(tmp_path: Path, count: int = 3) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(count):
        image = images / f"product-{index}.png"
        pixels = np.random.default_rng(index).integers(
            0, 256, size=(32, 32, 3), dtype=np.uint8
        )
        Image.fromarray(pixels, mode="RGB").save(image)
        rows.append(
            {
                "product_id": f"product-{index}",
                "title": f"title {index}",
                "category_l1": "fashion",
                "category_l2": None,
                "category_l3": None,
                "ocr_text": None,
                "image_path": image.relative_to(tmp_path).as_posix(),
                "source": "muge",
            }
        )
    products = tmp_path / "products.parquet"
    pq.write_table(pa.Table.from_pylist(rows), products)
    return products


def _verified_inputs(
    products_path: Path, *, cloud_upload_allowed: bool | None = True
) -> dict[str, Any]:
    root = products_path.parent
    rows = pq.read_table(products_path).to_pylist()
    query_directory = root / "query-images"
    query_directory.mkdir(exist_ok=True)
    query_image = query_directory / "query.png"
    query_pixels = np.random.default_rng(100_000).integers(
        0, 256, size=(32, 32, 3), dtype=np.uint8
    )
    Image.fromarray(query_pixels, mode="RGB").save(query_image)

    drafts = [
        DatasetAssetDraft(
            source_dataset=row["source"],
            source_revision="product-index-test-v1",
            source_record_id=row["product_id"],
            local_path=row["image_path"],
            product_id=row["product_id"],
            license_id="test-only",
            cloud_upload_allowed=cloud_upload_allowed,
        )
        for row in rows
    ]
    drafts.append(
        DatasetAssetDraft(
            source_dataset="muge",
            source_revision="product-index-test-v1",
            source_record_id="query-only",
            local_path=query_image.relative_to(root).as_posix(),
            product_id="query-only",
            license_id="test-only",
            cloud_upload_allowed=cloud_upload_allowed,
        )
    )
    assets = tuple(inventory_dataset_asset(draft, root) for draft in drafts)
    catalog_path = root / "asset-catalog"
    publish_asset_catalog(
        assets,
        catalog_path,
        root,
        NearDuplicatePolicy(max_phash_hamming_distance=0),
        coverage_roots=("images", "query-images"),
    )
    catalog = load_asset_catalog(catalog_path, root, verify_files=True)
    by_product = {
        asset.product_id: asset
        for asset in catalog.assets
        if asset.product_id is not None
    }
    query_asset = by_product["query-only"]
    text = "compare these products"
    query = Query(
        schema_version=2,
        taxonomy_version="product-index-test-taxonomy-v1",
        task_spec_version="product-index-test-spec-v1",
        query_id="product-index-test-query",
        asset_id=query_asset.asset_id,
        image_path=query_asset.local_path,
        leakage_group_id=catalog.component_for_asset(query_asset.asset_id),
        template_family="product-index-test-family",
        generator_batch_id="product-index-test-batch",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent="multi_product",
        canonical_capability=None,
        acceptable_capabilities=[],
        requires_card=None,
        split="dev_mini",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="product-index-test",
                canonical_intent="multi_product",
                canonical_capability=None,
                acceptable_capabilities=[],
            )
        ],
    )
    queries_path = root / "index-queries.jsonl"
    queries_path.write_bytes(canonical_jsonl_bytes([query]))
    gallery_path = root / "index-gallery.jsonl"
    gallery_references = [
        GalleryAssetReference(
            asset_id=by_product[row["product_id"]].asset_id,
            image_path=by_product[row["product_id"]].local_path,
        )
        for row in reversed(rows)
    ]
    gallery_path.write_bytes(canonical_jsonl_bytes(gallery_references))
    eligibility = build_gallery_eligibility_manifest(
        queries_path,
        gallery_path,
        catalog,
    )
    eligibility_path = root / "index-eligibility.json"
    eligibility_path.write_bytes(canonical_json_bytes(eligibility))
    return {
        "eligibility_manifest_path": eligibility_path,
        "query_artifact_path": queries_path,
        "gallery_artifact_path": gallery_path,
        "asset_catalog": catalog,
    }


def _build_verified(
    products_path: Path,
    output: Path,
    backend,
    limit: int | None = None,
    *,
    verified_inputs: dict[str, Any] | None = None,
):
    inputs = verified_inputs or _verified_inputs(products_path)
    return build_product_index(
        products_path,
        output,
        backend,
        limit=limit,
        **inputs,
    )


def _load_build_index_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_writes_auditable_aligned_index_and_smoke_is_incomplete(tmp_path: Path):
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    backend = FakeBackend()

    report = build_product_index(
        products,
        output,
        backend,
        limit=2,
        allow_provisional_gallery=True,
    )

    assert report.products == 2
    assert report.complete is False
    assert report.batches == {"image": 1, "text": 1}
    assert report.api_calls == 0
    assert report.cache_hits == 0
    assert {path.name for path in output.iterdir()} == {
        "image.faiss",
        "text.faiss",
        "image_vectors.npy",
        "text_vectors.npy",
        "products.parquet",
        "manifest.json",
    }
    assert [paths[0].name for paths in backend.image_calls] == ["product-0.png"]
    assert backend.text_calls == [[CANARY_TEXT], ["title 0", "title 1"]]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "test-embedding-model"
    assert manifest["dimension"] == EMBEDDING_DIMENSION
    assert manifest["normalization"] == "l2"
    assert manifest["complete"] is False
    assert manifest["format_version"] == 4
    assert manifest["embedding_execution_location"] == "local"
    assert manifest["embedding_runtime_assurance"] == "diagnostic-unattested"
    assert manifest["embedding_runtime_binding_sha256"] is None
    assert manifest["query_gallery_eligibility"] == {
        "mode": "provisional",
        "reason": "explicit-limited-smoke",
    }
    assert manifest["source"]["row_count"] == 3
    assert manifest["product_count"] == 2
    assert manifest["vector_counts"] == {"image": 2, "text": 2}
    assert manifest["faiss"]["type"] == "IndexFlatIP"
    assert manifest["batch_parameters"] == {
        "image_batch_size": IMAGE_BATCH_SIZE,
        "text_batch_size": TEXT_BATCH_SIZE,
    }
    assert (
        manifest["canary"]["text_sha256"]
        == hashlib.sha256(CANARY_TEXT.encode("utf-8")).hexdigest()
    )
    assert manifest["canary"]["dtype"] == "float32"
    canary_vector = np.asarray(manifest["canary"]["vector"], dtype=np.float32)
    assert canary_vector.shape == (EMBEDDING_DIMENSION,)
    assert np.linalg.norm(canary_vector) == pytest.approx(1.0)
    assert set(manifest["artifacts"]) == {
        "image.faiss",
        "text.faiss",
        "image_vectors.npy",
        "text_vectors.npy",
        "products.parquet",
    }
    assert "D:" not in json.dumps(manifest)

    with pytest.raises(ValueError, match="manifest.json"):
        ProductIndex.load(output)
    with pytest.raises(ValueError, match="provisional gallery"):
        ProductIndex.load(output, allow_incomplete=True)
    index = ProductIndex.load(
        output,
        allow_incomplete=True,
        allow_provisional_gallery=True,
    )
    assert [product.product_id for product in index.products] == [
        "product-0",
        "product-1",
    ]
    assert index.image_index.d == EMBEDDING_DIMENSION
    assert index.image_index.ntotal == 2
    assert index.text_index.d == EMBEDDING_DIMENSION
    assert index.text_index.ntotal == 2
    assert np.array_equal(index.image_index.reconstruct_n(0, 2), index.image_vectors)
    assert np.array_equal(index.text_index.reconstruct_n(0, 2), index.text_vectors)


def test_build_fails_closed_without_complete_gate_or_explicit_limited_smoke(
    tmp_path: Path,
):
    products = _write_products(tmp_path)
    backend = FakeBackend()

    with pytest.raises(ValueError, match="formal product index|eligibility bundle"):
        build_product_index(products, tmp_path / "unbound", backend)
    with pytest.raises(ValueError, match="requires eligibility manifest"):
        build_product_index(
            products,
            tmp_path / "partial",
            backend,
            eligibility_manifest_path=tmp_path / "eligibility.json",
        )
    with pytest.raises(ValueError, match="limited smoke"):
        build_product_index(
            products,
            tmp_path / "full-provisional",
            backend,
            allow_provisional_gallery=True,
        )

    assert backend.image_calls == []
    assert backend.text_calls == []


def test_build_batches_without_reordering_and_marks_full_build_complete(tmp_path: Path):
    count = max(IMAGE_BATCH_SIZE + 1, TEXT_BATCH_SIZE + 1)
    products = _write_products(tmp_path, count=count)
    backend = FakeBackend()

    report = _build_verified(products, tmp_path / "index", backend)

    assert report.complete is True
    assert report.products == count
    assert [len(batch) for batch in backend.image_calls] == [
        min(IMAGE_BATCH_SIZE, count - start)
        for start in range(0, count, IMAGE_BATCH_SIZE)
    ]
    assert [len(batch) for batch in backend.text_calls[1:]] == [
        min(TEXT_BATCH_SIZE, count - start)
        for start in range(0, count, TEXT_BATCH_SIZE)
    ]
    assert [path.name for batch in backend.image_calls for path in batch] == [
        f"product-{index}.png" for index in range(count)
    ]
    assert backend.text_calls[0] == [CANARY_TEXT]
    assert [text for batch in backend.text_calls[1:] for text in batch] == [
        f"title {index}" for index in range(count)
    ]
    assert ProductIndex.load(tmp_path / "index").manifest["complete"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "mep3m", "source does not match"),
        ("product_id", "substituted-product", "product_id does not match"),
    ],
)
def test_build_rejects_product_identity_substitution(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    products = _write_products(tmp_path)
    verified_inputs = _verified_inputs(products)
    table = pq.read_table(products)
    rows = table.to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows), products)

    with pytest.raises(ValueError, match=message):
        _build_verified(
            products,
            tmp_path / "substituted-index",
            FakeBackend(),
            verified_inputs=verified_inputs,
        )


def test_build_rechecks_gallery_asset_bytes_after_embedding(tmp_path: Path):
    products = _write_products(tmp_path)
    verified_inputs = _verified_inputs(products)

    class MutatingBackend(FakeBackend):
        def embed_images(self, paths):
            vectors = super().embed_images(paths)
            Image.new("RGB", (32, 32), "white").save(paths[0])
            return vectors

    output = tmp_path / "mutated-index"
    with pytest.raises(ValueError, match="sha256|asset|catalog"):
        _build_verified(
            products,
            output,
            MutatingBackend(),
            verified_inputs=verified_inputs,
        )
    assert not output.exists()


def test_build_rejects_missing_images_duplicate_ids_and_backend_row_mismatches(
    tmp_path: Path,
):
    products = _write_products(tmp_path, count=2)
    table = pq.read_table(products)
    rows = table.to_pylist()

    missing_image_rows = [dict(row) for row in rows]
    missing_image_rows[0]["image_path"] = "images/not-there.png"
    missing_image = tmp_path / "missing-image.parquet"
    pq.write_table(pa.Table.from_pylist(missing_image_rows), missing_image)
    with pytest.raises(ValueError, match="absent from verified gallery"):
        _build_verified(
            missing_image,
            tmp_path / "missing-index",
            FakeBackend(),
            verified_inputs=_verified_inputs(products),
        )

    duplicate_rows = [dict(row) for row in rows]
    duplicate_rows[1]["product_id"] = duplicate_rows[0]["product_id"]
    duplicate = tmp_path / "duplicate.parquet"
    pq.write_table(pa.Table.from_pylist(duplicate_rows), duplicate)
    with pytest.raises(ValueError, match="duplicate product_id"):
        _build_verified(
            duplicate,
            tmp_path / "duplicate-index",
            FakeBackend(),
            verified_inputs=_verified_inputs(products),
        )

    class ShortBackend(FakeBackend):
        def embed_images(self, paths):
            super().embed_images(paths)
            return _unit_vectors(len(paths) - 1)

    with pytest.raises(ValueError, match="image backend response"):
        _build_verified(products, tmp_path / "short-index", ShortBackend())


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        (
            "manifest.json",
            lambda path: path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "test-embedding-model", "tampered-model"
                ),
                encoding="utf-8",
            ),
        ),
        ("image.faiss", lambda path: path.write_bytes(path.read_bytes() + b"tampered")),
        ("text.faiss", lambda path: path.write_bytes(path.read_bytes() + b"tampered")),
        (
            "image_vectors.npy",
            lambda path: path.write_bytes(path.read_bytes() + b"tampered"),
        ),
        (
            "text_vectors.npy",
            lambda path: path.write_bytes(path.read_bytes() + b"tampered"),
        ),
        (
            "products.parquet",
            lambda path: path.write_bytes(path.read_bytes() + b"tampered"),
        ),
        (
            "query-gallery-eligibility.json",
            lambda path: path.write_bytes(path.read_bytes() + b"tampered"),
        ),
        (
            "gallery-assets.jsonl",
            lambda path: path.write_bytes(path.read_bytes() + b"tampered"),
        ),
    ],
)
def test_load_reports_the_specific_tampered_artifact(tmp_path: Path, filename, mutate):
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    _build_verified(products, output, FakeBackend())

    mutate(output / filename)

    with pytest.raises(ValueError, match=filename):
        ProductIndex.load(output)


def test_load_requires_canonical_manifest_even_when_integrity_is_unchanged(
    tmp_path: Path,
):
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    _build_verified(products, output, FakeBackend())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        ProductIndex.load(output)


def test_retrieval_binding_requires_the_exact_audited_query_artifact(tmp_path: Path):
    products = _write_products(tmp_path)
    inputs = _verified_inputs(products)
    output = tmp_path / "index"
    _build_verified(
        products,
        output,
        FakeBackend(),
        verified_inputs=inputs,
    )
    index = ProductIndex.load(output)

    binding = index.retrieval_binding(inputs["query_artifact_path"])
    assert binding.mode == "verified"
    assert (
        binding.eligibility_sha256
        == index.manifest["query_gallery_eligibility"]["eligibility_sha256"]
    )
    with pytest.raises(ValueError, match="requires a query artifact"):
        index.retrieval_binding(None)
    other = tmp_path / "other-queries.jsonl"
    other.write_bytes(b"different\n")
    with pytest.raises(ValueError, match="does not match"):
        index.retrieval_binding(other)


def test_failed_build_never_replaces_the_existing_index(tmp_path: Path):
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    verified_inputs = _verified_inputs(products)
    _build_verified(products, output, FakeBackend(), verified_inputs=verified_inputs)
    old_manifest = (output / "manifest.json").read_bytes()

    class FailingBackend(FakeBackend):
        def embed_texts(self, texts):
            raise RuntimeError("backend interrupted")

    with pytest.raises(RuntimeError, match="backend interrupted"):
        _build_verified(
            products,
            output,
            FailingBackend(),
            verified_inputs=verified_inputs,
        )

    assert (output / "manifest.json").read_bytes() == old_manifest
    assert ProductIndex.load(output).manifest["complete"] is True


@pytest.mark.parametrize(
    "invalid_vectors",
    [
        _unit_vectors(1).astype(np.float64),
        np.full((1, EMBEDDING_DIMENSION), np.nan, dtype=np.float32),
        np.full((1, EMBEDDING_DIMENSION), np.inf, dtype=np.float32),
        np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32),
    ],
    ids=["float64", "nan", "infinite", "not-unit"],
)
def test_build_rejects_invalid_backend_canary_vectors(tmp_path: Path, invalid_vectors):
    products = _write_products(tmp_path)

    class InvalidCanaryBackend(FakeBackend):
        def embed_texts(self, texts):
            self.text_calls.append(list(texts))
            return invalid_vectors

    with pytest.raises(ValueError, match="canary backend response"):
        _build_verified(products, tmp_path / "index", InvalidCanaryBackend())


def test_load_rejects_a_structurally_tampered_canary(tmp_path: Path):
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    _build_verified(products, output, FakeBackend())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canary"]["vector"][0] = 0.5
    unsigned = dict(manifest)
    unsigned.pop("integrity_sha256")
    manifest["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json"):
        ProductIndex.load(output)


def test_load_rejects_complete_manifest_that_does_not_cover_its_source(tmp_path: Path):
    products = _write_products(tmp_path, count=3)
    output = tmp_path / "smoke-index"
    _build_verified(products, output, FakeBackend(), limit=2)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = True
    unsigned = dict(manifest)
    unsigned.pop("integrity_sha256")
    manifest["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json"):
        ProductIndex.load(output)


def test_product_build_report_reads_real_cached_backend_metrics(tmp_path: Path):
    products = _write_products(tmp_path)

    class RawBackend:
        execution_location = "local"

        def embed_images(self, paths):
            return _unit_vectors(len(paths))

        def embed_texts(self, texts):
            return _unit_vectors(len(texts))

    cached = CachedEmbeddingBackend(
        RawBackend(), EmbeddingCache(tmp_path / "cache.sqlite3")
    )

    verified_inputs = _verified_inputs(products)
    first = _build_verified(
        products,
        tmp_path / "first",
        cached,
        verified_inputs=verified_inputs,
    )
    second = _build_verified(
        products,
        tmp_path / "second",
        cached,
        verified_inputs=verified_inputs,
    )

    assert first.cache_hits == 0
    assert first.api_calls == 3
    assert first.cache_hits_by_modality == {"image": 0, "text": 0}
    assert first.api_calls_by_modality == {"image": 1, "text": 2}
    assert second.cache_hits == 7
    assert second.api_calls == 0
    assert second.cache_hits_by_modality == {"image": 3, "text": 4}
    assert second.api_calls_by_modality == {"image": 0, "text": 0}


def test_remote_index_build_checks_every_gallery_upload_permission_before_calls(
    tmp_path: Path,
):
    denied_root = tmp_path / "denied"
    denied_root.mkdir()
    products = _write_products(denied_root)

    # A custom backend's self-declared ``local`` location is not trusted.  It is
    # conservatively treated as remote until a concrete runtime is attested.
    denied_backend = FakeBackend()
    denied_inputs = _verified_inputs(products, cloud_upload_allowed=False)
    with pytest.raises(ValueError, match="not licensed"):
        _build_verified(
            products,
            denied_root / "index",
            denied_backend,
            verified_inputs=denied_inputs,
        )
    assert denied_backend.image_calls == []
    assert denied_backend.text_calls == []

    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    allowed_products = _write_products(allowed_root)
    allowed_backend = FakeBackend()
    allowed_inputs = _verified_inputs(allowed_products, cloud_upload_allowed=True)
    _build_verified(
        allowed_products,
        allowed_root / "index",
        allowed_backend,
        verified_inputs=allowed_inputs,
    )
    assert allowed_backend.image_calls

    manifest_path = allowed_root / "index" / "manifest.json"
    locked = ProductIndex.load(
        allowed_root / "index",
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="attested build runtime"):
        locked.require_formal_verified()


def test_build_reembeds_a_cached_canary_when_no_trusted_baseline_exists(tmp_path: Path):
    products = _write_products(tmp_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    stale_vector = _unit_vectors(1)[0]
    cache.put_many("text", [(CANARY_TEXT, stale_vector, "stale-request")])
    cache.put_many("text", [("title 0", stale_vector, "stale-product")])
    cache.put_many(
        "image",
        [(tmp_path / "images" / "product-0.png", stale_vector, "stale-image")],
    )
    fresh_vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    fresh_vector[1] = 1.0

    class NewModelBackend:
        execution_location = "local"

        def __init__(self):
            self.text_calls = []
            self.image_calls = []

        def embed_images(self, paths):
            self.image_calls.append(list(paths))
            return np.tile(fresh_vector, (len(paths), 1))

        def embed_texts(self, texts):
            self.text_calls.append(list(texts))
            return np.tile(fresh_vector, (len(texts), 1))

    raw_backend = NewModelBackend()
    cached = CachedEmbeddingBackend(raw_backend, cache)

    _build_verified(products, tmp_path / "index", cached)

    index = ProductIndex.load(tmp_path / "index")
    assert raw_backend.text_calls[0] == [CANARY_TEXT]
    assert raw_backend.text_calls[1] == ["title 0", "title 1", "title 2"]
    assert [path.name for path in raw_backend.image_calls[0]] == [
        "product-0.png",
        "product-1.png",
        "product-2.png",
    ]
    assert np.array_equal(index.canary_vector, fresh_vector)
    assert np.array_equal(cached.canary_vector, fresh_vector)


def test_default_cli_backend_reuses_a_valid_index_canary(tmp_path: Path, monkeypatch):
    script = _load_build_index_script()
    products = _write_products(tmp_path)
    output = tmp_path / "index"
    _build_verified(products, output, FakeBackend())
    monkeypatch.setattr(script, "EMBEDDING_CACHE_PATH", tmp_path / "cache.sqlite3")
    expected_canary_vector = ProductIndex.load_canary(output).vector

    monkeypatch.setattr(
        script.ProductIndex,
        "load",
        classmethod(
            lambda cls, *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("full load")
            )
        ),
    )

    backend = script._default_backend(output)

    assert backend.canary_text == CANARY_TEXT
    assert np.array_equal(backend.canary_vector, expected_canary_vector)


def test_cli_reports_json_status_and_errors(tmp_path: Path, capsys):
    script = _load_build_index_script()
    products = _write_products(tmp_path)
    output = tmp_path / "smoke-index"

    result = script.main(
        [
            "products",
            "--source",
            str(products),
            "--output",
            str(output),
            "--limit",
            "2",
            "--allow-provisional-gallery",
        ],
        backend_factory=FakeBackend,
    )
    build_stdout = json.loads(capsys.readouterr().out)
    assert result == 0
    assert {
        "products",
        "batches",
        "cache_hits",
        "api_calls",
        "cache_hits_by_modality",
        "api_calls_by_modality",
        "items_per_second",
        "elapsed_seconds",
        "complete",
    } <= set(build_stdout)
    assert build_stdout["products"] == 2
    assert build_stdout["complete"] is False

    assert script.main(["status", "--output", str(output)]) == 2
    assert "manifest.json" in capsys.readouterr().err
    assert script.main(["status", "--output", str(output), "--allow-incomplete"]) == 2
    assert "provisional gallery" in capsys.readouterr().err
    assert (
        script.main(
            [
                "status",
                "--output",
                str(output),
                "--allow-incomplete",
                "--allow-provisional-gallery",
            ]
        )
        == 0
    )
    status_stdout = json.loads(capsys.readouterr().out)
    assert status_stdout == {
        "complete": False,
        "dimension": EMBEDDING_DIMENSION,
        "products": 2,
        "eligibility_mode": "provisional",
        "eligibility_sha256": None,
        "query_artifact_sha256": None,
        "gallery_artifact_sha256": None,
        "asset_catalog_sha256": None,
    }

    assert script.main(["products", "--source", str(tmp_path / "missing.parquet")]) == 2
    assert "build-index:" in capsys.readouterr().err
