from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

import skillchain.tools.document_ocr as document_ocr_runtime
import skillchain.tools.object_detect as object_detect_runtime

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    GalleryAssetReference,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.gallery_eligibility import (
    GalleryEligibilityViolation,
    build_gallery_eligibility_manifest,
)
from skillchain.data.kb_catalog import build_kb_catalog
from skillchain.schemas import LabelDecision, Query
from skillchain.tools.document_ocr import (
    DocumentOCRService,
    OCRConfig,
    RapidOCROnnxBackend,
    build_document_safety_approval,
)
from skillchain.tools.document_safety import (
    DocumentSafetyCatalog,
    build_document_safety_record,
    document_safety_review_ledger_digest,
    publish_document_safety_catalog,
)
from skillchain.tools.embedding import DashScopeEmbeddingClient, FormalEmbeddingBackend
from skillchain.tools.kb_index import build_kb_bundle, load_kb_bundle
from skillchain.tools.kb_lookup import KBLookupService
from skillchain.tools.model_artifacts import (
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.object_detect import (
    DetectorClass,
    DetectorConfig,
    DetectorLabels,
    ObjectDetectionService,
    UltralyticsDetectorBackend,
)
from skillchain.tools.product_index import (
    CANARY_TEXT,
    ProductIndex,
    build_product_index,
)
from skillchain.tools.product_search import ProductSearchService
from skillchain.tools.registry import MVPToolServices, ToolRegistry, build_mvp_registry
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.settings import EMBEDDING_DIMENSION


@pytest.fixture(autouse=True)
def _block_network_in_nonintegration_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Make the default pytest profile mechanically reject Python socket use."""

    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    def denied(*_args, **_kwargs):
        raise RuntimeError(
            "network access is disabled for non-integration tests; "
            "mark the test integration and opt in explicitly"
        )

    class GuardedSocket(socket.socket):
        def connect(self, *_args, **_kwargs):
            return denied()

        def connect_ex(self, *_args, **_kwargs):
            return denied()

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "gethostbyaddr", denied)
    monkeypatch.setattr(socket, "gethostbyname", denied)
    monkeypatch.setattr(socket, "gethostbyname_ex", denied)
    yield


@dataclass(frozen=True)
class CanonicalRegistryFixture:
    registry: ToolRegistry
    asset_catalog: Any
    product_search: ProductSearchService
    kb_lookup: KBLookupService
    object_detection: ObjectDetectionService
    document_ocr: DocumentOCRService
    safety_catalog: DocumentSafetyCatalog


class _VectorResponseAdapter(requests.adapters.HTTPAdapter):
    def __init__(
        self,
        *,
        text_rows: Mapping[str, int],
        image_rows: Mapping[str, int],
        vectors: np.ndarray,
        canary_vector: np.ndarray,
    ) -> None:
        super().__init__()
        self._text_rows = dict(text_rows)
        self._image_rows = dict(image_rows)
        self._vectors = vectors
        self._canary_vector = canary_vector
        self._request_count = 0

    def send(self, request, **_kwargs):
        payload = json.loads(request.body)
        contents = payload["input"]["contents"]
        embeddings = []
        for index, content in enumerate(contents):
            vector = self._vector_for(content)
            embeddings.append({"index": index, "embedding": vector.tolist()})
        self._request_count += 1
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(  # noqa: SLF001
            {
                "request_id": f"fixture-request-{self._request_count}",
                "output": {"embeddings": embeddings},
            }
        ).encode("utf-8")
        return response

    def _vector_for(self, content: Mapping[str, str]) -> np.ndarray:
        text = content.get("text")
        if text is not None:
            if text == CANARY_TEXT:
                return self._canary_vector
            return self._vectors[self._text_rows.get(text, 0)]
        encoded = content["image"].split(",", maxsplit=1)[1]
        digest = sha256_bytes(base64.b64decode(encoded))
        return self._vectors[self._image_rows.get(digest, 0)]


class _ArrayView:
    def __init__(self, value: Sequence[Any]) -> None:
        self._value = list(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value


class _DetectorBoxes:
    xyxy = _ArrayView(((1.0, 1.0, 10.0, 10.0),))
    conf = _ArrayView((0.9,))
    cls = _ArrayView((0,))


class _DetectorResult:
    boxes = _DetectorBoxes()


class _DetectorModel:
    def predict(self, **_kwargs):
        return (_DetectorResult(),)


class _OCREngine:
    def __call__(self, *_args, **_kwargs):
        return (
            (
                (
                    ((1.0, 1.0), (10.0, 1.0), (10.0, 5.0), (1.0, 5.0)),
                    "Alice",
                    0.9,
                ),
            ),
            0.0,
        )


def _kb_entry(entry_id: str, kind: str, title: str, text: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "entry_id": entry_id,
        "title": title,
        "text": text,
        "kind": kind,
        "origin": "dump",
        "source_dataset": "canonical-registry-fixture",
        "source_revision": "fixture-v1",
        "source_record_id": entry_id,
        "source_uri": f"https://example.test/kb/{entry_id}",
        "license_id": "CC-BY-4.0",
        "attribution": "Fixture Author",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entity_group_id": f"entity-{entry_id}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": "source_verified",
    }


def _default_safety_catalog(
    root: Path,
    catalog: Any,
) -> DocumentSafetyCatalog:
    asset = next(
        (
            candidate
            for candidate in catalog.assets
            if candidate.source_record_id == "query-only"
        ),
        catalog.assets[0],
    )
    approval = build_document_safety_approval(
        asset_id=asset.asset_id,
        image_sha256=asset.sha256,
        decision="approved_no_pii",
        policy_version="canonical-fixture-safety-v1",
        reviewer_id="canonical-fixture-reviewer",
    )
    records = (build_document_safety_record("fixture-query", approval),)
    review_digest = document_safety_review_ledger_digest(records)
    return publish_document_safety_catalog(
        records,
        root / "document-safety",
        catalog,
        expected_review_ledger_sha256=review_digest,
    )


def _default_product_catalog(
    root: Path,
    product_ids: Sequence[str],
) -> tuple[Any, Path]:
    asset_root = root / "product-assets"
    image_root = asset_root / "images"
    image_root.mkdir(parents=True)
    drafts = []
    for index, product_id in enumerate(product_ids):
        path = image_root / f"gallery-{index}.png"
        pixels = np.random.default_rng(index + 10_000).integers(
            0,
            256,
            size=(32, 32, 3),
            dtype=np.uint8,
        )
        Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
        drafts.append(
            DatasetAssetDraft(
                source_dataset="mep3m",
                source_revision="canonical-fixture-v1",
                source_record_id=product_id,
                local_path=path.relative_to(asset_root).as_posix(),
                product_id=product_id,
                license_id="test-only",
                cloud_upload_allowed=True,
            )
        )
    query_path = image_root / "query.png"
    query_pixels = np.random.default_rng(99_999).integers(
        0,
        256,
        size=(32, 32, 3),
        dtype=np.uint8,
    )
    Image.fromarray(query_pixels, mode="RGB").save(query_path, format="PNG")
    drafts.append(
        DatasetAssetDraft(
            source_dataset="mep3m",
            source_revision="canonical-fixture-v1",
            source_record_id="query-only",
            local_path=query_path.relative_to(asset_root).as_posix(),
            product_id="query-only",
            license_id="test-only",
            cloud_upload_allowed=True,
        )
    )
    assets = tuple(inventory_dataset_asset(draft, asset_root) for draft in drafts)
    catalog_root = root / "product-asset-catalog"
    publish_asset_catalog(
        assets,
        catalog_root,
        asset_root,
        NearDuplicatePolicy(max_phash_hamming_distance=0),
        coverage_roots=("images",),
    )
    catalog = load_asset_catalog(catalog_root, asset_root, verify_files=True)
    query_asset = next(
        asset for asset in catalog.assets if asset.source_record_id == "query-only"
    )
    text = "canonical registry fixture query"
    query = Query(
        schema_version=2,
        taxonomy_version="canonical-fixture-taxonomy-v1",
        task_spec_version="canonical-fixture-task-v1",
        query_id="canonical-fixture-query",
        asset_id=query_asset.asset_id,
        image_path=query_asset.local_path,
        leakage_group_id=catalog.component_for_asset(query_asset.asset_id),
        template_family="canonical-fixture-family",
        generator_batch_id="canonical-fixture-batch",
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
                annotator_id="canonical-fixture",
                canonical_intent="multi_product",
                canonical_capability=None,
                acceptable_capabilities=[],
            )
        ],
    )
    artifact = root / "queries.jsonl"
    artifact.write_bytes(canonical_jsonl_bytes((query.model_dump(mode="json"),)))
    return catalog, artifact


def _model_services(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ObjectDetectionService, DocumentOCRService]:
    detector_root = root / "detector"
    detector_root.mkdir()
    detector_labels = DetectorLabels(
        classes=(DetectorClass(class_id=0, label="cat", label_zh="cat"),)
    )
    detector_config = DetectorConfig(
        image_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        max_detections=100,
    )
    detector_files = {
        "weights": detector_root / "weights.bin",
        "labels": detector_root / "labels.json",
        "config": detector_root / "config.json",
    }
    detector_files["weights"].write_bytes(b"fixture-detector-weights")
    detector_files["labels"].write_bytes(
        canonical_json_bytes(detector_labels.model_dump(mode="json"))
    )
    detector_files["config"].write_bytes(
        canonical_json_bytes(detector_config.model_dump(mode="json"))
    )
    published_detector = publish_model_artifact_manifest(
        detector_root / "manifest.json",
        artifact_kind="object_detector",
        model_id="canonical-fixture-detector",
        backend_name="ultralytics",
        backend_version="1.0.0",
        artifacts=detector_files,
    )
    detector_artifact = load_model_artifact_manifest(
        published_detector.manifest_path,
        expected_manifest_sha256=published_detector.manifest.manifest_sha256,
    )
    ultralytics = ModuleType("ultralytics")
    setattr(ultralytics, "YOLO", lambda _weights: _DetectorModel())
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setattr(
        object_detect_runtime,
        "metadata",
        SimpleNamespace(version=lambda _package: "1.0.0"),
    )
    detector_backend = UltralyticsDetectorBackend(detector_artifact)
    detector_backend.detect(Image.new("RGB", (16, 16)), config=detector_config)
    detector_service = ObjectDetectionService(detector_artifact, detector_backend)

    ocr_root = root / "ocr"
    ocr_root.mkdir()
    ocr_config = OCRConfig(
        languages=("en",),
        max_lines=100,
        max_characters=1_000,
        use_classifier=False,
    )
    ocr_files = {
        "detector_model": ocr_root / "detector.onnx",
        "recognizer_model": ocr_root / "recognizer.onnx",
        "labels": ocr_root / "characters.txt",
        "config": ocr_root / "config.json",
    }
    ocr_files["detector_model"].write_bytes(b"fixture-ocr-detector")
    ocr_files["recognizer_model"].write_bytes(b"fixture-ocr-recognizer")
    ocr_files["labels"].write_text("A\nl\ni\nc\ne\n", encoding="utf-8")
    ocr_files["config"].write_bytes(
        canonical_json_bytes(ocr_config.model_dump(mode="json"))
    )
    published_ocr = publish_model_artifact_manifest(
        ocr_root / "manifest.json",
        artifact_kind="document_ocr",
        model_id="canonical-fixture-ocr",
        backend_name="rapidocr-onnxruntime",
        backend_version="1.0.0",
        artifacts=ocr_files,
    )
    ocr_artifact = load_model_artifact_manifest(
        published_ocr.manifest_path,
        expected_manifest_sha256=published_ocr.manifest.manifest_sha256,
    )
    rapidocr = ModuleType("rapidocr_onnxruntime")
    setattr(rapidocr, "RapidOCR", lambda **_arguments: _OCREngine())
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", rapidocr)
    monkeypatch.setattr(
        document_ocr_runtime,
        "metadata",
        SimpleNamespace(version=lambda _package: "1.0.0"),
    )
    ocr_backend = RapidOCROnnxBackend(ocr_artifact)
    ocr_backend.recognize(Image.new("RGB", (16, 16)), config=ocr_config)
    ocr_service = DocumentOCRService(ocr_artifact, ocr_backend)
    return detector_service, ocr_service


def _kb_service(
    root: Path,
    entries: Mapping[str, tuple[str, str, str]] | None,
) -> KBLookupService:
    values = entries or {
        "encyclopedia": ("entry-q-1", "dress", "dress fixture evidence"),
        "recipe": ("recipe-q-1", "soup", "soup fixture evidence"),
    }
    encyclopedia_id, encyclopedia_title, encyclopedia_text = values["encyclopedia"]
    recipe_id, recipe_title, recipe_text = values["recipe"]
    encyclopedia = _kb_entry(
        encyclopedia_id,
        "encyclopedia",
        encyclopedia_title,
        encyclopedia_text,
    )
    recipe = _kb_entry(recipe_id, "recipe", recipe_title, recipe_text)
    encyclopedia_path = root / "encyclopedia.jsonl"
    recipe_path = root / "recipes.jsonl"
    encyclopedia_path.write_bytes(canonical_jsonl_bytes((encyclopedia,)))
    recipe_path.write_bytes(canonical_jsonl_bytes((recipe,)))
    catalog = build_kb_catalog(
        encyclopedia_path,
        recipe_path,
        root / "kb-catalog",
        expected_encyclopedia_sha256=sha256_bytes(encyclopedia_path.read_bytes()),
        expected_recipe_sha256=sha256_bytes(recipe_path.read_bytes()),
    )
    published = build_kb_bundle(
        catalog.root,
        root / "kb-index",
        expected_catalog_sha256=catalog.manifest.catalog_sha256,
    )
    bundle = load_kb_bundle(
        published.root,
        catalog=catalog,
        expected_bundle_sha256=published.manifest.bundle_sha256,
    )
    return KBLookupService(bundle)


def _product_service(
    *,
    root: Path,
    query_artifact: Path,
    asset_catalog: Any,
    product_ids: Sequence[str],
    text_product_ids: Mapping[str, str],
    image_product_ids: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> ProductSearchService:
    by_product = {
        asset.product_id: asset
        for asset in asset_catalog.assets
        if asset.product_id is not None and asset.source_record_id == asset.product_id
    }
    rows = []
    for product_id in product_ids:
        asset = by_product[product_id]
        rows.append(
            {
                "product_id": product_id,
                "title": f"fixture {product_id}",
                "category_l1": "fixture",
                "category_l2": None,
                "category_l3": None,
                "ocr_text": None,
                "image_path": asset.local_path,
                "source": "mep3m",
            }
        )
    products_path = asset_catalog.asset_root / f"{root.name}-products.parquet"
    pq.write_table(pa.Table.from_pylist(rows), products_path)
    gallery_path = root / "gallery-assets.jsonl"
    gallery_path.write_bytes(
        canonical_jsonl_bytes(
            tuple(
                GalleryAssetReference(
                    asset_id=by_product[product_id].asset_id,
                    image_path=by_product[product_id].local_path,
                ).model_dump(mode="json")
                for product_id in product_ids
            )
        )
    )
    try:
        eligibility = build_gallery_eligibility_manifest(
            query_artifact,
            gallery_path,
            asset_catalog,
        )
    except GalleryEligibilityViolation as error:
        raise AssertionError(
            f"canonical registry fixture is not query/gallery eligible: {error.payload}"
        ) from error
    eligibility_path = root / "query-gallery-eligibility.json"
    eligibility_path.write_bytes(
        canonical_json_bytes(eligibility.model_dump(mode="json"))
    )

    vectors = np.zeros((len(product_ids), EMBEDDING_DIMENSION), dtype=np.float32)
    for row in range(len(product_ids)):
        vectors[row, row + 1] = 1.0
    canary = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    canary[0] = 1.0
    row_by_product = {product_id: row for row, product_id in enumerate(product_ids)}
    gallery_image_rows = {
        by_product[product_id].sha256: row_by_product[product_id]
        for product_id in product_ids
    }
    transport = _VectorResponseAdapter(
        text_rows={
            text: row_by_product[product_id]
            for text, product_id in text_product_ids.items()
        }
        | {
            f"fixture {product_id}": row_by_product[product_id]
            for product_id in product_ids
        },
        image_rows={
            digest: row_by_product[product_id]
            for digest, product_id in image_product_ids.items()
        }
        | gallery_image_rows,
        vectors=vectors,
        canary_vector=canary,
    )

    def send(_adapter, request, **kwargs):
        return transport.send(request, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    client = DashScopeEmbeddingClient(api_key="fixture-key")
    backend = FormalEmbeddingBackend(
        client,
        canary_text=CANARY_TEXT,
        canary_vector=canary,
    )
    backend.verify_canary()
    output = root / "product-index"
    build_product_index(
        products_path,
        output,
        backend,
        eligibility_manifest_path=eligibility_path,
        query_artifact_path=query_artifact,
        gallery_artifact_path=gallery_path,
        asset_catalog=asset_catalog,
    )
    manifest_path = output / "manifest.json"
    index = ProductIndex.load(
        output,
        expected_manifest_sha256=sha256_bytes(manifest_path.read_bytes()),
    )
    return ProductSearchService(index, backend, query_artifact=query_artifact)


@pytest.fixture
def canonical_registry_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def build(
        *,
        name: str,
        query_artifact: Path | None = None,
        asset_catalog: Any | None = None,
        product_ids: Sequence[str] = ("product-q-1",),
        text_product_ids: Mapping[str, str] | None = None,
        image_product_ids: Mapping[str, str] | None = None,
        kb_entries: Mapping[str, tuple[str, str, str]] | None = None,
        safety_catalog: DocumentSafetyCatalog | None = None,
        include_multi_product: bool = False,
    ) -> CanonicalRegistryFixture:
        root = tmp_path / name
        root.mkdir()
        if asset_catalog is None:
            if query_artifact is not None:
                raise ValueError("external query artifact requires its asset catalog")
            asset_catalog, query_artifact = _default_product_catalog(root, product_ids)
        elif query_artifact is None:
            raise ValueError("asset catalog requires its authoritative query artifact")
        assert query_artifact is not None
        product_search = _product_service(
            root=root,
            query_artifact=query_artifact,
            asset_catalog=asset_catalog,
            product_ids=product_ids,
            text_product_ids=text_product_ids or {},
            image_product_ids=image_product_ids or {},
            monkeypatch=monkeypatch,
        )
        kb_lookup = _kb_service(root, kb_entries)
        object_detection, document_ocr = _model_services(root, monkeypatch)
        resolved_safety_catalog = safety_catalog or _default_safety_catalog(
            root,
            asset_catalog,
        )
        registry = build_mvp_registry(
            MVPToolServices(
                product_search=product_search,
                kb_lookup=kb_lookup,
                object_detection=object_detection,
                document_ocr=document_ocr,
                safety_approval_for=resolved_safety_catalog.approval_for,
            ),
            include_multi_product=include_multi_product,
        )
        registry.require_formal_runtime()
        return CanonicalRegistryFixture(
            registry=registry,
            asset_catalog=asset_catalog,
            product_search=product_search,
            kb_lookup=kb_lookup,
            object_detection=object_detection,
            document_ocr=document_ocr,
            safety_catalog=resolved_safety_catalog,
        )

    return build
