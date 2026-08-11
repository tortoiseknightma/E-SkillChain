from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from skillchain.tools import formal_evaluation as formal_evaluation_module
from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.evaluation.assistant_runs import (
    JudgeAuditSelectionEntry,
    load_verified_phase4_inputs,
    make_phase4_input_selection_manifest,
)
from skillchain.evaluation.packets import RubricSnapshot
from skillchain.schemas import LabelDecision, Product, Query
from skillchain.synthesis.splitting import FrozenSplitManifest, build_atomic_groups
from skillchain.taxonomy import (
    TASK_SPEC_VERSION,
    TAXONOMY_VERSION,
    capability_for_intent,
    requires_card_for_intent,
)
from skillchain.tools.benchmark import BenchmarkCaseResult
from skillchain.tools.contracts import (
    ProductHit,
    ProductSearchTrace,
    RetrievalArtifactBinding,
)
from skillchain.tools.document_ocr import (
    DocumentOCRResult,
    OCRField,
    OCRInputBinding,
    OCRLine,
    build_document_safety_approval,
)
from skillchain.tools.document_safety import (
    build_document_safety_record,
    document_safety_review_ledger_digest,
    publish_document_safety_catalog,
)
from skillchain.tools.formal_evaluation import (
    FormalEvaluationError,
    FormalGoldAssignment,
    VerifiedFormalToolRun,
    create_formal_tool_run,
    evaluate_formal_tool_run,
    load_formal_gold_assignments,
    load_verified_formal_tool_run,
    multi_product_runtime_sha256,
    require_verified_formal_evaluation,
    require_verified_formal_tool_run,
)
from skillchain.tools.formal_evaluation_cli import (
    FormalRuntimeContext,
    main as cli_main,
)
from skillchain.tools.kb_lookup import (
    KBCitation,
    KBHit,
    KBRetrievalArtifactBinding,
)
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.multi_product import MultiProductResult, MultiProductSearchService
from skillchain.tools.object_detect import (
    DetectedObject,
    DetectionInputBinding,
    ObjectDetectionResult,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.registry import MVPToolServices, build_mvp_registry


CLAIMS = (
    ("q-detect", "object_detection", "object_detect", "multi_product"),
    ("q-image", "image_product_search", "image_product_search", "exact_match"),
    ("q-kb", "kb_lookup", "encyclopedia_lookup", "encyclopedia"),
    ("q-multi", "multi_product_chain", "multi_product_chain", "multi_product"),
    ("q-ocr", "document_ocr", "document_ocr", "utility"),
    ("q-recipe", "kb_lookup", "recipe_lookup", "encyclopedia"),
    ("q-style", "style_similar_search", "style_similar_search", "divergent_rec"),
    ("q-text", "text_product_search", "text_product_search", "exact_match"),
)


def _write_image(path: Path, seed: int) -> None:
    image = Image.new("RGB", (32, 32), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(32):
        for x in range(32):
            if ((x * (seed + 3) + y * (seed + 5) + x * y) % (7 + seed)) < 3:
                pixels[x, y] = ((seed * 31) % 255, (x * 7) % 255, (y * 11) % 255)
    image.save(path, format="PNG")


def _query(catalog, asset_id: str, query_id: str, intent: str) -> Query:
    resolution = catalog.resolve_asset_id(asset_id)
    text = f"authoritative text for {query_id}"
    capability = capability_for_intent(intent)
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=asset_id,
        image_path=resolution.local_path,
        leakage_group_id=resolution.leakage_group_id,
        template_family="formal-evaluation-fixture-v1",
        generator_batch_id=f"batch-{query_id}",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        requires_card=requires_card_for_intent(intent),
        split="test_frozen",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="formal-evaluation-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _retrieval_binding(catalog, query_sha256: str) -> RetrievalArtifactBinding:
    return RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256=query_sha256,
        gallery_artifact_sha256="3" * 64,
        asset_catalog_sha256=catalog.catalog_sha256,
        products_parquet_sha256="4" * 64,
        leakage_policy_version=catalog.leakage_policy_version,
    )


def _model_runtime(kind: str) -> ModelRuntimeBinding:
    return ModelRuntimeBinding(
        artifact_kind=kind,
        model_id=f"fixture-{kind}",
        backend_name="fixture-backend",
        backend_version="1.0",
        manifest_sha256=("5" if kind == "object_detector" else "6") * 64,
        artifacts=(),
    )


def _product_trace(
    tool_name: str,
    *,
    query_id: str,
    binding: RetrievalArtifactBinding,
    asset_id: str | None = None,
    image_path: Path | None = None,
    text: str | None = None,
) -> ProductSearchTrace:
    product = Product(
        product_id=f"product-{query_id}",
        title="fixture product",
        category_l1="fixture",
        image_path="products/fixture.png",
        source="muge",
    )
    hit = ProductHit(
        rank=1,
        score=0.9,
        product=product,
        artifact_binding=binding,
    )
    if tool_name == "text_product_search":
        assert text is not None
        return ProductSearchTrace(
            tool_name=tool_name,
            input_kind="text",
            query_input_sha256=sha256_bytes(text.encode()),
            query_text=text,
            query_vector_sha256="7" * 64,
            artifact_binding=binding,
            hits=(hit,),
        )
    assert image_path is not None and asset_id is not None
    return ProductSearchTrace(
        tool_name=tool_name,
        input_kind="image",
        query_input_sha256=sha256_bytes(image_path.read_bytes()),
        query_asset_id=asset_id,
        query_vector_sha256="7" * 64,
        artifact_binding=binding,
        hits=(hit,),
    )


def _kb_binding(kind: str) -> KBRetrievalArtifactBinding:
    return KBRetrievalArtifactBinding(
        mode="verified",
        kind=kind,
        bundle_sha256="a" * 64,
        index_integrity_sha256=("b" if kind == "encyclopedia" else "f") * 64,
        catalog_sha256="c" * 64,
        catalog_entries_sha256="d" * 64,
        indexed_entries_sha256="e" * 64,
        tokenizer_policy_version="fixture-tokenizer-v1",
        ranking_policy_version="fixture-ranking-v1",
    )


class _ProductService:
    execution_location = "remote"

    def __init__(self, binding, query_id_by_asset):
        self.artifact_binding = binding
        self.query_id_by_asset = query_id_by_asset
        self.runtime_revision = "v1"

    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(
            f"formal-product-service-fixture:{self.runtime_revision}".encode()
        )

    def trace_image_product_search(self, path, *, asset_id):
        return _product_trace(
            "image_product_search",
            query_id=self.query_id_by_asset[asset_id],
            binding=self.artifact_binding,
            asset_id=asset_id,
            image_path=Path(path),
        )

    def trace_similar_styles(self, path, *, asset_id):
        return _product_trace(
            "style_similar_search",
            query_id=self.query_id_by_asset[asset_id],
            binding=self.artifact_binding,
            asset_id=asset_id,
            image_path=Path(path),
        )

    def trace_text_product_search(self, text):
        return _product_trace(
            "text_product_search",
            query_id="q-text",
            binding=self.artifact_binding,
            text=text,
        )


class _KBService:
    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(b"formal-kb-service-fixture-v1")

    def artifact_binding_for(self, kind):
        return _kb_binding(kind)

    def encyclopedia_lookup(self, _entity):
        return self._hits("encyclopedia", "q-kb")

    def recipe_lookup(self, _dish):
        return self._hits("recipe", "q-recipe")

    def _hits(self, kind, query_id):
        return [
            KBHit(
                rank=1,
                score=1.0,
                title="fixture entry",
                text="grounded fixture excerpt",
                kind=kind,
                origin="dump",
                verification_status="source_verified",
                citation=KBCitation(
                    entry_id=f"entry-{query_id}",
                    source_dataset="fixture-kb",
                    source_revision="revision-1",
                    source_record_id=query_id,
                    source_uri="https://example.invalid/source",
                    license_id="CC-BY-4.0",
                    attribution="fixture",
                    char_start=0,
                    char_end=24,
                    excerpt_sha256=sha256_bytes(b"grounded fixture excerpt"),
                ),
                artifact_binding=_kb_binding(kind),
            )
        ]


class _DetectionService:
    def __init__(self):
        self.artifact = _Artifact(_model_runtime("object_detector"))

    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(b"formal-detection-service-fixture-v1")

    def detect(self, path, *, asset_id):
        content = Path(path).read_bytes()
        return ObjectDetectionResult(
            input_binding=DetectionInputBinding(
                asset_id=asset_id,
                image_sha256=sha256_bytes(content),
                image_bytes=len(content),
                width=32,
                height=32,
            ),
            runtime_binding=self.artifact.runtime_binding,
            detections=(
                DetectedObject(
                    detection_id="f" * 64,
                    class_id=0,
                    label="cat",
                    label_zh="cat",
                    bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
                    confidence=0.9,
                ),
            ),
        )


class _OCRService:
    def __init__(self):
        self.artifact = _Artifact(_model_runtime("document_ocr"))

    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(b"formal-ocr-service-fixture-v1")

    def ocr(self, path, *, asset_id, safety_approval):
        content = Path(path).read_bytes()
        line = OCRLine(
            line_id="1" * 64,
            text="Alice",
            polygon=((1.0, 1.0), (10.0, 1.0), (10.0, 5.0), (1.0, 5.0)),
            confidence=0.9,
        )
        return DocumentOCRResult(
            input_binding=OCRInputBinding(
                asset_id=asset_id,
                image_sha256=sha256_bytes(content),
                image_bytes=len(content),
                width=32,
                height=32,
                safety_decision=safety_approval.decision,
                safety_approval_sha256=safety_approval.approval_sha256,
            ),
            runtime_binding=self.artifact.runtime_binding,
            languages=("en",),
            full_text="Alice",
            lines=(line,),
            fields=(
                OCRField(
                    field_id="2" * 64,
                    field_name="name",
                    value="Alice",
                    evidence_line_ids=(line.line_id,),
                ),
            ),
            truncated=False,
        )


@dataclass
class _Artifact:
    runtime_binding: ModelRuntimeBinding


@dataclass
class _Detector:
    artifact: _Artifact

    @property
    def formal_runtime_binding_sha256(self):
        return sha256_bytes(b"formal-multi-detector-fixture-v1")


@dataclass
class _ProductSearch:
    formal_runtime_binding_sha256: str
    artifact_binding: RetrievalArtifactBinding
    execution_location: str = "remote"


class _MultiExecutor:
    def __init__(self, binding: RetrievalArtifactBinding):
        self.detector = _Detector(_Artifact(_model_runtime("object_detector")))
        self.product_search = _ProductSearch("8" * 64, binding)
        self.binding = binding
        self.on_search = None

    @property
    def execution_location(self) -> str:
        return self.product_search.execution_location

    def search(self, image_path: str | Path, *, asset_id: str) -> MultiProductResult:
        path = Path(image_path)
        if self.on_search is not None:
            self.on_search()
        return MultiProductResult(
            input_binding=DetectionInputBinding(
                asset_id=asset_id,
                image_sha256=sha256_bytes(path.read_bytes()),
                image_bytes=len(path.read_bytes()),
                width=32,
                height=32,
            ),
            detection_runtime_binding=self.detector.artifact.runtime_binding,
            artifact_binding=self.binding,
            objects=(),
        )


def _gold_row(query: Query, claim: str, tool_name: str) -> dict:
    if claim == "style_similar_search":
        arguments = {"asset_id": query.asset_id, "query": query.text}
    elif claim in {
        "image_product_search",
        "object_detection",
        "document_ocr",
        "multi_product_chain",
    }:
        arguments = {"asset_id": query.asset_id}
    elif claim == "text_product_search":
        arguments = {"query": query.text}
    else:
        arguments = (
            {"entity": "fixture entity"}
            if tool_name == "encyclopedia_lookup"
            else {"dish": "fixture dish"}
        )
    common = {
        "arguments": arguments,
        "leakage_group_id": query.leakage_group_id,
        "query_id": query.query_id,
        "schema_version": 1,
        "split": "tool_test_frozen",
        "tool_name": tool_name,
    }
    if claim == "object_detection":
        return {
            **common,
            "task": "detection",
            "expected_detections": [
                {"label": "cat", "bbox_xyxy": [1.0, 1.0, 10.0, 10.0]}
            ],
            "iou_threshold": 0.5,
        }
    if claim == "document_ocr":
        return {
            **common,
            "task": "ocr",
            "expected_text": "Alice",
            "expected_fields": [{"field_name": "name", "expected_value": "Alice"}],
        }
    if claim == "kb_lookup":
        identifier = f"entry-{query.query_id}"
        result_id_path = "citation.entry_id"
        result_list_path = None
    else:
        identifier = f"product-{query.query_id}"
        result_id_path = "product.product_id"
        result_list_path = "hits"
    return {
        **common,
        "task": "ranking",
        "relevant_ids": [identifier],
        "k": 1,
        "result_id_path": result_id_path,
        "result_list_path": result_list_path,
    }


def _write_gold_review_ledger(path: Path, gold_path: Path, rows: list[dict]) -> Path:
    ledger = {
        "schema_version": 1,
        "kind": "formal-gold-review-ledger",
        "policy_version": "formal-gold-human-review-v1",
        "gold_sha256": sha256_bytes(gold_path.read_bytes()),
        "cases": [
            {
                "query_id": row["query_id"],
                "gold_case_sha256": sha256_bytes(canonical_json_bytes(row)),
                "decision": "approved",
                "reviewer_kind": "human",
                "reviewer_id": "formal-gold-human-reviewer",
                "reviewed_at_utc": "2026-07-20T12:00:00Z",
                "blind_to_system_results": True,
                "review_basis": "Compared this gold output with immutable evidence.",
                "evidence": [
                    {
                        "evidence_id": f"evidence-{row['query_id']}",
                        "evidence_uri": f"urn:formal-gold-test:{row['query_id']}",
                        "evidence_revision": "fixture-v1",
                        "evidence_sha256": sha256_bytes(
                            f"evidence:{row['query_id']}".encode()
                        ),
                        "basis": "Fixture source evidence for this exact expected output.",
                    }
                ],
            }
            for row in sorted(rows, key=lambda item: item["query_id"])
        ],
    }
    path.write_bytes(canonical_json_bytes(ledger))
    return path


def _publish_phase4_inputs(
    root: Path,
    catalog,
    queries: tuple[Query, ...],
    *,
    name: str,
):
    queries = tuple(sorted(queries, key=lambda item: item.query_id))
    query_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in queries)
    )
    query_path = root / f"{name}-assistant-queries.jsonl"
    query_path.write_bytes(query_bytes)
    frozen = tuple(item for item in queries if item.split == "test_frozen")
    frozen_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in frozen)
    )
    split_counts = Counter(item.split for item in queries)
    intent_counts = Counter(item.canonical_intent for item in frozen)
    capability_counts = Counter(
        item.canonical_capability
        for item in frozen
        if item.canonical_capability is not None
    )
    split_manifest = FrozenSplitManifest(
        profile="formal-assistant-isolation-test",
        split_sizes={
            split: split_counts[split]
            for split in ("dev_mini", "opt_pool", "val", "test_frozen")
        },
        count=len(frozen),
        sha256=sha256_bytes(frozen_bytes),
        assignment_sha256=sha256_bytes(query_bytes),
        seed=17,
        taxonomy_version=TAXONOMY_VERSION,
        full_plan_sha256=sha256_bytes(b"formal-assistant-isolation-plan-v1"),
        asset_catalog_sha256=catalog.catalog_sha256,
        leakage_policy_version=catalog.leakage_policy_version,
        grouping_policy_version="query-connected-components-v1",
        group_fields=(
            "leakage_group_id",
            "boundary_group_id",
            "template_family",
            "generator_batch_id",
        ),
        component_count=len(build_atomic_groups(list(queries))),
        leakage_violations=0,
        intent_counts=dict(intent_counts),
        capability_counts=dict(capability_counts),
        boundary_count=sum(item.is_boundary for item in frozen),
    )
    split_bytes = canonical_json_bytes(split_manifest.model_dump(mode="json"))
    split_path = root / f"{name}-assistant-split.json"
    split_path.write_bytes(split_bytes)
    rubric_content = "Frozen formal Assistant evaluation rubric."
    rubric = RubricSnapshot(
        rubric_id="formal-assistant-rubric-v1",
        rubric_version="1",
        content=rubric_content,
        content_sha256=sha256_bytes(rubric_content.encode()),
    )
    rubric_bytes = canonical_json_bytes(rubric.model_dump(mode="json"))
    rubric_path = root / f"{name}-assistant-rubric.json"
    rubric_path.write_bytes(rubric_bytes)
    audit_query = queries[0]
    selection = make_phase4_input_selection_manifest(
        matrix_run_id=f"{name}-matrix-v1",
        query_artifact_sha256=sha256_bytes(query_bytes),
        split_manifest_sha256=sha256_bytes(split_bytes),
        final_rubric_file_sha256=sha256_bytes(rubric_bytes),
        final_rubric_content_sha256=rubric.content_sha256,
        query_ids=tuple(item.query_id for item in queries),
        judge_audit_entries=(
            JudgeAuditSelectionEntry(
                query_id=audit_query.query_id,
                config="full",
                evaluation_id=sha256_bytes(
                    f"{name}:{audit_query.query_id}:full".encode()
                ),
            ),
        ),
    )
    selection_bytes = canonical_json_bytes(selection.model_dump(mode="json"))
    selection_path = root / f"{name}-assistant-selection.json"
    selection_path.write_bytes(selection_bytes)
    handle = load_verified_phase4_inputs(
        query_path=query_path,
        expected_query_sha256=sha256_bytes(query_bytes),
        split_manifest_path=split_path,
        expected_split_manifest_sha256=sha256_bytes(split_bytes),
        rubric_path=rubric_path,
        expected_rubric_file_sha256=sha256_bytes(rubric_bytes),
        selection_path=selection_path,
        expected_selection_file_sha256=sha256_bytes(selection_bytes),
    )
    return handle, {
        "query_path": query_path,
        "split_path": split_path,
        "rubric_path": rubric_path,
        "selection_path": selection_path,
    }


@pytest.fixture
def formal_fixture(tmp_path: Path, monkeypatch, canonical_registry_factory):
    # Downstream verifier tests use deterministic in-memory doubles; production
    # calls still pass the exact built-in executor/dependency gate.
    monkeypatch.setattr(
        formal_evaluation_module,
        "_require_builtin_multi_product_executor",
        lambda _executor: None,
    )
    asset_root = tmp_path / "assets"
    image_root = asset_root / "images"
    image_root.mkdir(parents=True)
    drafts = []
    for index, (query_id, _claim, _tool, _intent) in enumerate(CLAIMS):
        name = f"{query_id}.png"
        _write_image(image_root / name, index + 1)
        drafts.append(
            DatasetAssetDraft(
                source_dataset="mep3m",
                source_revision="fixture-v1",
                source_record_id=query_id,
                local_path=f"images/{name}",
                product_id=(
                    f"product-{query_id}"
                    if query_id in {"q-image", "q-text"}
                    else f"source-product-{query_id}"
                ),
                license_id="test-only",
                cloud_upload_allowed=True,
            )
        )
    for index in (1, 2):
        source_record_id = f"assistant-public-{index}"
        name = f"{source_record_id}.png"
        _write_image(image_root / name, 100 + index)
        drafts.append(
            DatasetAssetDraft(
                source_dataset="mep3m",
                source_revision="fixture-v1",
                source_record_id=source_record_id,
                local_path=f"images/{name}",
                product_id=f"assistant-product-{index}",
                license_id="test-only",
                cloud_upload_allowed=True,
            )
        )
    product_ids = ("product-q-image", "product-q-style", "product-q-text")
    for index, product_id in enumerate(product_ids):
        name = f"gallery-{product_id}.png"
        pixels = np.random.default_rng(index + 50_000).integers(
            0,
            256,
            size=(32, 32, 3),
            dtype=np.uint8,
        )
        Image.fromarray(pixels, mode="RGB").save(image_root / name, format="PNG")
        drafts.append(
            DatasetAssetDraft(
                source_dataset="mep3m",
                source_revision="fixture-v1",
                source_record_id=product_id,
                local_path=f"images/{name}",
                product_id=product_id,
                license_id="test-only",
                cloud_upload_allowed=True,
            )
        )
    assets = tuple(inventory_dataset_asset(item, asset_root) for item in drafts)
    catalog_dir = tmp_path / "catalog"
    publish_asset_catalog(
        assets,
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(max_phash_hamming_distance=0),
        coverage_roots=["images"],
    )
    catalog = load_asset_catalog(catalog_dir, asset_root, verify_files=True)
    by_record = {item.source_record_id: item.asset_id for item in catalog.assets}
    queries = tuple(
        _query(catalog, by_record[query_id], query_id, intent)
        for query_id, _claim, _tool, intent in CLAIMS
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_bytes(
        canonical_jsonl_bytes(tuple(item.model_dump(mode="json") for item in queries))
    )
    split_path = tmp_path / "split-assignment.jsonl"
    split_path.write_bytes(
        canonical_jsonl_bytes(tuple(item.model_dump(mode="json") for item in queries))
    )
    assistant_queries = tuple(
        _query(
            catalog,
            by_record[f"assistant-public-{index}"],
            f"assistant-{split}",
            "utility",
        ).model_copy(
            update={
                "split": split,
                "template_family": f"assistant-template-{split}",
                "generator_batch_id": f"assistant-batch-{split}",
            }
        )
        for index, split in ((1, "dev_mini"), (2, "test_frozen"))
    )
    assistant_inputs, assistant_paths = _publish_phase4_inputs(
        tmp_path,
        catalog,
        assistant_queries,
        name="formal",
    )
    ocr_query = next(
        query
        for query, (_query_id, claim, _tool, _intent) in zip(
            queries, CLAIMS, strict=True
        )
        if claim == "document_ocr"
    )
    ocr_asset = catalog.resolve_asset_id(ocr_query.asset_id).asset
    approval = build_document_safety_approval(
        asset_id=ocr_asset.asset_id,
        image_sha256=ocr_asset.sha256,
        decision="approved_no_pii",
        policy_version="formal-fixture-pii-review-v1",
        reviewer_id="human-privacy-reviewer-fixture",
    )
    safety_records = (build_document_safety_record(ocr_query.query_id, approval),)
    safety_review_digest = document_safety_review_ledger_digest(safety_records)
    safety_catalog = publish_document_safety_catalog(
        safety_records,
        tmp_path / "document-safety",
        catalog,
        expected_review_ledger_sha256=safety_review_digest,
    )
    binding = _retrieval_binding(catalog, sha256_bytes(query_path.read_bytes()))
    query_by_id = {query.query_id: query for query in queries}
    canonical_runtime = canonical_registry_factory(
        name="formal-registry",
        query_artifact=query_path,
        asset_catalog=catalog,
        product_ids=product_ids,
        text_product_ids={
            query_by_id["q-text"].text: "product-q-text",
        },
        image_product_ids={
            catalog.resolve_asset_id(query_by_id[query_id].asset_id).asset.sha256: (
                f"product-{query_id}"
            )
            for query_id in ("q-image", "q-style")
        },
        kb_entries={
            "encyclopedia": (
                "entry-q-kb",
                "fixture entity",
                "fixture entity grounded evidence",
            ),
            "recipe": (
                "entry-q-recipe",
                "fixture dish",
                "fixture dish grounded evidence",
            ),
        },
        safety_catalog=safety_catalog,
    )
    registry = canonical_runtime.registry
    product_service = canonical_runtime.product_search
    gold_rows = [
        _gold_row(query, claim, tool)
        for query, (_query_id, claim, tool, _intent) in zip(
            queries, CLAIMS, strict=True
        )
    ]
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_bytes(canonical_jsonl_bytes(tuple(gold_rows)))
    gold_review_path = _write_gold_review_ledger(
        tmp_path / "gold-review.json", gold_path, gold_rows
    )
    assignment_rows = []
    for query, row, (_query_id, claim, tool, _intent) in zip(
        queries, gold_rows, CLAIMS, strict=True
    ):
        resolution = catalog.resolve_asset_id(query.asset_id)
        assignment_rows.append(
            FormalGoldAssignment(
                query_id=query.query_id,
                query_row_sha256=sha256_bytes(
                    canonical_json_bytes(query.model_dump(mode="json"))
                ),
                asset_id=query.asset_id,
                asset_sha256=resolution.asset.sha256,
                leakage_group_id=query.leakage_group_id,
                corpus_split=query.split,
                benchmark_split="tool_test_frozen",
                claim_kind=claim,
                tool_name=tool,
                arguments=row["arguments"],
                gold_case_sha256=sha256_bytes(canonical_json_bytes(row)),
                document_safety_approval_sha256=(
                    approval.approval_sha256 if claim == "document_ocr" else None
                ),
            )
        )
    assignment_path = tmp_path / "gold-assignment.jsonl"
    assignment_path.write_bytes(
        canonical_jsonl_bytes(
            tuple(item.model_dump(mode="json") for item in assignment_rows)
        )
    )
    verified_assignment = load_formal_gold_assignments(
        assignment_path,
        expected_assignment_sha256=sha256_bytes(assignment_path.read_bytes()),
        gold_path=gold_path,
        expected_gold_sha256=sha256_bytes(gold_path.read_bytes()),
        gold_review_ledger_path=gold_review_path,
        expected_gold_review_ledger_sha256=sha256_bytes(gold_review_path.read_bytes()),
        query_path=query_path,
        expected_query_sha256=sha256_bytes(query_path.read_bytes()),
        split_assignment_path=split_path,
        expected_split_assignment_sha256=sha256_bytes(split_path.read_bytes()),
        catalog=catalog,
        expected_asset_catalog_sha256=catalog.catalog_sha256,
        document_safety_catalog=safety_catalog,
        expected_document_safety_catalog_sha256=safety_catalog.catalog_sha256,
        expected_document_safety_review_ledger_sha256=safety_review_digest,
        assistant_inputs=assistant_inputs,
    )
    multi = _MultiExecutor(binding)
    return {
        "tmp": tmp_path,
        "catalog": catalog,
        "queries": queries,
        "query_path": query_path,
        "split_path": split_path,
        "gold_path": gold_path,
        "gold_review_path": gold_review_path,
        "assignment_path": assignment_path,
        "assignments": verified_assignment,
        "assistant_inputs": assistant_inputs,
        "assistant_paths": assistant_paths,
        "registry": registry,
        "canonical_runtime": canonical_runtime,
        "product_service": product_service,
        "safety_catalog": safety_catalog,
        "safety_catalog_sha": safety_catalog.catalog_sha256,
        "safety_review_sha": safety_review_digest,
        "multi": multi,
        "multi_sha": multi_product_runtime_sha256(multi),
    }


def _upgrade_fixture_to_registry_v2(value):
    canonical_runtime = value["canonical_runtime"]
    registry = build_mvp_registry(
        MVPToolServices(
            product_search=canonical_runtime.product_search,
            kb_lookup=canonical_runtime.kb_lookup,
            object_detection=canonical_runtime.object_detection,
            document_ocr=canonical_runtime.document_ocr,
            safety_approval_for=canonical_runtime.safety_catalog.approval_for,
        ),
        include_multi_product=True,
    )
    registry.require_formal_runtime()
    claims = tuple(
        (
            query_id,
            claim,
            "multi_product_search" if claim == "multi_product_chain" else tool,
            intent,
        )
        for query_id, claim, tool, intent in CLAIMS
    )
    gold_rows = [
        _gold_row(query, claim, tool)
        for query, (_query_id, claim, tool, _intent) in zip(
            value["queries"], claims, strict=True
        )
    ]
    gold_path = value["tmp"] / "gold-v2.jsonl"
    gold_path.write_bytes(canonical_jsonl_bytes(tuple(gold_rows)))
    review_path = _write_gold_review_ledger(
        value["tmp"] / "gold-review-v2.json",
        gold_path,
        gold_rows,
    )
    approval_sha256 = next(
        item.document_safety_approval_sha256
        for item in value["assignments"].records
        if item.claim_kind == "document_ocr"
    )
    assignment_rows = []
    for query, row, (_query_id, claim, tool, _intent) in zip(
        value["queries"], gold_rows, claims, strict=True
    ):
        resolution = value["catalog"].resolve_asset_id(query.asset_id)
        assignment_rows.append(
            FormalGoldAssignment(
                query_id=query.query_id,
                query_row_sha256=sha256_bytes(
                    canonical_json_bytes(query.model_dump(mode="json"))
                ),
                asset_id=query.asset_id,
                asset_sha256=resolution.asset.sha256,
                leakage_group_id=query.leakage_group_id,
                corpus_split=query.split,
                benchmark_split="tool_test_frozen",
                claim_kind=claim,
                tool_name=tool,
                arguments=row["arguments"],
                gold_case_sha256=sha256_bytes(canonical_json_bytes(row)),
                document_safety_approval_sha256=(
                    approval_sha256 if claim == "document_ocr" else None
                ),
            )
        )
    assignment_path = value["tmp"] / "gold-assignment-v2.jsonl"
    assignment_path.write_bytes(
        canonical_jsonl_bytes(
            tuple(item.model_dump(mode="json") for item in assignment_rows)
        )
    )
    assignments = load_formal_gold_assignments(
        assignment_path,
        expected_assignment_sha256=sha256_bytes(assignment_path.read_bytes()),
        gold_path=gold_path,
        expected_gold_sha256=sha256_bytes(gold_path.read_bytes()),
        gold_review_ledger_path=review_path,
        expected_gold_review_ledger_sha256=sha256_bytes(review_path.read_bytes()),
        query_path=value["query_path"],
        expected_query_sha256=sha256_bytes(value["query_path"].read_bytes()),
        split_assignment_path=value["split_path"],
        expected_split_assignment_sha256=sha256_bytes(value["split_path"].read_bytes()),
        catalog=value["catalog"],
        expected_asset_catalog_sha256=value["catalog"].catalog_sha256,
        document_safety_catalog=value["safety_catalog"],
        expected_document_safety_catalog_sha256=value["safety_catalog_sha"],
        expected_document_safety_review_ledger_sha256=value["safety_review_sha"],
        assistant_inputs=value["assistant_inputs"],
    )
    return registry, assignments


def _create_and_verify(value):
    output = value["tmp"] / "formal-run"
    created = create_formal_tool_run(
        value["assignments"],
        value["registry"],
        output,
        run_id="fixture-run-v1",
        expected_registry_sha256=value["registry"].registry_sha256,
        expected_registry_runtime_sha256=value["registry"].registry_runtime_sha256,
        multi_product_executor=value["multi"],
        expected_multi_product_runtime_sha256=value["multi_sha"],
    )
    verified = load_verified_formal_tool_run(
        output,
        value["assignments"],
        value["registry"],
        expected_run_bundle_sha256=created.run_bundle_sha256,
        expected_registry_sha256=value["registry"].registry_sha256,
        expected_registry_runtime_sha256=value["registry"].registry_runtime_sha256,
        multi_product_executor=value["multi"],
        expected_multi_product_runtime_sha256=value["multi_sha"],
    )
    return created, verified


def _reload_assignments(value, *, assistant_inputs=None):
    return load_formal_gold_assignments(
        value["assignment_path"],
        expected_assignment_sha256=sha256_bytes(value["assignment_path"].read_bytes()),
        gold_path=value["gold_path"],
        expected_gold_sha256=sha256_bytes(value["gold_path"].read_bytes()),
        gold_review_ledger_path=value["gold_review_path"],
        expected_gold_review_ledger_sha256=sha256_bytes(
            value["gold_review_path"].read_bytes()
        ),
        query_path=value["query_path"],
        expected_query_sha256=sha256_bytes(value["query_path"].read_bytes()),
        split_assignment_path=value["split_path"],
        expected_split_assignment_sha256=sha256_bytes(value["split_path"].read_bytes()),
        catalog=value["catalog"],
        expected_asset_catalog_sha256=value["catalog"].catalog_sha256,
        document_safety_catalog=value["safety_catalog"],
        expected_document_safety_catalog_sha256=value["safety_catalog_sha"],
        expected_document_safety_review_ledger_sha256=value["safety_review_sha"],
        assistant_inputs=assistant_inputs or value["assistant_inputs"],
    )


def _rewrite_record_bundle(root: Path, mutate) -> str:
    records_path = root / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    row = mutate(rows)
    if row.get("evidence") is not None:
        row["evidence_sha256"] = sha256_bytes(canonical_json_bytes(row["evidence"]))
    unsigned_record = dict(row)
    unsigned_record.pop("record_sha256")
    row["record_sha256"] = sha256_bytes(canonical_json_bytes(unsigned_record))
    content = canonical_jsonl_bytes(tuple(rows))
    records_path.write_bytes(content)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["records.jsonl"]["bytes"] = len(content)
    manifest["artifacts"]["records.jsonl"]["sha256"] = sha256_bytes(content)
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("run_bundle_sha256")
    manifest["run_bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(unsigned_manifest)
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest["run_bundle_sha256"]


def test_formal_run_covers_all_claims_and_evaluator_requires_verified_handle(
    formal_fixture,
) -> None:
    created, verified = _create_and_verify(formal_fixture)

    assert len(created.run_bundle_sha256) == 64
    assert verified.spec.gold_review_ledger_sha256 == sha256_bytes(
        formal_fixture["gold_review_path"].read_bytes()
    )
    assert (
        verified.spec.tool_assistant_isolation_sha256
        == formal_fixture["assignments"].tool_assistant_isolation.isolation_sha256
    )
    assert {item.claim_kind for item in verified.records} == {
        item[1] for item in CLAIMS
    }
    assert [(item.query_id, item.status, item.error) for item in verified.records] == [
        (item[0], "ok", None) for item in CLAIMS
    ]
    with pytest.raises(TypeError, match="load_verified_formal_tool_run"):
        require_verified_formal_tool_run(
            VerifiedFormalToolRun(
                root=verified.root,
                manifest=verified.manifest,
                spec=verified.spec,
                records=verified.records,
                assignments=verified.assignments,
                registry=verified.registry,
                multi_product_executor=verified.multi_product_executor,
                _marker=object(),
            )
        )
    with pytest.raises((TypeError, AttributeError)):
        evaluate_formal_tool_run(
            [{"hits": [{"product": {"product_id": "manual"}}]}],
            formal_fixture["tmp"] / "manual-report",
            evaluation_id="manual",
        )

    evaluated = evaluate_formal_tool_run(
        verified,
        formal_fixture["tmp"] / "formal-report",
        evaluation_id="fixture-evaluation-v1",
    )
    assert evaluated.manifest.assurance_level == "formal-verified"
    assert evaluated.manifest.case_count == len(CLAIMS)
    assert evaluated.manifest.metrics.ranking is not None
    assert evaluated.manifest.metrics.detection is not None
    assert evaluated.manifest.metrics.ocr is not None
    with pytest.raises(FileExistsError):
        evaluate_formal_tool_run(
            verified,
            formal_fixture["tmp"] / "formal-report",
            evaluation_id="second",
        )


def test_registry_v2_executes_multi_product_claim_through_composite_toolspec(
    formal_fixture,
) -> None:
    registry, assignments = _upgrade_fixture_to_registry_v2(formal_fixture)

    with pytest.raises(
        FormalEvaluationError,
        match="unused multi-product runtime",
    ):
        create_formal_tool_run(
            assignments,
            registry,
            formal_fixture["tmp"] / "formal-run-v2-double-authority",
            run_id="fixture-run-v2-double-authority",
            expected_registry_sha256=registry.registry_sha256,
            expected_registry_runtime_sha256=registry.registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )

    output = formal_fixture["tmp"] / "formal-run-v2"
    created = create_formal_tool_run(
        assignments,
        registry,
        output,
        run_id="fixture-run-v2",
        expected_registry_sha256=registry.registry_sha256,
        expected_registry_runtime_sha256=registry.registry_runtime_sha256,
    )
    verified = load_verified_formal_tool_run(
        output,
        assignments,
        registry,
        expected_run_bundle_sha256=created.run_bundle_sha256,
        expected_registry_sha256=registry.registry_sha256,
        expected_registry_runtime_sha256=registry.registry_runtime_sha256,
    )

    record = next(
        item for item in verified.records if item.claim_kind == "multi_product_chain"
    )
    composite_spec = next(
        item for item in registry.specs() if item.name == "multi_product_search"
    )
    snapshot = registry.formal_runtime_snapshot()
    assert record.status == "ok"
    assert record.tool_name == "multi_product_search"
    assert record.runtime.tool_spec_sha256 == composite_spec.spec_sha256
    assert record.runtime.tool_runtime_sha256 == snapshot.tool_runtime_sha256(
        "multi_product_search"
    )
    assert record.runtime.tool_spec_sha256 is not None
    assert verified.spec.multi_product_runtime_sha256 is None
    assert "multi_product_search" in verified.spec.tool_counts
    assert "multi_product_chain" not in verified.spec.tool_counts
    assert verified.multi_product_executor is None
    evaluated = evaluate_formal_tool_run(
        verified,
        formal_fixture["tmp"] / "formal-report-v2",
        evaluation_id="fixture-evaluation-v2",
    )
    assert evaluated.manifest.assurance_level == "formal-verified"
    assert evaluated.manifest.case_count == len(CLAIMS)


def test_registry_v2_rejects_legacy_multi_product_assignment(
    formal_fixture,
) -> None:
    registry, _assignments = _upgrade_fixture_to_registry_v2(formal_fixture)

    with pytest.raises(
        FormalEvaluationError,
        match="does not match the registry generation",
    ):
        create_formal_tool_run(
            formal_fixture["assignments"],
            registry,
            formal_fixture["tmp"] / "formal-run-v2-legacy-assignment",
            run_id="fixture-run-v2-legacy-assignment",
            expected_registry_sha256=registry.registry_sha256,
            expected_registry_runtime_sha256=registry.registry_runtime_sha256,
        )


def test_formal_loader_rejects_wrong_runtime_and_external_bundle_digest(
    formal_fixture,
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    with pytest.raises(FormalEvaluationError, match="external bundle digest"):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256="0" * 64,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )
    with pytest.raises(FormalEvaluationError, match="runtime digest mismatch"):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=created.run_bundle_sha256,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256="0" * 64,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_assignment_requires_externally_pinned_document_safety_catalog(
    formal_fixture,
) -> None:
    common = {
        "expected_assignment_sha256": sha256_bytes(
            formal_fixture["assignment_path"].read_bytes()
        ),
        "gold_path": formal_fixture["gold_path"],
        "expected_gold_sha256": sha256_bytes(formal_fixture["gold_path"].read_bytes()),
        "gold_review_ledger_path": formal_fixture["gold_review_path"],
        "expected_gold_review_ledger_sha256": sha256_bytes(
            formal_fixture["gold_review_path"].read_bytes()
        ),
        "query_path": formal_fixture["query_path"],
        "expected_query_sha256": sha256_bytes(
            formal_fixture["query_path"].read_bytes()
        ),
        "split_assignment_path": formal_fixture["split_path"],
        "expected_split_assignment_sha256": sha256_bytes(
            formal_fixture["split_path"].read_bytes()
        ),
        "catalog": formal_fixture["catalog"],
        "expected_asset_catalog_sha256": formal_fixture["catalog"].catalog_sha256,
        "assistant_inputs": formal_fixture["assistant_inputs"],
    }
    with pytest.raises(FormalEvaluationError, match="pinned safety catalog"):
        load_formal_gold_assignments(
            formal_fixture["assignment_path"],
            **common,
        )
    with pytest.raises(FormalEvaluationError, match="external digest"):
        load_formal_gold_assignments(
            formal_fixture["assignment_path"],
            **common,
            document_safety_catalog=formal_fixture["safety_catalog"],
            expected_document_safety_catalog_sha256="0" * 64,
            expected_document_safety_review_ledger_sha256=formal_fixture[
                "safety_review_sha"
            ],
        )


def test_formal_assignment_rejects_coordinated_assistant_component_overlap(
    formal_fixture,
) -> None:
    tool_query = next(
        item for item in formal_fixture["queries"] if item.query_id == "q-image"
    )
    overlapping_query = tool_query.model_copy(
        update={
            "query_id": "assistant-overlap",
            "split": "test_frozen",
            "template_family": "assistant-overlap-template",
            "generator_batch_id": "assistant-overlap-batch",
        }
    )
    coordinated, _paths = _publish_phase4_inputs(
        formal_fixture["tmp"],
        formal_fixture["catalog"],
        (overlapping_query,),
        name="coordinated-overlap",
    )
    assert (
        coordinated.expected_query_sha256
        != formal_fixture["assistant_inputs"].expected_query_sha256
    )
    assert (
        coordinated.split_manifest.assignment_sha256
        == coordinated.expected_query_sha256
    )
    assert (
        coordinated.selection.query_artifact_sha256 == coordinated.expected_query_sha256
    )
    with pytest.raises(FormalEvaluationError, match="components overlap"):
        _reload_assignments(formal_fixture, assistant_inputs=coordinated)


def test_formal_assignment_deeply_rechecks_gold_review_ledger(
    formal_fixture,
) -> None:
    ledger = json.loads(formal_fixture["gold_review_path"].read_text(encoding="utf-8"))
    ledger["cases"][0]["review_basis"] = "Tampered after typed verification."
    formal_fixture["gold_review_path"].write_bytes(canonical_json_bytes(ledger))
    with pytest.raises(FormalEvaluationError, match="review ledger|external digest"):
        create_formal_tool_run(
            formal_fixture["assignments"],
            formal_fixture["registry"],
            formal_fixture["tmp"] / "tampered-ledger-run",
            run_id="tampered-ledger-v1",
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("provisional_hit", "invalid|artifact binding"),
        (
            "other_verified_product_bundle",
            "authoritative upstream|live runtime|inherit the trace artifact binding",
        ),
        ("different_detector_runtime", "live runtime"),
        ("contradictory_ocr_decision", "safety decision"),
        ("wrong_text_input_hash", "text product evidence input"),
    ],
)
def test_formal_loader_rejects_coordinated_evidence_rebinding(
    formal_fixture, tamper: str, match: str
) -> None:
    created, _verified = _create_and_verify(formal_fixture)

    def mutate(rows):
        query_id = {
            "provisional_hit": "q-image",
            "other_verified_product_bundle": "q-image",
            "different_detector_runtime": "q-detect",
            "contradictory_ocr_decision": "q-ocr",
            "wrong_text_input_hash": "q-text",
        }[tamper]
        row = next(item for item in rows if item["query_id"] == query_id)
        if tamper == "provisional_hit":
            row["evidence"]["hits"][0]["artifact_binding"] = {
                "mode": "provisional",
                "index_integrity_sha256": None,
                "eligibility_sha256": None,
                "query_artifact_sha256": None,
                "gallery_artifact_sha256": None,
                "asset_catalog_sha256": None,
                "products_parquet_sha256": None,
                "leakage_policy_version": None,
            }
        elif tamper == "other_verified_product_bundle":
            row["evidence"]["artifact_binding"]["asset_catalog_sha256"] = "0" * 64
            row["evidence"]["hits"][0]["artifact_binding"]["asset_catalog_sha256"] = (
                "0" * 64
            )
        elif tamper == "different_detector_runtime":
            row["evidence"]["runtime_binding"]["model_id"] = "forged-detector"
        elif tamper == "contradictory_ocr_decision":
            row["evidence"]["input_binding"]["safety_decision"] = "approved_redacted"
        else:
            row["evidence"]["query_input_sha256"] = "0" * 64
        return row

    relocked = _rewrite_record_bundle(created.root, mutate)
    with pytest.raises((FormalEvaluationError, ValueError), match=match):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=relocked,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_registry_rejects_live_service_runtime_drift(formal_fixture) -> None:
    registry = formal_fixture["registry"]
    registry.require_formal_runtime()
    backend = formal_fixture["product_service"].backend
    backend._canary_vector = np.roll(backend._canary_vector, 1)  # noqa: SLF001
    with pytest.raises(
        Exception, match="live runtime binding (changed|cannot be resolved)"
    ):
        registry.require_formal_runtime()


def test_formal_multi_runtime_rejects_same_sha_different_behavior_doubles() -> None:
    binding = RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256="3" * 64,
        gallery_artifact_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        products_parquet_sha256="6" * 64,
        leakage_policy_version="fixture-policy-v1",
    )

    class _DifferentBehaviorExecutor(_MultiExecutor):
        def search(self, image_path, *, asset_id):
            raise RuntimeError("different behavior")

    first = _MultiExecutor(binding)
    second = _DifferentBehaviorExecutor(binding)
    assert (
        first.detector.formal_runtime_binding_sha256
        == second.detector.formal_runtime_binding_sha256
    )
    assert (
        first.product_search.formal_runtime_binding_sha256
        == second.product_search.formal_runtime_binding_sha256
    )
    assert type(first).search is not type(second).search

    for executor in (first, second):
        with pytest.raises(
            FormalEvaluationError, match="requires MultiProductSearchService"
        ):
            multi_product_runtime_sha256(executor)


def test_formal_multi_runtime_rejects_self_attested_dependencies() -> None:
    binding = RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256="3" * 64,
        gallery_artifact_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        products_parquet_sha256="6" * 64,
        leakage_policy_version="fixture-policy-v1",
    )

    class _SelfAttestedDetector:
        artifact = _Artifact(_model_runtime("object_detector"))
        formal_runtime_binding_sha256 = "7" * 64

        def __init__(self, outcome):
            self.outcome = outcome

        def detect(self, *_args, **_kwargs):
            return self.outcome

    class _SelfAttestedProductSearch:
        artifact_binding = binding
        formal_runtime_binding_sha256 = "8" * 64

        def __init__(self, outcome):
            self.outcome = outcome

        def trace_image_product_search(self, *_args, **_kwargs):
            return self.outcome

    first = MultiProductSearchService(
        _SelfAttestedDetector("first detector behavior"),
        _SelfAttestedProductSearch("first search behavior"),
    )
    second = MultiProductSearchService(
        _SelfAttestedDetector("second detector behavior"),
        _SelfAttestedProductSearch("second search behavior"),
    )
    assert (
        first.detector.formal_runtime_binding_sha256
        == second.detector.formal_runtime_binding_sha256
    )
    assert (
        first.product_search.formal_runtime_binding_sha256
        == second.product_search.formal_runtime_binding_sha256
    )

    for executor in (first, second):
        with pytest.raises(FormalEvaluationError, match="trusted built-in runtime"):
            multi_product_runtime_sha256(executor)


def test_formal_create_rejects_multi_runtime_change_during_search(
    formal_fixture,
) -> None:
    multi = formal_fixture["multi"]
    multi.on_search = lambda: setattr(
        multi.product_search,
        "formal_runtime_binding_sha256",
        "9" * 64,
    )

    destination = formal_fixture["tmp"] / "multi-runtime-drift-run"
    with pytest.raises(FormalEvaluationError, match="multi-product runtime"):
        create_formal_tool_run(
            formal_fixture["assignments"],
            formal_fixture["registry"],
            destination,
            run_id="multi-runtime-drift-v1",
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=multi,
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )
    assert not destination.exists()


def test_formal_loader_rechecks_multi_runtime_at_verifier_exit(
    formal_fixture,
    monkeypatch,
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    original_verify_records = formal_evaluation_module._verify_records

    def verify_then_drift(*args, **kwargs):
        result = original_verify_records(*args, **kwargs)
        formal_fixture["multi"].product_search.formal_runtime_binding_sha256 = "9" * 64
        return result

    monkeypatch.setattr(
        formal_evaluation_module,
        "_verify_records",
        verify_then_drift,
    )
    with pytest.raises(
        FormalEvaluationError,
        match="multi-product runtime changed during run verification",
    ):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=created.run_bundle_sha256,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_evaluator_reloads_nested_mutable_evidence_from_disk(
    formal_fixture,
) -> None:
    _created, verified = _create_and_verify(formal_fixture)
    image_record = next(item for item in verified.records if item.query_id == "q-image")
    assert isinstance(image_record.evidence, ProductSearchTrace)
    image_record.evidence.hits[0].product.product_id = "in-memory-forgery"

    evaluated = evaluate_formal_tool_run(
        verified,
        formal_fixture["tmp"] / "deep-refresh-report",
        evaluation_id="deep-refresh-v1",
    )
    image_result = next(
        item for item in evaluated.results if item.query_id == "q-image"
    )
    assert image_result.metrics.metric_kind == "ranking"
    assert image_result.metrics.reciprocal_rank == 1.0


def test_verified_evaluation_guard_deeply_reloads_mutable_summary_and_rows(
    formal_fixture,
) -> None:
    _created, verified = _create_and_verify(formal_fixture)
    evaluated = evaluate_formal_tool_run(
        verified,
        formal_fixture["tmp"] / "mutable-evaluation",
        evaluation_id="mutable-evaluation-v1",
    )
    image_result = next(
        item for item in evaluated.results if item.query_id == "q-image"
    )
    image_result.output["hits"][0]["product"]["product_id"] = "forged"
    evaluated.manifest.split_counts["tool_test_frozen"] = 0

    refreshed = require_verified_formal_evaluation(evaluated)
    refreshed_image = next(
        item for item in refreshed.results if item.query_id == "q-image"
    )
    assert refreshed_image.output["hits"][0]["product"]["product_id"] == (
        "product-q-image"
    )
    assert refreshed.manifest.split_counts["tool_test_frozen"] == len(CLAIMS)


def test_formal_loader_rejects_coordinated_internal_rehash(
    formal_fixture,
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    records_path = created.root / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["runtime"]["tool_runtime_sha256"] = "0" * 64
    unsigned = dict(rows[0])
    unsigned.pop("record_sha256")
    rows[0]["record_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    records_path.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    manifest_path = created.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = records_path.read_bytes()
    manifest["artifacts"]["records.jsonl"]["bytes"] = len(content)
    manifest["artifacts"]["records.jsonl"]["sha256"] = sha256_bytes(content)
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("run_bundle_sha256")
    manifest["run_bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(unsigned_manifest)
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(FormalEvaluationError, match="external bundle digest"):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=created.run_bundle_sha256,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )
    with pytest.raises(FormalEvaluationError, match="tool runtime mismatch"):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=manifest["run_bundle_sha256"],
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_loader_rejects_duplicate_and_missing_run_query_even_if_relocked(
    formal_fixture,
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    records_path = created.root / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1] = rows[0]
    records_path.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    manifest_path = created.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = records_path.read_bytes()
    manifest["artifacts"]["records.jsonl"]["bytes"] = len(content)
    manifest["artifacts"]["records.jsonl"]["sha256"] = sha256_bytes(content)
    unsigned = dict(manifest)
    unsigned.pop("run_bundle_sha256")
    manifest["run_bundle_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(
        FormalEvaluationError,
        match="sorted|duplicate|missing|extra",
    ):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=manifest["run_bundle_sha256"],
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_assignment_rejects_cross_split_group_and_unapproved_document(
    formal_fixture,
) -> None:
    rows = [
        json.loads(line)
        for line in formal_fixture["assignment_path"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[1]["leakage_group_id"] = rows[0]["leakage_group_id"]
    rows[1]["benchmark_split"] = "tool_dev"
    crossed = formal_fixture["tmp"] / "crossed.jsonl"
    crossed.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    with pytest.raises(
        Exception,
        match="assignment mismatch|query/asset mismatch|crosses tool splits",
    ):
        load_formal_gold_assignments(
            crossed,
            expected_assignment_sha256=sha256_bytes(crossed.read_bytes()),
            gold_path=formal_fixture["gold_path"],
            expected_gold_sha256=sha256_bytes(formal_fixture["gold_path"].read_bytes()),
            gold_review_ledger_path=formal_fixture["gold_review_path"],
            expected_gold_review_ledger_sha256=sha256_bytes(
                formal_fixture["gold_review_path"].read_bytes()
            ),
            query_path=formal_fixture["query_path"],
            expected_query_sha256=sha256_bytes(
                formal_fixture["query_path"].read_bytes()
            ),
            split_assignment_path=formal_fixture["split_path"],
            expected_split_assignment_sha256=sha256_bytes(
                formal_fixture["split_path"].read_bytes()
            ),
            catalog=formal_fixture["catalog"],
            expected_asset_catalog_sha256=formal_fixture["catalog"].catalog_sha256,
            assistant_inputs=formal_fixture["assistant_inputs"],
            document_safety_catalog=formal_fixture["safety_catalog"],
            expected_document_safety_catalog_sha256=formal_fixture[
                "safety_catalog_sha"
            ],
            expected_document_safety_review_ledger_sha256=formal_fixture[
                "safety_review_sha"
            ],
        )

    ocr = next(row for row in rows if row["claim_kind"] == "document_ocr")
    ocr["document_safety_approval_sha256"] = None
    unsafe = formal_fixture["tmp"] / "unsafe.jsonl"
    unsafe.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    with pytest.raises(FormalEvaluationError, match="document OCR requires"):
        load_formal_gold_assignments(
            unsafe,
            expected_assignment_sha256=sha256_bytes(unsafe.read_bytes()),
            gold_path=formal_fixture["gold_path"],
            expected_gold_sha256=sha256_bytes(formal_fixture["gold_path"].read_bytes()),
            gold_review_ledger_path=formal_fixture["gold_review_path"],
            expected_gold_review_ledger_sha256=sha256_bytes(
                formal_fixture["gold_review_path"].read_bytes()
            ),
            query_path=formal_fixture["query_path"],
            expected_query_sha256=sha256_bytes(
                formal_fixture["query_path"].read_bytes()
            ),
            split_assignment_path=formal_fixture["split_path"],
            expected_split_assignment_sha256=sha256_bytes(
                formal_fixture["split_path"].read_bytes()
            ),
            catalog=formal_fixture["catalog"],
            expected_asset_catalog_sha256=formal_fixture["catalog"].catalog_sha256,
            assistant_inputs=formal_fixture["assistant_inputs"],
            document_safety_catalog=formal_fixture["safety_catalog"],
            expected_document_safety_catalog_sha256=formal_fixture[
                "safety_catalog_sha"
            ],
            expected_document_safety_review_ledger_sha256=formal_fixture[
                "safety_review_sha"
            ],
        )


def test_formal_run_error_row_remains_in_denominator(formal_fixture) -> None:
    class InvalidExternalCodeError(RuntimeError):
        code = " bad "

    class FailingMulti(_MultiExecutor):
        def search(
            self, image_path: str | Path, *, asset_id: str
        ) -> MultiProductResult:
            raise InvalidExternalCodeError("private failure details")

    failing = FailingMulti(
        _retrieval_binding(
            formal_fixture["catalog"],
            sha256_bytes(formal_fixture["query_path"].read_bytes()),
        )
    )
    output = formal_fixture["tmp"] / "error-run"
    created = create_formal_tool_run(
        formal_fixture["assignments"],
        formal_fixture["registry"],
        output,
        run_id="error-run-v1",
        expected_registry_sha256=formal_fixture["registry"].registry_sha256,
        expected_registry_runtime_sha256=formal_fixture[
            "registry"
        ].registry_runtime_sha256,
        multi_product_executor=failing,
        expected_multi_product_runtime_sha256=multi_product_runtime_sha256(failing),
    )
    verified = load_verified_formal_tool_run(
        output,
        formal_fixture["assignments"],
        formal_fixture["registry"],
        expected_run_bundle_sha256=created.run_bundle_sha256,
        expected_registry_sha256=formal_fixture["registry"].registry_sha256,
        expected_registry_runtime_sha256=formal_fixture[
            "registry"
        ].registry_runtime_sha256,
        multi_product_executor=failing,
        expected_multi_product_runtime_sha256=multi_product_runtime_sha256(failing),
    )
    result = evaluate_formal_tool_run(
        verified,
        formal_fixture["tmp"] / "error-report",
        evaluation_id="error-eval-v1",
    )
    assert result.manifest.case_count == len(CLAIMS)
    assert result.manifest.error_count == 1
    run_error = next(item for item in verified.records if item.status == "error")
    assert run_error.error is not None
    assert run_error.error.code == "tool_error"
    error_result = next(item for item in result.results if item.status == "error")
    assert isinstance(error_result, BenchmarkCaseResult)
    assert error_result.error is not None
    assert error_result.error.message == "formal tool execution failed"


def test_formal_evaluation_report_is_create_only_and_bundle_rejects_extra_file(
    formal_fixture,
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    (created.root / "cache.tmp").write_text("mutable cache", encoding="utf-8")
    with pytest.raises(FormalEvaluationError, match="file set"):
        load_verified_formal_tool_run(
            created.root,
            formal_fixture["assignments"],
            formal_fixture["registry"],
            expected_run_bundle_sha256=created.run_bundle_sha256,
            expected_registry_sha256=formal_fixture["registry"].registry_sha256,
            expected_registry_runtime_sha256=formal_fixture[
                "registry"
            ].registry_runtime_sha256,
            multi_product_executor=formal_fixture["multi"],
            expected_multi_product_runtime_sha256=formal_fixture["multi_sha"],
        )


def test_formal_assignment_requires_exact_query_coverage(formal_fixture) -> None:
    lines = formal_fixture["assignment_path"].read_bytes().splitlines(keepends=True)
    incomplete = formal_fixture["tmp"] / "incomplete.jsonl"
    incomplete.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(Exception, match="query set mismatch|exactly equal"):
        load_formal_gold_assignments(
            incomplete,
            expected_assignment_sha256=sha256_bytes(incomplete.read_bytes()),
            gold_path=formal_fixture["gold_path"],
            expected_gold_sha256=sha256_bytes(formal_fixture["gold_path"].read_bytes()),
            gold_review_ledger_path=formal_fixture["gold_review_path"],
            expected_gold_review_ledger_sha256=sha256_bytes(
                formal_fixture["gold_review_path"].read_bytes()
            ),
            query_path=formal_fixture["query_path"],
            expected_query_sha256=sha256_bytes(
                formal_fixture["query_path"].read_bytes()
            ),
            split_assignment_path=formal_fixture["split_path"],
            expected_split_assignment_sha256=sha256_bytes(
                formal_fixture["split_path"].read_bytes()
            ),
            catalog=formal_fixture["catalog"],
            expected_asset_catalog_sha256=formal_fixture["catalog"].catalog_sha256,
            assistant_inputs=formal_fixture["assistant_inputs"],
            document_safety_catalog=formal_fixture["safety_catalog"],
            expected_document_safety_catalog_sha256=formal_fixture[
                "safety_catalog_sha"
            ],
            expected_document_safety_review_ledger_sha256=formal_fixture[
                "safety_review_sha"
            ],
        )


def test_dedicated_formal_cli_verifies_with_external_digests(
    formal_fixture, capsys
) -> None:
    created, _verified = _create_and_verify(formal_fixture)
    args = [
        "verify-run",
        "--assignment",
        str(formal_fixture["assignment_path"]),
        "--expected-assignment-sha256",
        sha256_bytes(formal_fixture["assignment_path"].read_bytes()),
        "--gold",
        str(formal_fixture["gold_path"]),
        "--expected-gold-sha256",
        sha256_bytes(formal_fixture["gold_path"].read_bytes()),
        "--gold-review-ledger",
        str(formal_fixture["gold_review_path"]),
        "--expected-gold-review-ledger-sha256",
        sha256_bytes(formal_fixture["gold_review_path"].read_bytes()),
        "--queries",
        str(formal_fixture["query_path"]),
        "--expected-query-sha256",
        sha256_bytes(formal_fixture["query_path"].read_bytes()),
        "--split-assignment",
        str(formal_fixture["split_path"]),
        "--expected-split-assignment-sha256",
        sha256_bytes(formal_fixture["split_path"].read_bytes()),
        "--expected-asset-catalog-sha256",
        formal_fixture["catalog"].catalog_sha256,
        "--expected-document-safety-catalog-sha256",
        formal_fixture["safety_catalog_sha"],
        "--expected-document-safety-review-ledger-sha256",
        formal_fixture["safety_review_sha"],
        "--assistant-queries",
        str(formal_fixture["assistant_paths"]["query_path"]),
        "--expected-assistant-query-sha256",
        formal_fixture["assistant_inputs"].expected_query_sha256,
        "--assistant-split-manifest",
        str(formal_fixture["assistant_paths"]["split_path"]),
        "--expected-assistant-split-manifest-sha256",
        formal_fixture["assistant_inputs"].expected_split_manifest_sha256,
        "--assistant-rubric",
        str(formal_fixture["assistant_paths"]["rubric_path"]),
        "--expected-assistant-rubric-sha256",
        formal_fixture["assistant_inputs"].expected_rubric_file_sha256,
        "--assistant-selection",
        str(formal_fixture["assistant_paths"]["selection_path"]),
        "--expected-assistant-selection-sha256",
        formal_fixture["assistant_inputs"].expected_selection_file_sha256,
        "--expected-registry-sha256",
        formal_fixture["registry"].registry_sha256,
        "--expected-registry-runtime-sha256",
        formal_fixture["registry"].registry_runtime_sha256,
        "--expected-multi-product-runtime-sha256",
        formal_fixture["multi_sha"],
        "--run-dir",
        str(created.root),
        "--expected-run-bundle-sha256",
        created.run_bundle_sha256,
    ]
    assert (
        cli_main(
            args,
            runtime_context=FormalRuntimeContext(
                catalog=formal_fixture["catalog"],
                registry=formal_fixture["registry"],
                document_safety_catalog=formal_fixture["safety_catalog"],
                multi_product_executor=formal_fixture["multi"],
            ),
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "verified-formal-tool-run"
    assert output["query_count"] == len(CLAIMS)


def test_dedicated_formal_cli_v2_create_verify_and_evaluate_use_registry_only(
    formal_fixture,
    capsys,
) -> None:
    registry, _assignments = _upgrade_fixture_to_registry_v2(formal_fixture)
    assignment_path = formal_fixture["tmp"] / "gold-assignment-v2.jsonl"
    gold_path = formal_fixture["tmp"] / "gold-v2.jsonl"
    gold_review_path = formal_fixture["tmp"] / "gold-review-v2.json"
    common = [
        "--assignment",
        str(assignment_path),
        "--expected-assignment-sha256",
        sha256_bytes(assignment_path.read_bytes()),
        "--gold",
        str(gold_path),
        "--expected-gold-sha256",
        sha256_bytes(gold_path.read_bytes()),
        "--gold-review-ledger",
        str(gold_review_path),
        "--expected-gold-review-ledger-sha256",
        sha256_bytes(gold_review_path.read_bytes()),
        "--queries",
        str(formal_fixture["query_path"]),
        "--expected-query-sha256",
        sha256_bytes(formal_fixture["query_path"].read_bytes()),
        "--split-assignment",
        str(formal_fixture["split_path"]),
        "--expected-split-assignment-sha256",
        sha256_bytes(formal_fixture["split_path"].read_bytes()),
        "--expected-asset-catalog-sha256",
        formal_fixture["catalog"].catalog_sha256,
        "--expected-document-safety-catalog-sha256",
        formal_fixture["safety_catalog_sha"],
        "--expected-document-safety-review-ledger-sha256",
        formal_fixture["safety_review_sha"],
        "--assistant-queries",
        str(formal_fixture["assistant_paths"]["query_path"]),
        "--expected-assistant-query-sha256",
        formal_fixture["assistant_inputs"].expected_query_sha256,
        "--assistant-split-manifest",
        str(formal_fixture["assistant_paths"]["split_path"]),
        "--expected-assistant-split-manifest-sha256",
        formal_fixture["assistant_inputs"].expected_split_manifest_sha256,
        "--assistant-rubric",
        str(formal_fixture["assistant_paths"]["rubric_path"]),
        "--expected-assistant-rubric-sha256",
        formal_fixture["assistant_inputs"].expected_rubric_file_sha256,
        "--assistant-selection",
        str(formal_fixture["assistant_paths"]["selection_path"]),
        "--expected-assistant-selection-sha256",
        formal_fixture["assistant_inputs"].expected_selection_file_sha256,
        "--expected-registry-sha256",
        registry.registry_sha256,
        "--expected-registry-runtime-sha256",
        registry.registry_runtime_sha256,
    ]
    context = FormalRuntimeContext(
        catalog=formal_fixture["catalog"],
        registry=registry,
        document_safety_catalog=formal_fixture["safety_catalog"],
        multi_product_executor=None,
    )
    run_root = formal_fixture["tmp"] / "formal-cli-run-v2"

    assert (
        cli_main(
            [
                "create-run",
                *common,
                "--destination",
                str(run_root),
                "--run-id",
                "formal-cli-run-v2",
            ],
            runtime_context=context,
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["kind"] == "created-formal-tool-run"

    run_arguments = [
        "--run-dir",
        str(run_root),
        "--expected-run-bundle-sha256",
        created["run_bundle_sha256"],
    ]
    assert (
        cli_main(
            ["verify-run", *common, *run_arguments],
            runtime_context=context,
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["kind"] == "verified-formal-tool-run"
    assert verified["query_count"] == len(CLAIMS)

    evaluation_root = formal_fixture["tmp"] / "formal-cli-evaluation-v2"
    assert (
        cli_main(
            [
                "evaluate-run",
                *common,
                *run_arguments,
                "--destination",
                str(evaluation_root),
                "--evaluation-id",
                "formal-cli-evaluation-v2",
            ],
            runtime_context=context,
        )
        == 0
    )
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["kind"] == "verified-formal-tool-evaluation"
    assert evaluated["case_count"] == len(CLAIMS)
