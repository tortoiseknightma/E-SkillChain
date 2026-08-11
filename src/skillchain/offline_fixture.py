"""Build a tiny, deterministic, network-denied engineering fixture.

The fixture exercises repository contracts end to end, but it is deliberately
not an experiment result.  It uses six synthetic queries, generated images,
diagnostic tool doubles, and a deterministic Assistant backend.  Every
eligibility-bearing artifact remains formally ineligible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import socket
import stat
import sys

from PIL import Image

from skillchain import config
from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    NearDuplicatePolicy,
    QueryAssetReference,
    assert_query_gallery_eligible,
    audit_query_gallery_eligibility,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.evaluation.assistant_runs import (
    MAIN_CONFIG_ORDER,
    AssistantBackendResponse,
    JudgeAuditSelectionEntry,
    build_assistant_matrix_plan,
    create_assistant_matrix_plan_file,
    create_assistant_run_bundle,
    load_verified_assistant_matrix_plan,
    load_verified_five_config_runs,
    load_verified_phase4_inputs,
    make_backbone_lock,
    make_inference_budget,
    make_phase4_input_selection_manifest,
)
from skillchain.evaluation.evaluator_isolation import (
    build_bound_feedback_prompt,
    build_bound_final_prompt,
    make_evaluator_isolation_lock,
    make_feedback_evaluator_identity,
    make_final_evaluator_identity,
    serialize_feedback_packet,
    serialize_final_evaluation_packet,
)
from skillchain.evaluation.packets import (
    AssistantToolTrace,
    EvaluationImage,
    FeedbackPacket,
    FinalEvaluationPacket,
    RubricSnapshot,
    derive_blinded_evaluation_id,
)
from skillchain.llm import LLMUsage
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.static_authoring import (
    AuthoringBudgets,
    FixedDecoding,
    ModelIdentity,
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    PriceSchedule,
    PromptIdentity,
    build_authoring_packet,
    build_public_source_material,
    run_spec_baseline,
)
from skillchain.synthesis.planning import PHASE3_TASK_SPEC_VERSION
from skillchain.synthesis.splitting import (
    SplitSpec,
    assert_no_group_leakage,
    freeze_test_split,
    group_leakage_report,
    stratified_split,
    verify_frozen_split,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import (
    TAXONOMY_VERSION,
    capabilities_for_intent,
    load_default_taxonomy_registry,
)
from skillchain.tools.contracts import (
    ProductSearchTrace,
    RetrievalArtifactBinding,
)
from skillchain.tools.document_ocr import (
    DocumentOCRResult,
    DocumentSafetyApproval,
    OCRInputBinding,
    document_safety_approval_digest,
)
from skillchain.tools.kb_lookup import KBRetrievalArtifactBinding
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.object_detect import (
    DetectionInputBinding,
    ObjectDetectionResult,
)
from skillchain.tools.registry import (
    DiagnosticToolRegistry,
    MVP_TOOL_NAMES_V2,
    MVPToolServices,
    ToolExecutionContext,
    ToolRegistry,
    build_mvp_registry,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

FIXTURE_POLICY_VERSION = "offline-engineering-fixture-v2"
FIXTURE_FORMAL_INELIGIBILITY_REASON = (
    "synthetic-diagnostic-fixture-not-a-real-experiment"
)
DEFAULT_FIXTURE_SPEC = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "fixtures"
    / "offline-engineering-fixture-v2.json"
)


@dataclass(frozen=True)
class CreatedOfflineFixture:
    root: Path
    manifest_file_sha256: str
    artifact_tree_sha256: str


def _hash_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(value)))


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_fixture_spec(path: Path) -> tuple[dict[str, object], bytes]:
    content = read_stable_regular_file(path, label="offline fixture specification")
    raw = parse_canonical_json(content, label="offline fixture specification")
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise ValueError("offline fixture specification must be a canonical object")
    expected = {
        "fixture_id": FIXTURE_POLICY_VERSION,
        "formal_eligible": False,
        "network_policy": "python-socket-audit-guard",
        "query_count": 6,
        "schema_version": 1,
        "seed": 20260721,
        "split_sizes": {
            "dev_mini": 2,
            "opt_pool": 2,
            "test_frozen": 1,
            "val": 1,
        },
    }
    if raw != expected:
        raise ValueError("offline fixture specification differs from policy v2")
    return raw, content


def _install_network_guard() -> None:
    """Fail the process if any dependency attempts DNS or a socket connection."""

    blocked_events = {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
    }

    def audit(event: str, _arguments: tuple[object, ...]) -> None:
        if event in blocked_events:
            raise RuntimeError(
                f"network access is forbidden in offline fixture: {event}"
            )

    sys.addaudithook(audit)

    def denied(*_args, **_kwargs):
        raise RuntimeError("network access is forbidden in offline fixture")

    socket.create_connection = denied  # type: ignore[assignment]


def _remove_staging_tree(path: Path) -> None:
    """Remove generated staging bytes, including read-only frozen-split files."""

    def unlock_and_retry(function, target, _error):
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onerror=unlock_and_retry)


def _write_image(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    pixels = bytes(rng.randrange(256) for _ in range(32 * 32 * 3))
    image = Image.frombytes("RGB", (32, 32), pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _build_catalog(root: Path, seed: int):
    asset_root = root / "assets"
    drafts: list[DatasetAssetDraft] = []
    for index in range(1, 8):
        local_path = f"images/asset-{index:02d}.png"
        _write_image(asset_root / local_path, seed + index)
        product_id = "locked-product" if index in {1, 2, 7} else f"product-{index}"
        drafts.append(
            DatasetAssetDraft(
                source_dataset="offline-fixture",
                source_revision="offline-fixture-revision-v1",
                source_record_id=f"record-{index:02d}",
                local_path=local_path,
                product_id=product_id,
                license_id="synthetic-test-only",
                source_url=None,
                attribution="Deterministically generated diagnostic pixels",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
        )
    assets = tuple(inventory_dataset_asset(item, asset_root) for item in drafts)
    publish_asset_catalog(
        assets,
        root / "asset-catalog",
        asset_root,
        NearDuplicatePolicy(
            policy_version="offline-fixture-phash-exact-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=("images",),
    )
    catalog = load_asset_catalog(root / "asset-catalog", asset_root)
    return catalog, {index: assets[index - 1] for index in range(1, 8)}


def _make_query(
    *,
    index: int,
    intent: str,
    asset,
    leakage_group_id: str,
    split: str,
    template_family: str,
) -> Query:
    capability_spec = capabilities_for_intent(intent)[0]
    capability = capability_spec.capability_id
    text = f"Offline diagnostic query {index:02d}"
    decision = LabelDecision(
        decision_type="constructed",
        annotator_kind="planner",
        annotator_id="offline-deterministic-constructor",
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        reason="Deterministic engineering fixture; no human-label claim",
    )
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        query_id=f"offline-q-{index:02d}",
        asset_id=asset.asset_id,
        image_path=asset.local_path,
        leakage_group_id=leakage_group_id,
        template_family=template_family,
        generator_batch_id=f"offline-generator-{index:02d}",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        boundary_group_id=None,
        requires_card=capability_spec.requires_card,
        split=split,
        label_status="auto",
        label_provenance=[decision],
    )


def _build_and_freeze_queries(
    root: Path,
    *,
    catalog,
    assets: dict[int, object],
    seed: int,
    fixture_spec_sha256: str,
):
    intents = (
        "exact_match",
        "exact_match",
        "multi_product",
        "multi_product",
        "encyclopedia",
        "utility",
    )
    queries: list[Query] = []
    for index, intent in enumerate(intents, start=1):
        asset = assets[index]
        queries.append(
            _make_query(
                index=index,
                intent=intent,
                asset=asset,
                leakage_group_id=catalog.component_for_asset(asset.asset_id),
                split="dev_mini" if index <= 2 else "opt_pool",
                template_family=(
                    "offline-paired-template"
                    if index in {3, 4}
                    else f"template-{index}"
                ),
            )
        )
    split_spec = SplitSpec(
        profile="offline-engineering-fixture",
        sizes={
            "dev_mini": 2,
            "opt_pool": 2,
            "val": 1,
            "test_frozen": 1,
        },
        min_test_per_intent=0,
    )
    assigned = stratified_split(
        queries,
        locked_dev_query_ids={"offline-q-01", "offline-q-02"},
        seed=seed,
        split_spec=split_spec,
    )
    assert all(query.schema_version == 2 for query in assigned)
    assert Counter(query.split for query in assigned) == Counter(split_spec.sizes)
    split_by_query = {query.query_id: query.split for query in assigned}
    if (
        split_by_query["offline-q-01"] != "dev_mini"
        or split_by_query["offline-q-02"] != "dev_mini"
        or split_by_query["offline-q-03"] != split_by_query["offline-q-04"]
    ):
        raise AssertionError("catalog/template atomic query groups were split")
    assert_no_group_leakage(assigned)
    split_root = root / "queries"
    _, manifest_path = freeze_test_split(
        split_root,
        assigned,
        seed=seed,
        full_plan_sha256=fixture_spec_sha256,
        asset_catalog=catalog,
        split_spec=split_spec,
    )
    verified_split = verify_frozen_split(
        split_root,
        asset_catalog=catalog,
        expected_full_plan_sha256=fixture_spec_sha256,
    )
    if verified_split is None:
        raise AssertionError("frozen split verifier returned no manifest")
    group_report = group_leakage_report(assigned)
    atomic_create_file(
        root / "gates" / "group-leakage-report.json",
        canonical_json_bytes(group_report),
    )
    gallery_report = audit_query_gallery_eligibility(
        (
            QueryAssetReference(
                query_id=assigned[0].query_id,
                intent="exact_match",
                asset_id=assets[1].asset_id,
            ),
        ),
        (assets[7].asset_id,),
        catalog,
    )
    assert_query_gallery_eligible(gallery_report)
    atomic_create_file(
        root / "gates" / "query-gallery-report.json",
        canonical_json_bytes(gallery_report.model_dump(mode="json")),
    )
    return assigned, verified_split, manifest_path


def _runtime_digest(label: str) -> str:
    return _hash_payload({"diagnostic_runtime": label, "policy_version": 1})


class _DiagnosticProductService:
    execution_location = "local"

    def __init__(self) -> None:
        self.artifact_binding = RetrievalArtifactBinding(mode="provisional")

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return _runtime_digest("product-service")

    def _image_trace(self, tool_name: str, image_path: Path, asset_id: str):
        digest = sha256_bytes(Path(image_path).read_bytes())
        return ProductSearchTrace(
            tool_name=tool_name,
            input_kind="image",
            query_input_sha256=digest,
            query_image_sha256=digest,
            query_asset_id=asset_id,
            query_vector_sha256=_runtime_digest(f"vector:{tool_name}:{digest}"),
            artifact_binding=self.artifact_binding,
            hits=(),
        )

    def trace_image_product_search(self, image_path: Path, *, asset_id: str):
        return self._image_trace("image_product_search", image_path, asset_id)

    def trace_similar_styles(self, image_path: Path, *, asset_id: str):
        return self._image_trace("style_similar_search", image_path, asset_id)

    def trace_text_product_search(self, query: str):
        digest = sha256_bytes(query.encode("utf-8"))
        return ProductSearchTrace(
            tool_name="text_product_search",
            input_kind="text",
            query_input_sha256=digest,
            query_text=query,
            query_vector_sha256=_runtime_digest(f"vector:text:{digest}"),
            artifact_binding=self.artifact_binding,
            hits=(),
        )


class _DiagnosticKBService:
    @property
    def formal_runtime_binding_sha256(self) -> str:
        return _runtime_digest("kb-service")

    def artifact_binding_for(self, kind: str):
        return KBRetrievalArtifactBinding(mode="provisional", kind=kind)

    def encyclopedia_lookup(self, _entity: str):
        return []

    def recipe_lookup(self, _dish: str):
        return []


@dataclass(frozen=True)
class _DiagnosticArtifact:
    runtime_binding: ModelRuntimeBinding


class _DiagnosticDetectionService:
    def __init__(self) -> None:
        self.artifact = _DiagnosticArtifact(
            ModelRuntimeBinding(
                artifact_kind="object_detector",
                model_id="offline-diagnostic-detector",
                backend_name="typed-diagnostic-double",
                backend_version="1",
                manifest_sha256=_runtime_digest("detector-artifact-manifest"),
                artifacts=(),
            )
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return _runtime_digest("detector-service")

    def detect(self, image_path: Path, *, asset_id: str):
        content = Path(image_path).read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
        return ObjectDetectionResult(
            input_binding=DetectionInputBinding(
                asset_id=asset_id,
                image_sha256=sha256_bytes(content),
                image_bytes=len(content),
                width=width,
                height=height,
            ),
            runtime_binding=self.artifact.runtime_binding,
            detections=(),
        )


class _DiagnosticOCRService:
    def __init__(self) -> None:
        self.artifact = _DiagnosticArtifact(
            ModelRuntimeBinding(
                artifact_kind="document_ocr",
                model_id="offline-diagnostic-ocr",
                backend_name="typed-diagnostic-double",
                backend_version="1",
                manifest_sha256=_runtime_digest("ocr-artifact-manifest"),
                artifacts=(),
            )
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return _runtime_digest("ocr-service")

    def ocr(
        self,
        image_path: Path,
        *,
        asset_id: str,
        safety_approval: DocumentSafetyApproval,
    ):
        content = Path(image_path).read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
        if safety_approval.image_sha256 != sha256_bytes(content):
            raise ValueError("diagnostic OCR approval/image mismatch")
        return DocumentOCRResult(
            input_binding=OCRInputBinding(
                asset_id=asset_id,
                image_sha256=sha256_bytes(content),
                image_bytes=len(content),
                width=width,
                height=height,
                safety_decision=safety_approval.decision,
                safety_approval_sha256=safety_approval.approval_sha256,
            ),
            runtime_binding=self.artifact.runtime_binding,
            languages=("en",),
            full_text="",
            lines=(),
            fields=(),
            truncated=False,
        )


class _DiagnosticSafetyResolver:
    def __init__(self, approvals: dict[str, DocumentSafetyApproval]) -> None:
        self._approvals = approvals

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return _runtime_digest("document-safety-resolver")

    def __call__(self, _query_id: str, asset_id: str):
        return self._approvals[asset_id]


def _approval(asset_id: str, image_path: Path) -> DocumentSafetyApproval:
    unsigned = {
        "schema_version": 1,
        "asset_id": asset_id,
        "image_sha256": sha256_bytes(image_path.read_bytes()),
        "decision": "approved_no_pii",
        "policy_version": "offline-diagnostic-safety-v1",
        "reviewer_id": "deterministic-diagnostic-double",
        "redaction_parent_sha256": None,
    }
    return DocumentSafetyApproval(
        **unsigned,
        approval_sha256=document_safety_approval_digest(unsigned),
    )


def _build_diagnostic_registry(
    root: Path, catalog, assigned: list[Query]
) -> ToolRegistry:
    approvals = {
        query.asset_id: _approval(query.asset_id, catalog.asset_root / query.image_path)
        for query in assigned
    }
    registry = build_mvp_registry(
        MVPToolServices(
            product_search=_DiagnosticProductService(),
            kb_lookup=_DiagnosticKBService(),
            object_detection=_DiagnosticDetectionService(),
            document_ocr=_DiagnosticOCRService(),
            safety_approval_for=_DiagnosticSafetyResolver(approvals),
        ),
        include_multi_product=True,
    )
    if type(registry) is not DiagnosticToolRegistry:
        raise AssertionError("offline fixture must use the typed diagnostic registry")
    if registry.formal_runtime_ready:
        raise AssertionError("offline fixture registry must remain formally ineligible")
    atomic_create_file(
        root / "tool-diagnostics" / "registry-manifest.json",
        canonical_json_bytes(registry.manifest.model_dump(mode="json")),
    )
    return registry


def _exercise_all_tools(
    root: Path,
    *,
    registry: ToolRegistry,
    catalog,
    query: Query,
) -> None:
    context = ToolExecutionContext(
        query_id=query.query_id,
        query_asset_id=query.asset_id,
        query_text=query.text,
        resolve_asset=lambda asset_id: (
            catalog.asset_root / catalog.resolve_asset_id(asset_id).local_path
        ),
    )
    arguments = {
        "image_product_search": {"asset_id": query.asset_id},
        "multi_product_search": {"asset_id": query.asset_id},
        "text_product_search": {"query": query.text},
        "style_similar_search": {
            "asset_id": query.asset_id,
            "query": query.text,
        },
        "encyclopedia_lookup": {"entity": "offline fixture entity"},
        "recipe_lookup": {"dish": "offline fixture dish"},
        "object_detect": {"asset_id": query.asset_id},
        "document_ocr": {"asset_id": query.asset_id},
    }
    if set(arguments) != MVP_TOOL_NAMES_V2:
        raise AssertionError("diagnostic calls must cover all eight tools")
    for name in sorted(arguments):
        result = registry.invoke(name, arguments[name], context)
        atomic_create_file(
            root / "tool-diagnostics" / f"{name}.json", result.output_bytes
        )
    atomic_create_file(
        root / "tool-diagnostics" / "coverage.json",
        canonical_json_bytes(
            {
                "diagnostic_doubles": True,
                "formal_eligible": False,
                "invoked_tools": sorted(arguments),
                "tool_count": len(arguments),
            }
        ),
    )


def _build_spec_baseline(root: Path, registry: ToolRegistry):
    prompt_text = "Return canonical structured Skills from these bytes only."
    source = build_public_source_material(
        source_id="offline-diagnostic-material",
        url="https://example.invalid/offline-material/revision-v1",
        revision="revision-v1",
        media_type="text/plain",
        content=b"Offline deterministic diagnostic material version one.",
        license_id="synthetic-test-only",
        license_evidence_url="https://example.invalid/license/synthetic-v1",
        license_evidence=b"Synthetic bytes generated only for an engineering fixture.",
    )
    authoring_input = build_authoring_packet(
        taxonomy=load_default_taxonomy_registry(),
        task_specification=load_mvp_task_specification_v1(),
        tool_registry=registry,
        prompt=PromptIdentity(
            prompt_id="offline-spec-baseline",
            prompt_version="1.0.0",
            template=prompt_text,
            prompt_sha256=sha256_bytes(prompt_text.encode("utf-8")),
        ),
        model=ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model="offline-never-invoked",
            revision="invoked",
        ),
        decoding=FixedDecoding(
            seed=20260721,
            max_output_tokens=128,
            response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
        ),
        price_schedule=PriceSchedule(
            schedule_id="offline-zero-cost-v1",
            provider="qwen",
            model="offline-never-invoked",
            input_microusd_per_million_tokens=0,
            output_microusd_per_million_tokens=0,
            source_input_microunits_per_million_tokens=0,
            source_output_microunits_per_million_tokens=0,
            conversion_microusd_per_source_unit=1_000_000,
            tier_max_input_tokens=32_000,
            source_url="https://example.invalid/pricing/revision-v1",
            source_revision="revision-v1",
        ),
        budgets=AuthoringBudgets(
            max_input_tokens=64,
            max_output_tokens=128,
            max_total_tokens=192,
            max_cost_microusd=0,
            max_human_review_minutes=0,
        ),
        public_sources=(source,),
    )
    result = run_spec_baseline(
        authoring_input,
        root / "spec-baseline",
        require_formal_eligibility=False,
    )
    if result.manifest.formal_eligible:
        raise AssertionError("offline SpecBaseline must remain formally ineligible")
    return result


class _DeterministicAssistantBackend:
    def execute(self, request):
        skilled = request.config != "noskill"
        runtime_by_tool = {
            item.tool_name: item.runtime_binding_sha256
            for item in request.registry.tools
        }
        trace = (
            AssistantToolTrace(
                call_index=1,
                tool_name="text_product_search",
                status="success",
                arguments_sha256=_runtime_digest(
                    f"assistant-arguments:{request.query.query_id}"
                ),
                result_sha256=_runtime_digest(
                    f"assistant-result:{request.query.query_id}"
                ),
                runtime_binding_sha256=runtime_by_tool["text_product_search"],
                latency_ms=1,
            ),
        )
        return AssistantBackendResponse(
            request_sha256=request.request_sha256,
            backbone_provider=request.backbone.provider,
            backbone_model=request.backbone.model,
            backbone_endpoint=request.backbone.endpoint,
            backbone_identity_sha256=request.backbone.identity_sha256,
            registry_sha256=request.registry.registry_sha256,
            registry_runtime_sha256=request.registry.registry_runtime_sha256,
            budget_sha256=request.budget.budget_sha256,
            response_text=f"Offline diagnostic answer for {request.query.query_id}.",
            tool_trace=trace,
            selected_capability="utility.document_reading" if skilled else None,
            skill_slug="offline-diagnostic-skill" if skilled else None,
            route_trace_sha256=(
                _runtime_digest(f"route:{request.config}:{request.query.query_id}")
                if skilled
                else None
            ),
            backbone_request_id=(
                f"offline-diagnostic-{request.query_ordinal}-{request.config}"
            ),
            usage=LLMUsage(input_tokens=12, output_tokens=8),
            turn_count=1,
            latency_ms=2,
        )


def _build_five_assistant_runs(
    root: Path,
    *,
    registry: ToolRegistry,
    assigned: list[Query],
    spec_bank_sha256: str,
):
    selected = [
        next(query for query in assigned if query.query_id == "offline-q-05"),
        next(query for query in assigned if query.query_id == "offline-q-06"),
    ]
    assignment_path = root / "queries" / "split_assignment.jsonl"
    query_artifact_sha256 = sha256_bytes(assignment_path.read_bytes())
    split_manifest_path = root / "queries" / "test_frozen.manifest.json"
    split_manifest_sha256 = sha256_bytes(split_manifest_path.read_bytes())
    rubric_content = "Score visible task completion and answer quality only."
    rubric = RubricSnapshot(
        rubric_id="offline-diagnostic-rubric",
        rubric_version="1",
        content=rubric_content,
        content_sha256=sha256_bytes(rubric_content.encode("utf-8")),
    )
    rubric_bytes = canonical_json_bytes(rubric.model_dump(mode="json"))
    rubric_path = atomic_create_file(root / "evaluation" / "rubric.json", rubric_bytes)
    matrix_run_id = "offline-diagnostic-matrix-v1"
    evaluation_id = derive_blinded_evaluation_id(
        blinding_key=b"offline-diagnostic-blinding-key-v1",
        run_id=matrix_run_id,
        query_id=selected[0].query_id,
        config="full",
    )
    selection = make_phase4_input_selection_manifest(
        matrix_run_id=matrix_run_id,
        query_artifact_sha256=query_artifact_sha256,
        split_manifest_sha256=split_manifest_sha256,
        final_rubric_file_sha256=sha256_bytes(rubric_bytes),
        final_rubric_content_sha256=rubric.content_sha256,
        query_ids=tuple(query.query_id for query in selected),
        judge_audit_entries=(
            JudgeAuditSelectionEntry(
                query_id=selected[0].query_id,
                config="full",
                evaluation_id=evaluation_id,
            ),
        ),
    )
    selection_bytes = canonical_json_bytes(selection.model_dump(mode="json"))
    selection_path = atomic_create_file(
        root / "evaluation" / "judge-audit-selection.json", selection_bytes
    )
    inputs = load_verified_phase4_inputs(
        query_path=assignment_path,
        expected_query_sha256=query_artifact_sha256,
        split_manifest_path=split_manifest_path,
        expected_split_manifest_sha256=split_manifest_sha256,
        rubric_path=rubric_path,
        expected_rubric_file_sha256=sha256_bytes(rubric_bytes),
        selection_path=selection_path,
        expected_selection_file_sha256=sha256_bytes(selection_bytes),
    )
    backbone = make_backbone_lock(
        provider="offline-diagnostic",
        model="deterministic-backend-v1",
        endpoint="https://example.invalid/offline-only",
        temperature=0.0,
        top_p=1.0,
        seed=20260721,
        system_prompt_sha256=_runtime_digest("assistant-system-prompt"),
    )
    budget = make_inference_budget(
        max_input_tokens=64,
        max_output_tokens=32,
        max_tool_calls=1,
        max_turns=1,
        timeout_ms=1000,
    )
    bank_hashes = {
        name: _runtime_digest(f"diagnostic-bank:{name}:{spec_bank_sha256}")
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    plan = build_assistant_matrix_plan(
        inputs=inputs,
        backbone=backbone,
        budget=budget,
        registry=registry,
        bank_sha256_by_config=bank_hashes,
        spec_baseline_bank_sha256=spec_bank_sha256,
    )
    created_plan = create_assistant_matrix_plan_file(
        root / "assistant" / "matrix-plan.json", plan
    )
    verified_plan = load_verified_assistant_matrix_plan(
        created_plan.path,
        expected_file_sha256=created_plan.file_sha256,
        registry=registry,
    )
    locks = {}
    for run_config in MAIN_CONFIG_ORDER:
        created = create_assistant_run_bundle(
            verified_plan,
            config=run_config,
            backend=_DeterministicAssistantBackend(),
            registry=registry,
            output_dir=root / "assistant" / f"run-{run_config}",
        )
        locks[run_config] = (created.root, created.external_manifest_sha256)
    five = load_verified_five_config_runs(
        verified_plan,
        bundle_locks=locks,
        registry=registry,
    )
    if five.formal_eligible or any(run.manifest.formal_eligible for run in five.runs):
        raise AssertionError(
            "diagnostic Assistant runs must remain formally ineligible"
        )
    return five, selected, rubric, evaluation_id


def _packet_with_self_hash(model_type, payload: dict[str, object]):
    return model_type.model_validate(
        {**payload, "packet_sha256": _hash_payload(payload)}, strict=True
    )


def _build_evaluator_isolation(
    root: Path,
    *,
    catalog,
    selected: list[Query],
    five_runs,
    rubric: RubricSnapshot,
    evaluation_id: str,
) -> None:
    feedback = make_feedback_evaluator_identity(
        provider="offline-feedback-diagnostic",
        model="feedback-visual-double-v3",
        model_family="diagnostic-feedback-family",
        endpoint="https://feedback.example.invalid/offline",
    )
    final = make_final_evaluator_identity(
        provider="offline-final-diagnostic",
        model="final-visual-double-v3",
        model_family="diagnostic-final-family",
        endpoint="https://final.example.invalid/offline",
    )
    isolation = make_evaluator_isolation_lock(feedback, final)
    full_run = next(run for run in five_runs.runs if run.manifest.config == "full")
    visible_response = full_run.rows[0].result.response_text
    image_bytes = (catalog.asset_root / selected[0].image_path).read_bytes()
    image = EvaluationImage(
        mime_type="image/png",
        sha256=sha256_bytes(image_bytes),
    )
    final_payload = {
        "schema_version": 2,
        "packet_kind": "final",
        "cache_namespace": "final-evaluator-v2",
        "evaluation_id": evaluation_id,
        "turns": (ConversationTurn(role="user", content=selected[0].text),),
        "image": image,
        "response_text": visible_response,
        "cards": (),
        "tool_evidence": (),
        "card_requirement": "not_applicable",
        "rubric": rubric,
    }
    final_packet = _packet_with_self_hash(FinalEvaluationPacket, final_payload)
    feedback_payload = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": "feedback-evaluator-v2",
        "query_id": selected[0].query_id,
        "turns": (ConversationTurn(role="user", content=selected[0].text),),
        "image": image,
        "canonical_capability": selected[0].canonical_capability,
        "acceptable_capabilities": tuple(selected[0].acceptable_capabilities),
        "response_text": visible_response,
        "cards": (),
        "tool_evidence": (),
        "tool_trace": (),
        "rubric": rubric,
    }
    feedback_packet = _packet_with_self_hash(FeedbackPacket, feedback_payload)
    final_bytes = serialize_final_evaluation_packet(final_packet)
    feedback_bytes = serialize_feedback_packet(feedback_packet)
    final_prompt = build_bound_final_prompt(final_packet, isolation)
    feedback_prompt = build_bound_feedback_prompt(feedback_packet, isolation)
    if final_prompt.cache_namespace == feedback_prompt.cache_namespace:
        raise AssertionError("final and feedback evaluator caches must be isolated")
    artifacts = {
        "evaluator-isolation-lock.json": canonical_json_bytes(
            isolation.model_dump(mode="json")
        ),
        "feedback-packet.json": feedback_bytes,
        "feedback-prompt.json": canonical_json_bytes(
            feedback_prompt.model_dump(mode="json")
        ),
        "final-packet.json": final_bytes,
        "final-prompt.json": canonical_json_bytes(final_prompt.model_dump(mode="json")),
    }
    for name, content in artifacts.items():
        atomic_create_file(root / "evaluation" / name, content)


def _artifact_descriptors(root: Path) -> tuple[list[dict[str, object]], str]:
    descriptors: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("offline fixture must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "fixture-manifest.json":
            continue
        content = path.read_bytes()
        if b'"formal_eligible":true' in content:
            raise AssertionError(f"formal eligibility leaked into fixture: {relative}")
        descriptors.append(
            {
                "bytes": len(content),
                "path": relative,
                "sha256": sha256_bytes(content),
            }
        )
    return descriptors, _hash_payload(descriptors)


def create_offline_engineering_fixture(
    output_dir: str | Path,
    *,
    fixture_spec_path: str | Path = DEFAULT_FIXTURE_SPEC,
) -> CreatedOfflineFixture:
    """Create and deeply verify a new diagnostic fixture directory."""

    _install_network_guard()
    output = Path(output_dir).resolve()
    fixture_spec, fixture_spec_bytes = _load_fixture_spec(
        Path(fixture_spec_path).resolve()
    )
    staging = new_staging_directory(output)
    try:
        atomic_create_file(staging / "fixture-spec.json", fixture_spec_bytes)
        fixture_spec_sha256 = sha256_bytes(fixture_spec_bytes)
        catalog, assets = _build_catalog(staging, int(fixture_spec["seed"]))
        assigned, frozen_manifest, split_manifest_path = _build_and_freeze_queries(
            staging,
            catalog=catalog,
            assets=assets,
            seed=int(fixture_spec["seed"]),
            fixture_spec_sha256=fixture_spec_sha256,
        )
        registry = _build_diagnostic_registry(staging, catalog, assigned)
        _exercise_all_tools(
            staging,
            registry=registry,
            catalog=catalog,
            query=assigned[0],
        )
        spec_baseline = _build_spec_baseline(staging, registry)
        five_runs, selected, rubric, evaluation_id = _build_five_assistant_runs(
            staging,
            registry=registry,
            assigned=assigned,
            spec_bank_sha256=spec_baseline.bank.bank_sha256,
        )
        _build_evaluator_isolation(
            staging,
            catalog=catalog,
            selected=selected,
            five_runs=five_runs,
            rubric=rubric,
            evaluation_id=evaluation_id,
        )
        descriptors, artifact_tree_sha256 = _artifact_descriptors(staging)
        manifest_payload = {
            "artifact_count": len(descriptors),
            "artifact_tree_sha256": artifact_tree_sha256,
            "artifacts": descriptors,
            "coverage": {
                "assistant_configurations": list(MAIN_CONFIG_ORDER),
                "asset_catalog_sha256": catalog.catalog_sha256,
                "evaluator_cache_isolation": True,
                "group_constraints_exercised": [
                    "asset-catalog-leakage-component",
                    "template-family",
                ],
                "group_leakage_violation_count": 0,
                "query_gallery_violation_count": 0,
                "query_schema_version": 2,
                "task_spec_version": PHASE3_TASK_SPEC_VERSION,
                "spec_baseline_bank_sha256": spec_baseline.bank.bank_sha256,
                "split_assignment_sha256": frozen_manifest.assignment_sha256,
                "tool_interfaces": sorted(MVP_TOOL_NAMES_V2),
            },
            "diagnostic_registry_kind": "typed-provisional",
            "fixture_spec_sha256": fixture_spec_sha256,
            "formal_eligible": False,
            "formal_ineligibility_reason": FIXTURE_FORMAL_INELIGIBILITY_REASON,
            "kind": "offline-engineering-fixture",
            "limitations": [
                "six synthetic queries, not the real 200-query mini run",
                "no downloaded source dataset or human license review evidence",
                "document safety approval is a diagnostic double, not human review",
                "no real model, embedding, OCR, detector, or LLM execution",
                "no LLMStatic, S1, S2, or S3 quality claim",
                "no calibrated judge, human audit, statistics, or paper comparison",
            ],
            "network_policy": "python-socket-audit-guard",
            "policy_version": FIXTURE_POLICY_VERSION,
            "schema_version": 1,
        }
        manifest = {
            **manifest_payload,
            "manifest_sha256": _hash_payload(manifest_payload),
        }
        manifest_bytes = canonical_json_bytes(manifest)
        atomic_create_file(staging / "fixture-manifest.json", manifest_bytes)
        published = atomic_publish_new_directory(staging, output)
    except BaseException:
        _remove_staging_tree(staging)
        raise
    result = CreatedOfflineFixture(
        root=published,
        manifest_file_sha256=sha256_bytes(manifest_bytes),
        artifact_tree_sha256=artifact_tree_sha256,
    )
    verify_offline_engineering_fixture(
        result.root, expected_manifest_file_sha256=result.manifest_file_sha256
    )
    return result


def verify_offline_engineering_fixture(
    root: str | Path, *, expected_manifest_file_sha256: str
) -> dict[str, object]:
    root = Path(root)
    manifest_bytes = read_stable_regular_file(
        root / "fixture-manifest.json", label="offline fixture manifest"
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise ValueError("offline fixture manifest external digest mismatch")
    raw = parse_canonical_json(manifest_bytes, label="offline fixture manifest")
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != manifest_bytes:
        raise ValueError("offline fixture manifest must be a canonical object")
    unsigned = dict(raw)
    manifest_sha256 = unsigned.pop("manifest_sha256", None)
    if manifest_sha256 != _hash_payload(unsigned):
        raise ValueError("offline fixture manifest self hash mismatch")
    if raw.get("formal_eligible") is not False:
        raise ValueError("offline fixture must remain formally ineligible")
    if raw.get("policy_version") != FIXTURE_POLICY_VERSION:
        raise ValueError("offline fixture policy version mismatch")
    if raw.get("diagnostic_registry_kind") != "typed-provisional":
        raise ValueError("offline fixture registry kind mismatch")
    if raw.get("network_policy") != "python-socket-audit-guard":
        raise ValueError("offline fixture network policy mismatch")
    descriptors = raw.get("artifacts")
    if not isinstance(descriptors, list):
        raise ValueError("offline fixture manifest artifacts must be an array")
    if raw.get("artifact_count") != len(descriptors):
        raise ValueError("offline fixture artifact count mismatch")
    descriptor_paths = [
        item.get("path") if isinstance(item, dict) else None for item in descriptors
    ]
    if (
        any(not isinstance(path, str) for path in descriptor_paths)
        or descriptor_paths != sorted(descriptor_paths)
        or len(descriptor_paths) != len(set(descriptor_paths))
    ):
        raise ValueError("offline fixture artifact paths must be unique and sorted")
    expected_paths = {"fixture-manifest.json"}
    for item in descriptors:
        if not isinstance(item, dict):
            raise ValueError("offline fixture artifact descriptor is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise ValueError("offline fixture artifact path is unsafe")
        path = root / Path(relative)
        content = read_stable_regular_file(path, label=f"offline fixture {relative}")
        if len(content) != item.get("bytes") or sha256_bytes(content) != item.get(
            "sha256"
        ):
            raise ValueError(f"offline fixture artifact digest mismatch: {relative}")
        if b'"formal_eligible":true' in content:
            raise ValueError(f"formal eligibility leaked into fixture: {relative}")
        expected_paths.add(relative)
    discovered = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in discovered):
        raise ValueError("offline fixture must not contain symlinks")
    actual_paths = {
        path.relative_to(root).as_posix() for path in discovered if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("offline fixture file set differs from its manifest")
    if raw.get("artifact_tree_sha256") != _hash_payload(descriptors):
        raise ValueError("offline fixture artifact-tree hash mismatch")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory; existing paths are rejected.",
    )
    parser.add_argument(
        "--fixture-spec",
        type=Path,
        default=DEFAULT_FIXTURE_SPEC,
        help="Externally tracked canonical fixture specification.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    created = create_offline_engineering_fixture(
        args.output, fixture_spec_path=args.fixture_spec
    )
    print(
        json.dumps(
            {
                "artifact_tree_sha256": created.artifact_tree_sha256,
                "formal_eligible": False,
                "manifest_file_sha256": created.manifest_file_sha256,
                "output": str(created.root),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
