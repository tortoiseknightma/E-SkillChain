"""Formal, externally locked tool-run evidence and evaluation.

This module is intentionally separate from :mod:`skillchain.tools.benchmark`'s
diagnostic executor.  A formal run is published first and remains untrusted
until a second pass verifies its canonical bytes against an externally supplied
bundle digest, the live registry runtime, and the complete query/split/gold
assignment chain.  The evaluator accepts only the opaque handle returned by
that verifier.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Annotated, Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import AssetCatalog, CatalogError, load_asset_catalog
from skillchain.data.gallery_eligibility import load_query_artifact
from skillchain.evaluation.assistant_runs import (
    AssistantRunError,
    VerifiedPhase4Inputs,
    require_verified_phase4_inputs,
)
from skillchain.schemas import Query, Split
from skillchain.synthesis.splitting import assert_no_group_leakage
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools import benchmark as benchmark_module
from skillchain.tools.benchmark import (
    BenchmarkAggregateMetrics,
    BenchmarkCaseError,
    BenchmarkCaseResult,
    BenchmarkError,
    BenchmarkSplit,
    GoldCase,
    QueryAssignment,
    VerifiedBenchmarkGold,
    load_benchmark_gold,
)
from skillchain.tools.contracts import ProductSearchTrace
from skillchain.tools.document_ocr import DocumentOCRResult
from skillchain.tools.document_safety import (
    DocumentSafetyCatalog,
    DocumentSafetyCatalogError,
    load_document_safety_catalog,
)
from skillchain.tools.embedding import (
    DashScopeEmbeddingClient,
    FormalEmbeddingBackend,
    OpenCLIPEmbeddingBackend,
)
from skillchain.tools.kb_lookup import KBHit
from skillchain.tools.model_artifacts import VerifiedModelArtifact
from skillchain.tools.multi_product import MultiProductResult, MultiProductSearchService
from skillchain.tools.object_detect import (
    ObjectDetectionResult,
    ObjectDetectionService,
    UltralyticsDetectorBackend,
)
from skillchain.tools.product_index import ProductIndex
from skillchain.tools.product_search import ProductSearchService
from skillchain.tools.registry import (
    FormalRegistryRuntimeSnapshot,
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    RegistryError,
    ToolCallError,
    ToolExecutionContext,
    ToolRegistry,
    require_formal_registry_runtime,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ClaimKind = Literal[
    "image_product_search",
    "text_product_search",
    "style_similar_search",
    "kb_lookup",
    "object_detection",
    "document_ocr",
    "multi_product_chain",
]

FORMAL_TOOL_RUN_POLICY_VERSION = "formal-tool-run-v1"
FORMAL_EVALUATION_POLICY_VERSION = "formal-tool-evaluation-v1"

_SPEC_FILE = "spec.json"
_RECORDS_FILE = "records.jsonl"
_RUN_MANIFEST_FILE = "manifest.json"
_RUN_FILES = frozenset({_SPEC_FILE, _RECORDS_FILE, _RUN_MANIFEST_FILE})
_REPORT_FILE = "report.jsonl"
_REPORT_MANIFEST_FILE = "manifest.json"
_REPORT_FILES = frozenset({_REPORT_FILE, _REPORT_MANIFEST_FILE})

_ASSIGNMENTS_MARKER = object()
_TOOL_ASSISTANT_ISOLATION_MARKER = object()
_VERIFIED_RUN_MARKER = object()
_VERIFIED_EVALUATION_MARKER = object()

_CLAIM_TOOLS: dict[str, frozenset[str]] = {
    "image_product_search": frozenset({"image_product_search"}),
    "text_product_search": frozenset({"text_product_search"}),
    "style_similar_search": frozenset({"style_similar_search"}),
    "kb_lookup": frozenset({"encyclopedia_lookup", "recipe_lookup"}),
    "object_detection": frozenset({"object_detect"}),
    "document_ocr": frozenset({"document_ocr"}),
    # ``multi_product_chain`` is the stable semantic claim kind.  The legacy
    # seven-tool registry executes it through a separately locked composition,
    # while registry v2 exposes that composition as a first-class ToolSpec.
    "multi_product_chain": frozenset({"multi_product_chain", "multi_product_search"}),
}
_LEGACY_FORMAL_TOOL_NAMES = frozenset({*MVP_TOOL_NAMES, "multi_product_chain"})
_V2_FORMAL_TOOL_NAMES = MVP_TOOL_NAMES_V2
_RANKING_CLAIMS = frozenset(
    {
        "image_product_search",
        "text_product_search",
        "style_similar_search",
        "kb_lookup",
        "multi_product_chain",
    }
)
_PUBLIC_ERROR_MESSAGE = "formal tool execution failed"


class FormalEvaluationError(ValueError):
    """A formal assignment, run, or evaluation failed verification."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and already stripped")
    return value


class FormalGoldAssignment(_FrozenStrictModel):
    """Externally reviewed link between one query, one claim, and one gold row."""

    schema_version: Literal[1] = 1
    query_id: str
    query_row_sha256: Sha256
    asset_id: str
    asset_sha256: Sha256
    leakage_group_id: str
    corpus_split: Split
    benchmark_split: BenchmarkSplit
    claim_kind: ClaimKind
    tool_name: str
    arguments: dict[str, Any]
    gold_case_sha256: Sha256
    document_safety_approval_sha256: Sha256 | None = None

    @field_validator("query_id", "asset_id", "leakage_group_id", "tool_name")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if self.tool_name not in _CLAIM_TOOLS[self.claim_kind]:
            raise ValueError("claim_kind/tool_name mismatch")
        if self.claim_kind == "document_ocr":
            if self.document_safety_approval_sha256 is None:
                raise ValueError("document OCR requires an approved safety digest")
        elif self.document_safety_approval_sha256 is not None:
            raise ValueError("only document OCR may carry a safety approval digest")
        return self


class KBLookupEvidence(_FrozenStrictModel):
    tool_name: Literal["encyclopedia_lookup", "recipe_lookup"]
    hits: tuple[KBHit, ...]

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        expected = (
            "encyclopedia" if self.tool_name == "encyclopedia_lookup" else "recipe"
        )
        if any(hit.kind != expected for hit in self.hits):
            raise ValueError("KB hit kind does not match the invoked tool")
        if any(hit.artifact_binding.mode != "verified" for hit in self.hits):
            raise ValueError("formal KB evidence requires verified bindings")
        return self


FormalEvidence = (
    ProductSearchTrace
    | KBLookupEvidence
    | ObjectDetectionResult
    | DocumentOCRResult
    | MultiProductResult
)


class FormalInputEvidence(_FrozenStrictModel):
    query_id: str
    query_row_sha256: Sha256
    asset_id: str
    asset_sha256: Sha256
    query_text_sha256: Sha256
    arguments: dict[str, Any]
    arguments_sha256: Sha256

    @field_validator("query_id", "asset_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class FormalRuntimeEvidence(_FrozenStrictModel):
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    tool_spec_sha256: Sha256 | None
    tool_runtime_sha256: Sha256
    network_policy: Literal["offline", "runtime_bound_embedding"]


class FormalUpstreamBinding(_FrozenStrictModel):
    asset_catalog_sha256: Sha256
    query_artifact_sha256: Sha256
    split_assignment_sha256: Sha256
    gold_assignment_sha256: Sha256
    gold_sha256: Sha256
    gold_review_ledger_sha256: Sha256
    tool_assistant_isolation_sha256: Sha256
    document_safety_catalog_sha256: Sha256 | None
    document_safety_review_ledger_sha256: Sha256 | None
    leakage_policy_version: str
    leakage_group_id: str
    corpus_split: Split
    benchmark_split: BenchmarkSplit

    @field_validator("leakage_policy_version", "leakage_group_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class FormalRunError(_FrozenStrictModel):
    code: str
    error_type: str
    message: Literal["formal tool execution failed"] = _PUBLIC_ERROR_MESSAGE

    @field_validator("code", "error_type")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class FormalToolRunRecord(_FrozenStrictModel):
    """One typed, self-hashed claim record; failures remain first-class rows."""

    schema_version: Literal[1] = 1
    query_id: str
    claim_kind: ClaimKind
    tool_name: str
    status: Literal["ok", "error"]
    input: FormalInputEvidence
    evidence: FormalEvidence | None
    evidence_sha256: Sha256 | None
    runtime: FormalRuntimeEvidence
    upstream: FormalUpstreamBinding
    error: FormalRunError | None
    record_sha256: Sha256

    @field_validator("query_id", "tool_name")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.query_id != self.input.query_id:
            raise ValueError("record/input query_id mismatch")
        if self.tool_name not in _CLAIM_TOOLS[self.claim_kind]:
            raise ValueError("record claim_kind/tool_name mismatch")
        if self.status == "ok":
            if (
                self.error is not None
                or self.evidence is None
                or self.evidence_sha256 is None
            ):
                raise ValueError("successful record requires evidence and no error")
            _validate_evidence_kind(self.claim_kind, self.tool_name, self.evidence)
        elif (
            self.error is None
            or self.evidence is not None
            or self.evidence_sha256 is not None
        ):
            raise ValueError("error record requires an error and null evidence/hash")
        if (
            self.claim_kind == "multi_product_chain"
            and self.tool_name == "multi_product_chain"
        ):
            if self.runtime.tool_spec_sha256 is not None:
                raise ValueError("multi-product composition has no registry ToolSpec")
        elif self.runtime.tool_spec_sha256 is None:
            raise ValueError("registry claims require a ToolSpec binding")
        return self


class RunArtifactDescriptor(_FrozenStrictModel):
    path: Literal["spec.json", "records.jsonl"]
    bytes: int = Field(gt=0)
    rows: int = Field(gt=0)
    sha256: Sha256


class FormalEvaluationArtifactDescriptor(_FrozenStrictModel):
    path: Literal["report.jsonl"] = _REPORT_FILE
    bytes: int = Field(gt=0)
    rows: int = Field(gt=0)
    sha256: Sha256


class FormalToolRunSpec(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["formal-tool-run-spec"] = "formal-tool-run-spec"
    policy_version: Literal["formal-tool-run-v1"] = FORMAL_TOOL_RUN_POLICY_VERSION
    run_id: str
    query_count: int = Field(gt=0)
    claim_counts: dict[str, int]
    tool_counts: dict[str, int]
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    multi_product_runtime_sha256: Sha256 | None
    asset_catalog_sha256: Sha256
    query_artifact_sha256: Sha256
    split_assignment_sha256: Sha256
    gold_assignment_sha256: Sha256
    gold_sha256: Sha256
    gold_review_ledger_sha256: Sha256
    tool_assistant_isolation_sha256: Sha256
    document_safety_catalog_sha256: Sha256 | None
    document_safety_review_ledger_sha256: Sha256 | None
    spec_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "run_id")

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if set(self.claim_counts) != set(_CLAIM_TOOLS):
            raise ValueError("claim_counts must contain every formal claim kind")
        if any(count < 0 for count in self.claim_counts.values()):
            raise ValueError("claim counts must be non-negative")
        if sum(self.claim_counts.values()) != self.query_count:
            raise ValueError("claim counts do not cover the query set")
        actual_tools = frozenset(self.tool_counts)
        if actual_tools not in {
            _LEGACY_FORMAL_TOOL_NAMES,
            _V2_FORMAL_TOOL_NAMES,
        }:
            raise ValueError(
                "tool_counts must cover either the legacy composition or "
                "the registry-v2 composite ToolSpec"
            )
        if any(count <= 0 for count in self.tool_counts.values()):
            raise ValueError("every formal tool requires coverage")
        if sum(self.tool_counts.values()) != self.query_count:
            raise ValueError("tool counts do not cover the query set")
        legacy_multi = "multi_product_chain" in actual_tools
        if self.claim_counts["multi_product_chain"] > 0:
            if legacy_multi and self.multi_product_runtime_sha256 is None:
                raise ValueError("multi-product claims require a runtime binding")
            if not legacy_multi and self.multi_product_runtime_sha256 is not None:
                raise ValueError(
                    "registry-v2 multi-product claims use the registry runtime binding"
                )
        elif self.multi_product_runtime_sha256 is not None:
            raise ValueError("unused multi-product runtime binding")
        safety_values = (
            self.document_safety_catalog_sha256,
            self.document_safety_review_ledger_sha256,
        )
        if self.claim_counts["document_ocr"] > 0:
            if any(value is None for value in safety_values):
                raise ValueError("document OCR claims require safety-catalog locks")
        elif any(value is not None for value in safety_values):
            raise ValueError("unused document safety-catalog lock")
        return self


class FormalToolRunManifest(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["formal-tool-run"] = "formal-tool-run"
    policy_version: Literal["formal-tool-run-v1"] = FORMAL_TOOL_RUN_POLICY_VERSION
    run_id: str
    status: Literal["complete"] = "complete"
    spec_sha256: Sha256
    query_count: int = Field(gt=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    artifacts: dict[str, RunArtifactDescriptor]
    run_bundle_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "run_id")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.success_count + self.error_count != self.query_count:
            raise ValueError("run statuses do not cover every query")
        if set(self.artifacts) != {_SPEC_FILE, _RECORDS_FILE}:
            raise ValueError("formal run artifact set mismatch")
        if any(name != item.path for name, item in self.artifacts.items()):
            raise ValueError("formal run artifact path mismatch")
        if self.artifacts[_SPEC_FILE].rows != 1:
            raise ValueError("formal run spec descriptor must contain one row")
        if self.artifacts[_RECORDS_FILE].rows != self.query_count:
            raise ValueError("formal run records do not cover every query")
        return self


class FormalEvaluationManifest(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["formal-tool-evaluation"] = "formal-tool-evaluation"
    policy_version: Literal["formal-tool-evaluation-v1"] = (
        FORMAL_EVALUATION_POLICY_VERSION
    )
    evaluation_id: str
    assurance_level: Literal["formal-verified"] = "formal-verified"
    run_bundle_sha256: Sha256
    gold_sha256: Sha256
    gold_assignment_sha256: Sha256
    gold_review_ledger_sha256: Sha256
    tool_assistant_isolation_sha256: Sha256
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    document_safety_catalog_sha256: Sha256 | None
    document_safety_review_ledger_sha256: Sha256 | None
    case_count: int = Field(gt=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    metrics: BenchmarkAggregateMetrics
    metrics_by_split: dict[str, BenchmarkAggregateMetrics]
    report: FormalEvaluationArtifactDescriptor
    manifest_sha256: Sha256

    @field_validator("evaluation_id")
    @classmethod
    def validate_evaluation_id(cls, value: str) -> str:
        return _nonblank(value, "evaluation_id")

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.success_count + self.error_count != self.case_count:
            raise ValueError("evaluation statuses do not cover every case")
        if sum(self.split_counts.values()) != self.case_count:
            raise ValueError("evaluation split counts do not cover every case")
        if sum(self.task_counts.values()) != self.case_count:
            raise ValueError("evaluation task counts do not cover every case")
        if self.report.rows != self.case_count:
            raise ValueError("formal report does not cover every case")
        if set(self.split_counts) != {"tool_dev", "tool_test_frozen"}:
            raise ValueError("formal evaluation split counts are incomplete")
        if set(self.task_counts) != {"ranking", "detection", "ocr"}:
            raise ValueError("formal evaluation task counts are incomplete")
        if set(self.metrics_by_split) != {"tool_dev", "tool_test_frozen"}:
            raise ValueError("formal split metrics are incomplete")
        if any(value < 0 for value in self.split_counts.values()) or any(
            value < 0 for value in self.task_counts.values()
        ):
            raise ValueError("formal evaluation counts must be non-negative")
        return self


@dataclass(frozen=True)
class VerifiedToolAssistantComponentIsolation:
    """Catalog-derived proof that tool gold is outside Assistant val/test."""

    assistant_inputs: VerifiedPhase4Inputs = field(repr=False, compare=False)
    assignment_sha256: str
    assistant_query_sha256: str
    assistant_split_manifest_sha256: str
    asset_catalog_sha256: str
    leakage_policy_version: str
    tool_component_ids: tuple[str, ...]
    assistant_val_test_component_ids: tuple[str, ...]
    isolation_sha256: str
    _marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedFormalGoldAssignments:
    path: Path
    sha256: str
    query_path: Path
    query_sha256: str
    split_assignment_path: Path
    split_assignment_sha256: str
    catalog: AssetCatalog = field(repr=False, compare=False)
    document_safety_catalog: DocumentSafetyCatalog | None = field(
        repr=False, compare=False
    )
    expected_document_safety_catalog_sha256: str | None
    expected_document_safety_review_ledger_sha256: str | None
    gold: VerifiedBenchmarkGold = field(repr=False, compare=False)
    tool_assistant_isolation: VerifiedToolAssistantComponentIsolation = field(
        repr=False, compare=False
    )
    records: tuple[FormalGoldAssignment, ...]
    queries: tuple[Query, ...]
    _marker: object = field(repr=False, compare=False)

    def assignment_by_query_id(self) -> dict[str, FormalGoldAssignment]:
        return {item.query_id: item for item in self.records}

    def query_by_query_id(self) -> dict[str, Query]:
        return {item.query_id: item for item in self.queries}


@dataclass(frozen=True)
class CreatedFormalToolRun:
    """A published bundle descriptor, deliberately not an evaluator handle."""

    root: Path
    run_bundle_sha256: str


@dataclass(frozen=True)
class VerifiedFormalToolRun:
    root: Path
    manifest: FormalToolRunManifest
    spec: FormalToolRunSpec
    records: tuple[FormalToolRunRecord, ...]
    assignments: VerifiedFormalGoldAssignments = field(repr=False, compare=False)
    registry: ToolRegistry = field(repr=False, compare=False)
    multi_product_executor: object | None = field(repr=False, compare=False)
    _marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedFormalEvaluation:
    root: Path
    manifest: FormalEvaluationManifest
    results: tuple[BenchmarkCaseResult, ...]
    run: VerifiedFormalToolRun = field(repr=False, compare=False)
    manifest_file_sha256: str
    _marker: object = field(repr=False, compare=False)


class MultiProductExecutor(Protocol):
    detector: object
    product_search: object

    @property
    def execution_location(self) -> Literal["local", "remote"]: ...

    def search(
        self, image_path: str | Path, *, asset_id: str
    ) -> MultiProductResult: ...


def _formal_tool_names_for_registry(registry: ToolRegistry) -> frozenset[str]:
    if registry.manifest.schema_version == 1:
        return _LEGACY_FORMAL_TOOL_NAMES
    if registry.manifest.schema_version == 2:
        return _V2_FORMAL_TOOL_NAMES
    raise FormalEvaluationError("unsupported formal registry generation")


def _verify_assignment_registry_contract(
    assignments: Sequence[FormalGoldAssignment],
    registry: ToolRegistry,
) -> None:
    """Bind the stable multi-product claim to one generation's authority graph."""

    expected_multi_tool = (
        "multi_product_chain"
        if registry.manifest.schema_version == 1
        else "multi_product_search"
    )
    mismatches = tuple(
        item.query_id
        for item in assignments
        if item.claim_kind == "multi_product_chain"
        and item.tool_name != expected_multi_tool
    )
    if mismatches:
        raise FormalEvaluationError(
            "multi-product assignment does not match the registry generation: "
            f"{mismatches!r}"
        )
    actual_tools = {item.tool_name for item in assignments}
    expected_tools = _formal_tool_names_for_registry(registry)
    if actual_tools != expected_tools:
        raise FormalEvaluationError(
            "formal assignment tool coverage does not match the registry generation"
        )


def load_formal_gold_assignments(
    assignment_path: str | Path,
    *,
    expected_assignment_sha256: str,
    gold_path: str | Path,
    expected_gold_sha256: str,
    gold_review_ledger_path: str | Path,
    expected_gold_review_ledger_sha256: str,
    query_path: str | Path,
    expected_query_sha256: str,
    split_assignment_path: str | Path,
    expected_split_assignment_sha256: str,
    catalog: AssetCatalog,
    expected_asset_catalog_sha256: str,
    assistant_inputs: VerifiedPhase4Inputs,
    document_safety_catalog: DocumentSafetyCatalog | None = None,
    expected_document_safety_catalog_sha256: str | None = None,
    expected_document_safety_review_ledger_sha256: str | None = None,
) -> VerifiedFormalGoldAssignments:
    """Verify the canonical external assignment and all transitive identities."""

    expected_assignment_sha256 = _require_sha256(
        expected_assignment_sha256, "expected_assignment_sha256"
    )
    expected_gold_sha256 = _require_sha256(expected_gold_sha256, "expected_gold_sha256")
    expected_query_sha256 = _require_sha256(
        expected_query_sha256, "expected_query_sha256"
    )
    expected_split_assignment_sha256 = _require_sha256(
        expected_split_assignment_sha256, "expected_split_assignment_sha256"
    )
    expected_asset_catalog_sha256 = _require_sha256(
        expected_asset_catalog_sha256, "expected_asset_catalog_sha256"
    )
    fresh_catalog = _fresh_catalog(catalog, expected_asset_catalog_sha256)

    assignment_path = Path(assignment_path)
    assignment_bytes = _read_expected_canonical_jsonl(
        assignment_path,
        expected_assignment_sha256,
        "formal gold assignment",
    )
    assignment_values = parse_canonical_jsonl(
        assignment_bytes, label="formal gold assignment"
    )
    assignments = _parse_models(
        assignment_values, FormalGoldAssignment, "formal gold assignment"
    )
    if not assignments:
        raise FormalEvaluationError("formal gold assignment must not be empty")
    if tuple(item.query_id for item in assignments) != tuple(
        sorted(item.query_id for item in assignments)
    ):
        raise FormalEvaluationError(
            "formal gold assignments must be sorted by query_id"
        )
    if assignment_bytes != canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in assignments)
    ):
        raise FormalEvaluationError(
            "formal gold assignment must be canonical typed JSONL"
        )
    _reject_duplicate_ids(assignments, "formal gold assignment")
    refreshed_safety_catalog = _prepare_document_safety_catalog(
        assignments,
        fresh_catalog,
        document_safety_catalog=document_safety_catalog,
        expected_catalog_sha256=expected_document_safety_catalog_sha256,
        expected_review_ledger_sha256=(expected_document_safety_review_ledger_sha256),
    )

    query_path = Path(query_path)
    query_snapshot = _read_real_stable_file(query_path, label="formal queries")
    if sha256_bytes(query_snapshot) != expected_query_sha256:
        raise FormalEvaluationError(
            "query artifact does not match expected_query_sha256"
        )
    query_bytes, queries = load_query_artifact(query_path, fresh_catalog)
    if sha256_bytes(query_bytes) != expected_query_sha256:
        raise FormalEvaluationError(
            "query artifact does not match expected_query_sha256"
        )
    if _read_real_stable_file(query_path, label="formal queries") != query_bytes:
        raise FormalEvaluationError("query artifact changed during verification")
    if tuple(query.query_id for query in queries) != tuple(
        sorted(query.query_id for query in queries)
    ):
        raise FormalEvaluationError("formal query artifact must be sorted by query_id")

    split_assignment_path = Path(split_assignment_path)
    split_bytes = _read_expected_canonical_jsonl(
        split_assignment_path,
        expected_split_assignment_sha256,
        "split assignment",
    )
    split_values = parse_canonical_jsonl(split_bytes, label="split assignment")
    split_queries = _parse_models(split_values, Query, "split assignment")
    if not split_queries:
        raise FormalEvaluationError("split assignment must not be empty")
    if split_bytes != canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in split_queries)
    ):
        raise FormalEvaluationError(
            "split assignment must be canonical schema-v2 JSONL"
        )
    _reject_duplicate_ids(split_queries, "split assignment")
    try:
        assert_no_group_leakage(split_queries)
    except ValueError as error:
        raise FormalEvaluationError(str(error)) from error

    projected = tuple(
        QueryAssignment(
            query_id=item.query_id,
            leakage_group_id=item.leakage_group_id,
            split=item.benchmark_split,
        )
        for item in assignments
    )
    gold_path = Path(gold_path)
    gold_snapshot = _read_real_stable_file(gold_path, label="formal gold")
    if sha256_bytes(gold_snapshot) != expected_gold_sha256:
        raise FormalEvaluationError("formal gold does not match external digest")
    try:
        gold = load_benchmark_gold(
            gold_path,
            expected_gold_sha256=expected_gold_sha256,
            authoritative_assignments=projected,
            review_ledger_path=gold_review_ledger_path,
            expected_review_ledger_sha256=expected_gold_review_ledger_sha256,
        )
    except BenchmarkError as error:
        raise FormalEvaluationError(str(error)) from error
    _verify_assignment_semantics(
        assignments,
        queries,
        split_queries,
        fresh_catalog,
        gold,
    )
    _verify_document_safety_assignments(assignments, refreshed_safety_catalog)
    isolation = _verify_tool_assistant_component_isolation(
        assignments,
        assignment_sha256=expected_assignment_sha256,
        catalog=fresh_catalog,
        assistant_inputs=assistant_inputs,
    )
    fresh_catalog.verify_asset_ids(item.asset_id for item in assignments)
    if (
        _read_real_stable_file(assignment_path, label="formal gold assignment")
        != assignment_bytes
    ):
        raise FormalEvaluationError(
            "formal gold assignment changed during verification"
        )
    if (
        _read_real_stable_file(split_assignment_path, label="split assignment")
        != split_bytes
    ):
        raise FormalEvaluationError("split assignment changed during verification")
    if _read_real_stable_file(gold_path, label="formal gold") != gold_snapshot:
        raise FormalEvaluationError("formal gold changed during verification")
    return VerifiedFormalGoldAssignments(
        path=assignment_path.resolve(),
        sha256=expected_assignment_sha256,
        query_path=query_path.resolve(),
        query_sha256=expected_query_sha256,
        split_assignment_path=split_assignment_path.resolve(),
        split_assignment_sha256=expected_split_assignment_sha256,
        catalog=fresh_catalog,
        document_safety_catalog=refreshed_safety_catalog,
        expected_document_safety_catalog_sha256=(
            expected_document_safety_catalog_sha256
        ),
        expected_document_safety_review_ledger_sha256=(
            expected_document_safety_review_ledger_sha256
        ),
        gold=gold,
        tool_assistant_isolation=isolation,
        records=assignments,
        queries=queries,
        _marker=_ASSIGNMENTS_MARKER,
    )


def verify_tool_assistant_component_isolation(
    assignments: Sequence[FormalGoldAssignment],
    *,
    expected_assignment_sha256: str,
    catalog: AssetCatalog,
    expected_asset_catalog_sha256: str,
    assistant_inputs: VerifiedPhase4Inputs,
) -> VerifiedToolAssistantComponentIsolation:
    """Verify a catalog-component firewall across formal tool and Assistant data.

    The Assistant side is derived from every ``val`` and ``test_frozen`` row in
    its externally locked full split artifact, not merely from its selected
    matrix subset.  Both sides are re-resolved through the same externally
    pinned asset catalog before the zero-intersection proof is granted.
    """

    expected_assignment_sha256 = _require_sha256(
        expected_assignment_sha256, "expected_assignment_sha256"
    )
    expected_asset_catalog_sha256 = _require_sha256(
        expected_asset_catalog_sha256, "expected_asset_catalog_sha256"
    )
    parsed = tuple(assignments)
    if not parsed or any(type(item) is not FormalGoldAssignment for item in parsed):
        raise TypeError("assignments must contain FormalGoldAssignment values")
    if (
        sha256_bytes(
            canonical_jsonl_bytes(
                tuple(item.model_dump(mode="json") for item in parsed)
            )
        )
        != expected_assignment_sha256
    ):
        raise FormalEvaluationError(
            "tool assignment values do not match expected_assignment_sha256"
        )
    fresh_catalog = _fresh_catalog(catalog, expected_asset_catalog_sha256)
    return _verify_tool_assistant_component_isolation(
        parsed,
        assignment_sha256=expected_assignment_sha256,
        catalog=fresh_catalog,
        assistant_inputs=assistant_inputs,
    )


def _verify_tool_assistant_component_isolation(
    assignments: tuple[FormalGoldAssignment, ...],
    *,
    assignment_sha256: str,
    catalog: AssetCatalog,
    assistant_inputs: VerifiedPhase4Inputs,
) -> VerifiedToolAssistantComponentIsolation:
    try:
        refreshed_inputs = require_verified_phase4_inputs(assistant_inputs)
    except (AssistantRunError, TypeError) as error:
        raise FormalEvaluationError(
            "Assistant Phase 4 inputs failed deep verification"
        ) from error
    manifest = refreshed_inputs.split_manifest
    if (
        manifest.asset_catalog_sha256,
        manifest.leakage_policy_version,
    ) != (catalog.catalog_sha256, catalog.leakage_policy_version):
        raise FormalEvaluationError(
            "Assistant split manifest is not bound to the formal asset catalog"
        )

    try:
        tool_components = tuple(
            sorted({catalog.component_for_asset(item.asset_id) for item in assignments})
        )
        if any(
            catalog.component_for_asset(item.asset_id) != item.leakage_group_id
            for item in assignments
        ):
            raise FormalEvaluationError(
                "tool assignment leakage component differs from the asset catalog"
            )
        assistant_rows = tuple(
            item
            for item in refreshed_inputs.queries
            if item.split in {"val", "test_frozen"}
        )
        if not assistant_rows:
            raise FormalEvaluationError(
                "Assistant split must contain at least one val/test row"
            )
        assistant_components = tuple(
            sorted(
                {
                    catalog.verify_reference(
                        item.asset_id,
                        item.image_path,
                        item.leakage_group_id,
                    ).leakage_group_id
                    for item in assistant_rows
                }
            )
        )
    except CatalogError as error:
        raise FormalEvaluationError(
            "tool/Assistant partition references invalid catalog identities"
        ) from error

    overlap = tuple(sorted(set(tool_components) & set(assistant_components)))
    if overlap:
        raise FormalEvaluationError(
            f"tool gold and Assistant val/test catalog components overlap: {overlap!r}"
        )
    unsigned = {
        "policy_version": "tool-assistant-component-isolation-v1",
        "assignment_sha256": assignment_sha256,
        "assistant_query_sha256": refreshed_inputs.expected_query_sha256,
        "assistant_split_manifest_sha256": (
            refreshed_inputs.expected_split_manifest_sha256
        ),
        "asset_catalog_sha256": catalog.catalog_sha256,
        "leakage_policy_version": catalog.leakage_policy_version,
        "tool_component_ids": list(tool_components),
        "assistant_val_test_component_ids": list(assistant_components),
    }
    return VerifiedToolAssistantComponentIsolation(
        assistant_inputs=refreshed_inputs,
        assignment_sha256=assignment_sha256,
        assistant_query_sha256=refreshed_inputs.expected_query_sha256,
        assistant_split_manifest_sha256=(
            refreshed_inputs.expected_split_manifest_sha256
        ),
        asset_catalog_sha256=catalog.catalog_sha256,
        leakage_policy_version=catalog.leakage_policy_version,
        tool_component_ids=tool_components,
        assistant_val_test_component_ids=assistant_components,
        isolation_sha256=sha256_bytes(canonical_json_bytes(unsigned)),
        _marker=_TOOL_ASSISTANT_ISOLATION_MARKER,
    )


def create_formal_tool_run(
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    destination: str | Path,
    *,
    run_id: str,
    expected_registry_sha256: str,
    expected_registry_runtime_sha256: str,
    multi_product_executor: MultiProductExecutor | None = None,
    expected_multi_product_runtime_sha256: str | None = None,
) -> CreatedFormalToolRun:
    """Execute the locked assignment and publish an untrusted create-only bundle."""

    assignments = _require_assignments(assignments)
    run_id = _nonblank(run_id, "run_id")
    runtime_snapshot = _verify_registry_identity(
        registry,
        expected_registry_sha256,
        expected_registry_runtime_sha256,
    )
    refreshed = _refresh_assignments(assignments)
    formal_tool_names = _formal_tool_names_for_registry(registry)
    _verify_assignment_registry_contract(refreshed.records, registry)
    has_legacy_multi = any(
        item.claim_kind == "multi_product_chain"
        and item.tool_name == "multi_product_chain"
        for item in refreshed.records
    )
    multi_runtime = _prepare_multi_runtime(
        has_legacy_multi,
        multi_product_executor,
        expected_multi_product_runtime_sha256,
    )
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"formal run destination already exists: {destination}")

    spec = _build_run_spec(
        refreshed,
        runtime_snapshot,
        run_id=run_id,
        multi_product_runtime_sha256=multi_runtime,
        formal_tool_names=formal_tool_names,
    )
    records = tuple(
        _execute_assignment(
            item,
            refreshed,
            registry,
            runtime_snapshot,
            multi_product_executor=multi_product_executor,
            expected_multi_product_runtime_sha256=multi_runtime,
        )
        for item in refreshed.records
    )
    spec_bytes = canonical_json_bytes(spec.model_dump(mode="json"))
    records_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in records)
    )
    manifest = _build_run_manifest(spec, records, spec_bytes, records_bytes)
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))

    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / _SPEC_FILE, spec_bytes)
        atomic_create_file(staging / _RECORDS_FILE, records_bytes)
        atomic_create_file(staging / _RUN_MANIFEST_FILE, manifest_bytes)
        _load_run_bundle(
            staging,
            refreshed,
            registry,
            expected_run_bundle_sha256=manifest.run_bundle_sha256,
            expected_registry_sha256=expected_registry_sha256,
            expected_registry_runtime_sha256=expected_registry_runtime_sha256,
            multi_product_executor=multi_product_executor,
            expected_multi_product_runtime_sha256=multi_runtime,
            grant_handle=False,
        )
        _refresh_assignments(refreshed)
        atomic_publish_new_directory(staging, destination)
    except BaseException:
        _remove_owned_staging(staging, destination.parent)
        raise
    return CreatedFormalToolRun(
        root=destination.resolve(),
        run_bundle_sha256=manifest.run_bundle_sha256,
    )


def load_verified_formal_tool_run(
    run_dir: str | Path,
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    *,
    expected_run_bundle_sha256: str,
    expected_registry_sha256: str,
    expected_registry_runtime_sha256: str,
    multi_product_executor: MultiProductExecutor | None = None,
    expected_multi_product_runtime_sha256: str | None = None,
) -> VerifiedFormalToolRun:
    """Reverify a bundle and grant the only handle accepted by the evaluator."""

    assignments = _require_assignments(assignments)
    return _load_run_bundle(
        Path(run_dir),
        _refresh_assignments(assignments),
        registry,
        expected_run_bundle_sha256=expected_run_bundle_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_registry_runtime_sha256=expected_registry_runtime_sha256,
        multi_product_executor=multi_product_executor,
        expected_multi_product_runtime_sha256=expected_multi_product_runtime_sha256,
        grant_handle=True,
    )


def require_verified_formal_tool_run(value: object) -> VerifiedFormalToolRun:
    if (
        not isinstance(value, VerifiedFormalToolRun)
        or value._marker is not _VERIFIED_RUN_MARKER
    ):
        raise TypeError(
            "formal evaluator requires a handle returned by "
            "load_verified_formal_tool_run"
        )
    return value


def evaluate_formal_tool_run(
    run: VerifiedFormalToolRun,
    destination: str | Path,
    *,
    evaluation_id: str,
) -> VerifiedFormalEvaluation:
    """Evaluate every locked row; tool errors remain zero-valued denominator rows."""

    run = require_verified_formal_tool_run(run)
    evaluation_id = _nonblank(evaluation_id, "evaluation_id")
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"formal evaluation destination already exists: {destination}"
        )
    run = _refresh_verified_run(run)
    gold = run.assignments.gold
    case_by_id = gold.case_by_query_id()
    results: list[BenchmarkCaseResult] = []
    observations: list[Any] = []
    for record in run.records:
        case = case_by_id[record.query_id]
        result, case_observations = _evaluate_formal_record(record, case)
        results.append(result)
        observations.extend(case_observations)

    result_tuple = tuple(results)
    metrics = benchmark_module._aggregate_metrics(  # noqa: SLF001
        gold.cases, result_tuple, observations
    )
    metrics_by_split = benchmark_module._aggregate_metrics_by_split(  # noqa: SLF001
        gold.cases, result_tuple, observations
    )
    report_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in result_tuple)
    )
    descriptor = FormalEvaluationArtifactDescriptor(
        bytes=len(report_bytes),
        rows=len(result_tuple),
        sha256=sha256_bytes(report_bytes),
    )
    split_counts = {
        split: sum(case.split == split for case in gold.cases)
        for split in ("tool_dev", "tool_test_frozen")
    }
    task_counts = {
        task: sum(case.task == task for case in gold.cases)
        for task in ("ranking", "detection", "ocr")
    }
    success_count = sum(item.status == "ok" for item in result_tuple)
    unsigned = {
        "schema_version": 1,
        "kind": "formal-tool-evaluation",
        "policy_version": FORMAL_EVALUATION_POLICY_VERSION,
        "evaluation_id": evaluation_id,
        "assurance_level": "formal-verified",
        "run_bundle_sha256": run.manifest.run_bundle_sha256,
        "gold_sha256": gold.sha256,
        "gold_assignment_sha256": run.assignments.sha256,
        "gold_review_ledger_sha256": gold.review_ledger_sha256,
        "tool_assistant_isolation_sha256": (
            run.assignments.tool_assistant_isolation.isolation_sha256
        ),
        "registry_sha256": run.spec.registry_sha256,
        "registry_runtime_sha256": run.spec.registry_runtime_sha256,
        "document_safety_catalog_sha256": (run.spec.document_safety_catalog_sha256),
        "document_safety_review_ledger_sha256": (
            run.spec.document_safety_review_ledger_sha256
        ),
        "case_count": len(result_tuple),
        "success_count": success_count,
        "error_count": len(result_tuple) - success_count,
        "split_counts": split_counts,
        "task_counts": task_counts,
        "metrics": metrics.model_dump(mode="json"),
        "metrics_by_split": {
            key: value.model_dump(mode="json")
            for key, value in metrics_by_split.items()
        },
        "report": descriptor.model_dump(mode="json"),
    }
    manifest = FormalEvaluationManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / _REPORT_FILE, report_bytes)
        atomic_create_file(staging / _REPORT_MANIFEST_FILE, manifest_bytes)
        _load_formal_evaluation(
            staging,
            run,
            expected_manifest_file_sha256=sha256_bytes(manifest_bytes),
        )
        run = _refresh_verified_run(run)
        atomic_publish_new_directory(staging, destination)
    except BaseException:
        _remove_owned_staging(staging, destination.parent)
        raise
    return load_verified_formal_evaluation(
        destination,
        run,
        expected_manifest_file_sha256=sha256_bytes(manifest_bytes),
    )


def load_verified_formal_evaluation(
    path: str | Path,
    run: VerifiedFormalToolRun,
    *,
    expected_manifest_file_sha256: str,
) -> VerifiedFormalEvaluation:
    """Reverify a formal report against the still-valid verified run handle."""

    run = require_verified_formal_tool_run(run)
    run = _refresh_verified_run(run)
    result = _load_formal_evaluation(
        Path(path),
        run,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    _refresh_verified_run(run)
    return result


def require_verified_formal_evaluation(value: object) -> VerifiedFormalEvaluation:
    if (
        not isinstance(value, VerifiedFormalEvaluation)
        or value._marker is not _VERIFIED_EVALUATION_MARKER
    ):
        raise TypeError("verified formal evaluation handle required")
    return load_verified_formal_evaluation(
        value.root,
        value.run,
        expected_manifest_file_sha256=value.manifest_file_sha256,
    )


def _load_run_bundle(
    run_dir: Path,
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    *,
    expected_run_bundle_sha256: str,
    expected_registry_sha256: str,
    expected_registry_runtime_sha256: str,
    multi_product_executor: MultiProductExecutor | None,
    expected_multi_product_runtime_sha256: str | None,
    grant_handle: bool,
) -> VerifiedFormalToolRun:
    expected_run_bundle_sha256 = _require_sha256(
        expected_run_bundle_sha256, "expected_run_bundle_sha256"
    )
    runtime_snapshot = _verify_registry_identity(
        registry,
        expected_registry_sha256,
        expected_registry_runtime_sha256,
    )
    before = _read_exact_directory(run_dir, _RUN_FILES, "formal run")
    try:
        spec_value = parse_canonical_json(before[_SPEC_FILE], label="formal run spec")
        spec = FormalToolRunSpec.model_validate(spec_value)
        manifest_value = parse_canonical_json(
            before[_RUN_MANIFEST_FILE], label="formal run manifest"
        )
        manifest = FormalToolRunManifest.model_validate(manifest_value)
        row_values = parse_canonical_jsonl(
            before[_RECORDS_FILE], label="formal run records"
        )
        records = _parse_models(row_values, FormalToolRunRecord, "formal run record")
    except (ArtifactFormatError, ValidationError) as error:
        raise FormalEvaluationError(str(error)) from error
    if spec.spec_sha256 != _model_digest(spec, "spec_sha256"):
        raise FormalEvaluationError("formal run spec self-hash mismatch")
    if manifest.run_bundle_sha256 != _model_digest(manifest, "run_bundle_sha256"):
        raise FormalEvaluationError("formal run bundle self-hash mismatch")
    if manifest.run_bundle_sha256 != expected_run_bundle_sha256:
        raise FormalEvaluationError("formal run does not match external bundle digest")
    _verify_run_descriptors(manifest, before)
    multi_runtime_snapshot = _verify_spec_bindings(
        spec,
        manifest,
        assignments,
        runtime_snapshot,
        expected_multi_product_runtime_sha256,
        multi_product_executor,
    )
    _verify_records(
        records,
        spec,
        manifest,
        assignments,
        registry,
        runtime_snapshot,
        multi_product_executor,
    )
    after = _read_exact_directory(run_dir, _RUN_FILES, "formal run")
    if after != before:
        raise FormalEvaluationError("formal run bundle changed during verification")
    refreshed = _refresh_assignments(assignments)
    if (
        refreshed.records != assignments.records
        or refreshed.queries != assignments.queries
    ):
        raise FormalEvaluationError("formal run upstream assignments changed")
    try:
        require_formal_registry_runtime(registry).verify(runtime_snapshot)
    except (RegistryError, TypeError) as error:
        raise FormalEvaluationError(
            "formal registry runtime changed during run verification"
        ) from error
    _verify_multi_runtime_snapshot(
        multi_runtime_snapshot,
        multi_product_executor,
    )
    return VerifiedFormalToolRun(
        root=run_dir.resolve(),
        manifest=manifest,
        spec=spec,
        records=records,
        assignments=refreshed,
        registry=registry,
        multi_product_executor=multi_product_executor,
        _marker=_VERIFIED_RUN_MARKER if grant_handle else object(),
    )


def _verify_records(
    records: tuple[FormalToolRunRecord, ...],
    spec: FormalToolRunSpec,
    manifest: FormalToolRunManifest,
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    runtime_snapshot: FormalRegistryRuntimeSnapshot,
    multi_product_executor: MultiProductExecutor | None,
) -> None:
    if len(records) != spec.query_count or len(records) != manifest.query_count:
        raise FormalEvaluationError("formal run query count mismatch")
    if tuple(item.query_id for item in records) != tuple(
        sorted(item.query_id for item in records)
    ):
        raise FormalEvaluationError("formal run records must be sorted by query_id")
    _reject_duplicate_ids(records, "formal run record")
    assignment_by_id = assignments.assignment_by_query_id()
    query_by_id = assignments.query_by_query_id()
    _verify_assignment_registry_contract(assignments.records, registry)
    if set(item.query_id for item in records) != set(assignment_by_id):
        raise FormalEvaluationError(
            "formal run has duplicate, missing, or extra queries"
        )
    actual_claim_counts = Counter(item.claim_kind for item in records)
    actual_tool_counts = Counter(item.tool_name for item in records)
    formal_tool_names = _formal_tool_names_for_registry(registry)
    if spec.claim_counts != {
        key: actual_claim_counts[key] for key in sorted(_CLAIM_TOOLS)
    } or spec.tool_counts != {
        key: actual_tool_counts[key] for key in sorted(formal_tool_names)
    }:
        raise FormalEvaluationError("formal run claim/tool coverage mismatch")
    specs = {item.name: item for item in registry.specs()}
    success_count = 0
    for record in records:
        if record.record_sha256 != _model_digest(record, "record_sha256"):
            raise FormalEvaluationError(f"record self-hash mismatch: {record.query_id}")
        assignment = assignment_by_id[record.query_id]
        query = query_by_id[record.query_id]
        _verify_record_identity(
            record,
            assignment,
            query,
            assignments,
            specs,
            runtime_snapshot,
            spec,
        )
        if record.status == "ok":
            success_count += 1
            assert record.evidence is not None
            if record.evidence_sha256 != sha256_bytes(
                canonical_json_bytes(_json_model(record.evidence))
            ):
                raise FormalEvaluationError(
                    f"evidence hash mismatch: {record.query_id}"
                )
            _verify_evidence_upstream(
                record,
                assignment,
                query,
                assignments,
                registry,
                multi_product_executor,
            )
    if (manifest.success_count, manifest.error_count) != (
        success_count,
        len(records) - success_count,
    ):
        raise FormalEvaluationError("formal run status counts mismatch")


def _verify_record_identity(
    record: FormalToolRunRecord,
    assignment: FormalGoldAssignment,
    query: Query,
    assignments: VerifiedFormalGoldAssignments,
    specs: Mapping[str, Any],
    runtime_snapshot: FormalRegistryRuntimeSnapshot,
    run_spec: FormalToolRunSpec,
) -> None:
    if (
        record.claim_kind,
        record.tool_name,
        record.input.query_row_sha256,
        record.input.asset_id,
        record.input.asset_sha256,
        record.input.arguments,
    ) != (
        assignment.claim_kind,
        assignment.tool_name,
        assignment.query_row_sha256,
        assignment.asset_id,
        assignment.asset_sha256,
        assignment.arguments,
    ):
        raise FormalEvaluationError(f"record assignment mismatch: {record.query_id}")
    if record.input.query_text_sha256 != sha256_bytes(query.text.encode("utf-8")):
        raise FormalEvaluationError(f"record query text mismatch: {record.query_id}")
    if record.input.arguments_sha256 != sha256_bytes(
        canonical_json_bytes(record.input.arguments)
    ):
        raise FormalEvaluationError(f"record argument hash mismatch: {record.query_id}")
    expected_upstream = _upstream_binding(assignment, assignments)
    if record.upstream != expected_upstream:
        raise FormalEvaluationError(f"record upstream mismatch: {record.query_id}")
    if record.runtime.registry_sha256 != runtime_snapshot.registry_sha256 or (
        record.runtime.registry_runtime_sha256
        != runtime_snapshot.registry_runtime_sha256
    ):
        raise FormalEvaluationError(
            f"record runtime registry mismatch: {record.query_id}"
        )
    if (
        record.claim_kind == "multi_product_chain"
        and record.tool_name == "multi_product_chain"
    ):
        if (
            record.runtime.tool_runtime_sha256 != run_spec.multi_product_runtime_sha256
            or record.runtime.network_policy != "runtime_bound_embedding"
        ):
            raise FormalEvaluationError(
                f"record multi-product runtime mismatch: {record.query_id}"
            )
    else:
        spec = specs[record.tool_name]
        if (
            record.runtime.tool_spec_sha256 != spec.spec_sha256
            or record.runtime.tool_runtime_sha256
            != runtime_snapshot.tool_runtime_sha256(record.tool_name)
            or record.runtime.network_policy != spec.network_policy
        ):
            raise FormalEvaluationError(
                f"record tool runtime mismatch: {record.query_id}"
            )


def _verify_evidence_upstream(
    record: FormalToolRunRecord,
    assignment: FormalGoldAssignment,
    query: Query,
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    multi_product_executor: MultiProductExecutor | None,
) -> None:
    evidence = record.evidence
    assert evidence is not None
    if isinstance(evidence, ProductSearchTrace):
        if evidence.artifact_binding.mode != "verified":
            raise FormalEvaluationError("formal product evidence is provisional")
        if (
            evidence.artifact_binding.asset_catalog_sha256
            != assignments.catalog.catalog_sha256
            or evidence.artifact_binding.query_artifact_sha256
            != assignments.query_sha256
            or evidence.artifact_binding.leakage_policy_version
            != assignments.catalog.leakage_policy_version
            or any(
                hit.artifact_binding != evidence.artifact_binding
                for hit in evidence.hits
            )
        ):
            raise FormalEvaluationError(
                "formal product evidence does not bind the authoritative upstream"
            )
        if record.claim_kind == "text_product_search":
            if (
                evidence.input_kind != "text"
                or evidence.query_text != query.text
                or evidence.query_input_sha256
                != sha256_bytes(query.text.encode("utf-8"))
            ):
                raise FormalEvaluationError("text product evidence input mismatch")
        else:
            if (
                evidence.input_kind != "image"
                or evidence.query_asset_id != query.asset_id
                or evidence.query_input_sha256 != assignment.asset_sha256
            ):
                raise FormalEvaluationError("image product evidence input mismatch")
    elif isinstance(evidence, ObjectDetectionResult):
        if (
            evidence.input_binding.asset_id != query.asset_id
            or evidence.input_binding.image_sha256 != assignment.asset_sha256
        ):
            raise FormalEvaluationError("detection evidence input mismatch")
        _verify_registry_evidence(registry, record.tool_name, evidence)
    elif isinstance(evidence, DocumentOCRResult):
        if (
            evidence.input_binding.asset_id != query.asset_id
            or evidence.input_binding.image_sha256 != assignment.asset_sha256
            or evidence.input_binding.safety_approval_sha256
            != assignment.document_safety_approval_sha256
        ):
            raise FormalEvaluationError("OCR evidence lacks the approved input binding")
        safety_catalog = assignments.document_safety_catalog
        approval = (
            safety_catalog.approval_for(query.query_id, query.asset_id)
            if safety_catalog is not None
            else None
        )
        if (
            approval is None
            or evidence.input_binding.safety_decision != approval.decision
        ):
            raise FormalEvaluationError("OCR evidence safety decision mismatch")
        _verify_registry_evidence(registry, record.tool_name, evidence)
    elif isinstance(evidence, KBLookupEvidence):
        _verify_registry_evidence(registry, record.tool_name, evidence.hits)
    elif isinstance(evidence, MultiProductResult):
        if (
            evidence.input_binding.asset_id != query.asset_id
            or evidence.input_binding.image_sha256 != assignment.asset_sha256
            or evidence.artifact_binding.mode != "verified"
        ):
            raise FormalEvaluationError("multi-product evidence input mismatch")
        if (
            evidence.artifact_binding.asset_catalog_sha256
            != assignments.catalog.catalog_sha256
            or evidence.artifact_binding.query_artifact_sha256
            != assignments.query_sha256
            or evidence.artifact_binding.leakage_policy_version
            != assignments.catalog.leakage_policy_version
        ):
            raise FormalEvaluationError("multi-product evidence runtime mismatch")
        if record.tool_name == "multi_product_search":
            _verify_registry_evidence(registry, record.tool_name, evidence)
        else:
            if multi_product_executor is None:
                raise FormalEvaluationError(
                    "multi-product evidence runtime is unavailable"
                )
            try:
                expected_detection = (
                    multi_product_executor.detector.artifact.runtime_binding
                )
                expected_binding = (
                    multi_product_executor.product_search.artifact_binding
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise FormalEvaluationError(
                    "multi-product executor lacks structured evidence bindings"
                ) from error
            if (
                evidence.detection_runtime_binding != expected_detection
                or evidence.artifact_binding != expected_binding
            ):
                raise FormalEvaluationError("multi-product evidence runtime mismatch")

    if isinstance(evidence, ProductSearchTrace):
        _verify_registry_evidence(registry, record.tool_name, evidence)


def _verify_registry_evidence(
    registry: ToolRegistry, tool_name: str, evidence: object
) -> None:
    try:
        registry.verify_formal_evidence(tool_name, evidence, recheck_runtime=False)
    except RegistryError as error:
        raise FormalEvaluationError(
            f"formal evidence does not match the live runtime: {tool_name}"
        ) from error


def _execute_assignment(
    assignment: FormalGoldAssignment,
    assignments: VerifiedFormalGoldAssignments,
    registry: ToolRegistry,
    runtime_snapshot: FormalRegistryRuntimeSnapshot,
    *,
    multi_product_executor: MultiProductExecutor | None,
    expected_multi_product_runtime_sha256: str | None,
) -> FormalToolRunRecord:
    query = assignments.query_by_query_id()[assignment.query_id]
    resolution = assignments.catalog.verify_reference(
        query.asset_id,
        query.image_path,
        leakage_group_id=query.leakage_group_id,
    )
    image_path = assignments.catalog.asset_root.joinpath(
        *resolution.asset.local_path.split("/")
    )
    context = ToolExecutionContext(
        query_id=query.query_id,
        query_asset_id=query.asset_id,
        query_text=query.text,
        resolve_asset=lambda asset_id: (
            image_path if asset_id == query.asset_id else _raise_context_violation()
        ),
        query_cloud_upload_allowed=resolution.asset.cloud_upload_allowed,
    )
    input_evidence = FormalInputEvidence(
        query_id=query.query_id,
        query_row_sha256=assignment.query_row_sha256,
        asset_id=query.asset_id,
        asset_sha256=assignment.asset_sha256,
        query_text_sha256=sha256_bytes(query.text.encode("utf-8")),
        arguments=assignment.arguments,
        arguments_sha256=sha256_bytes(canonical_json_bytes(assignment.arguments)),
    )
    specs = {item.name: item for item in registry.specs()}
    legacy_multi = (
        assignment.claim_kind == "multi_product_chain"
        and assignment.tool_name == "multi_product_chain"
    )
    if legacy_multi:
        assert expected_multi_product_runtime_sha256 is not None
        runtime = FormalRuntimeEvidence(
            registry_sha256=runtime_snapshot.registry_sha256,
            registry_runtime_sha256=runtime_snapshot.registry_runtime_sha256,
            tool_spec_sha256=None,
            tool_runtime_sha256=expected_multi_product_runtime_sha256,
            network_policy="runtime_bound_embedding",
        )
    else:
        spec = specs[assignment.tool_name]
        runtime = FormalRuntimeEvidence(
            registry_sha256=runtime_snapshot.registry_sha256,
            registry_runtime_sha256=runtime_snapshot.registry_runtime_sha256,
            tool_spec_sha256=spec.spec_sha256,
            tool_runtime_sha256=runtime_snapshot.tool_runtime_sha256(
                assignment.tool_name
            ),
            network_policy=spec.network_policy,
        )
    try:
        if legacy_multi:
            if multi_product_executor is None:
                raise FormalEvaluationError("multi-product runtime is unavailable")
            if (
                multi_product_executor.execution_location == "remote"
                and resolution.asset.cloud_upload_allowed is not True
            ):
                raise ToolCallError(
                    "permission_denied",
                    "remote image processing is not permitted for this asset",
                )
            assert expected_multi_product_runtime_sha256 is not None
            raw_evidence: object = _search_multi_product_with_runtime_guard(
                multi_product_executor,
                image_path,
                asset_id=query.asset_id,
                expected_runtime_sha256=expected_multi_product_runtime_sha256,
            )
        else:
            invocation = registry.invoke(
                assignment.tool_name,
                assignment.arguments,
                context,
            )
            raw_evidence = invocation.output
        evidence = _coerce_evidence(
            assignment.claim_kind, assignment.tool_name, raw_evidence
        )
        _verify_evidence_upstream(
            _unsigned_record_for_evidence(
                assignment,
                input_evidence,
                evidence,
                runtime,
                assignments,
            ),
            assignment,
            query,
            assignments,
            registry,
            multi_product_executor,
        )
    except Exception as error:  # noqa: BLE001 - one failure remains in denominator
        stable_code = getattr(error, "code", None)
        code = (
            stable_code
            if isinstance(stable_code, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", stable_code)
            else "tool_error"
        )
        return _build_record(
            assignment,
            input_evidence,
            evidence=None,
            runtime=runtime,
            upstream=_upstream_binding(assignment, assignments),
            error=FormalRunError(
                code=code,
                error_type=type(error).__name__,
            ),
        )
    return _build_record(
        assignment,
        input_evidence,
        evidence=evidence,
        runtime=runtime,
        upstream=_upstream_binding(assignment, assignments),
        error=None,
    )


def _search_multi_product_with_runtime_guard(
    executor: MultiProductExecutor,
    image_path: Path,
    *,
    asset_id: str,
    expected_runtime_sha256: str,
) -> MultiProductResult:
    before = multi_product_runtime_sha256(executor)
    if before != expected_runtime_sha256:
        raise FormalEvaluationError(
            "multi-product runtime changed before formal execution"
        )
    try:
        return executor.search(image_path, asset_id=asset_id)
    finally:
        after = multi_product_runtime_sha256(executor)
        if after != before:
            raise FormalEvaluationError(
                "multi-product runtime changed during formal execution"
            )


def _unsigned_record_for_evidence(
    assignment: FormalGoldAssignment,
    input_evidence: FormalInputEvidence,
    evidence: FormalEvidence,
    runtime: FormalRuntimeEvidence,
    assignments: VerifiedFormalGoldAssignments,
) -> FormalToolRunRecord:
    return _build_record(
        assignment,
        input_evidence,
        evidence=evidence,
        runtime=runtime,
        upstream=_upstream_binding(assignment, assignments),
        error=None,
    )


def _build_record(
    assignment: FormalGoldAssignment,
    input_evidence: FormalInputEvidence,
    *,
    evidence: FormalEvidence | None,
    runtime: FormalRuntimeEvidence,
    upstream: FormalUpstreamBinding,
    error: FormalRunError | None,
) -> FormalToolRunRecord:
    evidence_value = _json_model(evidence) if evidence is not None else None
    unsigned = {
        "schema_version": 1,
        "query_id": assignment.query_id,
        "claim_kind": assignment.claim_kind,
        "tool_name": assignment.tool_name,
        "status": "error" if error is not None else "ok",
        "input": input_evidence.model_dump(mode="json"),
        "evidence": evidence_value,
        "evidence_sha256": (
            sha256_bytes(canonical_json_bytes(evidence_value))
            if evidence_value is not None
            else None
        ),
        "runtime": runtime.model_dump(mode="json"),
        "upstream": upstream.model_dump(mode="json"),
        "error": error.model_dump(mode="json") if error is not None else None,
    }
    return FormalToolRunRecord.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "record_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        )
    )


def _coerce_evidence(
    claim_kind: ClaimKind, tool_name: str, value: object
) -> FormalEvidence:
    if claim_kind in {
        "image_product_search",
        "text_product_search",
        "style_similar_search",
    }:
        evidence: FormalEvidence = ProductSearchTrace.model_validate_json(
            canonical_json_bytes(_json_model(value))
        )
    elif claim_kind == "kb_lookup":
        raw_hits = value if isinstance(value, (list, tuple)) else None
        if raw_hits is None:
            raise TypeError("KB evidence must be a sequence of typed hits")
        evidence = KBLookupEvidence.model_validate_json(
            canonical_json_bytes({"tool_name": tool_name, "hits": list(raw_hits)})
        )
    elif claim_kind == "object_detection":
        evidence = ObjectDetectionResult.model_validate_json(
            canonical_json_bytes(_json_model(value))
        )
    elif claim_kind == "document_ocr":
        evidence = DocumentOCRResult.model_validate_json(
            canonical_json_bytes(_json_model(value))
        )
    else:
        evidence = MultiProductResult.model_validate_json(
            canonical_json_bytes(_json_model(value))
        )
    _validate_evidence_kind(claim_kind, tool_name, evidence)
    return evidence


def _validate_evidence_kind(
    claim_kind: ClaimKind, tool_name: str, evidence: FormalEvidence
) -> None:
    expected: type[BaseModel]
    if claim_kind in {
        "image_product_search",
        "text_product_search",
        "style_similar_search",
    }:
        expected = ProductSearchTrace
        if not isinstance(evidence, expected) or evidence.tool_name != tool_name:
            raise ValueError("product evidence does not match the claim")
    elif claim_kind == "kb_lookup":
        if (
            not isinstance(evidence, KBLookupEvidence)
            or evidence.tool_name != tool_name
        ):
            raise ValueError("KB evidence does not match the claim")
    elif claim_kind == "object_detection":
        if not isinstance(evidence, ObjectDetectionResult):
            raise ValueError("detection claim requires ObjectDetectionResult")
    elif claim_kind == "document_ocr":
        if not isinstance(evidence, DocumentOCRResult):
            raise ValueError("OCR claim requires DocumentOCRResult")
    elif not isinstance(evidence, MultiProductResult):
        raise ValueError("multi-product claim requires MultiProductResult")


def _verify_assignment_semantics(
    assignments: tuple[FormalGoldAssignment, ...],
    queries: tuple[Query, ...],
    split_queries: tuple[Query, ...],
    catalog: AssetCatalog,
    gold: VerifiedBenchmarkGold,
) -> None:
    query_by_id = {item.query_id: item for item in queries}
    split_by_id = {item.query_id: item for item in split_queries}
    gold_by_id = gold.case_by_query_id()
    assignment_ids = {item.query_id for item in assignments}
    if assignment_ids != set(query_by_id) or assignment_ids != set(gold_by_id):
        raise FormalEvaluationError(
            "formal query/gold/assignment sets must be exactly equal"
        )
    missing_from_split = sorted(assignment_ids - set(split_by_id))
    if missing_from_split:
        raise FormalEvaluationError(
            f"formal queries missing from split assignment: {missing_from_split!r}"
        )
    group_benchmark_splits: dict[str, str] = {}
    for assignment in assignments:
        query = query_by_id[assignment.query_id]
        split_query = split_by_id[assignment.query_id]
        gold_case = gold_by_id[assignment.query_id]
        resolution = catalog.verify_reference(
            query.asset_id,
            query.image_path,
            leakage_group_id=query.leakage_group_id,
        )
        expected_asset_sha = resolution.asset.sha256
        expected_query_sha = sha256_bytes(
            canonical_json_bytes(query.model_dump(mode="json"))
        )
        if split_query != query:
            raise FormalEvaluationError(
                f"query is not the authoritative split row: {query.query_id}"
            )
        if (
            assignment.query_row_sha256,
            assignment.asset_id,
            assignment.asset_sha256,
            assignment.leakage_group_id,
            assignment.corpus_split,
        ) != (
            expected_query_sha,
            query.asset_id,
            expected_asset_sha,
            query.leakage_group_id,
            query.split,
        ):
            raise FormalEvaluationError(
                f"formal assignment query/asset mismatch: {query.query_id}"
            )
        if (
            assignment.benchmark_split,
            assignment.tool_name,
            assignment.arguments,
            assignment.gold_case_sha256,
        ) != (
            gold_case.split,
            gold_case.tool_name,
            gold_case.arguments,
            sha256_bytes(canonical_json_bytes(gold_case.model_dump(mode="json"))),
        ):
            raise FormalEvaluationError(
                f"formal assignment gold mismatch: {query.query_id}"
            )
        _validate_gold_task_for_claim(assignment.claim_kind, gold_case)
        _validate_arguments(assignment, query)
        prior = group_benchmark_splits.setdefault(
            assignment.leakage_group_id, assignment.benchmark_split
        )
        if prior != assignment.benchmark_split:
            raise FormalEvaluationError("leakage group crosses benchmark splits")


def _prepare_document_safety_catalog(
    assignments: tuple[FormalGoldAssignment, ...],
    asset_catalog: AssetCatalog,
    *,
    document_safety_catalog: DocumentSafetyCatalog | None,
    expected_catalog_sha256: str | None,
    expected_review_ledger_sha256: str | None,
) -> DocumentSafetyCatalog | None:
    """Reload the externally pinned privacy-review ledger for OCR assignments."""

    requires_catalog = any(item.claim_kind == "document_ocr" for item in assignments)
    supplied = (
        document_safety_catalog,
        expected_catalog_sha256,
        expected_review_ledger_sha256,
    )
    if not requires_catalog:
        if any(value is not None for value in supplied):
            raise FormalEvaluationError(
                "document safety catalog was supplied without an OCR assignment"
            )
        return None
    if any(value is None for value in supplied):
        raise FormalEvaluationError(
            "document OCR assignments require an externally pinned safety catalog"
        )
    if not isinstance(document_safety_catalog, DocumentSafetyCatalog):
        raise TypeError("document_safety_catalog must be a DocumentSafetyCatalog")
    assert expected_catalog_sha256 is not None
    assert expected_review_ledger_sha256 is not None
    expected_catalog_sha256 = _require_sha256(
        expected_catalog_sha256,
        "expected_document_safety_catalog_sha256",
    )
    expected_review_ledger_sha256 = _require_sha256(
        expected_review_ledger_sha256,
        "expected_document_safety_review_ledger_sha256",
    )
    try:
        refreshed = load_document_safety_catalog(
            document_safety_catalog.root,
            asset_catalog,
            expected_review_ledger_sha256=expected_review_ledger_sha256,
        )
    except DocumentSafetyCatalogError as error:
        raise FormalEvaluationError(
            "document safety catalog failed full re-verification"
        ) from error
    if refreshed.catalog_sha256 != expected_catalog_sha256:
        raise FormalEvaluationError(
            "document safety catalog does not match its external digest"
        )
    return refreshed


def _verify_document_safety_assignments(
    assignments: tuple[FormalGoldAssignment, ...],
    safety_catalog: DocumentSafetyCatalog | None,
) -> None:
    ocr_assignments = tuple(
        item for item in assignments if item.claim_kind == "document_ocr"
    )
    if not ocr_assignments:
        if safety_catalog is not None:
            raise FormalEvaluationError("unused document safety catalog")
        return
    if safety_catalog is None:
        raise FormalEvaluationError("OCR safety catalog is unavailable")
    expected_pairs = {(item.query_id, item.asset_id) for item in ocr_assignments}
    actual_pairs = {
        (record.query_id, record.asset_id) for record in safety_catalog.approvals
    }
    if actual_pairs != expected_pairs:
        raise FormalEvaluationError(
            "document safety catalog must exactly cover the OCR assignment set"
        )
    for assignment in ocr_assignments:
        approval = safety_catalog.approval_for(assignment.query_id, assignment.asset_id)
        if approval is None or (
            approval.approval_sha256,
            approval.asset_id,
            approval.image_sha256,
        ) != (
            assignment.document_safety_approval_sha256,
            assignment.asset_id,
            assignment.asset_sha256,
        ):
            raise FormalEvaluationError(
                f"document safety approval mismatch: {assignment.query_id}"
            )


def _validate_arguments(assignment: FormalGoldAssignment, query: Query) -> None:
    if assignment.claim_kind == "style_similar_search":
        expected = {"asset_id": query.asset_id, "query": query.text}
    elif assignment.claim_kind in {
        "image_product_search",
        "object_detection",
        "document_ocr",
        "multi_product_chain",
    }:
        expected = {"asset_id": query.asset_id}
    elif assignment.claim_kind == "text_product_search":
        expected = {"query": query.text}
    else:
        key = "entity" if assignment.tool_name == "encyclopedia_lookup" else "dish"
        if set(assignment.arguments) != {key}:
            raise FormalEvaluationError(
                "KB assignment must contain one derived-text argument"
            )
        value = assignment.arguments[key]
        if not isinstance(value, str) or not value.strip():
            raise FormalEvaluationError("KB assignment text must be non-blank")
        return
    if assignment.arguments != expected:
        raise FormalEvaluationError(
            f"formal assignment arguments are not authoritative: {assignment.query_id}"
        )


def _validate_gold_task_for_claim(claim_kind: ClaimKind, gold_case: GoldCase) -> None:
    expected = (
        "ranking"
        if claim_kind in _RANKING_CLAIMS
        else "detection"
        if claim_kind == "object_detection"
        else "ocr"
    )
    if gold_case.task != expected:
        raise FormalEvaluationError("claim kind and benchmark gold task mismatch")


def _build_run_spec(
    assignments: VerifiedFormalGoldAssignments,
    runtime_snapshot: FormalRegistryRuntimeSnapshot,
    *,
    run_id: str,
    multi_product_runtime_sha256: str | None,
    formal_tool_names: frozenset[str],
) -> FormalToolRunSpec:
    counts = Counter(item.claim_kind for item in assignments.records)
    tool_counts = Counter(item.tool_name for item in assignments.records)
    unsigned = {
        "schema_version": 1,
        "kind": "formal-tool-run-spec",
        "policy_version": FORMAL_TOOL_RUN_POLICY_VERSION,
        "run_id": run_id,
        "query_count": len(assignments.records),
        "claim_counts": {key: counts[key] for key in sorted(_CLAIM_TOOLS)},
        "tool_counts": {key: tool_counts[key] for key in sorted(formal_tool_names)},
        "registry_sha256": runtime_snapshot.registry_sha256,
        "registry_runtime_sha256": runtime_snapshot.registry_runtime_sha256,
        "multi_product_runtime_sha256": multi_product_runtime_sha256,
        "asset_catalog_sha256": assignments.catalog.catalog_sha256,
        "query_artifact_sha256": assignments.query_sha256,
        "split_assignment_sha256": assignments.split_assignment_sha256,
        "gold_assignment_sha256": assignments.sha256,
        "gold_sha256": assignments.gold.sha256,
        "gold_review_ledger_sha256": assignments.gold.review_ledger_sha256,
        "tool_assistant_isolation_sha256": (
            assignments.tool_assistant_isolation.isolation_sha256
        ),
        "document_safety_catalog_sha256": (
            assignments.expected_document_safety_catalog_sha256
        ),
        "document_safety_review_ledger_sha256": (
            assignments.expected_document_safety_review_ledger_sha256
        ),
    }
    return FormalToolRunSpec.model_validate(
        {**unsigned, "spec_sha256": sha256_bytes(canonical_json_bytes(unsigned))}
    )


def _build_run_manifest(
    spec: FormalToolRunSpec,
    records: tuple[FormalToolRunRecord, ...],
    spec_bytes: bytes,
    records_bytes: bytes,
) -> FormalToolRunManifest:
    success_count = sum(item.status == "ok" for item in records)
    unsigned = {
        "schema_version": 1,
        "kind": "formal-tool-run",
        "policy_version": FORMAL_TOOL_RUN_POLICY_VERSION,
        "run_id": spec.run_id,
        "status": "complete",
        "spec_sha256": spec.spec_sha256,
        "query_count": len(records),
        "success_count": success_count,
        "error_count": len(records) - success_count,
        "artifacts": {
            _RECORDS_FILE: RunArtifactDescriptor(
                path=_RECORDS_FILE,
                bytes=len(records_bytes),
                rows=len(records),
                sha256=sha256_bytes(records_bytes),
            ).model_dump(mode="json"),
            _SPEC_FILE: RunArtifactDescriptor(
                path=_SPEC_FILE,
                bytes=len(spec_bytes),
                rows=1,
                sha256=sha256_bytes(spec_bytes),
            ).model_dump(mode="json"),
        },
    }
    return FormalToolRunManifest.model_validate(
        {
            **unsigned,
            "run_bundle_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )


def _verify_run_descriptors(
    manifest: FormalToolRunManifest, files: Mapping[str, bytes]
) -> None:
    for name, descriptor in manifest.artifacts.items():
        content = files[name]
        if (
            len(content) != descriptor.bytes
            or sha256_bytes(content) != descriptor.sha256
        ):
            raise FormalEvaluationError(f"formal run artifact mismatch: {name}")


def _verify_spec_bindings(
    spec: FormalToolRunSpec,
    manifest: FormalToolRunManifest,
    assignments: VerifiedFormalGoldAssignments,
    runtime_snapshot: FormalRegistryRuntimeSnapshot,
    expected_multi_product_runtime_sha256: str | None,
    multi_product_executor: MultiProductExecutor | None,
) -> str | None:
    if (manifest.run_id, manifest.spec_sha256, manifest.query_count) != (
        spec.run_id,
        spec.spec_sha256,
        spec.query_count,
    ):
        raise FormalEvaluationError("formal run manifest/spec mismatch")
    if (
        spec.registry_sha256,
        spec.registry_runtime_sha256,
        spec.asset_catalog_sha256,
        spec.query_artifact_sha256,
        spec.split_assignment_sha256,
        spec.gold_assignment_sha256,
        spec.gold_sha256,
        spec.gold_review_ledger_sha256,
        spec.tool_assistant_isolation_sha256,
        spec.document_safety_catalog_sha256,
        spec.document_safety_review_ledger_sha256,
    ) != (
        runtime_snapshot.registry_sha256,
        runtime_snapshot.registry_runtime_sha256,
        assignments.catalog.catalog_sha256,
        assignments.query_sha256,
        assignments.split_assignment_sha256,
        assignments.sha256,
        assignments.gold.sha256,
        assignments.gold.review_ledger_sha256,
        assignments.tool_assistant_isolation.isolation_sha256,
        assignments.expected_document_safety_catalog_sha256,
        assignments.expected_document_safety_review_ledger_sha256,
    ):
        raise FormalEvaluationError("formal run upstream or registry binding mismatch")
    expected_tools = (
        _V2_FORMAL_TOOL_NAMES
        if "multi_product_search" in dict(runtime_snapshot.tool_runtime_bindings)
        else _LEGACY_FORMAL_TOOL_NAMES
    )
    if set(spec.tool_counts) != expected_tools:
        raise FormalEvaluationError(
            "formal run tool coverage does not match the registry runtime generation"
        )
    # Only the legacy pseudo-tool is governed by the out-of-registry runtime
    # authority.  Registry v2 binds the same semantic claim through the
    # ``multi_product_search`` ToolSpec and registry runtime snapshot.
    has_multi = spec.tool_counts.get("multi_product_chain", 0) > 0
    actual_multi = _prepare_multi_runtime(
        has_multi,
        multi_product_executor,
        expected_multi_product_runtime_sha256,
    )
    if spec.multi_product_runtime_sha256 != actual_multi:
        raise FormalEvaluationError("formal run multi-product runtime mismatch")
    return actual_multi


def _prepare_multi_runtime(
    required: bool,
    executor: MultiProductExecutor | None,
    expected_sha256: str | None,
) -> str | None:
    if not required:
        if executor is not None or expected_sha256 is not None:
            raise FormalEvaluationError("unused multi-product runtime was supplied")
        return None
    if executor is None or expected_sha256 is None:
        raise FormalEvaluationError(
            "multi-product claims require a live executor and external runtime digest"
        )
    expected = _require_sha256(expected_sha256, "expected_multi_product_runtime_sha256")
    actual = multi_product_runtime_sha256(executor)
    if actual != expected:
        raise FormalEvaluationError("multi-product runtime digest mismatch")
    return actual


def _verify_multi_runtime_snapshot(
    expected: str | None,
    executor: MultiProductExecutor | None,
) -> None:
    """Re-resolve the complete composition after all evidence was inspected."""

    if expected is None:
        if executor is not None:
            raise FormalEvaluationError("unused multi-product runtime was supplied")
        return
    if executor is None or multi_product_runtime_sha256(executor) != expected:
        raise FormalEvaluationError(
            "multi-product runtime changed during run verification"
        )


def multi_product_runtime_sha256(executor: MultiProductExecutor) -> str:
    """Derive the composition identity from both transitive live runtimes."""

    _require_builtin_multi_product_executor(executor)
    try:
        detector_binding = executor.detector.artifact.runtime_binding
        detector_runtime = executor.detector.formal_runtime_binding_sha256
        product_binding = executor.product_search.formal_runtime_binding_sha256
    except (AttributeError, TypeError, ValueError) as error:
        raise FormalEvaluationError(
            "multi-product executor is not formal-ready"
        ) from error
    if not isinstance(product_binding, str):
        raise FormalEvaluationError("multi-product product runtime identity is invalid")
    detector_runtime = _require_sha256(detector_runtime, "detector_runtime_sha256")
    product_binding = _require_sha256(product_binding, "product_runtime_sha256")
    payload = {
        "policy_version": "formal-multi-product-runtime-v1",
        "detector_artifact_runtime": _json_model(detector_binding),
        "detector_runtime_sha256": detector_runtime,
        "product_runtime_sha256": product_binding,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _require_builtin_multi_product_executor(executor: object) -> None:
    """Keep formal composition on the reviewed production implementation graph."""

    if type(executor) is not MultiProductSearchService:
        raise FormalEvaluationError(
            "formal multi-product runtime requires MultiProductSearchService"
        )
    detector = executor.detector
    product_search = executor.product_search
    if (
        type(detector) is not ObjectDetectionService
        or type(getattr(detector, "artifact", None)) is not VerifiedModelArtifact
        or type(getattr(detector, "backend", None)) is not UltralyticsDetectorBackend
        or "detect" in vars(detector)
        or "detect" in vars(detector.backend)
    ):
        raise FormalEvaluationError(
            "formal multi-product detector is not a trusted built-in runtime"
        )
    backend = getattr(product_search, "backend", None)
    if (
        type(product_search) is not ProductSearchService
        or type(getattr(product_search, "index", None)) is not ProductIndex
        or type(backend) is not FormalEmbeddingBackend
        or type(getattr(backend, "_backend", None))
        not in {DashScopeEmbeddingClient, OpenCLIPEmbeddingBackend}
        or "trace_image_product_search" in vars(product_search)
        or any(
            name in vars(backend)
            for name in (
                "embed_images",
                "embed_image_bytes",
                "embed_texts",
                "verify_canary",
            )
        )
        or any(
            name in vars(backend._backend)
            for name in ("embed_images", "embed_image_bytes", "embed_texts")
        )
        or "search" in vars(executor)
    ):
        raise FormalEvaluationError(
            "formal multi-product search is not a trusted built-in runtime"
        )


def _upstream_binding(
    assignment: FormalGoldAssignment,
    assignments: VerifiedFormalGoldAssignments,
) -> FormalUpstreamBinding:
    return FormalUpstreamBinding(
        asset_catalog_sha256=assignments.catalog.catalog_sha256,
        query_artifact_sha256=assignments.query_sha256,
        split_assignment_sha256=assignments.split_assignment_sha256,
        gold_assignment_sha256=assignments.sha256,
        gold_sha256=assignments.gold.sha256,
        gold_review_ledger_sha256=assignments.gold.review_ledger_sha256,
        tool_assistant_isolation_sha256=(
            assignments.tool_assistant_isolation.isolation_sha256
        ),
        document_safety_catalog_sha256=(
            assignments.expected_document_safety_catalog_sha256
        ),
        document_safety_review_ledger_sha256=(
            assignments.expected_document_safety_review_ledger_sha256
        ),
        leakage_policy_version=assignments.catalog.leakage_policy_version,
        leakage_group_id=assignment.leakage_group_id,
        corpus_split=assignment.corpus_split,
        benchmark_split=assignment.benchmark_split,
    )


def _benchmark_output(record: FormalToolRunRecord) -> Any:
    evidence = record.evidence
    if evidence is None:
        return None
    if isinstance(evidence, KBLookupEvidence):
        return [item.model_dump(mode="json") for item in evidence.hits]
    if isinstance(evidence, MultiProductResult):
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in evidence.objects:
            for hit in item.hits:
                identifier = hit.product.product_id
                if identifier not in seen:
                    seen.add(identifier)
                    hits.append(hit.model_dump(mode="json"))
        return {"hits": hits}
    return evidence.model_dump(mode="json")


def _evaluate_formal_record(
    record: FormalToolRunRecord,
    case: GoldCase,
) -> tuple[BenchmarkCaseResult, list[Any]]:
    if record.status == "error":
        error = BenchmarkCaseError(
            error_type=record.error.error_type if record.error else "ToolError",
            message=_PUBLIC_ERROR_MESSAGE,
        )
        return benchmark_module._evaluate_case(  # noqa: SLF001
            case,
            output=None,
            error=error,
        )
    try:
        return benchmark_module._evaluate_case(  # noqa: SLF001
            case,
            output=_benchmark_output(record),
            error=None,
        )
    except Exception as error:  # Invalid metric projection remains in denominator.
        captured = BenchmarkCaseError(
            error_type=type(error).__name__,
            message=_PUBLIC_ERROR_MESSAGE,
        )
        return benchmark_module._evaluate_case(  # noqa: SLF001
            case,
            output=None,
            error=captured,
        )


def _load_formal_evaluation(
    path: Path,
    run: VerifiedFormalToolRun,
    *,
    expected_manifest_file_sha256: str,
) -> VerifiedFormalEvaluation:
    expected_manifest_file_sha256 = _require_sha256(
        expected_manifest_file_sha256,
        "expected_manifest_file_sha256",
    )
    before = _read_exact_directory(path, _REPORT_FILES, "formal evaluation")
    if sha256_bytes(before[_REPORT_MANIFEST_FILE]) != expected_manifest_file_sha256:
        raise FormalEvaluationError(
            "formal evaluation manifest does not match external digest"
        )
    try:
        manifest = FormalEvaluationManifest.model_validate(
            parse_canonical_json(
                before[_REPORT_MANIFEST_FILE], label="formal evaluation manifest"
            )
        )
        rows = parse_canonical_jsonl(
            before[_REPORT_FILE], label="formal evaluation report"
        )
        results = _parse_models(rows, BenchmarkCaseResult, "formal evaluation result")
    except (ArtifactFormatError, ValidationError) as error:
        raise FormalEvaluationError(str(error)) from error
    if manifest.manifest_sha256 != _model_digest(manifest, "manifest_sha256"):
        raise FormalEvaluationError("formal evaluation manifest self-hash mismatch")
    if (
        manifest.run_bundle_sha256,
        manifest.gold_sha256,
        manifest.gold_assignment_sha256,
        manifest.gold_review_ledger_sha256,
        manifest.tool_assistant_isolation_sha256,
        manifest.registry_sha256,
        manifest.registry_runtime_sha256,
        manifest.document_safety_catalog_sha256,
        manifest.document_safety_review_ledger_sha256,
    ) != (
        run.manifest.run_bundle_sha256,
        run.assignments.gold.sha256,
        run.assignments.sha256,
        run.assignments.gold.review_ledger_sha256,
        run.assignments.tool_assistant_isolation.isolation_sha256,
        run.spec.registry_sha256,
        run.spec.registry_runtime_sha256,
        run.spec.document_safety_catalog_sha256,
        run.spec.document_safety_review_ledger_sha256,
    ):
        raise FormalEvaluationError("formal evaluation binding mismatch")
    if (
        len(before[_REPORT_FILE]) != manifest.report.bytes
        or sha256_bytes(before[_REPORT_FILE]) != manifest.report.sha256
    ):
        raise FormalEvaluationError("formal evaluation report descriptor mismatch")
    if len(results) != manifest.case_count:
        raise FormalEvaluationError("formal evaluation result count mismatch")
    expected_ids = [item.query_id for item in run.records]
    if [item.query_id for item in results] != expected_ids:
        raise FormalEvaluationError("formal evaluation query coverage mismatch")
    observations: list[Any] = []
    recomputed: list[BenchmarkCaseResult] = []
    gold_by_id = run.assignments.gold.case_by_query_id()
    record_by_id = {item.query_id: item for item in run.records}
    for result in results:
        record = record_by_id[result.query_id]
        expected, case_observations = _evaluate_formal_record(
            record,
            gold_by_id[result.query_id],
        )
        recomputed.append(expected)
        observations.extend(case_observations)
    if tuple(recomputed) != results:
        raise FormalEvaluationError("formal evaluation stored metrics mismatch")
    metrics = benchmark_module._aggregate_metrics(  # noqa: SLF001
        run.assignments.gold.cases, results, observations
    )
    by_split = benchmark_module._aggregate_metrics_by_split(  # noqa: SLF001
        run.assignments.gold.cases, results, observations
    )
    if metrics != manifest.metrics or by_split != manifest.metrics_by_split:
        raise FormalEvaluationError("formal evaluation aggregate metrics mismatch")
    expected_splits = {
        split: sum(case.split == split for case in run.assignments.gold.cases)
        for split in ("tool_dev", "tool_test_frozen")
    }
    expected_tasks = {
        task: sum(case.task == task for case in run.assignments.gold.cases)
        for task in ("ranking", "detection", "ocr")
    }
    success_count = sum(item.status == "ok" for item in results)
    if (
        manifest.split_counts != expected_splits
        or manifest.task_counts != expected_tasks
        or (manifest.success_count, manifest.error_count)
        != (success_count, len(results) - success_count)
    ):
        raise FormalEvaluationError("formal evaluation count summary mismatch")
    if _read_exact_directory(path, _REPORT_FILES, "formal evaluation") != before:
        raise FormalEvaluationError("formal evaluation changed during verification")
    return VerifiedFormalEvaluation(
        root=path.resolve(),
        manifest=manifest,
        results=results,
        run=run,
        manifest_file_sha256=expected_manifest_file_sha256,
        _marker=_VERIFIED_EVALUATION_MARKER,
    )


def _refresh_verified_run(run: VerifiedFormalToolRun) -> VerifiedFormalToolRun:
    """Deeply reparse a run so mutable objects in an old handle are never used."""

    run = require_verified_formal_tool_run(run)
    return _load_run_bundle(
        run.root,
        _refresh_assignments(run.assignments),
        run.registry,
        expected_run_bundle_sha256=run.manifest.run_bundle_sha256,
        expected_registry_sha256=run.spec.registry_sha256,
        expected_registry_runtime_sha256=run.spec.registry_runtime_sha256,
        multi_product_executor=run.multi_product_executor,
        expected_multi_product_runtime_sha256=(run.spec.multi_product_runtime_sha256),
        grant_handle=True,
    )


def _refresh_assignments(
    value: VerifiedFormalGoldAssignments,
) -> VerifiedFormalGoldAssignments:
    value = _require_assignments(value)
    return load_formal_gold_assignments(
        value.path,
        expected_assignment_sha256=value.sha256,
        gold_path=value.gold.path,
        expected_gold_sha256=value.gold.sha256,
        gold_review_ledger_path=value.gold.review_ledger_path,
        expected_gold_review_ledger_sha256=value.gold.review_ledger_sha256,
        query_path=value.query_path,
        expected_query_sha256=value.query_sha256,
        split_assignment_path=value.split_assignment_path,
        expected_split_assignment_sha256=value.split_assignment_sha256,
        catalog=value.catalog,
        expected_asset_catalog_sha256=value.catalog.catalog_sha256,
        document_safety_catalog=value.document_safety_catalog,
        expected_document_safety_catalog_sha256=(
            value.expected_document_safety_catalog_sha256
        ),
        expected_document_safety_review_ledger_sha256=(
            value.expected_document_safety_review_ledger_sha256
        ),
        assistant_inputs=value.tool_assistant_isolation.assistant_inputs,
    )


def _require_assignments(
    value: VerifiedFormalGoldAssignments,
) -> VerifiedFormalGoldAssignments:
    if (
        not isinstance(value, VerifiedFormalGoldAssignments)
        or value._marker is not _ASSIGNMENTS_MARKER
        or not isinstance(
            value.tool_assistant_isolation,
            VerifiedToolAssistantComponentIsolation,
        )
        or value.tool_assistant_isolation._marker
        is not _TOOL_ASSISTANT_ISOLATION_MARKER
    ):
        raise TypeError("verified formal gold assignment handle required")
    return value


def _verify_registry_identity(
    registry: ToolRegistry,
    expected_registry_sha256: str,
    expected_registry_runtime_sha256: str,
) -> FormalRegistryRuntimeSnapshot:
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    try:
        runtime_snapshot = require_formal_registry_runtime(registry).snapshot()
    except Exception as error:
        raise FormalEvaluationError(
            "formal evaluation requires explicit runtime bindings for all tools"
        ) from error
    expected_registry_sha256 = _require_sha256(
        expected_registry_sha256, "expected_registry_sha256"
    )
    expected_registry_runtime_sha256 = _require_sha256(
        expected_registry_runtime_sha256, "expected_registry_runtime_sha256"
    )
    if runtime_snapshot.registry_sha256 != expected_registry_sha256:
        raise FormalEvaluationError("live registry spec digest mismatch")
    if runtime_snapshot.registry_runtime_sha256 != expected_registry_runtime_sha256:
        raise FormalEvaluationError("live registry runtime digest mismatch")
    return runtime_snapshot


def _fresh_catalog(catalog: AssetCatalog, expected_sha256: str) -> AssetCatalog:
    if not isinstance(catalog, AssetCatalog):
        raise TypeError("catalog must be an AssetCatalog")
    catalog_chain = _snapshot_real_path_chain(
        Path(catalog.root), label="asset catalog", leaf_kind="directory"
    )
    asset_chain = _snapshot_real_path_chain(
        Path(catalog.asset_root), label="asset root", leaf_kind="directory"
    )
    fresh = load_asset_catalog(catalog.root, catalog.asset_root, verify_files=True)
    if fresh.catalog_sha256 != expected_sha256:
        raise FormalEvaluationError("asset catalog digest mismatch")
    if catalog_chain != _snapshot_real_path_chain(
        Path(catalog.root), label="asset catalog", leaf_kind="directory"
    ) or asset_chain != _snapshot_real_path_chain(
        Path(catalog.asset_root), label="asset root", leaf_kind="directory"
    ):
        raise FormalEvaluationError("asset catalog path changed during verification")
    return fresh


def _read_expected_canonical_jsonl(
    path: Path, expected_sha256: str, label: str
) -> bytes:
    try:
        content = _read_real_stable_file(path, label=label)
        values = parse_canonical_jsonl(content, label=label)
    except ArtifactFormatError as error:
        raise FormalEvaluationError(str(error)) from error
    if sha256_bytes(content) != expected_sha256:
        raise FormalEvaluationError(f"{label} does not match external digest")
    if content != canonical_jsonl_bytes(values):
        raise FormalEvaluationError(f"{label} is not canonical JSONL")
    return content


def _parse_models(
    values: Sequence[object], model: type[BaseModel], label: str
) -> tuple[Any, ...]:
    parsed: list[BaseModel] = []
    for ordinal, value in enumerate(values):
        try:
            parsed.append(model.model_validate_json(canonical_json_bytes(value)))
        except ValidationError as error:
            raise FormalEvaluationError(
                f"invalid {label} row {ordinal}: {error}"
            ) from error
    return tuple(parsed)


def _reject_duplicate_ids(values: Sequence[Any], label: str) -> None:
    identifiers = [item.query_id for item in values]
    if len(identifiers) != len(set(identifiers)):
        raise FormalEvaluationError(f"{label} contains duplicate query_id")


def _read_exact_directory(
    path: Path, allowed: frozenset[str], label: str
) -> dict[str, bytes]:
    before_chain = _snapshot_real_path_chain(path, label=label, leaf_kind="directory")
    try:
        names = {item.name for item in path.iterdir()}
    except OSError as error:
        raise FormalEvaluationError(f"{label} directory cannot be listed") from error
    if names != allowed:
        raise FormalEvaluationError(f"{label} file set is invalid: {sorted(names)!r}")
    result: dict[str, bytes] = {}
    for name in sorted(names):
        try:
            result[name] = _read_real_stable_file(path / name, label=f"{label} {name}")
        except ArtifactFormatError as error:
            raise FormalEvaluationError(str(error)) from error
    after_chain = _snapshot_real_path_chain(path, label=label, leaf_kind="directory")
    if after_chain != before_chain:
        raise FormalEvaluationError(f"{label} ancestor chain changed during read")
    return result


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _snapshot_real_path_chain(
    path: Path,
    *,
    label: str,
    leaf_kind: Literal["file", "directory"],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Reject symlink/junction traversal and snapshot every path component."""

    absolute = Path(os.path.abspath(path))
    chain = [absolute, *absolute.parents]
    snapshots: list[tuple[str, int, int, int, int, int]] = []
    for index, member in enumerate(reversed(chain)):
        try:
            metadata = member.lstat()
        except OSError as error:
            raise FormalEvaluationError(f"{label} path cannot be inspected") from error
        if _is_link_or_reparse(metadata):
            raise FormalEvaluationError(
                f"{label} path must not traverse a symlink, junction, or reparse point"
            )
        is_leaf = index == len(chain) - 1
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise FormalEvaluationError(f"{label} ancestor must be a directory")
        if is_leaf:
            expected = stat.S_ISREG if leaf_kind == "file" else stat.S_ISDIR
            if not expected(metadata.st_mode):
                raise FormalEvaluationError(f"{label} must be a real {leaf_kind}")
            if leaf_kind == "file" and int(metadata.st_nlink) != 1:
                raise FormalEvaluationError(f"{label} must not be a hard-linked file")
        snapshots.append(
            (
                os.path.normcase(str(member)),
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_mode),
                int(metadata.st_size) if is_leaf and leaf_kind == "file" else 0,
                int(metadata.st_mtime_ns) if is_leaf and leaf_kind == "file" else 0,
            )
        )
    return tuple(snapshots)


def _read_real_stable_file(path: Path, *, label: str) -> bytes:
    before = _snapshot_real_path_chain(path, label=label, leaf_kind="file")
    try:
        content = read_stable_regular_file(path, label=label)
    except ArtifactFormatError as error:
        raise FormalEvaluationError(str(error)) from error
    after = _snapshot_real_path_chain(path, label=label, leaf_kind="file")
    if after != before:
        raise FormalEvaluationError(f"{label} path changed during read")
    return content


def _model_digest(model: BaseModel, field_name: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(field_name)
    return sha256_bytes(canonical_json_bytes(payload))


def _json_model(value: object) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _require_sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalEvaluationError(f"{field_name} must be a lowercase sha256")
    return value


def _raise_context_violation() -> Path:
    raise FormalEvaluationError("registry requested a non-authoritative asset")


def _remove_owned_staging(staging: Path, destination_parent: Path) -> None:
    if not os.path.lexists(staging):
        return
    try:
        if staging.parent.resolve() != destination_parent.resolve():
            return
    except OSError:
        return
    if ".staging-" not in staging.name:
        return
    shutil.rmtree(staging)


__all__ = [
    "FORMAL_EVALUATION_POLICY_VERSION",
    "FORMAL_TOOL_RUN_POLICY_VERSION",
    "CreatedFormalToolRun",
    "FormalEvaluationError",
    "FormalEvaluationManifest",
    "FormalGoldAssignment",
    "FormalToolRunManifest",
    "FormalToolRunRecord",
    "FormalToolRunSpec",
    "VerifiedFormalEvaluation",
    "VerifiedFormalGoldAssignments",
    "VerifiedFormalToolRun",
    "VerifiedToolAssistantComponentIsolation",
    "create_formal_tool_run",
    "evaluate_formal_tool_run",
    "load_formal_gold_assignments",
    "load_verified_formal_evaluation",
    "load_verified_formal_tool_run",
    "multi_product_runtime_sha256",
    "require_verified_formal_evaluation",
    "require_verified_formal_tool_run",
    "verify_tool_assistant_component_isolation",
]
