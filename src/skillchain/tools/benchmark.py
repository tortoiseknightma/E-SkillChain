"""Strict, offline, tool-independent diagnostic benchmark evidence.

The benchmark boundary deliberately separates *what must be evaluated* from
tool execution. Formal gold JSONL is accepted only when its bytes match an
external digest and its split/group identities exactly match authoritative
query assignments. Executor and raw-prediction results remain explicitly
diagnostic. Every query remains in metric denominators even when the executor
raises or returns malformed output.

Detection ``ap50`` in this module is a label-aware, micro, all-point AP at one
IoU threshold (0.5).  It is intentionally **not** COCO mAP.  OCR CER iterates
Python strings directly, so its unit is the Unicode code point; no implicit
Unicode normalization is applied.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
    validate_json_value,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BenchmarkSplit = Literal["tool_dev", "tool_test_frozen"]
TaskKind = Literal["ranking", "detection", "ocr"]

BENCHMARK_POLICY_VERSION = "offline-tool-benchmark-v2"
GOLD_REVIEW_POLICY_VERSION = "formal-gold-human-review-v1"
_RESULTS_FILE = "results.jsonl"
_MANIFEST_FILE = "manifest.json"
_ALLOWED_RESULT_FILES = {_RESULTS_FILE, _MANIFEST_FILE}
_VERIFIED_GOLD_MARKER = object()
_DIAGNOSTIC_GOLD_MARKER = object()
_DIAGNOSTIC_RESULT_MARKER = object()


class BenchmarkError(ValueError):
    """Gold, predictions, or persisted benchmark evidence is invalid."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and already stripped")
    return value


def _validate_path(value: str, field_name: str) -> str:
    _nonblank(value, field_name)
    parts = value.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ValueError(f"{field_name} must be a dotted identifier path")
    return value


class GoldDetection(_FrozenStrictModel):
    label: str
    bbox_xyxy: list[float]

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        return _nonblank(value, "label")

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        _validate_bbox(value)
        return value


class GoldOCRField(_FrozenStrictModel):
    field_name: str
    expected_value: str

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, value: str) -> str:
        return _nonblank(value, "field_name")


class _GoldCaseBase(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    task: TaskKind
    query_id: str
    tool_name: str
    split: BenchmarkSplit
    leakage_group_id: str
    arguments: dict[str, Any]

    @field_validator("query_id", "tool_name", "leakage_group_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            validate_json_value(value)
        except ArtifactFormatError as error:
            raise ValueError(str(error)) from error
        return value


class RankingGoldCase(_GoldCaseBase):
    task: Literal["ranking"] = "ranking"
    relevant_ids: list[str]
    k: int = Field(gt=0, le=1000)
    result_id_path: str
    result_list_path: str | None = None

    @field_validator("relevant_ids")
    @classmethod
    def validate_relevant_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("relevant_ids must not be empty")
        for item in value:
            _nonblank(item, "relevant_id")
        if len(value) != len(set(value)):
            raise ValueError("relevant_ids must be unique")
        return value

    @field_validator("result_id_path")
    @classmethod
    def validate_id_path(cls, value: str) -> str:
        return _validate_path(value, "result_id_path")

    @field_validator("result_list_path")
    @classmethod
    def validate_list_path(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_path(value, "result_list_path")
        return value


class DetectionGoldCase(_GoldCaseBase):
    task: Literal["detection"] = "detection"
    expected_detections: list[GoldDetection]
    iou_threshold: Literal[0.5] = 0.5

    @field_validator("expected_detections")
    @classmethod
    def validate_expected(cls, value: list[GoldDetection]) -> list[GoldDetection]:
        if not value:
            raise ValueError("expected_detections must not be empty")
        return value


class OCRGoldCase(_GoldCaseBase):
    task: Literal["ocr"] = "ocr"
    expected_text: str
    expected_fields: list[GoldOCRField]

    @field_validator("expected_text")
    @classmethod
    def validate_expected_text(cls, value: str) -> str:
        if not value:
            raise ValueError("expected_text must not be empty")
        return value

    @field_validator("expected_fields")
    @classmethod
    def validate_expected_fields(cls, value: list[GoldOCRField]) -> list[GoldOCRField]:
        if not value:
            raise ValueError("expected_fields must not be empty")
        names = [item.field_name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("expected OCR field names must be unique")
        return value


GoldCase = RankingGoldCase | DetectionGoldCase | OCRGoldCase


class QueryAssignment(_FrozenStrictModel):
    """Authoritative split assignment kept outside the benchmark gold."""

    query_id: str
    leakage_group_id: str
    split: BenchmarkSplit

    @field_validator("query_id", "leakage_group_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class GoldReviewEvidence(_FrozenStrictModel):
    """Immutable evidence cited by a human review of one gold case."""

    evidence_id: str
    evidence_uri: str
    evidence_revision: str
    evidence_sha256: Sha256
    basis: str

    @field_validator("evidence_id", "evidence_revision", "basis")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("evidence_uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        value = _nonblank(value, "evidence_uri")
        parsed = urlsplit(value)
        if parsed.scheme not in {"file", "https", "urn"}:
            raise ValueError("evidence_uri must use file, https, or urn")
        if parsed.scheme == "https" and not parsed.netloc:
            raise ValueError("https evidence_uri requires an authority")
        if parsed.scheme in {"file", "urn"} and not parsed.path:
            raise ValueError("evidence_uri requires an immutable resource identity")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("evidence_uri must not contain credentials")
        return value


class GoldCaseReview(_FrozenStrictModel):
    """Accountable, result-blind human approval for one exact gold row."""

    query_id: str
    gold_case_sha256: Sha256
    decision: Literal["approved"] = "approved"
    reviewer_kind: Literal["human"] = "human"
    reviewer_id: str
    reviewed_at_utc: str
    blind_to_system_results: Literal[True]
    review_basis: str
    evidence: tuple[GoldReviewEvidence, ...]

    @field_validator("query_id", "reviewer_id", "review_basis")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("reviewed_at_utc")
    @classmethod
    def validate_review_time(cls, value: str) -> str:
        value = _nonblank(value, "reviewed_at_utc")
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError(
                "reviewed_at_utc must be a second-precision UTC timestamp ending in Z"
            ) from error
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_json_evidence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(
        cls, value: tuple[GoldReviewEvidence, ...]
    ) -> tuple[GoldReviewEvidence, ...]:
        if not value:
            raise ValueError("gold review requires at least one evidence item")
        identities = tuple(item.evidence_id for item in value)
        if identities != tuple(sorted(identities)) or len(identities) != len(
            set(identities)
        ):
            raise ValueError("gold review evidence IDs must be unique and sorted")
        return value


class GoldReviewLedger(_FrozenStrictModel):
    """Externally digest-locked human-review ledger for exact gold bytes."""

    schema_version: Literal[1] = 1
    kind: Literal["formal-gold-review-ledger"] = "formal-gold-review-ledger"
    policy_version: Literal["formal-gold-human-review-v1"] = GOLD_REVIEW_POLICY_VERSION
    gold_sha256: Sha256
    cases: tuple[GoldCaseReview, ...]

    @field_validator("cases", mode="before")
    @classmethod
    def coerce_json_cases(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls, value: tuple[GoldCaseReview, ...]
    ) -> tuple[GoldCaseReview, ...]:
        if not value:
            raise ValueError("gold review ledger must not be empty")
        query_ids = tuple(item.query_id for item in value)
        if query_ids != tuple(sorted(query_ids)) or len(query_ids) != len(
            set(query_ids)
        ):
            raise ValueError("gold review cases must have unique sorted query IDs")
        return value


class BenchmarkBindings(_FrozenStrictModel):
    """Immutable identities required to interpret one benchmark run."""

    registry_sha256: Sha256
    tool_binding_sha256: Sha256
    runtime_binding_sha256: Sha256
    run_binding_sha256: Sha256
    network_policy: Literal["offline"] = "offline"


class BenchmarkCaseError(_FrozenStrictModel):
    error_type: str
    message: str

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str) -> str:
        return _nonblank(value, "error_type")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if len(value) > 1000:
            raise ValueError("error message exceeds 1000 characters")
        return value


class RankingCaseMetrics(_FrozenStrictModel):
    metric_kind: Literal["ranking"] = "ranking"
    k: int = Field(gt=0)
    relevant_count: int = Field(gt=0)
    retrieved_count: int = Field(ge=0)
    true_positive_at_k: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)


class DetectionCaseMetrics(_FrozenStrictModel):
    metric_kind: Literal["detection"] = "detection"
    iou_threshold: Literal[0.5] = 0.5
    gold_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.true_positive + self.false_negative != self.gold_count:
            raise ValueError("detection TP/FN do not cover gold objects")
        if self.true_positive + self.false_positive != self.prediction_count:
            raise ValueError("detection TP/FP do not cover predictions")
        return self


class OCRCaseMetrics(_FrozenStrictModel):
    metric_kind: Literal["ocr"] = "ocr"
    reference_codepoints: int = Field(gt=0)
    predicted_codepoints: int = Field(ge=0)
    edit_distance: int = Field(ge=0)
    expected_field_count: int = Field(gt=0)
    exact_field_count: int = Field(ge=0)
    evidenced_field_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.exact_field_count > self.expected_field_count:
            raise ValueError("exact OCR field count exceeds denominator")
        if self.evidenced_field_count > self.expected_field_count:
            raise ValueError("evidenced OCR field count exceeds denominator")
        return self


CaseMetrics = RankingCaseMetrics | DetectionCaseMetrics | OCRCaseMetrics


class BenchmarkCaseResult(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    query_id: str
    task: TaskKind
    split: BenchmarkSplit
    leakage_group_id: str
    status: Literal["ok", "error"]
    error: BenchmarkCaseError | None
    output: Any
    output_sha256: Sha256 | None
    metrics: CaseMetrics
    result_sha256: Sha256

    @field_validator("query_id", "leakage_group_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: Any) -> Any:
        try:
            validate_json_value(value)
        except ArtifactFormatError as error:
            raise ValueError(str(error)) from error
        return value

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "ok":
            if self.error is not None or self.output_sha256 is None:
                raise ValueError("successful result requires output hash and no error")
        elif (
            self.error is None
            or self.output is not None
            or self.output_sha256 is not None
        ):
            raise ValueError("error result requires error and null output/hash")
        if self.metrics.metric_kind != self.task:
            raise ValueError("case metric kind does not match task")
        return self


class RankingAggregate(_FrozenStrictModel):
    k: int = Field(gt=0)
    case_count: int = Field(gt=0)
    error_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)


class DetectionAggregate(_FrozenStrictModel):
    metric_definition: Literal[
        "label-aware-micro-all-point-ap@iou=0.5;not-coco-map"
    ] = "label-aware-micro-all-point-ap@iou=0.5;not-coco-map"
    iou_threshold: Literal[0.5] = 0.5
    case_count: int = Field(gt=0)
    error_count: int = Field(ge=0)
    gold_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    ap50: float = Field(ge=0.0, le=1.0)


class OCRAggregate(_FrozenStrictModel):
    cer_unit: Literal["unicode-codepoint"] = "unicode-codepoint"
    unicode_normalization: Literal["none"] = "none"
    case_count: int = Field(gt=0)
    error_count: int = Field(ge=0)
    reference_codepoints: int = Field(gt=0)
    predicted_codepoints: int = Field(ge=0)
    edit_distance: int = Field(ge=0)
    cer: float = Field(ge=0.0)
    expected_field_count: int = Field(gt=0)
    exact_field_count: int = Field(ge=0)
    field_exact: float = Field(ge=0.0, le=1.0)
    evidenced_field_count: int = Field(ge=0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)


class BenchmarkAggregateMetrics(_FrozenStrictModel):
    ranking: RankingAggregate | None
    detection: DetectionAggregate | None
    ocr: OCRAggregate | None


class BenchmarkArtifactDescriptor(_FrozenStrictModel):
    path: Literal["results.jsonl"] = _RESULTS_FILE
    bytes: int = Field(ge=0)
    rows: int = Field(gt=0)
    sha256: Sha256


class BenchmarkManifest(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["offline-tool-benchmark"] = "offline-tool-benchmark"
    policy_version: Literal["offline-tool-benchmark-v2"] = BENCHMARK_POLICY_VERSION
    run_id: str
    status: Literal["complete"] = "complete"
    assurance_level: Literal["diagnostic"] = "diagnostic"
    gold_sha256: Sha256
    query_set_sha256: Sha256
    registry_sha256: Sha256
    tool_binding_sha256: Sha256
    runtime_binding_sha256: Sha256
    run_binding_sha256: Sha256
    network_policy: Literal["offline"] = "offline"
    case_count: int = Field(gt=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    metrics: BenchmarkAggregateMetrics
    metrics_by_split: dict[str, BenchmarkAggregateMetrics]
    results: BenchmarkArtifactDescriptor
    manifest_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "run_id")

    @model_validator(mode="after")
    def validate_counts(self):
        if self.success_count + self.error_count != self.case_count:
            raise ValueError("success/error counts must cover every gold case")
        if set(self.split_counts) != {"tool_dev", "tool_test_frozen"}:
            raise ValueError("split_counts must contain both tool splits")
        if set(self.task_counts) != {"ranking", "detection", "ocr"}:
            raise ValueError("task_counts must contain all supported tasks")
        if set(self.metrics_by_split) != {"tool_dev", "tool_test_frozen"}:
            raise ValueError("metrics_by_split must contain both tool splits")
        if any(value < 0 for value in self.split_counts.values()):
            raise ValueError("split counts must be non-negative")
        if any(value < 0 for value in self.task_counts.values()):
            raise ValueError("task counts must be non-negative")
        if sum(self.split_counts.values()) != self.case_count:
            raise ValueError("split counts do not cover cases")
        if sum(self.task_counts.values()) != self.case_count:
            raise ValueError("task counts do not cover cases")
        if self.results.rows != self.case_count:
            raise ValueError("results row count does not cover cases")
        for task in ("ranking", "detection", "ocr"):
            if (getattr(self.metrics, task) is None) != (self.task_counts[task] == 0):
                raise ValueError(f"aggregate {task} metrics/count mismatch")
        return self


class OfflineToolExecutor(Protocol):
    """Executor supplied by the caller; this module itself performs no I/O calls."""

    network_policy: Literal["offline"]

    def __call__(self, tool_name: str, arguments: Mapping[str, Any]) -> object: ...


@dataclass(frozen=True)
class VerifiedBenchmarkGold:
    path: Path
    sha256: str
    query_set_sha256: str
    review_ledger_path: Path
    review_ledger_sha256: str
    review_ledger: GoldReviewLedger
    cases: tuple[GoldCase, ...]
    _marker: object

    def case_by_query_id(self) -> dict[str, GoldCase]:
        return {case.query_id: case for case in self.cases}


@dataclass(frozen=True)
class DiagnosticBenchmarkGold:
    """Untrusted gold suitable only for local diagnostics."""

    path: Path
    sha256: str
    query_set_sha256: str
    cases: tuple[GoldCase, ...]
    _marker: object

    def case_by_query_id(self) -> dict[str, GoldCase]:
        return {case.query_id: case for case in self.cases}


@dataclass(frozen=True)
class DiagnosticBenchmarkResult:
    """Internally consistent metrics with no claim of formal verification."""

    path: Path
    manifest: BenchmarkManifest
    results: tuple[BenchmarkCaseResult, ...]
    _marker: object


@dataclass(frozen=True)
class VerifiedBenchmarkResult:
    """Reserved for a future evaluator over externally trusted output artifacts.

    The executor and raw-prediction paths in this module deliberately never
    construct this type.
    """

    path: Path
    manifest: BenchmarkManifest
    results: tuple[BenchmarkCaseResult, ...]
    _marker: object


@dataclass(frozen=True)
class _APObservation:
    confidence: float
    query_id: str
    prediction_ordinal: int
    is_true_positive: bool


def load_benchmark_gold(
    path: str | Path,
    *,
    expected_gold_sha256: str,
    authoritative_assignments: Sequence[QueryAssignment | Mapping[str, Any]],
    review_ledger_path: str | Path | None = None,
    expected_review_ledger_sha256: str | None = None,
) -> VerifiedBenchmarkGold:
    """Load formal gold using independent gold, assignment, and review locks.

    A digest of the gold supplied by the caller is not evidence of human
    review.  A formal handle is granted only when a second externally pinned,
    canonical ledger approves every exact gold row and records accountable,
    result-blind human review evidence.
    """

    if not isinstance(expected_gold_sha256, str) or not _is_sha256(
        expected_gold_sha256
    ):
        raise BenchmarkError("expected_gold_sha256 must be a lowercase sha256")
    content, ordered = _load_gold_cases(path)
    actual_sha256 = sha256_bytes(content)
    if actual_sha256 != expected_gold_sha256:
        raise BenchmarkError("benchmark gold does not match expected_gold_sha256")
    assignments = _validate_authoritative_assignments(authoritative_assignments)

    by_query = {case.query_id: case for case in ordered}
    expected_ids = set(assignments)
    seen = set(by_query)
    missing = sorted(expected_ids - seen)
    extra = sorted(seen - expected_ids)
    if missing or extra:
        raise BenchmarkError(
            f"gold query set mismatch; missing={missing!r}, extra={extra!r}"
        )
    for query_id, assignment in assignments.items():
        case = by_query[query_id]
        if (case.leakage_group_id, case.split) != (
            assignment.leakage_group_id,
            assignment.split,
        ):
            raise BenchmarkError(f"gold assignment mismatch for query_id: {query_id}")
    frozen_tasks = {case.task for case in ordered if case.split == "tool_test_frozen"}
    missing_frozen_tasks = sorted(set(("ranking", "detection", "ocr")) - frozen_tasks)
    if missing_frozen_tasks:
        raise BenchmarkError(
            "formal gold tool_test_frozen coverage is incomplete; "
            f"missing_tasks={missing_frozen_tasks!r}"
        )
    review_path, review_sha256, review_ledger, review_snapshot = (
        _load_gold_review_ledger(
            review_ledger_path,
            expected_review_ledger_sha256=expected_review_ledger_sha256,
            gold_sha256=actual_sha256,
            cases=ordered,
        )
    )
    query_set_sha256 = sha256_bytes(
        canonical_json_bytes(
            [
                assignment.model_dump(mode="json")
                for assignment in sorted(
                    assignments.values(), key=lambda item: item.query_id
                )
            ]
        )
    )
    try:
        if read_stable_regular_file(Path(path), label="benchmark gold") != content:
            raise BenchmarkError("benchmark gold changed during verification")
        if (
            read_stable_regular_file(review_path, label="gold review ledger")
            != review_snapshot
        ):
            raise BenchmarkError("gold review ledger changed during verification")
    except ArtifactFormatError as error:
        raise BenchmarkError(str(error)) from error
    return VerifiedBenchmarkGold(
        path=Path(path).resolve(),
        sha256=actual_sha256,
        query_set_sha256=query_set_sha256,
        review_ledger_path=review_path.resolve(),
        review_ledger_sha256=review_sha256,
        review_ledger=review_ledger,
        cases=ordered,
        _marker=_VERIFIED_GOLD_MARKER,
    )


def _load_gold_review_ledger(
    path: str | Path | None,
    *,
    expected_review_ledger_sha256: str | None,
    gold_sha256: str,
    cases: tuple[GoldCase, ...],
) -> tuple[Path, str, GoldReviewLedger, bytes]:
    if path is None or expected_review_ledger_sha256 is None:
        raise BenchmarkError(
            "formal gold requires an externally digest-locked GoldReviewLedger"
        )
    if not isinstance(expected_review_ledger_sha256, str) or not _is_sha256(
        expected_review_ledger_sha256
    ):
        raise BenchmarkError("expected_review_ledger_sha256 must be a lowercase sha256")
    ledger_path = Path(path)
    try:
        content = read_stable_regular_file(ledger_path, label="gold review ledger")
        if sha256_bytes(content) != expected_review_ledger_sha256:
            raise BenchmarkError("gold review ledger does not match external digest")
        value = parse_canonical_json(content, label="gold review ledger")
        ledger = GoldReviewLedger.model_validate(value)
    except (ArtifactFormatError, ValidationError) as error:
        raise BenchmarkError(f"invalid gold review ledger: {error}") from error
    if content != canonical_json_bytes(ledger.model_dump(mode="json")):
        raise BenchmarkError("gold review ledger must be canonical typed JSON")
    if ledger.gold_sha256 != gold_sha256:
        raise BenchmarkError("gold review ledger is bound to different gold bytes")

    expected = {
        case.query_id: sha256_bytes(canonical_json_bytes(case.model_dump(mode="json")))
        for case in cases
    }
    reviewed = {item.query_id: item.gold_case_sha256 for item in ledger.cases}
    missing = sorted(set(expected) - set(reviewed))
    extra = sorted(set(reviewed) - set(expected))
    if missing or extra:
        raise BenchmarkError(
            f"gold review query set mismatch; missing={missing!r}, extra={extra!r}"
        )
    mismatched = sorted(
        query_id
        for query_id, case_sha256 in expected.items()
        if reviewed[query_id] != case_sha256
    )
    if mismatched:
        raise BenchmarkError(f"gold review case digest mismatch: {mismatched!r}")
    return ledger_path, expected_review_ledger_sha256, ledger, content


def load_diagnostic_benchmark_gold(
    path: str | Path,
    *,
    expected_query_ids: Sequence[str],
) -> DiagnosticBenchmarkGold:
    """Load self-asserted gold for diagnostics without a formal trust claim."""

    expected = list(expected_query_ids)
    if not expected:
        raise BenchmarkError("expected_query_ids must not be empty")
    if any(not isinstance(item, str) for item in expected):
        raise BenchmarkError("expected_query_ids must contain only strings")
    for item in expected:
        try:
            _nonblank(item, "expected_query_id")
        except ValueError as error:
            raise BenchmarkError(str(error)) from error
    if len(expected) != len(set(expected)):
        raise BenchmarkError("expected_query_ids contains duplicates")

    content, ordered = _load_gold_cases(path)
    seen = {case.query_id for case in ordered}
    expected_set = set(expected)
    missing = sorted(expected_set - seen)
    extra = sorted(seen - expected_set)
    if missing or extra:
        raise BenchmarkError(
            f"gold query set mismatch; missing={missing!r}, extra={extra!r}"
        )
    query_set_sha256 = sha256_bytes(canonical_json_bytes(sorted(seen)))
    return DiagnosticBenchmarkGold(
        path=Path(path).resolve(),
        sha256=sha256_bytes(content),
        query_set_sha256=query_set_sha256,
        cases=ordered,
        _marker=_DIAGNOSTIC_GOLD_MARKER,
    )


def _load_gold_cases(path: str | Path) -> tuple[bytes, tuple[GoldCase, ...]]:
    source_path = Path(path)
    try:
        content = read_stable_regular_file(source_path, label="benchmark gold")
        rows = parse_canonical_jsonl(content, label="benchmark gold")
    except ArtifactFormatError as error:
        raise BenchmarkError(str(error)) from error
    if not rows:
        raise BenchmarkError("benchmark gold must not be empty")

    cases: list[GoldCase] = []
    seen: set[str] = set()
    group_splits: dict[str, BenchmarkSplit] = {}
    ranking_k: int | None = None
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkError(f"gold row {ordinal} must be an object")
        task = row.get("task")
        model = {
            "ranking": RankingGoldCase,
            "detection": DetectionGoldCase,
            "ocr": OCRGoldCase,
        }.get(task)
        if model is None:
            raise BenchmarkError(f"gold row {ordinal} has unsupported task")
        try:
            case = model.model_validate(row)
        except ValidationError as error:
            raise BenchmarkError(f"invalid gold row {ordinal}: {error}") from error
        if case.query_id in seen:
            raise BenchmarkError(f"duplicate gold query_id: {case.query_id}")
        seen.add(case.query_id)
        prior_split = group_splits.setdefault(case.leakage_group_id, case.split)
        if prior_split != case.split:
            raise BenchmarkError(
                "leakage group crosses tool splits: "
                f"{case.leakage_group_id} ({prior_split}, {case.split})"
            )
        if isinstance(case, RankingGoldCase):
            ranking_k = case.k if ranking_k is None else ranking_k
            if case.k != ranking_k:
                raise BenchmarkError("all ranking gold cases must use the same K")
        cases.append(case)

    ordered = tuple(sorted(cases, key=lambda item: item.query_id))
    return content, ordered


def _validate_authoritative_assignments(
    values: Sequence[QueryAssignment | Mapping[str, Any]],
) -> dict[str, QueryAssignment]:
    if not values:
        raise BenchmarkError("authoritative_assignments must not be empty")
    result: dict[str, QueryAssignment] = {}
    for ordinal, value in enumerate(values):
        try:
            assignment = (
                value
                if isinstance(value, QueryAssignment)
                else QueryAssignment.model_validate(value)
            )
        except ValidationError as error:
            raise BenchmarkError(
                f"invalid authoritative assignment {ordinal}: {error}"
            ) from error
        if assignment.query_id in result:
            raise BenchmarkError("authoritative_assignments contains duplicates")
        result[assignment.query_id] = assignment
    group_splits: dict[str, BenchmarkSplit] = {}
    for assignment in result.values():
        prior = group_splits.setdefault(assignment.leakage_group_id, assignment.split)
        if prior != assignment.split:
            raise BenchmarkError("authoritative leakage group crosses tool splits")
    return result


def run_offline_benchmark(
    gold: VerifiedBenchmarkGold | DiagnosticBenchmarkGold,
    executor: OfflineToolExecutor,
    destination: str | Path,
    *,
    run_id: str,
    bindings: BenchmarkBindings,
) -> DiagnosticBenchmarkResult:
    """Execute every gold case and publish an explicitly diagnostic result."""

    _require_benchmark_gold(gold)
    _nonblank(run_id, "run_id")
    if getattr(executor, "network_policy", None) != "offline":
        raise BenchmarkError(
            "benchmark executor must explicitly declare network_policy='offline'"
        )
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"benchmark destination already exists: {destination}")
    _assert_gold_unchanged(gold)

    results: list[BenchmarkCaseResult] = []
    observations: list[_APObservation] = []
    for case in gold.cases:
        try:
            arguments = json.loads(
                json.dumps(case.arguments, ensure_ascii=False, allow_nan=False)
            )
            raw_output = executor(case.tool_name, arguments)
            output = _to_json_value(raw_output)
            result, case_observations = _evaluate_case(case, output=output, error=None)
        except Exception as error:  # One failed tool call must not abort evaluation.
            captured = BenchmarkCaseError(
                error_type=type(error).__name__,
                message="offline tool execution or output validation failed",
            )
            result, case_observations = _evaluate_case(
                case,
                output=None,
                error=captured,
            )
        results.append(result)
        observations.extend(case_observations)

    _assert_gold_unchanged(gold)
    metrics = _aggregate_metrics(gold.cases, results, observations)
    metrics_by_split = _aggregate_metrics_by_split(
        gold.cases,
        results,
        observations,
    )
    result_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in results)
    )
    descriptor = BenchmarkArtifactDescriptor(
        bytes=len(result_bytes),
        rows=len(results),
        sha256=sha256_bytes(result_bytes),
    )
    success_count = sum(item.status == "ok" for item in results)
    split_counts = {
        split: sum(case.split == split for case in gold.cases)
        for split in ("tool_dev", "tool_test_frozen")
    }
    task_counts = {
        task: sum(case.task == task for case in gold.cases)
        for task in ("ranking", "detection", "ocr")
    }
    manifest = BenchmarkManifest(
        run_id=run_id,
        gold_sha256=gold.sha256,
        query_set_sha256=gold.query_set_sha256,
        registry_sha256=bindings.registry_sha256,
        tool_binding_sha256=bindings.tool_binding_sha256,
        runtime_binding_sha256=bindings.runtime_binding_sha256,
        run_binding_sha256=bindings.run_binding_sha256,
        case_count=len(results),
        success_count=success_count,
        error_count=len(results) - success_count,
        split_counts=split_counts,
        task_counts=task_counts,
        metrics=metrics,
        metrics_by_split=metrics_by_split,
        results=descriptor,
        manifest_sha256="0" * 64,
    )
    manifest = manifest.model_copy(
        update={"manifest_sha256": _model_digest(manifest, "manifest_sha256")}
    )

    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / _RESULTS_FILE, result_bytes)
        atomic_create_file(
            staging / _MANIFEST_FILE,
            canonical_json_bytes(manifest.model_dump(mode="json")),
        )
        _load_benchmark_result_directory(
            staging,
            gold=gold,
            expected_bindings=bindings,
        )
        _assert_gold_unchanged(gold)
        atomic_publish_new_directory(staging, destination)
    except Exception:
        _remove_owned_staging(staging, destination.parent)
        raise
    return load_benchmark_result(
        destination,
        gold=gold,
        expected_bindings=bindings,
    )


def load_benchmark_result(
    path: str | Path,
    *,
    gold: VerifiedBenchmarkGold | DiagnosticBenchmarkGold,
    expected_bindings: BenchmarkBindings,
) -> DiagnosticBenchmarkResult:
    """Verify diagnostic result integrity, coverage, metrics, and bindings."""

    _require_benchmark_gold(gold)
    _assert_gold_unchanged(gold)
    result = _load_benchmark_result_directory(
        Path(path),
        gold=gold,
        expected_bindings=expected_bindings,
    )
    _assert_gold_unchanged(gold)
    return result


def require_verified_benchmark_result(
    value: VerifiedBenchmarkResult,
) -> VerifiedBenchmarkResult:
    del value
    raise BenchmarkError(
        "formal benchmark results are not produced by this evaluator; "
        "executor/raw-prediction results are diagnostic only"
    )


def require_diagnostic_benchmark_result(
    value: DiagnosticBenchmarkResult,
) -> DiagnosticBenchmarkResult:
    if (
        not isinstance(value, DiagnosticBenchmarkResult)
        or value._marker is not _DIAGNOSTIC_RESULT_MARKER
        or value.manifest.assurance_level != "diagnostic"
    ):
        raise BenchmarkError(
            "a diagnostic result returned by load/run_offline_benchmark is required"
        )
    return value


def _load_benchmark_result_directory(
    path: Path,
    *,
    gold: VerifiedBenchmarkGold | DiagnosticBenchmarkGold,
    expected_bindings: BenchmarkBindings,
) -> DiagnosticBenchmarkResult:
    try:
        info = path.lstat()
    except OSError as error:
        raise BenchmarkError(
            "benchmark result directory cannot be inspected"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BenchmarkError("benchmark result must be a non-symlink directory")
    try:
        names = {item.name for item in path.iterdir()}
    except OSError as error:
        raise BenchmarkError("benchmark result directory cannot be listed") from error
    if names != _ALLOWED_RESULT_FILES:
        raise BenchmarkError(f"benchmark result file set is invalid: {sorted(names)!r}")

    try:
        manifest_bytes = read_stable_regular_file(
            path / _MANIFEST_FILE,
            label="benchmark manifest",
        )
        manifest_value = parse_canonical_json(
            manifest_bytes,
            label="benchmark manifest",
        )
        manifest = BenchmarkManifest.model_validate(manifest_value)
    except (ArtifactFormatError, ValidationError) as error:
        raise BenchmarkError(str(error)) from error
    if manifest.manifest_sha256 != _model_digest(manifest, "manifest_sha256"):
        raise BenchmarkError("benchmark manifest self-hash mismatch")
    if manifest.gold_sha256 != gold.sha256:
        raise BenchmarkError("benchmark manifest/gold hash mismatch")
    if manifest.query_set_sha256 != gold.query_set_sha256:
        raise BenchmarkError("benchmark manifest/query-set hash mismatch")
    for name in (
        "registry_sha256",
        "tool_binding_sha256",
        "runtime_binding_sha256",
        "run_binding_sha256",
        "network_policy",
    ):
        if getattr(manifest, name) != getattr(expected_bindings, name):
            raise BenchmarkError(f"benchmark {name} binding mismatch")

    try:
        result_bytes = read_stable_regular_file(
            path / _RESULTS_FILE,
            label="benchmark results",
        )
        rows = parse_canonical_jsonl(result_bytes, label="benchmark results")
    except ArtifactFormatError as error:
        raise BenchmarkError(str(error)) from error
    if len(result_bytes) != manifest.results.bytes:
        raise BenchmarkError("benchmark results byte count mismatch")
    if sha256_bytes(result_bytes) != manifest.results.sha256:
        raise BenchmarkError("benchmark results hash mismatch")
    if len(rows) != manifest.results.rows:
        raise BenchmarkError("benchmark results row count mismatch")

    results: list[BenchmarkCaseResult] = []
    by_query = gold.case_by_query_id()
    seen: set[str] = set()
    observations: list[_APObservation] = []
    for ordinal, row in enumerate(rows):
        try:
            result = BenchmarkCaseResult.model_validate(row)
        except ValidationError as error:
            raise BenchmarkError(f"invalid result row {ordinal}: {error}") from error
        if result.query_id in seen:
            raise BenchmarkError(f"duplicate result query_id: {result.query_id}")
        seen.add(result.query_id)
        case = by_query.get(result.query_id)
        if case is None:
            raise BenchmarkError(f"extra result query_id: {result.query_id}")
        if (result.task, result.split, result.leakage_group_id) != (
            case.task,
            case.split,
            case.leakage_group_id,
        ):
            raise BenchmarkError(f"result identity mismatch for {result.query_id}")
        if result.result_sha256 != _model_digest(result, "result_sha256"):
            raise BenchmarkError(f"result self-hash mismatch for {result.query_id}")
        if result.status == "ok":
            if result.output_sha256 != sha256_bytes(
                canonical_json_bytes(result.output)
            ):
                raise BenchmarkError(f"output hash mismatch for {result.query_id}")
        recomputed, case_observations = _evaluate_case(
            case,
            output=result.output,
            error=result.error,
        )
        if recomputed.model_dump(mode="json") != result.model_dump(mode="json"):
            raise BenchmarkError(f"stored metrics mismatch for {result.query_id}")
        results.append(result)
        observations.extend(case_observations)

    expected_ids = set(by_query)
    missing = sorted(expected_ids - seen)
    extra = sorted(seen - expected_ids)
    if missing or extra:
        raise BenchmarkError(
            f"result query set mismatch; missing={missing!r}, extra={extra!r}"
        )
    if [item.query_id for item in results] != sorted(seen):
        raise BenchmarkError("benchmark result rows must be sorted by query_id")
    recomputed_metrics = _aggregate_metrics(gold.cases, results, observations)
    if recomputed_metrics != manifest.metrics:
        raise BenchmarkError("benchmark aggregate metrics mismatch")
    recomputed_by_split = _aggregate_metrics_by_split(
        gold.cases,
        results,
        observations,
    )
    if recomputed_by_split != manifest.metrics_by_split:
        raise BenchmarkError("benchmark split aggregate metrics mismatch")
    success_count = sum(item.status == "ok" for item in results)
    if (manifest.success_count, manifest.error_count) != (
        success_count,
        len(results) - success_count,
    ):
        raise BenchmarkError("benchmark status counts mismatch")
    expected_splits = {
        split: sum(case.split == split for case in gold.cases)
        for split in ("tool_dev", "tool_test_frozen")
    }
    expected_tasks = {
        task: sum(case.task == task for case in gold.cases)
        for task in ("ranking", "detection", "ocr")
    }
    if (
        manifest.split_counts != expected_splits
        or manifest.task_counts != expected_tasks
    ):
        raise BenchmarkError("benchmark split/task counts mismatch")
    return DiagnosticBenchmarkResult(
        path=path.resolve(),
        manifest=manifest,
        results=tuple(results),
        _marker=_DIAGNOSTIC_RESULT_MARKER,
    )


def _evaluate_case(
    case: GoldCase,
    *,
    output: Any,
    error: BenchmarkCaseError | None,
) -> tuple[BenchmarkCaseResult, list[_APObservation]]:
    status: Literal["ok", "error"] = "error" if error is not None else "ok"
    observations: list[_APObservation] = []
    if isinstance(case, RankingGoldCase):
        metrics = _ranking_metrics(case, output if error is None else None)
    elif isinstance(case, DetectionGoldCase):
        metrics, observations = _detection_metrics(
            case,
            output if error is None else None,
        )
    else:
        metrics = _ocr_metrics(case, output if error is None else None)
    output_sha256 = (
        sha256_bytes(canonical_json_bytes(output)) if error is None else None
    )
    result = BenchmarkCaseResult(
        query_id=case.query_id,
        task=case.task,
        split=case.split,
        leakage_group_id=case.leakage_group_id,
        status=status,
        error=error,
        output=output if error is None else None,
        output_sha256=output_sha256,
        metrics=metrics,
        result_sha256="0" * 64,
    )
    result = result.model_copy(
        update={"result_sha256": _model_digest(result, "result_sha256")}
    )
    return result, observations


def _ranking_metrics(case: RankingGoldCase, output: Any) -> RankingCaseMetrics:
    predicted_ids: list[str] = []
    if output is not None:
        hits = (
            _extract_path(output, case.result_list_path)
            if case.result_list_path
            else output
        )
        if not isinstance(hits, list):
            raise BenchmarkError("ranking output must resolve to a list")
        for ordinal, hit in enumerate(hits):
            identifier = _extract_path(hit, case.result_id_path)
            if not isinstance(identifier, str):
                raise BenchmarkError(f"ranking hit {ordinal} id must be a string")
            _nonblank(identifier, f"ranking hit {ordinal} id")
            predicted_ids.append(identifier)
        if len(predicted_ids) != len(set(predicted_ids)):
            raise BenchmarkError("ranking output contains duplicate ids")
    relevant = set(case.relevant_ids)
    top = predicted_ids[: case.k]
    true_positive = sum(identifier in relevant for identifier in top)
    first_rank = next(
        (
            rank
            for rank, identifier in enumerate(predicted_ids, start=1)
            if identifier in relevant
        ),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, identifier in enumerate(top, start=1)
        if identifier in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(case.k, len(relevant)) + 1)
    )
    return RankingCaseMetrics(
        k=case.k,
        relevant_count=len(relevant),
        retrieved_count=len(predicted_ids),
        true_positive_at_k=true_positive,
        recall_at_k=true_positive / len(relevant),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        ndcg_at_k=0.0 if ideal == 0 else dcg / ideal,
    )


def _detection_metrics(
    case: DetectionGoldCase,
    output: Any,
) -> tuple[DetectionCaseMetrics, list[_APObservation]]:
    predictions: list[tuple[str, list[float], float, int]] = []
    if output is not None:
        raw = output.get("detections") if isinstance(output, dict) else output
        if not isinstance(raw, list):
            raise BenchmarkError(
                "detection output must be a list or contain detections"
            )
        for ordinal, item in enumerate(raw):
            if not isinstance(item, dict):
                raise BenchmarkError(f"detection {ordinal} must be an object")
            label = item.get("label")
            bbox = item.get("bbox_xyxy")
            confidence = item.get("confidence")
            if not isinstance(label, str):
                raise BenchmarkError(f"detection {ordinal} label must be a string")
            _nonblank(label, f"detection {ordinal} label")
            if not isinstance(bbox, list):
                raise BenchmarkError(f"detection {ordinal} bbox_xyxy must be a list")
            _validate_bbox(bbox)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise BenchmarkError(f"detection {ordinal} confidence must be numeric")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise BenchmarkError(
                    f"detection {ordinal} confidence must be in [0, 1]"
                )
            predictions.append(
                (label, [float(value) for value in bbox], confidence, ordinal)
            )

    matched: set[int] = set()
    observations: list[_APObservation] = []
    true_positive = 0
    for label, bbox, confidence, ordinal in sorted(
        predictions,
        key=lambda item: (-item[2], item[3]),
    ):
        candidates = [
            (_iou(bbox, gold.bbox_xyxy), gold_ordinal)
            for gold_ordinal, gold in enumerate(case.expected_detections)
            if gold_ordinal not in matched and gold.label == label
        ]
        best_iou, best_ordinal = max(candidates, default=(0.0, -1))
        is_true_positive = best_iou >= case.iou_threshold
        if is_true_positive:
            matched.add(best_ordinal)
            true_positive += 1
        observations.append(
            _APObservation(
                confidence=confidence,
                query_id=case.query_id,
                prediction_ordinal=ordinal,
                is_true_positive=is_true_positive,
            )
        )
    metrics = DetectionCaseMetrics(
        gold_count=len(case.expected_detections),
        prediction_count=len(predictions),
        true_positive=true_positive,
        false_positive=len(predictions) - true_positive,
        false_negative=len(case.expected_detections) - true_positive,
    )
    return metrics, observations


def _ocr_metrics(case: OCRGoldCase, output: Any) -> OCRCaseMetrics:
    predicted_text = ""
    predicted_fields: dict[str, tuple[str, list[Any]]] = {}
    if output is not None:
        if not isinstance(output, dict):
            raise BenchmarkError("OCR output must be an object")
        predicted_text = output.get("full_text")
        raw_lines = output.get("lines")
        raw_fields = output.get("fields")
        if not isinstance(predicted_text, str):
            raise BenchmarkError("OCR full_text must be a string")
        if not isinstance(raw_lines, list):
            raise BenchmarkError("OCR lines must be a list")
        if not isinstance(raw_fields, list):
            raise BenchmarkError("OCR fields must be a list")
        line_text_by_id: dict[str, str] = {}
        for ordinal, item in enumerate(raw_lines):
            if not isinstance(item, dict):
                raise BenchmarkError(f"OCR line {ordinal} must be an object")
            line_id = item.get("line_id")
            line_text = item.get("text")
            if not isinstance(line_id, str):
                raise BenchmarkError(f"OCR line {ordinal} line_id must be a string")
            _nonblank(line_id, f"OCR line {ordinal} line_id")
            if not isinstance(line_text, str):
                raise BenchmarkError(f"OCR line {ordinal} text must be a string")
            _nonblank(line_text, f"OCR line {ordinal} text")
            if line_id in line_text_by_id:
                raise BenchmarkError("OCR line IDs must be unique")
            line_text_by_id[line_id] = line_text
        for ordinal, item in enumerate(raw_fields):
            if not isinstance(item, dict):
                raise BenchmarkError(f"OCR field {ordinal} must be an object")
            name = item.get("field_name")
            value = item.get("value")
            evidence = item.get("evidence_line_ids")
            if not isinstance(name, str) or not isinstance(value, str):
                raise BenchmarkError(f"OCR field {ordinal} name/value must be strings")
            _nonblank(name, f"OCR field {ordinal} name")
            if not isinstance(evidence, list) or any(
                not isinstance(line_id, str) or not line_id for line_id in evidence
            ):
                raise BenchmarkError(
                    f"OCR field {ordinal} evidence_line_ids must be non-blank strings"
                )
            if len(evidence) != len(set(evidence)):
                raise BenchmarkError(f"OCR field {ordinal} evidence ids must be unique")
            if not set(evidence).issubset(line_text_by_id):
                raise BenchmarkError(f"OCR field {ordinal} cites missing line evidence")
            if name in predicted_fields:
                raise BenchmarkError(f"duplicate OCR field: {name}")
            predicted_fields[name] = (value, evidence)

    exact = 0
    evidenced = 0
    for expected in case.expected_fields:
        prediction = predicted_fields.get(expected.field_name)
        if prediction is not None:
            value_is_exact = prediction[0] == expected.expected_value
            exact += value_is_exact
            cited_text = "\n".join(
                line_text_by_id[line_id] for line_id in prediction[1]
            )
            evidenced += (
                value_is_exact
                and bool(prediction[1])
                and expected.expected_value in cited_text
            )
    return OCRCaseMetrics(
        reference_codepoints=len(case.expected_text),
        predicted_codepoints=len(predicted_text),
        edit_distance=_levenshtein_codepoints(case.expected_text, predicted_text),
        expected_field_count=len(case.expected_fields),
        exact_field_count=exact,
        evidenced_field_count=evidenced,
    )


def _aggregate_metrics(
    cases: Sequence[GoldCase],
    results: Sequence[BenchmarkCaseResult],
    observations: Sequence[_APObservation],
) -> BenchmarkAggregateMetrics:
    by_task: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)
    for result in results:
        by_task[result.task].append(result)

    ranking: RankingAggregate | None = None
    ranking_results = by_task["ranking"]
    if ranking_results:
        ranking_metrics = [
            item.metrics
            for item in ranking_results
            if isinstance(item.metrics, RankingCaseMetrics)
        ]
        k = ranking_metrics[0].k
        ranking = RankingAggregate(
            k=k,
            case_count=len(ranking_results),
            error_count=sum(item.status == "error" for item in ranking_results),
            recall_at_k=sum(item.recall_at_k for item in ranking_metrics)
            / len(ranking_metrics),
            mrr=sum(item.reciprocal_rank for item in ranking_metrics)
            / len(ranking_metrics),
            ndcg_at_k=sum(item.ndcg_at_k for item in ranking_metrics)
            / len(ranking_metrics),
        )

    detection: DetectionAggregate | None = None
    detection_results = by_task["detection"]
    if detection_results:
        detection_metrics = [
            item.metrics
            for item in detection_results
            if isinstance(item.metrics, DetectionCaseMetrics)
        ]
        true_positive = sum(item.true_positive for item in detection_metrics)
        false_positive = sum(item.false_positive for item in detection_metrics)
        false_negative = sum(item.false_negative for item in detection_metrics)
        gold_count = true_positive + false_negative
        prediction_count = true_positive + false_positive
        detection = DetectionAggregate(
            case_count=len(detection_results),
            error_count=sum(item.status == "error" for item in detection_results),
            gold_count=gold_count,
            prediction_count=prediction_count,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=(true_positive / prediction_count if prediction_count else 0.0),
            recall=true_positive / gold_count,
            ap50=_all_point_ap(observations, gold_count),
        )

    ocr: OCRAggregate | None = None
    ocr_results = by_task["ocr"]
    if ocr_results:
        ocr_metrics = [
            item.metrics
            for item in ocr_results
            if isinstance(item.metrics, OCRCaseMetrics)
        ]
        reference = sum(item.reference_codepoints for item in ocr_metrics)
        predicted = sum(item.predicted_codepoints for item in ocr_metrics)
        distance = sum(item.edit_distance for item in ocr_metrics)
        expected_fields = sum(item.expected_field_count for item in ocr_metrics)
        exact_fields = sum(item.exact_field_count for item in ocr_metrics)
        evidenced_fields = sum(item.evidenced_field_count for item in ocr_metrics)
        ocr = OCRAggregate(
            case_count=len(ocr_results),
            error_count=sum(item.status == "error" for item in ocr_results),
            reference_codepoints=reference,
            predicted_codepoints=predicted,
            edit_distance=distance,
            cer=distance / reference,
            expected_field_count=expected_fields,
            exact_field_count=exact_fields,
            field_exact=exact_fields / expected_fields,
            evidenced_field_count=evidenced_fields,
            evidence_coverage=evidenced_fields / expected_fields,
        )
    return BenchmarkAggregateMetrics(ranking=ranking, detection=detection, ocr=ocr)


def _aggregate_metrics_by_split(
    cases: Sequence[GoldCase],
    results: Sequence[BenchmarkCaseResult],
    observations: Sequence[_APObservation],
) -> dict[str, BenchmarkAggregateMetrics]:
    aggregates: dict[str, BenchmarkAggregateMetrics] = {}
    for split in ("tool_dev", "tool_test_frozen"):
        split_cases = [case for case in cases if case.split == split]
        split_ids = {case.query_id for case in split_cases}
        split_results = [result for result in results if result.query_id in split_ids]
        split_observations = [
            item for item in observations if item.query_id in split_ids
        ]
        aggregates[split] = _aggregate_metrics(
            split_cases,
            split_results,
            split_observations,
        )
    return aggregates


def _all_point_ap(observations: Sequence[_APObservation], gold_count: int) -> float:
    if not observations:
        return 0.0
    ordered = sorted(
        observations,
        key=lambda item: (-item.confidence, item.query_id, item.prediction_ordinal),
    )
    recalls = [0.0]
    precisions = [0.0]
    true_positive = 0
    false_positive = 0
    for item in ordered:
        true_positive += item.is_true_positive
        false_positive += not item.is_true_positive
        recalls.append(true_positive / gold_count)
        precisions.append(true_positive / (true_positive + false_positive))
    recalls.append(1.0)
    precisions.append(0.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls))
        if recalls[index] != recalls[index - 1]
    )


def _levenshtein_codepoints(reference: str, prediction: str) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def _validate_bbox(value: Sequence[object]) -> None:
    if len(value) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError("bbox_xyxy coordinates must be numeric")
    coordinates = [float(item) for item in value]
    if any(not math.isfinite(item) for item in coordinates):
        raise ValueError("bbox_xyxy coordinates must be finite")
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive area")


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _extract_path(value: Any, path: str | None) -> Any:
    current = value
    if path is None:
        return current
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise BenchmarkError(f"output path is missing: {path}")
        current = current[part]
    return current


def _to_json_value(value: object) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkError("tool output contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BenchmarkError("tool output object keys must be strings")
            result[key] = _to_json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    raise BenchmarkError(f"tool output is not JSON-compatible: {type(value).__name__}")


def _model_digest(model: BaseModel, digest_field: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(digest_field)
    return sha256_bytes(canonical_json_bytes(payload))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_verified_gold(gold: VerifiedBenchmarkGold) -> None:
    if (
        not isinstance(gold, VerifiedBenchmarkGold)
        or gold._marker is not _VERIFIED_GOLD_MARKER
    ):
        raise BenchmarkError("gold returned by load_benchmark_gold is required")


def _require_benchmark_gold(
    gold: VerifiedBenchmarkGold | DiagnosticBenchmarkGold,
) -> None:
    if isinstance(gold, VerifiedBenchmarkGold):
        _require_verified_gold(gold)
        return
    if (
        not isinstance(gold, DiagnosticBenchmarkGold)
        or gold._marker is not _DIAGNOSTIC_GOLD_MARKER
    ):
        raise BenchmarkError("gold returned by a benchmark gold loader is required")


def _assert_gold_unchanged(
    gold: VerifiedBenchmarkGold | DiagnosticBenchmarkGold,
) -> None:
    try:
        content = read_stable_regular_file(gold.path, label="benchmark gold")
    except ArtifactFormatError as error:
        raise BenchmarkError(str(error)) from error
    if sha256_bytes(content) != gold.sha256:
        raise BenchmarkError("benchmark gold changed after verification")
    if isinstance(gold, VerifiedBenchmarkGold):
        refreshed = load_benchmark_gold(
            gold.path,
            expected_gold_sha256=gold.sha256,
            authoritative_assignments=tuple(
                QueryAssignment(
                    query_id=case.query_id,
                    leakage_group_id=case.leakage_group_id,
                    split=case.split,
                )
                for case in gold.cases
            ),
            review_ledger_path=gold.review_ledger_path,
            expected_review_ledger_sha256=gold.review_ledger_sha256,
        )
        if (
            refreshed.cases != gold.cases
            or refreshed.query_set_sha256 != gold.query_set_sha256
            or refreshed.review_ledger != gold.review_ledger
        ):
            raise BenchmarkError("verified benchmark gold handle was mutated")


def _remove_owned_staging(staging: Path, destination_parent: Path) -> None:
    if not os.path.lexists(staging):
        return
    try:
        parent = staging.parent.resolve()
        expected_parent = destination_parent.resolve()
    except OSError:
        return
    if parent != expected_parent or ".staging-" not in staging.name:
        return
    shutil.rmtree(staging)


__all__ = [
    "BENCHMARK_POLICY_VERSION",
    "GOLD_REVIEW_POLICY_VERSION",
    "BenchmarkAggregateMetrics",
    "BenchmarkBindings",
    "BenchmarkCaseResult",
    "DiagnosticBenchmarkGold",
    "DiagnosticBenchmarkResult",
    "BenchmarkError",
    "BenchmarkManifest",
    "DetectionGoldCase",
    "GoldDetection",
    "GoldCaseReview",
    "GoldOCRField",
    "GoldReviewEvidence",
    "GoldReviewLedger",
    "OCRGoldCase",
    "OfflineToolExecutor",
    "QueryAssignment",
    "RankingGoldCase",
    "VerifiedBenchmarkGold",
    "VerifiedBenchmarkResult",
    "load_benchmark_gold",
    "load_diagnostic_benchmark_gold",
    "load_benchmark_result",
    "require_diagnostic_benchmark_result",
    "require_verified_benchmark_result",
    "run_offline_benchmark",
]
