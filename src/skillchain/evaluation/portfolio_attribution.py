"""Current-parent-bound attribution packets for Portfolio S2 and S3.

The source of truth is one completed ``run_portfolio_treatment_smoke``
directory.  Every packet is rebuilt from canonical, self-hashed source files
and is bound to the exact Bank that produced those files.  This prevents a
route/body optimizer from consuming attribution produced by an earlier
candidate or by a differently named configuration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import math
import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.evaluation.assistant_runs import (
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantRequestSnapshot,
)
from skillchain.evaluation.final_runtime import FinalJudgeEvaluationResult
from skillchain.evaluation.packets import (
    AssistantToolTrace,
    JudgeDimensionScore,
    VisibleCard,
    VisibleToolEvidence,
)
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_OPTIMIZATION_QUERY_IDS,
    PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION,
    PortfolioStageGateResultSet,
    calculate_portfolio_skill_adherence,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AttributionStage = Literal["s2_route_optimizer", "s3_body_refiner"]
SourceConfig = Literal["s1", "s1s2"]
TreatmentConfig = Literal["llm_static", "s1", "s1s2", "full"]

LEGACY_PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION = (
    "portfolio-current-parent-attribution-v1"
)
PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION = "portfolio-current-parent-attribution-v2"
PORTFOLIO_OPTIMIZATION_QUERY_COUNT = 25
PORTFOLIO_ATTRIBUTION_QUERY_COUNT = PORTFOLIO_OPTIMIZATION_QUERY_COUNT
PORTFOLIO_OPTIMIZATION_QUERY_IDS_SHA256 = sha256_bytes(
    canonical_json_bytes(list(PORTFOLIO_OPTIMIZATION_QUERY_IDS))
)
PORTFOLIO_ATTRIBUTION_CAPABILITIES = frozenset(
    {
        "knowledge.visual_encyclopedia",
        "product.exact_match",
        "product.multi_search",
        "product.style_recommendation",
        "utility.document_reading",
        "utility.recipe_guidance",
    }
)
PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS = {
    "knowledge.visual_encyclopedia": 4,
    "product.exact_match": 5,
    "product.multi_search": 5,
    "product.style_recommendation": 4,
    "utility.document_reading": 4,
    "utility.recipe_guidance": 3,
}

_STAGE_CONFIG: dict[AttributionStage, SourceConfig] = {
    "s2_route_optimizer": "s1",
    "s3_body_refiner": "s1s2",
}
_SMOKE_POLICY_VERSION = PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES = 64 * 1024 * 1024


class PortfolioAttributionError(ValueError):
    """A source smoke run or attribution packet failed closed."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and trimmed")
    return value


def _model_self_hash(value: BaseModel, field_name: str) -> str:
    payload = value.model_dump(mode="json", exclude={field_name})
    return sha256_bytes(canonical_json_bytes(payload))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_object_self_hash(
    value: dict[str, Any],
    field_name: str,
    *,
    label: str,
) -> str:
    supplied = value.get(field_name)
    unsigned = dict(value)
    unsigned.pop(field_name, None)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if supplied != expected:
        raise PortfolioAttributionError(f"{label} {field_name} mismatch")
    return expected


class PortfolioRouteAttributionRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-current-parent-route-attribution"] = (
        "portfolio-current-parent-route-attribution"
    )
    record_ordinal: int = Field(ge=0, lt=PORTFOLIO_ATTRIBUTION_QUERY_COUNT)
    query_id: str
    canonical_capability: str
    selected_capability: str | None
    route_correct: bool
    skill_slug: str | None
    route_trace_sha256: Sha256 | None
    assistant_error_code: str | None
    tool_names: tuple[str, ...]
    source_result_sha256: Sha256
    source_assistant_file_sha256: Sha256
    source_assistant_row_sha256: Sha256
    source_request_sha256: Sha256
    source_response_sha256: Sha256
    source_receipt_sha256: Sha256
    source_final_file_sha256: Sha256
    source_final_result_sha256: Sha256
    record_sha256: Sha256

    @field_validator("tool_names", mode="before")
    @classmethod
    def coerce_tool_names(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("query_id", "canonical_capability")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "selected_capability",
        "skill_slug",
        "assistant_error_code",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.canonical_capability not in PORTFOLIO_ATTRIBUTION_CAPABILITIES:
            raise ValueError("route attribution capability is outside the universe")
        if self.route_correct != (
            self.selected_capability == self.canonical_capability
        ):
            raise ValueError("route attribution correctness projection drifted")
        route_identity = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if any(item is not None for item in route_identity) and not all(
            item is not None for item in route_identity
        ):
            raise ValueError("route identity must be complete or absent")
        if any(not item or item != item.strip() for item in self.tool_names):
            raise ValueError("route tool names must be non-blank and trimmed")
        if self.record_sha256 != _model_self_hash(self, "record_sha256"):
            raise ValueError("route attribution record self hash mismatch")
        return self


class PortfolioBodyAttributionRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-current-parent-body-attribution"] = (
        "portfolio-current-parent-body-attribution"
    )
    record_ordinal: int = Field(ge=0, lt=PORTFOLIO_ATTRIBUTION_QUERY_COUNT)
    query_id: str
    canonical_capability: str
    selected_capability: str | None
    route_correct: bool
    skill_slug: str | None
    route_trace_sha256: Sha256 | None
    response_text: str
    visible_cards: tuple[VisibleCard, ...]
    visible_tool_evidence: tuple[VisibleToolEvidence, ...]
    tool_trace: tuple[AssistantToolTrace, ...]
    backend_error_code: str | None
    assistant_error_code: str | None
    hard_error: bool
    final_kind: Literal["visual_final_judge", "assistant_fixed_zero"]
    evaluation_id: Sha256 | None
    judge_status: str
    judge_error_code: str | None
    dimensions: tuple[JudgeDimensionScore, ...]
    j_project: float = Field(ge=0.0, le=100.0)
    source_result_sha256: Sha256
    source_assistant_file_sha256: Sha256
    source_assistant_row_sha256: Sha256
    source_request_sha256: Sha256
    source_response_sha256: Sha256
    source_receipt_sha256: Sha256
    source_final_file_sha256: Sha256
    source_final_result_sha256: Sha256
    record_sha256: Sha256

    @field_validator(
        "visible_cards",
        "visible_tool_evidence",
        "tool_trace",
        "dimensions",
        mode="before",
    )
    @classmethod
    def coerce_body_tuples(cls, value: object, info) -> object:
        if not isinstance(value, list):
            return value
        nested_models = {
            "visible_cards": VisibleCard,
            "visible_tool_evidence": VisibleToolEvidence,
            "tool_trace": AssistantToolTrace,
            "dimensions": JudgeDimensionScore,
        }
        model_type = nested_models[info.field_name]
        return tuple(
            model_type.model_validate(item, strict=False)
            if isinstance(item, dict)
            else item
            for item in value
        )

    @field_validator("query_id", "canonical_capability", "judge_status")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "selected_capability",
        "skill_slug",
        "backend_error_code",
        "assistant_error_code",
        "judge_error_code",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.canonical_capability not in PORTFOLIO_ATTRIBUTION_CAPABILITIES:
            raise ValueError("body attribution capability is outside the universe")
        if self.route_correct != (
            self.selected_capability == self.canonical_capability
        ):
            raise ValueError("body attribution correctness projection drifted")
        route_identity = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if any(item is not None for item in route_identity) and not all(
            item is not None for item in route_identity
        ):
            raise ValueError("body attribution route identity is partial")
        if self.hard_error != (
            self.assistant_error_code is not None or self.judge_status != "scored"
        ):
            raise ValueError("body attribution hard-error projection drifted")
        if self.final_kind == "assistant_fixed_zero":
            if (
                self.assistant_error_code is None
                or self.evaluation_id is not None
                or self.dimensions
                or self.j_project != 0.0
                or self.judge_status
                not in {
                    "assistant_fixed_zero",
                    "assistant_public_identity_leak_fixed_zero",
                }
            ):
                raise ValueError("fixed-zero body attribution is inconsistent")
        elif self.evaluation_id is None or not self.dimensions:
            raise ValueError("visual-Judge body attribution is incomplete")
        if self.record_sha256 != _model_self_hash(self, "record_sha256"):
            raise ValueError("body attribution record self hash mismatch")
        return self


PortfolioAttributionRecord = Annotated[
    PortfolioRouteAttributionRecord | PortfolioBodyAttributionRecord,
    Field(discriminator="kind"),
]


class PortfolioParentAttributionPacket(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-current-parent-attribution-packet"] = (
        "portfolio-current-parent-attribution-packet"
    )
    policy_version: Literal["portfolio-current-parent-attribution-v2"] = (
        PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION
    )
    stage: AttributionStage
    source_config: SourceConfig
    source_bank_sha256: Sha256
    optimization_query_ids_sha256: Sha256
    source_summary_file_sha256: Sha256
    source_summary_sha256: Sha256
    source_results_file_sha256: Sha256
    query_ids: tuple[str, ...]
    records: tuple[PortfolioAttributionRecord, ...]
    packet_sha256: Sha256

    @field_validator("query_ids", "records", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if self.source_config != _STAGE_CONFIG[self.stage]:
            raise ValueError("attribution stage/source config mismatch")
        if (
            len(self.query_ids) != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
            or len(set(self.query_ids)) != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
            or any(not item or item != item.strip() for item in self.query_ids)
        ):
            raise ValueError("attribution packet requires 25 unique query IDs")
        if self.query_ids != PORTFOLIO_OPTIMIZATION_QUERY_IDS:
            raise ValueError("attribution packet must use frozen dm-001..dm-025")
        if (
            self.optimization_query_ids_sha256
            != PORTFOLIO_OPTIMIZATION_QUERY_IDS_SHA256
        ):
            raise ValueError("optimization query-ID digest differs from frozen 25")
        if len(self.records) != PORTFOLIO_ATTRIBUTION_QUERY_COUNT:
            raise ValueError("attribution packet requires 25 records")
        if tuple(item.query_id for item in self.records) != self.query_ids:
            raise ValueError("attribution record order differs from query IDs")
        if tuple(item.record_ordinal for item in self.records) != tuple(
            range(PORTFOLIO_ATTRIBUTION_QUERY_COUNT)
        ):
            raise ValueError("attribution record ordinals are not contiguous")
        expected_kind = (
            "portfolio-current-parent-route-attribution"
            if self.stage == "s2_route_optimizer"
            else "portfolio-current-parent-body-attribution"
        )
        if any(item.kind != expected_kind for item in self.records):
            raise ValueError("attribution record kind differs from stage")
        capability_counts = Counter(item.canonical_capability for item in self.records)
        if dict(capability_counts) != PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS:
            raise ValueError(
                "attribution records do not match optimization25 capability counts"
            )
        if self.packet_sha256 != _model_self_hash(self, "packet_sha256"):
            raise ValueError("attribution packet self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class _SmokeAssistantEnvelope(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-treatment-smoke-assistant"] = (
        "portfolio-treatment-smoke-assistant"
    )
    policy_version: Literal["portfolio-treatment-optimization25-smoke-v3"] = (
        _SMOKE_POLICY_VERSION
    )
    query_id: str
    config: TreatmentConfig
    request: AssistantRequestSnapshot
    response: AssistantBackendResponse
    receipt: AssistantExecutionReceipt
    shared_route_file: str | None
    shared_route_file_sha256: Sha256 | None
    row_sha256: Sha256

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if self.row_sha256 != _model_self_hash(self, "row_sha256"):
            raise ValueError("smoke Assistant row self hash mismatch")
        if (
            self.request.query.query_id != self.query_id
            or self.request.config != self.config
            or self.request.treatment.config != self.config
            or self.response.request_sha256 != self.request.request_sha256
            or self.receipt.request_sha256 != self.request.request_sha256
            or self.receipt.response_sha256
            != sha256_bytes(canonical_json_bytes(self.response.model_dump(mode="json")))
            or self.receipt.tool_trace != self.response.tool_trace
            or self.receipt.aggregate_usage != self.response.usage
        ):
            raise ValueError("smoke Assistant request/response/receipt binding drifted")
        if (
            self.response.backbone_provider != self.request.backbone.provider
            or self.response.backbone_model != self.request.backbone.model
            or self.response.backbone_endpoint != self.request.backbone.endpoint
            or self.response.backbone_identity_sha256
            != self.request.backbone.identity_sha256
            or self.response.registry_sha256 != self.request.registry.registry_sha256
            or self.response.registry_runtime_sha256
            != self.request.registry.registry_runtime_sha256
            or self.response.budget_sha256 != self.request.budget.budget_sha256
        ):
            raise ValueError("smoke Assistant runtime echoes drifted")
        if (self.shared_route_file is None) != (self.shared_route_file_sha256 is None):
            raise ValueError("smoke shared-route file binding is partial")
        return self


class _SmokeResultRow(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-treatment-smoke-result"] = (
        "portfolio-treatment-smoke-result"
    )
    policy_version: Literal["portfolio-treatment-optimization25-smoke-v3"] = (
        _SMOKE_POLICY_VERSION
    )
    query_id: str
    canonical_capability: str
    config: TreatmentConfig
    target_bank_sha256: Sha256
    assistant_file: str
    assistant_file_sha256: Sha256
    final_file: str
    final_file_sha256: Sha256
    shared_route_file: str | None
    shared_route_file_sha256: Sha256 | None
    selected_capability: str | None
    route_correct: bool
    assistant_error_code: str | None
    final_status: str
    hard_error: bool
    skill_adherence: float = Field(ge=0.0, le=1.0)
    j_project: float = Field(ge=0.0, le=100.0)
    usage_and_cost: dict[str, Any]
    result_sha256: Sha256

    @field_validator(
        "query_id",
        "canonical_capability",
        "assistant_file",
        "final_file",
        "final_status",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("selected_capability", "assistant_error_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        if self.canonical_capability not in PORTFOLIO_ATTRIBUTION_CAPABILITIES:
            raise ValueError("smoke result capability is outside the universe")
        if self.route_correct != (
            self.selected_capability == self.canonical_capability
        ):
            raise ValueError("smoke route-correct projection drifted")
        if self.hard_error != (
            self.assistant_error_code is not None or self.final_status != "scored"
        ):
            raise ValueError("smoke hard-error projection drifted")
        if self.hard_error and self.j_project != 0.0:
            raise ValueError("hard-error smoke result must retain zero score")
        if (self.shared_route_file is None) != (self.shared_route_file_sha256 is None):
            raise ValueError("smoke result shared-route binding is partial")
        if self.result_sha256 != _model_self_hash(self, "result_sha256"):
            raise ValueError("smoke result self hash mismatch")
        return self


class _FinalProjection(_StrictFrozenModel):
    final_kind: Literal["visual_final_judge", "assistant_fixed_zero"]
    evaluation_id: Sha256 | None
    judge_status: str
    judge_error_code: str | None
    dimensions: tuple[JudgeDimensionScore, ...]
    j_project: float = Field(ge=0.0, le=100.0)
    result_sha256: Sha256


def _read_canonical_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        content = read_stable_regular_file(
            path,
            label=label,
            max_bytes=_MAX_FILE_BYTES,
        )
        raw = parse_canonical_json(content, label=label)
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioAttributionError(f"{label} is not safely readable") from error
    if not isinstance(raw, dict):
        raise PortfolioAttributionError(f"{label} must contain an object")
    return content, raw


def _load_results(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[bytes, tuple[_SmokeResultRow, ...]]:
    try:
        content = read_stable_regular_file(
            path,
            label="source smoke results",
            max_bytes=_MAX_FILE_BYTES,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioAttributionError(
            "source smoke results cannot be read"
        ) from error
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioAttributionError("source smoke results file digest mismatch")

    rows: list[_SmokeResultRow] = []
    raw_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        try:
            raw = parse_canonical_json(
                line,
                label=f"source smoke results line {line_number}",
            )
            if not isinstance(raw, dict):
                raise ArtifactFormatError("smoke result row is not an object")
            row = _SmokeResultRow.model_validate_json(line, strict=True)
        except (ArtifactFormatError, ValidationError) as error:
            raise PortfolioAttributionError(
                f"source smoke result line {line_number} is invalid"
            ) from error
        raw_rows.append(raw)
        rows.append(row)

    canonical = b"".join(canonical_json_bytes(item) for item in raw_rows)
    if not rows or content != canonical:
        raise PortfolioAttributionError("source smoke results are not canonical JSONL")
    return content, tuple(rows)


def _load_assistant(
    root: Path,
    row: _SmokeResultRow,
    *,
    expected_config: TreatmentConfig,
    expected_bank_sha256: str,
) -> tuple[_SmokeAssistantEnvelope, bytes]:
    expected_relative = f"assistant/{row.query_id}.json"
    if row.assistant_file != expected_relative:
        raise PortfolioAttributionError("source smoke Assistant path is not canonical")
    content, _ = _read_canonical_object(
        root / expected_relative,
        label=f"source smoke Assistant {row.query_id}",
    )
    if sha256_bytes(content) != row.assistant_file_sha256:
        raise PortfolioAttributionError("source smoke Assistant file digest mismatch")
    try:
        assistant = _SmokeAssistantEnvelope.model_validate_json(
            content,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioAttributionError(
            f"source smoke Assistant is invalid: {row.query_id}"
        ) from error
    if (
        assistant.query_id != row.query_id
        or assistant.config != expected_config
        or assistant.request.treatment.bank_sha256 != expected_bank_sha256
        or assistant.response.selected_capability != row.selected_capability
        or assistant.shared_route_file != row.shared_route_file
        or assistant.shared_route_file_sha256 != row.shared_route_file_sha256
    ):
        raise PortfolioAttributionError(
            f"source smoke Assistant binding differs: {row.query_id}"
        )
    return assistant, content


def _load_final(
    root: Path,
    row: _SmokeResultRow,
) -> tuple[_FinalProjection, bytes]:
    expected_relative = f"final/{row.query_id}.json"
    if row.final_file != expected_relative:
        raise PortfolioAttributionError("source smoke final path is not canonical")
    content, raw = _read_canonical_object(
        root / expected_relative,
        label=f"source smoke final {row.query_id}",
    )
    if sha256_bytes(content) != row.final_file_sha256:
        raise PortfolioAttributionError("source smoke final file digest mismatch")

    if raw.get("kind") == "portfolio-treatment-smoke-final-fixed-zero":
        result_sha256 = _require_object_self_hash(
            raw,
            "result_sha256",
            label=f"source smoke final {row.query_id}",
        )
        if (
            raw.get("schema_version") != 1
            or raw.get("query_id") != row.query_id
            or raw.get("assistant_error_code") != row.assistant_error_code
            or raw.get("judge_invoked") is not False
            or raw.get("j_project") != 0.0
            or row.final_status
            not in {
                "assistant_fixed_zero",
                "assistant_public_identity_leak_fixed_zero",
            }
        ):
            raise PortfolioAttributionError(
                f"source smoke fixed-zero binding differs: {row.query_id}"
            )
        projection = _FinalProjection(
            final_kind="assistant_fixed_zero",
            evaluation_id=None,
            judge_status=row.final_status,
            judge_error_code=row.assistant_error_code,
            dimensions=(),
            j_project=0.0,
            result_sha256=result_sha256,
        )
    else:
        try:
            final = FinalJudgeEvaluationResult.model_validate_json(
                content,
                strict=True,
            )
        except ValidationError as error:
            raise PortfolioAttributionError(
                f"source visual final is invalid: {row.query_id}"
            ) from error
        if row.assistant_error_code is not None:
            raise PortfolioAttributionError(
                "visual final cannot be paired with an Assistant hard error"
            )
        projection = _FinalProjection(
            final_kind="visual_final_judge",
            evaluation_id=final.evaluation_id,
            judge_status=final.outcome.status,
            judge_error_code=final.outcome.error_code,
            dimensions=final.outcome.scores.dimensions,
            j_project=final.outcome.scores.j_project,
            result_sha256=final.result_sha256,
        )

    if projection.judge_status != row.final_status or not math.isclose(
        projection.j_project,
        row.j_project,
        abs_tol=1e-12,
    ):
        raise PortfolioAttributionError(
            f"source smoke final projection differs: {row.query_id}"
        )
    return projection, content


def _validated_optimization_query_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PortfolioAttributionError("optimization query IDs must be a sequence")
    query_ids = tuple(values)
    if query_ids != PORTFOLIO_OPTIMIZATION_QUERY_IDS:
        raise PortfolioAttributionError(
            "optimization query universe must equal frozen dm-001..dm-025"
        )
    return query_ids


def build_portfolio_parent_attribution(
    *,
    stage: AttributionStage,
    smoke_root: str | Path,
    parent_bank: StaticBankArtifact,
    optimization_query_ids: Sequence[str],
) -> PortfolioParentAttributionPacket:
    """Build one S2/S3 packet from the exact current parent smoke run."""

    if stage not in _STAGE_CONFIG:
        raise PortfolioAttributionError(f"unsupported attribution stage: {stage}")
    try:
        parent_bank = StaticBankArtifact.model_validate(
            parent_bank.model_dump(mode="python"),
            strict=True,
        )
    except (AttributeError, ValidationError) as error:
        raise PortfolioAttributionError("current parent Bank is invalid") from error
    if {
        item.capability_id for item in parent_bank.capability_map
    } != PORTFOLIO_ATTRIBUTION_CAPABILITIES:
        raise PortfolioAttributionError(
            "current parent Bank does not cover the six Portfolio capabilities"
        )
    optimization_ids = _validated_optimization_query_ids(optimization_query_ids)
    optimization_set = set(optimization_ids)
    expected_config = _STAGE_CONFIG[stage]

    root = Path(smoke_root).absolute()
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise PortfolioAttributionError("source smoke root does not exist") from error
    if not root.is_dir():
        raise PortfolioAttributionError("source smoke root is not a directory")

    summary_bytes, summary = _read_canonical_object(
        root / "summary.json",
        label="source smoke summary",
    )
    summary_sha256 = _require_object_self_hash(
        summary,
        "summary_sha256",
        label="source smoke summary",
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != "portfolio-treatment-development-smoke-summary"
        or summary.get("policy_version") != _SMOKE_POLICY_VERSION
        or summary.get("status") != "complete"
        or summary.get("config") != expected_config
        or summary.get("target_bank_sha256") != parent_bank.bank_sha256
        or summary.get("query_count_requested") != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
        or summary.get("query_count_completed") != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
        or summary.get("results_file") != "results.jsonl"
        or not isinstance(summary.get("results_file_sha256"), str)
    ):
        raise PortfolioAttributionError(
            "source smoke summary is not bound to the current parent Bank/config"
        )

    results_bytes, rows = _load_results(
        root / "results.jsonl",
        expected_file_sha256=summary["results_file_sha256"],
    )
    query_ids = tuple(item.query_id for item in rows)
    capabilities = tuple(item.canonical_capability for item in rows)
    if (
        len(rows) != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
        or len(set(query_ids)) != PORTFOLIO_ATTRIBUTION_QUERY_COUNT
        or any(item not in optimization_set for item in query_ids)
        or list(query_ids) != summary.get("query_ids_requested")
        or list(query_ids) != summary.get("query_ids_completed")
        or list(capabilities) != summary.get("capabilities_completed")
        or dict(Counter(capabilities)) != PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS
    ):
        raise PortfolioAttributionError(
            "source smoke query/capability coverage differs from optimization25"
        )
    if any(
        row.config != expected_config
        or row.target_bank_sha256 != parent_bank.bank_sha256
        for row in rows
    ):
        raise PortfolioAttributionError(
            "source smoke result row is not current-parent-bound"
        )

    route_accuracy = sum(item.route_correct for item in rows) / len(rows)
    mean_j = sum(item.j_project for item in rows) / len(rows)
    hard_errors = sum(item.hard_error for item in rows)
    mean_skill_adherence = sum(item.skill_adherence for item in rows) / len(rows)
    if (
        not math.isclose(
            route_accuracy,
            float(summary.get("route_accuracy", -1.0)),
            abs_tol=1e-12,
        )
        or not math.isclose(
            mean_j,
            float(summary.get("mean_j_project", -1.0)),
            abs_tol=1e-12,
        )
        or hard_errors != summary.get("hard_error_count")
        or not math.isclose(
            mean_skill_adherence,
            float(summary.get("mean_skill_adherence", -1.0)),
            abs_tol=1e-12,
        )
    ):
        raise PortfolioAttributionError("source smoke summary metrics drifted")

    records: list[PortfolioAttributionRecord] = []
    for ordinal, row in enumerate(rows):
        assistant, assistant_bytes = _load_assistant(
            root,
            row,
            expected_config=expected_config,
            expected_bank_sha256=parent_bank.bank_sha256,
        )
        final, final_bytes = _load_final(root, row)
        expected_skill_adherence = calculate_portfolio_skill_adherence(
            bank=parent_bank,
            selected_capability=assistant.response.selected_capability,
            skill_slug=assistant.response.skill_slug,
            response_text=assistant.response.response_text,
            visible_cards=assistant.response.visible_cards,
            hard_error=row.hard_error,
        )
        if not math.isclose(
            row.skill_adherence,
            expected_skill_adherence,
            abs_tol=1e-12,
        ):
            raise PortfolioAttributionError(
                "source smoke Skill-adherence projection drifted"
            )
        common: dict[str, Any] = {
            "schema_version": 1,
            "record_ordinal": ordinal,
            "query_id": row.query_id,
            "canonical_capability": row.canonical_capability,
            "selected_capability": row.selected_capability,
            "route_correct": row.route_correct,
            "skill_slug": assistant.response.skill_slug,
            "route_trace_sha256": assistant.response.route_trace_sha256,
            "assistant_error_code": row.assistant_error_code,
            "source_result_sha256": row.result_sha256,
            "source_assistant_file_sha256": sha256_bytes(assistant_bytes),
            "source_assistant_row_sha256": assistant.row_sha256,
            "source_request_sha256": assistant.request.request_sha256,
            "source_response_sha256": assistant.receipt.response_sha256,
            "source_receipt_sha256": assistant.receipt.receipt_sha256,
            "source_final_file_sha256": sha256_bytes(final_bytes),
            "source_final_result_sha256": final.result_sha256,
        }
        if stage == "s2_route_optimizer":
            payload = {
                **common,
                "kind": "portfolio-current-parent-route-attribution",
                "tool_names": tuple(
                    item.tool_name for item in assistant.response.tool_trace
                ),
            }
            record = PortfolioRouteAttributionRecord.model_validate(
                {
                    **payload,
                    "record_sha256": sha256_bytes(
                        canonical_json_bytes(_jsonable(payload))
                    ),
                },
                strict=True,
            )
        else:
            payload = {
                **common,
                "kind": "portfolio-current-parent-body-attribution",
                "response_text": assistant.response.response_text,
                "visible_cards": assistant.response.visible_cards,
                "visible_tool_evidence": assistant.response.visible_tool_evidence,
                "tool_trace": assistant.response.tool_trace,
                "backend_error_code": assistant.response.error_code,
                "hard_error": row.hard_error,
                "final_kind": final.final_kind,
                "evaluation_id": final.evaluation_id,
                "judge_status": final.judge_status,
                "judge_error_code": final.judge_error_code,
                "dimensions": final.dimensions,
                "j_project": final.j_project,
            }
            record = PortfolioBodyAttributionRecord.model_validate(
                {
                    **payload,
                    "record_sha256": sha256_bytes(
                        canonical_json_bytes(_jsonable(payload))
                    ),
                },
                strict=True,
            )
        records.append(record)

    packet_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "portfolio-current-parent-attribution-packet",
        "policy_version": PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION,
        "stage": stage,
        "source_config": expected_config,
        "source_bank_sha256": parent_bank.bank_sha256,
        "optimization_query_ids_sha256": sha256_bytes(
            canonical_json_bytes(list(optimization_ids))
        ),
        "source_summary_file_sha256": sha256_bytes(summary_bytes),
        "source_summary_sha256": summary_sha256,
        "source_results_file_sha256": sha256_bytes(results_bytes),
        "query_ids": query_ids,
        "records": tuple(records),
    }
    json_payload = {
        key: (
            [item.model_dump(mode="json") for item in value]
            if key == "records"
            else list(value)
            if key == "query_ids"
            else value
        )
        for key, value in packet_payload.items()
    }
    return PortfolioParentAttributionPacket.model_validate(
        {
            **packet_payload,
            "packet_sha256": sha256_bytes(canonical_json_bytes(json_payload)),
        },
        strict=True,
    )


def load_portfolio_parent_attribution_packet(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioParentAttributionPacket:
    """Load one packet through an external file digest and replay its hashes."""

    if not _SHA256_RE.fullmatch(expected_file_sha256):
        raise PortfolioAttributionError("expected attribution file SHA-256 is invalid")
    try:
        content = read_stable_regular_file(
            path,
            label="Portfolio parent attribution packet",
            max_bytes=_MAX_FILE_BYTES,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioAttributionError("attribution packet cannot be read") from error
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioAttributionError("attribution packet file digest mismatch")
    try:
        packet = PortfolioParentAttributionPacket.model_validate_json(
            content,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioAttributionError("attribution packet is invalid") from error
    if packet.canonical_bytes() != content:
        raise PortfolioAttributionError("attribution packet is not canonical")
    return packet


def verify_portfolio_parent_attribution_gate_binding(
    packet: PortfolioParentAttributionPacket,
    parent_result: PortfolioStageGateResultSet,
) -> None:
    """Bind an S2/S3 attribution packet to the exact gate-parent smoke run."""

    try:
        attribution = PortfolioParentAttributionPacket.model_validate(
            packet.model_dump(mode="python"),
            strict=True,
        )
        gate_parent = PortfolioStageGateResultSet.model_validate(
            parent_result.model_dump(mode="python"),
            strict=True,
        )
    except (AttributeError, ValidationError) as error:
        raise PortfolioAttributionError(
            "attribution or gate-parent result set is invalid"
        ) from error
    source_result_sha256s = tuple(
        item.source_result_sha256
        for item in sorted(attribution.records, key=lambda item: item.query_id)
    )
    if (
        attribution.source_config != gate_parent.source_config
        or attribution.source_bank_sha256 != gate_parent.bank_sha256
        or tuple(sorted(attribution.query_ids)) != gate_parent.evaluation_query_ids
        or attribution.source_summary_file_sha256
        != gate_parent.source_summary_file_sha256
        or attribution.source_summary_sha256 != gate_parent.source_summary_sha256
        or attribution.source_results_file_sha256
        != gate_parent.source_results_file_sha256
        or source_result_sha256s != gate_parent.source_result_sha256s
    ):
        raise PortfolioAttributionError(
            "current-parent attribution differs from the gate-parent smoke run"
        )


__all__ = [
    "AttributionStage",
    "LEGACY_PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION",
    "PORTFOLIO_ATTRIBUTION_CAPABILITIES",
    "PORTFOLIO_ATTRIBUTION_QUERY_COUNT",
    "PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS",
    "PORTFOLIO_OPTIMIZATION_QUERY_IDS",
    "PORTFOLIO_OPTIMIZATION_QUERY_IDS_SHA256",
    "PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION",
    "PortfolioAttributionError",
    "PortfolioBodyAttributionRecord",
    "PortfolioParentAttributionPacket",
    "PortfolioRouteAttributionRecord",
    "build_portfolio_parent_attribution",
    "load_portfolio_parent_attribution_packet",
    "verify_portfolio_parent_attribution_gate_binding",
]
