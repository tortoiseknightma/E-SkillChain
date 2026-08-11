"""Deterministic Static-opt Feedback selection and typed S1 handoff artifacts.

The module deliberately reuses ``FeedbackPacket`` v2 and
``run_visual_feedback`` unchanged.  It owns only the trusted source binding,
48-row selection, an independent selected-asset authorization, immutable
per-row result envelopes, and the all-parsed S1 bundle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import AssetCatalog
from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.portfolio_core_inputs import VerifiedPortfolioCoreInputs
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
)
from skillchain.evaluation.packets import (
    FeedbackPacket,
    FeedbackGCSComponentBitsV1,
    FeedbackGCSContractV1,
    FeedbackGCSDiagnosticsV1,
    FeedbackPacketV3,
    RubricSnapshot,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
    VisibleCard,
    VisibleToolEvidence,
    build_feedback_packet,
    build_feedback_packet_v3,
)
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    gcs_policy_payload,
    gcs_v2_policy_payload,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    STATIC_GCS_CORPUS_POLICY_VERSION,
    VerifiedStaticGCSCorpus,
    VerifiedStaticGCSRow,
    accept_loaded_verified_static_gcs_corpus,
    require_verified_static_gcs_corpus,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS,
    QWEN37_FEEDBACK_JSON_SCHEMA_NAME,
    QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION,
    QWEN37_FEEDBACK_JSON_SCHEMA_SHA256,
    QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS,
    QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
    QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS,
    QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY,
    QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2,
    QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256_V2,
    QWEN37_FEEDBACK_PRICING_LOCK_SHA256_V2,
    QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2,
    QWEN37_FEEDBACK_SELECTED_COUNT_V2,
    QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V2,
    QWEN37_FEEDBACK_ROLE_SELECTION_SHA256_V2,
    QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
    QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
    QWEN37_FEEDBACK_THINKING_BUDGET,
    QWEN37_FEEDBACK_TIMEOUT_SECONDS,
    QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256,
    QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION,
    PortfolioS1QwenFeedbackAuthorizationV3,
    SelectedQwenFeedbackAssetV1,
    validate_selected_qwen_feedback_authorization,
    QWEN38_FEEDBACK_CACHE_NAMESPACE,
    QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS,
    QWEN38_FEEDBACK_JSON_SCHEMA_NAME,
    QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION,
    QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
    QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS,
    QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY,
    QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
    QWEN38_FEEDBACK_MODEL,
    QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS,
    QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY,
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V3,
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V3,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V4,
    QWEN38_FEEDBACK_PROCESSOR,
    QWEN38_FEEDBACK_PROVIDER_CALL_CEILING,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11,
    QWEN38_FEEDBACK_SELECTED_COUNT,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256,
    QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY,
    QWEN38_FEEDBACK_THINKING_BUDGET,
    QWEN38_FEEDBACK_TIMEOUT_SECONDS,
    QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256,
    QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION,
)
from skillchain.synthesis.portfolio_core_selection import CreatorSelectionEntry
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file
from skillchain.schemas import ConversationTurn


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FeedbackRole = Literal["failure", "success_anchor", "partial_anchor"]
FeedbackStatus = Literal["parsed", "parse_error", "provider_error", "timeout"]

FEEDBACK_SELECTION_POLICY_VERSION = "portfolio-s1-feedback-selection-v1"
FEEDBACK_AUTHORIZATION_POLICY_VERSION = (
    "portfolio-s1-feedback-selected-assets-authorization-v1"
)
FEEDBACK_AUTHORIZATION_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-selected-assets-authorization-v2"
)
FEEDBACK_CONTROL_POLICY_VERSION_V1 = "portfolio-s1-feedback-control-v1"
FEEDBACK_CONTROL_POLICY_VERSION_V2 = "portfolio-s1-feedback-control-v2"
FEEDBACK_CONTROL_POLICY_VERSION_V3 = "portfolio-s1-feedback-control-v3"
FEEDBACK_CONTROL_POLICY_VERSION_V4 = "portfolio-s1-feedback-control-v4"
FEEDBACK_CONTROL_POLICY_VERSION_V5 = "portfolio-s1-feedback-control-v5"
FEEDBACK_CONTROL_POLICY_VERSION_V6 = "portfolio-s1-feedback-control-v6"
FEEDBACK_CONTROL_POLICY_VERSION_V7 = "portfolio-s1-feedback-control-v7"
FEEDBACK_CONTROL_POLICY_VERSION = "portfolio-s1-feedback-control-v8"
BOUND_FEEDBACK_POLICY_VERSION = "portfolio-s1-bound-feedback-v1"
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-call-reservation-v1"
)
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-call-reservation-v2"
)
FEEDBACK_CALL_RESERVATION_POLICY_VERSION = "portfolio-s1-feedback-call-reservation-v3"
S1_FEEDBACK_BUNDLE_POLICY_VERSION = "portfolio-s1-feedback-bundle-v1"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V2 = "portfolio-s1-feedback-bundle-v2"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V3 = "portfolio-s1-feedback-bundle-v3"
S1_FEEDBACK_RUN_POLICY_VERSION = "portfolio-s1-feedback-run-v1"
FEEDBACK_SELECTION_SIZE = 48
FAILURES_PER_CAPABILITY = 6
ANCHORS_PER_CAPABILITY = 2

FEEDBACK_SELECTION_POLICY_VERSION_V2 = "portfolio-s1-feedback-selection-v2"
FEEDBACK_AUTHORIZATION_POLICY_VERSION_V4 = (
    "portfolio-s1-feedback-selected-assets-authorization-v4"
)
FEEDBACK_CONTROL_POLICY_VERSION_V9 = "portfolio-s1-feedback-control-v9"
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V4 = (
    "portfolio-s1-feedback-call-reservation-v4"
)
BOUND_FEEDBACK_POLICY_VERSION_V2 = "portfolio-s1-bound-feedback-v2"
S1_FEEDBACK_RUN_POLICY_VERSION_V2 = "portfolio-s1-feedback-run-v2"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V5 = "portfolio-s1-feedback-bundle-v5"
QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION_V2 = "portfolio-s1-qwen37-feedback-launch-lock-v2"
FEEDBACK_AUTHORIZATION_POLICY_VERSION_V5 = (
    "portfolio-s1-feedback-selected-assets-authorization-v5"
)
FEEDBACK_CONTROL_POLICY_VERSION_V10 = "portfolio-s1-feedback-control-v10"
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V5 = (
    "portfolio-s1-feedback-call-reservation-v5"
)
BOUND_FEEDBACK_POLICY_VERSION_V3 = "portfolio-s1-bound-feedback-v3"
S1_FEEDBACK_RUN_POLICY_VERSION_V3 = "portfolio-s1-feedback-run-v3"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V6 = "portfolio-s1-feedback-bundle-v6"
QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V3 = "portfolio-s1-qwen38-feedback-launch-lock-v3"
FEEDBACK_AUTHORIZATION_POLICY_VERSION_V6 = (
    "portfolio-s1-feedback-selected-assets-authorization-v6"
)
FEEDBACK_CONTROL_POLICY_VERSION_V11 = "portfolio-s1-feedback-control-v11"
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V6 = (
    "portfolio-s1-feedback-call-reservation-v6"
)
BOUND_FEEDBACK_POLICY_VERSION_V4 = "portfolio-s1-bound-feedback-v4"
S1_FEEDBACK_RUN_POLICY_VERSION_V4 = "portfolio-s1-feedback-run-v4"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V7 = "portfolio-s1-feedback-bundle-v7"
QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V4 = "portfolio-s1-qwen38-feedback-launch-lock-v4"
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-global-schema-retry-v1"
)
FEEDBACK_SELECTION_SIZE_V2 = 240
FEEDBACK_PHASE_COUNTS_V2 = (12, 60, 120, 240)
FEEDBACK_CAPABILITY_QUOTAS_V2 = {
    "knowledge.visual_encyclopedia": 55,
    "product.exact_match": 40,
    "product.multi_search": 40,
    "product.style_recommendation": 40,
    "utility.document_reading": 24,
    "utility.recipe_guidance": 41,
}

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CLUSTER_PRIORITY = (
    "style_evidence_invalid",
    "multi_mapping_invalid",
    "material_claim_uncited",
    "card_contract_failed",
    "evidence_or_output_contract",
    "route_or_tool_contract",
    "other",
)
_VERIFIED_STATIC_FEEDBACK_SOURCE_MARKER = object()

_DRIVE_PATH_RE = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)")
_POSIX_PRIVATE_PATH_RE = re.compile(r"(?i)(?:^|[\s\"'])/(?:home|mnt|users|tmp)/")
_DATA_URI_RE = re.compile(r"(?i)data:[^\s,;]{1,128}(?:;[^\s,;]{1,128})*;base64,")
_HASH_VALUE_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_BASE64_BLOB_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])"
)
_ASSET_V2_RE = re.compile(r"(?i)(?<![a-z0-9])asset(?:[._:-]?v2)[._:/-][^\s\"']+")
_PRIVATE_HANDLE_RE = re.compile(
    r"(?i)(?:scorer(?:[_ -](?:payload|handle|evidence))?|sidecar|"
    r"checkpoint(?:[_ -](?:row|file))?|leakage_group|packet_sha256|"
    r"artifact_sha256|feedback_result_sha256)"
)
_CREATOR_RUNTIME_HANDLE_RE = re.compile(
    r"(?i)(?:\bquery_id\b|\braw(?:_response(?:_text)?| response(?: text)?)\b|"
    r"\breasoning(?:_(?:tokens|bytes|sha256|content|text)| "
    r"(?:tokens|bytes|hash|content|trace))\b)"
)


class PortfolioS1FeedbackError(ValueError):
    """A Feedback selection, authorization, result, or bundle is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _hash_payload(payload: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(payload)))


def _model_hash(model: BaseModel, field: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field}))


def qwen38_feedback_global_retry_policy_v1() -> dict[str, object]:
    """Freeze the sole outer retry allowed around no-retry transport v6 calls."""

    return {
        "policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1,
        "scope": "exact-frozen-discovery-selected240",
        "selected_query_count": 240,
        "provider_call_ceiling": 241,
        "normal_attempts_per_selected_query": 1,
        "global_retry_token_count": 1,
        "max_attempts_per_retried_query": 2,
        "retry_policy": "one_global_same_entry_strict_schema_retry_v1",
        "retry_eligibility": {
            "status": "parse_error",
            "error_code": "invalid_feedback_json",
            "raw_response_text_required": True,
            "response_redaction_reason": None,
            "finish_reason": "stop",
            "tool_calls_required_empty": True,
            "parsed_feedback_required_null": True,
            "strict_parser_v3_must_raise": True,
        },
        "attempt_transport_policy_version": (
            "visual-feedback-qwen38-dashscope-json-schema-v6"
        ),
        "attempt_transport_retry_policy": ("no_internal_retry_each_provider_attempt"),
        "retry_identity": "same-selection-entry-no-replacement",
        "persistence_order": [
            "settle-first-attempt-create-only",
            "claim-global-retry-create-only",
            "reserve-second-attempt-create-only",
            "invoke-provider-once",
            "settle-second-attempt-create-only",
        ],
        "claim_without_second_reservation_resume": "allowed-same-claim-only",
        "reservation_without_settlement": "terminal-orphan-no-retry",
        "second_failure": "terminal-stop",
        "second-eligible-first-attempt-after-token-claimed": "terminal-stop",
        "provider_privacy_input_echo_or_orphan_retry": "forbidden",
    }


QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(qwen38_feedback_global_retry_policy_v1())
)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _nested_json_model(value: object, model_type: type[BaseModel]) -> object:
    """Re-enter JSON validation for a strict model nested in an envelope.

    Pydantic accepts JSON arrays for tuple fields when the model itself is the
    JSON-validation root.  A ``mode="before"`` validator on an outer envelope
    otherwise turns the nested value into an ordinary Python validation input,
    where strict tuple fields reject those same arrays.  Re-entering the nested
    model through its public strict JSON parser preserves both strictness and
    round-trip loading of canonical artifacts.
    """

    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate_json(canonical_json_bytes(value), strict=True)
    return value


def _nested_json_models(value: object, model_type: type[BaseModel]) -> object:
    if isinstance(value, list):
        return tuple(_nested_json_model(item, model_type) for item in value)
    return value


def _iter_string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, BaseModel):
        yield from _iter_string_values(value.model_dump(mode="json"))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_string_values(item)


def _validate_model_projection_privacy(
    projection: object,
    *,
    private_query_ids: tuple[str, ...],
) -> None:
    """Reject private values echoed into the Creator-visible projection.

    Public source, product, Style facet, and tool handles are intentionally
    allowed.  The policy targets execution identities, filesystem locations,
    hash-like secrets, and encoded payloads instead of banning every value
    whose field name happens to include ``handle``.
    """

    for text in _iter_string_values(projection):
        if any(query_id and query_id in text for query_id in private_query_ids):
            raise PortfolioS1FeedbackError(
                "S1 Feedback model projection contains a private query identity"
            )
        if (
            _ASSET_V2_RE.search(text)
            or _DRIVE_PATH_RE.search(text)
            or _POSIX_PRIVATE_PATH_RE.search(text)
            or "file://" in text.casefold()
            or _DATA_URI_RE.search(text)
            or _HASH_VALUE_RE.search(text)
            or _BASE64_BLOB_RE.search(text)
            or _PRIVATE_HANDLE_RE.search(text)
        ):
            raise PortfolioS1FeedbackError(
                "S1 Feedback model projection contains a forbidden private value"
            )


def require_feedback_v10_creator_projection_privacy(
    result: FeedbackEvaluationResult,
    *,
    private_query_ids: tuple[str, ...],
) -> FeedbackEvaluationResult:
    """Validate exactly the parsed fields that can flow into Creator V5.

    Provider raw text, request metadata, and thinking receipts are excluded by
    construction.  Values are then checked with the BundleV5 privacy scanner,
    plus explicit runtime-handle rejection, before a bound settlement may be
    published.
    """

    if (
        type(result) is not FeedbackEvaluationResult
        or result.cache_namespace != "feedback-evaluator-v10"
        or result.status != "parsed"
        or result.parsed_feedback is None
    ):
        raise PortfolioS1FeedbackError(
            "Creator projection privacy requires one parsed Feedback V10 result"
        )
    projection = result.parsed_feedback.model_dump(mode="json")
    _validate_model_projection_privacy(
        projection,
        private_query_ids=private_query_ids,
    )
    if any(
        _CREATOR_RUNTIME_HANDLE_RE.search(text)
        for text in _iter_string_values(projection)
    ):
        raise PortfolioS1FeedbackError(
            "S1 Feedback Creator projection contains private runtime metadata"
        )
    return result


def require_feedback_v11_creator_projection_privacy(
    result: FeedbackEvaluationResult,
    *,
    private_query_ids: tuple[str, ...],
) -> FeedbackEvaluationResult:
    """Validate the Qwen3.8-Max parsed fields that may flow into BundleV6."""

    if (
        type(result) is not FeedbackEvaluationResult
        or result.schema_version != 5
        or result.cache_namespace != "feedback-evaluator-v11"
        or result.model != "qwen3.8-max"
        or result.status != "parsed"
        or result.parsed_feedback is None
    ):
        raise PortfolioS1FeedbackError(
            "Creator projection privacy requires one parsed Feedback V11 result"
        )
    projection = result.parsed_feedback.model_dump(mode="json")
    _validate_model_projection_privacy(
        projection,
        private_query_ids=private_query_ids,
    )
    if any(
        _CREATOR_RUNTIME_HANDLE_RE.search(text)
        for text in _iter_string_values(projection)
    ):
        raise PortfolioS1FeedbackError(
            "S1 Feedback Creator projection contains private runtime metadata"
        )
    return result


def _seed_hash(seed: int, namespace: str, query_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\x00{namespace}\x00{query_id}".encode("utf-8")
    ).hexdigest()


class FeedbackSelectionStrataV1(_StrictFrozenModel):
    source_dataset: str
    repair_status: Literal["r3_language_repaired", "r3_carry_forward"]
    boundary_status: Literal["boundary", "non_boundary"]
    style_submode: str | None = None

    @field_validator("source_dataset")
    @classmethod
    def _source(cls, value: str) -> str:
        return _nonblank(value, "source_dataset")


class FeedbackSelectionEntryV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    selection_ordinal: int = Field(ge=1, le=FEEDBACK_SELECTION_SIZE)
    creator_selection_rank: int = Field(ge=1, le=240)
    query_id: str
    capability: str
    role: FeedbackRole
    primary_cluster: str
    leakage_group_id: str
    atomic_component_id: str
    asset_id: str
    image_sha256: Sha256
    gcs: Literal[0, 1]
    gcs_component_pass_count: int = Field(ge=0, le=5)
    reason_codes: tuple[str, ...]
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    strata: FeedbackSelectionStrataV1
    deterministic_tie_sha256: Sha256
    entry_sha256: Sha256

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _tuple_reasons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "query_id",
        "capability",
        "primary_cluster",
        "leakage_group_id",
        "atomic_component_id",
        "asset_id",
    )
    @classmethod
    def _text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_entry(self) -> Self:
        if self.capability not in GCS_CAPABILITY_ORDER:
            raise ValueError("selection capability is unknown")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("selection reason codes must be sorted and unique")
        if self.role == "success_anchor":
            if self.gcs != 1 or self.primary_cluster != "success":
                raise ValueError("success anchor must be a GCS success")
        elif self.role == "partial_anchor":
            if self.gcs != 0 or self.primary_cluster != "partial_anchor":
                raise ValueError("partial anchor must be an explicit near-pass")
        elif self.gcs != 0 or self.primary_cluster not in _CLUSTER_PRIORITY:
            raise ValueError("failure selection has an invalid cluster")
        if self.entry_sha256 != _model_hash(self, "entry_sha256"):
            raise ValueError("Feedback selection entry self hash mismatch")
        return self


class PortfolioS1FeedbackSelectionV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-selection"] = "portfolio-s1-feedback-selection"
    policy_version: Literal["portfolio-s1-feedback-selection-v1"] = (
        FEEDBACK_SELECTION_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    corpus_sha256: Sha256
    creator_selection_file_sha256: Sha256
    creator_selection_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    selected_count: Literal[48] = FEEDBACK_SELECTION_SIZE
    failure_count: Literal[36] = 36
    anchor_count: Literal[12] = 12
    excluded_execution_lapse_query_ids: tuple[str, ...]
    entries: tuple[FeedbackSelectionEntryV1, ...]
    canary_entry_sha256s: tuple[Sha256, ...]
    canary_query_ids: tuple[str, ...]
    remaining_entry_sha256s: tuple[Sha256, ...]
    remaining_query_ids: tuple[str, ...]
    selected_asset_set_sha256: Sha256
    selection_sha256: Sha256

    @field_validator(
        "excluded_execution_lapse_query_ids",
        "entries",
        "canary_entry_sha256s",
        "canary_query_ids",
        "remaining_entry_sha256s",
        "remaining_query_ids",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if len(self.entries) != FEEDBACK_SELECTION_SIZE:
            raise ValueError("Feedback selection must contain exactly 48 entries")
        if tuple(item.selection_ordinal for item in self.entries) != tuple(
            range(1, FEEDBACK_SELECTION_SIZE + 1)
        ):
            raise ValueError("Feedback selection ordinals must be contiguous")
        unique_fields = (
            tuple(item.query_id for item in self.entries),
            tuple(item.leakage_group_id for item in self.entries),
            tuple(item.asset_id for item in self.entries),
            tuple(item.entry_sha256 for item in self.entries),
        )
        if any(len(set(values)) != FEEDBACK_SELECTION_SIZE for values in unique_fields):
            raise ValueError(
                "Feedback selection requires unique query/leakage/asset/entry identity"
            )
        role_counts = Counter(item.role for item in self.entries)
        if (
            role_counts["failure"] != 36
            or sum(role_counts[item] for item in ("success_anchor", "partial_anchor"))
            != 12
        ):
            raise ValueError("Feedback selection role counts are invalid")
        for capability in GCS_CAPABILITY_ORDER:
            members = [item for item in self.entries if item.capability == capability]
            if (
                len(members) != 8
                or sum(item.role == "failure" for item in members) != 6
            ):
                raise ValueError("Feedback selection capability quota drifted")
        canaries = tuple(
            next(
                item
                for item in self.entries
                if item.capability == capability and item.role == "failure"
            )
            for capability in GCS_CAPABILITY_ORDER
        )
        canary_hashes = tuple(item.entry_sha256 for item in canaries)
        canary_queries = tuple(item.query_id for item in canaries)
        remaining = tuple(
            item for item in self.entries if item.entry_sha256 not in canary_hashes
        )
        if (
            self.canary_entry_sha256s != canary_hashes
            or self.canary_query_ids != canary_queries
            or self.remaining_entry_sha256s
            != tuple(item.entry_sha256 for item in remaining)
            or self.remaining_query_ids != tuple(item.query_id for item in remaining)
            or len(remaining) != 42
        ):
            raise ValueError("Feedback canary/remaining partition drifted")
        asset_payload = [
            {"asset_id": item.asset_id, "image_sha256": item.image_sha256}
            for item in sorted(self.entries, key=lambda entry: entry.asset_id)
        ]
        if self.selected_asset_set_sha256 != _hash_payload(asset_payload):
            raise ValueError("selected asset set hash mismatch")
        if self.selection_sha256 != _model_hash(self, "selection_sha256"):
            raise ValueError("Feedback selection self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _primary_cluster(row: VerifiedStaticGCSRow) -> str:
    reasons = set(row.score.reason_codes)
    if "style_evidence_invalid" in reasons:
        return "style_evidence_invalid"
    if "multi_mapping_invalid" in reasons:
        return "multi_mapping_invalid"
    if "material_claim_uncited" in reasons:
        return "material_claim_uncited"
    if "card_contract_failed" in reasons:
        return "card_contract_failed"
    if not row.score.evidence_grounded or not row.score.output_contract_pass:
        return "evidence_or_output_contract"
    if not row.score.route_acceptable or not row.score.tool_contract_pass:
        return "route_or_tool_contract"
    return "other"


def _pass_count(row: VerifiedStaticGCSRow) -> int:
    return sum(
        int(value)
        for value in (
            row.score.route_acceptable,
            row.score.no_hard_error,
            row.score.tool_contract_pass,
            row.score.evidence_grounded,
            row.score.output_contract_pass,
        )
    )


def _strata_key(row: VerifiedStaticGCSRow) -> tuple[str, ...]:
    return (
        row.strata.source_dataset,
        row.strata.repair_status,
        row.strata.boundary_status,
        row.strata.style_submode or "not_applicable",
    )


def _pick_failures(
    rows: list[VerifiedStaticGCSRow],
    *,
    seed: int,
    capability: str,
    excluded_query_ids: set[str],
    used_assets: set[str],
) -> list[VerifiedStaticGCSRow]:
    eligible = [
        row
        for row in rows
        if row.score.gcs == 0
        and row.score.hard_error == 0
        and row.query.query_id not in excluded_query_ids
        and row.query.asset_id not in used_assets
    ]
    grouped: dict[str, list[VerifiedStaticGCSRow]] = defaultdict(list)
    for row in eligible:
        grouped[_primary_cluster(row)].append(row)
    for cluster, members in grouped.items():
        members.sort(
            key=lambda row: _seed_hash(
                seed, f"failure:{capability}:{cluster}", row.query.query_id
            )
        )
    selected: list[VerifiedStaticGCSRow] = []
    selected_strata: set[tuple[str, ...]] = set()
    for cluster in _CLUSTER_PRIORITY:
        if len(selected) >= FAILURES_PER_CAPABILITY:
            break
        candidates = [
            row
            for row in grouped.get(cluster, ())
            if row.query.asset_id not in used_assets
        ]
        if not candidates:
            continue
        row = candidates[0]
        selected.append(row)
        selected_strata.add(_strata_key(row))
        used_assets.add(row.query.asset_id)
    while len(selected) < FAILURES_PER_CAPABILITY:
        candidates = [
            row
            for row in eligible
            if row not in selected and row.query.asset_id not in used_assets
        ]
        if not candidates:
            raise PortfolioS1FeedbackError(
                f"{capability} cannot supply six component-unique failures"
            )
        candidates.sort(
            key=lambda row: (
                -int(_strata_key(row) not in selected_strata),
                -len(grouped[_primary_cluster(row)]),
                _CLUSTER_PRIORITY.index(_primary_cluster(row)),
                _seed_hash(seed, f"failure-fill:{capability}", row.query.query_id),
            )
        )
        row = candidates[0]
        selected.append(row)
        selected_strata.add(_strata_key(row))
        used_assets.add(row.query.asset_id)
    return selected


def _pick_anchors(
    rows: list[VerifiedStaticGCSRow],
    *,
    seed: int,
    capability: str,
    used_assets: set[str],
) -> list[tuple[VerifiedStaticGCSRow, FeedbackRole]]:
    eligible = [
        row
        for row in rows
        if row.score.hard_error == 0 and row.query.asset_id not in used_assets
    ]
    successes = [row for row in eligible if row.score.gcs == 1]
    successes.sort(
        key=lambda row: _seed_hash(seed, f"success:{capability}", row.query.query_id)
    )
    chosen: list[tuple[VerifiedStaticGCSRow, FeedbackRole]] = []
    for row in successes:
        if len(chosen) == ANCHORS_PER_CAPABILITY:
            break
        chosen.append((row, "success_anchor"))
        used_assets.add(row.query.asset_id)
    if len(chosen) < ANCHORS_PER_CAPABILITY:
        partials = [
            row
            for row in eligible
            if row.score.gcs == 0 and row.query.asset_id not in used_assets
        ]
        partials.sort(
            key=lambda row: (
                -_pass_count(row),
                len(row.score.reason_codes),
                _seed_hash(seed, f"partial:{capability}", row.query.query_id),
            )
        )
        for row in partials:
            if len(chosen) == ANCHORS_PER_CAPABILITY:
                break
            chosen.append((row, "partial_anchor"))
            used_assets.add(row.query.asset_id)
    if len(chosen) != ANCHORS_PER_CAPABILITY:
        raise PortfolioS1FeedbackError(f"{capability} cannot supply two anchors")
    return chosen


def build_portfolio_s1_feedback_selection(
    corpus: VerifiedStaticGCSCorpus,
    creator_entries: tuple[CreatorSelectionEntry, ...],
    *,
    creator_selection_file_sha256: str,
    creator_selection_sha256: str,
    seed: int = 20260808,
) -> PortfolioS1FeedbackSelectionV1:
    """Choose 36 failures and 12 anchors only from the frozen Creator240."""

    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    if seed < 0:
        raise PortfolioS1FeedbackError("Feedback selection seed must be non-negative")
    if len(creator_entries) != 240:
        raise PortfolioS1FeedbackError("Feedback selection requires Creator240")
    if not _SHA_RE.fullmatch(creator_selection_file_sha256) or not _SHA_RE.fullmatch(
        creator_selection_sha256
    ):
        raise PortfolioS1FeedbackError("Creator selection hashes are invalid")
    if tuple(item.selection_rank for item in creator_entries) != tuple(range(1, 241)):
        raise PortfolioS1FeedbackError("Creator selection ranks are not contiguous")
    if len({item.component_id for item in creator_entries}) != 240:
        raise PortfolioS1FeedbackError("Creator240 leakage groups must be unique")
    row_by_query = verified.row_by_query_id()
    creator_by_query = {item.plan_id: item for item in creator_entries}
    if len(creator_by_query) != 240 or not set(creator_by_query).issubset(row_by_query):
        raise PortfolioS1FeedbackError("Creator240 differs from the Static corpus")
    for query_id, creator in creator_by_query.items():
        row = row_by_query[query_id]
        if (
            creator.component_id != row.leakage_group_id
            or creator.canonical_capability != row.query.canonical_capability
        ):
            raise PortfolioS1FeedbackError("Creator240 row binding drifted")

    selected_rows: list[tuple[VerifiedStaticGCSRow, FeedbackRole]] = []
    used_assets: set[str] = set()
    excluded_execution_lapses = tuple(
        sorted(
            query_id
            for query_id in creator_by_query
            if row_by_query[query_id].score.hard_error == 1
        )
    )
    for capability in GCS_CAPABILITY_ORDER:
        candidates = [
            row_by_query[query_id]
            for query_id, item in creator_by_query.items()
            if item.canonical_capability == capability
        ]
        anchors = _pick_anchors(
            candidates,
            seed=seed,
            capability=capability,
            used_assets=used_assets,
        )
        anchor_query_ids = {row.query.query_id for row, _role in anchors}
        failures = _pick_failures(
            candidates,
            seed=seed,
            capability=capability,
            excluded_query_ids=anchor_query_ids,
            used_assets=used_assets,
        )
        selected_rows.extend((row, "failure") for row in failures)
        selected_rows.extend(anchors)

    entries: list[FeedbackSelectionEntryV1] = []
    for ordinal, (row, role) in enumerate(selected_rows, start=1):
        creator = creator_by_query[row.query.query_id]
        cluster = (
            _primary_cluster(row)
            if role == "failure"
            else ("success" if role == "success_anchor" else "partial_anchor")
        )
        unsigned = {
            "schema_version": 1,
            "selection_ordinal": ordinal,
            "creator_selection_rank": creator.selection_rank,
            "query_id": row.query.query_id,
            "capability": row.query.canonical_capability,
            "role": role,
            "primary_cluster": cluster,
            "leakage_group_id": row.leakage_group_id,
            "atomic_component_id": row.atomic_component_id,
            "asset_id": row.query.asset_id,
            "image_sha256": row.image_sha256,
            "gcs": row.score.gcs,
            "gcs_component_pass_count": _pass_count(row),
            "reason_codes": row.score.reason_codes,
            "checkpoint_file_sha256": row.checkpoint_file_sha256,
            "checkpoint_row_sha256": row.checkpoint_row_sha256,
            "sidecar_sha256": row.sidecar.evidence_sha256,
            "strata": FeedbackSelectionStrataV1(
                source_dataset=row.strata.source_dataset,
                repair_status=row.strata.repair_status,  # type: ignore[arg-type]
                boundary_status=row.strata.boundary_status,  # type: ignore[arg-type]
                style_submode=row.strata.style_submode,
            ),
            "deterministic_tie_sha256": _seed_hash(
                seed, f"selected:{role}:{cluster}", row.query.query_id
            ),
        }
        entries.append(
            FeedbackSelectionEntryV1.model_validate(
                {**unsigned, "entry_sha256": _hash_payload(unsigned)}, strict=True
            )
        )
    asset_payload = [
        {"asset_id": item.asset_id, "image_sha256": item.image_sha256}
        for item in sorted(entries, key=lambda entry: entry.asset_id)
    ]
    canaries = tuple(
        next(
            item
            for item in entries
            if item.capability == capability and item.role == "failure"
        )
        for capability in GCS_CAPABILITY_ORDER
    )
    canary_hashes = tuple(item.entry_sha256 for item in canaries)
    remaining = tuple(
        item for item in entries if item.entry_sha256 not in canary_hashes
    )
    unsigned_selection = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-selection",
        "policy_version": FEEDBACK_SELECTION_POLICY_VERSION,
        "seed": seed,
        "corpus_sha256": verified.corpus_sha256,
        "creator_selection_file_sha256": creator_selection_file_sha256,
        "creator_selection_sha256": creator_selection_sha256,
        "parent_static_bank_sha256": verified.runtime.bank.bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": FEEDBACK_SELECTION_SIZE,
        "failure_count": 36,
        "anchor_count": 12,
        "excluded_execution_lapse_query_ids": excluded_execution_lapses,
        "entries": tuple(entries),
        "canary_entry_sha256s": canary_hashes,
        "canary_query_ids": tuple(item.query_id for item in canaries),
        "remaining_entry_sha256s": tuple(item.entry_sha256 for item in remaining),
        "remaining_query_ids": tuple(item.query_id for item in remaining),
        "selected_asset_set_sha256": _hash_payload(asset_payload),
    }
    return PortfolioS1FeedbackSelectionV1.model_validate(
        {
            **unsigned_selection,
            "selection_sha256": _hash_payload(_jsonable(unsigned_selection)),
        },
        strict=True,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def write_portfolio_s1_feedback_selection(
    path: str | Path,
    selection: PortfolioS1FeedbackSelectionV1,
) -> Path:
    return atomic_create_file(path, selection.canonical_bytes())


def load_portfolio_s1_feedback_selection(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackSelectionV1:
    content = read_stable_regular_file(
        path, label="S1 Feedback selection", max_bytes=8 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback selection file SHA-256 mismatch")
    selection = PortfolioS1FeedbackSelectionV1.model_validate_json(content, strict=True)
    if selection.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback selection is not canonical JSON")
    return selection


class SelectedFeedbackAssetV1(_StrictFrozenModel):
    query_id: str
    asset_id: str
    image_sha256: Sha256


class PortfolioS1FeedbackAuthorizationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v1"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-selected-48-assets"] = (
        "core-opt800-s1-feedback-selected-48-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    selection_sha256: Sha256
    provider: Literal["gemini"] = "gemini"
    model: Literal["gemini-3.6-flash"] = "gemini-3.6-flash"
    processor: Literal["aifast-gemini-feedback"] = "aifast-gemini-feedback"
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    selected_assets: tuple[SelectedFeedbackAssetV1, ...]
    selected_asset_set_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator("selected_assets", mode="before")
    @classmethod
    def _assets_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("authorization_id", "reviewer_id", "owner_statement")
    @classmethod
    def _texts(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Feedback authorization reviewed_at needs a timezone")
        if len(self.selected_assets) != FEEDBACK_SELECTION_SIZE:
            raise ValueError("Feedback authorization must bind exactly 48 assets")
        if tuple(item.query_id for item in self.selected_assets) != tuple(
            sorted(item.query_id for item in self.selected_assets)
        ):
            raise ValueError("authorized assets must be query-sorted")
        if (
            len({item.query_id for item in self.selected_assets}) != 48
            or len({item.asset_id for item in self.selected_assets}) != 48
        ):
            raise ValueError("authorized query and asset identities must be unique")
        payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != _hash_payload(payload):
            raise ValueError("authorized asset set hash mismatch")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Feedback authorization self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1FeedbackAuthorizationV2(_StrictFrozenModel):
    """Selected48 Kimi scope derived from the verified Core authority."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v2"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION_V2
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-selected-48-assets"] = (
        "core-opt800-s1-feedback-selected-48-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    selection_sha256: Sha256
    provider: Literal["kimi"] = "kimi"
    model: Literal["kimi-k2.6"] = "kimi-k2.6"
    processor: Literal["dashscope-kimi-feedback"] = "dashscope-kimi-feedback"
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    parent_remote_catalog_sha256: Sha256
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    selected_assets: tuple[SelectedFeedbackAssetV1, ...]
    selected_asset_set_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator("selected_assets", mode="before")
    @classmethod
    def _assets_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "authorization_id",
        "reviewer_id",
        "owner_statement",
        "parent_remote_authorization_id",
    )
    @classmethod
    def _texts(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Feedback authorization reviewed_at needs a timezone")
        if len(self.selected_assets) != FEEDBACK_SELECTION_SIZE:
            raise ValueError("Feedback authorization must bind exactly 48 assets")
        if tuple(item.query_id for item in self.selected_assets) != tuple(
            sorted(item.query_id for item in self.selected_assets)
        ):
            raise ValueError("authorized assets must be query-sorted")
        if (
            len({item.query_id for item in self.selected_assets}) != 48
            or len({item.asset_id for item in self.selected_assets}) != 48
        ):
            raise ValueError("authorized query and asset identities must be unique")
        payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != _hash_payload(payload):
            raise ValueError("authorized asset set hash mismatch")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Feedback authorization self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


PortfolioS1FeedbackAuthorization = (
    PortfolioS1FeedbackAuthorizationV1
    | PortfolioS1FeedbackAuthorizationV2
    | PortfolioS1QwenFeedbackAuthorizationV3
)


def write_selected_feedback_authorization(
    path: str | Path,
    authorization: PortfolioS1FeedbackAuthorization,
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def load_selected_feedback_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV1:
    content = read_stable_regular_file(
        path, label="selected Feedback authorization", max_bytes=2 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError(
            "selected Feedback authorization file SHA-256 mismatch"
        )
    authorization = PortfolioS1FeedbackAuthorizationV1.model_validate_json(
        content, strict=True
    )
    if authorization.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "selected Feedback authorization is not canonical JSON"
        )
    return authorization


def load_selected_kimi_feedback_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV2:
    """Load the active selected48 Kimi authorization without retyping v1."""

    content = read_stable_regular_file(
        path, label="selected Kimi Feedback authorization", max_bytes=2 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError(
            "selected Kimi Feedback authorization file SHA-256 mismatch"
        )
    authorization = PortfolioS1FeedbackAuthorizationV2.model_validate_json(
        content, strict=True
    )
    if authorization.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "selected Kimi Feedback authorization is not canonical JSON"
        )
    return authorization


def validate_selected_feedback_authorization(
    authorization: PortfolioS1FeedbackAuthorization,
    selection: PortfolioS1FeedbackSelectionV1,
) -> None:
    if type(authorization) is PortfolioS1QwenFeedbackAuthorizationV3:
        validate_selected_qwen_feedback_authorization(authorization, selection)
        return
    expected = tuple(
        SelectedFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )
    if (
        authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != expected
        or authorization.selected_asset_set_sha256
        != _hash_payload([item.model_dump(mode="json") for item in expected])
    ):
        raise PortfolioS1FeedbackError(
            "selected-asset authorization differs from the 48-row selection"
        )


def build_selected_feedback_authorization(
    selection: PortfolioS1FeedbackSelectionV1,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
) -> PortfolioS1FeedbackAuthorizationV1:
    """Build, but never infer, one explicit owner-approved selected-asset scope."""

    selected_assets = tuple(
        SelectedFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )
    draft = PortfolioS1FeedbackAuthorizationV1.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        owner_statement=owner_statement,
        selection_sha256=selection.selection_sha256,
        selected_assets=selected_assets,
        selected_asset_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in selected_assets]
        ),
        authorization_sha256="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackAuthorizationV1.model_validate_json(
        canonical_json_bytes(
            {**payload, "authorization_sha256": _hash_payload(payload)}
        ),
        strict=True,
    )


def build_selected_kimi_feedback_authorization(
    selection: PortfolioS1FeedbackSelectionV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
) -> PortfolioS1FeedbackAuthorizationV2:
    """Derive the new selected48 scope from genuine Core Kimi authority."""

    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-kimi-feedback",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    if "dashscope-kimi-feedback" not in parent.authorization.processor_scope:
        raise PortfolioS1FeedbackError(
            "parent Core authority does not authorize Kimi Feedback"
        )
    selected_assets = tuple(
        SelectedFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )
    draft = PortfolioS1FeedbackAuthorizationV2.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        owner_statement=owner_statement,
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=(parent.authorization_file_sha256),
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        parent_remote_catalog_sha256=parent.catalog.catalog_sha256,
        selected_assets=selected_assets,
        selected_asset_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in selected_assets]
        ),
        authorization_sha256="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackAuthorizationV2.model_validate_json(
        canonical_json_bytes(
            {**payload, "authorization_sha256": _hash_payload(payload)}
        ),
        strict=True,
    )


class PortfolioS1FeedbackControlV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-control"] = "portfolio-s1-feedback-control"
    policy_version: Literal[
        "portfolio-s1-feedback-control-v1",
        "portfolio-s1-feedback-control-v2",
        "portfolio-s1-feedback-control-v3",
        "portfolio-s1-feedback-control-v4",
        "portfolio-s1-feedback-control-v5",
        "portfolio-s1-feedback-control-v6",
        "portfolio-s1-feedback-control-v7",
        "portfolio-s1-feedback-control-v8",
    ] = FEEDBACK_CONTROL_POLICY_VERSION
    parser_policy_version: (
        Literal[
            "visual-feedback-complete-fence-wrapper-v2",
            "visual-feedback-free-text-trim-v3",
        ]
        | None
    ) = None
    parser_policy_sha256: Sha256 | None = None
    prompt_policy_version: (
        Literal[
            "visual-feedback-exact-shape-prompt-v4",
            "visual-feedback-response-schema-v1-prompt-v5",
        ]
        | None
    ) = None
    prompt_policy_sha256: Sha256 | None = None
    transport_policy_version: (
        Literal[
            "visual-feedback-json-object-response-format-v1",
            "visual-feedback-kimi-dashscope-plain-json-v2",
            "visual-feedback-kimi-dashscope-plain-json-v3",
            "visual-feedback-qwen-dashscope-json-object-v4",
            "visual-feedback-qwen-dashscope-json-schema-v5",
        ]
        | None
    ) = None
    transport_policy_sha256: Sha256 | None = None
    requested_response_format: Literal["json_object", "json_schema"] | None = None
    requested_json_schema_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    requested_thinking: bool | None = None
    requested_thinking_budget: Literal[2048] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    requested_timeout_seconds: Literal[600] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    requested_temperature: Literal[0.6] | None = None
    requested_top_p: Literal[0.95] | None = None
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_id: str
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    canary_entry_sha256s: tuple[Sha256, ...]
    canary_query_ids: tuple[str, ...]
    remaining_entry_sha256s: tuple[Sha256, ...]
    remaining_query_ids: tuple[str, ...]
    rubric: RubricSnapshot
    provider: Literal["gemini", "kimi", "qwen"] = "gemini"
    model: Literal[
        "gemini-3.6-flash",
        "kimi-k2.6",
        "qwen3.7-plus-2026-05-26",
    ] = "gemini-3.6-flash"
    processor: Literal[
        "aifast-gemini-feedback",
        "dashscope-kimi-feedback",
        "dashscope-qwen37-feedback",
    ] = "aifast-gemini-feedback"
    selected_query_count: Literal[48] = 48
    provider_call_ceiling: Literal[48] = 48
    max_tokens: Literal[2048] | None = 2048
    max_completion_tokens: Literal[4096] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    max_attempts: Literal[1] = 1
    feedback_concurrency: Literal[1, 2] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    create_only_resume: Literal[True] = True
    require_all_parsed_for_bundle: Literal[True] = True
    control_sha256: Sha256

    @field_validator(
        "canary_entry_sha256s",
        "canary_query_ids",
        "remaining_entry_sha256s",
        "remaining_query_ids",
        mode="before",
    )
    @classmethod
    def _partitions_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V1:
            if (
                self.parser_policy_version is not None
                or self.parser_policy_sha256 is not None
            ):
                raise ValueError("legacy Feedback control cannot claim a parser policy")
        else:
            expected_parser_identity = {
                FEEDBACK_CONTROL_POLICY_VERSION_V2: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION_V3: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION_V4: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION_V5: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION_V6: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION_V7: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                FEEDBACK_CONTROL_POLICY_VERSION: (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
            }[self.policy_version]
            if (
                self.parser_policy_version,
                self.parser_policy_sha256,
            ) != expected_parser_identity:
                raise ValueError("Feedback control parser policy identity mismatch")
        if self.policy_version in {
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
            ):
                raise ValueError("Feedback control prompt policy identity mismatch")
        elif self.policy_version in {
            FEEDBACK_CONTROL_POLICY_VERSION_V4,
            FEEDBACK_CONTROL_POLICY_VERSION_V5,
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
        }:
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
            ):
                raise ValueError("Feedback control prompt policy identity mismatch")
        elif (
            self.prompt_policy_version is not None
            or self.prompt_policy_sha256 is not None
        ):
            raise ValueError(
                "historical Feedback control cannot claim an active prompt policy"
            )
        if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V5:
            if (
                self.transport_policy_version,
                self.transport_policy_sha256,
                self.requested_response_format,
            ) != (
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
                "json_object",
            ):
                raise ValueError("Feedback control transport policy identity mismatch")
        elif self.policy_version in {
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
        }:
            expected_transport_identity = (
                (
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
                )
                if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V7
                else (
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2,
                )
            )
            if (
                self.transport_policy_version,
                self.transport_policy_sha256,
                self.requested_response_format,
                self.requested_json_schema_sha256,
                self.requested_thinking,
                self.requested_thinking_budget,
                self.requested_timeout_seconds,
                self.requested_temperature,
                self.requested_top_p,
            ) != (
                *expected_transport_identity,
                None,
                None,
                False,
                None,
                None,
                0.6,
                0.95,
            ):
                raise ValueError("Feedback control transport policy identity mismatch")
        elif self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION:
            if (
                self.transport_policy_version,
                self.transport_policy_sha256,
                self.requested_response_format,
                self.requested_json_schema_sha256,
                self.requested_thinking,
                self.requested_thinking_budget,
                self.requested_timeout_seconds,
                self.requested_temperature,
                self.requested_top_p,
            ) != (
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
                "json_schema",
                VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
                True,
                2048,
                600,
                None,
                None,
            ):
                raise ValueError("Feedback control transport policy identity mismatch")
        elif (
            self.transport_policy_version is not None
            or self.transport_policy_sha256 is not None
            or self.requested_response_format is not None
            or self.requested_json_schema_sha256 is not None
            or self.requested_thinking is not None
            or self.requested_thinking_budget is not None
            or self.requested_timeout_seconds is not None
            or self.requested_temperature is not None
            or self.requested_top_p is not None
        ):
            raise ValueError(
                "historical Feedback control cannot claim JSON-object transport"
            )
        payload = self.model_dump(mode="json", exclude={"control_sha256"})
        if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V1:
            payload.pop("parser_policy_version", None)
            payload.pop("parser_policy_sha256", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V4,
            FEEDBACK_CONTROL_POLICY_VERSION_V5,
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("prompt_policy_version", None)
            payload.pop("prompt_policy_sha256", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V5,
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("transport_policy_version", None)
            payload.pop("transport_policy_sha256", None)
            payload.pop("requested_response_format", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("requested_thinking", None)
            payload.pop("requested_temperature", None)
            payload.pop("requested_top_p", None)
        if self.policy_version != FEEDBACK_CONTROL_POLICY_VERSION:
            payload.pop("requested_json_schema_sha256", None)
            payload.pop("requested_thinking_budget", None)
            payload.pop("requested_timeout_seconds", None)
            payload.pop("max_completion_tokens", None)
        if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION:
            if self.max_tokens is not None or self.max_completion_tokens != 4096:
                raise ValueError(
                    "Qwen Feedback control completion-token identity mismatch"
                )
            if self.feedback_concurrency != 2:
                raise ValueError("Qwen Feedback control concurrency must be 2")
            expected_runtime = (
                "qwen",
                "qwen3.7-plus-2026-05-26",
                "dashscope-qwen37-feedback",
            )
        elif self.policy_version in {
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
        }:
            expected_runtime = ("kimi", "kimi-k2.6", "dashscope-kimi-feedback")
        else:
            expected_runtime = (
                "gemini",
                "gemini-3.6-flash",
                "aifast-gemini-feedback",
            )
        if (self.provider, self.model, self.processor) != expected_runtime:
            raise ValueError("Feedback control provider identity mismatch")
        if self.control_sha256 != _hash_payload(payload):
            raise ValueError("Feedback control self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        if self.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V1:
            payload.pop("parser_policy_version", None)
            payload.pop("parser_policy_sha256", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V4,
            FEEDBACK_CONTROL_POLICY_VERSION_V5,
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("prompt_policy_version", None)
            payload.pop("prompt_policy_sha256", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V5,
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("transport_policy_version", None)
            payload.pop("transport_policy_sha256", None)
            payload.pop("requested_response_format", None)
        if self.policy_version not in {
            FEEDBACK_CONTROL_POLICY_VERSION_V6,
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }:
            payload.pop("requested_thinking", None)
            payload.pop("requested_temperature", None)
            payload.pop("requested_top_p", None)
        if self.policy_version != FEEDBACK_CONTROL_POLICY_VERSION:
            payload.pop("requested_json_schema_sha256", None)
            payload.pop("requested_thinking_budget", None)
            payload.pop("requested_timeout_seconds", None)
            payload.pop("max_completion_tokens", None)
        return canonical_json_bytes(payload)


def write_portfolio_s1_feedback_control(
    path: str | Path,
    control: PortfolioS1FeedbackControlV1,
) -> Path:
    return atomic_create_file(path, control.canonical_bytes())


def load_portfolio_s1_feedback_control(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackControlV1:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback control", max_bytes=2 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback control file SHA-256 mismatch")
    control = PortfolioS1FeedbackControlV1.model_validate_json(content, strict=True)
    if control.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback control is not canonical JSON")
    return control


def validate_portfolio_s1_feedback_control(
    control: PortfolioS1FeedbackControlV1,
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorization,
) -> None:
    validate_selected_feedback_authorization(authorization, selection)
    common_drift = (
        control.selection_sha256 != selection.selection_sha256
        or control.authorization_sha256 != authorization.authorization_sha256
        or control.authorization_id != authorization.authorization_id
        or control.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or control.corpus_sha256 != selection.corpus_sha256
        or control.parent_static_bank_sha256 != selection.parent_static_bank_sha256
        or control.canary_entry_sha256s != selection.canary_entry_sha256s
        or control.canary_query_ids != selection.canary_query_ids
        or control.remaining_entry_sha256s != selection.remaining_entry_sha256s
        or control.remaining_query_ids != selection.remaining_query_ids
    )
    if control.policy_version == FEEDBACK_CONTROL_POLICY_VERSION:
        identity_drift = (
            type(authorization) is not PortfolioS1QwenFeedbackAuthorizationV3
            or control.provider != "qwen"
            or control.model != "qwen3.7-plus-2026-05-26"
            or control.processor != "dashscope-qwen37-feedback"
            or control.transport_policy_version
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5
            or control.transport_policy_sha256
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5
            or control.requested_response_format != "json_schema"
            or control.requested_json_schema_sha256
            != VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
            or control.requested_thinking is not True
            or control.requested_thinking_budget != 2048
            or control.requested_timeout_seconds != 600
            or control.requested_temperature is not None
            or control.requested_top_p is not None
            or authorization.requested_response_format != "json_schema"
            or authorization.requested_json_schema_sha256
            != VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
            or authorization.requested_thinking is not True
            or authorization.requested_thinking_budget != 2048
            or authorization.requested_timeout_seconds != 600
            or authorization.requested_temperature is not None
            or authorization.requested_top_p is not None
            or authorization.requested_max_tokens != control.max_tokens
            or authorization.max_completion_tokens != control.max_completion_tokens
            or authorization.max_attempts != control.max_attempts
        )
    elif control.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V7:
        identity_drift = (
            type(authorization) is not PortfolioS1FeedbackAuthorizationV2
            or control.provider != "kimi"
            or control.model != "kimi-k2.6"
            or control.processor != "dashscope-kimi-feedback"
            or control.transport_policy_version
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3
            or control.transport_policy_sha256
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3
            or control.requested_response_format is not None
            or control.requested_thinking is not False
            or control.requested_temperature != 0.6
            or control.requested_top_p != 0.95
        )
    elif control.policy_version == FEEDBACK_CONTROL_POLICY_VERSION_V5:
        identity_drift = (
            type(authorization) is not PortfolioS1FeedbackAuthorizationV1
            or control.provider != "gemini"
            or control.model != "gemini-3.6-flash"
            or control.processor != "aifast-gemini-feedback"
            or control.transport_policy_version
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1
            or control.transport_policy_sha256
            != VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1
            or control.requested_response_format != "json_object"
        )
    else:
        identity_drift = True
    expected_prompt_identity = (
        (
            VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
            VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
        )
        if control.policy_version
        in {
            FEEDBACK_CONTROL_POLICY_VERSION_V7,
            FEEDBACK_CONTROL_POLICY_VERSION,
        }
        else (
            VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
            VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
        )
    )
    if (
        common_drift
        or identity_drift
        or control.parser_policy_version != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or control.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or (
            control.prompt_policy_version,
            control.prompt_policy_sha256,
        )
        != expected_prompt_identity
    ):
        raise PortfolioS1FeedbackError("Feedback control binding drifted")


def build_portfolio_s1_feedback_control(
    selection: PortfolioS1FeedbackSelectionV1,
    authorization: PortfolioS1FeedbackAuthorization,
    *,
    rubric: RubricSnapshot,
    feedback_concurrency: Literal[1, 2] = 2,
) -> PortfolioS1FeedbackControlV1:
    if type(authorization) is PortfolioS1QwenFeedbackAuthorizationV3:
        identity = {
            "policy_version": FEEDBACK_CONTROL_POLICY_VERSION,
            "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
            "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
            "transport_policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
            "transport_policy_sha256": VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
            "requested_response_format": "json_schema",
            "requested_json_schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            "requested_thinking": True,
            "requested_thinking_budget": 2048,
            "requested_timeout_seconds": 600,
            "requested_temperature": None,
            "requested_top_p": None,
            "provider": "qwen",
            "model": "qwen3.7-plus-2026-05-26",
            "processor": "dashscope-qwen37-feedback",
            "max_tokens": None,
            "max_completion_tokens": 4096,
        }
    elif type(authorization) is PortfolioS1FeedbackAuthorizationV2:
        # Preserve the frozen Kimi control-v7 constructor for historical
        # canonical fixtures and terminal artifact replay.  New CLI runs never
        # select this branch.
        identity = {
            "policy_version": FEEDBACK_CONTROL_POLICY_VERSION_V7,
            "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
            "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
            "transport_policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
            "transport_policy_sha256": VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
            "requested_response_format": None,
            "requested_json_schema_sha256": None,
            "requested_thinking": False,
            "requested_thinking_budget": None,
            "requested_timeout_seconds": None,
            "requested_temperature": 0.6,
            "requested_top_p": 0.95,
            "provider": "kimi",
            "model": "kimi-k2.6",
            "processor": "dashscope-kimi-feedback",
            "max_tokens": 2048,
            "max_completion_tokens": None,
        }
    else:
        raise PortfolioS1FeedbackError(
            "active Qwen3.7 Feedback control requires authorization v3"
        )
    validate_selected_feedback_authorization(authorization, selection)
    draft = PortfolioS1FeedbackControlV1.model_construct(
        policy_version=identity["policy_version"],
        parser_policy_version=VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        parser_policy_sha256=VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        prompt_policy_version=identity["prompt_policy_version"],
        prompt_policy_sha256=identity["prompt_policy_sha256"],
        transport_policy_version=identity["transport_policy_version"],
        transport_policy_sha256=identity["transport_policy_sha256"],
        requested_response_format=identity["requested_response_format"],
        requested_json_schema_sha256=identity["requested_json_schema_sha256"],
        requested_thinking=identity["requested_thinking"],
        requested_thinking_budget=identity["requested_thinking_budget"],
        requested_timeout_seconds=identity["requested_timeout_seconds"],
        requested_temperature=identity["requested_temperature"],
        requested_top_p=identity["requested_top_p"],
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_id=authorization.authorization_id,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        corpus_sha256=selection.corpus_sha256,
        parent_static_bank_sha256=selection.parent_static_bank_sha256,
        canary_entry_sha256s=selection.canary_entry_sha256s,
        canary_query_ids=selection.canary_query_ids,
        remaining_entry_sha256s=selection.remaining_entry_sha256s,
        remaining_query_ids=selection.remaining_query_ids,
        rubric=rubric,
        provider=identity["provider"],
        model=identity["model"],
        processor=identity["processor"],
        max_tokens=identity["max_tokens"],
        max_completion_tokens=identity["max_completion_tokens"],
        feedback_concurrency=feedback_concurrency,
        control_sha256="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"control_sha256"})
    if identity["policy_version"] != FEEDBACK_CONTROL_POLICY_VERSION:
        payload.pop("requested_json_schema_sha256", None)
        payload.pop("requested_thinking_budget", None)
        payload.pop("requested_timeout_seconds", None)
        payload.pop("max_completion_tokens", None)
    return PortfolioS1FeedbackControlV1.model_validate_json(
        canonical_json_bytes({**payload, "control_sha256": _hash_payload(payload)}),
        strict=True,
    )


@dataclass(frozen=True)
class VerifiedStaticFeedbackSource:
    """One selected row reverified against Static GCS and rebuilt locally."""

    corpus_sha256: str
    selection_sha256: str
    control_sha256: str
    selection_entry_sha256: str
    corpus: VerifiedStaticGCSCorpus = field(repr=False, compare=False)
    row: VerifiedStaticGCSRow = field(repr=False, compare=False)
    packet: FeedbackPacket
    _marker: object = field(repr=False, compare=False)


def _entry_matches_verified_row(
    entry: FeedbackSelectionEntryV1,
    row: VerifiedStaticGCSRow,
) -> bool:
    return (
        row.query.query_id == entry.query_id
        and row.query.canonical_capability == entry.capability
        and row.query.asset_id == entry.asset_id
        and row.query.leakage_group_id == entry.leakage_group_id
        and row.atomic_component_id == entry.atomic_component_id
        and row.image_sha256 == entry.image_sha256
        and int(row.score.gcs) == entry.gcs
        and _pass_count(row) == entry.gcs_component_pass_count
        and tuple(row.score.reason_codes) == entry.reason_codes
        and row.checkpoint_file_sha256 == entry.checkpoint_file_sha256
        and row.checkpoint_row_sha256 == entry.checkpoint_row_sha256
        and row.sidecar.evidence_sha256 == entry.sidecar_sha256
        and row.strata.source_dataset == entry.strata.source_dataset
        and row.strata.repair_status == entry.strata.repair_status
        and row.strata.boundary_status == entry.strata.boundary_status
        and row.strata.style_submode == entry.strata.style_submode
    )


def build_verified_static_feedback_source(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
) -> VerifiedStaticFeedbackSource:
    """Reverify one selected row and deterministically rebuild FeedbackPacket v2."""

    verified = require_verified_static_gcs_corpus(corpus)
    return _build_verified_static_feedback_source(verified, selection, control, entry)


def _build_verified_static_feedback_source(
    verified: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
) -> VerifiedStaticFeedbackSource:
    selected_by_hash = {item.entry_sha256: item for item in selection.entries}
    row = verified.row_by_query_id().get(entry.query_id)
    if (
        selected_by_hash.get(entry.entry_sha256) != entry
        or selection.corpus_sha256 != verified.corpus_sha256
        or control.selection_sha256 != selection.selection_sha256
        or control.corpus_sha256 != verified.corpus_sha256
        or control.parent_static_bank_sha256 != verified.runtime.bank.bank_sha256
        or row is None
        or not _entry_matches_verified_row(entry, row)
    ):
        raise PortfolioS1FeedbackError(
            "selected Feedback entry differs from the verified Static GCS row"
        )
    packet = build_feedback_packet(
        row.query,
        row.result,
        asset_catalog=verified.core_inputs.runtime_asset_catalog(),
        rubric=control.rubric,
    )
    if (
        packet.query_id != entry.query_id
        or packet.canonical_capability != entry.capability
        or packet.image.sha256 != entry.image_sha256
    ):
        raise PortfolioS1FeedbackError(
            "rebuilt Feedback packet differs from the selected Static source"
        )
    return VerifiedStaticFeedbackSource(
        corpus_sha256=verified.corpus_sha256,
        selection_sha256=selection.selection_sha256,
        control_sha256=control.control_sha256,
        selection_entry_sha256=entry.entry_sha256,
        corpus=verified,
        row=row,
        packet=packet,
        _marker=_VERIFIED_STATIC_FEEDBACK_SOURCE_MARKER,
    )


def build_verified_static_feedback_sources(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
) -> tuple[VerifiedStaticFeedbackSource, ...]:
    """Deep-verify opt800 once, then rebuild all 48 selected packets."""

    verified = require_verified_static_gcs_corpus(corpus)
    sources = tuple(
        _build_verified_static_feedback_source(verified, selection, control, entry)
        for entry in selection.entries
    )
    if len(sources) != FEEDBACK_SELECTION_SIZE:
        raise PortfolioS1FeedbackError(
            "verified Feedback source set must contain exactly 48 rows"
        )
    return sources


def require_verified_static_feedback_source(
    source: VerifiedStaticFeedbackSource,
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
) -> VerifiedStaticFeedbackSource:
    if (
        type(source) is not VerifiedStaticFeedbackSource
        or source._marker is not _VERIFIED_STATIC_FEEDBACK_SOURCE_MARKER
        or source.corpus_sha256 != selection.corpus_sha256
        or source.selection_sha256 != selection.selection_sha256
        or source.control_sha256 != control.control_sha256
        or source.selection_entry_sha256 != entry.entry_sha256
        or not _entry_matches_verified_row(entry, source.row)
    ):
        raise PortfolioS1FeedbackError("verified Static Feedback source drifted")
    return source


class FeedbackCallReservationV1(_StrictFrozenModel):
    """Create-only proof that the one allowed provider attempt was consumed."""

    schema_version: Literal[1, 2, 3] = 1
    kind: Literal["portfolio-s1-feedback-call-reservation"] = (
        "portfolio-s1-feedback-call-reservation"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-call-reservation-v1",
        "portfolio-s1-feedback-call-reservation-v2",
        "portfolio-s1-feedback-call-reservation-v3",
    ] = FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V1
    selection_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=48)
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    provider: Literal["gemini", "kimi", "qwen"] = "gemini"
    model: Literal[
        "gemini-3.6-flash",
        "kimi-k2.6",
        "qwen3.7-plus-2026-05-26",
    ] = "gemini-3.6-flash"
    max_tokens: Literal[2048] | None = 2048
    max_completion_tokens: Literal[4096] | None = None
    max_attempts: Literal[1] = 1
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        expected = {
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V1: (
                1,
                "gemini",
                "gemini-3.6-flash",
            ),
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V2: (
                2,
                "kimi",
                "kimi-k2.6",
            ),
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION: (
                3,
                "qwen",
                "qwen3.7-plus-2026-05-26",
            ),
        }[self.policy_version]
        if (self.schema_version, self.provider, self.model) != expected:
            raise ValueError("Feedback call reservation provider identity mismatch")
        if self.policy_version == FEEDBACK_CALL_RESERVATION_POLICY_VERSION:
            token_identity_mismatch = (
                self.max_tokens is not None or self.max_completion_tokens != 4096
            )
        else:
            token_identity_mismatch = (
                self.max_tokens != 2048 or self.max_completion_tokens is not None
            )
        if token_identity_mismatch:
            raise ValueError(
                "Feedback call reservation completion-token identity mismatch"
            )
        if self.reservation_sha256 != _hash_payload(
            self._canonical_payload(exclude_hash=True)
        ):
            raise ValueError("Feedback call reservation self hash mismatch")
        return self

    def _canonical_payload(self, *, exclude_hash: bool) -> dict[str, object]:
        excluded = {"reservation_sha256"} if exclude_hash else set()
        payload = self.model_dump(mode="json", exclude=excluded)
        if self.policy_version != FEEDBACK_CALL_RESERVATION_POLICY_VERSION:
            payload.pop("max_completion_tokens", None)
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._canonical_payload(exclude_hash=False))


def build_feedback_call_reservation(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
    *,
    verified_source: VerifiedStaticFeedbackSource,
) -> FeedbackCallReservationV1:
    require_verified_static_feedback_source(verified_source, selection, control, entry)
    row = verified_source.row
    if control.policy_version == FEEDBACK_CONTROL_POLICY_VERSION:
        reservation_identity = (
            3,
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION,
        )
    elif control.policy_version in {
        FEEDBACK_CONTROL_POLICY_VERSION_V6,
        FEEDBACK_CONTROL_POLICY_VERSION_V7,
    }:
        reservation_identity = (
            2,
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V2,
        )
    else:
        reservation_identity = (
            1,
            FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V1,
        )
    unsigned = {
        "schema_version": reservation_identity[0],
        "kind": "portfolio-s1-feedback-call-reservation",
        "policy_version": reservation_identity[1],
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "corpus_sha256": verified_source.corpus_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "selection_ordinal": entry.selection_ordinal,
        "query_id": entry.query_id,
        "packet_sha256": verified_source.packet.packet_sha256,
        "checkpoint_file_sha256": row.checkpoint_file_sha256,
        "checkpoint_row_sha256": row.checkpoint_row_sha256,
        "sidecar_sha256": row.sidecar.evidence_sha256,
        "provider": control.provider,
        "model": control.model,
        "max_tokens": control.max_tokens,
        "max_completion_tokens": control.max_completion_tokens,
        "max_attempts": control.max_attempts,
    }
    draft = FeedbackCallReservationV1.model_construct(
        **unsigned,
        reservation_sha256="0" * 64,
    )
    payload = draft._canonical_payload(exclude_hash=True)
    return FeedbackCallReservationV1.model_validate_json(
        canonical_json_bytes({**payload, "reservation_sha256": _hash_payload(payload)}),
        strict=True,
    )


def write_feedback_call_reservation(
    path: str | Path, reservation: FeedbackCallReservationV1
) -> Path:
    return atomic_create_file(path, reservation.canonical_bytes())


def load_feedback_call_reservation(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_control_sha256: str,
    expected_entry_sha256: str,
) -> FeedbackCallReservationV1:
    content = read_stable_regular_file(
        path, label="Feedback call reservation", max_bytes=2 * 1024 * 1024
    )
    reservation = FeedbackCallReservationV1.model_validate_json(content, strict=True)
    if reservation.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "Feedback call reservation is not canonical JSON"
        )
    if (
        reservation.selection_sha256 != expected_selection_sha256
        or reservation.control_sha256 != expected_control_sha256
        or reservation.selection_entry_sha256 != expected_entry_sha256
    ):
        raise PortfolioS1FeedbackError(
            "Feedback call reservation resume identity conflict"
        )
    return reservation


class BoundFeedbackArtifact(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-bound-feedback"] = "portfolio-s1-bound-feedback"
    policy_version: Literal["portfolio-s1-bound-feedback-v1"] = (
        BOUND_FEEDBACK_POLICY_VERSION
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    reservation_sha256: Sha256
    query_id: str
    feedback_packet: FeedbackPacket
    feedback_result: FeedbackEvaluationResult
    status: FeedbackStatus
    provider_call_count: Literal[1] = 1
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet_from_json(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacket)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result_from_json(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackEvaluationResult)

    @model_validator(mode="after")
    def _validate_bound_artifact(self) -> Self:
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.status != self.status
        ):
            raise ValueError("bound Feedback packet/result identity mismatch")
        if self.artifact_sha256 != _hash_payload(
            self._canonical_payload(exclude_hash=True)
        ):
            raise ValueError("bound Feedback artifact self hash mismatch")
        return self

    def _canonical_payload(self, *, exclude_hash: bool) -> dict[str, object]:
        excluded = {"artifact_sha256"} if exclude_hash else set()
        payload = self.model_dump(mode="json", exclude=excluded)
        if self.feedback_result.cache_namespace == "feedback-evaluator-v2":
            nested = payload["feedback_result"]
            assert isinstance(nested, dict)
            nested.pop("parser_policy_version", None)
            nested.pop("parser_policy_sha256", None)
        if self.feedback_result.cache_namespace not in {
            "feedback-evaluator-v5",
            "feedback-evaluator-v6",
            "feedback-evaluator-v7",
            "feedback-evaluator-v8",
            "feedback-evaluator-v9",
        }:
            nested = payload["feedback_result"]
            assert isinstance(nested, dict)
            nested.pop("prompt_policy_version", None)
            nested.pop("prompt_policy_sha256", None)
        if self.feedback_result.cache_namespace not in {
            "feedback-evaluator-v6",
            "feedback-evaluator-v7",
            "feedback-evaluator-v8",
            "feedback-evaluator-v9",
        }:
            nested = payload["feedback_result"]
            assert isinstance(nested, dict)
            nested.pop("transport_policy_version", None)
            nested.pop("transport_policy_sha256", None)
            nested.pop("requested_response_format", None)
        if self.feedback_result.cache_namespace not in {
            "feedback-evaluator-v7",
            "feedback-evaluator-v8",
            "feedback-evaluator-v9",
        }:
            nested = payload["feedback_result"]
            assert isinstance(nested, dict)
            nested.pop("requested_thinking", None)
            nested.pop("requested_temperature", None)
            nested.pop("requested_top_p", None)
        if self.feedback_result.cache_namespace != "feedback-evaluator-v9":
            nested = payload["feedback_result"]
            assert isinstance(nested, dict)
            nested.pop("max_completion_tokens", None)
            nested.pop("requested_thinking_budget", None)
            nested.pop("requested_timeout_seconds", None)
            nested.pop("reasoning_present", None)
            nested.pop("reasoning_tokens", None)
            nested.pop("reasoning_bytes", None)
            nested.pop("reasoning_sha256", None)
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._canonical_payload(exclude_hash=False))


def build_bound_feedback_artifact(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
    packet: FeedbackPacket,
    result: FeedbackEvaluationResult,
    *,
    verified_source: VerifiedStaticFeedbackSource,
    reservation: FeedbackCallReservationV1,
) -> BoundFeedbackArtifact:
    """Bind one exact packet/result pair to its pre-registered selection row."""

    selected_by_hash = {item.entry_sha256: item for item in selection.entries}
    active_qwen = control.policy_version == FEEDBACK_CONTROL_POLICY_VERSION
    expected_result_cache = (
        "feedback-evaluator-v9" if active_qwen else "feedback-evaluator-v8"
    )
    expected_transport_identity = (
        (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
            VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
        )
        if active_qwen
        else (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
            VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
        )
    )
    try:
        require_verified_static_feedback_source(
            verified_source, selection, control, entry
        )
    except PortfolioS1FeedbackError as error:
        raise PortfolioS1FeedbackError(
            "bound Feedback source identity drifted"
        ) from error
    if (
        verified_source.packet != packet
        or reservation
        != build_feedback_call_reservation(
            selection,
            control,
            entry,
            verified_source=verified_source,
        )
        or selected_by_hash.get(entry.entry_sha256) != entry
        or control.selection_sha256 != selection.selection_sha256
        or control.corpus_sha256 != selection.corpus_sha256
        or packet.query_id != entry.query_id
        or packet.canonical_capability != entry.capability
        or packet.image.sha256 != entry.image_sha256
        or packet.rubric != control.rubric
        or result.query_id != entry.query_id
        or result.cache_namespace != expected_result_cache
        or result.parser_policy_version != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or result.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or result.prompt_policy_version != VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5
        or result.prompt_policy_sha256 != VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5
        or (
            result.transport_policy_version,
            result.transport_policy_sha256,
        )
        != expected_transport_identity
        or result.requested_response_format != control.requested_response_format
        or result.requested_json_schema_sha256 != control.requested_json_schema_sha256
        or result.requested_thinking != control.requested_thinking
        or result.requested_thinking_budget != control.requested_thinking_budget
        or result.requested_timeout_seconds != control.requested_timeout_seconds
        or result.requested_temperature != control.requested_temperature
        or result.requested_top_p != control.requested_top_p
        or result.packet_sha256 != packet.packet_sha256
        or result.image_sha256 != entry.image_sha256
        or result.remote_authorization_id != control.authorization_id
        or result.remote_authorization_file_sha256 != control.authorization_file_sha256
        or result.provider != control.provider
        or result.model != control.model
        or result.max_tokens != control.max_tokens
        or result.max_completion_tokens != control.max_completion_tokens
        or result.max_attempts != control.max_attempts
    ):
        raise PortfolioS1FeedbackError("bound Feedback source identity drifted")
    draft = BoundFeedbackArtifact.model_construct(
        selection_sha256=selection.selection_sha256,
        control_sha256=control.control_sha256,
        selection_entry_sha256=entry.entry_sha256,
        reservation_sha256=reservation.reservation_sha256,
        query_id=entry.query_id,
        feedback_packet=packet,
        feedback_result=result,
        status=result.status,
        artifact_sha256="0" * 64,
    )
    payload = draft._canonical_payload(exclude_hash=True)
    return BoundFeedbackArtifact.model_validate_json(
        canonical_json_bytes({**payload, "artifact_sha256": _hash_payload(payload)}),
        strict=True,
    )


def write_bound_feedback_artifact(
    path: str | Path,
    artifact: BoundFeedbackArtifact,
) -> Path:
    return atomic_create_file(path, artifact.canonical_bytes())


def load_bound_feedback_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_selection_sha256: str | None = None,
    expected_control_sha256: str | None = None,
    expected_entry_sha256: str | None = None,
) -> BoundFeedbackArtifact:
    content = read_stable_regular_file(
        path, label="bound S1 Feedback artifact", max_bytes=4 * 1024 * 1024
    )
    if expected_file_sha256 is not None and sha256_bytes(content) != (
        expected_file_sha256
    ):
        raise PortfolioS1FeedbackError("bound Feedback file SHA-256 mismatch")
    artifact = BoundFeedbackArtifact.model_validate_json(content, strict=True)
    if artifact.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("bound Feedback artifact is not canonical")
    expected = (
        (expected_selection_sha256, artifact.selection_sha256),
        (expected_control_sha256, artifact.control_sha256),
        (expected_entry_sha256, artifact.selection_entry_sha256),
    )
    if any(wanted is not None and wanted != actual for wanted, actual in expected):
        raise PortfolioS1FeedbackError("bound Feedback resume identity conflict")
    return artifact


class PortfolioS1FeedbackRunEntryV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=48)
    selection_entry_sha256: Sha256
    artifact_sha256: Sha256
    status: FeedbackStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class PortfolioS1FeedbackRunV1(_StrictFrozenModel):
    """One immutable terminal summary for the selected48 provider run."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-run"] = "portfolio-s1-feedback-run"
    policy_version: Literal["portfolio-s1-feedback-run-v1"] = (
        S1_FEEDBACK_RUN_POLICY_VERSION
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    expected_count: Literal[48] = 48
    completed_count: int = Field(ge=1, le=48)
    parsed_count: int = Field(ge=0, le=48)
    error_count: int = Field(ge=0, le=48)
    canary_completed_count: int = Field(ge=0, le=6)
    remaining_completed_count: int = Field(ge=0, le=42)
    terminal_phase: Literal["canary", "remaining", "completed"]
    status: Literal["stopped_nonparsed", "completed"]
    provider_calls: int = Field(ge=1, le=48)
    usage_known_count: int = Field(ge=0, le=48)
    usage_unknown_count: int = Field(ge=0, le=48)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    artifacts: tuple[PortfolioS1FeedbackRunEntryV1, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator("artifacts", mode="before")
    @classmethod
    def _artifacts_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        if (
            len(self.artifacts) != self.completed_count
            or tuple(item.selection_ordinal for item in self.artifacts)
            != tuple(sorted(item.selection_ordinal for item in self.artifacts))
            or len({item.selection_entry_sha256 for item in self.artifacts})
            != self.completed_count
            or self.parsed_count
            != sum(item.status == "parsed" for item in self.artifacts)
            or self.error_count != self.completed_count - self.parsed_count
            or self.provider_calls != self.completed_count
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count != self.completed_count - self.usage_known_count
            or self.input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
            or self.canary_completed_count + self.remaining_completed_count
            != self.completed_count
        ):
            raise ValueError("S1 Feedback run counts drifted")
        payload = [item.model_dump(mode="json") for item in self.artifacts]
        if self.artifact_set_sha256 != _hash_payload(payload):
            raise ValueError("S1 Feedback run artifact set hash mismatch")
        if self.status == "completed":
            if (
                self.completed_count != 48
                or self.parsed_count != 48
                or self.error_count != 0
                or self.canary_completed_count != 6
                or self.remaining_completed_count != 42
                or self.terminal_phase != "completed"
            ):
                raise ValueError("completed S1 Feedback run is not parsed48")
        elif self.error_count < 1 or self.terminal_phase == "completed":
            raise ValueError("stopped S1 Feedback run lacks a terminal error")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("S1 Feedback run self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_run(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    artifacts: tuple[BoundFeedbackArtifact, ...],
) -> PortfolioS1FeedbackRunV1:
    artifact_by_entry = {item.selection_entry_sha256: item for item in artifacts}
    if not artifacts or len(artifact_by_entry) != len(artifacts):
        raise PortfolioS1FeedbackError("Feedback run artifacts are empty or repeat")
    rows: list[PortfolioS1FeedbackRunEntryV1] = []
    for entry in selection.entries:
        artifact = artifact_by_entry.get(entry.entry_sha256)
        if artifact is None:
            continue
        if (
            artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != entry.query_id
        ):
            raise PortfolioS1FeedbackError("Feedback run artifact binding drifted")
        rows.append(
            PortfolioS1FeedbackRunEntryV1(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                artifact_sha256=artifact.artifact_sha256,
                status=artifact.status,
                input_tokens=(
                    None
                    if artifact.feedback_result.usage is None
                    else artifact.feedback_result.usage.input_tokens
                ),
                output_tokens=(
                    None
                    if artifact.feedback_result.usage is None
                    else artifact.feedback_result.usage.output_tokens
                ),
            )
        )
    errors = [item for item in rows if item.status != "parsed"]
    if not errors and len(rows) != 48:
        raise PortfolioS1FeedbackError(
            "nonterminal parsed Feedback artifacts cannot publish a run summary"
        )
    canary_hashes = set(selection.canary_entry_sha256s)
    canary_count = sum(item.selection_entry_sha256 in canary_hashes for item in rows)
    remaining_count = len(rows) - canary_count
    terminal_phase: Literal["canary", "remaining", "completed"] = (
        "completed"
        if len(rows) == 48 and not errors
        else (
            "canary"
            if any(item.selection_entry_sha256 in canary_hashes for item in errors)
            else "remaining"
        )
    )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-run",
        "policy_version": S1_FEEDBACK_RUN_POLICY_VERSION,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 48,
        "completed_count": len(rows),
        "parsed_count": len(rows) - len(errors),
        "error_count": len(errors),
        "canary_completed_count": canary_count,
        "remaining_completed_count": remaining_count,
        "terminal_phase": terminal_phase,
        "status": "completed" if not errors else "stopped_nonparsed",
        "provider_calls": len(rows),
        "usage_known_count": sum(item.input_tokens is not None for item in rows),
        "usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRunV1.model_validate(
        {**unsigned, "run_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_portfolio_s1_feedback_run(
    path: str | Path, run: PortfolioS1FeedbackRunV1
) -> Path:
    return atomic_create_file(path, run.canonical_bytes())


def load_portfolio_s1_feedback_run(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackRunV1:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback run", max_bytes=4 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback run file SHA-256 mismatch")
    run = PortfolioS1FeedbackRunV1.model_validate_json(content, strict=True)
    if run.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback run is not canonical JSON")
    return run


def resume_bound_feedback_artifact(
    path: str | Path,
    *,
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    entry: FeedbackSelectionEntryV1,
    verified_source: VerifiedStaticFeedbackSource,
    reservation: FeedbackCallReservationV1,
) -> BoundFeedbackArtifact | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    artifact = load_bound_feedback_artifact(
        candidate,
        expected_selection_sha256=selection.selection_sha256,
        expected_control_sha256=control.control_sha256,
        expected_entry_sha256=entry.entry_sha256,
    )
    if artifact.reservation_sha256 != reservation.reservation_sha256:
        raise PortfolioS1FeedbackError("bound Feedback reservation conflict")
    expected = build_bound_feedback_artifact(
        selection,
        control,
        entry,
        verified_source.packet,
        artifact.feedback_result,
        verified_source=verified_source,
        reservation=reservation,
    )
    if artifact != expected:
        raise PortfolioS1FeedbackError("bound Feedback resume source conflict")
    return artifact


class PortfolioS1FeedbackBundleEntryV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=48)
    query_id: str
    capability: str
    role: FeedbackRole
    primary_cluster: str
    leakage_group_id: str
    gcs: Literal[0, 1]
    reason_codes: tuple[str, ...]
    bound_artifact_sha256: Sha256
    feedback_result_sha256: Sha256
    feedback: VisualFeedbackOutput

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _reasons_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("feedback", mode="before")
    @classmethod
    def _feedback_from_json(cls, value: object) -> object:
        return _nested_json_model(value, VisualFeedbackOutput)


class PortfolioS1FeedbackClusterSummaryV1(_StrictFrozenModel):
    cluster_id: str
    selected_count: int = Field(ge=1)
    selection_ordinals: tuple[int, ...]
    rule_violation_count: int = Field(ge=0)
    ideal_response_gap_count: int = Field(ge=0)
    skill_suggestion_count: int = Field(ge=0)

    @field_validator("selection_ordinals", mode="before")
    @classmethod
    def _query_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioS1FeedbackModelEntryV1(_StrictFrozenModel):
    """Allowlisted structured Feedback row; no execution identity survives."""

    selection_ordinal: int = Field(ge=1, le=48)
    capability: str
    role: FeedbackRole
    primary_cluster: str
    gcs: Literal[0, 1]
    reason_codes: tuple[str, ...]
    feedback: VisualFeedbackOutput

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _reason_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("feedback", mode="before")
    @classmethod
    def _feedback_from_json(cls, value: object) -> object:
        return _nested_json_model(value, VisualFeedbackOutput)


class PortfolioS1FeedbackRepresentativeExampleV1(_StrictFrozenModel):
    """One of 18 bounded public examples carrying transcript/evidence text."""

    selection_ordinal: int = Field(ge=1, le=48)
    capability: str
    role: FeedbackRole
    primary_cluster: str
    turns: tuple[ConversationTurn, ...]
    response_text: str
    cards: tuple[VisibleCard, ...]
    tool_evidence: tuple[VisibleToolEvidence, ...]

    @field_validator("turns", mode="before")
    @classmethod
    def _turns_from_json(cls, value: object) -> object:
        return _nested_json_models(value, ConversationTurn)

    @field_validator("cards", mode="before")
    @classmethod
    def _cards_from_json(cls, value: object) -> object:
        return _nested_json_models(value, VisibleCard)

    @field_validator("tool_evidence", mode="before")
    @classmethod
    def _evidence_from_json(cls, value: object) -> object:
        return _nested_json_models(value, VisibleToolEvidence)


class PortfolioS1FeedbackModelProjectionV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-model-projection"] = (
        "portfolio-s1-feedback-model-projection"
    )
    feedback_entries: tuple[PortfolioS1FeedbackModelEntryV1, ...]
    representative_examples: tuple[PortfolioS1FeedbackRepresentativeExampleV1, ...]
    cluster_summaries: tuple[PortfolioS1FeedbackClusterSummaryV1, ...]

    @field_validator(
        "feedback_entries",
        "representative_examples",
        "cluster_summaries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioS1FeedbackBundleV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-bundle"] = "portfolio-s1-feedback-bundle"
    policy_version: Literal["portfolio-s1-feedback-bundle-v1"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    run_sha256: Sha256
    authorization_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    selected_count: Literal[48] = 48
    parsed_count: Literal[48] = 48
    provider_call_count: Literal[48] = 48
    excluded_execution_lapse_query_ids: tuple[str, ...]
    bound_artifact_set_sha256: Sha256
    entries: tuple[PortfolioS1FeedbackBundleEntryV1, ...]
    model_projection: PortfolioS1FeedbackModelProjectionV1
    bundle_sha256: Sha256

    @field_validator("excluded_execution_lapse_query_ids", "entries", mode="before")
    @classmethod
    def _excluded_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        entries = self.entries
        if len(entries) != 48 or tuple(item.selection_ordinal for item in entries) != (
            tuple(range(1, 49))
        ):
            raise ValueError("S1 Feedback bundle must contain ordered parsed48")
        if len({item.query_id for item in entries}) != 48:
            raise ValueError("S1 Feedback bundle query identities must be unique")
        projection_ordinals = tuple(
            item.selection_ordinal for item in self.model_projection.feedback_entries
        )
        if projection_ordinals != tuple(range(1, 49)):
            raise ValueError("S1 Feedback model projection must cover ordered48")
        for private, public in zip(
            entries, self.model_projection.feedback_entries, strict=True
        ):
            if (
                public.capability != private.capability
                or public.role != private.role
                or public.primary_cluster != private.primary_cluster
                or public.gcs != private.gcs
                or public.reason_codes != private.reason_codes
                or public.feedback != private.feedback
            ):
                raise ValueError("S1 Feedback model projection differs from bundle")
        canary_ordinals = tuple(
            next(
                item.selection_ordinal
                for item in entries
                if item.capability == capability and item.role == "failure"
            )
            for capability in GCS_CAPABILITY_ORDER
        )
        anchor_ordinals = tuple(
            item.selection_ordinal for item in entries if item.role != "failure"
        )
        examples = self.model_projection.representative_examples
        if (
            len(examples) != 18
            or tuple(item.selection_ordinal for item in examples)
            != canary_ordinals + anchor_ordinals
        ):
            raise ValueError(
                "S1 Feedback projection requires canary6 plus all 12 anchors"
            )
        private_by_ordinal = {item.selection_ordinal: item for item in entries}
        for example in examples:
            private = private_by_ordinal[example.selection_ordinal]
            if (
                example.capability != private.capability
                or example.role != private.role
                or example.primary_cluster != private.primary_cluster
            ):
                raise ValueError("representative example metadata drifted")
        try:
            _validate_model_projection_privacy(
                self.model_projection,
                private_query_ids=tuple(item.query_id for item in entries),
            )
        except PortfolioS1FeedbackError as error:
            raise ValueError(str(error)) from error
        if self.bundle_sha256 != _model_hash(self, "bundle_sha256"):
            raise ValueError("S1 Feedback bundle self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_projection_payload(self) -> dict[str, object]:
        """Return only the allowlisted content visible to the S1 Creator."""

        _validate_model_projection_privacy(
            self.model_projection,
            private_query_ids=tuple(item.query_id for item in self.entries),
        )
        return self.model_projection.model_dump(mode="json")


def _cluster_summaries(
    entries: tuple[PortfolioS1FeedbackBundleEntryV1, ...],
) -> tuple[PortfolioS1FeedbackClusterSummaryV1, ...]:
    grouped: dict[str, list[PortfolioS1FeedbackBundleEntryV1]] = defaultdict(list)
    for entry in entries:
        grouped[entry.primary_cluster].append(entry)
    summaries = []
    for cluster_id in sorted(grouped):
        members = sorted(grouped[cluster_id], key=lambda item: item.query_id)
        summaries.append(
            PortfolioS1FeedbackClusterSummaryV1(
                cluster_id=cluster_id,
                selected_count=len(members),
                selection_ordinals=tuple(
                    sorted(item.selection_ordinal for item in members)
                ),
                rule_violation_count=sum(
                    len(item.feedback.rule_violations) for item in members
                ),
                ideal_response_gap_count=sum(
                    len(item.feedback.ideal_response_gaps) for item in members
                ),
                skill_suggestion_count=sum(
                    len(item.feedback.skill_suggestions) for item in members
                ),
            )
        )
    return tuple(summaries)


def build_portfolio_s1_feedback_bundle(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    authorization: PortfolioS1FeedbackAuthorization,
    artifacts: tuple[BoundFeedbackArtifact, ...],
    run: PortfolioS1FeedbackRunV1,
) -> PortfolioS1FeedbackBundleV1:
    """Build the Creator handoff only when all 48 Feedback calls parsed."""

    validate_portfolio_s1_feedback_control(control, selection, authorization)
    if len(artifacts) != 48:
        raise PortfolioS1FeedbackError("S1 Feedback bundle requires 48 artifacts")
    expected_run = build_portfolio_s1_feedback_run(selection, control, artifacts)
    if run != expected_run or run.status != "completed" or run.parsed_count != 48:
        raise PortfolioS1FeedbackError(
            "S1 Feedback bundle requires the exact completed parsed48 run"
        )
    artifact_by_entry = {item.selection_entry_sha256: item for item in artifacts}
    if len(artifact_by_entry) != 48:
        raise PortfolioS1FeedbackError("bound Feedback artifact identities repeat")
    entries: list[PortfolioS1FeedbackBundleEntryV1] = []
    artifact_hashes: list[str] = []
    for selected in selection.entries:
        artifact = artifact_by_entry.get(selected.entry_sha256)
        if artifact is None:
            raise PortfolioS1FeedbackError("selection lacks a bound Feedback artifact")
        if (
            artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != selected.query_id
            or artifact.status != "parsed"
            or artifact.feedback_result.parsed_feedback is None
        ):
            raise PortfolioS1FeedbackError(
                "S1 bundle requires 48 parsed, identity-bound Feedback results"
            )
        entries.append(
            PortfolioS1FeedbackBundleEntryV1(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                leakage_group_id=selected.leakage_group_id,
                gcs=selected.gcs,
                reason_codes=selected.reason_codes,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
                feedback=artifact.feedback_result.parsed_feedback,
            )
        )
        artifact_hashes.append(artifact.artifact_sha256)
    model_entries = tuple(
        PortfolioS1FeedbackModelEntryV1(
            selection_ordinal=entry.selection_ordinal,
            capability=entry.capability,
            role=entry.role,
            primary_cluster=entry.primary_cluster,
            gcs=entry.gcs,
            reason_codes=entry.reason_codes,
            feedback=entry.feedback,
        )
        for entry in entries
    )
    representative_ordinals = tuple(
        selection.entries.index(
            next(
                item for item in selection.entries if item.entry_sha256 == entry_sha256
            )
        )
        + 1
        for entry_sha256 in selection.canary_entry_sha256s
    ) + tuple(entry.selection_ordinal for entry in entries if entry.role != "failure")
    representative_examples = tuple(
        PortfolioS1FeedbackRepresentativeExampleV1(
            selection_ordinal=entry.selection_ordinal,
            capability=entry.capability,
            role=entry.role,
            primary_cluster=entry.primary_cluster,
            turns=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.turns,
            response_text=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.response_text,
            cards=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.cards,
            tool_evidence=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.tool_evidence,
        )
        for entry in entries
        if entry.selection_ordinal in representative_ordinals
    )
    representative_by_ordinal = {
        item.selection_ordinal: item for item in representative_examples
    }
    model_projection = PortfolioS1FeedbackModelProjectionV1(
        feedback_entries=model_entries,
        representative_examples=tuple(
            representative_by_ordinal[item] for item in representative_ordinals
        ),
        cluster_summaries=_cluster_summaries(tuple(entries)),
    )
    _validate_model_projection_privacy(
        model_projection,
        private_query_ids=tuple(item.query_id for item in entries),
    )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "run_sha256": run.run_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 48,
        "parsed_count": 48,
        "provider_call_count": 48,
        "excluded_execution_lapse_query_ids": (
            selection.excluded_execution_lapse_query_ids
        ),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "entries": tuple(entries),
        "model_projection": model_projection,
    }
    return PortfolioS1FeedbackBundleV1.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_portfolio_s1_feedback_bundle(
    path: str | Path,
    bundle: PortfolioS1FeedbackBundleV1,
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundleV1:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("S1 Feedback bundle file SHA-256 mismatch")
    bundle = PortfolioS1FeedbackBundleV1.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("S1 Feedback bundle is not canonical JSON")
    return bundle


_FEEDBACK_ROLE_ORDER: tuple[FeedbackRole, ...] = (
    "failure",
    "success_anchor",
    "partial_anchor",
)


class PortfolioS1FeedbackCoverageCellV2(_StrictFrozenModel):
    """One explicit capability/selection-role coverage cell."""

    capability: str
    role: FeedbackRole
    selected_count: int = Field(ge=0, le=8)
    attempted_count: int = Field(ge=0, le=8)
    parsed_count: int = Field(ge=0, le=8)
    parse_error_count: int = Field(ge=0, le=8)
    unattempted_count: int = Field(ge=0, le=8)

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.capability not in GCS_CAPABILITY_ORDER:
            raise ValueError("Feedback coverage capability is unknown")
        if (
            self.attempted_count != self.parsed_count + self.parse_error_count
            or self.selected_count != self.attempted_count + self.unattempted_count
        ):
            raise ValueError("Feedback coverage cell counts drifted")
        return self


class PortfolioS1FeedbackTerminalArtifactRefV2(_StrictFrozenModel):
    """Private binding for every v5 attempt, including the parse error."""

    selection_ordinal: int = Field(ge=1, le=48)
    selection_entry_sha256: Sha256
    capability: str
    role: FeedbackRole
    status: Literal["parsed", "parse_error"]
    artifact_file_sha256: Sha256
    bound_artifact_sha256: Sha256
    feedback_result_sha256: Sha256


class PortfolioS1FeedbackFullGCSBindingV2(_StrictFrozenModel):
    """Private identity binding for the deeply re-scored Static opt800 corpus."""

    corpus_policy_version: Literal["portfolio-static-opt800-gcs-corpus-v1"] = (
        STATIC_GCS_CORPUS_POLICY_VERSION
    )
    corpus_sha256: Sha256
    execution_control_file_sha256: Sha256
    execution_control_sha256: Sha256
    launch_plan_sha256: Sha256
    runtime_lock_sha256: Sha256
    population_mapping_sha256: Sha256
    checkpoint_set_sha256: Sha256
    sidecar_set_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    query_count: Literal[800] = 800
    gcs_reason_strata_sha256: Sha256


class PortfolioS1FeedbackGCSCapabilitySummaryV2(_StrictFrozenModel):
    capability: str
    query_count: int = Field(ge=1, le=800)
    gcs_pass_count: int = Field(ge=0, le=800)
    gcs_failure_count: int = Field(ge=0, le=800)
    hard_error_count: int = Field(ge=0, le=800)

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.capability not in GCS_CAPABILITY_ORDER:
            raise ValueError("full GCS summary capability is unknown")
        if self.query_count != self.gcs_pass_count + self.gcs_failure_count:
            raise ValueError("full GCS capability counts drifted")
        return self


class PortfolioS1FeedbackGCSReasonSummaryV2(_StrictFrozenModel):
    capability: str
    reason_code: str
    count: int = Field(ge=1, le=800)

    @field_validator("reason_code")
    @classmethod
    def _reason(cls, value: str) -> str:
        return _nonblank(value, "reason_code")


class PortfolioS1FeedbackGCSStrataSummaryV2(_StrictFrozenModel):
    capability: str
    source_dataset: str
    repair_status: str
    boundary_status: str
    style_submode: str | None = None
    query_count: int = Field(ge=1, le=800)
    gcs_pass_count: int = Field(ge=0, le=800)
    gcs_failure_count: int = Field(ge=0, le=800)

    @field_validator("source_dataset", "repair_status", "boundary_status")
    @classmethod
    def _text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.query_count != self.gcs_pass_count + self.gcs_failure_count:
            raise ValueError("full GCS strata counts drifted")
        return self


class PortfolioS1FeedbackFullGCSSummaryV2(_StrictFrozenModel):
    """Allowlisted aggregate diagnostics from all 800 verified Static rows."""

    query_count: Literal[800] = 800
    gcs_pass_count: int = Field(ge=0, le=800)
    gcs_failure_count: int = Field(ge=0, le=800)
    hard_error_count: int = Field(ge=0, le=800)
    capability_summaries: tuple[PortfolioS1FeedbackGCSCapabilitySummaryV2, ...]
    reason_summaries: tuple[PortfolioS1FeedbackGCSReasonSummaryV2, ...]
    strata_summaries: tuple[PortfolioS1FeedbackGCSStrataSummaryV2, ...]

    @field_validator(
        "capability_summaries", "reason_summaries", "strata_summaries", mode="before"
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        if self.query_count != self.gcs_pass_count + self.gcs_failure_count:
            raise ValueError("full GCS headline counts drifted")
        if tuple(item.capability for item in self.capability_summaries) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("full GCS capability summary order drifted")
        if (
            sum(item.query_count for item in self.capability_summaries) != 800
            or sum(item.gcs_pass_count for item in self.capability_summaries)
            != self.gcs_pass_count
            or sum(item.gcs_failure_count for item in self.capability_summaries)
            != self.gcs_failure_count
            or sum(item.hard_error_count for item in self.capability_summaries)
            != self.hard_error_count
        ):
            raise ValueError("full GCS aggregate differs from capability summaries")
        reason_keys = tuple(
            (item.capability, item.reason_code) for item in self.reason_summaries
        )
        if reason_keys != tuple(sorted(set(reason_keys))):
            raise ValueError("full GCS reason summaries must be sorted and unique")
        strata_keys = tuple(
            (
                item.capability,
                item.source_dataset,
                item.repair_status,
                item.boundary_status,
                item.style_submode or "",
            )
            for item in self.strata_summaries
        )
        if strata_keys != tuple(sorted(set(strata_keys))):
            raise ValueError("full GCS strata summaries must be sorted and unique")
        if sum(item.query_count for item in self.strata_summaries) != 800:
            raise ValueError("full GCS strata summary denominator drifted")
        return self


class PortfolioS1FeedbackModelProjectionV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-model-projection"] = (
        "portfolio-s1-feedback-model-projection"
    )
    status: Literal["incomplete_diagnostic_feedback"] = "incomplete_diagnostic_feedback"
    selected_count: Literal[48] = 48
    attempted_count: Literal[14] = 14
    parsed_count: Literal[13] = 13
    parse_error_count: Literal[1] = 1
    unattempted_count: Literal[34] = 34
    missing_feedback_count: Literal[35] = 35
    parse_error_ordinals: tuple[int, ...]
    unattempted_ordinals: tuple[int, ...]
    missing_feedback_ordinals: tuple[int, ...]
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    feedback_entries: tuple[PortfolioS1FeedbackModelEntryV1, ...]
    representative_examples: tuple[PortfolioS1FeedbackRepresentativeExampleV1, ...]
    cluster_summaries: tuple[PortfolioS1FeedbackClusterSummaryV1, ...]
    full_gcs_summary: PortfolioS1FeedbackFullGCSSummaryV2

    @field_validator(
        "parse_error_ordinals",
        "unattempted_ordinals",
        "missing_feedback_ordinals",
        "coverage",
        "feedback_entries",
        "representative_examples",
        "cluster_summaries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioS1FeedbackBundleV2(_StrictFrozenModel):
    """Approved partial13 diagnostic handoff from the one terminal v5 run."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-bundle"] = "portfolio-s1-feedback-bundle"
    policy_version: Literal["portfolio-s1-feedback-bundle-v2"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V2
    )
    status: Literal["incomplete_diagnostic_feedback"] = "incomplete_diagnostic_feedback"
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    run_sha256: Sha256
    run_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    full_gcs_binding: PortfolioS1FeedbackFullGCSBindingV2
    selected_count: Literal[48] = 48
    attempted_count: Literal[14] = 14
    parsed_count: Literal[13] = 13
    parse_error_count: Literal[1] = 1
    provider_call_count: Literal[14] = 14
    unattempted_count: Literal[34] = 34
    missing_feedback_count: Literal[35] = 35
    parse_error_ordinals: tuple[int, ...]
    unattempted_ordinals: tuple[int, ...]
    missing_feedback_ordinals: tuple[int, ...]
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    excluded_execution_lapse_query_ids: tuple[str, ...]
    selected_query_ids: tuple[str, ...]
    terminal_artifacts: tuple[PortfolioS1FeedbackTerminalArtifactRefV2, ...]
    bound_artifact_set_sha256: Sha256
    entries: tuple[PortfolioS1FeedbackBundleEntryV1, ...]
    model_projection: PortfolioS1FeedbackModelProjectionV2
    bundle_sha256: Sha256

    @field_validator(
        "parse_error_ordinals",
        "unattempted_ordinals",
        "missing_feedback_ordinals",
        "coverage",
        "excluded_execution_lapse_query_ids",
        "selected_query_ids",
        "terminal_artifacts",
        "entries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        attempted_ordinals = tuple(
            item.selection_ordinal for item in self.terminal_artifacts
        )
        parsed_ordinals = tuple(
            item.selection_ordinal
            for item in self.terminal_artifacts
            if item.status == "parsed"
        )
        parse_error_ordinals = tuple(
            item.selection_ordinal
            for item in self.terminal_artifacts
            if item.status == "parse_error"
        )
        if (
            len(self.terminal_artifacts) != 14
            or attempted_ordinals != tuple(sorted(set(attempted_ordinals)))
            or len(parsed_ordinals) != 13
            or len(parse_error_ordinals) != 1
            or self.parse_error_ordinals != parse_error_ordinals
            or self.unattempted_ordinals
            != tuple(item for item in range(1, 49) if item not in attempted_ordinals)
            or self.missing_feedback_ordinals
            != tuple(sorted(parse_error_ordinals + self.unattempted_ordinals))
            or tuple(item.selection_ordinal for item in self.entries) != parsed_ordinals
        ):
            raise ValueError("partial S1 Feedback ordinal partition drifted")
        if (
            len(self.selected_query_ids) != 48
            or len(set(self.selected_query_ids)) != 48
        ):
            raise ValueError("partial S1 Feedback selected query identities drifted")
        terminal_by_ordinal = {
            item.selection_ordinal: item for item in self.terminal_artifacts
        }
        for entry in self.entries:
            terminal = terminal_by_ordinal[entry.selection_ordinal]
            if (
                terminal.capability != entry.capability
                or terminal.role != entry.role
                or terminal.bound_artifact_sha256 != entry.bound_artifact_sha256
                or terminal.feedback_result_sha256 != entry.feedback_result_sha256
            ):
                raise ValueError("partial S1 Feedback entry/artifact binding drifted")
        expected_coverage_order = tuple(
            (capability, role)
            for capability in GCS_CAPABILITY_ORDER
            for role in _FEEDBACK_ROLE_ORDER
        )
        if tuple((item.capability, item.role) for item in self.coverage) != (
            expected_coverage_order
        ):
            raise ValueError("partial S1 Feedback coverage order drifted")
        if (
            sum(item.selected_count for item in self.coverage) != 48
            or sum(item.attempted_count for item in self.coverage) != 14
            or sum(item.parsed_count for item in self.coverage) != 13
            or sum(item.parse_error_count for item in self.coverage) != 1
            or sum(item.unattempted_count for item in self.coverage) != 34
        ):
            raise ValueError("partial S1 Feedback coverage totals drifted")
        if self.bound_artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.terminal_artifacts]
        ):
            raise ValueError("partial S1 Feedback artifact-set hash drifted")
        projection = self.model_projection
        if (
            projection.parse_error_ordinals != self.parse_error_ordinals
            or projection.unattempted_ordinals != self.unattempted_ordinals
            or projection.missing_feedback_ordinals != self.missing_feedback_ordinals
            or projection.coverage != self.coverage
            or tuple(item.selection_ordinal for item in projection.feedback_entries)
            != parsed_ordinals
            or tuple(
                item.selection_ordinal for item in projection.representative_examples
            )
            != parsed_ordinals
        ):
            raise ValueError("partial S1 Feedback model projection drifted")
        for private, public in zip(
            self.entries, projection.feedback_entries, strict=True
        ):
            if (
                public.capability != private.capability
                or public.role != private.role
                or public.primary_cluster != private.primary_cluster
                or public.gcs != private.gcs
                or public.reason_codes != private.reason_codes
                or public.feedback != private.feedback
            ):
                raise ValueError("partial Feedback projection differs from bundle")
        if (
            self.full_gcs_binding.corpus_sha256 != self.corpus_sha256
            or self.full_gcs_binding.parent_static_bank_sha256
            != self.parent_static_bank_sha256
            or self.full_gcs_binding.gcs_policy_sha256 != self.gcs_policy_sha256
        ):
            raise ValueError("partial Feedback full-corpus binding drifted")
        try:
            _validate_model_projection_privacy(
                projection,
                private_query_ids=self.selected_query_ids,
            )
        except PortfolioS1FeedbackError as error:
            raise ValueError(str(error)) from error
        if self.bundle_sha256 != _model_hash(self, "bundle_sha256"):
            raise ValueError("S1 Feedback bundle v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_projection_payload(self) -> dict[str, object]:
        """Return only the privacy-scanned partial diagnostics visible to Creator."""

        _validate_model_projection_privacy(
            self.model_projection,
            private_query_ids=self.selected_query_ids,
        )
        return self.model_projection.model_dump(mode="json")


def _feedback_v2_coverage(
    selection: PortfolioS1FeedbackSelectionV1,
    terminal_by_entry: dict[str, BoundFeedbackArtifact],
) -> tuple[PortfolioS1FeedbackCoverageCellV2, ...]:
    cells = []
    for capability in GCS_CAPABILITY_ORDER:
        for role in _FEEDBACK_ROLE_ORDER:
            selected = tuple(
                item
                for item in selection.entries
                if item.capability == capability and item.role == role
            )
            attempted = tuple(
                terminal_by_entry[item.entry_sha256]
                for item in selected
                if item.entry_sha256 in terminal_by_entry
            )
            cells.append(
                PortfolioS1FeedbackCoverageCellV2(
                    capability=capability,
                    role=role,
                    selected_count=len(selected),
                    attempted_count=len(attempted),
                    parsed_count=sum(item.status == "parsed" for item in attempted),
                    parse_error_count=sum(
                        item.status == "parse_error" for item in attempted
                    ),
                    unattempted_count=len(selected) - len(attempted),
                )
            )
    return tuple(cells)


def _full_gcs_v2_diagnostics(
    corpus: VerifiedStaticGCSCorpus,
) -> tuple[PortfolioS1FeedbackFullGCSBindingV2, PortfolioS1FeedbackFullGCSSummaryV2]:
    rows = tuple(sorted(corpus.rows, key=lambda item: item.query_ordinal))
    if (
        len(rows) != 800
        or len({row.query.query_id for row in rows}) != 800
        or {row.query.canonical_capability for row in rows} != set(GCS_CAPABILITY_ORDER)
    ):
        raise PortfolioS1FeedbackError(
            "partial Feedback requires the full opt800 GCS corpus"
        )
    identity_rows = []
    for row in rows:
        identity_rows.append(
            {
                "query_ordinal": row.query_ordinal,
                "query_id": row.query.query_id,
                "capability": row.query.canonical_capability,
                "gcs": int(row.score.gcs),
                "hard_error": int(row.score.hard_error),
                "route_acceptable": int(row.score.route_acceptable),
                "no_hard_error": int(row.score.no_hard_error),
                "tool_contract_pass": int(row.score.tool_contract_pass),
                "evidence_grounded": int(row.score.evidence_grounded),
                "output_contract_pass": int(row.score.output_contract_pass),
                "reason_codes": tuple(row.score.reason_codes),
                "source_dataset": row.strata.source_dataset,
                "repair_status": row.strata.repair_status,
                "boundary_status": row.strata.boundary_status,
                "style_submode": row.strata.style_submode,
            }
        )
    capability_summaries = []
    for capability in GCS_CAPABILITY_ORDER:
        members = tuple(
            row for row in rows if row.query.canonical_capability == capability
        )
        capability_summaries.append(
            PortfolioS1FeedbackGCSCapabilitySummaryV2(
                capability=capability,
                query_count=len(members),
                gcs_pass_count=sum(int(row.score.gcs) for row in members),
                gcs_failure_count=sum(not row.score.gcs for row in members),
                hard_error_count=sum(int(row.score.hard_error) for row in members),
            )
        )
    reason_counter: Counter[tuple[str, str]] = Counter()
    strata_counter: Counter[tuple[str, str, str, str, str | None, int]] = Counter()
    for row in rows:
        capability = row.query.canonical_capability
        for reason in row.score.reason_codes:
            reason_counter[(capability, reason)] += 1
        strata_counter[
            (
                capability,
                row.strata.source_dataset,
                row.strata.repair_status,
                row.strata.boundary_status,
                row.strata.style_submode,
                int(row.score.gcs),
            )
        ] += 1
    reason_summaries = tuple(
        PortfolioS1FeedbackGCSReasonSummaryV2(
            capability=capability, reason_code=reason, count=count
        )
        for (capability, reason), count in sorted(reason_counter.items())
    )
    strata_keys = sorted(
        {
            (capability, source, repair, boundary, style)
            for capability, source, repair, boundary, style, _gcs in strata_counter
        },
        key=lambda item: (*item[:-1], item[-1] or ""),
    )
    strata_summaries = tuple(
        PortfolioS1FeedbackGCSStrataSummaryV2(
            capability=capability,
            source_dataset=source,
            repair_status=repair,
            boundary_status=boundary,
            style_submode=style,
            query_count=sum(
                strata_counter[(capability, source, repair, boundary, style, gcs)]
                for gcs in (0, 1)
            ),
            gcs_pass_count=strata_counter[
                (capability, source, repair, boundary, style, 1)
            ],
            gcs_failure_count=strata_counter[
                (capability, source, repair, boundary, style, 0)
            ],
        )
        for capability, source, repair, boundary, style in strata_keys
    )
    summary = PortfolioS1FeedbackFullGCSSummaryV2(
        gcs_pass_count=sum(int(row.score.gcs) for row in rows),
        gcs_failure_count=sum(not row.score.gcs for row in rows),
        hard_error_count=sum(int(row.score.hard_error) for row in rows),
        capability_summaries=tuple(capability_summaries),
        reason_summaries=reason_summaries,
        strata_summaries=strata_summaries,
    )
    try:
        launch_sha256 = corpus.launch.plan.launch_plan_sha256
        runtime_lock_sha256 = corpus.runtime.runtime_lock["runtime_lock_sha256"]
        population_sha256 = corpus.population.population_mapping_sha256
        parent_bank_sha256 = corpus.runtime.bank.bank_sha256
    except (AttributeError, KeyError, TypeError) as error:
        raise PortfolioS1FeedbackError(
            "verified opt800 corpus is missing frozen identity fields"
        ) from error
    binding = PortfolioS1FeedbackFullGCSBindingV2(
        corpus_sha256=corpus.corpus_sha256,
        execution_control_file_sha256=corpus.control_file_sha256,
        execution_control_sha256=corpus.control_sha256,
        launch_plan_sha256=launch_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        population_mapping_sha256=population_sha256,
        checkpoint_set_sha256=corpus.checkpoint_set_sha256,
        sidecar_set_sha256=corpus.sidecar_set_sha256,
        parent_static_bank_sha256=parent_bank_sha256,
        gcs_reason_strata_sha256=_hash_payload(identity_rows),
    )
    return binding, summary


def build_portfolio_s1_feedback_bundle_v2(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    authorization: PortfolioS1FeedbackAuthorizationV1,
    artifacts: tuple[BoundFeedbackArtifact, ...],
    run: PortfolioS1FeedbackRunV1,
    *,
    corpus: VerifiedStaticGCSCorpus,
) -> PortfolioS1FeedbackBundleV2:
    """Build the approved partial13 handoff from exactly one terminal v5 run."""

    validate_portfolio_s1_feedback_control(control, selection, authorization)
    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    if (
        selection.corpus_sha256 != verified.corpus_sha256
        or selection.parent_static_bank_sha256 != verified.runtime.bank.bank_sha256
    ):
        raise PortfolioS1FeedbackError(
            "partial Feedback selection/corpus binding drifted"
        )
    artifact_by_entry = {item.selection_entry_sha256: item for item in artifacts}
    if len(artifacts) != 14 or len(artifact_by_entry) != 14:
        raise PortfolioS1FeedbackError(
            "partial Feedback bundle requires exact attempted14"
        )
    expected_run = build_portfolio_s1_feedback_run(selection, control, artifacts)
    if (
        run != expected_run
        or run.status != "stopped_nonparsed"
        or run.terminal_phase != "remaining"
        or run.completed_count != 14
        or run.provider_calls != 14
        or run.parsed_count != 13
        or run.error_count != 1
        or tuple(item.status for item in run.artifacts).count("parse_error") != 1
        or any(
            item.feedback_result.cache_namespace != "feedback-evaluator-v6"
            for item in artifacts
        )
    ):
        raise PortfolioS1FeedbackError(
            "partial Feedback bundle requires the exact terminal v5 14/13/1 run"
        )
    selected_by_hash = {item.entry_sha256: item for item in selection.entries}
    terminal_artifacts = []
    parsed_entries = []
    for run_entry in run.artifacts:
        selected = selected_by_hash.get(run_entry.selection_entry_sha256)
        artifact = artifact_by_entry.get(run_entry.selection_entry_sha256)
        if (
            selected is None
            or artifact is None
            or artifact.artifact_sha256 != run_entry.artifact_sha256
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != selected.query_id
            or artifact.status not in {"parsed", "parse_error"}
        ):
            raise PortfolioS1FeedbackError("partial Feedback terminal artifact drifted")
        terminal_artifacts.append(
            PortfolioS1FeedbackTerminalArtifactRefV2(
                selection_ordinal=selected.selection_ordinal,
                selection_entry_sha256=selected.entry_sha256,
                capability=selected.capability,
                role=selected.role,
                status=artifact.status,
                artifact_file_sha256=sha256_bytes(artifact.canonical_bytes()),
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
            )
        )
        if artifact.status == "parsed":
            if artifact.feedback_result.parsed_feedback is None:
                raise PortfolioS1FeedbackError(
                    "parsed v5 artifact lacks typed Feedback"
                )
            parsed_entries.append(
                PortfolioS1FeedbackBundleEntryV1(
                    selection_ordinal=selected.selection_ordinal,
                    query_id=selected.query_id,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    leakage_group_id=selected.leakage_group_id,
                    gcs=selected.gcs,
                    reason_codes=selected.reason_codes,
                    bound_artifact_sha256=artifact.artifact_sha256,
                    feedback_result_sha256=artifact.feedback_result.result_sha256,
                    feedback=artifact.feedback_result.parsed_feedback,
                )
            )
    terminal_tuple = tuple(terminal_artifacts)
    entries = tuple(parsed_entries)
    attempted_ordinals = tuple(item.selection_ordinal for item in terminal_tuple)
    parse_error_ordinals = tuple(
        item.selection_ordinal
        for item in terminal_tuple
        if item.status == "parse_error"
    )
    unattempted_ordinals = tuple(
        item for item in range(1, 49) if item not in attempted_ordinals
    )
    missing_ordinals = tuple(sorted(parse_error_ordinals + unattempted_ordinals))
    coverage = _feedback_v2_coverage(selection, artifact_by_entry)
    full_gcs_binding, full_gcs_summary = _full_gcs_v2_diagnostics(verified)
    model_entries = tuple(
        PortfolioS1FeedbackModelEntryV1(
            selection_ordinal=item.selection_ordinal,
            capability=item.capability,
            role=item.role,
            primary_cluster=item.primary_cluster,
            gcs=item.gcs,
            reason_codes=item.reason_codes,
            feedback=item.feedback,
        )
        for item in entries
    )
    representative_examples = tuple(
        PortfolioS1FeedbackRepresentativeExampleV1(
            selection_ordinal=entry.selection_ordinal,
            capability=entry.capability,
            role=entry.role,
            primary_cluster=entry.primary_cluster,
            turns=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.turns,
            response_text=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.response_text,
            cards=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.cards,
            tool_evidence=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.tool_evidence,
        )
        for entry in entries
    )
    projection = PortfolioS1FeedbackModelProjectionV2(
        parse_error_ordinals=parse_error_ordinals,
        unattempted_ordinals=unattempted_ordinals,
        missing_feedback_ordinals=missing_ordinals,
        coverage=coverage,
        feedback_entries=model_entries,
        representative_examples=representative_examples,
        cluster_summaries=_cluster_summaries(entries),
        full_gcs_summary=full_gcs_summary,
    )
    _validate_model_projection_privacy(
        projection,
        private_query_ids=tuple(item.query_id for item in selection.entries),
    )
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V2,
        "status": "incomplete_diagnostic_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": verified.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "full_gcs_binding": full_gcs_binding,
        "selected_count": 48,
        "attempted_count": 14,
        "parsed_count": 13,
        "parse_error_count": 1,
        "provider_call_count": 14,
        "unattempted_count": 34,
        "missing_feedback_count": 35,
        "parse_error_ordinals": parse_error_ordinals,
        "unattempted_ordinals": unattempted_ordinals,
        "missing_feedback_ordinals": missing_ordinals,
        "coverage": coverage,
        "excluded_execution_lapse_query_ids": (
            selection.excluded_execution_lapse_query_ids
        ),
        "selected_query_ids": tuple(item.query_id for item in selection.entries),
        "terminal_artifacts": terminal_tuple,
        "bound_artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in terminal_tuple]
        ),
        "entries": entries,
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV2.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_portfolio_s1_feedback_bundle_v2(
    path: str | Path,
    bundle: PortfolioS1FeedbackBundleV2,
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v2(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundleV2:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle v2", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v2 file SHA-256 mismatch")
    bundle = PortfolioS1FeedbackBundleV2.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v2 is not canonical JSON")
    return bundle


class PortfolioS1FeedbackModelProjectionV3(_StrictFrozenModel):
    """Creator-visible diagnostics from the one terminal Kimi v3 run."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-model-projection"] = (
        "portfolio-s1-feedback-model-projection"
    )
    status: Literal["incomplete_diagnostic_feedback"] = "incomplete_diagnostic_feedback"
    selected_count: Literal[48] = 48
    attempted_count: Literal[12] = 12
    parsed_count: Literal[11] = 11
    parse_error_count: Literal[1] = 1
    unattempted_count: Literal[36] = 36
    missing_feedback_count: Literal[37] = 37
    parse_error_ordinals: tuple[int, ...]
    unattempted_ordinals: tuple[int, ...]
    missing_feedback_ordinals: tuple[int, ...]
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    feedback_entries: tuple[PortfolioS1FeedbackModelEntryV1, ...]
    representative_examples: tuple[PortfolioS1FeedbackRepresentativeExampleV1, ...]
    cluster_summaries: tuple[PortfolioS1FeedbackClusterSummaryV1, ...]
    full_gcs_summary: PortfolioS1FeedbackFullGCSSummaryV2

    @field_validator(
        "parse_error_ordinals",
        "unattempted_ordinals",
        "missing_feedback_ordinals",
        "coverage",
        "feedback_entries",
        "representative_examples",
        "cluster_summaries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioS1FeedbackBundleV3(_StrictFrozenModel):
    """Independent partial11 handoff from the terminal Kimi v3 12/11/1 run."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-bundle"] = "portfolio-s1-feedback-bundle"
    policy_version: Literal["portfolio-s1-feedback-bundle-v3"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V3
    )
    status: Literal["incomplete_diagnostic_feedback"] = "incomplete_diagnostic_feedback"
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    run_sha256: Sha256
    run_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    full_gcs_binding: PortfolioS1FeedbackFullGCSBindingV2
    selected_count: Literal[48] = 48
    attempted_count: Literal[12] = 12
    parsed_count: Literal[11] = 11
    parse_error_count: Literal[1] = 1
    provider_call_count: Literal[12] = 12
    unattempted_count: Literal[36] = 36
    missing_feedback_count: Literal[37] = 37
    parse_error_ordinals: tuple[int, ...]
    unattempted_ordinals: tuple[int, ...]
    missing_feedback_ordinals: tuple[int, ...]
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    excluded_execution_lapse_query_ids: tuple[str, ...]
    selected_query_ids: tuple[str, ...]
    terminal_artifacts: tuple[PortfolioS1FeedbackTerminalArtifactRefV2, ...]
    bound_artifact_set_sha256: Sha256
    entries: tuple[PortfolioS1FeedbackBundleEntryV1, ...]
    model_projection: PortfolioS1FeedbackModelProjectionV3
    bundle_sha256: Sha256

    @field_validator(
        "parse_error_ordinals",
        "unattempted_ordinals",
        "missing_feedback_ordinals",
        "coverage",
        "excluded_execution_lapse_query_ids",
        "selected_query_ids",
        "terminal_artifacts",
        "entries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        attempted_ordinals = tuple(
            item.selection_ordinal for item in self.terminal_artifacts
        )
        parsed_ordinals = tuple(
            item.selection_ordinal
            for item in self.terminal_artifacts
            if item.status == "parsed"
        )
        parse_error_ordinals = tuple(
            item.selection_ordinal
            for item in self.terminal_artifacts
            if item.status == "parse_error"
        )
        if (
            len(self.terminal_artifacts) != 12
            or attempted_ordinals != tuple(sorted(set(attempted_ordinals)))
            or len(parsed_ordinals) != 11
            or len(parse_error_ordinals) != 1
            or self.parse_error_ordinals != parse_error_ordinals
            or self.unattempted_ordinals
            != tuple(item for item in range(1, 49) if item not in attempted_ordinals)
            or self.missing_feedback_ordinals
            != tuple(sorted(parse_error_ordinals + self.unattempted_ordinals))
            or tuple(item.selection_ordinal for item in self.entries) != parsed_ordinals
        ):
            raise ValueError("Kimi v3 S1 Feedback ordinal partition drifted")
        if (
            len(self.selected_query_ids) != 48
            or len(set(self.selected_query_ids)) != 48
        ):
            raise ValueError("Kimi v3 selected query identities drifted")
        terminal_by_ordinal = {
            item.selection_ordinal: item for item in self.terminal_artifacts
        }
        for entry in self.entries:
            terminal = terminal_by_ordinal[entry.selection_ordinal]
            if (
                terminal.capability != entry.capability
                or terminal.role != entry.role
                or terminal.bound_artifact_sha256 != entry.bound_artifact_sha256
                or terminal.feedback_result_sha256 != entry.feedback_result_sha256
            ):
                raise ValueError("Kimi v3 Feedback entry/artifact binding drifted")
        expected_coverage_order = tuple(
            (capability, role)
            for capability in GCS_CAPABILITY_ORDER
            for role in _FEEDBACK_ROLE_ORDER
        )
        if tuple((item.capability, item.role) for item in self.coverage) != (
            expected_coverage_order
        ):
            raise ValueError("Kimi v3 Feedback coverage order drifted")
        if (
            sum(item.selected_count for item in self.coverage) != 48
            or sum(item.attempted_count for item in self.coverage) != 12
            or sum(item.parsed_count for item in self.coverage) != 11
            or sum(item.parse_error_count for item in self.coverage) != 1
            or sum(item.unattempted_count for item in self.coverage) != 36
        ):
            raise ValueError("Kimi v3 Feedback coverage totals drifted")
        if self.bound_artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.terminal_artifacts]
        ):
            raise ValueError("Kimi v3 Feedback artifact-set hash drifted")
        projection = self.model_projection
        if (
            projection.parse_error_ordinals != self.parse_error_ordinals
            or projection.unattempted_ordinals != self.unattempted_ordinals
            or projection.missing_feedback_ordinals != self.missing_feedback_ordinals
            or projection.coverage != self.coverage
            or tuple(item.selection_ordinal for item in projection.feedback_entries)
            != parsed_ordinals
            or tuple(
                item.selection_ordinal for item in projection.representative_examples
            )
            != parsed_ordinals
        ):
            raise ValueError("Kimi v3 Feedback model projection drifted")
        for private, public in zip(
            self.entries, projection.feedback_entries, strict=True
        ):
            if (
                public.capability != private.capability
                or public.role != private.role
                or public.primary_cluster != private.primary_cluster
                or public.gcs != private.gcs
                or public.reason_codes != private.reason_codes
                or public.feedback != private.feedback
            ):
                raise ValueError("Kimi v3 Feedback projection differs from bundle")
        if (
            self.full_gcs_binding.corpus_sha256 != self.corpus_sha256
            or self.full_gcs_binding.parent_static_bank_sha256
            != self.parent_static_bank_sha256
            or self.full_gcs_binding.gcs_policy_sha256 != self.gcs_policy_sha256
        ):
            raise ValueError("Kimi v3 Feedback full-corpus binding drifted")
        try:
            _validate_model_projection_privacy(
                projection,
                private_query_ids=self.selected_query_ids,
            )
        except PortfolioS1FeedbackError as error:
            raise ValueError(str(error)) from error
        if self.bundle_sha256 != _model_hash(self, "bundle_sha256"):
            raise ValueError("S1 Feedback bundle v3 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_projection_payload(self) -> dict[str, object]:
        """Return only the privacy-scanned Kimi partial diagnostics."""

        _validate_model_projection_privacy(
            self.model_projection,
            private_query_ids=self.selected_query_ids,
        )
        return self.model_projection.model_dump(mode="json")


def build_portfolio_s1_feedback_bundle_v3(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    authorization: PortfolioS1FeedbackAuthorizationV2,
    artifacts: tuple[BoundFeedbackArtifact, ...],
    run: PortfolioS1FeedbackRunV1,
    *,
    corpus: VerifiedStaticGCSCorpus,
) -> PortfolioS1FeedbackBundleV3:
    """Build only the exact terminal Kimi v3 12/11/1 diagnostic handoff."""

    if type(authorization) is not PortfolioS1FeedbackAuthorizationV2:
        raise PortfolioS1FeedbackError("bundle v3 requires Kimi authorization v2")
    validate_portfolio_s1_feedback_control(control, selection, authorization)
    if control.policy_version != FEEDBACK_CONTROL_POLICY_VERSION_V7:
        raise PortfolioS1FeedbackError("bundle v3 requires Feedback control v7")
    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    if (
        selection.corpus_sha256 != verified.corpus_sha256
        or selection.parent_static_bank_sha256 != verified.runtime.bank.bank_sha256
    ):
        raise PortfolioS1FeedbackError("Kimi v3 selection/corpus binding drifted")
    artifact_by_entry = {item.selection_entry_sha256: item for item in artifacts}
    if len(artifacts) != 12 or len(artifact_by_entry) != 12:
        raise PortfolioS1FeedbackError(
            "Kimi v3 Feedback bundle requires exact attempted12"
        )
    expected_run = build_portfolio_s1_feedback_run(selection, control, artifacts)
    if (
        run != expected_run
        or run.status != "stopped_nonparsed"
        or run.terminal_phase != "remaining"
        or run.completed_count != 12
        or run.provider_calls != 12
        or run.parsed_count != 11
        or run.error_count != 1
        or tuple(item.status for item in run.artifacts).count("parse_error") != 1
    ):
        raise PortfolioS1FeedbackError(
            "bundle v3 requires the exact terminal Kimi 12/11/1 run"
        )

    selected_by_hash = {item.entry_sha256: item for item in selection.entries}
    source_by_hash = {
        item.selection_entry_sha256: item
        for item in build_verified_static_feedback_sources(verified, selection, control)
    }
    terminal_artifacts = []
    parsed_entries = []
    for run_entry in run.artifacts:
        selected = selected_by_hash.get(run_entry.selection_entry_sha256)
        artifact = artifact_by_entry.get(run_entry.selection_entry_sha256)
        source = source_by_hash.get(run_entry.selection_entry_sha256)
        if selected is None or artifact is None or source is None:
            raise PortfolioS1FeedbackError("Kimi v3 terminal identity is unknown")
        reservation = build_feedback_call_reservation(
            selection, control, selected, verified_source=source
        )
        rebuilt = build_bound_feedback_artifact(
            selection,
            control,
            selected,
            source.packet,
            artifact.feedback_result,
            verified_source=source,
            reservation=reservation,
        )
        if (
            artifact != rebuilt
            or artifact.artifact_sha256 != run_entry.artifact_sha256
            or artifact.status not in {"parsed", "parse_error"}
            or artifact.feedback_result.cache_namespace != "feedback-evaluator-v8"
        ):
            raise PortfolioS1FeedbackError("Kimi v3 terminal bound artifact drifted")
        terminal_artifacts.append(
            PortfolioS1FeedbackTerminalArtifactRefV2(
                selection_ordinal=selected.selection_ordinal,
                selection_entry_sha256=selected.entry_sha256,
                capability=selected.capability,
                role=selected.role,
                status=artifact.status,
                artifact_file_sha256=sha256_bytes(artifact.canonical_bytes()),
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
            )
        )
        if artifact.status == "parsed":
            feedback = artifact.feedback_result.parsed_feedback
            if feedback is None:
                raise PortfolioS1FeedbackError(
                    "parsed Kimi v3 artifact lacks typed Feedback"
                )
            parsed_entries.append(
                PortfolioS1FeedbackBundleEntryV1(
                    selection_ordinal=selected.selection_ordinal,
                    query_id=selected.query_id,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    leakage_group_id=selected.leakage_group_id,
                    gcs=selected.gcs,
                    reason_codes=selected.reason_codes,
                    bound_artifact_sha256=artifact.artifact_sha256,
                    feedback_result_sha256=artifact.feedback_result.result_sha256,
                    feedback=feedback,
                )
            )

    terminal_tuple = tuple(terminal_artifacts)
    entries = tuple(parsed_entries)
    attempted_ordinals = tuple(item.selection_ordinal for item in terminal_tuple)
    parse_error_ordinals = tuple(
        item.selection_ordinal
        for item in terminal_tuple
        if item.status == "parse_error"
    )
    unattempted_ordinals = tuple(
        item for item in range(1, 49) if item not in attempted_ordinals
    )
    missing_ordinals = tuple(sorted(parse_error_ordinals + unattempted_ordinals))
    coverage = _feedback_v2_coverage(selection, artifact_by_entry)
    full_gcs_binding, full_gcs_summary = _full_gcs_v2_diagnostics(verified)
    model_entries = tuple(
        PortfolioS1FeedbackModelEntryV1(
            selection_ordinal=item.selection_ordinal,
            capability=item.capability,
            role=item.role,
            primary_cluster=item.primary_cluster,
            gcs=item.gcs,
            reason_codes=item.reason_codes,
            feedback=item.feedback,
        )
        for item in entries
    )
    representative_examples = tuple(
        PortfolioS1FeedbackRepresentativeExampleV1(
            selection_ordinal=entry.selection_ordinal,
            capability=entry.capability,
            role=entry.role,
            primary_cluster=entry.primary_cluster,
            turns=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.turns,
            response_text=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.response_text,
            cards=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.cards,
            tool_evidence=artifact_by_entry[
                selection.entries[entry.selection_ordinal - 1].entry_sha256
            ].feedback_packet.tool_evidence,
        )
        for entry in entries
    )
    projection = PortfolioS1FeedbackModelProjectionV3(
        parse_error_ordinals=parse_error_ordinals,
        unattempted_ordinals=unattempted_ordinals,
        missing_feedback_ordinals=missing_ordinals,
        coverage=coverage,
        feedback_entries=model_entries,
        representative_examples=representative_examples,
        cluster_summaries=_cluster_summaries(entries),
        full_gcs_summary=full_gcs_summary,
    )
    _validate_model_projection_privacy(
        projection,
        private_query_ids=tuple(item.query_id for item in selection.entries),
    )
    unsigned = {
        "schema_version": 3,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V3,
        "status": "incomplete_diagnostic_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": verified.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "full_gcs_binding": full_gcs_binding,
        "selected_count": 48,
        "attempted_count": 12,
        "parsed_count": 11,
        "parse_error_count": 1,
        "provider_call_count": 12,
        "unattempted_count": 36,
        "missing_feedback_count": 37,
        "parse_error_ordinals": parse_error_ordinals,
        "unattempted_ordinals": unattempted_ordinals,
        "missing_feedback_ordinals": missing_ordinals,
        "coverage": coverage,
        "excluded_execution_lapse_query_ids": (
            selection.excluded_execution_lapse_query_ids
        ),
        "selected_query_ids": tuple(item.query_id for item in selection.entries),
        "terminal_artifacts": terminal_tuple,
        "bound_artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in terminal_tuple]
        ),
        "entries": entries,
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV3.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_portfolio_s1_feedback_bundle_v3(
    path: str | Path,
    bundle: PortfolioS1FeedbackBundleV3,
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v3(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundleV3:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle v3", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v3 file SHA-256 mismatch")
    bundle = PortfolioS1FeedbackBundleV3.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v3 is not canonical JSON")
    return bundle


S1_FEEDBACK_BUNDLE_POLICY_VERSION_V4 = "portfolio-s1-feedback-bundle-v4"


class PortfolioS1FeedbackModelProjectionV4(_StrictFrozenModel):
    """Creator-visible diagnostics from one complete Qwen parsed48 run."""

    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-feedback-model-projection"] = (
        "portfolio-s1-feedback-model-projection"
    )
    status: Literal["complete_diagnostic_feedback"] = "complete_diagnostic_feedback"
    selected_count: Literal[48] = 48
    attempted_count: Literal[48] = 48
    parsed_count: Literal[48] = 48
    parse_error_count: Literal[0] = 0
    unattempted_count: Literal[0] = 0
    missing_feedback_count: Literal[0] = 0
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    feedback_entries: tuple[PortfolioS1FeedbackModelEntryV1, ...]
    representative_examples: tuple[PortfolioS1FeedbackRepresentativeExampleV1, ...]
    cluster_summaries: tuple[PortfolioS1FeedbackClusterSummaryV1, ...]
    full_gcs_summary: PortfolioS1FeedbackFullGCSSummaryV2

    @field_validator(
        "coverage",
        "feedback_entries",
        "representative_examples",
        "cluster_summaries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class PortfolioS1FeedbackBundleV4(_StrictFrozenModel):
    """Complete Qwen handoff from one immutable parsed48 Feedback run."""

    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-feedback-bundle"] = "portfolio-s1-feedback-bundle"
    policy_version: Literal["portfolio-s1-feedback-bundle-v4"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V4
    )
    status: Literal["complete_diagnostic_feedback"] = "complete_diagnostic_feedback"
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    run_sha256: Sha256
    run_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    full_gcs_binding: PortfolioS1FeedbackFullGCSBindingV2
    selected_count: Literal[48] = 48
    attempted_count: Literal[48] = 48
    parsed_count: Literal[48] = 48
    parse_error_count: Literal[0] = 0
    provider_call_count: Literal[48] = 48
    unattempted_count: Literal[0] = 0
    missing_feedback_count: Literal[0] = 0
    coverage: tuple[PortfolioS1FeedbackCoverageCellV2, ...]
    excluded_execution_lapse_query_ids: tuple[str, ...]
    selected_query_ids: tuple[str, ...]
    terminal_artifacts: tuple[PortfolioS1FeedbackTerminalArtifactRefV2, ...]
    bound_artifact_set_sha256: Sha256
    entries: tuple[PortfolioS1FeedbackBundleEntryV1, ...]
    model_projection: PortfolioS1FeedbackModelProjectionV4
    bundle_sha256: Sha256

    @field_validator(
        "coverage",
        "excluded_execution_lapse_query_ids",
        "selected_query_ids",
        "terminal_artifacts",
        "entries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        expected_ordinals = tuple(range(1, 49))
        if (
            tuple(item.selection_ordinal for item in self.terminal_artifacts)
            != expected_ordinals
            or any(item.status != "parsed" for item in self.terminal_artifacts)
            or tuple(item.selection_ordinal for item in self.entries)
            != expected_ordinals
            or len(self.selected_query_ids) != 48
            or len(set(self.selected_query_ids)) != 48
            or tuple(item.query_id for item in self.entries) != self.selected_query_ids
        ):
            raise ValueError("Qwen v4 Feedback parsed48 identities drifted")

        terminal_by_ordinal = {
            item.selection_ordinal: item for item in self.terminal_artifacts
        }
        for entry in self.entries:
            terminal = terminal_by_ordinal[entry.selection_ordinal]
            if (
                terminal.capability != entry.capability
                or terminal.role != entry.role
                or terminal.bound_artifact_sha256 != entry.bound_artifact_sha256
                or terminal.feedback_result_sha256 != entry.feedback_result_sha256
            ):
                raise ValueError("Qwen v4 Feedback entry/artifact binding drifted")

        expected_coverage_order = tuple(
            (capability, role)
            for capability in GCS_CAPABILITY_ORDER
            for role in _FEEDBACK_ROLE_ORDER
        )
        if tuple((item.capability, item.role) for item in self.coverage) != (
            expected_coverage_order
        ) or any(
            item.attempted_count != item.selected_count
            or item.parsed_count != item.selected_count
            or item.parse_error_count != 0
            or item.unattempted_count != 0
            for item in self.coverage
        ):
            raise ValueError("Qwen v4 Feedback coverage drifted")
        if sum(item.selected_count for item in self.coverage) != 48:
            raise ValueError("Qwen v4 Feedback coverage total drifted")
        if self.bound_artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.terminal_artifacts]
        ):
            raise ValueError("Qwen v4 Feedback artifact-set hash drifted")

        projection = self.model_projection
        if (
            projection.coverage != self.coverage
            or tuple(item.selection_ordinal for item in projection.feedback_entries)
            != expected_ordinals
        ):
            raise ValueError("Qwen v4 Feedback model projection drifted")
        for private, public in zip(
            self.entries, projection.feedback_entries, strict=True
        ):
            if (
                public.capability != private.capability
                or public.role != private.role
                or public.primary_cluster != private.primary_cluster
                or public.gcs != private.gcs
                or public.reason_codes != private.reason_codes
                or public.feedback != private.feedback
            ):
                raise ValueError("Qwen v4 Feedback projection differs from bundle")

        canary_ordinals = tuple(
            next(
                item.selection_ordinal
                for item in self.entries
                if item.capability == capability and item.role == "failure"
            )
            for capability in GCS_CAPABILITY_ORDER
        )
        anchor_ordinals = tuple(
            item.selection_ordinal for item in self.entries if item.role != "failure"
        )
        examples = projection.representative_examples
        if (
            len(examples) != 18
            or tuple(item.selection_ordinal for item in examples)
            != canary_ordinals + anchor_ordinals
        ):
            raise ValueError(
                "Qwen v4 Feedback projection requires canary6 plus all 12 anchors"
            )
        private_by_ordinal = {item.selection_ordinal: item for item in self.entries}
        for example in examples:
            private = private_by_ordinal[example.selection_ordinal]
            if (
                example.capability != private.capability
                or example.role != private.role
                or example.primary_cluster != private.primary_cluster
            ):
                raise ValueError("Qwen v4 representative example metadata drifted")

        if (
            self.full_gcs_binding.corpus_sha256 != self.corpus_sha256
            or self.full_gcs_binding.parent_static_bank_sha256
            != self.parent_static_bank_sha256
            or self.full_gcs_binding.gcs_policy_sha256 != self.gcs_policy_sha256
        ):
            raise ValueError("Qwen v4 Feedback full-corpus binding drifted")
        try:
            _validate_model_projection_privacy(
                projection,
                private_query_ids=self.selected_query_ids,
            )
        except PortfolioS1FeedbackError as error:
            raise ValueError(str(error)) from error
        if self.bundle_sha256 != _model_hash(self, "bundle_sha256"):
            raise ValueError("S1 Feedback bundle v4 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_projection_payload(self) -> dict[str, object]:
        """Return only the privacy-scanned complete diagnostics for Creator."""

        _validate_model_projection_privacy(
            self.model_projection,
            private_query_ids=self.selected_query_ids,
        )
        return self.model_projection.model_dump(mode="json")


def build_portfolio_s1_feedback_bundle_v4(
    selection: PortfolioS1FeedbackSelectionV1,
    control: PortfolioS1FeedbackControlV1,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    artifacts: tuple[BoundFeedbackArtifact, ...],
    run: PortfolioS1FeedbackRunV1,
    *,
    corpus: VerifiedStaticGCSCorpus,
) -> PortfolioS1FeedbackBundleV4:
    """Build only a complete Qwen v8/v9 parsed48 Creator handoff."""

    if type(authorization) is not PortfolioS1QwenFeedbackAuthorizationV3:
        raise PortfolioS1FeedbackError("bundle v4 requires Qwen authorization v3")
    validate_portfolio_s1_feedback_control(control, selection, authorization)
    if control.policy_version != FEEDBACK_CONTROL_POLICY_VERSION:
        raise PortfolioS1FeedbackError("bundle v4 requires Feedback control v8")
    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    if (
        selection.corpus_sha256 != verified.corpus_sha256
        or selection.parent_static_bank_sha256 != verified.runtime.bank.bank_sha256
    ):
        raise PortfolioS1FeedbackError("Qwen v4 selection/corpus binding drifted")

    artifact_by_entry = {item.selection_entry_sha256: item for item in artifacts}
    if len(artifacts) != 48 or len(artifact_by_entry) != 48:
        raise PortfolioS1FeedbackError("Qwen v4 Feedback bundle requires exact 48")
    expected_run = build_portfolio_s1_feedback_run(selection, control, artifacts)
    if (
        run != expected_run
        or run.status != "completed"
        or run.terminal_phase != "completed"
        or run.completed_count != 48
        or run.provider_calls != 48
        or run.parsed_count != 48
        or run.error_count != 0
    ):
        raise PortfolioS1FeedbackError(
            "bundle v4 requires the exact completed Qwen parsed48 run"
        )

    selected_by_hash = {item.entry_sha256: item for item in selection.entries}
    source_by_hash = {
        item.selection_entry_sha256: item
        for item in build_verified_static_feedback_sources(verified, selection, control)
    }
    terminal_artifacts = []
    entries = []
    for run_entry in run.artifacts:
        selected = selected_by_hash.get(run_entry.selection_entry_sha256)
        artifact = artifact_by_entry.get(run_entry.selection_entry_sha256)
        source = source_by_hash.get(run_entry.selection_entry_sha256)
        if selected is None or artifact is None or source is None:
            raise PortfolioS1FeedbackError("Qwen v4 terminal identity is unknown")
        reservation = build_feedback_call_reservation(
            selection, control, selected, verified_source=source
        )
        rebuilt = build_bound_feedback_artifact(
            selection,
            control,
            selected,
            source.packet,
            artifact.feedback_result,
            verified_source=source,
            reservation=reservation,
        )
        feedback = artifact.feedback_result.parsed_feedback
        if (
            artifact != rebuilt
            or artifact.artifact_sha256 != run_entry.artifact_sha256
            or artifact.status != "parsed"
            or artifact.feedback_result.cache_namespace != "feedback-evaluator-v9"
            or artifact.feedback_result.provider != "qwen"
            or artifact.feedback_result.model != "qwen3.7-plus-2026-05-26"
            or artifact.feedback_result.requested_thinking is not True
            or feedback is None
        ):
            raise PortfolioS1FeedbackError("Qwen v4 terminal bound artifact drifted")
        terminal_artifacts.append(
            PortfolioS1FeedbackTerminalArtifactRefV2(
                selection_ordinal=selected.selection_ordinal,
                selection_entry_sha256=selected.entry_sha256,
                capability=selected.capability,
                role=selected.role,
                status="parsed",
                artifact_file_sha256=sha256_bytes(artifact.canonical_bytes()),
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
            )
        )
        entries.append(
            PortfolioS1FeedbackBundleEntryV1(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                leakage_group_id=selected.leakage_group_id,
                gcs=selected.gcs,
                reason_codes=selected.reason_codes,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
                feedback=feedback,
            )
        )

    terminal_tuple = tuple(terminal_artifacts)
    entry_tuple = tuple(entries)
    coverage = _feedback_v2_coverage(selection, artifact_by_entry)
    full_gcs_binding, full_gcs_summary = _full_gcs_v2_diagnostics(verified)
    model_entries = tuple(
        PortfolioS1FeedbackModelEntryV1(
            selection_ordinal=item.selection_ordinal,
            capability=item.capability,
            role=item.role,
            primary_cluster=item.primary_cluster,
            gcs=item.gcs,
            reason_codes=item.reason_codes,
            feedback=item.feedback,
        )
        for item in entry_tuple
    )
    representative_ordinals = tuple(
        selected_by_hash[item].selection_ordinal
        for item in selection.canary_entry_sha256s
    ) + tuple(item.selection_ordinal for item in entry_tuple if item.role != "failure")
    entry_by_ordinal = {item.selection_ordinal: item for item in entry_tuple}
    representative_examples = tuple(
        PortfolioS1FeedbackRepresentativeExampleV1(
            selection_ordinal=ordinal,
            capability=entry_by_ordinal[ordinal].capability,
            role=entry_by_ordinal[ordinal].role,
            primary_cluster=entry_by_ordinal[ordinal].primary_cluster,
            turns=artifact_by_entry[
                selection.entries[ordinal - 1].entry_sha256
            ].feedback_packet.turns,
            response_text=artifact_by_entry[
                selection.entries[ordinal - 1].entry_sha256
            ].feedback_packet.response_text,
            cards=artifact_by_entry[
                selection.entries[ordinal - 1].entry_sha256
            ].feedback_packet.cards,
            tool_evidence=artifact_by_entry[
                selection.entries[ordinal - 1].entry_sha256
            ].feedback_packet.tool_evidence,
        )
        for ordinal in representative_ordinals
    )
    projection = PortfolioS1FeedbackModelProjectionV4(
        coverage=coverage,
        feedback_entries=model_entries,
        representative_examples=representative_examples,
        cluster_summaries=_cluster_summaries(entry_tuple),
        full_gcs_summary=full_gcs_summary,
    )
    selected_query_ids = tuple(item.query_id for item in selection.entries)
    _validate_model_projection_privacy(
        projection,
        private_query_ids=selected_query_ids,
    )
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V4,
        "status": "complete_diagnostic_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": verified.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "full_gcs_binding": full_gcs_binding,
        "selected_count": 48,
        "attempted_count": 48,
        "parsed_count": 48,
        "parse_error_count": 0,
        "provider_call_count": 48,
        "unattempted_count": 0,
        "missing_feedback_count": 0,
        "coverage": coverage,
        "excluded_execution_lapse_query_ids": (
            selection.excluded_execution_lapse_query_ids
        ),
        "selected_query_ids": selected_query_ids,
        "terminal_artifacts": terminal_tuple,
        "bound_artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in terminal_tuple]
        ),
        "entries": entry_tuple,
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV4.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_portfolio_s1_feedback_bundle_v4(
    path: str | Path,
    bundle: PortfolioS1FeedbackBundleV4,
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v4(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1FeedbackBundleV4:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle v4", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v4 file SHA-256 mismatch")
    bundle = PortfolioS1FeedbackBundleV4.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("S1 Feedback bundle v4 is not canonical JSON")
    return bundle


# ---------------------------------------------------------------------------
# Forward-only S1 Round-2 exact240 contracts.  Historical selected48 models
# and builders above are intentionally untouched so their canonical bytes stay
# replayable.


class FeedbackSelectionEntryV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    selection_ordinal: int = Field(ge=1, le=FEEDBACK_SELECTION_SIZE_V2)
    discovery_rank: int = Field(ge=1, le=600)
    query_id: str
    capability: str
    role: FeedbackRole
    primary_cluster: str
    interaction_pattern: str
    leakage_group_id: str
    atomic_component_id: str
    asset_id: str
    image_sha256: Sha256
    answer_mode: Literal["supported", "fallback", "unresolved"]
    gcs: Literal[0, 1]
    route_acceptable: Literal[0, 1]
    no_hard_error: Literal[0, 1]
    tool_contract_pass: Literal[0, 1]
    evidence_grounded: Literal[0, 1]
    output_contract_pass: Literal[0, 1]
    gcs_component_pass_count: int = Field(ge=0, le=5)
    reason_codes: tuple[str, ...]
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    strata: FeedbackSelectionStrataV1
    deterministic_tie_sha256: Sha256
    entry_sha256: Sha256

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _tuple_reasons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "query_id",
        "capability",
        "primary_cluster",
        "interaction_pattern",
        "leakage_group_id",
        "atomic_component_id",
        "asset_id",
    )
    @classmethod
    def _texts(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_entry(self) -> Self:
        if self.capability not in GCS_CAPABILITY_ORDER:
            raise ValueError("selection v2 capability is unknown")
        bits = (
            self.route_acceptable,
            self.no_hard_error,
            self.tool_contract_pass,
            self.evidence_grounded,
            self.output_contract_pass,
        )
        if self.gcs != int(all(bits)) or self.gcs_component_pass_count != sum(bits):
            raise ValueError("selection v2 GCS components drifted")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("selection v2 reason codes must be sorted and unique")
        if self.role == "success_anchor":
            if self.gcs != 1 or self.primary_cluster != "success":
                raise ValueError("selection v2 success anchor is not a GCS pass")
        elif self.role == "partial_anchor":
            if self.gcs != 0 or self.gcs_component_pass_count != 4:
                raise ValueError("selection v2 partial anchor is not a near-pass")
        elif self.gcs != 0:
            raise ValueError("selection v2 failure must be a GCS failure")
        if self.entry_sha256 != _model_hash(self, "entry_sha256"):
            raise ValueError("Feedback selection v2 entry self hash mismatch")
        return self


class FeedbackCapabilityQuotaV2(_StrictFrozenModel):
    capability: str
    selected_count: int = Field(ge=1, le=FEEDBACK_SELECTION_SIZE_V2)

    @field_validator("capability")
    @classmethod
    def _capability(cls, value: str) -> str:
        return _nonblank(value, "capability")


class PortfolioS1FeedbackSelectionV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-selection"] = "portfolio-s1-feedback-selection"
    policy_version: Literal["portfolio-s1-feedback-selection-v2"] = (
        FEEDBACK_SELECTION_POLICY_VERSION_V2
    )
    seed: int = Field(ge=0)
    corpus_sha256: Sha256
    fold_mapping_sha256: Sha256
    discovery_query_ids_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    discovery_count: Literal[600] = 600
    selected_count: Literal[240] = FEEDBACK_SELECTION_SIZE_V2
    capability_quotas: tuple[FeedbackCapabilityQuotaV2, ...]
    excluded_hard_error_component_ids: tuple[str, ...]
    excluded_hard_error_query_ids: tuple[str, ...]
    entries: tuple[FeedbackSelectionEntryV2, ...]
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = (
        12,
        60,
        120,
        240,
    )
    phase_entry_sha256s: tuple[tuple[Sha256, ...], ...]
    phase_query_ids: tuple[tuple[str, ...], ...]
    selected_asset_set_sha256: Sha256
    selection_sha256: Sha256

    @field_validator(
        "capability_quotas",
        "excluded_hard_error_component_ids",
        "excluded_hard_error_query_ids",
        "entries",
        "phase_counts",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("phase_entry_sha256s", "phase_query_ids", mode="before")
    @classmethod
    def _nested_tuples(cls, value: object) -> object:
        return (
            tuple(tuple(item) if isinstance(item, list) else item for item in value)
            if isinstance(value, list)
            else value
        )

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Feedback v2 phase counts drifted")
        if len(self.entries) != FEEDBACK_SELECTION_SIZE_V2 or tuple(
            item.selection_ordinal for item in self.entries
        ) != tuple(range(1, FEEDBACK_SELECTION_SIZE_V2 + 1)):
            raise ValueError("Feedback selection v2 must contain ordered240")
        unique_fields = (
            tuple(item.query_id for item in self.entries),
            tuple(item.leakage_group_id for item in self.entries),
            tuple(item.asset_id for item in self.entries),
            tuple(item.entry_sha256 for item in self.entries),
        )
        if any(
            len(set(values)) != FEEDBACK_SELECTION_SIZE_V2 for values in unique_fields
        ):
            raise ValueError(
                "Feedback selection v2 requires unique query/component/asset/entry"
            )
        expected_quotas = tuple(
            FeedbackCapabilityQuotaV2(capability=capability, selected_count=count)
            for capability, count in FEEDBACK_CAPABILITY_QUOTAS_V2.items()
        )
        if self.capability_quotas != expected_quotas:
            raise ValueError("Feedback selection v2 quotas drifted")
        observed = Counter(item.capability for item in self.entries)
        if observed != Counter(FEEDBACK_CAPABILITY_QUOTAS_V2):
            # Counter(mapping) preserves each mapping value as the expected count.
            raise ValueError("Feedback selection v2 capability counts drifted")
        encyclopedia_successes = tuple(
            item
            for item in self.entries
            if item.capability == "knowledge.visual_encyclopedia"
            and item.role == "success_anchor"
        )
        if len(encyclopedia_successes) != 5:
            raise ValueError(
                "Feedback selection v2 requires all five Encyclopedia passes"
            )
        quarantined = set(self.excluded_hard_error_component_ids)
        if quarantined & {item.leakage_group_id for item in self.entries}:
            raise ValueError("Feedback selection v2 admitted a quarantined component")
        expected_phases = tuple(
            tuple(item.entry_sha256 for item in self.entries[:count])
            for count in FEEDBACK_PHASE_COUNTS_V2
        )
        expected_queries = tuple(
            tuple(item.query_id for item in self.entries[:count])
            for count in FEEDBACK_PHASE_COUNTS_V2
        )
        if (
            self.phase_entry_sha256s != expected_phases
            or self.phase_query_ids != expected_queries
        ):
            raise ValueError("Feedback selection v2 phase partitions drifted")
        asset_payload = [
            {"asset_id": item.asset_id, "image_sha256": item.image_sha256}
            for item in sorted(self.entries, key=lambda entry: entry.asset_id)
        ]
        if self.selected_asset_set_sha256 != _hash_payload(asset_payload):
            raise ValueError("Feedback selection v2 asset-set hash mismatch")
        if self.selection_sha256 != _model_hash(self, "selection_sha256"):
            raise ValueError("Feedback selection v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _feedback_v2_role(row: VerifiedStaticGCSRow) -> FeedbackRole:
    if row.score.gcs == 1:
        return "success_anchor"
    if row.score.hard_error == 0 and _pass_count(row) == 4:
        return "partial_anchor"
    return "failure"


def _feedback_v2_interaction_pattern(row: VerifiedStaticGCSRow) -> str:
    names = tuple(item.tool_name for item in row.result.tool_trace)
    return ">".join(names) if names else "no_tool"


def _feedback_v2_features(row: VerifiedStaticGCSRow) -> frozenset[str]:
    role = _feedback_v2_role(row)
    values = {
        f"role:{role}",
        f"answer_mode:{row.score.answer_mode}",
        f"interaction:{_feedback_v2_interaction_pattern(row)}",
        f"boundary:{row.strata.boundary_status}",
        f"source:{row.strata.source_dataset}",
    }
    values.update(f"reason:{reason}" for reason in row.score.reason_codes)
    if row.strata.style_submode is not None:
        values.add(f"style:{row.strata.style_submode}")
    return frozenset(values)


def _select_feedback_v2_capability(
    rows: list[VerifiedStaticGCSRow],
    *,
    capability: str,
    quota: int,
    seed: int,
    used_components: set[str],
    used_assets: set[str],
) -> list[VerifiedStaticGCSRow]:
    candidates = [
        row
        for row in rows
        if row.leakage_group_id not in used_components
        and row.query.asset_id not in used_assets
    ]
    if capability == "knowledge.visual_encyclopedia":
        mandatory = [row for row in candidates if row.score.gcs == 1]
        if len(mandatory) != 5:
            raise PortfolioS1FeedbackError(
                "Discovery600 must contain exactly five Encyclopedia successes"
            )
    else:
        successes = [row for row in candidates if row.score.gcs == 1]
        mandatory = (
            [
                min(
                    successes,
                    key=lambda row: _seed_hash(
                        seed, f"v2-success:{capability}", row.query.query_id
                    ),
                )
            ]
            if successes
            else []
        )
    mandatory.sort(
        key=lambda row: _seed_hash(
            seed, f"v2-mandatory:{capability}", row.query.query_id
        )
    )
    selected: list[VerifiedStaticGCSRow] = []
    covered: set[str] = set()

    def admit(row: VerifiedStaticGCSRow) -> None:
        if row.leakage_group_id in used_components or row.query.asset_id in used_assets:
            raise PortfolioS1FeedbackError(
                "Feedback selection v2 mandatory identity collides globally"
            )
        selected.append(row)
        used_components.add(row.leakage_group_id)
        used_assets.add(row.query.asset_id)
        covered.update(_feedback_v2_features(row))

    for row in mandatory:
        admit(row)
    priority_reasons = {
        "fallback_contract_failed",
        "output_section_invalid",
        "tool_contract_failed",
    }
    while len(selected) < quota:
        available = [
            row
            for row in candidates
            if row not in selected
            and row.leakage_group_id not in used_components
            and row.query.asset_id not in used_assets
        ]
        if not available:
            raise PortfolioS1FeedbackError(
                f"{capability} cannot supply its Discovery600 Feedback quota"
            )

        def rank(row: VerifiedStaticGCSRow) -> tuple[object, ...]:
            features = _feedback_v2_features(row)
            priority = (
                len(priority_reasons.intersection(row.score.reason_codes))
                + int(row.score.answer_mode == "fallback")
                + int(_pass_count(row) == 4)
            )
            return (
                -len(features - covered),
                -priority if capability == "knowledge.visual_encyclopedia" else 0,
                -_pass_count(row),
                _seed_hash(seed, f"v2-fill:{capability}", row.query.query_id),
                row.query.query_id,
            )

        admit(min(available, key=rank))
    return selected


def _solve_feedback_v2_global_selection(
    eligible: list[VerifiedStaticGCSRow],
    *,
    seed: int,
) -> tuple[VerifiedStaticGCSRow, ...]:
    """Solve the exact quota/component/asset contract before deterministic ordering.

    A per-capability greedy walk can consume an asset needed by another atomic
    component even when the complete 240-row assignment is feasible.  The
    forward selector therefore solves the frozen binary problem in one pass.
    The objective first maximizes capability-local stratum coverage, then
    prefers success/near-pass and Encyclopedia contract diagnostics, and uses
    the seeded query hash only as the final deterministic tie-break.
    """

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    rows = tuple(sorted(eligible, key=lambda row: row.query.query_id))
    feature_names = tuple(
        sorted(
            {
                f"{row.query.canonical_capability}|{feature}"
                for row in rows
                for feature in _feedback_v2_features(row)
            }
        )
    )
    feature_index = {
        feature: len(rows) + index for index, feature in enumerate(feature_names)
    }
    variable_count = len(rows) + len(feature_names)
    objective = np.zeros(variable_count, dtype=float)
    priority_reasons = {
        "fallback_contract_failed",
        "output_section_invalid",
        "tool_contract_failed",
    }
    for index, row in enumerate(rows):
        role = _feedback_v2_role(row)
        quality = (
            400 * int(role == "success_anchor")
            + 250 * int(role == "partial_anchor")
            + 200 * int(row.score.answer_mode == "fallback")
            + 300 * len(priority_reasons.intersection(row.score.reason_codes))
        )
        tie = int(
            _seed_hash(seed, "v2-global-milp", row.query.query_id)[:12], 16
        ) / float(16**12)
        objective[index] = -float(quality) + tie * 0.01
    for feature in feature_names:
        objective[feature_index[feature]] = -10_000.0

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: dict[int, float], minimum: float, maximum: float
    ) -> None:
        constraint_index = len(lower)
        for column, value in coefficients.items():
            row_indices.append(constraint_index)
            column_indices.append(column)
            values.append(value)
        lower.append(minimum)
        upper.append(maximum)

    for capability, quota in FEEDBACK_CAPABILITY_QUOTAS_V2.items():
        capability_columns = {
            index: 1.0
            for index, row in enumerate(rows)
            if row.query.canonical_capability == capability
        }
        add_constraint(capability_columns, float(quota), float(quota))
        failure_columns = {
            index: 1.0
            for index, row in enumerate(rows)
            if row.query.canonical_capability == capability and row.score.gcs == 0
        }
        if not failure_columns:
            raise PortfolioS1FeedbackError(
                f"Discovery600 has no failure anchor for {capability}"
            )
        add_constraint(failure_columns, 1.0, np.inf)
        success_columns = {
            index: 1.0
            for index, row in enumerate(rows)
            if row.query.canonical_capability == capability and row.score.gcs == 1
        }
        if success_columns:
            add_constraint(success_columns, 1.0, np.inf)

    by_component: dict[str, dict[int, float]] = defaultdict(dict)
    by_asset: dict[str, dict[int, float]] = defaultdict(dict)
    for index, row in enumerate(rows):
        by_component[row.leakage_group_id][index] = 1.0
        by_asset[row.query.asset_id][index] = 1.0
    for coefficients in (*by_component.values(), *by_asset.values()):
        add_constraint(coefficients, -np.inf, 1.0)

    encyclopedia_successes = {
        index: 1.0
        for index, row in enumerate(rows)
        if row.query.canonical_capability == "knowledge.visual_encyclopedia"
        and row.score.gcs == 1
    }
    if len(encyclopedia_successes) != 5:
        raise PortfolioS1FeedbackError(
            "Discovery600 must contain exactly five Encyclopedia successes"
        )
    for index in encyclopedia_successes:
        add_constraint({index: 1.0}, 1.0, 1.0)

    for feature in feature_names:
        capability, feature_name = feature.split("|", 1)
        coefficients = {
            index: -1.0
            for index, row in enumerate(rows)
            if row.query.canonical_capability == capability
            and feature_name in _feedback_v2_features(row)
        }
        coefficients[feature_index[feature]] = 1.0
        add_constraint(coefficients, -np.inf, 0.0)

    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(lower), variable_count),
    ).tocsc()
    result = milp(
        objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise PortfolioS1FeedbackError(
            "Discovery600 cannot satisfy the exact global Feedback240 contract"
        )
    selected = tuple(row for index, row in enumerate(rows) if result.x[index] > 0.5)
    if len(selected) != FEEDBACK_SELECTION_SIZE_V2:
        raise PortfolioS1FeedbackError("global Feedback240 solver count drifted")
    return selected


def _phase_order_feedback_v2(
    selected_by_capability: dict[str, list[VerifiedStaticGCSRow]],
    *,
    seed: int,
) -> tuple[VerifiedStaticGCSRow, ...]:
    first: list[VerifiedStaticGCSRow] = []
    remaining: dict[str, list[VerifiedStaticGCSRow]] = {}
    for capability in GCS_CAPABILITY_ORDER:
        members = list(selected_by_capability[capability])
        failures = [row for row in members if row.score.gcs == 0]
        anchors = [row for row in members if row.score.gcs == 1]
        failure = min(
            failures,
            key=lambda row: (
                -len(
                    {
                        "fallback_contract_failed",
                        "output_section_invalid",
                        "tool_contract_failed",
                    }.intersection(row.score.reason_codes)
                ),
                _seed_hash(seed, f"v2-phase-failure:{capability}", row.query.query_id),
            ),
        )
        second_pool = anchors or [row for row in failures if row != failure]
        second = min(
            second_pool,
            key=lambda row: _seed_hash(
                seed, f"v2-phase-anchor:{capability}", row.query.query_id
            ),
        )
        first.extend((failure, second))
        phase_ids = {failure.query.query_id, second.query.query_id}
        remaining[capability] = [
            row for row in members if row.query.query_id not in phase_ids
        ]
    ordered = list(first)
    while any(remaining.values()):
        for capability in GCS_CAPABILITY_ORDER:
            if remaining[capability]:
                ordered.append(remaining[capability].pop(0))
    return tuple(ordered)


def build_portfolio_s1_feedback_selection_v2(
    corpus: VerifiedStaticGCSCorpus,
    discovery_query_ids: tuple[str, ...],
    *,
    discovery_query_ids_sha256: str,
    fold_mapping_sha256: str,
    seed: int = 20260808,
) -> PortfolioS1FeedbackSelectionV2:
    """Select exact240 only from the frozen Discovery600 with quarantine."""

    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    if seed < 0:
        raise PortfolioS1FeedbackError("Feedback selection v2 seed is invalid")
    if (
        len(discovery_query_ids) != 600
        or len(set(discovery_query_ids)) != 600
        or discovery_query_ids_sha256 != _hash_payload(sorted(discovery_query_ids))
        or _SHA_RE.fullmatch(fold_mapping_sha256) is None
    ):
        raise PortfolioS1FeedbackError("Discovery600 identity is invalid")
    row_by_query = verified.row_by_query_id()
    if not set(discovery_query_ids).issubset(row_by_query):
        raise PortfolioS1FeedbackError("Discovery600 differs from Static GCS corpus")
    discovery_rank = {
        query_id: index
        for index, query_id in enumerate(sorted(discovery_query_ids), start=1)
    }
    discovery_rows = [row_by_query[query_id] for query_id in discovery_query_ids]
    # Selection "component" is the frozen leakage_group unit, not the broader
    # GCS bootstrap connected component.  The latter has only 23 units per
    # capability and cannot satisfy the exact 55/40/... identity quotas.
    hard_components = {
        row.leakage_group_id for row in discovery_rows if row.score.hard_error == 1
    }
    excluded_query_ids = tuple(
        sorted(
            row.query.query_id
            for row in discovery_rows
            if row.leakage_group_id in hard_components
        )
    )
    eligible = [
        row for row in discovery_rows if row.leakage_group_id not in hard_components
    ]
    selected_by_capability: dict[str, list[VerifiedStaticGCSRow]] = {
        capability: [] for capability in FEEDBACK_CAPABILITY_QUOTAS_V2
    }
    for row in _solve_feedback_v2_global_selection(eligible, seed=seed):
        selected_by_capability[row.query.canonical_capability].append(row)
    for capability, rows in selected_by_capability.items():
        rows.sort(
            key=lambda row: (
                _seed_hash(seed, f"v2-selected-order:{capability}", row.query.query_id),
                row.query.query_id,
            )
        )
    ordered_rows = _phase_order_feedback_v2(selected_by_capability, seed=seed)
    if len(ordered_rows) != FEEDBACK_SELECTION_SIZE_V2:
        raise PortfolioS1FeedbackError("Feedback selection v2 did not produce 240 rows")
    entries: list[FeedbackSelectionEntryV2] = []
    for ordinal, row in enumerate(ordered_rows, start=1):
        role = _feedback_v2_role(row)
        cluster = (
            "success"
            if role == "success_anchor"
            else "partial_anchor"
            if role == "partial_anchor"
            else _primary_cluster(row)
        )
        unsigned = {
            "schema_version": 2,
            "selection_ordinal": ordinal,
            "discovery_rank": discovery_rank[row.query.query_id],
            "query_id": row.query.query_id,
            "capability": row.query.canonical_capability,
            "role": role,
            "primary_cluster": cluster,
            "interaction_pattern": _feedback_v2_interaction_pattern(row),
            "leakage_group_id": row.leakage_group_id,
            "atomic_component_id": row.atomic_component_id,
            "asset_id": row.query.asset_id,
            "image_sha256": row.image_sha256,
            "answer_mode": row.score.answer_mode,
            "gcs": row.score.gcs,
            "route_acceptable": row.score.route_acceptable,
            "no_hard_error": row.score.no_hard_error,
            "tool_contract_pass": row.score.tool_contract_pass,
            "evidence_grounded": row.score.evidence_grounded,
            "output_contract_pass": row.score.output_contract_pass,
            "gcs_component_pass_count": _pass_count(row),
            "reason_codes": row.score.reason_codes,
            "checkpoint_file_sha256": row.checkpoint_file_sha256,
            "checkpoint_row_sha256": row.checkpoint_row_sha256,
            "sidecar_sha256": row.sidecar.evidence_sha256,
            "strata": FeedbackSelectionStrataV1(
                source_dataset=row.strata.source_dataset,
                repair_status=row.strata.repair_status,  # type: ignore[arg-type]
                boundary_status=row.strata.boundary_status,  # type: ignore[arg-type]
                style_submode=row.strata.style_submode,
            ),
            "deterministic_tie_sha256": _seed_hash(
                seed, f"v2-selected:{role}:{cluster}", row.query.query_id
            ),
        }
        entries.append(
            FeedbackSelectionEntryV2.model_validate(
                {**unsigned, "entry_sha256": _hash_payload(unsigned)}, strict=True
            )
        )
    asset_payload = [
        {"asset_id": item.asset_id, "image_sha256": item.image_sha256}
        for item in sorted(entries, key=lambda entry: entry.asset_id)
    ]
    unsigned_selection = {
        "schema_version": 2,
        "kind": "portfolio-s1-feedback-selection",
        "policy_version": FEEDBACK_SELECTION_POLICY_VERSION_V2,
        "seed": seed,
        "corpus_sha256": verified.corpus_sha256,
        "fold_mapping_sha256": fold_mapping_sha256,
        "discovery_query_ids_sha256": discovery_query_ids_sha256,
        "parent_static_bank_sha256": verified.runtime.bank.bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "discovery_count": 600,
        "selected_count": 240,
        "capability_quotas": tuple(
            FeedbackCapabilityQuotaV2(capability=capability, selected_count=count)
            for capability, count in FEEDBACK_CAPABILITY_QUOTAS_V2.items()
        ),
        "excluded_hard_error_component_ids": tuple(sorted(hard_components)),
        "excluded_hard_error_query_ids": excluded_query_ids,
        "entries": tuple(entries),
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "phase_entry_sha256s": tuple(
            tuple(item.entry_sha256 for item in entries[:count])
            for count in FEEDBACK_PHASE_COUNTS_V2
        ),
        "phase_query_ids": tuple(
            tuple(item.query_id for item in entries[:count])
            for count in FEEDBACK_PHASE_COUNTS_V2
        ),
        "selected_asset_set_sha256": _hash_payload(asset_payload),
    }
    return PortfolioS1FeedbackSelectionV2.model_validate(
        {
            **unsigned_selection,
            "selection_sha256": _hash_payload(_jsonable(unsigned_selection)),
        },
        strict=True,
    )


def write_portfolio_s1_feedback_selection_v2(
    path: str | Path, selection: PortfolioS1FeedbackSelectionV2
) -> Path:
    return atomic_create_file(path, selection.canonical_bytes())


def load_portfolio_s1_feedback_selection_v2(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackSelectionV2:
    content = read_stable_regular_file(
        path, label="S1 Feedback selection v2", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback selection v2 file SHA-256 mismatch")
    selection = PortfolioS1FeedbackSelectionV2.model_validate_json(content, strict=True)
    if selection.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback selection v2 is not canonical JSON")
    return selection


class PortfolioS1FeedbackAuthorizationV4(_StrictFrozenModel):
    """Owner authority for exactly the forward selected240 Qwen images."""

    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v4"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION_V4
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-discovery-selected-240-assets"] = (
        "core-opt800-s1-feedback-discovery-selected-240-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = "qwen3.7-plus-2026-05-26"
    processor: Literal["dashscope-qwen37-feedback"] = "dashscope-qwen37-feedback"
    cache_namespace: Literal["feedback-evaluator-v10"] = "feedback-evaluator-v10"
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[QWEN37_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256] = (
        QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    )
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN37_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN37_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    selected_query_count: Literal[240] = QWEN37_FEEDBACK_SELECTED_COUNT_V2
    provider_call_ceiling: Literal[240] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    feedback_concurrency: Literal[2] = 2
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["17.483520000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )
    phase_hard_cap_cny: Literal["18.000000000000"] = (
        QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2
    )
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    parent_membership_processor: Literal["dashscope-qwen-assistant"] = (
        "dashscope-qwen-assistant"
    )
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    parent_remote_catalog_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    selected_assets: tuple[SelectedQwenFeedbackAssetV1, ...]
    selected_asset_set_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator("selected_assets", "phase_counts", mode="before")
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "authorization_id",
        "reviewer_id",
        "owner_statement",
        "parent_remote_authorization_id",
    )
    @classmethod
    def _texts(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if (
            info.field_name == "authorization_id"
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None
        ):
            raise ValueError("authorization_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen Feedback v4 reviewed_at needs a timezone")
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen Feedback v4 phases drifted")
        if (
            len(self.selected_assets) != 240
            or tuple(item.query_id for item in self.selected_assets)
            != tuple(sorted(item.query_id for item in self.selected_assets))
            or len({item.query_id for item in self.selected_assets}) != 240
            or len({item.asset_id for item in self.selected_assets}) != 240
        ):
            raise ValueError("Qwen Feedback v4 must bind query-sorted unique240")
        asset_payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != _hash_payload(asset_payload):
            raise ValueError("Qwen Feedback v4 asset-set hash drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256_V2,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256_V2,
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V2,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256_V2,
        ):
            raise ValueError("Qwen Feedback v4 source/role identity drifted")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Qwen Feedback v4 authorization self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _selected_qwen_feedback_assets_v4(
    selection: PortfolioS1FeedbackSelectionV2,
) -> tuple[SelectedQwenFeedbackAssetV1, ...]:
    return tuple(
        SelectedQwenFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )


def validate_selected_qwen_feedback_authorization_v4(
    authorization: PortfolioS1FeedbackAuthorizationV4,
    selection: PortfolioS1FeedbackSelectionV2,
) -> None:
    assets = _selected_qwen_feedback_assets_v4(selection)
    if (
        type(authorization) is not PortfolioS1FeedbackAuthorizationV4
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
        or authorization.selected_asset_set_sha256
        != _hash_payload([item.model_dump(mode="json") for item in assets])
    ):
        raise PortfolioS1FeedbackError(
            "Qwen Feedback authorization v4 differs from selected240"
        )


def build_selected_qwen_feedback_authorization_v4(
    selection: PortfolioS1FeedbackSelectionV2,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
    model_source_lock_file_sha256: str,
    model_source_lock_sha256: str,
    pricing_lock_file_sha256: str,
    pricing_lock_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV4:
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selected_qwen_feedback_assets_v4(selection)
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-feedback-selected-asset-authorization",
        "policy_version": FEEDBACK_AUTHORIZATION_POLICY_VERSION_V4,
        "authorization_id": authorization_id,
        "status": "owner-approved",
        "scope": "core-opt800-s1-feedback-discovery-selected-240-assets",
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "owner_statement": owner_statement,
        "selection_sha256": selection.selection_sha256,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "processor": "dashscope-qwen37-feedback",
        "cache_namespace": "feedback-evaluator-v10",
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "requested_response_format": "json_schema",
        "requested_json_schema_policy_version": QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION,
        "requested_json_schema_name": QWEN37_FEEDBACK_JSON_SCHEMA_NAME,
        "requested_json_schema_sha256": QWEN37_FEEDBACK_JSON_SCHEMA_SHA256,
        "transport_policy_version": QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_thinking": True,
        "requested_thinking_budget": QWEN37_FEEDBACK_THINKING_BUDGET,
        "requested_timeout_seconds": QWEN37_FEEDBACK_TIMEOUT_SECONDS,
        "requested_temperature": None,
        "requested_top_p": None,
        "requested_max_tokens": None,
        "max_completion_tokens": QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS,
        "selected_query_count": 240,
        "provider_call_ceiling": 240,
        "max_attempts": 1,
        "retry_policy": "no_retry",
        "feedback_concurrency": 2,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "input_token_reservation_ceiling_per_call": 20000,
        "output_token_reservation_ceiling_per_call": 4106,
        "per_call_reservation_cny": QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY,
        "maximum_reservation_cny": QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
        "phase_hard_cap_cny": QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2,
        "over_budget_policy": "fail_closed_before_provider_call",
        "parent_membership_processor": "dashscope-qwen-assistant",
        "parent_remote_authorization_id": parent.authorization.authorization_id,
        "parent_remote_authorization_file_sha256": parent.authorization_file_sha256,
        "parent_remote_receipt_file_sha256": parent.receipt_file_sha256,
        "parent_remote_receipt_sha256": parent.receipt.receipt_sha256,
        "parent_remote_catalog_sha256": parent.catalog.catalog_sha256,
        "model_source_lock_file_sha256": model_source_lock_file_sha256,
        "model_source_lock_sha256": model_source_lock_sha256,
        "pricing_lock_file_sha256": pricing_lock_file_sha256,
        "pricing_lock_sha256": pricing_lock_sha256,
        "role_selection_file_sha256": role_selection_file_sha256,
        "role_selection_sha256": role_selection_sha256,
        "cloud_upload_allowed": True,
        "remote_model_inference_allowed": True,
        "redistribution_allowed": False,
        "public_demo_allowed": False,
        "selected_assets": assets,
        "selected_asset_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in assets]
        ),
    }
    return PortfolioS1FeedbackAuthorizationV4.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1FeedbackControlV9(_StrictFrozenModel):
    schema_version: Literal[9] = 9
    kind: Literal["portfolio-s1-feedback-control"] = "portfolio-s1-feedback-control"
    policy_version: Literal["portfolio-s1-feedback-control-v9"] = (
        FEEDBACK_CONTROL_POLICY_VERSION_V9
    )
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_id: str
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    parser_policy_version: Literal["visual-feedback-free-text-trim-v3"] = (
        VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
    )
    parser_policy_sha256: Literal[VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3] = (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
    )
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256] = (
        QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    )
    requested_json_schema_sha256: Literal[QWEN37_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = "qwen3.7-plus-2026-05-26"
    processor: Literal["dashscope-qwen37-feedback"] = "dashscope-qwen37-feedback"
    cache_namespace: Literal["feedback-evaluator-v10"] = "feedback-evaluator-v10"
    selected_query_count: Literal[240] = 240
    provider_call_ceiling: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    phase_entry_sha256s: tuple[tuple[Sha256, ...], ...]
    phase_query_ids: tuple[tuple[str, ...], ...]
    rubric: RubricSnapshot
    feedback_concurrency: Literal[2] = 2
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    max_completion_tokens: Literal[4096] = 4096
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = 2048
    requested_timeout_seconds: Literal[600] = 600
    require_all_parsed_for_bundle: Literal[True] = True
    stop_on_nonparsed_or_orphan: Literal[True] = True
    control_sha256: Sha256

    @field_validator("phase_counts", mode="before")
    @classmethod
    def _phase_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("phase_entry_sha256s", "phase_query_ids", mode="before")
    @classmethod
    def _phase_nested(cls, value: object) -> object:
        return (
            tuple(tuple(item) if isinstance(item, list) else item for item in value)
            if isinstance(value, list)
            else value
        )

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Feedback control v9 phases drifted")
        if self.control_sha256 != _model_hash(self, "control_sha256"):
            raise ValueError("Feedback control v9 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_portfolio_s1_feedback_control_v9(
    control: PortfolioS1FeedbackControlV9,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV4,
) -> None:
    validate_selected_qwen_feedback_authorization_v4(authorization, selection)
    if (
        type(control) is not PortfolioS1FeedbackControlV9
        or control.selection_sha256 != selection.selection_sha256
        or control.authorization_sha256 != authorization.authorization_sha256
        or control.authorization_id != authorization.authorization_id
        or control.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or control.corpus_sha256 != selection.corpus_sha256
        or control.parent_static_bank_sha256 != selection.parent_static_bank_sha256
        or control.phase_entry_sha256s != selection.phase_entry_sha256s
        or control.phase_query_ids != selection.phase_query_ids
    ):
        raise PortfolioS1FeedbackError("Feedback control v9 binding drifted")


def build_portfolio_s1_feedback_control_v9(
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV4,
    *,
    rubric: RubricSnapshot,
) -> PortfolioS1FeedbackControlV9:
    validate_selected_qwen_feedback_authorization_v4(authorization, selection)
    unsigned = {
        "schema_version": 9,
        "kind": "portfolio-s1-feedback-control",
        "policy_version": FEEDBACK_CONTROL_POLICY_VERSION_V9,
        "selection_sha256": selection.selection_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_id": authorization.authorization_id,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_json_schema_sha256": QWEN37_FEEDBACK_JSON_SCHEMA_SHA256,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "processor": "dashscope-qwen37-feedback",
        "cache_namespace": "feedback-evaluator-v10",
        "selected_query_count": 240,
        "provider_call_ceiling": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "phase_entry_sha256s": selection.phase_entry_sha256s,
        "phase_query_ids": selection.phase_query_ids,
        "rubric": rubric,
        "feedback_concurrency": 2,
        "max_attempts": 1,
        "retry_policy": "no_retry",
        "max_completion_tokens": 4096,
        "requested_thinking": True,
        "requested_thinking_budget": 2048,
        "requested_timeout_seconds": 600,
        "require_all_parsed_for_bundle": True,
        "stop_on_nonparsed_or_orphan": True,
    }
    return PortfolioS1FeedbackControlV9.model_validate(
        {**unsigned, "control_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1QwenFeedbackLaunchLockV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-qwen37-feedback-launch-lock"] = (
        "portfolio-s1-qwen37-feedback-launch-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-launch-lock-v2"] = (
        QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION_V2
    )
    run_id: str
    status: Literal["prepared-no-provider-calls"] = "prepared-no-provider-calls"
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    corpus_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = "qwen3.7-plus-2026-05-26"
    processor: Literal["dashscope-qwen37-feedback"] = "dashscope-qwen37-feedback"
    cache_namespace: Literal["feedback-evaluator-v10"] = "feedback-evaluator-v10"
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    provider_call_ceiling: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    feedback_concurrency: Literal[2] = 2
    max_attempts_per_selected_query: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["17.483520000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )
    phase_hard_cap_cny: Literal["18.000000000000"] = (
        QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2
    )
    provider_calls_performed: Literal[0] = 0
    launch_lock_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        value = _nonblank(value, "run_id")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None:
            raise ValueError("run_id must be canonical")
        return value

    @field_validator("phase_counts", mode="before")
    @classmethod
    def _phases(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen Feedback launch v2 phases drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256_V2,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256_V2,
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V2,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256_V2,
        ):
            raise ValueError("Qwen Feedback launch v2 source/role drifted")
        if self.launch_lock_sha256 != _model_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen Feedback launch v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_qwen37_feedback_launch_lock_v2(
    *,
    run_id: str,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV4,
    control: PortfolioS1FeedbackControlV9,
) -> PortfolioS1QwenFeedbackLaunchLockV2:
    validate_portfolio_s1_feedback_control_v9(control, selection, authorization)
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-qwen37-feedback-launch-lock",
        "policy_version": QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION_V2,
        "run_id": run_id,
        "status": "prepared-no-provider-calls",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "model_source_lock_file_sha256": authorization.model_source_lock_file_sha256,
        "model_source_lock_sha256": authorization.model_source_lock_sha256,
        "pricing_lock_file_sha256": authorization.pricing_lock_file_sha256,
        "pricing_lock_sha256": authorization.pricing_lock_sha256,
        "role_selection_file_sha256": authorization.role_selection_file_sha256,
        "role_selection_sha256": authorization.role_selection_sha256,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "processor": "dashscope-qwen37-feedback",
        "cache_namespace": "feedback-evaluator-v10",
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "provider_call_ceiling": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "feedback_concurrency": 2,
        "max_attempts_per_selected_query": 1,
        "retry_policy": "no_retry",
        "per_call_reservation_cny": QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY,
        "maximum_reservation_cny": QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
        "phase_hard_cap_cny": QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2,
        "provider_calls_performed": 0,
    }
    return PortfolioS1QwenFeedbackLaunchLockV2.model_validate(
        {**unsigned, "launch_lock_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1FeedbackRunEntryV2(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    artifact_sha256: Sha256 | None = None
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_entry(self) -> Self:
        if (self.status == "orphan") != (self.artifact_sha256 is None):
            raise ValueError("Feedback run v2 orphan/artifact identity drifted")
        if self.status == "orphan" and (
            self.input_tokens is not None or self.output_tokens is not None
        ):
            raise ValueError("Feedback run v2 orphan cannot claim usage")
        return self


class PortfolioS1FeedbackRunV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-run"] = "portfolio-s1-feedback-run"
    policy_version: Literal["portfolio-s1-feedback-run-v2"] = (
        S1_FEEDBACK_RUN_POLICY_VERSION_V2
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    expected_count: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    attempted_count: int = Field(ge=1, le=240)
    parsed_count: int = Field(ge=0, le=240)
    error_count: int = Field(ge=0, le=240)
    orphan_count: int = Field(ge=0, le=240)
    provider_calls_reserved: int = Field(ge=1, le=240)
    terminal_phase_count: Literal[12, 60, 120, 240]
    status: Literal["stopped_nonparsed", "stopped_orphan", "completed"]
    usage_known_count: int = Field(ge=0, le=240)
    usage_unknown_count: int = Field(ge=0, le=240)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    artifacts: tuple[PortfolioS1FeedbackRunEntryV2, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator("phase_counts", "artifacts", mode="before")
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        statuses = Counter(item.status for item in self.artifacts)
        ordinals = tuple(item.selection_ordinal for item in self.artifacts)
        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or len(self.artifacts) != self.attempted_count
            or ordinals != tuple(sorted(set(ordinals)))
            or len({item.selection_entry_sha256 for item in self.artifacts})
            != self.attempted_count
            or self.parsed_count != statuses["parsed"]
            or self.orphan_count != statuses["orphan"]
            or self.error_count != self.attempted_count - self.parsed_count
            or self.provider_calls_reserved != self.attempted_count
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count != self.attempted_count - self.usage_known_count
            or self.input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
        ):
            raise ValueError("S1 Feedback run v2 counts drifted")
        if self.status == "completed":
            if (
                self.attempted_count != 240
                or self.parsed_count != 240
                or ordinals != tuple(range(1, 241))
            ):
                raise ValueError("completed S1 Feedback run v2 is not parsed240")
        elif self.status == "stopped_orphan":
            if self.orphan_count < 1:
                raise ValueError("stopped-orphan run v2 lacks an orphan")
        elif self.error_count < 1 or self.orphan_count:
            raise ValueError("stopped-nonparsed run v2 status drifted")
        if self.terminal_phase_count not in FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Feedback run v2 terminal phase drifted")
        if self.artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.artifacts]
        ):
            raise ValueError("Feedback run v2 artifact set hash drifted")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Feedback run v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_run_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    artifacts: tuple[object, ...],
    *,
    orphaned_entry_sha256s: tuple[str, ...] = (),
) -> PortfolioS1FeedbackRunV2:
    if control.selection_sha256 != selection.selection_sha256:
        raise PortfolioS1FeedbackError("Feedback run v2 control binding drifted")
    artifact_by_entry: dict[str, object] = {}
    for artifact in artifacts:
        entry_sha = getattr(artifact, "selection_entry_sha256", None)
        if not isinstance(entry_sha, str) or entry_sha in artifact_by_entry:
            raise PortfolioS1FeedbackError("Feedback run v2 artifacts repeat")
        artifact_by_entry[entry_sha] = artifact
    orphaned = set(orphaned_entry_sha256s)
    if len(orphaned) != len(orphaned_entry_sha256s) or orphaned & set(
        artifact_by_entry
    ):
        raise PortfolioS1FeedbackError("Feedback run v2 orphan identity drifted")
    rows: list[PortfolioS1FeedbackRunEntryV2] = []
    for entry in selection.entries:
        artifact = artifact_by_entry.get(entry.entry_sha256)
        if artifact is None and entry.entry_sha256 not in orphaned:
            continue
        if artifact is None:
            rows.append(
                PortfolioS1FeedbackRunEntryV2(
                    selection_ordinal=entry.selection_ordinal,
                    selection_entry_sha256=entry.entry_sha256,
                    status="orphan",
                )
            )
            continue
        result = getattr(artifact, "feedback_result", None)
        if (
            getattr(artifact, "selection_sha256", None) != selection.selection_sha256
            or getattr(artifact, "control_sha256", None) != control.control_sha256
            or getattr(artifact, "query_id", None) != entry.query_id
            or result is None
        ):
            raise PortfolioS1FeedbackError("Feedback run v2 artifact binding drifted")
        usage = getattr(result, "usage", None)
        rows.append(
            PortfolioS1FeedbackRunEntryV2(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                artifact_sha256=getattr(artifact, "artifact_sha256"),
                status=getattr(artifact, "status"),
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
            )
        )
    if not rows or len(rows) != len(artifacts) + len(orphaned):
        raise PortfolioS1FeedbackError("Feedback run v2 attempted identities drifted")
    errors = [item for item in rows if item.status != "parsed"]
    if not errors and len(rows) != 240:
        raise PortfolioS1FeedbackError(
            "nonterminal parsed Feedback v2 artifacts cannot publish a run"
        )
    terminal_phase = next(
        count
        for count in FEEDBACK_PHASE_COUNTS_V2
        if max(item.selection_ordinal for item in rows) <= count
    )
    status = (
        "completed"
        if not errors
        else "stopped_orphan"
        if any(item.status == "orphan" for item in errors)
        else "stopped_nonparsed"
    )
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-feedback-run",
        "policy_version": S1_FEEDBACK_RUN_POLICY_VERSION_V2,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "attempted_count": len(rows),
        "parsed_count": len(rows) - len(errors),
        "error_count": len(errors),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "provider_calls_reserved": len(rows),
        "terminal_phase_count": terminal_phase,
        "status": status,
        "usage_known_count": sum(item.input_tokens is not None for item in rows),
        "usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRunV2.model_validate(
        {**unsigned, "run_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


@dataclass(frozen=True)
class VerifiedStaticFeedbackSourceV2:
    corpus_sha256: str
    selection_sha256: str
    control_sha256: str
    selection_entry_sha256: str
    corpus: VerifiedStaticGCSCorpus = field(repr=False, compare=False)
    row: VerifiedStaticGCSRow = field(repr=False, compare=False)
    asset_catalog: AssetCatalog = field(repr=False, compare=False)
    packet: FeedbackPacketV3
    _marker: object = field(repr=False, compare=False)


_VERIFIED_STATIC_FEEDBACK_SOURCE_V2_MARKER = object()


def _accept_loaded_feedback_source_context_v2(
    corpus: VerifiedStaticGCSCorpus,
) -> tuple[VerifiedStaticGCSCorpus, AssetCatalog]:
    """Reuse the exact catalog proof created by the first deep corpus load."""

    verified = accept_loaded_verified_static_gcs_corpus(corpus)
    core_inputs = verified.core_inputs
    catalogs = core_inputs._verified_catalogs
    if (
        type(core_inputs) is not VerifiedPortfolioCoreInputs
        or not isinstance(catalogs, tuple)
        or len(catalogs) != 2
        or any(type(item) is not AssetCatalog for item in catalogs)
    ):
        raise PortfolioS1FeedbackError(
            "loaded Feedback source v2 lacks its typed Core catalog proof"
        )
    catalog = catalogs[1]
    try:
        catalog.require_verified_files()
    except (OSError, ValueError) as error:
        raise PortfolioS1FeedbackError(
            "loaded Feedback source v2 catalog proof is invalid"
        ) from error
    if catalog.catalog_sha256 != core_inputs.expected_output_catalog_sha256:
        raise PortfolioS1FeedbackError(
            "loaded Feedback source v2 catalog identity drifted"
        )
    return verified, catalog


def _feedback_gcs_contract_v1(
    verified: VerifiedStaticGCSCorpus, capability: str
) -> FeedbackGCSContractV1:
    # GCS v2 is an incremental policy: it inherits the v1 non-Style oracle and
    # fallback grammar, then adds the public-safe Style adapter. Projecting
    # from v2 alone would omit five capabilities and every fallback marker.
    predecessor = gcs_policy_payload()
    policy = gcs_v2_policy_payload()
    predecessor_oracle = predecessor["oracle_contract"]
    oracle = policy["oracle_contract"]
    assert isinstance(predecessor_oracle, dict) and isinstance(oracle, dict)
    predecessor_capabilities = predecessor_oracle["capabilities"]
    assert isinstance(predecessor_capabilities, dict)
    if capability == "product.style_recommendation":
        style = oracle["style"]
        assert isinstance(style, dict)
        sequences = style["allowed_tool_sequences"]
    else:
        capability_policy = predecessor_capabilities[capability]
        assert isinstance(capability_policy, dict)
        sequences = capability_policy["allowed_tool_sequences"]
    fallback = predecessor["fallback_markers"]
    assert isinstance(sequences, list) and isinstance(fallback, dict)
    markers = fallback[capability]
    assert isinstance(markers, list)
    task = verified.task_spec.capabilities_by_id[capability]
    return FeedbackGCSContractV1(
        policy_sha256=GCS_V2_POLICY_SHA256,
        required_sections=task.output_contract.required_sections,
        fallback_markers=tuple(markers),
        preferred_fallback_marker=markers[0],
        card_requirement=task.output_contract.card_requirement,
        legal_tool_sequences=tuple(tuple(item) for item in sequences),
    )


def _entry_v2_matches_verified_row(
    entry: FeedbackSelectionEntryV2, row: VerifiedStaticGCSRow
) -> bool:
    return (
        row.query.query_id == entry.query_id
        and row.query.canonical_capability == entry.capability
        and row.query.asset_id == entry.asset_id
        and row.leakage_group_id == entry.leakage_group_id
        and row.atomic_component_id == entry.atomic_component_id
        and row.image_sha256 == entry.image_sha256
        and row.score.answer_mode == entry.answer_mode
        and row.score.gcs == entry.gcs
        and row.score.route_acceptable == entry.route_acceptable
        and row.score.no_hard_error == entry.no_hard_error
        and row.score.tool_contract_pass == entry.tool_contract_pass
        and row.score.evidence_grounded == entry.evidence_grounded
        and row.score.output_contract_pass == entry.output_contract_pass
        and tuple(row.score.reason_codes) == entry.reason_codes
        and row.checkpoint_file_sha256 == entry.checkpoint_file_sha256
        and row.checkpoint_row_sha256 == entry.checkpoint_row_sha256
        and row.sidecar.evidence_sha256 == entry.sidecar_sha256
    )


def build_verified_static_feedback_sources_v2(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
) -> tuple[VerifiedStaticFeedbackSourceV2, ...]:
    verified, catalog = _accept_loaded_feedback_source_context_v2(corpus)
    if (
        selection.corpus_sha256 != verified.corpus_sha256
        or control.selection_sha256 != selection.selection_sha256
        or control.corpus_sha256 != verified.corpus_sha256
        or control.parent_static_bank_sha256 != verified.runtime.bank.bank_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback source v2 root binding drifted")
    row_by_query = verified.row_by_query_id()
    sources: list[VerifiedStaticFeedbackSourceV2] = []
    for entry in selection.entries:
        row = row_by_query.get(entry.query_id)
        if row is None or not _entry_v2_matches_verified_row(entry, row):
            raise PortfolioS1FeedbackError(
                "Feedback source v2 differs from verified Static GCS"
            )
        packet = build_feedback_packet_v3(
            row.query,
            row.result,
            asset_catalog=catalog,
            rubric=control.rubric,
            gcs_diagnostics=FeedbackGCSDiagnosticsV1(
                answer_mode=row.score.answer_mode,
                gcs=row.score.gcs,
                components=FeedbackGCSComponentBitsV1(
                    route_acceptable=row.score.route_acceptable,
                    no_hard_error=row.score.no_hard_error,
                    tool_contract_pass=row.score.tool_contract_pass,
                    evidence_grounded=row.score.evidence_grounded,
                    output_contract_pass=row.score.output_contract_pass,
                ),
                reason_codes=row.score.reason_codes,
            ),
            gcs_contract=_feedback_gcs_contract_v1(verified, entry.capability),
        )
        sources.append(
            VerifiedStaticFeedbackSourceV2(
                corpus_sha256=verified.corpus_sha256,
                selection_sha256=selection.selection_sha256,
                control_sha256=control.control_sha256,
                selection_entry_sha256=entry.entry_sha256,
                corpus=verified,
                row=row,
                asset_catalog=catalog,
                packet=packet,
                _marker=_VERIFIED_STATIC_FEEDBACK_SOURCE_V2_MARKER,
            )
        )
    return tuple(sources)


def require_verified_static_feedback_source_v2(
    source: VerifiedStaticFeedbackSourceV2,
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    entry: FeedbackSelectionEntryV2,
) -> VerifiedStaticFeedbackSourceV2:
    if (
        type(source) is not VerifiedStaticFeedbackSourceV2
        or source._marker is not _VERIFIED_STATIC_FEEDBACK_SOURCE_V2_MARKER
        or source.corpus_sha256 != selection.corpus_sha256
        or source.selection_sha256 != selection.selection_sha256
        or source.control_sha256 != control.control_sha256
        or source.selection_entry_sha256 != entry.entry_sha256
        or not _entry_v2_matches_verified_row(entry, source.row)
        or source.packet.query_id != entry.query_id
        or source.packet.canonical_capability != entry.capability
        or source.packet.image.sha256 != entry.image_sha256
        or source.packet.gcs_diagnostics.answer_mode != entry.answer_mode
        or source.packet.gcs_diagnostics.reason_codes != entry.reason_codes
        or source.packet.gcs_contract
        != _feedback_gcs_contract_v1(source.corpus, entry.capability)
    ):
        raise PortfolioS1FeedbackError("verified Static Feedback source v2 drifted")
    verified, catalog = _accept_loaded_feedback_source_context_v2(source.corpus)
    if source.corpus is not verified or source.asset_catalog is not catalog:
        raise PortfolioS1FeedbackError("verified Static Feedback source v2 drifted")
    expected_packet = build_feedback_packet_v3(
        source.row.query,
        source.row.result,
        asset_catalog=catalog,
        rubric=control.rubric,
        gcs_diagnostics=source.packet.gcs_diagnostics,
        gcs_contract=source.packet.gcs_contract,
    )
    if source.packet != expected_packet:
        raise PortfolioS1FeedbackError("verified Static Feedback packet v3 drifted")
    return source


class FeedbackCallReservationV2(_StrictFrozenModel):
    """Create-only proof of the sole allowed call for one selected240 row."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-call-reservation"] = (
        "portfolio-s1-feedback-call-reservation"
    )
    policy_version: Literal["portfolio-s1-feedback-call-reservation-v4"] = (
        FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V4
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = "qwen3.7-plus-2026-05-26"
    cache_namespace: Literal["feedback-evaluator-v10"] = "feedback-evaluator-v10"
    max_tokens: None = None
    max_completion_tokens: Literal[4096] = 4096
    max_attempts: Literal[1] = 1
    reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Feedback call reservation v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_feedback_call_reservation_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    entry: FeedbackSelectionEntryV2,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
) -> FeedbackCallReservationV2:
    require_verified_static_feedback_source_v2(
        verified_source, selection, control, entry
    )
    row = verified_source.row
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-feedback-call-reservation",
        "policy_version": FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V4,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "selection_ordinal": entry.selection_ordinal,
        "query_id": entry.query_id,
        "packet_sha256": verified_source.packet.packet_sha256,
        "checkpoint_file_sha256": row.checkpoint_file_sha256,
        "checkpoint_row_sha256": row.checkpoint_row_sha256,
        "sidecar_sha256": row.sidecar.evidence_sha256,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "cache_namespace": "feedback-evaluator-v10",
        "max_tokens": None,
        "max_completion_tokens": 4096,
        "max_attempts": 1,
        "reservation_cny": QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY,
    }
    return FeedbackCallReservationV2.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class BoundFeedbackArtifactV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-bound-feedback"] = "portfolio-s1-bound-feedback"
    policy_version: Literal["portfolio-s1-bound-feedback-v2"] = (
        BOUND_FEEDBACK_POLICY_VERSION_V2
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    reservation_sha256: Sha256
    query_id: str
    feedback_packet: FeedbackPacketV3
    feedback_result: FeedbackEvaluationResult
    status: FeedbackStatus
    provider_call_count: Literal[1] = 1
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackEvaluationResult)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.status != self.status
            or self.feedback_result.cache_namespace != "feedback-evaluator-v10"
        ):
            raise ValueError("bound Feedback artifact v2 identity drifted")
        if self.artifact_sha256 != _model_hash(self, "artifact_sha256"):
            raise ValueError("bound Feedback artifact v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_bound_feedback_artifact_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    entry: FeedbackSelectionEntryV2,
    packet: FeedbackPacketV3,
    result: FeedbackEvaluationResult,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    reservation: FeedbackCallReservationV2,
) -> BoundFeedbackArtifactV2:
    require_verified_static_feedback_source_v2(
        verified_source, selection, control, entry
    )
    expected_reservation = build_feedback_call_reservation_v2(
        selection, control, entry, verified_source=verified_source
    )
    if (
        verified_source.packet != packet
        or reservation != expected_reservation
        or result.schema_version != 4
        or result.cache_namespace != "feedback-evaluator-v10"
        or result.parser_policy_version != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or result.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or result.prompt_policy_version != VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
        or result.prompt_policy_sha256 != VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
        or result.transport_policy_version != QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
        or result.transport_policy_sha256 != QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
        or result.requested_response_format != "json_schema"
        or result.requested_json_schema_sha256 != QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
        or result.requested_thinking is not True
        or result.requested_thinking_budget != 2048
        or result.requested_timeout_seconds != 600
        or result.requested_temperature is not None
        or result.requested_top_p is not None
        or result.packet_sha256 != packet.packet_sha256
        or result.image_sha256 != entry.image_sha256
        or result.remote_authorization_id != control.authorization_id
        or result.remote_authorization_file_sha256 != control.authorization_file_sha256
        or result.provider != "qwen"
        or result.model != "qwen3.7-plus-2026-05-26"
        or result.max_tokens is not None
        or result.max_completion_tokens != 4096
        or result.max_attempts != 1
    ):
        raise PortfolioS1FeedbackError("bound Feedback artifact v2 source drifted")
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-bound-feedback",
        "policy_version": BOUND_FEEDBACK_POLICY_VERSION_V2,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "query_id": entry.query_id,
        "feedback_packet": packet,
        "feedback_result": result,
        "status": result.status,
        "provider_call_count": 1,
    }
    return BoundFeedbackArtifactV2.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_feedback_call_reservation_v2(
    path: str | Path, reservation: FeedbackCallReservationV2
) -> Path:
    return atomic_create_file(path, reservation.canonical_bytes())


def load_feedback_call_reservation_v2(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_control_sha256: str,
    expected_entry_sha256: str,
) -> FeedbackCallReservationV2:
    content = read_stable_regular_file(
        path, label="Feedback call reservation v2", max_bytes=2 * 1024 * 1024
    )
    reservation = FeedbackCallReservationV2.model_validate_json(content, strict=True)
    if reservation.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v2 is not canonical JSON"
        )
    if (
        reservation.selection_sha256 != expected_selection_sha256
        or reservation.control_sha256 != expected_control_sha256
        or reservation.selection_entry_sha256 != expected_entry_sha256
    ):
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v2 resume identity conflict"
        )
    return reservation


def write_bound_feedback_artifact_v2(
    path: str | Path, artifact: BoundFeedbackArtifactV2
) -> Path:
    return atomic_create_file(path, artifact.canonical_bytes())


def load_bound_feedback_artifact_v2(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_selection_sha256: str | None = None,
    expected_control_sha256: str | None = None,
    expected_entry_sha256: str | None = None,
) -> BoundFeedbackArtifactV2:
    content = read_stable_regular_file(
        path, label="bound S1 Feedback artifact v2", max_bytes=8 * 1024 * 1024
    )
    if expected_file_sha256 is not None and sha256_bytes(content) != (
        expected_file_sha256
    ):
        raise PortfolioS1FeedbackError("bound Feedback artifact v2 SHA-256 mismatch")
    artifact = BoundFeedbackArtifactV2.model_validate_json(content, strict=True)
    if artifact.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("bound Feedback artifact v2 is not canonical")
    expected = (
        (expected_selection_sha256, artifact.selection_sha256),
        (expected_control_sha256, artifact.control_sha256),
        (expected_entry_sha256, artifact.selection_entry_sha256),
    )
    if any(wanted is not None and wanted != actual for wanted, actual in expected):
        raise PortfolioS1FeedbackError(
            "bound Feedback artifact v2 resume identity conflict"
        )
    return artifact


def resume_bound_feedback_artifact_v2(
    artifact: BoundFeedbackArtifactV2,
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    entry: FeedbackSelectionEntryV2,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    reservation: FeedbackCallReservationV2,
) -> BoundFeedbackArtifactV2:
    expected = build_bound_feedback_artifact_v2(
        selection,
        control,
        entry,
        verified_source.packet,
        artifact.feedback_result,
        verified_source=verified_source,
        reservation=reservation,
    )
    if artifact != expected:
        raise PortfolioS1FeedbackError("bound Feedback v2 resume source conflict")
    return artifact


SuggestionDispositionV1 = Literal[
    "policy_compatible", "requires_new_evidence", "rejected"
]
_SUGGESTION_LABEL_RE = re.compile(
    r"^\[(policy_compatible|requires_new_evidence|rejected)\] (?P<text>\S(?:.*\S)?)$"
)


class PolicyLabeledFeedbackSuggestionV1(_StrictFrozenModel):
    disposition: SuggestionDispositionV1
    text: str

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        return _nonblank(value, "suggestion text")


def parse_policy_labeled_feedback_suggestion_v1(
    value: str,
) -> PolicyLabeledFeedbackSuggestionV1:
    if not isinstance(value, str):
        raise PortfolioS1FeedbackError("Feedback suggestion is not text")
    match = _SUGGESTION_LABEL_RE.fullmatch(value)
    if match is None:
        raise PortfolioS1FeedbackError(
            "Feedback suggestion lacks one exact policy disposition"
        )
    return PolicyLabeledFeedbackSuggestionV1(
        disposition=match.group(1),  # type: ignore[arg-type]
        text=match.group("text"),
    )


class PortfolioS1FeedbackCoverageCellV5(_StrictFrozenModel):
    capability: str
    role: FeedbackRole
    selected_count: int = Field(ge=0, le=240)
    parsed_count: int = Field(ge=0, le=240)


class FeedbackGCSContractProjectionV5(_StrictFrozenModel):
    capability: str
    required_sections: tuple[str, ...]
    fallback_markers: tuple[str, ...]
    preferred_fallback_marker: str
    card_requirement: Literal["required", "forbidden"]
    legal_tool_sequences: tuple[tuple[str, ...], ...]
    detector_label_policy: Literal["prediction_only_never_verified_identity"] = (
        "prediction_only_never_verified_identity"
    )
    evidence_policy: Literal["material_facts_require_visible_tool_evidence"] = (
        "material_facts_require_visible_tool_evidence"
    )

    @field_validator(
        "required_sections", "fallback_markers", "legal_tool_sequences", mode="before"
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        if value and isinstance(value[0], list):
            return tuple(tuple(item) for item in value)
        return tuple(value)


class PortfolioS1FeedbackModelEntryV5(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    capability: str
    role: FeedbackRole
    primary_cluster: str
    answer_mode: Literal["supported", "fallback", "unresolved"]
    gcs: Literal[0, 1]
    gcs_components: FeedbackGCSComponentBitsV1
    reason_codes: tuple[str, ...]
    diagnostic_feedback: VisualFeedbackOutput
    labeled_suggestions: tuple[PolicyLabeledFeedbackSuggestionV1, ...]
    actionable_suggestions: tuple[str, ...]

    @field_validator(
        "reason_codes", "labeled_suggestions", "actionable_suggestions", mode="before"
    )
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("diagnostic_feedback", mode="before")
    @classmethod
    def _feedback(cls, value: object) -> object:
        return _nested_json_model(value, VisualFeedbackOutput)

    @model_validator(mode="after")
    def _validate_entry(self) -> Self:
        if self.diagnostic_feedback.skill_suggestions:
            raise ValueError("Creator diagnostic Feedback must remove raw suggestions")
        expected = tuple(
            item.text
            for item in self.labeled_suggestions
            if item.disposition == "policy_compatible"
        )
        if self.actionable_suggestions != expected:
            raise ValueError("Creator actionable suggestions were not policy-filtered")
        return self


class PortfolioS1FeedbackActionableSuggestionV5(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    capability: str
    text: str

    @field_validator("capability", "text")
    @classmethod
    def _text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class PortfolioS1FeedbackGCSContractSetV5(_StrictFrozenModel):
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        "portfolio-grounded-contract-success-v2"
    )
    capability_contracts: tuple[FeedbackGCSContractProjectionV5, ...]
    suggestion_policy: Literal["only_policy_compatible_is_actionable"] = (
        "only_policy_compatible_is_actionable"
    )

    @field_validator("capability_contracts", mode="before")
    @classmethod
    def _contracts(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_contracts(self) -> Self:
        if tuple(item.capability for item in self.capability_contracts) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("Feedback GCS contract set order drifted")
        return self


class PortfolioS1FeedbackRepresentativeExampleV5(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    capability: str
    role: FeedbackRole
    primary_cluster: str
    gcs_diagnostics: FeedbackGCSDiagnosticsV1
    turns: tuple[ConversationTurn, ...]
    response_text: str
    cards: tuple[VisibleCard, ...]
    tool_evidence: tuple[VisibleToolEvidence, ...]

    @field_validator("turns", mode="before")
    @classmethod
    def _turns(cls, value: object) -> object:
        return _nested_json_models(value, ConversationTurn)

    @field_validator("cards", mode="before")
    @classmethod
    def _cards(cls, value: object) -> object:
        return _nested_json_models(value, VisibleCard)

    @field_validator("tool_evidence", mode="before")
    @classmethod
    def _evidence(cls, value: object) -> object:
        return _nested_json_models(value, VisibleToolEvidence)


class PortfolioS1FeedbackModelProjectionV5(_StrictFrozenModel):
    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-feedback-model-projection"] = (
        "portfolio-s1-feedback-model-projection"
    )
    status: Literal["complete_policy_filtered_feedback"] = (
        "complete_policy_filtered_feedback"
    )
    selected_count: Literal[240] = 240
    parsed_count: Literal[240] = 240
    coverage: tuple[PortfolioS1FeedbackCoverageCellV5, ...]
    gcs_contract: PortfolioS1FeedbackGCSContractSetV5
    feedback_entries: tuple[PortfolioS1FeedbackModelEntryV5, ...]
    representative_examples: tuple[PortfolioS1FeedbackRepresentativeExampleV5, ...]
    actionable_suggestions: tuple[PortfolioS1FeedbackActionableSuggestionV5, ...]
    actionable_suggestion_count: int = Field(ge=0)

    @field_validator(
        "coverage",
        "feedback_entries",
        "representative_examples",
        "actionable_suggestions",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_projection(self) -> Self:
        if (
            len(self.feedback_entries) != 240
            or tuple(item.selection_ordinal for item in self.feedback_entries)
            != tuple(range(1, 241))
            or len(self.representative_examples) != 12
            or tuple(item.selection_ordinal for item in self.representative_examples)
            != tuple(range(1, 13))
            or self.actionable_suggestion_count != len(self.actionable_suggestions)
        ):
            raise ValueError("Feedback model projection v5 drifted")
        expected_actionable = tuple(
            PortfolioS1FeedbackActionableSuggestionV5(
                selection_ordinal=item.selection_ordinal,
                capability=item.capability,
                text=text,
            )
            for item in self.feedback_entries
            for text in item.actionable_suggestions
        )
        if self.actionable_suggestions != expected_actionable:
            raise ValueError("Feedback projection v5 actionable list drifted")
        return self


class PortfolioS1FeedbackBundleEntryV5(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    capability: str
    role: FeedbackRole
    primary_cluster: str
    atomic_component_id: str
    packet_sha256: Sha256
    bound_artifact_sha256: Sha256
    feedback_result_sha256: Sha256


class PortfolioS1FeedbackBundleV5(_StrictFrozenModel):
    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-feedback-bundle"] = "portfolio-s1-feedback-bundle"
    policy_version: Literal["portfolio-s1-feedback-bundle-v5"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V5
    )
    status: Literal["complete_policy_filtered_feedback"] = (
        "complete_policy_filtered_feedback"
    )
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    run_sha256: Sha256
    run_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    selected_count: Literal[240] = 240
    attempted_count: Literal[240] = 240
    parsed_count: Literal[240] = 240
    provider_call_count: Literal[240] = 240
    selected_query_ids: tuple[str, ...]
    entries: tuple[PortfolioS1FeedbackBundleEntryV5, ...]
    bound_artifact_set_sha256: Sha256
    model_projection: PortfolioS1FeedbackModelProjectionV5
    bundle_sha256: Sha256

    @field_validator("selected_query_ids", "entries", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        if (
            len(self.entries) != 240
            or tuple(item.selection_ordinal for item in self.entries)
            != tuple(range(1, 241))
            or tuple(item.query_id for item in self.entries) != self.selected_query_ids
            or len(set(self.selected_query_ids)) != 240
            or tuple(
                item.selection_ordinal
                for item in self.model_projection.feedback_entries
            )
            != tuple(range(1, 241))
        ):
            raise ValueError("Feedback bundle v5 parsed240 identities drifted")
        expected_artifact_set = _hash_payload(
            [
                {
                    "selection_ordinal": item.selection_ordinal,
                    "bound_artifact_sha256": item.bound_artifact_sha256,
                    "feedback_result_sha256": item.feedback_result_sha256,
                }
                for item in self.entries
            ]
        )
        if self.bound_artifact_set_sha256 != expected_artifact_set:
            raise ValueError("Feedback bundle v5 artifact-set hash drifted")
        try:
            _validate_model_projection_privacy(
                self.model_projection, private_query_ids=self.selected_query_ids
            )
        except PortfolioS1FeedbackError as error:
            raise ValueError(str(error)) from error
        if self.bundle_sha256 != _model_hash(self, "bundle_sha256"):
            raise ValueError("Feedback bundle v5 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_projection_payload(self) -> dict[str, object]:
        _validate_model_projection_privacy(
            self.model_projection, private_query_ids=self.selected_query_ids
        )
        projection = self.model_projection
        component_names = (
            "route_acceptable",
            "no_hard_error",
            "tool_contract_pass",
            "evidence_grounded",
            "output_contract_pass",
        )
        capability_aggregates: list[dict[str, object]] = []
        for capability in GCS_CAPABILITY_ORDER:
            entries = tuple(
                item
                for item in projection.feedback_entries
                if item.capability == capability
            )
            answer_modes = Counter(item.answer_mode for item in entries)
            roles = Counter(item.role for item in entries)
            clusters = Counter(item.primary_cluster for item in entries)
            reasons = Counter(
                reason for item in entries for reason in item.reason_codes
            )
            non_actionable_count = sum(
                suggestion.disposition != "policy_compatible"
                for item in entries
                for suggestion in item.labeled_suggestions
            )
            capability_aggregates.append(
                {
                    "capability": capability,
                    "selected_count": len(entries),
                    "gcs_success_count": sum(item.gcs for item in entries),
                    "gcs_failure_count": sum(1 - item.gcs for item in entries),
                    "answer_mode_counts": [
                        {"answer_mode": mode, "count": answer_modes[mode]}
                        for mode in ("supported", "fallback", "unresolved")
                    ],
                    "role_counts": [
                        {"role": role, "count": roles[role]}
                        for role in _FEEDBACK_ROLE_ORDER
                    ],
                    "primary_cluster_counts": [
                        {"primary_cluster": cluster, "count": count}
                        for cluster, count in sorted(clusters.items())
                    ],
                    "reason_code_counts": [
                        {"reason_code": reason, "count": count}
                        for reason, count in sorted(reasons.items())
                    ],
                    "gcs_component_failure_counts": {
                        name: sum(
                            getattr(item.gcs_components, name) == 0 for item in entries
                        )
                        for name in component_names
                    },
                    "diagnostic_counts": {
                        "entries_with_rule_violations": sum(
                            bool(item.diagnostic_feedback.rule_violations)
                            for item in entries
                        ),
                        "rule_violation_count": sum(
                            len(item.diagnostic_feedback.rule_violations)
                            for item in entries
                        ),
                        "entries_with_ideal_response_gaps": sum(
                            bool(item.diagnostic_feedback.ideal_response_gaps)
                            for item in entries
                        ),
                        "ideal_response_gap_count": sum(
                            len(item.diagnostic_feedback.ideal_response_gaps)
                            for item in entries
                        ),
                        "policy_compatible_actionable_suggestion_count": sum(
                            len(item.actionable_suggestions) for item in entries
                        ),
                        "filtered_non_actionable_suggestion_count": (
                            non_actionable_count
                        ),
                    },
                }
            )

        entry_by_ordinal = {
            item.selection_ordinal: item for item in projection.feedback_entries
        }
        representative_examples: list[dict[str, object]] = []
        for example in projection.representative_examples:
            entry = entry_by_ordinal[example.selection_ordinal]
            diagnostic = entry.diagnostic_feedback.model_dump(
                mode="json", exclude={"skill_suggestions"}
            )
            representative_examples.append(
                {
                    **example.model_dump(mode="json"),
                    "structured_feedback": {
                        **diagnostic,
                        "policy_compatible_actionable_suggestions": list(
                            entry.actionable_suggestions
                        ),
                    },
                }
            )

        payload: dict[str, object] = {
            "schema_version": projection.schema_version,
            "kind": projection.kind,
            "status": projection.status,
            "selected_count": projection.selected_count,
            "parsed_count": projection.parsed_count,
            "coverage": [item.model_dump(mode="json") for item in projection.coverage],
            "gcs_contract": projection.gcs_contract.model_dump(mode="json"),
            "aggregate_diagnostics": {
                "policy_version": "portfolio-s1-feedback-aggregate-diagnostics-v1",
                "selected_count": projection.selected_count,
                "capabilities": capability_aggregates,
            },
            "representative_examples": representative_examples,
            "actionable_suggestions": [
                item.model_dump(mode="json")
                for item in projection.actionable_suggestions
            ],
            "actionable_suggestion_count": projection.actionable_suggestion_count,
        }
        _validate_model_projection_privacy(
            payload, private_query_ids=self.selected_query_ids
        )
        return payload


def _feedback_v5_coverage(
    selection: PortfolioS1FeedbackSelectionV2,
) -> tuple[PortfolioS1FeedbackCoverageCellV5, ...]:
    cells = []
    for capability in GCS_CAPABILITY_ORDER:
        for role in _FEEDBACK_ROLE_ORDER:
            count = sum(
                item.capability == capability and item.role == role
                for item in selection.entries
            )
            cells.append(
                PortfolioS1FeedbackCoverageCellV5(
                    capability=capability,
                    role=role,
                    selected_count=count,
                    parsed_count=count,
                )
            )
    return tuple(cells)


def build_portfolio_s1_feedback_bundle_v5(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV9,
    authorization: PortfolioS1FeedbackAuthorizationV4,
    artifacts: tuple[object, ...],
    run: PortfolioS1FeedbackRunV2,
) -> PortfolioS1FeedbackBundleV5:
    validate_portfolio_s1_feedback_control_v9(control, selection, authorization)
    artifact_by_entry = {
        getattr(item, "selection_entry_sha256", None): item for item in artifacts
    }
    if (
        len(artifacts) != 240
        or len(artifact_by_entry) != 240
        or None in artifact_by_entry
    ):
        raise PortfolioS1FeedbackError("Feedback bundle v5 requires exact240 artifacts")
    expected_run = build_portfolio_s1_feedback_run_v2(selection, control, artifacts)
    if run != expected_run or run.status != "completed":
        raise PortfolioS1FeedbackError(
            "Feedback bundle v5 requires completed parsed240"
        )
    private_entries: list[PortfolioS1FeedbackBundleEntryV5] = []
    model_entries: list[PortfolioS1FeedbackModelEntryV5] = []
    representatives: list[PortfolioS1FeedbackRepresentativeExampleV5] = []
    contracts: dict[str, FeedbackGCSContractProjectionV5] = {}
    artifact_hashes: list[dict[str, object]] = []
    selected_query_ids = tuple(item.query_id for item in selection.entries)
    for selected in selection.entries:
        artifact = artifact_by_entry[selected.entry_sha256]
        packet = getattr(artifact, "feedback_packet", None)
        result = getattr(artifact, "feedback_result", None)
        feedback = None if result is None else getattr(result, "parsed_feedback", None)
        if (
            type(packet) is not FeedbackPacketV3
            or type(feedback) is not VisualFeedbackOutput
            or getattr(artifact, "status", None) != "parsed"
            or getattr(artifact, "selection_sha256", None) != selection.selection_sha256
            or getattr(artifact, "control_sha256", None) != control.control_sha256
            or getattr(artifact, "query_id", None) != selected.query_id
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError(
                "Feedback bundle v5 artifact binding drifted"
            )
        require_feedback_v10_creator_projection_privacy(
            result,
            private_query_ids=selected_query_ids,
        )
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("Feedback bundle v5 suggestions repeat")
        diagnostic_feedback = VisualFeedbackOutput(
            schema_version=1,
            summary=feedback.summary,
            rule_violations=feedback.rule_violations,
            ideal_response_gaps=feedback.ideal_response_gaps,
            skill_suggestions=(),
        )
        components = packet.gcs_diagnostics.components
        model_entries.append(
            PortfolioS1FeedbackModelEntryV5(
                selection_ordinal=selected.selection_ordinal,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                answer_mode=selected.answer_mode,
                gcs=selected.gcs,
                gcs_components=components,
                reason_codes=selected.reason_codes,
                diagnostic_feedback=diagnostic_feedback,
                labeled_suggestions=labeled,
                actionable_suggestions=tuple(
                    item.text
                    for item in labeled
                    if item.disposition == "policy_compatible"
                ),
            )
        )
        contract = packet.gcs_contract
        projected_contract = FeedbackGCSContractProjectionV5(
            capability=selected.capability,
            required_sections=contract.required_sections,
            fallback_markers=contract.fallback_markers,
            preferred_fallback_marker=contract.preferred_fallback_marker,
            card_requirement=contract.card_requirement,
            legal_tool_sequences=contract.legal_tool_sequences,
        )
        previous = contracts.setdefault(selected.capability, projected_contract)
        if previous != projected_contract:
            raise PortfolioS1FeedbackError("Feedback bundle v5 GCS contract drifted")
        artifact_sha = getattr(artifact, "artifact_sha256")
        result_sha = getattr(result, "result_sha256")
        private_entries.append(
            PortfolioS1FeedbackBundleEntryV5(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                atomic_component_id=selected.atomic_component_id,
                packet_sha256=packet.packet_sha256,
                bound_artifact_sha256=artifact_sha,
                feedback_result_sha256=result_sha,
            )
        )
        artifact_hashes.append(
            {
                "selection_ordinal": selected.selection_ordinal,
                "bound_artifact_sha256": artifact_sha,
                "feedback_result_sha256": result_sha,
            }
        )
        if selected.selection_ordinal <= 12:
            representatives.append(
                PortfolioS1FeedbackRepresentativeExampleV5(
                    selection_ordinal=selected.selection_ordinal,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    gcs_diagnostics=packet.gcs_diagnostics,
                    turns=packet.turns,
                    response_text=packet.response_text,
                    cards=packet.cards,
                    tool_evidence=packet.tool_evidence,
                )
            )
    projection = PortfolioS1FeedbackModelProjectionV5(
        coverage=_feedback_v5_coverage(selection),
        gcs_contract=PortfolioS1FeedbackGCSContractSetV5(
            capability_contracts=tuple(contracts[item] for item in GCS_CAPABILITY_ORDER)
        ),
        feedback_entries=tuple(model_entries),
        representative_examples=tuple(representatives),
        actionable_suggestions=tuple(
            PortfolioS1FeedbackActionableSuggestionV5(
                selection_ordinal=item.selection_ordinal,
                capability=item.capability,
                text=text,
            )
            for item in model_entries
            for text in item.actionable_suggestions
        ),
        actionable_suggestion_count=sum(
            len(item.actionable_suggestions) for item in model_entries
        ),
    )
    _validate_model_projection_privacy(projection, private_query_ids=selected_query_ids)
    unsigned = {
        "schema_version": 5,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V5,
        "status": "complete_policy_filtered_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 240,
        "attempted_count": 240,
        "parsed_count": 240,
        "provider_call_count": 240,
        "selected_query_ids": selected_query_ids,
        "entries": tuple(private_entries),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV5.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def write_portfolio_s1_feedback_bundle_v5(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV5
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v5(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV5:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle v5", max_bytes=128 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback bundle v5 file SHA-256 mismatch")
    bundle = PortfolioS1FeedbackBundleV5.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback bundle v5 is not canonical JSON")
    return bundle


def _load_forward_canonical_model(
    path: str | Path,
    *,
    expected_file_sha256: str,
    label: str,
    model_type: type[BaseModel],
    max_bytes: int = 32 * 1024 * 1024,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError(f"{label} file SHA-256 mismatch")
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical JSON")
    return model


def write_selected_qwen_feedback_authorization_v4(
    path: str | Path, authorization: PortfolioS1FeedbackAuthorizationV4
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def load_selected_qwen_feedback_authorization_v4(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackAuthorizationV4:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen Feedback authorization v4",
        model_type=PortfolioS1FeedbackAuthorizationV4,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_control_v9(
    path: str | Path, control: PortfolioS1FeedbackControlV9
) -> Path:
    return atomic_create_file(path, control.canonical_bytes())


def load_portfolio_s1_feedback_control_v9(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackControlV9:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback control v9",
        model_type=PortfolioS1FeedbackControlV9,
    )  # type: ignore[return-value]


def write_qwen37_feedback_launch_lock_v2(
    path: str | Path, launch: PortfolioS1QwenFeedbackLaunchLockV2
) -> Path:
    return atomic_create_file(path, launch.canonical_bytes())


def load_qwen37_feedback_launch_lock_v2(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1QwenFeedbackLaunchLockV2:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen Feedback launch lock v2",
        model_type=PortfolioS1QwenFeedbackLaunchLockV2,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_run_v2(
    path: str | Path, run: PortfolioS1FeedbackRunV2
) -> Path:
    return atomic_create_file(path, run.canonical_bytes())


def load_portfolio_s1_feedback_run_v2(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackRunV2:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback run v2",
        model_type=PortfolioS1FeedbackRunV2,
    )  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Forward-only Qwen3.8-Max Discovery240 identities.  The V4/V9/V2/V5 types
# above remain the byte-exact Qwen3.7 history and are intentionally not reused
# for new provider attempts.


class PortfolioS1FeedbackAuthorizationV5(_StrictFrozenModel):
    """Explicit owner authority for the Qwen3.8-Max selected240 run.

    Merely having a technical CNY94 ceiling in code is not authorization.  A
    production caller must provide the exact approved cap and an attributable
    owner statement; the CLI refuses to build this artifact when either is
    absent.
    """

    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v5"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION_V5
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-discovery-selected-240-assets"] = (
        "core-opt800-s1-feedback-discovery-selected-240-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    owner_approved_phase_hard_cap_cny: Literal["94.000000000000"]
    selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen38-feedback"] = QWEN38_FEEDBACK_PROCESSOR
    cache_namespace: Literal["feedback-evaluator-v11"] = QWEN38_FEEDBACK_CACHE_NAMESPACE
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v6"
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256] = (
        QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256
    )
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN38_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN38_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS
    selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    provider_call_ceiling: Literal[240] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    feedback_concurrency: Literal[2] = 2
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.387816000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["93.075840000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["94.000000000000"] = (
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
    )
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    parent_membership_processor: Literal["dashscope-qwen-assistant"] = (
        "dashscope-qwen-assistant"
    )
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    parent_remote_catalog_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    selected_assets: tuple[SelectedQwenFeedbackAssetV1, ...]
    selected_asset_set_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator("selected_assets", "phase_counts", mode="before")
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "authorization_id",
        "reviewer_id",
        "owner_statement",
        "parent_remote_authorization_id",
    )
    @classmethod
    def _texts(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if (
            info.field_name == "authorization_id"
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None
        ):
            raise ValueError("authorization_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen3.8 Feedback reviewed_at needs a timezone")
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback phases drifted")
        if self.owner_approved_phase_hard_cap_cny != self.phase_hard_cap_cny:
            raise ValueError("Qwen3.8 Feedback owner budget approval drifted")
        if (
            len(self.selected_assets) != 240
            or tuple(item.query_id for item in self.selected_assets)
            != tuple(sorted(item.query_id for item in self.selected_assets))
            or len({item.query_id for item in self.selected_assets}) != 240
            or len({item.asset_id for item in self.selected_assets}) != 240
        ):
            raise ValueError("Qwen3.8 Feedback must bind query-sorted unique240")
        asset_payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != _hash_payload(asset_payload):
            raise ValueError("Qwen3.8 Feedback asset-set hash drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN38_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V3,
            QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V3,
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen3.8 Feedback source/pricing identity drifted")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Qwen3.8 Feedback authorization self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_selected_qwen38_feedback_authorization_v5(
    authorization: PortfolioS1FeedbackAuthorizationV5,
    selection: PortfolioS1FeedbackSelectionV2,
) -> None:
    assets = _selected_qwen_feedback_assets_v4(selection)
    if (
        type(authorization) is not PortfolioS1FeedbackAuthorizationV5
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
        or authorization.selected_asset_set_sha256
        != _hash_payload([item.model_dump(mode="json") for item in assets])
    ):
        raise PortfolioS1FeedbackError(
            "Qwen3.8 Feedback authorization v5 differs from selected240"
        )


def build_selected_qwen38_feedback_authorization_v5(
    selection: PortfolioS1FeedbackSelectionV2,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
    owner_approved_phase_hard_cap_cny: str,
    model_source_lock_file_sha256: str,
    model_source_lock_sha256: str,
    pricing_lock_file_sha256: str,
    pricing_lock_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV5:
    if (
        owner_approved_phase_hard_cap_cny
        != QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
    ):
        raise PortfolioS1FeedbackError(
            "Qwen3.8 Feedback requires explicit owner approval of the CNY94 cap"
        )
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selected_qwen_feedback_assets_v4(selection)
    unsigned = {
        "schema_version": 5,
        "kind": "portfolio-s1-feedback-selected-asset-authorization",
        "policy_version": FEEDBACK_AUTHORIZATION_POLICY_VERSION_V5,
        "authorization_id": authorization_id,
        "status": "owner-approved",
        "scope": "core-opt800-s1-feedback-discovery-selected-240-assets",
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "owner_statement": owner_statement,
        "owner_approved_phase_hard_cap_cny": owner_approved_phase_hard_cap_cny,
        "selection_sha256": selection.selection_sha256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "requested_response_format": "json_schema",
        "requested_json_schema_policy_version": QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION,
        "requested_json_schema_name": QWEN38_FEEDBACK_JSON_SCHEMA_NAME,
        "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
        "transport_policy_version": QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_thinking": True,
        "requested_thinking_budget": QWEN38_FEEDBACK_THINKING_BUDGET,
        "requested_timeout_seconds": QWEN38_FEEDBACK_TIMEOUT_SECONDS,
        "requested_temperature": None,
        "requested_top_p": None,
        "requested_max_tokens": None,
        "max_completion_tokens": QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS,
        "selected_query_count": 240,
        "provider_call_ceiling": 240,
        "max_attempts": 1,
        "retry_policy": "no_retry",
        "feedback_concurrency": 2,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "input_token_reservation_ceiling_per_call": QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS,
        "output_token_reservation_ceiling_per_call": QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS,
        "per_call_reservation_cny": QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY,
        "maximum_reservation_cny": QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY,
        "phase_hard_cap_cny": QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY,
        "over_budget_policy": "fail_closed_before_provider_call",
        "parent_membership_processor": "dashscope-qwen-assistant",
        "parent_remote_authorization_id": parent.authorization.authorization_id,
        "parent_remote_authorization_file_sha256": parent.authorization_file_sha256,
        "parent_remote_receipt_file_sha256": parent.receipt_file_sha256,
        "parent_remote_receipt_sha256": parent.receipt.receipt_sha256,
        "parent_remote_catalog_sha256": parent.catalog.catalog_sha256,
        "model_source_lock_file_sha256": model_source_lock_file_sha256,
        "model_source_lock_sha256": model_source_lock_sha256,
        "pricing_lock_file_sha256": pricing_lock_file_sha256,
        "pricing_lock_sha256": pricing_lock_sha256,
        "role_selection_file_sha256": role_selection_file_sha256,
        "role_selection_sha256": role_selection_sha256,
        "cloud_upload_allowed": True,
        "remote_model_inference_allowed": True,
        "redistribution_allowed": False,
        "public_demo_allowed": False,
        "selected_assets": assets,
        "selected_asset_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in assets]
        ),
    }
    return PortfolioS1FeedbackAuthorizationV5.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1FeedbackControlV10(_StrictFrozenModel):
    schema_version: Literal[10] = 10
    kind: Literal["portfolio-s1-feedback-control"] = "portfolio-s1-feedback-control"
    policy_version: Literal["portfolio-s1-feedback-control-v10"] = (
        FEEDBACK_CONTROL_POLICY_VERSION_V10
    )
    selection_sha256: Sha256
    authorization_sha256: Sha256
    authorization_id: str
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    parent_static_bank_sha256: Sha256
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    parser_policy_version: Literal["visual-feedback-free-text-trim-v3"] = (
        VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
    )
    parser_policy_sha256: Literal[VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3] = (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
    )
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v6"
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256] = (
        QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256
    )
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen38-feedback"] = QWEN38_FEEDBACK_PROCESSOR
    cache_namespace: Literal["feedback-evaluator-v11"] = QWEN38_FEEDBACK_CACHE_NAMESPACE
    selected_query_count: Literal[240] = 240
    provider_call_ceiling: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    phase_entry_sha256s: tuple[tuple[Sha256, ...], ...]
    phase_query_ids: tuple[tuple[str, ...], ...]
    rubric: RubricSnapshot
    feedback_concurrency: Literal[2] = 2
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    max_completion_tokens: Literal[4096] = 4096
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = 2048
    requested_timeout_seconds: Literal[600] = 600
    require_all_parsed_for_bundle: Literal[True] = True
    stop_on_nonparsed_or_orphan: Literal[True] = True
    control_sha256: Sha256

    @field_validator("phase_counts", mode="before")
    @classmethod
    def _phase_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("phase_entry_sha256s", "phase_query_ids", mode="before")
    @classmethod
    def _phase_nested(cls, value: object) -> object:
        return (
            tuple(tuple(item) if isinstance(item, list) else item for item in value)
            if isinstance(value, list)
            else value
        )

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Feedback control v10 phases drifted")
        if self.control_sha256 != _model_hash(self, "control_sha256"):
            raise ValueError("Feedback control v10 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_portfolio_s1_feedback_control_v10(
    control: PortfolioS1FeedbackControlV10,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV5,
) -> None:
    validate_selected_qwen38_feedback_authorization_v5(authorization, selection)
    if (
        type(control) is not PortfolioS1FeedbackControlV10
        or control.selection_sha256 != selection.selection_sha256
        or control.authorization_sha256 != authorization.authorization_sha256
        or control.authorization_id != authorization.authorization_id
        or control.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or control.corpus_sha256 != selection.corpus_sha256
        or control.parent_static_bank_sha256 != selection.parent_static_bank_sha256
        or control.phase_entry_sha256s != selection.phase_entry_sha256s
        or control.phase_query_ids != selection.phase_query_ids
    ):
        raise PortfolioS1FeedbackError("Feedback control v10 binding drifted")


def build_portfolio_s1_feedback_control_v10(
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV5,
    *,
    rubric: RubricSnapshot,
) -> PortfolioS1FeedbackControlV10:
    validate_selected_qwen38_feedback_authorization_v5(authorization, selection)
    unsigned = {
        "schema_version": 10,
        "kind": "portfolio-s1-feedback-control",
        "policy_version": FEEDBACK_CONTROL_POLICY_VERSION_V10,
        "selection_sha256": selection.selection_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_id": authorization.authorization_id,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "selected_query_count": 240,
        "provider_call_ceiling": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "phase_entry_sha256s": selection.phase_entry_sha256s,
        "phase_query_ids": selection.phase_query_ids,
        "rubric": rubric,
        "feedback_concurrency": 2,
        "max_attempts": 1,
        "retry_policy": "no_retry",
        "max_completion_tokens": 4096,
        "requested_thinking": True,
        "requested_thinking_budget": 2048,
        "requested_timeout_seconds": 600,
        "require_all_parsed_for_bundle": True,
        "stop_on_nonparsed_or_orphan": True,
    }
    return PortfolioS1FeedbackControlV10.model_validate(
        {**unsigned, "control_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


class PortfolioS1Qwen38FeedbackLaunchLockV3(_StrictFrozenModel):
    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-qwen38-feedback-launch-lock"] = (
        "portfolio-s1-qwen38-feedback-launch-lock"
    )
    policy_version: Literal["portfolio-s1-qwen38-feedback-launch-lock-v3"] = (
        QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V3
    )
    run_id: str
    status: Literal["prepared-no-provider-calls"] = "prepared-no-provider-calls"
    selection_sha256: Sha256
    selection_file_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    corpus_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen38-feedback"] = QWEN38_FEEDBACK_PROCESSOR
    cache_namespace: Literal["feedback-evaluator-v11"] = QWEN38_FEEDBACK_CACHE_NAMESPACE
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6] = (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    )
    provider_call_ceiling: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    feedback_concurrency: Literal[2] = 2
    max_attempts_per_selected_query: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    per_call_reservation_cny: Literal["0.387816000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["93.075840000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    owner_approved_phase_hard_cap_cny: Literal["94.000000000000"]
    phase_hard_cap_cny: Literal["94.000000000000"] = (
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
    )
    provider_calls_performed: Literal[0] = 0
    launch_lock_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        value = _nonblank(value, "run_id")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None:
            raise ValueError("run_id must be canonical")
        return value

    @field_validator("phase_counts", mode="before")
    @classmethod
    def _phases(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback launch phases drifted")
        if self.owner_approved_phase_hard_cap_cny != self.phase_hard_cap_cny:
            raise ValueError("Qwen3.8 Feedback launch lacks owner budget approval")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN38_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V3,
            QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V3,
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen3.8 Feedback launch source/pricing drifted")
        if self.launch_lock_sha256 != _model_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback launch self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_qwen38_feedback_launch_lock_v3(
    *,
    run_id: str,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV5,
    control: PortfolioS1FeedbackControlV10,
) -> PortfolioS1Qwen38FeedbackLaunchLockV3:
    validate_portfolio_s1_feedback_control_v10(control, selection, authorization)
    unsigned = {
        "schema_version": 3,
        "kind": "portfolio-s1-qwen38-feedback-launch-lock",
        "policy_version": QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V3,
        "run_id": run_id,
        "status": "prepared-no-provider-calls",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "model_source_lock_file_sha256": authorization.model_source_lock_file_sha256,
        "model_source_lock_sha256": authorization.model_source_lock_sha256,
        "pricing_lock_file_sha256": authorization.pricing_lock_file_sha256,
        "pricing_lock_sha256": authorization.pricing_lock_sha256,
        "role_selection_file_sha256": authorization.role_selection_file_sha256,
        "role_selection_sha256": authorization.role_selection_sha256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "provider_call_ceiling": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "feedback_concurrency": 2,
        "max_attempts_per_selected_query": 1,
        "retry_policy": "no_retry",
        "per_call_reservation_cny": QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY,
        "maximum_reservation_cny": QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY,
        "owner_approved_phase_hard_cap_cny": (
            authorization.owner_approved_phase_hard_cap_cny
        ),
        "phase_hard_cap_cny": QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY,
        "provider_calls_performed": 0,
    }
    return PortfolioS1Qwen38FeedbackLaunchLockV3.model_validate(
        {**unsigned, "launch_lock_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class FeedbackCallReservationV3(_StrictFrozenModel):
    """Create-only proof of one Qwen3.8-Max selected240 provider attempt."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-call-reservation"] = (
        "portfolio-s1-feedback-call-reservation"
    )
    policy_version: Literal["portfolio-s1-feedback-call-reservation-v5"] = (
        FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V5
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    cache_namespace: Literal["feedback-evaluator-v11"] = QWEN38_FEEDBACK_CACHE_NAMESPACE
    max_tokens: None = None
    max_completion_tokens: Literal[4096] = 4096
    max_attempts: Literal[1] = 1
    reservation_cny: Literal["0.387816000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Feedback call reservation v3 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_feedback_call_reservation_v3(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV10,
    entry: FeedbackSelectionEntryV2,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
) -> FeedbackCallReservationV3:
    require_verified_static_feedback_source_v2(
        verified_source, selection, control, entry
    )
    row = verified_source.row
    unsigned = {
        "schema_version": 3,
        "kind": "portfolio-s1-feedback-call-reservation",
        "policy_version": FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V5,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "selection_ordinal": entry.selection_ordinal,
        "query_id": entry.query_id,
        "packet_sha256": verified_source.packet.packet_sha256,
        "checkpoint_file_sha256": row.checkpoint_file_sha256,
        "checkpoint_row_sha256": row.checkpoint_row_sha256,
        "sidecar_sha256": row.sidecar.evidence_sha256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "max_tokens": None,
        "max_completion_tokens": 4096,
        "max_attempts": 1,
        "reservation_cny": QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY,
    }
    return FeedbackCallReservationV3.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class BoundFeedbackArtifactV3(_StrictFrozenModel):
    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-bound-feedback"] = "portfolio-s1-bound-feedback"
    policy_version: Literal["portfolio-s1-bound-feedback-v3"] = (
        BOUND_FEEDBACK_POLICY_VERSION_V3
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    reservation_sha256: Sha256
    query_id: str
    feedback_packet: FeedbackPacketV3
    feedback_result: FeedbackEvaluationResult
    status: FeedbackStatus
    provider_call_count: Literal[1] = 1
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackEvaluationResult)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.status != self.status
            or self.feedback_result.cache_namespace != "feedback-evaluator-v11"
            or self.feedback_result.model != "qwen3.8-max"
        ):
            raise ValueError("bound Feedback artifact v3 identity drifted")
        if self.artifact_sha256 != _model_hash(self, "artifact_sha256"):
            raise ValueError("bound Feedback artifact v3 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_bound_feedback_artifact_v3(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV10,
    entry: FeedbackSelectionEntryV2,
    packet: FeedbackPacketV3,
    result: FeedbackEvaluationResult,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    reservation: FeedbackCallReservationV3,
) -> BoundFeedbackArtifactV3:
    require_verified_static_feedback_source_v2(
        verified_source, selection, control, entry
    )
    expected_reservation = build_feedback_call_reservation_v3(
        selection, control, entry, verified_source=verified_source
    )
    if (
        verified_source.packet != packet
        or reservation != expected_reservation
        or result.schema_version != 5
        or result.cache_namespace != "feedback-evaluator-v11"
        or result.parser_policy_version != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or result.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or result.prompt_policy_version != VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
        or result.prompt_policy_sha256 != VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
        or result.transport_policy_version != QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION
        or result.transport_policy_sha256 != QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256
        or result.requested_response_format != "json_schema"
        or result.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
        or result.requested_thinking is not True
        or result.requested_thinking_budget != 2048
        or result.requested_timeout_seconds != 600
        or result.requested_temperature is not None
        or result.requested_top_p is not None
        or result.packet_sha256 != packet.packet_sha256
        or result.image_sha256 != entry.image_sha256
        or result.remote_authorization_id != control.authorization_id
        or result.remote_authorization_file_sha256 != control.authorization_file_sha256
        or result.provider != "qwen"
        or result.model != "qwen3.8-max"
        or result.max_tokens is not None
        or result.max_completion_tokens != 4096
        or result.max_attempts != 1
    ):
        raise PortfolioS1FeedbackError("bound Feedback artifact v3 source drifted")
    unsigned = {
        "schema_version": 3,
        "kind": "portfolio-s1-bound-feedback",
        "policy_version": BOUND_FEEDBACK_POLICY_VERSION_V3,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "query_id": entry.query_id,
        "feedback_packet": packet,
        "feedback_result": result,
        "status": result.status,
        "provider_call_count": 1,
    }
    return BoundFeedbackArtifactV3.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def write_feedback_call_reservation_v3(
    path: str | Path, reservation: FeedbackCallReservationV3
) -> Path:
    return atomic_create_file(path, reservation.canonical_bytes())


def load_feedback_call_reservation_v3(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_control_sha256: str,
    expected_entry_sha256: str,
) -> FeedbackCallReservationV3:
    content = read_stable_regular_file(
        path, label="Feedback call reservation v3", max_bytes=2 * 1024 * 1024
    )
    reservation = FeedbackCallReservationV3.model_validate_json(content, strict=True)
    if reservation.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v3 is not canonical JSON"
        )
    if (
        reservation.selection_sha256 != expected_selection_sha256
        or reservation.control_sha256 != expected_control_sha256
        or reservation.selection_entry_sha256 != expected_entry_sha256
    ):
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v3 resume identity conflict"
        )
    return reservation


def write_bound_feedback_artifact_v3(
    path: str | Path, artifact: BoundFeedbackArtifactV3
) -> Path:
    return atomic_create_file(path, artifact.canonical_bytes())


def load_bound_feedback_artifact_v3(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_selection_sha256: str | None = None,
    expected_control_sha256: str | None = None,
    expected_entry_sha256: str | None = None,
) -> BoundFeedbackArtifactV3:
    content = read_stable_regular_file(
        path, label="bound S1 Feedback artifact v3", max_bytes=8 * 1024 * 1024
    )
    if (
        expected_file_sha256 is not None
        and sha256_bytes(content) != expected_file_sha256
    ):
        raise PortfolioS1FeedbackError("bound Feedback artifact v3 SHA-256 mismatch")
    artifact = BoundFeedbackArtifactV3.model_validate_json(content, strict=True)
    if artifact.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("bound Feedback artifact v3 is not canonical")
    expected = (
        (expected_selection_sha256, artifact.selection_sha256),
        (expected_control_sha256, artifact.control_sha256),
        (expected_entry_sha256, artifact.selection_entry_sha256),
    )
    if any(wanted is not None and wanted != actual for wanted, actual in expected):
        raise PortfolioS1FeedbackError(
            "bound Feedback artifact v3 resume identity conflict"
        )
    return artifact


def resume_bound_feedback_artifact_v3(
    artifact: BoundFeedbackArtifactV3,
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV10,
    entry: FeedbackSelectionEntryV2,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    reservation: FeedbackCallReservationV3,
) -> BoundFeedbackArtifactV3:
    expected = build_bound_feedback_artifact_v3(
        selection,
        control,
        entry,
        verified_source.packet,
        artifact.feedback_result,
        verified_source=verified_source,
        reservation=reservation,
    )
    if artifact != expected:
        raise PortfolioS1FeedbackError("bound Feedback v3 resume source conflict")
    return artifact


class PortfolioS1FeedbackRunV3(PortfolioS1FeedbackRunV2):
    schema_version: Literal[3] = 3
    policy_version: Literal["portfolio-s1-feedback-run-v3"] = (
        S1_FEEDBACK_RUN_POLICY_VERSION_V3
    )


def build_portfolio_s1_feedback_run_v3(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV10,
    artifacts: tuple[object, ...],
    *,
    orphaned_entry_sha256s: tuple[str, ...] = (),
) -> PortfolioS1FeedbackRunV3:
    if control.selection_sha256 != selection.selection_sha256:
        raise PortfolioS1FeedbackError("Feedback run v3 control binding drifted")
    artifact_by_entry: dict[str, object] = {}
    for artifact in artifacts:
        entry_sha = getattr(artifact, "selection_entry_sha256", None)
        if not isinstance(entry_sha, str) or entry_sha in artifact_by_entry:
            raise PortfolioS1FeedbackError("Feedback run v3 artifacts repeat")
        artifact_by_entry[entry_sha] = artifact
    orphaned = set(orphaned_entry_sha256s)
    if len(orphaned) != len(orphaned_entry_sha256s) or orphaned & set(
        artifact_by_entry
    ):
        raise PortfolioS1FeedbackError("Feedback run v3 orphan identity drifted")
    rows: list[PortfolioS1FeedbackRunEntryV2] = []
    for entry in selection.entries:
        artifact = artifact_by_entry.get(entry.entry_sha256)
        if artifact is None and entry.entry_sha256 not in orphaned:
            continue
        if artifact is None:
            rows.append(
                PortfolioS1FeedbackRunEntryV2(
                    selection_ordinal=entry.selection_ordinal,
                    selection_entry_sha256=entry.entry_sha256,
                    status="orphan",
                )
            )
            continue
        result = getattr(artifact, "feedback_result", None)
        if (
            type(artifact) is not BoundFeedbackArtifactV3
            or getattr(artifact, "selection_sha256", None) != selection.selection_sha256
            or getattr(artifact, "control_sha256", None) != control.control_sha256
            or getattr(artifact, "query_id", None) != entry.query_id
            or result is None
            or result.cache_namespace != "feedback-evaluator-v11"
        ):
            raise PortfolioS1FeedbackError("Feedback run v3 artifact binding drifted")
        usage = result.usage
        rows.append(
            PortfolioS1FeedbackRunEntryV2(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                artifact_sha256=artifact.artifact_sha256,
                status=artifact.status,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
            )
        )
    if not rows or len(rows) != len(artifacts) + len(orphaned):
        raise PortfolioS1FeedbackError("Feedback run v3 attempted identities drifted")
    errors = [item for item in rows if item.status != "parsed"]
    if not errors and len(rows) != 240:
        raise PortfolioS1FeedbackError(
            "nonterminal parsed Feedback v3 artifacts cannot publish a run"
        )
    terminal_phase = next(
        count
        for count in FEEDBACK_PHASE_COUNTS_V2
        if max(item.selection_ordinal for item in rows) <= count
    )
    status = (
        "completed"
        if not errors
        else "stopped_orphan"
        if any(item.status == "orphan" for item in errors)
        else "stopped_nonparsed"
    )
    unsigned = {
        "schema_version": 3,
        "kind": "portfolio-s1-feedback-run",
        "policy_version": S1_FEEDBACK_RUN_POLICY_VERSION_V3,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "attempted_count": len(rows),
        "parsed_count": len(rows) - len(errors),
        "error_count": len(errors),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "provider_calls_reserved": len(rows),
        "terminal_phase_count": terminal_phase,
        "status": status,
        "usage_known_count": sum(item.input_tokens is not None for item in rows),
        "usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRunV3.model_validate(
        {**unsigned, "run_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def write_portfolio_s1_feedback_run_v3(
    path: str | Path, run: PortfolioS1FeedbackRunV3
) -> Path:
    return atomic_create_file(path, run.canonical_bytes())


def load_portfolio_s1_feedback_run_v3(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackRunV3:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback run v3",
        model_type=PortfolioS1FeedbackRunV3,
    )  # type: ignore[return-value]


class PortfolioS1FeedbackBundleV6(PortfolioS1FeedbackBundleV5):
    schema_version: Literal[6] = 6
    policy_version: Literal["portfolio-s1-feedback-bundle-v6"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V6
    )


def build_portfolio_s1_feedback_bundle_v6(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV10,
    authorization: PortfolioS1FeedbackAuthorizationV5,
    artifacts: tuple[object, ...],
    run: PortfolioS1FeedbackRunV3,
) -> PortfolioS1FeedbackBundleV6:
    validate_portfolio_s1_feedback_control_v10(control, selection, authorization)
    artifact_by_entry = {
        getattr(item, "selection_entry_sha256", None): item for item in artifacts
    }
    if (
        len(artifacts) != 240
        or len(artifact_by_entry) != 240
        or None in artifact_by_entry
    ):
        raise PortfolioS1FeedbackError("Feedback bundle v6 requires exact240 artifacts")
    expected_run = build_portfolio_s1_feedback_run_v3(selection, control, artifacts)
    if run != expected_run or run.status != "completed":
        raise PortfolioS1FeedbackError(
            "Feedback bundle v6 requires completed parsed240"
        )
    private_entries: list[PortfolioS1FeedbackBundleEntryV5] = []
    model_entries: list[PortfolioS1FeedbackModelEntryV5] = []
    representatives: list[PortfolioS1FeedbackRepresentativeExampleV5] = []
    contracts: dict[str, FeedbackGCSContractProjectionV5] = {}
    artifact_hashes: list[dict[str, object]] = []
    selected_query_ids = tuple(item.query_id for item in selection.entries)
    for selected in selection.entries:
        artifact = artifact_by_entry[selected.entry_sha256]
        packet = getattr(artifact, "feedback_packet", None)
        result = getattr(artifact, "feedback_result", None)
        feedback = None if result is None else getattr(result, "parsed_feedback", None)
        if (
            type(artifact) is not BoundFeedbackArtifactV3
            or type(packet) is not FeedbackPacketV3
            or type(feedback) is not VisualFeedbackOutput
            or artifact.status != "parsed"
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != selected.query_id
            or result.cache_namespace != "feedback-evaluator-v11"
            or result.model != "qwen3.8-max"
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError(
                "Feedback bundle v6 artifact binding drifted"
            )
        require_feedback_v11_creator_projection_privacy(
            result,
            private_query_ids=selected_query_ids,
        )
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("Feedback bundle v6 suggestions repeat")
        diagnostic_feedback = VisualFeedbackOutput(
            schema_version=1,
            summary=feedback.summary,
            rule_violations=feedback.rule_violations,
            ideal_response_gaps=feedback.ideal_response_gaps,
            skill_suggestions=(),
        )
        components = packet.gcs_diagnostics.components
        model_entries.append(
            PortfolioS1FeedbackModelEntryV5(
                selection_ordinal=selected.selection_ordinal,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                answer_mode=selected.answer_mode,
                gcs=selected.gcs,
                gcs_components=components,
                reason_codes=selected.reason_codes,
                diagnostic_feedback=diagnostic_feedback,
                labeled_suggestions=labeled,
                actionable_suggestions=tuple(
                    item.text
                    for item in labeled
                    if item.disposition == "policy_compatible"
                ),
            )
        )
        contract = packet.gcs_contract
        projected_contract = FeedbackGCSContractProjectionV5(
            capability=selected.capability,
            required_sections=contract.required_sections,
            fallback_markers=contract.fallback_markers,
            preferred_fallback_marker=contract.preferred_fallback_marker,
            card_requirement=contract.card_requirement,
            legal_tool_sequences=contract.legal_tool_sequences,
        )
        previous = contracts.setdefault(selected.capability, projected_contract)
        if previous != projected_contract:
            raise PortfolioS1FeedbackError("Feedback bundle v6 GCS contract drifted")
        private_entries.append(
            PortfolioS1FeedbackBundleEntryV5(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                atomic_component_id=selected.atomic_component_id,
                packet_sha256=packet.packet_sha256,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=result.result_sha256,
            )
        )
        artifact_hashes.append(
            {
                "selection_ordinal": selected.selection_ordinal,
                "bound_artifact_sha256": artifact.artifact_sha256,
                "feedback_result_sha256": result.result_sha256,
            }
        )
        if selected.selection_ordinal <= 12:
            representatives.append(
                PortfolioS1FeedbackRepresentativeExampleV5(
                    selection_ordinal=selected.selection_ordinal,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    gcs_diagnostics=packet.gcs_diagnostics,
                    turns=packet.turns,
                    response_text=packet.response_text,
                    cards=packet.cards,
                    tool_evidence=packet.tool_evidence,
                )
            )
    projection = PortfolioS1FeedbackModelProjectionV5(
        coverage=_feedback_v5_coverage(selection),
        gcs_contract=PortfolioS1FeedbackGCSContractSetV5(
            capability_contracts=tuple(contracts[item] for item in GCS_CAPABILITY_ORDER)
        ),
        feedback_entries=tuple(model_entries),
        representative_examples=tuple(representatives),
        actionable_suggestions=tuple(
            PortfolioS1FeedbackActionableSuggestionV5(
                selection_ordinal=item.selection_ordinal,
                capability=item.capability,
                text=text,
            )
            for item in model_entries
            for text in item.actionable_suggestions
        ),
        actionable_suggestion_count=sum(
            len(item.actionable_suggestions) for item in model_entries
        ),
    )
    _validate_model_projection_privacy(projection, private_query_ids=selected_query_ids)
    unsigned = {
        "schema_version": 6,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V6,
        "status": "complete_policy_filtered_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 240,
        "attempted_count": 240,
        "parsed_count": 240,
        "provider_call_count": 240,
        "selected_query_ids": selected_query_ids,
        "entries": tuple(private_entries),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV6.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def write_portfolio_s1_feedback_bundle_v6(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV6
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v6(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV6:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Portfolio S1 Feedback bundle v6",
        model_type=PortfolioS1FeedbackBundleV6,
        max_bytes=128 * 1024 * 1024,
    )  # type: ignore[return-value]


def write_selected_qwen38_feedback_authorization_v5(
    path: str | Path, authorization: PortfolioS1FeedbackAuthorizationV5
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def load_selected_qwen38_feedback_authorization_v5(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackAuthorizationV5:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen3.8 Feedback authorization v5",
        model_type=PortfolioS1FeedbackAuthorizationV5,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_control_v10(
    path: str | Path, control: PortfolioS1FeedbackControlV10
) -> Path:
    return atomic_create_file(path, control.canonical_bytes())


def load_portfolio_s1_feedback_control_v10(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackControlV10:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback control v10",
        model_type=PortfolioS1FeedbackControlV10,
    )  # type: ignore[return-value]


def write_qwen38_feedback_launch_lock_v3(
    path: str | Path, launch: PortfolioS1Qwen38FeedbackLaunchLockV3
) -> Path:
    return atomic_create_file(path, launch.canonical_bytes())


def load_qwen38_feedback_launch_lock_v3(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1Qwen38FeedbackLaunchLockV3:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen3.8 Feedback launch lock v3",
        model_type=PortfolioS1Qwen38FeedbackLaunchLockV3,
    )  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fresh Qwen3.8-Max v2 identity.  This chain is deliberately separate from
# the terminal v1 run above: transport v6 still performs exactly one provider
# attempt, while this outer orchestration may spend one globally claimed
# second attempt for one strict-schema failure on the same selected entry.


class PortfolioS1FeedbackAuthorizationV6(PortfolioS1FeedbackAuthorizationV5):
    schema_version: Literal[6] = 6
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v6"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION_V6
    scope: Literal[
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-same-entry-invalid-json-retry"
    ] = "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-same-entry-invalid-json-retry"
    provider_call_ceiling: Literal[241] = 241
    retry_policy: Literal["one_global_same_entry_strict_schema_retry_v1"] = (
        "one_global_same_entry_strict_schema_retry_v1"
    )
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[1] = 1
    max_attempts_per_retried_query: Literal[2] = 2
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-retry-v1"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["93.463656000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )

    @field_validator("retry_eligible_error_codes", mode="before")
    @classmethod
    def _retry_codes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen3.8 Feedback v2 reviewed_at needs a timezone")
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback v2 phases drifted")
        if self.owner_approved_phase_hard_cap_cny != self.phase_hard_cap_cny:
            raise ValueError("Qwen3.8 Feedback v2 owner budget approval drifted")
        if (
            len(self.selected_assets) != 240
            or tuple(item.query_id for item in self.selected_assets)
            != tuple(sorted(item.query_id for item in self.selected_assets))
            or len({item.query_id for item in self.selected_assets}) != 240
            or len({item.asset_id for item in self.selected_assets}) != 240
        ):
            raise ValueError("Qwen3.8 Feedback v2 must bind query-sorted unique240")
        asset_payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != _hash_payload(asset_payload):
            raise ValueError("Qwen3.8 Feedback v2 asset-set hash drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN38_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4,
            QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V4,
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11,
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11,
        ):
            raise ValueError("Qwen3.8 Feedback v2 governance identity drifted")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Qwen3.8 Feedback authorization v6 self hash mismatch")
        return self


def validate_selected_qwen38_feedback_authorization_v6(
    authorization: PortfolioS1FeedbackAuthorizationV6,
    selection: PortfolioS1FeedbackSelectionV2,
) -> None:
    assets = _selected_qwen_feedback_assets_v4(selection)
    if (
        type(authorization) is not PortfolioS1FeedbackAuthorizationV6
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
        or authorization.selected_asset_set_sha256
        != _hash_payload([item.model_dump(mode="json") for item in assets])
    ):
        raise PortfolioS1FeedbackError(
            "Qwen3.8 Feedback authorization v6 differs from selected240"
        )


def build_selected_qwen38_feedback_authorization_v6(
    selection: PortfolioS1FeedbackSelectionV2,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
    owner_approved_phase_hard_cap_cny: str,
    model_source_lock_file_sha256: str,
    model_source_lock_sha256: str,
    pricing_lock_file_sha256: str,
    pricing_lock_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV6:
    if owner_approved_phase_hard_cap_cny != "94.000000000000":
        raise PortfolioS1FeedbackError(
            "Qwen3.8 Feedback v2 requires explicit owner approval of the CNY94 cap"
        )
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selected_qwen_feedback_assets_v4(selection)
    unsigned = {
        "schema_version": 6,
        "kind": "portfolio-s1-feedback-selected-asset-authorization",
        "policy_version": FEEDBACK_AUTHORIZATION_POLICY_VERSION_V6,
        "authorization_id": authorization_id,
        "status": "owner-approved",
        "scope": (
            "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-same-entry-invalid-json-retry"
        ),
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "owner_statement": owner_statement,
        "owner_approved_phase_hard_cap_cny": owner_approved_phase_hard_cap_cny,
        "selection_sha256": selection.selection_sha256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "requested_response_format": "json_schema",
        "requested_json_schema_policy_version": QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION,
        "requested_json_schema_name": QWEN38_FEEDBACK_JSON_SCHEMA_NAME,
        "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
        "transport_policy_version": QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_thinking": True,
        "requested_thinking_budget": QWEN38_FEEDBACK_THINKING_BUDGET,
        "requested_timeout_seconds": QWEN38_FEEDBACK_TIMEOUT_SECONDS,
        "requested_temperature": None,
        "requested_top_p": None,
        "requested_max_tokens": None,
        "max_completion_tokens": QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS,
        "selected_query_count": 240,
        "provider_call_ceiling": 241,
        # Every provider invocation remains a one-attempt transport-v6 call.
        "max_attempts": 1,
        "retry_policy": "one_global_same_entry_strict_schema_retry_v1",
        "normal_attempts_per_selected_query": 1,
        "global_retry_token_count": 1,
        "max_attempts_per_retried_query": 2,
        "retry_eligible_error_codes": ("invalid_feedback_json",),
        "outer_retry_policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1,
        "outer_retry_policy_sha256": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1,
        "attempt_transport_retry_policy": ("no_internal_retry_each_provider_attempt"),
        "feedback_concurrency": 2,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "input_token_reservation_ceiling_per_call": 20000,
        "output_token_reservation_ceiling_per_call": 4106,
        "per_call_reservation_cny": "0.387816000000",
        "maximum_reservation_cny": QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
        "phase_hard_cap_cny": "94.000000000000",
        "over_budget_policy": "fail_closed_before_provider_call",
        "parent_membership_processor": "dashscope-qwen-assistant",
        "parent_remote_authorization_id": parent.authorization.authorization_id,
        "parent_remote_authorization_file_sha256": parent.authorization_file_sha256,
        "parent_remote_receipt_file_sha256": parent.receipt_file_sha256,
        "parent_remote_receipt_sha256": parent.receipt.receipt_sha256,
        "parent_remote_catalog_sha256": parent.catalog.catalog_sha256,
        "model_source_lock_file_sha256": model_source_lock_file_sha256,
        "model_source_lock_sha256": model_source_lock_sha256,
        "pricing_lock_file_sha256": pricing_lock_file_sha256,
        "pricing_lock_sha256": pricing_lock_sha256,
        "role_selection_file_sha256": role_selection_file_sha256,
        "role_selection_sha256": role_selection_sha256,
        "cloud_upload_allowed": True,
        "remote_model_inference_allowed": True,
        "redistribution_allowed": False,
        "public_demo_allowed": False,
        "selected_assets": assets,
        "selected_asset_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in assets]
        ),
    }
    return PortfolioS1FeedbackAuthorizationV6.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1FeedbackControlV11(PortfolioS1FeedbackControlV10):
    schema_version: Literal[11] = 11
    policy_version: Literal["portfolio-s1-feedback-control-v11"] = (
        FEEDBACK_CONTROL_POLICY_VERSION_V11
    )
    provider_call_ceiling: Literal[241] = 241
    retry_policy: Literal["one_global_same_entry_strict_schema_retry_v1"] = (
        "one_global_same_entry_strict_schema_retry_v1"
    )
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[1] = 1
    max_attempts_per_retried_query: Literal[2] = 2
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-retry-v1"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"

    @field_validator("retry_eligible_error_codes", mode="before")
    @classmethod
    def _retry_codes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Feedback control v11 phases drifted")
        if self.control_sha256 != _model_hash(self, "control_sha256"):
            raise ValueError("Feedback control v11 self hash mismatch")
        return self


def validate_portfolio_s1_feedback_control_v11(
    control: PortfolioS1FeedbackControlV11,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV6,
) -> None:
    validate_selected_qwen38_feedback_authorization_v6(authorization, selection)
    if (
        type(control) is not PortfolioS1FeedbackControlV11
        or control.selection_sha256 != selection.selection_sha256
        or control.authorization_sha256 != authorization.authorization_sha256
        or control.authorization_id != authorization.authorization_id
        or control.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or control.corpus_sha256 != selection.corpus_sha256
        or control.parent_static_bank_sha256 != selection.parent_static_bank_sha256
        or control.phase_entry_sha256s != selection.phase_entry_sha256s
        or control.phase_query_ids != selection.phase_query_ids
    ):
        raise PortfolioS1FeedbackError("Feedback control v11 binding drifted")


def build_portfolio_s1_feedback_control_v11(
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV6,
    *,
    rubric: RubricSnapshot,
) -> PortfolioS1FeedbackControlV11:
    validate_selected_qwen38_feedback_authorization_v6(authorization, selection)
    unsigned = {
        "schema_version": 11,
        "kind": "portfolio-s1-feedback-control",
        "policy_version": FEEDBACK_CONTROL_POLICY_VERSION_V11,
        "selection_sha256": selection.selection_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_id": authorization.authorization_id,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256,
        "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "selected_query_count": 240,
        "provider_call_ceiling": 241,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "phase_entry_sha256s": selection.phase_entry_sha256s,
        "phase_query_ids": selection.phase_query_ids,
        "rubric": rubric,
        "feedback_concurrency": 2,
        # max_attempts is the unchanged per-provider-call transport setting.
        "max_attempts": 1,
        "retry_policy": "one_global_same_entry_strict_schema_retry_v1",
        "normal_attempts_per_selected_query": 1,
        "global_retry_token_count": 1,
        "max_attempts_per_retried_query": 2,
        "retry_eligible_error_codes": ("invalid_feedback_json",),
        "outer_retry_policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1,
        "outer_retry_policy_sha256": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1,
        "attempt_transport_retry_policy": ("no_internal_retry_each_provider_attempt"),
        "max_completion_tokens": 4096,
        "requested_thinking": True,
        "requested_thinking_budget": 2048,
        "requested_timeout_seconds": 600,
        "require_all_parsed_for_bundle": True,
        "stop_on_nonparsed_or_orphan": True,
    }
    return PortfolioS1FeedbackControlV11.model_validate(
        {**unsigned, "control_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


class PortfolioS1Qwen38FeedbackLaunchLockV4(PortfolioS1Qwen38FeedbackLaunchLockV3):
    schema_version: Literal[4] = 4
    policy_version: Literal["portfolio-s1-qwen38-feedback-launch-lock-v4"] = (
        QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V4
    )
    provider_call_ceiling: Literal[241] = 241
    max_attempts_per_selected_query: Literal[2] = 2
    retry_policy: Literal["one_global_same_entry_strict_schema_retry_v1"] = (
        "one_global_same_entry_strict_schema_retry_v1"
    )
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[1] = 1
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-retry-v1"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["93.463656000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )

    @field_validator("retry_eligible_error_codes", mode="before")
    @classmethod
    def _retry_codes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback v2 launch phases drifted")
        if self.owner_approved_phase_hard_cap_cny != self.phase_hard_cap_cny:
            raise ValueError("Qwen3.8 Feedback v2 launch lacks budget approval")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN38_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4,
            QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V4,
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11,
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11,
        ):
            raise ValueError("Qwen3.8 Feedback v2 launch governance drifted")
        if self.launch_lock_sha256 != _model_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback launch v4 self hash mismatch")
        return self


def build_qwen38_feedback_launch_lock_v4(
    *,
    run_id: str,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV6,
    control: PortfolioS1FeedbackControlV11,
) -> PortfolioS1Qwen38FeedbackLaunchLockV4:
    validate_portfolio_s1_feedback_control_v11(control, selection, authorization)
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-qwen38-feedback-launch-lock",
        "policy_version": QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V4,
        "run_id": run_id,
        "status": "prepared-no-provider-calls",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "model_source_lock_file_sha256": authorization.model_source_lock_file_sha256,
        "model_source_lock_sha256": authorization.model_source_lock_sha256,
        "pricing_lock_file_sha256": authorization.pricing_lock_file_sha256,
        "pricing_lock_sha256": authorization.pricing_lock_sha256,
        "role_selection_file_sha256": authorization.role_selection_file_sha256,
        "role_selection_sha256": authorization.role_selection_sha256,
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "processor": QWEN38_FEEDBACK_PROCESSOR,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "provider_call_ceiling": 241,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "feedback_concurrency": 2,
        "max_attempts_per_selected_query": 2,
        "retry_policy": "one_global_same_entry_strict_schema_retry_v1",
        "normal_attempts_per_selected_query": 1,
        "global_retry_token_count": 1,
        "retry_eligible_error_codes": ("invalid_feedback_json",),
        "outer_retry_policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1,
        "outer_retry_policy_sha256": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1,
        "attempt_transport_retry_policy": ("no_internal_retry_each_provider_attempt"),
        "per_call_reservation_cny": "0.387816000000",
        "maximum_reservation_cny": QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2,
        "owner_approved_phase_hard_cap_cny": (
            authorization.owner_approved_phase_hard_cap_cny
        ),
        "phase_hard_cap_cny": "94.000000000000",
        "provider_calls_performed": 0,
    }
    return PortfolioS1Qwen38FeedbackLaunchLockV4.model_validate(
        {**unsigned, "launch_lock_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def is_qwen38_strict_schema_retry_eligible(
    result: FeedbackEvaluationResult,
) -> bool:
    """Admit only a genuine stopped text response rejected by parser v3."""

    envelope_eligible = (
        type(result) is FeedbackEvaluationResult
        and result.status == "parse_error"
        and result.error_code == "invalid_feedback_json"
        and result.raw_response_text is not None
        and result.response_redaction_reason is None
        and result.finish_reason == "stop"
        and not result.tool_calls
        and result.parsed_feedback is None
        and result.request_id is not None
    )
    if not envelope_eligible:
        return False
    assert result.raw_response_text is not None
    try:
        parse_visual_feedback_output_v3(result.raw_response_text)
    except EvaluatorOutputParseError:
        return True
    return False


class FeedbackGlobalRetryClaimV1(_StrictFrozenModel):
    """Create-only ownership of the run's sole outer retry token."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-global-retry-claim"] = (
        "portfolio-s1-feedback-global-retry-claim"
    )
    policy_version: Literal["portfolio-s1-feedback-global-schema-retry-v1"] = (
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1
    )
    policy_sha256: Literal[QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1] = (
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    first_artifact_sha256: Sha256
    first_feedback_result_sha256: Sha256
    trigger_status: Literal["parse_error"] = "parse_error"
    trigger_error_code: Literal["invalid_feedback_json"] = "invalid_feedback_json"
    retry_attempt_index: Literal[2] = 2
    retry_identity: Literal["same-selection-entry-no-replacement"] = (
        "same-selection-entry-no-replacement"
    )
    claim_sha256: Sha256

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        if self.claim_sha256 != _model_hash(self, "claim_sha256"):
            raise ValueError("Feedback global retry claim self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class FeedbackCallReservationV4(_StrictFrozenModel):
    """Create-only proof for one external provider attempt in fresh v2."""

    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-feedback-call-reservation"] = (
        "portfolio-s1-feedback-call-reservation"
    )
    policy_version: Literal["portfolio-s1-feedback-call-reservation-v6"] = (
        FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V6
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    corpus_sha256: Sha256
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=241)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_trigger_error_code: Literal["invalid_feedback_json"] | None = None
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    cache_namespace: Literal["feedback-evaluator-v11"] = QWEN38_FEEDBACK_CACHE_NAMESPACE
    max_tokens: None = None
    max_completion_tokens: Literal[4096] = 4096
    provider_internal_max_attempts: Literal[1] = 1
    reservation_cny: Literal["0.387816000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        retry_fields = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_trigger_error_code,
        )
        if self.attempt_index == 1 and any(value is not None for value in retry_fields):
            raise ValueError("first Feedback attempt cannot claim retry ancestry")
        if self.attempt_index == 2 and any(value is None for value in retry_fields):
            raise ValueError("second Feedback attempt lacks retry ancestry")
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Feedback call reservation v4 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class BoundFeedbackArtifactV4(_StrictFrozenModel):
    """Immutable settlement for exactly one V4 reservation/provider attempt."""

    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-bound-feedback"] = "portfolio-s1-bound-feedback"
    policy_version: Literal["portfolio-s1-bound-feedback-v4"] = (
        BOUND_FEEDBACK_POLICY_VERSION_V4
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    reservation_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=241)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    query_id: str
    feedback_packet: FeedbackPacketV3
    feedback_result: FeedbackEvaluationResult
    status: FeedbackStatus
    provider_call_count: Literal[1] = 1
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackEvaluationResult)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.status != self.status
            or self.feedback_result.cache_namespace != "feedback-evaluator-v11"
            or self.feedback_result.model != "qwen3.8-max"
        ):
            raise ValueError("bound Feedback artifact v4 identity drifted")
        if self.attempt_index == 1 and (
            self.previous_artifact_sha256 is not None
            or self.retry_claim_sha256 is not None
        ):
            raise ValueError("first bound Feedback artifact has retry ancestry")
        if self.attempt_index == 2 and (
            self.previous_artifact_sha256 is None or self.retry_claim_sha256 is None
        ):
            raise ValueError("second bound Feedback artifact lacks retry ancestry")
        if self.artifact_sha256 != _model_hash(self, "artifact_sha256"):
            raise ValueError("bound Feedback artifact v4 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_feedback_global_retry_claim_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV11,
    entry: FeedbackSelectionEntryV2,
    first_artifact: BoundFeedbackArtifactV4,
) -> FeedbackGlobalRetryClaimV1:
    if (
        type(first_artifact) is not BoundFeedbackArtifactV4
        or first_artifact.selection_sha256 != selection.selection_sha256
        or first_artifact.control_sha256 != control.control_sha256
        or first_artifact.selection_entry_sha256 != entry.entry_sha256
        or first_artifact.query_id != entry.query_id
        or first_artifact.attempt_index != 1
        or not is_qwen38_strict_schema_retry_eligible(first_artifact.feedback_result)
    ):
        raise PortfolioS1FeedbackError(
            "Feedback global retry claim requires one eligible first settlement"
        )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-global-retry-claim",
        "policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1,
        "policy_sha256": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "selection_ordinal": entry.selection_ordinal,
        "query_id": entry.query_id,
        "first_artifact_sha256": first_artifact.artifact_sha256,
        "first_feedback_result_sha256": first_artifact.feedback_result.result_sha256,
        "trigger_status": "parse_error",
        "trigger_error_code": "invalid_feedback_json",
        "retry_attempt_index": 2,
        "retry_identity": "same-selection-entry-no-replacement",
    }
    return FeedbackGlobalRetryClaimV1.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def build_feedback_call_reservation_v4(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV11,
    entry: FeedbackSelectionEntryV2,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
    global_call_ordinal: int,
    first_artifact: BoundFeedbackArtifactV4 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV1 | None = None,
) -> FeedbackCallReservationV4:
    require_verified_static_feedback_source_v2(
        verified_source, selection, control, entry
    )
    if attempt_index == 1:
        if first_artifact is not None or retry_claim is not None:
            raise PortfolioS1FeedbackError("first Feedback reservation has retry state")
    else:
        if first_artifact is None or retry_claim is None:
            raise PortfolioS1FeedbackError("retry reservation lacks settled ancestry")
        expected_claim = build_feedback_global_retry_claim_v1(
            selection, control, entry, first_artifact
        )
        if retry_claim != expected_claim:
            raise PortfolioS1FeedbackError("retry reservation claim drifted")
    row = verified_source.row
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-feedback-call-reservation",
        "policy_version": FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V6,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "selection_ordinal": entry.selection_ordinal,
        "query_id": entry.query_id,
        "packet_sha256": verified_source.packet.packet_sha256,
        "checkpoint_file_sha256": row.checkpoint_file_sha256,
        "checkpoint_row_sha256": row.checkpoint_row_sha256,
        "sidecar_sha256": row.sidecar.evidence_sha256,
        "attempt_index": attempt_index,
        "global_call_ordinal": global_call_ordinal,
        "previous_artifact_sha256": (
            None if first_artifact is None else first_artifact.artifact_sha256
        ),
        "retry_claim_sha256": None if retry_claim is None else retry_claim.claim_sha256,
        "retry_trigger_error_code": (
            None if attempt_index == 1 else "invalid_feedback_json"
        ),
        "provider": "qwen",
        "model": QWEN38_FEEDBACK_MODEL,
        "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE,
        "max_tokens": None,
        "max_completion_tokens": 4096,
        "provider_internal_max_attempts": 1,
        "reservation_cny": "0.387816000000",
    }
    return FeedbackCallReservationV4.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


def build_bound_feedback_artifact_v4(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV11,
    entry: FeedbackSelectionEntryV2,
    packet: FeedbackPacketV3,
    result: FeedbackEvaluationResult,
    *,
    verified_source: VerifiedStaticFeedbackSourceV2,
    reservation: FeedbackCallReservationV4,
    first_artifact: BoundFeedbackArtifactV4 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV1 | None = None,
) -> BoundFeedbackArtifactV4:
    expected_reservation = build_feedback_call_reservation_v4(
        selection,
        control,
        entry,
        verified_source=verified_source,
        attempt_index=reservation.attempt_index,
        global_call_ordinal=reservation.global_call_ordinal,
        first_artifact=first_artifact,
        retry_claim=retry_claim,
    )
    same_wire_retry = first_artifact is None or (
        reservation.attempt_index == 2
        and result.packet_sha256 == first_artifact.feedback_result.packet_sha256
        and result.image_sha256 == first_artifact.feedback_result.image_sha256
        and result.prompt_sha256 == first_artifact.feedback_result.prompt_sha256
        and result.wire_sha256 == first_artifact.feedback_result.wire_sha256
        and result.asset_catalog_sha256
        == first_artifact.feedback_result.asset_catalog_sha256
        and result.remote_authorization_id
        == first_artifact.feedback_result.remote_authorization_id
        and result.remote_authorization_file_sha256
        == first_artifact.feedback_result.remote_authorization_file_sha256
        and result.remote_receipt_file_sha256
        == first_artifact.feedback_result.remote_receipt_file_sha256
        and result.remote_receipt_sha256
        == first_artifact.feedback_result.remote_receipt_sha256
        and result.endpoint == first_artifact.feedback_result.endpoint
    )
    if (
        verified_source.packet != packet
        or reservation != expected_reservation
        or not same_wire_retry
        or result.schema_version != 5
        or result.cache_namespace != "feedback-evaluator-v11"
        or result.parser_policy_version != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or result.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or result.prompt_policy_version != VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
        or result.prompt_policy_sha256 != VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
        or result.transport_policy_version != QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION
        or result.transport_policy_sha256 != QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256
        or result.requested_response_format != "json_schema"
        or result.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
        or result.requested_thinking is not True
        or result.requested_thinking_budget != 2048
        or result.requested_timeout_seconds != 600
        or result.requested_temperature is not None
        or result.requested_top_p is not None
        or result.packet_sha256 != packet.packet_sha256
        or result.image_sha256 != entry.image_sha256
        or result.remote_authorization_id != control.authorization_id
        or result.remote_authorization_file_sha256 != control.authorization_file_sha256
        or result.provider != "qwen"
        or result.model != "qwen3.8-max"
        or result.max_tokens is not None
        or result.max_completion_tokens != 4096
        or result.max_attempts != 1
    ):
        raise PortfolioS1FeedbackError("bound Feedback artifact v4 source drifted")
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-bound-feedback",
        "policy_version": BOUND_FEEDBACK_POLICY_VERSION_V4,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "selection_entry_sha256": entry.entry_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "attempt_index": reservation.attempt_index,
        "global_call_ordinal": reservation.global_call_ordinal,
        "previous_artifact_sha256": reservation.previous_artifact_sha256,
        "retry_claim_sha256": reservation.retry_claim_sha256,
        "query_id": entry.query_id,
        "feedback_packet": packet,
        "feedback_result": result,
        "status": result.status,
        "provider_call_count": 1,
    }
    return BoundFeedbackArtifactV4.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(_jsonable(unsigned))},
        strict=True,
    )


class PortfolioS1FeedbackRunAttemptV4(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=241)
    artifact_sha256: Sha256 | None = None
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    error_code: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        if (self.status == "orphan") != (self.artifact_sha256 is None):
            raise ValueError("Feedback run v4 orphan/artifact identity drifted")
        if self.status == "orphan" and any(
            value is not None
            for value in (self.error_code, self.input_tokens, self.output_tokens)
        ):
            raise ValueError("Feedback run v4 orphan cannot claim a result")
        if self.status == "parsed" and self.error_code is not None:
            raise ValueError("parsed Feedback run v4 attempt claims an error")
        return self


class PortfolioS1FeedbackRunV4(_StrictFrozenModel):
    schema_version: Literal[4] = 4
    kind: Literal["portfolio-s1-feedback-run"] = "portfolio-s1-feedback-run"
    policy_version: Literal["portfolio-s1-feedback-run-v4"] = (
        S1_FEEDBACK_RUN_POLICY_VERSION_V4
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    expected_count: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    attempted_count: int = Field(ge=1, le=240)
    parsed_count: int = Field(ge=0, le=240)
    error_count: int = Field(ge=0, le=240)
    orphan_count: int = Field(ge=0, le=241)
    provider_calls_reserved: int = Field(ge=1, le=241)
    retry_count: int = Field(ge=0, le=1)
    retry_claim_entry_sha256: Sha256 | None = None
    terminal_phase_count: Literal[12, 60, 120, 240]
    status: Literal["stopped_nonparsed", "stopped_orphan", "completed"]
    usage_known_count: int = Field(ge=0, le=241)
    usage_unknown_count: int = Field(ge=0, le=241)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    artifacts: tuple[PortfolioS1FeedbackRunAttemptV4, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator("phase_counts", "artifacts", mode="before")
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        keys = tuple(
            (item.selection_ordinal, item.attempt_index) for item in self.artifacts
        )
        call_ordinals = tuple(item.global_call_ordinal for item in self.artifacts)
        final_by_entry = {item.selection_entry_sha256: item for item in self.artifacts}
        statuses = Counter(item.status for item in final_by_entry.values())
        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or keys != tuple(sorted(set(keys)))
            or tuple(sorted(call_ordinals))
            != tuple(range(1, self.provider_calls_reserved + 1))
            or self.provider_calls_reserved != len(self.artifacts)
            or self.attempted_count != len(final_by_entry)
            or self.parsed_count != statuses["parsed"]
            or self.error_count != self.attempted_count - self.parsed_count
            or self.orphan_count
            != sum(item.status == "orphan" for item in self.artifacts)
            or self.retry_count
            != sum(item.attempt_index == 2 for item in self.artifacts)
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count
            != self.provider_calls_reserved - self.usage_known_count
            or self.input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
        ):
            raise ValueError("S1 Feedback run v4 counts drifted")
        if (self.retry_count == 1) != (self.retry_claim_entry_sha256 is not None):
            raise ValueError("S1 Feedback run v4 retry claim/count drifted")
        if self.status == "completed":
            if (
                self.attempted_count != 240
                or self.parsed_count != 240
                or tuple(
                    sorted(item.selection_ordinal for item in final_by_entry.values())
                )
                != tuple(range(1, 241))
            ):
                raise ValueError("completed S1 Feedback run v4 is not parsed240")
        elif self.status == "stopped_orphan":
            if self.orphan_count < 1:
                raise ValueError("stopped-orphan run v4 lacks an orphan")
        elif self.error_count < 1 or self.orphan_count:
            raise ValueError("stopped-nonparsed run v4 status drifted")
        if self.artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.artifacts]
        ):
            raise ValueError("Feedback run v4 artifact set hash drifted")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Feedback run v4 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_run_v4(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV11,
    artifacts: tuple[BoundFeedbackArtifactV4, ...],
    *,
    retry_claim: FeedbackGlobalRetryClaimV1 | None = None,
    orphaned_attempts: tuple[FeedbackCallReservationV4, ...] = (),
) -> PortfolioS1FeedbackRunV4:
    if control.selection_sha256 != selection.selection_sha256:
        raise PortfolioS1FeedbackError("Feedback run v4 control binding drifted")
    entry_by_sha = {item.entry_sha256: item for item in selection.entries}
    artifact_by_key: dict[tuple[str, int], BoundFeedbackArtifactV4] = {}
    for artifact in artifacts:
        key = (artifact.selection_entry_sha256, artifact.attempt_index)
        entry = entry_by_sha.get(artifact.selection_entry_sha256)
        if (
            type(artifact) is not BoundFeedbackArtifactV4
            or entry is None
            or key in artifact_by_key
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != entry.query_id
        ):
            raise PortfolioS1FeedbackError("Feedback run v4 artifacts drifted")
        artifact_by_key[key] = artifact
    orphan_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item
        for item in orphaned_attempts
    }
    if (
        len(orphan_by_key) != len(orphaned_attempts)
        or set(orphan_by_key) & set(artifact_by_key)
        or any(entry_sha not in entry_by_sha for entry_sha, _ in orphan_by_key)
        or any(
            type(item) is not FeedbackCallReservationV4
            or item.selection_sha256 != selection.selection_sha256
            or item.control_sha256 != control.control_sha256
            for item in orphaned_attempts
        )
    ):
        raise PortfolioS1FeedbackError("Feedback run v4 orphan identity drifted")
    attempt2 = [item for item in artifacts if item.attempt_index == 2]
    orphan2 = [key for key in orphan_by_key if key[1] == 2]
    if len(attempt2) + len(orphan2) > 1:
        raise PortfolioS1FeedbackError("Feedback run v4 exceeds one global retry")
    if retry_claim is not None:
        claimed_entry = entry_by_sha.get(retry_claim.selection_entry_sha256)
        first = artifact_by_key.get((retry_claim.selection_entry_sha256, 1))
        if (
            claimed_entry is None
            or first is None
            or retry_claim
            != build_feedback_global_retry_claim_v1(
                selection, control, claimed_entry, first
            )
        ):
            raise PortfolioS1FeedbackError("Feedback run v4 retry claim drifted")
    elif attempt2 or orphan2:
        raise PortfolioS1FeedbackError("Feedback run v4 retry lacks global claim")
    if orphan2:
        assert retry_claim is not None
        orphan_key = orphan2[0]
        reservation = orphan_by_key[orphan_key]
        first = artifact_by_key.get((retry_claim.selection_entry_sha256, 1))
        if (
            orphan_key[0] != retry_claim.selection_entry_sha256
            or first is None
            or reservation.previous_artifact_sha256 != first.artifact_sha256
            or reservation.retry_claim_sha256 != retry_claim.claim_sha256
            or reservation.global_call_ordinal <= first.global_call_ordinal
            or not is_qwen38_strict_schema_retry_eligible(first.feedback_result)
        ):
            raise PortfolioS1FeedbackError(
                "Feedback run v4 orphan retry ancestry drifted"
            )
    rows: list[PortfolioS1FeedbackRunAttemptV4] = []
    for entry in selection.entries:
        first = artifact_by_key.get((entry.entry_sha256, 1))
        second = artifact_by_key.get((entry.entry_sha256, 2))
        if second is not None:
            if (
                first is None
                or retry_claim is None
                or retry_claim.selection_entry_sha256 != entry.entry_sha256
                or second.previous_artifact_sha256 != first.artifact_sha256
                or second.retry_claim_sha256 != retry_claim.claim_sha256
                or second.global_call_ordinal <= first.global_call_ordinal
                or not is_qwen38_strict_schema_retry_eligible(first.feedback_result)
            ):
                raise PortfolioS1FeedbackError("Feedback run v4 retry chain drifted")
        for attempt in (1, 2):
            artifact = artifact_by_key.get((entry.entry_sha256, attempt))
            orphan_reservation = orphan_by_key.get((entry.entry_sha256, attempt))
            if artifact is None and orphan_reservation is None:
                continue
            if artifact is None:
                rows.append(
                    PortfolioS1FeedbackRunAttemptV4(
                        selection_ordinal=entry.selection_ordinal,
                        selection_entry_sha256=entry.entry_sha256,
                        attempt_index=attempt,
                        global_call_ordinal=orphan_reservation.global_call_ordinal,
                        status="orphan",
                    )
                )
                continue
            usage = artifact.feedback_result.usage
            rows.append(
                PortfolioS1FeedbackRunAttemptV4(
                    selection_ordinal=entry.selection_ordinal,
                    selection_entry_sha256=entry.entry_sha256,
                    attempt_index=attempt,
                    global_call_ordinal=artifact.global_call_ordinal,
                    artifact_sha256=artifact.artifact_sha256,
                    status=artifact.status,
                    error_code=artifact.feedback_result.error_code,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                )
            )
    if not rows:
        raise PortfolioS1FeedbackError("Feedback run v4 has no attempts")
    rows.sort(key=lambda item: (item.selection_ordinal, item.attempt_index))
    final_by_entry = {item.selection_entry_sha256: item for item in rows}
    errors = [item for item in final_by_entry.values() if item.status != "parsed"]
    if not errors and len(final_by_entry) != 240:
        raise PortfolioS1FeedbackError(
            "nonterminal parsed Feedback v4 artifacts cannot publish a run"
        )
    # An eligible first result owned by this run is pending, not terminal, until
    # its second reservation settles or is proven orphaned.
    if (
        retry_claim is not None
        and retry_claim.selection_entry_sha256 in final_by_entry
        and final_by_entry[retry_claim.selection_entry_sha256].attempt_index == 1
    ):
        raise PortfolioS1FeedbackError("claimed Feedback retry is still pending")
    terminal_phase = next(
        count
        for count in FEEDBACK_PHASE_COUNTS_V2
        if max(item.selection_ordinal for item in rows) <= count
    )
    status = (
        "completed"
        if not errors
        else "stopped_orphan"
        if any(item.status == "orphan" for item in rows)
        else "stopped_nonparsed"
    )
    unsigned = {
        "schema_version": 4,
        "kind": "portfolio-s1-feedback-run",
        "policy_version": S1_FEEDBACK_RUN_POLICY_VERSION_V4,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "attempted_count": len(final_by_entry),
        "parsed_count": sum(
            item.status == "parsed" for item in final_by_entry.values()
        ),
        "error_count": len(errors),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "provider_calls_reserved": len(rows),
        "retry_count": sum(item.attempt_index == 2 for item in rows),
        "retry_claim_entry_sha256": (
            None if retry_claim is None else retry_claim.selection_entry_sha256
        ),
        "terminal_phase_count": terminal_phase,
        "status": status,
        "usage_known_count": sum(item.input_tokens is not None for item in rows),
        "usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRunV4.model_validate(
        {**unsigned, "run_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def write_selected_qwen38_feedback_authorization_v6(
    path: str | Path, authorization: PortfolioS1FeedbackAuthorizationV6
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def load_selected_qwen38_feedback_authorization_v6(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackAuthorizationV6:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen3.8 Feedback authorization v6",
        model_type=PortfolioS1FeedbackAuthorizationV6,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_control_v11(
    path: str | Path, control: PortfolioS1FeedbackControlV11
) -> Path:
    return atomic_create_file(path, control.canonical_bytes())


def load_portfolio_s1_feedback_control_v11(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackControlV11:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback control v11",
        model_type=PortfolioS1FeedbackControlV11,
    )  # type: ignore[return-value]


def write_qwen38_feedback_launch_lock_v4(
    path: str | Path, launch: PortfolioS1Qwen38FeedbackLaunchLockV4
) -> Path:
    return atomic_create_file(path, launch.canonical_bytes())


def load_qwen38_feedback_launch_lock_v4(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1Qwen38FeedbackLaunchLockV4:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Qwen3.8 Feedback launch lock v4",
        model_type=PortfolioS1Qwen38FeedbackLaunchLockV4,
    )  # type: ignore[return-value]


def write_feedback_global_retry_claim_v1(
    path: str | Path, claim: FeedbackGlobalRetryClaimV1
) -> Path:
    return atomic_create_file(path, claim.canonical_bytes())


def load_feedback_global_retry_claim_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> FeedbackGlobalRetryClaimV1:
    content = read_stable_regular_file(
        path, label="Feedback global retry claim", max_bytes=2 * 1024 * 1024
    )
    if (
        expected_file_sha256 is not None
        and sha256_bytes(content) != expected_file_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback retry claim file SHA-256 mismatch")
    claim = FeedbackGlobalRetryClaimV1.model_validate_json(content, strict=True)
    if claim.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback retry claim is not canonical")
    return claim


def write_feedback_call_reservation_v4(
    path: str | Path, reservation: FeedbackCallReservationV4
) -> Path:
    return atomic_create_file(path, reservation.canonical_bytes())


def load_feedback_call_reservation_v4(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_control_sha256: str,
    expected_entry_sha256: str,
    expected_attempt_index: int,
) -> FeedbackCallReservationV4:
    content = read_stable_regular_file(
        path, label="Feedback call reservation v4", max_bytes=2 * 1024 * 1024
    )
    reservation = FeedbackCallReservationV4.model_validate_json(content, strict=True)
    if reservation.canonical_bytes() != content:
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v4 is not canonical JSON"
        )
    if (
        reservation.selection_sha256 != expected_selection_sha256
        or reservation.control_sha256 != expected_control_sha256
        or reservation.selection_entry_sha256 != expected_entry_sha256
        or reservation.attempt_index != expected_attempt_index
    ):
        raise PortfolioS1FeedbackError(
            "Feedback call reservation v4 resume identity conflict"
        )
    return reservation


def write_bound_feedback_artifact_v4(
    path: str | Path, artifact: BoundFeedbackArtifactV4
) -> Path:
    return atomic_create_file(path, artifact.canonical_bytes())


def load_bound_feedback_artifact_v4(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_control_sha256: str,
    expected_entry_sha256: str,
    expected_attempt_index: int,
) -> BoundFeedbackArtifactV4:
    content = read_stable_regular_file(
        path, label="bound S1 Feedback artifact v4", max_bytes=8 * 1024 * 1024
    )
    artifact = BoundFeedbackArtifactV4.model_validate_json(content, strict=True)
    if artifact.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("bound Feedback artifact v4 is not canonical")
    if (
        artifact.selection_sha256 != expected_selection_sha256
        or artifact.control_sha256 != expected_control_sha256
        or artifact.selection_entry_sha256 != expected_entry_sha256
        or artifact.attempt_index != expected_attempt_index
    ):
        raise PortfolioS1FeedbackError(
            "bound Feedback artifact v4 resume identity conflict"
        )
    return artifact


def write_portfolio_s1_feedback_run_v4(
    path: str | Path, run: PortfolioS1FeedbackRunV4
) -> Path:
    return atomic_create_file(path, run.canonical_bytes())


def load_portfolio_s1_feedback_run_v4(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackRunV4:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Feedback run v4",
        model_type=PortfolioS1FeedbackRunV4,
    )  # type: ignore[return-value]


class PortfolioS1FeedbackBundleV7(PortfolioS1FeedbackBundleV5):
    schema_version: Literal[7] = 7
    policy_version: Literal["portfolio-s1-feedback-bundle-v7"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V7
    )
    provider_call_count: Literal[240, 241]


def build_portfolio_s1_feedback_bundle_v7(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV11,
    authorization: PortfolioS1FeedbackAuthorizationV6,
    artifacts: tuple[BoundFeedbackArtifactV4, ...],
    run: PortfolioS1FeedbackRunV4,
    *,
    retry_claim: FeedbackGlobalRetryClaimV1 | None = None,
) -> PortfolioS1FeedbackBundleV7:
    validate_portfolio_s1_feedback_control_v11(control, selection, authorization)
    if len(artifacts) not in {240, 241}:
        raise PortfolioS1FeedbackError(
            "Feedback bundle v7 requires parsed240 plus at most one first failure"
        )
    final_by_entry: dict[str, BoundFeedbackArtifactV4] = {}
    for artifact in artifacts:
        if type(artifact) is not BoundFeedbackArtifactV4:
            raise PortfolioS1FeedbackError("Feedback bundle v7 artifact type drifted")
        previous = final_by_entry.get(artifact.selection_entry_sha256)
        if previous is None or artifact.attempt_index > previous.attempt_index:
            final_by_entry[artifact.selection_entry_sha256] = artifact
    if len(final_by_entry) != 240:
        raise PortfolioS1FeedbackError("Feedback bundle v7 requires exact240 entries")
    expected_run = build_portfolio_s1_feedback_run_v4(
        selection, control, artifacts, retry_claim=retry_claim
    )
    if run != expected_run or run.status != "completed":
        raise PortfolioS1FeedbackError(
            "Feedback bundle v7 requires completed parsed240"
        )
    private_entries: list[PortfolioS1FeedbackBundleEntryV5] = []
    model_entries: list[PortfolioS1FeedbackModelEntryV5] = []
    representatives: list[PortfolioS1FeedbackRepresentativeExampleV5] = []
    contracts: dict[str, FeedbackGCSContractProjectionV5] = {}
    artifact_hashes: list[dict[str, object]] = []
    selected_query_ids = tuple(item.query_id for item in selection.entries)
    for selected in selection.entries:
        artifact = final_by_entry[selected.entry_sha256]
        packet = artifact.feedback_packet
        result = artifact.feedback_result
        feedback = result.parsed_feedback
        if (
            type(packet) is not FeedbackPacketV3
            or type(feedback) is not VisualFeedbackOutput
            or artifact.status != "parsed"
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != selected.query_id
            or result.cache_namespace != "feedback-evaluator-v11"
            or result.model != "qwen3.8-max"
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError(
                "Feedback bundle v7 artifact binding drifted"
            )
        require_feedback_v11_creator_projection_privacy(
            result, private_query_ids=selected_query_ids
        )
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("Feedback bundle v7 suggestions repeat")
        diagnostic_feedback = VisualFeedbackOutput(
            schema_version=1,
            summary=feedback.summary,
            rule_violations=feedback.rule_violations,
            ideal_response_gaps=feedback.ideal_response_gaps,
            skill_suggestions=(),
        )
        model_entries.append(
            PortfolioS1FeedbackModelEntryV5(
                selection_ordinal=selected.selection_ordinal,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                answer_mode=selected.answer_mode,
                gcs=selected.gcs,
                gcs_components=packet.gcs_diagnostics.components,
                reason_codes=selected.reason_codes,
                diagnostic_feedback=diagnostic_feedback,
                labeled_suggestions=labeled,
                actionable_suggestions=tuple(
                    item.text
                    for item in labeled
                    if item.disposition == "policy_compatible"
                ),
            )
        )
        contract = packet.gcs_contract
        projected_contract = FeedbackGCSContractProjectionV5(
            capability=selected.capability,
            required_sections=contract.required_sections,
            fallback_markers=contract.fallback_markers,
            preferred_fallback_marker=contract.preferred_fallback_marker,
            card_requirement=contract.card_requirement,
            legal_tool_sequences=contract.legal_tool_sequences,
        )
        previous_contract = contracts.setdefault(
            selected.capability, projected_contract
        )
        if previous_contract != projected_contract:
            raise PortfolioS1FeedbackError("Feedback bundle v7 GCS contract drifted")
        private_entries.append(
            PortfolioS1FeedbackBundleEntryV5(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                atomic_component_id=selected.atomic_component_id,
                packet_sha256=packet.packet_sha256,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=result.result_sha256,
            )
        )
        artifact_hashes.append(
            {
                "selection_ordinal": selected.selection_ordinal,
                "bound_artifact_sha256": artifact.artifact_sha256,
                "feedback_result_sha256": result.result_sha256,
            }
        )
        if selected.selection_ordinal <= 12:
            representatives.append(
                PortfolioS1FeedbackRepresentativeExampleV5(
                    selection_ordinal=selected.selection_ordinal,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    gcs_diagnostics=packet.gcs_diagnostics,
                    turns=packet.turns,
                    response_text=packet.response_text,
                    cards=packet.cards,
                    tool_evidence=packet.tool_evidence,
                )
            )
    projection = PortfolioS1FeedbackModelProjectionV5(
        coverage=_feedback_v5_coverage(selection),
        gcs_contract=PortfolioS1FeedbackGCSContractSetV5(
            capability_contracts=tuple(contracts[item] for item in GCS_CAPABILITY_ORDER)
        ),
        feedback_entries=tuple(model_entries),
        representative_examples=tuple(representatives),
        actionable_suggestions=tuple(
            PortfolioS1FeedbackActionableSuggestionV5(
                selection_ordinal=item.selection_ordinal,
                capability=item.capability,
                text=text,
            )
            for item in model_entries
            for text in item.actionable_suggestions
        ),
        actionable_suggestion_count=sum(
            len(item.actionable_suggestions) for item in model_entries
        ),
    )
    _validate_model_projection_privacy(projection, private_query_ids=selected_query_ids)
    unsigned = {
        "schema_version": 7,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V7,
        "status": "complete_policy_filtered_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": sha256_bytes(selection.canonical_bytes()),
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": sha256_bytes(run.canonical_bytes()),
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 240,
        "attempted_count": 240,
        "parsed_count": 240,
        "provider_call_count": len(artifacts),
        "selected_query_ids": selected_query_ids,
        "entries": tuple(private_entries),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "model_projection": projection,
    }
    return PortfolioS1FeedbackBundleV7.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(_jsonable(unsigned))}, strict=True
    )


def write_portfolio_s1_feedback_bundle_v7(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV7
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v7(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV7:
    return _load_forward_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        label="Portfolio S1 Feedback bundle v7",
        model_type=PortfolioS1FeedbackBundleV7,
        max_bytes=128 * 1024 * 1024,
    )  # type: ignore[return-value]


__all__ = [
    "BoundFeedbackArtifact",
    "FeedbackCallReservationV1",
    "FeedbackSelectionEntryV1",
    "PortfolioS1FeedbackAuthorizationV1",
    "PortfolioS1FeedbackAuthorizationV2",
    "PortfolioS1FeedbackBundleEntryV1",
    "PortfolioS1FeedbackBundleV1",
    "PortfolioS1FeedbackBundleV2",
    "PortfolioS1FeedbackBundleV3",
    "PortfolioS1FeedbackBundleV4",
    "PortfolioS1FeedbackCoverageCellV2",
    "PortfolioS1FeedbackControlV1",
    "PortfolioS1FeedbackError",
    "PortfolioS1FeedbackModelProjectionV1",
    "PortfolioS1FeedbackModelProjectionV2",
    "PortfolioS1FeedbackModelProjectionV3",
    "PortfolioS1FeedbackModelProjectionV4",
    "PortfolioS1FeedbackModelEntryV1",
    "PortfolioS1FeedbackRunEntryV1",
    "PortfolioS1FeedbackRunV1",
    "PortfolioS1FeedbackRepresentativeExampleV1",
    "PortfolioS1FeedbackSelectionV1",
    "SelectedFeedbackAssetV1",
    "VerifiedStaticFeedbackSource",
    "build_bound_feedback_artifact",
    "build_feedback_call_reservation",
    "build_portfolio_s1_feedback_control",
    "build_portfolio_s1_feedback_bundle",
    "build_portfolio_s1_feedback_bundle_v2",
    "build_portfolio_s1_feedback_bundle_v3",
    "build_portfolio_s1_feedback_bundle_v4",
    "build_portfolio_s1_feedback_run",
    "build_portfolio_s1_feedback_selection",
    "build_selected_feedback_authorization",
    "build_selected_kimi_feedback_authorization",
    "build_verified_static_feedback_source",
    "build_verified_static_feedback_sources",
    "load_bound_feedback_artifact",
    "load_feedback_call_reservation",
    "load_portfolio_s1_feedback_control",
    "load_portfolio_s1_feedback_run",
    "load_portfolio_s1_feedback_bundle",
    "load_portfolio_s1_feedback_bundle_v2",
    "load_portfolio_s1_feedback_bundle_v3",
    "load_portfolio_s1_feedback_bundle_v4",
    "load_portfolio_s1_feedback_selection",
    "load_selected_feedback_authorization",
    "load_selected_kimi_feedback_authorization",
    "resume_bound_feedback_artifact",
    "require_verified_static_feedback_source",
    "validate_portfolio_s1_feedback_control",
    "validate_selected_feedback_authorization",
    "write_bound_feedback_artifact",
    "write_feedback_call_reservation",
    "write_portfolio_s1_feedback_control",
    "write_portfolio_s1_feedback_run",
    "write_portfolio_s1_feedback_bundle",
    "write_portfolio_s1_feedback_bundle_v2",
    "write_portfolio_s1_feedback_bundle_v3",
    "write_portfolio_s1_feedback_bundle_v4",
    "write_portfolio_s1_feedback_selection",
    "write_selected_feedback_authorization",
    # Forward-only S1 Round-2 contracts.
    "FeedbackCapabilityQuotaV2",
    "FeedbackCallReservationV2",
    "FeedbackGCSContractProjectionV5",
    "FeedbackSelectionEntryV2",
    "BoundFeedbackArtifactV2",
    "PolicyLabeledFeedbackSuggestionV1",
    "PortfolioS1FeedbackAuthorizationV4",
    "PortfolioS1FeedbackBundleV5",
    "PortfolioS1FeedbackControlV9",
    "PortfolioS1FeedbackModelProjectionV5",
    "PortfolioS1FeedbackRunV2",
    "PortfolioS1FeedbackSelectionV2",
    "PortfolioS1QwenFeedbackLaunchLockV2",
    "VerifiedStaticFeedbackSourceV2",
    "build_portfolio_s1_feedback_bundle_v5",
    "build_bound_feedback_artifact_v2",
    "build_feedback_call_reservation_v2",
    "build_portfolio_s1_feedback_control_v9",
    "build_portfolio_s1_feedback_run_v2",
    "build_portfolio_s1_feedback_selection_v2",
    "build_qwen37_feedback_launch_lock_v2",
    "build_selected_qwen_feedback_authorization_v4",
    "build_verified_static_feedback_sources_v2",
    "load_portfolio_s1_feedback_bundle_v5",
    "load_bound_feedback_artifact_v2",
    "load_feedback_call_reservation_v2",
    "load_portfolio_s1_feedback_control_v9",
    "load_portfolio_s1_feedback_run_v2",
    "load_portfolio_s1_feedback_selection_v2",
    "load_qwen37_feedback_launch_lock_v2",
    "load_selected_qwen_feedback_authorization_v4",
    "parse_policy_labeled_feedback_suggestion_v1",
    "resume_bound_feedback_artifact_v2",
    "require_verified_static_feedback_source_v2",
    "require_feedback_v10_creator_projection_privacy",
    "validate_portfolio_s1_feedback_control_v9",
    "validate_selected_qwen_feedback_authorization_v4",
    "write_portfolio_s1_feedback_bundle_v5",
    "write_bound_feedback_artifact_v2",
    "write_feedback_call_reservation_v2",
    "write_portfolio_s1_feedback_control_v9",
    "write_portfolio_s1_feedback_run_v2",
    "write_portfolio_s1_feedback_selection_v2",
    "write_qwen37_feedback_launch_lock_v2",
    "write_selected_qwen_feedback_authorization_v4",
    # Forward-only Qwen3.8-Max Discovery240 contracts.
    "BoundFeedbackArtifactV3",
    "FeedbackCallReservationV3",
    "PortfolioS1FeedbackAuthorizationV5",
    "PortfolioS1FeedbackBundleV6",
    "PortfolioS1FeedbackControlV10",
    "PortfolioS1FeedbackRunV3",
    "PortfolioS1Qwen38FeedbackLaunchLockV3",
    "build_bound_feedback_artifact_v3",
    "build_feedback_call_reservation_v3",
    "build_portfolio_s1_feedback_bundle_v6",
    "build_portfolio_s1_feedback_control_v10",
    "build_portfolio_s1_feedback_run_v3",
    "build_qwen38_feedback_launch_lock_v3",
    "build_selected_qwen38_feedback_authorization_v5",
    "load_bound_feedback_artifact_v3",
    "load_feedback_call_reservation_v3",
    "load_portfolio_s1_feedback_bundle_v6",
    "load_portfolio_s1_feedback_control_v10",
    "load_portfolio_s1_feedback_run_v3",
    "load_qwen38_feedback_launch_lock_v3",
    "load_selected_qwen38_feedback_authorization_v5",
    "require_feedback_v11_creator_projection_privacy",
    "resume_bound_feedback_artifact_v3",
    "validate_portfolio_s1_feedback_control_v10",
    "validate_selected_qwen38_feedback_authorization_v5",
    "write_bound_feedback_artifact_v3",
    "write_feedback_call_reservation_v3",
    "write_portfolio_s1_feedback_bundle_v6",
    "write_portfolio_s1_feedback_control_v10",
    "write_portfolio_s1_feedback_run_v3",
    "write_qwen38_feedback_launch_lock_v3",
    "write_selected_qwen38_feedback_authorization_v5",
    # Fresh Qwen3.8-Max v2 plus one globally claimed strict-schema retry.
    "BoundFeedbackArtifactV4",
    "FeedbackCallReservationV4",
    "FeedbackGlobalRetryClaimV1",
    "PortfolioS1FeedbackAuthorizationV6",
    "PortfolioS1FeedbackBundleV7",
    "PortfolioS1FeedbackControlV11",
    "PortfolioS1FeedbackRunAttemptV4",
    "PortfolioS1FeedbackRunV4",
    "PortfolioS1Qwen38FeedbackLaunchLockV4",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1",
    "build_bound_feedback_artifact_v4",
    "build_feedback_call_reservation_v4",
    "build_feedback_global_retry_claim_v1",
    "build_portfolio_s1_feedback_bundle_v7",
    "build_portfolio_s1_feedback_control_v11",
    "build_portfolio_s1_feedback_run_v4",
    "build_qwen38_feedback_launch_lock_v4",
    "build_selected_qwen38_feedback_authorization_v6",
    "is_qwen38_strict_schema_retry_eligible",
    "load_bound_feedback_artifact_v4",
    "load_feedback_call_reservation_v4",
    "load_feedback_global_retry_claim_v1",
    "load_portfolio_s1_feedback_bundle_v7",
    "load_portfolio_s1_feedback_control_v11",
    "load_portfolio_s1_feedback_run_v4",
    "load_qwen38_feedback_launch_lock_v4",
    "load_selected_qwen38_feedback_authorization_v6",
    "qwen38_feedback_global_retry_policy_v1",
    "validate_portfolio_s1_feedback_control_v11",
    "validate_selected_qwen38_feedback_authorization_v6",
    "write_bound_feedback_artifact_v4",
    "write_feedback_call_reservation_v4",
    "write_feedback_global_retry_claim_v1",
    "write_portfolio_s1_feedback_bundle_v7",
    "write_portfolio_s1_feedback_control_v11",
    "write_portfolio_s1_feedback_run_v4",
    "write_qwen38_feedback_launch_lock_v4",
    "write_selected_qwen38_feedback_authorization_v6",
]
