"""Forward-only fresh-v3 Qwen3.8 Feedback contracts and retry ledger.

The historical v1/v2 contracts remain in :mod:`portfolio_s1_feedback`.  This
module owns the new 6,144-token wire identity and the three create-only global
retry slots; no old run root is resumable through these APIs.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
)
from skillchain.evaluation.packets import (
    FeedbackPacketV3,
    RubricSnapshot,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.evaluation.portfolio_gcs import GCS_CAPABILITY_ORDER, GCS_V2_POLICY_SHA256
from skillchain.evaluation.portfolio_s1_feedback import (
    FEEDBACK_PHASE_COUNTS_V2,
    FeedbackSelectionEntryV2,
    PortfolioS1FeedbackAuthorizationV6,
    PortfolioS1FeedbackActionableSuggestionV5,
    PortfolioS1FeedbackBundleEntryV5,
    PortfolioS1FeedbackBundleV7,
    PortfolioS1FeedbackControlV11,
    PortfolioS1FeedbackGCSContractSetV5,
    PortfolioS1FeedbackModelEntryV5,
    PortfolioS1FeedbackModelProjectionV5,
    PortfolioS1FeedbackRepresentativeExampleV5,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    PortfolioS1Qwen38FeedbackLaunchLockV4,
    VerifiedStaticFeedbackSourceV2,
    FeedbackGCSContractProjectionV5,
    _feedback_v5_coverage,
    _hash_payload,
    _CREATOR_RUNTIME_HANDLE_RE,
    _iter_string_values,
    _model_hash,
    _nested_json_model,
    _selected_qwen_feedback_assets_v4,
    _validate_model_projection_privacy,
    parse_policy_labeled_feedback_suggestion_v1,
    require_verified_static_feedback_source_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV6,
    SelectedFeedbackRemoteBindingV2,
    SelectedFeedbackRemoteRuntimeError,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import VerifiedStaticGCSCorpus
from skillchain.evaluation import portfolio_s1_qwen_governance as governance
from skillchain.evaluation.visual_runtime import (
    SelectedFeedbackImageBinding,
    VerifiedSelectedFeedbackRemoteRuntime,
    _make_verified_selected_feedback_remote_runtime,
)
from skillchain.llm import LLMUsage
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FeedbackStatus = Literal["parsed", "parse_error", "provider_error", "timeout"]

FEEDBACK_AUTHORIZATION_POLICY_VERSION_V7 = (
    "portfolio-s1-feedback-selected-assets-authorization-v7"
)
FRESH_V3_OWNER_BUDGET_STATEMENT = (
    "Owner authorized a CNY150 budget ceiling; the compiler/runtime technical "
    "hard cap remains CNY113."
)
FEEDBACK_CONTROL_POLICY_VERSION_V12 = "portfolio-s1-feedback-control-v12"
QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V5 = (
    "portfolio-s1-qwen38-feedback-launch-lock-v5"
)
FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V7 = (
    "portfolio-s1-feedback-call-reservation-v7"
)
BOUND_FEEDBACK_POLICY_VERSION_V5 = "portfolio-s1-bound-feedback-v5"
S1_FEEDBACK_RUN_POLICY_VERSION_V5 = "portfolio-s1-feedback-run-v5"
S1_FEEDBACK_BUNDLE_POLICY_VERSION_V8 = "portfolio-s1-feedback-bundle-v8"
SELECTED_QWEN38_FEEDBACK_REMOTE_POLICY_VERSION_V7 = (
    "portfolio-s1-feedback-selected240-remote-runtime-v7"
)
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-global-schema-or-length-retry-v2"
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def qwen38_feedback_global_retry_policy_v2() -> dict[str, object]:
    """Return the complete outer-orchestration identity for fresh v3."""

    return {
        "policy_version": QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2,
        "scope": "exact-frozen-discovery-selected240",
        "selected_query_count": 240,
        "provider_call_ceiling": 243,
        "normal_attempts_per_selected_query": 1,
        "global_retry_token_count": 3,
        "max_attempts_per_retried_query": 2,
        "retry_policy": "three_global_same_entry_schema_or_length_retries_v2",
        "retry_eligibility": {
            "common": {
                "result_schema_version": 6,
                "cache_namespace": "feedback-evaluator-v12",
                "transport_policy_version": (
                    "visual-feedback-qwen38-dashscope-json-schema-v7"
                ),
                "transport_policy_sha256": (
                    "462b2ef6f7afb0d618f0f29f0d24590aaac45a00ece52cd99643151045a36a0b"
                ),
                "max_completion_tokens": 6144,
                "status": "parse_error",
                "error_code": "invalid_feedback_json",
                "raw_response_text_required": True,
                "request_id_required": True,
                "response_redaction_reason": None,
                "tool_calls_required_empty": True,
                "parsed_feedback_required_null": True,
            },
            "branches": [
                {
                    "finish_reason": "stop",
                    "strict_parser_v3_must_raise": True,
                },
                {
                    "finish_reason": "length",
                    "strict_parser_v3_reparse": (
                        "not_required_due_to_truncation"
                    ),
                },
            ],
            "policy_label_only_failure": "forbidden",
        },
        "attempt_transport_policy_version": (
            "visual-feedback-qwen38-dashscope-json-schema-v7"
        ),
        "attempt_transport_retry_policy": (
            "no_internal_retry_each_provider_attempt"
        ),
        "retry_identity": "same-selection-entry-no-replacement",
        "claim_policy": {
            "claim_count_ceiling": 3,
            "storage": "individual-create-only-claims",
            "ordering": "ascending-first-global-call-ordinal",
            "eligible_settlements_before_new_first_attempt": (
                "claim-and-retry-first"
            ),
            "same_wave_noneligible": "terminal-before-any-retry",
            "fourth_eligible": "terminal-no-provider-call",
        },
        "persistence_order": [
            "settle-first-attempt-create-only",
            "claim-global-retry-slot-create-only",
            "reserve-second-attempt-create-only",
            "invoke-provider-once",
            "settle-second-attempt-create-only",
        ],
        "claim_without_second_reservation_resume": "allowed-same-claim-only",
        "reservation_without_settlement": "terminal-orphan-no-retry",
        "retry_attempt_failure": "terminal-stop",
        "provider_privacy_input_echo_or_orphan_retry": "forbidden",
    }


QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(qwen38_feedback_global_retry_policy_v2())
)


def _fresh_governance_identity() -> tuple[str, str, str, str, str, str]:
    names = (
        "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2",
        "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2",
        "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5",
        "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5",
        "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12",
        "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12",
    )
    try:
        return tuple(getattr(governance, name) for name in names)  # type: ignore[return-value]
    except AttributeError as error:
        raise PortfolioS1FeedbackError(
            "fresh-v3 governance identities are not finalized"
        ) from error


def is_qwen38_schema_or_length_retry_eligible(
    result: FeedbackEvaluationResult,
) -> bool:
    """Admit only strict-parser failures or unredacted length truncation."""

    common = (
        type(result) is FeedbackEvaluationResult
        and result.schema_version == 6
        and result.cache_namespace == "feedback-evaluator-v12"
        and result.transport_policy_version
        == VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
        and result.transport_policy_sha256
        == VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
        and result.max_completion_tokens == 6144
        and result.status == "parse_error"
        and result.error_code == "invalid_feedback_json"
        and result.raw_response_text is not None
        and result.response_redaction_reason is None
        and not result.tool_calls
        and result.parsed_feedback is None
        and result.request_id is not None
    )
    if not common:
        return False
    if result.finish_reason == "length":
        return True
    if result.finish_reason != "stop":
        return False
    assert result.raw_response_text is not None
    try:
        parse_visual_feedback_output_v3(result.raw_response_text)
    except EvaluatorOutputParseError:
        return True
    return False


class PortfolioS1FeedbackAuthorizationV7(PortfolioS1FeedbackAuthorizationV6):
    schema_version: Literal[7] = 7
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v7"
    ] = FEEDBACK_AUTHORIZATION_POLICY_VERSION_V7
    scope: Literal[
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-three-global-same-entry-schema-or-length-retries"
    ] = "core-opt800-s1-feedback-discovery-selected-240-assets-plus-three-global-same-entry-schema-or-length-retries"
    owner_authorized_budget_ceiling_cny: Literal["150.000000000000"]
    approved_technical_phase_hard_cap_cny: Literal["113.000000000000"]
    # Compatibility alias inherited from v6; in v7 it is explicitly the
    # technical execution stop, never the owner's larger budget ceiling.
    owner_approved_phase_hard_cap_cny: Literal["113.000000000000"]
    owner_statement: Literal[
        "Owner authorized a CNY150 budget ceiling; the compiler/runtime technical hard cap remains CNY113."
    ] = FRESH_V3_OWNER_BUDGET_STATEMENT
    cache_namespace: Literal["feedback-evaluator-v12"] = "feedback-evaluator-v12"
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7"
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    transport_policy_sha256: Literal[
        VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    max_completion_tokens: Literal[6144] = 6144
    provider_call_ceiling: Literal[243] = 243
    retry_policy: Literal[
        "three_global_same_entry_schema_or_length_retries_v2"
    ] = "three_global_same_entry_schema_or_length_retries_v2"
    global_retry_token_count: Literal[3] = 3
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-or-length-retry-v2"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = 6154
    per_call_reservation_cny: Literal["0.461544000000"] = "0.461544000000"
    maximum_reservation_cny: Literal["112.155192000000"] = (
        "112.155192000000"
    )
    phase_hard_cap_cny: Literal["113.000000000000"] = "113.000000000000"

    @field_validator("retry_eligible_finish_reasons", mode="before")
    @classmethod
    def _finish_reasons_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen3.8 Feedback v3 reviewed_at needs a timezone")
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback v3 phases drifted")
        if (
            self.owner_authorized_budget_ceiling_cny != "150.000000000000"
            or self.approved_technical_phase_hard_cap_cny
            != self.phase_hard_cap_cny
            or self.owner_approved_phase_hard_cap_cny
            != self.approved_technical_phase_hard_cap_cny
            or Decimal(self.maximum_reservation_cny)
            >= Decimal(self.approved_technical_phase_hard_cap_cny)
            or Decimal(self.approved_technical_phase_hard_cap_cny)
            > Decimal(self.owner_authorized_budget_ceiling_cny)
        ):
            raise ValueError("Qwen3.8 Feedback v3 budget authority drifted")
        if (
            len(self.selected_assets) != 240
            or len({item.query_id for item in self.selected_assets}) != 240
            or len({item.asset_id for item in self.selected_assets}) != 240
            or tuple(item.query_id for item in self.selected_assets)
            != tuple(sorted(item.query_id for item in self.selected_assets))
        ):
            raise ValueError("Qwen3.8 Feedback v3 must bind query-sorted unique240")
        if self.selected_asset_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.selected_assets]
        ):
            raise ValueError("Qwen3.8 Feedback v3 asset-set hash drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != _fresh_governance_identity():
            raise ValueError("Qwen3.8 Feedback v3 governance identity drifted")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Qwen3.8 Feedback authorization v7 self hash mismatch")
        return self


def build_selected_qwen38_feedback_authorization_v7(
    selection: PortfolioS1FeedbackSelectionV2,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_authorized_budget_ceiling_cny: str,
    approved_technical_phase_hard_cap_cny: str,
    model_source_lock_file_sha256: str,
    model_source_lock_sha256: str,
    pricing_lock_file_sha256: str,
    pricing_lock_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1FeedbackAuthorizationV7:
    if owner_authorized_budget_ceiling_cny != "150.000000000000":
        raise PortfolioS1FeedbackError(
            "fresh-v3 requires explicit owner authorization of the CNY150 ceiling"
        )
    if approved_technical_phase_hard_cap_cny != "113.000000000000":
        raise PortfolioS1FeedbackError(
            "fresh-v3 technical execution hard cap must remain CNY113"
        )
    supplied = (
        model_source_lock_file_sha256,
        model_source_lock_sha256,
        pricing_lock_file_sha256,
        pricing_lock_sha256,
        role_selection_file_sha256,
        role_selection_sha256,
    )
    if supplied != _fresh_governance_identity():
        raise PortfolioS1FeedbackError("fresh-v3 governance identity differs")
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selected_qwen_feedback_assets_v4(selection)
    draft = PortfolioS1FeedbackAuthorizationV7.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        owner_statement=FRESH_V3_OWNER_BUDGET_STATEMENT,
        owner_authorized_budget_ceiling_cny=(
            owner_authorized_budget_ceiling_cny
        ),
        approved_technical_phase_hard_cap_cny=(
            approved_technical_phase_hard_cap_cny
        ),
        owner_approved_phase_hard_cap_cny=(
            approved_technical_phase_hard_cap_cny
        ),
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        parent_remote_catalog_sha256=parent.catalog.catalog_sha256,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        model_source_lock_sha256=model_source_lock_sha256,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
        pricing_lock_sha256=pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        selected_assets=assets,
        selected_asset_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in assets]
        ),
        authorization_sha256="0" * 64,
    )
    # Hash the canonical JSON projection, but feed strict Pydantic validation
    # the Python projection so timezone-aware ``datetime`` does not round-trip
    # through a JSON string and fail strict validation.
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    strict_payload = draft.model_dump(
        mode="python", exclude={"authorization_sha256"}
    )
    return PortfolioS1FeedbackAuthorizationV7.model_validate(
        {
            **strict_payload,
            "authorization_sha256": _hash_payload(unsigned),
        },
        strict=True,
    )


def validate_selected_qwen38_feedback_authorization_v7(
    authorization: PortfolioS1FeedbackAuthorizationV7,
    selection: PortfolioS1FeedbackSelectionV2,
) -> None:
    assets = _selected_qwen_feedback_assets_v4(selection)
    if (
        type(authorization) is not PortfolioS1FeedbackAuthorizationV7
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
    ):
        raise PortfolioS1FeedbackError(
            "Qwen3.8 Feedback authorization v7 differs from selected240"
        )


class PortfolioS1FeedbackControlV12(PortfolioS1FeedbackControlV11):
    schema_version: Literal[12] = 12
    policy_version: Literal["portfolio-s1-feedback-control-v12"] = (
        FEEDBACK_CONTROL_POLICY_VERSION_V12
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7"
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    transport_policy_sha256: Literal[
        VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    cache_namespace: Literal["feedback-evaluator-v12"] = "feedback-evaluator-v12"
    provider_call_ceiling: Literal[243] = 243
    retry_policy: Literal[
        "three_global_same_entry_schema_or_length_retries_v2"
    ] = "three_global_same_entry_schema_or_length_retries_v2"
    global_retry_token_count: Literal[3] = 3
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-or-length-retry-v2"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    max_completion_tokens: Literal[6144] = 6144
    owner_authorized_budget_ceiling_cny: Literal["150.000000000000"]
    approved_technical_phase_hard_cap_cny: Literal["113.000000000000"]

    @field_validator("retry_eligible_finish_reasons", mode="before")
    @classmethod
    def _finish_reasons_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or self.owner_authorized_budget_ceiling_cny
            != "150.000000000000"
            or self.approved_technical_phase_hard_cap_cny
            != "113.000000000000"
        ):
            raise ValueError("Feedback control v12 phases drifted")
        if self.control_sha256 != _model_hash(self, "control_sha256"):
            raise ValueError("Feedback control v12 self hash mismatch")
        return self


def build_portfolio_s1_feedback_control_v12(
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV7,
    *,
    rubric: RubricSnapshot,
) -> PortfolioS1FeedbackControlV12:
    validate_selected_qwen38_feedback_authorization_v7(authorization, selection)
    draft = PortfolioS1FeedbackControlV12.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_id=authorization.authorization_id,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        corpus_sha256=selection.corpus_sha256,
        parent_static_bank_sha256=selection.parent_static_bank_sha256,
        gcs_policy_sha256=GCS_V2_POLICY_SHA256,
        phase_entry_sha256s=selection.phase_entry_sha256s,
        phase_query_ids=selection.phase_query_ids,
        rubric=rubric,
        owner_authorized_budget_ceiling_cny=(
            authorization.owner_authorized_budget_ceiling_cny
        ),
        approved_technical_phase_hard_cap_cny=(
            authorization.approved_technical_phase_hard_cap_cny
        ),
        control_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"control_sha256"})
    return PortfolioS1FeedbackControlV12.model_validate(
        {**unsigned, "control_sha256": _hash_payload(unsigned)}, strict=True
    )


def validate_portfolio_s1_feedback_control_v12(
    control: PortfolioS1FeedbackControlV12,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV7,
) -> None:
    validate_selected_qwen38_feedback_authorization_v7(authorization, selection)
    if (
        type(control) is not PortfolioS1FeedbackControlV12
        or control.selection_sha256 != selection.selection_sha256
        or control.authorization_sha256 != authorization.authorization_sha256
        or control.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or control.phase_entry_sha256s != selection.phase_entry_sha256s
        or control.phase_query_ids != selection.phase_query_ids
        or control.owner_authorized_budget_ceiling_cny
        != authorization.owner_authorized_budget_ceiling_cny
        or control.approved_technical_phase_hard_cap_cny
        != authorization.approved_technical_phase_hard_cap_cny
    ):
        raise PortfolioS1FeedbackError("Feedback control v12 binding drifted")


class PortfolioS1Qwen38FeedbackLaunchLockV5(
    PortfolioS1Qwen38FeedbackLaunchLockV4
):
    schema_version: Literal[5] = 5
    policy_version: Literal[
        "portfolio-s1-qwen38-feedback-launch-lock-v5"
    ] = QWEN38_FEEDBACK_LAUNCH_POLICY_VERSION_V5
    cache_namespace: Literal["feedback-evaluator-v12"] = "feedback-evaluator-v12"
    provider_call_ceiling: Literal[243] = 243
    retry_policy: Literal[
        "three_global_same_entry_schema_or_length_retries_v2"
    ] = "three_global_same_entry_schema_or_length_retries_v2"
    global_retry_token_count: Literal[3] = 3
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_retry_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-or-length-retry-v2"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
    outer_retry_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    per_call_reservation_cny: Literal["0.461544000000"] = "0.461544000000"
    maximum_reservation_cny: Literal["112.155192000000"] = (
        "112.155192000000"
    )
    owner_authorized_budget_ceiling_cny: Literal["150.000000000000"]
    approved_technical_phase_hard_cap_cny: Literal["113.000000000000"]
    # Compatibility alias inherited from v4 with the same technical-only meaning.
    owner_approved_phase_hard_cap_cny: Literal["113.000000000000"]
    phase_hard_cap_cny: Literal["113.000000000000"] = "113.000000000000"

    @field_validator("retry_eligible_finish_reasons", mode="before")
    @classmethod
    def _finish_reasons_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.phase_counts != FEEDBACK_PHASE_COUNTS_V2:
            raise ValueError("Qwen3.8 Feedback v3 launch phases drifted")
        if (
            self.owner_authorized_budget_ceiling_cny != "150.000000000000"
            or self.approved_technical_phase_hard_cap_cny
            != self.phase_hard_cap_cny
            or self.owner_approved_phase_hard_cap_cny
            != self.approved_technical_phase_hard_cap_cny
        ):
            raise ValueError("Qwen3.8 Feedback v3 launch budget authority drifted")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != _fresh_governance_identity():
            raise ValueError("Qwen3.8 Feedback v3 launch governance drifted")
        if self.launch_lock_sha256 != _model_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback launch v5 self hash mismatch")
        return self


def build_qwen38_feedback_launch_lock_v5(
    *,
    run_id: str,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV7,
    control: PortfolioS1FeedbackControlV12,
) -> PortfolioS1Qwen38FeedbackLaunchLockV5:
    validate_portfolio_s1_feedback_control_v12(control, selection, authorization)
    draft = PortfolioS1Qwen38FeedbackLaunchLockV5.model_construct(
        run_id=run_id,
        selection_sha256=selection.selection_sha256,
        selection_file_sha256=sha256_bytes(selection.canonical_bytes()),
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        control_file_sha256=sha256_bytes(control.canonical_bytes()),
        corpus_sha256=selection.corpus_sha256,
        model_source_lock_file_sha256=authorization.model_source_lock_file_sha256,
        model_source_lock_sha256=authorization.model_source_lock_sha256,
        pricing_lock_file_sha256=authorization.pricing_lock_file_sha256,
        pricing_lock_sha256=authorization.pricing_lock_sha256,
        role_selection_file_sha256=authorization.role_selection_file_sha256,
        role_selection_sha256=authorization.role_selection_sha256,
        owner_approved_phase_hard_cap_cny=(
            authorization.owner_approved_phase_hard_cap_cny
        ),
        owner_authorized_budget_ceiling_cny=(
            authorization.owner_authorized_budget_ceiling_cny
        ),
        approved_technical_phase_hard_cap_cny=(
            authorization.approved_technical_phase_hard_cap_cny
        ),
        launch_lock_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_lock_sha256"})
    return PortfolioS1Qwen38FeedbackLaunchLockV5.model_validate(
        {**unsigned, "launch_lock_sha256": _hash_payload(unsigned)}, strict=True
    )


class FeedbackGlobalRetryClaimV2(_StrictFrozenModel):
    """One of three ordered, create-only outer retry claims."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-global-retry-claim"] = (
        "portfolio-s1-feedback-global-retry-claim"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-global-schema-or-length-retry-v2"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
    policy_sha256: Literal[QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2] = (
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    claim_ordinal: Literal[1, 2, 3]
    previous_claim_sha256: Sha256 | None = None
    selection_entry_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    query_id: str
    first_global_call_ordinal: int = Field(ge=1, le=242)
    first_artifact_sha256: Sha256
    first_feedback_result_sha256: Sha256
    trigger_status: Literal["parse_error"] = "parse_error"
    trigger_error_code: Literal["invalid_feedback_json"] = "invalid_feedback_json"
    trigger_finish_reason: Literal["stop", "length"]
    retry_attempt_index: Literal[2] = 2
    retry_identity: Literal["same-selection-entry-no-replacement"] = (
        "same-selection-entry-no-replacement"
    )
    claim_sha256: Sha256

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        if (self.claim_ordinal == 1) != (self.previous_claim_sha256 is None):
            raise ValueError("Feedback retry claim chain is not contiguous")
        if self.claim_sha256 != _model_hash(self, "claim_sha256"):
            raise ValueError("Feedback global retry claim v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_feedback_global_retry_claim_set_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    claims: tuple[FeedbackGlobalRetryClaimV2, ...],
    first_artifacts: tuple["BoundFeedbackArtifactV5", ...],
) -> None:
    if len(claims) > 3:
        raise PortfolioS1FeedbackError("fresh-v3 exceeds three global retry claims")
    first_by_sha = {item.artifact_sha256: item for item in first_artifacts}
    previous: FeedbackGlobalRetryClaimV2 | None = None
    seen_entries: set[str] = set()
    for expected_ordinal, claim in enumerate(claims, 1):
        first = first_by_sha.get(claim.first_artifact_sha256)
        if (
            type(claim) is not FeedbackGlobalRetryClaimV2
            or claim.claim_ordinal != expected_ordinal
            or claim.selection_sha256 != selection.selection_sha256
            or claim.control_sha256 != control.control_sha256
            or claim.selection_entry_sha256 in seen_entries
            or first is None
            or first.attempt_index != 1
            or claim.selection_entry_sha256 != first.selection_entry_sha256
            or selection.entries[claim.selection_ordinal - 1].entry_sha256
            != claim.selection_entry_sha256
            or selection.entries[claim.selection_ordinal - 1].query_id
            != claim.query_id
            or claim.query_id != first.query_id
            or claim.first_global_call_ordinal != first.global_call_ordinal
            or claim.first_feedback_result_sha256
            != first.feedback_result.result_sha256
            or claim.trigger_finish_reason != first.feedback_result.finish_reason
            or not is_qwen38_schema_or_length_retry_eligible(first.feedback_result)
        ):
            raise PortfolioS1FeedbackError("Feedback global retry claim set drifted")
        if previous is None:
            if claim.previous_claim_sha256 is not None:
                raise PortfolioS1FeedbackError("first retry claim has ancestry")
        elif (
            claim.previous_claim_sha256 != previous.claim_sha256
            or claim.first_global_call_ordinal <= previous.first_global_call_ordinal
        ):
            raise PortfolioS1FeedbackError("Feedback retry claim ordering drifted")
        seen_entries.add(claim.selection_entry_sha256)
        previous = claim


def build_feedback_global_retry_claim_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    entry: FeedbackSelectionEntryV2,
    first_artifact: "BoundFeedbackArtifactV5",
    *,
    existing_claims: tuple[FeedbackGlobalRetryClaimV2, ...] = (),
    existing_first_artifacts: tuple["BoundFeedbackArtifactV5", ...] = (),
) -> FeedbackGlobalRetryClaimV2:
    if existing_claims:
        validate_feedback_global_retry_claim_set_v2(
            selection, control, existing_claims, existing_first_artifacts
        )
    if len(existing_claims) >= 3:
        raise PortfolioS1FeedbackError("fourth eligible Feedback result has no retry")
    if (
        type(first_artifact) is not BoundFeedbackArtifactV5
        or first_artifact.attempt_index != 1
        or first_artifact.selection_entry_sha256 != entry.entry_sha256
        or first_artifact.selection_sha256 != selection.selection_sha256
        or first_artifact.control_sha256 != control.control_sha256
        or first_artifact.query_id != entry.query_id
        or any(
            item.selection_entry_sha256 == entry.entry_sha256
            for item in existing_claims
        )
        or not is_qwen38_schema_or_length_retry_eligible(
            first_artifact.feedback_result
        )
    ):
        raise PortfolioS1FeedbackError("Feedback result is not retry-claim eligible")
    previous = existing_claims[-1] if existing_claims else None
    if (
        previous is not None
        and first_artifact.global_call_ordinal <= previous.first_global_call_ordinal
    ):
        raise PortfolioS1FeedbackError("Feedback retry claims must follow call order")
    draft = FeedbackGlobalRetryClaimV2.model_construct(
        selection_sha256=selection.selection_sha256,
        control_sha256=control.control_sha256,
        claim_ordinal=len(existing_claims) + 1,
        previous_claim_sha256=(None if previous is None else previous.claim_sha256),
        selection_entry_sha256=entry.entry_sha256,
        selection_ordinal=entry.selection_ordinal,
        query_id=entry.query_id,
        first_global_call_ordinal=first_artifact.global_call_ordinal,
        first_artifact_sha256=first_artifact.artifact_sha256,
        first_feedback_result_sha256=first_artifact.feedback_result.result_sha256,
        trigger_finish_reason=first_artifact.feedback_result.finish_reason,
        claim_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"claim_sha256"})
    return FeedbackGlobalRetryClaimV2.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(unsigned)}, strict=True
    )


class FeedbackCallReservationV5(_StrictFrozenModel):
    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-feedback-call-reservation"] = (
        "portfolio-s1-feedback-call-reservation"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-call-reservation-v7"
    ] = FEEDBACK_CALL_RESERVATION_POLICY_VERSION_V7
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
    global_call_ordinal: int = Field(ge=1, le=243)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    retry_trigger_error_code: Literal["invalid_feedback_json"] | None = None
    retry_trigger_finish_reason: Literal["stop", "length"] | None = None
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    cache_namespace: Literal["feedback-evaluator-v12"] = "feedback-evaluator-v12"
    max_tokens: None = None
    max_completion_tokens: Literal[6144] = 6144
    provider_internal_max_attempts: Literal[1] = 1
    reservation_cny: Literal["0.461544000000"] = "0.461544000000"
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        retry_fields = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
            self.retry_trigger_error_code,
            self.retry_trigger_finish_reason,
        )
        if self.attempt_index == 1 and any(value is not None for value in retry_fields):
            raise ValueError("first Feedback attempt cannot claim retry ancestry")
        if self.attempt_index == 2 and any(value is None for value in retry_fields):
            raise ValueError("second Feedback attempt lacks retry ancestry")
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Feedback call reservation v5 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class BoundFeedbackArtifactV5(_StrictFrozenModel):
    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-bound-feedback"] = "portfolio-s1-bound-feedback"
    policy_version: Literal["portfolio-s1-bound-feedback-v5"] = (
        BOUND_FEEDBACK_POLICY_VERSION_V5
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    selection_entry_sha256: Sha256
    reservation_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=243)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
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
            or self.feedback_result.schema_version != 6
            or self.feedback_result.cache_namespace != "feedback-evaluator-v12"
            or self.feedback_result.model != "qwen3.8-max"
            or self.feedback_result.max_completion_tokens != 6144
        ):
            raise ValueError("bound Feedback artifact v5 identity drifted")
        ancestry = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
        )
        if self.attempt_index == 1 and any(value is not None for value in ancestry):
            raise ValueError("first bound Feedback artifact has retry ancestry")
        if self.attempt_index == 2 and any(value is None for value in ancestry):
            raise ValueError("second bound Feedback artifact lacks retry ancestry")
        if self.artifact_sha256 != _model_hash(self, "artifact_sha256"):
            raise ValueError("bound Feedback artifact v5 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_feedback_call_reservation_v5(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    entry: FeedbackSelectionEntryV2,
    source: VerifiedStaticFeedbackSourceV2,
    *,
    attempt_index: Literal[1, 2],
    global_call_ordinal: int,
    first_artifact: BoundFeedbackArtifactV5 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV2 | None = None,
) -> FeedbackCallReservationV5:
    require_verified_static_feedback_source_v2(source, selection, control, entry)
    if attempt_index == 1:
        if first_artifact is not None or retry_claim is not None:
            raise PortfolioS1FeedbackError("first attempt cannot carry retry ancestry")
    elif (
        first_artifact is None
        or retry_claim is None
        or first_artifact.attempt_index != 1
        or first_artifact.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.selection_sha256 != selection.selection_sha256
        or retry_claim.control_sha256 != control.control_sha256
        or retry_claim.selection_ordinal != entry.selection_ordinal
        or retry_claim.query_id != entry.query_id
        or retry_claim.first_artifact_sha256 != first_artifact.artifact_sha256
        or retry_claim.first_feedback_result_sha256
        != first_artifact.feedback_result.result_sha256
        or retry_claim.first_global_call_ordinal != first_artifact.global_call_ordinal
        or retry_claim.trigger_finish_reason
        != first_artifact.feedback_result.finish_reason
        or global_call_ordinal <= first_artifact.global_call_ordinal
        or not is_qwen38_schema_or_length_retry_eligible(
            first_artifact.feedback_result
        )
    ):
        raise PortfolioS1FeedbackError("second reservation lacks eligible claim chain")
    draft = FeedbackCallReservationV5.model_construct(
        selection_sha256=selection.selection_sha256,
        control_sha256=control.control_sha256,
        corpus_sha256=selection.corpus_sha256,
        selection_entry_sha256=entry.entry_sha256,
        selection_ordinal=entry.selection_ordinal,
        query_id=entry.query_id,
        packet_sha256=source.packet.packet_sha256,
        checkpoint_file_sha256=source.row.checkpoint_file_sha256,
        checkpoint_row_sha256=source.row.checkpoint_row_sha256,
        sidecar_sha256=source.row.sidecar.evidence_sha256,
        attempt_index=attempt_index,
        global_call_ordinal=global_call_ordinal,
        previous_artifact_sha256=(
            None if first_artifact is None else first_artifact.artifact_sha256
        ),
        retry_claim_sha256=(
            None if retry_claim is None else retry_claim.claim_sha256
        ),
        retry_claim_ordinal=(
            None if retry_claim is None else retry_claim.claim_ordinal
        ),
        retry_trigger_error_code=(
            None if retry_claim is None else retry_claim.trigger_error_code
        ),
        retry_trigger_finish_reason=(
            None if retry_claim is None else retry_claim.trigger_finish_reason
        ),
        reservation_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"reservation_sha256"})
    return FeedbackCallReservationV5.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(unsigned)}, strict=True
    )


def build_bound_feedback_artifact_v5(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    source: VerifiedStaticFeedbackSourceV2,
    result: FeedbackEvaluationResult,
    *,
    reservation: FeedbackCallReservationV5,
    first_artifact: BoundFeedbackArtifactV5 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV2 | None = None,
) -> BoundFeedbackArtifactV5:
    entry = selection.entries[reservation.selection_ordinal - 1]
    expected_reservation = build_feedback_call_reservation_v5(
        selection,
        control,
        entry,
        source,
        attempt_index=reservation.attempt_index,
        global_call_ordinal=reservation.global_call_ordinal,
        first_artifact=first_artifact,
        retry_claim=retry_claim,
    )
    if reservation != expected_reservation:
        raise PortfolioS1FeedbackError("Feedback reservation v5 binding drifted")
    if (
        type(result) is not FeedbackEvaluationResult
        or result.schema_version != 6
        or result.cache_namespace != "feedback-evaluator-v12"
        or result.transport_policy_version
        != VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
        or result.transport_policy_sha256
        != VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
        or result.parser_policy_version
        != VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
        or result.parser_policy_sha256 != VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        or result.prompt_policy_version != VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
        or result.prompt_policy_sha256 != VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
        or result.requested_response_format != "json_schema"
        or result.requested_json_schema_sha256
        != VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
        or result.requested_thinking is not True
        or result.requested_thinking_budget != 2048
        or result.requested_timeout_seconds != 600
        or result.requested_temperature is not None
        or result.requested_top_p is not None
        or result.max_completion_tokens != 6144
        or result.max_tokens is not None
        or result.max_attempts != 1
        or result.provider != "qwen"
        or result.packet_sha256 != source.packet.packet_sha256
        or result.query_id != entry.query_id
        or result.image_sha256 != entry.image_sha256
        or result.image_sha256 != source.packet.image.sha256
        or result.remote_authorization_id != control.authorization_id
        or result.remote_authorization_file_sha256
        != control.authorization_file_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback result v5 source drifted")
    if first_artifact is not None:
        first = first_artifact.feedback_result
        same_wire = (
            result.packet_sha256 == first.packet_sha256
            and result.image_sha256 == first.image_sha256
            and result.prompt_sha256 == first.prompt_sha256
            and result.wire_sha256 == first.wire_sha256
            and result.asset_catalog_sha256 == first.asset_catalog_sha256
            and result.remote_authorization_id == first.remote_authorization_id
            and result.remote_authorization_file_sha256
            == first.remote_authorization_file_sha256
            and result.remote_receipt_file_sha256
            == first.remote_receipt_file_sha256
            and result.remote_receipt_sha256 == first.remote_receipt_sha256
            and result.endpoint == first.endpoint
        )
        if not same_wire:
            raise PortfolioS1FeedbackError("Feedback retry same-wire identity drifted")
    draft = BoundFeedbackArtifactV5.model_construct(
        selection_sha256=selection.selection_sha256,
        control_sha256=control.control_sha256,
        selection_entry_sha256=entry.entry_sha256,
        reservation_sha256=reservation.reservation_sha256,
        attempt_index=reservation.attempt_index,
        global_call_ordinal=reservation.global_call_ordinal,
        previous_artifact_sha256=reservation.previous_artifact_sha256,
        retry_claim_sha256=reservation.retry_claim_sha256,
        retry_claim_ordinal=reservation.retry_claim_ordinal,
        query_id=entry.query_id,
        feedback_packet=source.packet,
        feedback_result=result,
        status=result.status,
        artifact_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"artifact_sha256"})
    return BoundFeedbackArtifactV5.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_feedback_global_retry_claim_v2(
    path: str | Path, claim: FeedbackGlobalRetryClaimV2
) -> Path:
    return atomic_create_file(path, claim.canonical_bytes())


def load_feedback_global_retry_claim_v2(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> FeedbackGlobalRetryClaimV2:
    content = read_stable_regular_file(
        path, label="Feedback global retry claim v2", max_bytes=2 * 1024 * 1024
    )
    if expected_file_sha256 is not None and sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback retry claim v2 file hash mismatch")
    claim = FeedbackGlobalRetryClaimV2.model_validate_json(content, strict=True)
    if claim.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback retry claim v2 is not canonical")
    return claim


def write_feedback_call_reservation_v5(
    path: str | Path, reservation: FeedbackCallReservationV5
) -> Path:
    return atomic_create_file(path, reservation.canonical_bytes())


def load_feedback_call_reservation_v5(path: str | Path) -> FeedbackCallReservationV5:
    content = read_stable_regular_file(
        path, label="Feedback call reservation v5", max_bytes=2 * 1024 * 1024
    )
    reservation = FeedbackCallReservationV5.model_validate_json(content, strict=True)
    if reservation.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback reservation v5 is not canonical")
    return reservation


def write_bound_feedback_artifact_v5(
    path: str | Path, artifact: BoundFeedbackArtifactV5
) -> Path:
    return atomic_create_file(path, artifact.canonical_bytes())


def load_bound_feedback_artifact_v5(path: str | Path) -> BoundFeedbackArtifactV5:
    content = read_stable_regular_file(
        path, label="bound S1 Feedback artifact v5", max_bytes=8 * 1024 * 1024
    )
    artifact = BoundFeedbackArtifactV5.model_validate_json(content, strict=True)
    if artifact.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("bound Feedback artifact v5 is not canonical")
    return artifact


class PortfolioS1FeedbackRunAttemptV5(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=243)
    artifact_sha256: Sha256 | None = None
    reservation_sha256: Sha256
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    error_code: str | None = None
    request_id: str | None = None
    wire_sha256: Sha256 | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    actual_cost_cny: str | None = None

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        result_fields = (
            self.artifact_sha256,
            self.request_id,
            self.wire_sha256,
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.reasoning_bytes,
            self.actual_cost_cny,
        )
        if self.status == "orphan" and any(value is not None for value in result_fields):
            raise ValueError("Feedback run v5 orphan cannot claim a result")
        if self.status != "orphan" and self.artifact_sha256 is None:
            raise ValueError("settled Feedback run v5 attempt lacks an artifact")
        if self.status == "parsed" and self.error_code is not None:
            raise ValueError("parsed Feedback run v5 attempt claims an error")
        if (self.attempt_index == 2) != (self.retry_claim_ordinal is not None):
            raise ValueError("Feedback run v5 retry claim ordinal drifted")
        if self.actual_cost_cny is not None:
            try:
                if Decimal(self.actual_cost_cny) < 0:
                    raise ValueError
            except Exception as error:
                raise ValueError("Feedback run v5 cost is not canonical") from error
        return self


class PortfolioS1FeedbackRunV5(_StrictFrozenModel):
    schema_version: Literal[5] = 5
    kind: Literal["portfolio-s1-feedback-run"] = "portfolio-s1-feedback-run"
    policy_version: Literal["portfolio-s1-feedback-run-v5"] = (
        S1_FEEDBACK_RUN_POLICY_VERSION_V5
    )
    selection_sha256: Sha256
    control_sha256: Sha256
    expected_count: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = FEEDBACK_PHASE_COUNTS_V2
    attempted_count: int = Field(ge=1, le=240)
    parsed_count: int = Field(ge=0, le=240)
    error_count: int = Field(ge=0, le=240)
    orphan_count: int = Field(ge=0, le=243)
    provider_calls_reserved: int = Field(ge=1, le=243)
    retry_count: int = Field(ge=0, le=3)
    retry_claim_sha256s: tuple[Sha256, ...]
    terminal_phase_count: Literal[12, 60, 120, 240]
    status: Literal[
        "stopped_nonparsed", "stopped_orphan", "stopped_budget", "completed"
    ]
    terminal_reason: Literal[
        "usage_limit_exceeded", "accountable_cost_exceeded"
    ] | None = None
    usage_known_count: int = Field(ge=0, le=243)
    usage_unknown_count: int = Field(ge=0, le=243)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    actual_cost_cny: str
    artifacts: tuple[PortfolioS1FeedbackRunAttemptV5, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator(
        "phase_counts", "retry_claim_sha256s", "artifacts", mode="before"
    )
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        keys = tuple(
            (item.selection_ordinal, item.attempt_index) for item in self.artifacts
        )
        ordinals = tuple(item.global_call_ordinal for item in self.artifacts)
        final_by_entry = {item.selection_entry_sha256: item for item in self.artifacts}
        statuses = Counter(item.status for item in final_by_entry.values())
        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or keys != tuple(sorted(set(keys)))
            or tuple(sorted(ordinals))
            != tuple(range(1, self.provider_calls_reserved + 1))
            or self.provider_calls_reserved != len(self.artifacts)
            or self.attempted_count != len(final_by_entry)
            or self.parsed_count != statuses["parsed"]
            or self.error_count != self.attempted_count - self.parsed_count
            or self.orphan_count
            != sum(item.status == "orphan" for item in self.artifacts)
            or self.retry_count
            != sum(item.attempt_index == 2 for item in self.artifacts)
            or len(self.retry_claim_sha256s) != self.retry_count
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count
            != self.provider_calls_reserved - self.usage_known_count
            or self.input_tokens != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
        ):
            raise ValueError("S1 Feedback run v5 counts drifted")
        if self.status == "completed":
            if (
                self.terminal_reason is not None
                or self.attempted_count != 240
                or self.parsed_count != 240
                or tuple(
                    sorted(item.selection_ordinal for item in final_by_entry.values())
                )
                != tuple(range(1, 241))
            ):
                raise ValueError("completed S1 Feedback run v5 is not parsed240")
        elif self.status == "stopped_orphan":
            if self.orphan_count < 1:
                raise ValueError("stopped-orphan run v5 lacks an orphan")
        elif self.status == "stopped_budget":
            if self.terminal_reason is None or self.orphan_count:
                raise ValueError("stopped-budget run v5 reason drifted")
        elif (
            self.error_count < 1
            or self.orphan_count
            or self.terminal_reason is not None
        ):
            raise ValueError("stopped-nonparsed run v5 status drifted")
        expected_cost = sum(
            (Decimal(item.actual_cost_cny) for item in self.artifacts if item.actual_cost_cny),
            Decimal("0"),
        )
        if Decimal(self.actual_cost_cny) != expected_cost:
            raise ValueError("Feedback run v5 actual cost drifted")
        if self.artifact_set_sha256 != _hash_payload(
            [item.model_dump(mode="json") for item in self.artifacts]
        ):
            raise ValueError("Feedback run v5 artifact set hash drifted")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Feedback run v5 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _actual_cost(usage: LLMUsage | None) -> str | None:
    if usage is None:
        return None
    value = (
        Decimal(usage.input_tokens) * Decimal(12)
        + Decimal(usage.output_tokens) * Decimal(36)
    ) / Decimal(1_000_000)
    return format(value.quantize(Decimal("0.000000000001")), "f")


def build_portfolio_s1_feedback_run_v5(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    artifacts: tuple[BoundFeedbackArtifactV5, ...],
    *,
    retry_claims: tuple[FeedbackGlobalRetryClaimV2, ...] = (),
    orphaned_attempts: tuple[FeedbackCallReservationV5, ...] = (),
    terminal_reason: Literal[
        "usage_limit_exceeded", "accountable_cost_exceeded"
    ] | None = None,
) -> PortfolioS1FeedbackRunV5:
    if control.selection_sha256 != selection.selection_sha256:
        raise PortfolioS1FeedbackError("Feedback run v5 control binding drifted")
    entry_by_sha = {item.entry_sha256: item for item in selection.entries}
    artifact_by_key: dict[tuple[str, int], BoundFeedbackArtifactV5] = {}
    for artifact in artifacts:
        key = (artifact.selection_entry_sha256, artifact.attempt_index)
        entry = entry_by_sha.get(artifact.selection_entry_sha256)
        if (
            type(artifact) is not BoundFeedbackArtifactV5
            or entry is None
            or key in artifact_by_key
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != entry.query_id
        ):
            raise PortfolioS1FeedbackError("Feedback run v5 artifacts drifted")
        artifact_by_key[key] = artifact
    orphan_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item
        for item in orphaned_attempts
    }
    if (
        len(orphan_by_key) != len(orphaned_attempts)
        or set(orphan_by_key) & set(artifact_by_key)
        or any(
            type(item) is not FeedbackCallReservationV5
            or item.selection_sha256 != selection.selection_sha256
            or item.control_sha256 != control.control_sha256
            or item.selection_entry_sha256 not in entry_by_sha
            for item in orphaned_attempts
        )
    ):
        raise PortfolioS1FeedbackError("Feedback run v5 orphan identity drifted")
    first_artifacts = tuple(
        item for item in artifacts if item.attempt_index == 1
    )
    validate_feedback_global_retry_claim_set_v2(
        selection, control, retry_claims, first_artifacts
    )
    claim_by_entry = {item.selection_entry_sha256: item for item in retry_claims}
    retry_keys = {
        key for key in (*artifact_by_key, *orphan_by_key) if key[1] == 2
    }
    if len(retry_keys) > 3 or len(retry_keys) != len(retry_claims):
        raise PortfolioS1FeedbackError("Feedback run v5 retry/claim count drifted")
    for entry_sha, _attempt in retry_keys:
        first = artifact_by_key.get((entry_sha, 1))
        second = artifact_by_key.get((entry_sha, 2))
        orphan = orphan_by_key.get((entry_sha, 2))
        claim = claim_by_entry.get(entry_sha)
        retry_item = second or orphan
        if (
            first is None
            or claim is None
            or retry_item is None
            or retry_item.previous_artifact_sha256 != first.artifact_sha256
            or retry_item.retry_claim_sha256 != claim.claim_sha256
            or retry_item.retry_claim_ordinal != claim.claim_ordinal
            or retry_item.global_call_ordinal <= first.global_call_ordinal
            or not is_qwen38_schema_or_length_retry_eligible(first.feedback_result)
        ):
            raise PortfolioS1FeedbackError("Feedback run v5 retry ancestry drifted")
    rows: list[PortfolioS1FeedbackRunAttemptV5] = []
    for entry in selection.entries:
        for attempt in (1, 2):
            artifact = artifact_by_key.get((entry.entry_sha256, attempt))
            orphan = orphan_by_key.get((entry.entry_sha256, attempt))
            if artifact is None and orphan is None:
                continue
            if artifact is None:
                assert orphan is not None
                rows.append(
                    PortfolioS1FeedbackRunAttemptV5(
                        selection_ordinal=entry.selection_ordinal,
                        selection_entry_sha256=entry.entry_sha256,
                        attempt_index=attempt,
                        global_call_ordinal=orphan.global_call_ordinal,
                        reservation_sha256=orphan.reservation_sha256,
                        retry_claim_ordinal=orphan.retry_claim_ordinal,
                        status="orphan",
                    )
                )
                continue
            usage = artifact.feedback_result.usage
            rows.append(
                PortfolioS1FeedbackRunAttemptV5(
                    selection_ordinal=entry.selection_ordinal,
                    selection_entry_sha256=entry.entry_sha256,
                    attempt_index=attempt,
                    global_call_ordinal=artifact.global_call_ordinal,
                    artifact_sha256=artifact.artifact_sha256,
                    reservation_sha256=artifact.reservation_sha256,
                    retry_claim_ordinal=artifact.retry_claim_ordinal,
                    status=artifact.status,
                    error_code=artifact.feedback_result.error_code,
                    request_id=artifact.feedback_result.request_id,
                    wire_sha256=artifact.feedback_result.wire_sha256,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                    reasoning_tokens=artifact.feedback_result.reasoning_tokens,
                    reasoning_bytes=artifact.feedback_result.reasoning_bytes,
                    actual_cost_cny=_actual_cost(usage),
                )
            )
    if not rows:
        raise PortfolioS1FeedbackError("Feedback run v5 has no attempts")
    rows.sort(key=lambda item: (item.selection_ordinal, item.attempt_index))
    final_by_entry = {item.selection_entry_sha256: item for item in rows}
    errors = [item for item in final_by_entry.values() if item.status != "parsed"]
    if not errors and len(final_by_entry) != 240 and terminal_reason is None:
        raise PortfolioS1FeedbackError(
            "nonterminal parsed Feedback v5 artifacts cannot publish a run"
        )
    terminal_phase = next(
        count
        for count in FEEDBACK_PHASE_COUNTS_V2
        if max(item.selection_ordinal for item in rows) <= count
    )
    status = (
        "completed"
        if not errors and terminal_reason is None
        else "stopped_orphan"
        if any(item.status == "orphan" for item in rows)
        else "stopped_budget"
        if terminal_reason is not None
        else "stopped_nonparsed"
    )
    total_cost = sum(
        (Decimal(item.actual_cost_cny) for item in rows if item.actual_cost_cny),
        Decimal("0"),
    )
    unsigned = {
        "schema_version": 5,
        "kind": "portfolio-s1-feedback-run",
        "policy_version": S1_FEEDBACK_RUN_POLICY_VERSION_V5,
        "selection_sha256": selection.selection_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 240,
        "phase_counts": FEEDBACK_PHASE_COUNTS_V2,
        "attempted_count": len(final_by_entry),
        "parsed_count": sum(item.status == "parsed" for item in final_by_entry.values()),
        "error_count": len(errors),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "provider_calls_reserved": len(rows),
        "retry_count": sum(item.attempt_index == 2 for item in rows),
        "retry_claim_sha256s": tuple(item.claim_sha256 for item in retry_claims),
        "terminal_phase_count": terminal_phase,
        "status": status,
        "terminal_reason": terminal_reason,
        "usage_known_count": sum(item.input_tokens is not None for item in rows),
        "usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "actual_cost_cny": format(total_cost.quantize(Decimal("0.000000000001")), "f"),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRunV5.model_validate(
        {**unsigned, "run_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_portfolio_s1_feedback_run_v5(
    path: str | Path, run: PortfolioS1FeedbackRunV5
) -> Path:
    return atomic_create_file(path, run.canonical_bytes())


def load_portfolio_s1_feedback_run_v5(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackRunV5:
    content = read_stable_regular_file(
        path, label="Feedback run v5", max_bytes=32 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback run v5 file hash mismatch")
    run = PortfolioS1FeedbackRunV5.model_validate_json(content, strict=True)
    if run.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback run v5 is not canonical")
    return run


def require_feedback_v12_creator_projection_privacy(
    result: FeedbackEvaluationResult,
    *,
    private_query_ids: tuple[str, ...],
) -> FeedbackEvaluationResult:
    if (
        type(result) is not FeedbackEvaluationResult
        or result.schema_version != 6
        or result.cache_namespace != "feedback-evaluator-v12"
        or result.model != "qwen3.8-max"
        or result.status != "parsed"
        or result.parsed_feedback is None
    ):
        raise PortfolioS1FeedbackError(
            "Creator projection privacy requires one parsed Feedback V12 result"
        )
    projection = result.parsed_feedback.model_dump(mode="json")
    _validate_model_projection_privacy(
        projection, private_query_ids=private_query_ids
    )
    if any(
        _CREATOR_RUNTIME_HANDLE_RE.search(text)
        for text in _iter_string_values(projection)
    ):
        raise PortfolioS1FeedbackError(
            "S1 Feedback Creator projection contains private runtime metadata"
        )
    return result


class PortfolioS1FeedbackBundleV8(PortfolioS1FeedbackBundleV7):
    schema_version: Literal[8] = 8
    policy_version: Literal["portfolio-s1-feedback-bundle-v8"] = (
        S1_FEEDBACK_BUNDLE_POLICY_VERSION_V8
    )
    provider_call_count: Literal[240, 241, 242, 243]


def build_portfolio_s1_feedback_bundle_v8(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackControlV12,
    authorization: PortfolioS1FeedbackAuthorizationV7,
    artifacts: tuple[BoundFeedbackArtifactV5, ...],
    run: PortfolioS1FeedbackRunV5,
    *,
    retry_claims: tuple[FeedbackGlobalRetryClaimV2, ...] = (),
) -> PortfolioS1FeedbackBundleV8:
    validate_portfolio_s1_feedback_control_v12(control, selection, authorization)
    if len(artifacts) not in {240, 241, 242, 243}:
        raise PortfolioS1FeedbackError(
            "Feedback bundle v8 requires parsed240 plus at most three first failures"
        )
    final_by_entry: dict[str, BoundFeedbackArtifactV5] = {}
    for artifact in artifacts:
        if type(artifact) is not BoundFeedbackArtifactV5:
            raise PortfolioS1FeedbackError("Feedback bundle v8 artifact type drifted")
        previous = final_by_entry.get(artifact.selection_entry_sha256)
        if previous is None or artifact.attempt_index > previous.attempt_index:
            final_by_entry[artifact.selection_entry_sha256] = artifact
    if len(final_by_entry) != 240:
        raise PortfolioS1FeedbackError("Feedback bundle v8 requires exact240 entries")
    expected_run = build_portfolio_s1_feedback_run_v5(
        selection, control, artifacts, retry_claims=retry_claims
    )
    if run != expected_run or run.status != "completed":
        raise PortfolioS1FeedbackError("Feedback bundle v8 requires completed parsed240")
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
            or result.schema_version != 6
            or result.cache_namespace != "feedback-evaluator-v12"
            or result.model != "qwen3.8-max"
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError("Feedback bundle v8 artifact binding drifted")
        require_feedback_v12_creator_projection_privacy(
            result, private_query_ids=selected_query_ids
        )
        assert feedback is not None
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("Feedback bundle v8 suggestions repeat")
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
            raise PortfolioS1FeedbackError("Feedback bundle v8 GCS contract drifted")
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
    _validate_model_projection_privacy(
        projection, private_query_ids=selected_query_ids
    )
    unsigned = {
        "schema_version": 8,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": S1_FEEDBACK_BUNDLE_POLICY_VERSION_V8,
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
    return PortfolioS1FeedbackBundleV8.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_portfolio_s1_feedback_bundle_v8(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV8
) -> Path:
    return atomic_create_file(path, bundle.canonical_bytes())


def load_portfolio_s1_feedback_bundle_v8(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV8:
    content = read_stable_regular_file(
        path, label="Portfolio S1 Feedback bundle v8", max_bytes=128 * 1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError("Feedback bundle v8 file hash mismatch")
    bundle = PortfolioS1FeedbackBundleV8.model_validate_json(content, strict=True)
    if bundle.canonical_bytes() != content:
        raise PortfolioS1FeedbackError("Feedback bundle v8 is not canonical")
    return bundle


class PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7(
    PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV6
):
    schema_version: Literal[7] = 7
    policy_version: Literal[
        "portfolio-s1-feedback-selected240-remote-runtime-v7"
    ] = SELECTED_QWEN38_FEEDBACK_REMOTE_POLICY_VERSION_V7


def prepare_selected_qwen38_feedback_remote_runtime_v7(
    corpus: VerifiedStaticGCSCorpus,
    selection: PortfolioS1FeedbackSelectionV2,
    authorization: PortfolioS1FeedbackAuthorizationV7,
    control: PortfolioS1FeedbackControlV12,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create/resume only the fresh-v3 exact240 remote membership receipt."""

    validate_portfolio_s1_feedback_control_v12(control, selection, authorization)
    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    if (
        authorization.parent_remote_authorization_id,
        authorization.parent_remote_authorization_file_sha256,
        authorization.parent_remote_receipt_file_sha256,
        authorization.parent_remote_receipt_sha256,
        authorization.parent_remote_catalog_sha256,
    ) != (
        parent.authorization.authorization_id,
        parent.authorization_file_sha256,
        parent.receipt_file_sha256,
        parent.receipt.receipt_sha256,
        parent.catalog.catalog_sha256,
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "fresh-v3 remote authorization parent drifted"
        )
    if (
        len(verified_sources) != 240
        or tuple(item.selection_entry_sha256 for item in verified_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "fresh-v3 remote sources differ from selected240"
        )
    receipt_bindings: list[SelectedFeedbackRemoteBindingV2] = []
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, verified_sources, strict=True):
        require_verified_static_feedback_source_v2(source, selection, control, entry)
        resolution = parent.catalog.verify_reference(
            entry.asset_id,
            source.row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
        ):
            raise SelectedFeedbackRemoteRuntimeError(
                "fresh-v3 remote image membership drifted"
            )
        receipt_bindings.append(
            SelectedFeedbackRemoteBindingV2(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
        image_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
    payload = [item.model_dump(mode="json") for item in receipt_bindings]
    draft = PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        corpus_sha256=corpus.corpus_sha256,
        catalog_sha256=parent.catalog.catalog_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        selected_bindings=tuple(receipt_bindings),
        selected_binding_set_sha256=_hash_payload(payload),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    expected = PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target, label="fresh-v3 remote receipt", max_bytes=8 * 1024 * 1024
        )
        receipt = PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7.model_validate_json(
            content, strict=True
        )
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "fresh-v3 remote receipt resume conflict"
            )
    else:
        atomic_create_file(target, expected.canonical_bytes())
        receipt = expected
        content = expected.canonical_bytes()
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=receipt,
        catalog=parent.catalog,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        receipt_file_sha256=sha256_bytes(content),
        selected_bindings=tuple(image_bindings),
        processor="dashscope-qwen38-feedback",
        expected_binding_count=240,
    )


def _write_model(path: str | Path, model: BaseModel) -> Path:
    return atomic_create_file(
        path, canonical_json_bytes(model.model_dump(mode="json"))
    )


def _load_model(
    path: str | Path,
    *,
    expected_file_sha256: str,
    model_type: type[BaseModel],
    label: str,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=32 * 1024 * 1024)
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError(f"{label} file hash mismatch")
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical")
    return model


def write_selected_qwen38_feedback_authorization_v7(
    path: str | Path, authorization: PortfolioS1FeedbackAuthorizationV7
) -> Path:
    return _write_model(path, authorization)


def load_selected_qwen38_feedback_authorization_v7(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackAuthorizationV7:
    return _load_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=PortfolioS1FeedbackAuthorizationV7,
        label="Feedback authorization v7",
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_control_v12(
    path: str | Path, control: PortfolioS1FeedbackControlV12
) -> Path:
    return _write_model(path, control)


def load_portfolio_s1_feedback_control_v12(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackControlV12:
    return _load_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=PortfolioS1FeedbackControlV12,
        label="Feedback control v12",
    )  # type: ignore[return-value]


def write_qwen38_feedback_launch_lock_v5(
    path: str | Path, launch: PortfolioS1Qwen38FeedbackLaunchLockV5
) -> Path:
    return _write_model(path, launch)


def load_qwen38_feedback_launch_lock_v5(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1Qwen38FeedbackLaunchLockV5:
    return _load_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=PortfolioS1Qwen38FeedbackLaunchLockV5,
        label="Feedback launch v5",
    )  # type: ignore[return-value]


__all__ = [
    "BOUND_FEEDBACK_POLICY_VERSION_V5",
    "BoundFeedbackArtifactV5",
    "FeedbackCallReservationV5",
    "FeedbackGlobalRetryClaimV2",
    "FRESH_V3_OWNER_BUDGET_STATEMENT",
    "PortfolioS1FeedbackAuthorizationV7",
    "PortfolioS1FeedbackBundleV8",
    "PortfolioS1FeedbackControlV12",
    "PortfolioS1FeedbackRunAttemptV5",
    "PortfolioS1FeedbackRunV5",
    "PortfolioS1Qwen38FeedbackLaunchLockV5",
    "PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2",
    "build_bound_feedback_artifact_v5",
    "build_feedback_call_reservation_v5",
    "build_feedback_global_retry_claim_v2",
    "build_portfolio_s1_feedback_bundle_v8",
    "build_portfolio_s1_feedback_control_v12",
    "build_portfolio_s1_feedback_run_v5",
    "build_qwen38_feedback_launch_lock_v5",
    "build_selected_qwen38_feedback_authorization_v7",
    "is_qwen38_schema_or_length_retry_eligible",
    "load_bound_feedback_artifact_v5",
    "load_feedback_call_reservation_v5",
    "load_feedback_global_retry_claim_v2",
    "load_portfolio_s1_feedback_bundle_v8",
    "load_portfolio_s1_feedback_control_v12",
    "load_portfolio_s1_feedback_run_v5",
    "load_qwen38_feedback_launch_lock_v5",
    "load_selected_qwen38_feedback_authorization_v7",
    "prepare_selected_qwen38_feedback_remote_runtime_v7",
    "qwen38_feedback_global_retry_policy_v2",
    "require_feedback_v12_creator_projection_privacy",
    "validate_feedback_global_retry_claim_set_v2",
    "validate_portfolio_s1_feedback_control_v12",
    "validate_selected_qwen38_feedback_authorization_v7",
    "write_bound_feedback_artifact_v5",
    "write_feedback_call_reservation_v5",
    "write_feedback_global_retry_claim_v2",
    "write_portfolio_s1_feedback_bundle_v8",
    "write_portfolio_s1_feedback_control_v12",
    "write_portfolio_s1_feedback_run_v5",
    "write_qwen38_feedback_launch_lock_v5",
    "write_selected_qwen38_feedback_authorization_v7",
]
