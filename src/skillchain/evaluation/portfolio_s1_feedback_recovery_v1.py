"""Forward-only recovery contracts for the terminal fresh-v3 Feedback run.

The recovery vertical is a new algorithm and transport identity.  It imports
only immutable references to the 73 final parsed parent entries, calls the
provider only for the fixed unresolved identities, and never resumes or
rewrites the terminal parent root.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain import config, llm
from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.evaluator_isolation import (
    EvaluatorIsolationLock,
    build_bound_feedback_prompt_v6,
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
    VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1,
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
    _require_policy_labeled_suggestions,
    visual_feedback_response_format_v1,
)
from skillchain.evaluation.packets import (
    FeedbackPacketV3,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
    evaluator_wire_messages,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    FeedbackGCSContractProjectionV5,
    PortfolioS1FeedbackActionableSuggestionV5,
    PortfolioS1FeedbackBundleEntryV5,
    PortfolioS1FeedbackBundleV5,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackGCSContractSetV5,
    PortfolioS1FeedbackModelEntryV5,
    PortfolioS1FeedbackModelProjectionV5,
    PortfolioS1FeedbackRepresentativeExampleV5,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    _CREATOR_RUNTIME_HANDLE_RE,
    _feedback_v5_coverage,
    _hash_payload,
    _iter_string_values,
    _model_hash,
    _nested_json_model,
    _validate_model_projection_privacy,
    load_portfolio_s1_feedback_selection_v2,
    parse_policy_labeled_feedback_suggestion_v1,
    require_verified_static_feedback_source_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_retry_v3 import (
    BoundFeedbackArtifactV5,
    FeedbackCallReservationV5,
    FeedbackGlobalRetryClaimV2,
    PortfolioS1FeedbackAuthorizationV7,
    PortfolioS1FeedbackControlV12,
    PortfolioS1FeedbackRunV5,
    PortfolioS1Qwen38FeedbackLaunchLockV5,
    PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7,
    load_bound_feedback_artifact_v5,
    load_feedback_call_reservation_v5,
    load_feedback_global_retry_claim_v2,
    load_portfolio_s1_feedback_control_v12,
    load_portfolio_s1_feedback_run_v5,
    load_selected_qwen38_feedback_authorization_v7,
    validate_feedback_global_retry_claim_set_v2,
)
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
)
from skillchain.evaluation.visual_runtime import (
    EvaluatorImageLoadError,
    SelectedFeedbackImageBinding,
    VerifiedSelectedFeedbackRemoteRuntime,
    _make_verified_selected_feedback_remote_runtime,
    contains_encoded_image_echo,
    load_verified_evaluator_image,
)
from skillchain.llm import LLMToolCall, LLMUsage
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RecoveryStatus = Literal["parsed", "parse_error", "provider_error", "timeout"]
RecoveryWireKind = Literal[
    "primary_v7",
    "recovery_v8_stage1",
    "recovery_v8_stage2",
    "round3_primary_json_object_v1",
    "round3_primary_json_schema_v1",
]
RecoveryTriggerKind = Literal[
    "legacy_parent_reasoning_empty_ambiguity",
    "reasoning_empty",
    "strict_parser_failure",
    "length_parser_failure",
]

PARENT_SELECTION_FILE_SHA256 = (
    "dbe400335c3d5f507189e1d787fa47a9518f0c0e8f65a0aeadc9ff8eea7a29ce"
)
PARENT_SELECTION_SHA256 = (
    "ce55b4d47d57a8313aabf014947c85a05150d3a6a4d3338d53548f4a4b983674"
)
PARENT_AUTHORIZATION_FILE_SHA256 = (
    "af59791aa575676c403ecbefa056641baa306cc897bc40e2bf85dacb6347d074"
)
PARENT_AUTHORIZATION_SHA256 = (
    "8bec01caf06239b0062bac92f034f58ffdea340579546bbc171792c13c1f7c08"
)
PARENT_CONTROL_FILE_SHA256 = (
    "f2f6b9fbeee4a2acad619929f932ce89c89b3531e52d453362b722303ebf16aa"
)
PARENT_CONTROL_SHA256 = (
    "a1ac395fcf2037c751316dc72039ebc5ab7ea691716dc156a47e5f5c2460b640"
)
PARENT_RUN_FILE_SHA256 = (
    "40ec6b8a6ce80f23efe988ae1ef8545dbd934c4761a2e933665bfe7adbb46ae8"
)
PARENT_RUN_SHA256 = "c70ab61e80ed8c040b4ff6c991fe408b9a235dcf76ffbe399d608eb511200ebc"
PARENT_ARTIFACT_SET_SHA256 = (
    "a69527efcc5e5bdae8fc441ff7ee7063168b06abaec43bfe45e0b183071e01f1"
)
PARENT_LAUNCH_FILE_SHA256 = (
    "acebb7e9054bdf04da21cb21bcddcd4caf6712627a2dad681212cfd9f44a502c"
)
PARENT_LAUNCH_SHA256 = (
    "d64b3e898a81383139f4c79a1d08752f40380709734de2092d513e6e002b9d66"
)
PARENT_REMOTE_RECEIPT_FILE_SHA256 = (
    "07dc24308aa721605ab7bcde43e794de892fe9dccc8059051acfe1cea63ce8ab"
)
PARENT_REMOTE_RECEIPT_SHA256 = (
    "6e1ef1d04e62fd1885048e3108e4730e23e397e7fb3de75d5c11367acbb63c15"
)
PARENT_TOP_FILE_INVENTORY_SHA256 = (
    "1c45d4ba491b01ad3e74cb9a5e3f6b678654984379e7786246908bf7d99fbc2d"
)
PARENT_BOUND_FILE_INVENTORY_SHA256 = (
    "38b8aa1c7a3bbeb9aefd04bd51a026a6efc8c13b138858c6c7a3b7fb784bd40a"
)
PARENT_RESERVATION_FILE_INVENTORY_SHA256 = (
    "c0d0c7a3450e1d9e741c0cba76f6c95213c84e1da1225b168eaa10160f7a077a"
)
PARENT_CLAIM_FILE_INVENTORY_SHA256 = (
    "a2835014badb55b5c82fefb794b47f9a466abf9f68570993a9b5e75f1815f78d"
)
PARENT_ACTUAL_COST_CNY = "10.747284000000"

IMPORTED_SELECTION_ORDINALS = tuple((*range(1, 73), 74))
UNRESOLVED_SELECTION_ORDINALS = tuple((73, *range(75, 241)))
# A phase boundary is a maximum *selection ordinal*, not a parsed-result count.
# Ordinal 74 is an immutable parsed parent import, so successful recovery of
# ordinal 73 produces 74 parsed entries at the first boundary.
RECOVERY_PHASE_MAX_SELECTION_ORDINALS = (73, 120, 240)
RECOVERY_PHASE_EXPECTED_PARSED_TOTALS = (74, 120, 240)
RECOVERY_GLOBAL_CLAIM_CEILING = 3
RECOVERY_NEW_PROVIDER_CALL_CEILING = 169
RECOVERY_PER_CALL_RESERVATION_CNY = "0.461544000000"
RECOVERY_MAXIMUM_RESERVATION_CNY = "78.000936000000"
RECOVERY_CUMULATIVE_MAXIMUM_CNY = "88.748220000000"
RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY = "89.000000000000"
RECOVERY_OWNER_BUDGET_CEILING_CNY = "150.000000000000"
EMPTY_SHA256 = sha256_bytes(b"")

RECOVERY_PROMPT_POLICY_VERSION_V7 = (
    "visual-feedback-gcs-empty-content-recovery-prompt-v7"
)
RECOVERY_PROMPT_POLICY_VERSION_V8 = (
    "visual-feedback-gcs-strict-parser-recovery-prompt-v8"
)
RECOVERY_TRANSPORT_POLICY_VERSION_V8 = (
    "visual-feedback-qwen38-dashscope-json-object-recovery-v8"
)
ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1 = (
    "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
)
ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1 = (
    "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
)
RECOVERY_ORCHESTRATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-terminal-v3-derived-recovery-v1"
)
RECOVERY_AUTHORIZATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-derived-recovery-authorization-v1"
)
RECOVERY_CONTROL_POLICY_VERSION_V1 = "portfolio-s1-feedback-derived-recovery-control-v1"
RECOVERY_LAUNCH_POLICY_VERSION_V1 = (
    "portfolio-s1-qwen38-feedback-derived-recovery-launch-v1"
)
RECOVERY_PARENT_EVIDENCE_POLICY_VERSION_V1 = "portfolio-s1-feedback-parent-evidence-v1"
RECOVERY_CLAIM_POLICY_VERSION_V1 = "portfolio-s1-feedback-derived-recovery-claim-v1"
RECOVERY_RESERVATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-derived-recovery-reservation-v1"
)
RECOVERY_BOUND_ARTIFACT_POLICY_VERSION_V1 = (
    "portfolio-s1-bound-feedback-derived-recovery-v1"
)
RECOVERY_RUN_POLICY_VERSION_V1 = "portfolio-s1-feedback-derived-recovery-run-v1"
RECOVERY_BUNDLE_POLICY_VERSION_V9 = "portfolio-s1-feedback-bundle-v9"
RECOVERY_OWNER_STATEMENT = (
    "Owner authorized the CNY150 cumulative ceiling; the derived recovery "
    "runtime keeps the stricter CNY89 cumulative technical stop."
)

_RECOVERY_SUFFIX_STAGE1 = (
    " Transport recovery: a prior invocation produced reasoning metadata but no "
    "assistant content, or otherwise failed strict local JSON parsing. "
    "Independently evaluate the same packet. After thinking, place the final "
    "answer in assistant content as exactly one JSON object with the five "
    "output_contract keys. Never return empty assistant content. Do not mention, "
    "infer, or reconstruct prior reasoning."
)
_RECOVERY_SUFFIX_STAGE2 = (
    " Final transport recovery: an earlier recovery response did not satisfy the "
    "strict local response contract. Independently evaluate the same packet and "
    "return the smallest complete JSON object that contains exactly the five "
    "output_contract keys. Use empty arrays where appropriate, keep every "
    "skill_suggestions item policy-prefixed, and emit no Markdown or prose. "
    "Never mention or reconstruct any prior response or reasoning."
)


def _policy_hash(payload: dict[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def feedback_recovery_prompt_policy_v7() -> dict[str, object]:
    return {
        "policy_version": RECOVERY_PROMPT_POLICY_VERSION_V7,
        "parent_prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "parent_prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "system_suffix_sha256": sha256_bytes(_RECOVERY_SUFFIX_STAGE1.encode("utf-8")),
        "rules": [
            "Use the exact same packet and image without prior response text or reasoning.",
            "Emit one nonempty five-key JSON object in assistant content.",
            "Retain parser-v3 and policy-label validation without salvage.",
        ],
    }


def feedback_recovery_prompt_policy_v8() -> dict[str, object]:
    return {
        "policy_version": RECOVERY_PROMPT_POLICY_VERSION_V8,
        "parent_prompt_policy_version": RECOVERY_PROMPT_POLICY_VERSION_V7,
        "parent_prompt_policy_sha256": _policy_hash(
            feedback_recovery_prompt_policy_v7()
        ),
        "system_suffix_sha256": sha256_bytes(_RECOVERY_SUFFIX_STAGE2.encode("utf-8")),
        "rules": [
            "Use only after a stage-one recovery remains retry-eligible.",
            "Use the exact same packet and image without prior response text or reasoning.",
            "Emit the smallest complete five-key JSON object; no salvage.",
        ],
    }


RECOVERY_PROMPT_POLICY_SHA256_V7 = _policy_hash(feedback_recovery_prompt_policy_v7())
RECOVERY_PROMPT_POLICY_SHA256_V8 = _policy_hash(feedback_recovery_prompt_policy_v8())


def feedback_recovery_transport_policy_v8() -> dict[str, object]:
    return {
        "policy_version": RECOVERY_TRANSPORT_POLICY_VERSION_V8,
        "provider": "qwen",
        "model": "qwen3.8-max",
        "response_format": "json_object",
        "stream": False,
        "enable_thinking": True,
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 6144,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_internal_max_attempts": 1,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "rules": [
            "Only a create-only recovery claim may select this wire.",
            "Never include a prior raw response or reasoning transcript.",
            "Provider refusal, privacy failure, input echo, tool call, or provider error is terminal.",
        ],
    }


RECOVERY_TRANSPORT_POLICY_SHA256_V8 = _policy_hash(
    feedback_recovery_transport_policy_v8()
)


def round3_primary_transport_policy_v1() -> dict[str, object]:
    """Fresh Round3 JSON-object wire; it is a primary, never a salvage wire."""

    return {
        "policy_version": ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
        "provider": "qwen",
        "model": "qwen3.8-max",
        "response_format": "json_object",
        "stream": False,
        "enable_thinking": True,
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 6144,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_internal_max_attempts": 1,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "rules": [
            "Use the unchanged FeedbackPacketV3 and base prompt-v6.",
            "This is the first attempt for a fully fresh fixed identity.",
            "Only a separately claimed global retry may issue a second call.",
            "Provider refusal, privacy failure, input echo, or tool call is terminal.",
        ],
    }


ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1 = _policy_hash(
    round3_primary_transport_policy_v1()
)


def round3_primary_json_schema_transport_policy_v1() -> dict[str, object]:
    """Fresh Round3 strict-schema wire after the terminated object canary."""

    return {
        "policy_version": ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
        "provider": "qwen",
        "model": "qwen3.8-max",
        "response_format": "json_schema",
        "json_schema_policy_version": VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1,
        "json_schema_name": VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
        "json_schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
        "json_schema_strict": True,
        "stream": False,
        "enable_thinking": True,
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 6144,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_internal_max_attempts": 1,
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "rules": [
            "Use the unchanged FeedbackPacketV3 and base prompt-v6.",
            "Request the exact strict provider JSON Schema and retain parser-v3 as independent local validation.",
            "This is the first attempt for a wholly fresh fixed identity; import no terminated-canary output.",
            "Only a separately claimed global retry may issue a second call.",
            "Provider refusal, privacy failure, input echo, or tool call is terminal.",
        ],
    }


ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1 = _policy_hash(
    round3_primary_json_schema_transport_policy_v1()
)


def feedback_recovery_orchestration_policy_v1() -> dict[str, object]:
    return {
        "policy_version": RECOVERY_ORCHESTRATION_POLICY_VERSION_V1,
        "parent_run_file_sha256": PARENT_RUN_FILE_SHA256,
        "parent_run_sha256": PARENT_RUN_SHA256,
        "parent_artifact_set_sha256": PARENT_ARTIFACT_SET_SHA256,
        "imported_selection_ordinals": list(IMPORTED_SELECTION_ORDINALS),
        "unresolved_selection_ordinals": list(UNRESOLVED_SELECTION_ORDINALS),
        "phase_max_selection_ordinals": list(RECOVERY_PHASE_MAX_SELECTION_ORDINALS),
        "phase_expected_parsed_totals": list(RECOVERY_PHASE_EXPECTED_PARSED_TOTALS),
        "claim_ceiling": RECOVERY_GLOBAL_CLAIM_CEILING,
        "new_provider_call_ceiling": RECOVERY_NEW_PROVIDER_CALL_CEILING,
        "feedback_concurrency": 1,
        "execution_concurrency": 1,
        "max_active_calls": 1,
        "not_parent_resume": True,
        "concurrency_rationale": (
            "serial-forward-recovery-for-deterministic-earliest-claim-and-fail-fast"
        ),
        "max_lifetime_attempts_per_entry": 3,
        "claim1": "selection73-parent-terminal-lifetime-attempt3",
        "remaining_claims": (
            "selection75-240-ordered-eligible-only-stage1-then-stage2"
        ),
        "primary_wire": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
        "recovery_wire": RECOVERY_TRANSPORT_POLICY_VERSION_V8,
        "retry_identity": "same-selection-entry-no-replacement-first-parsed-wins",
        "persistence_order": [
            "claim-create-only-if-needed",
            "reservation-create-only",
            "provider-invoke-once",
            "settlement-create-only",
        ],
        "orphan_policy": "terminal-no-further-provider-call",
    }


RECOVERY_ORCHESTRATION_POLICY_SHA256_V1 = _policy_hash(
    feedback_recovery_orchestration_policy_v1()
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RecoveryFeedbackEvaluationResultV1(_StrictFrozenModel):
    """One schema-v7 recovery-root provider result with refusal observability."""

    schema_version: Literal[7] = 7
    result_kind: Literal["visual-feedback"] = "visual-feedback"
    cache_namespace: Literal["feedback-evaluator-v13"] = "feedback-evaluator-v13"
    wire_kind: RecoveryWireKind
    parser_policy_version: Literal["visual-feedback-free-text-trim-v3"] = (
        VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
    )
    parser_policy_sha256: Literal[VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3] = (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
    )
    prompt_policy_version: Literal[
        "visual-feedback-gcs-policy-labels-prompt-v6",
        "visual-feedback-gcs-empty-content-recovery-prompt-v7",
        "visual-feedback-gcs-strict-parser-recovery-prompt-v8",
    ]
    prompt_policy_sha256: Sha256
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7",
        "visual-feedback-qwen38-dashscope-json-object-recovery-v8",
        "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1",
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1",
    ]
    transport_policy_sha256: Sha256
    requested_response_format: Literal["json_schema", "json_object"]
    requested_json_schema_sha256: Sha256 | None = None
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = 2048
    requested_timeout_seconds: Literal[600] = 600
    requested_temperature: None = None
    requested_top_p: None = None
    requested_stream: Literal[False] = False
    formal_eligible: Literal[False] = False
    query_id: str
    packet_sha256: Sha256
    prompt_sha256: Sha256
    image_sha256: Sha256
    wire_sha256: Sha256
    asset_catalog_sha256: Sha256
    remote_authorization_id: str
    remote_authorization_file_sha256: Sha256
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    endpoint: str
    max_tokens: None = None
    max_completion_tokens: Literal[6144] = 6144
    attempts: Literal[1] = 1
    max_attempts: Literal[1] = 1
    status: RecoveryStatus
    request_id: str | None = None
    raw_response_text: str | None = None
    raw_response_sha256: Sha256 | None = None
    raw_response_bytes: int | None = Field(default=None, ge=0)
    tool_calls: tuple[LLMToolCall, ...] | None = None
    tool_call_count: int | None = Field(default=None, ge=0)
    response_redaction_reason: (
        Literal["input_image_echo", "creator_projection_privacy"] | None
    ) = None
    parsed_feedback: VisualFeedbackOutput | None = None
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    reasoning_present: bool | None = None
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    reasoning_sha256: Sha256 | None = None
    refusal_present: bool | None = None
    refusal_bytes: int | None = Field(default=None, ge=0)
    refusal_sha256: Sha256 | None = None
    error_code: str | None = None
    result_sha256: Sha256

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _tool_calls_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("parsed_feedback", mode="before")
    @classmethod
    def _feedback(cls, value: object) -> object:
        return _nested_json_model(value, VisualFeedbackOutput)

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        primary = self.wire_kind == "primary_v7"
        round3_primary = self.wire_kind == "round3_primary_json_object_v1"
        round3_schema_primary = self.wire_kind == "round3_primary_json_schema_v1"
        expected_prompt = (
            (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
            )
            if primary or round3_primary or round3_schema_primary
            else (
                RECOVERY_PROMPT_POLICY_VERSION_V7,
                RECOVERY_PROMPT_POLICY_SHA256_V7,
            )
            if self.wire_kind == "recovery_v8_stage1"
            else (
                RECOVERY_PROMPT_POLICY_VERSION_V8,
                RECOVERY_PROMPT_POLICY_SHA256_V8,
            )
        )
        expected_transport = (
            (
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
                "json_schema",
                VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            )
            if primary
            else (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1,
                "json_schema",
                VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            )
            if round3_schema_primary
            else (
                ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
                ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
                "json_object",
                None,
            )
            if round3_primary
            else (
                RECOVERY_TRANSPORT_POLICY_VERSION_V8,
                RECOVERY_TRANSPORT_POLICY_SHA256_V8,
                "json_object",
                None,
            )
        )
        expected_result_identity = (
            (8, "feedback-evaluator-v14")
            if round3_schema_primary
            else (7, "feedback-evaluator-v13")
        )
        if (
            (self.schema_version, self.cache_namespace) != expected_result_identity
            or
            (self.prompt_policy_version, self.prompt_policy_sha256) != expected_prompt
            or (
                self.transport_policy_version,
                self.transport_policy_sha256,
                self.requested_response_format,
                self.requested_json_schema_sha256,
            )
            != expected_transport
            or self.endpoint != config.PROVIDER_ENDPOINTS["qwen"]
        ):
            raise ValueError("Feedback recovery result wire identity drifted")
        has_response = self.request_id is not None
        receipt_fields = (
            self.raw_response_sha256,
            self.raw_response_bytes,
            self.tool_call_count,
            self.usage,
            self.finish_reason,
            self.latency_ms,
            self.reasoning_present,
            self.reasoning_bytes,
            self.refusal_present,
            self.refusal_bytes,
            self.refusal_sha256,
        )
        if has_response:
            if any(item is None for item in receipt_fields):
                raise ValueError("Feedback recovery response receipt is incomplete")
            if self.response_redaction_reason is None:
                if self.raw_response_text is None or self.tool_calls is None:
                    raise ValueError("Feedback recovery response content is incomplete")
                if (
                    sha256_bytes(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_sha256
                    or len(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_bytes
                    or len(self.tool_calls) != self.tool_call_count
                ):
                    raise ValueError("Feedback recovery response commitment drifted")
            elif self.raw_response_text is not None or self.tool_calls is not None:
                raise ValueError("redacted Feedback recovery response leaked content")
            if (
                self.refusal_present is not (self.refusal_bytes > 0)
                or (self.refusal_bytes == 0 and self.refusal_sha256 != EMPTY_SHA256)
                or (self.reasoning_bytes > 0) is (self.reasoning_sha256 is None)
                or self.reasoning_present
                is not (self.reasoning_bytes > 0 or bool(self.reasoning_tokens))
            ):
                raise ValueError("Feedback recovery metadata commitments drifted")
        elif any(
            item is not None
            for item in (
                *receipt_fields,
                self.raw_response_text,
                self.tool_calls,
                self.response_redaction_reason,
                self.parsed_feedback,
            )
        ):
            raise ValueError("failed Feedback recovery call contains response fields")
        if self.status == "parsed":
            if (
                not has_response
                or self.finish_reason != "stop"
                or self.tool_calls
                or self.refusal_present
                or self.parsed_feedback is None
                or self.error_code is not None
                or self.response_redaction_reason is not None
                or not self.raw_response_text
            ):
                raise ValueError("parsed Feedback recovery result field set drifted")
            try:
                parsed = parse_visual_feedback_output_v3(self.raw_response_text)
                _require_policy_labeled_suggestions(parsed)
            except EvaluatorOutputParseError as error:
                raise ValueError(
                    "parsed recovery output cannot be reconstructed"
                ) from error
            if parsed != self.parsed_feedback:
                raise ValueError("parsed recovery output differs from raw response")
        elif self.status == "parse_error":
            if (
                not has_response
                or self.parsed_feedback is not None
                or self.error_code
                not in {
                    "invalid_feedback_json",
                    "input_image_echo",
                    "creator_projection_privacy",
                    "provider_refusal",
                }
            ):
                raise ValueError("Feedback recovery parse-error field set drifted")
            if self.error_code == "provider_refusal" and not self.refusal_present:
                raise ValueError("provider-refusal result lacks refusal metadata")
            if self.response_redaction_reason is not None and (
                self.error_code != self.response_redaction_reason
            ):
                raise ValueError("Feedback recovery redaction reason drifted")
            if (
                self.error_code == "invalid_feedback_json"
                and self.finish_reason == "stop"
                and not self.tool_calls
                and not self.refusal_present
            ):
                assert self.raw_response_text is not None
                try:
                    parsed = parse_visual_feedback_output_v3(self.raw_response_text)
                    _require_policy_labeled_suggestions(parsed)
                except EvaluatorOutputParseError:
                    pass
                else:
                    raise ValueError(
                        "parse-error recovery result contains valid output"
                    )
        elif (
            has_response
            or self.parsed_feedback is not None
            or self.error_code != self.status
        ):
            raise ValueError("provider-error recovery result field set drifted")
        if self.result_sha256 != _model_hash(self, "result_sha256"):
            raise ValueError("Feedback recovery result self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class RecoveryFeedbackEvaluationResultV2(RecoveryFeedbackEvaluationResultV1):
    """Forward-only schema-v8 result for strict Round3 JSON-Schema calls."""

    schema_version: Literal[8] = 8
    cache_namespace: Literal["feedback-evaluator-v14"] = "feedback-evaluator-v14"
    wire_kind: Literal["round3_primary_json_schema_v1"] = (
        "round3_primary_json_schema_v1"
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1] = (
        VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
    )


def _make_recovery_result(
    **payload: object,
) -> RecoveryFeedbackEvaluationResultV1 | RecoveryFeedbackEvaluationResultV2:
    result_type = (
        RecoveryFeedbackEvaluationResultV2
        if payload.get("wire_kind") == "round3_primary_json_schema_v1"
        else RecoveryFeedbackEvaluationResultV1
    )
    draft = result_type.model_construct(
        **payload, result_sha256="0" * 64
    )
    unsigned = draft.model_dump(mode="json", exclude={"result_sha256"})
    return result_type.model_validate(
        {**payload, "result_sha256": _hash_payload(unsigned)}, strict=True
    )


def recovery_trigger_kind(
    result: RecoveryFeedbackEvaluationResultV1 | RecoveryFeedbackEvaluationResultV2,
) -> RecoveryTriggerKind | None:
    """Classify only the frozen retryable response failures."""

    if (
        type(result)
        not in {RecoveryFeedbackEvaluationResultV1, RecoveryFeedbackEvaluationResultV2}
        or result.status != "parse_error"
        or result.error_code != "invalid_feedback_json"
        or result.request_id is None
        or result.usage is None
        or result.response_redaction_reason is not None
        or result.refusal_present is not False
        or result.tool_calls
        or result.raw_response_text is None
    ):
        return None
    if result.finish_reason == "length":
        return "length_parser_failure"
    if result.finish_reason != "stop":
        return None
    if not result.raw_response_text:
        # The recovery identity only admits the observed reasoning-without-
        # answer failure.  A wholly empty response with no reasoning evidence
        # is ambiguous (for example, an unobserved refusal) and terminates.
        return "reasoning_empty" if result.reasoning_present else None
    try:
        parsed = parse_visual_feedback_output_v3(result.raw_response_text)
    except EvaluatorOutputParseError:
        return "strict_parser_failure"
    # A syntactically valid output with only a missing/invalid policy label is
    # a policy failure, not a transport/parser failure.  It is terminal.
    try:
        _require_policy_labeled_suggestions(parsed)
    except EvaluatorOutputParseError:
        return None
    return None


def _wire_messages_for_kind(
    packet: FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
    image_bytes: bytes,
    wire_kind: RecoveryWireKind,
) -> tuple[list[dict[str, object]], str]:
    bound = build_bound_feedback_prompt_v6(packet, evaluator_isolation)
    messages: list[dict[str, object]] = evaluator_wire_messages(
        bound, image_bytes=image_bytes
    )
    if wire_kind in {
        "primary_v7",
        "round3_primary_json_object_v1",
        "round3_primary_json_schema_v1",
    }:
        return messages, bound.prompt_sha256
    suffix = (
        _RECOVERY_SUFFIX_STAGE1
        if wire_kind == "recovery_v8_stage1"
        else _RECOVERY_SUFFIX_STAGE2
    )
    if (
        not messages
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise PortfolioS1FeedbackError("Feedback recovery system prompt shape drifted")
    messages[0] = {
        **messages[0],
        "content": str(messages[0]["content"]) + suffix,
    }
    prompt_policy = (
        RECOVERY_PROMPT_POLICY_SHA256_V7
        if wire_kind == "recovery_v8_stage1"
        else RECOVERY_PROMPT_POLICY_SHA256_V8
    )
    return messages, _hash_payload(
        {
            "base_prompt_sha256": bound.prompt_sha256,
            "recovery_prompt_policy_sha256": prompt_policy,
            "messages_without_image_bytes": [
                messages[0],
                {
                    "role": "user",
                    "content_sha256": sha256_bytes(
                        canonical_json_bytes(messages[1]["content"])
                    ),
                },
            ],
        }
    )


def run_visual_feedback_recovery_v1(
    packet: FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
    *,
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime,
    wire_kind: RecoveryWireKind,
    record_usage: bool = True,
) -> RecoveryFeedbackEvaluationResultV1 | RecoveryFeedbackEvaluationResultV2:
    """Invoke one primary-v7 or conditional recovery-v8 provider call."""

    if type(packet) is not FeedbackPacketV3:
        raise TypeError("Feedback recovery requires exact FeedbackPacketV3")
    if type(wire_kind) is not str or wire_kind not in {
        "primary_v7",
        "recovery_v8_stage1",
        "recovery_v8_stage2",
        "round3_primary_json_object_v1",
        "round3_primary_json_schema_v1",
    }:
        raise PortfolioS1FeedbackError("Feedback recovery wire kind is unsupported")
    # Identity validation deliberately precedes image loading and client
    # construction.  A stale/shared evaluator lock must remain a zero-provider,
    # zero-asset-read failure.
    expected_isolation = make_active_portfolio_evaluator_isolation_lock()
    if (
        type(evaluator_isolation) is not EvaluatorIsolationLock
        or evaluator_isolation != expected_isolation
        or evaluator_isolation.shared_model_runtime
        or (
            evaluator_isolation.feedback.provider,
            evaluator_isolation.feedback.model,
            evaluator_isolation.feedback.model_family,
            evaluator_isolation.feedback.endpoint,
        )
        != (
            "qwen",
            "qwen3.8-max",
            "qwen3.8-max",
            config.PROVIDER_ENDPOINTS["qwen"],
        )
    ):
        raise PortfolioS1FeedbackError(
            "Feedback recovery evaluator lock differs from the active isolated role"
        )
    try:
        verified_runtime, image_bytes = load_verified_evaluator_image(
            remote_runtime,
            processor="dashscope-qwen38-feedback",
            image=packet.image,
            query_id=packet.query_id,
        )
    except EvaluatorImageLoadError as error:
        raise PortfolioS1FeedbackError(str(error)) from error
    messages, prompt_sha256 = _wire_messages_for_kind(
        packet, evaluator_isolation, image_bytes, wire_kind
    )
    primary = wire_kind == "primary_v7"
    round3_primary = wire_kind == "round3_primary_json_object_v1"
    round3_schema_primary = wire_kind == "round3_primary_json_schema_v1"
    response_format = (
        visual_feedback_response_format_v1()
        if primary or round3_schema_primary
        else None
    )
    invocation_controls = {
        "enable_thinking": True,
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 6144,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
    }
    requested_response_format: dict[str, object] = (
        response_format.model_dump(mode="json", by_alias=True)
        if response_format is not None
        else {"type": "json_object"}
    )
    wire_sha256 = _hash_payload(
        {
            "messages": messages,
            "response_format": requested_response_format,
            "stream": False,
            "invocation_controls": invocation_controls,
        }
    )
    common: dict[str, object] = {
        "wire_kind": wire_kind,
        "prompt_policy_version": (
            VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
            if primary or round3_primary or round3_schema_primary
            else RECOVERY_PROMPT_POLICY_VERSION_V7
            if wire_kind == "recovery_v8_stage1"
            else RECOVERY_PROMPT_POLICY_VERSION_V8
        ),
        "prompt_policy_sha256": (
            VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
            if primary or round3_primary or round3_schema_primary
            else RECOVERY_PROMPT_POLICY_SHA256_V7
            if wire_kind == "recovery_v8_stage1"
            else RECOVERY_PROMPT_POLICY_SHA256_V8
        ),
        "transport_policy_version": (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
            if primary
            else ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
            if round3_schema_primary
            else ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
            if round3_primary
            else RECOVERY_TRANSPORT_POLICY_VERSION_V8
        ),
        "transport_policy_sha256": (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
            if primary
            else ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
            if round3_schema_primary
            else ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
            if round3_primary
            else RECOVERY_TRANSPORT_POLICY_SHA256_V8
        ),
        "requested_response_format": (
            "json_schema" if primary or round3_schema_primary else "json_object"
        ),
        "requested_json_schema_sha256": (
            VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
            if primary or round3_schema_primary
            else None
        ),
        "query_id": packet.query_id,
        "packet_sha256": packet.packet_sha256,
        "prompt_sha256": prompt_sha256,
        "image_sha256": packet.image.sha256,
        "wire_sha256": wire_sha256,
        "asset_catalog_sha256": verified_runtime.catalog.catalog_sha256,
        "remote_authorization_id": verified_runtime.authorization.authorization_id,
        "remote_authorization_file_sha256": verified_runtime.authorization_file_sha256,
        "remote_receipt_file_sha256": verified_runtime.receipt_file_sha256,
        "remote_receipt_sha256": verified_runtime.receipt.receipt_sha256,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
    }
    try:
        response = llm.chat(
            "qwen",
            messages,
            model="qwen3.8-max",
            temperature=None,
            top_p=None,
            thinking=True,
            thinking_budget=2048,
            max_tokens=None,
            max_completion_tokens=6144,
            json_mode=not (primary or round3_schema_primary),
            response_format=response_format,
            qwen_feedback_recovery_json_object=not (
                primary or round3_schema_primary
            ),
            max_attempts=1,
            record_usage=record_usage,
            timeout_seconds=600,
        )
    except (APITimeoutError, llm.LLMTimeoutError):
        return _make_recovery_result(**common, status="timeout", error_code="timeout")
    except (APIConnectionError, APIStatusError, RateLimitError, llm.LLMContractError):
        return _make_recovery_result(
            **common, status="provider_error", error_code="provider_error"
        )
    raw_text = response.text
    response_receipt = {
        "request_id": response.request_id,
        "raw_response_sha256": sha256_bytes(raw_text.encode("utf-8")),
        "raw_response_bytes": len(raw_text.encode("utf-8")),
        "tool_call_count": len(response.tool_calls),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "latency_ms": response.latency_ms,
        "reasoning_present": response.reasoning_present,
        "reasoning_tokens": response.reasoning_tokens,
        "reasoning_bytes": response.reasoning_bytes,
        "reasoning_sha256": response.reasoning_sha256,
        "refusal_present": response.refusal_present,
        "refusal_bytes": response.refusal_bytes,
        "refusal_sha256": response.refusal_sha256 or EMPTY_SHA256,
    }
    if contains_encoded_image_echo(
        response_text=raw_text,
        tool_argument_texts=(item.arguments_json for item in response.tool_calls),
        image_bytes=image_bytes,
        mime_type=packet.image.mime_type,
    ):
        return _make_recovery_result(
            **common,
            **response_receipt,
            response_redaction_reason="input_image_echo",
            status="parse_error",
            error_code="input_image_echo",
        )
    response_receipt.update(
        {"raw_response_text": raw_text, "tool_calls": response.tool_calls}
    )
    if response.refusal_present:
        return _make_recovery_result(
            **common,
            **response_receipt,
            status="parse_error",
            error_code="provider_refusal",
        )
    try:
        if response.finish_reason != "stop" or response.tool_calls:
            raise EvaluatorOutputParseError(
                "Feedback recovery response did not stop as one text answer"
            )
        parsed = parse_visual_feedback_output_v3(raw_text)
        _require_policy_labeled_suggestions(parsed)
    except EvaluatorOutputParseError:
        return _make_recovery_result(
            **common,
            **response_receipt,
            status="parse_error",
            error_code="invalid_feedback_json",
        )
    return _make_recovery_result(
        **common,
        **response_receipt,
        status="parsed",
        parsed_feedback=parsed,
    )


def run_visual_feedback_round3_primary_v1(
    packet: FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
    *,
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime,
    record_usage: bool = True,
) -> RecoveryFeedbackEvaluationResultV1:
    """Issue one fresh Round3 JSON-object attempt with no internal retry."""

    return run_visual_feedback_recovery_v1(
        packet,
        evaluator_isolation,
        remote_runtime=remote_runtime,
        wire_kind="round3_primary_json_object_v1",
        record_usage=record_usage,
    )


def run_visual_feedback_round3_schema_primary_v1(
    packet: FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
    *,
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime,
    record_usage: bool = True,
) -> RecoveryFeedbackEvaluationResultV2:
    """Issue one fresh strict-schema Round3 attempt with no internal retry."""

    result = run_visual_feedback_recovery_v1(
        packet,
        evaluator_isolation,
        remote_runtime=remote_runtime,
        wire_kind="round3_primary_json_schema_v1",
        record_usage=record_usage,
    )
    if type(result) is not RecoveryFeedbackEvaluationResultV2:
        raise AssertionError("Round3 JSON-Schema runner returned the wrong result type")
    return result


class ParentFeedbackArtifactReferenceV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    attempt_index: Literal[1, 2]
    artifact_sha256: Sha256
    feedback_result_sha256: Sha256


class PortfolioS1FeedbackParentEvidenceReceiptV1(_StrictFrozenModel):
    """Canonical commitment to the immutable terminal fresh-v3 evidence."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-parent-evidence"] = (
        "portfolio-s1-feedback-parent-evidence"
    )
    policy_version: Literal["portfolio-s1-feedback-parent-evidence-v1"] = (
        RECOVERY_PARENT_EVIDENCE_POLICY_VERSION_V1
    )
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    authorization_file_sha256: Literal[PARENT_AUTHORIZATION_FILE_SHA256] = (
        PARENT_AUTHORIZATION_FILE_SHA256
    )
    authorization_sha256: Literal[PARENT_AUTHORIZATION_SHA256] = (
        PARENT_AUTHORIZATION_SHA256
    )
    control_file_sha256: Literal[PARENT_CONTROL_FILE_SHA256] = (
        PARENT_CONTROL_FILE_SHA256
    )
    control_sha256: Literal[PARENT_CONTROL_SHA256] = PARENT_CONTROL_SHA256
    run_file_sha256: Literal[PARENT_RUN_FILE_SHA256] = PARENT_RUN_FILE_SHA256
    run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    artifact_set_sha256: Literal[PARENT_ARTIFACT_SET_SHA256] = (
        PARENT_ARTIFACT_SET_SHA256
    )
    launch_file_sha256: Literal[PARENT_LAUNCH_FILE_SHA256] = PARENT_LAUNCH_FILE_SHA256
    launch_sha256: Literal[PARENT_LAUNCH_SHA256] = PARENT_LAUNCH_SHA256
    remote_receipt_file_sha256: Literal[PARENT_REMOTE_RECEIPT_FILE_SHA256] = (
        PARENT_REMOTE_RECEIPT_FILE_SHA256
    )
    remote_receipt_sha256: Literal[PARENT_REMOTE_RECEIPT_SHA256] = (
        PARENT_REMOTE_RECEIPT_SHA256
    )
    top_file_inventory_sha256: Literal[PARENT_TOP_FILE_INVENTORY_SHA256] = (
        PARENT_TOP_FILE_INVENTORY_SHA256
    )
    bound_file_inventory_sha256: Literal[PARENT_BOUND_FILE_INVENTORY_SHA256] = (
        PARENT_BOUND_FILE_INVENTORY_SHA256
    )
    reservation_file_inventory_sha256: Literal[
        PARENT_RESERVATION_FILE_INVENTORY_SHA256
    ] = PARENT_RESERVATION_FILE_INVENTORY_SHA256
    claim_file_inventory_sha256: Literal[PARENT_CLAIM_FILE_INVENTORY_SHA256] = (
        PARENT_CLAIM_FILE_INVENTORY_SHA256
    )
    parent_actual_cost_cny: Literal[PARENT_ACTUAL_COST_CNY] = PARENT_ACTUAL_COST_CNY
    imported_selection_ordinals: tuple[int, ...] = IMPORTED_SELECTION_ORDINALS
    unresolved_selection_ordinals: tuple[int, ...] = UNRESOLVED_SELECTION_ORDINALS
    imported_artifacts: tuple[ParentFeedbackArtifactReferenceV1, ...]
    terminal_ordinal73_attempts: tuple[ParentFeedbackArtifactReferenceV1, ...]
    bound_artifact_file_set_sha256: Sha256
    reservation_file_set_sha256: Sha256
    parent_retry_claim_set_sha256: Sha256
    evidence_sha256: Sha256

    @field_validator(
        "imported_selection_ordinals",
        "unresolved_selection_ordinals",
        "imported_artifacts",
        "terminal_ordinal73_attempts",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if (
            self.imported_selection_ordinals != IMPORTED_SELECTION_ORDINALS
            or self.unresolved_selection_ordinals != UNRESOLVED_SELECTION_ORDINALS
            or tuple(item.selection_ordinal for item in self.imported_artifacts)
            != IMPORTED_SELECTION_ORDINALS
            or len(self.imported_artifacts) != 73
            or tuple(
                (item.selection_ordinal, item.attempt_index)
                for item in self.terminal_ordinal73_attempts
            )
            != ((73, 1), (73, 2))
        ):
            raise ValueError("parent Feedback evidence membership drifted")
        if self.evidence_sha256 != _model_hash(self, "evidence_sha256"):
            raise ValueError("parent Feedback evidence self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackParentEvidenceV1:
    root: Path
    selection: PortfolioS1FeedbackSelectionV2
    authorization: PortfolioS1FeedbackAuthorizationV7
    control: PortfolioS1FeedbackControlV12
    launch: PortfolioS1Qwen38FeedbackLaunchLockV5
    remote_receipt: PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7
    run: PortfolioS1FeedbackRunV5
    artifacts: tuple[BoundFeedbackArtifactV5, ...]
    reservations: tuple[FeedbackCallReservationV5, ...]
    retry_claims: tuple[FeedbackGlobalRetryClaimV2, ...]
    imported_artifacts: tuple[BoundFeedbackArtifactV5, ...]
    terminal_ordinal73_artifacts: tuple[BoundFeedbackArtifactV5, ...]
    receipt: PortfolioS1FeedbackParentEvidenceReceiptV1


def _file_inventory_sha256(paths: tuple[Path, ...]) -> str:
    return _hash_payload(
        [
            {
                "name": path.name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        path,
                        label=f"frozen parent {path.name}",
                        max_bytes=16 * 1024 * 1024,
                    )
                ),
            }
            for path in paths
        ]
    )


def _exact_json_inventory(
    directory: Path,
    *,
    count: int,
    expected_inventory_sha256: str,
    label: str,
) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} directory is missing or unsafe")
    members = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if len(members) != count or any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in members
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory differs from frozen parent")
    if _file_inventory_sha256(members) != expected_inventory_sha256:
        raise PortfolioS1FeedbackError(f"{label} filenames or bytes drifted")
    return members


def _parent_reference(
    artifact: BoundFeedbackArtifactV5, selection_ordinal: int
) -> ParentFeedbackArtifactReferenceV1:
    return ParentFeedbackArtifactReferenceV1(
        selection_ordinal=selection_ordinal,
        selection_entry_sha256=artifact.selection_entry_sha256,
        attempt_index=artifact.attempt_index,
        artifact_sha256=artifact.artifact_sha256,
        feedback_result_sha256=artifact.feedback_result.result_sha256,
    )


def _make_parent_evidence_receipt_v1(
    *,
    imported: tuple[tuple[int, BoundFeedbackArtifactV5], ...],
    terminal: tuple[BoundFeedbackArtifactV5, ...],
    artifacts: tuple[BoundFeedbackArtifactV5, ...],
    reservations: tuple[FeedbackCallReservationV5, ...],
    claims: tuple[FeedbackGlobalRetryClaimV2, ...],
) -> PortfolioS1FeedbackParentEvidenceReceiptV1:
    artifact_payload = tuple(
        _parent_reference(item, ordinal) for ordinal, item in imported
    )
    terminal_payload = tuple(_parent_reference(item, 73) for item in terminal)
    all_artifact_refs = tuple(
        {
            "selection_entry_sha256": item.selection_entry_sha256,
            "attempt_index": item.attempt_index,
            "artifact_sha256": item.artifact_sha256,
            "feedback_result_sha256": item.feedback_result.result_sha256,
        }
        for item in artifacts
    )
    reservation_refs = tuple(
        {
            "selection_entry_sha256": item.selection_entry_sha256,
            "attempt_index": item.attempt_index,
            "global_call_ordinal": item.global_call_ordinal,
            "reservation_sha256": item.reservation_sha256,
        }
        for item in reservations
    )
    claim_refs = tuple(item.claim_sha256 for item in claims)
    draft = PortfolioS1FeedbackParentEvidenceReceiptV1.model_construct(
        imported_artifacts=artifact_payload,
        terminal_ordinal73_attempts=terminal_payload,
        bound_artifact_file_set_sha256=_hash_payload(all_artifact_refs),
        reservation_file_set_sha256=_hash_payload(reservation_refs),
        parent_retry_claim_set_sha256=_hash_payload(claim_refs),
        evidence_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"evidence_sha256"})
    return PortfolioS1FeedbackParentEvidenceReceiptV1.model_validate(
        {**unsigned, "evidence_sha256": _hash_payload(unsigned)}, strict=True
    )


def load_verified_parent_feedback_evidence_v1(
    parent_root: str | Path,
) -> VerifiedPortfolioS1FeedbackParentEvidenceV1:
    """Load the one frozen terminal v3 root without accepting substitutes."""

    supplied_root = Path(parent_root)
    if supplied_root.is_symlink():
        raise PortfolioS1FeedbackError("parent Feedback root cannot be a symlink")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise PortfolioS1FeedbackError("parent Feedback root is not a safe directory")
    expected_top_names = {
        "authorization-v7.json",
        "control-v12.json",
        "launch-lock-v5.json",
        "remote-runtime-receipt-v7.json",
        "run-v5.json",
        "selection-v2.json",
    }
    top_files = tuple(
        sorted(
            (item for item in root.iterdir() if item.is_file()),
            key=lambda item: item.name,
        )
    )
    top_directories = {
        item.name for item in root.iterdir() if item.is_dir() and not item.is_symlink()
    }
    if (
        {item.name for item in top_files} != expected_top_names
        or top_directories
        != {"bound-feedback-v3", "global-retry-claims-v2", "provider-attempts-v3"}
        or any(item.is_symlink() for item in root.iterdir())
        or _file_inventory_sha256(top_files) != PARENT_TOP_FILE_INVENTORY_SHA256
    ):
        raise PortfolioS1FeedbackError("parent Feedback top-level inventory drifted")
    selection = load_portfolio_s1_feedback_selection_v2(
        root / "selection-v2.json", expected_file_sha256=PARENT_SELECTION_FILE_SHA256
    )
    authorization = load_selected_qwen38_feedback_authorization_v7(
        root / "authorization-v7.json",
        expected_file_sha256=PARENT_AUTHORIZATION_FILE_SHA256,
    )
    control = load_portfolio_s1_feedback_control_v12(
        root / "control-v12.json", expected_file_sha256=PARENT_CONTROL_FILE_SHA256
    )
    run = load_portfolio_s1_feedback_run_v5(
        root / "run-v5.json", expected_file_sha256=PARENT_RUN_FILE_SHA256
    )
    launch_content = read_stable_regular_file(
        root / "launch-lock-v5.json",
        label="parent Feedback launch v5",
        max_bytes=2 * 1024 * 1024,
    )
    remote_content = read_stable_regular_file(
        root / "remote-runtime-receipt-v7.json",
        label="parent Feedback remote receipt v7",
        max_bytes=8 * 1024 * 1024,
    )
    if (
        sha256_bytes(launch_content) != PARENT_LAUNCH_FILE_SHA256
        or sha256_bytes(remote_content) != PARENT_REMOTE_RECEIPT_FILE_SHA256
    ):
        raise PortfolioS1FeedbackError("parent Feedback launch/remote file drifted")
    launch = PortfolioS1Qwen38FeedbackLaunchLockV5.model_validate_json(
        launch_content, strict=True
    )
    remote_receipt = (
        PortfolioS1Qwen38FeedbackRemoteRuntimeReceiptV7.model_validate_json(
            remote_content, strict=True
        )
    )
    if (
        launch.canonical_bytes() != launch_content
        or launch.launch_lock_sha256 != PARENT_LAUNCH_SHA256
        or remote_receipt.canonical_bytes() != remote_content
        or remote_receipt.receipt_sha256 != PARENT_REMOTE_RECEIPT_SHA256
    ):
        raise PortfolioS1FeedbackError("parent Feedback launch/remote identity drifted")
    if (
        selection.selection_sha256 != PARENT_SELECTION_SHA256
        or authorization.authorization_sha256 != PARENT_AUTHORIZATION_SHA256
        or control.control_sha256 != PARENT_CONTROL_SHA256
        or run.run_sha256 != PARENT_RUN_SHA256
        or run.artifact_set_sha256 != PARENT_ARTIFACT_SET_SHA256
        or run.status != "stopped_nonparsed"
        or run.attempted_count != 74
        or run.parsed_count != 73
        or run.error_count != 1
        or run.provider_calls_reserved != 76
        or run.retry_count != 2
        or run.actual_cost_cny != PARENT_ACTUAL_COST_CNY
    ):
        raise PortfolioS1FeedbackError("terminal parent Feedback identity drifted")

    artifact_paths = _exact_json_inventory(
        root / "bound-feedback-v3",
        count=76,
        expected_inventory_sha256=PARENT_BOUND_FILE_INVENTORY_SHA256,
        label="parent bound Feedback",
    )
    reservation_paths = _exact_json_inventory(
        root / "provider-attempts-v3",
        count=76,
        expected_inventory_sha256=PARENT_RESERVATION_FILE_INVENTORY_SHA256,
        label="parent reservation",
    )
    claim_paths = _exact_json_inventory(
        root / "global-retry-claims-v2",
        count=2,
        expected_inventory_sha256=PARENT_CLAIM_FILE_INVENTORY_SHA256,
        label="parent retry claim",
    )
    artifacts = tuple(load_bound_feedback_artifact_v5(path) for path in artifact_paths)
    reservations = tuple(
        load_feedback_call_reservation_v5(path) for path in reservation_paths
    )
    claims = tuple(load_feedback_global_retry_claim_v2(path) for path in claim_paths)
    ordered_claims = tuple(sorted(claims, key=lambda item: item.claim_ordinal))
    validate_feedback_global_retry_claim_set_v2(
        selection,
        control,
        ordered_claims,
        tuple(item for item in artifacts if item.attempt_index == 1),
    )
    if run.retry_claim_sha256s != tuple(item.claim_sha256 for item in ordered_claims):
        raise PortfolioS1FeedbackError("parent Feedback run/claim ancestry drifted")
    artifact_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item for item in artifacts
    }
    reservation_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item for item in reservations
    }
    if len(artifact_by_key) != 76 or len(reservation_by_key) != 76:
        raise PortfolioS1FeedbackError("parent Feedback attempt identities repeat")
    run_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item
        for item in run.artifacts
    }
    if set(artifact_by_key) != set(run_by_key) or set(reservation_by_key) != set(
        run_by_key
    ):
        raise PortfolioS1FeedbackError(
            "parent Feedback attempt inventory is incomplete"
        )
    for key, row in run_by_key.items():
        artifact = artifact_by_key[key]
        reservation = reservation_by_key[key]
        if (
            row.artifact_sha256 != artifact.artifact_sha256
            or row.reservation_sha256 != reservation.reservation_sha256
            or row.global_call_ordinal != artifact.global_call_ordinal
            or row.global_call_ordinal != reservation.global_call_ordinal
            or artifact.reservation_sha256 != reservation.reservation_sha256
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.control_sha256 != control.control_sha256
            or reservation.selection_sha256 != selection.selection_sha256
            or reservation.control_sha256 != control.control_sha256
        ):
            raise PortfolioS1FeedbackError("parent Feedback attempt lineage drifted")
        if key[1] == 2:
            claim = next(
                (
                    item
                    for item in ordered_claims
                    if item.selection_entry_sha256 == key[0]
                ),
                None,
            )
            first = artifact_by_key.get((key[0], 1))
            if (
                claim is None
                or first is None
                or artifact.previous_artifact_sha256 != first.artifact_sha256
                or artifact.retry_claim_sha256 != claim.claim_sha256
                or artifact.retry_claim_ordinal != claim.claim_ordinal
                or reservation.previous_artifact_sha256 != first.artifact_sha256
                or reservation.retry_claim_sha256 != claim.claim_sha256
                or reservation.retry_claim_ordinal != claim.claim_ordinal
            ):
                raise PortfolioS1FeedbackError(
                    "parent Feedback second-attempt claim lineage drifted"
                )

    final_by_entry: dict[str, BoundFeedbackArtifactV5] = {}
    for artifact in artifacts:
        prior = final_by_entry.get(artifact.selection_entry_sha256)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_entry[artifact.selection_entry_sha256] = artifact
    imported_pairs = tuple(
        (ordinal, final_by_entry[selection.entries[ordinal - 1].entry_sha256])
        for ordinal in IMPORTED_SELECTION_ORDINALS
    )
    if any(item.status != "parsed" for _ordinal, item in imported_pairs):
        raise PortfolioS1FeedbackError("parent imported Feedback is not parsed73")
    ordinal73_sha = selection.entries[72].entry_sha256
    terminal = tuple(artifact_by_key[(ordinal73_sha, attempt)] for attempt in (1, 2))
    if any(
        item.status != "parse_error"
        or item.feedback_result.error_code != "invalid_feedback_json"
        or item.feedback_result.finish_reason != "stop"
        or item.feedback_result.raw_response_text != ""
        or item.feedback_result.reasoning_present is not True
        for item in terminal
    ):
        raise PortfolioS1FeedbackError("parent ordinal73 failure evidence drifted")
    receipt = _make_parent_evidence_receipt_v1(
        imported=imported_pairs,
        terminal=terminal,
        artifacts=tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    next(
                        entry.selection_ordinal
                        for entry in selection.entries
                        if entry.entry_sha256 == item.selection_entry_sha256
                    ),
                    item.attempt_index,
                ),
            )
        ),
        reservations=tuple(
            sorted(reservations, key=lambda item: item.global_call_ordinal)
        ),
        claims=ordered_claims,
    )
    return VerifiedPortfolioS1FeedbackParentEvidenceV1(
        root=root,
        selection=selection,
        authorization=authorization,
        control=control,
        launch=launch,
        remote_receipt=remote_receipt,
        run=run,
        artifacts=artifacts,
        reservations=reservations,
        retry_claims=ordered_claims,
        imported_artifacts=tuple(item for _ordinal, item in imported_pairs),
        terminal_ordinal73_artifacts=terminal,
        receipt=receipt,
    )


def load_parent_selected_feedback_remote_runtime_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    verified_sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Bind the frozen v7 receipt read-only; this function never creates a file."""

    if type(parent) is not VerifiedPortfolioS1FeedbackParentEvidenceV1:
        raise TypeError("read-only recovery remote loader requires verified parent")
    remote = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    authorization = parent.authorization
    if (
        authorization.parent_remote_authorization_id,
        authorization.parent_remote_authorization_file_sha256,
        authorization.parent_remote_receipt_file_sha256,
        authorization.parent_remote_receipt_sha256,
        authorization.parent_remote_catalog_sha256,
    ) != (
        remote.authorization.authorization_id,
        remote.authorization_file_sha256,
        remote.receipt_file_sha256,
        remote.receipt.receipt_sha256,
        remote.catalog.catalog_sha256,
    ):
        raise PortfolioS1FeedbackError("read-only recovery remote parent drifted")
    if len(verified_sources) != 240 or tuple(
        item.selection_entry_sha256 for item in verified_sources
    ) != tuple(item.entry_sha256 for item in parent.selection.entries):
        raise PortfolioS1FeedbackError(
            "read-only recovery remote sources differ from selected240"
        )
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source, receipt_binding in zip(
        parent.selection.entries,
        verified_sources,
        parent.remote_receipt.selected_bindings,
        strict=True,
    ):
        require_verified_static_feedback_source_v2(
            source, parent.selection, parent.control, entry
        )
        resolution = remote.catalog.verify_reference(
            entry.asset_id,
            source.row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
            or receipt_binding.selection_ordinal != entry.selection_ordinal
            or receipt_binding.selection_entry_sha256 != entry.entry_sha256
            or receipt_binding.query_id != entry.query_id
            or receipt_binding.asset_id != entry.asset_id
            or receipt_binding.image_sha256 != entry.image_sha256
        ):
            raise PortfolioS1FeedbackError(
                "read-only recovery remote membership drifted"
            )
        image_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=parent.remote_receipt,
        catalog=remote.catalog,
        authorization_file_sha256=PARENT_AUTHORIZATION_FILE_SHA256,
        receipt_file_sha256=PARENT_REMOTE_RECEIPT_FILE_SHA256,
        selected_bindings=tuple(image_bindings),
        processor="dashscope-qwen38-feedback",
        expected_binding_count=240,
    )


def load_selected_qwen38_feedback_remote_runtime_v7(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    verified_sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Conventional public name for the read-only frozen-v7 receipt loader."""

    return load_parent_selected_feedback_remote_runtime_v1(
        parent,
        parent_remote_runtime,
        verified_sources=verified_sources,
    )


class FeedbackRecoveryEntryBindingV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    asset_id: str
    image_sha256: Sha256


class PortfolioS1FeedbackRecoveryAuthorizationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-recovery-authorization"] = (
        "portfolio-s1-feedback-recovery-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-derived-recovery-authorization-v1"
    ] = RECOVERY_AUTHORIZATION_POLICY_VERSION_V1
    status: Literal["owner-approved"] = "owner-approved"
    authorization_id: str
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: Literal[RECOVERY_OWNER_STATEMENT] = RECOVERY_OWNER_STATEMENT
    owner_authorized_budget_ceiling_cny: Literal[RECOVERY_OWNER_BUDGET_CEILING_CNY] = (
        RECOVERY_OWNER_BUDGET_CEILING_CNY
    )
    technical_cumulative_hard_cap_cny: Literal[
        RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
    ] = RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
    parent_actual_cost_cny: Literal[PARENT_ACTUAL_COST_CNY] = PARENT_ACTUAL_COST_CNY
    maximum_new_reservation_cny: Literal[RECOVERY_MAXIMUM_RESERVATION_CNY] = (
        RECOVERY_MAXIMUM_RESERVATION_CNY
    )
    cumulative_maximum_reservation_cny: Literal[RECOVERY_CUMULATIVE_MAXIMUM_CNY] = (
        RECOVERY_CUMULATIVE_MAXIMUM_CNY
    )
    per_call_reservation_cny: Literal[RECOVERY_PER_CALL_RESERVATION_CNY] = (
        RECOVERY_PER_CALL_RESERVATION_CNY
    )
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    parent_authorization_file_sha256: Literal[PARENT_AUTHORIZATION_FILE_SHA256] = (
        PARENT_AUTHORIZATION_FILE_SHA256
    )
    parent_authorization_sha256: Literal[PARENT_AUTHORIZATION_SHA256] = (
        PARENT_AUTHORIZATION_SHA256
    )
    parent_control_file_sha256: Literal[PARENT_CONTROL_FILE_SHA256] = (
        PARENT_CONTROL_FILE_SHA256
    )
    parent_control_sha256: Literal[PARENT_CONTROL_SHA256] = PARENT_CONTROL_SHA256
    parent_run_file_sha256: Literal[PARENT_RUN_FILE_SHA256] = PARENT_RUN_FILE_SHA256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    parent_evidence_sha256: Sha256
    imported_selection_ordinals: tuple[int, ...] = IMPORTED_SELECTION_ORDINALS
    unresolved_selection_ordinals: tuple[int, ...] = UNRESOLVED_SELECTION_ORDINALS
    phase_max_selection_ordinals: tuple[int, ...] = (
        RECOVERY_PHASE_MAX_SELECTION_ORDINALS
    )
    phase_expected_parsed_totals: tuple[int, ...] = (
        RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
    )
    recovery_entries: tuple[FeedbackRecoveryEntryBindingV1, ...]
    recovery_entry_set_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    endpoint: str
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = 2048
    max_completion_tokens: Literal[6144] = 6144
    timeout_seconds: Literal[600] = 600
    provider_internal_max_attempts: Literal[1] = 1
    global_recovery_claim_ceiling: Literal[3] = 3
    new_provider_call_ceiling: Literal[169] = 169
    feedback_concurrency: Literal[1] = 1
    execution_concurrency: Literal[1] = 1
    max_active_calls: Literal[1] = 1
    not_parent_resume: Literal[True] = True
    primary_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7"
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    primary_transport_policy_sha256: Literal[
        VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    ] = VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    recovery_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-object-recovery-v8"
    ] = RECOVERY_TRANSPORT_POLICY_VERSION_V8
    recovery_transport_policy_sha256: Literal[RECOVERY_TRANSPORT_POLICY_SHA256_V8] = (
        RECOVERY_TRANSPORT_POLICY_SHA256_V8
    )
    orchestration_policy_version: Literal[
        "portfolio-s1-feedback-terminal-v3-derived-recovery-v1"
    ] = RECOVERY_ORCHESTRATION_POLICY_VERSION_V1
    orchestration_policy_sha256: Literal[RECOVERY_ORCHESTRATION_POLICY_SHA256_V1] = (
        RECOVERY_ORCHESTRATION_POLICY_SHA256_V1
    )
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    remote_authorization_id: str
    remote_authorization_file_sha256: Sha256
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    remote_catalog_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator(
        "imported_selection_ordinals",
        "unresolved_selection_ordinals",
        "phase_max_selection_ordinals",
        "phase_expected_parsed_totals",
        "recovery_entries",
        mode="before",
    )
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Feedback recovery reviewed_at needs a timezone")
        if (
            self.endpoint != config.PROVIDER_ENDPOINTS["qwen"]
            or self.imported_selection_ordinals != IMPORTED_SELECTION_ORDINALS
            or self.unresolved_selection_ordinals != UNRESOLVED_SELECTION_ORDINALS
            or self.phase_max_selection_ordinals
            != RECOVERY_PHASE_MAX_SELECTION_ORDINALS
            or self.phase_expected_parsed_totals
            != RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
            or tuple(item.selection_ordinal for item in self.recovery_entries)
            != UNRESOLVED_SELECTION_ORDINALS
            or len({item.query_id for item in self.recovery_entries}) != 167
            or len({item.asset_id for item in self.recovery_entries}) != 167
            or self.recovery_entry_set_sha256
            != _hash_payload(
                [item.model_dump(mode="json") for item in self.recovery_entries]
            )
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.technical_cumulative_hard_cap_cny)
            or Decimal(self.technical_cumulative_hard_cap_cny)
            > Decimal(self.owner_authorized_budget_ceiling_cny)
        ):
            raise ValueError("Feedback recovery authorization identity drifted")
        if self.authorization_sha256 != _model_hash(self, "authorization_sha256"):
            raise ValueError("Feedback recovery authorization self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_recovery_authorization_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
) -> PortfolioS1FeedbackRecoveryAuthorizationV1:
    if type(parent) is not VerifiedPortfolioS1FeedbackParentEvidenceV1:
        raise TypeError("Feedback recovery requires verified parent evidence")
    entries = tuple(
        FeedbackRecoveryEntryBindingV1(
            selection_ordinal=ordinal,
            selection_entry_sha256=parent.selection.entries[ordinal - 1].entry_sha256,
            query_id=parent.selection.entries[ordinal - 1].query_id,
            asset_id=parent.selection.entries[ordinal - 1].asset_id,
            image_sha256=parent.selection.entries[ordinal - 1].image_sha256,
        )
        for ordinal in UNRESOLVED_SELECTION_ORDINALS
    )
    parent_authorization = parent.authorization
    draft = PortfolioS1FeedbackRecoveryAuthorizationV1.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        parent_evidence_sha256=parent.receipt.evidence_sha256,
        recovery_entries=entries,
        recovery_entry_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in entries]
        ),
        model_source_lock_file_sha256=(
            parent_authorization.model_source_lock_file_sha256
        ),
        model_source_lock_sha256=parent_authorization.model_source_lock_sha256,
        pricing_lock_file_sha256=parent_authorization.pricing_lock_file_sha256,
        pricing_lock_sha256=parent_authorization.pricing_lock_sha256,
        role_selection_file_sha256=(parent_authorization.role_selection_file_sha256),
        role_selection_sha256=parent_authorization.role_selection_sha256,
        remote_authorization_id=parent_authorization.authorization_id,
        remote_authorization_file_sha256=PARENT_AUTHORIZATION_FILE_SHA256,
        remote_receipt_file_sha256=(
            parent.terminal_ordinal73_artifacts[
                -1
            ].feedback_result.remote_receipt_file_sha256
        ),
        remote_receipt_sha256=(
            parent.terminal_ordinal73_artifacts[
                -1
            ].feedback_result.remote_receipt_sha256
        ),
        remote_catalog_sha256=(
            parent.terminal_ordinal73_artifacts[-1].feedback_result.asset_catalog_sha256
        ),
        authorization_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    strict_payload = draft.model_dump(mode="python", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackRecoveryAuthorizationV1.model_validate(
        {**strict_payload, "authorization_sha256": _hash_payload(unsigned)},
        strict=True,
    )


class PortfolioS1FeedbackRecoveryControlV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-recovery-control"] = (
        "portfolio-s1-feedback-recovery-control"
    )
    policy_version: Literal["portfolio-s1-feedback-derived-recovery-control-v1"] = (
        RECOVERY_CONTROL_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    parent_control_sha256: Literal[PARENT_CONTROL_SHA256] = PARENT_CONTROL_SHA256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    parent_evidence_sha256: Sha256
    authorization_id: str
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    phase_max_selection_ordinals: tuple[int, ...] = (
        RECOVERY_PHASE_MAX_SELECTION_ORDINALS
    )
    phase_expected_parsed_totals: tuple[int, ...] = (
        RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
    )
    imported_count: Literal[73] = 73
    unresolved_count: Literal[167] = 167
    new_provider_call_ceiling: Literal[169] = 169
    global_recovery_claim_ceiling: Literal[3] = 3
    feedback_concurrency: Literal[1] = 1
    execution_concurrency: Literal[1] = 1
    max_active_calls: Literal[1] = 1
    not_parent_resume: Literal[True] = True
    stop_on_nonparsed_or_orphan: Literal[True] = True
    require_all_parsed_for_bundle: Literal[True] = True
    ordinal73_failure_terminal: Literal[True] = True
    parent_rubric_sha256: Sha256
    orchestration_policy_sha256: Literal[RECOVERY_ORCHESTRATION_POLICY_SHA256_V1] = (
        RECOVERY_ORCHESTRATION_POLICY_SHA256_V1
    )
    control_sha256: Sha256

    @field_validator(
        "phase_max_selection_ordinals",
        "phase_expected_parsed_totals",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if (
            self.phase_max_selection_ordinals != RECOVERY_PHASE_MAX_SELECTION_ORDINALS
            or self.phase_expected_parsed_totals
            != RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
            or self.control_sha256 != _model_hash(self, "control_sha256")
        ):
            raise ValueError("Feedback recovery control identity drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_recovery_control_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
) -> PortfolioS1FeedbackRecoveryControlV1:
    if (
        type(parent) is not VerifiedPortfolioS1FeedbackParentEvidenceV1
        or type(authorization) is not PortfolioS1FeedbackRecoveryAuthorizationV1
        or authorization.parent_evidence_sha256 != parent.receipt.evidence_sha256
        or authorization.selection_sha256 != parent.selection.selection_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback recovery control inputs drifted")
    draft = PortfolioS1FeedbackRecoveryControlV1.model_construct(
        parent_evidence_sha256=parent.receipt.evidence_sha256,
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        parent_rubric_sha256=_hash_payload(
            parent.control.rubric.model_dump(mode="json")
        ),
        control_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"control_sha256"})
    return PortfolioS1FeedbackRecoveryControlV1.model_validate(
        {**unsigned, "control_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen38-feedback-recovery-launch-lock"] = (
        "portfolio-s1-qwen38-feedback-recovery-launch-lock"
    )
    policy_version: Literal[
        "portfolio-s1-qwen38-feedback-derived-recovery-launch-v1"
    ] = RECOVERY_LAUNCH_POLICY_VERSION_V1
    run_id: str
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    parent_evidence_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    new_provider_call_ceiling: Literal[169] = 169
    feedback_concurrency: Literal[1] = 1
    execution_concurrency: Literal[1] = 1
    max_active_calls: Literal[1] = 1
    not_parent_resume: Literal[True] = True
    technical_cumulative_hard_cap_cny: Literal[
        RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
    ] = RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
    owner_authorized_budget_ceiling_cny: Literal[RECOVERY_OWNER_BUDGET_CEILING_CNY] = (
        RECOVERY_OWNER_BUDGET_CEILING_CNY
    )
    orchestration_policy_sha256: Literal[RECOVERY_ORCHESTRATION_POLICY_SHA256_V1] = (
        RECOVERY_ORCHESTRATION_POLICY_SHA256_V1
    )
    launch_lock_sha256: Sha256

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.launch_lock_sha256 != _model_hash(self, "launch_lock_sha256"):
            raise ValueError("Feedback recovery launch self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    *,
    run_id: str,
) -> PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1:
    expected_control = build_portfolio_s1_feedback_recovery_control_v1(
        parent, authorization
    )
    if control != expected_control:
        raise PortfolioS1FeedbackError("Feedback recovery launch control drifted")
    draft = PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1.model_construct(
        run_id=run_id,
        parent_evidence_sha256=parent.receipt.evidence_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        control_file_sha256=sha256_bytes(control.canonical_bytes()),
        launch_lock_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_lock_sha256"})
    return PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1.model_validate(
        {**unsigned, "launch_lock_sha256": _hash_payload(unsigned)}, strict=True
    )


class FeedbackRecoveryGlobalClaimV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-recovery-claim"] = (
        "portfolio-s1-feedback-recovery-claim"
    )
    policy_version: Literal["portfolio-s1-feedback-derived-recovery-claim-v1"] = (
        RECOVERY_CLAIM_POLICY_VERSION_V1
    )
    orchestration_policy_sha256: Literal[RECOVERY_ORCHESTRATION_POLICY_SHA256_V1] = (
        RECOVERY_ORCHESTRATION_POLICY_SHA256_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    control_sha256: Sha256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    claim_ordinal: Literal[1, 2, 3]
    previous_claim_sha256: Sha256 | None = None
    trigger_origin: Literal["parent_v3_terminal", "recovery_v1"]
    trigger_new_call_ordinal: int | None = Field(default=None, ge=1, le=169)
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    trigger_artifact_sha256: Sha256
    trigger_feedback_result_sha256: Sha256
    trigger_kind: RecoveryTriggerKind
    trigger_error_code: Literal["invalid_feedback_json"] = "invalid_feedback_json"
    trigger_finish_reason: Literal["stop", "length"]
    lifetime_attempt_index: Literal[2, 3]
    wire_kind: Literal["recovery_v8_stage1", "recovery_v8_stage2"]
    retry_identity: Literal["same-selection-entry-no-replacement"] = (
        "same-selection-entry-no-replacement"
    )
    claim_sha256: Sha256

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        first = self.claim_ordinal == 1
        if first:
            if (
                self.previous_claim_sha256 is not None
                or self.trigger_origin != "parent_v3_terminal"
                or self.trigger_new_call_ordinal is not None
                or self.selection_ordinal != 73
                or self.trigger_kind != "legacy_parent_reasoning_empty_ambiguity"
                or self.trigger_finish_reason != "stop"
                or self.lifetime_attempt_index != 3
                or self.wire_kind != "recovery_v8_stage1"
            ):
                raise ValueError("first recovery claim is not fixed ordinal73")
        elif (
            self.previous_claim_sha256 is None
            or self.trigger_origin != "recovery_v1"
            or self.trigger_new_call_ordinal is None
            or self.selection_ordinal not in UNRESOLVED_SELECTION_ORDINALS[1:]
            or (self.lifetime_attempt_index == 2)
            != (self.wire_kind == "recovery_v8_stage1")
            or (self.lifetime_attempt_index == 3)
            != (self.wire_kind == "recovery_v8_stage2")
        ):
            raise ValueError("ordered recovery claim identity drifted")
        if self.claim_sha256 != _model_hash(self, "claim_sha256"):
            raise ValueError("Feedback recovery claim self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_feedback_recovery_claim_set_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    claims: tuple[FeedbackRecoveryGlobalClaimV1, ...],
    artifacts: tuple["BoundFeedbackRecoveryArtifactV1", ...] = (),
) -> None:
    if len(claims) > RECOVERY_GLOBAL_CLAIM_CEILING:
        raise PortfolioS1FeedbackError("Feedback recovery exceeds three claims")
    artifact_by_sha = {item.artifact_sha256: item for item in artifacts}
    if len(artifact_by_sha) != len(artifacts):
        raise PortfolioS1FeedbackError("Feedback recovery artifacts repeat")
    prior: FeedbackRecoveryGlobalClaimV1 | None = None
    used_trigger_sha256s: set[str] = set()
    for ordinal, claim in enumerate(claims, 1):
        if (
            type(claim) is not FeedbackRecoveryGlobalClaimV1
            or claim.claim_ordinal != ordinal
            or claim.control_sha256 != control.control_sha256
            or claim.selection_sha256 != parent.selection.selection_sha256
            or claim.selection_entry_sha256
            != parent.selection.entries[claim.selection_ordinal - 1].entry_sha256
            or claim.query_id
            != parent.selection.entries[claim.selection_ordinal - 1].query_id
            or claim.previous_claim_sha256
            != (None if prior is None else prior.claim_sha256)
            or claim.claim_sha256 != _model_hash(claim, "claim_sha256")
        ):
            raise PortfolioS1FeedbackError("Feedback recovery claim chain drifted")
        if ordinal == 1:
            trigger = parent.terminal_ordinal73_artifacts[-1]
            if (
                claim.trigger_artifact_sha256 != trigger.artifact_sha256
                or claim.trigger_feedback_result_sha256
                != trigger.feedback_result.result_sha256
            ):
                raise PortfolioS1FeedbackError(
                    "first recovery claim differs from parent ordinal73"
                )
        else:
            trigger = artifact_by_sha.get(claim.trigger_artifact_sha256)
            candidates = tuple(
                sorted(
                    (
                        item
                        for item in artifacts
                        if item.selection_ordinal != 73
                        and item.new_call_ordinal
                        <= (claim.trigger_new_call_ordinal or 0)
                        and item.artifact_sha256 not in used_trigger_sha256s
                        and recovery_trigger_kind(item.feedback_result) is not None
                        and item.lifetime_attempt_index < 3
                    ),
                    key=lambda item: (
                        item.new_call_ordinal,
                        item.selection_ordinal,
                        item.lifetime_attempt_index,
                    ),
                )
            )
            if (
                trigger is None
                or not candidates
                or trigger != candidates[0]
                or claim.trigger_artifact_sha256 in used_trigger_sha256s
                or trigger.feedback_result.result_sha256
                != claim.trigger_feedback_result_sha256
                or trigger.feedback_result.finish_reason != claim.trigger_finish_reason
                or trigger.selection_ordinal != claim.selection_ordinal
                or trigger.new_call_ordinal != claim.trigger_new_call_ordinal
                or trigger.lifetime_attempt_index + 1 != claim.lifetime_attempt_index
                or recovery_trigger_kind(trigger.feedback_result) != claim.trigger_kind
            ):
                raise PortfolioS1FeedbackError(
                    "Feedback recovery claim trigger ancestry drifted"
                )
            used_trigger_sha256s.add(claim.trigger_artifact_sha256)
        prior = claim


def build_feedback_recovery_global_claim_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    *,
    existing_claims: tuple[FeedbackRecoveryGlobalClaimV1, ...] = (),
    existing_artifacts: tuple["BoundFeedbackRecoveryArtifactV1", ...] = (),
    trigger_artifact: "BoundFeedbackRecoveryArtifactV1 | None" = None,
) -> FeedbackRecoveryGlobalClaimV1:
    validate_feedback_recovery_claim_set_v1(
        parent, control, existing_claims, existing_artifacts
    )
    claim_ordinal = len(existing_claims) + 1
    if claim_ordinal > RECOVERY_GLOBAL_CLAIM_CEILING:
        raise PortfolioS1FeedbackError("fourth Feedback recovery claim is forbidden")
    if claim_ordinal == 1:
        if trigger_artifact is not None:
            raise PortfolioS1FeedbackError(
                "fixed first recovery claim has no new trigger"
            )
        entry = parent.selection.entries[72]
        trigger = parent.terminal_ordinal73_artifacts[-1]
        payload: dict[str, object] = {
            "trigger_origin": "parent_v3_terminal",
            "trigger_new_call_ordinal": None,
            "selection_ordinal": 73,
            "selection_entry_sha256": entry.entry_sha256,
            "query_id": entry.query_id,
            "trigger_artifact_sha256": trigger.artifact_sha256,
            "trigger_feedback_result_sha256": trigger.feedback_result.result_sha256,
            "trigger_kind": "legacy_parent_reasoning_empty_ambiguity",
            "trigger_finish_reason": "stop",
            "lifetime_attempt_index": 3,
            "wire_kind": "recovery_v8_stage1",
        }
    else:
        ordered_artifacts = tuple(
            sorted(existing_artifacts, key=lambda item: item.new_call_ordinal)
        )
        if (
            tuple(item.new_call_ordinal for item in ordered_artifacts)
            != tuple(range(1, len(ordered_artifacts) + 1))
            or not ordered_artifacts
            or ordered_artifacts[0].selection_ordinal != 73
            or ordered_artifacts[0].status != "parsed"
        ):
            raise PortfolioS1FeedbackError(
                "Feedback recovery claim lacks the complete settled prefix"
            )
        already_claimed = {item.trigger_artifact_sha256 for item in existing_claims[1:]}
        eligible = tuple(
            sorted(
                (
                    item
                    for item in existing_artifacts
                    if item.selection_ordinal != 73
                    and item.artifact_sha256 not in already_claimed
                    and recovery_trigger_kind(item.feedback_result) is not None
                    and item.lifetime_attempt_index < 3
                ),
                key=lambda item: (
                    item.new_call_ordinal,
                    item.selection_ordinal,
                    item.lifetime_attempt_index,
                ),
            )
        )
        expected_trigger = None if not eligible else eligible[0]
        if (
            type(trigger_artifact) is not BoundFeedbackRecoveryArtifactV1
            or trigger_artifact not in existing_artifacts
            or trigger_artifact.selection_ordinal == 73
            or trigger_artifact != expected_trigger
        ):
            raise PortfolioS1FeedbackError(
                "ordered recovery claim did not select the earliest eligible settlement"
            )
        trigger_kind = recovery_trigger_kind(trigger_artifact.feedback_result)
        lifetime_attempt = trigger_artifact.lifetime_attempt_index + 1
        if trigger_kind is None or lifetime_attempt not in {2, 3}:
            raise PortfolioS1FeedbackError("Feedback recovery trigger is terminal")
        payload = {
            "trigger_origin": "recovery_v1",
            "trigger_new_call_ordinal": trigger_artifact.new_call_ordinal,
            "selection_ordinal": trigger_artifact.selection_ordinal,
            "selection_entry_sha256": trigger_artifact.selection_entry_sha256,
            "query_id": trigger_artifact.query_id,
            "trigger_artifact_sha256": trigger_artifact.artifact_sha256,
            "trigger_feedback_result_sha256": (
                trigger_artifact.feedback_result.result_sha256
            ),
            "trigger_kind": trigger_kind,
            "trigger_finish_reason": trigger_artifact.feedback_result.finish_reason,
            "lifetime_attempt_index": lifetime_attempt,
            "wire_kind": (
                "recovery_v8_stage1" if lifetime_attempt == 2 else "recovery_v8_stage2"
            ),
        }
    draft = FeedbackRecoveryGlobalClaimV1.model_construct(
        selection_sha256=parent.selection.selection_sha256,
        control_sha256=control.control_sha256,
        claim_ordinal=claim_ordinal,
        previous_claim_sha256=(
            None if not existing_claims else existing_claims[-1].claim_sha256
        ),
        **payload,
        claim_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"claim_sha256"})
    claim = FeedbackRecoveryGlobalClaimV1.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(unsigned)}, strict=True
    )
    validate_feedback_recovery_claim_set_v1(
        parent, control, (*existing_claims, claim), existing_artifacts
    )
    return claim


class FeedbackRecoveryCallReservationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-recovery-call-reservation"] = (
        "portfolio-s1-feedback-recovery-call-reservation"
    )
    policy_version: Literal["portfolio-s1-feedback-derived-recovery-reservation-v1"] = (
        RECOVERY_RESERVATION_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    control_sha256: Sha256
    authorization_sha256: Sha256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    new_call_ordinal: int = Field(ge=1, le=169)
    new_attempt_index: Literal[1, 2, 3]
    lifetime_attempt_index: Literal[1, 2, 3]
    wire_kind: RecoveryWireKind
    previous_artifact_sha256: Sha256 | None = None
    recovery_claim_ordinal: Literal[1, 2, 3] | None = None
    recovery_claim_sha256: Sha256 | None = None
    retry_trigger_kind: RecoveryTriggerKind | None = None
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    max_completion_tokens: Literal[6144] = 6144
    provider_internal_max_attempts: Literal[1] = 1
    reservation_cny: Literal[RECOVERY_PER_CALL_RESERVATION_CNY] = (
        RECOVERY_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        ancestry = (
            self.previous_artifact_sha256,
            self.recovery_claim_ordinal,
            self.recovery_claim_sha256,
            self.retry_trigger_kind,
        )
        if self.selection_ordinal == 73:
            if (
                self.new_call_ordinal != 1
                or self.new_attempt_index != 1
                or self.lifetime_attempt_index != 3
                or self.wire_kind != "recovery_v8_stage1"
                or any(item is None for item in ancestry)
                or self.recovery_claim_ordinal != 1
            ):
                raise ValueError("ordinal73 recovery reservation drifted")
        elif self.selection_ordinal not in UNRESOLVED_SELECTION_ORDINALS[1:]:
            raise ValueError("recovery reservation targets an imported identity")
        elif self.lifetime_attempt_index == 1:
            if (
                self.new_attempt_index != 1
                or self.wire_kind != "primary_v7"
                or any(item is not None for item in ancestry)
            ):
                raise ValueError("primary recovery-root reservation has ancestry")
        elif (
            self.new_attempt_index != self.lifetime_attempt_index
            or any(item is None for item in ancestry)
            or (self.lifetime_attempt_index == 2)
            != (self.wire_kind == "recovery_v8_stage1")
            or (self.lifetime_attempt_index == 3)
            != (self.wire_kind == "recovery_v8_stage2")
        ):
            raise ValueError("retry recovery-root reservation ancestry drifted")
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Feedback recovery reservation self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_feedback_recovery_call_reservation_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    source: VerifiedStaticFeedbackSourceV2,
    *,
    new_call_ordinal: int,
    lifetime_attempt_index: Literal[1, 2, 3],
    claim: FeedbackRecoveryGlobalClaimV1 | None = None,
    previous_artifact: "BoundFeedbackRecoveryArtifactV1 | None" = None,
    existing_claims: tuple[FeedbackRecoveryGlobalClaimV1, ...] = (),
    existing_artifacts: tuple["BoundFeedbackRecoveryArtifactV1", ...] = (),
) -> FeedbackRecoveryCallReservationV1:
    if new_call_ordinal > RECOVERY_NEW_PROVIDER_CALL_CEILING:
        raise PortfolioS1FeedbackError("Feedback recovery call ceiling exceeded")
    entry_by_query = {item.query_id: item for item in parent.selection.entries}
    entry = entry_by_query.get(source.packet.query_id)
    if entry is None:
        raise PortfolioS1FeedbackError("Feedback recovery source is not selected")
    require_verified_static_feedback_source_v2(
        source, parent.selection, parent.control, entry
    )
    validate_feedback_recovery_claim_set_v1(
        parent, control, existing_claims, existing_artifacts
    )
    if (
        not existing_claims
        or new_call_ordinal != len(existing_artifacts) + 1
        or tuple(
            item.new_call_ordinal
            for item in sorted(
                existing_artifacts, key=lambda item: item.new_call_ordinal
            )
        )
        != tuple(range(1, len(existing_artifacts) + 1))
    ):
        raise PortfolioS1FeedbackError(
            "Feedback recovery reservation lacks the complete settled prefix"
        )
    if (
        control.authorization_sha256 != authorization.authorization_sha256
        or control.parent_evidence_sha256 != parent.receipt.evidence_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback recovery reservation control drifted")
    if entry.selection_ordinal == 73:
        expected_previous_sha = parent.terminal_ordinal73_artifacts[-1].artifact_sha256
        if (
            claim is None
            or not existing_claims
            or claim != existing_claims[-1]
            or claim.claim_ordinal != 1
            or previous_artifact is not None
            or lifetime_attempt_index != 3
            or claim.selection_entry_sha256 != entry.entry_sha256
        ):
            raise PortfolioS1FeedbackError("ordinal73 requires fixed recovery claim1")
        new_attempt_index = 1
        previous_sha = expected_previous_sha
    elif lifetime_attempt_index == 1:
        if claim is not None or previous_artifact is not None:
            raise PortfolioS1FeedbackError("primary new-root call has retry ancestry")
        new_attempt_index = 1
        previous_sha = None
    else:
        if (
            claim is None
            or not existing_claims
            or claim != existing_claims[-1]
            or type(previous_artifact) is not BoundFeedbackRecoveryArtifactV1
            or previous_artifact not in existing_artifacts
            or claim.selection_entry_sha256 != entry.entry_sha256
            or previous_artifact.selection_entry_sha256 != entry.entry_sha256
            or claim.trigger_artifact_sha256 != previous_artifact.artifact_sha256
            or claim.lifetime_attempt_index != lifetime_attempt_index
        ):
            raise PortfolioS1FeedbackError("recovery retry lacks exact claim ancestry")
        new_attempt_index = lifetime_attempt_index
        previous_sha = previous_artifact.artifact_sha256
    final_by_ordinal = _final_recovery_artifacts_by_ordinal(
        PortfolioS1FeedbackRecoveryLedgerV1(
            claims=existing_claims,
            reservations=(),
            artifacts=existing_artifacts,
            orphaned_reservations=(),
            pending_claims=(),
        )
    )
    required_predecessors = (73, *range(75, entry.selection_ordinal))
    if entry.selection_ordinal != 73 and any(
        ordinal not in final_by_ordinal or final_by_ordinal[ordinal].status != "parsed"
        for ordinal in required_predecessors
    ):
        raise PortfolioS1FeedbackError(
            "Feedback recovery reservation skipped an unresolved predecessor"
        )
    wire_kind: RecoveryWireKind = "primary_v7" if claim is None else claim.wire_kind
    draft = FeedbackRecoveryCallReservationV1.model_construct(
        selection_sha256=parent.selection.selection_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        packet_sha256=source.packet.packet_sha256,
        checkpoint_file_sha256=source.row.checkpoint_file_sha256,
        checkpoint_row_sha256=source.row.checkpoint_row_sha256,
        sidecar_sha256=source.row.sidecar.evidence_sha256,
        new_call_ordinal=new_call_ordinal,
        new_attempt_index=new_attempt_index,
        lifetime_attempt_index=lifetime_attempt_index,
        wire_kind=wire_kind,
        previous_artifact_sha256=previous_sha,
        recovery_claim_ordinal=None if claim is None else claim.claim_ordinal,
        recovery_claim_sha256=None if claim is None else claim.claim_sha256,
        retry_trigger_kind=None if claim is None else claim.trigger_kind,
        reservation_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"reservation_sha256"})
    return FeedbackRecoveryCallReservationV1.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(unsigned)}, strict=True
    )


class BoundFeedbackRecoveryArtifactV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-bound-feedback-recovery"] = (
        "portfolio-s1-bound-feedback-recovery"
    )
    policy_version: Literal["portfolio-s1-bound-feedback-derived-recovery-v1"] = (
        RECOVERY_BOUND_ARTIFACT_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    new_call_ordinal: int = Field(ge=1, le=169)
    new_attempt_index: Literal[1, 2, 3]
    lifetime_attempt_index: Literal[1, 2, 3]
    wire_kind: RecoveryWireKind
    reservation_sha256: Sha256
    previous_artifact_sha256: Sha256 | None = None
    recovery_claim_ordinal: Literal[1, 2, 3] | None = None
    recovery_claim_sha256: Sha256 | None = None
    feedback_packet: FeedbackPacketV3
    feedback_result: RecoveryFeedbackEvaluationResultV1
    status: RecoveryStatus
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, RecoveryFeedbackEvaluationResultV1)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.status != self.status
            or self.feedback_result.wire_kind != self.wire_kind
            or (self.recovery_claim_ordinal is None)
            != (self.recovery_claim_sha256 is None)
            or self.artifact_sha256 != _model_hash(self, "artifact_sha256")
        ):
            raise ValueError("bound Feedback recovery artifact identity drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_bound_feedback_recovery_artifact_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    source: VerifiedStaticFeedbackSourceV2,
    result: RecoveryFeedbackEvaluationResultV1,
    *,
    reservation: FeedbackRecoveryCallReservationV1,
    claim: FeedbackRecoveryGlobalClaimV1 | None = None,
    previous_artifact: BoundFeedbackRecoveryArtifactV1 | None = None,
    existing_claims: tuple[FeedbackRecoveryGlobalClaimV1, ...] = (),
    existing_artifacts: tuple[BoundFeedbackRecoveryArtifactV1, ...] = (),
) -> BoundFeedbackRecoveryArtifactV1:
    if type(result) is not RecoveryFeedbackEvaluationResultV1:
        raise TypeError("bound Feedback recovery requires exact result v1")
    result = redact_recovery_result_for_creator_privacy_v1(
        result,
        private_query_ids=tuple(item.query_id for item in parent.selection.entries),
    )
    expected = build_feedback_recovery_call_reservation_v1(
        parent,
        authorization,
        control,
        source,
        new_call_ordinal=reservation.new_call_ordinal,
        lifetime_attempt_index=reservation.lifetime_attempt_index,
        claim=claim,
        previous_artifact=previous_artifact,
        existing_claims=existing_claims,
        existing_artifacts=existing_artifacts,
    )
    entry = parent.selection.entries[reservation.selection_ordinal - 1]
    if (
        reservation != expected
        or result.query_id != entry.query_id
        or result.packet_sha256 != source.packet.packet_sha256
        or result.image_sha256 != entry.image_sha256
        or result.wire_kind != reservation.wire_kind
        or result.remote_authorization_id != authorization.remote_authorization_id
        or result.remote_authorization_file_sha256
        != authorization.remote_authorization_file_sha256
        or result.remote_receipt_file_sha256 != authorization.remote_receipt_file_sha256
        or result.remote_receipt_sha256 != authorization.remote_receipt_sha256
        or result.asset_catalog_sha256 != authorization.remote_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("Feedback recovery result source drifted")
    draft = BoundFeedbackRecoveryArtifactV1.model_construct(
        selection_sha256=parent.selection.selection_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        new_call_ordinal=reservation.new_call_ordinal,
        new_attempt_index=reservation.new_attempt_index,
        lifetime_attempt_index=reservation.lifetime_attempt_index,
        wire_kind=reservation.wire_kind,
        reservation_sha256=reservation.reservation_sha256,
        previous_artifact_sha256=reservation.previous_artifact_sha256,
        recovery_claim_ordinal=reservation.recovery_claim_ordinal,
        recovery_claim_sha256=reservation.recovery_claim_sha256,
        feedback_packet=source.packet,
        feedback_result=result,
        status=result.status,
        artifact_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"artifact_sha256"})
    return BoundFeedbackRecoveryArtifactV1.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(unsigned)}, strict=True
    )


def redact_recovery_result_for_creator_privacy_v1(
    result: RecoveryFeedbackEvaluationResultV1,
    *,
    private_query_ids: tuple[str, ...],
) -> RecoveryFeedbackEvaluationResultV1:
    """Turn projection leakage into a terminal, committed parse-error result."""

    if result.status != "parsed" or result.parsed_feedback is None:
        return result
    projection = result.parsed_feedback.model_dump(mode="json")
    try:
        _validate_model_projection_privacy(
            projection, private_query_ids=private_query_ids
        )
        if any(
            _CREATOR_RUNTIME_HANDLE_RE.search(text)
            for text in _iter_string_values(projection)
        ):
            raise PortfolioS1FeedbackError(
                "Feedback recovery result contains private runtime metadata"
            )
    except PortfolioS1FeedbackError:
        payload = result.model_dump(
            mode="python",
            exclude={
                "result_sha256",
                "raw_response_text",
                "tool_calls",
                "parsed_feedback",
                "status",
                "error_code",
                "response_redaction_reason",
            },
        )
        return _make_recovery_result(
            **payload,
            status="parse_error",
            error_code="creator_projection_privacy",
            response_redaction_reason="creator_projection_privacy",
        )
    return result


def _write_recovery_model(path: str | Path, model: BaseModel) -> Path:
    return atomic_create_file(path, canonical_json_bytes(model.model_dump(mode="json")))


def _load_recovery_model(
    path: str | Path,
    *,
    model_type: type[BaseModel],
    label: str,
    expected_file_sha256: str | None = None,
    max_bytes: int = 32 * 1024 * 1024,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    if (
        expected_file_sha256 is not None
        and sha256_bytes(content) != expected_file_sha256
    ):
        raise PortfolioS1FeedbackError(f"{label} file hash mismatch")
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical")
    return model


def write_portfolio_s1_feedback_parent_evidence_receipt_v1(
    path: str | Path, receipt: PortfolioS1FeedbackParentEvidenceReceiptV1
) -> Path:
    return _write_recovery_model(path, receipt)


def load_portfolio_s1_feedback_parent_evidence_receipt_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1FeedbackParentEvidenceReceiptV1:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1FeedbackParentEvidenceReceiptV1,
        label="Feedback parent evidence v1",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_recovery_authorization_v1(
    path: str | Path, authorization: PortfolioS1FeedbackRecoveryAuthorizationV1
) -> Path:
    return _write_recovery_model(path, authorization)


def load_portfolio_s1_feedback_recovery_authorization_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1FeedbackRecoveryAuthorizationV1:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1FeedbackRecoveryAuthorizationV1,
        label="Feedback recovery authorization v1",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_recovery_control_v1(
    path: str | Path, control: PortfolioS1FeedbackRecoveryControlV1
) -> Path:
    return _write_recovery_model(path, control)


def load_portfolio_s1_feedback_recovery_control_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1FeedbackRecoveryControlV1:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1FeedbackRecoveryControlV1,
        label="Feedback recovery control v1",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


def write_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
    path: str | Path, launch: PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1
) -> Path:
    return _write_recovery_model(path, launch)


def load_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1,
        label="Feedback recovery launch v1",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


def feedback_recovery_claim_filename_v1(claim_ordinal: int) -> str:
    if claim_ordinal not in {1, 2, 3}:
        raise PortfolioS1FeedbackError("Feedback recovery claim ordinal is invalid")
    return f"claim-{claim_ordinal:02d}.json"


def feedback_recovery_attempt_filename_v1(
    reservation: FeedbackRecoveryCallReservationV1,
) -> str:
    return (
        f"{reservation.new_call_ordinal:04d}-"
        f"{reservation.selection_entry_sha256[:16]}-"
        f"attempt-{reservation.new_attempt_index}.json"
    )


def write_feedback_recovery_global_claim_v1(
    path: str | Path, claim: FeedbackRecoveryGlobalClaimV1
) -> Path:
    if Path(path).name != feedback_recovery_claim_filename_v1(claim.claim_ordinal):
        raise PortfolioS1FeedbackError("Feedback recovery claim filename drifted")
    return _write_recovery_model(path, claim)


def load_feedback_recovery_global_claim_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> FeedbackRecoveryGlobalClaimV1:
    claim = _load_recovery_model(
        path,
        model_type=FeedbackRecoveryGlobalClaimV1,
        label="Feedback recovery claim v1",
        expected_file_sha256=expected_file_sha256,
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(claim, FeedbackRecoveryGlobalClaimV1)
    if Path(path).name != feedback_recovery_claim_filename_v1(claim.claim_ordinal):
        raise PortfolioS1FeedbackError("Feedback recovery claim filename drifted")
    return claim


def write_feedback_recovery_call_reservation_v1(
    path: str | Path, reservation: FeedbackRecoveryCallReservationV1
) -> Path:
    if Path(path).name != feedback_recovery_attempt_filename_v1(reservation):
        raise PortfolioS1FeedbackError("Feedback recovery reservation filename drifted")
    return _write_recovery_model(path, reservation)


def load_feedback_recovery_call_reservation_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> FeedbackRecoveryCallReservationV1:
    reservation = _load_recovery_model(
        path,
        model_type=FeedbackRecoveryCallReservationV1,
        label="Feedback recovery reservation v1",
        expected_file_sha256=expected_file_sha256,
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(reservation, FeedbackRecoveryCallReservationV1)
    if Path(path).name != feedback_recovery_attempt_filename_v1(reservation):
        raise PortfolioS1FeedbackError("Feedback recovery reservation filename drifted")
    return reservation


def write_bound_feedback_recovery_artifact_v1(
    path: str | Path, artifact: BoundFeedbackRecoveryArtifactV1
) -> Path:
    expected_name = (
        f"{artifact.new_call_ordinal:04d}-"
        f"{artifact.selection_entry_sha256[:16]}-"
        f"attempt-{artifact.new_attempt_index}.json"
    )
    if Path(path).name != expected_name:
        raise PortfolioS1FeedbackError("bound Feedback recovery filename drifted")
    return _write_recovery_model(path, artifact)


def load_bound_feedback_recovery_artifact_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> BoundFeedbackRecoveryArtifactV1:
    artifact = _load_recovery_model(
        path,
        model_type=BoundFeedbackRecoveryArtifactV1,
        label="bound Feedback recovery artifact v1",
        expected_file_sha256=expected_file_sha256,
        max_bytes=8 * 1024 * 1024,
    )
    assert isinstance(artifact, BoundFeedbackRecoveryArtifactV1)
    expected_name = (
        f"{artifact.new_call_ordinal:04d}-"
        f"{artifact.selection_entry_sha256[:16]}-"
        f"attempt-{artifact.new_attempt_index}.json"
    )
    if Path(path).name != expected_name:
        raise PortfolioS1FeedbackError("bound Feedback recovery filename drifted")
    return artifact


@dataclass(frozen=True)
class PortfolioS1FeedbackRecoveryLedgerV1:
    claims: tuple[FeedbackRecoveryGlobalClaimV1, ...]
    reservations: tuple[FeedbackRecoveryCallReservationV1, ...]
    artifacts: tuple[BoundFeedbackRecoveryArtifactV1, ...]
    orphaned_reservations: tuple[FeedbackRecoveryCallReservationV1, ...]
    pending_claims: tuple[FeedbackRecoveryGlobalClaimV1, ...]


def _optional_dynamic_inventory(directory: Path, *, label: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} directory is unsafe")
    members = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in members
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory is unsafe")
    return members


def _validate_feedback_recovery_ledger_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
) -> None:
    claims = ledger.claims
    reservations = ledger.reservations
    artifacts = ledger.artifacts
    validate_feedback_recovery_claim_set_v1(parent, control, claims, artifacts)
    if (
        control.authorization_sha256 != authorization.authorization_sha256
        or tuple(item.new_call_ordinal for item in reservations)
        != tuple(range(1, len(reservations) + 1))
        or len(reservations) > RECOVERY_NEW_PROVIDER_CALL_CEILING
        or len({item.reservation_sha256 for item in reservations}) != len(reservations)
        or len({item.artifact_sha256 for item in artifacts}) != len(artifacts)
    ):
        raise PortfolioS1FeedbackError("Feedback recovery ledger identity drifted")
    reservation_by_sha = {item.reservation_sha256: item for item in reservations}
    artifact_by_reservation: dict[str, BoundFeedbackRecoveryArtifactV1] = {}
    for artifact in artifacts:
        reservation = reservation_by_sha.get(artifact.reservation_sha256)
        if (
            reservation is None
            or artifact.reservation_sha256 in artifact_by_reservation
            or artifact.selection_ordinal != reservation.selection_ordinal
            or artifact.selection_entry_sha256 != reservation.selection_entry_sha256
            or artifact.new_call_ordinal != reservation.new_call_ordinal
            or artifact.new_attempt_index != reservation.new_attempt_index
            or artifact.lifetime_attempt_index != reservation.lifetime_attempt_index
            or artifact.control_sha256 != control.control_sha256
            or artifact.authorization_sha256 != authorization.authorization_sha256
            or artifact.previous_artifact_sha256 != reservation.previous_artifact_sha256
            or artifact.recovery_claim_ordinal != reservation.recovery_claim_ordinal
            or artifact.recovery_claim_sha256 != reservation.recovery_claim_sha256
        ):
            raise PortfolioS1FeedbackError(
                "Feedback recovery artifact/reservation binding drifted"
            )
        artifact_by_reservation[artifact.reservation_sha256] = artifact
    expected_orphans = tuple(
        item
        for item in reservations
        if item.reservation_sha256 not in artifact_by_reservation
    )
    if ledger.orphaned_reservations != expected_orphans or (
        expected_orphans and expected_orphans[-1] != reservations[-1]
    ):
        raise PortfolioS1FeedbackError("Feedback recovery orphan ledger drifted")
    used_claim_sha256s = tuple(
        item.recovery_claim_sha256
        for item in reservations
        if item.recovery_claim_sha256 is not None
    )
    claim_sha256s = tuple(item.claim_sha256 for item in claims)
    if used_claim_sha256s != claim_sha256s[: len(used_claim_sha256s)]:
        raise PortfolioS1FeedbackError("Feedback recovery claim use ordering drifted")
    expected_pending = claims[len(used_claim_sha256s) :]
    if ledger.pending_claims != expected_pending or len(expected_pending) > 1:
        raise PortfolioS1FeedbackError("Feedback recovery pending claim drifted")
    claim_by_sha = {item.claim_sha256: item for item in claims}
    artifact_by_sha = {item.artifact_sha256: item for item in artifacts}
    for reservation in reservations:
        if reservation.recovery_claim_sha256 is None:
            continue
        claim = claim_by_sha.get(reservation.recovery_claim_sha256)
        expected_previous_sha = (
            parent.terminal_ordinal73_artifacts[-1].artifact_sha256
            if claim is not None and claim.claim_ordinal == 1
            else None
            if claim is None
            else claim.trigger_artifact_sha256
        )
        trigger = (
            None
            if claim is None or claim.claim_ordinal == 1
            else artifact_by_sha.get(claim.trigger_artifact_sha256)
        )
        entry = parent.selection.entries[reservation.selection_ordinal - 1]
        if (
            claim is None
            or reservation.recovery_claim_ordinal != claim.claim_ordinal
            or reservation.retry_trigger_kind != claim.trigger_kind
            or reservation.selection_ordinal != claim.selection_ordinal
            or reservation.selection_entry_sha256 != claim.selection_entry_sha256
            or reservation.query_id != claim.query_id
            or reservation.selection_entry_sha256 != entry.entry_sha256
            or reservation.query_id != entry.query_id
            or reservation.previous_artifact_sha256 != expected_previous_sha
            or reservation.lifetime_attempt_index != claim.lifetime_attempt_index
            or reservation.wire_kind != claim.wire_kind
            or (
                claim.claim_ordinal > 1
                and (
                    trigger is None
                    or trigger.selection_ordinal != claim.selection_ordinal
                    or trigger.selection_entry_sha256 != claim.selection_entry_sha256
                    or trigger.query_id != claim.query_id
                    or trigger.feedback_result.result_sha256
                    != claim.trigger_feedback_result_sha256
                    or trigger.feedback_result.finish_reason
                    != claim.trigger_finish_reason
                    or recovery_trigger_kind(trigger.feedback_result)
                    != claim.trigger_kind
                )
            )
        ):
            raise PortfolioS1FeedbackError(
                "Feedback recovery reservation/claim ancestry drifted"
            )
    if reservations:
        first = reservations[0]
        if (
            first.selection_ordinal != 73
            or first.recovery_claim_ordinal != 1
            or first.new_call_ordinal != 1
        ):
            raise PortfolioS1FeedbackError("Feedback recovery ledger must start at 73")
    primary_ordinals = tuple(
        item.selection_ordinal
        for item in reservations
        if item.wire_kind == "primary_v7"
    )
    if primary_ordinals != UNRESOLVED_SELECTION_ORDINALS[1 : 1 + len(primary_ordinals)]:
        raise PortfolioS1FeedbackError("Feedback recovery primary order drifted")
    by_entry: dict[int, list[FeedbackRecoveryCallReservationV1]] = {}
    for reservation in reservations:
        by_entry.setdefault(reservation.selection_ordinal, []).append(reservation)
    for selection_ordinal, entry_reservations in by_entry.items():
        expected_lifetimes = (
            (3,)
            if selection_ordinal == 73
            else tuple(range(1, len(entry_reservations) + 1))
        )
        if tuple(item.lifetime_attempt_index for item in entry_reservations) != (
            expected_lifetimes
        ):
            raise PortfolioS1FeedbackError(
                "Feedback recovery per-entry attempt ordering drifted"
            )
    if expected_orphans and any(
        item.new_call_ordinal > expected_orphans[0].new_call_ordinal
        for item in reservations
    ):
        raise PortfolioS1FeedbackError("provider work exists after a recovery orphan")


def load_feedback_recovery_ledger_v1(
    output_root: str | Path,
    *,
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
) -> PortfolioS1FeedbackRecoveryLedgerV1:
    root = Path(output_root)
    claim_paths = _optional_dynamic_inventory(
        root / "recovery-claims-v1", label="Feedback recovery claims"
    )
    reservation_paths = _optional_dynamic_inventory(
        root / "provider-attempts-recovery-v1",
        label="Feedback recovery reservations",
    )
    artifact_paths = _optional_dynamic_inventory(
        root / "bound-feedback-recovery-v1",
        label="bound Feedback recovery artifacts",
    )
    claims = tuple(load_feedback_recovery_global_claim_v1(path) for path in claim_paths)
    reservations = tuple(
        sorted(
            (
                load_feedback_recovery_call_reservation_v1(path)
                for path in reservation_paths
            ),
            key=lambda item: item.new_call_ordinal,
        )
    )
    artifacts = tuple(
        sorted(
            (load_bound_feedback_recovery_artifact_v1(path) for path in artifact_paths),
            key=lambda item: item.new_call_ordinal,
        )
    )
    settled_reservations = {item.reservation_sha256 for item in artifacts}
    used_claim_count = sum(
        item.recovery_claim_sha256 is not None for item in reservations
    )
    ledger = PortfolioS1FeedbackRecoveryLedgerV1(
        claims=claims,
        reservations=reservations,
        artifacts=artifacts,
        orphaned_reservations=tuple(
            item
            for item in reservations
            if item.reservation_sha256 not in settled_reservations
        ),
        pending_claims=claims[used_claim_count:],
    )
    _validate_feedback_recovery_ledger_v1(parent, authorization, control, ledger)
    return ledger


class FeedbackRecoveryNextStepV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal[
        "create_claim",
        "reserve_claimed_call",
        "reserve_primary_call",
        "phase_complete",
        "finalize_complete",
        "terminal_nonparsed",
        "terminal_orphan",
        "terminal_budget",
    ]
    selection_ordinal: int | None = Field(default=None, ge=1, le=240)
    lifetime_attempt_index: Literal[1, 2, 3] | None = None
    wire_kind: RecoveryWireKind | None = None
    claim_ordinal: Literal[1, 2, 3] | None = None
    new_call_ordinal: int | None = Field(default=None, ge=1, le=169)
    parsed_total: int = Field(ge=73, le=240)
    terminal_error_code: str | None = None


def _final_recovery_artifacts_by_ordinal(
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
) -> dict[int, BoundFeedbackRecoveryArtifactV1]:
    final: dict[int, BoundFeedbackRecoveryArtifactV1] = {}
    for artifact in ledger.artifacts:
        prior = final.get(artifact.selection_ordinal)
        if (
            prior is None
            or artifact.lifetime_attempt_index > prior.lifetime_attempt_index
        ):
            final[artifact.selection_ordinal] = artifact
    return final


def _next_feedback_recovery_call_budget_error(
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
) -> Literal["provider_call_ceiling_exceeded", "accountable_cost_exceeded"] | None:
    if len(ledger.reservations) >= RECOVERY_NEW_PROVIDER_CALL_CEILING:
        return "provider_call_ceiling_exceeded"
    cumulative_reservation = Decimal(PARENT_ACTUAL_COST_CNY) + (
        Decimal(len(ledger.reservations) + 1)
        * Decimal(RECOVERY_PER_CALL_RESERVATION_CNY)
    )
    if cumulative_reservation > Decimal(RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY):
        return "accountable_cost_exceeded"
    return None


def next_feedback_recovery_step_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    *,
    stop_after_max_selection_ordinal: Literal[73, 120, 240] | None = None,
) -> FeedbackRecoveryNextStepV1:
    _validate_feedback_recovery_ledger_v1(parent, authorization, control, ledger)
    if stop_after_max_selection_ordinal not in {
        None,
        *RECOVERY_PHASE_MAX_SELECTION_ORDINALS,
    }:
        raise PortfolioS1FeedbackError("Feedback recovery phase boundary is invalid")
    final = _final_recovery_artifacts_by_ordinal(ledger)
    parsed_total = 73 + sum(item.status == "parsed" for item in final.values())
    if ledger.orphaned_reservations:
        orphan = ledger.orphaned_reservations[0]
        return FeedbackRecoveryNextStepV1(
            kind="terminal_orphan",
            selection_ordinal=orphan.selection_ordinal,
            lifetime_attempt_index=orphan.lifetime_attempt_index,
            wire_kind=orphan.wire_kind,
            claim_ordinal=orphan.recovery_claim_ordinal,
            new_call_ordinal=orphan.new_call_ordinal,
            parsed_total=parsed_total,
            terminal_error_code="orphan",
        )
    if ledger.pending_claims:
        claim = ledger.pending_claims[0]
        budget_error = _next_feedback_recovery_call_budget_error(ledger)
        if budget_error is not None:
            return FeedbackRecoveryNextStepV1(
                kind="terminal_budget",
                selection_ordinal=claim.selection_ordinal,
                lifetime_attempt_index=claim.lifetime_attempt_index,
                wire_kind=claim.wire_kind,
                claim_ordinal=claim.claim_ordinal,
                parsed_total=parsed_total,
                terminal_error_code=budget_error,
            )
        return FeedbackRecoveryNextStepV1(
            kind="reserve_claimed_call",
            selection_ordinal=claim.selection_ordinal,
            lifetime_attempt_index=claim.lifetime_attempt_index,
            wire_kind=claim.wire_kind,
            claim_ordinal=claim.claim_ordinal,
            new_call_ordinal=len(ledger.reservations) + 1,
            parsed_total=parsed_total,
        )
    if not ledger.claims:
        budget_error = _next_feedback_recovery_call_budget_error(ledger)
        if budget_error is not None:
            return FeedbackRecoveryNextStepV1(
                kind="terminal_budget",
                selection_ordinal=73,
                lifetime_attempt_index=3,
                wire_kind="recovery_v8_stage1",
                claim_ordinal=1,
                parsed_total=parsed_total,
                terminal_error_code=budget_error,
            )
        return FeedbackRecoveryNextStepV1(
            kind="create_claim",
            selection_ordinal=73,
            lifetime_attempt_index=3,
            wire_kind="recovery_v8_stage1",
            claim_ordinal=1,
            parsed_total=parsed_total,
        )
    ordinal73 = final.get(73)
    if ordinal73 is None:
        raise PortfolioS1FeedbackError(
            "claim1 was consumed without ordinal73 settlement"
        )
    if ordinal73.status != "parsed":
        return FeedbackRecoveryNextStepV1(
            kind="terminal_nonparsed",
            selection_ordinal=73,
            lifetime_attempt_index=3,
            wire_kind=ordinal73.wire_kind,
            claim_ordinal=1,
            new_call_ordinal=ordinal73.new_call_ordinal,
            parsed_total=parsed_total,
            terminal_error_code=ordinal73.feedback_result.error_code,
        )
    boundary = stop_after_max_selection_ordinal or 240
    for selection_ordinal in range(75, boundary + 1):
        artifact = final.get(selection_ordinal)
        if artifact is None:
            budget_error = _next_feedback_recovery_call_budget_error(ledger)
            if budget_error is not None:
                return FeedbackRecoveryNextStepV1(
                    kind="terminal_budget",
                    selection_ordinal=selection_ordinal,
                    lifetime_attempt_index=1,
                    wire_kind="primary_v7",
                    parsed_total=parsed_total,
                    terminal_error_code=budget_error,
                )
            return FeedbackRecoveryNextStepV1(
                kind="reserve_primary_call",
                selection_ordinal=selection_ordinal,
                lifetime_attempt_index=1,
                wire_kind="primary_v7",
                new_call_ordinal=len(ledger.reservations) + 1,
                parsed_total=parsed_total,
            )
        if artifact.status == "parsed":
            continue
        trigger = recovery_trigger_kind(artifact.feedback_result)
        if (
            trigger is not None
            and artifact.lifetime_attempt_index < 3
            and len(ledger.claims) < RECOVERY_GLOBAL_CLAIM_CEILING
        ):
            budget_error = _next_feedback_recovery_call_budget_error(ledger)
            if budget_error is not None:
                return FeedbackRecoveryNextStepV1(
                    kind="terminal_budget",
                    selection_ordinal=selection_ordinal,
                    lifetime_attempt_index=artifact.lifetime_attempt_index + 1,
                    wire_kind=(
                        "recovery_v8_stage1"
                        if artifact.lifetime_attempt_index == 1
                        else "recovery_v8_stage2"
                    ),
                    parsed_total=parsed_total,
                    terminal_error_code=budget_error,
                )
            return FeedbackRecoveryNextStepV1(
                kind="create_claim",
                selection_ordinal=selection_ordinal,
                lifetime_attempt_index=artifact.lifetime_attempt_index + 1,
                wire_kind=(
                    "recovery_v8_stage1"
                    if artifact.lifetime_attempt_index == 1
                    else "recovery_v8_stage2"
                ),
                claim_ordinal=len(ledger.claims) + 1,
                parsed_total=parsed_total,
            )
        return FeedbackRecoveryNextStepV1(
            kind="terminal_nonparsed",
            selection_ordinal=selection_ordinal,
            lifetime_attempt_index=artifact.lifetime_attempt_index,
            wire_kind=artifact.wire_kind,
            claim_ordinal=artifact.recovery_claim_ordinal,
            new_call_ordinal=artifact.new_call_ordinal,
            parsed_total=parsed_total,
            terminal_error_code=artifact.feedback_result.error_code,
        )
    if boundary < 240:
        expected_total = RECOVERY_PHASE_EXPECTED_PARSED_TOTALS[
            RECOVERY_PHASE_MAX_SELECTION_ORDINALS.index(boundary)
        ]
        if parsed_total != expected_total:
            raise PortfolioS1FeedbackError(
                "Feedback recovery phase parsed total drifted"
            )
        return FeedbackRecoveryNextStepV1(
            kind="phase_complete", parsed_total=parsed_total
        )
    if parsed_total != 240:
        raise PortfolioS1FeedbackError(
            "Feedback recovery cannot finalize before parsed240"
        )
    return FeedbackRecoveryNextStepV1(
        kind="finalize_complete", parsed_total=parsed_total
    )


class PortfolioS1FeedbackRecoveryRunAttemptV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    new_call_ordinal: int = Field(ge=1, le=169)
    new_attempt_index: Literal[1, 2, 3]
    lifetime_attempt_index: Literal[1, 2, 3]
    wire_kind: RecoveryWireKind
    reservation_sha256: Sha256
    artifact_sha256: Sha256 | None = None
    recovery_claim_ordinal: Literal[1, 2, 3] | None = None
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    error_code: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    wire_sha256: Sha256 | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    refusal_present: bool | None = None
    refusal_bytes: int | None = Field(default=None, ge=0)
    actual_cost_cny: str | None = None

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        provider_response_fields = (
            self.request_id,
            self.finish_reason,
            self.input_tokens,
            self.output_tokens,
            self.refusal_present,
            self.refusal_bytes,
            self.actual_cost_cny,
        )
        if self.status == "orphan":
            if (
                self.artifact_sha256 is not None
                or self.wire_sha256 is not None
                or any(item is not None for item in provider_response_fields)
            ):
                raise ValueError("orphan recovery run attempt contains response data")
        elif self.status in {"provider_error", "timeout"}:
            if (
                self.artifact_sha256 is None
                or self.wire_sha256 is None
                or any(item is not None for item in provider_response_fields)
            ):
                raise ValueError(
                    "provider-failed recovery attempt receipt fields drifted"
                )
        elif (
            self.artifact_sha256 is None
            or self.wire_sha256 is None
            or any(item is None for item in provider_response_fields)
        ):
            raise ValueError("response-bearing recovery run attempt lacks receipt data")
        if self.status == "parsed" and self.error_code is not None:
            raise ValueError("parsed recovery run attempt claims an error")
        if (
            self.status != "parsed"
            and self.status != "orphan"
            and self.error_code is None
        ):
            raise ValueError("failed recovery run attempt lacks an error code")
        if self.actual_cost_cny is not None:
            try:
                if Decimal(self.actual_cost_cny) < 0:
                    raise ValueError
            except Exception as error:
                raise ValueError("recovery run attempt cost is invalid") from error
            assert self.input_tokens is not None and self.output_tokens is not None
            expected_cost = _recovery_actual_cost(
                LLMUsage(
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                )
            )
            if self.actual_cost_cny != expected_cost:
                raise ValueError("recovery run attempt cost differs from token usage")
        return self


class PortfolioS1FeedbackRecoveryRunV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-recovery-run"] = (
        "portfolio-s1-feedback-recovery-run"
    )
    policy_version: Literal["portfolio-s1-feedback-derived-recovery-run-v1"] = (
        RECOVERY_RUN_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    authorization_sha256: Sha256
    control_sha256: Sha256
    parent_run_file_sha256: Literal[PARENT_RUN_FILE_SHA256] = PARENT_RUN_FILE_SHA256
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    parent_evidence_sha256: Sha256
    expected_count: Literal[240] = 240
    imported_count: Literal[73] = 73
    unresolved_count: Literal[167] = 167
    phase_max_selection_ordinals: tuple[int, ...] = (
        RECOVERY_PHASE_MAX_SELECTION_ORDINALS
    )
    phase_expected_parsed_totals: tuple[int, ...] = (
        RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
    )
    attempted_total: int = Field(ge=74, le=240)
    parsed_total: int = Field(ge=73, le=240)
    new_attempted_count: int = Field(ge=1, le=167)
    new_parsed_count: int = Field(ge=0, le=167)
    new_error_count: int = Field(ge=0, le=167)
    new_provider_call_count: int = Field(ge=1, le=169)
    cumulative_provider_call_count: int = Field(ge=77, le=245)
    recovery_claim_count: int = Field(ge=1, le=3)
    recovery_claim_sha256s: tuple[Sha256, ...]
    orphan_count: int = Field(ge=0, le=1)
    terminal_phase_max_selection_ordinal: Literal[73, 120, 240]
    status: Literal[
        "stopped_nonparsed", "stopped_orphan", "stopped_budget", "completed"
    ]
    terminal_reason: (
        Literal[
            "usage_limit_exceeded",
            "accountable_cost_exceeded",
            "provider_call_ceiling_exceeded",
        ]
        | None
    ) = None
    new_usage_known_count: int = Field(ge=0, le=169)
    new_usage_unknown_count: int = Field(ge=0, le=169)
    parent_input_tokens: int = Field(ge=0)
    parent_output_tokens: int = Field(ge=0)
    new_input_tokens: int = Field(ge=0)
    new_output_tokens: int = Field(ge=0)
    cumulative_input_tokens: int = Field(ge=0)
    cumulative_output_tokens: int = Field(ge=0)
    parent_actual_cost_cny: Literal[PARENT_ACTUAL_COST_CNY] = PARENT_ACTUAL_COST_CNY
    new_actual_cost_cny: str
    cumulative_actual_cost_cny: str
    cumulative_accountable_cost_cny: str
    artifacts: tuple[PortfolioS1FeedbackRecoveryRunAttemptV1, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator(
        "phase_max_selection_ordinals",
        "phase_expected_parsed_totals",
        "recovery_claim_sha256s",
        "artifacts",
        mode="before",
    )
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        final_by_entry = {item.selection_entry_sha256: item for item in self.artifacts}
        statuses = Counter(item.status for item in final_by_entry.values())
        new_actual = sum(
            (
                Decimal(item.actual_cost_cny)
                for item in self.artifacts
                if item.actual_cost_cny is not None
            ),
            Decimal("0"),
        )
        cumulative_actual = Decimal(PARENT_ACTUAL_COST_CNY) + new_actual
        cumulative_accountable = cumulative_actual + (
            Decimal(self.new_usage_unknown_count)
            * Decimal(RECOVERY_PER_CALL_RESERVATION_CNY)
        )

        def money(value: Decimal) -> str:
            return format(value.quantize(Decimal("0.000000000001")), "f")

        if (
            self.phase_max_selection_ordinals != RECOVERY_PHASE_MAX_SELECTION_ORDINALS
            or self.phase_expected_parsed_totals
            != RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
            or tuple(item.new_call_ordinal for item in self.artifacts)
            != tuple(range(1, self.new_provider_call_count + 1))
            or self.new_provider_call_count != len(self.artifacts)
            or self.cumulative_provider_call_count != 76 + self.new_provider_call_count
            or self.new_attempted_count != len(final_by_entry)
            or self.attempted_total != 73 + self.new_attempted_count
            or self.new_parsed_count != statuses["parsed"]
            or self.parsed_total != 73 + self.new_parsed_count
            or self.new_error_count != self.new_attempted_count - self.new_parsed_count
            or self.orphan_count
            != sum(item.status == "orphan" for item in self.artifacts)
            or self.recovery_claim_count != len(self.recovery_claim_sha256s)
            or self.new_usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.new_usage_unknown_count
            != self.new_provider_call_count - self.new_usage_known_count
            or self.new_input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.new_output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
            or self.cumulative_input_tokens
            != self.parent_input_tokens + self.new_input_tokens
            or self.cumulative_output_tokens
            != self.parent_output_tokens + self.new_output_tokens
            or self.artifact_set_sha256
            != _hash_payload([item.model_dump(mode="json") for item in self.artifacts])
            or self.new_actual_cost_cny != money(new_actual)
            or self.cumulative_actual_cost_cny != money(cumulative_actual)
            or self.cumulative_accountable_cost_cny != money(cumulative_accountable)
            or cumulative_accountable > Decimal(RECOVERY_OWNER_BUDGET_CEILING_CNY)
        ):
            raise ValueError("Feedback recovery run counts drifted")
        if self.status == "completed":
            if (
                self.terminal_reason is not None
                or self.parsed_total != 240
                or self.new_attempted_count != 167
                or self.orphan_count
            ):
                raise ValueError("completed Feedback recovery run is not parsed240")
        elif self.status == "stopped_orphan":
            if self.orphan_count != 1:
                raise ValueError("orphan recovery run lacks one orphan")
        elif self.status == "stopped_budget":
            if self.terminal_reason is None or self.orphan_count:
                raise ValueError("budget recovery run reason drifted")
        elif self.terminal_reason is not None or self.new_error_count < 1:
            raise ValueError("nonparsed recovery run terminal state drifted")
        if cumulative_accountable > Decimal(
            RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
        ) and self.status not in {"stopped_budget", "stopped_orphan"}:
            raise ValueError("Feedback recovery run exceeds the CNY89 technical stop")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Feedback recovery run self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _recovery_actual_cost(usage: LLMUsage | None) -> str | None:
    if usage is None:
        return None
    value = (
        Decimal(usage.input_tokens) * Decimal(12)
        + Decimal(usage.output_tokens) * Decimal(36)
    ) / Decimal(1_000_000)
    return format(value.quantize(Decimal("0.000000000001")), "f")


def build_portfolio_s1_feedback_recovery_run_v1(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None = None,
) -> PortfolioS1FeedbackRecoveryRunV1:
    _validate_feedback_recovery_ledger_v1(parent, authorization, control, ledger)
    rows: list[PortfolioS1FeedbackRecoveryRunAttemptV1] = []
    artifact_by_reservation = {
        item.reservation_sha256: item for item in ledger.artifacts
    }
    for reservation in ledger.reservations:
        artifact = artifact_by_reservation.get(reservation.reservation_sha256)
        if artifact is None:
            rows.append(
                PortfolioS1FeedbackRecoveryRunAttemptV1(
                    selection_ordinal=reservation.selection_ordinal,
                    selection_entry_sha256=reservation.selection_entry_sha256,
                    new_call_ordinal=reservation.new_call_ordinal,
                    new_attempt_index=reservation.new_attempt_index,
                    lifetime_attempt_index=reservation.lifetime_attempt_index,
                    wire_kind=reservation.wire_kind,
                    reservation_sha256=reservation.reservation_sha256,
                    recovery_claim_ordinal=reservation.recovery_claim_ordinal,
                    status="orphan",
                )
            )
            continue
        result = artifact.feedback_result
        usage = result.usage
        rows.append(
            PortfolioS1FeedbackRecoveryRunAttemptV1(
                selection_ordinal=artifact.selection_ordinal,
                selection_entry_sha256=artifact.selection_entry_sha256,
                new_call_ordinal=artifact.new_call_ordinal,
                new_attempt_index=artifact.new_attempt_index,
                lifetime_attempt_index=artifact.lifetime_attempt_index,
                wire_kind=artifact.wire_kind,
                reservation_sha256=artifact.reservation_sha256,
                artifact_sha256=artifact.artifact_sha256,
                recovery_claim_ordinal=artifact.recovery_claim_ordinal,
                status=artifact.status,
                error_code=result.error_code,
                request_id=result.request_id,
                finish_reason=result.finish_reason,
                wire_sha256=result.wire_sha256,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                reasoning_tokens=result.reasoning_tokens,
                reasoning_bytes=result.reasoning_bytes,
                refusal_present=result.refusal_present,
                refusal_bytes=result.refusal_bytes,
                actual_cost_cny=_recovery_actual_cost(usage),
            )
        )
    if not rows:
        raise PortfolioS1FeedbackError("Feedback recovery run has no new attempts")
    final_by_entry = {item.selection_entry_sha256: item for item in rows}
    new_parsed_count = sum(item.status == "parsed" for item in final_by_entry.values())
    parsed_total = 73 + new_parsed_count
    usage_exceeded = any(
        (item.input_tokens or 0) > 20_000 or (item.output_tokens or 0) > 6_154
        for item in rows
        if item.input_tokens is not None
    )
    new_actual = sum(
        (Decimal(item.actual_cost_cny) for item in rows if item.actual_cost_cny),
        Decimal("0"),
    )
    cumulative_actual = Decimal(PARENT_ACTUAL_COST_CNY) + new_actual
    unknown_usage_reserve = Decimal(RECOVERY_PER_CALL_RESERVATION_CNY) * sum(
        item.input_tokens is None for item in rows
    )
    cumulative_accountable = cumulative_actual + unknown_usage_reserve
    if usage_exceeded:
        terminal_reason = "usage_limit_exceeded"
    elif cumulative_accountable > Decimal(RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY):
        terminal_reason = "accountable_cost_exceeded"
    if ledger.orphaned_reservations:
        status = "stopped_orphan"
    elif terminal_reason is not None:
        status = "stopped_budget"
    elif parsed_total == 240 and len(final_by_entry) == 167:
        status = "completed"
    elif any(item.status != "parsed" for item in final_by_entry.values()):
        status = "stopped_nonparsed"
    else:
        raise PortfolioS1FeedbackError(
            "nonterminal recovery phase cannot publish the terminal run"
        )
    max_selection_ordinal = max(item.selection_ordinal for item in rows)
    terminal_phase = next(
        boundary
        for boundary in RECOVERY_PHASE_MAX_SELECTION_ORDINALS
        if max_selection_ordinal <= boundary
    )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-recovery-run",
        "policy_version": RECOVERY_RUN_POLICY_VERSION_V1,
        "selection_sha256": parent.selection.selection_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "control_sha256": control.control_sha256,
        "parent_run_file_sha256": PARENT_RUN_FILE_SHA256,
        "parent_run_sha256": PARENT_RUN_SHA256,
        "parent_evidence_sha256": parent.receipt.evidence_sha256,
        "expected_count": 240,
        "imported_count": 73,
        "unresolved_count": 167,
        "phase_max_selection_ordinals": RECOVERY_PHASE_MAX_SELECTION_ORDINALS,
        "phase_expected_parsed_totals": RECOVERY_PHASE_EXPECTED_PARSED_TOTALS,
        "attempted_total": 73 + len(final_by_entry),
        "parsed_total": parsed_total,
        "new_attempted_count": len(final_by_entry),
        "new_parsed_count": new_parsed_count,
        "new_error_count": len(final_by_entry) - new_parsed_count,
        "new_provider_call_count": len(rows),
        "cumulative_provider_call_count": 76 + len(rows),
        "recovery_claim_count": len(ledger.claims),
        "recovery_claim_sha256s": tuple(item.claim_sha256 for item in ledger.claims),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "terminal_phase_max_selection_ordinal": terminal_phase,
        "status": status,
        "terminal_reason": terminal_reason,
        "new_usage_known_count": sum(item.input_tokens is not None for item in rows),
        "new_usage_unknown_count": sum(item.input_tokens is None for item in rows),
        "parent_input_tokens": parent.run.input_tokens,
        "parent_output_tokens": parent.run.output_tokens,
        "new_input_tokens": sum(item.input_tokens or 0 for item in rows),
        "new_output_tokens": sum(item.output_tokens or 0 for item in rows),
        "cumulative_input_tokens": parent.run.input_tokens
        + sum(item.input_tokens or 0 for item in rows),
        "cumulative_output_tokens": parent.run.output_tokens
        + sum(item.output_tokens or 0 for item in rows),
        "parent_actual_cost_cny": PARENT_ACTUAL_COST_CNY,
        "new_actual_cost_cny": format(
            new_actual.quantize(Decimal("0.000000000001")), "f"
        ),
        "cumulative_actual_cost_cny": format(
            cumulative_actual.quantize(Decimal("0.000000000001")), "f"
        ),
        "cumulative_accountable_cost_cny": format(
            cumulative_accountable.quantize(Decimal("0.000000000001")), "f"
        ),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRecoveryRunV1.model_validate(
        {**unsigned, "run_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_portfolio_s1_feedback_recovery_run_v1(
    path: str | Path, run: PortfolioS1FeedbackRecoveryRunV1
) -> Path:
    return _write_recovery_model(path, run)


def load_portfolio_s1_feedback_recovery_run_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1FeedbackRecoveryRunV1:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1FeedbackRecoveryRunV1,
        label="Feedback recovery run v1",
        expected_file_sha256=expected_file_sha256,
        max_bytes=64 * 1024 * 1024,
    )  # type: ignore[return-value]


class PortfolioS1FeedbackBundleProvenanceV9(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    origin: Literal["parent_v3_import", "derived_recovery_v1"]
    bound_artifact_sha256: Sha256
    feedback_result_sha256: Sha256
    parent_attempt_index: Literal[1, 2] | None = None
    recovery_new_call_ordinal: int | None = Field(default=None, ge=1, le=169)
    recovery_lifetime_attempt_index: Literal[1, 2, 3] | None = None

    @model_validator(mode="after")
    def _validate_provenance(self) -> Self:
        recovery_fields = (
            self.recovery_new_call_ordinal,
            self.recovery_lifetime_attempt_index,
        )
        if self.origin == "parent_v3_import":
            if self.parent_attempt_index is None or any(
                item is not None for item in recovery_fields
            ):
                raise ValueError("parent bundle provenance drifted")
        elif self.parent_attempt_index is not None or any(
            item is None for item in recovery_fields
        ):
            raise ValueError("recovery bundle provenance drifted")
        return self


class PortfolioS1FeedbackBundleV9(PortfolioS1FeedbackBundleV5):
    schema_version: Literal[9] = 9
    policy_version: Literal["portfolio-s1-feedback-bundle-v9"] = (
        RECOVERY_BUNDLE_POLICY_VERSION_V9
    )
    provider_call_count: Literal[243, 244, 245]
    parent_provider_call_count: Literal[76] = 76
    recovery_provider_call_count: Literal[167, 168, 169]
    imported_count: Literal[73] = 73
    parent_run_sha256: Literal[PARENT_RUN_SHA256] = PARENT_RUN_SHA256
    parent_run_file_sha256: Literal[PARENT_RUN_FILE_SHA256] = PARENT_RUN_FILE_SHA256
    parent_evidence_sha256: Sha256
    parent_evidence_file_sha256: Sha256
    recovery_run_sha256: Sha256
    recovery_run_file_sha256: Sha256
    recovery_authorization_sha256: Sha256
    recovery_control_sha256: Sha256
    recovery_artifact_set_sha256: Sha256
    recovery_claim_count: Literal[1, 2, 3]
    recovery_claim_sha256s: tuple[Sha256, ...]
    imported_selection_ordinals: tuple[int, ...] = IMPORTED_SELECTION_ORDINALS
    recovery_selection_ordinals: tuple[int, ...] = UNRESOLVED_SELECTION_ORDINALS
    recovery_output_count: Literal[167] = 167
    entry_provenance: tuple[PortfolioS1FeedbackBundleProvenanceV9, ...]

    @field_validator(
        "imported_selection_ordinals",
        "recovery_selection_ordinals",
        "recovery_claim_sha256s",
        "entry_provenance",
        mode="before",
    )
    @classmethod
    def _tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_recovery_bundle(self) -> Self:
        if (
            self.provider_call_count
            != self.parent_provider_call_count + self.recovery_provider_call_count
            or self.run_sha256 != self.recovery_run_sha256
            or self.run_file_sha256 != self.recovery_run_file_sha256
            or self.authorization_sha256 != self.recovery_authorization_sha256
            or self.control_sha256 != self.recovery_control_sha256
            or self.recovery_claim_count != len(self.recovery_claim_sha256s)
            or self.imported_selection_ordinals != IMPORTED_SELECTION_ORDINALS
            or self.recovery_selection_ordinals != UNRESOLVED_SELECTION_ORDINALS
            or len(self.entry_provenance) != 240
            or tuple(item.selection_ordinal for item in self.entry_provenance)
            != tuple(range(1, 241))
        ):
            raise ValueError("Feedback bundle v9 recovery lineage drifted")
        provenance_by_ordinal = {
            item.selection_ordinal: item for item in self.entry_provenance
        }
        if any(
            provenance_by_ordinal[ordinal].origin != "parent_v3_import"
            for ordinal in IMPORTED_SELECTION_ORDINALS
        ) or any(
            provenance_by_ordinal[ordinal].origin != "derived_recovery_v1"
            for ordinal in UNRESOLVED_SELECTION_ORDINALS
        ):
            raise ValueError("Feedback bundle v9 heterogeneous origins drifted")
        for entry in self.entries:
            provenance = provenance_by_ordinal[entry.selection_ordinal]
            if (
                provenance.bound_artifact_sha256 != entry.bound_artifact_sha256
                or provenance.feedback_result_sha256 != entry.feedback_result_sha256
            ):
                raise ValueError("Feedback bundle v9 provenance hashes drifted")
        return self


def build_portfolio_s1_feedback_bundle_v9(
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1,
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1,
    control: PortfolioS1FeedbackRecoveryControlV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    run: PortfolioS1FeedbackRecoveryRunV1,
) -> PortfolioS1FeedbackBundleV9:
    _validate_feedback_recovery_ledger_v1(parent, authorization, control, ledger)
    expected_run = build_portfolio_s1_feedback_recovery_run_v1(
        parent, authorization, control, ledger
    )
    if run != expected_run or run.status != "completed":
        raise PortfolioS1FeedbackError(
            "Feedback bundle v9 requires completed parsed240"
        )
    artifact_by_ordinal: dict[
        int, BoundFeedbackArtifactV5 | BoundFeedbackRecoveryArtifactV1
    ] = {
        ordinal: artifact
        for ordinal, artifact in zip(
            IMPORTED_SELECTION_ORDINALS,
            parent.imported_artifacts,
            strict=True,
        )
    }
    for artifact in ledger.artifacts:
        prior = artifact_by_ordinal.get(artifact.selection_ordinal)
        if (
            prior is None
            or isinstance(prior, BoundFeedbackRecoveryArtifactV1)
            and artifact.lifetime_attempt_index > prior.lifetime_attempt_index
        ):
            artifact_by_ordinal[artifact.selection_ordinal] = artifact
    if set(artifact_by_ordinal) != set(range(1, 241)):
        raise PortfolioS1FeedbackError(
            "Feedback bundle v9 mixed artifact set is incomplete"
        )

    private_entries: list[PortfolioS1FeedbackBundleEntryV5] = []
    model_entries: list[PortfolioS1FeedbackModelEntryV5] = []
    representatives: list[PortfolioS1FeedbackRepresentativeExampleV5] = []
    contracts: dict[str, FeedbackGCSContractProjectionV5] = {}
    artifact_hashes: list[dict[str, object]] = []
    provenance_rows: list[PortfolioS1FeedbackBundleProvenanceV9] = []
    selected_query_ids = tuple(item.query_id for item in parent.selection.entries)
    for selected in parent.selection.entries:
        artifact = artifact_by_ordinal[selected.selection_ordinal]
        packet = artifact.feedback_packet
        result = artifact.feedback_result
        feedback = result.parsed_feedback
        artifact_control_sha256 = artifact.control_sha256
        expected_control_sha256 = (
            parent.control.control_sha256
            if isinstance(artifact, BoundFeedbackArtifactV5)
            else control.control_sha256
        )
        if (
            type(packet) is not FeedbackPacketV3
            or type(feedback) is not VisualFeedbackOutput
            or artifact.status != "parsed"
            or artifact.selection_sha256 != parent.selection.selection_sha256
            or artifact_control_sha256 != expected_control_sha256
            or artifact.query_id != selected.query_id
            or result.model != "qwen3.8-max"
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError(
                "Feedback bundle v9 mixed artifact binding drifted"
            )
        assert feedback is not None
        try:
            _validate_model_projection_privacy(
                feedback.model_dump(mode="json"),
                private_query_ids=selected_query_ids,
            )
        except PortfolioS1FeedbackError as error:
            raise PortfolioS1FeedbackError(
                "Feedback bundle v9 contains a private model projection"
            ) from error
        if any(
            _CREATOR_RUNTIME_HANDLE_RE.search(text)
            for text in _iter_string_values(feedback.model_dump(mode="json"))
        ):
            raise PortfolioS1FeedbackError(
                "Feedback bundle v9 contains private runtime metadata"
            )
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("Feedback bundle v9 suggestions repeat")
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
            raise PortfolioS1FeedbackError("Feedback bundle v9 GCS contract drifted")
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
        if isinstance(artifact, BoundFeedbackArtifactV5):
            provenance_rows.append(
                PortfolioS1FeedbackBundleProvenanceV9(
                    selection_ordinal=selected.selection_ordinal,
                    origin="parent_v3_import",
                    bound_artifact_sha256=artifact.artifact_sha256,
                    feedback_result_sha256=result.result_sha256,
                    parent_attempt_index=artifact.attempt_index,
                )
            )
        else:
            provenance_rows.append(
                PortfolioS1FeedbackBundleProvenanceV9(
                    selection_ordinal=selected.selection_ordinal,
                    origin="derived_recovery_v1",
                    bound_artifact_sha256=artifact.artifact_sha256,
                    feedback_result_sha256=result.result_sha256,
                    recovery_new_call_ordinal=artifact.new_call_ordinal,
                    recovery_lifetime_attempt_index=(artifact.lifetime_attempt_index),
                )
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
        coverage=_feedback_v5_coverage(parent.selection),
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
    run_file_sha256 = sha256_bytes(run.canonical_bytes())
    unsigned = {
        "schema_version": 9,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": RECOVERY_BUNDLE_POLICY_VERSION_V9,
        "status": "complete_policy_filtered_feedback",
        "selection_sha256": parent.selection.selection_sha256,
        "selection_file_sha256": PARENT_SELECTION_FILE_SHA256,
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": run_file_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": parent.selection.corpus_sha256,
        "parent_static_bank_sha256": parent.selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 240,
        "attempted_count": 240,
        "parsed_count": 240,
        "provider_call_count": run.cumulative_provider_call_count,
        "selected_query_ids": selected_query_ids,
        "entries": tuple(private_entries),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "model_projection": projection,
        "parent_provider_call_count": 76,
        "recovery_provider_call_count": run.new_provider_call_count,
        "imported_count": 73,
        "parent_run_sha256": PARENT_RUN_SHA256,
        "parent_run_file_sha256": PARENT_RUN_FILE_SHA256,
        "parent_evidence_sha256": parent.receipt.evidence_sha256,
        "parent_evidence_file_sha256": sha256_bytes(parent.receipt.canonical_bytes()),
        "recovery_run_sha256": run.run_sha256,
        "recovery_run_file_sha256": run_file_sha256,
        "recovery_authorization_sha256": authorization.authorization_sha256,
        "recovery_control_sha256": control.control_sha256,
        "recovery_artifact_set_sha256": run.artifact_set_sha256,
        "recovery_claim_count": run.recovery_claim_count,
        "recovery_claim_sha256s": run.recovery_claim_sha256s,
        "imported_selection_ordinals": IMPORTED_SELECTION_ORDINALS,
        "recovery_selection_ordinals": UNRESOLVED_SELECTION_ORDINALS,
        "recovery_output_count": 167,
        "entry_provenance": tuple(provenance_rows),
    }
    return PortfolioS1FeedbackBundleV9.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_portfolio_s1_feedback_bundle_v9(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV9
) -> Path:
    return _write_recovery_model(path, bundle)


def load_portfolio_s1_feedback_bundle_v9(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV9:
    return _load_recovery_model(
        path,
        model_type=PortfolioS1FeedbackBundleV9,
        label="Portfolio S1 Feedback bundle v9",
        expected_file_sha256=expected_file_sha256,
        max_bytes=128 * 1024 * 1024,
    )  # type: ignore[return-value]


__all__ = [
    "BoundFeedbackRecoveryArtifactV1",
    "FeedbackRecoveryCallReservationV1",
    "FeedbackRecoveryEntryBindingV1",
    "FeedbackRecoveryGlobalClaimV1",
    "FeedbackRecoveryNextStepV1",
    "ParentFeedbackArtifactReferenceV1",
    "PortfolioS1FeedbackBundleProvenanceV9",
    "PortfolioS1FeedbackBundleV9",
    "PortfolioS1FeedbackParentEvidenceReceiptV1",
    "PortfolioS1FeedbackRecoveryAuthorizationV1",
    "PortfolioS1FeedbackRecoveryControlV1",
    "PortfolioS1FeedbackRecoveryLedgerV1",
    "PortfolioS1FeedbackRecoveryRunAttemptV1",
    "PortfolioS1FeedbackRecoveryRunV1",
    "PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1",
    "RECOVERY_CUMULATIVE_MAXIMUM_CNY",
    "RECOVERY_GLOBAL_CLAIM_CEILING",
    "RECOVERY_MAXIMUM_RESERVATION_CNY",
    "RECOVERY_NEW_PROVIDER_CALL_CEILING",
    "RECOVERY_ORCHESTRATION_POLICY_SHA256_V1",
    "RECOVERY_ORCHESTRATION_POLICY_VERSION_V1",
    "ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1",
    "ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1",
    "ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1",
    "ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1",
    "RECOVERY_OWNER_BUDGET_CEILING_CNY",
    "RECOVERY_PER_CALL_RESERVATION_CNY",
    "RECOVERY_PHASE_EXPECTED_PARSED_TOTALS",
    "RECOVERY_PHASE_MAX_SELECTION_ORDINALS",
    "RECOVERY_PROMPT_POLICY_SHA256_V7",
    "RECOVERY_PROMPT_POLICY_SHA256_V8",
    "RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY",
    "RECOVERY_TRANSPORT_POLICY_SHA256_V8",
    "RECOVERY_TRANSPORT_POLICY_VERSION_V8",
    "RecoveryFeedbackEvaluationResultV1",
    "RecoveryFeedbackEvaluationResultV2",
    "RecoveryTriggerKind",
    "RecoveryWireKind",
    "UNRESOLVED_SELECTION_ORDINALS",
    "VerifiedPortfolioS1FeedbackParentEvidenceV1",
    "build_bound_feedback_recovery_artifact_v1",
    "build_feedback_recovery_call_reservation_v1",
    "build_feedback_recovery_global_claim_v1",
    "build_portfolio_s1_feedback_bundle_v9",
    "build_portfolio_s1_feedback_recovery_authorization_v1",
    "build_portfolio_s1_feedback_recovery_control_v1",
    "build_portfolio_s1_feedback_recovery_run_v1",
    "build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1",
    "feedback_recovery_attempt_filename_v1",
    "feedback_recovery_claim_filename_v1",
    "feedback_recovery_orchestration_policy_v1",
    "feedback_recovery_prompt_policy_v7",
    "feedback_recovery_prompt_policy_v8",
    "feedback_recovery_transport_policy_v8",
    "round3_primary_json_schema_transport_policy_v1",
    "load_bound_feedback_recovery_artifact_v1",
    "load_feedback_recovery_call_reservation_v1",
    "load_feedback_recovery_global_claim_v1",
    "load_feedback_recovery_ledger_v1",
    "load_parent_selected_feedback_remote_runtime_v1",
    "load_portfolio_s1_feedback_bundle_v9",
    "load_portfolio_s1_feedback_parent_evidence_receipt_v1",
    "load_portfolio_s1_feedback_recovery_authorization_v1",
    "load_portfolio_s1_feedback_recovery_control_v1",
    "load_portfolio_s1_feedback_recovery_run_v1",
    "load_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1",
    "load_selected_qwen38_feedback_remote_runtime_v7",
    "load_verified_parent_feedback_evidence_v1",
    "next_feedback_recovery_step_v1",
    "recovery_trigger_kind",
    "redact_recovery_result_for_creator_privacy_v1",
    "run_visual_feedback_recovery_v1",
    "run_visual_feedback_round3_primary_v1",
    "run_visual_feedback_round3_schema_primary_v1",
    "validate_feedback_recovery_claim_set_v1",
    "write_bound_feedback_recovery_artifact_v1",
    "write_feedback_recovery_call_reservation_v1",
    "write_feedback_recovery_global_claim_v1",
    "write_portfolio_s1_feedback_bundle_v9",
    "write_portfolio_s1_feedback_parent_evidence_receipt_v1",
    "write_portfolio_s1_feedback_recovery_authorization_v1",
    "write_portfolio_s1_feedback_recovery_control_v1",
    "write_portfolio_s1_feedback_recovery_run_v1",
    "write_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1",
]
