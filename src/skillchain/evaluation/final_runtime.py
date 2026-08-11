"""Executable, fail-closed visual final Judge for the Portfolio track."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from skillchain.evaluation.evaluator_isolation import (
    EvaluatorIsolationLock,
    build_bound_final_prompt,
)
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
    EvaluatorOutputParseError,
    FinalJudgeDimensionsShapeV3,
    FinalJudgeOutput,
    parse_final_judge_output,
    parse_final_judge_output_v2,
    parse_final_judge_output_v3,
    parse_final_judge_output_v4,
)
from skillchain.evaluation.packets import (
    FinalEvaluationPacket,
    JudgeOutcome,
    JudgeScores,
    build_judge_scores,
    conservative_judge_error,
    evaluator_wire_messages,
)
from skillchain.evaluation.portfolio_execution import (
    forfeit_portfolio_provider_call,
    make_portfolio_budget_call_identity,
    reserve_portfolio_provider_call,
    settle_portfolio_provider_call_success,
)
from skillchain.evaluation.visual_runtime import (
    contains_base64_data_url,
    contains_encoded_image_echo,
    EvaluatorImageLoadError,
    load_verified_evaluator_image,
)
from skillchain.llm import LLMToolCall, LLMUsage
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    read_stable_regular_file,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FINAL_JUDGE_TIMEOUT_SECONDS = 600.0
FINAL_JUDGE_RESULT_SCHEMA_VERSION = 10
FINAL_JUDGE_CACHE_NAMESPACE = "final-evaluator-v11"
FINAL_JUDGE_ANSWER_MAX_TOKENS = 2_048
FINAL_JUDGE_THINKING_BUDGET = config.PORTFOLIO_JUDGE_THINKING_BUDGET
# Historical schemas 6-9 describe the former Kimi K2.6 Judge request.  Keep
# those values independent from the active Gemini request so immutable Kimi
# receipts continue to validate byte-for-byte after the role swap.
_LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET = 6_144
_LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS = 8_192
# These are local request/receipt caps, not AIFast provider ceilings.  Provider
# ceilings and pricing remain deliberately unavailable, so the budget layer
# fail-closes before every active Gemini call until a separately authorized
# profile is frozen.
FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS = 32_768
FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE = None
FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_LIMIT = None
FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS = FINAL_JUDGE_ANSWER_MAX_TOKENS
FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE = None
CARD_REQUIREMENT_GUARD_POLICY_VERSION = "bidirectional-card-requirement-score-guard-v1"
_CARD_REQUIREMENT_GUARD_POLICY = {
    "policy_version": CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    "applies_to_status": "scored",
    "rules": [
        {
            "when": {"card_requirement": "required", "visible_card_count": 0},
            "compiled_score_override": {"CCC": 0},
        },
        {
            "when": {
                "card_requirement": "forbidden",
                "visible_card_count_min": 1,
            },
            "compiled_score_override": {"CA": 0},
        },
    ],
    "raw_submission_policy": "preserve",
}
CARD_REQUIREMENT_GUARD_POLICY_SHA256 = sha256_bytes(
    canonical_json_bytes(_CARD_REQUIREMENT_GUARD_POLICY)
)
FINAL_JUDGE_TRANSPORT_POLICY_VERSION = "final-judge-json-object-response-format-v1"


def final_judge_transport_policy() -> dict[str, object]:
    """Return the frozen AIFast Gemini JSON-object wire contract."""

    return {
        "policy_version": FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
        "json_mode": True,
        "provider_response_format": {"type": "json_object"},
        "provider_guarantee": "requested_json_syntax_only_not_schema_attested",
        "schema_enforcement": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "wire_hash_payload": ["messages", "response_format"],
        "sampling_controls": "temperature_top_p_omitted",
        "thinking_controls": "omitted",
    }


FINAL_JUDGE_TRANSPORT_POLICY_SHA256 = sha256_bytes(
    canonical_json_bytes(final_judge_transport_policy())
)
FINAL_JUDGE_MAX_ATTEMPTS = 2
_FINAL_JUDGE_RETRY_POLICY_VERSION_V1 = "empty-final-response-single-retry-v1"
_FINAL_JUDGE_RETRY_POLICY_V1 = {
    "policy_version": _FINAL_JUDGE_RETRY_POLICY_VERSION_V1,
    "max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "retry_condition": {
        "finish_reason": "stop",
        "tool_call_count": 0,
        "visible_text_after_strip": "empty",
    },
    "retry_count": 1,
    "retry_wire_policy": "identical_messages_and_parameters",
    "non_retryable_errors": [
        "input_image_echo",
        "invalid_judge_json",
        "reasoning_budget_exhausted",
    ],
    "exhausted_disposition": "retain_fixed_zero_in_denominator",
}
_FINAL_JUDGE_RETRY_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(_FINAL_JUDGE_RETRY_POLICY_V1)
)
_FINAL_JUDGE_RETRY_POLICY_VERSION_V2 = "empty-final-response-single-retry-v2"
_FINAL_JUDGE_RETRY_POLICY_V2 = {
    "policy_version": _FINAL_JUDGE_RETRY_POLICY_VERSION_V2,
    "max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "retry_condition": {
        "finish_reason": "stop",
        "tool_call_count": 0,
        "visible_text_after_strip": "empty",
    },
    "retry_count": 1,
    "retry_wire_policy": "identical_messages_and_parameters",
    "non_retryable_errors": [
        "input_image_echo",
        "invalid_judge_json",
        "reasoning_budget_exhausted",
    ],
    "exhausted_disposition": "retain_fixed_zero_in_denominator",
    "request_contract": {
        "enable_thinking": True,
        "thinking_budget": _LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET,
        "answer_max_tokens": FINAL_JUDGE_ANSWER_MAX_TOKENS,
        "max_billable_input_tokens": FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
        "max_billable_output_tokens": (
            _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        ),
    },
}
_FINAL_JUDGE_RETRY_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(_FINAL_JUDGE_RETRY_POLICY_V2)
)
FINAL_JUDGE_RETRY_POLICY_VERSION_V2 = _FINAL_JUDGE_RETRY_POLICY_VERSION_V2
FINAL_JUDGE_RETRY_POLICY_SHA256_V2 = _FINAL_JUDGE_RETRY_POLICY_SHA256_V2
FINAL_JUDGE_RETRY_POLICY_VERSION_V3 = "empty-or-invalid-final-response-single-retry-v3"
_FINAL_JUDGE_RETRY_POLICY_V3 = {
    "policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION_V3,
    "max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "retry_conditions": [
        {
            "error_code": "empty_final_response",
            "finish_reason": "stop",
            "tool_call_count": 0,
            "visible_text_after_strip": "empty",
        },
        {
            "error_code": "invalid_judge_json",
            "finish_reason": "stop",
            "tool_call_count": 0,
            "visible_text_after_strip": "nonempty",
        },
    ],
    "retry_count": 1,
    "retry_wire_policy": "identical_messages_and_parameters",
    "non_retryable_errors": [
        "input_image_echo",
        "reasoning_budget_exhausted",
    ],
    "exhausted_disposition": "retain_fixed_zero_in_denominator",
    "request_contract": {
        "enable_thinking": True,
        "thinking_budget": _LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET,
        "answer_max_tokens": FINAL_JUDGE_ANSWER_MAX_TOKENS,
        "max_billable_input_tokens": FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
        "max_billable_output_tokens": (
            _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        ),
    },
}
FINAL_JUDGE_RETRY_POLICY_SHA256_V3 = sha256_bytes(
    canonical_json_bytes(_FINAL_JUDGE_RETRY_POLICY_V3)
)
FINAL_JUDGE_RETRY_POLICY_VERSION = "empty-or-invalid-json-object-single-retry-v4"
_FINAL_JUDGE_RETRY_POLICY = {
    "policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
    "max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "retry_conditions": [
        {
            "error_code": "empty_final_response",
            "finish_reason": "stop",
            "tool_call_count": 0,
            "visible_text_after_strip": "empty",
        },
        {
            "error_code": "invalid_judge_json",
            "finish_reason": "stop",
            "tool_call_count": 0,
            "visible_text_after_strip": "nonempty",
        },
    ],
    "retry_count": 1,
    "retry_wire_policy": "identical_messages_response_format_and_parameters",
    "non_retryable_errors": ["input_image_echo"],
    "exhausted_disposition": "retain_fixed_zero_in_denominator",
    "request_contract": {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "json_mode": True,
        "provider_response_format": {"type": "json_object"},
        "temperature": "omitted",
        "top_p": "omitted",
        "thinking": "omitted",
        "thinking_budget": "omitted",
        "answer_max_tokens": FINAL_JUDGE_ANSWER_MAX_TOKENS,
        "max_billable_input_tokens": FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
        "max_billable_output_tokens": FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS,
    },
}
FINAL_JUDGE_RETRY_POLICY_SHA256 = sha256_bytes(
    canonical_json_bytes(_FINAL_JUDGE_RETRY_POLICY)
)


class FinalJudgeRuntimeError(RuntimeError):
    """The final-Judge input or runtime identity violated its contract."""


@dataclass(frozen=True, slots=True)
class FinalJudgeBudgetContext:
    """Immutable matrix identity used to authorize final-Judge provider calls.

    ``attempt_index`` identifies the enclosing query execution attempt.  The
    final-Judge retry loop deterministically assigns call indexes one and two
    inside that attempt, so an empty-response retry can never reuse the first
    call's create-only reservation identity.
    """

    ledger_root: Path
    matrix_run_id: str
    shard_id: str
    config: str
    query_id: str
    instance_sha256: str
    request_sha256: str
    attempt_index: int


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_reasoning_metadata(
    *,
    present: bool,
    tokens: int | None,
    byte_count: int,
    content_sha256: str | None,
    label: str,
) -> None:
    observed = byte_count > 0 or (tokens is not None and tokens > 0)
    if present is not observed:
        raise ValueError(f"{label} reasoning-present flag is inconsistent")
    if (byte_count > 0) is (content_sha256 is None):
        raise ValueError(f"{label} reasoning bytes/hash are inconsistent")


class FinalJudgeInitialEmptyResponseReceipt(_StrictFrozenModel):
    """The first captured response that authorizes the one fixed retry.

    The field name is retained for canonical compatibility with schema v5/v6.
    Schema v7 additionally permits a non-empty, syntactically invalid Judge
    answer and commits the reason explicitly.  Legacy empty receipts omit the
    default reason from their canonical representation.
    """

    attempt_index: Literal[1] = 1
    retry_reason: Literal["empty_final_response", "invalid_judge_json"] = Field(
        default="empty_final_response",
        exclude_if=lambda value: value == "empty_final_response",
    )
    request_id: str
    raw_response_text: str
    raw_response_sha256: Sha256
    raw_response_bytes: int = Field(ge=0)
    tool_calls: tuple[LLMToolCall, ...]
    tool_call_count: Literal[0] = 0
    usage: LLMUsage
    finish_reason: Literal["stop"] = "stop"
    latency_ms: int = Field(ge=0)
    reasoning_present: bool
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    reasoning_bytes: int = Field(ge=0)
    reasoning_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("tool_calls", mode="before")
    @classmethod
    def coerce_tool_calls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("initial empty response request_id must be non-blank")
        return value

    @model_validator(mode="after")
    def validate_retryable_response(self) -> Self:
        encoded = self.raw_response_text.encode("utf-8")
        if self.tool_calls:
            raise ValueError("initial retry receipt cannot contain tool calls")
        if self.retry_reason == "empty_final_response":
            if self.raw_response_text.strip():
                raise ValueError("empty-response retry receipt contains visible text")
        elif not self.raw_response_text.strip():
            raise ValueError("invalid-JSON retry receipt must contain visible text")
        if self.raw_response_sha256 != sha256_bytes(
            encoded
        ) or self.raw_response_bytes != len(encoded):
            raise ValueError("initial retry response commitment mismatch")
        _validate_reasoning_metadata(
            present=self.reasoning_present,
            tokens=self.reasoning_tokens,
            byte_count=self.reasoning_bytes,
            content_sha256=self.reasoning_sha256,
            label="initial retry response",
        )
        return self


class FinalJudgeEvaluationResult(_StrictFrozenModel):
    """One content-addressed provider receipt plus its fail-closed outcome."""

    schema_version: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 1
    result_kind: Literal["visual-final-judge"] = "visual-final-judge"
    cache_namespace: Literal[
        "final-evaluator-v2",
        "final-evaluator-v3",
        "final-evaluator-v4",
        "final-evaluator-v5",
        "final-evaluator-v6",
        "final-evaluator-v7",
        "final-evaluator-v8",
        "final-evaluator-v9",
        "final-evaluator-v10",
        "final-evaluator-v11",
    ] = "final-evaluator-v2"
    formal_eligible: Literal[False] = False
    evaluation_id: Sha256
    packet_sha256: Sha256
    prompt_sha256: Sha256
    image_sha256: Sha256
    wire_sha256: Sha256
    asset_catalog_sha256: Sha256
    remote_authorization_id: str
    remote_authorization_file_sha256: Sha256
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    provider: Literal["kimi", "gemini"]
    model: Literal["kimi-k2.6", "gemini-3.6-flash"]
    endpoint: str
    max_tokens: int = Field(gt=0)
    thinking_budget: Literal[6_144] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    max_billable_input_tokens: Literal[32_768] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    max_billable_output_tokens: Literal[2_048, 8_192] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    transport_policy_version: (
        Literal["final-judge-json-object-response-format-v1"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    transport_policy_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    requested_response_format: Literal["json_object"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    attempts: Literal[1, 2] = 1
    max_attempts: Literal[1, 2] = 1
    request_id: str | None = None
    raw_response_text: str | None = None
    raw_response_sha256: Sha256 | None = None
    raw_response_bytes: int | None = Field(default=None, ge=0)
    tool_calls: tuple[LLMToolCall, ...] | None = None
    tool_call_count: int | None = Field(default=None, ge=0)
    response_redaction_reason: Literal["input_image_echo"] | None = None
    parsed_submission: FinalJudgeOutput | None = None
    parser_policy_version: (
        Literal[
            "final-judge-equivalent-shapes-v2",
            "final-judge-semantic-score-shapes-v3",
            "final-judge-complete-fence-wrapper-v4",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    parser_policy_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    raw_dimensions_shape: FinalJudgeDimensionsShapeV3 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    canonical_submission_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    visible_card_count: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    card_requirement_guard_policy_version: (
        Literal["bidirectional-card-requirement-score-guard-v1"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    card_requirement_guard_policy_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    card_requirement_guard_adjusted: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    retry_policy_version: (
        Literal[
            "empty-final-response-single-retry-v1",
            "empty-final-response-single-retry-v2",
            "empty-or-invalid-final-response-single-retry-v3",
            "empty-or-invalid-json-object-single-retry-v4",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    retry_policy_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    initial_empty_response: FinalJudgeInitialEmptyResponseReceipt | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    reasoning_present: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    reasoning_bytes: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    reasoning_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    outcome: JudgeOutcome
    forfeited_reservation_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    budget_forfeit_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    result_sha256: Sha256

    @field_validator("remote_authorization_id", "endpoint")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank and trimmed")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        parser_fields = (
            self.parser_policy_version,
            self.parser_policy_sha256,
            self.raw_dimensions_shape,
            self.canonical_submission_sha256,
        )
        guard_fields = (
            self.visible_card_count,
            self.card_requirement_guard_policy_version,
            self.card_requirement_guard_policy_sha256,
            self.card_requirement_guard_adjusted,
        )
        retry_fields = (
            self.retry_policy_version,
            self.retry_policy_sha256,
            self.initial_empty_response,
        )
        reasoning_fields = (
            self.reasoning_present,
            self.reasoning_tokens,
            self.reasoning_bytes,
            self.reasoning_sha256,
        )
        token_contract_fields = (
            self.thinking_budget,
            self.max_billable_input_tokens,
            self.max_billable_output_tokens,
        )
        transport_fields = (
            self.transport_policy_version,
            self.transport_policy_sha256,
            self.requested_response_format,
        )
        budget_forfeit_fields = (
            self.forfeited_reservation_sha256,
            self.budget_forfeit_sha256,
        )
        if self.schema_version == 1:
            if (
                self.cache_namespace != "final-evaluator-v2"
                or any(value is not None for value in parser_fields)
                or any(value is not None for value in guard_fields)
            ):
                raise ValueError(
                    "legacy final-Judge receipt has v2 parser-policy fields"
                )
        elif self.schema_version == 2:
            if (
                self.cache_namespace != "final-evaluator-v3"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V2
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V2
                or any(value is not None for value in guard_fields)
            ):
                raise ValueError("final-Judge v2 parser-policy identity mismatch")
        elif self.schema_version == 3:
            if (
                self.cache_namespace != "final-evaluator-v4"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
                or any(value is not None for value in guard_fields)
            ):
                raise ValueError("final-Judge v3 parser-policy identity mismatch")
        elif self.schema_version == 4:
            if (
                self.cache_namespace != "final-evaluator-v5"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
                or self.visible_card_count is None
                or self.card_requirement_guard_policy_version
                != CARD_REQUIREMENT_GUARD_POLICY_VERSION
                or self.card_requirement_guard_policy_sha256
                != CARD_REQUIREMENT_GUARD_POLICY_SHA256
                or self.card_requirement_guard_adjusted is None
            ):
                raise ValueError("final-Judge v4 card-guard policy identity mismatch")
        elif self.schema_version == 5:
            if (
                self.cache_namespace != "final-evaluator-v6"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
                or self.visible_card_count is None
                or self.card_requirement_guard_policy_version
                != CARD_REQUIREMENT_GUARD_POLICY_VERSION
                or self.card_requirement_guard_policy_sha256
                != CARD_REQUIREMENT_GUARD_POLICY_SHA256
                or self.card_requirement_guard_adjusted is None
                or self.retry_policy_version != _FINAL_JUDGE_RETRY_POLICY_VERSION_V1
                or self.retry_policy_sha256 != _FINAL_JUDGE_RETRY_POLICY_SHA256_V1
                or self.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                or (self.attempts == 2) is (self.initial_empty_response is None)
                or (
                    self.initial_empty_response is not None
                    and self.initial_empty_response.retry_reason
                    != "empty_final_response"
                )
            ):
                raise ValueError(
                    "final-Judge v5 retry/card-guard policy identity mismatch"
                )
        elif self.schema_version == 6:
            if (
                self.cache_namespace != "final-evaluator-v7"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
                or self.visible_card_count is None
                or self.card_requirement_guard_policy_version
                != CARD_REQUIREMENT_GUARD_POLICY_VERSION
                or self.card_requirement_guard_policy_sha256
                != CARD_REQUIREMENT_GUARD_POLICY_SHA256
                or self.card_requirement_guard_adjusted is None
                or self.retry_policy_version != _FINAL_JUDGE_RETRY_POLICY_VERSION_V2
                or self.retry_policy_sha256 != _FINAL_JUDGE_RETRY_POLICY_SHA256_V2
                or self.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                or (self.attempts == 2) is (self.initial_empty_response is None)
                or (
                    self.initial_empty_response is not None
                    and self.initial_empty_response.retry_reason
                    != "empty_final_response"
                )
                or self.max_tokens != FINAL_JUDGE_ANSWER_MAX_TOKENS
                or self.thinking_budget != _LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET
                or self.max_billable_input_tokens
                != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
                or self.max_billable_output_tokens
                != _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
                or self.max_billable_output_tokens
                != self.thinking_budget + self.max_tokens
            ):
                raise ValueError(
                    "final-Judge v6 request/retry policy identity mismatch"
                )
        elif self.schema_version == 7:
            if (
                self.cache_namespace != "final-evaluator-v8"
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
                or self.visible_card_count is None
                or self.card_requirement_guard_policy_version
                != CARD_REQUIREMENT_GUARD_POLICY_VERSION
                or self.card_requirement_guard_policy_sha256
                != CARD_REQUIREMENT_GUARD_POLICY_SHA256
                or self.card_requirement_guard_adjusted is None
                or self.retry_policy_version != FINAL_JUDGE_RETRY_POLICY_VERSION_V3
                or self.retry_policy_sha256 != FINAL_JUDGE_RETRY_POLICY_SHA256_V3
                or self.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                or (self.attempts == 2) is (self.initial_empty_response is None)
                or self.max_tokens != FINAL_JUDGE_ANSWER_MAX_TOKENS
                or self.thinking_budget != _LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET
                or self.max_billable_input_tokens
                != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
                or self.max_billable_output_tokens
                != _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
                or self.max_billable_output_tokens
                != self.thinking_budget + self.max_tokens
            ):
                raise ValueError(
                    "final-Judge v7 request/retry policy identity mismatch"
                )
        elif self.schema_version in {8, 9}:
            expected_cache_namespace = (
                "final-evaluator-v9"
                if self.schema_version == 8
                else "final-evaluator-v10"
            )
            if (
                self.cache_namespace != expected_cache_namespace
                or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
                or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
                or self.visible_card_count is None
                or self.card_requirement_guard_policy_version
                != CARD_REQUIREMENT_GUARD_POLICY_VERSION
                or self.card_requirement_guard_policy_sha256
                != CARD_REQUIREMENT_GUARD_POLICY_SHA256
                or self.card_requirement_guard_adjusted is None
                or self.retry_policy_version != FINAL_JUDGE_RETRY_POLICY_VERSION_V3
                or self.retry_policy_sha256 != FINAL_JUDGE_RETRY_POLICY_SHA256_V3
                or self.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                or (self.attempts == 2) is (self.initial_empty_response is None)
                or self.max_tokens != FINAL_JUDGE_ANSWER_MAX_TOKENS
                or self.thinking_budget != _LEGACY_KIMI_FINAL_JUDGE_THINKING_BUDGET
                or self.max_billable_input_tokens
                != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
                or self.max_billable_output_tokens
                != _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
                or self.max_billable_output_tokens
                != self.thinking_budget + self.max_tokens
                or any(value is not None for value in transport_fields)
            ):
                raise ValueError(
                    f"final-Judge v{self.schema_version} request/retry policy "
                    "identity mismatch"
                )
        elif (
            self.cache_namespace != FINAL_JUDGE_CACHE_NAMESPACE
            or self.parser_policy_version != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
            or self.parser_policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
            or self.visible_card_count is None
            or self.card_requirement_guard_policy_version
            != CARD_REQUIREMENT_GUARD_POLICY_VERSION
            or self.card_requirement_guard_policy_sha256
            != CARD_REQUIREMENT_GUARD_POLICY_SHA256
            or self.card_requirement_guard_adjusted is None
            or self.retry_policy_version != FINAL_JUDGE_RETRY_POLICY_VERSION
            or self.retry_policy_sha256 != FINAL_JUDGE_RETRY_POLICY_SHA256
            or self.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
            or (self.attempts == 2) is (self.initial_empty_response is None)
            or self.max_tokens != FINAL_JUDGE_ANSWER_MAX_TOKENS
            or self.thinking_budget is not None
            or self.max_billable_input_tokens != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
            or self.max_billable_output_tokens != FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
            or self.max_billable_output_tokens != self.max_tokens
            or transport_fields
            != (
                FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
                FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
                "json_object",
            )
        ):
            raise ValueError(
                f"final-Judge v{self.schema_version} request/retry policy "
                "identity mismatch"
            )
        if self.schema_version < 6 and any(
            value is not None for value in token_contract_fields
        ):
            raise ValueError("legacy final-Judge receipt has v6 token-contract fields")
        if self.schema_version < 10 and any(
            value is not None for value in transport_fields
        ):
            raise ValueError("legacy final-Judge receipt has Gemini transport fields")
        expected_provider_model = (
            ("gemini", "gemini-3.6-flash")
            if self.schema_version == 10
            else ("kimi", "kimi-k2.6")
        )
        if (self.provider, self.model) != expected_provider_model:
            raise ValueError("final-Judge provider/model differs from its schema")
        if (
            self.schema_version == 10
            and self.endpoint != config.PROVIDER_ENDPOINTS["gemini"]
        ):
            raise ValueError("active final-Judge endpoint differs from AIFast Gemini")
        if self.schema_version < 5:
            if (
                self.attempts != 1
                or self.max_attempts != 1
                or any(value is not None for value in retry_fields)
                or any(value is not None for value in reasoning_fields)
            ):
                raise ValueError("legacy final-Judge receipt has v5 retry fields")
        if self.schema_version < 9 and any(
            value is not None for value in budget_forfeit_fields
        ):
            raise ValueError("legacy final-Judge receipt has budget-forfeit fields")
        if any(value is None for value in budget_forfeit_fields) != all(
            value is None for value in budget_forfeit_fields
        ):
            raise ValueError("final-Judge budget-forfeit binding must be complete")
        provider_pre_response = self.outcome.status in {"provider_error", "timeout"}
        if self.schema_version in {9, 10} and provider_pre_response != all(
            value is not None for value in budget_forfeit_fields
        ):
            raise ValueError(
                "active final-Judge pre-response failure must bind its budget forfeit"
            )
        if (
            self.outcome.evaluation_id != self.evaluation_id
            or self.outcome.prompt_sha256 != self.prompt_sha256
            or self.outcome.judge_provider != self.provider
            or self.outcome.judge_model != self.model
            or self.outcome.attempts != self.attempts
            or self.outcome.max_attempts != self.max_attempts
        ):
            raise ValueError("final-Judge outcome is not bound to its provider receipt")
        initial_retry = self.initial_empty_response
        if initial_retry is not None and (
            contains_base64_data_url(
                initial_retry.raw_response_text,
                *(call.arguments_json for call in initial_retry.tool_calls),
            )
        ):
            raise ValueError("initial Judge retry response leaked a Base64 data URL")
        if (
            initial_retry is not None
            and initial_retry.retry_reason == "invalid_judge_json"
        ):
            if self.schema_version not in {7, 8, 9, 10}:
                raise ValueError(
                    "legacy final-Judge result contains an invalid-JSON retry"
                )
            try:
                if self.schema_version == 7:
                    parse_final_judge_output_v3(
                        initial_retry.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
                else:
                    parse_final_judge_output_v4(
                        initial_retry.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
            except (EvaluatorOutputParseError, TypeError, ValueError):
                pass
            else:
                raise ValueError(
                    "invalid-JSON retry receipt contains a valid Judge response"
                )
        has_response = self.request_id is not None
        response_receipt_fields = (
            self.raw_response_sha256,
            self.raw_response_bytes,
            self.tool_call_count,
            self.usage,
            self.finish_reason,
            self.latency_ms,
        )
        if self.schema_version in {5, 6, 7, 8, 9, 10} and has_response:
            if self.reasoning_present is None or self.reasoning_bytes is None:
                raise ValueError("final-Judge reasoning metadata is incomplete")
            _validate_reasoning_metadata(
                present=self.reasoning_present,
                tokens=self.reasoning_tokens,
                byte_count=self.reasoning_bytes,
                content_sha256=self.reasoning_sha256,
                label="terminal final-Judge response",
            )
        elif self.schema_version in {5, 6, 7, 8, 9, 10} and any(
            value is not None for value in reasoning_fields
        ):
            raise ValueError("failed final-Judge call has reasoning metadata")
        if self.schema_version in {6, 7, 8, 9, 10}:
            output_token_ceiling = (
                FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
                if self.schema_version == 10
                else _LEGACY_KIMI_FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
            )
            usage_receipts = (
                (
                    "initial retry response",
                    None
                    if self.initial_empty_response is None
                    else self.initial_empty_response.usage,
                ),
                ("terminal response", self.usage),
            )
            for label, usage in usage_receipts:
                if usage is None:
                    continue
                if usage.input_tokens > FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS:
                    raise ValueError(
                        f"{label} exceeds the final-Judge billable input threshold"
                    )
                if usage.output_tokens > output_token_ceiling:
                    raise ValueError(
                        f"{label} exceeds the final-Judge billable output ceiling"
                    )
        if has_response:
            if any(value is None for value in response_receipt_fields):
                raise ValueError("final-Judge provider response receipt is incomplete")
            if self.outcome.raw_response_sha256 != self.raw_response_sha256:
                raise ValueError("final-Judge raw response commitment mismatch")
            if self.response_redaction_reason is None:
                if self.raw_response_text is None or self.tool_calls is None:
                    raise ValueError(
                        "unredacted final-Judge response content is incomplete"
                    )
                if contains_base64_data_url(
                    self.raw_response_text,
                    *(call.arguments_json for call in self.tool_calls),
                ):
                    raise ValueError(
                        "final-Judge receipt cannot persist a Base64 data URL"
                    )
                if (
                    sha256_bytes(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_sha256
                    or len(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_bytes
                    or len(self.tool_calls) != self.tool_call_count
                ):
                    raise ValueError("final-Judge response content commitment mismatch")
            elif (
                self.raw_response_text is not None
                or self.tool_calls is not None
                or self.response_redaction_reason != "input_image_echo"
            ):
                raise ValueError("redacted final-Judge response leaked content")
        elif (
            any(
                value is not None
                for value in (
                    *response_receipt_fields,
                    self.raw_response_text,
                    self.tool_calls,
                    self.response_redaction_reason,
                )
            )
            or self.outcome.raw_response_sha256 is not None
        ):
            raise ValueError("failed provider call cannot contain response fields")

        if self.outcome.status == "scored":
            if (
                not has_response
                or self.finish_reason != "stop"
                or self.tool_calls
                or self.parsed_submission is None
                or self.outcome.error_code is not None
                or self.response_redaction_reason is not None
            ):
                raise ValueError("scored final-Judge result has an invalid field set")
            assert self.raw_response_text is not None
            try:
                if self.schema_version == 1:
                    reconstructed_submission = parse_final_judge_output(
                        self.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
                    reconstructed_shape = None
                elif self.schema_version == 2:
                    reparsed = parse_final_judge_output_v2(
                        self.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
                    reconstructed_submission = reparsed.submission
                    reconstructed_shape = reparsed.raw_dimensions_shape
                elif self.schema_version in {3, 4, 5, 6, 7}:
                    reparsed_v3 = parse_final_judge_output_v3(
                        self.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
                    reconstructed_submission = reparsed_v3.submission
                    reconstructed_shape = reparsed_v3.raw_dimensions_shape
                else:
                    reparsed_v4 = parse_final_judge_output_v4(
                        self.raw_response_text,
                        expected_requires_card=self.outcome.scores.requires_card,
                    )
                    reconstructed_submission = reparsed_v4.submission
                    reconstructed_shape = reparsed_v4.raw_dimensions_shape
                raw_scores = {
                    item.dimension: item.score
                    for item in reconstructed_submission.dimensions
                }
                if self.schema_version in {4, 5, 6, 7, 8, 9, 10}:
                    assert self.visible_card_count is not None
                    reconstructed_scores, reconstructed_adjusted = (
                        _compile_guarded_judge_scores(
                            evaluation_id=self.evaluation_id,
                            requires_card=self.outcome.scores.requires_card,
                            visible_card_count=self.visible_card_count,
                            raw_scores=raw_scores,
                        )
                    )
                else:
                    reconstructed_scores = build_judge_scores(
                        evaluation_id=self.evaluation_id,
                        requires_card=self.outcome.scores.requires_card,
                        raw_scores=raw_scores,
                    )
                    reconstructed_adjusted = False
            except (EvaluatorOutputParseError, TypeError, ValueError) as error:
                raise ValueError(
                    "scored final-Judge result cannot be reconstructed "
                    "from raw response"
                ) from error
            if reconstructed_submission != self.parsed_submission:
                raise ValueError("final-Judge submission differs from raw response")
            if self.schema_version in {2, 3, 4, 5, 6, 7, 8, 9, 10} and (
                self.raw_dimensions_shape != reconstructed_shape
                or self.canonical_submission_sha256
                != sha256_bytes(
                    canonical_json_bytes(
                        reconstructed_submission.model_dump(mode="json")
                    )
                )
            ):
                raise ValueError(
                    "final-Judge normalized submission receipt differs "
                    "from raw response"
                )
            if reconstructed_scores != self.outcome.scores:
                raise ValueError("final-Judge scores differ from raw response")
            if (
                self.schema_version in {4, 5, 6, 7, 8, 9, 10}
                and self.card_requirement_guard_adjusted is not reconstructed_adjusted
            ):
                raise ValueError(
                    "final-Judge card-guard adjustment differs from raw response"
                )
        elif self.outcome.status == "parse_error":
            if (
                not has_response
                or self.parsed_submission is not None
                or self.outcome.error_code
                not in {
                    "empty_final_response",
                    "input_image_echo",
                    "invalid_judge_json",
                    "reasoning_budget_exhausted",
                }
            ):
                raise ValueError(
                    "parse-error final-Judge result has an invalid field set"
                )
            if self.response_redaction_reason is not None:
                if (
                    self.response_redaction_reason != "input_image_echo"
                    or self.outcome.error_code != "input_image_echo"
                ):
                    raise ValueError(
                        "redacted final-Judge parse error has an invalid reason"
                    )
            elif self.outcome.error_code == "empty_final_response":
                if (
                    self.schema_version not in {5, 6, 7, 8, 9, 10}
                    or self.attempts != 2
                    or self.initial_empty_response is None
                    or self.finish_reason != "stop"
                    or self.tool_calls
                    or self.raw_response_text is None
                    or self.raw_response_text.strip()
                ):
                    raise ValueError(
                        "empty-final-response result lacks the exhausted retry evidence"
                    )
            elif self.outcome.error_code == "reasoning_budget_exhausted":
                if (
                    self.schema_version not in {5, 6, 7, 8, 9, 10}
                    or self.finish_reason != "length"
                    or self.tool_calls
                    or self.raw_response_text is None
                    or self.raw_response_text.strip()
                    or self.reasoning_present is not True
                ):
                    raise ValueError(
                        "reasoning-budget result lacks a reasoning-only length response"
                    )
            elif self.outcome.error_code == "invalid_judge_json" and (
                self.schema_version in {5, 6, 7, 8, 9, 10}
                and not self.tool_calls
                and self.raw_response_text is not None
                and not self.raw_response_text.strip()
                and (
                    self.finish_reason == "stop"
                    or (
                        self.finish_reason == "length"
                        and self.reasoning_present is True
                    )
                )
            ):
                raise ValueError(
                    "specialized empty/reasoning response has the wrong error code"
                )
            elif self.finish_reason == "stop" and not self.tool_calls:
                assert self.raw_response_text is not None
                try:
                    if self.schema_version == 1:
                        reconstructed_submission = parse_final_judge_output(
                            self.raw_response_text,
                            expected_requires_card=self.outcome.scores.requires_card,
                        )
                    elif self.schema_version == 2:
                        reconstructed_submission = parse_final_judge_output_v2(
                            self.raw_response_text,
                            expected_requires_card=self.outcome.scores.requires_card,
                        ).submission
                    elif self.schema_version in {3, 4, 5, 6, 7}:
                        reconstructed_submission = parse_final_judge_output_v3(
                            self.raw_response_text,
                            expected_requires_card=self.outcome.scores.requires_card,
                        ).submission
                    else:
                        reconstructed_submission = parse_final_judge_output_v4(
                            self.raw_response_text,
                            expected_requires_card=self.outcome.scores.requires_card,
                        ).submission
                    build_judge_scores(
                        evaluation_id=self.evaluation_id,
                        requires_card=self.outcome.scores.requires_card,
                        raw_scores={
                            item.dimension: item.score
                            for item in reconstructed_submission.dimensions
                        },
                    )
                except (EvaluatorOutputParseError, TypeError, ValueError):
                    pass
                else:
                    raise ValueError(
                        "parse-error final-Judge result contains a valid response"
                    )
            if (
                self.raw_dimensions_shape is not None
                or self.canonical_submission_sha256 is not None
            ):
                raise ValueError(
                    "parse-error final-Judge result has normalized score fields"
                )
            if (
                self.schema_version in {4, 5, 6, 7, 8, 9, 10}
                and self.card_requirement_guard_adjusted
            ):
                raise ValueError(
                    "parse-error final-Judge result cannot apply the card guard"
                )
        elif self.outcome.status in {"provider_error", "timeout"}:
            if (
                has_response
                or self.parsed_submission is not None
                or self.outcome.error_code != self.outcome.status
                or self.raw_dimensions_shape is not None
                or self.canonical_submission_sha256 is not None
                or (
                    self.schema_version in {4, 5, 6, 7, 8, 9, 10}
                    and self.card_requirement_guard_adjusted
                )
            ):
                raise ValueError(
                    "provider-error final-Judge result has an invalid field set"
                )
        else:
            raise ValueError("final runtime cannot emit an Assistant-error outcome")
        if self.result_sha256 != _result_hash(self):
            raise ValueError("final-Judge result self hash mismatch")
        return self

    @property
    def aggregate_usage(self) -> LLMUsage:
        """Provider-reported usage across the initial response and fixed retry."""

        initial = (
            LLMUsage(input_tokens=0, output_tokens=0)
            if self.initial_empty_response is None
            else self.initial_empty_response.usage
        )
        terminal = (
            LLMUsage(input_tokens=0, output_tokens=0)
            if self.usage is None
            else self.usage
        )
        return LLMUsage(
            input_tokens=initial.input_tokens + terminal.input_tokens,
            output_tokens=initial.output_tokens + terminal.output_tokens,
        )

    @property
    def captured_response_count(self) -> int:
        """Number of provider responses with usable receipt metadata."""

        return int(self.initial_empty_response is not None) + int(
            self.request_id is not None
        )


def _result_hash(result: FinalJudgeEvaluationResult) -> str:
    payload = result.model_dump(mode="json", exclude={"result_sha256"})
    return sha256_bytes(canonical_json_bytes(payload))


def _make_result(**payload) -> FinalJudgeEvaluationResult:
    unsigned = FinalJudgeEvaluationResult.model_construct(
        **payload,
        result_sha256="0" * 64,
    )
    return FinalJudgeEvaluationResult.model_validate(
        {
            **payload,
            "result_sha256": _result_hash(unsigned),
        },
        strict=True,
    )


def _compile_guarded_judge_scores(
    *,
    evaluation_id: str,
    requires_card: bool,
    visible_card_count: int,
    raw_scores: dict[str, int],
) -> tuple[JudgeScores, bool]:
    """Apply the frozen bidirectional card guard after preserving Judge output."""

    compiled_scores = dict(raw_scores)
    adjusted = False
    if requires_card and visible_card_count == 0:
        adjusted = compiled_scores.get("CCC") != 0
        compiled_scores["CCC"] = 0
    elif not requires_card and visible_card_count > 0:
        adjusted = compiled_scores.get("CA") != 0
        compiled_scores["CA"] = 0
    return (
        build_judge_scores(
            evaluation_id=evaluation_id,
            requires_card=requires_card,
            raw_scores=compiled_scores,
        ),
        adjusted,
    )


def run_visual_final_judge(
    packet: FinalEvaluationPacket,
    evaluator_isolation: EvaluatorIsolationLock,
    *,
    remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    budget_context: FinalJudgeBudgetContext,
    max_tokens: int = FINAL_JUDGE_ANSWER_MAX_TOKENS,
    record_usage: bool = True,
) -> FinalJudgeEvaluationResult:
    """Call Gemini 3.6 and retry one empty or invalid JSON answer exactly once."""

    if type(budget_context) is not FinalJudgeBudgetContext:
        raise TypeError("budget_context must be a FinalJudgeBudgetContext")
    if max_tokens != FINAL_JUDGE_ANSWER_MAX_TOKENS:
        raise FinalJudgeRuntimeError(
            "final-Judge answer max_tokens must remain frozen at "
            f"{FINAL_JUDGE_ANSWER_MAX_TOKENS}"
        )

    try:
        verified_runtime, image_bytes = load_verified_evaluator_image(
            remote_runtime,
            processor="aifast-gemini-judge",
            image=packet.image,
        )
    except EvaluatorImageLoadError as error:
        raise FinalJudgeRuntimeError(str(error)) from error
    bound = build_bound_final_prompt(packet, evaluator_isolation)
    evaluator = bound.evaluator
    expected_identity = (
        config.PORTFOLIO_JUDGE_PROVIDER,
        config.PORTFOLIO_JUDGE_MODEL,
        config.PROVIDER_ENDPOINTS[config.PORTFOLIO_JUDGE_PROVIDER],
    )
    actual_identity = (evaluator.provider, evaluator.model, evaluator.endpoint)
    if actual_identity != expected_identity:
        raise FinalJudgeRuntimeError(
            "final evaluator lock does not match the active Portfolio runtime"
        )
    if evaluator_isolation.shared_model_runtime:
        raise FinalJudgeRuntimeError(
            "active final Judge requires cross-provider evaluator isolation"
        )

    wire_messages = evaluator_wire_messages(bound, image_bytes=image_bytes)
    requested_response_format = {"type": "json_object"}
    wire_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "messages": wire_messages,
                "response_format": requested_response_format,
            }
        )
    )
    requires_card = packet.card_requirement == "required"
    visible_card_count = len(packet.cards)
    common = {
        "schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        "cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
        "visible_card_count": visible_card_count,
        "card_requirement_guard_policy_version": (
            CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (CARD_REQUIREMENT_GUARD_POLICY_SHA256),
        "retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
        "retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
        "transport_policy_version": FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
        "requested_response_format": "json_object",
        "evaluation_id": packet.evaluation_id,
        "packet_sha256": packet.packet_sha256,
        "prompt_sha256": bound.prompt_sha256,
        "image_sha256": packet.image.sha256,
        "wire_sha256": wire_sha256,
        "asset_catalog_sha256": verified_runtime.catalog.catalog_sha256,
        "remote_authorization_id": (verified_runtime.authorization.authorization_id),
        "remote_authorization_file_sha256": (
            verified_runtime.authorization_file_sha256
        ),
        "remote_receipt_file_sha256": verified_runtime.receipt_file_sha256,
        "remote_receipt_sha256": verified_runtime.receipt.receipt_sha256,
        "provider": config.PORTFOLIO_JUDGE_PROVIDER,
        "model": config.PORTFOLIO_JUDGE_MODEL,
        "endpoint": config.PROVIDER_ENDPOINTS[config.PORTFOLIO_JUDGE_PROVIDER],
        "max_tokens": max_tokens,
        "thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
        "max_billable_input_tokens": FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
        "max_billable_output_tokens": FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS,
        "max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    }
    outcome_common = {
        "evaluation_id": packet.evaluation_id,
        "requires_card": requires_card,
        "judge_provider": config.PORTFOLIO_JUDGE_PROVIDER,
        "judge_model": config.PORTFOLIO_JUDGE_MODEL,
        "prompt_sha256": bound.prompt_sha256,
    }
    initial_empty_response: FinalJudgeInitialEmptyResponseReceipt | None = None
    for attempt_index in range(1, FINAL_JUDGE_MAX_ATTEMPTS + 1):
        attempt_fields = {
            "attempts": attempt_index,
            "initial_empty_response": initial_empty_response,
        }
        budget_identity = make_portfolio_budget_call_identity(
            matrix_run_id=budget_context.matrix_run_id,
            shard_id=budget_context.shard_id,
            config=budget_context.config,
            query_id=budget_context.query_id,
            instance_sha256=budget_context.instance_sha256,
            request_sha256=budget_context.request_sha256,
            wire_request_sha256=wire_sha256,
            stage="final_judge",
            attempt_index=budget_context.attempt_index,
            call_index=attempt_index,
        )
        reservation, _ = reserve_portfolio_provider_call(
            budget_context.ledger_root,
            identity=budget_identity,
            final_judge_max_output_tokens=FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
        )
        try:
            response = llm.chat(
                config.PORTFOLIO_JUDGE_PROVIDER,
                wire_messages,
                model=config.PORTFOLIO_JUDGE_MODEL,
                max_tokens=max_tokens,
                json_mode=True,
                max_attempts=1,
                record_usage=record_usage,
                timeout_seconds=FINAL_JUDGE_TIMEOUT_SECONDS,
            )
        except (APITimeoutError, llm.LLMTimeoutError):
            budget_forfeit, _ = forfeit_portfolio_provider_call(
                budget_context.ledger_root,
                reservation_sha256=reservation.reservation_sha256,
                reason="provider_call_ended_without_captured_response",
            )
            outcome = conservative_judge_error(
                **outcome_common,
                attempts=attempt_index,
                max_attempts=FINAL_JUDGE_MAX_ATTEMPTS,
                status="timeout",
                error_code="timeout",
            )
            return _make_result(
                **common,
                **attempt_fields,
                forfeited_reservation_sha256=reservation.reservation_sha256,
                budget_forfeit_sha256=budget_forfeit.forfeit_sha256,
                card_requirement_guard_adjusted=False,
                outcome=outcome,
            )
        except (
            APIConnectionError,
            APIStatusError,
            RateLimitError,
            llm.LLMContractError,
        ):
            budget_forfeit, _ = forfeit_portfolio_provider_call(
                budget_context.ledger_root,
                reservation_sha256=reservation.reservation_sha256,
                reason="provider_call_ended_without_captured_response",
            )
            outcome = conservative_judge_error(
                **outcome_common,
                attempts=attempt_index,
                max_attempts=FINAL_JUDGE_MAX_ATTEMPTS,
                status="provider_error",
                error_code="provider_error",
            )
            return _make_result(
                **common,
                **attempt_fields,
                forfeited_reservation_sha256=reservation.reservation_sha256,
                budget_forfeit_sha256=budget_forfeit.forfeit_sha256,
                card_requirement_guard_adjusted=False,
                outcome=outcome,
            )
        except Exception:
            forfeit_portfolio_provider_call(
                budget_context.ledger_root,
                reservation_sha256=reservation.reservation_sha256,
                reason="provider_call_ended_without_captured_response",
            )
            raise

        if type(response) is not llm.LLMResponse:
            forfeit_portfolio_provider_call(
                budget_context.ledger_root,
                reservation_sha256=reservation.reservation_sha256,
                reason="provider_call_ended_without_captured_response",
            )
            raise TypeError("final-Judge model entry returned wrong response type")

        # Settle a captured provider response before inspecting or parsing its
        # contents.  A later response-contract or evaluator parse failure must
        # not leave a successfully returned call charged at the full reserve.
        safe_response_sha256 = sha256_bytes(
            canonical_json_bytes(response.model_dump(mode="json"))
        )
        settle_portfolio_provider_call_success(
            budget_context.ledger_root,
            reservation_sha256=reservation.reservation_sha256,
            actual_input_tokens=response.usage.input_tokens,
            actual_output_tokens=response.usage.output_tokens,
            provider_request_id=response.request_id,
            response_sha256=safe_response_sha256,
        )

        raw_response_text = response.text
        encoded_response = raw_response_text.encode("utf-8")
        raw_response_sha256 = sha256_bytes(encoded_response)
        raw_response_bytes = len(encoded_response)
        response_echoes_image = contains_encoded_image_echo(
            response_text=raw_response_text,
            tool_argument_texts=(call.arguments_json for call in response.tool_calls),
            image_bytes=image_bytes,
            mime_type=packet.image.mime_type,
        )
        response_receipt = {
            "request_id": response.request_id,
            "raw_response_sha256": raw_response_sha256,
            "raw_response_bytes": raw_response_bytes,
            "tool_call_count": len(response.tool_calls),
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
            "reasoning_present": response.reasoning_present,
            "reasoning_tokens": response.reasoning_tokens,
            "reasoning_bytes": response.reasoning_bytes,
            "reasoning_sha256": response.reasoning_sha256,
        }
        if response_echoes_image:
            outcome = conservative_judge_error(
                **outcome_common,
                attempts=attempt_index,
                max_attempts=FINAL_JUDGE_MAX_ATTEMPTS,
                status="parse_error",
                error_code="input_image_echo",
                raw_response_sha256=raw_response_sha256,
            )
            return _make_result(
                **common,
                **attempt_fields,
                **response_receipt,
                response_redaction_reason="input_image_echo",
                card_requirement_guard_adjusted=False,
                outcome=outcome,
            )
        response_receipt.update(
            {
                "raw_response_text": raw_response_text,
                "tool_calls": response.tool_calls,
            }
        )

        is_empty_final = (
            response.finish_reason == "stop"
            and not response.tool_calls
            and not raw_response_text.strip()
        )
        if is_empty_final and attempt_index == 1:
            initial_empty_response = FinalJudgeInitialEmptyResponseReceipt(
                attempt_index=1,
                **response_receipt,
            )
            continue

        parse_error_code: str | None = None
        if is_empty_final:
            parse_error_code = "empty_final_response"
        elif (
            response.finish_reason == "length"
            and not response.tool_calls
            and not raw_response_text.strip()
            and response.reasoning_present
        ):
            parse_error_code = "reasoning_budget_exhausted"

        parsed = None
        submission = None
        scores = None
        card_requirement_guard_adjusted = False
        if parse_error_code is None:
            try:
                if response.finish_reason != "stop" or response.tool_calls:
                    raise EvaluatorOutputParseError(
                        "final-Judge response did not stop as one text answer"
                    )
                parsed = parse_final_judge_output_v4(
                    raw_response_text,
                    expected_requires_card=requires_card,
                )
                submission = parsed.submission
                scores, card_requirement_guard_adjusted = _compile_guarded_judge_scores(
                    evaluation_id=packet.evaluation_id,
                    requires_card=requires_card,
                    visible_card_count=visible_card_count,
                    raw_scores={
                        item.dimension: item.score for item in submission.dimensions
                    },
                )
            except (EvaluatorOutputParseError, TypeError, ValueError):
                parse_error_code = "invalid_judge_json"

        if (
            parse_error_code == "invalid_judge_json"
            and attempt_index == 1
            and response.finish_reason == "stop"
            and not response.tool_calls
            and bool(raw_response_text.strip())
        ):
            initial_empty_response = FinalJudgeInitialEmptyResponseReceipt(
                attempt_index=1,
                retry_reason="invalid_judge_json",
                **response_receipt,
            )
            continue

        if parse_error_code is not None:
            outcome = conservative_judge_error(
                **outcome_common,
                attempts=attempt_index,
                max_attempts=FINAL_JUDGE_MAX_ATTEMPTS,
                status="parse_error",
                error_code=parse_error_code,
                raw_response_sha256=raw_response_sha256,
            )
            return _make_result(
                **common,
                **attempt_fields,
                **response_receipt,
                card_requirement_guard_adjusted=False,
                outcome=outcome,
            )

        assert parsed is not None and submission is not None and scores is not None
        outcome = JudgeOutcome(
            evaluation_id=packet.evaluation_id,
            status="scored",
            attempts=attempt_index,
            max_attempts=FINAL_JUDGE_MAX_ATTEMPTS,
            scores=scores,
            judge_provider=config.PORTFOLIO_JUDGE_PROVIDER,
            judge_model=config.PORTFOLIO_JUDGE_MODEL,
            prompt_sha256=bound.prompt_sha256,
            raw_response_sha256=raw_response_sha256,
        )
        return _make_result(
            **common,
            **attempt_fields,
            **response_receipt,
            parsed_submission=submission,
            raw_dimensions_shape=parsed.raw_dimensions_shape,
            canonical_submission_sha256=sha256_bytes(
                canonical_json_bytes(submission.model_dump(mode="json"))
            ),
            card_requirement_guard_adjusted=card_requirement_guard_adjusted,
            outcome=outcome,
        )

    raise AssertionError("final-Judge retry loop exhausted without a result")


def write_final_judge_evaluation_result(
    path: str | Path,
    result: FinalJudgeEvaluationResult,
) -> Path:
    """Create one immutable final-Judge provider receipt."""

    if type(result) is not FinalJudgeEvaluationResult:
        raise TypeError("final-Judge receipt requires FinalJudgeEvaluationResult")
    content = canonical_json_bytes(result.model_dump(mode="json"))
    FinalJudgeEvaluationResult.model_validate_json(content, strict=True)
    return atomic_create_file(path, content)


def load_final_judge_evaluation_result(
    path: str | Path,
    *,
    expected_result_sha256: str | None = None,
) -> FinalJudgeEvaluationResult:
    """Load and reverify one canonical final-Judge receipt."""

    content = read_stable_regular_file(
        path,
        label="visual final-Judge receipt",
        max_bytes=1024 * 1024,
    )
    try:
        result = FinalJudgeEvaluationResult.model_validate_json(content, strict=True)
    except ValueError as error:
        raise ArtifactFormatError("visual final-Judge receipt is invalid") from error
    if canonical_json_bytes(result.model_dump(mode="json")) != content:
        raise ArtifactFormatError("visual final-Judge receipt is not canonical JSON")
    if (
        expected_result_sha256 is not None
        and result.result_sha256 != expected_result_sha256
    ):
        raise ArtifactFormatError("visual final-Judge receipt hash mismatch")
    return result


__all__ = [
    "CARD_REQUIREMENT_GUARD_POLICY_SHA256",
    "CARD_REQUIREMENT_GUARD_POLICY_VERSION",
    "FINAL_JUDGE_ANSWER_MAX_TOKENS",
    "FINAL_JUDGE_CACHE_NAMESPACE",
    "FINAL_JUDGE_MAX_ATTEMPTS",
    "FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS",
    "FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS",
    "FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE",
    "FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_LIMIT",
    "FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE",
    "FINAL_JUDGE_RESULT_SCHEMA_VERSION",
    "FINAL_JUDGE_RETRY_POLICY_SHA256",
    "FINAL_JUDGE_RETRY_POLICY_SHA256_V2",
    "FINAL_JUDGE_RETRY_POLICY_SHA256_V3",
    "FINAL_JUDGE_RETRY_POLICY_VERSION",
    "FINAL_JUDGE_RETRY_POLICY_VERSION_V2",
    "FINAL_JUDGE_RETRY_POLICY_VERSION_V3",
    "FINAL_JUDGE_THINKING_BUDGET",
    "FINAL_JUDGE_TRANSPORT_POLICY_SHA256",
    "FINAL_JUDGE_TRANSPORT_POLICY_VERSION",
    "FinalJudgeEvaluationResult",
    "FinalJudgeBudgetContext",
    "FinalJudgeInitialEmptyResponseReceipt",
    "FinalJudgeRuntimeError",
    "load_final_judge_evaluation_result",
    "final_judge_transport_policy",
    "run_visual_final_judge",
    "write_final_judge_evaluation_result",
]
