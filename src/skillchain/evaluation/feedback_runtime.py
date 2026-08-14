"""Executable visual Feedback evaluator for the Portfolio track.

The runner consumes only a verified :class:`FeedbackPacket` and the bound
evaluator-isolation lock.  It sends the packet image as a Base64 Data URL in a
real multimodal message and keeps active DashScope Qwen Feedback artifacts
separate from historical Gemini/Kimi Feedback and active Gemini Judge artifacts.
"""

from __future__ import annotations

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
    build_bound_feedback_prompt,
    build_bound_feedback_prompt_v6,
)
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
    parse_visual_feedback_output,
    parse_visual_feedback_output_v2,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.packets import (
    FeedbackPacket,
    FeedbackPacketV3,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
    evaluator_wire_messages,
)
from skillchain.evaluation.visual_runtime import (
    contains_base64_data_url,
    contains_encoded_image_echo,
    EvaluatorImageLoadError,
    VerifiedSelectedFeedbackRemoteRuntime,
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

VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1 = (
    "visual-feedback-json-object-response-format-v1"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2 = (
    "visual-feedback-kimi-dashscope-plain-json-v2"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3 = (
    "visual-feedback-kimi-dashscope-plain-json-v3"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V4 = (
    "visual-feedback-qwen-dashscope-json-object-v4"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5 = (
    "visual-feedback-qwen-dashscope-json-schema-v5"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6 = (
    "visual-feedback-qwen38-dashscope-json-schema-v6"
)
VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7 = (
    "visual-feedback-qwen38-dashscope-json-schema-v7"
)

VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1 = "visual-feedback-output-json-schema-v1"
VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1 = "visual_feedback_output_v1"


def visual_feedback_json_schema_v1() -> dict[str, object]:
    """Return the provider-facing exact VisualFeedbackOutput JSON Schema."""

    finding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dimension": {
                "type": "string",
                "enum": ["TCR", "CCC", "CQ", "CA", "routing", "tool_use"],
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "grounded_in_image": {"type": "boolean"},
            "description": {"type": "string", "minLength": 1},
            "evidence": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": [
            "dimension",
            "severity",
            "grounded_in_image",
            "description",
            "evidence",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [1]},
            "summary": {"type": "string", "minLength": 1},
            "rule_violations": {
                "type": "array",
                "items": finding,
                "maxItems": 20,
            },
            "ideal_response_gaps": {
                "type": "array",
                "items": finding,
                "maxItems": 20,
            },
            "skill_suggestions": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 8,
            },
            "evidence_quality_notes": {
                "type": ["string", "null"],
                "minLength": 1,
            },
        },
        "required": [
            "schema_version",
            "summary",
            "rule_violations",
            "ideal_response_gaps",
            "skill_suggestions",
            "evidence_quality_notes",
        ],
    }


VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(visual_feedback_json_schema_v1())
)


def visual_feedback_response_format_v1() -> llm.LLMJsonSchemaResponseFormat:
    return llm.LLMJsonSchemaResponseFormat.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
                "strict": True,
                "schema": visual_feedback_json_schema_v1(),
            },
        },
        strict=True,
    )


def visual_feedback_transport_policy_v1() -> dict[str, object]:
    """Return the frozen historical JSON framing requested from AIFast."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
        "json_mode": True,
        "provider_response_format": {"type": "json_object"},
        "provider_guarantee": ("requested_json_syntax_only_not_gateway_attested"),
        "schema_enforcement": "strict_local_parser_v3",
        "wire_hash_payload": ["messages", "response_format"],
        "rules": [
            "Keep the exact-shape prompt v4, output contract v4, and parser v3 "
            "unchanged.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v1())
)


def visual_feedback_transport_policy_v2() -> dict[str, object]:
    """Return the active non-thinking DashScope Kimi Feedback transport."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
        "provider": "kimi",
        "model": "kimi-k2.6",
        "json_mode": False,
        "provider_response_format": "omitted",
        "enable_thinking": False,
        "thinking_control": "explicit-enable_thinking-false",
        "temperature": 0.6,
        "top_p": 0.95,
        "provider_guarantee": "none_strict_local_parser_only",
        "schema_enforcement": "strict_local_parser_v3",
        "wire_hash_payload": ["messages", "invocation_controls"],
        "rules": [
            "Keep the exact-shape prompt v4, output contract v4, and parser v3 "
            "unchanged.",
            "Do not request response_format from the Kimi adapter.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v2())
)


def visual_feedback_transport_policy_v3() -> dict[str, object]:
    """Return the active Kimi transport bound to Feedback prompt v5."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
        "provider": "kimi",
        "model": "kimi-k2.6",
        "json_mode": False,
        "provider_response_format": "omitted",
        "enable_thinking": False,
        "thinking_control": "explicit-enable_thinking-false",
        "temperature": 0.6,
        "top_p": 0.95,
        "provider_guarantee": "none_strict_local_parser_only",
        "schema_enforcement": "strict_local_parser_v3",
        "wire_hash_payload": ["messages", "invocation_controls"],
        "rules": [
            "Use response-schema-v1 prompt v5, output contract v4, and parser "
            "v3 unchanged.",
            "Do not request response_format from the Kimi adapter.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v3())
)


def visual_feedback_transport_policy_v4() -> dict[str, object]:
    """Return the active Qwen3.7 JSON-object Feedback transport."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V4,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "json_mode": True,
        "provider_response_format": {"type": "json_object"},
        "stream": False,
        "stream_options": "omitted",
        "enable_thinking": True,
        "thinking_control": "explicit-enable_thinking-true",
        "thinking_budget": 2048,
        "max_tokens": 4096,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_guarantee": "requested_json_syntax_only_not_schema_attested",
        "schema_enforcement": "strict_local_parser_v3",
        "wire_hash_payload": [
            "messages",
            "response_format",
            "stream",
            "invocation_controls",
        ],
        "rules": [
            "Use response-schema-v1 prompt v5, output contract v4, and parser "
            "v3 unchanged.",
            "Request response_format type json_object; do not claim provider "
            "JSON-Schema enforcement.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V4 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v4())
)


def visual_feedback_transport_policy_v5() -> dict[str, object]:
    """Return the active strict Qwen3.7 JSON-Schema Feedback transport."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
        "provider": "qwen",
        "model": "qwen3.7-plus-2026-05-26",
        "json_mode": False,
        "provider_response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
                "strict": True,
                "schema_policy_version": VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1,
                "schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            },
        },
        "stream": False,
        "stream_options": "omitted",
        "enable_thinking": True,
        "thinking_control": "explicit-enable_thinking-true",
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 4096,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_guarantee": "requested_strict_json_schema",
        "schema_enforcement": "provider_strict_plus_strict_local_parser_v3",
        "wire_hash_payload": [
            "messages",
            "response_format",
            "stream",
            "invocation_controls",
        ],
        "rules": [
            "Use response-schema-v1 prompt v5, output contract v4, and parser "
            "v3 unchanged.",
            "Request one strict provider JSON Schema with exact root and nested "
            "finding fields; retain parser v3 as independent local validation.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v5())
)


def visual_feedback_transport_policy_v6() -> dict[str, object]:
    """Return the active strict Qwen3.8-Max JSON-Schema transport."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
        "provider": "qwen",
        "model": "qwen3.8-max",
        "json_mode": False,
        "provider_response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
                "strict": True,
                "schema_policy_version": VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1,
                "schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            },
        },
        "stream": False,
        "stream_options": "omitted",
        "enable_thinking": True,
        "thinking_control": "explicit-enable_thinking-true",
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 4096,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_guarantee": "requested_strict_json_schema",
        "schema_enforcement": "provider_strict_plus_strict_local_parser_v3",
        "wire_hash_payload": [
            "messages",
            "response_format",
            "stream",
            "invocation_controls",
        ],
        "rules": [
            "Keep response-schema-v1, Feedback prompt v6, and parser v3 "
            "unchanged for Discovery240.",
            "Request one strict provider JSON Schema with exact root and nested "
            "finding fields; retain parser v3 as independent local validation.",
            "Do not salvage JSON prefixes, repair responses, retry, or replace "
            "fixed samples.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v6())
)


def visual_feedback_transport_policy_v7() -> dict[str, object]:
    """Return the fresh-v3 Qwen3.8-Max transport with a larger wire limit."""

    return {
        "policy_version": VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
        "provider": "qwen",
        "model": "qwen3.8-max",
        "json_mode": False,
        "provider_response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": VISUAL_FEEDBACK_JSON_SCHEMA_NAME_V1,
                "strict": True,
                "schema_policy_version": VISUAL_FEEDBACK_JSON_SCHEMA_POLICY_VERSION_V1,
                "schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
            },
        },
        "stream": False,
        "stream_options": "omitted",
        "enable_thinking": True,
        "thinking_control": "explicit-enable_thinking-true",
        "thinking_budget": 2048,
        "max_tokens": "omitted",
        "max_completion_tokens": 6144,
        "timeout_seconds": 600,
        "temperature": "omitted",
        "top_p": "omitted",
        "seed": "omitted",
        "provider_guarantee": "requested_strict_json_schema",
        "schema_enforcement": "provider_strict_plus_strict_local_parser_v3",
        "wire_hash_payload": [
            "messages",
            "response_format",
            "stream",
            "invocation_controls",
        ],
        "outer_orchestration_policy_version": (
            "portfolio-s1-feedback-global-schema-or-length-retry-v2"
        ),
        "rules": [
            "Keep response-schema-v1, Feedback prompt v6, and parser v3 "
            "unchanged for Discovery240.",
            "Request one strict provider JSON Schema with exact root and nested "
            "finding fields; retain parser v3 as independent local validation.",
            "Each provider invocation is exactly one internal attempt; do not "
            "salvage JSON prefixes, repair responses, or replace fixed samples. "
            "Only the separately hashed outer retry policy v2 may issue a "
            "same-entry second call.",
        ],
    }


VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7 = sha256_bytes(
    canonical_json_bytes(visual_feedback_transport_policy_v7())
)


class FeedbackRuntimeError(RuntimeError):
    """The visual Feedback runtime or provider response violated its contract."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class FeedbackEvaluationResult(_StrictFrozenModel):
    """One auditable visual Feedback response."""

    schema_version: Literal[2, 3, 4, 5, 6] = 2
    result_kind: Literal["visual-feedback"] = "visual-feedback"
    cache_namespace: Literal[
        "feedback-evaluator-v2",
        "feedback-evaluator-v3",
        "feedback-evaluator-v4",
        "feedback-evaluator-v5",
        "feedback-evaluator-v6",
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    ] = "feedback-evaluator-v2"
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
            "visual-feedback-gcs-policy-labels-prompt-v6",
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
            "visual-feedback-qwen38-dashscope-json-schema-v6",
            "visual-feedback-qwen38-dashscope-json-schema-v7",
        ]
        | None
    ) = None
    transport_policy_sha256: Sha256 | None = None
    requested_response_format: Literal["json_object", "json_schema"] | None = None
    requested_json_schema_sha256: Sha256 | None = None
    requested_thinking: bool | None = None
    requested_thinking_budget: Literal[2048] | None = None
    requested_timeout_seconds: Literal[600] | None = None
    requested_temperature: Literal[0.6] | None = None
    requested_top_p: Literal[0.95] | None = None
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
    provider: Literal["gemini", "kimi", "qwen"]
    model: Literal[
        "gemini-3.6-flash",
        "kimi-k2.6",
        "qwen3.7-plus-2026-05-26",
        "qwen3.8-max",
    ]
    endpoint: str
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    attempts: Literal[1] = 1
    max_attempts: Literal[1] = 1
    status: Literal["parsed", "parse_error", "provider_error", "timeout"]
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
    error_code: str | None = None
    result_sha256: Sha256

    @field_validator(
        "query_id",
        "remote_authorization_id",
        "endpoint",
    )
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank and trimmed")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        expected_schema_version = {
            "feedback-evaluator-v9": 3,
            "feedback-evaluator-v10": 4,
            "feedback-evaluator-v11": 5,
            "feedback-evaluator-v12": 6,
        }.get(self.cache_namespace, 2)
        if self.schema_version != expected_schema_version:
            raise ValueError("Feedback result schema/cache identity mismatch")
        if self.cache_namespace == "feedback-evaluator-v2":
            if (
                self.parser_policy_version is not None
                or self.parser_policy_sha256 is not None
            ):
                raise ValueError("legacy Feedback result cannot claim a parser policy")
        else:
            expected_parser_identity = {
                "feedback-evaluator-v3": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
                ),
                "feedback-evaluator-v4": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v5": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v6": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v7": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v8": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v9": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v10": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v11": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
                "feedback-evaluator-v12": (
                    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                ),
            }[self.cache_namespace]
            if (
                self.parser_policy_version,
                self.parser_policy_sha256,
            ) != expected_parser_identity:
                raise ValueError("Feedback parser policy identity mismatch")
        if self.cache_namespace == "feedback-evaluator-v10":
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
            ):
                raise ValueError("Feedback prompt policy identity mismatch")
        elif self.cache_namespace == "feedback-evaluator-v11":
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) not in {
                (
                    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
                    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
                ),
                (
                    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
                    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
                ),
            }:
                raise ValueError("Feedback prompt policy identity mismatch")
        elif self.cache_namespace == "feedback-evaluator-v12":
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
            ):
                raise ValueError("Feedback prompt policy identity mismatch")
        elif self.cache_namespace in {
            "feedback-evaluator-v8",
            "feedback-evaluator-v9",
        }:
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
            ):
                raise ValueError("Feedback prompt policy identity mismatch")
        elif self.cache_namespace in {
            "feedback-evaluator-v5",
            "feedback-evaluator-v6",
            "feedback-evaluator-v7",
        }:
            if (
                self.prompt_policy_version,
                self.prompt_policy_sha256,
            ) != (
                VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
                VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
            ):
                raise ValueError("Feedback prompt policy identity mismatch")
        elif (
            self.prompt_policy_version is not None
            or self.prompt_policy_sha256 is not None
        ):
            raise ValueError(
                "historical Feedback result cannot claim an active prompt policy"
            )
        if self.cache_namespace == "feedback-evaluator-v6":
            if (
                self.transport_policy_version,
                self.transport_policy_sha256,
                self.requested_response_format,
            ) != (
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
                "json_object",
            ):
                raise ValueError("Feedback transport policy identity mismatch")
        elif self.cache_namespace in {
            "feedback-evaluator-v7",
            "feedback-evaluator-v8",
        }:
            expected_transport_identity = (
                (
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
                    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
                )
                if self.cache_namespace == "feedback-evaluator-v8"
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
                raise ValueError("Feedback transport policy identity mismatch")
        elif self.cache_namespace in {
            "feedback-evaluator-v9",
            "feedback-evaluator-v10",
        }:
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
                raise ValueError("Feedback transport policy identity mismatch")
        elif self.cache_namespace == "feedback-evaluator-v11":
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
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
                "json_schema",
                VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
                True,
                2048,
                600,
                None,
                None,
            ):
                raise ValueError("Feedback transport policy identity mismatch")
        elif self.cache_namespace == "feedback-evaluator-v12":
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
                VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
                VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
                "json_schema",
                VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
                True,
                2048,
                600,
                None,
                None,
            ):
                raise ValueError("Feedback transport policy identity mismatch")
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
                "historical Feedback result cannot claim JSON-object transport"
            )
        if self.cache_namespace in {
            "feedback-evaluator-v9",
            "feedback-evaluator-v10",
            "feedback-evaluator-v11",
            "feedback-evaluator-v12",
        }:
            expected_completion_tokens = (
                6144 if self.cache_namespace == "feedback-evaluator-v12" else 4096
            )
            if (
                self.max_tokens is not None
                or self.max_completion_tokens != expected_completion_tokens
            ):
                raise ValueError("Qwen Feedback completion-token identity mismatch")
            expected_model = (
                "qwen3.8-max"
                if self.cache_namespace
                in {"feedback-evaluator-v11", "feedback-evaluator-v12"}
                else "qwen3.7-plus-2026-05-26"
            )
            if (self.provider, self.model) != ("qwen", expected_model):
                raise ValueError("active Feedback provider identity mismatch")
            if self.endpoint != config.PROVIDER_ENDPOINTS["qwen"]:
                raise ValueError("active Feedback endpoint identity mismatch")
        elif self.max_tokens is None or self.max_completion_tokens is not None:
            raise ValueError("historical Feedback token identity mismatch")
        elif self.cache_namespace in {
            "feedback-evaluator-v7",
            "feedback-evaluator-v8",
        }:
            if (self.provider, self.model) != ("kimi", "kimi-k2.6"):
                raise ValueError("active Feedback provider identity mismatch")
            if self.endpoint != config.PROVIDER_ENDPOINTS["kimi"]:
                raise ValueError("active Feedback endpoint identity mismatch")
        elif (self.provider, self.model) != ("gemini", "gemini-3.6-flash"):
            raise ValueError("historical Feedback provider identity mismatch")
        has_response = self.request_id is not None
        response_receipt_fields = (
            self.raw_response_sha256,
            self.raw_response_bytes,
            self.tool_call_count,
            self.usage,
            self.finish_reason,
            self.latency_ms,
        )
        if has_response:
            if any(value is None for value in response_receipt_fields):
                raise ValueError("Feedback provider response receipt is incomplete")
            if self.response_redaction_reason is None:
                if self.raw_response_text is None or self.tool_calls is None:
                    raise ValueError(
                        "unredacted Feedback response content is incomplete"
                    )
                if contains_base64_data_url(
                    self.raw_response_text,
                    *(call.arguments_json for call in self.tool_calls),
                ):
                    raise ValueError(
                        "Feedback receipt cannot persist a Base64 data URL"
                    )
                if (
                    sha256_bytes(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_sha256
                    or len(self.raw_response_text.encode("utf-8"))
                    != self.raw_response_bytes
                    or len(self.tool_calls) != self.tool_call_count
                ):
                    raise ValueError("feedback response commitment mismatch")
            elif self.raw_response_text is not None or self.tool_calls is not None:
                raise ValueError("redacted Feedback response leaked content")
            elif (
                self.response_redaction_reason == "creator_projection_privacy"
                and self.cache_namespace
                not in {
                    "feedback-evaluator-v10",
                    "feedback-evaluator-v11",
                    "feedback-evaluator-v12",
                }
            ):
                raise ValueError(
                    "historical Feedback result cannot claim Creator privacy redaction"
                )
        elif any(
            value is not None
            for value in (
                *response_receipt_fields,
                self.raw_response_text,
                self.tool_calls,
                self.response_redaction_reason,
            )
        ):
            raise ValueError("failed provider call cannot contain response fields")
        reasoning_fields = (
            self.reasoning_present,
            self.reasoning_tokens,
            self.reasoning_bytes,
            self.reasoning_sha256,
        )
        if self.cache_namespace in {
            "feedback-evaluator-v9",
            "feedback-evaluator-v10",
            "feedback-evaluator-v11",
            "feedback-evaluator-v12",
        }:
            if has_response:
                if (
                    self.reasoning_present is None
                    or self.reasoning_bytes is None
                    or (self.reasoning_bytes > 0) is (self.reasoning_sha256 is None)
                ):
                    raise ValueError("Qwen Feedback reasoning metadata is incomplete")
                observed = self.reasoning_bytes > 0 or bool(self.reasoning_tokens)
                if self.reasoning_present is not observed:
                    raise ValueError(
                        "Qwen Feedback reasoning presence metadata drifted"
                    )
            elif any(value is not None for value in reasoning_fields):
                raise ValueError(
                    "failed Qwen Feedback call cannot contain reasoning metadata"
                )
        elif any(value is not None for value in reasoning_fields):
            raise ValueError(
                "historical Feedback result cannot claim reasoning metadata"
            )
        if self.status == "parsed":
            if (
                not has_response
                or self.finish_reason != "stop"
                or self.tool_calls
                or self.parsed_feedback is None
                or self.error_code is not None
                or self.response_redaction_reason is not None
            ):
                raise ValueError("parsed Feedback result has an invalid field set")
            assert self.raw_response_text is not None
            parser = _parser_for_cache_namespace(self.cache_namespace)
            try:
                reconstructed = parser(self.raw_response_text)
                if self.prompt_policy_version == (
                    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
                ):
                    _require_policy_labeled_suggestions(reconstructed)
            except EvaluatorOutputParseError as error:
                raise ValueError(
                    "parsed Feedback result cannot be reconstructed from raw response"
                ) from error
            if reconstructed != self.parsed_feedback:
                raise ValueError("parsed Feedback result differs from raw response")
        elif self.status == "parse_error":
            if (
                not has_response
                or self.parsed_feedback is not None
                or self.error_code
                not in {
                    "invalid_feedback_json",
                    "input_image_echo",
                    "creator_projection_privacy",
                }
            ):
                raise ValueError("parse-error Feedback result has an invalid field set")
            if (
                self.error_code == "creator_projection_privacy"
                and self.response_redaction_reason != "creator_projection_privacy"
            ):
                raise ValueError(
                    "Creator privacy failure must redact its provider response"
                )
            if self.response_redaction_reason is not None:
                if self.error_code != self.response_redaction_reason:
                    raise ValueError(
                        "redacted Feedback parse error has an invalid reason"
                    )
                if (
                    self.response_redaction_reason == "creator_projection_privacy"
                    and self.cache_namespace
                    not in {
                        "feedback-evaluator-v10",
                        "feedback-evaluator-v11",
                        "feedback-evaluator-v12",
                    }
                ):
                    raise ValueError(
                        "historical Feedback result cannot claim Creator privacy error"
                    )
            elif self.finish_reason == "stop" and not self.tool_calls:
                assert self.raw_response_text is not None
                try:
                    parser = _parser_for_cache_namespace(self.cache_namespace)
                    reconstructed = parser(self.raw_response_text)
                    if self.prompt_policy_version == (
                        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
                    ):
                        _require_policy_labeled_suggestions(reconstructed)
                except EvaluatorOutputParseError:
                    pass
                else:
                    raise ValueError(
                        "parse-error Feedback result contains a valid response"
                    )
        else:
            if (
                has_response
                or self.parsed_feedback is not None
                or self.error_code != self.status
            ):
                raise ValueError(
                    "provider-error Feedback result has an invalid field set"
                )
        if self.result_sha256 != _result_hash(self):
            raise ValueError("Feedback result self hash mismatch")
        return self


def _result_hash(result: FeedbackEvaluationResult) -> str:
    payload = result.model_dump(mode="json", exclude={"result_sha256"})
    if result.cache_namespace == "feedback-evaluator-v2":
        payload.pop("parser_policy_version", None)
        payload.pop("parser_policy_sha256", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v5",
        "feedback-evaluator-v6",
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("prompt_policy_version", None)
        payload.pop("prompt_policy_sha256", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v6",
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("transport_policy_version", None)
        payload.pop("transport_policy_sha256", None)
        payload.pop("requested_response_format", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("requested_thinking", None)
        payload.pop("requested_temperature", None)
        payload.pop("requested_top_p", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("max_completion_tokens", None)
        payload.pop("requested_json_schema_sha256", None)
        payload.pop("requested_thinking_budget", None)
        payload.pop("requested_timeout_seconds", None)
        payload.pop("reasoning_present", None)
        payload.pop("reasoning_tokens", None)
        payload.pop("reasoning_bytes", None)
        payload.pop("reasoning_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _parser_for_cache_namespace(cache_namespace: str):
    return {
        "feedback-evaluator-v2": parse_visual_feedback_output,
        "feedback-evaluator-v3": parse_visual_feedback_output_v2,
        "feedback-evaluator-v4": parse_visual_feedback_output_v3,
        "feedback-evaluator-v5": parse_visual_feedback_output_v3,
        "feedback-evaluator-v6": parse_visual_feedback_output_v3,
        "feedback-evaluator-v7": parse_visual_feedback_output_v3,
        "feedback-evaluator-v8": parse_visual_feedback_output_v3,
        "feedback-evaluator-v9": parse_visual_feedback_output_v3,
        "feedback-evaluator-v10": parse_visual_feedback_output_v3,
        "feedback-evaluator-v11": parse_visual_feedback_output_v3,
        "feedback-evaluator-v12": parse_visual_feedback_output_v3,
    }[cache_namespace]


def _result_bytes(result: FeedbackEvaluationResult) -> bytes:
    payload = result.model_dump(mode="json")
    if result.cache_namespace == "feedback-evaluator-v2":
        payload.pop("parser_policy_version", None)
        payload.pop("parser_policy_sha256", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v5",
        "feedback-evaluator-v6",
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("prompt_policy_version", None)
        payload.pop("prompt_policy_sha256", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v6",
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("transport_policy_version", None)
        payload.pop("transport_policy_sha256", None)
        payload.pop("requested_response_format", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v7",
        "feedback-evaluator-v8",
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("requested_thinking", None)
        payload.pop("requested_temperature", None)
        payload.pop("requested_top_p", None)
    if result.cache_namespace not in {
        "feedback-evaluator-v9",
        "feedback-evaluator-v10",
        "feedback-evaluator-v11",
        "feedback-evaluator-v12",
    }:
        payload.pop("max_completion_tokens", None)
        payload.pop("requested_json_schema_sha256", None)
        payload.pop("requested_thinking_budget", None)
        payload.pop("requested_timeout_seconds", None)
        payload.pop("reasoning_present", None)
        payload.pop("reasoning_tokens", None)
        payload.pop("reasoning_bytes", None)
        payload.pop("reasoning_sha256", None)
    return canonical_json_bytes(payload)


def _make_result(**payload) -> FeedbackEvaluationResult:
    unsigned = FeedbackEvaluationResult.model_construct(
        **payload,
        result_sha256="0" * 64,
    )
    return FeedbackEvaluationResult.model_validate(
        {
            **payload,
            "result_sha256": _result_hash(unsigned),
        },
        strict=True,
    )


def redact_feedback_result_for_creator_privacy(
    result: FeedbackEvaluationResult,
) -> FeedbackEvaluationResult:
    """Turn one parsed V10 provider receipt into a terminal privacy failure.

    The provider call remains fully accountable through its immutable response
    commitments and usage metadata, while the unsafe response body and parsed
    projection are deliberately omitted.  This is forward-only: historical
    Feedback receipts retain their exact schemas and byte projections.
    """

    if (
        type(result) is not FeedbackEvaluationResult
        or result.cache_namespace
        not in {
            "feedback-evaluator-v10",
            "feedback-evaluator-v11",
            "feedback-evaluator-v12",
        }
        or result.status != "parsed"
        or result.parsed_feedback is None
        or result.request_id is None
    ):
        raise FeedbackRuntimeError(
            "Creator privacy redaction requires one parsed forward provider receipt"
        )
    payload = {
        field_name: getattr(result, field_name)
        for field_name in type(result).model_fields
        if field_name != "result_sha256"
    }
    payload.update(
        {
            "status": "parse_error",
            "raw_response_text": None,
            "tool_calls": None,
            "response_redaction_reason": "creator_projection_privacy",
            "parsed_feedback": None,
            "error_code": "creator_projection_privacy",
        }
    )
    return _make_result(**payload)


_POLICY_SUGGESTION_PREFIXES = (
    "[policy_compatible] ",
    "[requires_new_evidence] ",
    "[rejected] ",
)


def _require_policy_labeled_suggestions(feedback: VisualFeedbackOutput) -> None:
    for suggestion in feedback.skill_suggestions:
        matches = tuple(
            prefix
            for prefix in _POLICY_SUGGESTION_PREFIXES
            if suggestion.startswith(prefix)
        )
        if (
            len(matches) != 1
            or suggestion != suggestion.strip()
            or not suggestion.removeprefix(matches[0]).strip()
        ):
            raise EvaluatorOutputParseError(
                "Feedback suggestion lacks one exact GCS policy disposition"
            )


def require_policy_labeled_suggestions(feedback: VisualFeedbackOutput) -> None:
    """Validate the active v6 disposition prefix on every suggestion."""

    _require_policy_labeled_suggestions(feedback)


def run_visual_feedback(
    packet: FeedbackPacket | FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
    *,
    remote_runtime: (
        VerifiedPortfolioRemoteProcessingRuntime | VerifiedSelectedFeedbackRemoteRuntime
    ),
    max_tokens: int | None = config.FEEDBACK_JUDGE_MAX_TOKENS,
    max_completion_tokens: int = config.FEEDBACK_JUDGE_MAX_COMPLETION_TOKENS,
    timeout_seconds: int = config.FEEDBACK_JUDGE_TIMEOUT_SECONDS,
    record_usage: bool = True,
) -> FeedbackEvaluationResult:
    """Invoke and strictly parse one active DashScope Qwen3.8-Max call.

    Historical Qwen3.7 result/cache identities remain loadable, but this live
    entry point is forward-only after the permanent model switch.
    """

    if type(packet) is FeedbackPacketV3:
        policy_labeled = True
    elif type(packet) is FeedbackPacket:
        policy_labeled = False
    else:
        raise TypeError("Feedback runtime requires an exact Feedback packet type")
    if max_completion_tokens not in {4096, 6144}:
        raise FeedbackRuntimeError(
            "Feedback max_completion_tokens is outside a frozen wire identity"
        )
    fresh_v3 = max_completion_tokens == 6144
    if fresh_v3 and not policy_labeled:
        raise FeedbackRuntimeError(
            "fresh-v3 Feedback requires an exact policy-labeled FeedbackPacketV3"
        )

    try:
        verified_runtime, image_bytes = load_verified_evaluator_image(
            remote_runtime,
            processor="dashscope-qwen38-feedback",
            image=packet.image,
            query_id=packet.query_id,
        )
    except EvaluatorImageLoadError as error:
        raise FeedbackRuntimeError(str(error)) from error
    bound = (
        build_bound_feedback_prompt_v6(packet, evaluator_isolation)
        if policy_labeled
        else build_bound_feedback_prompt(packet, evaluator_isolation)
    )
    evaluator = bound.evaluator
    expected_identity = (
        "qwen",
        "qwen3.8-max",
        config.PROVIDER_ENDPOINTS["qwen"],
    )
    actual_identity = (evaluator.provider, evaluator.model, evaluator.endpoint)
    if actual_identity != expected_identity:
        raise FeedbackRuntimeError(
            "feedback evaluator lock does not match the active Portfolio runtime"
        )
    if evaluator_isolation.shared_model_runtime:
        raise FeedbackRuntimeError(
            "active Feedback requires cross-provider evaluator isolation"
        )

    wire_messages = evaluator_wire_messages(bound, image_bytes=image_bytes)
    invocation_controls = {
        "enable_thinking": config.FEEDBACK_JUDGE_THINKING,
        "thinking_budget": config.FEEDBACK_JUDGE_THINKING_BUDGET,
        "max_tokens": "omitted",
        "max_completion_tokens": max_completion_tokens,
        "timeout_seconds": timeout_seconds,
        "temperature": config.FEEDBACK_JUDGE_TEMPERATURE,
        "top_p": config.FEEDBACK_JUDGE_TOP_P,
    }
    response_format = visual_feedback_response_format_v1()
    requested_response_format = response_format.model_dump(mode="json", by_alias=True)
    wire_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "messages": wire_messages,
                "response_format": requested_response_format,
                "stream": False,
                "invocation_controls": invocation_controls,
            }
        )
    )
    common = {
        "schema_version": 6 if fresh_v3 else 5,
        "cache_namespace": (
            "feedback-evaluator-v12" if fresh_v3 else "feedback-evaluator-v11"
        ),
        "parser_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        "prompt_policy_version": (
            VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
            if policy_labeled
            else VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5
        ),
        "prompt_policy_sha256": (
            VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
            if policy_labeled
            else VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5
        ),
        "transport_policy_version": (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
            if fresh_v3
            else VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6
        ),
        "transport_policy_sha256": (
            VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
            if fresh_v3
            else VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6
        ),
        "requested_response_format": "json_schema",
        "requested_json_schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
        "requested_thinking": config.FEEDBACK_JUDGE_THINKING,
        "requested_thinking_budget": config.FEEDBACK_JUDGE_THINKING_BUDGET,
        "requested_timeout_seconds": timeout_seconds,
        "requested_temperature": config.FEEDBACK_JUDGE_TEMPERATURE,
        "requested_top_p": config.FEEDBACK_JUDGE_TOP_P,
        "query_id": packet.query_id,
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
        "provider": "qwen",
        "model": "qwen3.8-max",
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
    }
    try:
        response = llm.chat(
            "qwen",
            wire_messages,
            model="qwen3.8-max",
            temperature=config.FEEDBACK_JUDGE_TEMPERATURE,
            top_p=config.FEEDBACK_JUDGE_TOP_P,
            thinking=config.FEEDBACK_JUDGE_THINKING,
            thinking_budget=config.FEEDBACK_JUDGE_THINKING_BUDGET,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            json_mode=False,
            response_format=response_format,
            max_attempts=1,
            record_usage=record_usage,
            timeout_seconds=timeout_seconds,
        )
    except (APITimeoutError, llm.LLMTimeoutError):
        return _make_result(
            **common,
            status="timeout",
            error_code="timeout",
        )
    except (APIConnectionError, APIStatusError, RateLimitError, llm.LLMContractError):
        return _make_result(
            **common,
            status="provider_error",
            error_code="provider_error",
        )

    raw_response_text = response.text
    raw_response_sha256 = sha256_bytes(raw_response_text.encode("utf-8"))
    raw_response_bytes = len(raw_response_text.encode("utf-8"))
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
        return _make_result(
            **common,
            **response_receipt,
            response_redaction_reason="input_image_echo",
            status="parse_error",
            error_code="input_image_echo",
        )
    response_receipt.update(
        {
            "raw_response_text": raw_response_text,
            "tool_calls": response.tool_calls,
        }
    )
    try:
        if response.finish_reason != "stop" or response.tool_calls:
            raise EvaluatorOutputParseError(
                "Feedback response did not stop as one text answer"
            )
        parsed_feedback = parse_visual_feedback_output_v3(raw_response_text)
        if policy_labeled:
            _require_policy_labeled_suggestions(parsed_feedback)
    except EvaluatorOutputParseError:
        return _make_result(
            **common,
            **response_receipt,
            status="parse_error",
            error_code="invalid_feedback_json",
        )
    return _make_result(
        **common,
        **response_receipt,
        status="parsed",
        parsed_feedback=parsed_feedback,
    )


def write_feedback_evaluation_result(
    path: str | Path,
    result: FeedbackEvaluationResult,
) -> Path:
    """Create one immutable structured Feedback receipt."""

    if type(result) is not FeedbackEvaluationResult:
        raise TypeError("Feedback receipt requires FeedbackEvaluationResult")
    content = _result_bytes(result)
    FeedbackEvaluationResult.model_validate_json(content, strict=True)
    return atomic_create_file(path, content)


def load_feedback_evaluation_result(
    path: str | Path,
    *,
    expected_result_sha256: str | None = None,
) -> FeedbackEvaluationResult:
    """Load and reverify one canonical Feedback receipt."""

    content = read_stable_regular_file(
        path,
        label="visual Feedback receipt",
        max_bytes=1024 * 1024,
    )
    try:
        result = FeedbackEvaluationResult.model_validate_json(content, strict=True)
    except ValueError as error:
        raise ArtifactFormatError("visual Feedback receipt is invalid") from error
    if _result_bytes(result) != content:
        raise ArtifactFormatError("visual Feedback receipt is not canonical JSON")
    if (
        expected_result_sha256 is not None
        and result.result_sha256 != expected_result_sha256
    ):
        raise ArtifactFormatError("visual Feedback receipt hash mismatch")
    return result


__all__ = [
    "FeedbackEvaluationResult",
    "FeedbackRuntimeError",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V4",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V4",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6",
    "VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7",
    "load_feedback_evaluation_result",
    "require_policy_labeled_suggestions",
    "redact_feedback_result_for_creator_privacy",
    "run_visual_feedback",
    "visual_feedback_transport_policy_v1",
    "visual_feedback_transport_policy_v2",
    "visual_feedback_transport_policy_v3",
    "visual_feedback_transport_policy_v4",
    "visual_feedback_transport_policy_v5",
    "visual_feedback_transport_policy_v6",
    "visual_feedback_transport_policy_v7",
    "write_feedback_evaluation_result",
]
