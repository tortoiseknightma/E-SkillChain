"""Forward-only governance for Qwen3.7 Portfolio S1 visual Feedback.

The historical Gemini and Kimi authorizations are intentionally not imported
or retyped here.  This module records the owner's new, exact selected48 scope
for ``qwen3.7-plus-2026-05-26`` and binds it to independently verified Core
catalog membership.  The parent Core authorization is used only to establish
the catalog/image membership; its ``dashscope-qwen-assistant`` processor does
not authorize the new Feedback role.

No function in this module calls a model provider.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

QWEN37_FEEDBACK_PROVIDER = "qwen"
QWEN37_FEEDBACK_MODEL = "qwen3.7-plus-2026-05-26"
QWEN37_FEEDBACK_PROCESSOR = "dashscope-qwen37-feedback"
QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION = "DASHSCOPE_BASE_URL"
QWEN37_FEEDBACK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN37_FEEDBACK_CACHE_NAMESPACE = "feedback-evaluator-v9"
QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION = (
    "portfolio-s1-feedback-selected-assets-authorization-v3"
)
QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION = (
    "portfolio-s1-qwen37-feedback-model-source-lock-v1"
)
QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION = (
    "portfolio-s1-qwen37-feedback-pricing-lock-v1"
)
QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION = "portfolio-s1-qwen37-feedback-launch-lock-v1"
QWEN37_FEEDBACK_SELECTED_COUNT = 48
QWEN37_FEEDBACK_PROVIDER_CALL_CEILING = 48
QWEN37_FEEDBACK_THINKING_BUDGET = 2048
QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS = 4096
QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE = 10
QWEN37_FEEDBACK_TIMEOUT_SECONDS = 600
QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS = 1_000_000
QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS = 131_072
QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS = 256_000
QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS = 2
QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS = 8
QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS = 20_000
QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS = 4_106
QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY = "0.072848000000"
QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY = "3.496704000000"
QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY = "4.000000000000"
QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION = "visual-feedback-output-json-schema-v1"
QWEN37_FEEDBACK_JSON_SCHEMA_NAME = "visual_feedback_output_v1"
# Updated only when the provider-facing schema itself is intentionally revised.
QWEN37_FEEDBACK_JSON_SCHEMA_SHA256 = (
    "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
)
QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION = (
    "visual-feedback-qwen-dashscope-json-schema-v5"
)
QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256 = (
    "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
)
QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256 = (
    "8621e1823d0828efcd3cad1aba09f390dbbefa01fca509c525309e112b719d69"
)
QWEN37_FEEDBACK_SOURCE_LOCK_SHA256 = (
    "107b484f63cb9f7d05be56bd8f5f1182437b3f125fd2911e130fed9ebad2da07"
)
QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256 = (
    "d6c8247b18d404a0c75102b1bed38ec59671b63c0a157f48d82fb8595e038015"
)
QWEN37_FEEDBACK_PRICING_LOCK_SHA256 = (
    "e68d1a0cde879174b06cc4230da848e595cd8ac7a7be8b4a8e383831cb4a8b30"
)
QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256 = (
    "31ebdbe0af7d844ce3bf88bb0ab17d0c05640082618a4c24e1f022e38868efe8"
)
QWEN37_FEEDBACK_ROLE_SELECTION_SHA256 = (
    "9120e8abe4f667127c8ffc94f4175fc118188b2749ea7405340a03c44d7c369c"
)

_AUTHORIZATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PortfolioS1QwenFeedbackGovernanceError(ValueError):
    """A Qwen3.7 Feedback governance artifact is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _self_hash(model: BaseModel, field: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(model.model_dump(mode="json", exclude={field}))
    )


def _canonical_text(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


class _SelectionEntry(Protocol):
    query_id: str
    asset_id: str
    image_sha256: str


class _Selection(Protocol):
    selection_sha256: str
    entries: tuple[_SelectionEntry, ...]


class Qwen37FeedbackSourceEvidenceV1(_StrictFrozenModel):
    source_id: Literal[
        "model-capabilities",
        "structured-output",
        "deep-thinking",
        "chat-completions",
        "pricing",
    ]
    url: str
    retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    facts: tuple[str, ...]

    @field_validator("facts", mode="before")
    @classmethod
    def _facts_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = _canonical_text(value, "source URL")
        if not value.startswith("https://help.aliyun.com/"):
            raise ValueError("Qwen source evidence must use official Alibaba docs")
        return value

    @model_validator(mode="after")
    def _validate_facts(self) -> Self:
        if not self.facts or any(
            not item or item != item.strip() for item in self.facts
        ):
            raise ValueError("source evidence facts must be non-empty and canonical")
        return self


class Qwen37FeedbackModelSourceLockV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-model-source-lock"] = (
        "portfolio-s1-qwen37-feedback-model-source-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-model-source-lock-v1"] = (
        QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    endpoint: Literal["https://dashscope.aliyuncs.com/compatible-mode/v1"] = (
        QWEN37_FEEDBACK_ENDPOINT
    )
    input_modalities: tuple[Literal["text", "image", "video"], ...] = (
        "text",
        "image",
        "video",
    )
    output_modalities: tuple[Literal["text"], ...] = ("text",)
    context_window_tokens: Literal[1000000] = QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS
    provider_max_output_tokens: Literal[131072] = (
        QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS
    )
    structured_output_supported: Literal[True] = True
    thinking_mode_supported: Literal[True] = True
    structured_output_requires_non_thinking: Literal[False] = False
    structured_output_with_thinking_supported: Literal[True] = True
    structured_output_json_schema_supported: Literal[True] = True
    structured_output_response_format: Literal["json_schema"] = "json_schema"
    structured_output_strict: Literal[True] = True
    structured_output_prompt_must_contain_json: Literal[False] = False
    structured_output_max_tokens_guidance: Literal[
        "official-docs-recommend-omitting-max_tokens-to-avoid-truncation"
    ] = "official-docs-recommend-omitting-max_tokens-to-avoid-truncation"
    max_tokens_excludes_reasoning: Literal[True] = True
    max_completion_tokens_includes_reasoning_and_answer: Literal[True] = True
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    visual_input_supported: Literal[True] = True
    frozen_snapshot: Literal[True] = True
    evidence: tuple[Qwen37FeedbackSourceEvidenceV1, ...]
    source_lock_sha256: Sha256

    @field_validator("input_modalities", "output_modalities", "evidence", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if tuple(item.source_id for item in self.evidence) != (
            "model-capabilities",
            "structured-output",
            "deep-thinking",
            "chat-completions",
            "pricing",
        ):
            raise ValueError("Qwen source evidence order or coverage differs")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen model source lock self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen37FeedbackPricingLockV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-pricing-lock"] = (
        "portfolio-s1-qwen37-feedback-pricing-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-pricing-lock-v1"] = (
        QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    pricing_mode: Literal["thinking-realtime-list-price"] = (
        "thinking-realtime-list-price"
    )
    currency: Literal["CNY"] = "CNY"
    tier_min_input_tokens_exclusive: Literal[0] = 0
    tier_max_input_tokens_inclusive: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_cny_per_million_tokens: Literal[2] = (
        QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[8] = (
        QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    reasoning_tokens_billed_as_output_tokens: Literal[True] = True
    thinking_and_answer_output_rate_equal: Literal[True] = True
    wire_max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    discounts_assumed: Literal[False] = False
    free_quota_assumed: Literal[False] = False
    context_cache_assumed: Literal[False] = False
    batch_discount_assumed: Literal[False] = False
    source_url: Literal["https://help.aliyun.com/zh/model-studio/model-pricing"] = (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )
    source_retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    pricing_lock_sha256: Sha256

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen Feedback pricing lock self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class SelectedQwenFeedbackAssetV1(_StrictFrozenModel):
    query_id: str
    asset_id: str
    image_sha256: Sha256

    @field_validator("query_id", "asset_id")
    @classmethod
    def _texts(cls, value: str, info) -> str:
        return _canonical_text(value, info.field_name)


class PortfolioS1QwenFeedbackAuthorizationV3(_StrictFrozenModel):
    """Independent owner approval for exactly the frozen selected48 assets."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v3"
    ] = QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-selected-48-assets"] = (
        "core-opt800-s1-feedback-selected-48-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen37-feedback"] = QWEN37_FEEDBACK_PROCESSOR
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[
        "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    requested_json_schema_strict: Literal[True] = True
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[
        "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    requested_stream: Literal[False] = False
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN37_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN37_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_seed: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    selected_query_count: Literal[48] = QWEN37_FEEDBACK_SELECTED_COUNT
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    pricing_tier_max_input_tokens: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    output_reservation_includes_reasoning_and_answer_tokens: Literal[True] = True
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    parent_membership_processor: Literal["dashscope-qwen-assistant"] = (
        "dashscope-qwen-assistant"
    )
    parent_authority_used_for_role_permission: Literal[False] = False
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
        value = _canonical_text(value, info.field_name)
        if info.field_name == "authorization_id" and not _AUTHORIZATION_ID_RE.fullmatch(
            value
        ):
            raise ValueError("authorization_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen Feedback reviewed_at needs a timezone")
        if len(self.selected_assets) != QWEN37_FEEDBACK_SELECTED_COUNT:
            raise ValueError("Qwen Feedback authorization must bind exactly 48 assets")
        if tuple(item.query_id for item in self.selected_assets) != tuple(
            sorted(item.query_id for item in self.selected_assets)
        ):
            raise ValueError("Qwen Feedback authorized assets must be query-sorted")
        if (
            len({item.query_id for item in self.selected_assets}) != 48
            or len({item.asset_id for item in self.selected_assets}) != 48
        ):
            raise ValueError("Qwen Feedback authorized identities must be unique")
        payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("Qwen Feedback authorized asset set hash mismatch")
        if self.authorization_sha256 != _self_hash(self, "authorization_sha256"):
            raise ValueError("Qwen Feedback authorization self hash mismatch")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        ):
            raise ValueError("Qwen Feedback source or pricing identity drifted")
        if (
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen Feedback role selection identity drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1QwenFeedbackLaunchLockV1(_StrictFrozenModel):
    """Pre-provider lock for one exact selected48 Qwen Feedback run."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-launch-lock"] = (
        "portfolio-s1-qwen37-feedback-launch-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-launch-lock-v1"] = (
        QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION
    )
    run_id: str
    status: Literal["prepared-no-provider-calls"] = "prepared-no-provider-calls"
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen37-feedback"] = QWEN37_FEEDBACK_PROCESSOR
    cache_namespace: Literal["feedback-evaluator-v9"] = QWEN37_FEEDBACK_CACHE_NAMESPACE
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[
        "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    requested_json_schema_strict: Literal[True] = True
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[
        "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    requested_stream: Literal[False] = False
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN37_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN37_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_seed: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    max_attempts_per_selected_query: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    feedback_concurrency: Literal[2] = 2
    canary_call_count: Literal[6] = 6
    selection_sha256: Sha256
    corpus_sha256: Sha256
    authorization_file_sha256: Sha256
    authorization_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    control_file_sha256: Sha256
    control_sha256: Sha256
    pricing_tier_max_input_tokens: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    output_reservation_includes_reasoning_and_answer_tokens: Literal[True] = True
    max_completion_tokens_owner_locked_despite_structured_output_guidance: Literal[
        True
    ] = True
    input_cny_per_million_tokens: Literal[2] = (
        QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[8] = (
        QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    above_pricing_tier_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    provider_calls_performed: Literal[0] = 0
    launch_lock_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        value = _canonical_text(value, "run_id")
        if not _AUTHORIZATION_ID_RE.fullmatch(value):
            raise ValueError("run_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.authorization_file_sha256 == self.authorization_sha256:
            # File/content hashes may coincide for non-self-hashed formats, but this
            # authorization always embeds a self hash and therefore must differ.
            raise ValueError("Qwen authorization file and self hashes must differ")
        if self.launch_lock_sha256 != _self_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen Feedback launch lock self hash mismatch")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        ):
            raise ValueError("Qwen Feedback launch source or pricing drifted")
        if (
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen Feedback launch role selection drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _selection_assets(selection: _Selection) -> tuple[SelectedQwenFeedbackAssetV1, ...]:
    selection_sha256 = getattr(selection, "selection_sha256", None)
    entries = getattr(selection, "entries", None)
    if not isinstance(selection_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", selection_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection lacks a canonical self hash"
        )
    if not isinstance(entries, tuple) or len(entries) != 48:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection must contain exactly 48 entries"
        )
    try:
        assets = tuple(
            SelectedQwenFeedbackAssetV1(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
            for entry in sorted(entries, key=lambda item: item.query_id)
        )
    except (AttributeError, ValueError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection entry binding is invalid"
        ) from error
    return assets


def validate_selected_qwen_feedback_authorization(
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    selection: _Selection,
) -> None:
    assets = _selection_assets(selection)
    expected_set_hash = sha256_bytes(
        canonical_json_bytes([item.model_dump(mode="json") for item in assets])
    )
    if (
        type(authorization) is not PortfolioS1QwenFeedbackAuthorizationV3
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
        or authorization.selected_asset_set_sha256 != expected_set_hash
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback authorization differs from the frozen selected48"
        )


def _require_frozen_source_and_pricing_locks(
    *,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
) -> None:
    if (
        type(model_source_lock) is not Qwen37FeedbackModelSourceLockV1
        or model_source_lock_file_sha256 != QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256
        or model_source_lock.source_lock_sha256 != QWEN37_FEEDBACK_SOURCE_LOCK_SHA256
        or sha256_bytes(model_source_lock.canonical_bytes())
        != model_source_lock_file_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen model source lock differs from the frozen active identity"
        )
    if (
        type(pricing_lock) is not Qwen37FeedbackPricingLockV1
        or pricing_lock_file_sha256 != QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256
        or pricing_lock.pricing_lock_sha256 != QWEN37_FEEDBACK_PRICING_LOCK_SHA256
        or sha256_bytes(pricing_lock.canonical_bytes()) != pricing_lock_file_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen pricing lock differs from the frozen active identity"
        )


def _require_frozen_role_selection(
    *, role_selection_file_sha256: str, role_selection_sha256: str
) -> None:
    if (
        role_selection_file_sha256,
        role_selection_sha256,
    ) != (
        QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
        QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback role selection differs from frozen v8"
        )


def build_selected_qwen_feedback_authorization(
    selection: _Selection,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1QwenFeedbackAuthorizationV3:
    """Build exact selected48 authority; never infer Qwen Feedback from Assistant."""

    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selection_assets(selection)
    parent.catalog.require_verified_files()
    catalog_by_id = {item.asset_id: item for item in parent.catalog.assets}
    for item in assets:
        catalog_asset = catalog_by_id.get(item.asset_id)
        if (
            catalog_asset is None
            or catalog_asset.sha256 != item.image_sha256
            or catalog_asset.cloud_upload_allowed is not True
        ):
            raise PortfolioS1QwenFeedbackGovernanceError(
                "selected48 Qwen asset differs from verified Core catalog membership"
            )
    _require_frozen_source_and_pricing_locks(
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
    )
    _require_frozen_role_selection(
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
    )
    selected_payload = [item.model_dump(mode="json") for item in assets]
    draft = PortfolioS1QwenFeedbackAuthorizationV3.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        owner_statement=owner_statement,
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        parent_remote_catalog_sha256=parent.catalog.catalog_sha256,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        model_source_lock_sha256=model_source_lock.source_lock_sha256,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
        pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        selected_assets=assets,
        selected_asset_set_sha256=sha256_bytes(canonical_json_bytes(selected_payload)),
        authorization_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    authorization = PortfolioS1QwenFeedbackAuthorizationV3.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "authorization_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )
    validate_selected_qwen_feedback_authorization(authorization, selection)
    return authorization


def load_selected_qwen_feedback_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1QwenFeedbackAuthorizationV3:
    content = read_stable_regular_file(
        path,
        label="selected Qwen3.7 Feedback authorization",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization file SHA-256 mismatch"
        )
    try:
        authorization = PortfolioS1QwenFeedbackAuthorizationV3.model_validate_json(
            content, strict=True
        )
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization is invalid"
        ) from error
    if authorization.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization is not canonical JSON"
        )
    return authorization


def build_qwen37_feedback_launch_lock(
    *,
    run_id: str,
    selection_sha256: str,
    corpus_sha256: str,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
    control_file_sha256: str,
    control_sha256: str,
) -> PortfolioS1QwenFeedbackLaunchLockV1:
    """Build a zero-call launch lock only when every governance binding agrees."""

    if (
        authorization.selection_sha256 != selection_sha256
        or authorization.model_source_lock_file_sha256 != model_source_lock_file_sha256
        or authorization.model_source_lock_sha256
        != model_source_lock.source_lock_sha256
        or authorization.pricing_lock_file_sha256 != pricing_lock_file_sha256
        or authorization.pricing_lock_sha256 != pricing_lock.pricing_lock_sha256
        or authorization.role_selection_file_sha256 != role_selection_file_sha256
        or authorization.role_selection_sha256 != role_selection_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback launch governance differs from its authorization"
        )
    _require_frozen_source_and_pricing_locks(
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
    )
    _require_frozen_role_selection(
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
    )
    draft = PortfolioS1QwenFeedbackLaunchLockV1.model_construct(
        run_id=run_id,
        selection_sha256=selection_sha256,
        corpus_sha256=corpus_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        authorization_sha256=authorization.authorization_sha256,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        model_source_lock_sha256=model_source_lock.source_lock_sha256,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
        pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        control_file_sha256=control_file_sha256,
        control_sha256=control_sha256,
        launch_lock_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_lock_sha256"})
    return PortfolioS1QwenFeedbackLaunchLockV1.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "launch_lock_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )


def require_qwen37_feedback_pre_call_budget(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Reserve one Qwen call or fail closed before any provider invocation.

    ``committed_cost_cny`` is settled actual cost plus forfeited and unresolved
    reservations.  The caller persists the returned fixed-scale reservation
    before invoking the provider.
    """

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback provider-call ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY)
    if committed + reservation > Decimal(QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback phase hard cap would be exceeded"
        )
    return QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY


def load_qwen37_feedback_launch_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1QwenFeedbackLaunchLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback launch lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock file SHA-256 mismatch"
        )
    try:
        lock = PortfolioS1QwenFeedbackLaunchLockV1.model_validate_json(
            content, strict=True
        )
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock is not canonical JSON"
        )
    return lock


def load_qwen37_feedback_model_source_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen37FeedbackModelSourceLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback model source lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen37FeedbackModelSourceLockV1.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock is not canonical JSON"
        )
    return lock


def load_qwen37_feedback_pricing_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen37FeedbackPricingLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback pricing lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen37FeedbackPricingLockV1.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock is not canonical JSON"
        )
    return lock


def write_selected_qwen_feedback_authorization(
    path: str | Path,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def write_qwen37_feedback_launch_lock(
    path: str | Path,
    launch_lock: PortfolioS1QwenFeedbackLaunchLockV1,
) -> Path:
    return atomic_create_file(path, launch_lock.canonical_bytes())


__all__ = [
    "PortfolioS1QwenFeedbackAuthorizationV3",
    "PortfolioS1QwenFeedbackGovernanceError",
    "PortfolioS1QwenFeedbackLaunchLockV1",
    "QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION",
    "QWEN37_FEEDBACK_CACHE_NAMESPACE",
    "QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS",
    "QWEN37_FEEDBACK_ENDPOINT",
    "QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION",
    "QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS",
    "QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS",
    "QWEN37_FEEDBACK_JSON_SCHEMA_NAME",
    "QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION",
    "QWEN37_FEEDBACK_JSON_SCHEMA_SHA256",
    "QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION",
    "QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS",
    "QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE",
    "QWEN37_FEEDBACK_MODEL",
    "QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS",
    "QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS",
    "QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY",
    "QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY",
    "QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY",
    "QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION",
    "QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256",
    "QWEN37_FEEDBACK_PRICING_LOCK_SHA256",
    "QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS",
    "QWEN37_FEEDBACK_PROCESSOR",
    "QWEN37_FEEDBACK_PROVIDER",
    "QWEN37_FEEDBACK_PROVIDER_CALL_CEILING",
    "QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS",
    "QWEN37_FEEDBACK_SELECTED_COUNT",
    "QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256",
    "QWEN37_FEEDBACK_ROLE_SELECTION_SHA256",
    "QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256",
    "QWEN37_FEEDBACK_SOURCE_LOCK_SHA256",
    "QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION",
    "QWEN37_FEEDBACK_THINKING_BUDGET",
    "QWEN37_FEEDBACK_TIMEOUT_SECONDS",
    "QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256",
    "QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION",
    "Qwen37FeedbackModelSourceLockV1",
    "Qwen37FeedbackPricingLockV1",
    "Qwen37FeedbackSourceEvidenceV1",
    "SelectedQwenFeedbackAssetV1",
    "build_selected_qwen_feedback_authorization",
    "build_qwen37_feedback_launch_lock",
    "load_qwen37_feedback_launch_lock",
    "load_qwen37_feedback_model_source_lock",
    "load_qwen37_feedback_pricing_lock",
    "load_selected_qwen_feedback_authorization",
    "require_qwen37_feedback_pre_call_budget",
    "validate_selected_qwen_feedback_authorization",
    "write_qwen37_feedback_launch_lock",
    "write_selected_qwen_feedback_authorization",
]
