"""Zero-call launch package for the Portfolio ``dev_mini`` 200 x 5 matrix.

This module prepares the complete ordered workload, but it cannot execute a
model.  It intentionally keeps three states separate:

* input/permission/model configuration is verified;
* treatment Banks and the Assistant tool runtime are present and bound;
* the operator has separately authorized a budget large enough for the run.

The package is create-only and records enough identity to resume by 25-query
shard without silently changing the corpus, model roles, or configuration
order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.evaluation.assistant_runs import (
    AssistantRunConfig,
    MAIN_CONFIG_ORDER,
)
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.final_runtime import (
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    FINAL_JUDGE_RETRY_POLICY_SHA256,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_THINKING_BUDGET,
    FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_inputs import (
    ACTIVE_PORTFOLIO_PROCESSOR_ORDER,
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
    PortfolioRemoteProcessingFiles,
    VerifiedPortfolioDevMiniInputs,
    require_verified_portfolio_dev_mini_inputs,
)
from skillchain.evaluation.portfolio_parallel import (
    MAX_SAFE_ASSISTANT_CONCURRENCY,
)
from skillchain.evaluation.portfolio_core_inputs import (
    CORE_BATCH_SIZE,
    CORE_QUERY_COUNT,
    CORE_SPLIT_ORDER,
    CoreSplit,
    PortfolioCoreInputError,
    PortfolioCoreInputFiles,
    VerifiedPortfolioCoreInputs,
    load_verified_portfolio_core_inputs,
    require_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
    PortfolioTreatmentChainManifest,
)
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_BUDGET_CURRENCY,
    PORTFOLIO_BUDGET_DECIMAL_PLACES,
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PORTFOLIO_GEMINI_MODEL,
    PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION,
    PORTFOLIO_KIMI_MODEL,
    PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION,
    PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS,
    PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_MODEL,
    PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS,
    PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION,
    PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes
from skillchain.tools.serialization import (
    ArtifactFormatError,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PORTFOLIO_LAUNCH_POLICY_VERSION = "portfolio-dev-mini-200x5-launch-v3"
PORTFOLIO_CORE_LAUNCH_POLICY_VERSION = "portfolio-core-split-25qx5-launch-v1"
PORTFOLIO_STATIC_OPT_LAUNCH_POLICY_VERSION = "portfolio-core-static-opt-800x1-launch-v1"
PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION = "portfolio-provider-pricing-contract-v3"
_PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V1 = (
    "portfolio-provider-pricing-contract-v1"
)
_PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V2 = (
    "portfolio-provider-pricing-contract-v2"
)
PORTFOLIO_ROLE_SELECTION_FILE_SHA256 = (
    "fbfbe9437731052743b3025962a22e4d3cd6432c2212b7cc418b2118e9fd2ba4"
)
PORTFOLIO_ROLE_SELECTION_SHA256 = (
    "fc3e8ad7a1b2ace95ffe3df4c775a8c38425274459853d555dabb2c836097633"
)
_PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256 = (
    "86d5b7763dfee3087d8a9df05659387ca1033093fc021d806643864080e770f9"
)
_PORTFOLIO_ROLE_SELECTION_V6_SHA256 = (
    "940f886e40fd0438197384b4de9042efe4f620d8b7a8557060e36f407b0294a0"
)
_PORTFOLIO_ROLE_SELECTION_V5_FILE_SHA256 = (
    "7fc6b0ba20ab542478ce5e3d5d53974ff619333a657c3b806eca88584ac62e8c"
)
_PORTFOLIO_ROLE_SELECTION_V5_SHA256 = (
    "b0d65f8ffa5b6ce237cd9907ba9fe4c64129706596cf611b62829c396e8458f3"
)
_PORTFOLIO_V5_PROCESSOR_ORDER = (
    "dashscope-qwen-assistant",
    "aifast-gemini-feedback",
    "dashscope-kimi-judge",
)
PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256 = (
    "da8481bb9855df69823db3276e31bd8eac318677b37c6f9f807b3befeec3d4aa"
)
PORTFOLIO_CORE_ROLE_SELECTION_SHA256 = (
    "8cbc91be884ac0809c8a99ffa254e36fa045a560e30b497b278e17565860a441"
)
EXPECTED_QUERY_COUNT = 200
EXPECTED_CONFIG_COUNT = 5
EXPECTED_INSTANCE_COUNT = 1000
EXPECTED_BATCH_COUNT = 8
EXPECTED_BATCH_SIZE = 25
MAX_QUERY_COUNT = CORE_QUERY_COUNT
MAX_INSTANCE_COUNT = CORE_QUERY_COUNT * EXPECTED_CONFIG_COUNT
MAX_SHARD_COUNT = (CORE_QUERY_COUNT // CORE_BATCH_SIZE) * EXPECTED_CONFIG_COUNT
FINAL_JUDGE_EMPTY_RESPONSE_RETRY_CALL_CEILING = EXPECTED_INSTANCE_COUNT * (
    FINAL_JUDGE_MAX_ATTEMPTS - 1
)
AUTONOMOUS_DASHSCOPE_BUDGET_CNY = 10.0
DEFAULT_PLANNING_CEILING_CNY = 51.0

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BANK_CONFIGS: tuple[AssistantRunConfig, ...] = (
    "llm_static",
    "s1",
    "s1s2",
    "full",
)
_PLAN_NORMALIZATION_MARKER = object()


def _runtime_execution_contract_success_detail() -> str:
    return (
        f"runtime binds Judge schema {FINAL_JUDGE_RESULT_SCHEMA_VERSION}, "
        "thinking reserves, and the provider-call hard-budget contract"
    )


class PortfolioLaunchError(ValueError):
    """Launch preparation found drift or an unsafe artifact."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _hash_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256_bytes(canonical_json_bytes(value))


_PORTFOLIO_BUDGET_POLICY = {
    "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
    "currency": PORTFOLIO_BUDGET_CURRENCY,
    "decimal_places": PORTFOLIO_BUDGET_DECIMAL_PLACES,
    "authorization": "create_only_phase_cap_before_first_provider_call",
    "accounting": (
        "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
    ),
    "exception_policy": "append_full_reserve_forfeit_without_captured_response",
    "terminal_outcome_policy": "exactly_one_of_settlement_or_forfeit",
    "over_budget_policy": "reserve_before_each_provider_call_halt_before_call",
}
PORTFOLIO_BUDGET_POLICY_SHA256 = _hash_payload(_PORTFOLIO_BUDGET_POLICY)
_PORTFOLIO_BUDGET_POLICY_V2 = {
    **_PORTFOLIO_BUDGET_POLICY,
    "policy_version": "portfolio-call-hard-cap-v2",
}
PORTFOLIO_BUDGET_POLICY_SHA256_V2 = _hash_payload(_PORTFOLIO_BUDGET_POLICY_V2)
_PORTFOLIO_BUDGET_POLICY_V1 = {
    "policy_version": "portfolio-call-hard-cap-v1",
    "currency": PORTFOLIO_BUDGET_CURRENCY,
    "decimal_places": PORTFOLIO_BUDGET_DECIMAL_PLACES,
    "authorization": "create_only_phase_cap_before_first_provider_call",
    "accounting": "settled_actual_plus_unresolved_reserve",
    "exception_policy": "retain_full_reserve",
    "over_budget_policy": "reserve_before_each_provider_call_halt_before_call",
}
PORTFOLIO_BUDGET_POLICY_SHA256_V1 = _hash_payload(_PORTFOLIO_BUDGET_POLICY_V1)

_PORTFOLIO_PROVIDER_PRICING_CONTRACT = {
    "contract_version": PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    "qwen": {
        "model": PORTFOLIO_QWEN_MODEL,
        "stage_max_input_tokens": PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS,
        "provider_max_input_tokens": PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS,
        "route_max_output_tokens": PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
        "action_max_output_tokens": PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS,
        "reserve_input_cny_per_million_tokens": str(
            PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION
        ),
        "reserve_output_cny_per_million_tokens": str(
            PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION
        ),
        "settlement_pricing_policy": "qwen_input_length_tiers_v1",
    },
    "gemini": {
        "model": PORTFOLIO_GEMINI_MODEL,
        "pricing_status": PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
        "provider_input_token_reserve": None,
        "provider_output_token_limit": None,
        "provider_output_token_reserve": None,
        "reservation_allowed": False,
    },
}
_PORTFOLIO_PROVIDER_PRICING_CONTRACT_V2 = {
    "contract_version": _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V2,
    "qwen": {
        **_PORTFOLIO_PROVIDER_PRICING_CONTRACT["qwen"],
    },
    "kimi": {
        "model": PORTFOLIO_KIMI_MODEL,
        "provider_input_token_reserve": PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS,
        "provider_output_token_limit": PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS,
        "provider_output_token_reserve": 8_192,
        "thinking_budget": 6_144,
        "reserve_input_cny_per_million_tokens": str(
            PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION
        ),
        "reserve_output_cny_per_million_tokens": str(
            PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION
        ),
        "settlement_pricing_policy": "kimi_fixed_v1",
    },
}
_PORTFOLIO_PROVIDER_PRICING_CONTRACT_V1 = {
    **_PORTFOLIO_PROVIDER_PRICING_CONTRACT_V2,
    "contract_version": _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V1,
    "qwen": {
        **_PORTFOLIO_PROVIDER_PRICING_CONTRACT_V2["qwen"],
        "route_max_output_tokens": PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS,
    },
}
PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V1 = _hash_payload(
    _PORTFOLIO_PROVIDER_PRICING_CONTRACT_V1
)
PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256 = _hash_payload(
    _PORTFOLIO_PROVIDER_PRICING_CONTRACT
)
PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2 = _hash_payload(
    _PORTFOLIO_PROVIDER_PRICING_CONTRACT_V2
)


def _self_hash(value: BaseModel, field_name: str) -> str:
    return _hash_payload(value.model_dump(mode="json", exclude={field_name}))


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


class PortfolioPreflightCheck(_StrictFrozenModel):
    check_id: str
    status: Literal["passed", "blocked", "warning"]
    detail: str

    @field_validator("check_id", "detail")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class PortfolioArtifactBinding(_StrictFrozenModel):
    artifact_id: str
    path: str | None
    file_sha256: Sha256 | None
    content_sha256: Sha256 | None
    status: Literal["verified", "missing", "invalid"]
    detail: str

    @field_validator("artifact_id", "detail")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> "PortfolioArtifactBinding":
        if self.status == "verified":
            if self.path is None or self.file_sha256 is None:
                raise ValueError("verified artifact requires path and file digest")
        elif self.file_sha256 is not None or self.content_sha256 is not None:
            raise ValueError("unverified artifact cannot retain trusted digests")
        return self


class PortfolioLaunchInstance(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-launch-instance"] = "portfolio-launch-instance"
    matrix_run_id: str
    instance_ordinal: int = Field(ge=0, lt=MAX_INSTANCE_COUNT)
    shard_id: str
    shard_ordinal: int = Field(ge=0, lt=MAX_SHARD_COUNT)
    query_ordinal: int = Field(ge=0, lt=MAX_QUERY_COUNT)
    config_ordinal: int = Field(ge=0, le=4)
    config: AssistantRunConfig
    accepted_batch_id: str
    query_id: str
    query_sha256: Sha256
    public_input_sha256: Sha256
    # Historical dev_mini packages retain the private catalog binding.  New
    # Core packages carry only the opaque model-visible token and the hash of
    # the private AssistantAssetBinding; the real asset/path stays in the
    # separately verified query artifact.
    asset_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    image_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    asset_token: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    asset_binding_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    image_sha256: Sha256
    assistant_output_relpath: str
    final_output_relpath: str
    instance_sha256: Sha256

    @field_validator(
        "matrix_run_id",
        "shard_id",
        "accepted_batch_id",
        "query_id",
        "assistant_output_relpath",
        "final_output_relpath",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID.fullmatch(value):
            raise ValueError("matrix_run_id contains unsafe characters")
        return value

    @field_validator("assistant_output_relpath", "final_output_relpath")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("output path must be normalized relative POSIX")
        return value

    @model_validator(mode="after")
    def validate_instance(self) -> "PortfolioLaunchInstance":
        if MAIN_CONFIG_ORDER[self.config_ordinal] != self.config:
            raise ValueError("config ordinal differs from canonical order")
        expected_assistant = f"shards/{self.shard_id}/assistant/{self.query_id}.json"
        expected_final = f"shards/{self.shard_id}/final/{self.query_id}.json"
        if (
            self.assistant_output_relpath != expected_assistant
            or self.final_output_relpath != expected_final
        ):
            raise ValueError("instance output layout drifted")
        legacy_private = self.asset_id is not None or self.image_path is not None
        opaque_core = (
            self.asset_token is not None or self.asset_binding_sha256 is not None
        )
        if legacy_private == opaque_core:
            raise ValueError(
                "launch instance requires exactly one private-dev or "
                "opaque-Core asset binding"
            )
        if legacy_private and (self.asset_id is None or self.image_path is None):
            raise ValueError("legacy launch asset binding is incomplete")
        if opaque_core and (
            self.asset_token is None or self.asset_binding_sha256 is None
        ):
            raise ValueError("Core launch asset binding is incomplete")
        for label, value in (
            ("asset_id", self.asset_id),
            ("image_path", self.image_path),
            ("asset_token", self.asset_token),
        ):
            if value is not None:
                _nonblank(value, label)
        if self.instance_sha256 != _self_hash(self, "instance_sha256"):
            raise ValueError("launch instance self hash mismatch")
        return self


class PortfolioLaunchShard(_StrictFrozenModel):
    shard_id: str
    shard_ordinal: int = Field(ge=0, lt=MAX_SHARD_COUNT)
    accepted_batch_id: str
    config: AssistantRunConfig
    config_ordinal: int = Field(ge=0, le=4)
    query_count: Literal[25] = 25
    query_ids: tuple[str, ...]
    instance_sha256s: tuple[Sha256, ...]
    output_relpath: str
    shard_sha256: Sha256

    @field_validator("query_ids", "instance_sha256s", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("shard_id", "accepted_batch_id", "output_relpath")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_shard(self) -> "PortfolioLaunchShard":
        if MAIN_CONFIG_ORDER[self.config_ordinal] != self.config:
            raise ValueError("shard config ordinal differs from canonical order")
        if (
            len(self.query_ids) != EXPECTED_BATCH_SIZE
            or len(set(self.query_ids)) != EXPECTED_BATCH_SIZE
            or len(self.instance_sha256s) != EXPECTED_BATCH_SIZE
            or len(set(self.instance_sha256s)) != EXPECTED_BATCH_SIZE
        ):
            raise ValueError("launch shard must bind 25 unique instances")
        if self.output_relpath != f"shards/{self.shard_id}":
            raise ValueError("launch shard output path drifted")
        if self.shard_sha256 != _self_hash(self, "shard_sha256"):
            raise ValueError("launch shard self hash mismatch")
        return self


class PortfolioBudgetPolicy(_StrictFrozenModel):
    policy_version: (
        Literal[
            "portfolio-call-hard-cap-v1",
            "portfolio-call-hard-cap-v2",
            "portfolio-call-hard-cap-v3",
        ]
        | None
    ) = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    policy_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    provider_pricing_contract_version: (
        Literal[
            "portfolio-provider-pricing-contract-v1",
            "portfolio-provider-pricing-contract-v2",
            "portfolio-provider-pricing-contract-v3",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    provider_pricing_contract_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    currency: Literal["CNY"] = "CNY"
    autonomous_dashscope_budget_cny: float = Field(gt=0)
    operator_approved_dashscope_budget_cny: float = Field(gt=0)
    planning_ceiling_cny: float = Field(gt=0)
    qwen_input_cny_per_million_tokens: float = Field(gt=0)
    qwen_output_cny_per_million_tokens: float = Field(gt=0)
    kimi_input_cny_per_million_tokens: float = Field(gt=0)
    kimi_output_cny_per_million_tokens: float = Field(gt=0)
    pricing_basis: str
    aifast_cost_status: Literal["gateway_price_not_locked"]
    checkpoint_policy: Literal["after_every_25_query_shard"]
    over_budget_policy: Literal[
        "halt_before_next_shard",
        "reserve_before_each_provider_call_halt_before_call",
    ]

    @model_validator(mode="after")
    def validate_contract(self, info: ValidationInfo) -> "PortfolioBudgetPolicy":
        contract = (
            self.policy_version,
            self.policy_sha256,
            self.provider_pricing_contract_version,
            self.provider_pricing_contract_sha256,
        )
        if all(value is None for value in contract):
            if self.over_budget_policy != "halt_before_next_shard":
                raise ValueError(
                    "historical launch budget cannot claim provider-call reserves"
                )
            return self
        current_contract = (
            self.provider_pricing_contract_version
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            and self.provider_pricing_contract_sha256
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        )
        historical_v2_contract = (
            self.provider_pricing_contract_version
            == _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V2
            and self.provider_pricing_contract_sha256
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2
        )
        legacy_v1_contract = (
            self.provider_pricing_contract_version
            == _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V1
            and self.provider_pricing_contract_sha256
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V1
        )
        reserve_policy = (
            self.over_budget_policy
            == "reserve_before_each_provider_call_halt_before_call"
        )
        active_budget = (
            self.policy_version == PORTFOLIO_BUDGET_POLICY_VERSION
            and self.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256
            and current_contract
            and reserve_policy
        )
        allow_legacy = bool(
            isinstance(info.context, dict)
            and info.context.get("allow_legacy_budget_contract") is True
        )
        historical_v2_budget = (
            self.policy_version == "portfolio-call-hard-cap-v2"
            and self.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256_V2
            and historical_v2_contract
            and reserve_policy
        )
        legacy_v1_budget = (
            self.policy_version == "portfolio-call-hard-cap-v1"
            and self.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256_V1
            # Frozen launch-v23 combined the v1 ledger policy with the then-active
            # v2 provider-pricing contract.  The internal legacy-read context may
            # accept either exact historical pricing identity, but never v3.
            and (historical_v2_contract or legacy_v1_contract)
            and reserve_policy
        )
        if active_budget or historical_v2_budget or (allow_legacy and legacy_v1_budget):
            return self
        raise ValueError("Portfolio launch budget contract drifted")


class PortfolioRatePolicy(_StrictFrozenModel):
    assistant_concurrency: Literal[1, 2] = MAX_SAFE_ASSISTANT_CONCURRENCY
    final_judge_concurrency: Literal[0, 1] = 1
    feedback_concurrency: Literal[1] = 1
    qwen_requests_per_minute_cap: Literal[50] = 50
    provider_retry_attempts: Literal[1] = Field(
        default=1,
        description=(
            "Generic adapter attempts per individual provider call; the final-Judge "
            "empty-response semantic retry is owned and accounted for separately by "
            "the final-Judge runtime and call plan."
        ),
    )
    checkpoint_policy: Literal["after_every_25_query_shard"]


class PortfolioCallPlan(_StrictFrozenModel):
    assistant_instance_count: int = Field(
        default=EXPECTED_INSTANCE_COUNT, ge=125, le=MAX_INSTANCE_COUNT
    )
    final_judge_instance_count: int = Field(
        default=EXPECTED_INSTANCE_COUNT, ge=0, le=MAX_INSTANCE_COUNT
    )
    assistant_call_floor: int = Field(ge=200)
    assistant_call_ceiling: int = Field(ge=200)
    final_judge_call_count: int = Field(
        default=EXPECTED_INSTANCE_COUNT, ge=0, le=MAX_INSTANCE_COUNT
    )
    final_judge_empty_response_retry_call_ceiling: int | None = Field(
        default=None,
        ge=0,
        le=MAX_INSTANCE_COUNT,
        exclude_if=lambda value: value is None,
    )
    feedback_calls_in_main_matrix: Literal[0] = 0
    total_call_floor: int = Field(ge=200)
    total_call_ceiling: int = Field(ge=200)
    note: str

    @model_validator(mode="after")
    def validate_call_bounds(self) -> "PortfolioCallPlan":
        retry_ceiling = self.final_judge_empty_response_retry_call_ceiling or 0
        final_enabled = (
            self.final_judge_instance_count == self.assistant_instance_count
            and self.final_judge_call_count == self.assistant_instance_count
        )
        final_disabled = (
            self.final_judge_instance_count == 0
            and self.final_judge_call_count == 0
            and retry_ceiling == 0
        )
        if (
            not (final_enabled or final_disabled)
            or self.assistant_call_floor > self.assistant_call_ceiling
            or self.total_call_floor
            != self.assistant_call_floor + self.final_judge_call_count
            or self.total_call_ceiling
            != self.assistant_call_ceiling + self.final_judge_call_count + retry_ceiling
        ):
            raise ValueError("Portfolio call-plan bounds are inconsistent")
        return self


def _bound_absolute_path(path: Path) -> str:
    return path.absolute().as_posix()


class PortfolioCoreRemoteProcessingFilesBinding(_StrictFrozenModel):
    """Exact permission-chain files needed to rebuild Core preflight."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-core-remote-processing-files-binding"] = (
        "portfolio-core-remote-processing-files-binding"
    )
    authorization_file: str
    expected_authorization_file_sha256: Sha256
    receipt_file: str
    expected_receipt_file_sha256: Sha256
    expected_receipt_sha256: Sha256
    selection_manifest: str
    expected_selection_manifest_sha256: Sha256
    dataset_assets: str
    expected_dataset_assets_sha256: Sha256
    base_catalog_dir: str
    expected_base_catalog_sha256: Sha256
    output_catalog_dir: str
    expected_output_catalog_sha256: Sha256
    asset_root: str
    processor_order: tuple[str, ...]

    @field_validator(
        "authorization_file",
        "receipt_file",
        "selection_manifest",
        "dataset_assets",
        "base_catalog_dir",
        "output_catalog_dir",
        "asset_root",
    )
    @classmethod
    def validate_absolute_paths(cls, value: str, info: ValidationInfo) -> str:
        _nonblank(value, info.field_name)
        candidate = Path(value)
        if not candidate.is_absolute() or candidate.absolute().as_posix() != value:
            raise ValueError(
                f"{info.field_name} must be a normalized absolute POSIX path"
            )
        return value

    @field_validator("processor_order", mode="before")
    @classmethod
    def coerce_processor_order(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_processor_order(self) -> "PortfolioCoreRemoteProcessingFilesBinding":
        if self.processor_order not in {
            CORE_PORTFOLIO_PROCESSOR_ORDER,
            ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
        }:
            raise ValueError("Core permission binding processor order drifted")
        return self


class PortfolioCoreInputFilesBinding(_StrictFrozenModel):
    """Self-contained local-file recipe for reconstructing verified Core inputs."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-core-input-files-binding"] = (
        "portfolio-core-input-files-binding"
    )
    plan_path: str
    expected_plan_sha256: Sha256
    pre_generation_manifest_path: str
    expected_pre_generation_manifest_file_sha256: Sha256
    split_assignment_path: str
    expected_split_assignment_sha256: Sha256
    query_artifact_path: str
    expected_query_artifact_sha256: Sha256
    materialization_manifest_path: str
    expected_materialization_manifest_file_sha256: Sha256
    capability_assignments_path: str
    expected_capability_assignments_sha256: Sha256
    base_catalog_dir: str
    expected_base_catalog_sha256: Sha256
    asset_root: str
    runtime_catalog_dir: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    expected_runtime_catalog_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    remote_files: PortfolioCoreRemoteProcessingFilesBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator(
        "plan_path",
        "pre_generation_manifest_path",
        "split_assignment_path",
        "query_artifact_path",
        "materialization_manifest_path",
        "capability_assignments_path",
        "base_catalog_dir",
        "asset_root",
        "runtime_catalog_dir",
    )
    @classmethod
    def validate_absolute_paths(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return value
        _nonblank(value, info.field_name)
        candidate = Path(value)
        if not candidate.is_absolute() or candidate.absolute().as_posix() != value:
            raise ValueError(
                f"{info.field_name} must be a normalized absolute POSIX path"
            )
        return value

    @model_validator(mode="after")
    def validate_file_recipe(self) -> "PortfolioCoreInputFilesBinding":
        if (self.runtime_catalog_dir is None) != (
            self.expected_runtime_catalog_sha256 is None
        ):
            raise ValueError(
                "Core runtime catalog path and SHA-256 must be supplied together"
            )
        if self.remote_files is not None:
            remote = self.remote_files
            if (
                remote.base_catalog_dir != self.base_catalog_dir
                or remote.expected_base_catalog_sha256
                != self.expected_base_catalog_sha256
                or remote.output_catalog_dir
                != (self.runtime_catalog_dir or self.base_catalog_dir)
                or remote.expected_output_catalog_sha256
                != (
                    self.expected_runtime_catalog_sha256
                    or self.expected_base_catalog_sha256
                )
                or remote.asset_root != self.asset_root
            ):
                raise ValueError(
                    "Core permission files differ from the input catalog recipe"
                )
        return self

    def to_input_files(self) -> PortfolioCoreInputFiles:
        remote = self.remote_files
        return PortfolioCoreInputFiles(
            plan_path=Path(self.plan_path),
            expected_plan_sha256=self.expected_plan_sha256,
            pre_generation_manifest_path=Path(self.pre_generation_manifest_path),
            expected_pre_generation_manifest_file_sha256=(
                self.expected_pre_generation_manifest_file_sha256
            ),
            split_assignment_path=Path(self.split_assignment_path),
            expected_split_assignment_sha256=self.expected_split_assignment_sha256,
            query_artifact_path=Path(self.query_artifact_path),
            expected_query_artifact_sha256=self.expected_query_artifact_sha256,
            materialization_manifest_path=Path(self.materialization_manifest_path),
            expected_materialization_manifest_file_sha256=(
                self.expected_materialization_manifest_file_sha256
            ),
            capability_assignments_path=Path(self.capability_assignments_path),
            expected_capability_assignments_sha256=(
                self.expected_capability_assignments_sha256
            ),
            base_catalog_dir=Path(self.base_catalog_dir),
            expected_base_catalog_sha256=self.expected_base_catalog_sha256,
            asset_root=Path(self.asset_root),
            runtime_catalog_dir=(
                None
                if self.runtime_catalog_dir is None
                else Path(self.runtime_catalog_dir)
            ),
            expected_runtime_catalog_sha256=(self.expected_runtime_catalog_sha256),
            remote_files=(
                None
                if remote is None
                else PortfolioRemoteProcessingFiles(
                    authorization_file=Path(remote.authorization_file),
                    expected_authorization_file_sha256=(
                        remote.expected_authorization_file_sha256
                    ),
                    receipt_file=Path(remote.receipt_file),
                    expected_receipt_file_sha256=(remote.expected_receipt_file_sha256),
                    selection_manifest=Path(remote.selection_manifest),
                    dataset_assets=Path(remote.dataset_assets),
                    base_catalog_dir=Path(remote.base_catalog_dir),
                    output_catalog_dir=Path(remote.output_catalog_dir),
                    asset_root=Path(remote.asset_root),
                    processor_order=remote.processor_order,
                )
            ),
        )


class PortfolioLaunchPlan(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal[
        "portfolio-dev-mini-200x5-launch-plan",
        "portfolio-core-split-x5-launch-plan",
        "portfolio-core-static-opt-800x1-launch-plan",
    ] = "portfolio-dev-mini-200x5-launch-plan"
    policy_version: Literal[
        "portfolio-dev-mini-200x5-launch-v1",
        "portfolio-dev-mini-200x5-launch-v2",
        "portfolio-dev-mini-200x5-launch-v3",
        "portfolio-core-split-25qx5-launch-v1",
        "portfolio-core-static-opt-800x1-launch-v1",
    ] = PORTFOLIO_LAUNCH_POLICY_VERSION
    execution_mode: Literal["full_matrix", "static_opt_rollout"] = Field(
        default="full_matrix",
        exclude_if=lambda value: value == "full_matrix",
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    dataset_profile: Literal["core"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_query_count: Literal[1500] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    selected_splits: tuple[CoreSplit, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    split_assignment_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    materialization_manifest_file_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    core_input_files: PortfolioCoreInputFilesBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    matrix_run_id: str
    status: Literal["prepared_ready", "prepared_blocked"]
    execution_ready: bool
    execution_authorized: Literal[False] = False
    model_calls_performed: Literal[0] = 0
    query_count: int = Field(ge=EXPECTED_BATCH_SIZE, le=MAX_QUERY_COUNT)
    config_count: Literal[1, 5] = 5
    instance_count: int = Field(ge=125, le=MAX_INSTANCE_COUNT)
    shard_count: int = Field(ge=5, le=MAX_SHARD_COUNT)
    config_order: tuple[AssistantRunConfig, ...]
    portfolio_plan_sha256: Sha256
    portfolio_plan_manifest_file_sha256: Sha256
    accepted_ledger_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    query_artifact_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed_set_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    base_catalog_sha256: Sha256
    runtime_catalog_sha256: Sha256
    authorization_file_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    receipt_file_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    receipt_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    active_processor_order: tuple[str, ...]
    remote_runtime_binding_sha256s: tuple[Sha256, ...]
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    assistant_provider: Literal["qwen"] = "qwen"
    assistant_model: Literal["qwen3-vl-flash-2026-01-22"] = "qwen3-vl-flash-2026-01-22"
    feedback_provider: Literal["gemini", "kimi"] = "gemini"
    feedback_model: Literal["gemini-3.6-flash", "kimi-k2.6"] = "gemini-3.6-flash"
    final_provider: Literal["gemini", "kimi"] = "gemini"
    final_model: Literal["gemini-3.6-flash", "kimi-k2.6"] = "gemini-3.6-flash"
    artifacts: tuple[PortfolioArtifactBinding, ...]
    preflight_checks: tuple[PortfolioPreflightCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    budget: PortfolioBudgetPolicy
    rate: PortfolioRatePolicy
    calls: PortfolioCallPlan
    resume_policy: Literal["create_only_shards_skip_only_verified_complete"]
    scheduling_policy: Literal[
        "balanced_cyclic_config_order_by_accepted_batch",
        "balanced_cyclic_config_order_by_split_atomic_generator_batch",
        "static_opt_pool_atomic_generator_batch",
    ]
    instances_file_sha256: Sha256
    shards: tuple[PortfolioLaunchShard, ...]
    launch_plan_sha256: Sha256

    @field_validator(
        "config_order",
        "selected_splits",
        "active_processor_order",
        "remote_runtime_binding_sha256s",
        "artifacts",
        "preflight_checks",
        "blockers",
        "warnings",
        "shards",
        mode="before",
    )
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        value = _nonblank(value, "matrix_run_id")
        if not _RUN_ID.fullmatch(value):
            raise ValueError("matrix_run_id contains unsafe characters")
        return value

    @model_validator(mode="after")
    def validate_plan(self, info: ValidationInfo) -> "PortfolioLaunchPlan":
        static_opt = self.kind == "portfolio-core-static-opt-800x1-launch-plan"
        core = self.kind in {
            "portfolio-core-split-x5-launch-plan",
            "portfolio-core-static-opt-800x1-launch-plan",
        }
        expected_config_order = ("llm_static",) if static_opt else MAIN_CONFIG_ORDER
        if self.config_order != expected_config_order:
            raise ValueError("launch config order differs from execution mode")
        active_budget = (
            self.budget.policy_version == PORTFOLIO_BUDGET_POLICY_VERSION
            and self.budget.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256
            and self.budget.provider_pricing_contract_version
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            and self.budget.provider_pricing_contract_sha256
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        )
        allow_legacy = bool(
            isinstance(info.context, dict)
            and info.context.get("allow_legacy_budget_contract") is True
        )
        historical_v2_budget = (
            self.budget.policy_version == "portfolio-call-hard-cap-v2"
            and self.budget.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256_V2
            and self.budget.provider_pricing_contract_version
            == _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V2
            and self.budget.provider_pricing_contract_sha256
            == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2
        )
        legacy_v1_budget = (
            self.budget.policy_version == "portfolio-call-hard-cap-v1"
            and self.budget.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256_V1
            and (
                (
                    self.budget.provider_pricing_contract_version
                    == _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V2
                    and self.budget.provider_pricing_contract_sha256
                    == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2
                )
                or (
                    self.budget.provider_pricing_contract_version
                    == _PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION_V1
                    and self.budget.provider_pricing_contract_sha256
                    == PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V1
                )
            )
        )
        if self.policy_version in {
            PORTFOLIO_LAUNCH_POLICY_VERSION,
            PORTFOLIO_CORE_LAUNCH_POLICY_VERSION,
            PORTFOLIO_STATIC_OPT_LAUNCH_POLICY_VERSION,
        } and not (
            active_budget or historical_v2_budget or (allow_legacy and legacy_v1_budget)
        ):
            raise ValueError("active launch does not bind the hard-budget contract")
        historical_dev_v5 = (
            not core
            and self.role_selection_file_sha256
            == _PORTFOLIO_ROLE_SELECTION_V5_FILE_SHA256
            and self.role_selection_sha256 == _PORTFOLIO_ROLE_SELECTION_V5_SHA256
        )
        historical_core_v4 = (
            core
            and self.role_selection_file_sha256
            == PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256
            and self.role_selection_sha256 == PORTFOLIO_CORE_ROLE_SELECTION_SHA256
        )
        role_swapped_v6_or_v7 = (
            (
                self.role_selection_file_sha256,
                self.role_selection_sha256,
            )
            in {
                (
                    _PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256,
                    _PORTFOLIO_ROLE_SELECTION_V6_SHA256,
                ),
                (
                    PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
                    PORTFOLIO_ROLE_SELECTION_SHA256,
                ),
            }
        )
        expected_processor_order = (
            CORE_PORTFOLIO_PROCESSOR_ORDER
            if historical_core_v4
            else ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
            if core and role_swapped_v6_or_v7
            else _PORTFOLIO_V5_PROCESSOR_ORDER
            if historical_dev_v5
            else ACTIVE_PORTFOLIO_PROCESSOR_ORDER
        )
        if self.active_processor_order != expected_processor_order:
            raise ValueError("launch plan processor order differs from its profile")
        if len(self.remote_runtime_binding_sha256s) not in ({0, 3} if core else {3}):
            raise ValueError("launch plan has an invalid remote-runtime cardinality")
        if core:
            if (
                self.policy_version
                != (
                    PORTFOLIO_STATIC_OPT_LAUNCH_POLICY_VERSION
                    if static_opt
                    else PORTFOLIO_CORE_LAUNCH_POLICY_VERSION
                )
                or self.dataset_profile != "core"
                or self.source_query_count != CORE_QUERY_COUNT
                or not self.selected_splits
                or tuple(
                    split for split in CORE_SPLIT_ORDER if split in self.selected_splits
                )
                != self.selected_splits
                or len(set(self.selected_splits)) != len(self.selected_splits)
                or self.split_assignment_sha256 is None
                or self.materialization_manifest_file_sha256 is None
                or self.core_input_files is None
                or (
                    (
                        self.role_selection_file_sha256,
                        self.role_selection_sha256,
                        self.feedback_provider,
                        self.feedback_model,
                        self.final_provider,
                        self.final_model,
                    )
                    not in {
                        (
                            PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256,
                            PORTFOLIO_CORE_ROLE_SELECTION_SHA256,
                            "kimi",
                            "kimi-k2.6",
                            "kimi",
                            "kimi-k2.6",
                        ),
                        (
                            _PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256,
                            _PORTFOLIO_ROLE_SELECTION_V6_SHA256,
                            "kimi",
                            "kimi-k2.6",
                            "gemini",
                            "gemini-3.6-flash",
                        ),
                        (
                            PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
                            PORTFOLIO_ROLE_SELECTION_SHA256,
                            "kimi",
                            "kimi-k2.6",
                            "gemini",
                            "gemini-3.6-flash",
                        ),
                    }
                )
                or (static_opt and not historical_core_v4)
                or self.scheduling_policy
                != (
                    "static_opt_pool_atomic_generator_batch"
                    if static_opt
                    else "balanced_cyclic_config_order_by_split_atomic_generator_batch"
                )
            ):
                raise ValueError("Core launch identity or split selection drifted")
            if static_opt:
                artifact_ids = tuple(item.artifact_id for item in self.artifacts)
                if (
                    self.execution_mode != "static_opt_rollout"
                    or self.selected_splits != ("opt_pool",)
                    or self.query_count != 800
                    or self.config_count != 1
                    or self.instance_count != 800
                    or self.shard_count != 32
                    or self.rate.final_judge_concurrency != 0
                    or self.calls.final_judge_instance_count != 0
                    or self.calls.final_judge_call_count != 0
                    or (self.calls.final_judge_empty_response_retry_call_ceiling or 0)
                    != 0
                    or artifact_ids
                    != (
                        "assistant_runtime_lock",
                        "static_contract_refresh_receipt",
                        "bank.llm_static",
                    )
                ):
                    raise ValueError("Static opt rollout identity or geometry drifted")
            elif (
                self.execution_mode != "full_matrix"
                or self.config_count != 5
                or self.rate.final_judge_concurrency != 1
            ):
                raise ValueError("Core x5 launch execution mode drifted")
            if not self.remote_runtime_binding_sha256s and (
                "permissions.core_remote_processing_missing" not in self.blockers
            ):
                raise ValueError(
                    "Core launch without remote runtimes must remain blocked"
                )
            assert self.core_input_files is not None
            core_files = self.core_input_files
            if (
                core_files.expected_plan_sha256 != self.portfolio_plan_sha256
                or core_files.expected_pre_generation_manifest_file_sha256
                != self.portfolio_plan_manifest_file_sha256
                or core_files.expected_split_assignment_sha256
                != self.split_assignment_sha256
                or core_files.expected_query_artifact_sha256
                != self.query_artifact_sha256
                or core_files.expected_materialization_manifest_file_sha256
                != self.materialization_manifest_file_sha256
                or core_files.expected_capability_assignments_sha256
                != self.capability_assignments_sha256
                or core_files.expected_base_catalog_sha256 != self.base_catalog_sha256
                or (
                    core_files.expected_runtime_catalog_sha256
                    or core_files.expected_base_catalog_sha256
                )
                != self.runtime_catalog_sha256
            ):
                raise ValueError("Core input-files binding differs from plan hashes")
            permission_files = core_files.remote_files
            if bool(permission_files) != bool(self.remote_runtime_binding_sha256s):
                raise ValueError(
                    "Core permission files differ from remote-runtime bindings"
                )
            if permission_files is None and any(
                value is not None
                for value in (
                    self.authorization_file_sha256,
                    self.receipt_file_sha256,
                    self.receipt_sha256,
                )
            ):
                raise ValueError(
                    "Core launch has permission hashes without permission files"
                )
            if permission_files is not None and (
                permission_files.expected_authorization_file_sha256
                != self.authorization_file_sha256
                or permission_files.expected_receipt_file_sha256
                != self.receipt_file_sha256
                or permission_files.expected_receipt_sha256 != self.receipt_sha256
                or permission_files.processor_order != self.active_processor_order
            ):
                raise ValueError(
                    "Core permission file hashes differ from launch identity"
                )
        elif (
            self.policy_version
            not in {
                "portfolio-dev-mini-200x5-launch-v1",
                "portfolio-dev-mini-200x5-launch-v2",
                "portfolio-dev-mini-200x5-launch-v3",
            }
            or self.dataset_profile is not None
            or self.source_query_count is not None
            or self.selected_splits
            or self.split_assignment_sha256 is not None
            or self.materialization_manifest_file_sha256 is not None
            or self.core_input_files is not None
            or (
                (
                    self.role_selection_file_sha256,
                    self.role_selection_sha256,
                    self.feedback_provider,
                    self.feedback_model,
                    self.final_provider,
                    self.final_model,
                )
                not in {
                    (
                        _PORTFOLIO_ROLE_SELECTION_V5_FILE_SHA256,
                        _PORTFOLIO_ROLE_SELECTION_V5_SHA256,
                        "gemini",
                        "gemini-3.6-flash",
                        "kimi",
                        "kimi-k2.6",
                    ),
                    (
                        _PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256,
                        _PORTFOLIO_ROLE_SELECTION_V6_SHA256,
                        "kimi",
                        "kimi-k2.6",
                        "gemini",
                        "gemini-3.6-flash",
                    ),
                    (
                        PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
                        PORTFOLIO_ROLE_SELECTION_SHA256,
                        "kimi",
                        "kimi-k2.6",
                        "gemini",
                        "gemini-3.6-flash",
                    ),
                }
            )
            or self.query_count != EXPECTED_QUERY_COUNT
            or self.instance_count != EXPECTED_INSTANCE_COUNT
            or self.shard_count != EXPECTED_BATCH_COUNT * EXPECTED_CONFIG_COUNT
            or self.scheduling_policy
            != "balanced_cyclic_config_order_by_accepted_batch"
        ):
            raise ValueError("historical dev_mini launch geometry drifted")
        elif (
            self.execution_mode != "full_matrix"
            or self.config_count != 5
            or self.rate.final_judge_concurrency != 1
        ):
            raise ValueError("historical dev_mini execution mode drifted")
        if (
            self.query_count % EXPECTED_BATCH_SIZE
            or self.instance_count != self.query_count * self.config_count
            or self.shard_count
            != (self.query_count // EXPECTED_BATCH_SIZE) * self.config_count
            or self.calls.assistant_instance_count != self.instance_count
            or self.calls.final_judge_instance_count
            != (0 if static_opt else self.instance_count)
        ):
            raise ValueError("launch query/instance/shard geometry is inconsistent")
        if len(self.shards) != self.shard_count:
            raise ValueError("launch plan shard count differs from its geometry")
        if tuple(item.shard_ordinal for item in self.shards) != tuple(
            range(self.shard_count)
        ):
            raise ValueError("launch shard ordinals are not contiguous")
        all_pairs = {(item.accepted_batch_id, item.config) for item in self.shards}
        if len(all_pairs) != self.shard_count:
            raise ValueError("launch shards do not cover every batch/config pair")
        if self.execution_ready != (not self.blockers):
            raise ValueError("execution_ready differs from blockers")
        expected_status = (
            "prepared_ready" if self.execution_ready else "prepared_blocked"
        )
        if self.status != expected_status:
            raise ValueError("launch status differs from readiness")
        if (
            info.context is not _PLAN_NORMALIZATION_MARKER
            and self.launch_plan_sha256 != _self_hash(self, "launch_plan_sha256")
        ):
            raise ValueError("launch plan self hash mismatch")
        return self


class PortfolioLaunchState(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-launch-state"] = "portfolio-launch-state"
    matrix_run_id: str
    launch_plan_sha256: Sha256
    status: Literal["not_started", "running", "completed", "failed"]
    completed_shard_ids: tuple[str, ...]
    failed_shard_ids: tuple[str, ...]
    model_calls_performed: int = Field(ge=0)
    dashscope_observed_cost_cny: float | None = Field(default=None, ge=0)
    aifast_observed_cost_cny: float | None = Field(default=None, ge=0)

    @field_validator("completed_shard_ids", "failed_shard_ids", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_state(self) -> "PortfolioLaunchState":
        if (
            len(self.completed_shard_ids) != len(set(self.completed_shard_ids))
            or len(self.failed_shard_ids) != len(set(self.failed_shard_ids))
            or set(self.completed_shard_ids) & set(self.failed_shard_ids)
        ):
            raise ValueError("launch state shard sets must be unique and disjoint")
        if self.status == "not_started" and (
            self.completed_shard_ids
            or self.failed_shard_ids
            or self.model_calls_performed
            or self.dashscope_observed_cost_cny is not None
            or self.aifast_observed_cost_cny is not None
        ):
            raise ValueError("not-started launch state must be empty")
        return self


@dataclass(frozen=True)
class PortfolioLaunchArtifactInputs:
    assistant_runtime_lock_path: Path | None = None
    assistant_runtime_lock_file_sha256: str | None = None
    treatment_chain_manifest_path: Path | None = None
    treatment_chain_manifest_file_sha256: str | None = None
    static_contract_refresh_receipt_path: Path | None = None
    static_contract_refresh_receipt_file_sha256: str | None = None
    bank_paths: dict[AssistantRunConfig, Path] | None = None
    bank_file_sha256s: dict[AssistantRunConfig, str] | None = None


@dataclass(frozen=True)
class CreatedPortfolioLaunchPackage:
    root: Path
    plan_path: Path
    instances_path: Path
    state_path: Path
    plan: PortfolioLaunchPlan
    plan_file_sha256: str


@dataclass(frozen=True)
class LoadedPortfolioLaunchPackage:
    root: Path
    plan: PortfolioLaunchPlan
    instances: tuple[PortfolioLaunchInstance, ...]
    state: PortfolioLaunchState
    plan_file_sha256: str


def _read_canonical_object(path: Path, *, label: str) -> tuple[bytes, dict]:
    content = read_stable_regular_file(path, label=label, max_bytes=32 * 1024 * 1024)
    try:
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise PortfolioLaunchError(f"{label} is not canonical JSON") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioLaunchError(f"{label} must be a canonical JSON object")
    return content, raw


def _verify_role_selection(
    path: Path,
    *,
    expected_file_sha256: str,
    core: bool = False,
) -> tuple[str, str]:
    content, raw = _read_canonical_object(path, label="Portfolio model role selection")
    file_sha256 = sha256_bytes(content)
    if file_sha256 != expected_file_sha256:
        raise PortfolioLaunchError("model role selection file digest mismatch")
    selection_sha256 = raw.get("selection_sha256")
    unsigned = dict(raw)
    unsigned.pop("selection_sha256", None)
    historical_v6 = (
        not core
        and expected_file_sha256 == _PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256
    )
    expected_selection_sha256 = (
        PORTFOLIO_CORE_ROLE_SELECTION_SHA256
        if core
        else (
            _PORTFOLIO_ROLE_SELECTION_V6_SHA256
            if historical_v6
            else PORTFOLIO_ROLE_SELECTION_SHA256
        )
    )
    expected_bound_file_sha256 = (
        PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256
        if core
        else (
            _PORTFOLIO_ROLE_SELECTION_V6_FILE_SHA256
            if historical_v6
            else PORTFOLIO_ROLE_SELECTION_FILE_SHA256
        )
    )
    expected_schema_version = 4 if core else (6 if historical_v6 else 7)
    expected_feedback_provider = "kimi"
    expected_feedback_model = "kimi-k2.6"
    expected_final_provider = "kimi" if core else "gemini"
    expected_final_model = "kimi-k2.6" if core else config.PORTFOLIO_JUDGE_MODEL
    feedback = raw.get("feedback_evaluator", {})
    expected_feedback_policy = (
        None
        if core
        else (
            (
                "feedback-evaluator-v7",
                "visual-feedback-exact-shape-prompt-v4",
                "visual-feedback-kimi-dashscope-plain-json-v2",
            )
            if historical_v6
            else (
                "feedback-evaluator-v8",
                "visual-feedback-response-schema-v1-prompt-v5",
                "visual-feedback-kimi-dashscope-plain-json-v3",
            )
        )
    )
    if (
        selection_sha256 != sha256_bytes(canonical_json_bytes(unsigned))
        or selection_sha256 != expected_selection_sha256
        or file_sha256 != expected_bound_file_sha256
        or raw.get("schema_version") != expected_schema_version
        or raw.get("assistant", {}).get("model") != PORTFOLIO_QWEN_MODEL
        or feedback.get("provider") != expected_feedback_provider
        or feedback.get("model") != expected_feedback_model
        or (
            expected_feedback_policy is not None
            and (
                feedback.get("cache_namespace"),
                feedback.get("prompt_policy_version"),
                feedback.get("transport_policy_version"),
            )
            != expected_feedback_policy
        )
        or raw.get("offline_judge", {}).get("model") != expected_final_model
        or raw.get("offline_judge", {}).get("provider") != expected_final_provider
    ):
        raise PortfolioLaunchError("model role selection content drifted")
    return file_sha256, selection_sha256


def _artifact_binding(
    *,
    artifact_id: str,
    path: Path | None,
    expected_file_sha256: str | None,
    kind: Literal[
        "runtime",
        "static_runtime",
        "bank",
        "treatment_chain",
        "receipt",
    ],
) -> tuple[
    PortfolioArtifactBinding,
    StaticBankArtifact | PortfolioTreatmentChainManifest | dict | None,
]:
    if path is None or expected_file_sha256 is None:
        return (
            PortfolioArtifactBinding(
                artifact_id=artifact_id,
                path=None if path is None else path.as_posix(),
                file_sha256=None,
                content_sha256=None,
                status="missing",
                detail="path and external SHA-256 are required",
            ),
            None,
        )
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
            raise PortfolioLaunchError("external SHA-256 is invalid")
        content, raw = _read_canonical_object(path, label=artifact_id)
        if sha256_bytes(content) != expected_file_sha256:
            raise PortfolioLaunchError("external file digest mismatch")
        if kind == "bank":
            parsed: StaticBankArtifact | dict = StaticBankArtifact.model_validate(
                raw, strict=True
            )
            content_sha256 = parsed.bank_sha256
        elif kind == "treatment_chain":
            parsed = PortfolioTreatmentChainManifest.model_validate(raw, strict=True)
            content_sha256 = parsed.chain_sha256
        elif kind == "runtime":
            evaluator_root = Path(__file__).resolve().parent
            if (
                raw.get("tool_registry_sha256") is None
                or raw.get("tool_registry_runtime_sha256") is None
                or raw.get("evaluator_outputs_file_sha256")
                != sha256_bytes((evaluator_root / "evaluator_outputs.py").read_bytes())
                or raw.get("final_runtime_file_sha256")
                != sha256_bytes((evaluator_root / "final_runtime.py").read_bytes())
                or raw.get("final_judge_parser_policy_version")
                != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
                or raw.get("final_judge_parser_policy_sha256")
                != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
            ):
                raise PortfolioLaunchError(
                    "runtime lock lacks active registry or final-Judge identities"
                )
            content_sha256 = raw.get("runtime_lock_sha256")
            if content_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", content_sha256
            ):
                raise PortfolioLaunchError("runtime lock content digest is invalid")
            parsed = raw
        else:
            hash_field = (
                "runtime_lock_sha256" if kind == "static_runtime" else "receipt_sha256"
            )
            content_sha256 = raw.get(hash_field)
            unsigned = dict(raw)
            unsigned.pop(hash_field, None)
            if (
                not isinstance(content_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", content_sha256)
                or content_sha256 != sha256_bytes(canonical_json_bytes(unsigned))
            ):
                raise PortfolioLaunchError(f"{kind} content digest is invalid")
            parsed = raw
        return (
            PortfolioArtifactBinding(
                artifact_id=artifact_id,
                path=path.as_posix(),
                file_sha256=expected_file_sha256,
                content_sha256=content_sha256,
                status="verified",
                detail="canonical bytes match the external digest",
            ),
            parsed,
        )
    except (OSError, PortfolioLaunchError, ValidationError, ValueError) as error:
        return (
            PortfolioArtifactBinding(
                artifact_id=artifact_id,
                path=path.as_posix(),
                file_sha256=None,
                content_sha256=None,
                status="invalid",
                detail=str(error),
            ),
            None,
        )


def _environment_checks(
    *, core: bool = False, require_gemini: bool = False
) -> tuple[list[PortfolioPreflightCheck], list[str]]:
    checks: list[PortfolioPreflightCheck] = []
    blockers: list[str] = []
    variables = (
        (
            "DASHSCOPE_API_KEY",
            "GEMINI_API_KEY",
        )
        if require_gemini
        else ("DASHSCOPE_API_KEY",)
    )
    for variable in variables:
        present = bool(os.environ.get(variable, "").strip())
        checks.append(
            PortfolioPreflightCheck(
                check_id=f"env.{variable}",
                status="passed" if present else "blocked",
                detail="configured" if present else "not configured",
            )
        )
        if not present:
            blockers.append(f"missing_env:{variable}")
    endpoints = (
        (
            ("endpoint.dashscope", config.DASHSCOPE_BASE_URL),
            ("endpoint.kimi_dashscope", config.KIMI_DASHSCOPE_BASE_URL),
        )
        if core and not require_gemini
        else (
            ("endpoint.dashscope", config.DASHSCOPE_BASE_URL),
            ("endpoint.kimi_dashscope", config.KIMI_DASHSCOPE_BASE_URL),
            ("endpoint.aifast", config.AIFAST_BASE_URL),
        )
    )
    for check_id, endpoint in endpoints:
        parsed = urlparse(endpoint)
        valid = parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username
        checks.append(
            PortfolioPreflightCheck(
                check_id=check_id,
                status="passed" if valid else "blocked",
                detail=endpoint if valid else "endpoint must be credential-free HTTPS",
            )
        )
        if not valid:
            blockers.append(f"invalid_{check_id.replace('.', '_')}")
    return checks, blockers


def _balanced_config_order(batch_ordinal: int) -> tuple[AssistantRunConfig, ...]:
    offset = batch_ordinal % len(MAIN_CONFIG_ORDER)
    return MAIN_CONFIG_ORDER[offset:] + MAIN_CONFIG_ORDER[:offset]


def _build_instances_and_shards(
    inputs: VerifiedPortfolioDevMiniInputs | VerifiedPortfolioCoreInputs,
    *,
    matrix_run_id: str,
    selected_splits: tuple[CoreSplit, ...] | None = None,
    config_order: tuple[AssistantRunConfig, ...] = MAIN_CONFIG_ORDER,
) -> tuple[tuple[PortfolioLaunchInstance, ...], tuple[PortfolioLaunchShard, ...]]:
    core = type(inputs) is VerifiedPortfolioCoreInputs
    if core:
        if not selected_splits:
            raise PortfolioLaunchError(
                "Core launch requires at least one selected split"
            )
        if tuple(
            split for split in CORE_SPLIT_ORDER if split in selected_splits
        ) != selected_splits or len(set(selected_splits)) != len(selected_splits):
            raise PortfolioLaunchError(
                "Core split selection is duplicated or non-canonical"
            )
        selected_batches = tuple(
            batch for batch in inputs.batches if batch.split in selected_splits
        )
        batch_order = tuple(batch.batch_id for batch in selected_batches)
        grouped = {batch.batch_id: list(batch.query_ids) for batch in selected_batches}
        selected_query_ids = {
            query_id for batch in selected_batches for query_id in batch.query_ids
        }
        selected_queries = tuple(
            query for query in inputs.queries if query.query_id in selected_query_ids
        )
    else:
        if selected_splits not in (None, ("dev_mini",)):
            raise PortfolioLaunchError("dev_mini launch cannot select Core splits")
        grouped_dict: dict[str, list[str]] = defaultdict(list)
        for query in inputs.queries:
            if query.synthesis_batch_id is None:
                raise PortfolioLaunchError(
                    "dev_mini query lacks accepted batch identity"
                )
            grouped_dict[query.synthesis_batch_id].append(query.query_id)
        grouped = dict(grouped_dict)
        batch_order = tuple(entry.batch_id for entry in inputs.ledger)
        if set(grouped) != set(batch_order):
            raise PortfolioLaunchError("accepted batch membership drifted")
        selected_queries = inputs.queries
    if any(len(grouped[batch]) != EXPECTED_BATCH_SIZE for batch in batch_order):
        raise PortfolioLaunchError("launch batches must each contain 25 queries")

    query_by_id = {query.query_id: query for query in selected_queries}
    assistant_by_id = {query.query_id: query for query in inputs.assistant_queries}
    asset_by_id = {item.query_id: item for item in inputs.query_assets}
    query_ordinals = {
        query.query_id: ordinal for ordinal, query in enumerate(selected_queries)
    }

    instances: list[PortfolioLaunchInstance] = []
    shards: list[PortfolioLaunchShard] = []
    if config_order not in (MAIN_CONFIG_ORDER, ("llm_static",)):
        raise PortfolioLaunchError("launch config selection is unsupported")
    for batch_ordinal, batch_id in enumerate(batch_order):
        query_ids = tuple(grouped[batch_id])
        ordered_configs = (
            _balanced_config_order(batch_ordinal)
            if config_order == MAIN_CONFIG_ORDER
            else config_order
        )
        for config_name in ordered_configs:
            config_ordinal = MAIN_CONFIG_ORDER.index(config_name)
            shard_ordinal = len(shards)
            ordinal_width = 3 if core else 2
            shard_id = (
                f"{shard_ordinal:0{ordinal_width}d}-{batch_id}-"
                f"{config_ordinal:02d}-{config_name}"
            )
            shard_instances: list[PortfolioLaunchInstance] = []
            for query_id in query_ids:
                query = query_by_id[query_id]
                assistant_query = assistant_by_id[query_id]
                asset = asset_by_id[query_id]
                payload: dict[str, object] = {
                    "schema_version": 1,
                    "kind": "portfolio-launch-instance",
                    "matrix_run_id": matrix_run_id,
                    "instance_ordinal": len(instances),
                    "shard_id": shard_id,
                    "shard_ordinal": shard_ordinal,
                    "query_ordinal": query_ordinals[query_id],
                    "config_ordinal": config_ordinal,
                    "config": config_name,
                    "accepted_batch_id": batch_id,
                    "query_id": query_id,
                    "query_sha256": assistant_query.query_sha256,
                    "public_input_sha256": assistant_query.public_input_sha256,
                    "image_sha256": asset.image_sha256,
                    "assistant_output_relpath": (
                        f"shards/{shard_id}/assistant/{query_id}.json"
                    ),
                    "final_output_relpath": (
                        f"shards/{shard_id}/final/{query_id}.json"
                    ),
                }
                if core:
                    if assistant_query.asset_binding is None:
                        raise PortfolioLaunchError(
                            "Core Assistant projection lacks its private asset binding"
                        )
                    payload.update(
                        {
                            "asset_token": assistant_query.asset_binding.asset_token,
                            "asset_binding_sha256": (
                                assistant_query.asset_binding.binding_sha256
                            ),
                        }
                    )
                else:
                    payload.update(
                        {"asset_id": query.asset_id, "image_path": query.image_path}
                    )
                instance = PortfolioLaunchInstance.model_validate(
                    {
                        **payload,
                        "instance_sha256": _hash_payload(payload),
                    },
                    strict=True,
                )
                instances.append(instance)
                shard_instances.append(instance)
            shard_payload = {
                "shard_id": shard_id,
                "shard_ordinal": shard_ordinal,
                "accepted_batch_id": batch_id,
                "config": config_name,
                "config_ordinal": config_ordinal,
                "query_count": 25,
                "query_ids": tuple(item.query_id for item in shard_instances),
                "instance_sha256s": tuple(
                    item.instance_sha256 for item in shard_instances
                ),
                "output_relpath": f"shards/{shard_id}",
            }
            shards.append(
                PortfolioLaunchShard.model_validate(
                    {
                        **shard_payload,
                        "shard_sha256": _hash_payload(shard_payload),
                    },
                    strict=True,
                )
            )
    expected_instances = len(selected_queries) * len(config_order)
    expected_shards = len(batch_order) * len(config_order)
    if len(instances) != expected_instances or len(shards) != expected_shards:
        raise PortfolioLaunchError(
            "launch coverage differs from selected query x config selection"
        )
    if len({(item.query_id, item.config) for item in instances}) != expected_instances:
        raise PortfolioLaunchError("launch instances are not unique")
    return tuple(instances), tuple(shards)


def _instances_bytes(instances: tuple[PortfolioLaunchInstance, ...]) -> bytes:
    return b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) for item in instances
    )


def _remote_runtime_binding_sha256s(remote_runtimes: tuple) -> tuple[str, ...]:
    return tuple(
        _hash_payload(
            {
                "processor": item.processor,
                "authorization_id": item.authorization.authorization_id,
                "authorization_file_sha256": item.authorization_file_sha256,
                "receipt_file_sha256": item.receipt_file_sha256,
                "receipt_sha256": item.receipt.receipt_sha256,
                "selection_manifest_sha256": (
                    item.receipt.base_selection_manifest_sha256
                ),
                "dataset_assets_sha256": item.dataset_assets_sha256,
                "plan_sha256": item.plan_sha256,
                "query_artifact_sha256": item.query_artifact_sha256,
                "base_catalog_sha256": item.receipt.base_catalog_sha256,
                "output_catalog_sha256": item.catalog.catalog_sha256,
            }
        )
        for item in remote_runtimes
    )


def _build_core_input_files_binding(
    inputs: VerifiedPortfolioCoreInputs,
) -> PortfolioCoreInputFilesBinding:
    files = inputs.files
    remote_files = files.remote_files
    permission_binding: dict[str, object] | None = None
    if remote_files is not None:
        runtimes = inputs.remote_runtimes
        if tuple(
            item.processor for item in runtimes
        ) != remote_files.processor_order or len(runtimes) != len(
            remote_files.processor_order
        ):
            raise PortfolioLaunchError(
                "Core permission files lack three verified processor runtimes"
            )
        first = runtimes[0]
        if (
            remote_files.expected_authorization_file_sha256
            != first.authorization_file_sha256
            or remote_files.expected_receipt_file_sha256 != first.receipt_file_sha256
            or any(
                item.authorization_file_sha256 != first.authorization_file_sha256
                or item.receipt_file_sha256 != first.receipt_file_sha256
                or item.receipt.receipt_sha256 != first.receipt.receipt_sha256
                or item.receipt.base_selection_manifest_sha256
                != first.receipt.base_selection_manifest_sha256
                or item.dataset_assets_sha256 != first.dataset_assets_sha256
                or item.receipt.base_catalog_sha256 != first.receipt.base_catalog_sha256
                or item.catalog.catalog_sha256 != first.catalog.catalog_sha256
                for item in runtimes[1:]
            )
        ):
            raise PortfolioLaunchError(
                "Core processor runtimes do not share one input/permission chain"
            )
        permission_binding = {
            "authorization_file": _bound_absolute_path(remote_files.authorization_file),
            "expected_authorization_file_sha256": (
                remote_files.expected_authorization_file_sha256
            ),
            "receipt_file": _bound_absolute_path(remote_files.receipt_file),
            "expected_receipt_file_sha256": (remote_files.expected_receipt_file_sha256),
            "expected_receipt_sha256": first.receipt.receipt_sha256,
            "selection_manifest": _bound_absolute_path(remote_files.selection_manifest),
            "expected_selection_manifest_sha256": (
                first.receipt.base_selection_manifest_sha256
            ),
            "dataset_assets": _bound_absolute_path(remote_files.dataset_assets),
            "expected_dataset_assets_sha256": first.dataset_assets_sha256,
            "base_catalog_dir": _bound_absolute_path(remote_files.base_catalog_dir),
            "expected_base_catalog_sha256": first.receipt.base_catalog_sha256,
            "output_catalog_dir": _bound_absolute_path(remote_files.output_catalog_dir),
            "expected_output_catalog_sha256": first.catalog.catalog_sha256,
            "asset_root": _bound_absolute_path(remote_files.asset_root),
            "processor_order": remote_files.processor_order,
        }
    payload = {
        "plan_path": _bound_absolute_path(files.plan_path),
        "expected_plan_sha256": files.expected_plan_sha256,
        "pre_generation_manifest_path": _bound_absolute_path(
            files.pre_generation_manifest_path
        ),
        "expected_pre_generation_manifest_file_sha256": (
            files.expected_pre_generation_manifest_file_sha256
        ),
        "split_assignment_path": _bound_absolute_path(files.split_assignment_path),
        "expected_split_assignment_sha256": (files.expected_split_assignment_sha256),
        "query_artifact_path": _bound_absolute_path(files.query_artifact_path),
        "expected_query_artifact_sha256": files.expected_query_artifact_sha256,
        "materialization_manifest_path": _bound_absolute_path(
            files.materialization_manifest_path
        ),
        "expected_materialization_manifest_file_sha256": (
            files.expected_materialization_manifest_file_sha256
        ),
        "capability_assignments_path": _bound_absolute_path(
            files.capability_assignments_path
        ),
        "expected_capability_assignments_sha256": (
            files.expected_capability_assignments_sha256
        ),
        "base_catalog_dir": _bound_absolute_path(files.base_catalog_dir),
        "expected_base_catalog_sha256": files.expected_base_catalog_sha256,
        "asset_root": _bound_absolute_path(files.asset_root),
        "runtime_catalog_dir": (
            None
            if files.runtime_catalog_dir is None
            else _bound_absolute_path(files.runtime_catalog_dir)
        ),
        "expected_runtime_catalog_sha256": (files.expected_runtime_catalog_sha256),
        "remote_files": permission_binding,
    }
    try:
        return PortfolioCoreInputFilesBinding.model_validate(payload, strict=True)
    except ValidationError as error:
        raise PortfolioLaunchError("Core input-files binding is invalid") from error


def build_portfolio_launch_plan(
    inputs: VerifiedPortfolioDevMiniInputs | VerifiedPortfolioCoreInputs,
    *,
    matrix_run_id: str,
    role_selection_path: str | Path,
    expected_role_selection_file_sha256: str,
    artifacts: PortfolioLaunchArtifactInputs | None = None,
    operator_approved_dashscope_budget_cny: float = (AUTONOMOUS_DASHSCOPE_BUDGET_CNY),
    planning_ceiling_cny: float = DEFAULT_PLANNING_CEILING_CNY,
    available_disk_bytes: int | None = None,
    selected_splits: tuple[CoreSplit, ...] | None = None,
    execution_mode: Literal["full_matrix", "static_opt_rollout"] = "full_matrix",
) -> tuple[PortfolioLaunchPlan, bytes]:
    """Build a complete zero-call plan and return its canonical JSONL workload."""

    core = type(inputs) is VerifiedPortfolioCoreInputs
    static_opt = execution_mode == "static_opt_rollout"
    if core:
        verified = require_verified_portfolio_core_inputs(inputs)
        effective_splits = selected_splits or CORE_SPLIT_ORDER
    else:
        verified = require_verified_portfolio_dev_mini_inputs(inputs)
        if selected_splits not in (None, ("dev_mini",)):
            raise PortfolioLaunchError("dev_mini launch cannot select Core splits")
        effective_splits = None
    if static_opt and (not core or effective_splits != ("opt_pool",)):
        raise PortfolioLaunchError(
            "static_opt_rollout requires Core selected_splits=('opt_pool',)"
        )
    if not _RUN_ID.fullmatch(matrix_run_id):
        raise PortfolioLaunchError("matrix_run_id contains unsafe characters")
    if planning_ceiling_cny <= 0:
        raise PortfolioLaunchError("planning ceiling must be positive")
    runtime_processors = tuple(item.processor for item in verified.remote_runtimes)
    historical_core_role = (
        core
        and expected_role_selection_file_sha256
        == PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256
    )
    if static_opt and not historical_core_role:
        raise PortfolioLaunchError(
            "static_opt_rollout remains bound to the historical Core v4 role identity"
        )
    role_file_sha256, role_sha256 = _verify_role_selection(
        Path(role_selection_path),
        expected_file_sha256=expected_role_selection_file_sha256,
        core=historical_core_role,
    )
    role_swapped = role_sha256 in {
        _PORTFOLIO_ROLE_SELECTION_V6_SHA256,
        PORTFOLIO_ROLE_SELECTION_SHA256,
    }
    valid_runtime_orders = (
        {
            (
                ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
                if role_swapped
                else CORE_PORTFOLIO_PROCESSOR_ORDER
            ),
            (),
        }
        if core
        else {ACTIVE_PORTFOLIO_PROCESSOR_ORDER}
    )
    if runtime_processors not in valid_runtime_orders:
        raise PortfolioLaunchError(
            "inputs are not bound to the selected model-role processors"
        )
    instances, shards = _build_instances_and_shards(
        verified,
        matrix_run_id=matrix_run_id,
        selected_splits=effective_splits,
        config_order=("llm_static",) if static_opt else MAIN_CONFIG_ORDER,
    )
    instance_bytes = _instances_bytes(instances)
    instances_file_sha256 = sha256_bytes(instance_bytes)

    artifact_inputs = artifacts or PortfolioLaunchArtifactInputs()
    bindings: list[PortfolioArtifactBinding] = []
    parsed_runtime: dict | None = None
    runtime_binding, runtime_value = _artifact_binding(
        artifact_id="assistant_runtime_lock",
        path=artifact_inputs.assistant_runtime_lock_path,
        expected_file_sha256=(artifact_inputs.assistant_runtime_lock_file_sha256),
        kind="static_runtime" if static_opt else "runtime",
    )
    bindings.append(runtime_binding)
    if isinstance(runtime_value, dict):
        parsed_runtime = runtime_value

    parsed_treatment_chain: PortfolioTreatmentChainManifest | None = None
    treatment_binding: PortfolioArtifactBinding | None = None
    refresh_binding: PortfolioArtifactBinding | None = None
    parsed_refresh_receipt: dict | None = None
    if static_opt:
        refresh_binding, refresh_value = _artifact_binding(
            artifact_id="static_contract_refresh_receipt",
            path=artifact_inputs.static_contract_refresh_receipt_path,
            expected_file_sha256=(
                artifact_inputs.static_contract_refresh_receipt_file_sha256
            ),
            kind="receipt",
        )
        bindings.append(refresh_binding)
        if isinstance(refresh_value, dict):
            parsed_refresh_receipt = refresh_value
    else:
        treatment_binding, treatment_value = _artifact_binding(
            artifact_id="treatment_chain_manifest",
            path=artifact_inputs.treatment_chain_manifest_path,
            expected_file_sha256=(artifact_inputs.treatment_chain_manifest_file_sha256),
            kind="treatment_chain",
        )
        bindings.append(treatment_binding)
        if isinstance(treatment_value, PortfolioTreatmentChainManifest):
            parsed_treatment_chain = treatment_value

    parsed_banks: dict[AssistantRunConfig, StaticBankArtifact] = {}
    bank_paths = artifact_inputs.bank_paths or {}
    bank_hashes = artifact_inputs.bank_file_sha256s or {}
    for config_name in ("llm_static",) if static_opt else _BANK_CONFIGS:
        binding, parsed = _artifact_binding(
            artifact_id=f"bank.{config_name}",
            path=bank_paths.get(config_name),
            expected_file_sha256=bank_hashes.get(config_name),
            kind="bank",
        )
        bindings.append(binding)
        if isinstance(parsed, StaticBankArtifact):
            parsed_banks[config_name] = parsed

    compatibility_projection = None
    compatibility_projection_error: str | None = None
    compatibility_declared = (
        not static_opt
        and parsed_runtime is not None
        and parsed_runtime.get("runtime_compatibility_rebind_file") is not None
    )
    if compatibility_declared:
        try:
            from skillchain.evaluation.portfolio_treatment_io import (
                load_verified_portfolio_treatment_runtime,
            )

            runtime_lock_path = artifact_inputs.assistant_runtime_lock_path
            runtime_lock_file_sha256 = (
                artifact_inputs.assistant_runtime_lock_file_sha256
            )
            treatment_path = artifact_inputs.treatment_chain_manifest_path
            if (
                runtime_lock_path is None
                or runtime_lock_file_sha256 is None
                or treatment_path is None
            ):
                raise PortfolioLaunchError(
                    "compatibility rebind lacks its runtime-root artifacts"
                )
            runtime_root = runtime_lock_path.resolve(strict=True).parent
            if treatment_path.resolve(strict=True) != (
                runtime_root / "treatment-chain-manifest.json"
            ).resolve(strict=True):
                raise PortfolioLaunchError(
                    "compatibility rebind treatment manifest is outside its runtime"
                )
            compatibility_projection = load_verified_portfolio_treatment_runtime(
                runtime_root,
                expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
            )
            if (
                dict(compatibility_projection.runtime_lock) != parsed_runtime
                or compatibility_projection.chain.manifest != parsed_treatment_chain
                or compatibility_projection.chain.compatibility_rebind is None
            ):
                raise PortfolioLaunchError(
                    "compatibility rebind projection differs from launch artifacts"
                )
            output_bindings = {
                item.config: item
                for item in compatibility_projection.chain.compatibility_rebind.bank_bindings
                if item.role == "output"
            }
            for config_name in _BANK_CONFIGS:
                supplied_path = bank_paths.get(config_name)
                binding = output_bindings.get(config_name)
                if (
                    supplied_path is None
                    or binding is None
                    or supplied_path.resolve(strict=True)
                    != runtime_root.joinpath(
                        *binding.rebound_bank_file.split("/")
                    ).resolve(strict=True)
                ):
                    raise PortfolioLaunchError(
                        f"compatibility rebind Bank path differs: {config_name}"
                    )
        except (OSError, TypeError, ValueError) as error:
            compatibility_projection = None
            compatibility_projection_error = str(error)

    static_runtime_projection = None
    static_runtime_projection_error: str | None = None
    if static_opt and artifact_inputs.assistant_runtime_lock_path is not None:
        try:
            from skillchain.evaluation.portfolio_static_opt_runtime import (
                STATIC_OPT_BANK_FILE,
                STATIC_OPT_REFRESH_RECEIPT_FILE,
                load_verified_portfolio_static_opt_runtime,
            )

            runtime_lock_path = artifact_inputs.assistant_runtime_lock_path
            runtime_lock_file_sha256 = (
                artifact_inputs.assistant_runtime_lock_file_sha256
            )
            static_bank_path = bank_paths.get("llm_static")
            refresh_path = artifact_inputs.static_contract_refresh_receipt_path
            if (
                runtime_lock_file_sha256 is None
                or static_bank_path is None
                or refresh_path is None
            ):
                raise PortfolioLaunchError(
                    "Static opt runtime requires its lock, one Bank, and refresh receipt"
                )
            runtime_root = runtime_lock_path.resolve(strict=True).parent
            if static_bank_path.resolve(strict=True) != (
                runtime_root / STATIC_OPT_BANK_FILE
            ).resolve(strict=True) or refresh_path.resolve(strict=True) != (
                runtime_root / STATIC_OPT_REFRESH_RECEIPT_FILE
            ).resolve(strict=True):
                raise PortfolioLaunchError(
                    "Static opt Bank or refresh receipt is outside its runtime"
                )
            static_runtime_projection = load_verified_portfolio_static_opt_runtime(
                runtime_root,
                expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
            )
            from skillchain.evaluation.portfolio_core_runtime_sources import (
                load_verified_portfolio_core_runtime_sources,
            )

            load_verified_portfolio_core_runtime_sources(
                verified,
                output_dir=runtime_root / "core-runtime-sources",
                expected_receipt_file_sha256=(
                    static_runtime_projection.core_source_receipt_file_sha256
                ),
            )
            if (
                dict(static_runtime_projection.runtime_lock) != parsed_runtime
                or parsed_banks.get("llm_static") != static_runtime_projection.bank
                or dict(static_runtime_projection.refresh_receipt)
                != parsed_refresh_receipt
            ):
                raise PortfolioLaunchError(
                    "Static opt runtime projection differs from launch artifacts"
                )
        except (OSError, TypeError, ValueError) as error:
            static_runtime_projection = None
            static_runtime_projection_error = str(error)

    config_order = ("llm_static",) if static_opt else MAIN_CONFIG_ORDER
    selected_query_count = len(instances) // len(config_order)
    checks: list[PortfolioPreflightCheck] = [
        PortfolioPreflightCheck(
            check_id="inputs.core_r3" if core else "inputs.dev_mini",
            status="passed",
            detail=(
                f"{selected_query_count} split-selected Core r3 queries and "
                "private catalog bytes reverified"
                if core
                else "200 accepted queries and 184 image bytes reverified"
            ),
        ),
        PortfolioPreflightCheck(
            check_id=(
                "permissions.core_remote_processing"
                if core
                else "permissions.remote_processing_v3"
            ),
            status="passed" if verified.remote_runtimes else "blocked",
            detail=(
                "Assistant and schema-versioned evaluator scopes reverified"
                if verified.remote_runtimes
                else (
                    "Core remote-processing overlay and three runtime preflights "
                    "are missing"
                )
            ),
        ),
        PortfolioPreflightCheck(
            check_id=(
                "models.role_selection_v4_core"
                if core and not role_swapped
                else (
                    "models.role_selection_v7"
                    if role_sha256 == PORTFOLIO_ROLE_SELECTION_SHA256
                    else "models.role_selection_v6"
                )
            ),
            status="passed",
            detail=(
                "Core model-role-selection-v4 file and self hashes verified"
                if core and not role_swapped
                else (
                    "model-role-selection-v7 file and self hashes verified"
                    if role_sha256 == PORTFOLIO_ROLE_SELECTION_SHA256
                    else "model-role-selection-v6 file and self hashes verified"
                )
            ),
        ),
    ]
    env_checks, blockers = _environment_checks(
        core=core,
        require_gemini=role_swapped and not static_opt,
    )
    checks.extend(env_checks)
    if core and not verified.remote_runtimes:
        blockers.append("permissions.core_remote_processing_missing")
    for binding in bindings:
        checks.append(
            PortfolioPreflightCheck(
                check_id=f"artifact.{binding.artifact_id}",
                status="passed" if binding.status == "verified" else "blocked",
                detail=binding.detail,
            )
        )
        if binding.status != "verified":
            blockers.append(f"{binding.artifact_id}_{binding.status}")

    if compatibility_declared:
        compatibility_ready = compatibility_projection is not None
        checks.append(
            PortfolioPreflightCheck(
                check_id="binding.runtime_compatibility_rebind",
                status="passed" if compatibility_ready else "blocked",
                detail=(
                    "accepted parent chain and zero-authoring Core Bank projection reverified"
                    if compatibility_ready
                    else (
                        "runtime compatibility rebind is invalid: "
                        + (compatibility_projection_error or "unknown error")
                    )
                ),
            )
        )
        if not compatibility_ready:
            blockers.append("runtime_compatibility_rebind_invalid")

    if static_opt:
        static_runtime_ready = static_runtime_projection is not None
        checks.append(
            PortfolioPreflightCheck(
                check_id="binding.static_opt_runtime",
                status="passed" if static_runtime_ready else "blocked",
                detail=(
                    "one-Bank Static runtime and refresh receipt reverified"
                    if static_runtime_ready
                    else (
                        "Static runtime is invalid: "
                        + (static_runtime_projection_error or "missing artifacts")
                    )
                ),
            )
        )
        if not static_runtime_ready:
            blockers.append("static_opt_runtime_invalid")

    if parsed_runtime is not None:
        runtime_registry = parsed_runtime.get("tool_registry_sha256")
        runtime_identity = parsed_runtime.get("tool_registry_runtime_sha256")
        declared_bank_sha256s = parsed_runtime.get("bank_sha256s")
        for config_name, bank in parsed_banks.items():
            if (
                bank.tool_registry_sha256 != runtime_registry
                or bank.tool_registry_runtime_sha256 != runtime_identity
                or not isinstance(declared_bank_sha256s, dict)
                or declared_bank_sha256s.get(config_name) != bank.bank_sha256
            ):
                blockers.append(f"bank_runtime_mismatch:{config_name}")
                checks.append(
                    PortfolioPreflightCheck(
                        check_id=f"binding.bank_runtime.{config_name}",
                        status="blocked",
                        detail=(
                            "Bank registry or content identity differs from "
                            "runtime lock"
                        ),
                    )
                )
            else:
                checks.append(
                    PortfolioPreflightCheck(
                        check_id=f"binding.bank_runtime.{config_name}",
                        status="passed",
                        detail="Bank and runtime registry identities match",
                    )
                )

        if not static_opt:
            assert treatment_binding is not None
            runtime_chain_ready = (
                parsed_runtime.get("bank_policy")
                == PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
                and parsed_runtime.get("official_matrix_eligible") is True
                and parsed_runtime.get("treatment_chain_status") == "ready_for_matrix"
                and treatment_binding.status == "verified"
                and parsed_runtime.get("treatment_chain_manifest_file_sha256")
                == treatment_binding.file_sha256
                and parsed_runtime.get("treatment_chain_sha256")
                == treatment_binding.content_sha256
                and (not compatibility_declared or compatibility_projection is not None)
            )
            checks.append(
                PortfolioPreflightCheck(
                    check_id="binding.runtime_treatment_chain",
                    status="passed" if runtime_chain_ready else "blocked",
                    detail=(
                        "runtime lock binds a matrix-ready real treatment chain"
                        if runtime_chain_ready
                        else (
                            "runtime lock lacks a matrix-ready real treatment "
                            "chain or still identifies a scaffold"
                        )
                    ),
                )
            )
            if not runtime_chain_ready:
                blockers.append("runtime_treatment_chain_invalid")

        active_runtime_contract = {
            "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
            "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
            "provider_pricing_contract_version": (
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            ),
            "provider_pricing_contract_sha256": (
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
            ),
        }
        if not static_opt:
            active_runtime_contract.update(
                {
                    "final_judge_result_schema_version": (
                        FINAL_JUDGE_RESULT_SCHEMA_VERSION
                    ),
                    "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
                    "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
                    "final_judge_retry_policy_version": (
                        FINAL_JUDGE_RETRY_POLICY_VERSION
                    ),
                    "final_judge_retry_policy_sha256": (
                        FINAL_JUDGE_RETRY_POLICY_SHA256
                    ),
                    "final_judge_thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
                    "final_judge_provider_input_token_reserve": (
                        FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
                    ),
                    "final_judge_provider_output_token_reserve": (
                        FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
                    ),
                    "final_judge_transport_policy_version": (
                        FINAL_JUDGE_TRANSPORT_POLICY_VERSION
                    ),
                    "final_judge_transport_policy_sha256": (
                        FINAL_JUDGE_TRANSPORT_POLICY_SHA256
                    ),
                    "final_judge_requested_response_format": "json_object",
                    "final_judge_provider_pricing_status": (
                        PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS
                    ),
                }
            )
        runtime_execution_ready = all(
            parsed_runtime.get(field) == expected
            for field, expected in active_runtime_contract.items()
        )
        checks.append(
            PortfolioPreflightCheck(
                check_id="binding.runtime_execution_contract",
                status="passed" if runtime_execution_ready else "blocked",
                detail=(
                    (
                        "runtime binds active GCS v2 and hard-budget contracts"
                        if static_opt
                        else _runtime_execution_contract_success_detail()
                    )
                    if runtime_execution_ready
                    else "runtime lacks the active Judge or hard-budget contract"
                ),
            )
        )
        if not runtime_execution_ready:
            blockers.append("runtime_execution_contract_invalid")

    if parsed_treatment_chain is not None:
        record_hashes = {
            item.config: item.receipt_sha256 for item in parsed_treatment_chain.records
        }
        bank_bindings = {item.config: item for item in parsed_treatment_chain.banks}
        supplied_file_hashes = artifact_inputs.bank_file_sha256s or {}
        if compatibility_declared:
            chain_banks_match = (
                compatibility_projection is not None
                and parsed_treatment_chain.status == "ready_for_matrix"
                and all(
                    config_name in parsed_banks
                    and parsed_banks[config_name]
                    == compatibility_projection.chain.output_banks[config_name]
                    and supplied_file_hashes.get(config_name)
                    == sha256_bytes(parsed_banks[config_name].canonical_bytes())
                    for config_name in _BANK_CONFIGS
                )
            )
        else:
            chain_banks_match = (
                parsed_treatment_chain.status == "ready_for_matrix"
                and all(
                    config_name in parsed_banks
                    and config_name in bank_bindings
                    and parsed_banks[config_name].bank_sha256
                    == bank_bindings[config_name].bank_sha256
                    and supplied_file_hashes.get(config_name)
                    == bank_bindings[config_name].bank_file_sha256
                    for config_name in _BANK_CONFIGS
                )
            )
        expected_query_ids = {item.query_id for item in verified.queries}
        declared_query_ids = set(parsed_treatment_chain.optimization_query_ids) | set(
            parsed_treatment_chain.evaluation_query_ids
        )
        chain_split_matches = (
            len(parsed_treatment_chain.optimization_query_ids) == 25
            and len(parsed_treatment_chain.evaluation_query_ids) == 175
            and (
                declared_query_ids <= expected_query_ids
                if core
                else declared_query_ids == expected_query_ids
            )
        )
        runtime_records_match = (
            parsed_runtime is not None
            and parsed_runtime.get("treatment_record_sha256s") == record_hashes
        )
        chain_semantics_ready = (
            chain_banks_match and chain_split_matches and runtime_records_match
        )
        checks.append(
            PortfolioPreflightCheck(
                check_id="binding.treatment_chain_semantics",
                status="passed" if chain_semantics_ready else "blocked",
                detail=(
                    "chain binds all four Banks, stage records, and its "
                    "25/175 development provenance"
                    if chain_semantics_ready
                    else (
                        "chain Bank, stage-record, or 25/175 split binding "
                        "is incomplete"
                    )
                ),
            )
        )
        if not chain_semantics_ready:
            blockers.append("treatment_chain_semantics_invalid")

    disk_bytes = available_disk_bytes
    if disk_bytes is None:
        disk_bytes = shutil.disk_usage(Path.cwd()).free
    disk_ready = disk_bytes >= 2 * 1024 * 1024 * 1024
    checks.append(
        PortfolioPreflightCheck(
            check_id="filesystem.free_space",
            status="passed" if disk_ready else "blocked",
            detail=(
                "at least 2 GiB available"
                if disk_ready
                else "less than 2 GiB available"
            ),
        )
    )
    if not disk_ready:
        blockers.append("insufficient_disk_space")

    planning_ceiling = float(planning_ceiling_cny)
    if operator_approved_dashscope_budget_cny < planning_ceiling:
        blockers.append("dashscope_budget_approval_required")
        checks.append(
            PortfolioPreflightCheck(
                check_id="budget.dashscope",
                status="blocked",
                detail=(
                    f"planning ceiling CNY {planning_ceiling:g} exceeds the "
                    "currently approved "
                    f"CNY {operator_approved_dashscope_budget_cny:g}"
                ),
            )
        )
    else:
        checks.append(
            PortfolioPreflightCheck(
                check_id="budget.dashscope",
                status="passed",
                detail="operator-approved cap covers the conservative planning ceiling",
            )
        )

    if role_swapped and not static_opt:
        blockers.append("aifast_gemini_judge_pricing_and_ceiling_required")
        checks.append(
            PortfolioPreflightCheck(
                check_id="budget.aifast_gemini_judge",
                status="blocked",
                detail=(
                    "AIFast Gemini Judge pricing and provider ceilings are not "
                    "frozen; provider reservation is prohibited"
                ),
            )
        )

    blockers = sorted(set(blockers))
    feedback_warning = (
        "DashScope Kimi Feedback is excluded from the "
        f"{len(instances)}-row selected-matrix call estimate."
    )
    warnings = (
        feedback_warning,
        (
            f"CNY {planning_ceiling:g} is a conservative planning ceiling, "
            "not a provider bill quote."
        ),
        "Formal Research eligibility remains false; this is the Portfolio Track.",
        (
            "The Core plan may include frozen test batches, but this zero-call "
            "schedule does not unseal or authorize their execution."
            if core
            else (
                "The first 25 queries are optimization evidence; headline effect "
                "claims must use the disjoint 175-query evaluation partition."
            )
        ),
    )
    remote = verified.remote_runtimes
    instance_count = len(instances)
    shard_count = len(shards)
    assistant_call_floor = selected_query_count * (2 if static_opt else 8)
    assistant_call_ceiling = selected_query_count * (5 if static_opt else 24)
    final_judge_count = 0 if static_opt else instance_count
    final_judge_retry_ceiling = final_judge_count * (FINAL_JUDGE_MAX_ATTEMPTS - 1)
    plan_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": (
            "portfolio-core-static-opt-800x1-launch-plan"
            if static_opt
            else "portfolio-core-split-x5-launch-plan"
            if core
            else "portfolio-dev-mini-200x5-launch-plan"
        ),
        "policy_version": (
            PORTFOLIO_STATIC_OPT_LAUNCH_POLICY_VERSION
            if static_opt
            else PORTFOLIO_CORE_LAUNCH_POLICY_VERSION
            if core
            else PORTFOLIO_LAUNCH_POLICY_VERSION
        ),
        "execution_mode": execution_mode,
        "track": "portfolio",
        "formal_eligible": False,
        "matrix_run_id": matrix_run_id,
        "status": "prepared_ready" if not blockers else "prepared_blocked",
        "execution_ready": not blockers,
        "execution_authorized": False,
        "model_calls_performed": 0,
        "query_count": selected_query_count,
        "config_count": len(config_order),
        "instance_count": instance_count,
        "shard_count": shard_count,
        "config_order": config_order,
        "portfolio_plan_sha256": verified.expected_plan_sha256,
        "portfolio_plan_manifest_file_sha256": (
            verified.expected_plan_manifest_file_sha256
        ),
        "query_artifact_sha256": verified.expected_query_artifact_sha256,
        "capability_assignments_sha256": (
            verified.expected_capability_assignments_sha256
        ),
        "base_catalog_sha256": verified.expected_base_catalog_sha256,
        "runtime_catalog_sha256": verified.expected_output_catalog_sha256,
        "active_processor_order": (
            ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
            if core and role_swapped
            else CORE_PORTFOLIO_PROCESSOR_ORDER
            if core
            else ACTIVE_PORTFOLIO_PROCESSOR_ORDER
        ),
        "remote_runtime_binding_sha256s": _remote_runtime_binding_sha256s(remote),
        "role_selection_file_sha256": role_file_sha256,
        "role_selection_sha256": role_sha256,
        "assistant_provider": "qwen",
        "assistant_model": PORTFOLIO_QWEN_MODEL,
        "feedback_provider": "kimi",
        # Launch-plan v4/v6/v7 identities remain historical Kimi contracts.
        # Active Qwen Feedback is bound separately by the typed S1 pipeline.
        "feedback_model": config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL,
        "final_provider": "gemini" if role_swapped else "kimi",
        "final_model": config.PORTFOLIO_JUDGE_MODEL if role_swapped else "kimi-k2.6",
        "artifacts": tuple(item.model_dump(mode="python") for item in bindings),
        "preflight_checks": tuple(item.model_dump(mode="python") for item in checks),
        "blockers": tuple(blockers),
        "warnings": warnings,
        "budget": {
            "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
            "policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
            "provider_pricing_contract_version": (
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            ),
            "provider_pricing_contract_sha256": (
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
            ),
            "currency": "CNY",
            "autonomous_dashscope_budget_cny": AUTONOMOUS_DASHSCOPE_BUDGET_CNY,
            "operator_approved_dashscope_budget_cny": (
                operator_approved_dashscope_budget_cny
            ),
            "planning_ceiling_cny": planning_ceiling,
            "qwen_input_cny_per_million_tokens": 0.15,
            "qwen_output_cny_per_million_tokens": 1.5,
            "kimi_input_cny_per_million_tokens": 6.5,
            "kimi_output_cny_per_million_tokens": 27.0,
            "pricing_basis": (
                "2026-07-30 public list-price planning assumptions; "
                "single-request image/tokenization and discounts may differ"
            ),
            "aifast_cost_status": "gateway_price_not_locked",
            "checkpoint_policy": "after_every_25_query_shard",
            "over_budget_policy": (
                "reserve_before_each_provider_call_halt_before_call"
            ),
        },
        "rate": {
            "assistant_concurrency": MAX_SAFE_ASSISTANT_CONCURRENCY,
            "final_judge_concurrency": 0 if static_opt else 1,
            "feedback_concurrency": 1,
            "qwen_requests_per_minute_cap": 50,
            "provider_retry_attempts": 1,
            "checkpoint_policy": "after_every_25_query_shard",
        },
        "calls": {
            "assistant_instance_count": instance_count,
            "final_judge_instance_count": final_judge_count,
            "assistant_call_floor": assistant_call_floor,
            "assistant_call_ceiling": assistant_call_ceiling,
            "final_judge_call_count": final_judge_count,
            "final_judge_empty_response_retry_call_ceiling": final_judge_retry_ceiling,
            "feedback_calls_in_main_matrix": 0,
            "total_call_floor": assistant_call_floor + final_judge_count,
            "total_call_ceiling": (
                assistant_call_ceiling + final_judge_count + final_judge_retry_ceiling
            ),
            "note": (
                "Static opt rollout runs only LLMStaticSkill Assistant routing "
                "and action calls; GCS v2 analysis is deterministic and performs "
                "zero provider calls. Pairwise and Final Judge are prohibited."
                if static_opt
                else "NoSkill has only action calls; LLMStaticSkill and S1 reserve "
                "one private route call; S1+S2 and Full reference one shared "
                "Stage-2 route call per query and each retains the same four-turn "
                "action budget. The final Judge is planned once per successful "
                "row, while the worst-case ceiling reserves one additional call "
                "per row for the fixed empty-response retry. Actual execution "
                "accounting and cost use each persisted Judge result's attempts."
            ),
        },
        "resume_policy": "create_only_shards_skip_only_verified_complete",
        "scheduling_policy": (
            "static_opt_pool_atomic_generator_batch"
            if static_opt
            else "balanced_cyclic_config_order_by_split_atomic_generator_batch"
            if core
            else "balanced_cyclic_config_order_by_accepted_batch"
        ),
        "instances_file_sha256": instances_file_sha256,
        "shards": tuple(item.model_dump(mode="python") for item in shards),
    }
    if core:
        assert effective_splits is not None
        core_input_files = _build_core_input_files_binding(verified)
        plan_payload.update(
            {
                "dataset_profile": "core",
                "source_query_count": CORE_QUERY_COUNT,
                "selected_splits": effective_splits,
                "split_assignment_sha256": (
                    verified.files.expected_split_assignment_sha256
                ),
                "materialization_manifest_file_sha256": (
                    verified.files.expected_materialization_manifest_file_sha256
                ),
                "core_input_files": core_input_files.model_dump(mode="python"),
            }
        )
    else:
        plan_payload.update(
            {
                "accepted_ledger_sha256": verified.expected_accepted_ledger_sha256,
                "seed_set_sha256": verified.expected_seed_set_sha256,
            }
        )
    if remote:
        plan_payload.update(
            {
                "authorization_file_sha256": remote[0].authorization_file_sha256,
                "receipt_file_sha256": remote[0].receipt_file_sha256,
                "receipt_sha256": remote[0].receipt.receipt_sha256,
            }
        )
    # Normalize defaulted/excluded compatibility fields before computing the
    # self hash.  Historical dev_mini payloads have no Core-only fields, while
    # Core payloads intentionally omit legacy accepted-ledger/seed fields.
    draft = PortfolioLaunchPlan.model_validate(
        {**plan_payload, "launch_plan_sha256": "0" * 64},
        strict=True,
        context=_PLAN_NORMALIZATION_MARKER,
    )
    normalized_unsigned = draft.model_dump(mode="json", exclude={"launch_plan_sha256"})
    plan = PortfolioLaunchPlan.model_validate(
        {
            **plan_payload,
            "launch_plan_sha256": _hash_payload(normalized_unsigned),
        },
        strict=True,
    )
    return plan, instance_bytes


def reconstruct_verified_portfolio_core_inputs(
    plan: PortfolioLaunchPlan,
) -> VerifiedPortfolioCoreInputs:
    """Rebuild and deeply reverify Core inputs solely from a loaded launch plan."""

    if type(plan) is not PortfolioLaunchPlan:
        raise TypeError("plan must be PortfolioLaunchPlan")
    if (
        plan.kind
        not in {
            "portfolio-core-split-x5-launch-plan",
            "portfolio-core-static-opt-800x1-launch-plan",
        }
        or plan.dataset_profile != "core"
        or plan.core_input_files is None
    ):
        raise PortfolioLaunchError(
            "only a Core launch plan carries a reconstructable input binding"
        )
    binding = plan.core_input_files
    permission_files = binding.remote_files
    try:
        if permission_files is not None:
            for path, expected_sha256, label in (
                (
                    permission_files.selection_manifest,
                    permission_files.expected_selection_manifest_sha256,
                    "Core selection manifest",
                ),
                (
                    permission_files.dataset_assets,
                    permission_files.expected_dataset_assets_sha256,
                    "Core dataset assets",
                ),
            ):
                content = read_stable_regular_file(
                    Path(path), label=label, max_bytes=64 * 1024 * 1024
                )
                if sha256_bytes(content) != expected_sha256:
                    raise PortfolioLaunchError(f"{label} binding digest mismatch")
        verified = load_verified_portfolio_core_inputs(binding.to_input_files())
        verified = require_verified_portfolio_core_inputs(verified)
    except (ArtifactFormatError, OSError, PortfolioCoreInputError, ValueError) as error:
        if isinstance(error, PortfolioLaunchError):
            raise
        raise PortfolioLaunchError(
            "Core launch input binding could not be reconstructed"
        ) from error

    selected_query_count = sum(
        1 for query in verified.queries if query.split in plan.selected_splits
    )
    if (
        len(verified.queries) != plan.source_query_count
        or selected_query_count != plan.query_count
        or verified.expected_plan_sha256 != plan.portfolio_plan_sha256
        or verified.expected_plan_manifest_file_sha256
        != plan.portfolio_plan_manifest_file_sha256
        or verified.files.expected_split_assignment_sha256
        != plan.split_assignment_sha256
        or verified.expected_query_artifact_sha256 != plan.query_artifact_sha256
        or verified.files.expected_materialization_manifest_file_sha256
        != plan.materialization_manifest_file_sha256
        or verified.expected_capability_assignments_sha256
        != plan.capability_assignments_sha256
        or verified.expected_base_catalog_sha256 != plan.base_catalog_sha256
        or verified.expected_output_catalog_sha256 != plan.runtime_catalog_sha256
        or _remote_runtime_binding_sha256s(verified.remote_runtimes)
        != plan.remote_runtime_binding_sha256s
    ):
        raise PortfolioLaunchError(
            "reconstructed Core inputs differ from the launch profile or hashes"
        )
    return verified


def create_portfolio_launch_package(
    output_dir: str | Path,
    *,
    plan: PortfolioLaunchPlan,
    instances_bytes: bytes,
) -> CreatedPortfolioLaunchPackage:
    """Publish a create-only launch package through a sibling staging directory."""

    if type(plan) is not PortfolioLaunchPlan:
        raise TypeError("plan must be PortfolioLaunchPlan")
    if sha256_bytes(instances_bytes) != plan.instances_file_sha256:
        raise PortfolioLaunchError("instances bytes differ from launch plan")
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"launch output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        plan_bytes = canonical_json_bytes(plan.model_dump(mode="json"))
        state_payload = {
            "schema_version": 1,
            "kind": "portfolio-launch-state",
            "matrix_run_id": plan.matrix_run_id,
            "launch_plan_sha256": plan.launch_plan_sha256,
            "status": "not_started",
            "completed_shard_ids": (),
            "failed_shard_ids": (),
            "model_calls_performed": 0,
            "dashscope_observed_cost_cny": None,
            "aifast_observed_cost_cny": None,
        }
        atomic_create_file(staging / "instances.jsonl", instances_bytes)
        atomic_create_file(staging / "launch-plan.json", plan_bytes)
        state = PortfolioLaunchState.model_validate(state_payload, strict=True)
        atomic_create_file(
            staging / "run-state.json",
            canonical_json_bytes(state.model_dump(mode="json")),
        )
        staged = load_portfolio_launch_package(
            staging,
            expected_plan_file_sha256=sha256_bytes(plan_bytes),
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return CreatedPortfolioLaunchPackage(
        root=destination,
        plan_path=destination / "launch-plan.json",
        instances_path=destination / "instances.jsonl",
        state_path=destination / "run-state.json",
        plan=plan,
        plan_file_sha256=staged.plan_file_sha256,
    )


def load_portfolio_launch_package(
    root: str | Path,
    *,
    expected_plan_file_sha256: str,
    _allow_legacy_budget_contract: bool = False,
) -> LoadedPortfolioLaunchPackage:
    """Deeply verify a published plan, workload, and current resume state."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_file_sha256):
        raise PortfolioLaunchError("expected launch-plan SHA-256 is invalid")
    package_root = Path(root).absolute()
    plan_content, plan_raw = _read_canonical_object(
        package_root / "launch-plan.json",
        label="Portfolio launch plan",
    )
    if sha256_bytes(plan_content) != expected_plan_file_sha256:
        raise PortfolioLaunchError("Portfolio launch-plan file digest mismatch")
    try:
        plan = PortfolioLaunchPlan.model_validate(
            plan_raw,
            strict=True,
            context={
                "allow_legacy_budget_contract": _allow_legacy_budget_contract,
            },
        )
    except ValidationError as error:
        raise PortfolioLaunchError("Portfolio launch plan violates schema") from error
    instance_content = read_stable_regular_file(
        package_root / "instances.jsonl",
        label="Portfolio launch instances",
        max_bytes=64 * 1024 * 1024,
    )
    if sha256_bytes(instance_content) != plan.instances_file_sha256:
        raise PortfolioLaunchError("Portfolio instances file digest mismatch")
    try:
        instance_rows = parse_canonical_jsonl(
            instance_content,
            label="Portfolio launch instances",
        )
    except ArtifactFormatError as error:
        raise PortfolioLaunchError(
            "Portfolio launch instances are not canonical JSONL"
        ) from error
    instances: list[PortfolioLaunchInstance] = []
    for line_number, raw in enumerate(instance_rows, start=1):
        try:
            if not isinstance(raw, dict):
                raise PortfolioLaunchError("instance line is not an object")
            instances.append(PortfolioLaunchInstance.model_validate(raw, strict=True))
        except ValidationError as error:
            raise PortfolioLaunchError(
                f"Portfolio launch instance line {line_number} is invalid"
            ) from error
    if (
        len(instances) != plan.instance_count
        or tuple(item.instance_ordinal for item in instances)
        != tuple(range(plan.instance_count))
        or any(item.matrix_run_id != plan.matrix_run_id for item in instances)
        or len({(item.query_id, item.config) for item in instances})
        != plan.instance_count
    ):
        raise PortfolioLaunchError("Portfolio launch instance coverage drifted")
    by_shard: dict[str, list[PortfolioLaunchInstance]] = defaultdict(list)
    for instance in instances:
        by_shard[instance.shard_id].append(instance)
    if set(by_shard) != {item.shard_id for item in plan.shards}:
        raise PortfolioLaunchError("Portfolio launch shard membership drifted")
    for shard in plan.shards:
        members = by_shard[shard.shard_id]
        if (
            tuple(item.query_id for item in members) != shard.query_ids
            or tuple(item.instance_sha256 for item in members) != shard.instance_sha256s
            or any(
                item.shard_ordinal != shard.shard_ordinal
                or item.config != shard.config
                or item.accepted_batch_id != shard.accepted_batch_id
                for item in members
            )
        ):
            raise PortfolioLaunchError(
                f"Portfolio launch shard content drifted: {shard.shard_id}"
            )
    state_content, state_raw = _read_canonical_object(
        package_root / "run-state.json",
        label="Portfolio launch state",
    )
    del state_content
    try:
        state = PortfolioLaunchState.model_validate(state_raw, strict=True)
    except ValidationError as error:
        raise PortfolioLaunchError("Portfolio launch state violates schema") from error
    shard_ids = {item.shard_id for item in plan.shards}
    if (
        state.matrix_run_id != plan.matrix_run_id
        or state.launch_plan_sha256 != plan.launch_plan_sha256
        or not set(state.completed_shard_ids) <= shard_ids
        or not set(state.failed_shard_ids) <= shard_ids
    ):
        raise PortfolioLaunchError("Portfolio launch state differs from its plan")
    return LoadedPortfolioLaunchPackage(
        root=package_root,
        plan=plan,
        instances=tuple(instances),
        state=state,
        plan_file_sha256=expected_plan_file_sha256,
    )


__all__ = [
    "AUTONOMOUS_DASHSCOPE_BUDGET_CNY",
    "DEFAULT_PLANNING_CEILING_CNY",
    "CreatedPortfolioLaunchPackage",
    "LoadedPortfolioLaunchPackage",
    "PortfolioCoreInputFilesBinding",
    "PortfolioCoreRemoteProcessingFilesBinding",
    "PortfolioLaunchArtifactInputs",
    "PortfolioLaunchError",
    "PortfolioLaunchInstance",
    "PortfolioLaunchPlan",
    "PortfolioLaunchShard",
    "PortfolioLaunchState",
    "PORTFOLIO_CORE_LAUNCH_POLICY_VERSION",
    "PORTFOLIO_STATIC_OPT_LAUNCH_POLICY_VERSION",
    "PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256",
    "PORTFOLIO_CORE_ROLE_SELECTION_SHA256",
    "PORTFOLIO_LAUNCH_POLICY_VERSION",
    "PORTFOLIO_BUDGET_POLICY_SHA256",
    "PORTFOLIO_BUDGET_POLICY_SHA256_V1",
    "PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256",
    "PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V1",
    "PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION",
    "PORTFOLIO_ROLE_SELECTION_FILE_SHA256",
    "build_portfolio_launch_plan",
    "create_portfolio_launch_package",
    "load_portfolio_launch_package",
    "reconstruct_verified_portfolio_core_inputs",
]
