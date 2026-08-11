"""Portfolio shard retry receipts and provider pre-response circuit breaker.

This module deliberately owns only Portfolio execution control.  Retryable
infrastructure incidents are not evaluation outcomes and must never be
converted into task-level zero scores.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    model_validator,
)

from skillchain.evaluation.assistant_runs import AssistantRouteCallEvidence
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FailureStage = Literal[
    "shared_route",
    "assistant_route",
    "assistant_action",
    "final_judge",
]

PORTFOLIO_FAILURE_POLICY_VERSION_V1 = "portfolio-shard-attempt-v1"
PORTFOLIO_FAILURE_POLICY_VERSION_V2 = "portfolio-shard-attempt-v2"
PORTFOLIO_FAILURE_POLICY_VERSION_V3 = "portfolio-shard-attempt-v3"
PORTFOLIO_FAILURE_POLICY_VERSION = "portfolio-shard-attempt-v4"
PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD = 2
PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY = 2
PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE = 5

_ATTEMPT_FILE_RE = re.compile(r"^(?P<ordinal>\d{4})-attempt-(?P<attempt>\d{2})\.json$")
_SAFE_FAILURE_SUBTYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_EXCEPTION_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")

PORTFOLIO_BUDGET_POLICY_VERSION_V1 = "portfolio-call-hard-cap-v1"
PORTFOLIO_BUDGET_POLICY_VERSION_V2 = "portfolio-call-hard-cap-v2"
PORTFOLIO_BUDGET_POLICY_VERSION_V3 = "portfolio-call-hard-cap-v3"
PORTFOLIO_BUDGET_POLICY_VERSION = PORTFOLIO_BUDGET_POLICY_VERSION_V3
PORTFOLIO_BUDGET_CURRENCY = "CNY"
PORTFOLIO_BUDGET_DECIMAL_PLACES = 12
PORTFOLIO_BUDGET_CNY_QUANTUM = Decimal("0.000000000001")
PORTFOLIO_QWEN_PROVIDER = "qwen"
PORTFOLIO_QWEN_MODEL = "qwen3-vl-flash-2026-01-22"
# Historical Kimi constants remain exported because immutable v1/v2 ledgers
# embed them and must continue to replay exactly.
PORTFOLIO_KIMI_PROVIDER = "kimi"
PORTFOLIO_KIMI_MODEL = "kimi-k2.6"
PORTFOLIO_GEMINI_PROVIDER = "gemini"
PORTFOLIO_GEMINI_MODEL = "gemini-3.6-flash"
PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER = PORTFOLIO_GEMINI_PROVIDER
PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL = PORTFOLIO_GEMINI_MODEL
PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS = "aifast-pricing-and-ceilings-not-frozen"

PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS = 32_768
PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS = 128
# The active Router deliberately requests only the compact selector budget.
# The separate 512-token value remains the conservative, historically bound
# pricing reservation ceiling so old ledgers and runtime locks replay exactly.
PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS = 64
PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS = 512
PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS = 4_096
PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS = 258_048
PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS = 229_376
PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS = 16_384
PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS = 8_192

PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION = Decimal("0.600000000000")
PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION = Decimal("6.000000000000")
PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION = Decimal("6.500000000000")
PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION = Decimal("27.000000000000")

_QWEN_SETTLEMENT_TIERS = (
    (32_000, Decimal("0.150000000000"), Decimal("1.500000000000")),
    (128_000, Decimal("0.300000000000"), Decimal("3.000000000000")),
    (
        PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS,
        PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION,
        PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION,
    ),
)

_MILLION_TOKENS = Decimal(1_000_000)
_BUDGET_EVENT_FILE_RE = re.compile(r"^(?P<ordinal>\d{8})\.json$")
_BUDGET_APPEND_ATTEMPTS = 16
_BUDGET_IDENTITY_TEXT_MAX_LENGTH = 256

PortfolioProviderStage = Literal[
    "shared_route",
    "assistant_route",
    "assistant_action",
    "final_judge",
]


class PortfolioAttemptLimitError(RuntimeError):
    """The predeclared per-query retryable-infrastructure limit was reached."""


class PortfolioBudgetError(RuntimeError):
    """A persisted Portfolio hard-budget ledger cannot authorize an operation."""


class PortfolioBudgetExceededError(PortfolioBudgetError):
    """The next provider reservation would exceed the frozen phase cap."""


class PortfolioBudgetDuplicateCallError(PortfolioBudgetError):
    """A provider-call identity already has an immutable reservation."""


class PortfolioBudgetOrphanedCallError(PortfolioBudgetError):
    """A persisted provider call lacks its enclosing query checkpoint."""


class PortfolioBudgetSettlementError(PortfolioBudgetError):
    """A success settlement is absent, duplicated, or exceeds its reservation."""


class PortfolioBudgetForfeitError(PortfolioBudgetError):
    """A full-reserve forfeit is absent, duplicated, or conflicts with settlement."""


class PortfolioBudgetLedgerBusyError(PortfolioBudgetError):
    """Concurrent create-only writers prevented a bounded ledger append."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _self_hash(model: BaseModel, hash_field: str) -> str:
    unsigned = model.model_dump(mode="json")
    unsigned.pop(hash_field)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _validate_identity_text(value: str, label: str) -> None:
    if (
        not value
        or value != value.strip()
        or len(value) > _BUDGET_IDENTITY_TEXT_MAX_LENGTH
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{label} must be safe, non-blank, and trimmed")


def _normalize_cny(value: Decimal, label: str, *, positive: bool) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError(f"{label} must be a finite Decimal")
    try:
        normalized = value.quantize(PORTFOLIO_BUDGET_CNY_QUANTUM)
    except InvalidOperation as error:
        raise ValueError(f"{label} cannot be represented as CNY") from error
    if normalized != value:
        raise ValueError(
            f"{label} must have at most {PORTFOLIO_BUDGET_DECIMAL_PLACES} "
            "decimal places"
        )
    if normalized < 0 or (positive and normalized == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return normalized


def _validate_persisted_cny(
    value: Decimal,
    label: str,
    *,
    positive: bool,
) -> None:
    normalized = _normalize_cny(value, label, positive=positive)
    if normalized.as_tuple().exponent != -PORTFOLIO_BUDGET_DECIMAL_PLACES:
        raise ValueError(f"{label} must use the canonical 12-place CNY scale")


def _cny_json(value: Decimal) -> str:
    return format(value, f".{PORTFOLIO_BUDGET_DECIMAL_PLACES}f")


def _validate_token_count(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")


def calculate_portfolio_usage_cost_cny(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cny_per_million_tokens: Decimal,
    output_cny_per_million_tokens: Decimal,
) -> Decimal:
    """Price captured usage with exact Decimal arithmetic, rounded upward.

    Upward rounding at twelve CNY decimal places prevents the hard-cap ledger
    from understating a fractional-token-price result.
    """

    _validate_token_count(input_tokens, "input_tokens")
    _validate_token_count(output_tokens, "output_tokens")
    input_rate = _normalize_cny(
        input_cny_per_million_tokens,
        "input_cny_per_million_tokens",
        positive=True,
    )
    output_rate = _normalize_cny(
        output_cny_per_million_tokens,
        "output_cny_per_million_tokens",
        positive=True,
    )
    exact = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / _MILLION_TOKENS
    return exact.quantize(PORTFOLIO_BUDGET_CNY_QUANTUM, rounding=ROUND_CEILING)


class PortfolioBudgetCallIdentity(_StrictFrozenModel):
    """Stable identity of exactly one provider call authorization."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-budget-call-identity"] = "portfolio-budget-call-identity"
    matrix_run_id: str
    shard_id: str
    config: str
    query_id: str
    instance_sha256: Sha256
    request_sha256: Sha256
    wire_request_sha256: Sha256 | None
    stage: PortfolioProviderStage
    attempt_index: int = Field(ge=1)
    call_index: int = Field(ge=1)
    provider: str
    model: str
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        for name in (
            "matrix_run_id",
            "shard_id",
            "config",
            "query_id",
            "provider",
            "model",
        ):
            _validate_identity_text(getattr(self, name), name)
        allowed_provider_models = (
            {
                (PORTFOLIO_KIMI_PROVIDER, PORTFOLIO_KIMI_MODEL),
                (
                    PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER,
                    PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL,
                ),
            }
            if self.stage == "final_judge"
            else {(PORTFOLIO_QWEN_PROVIDER, PORTFOLIO_QWEN_MODEL)}
        )
        if (self.provider, self.model) not in allowed_provider_models:
            raise ValueError(
                "Portfolio provider/model differs from the frozen stage profile"
            )
        if self.identity_sha256 != _self_hash(self, "identity_sha256"):
            raise ValueError("Portfolio budget call identity self hash mismatch")
        return self


PortfolioPricingProfileId = Literal[
    "qwen-shared-route-v1",
    "qwen-shared-route-v2",
    "qwen-assistant-route-v1",
    "qwen-assistant-route-v2",
    "qwen-assistant-action-v1",
    "kimi-final-judge-v1",
]


class PortfolioProviderPricingProfile(_StrictFrozenModel):
    """Frozen model, stage, reserve ceiling, and settlement-price contract."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-provider-pricing-profile"] = (
        "portfolio-provider-pricing-profile"
    )
    profile_id: PortfolioPricingProfileId
    provider: str
    model: str
    stage: PortfolioProviderStage
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    provider_max_input_tokens: int = Field(ge=1)
    provider_max_output_tokens: int | None = Field(default=None, ge=1)
    reserve_input_cny_per_million_tokens: Decimal
    reserve_output_cny_per_million_tokens: Decimal
    settlement_pricing_policy: Literal[
        "qwen_input_length_tiers_v1",
        "kimi_fixed_v1",
    ]
    profile_sha256: Sha256

    @field_serializer(
        "reserve_input_cny_per_million_tokens",
        "reserve_output_cny_per_million_tokens",
        when_used="json",
    )
    def serialize_cny(self, value: Decimal) -> str:
        return _cny_json(value)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        _validate_persisted_cny(
            self.reserve_input_cny_per_million_tokens,
            "reserve_input_cny_per_million_tokens",
            positive=True,
        )
        _validate_persisted_cny(
            self.reserve_output_cny_per_million_tokens,
            "reserve_output_cny_per_million_tokens",
            positive=True,
        )
        if self.stage == "final_judge":
            valid = (
                self.profile_id == "kimi-final-judge-v1"
                and self.provider == PORTFOLIO_KIMI_PROVIDER
                and self.model == PORTFOLIO_KIMI_MODEL
                and self.max_input_tokens == PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS
                and self.provider_max_input_tokens
                == PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS
                and self.max_output_tokens
                == PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS
                and self.provider_max_output_tokens
                == PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS
                and self.reserve_input_cny_per_million_tokens
                == PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION
                and self.reserve_output_cny_per_million_tokens
                == PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION
                and self.settlement_pricing_policy == "kimi_fixed_v1"
            )
        else:
            if self.stage == "assistant_action":
                valid_profile_and_output = (
                    self.profile_id == "qwen-assistant-action-v1"
                    and self.max_output_tokens
                    == PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS
                )
            else:
                valid_profile_and_output = (
                    self.profile_id
                    in {
                        "qwen-shared-route-v1",
                        "qwen-assistant-route-v1",
                    }
                    and self.max_output_tokens
                    == PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS
                ) or (
                    self.profile_id
                    in {
                        "qwen-shared-route-v2",
                        "qwen-assistant-route-v2",
                    }
                    and self.max_output_tokens == PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
                )
                expected_stage_prefix = (
                    "qwen-shared-route"
                    if self.stage == "shared_route"
                    else "qwen-assistant-route"
                )
                valid_profile_and_output = (
                    valid_profile_and_output
                    and self.profile_id.startswith(expected_stage_prefix)
                )
            valid = (
                valid_profile_and_output
                and self.provider == PORTFOLIO_QWEN_PROVIDER
                and self.model == PORTFOLIO_QWEN_MODEL
                and self.max_input_tokens == PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS
                and self.provider_max_input_tokens
                == PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS
                and self.provider_max_output_tokens is None
                and self.reserve_input_cny_per_million_tokens
                == PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION
                and self.reserve_output_cny_per_million_tokens
                == PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION
                and self.settlement_pricing_policy == "qwen_input_length_tiers_v1"
            )
        if not valid:
            raise ValueError("Portfolio provider pricing profile drifted")
        if self.profile_sha256 != _self_hash(self, "profile_sha256"):
            raise ValueError("Portfolio provider pricing profile self hash mismatch")
        return self


def make_portfolio_provider_pricing_profile(
    identity: PortfolioBudgetCallIdentity,
    *,
    final_judge_max_output_tokens: int | None = None,
) -> PortfolioProviderPricingProfile:
    """Derive the only allowed reserve/pricing profile from the call stage."""

    if type(identity) is not PortfolioBudgetCallIdentity:
        raise TypeError("identity must be a PortfolioBudgetCallIdentity")
    if identity.stage == "final_judge":
        if (identity.provider, identity.model) == (
            PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER,
            PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL,
        ):
            raise PortfolioBudgetError(
                "AIFast Gemini final-Judge pricing and provider token ceilings "
                "are not frozen; provider reservation is forbidden"
            )
        if (identity.provider, identity.model) != (
            PORTFOLIO_KIMI_PROVIDER,
            PORTFOLIO_KIMI_MODEL,
        ):
            raise PortfolioBudgetError("unknown historical final-Judge pricing profile")
        if (
            type(final_judge_max_output_tokens) is not int
            or final_judge_max_output_tokens
            != PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS
        ):
            raise ValueError(
                "final-Judge stage output reserve must equal the frozen 8,192 "
                "tokens (6,144 thinking + 2,048 answer)"
            )
        profile_id = "kimi-final-judge-v1"
        max_input_tokens = PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS
        max_output_tokens = PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS
        provider_max_input_tokens = PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS
        provider_max_output_tokens = PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS
        reserve_input_rate = PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION
        reserve_output_rate = PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION
        settlement_policy = "kimi_fixed_v1"
    else:
        if final_judge_max_output_tokens is not None:
            raise ValueError("Qwen stages cannot accept a final-Judge output contract")
        profile_id = {
            "shared_route": "qwen-shared-route-v2",
            "assistant_route": "qwen-assistant-route-v2",
            "assistant_action": "qwen-assistant-action-v1",
        }[identity.stage]
        max_input_tokens = PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS
        max_output_tokens = (
            PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS
            if identity.stage == "assistant_action"
            else PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
        )
        provider_max_input_tokens = PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS
        provider_max_output_tokens = None
        reserve_input_rate = PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION
        reserve_output_rate = PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION
        settlement_policy = "qwen_input_length_tiers_v1"
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-provider-pricing-profile",
        "profile_id": profile_id,
        "provider": identity.provider,
        "model": identity.model,
        "stage": identity.stage,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "provider_max_input_tokens": provider_max_input_tokens,
        "provider_max_output_tokens": provider_max_output_tokens,
        "reserve_input_cny_per_million_tokens": _cny_json(reserve_input_rate),
        "reserve_output_cny_per_million_tokens": _cny_json(reserve_output_rate),
        "settlement_pricing_policy": settlement_policy,
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "profile_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    return PortfolioProviderPricingProfile.model_validate_json(content, strict=True)


def calculate_portfolio_settled_usage_cost_cny(
    profile: PortfolioProviderPricingProfile,
    *,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Price successful usage under the profile's frozen settlement policy."""

    if type(profile) is not PortfolioProviderPricingProfile:
        raise TypeError("profile must be a PortfolioProviderPricingProfile")
    _validate_token_count(input_tokens, "input_tokens")
    _validate_token_count(output_tokens, "output_tokens")
    if (
        input_tokens > profile.provider_max_input_tokens
        or output_tokens > profile.max_output_tokens
    ):
        raise PortfolioBudgetSettlementError(
            "captured success usage exceeds the frozen provider profile"
        )
    if profile.settlement_pricing_policy == "kimi_fixed_v1":
        input_rate = PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION
        output_rate = PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION
    else:
        input_rate: Decimal | None = None
        output_rate: Decimal | None = None
        for maximum, candidate_input, candidate_output in _QWEN_SETTLEMENT_TIERS:
            if input_tokens <= maximum:
                input_rate = candidate_input
                output_rate = candidate_output
                break
        if input_rate is None or output_rate is None:
            raise PortfolioBudgetSettlementError(
                "captured Qwen input exceeds the frozen pricing tiers"
            )
    return calculate_portfolio_usage_cost_cny(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cny_per_million_tokens=input_rate,
        output_cny_per_million_tokens=output_rate,
    )


class PortfolioBudgetAuthority(_StrictFrozenModel):
    """Create-only phase cap and pre-ledger observed-cost authority."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-budget-authority"] = "portfolio-budget-authority"
    policy_version: Literal[
        "portfolio-call-hard-cap-v1",
        "portfolio-call-hard-cap-v2",
        "portfolio-call-hard-cap-v3",
    ] = PORTFOLIO_BUDGET_POLICY_VERSION
    matrix_run_id: str
    currency: Literal["CNY"] = PORTFOLIO_BUDGET_CURRENCY
    phase_cap_cny: Decimal
    prior_observed_cost_cny: Decimal
    accounting_policy: Literal[
        "settled_actual_plus_unresolved_reserve",
        "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve",
    ] = "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
    exception_policy: Literal[
        "retain_full_reserve",
        "append_full_reserve_forfeit_without_captured_response",
    ] = "append_full_reserve_forfeit_without_captured_response"
    authority_sha256: Sha256

    @field_serializer("phase_cap_cny", "prior_observed_cost_cny", when_used="json")
    def serialize_cny(self, value: Decimal) -> str:
        return _cny_json(value)

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        _validate_identity_text(self.matrix_run_id, "matrix_run_id")
        _validate_persisted_cny(self.phase_cap_cny, "phase_cap_cny", positive=True)
        _validate_persisted_cny(
            self.prior_observed_cost_cny,
            "prior_observed_cost_cny",
            positive=False,
        )
        if self.prior_observed_cost_cny >= self.phase_cap_cny:
            raise ValueError("prior observed cost must be below the phase cap")
        expected_policies = {
            PORTFOLIO_BUDGET_POLICY_VERSION_V1: (
                "settled_actual_plus_unresolved_reserve",
                "retain_full_reserve",
            ),
            PORTFOLIO_BUDGET_POLICY_VERSION_V2: (
                "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve",
                "append_full_reserve_forfeit_without_captured_response",
            ),
            PORTFOLIO_BUDGET_POLICY_VERSION: (
                "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve",
                "append_full_reserve_forfeit_without_captured_response",
            ),
        }[self.policy_version]
        if (self.accounting_policy, self.exception_policy) != expected_policies:
            raise ValueError(
                "Portfolio budget authority policies differ from its version"
            )
        if self.authority_sha256 != _self_hash(self, "authority_sha256"):
            raise ValueError("Portfolio budget authority self hash mismatch")
        return self


class PortfolioBudgetReservation(_StrictFrozenModel):
    """One create-only, pre-provider maximum-cost reservation event."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-budget-reservation"] = "portfolio-budget-reservation"
    policy_version: Literal[
        "portfolio-call-hard-cap-v1",
        "portfolio-call-hard-cap-v2",
        "portfolio-call-hard-cap-v3",
    ] = PORTFOLIO_BUDGET_POLICY_VERSION
    authority_sha256: Sha256
    ledger_event_index: int = Field(ge=1)
    previous_event_sha256: Sha256 | None
    identity: PortfolioBudgetCallIdentity
    pricing_profile: PortfolioProviderPricingProfile
    reserved_cost_cny: Decimal
    disposition: Literal["reserved_before_provider_call"] = (
        "reserved_before_provider_call"
    )
    reservation_sha256: Sha256

    @field_serializer(
        "reserved_cost_cny",
        when_used="json",
    )
    def serialize_cny(self, value: Decimal) -> str:
        return _cny_json(value)

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        _validate_persisted_cny(
            self.reserved_cost_cny,
            "reserved_cost_cny",
            positive=True,
        )
        if (
            self.identity.provider != self.pricing_profile.provider
            or self.identity.model != self.pricing_profile.model
            or self.identity.stage != self.pricing_profile.stage
        ):
            raise ValueError("Portfolio reservation identity/profile binding drifted")
        expected_cost = calculate_portfolio_usage_cost_cny(
            input_tokens=self.pricing_profile.provider_max_input_tokens,
            output_tokens=self.pricing_profile.max_output_tokens,
            input_cny_per_million_tokens=(
                self.pricing_profile.reserve_input_cny_per_million_tokens
            ),
            output_cny_per_million_tokens=(
                self.pricing_profile.reserve_output_cny_per_million_tokens
            ),
        )
        if self.reserved_cost_cny != expected_cost:
            raise ValueError("Portfolio reservation differs from maximum usage cost")
        if self.reservation_sha256 != _self_hash(self, "reservation_sha256"):
            raise ValueError("Portfolio budget reservation self hash mismatch")
        return self

    @property
    def max_input_tokens(self) -> int:
        return self.pricing_profile.max_input_tokens

    @property
    def max_output_tokens(self) -> int:
        return self.pricing_profile.max_output_tokens


class PortfolioBudgetSettlement(_StrictFrozenModel):
    """One create-only success event settled from captured provider usage."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-budget-settlement"] = "portfolio-budget-settlement"
    policy_version: Literal[
        "portfolio-call-hard-cap-v1",
        "portfolio-call-hard-cap-v2",
        "portfolio-call-hard-cap-v3",
    ] = PORTFOLIO_BUDGET_POLICY_VERSION
    authority_sha256: Sha256
    ledger_event_index: int = Field(ge=1)
    previous_event_sha256: Sha256 | None
    reservation_sha256: Sha256
    identity_sha256: Sha256
    actual_input_tokens: int = Field(ge=0)
    actual_output_tokens: int = Field(ge=0)
    actual_cost_cny: Decimal
    provider_request_id: str
    response_sha256: Sha256
    outcome: Literal["success"] = "success"
    disposition: Literal["settled_from_actual_usage"] = "settled_from_actual_usage"
    settlement_sha256: Sha256

    @field_serializer("actual_cost_cny", when_used="json")
    def serialize_cny(self, value: Decimal) -> str:
        return _cny_json(value)

    @model_validator(mode="after")
    def validate_settlement(self) -> Self:
        _validate_identity_text(self.provider_request_id, "provider_request_id")
        if self.actual_input_tokens == 0 and self.actual_output_tokens == 0:
            raise ValueError("successful provider settlement cannot have zero usage")
        _validate_persisted_cny(
            self.actual_cost_cny,
            "actual_cost_cny",
            positive=False,
        )
        if self.settlement_sha256 != _self_hash(self, "settlement_sha256"):
            raise ValueError("Portfolio budget settlement self hash mismatch")
        return self


PortfolioBudgetForfeitReason = Literal[
    "provider_call_ended_without_captured_response",
    "orphan_recovered_after_owner_exit",
]


class PortfolioBudgetForfeit(_StrictFrozenModel):
    """One create-only terminal event charged at the complete reservation.

    A forfeit means the client captured no valid provider response and therefore
    cannot price actual usage.  It closes the reservation without releasing any
    budget or fabricating a successful usage settlement.
    """

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-budget-forfeit"] = "portfolio-budget-forfeit"
    policy_version: Literal[
        "portfolio-call-hard-cap-v2",
        "portfolio-call-hard-cap-v3",
    ] = PORTFOLIO_BUDGET_POLICY_VERSION
    authority_sha256: Sha256
    ledger_event_index: int = Field(ge=1)
    previous_event_sha256: Sha256 | None
    reservation_sha256: Sha256
    identity_sha256: Sha256
    forfeited_cost_cny: Decimal
    captured_provider_response: Literal[False] = False
    actual_usage_status: Literal["unknown"] = "unknown"
    reason: PortfolioBudgetForfeitReason
    disposition: Literal["charged_full_reserve"] = "charged_full_reserve"
    forfeit_sha256: Sha256

    @field_serializer("forfeited_cost_cny", when_used="json")
    def serialize_cny(self, value: Decimal) -> str:
        return _cny_json(value)

    @model_validator(mode="after")
    def validate_forfeit(self) -> Self:
        _validate_persisted_cny(
            self.forfeited_cost_cny,
            "forfeited_cost_cny",
            positive=True,
        )
        if self.forfeit_sha256 != _self_hash(self, "forfeit_sha256"):
            raise ValueError("Portfolio budget forfeit self hash mismatch")
        return self


PortfolioBudgetEvent = (
    PortfolioBudgetReservation | PortfolioBudgetSettlement | PortfolioBudgetForfeit
)


def _portfolio_budget_event_sha256(event: PortfolioBudgetEvent) -> str:
    if isinstance(event, PortfolioBudgetReservation):
        return event.reservation_sha256
    if isinstance(event, PortfolioBudgetSettlement):
        return event.settlement_sha256
    if isinstance(event, PortfolioBudgetForfeit):
        return event.forfeit_sha256
    raise TypeError("event must be a Portfolio budget ledger event")


@dataclass(frozen=True)
class PortfolioBudgetLedgerState:
    """Validated restart state; unresolved reservations remain fully charged."""

    authority: PortfolioBudgetAuthority
    reservations: tuple[PortfolioBudgetReservation, ...]
    settlements: tuple[PortfolioBudgetSettlement, ...]
    forfeits: tuple[PortfolioBudgetForfeit, ...]
    unresolved_reservations: tuple[PortfolioBudgetReservation, ...]
    settled_actual_cost_cny: Decimal
    forfeited_reserved_cost_cny: Decimal
    unresolved_reserved_cost_cny: Decimal
    accountable_cost_cny: Decimal
    remaining_cost_cny: Decimal
    last_event_index: int
    last_event_sha256: str | None


def portfolio_budget_settled_cost_cny(
    state: PortfolioBudgetLedgerState,
    *,
    shard_ids: Collection[str] | None = None,
    stages: Collection[PortfolioProviderStage] | None = None,
) -> Decimal:
    """Return exact settled cost for an optional shard/stage projection.

    Settlement events are the pricing authority: unlike artifact-level token
    summaries, they retain the frozen provider pricing profile used for each
    call, including Qwen's input-length tiers.
    """

    shard_filter = None if shard_ids is None else frozenset(shard_ids)
    stage_filter = None if stages is None else frozenset(stages)
    if shard_filter is not None and any(
        not value or value != value.strip() for value in shard_filter
    ):
        raise ValueError("shard_ids must contain non-blank, trimmed values")
    valid_stages = frozenset(
        {"shared_route", "assistant_route", "assistant_action", "final_judge"}
    )
    if stage_filter is not None and not stage_filter <= valid_stages:
        raise ValueError("stages contains an unsupported provider stage")

    reservations = {item.reservation_sha256: item for item in state.reservations}
    return sum(
        (
            settlement.actual_cost_cny
            for settlement in state.settlements
            if (
                shard_filter is None
                or reservations[settlement.reservation_sha256].identity.shard_id
                in shard_filter
            )
            and (
                stage_filter is None
                or reservations[settlement.reservation_sha256].identity.stage
                in stage_filter
            )
        ),
        Decimal("0.000000000000"),
    )


@dataclass
class PortfolioBudgetLedgerSession:
    """One-process validated-prefix cache for bounded incremental appends.

    A session performs one full ledger validation when opened.  Before every
    append it revalidates the immutable authority and current tail event.  If
    another writer has appended the next event, it falls back to a full reload;
    otherwise the locally created event advances the already-validated prefix
    in memory.  Shard finalization still performs an independent full replay.
    """

    ledger_root: Path
    _state: PortfolioBudgetLedgerState

    @property
    def state(self) -> PortfolioBudgetLedgerState:
        return self._state

    def require_current(self) -> PortfolioBudgetLedgerState:
        authority = load_portfolio_budget_authority(self.ledger_root)
        if authority != self._state.authority:
            raise PortfolioBudgetError(
                "Portfolio budget authority changed during an active session"
            )
        directory = portfolio_budget_event_directory(self.ledger_root)
        if self._state.last_event_index:
            tail_path = directory / f"{self._state.last_event_index:08d}.json"
            tail = _load_portfolio_budget_event(tail_path)
            tail_sha256 = _portfolio_budget_event_sha256(tail)
            if tail_sha256 != self._state.last_event_sha256:
                raise PortfolioBudgetError(
                    "Portfolio budget validated-prefix tail changed"
                )
        next_path = directory / f"{self._state.last_event_index + 1:08d}.json"
        if next_path.exists():
            self._state = load_portfolio_budget_ledger(self.ledger_root)
        return self._state

    def reload(self) -> PortfolioBudgetLedgerState:
        self._state = load_portfolio_budget_ledger(self.ledger_root)
        return self._state

    def accept_reservation(self, reservation: PortfolioBudgetReservation) -> None:
        state = self._state
        if (
            reservation.authority_sha256 != state.authority.authority_sha256
            or reservation.policy_version != state.authority.policy_version
            or reservation.ledger_event_index != state.last_event_index + 1
            or reservation.previous_event_sha256 != state.last_event_sha256
            or any(
                item.identity.identity_sha256 == reservation.identity.identity_sha256
                for item in state.reservations
            )
        ):
            raise PortfolioBudgetError(
                "Portfolio budget reservation cannot advance validated prefix"
            )
        reservations = state.reservations + (reservation,)
        unresolved = state.unresolved_reservations + (reservation,)
        unresolved_cost = (
            state.unresolved_reserved_cost_cny + reservation.reserved_cost_cny
        )
        accountable = (
            state.authority.prior_observed_cost_cny
            + state.settled_actual_cost_cny
            + state.forfeited_reserved_cost_cny
            + unresolved_cost
        )
        if accountable > state.authority.phase_cap_cny:
            raise PortfolioBudgetError(
                "Portfolio budget session advanced beyond the frozen cap"
            )
        self._state = PortfolioBudgetLedgerState(
            authority=state.authority,
            reservations=reservations,
            settlements=state.settlements,
            forfeits=state.forfeits,
            unresolved_reservations=unresolved,
            settled_actual_cost_cny=state.settled_actual_cost_cny,
            forfeited_reserved_cost_cny=state.forfeited_reserved_cost_cny,
            unresolved_reserved_cost_cny=unresolved_cost,
            accountable_cost_cny=accountable,
            remaining_cost_cny=state.authority.phase_cap_cny - accountable,
            last_event_index=reservation.ledger_event_index,
            last_event_sha256=reservation.reservation_sha256,
        )

    def accept_settlement(self, settlement: PortfolioBudgetSettlement) -> None:
        state = self._state
        reservation = next(
            (
                item
                for item in state.reservations
                if item.reservation_sha256 == settlement.reservation_sha256
            ),
            None,
        )
        if (
            reservation is None
            or settlement.authority_sha256 != state.authority.authority_sha256
            or settlement.policy_version != state.authority.policy_version
            or settlement.identity_sha256 != reservation.identity.identity_sha256
            or settlement.ledger_event_index != state.last_event_index + 1
            or settlement.previous_event_sha256 != state.last_event_sha256
            or any(
                item.reservation_sha256 == settlement.reservation_sha256
                for item in state.settlements
            )
            or any(
                item.reservation_sha256 == settlement.reservation_sha256
                for item in state.forfeits
            )
        ):
            raise PortfolioBudgetError(
                "Portfolio budget settlement cannot advance validated prefix"
            )
        settlements = state.settlements + (settlement,)
        unresolved = tuple(
            item
            for item in state.unresolved_reservations
            if item.reservation_sha256 != reservation.reservation_sha256
        )
        unresolved_cost = (
            state.unresolved_reserved_cost_cny - reservation.reserved_cost_cny
        )
        settled_cost = state.settled_actual_cost_cny + settlement.actual_cost_cny
        accountable = (
            state.authority.prior_observed_cost_cny
            + settled_cost
            + state.forfeited_reserved_cost_cny
            + unresolved_cost
        )
        self._state = PortfolioBudgetLedgerState(
            authority=state.authority,
            reservations=state.reservations,
            settlements=settlements,
            forfeits=state.forfeits,
            unresolved_reservations=unresolved,
            settled_actual_cost_cny=settled_cost,
            forfeited_reserved_cost_cny=state.forfeited_reserved_cost_cny,
            unresolved_reserved_cost_cny=unresolved_cost,
            accountable_cost_cny=accountable,
            remaining_cost_cny=state.authority.phase_cap_cny - accountable,
            last_event_index=settlement.ledger_event_index,
            last_event_sha256=settlement.settlement_sha256,
        )

    def accept_forfeit(self, forfeit: PortfolioBudgetForfeit) -> None:
        state = self._state
        reservation = next(
            (
                item
                for item in state.reservations
                if item.reservation_sha256 == forfeit.reservation_sha256
            ),
            None,
        )
        if (
            reservation is None
            or forfeit.authority_sha256 != state.authority.authority_sha256
            or forfeit.policy_version != state.authority.policy_version
            or forfeit.identity_sha256 != reservation.identity.identity_sha256
            or forfeit.forfeited_cost_cny != reservation.reserved_cost_cny
            or forfeit.ledger_event_index != state.last_event_index + 1
            or forfeit.previous_event_sha256 != state.last_event_sha256
            or any(
                item.reservation_sha256 == forfeit.reservation_sha256
                for item in state.settlements
            )
            or any(
                item.reservation_sha256 == forfeit.reservation_sha256
                for item in state.forfeits
            )
        ):
            raise PortfolioBudgetError(
                "Portfolio budget forfeit cannot advance validated prefix"
            )
        forfeits = state.forfeits + (forfeit,)
        unresolved = tuple(
            item
            for item in state.unresolved_reservations
            if item.reservation_sha256 != reservation.reservation_sha256
        )
        unresolved_cost = (
            state.unresolved_reserved_cost_cny - reservation.reserved_cost_cny
        )
        forfeited_cost = state.forfeited_reserved_cost_cny + forfeit.forfeited_cost_cny
        accountable = (
            state.authority.prior_observed_cost_cny
            + state.settled_actual_cost_cny
            + forfeited_cost
            + unresolved_cost
        )
        if accountable != state.accountable_cost_cny:
            raise PortfolioBudgetError(
                "full-reserve forfeit changed accountable Portfolio cost"
            )
        self._state = PortfolioBudgetLedgerState(
            authority=state.authority,
            reservations=state.reservations,
            settlements=state.settlements,
            forfeits=forfeits,
            unresolved_reservations=unresolved,
            settled_actual_cost_cny=state.settled_actual_cost_cny,
            forfeited_reserved_cost_cny=forfeited_cost,
            unresolved_reserved_cost_cny=unresolved_cost,
            accountable_cost_cny=accountable,
            remaining_cost_cny=state.authority.phase_cap_cny - accountable,
            last_event_index=forfeit.ledger_event_index,
            last_event_sha256=forfeit.forfeit_sha256,
        )


_ACTIVE_BUDGET_LEDGER_SESSIONS: dict[Path, PortfolioBudgetLedgerSession] = {}


class ValidatedRouteIdentity(_StrictFrozenModel):
    """A route identity retained after a later execution-stage failure."""

    selected_capability: str
    skill_slug: str
    route_trace_sha256: Sha256
    bank_sha256: Sha256

    @model_validator(mode="after")
    def validate_text(self) -> Self:
        for name in ("selected_capability", "skill_slug"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-blank and trimmed")
        return self


class PortfolioAttemptReceipt(_StrictFrozenModel):
    """Create-only evidence for one retryable, non-scoring infrastructure attempt."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-shard-attempt-receipt"] = "portfolio-shard-attempt-receipt"
    policy_version: Literal[
        "portfolio-shard-attempt-v1",
        "portfolio-shard-attempt-v2",
        "portfolio-shard-attempt-v3",
        "portfolio-shard-attempt-v4",
    ] = PORTFOLIO_FAILURE_POLICY_VERSION
    matrix_run_id: str
    shard_id: str
    config: str
    instance_sha256: Sha256
    query_id: str
    query_ordinal: int = Field(ge=0)
    request_sha256: Sha256
    attempt_index: int = Field(ge=1, le=PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY)
    failure_stage: FailureStage
    failure_subtype: str
    retryable: Literal[True] = True
    score_disposition: Literal["not_scored_no_fixed_zero"] = "not_scored_no_fixed_zero"
    recovery_policy: Literal["resume_earliest_incomplete_frozen_member"] = (
        "resume_earliest_incomplete_frozen_member"
    )
    automatic_same_process_retry: Literal[False] = False
    max_retryable_attempts_per_query: Literal[2] = (
        PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
    )
    circuit_id: str
    circuit_breaker_threshold: Literal[2] = PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
    consecutive_retryable_failures: int = Field(ge=1)
    circuit_open: bool
    captured_provider_response_count: int = Field(ge=0)
    captured_input_tokens: int = Field(ge=0)
    captured_output_tokens: int = Field(ge=0)
    assistant_response_sha256: Sha256 | None = None
    assistant_receipt_sha256: Sha256 | None = None
    judge_result_sha256: Sha256 | None = None
    validated_route_identity: ValidatedRouteIdentity | None = None
    route_call_evidence: AssistantRouteCallEvidence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    forfeited_reservation_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    budget_forfeit_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    exception_type: str | None = None
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        for name in (
            "matrix_run_id",
            "shard_id",
            "config",
            "query_id",
            "circuit_id",
        ):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-blank and trimmed")
        if not _SAFE_FAILURE_SUBTYPE_RE.fullmatch(self.failure_subtype):
            raise ValueError("failure_subtype is not a safe predeclared identifier")
        if self.exception_type is not None and not _SAFE_EXCEPTION_TYPE_RE.fullmatch(
            self.exception_type
        ):
            raise ValueError("exception_type must contain only a type identity")
        if self.circuit_open != (
            self.consecutive_retryable_failures >= self.circuit_breaker_threshold
        ):
            raise ValueError("circuit-open state differs from the frozen threshold")
        if self.captured_provider_response_count == 0 and (
            self.captured_input_tokens != 0 or self.captured_output_tokens != 0
        ):
            raise ValueError("uncaptured provider responses cannot carry token usage")
        evidence = (
            self.assistant_response_sha256,
            self.assistant_receipt_sha256,
        )
        if any(value is None for value in evidence) != all(
            value is None for value in evidence
        ):
            raise ValueError("Assistant failure evidence must be complete or absent")
        if self.failure_stage == "final_judge":
            if self.judge_result_sha256 is None and self.exception_type is None:
                raise ValueError(
                    "final-Judge failure lacks result or exception identity"
                )
        elif self.judge_result_sha256 is not None:
            raise ValueError("non-Judge failure cannot bind a Judge result")
        forfeit_fields = (
            self.forfeited_reservation_sha256,
            self.budget_forfeit_sha256,
        )
        if any(value is None for value in forfeit_fields) != all(
            value is None for value in forfeit_fields
        ):
            raise ValueError("budget forfeit evidence must be complete or absent")
        has_forfeit = self.budget_forfeit_sha256 is not None
        provider_pre_response = self.failure_subtype.startswith("provider_pre_response")
        orphaned_provider_call = self.failure_subtype == "orphaned_provider_call"
        if self.policy_version == PORTFOLIO_FAILURE_POLICY_VERSION:
            if provider_pre_response and not has_forfeit:
                raise ValueError(
                    "active provider-pre-response receipt requires budget forfeit"
                )
            if (
                orphaned_provider_call
                and self.captured_provider_response_count == 0
                and not has_forfeit
            ):
                raise ValueError("unsettled orphan receipt requires budget forfeit")
            if has_forfeit and not (provider_pre_response or orphaned_provider_call):
                raise ValueError(
                    "non-provider failure cannot bind budget forfeit evidence"
                )
        elif has_forfeit:
            raise ValueError("legacy attempt receipt cannot bind budget forfeit")
        route_contract_failure = self.failure_subtype.startswith("route_contract_")
        if self.route_call_evidence is not None:
            evidence = self.route_call_evidence
            common_evidence_drift = (
                not route_contract_failure
                or self.failure_stage not in {"shared_route", "assistant_route"}
                or evidence.attempt_index != self.attempt_index
                or evidence.call_receipt.input_tokens != self.captured_input_tokens
                or evidence.call_receipt.output_tokens != self.captured_output_tokens
                or self.captured_provider_response_count < 1
            )
            if self.policy_version in {
                PORTFOLIO_FAILURE_POLICY_VERSION_V3,
                PORTFOLIO_FAILURE_POLICY_VERSION,
            }:
                expected_route_failure = {
                    "route_contract_invalid_json": (
                        "invalid_route_json",
                        "invalid_json",
                    ),
                    "route_contract_non_object": (
                        "invalid_route_json",
                        "non_object",
                    ),
                    "route_contract_unexpected_keys": (
                        "invalid_route_json",
                        "unexpected_keys",
                    ),
                    "route_contract_schema_invalid": (
                        "invalid_route_json",
                        "schema_invalid",
                    ),
                    "route_contract_length": ("length", "not_examined"),
                    "route_contract_empty": ("response_empty_text", "empty"),
                }.get(self.failure_subtype)
                active_evidence_drift = (
                    expected_route_failure is None
                    or evidence.policy_version != "assistant-route-call-evidence-v2"
                    or (evidence.failure_reason, evidence.payload_status)
                    != expected_route_failure
                    or evidence.request_variant != "initial"
                    or evidence.repair_of_wire_request_sha256 is not None
                    or evidence.ignored_response_keys != ()
                )
            else:
                active_evidence_drift = (
                    evidence.payload_status
                    != self.failure_subtype.removeprefix("route_contract_")
                    or evidence.failure_reason != "invalid_route_json"
                )
            if common_evidence_drift or active_evidence_drift:
                raise ValueError("route-call evidence differs from attempt receipt")
        if (
            self.policy_version
            in {
                PORTFOLIO_FAILURE_POLICY_VERSION_V2,
                PORTFOLIO_FAILURE_POLICY_VERSION_V3,
                PORTFOLIO_FAILURE_POLICY_VERSION,
            }
            and route_contract_failure
            and self.route_call_evidence is None
        ):
            raise ValueError("active route-contract retry requires call evidence")
        if not route_contract_failure and self.route_call_evidence is not None:
            raise ValueError("non-route retry cannot retain route-call evidence")
        unsigned = self.model_dump(mode="json")
        observed = unsigned.pop("receipt_sha256")
        if observed != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("Portfolio attempt receipt self hash mismatch")
        return self


def portfolio_attempt_provider_call_count(
    receipt: PortfolioAttemptReceipt,
) -> int:
    """Return calls represented by one abandoned-attempt receipt.

    ``captured_provider_response_count`` already includes every provider call
    whose response was persisted in the receipt. Only a provider-pre-response
    terminal failure represents one additional invocation without a captured
    response. Route-contract failures captured their terminal response, while
    settled-orphan receipts describe calls already represented by settlements.
    """

    if getattr(receipt, "policy_version", None) == PORTFOLIO_FAILURE_POLICY_VERSION:
        return receipt.captured_provider_response_count + int(
            getattr(receipt, "budget_forfeit_sha256", None) is not None
        )
    return receipt.captured_provider_response_count + int(
        receipt.failure_subtype.startswith("provider_pre_response")
    )


@dataclass
class ProviderPreResponseCircuitBreaker:
    """In-process, provider-scoped breaker with a frozen two-failure threshold."""

    threshold: int = PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
    _consecutive_failures: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold != PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD:
            raise ValueError("Portfolio circuit threshold is frozen at two")

    def record_valid_response(self, circuit_id: str) -> None:
        _validate_circuit_id(circuit_id)
        self._consecutive_failures[circuit_id] = 0

    def restore_failure_count(self, circuit_id: str, count: int) -> None:
        """Restore persisted state before a resumed provider attempt."""

        _validate_circuit_id(circuit_id)
        if count < 0 or count >= self.threshold:
            raise ValueError("restored circuit count must be below the threshold")
        self._consecutive_failures[circuit_id] = count

    def record_retryable_failure(self, circuit_id: str) -> tuple[int, bool]:
        _validate_circuit_id(circuit_id)
        count = self._consecutive_failures.get(circuit_id, 0) + 1
        self._consecutive_failures[circuit_id] = count
        return count, count >= self.threshold


def _validate_circuit_id(value: str) -> None:
    if not value or value != value.strip():
        raise ValueError("circuit_id must be non-blank and trimmed")


def attempt_receipt_directory(shard_root: Path) -> Path:
    return shard_root / "attempt-receipts"


def load_query_attempt_receipts(
    shard_root: Path,
    *,
    query_ordinal: int,
    query_id: str,
    instance_sha256: str,
) -> tuple[PortfolioAttemptReceipt, ...]:
    """Load a query's immutable receipts and reject gaps or binding drift."""

    directory = attempt_receipt_directory(shard_root)
    if not directory.exists():
        return ()
    loaded: list[PortfolioAttemptReceipt] = []
    prefix = f"{query_ordinal:04d}-attempt-"
    for path in sorted(directory.glob(f"{prefix}*.json")):
        match = _ATTEMPT_FILE_RE.fullmatch(path.name)
        if match is None or int(match.group("ordinal")) != query_ordinal:
            raise ValueError(f"invalid Portfolio attempt receipt filename: {path}")
        raw = parse_canonical_json(
            path.read_bytes(),
            label=f"Portfolio attempt receipt {path.name}",
        )
        if not isinstance(raw, dict):
            raise ValueError("Portfolio attempt receipt must contain an object")
        receipt = PortfolioAttemptReceipt.model_validate(raw, strict=True)
        expected_index = len(loaded) + 1
        if (
            int(match.group("attempt")) != expected_index
            or receipt.attempt_index != expected_index
            or receipt.query_ordinal != query_ordinal
            or receipt.query_id != query_id
            or receipt.instance_sha256 != instance_sha256
        ):
            raise ValueError("Portfolio attempt receipt order or binding mismatch")
        loaded.append(receipt)
    return tuple(loaded)


def require_retryable_attempt_available(
    shard_root: Path,
    *,
    query_ordinal: int,
    query_id: str,
    instance_sha256: str,
) -> int:
    """Return the next attempt index or fail before any provider invocation."""

    receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=query_ordinal,
        query_id=query_id,
        instance_sha256=instance_sha256,
    )
    if len(receipts) >= PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY:
        raise PortfolioAttemptLimitError(
            "retryable infrastructure attempt limit reached for the earliest "
            "incomplete frozen member"
        )
    return len(receipts) + 1


def create_retryable_attempt_receipt(
    shard_root: Path,
    *,
    breaker: ProviderPreResponseCircuitBreaker,
    matrix_run_id: str,
    shard_id: str,
    config: str,
    instance_sha256: str,
    query_id: str,
    query_ordinal: int,
    request_sha256: str,
    failure_stage: FailureStage,
    failure_subtype: str,
    circuit_id: str,
    captured_provider_response_count: int = 0,
    captured_input_tokens: int = 0,
    captured_output_tokens: int = 0,
    assistant_response_sha256: str | None = None,
    assistant_receipt_sha256: str | None = None,
    judge_result_sha256: str | None = None,
    validated_route_identity: ValidatedRouteIdentity | None = None,
    route_call_evidence: AssistantRouteCallEvidence | None = None,
    forfeited_reservation_sha256: str | None = None,
    budget_forfeit_sha256: str | None = None,
    exception_type: str | None = None,
) -> tuple[PortfolioAttemptReceipt, Path]:
    """Atomically create the next ordered receipt; never overwrite or append."""

    prior_receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=query_ordinal,
        query_id=query_id,
        instance_sha256=instance_sha256,
    )
    if len(prior_receipts) >= PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY:
        raise PortfolioAttemptLimitError(
            "retryable infrastructure attempt limit reached for the earliest "
            "incomplete frozen member"
        )
    attempt_index = len(prior_receipts) + 1
    if prior_receipts and prior_receipts[-1].circuit_id == circuit_id:
        breaker.restore_failure_count(
            circuit_id,
            prior_receipts[-1].consecutive_retryable_failures,
        )
    consecutive, circuit_open = breaker.record_retryable_failure(circuit_id)
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-shard-attempt-receipt",
        "policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
        "matrix_run_id": matrix_run_id,
        "shard_id": shard_id,
        "config": config,
        "instance_sha256": instance_sha256,
        "query_id": query_id,
        "query_ordinal": query_ordinal,
        "request_sha256": request_sha256,
        "attempt_index": attempt_index,
        "failure_stage": failure_stage,
        "failure_subtype": failure_subtype,
        "retryable": True,
        "score_disposition": "not_scored_no_fixed_zero",
        "recovery_policy": "resume_earliest_incomplete_frozen_member",
        "automatic_same_process_retry": False,
        "max_retryable_attempts_per_query": (
            PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
        ),
        "circuit_id": circuit_id,
        "circuit_breaker_threshold": PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
        "consecutive_retryable_failures": consecutive,
        "circuit_open": circuit_open,
        "captured_provider_response_count": captured_provider_response_count,
        "captured_input_tokens": captured_input_tokens,
        "captured_output_tokens": captured_output_tokens,
        "assistant_response_sha256": assistant_response_sha256,
        "assistant_receipt_sha256": assistant_receipt_sha256,
        "judge_result_sha256": judge_result_sha256,
        "validated_route_identity": (
            None
            if validated_route_identity is None
            else validated_route_identity.model_dump(mode="json")
        ),
        "exception_type": exception_type,
    }
    if route_call_evidence is not None:
        unsigned["route_call_evidence"] = route_call_evidence.model_dump(mode="json")
    if forfeited_reservation_sha256 is not None:
        unsigned["forfeited_reservation_sha256"] = forfeited_reservation_sha256
    if budget_forfeit_sha256 is not None:
        unsigned["budget_forfeit_sha256"] = budget_forfeit_sha256
    receipt = PortfolioAttemptReceipt.model_validate(
        {
            **unsigned,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        },
        strict=True,
    )
    directory = attempt_receipt_directory(shard_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{query_ordinal:04d}-attempt-{attempt_index:02d}.json"
    atomic_create_file(path, canonical_json_bytes(receipt.model_dump(mode="json")))
    return receipt, path


def portfolio_budget_authority_path(ledger_root: Path) -> Path:
    return ledger_root / "budget-authority.json"


def portfolio_budget_event_directory(ledger_root: Path) -> Path:
    return ledger_root / "events"


def _budget_ledger_session_key(ledger_root: Path) -> Path:
    if not isinstance(ledger_root, Path):
        raise TypeError("ledger_root must be a pathlib.Path")
    return ledger_root.absolute().resolve(strict=False)


def make_portfolio_budget_call_identity(
    *,
    matrix_run_id: str,
    shard_id: str,
    config: str,
    query_id: str,
    instance_sha256: str,
    request_sha256: str,
    stage: PortfolioProviderStage,
    attempt_index: int,
    call_index: int,
    wire_request_sha256: str | None = None,
) -> PortfolioBudgetCallIdentity:
    """Create the self-hashed matrix/shard/query/stage/attempt/call identity."""

    provider, model = (
        (
            PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER,
            PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL,
        )
        if stage == "final_judge"
        else (PORTFOLIO_QWEN_PROVIDER, PORTFOLIO_QWEN_MODEL)
    )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-call-identity",
        "matrix_run_id": matrix_run_id,
        "shard_id": shard_id,
        "config": config,
        "query_id": query_id,
        "instance_sha256": instance_sha256,
        "request_sha256": request_sha256,
        "wire_request_sha256": wire_request_sha256,
        "stage": stage,
        "attempt_index": attempt_index,
        "call_index": call_index,
        "provider": provider,
        "model": model,
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "identity_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    return PortfolioBudgetCallIdentity.model_validate_json(content, strict=True)


def initialize_portfolio_budget_ledger(
    ledger_root: Path,
    *,
    matrix_run_id: str,
    phase_cap_cny: Decimal,
    prior_observed_cost_cny: Decimal = Decimal("0"),
) -> tuple[PortfolioBudgetAuthority, Path]:
    """Create a frozen hard-cap authority without overwriting an existing one."""

    cap = _normalize_cny(phase_cap_cny, "phase_cap_cny", positive=True)
    prior = _normalize_cny(
        prior_observed_cost_cny,
        "prior_observed_cost_cny",
        positive=False,
    )
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-authority",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "matrix_run_id": matrix_run_id,
        "currency": PORTFOLIO_BUDGET_CURRENCY,
        "phase_cap_cny": _cny_json(cap),
        "prior_observed_cost_cny": _cny_json(prior),
        "accounting_policy": (
            "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
        ),
        "exception_policy": ("append_full_reserve_forfeit_without_captured_response"),
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "authority_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    authority = PortfolioBudgetAuthority.model_validate_json(content, strict=True)
    _ACTIVE_BUDGET_LEDGER_SESSIONS.pop(
        _budget_ledger_session_key(ledger_root),
        None,
    )
    ledger_root.mkdir(parents=True, exist_ok=True)
    path = portfolio_budget_authority_path(ledger_root)
    atomic_create_file(path, content)
    return authority, path


def load_portfolio_budget_authority(ledger_root: Path) -> PortfolioBudgetAuthority:
    path = portfolio_budget_authority_path(ledger_root)
    content = read_stable_regular_file(
        path,
        label="Portfolio budget authority",
        max_bytes=64 * 1024,
    )
    raw = parse_canonical_json(content, label="Portfolio budget authority")
    if not isinstance(raw, dict):
        raise ValueError("Portfolio budget authority must contain an object")
    return PortfolioBudgetAuthority.model_validate_json(content, strict=True)


def _load_portfolio_budget_event(
    path: Path,
) -> PortfolioBudgetEvent:
    content = read_stable_regular_file(
        path,
        label=f"Portfolio budget event {path.name}",
        max_bytes=128 * 1024,
    )
    raw = parse_canonical_json(
        content,
        label=f"Portfolio budget event {path.name}",
    )
    if not isinstance(raw, dict):
        raise ValueError("Portfolio budget event must contain an object")
    kind = raw.get("kind")
    if kind == "portfolio-budget-reservation":
        return PortfolioBudgetReservation.model_validate_json(content, strict=True)
    if kind == "portfolio-budget-settlement":
        return PortfolioBudgetSettlement.model_validate_json(content, strict=True)
    if kind == "portfolio-budget-forfeit":
        return PortfolioBudgetForfeit.model_validate_json(content, strict=True)
    raise ValueError(f"unknown Portfolio budget event kind: {kind!r}")


def load_portfolio_budget_ledger(ledger_root: Path) -> PortfolioBudgetLedgerState:
    """Recover and validate the full create-only ledger after any restart."""

    authority = load_portfolio_budget_authority(ledger_root)
    directory = portfolio_budget_event_directory(ledger_root)
    event_paths: list[Path] = []
    if directory.exists():
        for path in directory.iterdir():
            # ``atomic_create_file`` may expose a dot-prefixed temporary file to
            # another process before the immutable link is published.
            if path.name.startswith("."):
                continue
            if not path.is_file() or _BUDGET_EVENT_FILE_RE.fullmatch(path.name) is None:
                raise ValueError(f"invalid Portfolio budget ledger entry: {path}")
            event_paths.append(path)
    event_paths.sort(key=lambda item: item.name)

    reservations: list[PortfolioBudgetReservation] = []
    settlements: list[PortfolioBudgetSettlement] = []
    forfeits: list[PortfolioBudgetForfeit] = []
    reservation_by_sha: dict[str, PortfolioBudgetReservation] = {}
    reservation_by_identity: dict[str, PortfolioBudgetReservation] = {}
    settlement_by_reservation: dict[str, PortfolioBudgetSettlement] = {}
    forfeit_by_reservation: dict[str, PortfolioBudgetForfeit] = {}
    accountable = authority.prior_observed_cost_cny
    last_event_sha256: str | None = None

    for expected_index, path in enumerate(event_paths, start=1):
        match = _BUDGET_EVENT_FILE_RE.fullmatch(path.name)
        assert match is not None
        if int(match.group("ordinal")) != expected_index:
            raise ValueError("Portfolio budget ledger event order contains a gap")
        event = _load_portfolio_budget_event(path)
        if (
            event.ledger_event_index != expected_index
            or event.authority_sha256 != authority.authority_sha256
            or event.policy_version != authority.policy_version
            or event.previous_event_sha256 != last_event_sha256
        ):
            raise ValueError("Portfolio budget ledger event chain mismatch")

        if isinstance(event, PortfolioBudgetReservation):
            identity = event.identity
            if identity.matrix_run_id != authority.matrix_run_id:
                raise ValueError("Portfolio reservation differs from budget authority")
            if identity.identity_sha256 in reservation_by_identity:
                raise ValueError(
                    "duplicate Portfolio provider-call reservation identity"
                )
            if event.reservation_sha256 in reservation_by_sha:
                raise ValueError("duplicate Portfolio budget reservation hash")
            reservations.append(event)
            reservation_by_sha[event.reservation_sha256] = event
            reservation_by_identity[identity.identity_sha256] = event
            accountable += event.reserved_cost_cny
            last_event_sha256 = event.reservation_sha256
        elif isinstance(event, PortfolioBudgetSettlement):
            reservation = reservation_by_sha.get(event.reservation_sha256)
            if reservation is None:
                raise ValueError("Portfolio settlement precedes its reservation")
            if (
                event.reservation_sha256 in settlement_by_reservation
                or event.reservation_sha256 in forfeit_by_reservation
            ):
                raise ValueError("Portfolio reservation has multiple terminal outcomes")
            if event.identity_sha256 != reservation.identity.identity_sha256:
                raise ValueError("Portfolio settlement call identity drifted")
            if (
                event.actual_input_tokens
                > reservation.pricing_profile.provider_max_input_tokens
                or event.actual_output_tokens > reservation.max_output_tokens
            ):
                raise ValueError("Portfolio settlement usage exceeds its reservation")
            expected_cost = calculate_portfolio_settled_usage_cost_cny(
                reservation.pricing_profile,
                input_tokens=event.actual_input_tokens,
                output_tokens=event.actual_output_tokens,
            )
            if event.actual_cost_cny != expected_cost:
                raise ValueError("Portfolio settlement cost differs from actual usage")
            settlements.append(event)
            settlement_by_reservation[event.reservation_sha256] = event
            accountable -= reservation.reserved_cost_cny
            accountable += event.actual_cost_cny
            last_event_sha256 = event.settlement_sha256
        else:
            if authority.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION:
                raise ValueError(
                    "legacy Portfolio budget ledger cannot contain forfeits"
                )
            reservation = reservation_by_sha.get(event.reservation_sha256)
            if reservation is None:
                raise ValueError("Portfolio forfeit precedes its reservation")
            if (
                event.reservation_sha256 in settlement_by_reservation
                or event.reservation_sha256 in forfeit_by_reservation
            ):
                raise ValueError("Portfolio reservation has multiple terminal outcomes")
            if event.identity_sha256 != reservation.identity.identity_sha256:
                raise ValueError("Portfolio forfeit call identity drifted")
            if event.forfeited_cost_cny != reservation.reserved_cost_cny:
                raise ValueError(
                    "Portfolio forfeit differs from the complete reservation"
                )
            forfeits.append(event)
            forfeit_by_reservation[event.reservation_sha256] = event
            accountable -= reservation.reserved_cost_cny
            accountable += event.forfeited_cost_cny
            last_event_sha256 = event.forfeit_sha256

        if accountable > authority.phase_cap_cny:
            raise ValueError("Portfolio budget ledger exceeds the frozen phase cap")

    unresolved = tuple(
        reservation
        for reservation in reservations
        if reservation.reservation_sha256 not in settlement_by_reservation
        and reservation.reservation_sha256 not in forfeit_by_reservation
    )
    settled_cost = sum(
        (item.actual_cost_cny for item in settlements),
        start=Decimal("0.000000000000"),
    )
    forfeited_cost = sum(
        (item.forfeited_cost_cny for item in forfeits),
        start=Decimal("0.000000000000"),
    )
    unresolved_cost = sum(
        (item.reserved_cost_cny for item in unresolved),
        start=Decimal("0.000000000000"),
    )
    expected_accountable = (
        authority.prior_observed_cost_cny
        + settled_cost
        + forfeited_cost
        + unresolved_cost
    )
    if accountable != expected_accountable:
        raise ValueError("Portfolio budget ledger accounting reconstruction drifted")
    return PortfolioBudgetLedgerState(
        authority=authority,
        reservations=tuple(reservations),
        settlements=tuple(settlements),
        forfeits=tuple(forfeits),
        unresolved_reservations=unresolved,
        settled_actual_cost_cny=settled_cost,
        forfeited_reserved_cost_cny=forfeited_cost,
        unresolved_reserved_cost_cny=unresolved_cost,
        accountable_cost_cny=accountable,
        remaining_cost_cny=authority.phase_cap_cny - accountable,
        last_event_index=len(event_paths),
        last_event_sha256=last_event_sha256,
    )


def open_portfolio_budget_ledger_session(
    ledger_root: Path,
) -> PortfolioBudgetLedgerSession:
    """Fully validate a ledger once and activate incremental appends in-process."""

    key = _budget_ledger_session_key(ledger_root)
    session = PortfolioBudgetLedgerSession(
        ledger_root=key,
        _state=load_portfolio_budget_ledger(key),
    )
    _ACTIVE_BUDGET_LEDGER_SESSIONS[key] = session
    return session


def _portfolio_budget_append_state(
    ledger_root: Path,
) -> tuple[PortfolioBudgetLedgerState, PortfolioBudgetLedgerSession | None]:
    key = _budget_ledger_session_key(ledger_root)
    session = _ACTIVE_BUDGET_LEDGER_SESSIONS.get(key)
    if session is None:
        return load_portfolio_budget_ledger(key), None
    return session.require_current(), session


def _require_active_budget_policy_for_append(
    state: PortfolioBudgetLedgerState,
) -> None:
    if state.authority.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION:
        raise PortfolioBudgetError(
            "legacy Portfolio budget ledgers are read-only under the active policy"
        )


def _build_budget_reservation(
    *,
    state: PortfolioBudgetLedgerState,
    identity: PortfolioBudgetCallIdentity,
    pricing_profile: PortfolioProviderPricingProfile,
    reserved_cost: Decimal,
) -> PortfolioBudgetReservation:
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-reservation",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "authority_sha256": state.authority.authority_sha256,
        "ledger_event_index": state.last_event_index + 1,
        "previous_event_sha256": state.last_event_sha256,
        "identity": identity.model_dump(mode="json"),
        "pricing_profile": pricing_profile.model_dump(mode="json"),
        "reserved_cost_cny": _cny_json(reserved_cost),
        "disposition": "reserved_before_provider_call",
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "reservation_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    return PortfolioBudgetReservation.model_validate_json(content, strict=True)


def reserve_portfolio_provider_call(
    ledger_root: Path,
    *,
    identity: PortfolioBudgetCallIdentity,
    final_judge_max_output_tokens: int | None = None,
) -> tuple[PortfolioBudgetReservation, Path]:
    """Reserve the maximum call cost before invoking a provider.

    Every append uses the next fixed ordinal filename.  Concurrent writers
    therefore contend on the same create-only path, reload, and repeat the cap
    check against the winner rather than racing past the cap.
    """

    if type(identity) is not PortfolioBudgetCallIdentity:
        raise TypeError("identity must be a PortfolioBudgetCallIdentity")
    pricing_profile = make_portfolio_provider_pricing_profile(
        identity,
        final_judge_max_output_tokens=final_judge_max_output_tokens,
    )
    reserved_cost = calculate_portfolio_usage_cost_cny(
        input_tokens=pricing_profile.provider_max_input_tokens,
        output_tokens=pricing_profile.max_output_tokens,
        input_cny_per_million_tokens=(
            pricing_profile.reserve_input_cny_per_million_tokens
        ),
        output_cny_per_million_tokens=(
            pricing_profile.reserve_output_cny_per_million_tokens
        ),
    )
    if reserved_cost == 0:
        raise ValueError("provider reservation rounds to zero CNY")

    for _ in range(_BUDGET_APPEND_ATTEMPTS):
        state, session = _portfolio_budget_append_state(ledger_root)
        _require_active_budget_policy_for_append(state)
        if identity.matrix_run_id != state.authority.matrix_run_id:
            raise PortfolioBudgetError(
                "provider-call identity differs from the frozen budget authority"
            )
        if any(
            item.identity.identity_sha256 == identity.identity_sha256
            for item in state.reservations
        ):
            raise PortfolioBudgetDuplicateCallError(
                "provider-call identity already has an immutable reservation"
            )
        projected = state.accountable_cost_cny + reserved_cost
        if projected > state.authority.phase_cap_cny:
            raise PortfolioBudgetExceededError(
                "provider call not authorized: accountable CNY "
                f"{_cny_json(state.accountable_cost_cny)} + reserve CNY "
                f"{_cny_json(reserved_cost)} exceeds cap CNY "
                f"{_cny_json(state.authority.phase_cap_cny)}"
            )
        reservation = _build_budget_reservation(
            state=state,
            identity=identity,
            pricing_profile=pricing_profile,
            reserved_cost=reserved_cost,
        )
        directory = portfolio_budget_event_directory(ledger_root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{reservation.ledger_event_index:08d}.json"
        try:
            atomic_create_file(
                path,
                canonical_json_bytes(reservation.model_dump(mode="json")),
            )
        except FileExistsError:
            if session is not None:
                session.reload()
            continue
        if session is not None:
            session.accept_reservation(reservation)
        return reservation, path
    raise PortfolioBudgetLedgerBusyError(
        "unable to append Portfolio budget reservation after concurrent writes"
    )


def _build_budget_settlement(
    *,
    state: PortfolioBudgetLedgerState,
    reservation: PortfolioBudgetReservation,
    actual_input_tokens: int,
    actual_output_tokens: int,
    actual_cost: Decimal,
    provider_request_id: str,
    response_sha256: str,
) -> PortfolioBudgetSettlement:
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-settlement",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "authority_sha256": state.authority.authority_sha256,
        "ledger_event_index": state.last_event_index + 1,
        "previous_event_sha256": state.last_event_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "identity_sha256": reservation.identity.identity_sha256,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
        "actual_cost_cny": _cny_json(actual_cost),
        "provider_request_id": provider_request_id,
        "response_sha256": response_sha256,
        "outcome": "success",
        "disposition": "settled_from_actual_usage",
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "settlement_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    return PortfolioBudgetSettlement.model_validate_json(content, strict=True)


def settle_portfolio_provider_call_success(
    ledger_root: Path,
    *,
    reservation_sha256: str,
    actual_input_tokens: int,
    actual_output_tokens: int,
    provider_request_id: str,
    response_sha256: str,
) -> tuple[PortfolioBudgetSettlement, Path]:
    """Settle a successful call from captured usage.

    A caller must invoke this only after receiving a valid provider response.
    Calls ending without one use :func:`forfeit_portfolio_provider_call`, which
    closes the reservation while charging its complete maximum cost.
    """

    if re.fullmatch(r"[0-9a-f]{64}", reservation_sha256) is None:
        raise ValueError("reservation_sha256 must be a lowercase SHA-256")
    _validate_token_count(actual_input_tokens, "actual_input_tokens")
    _validate_token_count(actual_output_tokens, "actual_output_tokens")
    _validate_identity_text(provider_request_id, "provider_request_id")
    if re.fullmatch(r"[0-9a-f]{64}", response_sha256) is None:
        raise ValueError("response_sha256 must be a lowercase SHA-256")
    if actual_input_tokens == 0 and actual_output_tokens == 0:
        raise PortfolioBudgetSettlementError(
            "successful provider settlement cannot have zero usage"
        )

    for _ in range(_BUDGET_APPEND_ATTEMPTS):
        state, session = _portfolio_budget_append_state(ledger_root)
        _require_active_budget_policy_for_append(state)
        reservation = next(
            (
                item
                for item in state.reservations
                if item.reservation_sha256 == reservation_sha256
            ),
            None,
        )
        if reservation is None:
            raise PortfolioBudgetSettlementError(
                "success settlement does not bind an existing reservation"
            )
        if any(
            item.reservation_sha256 == reservation_sha256 for item in state.settlements
        ) or any(
            item.reservation_sha256 == reservation_sha256 for item in state.forfeits
        ):
            raise PortfolioBudgetSettlementError(
                "reservation already has a create-only terminal outcome"
            )
        if (
            actual_input_tokens > reservation.pricing_profile.provider_max_input_tokens
            or actual_output_tokens > reservation.max_output_tokens
        ):
            raise PortfolioBudgetSettlementError(
                "captured success usage exceeds the pre-provider reservation"
            )
        actual_cost = calculate_portfolio_settled_usage_cost_cny(
            reservation.pricing_profile,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
        )
        if actual_cost > reservation.reserved_cost_cny:
            raise PortfolioBudgetSettlementError(
                "captured success cost exceeds the pre-provider reservation"
            )
        settlement = _build_budget_settlement(
            state=state,
            reservation=reservation,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            actual_cost=actual_cost,
            provider_request_id=provider_request_id,
            response_sha256=response_sha256,
        )
        directory = portfolio_budget_event_directory(ledger_root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{settlement.ledger_event_index:08d}.json"
        try:
            atomic_create_file(
                path,
                canonical_json_bytes(settlement.model_dump(mode="json")),
            )
        except FileExistsError:
            if session is not None:
                session.reload()
            continue
        if session is not None:
            session.accept_settlement(settlement)
        return settlement, path
    raise PortfolioBudgetLedgerBusyError(
        "unable to append Portfolio budget settlement after concurrent writes"
    )


def _build_budget_forfeit(
    *,
    state: PortfolioBudgetLedgerState,
    reservation: PortfolioBudgetReservation,
    reason: PortfolioBudgetForfeitReason,
) -> PortfolioBudgetForfeit:
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-forfeit",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "authority_sha256": state.authority.authority_sha256,
        "ledger_event_index": state.last_event_index + 1,
        "previous_event_sha256": state.last_event_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "identity_sha256": reservation.identity.identity_sha256,
        "forfeited_cost_cny": _cny_json(reservation.reserved_cost_cny),
        "captured_provider_response": False,
        "actual_usage_status": "unknown",
        "reason": reason,
        "disposition": "charged_full_reserve",
    }
    content = canonical_json_bytes(
        {
            **unsigned,
            "forfeit_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    return PortfolioBudgetForfeit.model_validate_json(content, strict=True)


def forfeit_portfolio_provider_call(
    ledger_root: Path,
    *,
    reservation_sha256: str,
    reason: PortfolioBudgetForfeitReason,
) -> tuple[PortfolioBudgetForfeit, Path]:
    """Close a response-less provider call while retaining its full reserve.

    The append is create-only and does not release any accountable budget.  It
    is valid only for active-v2 ledgers and is mutually exclusive with a success
    settlement for the same reservation.
    """

    if re.fullmatch(r"[0-9a-f]{64}", reservation_sha256) is None:
        raise ValueError("reservation_sha256 must be a lowercase SHA-256")
    if type(reason) is not str or reason not in {
        "provider_call_ended_without_captured_response",
        "orphan_recovered_after_owner_exit",
    }:
        raise ValueError("reason must be a frozen Portfolio forfeit reason")

    for _ in range(_BUDGET_APPEND_ATTEMPTS):
        state, session = _portfolio_budget_append_state(ledger_root)
        _require_active_budget_policy_for_append(state)
        reservation = next(
            (
                item
                for item in state.reservations
                if item.reservation_sha256 == reservation_sha256
            ),
            None,
        )
        if reservation is None:
            raise PortfolioBudgetForfeitError(
                "forfeit does not bind an existing reservation"
            )
        if any(
            item.reservation_sha256 == reservation_sha256 for item in state.settlements
        ) or any(
            item.reservation_sha256 == reservation_sha256 for item in state.forfeits
        ):
            raise PortfolioBudgetForfeitError(
                "reservation already has a create-only terminal outcome"
            )
        forfeit = _build_budget_forfeit(
            state=state,
            reservation=reservation,
            reason=reason,
        )
        directory = portfolio_budget_event_directory(ledger_root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{forfeit.ledger_event_index:08d}.json"
        try:
            atomic_create_file(
                path,
                canonical_json_bytes(forfeit.model_dump(mode="json")),
            )
        except FileExistsError:
            if session is not None:
                session.reload()
            continue
        if session is not None:
            session.accept_forfeit(forfeit)
        return forfeit, path
    raise PortfolioBudgetLedgerBusyError(
        "unable to append Portfolio budget forfeit after concurrent writes"
    )


__all__ = [
    "PORTFOLIO_BUDGET_CURRENCY",
    "PORTFOLIO_BUDGET_CNY_QUANTUM",
    "PORTFOLIO_BUDGET_DECIMAL_PLACES",
    "PORTFOLIO_BUDGET_POLICY_VERSION",
    "PORTFOLIO_BUDGET_POLICY_VERSION_V1",
    "PORTFOLIO_BUDGET_POLICY_VERSION_V2",
    "PORTFOLIO_BUDGET_POLICY_VERSION_V3",
    "PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL",
    "PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER",
    "PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS",
    "PORTFOLIO_GEMINI_MODEL",
    "PORTFOLIO_GEMINI_PROVIDER",
    "PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS",
    "PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION",
    "PORTFOLIO_KIMI_MODEL",
    "PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION",
    "PORTFOLIO_KIMI_PROVIDER",
    "PORTFOLIO_KIMI_PROVIDER_MAX_INPUT_TOKENS",
    "PORTFOLIO_KIMI_PROVIDER_MAX_OUTPUT_TOKENS",
    "PORTFOLIO_QWEN_ACTION_MAX_OUTPUT_TOKENS",
    "PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS",
    "PORTFOLIO_QWEN_MODEL",
    "PORTFOLIO_QWEN_PROVIDER",
    "PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS",
    "PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION",
    "PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION",
    "PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS",
    "PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS",
    "PORTFOLIO_QWEN_STAGE_MAX_INPUT_TOKENS",
    "PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD",
    "PORTFOLIO_FAILURE_POLICY_VERSION",
    "PORTFOLIO_FAILURE_POLICY_VERSION_V1",
    "PORTFOLIO_FAILURE_POLICY_VERSION_V2",
    "PORTFOLIO_FAILURE_POLICY_VERSION_V3",
    "PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY",
    "PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE",
    "PortfolioAttemptLimitError",
    "PortfolioAttemptReceipt",
    "PortfolioBudgetAuthority",
    "PortfolioBudgetCallIdentity",
    "PortfolioBudgetDuplicateCallError",
    "PortfolioBudgetError",
    "PortfolioBudgetExceededError",
    "PortfolioBudgetForfeit",
    "PortfolioBudgetForfeitError",
    "PortfolioBudgetForfeitReason",
    "PortfolioBudgetLedgerBusyError",
    "PortfolioBudgetLedgerSession",
    "PortfolioBudgetLedgerState",
    "PortfolioBudgetOrphanedCallError",
    "PortfolioBudgetReservation",
    "PortfolioBudgetSettlement",
    "PortfolioBudgetSettlementError",
    "PortfolioPricingProfileId",
    "PortfolioProviderStage",
    "PortfolioProviderPricingProfile",
    "ProviderPreResponseCircuitBreaker",
    "ValidatedRouteIdentity",
    "attempt_receipt_directory",
    "calculate_portfolio_usage_cost_cny",
    "calculate_portfolio_settled_usage_cost_cny",
    "create_retryable_attempt_receipt",
    "forfeit_portfolio_provider_call",
    "initialize_portfolio_budget_ledger",
    "load_portfolio_budget_authority",
    "load_portfolio_budget_ledger",
    "load_query_attempt_receipts",
    "make_portfolio_budget_call_identity",
    "make_portfolio_provider_pricing_profile",
    "open_portfolio_budget_ledger_session",
    "portfolio_attempt_provider_call_count",
    "portfolio_budget_authority_path",
    "portfolio_budget_event_directory",
    "portfolio_budget_settled_cost_cny",
    "require_retryable_attempt_available",
    "reserve_portfolio_provider_call",
    "settle_portfolio_provider_call_success",
]
