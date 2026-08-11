"""Deterministic GCS-v2 replay and acceptance gates for the Core S1 Bank.

This module is deliberately offline.  It consumes complete, already-scored
Assistant populations and never launches an Assistant, Feedback model, Judge,
or any other provider call.  Population-integrity violations raise; ordinary
coverage failures produce an unavailable gate and therefore a byte-exact
rollback.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.evaluation.portfolio_gcs import (
    GCS_BOOTSTRAP_POLICY_VERSION,
    GCS_BOOTSTRAP_REPLICATES,
    GCS_BOOTSTRAP_ROOT_SEED,
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
    SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
    SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
    SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
    SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN,
    GCSConfigSummaryV2,
    GCSQueryScoreV2,
    PortfolioGCSError,
    build_gcs_population_v2,
    summarize_gcs_v2,
)
from skillchain.evaluation.assistant_runs import (
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantRunConfig,
)
from skillchain.schemas import Query
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
S1GCSGatePhase = Literal["replay", "body_gate"]
S1GCSGateStatus = Literal["passed", "failed", "unavailable"]
S1BankDecision = Literal["accepted", "rolled_back"]

S1_GCS_GATE_POLICY_VERSION = "portfolio-s1-gcs-replay-body-gate-v1"
S1_REPLAY_SCOPE = "s1-opt-replay200"
S1_BODY_GATE_SCOPE = "s1-body-gate75"
S1_REPLAY_QUERY_COUNT = 200
S1_BODY_GATE_QUERY_COUNT = 75
S1_REPLAY_MACRO_DELTA_PP_MIN = 0.0
S1_REPLAY_HARD_ERROR_DELTA_PP_MAX = 1.0
S1_REPLAY_PER_CAPABILITY_DELTA_PP_MIN = -3.0
S1_CONTRACT_REGRESSION_REASON_CODES = (
    "fallback_contract_failed",
    "output_section_invalid",
    "tool_contract_failed",
)
S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION = (
    "portfolio-s1-round2-response-contract-diagnostics-v1"
)

_RESPONSE_CONTRACT_REASON_MAP = {
    "response_fallback_marker_missing": "fallback_contract_failed",
    "response_control_token": "output_section_invalid",
    "response_empty": "output_section_invalid",
    "response_repair_new_material_atom": "output_section_invalid",
    "response_repair_wire_invalid": "output_section_invalid",
    "response_section_invalid": "output_section_invalid",
    "response_detector_only_final": "tool_contract_failed",
    "response_tool_sequence_invalid": "tool_contract_failed",
}

_PHASE_QUERY_COUNT: dict[S1GCSGatePhase, int] = {
    "replay": S1_REPLAY_QUERY_COUNT,
    "body_gate": S1_BODY_GATE_QUERY_COUNT,
}
_PHASE_SCOPE: dict[S1GCSGatePhase, str] = {
    "replay": S1_REPLAY_SCOPE,
    "body_gate": S1_BODY_GATE_SCOPE,
}
_PHASE_SPLIT: dict[S1GCSGatePhase, str] = {
    "replay": "opt_pool",
    "body_gate": "val",
}
_PHASE_RULE: dict[S1GCSGatePhase, str] = {
    "replay": "macro-nonnegative-hard-plus1-eachcap-minus3-coverage-v1",
    "body_gate": "gcs-system-gain-thresholds-v1",
}

_POLICY_PAYLOAD = {
    "policy_version": S1_GCS_GATE_POLICY_VERSION,
    "gcs_policy_version": GCS_V2_POLICY_VERSION,
    "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
    "bootstrap": {
        "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
        "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
        "replicates": GCS_BOOTSTRAP_REPLICATES,
        "unit": "query_connected_component",
    },
    "baseline_config": "llm_static",
    "candidate_config": "s1",
    "replay": {
        "scope": S1_REPLAY_SCOPE,
        "split": "opt_pool",
        "query_count": S1_REPLAY_QUERY_COUNT,
        "macro_delta_pp_min": S1_REPLAY_MACRO_DELTA_PP_MIN,
        "hard_error_delta_pp_max": S1_REPLAY_HARD_ERROR_DELTA_PP_MAX,
        "per_capability_delta_pp_min": S1_REPLAY_PER_CAPABILITY_DELTA_PP_MIN,
        "purpose": "overfit_screen_not_acceptance",
    },
    "body_gate": {
        "scope": S1_BODY_GATE_SCOPE,
        "split": "val",
        "query_count": S1_BODY_GATE_QUERY_COUNT,
        "macro_delta_pp_min": SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
        "macro_ci95_low_pp_min": SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
        "hard_error_delta_pp_max": SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
        "per_capability_delta_pp_min": SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN,
        "purpose": "single_accept_or_rollback",
    },
    "coverage": "six_capabilities_and_all_oracles_available",
    "integrity": "rectangular_typed_gcs_v2_population_or_raise",
    "bank_disposition": "accepted_candidate_bytes_else_parent_bytes",
    "forbidden": ["pairwise_judge", "legacy_final_judge", "test_frozen"],
    "provider_model_call_count": 0,
}
S1_GCS_GATE_POLICY_SHA256 = sha256_bytes(canonical_json_bytes(_POLICY_PAYLOAD))
_EXPECTED_S1_GCS_GATE_POLICY_SHA256 = (
    "9d0dc6d291203dc4d2eefff0f6abea65c4355964eb7696b14b1543034a556e5d"
)
if S1_GCS_GATE_POLICY_SHA256 != _EXPECTED_S1_GCS_GATE_POLICY_SHA256:
    raise RuntimeError(
        "tracked S1 GCS gate policy changed without an identity revision"
    )

S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION = (
    "portfolio-s1-round2-development-screen-v2"
)
S1_ROUND2_DEVELOPMENT_QUERY_COUNT = 200
S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS = (
    "r2-core-0669",
    "r2-core-0670",
    "r2-core-0695",
    "r2-core-0796",
)
S1_ROUND2_BODY_GATE_POLICY_VERSION = "portfolio-s1-round2-body-gate-v2"
_ROUND2_BODY_GATE_POLICY_PAYLOAD = {
    "policy_version": S1_ROUND2_BODY_GATE_POLICY_VERSION,
    "development_screen": {
        "policy_version": S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION,
        "query_count": S1_ROUND2_DEVELOPMENT_QUERY_COUNT,
        "required_regression_query_ids": list(S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS),
        "bindings": [
            "query_ids",
            "population_mapping",
            "baseline_scores",
            "raw_candidate_scores",
            "checkpoint_response_contract_diagnostics",
            "parent_bank",
            "raw_candidate_bank",
            "source_evidence",
        ],
    },
    "development_input": (
        "passed-composite-linked-to-screened-bank-receipt-and-runtime-v2"
    ),
    "candidate_freeze": {
        "parent_and_candidate": "artifact-and-canonical-file-sha256",
        "runtime": "runtime-lock-artifact-and-file-sha256",
        "screen": "artifact-and-canonical-file-sha256",
        "screened_bank_receipt": "artifact-and-canonical-file-sha256",
        "gate_policy_version": S1_GCS_GATE_POLICY_VERSION,
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
    },
    "body_gate": {
        "phase": "body_gate",
        "query_count": S1_BODY_GATE_QUERY_COUNT,
        "split": "val",
        "threshold_policy": S1_GCS_GATE_POLICY_VERSION,
        "freshness": "disjoint-from-development-query-ids",
        "paired_static_success_to_candidate_failure_max": 0,
        "new_contract_reason_occurrences_max": 0,
        "contract_reason_source": (
            "gcs-score-union-checkpoint-response-contract-diagnostics"
        ),
    },
    "disposition": "passed-candidate-bytes-else-parent-bytes",
    "historical_replay_body_api": "unchanged",
    "provider_model_call_count": 0,
}
S1_ROUND2_BODY_GATE_POLICY_SHA256 = sha256_bytes(
    canonical_json_bytes(_ROUND2_BODY_GATE_POLICY_PAYLOAD)
)
_EXPECTED_S1_ROUND2_BODY_GATE_POLICY_SHA256 = (
    "cb8b7265377336736ed08ad62e730060496e8c482a9a3452a65469d7b3014874"
)
if S1_ROUND2_BODY_GATE_POLICY_SHA256 != _EXPECTED_S1_ROUND2_BODY_GATE_POLICY_SHA256:
    raise RuntimeError("tracked Round 2 body gate policy changed without a revision")


class S1GCSGateError(ValueError):
    """The S1 gate evidence, population, or Bank disposition is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _hash_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256_bytes(canonical_json_bytes(value))  # type: ignore[arg-type]


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field_name}))


class S1Round2ResponseContractDiagnostic(_StrictFrozenModel):
    """Checkpoint-bound response-contract outcome used only by Round 2 gates."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-round2-response-contract-diagnostic"] = (
        "portfolio-s1-round2-response-contract-diagnostic"
    )
    policy_version: Literal[
        S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION
    ] = S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION
    query_id: str
    config: AssistantRunConfig
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    assistant_response_sha256: Sha256
    assistant_receipt_sha256: Sha256
    response_error_code: str | None
    repair_status: Literal["not_attempted", "passed", "failed"]
    repair_attempt_count: Literal[0, 1]
    repair_receipt_sha256: Sha256 | None
    repair_call_index: int | None = Field(default=None, ge=1)
    repair_input_tokens: int = Field(ge=0)
    repair_output_tokens: int = Field(ge=0)
    initial_reason_codes: tuple[str, ...]
    final_reason_codes: tuple[str, ...]
    mapped_contract_reason_codes: tuple[str, ...]
    mapping_disposition: Literal[
        "not_applicable",
        "final_repair_reasons_mapped",
        "unclassified_terminal_mapped_fail_closed",
    ]
    diagnostic_sha256: Sha256

    @field_validator(
        "initial_reason_codes",
        "final_reason_codes",
        "mapped_contract_reason_codes",
        mode="before",
    )
    @classmethod
    def coerce_reason_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_diagnostic(self) -> Self:
        if not self.query_id or self.query_id != self.query_id.strip():
            raise ValueError("Round 2 contract diagnostic query id is invalid")
        if self.config not in {"llm_static", "s1"}:
            raise ValueError("Round 2 contract diagnostic config is invalid")
        for reasons in (
            self.initial_reason_codes,
            self.final_reason_codes,
            self.mapped_contract_reason_codes,
        ):
            if reasons != tuple(sorted(set(reasons))):
                raise ValueError(
                    "Round 2 contract diagnostic reasons must be sorted and unique"
                )
        if any(
            reason not in S1_CONTRACT_REGRESSION_REASON_CODES
            for reason in self.mapped_contract_reason_codes
        ):
            raise ValueError("Round 2 contract diagnostic mapping is outside policy")
        attempted = self.repair_attempt_count == 1
        repair_fields_present = (
            self.repair_receipt_sha256 is not None
            and self.repair_call_index is not None
        )
        if attempted != repair_fields_present:
            raise ValueError("Round 2 repair identity is incomplete")
        if not attempted and (
            self.repair_status != "not_attempted"
            or self.repair_input_tokens
            or self.repair_output_tokens
            or self.initial_reason_codes
            or self.final_reason_codes
        ):
            raise ValueError("Round 2 non-repair diagnostic carries repair evidence")
        if attempted and self.repair_status == "not_attempted":
            raise ValueError("Round 2 attempted repair lacks a terminal status")
        if self.repair_status == "passed" and (
            self.response_error_code is not None
            or self.final_reason_codes
            or self.mapped_contract_reason_codes
            or self.mapping_disposition != "not_applicable"
        ):
            raise ValueError("Round 2 passed repair has inconsistent final evidence")
        if self.repair_status == "failed" and (
            self.response_error_code != "response_contract_error"
            or not self.final_reason_codes
            or not self.mapped_contract_reason_codes
            or self.mapping_disposition != "final_repair_reasons_mapped"
        ):
            raise ValueError("Round 2 failed repair has inconsistent final evidence")
        if self.repair_status == "not_attempted":
            terminal = self.response_error_code == "response_contract_error"
            if terminal != bool(self.mapped_contract_reason_codes):
                raise ValueError(
                    "Round 2 unattempted contract error is not mapped fail-closed"
                )
            expected_disposition = (
                "unclassified_terminal_mapped_fail_closed"
                if terminal
                else "not_applicable"
            )
            if self.mapping_disposition != expected_disposition:
                raise ValueError("Round 2 unattempted mapping disposition drifted")
        if self.diagnostic_sha256 != _self_hash(self, "diagnostic_sha256"):
            raise ValueError("Round 2 contract diagnostic self hash mismatch")
        return self


class S1Round2ResponseContractDiagnostics(_StrictFrozenModel):
    """Exact paired Static/S1 contract diagnostics for one scored population."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal[
        "portfolio-s1-round2-response-contract-diagnostics"
    ] = "portfolio-s1-round2-response-contract-diagnostics"
    policy_version: Literal[
        S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION
    ] = S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION
    query_count: int = Field(gt=0)
    query_ids: tuple[str, ...]
    rows: tuple[S1Round2ResponseContractDiagnostic, ...]
    diagnostics_sha256: Sha256

    @field_validator("query_ids", "rows", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_diagnostics(self) -> Self:
        if (
            self.query_ids != tuple(sorted(set(self.query_ids)))
            or len(self.query_ids) != self.query_count
        ):
            raise ValueError("Round 2 contract diagnostic query identity is invalid")
        expected_keys = tuple(
            (query_id, config)
            for query_id in self.query_ids
            for config in ("llm_static", "s1")
        )
        if tuple((row.query_id, row.config) for row in self.rows) != expected_keys:
            raise ValueError("Round 2 contract diagnostics are not exact and paired")
        if self.diagnostics_sha256 != _self_hash(self, "diagnostics_sha256"):
            raise ValueError("Round 2 contract diagnostics self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _mapped_response_contract_reasons(
    *, response_error_code: str | None, final_reason_codes: Sequence[str]
) -> tuple[tuple[str, ...], str]:
    mapped = tuple(
        sorted(
            {
                _RESPONSE_CONTRACT_REASON_MAP[reason]
                for reason in final_reason_codes
                if reason in _RESPONSE_CONTRACT_REASON_MAP
            }
        )
    )
    if final_reason_codes:
        # An unknown forward reason must not erase a terminal contract failure.
        if response_error_code == "response_contract_error" and not mapped:
            mapped = ("output_section_invalid",)
        return mapped, "final_repair_reasons_mapped"
    if response_error_code == "response_contract_error":
        return (
            S1_CONTRACT_REGRESSION_REASON_CODES,
            "unclassified_terminal_mapped_fail_closed",
        )
    return (), "not_applicable"


def make_s1_round2_response_contract_diagnostic(
    *,
    query_id: str,
    config: AssistantRunConfig,
    checkpoint_file_sha256: str,
    checkpoint_row_sha256: str,
    response: AssistantBackendResponse,
    receipt: AssistantExecutionReceipt,
) -> S1Round2ResponseContractDiagnostic:
    """Project one actual checkpoint receipt into a fail-closed gate diagnostic."""

    verified_response = AssistantBackendResponse.model_validate(
        response.model_dump(mode="python"), strict=True
    )
    verified_receipt = AssistantExecutionReceipt.model_validate(
        receipt.model_dump(mode="python"), strict=True
    )
    if verified_receipt.response_sha256 != sha256_bytes(
        canonical_json_bytes(verified_response.model_dump(mode="json"))
    ):
        raise S1GCSGateError("Round 2 diagnostic response differs from its receipt")
    repair = verified_receipt.response_contract_repair
    final_reasons = () if repair is None else repair.final_reason_codes
    mapped, disposition = _mapped_response_contract_reasons(
        response_error_code=verified_response.error_code,
        final_reason_codes=final_reasons,
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-round2-response-contract-diagnostic",
        "policy_version": S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION,
        "query_id": query_id,
        "config": config,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_row_sha256": checkpoint_row_sha256,
        "assistant_response_sha256": verified_receipt.response_sha256,
        "assistant_receipt_sha256": verified_receipt.receipt_sha256,
        "response_error_code": verified_response.error_code,
        "repair_status": "not_attempted" if repair is None else repair.status,
        "repair_attempt_count": 0 if repair is None else repair.attempt_count,
        "repair_receipt_sha256": None if repair is None else repair.receipt_sha256,
        "repair_call_index": None if repair is None else repair.repair_call_index,
        "repair_input_tokens": (
            0 if repair is None else repair.repair_call_usage.input_tokens
        ),
        "repair_output_tokens": (
            0 if repair is None else repair.repair_call_usage.output_tokens
        ),
        "initial_reason_codes": (
            [] if repair is None else list(repair.initial_reason_codes)
        ),
        "final_reason_codes": list(final_reasons),
        "mapped_contract_reason_codes": list(mapped),
        "mapping_disposition": disposition,
    }
    try:
        return S1Round2ResponseContractDiagnostic.model_validate(
            {**payload, "diagnostic_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("Round 2 response contract diagnostic is invalid") from error


def make_s1_round2_response_contract_diagnostics(
    *,
    query_ids: Sequence[str],
    rows: Sequence[S1Round2ResponseContractDiagnostic],
) -> S1Round2ResponseContractDiagnostics:
    ordered_ids = tuple(sorted(set(query_ids)))
    if len(ordered_ids) != len(query_ids):
        raise S1GCSGateError("Round 2 contract diagnostic query ids are duplicated")
    order = {"llm_static": 0, "s1": 1}
    try:
        ordered_rows = tuple(
            sorted(rows, key=lambda row: (row.query_id, order[row.config]))
        )
    except KeyError as error:
        raise S1GCSGateError("Round 2 contract diagnostic config is invalid") from error
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-round2-response-contract-diagnostics",
        "policy_version": S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION,
        "query_count": len(ordered_ids),
        "query_ids": list(ordered_ids),
        "rows": [row.model_dump(mode="json") for row in ordered_rows],
    }
    try:
        return S1Round2ResponseContractDiagnostics.model_validate(
            {**payload, "diagnostics_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("Round 2 response contract diagnostics are invalid") from error


def _query_ids_sha256(query_ids: Sequence[str]) -> str:
    return _hash_payload({"query_ids": list(query_ids)})


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise S1GCSGateError("bootstrap percentile requires observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class S1GCSEvidenceBinding(_StrictFrozenModel):
    """Exact input-file commitments for one offline GCS gate."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-gcs-evidence-binding"] = (
        "portfolio-s1-gcs-evidence-binding"
    )
    policy_version: Literal[S1_GCS_GATE_POLICY_VERSION] = S1_GCS_GATE_POLICY_VERSION
    phase: S1GCSGatePhase
    population_binding_file_sha256: Sha256
    queries_file_sha256: Sha256
    baseline_scores_file_sha256: Sha256
    candidate_scores_file_sha256: Sha256
    parent_bank_file_sha256: Sha256
    candidate_bank_file_sha256: Sha256
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.binding_sha256 != _self_hash(self, "binding_sha256"):
            raise ValueError("S1 GCS evidence binding self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def make_s1_gcs_evidence_binding(
    *,
    phase: S1GCSGatePhase,
    population_binding_file_sha256: str,
    queries_file_sha256: str,
    baseline_scores_file_sha256: str,
    candidate_scores_file_sha256: str,
    parent_bank_file_sha256: str,
    candidate_bank_file_sha256: str,
) -> S1GCSEvidenceBinding:
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-evidence-binding",
        "policy_version": S1_GCS_GATE_POLICY_VERSION,
        "phase": phase,
        "population_binding_file_sha256": population_binding_file_sha256,
        "queries_file_sha256": queries_file_sha256,
        "baseline_scores_file_sha256": baseline_scores_file_sha256,
        "candidate_scores_file_sha256": candidate_scores_file_sha256,
        "parent_bank_file_sha256": parent_bank_file_sha256,
        "candidate_bank_file_sha256": candidate_bank_file_sha256,
    }
    try:
        return S1GCSEvidenceBinding.model_validate(
            {**payload, "binding_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 GCS evidence binding is invalid") from error


class S1GCSCapabilityDelta(_StrictFrozenModel):
    capability_id: str
    point_delta_pp: float

    @field_validator("capability_id")
    @classmethod
    def validate_capability(cls, value: str) -> str:
        if value not in GCS_CAPABILITY_ORDER:
            raise ValueError("unknown S1 GCS capability")
        return value

    @field_validator("point_delta_pp")
    @classmethod
    def validate_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("S1 GCS capability delta must be finite")
        return value


class S1GCSContractReasonSafety(_StrictFrozenModel):
    reason_code: str
    baseline_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    new_occurrence_count: int = Field(ge=0)

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        if value not in S1_CONTRACT_REGRESSION_REASON_CODES:
            raise ValueError("unknown S1 contract-regression reason code")
        return value


class S1GCSCapabilitySafety(_StrictFrozenModel):
    capability_id: str
    baseline_success_count: int = Field(ge=0)
    candidate_success_count: int = Field(ge=0)
    static_success_to_candidate_failure_count: int = Field(ge=0)
    contract_reasons: tuple[S1GCSContractReasonSafety, ...]

    @field_validator("capability_id")
    @classmethod
    def validate_capability(cls, value: str) -> str:
        if value not in GCS_CAPABILITY_ORDER:
            raise ValueError("unknown S1 safety capability")
        return value

    @field_validator("contract_reasons", mode="before")
    @classmethod
    def coerce_contract_reasons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_safety(self) -> Self:
        if tuple(item.reason_code for item in self.contract_reasons) != (
            S1_CONTRACT_REGRESSION_REASON_CODES
        ):
            raise ValueError("S1 contract reason order is not frozen")
        if self.static_success_to_candidate_failure_count > (
            self.baseline_success_count
        ):
            raise ValueError("S1 paired regressions exceed baseline successes")
        return self


class S1GCSPairedSafety(_StrictFrozenModel):
    static_success_to_candidate_failure_count: int = Field(ge=0)
    new_contract_reason_occurrence_count: int = Field(ge=0)
    per_capability: tuple[S1GCSCapabilitySafety, ...]

    @field_validator("per_capability", mode="before")
    @classmethod
    def coerce_capabilities(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_safety(self) -> Self:
        if tuple(item.capability_id for item in self.per_capability) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("S1 paired-safety capability order is not frozen")
        if self.static_success_to_candidate_failure_count != sum(
            item.static_success_to_candidate_failure_count
            for item in self.per_capability
        ):
            raise ValueError("S1 paired regression total differs from capabilities")
        if self.new_contract_reason_occurrence_count != sum(
            reason.new_occurrence_count
            for item in self.per_capability
            for reason in item.contract_reasons
        ):
            raise ValueError("S1 contract regression total differs from capabilities")
        return self


S1DevelopmentPatchDecision = Literal["retain_patch", "inherit_parent"]


class S1DevelopmentCapabilityScreen(_StrictFrozenModel):
    capability_id: str
    decision: S1DevelopmentPatchDecision
    baseline_success_count: int = Field(ge=0)
    candidate_success_count: int = Field(ge=0)
    static_success_to_candidate_failure_count: int = Field(ge=0)
    contract_reasons: tuple[S1GCSContractReasonSafety, ...]
    reason_codes: tuple[str, ...]

    @field_validator("contract_reasons", "reason_codes", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_screen(self) -> Self:
        if self.capability_id not in GCS_CAPABILITY_ORDER:
            raise ValueError("unknown development-screen capability")
        expected_reasons: set[str] = set()
        if self.candidate_success_count < self.baseline_success_count:
            expected_reasons.add("candidate_success_count_decreased")
        if self.static_success_to_candidate_failure_count:
            expected_reasons.add("paired_static_success_regressed")
        if any(
            item.candidate_count > item.baseline_count or item.new_occurrence_count
            for item in self.contract_reasons
        ):
            expected_reasons.add("contract_reason_regressed")
        if self.reason_codes != tuple(sorted(expected_reasons)):
            raise ValueError("development-screen reasons are inconsistent")
        if (self.decision == "retain_patch") != (not expected_reasons):
            raise ValueError("development-screen decision is inconsistent")
        return self


class S1DevelopmentScreen(_StrictFrozenModel):
    policy_version: Literal["portfolio-s1-development-screen-v1"] = (
        "portfolio-s1-development-screen-v1"
    )
    decisions: tuple[S1DevelopmentCapabilityScreen, ...]

    @field_validator("decisions", mode="before")
    @classmethod
    def coerce_decisions(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_screen(self) -> Self:
        if tuple(item.capability_id for item in self.decisions) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("development-screen capability order is not frozen")
        return self


class S1Round2DevelopmentScreen(_StrictFrozenModel):
    """Exact replay200 identity and sparse patch decisions for Round 2."""

    schema_version: Literal[2] = 2
    artifact_kind: Literal["portfolio-s1-round2-development-screen"] = (
        "portfolio-s1-round2-development-screen"
    )
    policy_version: Literal[S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION] = (
        S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION
    )
    purpose: Literal["development_only_not_acceptance"] = (
        "development_only_not_acceptance"
    )
    source_split: Literal["opt_pool"] = "opt_pool"
    query_count: Literal[S1_ROUND2_DEVELOPMENT_QUERY_COUNT] = (
        S1_ROUND2_DEVELOPMENT_QUERY_COUNT
    )
    query_ids: tuple[str, ...]
    query_ids_sha256: Sha256
    required_regression_query_ids: tuple[str, ...]
    population_mapping_sha256: Sha256
    baseline_scores_sha256: Sha256
    raw_candidate_scores_sha256: Sha256
    response_contract_diagnostics_sha256: Sha256
    parent_bank_sha256: Sha256
    parent_bank_file_sha256: Sha256
    raw_candidate_bank_sha256: Sha256
    raw_candidate_bank_file_sha256: Sha256
    source_evidence: S1GCSEvidenceBinding
    decisions: tuple[S1DevelopmentCapabilityScreen, ...]
    screen_sha256: Sha256

    @field_validator(
        "query_ids", "required_regression_query_ids", "decisions", mode="before"
    )
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_screen(self) -> Self:
        if (
            self.query_ids != tuple(sorted(set(self.query_ids)))
            or len(self.query_ids) != self.query_count
            or self.query_ids_sha256 != _query_ids_sha256(self.query_ids)
        ):
            raise ValueError("Round 2 development population identity is invalid")
        if self.required_regression_query_ids != (
            S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS
        ) or not set(self.required_regression_query_ids).issubset(self.query_ids):
            raise ValueError("Round 2 required regression rows are missing")
        if tuple(item.capability_id for item in self.decisions) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("Round 2 screen capability order is not frozen")
        if self.source_evidence.phase != "replay":
            raise ValueError("Round 2 screen evidence must be replay evidence")
        if self.parent_bank_sha256 == self.raw_candidate_bank_sha256:
            raise ValueError("Round 2 raw candidate Bank must differ from its parent")
        if (
            self.source_evidence.parent_bank_file_sha256 != self.parent_bank_file_sha256
            or self.source_evidence.candidate_bank_file_sha256
            != self.raw_candidate_bank_file_sha256
        ):
            raise ValueError("Round 2 screen evidence Bank files drifted")
        if self.screen_sha256 != _self_hash(self, "screen_sha256"):
            raise ValueError("Round 2 development screen self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class S1GCSContrast(_StrictFrozenModel):
    """Paired GCS-v2 point estimates and component-bootstrap macro interval."""

    bootstrap_policy_version: Literal[GCS_BOOTSTRAP_POLICY_VERSION] = (
        GCS_BOOTSTRAP_POLICY_VERSION
    )
    root_seed: Literal[GCS_BOOTSTRAP_ROOT_SEED] = GCS_BOOTSTRAP_ROOT_SEED
    replicates: Literal[GCS_BOOTSTRAP_REPLICATES] = GCS_BOOTSTRAP_REPLICATES
    derived_seed: int = Field(ge=0)
    draw_stream_sha256: Sha256
    query_count: int = Field(ge=1)
    component_count: int = Field(ge=1)
    macro_delta_pp: float
    macro_ci95_low_pp: float | None
    macro_ci95_high_pp: float | None
    macro_available_replicates: int = Field(ge=0, le=GCS_BOOTSTRAP_REPLICATES)
    query_micro_delta_pp: float
    hard_error_delta_pp: float
    per_capability: tuple[S1GCSCapabilityDelta, ...]
    baseline_oracle_coverage_complete: bool
    candidate_oracle_coverage_complete: bool
    zero_variance: bool
    precision_warnings: tuple[str, ...]

    @field_validator("per_capability", "precision_warnings", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_contrast(self) -> Self:
        if tuple(item.capability_id for item in self.per_capability) != (
            GCS_CAPABILITY_ORDER
        ):
            raise ValueError("S1 GCS capability delta order is not frozen")
        finite_values = (
            self.macro_delta_pp,
            self.query_micro_delta_pp,
            self.hard_error_delta_pp,
            *(item.point_delta_pp for item in self.per_capability),
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("S1 GCS contrast contains a non-finite point estimate")
        interval_present = (
            self.macro_ci95_low_pp is not None and self.macro_ci95_high_pp is not None
        )
        if interval_present != (self.macro_available_replicates == self.replicates):
            raise ValueError("S1 GCS macro interval availability is inconsistent")
        if (self.macro_ci95_low_pp is None) != (self.macro_ci95_high_pp is None):
            raise ValueError("S1 GCS macro interval is only partially present")
        if interval_present:
            assert self.macro_ci95_low_pp is not None
            assert self.macro_ci95_high_pp is not None
            if (
                not math.isfinite(self.macro_ci95_low_pp)
                or not math.isfinite(self.macro_ci95_high_pp)
                or self.macro_ci95_low_pp > self.macro_ci95_high_pp
            ):
                raise ValueError("S1 GCS macro interval is invalid")
        if self.precision_warnings != tuple(dict.fromkeys(self.precision_warnings)):
            raise ValueError("S1 GCS precision warnings must be unique")
        return self


class S1DevelopmentCompositeReport(_StrictFrozenModel):
    """Non-gate development check for the parent-composed sparse Bank."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-development-composite-report"] = (
        "portfolio-s1-development-composite-report"
    )
    policy_version: Literal["portfolio-s1-development-composite-v1"] = (
        "portfolio-s1-development-composite-v1"
    )
    purpose: Literal["development_only_not_acceptance"] = (
        "development_only_not_acceptance"
    )
    query_count: int = Field(ge=1)
    query_ids: tuple[str, ...]
    query_ids_sha256: Sha256
    population_mapping_sha256: Sha256
    baseline_scores_sha256: Sha256
    candidate_scores_sha256: Sha256
    response_contract_diagnostics_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    contrast: S1GCSContrast
    paired_safety: S1GCSPairedSafety
    macro_nonnegative: bool
    all_capabilities_nonnegative: bool
    hard_error_nonincreasing: bool
    paired_success_preserved: bool
    contract_reasons_preserved: bool
    passed: bool
    reason_codes: tuple[str, ...]
    report_sha256: Sha256

    @field_validator("query_ids", "reason_codes", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if (
            self.query_ids != tuple(sorted(set(self.query_ids)))
            or len(self.query_ids) != self.query_count
            or self.query_ids_sha256 != _query_ids_sha256(self.query_ids)
            or self.contrast.query_count != self.query_count
        ):
            raise ValueError("development composite population identity is invalid")
        expected = {
            "macro_negative": not self.macro_nonnegative,
            "capability_negative": not self.all_capabilities_nonnegative,
            "hard_error_increased": not self.hard_error_nonincreasing,
            "paired_static_success_regressed": not self.paired_success_preserved,
            "contract_reason_regressed": not self.contract_reasons_preserved,
        }
        reasons = tuple(sorted(key for key, failed in expected.items() if failed))
        if self.reason_codes != reasons or self.passed != (not reasons):
            raise ValueError("development composite decision is inconsistent")
        if self.report_sha256 != _self_hash(self, "report_sha256"):
            raise ValueError("development composite report self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class S1Round2CandidateFreeze(_StrictFrozenModel):
    """The single sparse candidate/runtime identity authorized for body_gate75."""

    schema_version: Literal[2] = 2
    artifact_kind: Literal["portfolio-s1-round2-candidate-freeze"] = (
        "portfolio-s1-round2-candidate-freeze"
    )
    policy_version: Literal[S1_ROUND2_BODY_GATE_POLICY_VERSION] = (
        S1_ROUND2_BODY_GATE_POLICY_VERSION
    )
    policy_sha256: Literal[S1_ROUND2_BODY_GATE_POLICY_SHA256] = (
        S1_ROUND2_BODY_GATE_POLICY_SHA256
    )
    gate_policy_version: Literal[S1_GCS_GATE_POLICY_VERSION] = (
        S1_GCS_GATE_POLICY_VERSION
    )
    gate_policy_sha256: Literal[S1_GCS_GATE_POLICY_SHA256] = S1_GCS_GATE_POLICY_SHA256
    development_report_sha256: Sha256
    development_report_file_sha256: Sha256
    development_screen_sha256: Sha256
    development_screen_file_sha256: Sha256
    screened_bank_receipt_sha256: Sha256
    screened_bank_receipt_file_sha256: Sha256
    retained_capability_ids: tuple[str, ...]
    parent_bank_sha256: Sha256
    parent_bank_file_sha256: Sha256
    candidate_bank_sha256: Sha256
    candidate_bank_file_sha256: Sha256
    runtime_lock_sha256: Sha256
    runtime_lock_file_sha256: Sha256
    body_gate_access_count: Literal[1] = 1
    freeze_sha256: Sha256

    @field_validator("retained_capability_ids", mode="before")
    @classmethod
    def coerce_capabilities(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_freeze(self) -> Self:
        if self.parent_bank_sha256 == self.candidate_bank_sha256:
            raise ValueError("Round 2 candidate Bank must differ from its parent")
        if self.retained_capability_ids != tuple(
            sorted(set(self.retained_capability_ids))
        ) or not set(self.retained_capability_ids).issubset(GCS_CAPABILITY_ORDER):
            raise ValueError("Round 2 retained capability ids are invalid")
        if self.freeze_sha256 != _self_hash(self, "freeze_sha256"):
            raise ValueError("Round 2 candidate freeze self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class S1GCSGateCriterion(_StrictFrozenModel):
    name: str
    operator: Literal[">=", "<="]
    threshold_pp: float
    observed_pp: float | None
    passed: bool | None

    @model_validator(mode="after")
    def validate_criterion(self) -> Self:
        if not self.name or self.name != self.name.strip():
            raise ValueError("S1 GCS criterion name must be non-blank and trimmed")
        for value in (self.threshold_pp, self.observed_pp):
            if value is not None and not math.isfinite(value):
                raise ValueError("S1 GCS criterion value must be finite")
        if self.passed is not None:
            if self.observed_pp is None:
                raise ValueError("resolved S1 GCS criterion lacks an observation")
            expected = (
                self.observed_pp >= self.threshold_pp
                if self.operator == ">="
                else self.observed_pp <= self.threshold_pp
            )
            if self.passed != expected:
                raise ValueError("S1 GCS criterion result differs from threshold")
        return self


class S1GCSGateReport(_StrictFrozenModel):
    """One replay screen or body-gate decision over a fixed paired population."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-gcs-gate-report"] = (
        "portfolio-s1-gcs-gate-report"
    )
    policy_version: Literal[S1_GCS_GATE_POLICY_VERSION] = S1_GCS_GATE_POLICY_VERSION
    policy_sha256: Literal[S1_GCS_GATE_POLICY_SHA256] = S1_GCS_GATE_POLICY_SHA256
    gcs_policy_version: Literal[GCS_V2_POLICY_VERSION] = GCS_V2_POLICY_VERSION
    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    phase: S1GCSGatePhase
    scope: str
    source_split: Literal["opt_pool", "val"]
    decision_rule: str
    evidence: S1GCSEvidenceBinding
    query_count: int = Field(ge=1)
    component_count: int = Field(ge=1)
    query_ids: tuple[str, ...]
    query_ids_sha256: Sha256
    population_mapping_sha256: Sha256
    baseline_config: Literal["llm_static"] = "llm_static"
    candidate_config: Literal["s1"] = "s1"
    parent_bank_sha256: Sha256
    candidate_bank_sha256: Sha256
    baseline_summary: GCSConfigSummaryV2
    candidate_summary: GCSConfigSummaryV2
    contrast: S1GCSContrast
    criteria: tuple[S1GCSGateCriterion, ...]
    coverage_complete: bool
    integrity_complete: Literal[True] = True
    status: S1GCSGateStatus
    reason_codes: tuple[str, ...]
    pairwise_judge_call_count: Literal[0] = 0
    legacy_final_judge_call_count: Literal[0] = 0
    provider_model_call_count: Literal[0] = 0
    report_sha256: Sha256

    @field_validator("query_ids", "criteria", "reason_codes", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if (
            self.scope != _PHASE_SCOPE[self.phase]
            or self.source_split != _PHASE_SPLIT[self.phase]
            or self.decision_rule != _PHASE_RULE[self.phase]
            or self.query_count != _PHASE_QUERY_COUNT[self.phase]
            or self.evidence.phase != self.phase
        ):
            raise ValueError("S1 GCS phase geometry or identity is inconsistent")
        if (
            self.query_ids != tuple(sorted(set(self.query_ids)))
            or len(self.query_ids) != self.query_count
            or self.query_ids_sha256 != _query_ids_sha256(self.query_ids)
        ):
            raise ValueError("S1 GCS query population identity is invalid")
        if self.parent_bank_sha256 == self.candidate_bank_sha256:
            raise ValueError("S1 GCS candidate Bank must differ from its parent")
        for summary, config in (
            (self.baseline_summary, self.baseline_config),
            (self.candidate_summary, self.candidate_config),
        ):
            if (
                summary.config != config
                or summary.scope != self.scope
                or summary.query_count != self.query_count
                or summary.component_count != self.component_count
                or summary.population_mapping_sha256 != self.population_mapping_sha256
            ):
                raise ValueError("S1 GCS summary differs from report population")
        if (
            self.contrast.query_count != self.query_count
            or self.contrast.component_count != self.component_count
        ):
            raise ValueError("S1 GCS contrast differs from report population")
        expected_coverage = (
            self.baseline_summary.headline_available
            and self.candidate_summary.headline_available
            and self.contrast.baseline_oracle_coverage_complete
            and self.contrast.candidate_oracle_coverage_complete
        )
        if self.coverage_complete != expected_coverage:
            raise ValueError("S1 GCS coverage flag is inconsistent")
        expected_names = ["macro_delta_pp"]
        if self.phase == "body_gate":
            expected_names.append("macro_ci95_low_pp")
        expected_names.append("hard_error_delta_pp")
        expected_names.extend(
            f"capability_delta_pp:{capability}" for capability in GCS_CAPABILITY_ORDER
        )
        if [item.name for item in self.criteria] != expected_names:
            raise ValueError("S1 GCS criterion universe/order is not frozen")
        if self.status == "unavailable":
            if self.coverage_complete and all(
                item.observed_pp is not None for item in self.criteria
            ):
                raise ValueError("available S1 GCS evidence marked unavailable")
            if any(item.passed is not None for item in self.criteria):
                raise ValueError("unavailable S1 GCS gate claims resolved criteria")
        else:
            if not self.coverage_complete or any(
                item.passed is None for item in self.criteria
            ):
                raise ValueError("available S1 GCS gate lacks coverage or criteria")
            all_passed = all(item.passed for item in self.criteria)
            if (self.status == "passed") != all_passed:
                raise ValueError("S1 GCS gate status differs from criteria")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("S1 GCS reason codes must be sorted and unique")
        if self.report_sha256 != _self_hash(self, "report_sha256"):
            raise ValueError("S1 GCS gate report self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class S1Round2BodyGateReport(_StrictFrozenModel):
    """Forward-only body decision linked to the frozen development candidate."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-round2-body-gate-report"] = (
        "portfolio-s1-round2-body-gate-report"
    )
    policy_version: Literal[S1_ROUND2_BODY_GATE_POLICY_VERSION] = (
        S1_ROUND2_BODY_GATE_POLICY_VERSION
    )
    policy_sha256: Literal[S1_ROUND2_BODY_GATE_POLICY_SHA256] = (
        S1_ROUND2_BODY_GATE_POLICY_SHA256
    )
    development_report: S1DevelopmentCompositeReport
    candidate_freeze: S1Round2CandidateFreeze
    body_gate_report: S1GCSGateReport
    response_contract_diagnostics_sha256: Sha256
    paired_safety: S1GCSPairedSafety
    paired_success_preserved: bool
    contract_reasons_preserved: bool
    status: S1GCSGateStatus
    reason_codes: tuple[str, ...]
    provider_model_call_count: Literal[0] = 0
    report_sha256: Sha256

    @field_validator("reason_codes", mode="before")
    @classmethod
    def coerce_reasons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        development = self.development_report
        freeze = self.candidate_freeze
        body = self.body_gate_report
        if not development.passed:
            raise ValueError("Round 2 body gate requires a passed development report")
        if (
            freeze.development_report_sha256 != development.report_sha256
            or freeze.development_report_file_sha256
            != sha256_bytes(development.canonical_bytes())
        ):
            raise ValueError("Round 2 freeze differs from the development report")
        if (
            body.phase != "body_gate"
            or body.parent_bank_sha256 != freeze.parent_bank_sha256
            or body.candidate_bank_sha256 != freeze.candidate_bank_sha256
            or body.evidence.parent_bank_file_sha256 != freeze.parent_bank_file_sha256
            or body.evidence.candidate_bank_file_sha256
            != freeze.candidate_bank_file_sha256
        ):
            raise ValueError("Round 2 body report differs from the candidate freeze")
        if set(development.query_ids) & set(body.query_ids):
            raise ValueError("Round 2 body and development populations overlap")
        if not self.response_contract_diagnostics_sha256:
            raise ValueError("Round 2 body contract diagnostics are absent")
        paired_preserved = (
            self.paired_safety.static_success_to_candidate_failure_count == 0
        )
        contract_preserved = (
            self.paired_safety.new_contract_reason_occurrence_count == 0
        )
        if (
            self.paired_success_preserved != paired_preserved
            or self.contract_reasons_preserved != contract_preserved
        ):
            raise ValueError("Round 2 preservation flags differ from paired evidence")
        reasons = set(body.reason_codes)
        if not paired_preserved:
            reasons.add("threshold_failed:static_success_to_candidate_failure_count")
        if not contract_preserved:
            reasons.add("threshold_failed:new_contract_reason_occurrence_count")
        if body.status == "unavailable":
            expected_status: S1GCSGateStatus = "unavailable"
        elif body.status == "passed" and paired_preserved and contract_preserved:
            expected_status = "passed"
        else:
            expected_status = "failed"
        if self.status != expected_status or self.reason_codes != tuple(
            sorted(reasons)
        ):
            raise ValueError("Round 2 body decision differs from frozen criteria")
        if self.report_sha256 != _self_hash(self, "report_sha256"):
            raise ValueError("Round 2 body gate report self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _strict_scores(
    values: Sequence[GCSQueryScoreV2], *, label: str
) -> tuple[GCSQueryScoreV2, ...]:
    result: list[GCSQueryScoreV2] = []
    for value in values:
        if type(value) is not GCSQueryScoreV2:
            raise S1GCSGateError(f"{label} must contain exact GCSQueryScoreV2 rows")
        try:
            result.append(
                GCSQueryScoreV2.model_validate(
                    value.model_dump(mode="python"), strict=True
                )
            )
        except ValidationError as error:
            raise S1GCSGateError(f"{label} contains an invalid GCS v2 row") from error
    return tuple(result)


def _verified_round2_contract_diagnostics(
    value: S1Round2ResponseContractDiagnostics,
    *,
    queries: Sequence[Query],
) -> S1Round2ResponseContractDiagnostics:
    if type(value) is not S1Round2ResponseContractDiagnostics:
        raise S1GCSGateError("Round 2 contract diagnostics have the wrong type")
    try:
        verified = S1Round2ResponseContractDiagnostics.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("Round 2 contract diagnostics are invalid") from error
    query_ids = tuple(sorted(item.query_id for item in queries))
    if verified.query_ids != query_ids:
        raise S1GCSGateError(
            "Round 2 contract diagnostics differ from the scored population"
        )
    return verified


def _paired_safety(
    *,
    queries: Sequence[Query],
    baseline: Sequence[GCSQueryScoreV2],
    candidate: Sequence[GCSQueryScoreV2],
    response_contract_diagnostics: S1Round2ResponseContractDiagnostics | None = None,
) -> S1GCSPairedSafety:
    query_by_id = {item.query_id: item for item in queries}
    baseline_by_id = {item.query_id: item for item in baseline}
    candidate_by_id = {item.query_id: item for item in candidate}
    expected_ids = set(query_by_id)
    if (
        len(query_by_id) != len(queries)
        or set(baseline_by_id) != expected_ids
        or set(candidate_by_id) != expected_ids
        or len(baseline_by_id) != len(baseline)
        or len(candidate_by_id) != len(candidate)
    ):
        raise S1GCSGateError("S1 paired safety population is not rectangular")
    population = build_gcs_population_v2(tuple(queries))
    component_by_id = {item.query_id: item.component_id for item in population.bindings}
    for score in (*baseline, *candidate):
        query = query_by_id[score.query_id]
        if (
            score.component_id != component_by_id[score.query_id]
            or score.canonical_capability != query.canonical_capability
            or score.evaluated_capability != query.canonical_capability
        ):
            raise S1GCSGateError(
                "S1 paired safety score identity differs from its query"
            )
    diagnostic_reasons: dict[tuple[str, str], frozenset[str]] = {}
    if response_contract_diagnostics is not None:
        diagnostics = _verified_round2_contract_diagnostics(
            response_contract_diagnostics, queries=queries
        )
        diagnostic_reasons = {
            (row.query_id, row.config): frozenset(row.mapped_contract_reason_codes)
            for row in diagnostics.rows
        }
    capabilities: list[S1GCSCapabilitySafety] = []
    for capability in GCS_CAPABILITY_ORDER:
        query_ids = tuple(
            sorted(
                query_id
                for query_id, query in query_by_id.items()
                if query.canonical_capability == capability
            )
        )
        if not query_ids:
            raise S1GCSGateError("S1 paired safety lacks one capability")
        baseline_success = sum(baseline_by_id[item].gcs for item in query_ids)
        candidate_success = sum(candidate_by_id[item].gcs for item in query_ids)
        paired_regressions = sum(
            baseline_by_id[item].gcs == 1 and candidate_by_id[item].gcs == 0
            for item in query_ids
        )
        reason_safety = tuple(
            S1GCSContractReasonSafety(
                reason_code=reason,
                baseline_count=sum(
                    reason
                    in (
                        set(baseline_by_id[item].reason_codes)
                        | set(diagnostic_reasons.get((item, "llm_static"), ()))
                    )
                    for item in query_ids
                ),
                candidate_count=sum(
                    reason
                    in (
                        set(candidate_by_id[item].reason_codes)
                        | set(diagnostic_reasons.get((item, "s1"), ()))
                    )
                    for item in query_ids
                ),
                new_occurrence_count=sum(
                    reason
                    not in (
                        set(baseline_by_id[item].reason_codes)
                        | set(diagnostic_reasons.get((item, "llm_static"), ()))
                    )
                    and reason
                    in (
                        set(candidate_by_id[item].reason_codes)
                        | set(diagnostic_reasons.get((item, "s1"), ()))
                    )
                    for item in query_ids
                ),
            )
            for reason in S1_CONTRACT_REGRESSION_REASON_CODES
        )
        capabilities.append(
            S1GCSCapabilitySafety(
                capability_id=capability,
                baseline_success_count=baseline_success,
                candidate_success_count=candidate_success,
                static_success_to_candidate_failure_count=paired_regressions,
                contract_reasons=reason_safety,
            )
        )
    return S1GCSPairedSafety(
        static_success_to_candidate_failure_count=sum(
            item.static_success_to_candidate_failure_count for item in capabilities
        ),
        new_contract_reason_occurrence_count=sum(
            reason.new_occurrence_count
            for item in capabilities
            for reason in item.contract_reasons
        ),
        per_capability=tuple(capabilities),
    )


def screen_s1_development_patches(
    *,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    response_contract_diagnostics: S1Round2ResponseContractDiagnostics | None = None,
) -> S1DevelopmentScreen:
    """Select capability patches using paired development evidence only.

    This helper does not merge Bank bytes.  It emits the frozen retain/inherit
    decision which the sparse compiler must apply before the composite replay.
    """

    baseline = _strict_scores(baseline_scores, label="baseline scores")
    candidate = _strict_scores(candidate_scores, label="candidate scores")
    if not queries or any(item.split != "opt_pool" for item in queries):
        raise S1GCSGateError("development screen requires opt_pool queries")
    if any(item.config != "llm_static" for item in baseline) or any(
        item.config != "s1" for item in candidate
    ):
        raise S1GCSGateError("development screen requires llm_static and s1 scores")
    safety = _paired_safety(
        queries=queries,
        baseline=baseline,
        candidate=candidate,
        response_contract_diagnostics=response_contract_diagnostics,
    )
    decisions: list[S1DevelopmentCapabilityScreen] = []
    for item in safety.per_capability:
        reasons: set[str] = set()
        if item.candidate_success_count < item.baseline_success_count:
            reasons.add("candidate_success_count_decreased")
        if item.static_success_to_candidate_failure_count:
            reasons.add("paired_static_success_regressed")
        if any(
            reason.candidate_count > reason.baseline_count
            or reason.new_occurrence_count
            for reason in item.contract_reasons
        ):
            reasons.add("contract_reason_regressed")
        decisions.append(
            S1DevelopmentCapabilityScreen(
                capability_id=item.capability_id,
                decision="inherit_parent" if reasons else "retain_patch",
                baseline_success_count=item.baseline_success_count,
                candidate_success_count=item.candidate_success_count,
                static_success_to_candidate_failure_count=(
                    item.static_success_to_candidate_failure_count
                ),
                contract_reasons=item.contract_reasons,
                reason_codes=tuple(sorted(reasons)),
            )
        )
    return S1DevelopmentScreen(decisions=tuple(decisions))


def screen_s1_round2_development_patches(
    *,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    raw_candidate_scores: Sequence[GCSQueryScoreV2],
    response_contract_diagnostics: S1Round2ResponseContractDiagnostics,
    source_evidence: S1GCSEvidenceBinding,
    parent_bank_sha256: str,
    raw_candidate_bank_sha256: str,
) -> S1Round2DevelopmentScreen:
    """Create the exact replay200 screen consumed by the Round 2 compiler."""

    ordered_queries = tuple(sorted(queries, key=lambda item: item.query_id))
    if (
        len(ordered_queries) != S1_ROUND2_DEVELOPMENT_QUERY_COUNT
        or len({item.query_id for item in ordered_queries}) != len(ordered_queries)
        or any(item.split != "opt_pool" for item in ordered_queries)
        or {item.canonical_capability for item in ordered_queries}
        != set(GCS_CAPABILITY_ORDER)
    ):
        raise S1GCSGateError(
            "Round 2 development screen requires exact replay200 across six capabilities"
        )
    query_ids = tuple(item.query_id for item in ordered_queries)
    missing = set(S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS) - set(query_ids)
    if missing:
        raise S1GCSGateError(
            "Round 2 development screen is missing required regressions: "
            + ", ".join(sorted(missing))
        )
    baseline = _strict_scores(baseline_scores, label="baseline scores")
    raw_candidate = _strict_scores(raw_candidate_scores, label="raw candidate scores")
    diagnostics = _verified_round2_contract_diagnostics(
        response_contract_diagnostics, queries=ordered_queries
    )
    try:
        evidence = S1GCSEvidenceBinding.model_validate(
            source_evidence.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise S1GCSGateError("Round 2 source evidence is invalid") from error
    if evidence.phase != "replay":
        raise S1GCSGateError("Round 2 source evidence must be replay evidence")
    generic = screen_s1_development_patches(
        queries=ordered_queries,
        baseline_scores=baseline,
        candidate_scores=raw_candidate,
        response_contract_diagnostics=diagnostics,
    )
    population = build_gcs_population_v2(ordered_queries)
    payload = {
        "schema_version": 2,
        "artifact_kind": "portfolio-s1-round2-development-screen",
        "policy_version": S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION,
        "purpose": "development_only_not_acceptance",
        "source_split": "opt_pool",
        "query_count": S1_ROUND2_DEVELOPMENT_QUERY_COUNT,
        "query_ids": list(query_ids),
        "query_ids_sha256": _query_ids_sha256(query_ids),
        "required_regression_query_ids": list(S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS),
        "population_mapping_sha256": population.population_mapping_sha256,
        "baseline_scores_sha256": _hash_payload(
            [
                item.model_dump(mode="json")
                for item in sorted(baseline, key=lambda value: value.query_id)
            ]
        ),
        "raw_candidate_scores_sha256": _hash_payload(
            [
                item.model_dump(mode="json")
                for item in sorted(raw_candidate, key=lambda value: value.query_id)
            ]
        ),
        "response_contract_diagnostics_sha256": diagnostics.diagnostics_sha256,
        "parent_bank_sha256": parent_bank_sha256,
        "parent_bank_file_sha256": evidence.parent_bank_file_sha256,
        "raw_candidate_bank_sha256": raw_candidate_bank_sha256,
        "raw_candidate_bank_file_sha256": evidence.candidate_bank_file_sha256,
        "source_evidence": evidence.model_dump(mode="json"),
        "decisions": [item.model_dump(mode="json") for item in generic.decisions],
    }
    try:
        return S1Round2DevelopmentScreen.model_validate(
            {**payload, "screen_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("Round 2 development screen is invalid") from error


def _rates(
    query_ids: Sequence[str],
    scores: Mapping[str, GCSQueryScoreV2],
    queries: Mapping[str, Query],
) -> tuple[float, float, float, dict[str, float]] | None:
    per_capability: dict[str, float] = {}
    for capability in GCS_CAPABILITY_ORDER:
        ids = tuple(
            query_id
            for query_id in query_ids
            if queries[query_id].canonical_capability == capability
        )
        if not ids:
            return None
        per_capability[capability] = sum(scores[item].gcs for item in ids) / len(ids)
    total = len(query_ids)
    return (
        sum(per_capability.values()) / len(GCS_CAPABILITY_ORDER),
        sum(scores[item].gcs for item in query_ids) / total,
        sum(scores[item].hard_error for item in query_ids) / total,
        per_capability,
    )


def _paired_gcs_v2_contrast(
    *,
    baseline: tuple[GCSQueryScoreV2, ...],
    candidate: tuple[GCSQueryScoreV2, ...],
    queries: tuple[Query, ...],
    scope: str,
) -> S1GCSContrast:
    population = build_gcs_population_v2(queries)
    query_by_id = {item.query_id: item for item in queries}
    baseline_by_id = {item.query_id: item for item in baseline}
    candidate_by_id = {item.query_id: item for item in candidate}
    ordered_ids = tuple(sorted(query_by_id))
    baseline_point = _rates(ordered_ids, baseline_by_id, query_by_id)
    candidate_point = _rates(ordered_ids, candidate_by_id, query_by_id)
    if baseline_point is None or candidate_point is None:
        raise S1GCSGateError("S1 GCS population lacks one of the six capabilities")

    mutable_members: dict[str, list[str]] = defaultdict(list)
    for binding in population.bindings:
        mutable_members[binding.component_id].append(binding.query_id)
    component_ids = tuple(component for component, _size in population.component_sizes)
    members = {
        component: tuple(sorted(query_ids))
        for component, query_ids in mutable_members.items()
    }
    seed_payload = {
        "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
        "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
        "scope": scope,
        "population_mapping_sha256": population.population_mapping_sha256,
    }
    seed_digest = hashlib.sha256(canonical_json_bytes(seed_payload)).digest()
    derived_seed = int.from_bytes(seed_digest[:16], "big")
    rng = random.Random(derived_seed)
    draw_hasher = hashlib.sha256()
    macro_deltas: list[float] = []
    for _replicate in range(GCS_BOOTSTRAP_REPLICATES):
        draw = tuple(
            component_ids[rng.randrange(len(component_ids))] for _ in component_ids
        )
        draw_hasher.update(canonical_json_bytes(list(draw)))
        sampled_ids = tuple(
            query_id for component in draw for query_id in members[component]
        )
        baseline_rates = _rates(sampled_ids, baseline_by_id, query_by_id)
        candidate_rates = _rates(sampled_ids, candidate_by_id, query_by_id)
        if baseline_rates is not None and candidate_rates is not None:
            macro_deltas.append(100.0 * (candidate_rates[0] - baseline_rates[0]))

    point_per_capability = tuple(
        S1GCSCapabilityDelta(
            capability_id=capability,
            point_delta_pp=100.0
            * (candidate_point[3][capability] - baseline_point[3][capability]),
        )
        for capability in GCS_CAPABILITY_ORDER
    )
    all_point_values = (
        100.0 * (candidate_point[0] - baseline_point[0]),
        100.0 * (candidate_point[1] - baseline_point[1]),
        100.0 * (candidate_point[2] - baseline_point[2]),
        *(item.point_delta_pp for item in point_per_capability),
    )
    zero_variance = bool(macro_deltas) and min(macro_deltas) == max(macro_deltas)
    warnings: list[str] = []
    if population.component_count < 20:
        warnings.append("low_component_count_precision")
    if len(macro_deltas) != GCS_BOOTSTRAP_REPLICATES:
        warnings.append("replicate_capability_missing")
    if zero_variance:
        warnings.append("zero_variance_descriptive_only")
    interval_complete = len(macro_deltas) == GCS_BOOTSTRAP_REPLICATES
    return S1GCSContrast(
        derived_seed=derived_seed,
        draw_stream_sha256=draw_hasher.hexdigest(),
        query_count=len(ordered_ids),
        component_count=population.component_count,
        macro_delta_pp=all_point_values[0],
        macro_ci95_low_pp=(
            _percentile(macro_deltas, 0.025) if interval_complete else None
        ),
        macro_ci95_high_pp=(
            _percentile(macro_deltas, 0.975) if interval_complete else None
        ),
        macro_available_replicates=len(macro_deltas),
        query_micro_delta_pp=all_point_values[1],
        hard_error_delta_pp=all_point_values[2],
        per_capability=point_per_capability,
        baseline_oracle_coverage_complete=all(
            item.oracle_available for item in baseline
        ),
        candidate_oracle_coverage_complete=all(
            item.oracle_available for item in candidate
        ),
        zero_variance=zero_variance,
        precision_warnings=tuple(warnings),
    )


def evaluate_s1_development_composite(
    *,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    response_contract_diagnostics: S1Round2ResponseContractDiagnostics | None = None,
) -> S1DevelopmentCompositeReport:
    """Evaluate the parent-composed candidate without creating acceptance evidence."""

    ordered_queries = tuple(sorted(queries, key=lambda item: item.query_id))
    if (
        not ordered_queries
        or len({item.query_id for item in ordered_queries}) != len(ordered_queries)
        or any(item.split != "opt_pool" for item in ordered_queries)
        or {item.canonical_capability for item in ordered_queries}
        != set(GCS_CAPABILITY_ORDER)
    ):
        raise S1GCSGateError(
            "development composite requires unique rows across all capabilities"
        )
    baseline = _strict_scores(baseline_scores, label="baseline scores")
    candidate = _strict_scores(candidate_scores, label="candidate scores")
    diagnostics = (
        None
        if response_contract_diagnostics is None
        else _verified_round2_contract_diagnostics(
            response_contract_diagnostics, queries=ordered_queries
        )
    )
    if any(item.config != "llm_static" for item in baseline) or any(
        item.config != "s1" for item in candidate
    ):
        raise S1GCSGateError("development composite requires llm_static and s1 scores")
    contrast = _paired_gcs_v2_contrast(
        baseline=baseline,
        candidate=candidate,
        queries=ordered_queries,
        scope="s1-development-composite",
    )
    safety = _paired_safety(
        queries=ordered_queries,
        baseline=baseline,
        candidate=candidate,
        response_contract_diagnostics=diagnostics,
    )
    flags = {
        "macro_nonnegative": contrast.macro_delta_pp >= 0.0,
        "all_capabilities_nonnegative": all(
            item.point_delta_pp >= 0.0 for item in contrast.per_capability
        ),
        "hard_error_nonincreasing": contrast.hard_error_delta_pp <= 0.0,
        "paired_success_preserved": (
            safety.static_success_to_candidate_failure_count == 0
        ),
        "contract_reasons_preserved": (
            safety.new_contract_reason_occurrence_count == 0
        ),
    }
    reason_map = {
        "macro_nonnegative": "macro_negative",
        "all_capabilities_nonnegative": "capability_negative",
        "hard_error_nonincreasing": "hard_error_increased",
        "paired_success_preserved": "paired_static_success_regressed",
        "contract_reasons_preserved": "contract_reason_regressed",
    }
    reasons = tuple(
        sorted(reason_map[key] for key, value in flags.items() if not value)
    )
    population = build_gcs_population_v2(ordered_queries)
    query_ids = tuple(item.query_id for item in ordered_queries)
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-development-composite-report",
        "policy_version": "portfolio-s1-development-composite-v1",
        "purpose": "development_only_not_acceptance",
        "query_count": len(query_ids),
        "query_ids": list(query_ids),
        "query_ids_sha256": _query_ids_sha256(query_ids),
        "population_mapping_sha256": population.population_mapping_sha256,
        "baseline_scores_sha256": _hash_payload(
            [
                item.model_dump(mode="json")
                for item in sorted(baseline, key=lambda value: value.query_id)
            ]
        ),
        "candidate_scores_sha256": _hash_payload(
            [
                item.model_dump(mode="json")
                for item in sorted(candidate, key=lambda value: value.query_id)
            ]
        ),
        **(
            {}
            if diagnostics is None
            else {
                "response_contract_diagnostics_sha256": (
                    diagnostics.diagnostics_sha256
                )
            }
        ),
        "contrast": contrast.model_dump(mode="json"),
        "paired_safety": safety.model_dump(mode="json"),
        **flags,
        "passed": not reasons,
        "reason_codes": list(reasons),
    }
    try:
        return S1DevelopmentCompositeReport.model_validate(
            {**payload, "report_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("development composite report is invalid") from error


def _criteria_for(
    phase: S1GCSGatePhase,
    contrast: S1GCSContrast,
    *,
    available: bool,
) -> tuple[S1GCSGateCriterion, ...]:
    per_capability = {
        item.capability_id: item.point_delta_pp for item in contrast.per_capability
    }
    if phase == "replay":
        raw: list[tuple[str, Literal[">=", "<="], float, float | None]] = [
            (
                "macro_delta_pp",
                ">=",
                S1_REPLAY_MACRO_DELTA_PP_MIN,
                contrast.macro_delta_pp,
            ),
            (
                "hard_error_delta_pp",
                "<=",
                S1_REPLAY_HARD_ERROR_DELTA_PP_MAX,
                contrast.hard_error_delta_pp,
            ),
        ]
        capability_threshold = S1_REPLAY_PER_CAPABILITY_DELTA_PP_MIN
    else:
        raw = [
            (
                "macro_delta_pp",
                ">=",
                SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
                contrast.macro_delta_pp,
            ),
            (
                "macro_ci95_low_pp",
                ">=",
                SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
                contrast.macro_ci95_low_pp,
            ),
            (
                "hard_error_delta_pp",
                "<=",
                SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
                contrast.hard_error_delta_pp,
            ),
        ]
        capability_threshold = SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN
    raw.extend(
        (
            f"capability_delta_pp:{capability}",
            ">=",
            capability_threshold,
            per_capability[capability],
        )
        for capability in GCS_CAPABILITY_ORDER
    )
    resolved = available and all(item[3] is not None for item in raw)
    return tuple(
        S1GCSGateCriterion(
            name=name,
            operator=operator,
            threshold_pp=threshold,
            observed_pp=observed,
            passed=(
                None
                if not resolved
                else observed >= threshold
                if operator == ">="
                else observed <= threshold
            ),
        )
        for name, operator, threshold, observed in raw
    )


def evaluate_s1_gcs_gate(
    *,
    phase: S1GCSGatePhase,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    evidence: S1GCSEvidenceBinding,
    parent_bank_sha256: str,
    candidate_bank_sha256: str,
) -> S1GCSGateReport:
    """Evaluate one complete replay200 or body_gate75 paired population."""

    try:
        verified_evidence = S1GCSEvidenceBinding.model_validate(
            evidence.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 GCS evidence binding is invalid") from error
    if verified_evidence.phase != phase:
        raise S1GCSGateError("S1 GCS evidence binding has the wrong phase")
    ordered_queries = tuple(sorted(queries, key=lambda item: item.query_id))
    if len(ordered_queries) != _PHASE_QUERY_COUNT[phase] or len(
        {item.query_id for item in ordered_queries}
    ) != len(ordered_queries):
        raise S1GCSGateError("S1 GCS gate population has the wrong size or duplicates")
    if any(item.split == "test_frozen" for item in ordered_queries):
        raise S1GCSGateError("S1 GCS gate must never consume test_frozen")
    if any(item.split != _PHASE_SPLIT[phase] for item in ordered_queries):
        raise S1GCSGateError("S1 GCS gate population comes from the wrong split")
    if {item.canonical_capability for item in ordered_queries} != set(
        GCS_CAPABILITY_ORDER
    ):
        raise S1GCSGateError("S1 GCS gate must cover all six capabilities")
    baseline = _strict_scores(baseline_scores, label="baseline scores")
    candidate = _strict_scores(candidate_scores, label="candidate scores")
    scope = _PHASE_SCOPE[phase]
    try:
        baseline_summary = summarize_gcs_v2(
            baseline, ordered_queries, "llm_static", scope
        )
        candidate_summary = summarize_gcs_v2(candidate, ordered_queries, "s1", scope)
        contrast = _paired_gcs_v2_contrast(
            baseline=baseline,
            candidate=candidate,
            queries=ordered_queries,
            scope=scope,
        )
        population = build_gcs_population_v2(ordered_queries)
    except (PortfolioGCSError, ValidationError, KeyError) as error:
        raise S1GCSGateError("S1 GCS paired population integrity failed") from error
    coverage = (
        baseline_summary.headline_available
        and candidate_summary.headline_available
        and contrast.baseline_oracle_coverage_complete
        and contrast.candidate_oracle_coverage_complete
    )
    criteria = _criteria_for(
        phase,
        contrast,
        available=coverage,
    )
    unavailable_reasons: set[str] = set()
    if not baseline_summary.headline_available:
        unavailable_reasons.add("baseline_oracle_coverage_incomplete")
    if not candidate_summary.headline_available:
        unavailable_reasons.add("candidate_oracle_coverage_incomplete")
    if any(item.observed_pp is None for item in criteria):
        unavailable_reasons.add("criterion_unavailable")
    if unavailable_reasons:
        status: S1GCSGateStatus = "unavailable"
        reasons = unavailable_reasons
    elif all(item.passed for item in criteria):
        status = "passed"
        reasons = set()
    else:
        status = "failed"
        reasons = {
            f"threshold_failed:{item.name}" for item in criteria if item.passed is False
        }
    query_ids = tuple(item.query_id for item in ordered_queries)
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-gate-report",
        "policy_version": S1_GCS_GATE_POLICY_VERSION,
        "policy_sha256": S1_GCS_GATE_POLICY_SHA256,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "phase": phase,
        "scope": scope,
        "source_split": _PHASE_SPLIT[phase],
        "decision_rule": _PHASE_RULE[phase],
        "evidence": verified_evidence.model_dump(mode="json"),
        "query_count": len(query_ids),
        "component_count": population.component_count,
        "query_ids": list(query_ids),
        "query_ids_sha256": _query_ids_sha256(query_ids),
        "population_mapping_sha256": population.population_mapping_sha256,
        "baseline_config": "llm_static",
        "candidate_config": "s1",
        "parent_bank_sha256": parent_bank_sha256,
        "candidate_bank_sha256": candidate_bank_sha256,
        "baseline_summary": baseline_summary.model_dump(mode="json"),
        "candidate_summary": candidate_summary.model_dump(mode="json"),
        "contrast": contrast.model_dump(mode="json"),
        "criteria": [item.model_dump(mode="json") for item in criteria],
        "coverage_complete": coverage,
        "integrity_complete": True,
        "status": status,
        "reason_codes": sorted(reasons),
        "pairwise_judge_call_count": 0,
        "legacy_final_judge_call_count": 0,
        "provider_model_call_count": 0,
    }
    try:
        return S1GCSGateReport.model_validate(
            {**payload, "report_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 GCS gate report is invalid") from error


def evaluate_s1_replay(
    *,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    evidence: S1GCSEvidenceBinding,
    parent_bank_sha256: str,
    candidate_bank_sha256: str,
) -> S1GCSGateReport:
    return evaluate_s1_gcs_gate(
        phase="replay",
        queries=queries,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        evidence=evidence,
        parent_bank_sha256=parent_bank_sha256,
        candidate_bank_sha256=candidate_bank_sha256,
    )


def evaluate_s1_body_gate(
    *,
    replay_report: S1GCSGateReport,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    evidence: S1GCSEvidenceBinding,
    parent_bank_sha256: str,
    candidate_bank_sha256: str,
) -> S1GCSGateReport:
    try:
        replay = S1GCSGateReport.model_validate(
            replay_report.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 replay report is invalid") from error
    if replay.phase != "replay" or replay.status != "passed":
        raise S1GCSGateError("body_gate75 requires a passed replay200 report")
    if (
        replay.parent_bank_sha256 != parent_bank_sha256
        or replay.candidate_bank_sha256 != candidate_bank_sha256
    ):
        raise S1GCSGateError("body gate Banks differ from replay Banks")
    body_query_ids = {item.query_id for item in queries}
    if body_query_ids & set(replay.query_ids):
        raise S1GCSGateError("body_gate75 and replay200 query populations overlap")
    return evaluate_s1_gcs_gate(
        phase="body_gate",
        queries=queries,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        evidence=evidence,
        parent_bank_sha256=parent_bank_sha256,
        candidate_bank_sha256=candidate_bank_sha256,
    )


def _verified_development_report(
    value: S1DevelopmentCompositeReport,
) -> S1DevelopmentCompositeReport:
    if type(value) is not S1DevelopmentCompositeReport:
        raise S1GCSGateError("S1 development composite has the wrong type")
    try:
        return S1DevelopmentCompositeReport.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 development composite is invalid") from error


def _verified_round2_screen(
    value: S1Round2DevelopmentScreen,
) -> S1Round2DevelopmentScreen:
    if type(value) is not S1Round2DevelopmentScreen:
        raise S1GCSGateError("S1 Round 2 development screen has the wrong type")
    try:
        return S1Round2DevelopmentScreen.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 development screen is invalid") from error


def _verified_round2_freeze(
    value: S1Round2CandidateFreeze,
) -> S1Round2CandidateFreeze:
    if type(value) is not S1Round2CandidateFreeze:
        raise S1GCSGateError("S1 Round 2 candidate freeze has the wrong type")
    try:
        return S1Round2CandidateFreeze.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 candidate freeze is invalid") from error


def make_s1_round2_candidate_freeze(
    *,
    development_report: S1DevelopmentCompositeReport,
    development_screen: S1Round2DevelopmentScreen,
    development_screen_file_sha256: str,
    screened_bank_receipt_sha256: str,
    screened_bank_receipt_file_sha256: str,
    retained_capability_ids: Sequence[str],
    parent_bank_sha256: str,
    parent_bank_file_sha256: str,
    candidate_bank_sha256: str,
    candidate_bank_file_sha256: str,
    runtime_lock_sha256: str,
    runtime_lock_file_sha256: str,
) -> S1Round2CandidateFreeze:
    """Freeze exactly one passed development candidate before body data access."""

    development = _verified_development_report(development_report)
    screen = _verified_round2_screen(development_screen)
    if not development.passed:
        raise S1GCSGateError("Round 2 candidate freeze requires passed development")
    if development.response_contract_diagnostics_sha256 is None:
        raise S1GCSGateError(
            "Round 2 candidate freeze requires checkpoint contract diagnostics"
        )
    if (
        development.query_count != S1_ROUND2_DEVELOPMENT_QUERY_COUNT
        or development.query_ids != screen.query_ids
        or development.query_ids_sha256 != screen.query_ids_sha256
        or development.population_mapping_sha256 != screen.population_mapping_sha256
    ):
        raise S1GCSGateError(
            "Round 2 development composite differs from its raw patch screen"
        )
    retained = tuple(sorted(set(retained_capability_ids)))
    if tuple(retained_capability_ids) != retained or not set(retained).issubset(
        GCS_CAPABILITY_ORDER
    ):
        raise S1GCSGateError("Round 2 retained capability identity is invalid")
    if (
        parent_bank_sha256 != screen.parent_bank_sha256
        or development_screen_file_sha256 != sha256_bytes(screen.canonical_bytes())
    ):
        raise S1GCSGateError("Round 2 screen or parent identity drifted before freeze")
    payload = {
        "schema_version": 2,
        "artifact_kind": "portfolio-s1-round2-candidate-freeze",
        "policy_version": S1_ROUND2_BODY_GATE_POLICY_VERSION,
        "policy_sha256": S1_ROUND2_BODY_GATE_POLICY_SHA256,
        "gate_policy_version": S1_GCS_GATE_POLICY_VERSION,
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
        "development_report_sha256": development.report_sha256,
        "development_report_file_sha256": sha256_bytes(development.canonical_bytes()),
        "development_screen_sha256": screen.screen_sha256,
        "development_screen_file_sha256": development_screen_file_sha256,
        "screened_bank_receipt_sha256": screened_bank_receipt_sha256,
        "screened_bank_receipt_file_sha256": screened_bank_receipt_file_sha256,
        "retained_capability_ids": list(retained),
        "parent_bank_sha256": parent_bank_sha256,
        "parent_bank_file_sha256": parent_bank_file_sha256,
        "candidate_bank_sha256": candidate_bank_sha256,
        "candidate_bank_file_sha256": candidate_bank_file_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "runtime_lock_file_sha256": runtime_lock_file_sha256,
        "body_gate_access_count": 1,
    }
    try:
        return S1Round2CandidateFreeze.model_validate(
            {**payload, "freeze_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 candidate freeze is invalid") from error


def evaluate_s1_round2_body_gate(
    *,
    development_report: S1DevelopmentCompositeReport,
    candidate_freeze: S1Round2CandidateFreeze,
    queries: Sequence[Query],
    baseline_scores: Sequence[GCSQueryScoreV2],
    candidate_scores: Sequence[GCSQueryScoreV2],
    response_contract_diagnostics: S1Round2ResponseContractDiagnostics,
    evidence: S1GCSEvidenceBinding,
    parent_bank_sha256: str,
    candidate_bank_sha256: str,
    runtime_lock_sha256: str,
    runtime_lock_file_sha256: str,
) -> S1Round2BodyGateReport:
    """Evaluate Round 2 body_gate75 without accepting a replay-report surrogate."""

    development = _verified_development_report(development_report)
    freeze = _verified_round2_freeze(candidate_freeze)
    if not development.passed:
        raise S1GCSGateError("Round 2 body gate requires passed development")
    if (
        freeze.development_report_sha256 != development.report_sha256
        or freeze.development_report_file_sha256
        != sha256_bytes(development.canonical_bytes())
    ):
        raise S1GCSGateError("Round 2 candidate freeze differs from development")
    if (
        parent_bank_sha256 != freeze.parent_bank_sha256
        or candidate_bank_sha256 != freeze.candidate_bank_sha256
        or runtime_lock_sha256 != freeze.runtime_lock_sha256
        or runtime_lock_file_sha256 != freeze.runtime_lock_file_sha256
    ):
        raise S1GCSGateError(
            "Round 2 candidate or runtime identity changed after freeze"
        )
    if (
        evidence.parent_bank_file_sha256 != freeze.parent_bank_file_sha256
        or evidence.candidate_bank_file_sha256 != freeze.candidate_bank_file_sha256
    ):
        raise S1GCSGateError("Round 2 body evidence Bank files differ from freeze")
    if {item.query_id for item in queries} & set(development.query_ids):
        raise S1GCSGateError("Round 2 body and development populations overlap")
    diagnostics = _verified_round2_contract_diagnostics(
        response_contract_diagnostics,
        queries=tuple(sorted(queries, key=lambda item: item.query_id)),
    )
    body = evaluate_s1_gcs_gate(
        phase="body_gate",
        queries=queries,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        evidence=evidence,
        parent_bank_sha256=parent_bank_sha256,
        candidate_bank_sha256=candidate_bank_sha256,
    )
    paired_safety = _paired_safety(
        queries=tuple(sorted(queries, key=lambda item: item.query_id)),
        baseline=_strict_scores(baseline_scores, label="baseline scores"),
        candidate=_strict_scores(candidate_scores, label="candidate scores"),
        response_contract_diagnostics=diagnostics,
    )
    paired_preserved = paired_safety.static_success_to_candidate_failure_count == 0
    contract_preserved = paired_safety.new_contract_reason_occurrence_count == 0
    reasons = set(body.reason_codes)
    if not paired_preserved:
        reasons.add("threshold_failed:static_success_to_candidate_failure_count")
    if not contract_preserved:
        reasons.add("threshold_failed:new_contract_reason_occurrence_count")
    if body.status == "unavailable":
        status: S1GCSGateStatus = "unavailable"
    elif body.status == "passed" and paired_preserved and contract_preserved:
        status = "passed"
    else:
        status = "failed"
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-round2-body-gate-report",
        "policy_version": S1_ROUND2_BODY_GATE_POLICY_VERSION,
        "policy_sha256": S1_ROUND2_BODY_GATE_POLICY_SHA256,
        "development_report": development.model_dump(mode="json"),
        "candidate_freeze": freeze.model_dump(mode="json"),
        "body_gate_report": body.model_dump(mode="json"),
        "response_contract_diagnostics_sha256": diagnostics.diagnostics_sha256,
        "paired_safety": paired_safety.model_dump(mode="json"),
        "paired_success_preserved": paired_preserved,
        "contract_reasons_preserved": contract_preserved,
        "status": status,
        "reason_codes": sorted(reasons),
        "provider_model_call_count": 0,
    }
    try:
        return S1Round2BodyGateReport.model_validate(
            {**payload, "report_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 body gate report is invalid") from error


class S1BankDispositionReceipt(_StrictFrozenModel):
    """Final whole-Bank S1 acceptance or byte-exact rollback receipt."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-bank-disposition-receipt"] = (
        "portfolio-s1-bank-disposition-receipt"
    )
    policy_version: Literal[S1_GCS_GATE_POLICY_VERSION] = S1_GCS_GATE_POLICY_VERSION
    policy_sha256: Literal[S1_GCS_GATE_POLICY_SHA256] = S1_GCS_GATE_POLICY_SHA256
    stage: Literal["s1_creator"] = "s1_creator"
    parent_bank_sha256: Sha256
    parent_bank_file_sha256: Sha256
    candidate_bank_sha256: Sha256
    candidate_bank_file_sha256: Sha256
    replay_report_sha256: Sha256
    replay_report_file_sha256: Sha256
    replay_status: S1GCSGateStatus
    body_gate_report_sha256: Sha256 | None
    body_gate_report_file_sha256: Sha256 | None
    body_gate_status: S1GCSGateStatus | None
    decision_basis: Literal["replay200", "body_gate75"]
    decision: S1BankDecision
    output_bank_sha256: Sha256
    output_bank_file: Literal["output-bank.json"] = "output-bank.json"
    output_bank_file_sha256: Sha256
    rejected_candidate_use: Literal["diagnostic_only"] = "diagnostic_only"
    pairwise_judge_call_count: Literal[0] = 0
    legacy_final_judge_call_count: Literal[0] = 0
    provider_model_call_count: Literal[0] = 0
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        body_fields = (
            self.body_gate_report_sha256,
            self.body_gate_report_file_sha256,
            self.body_gate_status,
        )
        if (self.replay_status == "passed") != all(
            item is not None for item in body_fields
        ):
            raise ValueError("S1 body gate presence differs from replay disposition")
        if self.replay_status == "passed":
            if self.decision_basis != "body_gate75":
                raise ValueError("passed replay must defer to body_gate75")
            expected: S1BankDecision = (
                "accepted" if self.body_gate_status == "passed" else "rolled_back"
            )
        else:
            if self.decision_basis != "replay200":
                raise ValueError("failed replay must be the rollback basis")
            expected = "rolled_back"
        if self.decision != expected:
            raise ValueError("S1 Bank decision differs from gate statuses")
        expected_bank = (
            self.candidate_bank_sha256
            if self.decision == "accepted"
            else self.parent_bank_sha256
        )
        expected_file = (
            self.candidate_bank_file_sha256
            if self.decision == "accepted"
            else self.parent_bank_file_sha256
        )
        if (
            self.output_bank_sha256 != expected_bank
            or self.output_bank_file_sha256 != expected_file
        ):
            raise ValueError(
                "S1 output Bank is not a byte-exact accepted/parent artifact"
            )
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("S1 Bank disposition receipt self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class S1BankDispositionPublication:
    receipt: S1BankDispositionReceipt
    output_bank_bytes: bytes


def _verified_report(value: S1GCSGateReport, phase: S1GCSGatePhase) -> S1GCSGateReport:
    if type(value) is not S1GCSGateReport:
        raise S1GCSGateError("S1 GCS report has the wrong type")
    try:
        report = S1GCSGateReport.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 GCS report is invalid") from error
    if report.phase != phase:
        raise S1GCSGateError("S1 GCS report has the wrong phase")
    return report


def _verified_bank_bytes(
    bank: StaticBankArtifact, content: bytes, *, label: str
) -> StaticBankArtifact:
    if type(bank) is not StaticBankArtifact:
        raise S1GCSGateError(f"{label} has the wrong type")
    try:
        verified = StaticBankArtifact.model_validate(
            bank.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError(f"{label} is invalid") from error
    if verified.canonical_bytes() != content:
        raise S1GCSGateError(
            f"{label} bytes are not canonical or differ from the model"
        )
    return verified


def build_s1_bank_disposition(
    *,
    replay_report: S1GCSGateReport,
    body_gate_report: S1GCSGateReport | None,
    parent_bank: StaticBankArtifact,
    parent_bank_bytes: bytes,
    candidate_bank: StaticBankArtifact,
    candidate_bank_bytes: bytes,
) -> S1BankDispositionPublication:
    """Select exact candidate bytes on pass, otherwise exact parent bytes."""

    replay = _verified_report(replay_report, "replay")
    body = (
        None
        if body_gate_report is None
        else _verified_report(body_gate_report, "body_gate")
    )
    parent = _verified_bank_bytes(parent_bank, parent_bank_bytes, label="parent Bank")
    candidate = _verified_bank_bytes(
        candidate_bank, candidate_bank_bytes, label="candidate Bank"
    )
    parent_file_sha = sha256_bytes(parent_bank_bytes)
    candidate_file_sha = sha256_bytes(candidate_bank_bytes)
    for report in (replay, *((body,) if body is not None else ())):
        if (
            report.parent_bank_sha256 != parent.bank_sha256
            or report.candidate_bank_sha256 != candidate.bank_sha256
            or report.evidence.parent_bank_file_sha256 != parent_file_sha
            or report.evidence.candidate_bank_file_sha256 != candidate_file_sha
        ):
            raise S1GCSGateError("S1 gate report Bank binding differs from exact bytes")
    if replay.status == "passed":
        if body is None:
            raise S1GCSGateError(
                "passed replay requires body_gate75 before disposition"
            )
        if set(replay.query_ids) & set(body.query_ids):
            raise S1GCSGateError("replay200 and body_gate75 overlap")
        decision: S1BankDecision = (
            "accepted" if body.status == "passed" else "rolled_back"
        )
        decision_basis: Literal["replay200", "body_gate75"] = "body_gate75"
    else:
        if body is not None:
            raise S1GCSGateError("failed replay forbids consuming a body gate report")
        decision = "rolled_back"
        decision_basis = "replay200"
    output_bytes = candidate_bank_bytes if decision == "accepted" else parent_bank_bytes
    output_bank = candidate if decision == "accepted" else parent
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-bank-disposition-receipt",
        "policy_version": S1_GCS_GATE_POLICY_VERSION,
        "policy_sha256": S1_GCS_GATE_POLICY_SHA256,
        "stage": "s1_creator",
        "parent_bank_sha256": parent.bank_sha256,
        "parent_bank_file_sha256": parent_file_sha,
        "candidate_bank_sha256": candidate.bank_sha256,
        "candidate_bank_file_sha256": candidate_file_sha,
        "replay_report_sha256": replay.report_sha256,
        "replay_report_file_sha256": sha256_bytes(replay.canonical_bytes()),
        "replay_status": replay.status,
        "body_gate_report_sha256": None if body is None else body.report_sha256,
        "body_gate_report_file_sha256": (
            None if body is None else sha256_bytes(body.canonical_bytes())
        ),
        "body_gate_status": None if body is None else body.status,
        "decision_basis": decision_basis,
        "decision": decision,
        "output_bank_sha256": output_bank.bank_sha256,
        "output_bank_file": "output-bank.json",
        "output_bank_file_sha256": sha256_bytes(output_bytes),
        "rejected_candidate_use": "diagnostic_only",
        "pairwise_judge_call_count": 0,
        "legacy_final_judge_call_count": 0,
        "provider_model_call_count": 0,
    }
    try:
        receipt = S1BankDispositionReceipt.model_validate(
            {**payload, "receipt_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Bank disposition receipt is invalid") from error
    return S1BankDispositionPublication(
        receipt=receipt,
        output_bank_bytes=output_bytes,
    )


class S1Round2BankDispositionReceipt(_StrictFrozenModel):
    """Whole-Bank disposition for the forward-only Round 2 body gate."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-round2-bank-disposition-receipt"] = (
        "portfolio-s1-round2-bank-disposition-receipt"
    )
    policy_version: Literal[S1_ROUND2_BODY_GATE_POLICY_VERSION] = (
        S1_ROUND2_BODY_GATE_POLICY_VERSION
    )
    policy_sha256: Literal[S1_ROUND2_BODY_GATE_POLICY_SHA256] = (
        S1_ROUND2_BODY_GATE_POLICY_SHA256
    )
    parent_bank_sha256: Sha256
    parent_bank_file_sha256: Sha256
    candidate_bank_sha256: Sha256
    candidate_bank_file_sha256: Sha256
    runtime_lock_sha256: Sha256
    runtime_lock_file_sha256: Sha256
    development_report_sha256: Sha256
    candidate_freeze_sha256: Sha256
    body_gate_report_sha256: Sha256
    body_gate_report_file_sha256: Sha256
    body_gate_status: S1GCSGateStatus
    decision_basis: Literal["body_gate75"] = "body_gate75"
    decision: S1BankDecision
    output_bank_sha256: Sha256
    output_bank_file: Literal["output-bank.json"] = "output-bank.json"
    output_bank_file_sha256: Sha256
    rejected_candidate_use: Literal["diagnostic_only"] = "diagnostic_only"
    provider_model_call_count: Literal[0] = 0
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        expected: S1BankDecision = (
            "accepted" if self.body_gate_status == "passed" else "rolled_back"
        )
        if self.decision != expected:
            raise ValueError("Round 2 Bank decision differs from body gate status")
        expected_bank = (
            self.candidate_bank_sha256
            if expected == "accepted"
            else self.parent_bank_sha256
        )
        expected_file = (
            self.candidate_bank_file_sha256
            if expected == "accepted"
            else self.parent_bank_file_sha256
        )
        if (
            self.output_bank_sha256 != expected_bank
            or self.output_bank_file_sha256 != expected_file
        ):
            raise ValueError("Round 2 output Bank is not the exact selected artifact")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("Round 2 disposition receipt self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class S1Round2BankDispositionPublication:
    receipt: S1Round2BankDispositionReceipt
    output_bank_bytes: bytes


def _verified_round2_body_report(
    value: S1Round2BodyGateReport,
) -> S1Round2BodyGateReport:
    if type(value) is not S1Round2BodyGateReport:
        raise S1GCSGateError("S1 Round 2 body gate report has the wrong type")
    try:
        return S1Round2BodyGateReport.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 body gate report is invalid") from error


def build_s1_round2_bank_disposition(
    *,
    body_gate_report: S1Round2BodyGateReport,
    parent_bank: StaticBankArtifact,
    parent_bank_bytes: bytes,
    candidate_bank: StaticBankArtifact,
    candidate_bank_bytes: bytes,
) -> S1Round2BankDispositionPublication:
    """Publish exact candidate bytes on Round 2 pass, exact parent bytes otherwise."""

    report = _verified_round2_body_report(body_gate_report)
    parent = _verified_bank_bytes(parent_bank, parent_bank_bytes, label="parent Bank")
    candidate = _verified_bank_bytes(
        candidate_bank, candidate_bank_bytes, label="candidate Bank"
    )
    freeze = report.candidate_freeze
    parent_file_sha = sha256_bytes(parent_bank_bytes)
    candidate_file_sha = sha256_bytes(candidate_bank_bytes)
    if (
        parent.bank_sha256 != freeze.parent_bank_sha256
        or parent_file_sha != freeze.parent_bank_file_sha256
        or candidate.bank_sha256 != freeze.candidate_bank_sha256
        or candidate_file_sha != freeze.candidate_bank_file_sha256
    ):
        raise S1GCSGateError("Round 2 disposition Bank bytes differ from freeze")
    decision: S1BankDecision = (
        "accepted" if report.status == "passed" else "rolled_back"
    )
    output_bank = candidate if decision == "accepted" else parent
    output_bytes = candidate_bank_bytes if decision == "accepted" else parent_bank_bytes
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-round2-bank-disposition-receipt",
        "policy_version": S1_ROUND2_BODY_GATE_POLICY_VERSION,
        "policy_sha256": S1_ROUND2_BODY_GATE_POLICY_SHA256,
        "parent_bank_sha256": parent.bank_sha256,
        "parent_bank_file_sha256": parent_file_sha,
        "candidate_bank_sha256": candidate.bank_sha256,
        "candidate_bank_file_sha256": candidate_file_sha,
        "runtime_lock_sha256": freeze.runtime_lock_sha256,
        "runtime_lock_file_sha256": freeze.runtime_lock_file_sha256,
        "development_report_sha256": report.development_report.report_sha256,
        "candidate_freeze_sha256": freeze.freeze_sha256,
        "body_gate_report_sha256": report.report_sha256,
        "body_gate_report_file_sha256": sha256_bytes(report.canonical_bytes()),
        "body_gate_status": report.status,
        "decision_basis": "body_gate75",
        "decision": decision,
        "output_bank_sha256": output_bank.bank_sha256,
        "output_bank_file": "output-bank.json",
        "output_bank_file_sha256": sha256_bytes(output_bytes),
        "rejected_candidate_use": "diagnostic_only",
        "provider_model_call_count": 0,
    }
    try:
        receipt = S1Round2BankDispositionReceipt.model_validate(
            {**payload, "receipt_sha256": _hash_payload(payload)}, strict=True
        )
    except ValidationError as error:
        raise S1GCSGateError("S1 Round 2 Bank disposition is invalid") from error
    return S1Round2BankDispositionPublication(
        receipt=receipt,
        output_bank_bytes=output_bytes,
    )


__all__ = [
    "S1_BODY_GATE_QUERY_COUNT",
    "S1_BODY_GATE_SCOPE",
    "S1_GCS_GATE_POLICY_SHA256",
    "S1_GCS_GATE_POLICY_VERSION",
    "S1_ROUND2_BODY_GATE_POLICY_SHA256",
    "S1_ROUND2_BODY_GATE_POLICY_VERSION",
    "S1_ROUND2_DEVELOPMENT_QUERY_COUNT",
    "S1_ROUND2_DEVELOPMENT_SCREEN_POLICY_VERSION",
    "S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS",
    "S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION",
    "S1_CONTRACT_REGRESSION_REASON_CODES",
    "S1_REPLAY_QUERY_COUNT",
    "S1_REPLAY_SCOPE",
    "S1BankDispositionPublication",
    "S1BankDispositionReceipt",
    "S1Round2BankDispositionPublication",
    "S1Round2BankDispositionReceipt",
    "S1Round2BodyGateReport",
    "S1Round2CandidateFreeze",
    "S1Round2DevelopmentScreen",
    "S1Round2ResponseContractDiagnostic",
    "S1Round2ResponseContractDiagnostics",
    "S1GCSContrast",
    "S1GCSCapabilitySafety",
    "S1GCSContractReasonSafety",
    "S1GCSPairedSafety",
    "S1DevelopmentCapabilityScreen",
    "S1DevelopmentCompositeReport",
    "S1DevelopmentScreen",
    "S1GCSEvidenceBinding",
    "S1GCSGateError",
    "S1GCSGateReport",
    "build_s1_bank_disposition",
    "build_s1_round2_bank_disposition",
    "evaluate_s1_body_gate",
    "evaluate_s1_development_composite",
    "evaluate_s1_gcs_gate",
    "evaluate_s1_replay",
    "evaluate_s1_round2_body_gate",
    "make_s1_gcs_evidence_binding",
    "make_s1_round2_candidate_freeze",
    "make_s1_round2_response_contract_diagnostic",
    "make_s1_round2_response_contract_diagnostics",
    "screen_s1_development_patches",
    "screen_s1_round2_development_patches",
]
