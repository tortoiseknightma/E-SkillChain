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
    raise RuntimeError("tracked S1 GCS gate policy changed without an identity revision")


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
            self.macro_ci95_low_pp is not None
            and self.macro_ci95_high_pp is not None
        )
        if interval_present != (
            self.macro_available_replicates == self.replicates
        ):
            raise ValueError("S1 GCS macro interval availability is inconsistent")
        if (self.macro_ci95_low_pp is None) != (
            self.macro_ci95_high_pp is None
        ):
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
                or summary.population_mapping_sha256
                != self.population_mapping_sha256
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
            f"capability_delta_pp:{capability}"
            for capability in GCS_CAPABILITY_ORDER
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
            * (
                candidate_point[3][capability]
                - baseline_point[3][capability]
            ),
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
    if (
        len(ordered_queries) != _PHASE_QUERY_COUNT[phase]
        or len({item.query_id for item in ordered_queries}) != len(ordered_queries)
    ):
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
    criteria = _criteria_for(phase, contrast, available=coverage)
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
            f"threshold_failed:{item.name}"
            for item in criteria
            if item.passed is False
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
            raise ValueError("S1 output Bank is not a byte-exact accepted/parent artifact")
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
        raise S1GCSGateError(f"{label} bytes are not canonical or differ from the model")
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
            raise S1GCSGateError("passed replay requires body_gate75 before disposition")
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


__all__ = [
    "S1_BODY_GATE_QUERY_COUNT",
    "S1_BODY_GATE_SCOPE",
    "S1_GCS_GATE_POLICY_SHA256",
    "S1_GCS_GATE_POLICY_VERSION",
    "S1_REPLAY_QUERY_COUNT",
    "S1_REPLAY_SCOPE",
    "S1BankDispositionPublication",
    "S1BankDispositionReceipt",
    "S1GCSContrast",
    "S1GCSEvidenceBinding",
    "S1GCSGateError",
    "S1GCSGateReport",
    "build_s1_bank_disposition",
    "evaluate_s1_body_gate",
    "evaluate_s1_gcs_gate",
    "evaluate_s1_replay",
    "make_s1_gcs_evidence_binding",
]
