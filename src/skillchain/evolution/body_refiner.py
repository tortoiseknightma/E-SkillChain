"""Minimal, typed S3 body refinement and rollback gate.

S3 receives the same complete skill surface proof as S2, but the allowed edit
surface is the inverse: only ``body`` may change.  It has no split-file or
runner access; callers must supply typed opt-pool examples and body-gate rows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from skillchain.evaluation.packets import AssistantRunConfig
from skillchain.evolution.attribution import (
    AttributionError,
    EvolutionExample,
    Sha256,
    body_opt_pool_examples,
)
from skillchain.evolution.models import CoreTraceRecord
from skillchain.evolution.route_optimizer import RouteSkillSurface
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


class BodyRefinementError(ValueError):
    """Raised when S3 evidence, candidate scope, or gate input is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash_payload(payload: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(payload)))


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field_name}))


def _identity_hash(kind: str, value: str) -> str:
    return _hash_payload({"kind": kind, "value": value})


def _verified_surface(value: RouteSkillSurface) -> RouteSkillSurface:
    if type(value) is not RouteSkillSurface:
        raise TypeError("skill surface must be a RouteSkillSurface")
    try:
        return RouteSkillSurface.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise BodyRefinementError("skill surface self hash or invariants are invalid") from error


def _verified_candidate(value: "BodyRefinementCandidate") -> "BodyRefinementCandidate":
    if type(value) is not BodyRefinementCandidate:
        raise TypeError("candidate must be a BodyRefinementCandidate")
    try:
        return BodyRefinementCandidate.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise BodyRefinementError("body candidate self hash or invariants are invalid") from error


def _surface_field_sha256(
    surfaces: tuple[RouteSkillSurface, ...], field_name: str
) -> str:
    return _hash_payload(
        [
            {"slug": item.slug, "value": _jsonable(getattr(item, field_name))}
            for item in surfaces
        ]
    )


def _surface_set_sha256(surfaces: tuple[RouteSkillSurface, ...]) -> str:
    return _hash_payload([item.model_dump(mode="json") for item in surfaces])


def _examples_sha256(examples: tuple[EvolutionExample, ...]) -> str:
    return _hash_payload([item.example_sha256 for item in examples])


def _validate_surface_sets(
    baseline: tuple[RouteSkillSurface, ...], candidate: tuple[RouteSkillSurface, ...]
) -> None:
    if not baseline:
        raise BodyRefinementError("S3 requires at least one skill surface")
    baseline_identities = tuple((item.slug, item.capability_id) for item in baseline)
    candidate_identities = tuple((item.slug, item.capability_id) for item in candidate)
    if len({item.slug for item in baseline}) != len(baseline) or len(
        {item.slug for item in candidate}
    ) != len(candidate):
        raise BodyRefinementError("S3 skill surfaces must have unique slugs")
    if baseline_identities != candidate_identities:
        raise BodyRefinementError("S3 candidate changes skill identity or capability")
    for before, after in zip(baseline, candidate, strict=True):
        if (
            before.route != after.route
            or before.description != after.description
            or before.static_refs != after.static_refs
            or before.operators != after.operators
        ):
            raise BodyRefinementError(
                "S3 may change only body; route/description/static refs/operators drifted"
            )


class BodyRefinementCandidate(_StrictFrozenModel):
    """A self-hashed S3 candidate with frozen S2 routing evidence."""

    schema_version: Literal[1] = 1
    baseline_bank_sha256: Sha256
    candidate_bank_sha256: Sha256
    baseline_skills: tuple[RouteSkillSurface, ...]
    candidate_skills: tuple[RouteSkillSurface, ...]
    opt_pool_examples: tuple[EvolutionExample, ...]
    opt_pool_examples_sha256: Sha256
    baseline_surface_set_sha256: Sha256
    candidate_surface_set_sha256: Sha256
    frozen_s2_route_sha256: Sha256
    candidate_route_sha256: Sha256
    baseline_description_sha256: Sha256
    candidate_description_sha256: Sha256
    baseline_body_sha256: Sha256
    candidate_body_sha256: Sha256
    baseline_static_refs_sha256: Sha256
    candidate_static_refs_sha256: Sha256
    baseline_operators_sha256: Sha256
    candidate_operators_sha256: Sha256
    candidate_sha256: Sha256

    @field_validator("baseline_skills", "candidate_skills", "opt_pool_examples", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        baseline = tuple(_verified_surface(item) for item in self.baseline_skills)
        candidate = tuple(_verified_surface(item) for item in self.candidate_skills)
        if tuple(item.slug for item in baseline) != tuple(
            sorted(item.slug for item in baseline)
        ):
            raise ValueError("baseline skills must be sorted by slug")
        if tuple(item.slug for item in candidate) != tuple(
            sorted(item.slug for item in candidate)
        ):
            raise ValueError("candidate skills must be sorted by slug")
        _validate_surface_sets(baseline, candidate)
        try:
            examples = body_opt_pool_examples(self.opt_pool_examples)
        except AttributionError as error:
            raise ValueError(str(error)) from error
        if not examples:
            raise ValueError("S3 requires at least one opt_pool body example")
        ordered_examples = tuple(
            sorted(
                examples,
                key=lambda item: (item.component_id, item.query_id, item.trace_sha256),
            )
        )
        if examples != ordered_examples:
            raise ValueError("opt_pool examples must use canonical order")
        expected_hashes = {
            "opt_pool_examples_sha256": _examples_sha256(examples),
            "baseline_surface_set_sha256": _surface_set_sha256(baseline),
            "candidate_surface_set_sha256": _surface_set_sha256(candidate),
            "frozen_s2_route_sha256": _surface_field_sha256(baseline, "route"),
            "candidate_route_sha256": _surface_field_sha256(candidate, "route"),
            "baseline_description_sha256": _surface_field_sha256(
                baseline, "description"
            ),
            "candidate_description_sha256": _surface_field_sha256(
                candidate, "description"
            ),
            "baseline_body_sha256": _surface_field_sha256(baseline, "body"),
            "candidate_body_sha256": _surface_field_sha256(candidate, "body"),
            "baseline_static_refs_sha256": _surface_field_sha256(
                baseline, "static_refs"
            ),
            "candidate_static_refs_sha256": _surface_field_sha256(
                candidate, "static_refs"
            ),
            "baseline_operators_sha256": _surface_field_sha256(baseline, "operators"),
            "candidate_operators_sha256": _surface_field_sha256(candidate, "operators"),
        }
        for field_name, expected in expected_hashes.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} mismatch")
        if not (
            self.frozen_s2_route_sha256 == self.candidate_route_sha256
            and self.baseline_description_sha256 == self.candidate_description_sha256
            and self.baseline_static_refs_sha256 == self.candidate_static_refs_sha256
            and self.baseline_operators_sha256 == self.candidate_operators_sha256
        ):
            raise ValueError(
                "S3 immutable route/description/static refs/operators hashes changed"
            )
        if self.candidate_sha256 != _self_hash(self, "candidate_sha256"):
            raise ValueError("body candidate self hash mismatch")
        return self


def build_body_candidate(
    *,
    baseline_bank_sha256: str,
    candidate_bank_sha256: str,
    baseline_skills: Sequence[RouteSkillSurface],
    candidate_skills: Sequence[RouteSkillSurface],
    opt_pool_examples: Sequence[EvolutionExample],
    s2_route_sha256: str | None = None,
) -> BodyRefinementCandidate:
    """Build a body-only candidate and bind it to frozen S2 route evidence."""

    baseline = tuple(sorted((_verified_surface(item) for item in baseline_skills), key=lambda item: item.slug))
    candidate = tuple(sorted((_verified_surface(item) for item in candidate_skills), key=lambda item: item.slug))
    _validate_surface_sets(baseline, candidate)
    expected_s2_route_sha256 = _surface_field_sha256(baseline, "route")
    if s2_route_sha256 is not None and s2_route_sha256 != expected_s2_route_sha256:
        raise BodyRefinementError("provided S2 route hash does not match the baseline route")
    try:
        examples = body_opt_pool_examples(tuple(opt_pool_examples))
    except AttributionError as error:
        raise BodyRefinementError(str(error)) from error
    examples = tuple(
        sorted(
            examples,
            key=lambda item: (item.component_id, item.query_id, item.trace_sha256),
        )
    )
    if not examples:
        raise BodyRefinementError("S3 requires at least one opt_pool body example")
    unsigned = {
        "schema_version": 1,
        "baseline_bank_sha256": baseline_bank_sha256,
        "candidate_bank_sha256": candidate_bank_sha256,
        "baseline_skills": baseline,
        "candidate_skills": candidate,
        "opt_pool_examples": examples,
        "opt_pool_examples_sha256": _examples_sha256(examples),
        "baseline_surface_set_sha256": _surface_set_sha256(baseline),
        "candidate_surface_set_sha256": _surface_set_sha256(candidate),
        "frozen_s2_route_sha256": expected_s2_route_sha256,
        "candidate_route_sha256": _surface_field_sha256(candidate, "route"),
        "baseline_description_sha256": _surface_field_sha256(
            baseline, "description"
        ),
        "candidate_description_sha256": _surface_field_sha256(
            candidate, "description"
        ),
        "baseline_body_sha256": _surface_field_sha256(baseline, "body"),
        "candidate_body_sha256": _surface_field_sha256(candidate, "body"),
        "baseline_static_refs_sha256": _surface_field_sha256(baseline, "static_refs"),
        "candidate_static_refs_sha256": _surface_field_sha256(candidate, "static_refs"),
        "baseline_operators_sha256": _surface_field_sha256(baseline, "operators"),
        "candidate_operators_sha256": _surface_field_sha256(candidate, "operators"),
    }
    return BodyRefinementCandidate.model_validate(
        {**unsigned, "candidate_sha256": _hash_payload(unsigned)}, strict=True
    )


class BodyGateRow(_StrictFrozenModel):
    """One frozen body-gate result with hard grounding and safety checks."""

    schema_version: Literal[1] = 1
    source_split: Literal["body_gate"] = "body_gate"
    query_id: str
    component_id: str
    run_id: str
    config: AssistantRunConfig
    query_sha256: Sha256
    route_trace_sha256: Sha256
    j_project: float
    grounding_passed: bool
    safety_passed: bool
    gate_row_sha256: Sha256

    @field_validator("query_id", "component_id", "run_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("j_project")
    @classmethod
    def validate_j_project(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("j_project must be finite and within [0, 100]")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        if self.query_sha256 != _identity_hash("query", self.query_id):
            raise ValueError("body gate query identity hash mismatch")
        if self.gate_row_sha256 != _self_hash(self, "gate_row_sha256"):
            raise ValueError("body gate row self hash mismatch")
        return self


def make_body_gate_row(
    *,
    query_id: str,
    component_id: str,
    run_id: str,
    config: AssistantRunConfig,
    route_trace_sha256: str,
    j_project: float,
    grounding_passed: bool,
    safety_passed: bool,
) -> BodyGateRow:
    """Create a body-gate row without opening a test or split artifact."""

    unsigned = {
        "schema_version": 1,
        "source_split": "body_gate",
        "query_id": query_id,
        "component_id": component_id,
        "run_id": run_id,
        "config": config,
        "query_sha256": _identity_hash("query", query_id),
        "route_trace_sha256": route_trace_sha256,
        "j_project": j_project,
        "grounding_passed": grounding_passed,
        "safety_passed": safety_passed,
    }
    return BodyGateRow.model_validate(
        {**unsigned, "gate_row_sha256": _hash_payload(unsigned)}, strict=True
    )


def body_gate_row_from_trace(
    trace: CoreTraceRecord,
    *,
    grounding_passed: bool,
    safety_passed: bool,
) -> BodyGateRow:
    """Adapt a scored, route-correct trace to a body-gate row.

    Grounding and safety are explicit booleans because their deterministic
    rules are project-configurable; callers cannot silently infer a pass from
    an absent rule score.
    """

    try:
        trace = CoreTraceRecord.model_validate(trace.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise BodyRefinementError("body-gate trace is invalid") from error
    if (
        trace.route_correctness != "correct"
        or trace.route_trace_sha256 is None
        or trace.evaluation is None
        or trace.evaluation.judge_outcome is None
        or trace.evaluation.judge_outcome.status != "scored"
    ):
        raise BodyRefinementError("body-gate trace lacks route-correct scored evidence")
    return make_body_gate_row(
        query_id=trace.query_id,
        component_id=trace.component_id,
        run_id=trace.run_id,
        config=trace.config,
        route_trace_sha256=trace.route_trace_sha256,
        j_project=trace.evaluation.judge_outcome.scores.j_project,
        grounding_passed=grounding_passed,
        safety_passed=safety_passed,
    )


def _verified_body_gate_row(value: BodyGateRow) -> BodyGateRow:
    if type(value) is not BodyGateRow:
        raise TypeError("body gate row must be a BodyGateRow")
    try:
        return BodyGateRow.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise BodyRefinementError("body-gate row self hash or invariants are invalid") from error


class BodyGateMetrics(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    row_count: int
    mean_j_project: float
    grounding_pass_rate: float
    safety_pass_rate: float
    metrics_sha256: Sha256

    @field_validator("row_count")
    @classmethod
    def validate_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("body-gate metrics require at least one row")
        return value

    @field_validator("mean_j_project")
    @classmethod
    def validate_j_project(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("mean_j_project must be finite and within [0, 100]")
        return value

    @field_validator("grounding_pass_rate", "safety_pass_rate")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("body-gate pass rate must be finite and within [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.metrics_sha256 != _self_hash(self, "metrics_sha256"):
            raise ValueError("body-gate metrics self hash mismatch")
        return self


def body_gate_metrics(rows: Sequence[BodyGateRow]) -> BodyGateMetrics:
    """Aggregate one frozen body-gate population."""

    verified = tuple(_verified_body_gate_row(item) for item in rows)
    if not verified:
        raise BodyRefinementError("body gate requires at least one row")
    count = len(verified)
    unsigned = {
        "schema_version": 1,
        "row_count": count,
        "mean_j_project": sum(item.j_project for item in verified) / count,
        "grounding_pass_rate": sum(item.grounding_passed for item in verified) / count,
        "safety_pass_rate": sum(item.safety_passed for item in verified) / count,
    }
    return BodyGateMetrics.model_validate(
        {**unsigned, "metrics_sha256": _hash_payload(unsigned)}, strict=True
    )


class RouteTraceComparison(_StrictFrozenModel):
    """Per-query S1+S2/Full route trace parity evidence.

    A mismatch is a valid, self-hashed observation rather than a validation
    error so the S3 gate can retain the evidence and explicitly roll back.
    """

    schema_version: Literal[1] = 1
    query_id: str
    component_id: str
    query_sha256: Sha256
    s1s2_route_trace_sha256: Sha256
    full_route_trace_sha256: Sha256
    matches: bool
    comparison_sha256: Sha256

    @field_validator("query_id", "component_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.query_sha256 != _identity_hash("query", self.query_id):
            raise ValueError("route comparison query identity hash mismatch")
        if self.matches != (
            self.s1s2_route_trace_sha256 == self.full_route_trace_sha256
        ):
            raise ValueError("route comparison match flag is inconsistent")
        if self.comparison_sha256 != _self_hash(self, "comparison_sha256"):
            raise ValueError("route comparison self hash mismatch")
        return self


def make_route_trace_comparison(
    *,
    query_id: str,
    component_id: str,
    s1s2_route_trace_sha256: str,
    full_route_trace_sha256: str,
) -> RouteTraceComparison:
    """Record whether Full reused S1+S2's route trace for one query."""

    unsigned = {
        "schema_version": 1,
        "query_id": query_id,
        "component_id": component_id,
        "query_sha256": _identity_hash("query", query_id),
        "s1s2_route_trace_sha256": s1s2_route_trace_sha256,
        "full_route_trace_sha256": full_route_trace_sha256,
        "matches": s1s2_route_trace_sha256 == full_route_trace_sha256,
    }
    return RouteTraceComparison.model_validate(
        {**unsigned, "comparison_sha256": _hash_payload(unsigned)}, strict=True
    )


def compare_s1s2_full_route_traces(
    s1s2_trace: CoreTraceRecord, full_trace: CoreTraceRecord
) -> RouteTraceComparison:
    """Build parity evidence from two verified traces of the same query."""

    try:
        s1s2_trace = CoreTraceRecord.model_validate(
            s1s2_trace.model_dump(mode="python"), strict=True
        )
        full_trace = CoreTraceRecord.model_validate(
            full_trace.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise BodyRefinementError("route parity trace is invalid") from error
    if s1s2_trace.config != "s1s2" or full_trace.config != "full":
        raise BodyRefinementError("route parity requires S1+S2 and Full traces")
    if (
        s1s2_trace.query_id != full_trace.query_id
        or s1s2_trace.component_id != full_trace.component_id
        or s1s2_trace.route_trace_sha256 is None
        or full_trace.route_trace_sha256 is None
    ):
        raise BodyRefinementError("route parity traces do not prove a common selected route")
    return make_route_trace_comparison(
        query_id=s1s2_trace.query_id,
        component_id=s1s2_trace.component_id,
        s1s2_route_trace_sha256=s1s2_trace.route_trace_sha256,
        full_route_trace_sha256=full_trace.route_trace_sha256,
    )


def _verified_route_comparison(value: RouteTraceComparison) -> RouteTraceComparison:
    if type(value) is not RouteTraceComparison:
        raise TypeError("route trace comparison must be a RouteTraceComparison")
    try:
        return RouteTraceComparison.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise BodyRefinementError("route comparison self hash or invariants are invalid") from error


def _paired_body_rows(
    baseline: Sequence[BodyGateRow], candidate: Sequence[BodyGateRow]
) -> tuple[tuple[BodyGateRow, ...], tuple[BodyGateRow, ...]]:
    before = tuple(_verified_body_gate_row(item) for item in baseline)
    after = tuple(_verified_body_gate_row(item) for item in candidate)
    if not before or not after:
        raise BodyRefinementError("body gate requires baseline and candidate rows")

    def key(item: BodyGateRow) -> tuple[str, str, str]:
        return item.query_id, item.component_id, item.query_sha256

    before_by_key = {key(item): item for item in before}
    after_by_key = {key(item): item for item in after}
    if len(before_by_key) != len(before) or len(after_by_key) != len(after):
        raise BodyRefinementError("body gate contains duplicate query bindings")
    if tuple(sorted(before_by_key)) != tuple(sorted(after_by_key)):
        raise BodyRefinementError("body gate baseline and candidate populations differ")
    ordered_keys = tuple(sorted(before_by_key))
    return (
        tuple(before_by_key[item] for item in ordered_keys),
        tuple(after_by_key[item] for item in ordered_keys),
    )


def _route_trace_gate_passed(
    baseline: tuple[BodyGateRow, ...],
    candidate: tuple[BodyGateRow, ...],
    comparisons: Sequence[RouteTraceComparison],
) -> bool:
    verified = tuple(_verified_route_comparison(item) for item in comparisons)

    def key(item: BodyGateRow) -> tuple[str, str, str]:
        return item.query_id, item.component_id, item.query_sha256

    comparison_by_key = {
        (item.query_id, item.component_id, item.query_sha256): item for item in verified
    }
    expected_keys = {key(item) for item in baseline}
    if len(comparison_by_key) != len(verified) or set(comparison_by_key) != expected_keys:
        return False
    return all(
        before.route_trace_sha256 == after.route_trace_sha256
        and comparison_by_key[key(before)].matches
        and before.route_trace_sha256
        == comparison_by_key[key(before)].s1s2_route_trace_sha256
        and after.route_trace_sha256
        == comparison_by_key[key(before)].full_route_trace_sha256
        for before, after in zip(baseline, candidate, strict=True)
    )


class BodyRefinementDecision(_StrictFrozenModel):
    """A self-hashed S3 accept/rollback decision with hard safety gates."""

    schema_version: Literal[1] = 1
    candidate_sha256: Sha256
    baseline_metrics: BodyGateMetrics
    candidate_metrics: BodyGateMetrics
    j_project_delta: float
    soft_j_project_improved: bool
    grounding_hard_gate_passed: bool
    safety_hard_gate_passed: bool
    route_trace_gate_passed: bool
    outcome: Literal["accept", "rollback"]
    rationale: str
    route_hash_preserved: Literal[True]
    description_hash_preserved: Literal[True]
    static_refs_hash_preserved: Literal[True]
    operators_hash_preserved: Literal[True]
    decision_sha256: Sha256

    @field_validator("j_project_delta")
    @classmethod
    def validate_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("j_project_delta must be finite")
        return value

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _nonblank(value, "rationale")

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        expected_delta = (
            self.candidate_metrics.mean_j_project - self.baseline_metrics.mean_j_project
        )
        if abs(self.j_project_delta - expected_delta) > 1e-12:
            raise ValueError("j_project_delta does not match body-gate metrics")
        if self.soft_j_project_improved != (self.j_project_delta > 0.0):
            raise ValueError("soft J_project flag is inconsistent")
        expected_accept = (
            self.grounding_hard_gate_passed
            and self.safety_hard_gate_passed
            and self.route_trace_gate_passed
        )
        if (self.outcome == "accept") != expected_accept:
            raise ValueError("body decision does not match the frozen hard gates")
        if self.decision_sha256 != _self_hash(self, "decision_sha256"):
            raise ValueError("body decision self hash mismatch")
        return self


def evaluate_body_candidate(
    candidate: BodyRefinementCandidate,
    *,
    baseline_rows: Sequence[BodyGateRow],
    candidate_rows: Sequence[BodyGateRow],
    route_trace_comparisons: Sequence[RouteTraceComparison],
) -> BodyRefinementDecision:
    """Apply S3's soft J_project target and hard safety/route rollback gates."""

    candidate = _verified_candidate(candidate)
    before, after = _paired_body_rows(baseline_rows, candidate_rows)
    baseline_metrics = body_gate_metrics(before)
    candidate_metrics = body_gate_metrics(after)
    grounding_hard_gate_passed = (
        candidate_metrics.grounding_pass_rate >= baseline_metrics.grounding_pass_rate
        and all(
            not baseline.grounding_passed or refined.grounding_passed
            for baseline, refined in zip(before, after, strict=True)
        )
    )
    safety_hard_gate_passed = (
        candidate_metrics.safety_pass_rate >= baseline_metrics.safety_pass_rate
        and all(
            not baseline.safety_passed or refined.safety_passed
            for baseline, refined in zip(before, after, strict=True)
        )
    )
    route_trace_gate_passed = _route_trace_gate_passed(
        before, after, route_trace_comparisons
    )
    accepted = (
        grounding_hard_gate_passed
        and safety_hard_gate_passed
        and route_trace_gate_passed
    )
    if accepted:
        rationale = (
            "route trace and hard grounding/safety gates passed; J_project is soft"
        )
    elif not route_trace_gate_passed:
        rationale = "S1+S2 and Full route traces differ or lack complete parity evidence"
    elif not grounding_hard_gate_passed:
        rationale = "grounding hard gate regressed"
    else:
        rationale = "safety hard gate regressed"
    unsigned = {
        "schema_version": 1,
        "candidate_sha256": candidate.candidate_sha256,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "j_project_delta": candidate_metrics.mean_j_project
        - baseline_metrics.mean_j_project,
        "soft_j_project_improved": candidate_metrics.mean_j_project
        > baseline_metrics.mean_j_project,
        "grounding_hard_gate_passed": grounding_hard_gate_passed,
        "safety_hard_gate_passed": safety_hard_gate_passed,
        "route_trace_gate_passed": route_trace_gate_passed,
        "outcome": "accept" if accepted else "rollback",
        "rationale": rationale,
        "route_hash_preserved": True,
        "description_hash_preserved": True,
        "static_refs_hash_preserved": True,
        "operators_hash_preserved": True,
    }
    return BodyRefinementDecision.model_validate(
        {**unsigned, "decision_sha256": _hash_payload(unsigned)}, strict=True
    )


run_body_gate = evaluate_body_candidate


__all__ = [
    "BodyGateMetrics",
    "BodyGateRow",
    "BodyRefinementCandidate",
    "BodyRefinementDecision",
    "BodyRefinementError",
    "RouteTraceComparison",
    "body_gate_metrics",
    "body_gate_row_from_trace",
    "build_body_candidate",
    "compare_s1s2_full_route_traces",
    "evaluate_body_candidate",
    "make_body_gate_row",
    "make_route_trace_comparison",
    "run_body_gate",
]
