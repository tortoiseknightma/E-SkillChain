"""Minimal, typed S2 route-surface optimization and rollback gate.

S2 is intentionally not a bank compiler.  It receives complete immutable
skill surfaces so it can prove that a candidate changed only ``route`` and
``description``.  Its only training input is typed ``opt_pool`` routing
attribution, and its only evaluation input is typed ``route_gate`` rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from skillchain.evaluation.packets import AssistantRunConfig
from skillchain.evolution.attribution import (
    AttributionError,
    EvolutionExample,
    Sha256,
    routing_opt_pool_examples,
)
from skillchain.evolution.models import CoreTraceRecord
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


class RouteOptimizationError(ValueError):
    """Raised when S2 evidence, candidate scope, or gate input is invalid."""


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


class RouteSkillSurface(_StrictFrozenModel):
    """The complete skill fields needed to prove S2's edit surface."""

    schema_version: Literal[1] = 1
    slug: str
    capability_id: str
    route: str
    description: str
    body: str
    static_refs: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    surface_sha256: Sha256

    @field_validator("slug", "capability_id", "route", "description")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        # Existing StrictSkillArtifact bodies are canonical Markdown and end in
        # a newline, so they intentionally cannot use the trimmed-text helper.
        if not value or value != value.lstrip():
            raise ValueError("body must be non-empty canonical Markdown")
        return value

    @field_validator("static_refs", "operators", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("static_refs", "operators")
    @classmethod
    def validate_sorted_unique(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        cleaned = tuple(_nonblank(item, info.field_name) for item in value)
        if cleaned != tuple(sorted(set(cleaned))):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_surface(self) -> Self:
        if self.surface_sha256 != _self_hash(self, "surface_sha256"):
            raise ValueError("route skill surface self hash mismatch")
        return self


def make_route_skill_surface(
    *,
    slug: str,
    capability_id: str,
    route: str,
    description: str,
    body: str,
    static_refs: Sequence[str] = (),
    operators: Sequence[str] = (),
) -> RouteSkillSurface:
    """Create one self-hashed surface in the canonical field order."""

    unsigned = {
        "schema_version": 1,
        "slug": slug,
        "capability_id": capability_id,
        "route": route,
        "description": description,
        "body": body,
        "static_refs": tuple(static_refs),
        "operators": tuple(operators),
    }
    return RouteSkillSurface.model_validate(
        {**unsigned, "surface_sha256": _hash_payload(unsigned)}, strict=True
    )


def _verified_surface(value: RouteSkillSurface) -> RouteSkillSurface:
    if type(value) is not RouteSkillSurface:
        raise TypeError("skill surface must be a RouteSkillSurface")
    try:
        return RouteSkillSurface.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise RouteOptimizationError("skill surface self hash or invariants are invalid") from error


def _verified_candidate(value: "RouteOptimizationCandidate") -> "RouteOptimizationCandidate":
    if type(value) is not RouteOptimizationCandidate:
        raise TypeError("candidate must be a RouteOptimizationCandidate")
    try:
        return RouteOptimizationCandidate.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise RouteOptimizationError("route candidate self hash or invariants are invalid") from error


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
        raise RouteOptimizationError("S2 requires at least one skill surface")
    baseline_identities = tuple((item.slug, item.capability_id) for item in baseline)
    candidate_identities = tuple((item.slug, item.capability_id) for item in candidate)
    if len({item.slug for item in baseline}) != len(baseline) or len(
        {item.slug for item in candidate}
    ) != len(candidate):
        raise RouteOptimizationError("S2 skill surfaces must have unique slugs")
    if baseline_identities != candidate_identities:
        raise RouteOptimizationError("S2 candidate changes skill identity or capability")
    for before, after in zip(baseline, candidate, strict=True):
        if (
            before.body != after.body
            or before.static_refs != after.static_refs
            or before.operators != after.operators
        ):
            raise RouteOptimizationError(
                "S2 may change only route and description; body/static refs/operators drifted"
            )


class RouteOptimizationCandidate(_StrictFrozenModel):
    """A self-hashed S2 candidate with explicit immutable-field proofs."""

    schema_version: Literal[1] = 1
    baseline_bank_sha256: Sha256
    candidate_bank_sha256: Sha256
    baseline_skills: tuple[RouteSkillSurface, ...]
    candidate_skills: tuple[RouteSkillSurface, ...]
    opt_pool_examples: tuple[EvolutionExample, ...]
    opt_pool_examples_sha256: Sha256
    baseline_surface_set_sha256: Sha256
    candidate_surface_set_sha256: Sha256
    baseline_route_sha256: Sha256
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
            examples = routing_opt_pool_examples(self.opt_pool_examples)
        except AttributionError as error:
            raise ValueError(str(error)) from error
        if not examples:
            raise ValueError("S2 requires at least one opt_pool routing example")
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
            "baseline_route_sha256": _surface_field_sha256(baseline, "route"),
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
            self.baseline_body_sha256 == self.candidate_body_sha256
            and self.baseline_static_refs_sha256 == self.candidate_static_refs_sha256
            and self.baseline_operators_sha256 == self.candidate_operators_sha256
        ):
            raise ValueError("S2 immutable body/static refs/operators hashes changed")
        if self.candidate_sha256 != _self_hash(self, "candidate_sha256"):
            raise ValueError("route candidate self hash mismatch")
        return self


def build_route_candidate(
    *,
    baseline_bank_sha256: str,
    candidate_bank_sha256: str,
    baseline_skills: Sequence[RouteSkillSurface],
    candidate_skills: Sequence[RouteSkillSurface],
    opt_pool_examples: Sequence[EvolutionExample],
) -> RouteOptimizationCandidate:
    """Build a candidate after proving it sees only S2-eligible evidence."""

    baseline = tuple(sorted((_verified_surface(item) for item in baseline_skills), key=lambda item: item.slug))
    candidate = tuple(sorted((_verified_surface(item) for item in candidate_skills), key=lambda item: item.slug))
    _validate_surface_sets(baseline, candidate)
    try:
        examples = routing_opt_pool_examples(tuple(opt_pool_examples))
    except AttributionError as error:
        raise RouteOptimizationError(str(error)) from error
    examples = tuple(
        sorted(
            examples,
            key=lambda item: (item.component_id, item.query_id, item.trace_sha256),
        )
    )
    if not examples:
        raise RouteOptimizationError("S2 requires at least one opt_pool routing example")
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
        "baseline_route_sha256": _surface_field_sha256(baseline, "route"),
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
    return RouteOptimizationCandidate.model_validate(
        {**unsigned, "candidate_sha256": _hash_payload(unsigned)}, strict=True
    )


class RouteGateRow(_StrictFrozenModel):
    """One non-test frozen route-gate prediction."""

    schema_version: Literal[1] = 1
    source_split: Literal["route_gate"] = "route_gate"
    query_id: str
    component_id: str
    canonical_capability: str
    selected_capability: str | None = None
    run_id: str
    config: AssistantRunConfig
    query_sha256: Sha256
    route_trace_sha256: Sha256 | None = None
    gate_row_sha256: Sha256

    @field_validator("query_id", "component_id", "canonical_capability", "run_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("selected_capability")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "selected_capability")

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        if self.query_sha256 != _identity_hash("query", self.query_id):
            raise ValueError("route gate query identity hash mismatch")
        if self.gate_row_sha256 != _self_hash(self, "gate_row_sha256"):
            raise ValueError("route gate row self hash mismatch")
        return self


def make_route_gate_row(
    *,
    query_id: str,
    component_id: str,
    canonical_capability: str,
    selected_capability: str | None,
    run_id: str,
    config: AssistantRunConfig,
    route_trace_sha256: str | None = None,
) -> RouteGateRow:
    """Create a route-gate row without any path or test-set input."""

    unsigned = {
        "schema_version": 1,
        "source_split": "route_gate",
        "query_id": query_id,
        "component_id": component_id,
        "canonical_capability": canonical_capability,
        "selected_capability": selected_capability,
        "run_id": run_id,
        "config": config,
        "query_sha256": _identity_hash("query", query_id),
        "route_trace_sha256": route_trace_sha256,
    }
    return RouteGateRow.model_validate(
        {**unsigned, "gate_row_sha256": _hash_payload(unsigned)}, strict=True
    )


def route_gate_row_from_trace(trace: CoreTraceRecord) -> RouteGateRow:
    """Convert a verified trace to a frozen route-gate observation."""

    try:
        trace = CoreTraceRecord.model_validate(trace.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise RouteOptimizationError("route-gate trace is invalid") from error
    if trace.canonical_capability is None:
        raise RouteOptimizationError("route-gate trace has no canonical capability")
    return make_route_gate_row(
        query_id=trace.query_id,
        component_id=trace.component_id,
        canonical_capability=trace.canonical_capability,
        selected_capability=trace.selected_capability,
        run_id=trace.run_id,
        config=trace.config,
        route_trace_sha256=trace.route_trace_sha256,
    )


def _verified_route_gate_row(value: RouteGateRow) -> RouteGateRow:
    if type(value) is not RouteGateRow:
        raise TypeError("route gate row must be a RouteGateRow")
    try:
        return RouteGateRow.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise RouteOptimizationError("route-gate row self hash or invariants are invalid") from error


def _f1(*, true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator


class RouteGateMetrics(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    row_count: int
    capability_macro_f1: float
    component_macro_f1: float
    metrics_sha256: Sha256

    @field_validator("row_count")
    @classmethod
    def validate_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("route-gate metrics require at least one row")
        return value

    @field_validator("capability_macro_f1", "component_macro_f1")
    @classmethod
    def validate_metric(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("route-gate metric must be finite and within [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.metrics_sha256 != _self_hash(self, "metrics_sha256"):
            raise ValueError("route-gate metrics self hash mismatch")
        return self


def route_gate_metrics(rows: Sequence[RouteGateRow]) -> RouteGateMetrics:
    """Compute capability and component macro-F1 on one frozen population.

    Component macro-F1 treats a correct route as the positive class within each
    component.  For a component with repeated queries this is its route
    accuracy expressed as binary F1; macro averaging prevents a large leakage
    component from dominating the gate.
    """

    verified = tuple(_verified_route_gate_row(item) for item in rows)
    if not verified:
        raise RouteOptimizationError("route gate requires at least one row")
    labels = tuple(sorted({item.canonical_capability for item in verified}))
    capability_scores: list[float] = []
    for label in labels:
        true_positive = sum(
            item.canonical_capability == label and item.selected_capability == label
            for item in verified
        )
        false_positive = sum(
            item.canonical_capability != label and item.selected_capability == label
            for item in verified
        )
        false_negative = sum(
            item.canonical_capability == label and item.selected_capability != label
            for item in verified
        )
        capability_scores.append(
            _f1(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
            )
        )
    by_component: dict[str, list[RouteGateRow]] = defaultdict(list)
    for item in verified:
        by_component[item.component_id].append(item)
    component_scores: list[float] = []
    for component_rows in by_component.values():
        correct = sum(
            item.selected_capability == item.canonical_capability
            for item in component_rows
        )
        incorrect = len(component_rows) - correct
        component_scores.append(
            _f1(
                true_positive=correct,
                false_positive=incorrect,
                false_negative=incorrect,
            )
        )
    unsigned = {
        "schema_version": 1,
        "row_count": len(verified),
        "capability_macro_f1": sum(capability_scores) / len(capability_scores),
        "component_macro_f1": sum(component_scores) / len(component_scores),
    }
    return RouteGateMetrics.model_validate(
        {**unsigned, "metrics_sha256": _hash_payload(unsigned)}, strict=True
    )


def _paired_route_rows(
    baseline: Sequence[RouteGateRow], candidate: Sequence[RouteGateRow]
) -> tuple[tuple[RouteGateRow, ...], tuple[RouteGateRow, ...]]:
    before = tuple(_verified_route_gate_row(item) for item in baseline)
    after = tuple(_verified_route_gate_row(item) for item in candidate)
    if not before or not after:
        raise RouteOptimizationError("route gate requires baseline and candidate rows")

    def key(item: RouteGateRow) -> tuple[str, str, str, str]:
        return (
            item.query_id,
            item.component_id,
            item.canonical_capability,
            item.query_sha256,
        )

    before_by_key = {key(item): item for item in before}
    after_by_key = {key(item): item for item in after}
    if len(before_by_key) != len(before) or len(after_by_key) != len(after):
        raise RouteOptimizationError("route gate contains duplicate query bindings")
    if tuple(sorted(before_by_key)) != tuple(sorted(after_by_key)):
        raise RouteOptimizationError("route gate baseline and candidate populations differ")
    ordered_keys = tuple(sorted(before_by_key))
    return (
        tuple(before_by_key[item] for item in ordered_keys),
        tuple(after_by_key[item] for item in ordered_keys),
    )


class RouteOptimizationDecision(_StrictFrozenModel):
    """A self-hashed S2 acceptance or rollback decision."""

    schema_version: Literal[1] = 1
    candidate_sha256: Sha256
    baseline_metrics: RouteGateMetrics
    candidate_metrics: RouteGateMetrics
    outcome: Literal["accept", "rollback"]
    rationale: str
    body_hash_preserved: Literal[True]
    static_refs_hash_preserved: Literal[True]
    operators_hash_preserved: Literal[True]
    decision_sha256: Sha256

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _nonblank(value, "rationale")

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        expected_accept = (
            self.candidate_metrics.capability_macro_f1
            >= self.baseline_metrics.capability_macro_f1
            and self.candidate_metrics.component_macro_f1
            >= self.baseline_metrics.component_macro_f1
        )
        if (self.outcome == "accept") != expected_accept:
            raise ValueError("route decision does not match the frozen two-metric gate")
        if self.decision_sha256 != _self_hash(self, "decision_sha256"):
            raise ValueError("route decision self hash mismatch")
        return self


def evaluate_route_candidate(
    candidate: RouteOptimizationCandidate,
    *,
    baseline_rows: Sequence[RouteGateRow],
    candidate_rows: Sequence[RouteGateRow],
) -> RouteOptimizationDecision:
    """Apply the r2 route gate and return an explicit accept/rollback record."""

    candidate = _verified_candidate(candidate)
    before, after = _paired_route_rows(baseline_rows, candidate_rows)
    baseline_metrics = route_gate_metrics(before)
    candidate_metrics = route_gate_metrics(after)
    accepted = (
        candidate_metrics.capability_macro_f1 >= baseline_metrics.capability_macro_f1
        and candidate_metrics.component_macro_f1 >= baseline_metrics.component_macro_f1
    )
    rationale = (
        "capability and component macro-F1 did not regress"
        if accepted
        else "capability or component macro-F1 regressed"
    )
    unsigned = {
        "schema_version": 1,
        "candidate_sha256": candidate.candidate_sha256,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "outcome": "accept" if accepted else "rollback",
        "rationale": rationale,
        "body_hash_preserved": True,
        "static_refs_hash_preserved": True,
        "operators_hash_preserved": True,
    }
    return RouteOptimizationDecision.model_validate(
        {**unsigned, "decision_sha256": _hash_payload(unsigned)}, strict=True
    )


run_route_gate = evaluate_route_candidate


__all__ = [
    "RouteGateMetrics",
    "RouteGateRow",
    "RouteOptimizationCandidate",
    "RouteOptimizationDecision",
    "RouteOptimizationError",
    "RouteSkillSurface",
    "build_route_candidate",
    "evaluate_route_candidate",
    "make_route_gate_row",
    "make_route_skill_surface",
    "route_gate_metrics",
    "route_gate_row_from_trace",
    "run_route_gate",
]
