"""Deterministic, fail-closed attribution for internal evolution traces.

The module deliberately works only with :class:`CoreTraceRecord` values.  It
does not parse logs, read split files, or expose Final Judge packets.  The
priority below makes each trace belong to exactly one attribution bucket:

``routing -> tool -> infra_error -> safety -> grounding -> body ->
insufficient_evidence``.

The first two positions implement the r2 contract directly: an unacceptable
route wins over every later symptom, and a failed tool wins over an answer
quality failure.  Consequently a tool or infrastructure failure can never be
used as a S3 body example.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.evaluation.packets import AssistantRunConfig
from skillchain.evolution.models import CoreTraceRecord
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
EvolutionSplit = Literal[
    "dev_mini",
    "opt_pool",
    "route_gate",
    "body_gate",
    "shadow_val",
    "test_frozen",
]
AttributionKind = Literal[
    "routing",
    "tool",
    "grounding",
    "body",
    "safety",
    "infra_error",
    "insufficient_evidence",
]
TrainingStage = Literal["s2", "s3"]
RouteCorrectness = Literal["correct", "incorrect", "not_routed", "unavailable"]
ToolOutcome = Literal["not_called", "success", "error", "mixed"]


class AttributionError(ValueError):
    """Raised when an attribution or its frozen evidence cannot be trusted."""


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


def _verified_trace(value: CoreTraceRecord) -> CoreTraceRecord:
    if type(value) is not CoreTraceRecord:
        raise TypeError("trace must be a CoreTraceRecord")
    try:
        return CoreTraceRecord.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise AttributionError("core trace self hash or invariants are invalid") from error


def _verified_example(value: "EvolutionExample") -> "EvolutionExample":
    if type(value) is not EvolutionExample:
        raise TypeError("example must be an EvolutionExample")
    try:
        return EvolutionExample.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise AttributionError("evolution example self hash or invariants are invalid") from error


def _training_stage(kind: AttributionKind) -> TrainingStage | None:
    if kind == "routing":
        return "s2"
    if kind == "body":
        return "s3"
    return None


def _rule_tokens(rule_id: str) -> frozenset[str]:
    return frozenset(token for token in re.split(r"[^a-z0-9]+", rule_id.casefold()) if token)


def _has_failed_rule(trace: CoreTraceRecord, family: frozenset[str]) -> bool:
    """Treat zero-or-negative deterministic rule scores as a failed check.

    Rule scores are intentionally generic in ``CoreTraceRecord``.  The r2
    convention is binary failure at ``<= 0``; a positive score is retained as
    evidence but is not itself a failure attribution.
    """

    if trace.evaluation is None:
        return False
    return any(
        score.score <= 0.0 and bool(_rule_tokens(score.rule_id) & family)
        for score in trace.evaluation.rule_scores
    )


_SAFETY_TOKENS = frozenset({"safety", "safe", "harm", "policy"})
_GROUNDING_TOKENS = frozenset(
    {
        "grounding",
        "grounded",
        "citation",
        "citations",
        "evidence",
        "factuality",
        "faithfulness",
    }
)


def _classify(trace: CoreTraceRecord) -> tuple[AttributionKind, str]:
    """Return the one deterministic attribution for a verified trace."""

    # A missing canonical label is not proof of an incorrect route.
    if trace.route_correctness == "unavailable":
        return "insufficient_evidence", "canonical-capability-unavailable"
    if trace.route_correctness != "correct":
        return "routing", "route-not-acceptable"
    if trace.tool_outcome not in {"not_called", "success"}:
        return "tool", "tool-trace-failed"
    if trace.error_code is not None:
        return "infra_error", "assistant-execution-failed"
    if trace.evaluation is None:
        return "insufficient_evidence", "no-offline-evaluation"
    if (
        trace.evaluation.judge_outcome is not None
        and trace.evaluation.judge_outcome.status != "scored"
    ):
        return "infra_error", "judge-execution-failed"
    if _has_failed_rule(trace, _SAFETY_TOKENS):
        return "safety", "safety-rule-failed"
    if _has_failed_rule(trace, _GROUNDING_TOKENS):
        return "grounding", "grounding-rule-failed"
    if any(score.score <= 0.0 for score in trace.evaluation.rule_scores):
        return "body", "answer-rule-failed"
    judge = trace.evaluation.judge_outcome
    if judge is not None and judge.scores.j_project < 100.0:
        return "body", "judge-project-score-below-perfect"
    return "insufficient_evidence", "no-attributable-failure"


class EvolutionExample(_StrictFrozenModel):
    """A trace-bound, mutually exclusive offline learning example.

    ``run_sha256``, ``config_sha256`` and ``component_sha256`` are derived
    identity hashes in addition to the original immutable trace hashes.  This
    lets a consumer bind an example to its source without treating a mutable
    string label as sufficient evidence.
    """

    schema_version: Literal[1] = 1
    source_split: EvolutionSplit
    attribution: AttributionKind
    attribution_reason: str
    training_stage: TrainingStage | None = None
    query_id: str
    component_id: str
    run_id: str
    config: AssistantRunConfig
    route_correctness: RouteCorrectness
    tool_outcome: ToolOutcome
    error_code: str | None = None
    trace_sha256: Sha256
    request_sha256: Sha256
    row_sha256: Sha256
    run_manifest_sha256: Sha256
    bank_sha256: Sha256 | None = None
    route_trace_sha256: Sha256 | None = None
    run_sha256: Sha256
    config_sha256: Sha256
    component_sha256: Sha256
    binding_sha256: Sha256
    example_sha256: Sha256

    @field_validator("attribution_reason", "query_id", "component_id", "run_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("error_code")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "error_code")

    @model_validator(mode="after")
    def validate_example(self) -> Self:
        if self.training_stage != _training_stage(self.attribution):
            raise ValueError("training stage does not match attribution")
        if self.attribution == "routing" and self.route_correctness not in {
            "incorrect",
            "not_routed",
        }:
            raise ValueError("routing attribution requires an unacceptable route")
        if self.attribution == "tool" and (
            self.route_correctness != "correct"
            or self.tool_outcome in {"not_called", "success"}
        ):
            raise ValueError("tool attribution requires a correct route and failed tool")
        if self.attribution in {"body", "grounding", "safety"} and (
            self.route_correctness != "correct"
            or self.tool_outcome not in {"not_called", "success"}
            or self.error_code is not None
        ):
            raise ValueError(
                "answer-quality attribution requires route-correct non-error evidence"
            )
        if self.run_sha256 != _identity_hash("run", self.run_id):
            raise ValueError("run identity hash mismatch")
        if self.config_sha256 != _identity_hash("config", self.config):
            raise ValueError("config identity hash mismatch")
        if self.component_sha256 != _identity_hash("component", self.component_id):
            raise ValueError("component identity hash mismatch")
        expected_binding = _hash_payload(
            {
                "trace_sha256": self.trace_sha256,
                "request_sha256": self.request_sha256,
                "row_sha256": self.row_sha256,
                "run_manifest_sha256": self.run_manifest_sha256,
                "bank_sha256": self.bank_sha256,
                "route_trace_sha256": self.route_trace_sha256,
                "route_correctness": self.route_correctness,
                "tool_outcome": self.tool_outcome,
                "error_code": self.error_code,
                "run_sha256": self.run_sha256,
                "config_sha256": self.config_sha256,
                "component_sha256": self.component_sha256,
            }
        )
        if self.binding_sha256 != expected_binding:
            raise ValueError("evolution example binding hash mismatch")
        if self.example_sha256 != _self_hash(self, "example_sha256"):
            raise ValueError("evolution example self hash mismatch")
        return self


def attribute_core_trace(
    trace: CoreTraceRecord,
    *,
    source_split: EvolutionSplit,
) -> EvolutionExample:
    """Classify one trace and bind every source identity into a frozen record."""

    trace = _verified_trace(trace)
    attribution, reason = _classify(trace)
    run_sha256 = _identity_hash("run", trace.run_id)
    config_sha256 = _identity_hash("config", trace.config)
    component_sha256 = _identity_hash("component", trace.component_id)
    binding_sha256 = _hash_payload(
        {
            "trace_sha256": trace.trace_sha256,
            "request_sha256": trace.request_sha256,
            "row_sha256": trace.row_sha256,
            "run_manifest_sha256": trace.run_manifest_sha256,
            "bank_sha256": trace.bank_sha256,
            "route_trace_sha256": trace.route_trace_sha256,
            "route_correctness": trace.route_correctness,
            "tool_outcome": trace.tool_outcome,
            "error_code": trace.error_code,
            "run_sha256": run_sha256,
            "config_sha256": config_sha256,
            "component_sha256": component_sha256,
        }
    )
    unsigned = {
        "schema_version": 1,
        "source_split": source_split,
        "attribution": attribution,
        "attribution_reason": reason,
        "training_stage": _training_stage(attribution),
        "query_id": trace.query_id,
        "component_id": trace.component_id,
        "run_id": trace.run_id,
        "config": trace.config,
        "route_correctness": trace.route_correctness,
        "tool_outcome": trace.tool_outcome,
        "error_code": trace.error_code,
        "trace_sha256": trace.trace_sha256,
        "request_sha256": trace.request_sha256,
        "row_sha256": trace.row_sha256,
        "run_manifest_sha256": trace.run_manifest_sha256,
        "bank_sha256": trace.bank_sha256,
        "route_trace_sha256": trace.route_trace_sha256,
        "run_sha256": run_sha256,
        "config_sha256": config_sha256,
        "component_sha256": component_sha256,
        "binding_sha256": binding_sha256,
    }
    return EvolutionExample.model_validate(
        {**unsigned, "example_sha256": _hash_payload(unsigned)}, strict=True
    )


def routing_opt_pool_examples(
    examples: tuple[EvolutionExample, ...] | list[EvolutionExample],
) -> tuple[EvolutionExample, ...]:
    """Return only the S2-readable opt-pool routing failures.

    Passing any other split, including ``test_frozen``, fails closed instead of
    silently filtering it away.  This makes accidental evaluation-set access
    visible to the caller.
    """

    verified = tuple(_verified_example(item) for item in examples)
    if any(item.source_split != "opt_pool" for item in verified):
        raise AttributionError("S2 may read only opt_pool attribution examples")
    if any(
        item.attribution != "routing" or item.training_stage != "s2"
        for item in verified
    ):
        raise AttributionError("S2 requires routing attribution examples")
    return verified


def body_opt_pool_examples(
    examples: tuple[EvolutionExample, ...] | list[EvolutionExample],
) -> tuple[EvolutionExample, ...]:
    """Return only route-correct S3-readable opt-pool body failures."""

    verified = tuple(_verified_example(item) for item in examples)
    if any(item.source_split != "opt_pool" for item in verified):
        raise AttributionError("S3 may read only opt_pool attribution examples")
    if any(
        item.attribution != "body" or item.training_stage != "s3"
        for item in verified
    ):
        raise AttributionError("S3 requires route-correct body attribution examples")
    return verified


# The shorter spelling is intentionally kept for callers that name the action
# rather than the record type.
attribute_trace = attribute_core_trace


__all__ = [
    "AttributionError",
    "AttributionKind",
    "EvolutionExample",
    "EvolutionSplit",
    "RouteCorrectness",
    "Sha256",
    "TrainingStage",
    "ToolOutcome",
    "attribute_core_trace",
    "attribute_trace",
    "body_opt_pool_examples",
    "routing_opt_pool_examples",
]
