"""Fail-closed internal trace records for later S1/S2/S3 attribution.

These records are deliberately separate from :mod:`skillchain.evaluation.packets`.
They retain route and tool execution evidence for learning/diagnostics, but are
never a model-visible input to the frozen Final Judge.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.evaluation.assistant_runs import (
    AssistantExecutionReceipt,
    AssistantRouteAttempt,
    AssistantRunError,
    VerifiedAssistantRunBundle,
    VerifiedPhase4Inputs,
    require_verified_assistant_run_bundle,
    require_verified_phase4_inputs,
)
from skillchain.evaluation.packets import (
    AssistantRunConfig,
    AssistantToolTrace,
    JudgeOutcome,
    VisibleCard,
    VisibleToolEvidence,
)
from skillchain.llm import LLMUsage
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CoreTraceError(ValueError):
    """Raised when internal evolution evidence cannot be proven safe to join."""


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


class CoreTraceRuleScore(_StrictFrozenModel):
    """An optional deterministic rule score, kept outside Final Judge input."""

    rule_id: str
    score: float

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        return _nonblank(value, "rule_id")

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rule score must be finite")
        return value


class CoreTraceEvaluation(_StrictFrozenModel):
    """Opt-in outcomes for offline evolution analysis only."""

    judge_outcome: JudgeOutcome | None = None
    rule_scores: tuple[CoreTraceRuleScore, ...] = ()

    @field_validator("rule_scores", mode="before")
    @classmethod
    def coerce_rule_scores(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_rule_scores(self) -> Self:
        identifiers = tuple(item.rule_id for item in self.rule_scores)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("rule scores must have unique, sorted rule IDs")
        return self


ToolOutcome = Literal["not_called", "success", "error", "mixed"]
RouteCorrectness = Literal["correct", "incorrect", "not_routed", "unavailable"]


class CoreTraceRecord(_StrictFrozenModel):
    """One fully verified, denominator-preserving internal Assistant trace.

    ``evaluation`` is not part of any FinalEvaluationPacket and must be added
    only through the explicit gate in :func:`build_core_trace_record`.
    """

    schema_version: Literal[1] = 1
    query_id: str
    component_id: str
    run_id: str
    config: AssistantRunConfig
    query_ordinal: int = Field(ge=0)
    request_sha256: Sha256
    row_sha256: Sha256
    run_manifest_sha256: Sha256
    canonical_capability: str | None = None
    acceptable_capabilities: tuple[str, ...] = ()
    selected_capability: str | None = None
    skill_slug: str | None = None
    bank_sha256: Sha256 | None = None
    route_trace_sha256: Sha256 | None = None
    route_attempt: AssistantRouteAttempt
    route_correctness: RouteCorrectness
    tool_outcome: ToolOutcome
    tool_error_codes: tuple[str, ...] = ()
    tool_trace: tuple[AssistantToolTrace, ...] = ()
    visible_cards: tuple[VisibleCard, ...] = ()
    visible_tool_evidence: tuple[VisibleToolEvidence, ...] = ()
    response_text: str
    usage: LLMUsage
    latency_ms: int = Field(ge=0)
    turn_count: int = Field(ge=1)
    error_code: str | None = None
    included_in_denominator: Literal[True] = True
    evaluation: CoreTraceEvaluation | None = None
    trace_sha256: Sha256

    @field_validator(
        "query_id",
        "component_id",
        "run_id",
        "canonical_capability",
        "selected_capability",
        "skill_slug",
        "error_code",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("response_text")
    @classmethod
    def validate_response_text(cls, value: str) -> str:
        # Error rows intentionally retain an empty public answer in the
        # denominator; successful rows are already required to be non-empty by
        # AssistantResult.
        if value and value != value.strip():
            raise ValueError("response_text must be trimmed when present")
        return value

    @field_validator(
        "acceptable_capabilities",
        "tool_error_codes",
        "tool_trace",
        "visible_cards",
        "visible_tool_evidence",
        mode="before",
    )
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("acceptable_capabilities")
    @classmethod
    def validate_acceptable_capabilities(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("acceptable capabilities must be unique")
        return tuple(
            _nonblank(item, "acceptable capability") for item in value
        )

    @field_validator("tool_error_codes")
    @classmethod
    def validate_tool_error_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_nonblank(item, "tool error code") for item in value)

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        if self.canonical_capability is None:
            if self.acceptable_capabilities:
                raise ValueError(
                    "unresolved capability cannot retain acceptable capabilities"
                )
        elif self.canonical_capability not in self.acceptable_capabilities:
            raise ValueError("canonical capability must be acceptable")

        routing = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        attempt = self.route_attempt
        if attempt.status == "selected":
            if routing != (
                attempt.selected_capability,
                attempt.skill_slug,
                attempt.route_trace_sha256,
            ):
                raise ValueError("trace routing differs from route-attempt receipt")
            expected_correctness: RouteCorrectness = (
                "unavailable"
                if self.canonical_capability is None
                else "correct"
                if self.selected_capability in self.acceptable_capabilities
                else "incorrect"
            )
        elif attempt.status == "not_applicable":
            if self.config != "noskill" or any(value is not None for value in routing):
                raise ValueError("not-applicable route attempt is inconsistent")
            expected_correctness = (
                "unavailable" if self.canonical_capability is None else "not_routed"
            )
        else:
            if self.config == "noskill" or any(value is not None for value in routing):
                raise ValueError("failed route attempt is inconsistent")
            expected_correctness = (
                "unavailable" if self.canonical_capability is None else "not_routed"
            )
        if self.route_correctness != expected_correctness:
            raise ValueError("route correctness differs from the verified route")

        statuses = {item.status for item in self.tool_trace}
        expected_tool_outcome: ToolOutcome = (
            "not_called"
            if not statuses
            else "success"
            if statuses == {"success"}
            else "error"
            if statuses == {"error"}
            else "mixed"
        )
        if self.tool_outcome != expected_tool_outcome:
            raise ValueError("tool outcome differs from the trace")
        expected_errors = tuple(
            item.error_code
            for item in self.tool_trace
            if item.error_code is not None
        )
        if self.tool_error_codes != expected_errors:
            raise ValueError("tool error codes differ from the trace")
        if self.trace_sha256 != _self_hash(self, "trace_sha256"):
            raise ValueError("core trace self hash mismatch")
        return self


def _route_correctness(
    *,
    canonical_capability: str | None,
    acceptable_capabilities: tuple[str, ...],
    route_attempt: AssistantRouteAttempt,
) -> RouteCorrectness:
    if canonical_capability is None:
        return "unavailable"
    if route_attempt.status != "selected":
        return "not_routed"
    return (
        "correct"
        if route_attempt.selected_capability in acceptable_capabilities
        else "incorrect"
    )


def _tool_outcome(trace: tuple[AssistantToolTrace, ...]) -> ToolOutcome:
    statuses = {item.status for item in trace}
    if not statuses:
        return "not_called"
    if statuses == {"success"}:
        return "success"
    if statuses == {"error"}:
        return "error"
    return "mixed"


def _require_exact_rule_scores(
    value: Sequence[CoreTraceRuleScore],
) -> tuple[CoreTraceRuleScore, ...]:
    scores = tuple(value)
    if any(type(item) is not CoreTraceRuleScore for item in scores):
        raise TypeError("rule_scores must contain CoreTraceRuleScore values")
    return scores


def _verified_inputs_and_run(
    inputs: VerifiedPhase4Inputs,
    run: VerifiedAssistantRunBundle,
) -> tuple[VerifiedPhase4Inputs, VerifiedAssistantRunBundle]:
    try:
        return require_verified_phase4_inputs(inputs), require_verified_assistant_run_bundle(
            run
        )
    except (AssistantRunError, TypeError) as error:
        raise CoreTraceError("verified trace inputs are stale or were mutated") from error


def build_core_trace_record(
    inputs: VerifiedPhase4Inputs,
    run: VerifiedAssistantRunBundle,
    *,
    query_ordinal: int,
    allow_judge_outcomes: bool = False,
    judge_outcome: JudgeOutcome | None = None,
    rule_scores: Sequence[CoreTraceRuleScore] = (),
) -> CoreTraceRecord:
    """Join one verified query/run/row/runner-receipt into internal evidence.

    The adapter intentionally rejects diagnostic/backend-reported rows and old
    runner receipts without a route-attempt record.  A failed execution remains
    a trace row with ``included_in_denominator=True`` rather than disappearing
    from the evolutionary attribution population.
    """

    if type(query_ordinal) is not int or query_ordinal < 0:
        raise CoreTraceError("query_ordinal must be a non-negative integer")
    if type(allow_judge_outcomes) is not bool:
        raise TypeError("allow_judge_outcomes must be a bool")
    if judge_outcome is not None and type(judge_outcome) is not JudgeOutcome:
        raise TypeError("judge_outcome must be a JudgeOutcome value")
    scores = _require_exact_rule_scores(rule_scores)
    if not allow_judge_outcomes and (judge_outcome is not None or scores):
        raise CoreTraceError(
            "judge outcomes and rule scores require allow_judge_outcomes=True"
        )

    inputs, run = _verified_inputs_and_run(inputs, run)
    if query_ordinal >= len(inputs.selected_queries) or query_ordinal >= len(run.rows):
        raise CoreTraceError("query_ordinal is absent from the verified inputs or run")
    if len(run.requests) != len(run.rows) or len(run.rows) != len(
        inputs.selected_queries
    ):
        raise CoreTraceError("verified run/query population has inconsistent coverage")
    if (
        run.manifest.matrix_run_id != inputs.selection.matrix_run_id
        or run.spec.matrix_run_id != inputs.selection.matrix_run_id
    ):
        raise CoreTraceError("run does not belong to the verified input selection")

    query = inputs.selected_queries[query_ordinal]
    assistant_query = inputs.assistant_queries[query_ordinal]
    request = run.requests[query_ordinal]
    row = run.rows[query_ordinal]
    result = row.result
    if (
        request.query_ordinal != query_ordinal
        or row.query_ordinal != query_ordinal
        or request.query != assistant_query
        or request.query.query_id != query.query_id
        or row.request_sha256 != request.request_sha256
        or result.query_id != query.query_id
        or result.run_id != inputs.selection.matrix_run_id
        or request.matrix_run_id != inputs.selection.matrix_run_id
        or result.config != run.manifest.config
        or request.config != run.manifest.config
        or result.query_artifact_sha256 != inputs.expected_query_sha256
        or result.split_manifest_sha256 != inputs.expected_split_manifest_sha256
    ):
        raise CoreTraceError("query, request, result, and run locks do not join")
    if row.execution_provenance != "runner-owned-v1" or row.execution_receipt is None:
        raise CoreTraceError("core trace requires a runner-owned execution receipt")
    receipt: AssistantExecutionReceipt = row.execution_receipt
    route_attempt = receipt.route_attempt
    if route_attempt is None:
        raise CoreTraceError("core trace requires a retained route-attempt receipt")
    if (
        receipt.request_sha256 != request.request_sha256
        or receipt.tool_trace != result.tool_trace
        or receipt.aggregate_usage != result.usage
        or receipt.runner_latency_ms != result.latency_ms
        or (receipt.outcome == "success") != (result.error_code is None)
    ):
        raise CoreTraceError("runner receipt and result do not join")
    result_routing = (
        result.selected_capability,
        result.skill_slug,
        result.route_trace_sha256,
    )
    attempt_routing = (
        route_attempt.selected_capability,
        route_attempt.skill_slug,
        route_attempt.route_trace_sha256,
    )
    if route_attempt.status == "selected":
        if result_routing != attempt_routing:
            raise CoreTraceError("result routing differs from the route-attempt receipt")
    elif any(value is not None for value in result_routing):
        raise CoreTraceError("unselected route attempt retained result route data")

    acceptable_capabilities = tuple(query.acceptable_capabilities)
    tool_trace = result.tool_trace
    evaluation = (
        CoreTraceEvaluation(judge_outcome=judge_outcome, rule_scores=scores)
        if allow_judge_outcomes and (judge_outcome is not None or scores)
        else None
    )
    unsigned = {
        "schema_version": 1,
        "query_id": query.query_id,
        "component_id": query.leakage_group_id,
        "run_id": result.run_id,
        "config": result.config,
        "query_ordinal": query_ordinal,
        "request_sha256": request.request_sha256,
        "row_sha256": row.row_sha256,
        "run_manifest_sha256": run.external_manifest_sha256,
        "canonical_capability": query.canonical_capability,
        "acceptable_capabilities": acceptable_capabilities,
        "selected_capability": result.selected_capability,
        "skill_slug": result.skill_slug,
        "bank_sha256": result.bank_sha256,
        "route_trace_sha256": result.route_trace_sha256,
        "route_attempt": route_attempt,
        "route_correctness": _route_correctness(
            canonical_capability=query.canonical_capability,
            acceptable_capabilities=acceptable_capabilities,
            route_attempt=route_attempt,
        ),
        "tool_outcome": _tool_outcome(tool_trace),
        "tool_error_codes": tuple(
            item.error_code for item in tool_trace if item.error_code is not None
        ),
        "tool_trace": tool_trace,
        "visible_cards": result.visible_cards,
        "visible_tool_evidence": result.visible_tool_evidence,
        "response_text": result.response_text,
        "usage": result.usage,
        "latency_ms": result.latency_ms,
        "turn_count": row.turn_count,
        "error_code": result.error_code,
        "included_in_denominator": True,
        "evaluation": evaluation,
    }
    record = CoreTraceRecord.model_validate(
        {**unsigned, "trace_sha256": _hash_payload(unsigned)}, strict=True
    )
    # Re-read both frozen handles before returning a joined trace.  This closes
    # the window in which an on-disk artifact could change after the first gate.
    _verified_inputs_and_run(inputs, run)
    return record


__all__ = [
    "CoreTraceError",
    "CoreTraceEvaluation",
    "CoreTraceRecord",
    "CoreTraceRuleScore",
    "build_core_trace_record",
]
