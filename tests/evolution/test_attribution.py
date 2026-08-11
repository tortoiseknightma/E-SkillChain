"""Focused tests for deterministic r2 evolution attribution."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest
from pydantic import ValidationError

from skillchain.evaluation.packets import (
    JudgeDimensionScore,
    JudgeOutcome,
    JudgeScores,
)
from skillchain.evolution.attribution import (
    AttributionError,
    attribute_core_trace,
    routing_opt_pool_examples,
)
from skillchain.evolution.models import (
    CoreTraceRecord,
    CoreTraceRuleScore,
    build_core_trace_record,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

_ROOT = Path(__file__).resolve().parents[2]
_CORE_TRACE_HELPERS = runpy.run_path(
    str(_ROOT / "tests" / "evolution" / "test_core_trace_readiness.py")
)


def _judge_outcome(*, j_project: float = 90.0) -> JudgeOutcome:
    evaluation_id = "a" * 64
    scores = JudgeScores(
        evaluation_id=evaluation_id,
        requires_card=False,
        dimensions=(
            JudgeDimensionScore(dimension="CA", score=10, tier="Good"),
            JudgeDimensionScore(dimension="CQ", score=16, tier="Good"),
            JudgeDimensionScore(dimension="TCR", score=10, tier="Good"),
        ),
        j_project=j_project,
    )
    return JudgeOutcome(
        evaluation_id=evaluation_id,
        status="scored",
        attempts=1,
        max_attempts=1,
        scores=scores,
        judge_provider="fixture",
        judge_model="fixture-v1",
        prompt_sha256="b" * 64,
        raw_response_sha256="c" * 64,
    )


def _record(tmp_path, monkeypatch, canonical_registry_factory, *, scenario: str):
    inputs, run = _CORE_TRACE_HELPERS["_runner_bundle"](
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario=scenario,
    )
    return inputs, run


def _rehash_trace(record: CoreTraceRecord, **updates: object) -> CoreTraceRecord:
    python_payload = record.model_dump(mode="python")
    python_payload.update(updates)
    python_payload.pop("trace_sha256")
    json_payload = record.model_dump(mode="json")
    json_payload.update(updates)
    json_payload.pop("trace_sha256")
    python_payload["trace_sha256"] = sha256_bytes(canonical_json_bytes(json_payload))
    return CoreTraceRecord.model_validate(python_payload, strict=True)


def test_attribution_is_mutually_exclusive_and_keeps_tool_infra_out_of_body(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    route_inputs, route_run = _record(
        tmp_path, monkeypatch, canonical_registry_factory, scenario="route_failed"
    )
    route_trace = build_core_trace_record(route_inputs, route_run, query_ordinal=0)
    route = attribute_core_trace(route_trace, source_split="opt_pool")
    assert route.attribution == "routing"
    assert route.training_stage == "s2"

    tool_inputs, tool_run = _record(
        tmp_path, monkeypatch, canonical_registry_factory, scenario="tool_failed"
    )
    tool_trace = build_core_trace_record(tool_inputs, tool_run, query_ordinal=0)
    tool = attribute_core_trace(tool_trace, source_split="opt_pool")
    assert tool.attribution == "tool"
    assert tool.training_stage is None

    success_inputs, success_run = _record(
        tmp_path, monkeypatch, canonical_registry_factory, scenario="success"
    )
    scored_trace = build_core_trace_record(
        success_inputs,
        success_run,
        query_ordinal=0,
        allow_judge_outcomes=True,
        judge_outcome=_judge_outcome(),
    )
    body = attribute_core_trace(scored_trace, source_split="opt_pool")
    assert body.attribution == "body"
    assert body.training_stage == "s3"

    infra = attribute_core_trace(
        _rehash_trace(scored_trace, error_code="runtime_error"), source_split="opt_pool"
    )
    assert infra.attribution == "infra_error"
    assert infra.training_stage is None

    safety_trace = build_core_trace_record(
        success_inputs,
        success_run,
        query_ordinal=0,
        allow_judge_outcomes=True,
        rule_scores=(CoreTraceRuleScore(rule_id="safety-check", score=0.0),),
    )
    safety = attribute_core_trace(safety_trace, source_split="opt_pool")
    assert safety.attribution == "safety"
    assert safety.training_stage is None

    assert tool.attribution != "body"
    assert infra.attribution != "body"
    assert body.component_sha256 != body.run_sha256
    assert len(body.binding_sha256) == 64


def test_attribution_tamper_and_test_split_access_fail_closed(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    inputs, run = _record(
        tmp_path, monkeypatch, canonical_registry_factory, scenario="route_failed"
    )
    trace = build_core_trace_record(inputs, run, query_ordinal=0)
    opt_example = attribute_core_trace(trace, source_split="opt_pool")
    frozen_example = attribute_core_trace(trace, source_split="test_frozen")

    assert routing_opt_pool_examples((opt_example,)) == (opt_example,)
    with pytest.raises(AttributionError, match="only opt_pool"):
        routing_opt_pool_examples((frozen_example,))

    tampered = opt_example.model_copy(update={"component_id": "tampered-component"})
    with pytest.raises(AttributionError, match="self hash"):
        routing_opt_pool_examples((tampered,))

    with pytest.raises(ValidationError, match="self hash"):
        CoreTraceRecord.model_validate(
            trace.model_dump(mode="python") | {"query_id": "tampered-query"},
            strict=True,
        )
