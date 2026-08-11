"""Focused typed smoke tests for the minimal S3 body refiner."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from skillchain.evaluation.packets import (
    JudgeDimensionScore,
    JudgeOutcome,
    JudgeScores,
)
from skillchain.evolution.attribution import attribute_core_trace
from skillchain.evolution.body_refiner import (
    BodyRefinementError,
    build_body_candidate,
    evaluate_body_candidate,
    make_body_gate_row,
    make_route_trace_comparison,
)
from skillchain.evolution.models import build_core_trace_record
from skillchain.evolution.route_optimizer import make_route_skill_surface

_ROOT = Path(__file__).resolve().parents[2]
_CORE_TRACE_HELPERS = runpy.run_path(
    str(_ROOT / "tests" / "evolution" / "test_core_trace_readiness.py")
)


def _judge_outcome() -> JudgeOutcome:
    evaluation_id = "a" * 64
    scores = JudgeScores(
        evaluation_id=evaluation_id,
        requires_card=False,
        dimensions=(
            JudgeDimensionScore(dimension="CA", score=10, tier="Good"),
            JudgeDimensionScore(dimension="CQ", score=16, tier="Good"),
            JudgeDimensionScore(dimension="TCR", score=10, tier="Good"),
        ),
        j_project=90.0,
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


def _body_example(tmp_path, monkeypatch, canonical_registry_factory, *, split="opt_pool"):
    inputs, run = _CORE_TRACE_HELPERS["_runner_bundle"](
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="success",
    )
    trace = build_core_trace_record(
        inputs,
        run,
        query_ordinal=0,
        allow_judge_outcomes=True,
        judge_outcome=_judge_outcome(),
    )
    return attribute_core_trace(trace, source_split=split)


def _surfaces(*, description_drift: bool = False):
    baseline = (
        make_route_skill_surface(
            slug="document-skill",
            capability_id="utility.document_reading",
            route="document OCR",
            description="Read documents and receipts.",
            body="Read the supplied document.\n",
            static_refs=("refs/document.md",),
            operators=("text_product_search",),
        ),
    )
    candidate = (
        make_route_skill_surface(
            slug="document-skill",
            capability_id="utility.document_reading",
            route="document OCR",
            description=(
                "Changed description." if description_drift else "Read documents and receipts."
            ),
            body="Read the supplied document and cite fields.\n",
            static_refs=("refs/document.md",),
            operators=("text_product_search",),
        ),
    )
    return baseline, candidate


def _gate_rows(*, safety_passed: bool = True):
    route_hash = "1" * 64
    baseline = (
        make_body_gate_row(
            query_id="q-a",
            component_id="component-a",
            run_id="s1s2-run",
            config="s1s2",
            route_trace_sha256=route_hash,
            j_project=75.0,
            grounding_passed=True,
            safety_passed=True,
        ),
    )
    candidate = (
        make_body_gate_row(
            query_id="q-a",
            component_id="component-a",
            run_id="full-run",
            config="full",
            route_trace_sha256=route_hash,
            j_project=85.0,
            grounding_passed=True,
            safety_passed=safety_passed,
        ),
    )
    return baseline, candidate


def test_body_candidate_accepts_when_hard_gates_hold_and_j_project_improves(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    example = _body_example(tmp_path, monkeypatch, canonical_registry_factory)
    baseline_skills, candidate_skills = _surfaces()
    candidate = build_body_candidate(
        baseline_bank_sha256="a" * 64,
        candidate_bank_sha256="b" * 64,
        baseline_skills=baseline_skills,
        candidate_skills=candidate_skills,
        opt_pool_examples=(example,),
    )
    baseline_rows, candidate_rows = _gate_rows()
    comparison = make_route_trace_comparison(
        query_id="q-a",
        component_id="component-a",
        s1s2_route_trace_sha256="1" * 64,
        full_route_trace_sha256="1" * 64,
    )

    decision = evaluate_body_candidate(
        candidate,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        route_trace_comparisons=(comparison,),
    )
    assert decision.outcome == "accept"
    assert decision.soft_j_project_improved is True
    assert decision.route_trace_gate_passed is True
    assert decision.route_hash_preserved is True


def test_body_candidate_rolls_back_on_safety_or_route_trace_regression(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    example = _body_example(tmp_path, monkeypatch, canonical_registry_factory)
    baseline_skills, candidate_skills = _surfaces()
    candidate = build_body_candidate(
        baseline_bank_sha256="a" * 64,
        candidate_bank_sha256="b" * 64,
        baseline_skills=baseline_skills,
        candidate_skills=candidate_skills,
        opt_pool_examples=(example,),
    )
    baseline_rows, unsafe_rows = _gate_rows(safety_passed=False)
    matching_comparison = make_route_trace_comparison(
        query_id="q-a",
        component_id="component-a",
        s1s2_route_trace_sha256="1" * 64,
        full_route_trace_sha256="1" * 64,
    )
    safety_rollback = evaluate_body_candidate(
        candidate,
        baseline_rows=baseline_rows,
        candidate_rows=unsafe_rows,
        route_trace_comparisons=(matching_comparison,),
    )
    assert safety_rollback.outcome == "rollback"
    assert safety_rollback.safety_hard_gate_passed is False

    _, candidate_rows = _gate_rows()
    mismatch = make_route_trace_comparison(
        query_id="q-a",
        component_id="component-a",
        s1s2_route_trace_sha256="1" * 64,
        full_route_trace_sha256="2" * 64,
    )
    route_rollback = evaluate_body_candidate(
        candidate,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        route_trace_comparisons=(mismatch,),
    )
    assert route_rollback.outcome == "rollback"
    assert route_rollback.route_trace_gate_passed is False


def test_body_candidate_rejects_test_frozen_and_surface_drift(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    frozen_example = _body_example(
        tmp_path, monkeypatch, canonical_registry_factory, split="test_frozen"
    )
    baseline_skills, candidate_skills = _surfaces()
    with pytest.raises(BodyRefinementError, match="only opt_pool"):
        build_body_candidate(
            baseline_bank_sha256="a" * 64,
            candidate_bank_sha256="b" * 64,
            baseline_skills=baseline_skills,
            candidate_skills=candidate_skills,
            opt_pool_examples=(frozen_example,),
        )

    _, description_drifted = _surfaces(description_drift=True)
    with pytest.raises(BodyRefinementError, match="route/description/static refs/operators drifted"):
        build_body_candidate(
            baseline_bank_sha256="a" * 64,
            candidate_bank_sha256="b" * 64,
            baseline_skills=baseline_skills,
            candidate_skills=description_drifted,
            opt_pool_examples=(frozen_example,),
        )
