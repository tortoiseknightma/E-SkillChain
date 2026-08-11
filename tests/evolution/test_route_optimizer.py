"""Focused typed smoke tests for the minimal S2 route optimizer."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from skillchain.evolution.attribution import attribute_core_trace
from skillchain.evolution.models import build_core_trace_record
from skillchain.evolution.route_optimizer import (
    RouteOptimizationError,
    build_route_candidate,
    evaluate_route_candidate,
    make_route_gate_row,
    make_route_skill_surface,
)

_ROOT = Path(__file__).resolve().parents[2]
_CORE_TRACE_HELPERS = runpy.run_path(
    str(_ROOT / "tests" / "evolution" / "test_core_trace_readiness.py")
)


def _routing_example(tmp_path, monkeypatch, canonical_registry_factory, *, split="opt_pool"):
    inputs, run = _CORE_TRACE_HELPERS["_runner_bundle"](
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="route_failed",
    )
    return attribute_core_trace(
        build_core_trace_record(inputs, run, query_ordinal=0), source_split=split
    )


def _surfaces(*, changed_body: bool = False):
    baseline = (
        make_route_skill_surface(
            slug="document-skill",
            capability_id="utility.document_reading",
            route="document",
            description="Read documents.",
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
            description="Read documents and receipts.",
            body=("Changed body.\n" if changed_body else "Read the supplied document.\n"),
            static_refs=("refs/document.md",),
            operators=("text_product_search",),
        ),
    )
    return baseline, candidate


def _rows(*, improved: bool):
    baseline = (
        make_route_gate_row(
            query_id="q-a",
            component_id="component-a",
            canonical_capability="product.exact_match",
            selected_capability="utility.document_reading",
            run_id="baseline-run",
            config="s1",
            route_trace_sha256="1" * 64,
        ),
        make_route_gate_row(
            query_id="q-b",
            component_id="component-b",
            canonical_capability="utility.document_reading",
            selected_capability="utility.document_reading",
            run_id="baseline-run",
            config="s1",
            route_trace_sha256="2" * 64,
        ),
    )
    candidate = (
        make_route_gate_row(
            query_id="q-a",
            component_id="component-a",
            canonical_capability="product.exact_match",
            selected_capability=(
                "product.exact_match" if improved else "utility.document_reading"
            ),
            run_id="candidate-run",
            config="s1s2",
            route_trace_sha256="3" * 64,
        ),
        make_route_gate_row(
            query_id="q-b",
            component_id="component-b",
            canonical_capability="utility.document_reading",
            selected_capability=(
                "utility.document_reading" if improved else "product.exact_match"
            ),
            run_id="candidate-run",
            config="s1s2",
            route_trace_sha256="4" * 64,
        ),
    )
    return baseline, candidate


def test_route_candidate_accepts_non_regressing_two_metric_gate(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    example = _routing_example(tmp_path, monkeypatch, canonical_registry_factory)
    baseline_skills, candidate_skills = _surfaces()
    candidate = build_route_candidate(
        baseline_bank_sha256="a" * 64,
        candidate_bank_sha256="b" * 64,
        baseline_skills=baseline_skills,
        candidate_skills=candidate_skills,
        opt_pool_examples=(example,),
    )
    baseline_rows, candidate_rows = _rows(improved=True)

    decision = evaluate_route_candidate(
        candidate, baseline_rows=baseline_rows, candidate_rows=candidate_rows
    )
    assert decision.outcome == "accept"
    assert decision.candidate_metrics.capability_macro_f1 > (
        decision.baseline_metrics.capability_macro_f1
    )
    assert decision.candidate_metrics.component_macro_f1 > (
        decision.baseline_metrics.component_macro_f1
    )
    assert decision.body_hash_preserved is True


def test_route_candidate_rolls_back_on_regression_and_fails_closed_on_tamper(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    example = _routing_example(tmp_path, monkeypatch, canonical_registry_factory)
    baseline_skills, candidate_skills = _surfaces()
    candidate = build_route_candidate(
        baseline_bank_sha256="a" * 64,
        candidate_bank_sha256="b" * 64,
        baseline_skills=baseline_skills,
        candidate_skills=candidate_skills,
        opt_pool_examples=(example,),
    )
    baseline_rows, candidate_rows = _rows(improved=False)

    rollback = evaluate_route_candidate(
        candidate, baseline_rows=baseline_rows, candidate_rows=candidate_rows
    )
    assert rollback.outcome == "rollback"

    tampered = candidate.model_copy(update={"candidate_description_sha256": "f" * 64})
    with pytest.raises(RouteOptimizationError, match="candidate self hash or invariants"):
        evaluate_route_candidate(
            tampered, baseline_rows=baseline_rows, candidate_rows=candidate_rows
        )


def test_route_candidate_rejects_test_frozen_and_non_route_surface_drift(
    tmp_path, monkeypatch, canonical_registry_factory
) -> None:
    frozen_example = _routing_example(
        tmp_path, monkeypatch, canonical_registry_factory, split="test_frozen"
    )
    baseline_skills, candidate_skills = _surfaces()
    with pytest.raises(RouteOptimizationError, match="only opt_pool"):
        build_route_candidate(
            baseline_bank_sha256="a" * 64,
            candidate_bank_sha256="b" * 64,
            baseline_skills=baseline_skills,
            candidate_skills=candidate_skills,
            opt_pool_examples=(frozen_example,),
        )

    _, body_drifted_candidate = _surfaces(changed_body=True)
    with pytest.raises(RouteOptimizationError, match="body/static refs/operators drifted"):
        build_route_candidate(
            baseline_bank_sha256="a" * 64,
            candidate_bank_sha256="b" * 64,
            baseline_skills=baseline_skills,
            candidate_skills=body_drifted_candidate,
            opt_pool_examples=(frozen_example,),
        )
