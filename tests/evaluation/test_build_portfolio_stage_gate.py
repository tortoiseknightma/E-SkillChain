from __future__ import annotations

import pytest
from types import SimpleNamespace

from scripts import build_portfolio_stage_gate as gate
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PORTFOLIO_OPTIMIZATION_QUERY_IDS,
)
from skillchain.tools.serialization import sha256_bytes


def _sha(label: str) -> str:
    return sha256_bytes(label.encode())


def _loaded(
    *,
    source_config: str,
    observations: tuple[gate.GateObservation, ...],
) -> gate.LoadedSmoke:
    return gate.LoadedSmoke(
        source_config=source_config,
        bank_sha256=_sha(f"{source_config}:bank"),
        adherence_contract_bank_sha256=_sha("parent:bank"),
        evaluation_query_ids=PORTFOLIO_OPTIMIZATION_QUERY_IDS,
        rubric_file_sha256=PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
        shared_route_artifact_sha256s=(
            tuple(_sha(f"route:{query_id}") for query_id in PORTFOLIO_OPTIMIZATION_QUERY_IDS)
            if source_config in {"s1s2", "full"}
            else (None,) * len(PORTFOLIO_OPTIMIZATION_QUERY_IDS)
        ),
        source_summary_file_sha256=_sha(f"{source_config}:summary-file"),
        source_summary_sha256=_sha(f"{source_config}:summary"),
        source_results_file_sha256=_sha(f"{source_config}:results-file"),
        source_result_sha256s=tuple(
            item.source_result_sha256 for item in observations
        ),
        observations=observations,
    )


def test_s2_causal_gate_reuses_only_unchanged_selected_routes() -> None:
    parent = tuple(
        gate.GateObservation(
            query_id=query_id,
            selected_capability="product.multi_search",
            route_correct=False,
            j_project=50.0,
            skill_adherence=0.6,
            assistant_hard_error=False,
            evaluator_anomaly=False,
            source_result_sha256=_sha(f"parent:{query_id}"),
            shared_route_artifact_sha256=None,
        )
        for query_id in PORTFOLIO_OPTIMIZATION_QUERY_IDS
    )
    candidate = tuple(
        gate.GateObservation(
            query_id=query_id,
            selected_capability=(
                "product.style_recommendation"
                if query_id == PORTFOLIO_OPTIMIZATION_QUERY_IDS[-1]
                else "product.multi_search"
            ),
            route_correct=query_id == PORTFOLIO_OPTIMIZATION_QUERY_IDS[-1],
            j_project=(60.0 if query_id == PORTFOLIO_OPTIMIZATION_QUERY_IDS[-1] else 0.0),
            skill_adherence=(0.8 if query_id == PORTFOLIO_OPTIMIZATION_QUERY_IDS[-1] else 0.0),
            assistant_hard_error=False,
            evaluator_anomaly=False,
            source_result_sha256=_sha(f"candidate:{query_id}"),
            shared_route_artifact_sha256=_sha(f"route:{query_id}"),
        )
        for query_id in PORTFOLIO_OPTIMIZATION_QUERY_IDS
    )

    result = gate._causal_candidate_result(
        config="s1s2",
        parent_loaded=_loaded(source_config="s1", observations=parent),
        candidate_loaded=_loaded(source_config="s1s2", observations=candidate),
        parent_bank=None,  # S2 causal scope depends only on selected route.
        candidate_bank=None,
    )

    assert result.causal_reuse_unaffected is True
    assert sum(
        item.metric_source == "reused_parent" for item in result.row_provenance
    ) == 24
    assert result.row_provenance[-1].metric_source == "candidate"
    assert result.route_accuracy == pytest.approx(1 / 25)
    assert result.mean_j == pytest.approx(50.4)
    assert result.mean_skill_adherence == pytest.approx(0.608)


def test_s3_causal_gate_uses_candidate_only_for_changed_selected_body() -> None:
    changed_queries = set(PORTFOLIO_OPTIMIZATION_QUERY_IDS[:2])
    parent = tuple(
        gate.GateObservation(
            query_id=query_id,
            selected_capability=(
                "product.multi_search"
                if query_id in changed_queries
                else "product.exact_match"
            ),
            route_correct=True,
            j_project=50.0,
            skill_adherence=0.6,
            assistant_hard_error=False,
            evaluator_anomaly=False,
            source_result_sha256=_sha(f"s3-parent:{query_id}"),
            shared_route_artifact_sha256=_sha(f"route:{query_id}"),
        )
        for query_id in PORTFOLIO_OPTIMIZATION_QUERY_IDS
    )
    candidate = tuple(
        gate.GateObservation(
            query_id=item.query_id,
            selected_capability=item.selected_capability,
            route_correct=True,
            j_project=60.0,
            skill_adherence=0.8,
            assistant_hard_error=False,
            evaluator_anomaly=False,
            source_result_sha256=_sha(f"s3-candidate:{item.query_id}"),
            shared_route_artifact_sha256=item.shared_route_artifact_sha256,
        )
        for item in parent
    )
    parent_bank = SimpleNamespace(
        skills=(
            SimpleNamespace(
                capability_id="product.multi_search",
                body="parent multi body",
                operators=("multi_product_search",),
            ),
            SimpleNamespace(
                capability_id="product.exact_match",
                body="unchanged exact body",
                operators=("text_product_search",),
            ),
        )
    )
    candidate_bank = SimpleNamespace(
        skills=(
            SimpleNamespace(
                capability_id="product.multi_search",
                body="refined multi body",
                operators=("multi_product_search",),
            ),
            parent_bank.skills[1],
        )
    )

    result = gate._causal_candidate_result(
        config="full",
        parent_loaded=_loaded(source_config="s1s2", observations=parent),
        candidate_loaded=_loaded(source_config="full", observations=candidate),
        parent_bank=parent_bank,
        candidate_bank=candidate_bank,
    )

    assert [
        item.query_id
        for item in result.row_provenance
        if item.metric_source == "candidate"
    ] == list(PORTFOLIO_OPTIMIZATION_QUERY_IDS[:2])
    assert sum(
        item.metric_source == "reused_parent" for item in result.row_provenance
    ) == 23
    assert result.mean_j == pytest.approx(50.8)
