"""Pure in-memory r2 creator-selection to legacy Stage 1 bundle coverage."""

from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.schemas import LabelDecision, Query
from skillchain.stage1 import (
    Stage1ContractError,
    build_core_r2_stage1_trajectory_bundle,
)
from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.portfolio_core_selection import (
    build_core_r2_creator_selection,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


@pytest.fixture(scope="module")
def r2_creator_inputs():
    namespace = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    selection = build_core_r2_creator_selection(
        result,
        bridge.realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    return result, plan_sha256, bridge.realism_sidecar, selection


def _accepted_queries(result, selection) -> tuple[Query, ...]:
    rows_by_plan_id = {row.plan_id: row for row in result.plan.queries}
    queries: list[Query] = []
    for entry in selection.entries:
        row = rows_by_plan_id[entry.plan_id]
        text = f"accepted Stage 1 request for {row.plan_id}"
        queries.append(
            Query(
                schema_version=2,
                taxonomy_version=row.taxonomy_version,
                task_spec_version=row.task_spec_version,
                query_id=row.plan_id,
                asset_id=row.asset_id,
                image_path=row.image_path,
                leakage_group_id=row.leakage_group_id,
                boundary_group_id=row.boundary_group_id,
                template_family=row.template_family,
                generator_batch_id=row.generator_batch_id,
                text=text,
                turns=[{"role": "user", "content": text}],
                canonical_intent=row.canonical_intent,
                canonical_capability=row.canonical_capability,
                acceptable_capabilities=row.acceptable_capabilities,
                is_boundary=row.is_boundary,
                boundary_strategy=row.boundary_strategy,
                requires_card=row.requires_card,
                split=result.final_split_by_plan_id[row.plan_id],
                label_status="auto",
                label_provenance=[
                    LabelDecision(
                        decision_type="constructed",
                        annotator_kind="planner",
                        annotator_id="r2-stage1-bridge-test",
                        canonical_intent=row.canonical_intent,
                        canonical_capability=row.canonical_capability,
                        acceptable_capabilities=row.acceptable_capabilities,
                    )
                ],
            )
        )
    return tuple(queries)


def test_r2_selection_projects_through_legacy_stage1_bundle_with_receipt(
    r2_creator_inputs,
) -> None:
    result, plan_sha256, realism_sidecar, selection = r2_creator_inputs
    accepted_queries = _accepted_queries(result, selection)

    projection = build_core_r2_stage1_trajectory_bundle(
        reversed(accepted_queries),
        selection,
        result,
        realism_sidecar,
        trusted_plan_sha256=plan_sha256,
    )
    ranked_ids = tuple(entry.plan_id for entry in selection.entries)

    assert len(projection.trajectory_bundle.trajectories) == 240
    assert {
        trajectory.query_id for trajectory in projection.trajectory_bundle.trajectories
    } == set(ranked_ids)
    assert projection.selection_receipt.plan_sha256 == plan_sha256
    assert projection.selection_receipt.selection_index_sha256 == (
        selection.manifest.selection_bytes_sha256
    )
    assert (
        projection.selection_receipt.selection_ranked_query_ids_sha256
        == sha256_bytes(canonical_json_bytes({"query_ids": list(ranked_ids)}))
    )
    assert projection.selection_receipt.accepted_corpus_sha256 == (
        projection.trajectory_bundle.accepted_corpus_sha256
    )
    assert projection.selection_receipt.trajectory_bundle_sha256 == (
        projection.trajectory_bundle.trajectory_bundle_sha256
    )


def test_r2_selection_bridge_rejects_missing_and_drifted_accepted_queries(
    r2_creator_inputs,
) -> None:
    result, plan_sha256, realism_sidecar, selection = r2_creator_inputs
    accepted_queries = list(_accepted_queries(result, selection))

    with pytest.raises(Stage1ContractError, match="absent from accepted corpus"):
        build_core_r2_stage1_trajectory_bundle(
            accepted_queries[1:],
            selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )

    drift_cases = (
        (
            {"leakage_group_id": "drifted-component"},
            "leakage component drifted",
        ),
        (
            {"canonical_capability": "drifted.capability"},
            "capability drifted",
        ),
        (
            {"split": "val"},
            "split must match opt_pool",
        ),
    )
    for update, message in drift_cases:
        drifted = list(accepted_queries)
        drifted[0] = drifted[0].model_copy(update=update)
        with pytest.raises(Stage1ContractError, match=message):
            build_core_r2_stage1_trajectory_bundle(
                drifted,
                selection,
                result,
                realism_sidecar,
                trusted_plan_sha256=plan_sha256,
            )


def test_r2_selection_bridge_rejects_tampered_selection_before_projection(
    r2_creator_inputs,
) -> None:
    result, plan_sha256, realism_sidecar, selection = r2_creator_inputs
    tampered_selection = replace(
        selection,
        manifest=selection.manifest.model_copy(
            update={"selection_bytes_sha256": "f" * 64}
        ),
    )

    with pytest.raises(Stage1ContractError, match="selection validation failed"):
        build_core_r2_stage1_trajectory_bundle(
            _accepted_queries(result, selection),
            tampered_selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )
