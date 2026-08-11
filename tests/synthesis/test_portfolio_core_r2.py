from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis.planning import MVP_CAPABILITY_ORDER
from skillchain.synthesis.portfolio_core_r2 import (
    R2CoreBridge,
    R2CoreBridgeError,
    build_r2_core_bridge,
    build_r2_split_constraints,
    extract_r2_val_interaction_by_query_id,
)


@pytest.fixture(scope="module")
def full_r2_result():
    """Reuse the full synthetic r2 planner fixture without touching E:."""

    fixture_path = Path(__file__).with_name("test_r2_planning.py")
    namespace = runpy.run_path(str(fixture_path))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    return result, plan_sha256


@pytest.fixture(scope="module")
def bridge(full_r2_result) -> R2CoreBridge:
    result, plan_sha256 = full_r2_result
    return build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )


def test_r2_bridge_projects_all_bindings_and_full_val_interactions(
    full_r2_result,
    bridge: R2CoreBridge,
) -> None:
    result, plan_sha256 = full_r2_result

    assert len(bridge.split_constraints.plan_id_to_split) == 1500
    assert bridge.split_constraints.plan_sha256 == plan_sha256
    assert (
        bridge.split_constraints.asset_catalog_sha256
        == result.plan.asset_catalog_sha256
    )
    assert (
        bridge.split_constraints.capability_assignments_sha256
        == result.plan.capability_assignments_sha256
    )
    assert len(bridge.split_constraints.abo_components) == 20
    assert len(bridge.split_constraints.food_components) == 10
    assert bridge.realism_sidecar.manifest.row_count == 1500
    assert bridge.realism_sidecar.manifest.plan_sha256 == plan_sha256
    assert len(bridge.val_interaction_by_query_id) == 200
    assert {
        plan_id
        for plan_id, split in bridge.split_constraints.plan_id_to_split.items()
        if split == "val"
    } == set(bridge.val_interaction_by_query_id)


def test_counterfactual_declarations_preserve_component_identity_and_capability_order(
    full_r2_result,
    bridge: R2CoreBridge,
) -> None:
    result, _ = full_r2_result
    row_by_plan_id = {row.plan_id: row for row in result.plan.queries}
    expected_capabilities = {
        "abo": (
            "product.exact_match",
            "product.style_recommendation",
            "knowledge.visual_encyclopedia",
        ),
        "food": (
            "knowledge.visual_encyclopedia",
            "utility.recipe_guidance",
        ),
    }
    ranks = {capability: index for index, capability in enumerate(MVP_CAPABILITY_ORDER)}

    for declaration in (
        *bridge.split_constraints.abo_components,
        *bridge.split_constraints.food_components,
    ):
        rows = [row_by_plan_id[plan_id] for plan_id in declaration.plan_ids]
        assert (
            tuple(row.canonical_capability for row in rows)
            == expected_capabilities[declaration.source]
        )
        assert list(declaration.plan_ids) == sorted(
            declaration.plan_ids,
            key=lambda plan_id: (
                ranks[row_by_plan_id[plan_id].canonical_capability],
                plan_id,
            ),
        )
        assert {row.leakage_group_id for row in rows} == {declaration.component_id}
        assert len({row.asset_id for row in rows}) == 1
        assert len({row.boundary_group_id for row in rows}) == 1


def test_r2_bridge_is_deterministic_for_the_same_audited_input(
    full_r2_result,
    bridge: R2CoreBridge,
) -> None:
    result, plan_sha256 = full_r2_result

    repeated = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )

    assert repeated == bridge


def test_r2_bridge_rejects_tampered_plan_trust_and_realism_bindings(
    full_r2_result,
    bridge: R2CoreBridge,
) -> None:
    result, plan_sha256 = full_r2_result

    with pytest.raises(R2CoreBridgeError, match="trusted plan SHA-256"):
        build_r2_split_constraints(
            result,
            trusted_plan_sha256="0" * 64,
        )

    changed_splits = dict(result.final_split_by_plan_id)
    first_val_plan_id = next(
        plan_id for plan_id, split in changed_splits.items() if split == "val"
    )
    changed_splits[first_val_plan_id] = "test_frozen"
    with pytest.raises(R2CoreBridgeError, match="plan audit failed"):
        build_r2_split_constraints(
            replace(result, final_split_by_plan_id=changed_splits),
            trusted_plan_sha256=plan_sha256,
        )

    bad_manifest = bridge.realism_sidecar.manifest.model_copy(
        update={"plan_sha256": "0" * 64}
    )
    with pytest.raises(R2CoreBridgeError, match="binding mismatch: plan_sha256"):
        extract_r2_val_interaction_by_query_id(
            result,
            replace(bridge.realism_sidecar, manifest=bad_manifest),
            trusted_plan_sha256=plan_sha256,
        )

    changed_recipes = list(bridge.realism_sidecar.recipes)
    changed_recipes[0] = changed_recipes[0].model_copy(
        update={"structural_slots": ("source-token-xyz",)}
    )
    with pytest.raises(R2CoreBridgeError, match="canonical prompt recipe"):
        extract_r2_val_interaction_by_query_id(
            result,
            replace(bridge.realism_sidecar, recipes=tuple(changed_recipes)),
            trusted_plan_sha256=plan_sha256,
        )
