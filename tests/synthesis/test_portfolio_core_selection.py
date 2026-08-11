"""Focused pure-metadata tests for the r2 S1 Creator selection index."""

from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.portfolio_core_selection import (
    CREATOR_SELECTION_SIZE,
    CreatorSelectionError,
    build_core_r2_creator_selection,
    canonical_creator_selection_index_bytes,
    compute_component_aware_quotas,
    validate_core_r2_creator_selection,
)
from skillchain.synthesis.store import sha256_bytes


@pytest.fixture(scope="module")
def r2_inputs():
    """Use the complete synthetic r2 planner and bridge fixtures without E:."""

    namespace = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    return result, plan_sha256, bridge.realism_sidecar


@pytest.fixture(scope="module")
def selection(r2_inputs):
    result, plan_sha256, realism_sidecar = r2_inputs
    return build_core_r2_creator_selection(
        result,
        realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )


def test_creator_selection_is_opt_only_component_unique_and_capacity_aware(
    r2_inputs,
    selection,
) -> None:
    result, plan_sha256, realism_sidecar = r2_inputs
    selected_plan_ids = {entry.plan_id for entry in selection.entries}
    selected_components = {entry.component_id for entry in selection.entries}
    available = {
        capability: len(
            {
                row.leakage_group_id
                for row in result.plan.queries
                if result.final_split_by_plan_id[row.plan_id] == "opt_pool"
                and row.canonical_capability == capability
            }
        )
        for capability in selection.audit.quota_by_capability
    }

    assert len(selection.entries) == CREATOR_SELECTION_SIZE
    assert len(selected_plan_ids) == CREATOR_SELECTION_SIZE
    assert len(selected_components) == CREATOR_SELECTION_SIZE
    assert all(entry.final_split == "opt_pool" for entry in selection.entries)
    assert all(
        result.final_split_by_plan_id[entry.plan_id] == "opt_pool"
        for entry in selection.entries
    )
    assert selection.manifest.plan_sha256 == plan_sha256
    assert selection.manifest.realism_assignments_sha256 == (
        realism_sidecar.manifest.assignments_sha256
    )
    assert selection.manifest.selection_bytes_sha256 == sha256_bytes(
        canonical_creator_selection_index_bytes(selection.entries)
    )
    assert selection.audit.available_unique_components_by_capability == available
    assert selection.audit.quota_by_capability == compute_component_aware_quotas(
        available
    )
    assert selection.audit.selected_by_capability == selection.audit.quota_by_capability
    assert sum(selection.audit.quota_by_capability.values()) == CREATOR_SELECTION_SIZE
    capacity_limited = [
        capability for capability, count in available.items() if count < 40
    ]
    assert capacity_limited
    assert all(
        selection.audit.quota_by_capability[capability] == available[capability]
        for capability in capacity_limited
    )
    assert all(
        count >= 1
        for count in selection.audit.selected_interaction_boundary_counts.values()
    )
    assert sum(selection.audit.target_interaction_boundary_counts.values()) == 240


def test_creator_selection_is_deterministic_for_the_same_seed(
    r2_inputs,
    selection,
) -> None:
    result, plan_sha256, realism_sidecar = r2_inputs

    repeated = build_core_r2_creator_selection(
        result,
        realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )

    assert repeated == selection
    assert canonical_creator_selection_index_bytes(
        repeated.entries
    ) == canonical_creator_selection_index_bytes(selection.entries)


@pytest.mark.parametrize("bad_split", ("dev_mini", "val", "test_frozen"))
def test_creator_selection_rejects_non_opt_pool_index_rows(
    r2_inputs,
    selection,
    bad_split: str,
) -> None:
    result, plan_sha256, realism_sidecar = r2_inputs
    altered_entries = list(selection.entries)
    altered_entries[0] = altered_entries[0].model_copy(
        update={"final_split": bad_split}
    )
    altered_entries = tuple(altered_entries)
    altered_manifest = selection.manifest.model_copy(
        update={
            "selection_bytes_sha256": sha256_bytes(
                canonical_creator_selection_index_bytes(altered_entries)
            )
        }
    )
    altered_selection = replace(
        selection,
        entries=altered_entries,
        manifest=altered_manifest,
    )

    with pytest.raises(CreatorSelectionError, match="only contain opt_pool"):
        validate_core_r2_creator_selection(
            altered_selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )


@pytest.mark.parametrize("forbidden_split", ("dev_mini", "val", "test_frozen"))
def test_creator_selection_rejects_non_opt_pool_plan_inputs(
    r2_inputs,
    selection,
    forbidden_split: str,
) -> None:
    result, plan_sha256, realism_sidecar = r2_inputs
    forbidden_row = next(
        row
        for row in result.plan.queries
        if result.final_split_by_plan_id[row.plan_id] == forbidden_split
    )
    assignment_by_plan_id = {
        assignment.plan_id: assignment for assignment in realism_sidecar.assignments
    }
    assignment = assignment_by_plan_id[forbidden_row.plan_id]
    altered_entries = list(selection.entries)
    altered_entries[0] = altered_entries[0].model_copy(
        update={
            "plan_id": forbidden_row.plan_id,
            "component_id": forbidden_row.leakage_group_id,
            "canonical_capability": forbidden_row.canonical_capability,
            "is_boundary": forbidden_row.is_boundary,
            "interaction_pattern": assignment.interaction_pattern,
        }
    )
    altered_entries = tuple(altered_entries)
    altered_selection = replace(
        selection,
        entries=altered_entries,
        manifest=selection.manifest.model_copy(
            update={
                "selection_bytes_sha256": sha256_bytes(
                    canonical_creator_selection_index_bytes(altered_entries)
                )
            }
        ),
    )

    with pytest.raises(CreatorSelectionError, match="non-opt-pool plan_id"):
        validate_core_r2_creator_selection(
            altered_selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )


def test_creator_selection_rejects_duplicate_component_and_tampered_realism(
    r2_inputs,
    selection,
) -> None:
    result, plan_sha256, realism_sidecar = r2_inputs
    altered_entries = list(selection.entries)
    altered_entries[1] = altered_entries[1].model_copy(
        update={"component_id": altered_entries[0].component_id}
    )
    altered_entries = tuple(altered_entries)
    altered_selection = replace(
        selection,
        entries=altered_entries,
        manifest=selection.manifest.model_copy(
            update={
                "selection_bytes_sha256": sha256_bytes(
                    canonical_creator_selection_index_bytes(altered_entries)
                )
            }
        ),
    )
    with pytest.raises(CreatorSelectionError, match="duplicate component"):
        validate_core_r2_creator_selection(
            altered_selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )

    with pytest.raises(CreatorSelectionError, match="bytes digest mismatch"):
        validate_core_r2_creator_selection(
            replace(
                selection,
                manifest=selection.manifest.model_copy(
                    update={"selection_bytes_sha256": "f" * 64}
                ),
            ),
            result,
            realism_sidecar,
            trusted_plan_sha256=plan_sha256,
        )

    tampered_realism = replace(
        realism_sidecar,
        manifest=realism_sidecar.manifest.model_copy(update={"plan_sha256": "0" * 64}),
    )
    with pytest.raises(CreatorSelectionError, match="realism sidecar binding mismatch"):
        build_core_r2_creator_selection(
            result,
            tampered_realism,
            trusted_plan_sha256=plan_sha256,
            seed=20260805,
        )
