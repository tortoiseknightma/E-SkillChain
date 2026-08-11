"""Focused zero-I/O tests for opt800 folds and Creator selection v2."""

from __future__ import annotations

import runpy
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.portfolio_core_selection import (
    CreatorSelectionError,
    build_core_r2_creator_selection,
    canonical_creator_selection_index_bytes,
)
from skillchain.synthesis.portfolio_opt_folds import (
    CREATOR_V2_DOCUMENT_QUOTA,
    DISCOVERY_FOLDS,
    GROUP_FIELDS,
    OPT_FOLD_IDS,
    OptFoldError,
    OptFoldQuery,
    audit_legacy_selection_conflict,
    build_opt_creator_selection_v2,
    build_opt_fold_plan,
    canonical_opt_fold_mapping_bytes,
    canonical_opt_projection_bytes,
    compute_creator_v2_quotas,
    validate_opt_creator_selection_v2,
    validate_opt_fold_plan,
)
from skillchain.synthesis.store import sha256_bytes


@pytest.fixture(scope="module")
def opt_inputs():
    namespace = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    legacy = build_core_r2_creator_selection(
        result,
        bridge.realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    queries = tuple(
        OptFoldQuery(
            query_id=row.plan_id,
            leakage_group_id=row.leakage_group_id,
            boundary_group_id=row.boundary_group_id,
            template_family=row.template_family,
            generator_batch_id=row.generator_batch_id,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
        )
        for row in result.plan.queries
        if result.final_split_by_plan_id[row.plan_id] == "opt_pool"
    )
    legacy_ids = tuple(entry.plan_id for entry in legacy.entries)
    return result, plan_sha256, bridge.realism_sidecar, legacy, queries, legacy_ids


@pytest.fixture(scope="module")
def fold_plan(opt_inputs):
    _, plan_sha256, _, legacy, queries, legacy_ids = opt_inputs
    return build_opt_fold_plan(
        queries,
        legacy_selected_query_ids=legacy_ids,
        seed=20260808,
        source_queries_sha256="a" * 64,
        trusted_plan_sha256=plan_sha256,
        legacy_selection_bytes_sha256=legacy.manifest.selection_bytes_sha256,
    )


@pytest.fixture(scope="module")
def creator_v2(opt_inputs, fold_plan):
    result, plan_sha256, realism_sidecar, _, _, _ = opt_inputs
    return build_opt_creator_selection_v2(
        result,
        realism_sidecar,
        fold_plan,
        trusted_plan_sha256=plan_sha256,
        seed=20260808,
    )


def test_historical_selection_closes_all_opt_batches(opt_inputs) -> None:
    _, _, _, legacy, queries, legacy_ids = opt_inputs
    audit = audit_legacy_selection_conflict(
        reversed(queries),
        reversed(legacy_ids),
        selection_bytes_sha256=legacy.manifest.selection_bytes_sha256,
    )

    assert audit.selected_query_count == 240
    assert audit.touched_atomic_batch_count == 32
    assert audit.closure_query_count == 800
    assert audit.remaining_replay_count == 0
    assert audit.incompatible_with_nonempty_replay is True


def test_fold_plan_is_four_complete_balanced_group_aware_folds(
    opt_inputs,
    fold_plan,
) -> None:
    _, plan_sha256, _, legacy, queries, _ = opt_inputs
    fold_by_query = {row.query_id: row.fold_id for row in fold_plan.assignments}

    assert fold_plan.audit.fold_query_counts == {fold: 200 for fold in OPT_FOLD_IDS}
    assert fold_plan.audit.fold_batch_counts == {fold: 8 for fold in OPT_FOLD_IDS}
    assert len([row for row in fold_plan.assignments if row.role == "replay"]) == 200
    assert len([row for row in fold_plan.assignments if row.role == "discovery"]) == 600
    assert {
        row.fold_id for row in fold_plan.assignments if row.role == "discovery"
    } == set(DISCOVERY_FOLDS)
    assert fold_plan.manifest.plan_sha256 == plan_sha256
    assert fold_plan.manifest.legacy_conflict_audit.selection_bytes_sha256 == (
        legacy.manifest.selection_bytes_sha256
    )
    assert fold_plan.manifest.mapping_sha256 == sha256_bytes(
        canonical_opt_fold_mapping_bytes(fold_plan.assignments)
    )
    assert fold_plan.manifest.opt_projection_sha256 == sha256_bytes(
        canonical_opt_projection_bytes(queries)
    )

    for field_name in GROUP_FIELDS:
        folds_by_value: dict[str, set[str]] = defaultdict(set)
        for query in queries:
            value = getattr(query, field_name)
            if value is not None:
                folds_by_value[value].add(fold_by_query[query.query_id])
        assert all(len(folds) == 1 for folds in folds_by_value.values())


def test_fold_plan_is_input_order_invariant_and_fail_closed(
    opt_inputs,
    fold_plan,
) -> None:
    _, plan_sha256, _, legacy, queries, legacy_ids = opt_inputs
    repeated = build_opt_fold_plan(
        reversed(queries),
        legacy_selected_query_ids=reversed(legacy_ids),
        seed=20260808,
        source_queries_sha256="a" * 64,
        trusted_plan_sha256=plan_sha256,
        legacy_selection_bytes_sha256=legacy.manifest.selection_bytes_sha256,
    )
    assert repeated == fold_plan

    first = fold_plan.assignments[0]
    changed_fold = "fold-02" if first.fold_id != "fold-02" else "fold-03"
    changed_role = "replay" if changed_fold == "fold-00" else "discovery"
    altered_assignments = (
        first.model_copy(update={"fold_id": changed_fold, "role": changed_role}),
        *fold_plan.assignments[1:],
    )
    altered = replace(fold_plan, assignments=altered_assignments)
    with pytest.raises(OptFoldError, match="deterministic optimum"):
        validate_opt_fold_plan(
            altered,
            queries,
            legacy_selected_query_ids=legacy_ids,
            source_queries_sha256="a" * 64,
            trusted_plan_sha256=plan_sha256,
            legacy_selection_bytes_sha256=legacy.manifest.selection_bytes_sha256,
        )


def test_creator_v2_is_discovery_only_component_unique_and_quota_bound(
    opt_inputs,
    fold_plan,
    creator_v2,
) -> None:
    _, _, _, legacy, queries, _ = opt_inputs
    role_by_id = {row.query_id: row.role for row in fold_plan.assignments}
    query_by_id = {row.query_id: row for row in queries}
    selected_ids = {row.plan_id for row in creator_v2.entries}
    replay_ids = {
        row.query_id for row in fold_plan.assignments if row.role == "replay"
    }
    replay_components = {
        query_by_id[query_id].leakage_group_id for query_id in replay_ids
    }

    assert len(creator_v2.entries) == 240
    assert len({row.component_id for row in creator_v2.entries}) == 240
    assert all(role_by_id[query_id] == "discovery" for query_id in selected_ids)
    assert not ({row.component_id for row in creator_v2.entries} & replay_components)
    assert creator_v2.audit.candidate_count == 600
    assert creator_v2.audit.quota_by_capability[
        "utility.document_reading"
    ] == CREATOR_V2_DOCUMENT_QUOTA
    assert creator_v2.audit.quota_by_capability == compute_creator_v2_quotas(
        creator_v2.audit.available_unique_components_by_capability
    )
    assert creator_v2.manifest.supersedes_selection_bytes_sha256 == (
        legacy.manifest.selection_bytes_sha256
    )
    assert creator_v2.manifest.fold_mapping_sha256 == (
        fold_plan.manifest.mapping_sha256
    )


def test_creator_v2_validation_and_v1_bytes_remain_stable(
    opt_inputs,
    fold_plan,
    creator_v2,
) -> None:
    result, plan_sha256, realism_sidecar, legacy, _, _ = opt_inputs
    v1_bytes_before = canonical_creator_selection_index_bytes(legacy.entries)

    validate_opt_creator_selection_v2(
        creator_v2,
        result,
        realism_sidecar,
        fold_plan,
        trusted_plan_sha256=plan_sha256,
    )
    assert canonical_creator_selection_index_bytes(legacy.entries) == v1_bytes_before
    assert sha256_bytes(v1_bytes_before) == legacy.manifest.selection_bytes_sha256

    altered_entries = list(creator_v2.entries)
    altered_entries[0] = altered_entries[0].model_copy(
        update={"component_id": altered_entries[1].component_id}
    )
    altered = replace(creator_v2, entries=tuple(altered_entries))
    with pytest.raises(CreatorSelectionError, match="deterministic optimum"):
        validate_opt_creator_selection_v2(
            altered,
            result,
            realism_sidecar,
            fold_plan,
            trusted_plan_sha256=plan_sha256,
        )


def test_fold_planner_rejects_incomplete_or_split_group(opt_inputs) -> None:
    _, plan_sha256, _, legacy, queries, legacy_ids = opt_inputs
    with pytest.raises(OptFoldError, match="exactly 800"):
        build_opt_fold_plan(
            queries[:-1],
            legacy_selected_query_ids=legacy_ids,
            source_queries_sha256="a" * 64,
            trusted_plan_sha256=plan_sha256,
            legacy_selection_bytes_sha256=legacy.manifest.selection_bytes_sha256,
        )
