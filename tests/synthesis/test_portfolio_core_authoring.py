"""Focused tests for r2 text-free authoring inputs.

These tests deliberately use only synthetic plan metadata.  They do not read
images and do not contain formal query text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from skillchain.synthesis.portfolio_core_authoring import (
    RealismAssignment,
    audit_realism_assignments,
    build_authoring_job,
    build_realism_sidecar,
    build_prompt_recipe_manifest,
    canonical_author_packet_bytes,
    canonical_realism_assignments_bytes,
    compute_generation_input_sha256,
    scan_author_packet_for_leaks,
    transition_authoring_checkpoint,
    validate_authoring_job,
    validate_realism_sidecar,
)
from skillchain.synthesis.store import sha256_bytes


_CAPABILITY_COUNTS = {
    "dev_mini": {
        "product.exact_match": 35,
        "product.multi_search": 35,
        "product.style_recommendation": 35,
        "knowledge.visual_encyclopedia": 35,
        "utility.document_reading": 30,
        "utility.recipe_guidance": 30,
    },
    "opt_pool": {
        "product.exact_match": 209,
        "product.multi_search": 117,
        "product.style_recommendation": 163,
        "knowledge.visual_encyclopedia": 163,
        "utility.document_reading": 32,
        "utility.recipe_guidance": 116,
    },
    "val": {
        "product.exact_match": 52,
        "product.multi_search": 29,
        "product.style_recommendation": 41,
        "knowledge.visual_encyclopedia": 41,
        "utility.document_reading": 10,
        "utility.recipe_guidance": 27,
    },
    "test_frozen": {
        "product.exact_match": 79,
        "product.multi_search": 44,
        "product.style_recommendation": 61,
        "knowledge.visual_encyclopedia": 61,
        "utility.document_reading": 18,
        "utility.recipe_guidance": 37,
    },
}
_BOUNDARY_COUNTS = {
    "dev_mini": {
        "product.exact_match": 11,
        "product.multi_search": 2,
        "product.style_recommendation": 10,
        "knowledge.visual_encyclopedia": 11,
        "utility.document_reading": 3,
        "utility.recipe_guidance": 3,
    },
    "opt_pool": {
        "product.exact_match": 34,
        "product.multi_search": 23,
        "product.style_recommendation": 26,
        "knowledge.visual_encyclopedia": 26,
        "utility.document_reading": 8,
        "utility.recipe_guidance": 20,
    },
    "val": {
        "product.exact_match": 8,
        "product.multi_search": 6,
        "product.style_recommendation": 7,
        "knowledge.visual_encyclopedia": 6,
        "utility.document_reading": 2,
        "utility.recipe_guidance": 5,
    },
    "test_frozen": {
        "product.exact_match": 13,
        "product.multi_search": 8,
        "product.style_recommendation": 10,
        "knowledge.visual_encyclopedia": 10,
        "utility.document_reading": 5,
        "utility.recipe_guidance": 6,
    },
}
_INTENT_BY_CAPABILITY = {
    "product.exact_match": "exact_match",
    "product.multi_search": "multi_product",
    "product.style_recommendation": "divergent_rec",
    "knowledge.visual_encyclopedia": "encyclopedia",
    "utility.document_reading": "utility",
    "utility.recipe_guidance": "utility",
}


def _full_plan_fixture():
    rows = []
    final_splits = {}
    reuse = {}
    index = 0
    for split, capabilities in _CAPABILITY_COUNTS.items():
        for capability, count in capabilities.items():
            for offset in range(count):
                index += 1
                plan_id = f"r2-plan-{index:04d}"
                rows.append(
                    {
                        "plan_id": plan_id,
                        "batch_id": f"r2-batch-{((index - 1) // 25) + 1:03d}",
                        "position": ((index - 1) % 25) + 1,
                        "canonical_intent": _INTENT_BY_CAPABILITY[capability],
                        "canonical_capability": capability,
                        "is_boundary": offset < _BOUNDARY_COUNTS[split][capability],
                        "asset_id": f"fixture-asset-{index:04d}",
                        "image_path": f"query_images/private-{index:04d}.jpg",
                    }
                )
                final_splits[plan_id] = split
                reuse[plan_id] = {
                    "reuse_variant": f"variant-{(offset % 3) + 1}",
                    "reuse_reason": "fixture",
                }
    assert len(rows) == 1500
    return rows, final_splits, reuse


def _sidecar():
    rows, final_splits, reuse = _full_plan_fixture()
    return build_realism_sidecar(
        rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        plan_sha256="a" * 64,
        asset_catalog_sha256="b" * 64,
        capability_assignments_sha256="c" * 64,
        seed=20260805,
    ), rows, final_splits, reuse


def test_allocator_is_canonical_and_hits_every_frozen_r2_quota():
    sidecar, rows, final_splits, reuse = _sidecar()
    repeated = build_realism_sidecar(
        rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        plan_sha256="a" * 64,
        asset_catalog_sha256="b" * 64,
        capability_assignments_sha256="c" * 64,
        seed=20260805,
    )

    assert len(sidecar.assignments) == 1500
    assert canonical_realism_assignments_bytes(sidecar.assignments) == canonical_realism_assignments_bytes(
        repeated.assignments
    )
    assert len(sidecar.recipes) == 12
    assert sidecar.manifest.row_count == 1500

    audit = audit_realism_assignments(sidecar.assignments, final_splits)
    assert audit["interaction"] == Counter(
        {
            "direct_request": 1050,
            "underspecified_clarification": 225,
            "constraint_correction": 75,
            "no_result_relaxation": 75,
            "goal_change_or_multi_query": 75,
        }
    )
    assert audit["expression"] == Counter({"S0": 225, "S1": 675, "S2": 450, "S3": 150})
    assert audit["user_style"] == Counter(
        {
            "terse_fragment": 375,
            "colloquial": 525,
            "neutral_complete": 375,
            "polite": 75,
            "code_mixed_or_numeric": 75,
            "typo_or_asr_like": 75,
        }
    )
    assert audit["constraint"] == Counter({"0": 450, "1": 600, "2": 375, "3_plus": 75})
    assert audit["ambiguity"] == Counter(
        {"resolved_near_boundary": 158, "clarification_required": 105}
    )
    assert audit["by_split"]["dev_mini"]["interaction"] == Counter(
        {
            "direct_request": 140,
            "underspecified_clarification": 30,
            "constraint_correction": 10,
            "no_result_relaxation": 10,
            "goal_change_or_multi_query": 10,
        }
    )
    assert all(
        assignment.trajectory_shape == "single_turn"
        for assignment in sidecar.assignments
        if assignment.interaction_pattern == "direct_request"
    )
    assert all(
        assignment.trajectory_shape == "three_turn"
        for assignment in sidecar.assignments
        if assignment.interaction_pattern != "direct_request"
    )
    assert all(
        assignment.interaction_pattern == "underspecified_clarification"
        for assignment in sidecar.assignments
        if assignment.ambiguity_subtype == "clarification_required"
    )


def test_realism_assignment_rejects_invalid_turn_and_ambiguity_combinations():
    common = {
        "plan_id": "fixture-plan",
        "expression_level": "S1",
        "user_style": "colloquial",
        "constraint_level": "1",
        "ambiguity_subtype": "none",
        "reuse_variant": "variant-1",
        "prompt_recipe_id": "fixture-recipe",
        "prompt_recipe_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="direct_request"):
        RealismAssignment(
            **common,
            trajectory_shape="three_turn",
            interaction_pattern="direct_request",
        )
    with pytest.raises(ValueError, match="clarification_required"):
        RealismAssignment(
            **{**common, "ambiguity_subtype": "clarification_required"},
            trajectory_shape="three_turn",
            interaction_pattern="constraint_correction",
        )


def test_realism_validator_rejects_quota_preserving_stratum_swaps():
    sidecar, rows, final_splits, reuse = _sidecar()
    rows_by_id = {row["plan_id"]: row for row in rows}
    boundary = next(
        assignment
        for assignment in sidecar.assignments
        if final_splits[assignment.plan_id] == "dev_mini"
        and rows_by_id[assignment.plan_id]["is_boundary"]
    )
    non_boundary = next(
        assignment
        for assignment in sidecar.assignments
        if final_splits[assignment.plan_id] == "dev_mini"
        and not rows_by_id[assignment.plan_id]["is_boundary"]
        and assignment.user_style != boundary.user_style
    )
    tampered_assignments = tuple(
        assignment.model_copy(update={"user_style": non_boundary.user_style})
        if assignment.plan_id == boundary.plan_id
        else assignment.model_copy(update={"user_style": boundary.user_style})
        if assignment.plan_id == non_boundary.plan_id
        else assignment
        for assignment in sidecar.assignments
    )
    tampered_manifest = sidecar.manifest.model_copy(
        update={
            "assignments_sha256": sha256_bytes(
                canonical_realism_assignments_bytes(tampered_assignments)
            )
        }
    )

    with pytest.raises(ValueError, match="exact seeded stratum allocation"):
        validate_realism_sidecar(
            replace(
                sidecar,
                assignments=tampered_assignments,
                manifest=tampered_manifest,
            ),
            plan_rows=rows,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=reuse,
        )


def test_core_realism_allocator_rejects_capability_matrix_drift():
    _, rows, final_splits, reuse = _sidecar()
    drifted_rows = [dict(row) for row in rows]
    target = next(
        row
        for row in drifted_rows
        if final_splits[row["plan_id"]] == "opt_pool"
        and row["canonical_capability"] == "utility.recipe_guidance"
        and not row["is_boundary"]
    )
    target["canonical_capability"] = "utility.document_reading"

    with pytest.raises(ValueError, match="capability x split matrix"):
        build_realism_sidecar(
            drifted_rows,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=reuse,
            plan_sha256="a" * 64,
            asset_catalog_sha256="b" * 64,
            capability_assignments_sha256="c" * 64,
            seed=20260805,
        )


def test_author_packet_is_opaque_and_checkpoint_is_monotonic_and_bound():
    sidecar, rows, final_splits, reuse = _sidecar()
    bindings = {
        row["asset_id"]: {
            "asset_sha256": f"{index:064x}",
            "byte_size": index,
            "canonical_path": f"E:/private-catalog/images/{index:04d}.jpg",
            "source_dataset": "private-catalog",
            "source_record_id": f"record-{index:04d}",
        }
        for index, row in enumerate(rows, start=1)
    }
    job = build_authoring_job(
        rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        realism_sidecar=sidecar,
        base_batch_id="r2-batch-001",
        asset_bindings=bindings,
    )

    packet = job.author_packet
    assert len(packet.items) == 25
    assert scan_author_packet_for_leaks(
        packet,
        forbidden_fragments=("private-catalog", "E:/private-catalog", "images/"),
    ) == ()
    packet_bytes = canonical_author_packet_bytes(packet)
    assert b"private-catalog" not in packet_bytes
    assert b"E:/private-catalog" not in packet_bytes
    assert b"image_path" not in packet_bytes
    assert b"source_dataset" not in packet_bytes
    assert b"final_split" not in packet_bytes
    assert b"capability_assignment" not in packet_bytes
    assert b"reuse_variant" not in packet_bytes
    assert b"counterfactual/abo" not in packet_bytes
    assert b"counterfactual/food" not in packet_bytes
    item = packet.items[0].model_dump(mode="json")
    assert set(item) == {
        "plan_id",
        "canonical_intent",
        "canonical_capability",
        "boundary",
        "realism",
        "recipe",
        "opaque_alias",
    }
    assert item["opaque_alias"].startswith("a0001.")
    assert job.work_order.aliases[0].canonical_path.startswith("E:/private-catalog")

    issued = job.checkpoint
    assert transition_authoring_checkpoint(issued, job.work_order, state="issued") == issued
    staged = transition_authoring_checkpoint(
        issued,
        job.work_order,
        state="staged",
        staged_batch_id="r2-batch-001-r1",
        staged_results_sha256="e" * 64,
    )
    assert transition_authoring_checkpoint(
        staged,
        job.work_order,
        state="staged",
        staged_batch_id="r2-batch-001-r1",
        staged_results_sha256="e" * 64,
    ) == staged
    accepted = transition_authoring_checkpoint(
        staged,
        job.work_order,
        state="accepted",
        staged_batch_id="r2-batch-001-r1",
        staged_results_sha256="e" * 64,
        accepted_batch_id="r2-batch-001-r1",
    )
    assert accepted.state == "accepted"
    with pytest.raises(ValueError, match="idempotent"):
        transition_authoring_checkpoint(
            staged,
            job.work_order,
            state="staged",
            staged_batch_id="r2-batch-001-r2",
            staged_results_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="backward"):
        transition_authoring_checkpoint(
            accepted,
            job.work_order,
            state="staged",
            staged_batch_id="r2-batch-001-r1",
            staged_results_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="skip"):
        transition_authoring_checkpoint(
            issued,
            job.work_order,
            state="accepted",
            accepted_batch_id="r2-batch-001-r1",
        )
    with pytest.raises(ValueError, match="does not bind"):
        transition_authoring_checkpoint(
            issued,
            job.work_order.model_copy(update={"generation_input_sha256": "f" * 64}),
            state="issued",
        )

    altered_recipes = list(job.work_order.prompt_recipes)
    altered_recipes[0] = altered_recipes[0].model_copy(
        update={"title": "altered_structure_label"}
    )
    altered_input_sha = compute_generation_input_sha256(
        base_batch_id=job.work_order.base_batch_id,
        plan_sha256=job.work_order.plan_sha256,
        asset_catalog_sha256=job.work_order.asset_catalog_sha256,
        capability_assignments_sha256=job.work_order.capability_assignments_sha256,
        realism_manifest=job.work_order.realism_manifest,
        recipe_manifest=build_prompt_recipe_manifest(altered_recipes),
        items=job.work_order.items,
        aliases=job.work_order.aliases,
    )
    assert altered_input_sha != job.work_order.generation_input_sha256

    changed_items = list(job.work_order.items)
    changed_items[0] = changed_items[0].model_copy(
        update={"boundary_strategy": "natural_ambiguity"}
    )
    boundary_changed_input_sha = compute_generation_input_sha256(
        base_batch_id=job.work_order.base_batch_id,
        plan_sha256=job.work_order.plan_sha256,
        asset_catalog_sha256=job.work_order.asset_catalog_sha256,
        capability_assignments_sha256=job.work_order.capability_assignments_sha256,
        realism_manifest=job.work_order.realism_manifest,
        recipe_manifest=job.work_order.recipe_manifest,
        items=changed_items,
        aliases=job.work_order.aliases,
    )
    assert boundary_changed_input_sha != job.work_order.generation_input_sha256
    with pytest.raises(ValueError, match="author packet no longer matches"):
        validate_authoring_job(
            replace(
                job,
                author_packet=job.author_packet.model_copy(
                    update={"generation_input_sha256": "f" * 64}
                ),
            )
        )
    assert scan_author_packet_for_leaks(
        {"source_path": "E:/private-catalog/images/0001.jpg"}
    )
