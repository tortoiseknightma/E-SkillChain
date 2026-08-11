"""Focused pure-metadata tests for the r2 200-row owner audit sampler."""

from __future__ import annotations

from dataclasses import replace

import pytest

from skillchain.synthesis.portfolio_core_audit import (
    AuditCoverageError,
    build_core_r2_audit_sample,
    canonical_audit_selection_index_bytes,
    validate_core_r2_audit_sample,
)
from skillchain.synthesis.portfolio_core_authoring import build_realism_sidecar
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


def _fixture_inputs():
    rows = []
    final_splits = {}
    reuse = {}
    sources = {}
    index = 0
    for split, capabilities in _CAPABILITY_COUNTS.items():
        for capability, count in capabilities.items():
            for offset in range(count):
                index += 1
                plan_id = f"r2-plan-{index:04d}"
                is_boundary = offset < _BOUNDARY_COUNTS[split][capability]
                asset_id = f"fixture-asset-{index:04d}"
                batch_id = f"r2-batch-{((index - 1) // 25) + 1:03d}"
                rows.append(
                    {
                        "plan_id": plan_id,
                        "batch_id": batch_id,
                        "generator_batch_id": batch_id,
                        "position": ((index - 1) % 25) + 1,
                        "canonical_intent": _INTENT_BY_CAPABILITY[capability],
                        "canonical_capability": capability,
                        "is_boundary": is_boundary,
                        "boundary_strategy": (
                            "cross_intent_triplet"
                            if is_boundary and index % 3 == 0
                            else "natural_ambiguity"
                            if is_boundary
                            else None
                        ),
                        "asset_id": asset_id,
                        "leakage_group_id": f"component-{(index - 1) // 2:04d}",
                    }
                )
                final_splits[plan_id] = split
                reuse[plan_id] = {
                    "reuse_variant": f"variant-{(offset % 3) + 1}",
                    "reuse_reason": "fixture",
                }
                sources[asset_id] = f"source-{index % 8}"
    assert len(rows) == 1500
    realism = build_realism_sidecar(
        rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        plan_sha256="a" * 64,
        asset_catalog_sha256="b" * 64,
        capability_assignments_sha256="c" * 64,
        seed=20260805,
    )
    return rows, final_splits, realism, sources


def test_core_r2_sampler_first_green_covers_all_mandatory_metadata_cells():
    rows, final_splits, realism, sources = _fixture_inputs()
    sample = build_core_r2_audit_sample(
        rows,
        final_split_by_plan_id=final_splits,
        realism_sidecar=realism,
        source_by_asset_id=sources,
        seed=20260805,
    )
    repeated = build_core_r2_audit_sample(
        rows,
        final_split_by_plan_id=final_splits,
        realism_sidecar=realism,
        source_by_asset_id=sources,
        seed=20260805,
    )

    assert len(sample.entries) == 200
    assert len({entry.plan_id for entry in sample.entries}) == 200
    assert len({entry.generator_batch_id for entry in sample.entries}) == 60
    assert len(sample.audit.recipe_counts) == 12
    assert set(sample.audit.source_counts or {}) == set(sources.values())
    assert sample.audit.rare_counts["document_boundary"] >= 4
    assert sample.audit.rare_counts["counterfactual"] >= 12
    assert sample.audit.rare_counts["s3"] >= 15
    assert sample.audit.rare_counts["no_result_relaxation"] >= 12
    assert sample.audit.rare_counts["goal_change_or_multi_query"] >= 12
    assert any(len(entry.selection_reason) > 1 for entry in sample.entries)
    assert canonical_audit_selection_index_bytes(
        sample.entries
    ) == canonical_audit_selection_index_bytes(repeated.entries)


def test_audit_index_and_manifest_tampering_fail_closed_and_overfull_mandatory_fails():
    rows, final_splits, realism, sources = _fixture_inputs()
    sample = build_core_r2_audit_sample(
        rows,
        final_split_by_plan_id=final_splits,
        realism_sidecar=realism,
        source_by_asset_id=sources,
        seed=20260805,
    )
    tampered_entries = list(sample.entries)
    tampered_entries[0] = tampered_entries[0].model_copy(
        update={"canonical_capability": "tampered.capability"}
    )
    with pytest.raises(ValueError, match="selection index digest mismatch"):
        validate_core_r2_audit_sample(
            replace(sample, entries=tuple(tampered_entries)),
            plan_rows=rows,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=sources,
        )
    with pytest.raises(ValueError, match="selection index digest mismatch"):
        validate_core_r2_audit_sample(
            replace(
                sample,
                manifest=sample.manifest.model_copy(
                    update={"selection_index_sha256": "f" * 64}
                ),
            ),
            plan_rows=rows,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=sources,
        )
    with pytest.raises(ValueError, match="plan_sha256 mismatch"):
        validate_core_r2_audit_sample(
            replace(
                sample,
                manifest=sample.manifest.model_copy(
                    update={"plan_sha256": "e" * 64}
                ),
            ),
            plan_rows=rows,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=sources,
        )
    multi_reason_index = next(
        index
        for index, entry in enumerate(sample.entries)
        if len(entry.selection_reason) > 1
    )
    rehashed_entries = list(sample.entries)
    original = rehashed_entries[multi_reason_index]
    rehashed_entries[multi_reason_index] = original.model_copy(
        update={"selection_reason": original.selection_reason[1:]}
    )
    rehashed_index = tuple(rehashed_entries)
    with pytest.raises(ValueError, match="deterministic sampler"):
        validate_core_r2_audit_sample(
            replace(
                sample,
                entries=rehashed_index,
                manifest=sample.manifest.model_copy(
                    update={
                        "selection_index_sha256": sha256_bytes(
                            canonical_audit_selection_index_bytes(rehashed_index)
                        )
                    }
                ),
            ),
            plan_rows=rows,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=sources,
        )
    altered_splits = dict(final_splits)
    altered_splits[rows[0]["plan_id"]] = "opt_pool"
    with pytest.raises(ValueError, match="final split sidecar digest mismatch"):
        validate_core_r2_audit_sample(
            sample,
            plan_rows=rows,
            final_split_by_plan_id=altered_splits,
            realism_sidecar=realism,
            source_by_asset_id=sources,
        )

    unique_sources = {
        row["asset_id"]: f"unique-source-{index:04d}"
        for index, row in enumerate(rows, start=1)
    }
    with pytest.raises(AuditCoverageError, match="more than 200"):
        build_core_r2_audit_sample(
            rows,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=unique_sources,
            seed=20260805,
        )
