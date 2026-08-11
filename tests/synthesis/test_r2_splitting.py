from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.planning import PHASE3_TASK_SPEC_VERSION
from skillchain.synthesis.splitting import (
    BoundaryCapabilityInfeasibleError,
    CapabilityInfeasibleError,
    CORE_R2_SPLIT_SPEC,
    GROUP_FIELDS,
    R2CounterfactualComponent,
    R2SplitConstraintError,
    R2SplitConstraints,
    SplitSpec,
    ValidationGateInfeasibleError,
    ValidationGateSpec,
    build_atomic_groups,
    plan_validation_gates,
    stratified_split,
    validate_exact_split_targets,
    validate_validation_gate_audit,
    verify_r2_split_constraint_binding,
)
from skillchain.taxonomy import TAXONOMY_VERSION, capabilities_for_intent


CAPABILITY_TO_INTENT = {
    "product.exact_match": "exact_match",
    "product.multi_search": "multi_product",
    "product.style_recommendation": "divergent_rec",
    "knowledge.visual_encyclopedia": "encyclopedia",
    "utility.document_reading": "utility",
    "utility.recipe_guidance": "utility",
}


def _query(
    index: int,
    *,
    capability: str,
    split: str,
    generator_batch_id: str,
    is_boundary: bool = False,
    boundary_strategy: str | None = None,
    boundary_group_id: str | None = None,
    shared_component_id: str | None = None,
) -> Query:
    intent = CAPABILITY_TO_INTENT[capability]
    requires_card = next(
        item.requires_card
        for item in capabilities_for_intent(intent)
        if item.capability_id == capability
    )
    query_id = f"r2-{index:04d}"
    text = f"mechanical r2 query {index:04d}"
    asset_id = shared_component_id or f"asset-{index:04d}"
    image_path = (
        f"query_images/{shared_component_id}.jpg"
        if shared_component_id is not None
        else f"query_images/r2-{index:04d}.jpg"
    )
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=PHASE3_TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=asset_id,
        image_path=image_path,
        leakage_group_id=shared_component_id or f"leakage-{index:04d}",
        boundary_group_id=boundary_group_id,
        template_family=f"template-{index:04d}",
        generator_batch_id=generator_batch_id,
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=is_boundary,
        boundary_strategy=boundary_strategy,
        requires_card=requires_card,
        split=split,
        label_status="cross_agreed",
        label_provenance=[
            LabelDecision(
                decision_type="cross_review",
                annotator_kind="llm_pair",
                annotator_id="mechanical-r2-fixture",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _r2_fixture() -> tuple[list[Query], set[str], R2SplitConstraints]:
    assert CORE_R2_SPLIT_SPEC.capability_targets is not None
    assert CORE_R2_SPLIT_SPEC.boundary_capability_targets is not None
    candidates: list[Query] = []
    mapping: dict[str, str] = {}
    declaration_ids: dict[str, list[str]] = {}
    declaration_splits: dict[str, str] = {}
    abo_component_ids: list[str] = []
    food_component_ids: list[str] = []
    counterfactual_counts = {
        "dev_mini": (0, 0),
        "opt_pool": (12, 6),
        "val": (3, 2),
        "test_frozen": (5, 2),
    }
    index = 0
    abo_sequence = 0
    food_sequence = 0
    for split in ("dev_mini", "opt_pool", "val", "test_frozen"):
        cap_targets = dict(CORE_R2_SPLIT_SPEC.capability_targets[split])
        boundary_targets = dict(
            CORE_R2_SPLIT_SPEC.boundary_capability_targets[split]
        )
        component_chunks: list[
            list[tuple[str, bool, str | None, str | None]]
        ] = []
        abo_count, food_count = counterfactual_counts[split]
        for _ in range(abo_count):
            abo_sequence += 1
            component_id = f"abo.{abo_sequence:03d}"
            abo_component_ids.append(component_id)
            declaration_ids[component_id] = []
            declaration_splits[component_id] = split
            chunk = [
                (
                    "product.exact_match",
                    True,
                    "cross_intent_triplet",
                    component_id,
                ),
                (
                    "product.style_recommendation",
                    True,
                    "cross_intent_triplet",
                    component_id,
                ),
                (
                    "knowledge.visual_encyclopedia",
                    True,
                    "cross_intent_triplet",
                    component_id,
                ),
            ]
            component_chunks.append(chunk)
            for capability, _, _, _ in chunk:
                cap_targets[capability] -= 1
                boundary_targets[capability] -= 1
        for _ in range(food_count):
            food_sequence += 1
            component_id = f"food.{food_sequence:03d}"
            food_component_ids.append(component_id)
            declaration_ids[component_id] = []
            declaration_splits[component_id] = split
            chunk = [
                (
                    "knowledge.visual_encyclopedia",
                    True,
                    "cross_intent_pair",
                    component_id,
                ),
                (
                    "utility.recipe_guidance",
                    True,
                    "cross_intent_pair",
                    component_id,
                ),
            ]
            component_chunks.append(chunk)
            for capability, _, _, _ in chunk:
                cap_targets[capability] -= 1
                boundary_targets[capability] -= 1
        normal_rows: list[tuple[str, bool, str | None, str | None]] = []
        for capability, count in cap_targets.items():
            boundary_count = boundary_targets[capability]
            normal_rows.extend(
                (
                    capability,
                    offset < boundary_count,
                    "natural_ambiguity" if offset < boundary_count else None,
                    None,
                )
                for offset in range(count)
            )
        batches: list[list[tuple[str, bool, str | None, str | None]]] = []
        current: list[tuple[str, bool, str | None, str | None]] = []
        for chunk in component_chunks:
            if len(current) + len(chunk) > 25:
                batches.append(current)
                current = []
            current.extend(chunk)
        if current:
            batches.append(current)
        normal_offset = 0
        for batch in batches:
            capacity = 25 - len(batch)
            batch.extend(normal_rows[normal_offset : normal_offset + capacity])
            normal_offset += capacity
        while normal_offset < len(normal_rows):
            batches.append(normal_rows[normal_offset : normal_offset + 25])
            normal_offset += 25
        assert len(batches) == CORE_R2_SPLIT_SPEC.sizes[split] // 25
        assert all(len(batch) == 25 for batch in batches)
        for batch_index, batch in enumerate(batches):
            for capability, is_boundary, strategy, component_id in batch:
                index += 1
                query = _query(
                    index,
                    capability=capability,
                    split="dev_mini" if split == "dev_mini" else "opt_pool",
                    generator_batch_id=f"batch-{split}-{batch_index:02d}",
                    is_boundary=is_boundary,
                    boundary_strategy=strategy,
                    boundary_group_id=component_id,
                    shared_component_id=component_id,
                )
                candidates.append(query)
                mapping[query.query_id] = split
                if component_id is not None:
                    declaration_ids[component_id].append(query.query_id)
    locked = {
        query.query_id for query in candidates if mapping[query.query_id] == "dev_mini"
    }
    constraints = R2SplitConstraints(
        plan_sha256="a" * 64,
        asset_catalog_sha256="b" * 64,
        capability_assignments_sha256="c" * 64,
        plan_id_to_split=mapping,
        abo_components=tuple(
            R2CounterfactualComponent(
                component_id=component_id,
                source="abo",
                split=declaration_splits[component_id],
                plan_ids=tuple(declaration_ids[component_id]),
            )
            for component_id in abo_component_ids
        ),
        food_components=tuple(
            R2CounterfactualComponent(
                component_id=component_id,
                source="food",
                split=declaration_splits[component_id],
                plan_ids=tuple(declaration_ids[component_id]),
            )
            for component_id in food_component_ids
        ),
    )
    return candidates, locked, constraints


def _balanced_val_fixture() -> tuple[list[Query], dict[str, str]]:
    assert CORE_R2_SPLIT_SPEC.capability_targets is not None
    targets = dict(CORE_R2_SPLIT_SPEC.capability_targets["val"])
    rows_by_batch: list[list[str]] = [[] for _ in range(8)]
    for batch in rows_by_batch:
        for capability in targets:
            batch.append(capability)
            targets[capability] -= 1
    batch_cursor = 0
    for capability, remaining in targets.items():
        for _ in range(remaining):
            while len(rows_by_batch[batch_cursor]) == 25:
                batch_cursor = (batch_cursor + 1) % len(rows_by_batch)
            rows_by_batch[batch_cursor].append(capability)
            batch_cursor = (batch_cursor + 1) % len(rows_by_batch)
    assert all(len(batch) == 25 for batch in rows_by_batch)

    queries: list[Query] = []
    interactions: dict[str, str] = {}
    index = 20_000
    for batch_index, capabilities in enumerate(rows_by_batch):
        for offset, capability in enumerate(capabilities):
            index += 1
            query = _query(
                index,
                capability=capability,
                split="val",
                generator_batch_id=f"val-batch-{batch_index:02d}",
                is_boundary=offset == 0,
                boundary_strategy="natural_ambiguity" if offset == 0 else None,
            )
            queries.append(query)
            interactions[query.query_id] = (
                "direct_request" if offset % 2 == 0 else "constraint_correction"
            )
    return queries, interactions


def test_core_r2_spec_freezes_revised_document_recipe_rows():
    assert CORE_R2_SPLIT_SPEC.profile == "core"
    assert CORE_R2_SPLIT_SPEC.capability_targets is not None
    assert CORE_R2_SPLIT_SPEC.capability_targets["opt_pool"][
        "utility.document_reading"
    ] == 32
    assert CORE_R2_SPLIT_SPEC.capability_targets["opt_pool"][
        "utility.recipe_guidance"
    ] == 116
    assert CORE_R2_SPLIT_SPEC.capability_targets["test_frozen"][
        "utility.document_reading"
    ] == 18
    assert CORE_R2_SPLIT_SPEC.capability_targets["test_frozen"][
        "utility.recipe_guidance"
    ] == 37

    with pytest.raises(ValidationError, match="both set"):
        SplitSpec(
            profile="invalid-half-matrix",
            sizes={
                "dev_mini": 1,
                "opt_pool": 1,
                "val": 1,
                "test_frozen": 1,
            },
            min_test_per_intent=0,
            capability_targets={
                split: {"product.exact_match": 1}
                for split in ("dev_mini", "opt_pool", "val", "test_frozen")
            },
        )


def test_core_r2_exact_split_uses_fixed_complete_mapping_and_postcheck():
    candidates, locked, constraints = _r2_fixture()

    assigned = stratified_split(
        list(reversed(candidates)),
        locked_dev_query_ids=locked,
        seed=20260804,
        split_spec=CORE_R2_SPLIT_SPEC,
        r2_constraints=constraints,
    )

    assert {query.query_id: query.split for query in assigned} == dict(
        constraints.plan_id_to_split
    )
    validate_exact_split_targets(assigned, CORE_R2_SPLIT_SPEC)
    assert all(
        len({query.split for query in group.queries}) == 1
        for group in build_atomic_groups(assigned)
    )
    first_group = next(
        group for group in build_atomic_groups(assigned) if group.size == 25
    )
    assert sum(first_group.capability_counts.values()) == 25
    assert sum(first_group.boundary_capability_counts.values()) <= 25


def test_core_r2_constraints_reject_incomplete_or_cross_group_mapping():
    candidates, locked, constraints = _r2_fixture()
    incomplete = dict(constraints.plan_id_to_split)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(R2SplitConstraintError, match="complete plan mapping"):
        stratified_split(
            candidates,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints.model_copy(
                update={"plan_id_to_split": incomplete}
            ),
        )

    mixed = dict(constraints.plan_id_to_split)
    opt_id = next(plan_id for plan_id, split in mixed.items() if split == "opt_pool")
    val_id = next(plan_id for plan_id, split in mixed.items() if split == "val")
    mixed[opt_id], mixed[val_id] = mixed[val_id], mixed[opt_id]
    with pytest.raises(R2SplitConstraintError, match="atomic GROUP_FIELDS"):
        stratified_split(
            candidates,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints.model_copy(update={"plan_id_to_split": mixed}),
        )


def test_r2_constraint_declarations_and_parent_hashes_fail_closed():
    candidates, locked, constraints = _r2_fixture()
    verify_r2_split_constraint_binding(
        constraints,
        expected_plan_sha256="a" * 64,
        expected_asset_catalog_sha256="b" * 64,
        expected_capability_assignments_sha256="c" * 64,
    )
    with pytest.raises(R2SplitConstraintError, match="plan_sha256"):
        verify_r2_split_constraint_binding(
            constraints,
            expected_plan_sha256="d" + "a" * 63,
            expected_asset_catalog_sha256="b" * 64,
            expected_capability_assignments_sha256="c" * 64,
        )
    with pytest.raises(ValidationError, match="exactly 20 ABO and 10 Food"):
        R2SplitConstraints(
            plan_sha256="a" * 64,
            asset_catalog_sha256="b" * 64,
            capability_assignments_sha256="c" * 64,
            plan_id_to_split=constraints.plan_id_to_split,
        )

    first_abo = constraints.abo_components[0]
    distinct_identifiers = [
        query.model_copy(
            update={
                "asset_id": "asset-distinct-from-leakage",
                "image_path": "query_images/asset-distinct-from-leakage.jpg",
                "boundary_group_id": "boundary-distinct-from-leakage",
            }
        )
        if query.query_id in first_abo.plan_ids
        else query
        for query in candidates
    ]
    stratified_split(
        distinct_identifiers,
        locked_dev_query_ids=locked,
        seed=20260804,
        split_spec=CORE_R2_SPLIT_SPEC,
        r2_constraints=constraints,
    )

    drift_id = first_abo.plan_ids[0]
    drifted_candidates = [
        query.model_copy(update={"asset_id": "different-asset"})
        if query.query_id == drift_id
        else query
        for query in candidates
    ]
    with pytest.raises(R2SplitConstraintError, match="shared asset/path"):
        stratified_split(
            drifted_candidates,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )

    boundary_drift = [
        query.model_copy(update={"boundary_group_id": "different-boundary"})
        if query.query_id == drift_id
        else query
        for query in candidates
    ]
    with pytest.raises(R2SplitConstraintError, match="boundary component"):
        stratified_split(
            boundary_drift,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )

    leakage_drift = [
        query.model_copy(update={"leakage_group_id": "different-leakage"})
        if query.query_id == drift_id
        else query
        for query in candidates
    ]
    with pytest.raises(R2SplitConstraintError, match="leakage component"):
        stratified_split(
            leakage_drift,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )


def test_core_r2_constraints_require_sixty_complete_generator_batches():
    candidates, locked, constraints = _r2_fixture()
    deatomized = [
        query.model_copy(update={"generator_batch_id": f"single-{query.query_id}"})
        for query in candidates
    ]

    with pytest.raises(R2SplitConstraintError, match="60 generator batches"):
        stratified_split(
            deatomized,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )


def test_core_r2_inventory_reports_capability_and_boundary_failures_separately():
    candidates, locked, constraints = _r2_fixture()
    boundary_target = next(query for query in candidates if not query.is_boundary)
    boundary_drift = boundary_target.model_copy(
        update={"is_boundary": True, "boundary_strategy": "natural_ambiguity"}
    )
    drifted = [
        boundary_drift if query.query_id == boundary_target.query_id else query
        for query in candidates
    ]
    with pytest.raises(BoundaryCapabilityInfeasibleError, match="global boundary"):
        stratified_split(
            drifted,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )

    capability_target = next(
        query
        for query in candidates
        if query.canonical_capability == "utility.document_reading"
    )
    recipe_capability = "utility.recipe_guidance"
    recipe_requires_card = next(
        item.requires_card
        for item in capabilities_for_intent("utility")
        if item.capability_id == recipe_capability
    )
    latest = capability_target.label_provenance[-1].model_copy(
        update={
            "canonical_capability": recipe_capability,
            "acceptable_capabilities": [recipe_capability],
        }
    )
    capability_drift = capability_target.model_copy(
        update={
            "canonical_capability": recipe_capability,
            "acceptable_capabilities": [recipe_capability],
            "requires_card": recipe_requires_card,
            "label_provenance": [latest],
        }
    )
    drifted = [
        capability_drift if query.query_id == capability_target.query_id else query
        for query in candidates
    ]
    with pytest.raises(CapabilityInfeasibleError, match="global capability"):
        stratified_split(
            drifted,
            locked_dev_query_ids=locked,
            seed=20260804,
            split_spec=CORE_R2_SPLIT_SPEC,
            r2_constraints=constraints,
        )


def test_validation_gates_are_75_75_50_atomic_and_cover_hard_sidecars():
    queries, interactions = _balanced_val_fixture()
    spec = ValidationGateSpec(
        boundary_minimums={
            "route_gate": 1,
            "body_gate": 1,
            "shadow_val": 1,
        },
        interaction_minimums={
            gate: {"direct_request": 1, "constraint_correction": 1}
            for gate in ("route_gate", "body_gate", "shadow_val")
        },
    )

    audit = plan_validation_gates(
        queries,
        seed=20260804,
        spec=spec,
        interaction_by_query_id=interactions,
    )
    reordered_audit = plan_validation_gates(
        list(reversed(queries)),
        seed=20260804,
        spec=spec,
        interaction_by_query_id=interactions,
    )

    assert audit.gate_sizes == {
        "route_gate": 75,
        "body_gate": 75,
        "shadow_val": 50,
    }
    assert reordered_audit.query_id_to_gate == audit.query_id_to_gate
    for gate, minimum in spec.capability_minimums.items():
        assert min(audit.capability_counts[gate].values()) >= minimum
        assert audit.boundary_counts[gate] >= 1
        assert set(audit.interaction_counts[gate]) == {
            "direct_request",
            "constraint_correction",
        }
    for group in build_atomic_groups(queries):
        assert len({audit.query_id_to_gate[item.query_id] for item in group.queries}) == 1
    validate_validation_gate_audit(
        queries,
        audit,
        spec=spec,
        interaction_by_query_id=interactions,
    )


def test_validation_gate_audit_tampering_and_incomplete_sidecar_fail_closed():
    queries, interactions = _balanced_val_fixture()
    spec = ValidationGateSpec(
        interaction_minimums={
            gate: {"direct_request": 1}
            for gate in ("route_gate", "body_gate", "shadow_val")
        }
    )
    with pytest.raises(ValidationGateInfeasibleError, match="sidecar"):
        plan_validation_gates(queries, seed=1, spec=spec)

    audit = plan_validation_gates(
        queries,
        seed=1,
        spec=spec,
        interaction_by_query_id=interactions,
    )
    changed = dict(audit.query_id_to_gate)
    target = queries[0].query_id
    changed[target] = (
        "body_gate" if changed[target] != "body_gate" else "route_gate"
    )
    with pytest.raises(ValidationGateInfeasibleError, match="GROUP_FIELDS atom"):
        validate_validation_gate_audit(
            queries,
            audit.model_copy(update={"query_id_to_gate": changed}),
            spec=spec,
            interaction_by_query_id=interactions,
        )


def test_legacy_80_80_40_gate_contract_is_explicitly_infeasible():
    queries, _ = _balanced_val_fixture()
    old_spec = ValidationGateSpec(
        sizes={"route_gate": 80, "body_gate": 80, "shadow_val": 40}
    )

    with pytest.raises(ValidationGateInfeasibleError, match="80/80/40"):
        plan_validation_gates(queries, seed=20260804, spec=old_spec)


def test_validation_gate_contract_retains_every_atomic_group_field():
    queries, interactions = _balanced_val_fixture()
    audit = plan_validation_gates(
        queries,
        seed=20260804,
        interaction_by_query_id=interactions,
    )

    assert audit.group_fields == GROUP_FIELDS
    assert audit.grouping_policy_version == "query-connected-components-v1"
    assert Counter(audit.query_id_to_gate.values()) == Counter(audit.gate_sizes)
