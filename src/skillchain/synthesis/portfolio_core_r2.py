"""Pure in-memory bridge from the audited r2 planner to authoring inputs.

The r2 planner intentionally keeps final-split and reuse decisions outside the
serialized :class:`~skillchain.synthesis.models.CorpusPlan`.  This module is
the narrow, fail-closed bridge that turns that audited layout into the two
typed packets consumed by splitting and authoring.  It never reads data,
writes files, or produces corpus text.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from skillchain.schemas import Split
from skillchain.synthesis.models import PlannedQuery
from skillchain.synthesis.planning import (
    MVP_CAPABILITY_ORDER,
    R2CoreInMemoryPlan,
    R2CorePlanAudit,
    audit_r2_core_in_memory_plan,
)
from skillchain.synthesis.portfolio_core_authoring import (
    RealismSidecar,
    ReuseAssignment,
    build_prompt_recipe_manifest,
    build_realism_sidecar,
    core_r2_realism_quota_spec,
    default_prompt_recipes,
    validate_realism_sidecar,
)
from skillchain.synthesis.splitting import (
    R2CounterfactualComponent,
    R2SplitConstraints,
    verify_r2_split_constraint_binding,
)
from skillchain.synthesis.store import canonical_json_bytes


class R2CoreBridgeError(ValueError):
    """An audited r2 layout cannot safely be projected to a downstream packet."""


@dataclass(frozen=True)
class R2CoreBridge:
    """Complete deterministic projection for the next r2 synthesis stages.

    The batch materializer makes ``query_id == plan_id``.  The interaction
    mapping is therefore already keyed exactly as ``plan_validation_gates``
    expects, while retaining the immutable planner identity.
    """

    split_constraints: R2SplitConstraints
    realism_sidecar: RealismSidecar
    val_interaction_by_query_id: Mapping[str, str]


_SPECIAL_COMPONENTS: dict[Literal["abo", "food"], tuple[str, tuple[str, ...], str]] = {
    "abo": (
        "counterfactual_abo",
        (
            "product.exact_match",
            "product.style_recommendation",
            "knowledge.visual_encyclopedia",
        ),
        "cross_intent_triplet",
    ),
    "food": (
        "counterfactual_food",
        (
            "knowledge.visual_encyclopedia",
            "utility.recipe_guidance",
        ),
        "cross_intent_pair",
    ),
}


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R2CoreBridgeError(f"{label} must be a 64-character lowercase SHA-256")


def _canonical_plan_sha256(result: R2CoreInMemoryPlan) -> str:
    return hashlib.sha256(canonical_json_bytes(result.plan)).hexdigest()


def _validated_result(
    result: R2CoreInMemoryPlan,
    *,
    trusted_plan_sha256: str,
) -> R2CorePlanAudit:
    """Re-audit mutable mappings and bind them to the caller's trusted digest."""

    _require_sha256(trusted_plan_sha256, "trusted_plan_sha256")
    actual_plan_sha256 = _canonical_plan_sha256(result)
    if actual_plan_sha256 != trusted_plan_sha256:
        raise R2CoreBridgeError(
            "trusted plan SHA-256 does not match the in-memory CorpusPlan"
        )
    try:
        audit = audit_r2_core_in_memory_plan(result)
    except (TypeError, ValueError, AssertionError) as exc:
        raise R2CoreBridgeError("r2 in-memory plan audit failed") from exc
    if audit != result.audit:
        raise R2CoreBridgeError("stored r2 plan audit no longer matches its layout")
    _require_sha256(result.plan.asset_catalog_sha256, "asset_catalog_sha256")
    _require_sha256(
        result.plan.capability_assignments_sha256,
        "capability_assignments_sha256",
    )
    return audit


def _plan_rows_by_component(
    result: R2CoreInMemoryPlan,
) -> dict[str, list[PlannedQuery]]:
    grouped: dict[str, list[PlannedQuery]] = defaultdict(list)
    for row in result.plan.queries:
        component_id = row.leakage_group_id
        if not component_id or component_id != component_id.strip():
            raise R2CoreBridgeError("r2 planned row has an invalid leakage_group_id")
        # The key is deliberately the leakage component itself: do not derive
        # it from asset_id or boundary_group_id, which are separate identities.
        grouped[component_id].append(row)
    return dict(grouped)


def _counterfactual_components(
    result: R2CoreInMemoryPlan,
    *,
    source: Literal["abo", "food"],
) -> tuple[R2CounterfactualComponent, ...]:
    reason, expected_capabilities, expected_boundary_strategy = _SPECIAL_COMPONENTS[
        source
    ]
    capability_rank = {
        capability: index for index, capability in enumerate(MVP_CAPABILITY_ORDER)
    }
    components: list[R2CounterfactualComponent] = []
    for component_id, rows in sorted(_plan_rows_by_component(result).items()):
        reasons = {result.reuse_reason_by_plan_id[row.plan_id] for row in rows}
        special_reasons = {
            candidate
            for candidate in reasons
            if candidate
            in {
                _SPECIAL_COMPONENTS["abo"][0],
                _SPECIAL_COMPONENTS["food"][0],
            }
        }
        if not special_reasons:
            continue
        if reasons != {reason}:
            if len(reasons) == 1:
                # This is a well-formed component for the other source; its
                # declaration is built by the corresponding call below.
                continue
            raise R2CoreBridgeError(
                "counterfactual component must have one consistent reuse_reason"
            )
        if len(rows) != len(expected_capabilities):
            raise R2CoreBridgeError(
                f"{source} counterfactual component has the wrong cardinality"
            )
        if {row.canonical_capability for row in rows} != set(expected_capabilities):
            raise R2CoreBridgeError(
                f"{source} counterfactual component has the wrong capabilities"
            )
        if not all(row.is_boundary for row in rows):
            raise R2CoreBridgeError(
                "counterfactual component must contain boundary rows"
            )
        if {row.boundary_strategy for row in rows} != {expected_boundary_strategy}:
            raise R2CoreBridgeError(
                "counterfactual component has the wrong boundary strategy"
            )
        boundary_groups = {row.boundary_group_id for row in rows}
        if len(boundary_groups) != 1 or None in boundary_groups:
            raise R2CoreBridgeError(
                "counterfactual component must share one boundary_group_id"
            )
        assets = {(row.asset_id, row.image_path) for row in rows}
        if len(assets) != 1:
            raise R2CoreBridgeError(
                "counterfactual component must share one asset/path binding"
            )
        splits = {result.final_split_by_plan_id[row.plan_id] for row in rows}
        if len(splits) != 1 or "dev_mini" in splits:
            raise R2CoreBridgeError(
                "counterfactual component must occupy one non-dev final split"
            )
        ordered_rows = sorted(
            rows,
            key=lambda row: (capability_rank[row.canonical_capability], row.plan_id),
        )
        components.append(
            R2CounterfactualComponent(
                component_id=component_id,
                source=source,
                split=next(iter(splits)),
                plan_ids=tuple(row.plan_id for row in ordered_rows),
            )
        )
    return tuple(components)


def build_r2_split_constraints(
    result: R2CoreInMemoryPlan,
    *,
    trusted_plan_sha256: str,
) -> R2SplitConstraints:
    """Build a complete r2 split packet from an independently trusted plan."""

    _validated_result(result, trusted_plan_sha256=trusted_plan_sha256)
    plan_id_to_split = {
        row.plan_id: result.final_split_by_plan_id[row.plan_id]
        for row in result.plan.queries
    }
    try:
        constraints = R2SplitConstraints(
            plan_sha256=trusted_plan_sha256,
            asset_catalog_sha256=result.plan.asset_catalog_sha256,
            capability_assignments_sha256=result.plan.capability_assignments_sha256,
            plan_id_to_split=plan_id_to_split,
            abo_components=_counterfactual_components(result, source="abo"),
            food_components=_counterfactual_components(result, source="food"),
        )
    except (TypeError, ValueError) as exc:
        raise R2CoreBridgeError("unable to build r2 split constraints") from exc
    try:
        verify_r2_split_constraint_binding(
            constraints,
            expected_plan_sha256=trusted_plan_sha256,
            expected_asset_catalog_sha256=result.plan.asset_catalog_sha256,
            expected_capability_assignments_sha256=(
                result.plan.capability_assignments_sha256
            ),
        )
    except (TypeError, ValueError) as exc:
        raise R2CoreBridgeError(
            "r2 split constraint binding validation failed"
        ) from exc
    return constraints


def _reuse_sidecar(result: R2CoreInMemoryPlan) -> dict[str, ReuseAssignment]:
    return {
        row.plan_id: ReuseAssignment(
            plan_id=row.plan_id,
            reuse_variant=result.reuse_variant_by_plan_id[row.plan_id],
            reuse_reason=result.reuse_reason_by_plan_id[row.plan_id],
        )
        for row in result.plan.queries
    }


def _final_split_sidecar(result: R2CoreInMemoryPlan) -> dict[str, Split]:
    return {
        row.plan_id: result.final_split_by_plan_id[row.plan_id]
        for row in result.plan.queries
    }


def _validate_realism_binding(
    result: R2CoreInMemoryPlan,
    sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
) -> None:
    _validated_result(result, trusted_plan_sha256=trusted_plan_sha256)
    expected_manifest = {
        "plan_sha256": trusted_plan_sha256,
        "asset_catalog_sha256": result.plan.asset_catalog_sha256,
        "capability_assignments_sha256": result.plan.capability_assignments_sha256,
    }
    for field_name, expected in expected_manifest.items():
        if getattr(sidecar.manifest, field_name) != expected:
            raise R2CoreBridgeError(f"realism sidecar binding mismatch: {field_name}")
    canonical_recipes = tuple(
        sorted(default_prompt_recipes(), key=lambda item: item.prompt_recipe_id)
    )
    if (
        sidecar.recipes != canonical_recipes
        or sidecar.recipe_manifest
        != build_prompt_recipe_manifest(canonical_recipes)
    ):
        raise R2CoreBridgeError(
            "core r2 requires the canonical prompt recipe library"
        )
    try:
        validate_realism_sidecar(
            sidecar,
            plan_rows=result.plan.queries,
            final_split_by_plan_id=_final_split_sidecar(result),
            reuse_by_plan_id=_reuse_sidecar(result),
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise R2CoreBridgeError("realism sidecar validation failed") from exc


def build_r2_realism_sidecar(
    result: R2CoreInMemoryPlan,
    *,
    trusted_plan_sha256: str,
    seed: int,
) -> RealismSidecar:
    """Build the full 1,500-row text-free realism sidecar for an r2 layout."""

    _validated_result(result, trusted_plan_sha256=trusted_plan_sha256)
    try:
        sidecar = build_realism_sidecar(
            result.plan.queries,
            final_split_by_plan_id=_final_split_sidecar(result),
            reuse_by_plan_id=_reuse_sidecar(result),
            plan_sha256=trusted_plan_sha256,
            asset_catalog_sha256=result.plan.asset_catalog_sha256,
            capability_assignments_sha256=(result.plan.capability_assignments_sha256),
            seed=seed,
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise R2CoreBridgeError("unable to build the r2 realism sidecar") from exc
    _validate_realism_binding(
        result,
        sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    return sidecar


def extract_r2_val_interaction_by_query_id(
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
) -> dict[str, str]:
    """Return the exact 200-row validation interaction mapping.

    Query IDs are the deterministic plan IDs in the batch materializer.  This
    makes omission or a stale writer-to-plan mapping fail before validation
    gates see a partial interaction assignment.
    """

    _validate_realism_binding(
        result,
        realism_sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    assignment_by_plan_id = {
        assignment.plan_id: assignment for assignment in realism_sidecar.assignments
    }
    val_mapping = {
        row.plan_id: assignment_by_plan_id[row.plan_id].interaction_pattern
        for row in result.plan.queries
        if result.final_split_by_plan_id[row.plan_id] == "val"
    }
    if len(val_mapping) != 200:
        raise R2CoreBridgeError(
            "r2 validation interaction mapping must contain exactly 200 plan IDs"
        )
    if any(not interaction.strip() for interaction in val_mapping.values()):
        raise R2CoreBridgeError("r2 validation interaction mapping contains a blank")
    return val_mapping


def build_r2_core_bridge(
    result: R2CoreInMemoryPlan,
    *,
    trusted_plan_sha256: str,
    seed: int,
) -> R2CoreBridge:
    """Build and cross-check all pure r2 downstream projections at once."""

    constraints = build_r2_split_constraints(
        result,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    realism_sidecar = build_r2_realism_sidecar(
        result,
        trusted_plan_sha256=trusted_plan_sha256,
        seed=seed,
    )
    return R2CoreBridge(
        split_constraints=constraints,
        realism_sidecar=realism_sidecar,
        val_interaction_by_query_id=extract_r2_val_interaction_by_query_id(
            result,
            realism_sidecar,
            trusted_plan_sha256=trusted_plan_sha256,
        ),
    )
