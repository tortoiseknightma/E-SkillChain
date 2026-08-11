"""Deterministic, component-unique S1 Creator selection for portfolio core r2.

This module operates exclusively on in-memory planning and realism metadata.
It neither composes user text nor reads or writes corpus assets.  The returned
selection index is suitable for a later create-only publisher or S1 runner.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from scipy.optimize import Bounds, LinearConstraint, milp

from skillchain.synthesis.planning import (
    MVP_CAPABILITY_ORDER,
    R2CoreInMemoryPlan,
    audit_r2_core_in_memory_plan,
)
from skillchain.synthesis.portfolio_core_authoring import (
    RealismSidecar,
    ReuseAssignment,
    core_r2_realism_quota_spec,
    validate_realism_sidecar,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CREATOR_SELECTION_SIZE = 240
CREATOR_INITIAL_CAPABILITY_TARGET = 40
CREATOR_SELECTION_POLICY_VERSION = "portfolio-core-r2-s1-component-unique-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreatorSelectionError(ValueError):
    """The frozen r2 inputs cannot safely support the requested S1 selection."""


class CreatorSelectionPolicy(_StrictModel):
    """Fixed, text-free policy for the first 240 S1 Creator trajectories."""

    policy_version: Literal["portfolio-core-r2-s1-component-unique-v1"] = (
        CREATOR_SELECTION_POLICY_VERSION
    )
    selection_size: Literal[240] = CREATOR_SELECTION_SIZE
    initial_capability_target: Literal[40] = CREATOR_INITIAL_CAPABILITY_TARGET


class CreatorSelectionEntry(_StrictModel):
    """One selected opt-pool planner row without any generated text or path."""

    schema_version: Literal[1] = 1
    selection_rank: int = Field(ge=1, le=CREATOR_SELECTION_SIZE)
    plan_id: str
    component_id: str
    canonical_capability: str
    final_split: Literal["opt_pool"] = "opt_pool"
    is_boundary: bool
    interaction_pattern: str

    @field_validator(
        "plan_id", "component_id", "canonical_capability", "interaction_pattern"
    )
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must be non-blank")
        return value


class CreatorSelectionManifest(_StrictModel):
    """Digest binding for one complete deterministic 240-row selection index."""

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-r2-s1-component-unique-v1"] = (
        CREATOR_SELECTION_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    selection_size: Literal[240]
    policy_sha256: Sha256
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    prompt_recipe_manifest_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    reuse_sidecar_sha256: Sha256
    selection_bytes_sha256: Sha256


class CreatorSelectionAudit(_StrictModel):
    """Count-only proof that the index meets its quota and coverage contract."""

    selection_size: Literal[240]
    candidate_count: int = Field(ge=CREATOR_SELECTION_SIZE)
    available_unique_components_by_capability: dict[str, int]
    quota_by_capability: dict[str, int]
    selected_by_capability: dict[str, int]
    selected_unique_component_count: Literal[240]
    candidate_interaction_boundary_cells: tuple[str, ...]
    target_interaction_boundary_counts: dict[str, int]
    selected_interaction_boundary_counts: dict[str, int]
    interaction_boundary_l1_deviation: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_counts(self):
        expected_capabilities = set(MVP_CAPABILITY_ORDER)
        count_maps = (
            self.available_unique_components_by_capability,
            self.quota_by_capability,
            self.selected_by_capability,
        )
        if any(set(values) != expected_capabilities for values in count_maps):
            raise ValueError("creator selection capability maps must cover MVP order")
        if any(value < 0 for values in count_maps for value in values.values()):
            raise ValueError("creator selection counts must be non-negative")
        if sum(self.quota_by_capability.values()) != CREATOR_SELECTION_SIZE:
            raise ValueError("creator selection quotas must sum to 240")
        if self.selected_by_capability != self.quota_by_capability:
            raise ValueError("creator selected capability counts must equal quotas")
        if any(
            self.quota_by_capability[capability]
            > self.available_unique_components_by_capability[capability]
            for capability in MVP_CAPABILITY_ORDER
        ):
            raise ValueError(
                "creator selection quota exceeds unique component capacity"
            )
        cells = self.candidate_interaction_boundary_cells
        if not cells or cells != tuple(sorted(cells)) or len(cells) != len(set(cells)):
            raise ValueError(
                "creator interaction/boundary cells must be sorted and unique"
            )
        if set(self.target_interaction_boundary_counts) != set(cells) or set(
            self.selected_interaction_boundary_counts
        ) != set(cells):
            raise ValueError(
                "creator interaction/boundary count maps drifted from cells"
            )
        if any(
            value < 1 for value in self.selected_interaction_boundary_counts.values()
        ):
            raise ValueError(
                "every non-empty interaction/boundary cell must be covered"
            )
        if (
            sum(self.selected_interaction_boundary_counts.values())
            != CREATOR_SELECTION_SIZE
        ):
            raise ValueError("creator interaction/boundary counts must sum to 240")
        return self


@dataclass(frozen=True)
class CoreR2CreatorSelection:
    entries: tuple[CreatorSelectionEntry, ...]
    manifest: CreatorSelectionManifest
    audit: CreatorSelectionAudit
    policy: CreatorSelectionPolicy


@dataclass(frozen=True)
class _Candidate:
    plan_id: str
    component_id: str
    canonical_capability: str
    is_boundary: bool
    interaction_pattern: str

    @property
    def cell(self) -> tuple[str, str, bool]:
        return (
            self.canonical_capability,
            self.interaction_pattern,
            self.is_boundary,
        )


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CreatorSelectionError(f"{label} must be a 64-character lowercase SHA-256")


def _canonical_plan_sha256(result: R2CoreInMemoryPlan) -> str:
    return sha256_bytes(canonical_json_bytes(result.plan))


def _final_split_sidecar(result: R2CoreInMemoryPlan) -> dict[str, str]:
    return {
        row.plan_id: result.final_split_by_plan_id[row.plan_id]
        for row in result.plan.queries
    }


def _reuse_sidecar(result: R2CoreInMemoryPlan) -> dict[str, ReuseAssignment]:
    return {
        row.plan_id: ReuseAssignment(
            plan_id=row.plan_id,
            reuse_variant=result.reuse_variant_by_plan_id[row.plan_id],
            reuse_reason=result.reuse_reason_by_plan_id[row.plan_id],
        )
        for row in result.plan.queries
    }


def _validate_inputs(
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
) -> tuple[_Candidate, ...]:
    """Re-audit all mutable mappings and validate the canonical realism packet."""

    _require_sha256(trusted_plan_sha256, "trusted_plan_sha256")
    if _canonical_plan_sha256(result) != trusted_plan_sha256:
        raise CreatorSelectionError(
            "trusted plan SHA-256 does not match the in-memory CorpusPlan"
        )
    try:
        audit = audit_r2_core_in_memory_plan(result)
    except (TypeError, ValueError, AssertionError) as exc:
        raise CreatorSelectionError("r2 in-memory plan audit failed") from exc
    if audit != result.audit:
        raise CreatorSelectionError("stored r2 plan audit no longer matches its layout")
    _require_sha256(result.plan.asset_catalog_sha256, "asset_catalog_sha256")
    _require_sha256(
        result.plan.capability_assignments_sha256,
        "capability_assignments_sha256",
    )
    expected_manifest = {
        "plan_sha256": trusted_plan_sha256,
        "asset_catalog_sha256": result.plan.asset_catalog_sha256,
        "capability_assignments_sha256": result.plan.capability_assignments_sha256,
    }
    for field_name, expected in expected_manifest.items():
        if getattr(realism_sidecar.manifest, field_name) != expected:
            raise CreatorSelectionError(
                f"realism sidecar binding mismatch: {field_name}"
            )
    try:
        validate_realism_sidecar(
            realism_sidecar,
            plan_rows=result.plan.queries,
            final_split_by_plan_id=_final_split_sidecar(result),
            reuse_by_plan_id=_reuse_sidecar(result),
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise CreatorSelectionError("realism sidecar validation failed") from exc
    assignments = {
        assignment.plan_id: assignment for assignment in realism_sidecar.assignments
    }
    plan_ids = {row.plan_id for row in result.plan.queries}
    if set(assignments) != plan_ids:
        raise CreatorSelectionError("realism sidecar does not cover the r2 plan")
    candidates: list[_Candidate] = []
    for row in result.plan.queries:
        if result.final_split_by_plan_id[row.plan_id] != "opt_pool":
            continue
        component_id = row.leakage_group_id
        if not component_id or component_id != component_id.strip():
            raise CreatorSelectionError("opt-pool row has an invalid leakage component")
        assignment = assignments[row.plan_id]
        candidates.append(
            _Candidate(
                plan_id=row.plan_id,
                component_id=component_id,
                canonical_capability=row.canonical_capability,
                is_boundary=row.is_boundary,
                interaction_pattern=assignment.interaction_pattern,
            )
        )
    if len(candidates) < CREATOR_SELECTION_SIZE:
        raise CreatorSelectionError("opt_pool has fewer than 240 available plan rows")
    if any(
        candidate.canonical_capability not in MVP_CAPABILITY_ORDER
        for candidate in candidates
    ):
        raise CreatorSelectionError("opt_pool row has an unknown canonical capability")
    return tuple(candidates)


def compute_component_aware_quotas(
    available_unique_components_by_capability: Mapping[str, int],
    *,
    selection_size: int = CREATOR_SELECTION_SIZE,
) -> dict[str, int]:
    """Allocate the fixed 240 slots under per-capability component capacity.

    Every capability first receives ``min(40, available_unique_components)``.
    Any remaining slots are given one at a time to the lowest selected-to-
    available ratio; frozen MVP capability order resolves equal ratios.
    """

    if selection_size != CREATOR_SELECTION_SIZE:
        raise CreatorSelectionError("creator selection size is frozen at 240")
    if set(available_unique_components_by_capability) != set(MVP_CAPABILITY_ORDER):
        raise CreatorSelectionError(
            "available component map must cover MVP capability order"
        )
    available = {
        capability: available_unique_components_by_capability[capability]
        for capability in MVP_CAPABILITY_ORDER
    }
    if any(not isinstance(value, int) or value < 0 for value in available.values()):
        raise CreatorSelectionError(
            "available component counts must be non-negative ints"
        )
    quotas = {
        capability: min(CREATOR_INITIAL_CAPABILITY_TARGET, available[capability])
        for capability in MVP_CAPABILITY_ORDER
    }
    while sum(quotas.values()) < selection_size:
        eligible = [
            capability
            for capability in MVP_CAPABILITY_ORDER
            if quotas[capability] < available[capability]
        ]
        if not eligible:
            raise CreatorSelectionError(
                "unique component capacity cannot supply the required 240 selections"
            )
        capability = min(
            eligible,
            key=lambda item: (
                Fraction(quotas[item], available[item]),
                MVP_CAPABILITY_ORDER.index(item),
            ),
        )
        quotas[capability] += 1
    if sum(quotas.values()) != selection_size:
        raise AssertionError(
            "creator quota allocator overshot its fixed selection size"
        )
    return quotas


def _cell_key(cell: tuple[str, str, bool]) -> str:
    capability, interaction_pattern, is_boundary = cell
    return (
        f"{capability}|{interaction_pattern}|"
        f"boundary={'true' if is_boundary else 'false'}"
    )


def _seed_key(seed: int, namespace: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}\x00{namespace}\x00{value}".encode("utf-8")
    ).hexdigest()
    return digest, value


def _target_cell_counts(
    cells: tuple[tuple[str, str, bool], ...],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> dict[tuple[str, str, bool], int]:
    if not cells:
        raise CreatorSelectionError("no eligible interaction/boundary cells exist")
    targets: dict[tuple[str, str, bool], int] = {}
    for capability in MVP_CAPABILITY_ORDER:
        capability_cells = tuple(
            cell for cell in cells if cell[0] == capability
        )
        if quotas[capability] > 0 and not capability_cells:
            raise CreatorSelectionError(
                f"no interaction/boundary cells exist for {capability}"
            )
        if not capability_cells:
            continue
        base, remainder = divmod(quotas[capability], len(capability_cells))
        extras = set(
            sorted(
                capability_cells,
                key=lambda cell: _seed_key(
                    seed,
                    f"cell-target/{capability}",
                    _cell_key(cell),
                ),
            )[:remainder]
        )
        targets.update(
            {cell: base + int(cell in extras) for cell in capability_cells}
        )
    if sum(targets.values()) != CREATOR_SELECTION_SIZE:
        raise AssertionError("creator cell targets do not sum to 240")
    return targets


def _solve_component_unique_selection(
    candidates: tuple[_Candidate, ...],
    quotas: Mapping[str, int],
    *,
    seed: int,
) -> tuple[tuple[_Candidate, ...], dict[str, int], dict[str, int], int]:
    """Solve exact capability quotas with global component uniqueness.

    Phase one minimizes L1 distance from an even capability-by-interaction-by-
    boundary allocation while requiring at least one row from every non-empty
    *global* interaction-by-boundary cell.  Some capability-specific rare cells
    share their sole component across capabilities, so hard-covering every fine
    cell would contradict global component uniqueness.  Phase two holds the L1
    optimum fixed and applies a stable seed-derived ordering as the
    deterministic tie break.
    """

    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda item: _seed_key(seed, "candidate", item.plan_id),
        )
    )
    cells = tuple(
        sorted(
            {
                candidate.cell
                for candidate in ordered_candidates
                if quotas[candidate.canonical_capability] > 0
            }
        )
    )
    global_cells = tuple(
        sorted(
            {
                (candidate.interaction_pattern, candidate.is_boundary)
                for candidate in ordered_candidates
                if quotas[candidate.canonical_capability] > 0
            }
        )
    )
    target_by_cell = _target_cell_counts(cells, quotas=quotas, seed=seed)
    candidate_count = len(ordered_candidates)
    cell_count = len(cells)
    variable_count = candidate_count + 2 * cell_count
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, CREATOR_SELECTION_SIZE, dtype=np.float64)
    upper_bounds[:candidate_count] = 1.0
    integrality = np.ones(variable_count, dtype=np.int32)
    index_by_cell = {cell: index for index, cell in enumerate(cells)}
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        row = np.zeros(variable_count, dtype=np.float64)
        for index, coefficient in coefficients.items():
            row[index] = coefficient
        rows.append(row)
        lower.append(lower_bound)
        upper.append(upper_bound)

    for capability in MVP_CAPABILITY_ORDER:
        add_constraint(
            {
                index: 1.0
                for index, candidate in enumerate(ordered_candidates)
                if candidate.canonical_capability == capability
            },
            quotas[capability],
            quotas[capability],
        )
    component_indices: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(ordered_candidates):
        component_indices[candidate.component_id].append(index)
    for indices in component_indices.values():
        add_constraint({index: 1.0 for index in indices}, 0.0, 1.0)
    # Cross-intent rows can make a capability-specific rare cell share its only
    # component with another capability.  Requiring every such cell would
    # violate global component uniqueness.  We therefore hard-cover each
    # user-behaviour interaction/boundary cell globally, while the L1 objective
    # below balances the finer capability x interaction x boundary cells.
    for interaction_pattern, is_boundary in global_cells:
        indices = [
            index
            for index, candidate in enumerate(ordered_candidates)
            if candidate.interaction_pattern == interaction_pattern
            and candidate.is_boundary == is_boundary
        ]
        add_constraint({index: 1.0 for index in indices}, 1.0, np.inf)
    for cell in cells:
        indices = [
            index
            for index, candidate in enumerate(ordered_candidates)
            if candidate.cell == cell
        ]
        if not indices:
            raise AssertionError(
                "a reported non-empty cell unexpectedly has no candidate"
            )
        cell_index = index_by_cell[cell]
        positive_deviation = candidate_count + 2 * cell_index
        negative_deviation = positive_deviation + 1
        coefficients = {index: 1.0 for index in indices}
        coefficients[positive_deviation] = -1.0
        coefficients[negative_deviation] = 1.0
        add_constraint(
            coefficients,
            target_by_cell[cell],
            target_by_cell[cell],
        )
    matrix = np.vstack(rows)
    lower_vector = np.asarray(lower, dtype=np.float64)
    upper_vector = np.asarray(upper, dtype=np.float64)

    def solve(
        objective: np.ndarray,
        *,
        extra_rows: Iterable[np.ndarray] = (),
        extra_lower: Iterable[float] = (),
        extra_upper: Iterable[float] = (),
    ) -> np.ndarray:
        extra_rows_tuple = tuple(extra_rows)
        if extra_rows_tuple:
            constraint_matrix = np.vstack((matrix, *extra_rows_tuple))
            constraint_lower = np.concatenate(
                (lower_vector, np.asarray(tuple(extra_lower), dtype=np.float64))
            )
            constraint_upper = np.concatenate(
                (upper_vector, np.asarray(tuple(extra_upper), dtype=np.float64))
            )
        else:
            constraint_matrix = matrix
            constraint_lower = lower_vector
            constraint_upper = upper_vector
        outcome = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(
                constraint_matrix,
                constraint_lower,
                constraint_upper,
            ),
            options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": 20.0},
        )
        if outcome.status != 0 or outcome.x is None:
            raise CreatorSelectionError(
                "component-unique creator selection is infeasible: "
                f"status={outcome.status}; message={outcome.message}"
            )
        rounded = np.rint(outcome.x)
        if np.max(np.abs(outcome.x - rounded), initial=0.0) > 1e-6:
            raise CreatorSelectionError(
                "creator selection MILP returned a non-integral result"
            )
        return rounded.astype(np.int64)

    primary_objective = np.zeros(variable_count, dtype=np.float64)
    primary_objective[candidate_count:] = 1.0
    primary = solve(primary_objective)
    minimum_deviation = int(primary[candidate_count:].sum())
    fixed_deviation_row = np.zeros(variable_count, dtype=np.float64)
    fixed_deviation_row[candidate_count:] = 1.0
    tie_objective = np.zeros(variable_count, dtype=np.float64)
    tie_objective[:candidate_count] = np.arange(1, candidate_count + 1)
    chosen = solve(
        tie_objective,
        extra_rows=(fixed_deviation_row,),
        extra_lower=(float(minimum_deviation),),
        extra_upper=(float(minimum_deviation),),
    )
    selected = tuple(
        candidate
        for index, candidate in enumerate(ordered_candidates)
        if chosen[index] == 1
    )
    if len(selected) != CREATOR_SELECTION_SIZE:
        raise CreatorSelectionError(
            "creator selection did not contain exactly 240 rows"
        )
    selected_capabilities = Counter(
        candidate.canonical_capability for candidate in selected
    )
    if {
        capability: selected_capabilities[capability]
        for capability in MVP_CAPABILITY_ORDER
    } != dict(quotas):
        raise CreatorSelectionError(
            "creator selection did not meet exact capability quotas"
        )
    if len({candidate.component_id for candidate in selected}) != len(selected):
        raise CreatorSelectionError("creator selection reused one leakage component")
    selected_cell_counts = Counter(_cell_key(candidate.cell) for candidate in selected)
    target_cell_counts = {_cell_key(cell): target_by_cell[cell] for cell in cells}
    observed_deviation = sum(
        abs(selected_cell_counts[key] - target)
        for key, target in target_cell_counts.items()
    )
    if observed_deviation != minimum_deviation:
        raise CreatorSelectionError("creator selection balance objective drifted")
    return (
        selected,
        target_cell_counts,
        {key: selected_cell_counts[key] for key in sorted(target_cell_counts)},
        minimum_deviation,
    )


def _available_component_counts(
    candidates: Iterable[_Candidate],
) -> dict[str, int]:
    components_by_capability: dict[str, set[str]] = {
        capability: set() for capability in MVP_CAPABILITY_ORDER
    }
    for candidate in candidates:
        components_by_capability[candidate.canonical_capability].add(
            candidate.component_id
        )
    return {
        capability: len(components_by_capability[capability])
        for capability in MVP_CAPABILITY_ORDER
    }


def _selection_entries(
    candidates: Iterable[_Candidate],
    *,
    seed: int,
) -> tuple[CreatorSelectionEntry, ...]:
    ordered = sorted(
        candidates,
        key=lambda item: _seed_key(seed, "selected-rank", item.plan_id),
    )
    return tuple(
        CreatorSelectionEntry(
            selection_rank=index,
            plan_id=candidate.plan_id,
            component_id=candidate.component_id,
            canonical_capability=candidate.canonical_capability,
            is_boundary=candidate.is_boundary,
            interaction_pattern=candidate.interaction_pattern,
        )
        for index, candidate in enumerate(ordered, start=1)
    )


def _selection_audit(
    candidates: tuple[_Candidate, ...],
    quotas: Mapping[str, int],
    selected: tuple[_Candidate, ...],
    target_cell_counts: Mapping[str, int],
    selected_cell_counts: Mapping[str, int],
    deviation: int,
) -> CreatorSelectionAudit:
    available = _available_component_counts(candidates)
    selected_capabilities = Counter(
        candidate.canonical_capability for candidate in selected
    )
    candidate_global_cells = tuple(
        sorted(
            {
                f"{candidate.interaction_pattern}|"
                f"boundary={'true' if candidate.is_boundary else 'false'}"
                for candidate in candidates
            }
        )
    )
    target_global_counts: Counter[str] = Counter()
    for key, value in target_cell_counts.items():
        target_global_counts[key.split("|", 1)[1]] += value
    selected_global_counts: Counter[str] = Counter(
        f"{candidate.interaction_pattern}|"
        f"boundary={'true' if candidate.is_boundary else 'false'}"
        for candidate in selected
    )
    return CreatorSelectionAudit(
        selection_size=CREATOR_SELECTION_SIZE,
        candidate_count=len(candidates),
        available_unique_components_by_capability=available,
        quota_by_capability={
            capability: quotas[capability] for capability in MVP_CAPABILITY_ORDER
        },
        selected_by_capability={
            capability: selected_capabilities[capability]
            for capability in MVP_CAPABILITY_ORDER
        },
        selected_unique_component_count=len(
            {candidate.component_id for candidate in selected}
        ),
        candidate_interaction_boundary_cells=candidate_global_cells,
        target_interaction_boundary_counts=dict(sorted(target_global_counts.items())),
        selected_interaction_boundary_counts={
            key: selected_global_counts[key] for key in candidate_global_cells
        },
        interaction_boundary_l1_deviation=deviation,
    )


def canonical_creator_selection_index_bytes(
    entries: Iterable[CreatorSelectionEntry],
) -> bytes:
    """Canonical bytes for a create-only selection-index publisher."""

    materialized = tuple(entries)
    ranks = [entry.selection_rank for entry in materialized]
    if ranks != list(range(1, len(materialized) + 1)):
        raise ValueError("creator selection ranks must be contiguous from one")
    plan_ids = [entry.plan_id for entry in materialized]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("creator selection plan IDs must be unique")
    return canonical_jsonl_bytes(materialized)


def _manifest(
    *,
    policy: CreatorSelectionPolicy,
    seed: int,
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    trusted_plan_sha256: str,
    entries: tuple[CreatorSelectionEntry, ...],
) -> CreatorSelectionManifest:
    return CreatorSelectionManifest(
        seed=seed,
        selection_size=CREATOR_SELECTION_SIZE,
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        plan_sha256=trusted_plan_sha256,
        asset_catalog_sha256=result.plan.asset_catalog_sha256,
        capability_assignments_sha256=result.plan.capability_assignments_sha256,
        realism_manifest_sha256=sha256_bytes(
            canonical_json_bytes(realism_sidecar.manifest)
        ),
        realism_assignments_sha256=realism_sidecar.manifest.assignments_sha256,
        prompt_recipe_manifest_sha256=(
            realism_sidecar.manifest.prompt_recipe_manifest_sha256
        ),
        final_split_sidecar_sha256=realism_sidecar.manifest.final_split_sidecar_sha256,
        reuse_sidecar_sha256=realism_sidecar.manifest.reuse_sidecar_sha256,
        selection_bytes_sha256=sha256_bytes(
            canonical_creator_selection_index_bytes(entries)
        ),
    )


def build_core_r2_creator_selection(
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
    seed: int,
    policy: CreatorSelectionPolicy | None = None,
) -> CoreR2CreatorSelection:
    """Build the deterministic 240-row opt-pool S1 Creator selection index."""

    effective_policy = policy or CreatorSelectionPolicy()
    if seed < 0:
        raise CreatorSelectionError("creator selection seed must be non-negative")
    candidates = _validate_inputs(
        result,
        realism_sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    available = _available_component_counts(candidates)
    quotas = compute_component_aware_quotas(
        available,
        selection_size=effective_policy.selection_size,
    )
    selected, target_cells, selected_cells, deviation = (
        _solve_component_unique_selection(
            candidates,
            quotas,
            seed=seed,
        )
    )
    entries = _selection_entries(selected, seed=seed)
    audit = _selection_audit(
        candidates,
        quotas,
        selected,
        target_cells,
        selected_cells,
        deviation,
    )
    selection = CoreR2CreatorSelection(
        entries=entries,
        manifest=_manifest(
            policy=effective_policy,
            seed=seed,
            result=result,
            realism_sidecar=realism_sidecar,
            trusted_plan_sha256=trusted_plan_sha256,
            entries=entries,
        ),
        audit=audit,
        policy=effective_policy,
    )
    validate_core_r2_creator_selection(
        selection,
        result,
        realism_sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    return selection


def validate_core_r2_creator_selection(
    selection: CoreR2CreatorSelection,
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
) -> None:
    """Fail closed on input, index, manifest, quota, or deterministic drift."""

    candidates = _validate_inputs(
        result,
        realism_sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    manifest = selection.manifest
    if (
        manifest.policy_version != selection.policy.policy_version
        or manifest.selection_size != selection.policy.selection_size
    ):
        raise CreatorSelectionError("creator selection manifest policy mismatch")
    expected_manifest = {
        "policy_sha256": sha256_bytes(canonical_json_bytes(selection.policy)),
        "plan_sha256": trusted_plan_sha256,
        "asset_catalog_sha256": result.plan.asset_catalog_sha256,
        "capability_assignments_sha256": result.plan.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(realism_sidecar.manifest)
        ),
        "realism_assignments_sha256": realism_sidecar.manifest.assignments_sha256,
        "prompt_recipe_manifest_sha256": (
            realism_sidecar.manifest.prompt_recipe_manifest_sha256
        ),
        "final_split_sidecar_sha256": realism_sidecar.manifest.final_split_sidecar_sha256,
        "reuse_sidecar_sha256": realism_sidecar.manifest.reuse_sidecar_sha256,
    }
    for field_name, expected in expected_manifest.items():
        if getattr(manifest, field_name) != expected:
            raise CreatorSelectionError(
                f"creator selection manifest {field_name} mismatch"
            )
    try:
        index_bytes = canonical_creator_selection_index_bytes(selection.entries)
    except ValueError as exc:
        raise CreatorSelectionError("creator selection index is malformed") from exc
    if manifest.selection_bytes_sha256 != sha256_bytes(index_bytes):
        raise CreatorSelectionError("creator selection bytes digest mismatch")
    if len(selection.entries) != CREATOR_SELECTION_SIZE:
        raise CreatorSelectionError("creator selection must contain exactly 240 rows")
    entry_plan_ids = [entry.plan_id for entry in selection.entries]
    component_ids = [entry.component_id for entry in selection.entries]
    if len(component_ids) != len(set(component_ids)):
        raise CreatorSelectionError("creator selection contains a duplicate component")
    if len(entry_plan_ids) != len(set(entry_plan_ids)):
        raise CreatorSelectionError("creator selection contains a duplicate plan_id")
    candidate_by_plan_id = {candidate.plan_id: candidate for candidate in candidates}
    for entry in selection.entries:
        candidate = candidate_by_plan_id.get(entry.plan_id)
        if candidate is None:
            raise CreatorSelectionError(
                "creator selection contains a non-opt-pool plan_id"
            )
        expected_entry = (
            candidate.component_id,
            candidate.canonical_capability,
            candidate.is_boundary,
            candidate.interaction_pattern,
        )
        observed_entry = (
            entry.component_id,
            entry.canonical_capability,
            entry.is_boundary,
            entry.interaction_pattern,
        )
        if observed_entry != expected_entry:
            raise CreatorSelectionError(
                f"creator selection row drifted from plan/realism inputs: {entry.plan_id}"
            )
        if entry.final_split != "opt_pool":
            raise CreatorSelectionError(
                "creator selection may only contain opt_pool rows"
            )
    available = _available_component_counts(candidates)
    quotas = compute_component_aware_quotas(
        available,
        selection_size=selection.policy.selection_size,
    )
    selected, target_cells, selected_cells, deviation = (
        _solve_component_unique_selection(
            candidates,
            quotas,
            seed=manifest.seed,
        )
    )
    expected_entries = _selection_entries(selected, seed=manifest.seed)
    if selection.entries != expected_entries:
        raise CreatorSelectionError(
            "creator selection drifted from deterministic optimum"
        )
    expected_audit = _selection_audit(
        candidates,
        quotas,
        selected,
        target_cells,
        selected_cells,
        deviation,
    )
    if selection.audit != expected_audit:
        raise CreatorSelectionError(
            "creator selection audit drifted from deterministic index"
        )


def creator_selection_index_bytes(selection: CoreR2CreatorSelection) -> bytes:
    """Convenience accessor for a later create-only selection-index publisher."""

    return canonical_creator_selection_index_bytes(selection.entries)
