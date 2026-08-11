"""Deterministic opt800 folds and the discovery-only S1 Creator v2 index.

The module is deliberately metadata-only.  It accepts a projection of the
already materialised ``opt_pool`` rows, keeps every field in ``GROUP_FIELDS``
atomic, and never reads query text, images, validation rows, or test rows.
Publishing is owned by a separate create-only command.
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

from skillchain.synthesis.models import CorpusPlan
from skillchain.synthesis.planning import (
    CORE_R2_CAPABILITY_BOUNDARY_COUNTS,
    CORE_R2_CAPABILITY_COUNTS,
    CORE_R2_FINAL_SPLITS,
    MVP_CAPABILITY_ORDER,
    R2CoreInMemoryPlan,
)
from skillchain.synthesis.portfolio_core_authoring import (
    RealismSidecar,
    ReuseAssignment,
    core_r2_realism_quota_spec,
    validate_realism_sidecar,
)
from skillchain.synthesis.portfolio_core_selection import (
    CREATOR_INITIAL_CAPABILITY_TARGET,
    CREATOR_SELECTION_SIZE,
    CreatorSelectionAudit,
    CreatorSelectionEntry,
    CreatorSelectionError,
    _Candidate,
    _available_component_counts,
    _selection_audit,
    _selection_entries,
    _solve_component_unique_selection,
    _validate_inputs,
    canonical_creator_selection_index_bytes,
)
from skillchain.synthesis.splitting import (
    GROUP_FIELDS,
    GROUPING_POLICY_VERSION,
    R2SplitConstraints,
    verify_r2_split_constraint_binding,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FoldId = Literal["fold-00", "fold-01", "fold-02", "fold-03"]
FoldRole = Literal["replay", "discovery"]

OPT_QUERY_COUNT = 800
OPT_FOLD_COUNT = 4
OPT_FOLD_SIZE = 200
OPT_BATCH_COUNT = 32
OPT_BATCH_SIZE = 25
OPT_BATCHES_PER_FOLD = 8
OPT_FOLD_IDS: tuple[FoldId, ...] = (
    "fold-00",
    "fold-01",
    "fold-02",
    "fold-03",
)
REPLAY_FOLD: FoldId = "fold-00"
DISCOVERY_FOLDS: tuple[FoldId, ...] = ("fold-01", "fold-02", "fold-03")
OPT_FOLD_POLICY_VERSION = "portfolio-core-opt800-group-folds-v1"
CREATOR_V2_POLICY_VERSION = "portfolio-core-r2-s1-discovery600-v2"
CORE_R3_QUERIES_SHA256 = (
    "e2899c5b0813d796692ab175cabfc722e3f3a59f61b658adb939c2e283895e87"
)
CORE_R2_PLAN_SHA256 = (
    "b1e637a6801eefa4a66eab8c05ec5a03ded31e1693e0296deb46aee82602dc11"
)
LEGACY_CREATOR_SELECTION_SHA256 = (
    "eeb4fc8e57788ded6393d3575c442e429c2aa03c8dd083ab2fd850da78b5c593"
)
DOCUMENT_CAPABILITY = "utility.document_reading"
CREATOR_V2_DOCUMENT_QUOTA = 24


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OptFoldError(ValueError):
    """The opt800 projection cannot satisfy the frozen fold contract."""


@dataclass(frozen=True)
class PublishedR2SelectionContext:
    """Typed selection inputs reconstructed from one immutable r2 publication.

    Unlike :class:`R2CoreInMemoryPlan`, this evidence does not claim that the
    current source tree can regenerate the historical capability assignments.
    It binds and cross-validates the already-published plan and sidecars, which
    is the correct replay boundary after a TaskSpec implementation update.
    """

    plan: CorpusPlan
    final_split_by_plan_id: Mapping[str, str]
    reuse_by_plan_id: Mapping[str, ReuseAssignment]
    split_constraints: R2SplitConstraints
    realism_sidecar: RealismSidecar
    trusted_plan_sha256: str


class OptFoldQuery(_StrictModel):
    """Text-free projection consumed by the opt fold planner."""

    schema_version: Literal[1] = 1
    query_id: str
    leakage_group_id: str
    boundary_group_id: str | None = None
    template_family: str
    generator_batch_id: str
    canonical_capability: str
    is_boundary: bool

    @field_validator(
        "query_id",
        "leakage_group_id",
        "template_family",
        "generator_batch_id",
        "canonical_capability",
    )
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank without padding")
        return value

    @field_validator("boundary_group_id")
    @classmethod
    def _optional_nonblank(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("boundary_group_id must be non-blank without padding")
        return value

    @model_validator(mode="after")
    def _known_capability(self):
        if self.canonical_capability not in MVP_CAPABILITY_ORDER:
            raise ValueError("canonical_capability is outside the frozen MVP order")
        if bool(self.boundary_group_id) and not self.is_boundary:
            raise ValueError("non-boundary query cannot carry boundary_group_id")
        return self


class OptFoldPolicy(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-opt800-group-folds-v1"] = (
        OPT_FOLD_POLICY_VERSION
    )
    fold_count: Literal[4] = OPT_FOLD_COUNT
    fold_size: Literal[200] = OPT_FOLD_SIZE
    batch_count: Literal[32] = OPT_BATCH_COUNT
    batch_size: Literal[25] = OPT_BATCH_SIZE
    batches_per_fold: Literal[8] = OPT_BATCHES_PER_FOLD
    replay_fold: Literal["fold-00"] = REPLAY_FOLD
    discovery_folds: tuple[FoldId, ...] = DISCOVERY_FOLDS
    group_fields: tuple[str, ...] = GROUP_FIELDS
    grouping_policy_version: Literal["query-connected-components-v1"] = (
        GROUPING_POLICY_VERSION
    )
    balance_objective: Literal["capability-boundary-l1"] = (
        "capability-boundary-l1"
    )

    @model_validator(mode="after")
    def _frozen_contract(self):
        if self.discovery_folds != DISCOVERY_FOLDS:
            raise ValueError("discovery_folds drifted from the frozen contract")
        if self.group_fields != GROUP_FIELDS:
            raise ValueError("opt folds must retain the complete GROUP_FIELDS")
        return self


class LegacySelectionConflictAudit(_StrictModel):
    schema_version: Literal[1] = 1
    selection_bytes_sha256: Sha256
    selected_query_count: Literal[240]
    touched_atomic_batch_count: Literal[32]
    closure_query_count: Literal[800]
    remaining_replay_count: Literal[0]
    incompatible_with_nonempty_replay: Literal[True] = True


class OptFoldAssignment(_StrictModel):
    schema_version: Literal[1] = 1
    query_id: str
    atomic_batch_id: str
    fold_id: FoldId
    role: FoldRole

    @field_validator("query_id", "atomic_batch_id")
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank without padding")
        return value

    @model_validator(mode="after")
    def _role_matches_fold(self):
        expected = "replay" if self.fold_id == REPLAY_FOLD else "discovery"
        if self.role != expected:
            raise ValueError("fold assignment role does not match fold_id")
        return self


class OptFoldAudit(_StrictModel):
    schema_version: Literal[1] = 1
    query_count: Literal[800]
    atomic_batch_count: Literal[32]
    fold_query_counts: dict[str, int]
    fold_batch_counts: dict[str, int]
    capability_counts_by_fold: dict[str, dict[str, int]]
    boundary_counts_by_fold: dict[str, dict[str, int]]
    discovery_unique_component_count: int = Field(ge=CREATOR_SELECTION_SIZE)
    discovery_unique_components_by_capability: dict[str, int]
    group_field_cross_fold_violations: Literal[0] = 0
    capability_boundary_l1_deviation: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_match_contract(self):
        expected_folds = set(OPT_FOLD_IDS)
        if set(self.fold_query_counts) != expected_folds or set(
            self.fold_batch_counts
        ) != expected_folds:
            raise ValueError("fold count maps must cover all four folds")
        if any(value != OPT_FOLD_SIZE for value in self.fold_query_counts.values()):
            raise ValueError("each fold must contain exactly 200 queries")
        if any(
            value != OPT_BATCHES_PER_FOLD
            for value in self.fold_batch_counts.values()
        ):
            raise ValueError("each fold must contain exactly eight batches")
        if set(self.capability_counts_by_fold) != expected_folds or any(
            set(values) != set(MVP_CAPABILITY_ORDER)
            or any(count < 1 for count in values.values())
            for values in self.capability_counts_by_fold.values()
        ):
            raise ValueError("every fold must cover all six capabilities")
        if set(self.discovery_unique_components_by_capability) != set(
            MVP_CAPABILITY_ORDER
        ):
            raise ValueError("discovery component map must cover MVP capability order")
        if self.discovery_unique_components_by_capability[DOCUMENT_CAPABILITY] < (
            CREATOR_V2_DOCUMENT_QUOTA
        ):
            raise ValueError("discovery does not support the frozen Document quota")
        return self


class OptFoldManifest(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-opt800-group-folds-v1"] = (
        OPT_FOLD_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    source_queries_sha256: Sha256
    plan_sha256: Sha256
    opt_projection_sha256: Sha256
    policy_sha256: Sha256
    mapping_sha256: Sha256
    discovery_query_ids_sha256: Sha256
    replay_query_ids_sha256: Sha256
    legacy_conflict_audit: LegacySelectionConflictAudit


@dataclass(frozen=True)
class OptFoldPlan:
    assignments: tuple[OptFoldAssignment, ...]
    manifest: OptFoldManifest
    audit: OptFoldAudit
    policy: OptFoldPolicy


class CreatorSelectionV2Policy(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-r2-s1-discovery600-v2"] = (
        CREATOR_V2_POLICY_VERSION
    )
    selection_size: Literal[240] = CREATOR_SELECTION_SIZE
    initial_capability_target: Literal[40] = CREATOR_INITIAL_CAPABILITY_TARGET
    document_quota: Literal[24] = CREATOR_V2_DOCUMENT_QUOTA
    eligible_folds: tuple[FoldId, ...] = DISCOVERY_FOLDS

    @model_validator(mode="after")
    def _eligible_folds(self):
        if self.eligible_folds != DISCOVERY_FOLDS:
            raise ValueError("Creator v2 must use exactly the three discovery folds")
        return self


class CreatorSelectionV2Manifest(_StrictModel):
    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-core-r2-s1-discovery600-v2"] = (
        CREATOR_V2_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    selection_size: Literal[240]
    policy_sha256: Sha256
    source_queries_sha256: Sha256
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    prompt_recipe_manifest_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    reuse_sidecar_sha256: Sha256
    fold_mapping_sha256: Sha256
    discovery_query_ids_sha256: Sha256
    replay_query_ids_sha256: Sha256
    supersedes_selection_bytes_sha256: Sha256
    legacy_closure_query_count: Literal[800]
    legacy_remaining_replay_count: Literal[0]
    selection_bytes_sha256: Sha256


@dataclass(frozen=True)
class CoreR2CreatorSelectionV2:
    entries: tuple[CreatorSelectionEntry, ...]
    manifest: CreatorSelectionV2Manifest
    audit: CreatorSelectionAudit
    policy: CreatorSelectionV2Policy


@dataclass(frozen=True)
class _AtomicBatch:
    atom_id: str
    queries: tuple[OptFoldQuery, ...]


def _seed_key(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{seed}\x00{namespace}\x00{value}".encode()).hexdigest(),
        value,
    )


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OptFoldError(f"{label} must be a lowercase SHA-256")


def canonical_opt_projection_bytes(queries: Iterable[OptFoldQuery]) -> bytes:
    rows = tuple(sorted(queries, key=lambda item: item.query_id))
    if len({row.query_id for row in rows}) != len(rows):
        raise OptFoldError("opt projection contains duplicate query_id")
    return canonical_jsonl_bytes(rows)


def canonical_opt_fold_mapping_bytes(
    assignments: Iterable[OptFoldAssignment],
) -> bytes:
    rows = tuple(sorted(assignments, key=lambda item: item.query_id))
    if len({row.query_id for row in rows}) != len(rows):
        raise OptFoldError("opt fold mapping contains duplicate query_id")
    return canonical_jsonl_bytes(rows)


def _ids_sha256(values: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(values)))


def _materialize_queries(queries: Iterable[OptFoldQuery]) -> tuple[OptFoldQuery, ...]:
    rows = tuple(sorted(queries, key=lambda item: item.query_id))
    if len(rows) != OPT_QUERY_COUNT:
        raise OptFoldError("opt fold planner requires exactly 800 opt_pool rows")
    if len({row.query_id for row in rows}) != len(rows):
        raise OptFoldError("opt fold planner received duplicate query_id")
    return rows


def _atomic_batches(queries: tuple[OptFoldQuery, ...]) -> tuple[_AtomicBatch, ...]:
    parent = list(range(len(queries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for field_name in GROUP_FIELDS:
        first_by_value: dict[str, int] = {}
        for index, query in enumerate(queries):
            value = getattr(query, field_name)
            if value is None:
                continue
            previous = first_by_value.setdefault(value, index)
            union(previous, index)
    indices_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(queries)):
        indices_by_root[find(index)].append(index)
    atoms: list[_AtomicBatch] = []
    for indices in indices_by_root.values():
        rows = tuple(sorted((queries[index] for index in indices), key=lambda q: q.query_id))
        batch_ids = {row.generator_batch_id for row in rows}
        template_families = {row.template_family for row in rows}
        if len(rows) != OPT_BATCH_SIZE or len(batch_ids) != 1:
            raise OptFoldError(
                "each GROUP_FIELDS atom must equal one complete 25-query generator batch"
            )
        if len(template_families) != 1:
            raise OptFoldError(
                "each GROUP_FIELDS atom must retain one template family"
            )
        atoms.append(_AtomicBatch(atom_id=next(iter(batch_ids)), queries=rows))
    ordered = tuple(sorted(atoms, key=lambda item: item.atom_id))
    if len(ordered) != OPT_BATCH_COUNT:
        raise OptFoldError("opt800 must contain exactly 32 GROUP_FIELDS atoms")
    return ordered


def audit_legacy_selection_conflict(
    queries: Iterable[OptFoldQuery],
    legacy_selected_query_ids: Iterable[str],
    *,
    selection_bytes_sha256: str = LEGACY_CREATOR_SELECTION_SHA256,
) -> LegacySelectionConflictAudit:
    """Prove that the historical 240 selection leaves no atomic replay rows."""

    rows = _materialize_queries(queries)
    atoms = _atomic_batches(rows)
    _require_sha256(selection_bytes_sha256, "legacy selection SHA-256")
    selected = tuple(legacy_selected_query_ids)
    if len(selected) != CREATOR_SELECTION_SIZE or len(set(selected)) != len(selected):
        raise OptFoldError("legacy selection must contain 240 unique query IDs")
    known = {row.query_id for row in rows}
    if not set(selected) <= known:
        raise OptFoldError("legacy selection contains a query outside opt800")
    selected_set = set(selected)
    touched = tuple(
        atom
        for atom in atoms
        if any(row.query_id in selected_set for row in atom.queries)
    )
    closure_count = sum(len(atom.queries) for atom in touched)
    remaining = len(rows) - closure_count
    if len(touched) != OPT_BATCH_COUNT or closure_count != OPT_QUERY_COUNT or remaining:
        raise OptFoldError(
            "historical selection conflict audit drifted: expected closure=800/replay=0"
        )
    return LegacySelectionConflictAudit(
        selection_bytes_sha256=selection_bytes_sha256,
        selected_query_count=CREATOR_SELECTION_SIZE,
        touched_atomic_batch_count=len(touched),
        closure_query_count=closure_count,
        remaining_replay_count=remaining,
        incompatible_with_nonempty_replay=True,
    )


def _atom_counts(
    atom: _AtomicBatch,
) -> tuple[Counter[tuple[str, bool]], dict[str, set[str]], set[str]]:
    cells = Counter(
        (row.canonical_capability, row.is_boundary) for row in atom.queries
    )
    components_by_capability = {
        capability: {
            row.leakage_group_id
            for row in atom.queries
            if row.canonical_capability == capability
        }
        for capability in MVP_CAPABILITY_ORDER
    }
    components = {row.leakage_group_id for row in atom.queries}
    return cells, components_by_capability, components


def _solve_fold_by_atom(
    atoms: tuple[_AtomicBatch, ...], *, seed: int
) -> tuple[dict[str, FoldId], int]:
    ordered_atoms = tuple(
        sorted(atoms, key=lambda item: _seed_key(seed, "atomic-batch", item.atom_id))
    )
    cells = tuple(
        (capability, boundary)
        for capability in MVP_CAPABILITY_ORDER
        for boundary in (False, True)
    )
    atom_data = [_atom_counts(atom) for atom in ordered_atoms]
    global_cells = Counter(
        (row.canonical_capability, row.is_boundary)
        for atom in ordered_atoms
        for row in atom.queries
    )
    assignment_variable_count = len(ordered_atoms) * OPT_FOLD_COUNT
    deviation_variable_count = OPT_FOLD_COUNT * len(cells)
    variable_count = assignment_variable_count + deviation_variable_count
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)
    upper_bounds[assignment_variable_count:] = 4 * OPT_FOLD_SIZE
    integrality = np.ones(variable_count, dtype=np.int32)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def x(atom_index: int, fold_index: int) -> int:
        return atom_index * OPT_FOLD_COUNT + fold_index

    def d(fold_index: int, cell_index: int) -> int:
        return assignment_variable_count + fold_index * len(cells) + cell_index

    def add(coefficients: Mapping[int, float], lo: float, hi: float) -> None:
        row = np.zeros(variable_count, dtype=np.float64)
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for atom_index in range(len(ordered_atoms)):
        add({x(atom_index, fold): 1.0 for fold in range(OPT_FOLD_COUNT)}, 1, 1)
    for fold in range(OPT_FOLD_COUNT):
        add(
            {x(atom_index, fold): 1.0 for atom_index in range(len(ordered_atoms))},
            OPT_BATCHES_PER_FOLD,
            OPT_BATCHES_PER_FOLD,
        )
        add(
            {
                x(atom_index, fold): float(len(atom.queries))
                for atom_index, atom in enumerate(ordered_atoms)
            },
            OPT_FOLD_SIZE,
            OPT_FOLD_SIZE,
        )
        for capability in MVP_CAPABILITY_ORDER:
            add(
                {
                    x(atom_index, fold): float(
                        sum(
                            count
                            for (candidate_capability, _), count in atom_data[
                                atom_index
                            ][0].items()
                            if candidate_capability == capability
                        )
                    )
                    for atom_index in range(len(ordered_atoms))
                },
                1,
                OPT_FOLD_SIZE,
            )
    # Discovery is the complement of fold-00.  Components cannot occur in two
    # atoms because leakage_group_id is itself a GROUP_FIELDS edge.
    total_components = sum(len(data[2]) for data in atom_data)
    add(
        {
            x(atom_index, 0): float(len(data[2]))
            for atom_index, data in enumerate(atom_data)
        },
        0,
        total_components - CREATOR_SELECTION_SIZE,
    )
    for capability in MVP_CAPABILITY_ORDER:
        required = (
            CREATOR_V2_DOCUMENT_QUOTA
            if capability == DOCUMENT_CAPABILITY
            else CREATOR_INITIAL_CAPABILITY_TARGET
        )
        total = sum(len(data[1][capability]) for data in atom_data)
        add(
            {
                x(atom_index, 0): float(len(data[1][capability]))
                for atom_index, data in enumerate(atom_data)
            },
            0,
            total - required,
        )
    # Integer-scaled L1 distance: |4 * fold_cell_count - global_cell_count|.
    for fold in range(OPT_FOLD_COUNT):
        for cell_index, cell in enumerate(cells):
            coefficients = {
                x(atom_index, fold): float(4 * data[0][cell])
                for atom_index, data in enumerate(atom_data)
                if data[0][cell]
            }
            coefficients[d(fold, cell_index)] = -1.0
            add(coefficients, -np.inf, float(global_cells[cell]))
            coefficients = {
                x(atom_index, fold): float(-4 * data[0][cell])
                for atom_index, data in enumerate(atom_data)
                if data[0][cell]
            }
            coefficients[d(fold, cell_index)] = -1.0
            add(coefficients, -np.inf, float(-global_cells[cell]))

    constraints = LinearConstraint(
        np.vstack(rows),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    bounds = Bounds(lower_bounds, upper_bounds)

    def solve(
        objective: np.ndarray,
        *,
        extra_row: np.ndarray | None = None,
        extra_value: float | None = None,
    ) -> np.ndarray:
        effective = constraints
        if extra_row is not None:
            effective = LinearConstraint(
                np.vstack((constraints.A, extra_row)),
                np.append(constraints.lb, extra_value),
                np.append(constraints.ub, extra_value),
            )
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=effective,
            options={"presolve": True, "time_limit": 60.0},
        )
        if not result.success or result.x is None:
            raise OptFoldError(
                "opt fold MILP could not satisfy grouping, coverage, and capacity"
            )
        return np.rint(result.x).astype(np.int64)

    tie_objective = np.zeros(variable_count, dtype=np.float64)
    # A rank over atom/fold pairs is non-separable (unlike atom_index+fold), so
    # it actually resolves symmetric fold layouts while remaining numerically
    # modest and independent of caller input order.
    pair_order = sorted(
        (
            (atom_index, fold)
            for atom_index in range(len(ordered_atoms))
            for fold in range(OPT_FOLD_COUNT)
        ),
        key=lambda pair: _seed_key(
            seed,
            "fold-tie",
            f"{ordered_atoms[pair[0]].atom_id}/{OPT_FOLD_IDS[pair[1]]}",
        ),
    )
    for rank, (atom_index, fold) in enumerate(pair_order, start=1):
        tie_objective[x(atom_index, fold)] = float(rank)
    # Exactly 32 assignment variables are selected and every tie coefficient
    # is at most 128, so the complete tie objective is < 4096.  A 10,000
    # coefficient therefore implements an exact lexicographic priority for
    # each integer unit of L1 deviation in one MILP solve.
    tie_objective[assignment_variable_count:] = 10_000.0
    chosen = solve(tie_objective)
    minimum_deviation = int(chosen[assignment_variable_count:].sum())
    mapping: dict[str, FoldId] = {}
    for atom_index, atom in enumerate(ordered_atoms):
        selected_folds = [
            fold
            for fold in range(OPT_FOLD_COUNT)
            if chosen[x(atom_index, fold)] == 1
        ]
        if len(selected_folds) != 1:
            raise OptFoldError("opt fold solver returned a non-integral assignment")
        mapping[atom.atom_id] = OPT_FOLD_IDS[selected_folds[0]]
    return mapping, minimum_deviation


def _fold_assignments(
    atoms: tuple[_AtomicBatch, ...], mapping: Mapping[str, FoldId]
) -> tuple[OptFoldAssignment, ...]:
    assignments = []
    for atom in atoms:
        fold_id = mapping[atom.atom_id]
        role: FoldRole = "replay" if fold_id == REPLAY_FOLD else "discovery"
        assignments.extend(
            OptFoldAssignment(
                query_id=row.query_id,
                atomic_batch_id=atom.atom_id,
                fold_id=fold_id,
                role=role,
            )
            for row in atom.queries
        )
    return tuple(sorted(assignments, key=lambda item: item.query_id))


def _fold_audit(
    queries: tuple[OptFoldQuery, ...],
    assignments: tuple[OptFoldAssignment, ...],
) -> OptFoldAudit:
    query_by_id = {row.query_id: row for row in queries}
    fold_query_counts = Counter(row.fold_id for row in assignments)
    batches_by_fold: dict[str, set[str]] = defaultdict(set)
    capabilities: dict[str, Counter[str]] = defaultdict(Counter)
    boundaries: dict[str, Counter[str]] = defaultdict(Counter)
    discovery_rows: list[OptFoldQuery] = []
    for assignment in assignments:
        query = query_by_id[assignment.query_id]
        batches_by_fold[assignment.fold_id].add(assignment.atomic_batch_id)
        capabilities[assignment.fold_id][query.canonical_capability] += 1
        boundaries[assignment.fold_id][str(query.is_boundary).lower()] += 1
        if assignment.role == "discovery":
            discovery_rows.append(query)
    discovery_components = {row.leakage_group_id for row in discovery_rows}
    by_capability = {
        capability: len(
            {
                row.leakage_group_id
                for row in discovery_rows
                if row.canonical_capability == capability
            }
        )
        for capability in MVP_CAPABILITY_ORDER
    }
    global_cells = Counter(
        (row.canonical_capability, row.is_boundary) for row in queries
    )
    fold_cells: dict[str, Counter[tuple[str, bool]]] = defaultdict(Counter)
    for assignment in assignments:
        query = query_by_id[assignment.query_id]
        fold_cells[assignment.fold_id][
            (query.canonical_capability, query.is_boundary)
        ] += 1
    observed_deviation = sum(
        abs(4 * fold_cells[fold][cell] - global_cells[cell])
        for fold in OPT_FOLD_IDS
        for cell in global_cells
    )
    return OptFoldAudit(
        query_count=len(queries),
        atomic_batch_count=len({row.atomic_batch_id for row in assignments}),
        fold_query_counts={fold: fold_query_counts[fold] for fold in OPT_FOLD_IDS},
        fold_batch_counts={fold: len(batches_by_fold[fold]) for fold in OPT_FOLD_IDS},
        capability_counts_by_fold={
            fold: {
                capability: capabilities[fold][capability]
                for capability in MVP_CAPABILITY_ORDER
            }
            for fold in OPT_FOLD_IDS
        },
        boundary_counts_by_fold={
            fold: {
                "false": boundaries[fold]["false"],
                "true": boundaries[fold]["true"],
            }
            for fold in OPT_FOLD_IDS
        },
        discovery_unique_component_count=len(discovery_components),
        discovery_unique_components_by_capability=by_capability,
        group_field_cross_fold_violations=0,
        capability_boundary_l1_deviation=observed_deviation,
    )


def _assert_zero_group_leakage(
    queries: tuple[OptFoldQuery, ...],
    assignments: tuple[OptFoldAssignment, ...],
) -> None:
    fold_by_query = {row.query_id: row.fold_id for row in assignments}
    for field_name in GROUP_FIELDS:
        folds_by_value: dict[str, set[FoldId]] = defaultdict(set)
        for query in queries:
            value = getattr(query, field_name)
            if value is not None:
                folds_by_value[value].add(fold_by_query[query.query_id])
        if any(len(folds) != 1 for folds in folds_by_value.values()):
            raise OptFoldError(f"fold mapping splits GROUP_FIELDS field {field_name}")


def _validate_fold_plan_integrity(
    fold_plan: OptFoldPlan,
    queries: tuple[OptFoldQuery, ...],
    *,
    require_deterministic_optimum: bool = False,
) -> None:
    """Validate a fold packet without needing the historical index rows."""

    assignments = fold_plan.assignments
    query_by_id = {row.query_id: row for row in queries}
    assignment_ids = [row.query_id for row in assignments]
    if (
        len(assignments) != OPT_QUERY_COUNT
        or len(set(assignment_ids)) != OPT_QUERY_COUNT
        or set(assignment_ids) != set(query_by_id)
    ):
        raise OptFoldError("opt fold mapping must cover each opt800 query exactly once")
    if fold_plan.policy != OptFoldPolicy():
        raise OptFoldError("opt fold policy drifted from the frozen contract")
    manifest = fold_plan.manifest
    expected_bindings = {
        "policy_version": fold_plan.policy.policy_version,
        "policy_sha256": sha256_bytes(canonical_json_bytes(fold_plan.policy)),
        "opt_projection_sha256": sha256_bytes(
            canonical_opt_projection_bytes(queries)
        ),
        "mapping_sha256": sha256_bytes(
            canonical_opt_fold_mapping_bytes(assignments)
        ),
        "discovery_query_ids_sha256": _ids_sha256(
            row.query_id for row in assignments if row.role == "discovery"
        ),
        "replay_query_ids_sha256": _ids_sha256(
            row.query_id for row in assignments if row.role == "replay"
        ),
    }
    if any(
        getattr(manifest, field_name) != expected
        for field_name, expected in expected_bindings.items()
    ):
        raise OptFoldError("opt fold manifest binding mismatch")
    for assignment in assignments:
        if (
            assignment.atomic_batch_id
            != query_by_id[assignment.query_id].generator_batch_id
        ):
            raise OptFoldError("opt fold atomic_batch_id drifted from query metadata")
    _assert_zero_group_leakage(queries, assignments)
    observed_audit = _fold_audit(queries, assignments)
    if fold_plan.audit != observed_audit:
        raise OptFoldError("opt fold audit drifted from mapping")
    if require_deterministic_optimum:
        atoms = _atomic_batches(queries)
        optimum_mapping, optimum_deviation = _solve_fold_by_atom(
            atoms,
            seed=manifest.seed,
        )
        optimum_assignments = _fold_assignments(atoms, optimum_mapping)
        if assignments != optimum_assignments or (
            observed_audit.capability_boundary_l1_deviation != optimum_deviation
        ):
            raise OptFoldError("opt fold mapping drifted from deterministic optimum")


def build_opt_fold_plan(
    queries: Iterable[OptFoldQuery],
    *,
    legacy_selected_query_ids: Iterable[str],
    seed: int = 20260808,
    fold_count: int = OPT_FOLD_COUNT,
    source_queries_sha256: str = CORE_R3_QUERIES_SHA256,
    trusted_plan_sha256: str = CORE_R2_PLAN_SHA256,
    legacy_selection_bytes_sha256: str = LEGACY_CREATOR_SELECTION_SHA256,
) -> OptFoldPlan:
    """Build four deterministic group-aware 200-row folds over opt800."""

    if seed < 0:
        raise OptFoldError("opt fold seed must be non-negative")
    if fold_count != OPT_FOLD_COUNT:
        raise OptFoldError("opt fold count is frozen at four")
    _require_sha256(source_queries_sha256, "source queries SHA-256")
    _require_sha256(trusted_plan_sha256, "trusted plan SHA-256")
    rows = _materialize_queries(queries)
    atoms = _atomic_batches(rows)
    conflict = audit_legacy_selection_conflict(
        rows,
        legacy_selected_query_ids,
        selection_bytes_sha256=legacy_selection_bytes_sha256,
    )
    policy = OptFoldPolicy()
    mapping, deviation = _solve_fold_by_atom(atoms, seed=seed)
    assignments = _fold_assignments(atoms, mapping)
    _assert_zero_group_leakage(rows, assignments)
    audit = _fold_audit(rows, assignments)
    if audit.capability_boundary_l1_deviation != deviation:
        raise OptFoldError("opt fold balance objective drifted from MILP optimum")
    discovery_ids = [row.query_id for row in assignments if row.role == "discovery"]
    replay_ids = [row.query_id for row in assignments if row.role == "replay"]
    manifest = OptFoldManifest(
        seed=seed,
        source_queries_sha256=source_queries_sha256,
        plan_sha256=trusted_plan_sha256,
        opt_projection_sha256=sha256_bytes(canonical_opt_projection_bytes(rows)),
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        mapping_sha256=sha256_bytes(canonical_opt_fold_mapping_bytes(assignments)),
        discovery_query_ids_sha256=_ids_sha256(discovery_ids),
        replay_query_ids_sha256=_ids_sha256(replay_ids),
        legacy_conflict_audit=conflict,
    )
    plan = OptFoldPlan(
        assignments=assignments,
        manifest=manifest,
        audit=audit,
        policy=policy,
    )
    _validate_fold_plan_integrity(plan, rows)
    return plan


def validate_opt_fold_plan(
    fold_plan: OptFoldPlan,
    queries: Iterable[OptFoldQuery],
    *,
    legacy_selected_query_ids: Iterable[str],
    source_queries_sha256: str = CORE_R3_QUERIES_SHA256,
    trusted_plan_sha256: str = CORE_R2_PLAN_SHA256,
    legacy_selection_bytes_sha256: str = LEGACY_CREATOR_SELECTION_SHA256,
) -> None:
    """Fail closed on source, mapping, grouping, or deterministic drift."""

    expected = build_opt_fold_plan(
        queries,
        legacy_selected_query_ids=legacy_selected_query_ids,
        seed=fold_plan.manifest.seed,
        source_queries_sha256=source_queries_sha256,
        trusted_plan_sha256=trusted_plan_sha256,
        legacy_selection_bytes_sha256=legacy_selection_bytes_sha256,
    )
    if fold_plan != expected:
        raise OptFoldError("opt fold plan drifted from its deterministic optimum")


def compute_creator_v2_quotas(
    available_unique_components_by_capability: Mapping[str, int],
) -> dict[str, int]:
    """Allocate 240 slots while freezing Document at its discovery-safe 24."""

    if set(available_unique_components_by_capability) != set(MVP_CAPABILITY_ORDER):
        raise CreatorSelectionError(
            "available component map must cover MVP capability order"
        )
    available = {
        capability: available_unique_components_by_capability[capability]
        for capability in MVP_CAPABILITY_ORDER
    }
    if any(not isinstance(value, int) or value < 0 for value in available.values()):
        raise CreatorSelectionError("available component counts must be non-negative ints")
    if available[DOCUMENT_CAPABILITY] < CREATOR_V2_DOCUMENT_QUOTA:
        raise CreatorSelectionError("discovery cannot supply the fixed Document quota")
    quotas = {
        capability: (
            CREATOR_V2_DOCUMENT_QUOTA
            if capability == DOCUMENT_CAPABILITY
            else min(CREATOR_INITIAL_CAPABILITY_TARGET, available[capability])
        )
        for capability in MVP_CAPABILITY_ORDER
    }
    while sum(quotas.values()) < CREATOR_SELECTION_SIZE:
        eligible = [
            capability
            for capability in MVP_CAPABILITY_ORDER
            if capability != DOCUMENT_CAPABILITY
            and quotas[capability] < available[capability]
        ]
        if not eligible:
            raise CreatorSelectionError(
                "discovery component capacity cannot supply 240 Creator rows"
            )
        capability = min(
            eligible,
            key=lambda item: (
                Fraction(quotas[item], available[item]),
                MVP_CAPABILITY_ORDER.index(item),
            ),
        )
        quotas[capability] += 1
    if sum(quotas.values()) != CREATOR_SELECTION_SIZE:
        raise CreatorSelectionError("Creator v2 quota allocation overshot 240")
    return quotas


def _plan_projection(result: R2CoreInMemoryPlan) -> tuple[OptFoldQuery, ...]:
    return tuple(
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


def _published_plan_projection(
    context: PublishedR2SelectionContext,
) -> tuple[OptFoldQuery, ...]:
    return tuple(
        OptFoldQuery(
            query_id=row.plan_id,
            leakage_group_id=row.leakage_group_id,
            boundary_group_id=row.boundary_group_id,
            template_family=row.template_family,
            generator_batch_id=row.generator_batch_id,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
        )
        for row in context.plan.queries
        if context.final_split_by_plan_id[row.plan_id] == "opt_pool"
    )


def validate_published_r2_selection_context(
    context: PublishedR2SelectionContext,
) -> None:
    """Cross-check frozen publication evidence without loading assignments.

    The root publication already binds the historical bytes.  This validator
    independently proves all selection-relevant structure: plan identity,
    exact split matrices, GROUP_FIELDS isolation, reuse and realism sidecars,
    and the typed split-constraint packet.  It intentionally does not consult
    the mutable current TaskSpec registry.
    """

    _require_sha256(context.trusted_plan_sha256, "published r2 plan SHA-256")
    if sha256_bytes(canonical_json_bytes(context.plan)) != context.trusted_plan_sha256:
        raise CreatorSelectionError("published r2 plan SHA-256 mismatch")
    if context.plan.scope != "core" or len(context.plan.queries) != 1500:
        raise CreatorSelectionError("published r2 plan must contain core1500")
    plan_ids = {row.plan_id for row in context.plan.queries}
    if len(plan_ids) != len(context.plan.queries):
        raise CreatorSelectionError("published r2 plan IDs must be unique")
    if set(context.final_split_by_plan_id) != plan_ids:
        raise CreatorSelectionError("published final-split sidecar must cover core1500")
    if set(context.reuse_by_plan_id) != plan_ids:
        raise CreatorSelectionError("published reuse sidecar must cover core1500")
    if any(
        assignment.plan_id != plan_id
        for plan_id, assignment in context.reuse_by_plan_id.items()
    ):
        raise CreatorSelectionError("published reuse sidecar key/plan_id mismatch")

    try:
        verify_r2_split_constraint_binding(
            context.split_constraints,
            expected_plan_sha256=context.trusted_plan_sha256,
            expected_asset_catalog_sha256=context.plan.asset_catalog_sha256,
            expected_capability_assignments_sha256=(
                context.plan.capability_assignments_sha256
            ),
        )
    except (TypeError, ValueError) as error:
        raise CreatorSelectionError(
            "published split constraints do not bind the r2 plan"
        ) from error
    if context.split_constraints.plan_id_to_split != dict(
        context.final_split_by_plan_id
    ):
        raise CreatorSelectionError(
            "published split constraints disagree with final-split sidecar"
        )

    capability_counts: Counter[tuple[str, str]] = Counter()
    boundary_counts: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(context.plan.queries):
        split = context.final_split_by_plan_id[row.plan_id]
        if split not in CORE_R2_FINAL_SPLITS:
            raise CreatorSelectionError("published final-split value is unknown")
        if index < 200:
            if split != "dev_mini" or row.provisional_split != "dev_mini":
                raise CreatorSelectionError("published r2 dev prefix split drifted")
        elif split == "dev_mini" or row.provisional_split != "opt_pool":
            raise CreatorSelectionError("published r2 tail split drifted")
        capability_counts[(split, row.canonical_capability)] += 1
        if row.is_boundary:
            boundary_counts[(split, row.canonical_capability)] += 1
    actual_capabilities = {
        split: {
            capability: capability_counts[(split, capability)]
            for capability in MVP_CAPABILITY_ORDER
        }
        for split in CORE_R2_FINAL_SPLITS
    }
    actual_boundaries = {
        split: {
            capability: boundary_counts[(split, capability)]
            for capability in MVP_CAPABILITY_ORDER
        }
        for split in CORE_R2_FINAL_SPLITS
    }
    if actual_capabilities != CORE_R2_CAPABILITY_COUNTS:
        raise CreatorSelectionError("published capability x split matrix drifted")
    if actual_boundaries != CORE_R2_CAPABILITY_BOUNDARY_COUNTS:
        raise CreatorSelectionError("published boundary x split matrix drifted")

    for field_name in GROUP_FIELDS:
        split_by_value: dict[str, str] = {}
        for row in context.plan.queries:
            value = getattr(row, field_name)
            if value is None:
                continue
            split = context.final_split_by_plan_id[row.plan_id]
            previous = split_by_value.setdefault(value, split)
            if previous != split:
                raise CreatorSelectionError(
                    f"published r2 plan splits GROUP_FIELDS field {field_name}"
                )

    realism = context.realism_sidecar
    expected_manifest = {
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.asset_catalog_sha256,
        "capability_assignments_sha256": (
            context.plan.capability_assignments_sha256
        ),
    }
    if any(
        getattr(realism.manifest, field_name) != expected
        for field_name, expected in expected_manifest.items()
    ):
        raise CreatorSelectionError("published realism manifest binding drifted")
    try:
        validate_realism_sidecar(
            realism,
            plan_rows=context.plan.queries,
            final_split_by_plan_id=context.final_split_by_plan_id,
            reuse_by_plan_id=context.reuse_by_plan_id,
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as error:
        raise CreatorSelectionError("published realism sidecar is invalid") from error


def _published_candidates(
    context: PublishedR2SelectionContext,
) -> tuple[_Candidate, ...]:
    validate_published_r2_selection_context(context)
    realism_by_id = {
        assignment.plan_id: assignment
        for assignment in context.realism_sidecar.assignments
    }
    candidates = tuple(
        _Candidate(
            plan_id=row.plan_id,
            component_id=row.leakage_group_id,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            interaction_pattern=realism_by_id[row.plan_id].interaction_pattern,
        )
        for row in context.plan.queries
        if context.final_split_by_plan_id[row.plan_id] == "opt_pool"
    )
    if len(candidates) != OPT_QUERY_COUNT:
        raise CreatorSelectionError("published selection context must expose opt800")
    if any(
        not candidate.component_id
        or candidate.component_id != candidate.component_id.strip()
        or candidate.canonical_capability not in MVP_CAPABILITY_ORDER
        for candidate in candidates
    ):
        raise CreatorSelectionError("published opt candidate metadata is invalid")
    return candidates


def _creator_v2_manifest(
    *,
    policy: CreatorSelectionV2Policy,
    seed: int,
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    trusted_plan_sha256: str,
    fold_plan: OptFoldPlan,
    entries: tuple[CreatorSelectionEntry, ...],
) -> CreatorSelectionV2Manifest:
    conflict = fold_plan.manifest.legacy_conflict_audit
    return CreatorSelectionV2Manifest(
        seed=seed,
        selection_size=CREATOR_SELECTION_SIZE,
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_queries_sha256=fold_plan.manifest.source_queries_sha256,
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
        final_split_sidecar_sha256=(
            realism_sidecar.manifest.final_split_sidecar_sha256
        ),
        reuse_sidecar_sha256=realism_sidecar.manifest.reuse_sidecar_sha256,
        fold_mapping_sha256=fold_plan.manifest.mapping_sha256,
        discovery_query_ids_sha256=fold_plan.manifest.discovery_query_ids_sha256,
        replay_query_ids_sha256=fold_plan.manifest.replay_query_ids_sha256,
        supersedes_selection_bytes_sha256=conflict.selection_bytes_sha256,
        legacy_closure_query_count=conflict.closure_query_count,
        legacy_remaining_replay_count=conflict.remaining_replay_count,
        selection_bytes_sha256=sha256_bytes(
            canonical_creator_selection_index_bytes(entries)
        ),
    )


def build_opt_creator_selection_v2(
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    fold_plan: OptFoldPlan,
    *,
    trusted_plan_sha256: str = CORE_R2_PLAN_SHA256,
    seed: int = 20260808,
) -> CoreR2CreatorSelectionV2:
    """Select 240 component-unique Creator inputs from discovery600 only."""

    if seed < 0:
        raise CreatorSelectionError("Creator v2 seed must be non-negative")
    projections = _plan_projection(result)
    if fold_plan.manifest.plan_sha256 != trusted_plan_sha256:
        raise CreatorSelectionError("Creator v2 fold plan binding mismatches plan")
    try:
        _validate_fold_plan_integrity(
            fold_plan,
            projections,
            require_deterministic_optimum=True,
        )
    except OptFoldError as exc:
        raise CreatorSelectionError("Creator v2 received an invalid fold plan") from exc
    assignment_by_id = {row.query_id: row for row in fold_plan.assignments}
    if set(assignment_by_id) != {row.query_id for row in projections}:
        raise CreatorSelectionError("Creator v2 fold mapping does not cover opt800")
    candidates = _validate_inputs(
        result,
        realism_sidecar,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    discovery_ids = {
        query_id
        for query_id, assignment in assignment_by_id.items()
        if assignment.role == "discovery"
    }
    if len(discovery_ids) != 600:
        raise CreatorSelectionError("Creator v2 requires exactly 600 discovery rows")
    eligible = tuple(
        candidate for candidate in candidates if candidate.plan_id in discovery_ids
    )
    available = _available_component_counts(eligible)
    quotas = compute_creator_v2_quotas(available)
    selected, target_cells, selected_cells, deviation = (
        _solve_component_unique_selection(eligible, quotas, seed=seed)
    )
    entries = _selection_entries(selected, seed=seed)
    audit = _selection_audit(
        eligible,
        quotas,
        selected,
        target_cells,
        selected_cells,
        deviation,
    )
    policy = CreatorSelectionV2Policy()
    return CoreR2CreatorSelectionV2(
        entries=entries,
        manifest=_creator_v2_manifest(
            policy=policy,
            seed=seed,
            result=result,
            realism_sidecar=realism_sidecar,
            trusted_plan_sha256=trusted_plan_sha256,
            fold_plan=fold_plan,
            entries=entries,
        ),
        audit=audit,
        policy=policy,
    )


def validate_opt_creator_selection_v2(
    selection: CoreR2CreatorSelectionV2,
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    fold_plan: OptFoldPlan,
    *,
    trusted_plan_sha256: str = CORE_R2_PLAN_SHA256,
) -> None:
    """Fail closed on discovery membership, bindings, or deterministic drift."""

    expected = build_opt_creator_selection_v2(
        result,
        realism_sidecar,
        fold_plan,
        trusted_plan_sha256=trusted_plan_sha256,
        seed=selection.manifest.seed,
    )
    if selection != expected:
        raise CreatorSelectionError(
            "Creator v2 selection drifted from its deterministic optimum"
        )


def build_opt_creator_selection_v2_from_publication(
    context: PublishedR2SelectionContext,
    fold_plan: OptFoldPlan,
    *,
    seed: int = 20260808,
) -> CoreR2CreatorSelectionV2:
    """Build Creator v2 from an immutable publication, not mutable registries."""

    if seed < 0:
        raise CreatorSelectionError("Creator v2 seed must be non-negative")
    projections = _published_plan_projection(context)
    if fold_plan.manifest.plan_sha256 != context.trusted_plan_sha256:
        raise CreatorSelectionError("Creator v2 fold plan binding mismatches plan")
    try:
        _validate_fold_plan_integrity(
            fold_plan,
            projections,
            require_deterministic_optimum=True,
        )
    except OptFoldError as error:
        raise CreatorSelectionError(
            "Creator v2 received an invalid published fold plan"
        ) from error
    assignment_by_id = {row.query_id: row for row in fold_plan.assignments}
    if set(assignment_by_id) != {row.query_id for row in projections}:
        raise CreatorSelectionError("Creator v2 fold mapping does not cover opt800")
    candidates = _published_candidates(context)
    discovery_ids = {
        query_id
        for query_id, assignment in assignment_by_id.items()
        if assignment.role == "discovery"
    }
    if len(discovery_ids) != 600:
        raise CreatorSelectionError("Creator v2 requires exactly 600 discovery rows")
    eligible = tuple(
        candidate for candidate in candidates if candidate.plan_id in discovery_ids
    )
    available = _available_component_counts(eligible)
    quotas = compute_creator_v2_quotas(available)
    selected, target_cells, selected_cells, deviation = (
        _solve_component_unique_selection(eligible, quotas, seed=seed)
    )
    entries = _selection_entries(selected, seed=seed)
    audit = _selection_audit(
        eligible,
        quotas,
        selected,
        target_cells,
        selected_cells,
        deviation,
    )
    policy = CreatorSelectionV2Policy()
    realism = context.realism_sidecar
    conflict = fold_plan.manifest.legacy_conflict_audit
    manifest = CreatorSelectionV2Manifest(
        seed=seed,
        selection_size=CREATOR_SELECTION_SIZE,
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_queries_sha256=fold_plan.manifest.source_queries_sha256,
        plan_sha256=context.trusted_plan_sha256,
        asset_catalog_sha256=context.plan.asset_catalog_sha256,
        capability_assignments_sha256=(
            context.plan.capability_assignments_sha256
        ),
        realism_manifest_sha256=sha256_bytes(
            canonical_json_bytes(realism.manifest)
        ),
        realism_assignments_sha256=realism.manifest.assignments_sha256,
        prompt_recipe_manifest_sha256=(
            realism.manifest.prompt_recipe_manifest_sha256
        ),
        final_split_sidecar_sha256=(
            realism.manifest.final_split_sidecar_sha256
        ),
        reuse_sidecar_sha256=realism.manifest.reuse_sidecar_sha256,
        fold_mapping_sha256=fold_plan.manifest.mapping_sha256,
        discovery_query_ids_sha256=fold_plan.manifest.discovery_query_ids_sha256,
        replay_query_ids_sha256=fold_plan.manifest.replay_query_ids_sha256,
        supersedes_selection_bytes_sha256=conflict.selection_bytes_sha256,
        legacy_closure_query_count=conflict.closure_query_count,
        legacy_remaining_replay_count=conflict.remaining_replay_count,
        selection_bytes_sha256=sha256_bytes(
            canonical_creator_selection_index_bytes(entries)
        ),
    )
    return CoreR2CreatorSelectionV2(
        entries=entries,
        manifest=manifest,
        audit=audit,
        policy=policy,
    )


def validate_opt_creator_selection_v2_from_publication(
    selection: CoreR2CreatorSelectionV2,
    context: PublishedR2SelectionContext,
    fold_plan: OptFoldPlan,
) -> None:
    expected = build_opt_creator_selection_v2_from_publication(
        context,
        fold_plan,
        seed=selection.manifest.seed,
    )
    if selection != expected:
        raise CreatorSelectionError(
            "published Creator v2 selection drifted from deterministic optimum"
        )


def creator_selection_v2_index_bytes(
    selection: CoreR2CreatorSelectionV2,
) -> bytes:
    return canonical_creator_selection_index_bytes(selection.entries)
