"""Pure in-memory, deterministic r2 owner-audit sampling.

The r2 corpus is intentionally too large for a record-by-record review before
use.  This module selects a fixed 200-row audit sample from frozen plan and
realism metadata.  It never opens assets, emits author packets, composes text,
or writes files.  An orchestration layer may later publish the returned
canonical JSONL index create-only.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from skillchain.schemas import Intent
from skillchain.synthesis.portfolio_core_authoring import (
    FinalSplit,
    RealismAssignment,
    RealismSidecar,
    canonical_final_split_sidecar_bytes,
    build_prompt_recipe_manifest,
    canonical_realism_assignments_bytes,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AUDIT_SAMPLE_SIZE = 200
AUDIT_POLICY_VERSION = "portfolio-core-r2-cluster-audit-v1"
_FINAL_SPLIT_ORDER: tuple[FinalSplit, ...] = (
    "dev_mini",
    "opt_pool",
    "val",
    "test_frozen",
)
_COUNTERFACTUAL_STRATEGIES = {"cross_intent_triplet", "cross_intent_pair"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must be non-blank")
    return value


class AuditCoverageError(ValueError):
    """Raised before sampling when a mandatory coverage contract is infeasible."""


class AuditPlanRow(_StrictModel):
    """Minimal internal plan projection required for audit selection only."""

    plan_id: str
    generator_batch_id: str
    canonical_intent: Intent
    canonical_capability: str
    is_boundary: bool
    boundary_strategy: str | None = None
    asset_id: str | None = None
    component_id: str

    @field_validator(
        "plan_id",
        "generator_batch_id",
        "canonical_capability",
        "boundary_strategy",
        "asset_id",
        "component_id",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)


class AuditSamplingPolicy(_StrictModel):
    """Fixed defaults for the owner-facing r2 200-row review sample."""

    policy_version: Literal["portfolio-core-r2-cluster-audit-v1"] = (
        AUDIT_POLICY_VERSION
    )
    sample_size: Literal[200] = AUDIT_SAMPLE_SIZE
    generator_batch_minimum: int = Field(default=1, ge=1)
    capability_boundary_shape_minimum: int = Field(default=2, ge=1)
    document_boundary_minimum: int = Field(default=4, ge=1)
    counterfactual_minimum: int = Field(default=12, ge=1)
    s3_minimum: int = Field(default=15, ge=1)
    no_result_minimum: int = Field(default=12, ge=1)
    goal_change_minimum: int = Field(default=12, ge=1)


class AuditSelectionEntry(_StrictModel):
    """One internal audit-index row.  It never carries a path or query text."""

    schema_version: Literal[1] = 1
    selection_rank: int = Field(ge=1)
    plan_id: str
    generator_batch_id: str
    component_id: str
    final_split: FinalSplit
    canonical_intent: Intent
    canonical_capability: str
    is_boundary: bool
    boundary_strategy: str | None = None
    trajectory_shape: Literal["single_turn", "three_turn"]
    interaction_pattern: str
    expression_level: str
    user_style: str
    constraint_level: str
    ambiguity_subtype: str
    prompt_recipe_id: str
    prompt_recipe_sha256: Sha256
    is_counterfactual: bool
    source_id: str | None = None
    selection_reason: tuple[str, ...]

    @field_validator(
        "plan_id",
        "generator_batch_id",
        "component_id",
        "canonical_capability",
        "boundary_strategy",
        "interaction_pattern",
        "expression_level",
        "user_style",
        "constraint_level",
        "ambiguity_subtype",
        "prompt_recipe_id",
        "source_id",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("selection_reason")
    @classmethod
    def _validate_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_nonblank(item, "selection_reason") for item in value)
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ValueError("selection_reason must be non-empty and unique")
        return tuple(sorted(cleaned))


class AuditSelectionManifest(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-r2-cluster-audit-v1"] = (
        AUDIT_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    sample_size: Literal[200] = AUDIT_SAMPLE_SIZE
    policy_sha256: Sha256
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    prompt_recipe_manifest_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    reuse_sidecar_sha256: Sha256
    source_sidecar_sha256: Sha256 | None = None
    selection_index_sha256: Sha256


class AuditSampleAudit(_StrictModel):
    sample_size: Literal[200] = AUDIT_SAMPLE_SIZE
    mandatory_selected_count: int = Field(ge=0, le=AUDIT_SAMPLE_SIZE)
    generator_batch_count: int = Field(ge=0)
    capability_boundary_shape_counts: dict[str, int]
    split_intent_capability_boundary_targets: dict[str, int]
    split_intent_capability_boundary_counts: dict[str, int]
    interaction_counts: dict[str, int]
    expression_counts: dict[str, int]
    user_style_counts: dict[str, int]
    ambiguity_counts: dict[str, int]
    recipe_counts: dict[str, int]
    rare_counts: dict[str, int]
    source_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class CoreR2AuditSample:
    entries: tuple[AuditSelectionEntry, ...]
    manifest: AuditSelectionManifest
    audit: AuditSampleAudit
    policy: AuditSamplingPolicy


@dataclass(frozen=True)
class _Requirement:
    label: str
    candidates: frozenset[str]
    minimum: int


def _payload(value: object) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"unsupported plan-like row: {type(value)!r}")


def coerce_audit_plan_rows(rows_or_plan: object) -> tuple[AuditPlanRow, ...]:
    """Coerce ``CorpusPlan.queries``-like objects without importing the planner."""

    source: object = getattr(rows_or_plan, "queries", rows_or_plan)
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        raise TypeError("audit input must be a plan-like sequence or object with queries")
    result: list[AuditPlanRow] = []
    for raw in source:
        data = _payload(raw)
        plan_id = data.get("plan_id")
        generator_batch_id = data.get("generator_batch_id", data.get("batch_id"))
        component_id = data.get(
            "component_id",
            data.get("leakage_group_id", data.get("asset_id", f"component:{plan_id}")),
        )
        result.append(
            AuditPlanRow(
                plan_id=plan_id,
                generator_batch_id=generator_batch_id,
                canonical_intent=data.get("canonical_intent"),
                canonical_capability=data.get("canonical_capability"),
                is_boundary=data.get("is_boundary"),
                boundary_strategy=data.get("boundary_strategy"),
                asset_id=data.get("asset_id"),
                component_id=component_id,
            )
        )
    if len(result) != 1500:
        raise ValueError("core r2 audit sampler requires exactly 1,500 plan rows")
    plan_ids = [row.plan_id for row in result]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("audit plan rows must have unique plan_id values")
    if len({row.generator_batch_id for row in result}) != 60:
        raise ValueError("core r2 audit sampler requires exactly 60 generator batches")
    return tuple(result)


def _normalize_final_splits(
    rows: Sequence[AuditPlanRow], sidecar: Mapping[str, object] | Sequence[object]
) -> dict[str, FinalSplit]:
    result: dict[str, FinalSplit] = {}
    if isinstance(sidecar, Mapping):
        for plan_id, raw in sidecar.items():
            if isinstance(raw, Mapping):
                raw_plan_id = raw.get("plan_id", plan_id)
                raw_split = raw.get("final_split")
                if raw_plan_id != plan_id:
                    raise ValueError("final split mapping key does not match plan_id")
            elif isinstance(raw, BaseModel):
                raw_plan_id = getattr(raw, "plan_id")
                raw_split = getattr(raw, "final_split")
                if raw_plan_id != plan_id:
                    raise ValueError("final split mapping key does not match plan_id")
            else:
                raw_split = raw
            result[_nonblank(str(plan_id), "plan_id")] = raw_split  # type: ignore[assignment]
    else:
        for raw in sidecar:
            data = _payload(raw)
            plan_id = _nonblank(str(data["plan_id"]), "plan_id")
            if plan_id in result:
                raise ValueError("final split sidecar must not repeat plan_id")
            result[plan_id] = data["final_split"]  # type: ignore[assignment]
    valid = set(_FINAL_SPLIT_ORDER)
    if any(value not in valid for value in result.values()):
        raise ValueError("final split sidecar contains an invalid split")
    _require_exact_coverage(rows, result, "final split sidecar")
    return result


def _require_exact_coverage(
    rows: Sequence[AuditPlanRow], values: Mapping[str, object], label: str
) -> None:
    expected = {row.plan_id for row in rows}
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{label} must exactly cover plan IDs; missing={sorted(expected - actual)[:3]}, "
            f"extra={sorted(actual - expected)[:3]}"
        )


def _normalise_sources(
    rows: Sequence[AuditPlanRow], source_by_asset_id: Mapping[str, str] | None
) -> tuple[dict[str, str], str | None]:
    if source_by_asset_id is None:
        return {}, None
    known_assets = {row.asset_id for row in rows if row.asset_id is not None}
    missing = known_assets - set(source_by_asset_id)
    if missing:
        raise ValueError(
            f"source sidecar is missing plan asset IDs: {sorted(missing)[:3]}"
        )
    normalized: dict[str, str] = {}
    # A catalog-wide mapping is the normal caller input.  Bind only the assets
    # referenced by this plan so unrelated catalog growth cannot perturb the
    # audit sample identity.
    for asset_id in sorted(known_assets):
        source_id = source_by_asset_id[asset_id]
        if not isinstance(source_id, str):
            raise ValueError("source sidecar values must be strings")
        if not source_id.strip():
            raise ValueError("source sidecar values must be non-blank")
        normalized[_nonblank(asset_id, "asset_id")] = _nonblank(source_id, "source_id")
    source_bytes = canonical_jsonl_bytes(
        {"asset_id": asset_id, "source_id": normalized[asset_id]}
        for asset_id in sorted(normalized)
    )
    return normalized, sha256_bytes(source_bytes)


def _realism_by_id(
    sidecar: RealismSidecar,
    rows: Sequence[AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
) -> dict[str, RealismAssignment]:
    assignments = {assignment.plan_id: assignment for assignment in sidecar.assignments}
    if len(assignments) != len(sidecar.assignments):
        raise ValueError("realism sidecar contains duplicate plan_id values")
    _require_exact_coverage(rows, assignments, "realism sidecar")
    if sidecar.manifest.row_count != 1500 or sidecar.manifest.assignments_sha256 != sha256_bytes(
        canonical_realism_assignments_bytes(sidecar.assignments)
    ):
        raise ValueError("realism sidecar manifest does not bind its assignments")
    if sidecar.manifest.final_split_sidecar_sha256 != sha256_bytes(
        canonical_final_split_sidecar_bytes(final_splits)
    ):
        raise ValueError("realism sidecar final split sidecar digest mismatch")
    recipe_manifest = build_prompt_recipe_manifest(sidecar.recipes)
    if sidecar.recipe_manifest != recipe_manifest:
        raise ValueError("realism sidecar prompt recipe manifest is invalid")
    if sidecar.manifest.prompt_recipe_manifest_sha256 != sha256_bytes(
        canonical_json_bytes(recipe_manifest)
    ):
        raise ValueError("realism sidecar recipe digest mismatch")
    if len(sidecar.recipes) != 12:
        raise ValueError("core r2 audit sampler requires exactly 12 prompt recipes")
    recipe_ids = {recipe.prompt_recipe_id for recipe in sidecar.recipes}
    used_recipe_ids = {assignment.prompt_recipe_id for assignment in sidecar.assignments}
    if recipe_ids != used_recipe_ids:
        raise ValueError("each of the 12 prompt recipes must be non-empty before audit")
    return assignments


def _stable_rank(seed: int, namespace: str, plan_id: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{seed}\x00{namespace}\x00{plan_id}".encode("utf-8")).hexdigest(),
        plan_id,
    )


def _largest_remainder(
    weights: Mapping[str, int], total: int, *, seed: int, namespace: str
) -> dict[str, int]:
    if total < 0 or any(weight < 0 for weight in weights.values()):
        raise ValueError("largest-remainder inputs must be non-negative")
    capacity = sum(weights.values())
    if total > capacity:
        raise ValueError("largest-remainder target exceeds population")
    if total == 0:
        return {key: 0 for key in weights}
    if capacity == 0:
        raise ValueError("cannot allocate a positive target to an empty population")
    result = {key: (weight * total) // capacity for key, weight in weights.items()}
    remainder = total - sum(result.values())
    ranking = sorted(
        weights,
        key=lambda key: (
            -((weights[key] * total) % capacity),
            _stable_rank(seed, f"{namespace}:quota", key),
        ),
    )
    for key in ranking[:remainder]:
        result[key] += 1
    return result


def _cell_key(row: AuditPlanRow, realism: RealismAssignment) -> str:
    return (
        f"{row.canonical_capability}|{'boundary' if row.is_boundary else 'non_boundary'}"
        f"|{realism.trajectory_shape}"
    )


def _fill_stratum(row: AuditPlanRow, split: FinalSplit) -> str:
    return (
        f"{split}|{row.canonical_intent}|{row.canonical_capability}|"
        f"{'boundary' if row.is_boundary else 'non_boundary'}"
    )


def _build_requirements(
    rows: Sequence[AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
    realism: Mapping[str, RealismAssignment],
    sources: Mapping[str, str],
    policy: AuditSamplingPolicy,
) -> tuple[_Requirement, ...]:
    groups: dict[str, set[str]] = defaultdict(set)

    def add(label: str, predicate) -> None:
        groups[label].update(row.plan_id for row in rows if predicate(row, realism[row.plan_id]))

    for batch_id in sorted({row.generator_batch_id for row in rows}):
        add(f"mandatory:generator_batch:{batch_id}", lambda row, _: row.generator_batch_id == batch_id)
    for key in sorted({_cell_key(row, realism[row.plan_id]) for row in rows}):
        add(
            f"mandatory:capability_boundary_shape:{key}",
            lambda row, assignment, key=key: _cell_key(row, assignment) == key,
        )
    for field, prefix in (
        ("interaction_pattern", "interaction"),
        ("expression_level", "expression"),
        ("user_style", "user_style"),
        ("ambiguity_subtype", "ambiguity"),
        ("prompt_recipe_id", "prompt_recipe"),
    ):
        values = sorted({getattr(assignment, field) for assignment in realism.values()})
        for value in values:
            add(
                f"mandatory:{prefix}:{value}",
                lambda row, assignment, field=field, value=value: getattr(assignment, field) == value,
            )
    for source_id in sorted(set(sources.values())):
        add(
            f"mandatory:source:{source_id}",
            lambda row, _, source_id=source_id: row.asset_id is not None
            and sources.get(row.asset_id) == source_id,
        )

    add(
        "mandatory:rare:document_boundary",
        lambda row, _: row.is_boundary and row.canonical_capability == "utility.document_reading",
    )
    add(
        "mandatory:rare:counterfactual",
        lambda row, _: row.boundary_strategy in _COUNTERFACTUAL_STRATEGIES,
    )
    add(
        "mandatory:rare:s3",
        lambda _, assignment: assignment.expression_level == "S3",
    )
    add(
        "mandatory:rare:no_result_relaxation",
        lambda _, assignment: assignment.interaction_pattern == "no_result_relaxation",
    )
    add(
        "mandatory:rare:goal_change_or_multi_query",
        lambda _, assignment: assignment.interaction_pattern
        == "goal_change_or_multi_query",
    )

    requirements: list[_Requirement] = []
    for label, candidates in sorted(groups.items()):
        if label.startswith("mandatory:generator_batch:"):
            minimum = policy.generator_batch_minimum
        elif label.startswith("mandatory:capability_boundary_shape:"):
            minimum = policy.capability_boundary_shape_minimum
        elif label == "mandatory:rare:document_boundary":
            minimum = policy.document_boundary_minimum
        elif label == "mandatory:rare:counterfactual":
            minimum = policy.counterfactual_minimum
        elif label == "mandatory:rare:s3":
            minimum = policy.s3_minimum
        elif label == "mandatory:rare:no_result_relaxation":
            minimum = policy.no_result_minimum
        elif label == "mandatory:rare:goal_change_or_multi_query":
            minimum = policy.goal_change_minimum
        else:
            minimum = 1
        if len(candidates) < minimum:
            raise AuditCoverageError(
                f"mandatory coverage is infeasible for {label}: need {minimum}, have {len(candidates)}"
            )
        requirements.append(
            _Requirement(label=label, candidates=frozenset(candidates), minimum=minimum)
        )
    return tuple(requirements)


def _select_mandatory(
    requirements: Sequence[_Requirement], *, seed: int, sample_size: int
) -> tuple[list[str], dict[str, set[str]]]:
    selected: list[str] = []
    selected_set: set[str] = set()
    reasons: dict[str, set[str]] = defaultdict(set)
    remaining = {requirement.label: requirement.minimum for requirement in requirements}
    reverse: dict[str, list[_Requirement]] = defaultdict(list)
    for requirement in requirements:
        for plan_id in requirement.candidates:
            reverse[plan_id].append(requirement)

    while any(value > 0 for value in remaining.values()):
        eligible = [
            plan_id
            for plan_id, linked in reverse.items()
            if plan_id not in selected_set
            and any(remaining[requirement.label] > 0 for requirement in linked)
        ]
        if not eligible:
            unsatisfied = sorted(label for label, value in remaining.items() if value > 0)
            raise AuditCoverageError(f"mandatory coverage cannot be completed: {unsatisfied[:3]}")

        def score(plan_id: str) -> tuple[int, int, tuple[str, str]]:
            linked = [
                requirement
                for requirement in reverse[plan_id]
                if remaining[requirement.label] > 0
            ]
            rarity = sum(
                (remaining[requirement.label] * 1_000_000)
                // max(1, len(requirement.candidates - selected_set))
                for requirement in linked
            )
            return rarity, len(linked), _stable_rank(seed, "mandatory", plan_id)

        chosen = min(
            eligible,
            key=lambda plan_id: (-score(plan_id)[0], -score(plan_id)[1], score(plan_id)[2]),
        )
        selected.append(chosen)
        selected_set.add(chosen)
        for requirement in reverse[chosen]:
            if remaining[requirement.label] > 0:
                remaining[requirement.label] -= 1
                reasons[chosen].add(requirement.label)
        if len(selected) > sample_size:
            raise AuditCoverageError(
                f"mandatory coverage requires more than {sample_size} samples"
            )
    return selected, reasons


def _fill_largest_deficit(
    rows: Sequence[AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
    selected: list[str],
    reasons: dict[str, set[str]],
    *,
    seed: int,
    sample_size: int,
) -> None:
    strata_by_id = {
        row.plan_id: _fill_stratum(row, final_splits[row.plan_id]) for row in rows
    }
    population = Counter(strata_by_id.values())
    targets = _largest_remainder(
        dict(population), sample_size, seed=seed, namespace="fill-strata"
    )
    current = Counter(strata_by_id[plan_id] for plan_id in selected)
    selected_set = set(selected)
    candidates_by_stratum: dict[str, list[str]] = defaultdict(list)
    for plan_id, stratum in strata_by_id.items():
        if plan_id not in selected_set:
            candidates_by_stratum[stratum].append(plan_id)

    while len(selected) < sample_size:
        deficits = {
            stratum: targets[stratum] - current[stratum]
            for stratum in targets
            if candidates_by_stratum[stratum]
        }
        max_deficit = max(deficits.values(), default=0)
        if max_deficit <= 0:
            raise AuditCoverageError("largest-deficit fill exhausted before reaching sample size")
        candidate_strata = [
            stratum for stratum, deficit in deficits.items() if deficit == max_deficit
        ]
        chosen_stratum = min(
            candidate_strata,
            key=lambda value: _stable_rank(seed, "fill-stratum", value),
        )
        chosen = min(
            candidates_by_stratum[chosen_stratum],
            key=lambda plan_id: _stable_rank(seed, "fill-row", plan_id),
        )
        candidates_by_stratum[chosen_stratum].remove(chosen)
        selected.append(chosen)
        selected_set.add(chosen)
        current[chosen_stratum] += 1
        reasons[chosen].add("fill:split_intent_capability_boundary_largest_deficit")


def _build_entries(
    selected: Sequence[str],
    reasons: Mapping[str, set[str]],
    rows_by_id: Mapping[str, AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
    realism: Mapping[str, RealismAssignment],
    sources: Mapping[str, str],
) -> tuple[AuditSelectionEntry, ...]:
    entries: list[AuditSelectionEntry] = []
    for rank, plan_id in enumerate(selected, start=1):
        row = rows_by_id[plan_id]
        assignment = realism[plan_id]
        source_id = sources.get(row.asset_id) if row.asset_id is not None else None
        entries.append(
            AuditSelectionEntry(
                selection_rank=rank,
                plan_id=plan_id,
                generator_batch_id=row.generator_batch_id,
                component_id=row.component_id,
                final_split=final_splits[plan_id],
                canonical_intent=row.canonical_intent,
                canonical_capability=row.canonical_capability,
                is_boundary=row.is_boundary,
                boundary_strategy=row.boundary_strategy,
                trajectory_shape=assignment.trajectory_shape,
                interaction_pattern=assignment.interaction_pattern,
                expression_level=assignment.expression_level,
                user_style=assignment.user_style,
                constraint_level=assignment.constraint_level,
                ambiguity_subtype=assignment.ambiguity_subtype,
                prompt_recipe_id=assignment.prompt_recipe_id,
                prompt_recipe_sha256=assignment.prompt_recipe_sha256,
                is_counterfactual=row.boundary_strategy in _COUNTERFACTUAL_STRATEGIES,
                source_id=source_id,
                selection_reason=tuple(reasons[plan_id]),
            )
        )
    return tuple(entries)


def canonical_audit_selection_index_bytes(
    entries: Iterable[AuditSelectionEntry],
) -> bytes:
    materialized = tuple(entries)
    if [entry.selection_rank for entry in materialized] != list(
        range(1, len(materialized) + 1)
    ):
        raise ValueError("audit index ranks must be contiguous and ordered")
    if len({entry.plan_id for entry in materialized}) != len(materialized):
        raise ValueError("audit index plan IDs must be unique")
    return canonical_jsonl_bytes(materialized)


def _build_audit(
    entries: Sequence[AuditSelectionEntry],
    rows: Sequence[AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
    mandatory_selected_count: int,
    sources_present: bool,
    *,
    seed: int,
) -> AuditSampleAudit:
    population = Counter(_fill_stratum(row, final_splits[row.plan_id]) for row in rows)
    targets = _largest_remainder(
        dict(population), AUDIT_SAMPLE_SIZE, seed=seed, namespace="fill-strata"
    )
    cell_counts = Counter(
        f"{entry.canonical_capability}|{'boundary' if entry.is_boundary else 'non_boundary'}"
        f"|{entry.trajectory_shape}"
        for entry in entries
    )
    sampled_strata = Counter(
        f"{entry.final_split}|{entry.canonical_intent}|{entry.canonical_capability}|"
        f"{'boundary' if entry.is_boundary else 'non_boundary'}"
        for entry in entries
    )
    interactions = Counter(entry.interaction_pattern for entry in entries)
    expressions = Counter(entry.expression_level for entry in entries)
    styles = Counter(entry.user_style for entry in entries)
    ambiguities = Counter(entry.ambiguity_subtype for entry in entries)
    recipes = Counter(entry.prompt_recipe_id for entry in entries)
    rare = {
        "document_boundary": sum(
            entry.is_boundary
            and entry.canonical_capability == "utility.document_reading"
            for entry in entries
        ),
        "counterfactual": sum(entry.is_counterfactual for entry in entries),
        "s3": expressions["S3"],
        "no_result_relaxation": interactions["no_result_relaxation"],
        "goal_change_or_multi_query": interactions["goal_change_or_multi_query"],
    }
    source_counts = (
        dict(Counter(entry.source_id for entry in entries if entry.source_id is not None))
        if sources_present
        else None
    )
    return AuditSampleAudit(
        sample_size=AUDIT_SAMPLE_SIZE,
        mandatory_selected_count=mandatory_selected_count,
        generator_batch_count=len({entry.generator_batch_id for entry in entries}),
        capability_boundary_shape_counts=dict(sorted(cell_counts.items())),
        split_intent_capability_boundary_targets=dict(sorted(targets.items())),
        split_intent_capability_boundary_counts=dict(sorted(sampled_strata.items())),
        interaction_counts=dict(sorted(interactions.items())),
        expression_counts=dict(sorted(expressions.items())),
        user_style_counts=dict(sorted(styles.items())),
        ambiguity_counts=dict(sorted(ambiguities.items())),
        recipe_counts=dict(sorted(recipes.items())),
        rare_counts=rare,
        source_counts=None if source_counts is None else dict(sorted(source_counts.items())),
    )


def _assert_hard_coverage(
    entries: Sequence[AuditSelectionEntry],
    rows: Sequence[AuditPlanRow],
    final_splits: Mapping[str, FinalSplit],
    realism: Mapping[str, RealismAssignment],
    sources: Mapping[str, str],
    policy: AuditSamplingPolicy,
) -> None:
    if len(entries) != policy.sample_size:
        raise AuditCoverageError("audit sample does not have the configured size")
    if len({entry.generator_batch_id for entry in entries}) != 60:
        raise AuditCoverageError("audit sample does not cover every generator batch")
    available_cells = {_cell_key(row, realism[row.plan_id]) for row in rows}
    sampled_cells = Counter(
        f"{entry.canonical_capability}|{'boundary' if entry.is_boundary else 'non_boundary'}"
        f"|{entry.trajectory_shape}"
        for entry in entries
    )
    if any(sampled_cells[key] < policy.capability_boundary_shape_minimum for key in available_cells):
        raise AuditCoverageError("audit sample misses a capability/boundary/shape minimum")
    for field in (
        "interaction_pattern",
        "expression_level",
        "user_style",
        "ambiguity_subtype",
        "prompt_recipe_id",
    ):
        available = {getattr(value, field) for value in realism.values()}
        observed = {getattr(entry, field) for entry in entries}
        if not available <= observed:
            raise AuditCoverageError(f"audit sample misses non-empty {field} values")
    rare = {
        "document_boundary": sum(
            entry.is_boundary
            and entry.canonical_capability == "utility.document_reading"
            for entry in entries
        ),
        "counterfactual": sum(entry.is_counterfactual for entry in entries),
        "s3": sum(entry.expression_level == "S3" for entry in entries),
        "no_result": sum(entry.interaction_pattern == "no_result_relaxation" for entry in entries),
        "goal_change": sum(
            entry.interaction_pattern == "goal_change_or_multi_query" for entry in entries
        ),
    }
    required = {
        "document_boundary": policy.document_boundary_minimum,
        "counterfactual": policy.counterfactual_minimum,
        "s3": policy.s3_minimum,
        "no_result": policy.no_result_minimum,
        "goal_change": policy.goal_change_minimum,
    }
    if any(rare[key] < required[key] for key in required):
        raise AuditCoverageError("audit sample misses a rare-cell explicit minimum")
    if sources:
        observed_sources = {entry.source_id for entry in entries if entry.source_id is not None}
        expected_sources = set(sources.values())
        if observed_sources != expected_sources:
            raise AuditCoverageError("audit sample misses a non-empty source")
    del final_splits  # Explicitly document that coverage is independent of source paths.


def build_core_r2_audit_sample(
    plan_rows: object,
    *,
    final_split_by_plan_id: Mapping[str, object] | Sequence[object],
    realism_sidecar: RealismSidecar,
    seed: int,
    source_by_asset_id: Mapping[str, str] | None = None,
    policy: AuditSamplingPolicy | None = None,
) -> CoreR2AuditSample:
    """Return a fully validated 200-row audit index without any filesystem I/O."""

    rows = coerce_audit_plan_rows(plan_rows)
    final_splits = _normalize_final_splits(rows, final_split_by_plan_id)
    realism = _realism_by_id(realism_sidecar, rows, final_splits)
    sources, source_sha = _normalise_sources(rows, source_by_asset_id)
    effective_policy = policy or AuditSamplingPolicy()
    requirements = _build_requirements(
        rows, final_splits, realism, sources, effective_policy
    )
    selected, reasons = _select_mandatory(
        requirements, seed=seed, sample_size=effective_policy.sample_size
    )
    mandatory_selected_count = len(selected)
    _fill_largest_deficit(
        rows,
        final_splits,
        selected,
        reasons,
        seed=seed,
        sample_size=effective_policy.sample_size,
    )
    rows_by_id = {row.plan_id: row for row in rows}
    entries = _build_entries(
        selected, reasons, rows_by_id, final_splits, realism, sources
    )
    _assert_hard_coverage(
        entries,
        rows,
        final_splits,
        realism,
        sources,
        effective_policy,
    )
    audit = _build_audit(
        entries,
        rows,
        final_splits,
        mandatory_selected_count,
        bool(sources),
        seed=seed,
    )
    manifest = AuditSelectionManifest(
        seed=seed,
        policy_sha256=sha256_bytes(canonical_json_bytes(effective_policy)),
        plan_sha256=realism_sidecar.manifest.plan_sha256,
        asset_catalog_sha256=realism_sidecar.manifest.asset_catalog_sha256,
        capability_assignments_sha256=realism_sidecar.manifest.capability_assignments_sha256,
        realism_manifest_sha256=sha256_bytes(canonical_json_bytes(realism_sidecar.manifest)),
        realism_assignments_sha256=realism_sidecar.manifest.assignments_sha256,
        prompt_recipe_manifest_sha256=realism_sidecar.manifest.prompt_recipe_manifest_sha256,
        final_split_sidecar_sha256=realism_sidecar.manifest.final_split_sidecar_sha256,
        reuse_sidecar_sha256=realism_sidecar.manifest.reuse_sidecar_sha256,
        source_sidecar_sha256=source_sha,
        selection_index_sha256=sha256_bytes(canonical_audit_selection_index_bytes(entries)),
    )
    sample = CoreR2AuditSample(
        entries=entries,
        manifest=manifest,
        audit=audit,
        policy=effective_policy,
    )
    validate_core_r2_audit_sample(
        sample,
        plan_rows=rows,
        final_split_by_plan_id=final_splits,
        realism_sidecar=realism_sidecar,
        source_by_asset_id=source_by_asset_id,
    )
    return sample


def validate_core_r2_audit_sample(
    sample: CoreR2AuditSample,
    *,
    plan_rows: object,
    final_split_by_plan_id: Mapping[str, object] | Sequence[object],
    realism_sidecar: RealismSidecar,
    source_by_asset_id: Mapping[str, str] | None = None,
) -> None:
    """Fail closed on index, manifest, sidecar, or selection drift."""

    rows = coerce_audit_plan_rows(plan_rows)
    final_splits = _normalize_final_splits(rows, final_split_by_plan_id)
    realism = _realism_by_id(realism_sidecar, rows, final_splits)
    sources, source_sha = _normalise_sources(rows, source_by_asset_id)
    manifest = sample.manifest
    if manifest.policy_sha256 != sha256_bytes(canonical_json_bytes(sample.policy)):
        raise ValueError("audit manifest policy digest mismatch")
    expected_manifest_values = {
        "plan_sha256": realism_sidecar.manifest.plan_sha256,
        "asset_catalog_sha256": realism_sidecar.manifest.asset_catalog_sha256,
        "capability_assignments_sha256": realism_sidecar.manifest.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(canonical_json_bytes(realism_sidecar.manifest)),
        "realism_assignments_sha256": realism_sidecar.manifest.assignments_sha256,
        "prompt_recipe_manifest_sha256": realism_sidecar.manifest.prompt_recipe_manifest_sha256,
        "final_split_sidecar_sha256": realism_sidecar.manifest.final_split_sidecar_sha256,
        "reuse_sidecar_sha256": realism_sidecar.manifest.reuse_sidecar_sha256,
        "source_sidecar_sha256": source_sha,
    }
    for field, expected in expected_manifest_values.items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"audit manifest {field} mismatch")
    index_bytes = canonical_audit_selection_index_bytes(sample.entries)
    if manifest.selection_index_sha256 != sha256_bytes(index_bytes):
        raise ValueError("audit manifest selection index digest mismatch")
    rows_by_id = {row.plan_id: row for row in rows}
    if any(entry.plan_id not in rows_by_id for entry in sample.entries):
        raise ValueError("audit index contains an unknown plan_id")
    for entry in sample.entries:
        row = rows_by_id[entry.plan_id]
        assignment = realism[entry.plan_id]
        source_id = sources.get(row.asset_id) if row.asset_id is not None else None
        expected = (
            row.generator_batch_id,
            row.component_id,
            final_splits[entry.plan_id],
            row.canonical_intent,
            row.canonical_capability,
            row.is_boundary,
            row.boundary_strategy,
            assignment.trajectory_shape,
            assignment.interaction_pattern,
            assignment.expression_level,
            assignment.user_style,
            assignment.constraint_level,
            assignment.ambiguity_subtype,
            assignment.prompt_recipe_id,
            assignment.prompt_recipe_sha256,
            row.boundary_strategy in _COUNTERFACTUAL_STRATEGIES,
            source_id,
        )
        observed = (
            entry.generator_batch_id,
            entry.component_id,
            entry.final_split,
            entry.canonical_intent,
            entry.canonical_capability,
            entry.is_boundary,
            entry.boundary_strategy,
            entry.trajectory_shape,
            entry.interaction_pattern,
            entry.expression_level,
            entry.user_style,
            entry.constraint_level,
            entry.ambiguity_subtype,
            entry.prompt_recipe_id,
            entry.prompt_recipe_sha256,
            entry.is_counterfactual,
            entry.source_id,
        )
        if observed != expected:
            raise ValueError(f"audit index row drifted from input bindings: {entry.plan_id}")
    requirements = {
        requirement.label: requirement
        for requirement in _build_requirements(
            rows, final_splits, realism, sources, sample.policy
        )
    }
    for entry in sample.entries:
        for reason in entry.selection_reason:
            if reason == "fill:split_intent_capability_boundary_largest_deficit":
                continue
            requirement = requirements.get(reason)
            if requirement is None or entry.plan_id not in requirement.candidates:
                raise ValueError("audit selection reason does not match its plan row")
    expected_selected, expected_reasons = _select_mandatory(
        tuple(requirements.values()),
        seed=manifest.seed,
        sample_size=sample.policy.sample_size,
    )
    expected_mandatory_selected_count = len(expected_selected)
    _fill_largest_deficit(
        rows,
        final_splits,
        expected_selected,
        expected_reasons,
        seed=manifest.seed,
        sample_size=sample.policy.sample_size,
    )
    expected_entries = _build_entries(
        expected_selected,
        expected_reasons,
        rows_by_id,
        final_splits,
        realism,
        sources,
    )
    if sample.entries != expected_entries:
        raise ValueError("audit selection index drifted from deterministic sampler")
    _assert_hard_coverage(
        sample.entries,
        rows,
        final_splits,
        realism,
        sources,
        sample.policy,
    )
    if sample.audit.mandatory_selected_count != expected_mandatory_selected_count:
        raise ValueError("audit mandatory-selected count drifted from selection reasons")
    expected_audit = _build_audit(
        sample.entries,
        rows,
        final_splits,
        expected_mandatory_selected_count,
        bool(sources),
        seed=manifest.seed,
    )
    if sample.audit != expected_audit:
        raise ValueError("audit summary drifted from selection index")


def audit_sample_selection_bytes(sample: CoreR2AuditSample) -> bytes:
    """Convenience accessor for a create-only orchestration publisher."""

    return canonical_audit_selection_index_bytes(sample.entries)
