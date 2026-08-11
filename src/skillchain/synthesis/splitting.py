"""Schema-v2 group-constrained corpus splitting and frozen-test manifests."""

from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.optimize import Bounds, LinearConstraint, milp

from skillchain import config
from skillchain.data.asset_catalog import AssetCatalog
from skillchain.schemas import Intent, Query, Split
from skillchain.synthesis.batches import AcceptedLedgerEntry, verify_accepted_corpus
from skillchain.synthesis.labeling import (
    ArbitrationItem,
    ArbitrationRecord,
    CrossReviewResult,
    LabelsManifest,
    validate_review_queue_semantics,
)
from skillchain.synthesis.planning import (
    INTENT_ORDER,
    load_plan,
    validate_capability_binding,
)
from skillchain.synthesis.models import PROVISIONAL_LEAKAGE_POLICY_VERSION
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import load_default_taxonomy_registry

Stratum = tuple[Intent, bool]
Capability = str
CapabilityStratum = tuple[Capability, bool]
GateName = Literal["route_gate", "body_gate", "shadow_val"]
GROUP_FIELDS: tuple[str, ...] = (
    "leakage_group_id",
    "boundary_group_id",
    "template_family",
    "generator_batch_id",
)
GROUPING_POLICY_VERSION = "query-connected-components-v1"
_ASSIGNABLE_SPLITS: tuple[Split, ...] = ("opt_pool", "val", "test_frozen")
_VALIDATION_GATES: tuple[GateName, ...] = (
    "route_gate",
    "body_gate",
    "shadow_val",
)
R2_CAPABILITY_ORDER: tuple[Capability, ...] = (
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "knowledge.visual_encyclopedia",
    "utility.document_reading",
    "utility.recipe_guidance",
)


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    sizes: dict[Split, int]
    min_test_per_intent: int = Field(ge=0)
    # These are deliberately optional. The historic full/core split APIs use
    # proportional intent/boundary balancing; r2 additionally freezes the
    # exact capability matrix before any authoring begins.
    capability_targets: dict[Split, dict[Capability, int]] | None = None
    boundary_capability_targets: dict[Split, dict[Capability, int]] | None = None

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("profile 不得为空")
        return value

    @model_validator(mode="after")
    def validate_sizes(self):
        expected = {"dev_mini", "opt_pool", "val", "test_frozen"}
        if set(self.sizes) != expected or any(
            value < 0 for value in self.sizes.values()
        ):
            raise ValueError("sizes 必须完整包含四个非负 split")
        if self.sizes["dev_mini"] <= 0 or self.sizes["test_frozen"] <= 0:
            raise ValueError("dev_mini/test_frozen 必须大于 0")
        if self.min_test_per_intent * len(INTENT_ORDER) > self.sizes["test_frozen"]:
            raise ValueError("min_test_per_intent 与 test_frozen 容量矛盾")
        if (self.capability_targets is None) != (
            self.boundary_capability_targets is None
        ):
            raise ValueError(
                "capability_targets and boundary_capability_targets must be both set "
                "or both omitted"
            )
        self._validate_exact_matrix(
            self.capability_targets,
            label="capability_targets",
            require_size_equality=True,
        )
        self._validate_exact_matrix(
            self.boundary_capability_targets,
            label="boundary_capability_targets",
            require_size_equality=False,
        )
        if (
            self.capability_targets is not None
            and self.boundary_capability_targets is not None
        ):
            for split in self.sizes:
                capabilities = self.capability_targets[split]
                boundaries = self.boundary_capability_targets[split]
                if set(capabilities) != set(boundaries):
                    raise ValueError(
                        "boundary_capability_targets must use the same capability keys"
                    )
                for capability, boundary_count in boundaries.items():
                    if boundary_count > capabilities[capability]:
                        raise ValueError(
                            "boundary capability target cannot exceed capability target: "
                            f"{split}/{capability}"
                        )
        return self

    def _validate_exact_matrix(
        self,
        matrix: dict[Split, dict[Capability, int]] | None,
        *,
        label: str,
        require_size_equality: bool,
    ) -> None:
        if matrix is None:
            return
        if set(matrix) != set(self.sizes):
            raise ValueError(f"{label} must contain every split exactly once")
        expected_capabilities: set[Capability] | None = None
        for split, counts in matrix.items():
            if not counts or any(value < 0 for value in counts.values()):
                raise ValueError(f"{label} contains an empty or negative row: {split}")
            if expected_capabilities is None:
                expected_capabilities = set(counts)
            elif set(counts) != expected_capabilities:
                raise ValueError(f"{label} capability keys drift across splits")
            total = sum(counts.values())
            if require_size_equality and total != self.sizes[split]:
                raise ValueError(
                    f"{label} row total must equal split size: {split}={total}, "
                    f"expected {self.sizes[split]}"
                )
            if not require_size_equality and total > self.sizes[split]:
                raise ValueError(
                    f"{label} row total exceeds split size: {split}={total}, "
                    f"capacity {self.sizes[split]}"
                )


FULL_SPLIT_SPEC = SplitSpec(
    profile="full",
    sizes=dict(config.SPLIT_SIZES),
    min_test_per_intent=config.MIN_TEST_PER_INTENT,
)

CORE_SPLIT_SPEC = SplitSpec(
    profile="core",
    sizes={"dev_mini": 200, "opt_pool": 800, "val": 200, "test_frozen": 300},
    min_test_per_intent=40,
)


CORE_R2_SPLIT_SPEC = SplitSpec(
    profile="core",
    sizes={"dev_mini": 200, "opt_pool": 800, "val": 200, "test_frozen": 300},
    min_test_per_intent=40,
    capability_targets={
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
    },
    boundary_capability_targets={
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
    },
)


class GroupSplitError(ValueError):
    """Base class for fail-closed group allocation errors."""


class LockedGroupConflictError(GroupSplitError):
    pass


class GroupCapacityError(GroupSplitError):
    pass


class ExactSizeInfeasibleError(GroupSplitError):
    pass


class StratificationInfeasibleError(GroupSplitError):
    pass


class SplitSolverError(GroupSplitError):
    pass


class CapabilityInfeasibleError(StratificationInfeasibleError):
    """Exact capability matrix cannot be satisfied without relaxing a quota."""


class BoundaryCapabilityInfeasibleError(CapabilityInfeasibleError):
    """Capability quotas fit, but the exact boundary matrix does not."""


class R2SplitConstraintError(GroupSplitError):
    """The create-only r2 split binding is incomplete or internally inconsistent."""


class ValidationGateInfeasibleError(GroupSplitError):
    """The group-atomic val cohort cannot satisfy the requested gate contract."""


class R2CounterfactualComponent(BaseModel):
    """Declared cross-intent component used by the r2 frozen split binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str
    source: Literal["abo", "food"]
    split: Split
    plan_ids: tuple[str, ...]

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("counterfactual component_id must be non-empty")
        return value

    @field_validator("plan_ids")
    @classmethod
    def validate_plan_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("counterfactual plan_ids must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("counterfactual plan_ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_component_shape(self):
        expected_size = 3 if self.source == "abo" else 2
        if len(self.plan_ids) != expected_size:
            raise ValueError(
                f"{self.source} counterfactual component must contain "
                f"{expected_size} plan_ids"
            )
        if self.split == "dev_mini":
            raise ValueError("r2 counterfactual components must be non-dev")
        return self


class R2SplitConstraints(BaseModel):
    """Typed, create-only-ready r2 binding; it intentionally writes no files.

    ``plan_id_to_split`` is complete rather than a hint.  This makes the MILP a
    deterministic feasibility proof for the planner's frozen assignment instead
    of a post-hoc optimizer that can silently move a component to improve a
    metric after authoring has begun.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    plan_sha256: str
    asset_catalog_sha256: str
    capability_assignments_sha256: str
    grouping_policy_version: Literal["query-connected-components-v1"] = (
        GROUPING_POLICY_VERSION
    )
    group_fields: tuple[str, ...] = GROUP_FIELDS
    plan_id_to_split: dict[str, Split]
    abo_components: tuple[R2CounterfactualComponent, ...] = ()
    food_components: tuple[R2CounterfactualComponent, ...] = ()

    @field_validator(
        "plan_sha256", "asset_catalog_sha256", "capability_assignments_sha256"
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("r2 binding SHA-256 must be 64 lowercase hex characters")
        return value

    @field_validator("plan_id_to_split")
    @classmethod
    def validate_plan_mapping(cls, value: dict[str, Split]) -> dict[str, Split]:
        if not value or any(not key.strip() for key in value):
            raise ValueError("r2 plan_id_to_split must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_contract(self):
        if self.group_fields != GROUP_FIELDS:
            raise ValueError("r2 split constraints must retain the complete GROUP_FIELDS")
        declarations = (*self.abo_components, *self.food_components)
        component_ids = [component.component_id for component in declarations]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("r2 counterfactual component_id values must be unique")
        declared_plan_ids = [
            plan_id for component in declarations for plan_id in component.plan_ids
        ]
        if len(declared_plan_ids) != len(set(declared_plan_ids)):
            raise ValueError("r2 counterfactual declarations may not overlap")
        if any(component.source != "abo" for component in self.abo_components):
            raise ValueError("abo_components must contain source='abo' declarations")
        if any(component.source != "food" for component in self.food_components):
            raise ValueError("food_components must contain source='food' declarations")
        if len(self.abo_components) != 20 or len(self.food_components) != 10:
            raise ValueError(
                "r2 split constraints require exactly 20 ABO and 10 Food components"
            )
        if Counter(component.split for component in self.abo_components) != Counter(
            {"opt_pool": 12, "val": 3, "test_frozen": 5}
        ):
            raise ValueError("ABO component split counts must be opt/val/test=12/3/5")
        if Counter(component.split for component in self.food_components) != Counter(
            {"opt_pool": 6, "val": 2, "test_frozen": 2}
        ):
            raise ValueError("Food component split counts must be opt/val/test=6/2/2")
        for component in declarations:
            if any(plan_id not in self.plan_id_to_split for plan_id in component.plan_ids):
                raise ValueError("counterfactual declaration references an unknown plan_id")
            if any(
                self.plan_id_to_split[plan_id] != component.split
                for plan_id in component.plan_ids
            ):
                raise ValueError("counterfactual declaration split disagrees with plan mapping")
        return self


def verify_r2_split_constraint_binding(
    constraints: R2SplitConstraints,
    *,
    expected_plan_sha256: str,
    expected_asset_catalog_sha256: str,
    expected_capability_assignments_sha256: str,
) -> None:
    """Bind a typed constraint packet to independently trusted parent hashes."""

    expected = {
        "plan_sha256": expected_plan_sha256,
        "asset_catalog_sha256": expected_asset_catalog_sha256,
        "capability_assignments_sha256": expected_capability_assignments_sha256,
    }
    for field_name, expected_sha256 in expected.items():
        if (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
            or getattr(constraints, field_name) != expected_sha256
        ):
            raise R2SplitConstraintError(
                f"r2 split constraint binding mismatch: {field_name}"
            )
    if (
        constraints.grouping_policy_version != GROUPING_POLICY_VERSION
        or constraints.group_fields != GROUP_FIELDS
    ):
        raise R2SplitConstraintError("r2 split constraint grouping binding mismatch")


class ValidationGateSpec(BaseModel):
    """Exact, group-atomic partition of the frozen 200-query val cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sizes: dict[GateName, int] = Field(
        default_factory=lambda: {
            "route_gate": 75,
            "body_gate": 75,
            "shadow_val": 50,
        }
    )
    capability_minimums: dict[GateName, int] = Field(
        default_factory=lambda: {
            "route_gate": 3,
            "body_gate": 3,
            "shadow_val": 2,
        }
    )
    boundary_minimums: dict[GateName, int] = Field(
        default_factory=lambda: {
            "route_gate": 1,
            "body_gate": 1,
            "shadow_val": 1,
        }
    )
    interaction_minimums: dict[GateName, dict[str, int]] = Field(
        default_factory=lambda: {
            "route_gate": {},
            "body_gate": {},
            "shadow_val": {},
        }
    )

    @model_validator(mode="after")
    def validate_contract(self):
        if set(self.sizes) != set(_VALIDATION_GATES):
            raise ValueError("validation gate sizes must name route/body/shadow exactly")
        if set(self.capability_minimums) != set(_VALIDATION_GATES):
            raise ValueError("validation gate capability minimums must name every gate")
        if set(self.boundary_minimums) != set(_VALIDATION_GATES):
            raise ValueError("validation gate boundary minimums must name every gate")
        if set(self.interaction_minimums) != set(_VALIDATION_GATES):
            raise ValueError("validation gate interaction minimums must name every gate")
        if any(value < 0 for value in self.sizes.values()):
            raise ValueError("validation gate sizes must be non-negative")
        if any(value < 0 for value in self.capability_minimums.values()):
            raise ValueError("validation gate capability minimums must be non-negative")
        required_capability_minimums = {
            "route_gate": 3,
            "body_gate": 3,
            "shadow_val": 2,
        }
        if any(
            self.capability_minimums[gate] < required_capability_minimums[gate]
            for gate in _VALIDATION_GATES
        ):
            raise ValueError(
                "validation gate capability minimums may not weaken r2's 3/3/2 floor"
            )
        if any(value < 0 for value in self.boundary_minimums.values()):
            raise ValueError("validation gate boundary minimums must be non-negative")
        if any(
            count < 0 or not pattern.strip()
            for values in self.interaction_minimums.values()
            for pattern, count in values.items()
        ):
            raise ValueError("validation gate interaction minimums must be non-negative")
        if sum(self.sizes.values()) != 200:
            raise ValueError("validation gates must partition exactly 200 val queries")
        for gate in _VALIDATION_GATES:
            if self.boundary_minimums[gate] > self.sizes[gate]:
                raise ValueError("validation gate boundary minimum exceeds gate size")
            if sum(self.interaction_minimums[gate].values()) > self.sizes[gate]:
                raise ValueError("validation gate interaction minima exceed gate size")
        return self


CORE_R2_VALIDATION_GATE_SPEC = ValidationGateSpec()


class ValidationGateAudit(BaseModel):
    """Pure in-memory, typed receipt for a deterministic r2 validation partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    seed: int
    source_split: Literal["val"] = "val"
    grouping_policy_version: Literal["query-connected-components-v1"] = (
        GROUPING_POLICY_VERSION
    )
    group_fields: tuple[str, ...] = GROUP_FIELDS
    query_id_to_gate: dict[str, GateName]
    gate_sizes: dict[GateName, int]
    capability_counts: dict[GateName, dict[str, int]]
    boundary_counts: dict[GateName, int]
    interaction_counts: dict[GateName, dict[str, int]]

    @model_validator(mode="after")
    def validate_contract(self):
        if self.group_fields != GROUP_FIELDS:
            raise ValueError("validation gate audit must retain the complete GROUP_FIELDS")
        if set(self.gate_sizes) != set(_VALIDATION_GATES):
            raise ValueError("validation gate audit sizes must name every gate")
        if sum(self.gate_sizes.values()) != 200:
            raise ValueError("validation gate audit must contain exactly 200 assignments")
        if len(self.query_id_to_gate) != 200:
            raise ValueError("validation gate audit must map exactly 200 query IDs")
        return self


class FrozenSplitError(RuntimeError):
    """test_frozen is missing, modified or inconsistent with its manifest."""


class FrozenSplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    query_schema_version: Literal[2] = 2
    profile: str
    split_sizes: dict[Split, int]
    count: int = Field(gt=0)
    sha256: str
    assignment_sha256: str
    seed: int
    taxonomy_version: str
    full_plan_sha256: str
    asset_catalog_sha256: str | None
    leakage_policy_version: str
    grouping_policy_version: Literal["query-connected-components-v1"]
    group_fields: tuple[str, ...]
    component_count: int = Field(gt=0)
    leakage_violations: Literal[0] = 0
    intent_counts: dict[Intent, int]
    capability_counts: dict[str, int]
    boundary_count: int

    @field_validator(
        "sha256",
        "assignment_sha256",
        "full_plan_sha256",
        "asset_catalog_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("sha256 必须是 64 位小写十六进制")
        return value

    @model_validator(mode="after")
    def validate_contract(self):
        if (
            not self.profile.strip()
            or not self.taxonomy_version.strip()
            or not self.leakage_policy_version.strip()
        ):
            raise ValueError("profile/taxonomy_version/leakage_policy_version 不得为空")
        expected_splits = {"dev_mini", "opt_pool", "val", "test_frozen"}
        if set(self.split_sizes) != expected_splits or any(
            value < 0 for value in self.split_sizes.values()
        ):
            raise ValueError("split_sizes 必须完整包含四个非负 split")
        if self.count != self.split_sizes["test_frozen"]:
            raise ValueError("count 必须等于 test_frozen split size")
        if self.group_fields != GROUP_FIELDS:
            raise ValueError("group_fields 与 grouping policy 不一致")
        is_provisional = (
            self.leakage_policy_version == PROVISIONAL_LEAKAGE_POLICY_VERSION
        )
        if self.asset_catalog_sha256 is None and not is_provisional:
            raise ValueError(
                "无 asset catalog 的冻结 manifest 必须显式标记 provisional"
            )
        if self.asset_catalog_sha256 is not None and is_provisional:
            raise ValueError("catalog-bound 冻结 manifest 不得使用 provisional policy")
        return self


@dataclass(frozen=True)
class AtomicGroup:
    group_id: str
    queries: tuple[Query, ...]
    stratum_counts: Counter[Stratum]
    intent_counts: Counter[Intent]
    capability_counts: Counter[Capability]
    boundary_capability_counts: Counter[Capability]

    @property
    def size(self) -> int:
        return len(self.queries)


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_atomic_groups(candidates: list[Query]) -> list[AtomicGroup]:
    """Build transitive hard groups from namespaced primitive group fields."""

    by_id = {query.query_id: query for query in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("full 切分候选包含重复 query_id")
    ordered_ids = sorted(by_id)
    union_find = _UnionFind(ordered_ids)
    for field_name in GROUP_FIELDS:
        first_by_value: dict[str, str] = {}
        for query_id in ordered_ids:
            value = getattr(by_id[query_id], field_name)
            if value is None:
                continue
            first = first_by_value.setdefault(value, query_id)
            union_find.union(first, query_id)

    members: dict[str, list[Query]] = defaultdict(list)
    for query_id in ordered_ids:
        members[union_find.find(query_id)].append(by_id[query_id])

    groups: list[AtomicGroup] = []
    for queries in members.values():
        ordered = tuple(sorted(queries, key=lambda query: query.query_id))
        digest = hashlib.sha256(
            "\n".join(query.query_id for query in ordered).encode("utf-8")
        ).hexdigest()
        groups.append(
            AtomicGroup(
                group_id=f"component.{digest}",
                queries=ordered,
                stratum_counts=Counter(
                    (query.canonical_intent, query.is_boundary) for query in ordered
                ),
                intent_counts=Counter(query.canonical_intent for query in ordered),
                capability_counts=Counter(
                    query.canonical_capability
                    for query in ordered
                    if query.canonical_capability is not None
                ),
                boundary_capability_counts=Counter(
                    query.canonical_capability
                    for query in ordered
                    if query.is_boundary and query.canonical_capability is not None
                ),
            )
        )
    return sorted(groups, key=lambda group: group.group_id)


def _validate_query_capability_bindings(queries: list[Query]) -> None:
    """Reject unresolved or cross-intent capability labels before splitting."""

    taxonomy = load_default_taxonomy_registry()
    task_specification = load_mvp_task_specification_v1()
    for query in queries:
        if query.canonical_capability is None or query.requires_card is None:
            raise ValueError(
                "formal split rejects unresolved capability/requires_card Query: "
                f"{query.query_id}"
            )
        try:
            validate_capability_binding(
                taxonomy_version=query.taxonomy_version,
                task_spec_version=query.task_spec_version,
                canonical_intent=query.canonical_intent,
                canonical_capability=query.canonical_capability,
                acceptable_capabilities=query.acceptable_capabilities,
                requires_card=query.requires_card,
                taxonomy=taxonomy,
                task_specification=task_specification,
            )
        except ValueError as exc:
            raise ValueError(
                f"query capability binding is invalid: {query.query_id}: {exc}"
            ) from exc


def _validate_exact_target_inventory(
    candidates: list[Query],
    locked: set[str],
    split_spec: SplitSpec,
) -> None:
    """Reject impossible r2 inventories before invoking the MILP."""

    capability_targets = split_spec.capability_targets
    boundary_targets = split_spec.boundary_capability_targets
    if capability_targets is None or boundary_targets is None:
        return
    expected_capabilities = Counter()
    expected_boundaries = Counter()
    for split in split_spec.sizes:
        expected_capabilities.update(capability_targets[split])
        expected_boundaries.update(boundary_targets[split])
    actual_capabilities = Counter(
        query.canonical_capability for query in candidates if query.canonical_capability
    )
    actual_boundaries = Counter(
        query.canonical_capability
        for query in candidates
        if query.is_boundary and query.canonical_capability
    )
    if actual_capabilities != expected_capabilities:
        raise CapabilityInfeasibleError(
            "candidate inventory does not match the frozen global capability totals: "
            f"actual={dict(actual_capabilities)}, "
            f"expected={dict(expected_capabilities)}"
        )
    if actual_boundaries != expected_boundaries:
        raise BoundaryCapabilityInfeasibleError(
            "candidate inventory does not match the frozen global boundary x "
            f"capability totals: actual={dict(actual_boundaries)}, "
            f"expected={dict(expected_boundaries)}"
        )
    locked_queries = [query for query in candidates if query.query_id in locked]
    locked_capabilities = Counter(
        query.canonical_capability
        for query in locked_queries
        if query.canonical_capability
    )
    if locked_capabilities != Counter(capability_targets["dev_mini"]):
        raise CapabilityInfeasibleError(
            "locked dev prefix does not match the frozen dev capability row"
        )
    locked_boundaries = Counter(
        query.canonical_capability
        for query in locked_queries
        if query.is_boundary and query.canonical_capability
    )
    if locked_boundaries != Counter(boundary_targets["dev_mini"]):
        raise BoundaryCapabilityInfeasibleError(
            "locked dev prefix does not match the frozen dev boundary capability row"
        )


def _validate_r2_split_constraints(
    candidates: list[Query],
    groups: list[AtomicGroup],
    *,
    locked: set[str],
    split_spec: SplitSpec,
    constraints: R2SplitConstraints,
) -> dict[str, Split]:
    if sum(split_spec.sizes.values()) != 1500 or len(candidates) != 1500:
        raise R2SplitConstraintError(
            "R2SplitConstraints require the complete 1,500-query core plan"
        )
    if split_spec.capability_targets is None:
        raise R2SplitConstraintError(
            "R2SplitConstraints require an exact capability/boundary SplitSpec"
        )
    query_by_id = {query.query_id: query for query in candidates}
    if set(constraints.plan_id_to_split) != set(query_by_id):
        missing = sorted(set(query_by_id) - set(constraints.plan_id_to_split))
        extra = sorted(set(constraints.plan_id_to_split) - set(query_by_id))
        raise R2SplitConstraintError(
            "r2 plan_id_to_split is not a complete plan mapping: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if Counter(constraints.plan_id_to_split.values()) != Counter(split_spec.sizes):
        raise R2SplitConstraintError(
            "r2 plan_id_to_split counts do not match the frozen split sizes"
        )
    mapped_dev = {
        plan_id
        for plan_id, split in constraints.plan_id_to_split.items()
        if split == "dev_mini"
    }
    if mapped_dev != locked:
        raise R2SplitConstraintError(
            "r2 plan mapping must preserve the locked dev prefix exactly"
        )

    group_by_plan_id: dict[str, AtomicGroup] = {}
    fixed_group_splits: dict[str, Split] = {}
    batch_sizes = Counter(
        query.generator_batch_id for query in candidates
    )
    if len(batch_sizes) != 60 or set(batch_sizes.values()) != {25}:
        raise R2SplitConstraintError(
            "r2 split constraints require exactly 60 generator batches of 25 queries"
        )
    for group in groups:
        generator_batch_ids = {
            query.generator_batch_id for query in group.queries
        }
        if len(group.queries) != 25 or len(generator_batch_ids) != 1:
            raise R2SplitConstraintError(
                "r2 GROUP_FIELDS atoms must each equal one complete 25-query "
                f"generator batch: {group.group_id}"
            )
        declared_splits = {
            constraints.plan_id_to_split[query.query_id] for query in group.queries
        }
        if len(declared_splits) != 1:
            raise R2SplitConstraintError(
                "r2 plan mapping splits an atomic GROUP_FIELDS component: "
                f"{group.group_id}"
            )
        fixed_group_splits[group.group_id] = next(iter(declared_splits))
        for query in group.queries:
            group_by_plan_id[query.query_id] = group

    expected_capabilities = {
        "abo": {
            "product.exact_match",
            "product.style_recommendation",
            "knowledge.visual_encyclopedia",
        },
        "food": {
            "knowledge.visual_encyclopedia",
            "utility.recipe_guidance",
        },
    }
    expected_strategies = {
        "abo": "cross_intent_triplet",
        "food": "cross_intent_pair",
    }
    for declaration in (
        *constraints.abo_components,
        *constraints.food_components,
    ):
        queries = [query_by_id[plan_id] for plan_id in declaration.plan_ids]
        containing_groups = {
            group_by_plan_id[query.query_id].group_id for query in queries
        }
        if len(containing_groups) != 1:
            raise R2SplitConstraintError(
                "counterfactual component crosses atomic GROUP_FIELDS components: "
                f"{declaration.component_id}"
            )
        if any(
            query.leakage_group_id != declaration.component_id
            for query in queries
        ):
            raise R2SplitConstraintError(
                "counterfactual declaration does not bind its shared leakage component: "
                f"{declaration.component_id}"
            )
        if (
            len({query.asset_id for query in queries}) != 1
            or len({query.image_path for query in queries}) != 1
        ):
            raise R2SplitConstraintError(
                "counterfactual declaration does not bind one shared asset/path: "
                f"{declaration.component_id}"
            )
        capabilities = {query.canonical_capability for query in queries}
        if capabilities != expected_capabilities[declaration.source]:
            raise R2SplitConstraintError(
                "counterfactual declaration has the wrong capability shape: "
                f"{declaration.component_id}"
            )
        boundary_group_ids = {query.boundary_group_id for query in queries}
        if len(boundary_group_ids) != 1 or None in boundary_group_ids:
            raise R2SplitConstraintError(
                "counterfactual declaration does not share one boundary component: "
                f"{declaration.component_id}"
            )
        if any(
            not query.is_boundary
            or query.boundary_strategy != expected_strategies[declaration.source]
            for query in queries
        ):
            raise R2SplitConstraintError(
                "counterfactual declaration is missing its boundary strategy: "
                f"{declaration.component_id}"
            )
    return fixed_group_splits


def validate_exact_split_targets(
    assigned: list[Query],
    split_spec: SplitSpec,
) -> None:
    """Pure-Python recomputation of every exact r2 split equality."""

    if Counter(query.split for query in assigned) != Counter(split_spec.sizes):
        raise ExactSizeInfeasibleError("assigned rows do not match exact split sizes")
    capability_targets = split_spec.capability_targets
    boundary_targets = split_spec.boundary_capability_targets
    if capability_targets is None or boundary_targets is None:
        return
    actual_capabilities: dict[Split, Counter[Capability]] = {
        split: Counter() for split in split_spec.sizes
    }
    actual_boundaries: dict[Split, Counter[Capability]] = {
        split: Counter() for split in split_spec.sizes
    }
    for query in assigned:
        if query.canonical_capability is None:
            raise CapabilityInfeasibleError(
                f"assigned query has no capability: {query.query_id}"
            )
        actual_capabilities[query.split][query.canonical_capability] += 1
        if query.is_boundary:
            actual_boundaries[query.split][query.canonical_capability] += 1
    for split in split_spec.sizes:
        if actual_capabilities[split] != Counter(capability_targets[split]):
            raise CapabilityInfeasibleError(
                f"exact capability post-check failed for {split}: "
                f"actual={dict(actual_capabilities[split])}, "
                f"expected={capability_targets[split]}"
            )
        if actual_boundaries[split] != Counter(boundary_targets[split]):
            raise BoundaryCapabilityInfeasibleError(
                f"exact boundary capability post-check failed for {split}: "
                f"actual={dict(actual_boundaries[split])}, "
                f"expected={boundary_targets[split]}"
            )


def stratified_split(
    candidates: list[Query],
    *,
    locked_dev_query_ids: set[str],
    seed: int,
    split_spec: SplitSpec = FULL_SPLIT_SPEC,
    r2_constraints: R2SplitConstraints | None = None,
) -> list[Query]:
    """Assign complete connected components with exact split capacities."""

    expected_total = sum(split_spec.sizes.values())
    if len(candidates) != expected_total:
        raise ValueError(
            f"{split_spec.profile} 切分必须恰好有 {expected_total} 条，实际 {len(candidates)}"
        )
    if any(query.schema_version != 2 for query in candidates):
        raise ValueError("formal split 只接受 schema v2 Query")
    unresolved = [
        query.query_id
        for query in candidates
        if query.canonical_capability is None or query.requires_card is None
    ]
    if unresolved:
        raise ValueError(
            "formal split 拒绝 capability/requires_card 未解析的 Query: "
            + ", ".join(unresolved[:5])
        )
    _validate_query_capability_bindings(candidates)
    taxonomy_versions = {query.taxonomy_version for query in candidates}
    if len(taxonomy_versions) != 1:
        raise ValueError("formal split 要求单一 taxonomy_version")

    locked = set(locked_dev_query_ids)
    if len(locked) != split_spec.sizes["dev_mini"]:
        raise ValueError("locked_dev_query_ids 数量与 SplitSpec 不一致")
    all_ids = {query.query_id for query in candidates}
    if not locked <= all_ids:
        raise ValueError("locked dev query_id 不完整")
    current_dev = {query.query_id for query in candidates if query.split == "dev_mini"}
    if current_dev != locked:
        raise ValueError("locked dev IDs 必须与当前 dev_mini 记录一一对应")

    groups = build_atomic_groups(candidates)
    fixed_group_splits: dict[str, Split] | None = None
    if r2_constraints is not None:
        fixed_group_splits = _validate_r2_split_constraints(
            candidates,
            groups,
            locked=locked,
            split_spec=split_spec,
            constraints=r2_constraints,
        )
    _validate_exact_target_inventory(candidates, locked, split_spec)
    unlocked_groups: list[AtomicGroup] = []
    for group in groups:
        member_ids = {query.query_id for query in group.queries}
        locked_members = member_ids & locked
        if locked_members and locked_members != member_ids:
            sample = ", ".join(sorted(member_ids)[:6])
            raise LockedGroupConflictError(
                f"locked dev 与非 dev 共享原子 group {group.group_id}: {sample}"
            )
        if not locked_members:
            unlocked_groups.append(group)

    max_capacity = max(split_spec.sizes[split] for split in _ASSIGNABLE_SPLITS)
    oversized = [group for group in unlocked_groups if group.size > max_capacity]
    if oversized:
        group = max(oversized, key=lambda item: item.size)
        raise GroupCapacityError(
            f"原子 group 大小 {group.size} 超过所有可分配 split 容量: {group.group_id}"
        )

    unlocked_fixed = (
        None
        if fixed_group_splits is None
        else {
            group.group_id: fixed_group_splits[group.group_id]
            for group in unlocked_groups
        }
    )
    assignment = _solve_group_assignment(
        unlocked_groups,
        split_spec,
        seed,
        fixed_group_splits=unlocked_fixed,
    )
    split_by_id = {query_id: "dev_mini" for query_id in locked}
    for group in unlocked_groups:
        split = assignment[group.group_id]
        split_by_id.update({query.query_id: split for query in group.queries})

    assigned = [
        Query.model_validate(
            {**query.model_dump(mode="json"), "split": split_by_id[query.query_id]}
        )
        for query in sorted(candidates, key=lambda item: item.query_id)
    ]
    counts = Counter(query.split for query in assigned)
    if counts != Counter(split_spec.sizes):
        raise AssertionError(f"切分数量异常: {dict(counts)}")
    test_intents = Counter(
        query.canonical_intent for query in assigned if query.split == "test_frozen"
    )
    if any(
        test_intents[intent] < split_spec.min_test_per_intent for intent in INTENT_ORDER
    ):
        raise AssertionError(f"求解结果违反 test intent minimum: {dict(test_intents)}")
    validate_exact_split_targets(assigned, split_spec)
    if r2_constraints is not None and any(
        query.split != r2_constraints.plan_id_to_split[query.query_id]
        for query in assigned
    ):
        raise R2SplitConstraintError("MILP result drifted from the frozen r2 mapping")
    assert_no_group_leakage(assigned)
    return assigned


def _solve_group_assignment(
    groups: list[AtomicGroup],
    split_spec: SplitSpec,
    seed: int,
    *,
    fixed_group_splits: dict[str, Split] | None = None,
) -> dict[str, Split]:
    if not groups:
        raise ExactSizeInfeasibleError("没有可分配的非 dev group")
    # Stable seed-dependent row order gives deterministic tie resolution without
    # adding tiny objective coefficients that make HiGHS prove a degenerate
    # optimum unnecessarily slowly.
    groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}|{group.group_id}".encode("utf-8")
        ).hexdigest(),
    )
    strata = sorted(
        {(intent, boundary) for intent in INTENT_ORDER for boundary in (False, True)}
    )
    total_strata = Counter()
    for group in groups:
        total_strata.update(group.stratum_counts)
    targets = _stratum_targets(total_strata, split_spec)
    feasibility = _run_milp(
        groups,
        split_spec,
        strata,
        targets,
        enforce_minimum=True,
        optimize_balance=False,
        time_limit=15.0,
        fixed_group_splits=fixed_group_splits,
    )
    if feasibility.assignment is None:
        if feasibility.status not in {"infeasible"}:
            raise SplitSolverError(
                "group split feasibility 求解未得出结论: "
                f"status={feasibility.status}; {feasibility.message}"
            )
        capacity_result = _run_milp(
            groups,
            split_spec,
            strata,
            targets,
            enforce_minimum=False,
            optimize_balance=False,
            time_limit=15.0,
            fixed_group_splits=fixed_group_splits,
            enforce_exact_targets=False,
        )
        histogram = dict(sorted(Counter(group.size for group in groups).items()))
        if (
            capacity_result.assignment is None
            and capacity_result.status == "infeasible"
        ):
            raise ExactSizeInfeasibleError(
                f"原子 group 无法组成精确 split sizes={split_spec.sizes}; "
                f"group_size_histogram={histogram}"
            )
        if capacity_result.assignment is None:
            raise SplitSolverError(
                "group capacity feasibility 求解未得出结论: "
                f"status={capacity_result.status}; {capacity_result.message}"
            )
        if split_spec.capability_targets is not None:
            capability_result = _run_milp(
                groups,
                split_spec,
                strata,
                targets,
                enforce_minimum=False,
                optimize_balance=False,
                time_limit=15.0,
                fixed_group_splits=fixed_group_splits,
                enforce_boundary_targets=False,
            )
            if capability_result.assignment is None:
                if capability_result.status == "infeasible":
                    raise CapabilityInfeasibleError(
                        "exact split sizes are feasible, but the frozen capability "
                        "matrix is infeasible"
                    )
                raise SplitSolverError(
                    "capability feasibility solve was inconclusive: "
                    f"status={capability_result.status}; "
                    f"{capability_result.message}"
                )
            boundary_result = _run_milp(
                groups,
                split_spec,
                strata,
                targets,
                enforce_minimum=False,
                optimize_balance=False,
                time_limit=15.0,
                fixed_group_splits=fixed_group_splits,
            )
            if boundary_result.assignment is None:
                if boundary_result.status == "infeasible":
                    raise BoundaryCapabilityInfeasibleError(
                        "capability matrix is feasible, but the frozen boundary x "
                        "capability matrix is infeasible"
                    )
                raise SplitSolverError(
                    "boundary capability feasibility solve was inconclusive: "
                    f"status={boundary_result.status}; {boundary_result.message}"
                )
        raise StratificationInfeasibleError(
            "精确容量可行，但 test intent minimum 不可行: "
            f"minimum={split_spec.min_test_per_intent}"
        )

    # Improve proportional intent/boundary balance under a short bounded solve.
    # A valid feasibility incumbent is always retained, so an optimization time
    # limit can never be misreported as mathematical infeasibility.
    balanced = _run_milp(
        groups,
        split_spec,
        strata,
        targets,
        enforce_minimum=True,
        optimize_balance=True,
        time_limit=5.0,
        fixed_group_splits=fixed_group_splits,
    )
    return balanced.assignment or feasibility.assignment


@dataclass(frozen=True)
class _MilpOutcome:
    assignment: dict[str, Split] | None
    status: Literal["feasible", "infeasible", "timeout", "error"]
    message: str


def _run_milp(
    groups: list[AtomicGroup],
    split_spec: SplitSpec,
    strata: list[Stratum],
    targets: dict[Split, dict[Stratum, int]],
    *,
    enforce_minimum: bool,
    optimize_balance: bool,
    time_limit: float,
    fixed_group_splits: dict[str, Split] | None = None,
    enforce_exact_targets: bool = True,
    enforce_boundary_targets: bool = True,
) -> _MilpOutcome:
    split_count = len(_ASSIGNABLE_SPLITS)
    x_count = len(groups) * split_count
    deviation_count = split_count * len(strata) * 2 if optimize_balance else 0
    variable_count = x_count + deviation_count
    objective = np.zeros(variable_count, dtype=np.float64)
    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[:x_count] = 1
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    upper_bounds[:x_count] = 1.0

    def x_index(group_index: int, split_index: int) -> int:
        return group_index * split_count + split_index

    def deviation_index(split_index: int, stratum_index: int, positive: bool) -> int:
        if not optimize_balance:  # pragma: no cover - internal misuse guard
            raise AssertionError("feasibility model has no deviation variables")
        base = x_count + (split_index * len(strata) + stratum_index) * 2
        return base + (0 if positive else 1)

    if optimize_balance:
        for split_index, split in enumerate(_ASSIGNABLE_SPLITS):
            for stratum_index, stratum in enumerate(strata):
                weight = 1.0 / max(1, targets[split][stratum])
                objective[deviation_index(split_index, stratum_index, True)] = weight
                objective[deviation_index(split_index, stratum_index, False)] = weight

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for group_index in range(len(groups)):
        row = np.zeros(variable_count)
        for split_index in range(split_count):
            row[x_index(group_index, split_index)] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)
    if fixed_group_splits is not None:
        if set(fixed_group_splits) != {group.group_id for group in groups}:
            raise R2SplitConstraintError(
                "fixed group mapping must cover every non-dev atomic group exactly"
            )
        for group_index, group in enumerate(groups):
            split = fixed_group_splits[group.group_id]
            if split not in _ASSIGNABLE_SPLITS:
                raise R2SplitConstraintError(
                    f"non-dev atomic group is fixed to an invalid split: {split}"
                )
            row = np.zeros(variable_count)
            split_index = _ASSIGNABLE_SPLITS.index(split)
            row[x_index(group_index, split_index)] = 1
            rows.append(row)
            lower.append(1)
            upper.append(1)
    for split_index, split in enumerate(_ASSIGNABLE_SPLITS):
        row = np.zeros(variable_count)
        for group_index, group in enumerate(groups):
            row[x_index(group_index, split_index)] = group.size
        rows.append(row)
        target_size = split_spec.sizes[split]
        lower.append(target_size)
        upper.append(target_size)
    if enforce_exact_targets and split_spec.capability_targets is not None:
        for split_index, split in enumerate(_ASSIGNABLE_SPLITS):
            for capability, target in split_spec.capability_targets[split].items():
                row = np.zeros(variable_count)
                for group_index, group in enumerate(groups):
                    row[x_index(group_index, split_index)] = group.capability_counts[
                        capability
                    ]
                rows.append(row)
                lower.append(target)
                upper.append(target)
    if (
        enforce_exact_targets
        and enforce_boundary_targets
        and split_spec.boundary_capability_targets is not None
    ):
        for split_index, split in enumerate(_ASSIGNABLE_SPLITS):
            boundary_targets = split_spec.boundary_capability_targets[split]
            for capability, target in boundary_targets.items():
                row = np.zeros(variable_count)
                for group_index, group in enumerate(groups):
                    row[x_index(group_index, split_index)] = (
                        group.boundary_capability_counts[capability]
                    )
                rows.append(row)
                lower.append(target)
                upper.append(target)
    if enforce_minimum:
        test_index = _ASSIGNABLE_SPLITS.index("test_frozen")
        for intent in INTENT_ORDER:
            row = np.zeros(variable_count)
            for group_index, group in enumerate(groups):
                row[x_index(group_index, test_index)] = group.intent_counts[intent]
            rows.append(row)
            lower.append(split_spec.min_test_per_intent)
            upper.append(np.inf)
    if optimize_balance:
        for split_index, split in enumerate(_ASSIGNABLE_SPLITS):
            for stratum_index, stratum in enumerate(strata):
                row = np.zeros(variable_count)
                for group_index, group in enumerate(groups):
                    row[x_index(group_index, split_index)] = group.stratum_counts[
                        stratum
                    ]
                row[deviation_index(split_index, stratum_index, True)] = -1
                row[deviation_index(split_index, stratum_index, False)] = 1
                rows.append(row)
                lower.append(targets[split][stratum])
                upper.append(targets[split][stratum])

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={
            "presolve": True,
            "time_limit": time_limit,
            "mip_rel_gap": 0.01 if optimize_balance else 0.0,
        },
    )
    assignment = _extract_feasible_assignment(
        result.x,
        groups,
        split_spec,
        enforce_minimum=enforce_minimum,
    )
    if assignment is not None:
        return _MilpOutcome(assignment, "feasible", str(result.message))
    if result.status == 2:
        return _MilpOutcome(None, "infeasible", str(result.message))
    if result.status == 1:
        return _MilpOutcome(None, "timeout", str(result.message))
    return _MilpOutcome(None, "error", str(result.message))


def _extract_feasible_assignment(
    solution: np.ndarray | None,
    groups: list[AtomicGroup],
    split_spec: SplitSpec,
    *,
    enforce_minimum: bool,
) -> dict[str, Split] | None:
    """Accept an integral feasible incumbent even when optimization timed out."""

    if solution is None:
        return None
    split_count = len(_ASSIGNABLE_SPLITS)
    assignment: dict[str, Split] = {}
    for group_index, group in enumerate(groups):
        values = solution[group_index * split_count : (group_index + 1) * split_count]
        if len(values) != split_count or any(
            min(abs(value), abs(value - 1.0)) > 1e-5 for value in values
        ):
            return None
        selected = [
            split
            for split, value in zip(_ASSIGNABLE_SPLITS, values, strict=True)
            if value > 0.5
        ]
        if len(selected) != 1:
            return None
        assignment[group.group_id] = selected[0]

    sizes: Counter[Split] = Counter()
    test_intents: Counter[Intent] = Counter()
    for group in groups:
        split = assignment[group.group_id]
        sizes[split] += group.size
        if split == "test_frozen":
            test_intents.update(group.intent_counts)
    if any(sizes[split] != split_spec.sizes[split] for split in _ASSIGNABLE_SPLITS):
        return None
    if enforce_minimum and any(
        test_intents[intent] < split_spec.min_test_per_intent for intent in INTENT_ORDER
    ):
        return None
    return assignment


def _stratum_targets(
    total_strata: Counter[Stratum], split_spec: SplitSpec
) -> dict[Split, dict[Stratum, int]]:
    # Keep the complete registered stratum space even in small/unit-test cohorts.
    # The MILP objective and equality rows address all ten strata explicitly.
    remaining = {
        (intent, boundary): total_strata[(intent, boundary)]
        for intent in INTENT_ORDER
        for boundary in (False, True)
    }
    targets: dict[Split, dict[Stratum, int]] = {}
    for split in ("test_frozen", "val"):
        allocation = _largest_remainder(remaining, split_spec.sizes[split])
        targets[split] = allocation
        remaining = {key: remaining[key] - allocation[key] for key in remaining}
    targets["opt_pool"] = remaining
    return targets


def plan_validation_gates(
    val_queries: list[Query],
    *,
    seed: int,
    spec: ValidationGateSpec = CORE_R2_VALIDATION_GATE_SPEC,
    interaction_by_query_id: Mapping[str, str] | None = None,
) -> ValidationGateAudit:
    """Partition the frozen val cohort while preserving every GROUP_FIELDS atom."""

    if spec.sizes == {
        "route_gate": 80,
        "body_gate": 80,
        "shadow_val": 40,
    }:
        raise ValidationGateInfeasibleError(
            "legacy 80/80/40 gates are infeasible under 25-query generator-batch "
            "atomicity; r2 uses 75/75/50"
        )
    if spec.sizes != CORE_R2_VALIDATION_GATE_SPEC.sizes:
        raise ValidationGateInfeasibleError(
            "r2 validation gates are frozen at 75/75/50"
        )
    if len(val_queries) != 200 or any(query.split != "val" for query in val_queries):
        raise ValidationGateInfeasibleError(
            "r2 validation gates require exactly 200 val queries"
        )
    query_ids = {query.query_id for query in val_queries}
    if len(query_ids) != len(val_queries):
        raise ValidationGateInfeasibleError("validation cohort has duplicate query IDs")
    capabilities = {query.canonical_capability for query in val_queries}
    if capabilities != set(R2_CAPABILITY_ORDER):
        raise ValidationGateInfeasibleError(
            "validation cohort must contain exactly the six r2 capabilities"
        )

    needs_interactions = any(
        count > 0
        for gate in _VALIDATION_GATES
        for count in spec.interaction_minimums[gate].values()
    )
    interactions: dict[str, str] = {}
    if interaction_by_query_id is not None:
        interactions = {
            query_id: pattern.strip()
            for query_id, pattern in interaction_by_query_id.items()
        }
        if set(interactions) != query_ids or any(not value for value in interactions.values()):
            raise ValidationGateInfeasibleError(
                "interaction sidecar must cover every val query exactly"
            )
    elif needs_interactions:
        raise ValidationGateInfeasibleError(
            "interaction minima require a complete interaction sidecar"
        )

    groups = build_atomic_groups(val_queries)
    group_histogram = Counter(group.size for group in groups)
    if group_histogram != Counter({25: 8}):
        raise ValidationGateInfeasibleError(
            "r2 val must be eight complete 25-query GROUP_FIELDS atoms: "
            f"group_size_histogram={dict(sorted(group_histogram.items()))}"
        )
    groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"validation-gates|{seed}|{group.group_id}".encode("utf-8")
        ).hexdigest(),
    )
    gate_count = len(_VALIDATION_GATES)
    variable_count = len(groups) * gate_count
    objective = np.zeros(variable_count, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int32)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)

    def x_index(group_index: int, gate_index: int) -> int:
        return group_index * gate_count + gate_index

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for group_index in range(len(groups)):
        row = np.zeros(variable_count)
        for gate_index in range(gate_count):
            row[x_index(group_index, gate_index)] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)
    for gate_index, gate in enumerate(_VALIDATION_GATES):
        size_row = np.zeros(variable_count)
        for group_index, group in enumerate(groups):
            size_row[x_index(group_index, gate_index)] = group.size
        rows.append(size_row)
        lower.append(spec.sizes[gate])
        upper.append(spec.sizes[gate])
        for capability in R2_CAPABILITY_ORDER:
            capability_row = np.zeros(variable_count)
            for group_index, group in enumerate(groups):
                capability_row[x_index(group_index, gate_index)] = (
                    group.capability_counts[capability]
                )
            rows.append(capability_row)
            lower.append(spec.capability_minimums[gate])
            upper.append(np.inf)
        boundary_row = np.zeros(variable_count)
        for group_index, group in enumerate(groups):
            boundary_row[x_index(group_index, gate_index)] = sum(
                group.boundary_capability_counts.values()
            )
        rows.append(boundary_row)
        lower.append(spec.boundary_minimums[gate])
        upper.append(np.inf)
        for pattern, minimum in spec.interaction_minimums[gate].items():
            interaction_row = np.zeros(variable_count)
            for group_index, group in enumerate(groups):
                interaction_row[x_index(group_index, gate_index)] = sum(
                    interactions.get(query.query_id) == pattern
                    for query in group.queries
                )
            rows.append(interaction_row)
            lower.append(minimum)
            upper.append(np.inf)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={"presolve": True, "time_limit": 15.0, "mip_rel_gap": 0.0},
    )
    if result.x is None or result.status not in {0, 1}:
        raise ValidationGateInfeasibleError(
            "validation gate constraints are infeasible: "
            f"sizes={spec.sizes}; status={result.status}; {result.message}"
        )
    group_to_gate: dict[str, GateName] = {}
    for group_index, group in enumerate(groups):
        values = result.x[
            group_index * gate_count : (group_index + 1) * gate_count
        ]
        selected = [
            gate
            for gate, value in zip(_VALIDATION_GATES, values, strict=True)
            if value > 0.5
        ]
        if len(selected) != 1 or any(
            min(abs(value), abs(value - 1.0)) > 1e-5 for value in values
        ):
            raise ValidationGateInfeasibleError(
                "validation gate solver returned a non-integral incumbent"
            )
        group_to_gate[group.group_id] = selected[0]
    query_id_to_gate = {
        query.query_id: group_to_gate[group.group_id]
        for group in groups
        for query in group.queries
    }
    gate_sizes: Counter[GateName] = Counter(query_id_to_gate.values())
    capability_counts: dict[GateName, Counter[str]] = {
        gate: Counter() for gate in _VALIDATION_GATES
    }
    boundary_counts: Counter[GateName] = Counter()
    interaction_counts: dict[GateName, Counter[str]] = {
        gate: Counter() for gate in _VALIDATION_GATES
    }
    for query in val_queries:
        gate = query_id_to_gate[query.query_id]
        if query.canonical_capability is not None:
            capability_counts[gate][query.canonical_capability] += 1
        if query.is_boundary:
            boundary_counts[gate] += 1
        if query.query_id in interactions:
            interaction_counts[gate][interactions[query.query_id]] += 1
    audit = ValidationGateAudit(
        seed=seed,
        query_id_to_gate=query_id_to_gate,
        gate_sizes={gate: gate_sizes[gate] for gate in _VALIDATION_GATES},
        capability_counts={
            gate: dict(sorted(capability_counts[gate].items()))
            for gate in _VALIDATION_GATES
        },
        boundary_counts={gate: boundary_counts[gate] for gate in _VALIDATION_GATES},
        interaction_counts={
            gate: dict(sorted(interaction_counts[gate].items()))
            for gate in _VALIDATION_GATES
        },
    )
    validate_validation_gate_audit(
        val_queries,
        audit,
        spec=spec,
        interaction_by_query_id=interactions or None,
    )
    return audit


def validate_validation_gate_audit(
    val_queries: list[Query],
    audit: ValidationGateAudit,
    *,
    spec: ValidationGateSpec = CORE_R2_VALIDATION_GATE_SPEC,
    interaction_by_query_id: Mapping[str, str] | None = None,
) -> None:
    """Fail closed when a typed gate audit is incomplete or has been altered."""

    query_by_id = {query.query_id: query for query in val_queries}
    if len(query_by_id) != 200 or set(audit.query_id_to_gate) != set(query_by_id):
        raise ValidationGateInfeasibleError(
            "validation gate audit does not cover the 200-query val cohort"
        )
    interactions = dict(interaction_by_query_id or {})
    if interactions and set(interactions) != set(query_by_id):
        raise ValidationGateInfeasibleError(
            "validation gate interaction sidecar is incomplete"
        )
    for group in build_atomic_groups(val_queries):
        gates = {audit.query_id_to_gate[query.query_id] for query in group.queries}
        if len(gates) != 1:
            raise ValidationGateInfeasibleError(
                f"validation gate audit splits GROUP_FIELDS atom {group.group_id}"
            )
    sizes: Counter[GateName] = Counter(audit.query_id_to_gate.values())
    capabilities: dict[GateName, Counter[str]] = {
        gate: Counter() for gate in _VALIDATION_GATES
    }
    boundaries: Counter[GateName] = Counter()
    interaction_counts: dict[GateName, Counter[str]] = {
        gate: Counter() for gate in _VALIDATION_GATES
    }
    for query_id, gate in audit.query_id_to_gate.items():
        query = query_by_id[query_id]
        if query.canonical_capability is not None:
            capabilities[gate][query.canonical_capability] += 1
        if query.is_boundary:
            boundaries[gate] += 1
        if query_id in interactions:
            interaction_counts[gate][interactions[query_id]] += 1
    expected_capability_counts = {
        gate: dict(sorted(capabilities[gate].items())) for gate in _VALIDATION_GATES
    }
    expected_interaction_counts = {
        gate: dict(sorted(interaction_counts[gate].items()))
        for gate in _VALIDATION_GATES
    }
    if dict(sizes) != dict(spec.sizes) or audit.gate_sizes != dict(spec.sizes):
        raise ValidationGateInfeasibleError("validation gate exact sizes drifted")
    if audit.capability_counts != expected_capability_counts:
        raise ValidationGateInfeasibleError(
            "validation gate capability audit does not match its assignments"
        )
    if audit.boundary_counts != {
        gate: boundaries[gate] for gate in _VALIDATION_GATES
    }:
        raise ValidationGateInfeasibleError(
            "validation gate boundary audit does not match its assignments"
        )
    if audit.interaction_counts != expected_interaction_counts:
        raise ValidationGateInfeasibleError(
            "validation gate interaction audit does not match its assignments"
        )
    for gate in _VALIDATION_GATES:
        if any(
            capabilities[gate][capability] < spec.capability_minimums[gate]
            for capability in R2_CAPABILITY_ORDER
        ):
            raise ValidationGateInfeasibleError(
                f"validation gate capability minimum failed: {gate}"
            )
        if boundaries[gate] < spec.boundary_minimums[gate]:
            raise ValidationGateInfeasibleError(
                f"validation gate boundary minimum failed: {gate}"
            )
        if any(
            interaction_counts[gate][pattern] < minimum
            for pattern, minimum in spec.interaction_minimums[gate].items()
        ):
            raise ValidationGateInfeasibleError(
                f"validation gate interaction minimum failed: {gate}"
            )


def group_leakage_report(queries: list[Query]) -> dict[str, object]:
    violations: dict[str, dict[str, list[str]]] = {}
    field_counts: dict[str, int] = {}
    for field_name in GROUP_FIELDS:
        splits_by_value: dict[str, set[str]] = defaultdict(set)
        for query in queries:
            value = getattr(query, field_name)
            if value is not None:
                splits_by_value[value].add(query.split)
        field_counts[field_name] = len(splits_by_value)
        bad = {
            value: sorted(splits)
            for value, splits in splits_by_value.items()
            if len(splits) > 1
        }
        if bad:
            violations[field_name] = bad
    return {
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "group_fields": list(GROUP_FIELDS),
        "field_counts": field_counts,
        "violation_count": sum(len(values) for values in violations.values()),
        "violations": violations,
    }


def assert_no_group_leakage(queries: list[Query]) -> None:
    report = group_leakage_report(queries)
    if report["violation_count"]:
        raise GroupSplitError(f"group 跨 split 泄漏: {report['violations']}")


def _validate_plan_catalog_binding(
    *,
    plan_catalog_sha256: str | None,
    plan_leakage_policy_version: str,
    manifest_catalog_sha256: str | None,
    manifest_leakage_policy_version: str,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> None:
    if (
        plan_catalog_sha256 != manifest_catalog_sha256
        or plan_leakage_policy_version != manifest_leakage_policy_version
    ):
        raise ValueError("full plan 与 plan manifest 的 asset catalog 绑定不一致")

    if plan_catalog_sha256 is None:
        if plan_leakage_policy_version != PROVISIONAL_LEAKAGE_POLICY_VERSION:
            raise ValueError("未绑定 catalog 的 full plan 未标记 provisional policy")
        if asset_catalog is not None:
            raise ValueError("provisional full plan 不能事后绑定未记录的 asset catalog")
        if not allow_provisional_asset_groups:
            raise ValueError(
                "正式 split 禁止 provisional asset groups；"
                "仅兼容测试可显式 allow_provisional_asset_groups=True"
            )
        return

    if asset_catalog is None:
        raise ValueError("catalog-bound full plan 缺少匹配的 asset_catalog")
    if asset_catalog.manifest.catalog_sha256 != plan_catalog_sha256:
        raise ValueError("asset catalog hash 与 full plan 不匹配")
    if asset_catalog.manifest.leakage_policy_version != plan_leakage_policy_version:
        raise ValueError("asset catalog leakage policy 与 full plan 不匹配")


def _verify_catalog_references(
    asset_catalog: AssetCatalog,
    queries: list[Query],
    *,
    artifact: str,
) -> None:
    asset_catalog.require_verified_files()
    asset_catalog.verify_asset_ids(query.asset_id for query in queries)
    for query in queries:
        try:
            asset_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            )
        except ValueError as exc:
            raise ValueError(
                f"{artifact} 的 asset catalog 引用不一致: {query.query_id}: {exc}"
            ) from exc


def load_labeled_queries_for_split(
    queries_root: str | Path,
    full_plan_path: str | Path,
    *,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> tuple[list[Query], set[str]]:
    """Validate the label audit chain and return schema-v2 split candidates."""

    queries_root = Path(queries_root)
    accepted_verified = verify_accepted_corpus(queries_root)
    labels_dir = queries_root / "labels"
    try:
        manifest_bytes = (labels_dir / "manifest.json").read_bytes()
        manifest = LabelsManifest.model_validate_json(manifest_bytes)
        review_bytes = (labels_dir / "reviews.jsonl").read_bytes()
        queue_bytes = (labels_dir / "arbitration_queue.jsonl").read_bytes()
        arbitration_bytes = (labels_dir / "arbitrations.jsonl").read_bytes()
        labeled_bytes = (labels_dir / "labeled_queries.jsonl").read_bytes()
        accepted_bytes = (queries_root / "queries.jsonl").read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"标签产物不完整: {exc.filename}") from None
    except ValidationError as exc:
        raise ValueError(f"标签 manifest 校验失败: {exc}") from exc

    if manifest.query_schema_version != 2:
        raise ValueError("标签切分只接受 query_schema_version=2")
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ValueError("标签 manifest 不是 canonical JSON")
    if sha256_bytes(accepted_bytes) != manifest.accepted_source_sha256:
        raise ValueError("标签 accepted source 哈希与当前 accepted 派生文件不匹配")
    if sha256_bytes(review_bytes) != manifest.review_results_sha256:
        raise ValueError("标签 reviews 哈希与 manifest 不匹配")
    if sha256_bytes(queue_bytes) != manifest.arbitration_queue_sha256:
        raise ValueError("标签仲裁队列哈希与 manifest 不匹配")

    reviews = _parse_models(review_bytes, CrossReviewResult, "交叉审核记录")
    queue = _parse_models(queue_bytes, ArbitrationItem, "标签仲裁队列")
    arbitrations = _parse_models(arbitration_bytes, ArbitrationRecord, "标签仲裁记录")
    if review_bytes != canonical_jsonl_bytes(reviews):
        raise ValueError("标签 reviews 不是 canonical JSONL")
    if queue_bytes != canonical_jsonl_bytes(queue):
        raise ValueError("标签仲裁队列不是 canonical JSONL")
    validate_review_queue_semantics(accepted_verified, reviews, queue, manifest)
    queue_ids = [item.query_id for item in queue]
    arbitration_ids = [item.query_id for item in arbitrations]
    if len(queue_ids) != len(set(queue_ids)) or len(arbitration_ids) != len(
        set(arbitration_ids)
    ):
        raise ValueError("标签仲裁 query_id 重复")
    if not set(arbitration_ids) <= set(queue_ids):
        raise ValueError("标签仲裁记录包含未入队 query_id")
    unresolved = set(queue_ids) - set(arbitration_ids)
    if unresolved:
        raise ValueError(f"仲裁尚未完成，共 {len(unresolved)} 条")

    labeled = _parse_models(labeled_bytes, Query, "标签查询")
    accepted = accepted_verified
    if len(labeled) != manifest.count or manifest.count != 4500:
        raise ValueError("标签数量必须与 full accepted 4500 条一致")
    if any(query.label_status == "auto" for query in labeled):
        raise ValueError("标签查询仍包含未审核的 auto 状态")
    if {query.taxonomy_version for query in labeled} != {manifest.taxonomy_version}:
        raise ValueError("标签 taxonomy_version 与 manifest 不一致")
    _validate_query_capability_bindings(labeled)

    full_plan, full_manifest = load_plan(full_plan_path)
    if full_plan.scope != "full" or full_manifest.parent_plan_sha256 is None:
        raise ValueError("标签切分必须绑定合法 full plan")
    if (
        manifest.plan_sha256 != full_manifest.plan_sha256
        or manifest.asset_catalog_sha256 != full_manifest.asset_catalog_sha256
        or manifest.leakage_policy_version != full_manifest.leakage_policy_version
    ):
        raise ValueError("labels manifest 与 full plan/catalog binding 不一致")
    _validate_plan_catalog_binding(
        plan_catalog_sha256=full_plan.asset_catalog_sha256,
        plan_leakage_policy_version=full_plan.leakage_policy_version,
        manifest_catalog_sha256=full_manifest.asset_catalog_sha256,
        manifest_leakage_policy_version=full_manifest.leakage_policy_version,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    plan_by_id = {item.plan_id: item for item in full_plan.queries}
    labeled_by_id = _unique_queries(labeled, "标签")
    accepted_by_id = _unique_queries(accepted, "accepted")
    review_by_id = {review.query_id: review for review in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("交叉审核记录 query_id 重复")
    if set(labeled_by_id) != set(plan_by_id) or set(accepted_by_id) != set(plan_by_id):
        raise ValueError("标签、accepted 与 full plan 的 query_id 不完整")
    if set(review_by_id) != set(plan_by_id):
        raise ValueError("交叉审核记录与 full plan 的 query_id 不完整")
    arbitration_by_id = {record.query_id: record for record in arbitrations}
    for query_id, query in labeled_by_id.items():
        accepted_query = accepted_by_id[query_id]
        review = review_by_id[query_id]
        if review.constructed_intent != accepted_query.canonical_intent:
            raise ValueError(
                f"交叉审核 constructed_intent 与 accepted 不一致: {query_id}"
            )
        decision = arbitration_by_id.get(query_id)
        if decision is not None and (
            query.label_status != "arbitrated"
            or query.canonical_intent != decision.intent
            or query.canonical_capability != decision.canonical_capability
            or query.acceptable_capabilities != decision.acceptable_capabilities
            or query.requires_card != decision.requires_card
        ):
            raise ValueError(f"仲裁记录未同步到标签查询: {query_id}")
        if decision is None:
            if query.label_status == "arbitrated":
                raise ValueError(f"标签查询缺少对应仲裁记录: {query_id}")
            if (
                review.label_status != "cross_agreed"
                or review.needs_arbitration
                or query.label_status != "cross_agreed"
                or query.canonical_intent != review.constructed_intent
                or query.canonical_capability != accepted_query.canonical_capability
                or query.acceptable_capabilities
                != accepted_query.acceptable_capabilities
                or query.requires_card != accepted_query.requires_card
                or review.qwen_intent != review.constructed_intent
                or review.deepseek_intent != review.constructed_intent
            ):
                raise ValueError(f"未仲裁标签与交叉审核记录不一致: {query_id}")
    locked_ids = {item.plan_id for item in full_plan.queries[:200]}
    for query_id, query in labeled_by_id.items():
        plan_item = plan_by_id[query_id]
        accepted_query = accepted_by_id[query_id]
        static_pairs = (
            (query.taxonomy_version, plan_item.taxonomy_version),
            (query.task_spec_version, plan_item.task_spec_version),
            (query.asset_id, plan_item.asset_id),
            (query.image_path, plan_item.image_path),
            (query.leakage_group_id, plan_item.leakage_group_id),
            (query.boundary_group_id, plan_item.boundary_group_id),
            (query.template_family, plan_item.template_family),
            (query.generator_batch_id, plan_item.generator_batch_id),
            (query.is_boundary, plan_item.is_boundary),
            (query.boundary_strategy, plan_item.boundary_strategy),
        )
        if any(actual != expected for actual, expected in static_pairs):
            raise ValueError(f"标签查询与 full plan 静态字段不一致: {query_id}")
        accepted_asset_reference = (
            accepted_query.asset_id,
            accepted_query.image_path,
            accepted_query.leakage_group_id,
        )
        plan_asset_reference = (
            plan_item.asset_id,
            plan_item.image_path,
            plan_item.leakage_group_id,
        )
        if accepted_asset_reference != plan_asset_reference:
            raise ValueError(f"accepted 查询与 full plan asset 引用不一致: {query_id}")
        immutable_accepted_pairs = (
            (query.text, accepted_query.text),
            (query.turns, accepted_query.turns),
            (query.episode, accepted_query.episode),
            (query.synth_provider, accepted_query.synth_provider),
            (query.synth_model, accepted_query.synth_model),
            (query.synthesis_batch_id, accepted_query.synthesis_batch_id),
            (query.synthesis_prompt_id, accepted_query.synthesis_prompt_id),
            (query.seed_set_sha256, accepted_query.seed_set_sha256),
        )
        if any(actual != expected for actual, expected in immutable_accepted_pairs):
            raise ValueError(f"标签查询篡改了 accepted 的非标签字段: {query_id}")
        if asset_catalog is not None:
            try:
                asset_catalog.verify_reference(
                    query.asset_id,
                    query.image_path,
                    leakage_group_id=query.leakage_group_id,
                )
                asset_catalog.verify_reference(
                    accepted_query.asset_id,
                    accepted_query.image_path,
                    leakage_group_id=accepted_query.leakage_group_id,
                )
            except ValueError as exc:
                raise ValueError(
                    f"标签切分的 asset catalog 引用不一致: {query_id}: {exc}"
                ) from exc
        if query_id in locked_ids and query.split != "dev_mini":
            raise ValueError(f"标签查询破坏 locked dev_mini split: {query_id}")

    try:
        ledger_bytes = (queries_root / "accepted-ledger.jsonl").read_bytes()
    except FileNotFoundError:
        raise ValueError("标签切分缺少 accepted ledger") from None
    ledger = _parse_models(ledger_bytes, AcceptedLedgerEntry, "accepted ledger")
    expected_batches = list(dict.fromkeys(item.batch_id for item in full_plan.queries))
    if (
        len(ledger) != 180
        or {entry.base_batch_id for entry in ledger} != set(expected_batches)
        or any(entry.count != 25 for entry in ledger)
    ):
        raise ValueError("标签切分要求 180 个完整 accepted 批次")
    for entry in ledger:
        expected_hash = (
            full_manifest.parent_plan_sha256
            if entry.base_batch_id.startswith("dev-mini-")
            else full_manifest.plan_sha256
        )
        if entry.plan_sha256 != expected_hash:
            raise ValueError("标签切分的 accepted ledger plan 哈希不匹配")
    return labeled, locked_ids


def freeze_test_split(
    queries_root: str | Path,
    assigned: list[Query],
    *,
    seed: int,
    full_plan_path: str | Path | None = None,
    full_plan_sha256: str | None = None,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    split_spec: SplitSpec = FULL_SPLIT_SPEC,
) -> tuple[Path, Path]:
    """Freeze test_frozen and bind it to the matching profile plan/audit."""

    queries_root = Path(queries_root)
    assert_no_group_leakage(assigned)
    if full_plan_path is not None:
        full_plan, full_manifest = load_plan(full_plan_path)
        if full_plan.scope != split_spec.profile:
            raise ValueError(
                "冻结 split 必须绑定与 profile 匹配的 plan: "
                f"expected {split_spec.profile}, got {full_plan.scope}"
            )
        if (
            full_plan_sha256 is not None
            and full_plan_sha256 != full_manifest.plan_sha256
        ):
            raise ValueError("调用方 plan SHA-256 与实际 plan 不一致")
        full_plan_sha256 = full_manifest.plan_sha256
        _validate_plan_catalog_binding(
            plan_catalog_sha256=full_plan.asset_catalog_sha256,
            plan_leakage_policy_version=full_plan.leakage_policy_version,
            manifest_catalog_sha256=full_manifest.asset_catalog_sha256,
            manifest_leakage_policy_version=full_manifest.leakage_policy_version,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional_asset_groups,
        )
        _verify_assignment_against_full_plan(assigned, full_plan)
        if split_spec.profile == "full":
            authoritative, _ = load_labeled_queries_for_split(
                queries_root,
                full_plan_path,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional_asset_groups,
            )
            _verify_assignment_derived_from_labels(assigned, authoritative)
    elif split_spec.profile in {"full", "core"}:
        raise ValueError(
            "正式 full/core split 必须提供 full_plan_path，禁止自报 plan hash"
        )
    elif full_plan_sha256 is None:
        raise ValueError("非正式测试 profile 也必须提供 full_plan_sha256")
    assert full_plan_sha256 is not None
    if asset_catalog is None:
        if not allow_provisional_asset_groups:
            raise ValueError(
                "冻结 split 必须绑定 asset catalog；"
                "仅兼容测试可显式 allow_provisional_asset_groups=True"
            )
        asset_catalog_sha256 = None
        leakage_policy_version = PROVISIONAL_LEAKAGE_POLICY_VERSION
    else:
        asset_catalog_sha256 = asset_catalog.manifest.catalog_sha256
        leakage_policy_version = asset_catalog.manifest.leakage_policy_version
        _verify_catalog_references(
            asset_catalog,
            assigned,
            artifact="冻结 split assignment",
        )
    test_path = queries_root / "test_frozen.jsonl"
    assignment_path = queries_root / "split_assignment.jsonl"
    manifest_path = queries_root / "test_frozen.manifest.json"
    frozen = sorted(
        (query for query in assigned if query.split == "test_frozen"),
        key=lambda query: query.query_id,
    )
    expected_count = split_spec.sizes["test_frozen"]
    if len(frozen) != expected_count:
        raise ValueError(f"冻结前必须恰好有 {expected_count} 条 test_frozen")
    ordered_assignment = sorted(assigned, key=lambda query: query.query_id)
    assignment_bytes = canonical_jsonl_bytes(ordered_assignment)
    test_bytes = canonical_jsonl_bytes(frozen)
    intent_counts = Counter(query.canonical_intent for query in frozen)
    capability_counts = Counter(query.canonical_capability for query in frozen)
    taxonomy_versions = {query.taxonomy_version for query in assigned}
    if len(taxonomy_versions) != 1:
        raise ValueError("冻结 assignment 必须使用单一 taxonomy_version")
    groups = build_atomic_groups(assigned)
    manifest = FrozenSplitManifest(
        profile=split_spec.profile,
        split_sizes=split_spec.sizes,
        count=expected_count,
        sha256=sha256_bytes(test_bytes),
        assignment_sha256=sha256_bytes(assignment_bytes),
        seed=seed,
        taxonomy_version=next(iter(taxonomy_versions)),
        full_plan_sha256=full_plan_sha256,
        asset_catalog_sha256=asset_catalog_sha256,
        leakage_policy_version=leakage_policy_version,
        grouping_policy_version=GROUPING_POLICY_VERSION,
        group_fields=GROUP_FIELDS,
        component_count=len(groups),
        leakage_violations=0,
        intent_counts={
            intent: intent_counts[intent] for intent in sorted(intent_counts)
        },
        capability_counts={
            capability: capability_counts[capability]
            for capability in sorted(capability_counts)
            if capability is not None
        },
        boundary_count=sum(query.is_boundary for query in frozen),
    )
    manifest_bytes = canonical_json_bytes(manifest)

    existing = verify_frozen_split(
        queries_root,
        asset_catalog=asset_catalog,
        full_plan_path=full_plan_path,
        expected_full_plan_sha256=full_plan_sha256,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    if existing is not None:
        if (
            test_path.read_bytes() == test_bytes
            and assignment_path.read_bytes() == assignment_bytes
            and manifest_path.read_bytes() == manifest_bytes
        ):
            return test_path, manifest_path
        raise FrozenSplitError("现有冻结测试集与本次内容不同，拒绝覆盖")

    queries_root.mkdir(parents=True, exist_ok=True)
    atomic_create_file(test_path, test_bytes)
    try:
        atomic_create_file(assignment_path, assignment_bytes)
        atomic_create_file(manifest_path, manifest_bytes)
    except BaseException:
        test_path.unlink(missing_ok=True)
        assignment_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    os.chmod(test_path, stat.S_IREAD)
    os.chmod(assignment_path, stat.S_IREAD)
    os.chmod(manifest_path, stat.S_IREAD)
    return test_path, manifest_path


def verify_frozen_split(
    queries_root: str | Path,
    *,
    asset_catalog: AssetCatalog | None = None,
    full_plan_path: str | Path | None = None,
    expected_full_plan_sha256: str | None = None,
    allow_provisional_asset_groups: bool = False,
) -> FrozenSplitManifest | None:
    queries_root = Path(queries_root)
    test_path = queries_root / "test_frozen.jsonl"
    assignment_path = queries_root / "split_assignment.jsonl"
    manifest_path = queries_root / "test_frozen.manifest.json"
    if (
        not test_path.exists()
        and not assignment_path.exists()
        and not manifest_path.exists()
    ):
        return None
    if not test_path.is_file() or not manifest_path.is_file():
        raise FrozenSplitError("冻结测试集或 manifest 缺失")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = FrozenSplitManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise FrozenSplitError(f"冻结 manifest 校验失败: {exc}") from exc
    if manifest_bytes != canonical_json_bytes(manifest):
        raise FrozenSplitError("冻结 manifest 不是 canonical JSON")
    full_plan = None
    if full_plan_path is not None:
        try:
            full_plan, full_manifest = load_plan(full_plan_path)
        except (ValueError, FileNotFoundError) as exc:
            raise FrozenSplitError(f"冻结 split 的 full plan 无效: {exc}") from exc
        if full_plan.scope != manifest.profile:
            raise FrozenSplitError(
                "冻结 split 必须绑定与 manifest profile 匹配的 plan: "
                f"expected {manifest.profile}, got {full_plan.scope}"
            )
        if manifest.full_plan_sha256 != full_manifest.plan_sha256:
            raise FrozenSplitError("冻结 manifest 的 full plan hash 不匹配")
        try:
            _validate_plan_catalog_binding(
                plan_catalog_sha256=full_plan.asset_catalog_sha256,
                plan_leakage_policy_version=full_plan.leakage_policy_version,
                manifest_catalog_sha256=full_manifest.asset_catalog_sha256,
                manifest_leakage_policy_version=full_manifest.leakage_policy_version,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional_asset_groups,
            )
        except ValueError as exc:
            raise FrozenSplitError(str(exc)) from exc
    elif manifest.profile in {"full", "core"}:
        raise FrozenSplitError("正式 full/core 冻结 split 必须提供 full_plan_path 验证")
    if (
        expected_full_plan_sha256 is not None
        and manifest.full_plan_sha256 != expected_full_plan_sha256
    ):
        raise FrozenSplitError("冻结 manifest 的 full plan hash 不匹配")
    if manifest.asset_catalog_sha256 is None:
        if asset_catalog is not None:
            raise FrozenSplitError("provisional 冻结 split 不能事后绑定 asset catalog")
        if not allow_provisional_asset_groups:
            raise FrozenSplitError(
                "冻结 split 使用 provisional asset groups，必须显式允许"
            )
    elif asset_catalog is None:
        raise FrozenSplitError("catalog-bound 冻结 split 必须提供匹配的 asset catalog")
    else:
        if asset_catalog.manifest.catalog_sha256 != manifest.asset_catalog_sha256:
            raise FrozenSplitError("冻结 manifest 的 asset catalog hash 不匹配")
        if (
            asset_catalog.manifest.leakage_policy_version
            != manifest.leakage_policy_version
        ):
            raise FrozenSplitError("冻结 manifest 的 leakage policy 不匹配")
    content = test_path.read_bytes()
    if sha256_bytes(content) != manifest.sha256:
        raise FrozenSplitError("冻结测试集哈希不匹配")
    if not assignment_path.is_file():
        raise FrozenSplitError("冻结 split assignment 缺失")
    assignment_content = assignment_path.read_bytes()
    if sha256_bytes(assignment_content) != manifest.assignment_sha256:
        raise FrozenSplitError("冻结 split assignment 哈希不匹配")
    try:
        queries = _parse_models(content, Query, "冻结测试集")
        assignment = _parse_models(assignment_content, Query, "冻结 split assignment")
    except ValueError as exc:
        raise FrozenSplitError(str(exc)) from exc
    if len(queries) != manifest.count or any(
        query.split != "test_frozen" for query in queries
    ):
        raise FrozenSplitError("冻结测试集 count 或 split 不一致")
    if {query.taxonomy_version for query in queries} != {manifest.taxonomy_version}:
        raise FrozenSplitError("冻结测试集 taxonomy_version 不一致")
    assignment_counts = Counter(query.split for query in assignment)
    if assignment_counts != Counter(manifest.split_sizes):
        raise FrozenSplitError("冻结 split assignment 的 split_sizes 不一致")
    if {query.taxonomy_version for query in assignment} != {manifest.taxonomy_version}:
        raise FrozenSplitError("冻结 split assignment taxonomy_version 不一致")
    try:
        assert_no_group_leakage(assignment)
    except GroupSplitError as exc:
        raise FrozenSplitError(str(exc)) from exc
    if full_plan is not None:
        try:
            _verify_assignment_against_full_plan(assignment, full_plan)
            if manifest.profile == "full":
                authoritative, _ = load_labeled_queries_for_split(
                    queries_root,
                    full_plan_path,
                    asset_catalog=asset_catalog,
                    allow_provisional_asset_groups=allow_provisional_asset_groups,
                )
                _verify_assignment_derived_from_labels(
                    assignment,
                    authoritative,
                )
        except ValueError as exc:
            raise FrozenSplitError(str(exc)) from exc
    if asset_catalog is not None:
        try:
            _verify_catalog_references(
                asset_catalog,
                assignment,
                artifact="冻结 split assignment",
            )
        except ValueError as exc:
            raise FrozenSplitError(str(exc)) from exc
    if len(build_atomic_groups(assignment)) != manifest.component_count:
        raise FrozenSplitError("冻结 split assignment component_count 不一致")
    assignment_test = canonical_jsonl_bytes(
        sorted(
            (query for query in assignment if query.split == "test_frozen"),
            key=lambda query: query.query_id,
        )
    )
    if assignment_test != content:
        raise FrozenSplitError("冻结测试集与 split assignment 不一致")
    intent_counts = Counter(query.canonical_intent for query in queries)
    if dict(sorted(intent_counts.items())) != manifest.intent_counts:
        raise FrozenSplitError("冻结测试集 intent_counts 不一致")
    capability_counts = Counter(query.canonical_capability for query in queries)
    actual_capabilities = {
        capability: capability_counts[capability]
        for capability in sorted(capability_counts)
        if capability is not None
    }
    if actual_capabilities != manifest.capability_counts:
        raise FrozenSplitError("冻结测试集 capability_counts 不一致")
    if sum(query.is_boundary for query in queries) != manifest.boundary_count:
        raise FrozenSplitError("冻结测试集 boundary_count 不一致")
    if (
        test_path.stat().st_mode & stat.S_IWRITE
        or assignment_path.stat().st_mode & stat.S_IWRITE
        or manifest_path.stat().st_mode & stat.S_IWRITE
    ):
        raise FrozenSplitError("冻结文件只读标志已丢失")
    return manifest


def _verify_assignment_against_full_plan(assigned: list[Query], full_plan) -> None:
    plan_by_id = {item.plan_id: item for item in full_plan.queries}
    assigned_by_id = _unique_queries(assigned, "split assignment")
    if set(assigned_by_id) != set(plan_by_id):
        raise ValueError("split assignment 与 full plan 的 query_id 不完整")
    for query_id, query in assigned_by_id.items():
        item = plan_by_id[query_id]
        static_pairs = (
            (query.taxonomy_version, item.taxonomy_version),
            (query.task_spec_version, item.task_spec_version),
            (query.asset_id, item.asset_id),
            (query.image_path, item.image_path),
            (query.leakage_group_id, item.leakage_group_id),
            (query.boundary_group_id, item.boundary_group_id),
            (query.template_family, item.template_family),
            (query.generator_batch_id, item.generator_batch_id),
            (query.canonical_intent, item.canonical_intent),
            (query.canonical_capability, item.canonical_capability),
            (query.acceptable_capabilities, item.acceptable_capabilities),
            (query.requires_card, item.requires_card),
            (query.is_boundary, item.is_boundary),
            (query.boundary_strategy, item.boundary_strategy),
        )
        if any(actual != expected for actual, expected in static_pairs):
            raise ValueError(
                f"split assignment 与 full plan 静态字段不一致: {query_id}"
            )
        if item.provisional_split == "dev_mini" and query.split != "dev_mini":
            raise ValueError(
                f"split assignment 破坏 full plan locked dev_mini: {query_id}"
            )


def _verify_assignment_derived_from_labels(
    assigned: list[Query],
    authoritative: list[Query],
) -> None:
    assigned_by_id = _unique_queries(assigned, "split assignment")
    authoritative_by_id = _unique_queries(authoritative, "authoritative labels")
    if set(assigned_by_id) != set(authoritative_by_id):
        raise ValueError("split assignment 与 authoritative labels 的 query_id 不完整")
    for query_id, query in assigned_by_id.items():
        actual = query.model_dump(mode="json", exclude={"split"})
        expected = authoritative_by_id[query_id].model_dump(
            mode="json",
            exclude={"split"},
        )
        if actual != expected:
            raise ValueError(
                f"split assignment 不是 authoritative labels 的纯 split 派生: {query_id}"
            )


def _largest_remainder(sizes: dict[Stratum, int], target: int) -> dict[Stratum, int]:
    total = sum(sizes.values())
    if target < 0 or target > total or total <= 0:
        raise ValueError("最大余数分配的 target/total 非法")
    allocations = {key: target * size // total for key, size in sizes.items()}
    remainder_count = target - sum(allocations.values())
    ranked = sorted(
        sizes,
        key=lambda key: (-(target * sizes[key] % total), key[0], key[1]),
    )
    for key in ranked[:remainder_count]:
        allocations[key] += 1
    return allocations


def _parse_models(content: bytes, model_type, label: str) -> list:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 不是 UTF-8") from exc
    values = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{label} 第 {line_number} 行为空")
        try:
            values.append(model_type.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{label} 第 {line_number} 行校验失败: {exc}") from exc
    return values


def _unique_queries(queries: list[Query], label: str) -> dict[str, Query]:
    by_id = {query.query_id: query for query in queries}
    if len(by_id) != len(queries):
        raise ValueError(f"{label} query_id 重复")
    return by_id
