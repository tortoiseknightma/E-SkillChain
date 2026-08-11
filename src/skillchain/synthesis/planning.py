"""确定性查询图片规划；不调用模型，也不生成任何正式话术。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.optimize import Bounds, LinearConstraint, milp

from skillchain.data.asset_catalog import AssetCatalog
from skillchain.schemas import Intent, Split
from skillchain.synthesis.models import (
    ActivePlanPointer,
    CorpusPlan,
    PlanManifest,
    PlannedQuery,
    PROVISIONAL_LEAKAGE_POLICY_VERSION,
)
from skillchain.synthesis.store import atomic_replace_file, canonical_jsonl_bytes
from skillchain.task_spec import (
    MVP_TASK_SPEC_V1_SHA256,
    TaskSpecification,
    load_mvp_task_specification_v1,
)
from skillchain.taxonomy import (
    DEFAULT_TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    TaxonomyRegistry,
    assert_public_inputs_only,
    capability_for_intent,
    load_default_taxonomy_registry,
    requires_card_for_intent,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    parse_canonical_jsonl,
    read_stable_regular_file,
)

INTENT_ORDER: tuple[Intent, ...] = (
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
)

PHASE3_TASK_SPEC_VERSION = "ecommerce-task-spec-v1"
PHASE3_TASK_SPEC_SHA256 = MVP_TASK_SPEC_V1_SHA256

MVP_CAPABILITY_ORDER: tuple[str, ...] = (
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "knowledge.visual_encyclopedia",
    "utility.document_reading",
    "utility.recipe_guidance",
)

MVP_CAPABILITY_INTENTS: dict[str, Intent] = {
    "product.exact_match": "exact_match",
    "product.multi_search": "multi_product",
    "product.style_recommendation": "divergent_rec",
    "knowledge.visual_encyclopedia": "encyclopedia",
    "utility.document_reading": "utility",
    "utility.recipe_guidance": "utility",
}

DEV_MINI_CAPABILITY_COUNTS: dict[str, int] = {
    "product.exact_match": 35,
    "product.multi_search": 35,
    "product.style_recommendation": 35,
    "knowledge.visual_encyclopedia": 35,
    "utility.document_reading": 30,
    "utility.recipe_guidance": 30,
}

_BATCH_COUNTS_A = {
    "product.exact_match": 5,
    "product.multi_search": 5,
    "product.style_recommendation": 4,
    "knowledge.visual_encyclopedia": 4,
    "utility.document_reading": 4,
    "utility.recipe_guidance": 3,
}
_BATCH_COUNTS_B = {
    "product.exact_match": 4,
    "product.multi_search": 4,
    "product.style_recommendation": 5,
    "knowledge.visual_encyclopedia": 5,
    "utility.document_reading": 3,
    "utility.recipe_guidance": 4,
}
_BATCH_COUNTS_C = {
    "product.exact_match": 4,
    "product.multi_search": 4,
    "product.style_recommendation": 4,
    "knowledge.visual_encyclopedia": 4,
    "utility.document_reading": 4,
    "utility.recipe_guidance": 5,
}
_BATCH_COUNTS_D = {
    "product.exact_match": 4,
    "product.multi_search": 4,
    "product.style_recommendation": 5,
    "knowledge.visual_encyclopedia": 5,
    "utility.document_reading": 4,
    "utility.recipe_guidance": 3,
}

DEV_MINI_BATCH_CAPABILITY_COUNTS: tuple[dict[str, int], ...] = (
    *(_BATCH_COUNTS_A.copy() for _ in range(3)),
    *(_BATCH_COUNTS_B.copy() for _ in range(2)),
    *(_BATCH_COUNTS_C.copy() for _ in range(2)),
    _BATCH_COUNTS_D,
)

_CROSS_INTENT_TRIPLET: tuple[str, ...] = (
    "product.exact_match",
    "product.style_recommendation",
    "knowledge.visual_encyclopedia",
)

# 每批 3 条 cross-intent triplet 和 2 条 natural，共 40 条 boundary。
_NATURAL_BOUNDARY_CAPABILITY_SCHEDULE: tuple[tuple[str, ...], ...] = (
    ("product.exact_match", "product.multi_search"),
    ("product.style_recommendation", "knowledge.visual_encyclopedia"),
    ("utility.document_reading", "utility.recipe_guidance"),
    ("product.style_recommendation", "utility.recipe_guidance"),
    ("knowledge.visual_encyclopedia", "utility.document_reading"),
    ("product.exact_match", "product.multi_search"),
    ("knowledge.visual_encyclopedia", "utility.recipe_guidance"),
    ("product.exact_match", "utility.document_reading"),
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GROUPING_POLICY_VERSION = PROVISIONAL_LEAKAGE_POLICY_VERSION

FULL_INTENT_COUNTS: dict[Intent, int] = {
    "exact_match": 1125,
    "multi_product": 675,
    "divergent_rec": 900,
    "encyclopedia": 900,
    "utility": 900,
}

FULL_BOUNDARY_COUNTS: dict[Intent, int] = {
    "exact_match": 197,
    "multi_product": 118,
    "divergent_rec": 158,
    "encyclopedia": 158,
    "utility": 157,
}

# The portfolio core profile retains the frozen dev_mini prefix and scales the
# registered full-profile intent and boundary mix to 1,500 total queries.
CORE_INTENT_COUNTS: dict[Intent, int] = {
    "exact_match": 375,
    "multi_product": 225,
    "divergent_rec": 300,
    "encyclopedia": 300,
    "utility": 300,
}

CORE_BOUNDARY_COUNTS: dict[Intent, int] = {
    "exact_match": 66,
    "multi_product": 39,
    "divergent_rec": 53,
    "encyclopedia": 53,
    "utility": 52,
}

# r2 keeps the registered 1,500-query core intent totals, but freezes the
# more precise capability x final-split matrix before any authoring happens.
# The existing CorpusPlan retains its historical ``opt_pool`` provisional
# value for every tail record; the final split is held by the r2 in-memory
# sidecar and audited against these targets.
CORE_R2_FINAL_SPLITS: tuple[Split, ...] = (
    "dev_mini",
    "opt_pool",
    "val",
    "test_frozen",
)
CORE_R2_CAPABILITY_COUNTS: dict[Split, dict[str, int]] = {
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
CORE_R2_CAPABILITY_BOUNDARY_COUNTS: dict[Split, dict[str, int]] = {
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

CORE_R2_TAIL_BATCH_COUNT = 52
CORE_R2_BATCH_SIZE = 25
CORE_R2_MIN_CAPABILITY_COUNTS_PER_TAIL_BATCH: dict[str, int] = {
    "product.exact_match": 6,
    "product.multi_search": 3,
    "product.style_recommendation": 5,
    "knowledge.visual_encyclopedia": 5,
    "utility.document_reading": 1,
    "utility.recipe_guidance": 3,
}
CORE_R2_MIN_UNIQUE_COMPONENTS = 850

_R2_TAIL_SPLITS: tuple[Split, ...] = ("opt_pool", "val", "test_frozen")
_R2_REGULAR_CATEGORY_CAPABILITY: dict[str, str] = {
    "exact": "product.exact_match",
    "multi": "product.multi_search",
    "style": "product.style_recommendation",
    "encyclopedia": "knowledge.visual_encyclopedia",
    "document": "utility.document_reading",
    "recipe": "utility.recipe_guidance",
}
_R2_COMPONENT_PROFILES: dict[str, frozenset[str]] = {
    "exact": frozenset({"product.exact_match"}),
    "multi": frozenset({"product.multi_search"}),
    "style": frozenset({"product.style_recommendation"}),
    "encyclopedia": frozenset({"knowledge.visual_encyclopedia"}),
    "document": frozenset({"utility.document_reading"}),
    "recipe": frozenset({"utility.recipe_guidance"}),
    "abo": frozenset(
        {
            "product.exact_match",
            "product.style_recommendation",
            "knowledge.visual_encyclopedia",
        }
    ),
    "food": frozenset(
        {"knowledge.visual_encyclopedia", "utility.recipe_guidance"}
    ),
}
_R2_REQUIRED_NONDEV_COMPONENT_COUNTS: dict[str, int] = {
    "exact": 136,
    "multi": 90,
    "style": 102,
    "encyclopedia": 95,
    "document": 65,
    "recipe": 170,
    "abo": 20,
    "food": 10,
}
# Every selected regular component is used.  Some extra 3-use / 1-use pairs
# can be necessary to express a capability's per-batch residual capacities.
# The planner searches these possibilities in increasing triple count and
# records the first feasible (therefore minimal) capacity-required solution.
_R2_REGULAR_COMPONENT_USE_COUNTS: dict[str, int] = {
    "exact": 136,
    "multi": 90,
    "style": 102,
    "encyclopedia": 95,
    "document": 60,
    "recipe": 170,
}
_R2_REGULAR_QUERY_SLOT_COUNTS: dict[str, int] = {
    "exact": 320,
    "multi": 190,
    "style": 245,
    "encyclopedia": 235,
    "document": 60,
    "recipe": 170,
}
_R2_MIN_CAPACITY_REQUIRED_TRIPLES: dict[str, int] = {
    "exact": 48,
    "multi": 14,
    "style": 41,
    "encyclopedia": 45,
    "document": 0,
    "recipe": 0,
}
_R2_REGULAR_USAGE_COUNTS: dict[str, dict[int, int]] = {
    "exact": {3: 48, 2: 88},
    "multi": {3: 14, 2: 72, 1: 4},
    "style": {3: 41, 2: 61},
    "encyclopedia": {3: 45, 2: 50},
    "document": {1: 60},
    "recipe": {1: 170},
}
_R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS: dict[Split, dict[str, int]] = {
    "opt_pool": {"abo": 12, "food": 6},
    "val": {"abo": 3, "food": 2},
    "test_frozen": {"abo": 5, "food": 2},
}

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CapabilityAssignmentKey = tuple[str, str, Intent]
_MAX_CAPABILITY_ASSIGNMENT_BYTES = 64 * 1024 * 1024


class CapabilityAssignment(BaseModel):
    """One pre-generation, public-input capability decision for an asset/intent.

    The record deliberately carries the frozen taxonomy/Task Specification
    identities and the canonical tool allowlist.  A planner may copy the
    resolved label fields into a plan, but it may not derive a formal label
    from the intent-only primary compatibility mapping.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    assignment_id: str
    asset_id: str
    image_path: str
    canonical_intent: Intent
    canonical_capability: str
    acceptable_capabilities: tuple[str, ...]
    requires_card: bool
    allowed_tools: tuple[str, ...]
    taxonomy_version: Literal["ecommerce-mvp-taxonomy-v0"] = TAXONOMY_VERSION
    taxonomy_sha256: Sha256
    task_spec_version: Literal[
        "ecommerce-task-spec-v0",
        "ecommerce-task-spec-v1",
    ] = PHASE3_TASK_SPEC_VERSION
    task_spec_sha256: Sha256
    source_ref: str
    source_artifact_sha256: Sha256
    rationale: str

    @field_validator("acceptable_capabilities", "allowed_tools", mode="before")
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "assignment_id",
        "asset_id",
        "image_path",
        "canonical_capability",
        "source_ref",
        "rationale",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank without whitespace")
        if info.field_name == "image_path" and Path(value).as_posix() != value:
            raise ValueError("image_path must use normalized POSIX separators")
        return value

    @field_validator("acceptable_capabilities", "allowed_tools")
    @classmethod
    def validate_ordered_unique_tuple(
        cls, value: tuple[str, ...], info
    ) -> tuple[str, ...]:
        if not value or value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be non-empty, sorted, and unique")
        if any(not item or item != item.strip() for item in value):
            raise ValueError(f"{info.field_name} contains a blank or padded value")
        return value

    @model_validator(mode="after")
    def validate_record(self):
        if self.canonical_capability not in self.acceptable_capabilities:
            raise ValueError("canonical_capability must be acceptable")
        assert_public_inputs_only(
            {
                "assignment_id": self.assignment_id,
                "source_ref": self.source_ref,
                "rationale": self.rationale,
            },
            "capability assignment",
        )
        return self


class DocumentPathRebindEntry(BaseModel):
    """One immutable r1-to-r2 Document Reading path rebinding decision."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    prefix_index: int = Field(ge=1, le=200)
    plan_id: str
    asset_id: str
    sha256: Sha256
    leakage_group_id: str
    canonical_intent: Literal["utility"] = "utility"
    canonical_capability: Literal["utility.document_reading"] = (
        "utility.document_reading"
    )
    old_image_path: str
    new_image_path: str
    old_assignment_id: str
    new_assignment_id: str
    old_assignment_source_sha256: Sha256
    new_assignment_source_sha256: Sha256

    @field_validator(
        "plan_id",
        "asset_id",
        "leakage_group_id",
        "old_image_path",
        "new_image_path",
        "old_assignment_id",
        "new_assignment_id",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank without whitespace")
        if info.field_name.endswith("image_path") and Path(value).as_posix() != value:
            raise ValueError(f"{info.field_name} must use normalized POSIX separators")
        return value

    @model_validator(mode="after")
    def validate_path_change(self):
        if self.old_image_path == self.new_image_path:
            raise ValueError("Document Reading rebind must change image_path")
        return self


class DocumentPathRebindManifest(BaseModel):
    """In-memory, fail-closed proof for the 30 carried Document assets."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    parent_core_plan_sha256: Sha256
    parent_catalog_sha256: Sha256
    parent_capability_assignments_sha256: Sha256
    target_catalog_sha256: Sha256
    target_capability_assignments_sha256: Sha256
    entries: tuple[DocumentPathRebindEntry, ...]

    @model_validator(mode="after")
    def validate_entries(self):
        if len(self.entries) != DEV_MINI_CAPABILITY_COUNTS[
            "utility.document_reading"
        ]:
            raise ValueError("Document rebind manifest must contain exactly 30 entries")
        indices = tuple(entry.prefix_index for entry in self.entries)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError(
                "Document rebind manifest entries must be uniquely ordered by prefix_index"
            )
        asset_ids = [entry.asset_id for entry in self.entries]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Document rebind manifest must bind 30 distinct assets")
        if len({entry.plan_id for entry in self.entries}) != len(self.entries):
            raise ValueError("Document rebind manifest plan_ids must be unique")
        return self


@dataclass(frozen=True)
class R2DevPrefixRebind:
    """The only r2 dev-prefix transformation permitted before tail planning."""

    dev_prefix: CorpusPlan
    document_path_rebind_manifest: DocumentPathRebindManifest


@dataclass(frozen=True)
class _R2Component:
    component_id: str
    bindings: Mapping[str, CapabilityAssignment]


@dataclass(frozen=True)
class _R2Bundle:
    component_id: str
    capability: str
    usage_count: int
    category: str

    @property
    def reuse_reason(self) -> str:
        if self.usage_count == 3:
            return "capacity_required"
        if self.usage_count == 2:
            return "repeat"
        return "singleton"


@dataclass
class _R2TailSlot:
    component_id: str
    capability: str
    assignment: CapabilityAssignment
    final_split: Split
    reuse_variant: str
    reuse_reason: str
    is_boundary: bool = False
    boundary_strategy: str | None = None
    boundary_group_id: str | None = None


@dataclass(frozen=True)
class _R2JointBatchSolution:
    """One exact, batch-atomic tail packing returned by the joint MILP.

    ``abo_by_batch`` and ``food_by_batch`` count semantic counterfactual
    components, while ``regular_bundle_counts_by_batch`` counts normal
    components by their within-batch reuse cardinality.  Keeping these facts
    separate makes the later slot materialisation a pure, auditable mapping
    from a solved integer layout rather than a second scheduler.
    """

    batch_final_splits: tuple[Split, ...]
    abo_by_batch: tuple[int, ...]
    food_by_batch: tuple[int, ...]
    regular_bundle_counts_by_batch: tuple[Mapping[str, Mapping[int, int]], ...]
    capability_counts_by_batch: tuple[Mapping[str, int], ...]
    stacked_counterfactual_batch_count: int


@dataclass(frozen=True)
class R2CorePlanAudit:
    """Compact exact counts for an in-memory r2 core planner result."""

    capability_counts_by_split: Mapping[Split, Mapping[str, int]]
    boundary_counts_by_split: Mapping[Split, Mapping[str, int]]
    tail_batch_capability_counts: Mapping[str, Mapping[str, int]]
    unique_component_count: int
    max_component_reuse: int
    capacity_required_triples_by_capability: Mapping[str, int]
    counterfactual_components_by_split: Mapping[Split, int]
    tail_multi_component_count: int


@dataclass(frozen=True)
class R2CoreInMemoryPlan:
    """A locked r2 layout plus non-serialized final-split/reuse sidecars."""

    plan: CorpusPlan
    dev_prefix_rebind: R2DevPrefixRebind
    final_split_by_plan_id: Mapping[str, Split]
    reuse_variant_by_plan_id: Mapping[str, str]
    reuse_reason_by_plan_id: Mapping[str, str]
    audit: R2CorePlanAudit


def load_capability_assignments(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[CapabilityAssignment, ...]:
    """Load canonical assignments bound to an independently frozen digest."""

    _require_sha256(expected_sha256, "expected capability assignments SHA-256")

    path = Path(path)
    try:
        content = read_stable_regular_file(
            path,
            label="capability assignment file",
            max_bytes=_MAX_CAPABILITY_ASSIGNMENT_BYTES,
        )
        rows = parse_canonical_jsonl(content, label="capability assignment file")
    except ArtifactFormatError as exc:
        raise ValueError(str(exc)) from exc
    if not rows:
        raise ValueError("capability assignment file must be non-empty canonical JSONL")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "capability assignment file does not match the external expected digest"
        )
    assignments: list[CapabilityAssignment] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            assignments.append(CapabilityAssignment.model_validate(row, strict=True))
        except ValidationError as exc:
            raise ValueError(
                f"capability assignment line {line_number} violates schema: {exc}"
            ) from exc
    if content != canonical_jsonl_bytes(assignments):  # pragma: no cover - parser guard
        raise ValueError("capability assignment file changed during validation")
    _index_capability_assignments(assignments)
    return tuple(assignments)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _capability_assignments_digest(
    assignments: Sequence[CapabilityAssignment] | None,
    *,
    expected_sha256: str | None,
    formal: bool,
) -> str | None:
    """Bind in-memory records to the caller-owned digest used by the formal job."""

    if assignments is None:
        if formal or expected_sha256 is not None:
            raise ValueError(
                "formal planning requires externally locked capability assignments"
            )
        return None
    if not formal:
        raise ValueError(
            "capability assignments are only accepted with a verified asset catalog"
        )
    if expected_sha256 is None:
        raise ValueError(
            "formal planning requires expected_capability_assignments_sha256"
        )
    _require_sha256(expected_sha256, "expected capability assignments SHA-256")
    actual_sha256 = hashlib.sha256(canonical_jsonl_bytes(assignments)).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "capability assignments do not match the external expected digest"
        )
    return actual_sha256


def validate_capability_binding(
    *,
    taxonomy_version: str,
    task_spec_version: str,
    canonical_intent: Intent,
    canonical_capability: str,
    acceptable_capabilities: Sequence[str],
    requires_card: bool,
    allowed_tools: Sequence[str] | None = None,
    taxonomy: TaxonomyRegistry | None = None,
    task_specification: TaskSpecification | None = None,
) -> None:
    """Validate a resolved label against all frozen capability contracts."""

    taxonomy = taxonomy or load_default_taxonomy_registry()
    task_specification = task_specification or load_mvp_task_specification_v1()
    if taxonomy_version != taxonomy.taxonomy_version:
        raise ValueError("capability binding taxonomy_version mismatch")
    if task_spec_version != task_specification.task_spec_version:
        raise ValueError("capability binding task_spec_version mismatch")
    acceptable = tuple(acceptable_capabilities)
    if not acceptable or len(acceptable) != len(set(acceptable)):
        raise ValueError("acceptable_capabilities must be non-empty and unique")
    if canonical_capability not in acceptable:
        raise ValueError("canonical_capability must be in acceptable_capabilities")

    taxonomy_capabilities = taxonomy.capabilities_by_id
    task_capabilities = task_specification.capabilities_by_id
    card_requirements: set[bool] = set()
    for capability_id in acceptable:
        capability = taxonomy_capabilities.get(capability_id)
        if capability is None:
            raise ValueError(f"unknown capability in assignment: {capability_id}")
        if capability.intent_id != canonical_intent:
            raise ValueError(
                f"capability {capability_id} does not belong to intent "
                f"{canonical_intent}"
            )
        task = task_capabilities.get(capability_id)
        if task is None or not task.allowed_tools:
            raise ValueError(
                f"capability {capability_id} is not allowed by the Task Specification"
            )
        task_requires_card = task.output_contract.card_requirement == "required"
        if task_requires_card != capability.requires_card:
            raise ValueError(
                f"taxonomy/Task Specification card mismatch for {capability_id}"
            )
        card_requirements.add(capability.requires_card)
    if card_requirements != {requires_card}:
        raise ValueError(
            "requires_card does not match every acceptable capability contract"
        )
    canonical_task = task_capabilities[canonical_capability]
    if (
        allowed_tools is not None
        and tuple(allowed_tools) != canonical_task.allowed_tools
    ):
        raise ValueError(
            "allowed_tools does not match the canonical capability Task Specification"
        )


def _index_capability_assignments(
    assignments: Sequence[CapabilityAssignment],
) -> dict[CapabilityAssignmentKey, CapabilityAssignment]:
    taxonomy = load_default_taxonomy_registry()
    task_specification = load_mvp_task_specification_v1()
    indexed: dict[CapabilityAssignmentKey, CapabilityAssignment] = {}
    assignment_ids: set[str] = set()
    for raw_assignment in assignments:
        try:
            assignment = CapabilityAssignment.model_validate(
                raw_assignment.model_dump(mode="json"), strict=True
            )
        except (AttributeError, ValidationError) as exc:
            raise ValueError("capability assignment violates schema") from exc
        if assignment.assignment_id in assignment_ids:
            raise ValueError(
                f"duplicate capability assignment_id: {assignment.assignment_id}"
            )
        assignment_ids.add(assignment.assignment_id)
        if assignment.taxonomy_sha256 != DEFAULT_TAXONOMY_SHA256:
            raise ValueError("capability assignment taxonomy SHA-256 mismatch")
        if assignment.task_spec_sha256 != PHASE3_TASK_SPEC_SHA256:
            raise ValueError(
                "capability assignment Task Specification SHA-256 mismatch"
            )
        validate_capability_binding(
            taxonomy_version=assignment.taxonomy_version,
            task_spec_version=assignment.task_spec_version,
            canonical_intent=assignment.canonical_intent,
            canonical_capability=assignment.canonical_capability,
            acceptable_capabilities=assignment.acceptable_capabilities,
            requires_card=assignment.requires_card,
            allowed_tools=assignment.allowed_tools,
            taxonomy=taxonomy,
            task_specification=task_specification,
        )
        key = (
            assignment.asset_id,
            assignment.image_path,
            assignment.canonical_intent,
        )
        if key in indexed:
            raise ValueError(
                "duplicate capability assignment for asset/path/intent: "
                f"{assignment.asset_id}, {assignment.image_path}, "
                f"{assignment.canonical_intent}"
            )
        indexed[key] = assignment
    return indexed


def _audit_planned_assignment_binding(
    queries: Sequence[PlannedQuery],
    assignments: Mapping[CapabilityAssignmentKey, CapabilityAssignment] | None,
    *,
    formal: bool,
) -> None:
    if assignments is None:
        if formal:
            raise ValueError(
                "formal planning requires explicit per-asset capability assignments"
            )
        return
    for query in queries:
        key = (query.asset_id, query.image_path, query.canonical_intent)
        assignment = assignments.get(key)
        if assignment is None:
            raise ValueError(
                "missing capability assignment for planned asset/path/intent: "
                f"{query.asset_id}, {query.image_path}, {query.canonical_intent}"
            )
        if (
            query.canonical_capability != assignment.canonical_capability
            or tuple(query.acceptable_capabilities)
            != assignment.acceptable_capabilities
            or query.requires_card != assignment.requires_card
            or query.capability_assignment_id != assignment.assignment_id
            or query.capability_assignment_source_sha256
            != assignment.source_artifact_sha256
        ):
            raise ValueError(
                f"planned capability fields drifted from assignment: {query.plan_id}"
            )


def _canonical_plan_sha256(plan: CorpusPlan) -> str:
    return hashlib.sha256(_canonical_json_bytes(plan.model_dump(mode="json"))).hexdigest()


def _project_dev_prefix(parent_core_plan: CorpusPlan) -> CorpusPlan:
    """Project and validate the immutable first 200 records of an r1 core plan."""

    if parent_core_plan.scope != "core" or len(parent_core_plan.queries) != 1500:
        raise ValueError("r2 rebind requires the complete 1,500-query r1 core plan")
    prefix = parent_core_plan.queries[:200]
    if any(item.provisional_split != "dev_mini" for item in prefix):
        raise ValueError("r1 core prefix must retain 200 dev_mini records")
    return CorpusPlan(
        scope="dev_mini",
        seed=parent_core_plan.seed,
        asset_catalog_sha256=parent_core_plan.asset_catalog_sha256,
        capability_assignments_sha256=(
            parent_core_plan.capability_assignments_sha256
        ),
        leakage_policy_version=parent_core_plan.leakage_policy_version,
        queries=prefix,
    )


def _assignment_semantic_key(
    assignment: CapabilityAssignment,
) -> tuple[object, ...]:
    """Fields whose drift would alter how an asset is labelled or executed."""

    return (
        assignment.canonical_intent,
        assignment.canonical_capability,
        assignment.acceptable_capabilities,
        assignment.requires_card,
        assignment.allowed_tools,
        assignment.taxonomy_version,
        assignment.taxonomy_sha256,
        assignment.task_spec_version,
        assignment.task_spec_sha256,
    )


def _assert_query_matches_assignment(
    query: PlannedQuery,
    assignment: CapabilityAssignment,
    *,
    label: str,
) -> None:
    expected = (
        assignment.canonical_intent,
        assignment.canonical_capability,
        assignment.acceptable_capabilities,
        assignment.requires_card,
        assignment.taxonomy_version,
        assignment.task_spec_version,
        assignment.assignment_id,
        assignment.source_artifact_sha256,
    )
    actual = (
        query.canonical_intent,
        query.canonical_capability,
        tuple(query.acceptable_capabilities),
        query.requires_card,
        query.taxonomy_version,
        query.task_spec_version,
        query.capability_assignment_id,
        query.capability_assignment_source_sha256,
    )
    if actual != expected:
        raise ValueError(
            f"{label} planned capability semantics drifted from assignment: "
            f"{query.plan_id}"
        )


def rebind_r2_dev_prefix(
    parent_core_plan: CorpusPlan,
    *,
    parent_core_plan_sha256: str,
    parent_catalog: AssetCatalog,
    expected_parent_catalog_sha256: str,
    parent_capability_assignments: Sequence[CapabilityAssignment],
    expected_parent_capability_assignments_sha256: str,
    target_catalog: AssetCatalog,
    expected_target_catalog_sha256: str,
    target_capability_assignments: Sequence[CapabilityAssignment],
    expected_target_capability_assignments_sha256: str,
) -> R2DevPrefixRebind:
    """Rebind only the carried Document paths from r1 to the locked r2 catalog.

    This intentionally accepts a complete r1 *core* plan rather than a freshly
    discovered image folder.  The first 200 records retain their plan IDs,
    order, batches and all semantics.  The only legal path change is the known
    Document Reading rebase; all other paths must remain byte-for-byte stable.
    """

    _require_sha256(parent_core_plan_sha256, "parent core plan SHA-256")
    _require_sha256(expected_parent_catalog_sha256, "parent catalog SHA-256")
    _require_sha256(
        expected_parent_capability_assignments_sha256,
        "parent capability assignments SHA-256",
    )
    _require_sha256(expected_target_catalog_sha256, "target catalog SHA-256")
    _require_sha256(
        expected_target_capability_assignments_sha256,
        "target capability assignments SHA-256",
    )
    actual_parent_plan_sha256 = _canonical_plan_sha256(parent_core_plan)
    if actual_parent_plan_sha256 != parent_core_plan_sha256:
        raise ValueError("parent core plan does not match the expected SHA-256")
    if parent_catalog.manifest.catalog_sha256 != expected_parent_catalog_sha256:
        raise ValueError("loaded parent catalog does not match the expected SHA-256")
    if target_catalog.manifest.catalog_sha256 != expected_target_catalog_sha256:
        raise ValueError("loaded target catalog does not match the expected SHA-256")

    parent_assignment_sha256 = _capability_assignments_digest(
        parent_capability_assignments,
        expected_sha256=expected_parent_capability_assignments_sha256,
        formal=True,
    )
    target_assignment_sha256 = _capability_assignments_digest(
        target_capability_assignments,
        expected_sha256=expected_target_capability_assignments_sha256,
        formal=True,
    )
    assert parent_assignment_sha256 is not None
    assert target_assignment_sha256 is not None

    parent_prefix = _project_dev_prefix(parent_core_plan)
    if (
        parent_prefix.asset_catalog_sha256 != expected_parent_catalog_sha256
        or parent_prefix.capability_assignments_sha256 != parent_assignment_sha256
    ):
        raise ValueError("parent core plan catalog or assignment binding drifted")
    audit_plan_catalog_binding(parent_prefix, parent_catalog)

    parent_assignment_index = _index_capability_assignments(
        parent_capability_assignments
    )
    target_assignment_index = _index_capability_assignments(
        target_capability_assignments
    )
    _audit_planned_assignment_binding(
        parent_prefix.queries,
        parent_assignment_index,
        formal=True,
    )

    target_queries: list[PlannedQuery] = []
    document_entries: list[DocumentPathRebindEntry] = []
    for prefix_index, source in enumerate(parent_prefix.queries, start=1):
        parent_resolution = parent_catalog.verify_reference(
            asset_id=source.asset_id,
            image_path=source.image_path,
            leakage_group_id=source.leakage_group_id,
        )
        parent_assignment = parent_assignment_index.get(
            (source.asset_id, source.image_path, source.canonical_intent)
        )
        if parent_assignment is None:  # defensive; the audit above is explicit.
            raise ValueError(f"missing parent assignment for {source.plan_id}")
        _assert_query_matches_assignment(
            source,
            parent_assignment,
            label="parent",
        )

        target_resolution = target_catalog.resolve_asset_id(source.asset_id)
        target_path = target_resolution.asset.local_path
        target_catalog.verify_reference(
            asset_id=source.asset_id,
            image_path=target_path,
            leakage_group_id=target_resolution.leakage_group_id,
        )
        if parent_resolution.asset.sha256 != target_resolution.asset.sha256:
            raise ValueError(
                f"r2 rebind SHA-256 drift for {source.plan_id}: {source.asset_id}"
            )
        if parent_resolution.leakage_group_id != target_resolution.leakage_group_id:
            raise ValueError(
                f"r2 rebind leakage component drift for {source.plan_id}: "
                f"{source.asset_id}"
            )

        target_assignment = target_assignment_index.get(
            (source.asset_id, target_path, source.canonical_intent)
        )
        if target_assignment is None:
            raise ValueError(
                "missing target capability assignment for re-bound asset/path/intent: "
                f"{source.asset_id}, {target_path}, {source.canonical_intent}"
            )
        if _assignment_semantic_key(parent_assignment) != _assignment_semantic_key(
            target_assignment
        ):
            raise ValueError(
                f"r2 rebind capability/tools/card semantics drifted: {source.plan_id}"
            )

        is_document = source.canonical_capability == "utility.document_reading"
        if is_document:
            if target_path == source.image_path:
                raise ValueError(
                    f"Document Reading path did not rebind for {source.plan_id}"
                )
            document_entries.append(
                DocumentPathRebindEntry(
                    prefix_index=prefix_index,
                    plan_id=source.plan_id,
                    asset_id=source.asset_id,
                    sha256=parent_resolution.asset.sha256,
                    leakage_group_id=source.leakage_group_id,
                    old_image_path=source.image_path,
                    new_image_path=target_path,
                    old_assignment_id=parent_assignment.assignment_id,
                    new_assignment_id=target_assignment.assignment_id,
                    old_assignment_source_sha256=(
                        parent_assignment.source_artifact_sha256
                    ),
                    new_assignment_source_sha256=(
                        target_assignment.source_artifact_sha256
                    ),
                )
            )
        elif target_path != source.image_path:
            raise ValueError(
                f"non-Document path changed during r2 rebind: {source.plan_id}"
            )

        replacement = source.model_dump(mode="python")
        replacement.update(
            {
                "asset_id": target_resolution.asset.asset_id,
                "image_path": target_path,
                "leakage_group_id": target_resolution.leakage_group_id,
                "taxonomy_version": target_assignment.taxonomy_version,
                "task_spec_version": target_assignment.task_spec_version,
                "canonical_intent": target_assignment.canonical_intent,
                "canonical_capability": target_assignment.canonical_capability,
                "acceptable_capabilities": list(
                    target_assignment.acceptable_capabilities
                ),
                "requires_card": target_assignment.requires_card,
                "capability_assignment_id": target_assignment.assignment_id,
                "capability_assignment_source_sha256": (
                    target_assignment.source_artifact_sha256
                ),
            }
        )
        target_queries.append(PlannedQuery.model_validate(replacement, strict=True))

    document_manifest = DocumentPathRebindManifest(
        parent_core_plan_sha256=parent_core_plan_sha256,
        parent_catalog_sha256=expected_parent_catalog_sha256,
        parent_capability_assignments_sha256=parent_assignment_sha256,
        target_catalog_sha256=expected_target_catalog_sha256,
        target_capability_assignments_sha256=target_assignment_sha256,
        entries=tuple(document_entries),
    )
    target_prefix = CorpusPlan(
        scope="dev_mini",
        seed=parent_core_plan.seed,
        asset_catalog_sha256=expected_target_catalog_sha256,
        capability_assignments_sha256=target_assignment_sha256,
        leakage_policy_version=target_catalog.manifest.leakage_policy_version,
        queries=target_queries,
    )
    audit_plan_catalog_binding(target_prefix, target_catalog)
    _audit_planned_assignment_binding(
        target_prefix.queries,
        target_assignment_index,
        formal=True,
    )
    return R2DevPrefixRebind(
        dev_prefix=target_prefix,
        document_path_rebind_manifest=document_manifest,
    )


def _r2_tail_capability_totals() -> dict[str, int]:
    return {
        capability: sum(
            CORE_R2_CAPABILITY_COUNTS[split][capability]
            for split in _R2_TAIL_SPLITS
        )
        for capability in MVP_CAPABILITY_ORDER
    }


def _r2_tail_batch_final_splits() -> tuple[Split, ...]:
    """Return the fixed 32/8/12 split-atomic tail batch order."""

    batch_splits: list[Split] = []
    for split in _R2_TAIL_SPLITS:
        split_total = sum(CORE_R2_CAPABILITY_COUNTS[split].values())
        if split_total % CORE_R2_BATCH_SIZE:
            raise ValueError(f"r2 {split} total cannot form whole 25-query batches")
        batch_splits.extend([split] * (split_total // CORE_R2_BATCH_SIZE))
    if len(batch_splits) != CORE_R2_TAIL_BATCH_COUNT:
        raise AssertionError("r2 split schedule must contain exactly 52 tail batches")
    return tuple(batch_splits)


def _r2_deterministic_capability_schedule() -> tuple[
    tuple[Split, ...], tuple[dict[str, int], ...]
]:
    """Freeze the floor/ceil schedule before choosing any component IDs.

    It is deliberately a pure Python largest-deficit construction.  The joint
    MILP below proves that the fixed schedule can be filled by the locked
    component-reuse inventory; it does not get to change a batch's semantic
    quota just because an alternative packing would be easier.
    """

    lower, _ = _r2_capability_batch_bounds()
    batch_splits = _r2_tail_batch_final_splits()
    schedule: list[dict[str, int]] = []
    capability_rank = {
        capability: index for index, capability in enumerate(MVP_CAPABILITY_ORDER)
    }
    cursor = 0
    for split in _R2_TAIL_SPLITS:
        batch_count = sum(1 for value in batch_splits if value == split)
        targets = CORE_R2_CAPABILITY_COUNTS[split]
        remaining = {
            capability: targets[capability] - lower[capability] * batch_count
            for capability in MVP_CAPABILITY_ORDER
        }
        if any(value < 0 for value in remaining.values()):
            raise ValueError(
                f"r2 {split} cannot satisfy the per-batch capability floors"
            )
        if sum(remaining.values()) != 2 * batch_count:
            raise ValueError(
                f"r2 {split} schedule does not leave exactly two extras per batch"
            )
        for local_index in range(batch_count):
            batches_including_current = batch_count - local_index
            forced = [
                capability
                for capability in MVP_CAPABILITY_ORDER
                if remaining[capability] == batches_including_current
            ]
            if len(forced) > 2:
                raise ValueError(
                    f"r2 {split} deterministic schedule has too many forced extras"
                )
            extras = list(forced)
            candidates = sorted(
                (
                    capability
                    for capability in MVP_CAPABILITY_ORDER
                    if capability not in extras and remaining[capability] > 0
                ),
                key=lambda capability: (-remaining[capability], capability_rank[capability]),
            )
            extras.extend(candidates[: 2 - len(extras)])
            if len(extras) != 2:
                raise ValueError(f"r2 {split} deterministic schedule cannot fill extras")
            counts = dict(lower)
            for capability in extras:
                counts[capability] += 1
                remaining[capability] -= 1
            if any(value < 0 for value in remaining.values()):
                raise AssertionError("r2 deterministic schedule over-allocated extras")
            schedule.append(counts)
            cursor += 1
        if any(remaining.values()):
            raise AssertionError("r2 deterministic schedule left extras unallocated")
    if cursor != CORE_R2_TAIL_BATCH_COUNT:
        raise AssertionError("r2 deterministic schedule has the wrong batch count")
    for split in _R2_TAIL_SPLITS:
        indices = [
            index for index, value in enumerate(batch_splits) if value == split
        ]
        actual = {
            capability: sum(schedule[index][capability] for index in indices)
            for capability in MVP_CAPABILITY_ORDER
        }
        if actual != CORE_R2_CAPABILITY_COUNTS[split]:
            raise AssertionError(f"r2 deterministic schedule drifted: {split}")
    return batch_splits, tuple(schedule)


def _r2_capability_batch_bounds() -> tuple[dict[str, int], dict[str, int]]:
    """Return the frozen floor/ceil capability range for every tail batch."""

    totals = _r2_tail_capability_totals()
    lower = {
        capability: totals[capability] // CORE_R2_TAIL_BATCH_COUNT
        for capability in MVP_CAPABILITY_ORDER
    }
    upper = {
        capability: -(-totals[capability] // CORE_R2_TAIL_BATCH_COUNT)
        for capability in MVP_CAPABILITY_ORDER
    }
    if sum(lower.values()) != CORE_R2_BATCH_SIZE - 2:
        raise AssertionError("r2 tail floor capacities no longer leave two extras")
    if any(lower[capability] < 1 for capability in MVP_CAPABILITY_ORDER):
        raise AssertionError("r2 tail batch would lose a capability floor")
    return lower, upper


def _r2_joint_tie_weight(*, seed: int, label: tuple[object, ...]) -> float:
    """A bounded stable secondary objective coefficient for one MILP variable."""

    digest = hashlib.sha256(
        f"{seed}\x00{label!r}".encode("utf-8")
    ).digest()
    return float(1 + int.from_bytes(digest[:4], "big") % 997)


def _r2_solve_joint_batch_layout(*, seed: int) -> _R2JointBatchSolution:
    """Solve the tail's schedule and component reuse packing together.

    The old split-first scheduler could prove marginal capability quotas but
    could not prove that the locked 1/2/3-use component inventory filled the
    same 25-slot batches.  This model instead treats counterfactual vectors
    and regular reuse bundles as the only fill primitives.  Thus every output
    batch is already component-atomic before an asset ID is materialised.
    """

    batch_final_splits, frozen_schedule = _r2_deterministic_capability_schedule()
    lower, upper = _r2_capability_batch_bounds()
    categories = tuple(_R2_REGULAR_CATEGORY_CAPABILITY)
    abo_capabilities = frozenset(_CROSS_INTENT_TRIPLET)
    food_capabilities = frozenset(
        {
            "knowledge.visual_encyclopedia",
            "utility.recipe_guidance",
        }
    )

    variable_labels: list[tuple[object, ...]] = []
    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT):
        variable_labels.extend(
            (
                ("abo", batch_index),
                ("food", batch_index),
                ("abo_stack", batch_index),
                ("food_stack", batch_index),
            )
        )
        for category in categories:
            for usage_count in sorted(_R2_REGULAR_USAGE_COUNTS[category]):
                variable_labels.append(
                    ("regular", batch_index, category, usage_count)
                )
    variable_index = {label: index for index, label in enumerate(variable_labels)}
    variable_count = len(variable_labels)

    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int32)
    max_abo = min(upper[capability] for capability in abo_capabilities)
    max_food = min(upper[capability] for capability in food_capabilities)
    for index, label in enumerate(variable_labels):
        kind = label[0]
        if kind == "abo":
            upper_bounds[index] = max_abo
        elif kind == "food":
            upper_bounds[index] = max_food
        elif kind in {"abo_stack", "food_stack"}:
            upper_bounds[index] = 1.0
        else:
            _, _, category, usage_count = label
            capability = _R2_REGULAR_CATEGORY_CAPABILITY[cast(str, category)]
            upper_bounds[index] = upper[capability] // cast(int, usage_count)

    rows: list[np.ndarray] = []
    lower_constraints: list[float] = []
    upper_constraints: list[float] = []

    def add_constraint(
        coefficients: Mapping[tuple[object, ...], int],
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        row = np.zeros(variable_count, dtype=np.float64)
        for label, coefficient in coefficients.items():
            row[variable_index[label]] = coefficient
        rows.append(row)
        lower_constraints.append(lower_bound)
        upper_constraints.append(upper_bound)

    def capability_expression(
        batch_index: int,
        capability: str,
    ) -> dict[tuple[object, ...], int]:
        coefficients: dict[tuple[object, ...], int] = {}
        if capability in abo_capabilities:
            coefficients[("abo", batch_index)] = 1
        if capability in food_capabilities:
            coefficients[("food", batch_index)] = 1
        for category in categories:
            if _R2_REGULAR_CATEGORY_CAPABILITY[category] != capability:
                continue
            for usage_count in _R2_REGULAR_USAGE_COUNTS[category]:
                coefficients[("regular", batch_index, category, usage_count)] = (
                    usage_count
                )
        return coefficients

    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT):
        total_expression: Counter[tuple[object, ...]] = Counter()
        for capability in MVP_CAPABILITY_ORDER:
            expression = capability_expression(batch_index, capability)
            target = frozen_schedule[batch_index][capability]
            add_constraint(expression, target, target)
            total_expression.update(expression)
        add_constraint(total_expression, CORE_R2_BATCH_SIZE, CORE_R2_BATCH_SIZE)

        # A second counterfactual of either kind is a true same-batch stack.
        # The joint feasibility proof establishes that a maximum multiplicity
        # of two is sufficient, so the binary excess flags make the primary
        # objective an exact count of stacked batches rather than a proxy.
        add_constraint(
            {
                ("abo", batch_index): 1,
                ("abo_stack", batch_index): -1,
            },
            -np.inf,
            1,
        )
        add_constraint(
            {
                ("food", batch_index): 1,
                ("food_stack", batch_index): -1,
            },
            -np.inf,
            1,
        )

    for split in _R2_TAIL_SPLITS:
        batch_indices = tuple(
            index
            for index, batch_split in enumerate(batch_final_splits)
            if batch_split == split
        )
        for capability in MVP_CAPABILITY_ORDER:
            expression: Counter[tuple[object, ...]] = Counter()
            for batch_index in batch_indices:
                expression.update(capability_expression(batch_index, capability))
            target = CORE_R2_CAPABILITY_COUNTS[split][capability]
            add_constraint(expression, target, target)
        for category, label in (("abo", "abo"), ("food", "food")):
            expected = _R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS[split][category]
            add_constraint(
                {(label, batch_index): 1 for batch_index in batch_indices},
                expected,
                expected,
            )

    for category in categories:
        for usage_count, expected in _R2_REGULAR_USAGE_COUNTS[category].items():
            add_constraint(
                {
                    ("regular", batch_index, category, usage_count): 1
                    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT)
                },
                expected,
                expected,
            )

    constraint_matrix = np.vstack(rows)
    constraint_lower = np.asarray(lower_constraints, dtype=np.float64)
    constraint_upper = np.asarray(upper_constraints, dtype=np.float64)

    def solve(
        objective: np.ndarray,
        *,
        extra_rows: Sequence[np.ndarray] = (),
        extra_lower: Sequence[float] = (),
        extra_upper: Sequence[float] = (),
    ) -> np.ndarray:
        matrix = constraint_matrix
        lower_vector = constraint_lower
        upper_vector = constraint_upper
        if extra_rows:
            matrix = np.vstack((matrix, *extra_rows))
            lower_vector = np.concatenate((lower_vector, np.asarray(extra_lower)))
            upper_vector = np.concatenate((upper_vector, np.asarray(extra_upper)))
        outcome = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(matrix, lower_vector, upper_vector),
            options={"presolve": True, "time_limit": 20.0, "mip_rel_gap": 0.0},
        )
        if outcome.status != 0 or outcome.x is None:
            raise ValueError(
                "r2 joint batch MILP did not reach an exact optimum: "
                f"status={outcome.status}; message={outcome.message}"
            )
        rounded = np.rint(outcome.x)
        if np.max(np.abs(outcome.x - rounded), initial=0.0) > 1e-6:
            raise ValueError("r2 joint batch MILP returned a non-integral solution")
        return rounded.astype(np.int64)

    primary_objective = np.zeros(variable_count, dtype=np.float64)
    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT):
        primary_objective[variable_index[("abo_stack", batch_index)]] = 1.0
        primary_objective[variable_index[("food_stack", batch_index)]] = 1.0
    primary = solve(primary_objective)
    minimum_stack = int(
        sum(
            primary[variable_index[("abo_stack", batch_index)]]
            + primary[variable_index[("food_stack", batch_index)]]
            for batch_index in range(CORE_R2_TAIL_BATCH_COUNT)
        )
    )
    stack_row = np.zeros(variable_count, dtype=np.float64)
    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT):
        stack_row[variable_index[("abo_stack", batch_index)]] = 1.0
        stack_row[variable_index[("food_stack", batch_index)]] = 1.0
    # Only A/F placement is extracted from the MILP.  The y bundle incumbent
    # is intentionally discarded below in favour of the deterministic DP, so
    # it cannot leak a solver-dependent ordering into a canonical plan.
    tiebreak_objective = np.zeros(variable_count, dtype=np.float64)
    for label in variable_labels:
        if label[0] in {"abo", "food", "abo_stack", "food_stack"}:
            tiebreak_objective[variable_index[label]] = _r2_joint_tie_weight(
                seed=seed,
                label=label,
            )
    solution_vector = solve(
        tiebreak_objective,
        extra_rows=(stack_row,),
        extra_lower=(minimum_stack,),
        extra_upper=(minimum_stack,),
    )

    abo_by_batch: list[int] = []
    food_by_batch: list[int] = []
    for batch_index in range(CORE_R2_TAIL_BATCH_COUNT):
        abo_by_batch.append(int(solution_vector[variable_index[("abo", batch_index)]]))
        food_by_batch.append(int(solution_vector[variable_index[("food", batch_index)]]))

    regular_bundle_counts_by_batch: list[dict[str, dict[int, int]]] = [
        {} for _ in range(CORE_R2_TAIL_BATCH_COUNT)
    ]
    for category, capability in _R2_REGULAR_CATEGORY_CAPABILITY.items():
        residual_capacities: list[int] = []
        for batch_index, counts in enumerate(frozen_schedule):
            special_count = 0
            if capability in abo_capabilities:
                special_count += abo_by_batch[batch_index]
            if capability in food_capabilities:
                special_count += food_by_batch[batch_index]
            residual = counts[capability] - special_count
            if residual < 0:
                raise ValueError(
                    f"r2 joint A/F placement overfills {capability} batch "
                    f"{batch_index + 1}"
                )
            residual_capacities.append(residual)
        patterns = _r2_pack_bundle_usage_counts(
            residual_capacities,
            _R2_REGULAR_USAGE_COUNTS[category],
        )
        for batch_index, pattern in enumerate(patterns):
            regular_bundle_counts_by_batch[batch_index][category] = dict(pattern)

    solution = _R2JointBatchSolution(
        batch_final_splits=batch_final_splits,
        abo_by_batch=tuple(abo_by_batch),
        food_by_batch=tuple(food_by_batch),
        regular_bundle_counts_by_batch=tuple(regular_bundle_counts_by_batch),
        capability_counts_by_batch=tuple(
            dict(counts) for counts in frozen_schedule
        ),
        stacked_counterfactual_batch_count=minimum_stack,
    )
    _verify_r2_joint_batch_solution(solution)
    return solution


def _verify_r2_joint_batch_solution(solution: _R2JointBatchSolution) -> None:
    """Re-evaluate the MILP equations in ordinary Python before materialising."""

    if len(solution.batch_final_splits) != CORE_R2_TAIL_BATCH_COUNT:
        raise ValueError("r2 joint solution has the wrong tail batch count")
    if not (
        len(solution.abo_by_batch)
        == len(solution.food_by_batch)
        == len(solution.regular_bundle_counts_by_batch)
        == len(solution.capability_counts_by_batch)
        == CORE_R2_TAIL_BATCH_COUNT
    ):
        raise ValueError("r2 joint solution sidecars have inconsistent batch lengths")

    lower, upper = _r2_capability_batch_bounds()
    split_capability_counts: dict[Split, Counter[str]] = {
        split: Counter() for split in _R2_TAIL_SPLITS
    }
    split_special_counts: dict[Split, Counter[str]] = {
        split: Counter() for split in _R2_TAIL_SPLITS
    }
    global_regular_counts: dict[str, Counter[int]] = {
        category: Counter() for category in _R2_REGULAR_CATEGORY_CAPABILITY
    }
    actual_stack_count = 0
    for batch_index, split in enumerate(solution.batch_final_splits):
        abo_count = solution.abo_by_batch[batch_index]
        food_count = solution.food_by_batch[batch_index]
        if abo_count < 0 or food_count < 0:
            raise ValueError("r2 joint solution has negative counterfactual counts")
        actual_stack_count += max(0, abo_count - 1) + max(0, food_count - 1)
        split_special_counts[split]["abo"] += abo_count
        split_special_counts[split]["food"] += food_count

        derived_capability_counts: Counter[str] = Counter()
        for capability in _CROSS_INTENT_TRIPLET:
            derived_capability_counts[capability] += abo_count
        derived_capability_counts["knowledge.visual_encyclopedia"] += food_count
        derived_capability_counts["utility.recipe_guidance"] += food_count
        for category, usage_counts in solution.regular_bundle_counts_by_batch[
            batch_index
        ].items():
            if category not in _R2_REGULAR_CATEGORY_CAPABILITY:
                raise ValueError(f"r2 joint solution has unknown category: {category}")
            capability = _R2_REGULAR_CATEGORY_CAPABILITY[category]
            for usage_count, component_count in usage_counts.items():
                if usage_count not in _R2_REGULAR_USAGE_COUNTS[category]:
                    raise ValueError("r2 joint solution has unknown reuse cardinality")
                if component_count < 0:
                    raise ValueError("r2 joint solution has negative bundle count")
                global_regular_counts[category][usage_count] += component_count
                derived_capability_counts[capability] += usage_count * component_count
        expected_capability_counts = solution.capability_counts_by_batch[batch_index]
        if dict(derived_capability_counts) != dict(expected_capability_counts):
            raise ValueError(
                f"r2 joint capability equation drifted in batch {batch_index + 1}"
            )
        if sum(derived_capability_counts.values()) != CORE_R2_BATCH_SIZE:
            raise ValueError(f"r2 joint batch {batch_index + 1} is not 25 slots")
        for capability in MVP_CAPABILITY_ORDER:
            value = derived_capability_counts[capability]
            if not lower[capability] <= value <= upper[capability]:
                raise ValueError(
                    f"r2 joint batch {batch_index + 1} violates {capability} floor/ceil"
                )
        split_capability_counts[split].update(derived_capability_counts)

    if actual_stack_count != solution.stacked_counterfactual_batch_count:
        raise ValueError("r2 joint counterfactual stack indicator drifted")
    for split in _R2_TAIL_SPLITS:
        if dict(split_capability_counts[split]) != CORE_R2_CAPABILITY_COUNTS[split]:
            raise ValueError(f"r2 joint split capability matrix drifted: {split}")
        if dict(split_special_counts[split]) != _R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS[
            split
        ]:
            raise ValueError(f"r2 joint counterfactual split count drifted: {split}")
    for category, expected in _R2_REGULAR_USAGE_COUNTS.items():
        if dict(global_regular_counts[category]) != expected:
            raise ValueError(f"r2 joint regular reuse inventory drifted: {category}")


def build_r2_split_atomic_batch_schedule(
    *,
    seed: int = 20260804,
) -> tuple[tuple[Split, ...], tuple[dict[str, int], ...]]:
    """Return the jointly proven split-atomic 52 x 25 capability schedule."""

    solution = _r2_solve_joint_batch_layout(seed=seed)
    return (
        solution.batch_final_splits,
        tuple(dict(counts) for counts in solution.capability_counts_by_batch),
    )


def _r2_component_order(
    components: Sequence[_R2Component],
    *,
    seed: int,
    namespace: str,
) -> list[_R2Component]:
    """Use a stable seeded ordering without depending on input file ordering."""

    def sort_key(component: _R2Component) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{seed}\x00{namespace}\x00{component.component_id}".encode("utf-8")
        ).hexdigest()
        return digest, component.component_id

    return sorted(components, key=sort_key)


def _r2_component_pools(
    *,
    target_catalog: AssetCatalog,
    target_capability_assignments: Sequence[CapabilityAssignment],
    dev_component_ids: set[str],
) -> dict[str, list[_R2Component]]:
    """Classify every non-dev v9 component by its explicit capability bindings."""

    target_catalog.require_verified_files()
    raw: dict[str, dict[str, list[CapabilityAssignment]]] = {}
    for assignment in target_capability_assignments:
        if MVP_CAPABILITY_INTENTS.get(assignment.canonical_capability) != (
            assignment.canonical_intent
        ):
            raise ValueError(
                "r2 assignment canonical capability does not belong to its intent: "
                f"{assignment.assignment_id}"
            )
        resolution = target_catalog.resolve_asset_id(assignment.asset_id)
        target_catalog.verify_reference(
            asset_id=assignment.asset_id,
            image_path=assignment.image_path,
            leakage_group_id=resolution.leakage_group_id,
        )
        if resolution.asset.local_path != assignment.image_path:
            raise ValueError(
                "r2 assignment image_path drifted from target catalog: "
                f"{assignment.assignment_id}"
            )
        component = raw.setdefault(resolution.leakage_group_id, {})
        component.setdefault(assignment.canonical_capability, []).append(assignment)

    profile_to_category = {
        profile: category for category, profile in _R2_COMPONENT_PROFILES.items()
    }
    pools: dict[str, list[_R2Component]] = {
        category: [] for category in _R2_COMPONENT_PROFILES
    }
    unexpected: dict[str, tuple[str, ...]] = {}
    for component_id, capability_assignments in sorted(raw.items()):
        if component_id in dev_component_ids:
            continue
        profile = frozenset(capability_assignments)
        category = profile_to_category.get(profile)
        if category is None:
            unexpected[component_id] = tuple(sorted(profile))
            continue
        if category in {"abo", "food"}:
            common_asset_paths = set.intersection(
                *(
                    {
                        (assignment.asset_id, assignment.image_path)
                        for assignment in assignments
                    }
                    for assignments in capability_assignments.values()
                )
            )
            if not common_asset_paths:
                raise ValueError(
                    "r2 counterfactual component lacks one shared asset/path across "
                    f"all capability bindings: {component_id}"
                )
            asset_id, image_path = min(common_asset_paths)
            bindings = {
                capability: min(
                    (
                        assignment
                        for assignment in assignments
                        if (assignment.asset_id, assignment.image_path)
                        == (asset_id, image_path)
                    ),
                    key=lambda item: item.assignment_id,
                )
                for capability, assignments in capability_assignments.items()
            }
        else:
            bindings = {
                capability: min(
                    assignments,
                    key=lambda item: (
                        item.asset_id,
                        item.image_path,
                        item.assignment_id,
                    ),
                )
                for capability, assignments in capability_assignments.items()
            }
        pools[category].append(
            _R2Component(component_id=component_id, bindings=bindings)
        )
    if unexpected:
        sample = dict(list(unexpected.items())[:5])
        raise ValueError(f"unexpected r2 non-dev capability profiles: {sample}")
    actual = {category: len(pool) for category, pool in pools.items()}
    if actual != _R2_REQUIRED_NONDEV_COMPONENT_COUNTS:
        raise ValueError(
            "r2 non-dev capability component profile is not the locked v9 pool: "
            f"{actual}"
        )
    return pools


def _r2_regular_bundles(
    *,
    category: str,
    components: Sequence[_R2Component],
    usage_counts: Mapping[int, int],
    seed: int,
) -> list[_R2Bundle]:
    if sum(usage_counts.values()) != len(components):
        raise ValueError(
            f"r2 {category} component count does not match the locked reuse plan"
        )
    usage_sequence = [
        usage_count
        for usage_count in sorted(usage_counts, reverse=True)
        for _ in range(usage_counts[usage_count])
    ]
    ordered_components = _r2_component_order(
        components,
        seed=seed,
        namespace=f"regular/{category}",
    )
    if len(usage_sequence) != len(ordered_components):
        raise AssertionError("r2 regular bundle assignment count drifted")
    capability = _R2_REGULAR_CATEGORY_CAPABILITY[category]
    return [
        _R2Bundle(
            component_id=component.component_id,
            capability=capability,
            usage_count=usage_count,
            category=category,
        )
        for component, usage_count in zip(
            ordered_components,
            usage_sequence,
            strict=True,
        )
    ]


def _r2_special_component_splits(
    *,
    category: Literal["abo", "food"],
    components: Sequence[_R2Component],
    seed: int,
) -> list[tuple[_R2Component, Split]]:
    expected_count = _R2_REQUIRED_NONDEV_COMPONENT_COUNTS[category]
    if len(components) != expected_count:
        raise ValueError(f"r2 {category} component count drifted from the v9 lock")
    splits: list[Split] = []
    for split in _R2_TAIL_SPLITS:
        splits.extend(
            [split] * _R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS[split][category]
        )
    if len(splits) != expected_count:
        raise AssertionError("r2 counterfactual split allocation count drifted")
    ordered_components = _r2_component_order(
        components,
        seed=seed,
        namespace=f"counterfactual/{category}",
    )
    return list(zip(ordered_components, splits, strict=True))


def _r2_patterns_for_capacity(
    capacity: int,
    sizes: tuple[int, ...],
    limits: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = []

    def visit(index: int, remaining: int, selected: list[int]) -> None:
        if index == len(sizes):
            if remaining == 0:
                patterns.append(tuple(selected))
            return
        size = sizes[index]
        maximum = min(limits[index], remaining // size)
        for count in range(maximum, -1, -1):
            selected.append(count)
            visit(index + 1, remaining - size * count, selected)
            selected.pop()

    visit(0, capacity, [])
    return tuple(patterns)


def _r2_pack_bundle_usage_counts(
    capacities: Sequence[int],
    usage_counts: Mapping[int, int],
) -> list[dict[int, int]]:
    """Small deterministic integer packing for one capability's 52 bins."""

    sizes = tuple(sorted(usage_counts, reverse=True))
    limits = tuple(usage_counts[size] for size in sizes)
    if sum(capacities) != sum(
        size * usage_counts[size] for size in sizes
    ):
        raise ValueError("r2 component bundle capacity does not match target slots")
    pattern_cache = {
        capacity: _r2_patterns_for_capacity(capacity, sizes, limits)
        for capacity in set(capacities)
    }
    if any(not pattern_cache[capacity] for capacity in capacities):
        raise ValueError("r2 batch capacity cannot be expressed by component bundles")

    states: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {
        (0,) * len(sizes): ()
    }
    for capacity in capacities:
        next_states: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
        for used, path in states.items():
            for pattern in pattern_cache[capacity]:
                updated = tuple(
                    before + added
                    for before, added in zip(used, pattern, strict=True)
                )
                if any(
                    value > limit
                    for value, limit in zip(updated, limits, strict=True)
                ):
                    continue
                next_states.setdefault(updated, (*path, pattern))
        states = next_states
        if not states:
            raise ValueError("r2 batch packing became infeasible")
    target = limits
    try:
        patterns = states[target]
    except KeyError:
        raise ValueError("r2 batch packing cannot use every selected component") from None
    return [
        {
            size: count
            for size, count in zip(sizes, pattern, strict=True)
            if count
        }
        for pattern in patterns
    ]


def _r2_distributed_batch_indices(
    *,
    count: int,
    candidate_indices: Sequence[int],
    seed: int,
    namespace: str,
) -> tuple[int, ...]:
    """Spread a vector group across its own final-split batches reproducibly."""

    if count <= 0 or count > len(candidate_indices):
        raise ValueError("r2 counterfactual batch count is out of range")
    phase = int.from_bytes(
        hashlib.sha256(f"{seed}\x00{namespace}".encode("utf-8")).digest()[:4],
        "big",
    ) % len(candidate_indices)
    base = tuple(
        ((2 * offset + 1) * len(candidate_indices)) // (2 * count)
        for offset in range(count)
    )
    result = tuple(candidate_indices[(value + phase) % len(candidate_indices)] for value in base)
    if len(set(result)) != count:
        raise AssertionError("r2 counterfactual batch spacing unexpectedly collided")
    return result


def _r2_mark_tail_boundaries(slots_by_batch: Sequence[list[_R2TailSlot]]) -> None:
    """Add natural boundaries after fixed semantic counterfactual vectors."""

    capability_rank = {
        capability: index for index, capability in enumerate(MVP_CAPABILITY_ORDER)
    }
    for split in _R2_TAIL_SPLITS:
        for capability in MVP_CAPABILITY_ORDER:
            selected: list[tuple[int, _R2TailSlot]] = [
                (batch_index, slot)
                for batch_index, batch in enumerate(slots_by_batch)
                for slot in batch
                if slot.final_split == split and slot.capability == capability
            ]
            expected = CORE_R2_CAPABILITY_BOUNDARY_COUNTS[split][capability]
            existing = [slot for _, slot in selected if slot.is_boundary]
            remaining = expected - len(existing)
            if remaining < 0:
                raise ValueError(
                    "r2 counterfactual vectors exceed the frozen boundary target: "
                    f"{split}/{capability}"
                )
            candidates = sorted(
                (
                    (batch_index, slot)
                    for batch_index, slot in selected
                    if not slot.is_boundary
                ),
                key=lambda item: (
                    item[0],
                    capability_rank[item[1].capability],
                    item[1].component_id,
                    item[1].reuse_variant,
                ),
            )
            if len(candidates) < remaining:
                raise ValueError(
                    "r2 cannot fill the frozen natural-boundary target: "
                    f"{split}/{capability}"
                )
            for _, slot in candidates[:remaining]:
                slot.is_boundary = True
                slot.boundary_strategy = "natural_ambiguity"


def build_r2_core_in_memory_plan(
    dev_prefix_rebind: R2DevPrefixRebind,
    *,
    target_catalog: AssetCatalog,
    target_capability_assignments: Sequence[CapabilityAssignment],
    expected_target_catalog_sha256: str,
    expected_target_capability_assignments_sha256: str,
    seed: int = 20260804,
) -> R2CoreInMemoryPlan:
    """Build and audit the r2 60 x 25 layout without writing any artifact.

    The historic ``CorpusPlan`` stays intentionally serialization-compatible:
    all tail records are still ``opt_pool`` provisionally.  Final split and
    reuse facts are returned only in the typed in-memory sidecars below.
    """

    _require_sha256(expected_target_catalog_sha256, "target catalog SHA-256")
    _require_sha256(
        expected_target_capability_assignments_sha256,
        "target capability assignments SHA-256",
    )
    if target_catalog.manifest.catalog_sha256 != expected_target_catalog_sha256:
        raise ValueError("loaded target catalog does not match the expected SHA-256")
    target_assignment_sha256 = _capability_assignments_digest(
        target_capability_assignments,
        expected_sha256=expected_target_capability_assignments_sha256,
        formal=True,
    )
    assert target_assignment_sha256 is not None
    dev_prefix = dev_prefix_rebind.dev_prefix
    if (
        dev_prefix.asset_catalog_sha256 != expected_target_catalog_sha256
        or dev_prefix.capability_assignments_sha256 != target_assignment_sha256
        or dev_prefix.leakage_policy_version
        != target_catalog.manifest.leakage_policy_version
    ):
        raise ValueError("r2 dev prefix is not bound to the locked target inputs")
    audit_plan_catalog_binding(dev_prefix, target_catalog)
    assignment_index = _index_capability_assignments(target_capability_assignments)
    _audit_planned_assignment_binding(
        dev_prefix.queries,
        assignment_index,
        formal=True,
    )

    dev_component_ids = {item.leakage_group_id for item in dev_prefix.queries}
    pools = _r2_component_pools(
        target_catalog=target_catalog,
        target_capability_assignments=target_capability_assignments,
        dev_component_ids=dev_component_ids,
    )
    component_by_id = {
        component.component_id: component
        for pool in pools.values()
        for component in pool
    }
    if len(component_by_id) != sum(len(pool) for pool in pools.values()):
        raise ValueError("r2 non-dev component is present in multiple capability pools")

    joint_solution = _r2_solve_joint_batch_layout(seed=seed)
    batch_final_splits = joint_solution.batch_final_splits
    schedule = joint_solution.capability_counts_by_batch
    slots_by_batch: list[list[_R2TailSlot]] = [
        [] for _ in range(CORE_R2_TAIL_BATCH_COUNT)
    ]
    special_capabilities: dict[str, tuple[str, ...]] = {
        "abo": _CROSS_INTENT_TRIPLET,
        "food": (
            "knowledge.visual_encyclopedia",
            "utility.recipe_guidance",
        ),
    }
    for category in ("abo", "food"):
        strategy = (
            "cross_intent_triplet" if category == "abo" else "cross_intent_pair"
        )
        counts_by_batch = (
            joint_solution.abo_by_batch
            if category == "abo"
            else joint_solution.food_by_batch
        )
        components = deque(
            _r2_component_order(
                pools[category],
                seed=seed,
                namespace=f"counterfactual/{category}",
            )
        )
        vector_index = 0
        for batch_index, count in enumerate(counts_by_batch):
            split = batch_final_splits[batch_index]
            for _ in range(count):
                vector_index += 1
                try:
                    component = components.popleft()
                except IndexError:
                    raise AssertionError(
                        f"r2 {category} joint placement exhausted its component pool"
                    ) from None
                group_id = f"r2-{category}-{vector_index:03d}"
                for capability in special_capabilities[category]:
                    assignment = component.bindings.get(capability)
                    if assignment is None:
                        raise ValueError(
                            f"r2 {category} component is missing {capability}: "
                            f"{component.component_id}"
                        )
                    slots_by_batch[batch_index].append(
                        _R2TailSlot(
                            component_id=component.component_id,
                            capability=capability,
                            assignment=assignment,
                            final_split=split,
                            reuse_variant=(
                                f"counterfactual/{category}/"
                                f"{capability.rsplit('.', 1)[1]}"
                            ),
                            reuse_reason=f"counterfactual_{category}",
                            is_boundary=True,
                            boundary_strategy=strategy,
                            boundary_group_id=group_id,
                        )
                    )
        if components:
            raise AssertionError(
                f"r2 {category} joint placement left unassigned components"
            )

    for category, capability in _R2_REGULAR_CATEGORY_CAPABILITY.items():
        selected_count = sum(_R2_REGULAR_USAGE_COUNTS[category].values())
        selected = _r2_component_order(
            pools[category],
            seed=seed,
            namespace=f"selected/{category}",
        )[:selected_count]
        bundles = _r2_regular_bundles(
            category=category,
            components=selected,
            usage_counts=_R2_REGULAR_USAGE_COUNTS[category],
            seed=seed,
        )
        usage_counts = Counter(bundle.usage_count for bundle in bundles)
        queues = {
            usage_count: deque(
                sorted(
                    (
                        bundle
                        for bundle in bundles
                        if bundle.usage_count == usage_count
                    ),
                    key=lambda bundle: (bundle.component_id, bundle.category),
                )
            )
            for usage_count in usage_counts
        }
        for batch_index, regular_counts in enumerate(
            joint_solution.regular_bundle_counts_by_batch
        ):
            pattern = regular_counts[category]
            for usage_count in sorted(pattern, reverse=True):
                for _ in range(pattern[usage_count]):
                    try:
                        bundle = queues[usage_count].popleft()
                    except IndexError:
                        raise AssertionError(
                            "r2 bundle packing exhausted a queue"
                        ) from None
                    component = component_by_id[bundle.component_id]
                    assignment = component.bindings.get(capability)
                    if assignment is None:
                        raise AssertionError(
                            "r2 regular bundle lost its capability binding"
                        )
                    for occurrence in range(1, usage_count + 1):
                        slots_by_batch[batch_index].append(
                            _R2TailSlot(
                                component_id=bundle.component_id,
                                capability=capability,
                                assignment=assignment,
                                final_split=batch_final_splits[batch_index],
                                reuse_variant=(
                                    f"{bundle.category}/{bundle.reuse_reason}/"
                                    f"{occurrence:02d}"
                                ),
                                reuse_reason=bundle.reuse_reason,
                            )
                        )
        if any(queues[usage_count] for usage_count in queues):
            raise AssertionError("r2 bundle packing left an unassigned component")

    for batch_index, (batch, target) in enumerate(
        zip(slots_by_batch, schedule, strict=True),
        start=1,
    ):
        counts = Counter(slot.capability for slot in batch)
        if len(batch) != CORE_R2_BATCH_SIZE or dict(counts) != target:
            raise AssertionError(
                f"r2 tail batch {batch_index:03d} does not match its capacity schedule"
            )
    _r2_mark_tail_boundaries(slots_by_batch)

    capability_rank = {
        capability: index for index, capability in enumerate(MVP_CAPABILITY_ORDER)
    }
    tail_queries: list[PlannedQuery] = []
    final_split_by_plan_id: dict[str, Split] = {
        item.plan_id: "dev_mini" for item in dev_prefix.queries
    }
    reuse_variant_by_plan_id: dict[str, str] = {}
    reuse_reason_by_plan_id: dict[str, str] = {}
    dev_variant_counts: Counter[str] = Counter()
    for item in dev_prefix.queries:
        dev_variant_counts[item.leakage_group_id] += 1
        reuse_variant_by_plan_id[item.plan_id] = (
            f"legacy-prefix/{dev_variant_counts[item.leakage_group_id]:02d}"
        )
        reuse_reason_by_plan_id[item.plan_id] = "legacy_prefix"

    global_index = 200
    for batch_offset, slots in enumerate(slots_by_batch, start=9):
        batch_id = f"core-{batch_offset:03d}"
        template_family = f"generation-prompt-family-v2/r2-batch-{batch_offset:03d}"
        ordered_slots = sorted(
            slots,
            key=lambda slot: (
                capability_rank[slot.capability],
                slot.component_id,
                slot.reuse_variant,
            ),
        )
        for position, slot in enumerate(ordered_slots, start=1):
            global_index += 1
            target_catalog.verify_reference(
                asset_id=slot.assignment.asset_id,
                image_path=slot.assignment.image_path,
                leakage_group_id=slot.component_id,
            )
            plan_id = f"r2-core-{global_index:04d}"
            tail_queries.append(
                PlannedQuery(
                    plan_id=plan_id,
                    batch_id=batch_id,
                    position=position,
                    taxonomy_version=slot.assignment.taxonomy_version,
                    task_spec_version=slot.assignment.task_spec_version,
                    asset_id=slot.assignment.asset_id,
                    image_path=slot.assignment.image_path,
                    leakage_group_id=slot.component_id,
                    template_family=template_family,
                    generator_batch_id=batch_id,
                    canonical_intent=slot.assignment.canonical_intent,
                    canonical_capability=slot.assignment.canonical_capability,
                    acceptable_capabilities=list(
                        slot.assignment.acceptable_capabilities
                    ),
                    requires_card=slot.assignment.requires_card,
                    capability_assignment_id=slot.assignment.assignment_id,
                    capability_assignment_source_sha256=(
                        slot.assignment.source_artifact_sha256
                    ),
                    is_boundary=slot.is_boundary,
                    boundary_strategy=slot.boundary_strategy,
                    boundary_group_id=slot.boundary_group_id,
                    provisional_split="opt_pool",
                )
            )
            final_split_by_plan_id[plan_id] = slot.final_split
            reuse_variant_by_plan_id[plan_id] = slot.reuse_variant
            reuse_reason_by_plan_id[plan_id] = slot.reuse_reason

    plan = CorpusPlan(
        scope="core",
        seed=seed,
        asset_catalog_sha256=expected_target_catalog_sha256,
        capability_assignments_sha256=target_assignment_sha256,
        leakage_policy_version=target_catalog.manifest.leakage_policy_version,
        queries=[*dev_prefix.queries, *tail_queries],
    )
    audit_plan_catalog_binding(plan, target_catalog)
    _audit_planned_assignment_binding(
        plan.queries,
        assignment_index,
        formal=True,
    )
    provisional = R2CoreInMemoryPlan(
        plan=plan,
        dev_prefix_rebind=dev_prefix_rebind,
        final_split_by_plan_id=final_split_by_plan_id,
        reuse_variant_by_plan_id=reuse_variant_by_plan_id,
        reuse_reason_by_plan_id=reuse_reason_by_plan_id,
        audit=cast(R2CorePlanAudit, None),
    )
    audit = audit_r2_core_in_memory_plan(provisional)
    return R2CoreInMemoryPlan(
        plan=plan,
        dev_prefix_rebind=dev_prefix_rebind,
        final_split_by_plan_id=final_split_by_plan_id,
        reuse_variant_by_plan_id=reuse_variant_by_plan_id,
        reuse_reason_by_plan_id=reuse_reason_by_plan_id,
        audit=audit,
    )


def audit_r2_core_in_memory_plan(result: R2CoreInMemoryPlan) -> R2CorePlanAudit:
    """Fail closed on every frozen r2 layout, split, and reuse invariant."""

    plan = result.plan
    if plan.scope != "core" or len(plan.queries) != 1500:
        raise ValueError("r2 in-memory result must contain one 1,500-query core plan")
    if plan.queries[:200] != result.dev_prefix_rebind.dev_prefix.queries:
        raise ValueError("r2 plan changed the locked re-bound dev prefix")
    if (
        plan.asset_catalog_sha256
        != result.dev_prefix_rebind.dev_prefix.asset_catalog_sha256
        or plan.capability_assignments_sha256
        != result.dev_prefix_rebind.dev_prefix.capability_assignments_sha256
        or plan.leakage_policy_version
        != result.dev_prefix_rebind.dev_prefix.leakage_policy_version
    ):
        raise ValueError("r2 plan top-level bindings drifted from its re-bound prefix")

    plan_ids = {item.plan_id for item in plan.queries}
    if (
        set(result.final_split_by_plan_id) != plan_ids
        or set(result.reuse_variant_by_plan_id) != plan_ids
        or set(result.reuse_reason_by_plan_id) != plan_ids
    ):
        raise ValueError("r2 final-split/reuse sidecars must cover every plan_id once")
    if any(
        split not in CORE_R2_FINAL_SPLITS
        for split in result.final_split_by_plan_id.values()
    ):
        raise ValueError("r2 final-split sidecar has an unknown split")

    capability_counts: Counter[tuple[Split, str]] = Counter()
    boundary_counts: Counter[tuple[Split, str]] = Counter()
    for index, item in enumerate(plan.queries):
        split = result.final_split_by_plan_id[item.plan_id]
        if index < 200:
            if split != "dev_mini" or item.provisional_split != "dev_mini":
                raise ValueError("r2 dev prefix split drifted")
        elif split not in _R2_TAIL_SPLITS or item.provisional_split != "opt_pool":
            raise ValueError("r2 tail split/provisional split drifted")
        capability_counts[(split, item.canonical_capability)] += 1
        if item.is_boundary:
            boundary_counts[(split, item.canonical_capability)] += 1

    actual_capability_counts = {
        split: {
            capability: capability_counts[(split, capability)]
            for capability in MVP_CAPABILITY_ORDER
        }
        for split in CORE_R2_FINAL_SPLITS
    }
    actual_boundary_counts = {
        split: {
            capability: boundary_counts[(split, capability)]
            for capability in MVP_CAPABILITY_ORDER
        }
        for split in CORE_R2_FINAL_SPLITS
    }
    if actual_capability_counts != CORE_R2_CAPABILITY_COUNTS:
        raise ValueError(
            "r2 capability x split matrix drifted: "
            f"{actual_capability_counts}"
        )
    if actual_boundary_counts != CORE_R2_CAPABILITY_BOUNDARY_COUNTS:
        raise ValueError(
            "r2 boundary capability x split matrix drifted: "
            f"{actual_boundary_counts}"
        )

    tail_batches: dict[str, list[PlannedQuery]] = {}
    for item in plan.queries[200:]:
        tail_batches.setdefault(item.generator_batch_id, []).append(item)
    if len(tail_batches) != CORE_R2_TAIL_BATCH_COUNT:
        raise ValueError("r2 tail must contain exactly 52 generator batches")
    tail_batch_capability_counts: dict[str, dict[str, int]] = {}
    tail_totals = _r2_tail_capability_totals()
    for batch_id, items in sorted(tail_batches.items()):
        batch_splits = {
            result.final_split_by_plan_id[item.plan_id] for item in items
        }
        if len(batch_splits) != 1:
            raise ValueError(
                f"r2 generator batch crosses final splits: {batch_id}/{batch_splits}"
            )
        if len({item.template_family for item in items}) != 1:
            raise ValueError(
                f"r2 generator batch has multiple template-family instances: {batch_id}"
            )
        counts = Counter(item.canonical_capability for item in items)
        if len(items) != CORE_R2_BATCH_SIZE or set(counts) != set(
            MVP_CAPABILITY_ORDER
        ):
            raise ValueError(f"r2 tail batch lacks 25 slots or a capability: {batch_id}")
        for capability in MVP_CAPABILITY_ORDER:
            lower = tail_totals[capability] // CORE_R2_TAIL_BATCH_COUNT
            upper = -(-tail_totals[capability] // CORE_R2_TAIL_BATCH_COUNT)
            if not lower <= counts[capability] <= upper:
                raise ValueError(
                    f"r2 tail batch capability exceeds floor/ceil: "
                    f"{batch_id}/{capability}={counts[capability]}"
                )
        tail_batch_capability_counts[batch_id] = {
            capability: counts[capability] for capability in MVP_CAPABILITY_ORDER
        }
    if len({item.template_family for item in plan.queries[200:]}) != (
        CORE_R2_TAIL_BATCH_COUNT
    ):
        raise ValueError("r2 tail batches must have unique template-family instances")

    items_by_component: dict[str, list[PlannedQuery]] = {}
    for item in plan.queries:
        items_by_component.setdefault(item.leakage_group_id, []).append(item)
    max_component_reuse = max(len(items) for items in items_by_component.values())
    if len(items_by_component) < CORE_R2_MIN_UNIQUE_COMPONENTS:
        raise ValueError("r2 plan does not meet the minimum unique component floor")
    if max_component_reuse > 3:
        raise ValueError("r2 plan exceeds the three-query component reuse ceiling")

    capacity_triples = Counter({capability: 0 for capability in MVP_CAPABILITY_ORDER})
    abo_by_split: Counter[Split] = Counter()
    food_by_split: Counter[Split] = Counter()
    tail_multi_components: set[str] = set()
    for component_id, items in items_by_component.items():
        batches = {item.generator_batch_id for item in items}
        splits = {result.final_split_by_plan_id[item.plan_id] for item in items}
        variants = [result.reuse_variant_by_plan_id[item.plan_id] for item in items]
        reasons = {result.reuse_reason_by_plan_id[item.plan_id] for item in items}
        if len(batches) != 1:
            raise ValueError(
                f"r2 leakage component crosses generator batches: {component_id}"
            )
        if len(splits) != 1:
            raise ValueError(f"r2 leakage component crosses final splits: {component_id}")
        if any(not variant or variant != variant.strip() for variant in variants):
            raise ValueError(f"r2 component has a blank reuse_variant: {component_id}")
        if len(items) > 1 and len(set(variants)) != len(variants):
            raise ValueError(
                f"r2 repeated component has a duplicate reuse_variant: {component_id}"
            )

        is_tail_component = next(iter(splits)) in _R2_TAIL_SPLITS
        if is_tail_component:
            capabilities = {item.canonical_capability for item in items}
            if "product.multi_search" in capabilities:
                tail_multi_components.add(component_id)
            if reasons == {"capacity_required"}:
                if len(items) != 3 or len(capabilities) != 1:
                    raise ValueError(
                        "r2 capacity-required triple must be one capability in one batch"
                    )
                capacity_triples[next(iter(capabilities))] += 1
            elif reasons == {"counterfactual_abo"}:
                if (
                    len(items) != 3
                    or capabilities != set(_CROSS_INTENT_TRIPLET)
                    or not all(item.is_boundary for item in items)
                    or {item.boundary_strategy for item in items}
                    != {"cross_intent_triplet"}
                    or len({item.boundary_group_id for item in items}) != 1
                    or len({(item.asset_id, item.image_path) for item in items}) != 1
                ):
                    raise ValueError("r2 ABO counterfactual component drifted")
                abo_by_split[next(iter(splits))] += 1
            elif reasons == {"counterfactual_food"}:
                if (
                    len(items) != 2
                    or capabilities
                    != {
                        "knowledge.visual_encyclopedia",
                        "utility.recipe_guidance",
                    }
                    or not all(item.is_boundary for item in items)
                    or {item.boundary_strategy for item in items}
                    != {"cross_intent_pair"}
                    or len({item.boundary_group_id for item in items}) != 1
                    or len({(item.asset_id, item.image_path) for item in items}) != 1
                ):
                    raise ValueError("r2 Food counterfactual component drifted")
                food_by_split[next(iter(splits))] += 1
            elif len(items) == 3:
                raise ValueError(
                    "r2 normal triple must carry capacity_required reuse reason"
                )
            elif "capacity_required" in reasons:
                raise ValueError("r2 capacity_required reason must describe a triple")

    expected_capacity_triples = {
        "product.exact_match": 48,
        "product.multi_search": 14,
        "product.style_recommendation": 41,
        "knowledge.visual_encyclopedia": 45,
        "utility.document_reading": 0,
        "utility.recipe_guidance": 0,
    }
    if dict(capacity_triples) != expected_capacity_triples:
        raise ValueError(
            "r2 capacity-required triple counts drifted: "
            f"{dict(capacity_triples)}"
        )
    expected_abo = {
        split: _R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS[split]["abo"]
        for split in _R2_TAIL_SPLITS
    }
    expected_food = {
        split: _R2_COUNTERFACTUAL_SPLIT_COMPONENT_COUNTS[split]["food"]
        for split in _R2_TAIL_SPLITS
    }
    if dict(abo_by_split) != expected_abo or dict(food_by_split) != expected_food:
        raise ValueError(
            "r2 ABO/Food counterfactual split allocation drifted: "
            f"abo={dict(abo_by_split)}, food={dict(food_by_split)}"
        )
    if len(tail_multi_components) != _R2_REQUIRED_NONDEV_COMPONENT_COUNTS["multi"]:
        raise ValueError("r2 non-dev Multi components are not batch-isolated exactly once")

    counterfactual_components_by_split = {
        split: abo_by_split[split] + food_by_split[split]
        for split in _R2_TAIL_SPLITS
    }
    expected_counterfactual_components = {"opt_pool": 18, "val": 5, "test_frozen": 7}
    if counterfactual_components_by_split != expected_counterfactual_components:
        raise AssertionError("r2 counterfactual component totals drifted")
    return R2CorePlanAudit(
        capability_counts_by_split=actual_capability_counts,
        boundary_counts_by_split=actual_boundary_counts,
        tail_batch_capability_counts=tail_batch_capability_counts,
        unique_component_count=len(items_by_component),
        max_component_reuse=max_component_reuse,
        capacity_required_triples_by_capability=dict(capacity_triples),
        counterfactual_components_by_split=counterfactual_components_by_split,
        tail_multi_component_count=len(tail_multi_components),
    )


def asset_identity_for_path(image_path: str | Path) -> tuple[str, str]:
    """Return provisional asset/group IDs for the current path-based catalog.

    The grouping policy is recorded explicitly so a later content/product/pHash
    catalog can migrate these IDs instead of treating them as universal IDs.
    """

    normalized = Path(image_path).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"asset.path.{digest}", f"leakage.path.{digest}"


def _asset_identity(
    image_path: str,
    *,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> tuple[str, str]:
    if asset_catalog is not None:
        resolution = asset_catalog.resolve_path(image_path)
        return resolution.asset.asset_id, resolution.leakage_group_id
    if allow_provisional_asset_groups:
        return asset_identity_for_path(image_path)
    raise ValueError(
        "formal planning requires an asset catalog; "
        "set allow_provisional_asset_groups=True only for provisional/debug plans"
    )


def _plan_catalog_binding(
    asset_catalog: AssetCatalog | None,
    *,
    allow_provisional_asset_groups: bool,
) -> tuple[str | None, str]:
    if asset_catalog is not None:
        return (
            asset_catalog.manifest.catalog_sha256,
            asset_catalog.manifest.leakage_policy_version,
        )
    if allow_provisional_asset_groups:
        return None, GROUPING_POLICY_VERSION
    raise ValueError(
        "formal planning requires an asset catalog; "
        "provisional path identities require explicit opt-in"
    )


def _planning_fields(
    *,
    image_path: str,
    intent: Intent,
    desired_capability: str | None = None,
    batch_id: str,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
    capability_assignments: Mapping[CapabilityAssignmentKey, CapabilityAssignment]
    | None,
) -> dict[str, object]:
    asset_id, leakage_group_id = _asset_identity(
        image_path,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    assignment = (
        None
        if capability_assignments is None
        else capability_assignments.get((asset_id, image_path, intent))
    )
    if assignment is None:
        if capability_assignments is not None:
            raise ValueError(
                "missing capability assignment for selected asset/path/intent: "
                f"{asset_id}, {image_path}, {intent}"
            )
        if asset_catalog is not None or not allow_provisional_asset_groups:
            raise ValueError(
                "formal planning requires explicit per-asset capability assignments"
            )
        capability = desired_capability or capability_for_intent(intent)
        if (
            desired_capability is not None
            and MVP_CAPABILITY_INTENTS.get(desired_capability) != intent
        ):
            raise ValueError(
                f"capability {desired_capability} does not belong to intent {intent}"
            )
        acceptable_capabilities = [capability]
        requires_card = requires_card_for_intent(intent)
    else:
        if (
            desired_capability is not None
            and assignment.canonical_capability != desired_capability
        ):
            raise ValueError(
                "selected capability assignment does not match the planned quota: "
                f"{assignment.canonical_capability} != {desired_capability}"
            )
        capability = assignment.canonical_capability
        acceptable_capabilities = list(assignment.acceptable_capabilities)
        requires_card = assignment.requires_card
    assignment_id = None if assignment is None else assignment.assignment_id
    assignment_source_sha256 = (
        None if assignment is None else assignment.source_artifact_sha256
    )
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "task_spec_version": PHASE3_TASK_SPEC_VERSION,
        "asset_id": asset_id,
        "leakage_group_id": leakage_group_id,
        "template_family": _template_family_for_batch(batch_id),
        "generator_batch_id": batch_id,
        "canonical_intent": intent,
        "canonical_capability": capability,
        "acceptable_capabilities": acceptable_capabilities,
        "requires_card": requires_card,
        "capability_assignment_id": assignment_id,
        "capability_assignment_source_sha256": assignment_source_sha256,
    }


def _template_family_for_batch(batch_id: str) -> str:
    """Assign one real prompt family to an isolated two-batch generation block."""

    try:
        batch_number = int(batch_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"无法从 batch_id 推导 template family: {batch_id}") from exc
    if batch_number < 1 or not (
        batch_id.startswith("dev-mini-")
        or batch_id.startswith("core-")
        or batch_id.startswith("full-")
    ):
        raise ValueError(f"无法从 batch_id 推导 template family: {batch_id}")
    block_number = (batch_number - 1) // 2 + 1
    return f"generation-prompt-family-v1/block-{block_number:03d}"


def discover_image_pools(image_root: str | Path) -> dict[Intent, list[Path]]:
    """发现五意图图片池，返回相对 data/clean 的审计路径。"""

    image_root = Path(image_root)
    pools: dict[Intent, list[Path]] = {}
    for intent in INTENT_ORDER:
        directory = image_root / intent
        if not directory.is_dir():
            raise ValueError(f"{intent} 图片目录不存在: {directory}")
        paths = sorted(
            (
                Path("query_images") / path.relative_to(image_root)
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            ),
            key=lambda path: path.as_posix(),
        )
        if not paths:
            raise ValueError(f"{intent} 图片不足: 未发现可用图片")
        pools[intent] = paths
    return pools


def build_dev_mini_plan(
    image_root: str | Path,
    *,
    seed: int = 20260711,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    capability_assignments: Sequence[CapabilityAssignment] | None = None,
    expected_capability_assignments_sha256: str | None = None,
) -> CorpusPlan:
    """构建 200 条、8 批、六 capability、40 条 boundary 的开发计划。"""

    catalog_sha256, leakage_policy_version = _plan_catalog_binding(
        asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    assignment_index = (
        _index_capability_assignments(capability_assignments)
        if capability_assignments is not None
        else None
    )
    assignment_sha256 = _capability_assignments_digest(
        capability_assignments,
        expected_sha256=expected_capability_assignments_sha256,
        formal=asset_catalog is not None,
    )
    discovered = discover_image_pools(image_root)
    rng = random.Random(seed)
    pools: dict[str, deque[Path]] = {}
    for capability in MVP_CAPABILITY_ORDER:
        intent = MVP_CAPABILITY_INTENTS[capability]
        shuffled = list(discovered[intent])
        rng.shuffle(shuffled)
        if assignment_index is not None:
            eligible: list[Path] = []
            for path in shuffled:
                image_path = path.as_posix()
                asset_id, _ = _asset_identity(
                    image_path,
                    asset_catalog=asset_catalog,
                    allow_provisional_asset_groups=allow_provisional_asset_groups,
                )
                assignment = assignment_index.get((asset_id, image_path, intent))
                if assignment is None:
                    raise ValueError(
                        "missing capability assignment for discovered "
                        f"asset/path/intent: {asset_id}, {image_path}, {intent}"
                    )
                if assignment.canonical_capability == capability:
                    eligible.append(path)
            shuffled = eligible
        pools[capability] = deque(shuffled)

    def supports_required_capabilities(
        path: Path,
        required_capabilities: Sequence[str],
    ) -> bool:
        if assignment_index is None:
            return True
        image_path = path.as_posix()
        asset_id, _ = _asset_identity(
            image_path,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional_asset_groups,
        )
        return all(
            (
                assignment := assignment_index.get(
                    (
                        asset_id,
                        image_path,
                        MVP_CAPABILITY_INTENTS[required_capability],
                    )
                )
            )
            is not None
            and assignment.canonical_capability == required_capability
            for required_capability in required_capabilities
        )

    # Reserve one independent triplet image per dev batch before ordinary Exact
    # slots consume that pool. Exact-only candidates retain their deterministic
    # order and remain available to ordinary Exact slots.
    triplet_pool: deque[Path] = deque()
    exact_capability = _CROSS_INTENT_TRIPLET[0]
    regular_exact_pool: deque[Path] = deque()
    reserved_triplet_components: set[str] = set()
    required_triplet_count = len(DEV_MINI_BATCH_CAPABILITY_COUNTS)
    while pools[exact_capability]:
        path = pools[exact_capability].popleft()
        image_path = path.as_posix()
        _, component_id = _asset_identity(
            image_path,
            asset_catalog=asset_catalog,
            allow_provisional_asset_groups=allow_provisional_asset_groups,
        )
        if (
            len(triplet_pool) < required_triplet_count
            and component_id not in reserved_triplet_components
            and supports_required_capabilities(path, _CROSS_INTENT_TRIPLET)
        ):
            triplet_pool.append(path)
            reserved_triplet_components.add(component_id)
        else:
            regular_exact_pool.append(path)
    pools[exact_capability] = regular_exact_pool
    if len(triplet_pool) != required_triplet_count:
        raise ValueError(
            "exact_match cross-intent triplet images are insufficient: "
            f"required {required_triplet_count}, found {len(triplet_pool)}"
        )

    queries: list[PlannedQuery] = []
    global_index = 0
    component_batch: dict[str, str] = {}

    def take_image(
        capability: str,
        batch_id: str,
        *,
        required_capabilities: Sequence[str] = (),
        candidate_pool: deque[Path] | None = None,
    ) -> str:
        intent = MVP_CAPABILITY_INTENTS[capability]
        active_pool = pools[capability] if candidate_pool is None else candidate_pool
        deferred: deque[Path] = deque()
        candidates_to_scan = len(active_pool)
        for _ in range(candidates_to_scan):
            path = active_pool.popleft()
            image_path = path.as_posix()
            _, component_id = _asset_identity(
                image_path,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional_asset_groups,
            )
            if component_id in component_batch:
                continue
            if not supports_required_capabilities(path, required_capabilities):
                deferred.append(path)
                continue
            active_pool.extendleft(reversed(deferred))
            component_batch[component_id] = batch_id
            return image_path
        active_pool.extendleft(reversed(deferred))
        raise ValueError(
            f"{intent}/{capability} 图片不足："
            "独立 asset component 无法满足 dev_mini 计划"
        )

    def append_query(
        *,
        batch_id: str,
        position: int,
        image_path: str,
        capability: str,
        is_boundary: bool,
        boundary_strategy: str | None = None,
        boundary_group_id: str | None = None,
    ) -> None:
        nonlocal global_index
        intent = MVP_CAPABILITY_INTENTS[capability]
        global_index += 1
        queries.append(
            PlannedQuery(
                plan_id=f"dm-{global_index:03d}",
                batch_id=batch_id,
                position=position,
                image_path=image_path,
                **_planning_fields(
                    image_path=image_path,
                    intent=intent,
                    desired_capability=capability,
                    batch_id=batch_id,
                    asset_catalog=asset_catalog,
                    allow_provisional_asset_groups=allow_provisional_asset_groups,
                    capability_assignments=assignment_index,
                ),
                is_boundary=is_boundary,
                boundary_strategy=boundary_strategy,
                boundary_group_id=boundary_group_id,
                provisional_split="dev_mini",
            )
        )

    for batch_offset, target_counts in enumerate(
        DEV_MINI_BATCH_CAPABILITY_COUNTS, start=1
    ):
        batch_id = f"dev-mini-{batch_offset:03d}"
        remaining = dict(target_counts)
        position = 0

        triplet_image = take_image(
            "product.exact_match",
            batch_id,
            required_capabilities=_CROSS_INTENT_TRIPLET,
            candidate_pool=triplet_pool,
        )
        triplet_group = f"triplet-{batch_offset:03d}"
        for capability in _CROSS_INTENT_TRIPLET:
            position += 1
            append_query(
                batch_id=batch_id,
                position=position,
                image_path=triplet_image,
                capability=capability,
                is_boundary=True,
                boundary_strategy="cross_intent_triplet",
                boundary_group_id=triplet_group,
            )
            remaining[capability] -= 1

        for capability in _NATURAL_BOUNDARY_CAPABILITY_SCHEDULE[batch_offset - 1]:
            position += 1
            append_query(
                batch_id=batch_id,
                position=position,
                image_path=take_image(capability, batch_id),
                capability=capability,
                is_boundary=True,
                boundary_strategy="natural_ambiguity",
            )
            remaining[capability] -= 1

        for capability in MVP_CAPABILITY_ORDER:
            for _ in range(remaining[capability]):
                position += 1
                append_query(
                    batch_id=batch_id,
                    position=position,
                    image_path=take_image(capability, batch_id),
                    capability=capability,
                    is_boundary=False,
                )
        if position != 25:
            raise AssertionError(f"{batch_id} 计划数量异常: {position}")

    plan = CorpusPlan(
        scope="dev_mini",
        seed=seed,
        asset_catalog_sha256=catalog_sha256,
        capability_assignments_sha256=assignment_sha256,
        leakage_policy_version=leakage_policy_version,
        queries=queries,
    )
    if asset_catalog is not None:
        audit_plan_catalog_binding(plan, asset_catalog)
    return plan


def audit_plan_catalog_binding(
    plan: CorpusPlan,
    asset_catalog: AssetCatalog,
) -> None:
    """Fail closed unless every planned asset is bound to one catalog component."""

    asset_catalog.require_verified_files()
    if plan.asset_catalog_sha256 != asset_catalog.manifest.catalog_sha256:
        raise ValueError("plan asset catalog hash 与加载的 catalog 不一致")
    if plan.leakage_policy_version != asset_catalog.manifest.leakage_policy_version:
        raise ValueError("plan leakage policy 与加载的 catalog 不一致")

    batches_by_component: dict[str, set[str]] = {}
    for item in plan.queries:
        resolution = asset_catalog.verify_reference(
            asset_id=item.asset_id,
            image_path=item.image_path,
            leakage_group_id=item.leakage_group_id,
        )
        component_id = resolution.leakage_group_id
        batches_by_component.setdefault(component_id, set()).add(
            item.generator_batch_id
        )

    crossing = {
        component_id: sorted(batch_ids)
        for component_id, batch_ids in batches_by_component.items()
        if len(batch_ids) > 1
    }
    if crossing:
        sample = dict(list(sorted(crossing.items()))[:5])
        raise ValueError(
            f"catalog component 跨 generator batch，必须重新规划: {sample}"
        )
    asset_catalog.verify_asset_ids(item.asset_id for item in plan.queries)


def _audit_activation_binding(
    plan: CorpusPlan,
    *,
    asset_catalog: AssetCatalog | None,
    allow_provisional_asset_groups: bool,
) -> None:
    if plan.asset_catalog_sha256 is None:
        if asset_catalog is not None:
            raise ValueError("provisional plan 不能伪装为 catalog-bound plan")
        if not allow_provisional_asset_groups:
            raise ValueError(
                "provisional plan activation requires explicit "
                "allow_provisional_asset_groups=True"
            )
        return
    if asset_catalog is None:
        raise ValueError("catalog-bound plan activation requires the asset catalog")
    audit_plan_catalog_binding(plan, asset_catalog)


def write_dev_mini_plan(plan: CorpusPlan, output_path: str | Path) -> tuple[Path, Path]:
    """以规范 JSON 写计划与哈希 manifest；不同内容禁止覆盖。"""

    if plan.scope != "dev_mini":
        raise ValueError("write_dev_mini_plan 只接受 dev_mini scope")
    output_path = Path(output_path)
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    plan_bytes = _canonical_json_bytes(plan.model_dump(mode="json"))
    manifest = PlanManifest(
        scope="dev_mini",
        count=len(plan.queries),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        asset_catalog_sha256=plan.asset_catalog_sha256,
        capability_assignments_sha256=plan.capability_assignments_sha256,
        leakage_policy_version=plan.leakage_policy_version,
    )
    manifest_bytes = _canonical_json_bytes(manifest.model_dump(mode="json"))

    if output_path.exists() or manifest_path.exists():
        if (
            output_path.is_file()
            and manifest_path.is_file()
            and output_path.read_bytes() == plan_bytes
            and manifest_path.read_bytes() == manifest_bytes
        ):
            return output_path, manifest_path
        raise FileExistsError("计划目标已存在且内容不同，拒绝覆盖")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output_path, plan_bytes)
    try:
        _atomic_create(manifest_path, manifest_bytes)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path, manifest_path


def _extend_plan_from_dev(
    dev_plan: CorpusPlan,
    image_pools: Mapping[Intent, list[Path]],
    *,
    target_scope: Literal["core", "full"],
    target_intent_counts: Mapping[Intent, int],
    target_boundary_counts: Mapping[Intent, int],
    seed: int,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    capability_assignments: Sequence[CapabilityAssignment] | None = None,
    expected_capability_assignments_sha256: str | None = None,
) -> CorpusPlan:
    """Retain the dev_mini prefix and deterministically append a profile tail."""

    catalog_sha256, leakage_policy_version = _plan_catalog_binding(
        asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    assignment_index = (
        _index_capability_assignments(capability_assignments)
        if capability_assignments is not None
        else None
    )
    assignment_sha256 = _capability_assignments_digest(
        capability_assignments,
        expected_sha256=expected_capability_assignments_sha256,
        formal=asset_catalog is not None,
    )
    if dev_plan.scope != "dev_mini" or len(dev_plan.queries) != 200:
        raise ValueError(f"{target_scope} plan 只能从完整 dev_mini plan 延伸")
    if (
        dev_plan.asset_catalog_sha256 != catalog_sha256
        or dev_plan.leakage_policy_version != leakage_policy_version
    ):
        raise ValueError(
            f"dev_mini plan 与 {target_scope} planner 的 asset catalog 绑定不一致"
        )
    if asset_catalog is not None:
        audit_plan_catalog_binding(dev_plan, asset_catalog)
    _audit_planned_assignment_binding(
        dev_plan.queries,
        assignment_index,
        formal=asset_catalog is not None,
    )
    if set(image_pools) != set(INTENT_ORDER):
        raise ValueError(f"{target_scope} image_pools 必须完整覆盖五个意图")

    dev_intents = Counter(item.canonical_intent for item in dev_plan.queries)
    dev_boundaries = Counter(
        item.canonical_intent for item in dev_plan.queries if item.is_boundary
    )
    pair_counts = Counter(
        (item.image_path, item.canonical_intent) for item in dev_plan.queries
    )
    if pair_counts and max(pair_counts.values()) > 3:
        raise ValueError("dev_mini 已违反 image_path/intent 最多复用 3 次")

    rng = random.Random(seed)
    remaining_by_intent: dict[Intent, int] = {}
    remaining_boundary_by_intent: dict[Intent, int] = {}
    candidate_components: dict[Intent, list[tuple[str, str]]] = {}
    dev_leakage_groups = {item.leakage_group_id for item in dev_plan.queries}
    for intent in INTENT_ORDER:
        remaining = target_intent_counts[intent] - dev_intents[intent]
        remaining_boundary = target_boundary_counts[intent] - dev_boundaries[intent]
        if remaining < 0 or remaining_boundary < 0 or remaining_boundary > remaining:
            raise ValueError(
                f"dev_mini 的 {intent} 计数无法延伸到 {target_scope} 目标"
            )
        remaining_by_intent[intent] = remaining
        remaining_boundary_by_intent[intent] = remaining_boundary

        candidates: list[tuple[str, str]] = []
        seen_components: set[str] = set()
        for path in sorted({Path(path).as_posix() for path in image_pools[intent]}):
            _, component_id = _asset_identity(
                path,
                asset_catalog=asset_catalog,
                allow_provisional_asset_groups=allow_provisional_asset_groups,
            )
            if component_id in dev_leakage_groups or component_id in seen_components:
                continue
            seen_components.add(component_id)
            candidates.append((component_id, path))
        if not candidates:
            raise ValueError(f"{intent} {target_scope} 独立 asset component 池为空")
        rng.shuffle(candidates)
        candidate_components[intent] = candidates

    appended_count = sum(remaining_by_intent.values())
    if appended_count % 25:
        raise AssertionError(f"{target_scope} 追加 query 数量必须可被 25 整除")
    batch_count = appended_count // 25
    singleton_counts = _allocate_singleton_counts(
        remaining_by_intent,
        {intent: len(candidate_components[intent]) for intent in INTENT_ORDER},
        total_singletons=batch_count,
    )
    singleton_groups: list[tuple[Intent, list[bool], str]] = []
    triple_groups: list[tuple[Intent, list[bool], str]] = []
    selected_components = set(dev_leakage_groups)
    for intent in INTENT_ORDER:
        flags = [True] * remaining_boundary_by_intent[intent] + [False] * (
            remaining_by_intent[intent] - remaining_boundary_by_intent[intent]
        )
        rng.shuffle(flags)
        single_count = singleton_counts[intent]
        group_count = single_count + (len(flags) - single_count) // 3
        available = [
            (component_id, path)
            for component_id, path in candidate_components[intent]
            if component_id not in selected_components
        ]
        if len(available) < group_count:
            raise ValueError(
                f"{intent} {target_scope} 独立 asset component 不足：需要 {group_count}，"
                f"实际 {len(available)}"
            )
        selected = available[:group_count]
        selected_components.update(component_id for component_id, _ in selected)
        paths = iter(path for _, path in selected)
        for flag in flags[:single_count]:
            singleton_groups.append((intent, [flag], next(paths)))
        grouped_flags = flags[single_count:]
        for offset in range(0, len(grouped_flags), 3):
            triple_groups.append(
                (intent, grouped_flags[offset : offset + 3], next(paths))
            )
    if len(singleton_groups) != batch_count or len(triple_groups) != batch_count * 8:
        raise AssertionError(
            f"{target_scope} asset-group packing 无法形成 {batch_count} 个 25-query batch"
        )
    rng.shuffle(singleton_groups)
    rng.shuffle(triple_groups)

    appended: list[PlannedQuery] = []
    global_index = 200
    for batch_offset in range(batch_count):
        batch_number = batch_offset + 9
        batch_id = f"{target_scope}-{batch_number:03d}"
        asset_groups = [singleton_groups[batch_offset]] + triple_groups[
            batch_offset * 8 : (batch_offset + 1) * 8
        ]
        rng.shuffle(asset_groups)
        position = 0
        for intent, boundary_flags, image_path in asset_groups:
            for is_boundary in boundary_flags:
                global_index += 1
                position += 1
                appended.append(
                    PlannedQuery(
                        plan_id=f"{target_scope}-{global_index:04d}",
                        batch_id=batch_id,
                        position=position,
                        image_path=image_path,
                        **_planning_fields(
                            image_path=image_path,
                            intent=intent,
                            batch_id=batch_id,
                            asset_catalog=asset_catalog,
                            allow_provisional_asset_groups=allow_provisional_asset_groups,
                            capability_assignments=assignment_index,
                        ),
                        is_boundary=is_boundary,
                        boundary_strategy=(
                            "natural_ambiguity" if is_boundary else None
                        ),
                        provisional_split="opt_pool",
                    )
                )
        if position != 25:
            raise AssertionError(f"{batch_id} asset-group packing 数量异常")
    plan = CorpusPlan(
        scope=target_scope,
        seed=seed,
        asset_catalog_sha256=catalog_sha256,
        capability_assignments_sha256=assignment_sha256,
        leakage_policy_version=leakage_policy_version,
        queries=[*dev_plan.queries, *appended],
    )
    if asset_catalog is not None:
        audit_plan_catalog_binding(plan, asset_catalog)
    return plan


def extend_core_plan(
    dev_plan: CorpusPlan,
    image_pools: Mapping[Intent, list[Path]],
    *,
    seed: int,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    capability_assignments: Sequence[CapabilityAssignment] | None = None,
    expected_capability_assignments_sha256: str | None = None,
) -> CorpusPlan:
    """Build the registered 1,500-query core profile from dev_mini."""

    return _extend_plan_from_dev(
        dev_plan,
        image_pools,
        target_scope="core",
        target_intent_counts=CORE_INTENT_COUNTS,
        target_boundary_counts=CORE_BOUNDARY_COUNTS,
        seed=seed,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
        capability_assignments=capability_assignments,
        expected_capability_assignments_sha256=(
            expected_capability_assignments_sha256
        ),
    )


def build_core_plan_from_resources(
    data_root: str | Path,
    *,
    seed: int,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    capability_assignments: Sequence[CapabilityAssignment] | None = None,
    expected_capability_assignments_sha256: str | None = None,
) -> tuple[CorpusPlan, CorpusPlan]:
    """Create a fresh core plan and its reproducible 200-query dev prefix.

    Portfolio core runs deliberately rebuild the prefix against the same core
    asset catalog rather than inheriting an older dev_mini catalog hash.
    """

    dev_plan = build_dev_mini_plan(
        data_root,
        seed=seed,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
        capability_assignments=capability_assignments,
        expected_capability_assignments_sha256=(
            expected_capability_assignments_sha256
        ),
    )
    return dev_plan, extend_core_plan(
        dev_plan,
        discover_image_pools(data_root),
        seed=seed,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
        capability_assignments=capability_assignments,
        expected_capability_assignments_sha256=(
            expected_capability_assignments_sha256
        ),
    )


def extend_full_plan(
    dev_plan: CorpusPlan,
    image_pools: Mapping[Intent, list[Path]],
    *,
    seed: int,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
    capability_assignments: Sequence[CapabilityAssignment] | None = None,
    expected_capability_assignments_sha256: str | None = None,
) -> CorpusPlan:
    """Build the registered 4,500-query full profile from dev_mini."""

    return _extend_plan_from_dev(
        dev_plan,
        image_pools,
        target_scope="full",
        target_intent_counts=FULL_INTENT_COUNTS,
        target_boundary_counts=FULL_BOUNDARY_COUNTS,
        seed=seed,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
        capability_assignments=capability_assignments,
        expected_capability_assignments_sha256=(
            expected_capability_assignments_sha256
        ),
    )


def _allocate_singleton_counts(
    totals: Mapping[Intent, int],
    candidate_counts: Mapping[Intent, int],
    *,
    total_singletons: int,
) -> dict[Intent, int]:
    """Choose singleton asset groups so every 25-query batch is 1 + 8*3."""

    base = {intent: totals[intent] % 3 for intent in INTENT_ORDER}
    remainder = total_singletons - sum(base.values())
    if remainder < 0 or remainder % 3:
        raise ValueError("full singleton packing 与 intent totals 不兼容")
    units = remainder // 3
    total = sum(totals.values())
    allocations = {intent: units * totals[intent] // total for intent in INTENT_ORDER}
    missing = units - sum(allocations.values())
    ranked = sorted(
        INTENT_ORDER,
        key=lambda intent: (-(units * totals[intent] % total), intent),
    )
    for intent in ranked[:missing]:
        allocations[intent] += 1

    def capacity_units(intent: Intent) -> int:
        # groups = singletons + (total-singletons)/3 <= candidate_count
        max_singletons = (3 * candidate_counts[intent] - totals[intent]) // 2
        return max(0, (max_singletons - base[intent]) // 3)

    overflow = 0
    for intent in INTENT_ORDER:
        cap = capacity_units(intent)
        if allocations[intent] > cap:
            overflow += allocations[intent] - cap
            allocations[intent] = cap
    while overflow:
        candidates = [
            intent
            for intent in INTENT_ORDER
            if allocations[intent] < capacity_units(intent)
        ]
        if not candidates:
            raise ValueError("full 图片池不足以隔离 generator batch 与 asset group")
        intent = max(candidates, key=lambda item: (totals[item], item))
        allocations[intent] += 1
        overflow -= 1
    return {intent: base[intent] + 3 * allocations[intent] for intent in INTENT_ORDER}


def _write_derived_plan(
    plan: CorpusPlan,
    output_path: str | Path,
    *,
    scope: Literal["core", "full"],
    parent_plan_sha256: str,
) -> tuple[Path, Path]:
    """Create-only write of a dev_mini-derived plan and parent binding."""

    if plan.scope != scope:
        raise ValueError(f"write_{scope}_plan 只接受 {scope} scope")
    output_path = Path(output_path)
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    plan_bytes = _canonical_json_bytes(plan.model_dump(mode="json"))
    manifest = PlanManifest(
        scope=scope,
        count=len(plan.queries),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        parent_plan_sha256=parent_plan_sha256,
        asset_catalog_sha256=plan.asset_catalog_sha256,
        capability_assignments_sha256=plan.capability_assignments_sha256,
        leakage_policy_version=plan.leakage_policy_version,
    )
    manifest_bytes = _canonical_json_bytes(manifest.model_dump(mode="json"))

    if output_path.exists() or manifest_path.exists():
        if (
            output_path.is_file()
            and manifest_path.is_file()
            and output_path.read_bytes() == plan_bytes
            and manifest_path.read_bytes() == manifest_bytes
        ):
            return output_path, manifest_path
        raise FileExistsError(f"{scope} 计划目标已存在且内容不同，拒绝覆盖")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output_path, plan_bytes)
    try:
        _atomic_create(manifest_path, manifest_bytes)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path, manifest_path


def write_core_plan(
    plan: CorpusPlan,
    output_path: str | Path,
    *,
    parent_plan_sha256: str,
) -> tuple[Path, Path]:
    return _write_derived_plan(
        plan,
        output_path,
        scope="core",
        parent_plan_sha256=parent_plan_sha256,
    )


def write_full_plan(
    plan: CorpusPlan,
    output_path: str | Path,
    *,
    parent_plan_sha256: str,
) -> tuple[Path, Path]:
    return _write_derived_plan(
        plan,
        output_path,
        scope="full",
        parent_plan_sha256=parent_plan_sha256,
    )


def load_plan(plan_path: str | Path) -> tuple[CorpusPlan, PlanManifest]:
    plan_path = Path(plan_path)
    manifest_path = plan_path.with_name(f"{plan_path.stem}.manifest.json")
    try:
        plan_bytes = plan_path.read_bytes()
        manifest = PlanManifest.model_validate_json(manifest_path.read_bytes())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"plan 或 manifest 不存在: {exc.filename}") from None
    except ValidationError as exc:
        raise ValueError(f"plan manifest 校验失败: {exc}") from exc
    actual_hash = hashlib.sha256(plan_bytes).hexdigest()
    if actual_hash != manifest.plan_sha256:
        raise ValueError("plan 哈希与 manifest 不一致")
    try:
        plan = CorpusPlan.model_validate_json(plan_bytes)
    except ValidationError as exc:
        raise ValueError(f"plan 内容校验失败: {exc}") from exc
    if (
        plan.scope != manifest.scope
        or len(plan.queries) != manifest.count
        or plan.asset_catalog_sha256 != manifest.asset_catalog_sha256
        or plan.capability_assignments_sha256 != manifest.capability_assignments_sha256
        or plan.leakage_policy_version != manifest.leakage_policy_version
    ):
        raise ValueError("plan 与 manifest scope/count/catalog binding 不一致")
    return plan, manifest


def activate_dev_plan(
    plan_path: str | Path,
    queries_root: str | Path,
    *,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> ActivePlanPointer:
    plan_path = Path(plan_path).resolve()
    plan, manifest = load_plan(plan_path)
    if plan.scope != "dev_mini":
        raise ValueError("activate_dev_plan 只接受 dev_mini plan")
    _audit_activation_binding(
        plan,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    pointer = ActivePlanPointer(
        scope="dev_mini",
        plan_path=str(plan_path),
        plan_sha256=manifest.plan_sha256,
        asset_catalog_sha256=manifest.asset_catalog_sha256,
        leakage_policy_version=manifest.leakage_policy_version,
    )
    active_path = Path(queries_root) / "plans" / "active.json"
    pointer_bytes = _canonical_json_bytes(pointer.model_dump(mode="json"))
    if active_path.exists():
        if active_path.read_bytes() == pointer_bytes:
            return pointer
        raise FileExistsError("active plan 已存在且不同，拒绝隐式切换")
    active_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(active_path, pointer_bytes)
    return pointer


def _activate_derived_plan(
    queries_root: str | Path,
    derived_plan_path: str | Path,
    *,
    scope: Literal["core", "full"],
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> ActivePlanPointer:
    """Switch a completed dev_mini root to an explicitly derived profile."""

    queries_root = Path(queries_root)
    derived_plan_path = Path(derived_plan_path).resolve()
    derived_plan, derived_manifest = load_plan(derived_plan_path)
    if derived_plan.scope != scope or derived_manifest.parent_plan_sha256 is None:
        raise ValueError(f"activate_{scope}_plan 只接受带 parent 的 {scope} plan")
    _audit_activation_binding(
        derived_plan,
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )
    target = ActivePlanPointer(
        scope=scope,
        plan_path=str(derived_plan_path),
        plan_sha256=derived_manifest.plan_sha256,
        asset_catalog_sha256=derived_manifest.asset_catalog_sha256,
        leakage_policy_version=derived_manifest.leakage_policy_version,
    )
    active_path = queries_root / "plans" / "active.json"
    target_bytes = _canonical_json_bytes(target.model_dump(mode="json"))
    already_active = active_path.is_file() and active_path.read_bytes() == target_bytes
    projected_parent = CorpusPlan(
        scope="dev_mini",
        seed=derived_plan.seed,
        asset_catalog_sha256=derived_plan.asset_catalog_sha256,
        capability_assignments_sha256=derived_plan.capability_assignments_sha256,
        leakage_policy_version=derived_plan.leakage_policy_version,
        queries=derived_plan.queries[:200],
    )
    projected_parent_sha256 = hashlib.sha256(
        _canonical_json_bytes(projected_parent.model_dump(mode="json"))
    ).hexdigest()
    if projected_parent_sha256 != derived_manifest.parent_plan_sha256:
        raise ValueError(
            f"{scope} plan 的前 200 条与 seed 无法重建 parent plan hash"
        )
    if not already_active:
        _, dev_plan, dev_manifest, active = read_active_plan(queries_root)
        if active.scope != "dev_mini":
            raise ValueError(f"{scope} 激活前 active plan 必须是 dev_mini")
        if (
            dev_plan != projected_parent
            or dev_manifest.plan_sha256 != projected_parent_sha256
            or active.plan_sha256 != projected_parent_sha256
        ):
            raise ValueError(
                f"{scope} manifest 的 parent plan hash 与 active dev_mini 不匹配"
            )

    staging_root = queries_root / "staging"
    if staging_root.is_dir() and any(staging_root.iterdir()):
        raise ValueError(f"存在未决 staging，禁止激活 {scope} plan")

    from skillchain.synthesis.batches import AcceptedLedgerEntry

    ledger_path = queries_root / "accepted-ledger.jsonl"
    try:
        ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        ledger_lines = []
    entries: list[AcceptedLedgerEntry] = []
    for line_number, line in enumerate(ledger_lines, start=1):
        try:
            entries.append(AcceptedLedgerEntry.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(
                f"前 8 批 ledger 第 {line_number} 行校验失败: {exc}"
            ) from exc
    expected_bases = {f"dev-mini-{index:03d}" for index in range(1, 9)}
    if (
        len(entries) != 8
        or {entry.base_batch_id for entry in entries} != expected_bases
    ):
        raise ValueError("前 8 批必须全部 accepted 才能激活 full plan")
    if any(entry.count != 25 for entry in entries):
        raise ValueError("前 8 批 accepted count 必须各为 25")
    if any(entry.plan_sha256 != derived_manifest.parent_plan_sha256 for entry in entries):
        raise ValueError("前 8 批 ledger 的 parent plan hash 不匹配")
    missing_directories = [
        entry.batch_id
        for entry in entries
        if not (queries_root / "accepted" / entry.batch_id).is_dir()
    ]
    if missing_directories:
        raise ValueError("前 8 批 accepted 目录缺失: " + ", ".join(missing_directories))

    if not already_active:
        atomic_replace_file(active_path, target_bytes)
    return target


def activate_core_plan(
    queries_root: str | Path,
    core_plan_path: str | Path,
    *,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> ActivePlanPointer:
    return _activate_derived_plan(
        queries_root,
        core_plan_path,
        scope="core",
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )


def activate_full_plan(
    queries_root: str | Path,
    full_plan_path: str | Path,
    *,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_asset_groups: bool = False,
) -> ActivePlanPointer:
    return _activate_derived_plan(
        queries_root,
        full_plan_path,
        scope="full",
        asset_catalog=asset_catalog,
        allow_provisional_asset_groups=allow_provisional_asset_groups,
    )


def read_active_plan(
    queries_root: str | Path,
) -> tuple[Path, CorpusPlan, PlanManifest, ActivePlanPointer]:
    active_path = Path(queries_root) / "plans" / "active.json"
    try:
        pointer = ActivePlanPointer.model_validate_json(active_path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"active plan 不存在: {active_path}") from None
    except ValidationError as exc:
        raise ValueError(f"active plan 校验失败: {exc}") from exc
    plan_path = Path(pointer.plan_path)
    plan, manifest = load_plan(plan_path)
    if (
        manifest.plan_sha256 != pointer.plan_sha256
        or plan.scope != pointer.scope
        or manifest.asset_catalog_sha256 != pointer.asset_catalog_sha256
        or manifest.leakage_policy_version != pointer.leakage_policy_version
    ):
        raise ValueError("active pointer 与 plan/manifest/catalog binding 不一致")
    return plan_path, plan, manifest, pointer


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_create(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"目标已存在，拒绝覆盖: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
