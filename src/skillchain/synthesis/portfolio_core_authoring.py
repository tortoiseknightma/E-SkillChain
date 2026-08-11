"""Deterministic, text-free r2 authoring inputs.

This module deliberately stops before image inspection and corpus composition.
It turns a frozen r2 plan plus its split/reuse sidecars into two different
in-memory objects:

* an internal work order, which may retain catalog binding information; and
* a model-visible author packet, which contains only opaque asset aliases and
  the canonical task/realism card needed to write a trajectory.

There is no filesystem, model, image, or corpus-text I/O here.  Publication of
an author packet and materialisation of its aliases belong to the orchestration
layer and must remain create-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.schemas import Intent
from skillchain.synthesis.planning import (
    CORE_R2_CAPABILITY_BOUNDARY_COUNTS,
    CORE_R2_CAPABILITY_COUNTS,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FinalSplit = Literal["dev_mini", "opt_pool", "val", "test_frozen"]
TrajectoryShape = Literal["single_turn", "three_turn"]
InteractionPattern = Literal[
    "direct_request",
    "underspecified_clarification",
    "constraint_correction",
    "no_result_relaxation",
    "goal_change_or_multi_query",
]
ExpressionLevel = Literal["S0", "S1", "S2", "S3"]
UserStyle = Literal[
    "terse_fragment",
    "colloquial",
    "neutral_complete",
    "polite",
    "code_mixed_or_numeric",
    "typo_or_asr_like",
]
ConstraintLevel = Literal["0", "1", "2", "3_plus"]
AmbiguitySubtype = Literal[
    "none",
    "resolved_near_boundary",
    "clarification_required",
]

AUTHORING_SCHEMA_VERSION = 2
REALISM_ALLOCATION_POLICY_VERSION = "r2-split-capability-boundary-lrm-v1"
PROMPT_RECIPE_SCHEMA_VERSION = 1

_FINAL_SPLIT_ORDER: tuple[FinalSplit, ...] = (
    "dev_mini",
    "opt_pool",
    "val",
    "test_frozen",
)
_INTERACTION_ORDER: tuple[InteractionPattern, ...] = (
    "direct_request",
    "underspecified_clarification",
    "constraint_correction",
    "no_result_relaxation",
    "goal_change_or_multi_query",
)
_EXPRESSION_ORDER: tuple[ExpressionLevel, ...] = ("S0", "S1", "S2", "S3")
_STYLE_ORDER: tuple[UserStyle, ...] = (
    "terse_fragment",
    "colloquial",
    "neutral_complete",
    "polite",
    "code_mixed_or_numeric",
    "typo_or_asr_like",
)
_CONSTRAINT_ORDER: tuple[ConstraintLevel, ...] = ("0", "1", "2", "3_plus")
_INTERACTION_RATIOS: dict[InteractionPattern, int] = {
    "direct_request": 70,
    "underspecified_clarification": 15,
    "constraint_correction": 5,
    "no_result_relaxation": 5,
    "goal_change_or_multi_query": 5,
}
_EXPRESSION_RATIOS: dict[ExpressionLevel, int] = {
    "S0": 15,
    "S1": 45,
    "S2": 30,
    "S3": 10,
}
_STYLE_RATIOS: dict[UserStyle, int] = {
    "terse_fragment": 25,
    "colloquial": 35,
    "neutral_complete": 25,
    "polite": 5,
    "code_mixed_or_numeric": 5,
    "typo_or_asr_like": 5,
}
_CONSTRAINT_RATIOS: dict[ConstraintLevel, int] = {
    "0": 30,
    "1": 40,
    "2": 25,
    "3_plus": 5,
}
_CORE_SPLIT_SIZES: dict[FinalSplit, int] = {
    "dev_mini": 200,
    "opt_pool": 800,
    "val": 200,
    "test_frozen": 300,
}
_CORE_INTERACTION_TARGETS: dict[FinalSplit, dict[InteractionPattern, int]] = {
    "dev_mini": {
        "direct_request": 140,
        "underspecified_clarification": 30,
        "constraint_correction": 10,
        "no_result_relaxation": 10,
        "goal_change_or_multi_query": 10,
    },
    "opt_pool": {
        "direct_request": 560,
        "underspecified_clarification": 120,
        "constraint_correction": 40,
        "no_result_relaxation": 40,
        "goal_change_or_multi_query": 40,
    },
    "val": {
        "direct_request": 140,
        "underspecified_clarification": 30,
        "constraint_correction": 10,
        "no_result_relaxation": 10,
        "goal_change_or_multi_query": 10,
    },
    "test_frozen": {
        "direct_request": 210,
        "underspecified_clarification": 45,
        "constraint_correction": 15,
        "no_result_relaxation": 15,
        "goal_change_or_multi_query": 15,
    },
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must be non-blank")
    return value


def _normalise_shape(value: object) -> str:
    aliases = {
        "one_turn": "single_turn",
        "single": "single_turn",
        "[user]": "single_turn",
        "three": "three_turn",
        "[user,assistant,user]": "three_turn",
        "[user, assistant, user]": "three_turn",
    }
    if isinstance(value, str):
        return aliases.get(value.strip(), value.strip())
    return str(value)


def _normalise_constraint(value: object) -> str:
    aliases = {0: "0", 1: "1", 2: "2", 3: "3_plus", "3": "3_plus"}
    if value in aliases:
        return aliases[value]
    return str(value).strip()


class PlanLikeRow(_StrictModel):
    """The plan projection needed by allocation and internal orchestration.

    ``asset_id`` and binding/provenance fields are deliberately optional here
    so small planner fixtures can be allocated without pretending to be an
    asset catalog.  A real authoring work order requires ``asset_id`` and an
    accompanying :class:`InternalAssetBinding` for every selected item.
    """

    plan_id: str
    batch_id: str = "unbatched"
    position: int = Field(default=1, ge=1, le=25)
    canonical_intent: Intent
    canonical_capability: str
    is_boundary: bool
    boundary_strategy: str | None = None
    asset_id: str | None = None
    image_path: str | None = None
    leakage_group_id: str | None = None
    capability_assignment_id: str | None = None
    capability_assignment_source_sha256: Sha256 | None = None

    @field_validator(
        "plan_id",
        "batch_id",
        "canonical_capability",
        "asset_id",
        "image_path",
        "leakage_group_id",
        "capability_assignment_id",
        "boundary_strategy",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)


class FinalSplitAssignment(_StrictModel):
    plan_id: str
    final_split: FinalSplit

    @field_validator("plan_id")
    @classmethod
    def _validate_plan_id(cls, value: str) -> str:
        return _nonblank(value, "plan_id")


class ReuseAssignment(_StrictModel):
    plan_id: str
    reuse_variant: str
    reuse_reason: str = "unspecified"

    @field_validator("plan_id", "reuse_variant", "reuse_reason")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class PromptRecipe(_StrictModel):
    """A structural authoring recipe, intentionally without user utterances."""

    schema_version: Literal[1] = PROMPT_RECIPE_SCHEMA_VERSION
    prompt_recipe_id: str
    title: str
    trajectory_shape: TrajectoryShape
    allowed_interaction_patterns: tuple[InteractionPattern, ...]
    boundary_mode: Literal["any", "required", "forbidden"] = "any"
    allowed_capabilities: tuple[str, ...] | None = None
    structural_slots: tuple[str, ...]
    required_elements: tuple[str, ...]
    prohibited_elements: tuple[str, ...] = ()

    @field_validator("prompt_recipe_id", "title")
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("trajectory_shape", mode="before")
    @classmethod
    def _validate_shape(cls, value: object) -> str:
        return _normalise_shape(value)

    @field_validator("allowed_interaction_patterns")
    @classmethod
    def _validate_patterns(
        cls, value: tuple[InteractionPattern, ...]
    ) -> tuple[InteractionPattern, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("allowed_interaction_patterns must be non-empty and unique")
        return value

    @field_validator("allowed_capabilities")
    @classmethod
    def _validate_capabilities(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        cleaned = tuple(_nonblank(item, "allowed_capabilities") for item in value)
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ValueError("allowed_capabilities must be non-empty and unique")
        return cleaned

    @field_validator("structural_slots", "required_elements", "prohibited_elements")
    @classmethod
    def _validate_structural_labels(
        cls, value: tuple[str, ...], info
    ) -> tuple[str, ...]:
        cleaned = tuple(_nonblank(item, info.field_name) for item in value)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} must not repeat labels")
        return cleaned

    @model_validator(mode="after")
    def _validate_recipe(self):
        if not self.structural_slots or not self.required_elements:
            raise ValueError("a prompt recipe must name structural slots and requirements")
        return self

    def matches(
        self,
        *,
        canonical_capability: str,
        is_boundary: bool,
        interaction_pattern: InteractionPattern,
        trajectory_shape: TrajectoryShape,
    ) -> bool:
        return (
            self.trajectory_shape == trajectory_shape
            and interaction_pattern in self.allowed_interaction_patterns
        ) and (
            self.boundary_mode == "any"
            or (self.boundary_mode == "required" and is_boundary)
            or (self.boundary_mode == "forbidden" and not is_boundary)
        ) and (
            self.allowed_capabilities is None
            or canonical_capability in self.allowed_capabilities
        )


class PromptRecipeManifest(_StrictModel):
    schema_version: Literal[1] = PROMPT_RECIPE_SCHEMA_VERSION
    recipe_count: int = Field(gt=0)
    recipes_sha256: Sha256


class RealismAssignment(_StrictModel):
    """One text-free realism card bound to one immutable plan ID."""

    plan_id: str
    trajectory_shape: TrajectoryShape
    interaction_pattern: InteractionPattern
    expression_level: ExpressionLevel
    user_style: UserStyle
    constraint_level: ConstraintLevel
    ambiguity_subtype: AmbiguitySubtype
    reuse_variant: str
    prompt_recipe_id: str
    prompt_recipe_sha256: Sha256

    @field_validator("plan_id", "reuse_variant", "prompt_recipe_id")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("trajectory_shape", mode="before")
    @classmethod
    def _validate_shape(cls, value: object) -> str:
        return _normalise_shape(value)

    @field_validator("constraint_level", mode="before")
    @classmethod
    def _validate_constraint(cls, value: object) -> str:
        return _normalise_constraint(value)

    @model_validator(mode="after")
    def _validate_combinations(self):
        if self.interaction_pattern == "direct_request":
            if self.trajectory_shape != "single_turn":
                raise ValueError("direct_request must use single_turn")
        elif self.trajectory_shape != "three_turn":
            raise ValueError("non-direct interaction patterns must use three_turn")
        if self.ambiguity_subtype == "clarification_required":
            if (
                self.interaction_pattern != "underspecified_clarification"
                or self.trajectory_shape != "three_turn"
            ):
                raise ValueError(
                    "clarification_required must map to underspecified three_turn"
                )
        return self


class RealismAssignmentsManifest(_StrictModel):
    """Canonical binding for a complete realism sidecar."""

    schema_version: Literal[1] = 1
    allocation_policy_version: Literal[
        "r2-split-capability-boundary-lrm-v1"
    ] = REALISM_ALLOCATION_POLICY_VERSION
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed: int = Field(ge=0)
    row_count: int = Field(gt=0)
    assignments_sha256: Sha256
    prompt_recipe_manifest_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    reuse_sidecar_sha256: Sha256


@dataclass(frozen=True)
class RealismSidecar:
    assignments: tuple[RealismAssignment, ...]
    manifest: RealismAssignmentsManifest
    recipes: tuple[PromptRecipe, ...]
    recipe_manifest: PromptRecipeManifest


class RealismQuotaSpec(_StrictModel):
    """Exact row-count targets after final split allocation."""

    split_sizes: dict[FinalSplit, int]
    interaction_targets: dict[FinalSplit, dict[InteractionPattern, int]]
    expression_targets: dict[FinalSplit, dict[ExpressionLevel, int]]
    user_style_targets: dict[FinalSplit, dict[UserStyle, int]]
    constraint_targets: dict[FinalSplit, dict[ConstraintLevel, int]]
    ambiguity_targets: dict[AmbiguitySubtype, int]

    @model_validator(mode="after")
    def _validate_target_shapes(self):
        if set(self.split_sizes) != set(_FINAL_SPLIT_ORDER):
            raise ValueError("split_sizes must cover every final split")
        for split, size in self.split_sizes.items():
            if size < 0:
                raise ValueError(f"split size must be non-negative: {split}")
            for label, matrix, expected_keys in (
                ("interaction", self.interaction_targets, set(_INTERACTION_ORDER)),
                ("expression", self.expression_targets, set(_EXPRESSION_ORDER)),
                ("user_style", self.user_style_targets, set(_STYLE_ORDER)),
                ("constraint", self.constraint_targets, set(_CONSTRAINT_ORDER)),
            ):
                values = matrix.get(split)
                if values is None or set(values) != expected_keys:
                    raise ValueError(f"{label} targets must cover {split} exactly")
                if any(value < 0 for value in values.values()) or sum(values.values()) != size:
                    raise ValueError(f"{label} targets do not sum to {split} size")
        if set(self.ambiguity_targets) != {
            "resolved_near_boundary",
            "clarification_required",
        }:
            raise ValueError("ambiguity targets must name both boundary subtypes")
        if any(value < 0 for value in self.ambiguity_targets.values()):
            raise ValueError("ambiguity targets must be non-negative")
        return self


class InternalAssetBinding(_StrictModel):
    """Internal-only catalog evidence for an asset alias.

    ``canonical_path`` and the optional source identifiers never appear in an
    :class:`AuthorPacket`; their retention here makes later materialisation
    verifiable without changing model-visible input.
    """

    asset_id: str
    asset_sha256: Sha256
    byte_size: int = Field(ge=0)
    canonical_path: str
    source_dataset: str | None = None
    source_record_id: str | None = None

    @field_validator(
        "asset_id", "canonical_path", "source_dataset", "source_record_id"
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)


class OpaqueAssetBinding(_StrictModel):
    """Internal alias map for later hard-link/verified-copy materialisation."""

    opaque_alias: str
    materialization_relative_path: str
    asset_id: str
    asset_sha256: Sha256
    byte_size: int = Field(ge=0)
    canonical_path: str
    source_dataset: str | None = None
    source_record_id: str | None = None

    @field_validator("opaque_alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        value = _nonblank(value, "opaque_alias")
        if not re.fullmatch(r"a[0-9]{4}\.[a-z0-9]{1,8}", value):
            raise ValueError("opaque_alias must be an opaque aNNNN.ext name")
        return value

    @field_validator("materialization_relative_path")
    @classmethod
    def _validate_materialization_path(cls, value: str) -> str:
        value = _nonblank(value, "materialization_relative_path").replace("\\", "/")
        if (
            value.startswith("/")
            or re.match(r"^[A-Za-z]:/", value)
            or ".." in value.split("/")
            or not value.startswith("author-assets/")
        ):
            raise ValueError("materialization path must be a safe author-assets relative path")
        return value

    @field_validator(
        "asset_id", "canonical_path", "source_dataset", "source_record_id"
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_alias_path_pair(self):
        if self.materialization_relative_path != f"author-assets/{self.opaque_alias}":
            raise ValueError("materialization path must match opaque alias")
        return self


class InternalWorkOrderItem(_StrictModel):
    """An internal item; it may retain provenance but is never author-visible."""

    plan_id: str
    batch_id: str
    position: int = Field(ge=1, le=25)
    asset_id: str
    image_path: str | None = None
    leakage_group_id: str | None = None
    capability_assignment_id: str | None = None
    capability_assignment_source_sha256: Sha256 | None = None
    canonical_intent: Intent
    canonical_capability: str
    is_boundary: bool
    boundary_strategy: str | None = None
    final_split: FinalSplit
    reuse_variant: str
    reuse_reason: str
    realism: RealismAssignment
    opaque_alias: str

    @field_validator(
        "plan_id",
        "batch_id",
        "asset_id",
        "image_path",
        "leakage_group_id",
        "capability_assignment_id",
        "canonical_capability",
        "boundary_strategy",
        "reuse_variant",
        "reuse_reason",
        "opaque_alias",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_binding(self):
        if self.realism.plan_id != self.plan_id:
            raise ValueError("realism card must bind the same plan_id")
        if self.realism.reuse_variant != self.reuse_variant:
            raise ValueError("realism card reuse_variant must match work order")
        return self


class InternalWorkOrder(_StrictModel):
    """Schema-v2 internal, deterministic author handoff for one 25-item job."""

    schema_version: Literal[2] = AUTHORING_SCHEMA_VERSION
    job_id: str
    base_batch_id: str
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest: RealismAssignmentsManifest
    recipe_manifest: PromptRecipeManifest
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    prompt_recipe_manifest_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    reuse_sidecar_sha256: Sha256
    generation_input_sha256: Sha256
    author_packet_sha256: Sha256
    prompt_recipes: tuple[PromptRecipe, ...]
    items: tuple[InternalWorkOrderItem, ...]
    aliases: tuple[OpaqueAssetBinding, ...]

    @field_validator("job_id", "base_batch_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_work_order(self):
        if self.realism_manifest.plan_sha256 != self.plan_sha256:
            raise ValueError("work order realism manifest plan digest mismatch")
        if self.realism_manifest.asset_catalog_sha256 != self.asset_catalog_sha256:
            raise ValueError("work order realism manifest catalog digest mismatch")
        if (
            self.realism_manifest.capability_assignments_sha256
            != self.capability_assignments_sha256
        ):
            raise ValueError("work order realism manifest assignment digest mismatch")
        if self.realism_manifest.assignments_sha256 != self.realism_assignments_sha256:
            raise ValueError("work order realism assignment digest mismatch")
        if (
            self.realism_manifest.prompt_recipe_manifest_sha256
            != self.prompt_recipe_manifest_sha256
        ):
            raise ValueError("work order realism recipe digest mismatch")
        if (
            self.realism_manifest.final_split_sidecar_sha256
            != self.final_split_sidecar_sha256
            or self.realism_manifest.reuse_sidecar_sha256 != self.reuse_sidecar_sha256
        ):
            raise ValueError("work order realism sidecar digest mismatch")
        if self.realism_manifest_sha256 != sha256_bytes(
            canonical_json_bytes(self.realism_manifest)
        ):
            raise ValueError("work order realism manifest digest mismatch")
        if self.recipe_manifest != build_prompt_recipe_manifest(self.prompt_recipes):
            raise ValueError("work order prompt recipe manifest mismatch")
        if self.prompt_recipe_manifest_sha256 != sha256_bytes(
            canonical_json_bytes(self.recipe_manifest)
        ):
            raise ValueError("work order prompt recipe manifest digest mismatch")
        if len(self.items) != 25:
            raise ValueError("r2 authoring work order must contain exactly 25 items")
        if [item.position for item in self.items] != list(range(1, 26)):
            raise ValueError("work-order positions must cover 1..25 in order")
        if any(item.batch_id != self.base_batch_id for item in self.items):
            raise ValueError("work-order item batch_id mismatch")
        if len({item.plan_id for item in self.items}) != 25:
            raise ValueError("work-order plan IDs must be unique")
        alias_by_asset = {binding.asset_id: binding.opaque_alias for binding in self.aliases}
        if len(alias_by_asset) != len(self.aliases) or len(
            {binding.opaque_alias for binding in self.aliases}
        ) != len(self.aliases):
            raise ValueError("aliases must be one-to-one with selected assets")
        for item in self.items:
            if alias_by_asset.get(item.asset_id) != item.opaque_alias:
                raise ValueError("work-order item alias does not match asset binding")
        recipes = _recipe_index(self.prompt_recipes)
        for item in self.items:
            recipe = recipes.get(item.realism.prompt_recipe_id)
            if recipe is None or item.realism.prompt_recipe_sha256 != prompt_recipe_sha256(
                recipe
            ):
                raise ValueError("work-order realism card recipe digest mismatch")
            if not recipe.matches(
                canonical_capability=item.canonical_capability,
                is_boundary=item.is_boundary,
                interaction_pattern=item.realism.interaction_pattern,
                trajectory_shape=item.realism.trajectory_shape,
            ):
                raise ValueError("work-order realism card recipe is incompatible")
        return self


class AuthorPacketBoundary(_StrictModel):
    is_boundary: bool
    boundary_strategy: str | None = None

    @field_validator("boundary_strategy")
    @classmethod
    def _validate_strategy(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "boundary_strategy")

    @model_validator(mode="after")
    def _validate_boundary(self):
        if not self.is_boundary and self.boundary_strategy is not None:
            raise ValueError("non-boundary packet item must not include a strategy")
        return self


class AuthorPacketRealismCard(_StrictModel):
    trajectory_shape: TrajectoryShape
    interaction_pattern: InteractionPattern
    expression_level: ExpressionLevel
    user_style: UserStyle
    constraint_level: ConstraintLevel
    ambiguity_subtype: AmbiguitySubtype

    @field_validator("trajectory_shape", mode="before")
    @classmethod
    def _validate_shape(cls, value: object) -> str:
        return _normalise_shape(value)

    @field_validator("constraint_level", mode="before")
    @classmethod
    def _validate_constraint(cls, value: object) -> str:
        return _normalise_constraint(value)

class AuthorPacketRecipe(_StrictModel):
    prompt_recipe_id: str
    prompt_recipe_sha256: Sha256
    trajectory_shape: TrajectoryShape
    structural_slots: tuple[str, ...]
    required_elements: tuple[str, ...]
    prohibited_elements: tuple[str, ...]

    @field_validator("prompt_recipe_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _nonblank(value, "prompt_recipe_id")

    @field_validator("trajectory_shape", mode="before")
    @classmethod
    def _validate_shape(cls, value: object) -> str:
        return _normalise_shape(value)


class AuthorPacketItem(_StrictModel):
    plan_id: str
    canonical_intent: Intent
    canonical_capability: str
    boundary: AuthorPacketBoundary
    realism: AuthorPacketRealismCard
    recipe: AuthorPacketRecipe
    opaque_alias: str

    @field_validator("plan_id", "canonical_capability", "opaque_alias")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class AuthorPacket(_StrictModel):
    """The only projection intended for a future model-visible author step."""

    schema_version: Literal[1] = 1
    job_id: str
    base_batch_id: str
    generation_input_sha256: Sha256
    items: tuple[AuthorPacketItem, ...]

    @field_validator("job_id", "base_batch_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_items(self):
        if len(self.items) != 25 or len({item.plan_id for item in self.items}) != 25:
            raise ValueError("author packet must contain exactly 25 unique plan IDs")
        return self


class PacketLeakFinding(_StrictModel):
    json_pointer: str
    rule: str

    @field_validator("json_pointer", "rule")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class AuthoringCheckpoint(_StrictModel):
    """Pure, monotonic checkpoint for the immutable schema-v2 work order."""

    schema_version: Literal[1] = 1
    job_id: str
    work_order_sha256: Sha256
    generation_input_sha256: Sha256
    state: Literal["issued", "staged", "accepted"]
    staged_batch_id: str | None = None
    staged_results_sha256: Sha256 | None = None
    accepted_batch_id: str | None = None

    @field_validator("job_id", "staged_batch_id", "accepted_batch_id")
    @classmethod
    def _validate_text(cls, value: str | None, info):
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_state(self):
        if self.state == "issued":
            if any(
                value is not None
                for value in (
                    self.staged_batch_id,
                    self.staged_results_sha256,
                    self.accepted_batch_id,
                )
            ):
                raise ValueError("issued checkpoint cannot name output")
        elif self.state == "staged":
            if (
                self.staged_batch_id is None
                or self.staged_results_sha256 is None
                or self.accepted_batch_id is not None
            ):
                raise ValueError("staged checkpoint is incomplete")
        elif (
            self.staged_batch_id is None
            or self.staged_results_sha256 is None
            or self.accepted_batch_id is None
        ):
            raise ValueError("accepted checkpoint is incomplete")
        return self


@dataclass(frozen=True)
class AuthoringJob:
    work_order: InternalWorkOrder
    author_packet: AuthorPacket
    checkpoint: AuthoringCheckpoint


def _stable_rank(seed: int, namespace: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\x00{namespace}\x00{value}".encode("utf-8")).hexdigest()
    return digest, value


def _canonical_mapping_bytes(values: Mapping[str, object]) -> bytes:
    return canonical_jsonl_bytes(
        {"plan_id": plan_id, "value": values[plan_id]}
        for plan_id in sorted(values)
    )


def _coerce_payload(value: object) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"unsupported plan-like value: {type(value)!r}")


def coerce_plan_rows(rows_or_plan: object) -> tuple[PlanLikeRow, ...]:
    """Accept ``CorpusPlan.queries``-like records without importing planner code."""

    source: object = getattr(rows_or_plan, "queries", rows_or_plan)
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        raise TypeError("plan rows must be a sequence or an object with queries")
    fields = set(PlanLikeRow.model_fields)
    rows: list[PlanLikeRow] = []
    for value in source:
        payload = _coerce_payload(value)
        selected = {name: payload[name] for name in fields if name in payload}
        rows.append(PlanLikeRow.model_validate(selected))
    plan_ids = [row.plan_id for row in rows]
    if not rows or len(plan_ids) != len(set(plan_ids)):
        raise ValueError("plan rows must be non-empty with globally unique plan_id values")
    return tuple(rows)


def _normalize_final_splits(
    rows: Sequence[PlanLikeRow],
    sidecar: Mapping[str, object] | Sequence[object],
) -> dict[str, FinalSplit]:
    values: dict[str, FinalSplit] = {}
    if isinstance(sidecar, Mapping):
        iterator = sidecar.items()
        for plan_id, raw in iterator:
            if isinstance(raw, FinalSplitAssignment):
                assignment = raw
            elif isinstance(raw, Mapping):
                assignment = FinalSplitAssignment.model_validate(raw)
                if assignment.plan_id != plan_id:
                    raise ValueError("final split mapping key does not match plan_id")
            else:
                assignment = FinalSplitAssignment(plan_id=plan_id, final_split=raw)
            values[assignment.plan_id] = assignment.final_split
    else:
        for raw in sidecar:
            assignment = (
                raw
                if isinstance(raw, FinalSplitAssignment)
                else FinalSplitAssignment.model_validate(_coerce_payload(raw))
            )
            if assignment.plan_id in values:
                raise ValueError("final split sidecar has duplicate plan_id")
            values[assignment.plan_id] = assignment.final_split
    _require_exact_plan_id_coverage(rows, values, "final split sidecar")
    return values


def _normalize_reuse_assignments(
    rows: Sequence[PlanLikeRow],
    sidecar: Mapping[str, object] | Sequence[object],
) -> dict[str, ReuseAssignment]:
    values: dict[str, ReuseAssignment] = {}
    if isinstance(sidecar, Mapping):
        for plan_id, raw in sidecar.items():
            if isinstance(raw, ReuseAssignment):
                assignment = raw
            elif isinstance(raw, Mapping):
                payload = dict(raw)
                payload.setdefault("plan_id", plan_id)
                assignment = ReuseAssignment.model_validate(payload)
            else:
                assignment = ReuseAssignment(plan_id=plan_id, reuse_variant=raw)
            if assignment.plan_id != plan_id:
                raise ValueError("reuse mapping key does not match plan_id")
            values[assignment.plan_id] = assignment
    else:
        for raw in sidecar:
            assignment = (
                raw
                if isinstance(raw, ReuseAssignment)
                else ReuseAssignment.model_validate(_coerce_payload(raw))
            )
            if assignment.plan_id in values:
                raise ValueError("reuse sidecar has duplicate plan_id")
            values[assignment.plan_id] = assignment
    _require_exact_plan_id_coverage(rows, values, "reuse sidecar")
    return values


def _require_exact_plan_id_coverage(
    rows: Sequence[PlanLikeRow], values: Mapping[str, object], label: str
) -> None:
    expected = {row.plan_id for row in rows}
    actual = set(values)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} must exactly cover plan IDs; missing={missing[:3]}, extra={extra[:3]}")


def canonical_final_split_sidecar_bytes(
    final_split_by_plan_id: Mapping[str, FinalSplit],
) -> bytes:
    return canonical_jsonl_bytes(
        FinalSplitAssignment(plan_id=plan_id, final_split=final_split_by_plan_id[plan_id])
        for plan_id in sorted(final_split_by_plan_id)
    )


def canonical_reuse_sidecar_bytes(
    reuse_by_plan_id: Mapping[str, ReuseAssignment],
) -> bytes:
    return canonical_jsonl_bytes(reuse_by_plan_id[plan_id] for plan_id in sorted(reuse_by_plan_id))


def canonical_prompt_recipe_bytes(recipes: Iterable[PromptRecipe]) -> bytes:
    materialized = tuple(recipes)
    _recipe_index(materialized)
    return canonical_jsonl_bytes(
        recipe for recipe in sorted(materialized, key=lambda item: item.prompt_recipe_id)
    )


def prompt_recipe_sha256(recipe: PromptRecipe) -> str:
    return sha256_bytes(canonical_json_bytes(recipe))


def build_prompt_recipe_manifest(recipes: Iterable[PromptRecipe]) -> PromptRecipeManifest:
    materialized = tuple(recipes)
    return PromptRecipeManifest(
        recipe_count=len(materialized),
        recipes_sha256=sha256_bytes(canonical_prompt_recipe_bytes(materialized)),
    )


def default_prompt_recipes() -> tuple[PromptRecipe, ...]:
    """Return the frozen structured recipe library (never example utterances)."""

    direct = ("direct_request",)
    capability_specs = (
        ("exact_lookup", "exact_lookup_structure", "product.exact_match"),
        ("multi_product_search", "multi_product_search_structure", "product.multi_search"),
        ("style_discovery", "style_discovery_structure", "product.style_recommendation"),
        ("visual_encyclopedia", "visual_encyclopedia_structure", "knowledge.visual_encyclopedia"),
        ("document_reading", "document_reading_structure", "utility.document_reading"),
        ("recipe_guidance", "recipe_guidance_structure", "utility.recipe_guidance"),
    )
    recipes: list[PromptRecipe] = [
        PromptRecipe(
            prompt_recipe_id=recipe_id,
            title=title,
            trajectory_shape="single_turn",
            allowed_interaction_patterns=direct,
            boundary_mode="forbidden",
            allowed_capabilities=(capability,),
            structural_slots=("intent_goal", "visual_anchor", "constraint_bundle"),
            required_elements=("canonical_goal", "visual_reference"),
            prohibited_elements=("invented_visual_fact", "dataset_metadata"),
        )
        for recipe_id, title, capability in capability_specs
    ]
    recipes.extend(
        (
            PromptRecipe(
                prompt_recipe_id="boundary_resolution_request",
                title="boundary_resolution_structure",
                trajectory_shape="single_turn",
                allowed_interaction_patterns=direct,
                boundary_mode="required",
                structural_slots=("intent_goal", "ambiguity_cue", "output_scope"),
                required_elements=("canonical_goal", "bounded_interpretation"),
                prohibited_elements=("alternate_label_claim", "dataset_metadata"),
            ),
            PromptRecipe(
                prompt_recipe_id="clarification_dialogue",
                title="underspecified_clarification_structure",
                trajectory_shape="three_turn",
                allowed_interaction_patterns=("underspecified_clarification",),
                boundary_mode="any",
                structural_slots=("initial_goal", "clarifying_question", "resolved_goal"),
                required_elements=("goal_narrowing", "final_user_goal"),
                prohibited_elements=("assistant_final_answer", "dataset_metadata"),
            ),
            PromptRecipe(
                prompt_recipe_id="boundary_clarification_dialogue",
                title="boundary_clarification_structure",
                trajectory_shape="three_turn",
                allowed_interaction_patterns=("underspecified_clarification",),
                boundary_mode="required",
                structural_slots=("ambiguous_goal", "clarifying_question", "resolved_goal"),
                required_elements=("ambiguity_resolution", "final_user_goal"),
                prohibited_elements=("alternate_label_claim", "dataset_metadata"),
            ),
            PromptRecipe(
                prompt_recipe_id="constraint_correction_dialogue",
                title="constraint_correction_structure",
                trajectory_shape="three_turn",
                allowed_interaction_patterns=("constraint_correction",),
                boundary_mode="any",
                structural_slots=("initial_goal", "correction_prompt", "corrected_constraint"),
                required_elements=("changed_constraint", "final_user_goal"),
                prohibited_elements=("assistant_final_answer", "dataset_metadata"),
            ),
            PromptRecipe(
                prompt_recipe_id="no_result_relaxation_dialogue",
                title="no_result_relaxation_structure",
                trajectory_shape="three_turn",
                allowed_interaction_patterns=("no_result_relaxation",),
                boundary_mode="any",
                structural_slots=("initial_goal", "availability_signal", "relaxed_constraint"),
                required_elements=("relaxation", "final_user_goal"),
                prohibited_elements=("assistant_final_answer", "dataset_metadata"),
            ),
            PromptRecipe(
                prompt_recipe_id="goal_change_dialogue",
                title="goal_change_or_multi_query_structure",
                trajectory_shape="three_turn",
                allowed_interaction_patterns=("goal_change_or_multi_query",),
                boundary_mode="any",
                structural_slots=("initial_goal", "followup_prompt", "revised_or_second_goal"),
                required_elements=("goal_transition", "final_user_goal"),
                prohibited_elements=("assistant_final_answer", "dataset_metadata"),
            ),
        )
    )
    library = tuple(recipes)
    if len(library) != 12:  # pragma: no cover - protects the frozen recipe library
        raise AssertionError("r2 recipe library must contain exactly twelve recipes")
    _recipe_index(library)
    return library


def _recipe_index(recipes: Iterable[PromptRecipe]) -> dict[str, PromptRecipe]:
    index: dict[str, PromptRecipe] = {}
    for recipe in recipes:
        if recipe.prompt_recipe_id in index:
            raise ValueError("prompt recipe IDs must be unique")
        index[recipe.prompt_recipe_id] = recipe
    if not index:
        raise ValueError("at least one prompt recipe is required")
    return index


def _largest_remainder(
    weights: Mapping[str, int], total: int, *, seed: int, namespace: str
) -> dict[str, int]:
    if total < 0 or any(value < 0 for value in weights.values()):
        raise ValueError("largest-remainder inputs must be non-negative")
    denominator = sum(weights.values())
    if total > denominator:
        raise ValueError("largest-remainder target exceeds capacity")
    if total == 0:
        return {key: 0 for key in weights}
    if denominator == 0:
        raise ValueError("cannot allocate a positive target to zero capacity")
    result = {key: (weight * total) // denominator for key, weight in weights.items()}
    remainder = total - sum(result.values())
    ranking = sorted(
        weights,
        key=lambda key: (
            -((weights[key] * total) % denominator),
            _stable_rank(seed, namespace, key),
        ),
    )
    for key in ranking[:remainder]:
        result[key] += 1
    if any(result[key] > weights[key] for key in weights):  # pragma: no cover
        raise AssertionError("largest remainder overflowed a capacity")
    return result


def _ratio_targets(
    size: int,
    ratios: Mapping[str, int],
    *,
    seed: int,
    namespace: str,
) -> dict[str, int]:
    if sum(ratios.values()) != 100:
        raise ValueError("ratios must sum to 100")
    expanded = {key: size * ratio for key, ratio in ratios.items()}
    base = {key: value // 100 for key, value in expanded.items()}
    residual = size - sum(base.values())
    ranking = sorted(
        ratios,
        key=lambda key: (-(expanded[key] % 100), _stable_rank(seed, namespace, key)),
    )
    for key in ranking[:residual]:
        base[key] += 1
    return base


def core_r2_realism_quota_spec() -> RealismQuotaSpec:
    """The exact 1,500-row r2 behavior distribution frozen in the plan."""

    def from_ratios(ratios: Mapping[str, int], label: str) -> dict[FinalSplit, dict]:
        return {
            split: _ratio_targets(
                size, ratios, seed=0, namespace=f"core-r2-{label}-{split}"
            )
            for split, size in _CORE_SPLIT_SIZES.items()
        }

    return RealismQuotaSpec(
        split_sizes=_CORE_SPLIT_SIZES,
        interaction_targets=_CORE_INTERACTION_TARGETS,
        expression_targets=from_ratios(_EXPRESSION_RATIOS, "expression"),
        user_style_targets=from_ratios(_STYLE_RATIOS, "user-style"),
        constraint_targets=from_ratios(_CONSTRAINT_RATIOS, "constraint"),
        ambiguity_targets={
            "resolved_near_boundary": 158,
            "clarification_required": 105,
        },
    )


def scaled_realism_quota_spec(
    split_sizes: Mapping[FinalSplit, int], *, boundary_count: int, seed: int
) -> RealismQuotaSpec:
    """Fixture-friendly proportional quota spec retaining the r2 legal rules."""

    if set(split_sizes) != set(_FINAL_SPLIT_ORDER):
        raise ValueError("scaled quota spec requires all final splits")
    interaction_targets = {
        split: _ratio_targets(size, _INTERACTION_RATIOS, seed=seed, namespace=f"interaction-{split}")
        for split, size in split_sizes.items()
    }
    expression_targets = {
        split: _ratio_targets(size, _EXPRESSION_RATIOS, seed=seed, namespace=f"expression-{split}")
        for split, size in split_sizes.items()
    }
    style_targets = {
        split: _ratio_targets(size, _STYLE_RATIOS, seed=seed, namespace=f"style-{split}")
        for split, size in split_sizes.items()
    }
    constraint_targets = {
        split: _ratio_targets(size, _CONSTRAINT_RATIOS, seed=seed, namespace=f"constraint-{split}")
        for split, size in split_sizes.items()
    }
    denominator = 263
    raw = {
        "resolved_near_boundary": (158 * boundary_count) // denominator,
        "clarification_required": (105 * boundary_count) // denominator,
    }
    residual = boundary_count - sum(raw.values())
    ranking = sorted(
        raw,
        key=lambda key: (
            -((({"resolved_near_boundary": 158, "clarification_required": 105}[key] * boundary_count) % denominator)),
            _stable_rank(seed, "scaled-ambiguity", key),
        ),
    )
    for key in ranking[:residual]:
        raw[key] += 1
    underspecified_capacity = sum(
        values["underspecified_clarification"]
        for values in interaction_targets.values()
    )
    if raw["clarification_required"] > underspecified_capacity:
        raw["clarification_required"] = underspecified_capacity
        raw["resolved_near_boundary"] = boundary_count - underspecified_capacity
    ambiguity = raw
    return RealismQuotaSpec(
        split_sizes=dict(split_sizes),
        interaction_targets=interaction_targets,
        expression_targets=expression_targets,
        user_style_targets=style_targets,
        constraint_targets=constraint_targets,
        ambiguity_targets=ambiguity,
    )


def _resolve_quota_spec(
    rows: Sequence[PlanLikeRow],
    final_splits: Mapping[str, FinalSplit],
    *,
    seed: int,
    quota_spec: RealismQuotaSpec | None,
) -> RealismQuotaSpec:
    counts: Counter[FinalSplit] = Counter(final_splits.values())
    split_sizes = {split: counts[split] for split in _FINAL_SPLIT_ORDER}
    boundary_count = sum(row.is_boundary for row in rows)
    if quota_spec is not None:
        if quota_spec.split_sizes != split_sizes:
            raise ValueError("provided realism quota spec does not match final split sidecar")
        if sum(quota_spec.ambiguity_targets.values()) != boundary_count:
            raise ValueError("provided ambiguity quotas do not equal boundary row count")
        return quota_spec
    if split_sizes == _CORE_SPLIT_SIZES and len(rows) == 1500:
        capability_counts: dict[FinalSplit, Counter[str]] = {
            split: Counter() for split in _FINAL_SPLIT_ORDER
        }
        boundary_counts: dict[FinalSplit, Counter[str]] = {
            split: Counter() for split in _FINAL_SPLIT_ORDER
        }
        for row in rows:
            split = final_splits[row.plan_id]
            capability_counts[split][row.canonical_capability] += 1
            if row.is_boundary:
                boundary_counts[split][row.canonical_capability] += 1
        if any(
            dict(capability_counts[split]) != CORE_R2_CAPABILITY_COUNTS[split]
            for split in _FINAL_SPLIT_ORDER
        ):
            raise ValueError("core r2 capability x split matrix drifted")
        if any(
            dict(boundary_counts[split])
            != CORE_R2_CAPABILITY_BOUNDARY_COUNTS[split]
            for split in _FINAL_SPLIT_ORDER
        ):
            raise ValueError("core r2 boundary x capability x split matrix drifted")
        spec = core_r2_realism_quota_spec()
        if boundary_count != sum(spec.ambiguity_targets.values()):
            raise ValueError("core r2 plan must contain exactly 263 boundary rows")
        return spec
    return scaled_realism_quota_spec(split_sizes, boundary_count=boundary_count, seed=seed)


def _stratum(row: PlanLikeRow) -> str:
    return f"{row.canonical_capability}|{'boundary' if row.is_boundary else 'non_boundary'}"


def _select_stratified(
    rows: Sequence[PlanLikeRow],
    target: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[PlanLikeRow, ...]:
    if target < 0 or target > len(rows):
        raise ValueError(f"selection target is outside capacity for {namespace}")
    groups: dict[str, list[PlanLikeRow]] = defaultdict(list)
    for row in rows:
        groups[_stratum(row)].append(row)
    quota = _largest_remainder(
        {key: len(group) for key, group in groups.items()},
        target,
        seed=seed,
        namespace=f"{namespace}:quota",
    )
    selected: list[PlanLikeRow] = []
    for key in sorted(groups):
        selected.extend(
            sorted(
                groups[key],
                key=lambda row: _stable_rank(seed, f"{namespace}:{key}", row.plan_id),
            )[: quota[key]]
        )
    if len(selected) != target:  # pragma: no cover - arithmetic guard
        raise AssertionError("stratified selection count drifted")
    return tuple(selected)


def _set_selection(
    target: dict[str, Any], rows: Iterable[PlanLikeRow], field: str, value: object
) -> None:
    for row in rows:
        if field in target[row.plan_id]:
            raise ValueError(f"{field} was assigned twice for {row.plan_id}")
        target[row.plan_id][field] = value


def _remaining(
    rows: Sequence[PlanLikeRow], target: Mapping[str, Mapping[str, Any]], field: str
) -> list[PlanLikeRow]:
    return [row for row in rows if field not in target[row.plan_id]]


def _assign_interactions(
    rows: Sequence[PlanLikeRow],
    final_splits: Mapping[str, FinalSplit],
    spec: RealismQuotaSpec,
    target: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    boundaries = [row for row in rows if row.is_boundary]
    clarification_count = spec.ambiguity_targets["clarification_required"]
    boundaries_by_split = {
        split: [row for row in boundaries if final_splits[row.plan_id] == split]
        for split in _FINAL_SPLIT_ORDER
    }
    split_targets = _largest_remainder(
        {split: len(split_rows) for split, split_rows in boundaries_by_split.items()},
        clarification_count,
        seed=seed,
        namespace="ambiguity-clarification-split",
    )
    if any(
        split_targets[split]
        > spec.interaction_targets[split]["underspecified_clarification"]
        for split in _FINAL_SPLIT_ORDER
    ):
        split_targets = _largest_remainder(
            {
                split: min(
                    len(split_rows),
                    spec.interaction_targets[split]["underspecified_clarification"],
                )
                for split, split_rows in boundaries_by_split.items()
            },
            clarification_count,
            seed=seed,
            namespace="ambiguity-clarification-capped-split",
        )
    clarification_rows = tuple(
        row
        for split in _FINAL_SPLIT_ORDER
        for row in _select_stratified(
            boundaries_by_split[split],
            split_targets[split],
            seed=seed,
            namespace=f"ambiguity-clarification-{split}",
        )
    )
    clarification_ids = {row.plan_id for row in clarification_rows}
    for row in boundaries:
        target[row.plan_id]["ambiguity_subtype"] = (
            "clarification_required"
            if row.plan_id in clarification_ids
            else "resolved_near_boundary"
        )
    for row in rows:
        if not row.is_boundary:
            target[row.plan_id]["ambiguity_subtype"] = "none"
    _set_selection(
        target,
        clarification_rows,
        "interaction_pattern",
        "underspecified_clarification",
    )

    for split in _FINAL_SPLIT_ORDER:
        split_rows = [row for row in rows if final_splits[row.plan_id] == split]
        quotas = spec.interaction_targets[split]
        preassigned = sum(
            target[row.plan_id].get("interaction_pattern") == "underspecified_clarification"
            for row in split_rows
        )
        if preassigned > quotas["underspecified_clarification"]:
            raise ValueError("clarification allocation exceeds its split interaction quota")
        for pattern in (
            "no_result_relaxation",
            "constraint_correction",
            "goal_change_or_multi_query",
            "underspecified_clarification",
            "direct_request",
        ):
            required = quotas[pattern]
            if pattern == "underspecified_clarification":
                required -= preassigned
            candidates = _remaining(split_rows, target, "interaction_pattern")
            if pattern == "direct_request":
                if required != len(candidates):
                    raise ValueError("interaction quota leaves an invalid direct-request remainder")
                selected = tuple(candidates)
            else:
                selected = _select_stratified(
                    candidates,
                    required,
                    seed=seed,
                    namespace=f"interaction-{split}-{pattern}",
                )
            _set_selection(target, selected, "interaction_pattern", pattern)


def _assign_expression_levels(
    rows: Sequence[PlanLikeRow],
    final_splits: Mapping[str, FinalSplit],
    spec: RealismQuotaSpec,
    target: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    for split in _FINAL_SPLIT_ORDER:
        split_rows = [row for row in rows if final_splits[row.plan_id] == split]
        quotas = spec.expression_targets[split]
        clarifications = [
            row
            for row in split_rows
            if target[row.plan_id]["ambiguity_subtype"] == "clarification_required"
        ]
        special_targets = _largest_remainder(
            {"S2": quotas["S2"], "S3": quotas["S3"]},
            len(clarifications),
            seed=seed,
            namespace=f"clarification-expression-{split}",
        )
        s3 = _select_stratified(
            clarifications,
            special_targets["S3"],
            seed=seed,
            namespace=f"clarification-expression-s3-{split}",
        )
        s3_ids = {row.plan_id for row in s3}
        _set_selection(target, s3, "expression_level", "S3")
        _set_selection(
            target,
            [row for row in clarifications if row.plan_id not in s3_ids],
            "expression_level",
            "S2",
        )
        for level, predicate in (
            ("S0", lambda row: target[row.plan_id]["interaction_pattern"] == "direct_request"),
            ("S1", lambda row: True),
            ("S3", lambda row: True),
            ("S2", lambda row: True),
        ):
            already = sum(
                target[row.plan_id].get("expression_level") == level
                for row in split_rows
            )
            candidates = [
                row
                for row in _remaining(split_rows, target, "expression_level")
                if predicate(row)
            ]
            required = quotas[level] - already
            if level == "S2":
                all_remaining = _remaining(split_rows, target, "expression_level")
                if required != len(all_remaining):
                    raise ValueError("expression quotas leave an invalid S2 remainder")
                selected = tuple(all_remaining)
            else:
                selected = _select_stratified(
                    candidates,
                    required,
                    seed=seed,
                    namespace=f"expression-{split}-{level}",
                )
            _set_selection(target, selected, "expression_level", level)


def _assign_constraint_levels(
    rows: Sequence[PlanLikeRow],
    final_splits: Mapping[str, FinalSplit],
    spec: RealismQuotaSpec,
    target: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    predicates = (
        ("3_plus", lambda expression: expression == "S3"),
        ("2", lambda expression: expression in {"S2", "S3"}),
        ("1", lambda expression: expression != "S0"),
        ("0", lambda expression: True),
    )
    for split in _FINAL_SPLIT_ORDER:
        split_rows = [row for row in rows if final_splits[row.plan_id] == split]
        quotas = spec.constraint_targets[split]
        for level, predicate in predicates:
            candidates = [
                row
                for row in _remaining(split_rows, target, "constraint_level")
                if predicate(target[row.plan_id]["expression_level"])
            ]
            required = quotas[level]
            if level == "0":
                all_remaining = _remaining(split_rows, target, "constraint_level")
                if required != len(all_remaining):
                    raise ValueError("constraint quotas leave an invalid zero remainder")
                selected = tuple(all_remaining)
            else:
                selected = _select_stratified(
                    candidates,
                    required,
                    seed=seed,
                    namespace=f"constraint-{split}-{level}",
                )
            _set_selection(target, selected, "constraint_level", level)


def _assign_user_styles(
    rows: Sequence[PlanLikeRow],
    final_splits: Mapping[str, FinalSplit],
    spec: RealismQuotaSpec,
    target: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    for split in _FINAL_SPLIT_ORDER:
        split_rows = [row for row in rows if final_splits[row.plan_id] == split]
        quotas = spec.user_style_targets[split]
        for style in (
            "code_mixed_or_numeric",
            "typo_or_asr_like",
            "polite",
            "terse_fragment",
            "neutral_complete",
            "colloquial",
        ):
            candidates = _remaining(split_rows, target, "user_style")
            required = quotas[style]
            if style == "colloquial":
                if required != len(candidates):
                    raise ValueError("style quotas leave an invalid colloquial remainder")
                selected = tuple(candidates)
            else:
                selected = _select_stratified(
                    candidates,
                    required,
                    seed=seed,
                    namespace=f"style-{split}-{style}",
                )
            _set_selection(target, selected, "user_style", style)


def _recipe_id_for(
    row: PlanLikeRow,
    values: Mapping[str, Any],
) -> str:
    interaction = values["interaction_pattern"]
    ambiguity = values["ambiguity_subtype"]
    if interaction == "underspecified_clarification":
        return (
            "boundary_clarification_dialogue"
            if ambiguity == "clarification_required"
            else "clarification_dialogue"
        )
    dialogue_recipes = {
        "constraint_correction": "constraint_correction_dialogue",
        "no_result_relaxation": "no_result_relaxation_dialogue",
        "goal_change_or_multi_query": "goal_change_dialogue",
    }
    if interaction in dialogue_recipes:
        return dialogue_recipes[interaction]
    if row.is_boundary:
        return "boundary_resolution_request"
    direct_recipes = {
        "product.exact_match": "exact_lookup",
        "product.multi_search": "multi_product_search",
        "product.style_recommendation": "style_discovery",
        "knowledge.visual_encyclopedia": "visual_encyclopedia",
        "utility.document_reading": "document_reading",
        "utility.recipe_guidance": "recipe_guidance",
    }
    try:
        return direct_recipes[row.canonical_capability]
    except KeyError as exc:
        raise ValueError(f"no direct prompt recipe for {row.canonical_capability}") from exc


def canonical_realism_assignments_bytes(
    assignments: Iterable[RealismAssignment],
) -> bytes:
    materialized = tuple(assignments)
    plan_ids = [assignment.plan_id for assignment in materialized]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("realism assignment plan IDs must be unique")
    return canonical_jsonl_bytes(
        assignment for assignment in sorted(materialized, key=lambda item: item.plan_id)
    )


def _allocate_realism_assignments(
    rows: Sequence[PlanLikeRow],
    *,
    final_splits: Mapping[str, FinalSplit],
    reuse: Mapping[str, ReuseAssignment],
    recipe_library: Sequence[PromptRecipe],
    spec: RealismQuotaSpec,
    seed: int,
) -> tuple[RealismAssignment, ...]:
    """Rebuild the exact seeded allocation, including every stratum choice."""

    recipe_index = _recipe_index(recipe_library)
    state: dict[str, dict[str, Any]] = {row.plan_id: {} for row in rows}
    _assign_interactions(rows, final_splits, spec, state, seed=seed)
    _assign_expression_levels(rows, final_splits, spec, state, seed=seed)
    _assign_constraint_levels(rows, final_splits, spec, state, seed=seed)
    _assign_user_styles(rows, final_splits, spec, state, seed=seed)

    assignments: list[RealismAssignment] = []
    for row in rows:
        values = state[row.plan_id]
        recipe_id = _recipe_id_for(row, values)
        recipe = recipe_index[recipe_id]
        trajectory_shape: TrajectoryShape = (
            "single_turn"
            if values["interaction_pattern"] == "direct_request"
            else "three_turn"
        )
        if not recipe.matches(
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            interaction_pattern=values["interaction_pattern"],
            trajectory_shape=trajectory_shape,
        ):
            raise ValueError(f"selected prompt recipe is incompatible: {row.plan_id}")
        assignments.append(
            RealismAssignment(
                plan_id=row.plan_id,
                trajectory_shape=trajectory_shape,
                interaction_pattern=values["interaction_pattern"],
                expression_level=values["expression_level"],
                user_style=values["user_style"],
                constraint_level=values["constraint_level"],
                ambiguity_subtype=values["ambiguity_subtype"],
                reuse_variant=reuse[row.plan_id].reuse_variant,
                prompt_recipe_id=recipe_id,
                prompt_recipe_sha256=prompt_recipe_sha256(recipe),
            )
        )
    return tuple(sorted(assignments, key=lambda item: item.plan_id))


def build_realism_sidecar(
    plan_rows: object,
    *,
    final_split_by_plan_id: Mapping[str, object] | Sequence[object],
    reuse_by_plan_id: Mapping[str, object] | Sequence[object],
    plan_sha256: str,
    asset_catalog_sha256: str,
    capability_assignments_sha256: str,
    seed: int,
    recipes: Sequence[PromptRecipe] | None = None,
    quota_spec: RealismQuotaSpec | None = None,
) -> RealismSidecar:
    """Allocate all text-free realism cards deterministically.

    The allocator is intentionally independent of image bytes and natural
    language.  It validates every frozen distribution before returning.
    """

    rows = coerce_plan_rows(plan_rows)
    final_splits = _normalize_final_splits(rows, final_split_by_plan_id)
    reuse = _normalize_reuse_assignments(rows, reuse_by_plan_id)
    recipe_library = tuple(default_prompt_recipes() if recipes is None else recipes)
    recipe_manifest = build_prompt_recipe_manifest(recipe_library)
    spec = _resolve_quota_spec(rows, final_splits, seed=seed, quota_spec=quota_spec)
    assignment_tuple = _allocate_realism_assignments(
        rows,
        final_splits=final_splits,
        reuse=reuse,
        recipe_library=recipe_library,
        spec=spec,
        seed=seed,
    )
    assignments_sha256 = sha256_bytes(canonical_realism_assignments_bytes(assignment_tuple))
    final_split_digest = sha256_bytes(canonical_final_split_sidecar_bytes(final_splits))
    reuse_digest = sha256_bytes(canonical_reuse_sidecar_bytes(reuse))
    manifest = RealismAssignmentsManifest(
        plan_sha256=plan_sha256,
        asset_catalog_sha256=asset_catalog_sha256,
        capability_assignments_sha256=capability_assignments_sha256,
        seed=seed,
        row_count=len(assignment_tuple),
        assignments_sha256=assignments_sha256,
        prompt_recipe_manifest_sha256=sha256_bytes(canonical_json_bytes(recipe_manifest)),
        final_split_sidecar_sha256=final_split_digest,
        reuse_sidecar_sha256=reuse_digest,
    )
    sidecar = RealismSidecar(
        assignments=assignment_tuple,
        manifest=manifest,
        recipes=tuple(sorted(recipe_library, key=lambda item: item.prompt_recipe_id)),
        recipe_manifest=recipe_manifest,
    )
    validate_realism_sidecar(
        sidecar,
        plan_rows=rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        quota_spec=spec,
    )
    return sidecar


def audit_realism_assignments(
    assignments: Sequence[RealismAssignment],
    final_split_by_plan_id: Mapping[str, FinalSplit],
) -> dict[str, dict[str, Counter[str]] | Counter[str]]:
    """Return count-only diagnostics; it never inspects or emits user text."""

    result: dict[str, dict[str, Counter[str]] | Counter[str]] = {
        "interaction": Counter(),
        "expression": Counter(),
        "user_style": Counter(),
        "constraint": Counter(),
        "ambiguity": Counter(),
        "by_split": {},
    }
    for split in _FINAL_SPLIT_ORDER:
        result["by_split"][split] = {
            "interaction": Counter(),
            "expression": Counter(),
            "user_style": Counter(),
            "constraint": Counter(),
        }
    for assignment in assignments:
        split = final_split_by_plan_id[assignment.plan_id]
        cast_global = result
        cast_global["interaction"][assignment.interaction_pattern] += 1
        cast_global["expression"][assignment.expression_level] += 1
        cast_global["user_style"][assignment.user_style] += 1
        cast_global["constraint"][assignment.constraint_level] += 1
        if assignment.ambiguity_subtype != "none":
            cast_global["ambiguity"][assignment.ambiguity_subtype] += 1
        split_counts = cast_global["by_split"][split]
        split_counts["interaction"][assignment.interaction_pattern] += 1
        split_counts["expression"][assignment.expression_level] += 1
        split_counts["user_style"][assignment.user_style] += 1
        split_counts["constraint"][assignment.constraint_level] += 1
    return result


def validate_realism_sidecar(
    sidecar: RealismSidecar,
    *,
    plan_rows: object,
    final_split_by_plan_id: Mapping[str, object] | Sequence[object],
    reuse_by_plan_id: Mapping[str, object] | Sequence[object],
    quota_spec: RealismQuotaSpec | None = None,
) -> None:
    rows = coerce_plan_rows(plan_rows)
    final_splits = _normalize_final_splits(rows, final_split_by_plan_id)
    reuse = _normalize_reuse_assignments(rows, reuse_by_plan_id)
    _require_exact_plan_id_coverage(rows, {item.plan_id: item for item in sidecar.assignments}, "realism sidecar")
    if sidecar.manifest.row_count != len(sidecar.assignments):
        raise ValueError("realism manifest row count mismatch")
    if sidecar.manifest.assignments_sha256 != sha256_bytes(
        canonical_realism_assignments_bytes(sidecar.assignments)
    ):
        raise ValueError("realism assignment digest mismatch")
    if sidecar.recipe_manifest != build_prompt_recipe_manifest(sidecar.recipes):
        raise ValueError("prompt recipe manifest mismatch")
    if sidecar.manifest.prompt_recipe_manifest_sha256 != sha256_bytes(
        canonical_json_bytes(sidecar.recipe_manifest)
    ):
        raise ValueError("prompt recipe manifest digest mismatch")
    if sidecar.manifest.final_split_sidecar_sha256 != sha256_bytes(
        canonical_final_split_sidecar_bytes(final_splits)
    ):
        raise ValueError("final split sidecar digest mismatch")
    if sidecar.manifest.reuse_sidecar_sha256 != sha256_bytes(
        canonical_reuse_sidecar_bytes(reuse)
    ):
        raise ValueError("reuse sidecar digest mismatch")
    recipe_index = _recipe_index(sidecar.recipes)
    rows_by_id = {row.plan_id: row for row in rows}
    for assignment in sidecar.assignments:
        row = rows_by_id[assignment.plan_id]
        if assignment.reuse_variant != reuse[assignment.plan_id].reuse_variant:
            raise ValueError("realism sidecar drifted from reuse sidecar")
        if row.is_boundary:
            if assignment.ambiguity_subtype == "none":
                raise ValueError("boundary row is missing ambiguity subtype")
        elif assignment.ambiguity_subtype != "none":
            raise ValueError("non-boundary row cannot carry ambiguity subtype")
        recipe = recipe_index.get(assignment.prompt_recipe_id)
        if recipe is None or assignment.prompt_recipe_sha256 != prompt_recipe_sha256(recipe):
            raise ValueError("realism assignment prompt recipe digest mismatch")
        if not recipe.matches(
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            interaction_pattern=assignment.interaction_pattern,
            trajectory_shape=assignment.trajectory_shape,
        ):
            raise ValueError("realism assignment selects an incompatible recipe")
    spec = _resolve_quota_spec(
        rows,
        final_splits,
        seed=sidecar.manifest.seed,
        quota_spec=quota_spec,
    )
    expected_assignments = _allocate_realism_assignments(
        rows,
        final_splits=final_splits,
        reuse=reuse,
        recipe_library=sidecar.recipes,
        spec=spec,
        seed=sidecar.manifest.seed,
    )
    if sidecar.assignments != expected_assignments:
        raise ValueError(
            "realism sidecar does not match the exact seeded stratum allocation"
        )
    audit = audit_realism_assignments(sidecar.assignments, final_splits)
    if audit["ambiguity"] != Counter(spec.ambiguity_targets):
        raise ValueError("ambiguity quotas are not exact")
    for split in _FINAL_SPLIT_ORDER:
        split_counts = audit["by_split"][split]
        if split_counts["interaction"] != Counter(spec.interaction_targets[split]):
            raise ValueError(f"interaction quotas are not exact for {split}")
        if split_counts["expression"] != Counter(spec.expression_targets[split]):
            raise ValueError(f"expression quotas are not exact for {split}")
        if split_counts["user_style"] != Counter(spec.user_style_targets[split]):
            raise ValueError(f"user style quotas are not exact for {split}")
        if split_counts["constraint"] != Counter(spec.constraint_targets[split]):
            raise ValueError(f"constraint quotas are not exact for {split}")


def _normalize_asset_bindings(
    asset_bindings: Mapping[str, object] | Sequence[object],
) -> dict[str, InternalAssetBinding]:
    result: dict[str, InternalAssetBinding] = {}
    if isinstance(asset_bindings, Mapping):
        iterator = asset_bindings.items()
        for asset_id, raw in iterator:
            if isinstance(raw, InternalAssetBinding):
                binding = raw
            else:
                payload = dict(_coerce_payload(raw))
                payload.setdefault("asset_id", asset_id)
                binding = InternalAssetBinding.model_validate(payload)
            if binding.asset_id != asset_id:
                raise ValueError("asset binding mapping key does not match asset_id")
            result[binding.asset_id] = binding
    else:
        for raw in asset_bindings:
            binding = (
                raw
                if isinstance(raw, InternalAssetBinding)
                else InternalAssetBinding.model_validate(_coerce_payload(raw))
            )
            if binding.asset_id in result:
                raise ValueError("asset bindings must not repeat asset_id")
            result[binding.asset_id] = binding
    return result


def _opaque_extension(binding: InternalAssetBinding, row: PlanLikeRow) -> str:
    candidates = (binding.canonical_path, row.image_path or "")
    for candidate in candidates:
        suffix = Path(candidate).suffix.lower().lstrip(".")
        if re.fullmatch(r"[a-z0-9]{1,8}", suffix):
            return suffix
    return "bin"


def _build_aliases(
    rows: Sequence[PlanLikeRow],
    bindings: Mapping[str, InternalAssetBinding],
) -> tuple[OpaqueAssetBinding, ...]:
    seen: set[str] = set()
    aliases: list[OpaqueAssetBinding] = []
    for row in rows:
        if row.asset_id is None:
            raise ValueError("a real authoring work order requires asset_id")
        if row.asset_id in seen:
            continue
        try:
            binding = bindings[row.asset_id]
        except KeyError as exc:
            raise ValueError(f"missing internal asset binding: {row.asset_id}") from exc
        seen.add(row.asset_id)
        alias = f"a{len(aliases) + 1:04d}.{_opaque_extension(binding, row)}"
        aliases.append(
            OpaqueAssetBinding(
                opaque_alias=alias,
                materialization_relative_path=f"author-assets/{alias}",
                asset_id=binding.asset_id,
                asset_sha256=binding.asset_sha256,
                byte_size=binding.byte_size,
                canonical_path=binding.canonical_path,
                source_dataset=binding.source_dataset,
                source_record_id=binding.source_record_id,
            )
        )
    return tuple(aliases)


def _selected_recipe_projection(recipe: PromptRecipe) -> AuthorPacketRecipe:
    return AuthorPacketRecipe(
        prompt_recipe_id=recipe.prompt_recipe_id,
        prompt_recipe_sha256=prompt_recipe_sha256(recipe),
        trajectory_shape=recipe.trajectory_shape,
        structural_slots=recipe.structural_slots,
        required_elements=recipe.required_elements,
        prohibited_elements=recipe.prohibited_elements,
    )


def _packet_from_items(
    *,
    job_id: str,
    base_batch_id: str,
    generation_input_sha256: str,
    items: Sequence[InternalWorkOrderItem],
    recipes: Mapping[str, PromptRecipe],
) -> AuthorPacket:
    return AuthorPacket(
        job_id=job_id,
        base_batch_id=base_batch_id,
        generation_input_sha256=generation_input_sha256,
        items=tuple(
            AuthorPacketItem(
                plan_id=item.plan_id,
                canonical_intent=item.canonical_intent,
                canonical_capability=item.canonical_capability,
                boundary=AuthorPacketBoundary(
                    is_boundary=item.is_boundary,
                    boundary_strategy=item.boundary_strategy if item.is_boundary else None,
                ),
                realism=AuthorPacketRealismCard(
                    trajectory_shape=item.realism.trajectory_shape,
                    interaction_pattern=item.realism.interaction_pattern,
                    expression_level=item.realism.expression_level,
                    user_style=item.realism.user_style,
                    constraint_level=item.realism.constraint_level,
                    ambiguity_subtype=item.realism.ambiguity_subtype,
                ),
                recipe=_selected_recipe_projection(recipes[item.realism.prompt_recipe_id]),
                opaque_alias=item.opaque_alias,
            )
            for item in items
        ),
    )


def _generation_input_payload(
    *,
    base_batch_id: str,
    plan_sha256: str,
    asset_catalog_sha256: str,
    capability_assignments_sha256: str,
    realism_manifest: RealismAssignmentsManifest,
    recipe_manifest: PromptRecipeManifest,
    items: Sequence[InternalWorkOrderItem],
    aliases: Sequence[OpaqueAssetBinding],
) -> dict[str, object]:
    """Stable internal inputs whose digest invalidates stale author drafts."""

    return {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "base_batch_id": base_batch_id,
        "plan_sha256": plan_sha256,
        "asset_catalog_sha256": asset_catalog_sha256,
        "capability_assignments_sha256": capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(canonical_json_bytes(realism_manifest)),
        "realism_assignments_sha256": realism_manifest.assignments_sha256,
        "prompt_recipe_manifest_sha256": sha256_bytes(canonical_json_bytes(recipe_manifest)),
        "final_split_sidecar_sha256": realism_manifest.final_split_sidecar_sha256,
        "reuse_sidecar_sha256": realism_manifest.reuse_sidecar_sha256,
        "items": [
            {
                "plan_id": item.plan_id,
                "asset_id": item.asset_id,
                "canonical_intent": item.canonical_intent,
                "canonical_capability": item.canonical_capability,
                "is_boundary": item.is_boundary,
                "boundary_strategy": item.boundary_strategy,
                "realism": item.realism.model_dump(mode="json"),
                "opaque_alias": item.opaque_alias,
            }
            for item in items
        ],
        "aliases": [
            {
                "opaque_alias": alias.opaque_alias,
                "asset_id": alias.asset_id,
                "asset_sha256": alias.asset_sha256,
                "byte_size": alias.byte_size,
            }
            for alias in aliases
        ],
    }


def compute_generation_input_sha256(
    *,
    base_batch_id: str,
    plan_sha256: str,
    asset_catalog_sha256: str,
    capability_assignments_sha256: str,
    realism_manifest: RealismAssignmentsManifest,
    recipe_manifest: PromptRecipeManifest,
    items: Sequence[InternalWorkOrderItem],
    aliases: Sequence[OpaqueAssetBinding],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            _generation_input_payload(
                base_batch_id=base_batch_id,
                plan_sha256=plan_sha256,
                asset_catalog_sha256=asset_catalog_sha256,
                capability_assignments_sha256=capability_assignments_sha256,
                realism_manifest=realism_manifest,
                recipe_manifest=recipe_manifest,
                items=items,
                aliases=aliases,
            )
        )
    )


def canonical_author_packet_bytes(packet: AuthorPacket) -> bytes:
    return canonical_json_bytes(packet)


def canonical_work_order_bytes(order: InternalWorkOrder) -> bytes:
    return canonical_json_bytes(order)


def work_order_sha256(order: InternalWorkOrder) -> str:
    return sha256_bytes(canonical_work_order_bytes(order))


def _work_order_identity_payload(
    *,
    base_batch_id: str,
    plan_sha256: str,
    asset_catalog_sha256: str,
    capability_assignments_sha256: str,
    realism_manifest: RealismAssignmentsManifest,
    recipe_manifest: PromptRecipeManifest,
    items: Sequence[InternalWorkOrderItem],
    aliases: Sequence[OpaqueAssetBinding],
) -> dict[str, object]:
    return {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "base_batch_id": base_batch_id,
        "plan_sha256": plan_sha256,
        "asset_catalog_sha256": asset_catalog_sha256,
        "capability_assignments_sha256": capability_assignments_sha256,
        "realism_manifest": realism_manifest.model_dump(mode="json"),
        "recipe_manifest": recipe_manifest.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in items],
        "aliases": [alias.model_dump(mode="json") for alias in aliases],
    }


def _deterministic_job_id(identity_payload: Mapping[str, object]) -> str:
    return "r2-author-" + sha256_bytes(canonical_json_bytes(dict(identity_payload)))[:24]


def author_packet_from_work_order(order: InternalWorkOrder) -> AuthorPacket:
    recipes = _recipe_index(order.prompt_recipes)
    return _packet_from_items(
        job_id=order.job_id,
        base_batch_id=order.base_batch_id,
        generation_input_sha256=order.generation_input_sha256,
        items=order.items,
        recipes=recipes,
    )


def build_authoring_job(
    plan_rows: object,
    *,
    final_split_by_plan_id: Mapping[str, object] | Sequence[object],
    reuse_by_plan_id: Mapping[str, object] | Sequence[object],
    realism_sidecar: RealismSidecar,
    base_batch_id: str,
    asset_bindings: Mapping[str, object] | Sequence[object],
) -> AuthoringJob:
    """Build a pure in-memory 25-item authoring job and its safe projection."""

    rows = coerce_plan_rows(plan_rows)
    final_splits = _normalize_final_splits(rows, final_split_by_plan_id)
    reuse = _normalize_reuse_assignments(rows, reuse_by_plan_id)
    validate_realism_sidecar(
        realism_sidecar,
        plan_rows=rows,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
    )
    selected_rows = tuple(
        sorted(
            (row for row in rows if row.batch_id == base_batch_id),
            key=lambda row: row.position,
        )
    )
    if len(selected_rows) != 25 or [row.position for row in selected_rows] != list(range(1, 26)):
        raise ValueError("authoring job must select one complete 25-item batch")
    bindings = _normalize_asset_bindings(asset_bindings)
    aliases = _build_aliases(selected_rows, bindings)
    alias_by_asset = {alias.asset_id: alias.opaque_alias for alias in aliases}
    realism_by_id = {assignment.plan_id: assignment for assignment in realism_sidecar.assignments}
    items = tuple(
        InternalWorkOrderItem(
            plan_id=row.plan_id,
            batch_id=row.batch_id,
            position=row.position,
            asset_id=row.asset_id or "",
            image_path=row.image_path,
            leakage_group_id=row.leakage_group_id,
            capability_assignment_id=row.capability_assignment_id,
            capability_assignment_source_sha256=row.capability_assignment_source_sha256,
            canonical_intent=row.canonical_intent,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            boundary_strategy=row.boundary_strategy,
            final_split=final_splits[row.plan_id],
            reuse_variant=reuse[row.plan_id].reuse_variant,
            reuse_reason=reuse[row.plan_id].reuse_reason,
            realism=realism_by_id[row.plan_id],
            opaque_alias=alias_by_asset[row.asset_id or ""],
        )
        for row in selected_rows
    )
    generation_input = compute_generation_input_sha256(
        base_batch_id=base_batch_id,
        plan_sha256=realism_sidecar.manifest.plan_sha256,
        asset_catalog_sha256=realism_sidecar.manifest.asset_catalog_sha256,
        capability_assignments_sha256=realism_sidecar.manifest.capability_assignments_sha256,
        realism_manifest=realism_sidecar.manifest,
        recipe_manifest=realism_sidecar.recipe_manifest,
        items=items,
        aliases=aliases,
    )
    identity = _work_order_identity_payload(
        base_batch_id=base_batch_id,
        plan_sha256=realism_sidecar.manifest.plan_sha256,
        asset_catalog_sha256=realism_sidecar.manifest.asset_catalog_sha256,
        capability_assignments_sha256=realism_sidecar.manifest.capability_assignments_sha256,
        realism_manifest=realism_sidecar.manifest,
        recipe_manifest=realism_sidecar.recipe_manifest,
        items=items,
        aliases=aliases,
    )
    job_id = _deterministic_job_id(identity)
    packet = _packet_from_items(
        job_id=job_id,
        base_batch_id=base_batch_id,
        generation_input_sha256=generation_input,
        items=items,
        recipes=_recipe_index(realism_sidecar.recipes),
    )
    assert_author_packet_safe(packet)
    order = InternalWorkOrder(
        job_id=job_id,
        base_batch_id=base_batch_id,
        plan_sha256=realism_sidecar.manifest.plan_sha256,
        asset_catalog_sha256=realism_sidecar.manifest.asset_catalog_sha256,
        capability_assignments_sha256=realism_sidecar.manifest.capability_assignments_sha256,
        realism_manifest=realism_sidecar.manifest,
        recipe_manifest=realism_sidecar.recipe_manifest,
        realism_manifest_sha256=sha256_bytes(canonical_json_bytes(realism_sidecar.manifest)),
        realism_assignments_sha256=realism_sidecar.manifest.assignments_sha256,
        prompt_recipe_manifest_sha256=sha256_bytes(
            canonical_json_bytes(realism_sidecar.recipe_manifest)
        ),
        final_split_sidecar_sha256=realism_sidecar.manifest.final_split_sidecar_sha256,
        reuse_sidecar_sha256=realism_sidecar.manifest.reuse_sidecar_sha256,
        generation_input_sha256=generation_input,
        author_packet_sha256=sha256_bytes(canonical_author_packet_bytes(packet)),
        prompt_recipes=realism_sidecar.recipes,
        items=items,
        aliases=aliases,
    )
    checkpoint = issued_checkpoint(order)
    job = AuthoringJob(work_order=order, author_packet=packet, checkpoint=checkpoint)
    validate_authoring_job(job)
    return job


def scan_author_packet_for_leaks(
    packet: AuthorPacket | Mapping[str, object] | str | bytes,
    *,
    forbidden_fragments: Iterable[str] = (),
) -> tuple[PacketLeakFinding, ...]:
    """Scan a visible packet for path, provenance, and caller-provided leaks."""

    if isinstance(packet, AuthorPacket):
        payload: object = packet.model_dump(mode="json")
    elif isinstance(packet, Mapping):
        payload = packet
    elif isinstance(packet, bytes):
        payload = packet.decode("utf-8", errors="replace")
    else:
        payload = packet
    forbidden = tuple(
        _nonblank(fragment, "forbidden fragment").casefold()
        for fragment in forbidden_fragments
    )
    findings: list[PacketLeakFinding] = []
    banned_key_fragments = (
        "source",
        "path",
        "split",
        "assignment",
        "provenance",
        "asset_id",
        "leakage",
        "catalog",
    )
    absolute_path = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|(?:^|[\"'\s])/(?:[^\s\"']+))")

    def add(pointer: str, rule: str) -> None:
        candidate = PacketLeakFinding(json_pointer=pointer or "/", rule=rule)
        if candidate not in findings:
            findings.append(candidate)

    def walk(value: object, pointer: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_pointer = f"{pointer}/{key_text}" if pointer else f"/{key_text}"
                if any(fragment in key_text.casefold() for fragment in banned_key_fragments):
                    add(child_pointer, "forbidden_field")
                walk(child, child_pointer)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}" if pointer else f"/{index}")
        elif isinstance(value, str):
            if absolute_path.search(value):
                add(pointer, "absolute_path")
            lowered = value.casefold()
            for fragment in forbidden:
                if fragment in lowered:
                    add(pointer, "forbidden_fragment")

    walk(payload, "")
    return tuple(findings)


def assert_author_packet_safe(
    packet: AuthorPacket | Mapping[str, object] | str | bytes,
    *,
    forbidden_fragments: Iterable[str] = (),
) -> None:
    findings = scan_author_packet_for_leaks(packet, forbidden_fragments=forbidden_fragments)
    if findings:
        summary = ", ".join(f"{item.json_pointer}:{item.rule}" for item in findings[:3])
        raise ValueError(f"author packet leakage scanner rejected payload: {summary}")


def issued_checkpoint(order: InternalWorkOrder) -> AuthoringCheckpoint:
    return AuthoringCheckpoint(
        job_id=order.job_id,
        work_order_sha256=work_order_sha256(order),
        generation_input_sha256=order.generation_input_sha256,
        state="issued",
    )


def transition_authoring_checkpoint(
    checkpoint: AuthoringCheckpoint,
    order: InternalWorkOrder,
    *,
    state: Literal["issued", "staged", "accepted"],
    staged_batch_id: str | None = None,
    staged_results_sha256: str | None = None,
    accepted_batch_id: str | None = None,
) -> AuthoringCheckpoint:
    """Pure state transition; backward or skipping transitions fail closed."""

    expected_work_order_sha = work_order_sha256(order)
    if (
        checkpoint.job_id != order.job_id
        or checkpoint.work_order_sha256 != expected_work_order_sha
        or checkpoint.generation_input_sha256 != order.generation_input_sha256
    ):
        raise ValueError("checkpoint does not bind this work order")
    rank = {"issued": 0, "staged": 1, "accepted": 2}
    if rank[state] < rank[checkpoint.state]:
        raise ValueError("checkpoint may not move backward")
    if rank[state] - rank[checkpoint.state] > 1:
        raise ValueError("checkpoint may not skip a state")
    if state == checkpoint.state:
        supplied = (staged_batch_id, staged_results_sha256, accepted_batch_id)
        existing = (
            checkpoint.staged_batch_id,
            checkpoint.staged_results_sha256,
            checkpoint.accepted_batch_id,
        )
        if any(value is not None for value in supplied) and supplied != existing:
            raise ValueError("idempotent checkpoint transition cannot change bindings")
        desired = AuthoringCheckpoint(
            job_id=checkpoint.job_id,
            work_order_sha256=checkpoint.work_order_sha256,
            generation_input_sha256=checkpoint.generation_input_sha256,
            state=checkpoint.state,
            staged_batch_id=checkpoint.staged_batch_id,
            staged_results_sha256=checkpoint.staged_results_sha256,
            accepted_batch_id=checkpoint.accepted_batch_id,
        )
        if desired != checkpoint:
            raise ValueError("idempotent checkpoint transition cannot change bindings")
        return checkpoint
    if state == "staged":
        return AuthoringCheckpoint(
            job_id=checkpoint.job_id,
            work_order_sha256=checkpoint.work_order_sha256,
            generation_input_sha256=checkpoint.generation_input_sha256,
            state="staged",
            staged_batch_id=staged_batch_id,
            staged_results_sha256=staged_results_sha256,
        )
    if (
        checkpoint.staged_batch_id != staged_batch_id
        or checkpoint.staged_results_sha256 != staged_results_sha256
        or accepted_batch_id is None
    ):
        raise ValueError("accepted checkpoint must retain the exact staged bindings")
    return AuthoringCheckpoint(
        job_id=checkpoint.job_id,
        work_order_sha256=checkpoint.work_order_sha256,
        generation_input_sha256=checkpoint.generation_input_sha256,
        state="accepted",
        staged_batch_id=checkpoint.staged_batch_id,
        staged_results_sha256=checkpoint.staged_results_sha256,
        accepted_batch_id=accepted_batch_id,
    )


def validate_authoring_job(job: AuthoringJob) -> None:
    """Rebuild the visible projection and verify all pure in-memory bindings."""

    order = job.work_order
    expected_job_id = _deterministic_job_id(
        _work_order_identity_payload(
            base_batch_id=order.base_batch_id,
            plan_sha256=order.plan_sha256,
            asset_catalog_sha256=order.asset_catalog_sha256,
            capability_assignments_sha256=order.capability_assignments_sha256,
            realism_manifest=order.realism_manifest,
            recipe_manifest=order.recipe_manifest,
            items=order.items,
            aliases=order.aliases,
        )
    )
    if order.job_id != expected_job_id:
        raise ValueError("work order has an invalid deterministic job ID")
    packet = author_packet_from_work_order(order)
    if packet != job.author_packet:
        raise ValueError("author packet no longer matches internal work order")
    assert_author_packet_safe(packet)
    if order.author_packet_sha256 != sha256_bytes(canonical_author_packet_bytes(packet)):
        raise ValueError("internal work order author packet digest mismatch")
    recomputed_generation = compute_generation_input_sha256(
        base_batch_id=order.base_batch_id,
        plan_sha256=order.plan_sha256,
        asset_catalog_sha256=order.asset_catalog_sha256,
        capability_assignments_sha256=order.capability_assignments_sha256,
        realism_manifest=order.realism_manifest,
        recipe_manifest=order.recipe_manifest,
        items=order.items,
        aliases=order.aliases,
    )
    if recomputed_generation != order.generation_input_sha256:
        raise ValueError("work order generation input digest mismatch")
    if job.checkpoint.job_id != order.job_id:
        raise ValueError("checkpoint job ID mismatch")
    if job.checkpoint.work_order_sha256 != work_order_sha256(order):
        raise ValueError("checkpoint work order digest mismatch")
    if job.checkpoint.generation_input_sha256 != order.generation_input_sha256:
        raise ValueError("checkpoint generation input digest mismatch")


def canonical_json_text(value: BaseModel | Mapping[str, object] | Sequence[object]) -> str:
    """A small inspection helper for tests and non-publishing diagnostics."""

    if isinstance(value, BaseModel):
        return canonical_json_bytes(value).decode("utf-8")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
