"""Phase 3 内部数据契约。正式 Query 仍以 schemas.Query 为唯一权威。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.schemas import BoundaryStrategy, ConversationTurn, Intent

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BatchState = Literal["planned", "staging", "accepted", "rejected"]
PROVISIONAL_LEAKAGE_POLICY_VERSION = "relative-path-plus-plan-groups-v1"

_INTENTS = {
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_nonblank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} 不得为空")
    return value


def _validate_turn_shape(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    roles = [turn.role for turn in turns]
    if roles not in (["user"], ["user", "assistant", "user"]):
        raise ValueError("turns 只能是 [user] 或 [user, assistant, user]")
    return turns


def _validate_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} 必须包含时区")
    return value


def _validate_catalog_binding(
    catalog_sha256: str | None,
    leakage_policy_version: str,
    *,
    artifact: str,
) -> None:
    policy = _require_nonblank(leakage_policy_version, "leakage_policy_version")
    if policy != leakage_policy_version:
        raise ValueError("leakage_policy_version 不得包含首尾空白")
    is_provisional = policy == PROVISIONAL_LEAKAGE_POLICY_VERSION
    if catalog_sha256 is None and not is_provisional:
        raise ValueError(f"non-provisional {artifact} 必须绑定 asset_catalog_sha256")
    if catalog_sha256 is not None and is_provisional:
        raise ValueError(
            f"catalog-bound {artifact} 不得使用 provisional leakage policy"
        )


class PlannedQuery(StrictModel):
    schema_version: Literal[2] = 2
    plan_id: str
    batch_id: str
    position: int = Field(ge=1, le=25)
    taxonomy_version: str
    task_spec_version: str
    asset_id: str
    image_path: str
    leakage_group_id: str
    template_family: str
    generator_batch_id: str
    canonical_intent: Intent
    canonical_capability: str
    acceptable_capabilities: list[str]
    requires_card: bool
    capability_assignment_id: str | None = None
    capability_assignment_source_sha256: Sha256 | None = None
    is_boundary: bool
    boundary_strategy: BoundaryStrategy | None = None
    boundary_group_id: str | None = None
    provisional_split: Literal["dev_mini", "opt_pool"]

    @field_validator(
        "plan_id",
        "batch_id",
        "taxonomy_version",
        "task_spec_version",
        "asset_id",
        "image_path",
        "leakage_group_id",
        "template_family",
        "generator_batch_id",
        "canonical_capability",
    )
    @classmethod
    def require_text_fields(cls, value: str, info) -> str:
        return _require_nonblank(value, info.field_name)

    @field_validator("capability_assignment_id")
    @classmethod
    def normalize_assignment_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _require_nonblank(value, "capability_assignment_id")
        )

    @field_validator("boundary_group_id")
    @classmethod
    def normalize_group(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonblank(value, "boundary_group_id")

    @model_validator(mode="after")
    def validate_boundary_fields(self):
        if not self.is_boundary:
            if self.boundary_strategy is not None or self.boundary_group_id is not None:
                raise ValueError("非 boundary 样本不得设置 boundary 字段")
        elif self.boundary_strategy is None:
            raise ValueError("boundary 样本必须设置 boundary_strategy")
        elif self.boundary_strategy == "natural_ambiguity":
            if self.boundary_group_id is not None:
                raise ValueError("natural_ambiguity boundary 不得设置 group")
        elif self.boundary_group_id is None:
            raise ValueError("cross_intent boundary 必须设置 group")
        if self.canonical_capability not in self.acceptable_capabilities:
            raise ValueError("canonical_capability 必须包含在 acceptable_capabilities")
        if len(self.acceptable_capabilities) != len(set(self.acceptable_capabilities)):
            raise ValueError("acceptable_capabilities 不得重复")
        if self.generator_batch_id != self.batch_id:
            raise ValueError("计划阶段 generator_batch_id 必须等于 batch_id")
        assignment_fields = (
            self.capability_assignment_id,
            self.capability_assignment_source_sha256,
        )
        if any(value is None for value in assignment_fields) and any(
            value is not None for value in assignment_fields
        ):
            raise ValueError("capability assignment provenance 字段必须同时提供")
        return self

    @property
    def gt_intent(self) -> Intent:
        return self.canonical_intent


class CorpusPlan(StrictModel):
    schema_version: Literal[2] = 2
    scope: Literal["dev_mini", "core", "full"]
    seed: int
    asset_catalog_sha256: Sha256 | None = None
    capability_assignments_sha256: Sha256 | None = None
    leakage_policy_version: str = PROVISIONAL_LEAKAGE_POLICY_VERSION
    queries: list[PlannedQuery]

    @model_validator(mode="after")
    def validate_scope(self):
        _validate_catalog_binding(
            self.asset_catalog_sha256,
            self.leakage_policy_version,
            artifact="plan",
        )
        has_assignment_provenance = all(
            item.capability_assignment_id is not None
            and item.capability_assignment_source_sha256 is not None
            for item in self.queries
        )
        if self.asset_catalog_sha256 is not None:
            if self.capability_assignments_sha256 is None:
                raise ValueError(
                    "catalog-bound plan 必须绑定 capability assignments SHA-256"
                )
            if not has_assignment_provenance:
                raise ValueError(
                    "catalog-bound plan 的每条 query 必须保留 capability assignment provenance"
                )
        elif self.capability_assignments_sha256 is not None or any(
            item.capability_assignment_id is not None
            or item.capability_assignment_source_sha256 is not None
            for item in self.queries
        ):
            raise ValueError(
                "provisional plan 不得伪装成 externally locked capability assignment"
            )
        expected_size = {
            "dev_mini": 200,
            "core": 1500,
            "full": 4500,
        }[self.scope]
        if len(self.queries) != expected_size:
            raise ValueError(f"{self.scope} plan 必须恰好包含 {expected_size} 条")
        if self.scope == "dev_mini":
            if any(item.provisional_split != "dev_mini" for item in self.queries):
                raise ValueError("dev_mini plan 只能包含 dev_mini provisional split")
        else:
            if any(
                item.provisional_split != "dev_mini" for item in self.queries[:200]
            ) or any(
                item.provisional_split != "opt_pool" for item in self.queries[200:]
            ):
                raise ValueError(
                    f"{self.scope} plan 前 200 条必须锁定 dev_mini，其余为 opt_pool"
                )
        plan_ids = [item.plan_id for item in self.queries]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("plan_id 必须全局唯一")
        positions_by_batch: dict[str, list[int]] = defaultdict(list)
        for item in self.queries:
            positions_by_batch[item.batch_id].append(item.position)
        expected_positions = list(range(1, 26))
        invalid_batches = [
            batch_id
            for batch_id, positions in positions_by_batch.items()
            if sorted(positions) != expected_positions
        ]
        if invalid_batches:
            raise ValueError(
                "每个 batch 的 position 必须恰好覆盖 1..25: "
                + ", ".join(sorted(invalid_batches))
            )
        expected_batches = expected_size // 25
        if len(positions_by_batch) != expected_batches:
            raise ValueError(f"{self.scope} plan batch 数量必须为 {expected_batches}")
        return self


class GeneratedTrajectory(StrictModel):
    plan_id: str
    turns: list[ConversationTurn]

    @field_validator("plan_id")
    @classmethod
    def require_plan_id(cls, value: str) -> str:
        return _require_nonblank(value, "plan_id")

    @field_validator("turns")
    @classmethod
    def validate_turns(cls, value: list[ConversationTurn]) -> list[ConversationTurn]:
        return _validate_turn_shape(value)


class SeedDraft(StrictModel):
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    examples: dict[Intent, list[str]]

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware_datetime(value, "generated_at")

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, value: dict[Intent, list[str]]):
        if set(value) != _INTENTS:
            raise ValueError("种子必须完整覆盖五个意图")
        normalized: dict[Intent, list[str]] = {}
        for intent, examples in value.items():
            cleaned = [_require_nonblank(item, f"{intent} seed") for item in examples]
            if (
                len(cleaned) != 3
                or len({"".join(item.split()) for item in cleaned}) != 3
            ):
                raise ValueError(f"{intent} 必须恰好包含 3 条不同种子")
            normalized[intent] = cleaned
        return normalized


class SeedManifest(StrictModel):
    schema_version: Literal[1] = 1
    seed_batch_id: str
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    staged_at: datetime
    seed_set_sha256: Sha256

    @field_validator("generated_at", "staged_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info) -> datetime:
        return _validate_aware_datetime(value, info.field_name)


class PlanManifest(StrictModel):
    schema_version: Literal[2] = 2
    query_schema_version: Literal[2] = 2
    scope: Literal["dev_mini", "core", "full"]
    count: int = Field(gt=0)
    plan_sha256: Sha256
    parent_plan_sha256: Sha256 | None = None
    asset_catalog_sha256: Sha256 | None = None
    capability_assignments_sha256: Sha256 | None = None
    leakage_policy_version: str = PROVISIONAL_LEAKAGE_POLICY_VERSION

    @model_validator(mode="after")
    def validate_parent(self):
        _validate_catalog_binding(
            self.asset_catalog_sha256,
            self.leakage_policy_version,
            artifact="plan manifest",
        )
        if (self.asset_catalog_sha256 is None) != (
            self.capability_assignments_sha256 is None
        ):
            raise ValueError(
                "formal plan manifest 必须同时绑定 asset catalog 与 capability assignments"
            )
        if self.scope == "dev_mini" and self.parent_plan_sha256 is not None:
            raise ValueError("dev_mini manifest 不得设置 parent_plan_sha256")
        if self.scope in {"core", "full"} and self.parent_plan_sha256 is None:
            raise ValueError(f"{self.scope} manifest 必须设置 parent_plan_sha256")
        return self


class ActivePlanPointer(StrictModel):
    schema_version: Literal[2] = 2
    plan_schema_version: Literal[2] = 2
    scope: Literal["dev_mini", "core", "full"]
    plan_path: str
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256 | None = None
    leakage_policy_version: str = PROVISIONAL_LEAKAGE_POLICY_VERSION

    @field_validator("plan_path")
    @classmethod
    def require_path(cls, value: str) -> str:
        return _require_nonblank(value, "plan_path")

    @model_validator(mode="after")
    def validate_catalog_binding(self):
        _validate_catalog_binding(
            self.asset_catalog_sha256,
            self.leakage_policy_version,
            artifact="active plan pointer",
        )
        return self


class BatchDraftManifest(StrictModel):
    schema_version: Literal[2] = 2
    base_batch_id: str
    data_origin: Literal["synthetic_derived"]
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256 | None
    leakage_policy_version: str
    seed_set_sha256: Sha256
    draft_sha256: Sha256
    generation_input_sha256: Sha256

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _validate_aware_datetime(value, "generated_at")

    @model_validator(mode="after")
    def validate_catalog_binding(self):
        _validate_catalog_binding(
            self.asset_catalog_sha256,
            self.leakage_policy_version,
            artifact="batch draft manifest",
        )
        return self


class BatchManifest(StrictModel):
    schema_version: Literal[2] = 2
    query_schema_version: Literal[2] = 2
    plan_schema_version: Literal[2] = 2
    base_batch_id: str
    revision: int = Field(ge=1)
    data_origin: Literal["synthetic_derived"]
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    staged_at: datetime
    count: Literal[25]
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256 | None = None
    leakage_policy_version: str = PROVISIONAL_LEAKAGE_POLICY_VERSION
    seed_set_sha256: Sha256
    results_sha256: Sha256

    @field_validator("generated_at", "staged_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info) -> datetime:
        return _validate_aware_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_catalog_binding(self):
        _validate_catalog_binding(
            self.asset_catalog_sha256,
            self.leakage_policy_version,
            artifact="batch manifest",
        )
        return self
