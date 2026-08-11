"""Public data contracts.

Schema v2 uses explicit versions and migrations.  Query v1 is retained only as
an input contract for the audited migration path; new artifacts must use v2.
"""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Intent = Literal[
    "exact_match",  # 找同款
    "multi_product",  # 多商品分解比较
    "divergent_rec",  # 风格发散推荐
    "encyclopedia",  # 视觉百科
    "utility",  # 工具型任务
]

Split = Literal["dev_mini", "opt_pool", "val", "test_frozen"]

Tier = Literal["Good", "Average", "Poor"]

BoundaryStrategy = Literal[
    "cross_intent_triplet",
    "cross_intent_pair",
    "natural_ambiguity",
]
Episode = Literal["t0", "t1", "t2", "t3"]
LabelStatus = Literal["auto", "cross_agreed", "arbitrated"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _nonblank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} 不得为空")
    return value


class DatasetAsset(StrictModel):
    """Auditable asset identity; large image bytes remain outside Git."""

    schema_version: Literal[2]
    asset_id: str
    source_dataset: str
    source_revision: str
    source_record_id: str
    transform_policy_version: str
    local_path: str
    sha256: str
    phash: str | None = None
    near_duplicate_cluster_id: str | None = None
    product_id: str | None = None
    derivation_parent_asset_ids: list[str] = Field(default_factory=list)
    license_id: str
    source_url: str | None = None
    attribution: str | None = None
    cloud_upload_allowed: bool | None = None
    public_demo_allowed: bool = False

    @field_validator(
        "asset_id",
        "source_dataset",
        "source_revision",
        "source_record_id",
        "transform_policy_version",
        "local_path",
        "license_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("sha256 必须是 64 位小写十六进制")
        return value

    @field_validator("phash")
    @classmethod
    def validate_phash(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{16}", value):
            raise ValueError("phash 必须是 16 位小写十六进制（64 bit）")
        return value

    @field_validator(
        "near_duplicate_cluster_id",
        "product_id",
        "source_url",
        "attribution",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("derivation_parent_asset_ids")
    @classmethod
    def validate_parent_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "derivation_parent_asset_ids") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("derivation_parent_asset_ids 不得重复")
        return sorted(cleaned)

    @field_validator("local_path")
    @classmethod
    def validate_relative_posix_path(cls, value: str) -> str:
        value = value.replace("\\", "/")
        if value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
            raise ValueError("local_path 必须是相对 POSIX 路径")
        if ".." in value.split("/"):
            raise ValueError("local_path 不得包含 ..")
        return value

    @model_validator(mode="after")
    def validate_derivation(self):
        if self.asset_id in self.derivation_parent_asset_ids:
            raise ValueError("asset 不得把自身声明为 derivation parent")
        return self


class Product(StrictModel):
    """Strict product evidence embedded in tool results and index snapshots."""

    product_id: str
    title: str
    category_l1: str
    category_l2: str | None = None
    category_l3: str | None = None
    ocr_text: str | None = None
    image_path: str
    # The formal source portfolio is versioned independently from this wire
    # schema. Verified index builds already prove that this value equals the
    # source_dataset of the exact AssetCatalog row, so a historical closed
    # MUGE/MEP-3M Literal only prevents legitimate ABO/RPC/FashionIQ rows.
    source: str

    @field_validator("product_id", "title", "category_l1", "image_path", "source")
    @classmethod
    def validate_required_product_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("category_l2", "category_l3", "ocr_text")
    @classmethod
    def normalize_optional_product_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)


class KBEntry(BaseModel):
    entry_id: str
    title: str
    text: str
    kind: Literal["encyclopedia", "recipe"]
    origin: Literal["dump", "llm_synth"]  # LLM 离线补齐的条目必须标注来源
    synth_provider: Literal["fable", "gpt_5_6_sol"] | None = None
    synth_model: str | None = None

    @field_validator("synth_model")
    @classmethod
    def normalize_synth_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("synth_model 不得为空")
        return value

    @model_validator(mode="after")
    def validate_synthesis_provenance(self):
        if self.origin == "llm_synth" and (
            not self.synth_provider or not self.synth_model
        ):
            raise ValueError("llm_synth 条目必须提供 synth_provider 与 synth_model")
        if self.origin == "dump" and (self.synth_provider or self.synth_model):
            raise ValueError("dump 条目不得填写 synth_provider 或 synth_model")
        return self


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def require_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content 不得为空")
        return value


class QueryV1(StrictModel):
    """Exact legacy input contract.  It is never accepted by v2 loaders."""

    schema_version: Literal[1] = 1
    query_id: str
    image_path: str
    text: str
    gt_intent: Intent
    gt_skill: str | None = None
    is_boundary: bool = False
    split: Split
    label_status: LabelStatus = "auto"
    turns: list[ConversationTurn] = Field(default_factory=list)
    synth_provider: Literal["codex"] | None = None
    synth_model: str | None = None
    synthesis_batch_id: str | None = None
    synthesis_prompt_id: str | None = None
    seed_set_sha256: str | None = None


class LabelDecision(StrictModel):
    decision_type: Literal[
        "constructed", "cross_review", "arbitration", "migrated_unverified"
    ]
    annotator_kind: Literal["planner", "llm_pair", "human", "migration"]
    annotator_id: str
    canonical_intent: Intent
    canonical_capability: str | None
    acceptable_capabilities: list[str] = Field(default_factory=list)
    reason: str | None = None
    legacy_skill_slug: str | None = None
    source_artifact_sha256: str | None = None
    decided_at: datetime | None = None

    @field_validator("annotator_id")
    @classmethod
    def validate_annotator(cls, value: str) -> str:
        return _nonblank(value, "annotator_id")

    @field_validator("canonical_capability", "reason", "legacy_skill_slug")
    @classmethod
    def normalize_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("source_artifact_sha256")
    @classmethod
    def validate_source_hash(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("source_artifact_sha256 必须是 64 位小写十六进制")
        return value

    @field_validator("acceptable_capabilities")
    @classmethod
    def validate_acceptable_capabilities(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "acceptable_capabilities") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("acceptable_capabilities 不得重复")
        return cleaned

    @field_validator("decided_at")
    @classmethod
    def validate_decided_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("decided_at 必须包含时区")
        return value


class Query(StrictModel):
    """Schema-v2 query.  Bank skill slugs are deliberately absent."""

    schema_version: Literal[2]
    taxonomy_version: str
    task_spec_version: str
    query_id: str
    asset_id: str
    image_path: str
    leakage_group_id: str
    boundary_group_id: str | None = None
    template_family: str
    generator_batch_id: str
    text: str
    turns: list[ConversationTurn]
    canonical_intent: Intent
    canonical_capability: str | None
    acceptable_capabilities: list[str] = Field(default_factory=list)
    is_boundary: bool = False
    boundary_strategy: BoundaryStrategy | None = None
    requires_card: bool | None
    episode: Episode = "t0"
    split: Split
    label_status: LabelStatus = "auto"
    label_provenance: list[LabelDecision]
    synth_provider: str | None = None
    synth_model: str | None = None
    synthesis_batch_id: str | None = None
    synthesis_prompt_id: str | None = None
    seed_set_sha256: str | None = None

    @field_validator(
        "taxonomy_version",
        "task_spec_version",
        "query_id",
        "asset_id",
        "image_path",
        "leakage_group_id",
        "template_family",
        "generator_batch_id",
        "text",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "boundary_group_id",
        "canonical_capability",
        "synth_provider",
        "synth_model",
        "synthesis_batch_id",
        "synthesis_prompt_id",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("seed_set_sha256")
    @classmethod
    def validate_seed_hash(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("seed_set_sha256 必须是 64 位小写十六进制")
        return value

    @field_validator("acceptable_capabilities")
    @classmethod
    def validate_acceptable_capabilities(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "acceptable_capabilities") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("acceptable_capabilities 不得重复")
        return cleaned

    @model_validator(mode="after")
    def validate_query_contract(self):
        if not self.turns:
            raise ValueError("schema v2 Query.turns 不得为空")
        roles = [turn.role for turn in self.turns]
        if roles not in (["user"], ["user", "assistant", "user"]):
            raise ValueError("turns 只能是 [user] 或 [user, assistant, user]")
        if self.text != self.turns[-1].content:
            raise ValueError("Query.text 必须等于 turns 中最后一个 user content")

        if not self.is_boundary:
            if self.boundary_strategy is not None or self.boundary_group_id is not None:
                raise ValueError("非 boundary 样本不得设置 boundary 字段")
        elif self.boundary_strategy is None:
            raise ValueError("boundary 样本必须设置 boundary_strategy")
        elif self.boundary_strategy in {
            "cross_intent_triplet",
            "cross_intent_pair",
        }:
            if self.boundary_group_id is None:
                raise ValueError(
                    "cross_intent boundary 必须设置 boundary_group_id"
                )
        elif self.boundary_group_id is not None:
            raise ValueError("natural_ambiguity 不得设置 boundary_group_id")

        if self.canonical_capability is None:
            if self.acceptable_capabilities:
                raise ValueError("capability 未解析时 acceptable_capabilities 必须为空")
        elif self.canonical_capability not in self.acceptable_capabilities:
            raise ValueError("canonical_capability 必须包含在 acceptable_capabilities")

        if not self.label_provenance:
            raise ValueError("schema v2 Query 必须包含 label_provenance")
        latest = self.label_provenance[-1]
        if (
            latest.canonical_intent != self.canonical_intent
            or latest.canonical_capability != self.canonical_capability
            or latest.acceptable_capabilities != self.acceptable_capabilities
        ):
            raise ValueError("label_provenance 最后一项必须匹配当前标签")
        expected_decisions = {
            "auto": {"constructed", "migrated_unverified"},
            "cross_agreed": {"cross_review"},
            "arbitrated": {"arbitration"},
        }
        if latest.decision_type not in expected_decisions[self.label_status]:
            raise ValueError(
                "label_status 必须与最新 label_provenance decision_type 一致"
            )

        provenance = (
            self.synth_provider,
            self.synth_model,
            self.synthesis_batch_id,
            self.synthesis_prompt_id,
            self.seed_set_sha256,
        )
        has_any_provenance = any(value is not None for value in provenance)
        has_all_provenance = all(value is not None for value in provenance)
        if has_any_provenance and not has_all_provenance:
            raise ValueError("合成溯源字段必须同时提供")
        return self

    @property
    def gt_intent(self) -> Intent:
        """Read-only Python compatibility; JSON never serializes the legacy name."""

        return self.canonical_intent


class Skill(BaseModel):
    slug: str
    version: int
    description: str  # d —— 仅 Stage 2 可修改
    body: str  # b —— 仅 Stage 3 可修改
    static_refs: list[str] = []  # Cs：相对 Skill 目录的静态资源路径
    operators: list[str] = []  # Od：工具名，必须存在于 tools/registry


class JudgeResult(BaseModel):
    query_id: str
    skill_slug: str | None
    config: str  # noskill / s1 / s1s2 / full
    scores: dict[
        str, int
    ]  # {"TCR":0-10,"CCC":0-10,"CQ":0-20,"CA":0-10}；无卡查询无 CCC 键
    tiers: dict[str, Tier]
    rule_violations: list[dict] = []
    ideal_gaps: list[dict] = []
    skill_md_suggestions: list[str] = []
    raw: str  # Judge 原始输出，永久留档
