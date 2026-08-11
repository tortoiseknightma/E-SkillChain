"""Prepare create-only Portfolio inputs for S1, S2, or S3 evolution.

This script is deliberately not an evolution implementation.  It performs no
model calls, creates no Skill or Bank, and grants no execution authority.

* ``s1`` freezes one complete accepted 25-query batch as public user
  trajectories plus its private capability grouping.
* ``s2`` freezes route attribution from one completed S1 execution shard.
* ``s3`` freezes body attribution from one completed S1+S2 execution shard.

Every output contains exactly ``records.jsonl``, ``manifest.json``, and a
model-consumable ``packet.json`` containing the manifest plus all 25 records.
All files are canonical; every record, manifest, and packet is self-addressed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from scripts.run_portfolio_shard import _assistant_result  # noqa: E402
from skillchain.data.asset_catalog import AssetCatalog  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantQueryInput,
    AssistantRequestSnapshot,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    FINAL_JUDGE_MAX_ATTEMPTS,
    load_final_judge_evaluation_result,
)
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
)
from skillchain.evaluation.packets import (  # noqa: E402
    AssistantToolTrace,
    JudgeDimensionScore,
    RubricSnapshot,
    VisibleCard,
    VisibleToolEvidence,
    build_final_evaluation_packet,
    derive_blinded_evaluation_id,
)
from skillchain.evaluation.portfolio_inputs import (  # noqa: E402
    VerifiedPortfolioDevMiniInputs,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    LoadedPortfolioLaunchPackage,
    PortfolioLaunchInstance,
    PortfolioLaunchShard,
    load_portfolio_launch_package,
)
from skillchain.runners.assistant import (  # noqa: E402
    SharedStage2RouteArtifact,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
EvolutionStage = Literal["s1", "s2", "s3"]
EvolutionComponent = Literal["creator", "route_optimizer", "body_refiner"]

PORTFOLIO_EVOLUTION_INPUT_POLICY_VERSION = "portfolio-evolution-input-v1"
EVOLUTION_INPUT_FILES = ("manifest.json", "packet.json", "records.jsonl")
EXPECTED_RECORD_COUNT = 25
EXPECTED_CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)


def _locked_final_result_contract(runtime_lock: dict[str, object]) -> tuple[int, str]:
    """Resolve explicit active or historical parser-only result contracts."""

    schema = runtime_lock.get("final_judge_result_schema_version")
    cache = runtime_lock.get("final_judge_cache_namespace")
    if isinstance(schema, int) and isinstance(cache, str):
        return schema, cache
    if schema is not None or cache is not None:
        raise PortfolioEvolutionInputError(
            "source runtime has a partial final-Judge result contract"
        )
    parser = runtime_lock.get("final_judge_parser_policy_version")
    parser_sha256 = runtime_lock.get("final_judge_parser_policy_sha256")
    if (
        parser == FINAL_JUDGE_PARSER_POLICY_VERSION_V3
        and parser_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V3
    ):
        return 3, "final-evaluator-v4"
    if (
        parser == FINAL_JUDGE_PARSER_POLICY_VERSION_V2
        and parser_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V2
    ):
        return 2, "final-evaluator-v3"
    if parser is None and parser_sha256 is None:
        return 1, "final-evaluator-v2"
    raise PortfolioEvolutionInputError(
        "source runtime has an unknown historical final-Judge contract"
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_OBJECT_BYTES = 64 * 1024 * 1024


class PortfolioEvolutionInputError(ValueError):
    """An input source or frozen evolution bundle violated its contract."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _hash_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256_bytes(canonical_json_bytes(value))  # type: ignore[arg-type]


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field_name}))


class CapabilityCount(_StrictFrozenModel):
    capability_id: str
    count: int = Field(gt=0)

    @field_validator("capability_id")
    @classmethod
    def validate_capability(cls, value: str) -> str:
        return _nonblank(value, "capability_id")


class S1TrajectoryRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-user-trajectory"] = "portfolio-s1-user-trajectory"
    record_ordinal: int = Field(ge=0, lt=EXPECTED_RECORD_COUNT)
    source_query_ordinal: int = Field(ge=0)
    accepted_batch_id: str
    query_id: str
    canonical_capability: str
    canonical_intent: str
    asset_id: str
    image_path: str
    image_sha256: Sha256
    public_input_json: str
    public_input_sha256: Sha256
    query_sha256: Sha256
    record_sha256: Sha256

    @field_validator(
        "accepted_batch_id",
        "query_id",
        "canonical_capability",
        "canonical_intent",
        "asset_id",
        "image_path",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        try:
            public = parse_canonical_json(
                self.public_input_json.encode("utf-8"),
                label=f"S1 public input {self.query_id}",
            )
        except ArtifactFormatError as error:
            raise ValueError("S1 public input is not canonical JSON") from error
        if (
            not isinstance(public, dict)
            or public.get("asset_id") != self.asset_id
            or public.get("image_path") != self.image_path
            or self.public_input_sha256
            != sha256_bytes(self.public_input_json.encode("utf-8"))
        ):
            raise ValueError("S1 public input identity mismatch")
        if self.record_sha256 != _self_hash(self, "record_sha256"):
            raise ValueError("S1 trajectory record self hash mismatch")
        return self


class S2RouteAttributionRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s2-route-attribution"] = "portfolio-s2-route-attribution"
    record_ordinal: int = Field(ge=0, lt=EXPECTED_RECORD_COUNT)
    source_query_ordinal: int = Field(ge=0)
    accepted_batch_id: str
    query_id: str
    public_input_sha256: Sha256
    query_sha256: Sha256
    expected_capability: str
    selected_capability: str | None
    route_correct: bool
    skill_slug: str | None
    route_trace_sha256: Sha256 | None
    tool_names: tuple[str, ...]
    tool_trace: tuple[AssistantToolTrace, ...]
    assistant_error_code: str | None
    source_bank_sha256: Sha256
    request_sha256: Sha256
    response_sha256: Sha256
    receipt_sha256: Sha256
    assistant_checkpoint_file_sha256: Sha256
    record_sha256: Sha256

    @field_validator(
        "accepted_batch_id",
        "query_id",
        "expected_capability",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "selected_capability",
        "skill_slug",
        "assistant_error_code",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.expected_capability not in EXPECTED_CAPABILITIES:
            raise ValueError("S2 expected capability is outside the frozen universe")
        if self.tool_names != tuple(item.tool_name for item in self.tool_trace):
            raise ValueError("S2 tool-name projection differs from tool trace")
        if self.route_correct != (self.selected_capability == self.expected_capability):
            raise ValueError("S2 route-correct flag differs from attribution")
        route_identity = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if any(value is not None for value in route_identity) and not all(
            value is not None for value in route_identity
        ):
            raise ValueError("S2 route identity must be complete or absent")
        if self.record_sha256 != _self_hash(self, "record_sha256"):
            raise ValueError("S2 route attribution self hash mismatch")
        return self


class S3BodyAttributionRecord(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s3-body-attribution"] = "portfolio-s3-body-attribution"
    record_ordinal: int = Field(ge=0, lt=EXPECTED_RECORD_COUNT)
    source_query_ordinal: int = Field(ge=0)
    accepted_batch_id: str
    query_id: str
    public_input_sha256: Sha256
    query_sha256: Sha256
    canonical_capability: str
    selected_capability: str | None
    skill_slug: str | None
    route_trace_sha256: Sha256 | None
    shared_route_artifact_file_sha256: Sha256 | None
    response_text: str
    visible_cards: tuple[VisibleCard, ...]
    visible_tool_evidence: tuple[VisibleToolEvidence, ...]
    tool_trace: tuple[AssistantToolTrace, ...]
    assistant_error_code: str | None
    source_bank_sha256: Sha256
    request_sha256: Sha256
    response_sha256: Sha256
    receipt_sha256: Sha256
    assistant_checkpoint_file_sha256: Sha256
    final_kind: Literal["visual_final_judge", "assistant_fixed_zero"]
    evaluation_id: Sha256 | None
    judge_status: str
    judge_error_code: str | None
    dimensions: tuple[JudgeDimensionScore, ...]
    j_project: float = Field(ge=0.0, le=100.0)
    final_result_sha256: Sha256
    final_checkpoint_file_sha256: Sha256
    record_sha256: Sha256

    @field_validator(
        "accepted_batch_id",
        "query_id",
        "canonical_capability",
        "judge_status",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "selected_capability",
        "skill_slug",
        "assistant_error_code",
        "judge_error_code",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.canonical_capability not in EXPECTED_CAPABILITIES:
            raise ValueError("S3 capability is outside the frozen universe")
        route_identity = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if any(value is not None for value in route_identity) and not all(
            value is not None for value in route_identity
        ):
            raise ValueError("S3 route identity must be complete or absent")
        if self.final_kind == "assistant_fixed_zero":
            if (
                self.assistant_error_code is None
                or self.evaluation_id is not None
                or self.judge_status != "not_invoked_assistant_error"
                or self.dimensions
                or self.j_project != 0.0
            ):
                raise ValueError("S3 fixed-zero attribution is inconsistent")
        elif (
            self.assistant_error_code is not None
            or self.evaluation_id is None
            or not self.dimensions
        ):
            raise ValueError("S3 visual-Judge attribution is incomplete")
        if self.record_sha256 != _self_hash(self, "record_sha256"):
            raise ValueError("S3 body attribution self hash mismatch")
        return self


EvolutionRecord = Annotated[
    S1TrajectoryRecord | S2RouteAttributionRecord | S3BodyAttributionRecord,
    Field(discriminator="kind"),
]


class PortfolioEvolutionInputManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-evolution-input-manifest"] = (
        "portfolio-evolution-input-manifest"
    )
    policy_version: Literal["portfolio-evolution-input-v1"] = (
        PORTFOLIO_EVOLUTION_INPUT_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    stage: EvolutionStage
    component: EvolutionComponent
    status: Literal["prepared_input_only"] = "prepared_input_only"
    source_split: Literal["dev_mini"] = "dev_mini"
    accepted_batch_id: str
    record_kind: Literal[
        "portfolio-s1-user-trajectory",
        "portfolio-s2-route-attribution",
        "portfolio-s3-body-attribution",
    ]
    record_count: Literal[25] = EXPECTED_RECORD_COUNT
    query_ids: tuple[str, ...]
    query_order_sha256: Sha256
    capability_counts: tuple[CapabilityCount, ...]
    records_file: Literal["records.jsonl"] = "records.jsonl"
    records_file_sha256: Sha256
    portfolio_plan_sha256: Sha256
    accepted_ledger_sha256: Sha256
    query_artifact_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed_set_sha256: Sha256
    runtime_catalog_sha256: Sha256
    source_config: Literal["s1", "s1s2"] | None = None
    source_execution_control_file_sha256: Sha256 | None = None
    source_execution_control_sha256: Sha256 | None = None
    source_launch_plan_file_sha256: Sha256 | None = None
    source_launch_plan_sha256: Sha256 | None = None
    source_runtime_lock_file_sha256: Sha256 | None = None
    source_runtime_lock_sha256: Sha256 | None = None
    source_shard_id: str | None = None
    source_shard_sha256: Sha256 | None = None
    source_shard_summary_file_sha256: Sha256 | None = None
    source_shard_summary_sha256: Sha256 | None = None
    source_shard_audit_file_sha256: Sha256 | None = None
    source_shard_audit_sha256: Sha256 | None = None
    source_bank_sha256: Sha256 | None = None
    model_calls_performed: Literal[0] = 0
    evolution_component_invoked: Literal[False] = False
    bank_generated: Literal[False] = False
    execution_authorized: Literal[False] = False
    manifest_sha256: Sha256

    @field_validator("query_ids", "capability_counts", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @field_validator("accepted_batch_id")
    @classmethod
    def validate_batch(cls, value: str) -> str:
        return _nonblank(value, "accepted_batch_id")

    @field_validator("source_shard_id")
    @classmethod
    def validate_optional_shard(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "source_shard_id")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected = {
            "s1": ("creator", "portfolio-s1-user-trajectory", None),
            "s2": ("route_optimizer", "portfolio-s2-route-attribution", "s1"),
            "s3": ("body_refiner", "portfolio-s3-body-attribution", "s1s2"),
        }[self.stage]
        if (
            self.component,
            self.record_kind,
            self.source_config,
        ) != expected:
            raise ValueError("evolution stage/component/source configuration mismatch")
        if (
            len(self.query_ids) != EXPECTED_RECORD_COUNT
            or len(set(self.query_ids)) != EXPECTED_RECORD_COUNT
            or self.query_order_sha256 != _hash_payload(list(self.query_ids))
        ):
            raise ValueError("evolution manifest query order is invalid")
        observed_capabilities = tuple(
            item.capability_id for item in self.capability_counts
        )
        if (
            observed_capabilities != EXPECTED_CAPABILITIES
            or sum(item.count for item in self.capability_counts)
            != EXPECTED_RECORD_COUNT
        ):
            raise ValueError(
                "evolution input must cover all six capabilities in canonical order"
            )
        source_fields = (
            self.source_execution_control_file_sha256,
            self.source_execution_control_sha256,
            self.source_launch_plan_file_sha256,
            self.source_launch_plan_sha256,
            self.source_runtime_lock_file_sha256,
            self.source_runtime_lock_sha256,
            self.source_shard_id,
            self.source_shard_sha256,
            self.source_shard_summary_file_sha256,
            self.source_shard_summary_sha256,
            self.source_shard_audit_file_sha256,
            self.source_shard_audit_sha256,
            self.source_bank_sha256,
        )
        if self.stage == "s1":
            if any(value is not None for value in source_fields):
                raise ValueError(
                    "S1 trajectories cannot claim execution-shard provenance"
                )
        elif any(value is None for value in source_fields):
            raise ValueError("S2/S3 attribution lacks execution-shard provenance")
        if self.manifest_sha256 != _self_hash(self, "manifest_sha256"):
            raise ValueError("evolution input manifest self hash mismatch")
        return self


class PortfolioEvolutionInputPacket(_StrictFrozenModel):
    manifest: PortfolioEvolutionInputManifest
    records: tuple[EvolutionRecord, ...]
    packet_sha256: Sha256

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if (
            len(self.records) != EXPECTED_RECORD_COUNT
            or tuple(item.query_id for item in self.records) != self.manifest.query_ids
            or any(item.kind != self.manifest.record_kind for item in self.records)
        ):
            raise ValueError("evolution packet differs from its manifest")
        if self.packet_sha256 != _self_hash(self, "packet_sha256"):
            raise ValueError("evolution packet self hash mismatch")
        return self


@dataclass(frozen=True)
class CreatedEvolutionInputBundle:
    root: Path
    records_path: Path
    manifest_path: Path
    packet_path: Path
    manifest: PortfolioEvolutionInputManifest
    packet: PortfolioEvolutionInputPacket
    records: tuple[EvolutionRecord, ...]
    records_file_sha256: str
    manifest_file_sha256: str
    packet_file_sha256: str
    packet_sha256: str


@dataclass(frozen=True)
class _AssistantCheckpoint:
    member: PortfolioLaunchInstance
    request: AssistantRequestSnapshot
    response: AssistantBackendResponse
    receipt: AssistantExecutionReceipt
    checkpoint_file_sha256: str
    shared_route_file_sha256: str | None


@dataclass(frozen=True)
class _SourceShardContext:
    inputs: VerifiedPortfolioDevMiniInputs
    execution_root: Path
    control: dict[str, object]
    control_file_sha256: str
    control_sha256: str
    launch: LoadedPortfolioLaunchPackage
    runtime_lock: dict[str, object]
    runtime_lock_file_sha256: str
    runtime_lock_sha256: str
    bank: StaticBankArtifact
    bank_file_sha256: str
    shard: PortfolioLaunchShard
    members: tuple[PortfolioLaunchInstance, ...]
    shard_root: Path
    summary: dict[str, object]
    summary_file_sha256: str
    summary_sha256: str
    audit: dict[str, object]
    audit_file_sha256: str
    audit_sha256: str
    checkpoints: tuple[_AssistantCheckpoint, ...]


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PortfolioEvolutionInputError(f"{label} is not a SHA-256")
    return value


def _canonical_object(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_OBJECT_BYTES,
) -> tuple[bytes, dict[str, object]]:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise PortfolioEvolutionInputError(f"{label} must contain an object")
    return content, value


def _require_self_hash(
    value: dict[str, object],
    field_name: str,
    *,
    label: str,
) -> str:
    supplied = _require_sha256(value.get(field_name), f"{label} {field_name}")
    unsigned = dict(value)
    unsigned.pop(field_name, None)
    if supplied != _hash_payload(unsigned):
        raise PortfolioEvolutionInputError(f"{label} self hash mismatch")
    return supplied


def _capability_counts(
    values: list[str] | tuple[str, ...],
) -> tuple[CapabilityCount, ...]:
    counts = Counter(values)
    if set(counts) != set(EXPECTED_CAPABILITIES):
        raise PortfolioEvolutionInputError(
            "selected 25-query input does not cover all six capabilities"
        )
    return tuple(
        CapabilityCount(capability_id=capability, count=counts[capability])
        for capability in EXPECTED_CAPABILITIES
    )


def _record_model(stage: EvolutionStage) -> type[BaseModel]:
    return {
        "s1": S1TrajectoryRecord,
        "s2": S2RouteAttributionRecord,
        "s3": S3BodyAttributionRecord,
    }[stage]


def load_prepared_evolution_inputs(
    root: str | Path,
    *,
    expected_manifest_file_sha256: str,
) -> CreatedEvolutionInputBundle:
    """Reload one externally pinned two-file evolution input bundle."""

    destination = Path(root)
    expected_manifest_file_sha256 = _require_sha256(
        expected_manifest_file_sha256,
        "expected evolution manifest file SHA-256",
    )
    if not destination.is_dir() or destination.is_symlink():
        raise PortfolioEvolutionInputError(
            "evolution input root must be a real directory"
        )
    actual_files = {
        item.name
        for item in destination.iterdir()
        if item.is_file() and not item.is_symlink()
    }
    if actual_files != set(EVOLUTION_INPUT_FILES) or any(
        item.is_dir() or item.is_symlink() for item in destination.iterdir()
    ):
        raise PortfolioEvolutionInputError(
            "evolution input bundle must contain exactly manifest.json and records.jsonl"
        )
    manifest_path = destination / "manifest.json"
    manifest_bytes, manifest_raw = _canonical_object(
        manifest_path,
        label="evolution input manifest",
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise PortfolioEvolutionInputError(
            "evolution manifest external file digest mismatch"
        )
    try:
        manifest = PortfolioEvolutionInputManifest.model_validate(
            manifest_raw,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioEvolutionInputError(
            "evolution input manifest violates schema"
        ) from error

    records_path = destination / manifest.records_file
    records_bytes = read_stable_regular_file(
        records_path,
        label="evolution input records",
        max_bytes=_MAX_OBJECT_BYTES,
    )
    if sha256_bytes(records_bytes) != manifest.records_file_sha256:
        raise PortfolioEvolutionInputError("evolution records file digest mismatch")
    raw_records = parse_canonical_jsonl(
        records_bytes,
        label="evolution input records",
    )
    model_type = _record_model(manifest.stage)
    try:
        records = tuple(
            model_type.model_validate_json(
                canonical_json_bytes(value),
                strict=True,
            )
            for value in raw_records
        )
    except ValidationError as error:
        raise PortfolioEvolutionInputError(
            "evolution input record violates schema"
        ) from error
    if (
        len(records) != EXPECTED_RECORD_COUNT
        or tuple(item.record_ordinal for item in records)
        != tuple(range(EXPECTED_RECORD_COUNT))
        or tuple(item.query_id for item in records) != manifest.query_ids
    ):
        raise PortfolioEvolutionInputError(
            "evolution records are not the exact manifest-bound 25 rows"
        )
    capabilities = tuple(
        (
            item.canonical_capability
            if isinstance(item, (S1TrajectoryRecord, S3BodyAttributionRecord))
            else item.expected_capability
        )
        for item in records
    )
    if _capability_counts(capabilities) != manifest.capability_counts:
        raise PortfolioEvolutionInputError(
            "evolution record capability coverage differs from manifest"
        )

    packet_path = destination / "packet.json"
    packet_bytes = read_stable_regular_file(
        packet_path,
        label="evolution stage-input packet",
        max_bytes=_MAX_OBJECT_BYTES,
    )
    try:
        packet = PortfolioEvolutionInputPacket.model_validate_json(
            packet_bytes,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioEvolutionInputError(
            "evolution stage-input packet violates schema"
        ) from error
    if canonical_json_bytes(packet) != packet_bytes:
        raise PortfolioEvolutionInputError(
            "evolution stage-input packet is not canonical"
        )
    if packet.manifest != manifest or packet.records != records:
        raise PortfolioEvolutionInputError(
            "evolution stage-input packet differs from manifest or records"
        )
    return CreatedEvolutionInputBundle(
        root=destination,
        records_path=records_path,
        manifest_path=manifest_path,
        packet_path=packet_path,
        manifest=manifest,
        packet=packet,
        records=records,  # type: ignore[arg-type]
        records_file_sha256=manifest.records_file_sha256,
        manifest_file_sha256=expected_manifest_file_sha256,
        packet_file_sha256=sha256_bytes(packet_bytes),
        packet_sha256=packet.packet_sha256,
    )


def _publish_bundle(
    output_dir: Path,
    *,
    records: tuple[EvolutionRecord, ...],
    manifest_payload: dict[str, object],
) -> CreatedEvolutionInputBundle:
    if os.path.lexists(output_dir):
        raise FileExistsError(f"evolution input output is create-only: {output_dir}")
    records_bytes = canonical_jsonl_bytes(records)
    payload = {
        **manifest_payload,
        "record_count": EXPECTED_RECORD_COUNT,
        "query_ids": [item.query_id for item in records],
        "query_order_sha256": _hash_payload([item.query_id for item in records]),
        "capability_counts": [
            item.model_dump(mode="json")
            for item in _capability_counts(
                tuple(
                    (
                        item.canonical_capability
                        if isinstance(
                            item,
                            (S1TrajectoryRecord, S3BodyAttributionRecord),
                        )
                        else item.expected_capability
                    )
                    for item in records
                )
            )
        ],
        "records_file": "records.jsonl",
        "records_file_sha256": sha256_bytes(records_bytes),
        "model_calls_performed": 0,
        "evolution_component_invoked": False,
        "bank_generated": False,
        "execution_authorized": False,
    }
    manifest = PortfolioEvolutionInputManifest.model_validate(
        {
            **payload,
            "manifest_sha256": _hash_payload(payload),
        },
        strict=True,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    packet_payload = {
        "manifest": manifest.model_dump(mode="json"),
        "records": [item.model_dump(mode="json") for item in records],
    }
    packet = PortfolioEvolutionInputPacket.model_validate_json(
        canonical_json_bytes(
            {
                **packet_payload,
                "packet_sha256": _hash_payload(packet_payload),
            }
        ),
        strict=True,
    )
    packet_bytes = canonical_json_bytes(packet)
    staging = new_staging_directory(output_dir)
    try:
        atomic_create_file(staging / "records.jsonl", records_bytes)
        atomic_create_file(staging / "manifest.json", manifest_bytes)
        atomic_create_file(staging / "packet.json", packet_bytes)
        verified = load_prepared_evolution_inputs(
            staging,
            expected_manifest_file_sha256=sha256_bytes(manifest_bytes),
        )
        atomic_publish_new_directory(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return CreatedEvolutionInputBundle(
        root=output_dir,
        records_path=output_dir / "records.jsonl",
        manifest_path=output_dir / "manifest.json",
        packet_path=output_dir / "packet.json",
        manifest=verified.manifest,
        packet=verified.packet,
        records=verified.records,
        records_file_sha256=verified.records_file_sha256,
        manifest_file_sha256=verified.manifest_file_sha256,
        packet_file_sha256=verified.packet_file_sha256,
        packet_sha256=verified.packet_sha256,
    )


def _common_manifest_payload(
    inputs: VerifiedPortfolioDevMiniInputs,
    *,
    stage: EvolutionStage,
    accepted_batch_id: str,
) -> dict[str, object]:
    component, record_kind = {
        "s1": ("creator", "portfolio-s1-user-trajectory"),
        "s2": ("route_optimizer", "portfolio-s2-route-attribution"),
        "s3": ("body_refiner", "portfolio-s3-body-attribution"),
    }[stage]
    return {
        "schema_version": 1,
        "kind": "portfolio-evolution-input-manifest",
        "policy_version": PORTFOLIO_EVOLUTION_INPUT_POLICY_VERSION,
        "track": "portfolio",
        "formal_eligible": False,
        "stage": stage,
        "component": component,
        "status": "prepared_input_only",
        "source_split": "dev_mini",
        "accepted_batch_id": accepted_batch_id,
        "record_kind": record_kind,
        "portfolio_plan_sha256": inputs.expected_plan_sha256,
        "accepted_ledger_sha256": inputs.expected_accepted_ledger_sha256,
        "query_artifact_sha256": inputs.expected_query_artifact_sha256,
        "capability_assignments_sha256": (
            inputs.expected_capability_assignments_sha256
        ),
        "seed_set_sha256": inputs.expected_seed_set_sha256,
        "runtime_catalog_sha256": inputs.expected_output_catalog_sha256,
    }


def prepare_s1_inputs(
    *,
    accepted_batch_id: str,
    output_dir: str | Path,
) -> CreatedEvolutionInputBundle:
    """Freeze one complete accepted batch for a later, separate S1 Creator."""

    accepted_batch_id = _nonblank(accepted_batch_id, "accepted_batch_id")
    inputs = _load_active_inputs()
    matching_ledger = tuple(
        item for item in inputs.ledger if item.batch_id == accepted_batch_id
    )
    if len(matching_ledger) != 1 or matching_ledger[0].count != EXPECTED_RECORD_COUNT:
        raise PortfolioEvolutionInputError(
            "S1 accepted_batch_id is not one exact accepted 25-query batch"
        )
    selected = tuple(
        query
        for query in inputs.queries
        if query.synthesis_batch_id == accepted_batch_id
    )
    if len(selected) != EXPECTED_RECORD_COUNT:
        raise PortfolioEvolutionInputError(
            "S1 accepted batch does not contain exactly 25 verified queries"
        )
    assistant_by_id = {item.query_id: item for item in inputs.assistant_queries}
    asset_by_id = {item.query_id: item for item in inputs.query_assets}
    global_ordinal = {
        query.query_id: ordinal for ordinal, query in enumerate(inputs.queries)
    }
    records: list[S1TrajectoryRecord] = []
    for record_ordinal, query in enumerate(selected):
        if query.canonical_capability is None:
            raise PortfolioEvolutionInputError(
                f"S1 query lacks canonical capability: {query.query_id}"
            )
        public = assistant_by_id.get(query.query_id)
        asset = asset_by_id.get(query.query_id)
        if (
            public is None
            or asset is None
            or asset.asset_id != query.asset_id
            or asset.image_path != query.image_path
        ):
            raise PortfolioEvolutionInputError(
                f"S1 public/image binding is incomplete: {query.query_id}"
            )
        payload = {
            "schema_version": 1,
            "kind": "portfolio-s1-user-trajectory",
            "record_ordinal": record_ordinal,
            "source_query_ordinal": global_ordinal[query.query_id],
            "accepted_batch_id": accepted_batch_id,
            "query_id": query.query_id,
            "canonical_capability": query.canonical_capability,
            "canonical_intent": query.canonical_intent,
            "asset_id": query.asset_id,
            "image_path": query.image_path,
            "image_sha256": asset.image_sha256,
            "public_input_json": public.public_input_json,
            "public_input_sha256": public.public_input_sha256,
            "query_sha256": public.query_sha256,
        }
        records.append(
            S1TrajectoryRecord.model_validate_json(
                canonical_json_bytes(
                    {**payload, "record_sha256": _hash_payload(payload)}
                ),
                strict=True,
            )
        )
    _capability_counts(tuple(item.canonical_capability for item in records))
    payload = {
        **_common_manifest_payload(
            inputs,
            stage="s1",
            accepted_batch_id=accepted_batch_id,
        ),
        "source_config": None,
        "source_execution_control_file_sha256": None,
        "source_execution_control_sha256": None,
        "source_launch_plan_file_sha256": None,
        "source_launch_plan_sha256": None,
        "source_runtime_lock_file_sha256": None,
        "source_runtime_lock_sha256": None,
        "source_shard_id": None,
        "source_shard_sha256": None,
        "source_shard_summary_file_sha256": None,
        "source_shard_summary_sha256": None,
        "source_shard_audit_file_sha256": None,
        "source_shard_audit_sha256": None,
        "source_bank_sha256": None,
    }
    return _publish_bundle(
        Path(output_dir),
        records=tuple(records),
        manifest_payload=payload,
    )


def _assert_exact_checkpoint_files(
    directory: Path,
    *,
    query_ids: tuple[str, ...],
    label: str,
) -> None:
    expected = {f"{query_id}.json" for query_id in query_ids}
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioEvolutionInputError(f"{label} directory is missing or unsafe")
    observed: set[str] = set()
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file():
            raise PortfolioEvolutionInputError(
                f"{label} directory contains a non-regular entry"
            )
        observed.add(item.name)
    if observed != expected:
        raise PortfolioEvolutionInputError(
            f"{label} files are not the exact source shard query set"
        )


def _validate_launch_artifact_binding(
    launch: LoadedPortfolioLaunchPackage,
    *,
    artifact_id: str,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> None:
    matches = tuple(
        item for item in launch.plan.artifacts if item.artifact_id == artifact_id
    )
    if (
        len(matches) != 1
        or matches[0].status != "verified"
        or matches[0].file_sha256 != expected_file_sha256
        or matches[0].content_sha256 != expected_content_sha256
    ):
        raise PortfolioEvolutionInputError(
            f"source launch artifact binding differs: {artifact_id}"
        )


def _load_assistant_checkpoint(
    context: _SourceShardContext | None,
    *,
    execution_root: Path,
    accepted_batch_id: str,
    member: PortfolioLaunchInstance,
    expected_config: Literal["s1", "s1s2"],
    expected_bank_sha256: str,
    expected_query: AssistantQueryInput,
    expected_image_sha256: str,
) -> _AssistantCheckpoint:
    del context
    path = execution_root / member.assistant_output_relpath
    content, raw = _canonical_object(
        path,
        label=f"source Assistant checkpoint {member.query_id}",
    )
    expected_outer_fields = {
        "schema_version",
        "kind",
        "instance_sha256",
        "query_ordinal",
        "request",
        "response",
        "receipt",
        "row_sha256",
    }
    supplied_row_sha256 = raw.get("row_sha256")
    unsigned = dict(raw)
    unsigned.pop("row_sha256", None)
    if (
        set(raw) != expected_outer_fields
        or raw.get("schema_version") != 1
        or raw.get("kind") != "portfolio-assistant-checkpoint"
        or supplied_row_sha256 != _hash_payload(unsigned)
    ):
        raise PortfolioEvolutionInputError(
            f"Assistant checkpoint envelope is invalid: {member.query_id}"
        )
    try:
        request = AssistantRequestSnapshot.model_validate_json(
            canonical_json_bytes(raw["request"]),
            strict=True,
        )
        response = AssistantBackendResponse.model_validate_json(
            canonical_json_bytes(raw["response"]),
            strict=True,
        )
        receipt = AssistantExecutionReceipt.model_validate_json(
            canonical_json_bytes(raw["receipt"]),
            strict=True,
        )
    except (KeyError, ValidationError) as error:
        raise PortfolioEvolutionInputError(
            f"Assistant checkpoint schema is invalid: {member.query_id}"
        ) from error
    if (
        raw.get("instance_sha256") != member.instance_sha256
        or raw.get("query_ordinal") != member.query_ordinal
        or request.matrix_run_id != member.matrix_run_id
        or request.query_ordinal != member.query_ordinal
        or request.config != expected_config
        or request.query != expected_query
        or request.query.query_id != member.query_id
        or request.query.query_sha256 != member.query_sha256
        or request.query.public_input_sha256 != member.public_input_sha256
        or request.treatment.config != expected_config
        or request.treatment.bank_sha256 != expected_bank_sha256
        or response.request_sha256 != request.request_sha256
        or receipt.request_sha256 != request.request_sha256
        or receipt.response_sha256 != _hash_payload(response.model_dump(mode="json"))
        or receipt.aggregate_usage != response.usage
        or receipt.tool_trace != response.tool_trace
        or receipt.asset_catalog_sha256 is None
        or receipt.query_asset_id != member.asset_id
        or receipt.query_asset_sha256 != expected_image_sha256
        or response.backbone_identity_sha256 != request.backbone.identity_sha256
        or response.registry_sha256 != request.registry.registry_sha256
        or response.registry_runtime_sha256 != request.registry.registry_runtime_sha256
        or response.budget_sha256 != request.budget.budget_sha256
    ):
        raise PortfolioEvolutionInputError(
            f"Assistant checkpoint binding differs: {member.query_id}"
        )

    shared_route_file_sha256: str | None = None
    if expected_config == "s1":
        if receipt.shared_route_reference is not None:
            raise PortfolioEvolutionInputError(
                f"S1 checkpoint unexpectedly references shared route: {member.query_id}"
            )
    else:
        reference = receipt.shared_route_reference
        if reference is None:
            raise PortfolioEvolutionInputError(
                f"S1+S2 checkpoint lacks shared route: {member.query_id}"
            )
        route_path = (
            execution_root
            / "shared-routes"
            / accepted_batch_id
            / f"{member.query_id}.json"
        )
        route_content = read_stable_regular_file(
            route_path,
            label=f"shared Stage-2 route {member.query_id}",
            max_bytes=4 * 1024 * 1024,
        )
        try:
            route = SharedStage2RouteArtifact.model_validate_json(
                route_content,
                strict=True,
            )
        except ValidationError as error:
            raise PortfolioEvolutionInputError(
                f"shared Stage-2 route is invalid: {member.query_id}"
            ) from error
        if canonical_json_bytes(route) != route_content:
            raise PortfolioEvolutionInputError(
                f"shared Stage-2 route is not canonical: {member.query_id}"
            )
        if (
            route.matrix_run_id != request.matrix_run_id
            or route.query_id != member.query_id
            or route.query_ordinal != member.query_ordinal
            or route.query_sha256 != expected_query.query_sha256
            or route.public_input_sha256 != expected_query.public_input_sha256
            or route.artifact_sha256 != reference.artifact_sha256
            or route.route_call.response_sha256 != reference.route_call_response_sha256
            or route.route_call.input_tokens != reference.reserved_usage.input_tokens
            or route.route_call.output_tokens != reference.reserved_usage.output_tokens
            or route.artifact_sha256 != response.route_trace_sha256
            or route.selected_capability != response.selected_capability
            or route.capability_ids != EXPECTED_CAPABILITIES
        ):
            raise PortfolioEvolutionInputError(
                f"shared Stage-2 route binding differs: {member.query_id}"
            )
        shared_route_file_sha256 = sha256_bytes(route_content)

    return _AssistantCheckpoint(
        member=member,
        request=request,
        response=response,
        receipt=receipt,
        checkpoint_file_sha256=sha256_bytes(content),
        shared_route_file_sha256=shared_route_file_sha256,
    )


def _validate_assistant_audit(
    context: _SourceShardContext,
) -> None:
    errors = [
        item.member.query_id
        for item in context.checkpoints
        if item.response.error_code is not None
    ]
    if (
        context.audit.get("assistant_error_ids") != errors
        or context.audit.get("assistant_error_count") != len(errors)
        or context.audit.get("assistant_success_count")
        != EXPECTED_RECORD_COUNT - len(errors)
        or context.audit.get("assistant_model_calls")
        != sum(len(item.receipt.model_calls) for item in context.checkpoints)
        or context.audit.get("assistant_input_tokens")
        != sum(item.response.usage.input_tokens for item in context.checkpoints)
        or context.audit.get("assistant_output_tokens")
        != sum(item.response.usage.output_tokens for item in context.checkpoints)
        or context.audit.get("tool_call_count")
        != sum(len(item.response.tool_trace) for item in context.checkpoints)
    ):
        raise PortfolioEvolutionInputError(
            "source shard audit differs from Assistant checkpoints"
        )


def _load_source_shard(
    *,
    execution_root: str | Path,
    source_shard_id: str,
    expected_source_shard_audit_sha256: str,
    expected_config: Literal["s1", "s1s2"],
) -> _SourceShardContext:
    source_shard_id = _nonblank(source_shard_id, "source_shard_id")
    expected_source_shard_audit_sha256 = _require_sha256(
        expected_source_shard_audit_sha256,
        "expected source shard audit SHA-256",
    )
    root = Path(execution_root)
    if not root.is_dir() or root.is_symlink():
        raise PortfolioEvolutionInputError(
            "source execution root must be a real directory"
        )
    control_path = root / "execution-control.json"
    control_bytes, control = _canonical_object(
        control_path,
        label="source execution control",
    )
    control_sha256 = _require_self_hash(
        control,
        "control_sha256",
        label="source execution control",
    )
    if (
        control.get("kind") != "portfolio-matrix-execution-control"
        or control.get("track") != "portfolio"
        or control.get("formal_eligible") is not False
    ):
        raise PortfolioEvolutionInputError(
            "source execution control is not a Portfolio matrix control"
        )
    authorized = control.get("authorized_shard_ids")
    if (
        not isinstance(authorized, list)
        or source_shard_id not in authorized
        or any(not isinstance(value, str) for value in authorized)
    ):
        raise PortfolioEvolutionInputError(
            "source shard is not authorized by execution control"
        )
    key_content = read_stable_regular_file(
        root / "blinding-key.bin",
        label="source execution blinding key",
        max_bytes=32,
    )
    if len(key_content) != 32 or sha256_bytes(key_content) != control.get(
        "blinding_key_sha256"
    ):
        raise PortfolioEvolutionInputError(
            "source execution blinding key binding mismatch"
        )

    launch_root_value = control.get("launch_root")
    launch_file_sha256 = _require_sha256(
        control.get("launch_plan_file_sha256"),
        "source launch plan file SHA-256",
    )
    if not isinstance(launch_root_value, str):
        raise PortfolioEvolutionInputError("source launch root is absent")
    launch = load_portfolio_launch_package(
        Path(launch_root_value),
        expected_plan_file_sha256=launch_file_sha256,
    )
    if launch.plan.launch_plan_sha256 != control.get("launch_plan_sha256"):
        raise PortfolioEvolutionInputError(
            "source launch plan differs from execution control"
        )
    shards = tuple(
        item for item in launch.plan.shards if item.shard_id == source_shard_id
    )
    if len(shards) != 1:
        raise PortfolioEvolutionInputError("source shard is absent from launch")
    shard = shards[0]
    if (
        shard.config != expected_config
        or shard.query_count != EXPECTED_RECORD_COUNT
        or len(shard.query_ids) != EXPECTED_RECORD_COUNT
    ):
        raise PortfolioEvolutionInputError(
            f"source shard must be one complete {expected_config} 25-query shard"
        )
    members = tuple(
        item for item in launch.instances if item.shard_id == source_shard_id
    )
    if (
        len(members) != EXPECTED_RECORD_COUNT
        or tuple(item.query_id for item in members) != shard.query_ids
        or tuple(item.instance_sha256 for item in members) != shard.instance_sha256s
        or any(
            item.config != expected_config
            or item.accepted_batch_id != shard.accepted_batch_id
            for item in members
        )
    ):
        raise PortfolioEvolutionInputError(
            "source launch members differ from shard descriptor"
        )

    runtime_root_value = control.get("runtime_root")
    runtime_lock_file_sha256 = _require_sha256(
        control.get("runtime_lock_file_sha256"),
        "source runtime lock file SHA-256",
    )
    if not isinstance(runtime_root_value, str):
        raise PortfolioEvolutionInputError("source runtime root is absent")
    runtime_root = Path(runtime_root_value)
    runtime_lock_path = runtime_root / "runtime-lock.json"
    runtime_lock_bytes, runtime_lock = _canonical_object(
        runtime_lock_path,
        label="source runtime lock",
    )
    if sha256_bytes(runtime_lock_bytes) != runtime_lock_file_sha256:
        raise PortfolioEvolutionInputError("source runtime lock file digest mismatch")
    runtime_lock_sha256 = _require_self_hash(
        runtime_lock,
        "runtime_lock_sha256",
        label="source runtime lock",
    )
    if runtime_lock_sha256 != control.get("runtime_lock_sha256"):
        raise PortfolioEvolutionInputError(
            "source runtime lock differs from execution control"
        )
    _validate_launch_artifact_binding(
        launch,
        artifact_id="assistant_runtime_lock",
        expected_file_sha256=runtime_lock_file_sha256,
        expected_content_sha256=runtime_lock_sha256,
    )

    locked_banks = runtime_lock.get("bank_sha256s")
    if not isinstance(locked_banks, dict):
        raise PortfolioEvolutionInputError("source runtime lock lacks Bank identities")
    expected_bank_sha256 = _require_sha256(
        locked_banks.get(expected_config),
        f"source {expected_config} Bank SHA-256",
    )
    bank_path = runtime_root / f"bank-{expected_config}.json"
    bank_content = read_stable_regular_file(
        bank_path,
        label=f"source {expected_config} Bank",
        max_bytes=16 * 1024 * 1024,
    )
    try:
        bank = StaticBankArtifact.model_validate_json(
            bank_content,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioEvolutionInputError(
            f"source {expected_config} Bank violates schema"
        ) from error
    bank_file_sha256 = sha256_bytes(bank_content)
    if (
        canonical_json_bytes(bank) != bank_content
        or bank.bank_sha256 != expected_bank_sha256
        or bank.tool_registry_sha256 != runtime_lock.get("tool_registry_sha256")
        or bank.tool_registry_runtime_sha256
        != runtime_lock.get("tool_registry_runtime_sha256")
    ):
        raise PortfolioEvolutionInputError(
            f"source {expected_config} Bank binding mismatch"
        )
    _validate_launch_artifact_binding(
        launch,
        artifact_id=f"bank.{expected_config}",
        expected_file_sha256=bank_file_sha256,
        expected_content_sha256=bank.bank_sha256,
    )

    inputs = _load_active_inputs()
    private_by_id = {item.query_id: item for item in inputs.queries}
    public_by_id = {item.query_id: item for item in inputs.assistant_queries}
    asset_by_id = {item.query_id: item for item in inputs.query_assets}
    matching_ledger = tuple(
        item for item in inputs.ledger if item.batch_id == shard.accepted_batch_id
    )
    if len(matching_ledger) != 1 or matching_ledger[0].count != EXPECTED_RECORD_COUNT:
        raise PortfolioEvolutionInputError(
            "source shard accepted batch is not in the verified ledger"
        )
    capabilities: list[str] = []
    for member in members:
        private = private_by_id.get(member.query_id)
        public = public_by_id.get(member.query_id)
        asset = asset_by_id.get(member.query_id)
        if (
            private is None
            or public is None
            or asset is None
            or private.synthesis_batch_id != shard.accepted_batch_id
            or private.canonical_capability is None
            or member.query_sha256 != public.query_sha256
            or member.public_input_sha256 != public.public_input_sha256
            or member.asset_id != private.asset_id
            or member.image_path != private.image_path
            or member.image_sha256 != asset.image_sha256
        ):
            raise PortfolioEvolutionInputError(
                f"source shard query/private/public binding differs: {member.query_id}"
            )
        capabilities.append(private.canonical_capability)
    _capability_counts(capabilities)

    shard_root = root / shard.output_relpath
    summary_path = shard_root / "shard-summary.json"
    summary_bytes, summary = _canonical_object(
        summary_path,
        label="source shard summary",
    )
    summary_sha256 = _require_self_hash(
        summary,
        "summary_sha256",
        label="source shard summary",
    )
    audit_path = shard_root / "shard-audit.json"
    audit_bytes, audit = _canonical_object(
        audit_path,
        label="source shard audit",
    )
    audit_sha256 = _require_self_hash(
        audit,
        "audit_sha256",
        label="source shard audit",
    )
    if audit_sha256 != expected_source_shard_audit_sha256:
        raise PortfolioEvolutionInputError(
            "source shard audit differs from the caller-pinned self digest"
        )
    if (
        summary.get("kind") != "portfolio-shard-summary"
        or summary.get("status") != "complete"
        or summary.get("shard_id") != shard.shard_id
        or summary.get("config") != expected_config
        or summary.get("query_count") != EXPECTED_RECORD_COUNT
        or audit.get("kind") != "portfolio-shard-audit"
        or audit.get("shard_id") != shard.shard_id
        or audit.get("config") != expected_config
        or audit.get("query_count") != EXPECTED_RECORD_COUNT
        or audit.get("shard_summary_sha256") != summary_sha256
        or audit.get("launch_plan_sha256") != launch.plan.launch_plan_sha256
        or audit.get("runtime_lock_sha256") != runtime_lock_sha256
    ):
        raise PortfolioEvolutionInputError(
            "source shard summary/audit binding mismatch"
        )

    query_ids = tuple(item.query_id for item in members)
    _assert_exact_checkpoint_files(
        shard_root / "assistant",
        query_ids=query_ids,
        label="source Assistant checkpoint",
    )
    checkpoints = tuple(
        _load_assistant_checkpoint(
            None,
            execution_root=root,
            accepted_batch_id=shard.accepted_batch_id,
            member=member,
            expected_config=expected_config,
            expected_bank_sha256=expected_bank_sha256,
            expected_query=public_by_id[member.query_id],
            expected_image_sha256=asset_by_id[member.query_id].image_sha256,
        )
        for member in members
    )
    context = _SourceShardContext(
        inputs=inputs,
        execution_root=root,
        control=control,
        control_file_sha256=sha256_bytes(control_bytes),
        control_sha256=control_sha256,
        launch=launch,
        runtime_lock=runtime_lock,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        bank=bank,
        bank_file_sha256=bank_file_sha256,
        shard=shard,
        members=members,
        shard_root=shard_root,
        summary=summary,
        summary_file_sha256=sha256_bytes(summary_bytes),
        summary_sha256=summary_sha256,
        audit=audit,
        audit_file_sha256=sha256_bytes(audit_bytes),
        audit_sha256=audit_sha256,
        checkpoints=checkpoints,
    )
    _validate_assistant_audit(context)
    return context


def _source_manifest_payload(
    context: _SourceShardContext,
    *,
    stage: Literal["s2", "s3"],
) -> dict[str, object]:
    payload = _common_manifest_payload(
        context.inputs,
        stage=stage,
        accepted_batch_id=context.shard.accepted_batch_id,
    )
    return {
        **payload,
        "source_config": context.shard.config,
        "source_execution_control_file_sha256": context.control_file_sha256,
        "source_execution_control_sha256": context.control_sha256,
        "source_launch_plan_file_sha256": context.launch.plan_file_sha256,
        "source_launch_plan_sha256": context.launch.plan.launch_plan_sha256,
        "source_runtime_lock_file_sha256": (context.runtime_lock_file_sha256),
        "source_runtime_lock_sha256": context.runtime_lock_sha256,
        "source_shard_id": context.shard.shard_id,
        "source_shard_sha256": context.shard.shard_sha256,
        "source_shard_summary_file_sha256": context.summary_file_sha256,
        "source_shard_summary_sha256": context.summary_sha256,
        "source_shard_audit_file_sha256": context.audit_file_sha256,
        "source_shard_audit_sha256": context.audit_sha256,
        "source_bank_sha256": context.bank.bank_sha256,
    }


def prepare_s2_inputs(
    *,
    execution_root: str | Path,
    source_shard_id: str,
    expected_source_shard_audit_sha256: str,
    output_dir: str | Path,
) -> CreatedEvolutionInputBundle:
    """Freeze route attribution from one completed S1 source shard."""

    context = _load_source_shard(
        execution_root=execution_root,
        source_shard_id=source_shard_id,
        expected_source_shard_audit_sha256=(expected_source_shard_audit_sha256),
        expected_config="s1",
    )
    private_by_id = {item.query_id: item for item in context.inputs.queries}
    records: list[S2RouteAttributionRecord] = []
    for ordinal, checkpoint in enumerate(context.checkpoints):
        private = private_by_id[checkpoint.member.query_id]
        assert private.canonical_capability is not None
        response = checkpoint.response
        payload = {
            "schema_version": 1,
            "kind": "portfolio-s2-route-attribution",
            "record_ordinal": ordinal,
            "source_query_ordinal": checkpoint.member.query_ordinal,
            "accepted_batch_id": context.shard.accepted_batch_id,
            "query_id": checkpoint.member.query_id,
            "public_input_sha256": checkpoint.member.public_input_sha256,
            "query_sha256": checkpoint.member.query_sha256,
            "expected_capability": private.canonical_capability,
            "selected_capability": response.selected_capability,
            "route_correct": (
                response.selected_capability == private.canonical_capability
            ),
            "skill_slug": response.skill_slug,
            "route_trace_sha256": response.route_trace_sha256,
            "tool_names": [item.tool_name for item in response.tool_trace],
            "tool_trace": [
                item.model_dump(mode="json") for item in response.tool_trace
            ],
            "assistant_error_code": response.error_code,
            "source_bank_sha256": context.bank.bank_sha256,
            "request_sha256": checkpoint.request.request_sha256,
            "response_sha256": checkpoint.receipt.response_sha256,
            "receipt_sha256": checkpoint.receipt.receipt_sha256,
            "assistant_checkpoint_file_sha256": (checkpoint.checkpoint_file_sha256),
        }
        records.append(
            S2RouteAttributionRecord.model_validate_json(
                canonical_json_bytes(
                    {**payload, "record_sha256": _hash_payload(payload)}
                ),
                strict=True,
            )
        )
    return _publish_bundle(
        Path(output_dir),
        records=tuple(records),
        manifest_payload=_source_manifest_payload(context, stage="s2"),
    )


def _load_fixed_zero(
    path: Path,
    *,
    query_id: str,
    assistant_error_code: str,
) -> tuple[dict[str, object], bytes]:
    content, raw = _canonical_object(path, label=f"fixed-zero {query_id}")
    supplied = _require_self_hash(
        raw,
        "result_sha256",
        label=f"fixed-zero {query_id}",
    )
    if (
        raw.get("kind") != "portfolio-final-fixed-zero"
        or raw.get("query_id") != query_id
        or raw.get("assistant_error_code") != assistant_error_code
        or raw.get("j_project") != 0.0
        or supplied != raw.get("result_sha256")
    ):
        raise PortfolioEvolutionInputError(
            f"fixed-zero result binding differs: {query_id}"
        )
    return raw, content


def _rubric_and_key(
    context: _SourceShardContext,
) -> tuple[RubricSnapshot, bytes]:
    rubric_path_value = context.control.get("rubric_path")
    if not isinstance(rubric_path_value, str):
        raise PortfolioEvolutionInputError("source execution control lacks rubric path")
    rubric_content = read_stable_regular_file(
        Path(rubric_path_value),
        label="source final rubric",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(rubric_content) != context.control.get("rubric_file_sha256"):
        raise PortfolioEvolutionInputError("source final rubric file digest mismatch")
    try:
        rubric = RubricSnapshot.model_validate_json(
            rubric_content,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioEvolutionInputError(
            "source final rubric violates schema"
        ) from error
    if rubric.content_sha256 != context.control.get("rubric_content_sha256"):
        raise PortfolioEvolutionInputError(
            "source final rubric content binding mismatch"
        )
    key = read_stable_regular_file(
        context.execution_root / "blinding-key.bin",
        label="source execution blinding key",
        max_bytes=32,
    )
    return rubric, key


def _s3_final_projection(
    context: _SourceShardContext,
    checkpoint: _AssistantCheckpoint,
    *,
    asset_catalog: AssetCatalog,
    rubric: RubricSnapshot,
    blinding_key: bytes,
) -> dict[str, object]:
    member = checkpoint.member
    response = checkpoint.response
    final_path = context.execution_root / member.final_output_relpath
    if response.error_code is not None:
        fixed, fixed_content = _load_fixed_zero(
            final_path,
            query_id=member.query_id,
            assistant_error_code=response.error_code,
        )
        return {
            "final_kind": "assistant_fixed_zero",
            "evaluation_id": None,
            "judge_status": "not_invoked_assistant_error",
            "judge_error_code": response.error_code,
            "dimensions": [],
            "j_project": 0.0,
            "final_result_sha256": fixed["result_sha256"],
            "final_checkpoint_file_sha256": sha256_bytes(fixed_content),
        }

    final = load_final_judge_evaluation_result(final_path)
    final_content = read_stable_regular_file(
        final_path,
        label=f"source final checkpoint {member.query_id}",
        max_bytes=4 * 1024 * 1024,
    )
    if canonical_json_bytes(final) != final_content:
        raise PortfolioEvolutionInputError(
            f"source final checkpoint changed: {member.query_id}"
        )
    private = next(
        item for item in context.inputs.queries if item.query_id == member.query_id
    )
    assistant_result = _assistant_result(
        context.launch,
        checkpoint.request,
        response,
    )
    packet = build_final_evaluation_packet(
        private,
        assistant_result,
        asset_catalog=asset_catalog,
        rubric=rubric,
        blinding_key=blinding_key,
    )
    expected_evaluation_id = derive_blinded_evaluation_id(
        blinding_key=blinding_key,
        run_id=context.launch.plan.matrix_run_id,
        query_id=member.query_id,
        config="s1s2",
    )
    locked_schema_version, locked_cache_namespace = _locked_final_result_contract(
        context.runtime_lock
    )
    if (
        final.evaluation_id != expected_evaluation_id
        or final.evaluation_id != packet.evaluation_id
        or final.packet_sha256 != packet.packet_sha256
        or final.image_sha256 != member.image_sha256
        or final.asset_catalog_sha256 != context.inputs.expected_output_catalog_sha256
        or final.parser_policy_version
        != context.runtime_lock.get("final_judge_parser_policy_version")
        or final.parser_policy_sha256
        != context.runtime_lock.get("final_judge_parser_policy_sha256")
        or final.schema_version != locked_schema_version
        or final.cache_namespace != locked_cache_namespace
        or (
            final.schema_version in {4, 5, 6, 7, 8, 9}
            and (
                final.card_requirement_guard_policy_version
                != context.runtime_lock.get("card_requirement_guard_policy_version")
                or final.card_requirement_guard_policy_sha256
                != context.runtime_lock.get("card_requirement_guard_policy_sha256")
                or final.visible_card_count != len(response.visible_cards)
            )
        )
        or (
            final.schema_version in {5, 6, 7, 8, 9}
            and (
                final.max_attempts
                != context.runtime_lock.get("final_judge_max_attempts")
                or final.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                or final.retry_policy_version
                != context.runtime_lock.get("final_judge_retry_policy_version")
                or final.retry_policy_sha256
                != context.runtime_lock.get("final_judge_retry_policy_sha256")
            )
        )
        or (
            final.schema_version in {6, 7, 8, 9}
            and (
                final.thinking_budget
                != context.runtime_lock.get("final_judge_thinking_budget")
                or final.max_billable_input_tokens
                != context.runtime_lock.get("final_judge_max_billable_input_tokens")
                or final.max_billable_output_tokens
                != context.runtime_lock.get("final_judge_max_billable_output_tokens")
            )
        )
    ):
        raise PortfolioEvolutionInputError(
            f"source final checkpoint binding differs: {member.query_id}"
        )
    return {
        "final_kind": "visual_final_judge",
        "evaluation_id": final.evaluation_id,
        "judge_status": final.outcome.status,
        "judge_error_code": final.outcome.error_code,
        "dimensions": [
            item.model_dump(mode="json") for item in final.outcome.scores.dimensions
        ],
        "j_project": final.outcome.scores.j_project,
        "final_result_sha256": final.result_sha256,
        "final_checkpoint_file_sha256": sha256_bytes(final_content),
        "_query_id": member.query_id,
        "_judge_usage": final.aggregate_usage,
        "_judge_shape": final.raw_dimensions_shape,
        "_judge_attempts": final.attempts,
        "_judge_captured_response_count": final.captured_response_count,
        "_judge_initial_empty_response": (
            final.initial_empty_response is not None
            and final.initial_empty_response.retry_reason == "empty_final_response"
        ),
        "_judge_initial_retry_reason": (
            None
            if final.initial_empty_response is None
            else final.initial_empty_response.retry_reason
        ),
        "_judge_initial_reasoning_present": (
            None
            if final.initial_empty_response is None
            else final.initial_empty_response.reasoning_present
        ),
        "_judge_initial_reasoning_tokens": (
            None
            if final.initial_empty_response is None
            else final.initial_empty_response.reasoning_tokens
        ),
        "_judge_initial_reasoning_bytes": (
            None
            if final.initial_empty_response is None
            else final.initial_empty_response.reasoning_bytes
        ),
        "_judge_initial_reasoning_sha256": (
            None
            if final.initial_empty_response is None
            else final.initial_empty_response.reasoning_sha256
        ),
        "_judge_terminal_response_captured": final.request_id is not None,
        "_judge_terminal_reasoning_present": final.reasoning_present,
        "_judge_terminal_reasoning_tokens": final.reasoning_tokens,
        "_judge_terminal_reasoning_bytes": final.reasoning_bytes,
        "_judge_terminal_reasoning_sha256": final.reasoning_sha256,
    }


def _validate_final_audit(
    context: _SourceShardContext,
    projections: tuple[dict[str, object], ...],
) -> None:
    fixed_count = sum(
        item["final_kind"] == "assistant_fixed_zero" for item in projections
    )
    judged = [
        item for item in projections if item["final_kind"] == "visual_final_judge"
    ]
    statuses = Counter(str(item["judge_status"]) for item in judged)
    shapes = Counter(
        str(item["_judge_shape"])
        for item in judged
        if item.get("_judge_shape") is not None
    )
    scores = [float(item["j_project"]) for item in projections]
    judge_input_tokens = sum(
        int(item["_judge_usage"].input_tokens)  # type: ignore[union-attr]
        for item in judged
    )
    judge_output_tokens = sum(
        int(item["_judge_usage"].output_tokens)  # type: ignore[union-attr]
        for item in judged
    )
    if (
        context.audit.get("fixed_zero_count") != fixed_count
        or context.audit.get("judge_invoked_count") != len(judged)
        or context.audit.get("judge_status_counts") != dict(sorted(statuses.items()))
        or context.audit.get("judge_input_tokens") != judge_input_tokens
        or context.audit.get("judge_output_tokens") != judge_output_tokens
        or abs(
            float(context.audit.get("mean_j_project_all_rows", -1.0))
            - sum(scores) / EXPECTED_RECORD_COUNT
        )
        > 1e-9
        or abs(float(context.audit.get("min_j_project", -1.0)) - min(scores)) > 1e-9
        or abs(float(context.audit.get("max_j_project", -1.0)) - max(scores)) > 1e-9
    ):
        raise PortfolioEvolutionInputError(
            "source shard audit differs from final checkpoints"
        )
    audit_shapes = context.audit.get("judge_raw_dimensions_shape_counts")
    if audit_shapes is not None and audit_shapes != dict(sorted(shapes.items())):
        raise PortfolioEvolutionInputError(
            "source shard audit Judge shapes differ from final checkpoints"
        )
    result_schema_version = context.runtime_lock.get(
        "final_judge_result_schema_version"
    )
    if result_schema_version in {5, 6, 7, 8, 9}:
        judge_response_receipts = [
            {
                "query_id": item["_query_id"],
                "attempts": item["_judge_attempts"],
                "initial_empty_response": item["_judge_initial_empty_response"],
                **(
                    {"initial_retry_reason": item["_judge_initial_retry_reason"]}
                    if result_schema_version in {7, 8, 9}
                    else {}
                ),
                "initial_reasoning_present": item["_judge_initial_reasoning_present"],
                "initial_reasoning_tokens": item["_judge_initial_reasoning_tokens"],
                "initial_reasoning_bytes": item["_judge_initial_reasoning_bytes"],
                "initial_reasoning_sha256": item["_judge_initial_reasoning_sha256"],
                "terminal_response_captured": item["_judge_terminal_response_captured"],
                "terminal_reasoning_present": item["_judge_terminal_reasoning_present"],
                "terminal_reasoning_tokens": item["_judge_terminal_reasoning_tokens"],
                "terminal_reasoning_bytes": item["_judge_terminal_reasoning_bytes"],
                "terminal_reasoning_sha256": item["_judge_terminal_reasoning_sha256"],
            }
            for item in judged
        ]
        expected_retry_audit = {
            "judge_model_calls": sum(int(item["_judge_attempts"]) for item in judged),
            "judge_captured_response_count": sum(
                int(item["_judge_captured_response_count"]) for item in judged
            ),
            "judge_retried_row_count": sum(
                item["_judge_attempts"] == 2 for item in judged
            ),
            "judge_initial_empty_response_count": sum(
                bool(item["_judge_initial_empty_response"]) for item in judged
            ),
            "judge_reasoning_present_response_count": sum(
                item[field] is True
                for item in judged
                for field in (
                    "_judge_initial_reasoning_present",
                    "_judge_terminal_reasoning_present",
                )
            ),
            "judge_reasoning_tokens_reported_total": sum(
                int(item[field])
                for item in judged
                for field in (
                    "_judge_initial_reasoning_tokens",
                    "_judge_terminal_reasoning_tokens",
                )
                if item[field] is not None
            ),
            "judge_reasoning_tokens_unavailable_response_count": sum(
                item["_judge_initial_retry_reason"] is not None
                and item["_judge_initial_reasoning_tokens"] is None
                for item in judged
            )
            + sum(
                bool(item["_judge_terminal_response_captured"])
                and item["_judge_terminal_reasoning_tokens"] is None
                for item in judged
            ),
            "judge_reasoning_bytes_total": sum(
                int(item[field])
                for item in judged
                for field in (
                    "_judge_initial_reasoning_bytes",
                    "_judge_terminal_reasoning_bytes",
                )
                if item[field] is not None
            ),
            "judge_response_receipts": judge_response_receipts,
        }
        if result_schema_version in {7, 8, 9}:
            expected_retry_audit["judge_initial_invalid_json_response_count"] = sum(
                item["_judge_initial_retry_reason"] == "invalid_judge_json"
                for item in judged
            )
        if any(
            context.audit.get(field) != expected
            for field, expected in expected_retry_audit.items()
        ):
            raise PortfolioEvolutionInputError(
                "source shard audit retry/reasoning metadata differs from final "
                "checkpoints"
            )


def _judge_processor_for_launch(plan: object) -> str:
    identity = (
        getattr(plan, "final_provider", None),
        getattr(plan, "final_model", None),
    )
    if identity == ("gemini", "gemini-3.6-flash"):
        return "aifast-gemini-judge"
    if identity == ("kimi", "kimi-k2.6"):
        return "dashscope-kimi-judge"
    raise PortfolioEvolutionInputError("source launch has an unknown final-Judge role")


def prepare_s3_inputs(
    *,
    execution_root: str | Path,
    source_shard_id: str,
    expected_source_shard_audit_sha256: str,
    output_dir: str | Path,
) -> CreatedEvolutionInputBundle:
    """Freeze body attribution from one completed S1+S2 source shard."""

    context = _load_source_shard(
        execution_root=execution_root,
        source_shard_id=source_shard_id,
        expected_source_shard_audit_sha256=(expected_source_shard_audit_sha256),
        expected_config="s1s2",
    )
    _assert_exact_checkpoint_files(
        context.shard_root / "final",
        query_ids=tuple(item.query_id for item in context.members),
        label="source final checkpoint",
    )
    rubric, blinding_key = _rubric_and_key(context)
    judge_processor = _judge_processor_for_launch(context.launch.plan)
    judge_asset_catalog = context.inputs.runtime_for(judge_processor).catalog
    projections = tuple(
        _s3_final_projection(
            context,
            checkpoint,
            asset_catalog=judge_asset_catalog,
            rubric=rubric,
            blinding_key=blinding_key,
        )
        for checkpoint in context.checkpoints
    )
    _validate_final_audit(context, projections)
    private_by_id = {item.query_id: item for item in context.inputs.queries}
    records: list[S3BodyAttributionRecord] = []
    for ordinal, (checkpoint, final_projection) in enumerate(
        zip(context.checkpoints, projections, strict=True)
    ):
        private = private_by_id[checkpoint.member.query_id]
        assert private.canonical_capability is not None
        response = checkpoint.response
        public_projection = {
            key: value
            for key, value in final_projection.items()
            if not key.startswith("_")
        }
        payload = {
            "schema_version": 1,
            "kind": "portfolio-s3-body-attribution",
            "record_ordinal": ordinal,
            "source_query_ordinal": checkpoint.member.query_ordinal,
            "accepted_batch_id": context.shard.accepted_batch_id,
            "query_id": checkpoint.member.query_id,
            "public_input_sha256": checkpoint.member.public_input_sha256,
            "query_sha256": checkpoint.member.query_sha256,
            "canonical_capability": private.canonical_capability,
            "selected_capability": response.selected_capability,
            "skill_slug": response.skill_slug,
            "route_trace_sha256": response.route_trace_sha256,
            "shared_route_artifact_file_sha256": (checkpoint.shared_route_file_sha256),
            "response_text": response.response_text,
            "visible_cards": [
                item.model_dump(mode="json") for item in response.visible_cards
            ],
            "visible_tool_evidence": [
                item.model_dump(mode="json") for item in response.visible_tool_evidence
            ],
            "tool_trace": [
                item.model_dump(mode="json") for item in response.tool_trace
            ],
            "assistant_error_code": response.error_code,
            "source_bank_sha256": context.bank.bank_sha256,
            "request_sha256": checkpoint.request.request_sha256,
            "response_sha256": checkpoint.receipt.response_sha256,
            "receipt_sha256": checkpoint.receipt.receipt_sha256,
            "assistant_checkpoint_file_sha256": (checkpoint.checkpoint_file_sha256),
            **public_projection,
        }
        records.append(
            S3BodyAttributionRecord.model_validate_json(
                canonical_json_bytes(
                    {**payload, "record_sha256": _hash_payload(payload)}
                ),
                strict=True,
            )
        )
    return _publish_bundle(
        Path(output_dir),
        records=tuple(records),
        manifest_payload=_source_manifest_payload(context, stage="s3"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="stage", required=True)

    s1 = subcommands.add_parser("s1", help="Prepare S1 user trajectories.")
    s1.add_argument("--accepted-batch-id", required=True)
    s1.add_argument("--output-dir", type=Path, required=True)

    for stage, help_text in (
        ("s2", "Prepare S2 route attribution from one S1 shard."),
        ("s3", "Prepare S3 body attribution from one S1+S2 shard."),
    ):
        child = subcommands.add_parser(stage, help=help_text)
        child.add_argument("--execution-root", type=Path, required=True)
        child.add_argument("--source-shard-id", required=True)
        child.add_argument(
            "--expected-source-shard-audit-sha256",
            required=True,
            help="Caller-pinned shard-audit self SHA-256.",
        )
        child.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.stage == "s1":
            created = prepare_s1_inputs(
                accepted_batch_id=arguments.accepted_batch_id,
                output_dir=arguments.output_dir,
            )
        elif arguments.stage == "s2":
            created = prepare_s2_inputs(
                execution_root=arguments.execution_root,
                source_shard_id=arguments.source_shard_id,
                expected_source_shard_audit_sha256=(
                    arguments.expected_source_shard_audit_sha256
                ),
                output_dir=arguments.output_dir,
            )
        else:
            created = prepare_s3_inputs(
                execution_root=arguments.execution_root,
                source_shard_id=arguments.source_shard_id,
                expected_source_shard_audit_sha256=(
                    arguments.expected_source_shard_audit_sha256
                ),
                output_dir=arguments.output_dir,
            )
    except (
        ArtifactFormatError,
        FileExistsError,
        OSError,
        PortfolioEvolutionInputError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"prepare-portfolio-evolution-inputs: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "bank_generated": False,
                "component": created.manifest.component,
                "evolution_component_invoked": False,
                "manifest_file_sha256": created.manifest_file_sha256,
                "manifest_sha256": created.manifest.manifest_sha256,
                "model_calls_performed": 0,
                "output_dir": str(created.root),
                "packet_file_sha256": created.packet_file_sha256,
                "packet_path": str(created.packet_path),
                "packet_sha256": created.packet_sha256,
                "record_count": len(created.records),
                "records_file_sha256": created.records_file_sha256,
                "stage": created.manifest.stage,
                "status": created.manifest.status,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
