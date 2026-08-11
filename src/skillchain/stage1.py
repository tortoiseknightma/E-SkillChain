"""Fail-closed preparation contracts for the paper-aligned Stage 1 Creator.

This module only prepares immutable inputs.  It does not invoke an author model,
compile a Skill Bank, inspect evaluation feedback, or claim that a Stage 1 run
has happened.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Annotated, Iterable, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.codex_authoring import CodexAuthoringInput
from skillchain.codex_authoring_v5 import (
    CodexAuthoringContractError,
    load_codex_authoring_input,
)
from skillchain.schemas import Intent, Query
from skillchain.synthesis.batches import verify_accepted_corpus
from skillchain.synthesis.planning import R2CoreInMemoryPlan
from skillchain.synthesis.portfolio_core_authoring import RealismSidecar
from skillchain.synthesis.portfolio_core_selection import (
    CREATOR_SELECTION_SIZE,
    CoreR2CreatorSelection,
    canonical_creator_selection_index_bytes,
    validate_core_r2_creator_selection,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PAPER_INTENTS: tuple[Intent, ...] = (
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
)
_PAPER_INTENT_SET = frozenset(PAPER_INTENTS)
_MAX_AUTHORING_INPUT_BYTES = 16 * 1024 * 1024
_MAX_SELECTION_BYTES = 4 * 1024 * 1024
_MAX_PREPARED_ARTIFACT_BYTES = 64 * 1024 * 1024


class Stage1ContractError(ValueError):
    """A Stage 1 preparation input or persisted artifact is unsafe or invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _nonblank(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonblank without edge whitespace")
    return value


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


class QuerySelection(_StrictFrozenModel):
    """Explicit, deterministic opt-pool query selection."""

    schema_version: Literal[1] = 1
    query_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("query_ids", mode="before")
    @classmethod
    def coerce_query_ids(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("query_ids")
    @classmethod
    def validate_query_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for query_id in value:
            _nonblank(query_id, "query_id")
        if len(value) != len(set(value)):
            raise ValueError("query_ids must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("query_ids must be in lexical order")
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class TrajectoryTurn(_StrictFrozenModel):
    """One verbatim turn from an accepted Phase 3 interaction sequence."""

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _nonblank(value, "content")


class UserTrajectory(_StrictFrozenModel):
    """The Stage 1-visible projection of one accepted opt-pool query."""

    query_id: str
    asset_id: str
    image_path: str
    turns: tuple[TrajectoryTurn, ...] = Field(min_length=1)
    canonical_intent: Intent
    canonical_capability: str

    @field_validator("turns", mode="before")
    @classmethod
    def coerce_turns(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator(
        "query_id",
        "asset_id",
        "image_path",
        "canonical_capability",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_turn_shape(self) -> Self:
        roles = [turn.role for turn in self.turns]
        if roles not in (["user"], ["user", "assistant", "user"]):
            raise ValueError("turns must be exactly [user] or [user, assistant, user]")
        return self


class TrajectoryBundle(_StrictFrozenModel):
    """The only experimental input S1 may receive beyond its common input."""

    schema_version: Literal[1] = 1
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    accepted_corpus_sha256: Sha256
    taxonomy_version: str
    task_spec_version: str
    trajectories: tuple[UserTrajectory, ...] = Field(min_length=1)
    trajectory_bundle_sha256: Sha256

    @field_validator("trajectories", mode="before")
    @classmethod
    def coerce_trajectories(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("taxonomy_version", "task_spec_version")
    @classmethod
    def validate_versions(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        query_ids = tuple(item.query_id for item in self.trajectories)
        if query_ids != tuple(sorted(set(query_ids))):
            raise ValueError("trajectories must have unique query IDs in lexical order")
        observed_intents = {item.canonical_intent for item in self.trajectories}
        if observed_intents != _PAPER_INTENT_SET:
            missing = sorted(_PAPER_INTENT_SET - observed_intents)
            unexpected = sorted(observed_intents - _PAPER_INTENT_SET)
            raise ValueError(
                "trajectory bundle must cover exactly the paper five intents; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.trajectory_bundle_sha256 != _digest_without(
            self.model_dump(mode="json"), "trajectory_bundle_sha256"
        ):
            raise ValueError("trajectory_bundle_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class CoreR2Stage1SelectionReceipt(_StrictFrozenModel):
    """Strong binding from the r2 240-row index to a Stage 1 bundle.

    The receipt is deliberately separate from :class:`TrajectoryBundle` so
    legacy Stage 1 artifact bytes stay stable while the r2 S1 path cannot be
    invoked with a caller-supplied list of query IDs.
    """

    schema_version: Literal[1] = 1
    selection_size: Literal[240] = CREATOR_SELECTION_SIZE
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    selection_manifest_sha256: Sha256
    selection_index_sha256: Sha256
    selection_ranked_query_ids_sha256: Sha256
    accepted_corpus_sha256: Sha256
    trajectory_bundle_sha256: Sha256
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.receipt_sha256 != _digest_without(
            self.model_dump(mode="json"), "receipt_sha256"
        ):
            raise ValueError("r2 Stage 1 selection receipt SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CoreR2Stage1TrajectoryBundle:
    """The existing Stage 1 bundle plus its mandatory r2 selection receipt."""

    trajectory_bundle: TrajectoryBundle
    selection_receipt: CoreR2Stage1SelectionReceipt


class S1CreatorPacket(_StrictFrozenModel):
    """Create-only Stage 1 input: common input plus the independent bundle."""

    schema_version: Literal[1] = 1
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    authoring_input: CodexAuthoringInput
    trajectory_bundle: TrajectoryBundle
    packet_sha256: Sha256

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if (
            self.trajectory_bundle.taxonomy_version
            != self.authoring_input.taxonomy.version
            or self.trajectory_bundle.task_spec_version
            != self.authoring_input.task_specification.version
        ):
            raise ValueError(
                "TrajectoryBundle specification versions differ from AuthoringInput"
            )
        if self.packet_sha256 != _digest_without(
            self.model_dump(mode="json"), "packet_sha256"
        ):
            raise ValueError("packet_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class PreparedStage1:
    output_dir: Path
    trajectory_bundle_path: Path
    s1_creator_packet_path: Path
    trajectory_bundle_file_sha256: str
    s1_creator_packet_file_sha256: str
    selected_query_count: int


def build_query_selection(query_ids: Iterable[str]) -> QuerySelection:
    values = tuple(query_ids)
    for query_id in values:
        if not isinstance(query_id, str):
            raise Stage1ContractError("query IDs must be strings")
        try:
            _nonblank(query_id, "query_id")
        except ValueError as error:
            raise Stage1ContractError(str(error)) from error
    if not values:
        raise Stage1ContractError("at least one query ID must be selected")
    if len(values) != len(set(values)):
        raise Stage1ContractError("selected query IDs must be unique")
    try:
        return QuerySelection(query_ids=tuple(sorted(values)))
    except ValidationError as error:
        raise Stage1ContractError("query selection is invalid") from error


def load_query_selection(
    path: str | Path, *, expected_file_sha256: str
) -> QuerySelection:
    return _load_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=QuerySelection,
        label="Stage 1 query selection",
        max_bytes=_MAX_SELECTION_BYTES,
    )


def build_trajectory_bundle(
    accepted_queries: Iterable[Query],
    selected_query_ids: Iterable[str],
) -> TrajectoryBundle:
    """Project explicit opt-pool selections without exposing later-stage signals."""

    queries = tuple(accepted_queries)
    if not queries:
        raise Stage1ContractError("accepted corpus is empty")
    if any(not isinstance(query, Query) for query in queries):
        raise Stage1ContractError("accepted corpus must contain schema-v2 Query values")
    selection = build_query_selection(selected_query_ids)
    by_id: dict[str, Query] = {}
    for query in queries:
        if query.query_id in by_id:
            raise Stage1ContractError(
                f"accepted corpus contains duplicate query ID: {query.query_id}"
            )
        by_id[query.query_id] = query

    missing = [query_id for query_id in selection.query_ids if query_id not in by_id]
    if missing:
        raise Stage1ContractError(
            "selected query IDs are absent from accepted corpus: " + ", ".join(missing)
        )

    selected_queries = tuple(by_id[query_id] for query_id in selection.query_ids)
    taxonomy_versions = {query.taxonomy_version for query in selected_queries}
    task_spec_versions = {query.task_spec_version for query in selected_queries}
    if len(taxonomy_versions) != 1 or len(task_spec_versions) != 1:
        raise Stage1ContractError(
            "selected queries must share one taxonomy_version and task_spec_version"
        )
    taxonomy_version = next(iter(taxonomy_versions))
    task_spec_version = next(iter(task_spec_versions))

    trajectories: list[UserTrajectory] = []
    for query in selected_queries:
        query_id = query.query_id
        if query.split != "opt_pool":
            raise Stage1ContractError(
                f"Stage 1 selection must contain only opt_pool queries: "
                f"{query_id}={query.split}"
            )
        if query.canonical_capability is None:
            raise Stage1ContractError(
                f"Stage 1 query lacks canonical capability grouping: {query_id}"
            )
        turns = tuple(
            TrajectoryTurn(role=turn.role, content=turn.content) for turn in query.turns
        )
        trajectories.append(
            UserTrajectory(
                query_id=query.query_id,
                asset_id=query.asset_id,
                image_path=query.image_path,
                turns=turns,
                canonical_intent=query.canonical_intent,
                canonical_capability=query.canonical_capability,
            )
        )

    corpus_bytes = canonical_jsonl_bytes(
        tuple(query.model_dump(mode="json") for query in queries)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared_not_invoked",
        "accepted_corpus_sha256": sha256_bytes(corpus_bytes),
        "taxonomy_version": taxonomy_version,
        "task_spec_version": task_spec_version,
        "trajectories": [
            trajectory.model_dump(mode="json") for trajectory in trajectories
        ],
    }
    try:
        return TrajectoryBundle.model_validate(
            {
                **payload,
                "trajectory_bundle_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise Stage1ContractError(f"trajectory bundle is invalid: {error}") from error


def build_core_r2_stage1_trajectory_bundle(
    accepted_queries: Iterable[Query],
    selection: CoreR2CreatorSelection,
    result: R2CoreInMemoryPlan,
    realism_sidecar: RealismSidecar,
    *,
    trusted_plan_sha256: str,
) -> CoreR2Stage1TrajectoryBundle:
    """Project the verified r2 S1 selection through the existing bundle builder.

    This adapter deliberately has no ``selected_query_ids`` parameter.  The
    only admissible selection is the independently validated 240-row creator
    index, preventing a caller from substituting a hand-maintained ID list.
    ``TrajectoryBundle`` retains its legacy lexical canonical ordering; the
    receipt separately binds the exact selection-rank ordering consumed here.
    """

    try:
        validate_core_r2_creator_selection(
            selection,
            result,
            realism_sidecar,
            trusted_plan_sha256=trusted_plan_sha256,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise Stage1ContractError(
            "r2 S1 creator selection validation failed"
        ) from error

    queries = tuple(accepted_queries)
    if not queries:
        raise Stage1ContractError("accepted corpus is empty")
    if any(
        not isinstance(query, Query) or query.schema_version != 2 for query in queries
    ):
        raise Stage1ContractError("accepted corpus must contain schema-v2 Query values")
    accepted_by_id: dict[str, Query] = {}
    for query in queries:
        if query.query_id in accepted_by_id:
            raise Stage1ContractError(
                f"accepted corpus contains duplicate query ID: {query.query_id}"
            )
        accepted_by_id[query.query_id] = query

    ranked_entries = tuple(
        sorted(selection.entries, key=lambda entry: entry.selection_rank)
    )
    if len(ranked_entries) != CREATOR_SELECTION_SIZE:
        raise Stage1ContractError(
            "r2 S1 creator selection must contain exactly 240 rows"
        )
    ranked_query_ids = tuple(entry.plan_id for entry in ranked_entries)
    missing = [
        query_id for query_id in ranked_query_ids if query_id not in accepted_by_id
    ]
    if missing:
        raise Stage1ContractError(
            "r2 S1 selection IDs are absent from accepted corpus: " + ", ".join(missing)
        )
    for entry in ranked_entries:
        query = accepted_by_id[entry.plan_id]
        if query.leakage_group_id != entry.component_id:
            raise Stage1ContractError(
                f"r2 S1 accepted query leakage component drifted: {entry.plan_id}"
            )
        if query.canonical_capability != entry.canonical_capability:
            raise Stage1ContractError(
                f"r2 S1 accepted query capability drifted: {entry.plan_id}"
            )
        if query.split != entry.final_split or query.split != "opt_pool":
            raise Stage1ContractError(
                f"r2 S1 accepted query split must match opt_pool selection: "
                f"{entry.plan_id}={query.split}"
            )

    # Reuse the established Stage 1 projection and its no-later-signal checks.
    trajectory_bundle = build_trajectory_bundle(queries, ranked_query_ids)
    if {trajectory.query_id for trajectory in trajectory_bundle.trajectories} != set(
        ranked_query_ids
    ):
        raise Stage1ContractError(
            "trajectory bundle did not preserve r2 S1 selection IDs"
        )
    selection_index_sha256 = sha256_bytes(
        canonical_creator_selection_index_bytes(selection.entries)
    )
    if selection_index_sha256 != selection.manifest.selection_bytes_sha256:
        raise Stage1ContractError("r2 S1 selection index SHA-256 drifted")
    selection_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(selection.manifest.model_dump(mode="json"))
    )
    ranked_query_ids_sha256 = sha256_bytes(
        canonical_json_bytes({"query_ids": list(ranked_query_ids)})
    )
    receipt_payload: dict[str, object] = {
        "schema_version": 1,
        "selection_size": CREATOR_SELECTION_SIZE,
        "plan_sha256": selection.manifest.plan_sha256,
        "asset_catalog_sha256": selection.manifest.asset_catalog_sha256,
        "capability_assignments_sha256": (
            selection.manifest.capability_assignments_sha256
        ),
        "realism_manifest_sha256": selection.manifest.realism_manifest_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "selection_index_sha256": selection_index_sha256,
        "selection_ranked_query_ids_sha256": ranked_query_ids_sha256,
        "accepted_corpus_sha256": trajectory_bundle.accepted_corpus_sha256,
        "trajectory_bundle_sha256": trajectory_bundle.trajectory_bundle_sha256,
    }
    try:
        receipt = CoreR2Stage1SelectionReceipt.model_validate(
            {
                **receipt_payload,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise Stage1ContractError("r2 S1 selection receipt is invalid") from error
    return CoreR2Stage1TrajectoryBundle(
        trajectory_bundle=trajectory_bundle,
        selection_receipt=receipt,
    )


def build_s1_creator_packet(
    authoring_input: CodexAuthoringInput,
    trajectory_bundle: TrajectoryBundle,
) -> S1CreatorPacket:
    try:
        authoring_input = CodexAuthoringInput.model_validate(
            authoring_input.model_dump(mode="python"), strict=True
        )
        trajectory_bundle = TrajectoryBundle.model_validate(
            trajectory_bundle.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise Stage1ContractError("S1 Creator inputs are invalid") from error
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared_not_invoked",
        "authoring_input": authoring_input.model_dump(mode="json"),
        "trajectory_bundle": trajectory_bundle.model_dump(mode="json"),
    }
    try:
        return S1CreatorPacket.model_validate(
            {
                **payload,
                "packet_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise Stage1ContractError(f"S1 Creator packet is invalid: {error}") from error


def load_trajectory_bundle(
    path: str | Path, *, expected_file_sha256: str
) -> TrajectoryBundle:
    return _load_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=TrajectoryBundle,
        label="TrajectoryBundle",
        max_bytes=_MAX_PREPARED_ARTIFACT_BYTES,
    )


def load_s1_creator_packet(
    path: str | Path, *, expected_file_sha256: str
) -> S1CreatorPacket:
    return _load_canonical_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=S1CreatorPacket,
        label="S1CreatorPacket",
        max_bytes=_MAX_PREPARED_ARTIFACT_BYTES,
    )


def prepare_stage1(
    *,
    authoring_input_path: str | Path,
    expected_authoring_input_file_sha256: str,
    queries_root: str | Path,
    selected_query_ids: Iterable[str],
    output_dir: str | Path,
) -> PreparedStage1:
    """Create immutable Stage 1 preparation artifacts without invoking a model."""

    authoring_path = Path(authoring_input_path)
    queries_path = Path(queries_root)
    destination = Path(output_dir)
    _require_real_directory(queries_path, "queries root")
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Stage 1 preparation is create-only; target exists: {destination}"
        )

    try:
        authoring_input = load_codex_authoring_input(
            authoring_path,
            expected_file_sha256=expected_authoring_input_file_sha256,
        )
        authoring_bytes = read_stable_regular_file(
            authoring_path,
            label="common CodexAuthoringInput",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        )
    except (ArtifactFormatError, CodexAuthoringContractError) as error:
        raise Stage1ContractError(
            "common CodexAuthoringInput verification failed"
        ) from error
    if (
        sha256_bytes(authoring_bytes) != expected_authoring_input_file_sha256
        or authoring_bytes != authoring_input.canonical_bytes()
    ):
        raise Stage1ContractError(
            "common CodexAuthoringInput bytes changed or do not match canonical "
            "embedding"
        )

    try:
        accepted_queries = tuple(verify_accepted_corpus(queries_path))
    except (FileNotFoundError, OSError, ValueError) as error:
        raise Stage1ContractError("accepted corpus verification failed") from error
    derived_bytes = _read_verified_corpus_bytes(queries_path, accepted_queries)
    trajectory_bundle = build_trajectory_bundle(accepted_queries, selected_query_ids)
    if trajectory_bundle.accepted_corpus_sha256 != sha256_bytes(derived_bytes):
        raise Stage1ContractError("accepted corpus changed after verification")
    creator_packet = build_s1_creator_packet(authoring_input, trajectory_bundle)

    bundle_bytes = trajectory_bundle.canonical_bytes()
    packet_bytes = creator_packet.canonical_bytes()
    packet_raw = parse_canonical_json(packet_bytes, label="S1CreatorPacket")
    if not isinstance(packet_raw, dict):
        raise Stage1ContractError("S1CreatorPacket must be an object")
    embedded_authoring = packet_raw.get("authoring_input")
    embedded_bundle = packet_raw.get("trajectory_bundle")
    if (
        canonical_json_bytes(embedded_authoring) != authoring_bytes
        or canonical_json_bytes(embedded_bundle) != bundle_bytes
    ):
        raise Stage1ContractError(
            "S1CreatorPacket embedding differs from common input or TrajectoryBundle"
        )

    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / "trajectory-bundle.json", bundle_bytes)
        atomic_create_file(staging / "s1-creator-packet.json", packet_bytes)
        atomic_publish_new_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    bundle_path = destination / "trajectory-bundle.json"
    packet_path = destination / "s1-creator-packet.json"
    return PreparedStage1(
        output_dir=destination,
        trajectory_bundle_path=bundle_path,
        s1_creator_packet_path=packet_path,
        trajectory_bundle_file_sha256=sha256_bytes(bundle_bytes),
        s1_creator_packet_file_sha256=sha256_bytes(packet_bytes),
        selected_query_count=len(trajectory_bundle.trajectories),
    )


def _read_verified_corpus_bytes(
    queries_root: Path, accepted_queries: tuple[Query, ...]
) -> bytes:
    try:
        content = read_stable_regular_file(
            queries_root / "queries.jsonl",
            label="accepted derived queries",
            max_bytes=_MAX_PREPARED_ARTIFACT_BYTES,
        )
    except ArtifactFormatError as error:
        raise Stage1ContractError("accepted derived queries are unsafe") from error
    expected = canonical_jsonl_bytes(
        tuple(query.model_dump(mode="json") for query in accepted_queries)
    )
    if content != expected:
        raise Stage1ContractError(
            "accepted derived queries changed after authoritative verification"
        )
    return content


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Stage1ContractError(f"{label} cannot be inspected: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Stage1ContractError(
            f"{label} must be a regular non-symlink directory: {path}"
        )
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(metadata, "st_file_attributes", 0) & reparse_attribute:
        raise Stage1ContractError(
            f"{label} must not be a Windows junction or reparse point: {path}"
        )


def _load_canonical_model(
    path: str | Path,
    *,
    expected_file_sha256: str,
    model_type: type[QuerySelection] | type[TrajectoryBundle] | type[S1CreatorPacket],
    label: str,
    max_bytes: int,
):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise Stage1ContractError(f"{label} expected SHA-256 is invalid")
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
        if sha256_bytes(content) != expected_file_sha256:
            raise Stage1ContractError(f"{label} file digest mismatch")
        raw = parse_canonical_json(content, label=label)
        if not isinstance(raw, dict):
            raise Stage1ContractError(f"{label} must be an object")
        value = model_type.model_validate(raw, strict=True)
    except Stage1ContractError:
        raise
    except (ArtifactFormatError, ValidationError) as error:
        raise Stage1ContractError(f"{label} is not canonical and valid") from error
    if canonical_json_bytes(value.model_dump(mode="json")) != content:
        raise Stage1ContractError(f"{label} canonical bytes changed during validation")
    return value


__all__ = [
    "CoreR2Stage1SelectionReceipt",
    "CoreR2Stage1TrajectoryBundle",
    "PAPER_INTENTS",
    "PreparedStage1",
    "QuerySelection",
    "S1CreatorPacket",
    "Stage1ContractError",
    "TrajectoryBundle",
    "TrajectoryTurn",
    "UserTrajectory",
    "build_query_selection",
    "build_core_r2_stage1_trajectory_bundle",
    "build_s1_creator_packet",
    "build_trajectory_bundle",
    "load_query_selection",
    "load_s1_creator_packet",
    "load_trajectory_bundle",
    "prepare_stage1",
]
