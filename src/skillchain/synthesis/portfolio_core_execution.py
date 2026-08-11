"""Recoverable, mechanical r2 author-draft execution without model calls.

This module is deliberately separate from the legacy human-acceptance batch
wrapper.  It accepts only a strongly bound r2 authoring job and an equally
bound 25-record draft, materialises schema-v2 ``Query`` records from the r2
final-split sidecar, and records the owner's approved mechanical
auto-approval policy in an isolated runtime root.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Mapping, Self, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.synthesis.batches import normalized_text
from skillchain.synthesis.models import GeneratedTrajectory, PlannedQuery
from skillchain.synthesis.planning import R2CoreInMemoryPlan
from skillchain.synthesis.portfolio_core_authoring import (
    AuthoringJob,
    canonical_author_packet_bytes,
    core_r2_realism_quota_spec,
    default_prompt_recipes,
    validate_authoring_job,
    validate_realism_sidecar,
    work_order_sha256,
)
from skillchain.synthesis.portfolio_core_r2 import (
    R2CoreBridge,
    build_r2_split_constraints,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    atomic_replace_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
EXECUTION_POLICY_VERSION = "portfolio-core-r2-mechanical-auto-approval-v2"
_REPARSE_POINT = 0x0400
_TARGET_QUERY_COUNT = 1500
_BATCH_SIZE = 25
_ROOT_FILES = {"accepted-ledger.jsonl", "runtime-checkpoint.json"}
_ROOT_DIRECTORIES = {"accepted", "staging"}
_BATCH_FILES = {
    "author-draft-receipt.json",
    "manifest.json",
    "results.jsonl",
}
_DRAFT_ARTIFACT_FILES = {"drafts.jsonl", "manifest.json", "receipt.json"}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LEGACY_DEV_MINI_PLAN_IDS = tuple(f"dm-{index:03d}" for index in range(1, 201))


class PortfolioCoreExecutionError(ValueError):
    """An r2 execution input or persisted runtime state is unsafe."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must be non-blank")
    return value


def _require_identifier(value: str, label: str) -> str:
    value = _nonblank(value, label)
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def canonical_turns_sha256(turns: Sequence[ConversationTurn]) -> str:
    """Return the canonical SHA-256 for a sequence of typed conversation turns."""

    if not isinstance(turns, (tuple, list)) or not turns:
        raise ValueError("turns must be a non-empty tuple or list")
    if any(not isinstance(turn, ConversationTurn) for turn in turns):
        raise ValueError("turns must contain only ConversationTurn records")
    return sha256_bytes(
        canonical_json_bytes([turn.model_dump(mode="json") for turn in turns])
    )


def build_legacy_turn_exclusion_receipt(
    *,
    legacy_artifact_sha256: str,
    turns_by_plan_id: Mapping[str, Sequence[ConversationTurn]],
) -> "LegacyTurnExclusionReceipt":
    """Build a body-free receipt from the already accepted dev-mini turns.

    The caller is responsible for calculating ``legacy_artifact_sha256`` from
    the immutable accepted dev artifact.  This pure helper never writes the
    artifact or the receipt and deliberately omits all turn content.
    """

    if set(turns_by_plan_id) != set(_LEGACY_DEV_MINI_PLAN_IDS):
        raise ValueError("legacy turns must contain exactly dm-001 through dm-200")
    turn_digests = tuple(
        LegacyTurnDigest(
            plan_id=plan_id,
            canonical_turns_sha256=canonical_turns_sha256(turns_by_plan_id[plan_id]),
        )
        for plan_id in _LEGACY_DEV_MINI_PLAN_IDS
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "legacy_artifact_sha256": legacy_artifact_sha256,
        "plan_ids": _LEGACY_DEV_MINI_PLAN_IDS,
        "turn_digests": [item.model_dump(mode="json") for item in turn_digests],
    }
    return LegacyTurnExclusionReceipt.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


class R2AuthorDraftManifest(_StrictModel):
    """The only draft manifest accepted by the r2 mechanical executor."""

    schema_version: Literal[1] = 1
    data_origin: Literal["synthetic_derived"] = "synthetic_derived"
    base_batch_id: str
    job_id: str
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    generation_input_sha256: Sha256
    author_packet_sha256: Sha256
    work_order_sha256: Sha256
    plan_ids_sha256: Sha256
    draft_sha256: Sha256

    @field_validator("base_batch_id", "job_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @field_validator("generated_at")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value


class LegacyTurnDigest(_StrictModel):
    """One body-free canonical-turn hash from the accepted legacy dev corpus."""

    plan_id: str
    canonical_turns_sha256: Sha256

    @field_validator("plan_id")
    @classmethod
    def _validate_plan_id(cls, value: str) -> str:
        return _require_identifier(value, "plan_id")


class LegacyTurnExclusionReceipt(_StrictModel):
    """Self-hashed body-free receipt excluding accepted dev-mini turns.

    This is an integrity receipt, not a claim that the historical corpus or a
    model provider has been cryptographically attested.  It deliberately
    persists only deterministic hashes of turn sequences.
    """

    schema_version: Literal[1] = 1
    legacy_artifact_sha256: Sha256
    plan_ids: tuple[str, ...]
    turn_digests: tuple[LegacyTurnDigest, ...]
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.plan_ids != _LEGACY_DEV_MINI_PLAN_IDS:
            raise ValueError(
                "legacy receipt must contain exactly dm-001 through dm-200"
            )
        if tuple(item.plan_id for item in self.turn_digests) != self.plan_ids:
            raise ValueError(
                "legacy receipt turn digests must follow the exact plan IDs"
            )
        if self.receipt_sha256 != _digest_without(
            self.model_dump(mode="json"), "receipt_sha256"
        ):
            raise ValueError("legacy turn exclusion receipt SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class R2AuthorDraftReceipt(_StrictModel):
    """Self-hashed receipt for a create-only r2 author-draft artifact.

    The receipt intentionally contains identifiers and hashes only.  The
    separately bound draft manifest retains workflow-declared provider/model
    fields, which are not a cryptographic model identity proof.
    """

    schema_version: Literal[1] = 1
    base_batch_id: str
    job_id: str
    count: Literal[25] = _BATCH_SIZE
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    generation_input_sha256: Sha256
    author_packet_sha256: Sha256
    work_order_sha256: Sha256
    plan_ids_sha256: Sha256
    draft_manifest_sha256: Sha256
    draft_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("base_batch_id", "job_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.receipt_sha256 != _digest_without(
            self.model_dump(mode="json"), "receipt_sha256"
        ):
            raise ValueError("author draft receipt SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class R2ExecutionBatchManifest(_StrictModel):
    """Immutable receipt for one mechanically accepted r2 author batch."""

    schema_version: Literal[1] = 1
    execution_policy_version: Literal[
        "portfolio-core-r2-mechanical-auto-approval-v2"
    ] = EXECUTION_POLICY_VERSION
    base_batch_id: str
    job_id: str
    count: Literal[25] = _BATCH_SIZE
    data_origin: Literal["synthetic_derived"] = "synthetic_derived"
    provider: Literal["codex"]
    model_display_name: Literal["5.6 Sol Ultra"]
    model_claim_source: Literal["user_confirmation"]
    generated_at: datetime
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    generation_input_sha256: Sha256
    author_packet_sha256: Sha256
    work_order_sha256: Sha256
    plan_ids_sha256: Sha256
    draft_sha256: Sha256
    legacy_turn_exclusion_receipt_sha256: Sha256
    author_draft_receipt_sha256: Sha256
    results_sha256: Sha256
    manifest_sha256: Sha256

    @field_validator("base_batch_id", "job_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @field_validator("generated_at")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.manifest_sha256 != _digest_without(
            self.model_dump(mode="json"), "manifest_sha256"
        ):
            raise ValueError("execution batch manifest SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class R2ExecutionLedgerEntry(_StrictModel):
    """One immutable prefix record in the independent execution ledger."""

    schema_version: Literal[1] = 1
    base_batch_id: str
    job_id: str
    count: Literal[25] = _BATCH_SIZE
    results_sha256: Sha256
    batch_manifest_sha256: Sha256
    plan_sha256: Sha256
    legacy_turn_exclusion_receipt_sha256: Sha256
    author_draft_receipt_sha256: Sha256

    @field_validator("base_batch_id", "job_id")
    @classmethod
    def _validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)


class R2ExecutionCheckpoint(_StrictModel):
    """Deterministic runtime state, derived solely from the accepted ledger."""

    schema_version: Literal[1] = 1
    execution_policy_version: Literal[
        "portfolio-core-r2-mechanical-auto-approval-v2"
    ] = EXECUTION_POLICY_VERSION
    target_query_count: Literal[1500] = _TARGET_QUERY_COUNT
    batch_size: Literal[25] = _BATCH_SIZE
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    final_split_sidecar_sha256: Sha256
    legacy_turn_exclusion_receipt_sha256: Sha256
    accepted_batch_ids: tuple[str, ...] = ()
    accepted_query_count: int = Field(ge=0, le=_TARGET_QUERY_COUNT)
    status: Literal[
        "ready",
        "running",
        "auto_approved_usable_pending_sample_review",
    ]
    accepted_ledger_sha256: Sha256
    checkpoint_sha256: Sha256

    @field_validator("accepted_batch_ids")
    @classmethod
    def _validate_batch_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for batch_id in value:
            _require_identifier(batch_id, "accepted_batch_id")
        if len(value) != len(set(value)):
            raise ValueError("accepted_batch_ids must be unique")
        return value

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        if self.accepted_query_count != len(self.accepted_batch_ids) * _BATCH_SIZE:
            raise ValueError("accepted_query_count does not match accepted batch count")
        expected_status = (
            "ready"
            if self.accepted_query_count == 0
            else "auto_approved_usable_pending_sample_review"
            if self.accepted_query_count == _TARGET_QUERY_COUNT
            else "running"
        )
        if self.status != expected_status:
            raise ValueError("runtime status does not match accepted query count")
        if self.checkpoint_sha256 != _digest_without(
            self.model_dump(mode="json"), "checkpoint_sha256"
        ):
            raise ValueError("execution checkpoint SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class R2ExecutionResult:
    """Result of an idempotent mechanical r2 batch execution."""

    accepted_dir: Path
    created: bool
    checkpoint: R2ExecutionCheckpoint


@dataclass(frozen=True)
class R2PublishedAuthorDraft:
    """A revalidated create-only author-draft directory."""

    directory: Path
    drafts: tuple[GeneratedTrajectory, ...]
    draft_manifest: R2AuthorDraftManifest
    receipt: R2AuthorDraftReceipt


@dataclass(frozen=True)
class _ExecutionContext:
    plan: R2CoreInMemoryPlan
    bridge: R2CoreBridge
    trusted_plan_sha256: str
    plan_rows: tuple[PlannedQuery, ...]
    rows_by_batch: dict[str, tuple[PlannedQuery, ...]]
    rows_by_plan_id: dict[str, PlannedQuery]
    final_split_by_plan_id: dict[str, str]
    realism_by_plan_id: dict[str, object]
    batch_ids: tuple[str, ...]
    legacy_turn_exclusion_receipt_sha256: str | None
    legacy_turns_sha256_by_plan_id: dict[str, str]


@dataclass(frozen=True)
class _ExecutionPayload:
    batch_manifest: R2ExecutionBatchManifest
    results: tuple[Query, ...]
    results_bytes: bytes
    manifest_bytes: bytes
    author_draft_receipt_bytes: bytes
    ledger_entry: R2ExecutionLedgerEntry


@dataclass(frozen=True)
class _RuntimeState:
    entries: tuple[R2ExecutionLedgerEntry, ...]
    checkpoint: R2ExecutionCheckpoint
    checkpoint_bytes: bytes
    accepted_texts: frozenset[str]
    checkpoint_lags_ledger: bool
    ledger_pending_promotion_batch_id: str | None
    uncommitted_staging_batch_id: str | None


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to inspect {label}") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise PortfolioCoreExecutionError(
            f"{label} must be a non-reparse real directory"
        )


def _require_safe_absolute_directory_chain(path: Path, label: str) -> None:
    if not path.is_absolute() or not path.anchor:
        raise PortfolioCoreExecutionError(f"{label} must be an absolute directory")
    anchor = Path(path.anchor)
    _require_real_directory(anchor, f"{label} anchor")
    current = anchor
    for part in path.parts[1:]:
        if part in {".", ".."}:
            raise PortfolioCoreExecutionError(f"{label} contains a traversal component")
        current = current / part
        _require_real_directory(current, label)


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to inspect {label}") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise PortfolioCoreExecutionError(f"{label} must be a regular non-link file")
    return metadata


def _safe_read(path: Path, label: str) -> bytes:
    before = _require_regular_file(path, label)
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ):
                raise PortfolioCoreExecutionError(f"{label} changed before reading")
            content = handle.read()
            after_read = os.fstat(handle.fileno())
        after = _require_regular_file(path, label)
    except PortfolioCoreExecutionError:
        raise
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to read {label}") from exc
    snapshots = (opened, after_read, after)
    first = (
        snapshots[0].st_dev,
        snapshots[0].st_ino,
        snapshots[0].st_size,
        snapshots[0].st_mtime_ns,
    )
    if (
        any(
            (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) != first
            for item in snapshots[1:]
        )
        or len(content) != after.st_size
    ):
        raise PortfolioCoreExecutionError(f"{label} changed during reading")
    return content


def _read_model(path: Path, model_type: type[_StrictModel], label: str) -> _StrictModel:
    content = _safe_read(path, label)
    try:
        model = model_type.model_validate_json(content)
    except ValidationError as exc:
        raise PortfolioCoreExecutionError(f"{label} violates its schema") from exc
    if canonical_json_bytes(model) != content:
        raise PortfolioCoreExecutionError(f"{label} is not canonical JSON")
    return model


def _parse_queries(content: bytes, label: str) -> tuple[Query, ...]:
    if not content:
        raise PortfolioCoreExecutionError(f"{label} must be non-empty")
    records: list[Query] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise PortfolioCoreExecutionError(f"{label} contains a blank line")
        try:
            records.append(Query.model_validate_json(line))
        except ValidationError as exc:
            raise PortfolioCoreExecutionError(
                f"{label} line {line_number} violates Query schema"
            ) from exc
    parsed = tuple(records)
    if canonical_jsonl_bytes(parsed) != content:
        raise PortfolioCoreExecutionError(f"{label} is not canonical Query JSONL")
    return parsed


def _parse_generated_trajectories(
    content: bytes, label: str
) -> tuple[GeneratedTrajectory, ...]:
    if not content:
        raise PortfolioCoreExecutionError(f"{label} must be non-empty")
    records: list[GeneratedTrajectory] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise PortfolioCoreExecutionError(f"{label} contains a blank line")
        try:
            records.append(GeneratedTrajectory.model_validate_json(line))
        except ValidationError as exc:
            raise PortfolioCoreExecutionError(
                f"{label} line {line_number} violates GeneratedTrajectory schema"
            ) from exc
    parsed = tuple(records)
    if canonical_jsonl_bytes(parsed) != content:
        raise PortfolioCoreExecutionError(
            f"{label} is not canonical GeneratedTrajectory JSONL"
        )
    return parsed


def _parse_ledger(content: bytes) -> tuple[R2ExecutionLedgerEntry, ...]:
    if not content:
        return ()
    entries: list[R2ExecutionLedgerEntry] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise PortfolioCoreExecutionError("accepted ledger contains a blank line")
        try:
            entries.append(R2ExecutionLedgerEntry.model_validate_json(line))
        except ValidationError as exc:
            raise PortfolioCoreExecutionError(
                f"accepted ledger line {line_number} violates its schema"
            ) from exc
    parsed = tuple(entries)
    if canonical_jsonl_bytes(parsed) != content:
        raise PortfolioCoreExecutionError("accepted ledger is not canonical JSONL")
    return parsed


def _plan_sha256(plan: R2CoreInMemoryPlan) -> str:
    return sha256_bytes(canonical_json_bytes(plan.plan))


def _batch_rows(plan: R2CoreInMemoryPlan) -> dict[str, tuple[PlannedQuery, ...]]:
    grouped: dict[str, list[PlannedQuery]] = {}
    for row in plan.plan.queries:
        grouped.setdefault(row.batch_id, []).append(row)
    result = {
        batch_id: tuple(sorted(rows, key=lambda row: row.position))
        for batch_id, rows in grouped.items()
    }
    if any(
        len(rows) != _BATCH_SIZE
        or [row.position for row in rows] != list(range(1, _BATCH_SIZE + 1))
        for rows in result.values()
    ):
        raise PortfolioCoreExecutionError("r2 plan batch layout is not complete")
    return result


def _legacy_receipt_binding(
    receipt: LegacyTurnExclusionReceipt | None,
) -> tuple[str | None, dict[str, str]]:
    if receipt is None:
        return None, {}
    if not isinstance(receipt, LegacyTurnExclusionReceipt):
        raise PortfolioCoreExecutionError(
            "legacy_turn_exclusion_receipt must be a LegacyTurnExclusionReceipt"
        )
    try:
        validated = LegacyTurnExclusionReceipt.model_validate(
            receipt.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise PortfolioCoreExecutionError(
            "legacy turn exclusion receipt validation failed"
        ) from exc
    canonical = validated.canonical_bytes()
    return sha256_bytes(canonical), {
        item.plan_id: item.canonical_turns_sha256 for item in validated.turn_digests
    }


def _validate_context(
    plan: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
    legacy_turn_exclusion_receipt: LegacyTurnExclusionReceipt | None = None,
) -> _ExecutionContext:
    if not isinstance(plan, R2CoreInMemoryPlan):
        raise PortfolioCoreExecutionError("plan must be an R2CoreInMemoryPlan")
    if not isinstance(bridge, R2CoreBridge):
        raise PortfolioCoreExecutionError("bridge must be an R2CoreBridge")
    actual_plan_sha256 = _plan_sha256(plan)
    if actual_plan_sha256 != trusted_plan_sha256:
        raise PortfolioCoreExecutionError(
            "trusted plan SHA-256 does not match the r2 in-memory plan"
        )
    try:
        expected_constraints = build_r2_split_constraints(
            plan,
            trusted_plan_sha256=trusted_plan_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioCoreExecutionError(
            "r2 plan audit or split binding failed"
        ) from exc
    if bridge.split_constraints != expected_constraints:
        raise PortfolioCoreExecutionError("r2 bridge split constraints drifted")
    if plan.plan.scope != "core" or len(plan.plan.queries) != _TARGET_QUERY_COUNT:
        raise PortfolioCoreExecutionError(
            "r2 execution requires the full 1,500-query core plan"
        )

    sidecar = bridge.realism_sidecar
    expected_recipes = tuple(
        sorted(default_prompt_recipes(), key=lambda recipe: recipe.prompt_recipe_id)
    )
    if sidecar.recipes != expected_recipes:
        raise PortfolioCoreExecutionError("r2 bridge realism recipe library drifted")
    expected_manifest_values = {
        "plan_sha256": trusted_plan_sha256,
        "asset_catalog_sha256": plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": plan.plan.capability_assignments_sha256,
    }
    if any(
        getattr(sidecar.manifest, field_name) != expected
        for field_name, expected in expected_manifest_values.items()
    ):
        raise PortfolioCoreExecutionError("r2 bridge realism manifest binding drifted")
    final_splits = dict(bridge.split_constraints.plan_id_to_split)
    if final_splits != dict(plan.final_split_by_plan_id):
        raise PortfolioCoreExecutionError("r2 bridge final-split sidecar drifted")
    reuse = {
        plan_id: {
            "plan_id": plan_id,
            "reuse_variant": plan.reuse_variant_by_plan_id[plan_id],
            "reuse_reason": plan.reuse_reason_by_plan_id[plan_id],
        }
        for plan_id in final_splits
    }
    try:
        validate_realism_sidecar(
            sidecar,
            plan_rows=plan.plan.queries,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=reuse,
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioCoreExecutionError("r2 bridge realism sidecar drifted") from exc
    realism_by_plan_id = {item.plan_id: item for item in sidecar.assignments}
    expected_val_interactions = {
        plan_id: realism_by_plan_id[plan_id].interaction_pattern
        for plan_id, split in final_splits.items()
        if split == "val"
    }
    if dict(bridge.val_interaction_by_query_id) != expected_val_interactions:
        raise PortfolioCoreExecutionError(
            "r2 bridge validation interaction binding drifted"
        )

    rows = tuple(plan.plan.queries)
    rows_by_batch = _batch_rows(plan)
    batch_ids = tuple(dict.fromkeys(row.batch_id for row in rows))
    if len(batch_ids) != _TARGET_QUERY_COUNT // _BATCH_SIZE:
        raise PortfolioCoreExecutionError("r2 plan does not contain exactly 60 batches")
    receipt_sha256, legacy_turns_sha256_by_plan_id = _legacy_receipt_binding(
        legacy_turn_exclusion_receipt
    )
    return _ExecutionContext(
        plan=plan,
        bridge=bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        plan_rows=rows,
        rows_by_batch=rows_by_batch,
        rows_by_plan_id={row.plan_id: row for row in rows},
        final_split_by_plan_id=final_splits,
        realism_by_plan_id=realism_by_plan_id,
        batch_ids=batch_ids,
        legacy_turn_exclusion_receipt_sha256=receipt_sha256,
        legacy_turns_sha256_by_plan_id=legacy_turns_sha256_by_plan_id,
    )


def _require_issued_authoring_job(job: AuthoringJob) -> None:
    if not isinstance(job, AuthoringJob):
        raise PortfolioCoreExecutionError("authoring_job must be an AuthoringJob")
    try:
        validate_authoring_job(job)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PortfolioCoreExecutionError("authoring job validation failed") from exc
    if job.checkpoint.state != "issued":
        raise PortfolioCoreExecutionError(
            "mechanical execution requires an immutable issued authoring job"
        )


def _job_binding(
    context: _ExecutionContext,
    job: AuthoringJob,
) -> tuple[PlannedQuery, ...]:
    _require_issued_authoring_job(job)
    order = job.work_order
    if order.base_batch_id not in context.rows_by_batch:
        raise PortfolioCoreExecutionError(
            "authoring job batch is absent from the r2 plan"
        )
    expected_rows = context.rows_by_batch[order.base_batch_id]
    expected_hashes = {
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": context.plan.plan.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(context.bridge.realism_sidecar.manifest)
        ),
        "realism_assignments_sha256": context.bridge.realism_sidecar.manifest.assignments_sha256,
        "final_split_sidecar_sha256": context.bridge.realism_sidecar.manifest.final_split_sidecar_sha256,
    }
    for field_name, expected in expected_hashes.items():
        if getattr(order, field_name) != expected:
            raise PortfolioCoreExecutionError(
                f"authoring job binding drifted: {field_name}"
            )
    if order.realism_manifest != context.bridge.realism_sidecar.manifest:
        raise PortfolioCoreExecutionError("authoring job realism manifest drifted")
    if order.recipe_manifest != context.bridge.realism_sidecar.recipe_manifest:
        raise PortfolioCoreExecutionError("authoring job recipe manifest drifted")
    if tuple(item.plan_id for item in order.items) != tuple(
        row.plan_id for row in expected_rows
    ):
        raise PortfolioCoreExecutionError(
            "authoring job plan IDs are not in r2 batch order"
        )
    for item, row in zip(order.items, expected_rows, strict=True):
        expected_assignment = context.realism_by_plan_id[row.plan_id]
        expected_values = (
            row.batch_id,
            row.position,
            row.asset_id,
            row.image_path,
            row.leakage_group_id,
            row.capability_assignment_id,
            row.capability_assignment_source_sha256,
            row.canonical_intent,
            row.canonical_capability,
            row.is_boundary,
            row.boundary_strategy,
            context.final_split_by_plan_id[row.plan_id],
            context.plan.reuse_variant_by_plan_id[row.plan_id],
            context.plan.reuse_reason_by_plan_id[row.plan_id],
            expected_assignment,
        )
        actual_values = (
            item.batch_id,
            item.position,
            item.asset_id,
            item.image_path,
            item.leakage_group_id,
            item.capability_assignment_id,
            item.capability_assignment_source_sha256,
            item.canonical_intent,
            item.canonical_capability,
            item.is_boundary,
            item.boundary_strategy,
            item.final_split,
            item.reuse_variant,
            item.reuse_reason,
            item.realism,
        )
        if actual_values != expected_values:
            raise PortfolioCoreExecutionError(
                f"authoring job item drifted from r2 bindings: {row.plan_id}"
            )
    return expected_rows


def _expected_plan_ids_sha256(rows: tuple[PlannedQuery, ...]) -> str:
    return sha256_bytes(canonical_json_bytes([row.plan_id for row in rows]))


def _draft_manifest_bytes(manifest: R2AuthorDraftManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json"))


def _validate_draft_manifest_against_job(
    job: AuthoringJob,
    drafts: object,
    draft_manifest: object,
) -> tuple[tuple[GeneratedTrajectory, ...], R2AuthorDraftManifest, bytes]:
    """Validate the author-side binding before publication or execution."""

    _require_issued_authoring_job(job)
    if not isinstance(draft_manifest, R2AuthorDraftManifest):
        raise PortfolioCoreExecutionError(
            "r2 execution requires an R2AuthorDraftManifest with author-job bindings"
        )
    try:
        manifest = R2AuthorDraftManifest.model_validate(
            draft_manifest.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise PortfolioCoreExecutionError("draft manifest validation failed") from exc
    if not isinstance(drafts, (list, tuple)) or len(drafts) != _BATCH_SIZE:
        raise PortfolioCoreExecutionError(
            "r2 execution drafts must contain exactly 25 records"
        )
    if any(not isinstance(draft, GeneratedTrajectory) for draft in drafts):
        raise PortfolioCoreExecutionError(
            "r2 execution drafts must be GeneratedTrajectory records"
        )
    draft_tuple = tuple(drafts)
    order = job.work_order
    actual_ids = tuple(draft.plan_id for draft in draft_tuple)
    expected_ids = tuple(item.plan_id for item in order.items)
    if actual_ids != expected_ids:
        raise PortfolioCoreExecutionError(
            "draft plan IDs must exactly match r2 batch order"
        )
    draft_bytes = canonical_jsonl_bytes(draft_tuple)
    expected_bindings = {
        "base_batch_id": order.base_batch_id,
        "job_id": order.job_id,
        "plan_sha256": order.plan_sha256,
        "asset_catalog_sha256": order.asset_catalog_sha256,
        "capability_assignments_sha256": order.capability_assignments_sha256,
        "realism_manifest_sha256": order.realism_manifest_sha256,
        "realism_assignments_sha256": order.realism_assignments_sha256,
        "final_split_sidecar_sha256": order.final_split_sidecar_sha256,
        "generation_input_sha256": order.generation_input_sha256,
        "author_packet_sha256": sha256_bytes(
            canonical_author_packet_bytes(job.author_packet)
        ),
        "work_order_sha256": work_order_sha256(order),
        "plan_ids_sha256": sha256_bytes(canonical_json_bytes(list(expected_ids))),
        "draft_sha256": sha256_bytes(draft_bytes),
    }
    for field_name, expected in expected_bindings.items():
        if getattr(manifest, field_name) != expected:
            raise PortfolioCoreExecutionError(
                f"draft manifest binding drifted: {field_name}"
            )
    return draft_tuple, manifest, draft_bytes


def build_r2_author_draft_manifest(
    plan: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
    authoring_job: AuthoringJob,
    drafts: tuple[GeneratedTrajectory, ...] | list[GeneratedTrajectory],
    generated_at: datetime,
) -> R2AuthorDraftManifest:
    """Purely build the strongly bound manifest for one r2 author draft.

    This helper validates the audited plan/bridge and the issued authoring job,
    but writes neither a draft nor a receipt.  ``generated_at`` must be aware.
    Model/provider values remain workflow declarations rather than a
    cryptographic provenance claim.
    """

    context = _validate_context(
        plan,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    rows = _job_binding(context, authoring_job)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise PortfolioCoreExecutionError("generated_at must include a timezone")
    if not isinstance(drafts, (list, tuple)) or len(drafts) != _BATCH_SIZE:
        raise PortfolioCoreExecutionError(
            "r2 execution drafts must contain exactly 25 records"
        )
    if any(not isinstance(draft, GeneratedTrajectory) for draft in drafts):
        raise PortfolioCoreExecutionError(
            "r2 execution drafts must be GeneratedTrajectory records"
        )
    draft_tuple = tuple(drafts)
    if tuple(draft.plan_id for draft in draft_tuple) != tuple(
        row.plan_id for row in rows
    ):
        raise PortfolioCoreExecutionError(
            "draft plan IDs must exactly match r2 batch order"
        )
    order = authoring_job.work_order
    return R2AuthorDraftManifest(
        base_batch_id=order.base_batch_id,
        job_id=order.job_id,
        provider="codex",
        model_display_name="5.6 Sol Ultra",
        model_claim_source="user_confirmation",
        generated_at=generated_at,
        plan_sha256=context.trusted_plan_sha256,
        asset_catalog_sha256=context.plan.plan.asset_catalog_sha256,
        capability_assignments_sha256=context.plan.plan.capability_assignments_sha256,
        realism_manifest_sha256=sha256_bytes(
            canonical_json_bytes(context.bridge.realism_sidecar.manifest)
        ),
        realism_assignments_sha256=(
            context.bridge.realism_sidecar.manifest.assignments_sha256
        ),
        final_split_sidecar_sha256=(
            context.bridge.realism_sidecar.manifest.final_split_sidecar_sha256
        ),
        generation_input_sha256=order.generation_input_sha256,
        author_packet_sha256=sha256_bytes(
            canonical_author_packet_bytes(authoring_job.author_packet)
        ),
        work_order_sha256=work_order_sha256(order),
        plan_ids_sha256=_expected_plan_ids_sha256(rows),
        draft_sha256=sha256_bytes(canonical_jsonl_bytes(draft_tuple)),
    )


def _author_draft_receipt(
    job: AuthoringJob,
    drafts: object,
    draft_manifest: object,
) -> tuple[
    R2AuthorDraftReceipt, tuple[GeneratedTrajectory, ...], R2AuthorDraftManifest
]:
    draft_tuple, manifest, draft_bytes = _validate_draft_manifest_against_job(
        job, drafts, draft_manifest
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "base_batch_id": manifest.base_batch_id,
        "job_id": manifest.job_id,
        "count": _BATCH_SIZE,
        "plan_sha256": manifest.plan_sha256,
        "asset_catalog_sha256": manifest.asset_catalog_sha256,
        "capability_assignments_sha256": manifest.capability_assignments_sha256,
        "realism_manifest_sha256": manifest.realism_manifest_sha256,
        "realism_assignments_sha256": manifest.realism_assignments_sha256,
        "final_split_sidecar_sha256": manifest.final_split_sidecar_sha256,
        "generation_input_sha256": manifest.generation_input_sha256,
        "author_packet_sha256": manifest.author_packet_sha256,
        "work_order_sha256": manifest.work_order_sha256,
        "plan_ids_sha256": manifest.plan_ids_sha256,
        "draft_manifest_sha256": sha256_bytes(_draft_manifest_bytes(manifest)),
        "draft_sha256": sha256_bytes(draft_bytes),
    }
    return (
        R2AuthorDraftReceipt.model_validate(
            {
                **payload,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
            }
        ),
        draft_tuple,
        manifest,
    )


def _expected_roles(assignment: object) -> tuple[str, ...]:
    shape = getattr(assignment, "trajectory_shape", None)
    interaction = getattr(assignment, "interaction_pattern", None)
    if interaction == "direct_request" and shape != "single_turn":
        raise PortfolioCoreExecutionError("realism direct_request shape drifted")
    if interaction != "direct_request" and shape != "three_turn":
        raise PortfolioCoreExecutionError(
            "realism non-direct interaction shape drifted"
        )
    if shape == "single_turn":
        return ("user",)
    if shape == "three_turn":
        return ("user", "assistant", "user")
    raise PortfolioCoreExecutionError("realism trajectory shape is unknown")


def _materialize_query(
    context: _ExecutionContext,
    row: PlannedQuery,
    turns: list,
) -> Query:
    assignment = context.realism_by_plan_id[row.plan_id]
    roles = tuple(turn.role for turn in turns)
    if roles != _expected_roles(assignment):
        raise PortfolioCoreExecutionError(
            f"draft turn roles do not match realism assignment: {row.plan_id}"
        )
    legacy_turns_sha256 = context.legacy_turns_sha256_by_plan_id.get(row.plan_id)
    if (
        legacy_turns_sha256 is not None
        and canonical_turns_sha256(turns) == legacy_turns_sha256
    ):
        raise PortfolioCoreExecutionError(
            f"draft reuses excluded legacy canonical turns: {row.plan_id}"
        )
    return Query(
        schema_version=2,
        taxonomy_version=row.taxonomy_version,
        task_spec_version=row.task_spec_version,
        query_id=row.plan_id,
        asset_id=row.asset_id,
        image_path=row.image_path,
        leakage_group_id=row.leakage_group_id,
        boundary_group_id=row.boundary_group_id,
        template_family=row.template_family,
        generator_batch_id=row.generator_batch_id,
        text=turns[-1].content,
        turns=turns,
        canonical_intent=row.canonical_intent,
        canonical_capability=row.canonical_capability,
        acceptable_capabilities=row.acceptable_capabilities,
        is_boundary=row.is_boundary,
        boundary_strategy=row.boundary_strategy,
        requires_card=row.requires_card,
        episode="t0",
        split=context.final_split_by_plan_id[row.plan_id],
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="portfolio-core-r2-execution-v1",
                canonical_intent=row.canonical_intent,
                canonical_capability=row.canonical_capability,
                acceptable_capabilities=row.acceptable_capabilities,
                reason="label copied from the audited r2 plan and final-split sidecar",
                source_artifact_sha256=context.trusted_plan_sha256,
            )
        ],
    )


def _build_payload(
    context: _ExecutionContext,
    job: AuthoringJob,
    published_draft: R2PublishedAuthorDraft,
) -> _ExecutionPayload:
    if context.legacy_turn_exclusion_receipt_sha256 is None:
        raise PortfolioCoreExecutionError(
            "mechanical execution requires a legacy turn exclusion receipt"
        )
    if not isinstance(published_draft, R2PublishedAuthorDraft):
        raise PortfolioCoreExecutionError(
            "r2 execution requires a reverified published author draft"
        )
    rows = _job_binding(context, job)
    receipt, draft_tuple, draft_manifest = _author_draft_receipt(
        job, published_draft.drafts, published_draft.draft_manifest
    )
    if receipt != published_draft.receipt:
        raise PortfolioCoreExecutionError("published author draft receipt drifted")
    if tuple(draft.plan_id for draft in draft_tuple) != tuple(
        row.plan_id for row in rows
    ):
        raise PortfolioCoreExecutionError(
            "draft plan IDs must exactly match r2 batch order"
        )
    draft_bytes = canonical_jsonl_bytes(draft_tuple)
    order = job.work_order
    expected_bindings = {
        "base_batch_id": order.base_batch_id,
        "job_id": order.job_id,
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": context.plan.plan.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(context.bridge.realism_sidecar.manifest)
        ),
        "realism_assignments_sha256": context.bridge.realism_sidecar.manifest.assignments_sha256,
        "final_split_sidecar_sha256": context.bridge.realism_sidecar.manifest.final_split_sidecar_sha256,
        "generation_input_sha256": order.generation_input_sha256,
        "author_packet_sha256": sha256_bytes(
            canonical_author_packet_bytes(job.author_packet)
        ),
        "work_order_sha256": work_order_sha256(order),
        "plan_ids_sha256": _expected_plan_ids_sha256(rows),
        "draft_sha256": sha256_bytes(draft_bytes),
    }
    for field_name, expected in expected_bindings.items():
        if getattr(draft_manifest, field_name) != expected:
            raise PortfolioCoreExecutionError(
                f"draft manifest binding drifted: {field_name}"
            )
    queries = tuple(
        _materialize_query(context, row, draft.turns)
        for row, draft in zip(rows, draft_tuple, strict=True)
    )
    normalized = [normalized_text(query.text) for query in queries]
    if any(not value for value in normalized):
        raise PortfolioCoreExecutionError(
            "draft contains an empty normalized final user text"
        )
    if len(normalized) != len(set(normalized)):
        raise PortfolioCoreExecutionError(
            "draft contains duplicate normalized final user text"
        )
    results_bytes = canonical_jsonl_bytes(queries)
    author_draft_receipt_bytes = receipt.canonical_bytes()
    manifest_payload: dict[str, object] = {
        "schema_version": 1,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "base_batch_id": draft_manifest.base_batch_id,
        "job_id": draft_manifest.job_id,
        "count": _BATCH_SIZE,
        "data_origin": draft_manifest.data_origin,
        "provider": draft_manifest.provider,
        "model_display_name": draft_manifest.model_display_name,
        "model_claim_source": draft_manifest.model_claim_source,
        "generated_at": draft_manifest.model_dump(mode="json")["generated_at"],
        **expected_bindings,
        "legacy_turn_exclusion_receipt_sha256": (
            context.legacy_turn_exclusion_receipt_sha256
        ),
        "author_draft_receipt_sha256": sha256_bytes(author_draft_receipt_bytes),
        "results_sha256": sha256_bytes(results_bytes),
    }
    try:
        manifest = R2ExecutionBatchManifest.model_validate(
            {
                **manifest_payload,
                "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_payload)),
            }
        )
    except ValidationError as exc:  # pragma: no cover - internal construction guard
        raise PortfolioCoreExecutionError(
            "execution batch manifest is invalid"
        ) from exc
    manifest_bytes = manifest.canonical_bytes()
    return _ExecutionPayload(
        batch_manifest=manifest,
        results=queries,
        results_bytes=results_bytes,
        manifest_bytes=manifest_bytes,
        author_draft_receipt_bytes=author_draft_receipt_bytes,
        ledger_entry=R2ExecutionLedgerEntry(
            base_batch_id=manifest.base_batch_id,
            job_id=manifest.job_id,
            results_sha256=manifest.results_sha256,
            batch_manifest_sha256=sha256_bytes(manifest_bytes),
            plan_sha256=manifest.plan_sha256,
            legacy_turn_exclusion_receipt_sha256=(
                manifest.legacy_turn_exclusion_receipt_sha256
            ),
            author_draft_receipt_sha256=manifest.author_draft_receipt_sha256,
        ),
    )


def _checkpoint(
    context: _ExecutionContext,
    entries: tuple[R2ExecutionLedgerEntry, ...],
) -> R2ExecutionCheckpoint:
    if context.legacy_turn_exclusion_receipt_sha256 is None:
        raise PortfolioCoreExecutionError(
            "mechanical execution requires a legacy turn exclusion receipt"
        )
    accepted_query_count = len(entries) * _BATCH_SIZE
    payload: dict[str, object] = {
        "schema_version": 1,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "target_query_count": _TARGET_QUERY_COUNT,
        "batch_size": _BATCH_SIZE,
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": context.plan.plan.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(context.bridge.realism_sidecar.manifest)
        ),
        "realism_assignments_sha256": context.bridge.realism_sidecar.manifest.assignments_sha256,
        "final_split_sidecar_sha256": context.bridge.realism_sidecar.manifest.final_split_sidecar_sha256,
        "legacy_turn_exclusion_receipt_sha256": (
            context.legacy_turn_exclusion_receipt_sha256
        ),
        "accepted_batch_ids": tuple(entry.base_batch_id for entry in entries),
        "accepted_query_count": accepted_query_count,
        "status": (
            "ready"
            if accepted_query_count == 0
            else "auto_approved_usable_pending_sample_review"
            if accepted_query_count == _TARGET_QUERY_COUNT
            else "running"
        ),
        "accepted_ledger_sha256": sha256_bytes(canonical_jsonl_bytes(entries)),
    }
    return R2ExecutionCheckpoint.model_validate(
        {
            **payload,
            "checkpoint_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


def _verify_batch_manifest(
    context: _ExecutionContext,
    manifest: R2ExecutionBatchManifest,
) -> tuple[PlannedQuery, ...]:
    if context.legacy_turn_exclusion_receipt_sha256 is None:
        raise PortfolioCoreExecutionError(
            "mechanical execution requires a legacy turn exclusion receipt"
        )
    try:
        rows = context.rows_by_batch[manifest.base_batch_id]
    except KeyError as exc:
        raise PortfolioCoreExecutionError(
            "batch manifest names an unknown r2 batch"
        ) from exc
    expected = {
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": context.plan.plan.capability_assignments_sha256,
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(context.bridge.realism_sidecar.manifest)
        ),
        "realism_assignments_sha256": context.bridge.realism_sidecar.manifest.assignments_sha256,
        "final_split_sidecar_sha256": context.bridge.realism_sidecar.manifest.final_split_sidecar_sha256,
        "plan_ids_sha256": _expected_plan_ids_sha256(rows),
        "legacy_turn_exclusion_receipt_sha256": (
            context.legacy_turn_exclusion_receipt_sha256
        ),
    }
    if any(
        getattr(manifest, field_name) != value for field_name, value in expected.items()
    ):
        raise PortfolioCoreExecutionError("accepted batch manifest binding drifted")
    return rows


def _direct_entries(directory: Path, label: str) -> dict[str, Path]:
    _require_real_directory(directory, label)
    try:
        entries = {entry.name: Path(entry.path) for entry in os.scandir(directory)}
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to list {label}") from exc
    for name, path in entries.items():
        if not _SAFE_IDENTIFIER.fullmatch(name):
            raise PortfolioCoreExecutionError(f"{label} contains an unsafe entry name")
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise PortfolioCoreExecutionError(
                f"unable to inspect {label} entry"
            ) from exc
        if _is_link_or_reparse(metadata):
            raise PortfolioCoreExecutionError(
                f"{label} contains a symlink or reparse point"
            )
    return entries


def _ensure_create_only_directory(root: Path, label: str) -> None:
    if not root.is_absolute():
        raise PortfolioCoreExecutionError(f"{label} must be an absolute directory")
    if os.path.lexists(root):
        _require_safe_absolute_directory_chain(root, label)
        return
    _require_safe_absolute_directory_chain(root.parent, f"{label} parent")
    transient: Path | None = None
    try:
        transient = new_staging_directory(root)
        _require_real_directory(transient, f"new {label} staging")
        atomic_publish_new_directory(transient, root)
        transient = None
    except FileExistsError:
        _require_safe_absolute_directory_chain(root, label)
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to create {label}") from exc
    finally:
        if transient is not None:
            _cleanup_new_staging(
                transient,
                parent=root.parent,
                target_name=root.name,
            )


def _read_r2_author_draft_manifest(path: Path) -> R2AuthorDraftManifest:
    content = _safe_read(path, "published author draft manifest")
    try:
        manifest = R2AuthorDraftManifest.model_validate_json(content)
    except ValidationError as exc:
        raise PortfolioCoreExecutionError(
            "published author draft manifest violates its schema"
        ) from exc
    if _draft_manifest_bytes(manifest) != content:
        raise PortfolioCoreExecutionError(
            "published author draft manifest is not canonical JSON"
        )
    return manifest


def _verify_published_r2_author_draft_directory(
    directory: Path,
    *,
    authoring_job: AuthoringJob,
    expected: R2PublishedAuthorDraft | None = None,
) -> R2PublishedAuthorDraft:
    entries = _direct_entries(directory, "published author draft directory")
    if set(entries) != _DRAFT_ARTIFACT_FILES:
        raise PortfolioCoreExecutionError(
            "published author draft directory has unexpected or missing files"
        )
    manifest = _read_r2_author_draft_manifest(entries["manifest.json"])
    drafts_bytes = _safe_read(entries["drafts.jsonl"], "published author drafts")
    drafts = _parse_generated_trajectories(drafts_bytes, "published author drafts")
    receipt = _read_model(
        entries["receipt.json"], R2AuthorDraftReceipt, "published author draft receipt"
    )
    assert isinstance(receipt, R2AuthorDraftReceipt)
    expected_receipt, expected_drafts, expected_manifest = _author_draft_receipt(
        authoring_job, drafts, manifest
    )
    if receipt != expected_receipt or receipt.canonical_bytes() != _safe_read(
        entries["receipt.json"], "published author draft receipt"
    ):
        raise PortfolioCoreExecutionError("published author draft receipt drifted")
    published = R2PublishedAuthorDraft(
        directory=directory,
        drafts=expected_drafts,
        draft_manifest=expected_manifest,
        receipt=expected_receipt,
    )
    if expected is not None and (
        published.receipt != expected.receipt
        or published.drafts != expected.drafts
        or published.draft_manifest != expected.draft_manifest
        or _safe_read(entries["drafts.jsonl"], "published author drafts")
        != canonical_jsonl_bytes(expected.drafts)
        or _safe_read(entries["manifest.json"], "published author draft manifest")
        != _draft_manifest_bytes(expected.draft_manifest)
    ):
        raise PortfolioCoreExecutionError(
            "existing published author draft is not byte-identical"
        )
    return published


def publish_r2_author_drafts(
    *,
    authoring_job: AuthoringJob,
    drafts: tuple[GeneratedTrajectory, ...] | list[GeneratedTrajectory],
    draft_manifest: R2AuthorDraftManifest,
    draft_artifact_root: str | Path,
) -> R2PublishedAuthorDraft:
    """Create-only publish a body-bearing r2 author draft and receipt.

    Each child directory is one exact batch and contains only the canonical
    ``drafts.jsonl``, ``manifest.json`` and self-hashed ``receipt.json``.  A
    byte-identical repeat is idempotent; any drift, extra file, symlink, or
    reparse point fails closed.
    """

    receipt, draft_tuple, manifest = _author_draft_receipt(
        authoring_job, drafts, draft_manifest
    )
    root = Path(draft_artifact_root)
    _ensure_create_only_directory(root, "draft_artifact_root")
    root_entries = _direct_entries(root, "draft_artifact_root")
    for path in root_entries.values():
        _require_real_directory(path, "draft_artifact_root child")
    target = root / receipt.base_batch_id
    expected = R2PublishedAuthorDraft(
        directory=target,
        drafts=draft_tuple,
        draft_manifest=manifest,
        receipt=receipt,
    )
    if os.path.lexists(target):
        return _verify_published_r2_author_draft_directory(
            target, authoring_job=authoring_job, expected=expected
        )
    transient: Path | None = None
    try:
        transient = new_staging_directory(target)
        _require_real_directory(transient, "new published author draft staging")
        atomic_create_file(
            transient / "drafts.jsonl", canonical_jsonl_bytes(draft_tuple)
        )
        atomic_create_file(transient / "manifest.json", _draft_manifest_bytes(manifest))
        atomic_create_file(transient / "receipt.json", receipt.canonical_bytes())
        _verify_published_r2_author_draft_directory(
            transient, authoring_job=authoring_job, expected=expected
        )
        try:
            atomic_publish_new_directory(transient, target)
        except FileExistsError:
            return _verify_published_r2_author_draft_directory(
                target, authoring_job=authoring_job, expected=expected
            )
        transient = None
        return _verify_published_r2_author_draft_directory(
            target, authoring_job=authoring_job, expected=expected
        )
    except PortfolioCoreExecutionError:
        raise
    except OSError as exc:
        raise PortfolioCoreExecutionError("unable to publish r2 author drafts") from exc
    finally:
        if transient is not None:
            _cleanup_new_staging(
                transient,
                parent=target.parent,
                target_name=target.name,
            )


def load_published_r2_author_drafts(
    *,
    authoring_job: AuthoringJob,
    draft_artifact_root: str | Path,
) -> R2PublishedAuthorDraft:
    """Load and reverify the create-only artifact required for execution."""

    _require_issued_authoring_job(authoring_job)
    root = Path(draft_artifact_root)
    _require_safe_absolute_directory_chain(root, "draft_artifact_root")
    entries = _direct_entries(root, "draft_artifact_root")
    for path in entries.values():
        _require_real_directory(path, "draft_artifact_root child")
    batch_id = authoring_job.work_order.base_batch_id
    try:
        directory = entries[batch_id]
    except KeyError as exc:
        raise PortfolioCoreExecutionError(
            "published author draft is absent for the issued authoring job"
        ) from exc
    return _verify_published_r2_author_draft_directory(
        directory, authoring_job=authoring_job
    )


def _verify_author_draft_receipt_against_execution_manifest(
    receipt: R2AuthorDraftReceipt,
    manifest: R2ExecutionBatchManifest,
) -> None:
    expected = {
        "base_batch_id": manifest.base_batch_id,
        "job_id": manifest.job_id,
        "plan_sha256": manifest.plan_sha256,
        "asset_catalog_sha256": manifest.asset_catalog_sha256,
        "capability_assignments_sha256": manifest.capability_assignments_sha256,
        "realism_manifest_sha256": manifest.realism_manifest_sha256,
        "realism_assignments_sha256": manifest.realism_assignments_sha256,
        "final_split_sidecar_sha256": manifest.final_split_sidecar_sha256,
        "generation_input_sha256": manifest.generation_input_sha256,
        "author_packet_sha256": manifest.author_packet_sha256,
        "work_order_sha256": manifest.work_order_sha256,
        "plan_ids_sha256": manifest.plan_ids_sha256,
        "draft_sha256": manifest.draft_sha256,
    }
    if any(
        getattr(receipt, field_name) != value for field_name, value in expected.items()
    ):
        raise PortfolioCoreExecutionError(
            "execution author draft receipt binding drifted"
        )


def _verify_batch_directory(
    directory: Path,
    context: _ExecutionContext,
    *,
    expected: _ExecutionPayload | None = None,
) -> tuple[R2ExecutionBatchManifest, tuple[Query, ...]]:
    entries = _direct_entries(directory, "execution batch directory")
    if set(entries) != _BATCH_FILES:
        raise PortfolioCoreExecutionError(
            "execution batch directory has unexpected or missing files"
        )
    manifest_bytes = _safe_read(entries["manifest.json"], "execution batch manifest")
    try:
        manifest = R2ExecutionBatchManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise PortfolioCoreExecutionError(
            "execution batch manifest violates its schema"
        ) from exc
    if manifest.canonical_bytes() != manifest_bytes:
        raise PortfolioCoreExecutionError(
            "execution batch manifest is not canonical JSON"
        )
    assert isinstance(manifest, R2ExecutionBatchManifest)
    rows = _verify_batch_manifest(context, manifest)
    author_draft_receipt_bytes = _safe_read(
        entries["author-draft-receipt.json"], "execution author draft receipt"
    )
    try:
        author_draft_receipt = R2AuthorDraftReceipt.model_validate_json(
            author_draft_receipt_bytes
        )
    except ValidationError as exc:
        raise PortfolioCoreExecutionError(
            "execution author draft receipt violates its schema"
        ) from exc
    if author_draft_receipt.canonical_bytes() != author_draft_receipt_bytes:
        raise PortfolioCoreExecutionError(
            "execution author draft receipt is not canonical JSON"
        )
    if sha256_bytes(author_draft_receipt_bytes) != manifest.author_draft_receipt_sha256:
        raise PortfolioCoreExecutionError(
            "execution author draft receipt SHA-256 drifted"
        )
    _verify_author_draft_receipt_against_execution_manifest(
        author_draft_receipt, manifest
    )
    results_bytes = _safe_read(entries["results.jsonl"], "execution batch results")
    if sha256_bytes(results_bytes) != manifest.results_sha256:
        raise PortfolioCoreExecutionError("execution batch results SHA-256 drifted")
    queries = _parse_queries(results_bytes, "execution batch results")
    if len(queries) != _BATCH_SIZE or tuple(
        query.query_id for query in queries
    ) != tuple(row.plan_id for row in rows):
        raise PortfolioCoreExecutionError(
            "execution batch results do not match r2 plan order"
        )
    for row, query in zip(rows, queries, strict=True):
        reconstructed = _materialize_query(context, row, query.turns)
        if query != reconstructed:
            raise PortfolioCoreExecutionError(
                f"execution query drifted from r2 plan or final-split sidecar: {row.plan_id}"
            )
    normalized = [normalized_text(query.text) for query in queries]
    if any(not value for value in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise PortfolioCoreExecutionError(
            "execution batch contains duplicate normalized text"
        )
    if expected is not None and (
        manifest != expected.batch_manifest
        or results_bytes != expected.results_bytes
        or manifest_bytes != expected.manifest_bytes
        or author_draft_receipt_bytes != expected.author_draft_receipt_bytes
    ):
        raise PortfolioCoreExecutionError(
            "existing execution batch is not byte-identical"
        )
    return manifest, queries


def _verify_runtime_root_shape(root: Path) -> tuple[Path, Path, Path, Path]:
    _require_safe_absolute_directory_chain(root, "execution_root")
    entries = _direct_entries(root, "execution_root")
    if set(entries) != _ROOT_FILES | _ROOT_DIRECTORIES:
        raise PortfolioCoreExecutionError(
            "execution_root has unexpected or missing entries"
        )
    accepted = entries["accepted"]
    staging = entries["staging"]
    ledger = entries["accepted-ledger.jsonl"]
    checkpoint = entries["runtime-checkpoint.json"]
    _require_real_directory(accepted, "execution accepted root")
    _require_real_directory(staging, "execution staging root")
    _require_regular_file(ledger, "execution accepted ledger")
    _require_regular_file(checkpoint, "execution runtime checkpoint")
    return accepted, staging, ledger, checkpoint


def _validate_ledger_prefix(
    context: _ExecutionContext,
    entries: tuple[R2ExecutionLedgerEntry, ...],
) -> None:
    if len(entries) > len(context.batch_ids):
        raise PortfolioCoreExecutionError("accepted ledger exceeds the r2 plan")
    actual_ids = tuple(entry.base_batch_id for entry in entries)
    expected_ids = context.batch_ids[: len(entries)]
    if actual_ids != expected_ids:
        raise PortfolioCoreExecutionError(
            "accepted ledger is not an exact r2 plan prefix"
        )
    if len({entry.base_batch_id for entry in entries}) != len(entries):
        raise PortfolioCoreExecutionError("accepted ledger repeats a base batch")
    if any(entry.plan_sha256 != context.trusted_plan_sha256 for entry in entries):
        raise PortfolioCoreExecutionError("accepted ledger plan binding drifted")
    if context.legacy_turn_exclusion_receipt_sha256 is None or any(
        entry.legacy_turn_exclusion_receipt_sha256
        != context.legacy_turn_exclusion_receipt_sha256
        for entry in entries
    ):
        raise PortfolioCoreExecutionError(
            "accepted ledger legacy turn exclusion binding drifted"
        )


def _verify_ledger_entry_against_batch(
    entry: R2ExecutionLedgerEntry,
    manifest: R2ExecutionBatchManifest,
) -> None:
    if (
        entry.base_batch_id != manifest.base_batch_id
        or entry.job_id != manifest.job_id
        or entry.results_sha256 != manifest.results_sha256
        or entry.batch_manifest_sha256 != sha256_bytes(manifest.canonical_bytes())
        or entry.plan_sha256 != manifest.plan_sha256
        or entry.legacy_turn_exclusion_receipt_sha256
        != manifest.legacy_turn_exclusion_receipt_sha256
        or entry.author_draft_receipt_sha256 != manifest.author_draft_receipt_sha256
    ):
        raise PortfolioCoreExecutionError("accepted ledger entry drifted from batch")


def _load_runtime_state(
    root: Path,
    context: _ExecutionContext,
    *,
    pending: _ExecutionPayload | None,
) -> _RuntimeState:
    accepted_root, staging_root, ledger_path, checkpoint_path = (
        _verify_runtime_root_shape(root)
    )
    ledger_bytes = _safe_read(ledger_path, "execution accepted ledger")
    entries = _parse_ledger(ledger_bytes)
    _validate_ledger_prefix(context, entries)
    accepted_children = _direct_entries(accepted_root, "execution accepted root")
    staging_children = _direct_entries(staging_root, "execution staging root")
    for path in (*accepted_children.values(), *staging_children.values()):
        _require_real_directory(path, "execution batch child")

    ledger_ids = tuple(entry.base_batch_id for entry in entries)
    ledger_id_set = set(ledger_ids)
    accepted_ids = set(accepted_children)
    staging_ids = set(staging_children)
    if not accepted_ids.issubset(ledger_id_set):
        raise PortfolioCoreExecutionError(
            "execution accepted directories are visible before their durable ledger entry"
        )
    if accepted_ids & staging_ids:
        raise PortfolioCoreExecutionError(
            "execution batch appears in both accepted and staging roots"
        )

    ledger_pending_promotion_batch_id: str | None = None
    if accepted_ids == ledger_id_set:
        pass
    elif entries and accepted_ids == set(ledger_ids[:-1]):
        candidate = ledger_ids[-1]
        if staging_ids != {candidate}:
            raise PortfolioCoreExecutionError(
                "ledger-last batch must remain only in exact staging before promotion"
            )
        ledger_pending_promotion_batch_id = candidate
    else:
        raise PortfolioCoreExecutionError(
            "execution accepted directories do not match the durable ledger prefix"
        )

    seen_texts: set[str] = set()
    accepted_entries = entries
    if ledger_pending_promotion_batch_id is not None:
        accepted_entries = entries[:-1]
    for entry in accepted_entries:
        manifest, queries = _verify_batch_directory(
            accepted_children[entry.base_batch_id], context
        )
        _verify_ledger_entry_against_batch(entry, manifest)
        batch_texts = {normalized_text(query.text) for query in queries}
        if seen_texts & batch_texts:
            raise PortfolioCoreExecutionError(
                "accepted batches contain duplicate normalized text"
            )
        seen_texts.update(batch_texts)

    uncommitted_staging_batch_id: str | None = None
    if ledger_pending_promotion_batch_id is not None:
        entry = entries[-1]
        expected_payload = (
            pending
            if pending is not None
            and pending.batch_manifest.base_batch_id == entry.base_batch_id
            else None
        )
        manifest, queries = _verify_batch_directory(
            staging_children[entry.base_batch_id], context, expected=expected_payload
        )
        _verify_ledger_entry_against_batch(entry, manifest)
        batch_texts = {normalized_text(query.text) for query in queries}
        if seen_texts & batch_texts:
            raise PortfolioCoreExecutionError(
                "accepted batches contain duplicate normalized text"
            )
    elif staging_ids:
        if len(staging_ids) != 1 or len(entries) >= len(context.batch_ids):
            raise PortfolioCoreExecutionError(
                "execution staging root contains an unexpected batch"
            )
        candidate = next(iter(staging_ids))
        expected_next = context.batch_ids[len(entries)]
        if candidate != expected_next:
            raise PortfolioCoreExecutionError(
                "execution staging batch is not the next r2 plan batch"
            )
        expected_payload = pending if pending is not None else None
        if (
            expected_payload is not None
            and candidate != expected_payload.batch_manifest.base_batch_id
        ):
            raise PortfolioCoreExecutionError(
                "execution staging batch awaits its own published draft retry"
            )
        _, queries = _verify_batch_directory(
            staging_children[candidate], context, expected=expected_payload
        )
        staged_texts = {normalized_text(query.text) for query in queries}
        if seen_texts & staged_texts:
            raise PortfolioCoreExecutionError(
                "execution staging batch duplicates normalized accepted text"
            )
        uncommitted_staging_batch_id = candidate

    checkpoint = _read_model(
        checkpoint_path, R2ExecutionCheckpoint, "execution runtime checkpoint"
    )
    assert isinstance(checkpoint, R2ExecutionCheckpoint)
    expected_checkpoint = _checkpoint(context, entries)
    checkpoint_lags_ledger = False
    allowed_checkpoints = {expected_checkpoint.canonical_bytes()}
    if entries:
        allowed_checkpoints.add(_checkpoint(context, entries[:-1]).canonical_bytes())
    checkpoint_bytes = checkpoint.canonical_bytes()
    if checkpoint_bytes not in allowed_checkpoints:
        raise PortfolioCoreExecutionError("execution runtime checkpoint drifted")
    if checkpoint_bytes != expected_checkpoint.canonical_bytes():
        checkpoint_lags_ledger = True
    return _RuntimeState(
        entries=entries,
        checkpoint=checkpoint,
        checkpoint_bytes=checkpoint_bytes,
        accepted_texts=frozenset(seen_texts),
        checkpoint_lags_ledger=checkpoint_lags_ledger,
        ledger_pending_promotion_batch_id=ledger_pending_promotion_batch_id,
        uncommitted_staging_batch_id=uncommitted_staging_batch_id,
    )


def _cleanup_new_staging(path: Path, *, parent: Path, target_name: str) -> None:
    if not os.path.lexists(path):
        return
    if path.parent != parent or not path.name.startswith(f".{target_name}.staging-"):
        raise PortfolioCoreExecutionError(
            "refusing to clean an unexpected staging directory"
        )
    _require_real_directory(path, "new execution staging cleanup target")
    descendants = [path]
    while descendants:
        current = descendants.pop()
        try:
            children = tuple(Path(entry.path) for entry in os.scandir(current))
        except OSError as exc:
            raise PortfolioCoreExecutionError(
                "unable to inspect new execution staging cleanup target"
            ) from exc
        for child in children:
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise PortfolioCoreExecutionError(
                    "unable to inspect new execution staging cleanup entry"
                ) from exc
            if _is_link_or_reparse(metadata):
                raise PortfolioCoreExecutionError(
                    "refusing to clean staging containing a symlink or reparse point"
                )
            if stat.S_ISDIR(metadata.st_mode):
                descendants.append(child)
            elif not stat.S_ISREG(metadata.st_mode):
                raise PortfolioCoreExecutionError(
                    "refusing to clean staging containing a non-regular entry"
                )
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise PortfolioCoreExecutionError(
            "unable to clean new execution staging"
        ) from exc


def _create_execution_root(root: Path, context: _ExecutionContext) -> None:
    if not root.is_absolute():
        raise PortfolioCoreExecutionError(
            "execution_root must be an absolute directory"
        )
    if os.path.lexists(root):
        return
    _require_safe_absolute_directory_chain(root.parent, "execution_root parent")
    staging: Path | None = None
    try:
        staging = new_staging_directory(root)
        _require_real_directory(staging, "new execution_root staging")
        (staging / "accepted").mkdir()
        (staging / "staging").mkdir()
        atomic_create_file(staging / "accepted-ledger.jsonl", b"")
        atomic_create_file(
            staging / "runtime-checkpoint.json",
            _checkpoint(context, ()).canonical_bytes(),
        )
        atomic_publish_new_directory(staging, root)
        staging = None
    except FileExistsError:
        return
    except OSError as exc:
        raise PortfolioCoreExecutionError("unable to create execution_root") from exc
    finally:
        if staging is not None:
            _cleanup_new_staging(staging, parent=root.parent, target_name=root.name)


def prepare_portfolio_core_r2_execution(
    plan: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
    legacy_turn_exclusion_receipt: LegacyTurnExclusionReceipt,
    execution_root: str | Path,
) -> R2ExecutionCheckpoint:
    """Create, verify, or repair an r2 execution root.

    This performs no text generation and never writes to an author-job
    directory.  It repairs only a ledger-durable promotion/checkpoint
    interruption.  A pre-ledger staging directory remains deliberately
    uncommitted until the same published draft artifact is retried.
    """

    context = _validate_context(
        plan,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        legacy_turn_exclusion_receipt=legacy_turn_exclusion_receipt,
    )
    root = Path(execution_root)
    _create_execution_root(root, context)
    state = _recover_runtime(root, context, pending=None)
    if state.ledger_pending_promotion_batch_id is not None:
        raise PortfolioCoreExecutionError(
            "ledger-last batch could not be promoted to accepted"
        )
    if state.uncommitted_staging_batch_id is not None:
        raise PortfolioCoreExecutionError(
            "uncommitted execution staging awaits the same published draft retry"
        )
    return state.checkpoint


def _stage_payload(
    root: Path,
    context: _ExecutionContext,
    payload: _ExecutionPayload,
) -> Path:
    persistent = root / "staging" / payload.batch_manifest.base_batch_id
    if os.path.lexists(persistent):
        _verify_batch_directory(persistent, context, expected=payload)
        return persistent
    transient: Path | None = None
    try:
        transient = new_staging_directory(persistent)
        _require_real_directory(transient, "new execution batch staging")
        atomic_create_file(transient / "results.jsonl", payload.results_bytes)
        atomic_create_file(transient / "manifest.json", payload.manifest_bytes)
        atomic_create_file(
            transient / "author-draft-receipt.json",
            payload.author_draft_receipt_bytes,
        )
        _verify_batch_directory(transient, context, expected=payload)
        try:
            atomic_publish_new_directory(transient, persistent)
        except FileExistsError:
            _verify_batch_directory(persistent, context, expected=payload)
        else:
            transient = None
        return persistent
    except PortfolioCoreExecutionError:
        raise
    except OSError as exc:
        raise PortfolioCoreExecutionError("unable to stage execution batch") from exc
    finally:
        if transient is not None:
            _cleanup_new_staging(
                transient,
                parent=persistent.parent,
                target_name=persistent.name,
            )


def _promote_ledgered_staging(
    root: Path,
    context: _ExecutionContext,
    entry: R2ExecutionLedgerEntry,
    *,
    expected: _ExecutionPayload | None = None,
) -> None:
    ledger = _parse_ledger(
        _safe_read(root / "accepted-ledger.jsonl", "execution accepted ledger")
    )
    _validate_ledger_prefix(context, ledger)
    if not ledger or ledger[-1] != entry:
        raise PortfolioCoreExecutionError(
            "cannot promote an execution batch before its durable ledger entry"
        )
    source = root / "staging" / entry.base_batch_id
    destination = root / "accepted" / entry.base_batch_id
    if os.path.lexists(destination):
        manifest, _ = _verify_batch_directory(destination, context, expected=expected)
        _verify_ledger_entry_against_batch(entry, manifest)
        return
    manifest, _ = _verify_batch_directory(source, context, expected=expected)
    _verify_ledger_entry_against_batch(entry, manifest)
    try:
        atomic_publish_new_directory(source, destination)
    except FileExistsError:
        manifest, _ = _verify_batch_directory(destination, context, expected=expected)
        _verify_ledger_entry_against_batch(entry, manifest)
    except OSError as exc:
        raise PortfolioCoreExecutionError(
            "unable to promote execution staging to accepted"
        ) from exc


def _replace_exact(path: Path, before: bytes, after: bytes, label: str) -> None:
    current = _safe_read(path, label)
    if current != before:
        raise PortfolioCoreExecutionError(f"{label} drifted before update")
    try:
        atomic_replace_file(path, after)
    except OSError as exc:
        raise PortfolioCoreExecutionError(f"unable to update {label}") from exc
    if _safe_read(path, label) != after:
        raise PortfolioCoreExecutionError(f"{label} drifted after update")


def _repair_checkpoint_if_needed(
    root: Path,
    context: _ExecutionContext,
    state: _RuntimeState,
) -> R2ExecutionCheckpoint:
    expected = _checkpoint(context, state.entries)
    if state.checkpoint_bytes == expected.canonical_bytes():
        return expected
    _replace_exact(
        root / "runtime-checkpoint.json",
        state.checkpoint_bytes,
        expected.canonical_bytes(),
        "execution runtime checkpoint",
    )
    return expected


def _append_durable_ledger_entry(
    root: Path,
    context: _ExecutionContext,
    state: _RuntimeState,
    payload: _ExecutionPayload,
) -> tuple[R2ExecutionLedgerEntry, ...]:
    if state.ledger_pending_promotion_batch_id is not None:
        raise PortfolioCoreExecutionError(
            "cannot append a ledger entry while the prior ledger-last batch awaits promotion"
        )
    entries = state.entries
    if payload.batch_manifest.base_batch_id in {
        entry.base_batch_id for entry in entries
    }:
        if entries[-1] != payload.ledger_entry:
            raise PortfolioCoreExecutionError("durable execution ledger entry drifted")
        return entries
    new_entries = (*entries, payload.ledger_entry)
    old_ledger = canonical_jsonl_bytes(entries)
    new_ledger = canonical_jsonl_bytes(new_entries)
    _replace_exact(
        root / "accepted-ledger.jsonl",
        old_ledger,
        new_ledger,
        "execution accepted ledger",
    )
    return new_entries


def _recover_runtime(
    root: Path,
    context: _ExecutionContext,
    *,
    pending: _ExecutionPayload | None,
) -> _RuntimeState:
    """Repair only ledger-durable interruptions; never auto-commit staging."""

    state = _load_runtime_state(root, context, pending=pending)
    if state.ledger_pending_promotion_batch_id is not None:
        entry = state.entries[-1]
        expected = (
            pending
            if pending is not None
            and pending.batch_manifest.base_batch_id == entry.base_batch_id
            else None
        )
        _promote_ledgered_staging(root, context, entry, expected=expected)
        state = _load_runtime_state(root, context, pending=pending)
    if state.checkpoint_lags_ledger:
        _repair_checkpoint_if_needed(root, context, state)
        state = _load_runtime_state(root, context, pending=pending)
    return state


def auto_approve_portfolio_core_r2_batch(
    plan: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
    legacy_turn_exclusion_receipt: LegacyTurnExclusionReceipt,
    authoring_job: AuthoringJob,
    draft_artifact_root: str | Path,
    execution_root: str | Path,
) -> R2ExecutionResult:
    """Mechanically accept one exact, in-order r2 author batch.

    The function performs no model call and generates no text.  It accepts
    only a create-only, reverified author-draft artifact; raw in-memory drafts
    are not a formal execution input.  It durably appends the ledger before an
    accepted directory becomes visible, then advances the runtime checkpoint.
    """

    context = _validate_context(
        plan,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        legacy_turn_exclusion_receipt=legacy_turn_exclusion_receipt,
    )
    published_draft = load_published_r2_author_drafts(
        authoring_job=authoring_job,
        draft_artifact_root=draft_artifact_root,
    )
    payload = _build_payload(context, authoring_job, published_draft)
    root = Path(execution_root)
    _create_execution_root(root, context)
    state = _recover_runtime(root, context, pending=payload)
    batch_id = payload.batch_manifest.base_batch_id
    try:
        batch_index = context.batch_ids.index(batch_id)
    except ValueError as exc:  # pragma: no cover - job binding already checks it
        raise PortfolioCoreExecutionError("authoring job batch is unknown") from exc

    accepted_ids = tuple(entry.base_batch_id for entry in state.entries)
    if batch_id in accepted_ids:
        _verify_batch_directory(root / "accepted" / batch_id, context, expected=payload)
        return R2ExecutionResult(
            accepted_dir=root / "accepted" / batch_id,
            created=False,
            checkpoint=state.checkpoint,
        )
    if batch_index != len(state.entries):
        raise PortfolioCoreExecutionError(
            "batch must follow the complete r2 plan order"
        )
    normalized = {normalized_text(query.text) for query in payload.results}
    if normalized & state.accepted_texts:
        raise PortfolioCoreExecutionError(
            "draft duplicates normalized text from an accepted r2 batch"
        )

    if state.uncommitted_staging_batch_id is None:
        _stage_payload(root, context, payload)
    elif state.uncommitted_staging_batch_id != batch_id:
        raise PortfolioCoreExecutionError(
            "execution staging batch awaits its own published draft retry"
        )
    ledger_entries = _append_durable_ledger_entry(root, context, state, payload)
    _promote_ledgered_staging(root, context, ledger_entries[-1], expected=payload)
    state = _load_runtime_state(root, context, pending=payload)
    checkpoint = _repair_checkpoint_if_needed(root, context, state)
    final_state = _load_runtime_state(root, context, pending=None)
    if (
        final_state.checkpoint != checkpoint
        or final_state.ledger_pending_promotion_batch_id is not None
        or final_state.uncommitted_staging_batch_id is not None
    ):
        raise PortfolioCoreExecutionError("execution runtime state did not converge")
    return R2ExecutionResult(
        accepted_dir=root / "accepted" / batch_id,
        created=True,
        checkpoint=checkpoint,
    )


__all__ = [
    "EXECUTION_POLICY_VERSION",
    "LegacyTurnDigest",
    "LegacyTurnExclusionReceipt",
    "PortfolioCoreExecutionError",
    "R2AuthorDraftManifest",
    "R2AuthorDraftReceipt",
    "R2ExecutionBatchManifest",
    "R2ExecutionCheckpoint",
    "R2ExecutionLedgerEntry",
    "R2ExecutionResult",
    "R2PublishedAuthorDraft",
    "auto_approve_portfolio_core_r2_batch",
    "build_legacy_turn_exclusion_receipt",
    "build_r2_author_draft_manifest",
    "canonical_turns_sha256",
    "load_published_r2_author_drafts",
    "publish_r2_author_drafts",
    "prepare_portfolio_core_r2_execution",
]
