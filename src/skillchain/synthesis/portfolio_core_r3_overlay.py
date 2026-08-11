"""Create-only Portfolio Core r3 text-repair overlay.

This module is intentionally independent of the r2 executor.  It does not
open a corpus, a draft artifact, or an image: callers must hand it either
typed :class:`~skillchain.schemas.Query` rows or canonical query JSONL bytes,
an already-frozen ten-by-twenty-five repair plan, and body-only replacement
drafts.  The complete r3 overlay is then materialised in memory, verified,
and published as one create-only directory.

The overlay is deliberately narrow.  It changes only ``text`` and ``turns``
for the 250 planned replacement rows; all other schema-v2 fields are copied
verbatim from the supplied r2 source rows.  It never changes r2 planning,
execution, or their artifacts.
"""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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

from skillchain.schemas import ConversationTurn, Query
from skillchain.synthesis.batches import normalized_text
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

R3_OVERLAY_VERSION = "portfolio-core-r3-overlay-v1"
_SOURCE_QUERY_COUNT = 1500
_PARENT_BATCH_COUNT = 60
_REPAIR_BATCH_COUNT = 10
_BATCH_SIZE = 25
_REPLACEMENT_COUNT = _REPAIR_BATCH_COUNT * _BATCH_SIZE
_CARRY_FORWARD_COUNT = _SOURCE_QUERY_COUNT - _REPLACEMENT_COUNT
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class PortfolioCoreR3OverlayError(ValueError):
    """An r3 overlay input is not safe to materialise."""


class R3OverlayPublicationError(RuntimeError):
    """An r3 overlay cannot safely be published."""

    def __init__(self, message: str, *, staging_path: Path | None = None) -> None:
        super().__init__(message)
        self.staging_path = staging_path


class R3OverlayConflictError(R3OverlayPublicationError):
    """A populated r3 target is not an identical prior publication."""


class R3OverlaySafetyError(R3OverlayPublicationError):
    """A target contains an unsafe filesystem entry."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must be non-blank")
    return value


def _safe_identifier(value: str, label: str) -> str:
    value = _nonblank(value, label)
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PortfolioCoreR3OverlayError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must not be absolute or traverse")
    return value


def _json_ready(value: object) -> object:
    """Convert nested strict models before canonical JSON hashing."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _digest_without(payload: Mapping[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(_json_ready(unsigned)))


def _query_ids_sha256(query_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(query_ids)))


def canonical_turns_sha256(turns: Sequence[ConversationTurn]) -> str:
    """Return the canonical digest for one complete conversation trajectory."""

    if not isinstance(turns, (tuple, list)) or not turns:
        raise PortfolioCoreR3OverlayError("turns must be a non-empty sequence")
    if any(not isinstance(turn, ConversationTurn) for turn in turns):
        raise PortfolioCoreR3OverlayError("turns must contain ConversationTurn values")
    return sha256_bytes(
        canonical_json_bytes([turn.model_dump(mode="json") for turn in turns])
    )


def normalized_final_text(value: str) -> str:
    """Use the project normalizer after NFKC for r3 global uniqueness."""

    return normalized_text(unicodedata.normalize("NFKC", value))


def _validate_ascii_free(value: str, *, query_id: str, field: str) -> None:
    normalized = unicodedata.normalize("NFKC", value)
    if _ASCII_LETTER.search(value) or _ASCII_LETTER.search(normalized):
        # Deliberately never include query text in an exception.  In production
        # these checks may run over an externally held corpus.
        raise PortfolioCoreR3OverlayError(
            f"{field} contains an ASCII letter before or after NFKC: {query_id}"
        )


def _validate_turn_language(turns: Sequence[ConversationTurn], *, query_id: str) -> None:
    for turn in turns:
        _validate_ascii_free(turn.content, query_id=query_id, field="turn content")
    _validate_ascii_free(turns[-1].content, query_id=query_id, field="final text")


def _role_shape(turns: Sequence[ConversationTurn]) -> tuple[str, ...]:
    return tuple(turn.role for turn in turns)


def _require_valid_role_shape(turns: Sequence[ConversationTurn]) -> None:
    if _role_shape(turns) not in (("user",), ("user", "assistant", "user")):
        raise ValueError("turn roles must be [user] or [user, assistant, user]")


class R3RepairBatchPlan(_StrictFrozenModel):
    """One self-hashed, fixed 25-query repair batch declaration."""

    schema_version: Literal[1] = 1
    batch_id: str
    query_ids: tuple[str, ...]
    batch_sha256: Sha256

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "batch_id")

    @field_validator("query_ids")
    @classmethod
    def _validate_query_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_nonblank(item, "query_id") for item in value)
        if len(cleaned) != _BATCH_SIZE or len(set(cleaned)) != _BATCH_SIZE:
            raise ValueError("each repair batch must contain exactly 25 unique query IDs")
        return cleaned

    @model_validator(mode="after")
    def _validate_self_hash(self) -> Self:
        if self.batch_sha256 != _digest_without(
            self.model_dump(mode="json"), "batch_sha256"
        ):
            raise ValueError("repair batch plan self-hash mismatch")
        return self

    @classmethod
    def create(cls, *, batch_id: str, query_ids: Sequence[str]) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "batch_id": batch_id,
            "query_ids": tuple(query_ids),
        }
        return cls(
            **payload,
            batch_sha256=_digest_without(payload, "batch_sha256"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class R3RepairPlan(_StrictFrozenModel):
    """External frozen r3 plan: exactly ten ordered repair batches."""

    schema_version: Literal[1] = 1
    batches: tuple[R3RepairBatchPlan, ...]
    repair_plan_sha256: Sha256

    @model_validator(mode="after")
    def _validate_plan(self) -> Self:
        if len(self.batches) != _REPAIR_BATCH_COUNT:
            raise ValueError("r3 repair plan must contain exactly ten batches")
        batch_ids = tuple(item.batch_id for item in self.batches)
        if len(set(batch_ids)) != _REPAIR_BATCH_COUNT:
            raise ValueError("r3 repair plan batch IDs must be unique")
        query_ids = tuple(
            query_id for batch in self.batches for query_id in batch.query_ids
        )
        if len(set(query_ids)) != _REPLACEMENT_COUNT:
            raise ValueError("r3 repair plan query IDs must be globally unique")
        if self.repair_plan_sha256 != _digest_without(
            self.model_dump(mode="json"), "repair_plan_sha256"
        ):
            raise ValueError("r3 repair plan self-hash mismatch")
        return self

    @classmethod
    def create(cls, *, batches: Sequence[R3RepairBatchPlan]) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "batches": tuple(batches),
        }
        return cls(
            **payload,
            repair_plan_sha256=_digest_without(payload, "repair_plan_sha256"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def build_r3_repair_plan(
    batches: Sequence[tuple[str, Sequence[str]]],
) -> R3RepairPlan:
    """Build a self-hashed frozen plan from ordered ``(batch_id, query_ids)`` pairs."""

    return R3RepairPlan.create(
        batches=tuple(
            R3RepairBatchPlan.create(batch_id=batch_id, query_ids=query_ids)
            for batch_id, query_ids in batches
        )
    )


class R3ReplacementDraft(_StrictFrozenModel):
    """The only replacement payload accepted from a draft producer."""

    query_id: str
    turns: tuple[ConversationTurn, ...]

    @field_validator("query_id")
    @classmethod
    def _validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")

    @model_validator(mode="after")
    def _validate_turns(self) -> Self:
        _require_valid_role_shape(self.turns)
        return self


class R3RepairDraftBatch(_StrictFrozenModel):
    """A strictly body-only submitted draft for one fixed repair batch."""

    batch_id: str
    drafts: tuple[R3ReplacementDraft, ...]

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "batch_id")

    @model_validator(mode="after")
    def _validate_drafts(self) -> Self:
        if len(self.drafts) != _BATCH_SIZE:
            raise ValueError("each repair draft batch must contain exactly 25 drafts")
        query_ids = tuple(draft.query_id for draft in self.drafts)
        if len(set(query_ids)) != _BATCH_SIZE:
            raise ValueError("repair draft query IDs must be unique within a batch")
        return self


class R3ParentSourceHash(_StrictFrozenModel):
    """A source parent-batch identity carried into a repair manifest."""

    parent_batch_id: str
    source_parent_batch_sha256: Sha256

    @field_validator("parent_batch_id")
    @classmethod
    def _validate_parent_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "parent_batch_id")


class R3RepairBatchManifest(_StrictFrozenModel):
    """Self-hashed record for one materialised repair result batch."""

    schema_version: Literal[1] = 1
    overlay_version: Literal["portfolio-core-r3-overlay-v1"] = R3_OVERLAY_VERSION
    batch_id: str
    ordinal: int = Field(ge=1, le=_REPAIR_BATCH_COUNT)
    count: Literal[25] = _BATCH_SIZE
    source_r2_sha256: Sha256
    repair_plan_sha256: Sha256
    query_ids_sha256: Sha256
    source_selected_queries_sha256: Sha256
    drafts_sha256: Sha256
    repair_results_sha256: Sha256
    parent_source_hashes: tuple[R3ParentSourceHash, ...]
    manifest_sha256: Sha256

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "batch_id")

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        parents = tuple(item.parent_batch_id for item in self.parent_source_hashes)
        if not parents or len(set(parents)) != len(parents):
            raise ValueError("repair batch parent source hashes must be non-empty and unique")
        if self.manifest_sha256 != _digest_without(
            self.model_dump(mode="json"), "manifest_sha256"
        ):
            raise ValueError("repair batch manifest self-hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: object) -> Self:
        payload = {
            "schema_version": 1,
            "overlay_version": R3_OVERLAY_VERSION,
            "count": _BATCH_SIZE,
            **payload,
        }
        return cls(
            **payload,
            manifest_sha256=_digest_without(payload, "manifest_sha256"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class R3RepairLedgerEntry(_StrictFrozenModel):
    """One self-hashed, ordered accepted repair-batch receipt."""

    schema_version: Literal[1] = 1
    batch_id: str
    ordinal: int = Field(ge=1, le=_REPAIR_BATCH_COUNT)
    count: Literal[25] = _BATCH_SIZE
    source_r2_sha256: Sha256
    repair_plan_sha256: Sha256
    repair_results_sha256: Sha256
    batch_manifest_sha256: Sha256
    entry_sha256: Sha256

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "batch_id")

    @model_validator(mode="after")
    def _validate_entry(self) -> Self:
        if self.entry_sha256 != _digest_without(
            self.model_dump(mode="json"), "entry_sha256"
        ):
            raise ValueError("repair ledger entry self-hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: object) -> Self:
        payload = {"schema_version": 1, "count": _BATCH_SIZE, **payload}
        return cls(**payload, entry_sha256=_digest_without(payload, "entry_sha256"))


class R3RepairCheckpoint(_StrictFrozenModel):
    """Self-hashed completion checkpoint derived only from the repair ledger."""

    schema_version: Literal[1] = 1
    overlay_version: Literal["portfolio-core-r3-overlay-v1"] = R3_OVERLAY_VERSION
    source_query_count: Literal[1500] = _SOURCE_QUERY_COUNT
    repair_batch_count: Literal[10] = _REPAIR_BATCH_COUNT
    repair_batch_size: Literal[25] = _BATCH_SIZE
    replacement_query_count: Literal[250] = _REPLACEMENT_COUNT
    carry_forward_query_count: Literal[1250] = _CARRY_FORWARD_COUNT
    source_r2_sha256: Sha256
    repair_plan_sha256: Sha256
    repair_ledger_sha256: Sha256
    queries_sha256: Sha256
    accepted_batch_ids: tuple[str, ...]
    checkpoint_sha256: Sha256

    @field_validator("accepted_batch_ids")
    @classmethod
    def _validate_batch_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_safe_identifier(item, "accepted_batch_id") for item in value)
        if len(cleaned) != _REPAIR_BATCH_COUNT or len(set(cleaned)) != len(cleaned):
            raise ValueError("checkpoint must contain exactly ten unique repair batch IDs")
        return cleaned

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        if self.checkpoint_sha256 != _digest_without(
            self.model_dump(mode="json"), "checkpoint_sha256"
        ):
            raise ValueError("repair checkpoint self-hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: object) -> Self:
        payload = {
            "schema_version": 1,
            "overlay_version": R3_OVERLAY_VERSION,
            "source_query_count": _SOURCE_QUERY_COUNT,
            "repair_batch_count": _REPAIR_BATCH_COUNT,
            "repair_batch_size": _BATCH_SIZE,
            "replacement_query_count": _REPLACEMENT_COUNT,
            "carry_forward_query_count": _CARRY_FORWARD_COUNT,
            **payload,
        }
        return cls(
            **payload,
            checkpoint_sha256=_digest_without(payload, "checkpoint_sha256"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


class R3ParentBatchIndexEntry(_StrictFrozenModel):
    """Source-to-final binding for one of the sixty parent batches."""

    schema_version: Literal[1] = 1
    parent_batch_id: str
    ordinal: int = Field(ge=1, le=_PARENT_BATCH_COUNT)
    count: Literal[25] = _BATCH_SIZE
    query_ids_sha256: Sha256
    source_parent_batch_sha256: Sha256
    final_parent_batch_sha256: Sha256
    replacement_count: int = Field(ge=0, le=_BATCH_SIZE)

    @field_validator("parent_batch_id")
    @classmethod
    def _validate_parent_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "parent_batch_id")


class R3RepairBatchIndexEntry(_StrictFrozenModel):
    """Index entry for one on-disk repair result batch."""

    schema_version: Literal[1] = 1
    batch_id: str
    ordinal: int = Field(ge=1, le=_REPAIR_BATCH_COUNT)
    count: Literal[25] = _BATCH_SIZE
    repair_results_sha256: Sha256
    batch_manifest_sha256: Sha256

    @field_validator("batch_id")
    @classmethod
    def _validate_batch_id(cls, value: str) -> str:
        return _safe_identifier(value, "batch_id")


class R3LanguageValidationEntry(_StrictFrozenModel):
    """Body-free language validation receipt for one final query."""

    schema_version: Literal[1] = 1
    query_id: str
    disposition: Literal["replacement", "source_carry_forward"]
    turns_sha256: Sha256
    final_text_sha256: Sha256
    nfkc_final_text_sha256: Sha256
    normalized_final_text_sha256: Sha256
    original_ascii_free: Literal[True] = True
    nfkc_ascii_free: Literal[True] = True

    @field_validator("query_id")
    @classmethod
    def _validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")


class R3PublishedFile(_StrictFrozenModel):
    """One canonical file digest in the root manifest."""

    relative_path: str
    sha256: Sha256
    bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class R3RepairManifest(_StrictFrozenModel):
    """Root self-hash over every r3 artifact other than this manifest."""

    schema_version: Literal[1] = 1
    overlay_version: Literal["portfolio-core-r3-overlay-v1"] = R3_OVERLAY_VERSION
    run_id: str
    source_r2_sha256: Sha256
    repair_plan_sha256: Sha256
    source_query_count: Literal[1500] = _SOURCE_QUERY_COUNT
    replacement_query_count: Literal[250] = _REPLACEMENT_COUNT
    carry_forward_query_count: Literal[1250] = _CARRY_FORWARD_COUNT
    parent_batch_count: Literal[60] = _PARENT_BATCH_COUNT
    repair_batch_count: Literal[10] = _REPAIR_BATCH_COUNT
    queries_sha256: Sha256
    parent_batch_index_sha256: Sha256
    repair_batch_index_sha256: Sha256
    language_validation_sha256: Sha256
    repair_ledger_sha256: Sha256
    repair_checkpoint_sha256: Sha256
    file_count: int = Field(gt=0)
    files: tuple[R3PublishedFile, ...]
    manifest_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _safe_identifier(value, "run_id")

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        paths = tuple(item.relative_path for item in self.files)
        if (
            self.file_count != len(self.files)
            or not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or "repair-manifest.json" in paths
        ):
            raise ValueError("root manifest file collection is invalid")
        if self.manifest_sha256 != _digest_without(
            self.model_dump(mode="json"), "manifest_sha256"
        ):
            raise ValueError("root repair manifest self-hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: object) -> Self:
        payload = {
            "schema_version": 1,
            "overlay_version": R3_OVERLAY_VERSION,
            "source_query_count": _SOURCE_QUERY_COUNT,
            "replacement_query_count": _REPLACEMENT_COUNT,
            "carry_forward_query_count": _CARRY_FORWARD_COUNT,
            "parent_batch_count": _PARENT_BATCH_COUNT,
            "repair_batch_count": _REPAIR_BATCH_COUNT,
            **payload,
        }
        return cls(
            **payload,
            manifest_sha256=_digest_without(payload, "manifest_sha256"),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


@dataclass(frozen=True)
class _ArtifactBytes:
    relative_path: str
    content: bytes


@dataclass(frozen=True)
class R3OverlayPayload:
    """Fully verified in-memory overlay ready for create-only publication."""

    run_id: str
    source_r2_sha256: str
    repair_plan: R3RepairPlan
    final_queries: tuple[Query, ...]
    manifest: R3RepairManifest
    artifacts: tuple[_ArtifactBytes, ...]


@dataclass(frozen=True)
class R3OverlayPublication:
    """Result of a create-only r3 overlay publication."""

    run_root: Path
    manifest: R3RepairManifest
    created: bool
    incident_staging_path: Path | None = None


def _parse_canonical_queries(content: bytes, *, label: str) -> tuple[Query, ...]:
    try:
        lines = content.splitlines()
    except AttributeError as exc:  # pragma: no cover - API type guard
        raise PortfolioCoreR3OverlayError(f"{label} must be bytes") from exc
    if not lines or any(not line.strip() for line in lines):
        raise PortfolioCoreR3OverlayError(f"{label} must be non-empty canonical JSONL")
    records: list[Query] = []
    for line in lines:
        try:
            records.append(Query.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise PortfolioCoreR3OverlayError(f"{label} contains an invalid Query") from exc
    parsed = tuple(records)
    if canonical_jsonl_bytes(parsed) != content:
        raise PortfolioCoreR3OverlayError(f"{label} is not canonical Query JSONL")
    return parsed


def _canonical_source(
    *,
    source_queries: Sequence[Query] | None,
    source_queries_bytes: bytes | None,
) -> tuple[tuple[Query, ...], bytes]:
    if (source_queries is None) == (source_queries_bytes is None):
        raise PortfolioCoreR3OverlayError(
            "provide exactly one of source_queries or source_queries_bytes"
        )
    if source_queries_bytes is not None:
        return (
            _parse_canonical_queries(source_queries_bytes, label="source query JSONL"),
            source_queries_bytes,
        )

    assert source_queries is not None
    if isinstance(source_queries, (str, bytes)) or not isinstance(
        source_queries, Sequence
    ):
        raise PortfolioCoreR3OverlayError("source_queries must be a Query sequence")
    if any(not isinstance(query, Query) for query in source_queries):
        raise PortfolioCoreR3OverlayError("source_queries must contain only Query values")
    # Reparse canonical bytes so model_construct() and equivalent caller-side
    # shortcuts cannot bypass the current schema-v2 validator.
    content = canonical_jsonl_bytes(source_queries)
    return _parse_canonical_queries(content, label="source Query rows"), content


def _source_parent_batches(
    source_queries: Sequence[Query],
) -> tuple[tuple[str, tuple[Query, ...]], ...]:
    if len(source_queries) != _SOURCE_QUERY_COUNT:
        raise PortfolioCoreR3OverlayError("r3 source must contain exactly 1,500 Query rows")
    query_ids = tuple(query.query_id for query in source_queries)
    if len(set(query_ids)) != _SOURCE_QUERY_COUNT:
        raise PortfolioCoreR3OverlayError("r3 source query IDs must be globally unique")

    groups: list[tuple[str, tuple[Query, ...]]] = []
    current_id: str | None = None
    current_rows: list[Query] = []
    seen: set[str] = set()
    for query in source_queries:
        parent_id = _safe_identifier(query.generator_batch_id, "source generator_batch_id")
        if current_id is None:
            current_id = parent_id
        if parent_id != current_id:
            if current_id in seen:
                raise PortfolioCoreR3OverlayError("source parent batches must be contiguous")
            seen.add(current_id)
            groups.append((current_id, tuple(current_rows)))
            current_id = parent_id
            current_rows = []
        current_rows.append(query)
    assert current_id is not None  # source count was checked above
    if current_id in seen:
        raise PortfolioCoreR3OverlayError("source parent batches must be contiguous")
    groups.append((current_id, tuple(current_rows)))

    if len(groups) != _PARENT_BATCH_COUNT or any(
        len(rows) != _BATCH_SIZE for _, rows in groups
    ):
        raise PortfolioCoreR3OverlayError(
            "r3 source must contain 60 ordered parent batches of 25 rows"
        )
    return tuple(groups)


def _validated_repair_plan(plan: R3RepairPlan) -> R3RepairPlan:
    if not isinstance(plan, R3RepairPlan):
        raise PortfolioCoreR3OverlayError("repair_plan must be an R3RepairPlan")
    try:
        return R3RepairPlan.model_validate(plan.model_dump(mode="json"))
    except (ValidationError, ValueError) as exc:
        raise PortfolioCoreR3OverlayError("repair plan validation failed") from exc


def _validated_draft_batches(
    drafts: Sequence[R3RepairDraftBatch],
) -> tuple[R3RepairDraftBatch, ...]:
    if isinstance(drafts, (str, bytes)) or not isinstance(drafts, Sequence):
        raise PortfolioCoreR3OverlayError("draft_batches must be a sequence")
    parsed: list[R3RepairDraftBatch] = []
    for item in drafts:
        try:
            raw = item.model_dump(mode="json") if isinstance(item, R3RepairDraftBatch) else item
            parsed.append(R3RepairDraftBatch.model_validate(raw))
        except (ValidationError, ValueError, AttributeError, TypeError) as exc:
            raise PortfolioCoreR3OverlayError("repair draft batch validation failed") from exc
    return tuple(parsed)


def _validate_plan_drafts_and_source_binding(
    *,
    source_queries: Sequence[Query],
    repair_plan: R3RepairPlan,
    draft_batches: Sequence[R3RepairDraftBatch],
) -> tuple[R3RepairPlan, tuple[R3RepairDraftBatch, ...]]:
    plan = _validated_repair_plan(repair_plan)
    drafts = _validated_draft_batches(draft_batches)
    if len(drafts) != _REPAIR_BATCH_COUNT:
        raise PortfolioCoreR3OverlayError("r3 requires exactly ten repair draft batches")
    if tuple(item.batch_id for item in drafts) != tuple(
        item.batch_id for item in plan.batches
    ):
        raise PortfolioCoreR3OverlayError("repair draft batch IDs/order do not match frozen plan")

    source_by_id = {query.query_id: query for query in source_queries}
    plan_ids = tuple(
        query_id for batch in plan.batches for query_id in batch.query_ids
    )
    if any(query_id not in source_by_id for query_id in plan_ids):
        raise PortfolioCoreR3OverlayError("repair plan refers to a query absent from r2 source")
    source_positions = {query.query_id: index for index, query in enumerate(source_queries)}
    if tuple(source_positions[query_id] for query_id in plan_ids) != tuple(
        sorted(source_positions[query_id] for query_id in plan_ids)
    ):
        raise PortfolioCoreR3OverlayError(
            "repair plan query IDs must preserve r2 source query order"
        )

    for planned, submitted in zip(plan.batches, drafts, strict=True):
        submitted_ids = tuple(item.query_id for item in submitted.drafts)
        if submitted_ids != planned.query_ids:
            raise PortfolioCoreR3OverlayError(
                "repair draft query IDs/order do not match frozen repair batch"
            )
        for draft in submitted.drafts:
            _validate_turn_language(draft.turns, query_id=draft.query_id)
            source_turns = source_by_id[draft.query_id].turns
            if _role_shape(draft.turns) != _role_shape(source_turns):
                raise PortfolioCoreR3OverlayError(
                    "replacement turn role shape must match r2 source: "
                    f"{draft.query_id}"
                )
            if canonical_turns_sha256(draft.turns) == canonical_turns_sha256(source_turns):
                raise PortfolioCoreR3OverlayError(
                    f"replacement turns must differ from r2 source: {draft.query_id}"
                )
    return plan, drafts


def _materialize_replacement(source: Query, draft: R3ReplacementDraft) -> Query:
    payload = source.model_dump(mode="json")
    payload["turns"] = [turn.model_dump(mode="json") for turn in draft.turns]
    payload["text"] = draft.turns[-1].content
    try:
        replacement = Query.model_validate(payload)
    except (ValidationError, ValueError) as exc:  # pragma: no cover - input guards above
        raise PortfolioCoreR3OverlayError("replacement Query schema validation failed") from exc
    source_payload = source.model_dump(mode="json")
    replacement_payload = replacement.model_dump(mode="json")
    if any(
        source_payload[field_name] != replacement_payload[field_name]
        for field_name in source_payload
        if field_name not in {"text", "turns"}
    ):
        raise PortfolioCoreR3OverlayError(
            f"replacement metadata drifted from r2 source: {source.query_id}"
        )
    return replacement


def _materialize_final_queries(
    source_queries: Sequence[Query],
    draft_batches: Sequence[R3RepairDraftBatch],
) -> tuple[Query, ...]:
    draft_by_id = {
        draft.query_id: draft
        for batch in draft_batches
        for draft in batch.drafts
    }
    final_queries = tuple(
        _materialize_replacement(query, draft_by_id[query.query_id])
        if query.query_id in draft_by_id
        else query
        for query in source_queries
    )
    if len(final_queries) != _SOURCE_QUERY_COUNT:
        raise AssertionError("source cardinality was validated before materialisation")
    for query in final_queries:
        # Query's own validator guarantees text == the final user turn.  This
        # explicitly repeats the language requirement for both carried rows and
        # replacement rows before anything can be staged.
        _require_valid_role_shape(query.turns)
        _validate_turn_language(query.turns, query_id=query.query_id)
        if query.text != query.turns[-1].content:
            raise PortfolioCoreR3OverlayError("final Query text/turn binding drifted")
    normalized = tuple(normalized_final_text(query.text) for query in final_queries)
    if any(not value for value in normalized):
        raise PortfolioCoreR3OverlayError("final query contains empty normalized text")
    if len(set(normalized)) != len(normalized):
        raise PortfolioCoreR3OverlayError("final corpus has duplicate normalized text")
    return final_queries


def _language_validation_bytes(
    final_queries: Sequence[Query], *, replacement_ids: set[str]
) -> bytes:
    entries = tuple(
        R3LanguageValidationEntry(
            query_id=query.query_id,
            disposition=(
                "replacement" if query.query_id in replacement_ids else "source_carry_forward"
            ),
            turns_sha256=canonical_turns_sha256(query.turns),
            final_text_sha256=sha256_bytes(canonical_json_bytes({"text": query.text})),
            nfkc_final_text_sha256=sha256_bytes(
                canonical_json_bytes({"text": unicodedata.normalize("NFKC", query.text)})
            ),
            normalized_final_text_sha256=sha256_bytes(
                canonical_json_bytes({"text": normalized_final_text(query.text)})
            ),
        )
        for query in final_queries
    )
    return canonical_jsonl_bytes(entries)


def _artifact(relative_path: str, content: bytes) -> _ArtifactBytes:
    return _ArtifactBytes(relative_path=_safe_relative_path(relative_path), content=content)


def build_portfolio_core_r3_overlay_payload(
    *,
    source_queries: Sequence[Query] | None = None,
    source_queries_bytes: bytes | None = None,
    expected_r2_source_sha256: str,
    repair_plan: R3RepairPlan,
    draft_batches: Sequence[R3RepairDraftBatch],
    run_id: str,
) -> R3OverlayPayload:
    """Build one complete validated r3 overlay without writing a file.

    ``expected_r2_source_sha256`` is an external frozen binding.  The supplied
    source is always canonicalized and hashed before drafts are considered, so
    any r2 source drift fails closed.  A caller may supply rows *or* canonical
    JSONL bytes, never a filesystem path.
    """

    run_id = _safe_identifier(run_id, "run_id")
    expected_r2_source_sha256 = _require_sha256(
        expected_r2_source_sha256, "expected_r2_source_sha256"
    )
    source, source_bytes = _canonical_source(
        source_queries=source_queries,
        source_queries_bytes=source_queries_bytes,
    )
    source_r2_sha256 = sha256_bytes(source_bytes)
    if source_r2_sha256 != expected_r2_source_sha256:
        raise PortfolioCoreR3OverlayError("r2 source SHA-256 drifted from frozen value")
    parent_batches = _source_parent_batches(source)
    plan, drafts = _validate_plan_drafts_and_source_binding(
        source_queries=source,
        repair_plan=repair_plan,
        draft_batches=draft_batches,
    )
    final_queries = _materialize_final_queries(source, drafts)
    final_by_id = {query.query_id: query for query in final_queries}
    source_by_id = {query.query_id: query for query in source}
    replacement_ids = {
        query_id for batch in plan.batches for query_id in batch.query_ids
    }
    if len(replacement_ids) != _REPLACEMENT_COUNT:
        raise AssertionError("repair plan validator guarantees 250 replacement IDs")
    queries_bytes = canonical_jsonl_bytes(final_queries)
    language_bytes = _language_validation_bytes(
        final_queries, replacement_ids=replacement_ids
    )

    source_parent_hash_by_id = {
        parent_id: sha256_bytes(canonical_jsonl_bytes(rows))
        for parent_id, rows in parent_batches
    }
    parent_index_entries: list[R3ParentBatchIndexEntry] = []
    artifacts: list[_ArtifactBytes] = [
        _artifact("queries.jsonl", queries_bytes),
        _artifact("repair-plan.json", plan.canonical_bytes()),
        _artifact("language-validation.jsonl", language_bytes),
    ]
    for ordinal, (parent_id, source_rows) in enumerate(parent_batches, start=1):
        final_rows = tuple(final_by_id[row.query_id] for row in source_rows)
        final_parent_bytes = canonical_jsonl_bytes(final_rows)
        artifacts.append(_artifact(f"parent-batches/{parent_id}.jsonl", final_parent_bytes))
        parent_index_entries.append(
            R3ParentBatchIndexEntry(
                parent_batch_id=parent_id,
                ordinal=ordinal,
                query_ids_sha256=_query_ids_sha256(
                    tuple(row.query_id for row in source_rows)
                ),
                source_parent_batch_sha256=source_parent_hash_by_id[parent_id],
                final_parent_batch_sha256=sha256_bytes(final_parent_bytes),
                replacement_count=sum(
                    row.query_id in replacement_ids for row in source_rows
                ),
            )
        )
    parent_index_bytes = canonical_jsonl_bytes(parent_index_entries)
    artifacts.append(_artifact("parent-batches/index.jsonl", parent_index_bytes))

    parent_order = tuple(parent_id for parent_id, _ in parent_batches)
    parent_for_query = {
        query.query_id: parent_id
        for parent_id, rows in parent_batches
        for query in rows
    }
    repair_index_entries: list[R3RepairBatchIndexEntry] = []
    ledger_entries: list[R3RepairLedgerEntry] = []
    for ordinal, (planned, submitted) in enumerate(
        zip(plan.batches, drafts, strict=True), start=1
    ):
        source_rows = tuple(source_by_id[query_id] for query_id in planned.query_ids)
        repaired_rows = tuple(final_by_id[query_id] for query_id in planned.query_ids)
        draft_bytes = canonical_jsonl_bytes(submitted.drafts)
        repair_results_bytes = canonical_jsonl_bytes(repaired_rows)
        touched_parent_ids = tuple(
            parent_id
            for parent_id in parent_order
            if any(parent_for_query[query_id] == parent_id for query_id in planned.query_ids)
        )
        parent_hashes = tuple(
            R3ParentSourceHash(
                parent_batch_id=parent_id,
                source_parent_batch_sha256=source_parent_hash_by_id[parent_id],
            )
            for parent_id in touched_parent_ids
        )
        batch_manifest = R3RepairBatchManifest.create(
            batch_id=planned.batch_id,
            ordinal=ordinal,
            source_r2_sha256=source_r2_sha256,
            repair_plan_sha256=plan.repair_plan_sha256,
            query_ids_sha256=_query_ids_sha256(planned.query_ids),
            source_selected_queries_sha256=sha256_bytes(
                canonical_jsonl_bytes(source_rows)
            ),
            drafts_sha256=sha256_bytes(draft_bytes),
            repair_results_sha256=sha256_bytes(repair_results_bytes),
            parent_source_hashes=parent_hashes,
        )
        batch_manifest_bytes = batch_manifest.canonical_bytes()
        ledger_entry = R3RepairLedgerEntry.create(
            batch_id=planned.batch_id,
            ordinal=ordinal,
            source_r2_sha256=source_r2_sha256,
            repair_plan_sha256=plan.repair_plan_sha256,
            repair_results_sha256=batch_manifest.repair_results_sha256,
            batch_manifest_sha256=sha256_bytes(batch_manifest_bytes),
        )
        repair_index_entries.append(
            R3RepairBatchIndexEntry(
                batch_id=planned.batch_id,
                ordinal=ordinal,
                repair_results_sha256=batch_manifest.repair_results_sha256,
                batch_manifest_sha256=sha256_bytes(batch_manifest_bytes),
            )
        )
        ledger_entries.append(ledger_entry)
        artifacts.extend(
            (
                _artifact(
                    f"repair-batches/{planned.batch_id}/results.jsonl",
                    repair_results_bytes,
                ),
                _artifact(
                    f"repair-batches/{planned.batch_id}/manifest.json",
                    batch_manifest_bytes,
                ),
            )
        )

    repair_index_bytes = canonical_jsonl_bytes(repair_index_entries)
    repair_ledger_bytes = canonical_jsonl_bytes(ledger_entries)
    checkpoint = R3RepairCheckpoint.create(
        source_r2_sha256=source_r2_sha256,
        repair_plan_sha256=plan.repair_plan_sha256,
        repair_ledger_sha256=sha256_bytes(repair_ledger_bytes),
        queries_sha256=sha256_bytes(queries_bytes),
        accepted_batch_ids=tuple(item.batch_id for item in plan.batches),
    )
    checkpoint_bytes = checkpoint.canonical_bytes()
    artifacts.extend(
        (
            _artifact("repair-batches/index.jsonl", repair_index_bytes),
            _artifact("repair-ledger.jsonl", repair_ledger_bytes),
            _artifact("repair-checkpoint.json", checkpoint_bytes),
        )
    )
    artifacts = sorted(artifacts, key=lambda item: item.relative_path)
    if len({item.relative_path for item in artifacts}) != len(artifacts):
        raise AssertionError("r3 artifact paths must be unique")

    manifest = R3RepairManifest.create(
        run_id=run_id,
        source_r2_sha256=source_r2_sha256,
        repair_plan_sha256=plan.repair_plan_sha256,
        queries_sha256=sha256_bytes(queries_bytes),
        parent_batch_index_sha256=sha256_bytes(parent_index_bytes),
        repair_batch_index_sha256=sha256_bytes(repair_index_bytes),
        language_validation_sha256=sha256_bytes(language_bytes),
        repair_ledger_sha256=sha256_bytes(repair_ledger_bytes),
        repair_checkpoint_sha256=sha256_bytes(checkpoint_bytes),
        file_count=len(artifacts),
        files=tuple(
            R3PublishedFile(
                relative_path=item.relative_path,
                sha256=sha256_bytes(item.content),
                bytes=len(item.content),
            )
            for item in artifacts
        ),
    )
    all_artifacts = tuple(
        [
            *artifacts,
            _artifact("repair-manifest.json", manifest.canonical_bytes()),
        ]
    )
    return R3OverlayPayload(
        run_id=run_id,
        source_r2_sha256=source_r2_sha256,
        repair_plan=plan,
        final_queries=final_queries,
        manifest=manifest,
        artifacts=all_artifacts,
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise R3OverlaySafetyError(f"unable to inspect {label}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise R3OverlaySafetyError(f"{label} must be a non-reparse real directory")


def _require_safe_target_path(target: Path) -> None:
    if not os.path.lexists(target):
        return
    metadata = target.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise R3OverlaySafetyError("r3 publication target must not be a link or reparse")
    if not stat.S_ISDIR(metadata.st_mode):
        raise R3OverlayConflictError("r3 publication target exists but is not a directory")


def _expected_directories(artifacts: Sequence[_ArtifactBytes]) -> set[str]:
    directories: set[str] = set()
    for artifact in artifacts:
        parent = PurePosixPath(artifact.relative_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _collect_tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    _require_real_directory(root, "r3 publication root")
    files: dict[str, Path] = {}
    directories: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda item: item.name)
        for entry in entries:
            path = Path(entry.path)
            relative = entry.name if not prefix else f"{prefix}/{entry.name}"
            _safe_relative_path(relative)
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                raise R3OverlaySafetyError("r3 publication tree contains a link or reparse")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(path, relative)
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = path
            else:
                raise R3OverlaySafetyError("r3 publication tree contains a special entry")

    visit(root, "")
    return files, directories


def _parse_canonical_model(
    content: bytes,
    model_type: type[_StrictFrozenModel],
    *,
    label: str,
) -> _StrictFrozenModel:
    try:
        parsed = model_type.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise R3OverlayPublicationError(f"{label} violates its schema") from exc
    if canonical_json_bytes(parsed) != content:
        raise R3OverlayPublicationError(f"{label} is not canonical JSON")
    return parsed


def _parse_canonical_jsonl_models(
    content: bytes,
    model_type: type[_StrictFrozenModel],
    *,
    label: str,
) -> tuple[_StrictFrozenModel, ...]:
    lines = content.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise R3OverlayPublicationError(f"{label} is not non-empty canonical JSONL")
    records: list[_StrictFrozenModel] = []
    for line in lines:
        try:
            records.append(model_type.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise R3OverlayPublicationError(f"{label} violates its schema") from exc
    parsed = tuple(records)
    if canonical_jsonl_bytes(parsed) != content:
        raise R3OverlayPublicationError(f"{label} is not canonical JSONL")
    return parsed


def _verify_overlay_tree(root: Path, payload: R3OverlayPayload) -> None:
    """Verify every staged artifact without trusting directory enumeration alone."""

    expected_files = {item.relative_path: item.content for item in payload.artifacts}
    actual_files, actual_directories = _collect_tree(root)
    if set(actual_files) != set(expected_files):
        raise R3OverlayConflictError("r3 publication has a different file collection")
    if actual_directories != _expected_directories(payload.artifacts):
        raise R3OverlayConflictError("r3 publication has a different directory collection")
    for relative_path, expected in expected_files.items():
        actual = actual_files[relative_path].read_bytes()
        if actual != expected:
            raise R3OverlayConflictError(
                f"r3 publication content differs: {relative_path}"
            )

    # Parsing all emitted machine contracts catches a bad writer or a future
    # accidental change in serialization even when a local expected payload was
    # built in the same process.
    manifest = _parse_canonical_model(
        expected_files["repair-manifest.json"], R3RepairManifest, label="repair manifest"
    )
    if manifest != payload.manifest:
        raise R3OverlayPublicationError("repair manifest does not match payload")
    expected_manifest_files = tuple(
        R3PublishedFile(
            relative_path=artifact.relative_path,
            sha256=sha256_bytes(artifact.content),
            bytes=len(artifact.content),
        )
        for artifact in sorted(
            (
                artifact
                for artifact in payload.artifacts
                if artifact.relative_path != "repair-manifest.json"
            ),
            key=lambda artifact: artifact.relative_path,
        )
    )
    if (
        manifest.run_id != payload.run_id
        or manifest.source_r2_sha256 != payload.source_r2_sha256
        or manifest.repair_plan_sha256 != payload.repair_plan.repair_plan_sha256
        or manifest.files != expected_manifest_files
    ):
        raise R3OverlayPublicationError("repair manifest root binding drifted")
    plan = _parse_canonical_model(
        expected_files["repair-plan.json"], R3RepairPlan, label="repair plan"
    )
    if plan != payload.repair_plan:
        raise R3OverlayPublicationError("repair plan does not match payload")
    try:
        queries = _parse_canonical_queries(expected_files["queries.jsonl"], label="queries")
    except PortfolioCoreR3OverlayError as exc:
        raise R3OverlayPublicationError("queries artifact validation failed") from exc
    if queries != payload.final_queries or len(queries) != _SOURCE_QUERY_COUNT:
        raise R3OverlayPublicationError("queries artifact does not match payload")
    final_by_id = {query.query_id: query for query in queries}
    if len(final_by_id) != _SOURCE_QUERY_COUNT:
        raise R3OverlayPublicationError("queries artifact query IDs are not globally unique")
    parent_index = _parse_canonical_jsonl_models(
        expected_files["parent-batches/index.jsonl"],
        R3ParentBatchIndexEntry,
        label="parent batch index",
    )
    repair_index = _parse_canonical_jsonl_models(
        expected_files["repair-batches/index.jsonl"],
        R3RepairBatchIndexEntry,
        label="repair batch index",
    )
    ledger = _parse_canonical_jsonl_models(
        expected_files["repair-ledger.jsonl"],
        R3RepairLedgerEntry,
        label="repair ledger",
    )
    language = _parse_canonical_jsonl_models(
        expected_files["language-validation.jsonl"],
        R3LanguageValidationEntry,
        label="language validation",
    )
    checkpoint = _parse_canonical_model(
        expected_files["repair-checkpoint.json"],
        R3RepairCheckpoint,
        label="repair checkpoint",
    )
    if (
        len(parent_index) != _PARENT_BATCH_COUNT
        or tuple(item.ordinal for item in parent_index)
        != tuple(range(1, _PARENT_BATCH_COUNT + 1))
        or len(repair_index) != _REPAIR_BATCH_COUNT
        or tuple(item.ordinal for item in repair_index)
        != tuple(range(1, _REPAIR_BATCH_COUNT + 1))
        or len(ledger) != _REPAIR_BATCH_COUNT
        or tuple(item.ordinal for item in ledger)
        != tuple(range(1, _REPAIR_BATCH_COUNT + 1))
        or len(language) != _SOURCE_QUERY_COUNT
    ):
        raise R3OverlayPublicationError("r3 staged artifact cardinality or order drifted")
    assert isinstance(checkpoint, R3RepairCheckpoint)
    if (
        tuple(item.batch_id for item in repair_index)
        != tuple(item.batch_id for item in payload.repair_plan.batches)
        or tuple(item.batch_id for item in ledger)
        != tuple(item.batch_id for item in payload.repair_plan.batches)
        or checkpoint.accepted_batch_ids
        != tuple(item.batch_id for item in payload.repair_plan.batches)
    ):
        raise R3OverlayPublicationError("r3 repair batch identity/order drifted")
    if (
        manifest.queries_sha256 != sha256_bytes(expected_files["queries.jsonl"])
        or manifest.parent_batch_index_sha256
        != sha256_bytes(expected_files["parent-batches/index.jsonl"])
        or manifest.repair_batch_index_sha256
        != sha256_bytes(expected_files["repair-batches/index.jsonl"])
        or manifest.language_validation_sha256
        != sha256_bytes(expected_files["language-validation.jsonl"])
        or manifest.repair_ledger_sha256
        != sha256_bytes(expected_files["repair-ledger.jsonl"])
        or manifest.repair_checkpoint_sha256
        != sha256_bytes(expected_files["repair-checkpoint.json"])
        or checkpoint.repair_ledger_sha256
        != sha256_bytes(expected_files["repair-ledger.jsonl"])
        or checkpoint.queries_sha256 != sha256_bytes(expected_files["queries.jsonl"])
    ):
        raise R3OverlayPublicationError("r3 root hash binding drifted")
    replacement_ids = {
        query_id
        for batch in payload.repair_plan.batches
        for query_id in batch.query_ids
    }
    if any(query_id not in final_by_id for query_id in replacement_ids):
        raise R3OverlayPublicationError("repair plan refers to a missing final query")
    if tuple(item.query_id for item in language) != tuple(query.query_id for query in queries):
        raise R3OverlayPublicationError("language validation query order drifted")
    for query, entry in zip(queries, language, strict=True):
        assert isinstance(entry, R3LanguageValidationEntry)
        _require_valid_role_shape(query.turns)
        try:
            _validate_turn_language(query.turns, query_id=query.query_id)
        except PortfolioCoreR3OverlayError as exc:
            raise R3OverlayPublicationError("language validation no longer holds") from exc
        expected_disposition = (
            "replacement"
            if query.query_id in replacement_ids
            else "source_carry_forward"
        )
        if (
            entry.disposition != expected_disposition
            or entry.turns_sha256 != canonical_turns_sha256(query.turns)
            or entry.final_text_sha256
            != sha256_bytes(canonical_json_bytes({"text": query.text}))
            or entry.nfkc_final_text_sha256
            != sha256_bytes(
                canonical_json_bytes(
                    {"text": unicodedata.normalize("NFKC", query.text)}
                )
            )
            or entry.normalized_final_text_sha256
            != sha256_bytes(
                canonical_json_bytes({"text": normalized_final_text(query.text)})
            )
        ):
            raise R3OverlayPublicationError("language validation receipt drifted")
    normalized = tuple(normalized_final_text(query.text) for query in queries)
    if len(set(normalized)) != len(normalized):
        raise R3OverlayPublicationError("staged corpus has duplicate normalized text")

    reconstructed_parent_queries: list[Query] = []
    for parent in parent_index:
        assert isinstance(parent, R3ParentBatchIndexEntry)
        try:
            parent_queries = _parse_canonical_queries(
                expected_files[f"parent-batches/{parent.parent_batch_id}.jsonl"],
                label="parent batch",
            )
        except PortfolioCoreR3OverlayError as exc:
            raise R3OverlayPublicationError("parent batch artifact validation failed") from exc
        if (
            len(parent_queries) != _BATCH_SIZE
            or any(
                query.generator_batch_id != parent.parent_batch_id
                for query in parent_queries
            )
            or parent.query_ids_sha256
            != _query_ids_sha256(tuple(query.query_id for query in parent_queries))
            or parent.final_parent_batch_sha256
            != sha256_bytes(
                expected_files[f"parent-batches/{parent.parent_batch_id}.jsonl"]
            )
        ):
            raise R3OverlayPublicationError("parent batch binding drifted")
        reconstructed_parent_queries.extend(parent_queries)
    if tuple(reconstructed_parent_queries) != queries:
        raise R3OverlayPublicationError("parent batch files do not reconstruct queries.jsonl")

    repair_manifest_by_id: dict[str, R3RepairBatchManifest] = {}
    for repair in repair_index:
        assert isinstance(repair, R3RepairBatchIndexEntry)
        try:
            repair_queries = _parse_canonical_queries(
                expected_files[f"repair-batches/{repair.batch_id}/results.jsonl"],
                label="repair batch results",
            )
        except PortfolioCoreR3OverlayError as exc:
            raise R3OverlayPublicationError("repair results artifact validation failed") from exc
        batch_manifest = _parse_canonical_model(
            expected_files[f"repair-batches/{repair.batch_id}/manifest.json"],
            R3RepairBatchManifest,
            label="repair batch manifest",
        )
        assert isinstance(batch_manifest, R3RepairBatchManifest)
        planned = payload.repair_plan.batches[repair.ordinal - 1]
        result_bytes = expected_files[f"repair-batches/{repair.batch_id}/results.jsonl"]
        manifest_bytes = expected_files[f"repair-batches/{repair.batch_id}/manifest.json"]
        if (
            repair.batch_id != planned.batch_id
            or len(repair_queries) != _BATCH_SIZE
            or tuple(query.query_id for query in repair_queries) != planned.query_ids
            or tuple(repair_queries)
            != tuple(final_by_id[query_id] for query_id in planned.query_ids)
            or repair.repair_results_sha256 != sha256_bytes(result_bytes)
            or repair.batch_manifest_sha256 != sha256_bytes(manifest_bytes)
            or batch_manifest.batch_id != repair.batch_id
            or batch_manifest.ordinal != repair.ordinal
            or batch_manifest.query_ids_sha256 != _query_ids_sha256(planned.query_ids)
            or batch_manifest.repair_results_sha256 != sha256_bytes(result_bytes)
            or batch_manifest.source_r2_sha256 != payload.source_r2_sha256
            or batch_manifest.repair_plan_sha256
            != payload.repair_plan.repair_plan_sha256
        ):
            raise R3OverlayPublicationError("repair batch binding drifted")
        repair_manifest_by_id[repair.batch_id] = batch_manifest
    for item, index in zip(ledger, repair_index, strict=True):
        assert isinstance(item, R3RepairLedgerEntry)
        batch_manifest = repair_manifest_by_id[index.batch_id]
        if (
            item.batch_id != index.batch_id
            or item.repair_results_sha256 != index.repair_results_sha256
            or item.batch_manifest_sha256 != index.batch_manifest_sha256
            or item.source_r2_sha256 != payload.source_r2_sha256
            or item.repair_plan_sha256 != payload.repair_plan.repair_plan_sha256
            or item.batch_manifest_sha256 != sha256_bytes(batch_manifest.canonical_bytes())
        ):
            raise R3OverlayPublicationError("repair ledger binding drifted")


def _write_staging_tree(staging: Path, payload: R3OverlayPayload) -> None:
    _require_real_directory(staging, "r3 publication staging directory")
    for artifact in payload.artifacts:
        relative = PurePosixPath(artifact.relative_path)
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _require_real_directory(target.parent, "r3 publication staging parent")
        try:
            with target.open("xb") as handle:
                handle.write(artifact.content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:  # pragma: no cover - hostile staging race
            raise R3OverlaySafetyError("r3 staging already contains an artifact") from exc
    _verify_overlay_tree(staging, payload)


def publish_portfolio_core_r3_overlay(
    payload: R3OverlayPayload,
    *,
    run_parent: str | Path,
) -> R3OverlayPublication:
    """Atomically create-only publish a fully built r3 overlay.

    On a target collision, publication succeeds only if every expected file,
    directory, canonical model, and byte is identical.  Failed staging trees
    are intentionally retained for inspection; this function never removes an
    existing run or a prior staging directory.
    """

    if not isinstance(payload, R3OverlayPayload):
        raise PortfolioCoreR3OverlayError("payload must be an R3OverlayPayload")
    parent = Path(run_parent)
    _require_real_directory(parent, "r3 publication parent")
    target = parent / _safe_identifier(payload.run_id, "payload run_id")
    _require_safe_target_path(target)
    if os.path.lexists(target):
        _verify_overlay_tree(target, payload)
        return R3OverlayPublication(
            run_root=target,
            manifest=payload.manifest,
            created=False,
        )

    try:
        staging = new_staging_directory(target)
    except FileExistsError:
        _require_safe_target_path(target)
        _verify_overlay_tree(target, payload)
        return R3OverlayPublication(
            run_root=target,
            manifest=payload.manifest,
            created=False,
        )
    try:
        _write_staging_tree(staging, payload)
        try:
            atomic_publish_new_directory(staging, target)
        except FileExistsError:
            _require_safe_target_path(target)
            _verify_overlay_tree(target, payload)
            return R3OverlayPublication(
                run_root=target,
                manifest=payload.manifest,
                created=False,
                incident_staging_path=staging,
            )
    except R3OverlayPublicationError:
        raise
    except Exception as exc:
        raise R3OverlayPublicationError(
            "r3 publication failed; staging retained for inspection",
            staging_path=staging,
        ) from exc
    return R3OverlayPublication(
        run_root=target,
        manifest=payload.manifest,
        created=True,
    )


def create_only_publish_portfolio_core_r3_overlay(
    *,
    source_queries: Sequence[Query] | None = None,
    source_queries_bytes: bytes | None = None,
    expected_r2_source_sha256: str,
    repair_plan: R3RepairPlan,
    draft_batches: Sequence[R3RepairDraftBatch],
    run_id: str,
    run_parent: str | Path,
) -> R3OverlayPublication:
    """Build, fully validate, and create-only publish a single r3 overlay."""

    payload = build_portfolio_core_r3_overlay_payload(
        source_queries=source_queries,
        source_queries_bytes=source_queries_bytes,
        expected_r2_source_sha256=expected_r2_source_sha256,
        repair_plan=repair_plan,
        draft_batches=draft_batches,
        run_id=run_id,
    )
    return publish_portfolio_core_r3_overlay(payload, run_parent=run_parent)
