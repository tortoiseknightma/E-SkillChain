"""External trust roots for formal wiki and recipe source adapters.

The lock file is an input approved outside the adapter.  Merely constructing a
``KBSourceLock`` object, hashing source bytes inside a builder, or asking an LLM
to attest provenance never creates a verified lock: formal entry points require
the canonical lock bytes *and* an independently supplied expected digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Annotated, Any, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
KBSourceKind = Literal["encyclopedia", "recipe"]
KBAdapterId = Literal["wiki-zh-filtered-v1", "xiachufang-recipe-zip-v1"]
ReviewerKind = Literal["human", "publisher"]

_MUTABLE_TOKEN = re.compile(
    r"(?:^|[-_./:=?&#])"
    r"(?:main|master|head|latest|current|unknown|unverified|unpinned|mutable|tbd)"
    r"(?:$|[-_./:=?&#])",
    flags=re.IGNORECASE,
)
_UNKNOWN_LICENSES = {
    "",
    "unknown",
    "unknown-unverified",
    "none",
    "null",
    "unlicensed",
    "tbd",
}
_LLM_REVIEWER_TOKEN = re.compile(
    r"(?:^|[-_./ ])(?:llm|gpt|chatgpt|claude|gemini|codex|fable|model)"
    r"(?:$|[-_./ ])",
    flags=re.IGNORECASE,
)


class KBSourceLockError(ValueError):
    """A formal KB source or its external provenance lock is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be blank")
    return cleaned


def _validate_absolute_uri(value: str, name: str) -> str:
    value = _nonblank(value, name)
    parsed = urlsplit(value)
    if not parsed.scheme or any(character.isspace() for character in value):
        raise ValueError(f"{name} must be an absolute URI")
    if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname is None:
        raise ValueError(f"{name} HTTP URI must contain a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain credentials")
    return value


class KBSourceLock(_StrictFrozenModel):
    """Canonical, human/publisher-reviewed identity for one immutable raw file."""

    schema_version: Literal[1] = 1
    adapter_id: KBAdapterId
    source_kind: KBSourceKind
    source_dataset: str
    source_revision: str
    source_sha256: Sha256
    source_uri: str
    license_id: str
    license_uri: str
    attribution: str
    local_research_allowed: Literal[True]
    cloud_processing_allowed: bool
    redistribution_allowed: bool
    public_demo_allowed: bool
    reviewer_kind: ReviewerKind
    reviewer_id: str
    reviewed_at: datetime
    review_record_uri: str
    review_record_sha256: Sha256

    @field_validator(
        "source_dataset",
        "source_revision",
        "license_id",
        "attribution",
        "reviewer_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("source_revision")
    @classmethod
    def validate_immutable_revision(cls, value: str) -> str:
        if _MUTABLE_TOKEN.search(value):
            raise ValueError("source_revision must identify an immutable revision")
        return value

    @field_validator("license_id")
    @classmethod
    def validate_known_license(cls, value: str) -> str:
        if value.casefold() in _UNKNOWN_LICENSES:
            raise ValueError("license_id must be explicit and externally reviewed")
        return value

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        value = _validate_absolute_uri(value, "source_uri")
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"https", "urn", "doi", "ipfs"}:
            raise ValueError("source_uri must use an immutable-capable URI scheme")
        if _MUTABLE_TOKEN.search(unquote(value)):
            raise ValueError("source_uri must not point at a mutable revision")
        return value

    @field_validator("license_uri")
    @classmethod
    def validate_license_uri(cls, value: str) -> str:
        value = _validate_absolute_uri(value, "license_uri")
        if urlsplit(value).scheme.casefold() not in {"https", "urn"}:
            raise ValueError("license_uri must use HTTPS or URN")
        return value

    @field_validator("review_record_uri")
    @classmethod
    def validate_review_uri(cls, value: str) -> str:
        value = _validate_absolute_uri(value, "review_record_uri")
        if urlsplit(value).scheme.casefold() not in {"https", "urn", "ipfs"}:
            raise ValueError("review_record_uri must use an immutable-capable scheme")
        return value

    @field_validator("reviewer_id")
    @classmethod
    def reject_llm_reviewer(cls, value: str) -> str:
        if _LLM_REVIEWER_TOKEN.search(value):
            raise ValueError("LLM self-attestation is not a provenance review")
        return value

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def parse_json_datetime(cls, value):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("reviewed_at must be an ISO-8601 datetime") from error
        return value

    @field_validator("reviewed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_adapter_and_revision_binding(self) -> Self:
        expected_kind = {
            "wiki-zh-filtered-v1": "encyclopedia",
            "xiachufang-recipe-zip-v1": "recipe",
        }[self.adapter_id]
        if self.source_kind != expected_kind:
            raise ValueError("adapter_id does not match source_kind")
        decoded_uri = unquote(self.source_uri).casefold()
        revision = self.source_revision.casefold()
        if revision not in decoded_uri:
            raise ValueError("source_uri must bind the immutable source_revision")
        if self.source_sha256 not in decoded_uri:
            raise ValueError("source_uri must be content-addressed by source_sha256")
        if self.review_record_sha256 not in unquote(self.review_record_uri).casefold():
            raise ValueError(
                "review_record_uri must be content-addressed by review_record_sha256"
            )
        return self


@dataclass(frozen=True)
class SourceFileSnapshot:
    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class VerifiedKBSourceLock:
    """Result of matching canonical lock bytes to an external expected digest."""

    lock: KBSourceLock
    lock_sha256: str
    lock_snapshot: SourceFileSnapshot


def canonical_json_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_verified_kb_source_lock(
    path: Path | str,
    *,
    expected_lock_sha256: str,
    expected_adapter_id: KBAdapterId,
) -> VerifiedKBSourceLock:
    """Load a canonical lock only after matching a caller-owned trust root."""

    if not _is_sha256(expected_lock_sha256):
        raise KBSourceLockError(
            "formal KB cleaning requires an external expected source-lock SHA-256"
        )
    snapshot = snapshot_regular_file(path, "KB source lock")
    if snapshot.sha256 != expected_lock_sha256:
        raise KBSourceLockError(
            "KB source lock does not match external expected digest"
        )
    raw = _parse_strict_json_object(snapshot.content, "KB source lock")
    try:
        lock = KBSourceLock.model_validate(raw)
    except Exception as error:
        raise KBSourceLockError(
            "KB source lock violates the strict contract"
        ) from error
    if snapshot.content != canonical_json_bytes(lock):
        raise KBSourceLockError("KB source lock must be canonical JSON")
    if lock.adapter_id != expected_adapter_id:
        raise KBSourceLockError("KB source lock targets a different adapter")
    verify_source_snapshot(snapshot, "KB source lock")
    return VerifiedKBSourceLock(
        lock=lock,
        lock_sha256=snapshot.sha256,
        lock_snapshot=snapshot,
    )


def snapshot_regular_file(path: Path | str, label: str) -> SourceFileSnapshot:
    """Read one regular file into stable bytes while rejecting link traversal."""

    path = Path(path).absolute()
    _reject_link_path(path, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise KBSourceLockError(f"unable to inspect {label}: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise KBSourceLockError(f"{label} must be a regular non-symlink file")
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise KBSourceLockError(f"{label} changed before it could be read")
            content = source.read()
            after_open = os.fstat(source.fileno())
    except OSError as error:
        raise KBSourceLockError(f"unable to read {label}: {path}") from error
    if _file_identity(opened) != _file_identity(
        after_open
    ) or after_open.st_size != len(content):
        raise KBSourceLockError(f"{label} changed while it was read")
    try:
        after_path = path.lstat()
    except OSError as error:
        raise KBSourceLockError(f"{label} changed after it was read") from error
    if _file_identity(after_open) != _file_identity(after_path):
        raise KBSourceLockError(f"{label} changed while it was read")
    return SourceFileSnapshot(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        identity=_file_identity(after_open),
    )


def verify_source_snapshot(snapshot: SourceFileSnapshot, label: str) -> None:
    """Re-read live bytes before publish to close the adapter TOCTOU window."""

    current = snapshot_regular_file(snapshot.path, label)
    if (
        current.identity != snapshot.identity
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise KBSourceLockError(f"{label} changed during formal KB cleaning")


def publish_staged_file_create_only(
    staging: Path | str, destination: Path | str
) -> None:
    """Atomically publish a staged regular file without replacing any path."""

    staging = Path(staging)
    destination = Path(destination)
    try:
        staging_metadata = staging.lstat()
    except OSError as error:
        raise KBSourceLockError("formal KB staging file is missing") from error
    if stat.S_ISLNK(staging_metadata.st_mode) or not stat.S_ISREG(
        staging_metadata.st_mode
    ):
        raise KBSourceLockError("formal KB staging path must be a regular file")
    _reject_link_path(destination.parent.absolute(), "formal KB output parent")
    _require_real_directory(destination.parent, "formal KB output parent")
    if os.path.lexists(destination):
        raise FileExistsError(
            f"formal KB output already exists; refusing overwrite: {destination}"
        )
    try:
        os.link(staging, destination)
    except FileExistsError:
        raise FileExistsError(
            f"formal KB output already exists; refusing overwrite: {destination}"
        ) from None


def prepare_formal_output_parent(destination: Path | str) -> Path:
    """Create and validate the real parent used for a formal temporary file."""

    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_path(destination.parent, "formal KB output parent")
    _require_real_directory(destination.parent, "formal KB output parent")
    if os.path.lexists(destination):
        raise FileExistsError(
            f"formal KB output already exists; refusing overwrite: {destination}"
        )
    return destination


def citation_uri(
    base_uri: str,
    *,
    source_sha256: str,
    source_lock_sha256: str,
    locator: str,
) -> str:
    """Bind an entry citation to the immutable source bytes and record locator."""

    parsed = urlsplit(base_uri)
    suffix = (
        f"sha256={source_sha256}&source-lock-sha256={source_lock_sha256}"
        f"&record={locator}"
    )
    fragment = f"{parsed.fragment}&{suffix}" if parsed.fragment else suffix
    return parsed._replace(fragment=fragment).geturl()


def _parse_strict_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _raise_invalid_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KBSourceLockError) as error:
        raise KBSourceLockError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KBSourceLockError(f"{label} must contain one JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KBSourceLockError(f"duplicate source-lock JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str):
    raise KBSourceLockError(f"non-finite source-lock JSON value: {value}")


def _reject_link_path(path: Path, label: str) -> None:
    anchor = Path(path.anchor)
    current = anchor
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            # The final error is reported by the caller; a missing path is not a link.
            continue
        is_junction = bool(getattr(current, "is_junction", lambda: False)())
        if stat.S_ISLNK(metadata.st_mode) or is_junction:
            raise KBSourceLockError(f"{label} path must not traverse a symlink")


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise KBSourceLockError(f"{label} is missing: {path}") from error
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    if (
        stat.S_ISLNK(metadata.st_mode)
        or is_junction
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise KBSourceLockError(f"{label} must be a real non-symlink directory")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "KBSourceLock",
    "KBSourceLockError",
    "SourceFileSnapshot",
    "VerifiedKBSourceLock",
    "canonical_json_bytes",
    "citation_uri",
    "load_verified_kb_source_lock",
    "prepare_formal_output_parent",
    "publish_staged_file_create_only",
    "snapshot_regular_file",
    "verify_source_snapshot",
]
