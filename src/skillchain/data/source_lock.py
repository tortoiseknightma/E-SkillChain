"""Compact, independently re-computable locks for externally acquired datasets.

The lock stores a digest over canonical per-file descriptors instead of copying a
large file inventory into Git.  Verification walks the same declared scope and
therefore detects additions, removals, byte changes, and path changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_MAX_LOCK_BYTES = 2 * 1024 * 1024
_CHUNK_BYTES = 8 * 1024 * 1024


class SourceLockError(ValueError):
    """A source lock or the bytes it commits to are invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ArtifactScope(_StrictFrozenModel):
    """A deterministic set of files below the RAW root."""

    scope_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    mode: Literal["explicit_files", "recursive_tree"]
    paths: tuple[str, ...] = ()
    root: str | None = None
    exclude_prefixes: tuple[str, ...] = ()
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)
    manifest_sha256: Sha256

    @field_validator("paths", "exclude_prefixes", mode="before")
    @classmethod
    def coerce_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.mode == "explicit_files":
            if self.root is not None or self.exclude_prefixes:
                raise ValueError("explicit scope cannot define a root or exclusions")
            if not self.paths:
                raise ValueError("explicit scope requires paths")
        else:
            if self.root is None or self.paths:
                raise ValueError("tree scope requires only a root")
        for value in self.paths:
            _validate_relative_path(value, "artifact path")
        if self.root is not None:
            _validate_relative_path(self.root, "artifact root")
        for value in self.exclude_prefixes:
            _validate_relative_path(value, "excluded prefix")
        if self.paths != tuple(sorted(set(self.paths))):
            raise ValueError("artifact paths must be sorted and unique")
        if self.exclude_prefixes != tuple(sorted(set(self.exclude_prefixes))):
            raise ValueError("excluded prefixes must be sorted and unique")
        return self


class AcquisitionIdentity(_StrictFrozenModel):
    """Upstream identity retained separately from local content hashes."""

    logical_path: str
    url: str
    bytes: int = Field(ge=1)
    etag: str | None = None
    last_modified: str | None = None
    publisher_sha256: Sha256 | None = None
    local_sha256: Sha256

    @field_validator("logical_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _validate_relative_path(value, "acquisition path")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not re.fullmatch(r"https?://[^\s]+", value):
            raise ValueError("acquisition URL must be absolute HTTP(S)")
        if "@" in value.split("://", 1)[1].split("/", 1)[0]:
            raise ValueError("acquisition URL must not contain credentials")
        return value


class RequiredSourceLock(_StrictFrozenModel):
    """One immutable snapshot of a required source's locally acquired bytes."""

    schema_version: Literal[1] = 1
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_revision: str = Field(min_length=1)
    lock_plan_sha256: Sha256
    artifact_scopes: tuple[ArtifactScope, ...]
    acquisition_identities: tuple[AcquisitionIdentity, ...]

    @field_validator("artifact_scopes", "acquisition_identities", mode="before")
    @classmethod
    def coerce_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        if self.source_revision != self.source_revision.strip():
            raise ValueError("source revision must be canonical nonblank text")
        scope_ids = tuple(item.scope_id for item in self.artifact_scopes)
        if not scope_ids or scope_ids != tuple(sorted(set(scope_ids))):
            raise ValueError("artifact scopes must be sorted and unique")
        identities = tuple(
            item.logical_path for item in self.acquisition_identities
        )
        if identities != tuple(sorted(set(identities))):
            raise ValueError("acquisition identities must be sorted and unique")
        return self


@dataclass(frozen=True)
class VerifiedRequiredSourceLock:
    lock: RequiredSourceLock
    lock_file_sha256: str


def build_artifact_scope(
    raw_root: str | Path,
    *,
    scope_id: str,
    mode: Literal["explicit_files", "recursive_tree"],
    paths: tuple[str, ...] = (),
    root: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> ArtifactScope:
    """Hash a declared scope without loading large files into memory."""

    root_path = Path(raw_root).resolve(strict=True)
    normalized_paths = tuple(sorted(set(paths)))
    normalized_exclusions = tuple(sorted(set(exclude_prefixes)))
    if mode == "explicit_files":
        if root is not None or normalized_exclusions or not normalized_paths:
            raise SourceLockError("invalid explicit artifact scope")
        logical_paths = normalized_paths
    else:
        if root is None or normalized_paths:
            raise SourceLockError("invalid recursive artifact scope")
        _validate_relative_path(root, "artifact root")
        tree_root = _resolve_below(root_path, root)
        _reject_non_directory_or_link(tree_root, f"{scope_id} tree root")
        logical_paths = tuple(
            _walk_tree(
                root_path,
                tree_root,
                excluded_prefixes=normalized_exclusions,
            )
        )
        if not logical_paths:
            raise SourceLockError(f"{scope_id}: artifact tree is empty")

    digest = hashlib.sha256()
    total_bytes = 0
    casefolded: set[str] = set()
    for logical_path in logical_paths:
        _validate_relative_path(logical_path, "artifact path")
        folded = logical_path.casefold()
        if folded in casefolded:
            raise SourceLockError(
                f"{scope_id}: case-insensitive artifact path collision"
            )
        casefolded.add(folded)
        file_path = _resolve_below(root_path, logical_path)
        file_sha256, file_bytes = _stable_file_digest(
            file_path, label=f"{scope_id}:{logical_path}"
        )
        digest.update(
            canonical_json_bytes(
                {
                    "bytes": file_bytes,
                    "path": logical_path,
                    "sha256": file_sha256,
                }
            )
        )
        total_bytes += file_bytes

    return ArtifactScope(
        scope_id=scope_id,
        mode=mode,
        paths=normalized_paths,
        root=root,
        exclude_prefixes=normalized_exclusions,
        file_count=len(logical_paths),
        total_bytes=total_bytes,
        manifest_sha256=digest.hexdigest(),
    )


def stable_file_digest(path: str | Path, *, label: str) -> tuple[str, int]:
    """Return a mutation-checked SHA-256 and byte count for one large file."""

    return _stable_file_digest(Path(path), label=label)


def load_and_verify_required_source_lock(
    path: str | Path,
    raw_root: str | Path,
    *,
    expected_lock_file_sha256: str,
) -> VerifiedRequiredSourceLock:
    """Verify the externally supplied lock digest and every committed RAW byte."""

    lock = load_required_source_lock(
        path,
        expected_lock_file_sha256=expected_lock_file_sha256,
    )

    for expected in lock.artifact_scopes:
        observed = build_artifact_scope(
            raw_root,
            scope_id=expected.scope_id,
            mode=expected.mode,
            paths=expected.paths,
            root=expected.root,
            exclude_prefixes=expected.exclude_prefixes,
        )
        if observed != expected:
            raise SourceLockError(
                f"{lock.source_id}:{expected.scope_id}: RAW scope differs from lock"
            )
    identities = {item.logical_path: item for item in lock.acquisition_identities}
    for scope in lock.artifact_scopes:
        if scope.mode != "explicit_files":
            continue
        for logical_path in scope.paths:
            identity = identities.get(logical_path)
            if identity is None:
                continue
            file_sha256, file_bytes = _stable_file_digest(
                _resolve_below(Path(raw_root).resolve(strict=True), logical_path),
                label=f"acquisition:{logical_path}",
            )
            if identity.bytes != file_bytes or identity.local_sha256 != file_sha256:
                raise SourceLockError(
                    f"{lock.source_id}:{logical_path}: acquisition identity mismatch"
                )
    return VerifiedRequiredSourceLock(
        lock=lock,
        lock_file_sha256=expected_lock_file_sha256,
    )


def load_required_source_lock(
    path: str | Path,
    *,
    expected_lock_file_sha256: str,
) -> RequiredSourceLock:
    """Load canonical lock bytes under an external digest without walking RAW.

    Adapter-specific static compatibility checks should use this lightweight
    loader before invoking :func:`load_and_verify_required_source_lock`, whose
    full tree re-hash can be expensive for large image collections.
    """

    if re.fullmatch(r"[0-9a-f]{64}", expected_lock_file_sha256) is None:
        raise SourceLockError("expected lock SHA-256 must be lowercase hex")
    content = read_stable_regular_file(
        path,
        label="required source lock",
        max_bytes=_MAX_LOCK_BYTES,
    )
    if sha256_bytes(content) != expected_lock_file_sha256:
        raise SourceLockError("required source lock external digest mismatch")
    value = parse_canonical_json(content, label="required source lock")
    if not isinstance(value, dict):
        raise SourceLockError("required source lock root must be an object")
    try:
        lock = RequiredSourceLock.model_validate_json(content, strict=True)
    except ValueError as error:
        raise SourceLockError("required source lock schema is invalid") from error
    if content != canonical_json_bytes(lock.model_dump(mode="json")):
        raise SourceLockError("required source lock must be canonical JSON")
    return lock


def _walk_tree(
    raw_root: Path,
    tree_root: Path,
    *,
    excluded_prefixes: tuple[str, ...],
) -> list[str]:
    results: list[str] = []
    for current, directories, files in os.walk(tree_root, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in directories:
            candidate = current_path / name
            logical = candidate.relative_to(raw_root).as_posix()
            if _is_excluded(logical, excluded_prefixes):
                continue
            _reject_non_directory_or_link(candidate, f"artifact directory {logical}")
            safe_directories.append(name)
        directories[:] = sorted(safe_directories)
        for name in sorted(files):
            candidate = current_path / name
            logical = candidate.relative_to(raw_root).as_posix()
            if _is_excluded(logical, excluded_prefixes):
                continue
            before = candidate.lstat()
            if _is_link_or_junction(candidate, before) or not stat.S_ISREG(
                before.st_mode
            ):
                raise SourceLockError(
                    f"artifact {logical} must be a regular non-symlink file"
                )
            results.append(logical)
    return sorted(results)


def _stable_file_digest(path: Path, *, label: str) -> tuple[str, int]:
    try:
        before = path.lstat()
        if _is_link_or_junction(path, before) or not stat.S_ISREG(before.st_mode):
            raise SourceLockError(f"{label}: must be a regular non-symlink file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_snapshot(before, opened):
                raise SourceLockError(f"{label}: changed before hashing")
            while chunk := handle.read(_CHUNK_BYTES):
                digest.update(chunk)
            after_read = os.fstat(handle.fileno())
        after = path.lstat()
    except SourceLockError:
        raise
    except OSError as error:
        raise SourceLockError(f"{label}: cannot be hashed") from error
    if not _same_snapshot(opened, after_read) or not _same_snapshot(
        after_read, after
    ):
        raise SourceLockError(f"{label}: changed while hashing")
    return digest.hexdigest(), after.st_size


def _resolve_below(root: Path, logical_path: str) -> Path:
    _validate_relative_path(logical_path, "logical path")
    candidate = (root / Path(*PurePosixPath(logical_path).parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SourceLockError("logical path escapes RAW root") from error
    return candidate


def _validate_relative_path(value: str, label: str) -> None:
    if not value or "\\" in value or value != value.strip():
        raise ValueError(f"{label} must be canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must stay below RAW root")


def _reject_non_directory_or_link(path: Path, label: str) -> None:
    before = path.lstat()
    if _is_link_or_junction(path, before) or not stat.S_ISDIR(before.st_mode):
        raise SourceLockError(f"{label} must be a non-symlink directory")


def _is_link_or_junction(path: Path, snapshot: os.stat_result) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return stat.S_ISLNK(snapshot.st_mode) or (
        is_junction is not None and is_junction()
    )


def _is_excluded(logical_path: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        logical_path == prefix or logical_path.startswith(prefix + "/")
        for prefix in prefixes
    )


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    identity_matches = (
        left.st_dev == right.st_dev and left.st_ino == right.st_ino
        if left.st_ino and right.st_ino
        else True
    )
    return (
        identity_matches
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


__all__ = [
    "AcquisitionIdentity",
    "ArtifactScope",
    "RequiredSourceLock",
    "SourceLockError",
    "VerifiedRequiredSourceLock",
    "build_artifact_scope",
    "load_and_verify_required_source_lock",
    "stable_file_digest",
]
