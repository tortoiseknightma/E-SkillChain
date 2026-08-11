"""Strict canonical serialization and stable-file primitives for tool artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


class ArtifactFormatError(ValueError):
    """A persisted tool artifact is unsafe, non-canonical, or malformed."""


def validate_json_value(
    value: object,
    *,
    max_depth: int = 32,
    _depth: int = 0,
    _active: set[int] | None = None,
) -> None:
    """Reject non-JSON values, non-finite floats, cycles, and excessive depth."""

    if _depth > max_depth:
        raise ArtifactFormatError(f"JSON nesting exceeds {max_depth}")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactFormatError("JSON numbers must be finite")
        return
    if not isinstance(value, (list, dict)):
        raise ArtifactFormatError(f"unsupported JSON value: {type(value).__name__}")

    active = _active if _active is not None else set()
    identity = id(value)
    if identity in active:
        raise ArtifactFormatError("JSON containers must not contain cycles")
    active.add(identity)
    try:
        if isinstance(value, list):
            for item in value:
                validate_json_value(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _active=active,
                )
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ArtifactFormatError("JSON object keys must be strings")
                validate_json_value(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _active=active,
                )
    finally:
        active.remove(identity)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value with a trailing newline and no NaN extension."""

    validate_json_value(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(values: list[object] | tuple[object, ...]) -> bytes:
    """Serialize an ordered sequence as canonical JSONL."""

    return b"".join(canonical_json_bytes(value) for value in values)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_stable_regular_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    """Read one non-symlink regular file and reject mutation during the read."""

    path = Path(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise ArtifactFormatError(f"{label}: cannot be inspected") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ArtifactFormatError(f"{label}: must be a regular non-symlink file")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ArtifactFormatError(f"{label}: exceeds {max_bytes} bytes")

    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_file_snapshot(before, opened):
                raise ArtifactFormatError(f"{label}: changed before read")
            content = handle.read()
            after_read = os.fstat(handle.fileno())
        after = path.lstat()
    except ArtifactFormatError:
        raise
    except OSError as error:
        raise ArtifactFormatError(f"{label}: cannot be read") from error

    if not _same_file_snapshot(opened, after_read) or not _same_file_snapshot(
        after_read, after
    ):
        raise ArtifactFormatError(f"{label}: changed during read")
    if len(content) != after.st_size:
        raise ArtifactFormatError(f"{label}: byte count changed during read")
    return content


def parse_canonical_json(content: bytes, *, label: str) -> object:
    """Parse one canonical JSON document, rejecting duplicate object keys."""

    value = parse_strict_json(content, label=label)
    if content != canonical_json_bytes(value):
        raise ArtifactFormatError(f"{label}: must be canonical JSON")
    return value


def parse_strict_json(content: bytes, *, label: str) -> object:
    """Parse UTF-8 JSON safely without requiring provider bytes to be canonical.

    Provider-owned structured-output envelopes are not trusted to use the
    repository's whitespace or key-order convention.  They must still reject
    duplicate keys, non-finite numbers, invalid UTF-8, and unsafe JSON values
    before schema validation and runner-side canonicalization.
    """

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=lambda pairs: _object_without_duplicates(pairs, label),
            parse_constant=lambda token: _reject_constant(token, label),
        )
    except ArtifactFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactFormatError(f"{label}: invalid UTF-8 JSON") from error
    validate_json_value(value)
    return value


def parse_canonical_jsonl(content: bytes, *, label: str) -> tuple[object, ...]:
    """Parse canonical, non-blank JSONL without accepting a partial last line."""

    if not content:
        return ()
    if not content.endswith(b"\n"):
        raise ArtifactFormatError(f"{label}: must end with a newline")
    rows: list[object] = []
    for line_number, raw_line in enumerate(content.splitlines(keepends=True), start=1):
        if raw_line == b"\n":
            raise ArtifactFormatError(f"{label}: blank line {line_number}")
        rows.append(
            parse_canonical_json(
                raw_line,
                label=f"{label} line {line_number}",
            )
        )
    return tuple(rows)


def artifact_descriptor(
    path: str | Path, *, label: str | None = None
) -> dict[str, Any]:
    """Return the name, byte count, and digest of a stable regular artifact."""

    path = Path(path)
    content = read_stable_regular_file(path, label=label or path.name)
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def _object_without_duplicates(
    pairs: list[tuple[str, Any]], label: str
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactFormatError(f"{label}: duplicate key {key}")
        value[key] = item
    return value


def _reject_constant(token: str, label: str) -> None:
    raise ArtifactFormatError(f"{label}: non-finite number {token}")


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
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
    "ArtifactFormatError",
    "artifact_descriptor",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "parse_canonical_json",
    "parse_canonical_jsonl",
    "parse_strict_json",
    "read_stable_regular_file",
    "sha256_bytes",
    "validate_json_value",
]
