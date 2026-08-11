"""Strict, auditable source catalog for the two local knowledge bases.

The catalog is deliberately independent from the BM25 implementation.  It
freezes source bytes, validates per-entry provenance, computes leakage groups,
and publishes one canonical create-only bundle that later indices can bind to.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from skillchain.synthesis.store import atomic_publish_new_directory

KBKind = Literal["encyclopedia", "recipe"]
KBOrigin = Literal["dump", "llm_synth"]
KBVerificationStatus = Literal["source_verified", "grounded_verified", "unverified"]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

KB_CATALOG_SCHEMA_VERSION = 2
KB_CATALOG_POLICY_VERSION = "kb-catalog-v2"
KB_LEAKAGE_POLICY_VERSION = "kb-evidence-components-v1"
_KINDS: tuple[KBKind, ...] = ("encyclopedia", "recipe")
_ENTRY_ARTIFACT = "entries.jsonl"
_GROUP_ARTIFACT = "groups.jsonl"


class KBCatalogError(ValueError):
    """A catalog or one of its source artifacts is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be blank")
    return cleaned


class KBEntryV2(_StrictFrozenModel):
    """A source-grounded KB document with stable provenance."""

    schema_version: Literal[2]
    entry_id: str
    title: str
    text: str
    kind: KBKind
    origin: KBOrigin
    source_dataset: str
    source_revision: str
    source_record_id: str
    source_uri: str
    license_id: str
    attribution: str | None = None
    content_sha256: Sha256
    entity_group_id: str
    near_duplicate_cluster_id: str | None = None
    derivation_parent_entry_ids: tuple[str, ...] = ()
    synth_provider: str | None = None
    synth_model: str | None = None
    synthesis_prompt_sha256: Sha256 | None = None
    verification_status: KBVerificationStatus

    @field_validator(
        "entry_id",
        "title",
        "source_dataset",
        "source_revision",
        "source_record_id",
        "source_uri",
        "license_id",
        "entity_group_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("text")
    @classmethod
    def validate_document_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator(
        "attribution",
        "near_duplicate_cluster_id",
        "synth_provider",
        "synth_model",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.scheme or any(character.isspace() for character in value):
            raise ValueError("source_uri must be an absolute URI")
        if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname is None:
            raise ValueError("HTTP source_uri must contain a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("source_uri must not contain credentials")
        return value

    @field_validator("derivation_parent_entry_ids", mode="before")
    @classmethod
    def coerce_json_parent_ids(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("derivation_parent_entry_ids")
    @classmethod
    def validate_parent_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(sorted(_nonblank(item, "parent entry id") for item in value))
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("derivation_parent_entry_ids must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> Self:
        expected_hash = sha256_bytes(self.text.encode("utf-8"))
        if self.content_sha256 != expected_hash:
            raise ValueError("content_sha256 does not match UTF-8 text bytes")
        if self.entry_id in self.derivation_parent_entry_ids:
            raise ValueError("entry cannot derive from itself")
        synthesis = (
            self.synth_provider,
            self.synth_model,
            self.synthesis_prompt_sha256,
        )
        if self.origin == "dump":
            if any(value is not None for value in synthesis):
                raise ValueError("dump entries must not carry synthesis identity")
        elif any(value is None for value in synthesis):
            raise ValueError(
                "llm_synth entries require provider, model, and prompt hash"
            )
        return self

    @property
    def formally_verified(self) -> bool:
        if self.origin == "dump":
            return self.verification_status == "source_verified"
        return self.verification_status == "grounded_verified" and bool(
            self.derivation_parent_entry_ids
        )


class KBLeakageGroupRecord(_StrictFrozenModel):
    schema_version: Literal[2]
    entry_id: str
    leakage_group_id: str

    @field_validator("entry_id", "leakage_group_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class ArtifactDescriptor(_StrictFrozenModel):
    path: str
    bytes: int = Field(ge=0)
    sha256: Sha256


class SourceDescriptor(_StrictFrozenModel):
    bytes: int = Field(ge=0)
    sha256: Sha256
    row_count: int = Field(ge=1)
    selected_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.selected_count > self.row_count:
            raise ValueError("selected_count cannot exceed row_count")
        return self


class KBCatalogManifest(_StrictFrozenModel):
    schema_version: Literal[2]
    policy_version: Literal["kb-catalog-v2"]
    leakage_policy_version: Literal["kb-evidence-components-v1"]
    mode: Literal["verified", "provisional"]
    complete: bool
    entry_count: int = Field(ge=2)
    kind_counts: dict[KBKind, int]
    group_count: int = Field(ge=1)
    group_binding_sha256: Sha256
    sources: dict[KBKind, SourceDescriptor]
    artifacts: dict[str, ArtifactDescriptor]
    catalog_sha256: Sha256

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if set(self.kind_counts) != set(_KINDS) or any(
            value < 1 for value in self.kind_counts.values()
        ):
            raise ValueError("kind_counts must contain both non-empty KB kinds")
        if sum(self.kind_counts.values()) != self.entry_count:
            raise ValueError("kind_counts do not sum to entry_count")
        if set(self.sources) != set(_KINDS):
            raise ValueError("sources must contain both KB kinds")
        for kind in _KINDS:
            if self.sources[kind].selected_count != self.kind_counts[kind]:
                raise ValueError(f"selected source count for {kind} is inconsistent")
            if (
                self.mode == "verified"
                and self.sources[kind].selected_count != self.sources[kind].row_count
            ):
                raise ValueError(
                    f"verified catalog does not cover the full {kind} source"
                )
        if set(self.artifacts) != {_ENTRY_ARTIFACT, _GROUP_ARTIFACT}:
            raise ValueError("catalog artifact set is invalid")
        for name, descriptor in self.artifacts.items():
            if descriptor.path != name:
                raise ValueError(f"artifact path for {name} is invalid")
        if self.mode == "verified" and not self.complete:
            raise ValueError("verified catalog must be complete")
        if self.mode == "provisional" and self.complete:
            raise ValueError("provisional catalog must not claim completeness")
        if self.catalog_sha256 != _catalog_digest(self):
            raise ValueError("catalog self-hash mismatch")
        return self


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class KBCatalog:
    root: Path
    manifest: KBCatalogManifest
    entries: tuple[KBEntryV2, ...]
    groups: tuple[KBLeakageGroupRecord, ...]
    manifest_bytes: bytes
    entries_bytes: bytes
    groups_bytes: bytes
    external_sha256_verified: bool

    @property
    def leakage_group_by_entry_id(self) -> dict[str, str]:
        return {record.entry_id: record.leakage_group_id for record in self.groups}


def build_kb_catalog(
    encyclopedia_source: Path | str,
    recipe_source: Path | str,
    output_dir: Path | str,
    *,
    expected_encyclopedia_sha256: str | None = None,
    expected_recipe_sha256: str | None = None,
    limit_per_kind: int | None = None,
    allow_provisional: bool = False,
) -> KBCatalog:
    """Create a canonical catalog without replacing an existing destination.

    A verified catalog has an external trust root: callers must supply the
    expected SHA-256 for both complete source JSONL files.  Provisional smoke
    builds may omit those locks, but any lock they do provide is still
    enforced.
    """

    if allow_provisional:
        if limit_per_kind is None or limit_per_kind <= 0:
            raise KBCatalogError(
                "provisional catalog is restricted to a positive limit_per_kind smoke build"
            )
    elif limit_per_kind is not None:
        raise KBCatalogError("limited catalogs require allow_provisional=True")

    expected_hashes: dict[KBKind, str | None] = {
        "encyclopedia": expected_encyclopedia_sha256,
        "recipe": expected_recipe_sha256,
    }
    if not allow_provisional and any(
        value is None for value in expected_hashes.values()
    ):
        raise KBCatalogError(
            "verified catalog requires expected SHA-256 locks for both sources"
        )
    for kind, expected_hash in expected_hashes.items():
        if expected_hash is not None and not _is_sha256(expected_hash):
            raise KBCatalogError(
                f"expected {kind} source SHA-256 must be 64 lowercase hex characters"
            )

    output_dir = Path(output_dir)
    if _lexists(output_dir):
        raise FileExistsError(f"catalog destination already exists: {output_dir}")

    source_paths: dict[KBKind, Path] = {
        "encyclopedia": Path(encyclopedia_source),
        "recipe": Path(recipe_source),
    }
    snapshots = {
        kind: read_regular_file_snapshot(path) for kind, path in source_paths.items()
    }
    for kind, snapshot in snapshots.items():
        expected_hash = expected_hashes[kind]
        if expected_hash is not None and snapshot.sha256 != expected_hash:
            raise KBCatalogError(f"{kind} source SHA-256 does not match expected lock")
    parsed = {
        kind: _parse_entry_source(snapshot.content, kind, snapshot.path)
        for kind, snapshot in snapshots.items()
    }
    selected = {
        kind: entries if limit_per_kind is None else entries[:limit_per_kind]
        for kind, entries in parsed.items()
    }
    if any(not entries for entries in selected.values()):
        raise KBCatalogError("each KB kind must contribute at least one entry")
    entries = tuple(entry for kind in _KINDS for entry in selected[kind])
    _validate_entry_set(entries, require_formal=not allow_provisional)
    groups = _compute_leakage_groups(entries)
    entries_bytes = canonical_jsonl_bytes(entries)
    groups_bytes = canonical_jsonl_bytes(groups)

    staging = _create_staging_directory(output_dir)
    try:
        (staging / _ENTRY_ARTIFACT).write_bytes(entries_bytes)
        (staging / _GROUP_ARTIFACT).write_bytes(groups_bytes)
        unsigned: dict[str, Any] = {
            "schema_version": KB_CATALOG_SCHEMA_VERSION,
            "policy_version": KB_CATALOG_POLICY_VERSION,
            "leakage_policy_version": KB_LEAKAGE_POLICY_VERSION,
            "mode": "provisional" if allow_provisional else "verified",
            "complete": not allow_provisional,
            "entry_count": len(entries),
            "kind_counts": {kind: len(selected[kind]) for kind in _KINDS},
            "group_count": len({record.leakage_group_id for record in groups}),
            "group_binding_sha256": sha256_bytes(groups_bytes),
            "sources": {
                kind: {
                    "bytes": len(snapshots[kind].content),
                    "sha256": snapshots[kind].sha256,
                    "row_count": len(parsed[kind]),
                    "selected_count": len(selected[kind]),
                }
                for kind in _KINDS
            },
            "artifacts": {
                name: artifact_descriptor(staging / name).model_dump(mode="json")
                for name in (_ENTRY_ARTIFACT, _GROUP_ARTIFACT)
            },
        }
        manifest_payload = {
            **unsigned,
            "catalog_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
        manifest = KBCatalogManifest.model_validate(manifest_payload)
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

        load_kb_catalog(
            staging,
            allow_provisional=True,
            expected_catalog_sha256=manifest.catalog_sha256,
        )
        for snapshot in snapshots.values():
            verify_file_snapshot(snapshot)
        _publish_create_only(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_kb_catalog(
        output_dir,
        allow_provisional=allow_provisional,
        expected_catalog_sha256=manifest.catalog_sha256,
    )


def load_kb_catalog(
    root: Path | str,
    *,
    allow_provisional: bool = False,
    expected_catalog_sha256: str | None = None,
) -> KBCatalog:
    """Load a catalog after verification against an external catalog lock."""

    if expected_catalog_sha256 is not None and not _is_sha256(expected_catalog_sha256):
        raise KBCatalogError(
            "expected catalog SHA-256 must be 64 lowercase hex characters"
        )

    root = Path(root)
    _require_regular_directory(root, "KB catalog")
    expected_names = {"manifest.json", _ENTRY_ARTIFACT, _GROUP_ARTIFACT}
    actual_names = {item.name for item in root.iterdir()}
    if actual_names != expected_names:
        raise KBCatalogError("catalog root artifact set is invalid")

    manifest_snapshot = read_regular_file_snapshot(root / "manifest.json")
    manifest_value = parse_json_object(manifest_snapshot.content, "manifest.json")
    try:
        manifest = KBCatalogManifest.model_validate(manifest_value)
    except Exception as error:
        raise KBCatalogError("manifest.json is invalid") from error
    expected_manifest = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    if manifest_snapshot.content != expected_manifest:
        raise KBCatalogError("manifest.json must be canonical JSON")
    if manifest.mode == "provisional" and not allow_provisional:
        raise KBCatalogError("provisional catalog requires explicit opt-in")
    if manifest.mode == "verified" and expected_catalog_sha256 is None:
        raise KBCatalogError(
            "verified catalog requires an external expected catalog SHA-256"
        )
    if (
        expected_catalog_sha256 is not None
        and manifest.catalog_sha256 != expected_catalog_sha256
    ):
        raise KBCatalogError("catalog SHA-256 does not match external expected lock")

    entry_snapshot = read_regular_file_snapshot(root / _ENTRY_ARTIFACT)
    group_snapshot = read_regular_file_snapshot(root / _GROUP_ARTIFACT)
    _verify_descriptor(entry_snapshot, manifest.artifacts[_ENTRY_ARTIFACT])
    _verify_descriptor(group_snapshot, manifest.artifacts[_GROUP_ARTIFACT])
    entries = _parse_canonical_jsonl(entry_snapshot.content, KBEntryV2, _ENTRY_ARTIFACT)
    groups = _parse_canonical_jsonl(
        group_snapshot.content, KBLeakageGroupRecord, _GROUP_ARTIFACT
    )
    typed_entries = tuple(entries)
    typed_groups = tuple(groups)
    _validate_entry_set(typed_entries, require_formal=manifest.mode == "verified")
    expected_groups = _compute_leakage_groups(typed_entries)
    if typed_groups != expected_groups:
        raise KBCatalogError("groups.jsonl does not match recomputed leakage closure")
    kind_counts = {
        kind: sum(entry.kind == kind for entry in typed_entries) for kind in _KINDS
    }
    if (
        len(typed_entries) != manifest.entry_count
        or kind_counts != manifest.kind_counts
    ):
        raise KBCatalogError("catalog entry counts do not match manifest")
    if sha256_bytes(group_snapshot.content) != manifest.group_binding_sha256:
        raise KBCatalogError("group binding hash mismatch")
    if (
        len({record.leakage_group_id for record in typed_groups})
        != manifest.group_count
    ):
        raise KBCatalogError("catalog group count does not match manifest")

    verify_file_snapshot(manifest_snapshot)
    verify_file_snapshot(entry_snapshot)
    verify_file_snapshot(group_snapshot)
    return KBCatalog(
        root=root,
        manifest=manifest,
        entries=typed_entries,
        groups=typed_groups,
        manifest_bytes=manifest_snapshot.content,
        entries_bytes=entry_snapshot.content,
        groups_bytes=group_snapshot.content,
        external_sha256_verified=expected_catalog_sha256 is not None,
    )


def verify_catalog_unchanged(catalog: KBCatalog) -> None:
    """Re-read every catalog byte used by an ongoing downstream build."""

    expected = {
        "manifest.json": catalog.manifest_bytes,
        _ENTRY_ARTIFACT: catalog.entries_bytes,
        _GROUP_ARTIFACT: catalog.groups_bytes,
    }
    for name, content in expected.items():
        snapshot = read_regular_file_snapshot(catalog.root / name)
        if snapshot.content != content:
            raise KBCatalogError(f"catalog artifact changed during build: {name}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_jsonl_bytes(values) -> bytes:
    chunks = []
    for value in values:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        chunks.append(canonical_json_bytes(value) + b"\n")
    return b"".join(chunks)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def artifact_descriptor(path: Path) -> ArtifactDescriptor:
    snapshot = read_regular_file_snapshot(path)
    return ArtifactDescriptor(
        path=path.name, bytes=len(snapshot.content), sha256=snapshot.sha256
    )


def read_regular_file_snapshot(path: Path | str) -> FileSnapshot:
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise KBCatalogError(f"missing artifact: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise KBCatalogError(f"artifact must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise KBCatalogError(f"artifact identity changed while opening: {path}")
            content = source.read()
            after = os.fstat(source.fileno())
    except OSError as error:
        raise KBCatalogError(f"artifact cannot be read: {path}") from error
    if _file_identity(opened) != _file_identity(after) or after.st_size != len(content):
        raise KBCatalogError(f"artifact changed while reading: {path}")
    return FileSnapshot(
        path=path,
        content=content,
        sha256=sha256_bytes(content),
        identity=_file_identity(after),
    )


def verify_file_snapshot(snapshot: FileSnapshot) -> None:
    current = read_regular_file_snapshot(snapshot.path)
    if current.identity != snapshot.identity or current.content != snapshot.content:
        raise KBCatalogError(f"artifact changed after snapshot: {snapshot.path}")


def parse_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: _raise_invalid_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KBCatalogError) as error:
        raise KBCatalogError(f"{label} is not valid strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KBCatalogError(f"{label} must contain a JSON object")
    return value


def _parse_entry_source(
    content: bytes, kind: KBKind, path: Path
) -> tuple[KBEntryV2, ...]:
    entries = _parse_jsonl(content, KBEntryV2, str(path))
    if not entries:
        raise KBCatalogError(f"source must not be empty: {path}")
    if any(entry.kind != kind for entry in entries):
        raise KBCatalogError(f"{path} contains an entry from the wrong KB kind")
    return tuple(entries)


def _parse_canonical_jsonl(content: bytes, model, label: str):
    values = _parse_jsonl(content, model, label)
    if content != canonical_jsonl_bytes(values):
        raise KBCatalogError(f"{label} must be canonical JSONL")
    return values


def _parse_jsonl(content: bytes, model, label: str):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KBCatalogError(f"{label} must be UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise KBCatalogError(f"{label} must contain non-empty JSONL records")
    values = []
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=lambda value: _raise_invalid_constant(value),
            )
            values.append(model.model_validate(raw))
        except Exception as error:
            raise KBCatalogError(f"{label}:{line_number} is invalid") from error
    return values


def _validate_entry_set(
    entries: tuple[KBEntryV2, ...], *, require_formal: bool
) -> None:
    entry_ids = [entry.entry_id for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise KBCatalogError("entry_id values must be globally unique")
    source_keys = [
        (entry.source_dataset, entry.source_revision, entry.source_record_id)
        for entry in entries
    ]
    if len(source_keys) != len(set(source_keys)):
        raise KBCatalogError("source record identities must be globally unique")
    entry_id_set = set(entry_ids)
    for entry in entries:
        missing = set(entry.derivation_parent_entry_ids) - entry_id_set
        if missing:
            raise KBCatalogError(
                f"entry {entry.entry_id} has missing derivation parents: {sorted(missing)}"
            )
        if require_formal and not entry.formally_verified:
            raise KBCatalogError(
                f"entry {entry.entry_id} is not eligible for a verified catalog"
            )
    _reject_derivation_cycles(entries)


def _reject_derivation_cycles(entries: tuple[KBEntryV2, ...]) -> None:
    parents = {entry.entry_id: entry.derivation_parent_entry_ids for entry in entries}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entry_id: str) -> None:
        if entry_id in visiting:
            raise KBCatalogError("derivation graph contains a cycle")
        if entry_id in visited:
            return
        visiting.add(entry_id)
        for parent_id in parents[entry_id]:
            visit(parent_id)
        visiting.remove(entry_id)
        visited.add(entry_id)

    for entry_id in sorted(parents):
        visit(entry_id)


def _compute_leakage_groups(
    entries: tuple[KBEntryV2, ...],
) -> tuple[KBLeakageGroupRecord, ...]:
    parent = {entry.entry_id: entry.entry_id for entry in entries}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    keyed: dict[tuple[str, ...], str] = {}
    for entry in entries:
        keys = [
            ("content", entry.content_sha256),
            ("entity", entry.entity_group_id),
            (
                "source",
                entry.source_dataset,
                entry.source_record_id,
            ),
        ]
        if entry.near_duplicate_cluster_id is not None:
            keys.append(
                (
                    "near",
                    entry.source_dataset,
                    entry.near_duplicate_cluster_id,
                )
            )
        for key in keys:
            previous = keyed.setdefault(key, entry.entry_id)
            union(previous, entry.entry_id)
        for parent_id in entry.derivation_parent_entry_ids:
            union(parent_id, entry.entry_id)

    members: dict[str, list[str]] = {}
    for entry_id in sorted(parent):
        members.setdefault(find(entry_id), []).append(entry_id)
    group_id_by_entry: dict[str, str] = {}
    for component in members.values():
        digest = sha256_bytes(canonical_json_bytes(sorted(component)))[:24]
        group_id = f"kbgrp-{digest}"
        group_id_by_entry.update({entry_id: group_id for entry_id in component})
    return tuple(
        KBLeakageGroupRecord(
            schema_version=KB_CATALOG_SCHEMA_VERSION,
            entry_id=entry_id,
            leakage_group_id=group_id_by_entry[entry_id],
        )
        for entry_id in sorted(parent)
    )


def _catalog_digest(manifest: KBCatalogManifest) -> str:
    unsigned = manifest.model_dump(mode="json", exclude={"catalog_sha256"})
    return sha256_bytes(canonical_json_bytes(unsigned))


def _verify_descriptor(snapshot: FileSnapshot, descriptor: ArtifactDescriptor) -> None:
    if snapshot.path.name != descriptor.path:
        raise KBCatalogError(f"artifact path mismatch: {descriptor.path}")
    if len(snapshot.content) != descriptor.bytes:
        raise KBCatalogError(f"artifact byte size mismatch: {descriptor.path}")
    if snapshot.sha256 != descriptor.sha256:
        raise KBCatalogError(f"artifact sha256 mismatch: {descriptor.path}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KBCatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str):
    raise KBCatalogError(f"non-finite JSON value: {value}")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size)


def _require_regular_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise KBCatalogError(f"{label} is missing: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise KBCatalogError(f"{label} must be a non-symlink directory")


def _create_staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )


def _publish_create_only(staging: Path, destination: Path) -> None:
    atomic_publish_new_directory(staging, destination)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = [
    "ArtifactDescriptor",
    "FileSnapshot",
    "KBCatalog",
    "KBCatalogError",
    "KBCatalogManifest",
    "KBEntryV2",
    "KBKind",
    "KBLeakageGroupRecord",
    "KB_CATALOG_POLICY_VERSION",
    "KB_CATALOG_SCHEMA_VERSION",
    "KB_LEAKAGE_POLICY_VERSION",
    "artifact_descriptor",
    "build_kb_catalog",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "load_kb_catalog",
    "parse_json_object",
    "read_regular_file_snapshot",
    "sha256_bytes",
    "verify_catalog_unchanged",
    "verify_file_snapshot",
]
