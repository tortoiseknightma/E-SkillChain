"""Create-only, content-addressed manifests for local tool model artifacts.

The manifest deliberately contains no machine-specific absolute paths.  Every
referenced file is a regular file below the manifest directory, and consumers
re-hash those files at each formal boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

MODEL_ARTIFACT_SCHEMA_VERSION = 1

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ArtifactKind = Literal[
    "multimodal_embedding",
    "object_detector",
    "document_ocr",
]


class ModelArtifactError(ValueError):
    """Raised when a model artifact or its manifest is not trustworthy."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ArtifactDescriptor(_StrictFrozenModel):
    """Identity of one file relative to its model manifest."""

    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str
    bytes: int = Field(ge=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if not value or value != value.strip() or "\\" in value:
            raise ValueError("artifact path must be a canonical POSIX path")
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or value in {".", ".."} or ".." in parsed.parts:
            raise ValueError("artifact path must stay below the manifest directory")
        if parsed.as_posix() != value or any(
            part in {"", "."} for part in parsed.parts
        ):
            raise ValueError("artifact path must be a canonical POSIX path")
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
            or part.endswith((".", " "))
            for part in parsed.parts
        ):
            raise ValueError("artifact path contains an unsafe component")
        return value


class ModelArtifactManifest(_StrictFrozenModel):
    """Canonical identity for one detector or OCR runtime."""

    schema_version: Literal[1] = MODEL_ARTIFACT_SCHEMA_VERSION
    artifact_kind: ArtifactKind
    model_id: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    artifacts: tuple[ArtifactDescriptor, ...]
    manifest_sha256: Sha256

    @field_validator("model_id", "backend_name", "backend_version")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(
                "model identity fields must not have surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_artifacts_and_hash(self) -> Self:
        ordered = tuple(sorted(self.artifacts, key=lambda item: (item.role, item.path)))
        if not ordered or ordered != self.artifacts:
            raise ValueError("artifacts must be non-empty and canonically ordered")
        roles = [item.role for item in ordered]
        paths = [item.path for item in ordered]
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if len(paths) != len({path.casefold() for path in paths}):
            raise ValueError("artifact paths must be unique under case-folding")

        role_set = set(roles)
        if self.artifact_kind == "multimodal_embedding":
            required = {"weights", "config", "tokenizer"}
            if not required.issubset(role_set):
                raise ValueError(
                    "multimodal embedding requires weights, config, and tokenizer"
                )
        elif self.artifact_kind == "object_detector":
            required = {"weights", "labels", "config"}
            if not required.issubset(role_set):
                raise ValueError("object detector requires weights, labels, and config")
        else:
            required = {"labels", "config"}
            if not required.issubset(role_set) or not any(
                role == "model" or role.endswith("_model") for role in roles
            ):
                raise ValueError(
                    "document OCR requires model, labels, and config artifacts"
                )

        expected = manifest_digest(self)
        if self.manifest_sha256 != expected:
            raise ValueError("model artifact manifest self hash mismatch")
        return self


class ModelRuntimeBinding(_StrictFrozenModel):
    """Runtime identity included in every successful, including empty, result."""

    artifact_kind: ArtifactKind
    model_id: str
    backend_name: str
    backend_version: str
    manifest_sha256: Sha256
    artifacts: tuple[ArtifactDescriptor, ...]


@dataclass(frozen=True)
class FileSnapshot:
    """One opened regular-file snapshot used for TOCTOU verification."""

    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class VerifiedModelArtifact:
    """A canonical manifest whose referenced files were verified."""

    root: Path
    manifest_path: Path
    manifest: ModelArtifactManifest
    external_sha256_verified: bool

    @property
    def runtime_binding(self) -> ModelRuntimeBinding:
        return ModelRuntimeBinding(
            artifact_kind=self.manifest.artifact_kind,
            model_id=self.manifest.model_id,
            backend_name=self.manifest.backend_name,
            backend_version=self.manifest.backend_version,
            manifest_sha256=self.manifest.manifest_sha256,
            artifacts=self.manifest.artifacts,
        )

    def artifact_path(self, role: str) -> Path:
        descriptor = self.descriptor(role)
        return _descriptor_path(self.root, descriptor)

    def descriptor(self, role: str) -> ArtifactDescriptor:
        for descriptor in self.manifest.artifacts:
            if descriptor.role == role:
                return descriptor
        raise ModelArtifactError(f"model artifact role is not registered: {role}")

    def read_artifact_bytes(self, role: str, *, max_bytes: int | None = None) -> bytes:
        descriptor = self.descriptor(role)
        snapshot = read_regular_file_snapshot(
            self.artifact_path(role),
            f"model artifact {role}",
            max_bytes=max_bytes,
        )
        _verify_descriptor_snapshot(descriptor, snapshot)
        return snapshot.content

    def verify_files(self) -> None:
        for descriptor in self.manifest.artifacts:
            size, digest = _digest_regular_file(
                _descriptor_path(self.root, descriptor),
                f"model artifact {descriptor.role}",
            )
            if size != descriptor.bytes or digest != descriptor.sha256:
                raise ModelArtifactError(
                    f"model artifact {descriptor.role} does not match its manifest"
                )


def canonical_json_bytes(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Serialize one JSON object with the repository's canonical byte contract."""

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


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def manifest_digest(manifest: ModelArtifactManifest | Mapping[str, Any]) -> str:
    if isinstance(manifest, BaseModel):
        unsigned = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    else:
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def publish_model_artifact_manifest(
    manifest_path: str | Path,
    *,
    artifact_kind: ArtifactKind,
    model_id: str,
    backend_name: str,
    backend_version: str,
    artifacts: Mapping[str, str | Path],
) -> VerifiedModelArtifact:
    """Create a canonical manifest without overwriting an existing path."""

    manifest_path = Path(manifest_path).absolute()
    root = manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(root, "model artifact root")
    descriptors: list[ArtifactDescriptor] = []
    for role, raw_path in artifacts.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.absolute()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            raise ModelArtifactError(
                f"model artifact must stay below the manifest directory: {path}"
            ) from None
        _require_safe_relative_parents(root, PurePosixPath(relative).parts[:-1])
        size, digest = _digest_regular_file(path, f"model artifact {role}")
        descriptors.append(
            ArtifactDescriptor(role=role, path=relative, bytes=size, sha256=digest)
        )

    unsigned: dict[str, Any] = {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "model_id": model_id,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "artifacts": [
            item.model_dump(mode="json")
            for item in sorted(descriptors, key=lambda item: (item.role, item.path))
        ],
    }
    manifest = ModelArtifactManifest.model_validate_json(
        canonical_json_bytes({**unsigned, "manifest_sha256": manifest_digest(unsigned)})
    )
    manifest_bytes = canonical_json_bytes(manifest)
    _atomic_create_file(manifest_path, manifest_bytes)
    verified = load_model_artifact_manifest(
        manifest_path,
        expected_kind=artifact_kind,
        expected_manifest_sha256=manifest.manifest_sha256,
        verify_files=True,
    )
    # The publisher computed this digest itself.  It is useful output for the
    # caller to pin, but is not an external trust root until a later formal load
    # supplies that independently persisted value.
    return replace(verified, external_sha256_verified=False)


def load_model_artifact_manifest(
    manifest_path: str | Path,
    *,
    expected_kind: ArtifactKind | None = None,
    expected_manifest_sha256: str | None = None,
    verify_files: bool = True,
) -> VerifiedModelArtifact:
    """Load a manifest only when its identity matches an external trust root."""

    if expected_manifest_sha256 is None:
        raise ModelArtifactError(
            "formal model artifact load requires an external expected manifest SHA-256"
        )
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ModelArtifactError(
            "expected model manifest SHA-256 must be 64 lowercase hex characters"
        )

    manifest_path = Path(manifest_path).absolute()
    _require_regular_directory(manifest_path.parent, "model artifact root")
    snapshot = read_regular_file_snapshot(
        manifest_path, "model artifact manifest", max_bytes=2 * 1024 * 1024
    )
    try:
        raw = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(
            f"model artifact manifest is invalid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ModelArtifactError("model artifact manifest root must be an object")
    try:
        manifest = ModelArtifactManifest.model_validate_json(canonical_json_bytes(raw))
    except ValueError as exc:
        raise ModelArtifactError(
            f"model artifact manifest schema is invalid: {exc}"
        ) from exc
    if snapshot.content != canonical_json_bytes(manifest):
        raise ModelArtifactError("model artifact manifest must be canonical JSON")
    if expected_kind is not None and manifest.artifact_kind != expected_kind:
        raise ModelArtifactError(
            f"expected {expected_kind} artifact, got {manifest.artifact_kind}"
        )
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise ModelArtifactError(
            "model manifest SHA-256 does not match external expected lock"
        )

    verified = VerifiedModelArtifact(
        root=manifest_path.parent,
        manifest_path=manifest_path,
        manifest=manifest,
        external_sha256_verified=True,
    )
    if verify_files:
        verified.verify_files()
    return verified


def read_regular_file_snapshot(
    path: str | Path,
    label: str,
    *,
    max_bytes: int | None = None,
) -> FileSnapshot:
    """Read one regular, non-symlink file and retain its opened identity."""

    path = Path(path).absolute()
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise ModelArtifactError(f"{label} does not exist") from None
    except OSError as exc:
        raise ModelArtifactError(f"unable to inspect {label}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ModelArtifactError(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        raise ModelArtifactError(f"{label} must be a regular file")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ModelArtifactError(f"{label} exceeds the byte limit")

    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ModelArtifactError(f"{label} must remain a regular file")
            if _file_identity(before) != _file_identity(opened):
                raise ModelArtifactError(f"{label} changed before it could be read")
            content = source.read() if max_bytes is None else source.read(max_bytes + 1)
            after_open = os.fstat(source.fileno())
    except OSError as exc:
        raise ModelArtifactError(f"unable to read {label}") from exc
    if max_bytes is not None and len(content) > max_bytes:
        raise ModelArtifactError(f"{label} exceeds the byte limit")
    if _file_identity(opened) != _file_identity(after_open):
        raise ModelArtifactError(f"{label} changed while it was read")
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ModelArtifactError(f"{label} changed after it was read") from exc
    if _file_identity(after_open) != _file_identity(after_path):
        raise ModelArtifactError(f"{label} changed while it was read")
    return FileSnapshot(
        path=path,
        content=content,
        sha256=sha256_bytes(content),
        identity=_file_identity(after_open),
    )


def verify_file_snapshot(snapshot: FileSnapshot, label: str) -> None:
    """Fail if a path no longer contains the exact bytes and identity read earlier."""

    current = read_regular_file_snapshot(
        snapshot.path,
        label,
        max_bytes=len(snapshot.content),
    )
    if (
        current.identity != snapshot.identity
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise ModelArtifactError(f"{label} changed during tool execution")


def _digest_regular_file(path: Path, label: str) -> tuple[int, str]:
    path = path.absolute()
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise ModelArtifactError(f"{label} does not exist") from None
    except OSError as exc:
        raise ModelArtifactError(f"unable to inspect {label}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ModelArtifactError(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        raise ModelArtifactError(f"{label} must be a regular file")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise ModelArtifactError(f"{label} changed before it could be hashed")
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after_open = os.fstat(source.fileno())
    except OSError as exc:
        raise ModelArtifactError(f"unable to hash {label}") from exc
    if _file_identity(opened) != _file_identity(after_open):
        raise ModelArtifactError(f"{label} changed while it was hashed")
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ModelArtifactError(f"{label} changed after it was hashed") from exc
    if _file_identity(after_open) != _file_identity(after_path):
        raise ModelArtifactError(f"{label} changed while it was hashed")
    return after_open.st_size, digest.hexdigest()


def _verify_descriptor_snapshot(
    descriptor: ArtifactDescriptor, snapshot: FileSnapshot
) -> None:
    if (
        len(snapshot.content) != descriptor.bytes
        or snapshot.sha256 != descriptor.sha256
    ):
        raise ModelArtifactError(
            f"model artifact {descriptor.role} does not match its manifest"
        )


def _descriptor_path(root: Path, descriptor: ArtifactDescriptor) -> Path:
    parts = PurePosixPath(descriptor.path).parts
    _require_regular_directory(root, "model artifact root")
    _require_safe_relative_parents(root, parts[:-1])
    return root.joinpath(*parts)


def _require_safe_relative_parents(root: Path, parts: tuple[str, ...]) -> None:
    current = root
    for part in parts:
        current = current / part
        _require_regular_directory(current, "model artifact directory")


def _require_regular_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelArtifactError(f"unable to inspect {label}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ModelArtifactError(f"{label} must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ModelArtifactError(f"{label} must be a directory")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ModelArtifactError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


def _atomic_create_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"target already exists; refusing overwrite: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(
            f"target already exists; refusing overwrite: {path}"
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ArtifactDescriptor",
    "FileSnapshot",
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "ModelArtifactError",
    "ModelArtifactManifest",
    "ModelRuntimeBinding",
    "VerifiedModelArtifact",
    "canonical_json_bytes",
    "load_model_artifact_manifest",
    "manifest_digest",
    "publish_model_artifact_manifest",
    "read_regular_file_snapshot",
    "sha256_bytes",
    "verify_file_snapshot",
]
