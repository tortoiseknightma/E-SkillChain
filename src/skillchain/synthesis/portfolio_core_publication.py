"""Create-only publication of the text-free r2 pre-generation bundle.

This module is deliberately the last metadata-only step before generation. It
serializes frozen planning, split, realism, audit, and S1 selection artifacts
without constructing author packets or corpus turns. A caller supplies a run
parent directory; publication uses a same-parent staging directory and an
atomic no-replace directory rename. Existing runs are accepted only when every
expected path and byte is identical.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.synthesis.planning import (
    R2CoreInMemoryPlan,
    audit_r2_core_in_memory_plan,
)
from skillchain.synthesis.portfolio_core_audit import (
    CoreR2AuditSample,
    canonical_audit_selection_index_bytes,
    validate_core_r2_audit_sample,
)
from skillchain.synthesis.portfolio_core_authoring import (
    ReuseAssignment,
    canonical_final_split_sidecar_bytes,
    canonical_prompt_recipe_bytes,
    canonical_realism_assignments_bytes,
    canonical_reuse_sidecar_bytes,
    core_r2_realism_quota_spec,
    validate_realism_sidecar,
)
from skillchain.synthesis.portfolio_core_r2 import R2CoreBridge
from skillchain.synthesis.portfolio_core_selection import (
    CoreR2CreatorSelection,
    canonical_creator_selection_index_bytes,
    validate_core_r2_creator_selection,
)
from skillchain.synthesis.splitting import verify_r2_split_constraint_binding
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PUBLICATION_VERSION = "portfolio-core-r2-pre-generation-v1"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-blank, trimmed identifier")
    if value in {".", ".."} or any(character in value for character in "/\\"):
        raise ValueError(f"{label} must not contain a path separator")
    if not all(character.isalnum() or character in "._-" for character in value):
        raise ValueError(f"{label} contains an unsafe character")
    return value


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("publication relative path must use non-empty POSIX syntax")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("publication relative path must not be absolute or traverse")
    return value


class CoreR2PublicationError(RuntimeError):
    """The pre-generation bundle cannot be safely published as requested."""

    def __init__(self, message: str, *, staging_path: Path | None = None) -> None:
        super().__init__(message)
        self.staging_path = staging_path


class PublicationConflictError(CoreR2PublicationError):
    """A populated target exists but is not byte-identical to this bundle."""


class PublicationSafetyError(CoreR2PublicationError):
    """A target, parent, or contained path is a link/reparse/special entry."""


class PublishedFile(_StrictModel):
    """Digest entry in the root manifest, deliberately excluding the manifest."""

    relative_path: str
    sha256: Sha256
    bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class PreGenerationManifest(_StrictModel):
    """Canonical root binding for every non-root pre-generation artifact."""

    schema_version: Literal[1] = 1
    publication_version: Literal["portfolio-core-r2-pre-generation-v1"] = (
        PUBLICATION_VERSION
    )
    run_id: str
    supersedes_run: str
    trusted_parent_r1_sha256: Sha256
    r2_plan_sha256: Sha256
    file_count: int = Field(gt=0)
    files: tuple[PublishedFile, ...]

    @field_validator("run_id", "supersedes_run")
    @classmethod
    def _validate_identifier(cls, value: str, info) -> str:
        return _safe_identifier(value, info.field_name)

    @field_validator("files")
    @classmethod
    def _validate_files(cls, value: tuple[PublishedFile, ...]) -> tuple[PublishedFile, ...]:
        paths = [item.relative_path for item in value]
        if not value or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("publication manifest files must be non-empty, sorted, unique")
        if "pre-generation-manifest.json" in paths:
            raise ValueError("root manifest must not self-reference")
        return value

    @model_validator(mode="after")
    def _validate_file_count(self):
        if self.file_count != len(self.files):
            raise ValueError("publication manifest file_count must match files")
        return self


@dataclass(frozen=True)
class CoreR2PublicationBundle:
    """All independently audited, text-free inputs required for publication."""

    r2_plan: R2CoreInMemoryPlan
    bridge: R2CoreBridge
    audit_sample: CoreR2AuditSample
    creator_selection: CoreR2CreatorSelection
    trusted_parent_r1_sha256: str
    supersedes_run: str
    run_id: str
    source_by_asset_id: Mapping[str, str] | None = None


@dataclass(frozen=True)
class _ArtifactBytes:
    relative_path: str
    content: bytes


@dataclass(frozen=True)
class CoreR2PublicationPayload:
    """Fully materialized bytes; building this value performs no filesystem I/O."""

    artifacts: tuple[_ArtifactBytes, ...]
    manifest: PreGenerationManifest


@dataclass(frozen=True)
class CoreR2PublicationResult:
    run_root: Path
    manifest: PreGenerationManifest
    created: bool
    incident_staging_path: Path | None = None


def _r2_plan_sha256(bundle: CoreR2PublicationBundle) -> str:
    return sha256_bytes(canonical_json_bytes(bundle.r2_plan.plan))


def _final_splits(bundle: CoreR2PublicationBundle) -> dict[str, str]:
    return {
        row.plan_id: bundle.r2_plan.final_split_by_plan_id[row.plan_id]
        for row in bundle.r2_plan.plan.queries
    }


def _reuse_assignments(bundle: CoreR2PublicationBundle) -> dict[str, ReuseAssignment]:
    return {
        row.plan_id: ReuseAssignment(
            plan_id=row.plan_id,
            reuse_variant=bundle.r2_plan.reuse_variant_by_plan_id[row.plan_id],
            reuse_reason=bundle.r2_plan.reuse_reason_by_plan_id[row.plan_id],
        )
        for row in bundle.r2_plan.plan.queries
    }


def _validate_bundle(bundle: CoreR2PublicationBundle) -> str:
    """Cross-check every typed input before its bytes become publishable."""

    _safe_identifier(bundle.run_id, "run_id")
    _safe_identifier(bundle.supersedes_run, "supersedes_run")
    if bundle.run_id == bundle.supersedes_run:
        raise CoreR2PublicationError("run_id must differ from supersedes_run")
    try:
        parent_sha = _require_sha256(
            bundle.trusted_parent_r1_sha256,
            "trusted_parent_r1_sha256",
        )
        plan_sha = _r2_plan_sha256(bundle)
        plan_audit = audit_r2_core_in_memory_plan(bundle.r2_plan)
    except (TypeError, ValueError, AssertionError) as exc:
        raise CoreR2PublicationError("r2 plan input failed its audited binding") from exc
    if plan_audit != bundle.r2_plan.audit:
        raise CoreR2PublicationError("stored r2 plan audit drifted from plan sidecars")
    if plan_sha == parent_sha:
        raise CoreR2PublicationError("r2 plan SHA must not equal its parent r1 SHA")

    constraints = bundle.bridge.split_constraints
    try:
        verify_r2_split_constraint_binding(
            constraints,
            expected_plan_sha256=plan_sha,
            expected_asset_catalog_sha256=bundle.r2_plan.plan.asset_catalog_sha256,
            expected_capability_assignments_sha256=(
                bundle.r2_plan.plan.capability_assignments_sha256
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CoreR2PublicationError("r2 split constraint binding drifted") from exc
    final_splits = _final_splits(bundle)
    if constraints.plan_id_to_split != final_splits:
        raise CoreR2PublicationError("r2 split constraints drifted from final split sidecar")

    reuse = _reuse_assignments(bundle)
    realism = bundle.bridge.realism_sidecar
    expected_realism_hashes = {
        "plan_sha256": plan_sha,
        "asset_catalog_sha256": bundle.r2_plan.plan.asset_catalog_sha256,
        "capability_assignments_sha256": (
            bundle.r2_plan.plan.capability_assignments_sha256
        ),
    }
    if any(
        getattr(realism.manifest, field_name) != expected
        for field_name, expected in expected_realism_hashes.items()
    ):
        raise CoreR2PublicationError("realism manifest drifted from r2 plan bindings")
    try:
        validate_realism_sidecar(
            realism,
            plan_rows=bundle.r2_plan.plan.queries,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=reuse,
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise CoreR2PublicationError("realism sidecar validation failed") from exc

    assignment_by_id = {item.plan_id: item for item in realism.assignments}
    expected_val_interactions = {
        row.plan_id: assignment_by_id[row.plan_id].interaction_pattern
        for row in bundle.r2_plan.plan.queries
        if final_splits[row.plan_id] == "val"
    }
    if bundle.bridge.val_interaction_by_query_id != expected_val_interactions:
        raise CoreR2PublicationError("val interaction sidecar drifted from realism")
    try:
        validate_core_r2_audit_sample(
            bundle.audit_sample,
            plan_rows=bundle.r2_plan.plan.queries,
            final_split_by_plan_id=final_splits,
            realism_sidecar=realism,
            source_by_asset_id=bundle.source_by_asset_id,
        )
        validate_core_r2_creator_selection(
            bundle.creator_selection,
            bundle.r2_plan,
            realism,
            trusted_plan_sha256=plan_sha,
        )
    except (TypeError, ValueError) as exc:
        raise CoreR2PublicationError("audit or stage1 selection validation failed") from exc
    return plan_sha


def _artifact(relative_path: str, content: bytes) -> _ArtifactBytes:
    return _ArtifactBytes(relative_path=_safe_relative_path(relative_path), content=content)


def build_core_r2_publication_payload(
    bundle: CoreR2PublicationBundle,
) -> CoreR2PublicationPayload:
    """Canonicalize a validated bundle without writing it anywhere."""

    plan_sha = _validate_bundle(bundle)
    final_splits = _final_splits(bundle)
    reuse = _reuse_assignments(bundle)
    realism = bundle.bridge.realism_sidecar
    artifacts = [
        _artifact("plan/core.json", canonical_json_bytes(bundle.r2_plan.plan)),
        _artifact(
            "sidecars/final-splits.jsonl",
            canonical_final_split_sidecar_bytes(final_splits),
        ),
        _artifact("sidecars/reuse.jsonl", canonical_reuse_sidecar_bytes(reuse)),
        _artifact(
            "sidecars/split-constraints.json",
            canonical_json_bytes(bundle.bridge.split_constraints),
        ),
        _artifact(
            "sidecars/val-interactions.jsonl",
            canonical_jsonl_bytes(
                {
                    "query_id": query_id,
                    "interaction_pattern": bundle.bridge.val_interaction_by_query_id[
                        query_id
                    ],
                }
                for query_id in sorted(bundle.bridge.val_interaction_by_query_id)
            ),
        ),
        _artifact(
            "authoring/realism.jsonl",
            canonical_realism_assignments_bytes(realism.assignments),
        ),
        _artifact(
            "authoring/realism-manifest.json",
            canonical_json_bytes(realism.manifest),
        ),
        _artifact(
            "authoring/prompt-recipes.jsonl",
            canonical_prompt_recipe_bytes(realism.recipes),
        ),
        _artifact(
            "authoring/prompt-recipes-manifest.json",
            canonical_json_bytes(realism.recipe_manifest),
        ),
        _artifact(
            "audit/index.jsonl",
            canonical_audit_selection_index_bytes(bundle.audit_sample.entries),
        ),
        _artifact(
            "audit/manifest.json",
            canonical_json_bytes(bundle.audit_sample.manifest),
        ),
        _artifact("audit/audit.json", canonical_json_bytes(bundle.audit_sample.audit)),
        _artifact(
            "stage1/index.jsonl",
            canonical_creator_selection_index_bytes(bundle.creator_selection.entries),
        ),
        _artifact(
            "stage1/manifest.json",
            canonical_json_bytes(bundle.creator_selection.manifest),
        ),
        _artifact(
            "stage1/audit.json",
            canonical_json_bytes(bundle.creator_selection.audit),
        ),
    ]
    artifacts = sorted(artifacts, key=lambda item: item.relative_path)
    if len({item.relative_path for item in artifacts}) != len(artifacts):
        raise AssertionError("publication artifact paths must be unique")
    manifest = PreGenerationManifest(
        run_id=bundle.run_id,
        supersedes_run=bundle.supersedes_run,
        trusted_parent_r1_sha256=bundle.trusted_parent_r1_sha256,
        r2_plan_sha256=plan_sha,
        file_count=len(artifacts),
        files=tuple(
            PublishedFile(
                relative_path=item.relative_path,
                sha256=sha256_bytes(item.content),
                bytes=len(item.content),
            )
            for item in artifacts
        ),
    )
    return CoreR2PublicationPayload(
        artifacts=tuple(
            [
                *artifacts,
                _artifact(
                    "pre-generation-manifest.json",
                    canonical_json_bytes(manifest),
                ),
            ]
        ),
        manifest=manifest,
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicationSafetyError(f"unable to inspect {label}: {path}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PublicationSafetyError(f"{label} must be a non-reparse real directory")


def _require_safe_target_path(target: Path) -> None:
    if not os.path.lexists(target):
        return
    metadata = target.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise PublicationSafetyError("publication target must not be a link or reparse")
    if not stat.S_ISDIR(metadata.st_mode):
        raise PublicationConflictError("publication target exists but is not a directory")


def _expected_directories(artifacts: tuple[_ArtifactBytes, ...]) -> set[str]:
    result: set[str] = set()
    for artifact in artifacts:
        path = PurePosixPath(artifact.relative_path)
        parent = path.parent
        while parent != PurePosixPath("."):
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _collect_tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    """Read a regular tree without following symlinks or Windows reparse points."""

    _require_real_directory(root, "publication run root")
    files: dict[str, Path] = {}
    directories: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            relative = entry.name if not prefix else f"{prefix}/{entry.name}"
            _safe_relative_path(relative)
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                raise PublicationSafetyError(
                    f"publication tree contains a link or reparse point: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(path, relative)
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = path
            else:
                raise PublicationSafetyError(
                    f"publication tree contains a non-regular entry: {relative}"
                )

    visit(root, "")
    return files, directories


def _assert_tree_is_exact(target: Path, payload: CoreR2PublicationPayload) -> None:
    _require_safe_target_path(target)
    expected_files = {item.relative_path: item.content for item in payload.artifacts}
    actual_files, actual_directories = _collect_tree(target)
    expected_directories = _expected_directories(payload.artifacts)
    if set(actual_files) != set(expected_files):
        raise PublicationConflictError(
            "existing publication has a different file collection"
        )
    if actual_directories != expected_directories:
        raise PublicationConflictError(
            "existing publication has a different directory collection"
        )
    for relative_path, expected in expected_files.items():
        actual = actual_files[relative_path].read_bytes()
        if len(actual) != len(expected) or sha256_bytes(actual) != sha256_bytes(expected):
            raise PublicationConflictError(
                f"existing publication content differs: {relative_path}"
            )


def _write_staging_tree(staging: Path, payload: CoreR2PublicationPayload) -> None:
    _require_real_directory(staging, "publication staging directory")
    for artifact in payload.artifacts:
        relative = PurePosixPath(artifact.relative_path)
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _require_real_directory(target.parent, "publication staging parent")
        try:
            with target.open("xb") as handle:
                handle.write(artifact.content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:  # pragma: no cover - hostile staging race
            raise PublicationSafetyError(
                f"staging already contains an artifact: {artifact.relative_path}",
                staging_path=staging,
            ) from exc
    _assert_tree_is_exact(staging, payload)


def publish_core_r2_pre_generation_bundle(
    bundle: CoreR2PublicationBundle,
    *,
    run_parent: str | Path,
) -> CoreR2PublicationResult:
    """Create-only publish a validated r2 bundle under ``run_parent / run_id``.

    The function never creates an active pointer. On a target collision it only
    succeeds if a safe, exact byte-for-byte run already exists. A failed staging
    directory is intentionally retained beside the requested run for incident
    inspection rather than being deleted.
    """

    payload = build_core_r2_publication_payload(bundle)
    parent = Path(run_parent)
    _require_real_directory(parent, "publication run parent")
    target = parent / bundle.run_id
    _require_safe_target_path(target)
    if os.path.lexists(target):
        _assert_tree_is_exact(target, payload)
        return CoreR2PublicationResult(
            run_root=target,
            manifest=payload.manifest,
            created=False,
        )

    try:
        staging = new_staging_directory(target)
    except FileExistsError:
        _require_safe_target_path(target)
        _assert_tree_is_exact(target, payload)
        return CoreR2PublicationResult(
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
            _assert_tree_is_exact(target, payload)
            return CoreR2PublicationResult(
                run_root=target,
                manifest=payload.manifest,
                created=False,
                incident_staging_path=staging,
            )
    except CoreR2PublicationError:
        raise
    except Exception as exc:
        raise CoreR2PublicationError(
            f"publication failed; staging retained at {staging}",
            staging_path=staging,
        ) from exc
    return CoreR2PublicationResult(
        run_root=target,
        manifest=payload.manifest,
        created=True,
    )
