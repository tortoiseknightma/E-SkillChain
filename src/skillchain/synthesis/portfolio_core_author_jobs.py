"""Create-only local publication for one verified r2 authoring job.

The publisher is intentionally a filesystem boundary only: it verifies a
fully-built :class:`AuthoringJob`, materialises its opaque asset aliases, and
never invokes a model or writes corpus text.  A future author may read only
``author-packet.json`` and the opaque ``author-assets`` names.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
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

from skillchain.synthesis.portfolio_core_authoring import (
    AuthoringJob,
    OpaqueAssetBinding,
    canonical_author_packet_bytes,
    canonical_work_order_bytes,
    scan_author_packet_for_leaks,
    validate_authoring_job,
    work_order_sha256,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    new_staging_directory,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AUTHOR_JOB_PUBLISH_POLICY_VERSION = "portfolio-core-r2-author-job-publish-v1"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_COPY_BUFFER_BYTES = 1024 * 1024
_METADATA_FILENAMES = (
    "work-order.json",
    "author-packet.json",
    "checkpoint.json",
    "manifest.json",
)


class AuthorJobPublishError(ValueError):
    """An authoring job or filesystem state is unsafe for create-only publication."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must be non-blank")
    return value


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


class PublishedAuthorAsset(_StrictModel):
    """One opaque file published under ``author-assets``."""

    opaque_alias: str
    relative_path: str
    asset_sha256: Sha256
    byte_size: int = Field(ge=0)

    @field_validator("opaque_alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        value = _nonblank(value, "opaque_alias")
        if not re.fullmatch(r"a[0-9]{4}\.[a-z0-9]{1,8}", value):
            raise ValueError("opaque_alias must be an opaque aNNNN.ext name")
        return value

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        value = _nonblank(value, "relative_path").replace("\\", "/")
        if value != f"author-assets/{Path(value).name}" or ".." in value.split("/"):
            raise ValueError("author asset path must be an opaque author-assets path")
        return value

    @model_validator(mode="after")
    def _validate_pair(self) -> Self:
        if self.relative_path != f"author-assets/{self.opaque_alias}":
            raise ValueError("author asset path must match its opaque alias")
        return self


class AuthorJobPublishManifest(_StrictModel):
    """Canonical receipt for a fully materialised, model-ready author job."""

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-r2-author-job-publish-v1"] = (
        AUTHOR_JOB_PUBLISH_POLICY_VERSION
    )
    job_id: str
    base_batch_id: str
    work_order_sha256: Sha256
    author_packet_sha256: Sha256
    checkpoint_sha256: Sha256
    asset_count: int = Field(ge=1)
    assets: tuple[PublishedAuthorAsset, ...]
    manifest_sha256: Sha256

    @field_validator("job_id", "base_batch_id")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("assets")
    @classmethod
    def _validate_assets(
        cls, value: tuple[PublishedAuthorAsset, ...]
    ) -> tuple[PublishedAuthorAsset, ...]:
        aliases = [asset.opaque_alias for asset in value]
        if not value or aliases != sorted(aliases) or len(aliases) != len(set(aliases)):
            raise ValueError("published author assets must be sorted and unique")
        return value

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.asset_count != len(self.assets):
            raise ValueError("published author asset_count does not match assets")
        if self.manifest_sha256 != _digest_without(
            self.model_dump(mode="json"), "manifest_sha256"
        ):
            raise ValueError("published author job manifest SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class PublishedAuthoringJob:
    """A published job path plus its exact in-memory manifest receipt."""

    job_dir: Path
    manifest: AuthorJobPublishManifest


def _is_reparse_point(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata)


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _assert_safe_path_chain(path: Path, *, label: str) -> os.stat_result:
    """Return final lstat metadata without traversing link/reparse components."""

    if not path.is_absolute():
        raise AuthorJobPublishError(f"{label} must be an absolute canonical path")
    anchor = Path(path.anchor)
    if not path.anchor:
        raise AuthorJobPublishError(f"{label} has no filesystem anchor")
    try:
        anchor_metadata = anchor.lstat()
    except OSError as exc:
        raise AuthorJobPublishError(f"{label} anchor cannot be inspected") from exc
    if _is_link_or_reparse(anchor_metadata) or not stat.S_ISDIR(
        anchor_metadata.st_mode
    ):
        raise AuthorJobPublishError(f"{label} anchor is a link or reparse point")
    current = anchor
    parts = path.parts
    for position, part in enumerate(parts[1:], start=1):
        if part in {".", ".."}:
            raise AuthorJobPublishError(f"{label} contains a traversal component")
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise AuthorJobPublishError(f"{label} cannot be inspected") from exc
        if _is_link_or_reparse(metadata):
            raise AuthorJobPublishError(
                f"{label} must not traverse a symlink or reparse point"
            )
        if position < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise AuthorJobPublishError(f"{label} has a non-directory ancestor")
    return anchor_metadata if path == anchor else metadata


def _assert_safe_real_directory(path: Path, *, label: str) -> None:
    metadata = _assert_safe_path_chain(path, label=label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise AuthorJobPublishError(f"{label} must be a real directory")


def _assert_safe_regular_file(path: Path, *, label: str) -> os.stat_result:
    metadata = _assert_safe_path_chain(path, label=label)
    if not stat.S_ISREG(metadata.st_mode):
        raise AuthorJobPublishError(f"{label} must be a regular file")
    return metadata


def _digest_regular_file(path: Path, *, label: str) -> tuple[int, str]:
    before = _assert_safe_regular_file(path, label=label)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_snapshot(before, opened):
                raise AuthorJobPublishError(f"{label} changed before reading")
            while block := handle.read(_COPY_BUFFER_BYTES):
                digest.update(block)
                byte_count += len(block)
            after_read = os.fstat(handle.fileno())
        after = _assert_safe_regular_file(path, label=label)
    except AuthorJobPublishError:
        raise
    except OSError as exc:
        raise AuthorJobPublishError(f"{label} cannot be read") from exc
    if not _same_snapshot(opened, after_read) or not _same_snapshot(after_read, after):
        raise AuthorJobPublishError(f"{label} changed during reading")
    if byte_count != after.st_size:
        raise AuthorJobPublishError(f"{label} byte count changed during reading")
    return byte_count, digest.hexdigest()


def _verify_asset_source(binding: OpaqueAssetBinding) -> Path:
    source = Path(binding.canonical_path)
    byte_count, digest = _digest_regular_file(
        source,
        label=f"source asset {binding.opaque_alias}",
    )
    if byte_count != binding.byte_size or digest != binding.asset_sha256:
        raise AuthorJobPublishError(
            f"source asset {binding.opaque_alias} size or SHA-256 drifted"
        )
    return source


def _verify_asset_file(path: Path, asset: PublishedAuthorAsset) -> None:
    byte_count, digest = _digest_regular_file(
        path,
        label=f"published asset {asset.opaque_alias}",
    )
    if byte_count != asset.byte_size or digest != asset.asset_sha256:
        raise AuthorJobPublishError(
            f"published asset {asset.opaque_alias} size or SHA-256 drifted"
        )


def _copy_source_to_staging(
    source: Path,
    destination: Path,
    asset: PublishedAuthorAsset,
) -> None:
    """Stream a verified source into a temporary staging file, then rename it."""

    _assert_safe_regular_file(source, label=f"source asset {asset.opaque_alias}")
    _assert_safe_real_directory(destination.parent, label="author-assets staging")
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=destination.parent,
            prefix=f".{asset.opaque_alias}.",
            suffix=".copy",
            delete=False,
        ) as target_handle:
            temporary_path = Path(target_handle.name)
            with source.open("rb") as source_handle:
                source_before = os.fstat(source_handle.fileno())
                while block := source_handle.read(_COPY_BUFFER_BYTES):
                    digest.update(block)
                    byte_count += len(block)
                    target_handle.write(block)
                source_after = os.fstat(source_handle.fileno())
            target_handle.flush()
            os.fsync(target_handle.fileno())
        _assert_safe_regular_file(source, label=f"source asset {asset.opaque_alias}")
        if not _same_snapshot(source_before, source_after):
            raise AuthorJobPublishError(
                f"source asset {asset.opaque_alias} changed during copy"
            )
        if byte_count != asset.byte_size or digest.hexdigest() != asset.asset_sha256:
            raise AuthorJobPublishError(
                f"source asset {asset.opaque_alias} size or SHA-256 drifted during copy"
            )
        _verify_asset_file(temporary_path, asset)
        os.replace(temporary_path, destination)
        temporary_path = None
    except AuthorJobPublishError:
        raise
    except OSError as exc:
        raise AuthorJobPublishError(
            f"source asset {asset.opaque_alias} cannot be copied"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _materialize_asset(
    binding: OpaqueAssetBinding,
    destination: Path,
    asset: PublishedAuthorAsset,
) -> None:
    source = _verify_asset_source(binding)
    try:
        os.link(source, destination)
    except OSError:
        _copy_source_to_staging(source, destination, asset)
    else:
        _assert_safe_regular_file(source, label=f"source asset {asset.opaque_alias}")
    _verify_asset_file(destination, asset)
    _verify_asset_source(binding)


def _canonical_checkpoint_bytes(job: AuthoringJob) -> bytes:
    return canonical_json_bytes(job.checkpoint)


def _published_assets(job: AuthoringJob) -> tuple[PublishedAuthorAsset, ...]:
    return tuple(
        PublishedAuthorAsset(
            opaque_alias=binding.opaque_alias,
            relative_path=binding.materialization_relative_path,
            asset_sha256=binding.asset_sha256,
            byte_size=binding.byte_size,
        )
        for binding in sorted(
            job.work_order.aliases, key=lambda item: item.opaque_alias
        )
    )


def _expected_manifest(job: AuthoringJob) -> AuthorJobPublishManifest:
    assets = _published_assets(job)
    payload: dict[str, object] = {
        "schema_version": 1,
        "policy_version": AUTHOR_JOB_PUBLISH_POLICY_VERSION,
        "job_id": job.work_order.job_id,
        "base_batch_id": job.work_order.base_batch_id,
        "work_order_sha256": work_order_sha256(job.work_order),
        "author_packet_sha256": sha256_bytes(
            canonical_author_packet_bytes(job.author_packet)
        ),
        "checkpoint_sha256": sha256_bytes(_canonical_checkpoint_bytes(job)),
        "asset_count": len(assets),
        "assets": tuple(asset.model_dump(mode="json") for asset in assets),
    }
    try:
        return AuthorJobPublishManifest.model_validate(
            {
                **payload,
                "manifest_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as exc:
        raise AuthorJobPublishError(
            "author job publication manifest is invalid"
        ) from exc


def _validate_publishable_job(job: AuthoringJob) -> None:
    if not isinstance(job, AuthoringJob):
        raise AuthorJobPublishError("author job must be an AuthoringJob")
    findings = scan_author_packet_for_leaks(job.author_packet)
    if findings:
        raise AuthorJobPublishError(
            "model-visible author packet leak scanner rejected job"
        )
    try:
        validate_authoring_job(job)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthorJobPublishError("author job validation failed") from exc
    aliases = job.work_order.aliases
    packet_aliases = tuple(item.opaque_alias for item in job.author_packet.items)
    expected_aliases = tuple(binding.opaque_alias for binding in aliases)
    if set(packet_aliases) != set(expected_aliases):
        raise AuthorJobPublishError(
            "author packet aliases do not match work-order aliases"
        )
    if len(expected_aliases) != len(set(expected_aliases)):
        raise AuthorJobPublishError("work-order opaque aliases must be unique")


def _job_target(jobs_root: Path, job_id: str) -> Path:
    if not re.fullmatch(r"r2-author-[0-9a-f]{24}", job_id):
        raise AuthorJobPublishError("author job ID is not safe for publication")
    target = jobs_root / job_id
    if target.parent != jobs_root:
        raise AuthorJobPublishError("author job target escaped jobs_root")
    return target


def _ensure_jobs_root(jobs_root: Path) -> None:
    if not jobs_root.is_absolute():
        raise AuthorJobPublishError("jobs_root must be an absolute local directory")
    if not os.path.lexists(jobs_root):
        parent = jobs_root.parent
        _assert_safe_real_directory(parent, label="jobs_root parent")
        try:
            jobs_root.mkdir()
        except OSError as exc:
            raise AuthorJobPublishError("jobs_root cannot be created") from exc
    _assert_safe_real_directory(jobs_root, label="jobs_root")


def _read_exact_file(path: Path, expected: bytes, *, label: str) -> None:
    before = _assert_safe_regular_file(path, label=label)
    if before.st_size != len(expected):
        raise AuthorJobPublishError(f"{label} bytes drifted")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_snapshot(before, opened):
                raise AuthorJobPublishError(f"{label} changed before reading")
            content = handle.read()
            after_read = os.fstat(handle.fileno())
        after = _assert_safe_regular_file(path, label=label)
    except AuthorJobPublishError:
        raise
    except OSError as exc:
        raise AuthorJobPublishError(f"{label} cannot be read") from exc
    if not _same_snapshot(opened, after_read) or not _same_snapshot(after_read, after):
        raise AuthorJobPublishError(f"{label} changed during reading")
    if content != expected:
        raise AuthorJobPublishError(f"{label} bytes drifted")


def _verify_published_target(
    target: Path,
    *,
    work_order_bytes: bytes,
    author_packet_bytes: bytes,
    checkpoint_bytes: bytes,
    manifest: AuthorJobPublishManifest,
) -> None:
    _assert_safe_real_directory(target, label="published author job directory")
    try:
        root_entries = {entry.name: entry for entry in target.iterdir()}
    except OSError as exc:
        raise AuthorJobPublishError(
            "published author job directory cannot be listed"
        ) from exc
    expected_root_names = set(_METADATA_FILENAMES) | {"author-assets"}
    if set(root_entries) != expected_root_names:
        raise AuthorJobPublishError(
            "published author job has unexpected or missing files"
        )
    _read_exact_file(
        root_entries["work-order.json"],
        work_order_bytes,
        label="published work-order.json",
    )
    _read_exact_file(
        root_entries["author-packet.json"],
        author_packet_bytes,
        label="published author-packet.json",
    )
    _read_exact_file(
        root_entries["checkpoint.json"],
        checkpoint_bytes,
        label="published checkpoint.json",
    )
    _read_exact_file(
        root_entries["manifest.json"],
        manifest.canonical_bytes(),
        label="published manifest.json",
    )
    assets_root = root_entries["author-assets"]
    _assert_safe_real_directory(assets_root, label="published author-assets directory")
    try:
        asset_entries = {entry.name: entry for entry in assets_root.iterdir()}
    except OSError as exc:
        raise AuthorJobPublishError("published author-assets cannot be listed") from exc
    expected_assets = {asset.opaque_alias: asset for asset in manifest.assets}
    if set(asset_entries) != set(expected_assets):
        raise AuthorJobPublishError(
            "published author-assets contain unexpected or missing files"
        )
    for alias, asset in expected_assets.items():
        _verify_asset_file(asset_entries[alias], asset)


def _cleanup_staging(staging: Path, *, jobs_root: Path, job_id: str) -> None:
    if not staging.exists() and not os.path.lexists(staging):
        return
    if staging.parent != jobs_root or not staging.name.startswith(
        f".{job_id}.staging-"
    ):
        raise AuthorJobPublishError("refusing to clean an unexpected staging directory")
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        raise AuthorJobPublishError("publisher staging cleanup failed") from exc


def publish_authoring_job(
    job: AuthoringJob,
    jobs_root: str | Path,
) -> PublishedAuthoringJob:
    """Publish one verified job atomically, or verify an identical prior publish.

    All source bytes are read solely from ``work_order.aliases`` canonical
    paths.  No existing target is modified; an existing job directory must be
    byte-for-byte and hash-for-hash equivalent to this immutable input.
    """

    _validate_publishable_job(job)
    root = Path(jobs_root)
    _ensure_jobs_root(root)
    target = _job_target(root, job.work_order.job_id)
    work_order_bytes = canonical_work_order_bytes(job.work_order)
    author_packet_bytes = canonical_author_packet_bytes(job.author_packet)
    checkpoint_bytes = _canonical_checkpoint_bytes(job)
    manifest = _expected_manifest(job)
    for binding in job.work_order.aliases:
        _verify_asset_source(binding)
    if os.path.lexists(target):
        _verify_published_target(
            target,
            work_order_bytes=work_order_bytes,
            author_packet_bytes=author_packet_bytes,
            checkpoint_bytes=checkpoint_bytes,
            manifest=manifest,
        )
        return PublishedAuthoringJob(job_dir=target, manifest=manifest)

    staging: Path | None = None
    published = False
    try:
        staging = new_staging_directory(target)
        _assert_safe_real_directory(staging, label="author job staging directory")
        assets_root = staging / "author-assets"
        assets_root.mkdir()
        _assert_safe_real_directory(assets_root, label="author-assets staging")
        aliases_by_name = {
            binding.opaque_alias: binding for binding in job.work_order.aliases
        }
        for asset in manifest.assets:
            binding = aliases_by_name.get(asset.opaque_alias)
            if binding is None:
                raise AuthorJobPublishError("manifest alias is missing from work order")
            destination = assets_root / asset.opaque_alias
            _materialize_asset(binding, destination, asset)
        atomic_create_file(staging / "work-order.json", work_order_bytes)
        atomic_create_file(staging / "author-packet.json", author_packet_bytes)
        atomic_create_file(staging / "checkpoint.json", checkpoint_bytes)
        atomic_create_file(staging / "manifest.json", manifest.canonical_bytes())
        _verify_published_target(
            staging,
            work_order_bytes=work_order_bytes,
            author_packet_bytes=author_packet_bytes,
            checkpoint_bytes=checkpoint_bytes,
            manifest=manifest,
        )
        try:
            atomic_publish_new_directory(staging, target)
        except FileExistsError:
            _verify_published_target(
                target,
                work_order_bytes=work_order_bytes,
                author_packet_bytes=author_packet_bytes,
                checkpoint_bytes=checkpoint_bytes,
                manifest=manifest,
            )
            return PublishedAuthoringJob(job_dir=target, manifest=manifest)
        published = True
        return PublishedAuthoringJob(job_dir=target, manifest=manifest)
    finally:
        if staging is not None and not published:
            _cleanup_staging(staging, jobs_root=root, job_id=job.work_order.job_id)


__all__ = [
    "AUTHOR_JOB_PUBLISH_POLICY_VERSION",
    "AuthorJobPublishError",
    "AuthorJobPublishManifest",
    "PublishedAuthorAsset",
    "PublishedAuthoringJob",
    "publish_authoring_job",
]
