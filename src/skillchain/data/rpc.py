"""Create a small, deterministic RPC multi-product query-image bundle.

The Kaggle v5 archive contains two mirrored directory roots.  This adapter
accepts both only when their annotation document and every selected image are
byte-identical, then publishes one exact-byte copy per selected validation
scene.  It never treats the mirrored members as additional observations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Annotated, Any, Literal
import unicodedata
import zipfile

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.data.source_lock import (
    RequiredSourceLock,
    SourceLockError,
    load_required_source_lock,
    stable_file_digest,
)
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

SOURCE_DATASET = "rpc"
SOURCE_LOCK_PATH = (
    config.ROOT / "specs" / "data_sources" / "c2" / "source-locks" / "rpc.source-lock.json"
)
EXPECTED_SOURCE_LOCK_SHA256 = (
    "4dd0f78a8ffb84019973bbf45e9010827086c3d187314f805c37bff114a7edc2"
)
DEFAULT_RAW_ROOT = config.DATA_DIR / "raw"
DEFAULT_OUTPUT = config.DATA_DIR / "clean" / "rpc-multi-product-query-v1"
DEFAULT_COUNT = 35

ARCHIVE_SCOPE_ID = "kaggle_archive"
ANNOTATION_BASENAME = "instances_val2019.json"
VALIDATION_DIRECTORY = "val2019"
ROOT_PREFERENCE = ("retail_product_checkout", ".")
SELECTION_POLICY_VERSION = "rpc-val-multi-scene-sha256-order-v1"
TRANSFORM_POLICY_VERSION = "rpc-val-exact-zip-member-v1"
ADAPTER_ID = "rpc-val-multi-product-query-images-v1"
LICENSE_ID = "CC-BY-NC-SA-4.0"
ATTRIBUTION = "Retail Product Checkout (RPC) dataset; exact Kaggle v5 member bytes"

DATASET_ASSETS_FILE = "dataset-assets.jsonl"
SCENES_FILE = "scenes.jsonl"
MANIFEST_FILE = "manifest.json"
IMAGES_DIRECTORY = "images"

MAX_ANNOTATION_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250
MAX_METADATA_BYTES = 32 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class RPCAdapterError(ValueError):
    """The RPC source or derived adapter bundle is invalid."""


class RPCArchiveError(RPCAdapterError):
    """The locked RPC archive is unsafe, ambiguous, or inconsistent."""


class RPCBundleError(RPCAdapterError):
    """A published RPC bundle is malformed or has drifted."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RPCInstance(_StrictFrozenModel):
    annotation_id: int = Field(ge=0)
    category_id: int = Field(ge=0)
    category_name: str = Field(min_length=1)
    sku_product_id: str = Field(min_length=1)
    bbox_xywh: tuple[float, float, float, float]
    bbox_area: float = Field(gt=0)

    @field_validator("bbox_xywh", mode="before")
    @classmethod
    def coerce_bbox_array(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("category_name", "sku_product_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("RPC instance text must be canonical")
        return value

    @field_validator("bbox_xywh")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("RPC bbox values must be finite")
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("RPC bbox must have non-negative origin and positive size")
        return value


class RPCScene(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    selection_rank: int = Field(ge=1)
    selection_key_sha256: Sha256
    source_record_id: str = Field(min_length=1)
    image_id: int = Field(ge=0)
    image_file_name: str = Field(min_length=1)
    image_member_path: str = Field(min_length=1)
    replica_image_member_paths: tuple[str, ...] = Field(min_length=1)
    image_sha256: Sha256
    image_bytes: int = Field(gt=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_zip_crc32: int = Field(ge=0, le=0xFFFFFFFF)
    annotation_member_paths: tuple[str, ...] = Field(min_length=1)
    annotation_document_sha256: Sha256
    annotation_subset_sha256: Sha256
    instance_count: int = Field(ge=2)
    unique_sku_count: int = Field(ge=1)
    category_ids: tuple[int, ...] = Field(min_length=1)
    category_names: tuple[str, ...] = Field(min_length=1)
    category_instance_counts: dict[str, int]
    bbox_count: int = Field(ge=2)
    bbox_area_min: float = Field(gt=0)
    bbox_area_max: float = Field(gt=0)
    bbox_area_sum: float = Field(gt=0)
    dataset_asset_local_path: str = Field(min_length=1)
    instances: tuple[RPCInstance, ...] = Field(min_length=2)

    @field_validator(
        "replica_image_member_paths",
        "annotation_member_paths",
        "category_ids",
        "category_names",
        "instances",
        mode="before",
    )
    @classmethod
    def coerce_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_scene(self):
        if self.source_record_id != f"val2019:{self.image_id}":
            raise ValueError("RPC scene source_record_id is not canonical")
        if (
            tuple(sorted(set(self.replica_image_member_paths)))
            != self.replica_image_member_paths
        ):
            raise ValueError("RPC replica image member paths must be sorted and unique")
        if (
            tuple(sorted(set(self.annotation_member_paths)))
            != self.annotation_member_paths
        ):
            raise ValueError("RPC annotation member paths must be sorted and unique")
        if self.image_member_path not in self.replica_image_member_paths:
            raise ValueError("RPC canonical image member is absent from replicas")
        if self.instance_count != len(self.instances):
            raise ValueError("RPC instance count differs from instances")
        if self.bbox_count != self.instance_count:
            raise ValueError("RPC bbox count differs from instances")
        category_ids = tuple(sorted({item.category_id for item in self.instances}))
        category_names = tuple(sorted({item.category_name for item in self.instances}))
        if self.category_ids != category_ids or self.category_names != category_names:
            raise ValueError("RPC scene category summary differs from instances")
        if self.unique_sku_count != len(category_ids):
            raise ValueError("RPC unique SKU count differs from category IDs")
        counts = Counter(item.category_name for item in self.instances)
        if self.category_instance_counts != dict(sorted(counts.items())):
            raise ValueError("RPC category instance counts differ from instances")
        areas = [item.bbox_area for item in self.instances]
        if (
            self.bbox_area_min != min(areas)
            or self.bbox_area_max != max(areas)
            or self.bbox_area_sum != sum(areas)
        ):
            raise ValueError("RPC bbox statistics differ from instances")
        return self


class RPCAdapterFile(_StrictFrozenModel):
    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: Sha256
    rows: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "adapter file path")


class RPCAdapterManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    adapter_id: Literal["rpc-val-multi-product-query-images-v1"] = ADAPTER_ID
    status: Literal["portfolio_query_image_bundle"] = "portfolio_query_image_bundle"
    source_dataset: Literal["rpc"] = SOURCE_DATASET
    source_revision: str = Field(min_length=1)
    source_lock_sha256: Sha256
    archive_logical_path: str = Field(min_length=1)
    archive_sha256: Sha256
    archive_bytes: int = Field(gt=0)
    annotation_member_paths: tuple[str, ...] = Field(min_length=1)
    annotation_document_sha256: Sha256
    replica_roots: tuple[str, ...] = Field(min_length=1)
    canonical_root: str = Field(min_length=1)
    duplicate_root_policy: Literal[
        "accept_only_byte_identical_annotation_and_selected_images"
    ] = "accept_only_byte_identical_annotation_and_selected_images"
    selection_policy_version: Literal[
        "rpc-val-multi-scene-sha256-order-v1"
    ] = SELECTION_POLICY_VERSION
    requested_count: int = Field(gt=0)
    candidate_scene_count: int = Field(ge=0)
    selected_scene_count: int = Field(gt=0)
    selected_instance_count: int = Field(ge=2)
    selected_unique_category_count: int = Field(ge=1)
    selected_category_instance_counts: dict[str, int]
    bbox_count: int = Field(ge=2)
    bbox_area_min: float = Field(gt=0)
    bbox_area_max: float = Field(gt=0)
    bbox_area_sum: float = Field(gt=0)
    bundle_directory_name: str = Field(min_length=1)
    asset_root_relation: Literal["bundle_parent"] = "bundle_parent"
    dataset_asset_transform_policy_version: Literal[
        "rpc-val-exact-zip-member-v1"
    ] = TRANSFORM_POLICY_VERSION
    license_id: Literal["CC-BY-NC-SA-4.0"] = LICENSE_ID
    local_research_allowed: Literal[True] = True
    local_embedding_allowed: Literal[True] = True
    cloud_upload_allowed: Literal[True] = True
    public_demo_allowed: Literal[True] = True
    files: tuple[RPCAdapterFile, ...] = Field(min_length=3)
    manifest_self_sha256: Sha256

    @field_validator(
        "annotation_member_paths",
        "replica_roots",
        "files",
        mode="before",
    )
    @classmethod
    def coerce_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.source_revision != self.source_revision.strip():
            raise ValueError("RPC source revision must be canonical")
        if (
            tuple(sorted(set(self.annotation_member_paths)))
            != self.annotation_member_paths
        ):
            raise ValueError("RPC annotation paths must be sorted and unique")
        if tuple(sorted(set(self.replica_roots))) != self.replica_roots:
            raise ValueError("RPC roots must be sorted and unique")
        if self.canonical_root not in self.replica_roots:
            raise ValueError("RPC canonical root is absent from replica roots")
        file_paths = tuple(item.path for item in self.files)
        if file_paths != tuple(sorted(set(file_paths))):
            raise ValueError("RPC adapter files must be sorted and unique")
        if self.requested_count != self.selected_scene_count:
            raise ValueError("RPC adapter did not satisfy the requested count")
        unsigned = self.model_dump(mode="json", exclude={"manifest_self_sha256"})
        if self.manifest_self_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("RPC manifest self hash differs")
        return self


@dataclass(frozen=True)
class RPCAdapterBuildResult:
    output_dir: Path
    drafts: tuple[DatasetAssetDraft, ...]
    scenes: tuple[RPCScene, ...]
    manifest: RPCAdapterManifest
    manifest_file_sha256: str


@dataclass(frozen=True)
class VerifiedRPCAdapterBundle:
    root: Path
    drafts: tuple[DatasetAssetDraft, ...]
    scenes: tuple[RPCScene, ...]
    manifest: RPCAdapterManifest
    manifest_file_sha256: str


@dataclass(frozen=True)
class _ImageRow:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class _PreparedScene:
    scene: RPCScene
    draft: DatasetAssetDraft
    bundle_image_path: str
    image_bytes: bytes


def build_rpc_val_query_adapter(
    *,
    raw_root: str | Path,
    source_lock_path: str | Path,
    expected_source_lock_sha256: str,
    output_dir: str | Path,
    count: int = DEFAULT_COUNT,
) -> RPCAdapterBuildResult:
    """Publish a create-only RPC validation-scene bundle.

    The output's ``DatasetAssetDraft.local_path`` values are relative to the
    parent of ``output_dir``, allowing the unified catalog builder to use that
    parent as its asset root.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RPCAdapterError("RPC adapter count must be a positive integer")
    output_dir = Path(output_dir).absolute()
    bundle_name = _validate_bundle_name(output_dir.name)
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"RPC adapter target already exists; refusing overwrite: {output_dir}"
        )

    lock = load_required_source_lock(
        source_lock_path,
        expected_lock_file_sha256=expected_source_lock_sha256,
    )
    archive_path, archive_logical_path, archive_sha256, archive_bytes = (
        _resolve_locked_archive(lock, Path(raw_root))
    )
    snapshot = _file_snapshot(archive_path)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validate_archive_members(archive)
            (
                annotation_bytes,
                annotation_member_paths,
                replica_roots,
                canonical_root,
            ) = _load_annotation_replicas(archive, members)
            annotation_sha256 = sha256_bytes(annotation_bytes)
            annotation = parse_strict_json(
                annotation_bytes, label="RPC validation annotations"
            )
            images, categories, instances_by_image = _validate_annotations(annotation)
            candidates = _select_candidates(
                images=images,
                instances_by_image=instances_by_image,
                archive_sha256=archive_sha256,
            )
            if len(candidates) < count:
                raise RPCArchiveError(
                    "RPC validation annotations contain only "
                    f"{len(candidates)} multi-instance scenes; requested {count}"
                )
            prepared = tuple(
                _prepare_scene(
                    archive=archive,
                    members=members,
                    roots=replica_roots,
                    canonical_root=canonical_root,
                    annotation_member_paths=annotation_member_paths,
                    annotation_sha256=annotation_sha256,
                    image=images[image_id],
                    raw_instances=instances_by_image[image_id],
                    categories=categories,
                    selection_rank=rank,
                    selection_key=selection_key,
                    bundle_name=bundle_name,
                    source_revision=lock.source_revision,
                    source_url=_archive_identity(lock, archive_logical_path).url,
                )
                for rank, (selection_key, image_id) in enumerate(
                    candidates[:count], start=1
                )
            )
    except zipfile.BadZipFile as error:
        raise RPCArchiveError("RPC locked archive is not a valid ZIP") from error

    if _file_snapshot(archive_path) != snapshot:
        raise RPCArchiveError("RPC locked archive changed during adapter preparation")

    drafts = tuple(item.draft for item in prepared)
    scenes = tuple(item.scene for item in prepared)
    draft_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in drafts)
    )
    scene_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in scenes)
    )

    staging = new_staging_directory(output_dir)
    try:
        images_dir = staging / IMAGES_DIRECTORY
        images_dir.mkdir()
        payloads: dict[str, tuple[bytes, int | None]] = {
            DATASET_ASSETS_FILE: (draft_bytes, len(drafts)),
            SCENES_FILE: (scene_bytes, len(scenes)),
        }
        for item in prepared:
            target = staging / item.bundle_image_path
            target.write_bytes(item.image_bytes)
            payloads[item.bundle_image_path] = (item.image_bytes, None)

        files = tuple(
            RPCAdapterFile(
                path=path,
                bytes=len(content),
                sha256=sha256_bytes(content),
                rows=rows,
            )
            for path, (content, rows) in sorted(payloads.items())
        )
        category_counts = Counter(
            instance.category_name for scene in scenes for instance in scene.instances
        )
        bbox_areas = [
            instance.bbox_area for scene in scenes for instance in scene.instances
        ]
        unsigned_manifest = {
            "schema_version": 1,
            "adapter_id": ADAPTER_ID,
            "status": "portfolio_query_image_bundle",
            "source_dataset": SOURCE_DATASET,
            "source_revision": lock.source_revision,
            "source_lock_sha256": expected_source_lock_sha256,
            "archive_logical_path": archive_logical_path,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
            "annotation_member_paths": annotation_member_paths,
            "annotation_document_sha256": annotation_sha256,
            "replica_roots": replica_roots,
            "canonical_root": canonical_root,
            "duplicate_root_policy": (
                "accept_only_byte_identical_annotation_and_selected_images"
            ),
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "requested_count": count,
            "candidate_scene_count": len(candidates),
            "selected_scene_count": len(scenes),
            "selected_instance_count": len(bbox_areas),
            "selected_unique_category_count": len(
                {
                    instance.category_id
                    for scene in scenes
                    for instance in scene.instances
                }
            ),
            "selected_category_instance_counts": dict(sorted(category_counts.items())),
            "bbox_count": len(bbox_areas),
            "bbox_area_min": min(bbox_areas),
            "bbox_area_max": max(bbox_areas),
            "bbox_area_sum": sum(bbox_areas),
            "bundle_directory_name": bundle_name,
            "asset_root_relation": "bundle_parent",
            "dataset_asset_transform_policy_version": TRANSFORM_POLICY_VERSION,
            "license_id": LICENSE_ID,
            "local_research_allowed": True,
            "local_embedding_allowed": True,
            "cloud_upload_allowed": True,
            "public_demo_allowed": True,
            "files": tuple(item.model_dump(mode="json") for item in files),
        }
        manifest = RPCAdapterManifest(
            **unsigned_manifest,
            manifest_self_sha256=sha256_bytes(
                canonical_json_bytes(_json_native(unsigned_manifest))
            ),
        )
        manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
        (staging / DATASET_ASSETS_FILE).write_bytes(draft_bytes)
        (staging / SCENES_FILE).write_bytes(scene_bytes)
        (staging / MANIFEST_FILE).write_bytes(manifest_bytes)
        manifest_file_sha256 = sha256_bytes(manifest_bytes)
        load_verified_rpc_val_adapter(
            staging,
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_bundle_directory_name=bundle_name,
        )
        if _file_snapshot(archive_path) != snapshot:
            raise RPCArchiveError("RPC locked archive changed before publication")
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return RPCAdapterBuildResult(
        output_dir=output_dir,
        drafts=drafts,
        scenes=scenes,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
    )


def load_verified_rpc_val_adapter(
    root: str | Path,
    *,
    expected_manifest_file_sha256: str,
    expected_bundle_directory_name: str | None = None,
) -> VerifiedRPCAdapterBundle:
    """Strictly reload a published RPC adapter and all exact image bytes."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_file_sha256) is None:
        raise RPCBundleError("expected RPC manifest SHA-256 must be lowercase hex")
    root = Path(root).absolute()
    if not root.is_dir() or root.is_symlink():
        raise RPCBundleError("RPC adapter root must be a real directory")
    manifest_bytes = read_stable_regular_file(
        root / MANIFEST_FILE,
        label="RPC adapter manifest",
        max_bytes=MAX_METADATA_BYTES,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise RPCBundleError("RPC adapter manifest external digest mismatch")
    try:
        manifest_value = parse_canonical_json(
            manifest_bytes, label="RPC adapter manifest"
        )
        manifest = RPCAdapterManifest.model_validate(manifest_value, strict=True)
    except (ArtifactFormatError, ValueError) as error:
        raise RPCBundleError("RPC adapter manifest is invalid") from error
    expected_bundle_name = (
        root.name
        if expected_bundle_directory_name is None
        else _validate_bundle_name(expected_bundle_directory_name)
    )
    if manifest.bundle_directory_name != expected_bundle_name:
        raise RPCBundleError("RPC adapter bundle directory name changed")

    expected_paths = {MANIFEST_FILE, *(item.path for item in manifest.files)}
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RPCBundleError(f"RPC adapter contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RPCBundleError(
                f"RPC adapter contains a non-regular file: {relative}"
            )
        actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise RPCBundleError("RPC adapter file set differs from manifest")

    payloads: dict[str, bytes] = {}
    for descriptor in manifest.files:
        maximum = (
            MAX_METADATA_BYTES
            if descriptor.path in {DATASET_ASSETS_FILE, SCENES_FILE}
            else MAX_IMAGE_BYTES
        )
        content = read_stable_regular_file(
            root / Path(*PurePosixPath(descriptor.path).parts),
            label=f"RPC adapter payload {descriptor.path}",
            max_bytes=maximum,
        )
        if (
            len(content) != descriptor.bytes
            or sha256_bytes(content) != descriptor.sha256
        ):
            raise RPCBundleError(f"RPC adapter payload drifted: {descriptor.path}")
        payloads[descriptor.path] = content

    drafts = _parse_canonical_rows(
        payloads[DATASET_ASSETS_FILE],
        DatasetAssetDraft,
        "RPC dataset asset drafts",
    )
    scenes = _parse_canonical_rows(
        payloads[SCENES_FILE],
        RPCScene,
        "RPC scenes",
    )
    file_by_path = {item.path: item for item in manifest.files}
    if (
        file_by_path[DATASET_ASSETS_FILE].rows != len(drafts)
        or file_by_path[SCENES_FILE].rows != len(scenes)
        or len(drafts) != manifest.selected_scene_count
        or len(scenes) != manifest.selected_scene_count
    ):
        raise RPCBundleError("RPC adapter row counts differ from manifest")

    draft_by_record = {item.source_record_id: item for item in drafts}
    if len(draft_by_record) != len(drafts):
        raise RPCBundleError("RPC dataset asset source_record_id is duplicated")
    for scene in scenes:
        draft = draft_by_record.get(scene.source_record_id)
        expected_local_path = (
            f"{manifest.bundle_directory_name}/"
            f"{_bundle_path_from_local_path(scene.dataset_asset_local_path, manifest)}"
        )
        if (
            draft is None
            or draft.source_dataset != SOURCE_DATASET
            or draft.source_revision != manifest.source_revision
            or draft.local_path != scene.dataset_asset_local_path
            or draft.local_path != expected_local_path
            or draft.product_id is not None
            or draft.transform_policy_version != TRANSFORM_POLICY_VERSION
            or draft.license_id != LICENSE_ID
            or draft.cloud_upload_allowed is not True
            or draft.public_demo_allowed is not True
            or draft.derivation_parent_asset_ids
            or draft.derivation_parent_asset_id is not None
        ):
            raise RPCBundleError("RPC scene and DatasetAssetDraft binding differs")
        bundle_path = _bundle_path_from_local_path(draft.local_path, manifest)
        image_bytes = payloads.get(bundle_path)
        if (
            image_bytes is None
            or sha256_bytes(image_bytes) != scene.image_sha256
            or len(image_bytes) != scene.image_bytes
        ):
            raise RPCBundleError("RPC scene image identity differs from payload")
        expected_subset_sha = sha256_bytes(
            canonical_json_bytes(
                [item.model_dump(mode="json") for item in scene.instances]
            )
        )
        if expected_subset_sha != scene.annotation_subset_sha256:
            raise RPCBundleError("RPC scene annotation subset identity differs")

    category_counts = Counter(
        instance.category_name for scene in scenes for instance in scene.instances
    )
    bbox_areas = [
        instance.bbox_area for scene in scenes for instance in scene.instances
    ]
    if (
        manifest.selected_instance_count != len(bbox_areas)
        or manifest.bbox_count != len(bbox_areas)
        or manifest.selected_category_instance_counts
        != dict(sorted(category_counts.items()))
        or manifest.bbox_area_min != min(bbox_areas)
        or manifest.bbox_area_max != max(bbox_areas)
        or manifest.bbox_area_sum != sum(bbox_areas)
    ):
        raise RPCBundleError("RPC manifest aggregate statistics differ")

    return VerifiedRPCAdapterBundle(
        root=root,
        drafts=drafts,
        scenes=scenes,
        manifest=manifest,
        manifest_file_sha256=expected_manifest_file_sha256,
    )


def _resolve_locked_archive(
    lock: RequiredSourceLock, raw_root: Path
) -> tuple[Path, str, str, int]:
    if lock.source_id != SOURCE_DATASET:
        raise RPCAdapterError("source lock does not target RPC")
    scopes = [item for item in lock.artifact_scopes if item.scope_id == ARCHIVE_SCOPE_ID]
    if len(scopes) != 1:
        raise RPCAdapterError("RPC source lock must contain one kaggle_archive scope")
    scope = scopes[0]
    if scope.mode != "explicit_files" or len(scope.paths) != 1:
        raise RPCAdapterError("RPC kaggle_archive scope must lock one explicit file")
    logical_path = scope.paths[0]
    identity = _archive_identity(lock, logical_path)
    raw_root = raw_root.resolve(strict=True)
    archive_path = (raw_root / Path(*PurePosixPath(logical_path).parts)).resolve(
        strict=True
    )
    try:
        archive_path.relative_to(raw_root)
    except ValueError:
        raise RPCAdapterError("RPC archive resolves outside the RAW root") from None
    archive_sha256, archive_bytes = stable_file_digest(
        archive_path, label="RPC Kaggle v5 archive"
    )
    if (
        archive_sha256 != identity.local_sha256
        or archive_bytes != identity.bytes
    ):
        raise RPCArchiveError("RPC archive SHA-256 or byte count differs from lock")
    scope_digest = sha256_bytes(
        canonical_json_bytes(
            {
                "bytes": archive_bytes,
                "path": logical_path,
                "sha256": archive_sha256,
            }
        )
    )
    if (
        scope.file_count != 1
        or scope.total_bytes != archive_bytes
        or scope.manifest_sha256 != scope_digest
    ):
        raise RPCArchiveError("RPC archive descriptor differs from locked scope")
    return archive_path, logical_path, archive_sha256, archive_bytes


def _archive_identity(lock: RequiredSourceLock, logical_path: str):
    identities = [
        item for item in lock.acquisition_identities if item.logical_path == logical_path
    ]
    if len(identities) != 1:
        raise RPCAdapterError("RPC archive must have one acquisition identity")
    return identities[0]


def _validate_archive_members(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    casefolded: dict[str, str] = {}
    for info in archive.infolist():
        name = _validate_zip_member_path(info.filename, info.is_dir())
        if name in members:
            raise RPCArchiveError(f"duplicate RPC ZIP member: {name}")
        folded = unicodedata.normalize("NFC", name).casefold()
        previous = casefolded.get(folded)
        if previous is not None and previous != name:
            raise RPCArchiveError(
                f"case-insensitive RPC ZIP member collision: {previous} / {name}"
            )
        casefolded[folded] = name
        members[name] = info
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise RPCArchiveError(f"RPC ZIP contains a symlink member: {name}")
        if info.flag_bits & 0x1:
            raise RPCArchiveError(f"RPC ZIP contains an encrypted member: {name}")
    return members


def _validate_zip_member_path(name: str, is_directory: bool) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise RPCArchiveError(f"unsafe RPC ZIP member path: {name!r}")
    normalized = unicodedata.normalize("NFC", name)
    if normalized != name:
        raise RPCArchiveError(f"non-canonical RPC ZIP member path: {name!r}")
    body = name[:-1] if is_directory and name.endswith("/") else name
    path = PurePosixPath(body)
    if (
        not body
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != body
        or re.match(r"^[A-Za-z]:", body)
    ):
        raise RPCArchiveError(f"unsafe RPC ZIP member path: {name!r}")
    return name


def _load_annotation_replicas(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
) -> tuple[bytes, tuple[str, ...], tuple[str, ...], str]:
    available: list[tuple[str, str, bytes]] = []
    for root in ROOT_PREFERENCE:
        member = _member_path(root, ANNOTATION_BASENAME)
        info = members.get(member)
        if info is None:
            continue
        available.append(
            (
                root,
                member,
                _read_bounded_member(
                    archive,
                    info,
                    maximum_bytes=MAX_ANNOTATION_BYTES,
                    label="RPC validation annotation",
                ),
            )
        )
    if not available:
        raise RPCArchiveError("RPC validation annotation member is missing")
    digests = {sha256_bytes(content) for _, _, content in available}
    if len(digests) != 1:
        raise RPCArchiveError("ambiguous RPC duplicate roots have different annotations")
    canonical_root, _, annotation_bytes = available[0]
    roots = tuple(sorted(root for root, _, _ in available))
    annotation_paths = tuple(sorted(member for _, member, _ in available))
    return annotation_bytes, annotation_paths, roots, canonical_root


def _validate_annotations(
    value: object,
) -> tuple[
    dict[int, _ImageRow],
    dict[int, str],
    dict[int, tuple[dict[str, Any], ...]],
]:
    if not isinstance(value, dict):
        raise RPCArchiveError("RPC annotation root must be an object")
    for key in ("images", "categories", "annotations"):
        if not isinstance(value.get(key), list):
            raise RPCArchiveError(f"RPC annotation field {key} must be an array")

    categories: dict[int, str] = {}
    category_names: set[str] = set()
    for raw in value["categories"]:
        if not isinstance(raw, dict):
            raise RPCArchiveError("RPC category rows must be objects")
        category_id = _strict_nonnegative_int(raw.get("id"), "category id")
        name = _strict_nonblank_text(raw.get("name"), "category name")
        if category_id in categories or name in category_names:
            raise RPCArchiveError("RPC category ID or name is duplicated")
        categories[category_id] = name
        category_names.add(name)
    if not categories:
        raise RPCArchiveError("RPC categories are empty")

    images: dict[int, _ImageRow] = {}
    image_names: set[str] = set()
    for raw in value["images"]:
        if not isinstance(raw, dict):
            raise RPCArchiveError("RPC image rows must be objects")
        image_id = _strict_nonnegative_int(raw.get("id"), "image id")
        file_name = _validate_image_file_name(raw.get("file_name"))
        width = _strict_positive_int(raw.get("width"), "image width")
        height = _strict_positive_int(raw.get("height"), "image height")
        if image_id in images or file_name in image_names:
            raise RPCArchiveError("RPC image ID or file_name is duplicated")
        images[image_id] = _ImageRow(image_id, file_name, width, height)
        image_names.add(file_name)
    if not images:
        raise RPCArchiveError("RPC validation images are empty")

    annotation_ids: set[int] = set()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in value["annotations"]:
        if not isinstance(raw, dict):
            raise RPCArchiveError("RPC annotation rows must be objects")
        annotation_id = _strict_nonnegative_int(raw.get("id"), "annotation id")
        image_id = _strict_nonnegative_int(raw.get("image_id"), "annotation image id")
        category_id = _strict_nonnegative_int(
            raw.get("category_id"), "annotation category id"
        )
        if annotation_id in annotation_ids:
            raise RPCArchiveError("RPC annotation ID is duplicated")
        annotation_ids.add(annotation_id)
        if image_id not in images or category_id not in categories:
            raise RPCArchiveError("RPC annotation references an unknown image/category")
        iscrowd = raw.get("iscrowd", 0)
        if isinstance(iscrowd, bool):
            iscrowd = int(iscrowd)
        if not isinstance(iscrowd, int) or iscrowd not in {0, 1}:
            raise RPCArchiveError("RPC annotation iscrowd must be 0 or 1")
        if iscrowd:
            continue
        bbox = _validate_raw_bbox(raw.get("bbox"), images[image_id])
        grouped[image_id].append(
            {
                "annotation_id": annotation_id,
                "category_id": category_id,
                "bbox_xywh": bbox,
            }
        )
    normalized = {
        image_id: tuple(
            sorted(rows, key=lambda row: (row["annotation_id"], row["category_id"]))
        )
        for image_id, rows in grouped.items()
    }
    return images, categories, normalized


def _select_candidates(
    *,
    images: dict[int, _ImageRow],
    instances_by_image: dict[int, tuple[dict[str, Any], ...]],
    archive_sha256: str,
) -> tuple[tuple[str, int], ...]:
    candidates: list[tuple[str, int]] = []
    for image_id, instances in instances_by_image.items():
        if len(instances) < 2:
            continue
        image = images[image_id]
        selection_key = hashlib.sha256(
            (
                f"{SELECTION_POLICY_VERSION}\n{archive_sha256}\n"
                f"{image_id}\n{image.file_name}"
            ).encode("utf-8")
        ).hexdigest()
        candidates.append((selection_key, image_id))
    return tuple(sorted(candidates))


def _prepare_scene(
    *,
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    roots: tuple[str, ...],
    canonical_root: str,
    annotation_member_paths: tuple[str, ...],
    annotation_sha256: str,
    image: _ImageRow,
    raw_instances: tuple[dict[str, Any], ...],
    categories: dict[int, str],
    selection_rank: int,
    selection_key: str,
    bundle_name: str,
    source_revision: str,
    source_url: str,
) -> _PreparedScene:
    replicas: list[tuple[str, zipfile.ZipInfo, bytes]] = []
    for root in roots:
        member = _member_path(root, f"{VALIDATION_DIRECTORY}/{image.file_name}")
        info = members.get(member)
        if info is None:
            raise RPCArchiveError(
                "ambiguous RPC duplicate root is missing selected image: "
                f"{root}:{image.file_name}"
            )
        replicas.append(
            (
                member,
                info,
                _read_bounded_member(
                    archive,
                    info,
                    maximum_bytes=MAX_IMAGE_BYTES,
                    label=f"RPC validation image {image.image_id}",
                ),
            )
        )
    image_hashes = {sha256_bytes(content) for _, _, content in replicas}
    if len(image_hashes) != 1:
        raise RPCArchiveError(
            f"ambiguous RPC duplicate roots differ for image {image.image_id}"
        )
    canonical_member = _member_path(
        canonical_root, f"{VALIDATION_DIRECTORY}/{image.file_name}"
    )
    canonical_replica = next(
        item for item in replicas if item[0] == canonical_member
    )
    raw_image = canonical_replica[2]
    try:
        with Image.open(io.BytesIO(raw_image)) as opened:
            actual_size = opened.size
            opened.verify()
    except (OSError, ValueError) as error:
        raise RPCArchiveError(
            f"RPC selected image is not decodable: {image.image_id}"
        ) from error
    if actual_size != (image.width, image.height):
        raise RPCArchiveError(
            f"RPC selected image dimensions differ from annotations: {image.image_id}"
        )

    instances = tuple(
        RPCInstance(
            annotation_id=raw["annotation_id"],
            category_id=raw["category_id"],
            category_name=categories[raw["category_id"]],
            sku_product_id=f"rpc:{raw['category_id']}",
            bbox_xywh=raw["bbox_xywh"],
            bbox_area=raw["bbox_xywh"][2] * raw["bbox_xywh"][3],
        )
        for raw in raw_instances
    )
    annotation_subset_sha256 = sha256_bytes(
        canonical_json_bytes(
            [item.model_dump(mode="json") for item in instances]
        )
    )
    suffix = PurePosixPath(image.file_name).suffix.lower()
    output_name = f"rpc-val-{image.image_id}{suffix}"
    bundle_image_path = f"{IMAGES_DIRECTORY}/{output_name}"
    local_path = f"{bundle_name}/{bundle_image_path}"
    counts = Counter(item.category_name for item in instances)
    bbox_areas = [item.bbox_area for item in instances]
    scene = RPCScene(
        selection_rank=selection_rank,
        selection_key_sha256=selection_key,
        source_record_id=f"val2019:{image.image_id}",
        image_id=image.image_id,
        image_file_name=image.file_name,
        image_member_path=canonical_member,
        replica_image_member_paths=tuple(sorted(item[0] for item in replicas)),
        image_sha256=sha256_bytes(raw_image),
        image_bytes=len(raw_image),
        image_width=image.width,
        image_height=image.height,
        image_zip_crc32=canonical_replica[1].CRC,
        annotation_member_paths=annotation_member_paths,
        annotation_document_sha256=annotation_sha256,
        annotation_subset_sha256=annotation_subset_sha256,
        instance_count=len(instances),
        unique_sku_count=len({item.category_id for item in instances}),
        category_ids=tuple(sorted({item.category_id for item in instances})),
        category_names=tuple(sorted({item.category_name for item in instances})),
        category_instance_counts=dict(sorted(counts.items())),
        bbox_count=len(instances),
        bbox_area_min=min(bbox_areas),
        bbox_area_max=max(bbox_areas),
        bbox_area_sum=sum(bbox_areas),
        dataset_asset_local_path=local_path,
        instances=instances,
    )
    draft = DatasetAssetDraft(
        source_dataset=SOURCE_DATASET,
        source_revision=source_revision,
        source_record_id=scene.source_record_id,
        transform_policy_version=TRANSFORM_POLICY_VERSION,
        local_path=local_path,
        product_id=None,
        derivation_parent_asset_ids=[],
        license_id=LICENSE_ID,
        source_url=source_url,
        attribution=ATTRIBUTION,
        cloud_upload_allowed=True,
        public_demo_allowed=True,
    )
    return _PreparedScene(
        scene=scene,
        draft=draft,
        bundle_image_path=bundle_image_path,
        image_bytes=raw_image,
    )


def _read_bounded_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if (
        info.is_dir()
        or info.file_size <= 0
        or info.file_size > maximum_bytes
        or info.compress_size < 0
        or (
            info.file_size > 0
            and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO
        )
    ):
        raise RPCArchiveError(f"{label} has unsafe ZIP size metadata")
    try:
        with archive.open(info) as source:
            content = source.read(maximum_bytes + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise RPCArchiveError(f"{label} cannot be read") from error
    if len(content) != info.file_size or len(content) > maximum_bytes:
        raise RPCArchiveError(f"{label} decompressed size differs from ZIP metadata")
    return content


def _validate_raw_bbox(
    value: object, image: _ImageRow
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise RPCArchiveError("RPC annotation bbox must contain four finite numbers")
    x, y, width, height = (float(item) for item in value)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image.width + 1e-6
        or y + height > image.height + 1e-6
    ):
        raise RPCArchiveError("RPC annotation bbox lies outside its image")
    return x, y, width, height


def _validate_image_file_name(value: object) -> str:
    name = _strict_nonblank_text(value, "image file_name")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != name
        or "\\" in name
        or path.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise RPCArchiveError(f"unsafe RPC image file_name: {name!r}")
    return name


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RPCArchiveError(f"RPC {label} must be a non-negative integer")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    result = _strict_nonnegative_int(value, label)
    if result == 0:
        raise RPCArchiveError(f"RPC {label} must be positive")
    return result


def _strict_nonblank_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RPCArchiveError(f"RPC {label} must be canonical nonblank text")
    return value


def _member_path(root: str, suffix: str) -> str:
    return suffix if root == "." else f"{root}/{suffix}"


def _json_native(value: object) -> object:
    """Convert immutable in-memory containers to strict JSON containers."""

    if isinstance(value, tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    return value


def _canonical_relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _validate_bundle_name(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise RPCAdapterError("RPC output directory name is not canonical")
    return value


def _bundle_path_from_local_path(
    local_path: str, manifest: RPCAdapterManifest
) -> str:
    prefix = f"{manifest.bundle_directory_name}/"
    if not local_path.startswith(prefix):
        raise RPCBundleError("RPC asset local_path is outside the bundle")
    return _canonical_relative_path(
        local_path[len(prefix) :], "RPC bundle payload path"
    )


def _parse_canonical_rows(content: bytes, model, label: str) -> tuple:
    try:
        raw_rows = parse_canonical_jsonl(content, label=label)
        rows = tuple(model.model_validate(row, strict=True) for row in raw_rows)
    except (ArtifactFormatError, ValueError) as error:
        raise RPCBundleError(f"{label} are invalid") from error
    if not rows or canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in rows)
    ) != content:
        raise RPCBundleError(f"{label} are not canonical")
    return rows


def _file_snapshot(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _summary(result: RPCAdapterBuildResult) -> dict[str, object]:
    return {
        "archive_sha256": result.manifest.archive_sha256,
        "candidate_scene_count": result.manifest.candidate_scene_count,
        "manifest_file_sha256": result.manifest_file_sha256,
        "output_dir": str(result.output_dir),
        "selected_instance_count": result.manifest.selected_instance_count,
        "selected_scene_count": result.manifest.selected_scene_count,
        "status": result.manifest.status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK_PATH)
    parser.add_argument(
        "--expected-source-lock-sha256",
        default=EXPECTED_SOURCE_LOCK_SHA256,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args(argv)
    try:
        result = build_rpc_val_query_adapter(
            raw_root=args.raw_root,
            source_lock_path=args.source_lock,
            expected_source_lock_sha256=args.expected_source_lock_sha256,
            output_dir=args.output,
            count=args.count,
        )
    except (OSError, SourceLockError, RPCAdapterError, FileExistsError) as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            _summary(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_ID",
    "DEFAULT_COUNT",
    "EXPECTED_SOURCE_LOCK_SHA256",
    "RPCAdapterBuildResult",
    "RPCAdapterError",
    "RPCAdapterManifest",
    "RPCArchiveError",
    "RPCBundleError",
    "RPCInstance",
    "RPCScene",
    "VerifiedRPCAdapterBundle",
    "build_rpc_val_query_adapter",
    "load_verified_rpc_val_adapter",
    "main",
]
