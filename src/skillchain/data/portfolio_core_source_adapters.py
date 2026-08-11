"""Selected-only ABO/RPC source adapters for the Portfolio core inventory.

The adapters in this module deliberately consume only bounded members from
the already locked source archives.  They never unpack an archive tree and
never mutate RAW.  Every selected image is decoded, fingerprinted, published
create-only, and followed by an immutable checkpoint, so an interrupted run
can safely continue.

The final ``candidates.jsonl`` files use the strict
``PortfolioCoreAssetCandidate`` contract consumed by
``portfolio_core_assets.py``.  The prior dev-mini selection manifest is a
required exclusion boundary: matching source records, products, and exact
content hashes can never re-enter the fresh core inventory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Literal, Mapping, Sequence
import zipfile

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from skillchain import config
from skillchain.data import rpc as rpc_source
from skillchain.data.abo import LICENSE_ID as ABO_LICENSE_ID
from skillchain.data.abo_archive_review import (
    _index_image_members,
    _select_candidates,
)
from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.data.portfolio_core_assets import (
    CORE_POOL_ASSET_FLOORS,
    CandidateCapabilityBinding,
    PortfolioCoreAssetCandidate,
    PortfolioCoreAssetError,
    load_candidate_inventory,
    load_source_policy,
)
from skillchain.data.source_lock import (
    RequiredSourceLock,
    SourceLockError,
    load_required_source_lock,
    stable_file_digest,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    parse_canonical_json,
    read_stable_regular_file,
)


ADAPTER_POLICY_VERSION = "portfolio-core-selected-source-adapters-v1"
ABO_TRANSFORM_POLICY_VERSION = "abo-compact-selected-member-v1"
RPC_TRANSFORM_POLICY_VERSION = "rpc-val-selected-member-v1"

DEFAULT_ABO_INCLUDE_COUNT = 340
DEFAULT_RPC_INCLUDE_COUNT = CORE_POOL_ASSET_FLOORS["multi_product"]
DEFAULT_ABO_RESERVE_COUNT = 8
DEFAULT_RESERVE_COUNT = 4
DEFAULT_ABO_CROSS_INTENT_COUNT = 25

ABO_SOURCE_LOCK_PATH = (
    config.ROOT
    / "specs"
    / "data_sources"
    / "c2"
    / "source-locks"
    / "abo.source-lock.json"
)
ABO_EXPECTED_SOURCE_LOCK_SHA256 = (
    "337a0b2dbf47701420fbf4d2b6b9ae4a5ca23b1990748c1bbb61c642234cc48e"
)
RPC_SOURCE_LOCK_PATH = rpc_source.SOURCE_LOCK_PATH
RPC_EXPECTED_SOURCE_LOCK_SHA256 = rpc_source.EXPECTED_SOURCE_LOCK_SHA256

_ABO_LISTINGS_LOGICAL_PATH = "abo/archives/abo-listings.tar"
_ABO_IMAGES_LOGICAL_PATH = "abo/archives/abo-images-small.tar"
_ABO_ATTRIBUTION = "Amazon Berkeley Objects (ABO); exact compact archive member bytes"
_ABO_SELECTION_BUFFER = 64

_RUN_FILE = "adapter-run.json"
_CHECKPOINT_DIRECTORY = "checkpoints"
_CANDIDATES_FILE = "candidates.jsonl"
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PortfolioCoreSourceAdapterError(ValueError):
    """A selected source run is inconsistent or cannot fill its quota."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _ArchiveIdentity(_StrictModel):
    logical_path: str
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("archive sha256 must be lowercase hexadecimal")
        return value


class _SourceRunManifest(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal[
        "portfolio-core-selected-source-adapters-v1"
    ] = ADAPTER_POLICY_VERSION
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    source_id: Literal["abo", "rpc"]
    source_revision: str
    source_lock_sha256: str
    source_policy_sha256: str
    dev_mini_selection_sha256: str
    include_count: int = Field(gt=0)
    reserve_count: int = Field(ge=0)
    cross_intent_count: int = Field(ge=0)
    raw_mutation_performed: Literal[False] = False
    archives: tuple[_ArchiveIdentity, ...] = Field(min_length=1)

    @field_validator(
        "source_lock_sha256",
        "source_policy_sha256",
        "dev_mini_selection_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("manifest sha256 fields must be lowercase hexadecimal")
        return value

    @field_validator("archives", mode="before")
    @classmethod
    def coerce_archives(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class _CandidateCheckpoint(_StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal[
        "portfolio-core-selected-source-adapters-v1"
    ] = ADAPTER_POLICY_VERSION
    archive_member: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    candidate_sha256: str
    candidate: PortfolioCoreAssetCandidate

    @field_validator("candidate_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("candidate_sha256 must be lowercase hexadecimal")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.candidate_sha256 != _candidate_sha256(self.candidate):
            raise ValueError("checkpoint candidate digest differs")


@dataclass(frozen=True)
class _Exclusions:
    manifest_sha256: str
    records: Mapping[str, frozenset[str]]
    products: Mapping[str, frozenset[str]]
    hashes: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class _ResolvedArchive:
    path: Path
    logical_path: str
    sha256: str
    bytes: int
    url: str
    snapshot: tuple[int, int, int, int]

    @property
    def identity(self) -> _ArchiveIdentity:
        return _ArchiveIdentity(
            logical_path=self.logical_path,
            bytes=self.bytes,
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class PortfolioCoreSourceAdapterResult:
    source_id: str
    output_root: Path
    complete: bool
    include_count: int
    reserve_count: int
    checkpoint_count: int
    published_this_call: int
    candidates_path: Path | None
    candidates_sha256: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "candidates_path": (
                str(self.candidates_path) if self.candidates_path is not None else None
            ),
            "candidates_sha256": self.candidates_sha256,
            "checkpoint_count": self.checkpoint_count,
            "complete": self.complete,
            "include_count": self.include_count,
            "output_root": str(self.output_root),
            "published_this_call": self.published_this_call,
            "reserve_count": self.reserve_count,
            "source_id": self.source_id,
            "status": "complete" if self.complete else "in_progress",
        }


def build_abo_portfolio_core_candidates(
    *,
    raw_root: str | Path,
    source_lock_path: str | Path,
    expected_source_lock_sha256: str,
    source_policy_path: str | Path,
    dev_mini_selection_manifest: str | Path,
    output_root: str | Path,
    include_count: int = DEFAULT_ABO_INCLUDE_COUNT,
    reserve_count: int = DEFAULT_ABO_RESERVE_COUNT,
    cross_intent_count: int = DEFAULT_ABO_CROSS_INTENT_COUNT,
    max_items: int | None = None,
) -> PortfolioCoreSourceAdapterResult:
    """Publish fresh ABO compact-archive candidates without unpacking RAW."""

    _validate_counts(
        include_count=include_count,
        reserve_count=reserve_count,
        cross_intent_count=cross_intent_count,
        max_items=max_items,
    )
    policy_sha256 = _require_ready_policy(
        source_policy_path, source_id="abo", pool="exact_match"
    )
    exclusions = load_dev_mini_exclusions(dev_mini_selection_manifest)
    lock = load_required_source_lock(
        source_lock_path,
        expected_lock_file_sha256=expected_source_lock_sha256,
    )
    listings, images = _resolve_abo_archives(lock, Path(raw_root))
    output = _initialize_output(
        output_root=output_root,
        forbidden_roots=(raw_root,),
        manifest=_SourceRunManifest(
            source_id="abo",
            source_revision=lock.source_revision,
            source_lock_sha256=expected_source_lock_sha256,
            source_policy_sha256=policy_sha256,
            dev_mini_selection_sha256=exclusions.manifest_sha256,
            include_count=include_count,
            reserve_count=reserve_count,
            cross_intent_count=cross_intent_count,
            archives=(images.identity, listings.identity),
        ),
    )

    target = include_count + reserve_count
    exclusion_weight = sum(
        len(values)
        for values in (
            exclusions.records.get("abo", frozenset()),
            exclusions.products.get("abo", frozenset()),
            exclusions.hashes.get("abo", frozenset()),
        )
    )
    pair_plan_count = max(
        math.ceil(target / 2) + exclusion_weight + _ABO_SELECTION_BUFFER,
        target,
    )
    image_members = _index_image_members(images.path)
    ranked_pairs, _ = _select_candidates(
        listings.path,
        image_members,
        maximum_pair_candidates=pair_plan_count,
    )

    selected: list[PortfolioCoreAssetCandidate] = []
    seen_hashes: set[str] = set()
    published = 0
    with tarfile.open(images.path, mode="r:") as archive:
        for pair in ranked_pairs:
            product_id = f"abo:{pair['item_id']}"
            if product_id in exclusions.products.get("abo", frozenset()):
                continue
            for role in ("main", "other"):
                image_id = pair[f"{role}_image_id"]
                record_id = f"listing:{pair['item_id']}/image:{image_id}"
                record_alias = f"{pair['item_id']}:{image_id}"
                excluded_records = exclusions.records.get("abo", frozenset())
                if record_id in excluded_records or record_alias in excluded_records:
                    continue
                member_name = pair[f"{role}_member"]
                content, width, height = _read_abo_image_member(
                    archive, member_name
                )
                content_sha256 = sha256_bytes(content)
                if (
                    content_sha256 in exclusions.hashes.get("abo", frozenset())
                    or content_sha256 in seen_hashes
                ):
                    continue
                selection_index = len(selected) + 1
                selection: Literal["include", "reserve"] = (
                    "include" if selection_index <= include_count else "reserve"
                )
                candidate = _abo_candidate(
                    pair=pair,
                    role=role,
                    content=content,
                    selection_index=selection_index,
                    selection=selection,
                    source_revision=lock.source_revision,
                    source_url=images.url,
                    cross_intent=(
                        selection == "include"
                        and selection_index <= cross_intent_count
                    ),
                )
                checkpoint_created = _publish_candidate_checkpoint(
                    output,
                    candidate=candidate,
                    content=content,
                    archive_member=member_name,
                    width=width,
                    height=height,
                )
                selected.append(candidate)
                seen_hashes.add(content_sha256)
                if checkpoint_created:
                    published += 1
                if len(selected) == target:
                    break
                if max_items is not None and published >= max_items:
                    break
            if len(selected) == target or (
                max_items is not None and published >= max_items
            ):
                break

    _verify_archive_unchanged(images)
    _verify_archive_unchanged(listings)
    return _finish_source_run(
        source_id="abo",
        output_root=output,
        candidates=selected,
        include_count=include_count,
        reserve_count=reserve_count,
        published_this_call=published,
        exhausted=len(selected) < target and not (
            max_items is not None and published >= max_items
        ),
    )


def build_rpc_portfolio_core_candidates(
    *,
    raw_root: str | Path,
    source_lock_path: str | Path,
    expected_source_lock_sha256: str,
    source_policy_path: str | Path,
    dev_mini_selection_manifest: str | Path,
    output_root: str | Path,
    include_count: int = DEFAULT_RPC_INCLUDE_COUNT,
    reserve_count: int = DEFAULT_RESERVE_COUNT,
    max_items: int | None = None,
) -> PortfolioCoreSourceAdapterResult:
    """Publish fresh RPC multi-scene candidates without unpacking RAW."""

    _validate_counts(
        include_count=include_count,
        reserve_count=reserve_count,
        cross_intent_count=0,
        max_items=max_items,
    )
    policy_sha256 = _require_ready_policy(
        source_policy_path, source_id="rpc", pool="multi_product"
    )
    exclusions = load_dev_mini_exclusions(dev_mini_selection_manifest)
    lock = load_required_source_lock(
        source_lock_path,
        expected_lock_file_sha256=expected_source_lock_sha256,
    )
    archive_path, logical_path, archive_sha256, archive_bytes = (
        rpc_source._resolve_locked_archive(lock, Path(raw_root))
    )
    archive = _ResolvedArchive(
        path=archive_path,
        logical_path=logical_path,
        sha256=archive_sha256,
        bytes=archive_bytes,
        url=rpc_source._archive_identity(lock, logical_path).url,
        snapshot=_file_snapshot(archive_path),
    )
    output = _initialize_output(
        output_root=output_root,
        forbidden_roots=(raw_root,),
        manifest=_SourceRunManifest(
            source_id="rpc",
            source_revision=lock.source_revision,
            source_lock_sha256=expected_source_lock_sha256,
            source_policy_sha256=policy_sha256,
            dev_mini_selection_sha256=exclusions.manifest_sha256,
            include_count=include_count,
            reserve_count=reserve_count,
            cross_intent_count=0,
            archives=(archive.identity,),
        ),
    )

    target = include_count + reserve_count
    selected: list[PortfolioCoreAssetCandidate] = []
    seen_hashes: set[str] = set()
    published = 0
    try:
        with zipfile.ZipFile(archive.path) as opened:
            members = rpc_source._validate_archive_members(opened)
            (
                annotation_bytes,
                annotation_member_paths,
                replica_roots,
                canonical_root,
            ) = rpc_source._load_annotation_replicas(opened, members)
            annotation_sha256 = sha256_bytes(annotation_bytes)
            annotation = rpc_source.parse_strict_json(
                annotation_bytes, label="RPC validation annotations"
            )
            images, categories, instances_by_image = (
                rpc_source._validate_annotations(annotation)
            )
            ranked = rpc_source._select_candidates(
                images=images,
                instances_by_image=instances_by_image,
                archive_sha256=archive.sha256,
            )
            for rank, (selection_key, image_id) in enumerate(ranked, start=1):
                record_id = f"val2019:{image_id}"
                if record_id in exclusions.records.get("rpc", frozenset()):
                    continue
                prepared = rpc_source._prepare_scene(
                    archive=opened,
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
                    bundle_name="portfolio-core-rpc-selected",
                    source_revision=lock.source_revision,
                    source_url=archive.url,
                )
                content_sha256 = prepared.scene.image_sha256
                if (
                    content_sha256 in exclusions.hashes.get("rpc", frozenset())
                    or content_sha256 in seen_hashes
                ):
                    continue
                selection_index = len(selected) + 1
                selection: Literal["include", "reserve"] = (
                    "include" if selection_index <= include_count else "reserve"
                )
                candidate = _rpc_candidate(
                    prepared=prepared,
                    selection_index=selection_index,
                    selection=selection,
                )
                checkpoint_created = _publish_candidate_checkpoint(
                    output,
                    candidate=candidate,
                    content=prepared.image_bytes,
                    archive_member=prepared.scene.image_member_path,
                    width=prepared.scene.image_width,
                    height=prepared.scene.image_height,
                )
                selected.append(candidate)
                seen_hashes.add(content_sha256)
                if checkpoint_created:
                    published += 1
                if len(selected) == target:
                    break
                if max_items is not None and published >= max_items:
                    break
    except zipfile.BadZipFile as error:
        raise PortfolioCoreSourceAdapterError(
            "RPC locked archive is not a valid ZIP"
        ) from error

    _verify_archive_unchanged(archive)
    return _finish_source_run(
        source_id="rpc",
        output_root=output,
        candidates=selected,
        include_count=include_count,
        reserve_count=reserve_count,
        published_this_call=published,
        exhausted=len(selected) < target and not (
            max_items is not None and published >= max_items
        ),
    )


def load_dev_mini_exclusions(path: str | Path) -> _Exclusions:
    """Load the prior Portfolio dev-mini manifest as an immutable exclusion set."""

    content = _read_metadata(Path(path), "dev-mini selection manifest")
    try:
        value = parse_canonical_json(content, label="dev-mini selection manifest")
    except ArtifactFormatError as exc:
        raise PortfolioCoreSourceAdapterError(str(exc)) from exc
    if content != canonical_json_bytes(value):
        raise PortfolioCoreSourceAdapterError(
            "dev-mini selection manifest must be canonical JSON"
        )
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("track") != "portfolio"
        or value.get("formal_eligible") is not False
        or value.get("formal_status") != "non_formal"
    ):
        raise PortfolioCoreSourceAdapterError(
            "dev-mini selection manifest boundary fields are invalid"
        )
    rows = value.get("selections")
    if not isinstance(rows, list) or not rows:
        raise PortfolioCoreSourceAdapterError(
            "dev-mini selection manifest has no selections"
        )
    if value.get("asset_count") != len(rows):
        raise PortfolioCoreSourceAdapterError(
            "dev-mini selection manifest asset_count differs"
        )
    records: dict[str, set[str]] = {"abo": set(), "rpc": set()}
    products: dict[str, set[str]] = {"abo": set(), "rpc": set()}
    hashes: dict[str, set[str]] = {"abo": set(), "rpc": set()}
    for row in rows:
        if not isinstance(row, dict):
            raise PortfolioCoreSourceAdapterError(
                "dev-mini selection manifest row is invalid"
            )
        source_id = row.get("source_dataset")
        if source_id not in records:
            continue
        record_id = row.get("source_record_id")
        image_sha256 = row.get("image_sha256")
        product_id = row.get("product_id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not isinstance(image_sha256, str)
            or not _SHA256_RE.fullmatch(image_sha256)
            or (product_id is not None and not isinstance(product_id, str))
        ):
            raise PortfolioCoreSourceAdapterError(
                f"dev-mini {source_id} exclusion row is invalid"
            )
        records[source_id].add(record_id)
        hashes[source_id].add(image_sha256)
        if product_id:
            products[source_id].add(product_id)
    for source_id in ("abo", "rpc"):
        if not records[source_id] or not hashes[source_id]:
            raise PortfolioCoreSourceAdapterError(
                f"dev-mini manifest lacks {source_id} exclusion evidence"
            )
    return _Exclusions(
        manifest_sha256=sha256_bytes(content),
        records={key: frozenset(value) for key, value in records.items()},
        products={key: frozenset(value) for key, value in products.items()},
        hashes={key: frozenset(value) for key, value in hashes.items()},
    )


def _validate_counts(
    *,
    include_count: int,
    reserve_count: int,
    cross_intent_count: int,
    max_items: int | None,
) -> None:
    if (
        isinstance(include_count, bool)
        or not isinstance(include_count, int)
        or include_count <= 0
        or isinstance(reserve_count, bool)
        or not isinstance(reserve_count, int)
        or reserve_count < 0
        or isinstance(cross_intent_count, bool)
        or not isinstance(cross_intent_count, int)
        or cross_intent_count < 0
        or cross_intent_count > include_count
    ):
        raise ValueError("adapter counts are invalid")
    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0
    ):
        raise ValueError("max_items must be a positive integer when supplied")


def _require_ready_policy(
    path: str | Path,
    *,
    source_id: str,
    pool: str,
) -> str:
    policy, policy_sha256 = load_source_policy(path)
    source = next(
        (item for item in policy.sources if item.source_id == source_id), None
    )
    if source is None or source.availability != "ready" or pool not in source.allowed_pools:
        raise PortfolioCoreSourceAdapterError(
            f"Portfolio source policy does not enable {source_id}:{pool}"
        )
    return policy_sha256


def _resolve_abo_archives(
    lock: RequiredSourceLock,
    raw_root: Path,
) -> tuple[_ResolvedArchive, _ResolvedArchive]:
    if lock.source_id != "abo":
        raise PortfolioCoreSourceAdapterError("source lock does not target ABO")
    identities = {item.logical_path: item for item in lock.acquisition_identities}
    if len(identities) != len(lock.acquisition_identities):
        raise PortfolioCoreSourceAdapterError("ABO source lock identities are duplicated")
    scopes = [
        scope
        for scope in lock.artifact_scopes
        if scope.mode == "explicit_files"
    ]
    root = raw_root.resolve(strict=True)
    resolved: list[_ResolvedArchive] = []
    for logical_path in (
        _ABO_LISTINGS_LOGICAL_PATH,
        _ABO_IMAGES_LOGICAL_PATH,
    ):
        if not any(logical_path in scope.paths for scope in scopes):
            raise PortfolioCoreSourceAdapterError(
                f"ABO source lock does not cover {logical_path}"
            )
        identity = identities.get(logical_path)
        if identity is None:
            raise PortfolioCoreSourceAdapterError(
                f"ABO source lock lacks identity for {logical_path}"
            )
        path = (root / Path(*PurePosixPath(logical_path).parts)).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError:
            raise PortfolioCoreSourceAdapterError(
                "ABO archive resolves outside RAW root"
            ) from None
        digest, size = stable_file_digest(path, label=f"ABO {Path(logical_path).name}")
        if digest != identity.local_sha256 or size != identity.bytes:
            raise PortfolioCoreSourceAdapterError(
                f"ABO locked archive identity drifted: {logical_path}"
            )
        resolved.append(
            _ResolvedArchive(
                path=path,
                logical_path=logical_path,
                sha256=digest,
                bytes=size,
                url=identity.url,
                snapshot=_file_snapshot(path),
            )
        )
    return resolved[0], resolved[1]


def _initialize_output(
    *,
    output_root: str | Path,
    manifest: _SourceRunManifest,
    forbidden_roots: Sequence[str | Path] = (),
) -> Path:
    output = Path(output_root).absolute()
    _require_disjoint_output_root(
        output,
        forbidden_roots,
        label="source adapter output",
    )
    _ensure_real_directory(output, create=True, label="source adapter output")
    _ensure_real_directory(
        output / _CHECKPOINT_DIRECTORY,
        create=True,
        label="source adapter checkpoints",
    )
    _publish_idempotent(output / _RUN_FILE, canonical_json_bytes(manifest))
    return output


def _require_disjoint_output_root(
    output_root: Path,
    forbidden_roots: Sequence[str | Path],
    *,
    label: str,
) -> None:
    """Reject every ancestor/descendant overlap before creating output bytes."""

    try:
        resolved_output = output_root.resolve(strict=False)
    except OSError as exc:
        raise PortfolioCoreSourceAdapterError(
            f"cannot resolve {label}: {output_root}"
        ) from exc
    for raw_root in forbidden_roots:
        raw_path = Path(raw_root).absolute()
        try:
            resolved_raw = raw_path.resolve(strict=True)
        except OSError as exc:
            raise PortfolioCoreSourceAdapterError(
                f"cannot resolve forbidden source root: {raw_path}"
            ) from exc
        if _paths_overlap(resolved_output, resolved_raw):
            raise PortfolioCoreSourceAdapterError(
                f"{label} must be disjoint from RAW/source root: {raw_path}"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _read_abo_image_member(
    archive: tarfile.TarFile,
    member_name: str,
) -> tuple[bytes, int, int]:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise PortfolioCoreSourceAdapterError(
            f"ABO selected member is missing: {member_name}"
        ) from exc
    if (
        not member.isfile()
        or member.issym()
        or member.islnk()
        or member.size <= 0
        or member.size > _MAX_IMAGE_BYTES
    ):
        raise PortfolioCoreSourceAdapterError(
            f"ABO selected member is unsafe: {member_name}"
        )
    stream = archive.extractfile(member)
    if stream is None:
        raise PortfolioCoreSourceAdapterError(
            f"ABO selected member cannot be read: {member_name}"
        )
    content = stream.read(_MAX_IMAGE_BYTES + 1)
    if len(content) != member.size or len(content) > _MAX_IMAGE_BYTES:
        raise PortfolioCoreSourceAdapterError(
            f"ABO selected member byte count differs: {member_name}"
        )
    width, height = _decode_image(content, f"ABO selected member {member_name}")
    return content, width, height


def _decode_image(content: bytes, label: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(content)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(content)) as opened:
            opened.load()
            width, height = opened.size
    except (OSError, ValueError) as exc:
        raise PortfolioCoreSourceAdapterError(f"{label} is not decodable") from exc
    if width <= 0 or height <= 0:
        raise PortfolioCoreSourceAdapterError(f"{label} has invalid dimensions")
    return width, height


def _abo_candidate(
    *,
    pair: Mapping[str, Any],
    role: str,
    content: bytes,
    selection_index: int,
    selection: Literal["include", "reserve"],
    source_revision: str,
    source_url: str,
    cross_intent: bool,
) -> PortfolioCoreAssetCandidate:
    image_id = pair[f"{role}_image_id"]
    source_record_id = f"listing:{pair['item_id']}/image:{image_id}"
    suffix = PurePosixPath(pair[f"{role}_member"]).suffix.lower()
    filename = f"abo-core-{selection_index:04d}{suffix}"
    local_path = f"query_images/exact_match/{filename}"
    identity = sha256_bytes(
        f"{ADAPTER_POLICY_VERSION}\nabo\n{pair['item_id']}\n{image_id}".encode()
    )
    bindings = [
        CandidateCapabilityBinding(
            canonical_intent="exact_match",
            canonical_capability="product.exact_match",
        )
    ]
    if cross_intent:
        bindings.extend(
            (
                CandidateCapabilityBinding(
                    canonical_intent="divergent_rec",
                    canonical_capability="product.style_recommendation",
                ),
                CandidateCapabilityBinding(
                    canonical_intent="encyclopedia",
                    canonical_capability="knowledge.visual_encyclopedia",
                ),
            )
        )
    return PortfolioCoreAssetCandidate(
        candidate_id=f"abo.core.{identity[:24]}",
        source_id="abo",
        selection=selection,
        pool="exact_match",
        source_local_path=local_path,
        destination_path=local_path,
        expected_bytes=len(content),
        expected_sha256=sha256_bytes(content),
        draft=DatasetAssetDraft(
            source_dataset="abo",
            source_revision=source_revision,
            source_record_id=source_record_id,
            transform_policy_version=ABO_TRANSFORM_POLICY_VERSION,
            local_path=local_path,
            product_id=f"abo:{pair['item_id']}",
            license_id=ABO_LICENSE_ID,
            source_url=source_url,
            attribution=_ABO_ATTRIBUTION,
            cloud_upload_allowed=True,
            public_demo_allowed=True,
        ),
        capability_bindings=tuple(bindings),
    )


def _rpc_candidate(
    *,
    prepared: Any,
    selection_index: int,
    selection: Literal["include", "reserve"],
) -> PortfolioCoreAssetCandidate:
    suffix = PurePosixPath(prepared.scene.image_file_name).suffix.lower()
    filename = f"rpc-core-{selection_index:04d}-{prepared.scene.image_id}{suffix}"
    local_path = f"query_images/multi_product/{filename}"
    identity = sha256_bytes(
        (
            f"{ADAPTER_POLICY_VERSION}\nrpc\n"
            f"{prepared.scene.source_record_id}\n"
            f"{prepared.scene.selection_key_sha256}"
        ).encode()
    )
    return PortfolioCoreAssetCandidate(
        candidate_id=f"rpc.core.{identity[:24]}",
        source_id="rpc",
        selection=selection,
        pool="multi_product",
        source_local_path=local_path,
        destination_path=local_path,
        expected_bytes=len(prepared.image_bytes),
        expected_sha256=prepared.scene.image_sha256,
        draft=prepared.draft.model_copy(
            update={
                "local_path": local_path,
                "transform_policy_version": RPC_TRANSFORM_POLICY_VERSION,
            }
        ),
        capability_bindings=(
            CandidateCapabilityBinding(
                canonical_intent="multi_product",
                canonical_capability="product.multi_search",
            ),
        ),
    )


def _publish_candidate_checkpoint(
    output_root: Path,
    *,
    candidate: PortfolioCoreAssetCandidate,
    content: bytes,
    archive_member: str,
    width: int,
    height: int,
) -> bool:
    destination = output_root / Path(*PurePosixPath(candidate.source_local_path).parts)
    _ensure_real_directory(
        destination.parent,
        create=True,
        label=f"candidate parent {candidate.candidate_id}",
    )
    _publish_or_verify_asset(destination, content, candidate)
    checkpoint = _CandidateCheckpoint(
        archive_member=archive_member,
        image_width=width,
        image_height=height,
        candidate_sha256=_candidate_sha256(candidate),
        candidate=candidate,
    )
    checkpoint_path = output_root / _CHECKPOINT_DIRECTORY / f"{candidate.candidate_id}.json"
    checkpoint_bytes = canonical_json_bytes(checkpoint)
    if os.path.lexists(checkpoint_path):
        existing = _read_metadata(
            checkpoint_path, f"checkpoint {candidate.candidate_id}"
        )
        try:
            loaded = _CandidateCheckpoint.model_validate_json(existing, strict=True)
        except ValidationError as exc:
            raise PortfolioCoreSourceAdapterError(
                f"candidate checkpoint is invalid: {candidate.candidate_id}"
            ) from exc
        if existing != canonical_json_bytes(loaded) or existing != checkpoint_bytes:
            raise PortfolioCoreSourceAdapterError(
                f"candidate checkpoint conflicts: {candidate.candidate_id}"
            )
        _verify_asset(destination, candidate)
        return False
    try:
        atomic_create_file(checkpoint_path, checkpoint_bytes)
    except FileExistsError:
        return _publish_candidate_checkpoint(
            output_root,
            candidate=candidate,
            content=content,
            archive_member=archive_member,
            width=width,
            height=height,
        )
    return True


def _publish_or_verify_asset(
    path: Path,
    content: bytes,
    candidate: PortfolioCoreAssetCandidate,
) -> None:
    if os.path.lexists(path):
        _verify_asset(path, candidate)
        return
    try:
        atomic_create_file(path, content)
    except FileExistsError:
        _verify_asset(path, candidate)


def _verify_asset(path: Path, candidate: PortfolioCoreAssetCandidate) -> None:
    content = _read_metadata(path, f"candidate image {candidate.candidate_id}")
    if (
        len(content) != candidate.expected_bytes
        or sha256_bytes(content) != candidate.expected_sha256
    ):
        raise PortfolioCoreSourceAdapterError(
            f"published candidate bytes conflict: {candidate.candidate_id}"
        )
    _decode_image(content, f"published candidate {candidate.candidate_id}")


def _finish_source_run(
    *,
    source_id: Literal["abo", "rpc"],
    output_root: Path,
    candidates: Sequence[PortfolioCoreAssetCandidate],
    include_count: int,
    reserve_count: int,
    published_this_call: int,
    exhausted: bool,
) -> PortfolioCoreSourceAdapterResult:
    target = include_count + reserve_count
    if exhausted:
        raise PortfolioCoreSourceAdapterError(
            f"{source_id} selected archive members cannot fill the requested fresh quota: "
            f"selected={len(candidates)} requested={target}"
        )
    if len(candidates) < target:
        return PortfolioCoreSourceAdapterResult(
            source_id=source_id,
            output_root=output_root,
            complete=False,
            include_count=sum(item.selection == "include" for item in candidates),
            reserve_count=sum(item.selection == "reserve" for item in candidates),
            checkpoint_count=len(candidates),
            published_this_call=published_this_call,
            candidates_path=None,
            candidates_sha256=None,
        )
    if len({item.expected_sha256 for item in candidates}) != target:
        raise PortfolioCoreSourceAdapterError(
            f"{source_id} completed selection is not content-unique"
        )
    checkpoint_names = {
        path.name
        for path in (output_root / _CHECKPOINT_DIRECTORY).iterdir()
        if path.is_file()
    }
    expected_checkpoint_names = {f"{item.candidate_id}.json" for item in candidates}
    if checkpoint_names != expected_checkpoint_names:
        raise PortfolioCoreSourceAdapterError(
            f"{source_id} checkpoint set contains unexpected or missing files"
        )
    inventory_path = output_root / _CANDIDATES_FILE
    inventory_bytes = canonical_jsonl_bytes(tuple(candidates))
    _publish_idempotent(inventory_path, inventory_bytes)
    loaded, inventory_sha256 = load_candidate_inventory(inventory_path)
    if loaded != tuple(candidates):
        raise PortfolioCoreSourceAdapterError(
            f"{source_id} candidate inventory changed after publication"
        )
    return PortfolioCoreSourceAdapterResult(
        source_id=source_id,
        output_root=output_root,
        complete=True,
        include_count=include_count,
        reserve_count=reserve_count,
        checkpoint_count=target,
        published_this_call=published_this_call,
        candidates_path=inventory_path,
        candidates_sha256=inventory_sha256,
    )


def _candidate_sha256(candidate: PortfolioCoreAssetCandidate) -> str:
    return sha256_bytes(canonical_json_bytes(candidate))


def _publish_idempotent(path: Path, content: bytes) -> Path:
    if os.path.lexists(path):
        existing = _read_metadata(path, f"existing {path.name}")
        if existing != content:
            raise FileExistsError(f"existing create-only file conflicts: {path}")
        return path
    try:
        atomic_create_file(path, content)
    except FileExistsError:
        return _publish_idempotent(path, content)
    return path


def _read_metadata(path: Path, label: str) -> bytes:
    try:
        return read_stable_regular_file(
            path,
            label=label,
            max_bytes=(
                _MAX_IMAGE_BYTES
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                else _MAX_METADATA_BYTES
            ),
        )
    except ArtifactFormatError as exc:
        raise PortfolioCoreSourceAdapterError(str(exc)) from exc


def _ensure_real_directory(
    path: Path,
    *,
    create: bool,
    label: str,
) -> None:
    if not os.path.lexists(path):
        if not create:
            raise PortfolioCoreSourceAdapterError(f"{label} does not exist")
        path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    is_junction = getattr(path, "is_junction", None)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or (is_junction is not None and is_junction())
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PortfolioCoreSourceAdapterError(
            f"{label} must be a real non-link directory"
        )


def _file_snapshot(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _verify_archive_unchanged(archive: _ResolvedArchive) -> None:
    if _file_snapshot(archive.path) != archive.snapshot:
        raise PortfolioCoreSourceAdapterError(
            f"locked archive changed during selected materialization: {archive.logical_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for source_id, include_count, reserve_count, lock_path, lock_sha in (
        (
            "abo",
            DEFAULT_ABO_INCLUDE_COUNT,
            DEFAULT_ABO_RESERVE_COUNT,
            ABO_SOURCE_LOCK_PATH,
            ABO_EXPECTED_SOURCE_LOCK_SHA256,
        ),
        (
            "rpc",
            DEFAULT_RPC_INCLUDE_COUNT,
            DEFAULT_RESERVE_COUNT,
            RPC_SOURCE_LOCK_PATH,
            RPC_EXPECTED_SOURCE_LOCK_SHA256,
        ),
    ):
        command = subparsers.add_parser(source_id)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--source-lock", type=Path, default=lock_path)
        command.add_argument("--source-lock-sha256", default=lock_sha)
        command.add_argument("--source-policy", type=Path, required=True)
        command.add_argument(
            "--dev-mini-selection-manifest", type=Path, required=True
        )
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--include-count", type=int, default=include_count)
        command.add_argument(
            "--reserve-count", type=int, default=reserve_count
        )
        command.add_argument("--max-items", type=int)
        if source_id == "abo":
            command.add_argument(
                "--cross-intent-count",
                type=int,
                default=DEFAULT_ABO_CROSS_INTENT_COUNT,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        common = {
            "raw_root": args.raw_root,
            "source_lock_path": args.source_lock,
            "expected_source_lock_sha256": args.source_lock_sha256,
            "source_policy_path": args.source_policy,
            "dev_mini_selection_manifest": args.dev_mini_selection_manifest,
            "output_root": args.output_root,
            "include_count": args.include_count,
            "reserve_count": args.reserve_count,
            "max_items": args.max_items,
        }
        if args.command == "abo":
            result = build_abo_portfolio_core_candidates(
                **common,
                cross_intent_count=args.cross_intent_count,
            )
        else:
            result = build_rpc_portfolio_core_candidates(**common)
    except (
        FileExistsError,
        OSError,
        PortfolioCoreAssetError,
        PortfolioCoreSourceAdapterError,
        SourceLockError,
        tarfile.TarError,
        rpc_source.RPCAdapterError,
    ) as error:
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
            result.as_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ABO_EXPECTED_SOURCE_LOCK_SHA256",
    "ABO_SOURCE_LOCK_PATH",
    "ADAPTER_POLICY_VERSION",
    "DEFAULT_ABO_CROSS_INTENT_COUNT",
    "DEFAULT_ABO_INCLUDE_COUNT",
    "DEFAULT_ABO_RESERVE_COUNT",
    "DEFAULT_RESERVE_COUNT",
    "DEFAULT_RPC_INCLUDE_COUNT",
    "PortfolioCoreSourceAdapterError",
    "PortfolioCoreSourceAdapterResult",
    "RPC_EXPECTED_SOURCE_LOCK_SHA256",
    "RPC_SOURCE_LOCK_PATH",
    "build_abo_portfolio_core_candidates",
    "build_rpc_portfolio_core_candidates",
    "load_dev_mini_exclusions",
    "main",
]
