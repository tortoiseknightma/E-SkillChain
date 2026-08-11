"""Strict, offline Amazon Berkeley Objects adapter for Exact Match.

The official ABO release is large and its archive layout is intentionally not
guessed here.  Formal callers must first produce two *canonical local
snapshots*:

* listings JSONL with ``item_id``, ``main_image_id`` and ``other_image_id``;
* image-manifest JSONL with the official image id, archive-relative path,
  original-byte SHA-256, source collection kind, and derivation kind.

Those bytes, both conflicting licence statements, and an immutable archive
identity are pinned by an externally supplied :class:`ABOSourceLock`.  A lock
object created in-process is never accepted by a formal entry point; the lock
must be loaded from canonical bytes under an independently held expected
digest.  This module performs no network access and never downloads the full
ABO release.

ABO item identity is useful for Exact Match only when two different original
catalog images of the same ``item_id`` survive all final-byte leakage gates.
Spin, render, 3D, derived, unknown, or auxiliary assets are rejected.  Because
metadata alone cannot reliably distinguish a product photograph from an
auxiliary graphic, every selected image also requires a digest-bound human
review decision.
"""

from __future__ import annotations

import hashlib
import csv
import gzip
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from itertools import combinations
from typing import Annotated, Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import (
    DEFAULT_NEAR_DUPLICATE_POLICY,
    AssetCatalog,
    DatasetAssetDraft,
    NearDuplicatePolicy,
    QueryAssetReference,
    audit_query_gallery_eligibility,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.source_lock import (
    SourceLockError,
    VerifiedRequiredSourceLock,
    load_and_verify_required_source_lock,
)
from skillchain.data.source_review import (
    PermissionName,
    Purpose,
    SourceReviewError,
    SourceReviewRecord,
    load_source_review_policy,
    load_verified_source_review_ledger,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SafeIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$"),
]

SOURCE_DATASET = "abo"
SOURCE_SNAPSHOT_FORMAT = "abo-normalized-snapshot-v1"
SELECTION_POLICY_VERSION = "abo-exact-mini-selection-v1"
TRANSFORM_POLICY_VERSION = "abo-original-byte-copy-v1"
_VERIFIED_HANDLE_TOKEN = object()
REVIEW_POLICY_VERSION = "abo-catalog-photo-human-review-v1"
LICENSE_ID = "CC-BY-NC-4.0-CONFLICT-REVIEW"
ABO_FORMAL_PURPOSES: tuple[Purpose, ...] = (
    "capability_gold",
    "product_gallery",
    "tool_gold",
)
ABO_FORMAL_PERMISSIONS: tuple[PermissionName, ...] = (
    "local_embedding_allowed",
    "local_research_allowed",
    "public_demo_allowed",
    "remote_embedding_allowed",
)

_PROVENANCE_FILE = "abo-provenance.jsonl"
_DRAFT_FILE = "dataset-assets.jsonl"
_PAIR_FILE = "pair-candidates.jsonl"
_COVERAGE_FILE = "coverage-audit.json"
_REVIEW_PACKET_FILE = "review-packet.jsonl"
_BUNDLE_FILE = "candidate-bundle-manifest.json"
_PROVISIONAL_CATALOG_DIR = "provisional-catalog"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_PROHIBITED_PATH_PARTS = {
    "3d",
    "derived",
    "derivatives",
    "render",
    "renders",
    "spin",
    "spins",
}


class ABOProvenanceError(ValueError):
    """ABO provenance, identity, review, or immutable bytes are invalid."""


class ABOExactCoverageError(ABOProvenanceError):
    """The candidate snapshot cannot meet a preregistered Exact minimum."""

    def __init__(self, audit: "ABOExactCoverageAudit") -> None:
        super().__init__(
            "ABO provisional Exact candidate coverage is insufficient: "
            f"pairs={audit.provisional_pair_item_count} "
            f"minimum={audit.minimum_pair_candidates}"
        )
        self.audit = audit


class ABOGlobalFinalizationError(ABOProvenanceError):
    """A provisional bundle cannot satisfy the cross-source global gate."""

    def __init__(self, manifest: "ABOGlobalFinalizationManifest") -> None:
        super().__init__(
            "ABO global Exact finalization is insufficient: "
            f"finalized={manifest.finalized_pair_count} "
            f"minimum={manifest.minimum_finalized_pairs}"
        )
        self.manifest = manifest


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


class ABOPermissionMatrix(_StrictFrozenModel):
    """Conservative permissions while the official licence conflict is open."""

    local_noncommercial_research_allowed: Literal[True] = True
    local_noncommercial_embedding_allowed: Literal[True] = True
    remote_embedding_allowed: Literal[True] = True
    cloud_upload_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[True] = True


class ABORawArtifactReceipt(_StrictFrozenModel):
    """One exact official raw artifact retained in the local acquisition cache."""

    artifact_kind: Literal["listing_shard", "image_metadata"]
    official_uri: str
    etag: str
    raw_sha256: Sha256
    cache_relative_path: str
    compression: Literal["gzip"] = "gzip"

    @field_validator("etag")
    @classmethod
    def validate_etag(cls, value: str) -> str:
        return _nonblank(value, "etag")

    @field_validator("cache_relative_path")
    @classmethod
    def validate_cache_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "cache_relative_path")

    @model_validator(mode="after")
    def validate_official_artifact_uri(self):
        _validate_official_raw_artifact_uri(self.official_uri, self.artifact_kind)
        return self


class ABORawRecordLocator(_StrictFrozenModel):
    artifact_uri: str
    decompressed_line_number: int = Field(gt=0)
    record_bytes_sha256: Sha256


class ABOOfficialImageReceipt(_StrictFrozenModel):
    item_id: SafeIdentifier
    image_id: SafeIdentifier
    image_role: Literal["main", "other"]
    metadata_locator: ABORawRecordLocator
    metadata_path: str
    official_image_uri: str
    image_etag: str
    source_image_sha256: Sha256
    cache_relative_path: str

    @field_validator("metadata_path")
    @classmethod
    def validate_metadata_path(cls, value: str) -> str:
        return _canonical_metadata_image_path(value)

    @field_validator("cache_relative_path")
    @classmethod
    def validate_image_cache_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "cache_relative_path")

    @field_validator("image_etag")
    @classmethod
    def validate_image_etag(cls, value: str) -> str:
        return _nonblank(value, "image_etag")

    @model_validator(mode="after")
    def validate_original_image_identity(self):
        expected_cache = f"images/original/{self.metadata_path}"
        if self.cache_relative_path != expected_cache:
            raise ValueError(
                "original image cache path must equal images/original/{metadata_path}"
            )
        _validate_official_original_image_uri(
            self.official_image_uri, self.metadata_path
        )
        return self


class ABOItemAcquisitionReceipt(_StrictFrozenModel):
    item_id: SafeIdentifier
    listing_locator: ABORawRecordLocator
    images: tuple[ABOOfficialImageReceipt, ...]

    @field_validator("images", mode="before")
    @classmethod
    def accept_json_images(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_item_images(self):
        if len(self.images) < 2:
            raise ValueError("receipt item requires main plus at least one other image")
        if any(image.item_id != self.item_id for image in self.images):
            raise ValueError("receipt image item_id must match its parent item")
        main = [image for image in self.images if image.image_role == "main"]
        if len(main) != 1:
            raise ValueError("receipt item must contain exactly one main image")
        image_ids = [image.image_id for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("receipt image ids must be unique within an item")
        expected = tuple(
            sorted(
                self.images,
                key=lambda image: (image.image_role != "main", image.image_id),
            )
        )
        if self.images != expected:
            raise ValueError("receipt images must be canonical main-first order")
        return self


class ABOAcquisitionReceipt(_StrictFrozenModel):
    """Raw-to-normalized chain of custody for a targeted ABO mini snapshot."""

    schema_version: Literal[1] = 1
    receipt_policy_version: Literal["abo-targeted-acquisition-receipt-v1"] = (
        "abo-targeted-acquisition-receipt-v1"
    )
    source_revision: str
    archive_identity: str
    listing_artifact: ABORawArtifactReceipt
    image_metadata_artifact: ABORawArtifactReceipt
    items: tuple[ABOItemAcquisitionReceipt, ...]

    @field_validator("items", mode="before")
    @classmethod
    def accept_json_items(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_receipt(self):
        if self.listing_artifact.artifact_kind != "listing_shard":
            raise ValueError("listing_artifact kind is invalid")
        if self.image_metadata_artifact.artifact_kind != "image_metadata":
            raise ValueError("image_metadata_artifact kind is invalid")
        if not self.items:
            raise ValueError("acquisition receipt must contain at least one item")
        item_ids = [item.item_id for item in self.items]
        if item_ids != sorted(set(item_ids)):
            raise ValueError("receipt items must be sorted and unique")
        listing_uri = self.listing_artifact.official_uri
        metadata_uri = self.image_metadata_artifact.official_uri
        if any(item.listing_locator.artifact_uri != listing_uri for item in self.items):
            raise ValueError("listing locators must bind the locked listing shard")
        if any(
            image.metadata_locator.artifact_uri != metadata_uri
            for item in self.items
            for image in item.images
        ):
            raise ValueError("image locators must bind the locked images.csv.gz")
        return self


class ABOSourceLock(_StrictFrozenModel):
    """External trust record binding raw artifacts, receipt, and normalized views."""

    schema_version: Literal[1] = 1
    source_dataset: Literal["abo"] = SOURCE_DATASET
    snapshot_format: Literal["abo-normalized-snapshot-v1"] = SOURCE_SNAPSHOT_FORMAT
    source_revision: str
    archive_identity: str
    acquisition_receipt_sha256: Sha256
    listing_artifact: ABORawArtifactReceipt
    image_metadata_artifact: ABORawArtifactReceipt
    listings_sha256: Sha256
    image_manifest_sha256: Sha256
    direct_license_sha256: Sha256
    registry_license_sha256: Sha256
    direct_license_id: Literal["CC-BY-4.0"]
    registry_license_id: Literal["CC-BY-NC-4.0"]
    license_id: Literal["CC-BY-NC-4.0-CONFLICT-REVIEW"] = LICENSE_ID
    license_review_status: Literal["unresolved_conservative_by_nc"] = (
        "unresolved_conservative_by_nc"
    )
    dataset_uri: str
    archive_uri: str
    direct_license_uri: str
    registry_license_uri: str
    attribution: str
    permissions: ABOPermissionMatrix

    @field_validator("source_revision", "archive_identity")
    @classmethod
    def validate_immutable_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        mutable = {"main", "master", "head", "latest", "current", "unknown"}
        normalized = value.casefold()
        tokens = {token for token in re.split(r"[/:._-]+", normalized) if token}
        if tokens.intersection(mutable):
            raise ValueError(f"{info.field_name} must identify an immutable snapshot")
        immutable_marker = re.search(
            r"(?:[0-9a-f]{7,64}|\d{4}[-_.]?\d{2}|\bv?\d+(?:\.\d+)+|etag)",
            normalized,
        )
        if immutable_marker is None:
            raise ValueError(
                f"{info.field_name} must contain a commit, date, version, or ETag"
            )
        return value

    @field_validator("attribution")
    @classmethod
    def validate_attribution(cls, value: str) -> str:
        return _nonblank(value, "attribution")

    @field_validator("dataset_uri", "archive_uri")
    @classmethod
    def validate_dataset_uri(cls, value: str) -> str:
        return _validate_official_dataset_uri(value)

    @field_validator("direct_license_uri")
    @classmethod
    def validate_direct_license_uri(cls, value: str) -> str:
        return _validate_official_license_uri(value)

    @field_validator("registry_license_uri")
    @classmethod
    def validate_registry_uri(cls, value: str) -> str:
        return _validate_registry_uri(value)

    @model_validator(mode="after")
    def validate_raw_artifact_kinds(self):
        if self.listing_artifact.artifact_kind != "listing_shard":
            raise ValueError("source lock listing artifact kind is invalid")
        if self.image_metadata_artifact.artifact_kind != "image_metadata":
            raise ValueError("source lock image metadata artifact kind is invalid")
        return self


class ABOListingRecord(_StrictFrozenModel):
    """Normalized subset of one official ABO listing record."""

    schema_version: Literal[1] = 1
    item_id: SafeIdentifier
    main_image_id: SafeIdentifier
    other_image_id: tuple[SafeIdentifier, ...]

    @field_validator("other_image_id", mode="before")
    @classmethod
    def accept_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_distinct_images(self):
        if len(self.other_image_id) != len(set(self.other_image_id)):
            raise ValueError("other_image_id must not contain duplicates")
        if self.main_image_id in self.other_image_id:
            raise ValueError("main_image_id must not be repeated in other_image_id")
        if tuple(sorted(self.other_image_id)) != self.other_image_id:
            raise ValueError("other_image_id must be sorted")
        return self


class ABOImageManifestRecord(_StrictFrozenModel):
    """Normalized view that must be re-derived from raw receipt evidence."""

    schema_version: Literal[1] = 1
    image_id: SafeIdentifier
    path: str
    source_collection: Literal["official_original"] = "official_original"
    derivation_kind: Literal["original"] = "original"
    official_image_uri: str
    official_image_etag: str
    source_image_sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_relative_image_path(cls, value: str) -> str:
        return _canonical_metadata_image_path(value)

    @field_validator("official_image_etag")
    @classmethod
    def validate_etag(cls, value: str) -> str:
        return _nonblank(value, "official_image_etag")

    @model_validator(mode="after")
    def validate_exact_original_uri(self):
        _validate_official_original_image_uri(self.official_image_uri, self.path)
        return self


ABOReviewDecision = Literal[
    "approve_catalog_product_photo",
    "reject_auxiliary_graphic",
    "reject_non_product",
    "reject_uncertain",
]


class ABOImageReviewDecision(_StrictFrozenModel):
    """Human decision that metadata cannot manufacture by itself."""

    schema_version: Literal[1] = 1
    review_policy_version: Literal["abo-catalog-photo-human-review-v1"] = (
        REVIEW_POLICY_VERSION
    )
    item_id: SafeIdentifier
    image_id: SafeIdentifier
    listing_record_sha256: Sha256
    image_record_sha256: Sha256
    source_image_sha256: Sha256
    decision: ABOReviewDecision
    reviewer_kind: Literal["human"]
    reviewer_id: str
    reviewed_at: str

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        value = _nonblank(value, "reviewer_id")
        if value.casefold().startswith(("llm", "gpt", "claude", "qwen", "model")):
            raise ValueError("reviewer_id must identify the accountable human reviewer")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_review_time(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
            raise ValueError("reviewed_at must be a second-precision UTC timestamp")
        return value


class ABOAssetProvenance(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    source_dataset: Literal["abo"] = SOURCE_DATASET
    source_revision: str
    archive_identity: str
    item_id: SafeIdentifier
    product_id: str
    image_id: SafeIdentifier
    image_role: Literal["main", "other"]
    source_record_id: str
    listing_record_sha256: Sha256
    image_record_sha256: Sha256
    acquisition_receipt_sha256: Sha256
    listing_artifact_sha256: Sha256
    listing_artifact_etag: str
    listing_line_number: int = Field(gt=0)
    raw_listing_record_sha256: Sha256
    image_metadata_artifact_sha256: Sha256
    image_metadata_artifact_etag: str
    image_metadata_line_number: int = Field(gt=0)
    raw_image_metadata_record_sha256: Sha256
    source_image_sha256: Sha256
    final_image_sha256: Sha256
    source_collection: Literal["official_original"]
    derivation_kind: Literal["original"]
    transform_policy_version: Literal["abo-original-byte-copy-v1"] = (
        TRANSFORM_POLICY_VERSION
    )
    derivation_parent_asset_ids: tuple[str, ...] = ()
    source_lock_sha256: Sha256
    review_ledger_sha256: Sha256
    review_decision: Literal["approve_catalog_product_photo"]
    reviewer_kind: Literal["human"]
    reviewer_id: str
    reviewed_at: str
    license_id: Literal["CC-BY-NC-4.0-CONFLICT-REVIEW"] = LICENSE_ID
    direct_license_sha256: Sha256
    registry_license_sha256: Sha256
    official_image_uri: str
    official_image_etag: str
    attribution: str
    permissions: ABOPermissionMatrix

    @field_validator("derivation_parent_asset_ids", mode="before")
    @classmethod
    def accept_json_parent_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("derivation_parent_asset_ids")
    @classmethod
    def reject_invented_catalog_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("external ABO records are not catalog derivation parents")
        return value


class ABOExactPair(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    eligibility_status: Literal["provisional_requires_cross_source_global_catalog"] = (
        "provisional_requires_cross_source_global_catalog"
    )
    item_id: SafeIdentifier
    product_id: str
    query_image_id: SafeIdentifier
    query_asset_id: str
    query_local_path: str
    gallery_image_id: SafeIdentifier
    gallery_asset_id: str
    gallery_local_path: str

    @model_validator(mode="after")
    def validate_pair_identity(self):
        if self.product_id != _product_id(self.item_id):
            raise ValueError("Exact pair product_id must map directly from item_id")
        if self.query_image_id == self.gallery_image_id:
            raise ValueError("Exact pair requires two different source image ids")
        if self.query_asset_id == self.gallery_asset_id:
            raise ValueError("Exact pair requires two different catalog assets")
        if self.query_local_path == self.gallery_local_path:
            raise ValueError("Exact pair requires two different final files")
        return self


ABOAssetDispositionReason = Literal[
    "provisional_query_candidate",
    "provisional_gallery_candidate",
    "not_selected_view",
    "cross_item_image_identity",
    "missing_image_record",
    "missing_image_file",
    "non_original_source",
    "raw_receipt_mismatch",
    "invalid_image",
    "missing_human_approval",
    "review_rejected",
    "local_near_duplicate",
    "local_cross_pair_duplicate",
    "pair_limit",
]


class ABOAssetDisposition(_StrictFrozenModel):
    asset_key: str
    item_id: SafeIdentifier
    image_id: SafeIdentifier
    disposition: Literal["provisional_candidate", "excluded"]
    reason: ABOAssetDispositionReason

    @model_validator(mode="after")
    def validate_status_reason(self):
        candidate_reasons = {
            "provisional_query_candidate",
            "provisional_gallery_candidate",
        }
        if (self.reason in candidate_reasons) != (
            self.disposition == "provisional_candidate"
        ):
            raise ValueError("asset disposition and reason are inconsistent")
        if self.asset_key != f"{self.item_id}/{self.image_id}":
            raise ValueError("asset_key must be the canonical item/image identity")
        return self


class ABOExactCoverageAudit(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    selection_policy_version: Literal["abo-exact-mini-selection-v1"] = (
        SELECTION_POLICY_VERSION
    )
    eligibility_scope: Literal["local_provisional_only"] = "local_provisional_only"
    global_eligibility_status: Literal[
        "not_evaluated_requires_cross_source_catalog"
    ] = "not_evaluated_requires_cross_source_catalog"
    candidate_item_count: int = Field(ge=0)
    candidate_asset_count: int = Field(ge=0)
    provisional_pair_item_count: int = Field(ge=0)
    provisional_pair_asset_count: int = Field(ge=0)
    excluded_item_count: int = Field(ge=0)
    excluded_asset_count: int = Field(ge=0)
    minimum_pair_candidates: int = Field(gt=0)
    maximum_pair_candidates: int = Field(gt=0)
    excluded_items_by_reason: dict[str, tuple[str, ...]]
    asset_dispositions: tuple[ABOAssetDisposition, ...]
    passed_provisional_gate: bool

    @field_validator("asset_dispositions", mode="before")
    @classmethod
    def accept_json_dispositions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("excluded_items_by_reason", mode="before")
    @classmethod
    def accept_json_item_reasons(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: tuple(items) if isinstance(items, list) else items
                for key, items in value.items()
            }
        return value

    @model_validator(mode="after")
    def validate_coverage_arithmetic(self):
        if (
            self.provisional_pair_item_count + self.excluded_item_count
            != self.candidate_item_count
        ):
            raise ValueError(
                "candidate items must partition into eligible and excluded"
            )
        if self.provisional_pair_asset_count != 2 * self.provisional_pair_item_count:
            raise ValueError("each provisional pair must contribute exactly two assets")
        if self.provisional_pair_item_count > self.maximum_pair_candidates:
            raise ValueError("provisional pair count exceeds maximum_pair_candidates")
        if self.passed_provisional_gate != (
            self.provisional_pair_item_count >= self.minimum_pair_candidates
        ):
            raise ValueError("provisional pass does not match preregistered minimum")
        excluded_items = {
            value
            for values in self.excluded_items_by_reason.values()
            for value in values
        }
        if len(excluded_items) != self.excluded_item_count:
            raise ValueError("excluded item reasons do not cover every excluded item")
        disposition_keys = [value.asset_key for value in self.asset_dispositions]
        if disposition_keys != sorted(set(disposition_keys)):
            raise ValueError("asset dispositions must be sorted, unique, and complete")
        if len(disposition_keys) != self.candidate_asset_count:
            raise ValueError("every candidate asset requires exactly one disposition")
        excluded_assets = sum(
            value.disposition == "excluded" for value in self.asset_dispositions
        )
        if excluded_assets != self.excluded_asset_count:
            raise ValueError("excluded disposition count is inconsistent")
        provisional_assets = sum(
            value.disposition == "provisional_candidate"
            for value in self.asset_dispositions
        )
        if provisional_assets != self.provisional_pair_asset_count:
            raise ValueError("provisional disposition count is inconsistent")
        return self


class ABOBundleFile(_StrictFrozenModel):
    path: str
    bytes: int = Field(ge=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "bundle file path")


class ABOCandidateBundleManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    bundle_policy_version: Literal["abo-exact-candidate-bundle-v2"] = (
        "abo-exact-candidate-bundle-v2"
    )
    status: Literal["provisional_requires_cross_source_global_catalog"] = (
        "provisional_requires_cross_source_global_catalog"
    )
    selection_policy_version: Literal["abo-exact-mini-selection-v1"] = (
        SELECTION_POLICY_VERSION
    )
    required_source_lock_sha256: Sha256
    source_review_policy_sha256: Sha256
    source_review_ledger_sha256: Sha256
    source_review_record_sha256: Sha256
    source_review_license_evidence_sha256: Sha256
    source_lock_sha256: Sha256
    acquisition_receipt_sha256: Sha256
    listings_sha256: Sha256
    image_manifest_sha256: Sha256
    direct_license_sha256: Sha256
    registry_license_sha256: Sha256
    review_ledger_sha256: Sha256
    dataset_assets_sha256: Sha256
    provenance_sha256: Sha256
    exact_pairs_sha256: Sha256
    coverage_audit_sha256: Sha256
    review_packet_sha256: Sha256
    provisional_asset_catalog_sha256: Sha256
    permissions: ABOPermissionMatrix
    provisional_pair_count: int = Field(gt=0)
    provisional_asset_count: int = Field(gt=0)
    files: tuple[ABOBundleFile, ...]
    bundle_self_sha256: Sha256

    @field_validator("files", mode="before")
    @classmethod
    def accept_json_files(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_bundle(self):
        if self.provisional_asset_count != 2 * self.provisional_pair_count:
            raise ValueError("candidate bundle requires two assets per pair")
        paths = [value.path for value in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("bundle file descriptors must be sorted and unique")
        if _BUNDLE_FILE in paths:
            raise ValueError(
                "bundle manifest must not describe itself as a payload file"
            )
        return self


class ABOReviewPacketEntry(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    packet_status: Literal["human_decision_required"] = "human_decision_required"
    item_id: SafeIdentifier
    image_id: SafeIdentifier
    image_role: Literal["main", "other"]
    local_path: str
    official_image_uri: str
    official_image_etag: str
    source_image_sha256: Sha256
    listing_record_sha256: Sha256
    image_record_sha256: Sha256


class ABOGlobalCatalogScope(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    scope_kind: Literal["cross_source_global_evaluation_catalog"] = (
        "cross_source_global_evaluation_catalog"
    )
    catalog_sha256: Sha256
    asset_count: int = Field(gt=0)
    coverage_roots: tuple[str, ...]
    source_datasets: tuple[str, ...]
    completeness_claim: Literal["all_assets_available_to_the_registered_evaluation"] = (
        "all_assets_available_to_the_registered_evaluation"
    )

    @field_validator("coverage_roots", "source_datasets", mode="before")
    @classmethod
    def accept_json_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_cross_source_scope(self):
        if self.source_datasets != tuple(sorted(set(self.source_datasets))):
            raise ValueError("global scope source_datasets must be sorted and unique")
        if SOURCE_DATASET not in self.source_datasets or len(self.source_datasets) < 2:
            raise ValueError("global catalog scope must cover ABO and another source")
        if self.coverage_roots != tuple(sorted(set(self.coverage_roots))):
            raise ValueError("global scope coverage_roots must be sorted and unique")
        return self


class ABOGlobalPairDisposition(_StrictFrozenModel):
    item_id: SafeIdentifier
    disposition: Literal["finalized", "excluded"]
    reason: Literal["globally_clean", "global_forbidden_leakage"]

    @model_validator(mode="after")
    def validate_disposition(self):
        if (self.disposition == "finalized") != (self.reason == "globally_clean"):
            raise ValueError("global pair disposition is inconsistent")
        return self


class ABOGlobalFinalizationManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    finalization_policy_version: Literal["abo-cross-source-global-finalization-v1"] = (
        "abo-cross-source-global-finalization-v1"
    )
    candidate_bundle_sha256: Sha256
    global_catalog_sha256: Sha256
    global_scope_sha256: Sha256
    finalized_pair_count: int = Field(ge=0)
    minimum_finalized_pairs: int = Field(gt=0)
    pair_dispositions: tuple[ABOGlobalPairDisposition, ...]
    finalization_self_sha256: Sha256

    @field_validator("pair_dispositions", mode="before")
    @classmethod
    def accept_json_dispositions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_finalization(self):
        ids = [value.item_id for value in self.pair_dispositions]
        if ids != sorted(set(ids)):
            raise ValueError("global pair dispositions must be sorted and unique")
        finalized = sum(
            value.disposition == "finalized" for value in self.pair_dispositions
        )
        if finalized != self.finalized_pair_count:
            raise ValueError("finalized pair count is inconsistent")
        return self


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, int, int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int, int, int]], ...]


@dataclass(frozen=True)
class VerifiedABOSourceBundle:
    lock: ABOSourceLock
    lock_sha256: str
    lock_snapshot: _FileSnapshot
    receipt: ABOAcquisitionReceipt
    receipt_sha256: str
    receipt_snapshot: _FileSnapshot
    listing_raw_snapshot: _FileSnapshot
    image_metadata_raw_snapshot: _FileSnapshot
    listings_snapshot: _FileSnapshot
    image_manifest_snapshot: _FileSnapshot
    direct_license_snapshot: _FileSnapshot
    registry_license_snapshot: _FileSnapshot
    listings: tuple[ABOListingRecord, ...]
    images: tuple[ABOImageManifestRecord, ...]
    receipt_images: dict[tuple[str, str], ABOOfficialImageReceipt]
    raw_cache_root: Path
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedABOSourceApproval:
    """Owner approval bound to a verified required lock and evidence bytes."""

    required_source_lock: VerifiedRequiredSourceLock = field(repr=False, compare=False)
    review_record: SourceReviewRecord
    policy_file_sha256: str
    ledger_file_sha256: str
    license_evidence_sha256: str
    purposes: tuple[Purpose, ...]
    permissions: tuple[PermissionName, ...]
    raw_root: Path = field(repr=False, compare=False)
    required_lock_snapshot: _FileSnapshot = field(repr=False, compare=False)
    policy_snapshot: _FileSnapshot = field(repr=False, compare=False)
    ledger_snapshot: _FileSnapshot = field(repr=False, compare=False)
    license_evidence_snapshot: _FileSnapshot = field(repr=False, compare=False)
    expected_portfolio_sha256: str = field(repr=False, compare=False)
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _PreparedAsset:
    item: ABOListingRecord
    image: ABOImageManifestRecord
    role: Literal["main", "other"]
    review: ABOImageReviewDecision
    snapshot: _FileSnapshot
    listing_record_sha256: str
    image_record_sha256: str
    receipt_image: ABOOfficialImageReceipt
    local_path: str


@dataclass(frozen=True)
class _PreparedPair:
    item_id: str
    main: _PreparedAsset
    other: _PreparedAsset


@dataclass(frozen=True)
class ABOBuildResult:
    output_dir: Path
    drafts: tuple[DatasetAssetDraft, ...]
    pairs: tuple[ABOExactPair, ...]
    audit: ABOExactCoverageAudit
    manifest: ABOCandidateBundleManifest
    bundle_manifest_sha256: str


@dataclass(frozen=True)
class _ParsedABOCandidateBundle:
    root: Path
    manifest: ABOCandidateBundleManifest
    manifest_sha256: str
    drafts: tuple[DatasetAssetDraft, ...]
    pairs: tuple[ABOExactPair, ...]
    audit: ABOExactCoverageAudit
    provenance: tuple[ABOAssetProvenance, ...]
    review_packet: tuple[ABOReviewPacketEntry, ...]
    catalog: AssetCatalog


@dataclass(frozen=True)
class VerifiedABOCandidateBundle(_ParsedABOCandidateBundle):
    """Candidate bytes re-derived from their independently verified sources."""

    source_bundle: VerifiedABOSourceBundle = field(repr=False, compare=False)
    source_approval: VerifiedABOSourceApproval = field(repr=False, compare=False)
    review_snapshot: _FileSnapshot = field(repr=False, compare=False)
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedABOGlobalFinalization:
    """A final pair selection re-derived under an external file digest."""

    path: Path
    file_sha256: str
    manifest: ABOGlobalFinalizationManifest
    candidate_bundle: VerifiedABOCandidateBundle = field(repr=False, compare=False)
    global_catalog: AssetCatalog = field(repr=False, compare=False)


@dataclass(frozen=True)
class _CoverageAssessment:
    bundle: VerifiedABOSourceBundle
    review_snapshot: _FileSnapshot
    image_snapshots: tuple[_FileSnapshot, ...]
    locally_viable_pairs: tuple[_PreparedPair, ...]
    selected_pairs: tuple[_PreparedPair, ...]
    audit: ABOExactCoverageAudit


TModel = TypeVar("TModel", bound=BaseModel)


def load_verified_abo_source_bundle(
    *,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    listings_path: Path | str,
    image_manifest_path: Path | str,
    direct_license_path: Path | str,
    registry_license_path: Path | str,
) -> VerifiedABOSourceBundle:
    """Re-derive normalized identity from raw official cache and receipt bytes."""

    _validate_expected_digest(expected_source_lock_sha256, "source-lock")
    lock_snapshot = _snapshot_regular_file(Path(source_lock_path), "ABO source lock")
    if lock_snapshot.sha256 != expected_source_lock_sha256:
        raise ABOProvenanceError(
            "ABO source lock does not match the independent expected digest"
        )
    lock = _load_canonical_json_model(lock_snapshot, ABOSourceLock, "ABO source lock")
    _validate_expected_digest(
        expected_acquisition_receipt_sha256, "acquisition-receipt"
    )
    receipt_snapshot = _snapshot_regular_file(
        Path(acquisition_receipt_path), "ABO acquisition receipt"
    )
    if receipt_snapshot.sha256 != expected_acquisition_receipt_sha256:
        raise ABOProvenanceError(
            "ABO acquisition receipt does not match the independent expected digest"
        )
    if receipt_snapshot.sha256 != lock.acquisition_receipt_sha256:
        raise ABOProvenanceError("ABO acquisition receipt does not match source lock")
    receipt = _load_canonical_json_model(
        receipt_snapshot, ABOAcquisitionReceipt, "ABO acquisition receipt"
    )
    if (
        receipt.source_revision != lock.source_revision
        or receipt.archive_identity != lock.archive_identity
        or receipt.listing_artifact != lock.listing_artifact
        or receipt.image_metadata_artifact != lock.image_metadata_artifact
    ):
        raise ABOProvenanceError(
            "ABO receipt raw identity does not match the external source lock"
        )
    raw_cache_root = Path(raw_cache_root).absolute()
    _require_real_directory(raw_cache_root, "ABO raw cache root")
    listing_raw_snapshot = _snapshot_regular_file(
        _cache_file(raw_cache_root, receipt.listing_artifact.cache_relative_path),
        "ABO raw listing shard",
    )
    image_metadata_raw_snapshot = _snapshot_regular_file(
        _cache_file(
            raw_cache_root, receipt.image_metadata_artifact.cache_relative_path
        ),
        "ABO raw images.csv.gz",
    )
    if listing_raw_snapshot.sha256 != receipt.listing_artifact.raw_sha256:
        raise ABOProvenanceError(
            "ABO raw listing shard drifted from acquisition receipt"
        )
    if image_metadata_raw_snapshot.sha256 != receipt.image_metadata_artifact.raw_sha256:
        raise ABOProvenanceError(
            "ABO raw images.csv.gz drifted from acquisition receipt"
        )
    listings_snapshot = _snapshot_regular_file(
        Path(listings_path), "ABO listings snapshot"
    )
    image_manifest_snapshot = _snapshot_regular_file(
        Path(image_manifest_path), "ABO image manifest"
    )
    direct_license_snapshot = _snapshot_regular_file(
        Path(direct_license_path), "ABO direct licence evidence"
    )
    registry_license_snapshot = _snapshot_regular_file(
        Path(registry_license_path), "ABO registry licence evidence"
    )
    expected = {
        "listings": (listings_snapshot.sha256, lock.listings_sha256),
        "image manifest": (
            image_manifest_snapshot.sha256,
            lock.image_manifest_sha256,
        ),
        "direct licence": (
            direct_license_snapshot.sha256,
            lock.direct_license_sha256,
        ),
        "registry licence": (
            registry_license_snapshot.sha256,
            lock.registry_license_sha256,
        ),
    }
    for label, (actual, locked) in expected.items():
        if actual != locked:
            raise ABOProvenanceError(f"ABO {label} bytes do not match the source lock")
    _verify_license_evidence(
        direct_license_snapshot.content,
        required_tokens="cc by 4 0",
        label="ABO direct licence evidence",
    )
    _verify_license_evidence(
        registry_license_snapshot.content,
        required_tokens="cc by nc 4 0",
        label="ABO registry licence evidence",
    )

    listings = _load_canonical_jsonl(
        listings_snapshot, ABOListingRecord, "ABO listings snapshot"
    )
    images = _load_canonical_jsonl(
        image_manifest_snapshot, ABOImageManifestRecord, "ABO image manifest"
    )
    _require_unique((row.item_id for row in listings), "ABO listing item_id")
    _require_unique((row.image_id for row in images), "ABO image_id")
    _require_unique(
        (row.path.casefold() for row in images),
        "ABO image path (case-insensitive)",
    )
    receipt_images = _cross_validate_raw_receipt(
        receipt=receipt,
        listings=listings,
        images=images,
        listing_raw_snapshot=listing_raw_snapshot,
        image_metadata_raw_snapshot=image_metadata_raw_snapshot,
    )
    for snapshot, label in (
        (lock_snapshot, "ABO source lock"),
        (receipt_snapshot, "ABO acquisition receipt"),
        (listing_raw_snapshot, "ABO raw listing shard"),
        (image_metadata_raw_snapshot, "ABO raw images.csv.gz"),
        (listings_snapshot, "ABO listings snapshot"),
        (image_manifest_snapshot, "ABO image manifest"),
        (direct_license_snapshot, "ABO direct licence evidence"),
        (registry_license_snapshot, "ABO registry licence evidence"),
    ):
        _verify_snapshot(snapshot, label)
    return VerifiedABOSourceBundle(
        lock=lock,
        lock_sha256=lock_snapshot.sha256,
        lock_snapshot=lock_snapshot,
        receipt=receipt,
        receipt_sha256=receipt_snapshot.sha256,
        receipt_snapshot=receipt_snapshot,
        listing_raw_snapshot=listing_raw_snapshot,
        image_metadata_raw_snapshot=image_metadata_raw_snapshot,
        listings_snapshot=listings_snapshot,
        image_manifest_snapshot=image_manifest_snapshot,
        direct_license_snapshot=direct_license_snapshot,
        registry_license_snapshot=registry_license_snapshot,
        listings=listings,
        images=images,
        receipt_images=receipt_images,
        raw_cache_root=raw_cache_root,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def load_verified_abo_source_approval(
    *,
    required_source_lock_path: Path | str,
    expected_required_source_lock_sha256: str,
    required_source_raw_root: Path | str,
    license_evidence_path: Path | str,
    source_review_policy_path: Path | str,
    expected_source_review_policy_sha256: str,
    expected_source_review_portfolio_sha256: str,
    source_review_ledger_path: Path | str,
    expected_source_review_ledger_sha256: str,
    purposes: tuple[Purpose, ...] = ABO_FORMAL_PURPOSES,
    permissions: tuple[PermissionName, ...] = ABO_FORMAL_PERMISSIONS,
) -> VerifiedABOSourceApproval:
    """Load the independent owner approval required by formal ABO adapters."""

    purposes = tuple(sorted(set(purposes)))
    permissions = tuple(sorted(set(permissions)))
    raw_root = Path(required_source_raw_root).absolute()
    _require_real_directory(raw_root, "ABO required-source RAW root")
    required_lock_snapshot = _snapshot_regular_file(
        Path(required_source_lock_path), "ABO required source lock"
    )
    policy_snapshot = _snapshot_regular_file(
        Path(source_review_policy_path), "ABO source-review policy"
    )
    ledger_snapshot = _snapshot_regular_file(
        Path(source_review_ledger_path), "ABO source-review ledger"
    )
    evidence_snapshot = _snapshot_regular_file(
        Path(license_evidence_path), "ABO source-review license evidence"
    )
    if required_lock_snapshot.sha256 != expected_required_source_lock_sha256:
        raise ABOProvenanceError(
            "ABO required source lock does not match the independent expected digest"
        )
    if policy_snapshot.sha256 != expected_source_review_policy_sha256:
        raise ABOProvenanceError(
            "ABO source-review policy does not match the independent expected digest"
        )
    if ledger_snapshot.sha256 != expected_source_review_ledger_sha256:
        raise ABOProvenanceError(
            "ABO source-review ledger does not match the independent expected digest"
        )

    evidence = _parse_abo_source_approval_evidence(evidence_snapshot)
    try:
        required_source = load_and_verify_required_source_lock(
            required_lock_snapshot.path,
            raw_root,
            expected_lock_file_sha256=expected_required_source_lock_sha256,
        )
        policy = load_source_review_policy(
            policy_snapshot.path,
            expected_policy_file_sha256=expected_source_review_policy_sha256,
            expected_portfolio_sha256=expected_source_review_portfolio_sha256,
        )
        ledger = load_verified_source_review_ledger(
            ledger_snapshot.path,
            policy,
            policy_file_sha256=expected_source_review_policy_sha256,
            expected_ledger_file_sha256=expected_source_review_ledger_sha256,
        )
        if required_source.lock.source_id != SOURCE_DATASET:
            raise ABOProvenanceError(
                "ABO source approval must bind an abo required source lock"
            )
        review_record = ledger.require_approval(
            SOURCE_DATASET,
            source_revision=required_source.lock.source_revision,
            source_lock_sha256=required_source.lock_file_sha256,
            license_id=evidence["license_id"],
            license_evidence_sha256=evidence_snapshot.sha256,
            purposes=purposes,
            permissions=permissions,
        )
    except (SourceLockError, SourceReviewError) as error:
        raise ABOProvenanceError("ABO owner source approval is invalid") from error

    for snapshot, label in (
        (required_lock_snapshot, "ABO required source lock"),
        (policy_snapshot, "ABO source-review policy"),
        (ledger_snapshot, "ABO source-review ledger"),
        (evidence_snapshot, "ABO source-review license evidence"),
    ):
        _verify_snapshot(snapshot, label)
    return VerifiedABOSourceApproval(
        required_source_lock=required_source,
        review_record=review_record,
        policy_file_sha256=policy_snapshot.sha256,
        ledger_file_sha256=ledger_snapshot.sha256,
        license_evidence_sha256=evidence_snapshot.sha256,
        purposes=purposes,
        permissions=permissions,
        raw_root=raw_root,
        required_lock_snapshot=required_lock_snapshot,
        policy_snapshot=policy_snapshot,
        ledger_snapshot=ledger_snapshot,
        license_evidence_snapshot=evidence_snapshot,
        expected_portfolio_sha256=expected_source_review_portfolio_sha256,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _parse_abo_source_approval_evidence(snapshot: _FileSnapshot) -> dict[str, Any]:
    try:
        value = json.loads(snapshot.content)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ABOProvenanceError(
            "ABO source-review license evidence is invalid JSON"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != snapshot.content:
        raise ABOProvenanceError(
            "ABO source-review license evidence must be canonical JSON"
        )
    if value.get("source_id") != SOURCE_DATASET:
        raise ABOProvenanceError(
            "ABO source-review license evidence names a different source"
        )
    license_id = value.get("license_id")
    if not isinstance(license_id, str) or not license_id.strip():
        raise ABOProvenanceError(
            "ABO source-review license evidence has no canonical license_id"
        )
    return value


def _refresh_verified_abo_source_approval(
    approval: VerifiedABOSourceApproval,
) -> VerifiedABOSourceApproval:
    if (
        not isinstance(approval, VerifiedABOSourceApproval)
        or approval._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise ABOProvenanceError("formal ABO use requires a VerifiedABOSourceApproval")
    for snapshot, label in (
        (approval.required_lock_snapshot, "ABO required source lock"),
        (approval.policy_snapshot, "ABO source-review policy"),
        (approval.ledger_snapshot, "ABO source-review ledger"),
        (
            approval.license_evidence_snapshot,
            "ABO source-review license evidence",
        ),
    ):
        _verify_snapshot(snapshot, label)
    refreshed = load_verified_abo_source_approval(
        required_source_lock_path=approval.required_lock_snapshot.path,
        expected_required_source_lock_sha256=(
            approval.required_source_lock.lock_file_sha256
        ),
        required_source_raw_root=approval.raw_root,
        license_evidence_path=approval.license_evidence_snapshot.path,
        source_review_policy_path=approval.policy_snapshot.path,
        expected_source_review_policy_sha256=approval.policy_file_sha256,
        expected_source_review_portfolio_sha256=approval.expected_portfolio_sha256,
        source_review_ledger_path=approval.ledger_snapshot.path,
        expected_source_review_ledger_sha256=approval.ledger_file_sha256,
        purposes=approval.purposes,
        permissions=approval.permissions,
    )
    if (
        refreshed.required_source_lock != approval.required_source_lock
        or refreshed.review_record != approval.review_record
    ):
        raise ABOProvenanceError("ABO source approval changed during processing")
    return refreshed


def _bind_abo_source_approval(
    approval: VerifiedABOSourceApproval,
    bundle: VerifiedABOSourceBundle,
    *,
    consumed_image_snapshots: tuple[_FileSnapshot, ...] = (),
) -> VerifiedABOSourceApproval:
    if (
        not isinstance(approval, VerifiedABOSourceApproval)
        or approval._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise ABOProvenanceError("formal ABO use requires a VerifiedABOSourceApproval")
    if approval.purposes != ABO_FORMAL_PURPOSES:
        raise ABOProvenanceError(
            "formal ABO Exact Match requires all registered source-review purposes"
        )
    if approval.permissions != ABO_FORMAL_PERMISSIONS:
        raise ABOProvenanceError(
            "formal ABO Exact Match requires local research and embedding approval"
        )
    record = approval.review_record
    if record.license_id != bundle.lock.registry_license_id:
        raise ABOProvenanceError(
            "ABO normalized lock license differs from owner-reviewed evidence"
        )
    permission_pairs = (
        (
            record.permissions.local_research_allowed,
            bundle.lock.permissions.local_noncommercial_research_allowed,
        ),
        (
            record.permissions.local_embedding_allowed,
            bundle.lock.permissions.local_noncommercial_embedding_allowed,
        ),
        (
            record.permissions.remote_embedding_allowed,
            bundle.lock.permissions.remote_embedding_allowed,
        ),
        (
            record.permissions.redistribution_allowed,
            bundle.lock.permissions.redistribution_allowed,
        ),
        (
            record.permissions.public_demo_allowed,
            bundle.lock.permissions.public_demo_allowed,
        ),
    )
    if any(reviewed != normalized for reviewed, normalized in permission_pairs):
        raise ABOProvenanceError(
            "ABO normalized lock permissions differ from owner-reviewed permissions"
        )
    _require_approval_lock_covers_consumed_raw(
        approval,
        (
            bundle.listing_raw_snapshot,
            bundle.image_metadata_raw_snapshot,
            *consumed_image_snapshots,
        ),
    )
    return approval


def _require_approval_lock_covers_consumed_raw(
    approval: VerifiedABOSourceApproval,
    snapshots: tuple[_FileSnapshot, ...],
) -> None:
    raw_root = approval.raw_root.resolve(strict=True)
    scopes = approval.required_source_lock.lock.artifact_scopes
    for snapshot in snapshots:
        try:
            relative = (
                snapshot.path.resolve(strict=True).relative_to(raw_root).as_posix()
            )
        except ValueError as error:
            raise ABOProvenanceError(
                "ABO consumed RAW file is outside the owner-approved lock root"
            ) from error
        covered = False
        for scope in scopes:
            if scope.mode == "explicit_files":
                covered = relative in scope.paths
            else:
                prefix = f"{scope.root}/"
                inside = relative.startswith(prefix)
                excluded = any(
                    relative == value or relative.startswith(f"{value}/")
                    for value in scope.exclude_prefixes
                )
                covered = inside and not excluded
            if covered:
                break
        if not covered:
            raise ABOProvenanceError(
                "ABO owner-approved required source lock does not cover consumed "
                f"RAW file: {relative}"
            )


def _cross_validate_raw_receipt(
    *,
    receipt: ABOAcquisitionReceipt,
    listings: tuple[ABOListingRecord, ...],
    images: tuple[ABOImageManifestRecord, ...],
    listing_raw_snapshot: _FileSnapshot,
    image_metadata_raw_snapshot: _FileSnapshot,
) -> dict[tuple[str, str], ABOOfficialImageReceipt]:
    """Prove every normalized relation against exact decompressed raw records."""

    listing_lines = _decompress_gzip_lines(
        listing_raw_snapshot.content, "ABO raw listing shard"
    )
    metadata_lines = _decompress_gzip_lines(
        image_metadata_raw_snapshot.content, "ABO raw images.csv.gz"
    )
    if not metadata_lines:
        raise ABOProvenanceError("ABO raw images.csv.gz has no CSV header")
    try:
        header = next(
            csv.reader([metadata_lines[0].decode("utf-8-sig").rstrip("\r\n")])
        )
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ABOProvenanceError("ABO raw images.csv.gz header is invalid") from error
    if (
        "image_id" not in header
        or "path" not in header
        or len(header) != len(set(header))
    ):
        raise ABOProvenanceError(
            "ABO raw images.csv.gz must have unique image_id and path columns"
        )

    listings_by_id = {row.item_id: row for row in listings}
    images_by_id = {row.image_id: row for row in images}
    if set(listings_by_id) != {item.item_id for item in receipt.items}:
        raise ABOProvenanceError(
            "normalized listings do not exactly match acquisition receipt items"
        )
    receipt_images: dict[tuple[str, str], ABOOfficialImageReceipt] = {}
    for item_receipt in receipt.items:
        listing = listings_by_id[item_receipt.item_id]
        raw_listing_bytes = _located_line(
            listing_lines,
            item_receipt.listing_locator,
            "ABO raw listing record",
        )
        try:
            raw_listing = json.loads(
                raw_listing_bytes.decode("utf-8").strip(),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except Exception as error:
            raise ABOProvenanceError(
                f"ABO raw listing row is invalid: {item_receipt.item_id}"
            ) from error
        required = {"item_id", "main_image_id", "other_image_id"}
        if not isinstance(raw_listing, dict) or not required.issubset(raw_listing):
            raise ABOProvenanceError("ABO raw listing row lacks identity fields")
        raw_item_id = str(raw_listing["item_id"])
        raw_main = str(raw_listing["main_image_id"])
        raw_other_value = raw_listing["other_image_id"]
        if not isinstance(raw_other_value, list):
            raise ABOProvenanceError("ABO raw other_image_id must be a list")
        raw_others = tuple(str(value) for value in raw_other_value)
        if (
            raw_item_id != listing.item_id
            or raw_main != listing.main_image_id
            or listing.other_image_id != tuple(sorted(raw_others))
        ):
            raise ABOProvenanceError(
                "normalized item/image relation is not present in raw listing row"
            )
        receipt_image_ids = {image.image_id for image in item_receipt.images}
        normalized_image_ids = {listing.main_image_id, *listing.other_image_id}
        if receipt_image_ids != normalized_image_ids:
            raise ABOProvenanceError(
                "receipt images do not exactly cover normalized listing candidates"
            )
        for image_receipt in item_receipt.images:
            expected_role = (
                "main" if image_receipt.image_id == listing.main_image_id else "other"
            )
            if image_receipt.image_role != expected_role:
                raise ABOProvenanceError("receipt image role contradicts raw listing")
            image = images_by_id.get(image_receipt.image_id)
            if image is None:
                raise ABOProvenanceError(
                    "receipt image is missing from normalized image manifest"
                )
            raw_metadata_bytes = _located_line(
                metadata_lines,
                image_receipt.metadata_locator,
                "ABO raw image metadata row",
            )
            try:
                values = next(
                    csv.reader([raw_metadata_bytes.decode("utf-8").rstrip("\r\n")])
                )
            except (UnicodeDecodeError, csv.Error, StopIteration) as error:
                raise ABOProvenanceError(
                    "ABO raw image metadata row is invalid"
                ) from error
            if len(values) != len(header):
                raise ABOProvenanceError(
                    "ABO raw image metadata row width does not match header"
                )
            raw_metadata = dict(zip(header, values, strict=True))
            if (
                raw_metadata["image_id"] != image_receipt.image_id
                or raw_metadata["path"] != image_receipt.metadata_path
            ):
                raise ABOProvenanceError(
                    "receipt image identity does not match raw images.csv.gz row"
                )
            if (
                image.path != image_receipt.metadata_path
                or image.official_image_uri != image_receipt.official_image_uri
                or image.official_image_etag != image_receipt.image_etag
                or image.source_image_sha256 != image_receipt.source_image_sha256
            ):
                raise ABOProvenanceError(
                    "normalized image manifest does not match raw acquisition receipt"
                )
            key = (item_receipt.item_id, image_receipt.image_id)
            if key in receipt_images:
                raise ABOProvenanceError("duplicate receipt item/image identity")
            receipt_images[key] = image_receipt
    if set(images_by_id) != {image_id for _, image_id in receipt_images}:
        raise ABOProvenanceError(
            "normalized image manifest contains images outside the raw receipt"
        )
    return receipt_images


def _decompress_gzip_lines(content: bytes, label: str) -> tuple[bytes, ...]:
    try:
        decompressed = gzip.decompress(content)
    except (OSError, EOFError) as error:
        raise ABOProvenanceError(f"{label} is not a complete gzip artifact") from error
    lines = tuple(decompressed.splitlines(keepends=True))
    if not lines:
        raise ABOProvenanceError(f"{label} is empty")
    return lines


def _located_line(
    lines: tuple[bytes, ...], locator: ABORawRecordLocator, label: str
) -> bytes:
    index = locator.decompressed_line_number - 1
    if index < 0 or index >= len(lines):
        raise ABOProvenanceError(f"{label} locator is outside the raw artifact")
    content = lines[index]
    if sha256_bytes(content) != locator.record_bytes_sha256:
        raise ABOProvenanceError(f"{label} bytes drifted from acquisition receipt")
    return content


class _HTTPResponse(Protocol):
    headers: Any
    content: bytes

    def raise_for_status(self) -> None: ...


class _HTTPSession(Protocol):
    def get(self, url: str, *, timeout: int) -> _HTTPResponse: ...


def acquire_targeted_abo_original_images(
    *,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    session: _HTTPSession,
) -> tuple[Path, ...]:
    """Download only receipt-selected ``images/original`` objects.

    Raw listing and images.csv.gz acquisition remains a separate, externally
    locked step.  This function never enumerates or downloads the full image
    archive and accepts a caller-supplied session for offline fake-session tests.
    """

    _validate_expected_digest(
        expected_acquisition_receipt_sha256, "acquisition-receipt"
    )
    receipt_snapshot = _snapshot_regular_file(
        Path(acquisition_receipt_path), "ABO acquisition receipt"
    )
    if receipt_snapshot.sha256 != expected_acquisition_receipt_sha256:
        raise ABOProvenanceError(
            "ABO acquisition receipt does not match independent expected digest"
        )
    receipt = _load_canonical_json_model(
        receipt_snapshot, ABOAcquisitionReceipt, "ABO acquisition receipt"
    )
    raw_cache_root = Path(raw_cache_root).absolute()
    _ensure_real_directory(raw_cache_root, "ABO raw cache root")
    unique_images: dict[str, ABOOfficialImageReceipt] = {}
    for item in receipt.items:
        for image in item.images:
            existing = unique_images.get(image.official_image_uri)
            if existing is not None and (
                existing.source_image_sha256 != image.source_image_sha256
                or existing.image_etag != image.image_etag
                or existing.cache_relative_path != image.cache_relative_path
            ):
                raise ABOProvenanceError(
                    "same official ABO image URI has contradictory receipt identity"
                )
            unique_images[image.official_image_uri] = image
    acquired = []
    for uri, image in sorted(unique_images.items()):
        target = _cache_file(raw_cache_root, image.cache_relative_path)
        if os.path.lexists(target):
            snapshot = _snapshot_regular_file(target, "cached ABO original image")
            if snapshot.sha256 != image.source_image_sha256:
                raise ABOProvenanceError(
                    "cached ABO original image differs from acquisition receipt"
                )
            acquired.append(target)
            continue
        response = session.get(uri, timeout=120)
        response.raise_for_status()
        response_etag = _response_header(response.headers, "etag")
        if response_etag != image.image_etag:
            raise ABOProvenanceError("ABO original image response ETag mismatch")
        content = bytes(response.content)
        if sha256_bytes(content) != image.source_image_sha256:
            raise ABOProvenanceError("ABO original image response SHA-256 mismatch")
        _ensure_real_directory(target.parent, "ABO original image cache directory")
        atomic_create_file(target, content)
        snapshot = _snapshot_regular_file(target, "cached ABO original image")
        if snapshot.sha256 != image.source_image_sha256:
            raise ABOProvenanceError("published ABO original image changed")
        acquired.append(target)
    _verify_snapshot(receipt_snapshot, "ABO acquisition receipt")
    return tuple(acquired)


def create_abo_candidate_review_packet(
    *,
    source_approval: VerifiedABOSourceApproval,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    listings_path: Path | str,
    image_manifest_path: Path | str,
    direct_license_path: Path | str,
    registry_license_path: Path | str,
    output_path: Path | str,
) -> tuple[ABOReviewPacketEntry, ...]:
    """Create review inputs without fabricating any human approval decision."""

    source_approval = _refresh_verified_abo_source_approval(source_approval)
    bundle = load_verified_abo_source_bundle(
        source_lock_path=source_lock_path,
        expected_source_lock_sha256=expected_source_lock_sha256,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_acquisition_receipt_sha256=expected_acquisition_receipt_sha256,
        raw_cache_root=raw_cache_root,
        listings_path=listings_path,
        image_manifest_path=image_manifest_path,
        direct_license_path=direct_license_path,
        registry_license_path=registry_license_path,
    )
    listings_by_id = {value.item_id: value for value in bundle.listings}
    images_by_id = {value.image_id: value for value in bundle.images}
    entries = []
    snapshots = []
    raw_cache_root = Path(raw_cache_root).absolute()
    for item in bundle.receipt.items:
        listing = listings_by_id[item.item_id]
        listing_sha = sha256_bytes(canonical_json_bytes(listing))
        for image_receipt in item.images:
            image = images_by_id[image_receipt.image_id]
            snapshot = _snapshot_regular_file(
                _cache_file(raw_cache_root, image_receipt.cache_relative_path),
                "ABO review candidate original image",
            )
            if snapshot.sha256 != image_receipt.source_image_sha256:
                raise ABOProvenanceError(
                    "ABO review candidate image differs from acquisition receipt"
                )
            snapshots.append(snapshot)
            entries.append(
                ABOReviewPacketEntry(
                    item_id=item.item_id,
                    image_id=image.image_id,
                    image_role=image_receipt.image_role,
                    local_path=image_receipt.cache_relative_path,
                    official_image_uri=image_receipt.official_image_uri,
                    official_image_etag=image_receipt.image_etag,
                    source_image_sha256=snapshot.sha256,
                    listing_record_sha256=listing_sha,
                    image_record_sha256=sha256_bytes(canonical_json_bytes(image)),
                )
            )
    packet = tuple(sorted(entries, key=lambda value: (value.item_id, value.image_id)))
    _bind_abo_source_approval(
        source_approval,
        bundle,
        consumed_image_snapshots=tuple(snapshots),
    )
    _verify_bundle_unchanged(bundle)
    for snapshot in snapshots:
        _verify_snapshot(snapshot, "ABO review candidate original image")
    output_path = Path(output_path).absolute()
    _ensure_real_directory(output_path.parent, "ABO review packet parent")
    _refresh_verified_abo_source_approval(source_approval)
    atomic_create_file(output_path, canonical_jsonl_bytes(packet))
    return packet


def build_abo_exact_mini(
    *,
    source_approval: VerifiedABOSourceApproval,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    listings_path: Path | str,
    image_manifest_path: Path | str,
    direct_license_path: Path | str,
    registry_license_path: Path | str,
    review_ledger_path: Path | str,
    expected_review_ledger_sha256: str,
    output_dir: Path | str,
    minimum_pair_candidates: int,
    maximum_pair_candidates: int,
    near_duplicate_policy: NearDuplicatePolicy = DEFAULT_NEAR_DUPLICATE_POLICY,
) -> ABOBuildResult:
    """Build a create-only Exact mini from already downloaded official bytes.

    The normalized snapshot contract is deliberately narrower than the remote
    archives.  Unknown archive layouts, compressed members, or inferred roles
    fail before publication; a separate, reviewed extractor must produce the
    canonical inputs accepted here.
    """

    source_approval = _refresh_verified_abo_source_approval(source_approval)
    if minimum_pair_candidates <= 0:
        raise ValueError("minimum_pair_candidates must be positive")
    if maximum_pair_candidates < minimum_pair_candidates:
        raise ValueError(
            "maximum_pair_candidates must be at least minimum_pair_candidates"
        )
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"ABO output already exists: {output_dir}")
    _ensure_real_directory(output_dir.parent, "ABO output parent")
    assessment = _assess_abo_exact_coverage(
        source_lock_path=source_lock_path,
        expected_source_lock_sha256=expected_source_lock_sha256,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_acquisition_receipt_sha256=expected_acquisition_receipt_sha256,
        raw_cache_root=raw_cache_root,
        listings_path=listings_path,
        image_manifest_path=image_manifest_path,
        direct_license_path=direct_license_path,
        registry_license_path=registry_license_path,
        review_ledger_path=review_ledger_path,
        expected_review_ledger_sha256=expected_review_ledger_sha256,
        minimum_pair_candidates=minimum_pair_candidates,
        maximum_pair_candidates=maximum_pair_candidates,
        near_duplicate_policy=near_duplicate_policy,
        temporary_parent=output_dir.parent,
    )
    bundle = assessment.bundle
    source_approval = _bind_abo_source_approval(
        source_approval,
        bundle,
        consumed_image_snapshots=assessment.image_snapshots,
    )
    review_snapshot = assessment.review_snapshot
    selected_prepared = list(assessment.selected_pairs)
    audit = assessment.audit
    if not audit.passed_provisional_gate:
        raise ABOExactCoverageError(audit)

    staging = new_staging_directory(output_dir)
    try:
        selected_by_item = {
            pair.item_id: (pair.main, pair.other) for pair in selected_prepared
        }
        drafts, prepared_by_record = _materialize_selected_assets(
            selected_by_item, staging, bundle.lock
        )
        assets = tuple(inventory_dataset_asset(draft, staging) for draft in drafts)
        publish_asset_catalog(
            assets,
            staging / _PROVISIONAL_CATALOG_DIR,
            staging,
            near_duplicate_policy,
            coverage_roots=["images/original"],
        )
        catalog = load_asset_catalog(
            staging / _PROVISIONAL_CATALOG_DIR, staging, verify_files=True
        )
        pairs = _build_exact_pairs(selected_prepared, catalog)

        provenance = tuple(
            _build_provenance(
                prepared_by_record[draft.source_record_id],
                bundle,
                review_snapshot.sha256,
            )
            for draft in drafts
        )
        review_packet = _build_review_packet(selected_prepared)
        draft_bytes = canonical_jsonl_bytes(drafts)
        provenance_bytes = canonical_jsonl_bytes(provenance)
        pair_bytes = canonical_jsonl_bytes(pairs)
        audit_bytes = canonical_json_bytes(audit)
        review_packet_bytes = canonical_jsonl_bytes(review_packet)
        (staging / _DRAFT_FILE).write_bytes(draft_bytes)
        (staging / _PROVENANCE_FILE).write_bytes(provenance_bytes)
        (staging / _PAIR_FILE).write_bytes(pair_bytes)
        (staging / _COVERAGE_FILE).write_bytes(audit_bytes)
        (staging / _REVIEW_PACKET_FILE).write_bytes(review_packet_bytes)
        manifest = _build_candidate_bundle_manifest(
            bundle=bundle,
            source_approval=source_approval,
            review_ledger_sha256=review_snapshot.sha256,
            drafts=drafts,
            catalog_sha256=catalog.catalog_sha256,
            draft_bytes=draft_bytes,
            provenance_bytes=provenance_bytes,
            pair_bytes=pair_bytes,
            audit_bytes=audit_bytes,
            review_packet_bytes=review_packet_bytes,
            staging=staging,
        )
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / _BUNDLE_FILE).write_bytes(manifest_bytes)
        bundle_manifest_sha256 = sha256_bytes(manifest_bytes)

        _verify_bundle_unchanged(bundle)
        _refresh_verified_abo_source_approval(source_approval)
        _verify_snapshot(review_snapshot, "ABO image review ledger")
        for snapshot in assessment.image_snapshots:
            _verify_snapshot(snapshot, "ABO source image")
        _parse_abo_candidate_bundle(
            staging, expected_bundle_manifest_sha256=bundle_manifest_sha256
        )
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ABOBuildResult(
        output_dir=output_dir,
        drafts=tuple(drafts),
        pairs=tuple(pairs),
        audit=audit,
        manifest=manifest,
        bundle_manifest_sha256=bundle_manifest_sha256,
    )


def _parse_abo_candidate_bundle(
    root: Path | str, *, expected_bundle_manifest_sha256: str
) -> _ParsedABOCandidateBundle:
    """Parse self-consistent candidate bytes without granting verified status."""

    _validate_expected_digest(expected_bundle_manifest_sha256, "bundle-manifest")
    root = Path(root).absolute()
    _require_real_directory(root, "ABO candidate bundle root")
    manifest_snapshot = _snapshot_regular_file(
        root / _BUNDLE_FILE, "ABO candidate bundle manifest"
    )
    if manifest_snapshot.sha256 != expected_bundle_manifest_sha256:
        raise ABOProvenanceError(
            "ABO candidate bundle does not match external expected digest"
        )
    manifest = _load_canonical_json_model(
        manifest_snapshot, ABOCandidateBundleManifest, "ABO candidate bundle manifest"
    )
    unsigned = manifest.model_dump(mode="json", exclude={"bundle_self_sha256"})
    if sha256_bytes(canonical_json_bytes(unsigned)) != manifest.bundle_self_sha256:
        raise ABOProvenanceError("ABO candidate bundle self hash is invalid")

    descriptors = {descriptor.path: descriptor for descriptor in manifest.files}
    actual_paths = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if _is_symlink_or_reparse(metadata):
            raise ABOProvenanceError("ABO candidate bundle contains a reparse path")
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_directory(path, "ABO candidate bundle directory")
        elif stat.S_ISREG(metadata.st_mode) and relative != _BUNDLE_FILE:
            actual_paths.add(relative)
        elif not stat.S_ISREG(metadata.st_mode):
            raise ABOProvenanceError("ABO candidate bundle contains a special file")
    if set(descriptors) != actual_paths:
        raise ABOProvenanceError(
            "ABO candidate bundle file set differs from the locked manifest"
        )
    payload_snapshots: dict[str, _FileSnapshot] = {}
    for relative, descriptor in descriptors.items():
        snapshot = _snapshot_regular_file(
            root / PurePosixPath(relative), f"ABO candidate payload {relative}"
        )
        if (
            len(snapshot.content) != descriptor.bytes
            or snapshot.sha256 != descriptor.sha256
        ):
            raise ABOProvenanceError(f"ABO candidate payload drifted: {relative}")
        payload_snapshots[relative] = snapshot
    required_static = {
        _DRAFT_FILE,
        _PROVENANCE_FILE,
        _PAIR_FILE,
        _COVERAGE_FILE,
        _REVIEW_PACKET_FILE,
        f"{_PROVISIONAL_CATALOG_DIR}/assets.jsonl",
        f"{_PROVISIONAL_CATALOG_DIR}/components.jsonl",
        f"{_PROVISIONAL_CATALOG_DIR}/manifest.json",
    }
    if not required_static.issubset(actual_paths):
        raise ABOProvenanceError("ABO candidate bundle is missing required metadata")

    drafts = _load_canonical_jsonl(
        payload_snapshots[_DRAFT_FILE], DatasetAssetDraft, "ABO dataset asset drafts"
    )
    provenance = _load_canonical_jsonl(
        payload_snapshots[_PROVENANCE_FILE],
        ABOAssetProvenance,
        "ABO asset provenance",
    )
    pairs = _load_canonical_jsonl(
        payload_snapshots[_PAIR_FILE], ABOExactPair, "ABO pair candidates"
    )
    audit = _load_canonical_json_model(
        payload_snapshots[_COVERAGE_FILE],
        ABOExactCoverageAudit,
        "ABO coverage audit",
    )
    review_packet = _load_canonical_jsonl(
        payload_snapshots[_REVIEW_PACKET_FILE],
        ABOReviewPacketEntry,
        "ABO review packet",
    )
    expected_image_paths = {draft.local_path for draft in drafts}
    if any(not path.startswith("images/original/") for path in expected_image_paths):
        raise ABOProvenanceError("ABO candidate images must use images/original only")
    permitted_paths = required_static | expected_image_paths
    if actual_paths != permitted_paths:
        raise ABOProvenanceError(
            "ABO candidate bundle contains unregistered or missing image assets"
        )
    if len(drafts) != manifest.provisional_asset_count:
        raise ABOProvenanceError("ABO candidate draft count differs from manifest")
    if len(pairs) != manifest.provisional_pair_count:
        raise ABOProvenanceError("ABO candidate pair count differs from manifest")
    if len(provenance) != len(drafts) or len(review_packet) != len(drafts):
        raise ABOProvenanceError("ABO candidate metadata does not cover every asset")
    artifact_hashes = {
        _DRAFT_FILE: manifest.dataset_assets_sha256,
        _PROVENANCE_FILE: manifest.provenance_sha256,
        _PAIR_FILE: manifest.exact_pairs_sha256,
        _COVERAGE_FILE: manifest.coverage_audit_sha256,
        _REVIEW_PACKET_FILE: manifest.review_packet_sha256,
    }
    for relative, expected in artifact_hashes.items():
        if payload_snapshots[relative].sha256 != expected:
            raise ABOProvenanceError(f"ABO manifest hash mismatch: {relative}")
    catalog = load_asset_catalog(
        root / _PROVISIONAL_CATALOG_DIR, root, verify_files=True
    )
    if catalog.catalog_sha256 != manifest.provisional_asset_catalog_sha256:
        raise ABOProvenanceError("ABO provisional catalog hash differs from bundle")
    if {asset.asset_id for asset in catalog.assets} != {
        pair.query_asset_id for pair in pairs
    } | {pair.gallery_asset_id for pair in pairs}:
        raise ABOProvenanceError(
            "ABO pair candidates do not exactly cover catalog assets"
        )
    for snapshot in payload_snapshots.values():
        _verify_snapshot(snapshot, "ABO candidate bundle payload")
    _verify_snapshot(manifest_snapshot, "ABO candidate bundle manifest")
    return _ParsedABOCandidateBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_snapshot.sha256,
        drafts=drafts,
        pairs=pairs,
        audit=audit,
        provenance=provenance,
        review_packet=review_packet,
        catalog=catalog,
    )


def load_verified_abo_candidate_bundle(
    root: Path | str,
    *,
    expected_bundle_manifest_sha256: str,
    source_bundle: VerifiedABOSourceBundle,
    source_approval: VerifiedABOSourceApproval,
    review_ledger_path: Path | str,
    expected_review_ledger_sha256: str,
) -> VerifiedABOCandidateBundle:
    """Re-derive a candidate bundle from its locked raw source and human review.

    A manifest digest proves only the identity of the candidate directory.  It is
    deliberately insufficient to grant ``VerifiedABOCandidateBundle``: every
    draft, image, provenance row, pair, review packet row, and disposition is
    rebuilt from the independently verified source bundle and review ledger.
    """

    source_approval = _refresh_verified_abo_source_approval(source_approval)
    parsed = _parse_abo_candidate_bundle(
        root, expected_bundle_manifest_sha256=expected_bundle_manifest_sha256
    )
    if (
        not isinstance(source_bundle, VerifiedABOSourceBundle)
        or source_bundle._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise ABOProvenanceError(
            "formal ABO candidate loading requires a VerifiedABOSourceBundle"
        )
    source_bundle = load_verified_abo_source_bundle(
        source_lock_path=source_bundle.lock_snapshot.path,
        expected_source_lock_sha256=source_bundle.lock_sha256,
        acquisition_receipt_path=source_bundle.receipt_snapshot.path,
        expected_acquisition_receipt_sha256=source_bundle.receipt_sha256,
        raw_cache_root=source_bundle.raw_cache_root,
        listings_path=source_bundle.listings_snapshot.path,
        image_manifest_path=source_bundle.image_manifest_snapshot.path,
        direct_license_path=source_bundle.direct_license_snapshot.path,
        registry_license_path=source_bundle.registry_license_snapshot.path,
    )
    source_approval = _bind_abo_source_approval(source_approval, source_bundle)
    review_snapshot, _ = _load_review_ledger(
        Path(review_ledger_path), expected_review_ledger_sha256
    )
    manifest_source_binding = (
        parsed.manifest.required_source_lock_sha256,
        parsed.manifest.source_review_policy_sha256,
        parsed.manifest.source_review_ledger_sha256,
        parsed.manifest.source_review_record_sha256,
        parsed.manifest.source_review_license_evidence_sha256,
        parsed.manifest.source_lock_sha256,
        parsed.manifest.acquisition_receipt_sha256,
        parsed.manifest.listings_sha256,
        parsed.manifest.image_manifest_sha256,
        parsed.manifest.direct_license_sha256,
        parsed.manifest.registry_license_sha256,
        parsed.manifest.review_ledger_sha256,
        parsed.manifest.permissions,
    )
    expected_source_binding = (
        source_approval.required_source_lock.lock_file_sha256,
        source_approval.policy_file_sha256,
        source_approval.ledger_file_sha256,
        sha256_bytes(canonical_json_bytes(source_approval.review_record)),
        source_approval.license_evidence_sha256,
        source_bundle.lock_sha256,
        source_bundle.receipt_sha256,
        source_bundle.listings_snapshot.sha256,
        source_bundle.image_manifest_snapshot.sha256,
        source_bundle.direct_license_snapshot.sha256,
        source_bundle.registry_license_snapshot.sha256,
        review_snapshot.sha256,
        source_bundle.lock.permissions,
    )
    if manifest_source_binding != expected_source_binding:
        raise ABOProvenanceError(
            "ABO candidate manifest is not bound to the verified source and review"
        )

    assessment = _assess_abo_exact_coverage(
        source_lock_path=source_bundle.lock_snapshot.path,
        expected_source_lock_sha256=source_bundle.lock_sha256,
        acquisition_receipt_path=source_bundle.receipt_snapshot.path,
        expected_acquisition_receipt_sha256=source_bundle.receipt_sha256,
        raw_cache_root=source_bundle.raw_cache_root,
        listings_path=source_bundle.listings_snapshot.path,
        image_manifest_path=source_bundle.image_manifest_snapshot.path,
        direct_license_path=source_bundle.direct_license_snapshot.path,
        registry_license_path=source_bundle.registry_license_snapshot.path,
        review_ledger_path=review_snapshot.path,
        expected_review_ledger_sha256=review_snapshot.sha256,
        minimum_pair_candidates=parsed.audit.minimum_pair_candidates,
        maximum_pair_candidates=parsed.audit.maximum_pair_candidates,
        near_duplicate_policy=parsed.catalog.manifest.near_duplicate_policy,
        temporary_parent=parsed.root.parent,
    )
    _bind_abo_source_approval(
        source_approval,
        source_bundle,
        consumed_image_snapshots=assessment.image_snapshots,
    )
    if assessment.audit != parsed.audit:
        raise ABOProvenanceError(
            "ABO candidate dispositions differ from source-bound recomputation"
        )

    with tempfile.TemporaryDirectory(
        prefix=".abo-candidate-reverify-", dir=str(parsed.root.parent)
    ) as temporary:
        expected_root = Path(temporary)
        selected_by_item = {
            pair.item_id: (pair.main, pair.other) for pair in assessment.selected_pairs
        }
        expected_drafts, prepared_by_record = _materialize_selected_assets(
            selected_by_item, expected_root, source_bundle.lock
        )
        if tuple(expected_drafts) != parsed.drafts:
            raise ABOProvenanceError(
                "ABO candidate drafts differ from source-bound recomputation"
            )
        for draft in expected_drafts:
            expected_image = _snapshot_regular_file(
                expected_root / draft.local_path, "re-derived ABO candidate image"
            )
            actual_image = _snapshot_regular_file(
                parsed.root / draft.local_path, "published ABO candidate image"
            )
            if expected_image.sha256 != actual_image.sha256:
                raise ABOProvenanceError(
                    "ABO candidate image differs from its verified original source"
                )
        expected_assets = tuple(
            inventory_dataset_asset(draft, expected_root) for draft in expected_drafts
        )
        expected_catalog_root = expected_root / _PROVISIONAL_CATALOG_DIR
        publish_asset_catalog(
            expected_assets,
            expected_catalog_root,
            expected_root,
            parsed.catalog.manifest.near_duplicate_policy,
            coverage_roots=["images/original"],
        )
        expected_catalog = load_asset_catalog(
            expected_catalog_root, expected_root, verify_files=True
        )
        if expected_catalog.catalog_sha256 != parsed.catalog.catalog_sha256:
            raise ABOProvenanceError(
                "ABO candidate catalog differs from source-bound recomputation"
            )
        expected_pairs = _build_exact_pairs(
            list(assessment.selected_pairs), expected_catalog
        )
        if expected_pairs != parsed.pairs:
            raise ABOProvenanceError(
                "ABO exact pairs differ from source-bound recomputation"
            )
        expected_provenance = tuple(
            _build_provenance(
                prepared_by_record[draft.source_record_id],
                source_bundle,
                review_snapshot.sha256,
            )
            for draft in expected_drafts
        )
        if expected_provenance != parsed.provenance:
            raise ABOProvenanceError(
                "ABO provenance differs from source-bound recomputation"
            )
        expected_review_packet = _build_review_packet(list(assessment.selected_pairs))
        if expected_review_packet != parsed.review_packet:
            raise ABOProvenanceError(
                "ABO review packet differs from source-bound recomputation"
            )

    expected_manifest = _build_candidate_bundle_manifest(
        bundle=source_bundle,
        source_approval=source_approval,
        review_ledger_sha256=review_snapshot.sha256,
        drafts=list(parsed.drafts),
        catalog_sha256=parsed.catalog.catalog_sha256,
        draft_bytes=canonical_jsonl_bytes(parsed.drafts),
        provenance_bytes=canonical_jsonl_bytes(parsed.provenance),
        pair_bytes=canonical_jsonl_bytes(parsed.pairs),
        audit_bytes=canonical_json_bytes(parsed.audit),
        review_packet_bytes=canonical_jsonl_bytes(parsed.review_packet),
        staging=parsed.root,
    )
    if expected_manifest != parsed.manifest:
        raise ABOProvenanceError(
            "ABO candidate manifest differs from source-bound recomputation"
        )
    _verify_bundle_unchanged(source_bundle)
    source_approval = _refresh_verified_abo_source_approval(source_approval)
    _verify_snapshot(review_snapshot, "ABO image review ledger")
    refreshed = _parse_abo_candidate_bundle(
        parsed.root,
        expected_bundle_manifest_sha256=parsed.manifest_sha256,
    )
    return VerifiedABOCandidateBundle(
        **refreshed.__dict__,
        source_bundle=source_bundle,
        source_approval=source_approval,
        review_snapshot=review_snapshot,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _refresh_verified_abo_candidate_bundle(
    bundle: VerifiedABOCandidateBundle,
) -> VerifiedABOCandidateBundle:
    if (
        not isinstance(bundle, VerifiedABOCandidateBundle)
        or bundle._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise ABOProvenanceError(
            "formal ABO finalization requires a source-bound verified candidate"
        )
    return load_verified_abo_candidate_bundle(
        bundle.root,
        expected_bundle_manifest_sha256=bundle.manifest_sha256,
        source_bundle=bundle.source_bundle,
        source_approval=bundle.source_approval,
        review_ledger_path=bundle.review_snapshot.path,
        expected_review_ledger_sha256=bundle.review_snapshot.sha256,
    )


def finalize_abo_exact_candidates(
    *,
    candidate_bundle: VerifiedABOCandidateBundle,
    global_catalog: AssetCatalog,
    global_scope_path: Path | str,
    expected_global_scope_sha256: str,
    output_path: Path | str,
    minimum_finalized_pairs: int,
) -> ABOGlobalFinalizationManifest:
    """Finalize candidates only against an externally scoped cross-source catalog."""

    if minimum_finalized_pairs <= 0:
        raise ValueError("minimum_finalized_pairs must be positive")
    bundle = _refresh_verified_abo_candidate_bundle(candidate_bundle)
    global_catalog.require_verified_files()
    _validate_expected_digest(expected_global_scope_sha256, "global-scope")
    scope_snapshot = _snapshot_regular_file(
        Path(global_scope_path), "ABO global catalog scope"
    )
    if scope_snapshot.sha256 != expected_global_scope_sha256:
        raise ABOProvenanceError(
            "ABO global scope does not match independent expected digest"
        )
    scope = _load_canonical_json_model(
        scope_snapshot, ABOGlobalCatalogScope, "ABO global catalog scope"
    )
    actual_sources = tuple(
        sorted({source.source_dataset for source in global_catalog.manifest.sources})
    )
    if (
        scope.catalog_sha256 != global_catalog.catalog_sha256
        or scope.asset_count != global_catalog.manifest.asset_count
        or scope.coverage_roots != global_catalog.manifest.coverage_roots
        or scope.source_datasets != actual_sources
    ):
        raise ABOProvenanceError(
            "ABO global scope does not exactly describe the supplied catalog"
        )

    candidate_ids = {
        asset_id
        for pair in bundle.pairs
        for asset_id in (pair.query_asset_id, pair.gallery_asset_id)
    }
    global_catalog.verify_asset_ids(candidate_ids)
    local_by_id = {asset.asset_id: asset for asset in bundle.catalog.assets}
    for asset_id in candidate_ids:
        try:
            global_asset = global_catalog.resolve_asset_id(asset_id).asset
        except Exception as error:
            raise ABOProvenanceError(
                f"ABO candidate is absent from global catalog: {asset_id}"
            ) from error
        local_asset = local_by_id[asset_id]
        local_payload = local_asset.model_dump(
            mode="json", exclude={"near_duplicate_cluster_id"}
        )
        global_payload = global_asset.model_dump(
            mode="json", exclude={"near_duplicate_cluster_id"}
        )
        if local_payload != global_payload:
            raise ABOProvenanceError(
                f"ABO candidate metadata differs in global catalog: {asset_id}"
            )

    finalized: list[ABOExactPair] = []
    dispositions: list[ABOGlobalPairDisposition] = []
    for pair in sorted(bundle.pairs, key=lambda value: value.item_id):
        trial = [*finalized, pair]
        if _pairs_are_globally_clean(
            trial, global_catalog, candidate_pool_ids=candidate_ids
        ):
            finalized.append(pair)
            dispositions.append(
                ABOGlobalPairDisposition(
                    item_id=pair.item_id,
                    disposition="finalized",
                    reason="globally_clean",
                )
            )
        else:
            dispositions.append(
                ABOGlobalPairDisposition(
                    item_id=pair.item_id,
                    disposition="excluded",
                    reason="global_forbidden_leakage",
                )
            )
    unsigned = {
        "schema_version": 1,
        "finalization_policy_version": "abo-cross-source-global-finalization-v1",
        "candidate_bundle_sha256": bundle.manifest_sha256,
        "global_catalog_sha256": global_catalog.catalog_sha256,
        "global_scope_sha256": scope_snapshot.sha256,
        "finalized_pair_count": len(finalized),
        "minimum_finalized_pairs": minimum_finalized_pairs,
        "pair_dispositions": [value.model_dump(mode="json") for value in dispositions],
    }
    manifest = ABOGlobalFinalizationManifest.model_validate(
        {
            **unsigned,
            "finalization_self_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )
    if len(finalized) < minimum_finalized_pairs:
        raise ABOGlobalFinalizationError(manifest)
    _verify_snapshot(scope_snapshot, "ABO global catalog scope")
    global_catalog.verify_asset_ids(candidate_ids)
    _refresh_verified_abo_candidate_bundle(bundle)
    output_path = Path(output_path).absolute()
    _ensure_real_directory(output_path.parent, "ABO finalization output parent")
    atomic_create_file(output_path, canonical_json_bytes(manifest))
    return manifest


def load_verified_abo_global_finalization(
    path: Path | str,
    *,
    expected_finalization_sha256: str,
    candidate_bundle: VerifiedABOCandidateBundle,
    global_catalog: AssetCatalog,
    global_scope_path: Path | str,
    expected_global_scope_sha256: str,
) -> VerifiedABOGlobalFinalization:
    """Recompute a published final selection from all still-verified inputs."""

    _validate_expected_digest(expected_finalization_sha256, "finalization")
    snapshot = _snapshot_regular_file(Path(path), "ABO global finalization")
    if snapshot.sha256 != expected_finalization_sha256:
        raise ABOProvenanceError(
            "ABO global finalization does not match independent expected digest"
        )
    manifest = _load_canonical_json_model(
        snapshot, ABOGlobalFinalizationManifest, "ABO global finalization"
    )
    unsigned = manifest.model_dump(mode="json", exclude={"finalization_self_sha256"})
    if (
        sha256_bytes(canonical_json_bytes(unsigned))
        != manifest.finalization_self_sha256
    ):
        raise ABOProvenanceError("ABO global finalization self hash is invalid")

    bundle = _refresh_verified_abo_candidate_bundle(candidate_bundle)
    global_catalog.require_verified_files()
    _validate_expected_digest(expected_global_scope_sha256, "global-scope")
    scope_snapshot = _snapshot_regular_file(
        Path(global_scope_path), "ABO global catalog scope"
    )
    if scope_snapshot.sha256 != expected_global_scope_sha256:
        raise ABOProvenanceError(
            "ABO global scope does not match independent expected digest"
        )
    scope = _load_canonical_json_model(
        scope_snapshot, ABOGlobalCatalogScope, "ABO global catalog scope"
    )
    actual_sources = tuple(
        sorted({source.source_dataset for source in global_catalog.manifest.sources})
    )
    if (
        scope.catalog_sha256 != global_catalog.catalog_sha256
        or scope.asset_count != global_catalog.manifest.asset_count
        or scope.coverage_roots != global_catalog.manifest.coverage_roots
        or scope.source_datasets != actual_sources
    ):
        raise ABOProvenanceError(
            "ABO global scope does not exactly describe the supplied catalog"
        )

    candidate_ids = {
        asset_id
        for pair in bundle.pairs
        for asset_id in (pair.query_asset_id, pair.gallery_asset_id)
    }
    global_catalog.verify_asset_ids(candidate_ids)
    local_by_id = {asset.asset_id: asset for asset in bundle.catalog.assets}
    for asset_id in candidate_ids:
        global_asset = global_catalog.resolve_asset_id(asset_id).asset
        local_payload = local_by_id[asset_id].model_dump(
            mode="json", exclude={"near_duplicate_cluster_id"}
        )
        global_payload = global_asset.model_dump(
            mode="json", exclude={"near_duplicate_cluster_id"}
        )
        if local_payload != global_payload:
            raise ABOProvenanceError(
                f"ABO candidate metadata differs in global catalog: {asset_id}"
            )
    finalized: list[ABOExactPair] = []
    expected_dispositions: list[ABOGlobalPairDisposition] = []
    for pair in sorted(bundle.pairs, key=lambda value: value.item_id):
        trial = [*finalized, pair]
        if _pairs_are_globally_clean(
            trial, global_catalog, candidate_pool_ids=candidate_ids
        ):
            finalized.append(pair)
            expected_dispositions.append(
                ABOGlobalPairDisposition(
                    item_id=pair.item_id,
                    disposition="finalized",
                    reason="globally_clean",
                )
            )
        else:
            expected_dispositions.append(
                ABOGlobalPairDisposition(
                    item_id=pair.item_id,
                    disposition="excluded",
                    reason="global_forbidden_leakage",
                )
            )
    expected_binding = (
        bundle.manifest_sha256,
        global_catalog.catalog_sha256,
        scope_snapshot.sha256,
        len(finalized),
        tuple(expected_dispositions),
    )
    actual_binding = (
        manifest.candidate_bundle_sha256,
        manifest.global_catalog_sha256,
        manifest.global_scope_sha256,
        manifest.finalized_pair_count,
        manifest.pair_dispositions,
    )
    if actual_binding != expected_binding:
        raise ABOProvenanceError(
            "ABO global finalization differs from deterministic recomputation"
        )
    if manifest.finalized_pair_count < manifest.minimum_finalized_pairs:
        raise ABOProvenanceError("ABO global finalization did not pass its frozen gate")
    _verify_snapshot(snapshot, "ABO global finalization")
    _verify_snapshot(scope_snapshot, "ABO global catalog scope")
    _refresh_verified_abo_candidate_bundle(bundle)
    return VerifiedABOGlobalFinalization(
        path=snapshot.path,
        file_sha256=snapshot.sha256,
        manifest=manifest,
        candidate_bundle=bundle,
        global_catalog=global_catalog,
    )


def _pairs_are_globally_clean(
    pairs: list[ABOExactPair],
    catalog: AssetCatalog,
    *,
    candidate_pool_ids: set[str],
) -> bool:
    queries = [
        QueryAssetReference(
            query_id=f"abo-global:{pair.item_id}",
            intent="exact_match",
            asset_id=pair.query_asset_id,
        )
        for pair in pairs
    ]
    galleries = [pair.gallery_asset_id for pair in pairs]
    cross_report = audit_query_gallery_eligibility(queries, galleries, catalog)
    if cross_report.violation_count:
        return False
    candidate_ids = {
        asset_id
        for pair in pairs
        for asset_id in (pair.query_asset_id, pair.gallery_asset_id)
    }
    all_asset_ids = {asset.asset_id for asset in catalog.assets}
    for asset_id in sorted(candidate_ids):
        # Conflicts within the candidate pool are resolved deterministically by
        # the incremental selection below.  A conflicting asset outside that
        # pool cannot be removed by this finalizer and therefore blocks the pair.
        other_ids = sorted(all_asset_ids - candidate_pool_ids)
        if not other_ids:
            continue
        report = audit_query_gallery_eligibility(
            [
                QueryAssetReference(
                    query_id=f"abo-global-component:{asset_id}",
                    intent="exact_match",
                    asset_id=asset_id,
                )
            ],
            other_ids,
            catalog,
        )
        if report.exact_or_near_duplicate_violations:
            return False
    query_ids = [pair.query_asset_id for pair in pairs]
    gallery_ids = [pair.gallery_asset_id for pair in pairs]
    if any(
        _share_forbidden_component(left, right, catalog)
        for left, right in combinations(query_ids, 2)
    ):
        return False
    if any(
        _share_forbidden_component(left, right, catalog)
        for left, right in combinations(gallery_ids, 2)
    ):
        return False
    return True


def _share_forbidden_component(
    left_asset_id: str, right_asset_id: str, catalog: AssetCatalog
) -> bool:
    report = audit_query_gallery_eligibility(
        [
            QueryAssetReference(
                query_id="abo-global-pairwise-check",
                intent="exact_match",
                asset_id=left_asset_id,
            )
        ],
        [right_asset_id],
        catalog,
    )
    return bool(report.exact_or_near_duplicate_violations)


def audit_abo_exact_coverage(
    *,
    source_approval: VerifiedABOSourceApproval,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    listings_path: Path | str,
    image_manifest_path: Path | str,
    direct_license_path: Path | str,
    registry_license_path: Path | str,
    review_ledger_path: Path | str,
    expected_review_ledger_sha256: str,
    minimum_pair_candidates: int,
    maximum_pair_candidates: int,
    near_duplicate_policy: NearDuplicatePolicy = DEFAULT_NEAR_DUPLICATE_POLICY,
) -> ABOExactCoverageAudit:
    """Audit the locked snapshot without publishing clean data.

    An insufficient result is carried on :class:`ABOExactCoverageError` so a
    caller cannot accidentally treat a merely serialized failure report as an
    authorization to continue.
    """

    source_approval = _refresh_verified_abo_source_approval(source_approval)
    assessment = _assess_abo_exact_coverage(
        source_lock_path=source_lock_path,
        expected_source_lock_sha256=expected_source_lock_sha256,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_acquisition_receipt_sha256=expected_acquisition_receipt_sha256,
        raw_cache_root=raw_cache_root,
        listings_path=listings_path,
        image_manifest_path=image_manifest_path,
        direct_license_path=direct_license_path,
        registry_license_path=registry_license_path,
        review_ledger_path=review_ledger_path,
        expected_review_ledger_sha256=expected_review_ledger_sha256,
        minimum_pair_candidates=minimum_pair_candidates,
        maximum_pair_candidates=maximum_pair_candidates,
        near_duplicate_policy=near_duplicate_policy,
        temporary_parent=None,
    )
    _bind_abo_source_approval(
        source_approval,
        assessment.bundle,
        consumed_image_snapshots=assessment.image_snapshots,
    )
    if not assessment.audit.passed_provisional_gate:
        raise ABOExactCoverageError(assessment.audit)
    return assessment.audit


def _assess_abo_exact_coverage(
    *,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    acquisition_receipt_path: Path | str,
    expected_acquisition_receipt_sha256: str,
    raw_cache_root: Path | str,
    listings_path: Path | str,
    image_manifest_path: Path | str,
    direct_license_path: Path | str,
    registry_license_path: Path | str,
    review_ledger_path: Path | str,
    expected_review_ledger_sha256: str,
    minimum_pair_candidates: int,
    maximum_pair_candidates: int,
    near_duplicate_policy: NearDuplicatePolicy,
    temporary_parent: Path | None,
) -> _CoverageAssessment:
    if minimum_pair_candidates <= 0:
        raise ValueError("minimum_pair_candidates must be positive")
    if maximum_pair_candidates < minimum_pair_candidates:
        raise ValueError(
            "maximum_pair_candidates must be at least minimum_pair_candidates"
        )
    bundle = load_verified_abo_source_bundle(
        source_lock_path=source_lock_path,
        expected_source_lock_sha256=expected_source_lock_sha256,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_acquisition_receipt_sha256=expected_acquisition_receipt_sha256,
        raw_cache_root=raw_cache_root,
        listings_path=listings_path,
        image_manifest_path=image_manifest_path,
        direct_license_path=direct_license_path,
        registry_license_path=registry_license_path,
    )
    review_snapshot, reviews = _load_review_ledger(
        Path(review_ledger_path), expected_review_ledger_sha256
    )
    prepared_by_item, item_reasons, asset_reasons, image_snapshots = _screen_assets(
        bundle=bundle,
        reviews=reviews,
        raw_cache_root=Path(raw_cache_root).absolute(),
    )
    temporary_options = {"prefix": ".abo-exact-audit-"}
    if temporary_parent is not None:
        temporary_options["dir"] = str(temporary_parent)
    with tempfile.TemporaryDirectory(**temporary_options) as temporary:
        audit_root = Path(temporary)
        all_drafts, _ = _materialize_prepared_assets(
            prepared_by_item, audit_root, bundle.lock
        )
        if all_drafts:
            all_assets = tuple(
                inventory_dataset_asset(draft, audit_root) for draft in all_drafts
            )
            publish_asset_catalog(
                all_assets,
                audit_root / _PROVISIONAL_CATALOG_DIR,
                audit_root,
                near_duplicate_policy,
                coverage_roots=["images/original"],
            )
            audit_catalog = load_asset_catalog(
                audit_root / _PROVISIONAL_CATALOG_DIR,
                audit_root,
                verify_files=True,
            )
            eligible_pairs = _choose_local_provisional_pairs(
                prepared_by_item,
                audit_catalog,
                item_reasons,
            )
        else:
            eligible_pairs = []
    selected_pairs = eligible_pairs[:maximum_pair_candidates]
    audit = _build_coverage_audit(
        bundle=bundle,
        eligible_pairs=eligible_pairs,
        selected_pairs=selected_pairs,
        item_reasons=item_reasons,
        asset_reasons=asset_reasons,
        minimum_pair_candidates=minimum_pair_candidates,
        maximum_pair_candidates=maximum_pair_candidates,
    )
    _verify_bundle_unchanged(bundle)
    _verify_snapshot(review_snapshot, "ABO image review ledger")
    for snapshot in image_snapshots:
        _verify_snapshot(snapshot, "ABO source image")
    return _CoverageAssessment(
        bundle=bundle,
        review_snapshot=review_snapshot,
        image_snapshots=image_snapshots,
        locally_viable_pairs=tuple(eligible_pairs),
        selected_pairs=tuple(selected_pairs),
        audit=audit,
    )


def _screen_assets(
    *,
    bundle: VerifiedABOSourceBundle,
    reviews: tuple[ABOImageReviewDecision, ...],
    raw_cache_root: Path,
) -> tuple[
    dict[str, tuple[_PreparedAsset | None, tuple[_PreparedAsset, ...]]],
    defaultdict[str, set[str]],
    defaultdict[str, set[str]],
    tuple[_FileSnapshot, ...],
]:
    images = {row.image_id: row for row in bundle.images}
    reviews_by_key = {(row.item_id, row.image_id): row for row in reviews}
    if len(reviews_by_key) != len(reviews):
        raise ABOProvenanceError("ABO review ledger decisions must be unique")
    candidate_review_keys = {
        (listing.item_id, image_id)
        for listing in bundle.listings
        for image_id in (listing.main_image_id, *listing.other_image_id)
    }
    unexpected_reviews = sorted(set(reviews_by_key).difference(candidate_review_keys))
    if unexpected_reviews:
        raise ABOProvenanceError(
            "ABO review ledger contains decisions outside the locked listings: "
            f"{unexpected_reviews}"
        )
    listing_hashes = {
        row.item_id: sha256_bytes(canonical_json_bytes(row)) for row in bundle.listings
    }
    image_hashes = {
        row.image_id: sha256_bytes(canonical_json_bytes(row)) for row in bundle.images
    }
    association: defaultdict[str, set[str]] = defaultdict(set)
    for listing in bundle.listings:
        for image_id in (listing.main_image_id, *listing.other_image_id):
            association[image_id].add(listing.item_id)

    prepared: dict[str, tuple[_PreparedAsset | None, tuple[_PreparedAsset, ...]]] = {}
    item_reasons: defaultdict[str, set[str]] = defaultdict(set)
    asset_reasons: defaultdict[str, set[str]] = defaultdict(set)
    snapshots: dict[Path, _FileSnapshot] = {}
    for listing in bundle.listings:
        accepted: dict[str, _PreparedAsset] = {}
        roles: dict[str, Literal["main", "other"]] = {
            listing.main_image_id: "main",
            **{image_id: "other" for image_id in listing.other_image_id},
        }
        for image_id, role in roles.items():
            asset_key = f"{listing.item_id}/{image_id}"
            if len(association[image_id]) != 1:
                asset_reasons["cross_item_image_identity"].add(asset_key)
                continue
            image = images.get(image_id)
            if image is None:
                asset_reasons["missing_image_record"].add(asset_key)
                continue
            receipt_image = bundle.receipt_images.get((listing.item_id, image_id))
            if receipt_image is None:
                raise ABOProvenanceError(
                    f"ABO normalized relation has no raw receipt identity: {asset_key}"
                )
            image_path = _cache_file(raw_cache_root, receipt_image.cache_relative_path)
            if not os.path.lexists(image_path):
                asset_reasons["missing_image_file"].add(asset_key)
                continue
            snapshot = _snapshot_regular_file(image_path, f"ABO image {image_id}")
            if (
                snapshot.sha256 != image.source_image_sha256
                or snapshot.sha256 != receipt_image.source_image_sha256
            ):
                raise ABOProvenanceError(
                    f"ABO original image bytes do not match raw receipt: {image_id}"
                )
            snapshots[snapshot.path] = snapshot
            try:
                with Image.open(io.BytesIO(snapshot.content)) as opened:
                    opened.load()
                    opened.convert("RGB")
            except (OSError, ValueError):
                asset_reasons["invalid_image"].add(asset_key)
                continue
            review = reviews_by_key.get((listing.item_id, image_id))
            if review is None:
                asset_reasons["missing_human_approval"].add(asset_key)
                continue
            if (
                review.listing_record_sha256 != listing_hashes[listing.item_id]
                or review.image_record_sha256 != image_hashes[image_id]
                or review.source_image_sha256 != snapshot.sha256
            ):
                raise ABOProvenanceError(
                    f"ABO review decision is not bound to current evidence: {asset_key}"
                )
            if review.decision != "approve_catalog_product_photo":
                asset_reasons["review_rejected"].add(asset_key)
                continue
            local_path = receipt_image.cache_relative_path
            accepted[image_id] = _PreparedAsset(
                item=listing,
                image=image,
                role=role,
                review=review,
                snapshot=snapshot,
                listing_record_sha256=listing_hashes[listing.item_id],
                image_record_sha256=image_hashes[image_id],
                receipt_image=receipt_image,
                local_path=local_path,
            )
        main = accepted.get(listing.main_image_id)
        others = tuple(
            accepted[image_id]
            for image_id in listing.other_image_id
            if image_id in accepted
        )
        if main is None or not others:
            item_reasons["single_approved_original_catalog_view"].add(listing.item_id)
        prepared[listing.item_id] = (main, others)
    return prepared, item_reasons, asset_reasons, tuple(snapshots.values())


def _materialize_prepared_assets(
    prepared_by_item: dict[
        str, tuple[_PreparedAsset | None, tuple[_PreparedAsset, ...]]
    ],
    root: Path,
    lock: ABOSourceLock,
) -> tuple[list[DatasetAssetDraft], dict[str, _PreparedAsset]]:
    selected: dict[str, tuple[_PreparedAsset, ...]] = {}
    for item_id, (main, others) in prepared_by_item.items():
        if main is not None and others:
            selected[item_id] = (main, *others)
    return _materialize_selected_assets(selected, root, lock)


def _materialize_selected_assets(
    selected_by_item: dict[str, tuple[_PreparedAsset, ...]],
    root: Path,
    lock: ABOSourceLock,
) -> tuple[list[DatasetAssetDraft], dict[str, _PreparedAsset]]:
    drafts: list[DatasetAssetDraft] = []
    prepared_by_record: dict[str, _PreparedAsset] = {}
    for item_id in sorted(selected_by_item):
        for prepared in sorted(
            selected_by_item[item_id],
            key=lambda value: (value.role != "main", value.image.image_id),
        ):
            destination = root / prepared.local_path
            _ensure_real_directory(destination.parent, "ABO staged image directory")
            destination.write_bytes(prepared.snapshot.content)
            source_record_id = _source_record_id(item_id, prepared.image.image_id)
            if source_record_id in prepared_by_record:
                raise ABOProvenanceError("ABO source record identity must be unique")
            prepared_by_record[source_record_id] = prepared
            drafts.append(
                DatasetAssetDraft(
                    source_dataset=SOURCE_DATASET,
                    source_revision=lock.source_revision,
                    source_record_id=source_record_id,
                    transform_policy_version=TRANSFORM_POLICY_VERSION,
                    local_path=prepared.local_path,
                    product_id=_product_id(item_id),
                    derivation_parent_asset_ids=[],
                    license_id=lock.license_id,
                    source_url=prepared.receipt_image.official_image_uri,
                    attribution=lock.attribution,
                    cloud_upload_allowed=lock.permissions.cloud_upload_allowed,
                    public_demo_allowed=lock.permissions.public_demo_allowed,
                )
            )
    drafts.sort(key=lambda value: (value.product_id or "", value.local_path))
    return drafts, prepared_by_record


def _choose_local_provisional_pairs(
    prepared_by_item: dict[
        str, tuple[_PreparedAsset | None, tuple[_PreparedAsset, ...]]
    ],
    catalog: AssetCatalog,
    item_reasons: defaultdict[str, set[str]],
) -> list[_PreparedPair]:
    chosen: list[_PreparedPair] = []
    for item_id in sorted(prepared_by_item):
        main, others = prepared_by_item[item_id]
        if main is None or not others:
            continue
        main_asset = catalog.resolve_path(main.local_path).asset
        individually_distinct = False
        locally_selected = False
        for other in others:
            other_asset = catalog.resolve_path(other.local_path).asset
            individual_report = audit_query_gallery_eligibility(
                [
                    QueryAssetReference(
                        query_id=f"abo-exact:{item_id}",
                        intent="exact_match",
                        asset_id=main_asset.asset_id,
                    )
                ],
                [other_asset.asset_id],
                catalog,
            )
            if individual_report.violation_count:
                continue
            individually_distinct = True
            pair = _PreparedPair(item_id=item_id, main=main, other=other)
            trial = [*chosen, pair]
            local_report = audit_query_gallery_eligibility(
                [
                    QueryAssetReference(
                        query_id=f"abo-exact:{candidate.item_id}",
                        intent="exact_match",
                        asset_id=catalog.resolve_path(
                            candidate.main.local_path
                        ).asset.asset_id,
                    )
                    for candidate in trial
                ],
                [
                    catalog.resolve_path(candidate.other.local_path).asset.asset_id
                    for candidate in trial
                ],
                catalog,
            )
            if not local_report.violation_count:
                chosen.append(pair)
                locally_selected = True
                break
        if locally_selected:
            continue
        if not individually_distinct:
            item_reasons["near_duplicate_or_no_distinct_gallery_view"].add(item_id)
        else:
            item_reasons["cross_item_near_duplicate"].add(item_id)
    return chosen


def _build_coverage_audit(
    *,
    bundle: VerifiedABOSourceBundle,
    eligible_pairs: list[_PreparedPair],
    selected_pairs: list[_PreparedPair],
    item_reasons: defaultdict[str, set[str]],
    asset_reasons: defaultdict[str, set[str]],
    minimum_pair_candidates: int,
    maximum_pair_candidates: int,
) -> ABOExactCoverageAudit:
    locally_viable_ids = {pair.item_id for pair in eligible_pairs}
    selected_by_item = {pair.item_id: pair for pair in selected_pairs}
    selected_ids = set(selected_by_item)
    all_item_ids = {listing.item_id for listing in bundle.listings}
    excluded_item_ids = all_item_ids.difference(selected_ids)
    for item_id in locally_viable_ids.difference(selected_ids):
        item_reasons["pair_limit"].add(item_id)
    normalized_item_reasons = {
        reason: tuple(sorted(item_ids.intersection(excluded_item_ids)))
        for reason, item_ids in sorted(item_reasons.items())
        if item_ids.intersection(excluded_item_ids)
    }
    reason_by_asset = {
        asset_id: reason
        for reason, asset_ids in sorted(asset_reasons.items())
        for asset_id in sorted(asset_ids)
    }
    dispositions: list[ABOAssetDisposition] = []
    for listing in bundle.listings:
        selected = selected_by_item.get(listing.item_id)
        for image_id in (listing.main_image_id, *listing.other_image_id):
            asset_key = f"{listing.item_id}/{image_id}"
            if selected is not None and image_id == selected.main.image.image_id:
                disposition = "provisional_candidate"
                reason = "provisional_query_candidate"
            elif selected is not None and image_id == selected.other.image.image_id:
                disposition = "provisional_candidate"
                reason = "provisional_gallery_candidate"
            else:
                disposition = "excluded"
                reason = reason_by_asset.get(asset_key)
                if reason is None and listing.item_id in locally_viable_ids:
                    reason = (
                        "pair_limit"
                        if listing.item_id not in selected_ids
                        else "not_selected_view"
                    )
                if reason is None and listing.item_id in item_reasons.get(
                    "cross_item_near_duplicate", set()
                ):
                    reason = "local_cross_pair_duplicate"
                if reason is None:
                    reason = "local_near_duplicate"
            dispositions.append(
                ABOAssetDisposition(
                    asset_key=asset_key,
                    item_id=listing.item_id,
                    image_id=image_id,
                    disposition=disposition,
                    reason=reason,
                )
            )
    dispositions.sort(key=lambda value: value.asset_key)
    excluded_asset_count = sum(
        value.disposition == "excluded" for value in dispositions
    )
    return ABOExactCoverageAudit(
        candidate_item_count=len(bundle.listings),
        candidate_asset_count=len(dispositions),
        provisional_pair_item_count=len(selected_pairs),
        provisional_pair_asset_count=2 * len(selected_pairs),
        excluded_item_count=len(excluded_item_ids),
        excluded_asset_count=excluded_asset_count,
        minimum_pair_candidates=minimum_pair_candidates,
        maximum_pair_candidates=maximum_pair_candidates,
        excluded_items_by_reason=normalized_item_reasons,
        asset_dispositions=tuple(dispositions),
        passed_provisional_gate=len(selected_pairs) >= minimum_pair_candidates,
    )


def _build_exact_pairs(
    selected: list[_PreparedPair], catalog: AssetCatalog
) -> tuple[ABOExactPair, ...]:
    result = []
    for pair in selected:
        query = catalog.resolve_path(pair.main.local_path).asset
        gallery = catalog.resolve_path(pair.other.local_path).asset
        result.append(
            ABOExactPair(
                item_id=pair.item_id,
                product_id=_product_id(pair.item_id),
                query_image_id=pair.main.image.image_id,
                query_asset_id=query.asset_id,
                query_local_path=query.local_path,
                gallery_image_id=pair.other.image.image_id,
                gallery_asset_id=gallery.asset_id,
                gallery_local_path=gallery.local_path,
            )
        )
    return tuple(result)


def _build_provenance(
    prepared: _PreparedAsset,
    bundle: VerifiedABOSourceBundle,
    review_ledger_sha256: str,
) -> ABOAssetProvenance:
    item_receipt = next(
        item for item in bundle.receipt.items if item.item_id == prepared.item.item_id
    )
    image_receipt = prepared.receipt_image
    return ABOAssetProvenance(
        source_revision=bundle.lock.source_revision,
        archive_identity=bundle.lock.archive_identity,
        item_id=prepared.item.item_id,
        product_id=_product_id(prepared.item.item_id),
        image_id=prepared.image.image_id,
        image_role=prepared.role,
        source_record_id=_source_record_id(
            prepared.item.item_id, prepared.image.image_id
        ),
        listing_record_sha256=prepared.listing_record_sha256,
        image_record_sha256=prepared.image_record_sha256,
        acquisition_receipt_sha256=bundle.receipt_sha256,
        listing_artifact_sha256=bundle.receipt.listing_artifact.raw_sha256,
        listing_artifact_etag=bundle.receipt.listing_artifact.etag,
        listing_line_number=item_receipt.listing_locator.decompressed_line_number,
        raw_listing_record_sha256=item_receipt.listing_locator.record_bytes_sha256,
        image_metadata_artifact_sha256=(
            bundle.receipt.image_metadata_artifact.raw_sha256
        ),
        image_metadata_artifact_etag=bundle.receipt.image_metadata_artifact.etag,
        image_metadata_line_number=(
            image_receipt.metadata_locator.decompressed_line_number
        ),
        raw_image_metadata_record_sha256=(
            image_receipt.metadata_locator.record_bytes_sha256
        ),
        source_image_sha256=prepared.snapshot.sha256,
        final_image_sha256=prepared.snapshot.sha256,
        source_collection="official_original",
        derivation_kind="original",
        source_lock_sha256=bundle.lock_sha256,
        review_ledger_sha256=review_ledger_sha256,
        review_decision="approve_catalog_product_photo",
        reviewer_kind=prepared.review.reviewer_kind,
        reviewer_id=prepared.review.reviewer_id,
        reviewed_at=prepared.review.reviewed_at,
        direct_license_sha256=bundle.lock.direct_license_sha256,
        registry_license_sha256=bundle.lock.registry_license_sha256,
        official_image_uri=image_receipt.official_image_uri,
        official_image_etag=image_receipt.image_etag,
        attribution=bundle.lock.attribution,
        permissions=bundle.lock.permissions,
    )


def _build_review_packet(
    selected: list[_PreparedPair],
) -> tuple[ABOReviewPacketEntry, ...]:
    entries = []
    for pair in selected:
        for prepared in (pair.main, pair.other):
            entries.append(
                ABOReviewPacketEntry(
                    item_id=prepared.item.item_id,
                    image_id=prepared.image.image_id,
                    image_role=prepared.role,
                    local_path=prepared.local_path,
                    official_image_uri=prepared.receipt_image.official_image_uri,
                    official_image_etag=prepared.receipt_image.image_etag,
                    source_image_sha256=prepared.snapshot.sha256,
                    listing_record_sha256=prepared.listing_record_sha256,
                    image_record_sha256=prepared.image_record_sha256,
                )
            )
    return tuple(sorted(entries, key=lambda value: (value.item_id, value.image_id)))


def _build_candidate_bundle_manifest(
    *,
    bundle: VerifiedABOSourceBundle,
    source_approval: VerifiedABOSourceApproval,
    review_ledger_sha256: str,
    drafts: list[DatasetAssetDraft],
    catalog_sha256: str,
    draft_bytes: bytes,
    provenance_bytes: bytes,
    pair_bytes: bytes,
    audit_bytes: bytes,
    review_packet_bytes: bytes,
    staging: Path,
) -> ABOCandidateBundleManifest:
    files = _bundle_file_descriptors(staging)
    review_record_sha256 = sha256_bytes(
        canonical_json_bytes(source_approval.review_record)
    )
    unsigned = {
        "schema_version": 1,
        "bundle_policy_version": "abo-exact-candidate-bundle-v2",
        "status": "provisional_requires_cross_source_global_catalog",
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "required_source_lock_sha256": (
            source_approval.required_source_lock.lock_file_sha256
        ),
        "source_review_policy_sha256": source_approval.policy_file_sha256,
        "source_review_ledger_sha256": source_approval.ledger_file_sha256,
        "source_review_record_sha256": review_record_sha256,
        "source_review_license_evidence_sha256": (
            source_approval.license_evidence_sha256
        ),
        "source_lock_sha256": bundle.lock_sha256,
        "acquisition_receipt_sha256": bundle.receipt_sha256,
        "listings_sha256": bundle.listings_snapshot.sha256,
        "image_manifest_sha256": bundle.image_manifest_snapshot.sha256,
        "direct_license_sha256": bundle.direct_license_snapshot.sha256,
        "registry_license_sha256": bundle.registry_license_snapshot.sha256,
        "review_ledger_sha256": review_ledger_sha256,
        "dataset_assets_sha256": sha256_bytes(draft_bytes),
        "provenance_sha256": sha256_bytes(provenance_bytes),
        "exact_pairs_sha256": sha256_bytes(pair_bytes),
        "coverage_audit_sha256": sha256_bytes(audit_bytes),
        "review_packet_sha256": sha256_bytes(review_packet_bytes),
        "provisional_asset_catalog_sha256": catalog_sha256,
        "permissions": bundle.lock.permissions.model_dump(mode="json"),
        "provisional_pair_count": len(drafts) // 2,
        "provisional_asset_count": len(drafts),
        "files": [value.model_dump(mode="json") for value in files],
    }
    return ABOCandidateBundleManifest.model_validate(
        {
            **unsigned,
            "bundle_self_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )


def _bundle_file_descriptors(staging: Path) -> tuple[ABOBundleFile, ...]:
    descriptors = []
    for path in sorted(staging.rglob("*")):
        metadata = path.lstat()
        if _is_symlink_or_reparse(metadata):
            raise ABOProvenanceError("ABO staged bundle contains a reparse path")
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_directory(path, "ABO staged bundle directory")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ABOProvenanceError("ABO staged bundle contains a special file")
        snapshot = _snapshot_regular_file(path, "ABO staged bundle payload")
        relative = path.relative_to(staging).as_posix()
        if relative == _BUNDLE_FILE:
            continue
        descriptors.append(
            ABOBundleFile(
                path=relative,
                bytes=len(snapshot.content),
                sha256=snapshot.sha256,
            )
        )
    return tuple(descriptors)


def _load_review_ledger(
    path: Path, expected_sha256: str
) -> tuple[_FileSnapshot, tuple[ABOImageReviewDecision, ...]]:
    _validate_expected_digest(expected_sha256, "review-ledger")
    snapshot = _snapshot_regular_file(path, "ABO image review ledger")
    if snapshot.sha256 != expected_sha256:
        raise ABOProvenanceError(
            "ABO review ledger does not match the independent expected digest"
        )
    records = _load_canonical_jsonl(
        snapshot, ABOImageReviewDecision, "ABO image review ledger"
    )
    _require_unique(
        (f"{record.item_id}/{record.image_id}" for record in records),
        "ABO review decision identity",
    )
    _verify_snapshot(snapshot, "ABO image review ledger")
    return snapshot, records


def _load_canonical_json_model(
    snapshot: _FileSnapshot, model_type: type[TModel], label: str
) -> TModel:
    try:
        raw = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        model = model_type.model_validate(raw, strict=True)
    except Exception as error:
        raise ABOProvenanceError(f"{label} violates its strict schema") from error
    if snapshot.content != canonical_json_bytes(model):
        raise ABOProvenanceError(f"{label} must be canonical JSON")
    return model


def _load_canonical_jsonl(
    snapshot: _FileSnapshot, model_type: type[TModel], label: str
) -> tuple[TModel, ...]:
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ABOProvenanceError(f"{label} must be UTF-8") from error
    if not text or not text.endswith("\n"):
        raise ABOProvenanceError(f"{label} must be non-empty newline-terminated JSONL")
    records: list[TModel] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ABOProvenanceError(f"{label} contains a blank row")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            record = model_type.model_validate(raw, strict=True)
        except Exception as error:
            raise ABOProvenanceError(
                f"{label} row {line_number} violates its strict schema"
            ) from error
        records.append(record)
    if snapshot.content != canonical_jsonl_bytes(records):
        raise ABOProvenanceError(f"{label} must be canonical JSONL")
    return tuple(records)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ABOProvenanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ABOProvenanceError(f"non-finite JSON constant is forbidden: {value}")


def _verify_license_evidence(
    content: bytes, *, required_tokens: str, label: str
) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ABOProvenanceError(f"{label} must be UTF-8 text") from error
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    if required_tokens not in normalized:
        raise ABOProvenanceError(
            f"{label} does not contain the licence named by the source lock"
        )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _is_symlink_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _snapshot_directory_chain(
    path: Path, label: str
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    path = Path(path).absolute()
    chain = [path, *path.parents]
    chain.reverse()
    result = []
    for member in chain:
        try:
            metadata = member.lstat()
        except OSError as error:
            raise ABOProvenanceError(
                f"unable to inspect ancestor of {label}"
            ) from error
        if _is_symlink_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ABOProvenanceError(
                f"{label} ancestor must be a real non-reparse directory: {member}"
            )
        result.append((str(member), _directory_identity(metadata)))
    return tuple(result)


def _snapshot_regular_file(path: Path, label: str) -> _FileSnapshot:
    path = Path(path).absolute()
    ancestors_before = _snapshot_directory_chain(path.parent, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise ABOProvenanceError(f"unable to inspect {label}") from error
    if _is_symlink_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ABOProvenanceError(
            f"{label} must be a regular non-symlink/non-reparse file"
        )
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise ABOProvenanceError(f"{label} changed before it could be read")
            content = source.read()
            after_open = os.fstat(source.fileno())
    except OSError as error:
        raise ABOProvenanceError(f"unable to read {label}") from error
    if _file_identity(opened) != _file_identity(after_open):
        raise ABOProvenanceError(f"{label} changed while it was read")
    try:
        after_path = path.lstat()
    except OSError as error:
        raise ABOProvenanceError(f"{label} changed after it was read") from error
    if _file_identity(after_open) != _file_identity(after_path):
        raise ABOProvenanceError(f"{label} changed while it was read")
    ancestors_after = _snapshot_directory_chain(path.parent, label)
    if ancestors_before != ancestors_after:
        raise ABOProvenanceError(f"{label} ancestor chain changed while it was read")
    return _FileSnapshot(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        identity=_file_identity(after_open),
        ancestor_identities=ancestors_after,
    )


def _verify_snapshot(snapshot: _FileSnapshot, label: str) -> None:
    current = _snapshot_regular_file(snapshot.path, label)
    if (
        current.identity != snapshot.identity
        or current.ancestor_identities != snapshot.ancestor_identities
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise ABOProvenanceError(f"{label} changed during ABO processing")


def _require_real_directory(path: Path, label: str) -> None:
    _snapshot_directory_chain(Path(path), label)


def _ensure_real_directory(path: Path, label: str) -> None:
    path = Path(path).absolute()
    missing = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise ABOProvenanceError(f"unable to find existing ancestor for {label}")
        current = parent
    _require_real_directory(current, label)
    for member in reversed(missing):
        try:
            member.mkdir()
        except FileExistsError:
            pass
        _require_real_directory(member, label)


def _canonical_relative_path(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    if "\\" in value:
        raise ValueError(f"{field_name} must use POSIX separators")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{field_name} must be a canonical relative path")
    if pure.as_posix() != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


def _canonical_metadata_image_path(value: str) -> str:
    value = _canonical_relative_path(value, "metadata image path")
    pure = PurePosixPath(value)
    forbidden = _PROHIBITED_PATH_PARTS | {"images", "original", "small"}
    if any(part.casefold() in forbidden for part in pure.parts):
        raise ValueError(
            "metadata path must not name an image collection or derivation"
        )
    if pure.suffix.casefold() not in _IMAGE_SUFFIXES:
        raise ValueError("metadata image path must use a supported raster suffix")
    return value


def _cache_file(root: Path, relative: str) -> Path:
    root = Path(root).absolute()
    _require_real_directory(root, "ABO cache root")
    relative = _canonical_relative_path(relative, "ABO cache relative path")
    return root.joinpath(*PurePosixPath(relative).parts)


def _parse_exact_https_uri(value: str):
    value = _nonblank(value, "official URI")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("official URI must use HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("official URI must not contain credentials")
    return value, parsed


def _official_s3_key(value: str) -> str:
    value, parsed = _parse_exact_https_uri(value)
    host = parsed.hostname.casefold()
    path = parsed.path.lstrip("/")
    if host in {
        "amazon-berkeley-objects.s3.amazonaws.com",
        "amazon-berkeley-objects.s3.us-east-1.amazonaws.com",
    }:
        key = path
    elif host == "s3.us-east-1.amazonaws.com":
        prefix = "amazon-berkeley-objects/"
        if not path.startswith(prefix):
            raise ValueError("path-style S3 URI must use the exact ABO bucket")
        key = path.removeprefix(prefix)
    else:
        raise ValueError("URI host is not an exact official ABO S3 endpoint")
    if not key or ".." in PurePosixPath(key).parts:
        raise ValueError("official ABO S3 key is invalid")
    return key


def _validate_official_dataset_uri(value: str) -> str:
    _official_s3_key(value)
    return value


def _validate_official_raw_artifact_uri(value: str, artifact_kind: str) -> str:
    key = _official_s3_key(value)
    if artifact_kind == "listing_shard":
        if re.fullmatch(r"listings/metadata/listings_[0-9a-f]\.json\.gz", key) is None:
            raise ValueError("listing shard URI must name official listings metadata")
    elif artifact_kind == "image_metadata":
        if key != "images/metadata/images.csv.gz":
            raise ValueError("image metadata URI must name official images.csv.gz")
    else:
        raise ValueError("unknown ABO raw artifact kind")
    return value


def _validate_official_original_image_uri(value: str, metadata_path: str) -> str:
    metadata_path = _canonical_metadata_image_path(metadata_path)
    key = _official_s3_key(value)
    expected = f"images/original/{metadata_path}"
    if key != expected:
        raise ValueError("official image URI must exactly name images/original path")
    return value


def _validate_official_license_uri(value: str) -> str:
    _, parsed = _parse_exact_https_uri(value)
    host = parsed.hostname.casefold()
    if host in {
        "amazon-berkeley-objects.s3.amazonaws.com",
        "amazon-berkeley-objects.s3.us-east-1.amazonaws.com",
        "s3.us-east-1.amazonaws.com",
    }:
        key = _official_s3_key(value)
        if key != "LICENSE-CC-BY-4.0.txt":
            raise ValueError("official S3 licence URI must name LICENSE-CC-BY-4.0.txt")
        return value
    parts = [part for part in parsed.path.split("/") if part]
    commit_pattern = re.compile(r"^[0-9a-fA-F]{40}$")
    if host == "raw.githubusercontent.com":
        valid = (
            len(parts) == 4
            and parts[0] == "amazon-science"
            and parts[1] == "amazon-berkeley-objects"
            and commit_pattern.fullmatch(parts[2]) is not None
            and parts[-1].casefold() in {"license", "license.txt"}
        )
    elif host == "github.com":
        valid = (
            len(parts) == 5
            and parts[0] == "amazon-science"
            and parts[1] == "amazon-berkeley-objects"
            and parts[2] == "blob"
            and commit_pattern.fullmatch(parts[3]) is not None
            and parts[4].casefold() in {"license", "license.txt"}
        )
    else:
        valid = False
    if not valid:
        raise ValueError("GitHub licence URI must pin exact owner/repo/commit/LICENSE")
    return value


def _validate_registry_uri(value: str) -> str:
    _, parsed = _parse_exact_https_uri(value)
    if (
        parsed.hostname.casefold() != "registry.opendata.aws"
        or parsed.path.rstrip("/") != "/amazon-berkeley-objects"
    ):
        raise ValueError("registry licence URI must be the exact ABO registry page")
    return value


def _response_header(headers: Any, name: str) -> str:
    for key, value in headers.items():
        if str(key).casefold() == name.casefold():
            return str(value)
    raise ABOProvenanceError(f"ABO response is missing required {name} header")


def _validate_expected_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ABOProvenanceError(
            f"formal ABO use requires an independent expected {label} SHA-256"
        )


def _require_unique(values, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ABOProvenanceError(f"duplicate {label}: {value}")
        seen.add(value)


def _has_prohibited_path(value: str) -> bool:
    return any(
        part.casefold() in _PROHIBITED_PATH_PARTS for part in PurePosixPath(value).parts
    )


def _source_record_id(item_id: str, image_id: str) -> str:
    return f"listing:{item_id}/image:{image_id}"


def _product_id(item_id: str) -> str:
    return f"abo:{item_id}"


def _verify_bundle_unchanged(bundle: VerifiedABOSourceBundle) -> None:
    for snapshot, label in (
        (bundle.lock_snapshot, "ABO source lock"),
        (bundle.receipt_snapshot, "ABO acquisition receipt"),
        (bundle.listing_raw_snapshot, "ABO raw listing shard"),
        (bundle.image_metadata_raw_snapshot, "ABO raw images.csv.gz"),
        (bundle.listings_snapshot, "ABO listings snapshot"),
        (bundle.image_manifest_snapshot, "ABO image manifest"),
        (bundle.direct_license_snapshot, "ABO direct licence evidence"),
        (bundle.registry_license_snapshot, "ABO registry licence evidence"),
    ):
        _verify_snapshot(snapshot, label)


__all__ = [
    "ABO_FORMAL_PERMISSIONS",
    "ABO_FORMAL_PURPOSES",
    "ABOAcquisitionReceipt",
    "ABOBuildResult",
    "ABOCandidateBundleManifest",
    "ABOExactCoverageAudit",
    "ABOExactCoverageError",
    "ABOGlobalCatalogScope",
    "ABOGlobalFinalizationError",
    "ABOGlobalFinalizationManifest",
    "ABOImageManifestRecord",
    "ABOImageReviewDecision",
    "ABOItemAcquisitionReceipt",
    "ABOListingRecord",
    "ABOOfficialImageReceipt",
    "ABOPermissionMatrix",
    "ABOProvenanceError",
    "ABORawArtifactReceipt",
    "ABORawRecordLocator",
    "ABOReviewPacketEntry",
    "ABOSourceLock",
    "LICENSE_ID",
    "REVIEW_POLICY_VERSION",
    "SELECTION_POLICY_VERSION",
    "SOURCE_SNAPSHOT_FORMAT",
    "TRANSFORM_POLICY_VERSION",
    "VerifiedABOCandidateBundle",
    "VerifiedABOGlobalFinalization",
    "VerifiedABOSourceApproval",
    "VerifiedABOSourceBundle",
    "acquire_targeted_abo_original_images",
    "audit_abo_exact_coverage",
    "build_abo_exact_mini",
    "create_abo_candidate_review_packet",
    "finalize_abo_exact_candidates",
    "load_verified_abo_candidate_bundle",
    "load_verified_abo_global_finalization",
    "load_verified_abo_source_approval",
    "load_verified_abo_source_bundle",
]
