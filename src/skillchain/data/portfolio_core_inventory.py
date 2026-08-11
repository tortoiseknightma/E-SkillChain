"""Deterministic Portfolio core inventory for already-regular image sources.

This builder is intentionally separate from archive adapters.  It verifies
regular files that are already present, selects fixed content-unique quotas,
and emits the strict :class:`PortfolioCoreAssetCandidate` JSONL consumed by
``portfolio_core_assets``.  It never extracts archives or composes corpus
text, except for the explicit ISIA selected-member extension, which reads and
materializes only its deterministic carry-forward/fresh/reserve members.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import mmap
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Literal, Mapping, Sequence
import zipfile

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from skillchain.data import isia_food500 as isia_food500_source
from skillchain.data import sroie as sroie_source
from skillchain.data.asset_catalog import DatasetAssetDraft, load_asset_catalog
from skillchain.data.fashioniq import (
    FashionIQProvenanceError,
    load_verified_fashioniq_adapter_bundle,
)
from skillchain.data.portfolio_core_assets import (
    CORE_R2_DOCUMENT_CARRY_FORWARD_FLOOR,
    CORE_R2_DOCUMENT_CORD_FRESH_FLOOR,
    CORE_R2_DOCUMENT_CORD_RESERVE_FLOOR,
    CORE_R2_DOCUMENT_FRESH_FLOOR,
    CORE_R2_DOCUMENT_SROIE_FRESH_FLOOR,
    CORE_R2_DOCUMENT_SROIE_RESERVE_FLOOR,
    CORE_POOL_ASSET_FLOORS,
    CandidateCapabilityBinding,
    PortfolioCoreAssetCandidate,
    PortfolioCoreDocumentSelectionEntry,
    PortfolioCoreDocumentSelectionManifest,
    PortfolioCoreAssetError,
    document_component_key,
    load_candidate_inventory,
)
from skillchain.data.portfolio_mini import (
    PortfolioMiniError,
    load_portfolio_mini_drafts,
    load_portfolio_mini_selection_manifest,
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
    parse_strict_json,
    read_stable_regular_file,
)


REGULAR_CORE_INVENTORY_POLICY_VERSION = "portfolio-core-regular-inventory-v1"
REGULAR_SOURCE_QUOTAS: dict[str, int] = {
    "fashioniq": 130,
    "inaturalist": 122,
    "isia_food500": 116,
    "wikimedia_commons_documents": 30,
}
R2_DOCUMENT_CARRY_FORWARD_COUNT = CORE_R2_DOCUMENT_CARRY_FORWARD_FLOOR
R2_DOCUMENT_FRESH_COUNT = CORE_R2_DOCUMENT_FRESH_FLOOR
R2_DOCUMENT_RESERVE_COUNT = 5
R2_DOCUMENT_CORD_FRESH_COUNT = CORE_R2_DOCUMENT_CORD_FRESH_FLOOR
R2_DOCUMENT_SROIE_FRESH_COUNT = CORE_R2_DOCUMENT_SROIE_FRESH_FLOOR
R2_DOCUMENT_CORD_RESERVE_COUNT = CORE_R2_DOCUMENT_CORD_RESERVE_FLOOR
R2_DOCUMENT_SROIE_RESERVE_COUNT = CORE_R2_DOCUMENT_SROIE_RESERVE_FLOOR
R2_RECIPE_CARRY_FORWARD_COUNT = REGULAR_SOURCE_QUOTAS["isia_food500"]
R2_RECIPE_TARGET_COUNT = 210
R2_RECIPE_RESERVE_COUNT = 5
ISIA_RECIPE_EXTENSION_POLICY_VERSION = "portfolio-core-isia-carry-forward-v1"
DOCUMENT_EXTENSION_POLICY_VERSION = "portfolio-core-document-multi-source-v2"
PORTFOLIO_CORE_R2_CATALOG_POLICY_VERSION = "portfolio-core-r2-catalog-v1"
R2_ABO_ADDITIONAL_CROSS_INTENT_COUNT = 5
CORD_V2_FIXED_COMMIT = "7f0115a4b758a71d6473b8d085751692da2fef98"
SROIE_IMAGE_ONLY_ROLE = "task3_test_images"

_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

_SOURCE_BINDINGS: dict[str, tuple[str, str, str]] = {
    "cord": (
        "utility",
        "utility",
        "utility.document_reading",
    ),
    "fashioniq": (
        "divergent_rec",
        "divergent_rec",
        "product.style_recommendation",
    ),
    "inaturalist": (
        "encyclopedia",
        "encyclopedia",
        "knowledge.visual_encyclopedia",
    ),
    "isia_food500": (
        "utility",
        "utility",
        "utility.recipe_guidance",
    ),
    "wikimedia_commons_documents": (
        "utility",
        "utility",
        "utility.document_reading",
    ),
    "recipe1m_plus": (
        "utility",
        "utility",
        "utility.recipe_guidance",
    ),
    "sroie": (
        "utility",
        "utility",
        "utility.document_reading",
    ),
}


class PortfolioCoreInventoryError(ValueError):
    """A regular-source manifest or deterministic selection is inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _INaturalistRow(_StrictModel):
    observation_id: int
    photo_id: int
    scientific_name: str
    common_name: str | None
    iconic_taxon: str
    license: str
    license_url: str
    attribution: str
    observation_url: str
    api_photo_url: str
    original_url: str
    source_url: str
    image: str
    cloud_upload_allowed: bool | None = True
    public_demo_allowed: bool = True


class _ISIAFoodRow(_StrictModel):
    image: str
    category: str
    source: str
    source_member: str
    source_page: str
    source_archive: str
    source_archive_sha256: str
    source_volume_index: int
    license: str
    license_note: str
    distribution: str
    retrieved_at: str
    cloud_upload_allowed: bool | None = True
    public_demo_allowed: bool = True


class _CommonsDocumentRow(_StrictModel):
    page_id: int
    title: str
    document_category: str
    width: int
    height: int
    mime: str
    license: str
    license_url: str | None
    attribution: str
    description_url: str
    original_url: str
    source_url: str
    image: str
    source: str
    cloud_upload_allowed: bool | None = True
    public_demo_allowed: bool = True


RecipeSelectionRole = Literal["carry_forward", "fresh", "reserve"]


class ISIARecipeSelectionEntry(_StrictModel):
    """One source-member binding in a create-only ISIA r2 extension."""

    image_path: str
    source_record_id: str
    content_sha256: str
    component_fingerprint: str
    component_key: str
    selection_role: RecipeSelectionRole

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        return _validate_safe_image_path(value, "image_path")

    @field_validator("source_record_id")
    @classmethod
    def validate_source_record_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("source_record_id must be canonical non-blank text")
        path = PurePosixPath(value)
        if path.is_absolute() or "\\" in value or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError("source_record_id must be a safe POSIX member path")
        return value

    @field_validator("content_sha256", "component_key")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("component_fingerprint")
    @classmethod
    def validate_component_fingerprint(cls, value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
            raise ValueError("component_fingerprint must be a 16-character lowercase hash")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.component_key != _recipe_component_key(
            source_record_id=self.source_record_id,
            content_sha256=self.content_sha256,
            component_fingerprint=self.component_fingerprint,
        ):
            raise ValueError("component_key does not bind the recipe selection")


class ISIARecipeExtensionManifest(_StrictModel):
    """Canonical binding for an ISIA carry-forward + selected-member tail."""

    schema_version: Literal[1] = 1
    selection_policy_version: Literal[
        "portfolio-core-isia-carry-forward-v1"
    ] = ISIA_RECIPE_EXTENSION_POLICY_VERSION
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    formal_status: Literal["non_formal"] = "non_formal"
    archive_bytes: int
    archive_sha256: str
    existing_staging_manifest_sha256: str
    output_staging_manifest_sha256: str
    dev_mini_selection_sha256: str
    carry_forward_count: int
    fresh_count: int
    reserve_count: int
    selections: tuple[ISIARecipeSelectionEntry, ...]

    @field_validator("archive_bytes", "carry_forward_count", "fresh_count", "reserve_count")
    @classmethod
    def validate_counts(cls, value: int, info) -> int:
        if value < 0 or (info.field_name == "archive_bytes" and value == 0):
            raise ValueError(f"{info.field_name} must be positive/non-negative")
        return value

    @field_validator(
        "archive_sha256",
        "existing_staging_manifest_sha256",
        "output_staging_manifest_sha256",
        "dev_mini_selection_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("selections", mode="before")
    @classmethod
    def coerce_selections(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("selections")
    @classmethod
    def validate_selections(
        cls, value: tuple[ISIARecipeSelectionEntry, ...]
    ) -> tuple[ISIARecipeSelectionEntry, ...]:
        if not value:
            raise ValueError("recipe extension must contain at least one selection")
        for attribute in (
            "image_path",
            "source_record_id",
            "content_sha256",
            "component_fingerprint",
            "component_key",
        ):
            items = [getattr(row, attribute) for row in value]
            if len(items) != len(set(items)):
                raise ValueError(f"recipe extension repeats {attribute}")
        role_order = {"carry_forward": 0, "fresh": 1, "reserve": 2}
        expected = tuple(
            sorted(
                value,
                key=lambda row: (
                    role_order[row.selection_role],
                    row.source_record_id,
                    row.image_path,
                ),
            )
        )
        if value != expected:
            raise ValueError("recipe extension rows are not in canonical order")
        return value

    def model_post_init(self, __context: Any) -> None:
        counts = Counter(row.selection_role for row in self.selections)
        if counts["carry_forward"] != self.carry_forward_count:
            raise ValueError("carry_forward_count does not match selections")
        if counts["fresh"] != self.fresh_count:
            raise ValueError("fresh_count does not match selections")
        if counts["reserve"] != self.reserve_count:
            raise ValueError("reserve_count does not match selections")


@dataclass(frozen=True)
class PortfolioCoreInventoryBuildResult:
    output_path: Path
    output_sha256: str
    include_count: int
    reserve_count: int
    source_counts: dict[str, int]
    pool_unique_content_counts: dict[str, int]
    dev_mini_selection_sha256: str
    document_selection_manifest_path: Path | None = None
    document_selection_manifest_sha256: str | None = None


@dataclass(frozen=True)
class PortfolioCoreInventoryMergeResult:
    output_path: Path
    output_sha256: str
    include_count: int
    reserve_count: int
    pool_unique_content_counts: dict[str, int]


@dataclass(frozen=True)
class PortfolioCoreR2CatalogResult:
    """Immutable r2 inventory plus the audit record for added ABO bindings."""

    output_path: Path
    output_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    include_count: int
    reserve_count: int
    selected_abo_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class ISIARecipeExtensionResult:
    output_root: Path
    output_staging_manifest_path: Path
    output_staging_manifest_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    archive_sha256: str
    carry_forward_count: int
    fresh_count: int
    reserve_count: int


@dataclass(frozen=True)
class ISIARecipeCandidateBuildResult:
    output_path: Path
    output_sha256: str
    include_count: int
    reserve_count: int
    dev_mini_selection_sha256: str
    extension_manifest_sha256: str


DocumentExtensionRole = Literal[
    "carry_forward", "fresh_cord", "fresh_sroie", "reserve"
]


@dataclass(frozen=True)
class _DocumentExtensionAsset:
    draft: DatasetAssetDraft
    content: bytes
    component_fingerprint: str
    selection_role: DocumentExtensionRole


@dataclass(frozen=True)
class PortfolioCoreDocumentExtensionResult:
    output_root: Path
    staging_drafts_path: Path
    staging_drafts_sha256: str
    candidate_inventory_path: Path
    candidate_inventory_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    dev_mini_selection_sha256: str
    source_counts: dict[str, int]
    role_counts: dict[str, int]


def build_regular_portfolio_core_inventory(
    *,
    dev_mini_selection_manifest: str | Path,
    fashioniq_adapter_root: str | Path,
    fashioniq_manifest_sha256: str,
    fashioniq_asset_root: str | Path,
    inaturalist_manifest: str | Path,
    inaturalist_asset_root: str | Path,
    commons_document_manifest: str | Path,
    commons_document_asset_root: str | Path,
    isia_food_manifest: str | Path,
    isia_food_asset_root: str | Path,
    output_path: str | Path,
    isia_food_quota: int = REGULAR_SOURCE_QUOTAS["isia_food500"],
    isia_food_selection_manifest: str | Path | None = None,
    document_fresh_count: int = 0,
    document_reserve_count: int = 0,
    document_selection_manifest_path: str | Path | None = None,
    recipe1m_plus_inventory: str | Path | None = None,
    recipe1m_plus_root: str | Path | None = None,
    recipe1m_plus_license_id: str | None = None,
    recipe1m_plus_attribution: str | None = None,
) -> PortfolioCoreInventoryBuildResult:
    """Build the fixed regular-source portion of the fresh core inventory.

    The old mini selection supplies exclusion identities for FashionIQ,
    iNaturalist and ISIA Food-500.  Its 30 Wikimedia document images are the
    deliberate carry-forward.  Callers may request a deterministic fresh
    document tail and reserve; when they do, a separate canonical manifest
    records exactly which document candidates are carry-forward, fresh, and
    reserve without changing the immutable v4 candidate schema.  A fresh
    ISIA selection preserves recipe guidance.  Recipe1M+ rows, when
    explicitly supplied, are reserves only.
    """

    if isia_food_quota <= 0:
        raise PortfolioCoreInventoryError("ISIA Food-500 quota must be positive")
    if document_fresh_count < 0 or document_reserve_count < 0:
        raise PortfolioCoreInventoryError(
            "document fresh and reserve counts must be non-negative"
        )
    if document_fresh_count or document_reserve_count:
        if document_selection_manifest_path is None:
            raise PortfolioCoreInventoryError(
                "expanded document selection requires a manifest output path"
            )
    elif document_selection_manifest_path is not None:
        raise PortfolioCoreInventoryError(
            "document selection manifest is only emitted for an expanded document selection"
        )

    manifest_path = Path(dev_mini_selection_manifest)
    try:
        mini_manifest, mini_sha256 = load_portfolio_mini_selection_manifest(
            manifest_path
        )
        mini_descriptor = mini_manifest.get("dataset_assets")
        if (
            not isinstance(mini_descriptor, dict)
            or mini_descriptor.get("path") != "dataset-assets.jsonl"
        ):
            raise PortfolioCoreInventoryError(
                "dev_mini DatasetAssetDraft path binding drifted"
            )
        mini_drafts = load_portfolio_mini_drafts(
            manifest_path.parent / "dataset-assets.jsonl"
        )
    except (PortfolioMiniError, KeyError, TypeError) as error:
        raise PortfolioCoreInventoryError(
            "dev_mini selection or bound DatasetAssetDraft inventory is invalid"
        ) from error
    _verify_bound_mini_drafts(mini_manifest, mini_drafts)

    exclusions = _dev_mini_exclusions(mini_manifest["selections"])
    document_bindings = _mini_source_bindings(
        mini_manifest["selections"],
        source_id="wikimedia_commons_documents",
        expected_pool="utility_document",
        expected_capability="utility.document_reading",
        quota=REGULAR_SOURCE_QUOTAS["wikimedia_commons_documents"],
    )

    if not _SHA256_RE.fullmatch(fashioniq_manifest_sha256):
        raise PortfolioCoreInventoryError(
            "FashionIQ manifest SHA-256 must be lowercase hexadecimal"
        )
    try:
        fashion_bundle = load_verified_fashioniq_adapter_bundle(
            fashioniq_adapter_root,
            expected_manifest_sha256=fashioniq_manifest_sha256,
        )
    except FashionIQProvenanceError as error:
        raise PortfolioCoreInventoryError(
            "FashionIQ verified adapter bundle is invalid"
        ) from error
    fashion_candidates = _select_draft_candidates(
        source_id="fashioniq",
        drafts=fashion_bundle.drafts,
        source_root=Path(fashioniq_asset_root),
        quota=REGULAR_SOURCE_QUOTAS["fashioniq"],
        excluded_record_ids=exclusions["fashioniq"][0],
        excluded_content_sha256=exclusions["fashioniq"][1],
    )

    inaturalist_rows, inaturalist_manifest_sha256 = _load_strict_jsonl_models(
        Path(inaturalist_manifest),
        _INaturalistRow,
        label="iNaturalist manifest",
    )
    inaturalist_drafts = tuple(
        DatasetAssetDraft(
            source_dataset="inaturalist",
            source_revision=f"manifest-{inaturalist_manifest_sha256}",
            source_record_id=f"photo:{row.photo_id}",
            transform_policy_version="inaturalist-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.source_url,
            attribution=row.attribution,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in inaturalist_rows
    )
    inaturalist_candidates = _select_draft_candidates(
        source_id="inaturalist",
        drafts=inaturalist_drafts,
        source_root=Path(inaturalist_asset_root),
        quota=REGULAR_SOURCE_QUOTAS["inaturalist"],
        excluded_record_ids=exclusions["inaturalist"][0],
        excluded_content_sha256=exclusions["inaturalist"][1],
    )

    document_rows, commons_document_manifest_sha256 = _load_strict_jsonl_models(
        Path(commons_document_manifest),
        _CommonsDocumentRow,
        label="Wikimedia Commons document manifest",
    )
    document_drafts = tuple(
        DatasetAssetDraft(
            source_dataset="wikimedia_commons_documents",
            source_revision=f"manifest-{commons_document_manifest_sha256}",
            source_record_id=f"page:{row.page_id}",
            transform_policy_version="wikimedia-document-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.description_url,
            attribution=row.attribution,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in document_rows
    )
    document_carry_forward_candidates = _select_bound_draft_candidates(
        source_id="wikimedia_commons_documents",
        drafts=document_drafts,
        source_root=Path(commons_document_asset_root),
        record_content_bindings=document_bindings,
    )
    document_fresh_candidates: tuple[PortfolioCoreAssetCandidate, ...] = ()
    document_reserve_candidates: tuple[PortfolioCoreAssetCandidate, ...] = ()
    if document_fresh_count:
        document_fresh_candidates = _select_draft_candidates(
            source_id="wikimedia_commons_documents",
            drafts=document_drafts,
            source_root=Path(commons_document_asset_root),
            quota=document_fresh_count,
            excluded_record_ids=set(document_bindings),
            excluded_content_sha256=set(document_bindings.values()),
            destination_index_offset=len(document_carry_forward_candidates),
        )
    if document_reserve_count:
        document_reserve_candidates = _select_draft_candidates(
            source_id="wikimedia_commons_documents",
            drafts=document_drafts,
            source_root=Path(commons_document_asset_root),
            quota=document_reserve_count,
            excluded_record_ids=(
                set(document_bindings)
                | {
                    item.draft.source_record_id
                    for item in document_fresh_candidates
                }
            ),
            excluded_content_sha256=(
                set(document_bindings.values())
                | {item.expected_sha256 for item in document_fresh_candidates}
            ),
            destination_index_offset=(
                len(document_carry_forward_candidates)
                + len(document_fresh_candidates)
            ),
            selection="reserve",
        )
    document_candidates = (
        *document_carry_forward_candidates,
        *document_fresh_candidates,
        *document_reserve_candidates,
    )

    isia_rows, isia_manifest_sha256 = _load_strict_jsonl_models(
        Path(isia_food_manifest),
        _ISIAFoodRow,
        label="ISIA Food-500 manifest",
    )
    isia_drafts = tuple(
        DatasetAssetDraft(
            source_dataset="isia_food500",
            source_revision=f"archive-{row.source_archive_sha256}",
            source_record_id=row.source_member,
            transform_policy_version="isia-food500-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.source_page,
            attribution=row.license_note,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in isia_rows
    )
    document_hashes = {item.expected_sha256 for item in document_candidates}
    if isia_food_selection_manifest is None:
        isia_candidates = _select_draft_candidates(
            source_id="isia_food500",
            drafts=isia_drafts,
            source_root=Path(isia_food_asset_root),
            quota=isia_food_quota,
            excluded_record_ids=exclusions["isia_food500"][0],
            excluded_content_sha256=(
                exclusions["isia_food500"][1] | document_hashes
            ),
        )
    else:
        recipe_extension, _ = load_isia_recipe_extension_manifest(
            isia_food_selection_manifest
        )
        if recipe_extension.dev_mini_selection_sha256 != mini_sha256:
            raise PortfolioCoreInventoryError(
                "ISIA recipe extension manifest does not bind the supplied dev-mini selection"
            )
        if recipe_extension.output_staging_manifest_sha256 != isia_manifest_sha256:
            raise PortfolioCoreInventoryError(
                "ISIA recipe extension manifest does not bind the supplied staging manifest"
            )
        isia_candidates = _select_isia_extension_candidates(
            drafts=isia_drafts,
            source_root=Path(isia_food_asset_root),
            extension=recipe_extension,
            excluded_record_ids=exclusions["isia_food500"][0],
            excluded_content_sha256=(
                exclusions["isia_food500"][1] | document_hashes
            ),
        )
        if sum(item.selection == "include" for item in isia_candidates) != isia_food_quota:
            raise PortfolioCoreInventoryError(
                "ISIA recipe extension include count does not match requested quota"
            )

    recipe_candidates: tuple[PortfolioCoreAssetCandidate, ...] = ()
    recipe_values = (
        recipe1m_plus_inventory,
        recipe1m_plus_root,
        recipe1m_plus_license_id,
    )
    if any(value is not None for value in recipe_values):
        if not all(value is not None for value in recipe_values):
            raise PortfolioCoreInventoryError(
                "Recipe1M+ inventory, root and license ID must be supplied together"
            )
        recipe_candidates = _load_recipe1m_plus_reserves(
            Path(recipe1m_plus_inventory),  # type: ignore[arg-type]
            Path(recipe1m_plus_root),  # type: ignore[arg-type]
            license_id=recipe1m_plus_license_id,  # type: ignore[arg-type]
            attribution=recipe1m_plus_attribution,
        )

    candidates = tuple(
        sorted(
            (
                *fashion_candidates,
                *inaturalist_candidates,
                *document_candidates,
                *isia_candidates,
                *recipe_candidates,
            ),
            key=_candidate_sort_key,
        )
    )
    source_quotas = dict(REGULAR_SOURCE_QUOTAS)
    source_quotas["wikimedia_commons_documents"] += document_fresh_count
    source_quotas["isia_food500"] = isia_food_quota
    _validate_regular_inventory(candidates, source_quotas=source_quotas)
    output = Path(output_path).absolute()
    content = canonical_jsonl_bytes(candidates)
    _publish_idempotent(output, content)
    loaded, output_sha256 = load_candidate_inventory(output)
    if loaded != candidates:
        raise PortfolioCoreInventoryError("published regular inventory changed on reload")
    document_manifest_path: Path | None = None
    document_selection_sha256: str | None = None
    if document_selection_manifest_path is not None:
        document_manifest_path = Path(document_selection_manifest_path).absolute()
        document_manifest = PortfolioCoreDocumentSelectionManifest(
            dev_mini_selection_sha256=mini_sha256,
            source_manifest_sha256={
                "wikimedia_commons_documents": commons_document_manifest_sha256,
            },
            inventory_sha256=output_sha256,
            selections=tuple(
                [
                    _document_selection_entry(
                        item,
                        "carry_forward",
                        component_fingerprint=_document_component_fingerprint(
                            _read_verified_image(
                                Path(commons_document_asset_root),
                                item.source_local_path,
                                "Wikimedia document selection asset",
                            )
                        )[0],
                    )
                    for item in document_carry_forward_candidates
                ]
                + [
                    _document_selection_entry(
                        item,
                        "fresh",
                        component_fingerprint=_document_component_fingerprint(
                            _read_verified_image(
                                Path(commons_document_asset_root),
                                item.source_local_path,
                                "Wikimedia document selection asset",
                            )
                        )[0],
                    )
                    for item in document_fresh_candidates
                ]
                + [
                    _document_selection_entry(
                        item,
                        "reserve",
                        component_fingerprint=_document_component_fingerprint(
                            _read_verified_image(
                                Path(commons_document_asset_root),
                                item.source_local_path,
                                "Wikimedia document selection asset",
                            )
                        )[0],
                    )
                    for item in document_reserve_candidates
                ]
            ),
        )
        document_content = canonical_json_bytes(document_manifest)
        _publish_idempotent(document_manifest_path, document_content)
        document_selection_sha256 = sha256_bytes(document_content)
    included = tuple(item for item in candidates if item.selection == "include")
    return PortfolioCoreInventoryBuildResult(
        output_path=output,
        output_sha256=output_sha256,
        include_count=len(included),
        reserve_count=len(candidates) - len(included),
        source_counts=dict(
            sorted(Counter(item.source_id for item in included).items())
        ),
        pool_unique_content_counts=_unique_pool_counts(included),
        dev_mini_selection_sha256=mini_sha256,
        document_selection_manifest_path=document_manifest_path,
        document_selection_manifest_sha256=document_selection_sha256,
    )


def merge_portfolio_core_candidate_inventories(
    *,
    inventory_paths: Sequence[str | Path],
    output_path: str | Path,
    require_capacity: bool = True,
) -> PortfolioCoreInventoryMergeResult:
    """Merge independently verified candidate JSONL files deterministically."""

    if not inventory_paths:
        raise PortfolioCoreInventoryError("at least one candidate inventory is required")
    candidates: list[PortfolioCoreAssetCandidate] = []
    for path in inventory_paths:
        try:
            rows, _ = load_candidate_inventory(path)
        except PortfolioCoreAssetError as error:
            raise PortfolioCoreInventoryError(
                f"candidate inventory is invalid: {Path(path)}"
            ) from error
        candidates.extend(rows)
    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    ids = [item.candidate_id for item in ordered]
    destinations = [item.destination_path for item in ordered]
    if len(ids) != len(set(ids)):
        raise PortfolioCoreInventoryError("merged inventory has duplicate candidate_id")
    if len(destinations) != len(set(destinations)):
        raise PortfolioCoreInventoryError("merged inventory has duplicate destination_path")
    included = tuple(item for item in ordered if item.selection == "include")
    pool_counts = _unique_pool_counts(included)
    shortfalls = {
        pool: floor - pool_counts.get(pool, 0)
        for pool, floor in CORE_POOL_ASSET_FLOORS.items()
        if pool_counts.get(pool, 0) < floor
    }
    if require_capacity and shortfalls:
        raise PortfolioCoreInventoryError(
            "merged inventory does not meet core pool floors: "
            + json.dumps(shortfalls, sort_keys=True, separators=(",", ":"))
        )
    output = Path(output_path).absolute()
    content = canonical_jsonl_bytes(ordered)
    _publish_idempotent(output, content)
    loaded, output_sha256 = load_candidate_inventory(output)
    if loaded != ordered:
        raise PortfolioCoreInventoryError("published merged inventory changed on reload")
    return PortfolioCoreInventoryMergeResult(
        output_path=output,
        output_sha256=output_sha256,
        include_count=len(included),
        reserve_count=len(ordered) - len(included),
        pool_unique_content_counts=pool_counts,
    )


def build_portfolio_core_r2_candidate_catalog(
    *,
    v4_inventory_path: str | Path,
    recipe_inventory_path: str | Path,
    document_inventory_path: str | Path,
    v4_asset_catalog_dir: str | Path,
    v4_asset_root: str | Path,
    output_path: str | Path,
    selection_manifest_path: str | Path,
    superseded_v5_inventory_sha256: str | None = None,
    superseded_v5_selection_manifest_sha256: str | None = None,
) -> PortfolioCoreR2CatalogResult:
    """Build the r2 candidate catalog without changing the immutable v4 input.

    r2 replaces only the old recipe and document candidate groups.  Five
    existing ABO exact-match candidates are additionally bound to the two
    already-allowed cross intents.  The candidates are selected solely from
    v4 catalog evidence, in stable destination-path order, and must each be a
    singleton leakage component that is absent from the prior ABO triplets.
    This keeps the extra reuse both deterministic and auditable.
    """

    if (superseded_v5_inventory_sha256 is None) != (
        superseded_v5_selection_manifest_sha256 is None
    ):
        raise PortfolioCoreInventoryError(
            "both superseded v5 SHA-256 values must be supplied together"
        )
    for value in (
        superseded_v5_inventory_sha256,
        superseded_v5_selection_manifest_sha256,
    ):
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise PortfolioCoreInventoryError(
                "superseded v5 SHA-256 values must be lowercase hexadecimal"
            )

    try:
        v4_candidates, v4_inventory_sha256 = load_candidate_inventory(
            v4_inventory_path
        )
        recipe_candidates, recipe_inventory_sha256 = load_candidate_inventory(
            recipe_inventory_path
        )
        document_candidates, document_inventory_sha256 = load_candidate_inventory(
            document_inventory_path
        )
    except PortfolioCoreAssetError as error:
        raise PortfolioCoreInventoryError("r2 catalog input inventory is invalid") from error

    replaced_sources = {"isia_food500", "wikimedia_commons_documents"}
    if {candidate.source_id for candidate in recipe_candidates} != {"isia_food500"}:
        raise PortfolioCoreInventoryError(
            "r2 recipe inventory must contain only isia_food500 candidates"
        )
    expected_document_sources = {
        "wikimedia_commons_documents",
        "cord",
        "sroie",
    }
    if {candidate.source_id for candidate in document_candidates} != expected_document_sources:
        raise PortfolioCoreInventoryError(
            "r2 document inventory must contain Wikimedia, CORD, and SROIE candidates"
        )
    retained_before_rebase = tuple(
        candidate
        for candidate in v4_candidates
        if candidate.source_id not in replaced_sources
    )
    if len(retained_before_rebase) == len(v4_candidates):
        raise PortfolioCoreInventoryError(
            "v4 inventory does not contain recipe/document candidates to replace"
        )

    def rebase_to_v4_materialization(
        candidate: PortfolioCoreAssetCandidate,
    ) -> PortfolioCoreAssetCandidate:
        """Make the published v4 destination the r2 source binding.

        Some v4 rows retain their source-adapter staging path while ABO/RPC
        rows already use their destination.  r2 intentionally reuses the
        verified v4 materialization rather than rediscovering those adapters,
        so both fields must bind the published destination explicitly.
        """

        if (
            candidate.source_local_path == candidate.destination_path
            and candidate.draft.local_path == candidate.destination_path
        ):
            return candidate
        value = candidate.model_dump(mode="json")
        value["source_local_path"] = candidate.destination_path
        value["draft"]["local_path"] = candidate.destination_path
        return PortfolioCoreAssetCandidate.model_validate(value, strict=True)

    retained = tuple(
        rebase_to_v4_materialization(candidate)
        for candidate in retained_before_rebase
    )
    rebased_source_ids = sorted(
        {
            candidate.source_id
            for candidate in retained_before_rebase
            if candidate.source_local_path != candidate.destination_path
            or candidate.draft.local_path != candidate.destination_path
        }
    )

    catalog = load_asset_catalog(
        v4_asset_catalog_dir,
        v4_asset_root,
        verify_files=True,
    )
    assets_by_path = {asset.local_path: asset for asset in catalog.assets}
    if len(assets_by_path) != len(catalog.assets):
        raise PortfolioCoreInventoryError("v4 asset catalog repeats local paths")
    component_member_counts = Counter(
        asset.near_duplicate_cluster_id for asset in catalog.assets
    )
    v4_hash_counts = Counter(candidate.expected_sha256 for candidate in v4_candidates)

    exact_binding = CandidateCapabilityBinding(
        canonical_intent="exact_match",
        canonical_capability="product.exact_match",
    )
    triplet_bindings = (
        CandidateCapabilityBinding(
            canonical_intent="divergent_rec",
            canonical_capability="product.style_recommendation",
        ),
        CandidateCapabilityBinding(
            canonical_intent="encyclopedia",
            canonical_capability="knowledge.visual_encyclopedia",
        ),
        exact_binding,
    )

    def component_for(candidate: PortfolioCoreAssetCandidate) -> str:
        asset = assets_by_path.get(candidate.destination_path)
        if asset is None:
            raise PortfolioCoreInventoryError(
                "v4 asset catalog does not contain ABO candidate destination: "
                + candidate.destination_path
            )
        if asset.sha256 != candidate.expected_sha256:
            raise PortfolioCoreInventoryError(
                "v4 asset catalog content binding drifted for " + candidate.candidate_id
            )
        if asset.near_duplicate_cluster_id is None:
            raise PortfolioCoreInventoryError(
                "v4 asset catalog does not assign a component to "
                + candidate.candidate_id
            )
        return asset.near_duplicate_cluster_id

    abo_candidates = tuple(
        candidate
        for candidate in retained
        if candidate.source_id == "abo" and candidate.selection == "include"
    )
    existing_cross_intent_components = {
        component_for(candidate)
        for candidate in abo_candidates
        if candidate.capability_bindings != (exact_binding,)
    }
    selected: list[tuple[PortfolioCoreAssetCandidate, str]] = []
    selected_hashes: set[str] = set()
    selected_components: set[str] = set()
    for candidate in sorted(
        abo_candidates,
        key=lambda item: (item.destination_path, item.candidate_id),
    ):
        if candidate.capability_bindings != (exact_binding,):
            continue
        if v4_hash_counts[candidate.expected_sha256] != 1:
            continue
        component_id = component_for(candidate)
        if (
            component_member_counts[component_id] != 1
            or component_id in existing_cross_intent_components
            or component_id in selected_components
            or candidate.expected_sha256 in selected_hashes
        ):
            continue
        selected.append((candidate, component_id))
        selected_components.add(component_id)
        selected_hashes.add(candidate.expected_sha256)
        if len(selected) == R2_ABO_ADDITIONAL_CROSS_INTENT_COUNT:
            break
    if len(selected) != R2_ABO_ADDITIONAL_CROSS_INTENT_COUNT:
        raise PortfolioCoreInventoryError(
            "v4 inventory cannot provide the required content/component-unique "
            "ABO cross-intent additions"
        )

    updates: dict[str, PortfolioCoreAssetCandidate] = {}
    for candidate, _component_id in selected:
        value = candidate.model_dump(mode="json")
        value["capability_bindings"] = [
            binding.model_dump(mode="json") for binding in triplet_bindings
        ]
        updates[candidate.candidate_id] = PortfolioCoreAssetCandidate.model_validate(
            value,
            strict=True,
        )
    merged = tuple(
        sorted(
            (
                *(updates.get(candidate.candidate_id, candidate) for candidate in retained),
                *recipe_candidates,
                *document_candidates,
            ),
            key=_candidate_sort_key,
        )
    )
    candidate_ids = [candidate.candidate_id for candidate in merged]
    destinations = [candidate.destination_path for candidate in merged]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise PortfolioCoreInventoryError("r2 catalog has duplicate candidate IDs")
    if len(destinations) != len(set(destinations)):
        raise PortfolioCoreInventoryError("r2 catalog has duplicate destinations")

    output = Path(output_path).absolute()
    selection_output = Path(selection_manifest_path).absolute()
    if output == selection_output:
        raise PortfolioCoreInventoryError(
            "r2 candidate output and selection manifest must be different files"
        )
    content = canonical_jsonl_bytes(merged)
    _publish_idempotent(output, content)
    loaded, output_sha256 = load_candidate_inventory(output)
    if loaded != merged:
        raise PortfolioCoreInventoryError("published r2 catalog changed on reload")

    selected_manifest_rows = [
        {
            "added_capability_bindings": [
                binding.model_dump(mode="json")
                for binding in triplet_bindings
                if binding != exact_binding
            ],
            "candidate_id": candidate.candidate_id,
            "component_id": component_id,
            "content_sha256": candidate.expected_sha256,
            "destination_path": candidate.destination_path,
            "prior_capability_bindings": [exact_binding.model_dump(mode="json")],
            "source_record_id": candidate.draft.source_record_id,
        }
        for candidate, component_id in selected
    ]
    selection_content = canonical_json_bytes(
        {
            "document_candidate_inventory_sha256": document_inventory_sha256,
            "dropped_v4_source_ids": sorted(replaced_sources),
            "formal_eligible": False,
            "formal_status": "non_formal",
            "include_count": sum(candidate.selection == "include" for candidate in merged),
            "output_inventory_sha256": output_sha256,
            "policy_version": PORTFOLIO_CORE_R2_CATALOG_POLICY_VERSION,
            "recipe_candidate_inventory_sha256": recipe_inventory_sha256,
            "reserve_count": sum(candidate.selection == "reserve" for candidate in merged),
            "schema_version": 1,
            "selected_abo_cross_intent_candidates": selected_manifest_rows,
            "selection_rule": {
                "candidate_source": "v4_abo_exact_only",
                "component_member_count": 1,
                "count": R2_ABO_ADDITIONAL_CROSS_INTENT_COUNT,
                "exclude_existing_abo_cross_intent_components": True,
                "non_dev_evidence": (
                    "v4 ABO candidates are supplied by the selected-source adapter, "
                    "which applies its bound dev_mini exclusions before selection"
                ),
                "order": "destination_path,candidate_id",
                "required_prior_binding": exact_binding.model_dump(mode="json"),
                "required_unique_content_sha256": True,
            },
            "source_counts": dict(
                sorted(Counter(candidate.source_id for candidate in merged).items())
            ),
            "superseded_v5": (
                None
                if superseded_v5_inventory_sha256 is None
                else {
                    "inventory_sha256": superseded_v5_inventory_sha256,
                    "repair_reason": (
                        "v5 source preflight could not resolve retained FashionIQ "
                        "and iNaturalist staging paths; v6 rebases those source "
                        "bindings to verified v4 materialized destinations"
                    ),
                    "selection_manifest_sha256": (
                        superseded_v5_selection_manifest_sha256
                    ),
                }
            ),
            "track": "portfolio",
            "v4_asset_catalog_sha256": catalog.manifest.catalog_sha256,
            "v4_inventory_sha256": v4_inventory_sha256,
            "v4_rebased_source_ids": rebased_source_ids,
        }
    )
    _publish_idempotent(selection_output, selection_content)
    return PortfolioCoreR2CatalogResult(
        output_path=output,
        output_sha256=output_sha256,
        selection_manifest_path=selection_output,
        selection_manifest_sha256=sha256_bytes(selection_content),
        include_count=sum(candidate.selection == "include" for candidate in merged),
        reserve_count=sum(candidate.selection == "reserve" for candidate in merged),
        selected_abo_candidate_ids=tuple(
            candidate.candidate_id for candidate, _component_id in selected
        ),
    )


@dataclass(frozen=True)
class _ISIAExtensionAsset:
    row: _ISIAFoodRow
    content: bytes
    component_fingerprint: str
    selection_role: RecipeSelectionRole


def load_isia_recipe_extension_manifest(
    path: str | Path,
) -> tuple[ISIARecipeExtensionManifest, str]:
    """Load the canonical selected-member binding for an ISIA r2 extension."""

    content = read_stable_regular_file(
        Path(path),
        label="ISIA recipe extension manifest",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        value = parse_canonical_json(
            content,
            label="ISIA recipe extension manifest",
        )
        manifest = ISIARecipeExtensionManifest.model_validate(value, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioCoreInventoryError(
            "ISIA recipe extension manifest violates schema"
        ) from error
    if content != canonical_json_bytes(manifest):
        raise PortfolioCoreInventoryError(
            "ISIA recipe extension manifest must be canonical JSON"
        )
    return manifest, sha256_bytes(content)


def materialize_isia_food500_recipe_extension(
    *,
    dev_mini_selection_manifest: str | Path,
    carry_forward_manifest: str | Path,
    carry_forward_asset_root: str | Path,
    raw_archive_path: str | Path,
    output_root: str | Path,
    include_count: int = R2_RECIPE_TARGET_COUNT,
    reserve_count: int = R2_RECIPE_RESERVE_COUNT,
    carry_forward_count: int = R2_RECIPE_CARRY_FORWARD_COUNT,
    expected_archive_sha256: str = isia_food500_source.ARCHIVE_SHA256,
    expected_archive_bytes: int = isia_food500_source.ARCHIVE_SIZE,
    min_side: int = 200,
) -> ISIARecipeExtensionResult:
    """Create a carry-forward + selected-member ISIA staging extension.

    The requested non-dev r1 bindings are copied byte-for-byte after their
    records, contents, and components are verified.  A staging manifest may
    include the 30 dev-mini rows; those are explicitly excluded before the
    carry-forward count is filled.  Only the fresh members needed to reach the
    target (plus the explicit reserve) are read from the locked ZIP.  The
    function never downloads, expands the archive tree, or rewrites an
    existing output.
    """

    if (
        include_count <= 0
        or reserve_count < 0
        or carry_forward_count <= 0
        or min_side <= 0
    ):
        raise PortfolioCoreInventoryError(
            "ISIA extension counts and minimum side must be positive/non-negative"
        )
    _validate_sha256_text(expected_archive_sha256, "expected archive SHA-256")
    if expected_archive_bytes <= 0:
        raise PortfolioCoreInventoryError("expected archive bytes must be positive")

    mini_manifest, mini_sha256, mini_exclusions = _load_dev_mini_exclusions(
        dev_mini_selection_manifest
    )
    # The explicit local variable is deliberately retained: it makes a future
    # caller unable to replace a verified mini manifest with an unbound set of
    # exclusion IDs without tripping the loader above.
    del mini_manifest

    archive = _verify_locked_isia_archive(
        Path(raw_archive_path),
        expected_sha256=expected_archive_sha256,
        expected_bytes=expected_archive_bytes,
    )
    existing_rows, existing_manifest_sha256 = _load_strict_jsonl_models(
        Path(carry_forward_manifest),
        _ISIAFoodRow,
        label="ISIA carry-forward staging manifest",
    )
    if include_count < carry_forward_count:
        raise PortfolioCoreInventoryError(
            "ISIA extension include count is below the carry-forward binding count"
        )
    if not existing_rows:
        raise PortfolioCoreInventoryError("ISIA carry-forward staging is empty")

    carry_root = Path(carry_forward_asset_root)
    carry_assets: list[_ISIAExtensionAsset] = []
    seen_records: set[str] = set()
    seen_content: set[str] = set()
    seen_components: set[str] = set()
    dev_records, dev_hashes = mini_exclusions["isia_food500"]
    for row in sorted(existing_rows, key=lambda item: (item.source_member, item.image)):
        if row.source_archive_sha256 != expected_archive_sha256:
            raise PortfolioCoreInventoryError(
                "ISIA carry-forward manifest archive SHA-256 does not match locked ZIP"
            )
        if row.source_member in dev_records:
            continue
        content = _read_verified_image(carry_root, row.image, "ISIA carry-forward asset")
        content_sha256 = sha256_bytes(content)
        if content_sha256 in dev_hashes:
            continue
        component_fingerprint = _isia_component_fingerprint(content, min_side=min_side)
        if (
            row.source_member in seen_records
            or content_sha256 in seen_content
            or component_fingerprint in seen_components
        ):
            raise PortfolioCoreInventoryError(
                "ISIA carry-forward staging repeats record, content, or component"
            )
        seen_records.add(row.source_member)
        seen_content.add(content_sha256)
        seen_components.add(component_fingerprint)
        carry_assets.append(
            _ISIAExtensionAsset(
                row=row,
                content=content,
                component_fingerprint=component_fingerprint,
                selection_role="carry_forward",
            )
        )
        if len(carry_assets) == carry_forward_count:
            break

    if len(carry_assets) != carry_forward_count:
        raise PortfolioCoreInventoryError(
            "ISIA staging cannot supply the requested non-dev carry-forward binding count"
        )

    fresh_needed = include_count - len(carry_assets)
    fresh_assets: list[_ISIAExtensionAsset] = []
    reserve_assets: list[_ISIAExtensionAsset] = []
    records = isia_food500_source.balanced_local_records(
        isia_food500_source.discover_local_records(archive)
    )
    if not records:
        raise PortfolioCoreInventoryError("locked ISIA ZIP has no valid local image records")
    template = carry_assets[0].row
    with archive.open("rb") as source, mmap.mmap(
        source.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        for record in records:
            if len(fresh_assets) == fresh_needed and len(reserve_assets) == reserve_count:
                break
            if record.name in dev_records or record.name in seen_records:
                continue
            try:
                raw = isia_food500_source.read_local_record(data, record)
                content, component_fingerprint = _normalize_isia_member(
                    raw,
                    min_side=min_side,
                )
            except (OSError, ValueError):
                continue
            content_sha256 = sha256_bytes(content)
            if (
                content_sha256 in dev_hashes
                or content_sha256 in seen_content
                or component_fingerprint in seen_components
            ):
                continue
            selection_role: RecipeSelectionRole = (
                "fresh" if len(fresh_assets) < fresh_needed else "reserve"
            )
            ordinal = (
                len(fresh_assets) + 1
                if selection_role == "fresh"
                else len(reserve_assets) + 1
            )
            image_name = (
                f"isia-r2-{selection_role}-{ordinal:04d}.jpg"
            )
            row = _ISIAFoodRow(
                image=image_name,
                category=record.category,
                source=template.source,
                source_member=record.name,
                source_page=template.source_page,
                source_archive=template.source_archive,
                source_archive_sha256=expected_archive_sha256,
                source_volume_index=template.source_volume_index,
                license=template.license,
                license_note=template.license_note,
                distribution=template.distribution,
                retrieved_at=template.retrieved_at,
                cloud_upload_allowed=template.cloud_upload_allowed,
                public_demo_allowed=template.public_demo_allowed,
            )
            asset = _ISIAExtensionAsset(
                row=row,
                content=content,
                component_fingerprint=component_fingerprint,
                selection_role=selection_role,
            )
            if selection_role == "fresh":
                fresh_assets.append(asset)
            else:
                reserve_assets.append(asset)
            seen_records.add(record.name)
            seen_content.add(content_sha256)
            seen_components.add(component_fingerprint)

    if len(fresh_assets) != fresh_needed or len(reserve_assets) != reserve_count:
        raise PortfolioCoreInventoryError(
            "locked ISIA ZIP cannot supply the requested fresh unique members: "
            + json.dumps(
                {
                    "fresh_available": len(fresh_assets),
                    "fresh_required": fresh_needed,
                    "reserve_available": len(reserve_assets),
                    "reserve_required": reserve_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    selected = tuple(
        sorted(
            (*carry_assets, *fresh_assets, *reserve_assets),
            key=lambda item: (
                {"carry_forward": 0, "fresh": 1, "reserve": 2}[item.selection_role],
                item.row.source_member,
                item.row.image,
            ),
        )
    )
    output = Path(output_root).absolute()
    _ensure_create_only_output_directory(
        output,
        forbidden_paths=(carry_root, archive),
    )
    for asset in selected:
        _publish_or_verify_recipe_extension_asset(
            output / asset.row.image,
            asset.content,
        )
    output_rows = tuple(asset.row for asset in selected)
    output_manifest_path = output / "manifest.jsonl"
    output_manifest_content = canonical_jsonl_bytes(output_rows)
    _publish_idempotent(output_manifest_path, output_manifest_content)
    output_manifest_sha256 = sha256_bytes(output_manifest_content)
    selection_manifest = ISIARecipeExtensionManifest(
        archive_bytes=expected_archive_bytes,
        archive_sha256=expected_archive_sha256,
        existing_staging_manifest_sha256=existing_manifest_sha256,
        output_staging_manifest_sha256=output_manifest_sha256,
        dev_mini_selection_sha256=mini_sha256,
        carry_forward_count=len(carry_assets),
        fresh_count=len(fresh_assets),
        reserve_count=len(reserve_assets),
        selections=tuple(
            ISIARecipeSelectionEntry(
                image_path=asset.row.image,
                source_record_id=asset.row.source_member,
                content_sha256=sha256_bytes(asset.content),
                component_fingerprint=asset.component_fingerprint,
                component_key=_recipe_component_key(
                    source_record_id=asset.row.source_member,
                    content_sha256=sha256_bytes(asset.content),
                    component_fingerprint=asset.component_fingerprint,
                ),
                selection_role=asset.selection_role,
            )
            for asset in selected
        ),
    )
    selection_manifest_path = output / "recipe-extension-manifest.json"
    selection_content = canonical_json_bytes(selection_manifest)
    _publish_idempotent(selection_manifest_path, selection_content)
    loaded_manifest, selection_manifest_sha256 = load_isia_recipe_extension_manifest(
        selection_manifest_path
    )
    if loaded_manifest != selection_manifest:
        raise PortfolioCoreInventoryError(
            "published ISIA recipe extension manifest changed on reload"
        )
    loaded_rows, loaded_output_sha256 = _load_strict_jsonl_models(
        output_manifest_path,
        _ISIAFoodRow,
        label="published ISIA recipe extension staging manifest",
    )
    if loaded_rows != output_rows or loaded_output_sha256 != output_manifest_sha256:
        raise PortfolioCoreInventoryError(
            "published ISIA recipe extension staging manifest changed on reload"
        )
    return ISIARecipeExtensionResult(
        output_root=output,
        output_staging_manifest_path=output_manifest_path,
        output_staging_manifest_sha256=output_manifest_sha256,
        selection_manifest_path=selection_manifest_path,
        selection_manifest_sha256=selection_manifest_sha256,
        archive_sha256=expected_archive_sha256,
        carry_forward_count=len(carry_assets),
        fresh_count=len(fresh_assets),
        reserve_count=len(reserve_assets),
    )


def build_isia_recipe_extension_candidates(
    *,
    dev_mini_selection_manifest: str | Path,
    staging_manifest: str | Path,
    staging_asset_root: str | Path,
    extension_manifest: str | Path,
    output_path: str | Path,
) -> ISIARecipeCandidateBuildResult:
    """Emit the 210+reserve ISIA candidate JSONL bound to an extension stage."""

    _mini_manifest, mini_sha256, exclusions = _load_dev_mini_exclusions(
        dev_mini_selection_manifest
    )
    del _mini_manifest
    extension, extension_sha256 = load_isia_recipe_extension_manifest(
        extension_manifest
    )
    if extension.dev_mini_selection_sha256 != mini_sha256:
        raise PortfolioCoreInventoryError(
            "ISIA recipe extension does not bind the supplied dev-mini selection"
        )
    rows, staging_manifest_sha256 = _load_strict_jsonl_models(
        Path(staging_manifest),
        _ISIAFoodRow,
        label="ISIA recipe extension staging manifest",
    )
    if extension.output_staging_manifest_sha256 != staging_manifest_sha256:
        raise PortfolioCoreInventoryError(
            "ISIA recipe extension does not bind the supplied staging manifest"
        )
    drafts = tuple(
        DatasetAssetDraft(
            source_dataset="isia_food500",
            source_revision=f"archive-{row.source_archive_sha256}",
            source_record_id=row.source_member,
            transform_policy_version="isia-food500-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.source_page,
            attribution=row.license_note,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in rows
    )
    candidates = _select_isia_extension_candidates(
        drafts=drafts,
        source_root=Path(staging_asset_root),
        extension=extension,
        excluded_record_ids=exclusions["isia_food500"][0],
        excluded_content_sha256=exclusions["isia_food500"][1],
    )
    include_count = sum(item.selection == "include" for item in candidates)
    reserve_count = sum(item.selection == "reserve" for item in candidates)
    if include_count != extension.carry_forward_count + extension.fresh_count:
        raise PortfolioCoreInventoryError(
            "ISIA extension candidate include count does not match its binding"
        )
    if reserve_count != extension.reserve_count:
        raise PortfolioCoreInventoryError(
            "ISIA extension candidate reserve count does not match its binding"
        )
    content = canonical_jsonl_bytes(candidates)
    output = Path(output_path).absolute()
    _publish_idempotent(output, content)
    loaded, output_sha256 = load_candidate_inventory(output)
    if loaded != candidates:
        raise PortfolioCoreInventoryError(
            "published ISIA extension candidates changed on reload"
        )
    return ISIARecipeCandidateBuildResult(
        output_path=output,
        output_sha256=output_sha256,
        include_count=include_count,
        reserve_count=reserve_count,
        dev_mini_selection_sha256=mini_sha256,
        extension_manifest_sha256=extension_sha256,
    )


def materialize_portfolio_core_document_extension(
    *,
    dev_mini_selection_manifest: str | Path,
    dev_mini_asset_root: str | Path,
    cord_root: str | Path,
    sroie_acquisition_manifest: str | Path,
    sroie_raw_root: str | Path,
    output_root: str | Path,
    cord_fresh_count: int = R2_DOCUMENT_CORD_FRESH_COUNT,
    cord_reserve_count: int = R2_DOCUMENT_CORD_RESERVE_COUNT,
    sroie_fresh_count: int = R2_DOCUMENT_SROIE_FRESH_COUNT,
    sroie_reserve_count: int = R2_DOCUMENT_SROIE_RESERVE_COUNT,
    expected_cord_commit: str = CORD_V2_FIXED_COMMIT,
) -> PortfolioCoreDocumentExtensionResult:
    """Create the r2 multi-source document staging and candidate inventory.

    The pre-existing 30 Wikimedia dev-mini images are copied byte-for-byte.
    CORD is bound to its fixed local commit and reads only the image field of
    the deterministic test Parquet shard.  SROIE is first checked with its
    existing acquisition validator and then reads only the image-only task3
    package.  Raw archives are never expanded and no label/text field is
    consumed by this adapter.
    """

    if cord_fresh_count <= 0 or cord_reserve_count < 0:
        raise PortfolioCoreInventoryError(
            "CORD fresh/reserve counts must be positive/non-negative"
        )
    if sroie_fresh_count < 0 or sroie_reserve_count < 0:
        raise PortfolioCoreInventoryError(
            "SROIE fresh/reserve counts must be non-negative"
        )
    if (sroie_fresh_count == 0) != (sroie_reserve_count == 0):
        raise PortfolioCoreInventoryError(
            "SROIE fresh and reserve counts must be both enabled or both zero"
        )
    if not isinstance(expected_cord_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", expected_cord_commit
    ):
        raise PortfolioCoreInventoryError("expected CORD commit must be lowercase SHA-1")

    mini_manifest, mini_sha256, _mini_exclusions = _load_dev_mini_exclusions(
        dev_mini_selection_manifest
    )
    try:
        mini_drafts = load_portfolio_mini_drafts(
            Path(dev_mini_selection_manifest).parent / "dataset-assets.jsonl"
        )
    except PortfolioMiniError as error:
        raise PortfolioCoreInventoryError(
            "dev_mini DatasetAssetDraft inventory is invalid"
        ) from error
    _verify_bound_mini_drafts(mini_manifest, mini_drafts)
    carry_bindings = _mini_source_bindings(
        mini_manifest["selections"],
        source_id="wikimedia_commons_documents",
        expected_pool="utility_document",
        expected_capability="utility.document_reading",
        quota=R2_DOCUMENT_CARRY_FORWARD_COUNT,
    )
    dev_root = _require_real_directory(
        Path(dev_mini_asset_root), "dev-mini document asset root"
    )
    cord_test_parquet, cord_descriptor = _validate_cord_v2_test_parquet(
        Path(cord_root), expected_commit=expected_cord_commit
    )
    sroie_manifest_value: dict[str, Any] | None = None
    sroie_manifest_sha256: str | None = None
    sroie_archive: Path | None = None
    if sroie_fresh_count:
        (
            sroie_manifest_value,
            sroie_manifest_sha256,
            sroie_archive,
        ) = _validate_sroie_image_only_source(
            Path(sroie_acquisition_manifest), Path(sroie_raw_root)
        )

    output = Path(output_root).absolute()
    forbidden_paths: list[Path] = [dev_root, Path(cord_root)]
    if sroie_fresh_count:
        forbidden_paths.append(Path(sroie_raw_root))
    _ensure_create_only_output_directory(
        output,
        forbidden_paths=tuple(forbidden_paths),
    )

    mini_documents = {
        draft.source_record_id: draft
        for draft in mini_drafts
        if draft.source_dataset == "wikimedia_commons_documents"
    }
    if len(mini_documents) != R2_DOCUMENT_CARRY_FORWARD_COUNT:
        raise PortfolioCoreInventoryError(
            "dev_mini does not expose exactly 30 Wikimedia document drafts"
        )

    assets: list[_DocumentExtensionAsset] = []
    seen_records: set[str] = set()
    seen_content: set[str] = set()
    seen_phashes: list[imagehash.ImageHash] = []
    for ordinal, (record_id, expected_sha256) in enumerate(
        sorted(carry_bindings.items()), start=1
    ):
        draft = mini_documents.get(record_id)
        if draft is None:
            raise PortfolioCoreInventoryError(
                "dev_mini document selection cannot resolve its DatasetAssetDraft"
            )
        content = _read_verified_image(
            dev_root, draft.local_path, "dev-mini Wikimedia document asset"
        )
        if sha256_bytes(content) != expected_sha256:
            raise PortfolioCoreInventoryError(
                "dev-mini Wikimedia document bytes drifted: " + record_id
            )
        staging_path = (
            "wikimedia/carry-forward-"
            f"{ordinal:04d}{PurePosixPath(draft.local_path).suffix.lower()}"
        )
        staging_draft = draft.model_copy(update={"local_path": staging_path})
        if not _try_add_document_extension_asset(
            assets=assets,
            seen_records=seen_records,
            seen_content=seen_content,
            seen_phashes=seen_phashes,
            draft=staging_draft,
            content=content,
            selection_role="carry_forward",
        ):
            raise PortfolioCoreInventoryError(
                "Wikimedia carry-forward overlaps record, content, or pHash component"
            )

    _append_cord_document_tail(
        assets=assets,
        seen_records=seen_records,
        seen_content=seen_content,
        seen_phashes=seen_phashes,
        test_parquet=cord_test_parquet,
        source_revision=f"git-{expected_cord_commit}",
        fresh_count=cord_fresh_count,
        reserve_count=cord_reserve_count,
    )
    if sroie_fresh_count:
        assert sroie_manifest_value is not None
        assert sroie_manifest_sha256 is not None
        assert sroie_archive is not None
        _append_sroie_document_tail(
            assets=assets,
            seen_records=seen_records,
            seen_content=seen_content,
            seen_phashes=seen_phashes,
            acquisition_manifest=sroie_manifest_value,
            archive=sroie_archive,
            acquisition_manifest_sha256=sroie_manifest_sha256,
            fresh_count=sroie_fresh_count,
            reserve_count=sroie_reserve_count,
        )

    expected_role_counts: dict[str, int] = {
        "carry_forward": R2_DOCUMENT_CARRY_FORWARD_COUNT,
        "fresh_cord": cord_fresh_count,
        "reserve": cord_reserve_count + sroie_reserve_count,
    }
    if sroie_fresh_count:
        expected_role_counts["fresh_sroie"] = sroie_fresh_count
    role_counts = Counter(item.selection_role for item in assets)
    if dict(role_counts) != expected_role_counts:
        raise PortfolioCoreInventoryError(
            "document extension role counts drifted: "
            + json.dumps(dict(sorted(role_counts.items())), separators=(",", ":"))
        )

    ordered_assets = tuple(
        sorted(
            assets,
            key=lambda item: (
                {
                    "carry_forward": 0,
                    "fresh_cord": 1,
                    "fresh_sroie": 2,
                    "reserve": 3,
                }[item.selection_role],
                item.draft.source_dataset,
                item.draft.source_record_id,
            ),
        )
    )
    for asset in ordered_assets:
        _publish_or_verify_document_extension_asset(
            output / Path(*PurePosixPath(asset.draft.local_path).parts), asset.content
        )

    staging_drafts = tuple(asset.draft for asset in ordered_assets)
    staging_drafts_content = canonical_jsonl_bytes(staging_drafts)
    staging_drafts_path = output / "dataset-assets.jsonl"
    _publish_idempotent(staging_drafts_path, staging_drafts_content)
    staging_drafts_sha256 = sha256_bytes(staging_drafts_content)

    candidates = tuple(
        sorted(
            (
                _candidate(
                    source_id=asset.draft.source_dataset,
                    draft=asset.draft,
                    expected_content=asset.content,
                    destination_path=_document_extension_destination_path(asset),
                    selection=(
                        "reserve"
                        if asset.selection_role == "reserve"
                        else "include"
                    ),
                )
                for asset in ordered_assets
            ),
            key=_candidate_sort_key,
        )
    )
    candidate_content = canonical_jsonl_bytes(candidates)
    candidate_path = output / "candidates.jsonl"
    _publish_idempotent(candidate_path, candidate_content)
    loaded_candidates, candidate_sha256 = load_candidate_inventory(candidate_path)
    if loaded_candidates != candidates:
        raise PortfolioCoreInventoryError(
            "published document extension candidates changed on reload"
        )

    assets_by_record = {asset.draft.source_record_id: asset for asset in ordered_assets}
    source_bindings: dict[str, object] = {
        "cord": cord_descriptor,
        "wikimedia_commons_documents": {
            "dev_mini_selection_sha256": mini_sha256,
        },
    }
    source_manifest_sha256 = {
        "cord": sha256_bytes(canonical_json_bytes(cord_descriptor)),
        "wikimedia_commons_documents": mini_sha256,
    }
    if sroie_fresh_count:
        assert sroie_manifest_value is not None
        assert sroie_manifest_sha256 is not None
        source_bindings["sroie"] = {
            "acquisition_manifest_sha256": sroie_manifest_sha256,
            "image_only_role": SROIE_IMAGE_ONLY_ROLE,
            "manifest_id": sroie_manifest_value["manifest_id"],
        }
        source_manifest_sha256["sroie"] = sroie_manifest_sha256
    source_bindings_content = canonical_json_bytes(source_bindings)
    _publish_idempotent(output / "source-bindings.json", source_bindings_content)
    source_binding_sha256 = sha256_bytes(source_bindings_content)
    document_entries = tuple(
        sorted(
            (
                _document_selection_entry(
                    candidate,
                    assets_by_record[candidate.draft.source_record_id].selection_role,
                    component_fingerprint=(
                        assets_by_record[candidate.draft.source_record_id]
                        .component_fingerprint
                    ),
                )
                for candidate in candidates
            ),
            key=lambda row: (
                {
                    "carry_forward": 0,
                    "fresh": 1,
                    "fresh_cord": 2,
                    "fresh_sroie": 3,
                    "reserve": 4,
                }[row.selection_role],
                row.source_id,
                row.source_record_id,
                row.candidate_id,
            ),
        )
    )
    document_manifest = PortfolioCoreDocumentSelectionManifest(
        dev_mini_selection_sha256=mini_sha256,
        source_manifest_sha256=source_manifest_sha256,
        inventory_sha256=candidate_sha256,
        selections=document_entries,
    )
    document_manifest_content = canonical_json_bytes(document_manifest)
    document_manifest_path = output / "document-selection-manifest.json"
    _publish_idempotent(document_manifest_path, document_manifest_content)
    document_manifest_sha256 = sha256_bytes(document_manifest_content)

    if source_binding_sha256 != sha256_bytes(
        read_stable_regular_file(
            output / "source-bindings.json",
            label="published document source bindings",
            max_bytes=_MAX_METADATA_BYTES,
        )
    ):
        raise PortfolioCoreInventoryError(
            "published document source bindings changed on reload"
        )
    source_counts = dict(
        sorted(Counter(asset.draft.source_dataset for asset in ordered_assets).items())
    )
    return PortfolioCoreDocumentExtensionResult(
        output_root=output,
        staging_drafts_path=staging_drafts_path,
        staging_drafts_sha256=staging_drafts_sha256,
        candidate_inventory_path=candidate_path,
        candidate_inventory_sha256=candidate_sha256,
        selection_manifest_path=document_manifest_path,
        selection_manifest_sha256=document_manifest_sha256,
        dev_mini_selection_sha256=mini_sha256,
        source_counts=source_counts,
        role_counts=dict(sorted(role_counts.items())),
    )


def _verify_bound_mini_drafts(
    manifest: Mapping[str, Any], drafts: Sequence[DatasetAssetDraft]
) -> None:
    descriptor = manifest.get("dataset_assets")
    if not isinstance(descriptor, dict):
        raise PortfolioCoreInventoryError("dev_mini dataset-assets binding is missing")
    content = canonical_jsonl_bytes(drafts)
    if (
        descriptor.get("rows") != len(drafts)
        or descriptor.get("bytes") != len(content)
        or descriptor.get("sha256") != sha256_bytes(content)
    ):
        raise PortfolioCoreInventoryError("dev_mini DatasetAssetDraft binding drifted")


def _require_real_directory(path: Path, label: str) -> Path:
    directory = Path(path).absolute()
    try:
        metadata = directory.lstat()
    except OSError as error:
        raise PortfolioCoreInventoryError(f"{label} is unavailable") from error
    if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PortfolioCoreInventoryError(f"{label} must be a real directory")
    return directory


def _sha256_regular_file(path: Path, label: str) -> str:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PortfolioCoreInventoryError(f"{label} is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PortfolioCoreInventoryError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PortfolioCoreInventoryError(f"cannot read {label}") from error
    return digest.hexdigest()


def _validate_cord_v2_test_parquet(
    root: Path,
    *,
    expected_commit: str,
) -> tuple[Path, dict[str, object]]:
    root = _require_real_directory(root, "CORD v2 fixed-commit root")
    if root.name != expected_commit:
        raise PortfolioCoreInventoryError(
            "CORD root does not match the required fixed commit"
        )
    data_root = _require_real_directory(root / "data", "CORD v2 data directory")
    test_paths = sorted(data_root.glob("test-*.parquet"))
    if len(test_paths) != 1:
        raise PortfolioCoreInventoryError(
            "CORD v2 fixed commit must expose exactly one test Parquet shard"
        )
    test_path = test_paths[0]
    try:
        metadata = test_path.lstat()
    except OSError as error:
        raise PortfolioCoreInventoryError("CORD test Parquet is unavailable") from error
    if test_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PortfolioCoreInventoryError("CORD test Parquet must be a regular file")
    try:
        parquet = pq.ParquetFile(test_path)
        schema = parquet.schema_arrow
    except (OSError, ValueError, pa.ArrowInvalid) as error:
        raise PortfolioCoreInventoryError("CORD test Parquet cannot be opened") from error
    if "image" not in schema.names:
        raise PortfolioCoreInventoryError("CORD test Parquet is missing image bytes")
    image_type = schema.field("image").type
    if (
        not hasattr(image_type, "get_field_index")
        or image_type.get_field_index("bytes") < 0
        or image_type.get_field_index("path") < 0
    ):
        raise PortfolioCoreInventoryError(
            "CORD test Parquet image field does not expose bytes/path"
        )
    return (
        test_path,
        {
            "commit": expected_commit,
            "parquet_bytes": metadata.st_size,
            "parquet_path": f"data/{test_path.name}",
            "parquet_sha256": _sha256_regular_file(test_path, "CORD test Parquet"),
            "row_count": parquet.metadata.num_rows,
            "schema": "image_struct_bytes_path_v1",
        },
    )


def _iter_cord_v2_test_images(test_path: Path) -> Iterable[tuple[int, bytes]]:
    try:
        parquet = pq.ParquetFile(test_path)
        for index, batch in enumerate(
            parquet.iter_batches(batch_size=1, columns=["image"])
        ):
            value = batch.column(0)[0].as_py()
            if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
                raise PortfolioCoreInventoryError(
                    "CORD test Parquet image record lacks byte content"
                )
            yield index, value["bytes"]
    except (OSError, ValueError, pa.ArrowInvalid) as error:
        raise PortfolioCoreInventoryError("cannot read CORD image bytes") from error


def _validate_sroie_image_only_source(
    manifest_path: Path,
    raw_root: Path,
) -> tuple[dict[str, Any], str, Path]:
    raw_root = _require_real_directory(raw_root, "SROIE RAW root")
    content = read_stable_regular_file(
        manifest_path,
        label="SROIE acquisition manifest",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        value = parse_strict_json(content, label="SROIE acquisition manifest")
    except ArtifactFormatError as error:
        raise PortfolioCoreInventoryError("SROIE acquisition manifest is invalid") from error
    if not isinstance(value, dict) or not isinstance(value.get("manifest_id"), str):
        raise PortfolioCoreInventoryError("SROIE acquisition manifest has no manifest ID")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise PortfolioCoreInventoryError("SROIE acquisition manifest has no artifacts")
    matching = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("role") == SROIE_IMAGE_ONLY_ROLE
    ]
    if len(matching) != 1:
        raise PortfolioCoreInventoryError(
            "SROIE acquisition manifest must expose one task3 image-only package"
        )
    artifact = matching[0]
    expected = artifact.get("expected")
    logical_members = (
        expected.get("logical_members_by_extension")
        if isinstance(expected, dict)
        else None
    )
    if (
        not isinstance(logical_members, dict)
        or set(logical_members) != {".jpg"}
        or not isinstance(logical_members.get(".jpg"), int)
        or logical_members[".jpg"] <= 0
    ):
        raise PortfolioCoreInventoryError(
            "SROIE task3 package must be a semantically unique JPG-only image pack"
        )
    relative = PurePosixPath(str(artifact.get("path", "")))
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative.as_posix()
    ):
        raise PortfolioCoreInventoryError("SROIE task3 archive path is unsafe")
    archive = raw_root.joinpath(*relative.parts)
    try:
        sroie_source.validate_acquisition(manifest_path, raw_root)
    except (OSError, sroie_source.SROIEValidationError) as error:
        raise PortfolioCoreInventoryError("SROIE acquisition validation failed") from error
    _sha256_regular_file(archive, "SROIE task3 image-only archive")
    return value, sha256_bytes(content), archive


def _document_component_fingerprint(
    content: bytes,
) -> tuple[str, imagehash.ImageHash]:
    try:
        with Image.open(BytesIO(content)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise PortfolioCoreInventoryError("document image is not decodable") from error
    fingerprint = imagehash.phash(image)
    return str(fingerprint), fingerprint


def _normalize_document_image(raw: bytes) -> bytes:
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise ValueError("source document image is not decodable") from error
    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


def _try_add_document_extension_asset(
    *,
    assets: list[_DocumentExtensionAsset],
    seen_records: set[str],
    seen_content: set[str],
    seen_phashes: list[imagehash.ImageHash],
    draft: DatasetAssetDraft,
    content: bytes,
    selection_role: DocumentExtensionRole,
) -> bool:
    content_sha256 = sha256_bytes(content)
    component_fingerprint, phash = _document_component_fingerprint(content)
    if (
        draft.source_record_id in seen_records
        or content_sha256 in seen_content
        or any(phash - existing <= 4 for existing in seen_phashes)
    ):
        return False
    seen_records.add(draft.source_record_id)
    seen_content.add(content_sha256)
    seen_phashes.append(phash)
    assets.append(
        _DocumentExtensionAsset(
            draft=draft,
            content=content,
            component_fingerprint=component_fingerprint,
            selection_role=selection_role,
        )
    )
    return True


def _append_cord_document_tail(
    *,
    assets: list[_DocumentExtensionAsset],
    seen_records: set[str],
    seen_content: set[str],
    seen_phashes: list[imagehash.ImageHash],
    test_parquet: Path,
    source_revision: str,
    fresh_count: int,
    reserve_count: int,
) -> None:
    fresh_added = 0
    reserve_added = 0
    for row_index, raw in _iter_cord_v2_test_images(test_parquet):
        if fresh_added == fresh_count and reserve_added == reserve_count:
            break
        try:
            content = _normalize_document_image(raw)
        except ValueError:
            continue
        selection_role: DocumentExtensionRole = (
            "fresh_cord" if fresh_added < fresh_count else "reserve"
        )
        ordinal = fresh_added + 1 if selection_role == "fresh_cord" else reserve_added + 1
        filename = (
            f"fresh-cord-{ordinal:04d}.jpg"
            if selection_role == "fresh_cord"
            else f"reserve-cord-{ordinal:04d}.jpg"
        )
        draft = DatasetAssetDraft(
            source_dataset="cord",
            source_revision=source_revision,
            source_record_id=f"cord-v2:{test_parquet.name}:{row_index:04d}",
            transform_policy_version="portfolio-core-document-normalize-jpeg-v1",
            local_path=f"cord/{filename}",
            license_id="CC-BY-4.0",
            source_url="https://github.com/clovaai/cord",
            attribution="CORD v2 fixed local commit",
            cloud_upload_allowed=True,
            public_demo_allowed=True,
        )
        if not _try_add_document_extension_asset(
            assets=assets,
            seen_records=seen_records,
            seen_content=seen_content,
            seen_phashes=seen_phashes,
            draft=draft,
            content=content,
            selection_role=selection_role,
        ):
            continue
        if selection_role == "fresh_cord":
            fresh_added += 1
        else:
            reserve_added += 1
    if fresh_added != fresh_count or reserve_added != reserve_count:
        raise PortfolioCoreInventoryError(
            "CORD test Parquet cannot supply the requested content/pHash-unique "
            "document tail"
        )


def _append_sroie_document_tail(
    *,
    assets: list[_DocumentExtensionAsset],
    seen_records: set[str],
    seen_content: set[str],
    seen_phashes: list[imagehash.ImageHash],
    acquisition_manifest: Mapping[str, Any],
    archive: Path,
    acquisition_manifest_sha256: str,
    fresh_count: int,
    reserve_count: int,
) -> None:
    source_url = acquisition_manifest.get("official_source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise PortfolioCoreInventoryError("SROIE acquisition manifest source URL is invalid")
    fresh_added = 0
    reserve_added = 0
    logical_ids: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as source:
            infos = sorted(
                (item for item in source.infolist() if not item.is_dir()),
                key=lambda item: item.filename,
            )
            for info in infos:
                if fresh_added == fresh_count and reserve_added == reserve_count:
                    break
                member = PurePosixPath(info.filename)
                if member.is_absolute() or any(
                    part in {"", ".", ".."} for part in member.parts
                ):
                    raise PortfolioCoreInventoryError("SROIE task3 archive has unsafe paths")
                if member.suffix.lower() != ".jpg":
                    raise PortfolioCoreInventoryError(
                        "SROIE task3 archive is not an image-only JPG package"
                    )
                logical_id = re.sub(r"\(\d+\)$", "", member.stem)
                if logical_id in logical_ids:
                    raise PortfolioCoreInventoryError(
                        "SROIE task3 archive repeats a logical image identifier"
                    )
                logical_ids.add(logical_id)
                try:
                    content = _normalize_document_image(source.read(info))
                except (OSError, ValueError):
                    continue
                selection_role: DocumentExtensionRole = (
                    "fresh_sroie" if fresh_added < fresh_count else "reserve"
                )
                ordinal = (
                    fresh_added + 1
                    if selection_role == "fresh_sroie"
                    else reserve_added + 1
                )
                filename = (
                    f"fresh-sroie-{ordinal:04d}.jpg"
                    if selection_role == "fresh_sroie"
                    else f"reserve-sroie-{ordinal:04d}.jpg"
                )
                draft = DatasetAssetDraft(
                    source_dataset="sroie",
                    source_revision=f"acquisition-{acquisition_manifest_sha256}",
                    source_record_id=(
                        f"sroie:{SROIE_IMAGE_ONLY_ROLE}:{logical_id}"
                    ),
                    transform_policy_version="portfolio-core-document-normalize-jpeg-v1",
                    local_path=f"sroie/{filename}",
                    license_id="LicenseRef-SROIE-terms-pending",
                    source_url=source_url,
                    attribution="SROIE official image-only task3 package",
                    cloud_upload_allowed=True,
                    public_demo_allowed=True,
                )
                if not _try_add_document_extension_asset(
                    assets=assets,
                    seen_records=seen_records,
                    seen_content=seen_content,
                    seen_phashes=seen_phashes,
                    draft=draft,
                    content=content,
                    selection_role=selection_role,
                ):
                    continue
                if selection_role == "fresh_sroie":
                    fresh_added += 1
                else:
                    reserve_added += 1
    except (OSError, zipfile.BadZipFile) as error:
        raise PortfolioCoreInventoryError("cannot read SROIE task3 image archive") from error
    if fresh_added != fresh_count or reserve_added != reserve_count:
        raise PortfolioCoreInventoryError(
            "SROIE task3 image package cannot supply the requested content/pHash-"
            "unique document tail"
        )


def _document_extension_destination_path(asset: _DocumentExtensionAsset) -> str:
    filename = PurePosixPath(asset.draft.local_path).name
    return (
        "query_images/utility/"
        f"{asset.draft.source_dataset.replace('_', '-')}-{filename}"
    )


def _publish_or_verify_document_extension_asset(
    destination: Path,
    content: bytes,
) -> None:
    try:
        destination.lstat()
    except FileNotFoundError:
        atomic_create_file(destination, content)
    existing = read_stable_regular_file(
        destination,
        label=f"document extension asset {destination.name}",
        max_bytes=_MAX_IMAGE_BYTES,
    )
    if existing != content:
        raise FileExistsError(
            "refusing to overwrite different document extension asset: "
            + str(destination)
        )
    _read_verified_image(
        destination.parent, destination.name, "published document extension asset"
    )


def _load_dev_mini_exclusions(
    path: str | Path,
) -> tuple[Mapping[str, Any], str, dict[str, tuple[set[str], set[str]]]]:
    manifest_path = Path(path)
    try:
        mini_manifest, mini_sha256 = load_portfolio_mini_selection_manifest(
            manifest_path
        )
        mini_descriptor = mini_manifest.get("dataset_assets")
        if (
            not isinstance(mini_descriptor, dict)
            or mini_descriptor.get("path") != "dataset-assets.jsonl"
        ):
            raise PortfolioCoreInventoryError(
                "dev_mini DatasetAssetDraft path binding drifted"
            )
        mini_drafts = load_portfolio_mini_drafts(
            manifest_path.parent / "dataset-assets.jsonl"
        )
    except (PortfolioMiniError, KeyError, TypeError) as error:
        raise PortfolioCoreInventoryError(
            "dev_mini selection or bound DatasetAssetDraft inventory is invalid"
        ) from error
    _verify_bound_mini_drafts(mini_manifest, mini_drafts)
    return mini_manifest, mini_sha256, _dev_mini_exclusions(mini_manifest["selections"])


def _validate_safe_image_path(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise ValueError(f"{field_name} must be canonical POSIX text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() not in _IMAGE_SUFFIXES
    ):
        raise ValueError(f"{field_name} must be a safe supported image path")
    return value


def _validate_sha256_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PortfolioCoreInventoryError(f"{label} must be lowercase SHA-256")


def _recipe_component_key(
    *,
    source_record_id: str,
    content_sha256: str,
    component_fingerprint: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "component_fingerprint": component_fingerprint,
                "content_sha256": content_sha256,
                "source_id": "isia_food500",
                "source_record_id": source_record_id,
            }
        )
    )


def _verify_locked_isia_archive(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> Path:
    archive = Path(path).absolute()
    try:
        metadata = archive.lstat()
    except OSError as error:
        raise PortfolioCoreInventoryError("locked ISIA archive is unavailable") from error
    if archive.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PortfolioCoreInventoryError("locked ISIA archive must be a regular file")
    if metadata.st_size != expected_bytes:
        raise PortfolioCoreInventoryError(
            "locked ISIA archive byte size does not match the expected lock"
        )
    actual_sha256 = isia_food500_source.sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise PortfolioCoreInventoryError(
            "locked ISIA archive SHA-256 does not match the expected lock"
        )
    return archive


def _isia_component_fingerprint(content: bytes, *, min_side: int) -> str:
    try:
        with Image.open(BytesIO(content)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise PortfolioCoreInventoryError("ISIA image is not decodable") from error
    if min(image.size) < min_side:
        raise PortfolioCoreInventoryError("ISIA image is below the required minimum side")
    return str(imagehash.phash(image))


def _normalize_isia_member(
    raw: bytes,
    *,
    min_side: int,
) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise ValueError("ISIA ZIP member is not decodable") from error
    if min(image.size) < min_side:
        raise ValueError("ISIA ZIP member is below the required minimum side")
    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    content = output.getvalue()
    return content, _isia_component_fingerprint(content, min_side=min_side)


def _ensure_create_only_output_directory(
    output: Path,
    *,
    forbidden_paths: Sequence[Path],
) -> None:
    output = Path(output).absolute()
    try:
        resolved_output = output.resolve(strict=False)
    except OSError as error:
        raise PortfolioCoreInventoryError(
            "cannot resolve ISIA extension output root"
        ) from error
    for forbidden in forbidden_paths:
        try:
            resolved_forbidden = Path(forbidden).absolute().resolve(strict=True)
        except OSError as error:
            raise PortfolioCoreInventoryError(
                "cannot resolve ISIA extension input for containment check"
            ) from error
        if _paths_overlap(resolved_output, resolved_forbidden):
            raise PortfolioCoreInventoryError(
                "ISIA extension output must be disjoint from carry-forward and RAW inputs"
            )
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise PortfolioCoreInventoryError(
                "ISIA extension output root must be a real directory"
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError:
        if output.is_symlink() or not output.is_dir():
            raise PortfolioCoreInventoryError(
                "ISIA extension output root is not a real directory"
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


def _publish_or_verify_recipe_extension_asset(destination: Path, content: bytes) -> None:
    try:
        destination.lstat()
    except FileNotFoundError:
        atomic_create_file(destination, content)
    existing = read_stable_regular_file(
        destination,
        label=f"ISIA extension asset {destination.name}",
        max_bytes=_MAX_IMAGE_BYTES,
    )
    if existing != content:
        raise FileExistsError(
            f"refusing to overwrite different ISIA extension asset: {destination}"
        )
    _read_verified_image(
        destination.parent,
        destination.name,
        "published ISIA extension asset",
    )


def _dev_mini_exclusions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[set[str], set[str]]]:
    records: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = row.get("source_dataset")
        record = row.get("source_record_id")
        content_sha256 = row.get("image_sha256")
        if not isinstance(source, str) or not isinstance(record, str):
            raise PortfolioCoreInventoryError("dev_mini exclusion identity is invalid")
        if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(content_sha256):
            raise PortfolioCoreInventoryError("dev_mini exclusion hash is invalid")
        records[source].add(record)
        hashes[source].add(content_sha256)
    return {
        source: (records[source], hashes[source])
        for source in (
            "fashioniq",
            "inaturalist",
            "isia_food500",
        )
    }


def _mini_source_bindings(
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    expected_pool: str,
    expected_capability: str,
    quota: int,
) -> dict[str, str]:
    rows = [
        row for row in selection_rows if row.get("source_dataset") == source_id
    ]
    if len(rows) != quota:
        raise PortfolioCoreInventoryError(
            f"dev_mini must bind exactly {quota} {source_id} assets"
        )
    bindings: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (item["source_record_id"], item["image_path"])):
        if (
            row.get("pool") != expected_pool
            or row.get("canonical_intent") != "utility"
            or row.get("canonical_capability") != expected_capability
        ):
            raise PortfolioCoreInventoryError(
                f"dev_mini {source_id} capability binding drifted"
            )
        record_id = row.get("source_record_id")
        content_sha256 = row.get("image_sha256")
        if not isinstance(record_id, str) or record_id in bindings:
            raise PortfolioCoreInventoryError(
                f"dev_mini {source_id} record binding is invalid"
            )
        if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(
            content_sha256
        ):
            raise PortfolioCoreInventoryError(
                f"dev_mini {source_id} image hash binding is invalid"
            )
        bindings[record_id] = content_sha256
    if len(set(bindings.values())) != len(bindings):
        raise PortfolioCoreInventoryError(
            f"dev_mini {source_id} assets contain duplicate bytes"
        )
    return bindings


def _select_bound_draft_candidates(
    *,
    source_id: str,
    drafts: Iterable[DatasetAssetDraft],
    source_root: Path,
    record_content_bindings: Mapping[str, str],
) -> tuple[PortfolioCoreAssetCandidate, ...]:
    rows = tuple(drafts)
    by_record = {item.source_record_id: item for item in rows}
    if len(by_record) != len(rows):
        raise PortfolioCoreInventoryError(f"{source_id} drafts repeat source records")
    candidates: list[PortfolioCoreAssetCandidate] = []
    for record_id, expected_sha256 in sorted(record_content_bindings.items()):
        draft = by_record.get(record_id)
        if draft is None or draft.source_dataset != source_id:
            raise PortfolioCoreInventoryError(
                f"{source_id} cannot resolve dev_mini record {record_id}"
            )
        content = _read_verified_image(source_root, draft.local_path, f"{source_id} asset")
        if sha256_bytes(content) != expected_sha256:
            raise PortfolioCoreInventoryError(
                f"{source_id} dev_mini-bound image bytes drifted: {record_id}"
            )
        pool = _SOURCE_BINDINGS[source_id][0]
        suffix = PurePosixPath(draft.local_path).suffix.lower()
        candidates.append(
            _candidate(
                source_id=source_id,
                draft=draft,
                expected_content=content,
                destination_path=(
                    f"query_images/{pool}/{source_id.replace('_', '-')}-core-"
                    f"{len(candidates) + 1:04d}{suffix}"
                ),
            )
        )
    return tuple(candidates)


def _select_isia_extension_candidates(
    *,
    drafts: Iterable[DatasetAssetDraft],
    source_root: Path,
    extension: ISIARecipeExtensionManifest,
    excluded_record_ids: set[str],
    excluded_content_sha256: set[str],
) -> tuple[PortfolioCoreAssetCandidate, ...]:
    """Turn a verified ISIA extension binding into candidate inventory rows."""

    rows = tuple(drafts)
    by_record = {item.source_record_id: item for item in rows}
    if len(by_record) != len(rows):
        raise PortfolioCoreInventoryError("ISIA extension staging repeats source records")
    entry_records = {entry.source_record_id for entry in extension.selections}
    if entry_records != set(by_record):
        raise PortfolioCoreInventoryError(
            "ISIA extension manifest does not enumerate the supplied staging rows"
        )
    candidates: list[PortfolioCoreAssetCandidate] = []
    seen_content: set[str] = set(excluded_content_sha256)
    for index, entry in enumerate(extension.selections, start=1):
        draft = by_record[entry.source_record_id]
        if draft.local_path != entry.image_path:
            raise PortfolioCoreInventoryError(
                "ISIA extension image path binding drifted for "
                + entry.source_record_id
            )
        if draft.source_record_id in excluded_record_ids:
            raise PortfolioCoreInventoryError(
                "ISIA extension record overlaps the dev-mini exclusion: "
                + draft.source_record_id
            )
        content = _read_verified_image(source_root, draft.local_path, "ISIA extension asset")
        digest = sha256_bytes(content)
        if digest != entry.content_sha256:
            raise PortfolioCoreInventoryError(
                "ISIA extension content binding drifted for "
                + draft.source_record_id
            )
        if digest in seen_content:
            raise PortfolioCoreInventoryError(
                "ISIA extension content overlaps a prior or excluded binding"
            )
        component_fingerprint = _isia_component_fingerprint(content, min_side=1)
        if component_fingerprint != entry.component_fingerprint:
            raise PortfolioCoreInventoryError(
                "ISIA extension component binding drifted for "
                + draft.source_record_id
            )
        seen_content.add(digest)
        selection = "reserve" if entry.selection_role == "reserve" else "include"
        suffix = PurePosixPath(draft.local_path).suffix.lower()
        candidates.append(
            _candidate(
                source_id="isia_food500",
                draft=draft,
                expected_content=content,
                destination_path=(
                    "query_images/utility/isia-food500-core-"
                    f"{index:04d}{suffix}"
                ),
                selection=selection,
            )
        )
    return tuple(candidates)


def _select_draft_candidates(
    *,
    source_id: str,
    drafts: Iterable[DatasetAssetDraft],
    source_root: Path,
    quota: int,
    excluded_record_ids: set[str],
    excluded_content_sha256: set[str],
    destination_index_offset: int = 0,
    selection: str = "include",
) -> tuple[PortfolioCoreAssetCandidate, ...]:
    if quota <= 0:
        raise PortfolioCoreInventoryError(f"{source_id} quota must be positive")
    if destination_index_offset < 0:
        raise PortfolioCoreInventoryError(
            f"{source_id} destination index offset must be non-negative"
        )
    if selection not in {"include", "reserve"}:
        raise PortfolioCoreInventoryError(
            f"{source_id} selection must be include or reserve"
        )
    rows = tuple(drafts)
    records = [item.source_record_id for item in rows]
    paths = [item.local_path for item in rows]
    if len(records) != len(set(records)) or len(paths) != len(set(paths)):
        raise PortfolioCoreInventoryError(f"{source_id} drafts contain duplicate identities")
    selected: list[PortfolioCoreAssetCandidate] = []
    seen_hashes: set[str] = set(excluded_content_sha256)
    for draft in sorted(rows, key=lambda item: (item.source_record_id, item.local_path)):
        if draft.source_dataset != source_id:
            raise PortfolioCoreInventoryError(f"{source_id} draft source_dataset drifted")
        if draft.source_record_id in excluded_record_ids:
            continue
        if draft.derivation_parent_asset_ids or draft.derivation_parent_asset_id:
            raise PortfolioCoreInventoryError(
                f"{source_id} core query asset depends on omitted derivation parents"
            )
        content = _read_verified_image(source_root, draft.local_path, f"{source_id} asset")
        digest = sha256_bytes(content)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        pool = _SOURCE_BINDINGS[source_id][0]
        suffix = PurePosixPath(draft.local_path).suffix.lower()
        destination = (
            f"query_images/{pool}/{source_id.replace('_', '-')}-core-"
            f"{destination_index_offset + len(selected) + 1:04d}{suffix}"
        )
        selected.append(
            _candidate(
                source_id=source_id,
                draft=draft,
                expected_content=content,
                destination_path=destination,
                selection=selection,
            )
        )
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise PortfolioCoreInventoryError(
            f"{source_id} cannot supply {quota} fresh content-unique regular assets"
        )
    return tuple(selected)


def _candidate(
    *,
    source_id: str,
    draft: DatasetAssetDraft,
    expected_content: bytes,
    destination_path: str,
    selection: str = "include",
) -> PortfolioCoreAssetCandidate:
    pool, intent, capability = _SOURCE_BINDINGS[source_id]
    digest = sha256_bytes(expected_content)
    identity = sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": REGULAR_CORE_INVENTORY_POLICY_VERSION,
                "source_id": source_id,
                "source_record_id": draft.source_record_id,
                "source_revision": draft.source_revision,
                "source_sha256": digest,
            }
        )
    )
    return PortfolioCoreAssetCandidate(
        candidate_id=f"{source_id}.core.{identity[:32]}",
        source_id=source_id,
        selection=selection,  # type: ignore[arg-type]
        pool=pool,  # type: ignore[arg-type]
        source_local_path=draft.local_path,
        destination_path=destination_path,
        expected_bytes=len(expected_content),
        expected_sha256=digest,
        draft=draft,
        capability_bindings=(
            CandidateCapabilityBinding(
                canonical_intent=intent,  # type: ignore[arg-type]
                canonical_capability=capability,
            ),
        ),
    )


def _document_selection_entry(
    candidate: PortfolioCoreAssetCandidate,
    selection_role: str,
    *,
    component_fingerprint: str,
) -> PortfolioCoreDocumentSelectionEntry:
    if candidate.source_id not in {
        "wikimedia_commons_documents",
        "cord",
        "sroie",
    }:
        raise PortfolioCoreInventoryError(
            "document selection entry must bind a configured document candidate"
        )
    return PortfolioCoreDocumentSelectionEntry(
        candidate_id=candidate.candidate_id,
        source_id=candidate.source_id,
        source_record_id=candidate.draft.source_record_id,
        content_sha256=candidate.expected_sha256,
        component_fingerprint=component_fingerprint,
        component_key=document_component_key(
            source_id=candidate.source_id,
            source_record_id=candidate.draft.source_record_id,
            content_sha256=candidate.expected_sha256,
            component_fingerprint=component_fingerprint,
        ),
        selection_role=selection_role,  # type: ignore[arg-type]
    )


def _load_recipe1m_plus_reserves(
    inventory_path: Path,
    source_root: Path,
    *,
    license_id: str,
    attribution: str | None,
) -> tuple[PortfolioCoreAssetCandidate, ...]:
    if not license_id or license_id != license_id.strip():
        raise PortfolioCoreInventoryError("Recipe1M+ license ID must be canonical text")
    content = read_stable_regular_file(
        inventory_path,
        label="Recipe1M+ bounded inventory",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        value = parse_canonical_json(content, label="Recipe1M+ bounded inventory")
    except ArtifactFormatError as error:
        raise PortfolioCoreInventoryError("Recipe1M+ bounded inventory is invalid") from error
    if content != canonical_json_bytes(value) or not isinstance(value, dict):
        raise PortfolioCoreInventoryError("Recipe1M+ inventory must be canonical JSON")
    entities = value.get("entities")
    if not isinstance(entities, list) or not entities:
        raise PortfolioCoreInventoryError("Recipe1M+ inventory has no selected entities")
    revision = sha256_bytes(content)
    records: list[tuple[str, str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("entity_id"), str):
            raise PortfolioCoreInventoryError("Recipe1M+ entity row is invalid")
        recipes = entity.get("recipes")
        if not isinstance(recipes, list):
            raise PortfolioCoreInventoryError("Recipe1M+ entity recipes are invalid")
        for recipe in recipes:
            if not isinstance(recipe, dict) or not isinstance(recipe.get("recipe_id"), str):
                raise PortfolioCoreInventoryError("Recipe1M+ recipe row is invalid")
            image_ids = recipe.get("image_ids")
            if not isinstance(image_ids, list) or not image_ids:
                raise PortfolioCoreInventoryError("Recipe1M+ recipe image IDs are invalid")
            for image_id in image_ids:
                if (
                    not isinstance(image_id, str)
                    or PurePosixPath(image_id).name != image_id
                    or not image_id.lower().endswith(".jpg")
                ):
                    raise PortfolioCoreInventoryError("Recipe1M+ image ID is unsafe")
                records.append((entity["entity_id"], recipe["recipe_id"], image_id))
    if len(records) != len({item[2] for item in records}):
        raise PortfolioCoreInventoryError("Recipe1M+ bounded inventory repeats image IDs")
    if value.get("image_count") != len(records):
        raise PortfolioCoreInventoryError("Recipe1M+ image_count binding drifted")
    candidates: list[PortfolioCoreAssetCandidate] = []
    for index, (entity_id, recipe_id, image_id) in enumerate(sorted(records), start=1):
        local_path = f"images/{image_id}"
        image = _read_verified_image(source_root, local_path, "Recipe1M+ selected image")
        draft = DatasetAssetDraft(
            source_dataset="recipe1m_plus",
            source_revision=f"inventory-{revision}",
            source_record_id=f"recipe:{recipe_id}:image:{image_id}",
            transform_policy_version="recipe1m-plus-approved-selection-v1",
            local_path=local_path,
            product_id=f"recipe1m_plus:{entity_id}:{recipe_id}",
            license_id=license_id,
            attribution=attribution,
            cloud_upload_allowed=True,
            public_demo_allowed=True,
        )
        candidates.append(
            _candidate(
                source_id="recipe1m_plus",
                draft=draft,
                expected_content=image,
                destination_path=(
                    f"query_images/utility/recipe1m-plus-reserve-{index:04d}.jpg"
                ),
                selection="reserve",
            )
        )
    return tuple(candidates)


def _load_strict_jsonl_models(
    path: Path,
    model: type[_StrictModel],
    *,
    label: str,
) -> tuple[tuple[Any, ...], str]:
    content = read_stable_regular_file(path, label=label, max_bytes=_MAX_METADATA_BYTES)
    if not content or not content.endswith(b"\n"):
        raise PortfolioCoreInventoryError(f"{label} must be non-empty newline-terminated JSONL")
    values: list[Any] = []
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        if line == b"\n":
            raise PortfolioCoreInventoryError(f"{label} contains a blank line")
        try:
            raw = parse_strict_json(line, label=f"{label} line {line_number}")
            values.append(model.model_validate(raw, strict=True))
        except (ArtifactFormatError, ValidationError) as error:
            raise PortfolioCoreInventoryError(
                f"{label} line {line_number} violates schema"
            ) from error
    # Source materializers predate the repository's canonical key-order
    # convention.  Preserve and bind their exact bytes, while requiring every
    # line to be strict JSON and every object to satisfy the frozen schema.
    # Only our emitted PortfolioCoreAssetCandidate inventory is canonicalized.
    return tuple(values), sha256_bytes(content)


def _read_verified_image(root: Path, local_path: str, label: str) -> bytes:
    path = _resolve_regular_file(root, local_path, label)
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=_MAX_IMAGE_BYTES)
    except ArtifactFormatError as error:
        raise PortfolioCoreInventoryError(str(error)) from error
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise PortfolioCoreInventoryError(f"{label} is not a decodable image") from error
    if not content:
        raise PortfolioCoreInventoryError(f"{label} is empty")
    return content


def _resolve_regular_file(root: Path, local_path: str, label: str) -> Path:
    if not isinstance(local_path, str) or "\\" in local_path:
        raise PortfolioCoreInventoryError(f"{label} local path is not canonical POSIX")
    relative = PurePosixPath(local_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PortfolioCoreInventoryError(f"{label} local path escapes its source root")
    current = Path(root).absolute()
    try:
        root_metadata = current.lstat()
    except OSError as error:
        raise PortfolioCoreInventoryError(f"{label} source root is unavailable") from error
    if not stat.S_ISDIR(root_metadata.st_mode) or current.is_symlink():
        raise PortfolioCoreInventoryError(f"{label} source root must be a real directory")
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise PortfolioCoreInventoryError(f"{label} source file is unavailable") from error
        if current.is_symlink():
            raise PortfolioCoreInventoryError(f"{label} path contains a symlink")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PortfolioCoreInventoryError(f"{label} parent is not a directory")
    if not stat.S_ISREG(current.lstat().st_mode) or current.suffix.lower() not in _IMAGE_SUFFIXES:
        raise PortfolioCoreInventoryError(f"{label} is not a supported regular image")
    return current


def _validate_regular_inventory(
    candidates: Sequence[PortfolioCoreAssetCandidate],
    *,
    source_quotas: Mapping[str, int] = REGULAR_SOURCE_QUOTAS,
) -> None:
    included = tuple(item for item in candidates if item.selection == "include")
    counts = Counter(item.source_id for item in included)
    if counts != Counter(source_quotas):
        raise PortfolioCoreInventoryError(
            "regular-source quota drift: "
            + json.dumps(dict(sorted(counts.items())), separators=(",", ":"))
        )
    ids = [item.candidate_id for item in candidates]
    destinations = [item.destination_path for item in candidates]
    if len(ids) != len(set(ids)) or len(destinations) != len(set(destinations)):
        raise PortfolioCoreInventoryError("regular inventory has duplicate identities")
    pool_counts = _unique_pool_counts(included)
    expected = {
        "divergent_rec": source_quotas["fashioniq"],
        "encyclopedia": source_quotas["inaturalist"],
        "utility": (
            source_quotas["isia_food500"]
            + source_quotas["wikimedia_commons_documents"]
        ),
    }
    if pool_counts != expected:
        raise PortfolioCoreInventoryError(
            "regular inventory unique pool counts drifted: "
            + json.dumps(pool_counts, sort_keys=True, separators=(",", ":"))
        )


def _unique_pool_counts(
    candidates: Iterable[PortfolioCoreAssetCandidate],
) -> dict[str, int]:
    hashes: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        hashes[item.pool].add(item.expected_sha256)
    return {pool: len(values) for pool, values in sorted(hashes.items())}


def _candidate_sort_key(candidate: PortfolioCoreAssetCandidate) -> tuple[Any, ...]:
    pool_order = {
        "exact_match": 0,
        "multi_product": 1,
        "divergent_rec": 2,
        "encyclopedia": 3,
        "utility": 4,
    }
    return (
        pool_order[candidate.pool],
        candidate.selection != "include",
        candidate.source_id,
        candidate.destination_path,
        candidate.candidate_id,
    )


def _publish_idempotent(path: Path, content: bytes) -> None:
    try:
        existing = read_stable_regular_file(
            path,
            label=f"existing output {path.name}",
            max_bytes=_MAX_METADATA_BYTES,
        )
    except (FileNotFoundError, ArtifactFormatError):
        if path.exists():
            raise PortfolioCoreInventoryError(f"output is not a regular file: {path}")
        atomic_create_file(path, content)
        return
    if existing != content:
        raise FileExistsError(f"refusing to overwrite different inventory: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    regular = commands.add_parser("build-regular")
    regular.add_argument("--dev-mini-selection", type=Path, required=True)
    regular.add_argument("--fashioniq-adapter-root", type=Path, required=True)
    regular.add_argument("--fashioniq-manifest-sha256", required=True)
    regular.add_argument("--fashioniq-asset-root", type=Path, required=True)
    regular.add_argument("--inaturalist-manifest", type=Path, required=True)
    regular.add_argument("--inaturalist-asset-root", type=Path, required=True)
    regular.add_argument("--commons-document-manifest", type=Path, required=True)
    regular.add_argument("--commons-document-asset-root", type=Path, required=True)
    regular.add_argument("--isia-food-manifest", type=Path, required=True)
    regular.add_argument("--isia-food-asset-root", type=Path, required=True)
    regular.add_argument(
        "--isia-food-quota",
        type=int,
        default=REGULAR_SOURCE_QUOTAS["isia_food500"],
    )
    regular.add_argument("--isia-food-selection-manifest", type=Path)
    regular.add_argument("--document-fresh-count", type=int, default=0)
    regular.add_argument("--document-reserve-count", type=int, default=0)
    regular.add_argument("--document-selection-manifest", type=Path)
    regular.add_argument("--recipe1m-plus-inventory", type=Path)
    regular.add_argument("--recipe1m-plus-root", type=Path)
    regular.add_argument("--recipe1m-plus-license-id")
    regular.add_argument("--recipe1m-plus-attribution")
    regular.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--inventory", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--allow-incomplete-capacity", action="store_true")

    recipe_extension = commands.add_parser("build-isia-recipe-extension")
    recipe_extension.add_argument("--dev-mini-selection", type=Path, required=True)
    recipe_extension.add_argument("--carry-forward-manifest", type=Path, required=True)
    recipe_extension.add_argument("--carry-forward-asset-root", type=Path, required=True)
    recipe_extension.add_argument("--raw-archive", type=Path, required=True)
    recipe_extension.add_argument("--output-root", type=Path, required=True)
    recipe_extension.add_argument("--include-count", type=int, default=R2_RECIPE_TARGET_COUNT)
    recipe_extension.add_argument("--reserve-count", type=int, default=R2_RECIPE_RESERVE_COUNT)
    recipe_extension.add_argument(
        "--carry-forward-count",
        type=int,
        default=R2_RECIPE_CARRY_FORWARD_COUNT,
    )

    recipe_candidates = commands.add_parser("build-isia-recipe-candidates")
    recipe_candidates.add_argument("--dev-mini-selection", type=Path, required=True)
    recipe_candidates.add_argument("--staging-manifest", type=Path, required=True)
    recipe_candidates.add_argument("--staging-asset-root", type=Path, required=True)
    recipe_candidates.add_argument("--extension-manifest", type=Path, required=True)
    recipe_candidates.add_argument("--output", type=Path, required=True)

    documents = commands.add_parser("build-document-extension")
    documents.add_argument("--dev-mini-selection", type=Path, required=True)
    documents.add_argument("--dev-mini-asset-root", type=Path, required=True)
    documents.add_argument("--cord-root", type=Path, required=True)
    documents.add_argument("--sroie-acquisition-manifest", type=Path, required=True)
    documents.add_argument("--sroie-raw-root", type=Path, required=True)
    documents.add_argument("--output-root", type=Path, required=True)
    documents.add_argument(
        "--cord-fresh-count", type=int, default=R2_DOCUMENT_CORD_FRESH_COUNT
    )
    documents.add_argument(
        "--cord-reserve-count", type=int, default=R2_DOCUMENT_CORD_RESERVE_COUNT
    )
    documents.add_argument(
        "--sroie-fresh-count", type=int, default=R2_DOCUMENT_SROIE_FRESH_COUNT
    )
    documents.add_argument(
        "--sroie-reserve-count", type=int, default=R2_DOCUMENT_SROIE_RESERVE_COUNT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "build-regular":
        result = build_regular_portfolio_core_inventory(
            dev_mini_selection_manifest=arguments.dev_mini_selection,
            fashioniq_adapter_root=arguments.fashioniq_adapter_root,
            fashioniq_manifest_sha256=arguments.fashioniq_manifest_sha256,
            fashioniq_asset_root=arguments.fashioniq_asset_root,
            inaturalist_manifest=arguments.inaturalist_manifest,
            inaturalist_asset_root=arguments.inaturalist_asset_root,
            commons_document_manifest=arguments.commons_document_manifest,
            commons_document_asset_root=arguments.commons_document_asset_root,
            isia_food_manifest=arguments.isia_food_manifest,
            isia_food_asset_root=arguments.isia_food_asset_root,
            output_path=arguments.output,
            isia_food_quota=arguments.isia_food_quota,
            isia_food_selection_manifest=arguments.isia_food_selection_manifest,
            document_fresh_count=arguments.document_fresh_count,
            document_reserve_count=arguments.document_reserve_count,
            document_selection_manifest_path=arguments.document_selection_manifest,
            recipe1m_plus_inventory=arguments.recipe1m_plus_inventory,
            recipe1m_plus_root=arguments.recipe1m_plus_root,
            recipe1m_plus_license_id=arguments.recipe1m_plus_license_id,
            recipe1m_plus_attribution=arguments.recipe1m_plus_attribution,
        )
        payload = {
            "dev_mini_selection_sha256": result.dev_mini_selection_sha256,
            "document_selection_manifest_path": (
                str(result.document_selection_manifest_path)
                if result.document_selection_manifest_path
                else None
            ),
            "document_selection_manifest_sha256": result.document_selection_manifest_sha256,
            "include_count": result.include_count,
            "output_path": str(result.output_path),
            "output_sha256": result.output_sha256,
            "pool_unique_content_counts": result.pool_unique_content_counts,
            "reserve_count": result.reserve_count,
            "source_counts": result.source_counts,
            "status": "ready",
            "track": "portfolio",
        }
    elif arguments.command == "merge":
        result = merge_portfolio_core_candidate_inventories(
            inventory_paths=arguments.inventory,
            output_path=arguments.output,
            require_capacity=not arguments.allow_incomplete_capacity,
        )
        payload = {
            "include_count": result.include_count,
            "output_path": str(result.output_path),
            "output_sha256": result.output_sha256,
            "pool_unique_content_counts": result.pool_unique_content_counts,
            "reserve_count": result.reserve_count,
            "status": "ready",
            "track": "portfolio",
        }
    elif arguments.command == "build-isia-recipe-extension":
        result = materialize_isia_food500_recipe_extension(
            dev_mini_selection_manifest=arguments.dev_mini_selection,
            carry_forward_manifest=arguments.carry_forward_manifest,
            carry_forward_asset_root=arguments.carry_forward_asset_root,
            raw_archive_path=arguments.raw_archive,
            output_root=arguments.output_root,
            include_count=arguments.include_count,
            reserve_count=arguments.reserve_count,
            carry_forward_count=arguments.carry_forward_count,
        )
        payload = {
            "archive_sha256": result.archive_sha256,
            "carry_forward_count": result.carry_forward_count,
            "fresh_count": result.fresh_count,
            "output_root": str(result.output_root),
            "output_staging_manifest_path": str(result.output_staging_manifest_path),
            "output_staging_manifest_sha256": result.output_staging_manifest_sha256,
            "reserve_count": result.reserve_count,
            "selection_manifest_path": str(result.selection_manifest_path),
            "selection_manifest_sha256": result.selection_manifest_sha256,
            "status": "ready",
            "track": "portfolio",
        }
    elif arguments.command == "build-document-extension":
        result = materialize_portfolio_core_document_extension(
            dev_mini_selection_manifest=arguments.dev_mini_selection,
            dev_mini_asset_root=arguments.dev_mini_asset_root,
            cord_root=arguments.cord_root,
            sroie_acquisition_manifest=arguments.sroie_acquisition_manifest,
            sroie_raw_root=arguments.sroie_raw_root,
            output_root=arguments.output_root,
            cord_fresh_count=arguments.cord_fresh_count,
            cord_reserve_count=arguments.cord_reserve_count,
            sroie_fresh_count=arguments.sroie_fresh_count,
            sroie_reserve_count=arguments.sroie_reserve_count,
        )
        payload = {
            "candidate_inventory_path": str(result.candidate_inventory_path),
            "candidate_inventory_sha256": result.candidate_inventory_sha256,
            "dev_mini_selection_sha256": result.dev_mini_selection_sha256,
            "output_root": str(result.output_root),
            "role_counts": result.role_counts,
            "selection_manifest_path": str(result.selection_manifest_path),
            "selection_manifest_sha256": result.selection_manifest_sha256,
            "source_counts": result.source_counts,
            "staging_drafts_path": str(result.staging_drafts_path),
            "staging_drafts_sha256": result.staging_drafts_sha256,
            "status": "ready",
            "track": "portfolio",
        }
    else:
        result = build_isia_recipe_extension_candidates(
            dev_mini_selection_manifest=arguments.dev_mini_selection,
            staging_manifest=arguments.staging_manifest,
            staging_asset_root=arguments.staging_asset_root,
            extension_manifest=arguments.extension_manifest,
            output_path=arguments.output,
        )
        payload = {
            "dev_mini_selection_sha256": result.dev_mini_selection_sha256,
            "extension_manifest_sha256": result.extension_manifest_sha256,
            "include_count": result.include_count,
            "output_path": str(result.output_path),
            "output_sha256": result.output_sha256,
            "reserve_count": result.reserve_count,
            "status": "ready",
            "track": "portfolio",
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ISIARecipeExtensionManifest",
    "ISIARecipeExtensionResult",
    "ISIARecipeSelectionEntry",
    "ISIARecipeCandidateBuildResult",
    "PORTFOLIO_CORE_R2_CATALOG_POLICY_VERSION",
    "PortfolioCoreDocumentExtensionResult",
    "PortfolioCoreInventoryBuildResult",
    "PortfolioCoreInventoryError",
    "PortfolioCoreInventoryMergeResult",
    "PortfolioCoreR2CatalogResult",
    "REGULAR_SOURCE_QUOTAS",
    "R2_DOCUMENT_CARRY_FORWARD_COUNT",
    "R2_DOCUMENT_CORD_FRESH_COUNT",
    "R2_DOCUMENT_CORD_RESERVE_COUNT",
    "R2_DOCUMENT_FRESH_COUNT",
    "R2_DOCUMENT_RESERVE_COUNT",
    "R2_DOCUMENT_SROIE_FRESH_COUNT",
    "R2_DOCUMENT_SROIE_RESERVE_COUNT",
    "R2_RECIPE_CARRY_FORWARD_COUNT",
    "R2_RECIPE_RESERVE_COUNT",
    "R2_RECIPE_TARGET_COUNT",
    "R2_ABO_ADDITIONAL_CROSS_INTENT_COUNT",
    "build_portfolio_core_r2_candidate_catalog",
    "build_regular_portfolio_core_inventory",
    "build_isia_recipe_extension_candidates",
    "load_isia_recipe_extension_manifest",
    "materialize_portfolio_core_document_extension",
    "materialize_isia_food500_recipe_extension",
    "merge_portfolio_core_candidate_inventories",
]
