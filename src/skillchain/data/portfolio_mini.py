"""Create the non-formal 184-image Portfolio mini pool and its assignments.

This module only selects and copies already prepared public/project assets.  It
does not create, label, or review query text.  The resulting pool is explicitly
Portfolio Track evidence and is not a formal-reproduction artifact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Iterable, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, ValidationError

from skillchain.data.abo import LICENSE_ID as ABO_LICENSE_ID
from skillchain.data.abo import TRANSFORM_POLICY_VERSION as ABO_TRANSFORM_POLICY
from skillchain.data.asset_catalog import (
    AssetCatalog,
    DatasetAssetDraft,
    load_asset_catalog,
)
from skillchain.data.fashioniq import load_verified_fashioniq_adapter_bundle
from skillchain.data.rpc import load_verified_rpc_val_adapter
from skillchain.schemas import Intent
from skillchain.synthesis.planning import (
    PHASE3_TASK_SPEC_SHA256,
    PHASE3_TASK_SPEC_VERSION,
    CapabilityAssignment,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import (
    DEFAULT_TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    load_default_taxonomy_registry,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_strict_json,
    read_stable_regular_file,
)

SELECTION_POLICY_VERSION = "portfolio-mini-query-images-v1"
ASSIGNMENT_POLICY_VERSION = "portfolio-mini-capability-assignments-v1"
SELECTION_MANIFEST_FILE = "selection-manifest.json"
DATASET_ASSETS_FILE = "dataset-assets.jsonl"

_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

Pool = Literal[
    "exact",
    "multi",
    "style",
    "encyclopedia",
    "utility_document",
    "utility_recipe",
]

POOL_ORDER: tuple[Pool, ...] = (
    "exact",
    "multi",
    "style",
    "encyclopedia",
    "utility_document",
    "utility_recipe",
)
POOL_QUOTAS: dict[Pool, int] = {
    "exact": 35,
    "multi": 35,
    "style": 27,
    "encyclopedia": 27,
    "utility_document": 30,
    "utility_recipe": 30,
}
POOL_BINDINGS: dict[Pool, tuple[Intent, str, str]] = {
    "exact": ("exact_match", "product.exact_match", "exact_match"),
    "multi": ("multi_product", "product.multi_search", "multi_product"),
    "style": ("divergent_rec", "product.style_recommendation", "divergent_rec"),
    "encyclopedia": (
        "encyclopedia",
        "knowledge.visual_encyclopedia",
        "encyclopedia",
    ),
    "utility_document": ("utility", "utility.document_reading", "utility"),
    "utility_recipe": ("utility", "utility.recipe_guidance", "utility"),
}


class PortfolioMiniError(ValueError):
    """The prepared source pool or published mini pool is inconsistent."""


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
    cloud_upload_allowed: bool | None = None
    public_demo_allowed: bool = False


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
    cloud_upload_allowed: bool | None = None
    public_demo_allowed: bool = False


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
    cloud_upload_allowed: bool | None = None
    public_demo_allowed: bool = False


@dataclass(frozen=True)
class _SourceArtifact:
    source_id: str
    artifact_format: str
    content_sha256: str
    bytes: int
    rows: int

    def as_json(self) -> dict[str, Any]:
        return {
            "artifact_format": self.artifact_format,
            "bytes": self.bytes,
            "rows": self.rows,
            "sha256": self.content_sha256,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class _Candidate:
    pool: Pool
    draft: DatasetAssetDraft
    source_path: Path
    source_local_path: str
    source_artifact_sha256: str
    source_bytes: int
    source_sha256: str
    destination_name: str


@dataclass(frozen=True)
class PortfolioMiniBuildResult:
    output_root: Path
    dataset_assets_path: Path
    dataset_assets_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    asset_count: int


@dataclass(frozen=True)
class PortfolioMiniAssignmentResult:
    output_path: Path
    assignment_sha256: str
    assignment_count: int
    selection_manifest_sha256: str


def assemble_portfolio_mini(
    *,
    abo_audit_receipt: str | Path,
    abo_original_root: str | Path,
    fashioniq_adapter_root: str | Path,
    fashioniq_manifest_sha256: str,
    fashioniq_asset_root: str | Path,
    rpc_adapter_root: str | Path,
    rpc_manifest_sha256: str,
    inaturalist_manifest: str | Path,
    inaturalist_asset_root: str | Path,
    commons_document_manifest: str | Path,
    commons_document_asset_root: str | Path,
    isia_food_manifest: str | Path,
    isia_food_asset_root: str | Path,
    output_root: str | Path,
) -> PortfolioMiniBuildResult:
    """Select 184 existing images and publish five create-only intent folders.

    Existing intent folders are accepted only when their complete supported
    image set and every image byte exactly match the deterministic selection.
    This permits adopting the already materialized iNaturalist folder without
    rewriting it while still failing closed on a conflicting partial run.
    """

    output_root = Path(output_root).absolute()
    _ensure_output_root(output_root)

    candidates: list[_Candidate] = []
    artifacts: list[_SourceArtifact] = []

    abo_candidates, abo_artifact = _load_abo_candidates(
        Path(abo_audit_receipt),
        Path(abo_original_root),
    )
    candidates.extend(abo_candidates)
    artifacts.append(abo_artifact)

    fashion_bundle = load_verified_fashioniq_adapter_bundle(
        fashioniq_adapter_root,
        expected_manifest_sha256=_require_sha256(
            fashioniq_manifest_sha256, "FashionIQ manifest SHA-256"
        ),
    )
    fashion_drafts = _ordered_drafts(
        fashion_bundle.drafts,
        expected_source_dataset="fashioniq",
        minimum_count=POOL_QUOTAS["style"],
        label="FashionIQ",
    )
    fashion_artifact = _stable_artifact(
        Path(fashioniq_adapter_root) / "manifest.json",
        source_id="fashioniq",
        artifact_format="verified-adapter-manifest-json",
        rows=len(fashion_bundle.drafts),
    )
    if fashion_artifact.content_sha256 != fashioniq_manifest_sha256:
        raise PortfolioMiniError("FashionIQ manifest changed after verification")
    candidates.extend(
        _draft_candidates(
            pool="style",
            drafts=fashion_drafts,
            asset_root=Path(fashioniq_asset_root),
            artifact=fashion_artifact,
            filename_prefix="fashioniq",
            quota=POOL_QUOTAS["style"],
            skip_duplicate_content=True,
        )
    )
    artifacts.append(fashion_artifact)

    rpc_bundle = load_verified_rpc_val_adapter(
        rpc_adapter_root,
        expected_manifest_file_sha256=_require_sha256(
            rpc_manifest_sha256, "RPC manifest SHA-256"
        ),
    )
    rpc_drafts = _select_drafts(
        rpc_bundle.drafts,
        expected_source_dataset="rpc",
        quota=POOL_QUOTAS["multi"],
        label="RPC",
    )
    rpc_artifact = _stable_artifact(
        Path(rpc_adapter_root) / "manifest.json",
        source_id="rpc",
        artifact_format="verified-adapter-manifest-json",
        rows=len(rpc_bundle.drafts),
    )
    if rpc_artifact.content_sha256 != rpc_manifest_sha256:
        raise PortfolioMiniError("RPC manifest changed after verification")
    candidates.extend(
        _draft_candidates(
            pool="multi",
            drafts=rpc_drafts,
            asset_root=Path(rpc_adapter_root).absolute().parent,
            artifact=rpc_artifact,
            filename_prefix="rpc",
        )
    )
    artifacts.append(rpc_artifact)

    inat_drafts, inat_artifact = _load_inaturalist_drafts(inaturalist_manifest)
    candidates.extend(
        _draft_candidates(
            pool="encyclopedia",
            drafts=_select_drafts(
                inat_drafts,
                expected_source_dataset="inaturalist",
                quota=POOL_QUOTAS["encyclopedia"],
                label="iNaturalist",
            ),
            asset_root=Path(inaturalist_asset_root),
            artifact=inat_artifact,
            filename_prefix="inaturalist",
            preserve_source_name=True,
        )
    )
    artifacts.append(inat_artifact)

    document_drafts, document_artifact = _load_commons_document_drafts(
        commons_document_manifest
    )
    candidates.extend(
        _draft_candidates(
            pool="utility_document",
            drafts=_select_drafts(
                document_drafts,
                expected_source_dataset="wikimedia_commons_documents",
                quota=POOL_QUOTAS["utility_document"],
                label="Wikimedia Commons documents",
            ),
            asset_root=Path(commons_document_asset_root),
            artifact=document_artifact,
            filename_prefix="document",
        )
    )
    artifacts.append(document_artifact)

    food_drafts, food_artifact = _load_isia_food_drafts(isia_food_manifest)
    candidates.extend(
        _draft_candidates(
            pool="utility_recipe",
            drafts=_select_drafts(
                food_drafts,
                expected_source_dataset="isia_food500",
                quota=POOL_QUOTAS["utility_recipe"],
                label="ISIA Food-500",
            ),
            asset_root=Path(isia_food_asset_root),
            artifact=food_artifact,
            filename_prefix="recipe",
        )
    )
    artifacts.append(food_artifact)

    ordered = _validate_complete_selection(candidates)
    published_drafts, selection_rows = _publish_intent_directories(
        ordered,
        output_root,
    )
    drafts_bytes = canonical_jsonl_bytes(published_drafts)
    drafts_path = output_root / DATASET_ASSETS_FILE
    _publish_idempotent_file(drafts_path, drafts_bytes)

    drafts_sha256 = sha256_bytes(drafts_bytes)
    manifest = _selection_manifest(
        selection_rows=selection_rows,
        artifacts=artifacts,
        drafts_bytes=drafts_bytes,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = output_root / SELECTION_MANIFEST_FILE
    _publish_idempotent_file(manifest_path, manifest_bytes)
    return PortfolioMiniBuildResult(
        output_root=output_root,
        dataset_assets_path=drafts_path,
        dataset_assets_sha256=drafts_sha256,
        selection_manifest_path=manifest_path,
        selection_manifest_sha256=sha256_bytes(manifest_bytes),
        asset_count=len(published_drafts),
    )


def build_portfolio_mini_capability_assignments(
    *,
    selection_manifest: str | Path,
    asset_catalog_dir: str | Path,
    asset_root: str | Path,
    output_path: str | Path,
) -> PortfolioMiniAssignmentResult:
    """Create canonical assignments bound to a verified catalog and selection."""

    selection_path = Path(selection_manifest)
    selection_bytes = read_stable_regular_file(
        selection_path,
        label="Portfolio mini selection manifest",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        raw_manifest = parse_canonical_json(
            selection_bytes,
            label="Portfolio mini selection manifest",
        )
    except ArtifactFormatError as error:
        raise PortfolioMiniError(str(error)) from error
    manifest = _validate_selection_manifest(raw_manifest)
    selection_sha256 = sha256_bytes(selection_bytes)

    catalog = load_asset_catalog(
        asset_catalog_dir,
        asset_root,
        verify_files=True,
    )
    catalog.require_verified_files()
    assignments = _assignments_for_selection(
        manifest["selections"],
        catalog,
        selection_sha256=selection_sha256,
    )
    assignment_bytes = canonical_jsonl_bytes(assignments)
    output_path = Path(output_path).absolute()
    _publish_idempotent_file(output_path, assignment_bytes)
    return PortfolioMiniAssignmentResult(
        output_path=output_path,
        assignment_sha256=sha256_bytes(assignment_bytes),
        assignment_count=len(assignments),
        selection_manifest_sha256=selection_sha256,
    )


def load_portfolio_mini_drafts(
    path: str | Path,
) -> tuple[DatasetAssetDraft, ...]:
    """Load the canonical draft file emitted by :func:`assemble_portfolio_mini`."""

    content = read_stable_regular_file(
        path,
        label="Portfolio mini DatasetAssetDraft file",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        rows = parse_canonical_jsonl(
            content,
            label="Portfolio mini DatasetAssetDraft file",
        )
        drafts = tuple(
            DatasetAssetDraft.model_validate(row, strict=True) for row in rows
        )
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioMiniError("Portfolio mini drafts are invalid") from error
    if len(drafts) != sum(POOL_QUOTAS.values()):
        raise PortfolioMiniError("Portfolio mini draft count is not 184")
    return drafts


def load_portfolio_mini_selection_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load the canonical, complete Portfolio mini selection manifest.

    Core preparation uses this public loader only to bind exclusions and the
    existing Wikimedia document assets.  It does not reinterpret or mutate
    any accepted mini query row.
    """

    content = read_stable_regular_file(
        path,
        label="Portfolio mini selection manifest",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        raw = parse_canonical_json(
            content,
            label="Portfolio mini selection manifest",
        )
    except ArtifactFormatError as error:
        raise PortfolioMiniError(str(error)) from error
    manifest = _validate_selection_manifest(raw)
    if content != canonical_json_bytes(manifest):
        raise PortfolioMiniError("Portfolio mini selection manifest is not canonical")
    return manifest, sha256_bytes(content)


def _load_abo_candidates(
    receipt_path: Path,
    asset_root: Path,
) -> tuple[list[_Candidate], _SourceArtifact]:
    content = read_stable_regular_file(
        receipt_path,
        label="ABO pair-audit receipt",
        max_bytes=_MAX_METADATA_BYTES,
    )
    try:
        value = parse_canonical_json(content, label="ABO pair-audit receipt")
    except ArtifactFormatError as error:
        raise PortfolioMiniError(str(error)) from error
    if not isinstance(value, dict):
        raise PortfolioMiniError("ABO pair-audit receipt must be a JSON object")
    self_hash = value.get("receipt_self_sha256")
    if not isinstance(self_hash, str) or not _SHA256.fullmatch(self_hash):
        raise PortfolioMiniError("ABO receipt self hash is invalid")
    unsigned = dict(value)
    del unsigned["receipt_self_sha256"]
    if sha256_bytes(canonical_json_bytes(unsigned)) != self_hash:
        raise PortfolioMiniError("ABO receipt self hash mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "local_pair_leakage_audit_complete"
        or value.get("formal_use_allowed") is not False
    ):
        raise PortfolioMiniError("ABO receipt is not the reviewed non-formal audit")

    retained = value.get("retained_pair_ids")
    pairs = value.get("approved_pairs")
    fingerprints = value.get("image_fingerprints")
    if not isinstance(retained, list) or not all(
        isinstance(item, str) and item for item in retained
    ):
        raise PortfolioMiniError("ABO retained pair ids are invalid")
    if retained != sorted(set(retained)):
        raise PortfolioMiniError("ABO retained pair ids must be sorted and unique")
    if value.get("retained_pair_count") != len(retained):
        raise PortfolioMiniError("ABO retained pair count mismatch")
    if len(retained) < POOL_QUOTAS["exact"]:
        raise PortfolioMiniError("ABO retained pool cannot supply 35 exact images")
    if not isinstance(pairs, list) or not isinstance(fingerprints, list):
        raise PortfolioMiniError("ABO receipt pair/fingerprint rows are invalid")
    pair_by_id = _unique_dict_rows(pairs, "pair_id", "ABO approved pairs")
    fingerprint_by_id = _unique_dict_rows(
        fingerprints,
        "image_id",
        "ABO image fingerprints",
    )

    artifact = _SourceArtifact(
        source_id="abo",
        artifact_format="canonical-pair-audit-receipt-json",
        content_sha256=sha256_bytes(content),
        bytes=len(content),
        rows=len(retained),
    )
    candidates: list[_Candidate] = []
    for index, pair_id in enumerate(retained[: POOL_QUOTAS["exact"]], start=1):
        pair = pair_by_id.get(pair_id)
        if pair is None:
            raise PortfolioMiniError(f"ABO retained pair is absent: {pair_id}")
        image_id = pair.get("main_image_id")
        local_path = pair.get("main_local_path")
        if not isinstance(image_id, str) or not isinstance(local_path, str):
            raise PortfolioMiniError(f"ABO pair is malformed: {pair_id}")
        fingerprint = fingerprint_by_id.get(image_id)
        if (
            fingerprint is None
            or fingerprint.get("local_path") != local_path
            or not isinstance(fingerprint.get("source_image_sha256"), str)
            or not isinstance(fingerprint.get("bytes"), int)
        ):
            raise PortfolioMiniError(f"ABO fingerprint binding failed: {pair_id}")
        source_path = _resolve_source_image(asset_root, local_path)
        snapshot = _read_image(source_path, f"ABO image {image_id}")
        actual_sha256 = sha256_bytes(snapshot)
        if (
            actual_sha256 != fingerprint["source_image_sha256"]
            or len(snapshot) != fingerprint["bytes"]
        ):
            raise PortfolioMiniError(f"ABO image bytes drifted: {image_id}")
        suffix = source_path.suffix.lower()
        draft = DatasetAssetDraft(
            source_dataset="abo",
            source_revision=f"pair-audit-{self_hash}",
            source_record_id=f"{pair_id}:{image_id}",
            transform_policy_version=ABO_TRANSFORM_POLICY,
            local_path=local_path,
            product_id=f"abo:{pair_id}",
            license_id=ABO_LICENSE_ID,
            attribution="Amazon Berkeley Objects (ABO)",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        )
        candidates.append(
            _Candidate(
                pool="exact",
                draft=draft,
                source_path=source_path,
                source_local_path=local_path,
                source_artifact_sha256=artifact.content_sha256,
                source_bytes=len(snapshot),
                source_sha256=actual_sha256,
                destination_name=f"abo-{index:04d}{suffix}",
            )
        )
    return candidates, artifact


def _load_inaturalist_drafts(
    manifest_path: str | Path,
) -> tuple[tuple[DatasetAssetDraft, ...], _SourceArtifact]:
    rows, artifact = _load_legacy_rows(
        manifest_path,
        _INaturalistRow,
        source_id="inaturalist",
    )
    drafts = tuple(
        DatasetAssetDraft(
            source_dataset="inaturalist",
            source_revision=f"manifest-{artifact.content_sha256}",
            source_record_id=f"photo:{row.photo_id}",
            transform_policy_version="inaturalist-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.source_url,
            attribution=row.attribution,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in rows
    )
    return drafts, artifact


def _load_commons_document_drafts(
    manifest_path: str | Path,
) -> tuple[tuple[DatasetAssetDraft, ...], _SourceArtifact]:
    rows, artifact = _load_legacy_rows(
        manifest_path,
        _CommonsDocumentRow,
        source_id="wikimedia_commons_documents",
    )
    drafts = tuple(
        DatasetAssetDraft(
            source_dataset="wikimedia_commons_documents",
            source_revision=f"manifest-{artifact.content_sha256}",
            source_record_id=f"page:{row.page_id}",
            transform_policy_version="wikimedia-document-jpeg-materialize-v1",
            local_path=row.image,
            license_id=row.license,
            source_url=row.description_url,
            attribution=row.attribution,
            cloud_upload_allowed=row.cloud_upload_allowed,
            public_demo_allowed=row.public_demo_allowed,
        )
        for row in rows
    )
    return drafts, artifact


def _load_isia_food_drafts(
    manifest_path: str | Path,
) -> tuple[tuple[DatasetAssetDraft, ...], _SourceArtifact]:
    rows, artifact = _load_legacy_rows(
        manifest_path,
        _ISIAFoodRow,
        source_id="isia_food500",
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
    return drafts, artifact


def _load_legacy_rows(
    manifest_path: str | Path,
    model: type[_StrictModel],
    *,
    source_id: str,
) -> tuple[tuple[Any, ...], _SourceArtifact]:
    content = read_stable_regular_file(
        manifest_path,
        label=f"{source_id} manifest",
        max_bytes=_MAX_METADATA_BYTES,
    )
    if not content or not content.endswith(b"\n"):
        raise PortfolioMiniError(f"{source_id} manifest must be non-empty JSONL")
    parsed: list[Any] = []
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        if line == b"\n":
            raise PortfolioMiniError(
                f"{source_id} manifest has blank line {line_number}"
            )
        try:
            value = parse_strict_json(
                line,
                label=f"{source_id} manifest line {line_number}",
            )
            parsed.append(model.model_validate(value, strict=True))
        except (ArtifactFormatError, ValidationError) as error:
            raise PortfolioMiniError(
                f"{source_id} manifest line {line_number} is invalid"
            ) from error
    artifact = _SourceArtifact(
        source_id=source_id,
        artifact_format="strict-source-manifest-jsonl",
        content_sha256=sha256_bytes(content),
        bytes=len(content),
        rows=len(parsed),
    )
    return tuple(parsed), artifact


def _select_drafts(
    drafts: Iterable[DatasetAssetDraft],
    *,
    expected_source_dataset: str,
    quota: int,
    label: str,
) -> tuple[DatasetAssetDraft, ...]:
    return _ordered_drafts(
        drafts,
        expected_source_dataset=expected_source_dataset,
        minimum_count=quota,
        label=label,
    )[:quota]


def _ordered_drafts(
    drafts: Iterable[DatasetAssetDraft],
    *,
    expected_source_dataset: str,
    minimum_count: int,
    label: str,
) -> tuple[DatasetAssetDraft, ...]:
    rows = tuple(drafts)
    if len(rows) < minimum_count:
        raise PortfolioMiniError(f"{label} cannot supply {minimum_count} assets")
    seen_records: set[str] = set()
    seen_paths: set[str] = set()
    for draft in rows:
        if draft.source_dataset != expected_source_dataset:
            raise PortfolioMiniError(f"{label} source_dataset drifted")
        if draft.derivation_parent_asset_ids or (
            draft.derivation_parent_asset_id is not None
        ):
            raise PortfolioMiniError(
                f"{label} mini query assets cannot depend on omitted parents"
            )
        if draft.source_record_id in seen_records:
            raise PortfolioMiniError(f"{label} source_record_id is duplicated")
        if draft.local_path in seen_paths:
            raise PortfolioMiniError(f"{label} local_path is duplicated")
        seen_records.add(draft.source_record_id)
        seen_paths.add(draft.local_path)
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.source_record_id,
                item.local_path,
            ),
        )
    )


def _draft_candidates(
    *,
    pool: Pool,
    drafts: Iterable[DatasetAssetDraft],
    asset_root: Path,
    artifact: _SourceArtifact,
    filename_prefix: str,
    preserve_source_name: bool = False,
    quota: int | None = None,
    skip_duplicate_content: bool = False,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    seen_content: set[str] = set()
    for draft in drafts:
        source_path = _resolve_source_image(asset_root, draft.local_path)
        content = _read_image(
            source_path,
            f"{draft.source_dataset} image {draft.source_record_id}",
        )
        content_sha256 = sha256_bytes(content)
        if skip_duplicate_content and content_sha256 in seen_content:
            continue
        seen_content.add(content_sha256)
        index = len(candidates) + 1
        suffix = source_path.suffix.lower()
        destination_name = (
            source_path.name
            if preserve_source_name
            else f"{filename_prefix}-{index:04d}{suffix}"
        )
        candidates.append(
            _Candidate(
                pool=pool,
                draft=draft,
                source_path=source_path,
                source_local_path=draft.local_path,
                source_artifact_sha256=artifact.content_sha256,
                source_bytes=len(content),
                source_sha256=content_sha256,
                destination_name=destination_name,
            )
        )
        if quota is not None and len(candidates) == quota:
            break
    if quota is not None and len(candidates) != quota:
        raise PortfolioMiniError(
            f"{artifact.source_id} cannot supply {quota} content-unique assets"
        )
    return candidates


def _validate_complete_selection(
    candidates: Iterable[_Candidate],
) -> tuple[_Candidate, ...]:
    candidates = tuple(candidates)
    counts = Counter(item.pool for item in candidates)
    if counts != Counter(POOL_QUOTAS):
        raise PortfolioMiniError(
            f"Portfolio mini pool counts differ: {dict(sorted(counts.items()))}"
        )
    content_hashes = [item.source_sha256 for item in candidates]
    if len(content_hashes) != len(set(content_hashes)):
        raise PortfolioMiniError("Portfolio mini contains exact duplicate image bytes")
    destination_keys = [
        (POOL_BINDINGS[item.pool][2], item.destination_name.casefold())
        for item in candidates
    ]
    if len(destination_keys) != len(set(destination_keys)):
        raise PortfolioMiniError("Portfolio mini destination paths collide")
    order = {pool: index for index, pool in enumerate(POOL_ORDER)}
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                order[item.pool],
                item.destination_name,
                item.draft.source_record_id,
            ),
        )
    )


def _publish_intent_directories(
    candidates: tuple[_Candidate, ...],
    output_root: Path,
) -> tuple[tuple[DatasetAssetDraft, ...], list[dict[str, Any]]]:
    by_directory: dict[str, list[_Candidate]] = {}
    for item in candidates:
        by_directory.setdefault(POOL_BINDINGS[item.pool][2], []).append(item)

    drafts: list[DatasetAssetDraft] = []
    selections: list[dict[str, Any]] = []
    for directory_name in (
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    ):
        items = by_directory[directory_name]
        destination = output_root / directory_name
        _publish_or_adopt_directory(items, destination)
        for item in items:
            image_path = (
                PurePosixPath("query_images")
                / directory_name
                / item.destination_name
            ).as_posix()
            published = item.draft.model_copy(
                update={
                    "local_path": image_path,
                    "derivation_parent_asset_ids": [],
                    "derivation_parent_asset_id": None,
                }
            )
            drafts.append(published)
            intent, capability, _ = POOL_BINDINGS[item.pool]
            selection_id = (
                "portfolio-mini."
                + sha256_bytes(
                    canonical_json_bytes(
                        {
                            "image_path": image_path,
                            "pool": item.pool,
                            "source_artifact_sha256": (
                                item.source_artifact_sha256
                            ),
                            "source_record_id": item.draft.source_record_id,
                            "source_sha256": item.source_sha256,
                        }
                    )
                )
            )
            selections.append(
                {
                    "canonical_capability": capability,
                    "canonical_intent": intent,
                    "cloud_upload_allowed": item.draft.cloud_upload_allowed,
                    "image_bytes": item.source_bytes,
                    "image_path": image_path,
                    "image_sha256": item.source_sha256,
                    "license_id": item.draft.license_id,
                    "pool": item.pool,
                    "product_id": item.draft.product_id,
                    "public_demo_allowed": item.draft.public_demo_allowed,
                    "selection_id": selection_id,
                    "source_artifact_sha256": item.source_artifact_sha256,
                    "source_dataset": item.draft.source_dataset,
                    "source_local_path": item.source_local_path,
                    "source_record_id": item.draft.source_record_id,
                    "source_revision": item.draft.source_revision,
                    "transform_policy_version": (
                        item.draft.transform_policy_version
                    ),
                }
            )
    return tuple(drafts), selections


def _publish_or_adopt_directory(
    items: list[_Candidate],
    destination: Path,
) -> None:
    expected = {item.destination_name: item for item in items}
    if os.path.lexists(destination):
        _require_real_directory(destination, f"intent directory {destination.name}")
        actual_images: set[str] = set()
        for path in destination.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PortfolioMiniError(
                    f"intent directory contains a symlink: {path}"
                )
            if stat.S_ISREG(metadata.st_mode) and path.suffix.lower() in _IMAGE_SUFFIXES:
                actual_images.add(path.relative_to(destination).as_posix())
        if actual_images != set(expected):
            raise FileExistsError(
                f"existing intent directory conflicts with selection: {destination}"
            )
        for name, item in expected.items():
            content = _read_image(destination / name, f"published image {name}")
            if (
                len(content) != item.source_bytes
                or sha256_bytes(content) != item.source_sha256
            ):
                raise FileExistsError(
                    f"existing intent image conflicts with selection: {destination / name}"
                )
        return

    staging = new_staging_directory(destination)
    try:
        for name, item in expected.items():
            content = _read_image(item.source_path, f"source image {name}")
            if (
                len(content) != item.source_bytes
                or sha256_bytes(content) != item.source_sha256
            ):
                raise PortfolioMiniError(f"source image changed during copy: {name}")
            atomic_create_file(staging / name, content)
        atomic_publish_new_directory(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _selection_manifest(
    *,
    selection_rows: list[dict[str, Any]],
    artifacts: list[_SourceArtifact],
    drafts_bytes: bytes,
) -> dict[str, Any]:
    pool_counts = Counter(row["pool"] for row in selection_rows)
    intent_counts = Counter(row["canonical_intent"] for row in selection_rows)
    capability_counts = Counter(
        row["canonical_capability"] for row in selection_rows
    )
    return {
        "asset_count": len(selection_rows),
        "assignment_count_expected": 254,
        "capability_asset_counts": dict(sorted(capability_counts.items())),
        "dataset_assets": {
            "bytes": len(drafts_bytes),
            "path": DATASET_ASSETS_FILE,
            "rows": len(selection_rows),
            "sha256": sha256_bytes(drafts_bytes),
        },
        "formal_eligible": False,
        "formal_status": "non_formal",
        "intent_asset_counts": dict(sorted(intent_counts.items())),
        "permission_policy": "preserve-source-value-no-upgrade-v1",
        "pool_counts": dict(sorted(pool_counts.items())),
        "query_image_root": "query_images",
        "schema_version": 1,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "selections": selection_rows,
        "source_artifacts": [
            item.as_json()
            for item in sorted(artifacts, key=lambda value: value.source_id)
        ],
        "track": "portfolio",
    }


def _validate_selection_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PortfolioMiniError("selection manifest must be an object")
    if (
        value.get("schema_version") != 1
        or value.get("selection_policy_version") != SELECTION_POLICY_VERSION
        or value.get("track") != "portfolio"
        or value.get("formal_eligible") is not False
        or value.get("formal_status") != "non_formal"
        or value.get("asset_count") != sum(POOL_QUOTAS.values())
        or value.get("assignment_count_expected") != 254
    ):
        raise PortfolioMiniError("selection manifest boundary fields drifted")
    selections = value.get("selections")
    if not isinstance(selections, list) or len(selections) != 184:
        raise PortfolioMiniError("selection manifest must contain 184 selections")
    required = {
        "canonical_capability",
        "canonical_intent",
        "cloud_upload_allowed",
        "image_bytes",
        "image_path",
        "image_sha256",
        "license_id",
        "pool",
        "product_id",
        "public_demo_allowed",
        "selection_id",
        "source_artifact_sha256",
        "source_dataset",
        "source_local_path",
        "source_record_id",
        "source_revision",
        "transform_policy_version",
    }
    counts: Counter[str] = Counter()
    paths: set[str] = set()
    for row in selections:
        if not isinstance(row, dict) or set(row) != required:
            raise PortfolioMiniError("selection row schema drifted")
        pool = row["pool"]
        if pool not in POOL_QUOTAS:
            raise PortfolioMiniError("selection row has unknown pool")
        intent, capability, directory = POOL_BINDINGS[pool]
        if (
            row["canonical_intent"] != intent
            or row["canonical_capability"] != capability
            or not isinstance(row["image_path"], str)
            or not row["image_path"].startswith(f"query_images/{directory}/")
            or not isinstance(row["image_sha256"], str)
            or not _SHA256.fullmatch(row["image_sha256"])
            or not isinstance(row["source_artifact_sha256"], str)
            or not _SHA256.fullmatch(row["source_artifact_sha256"])
        ):
            raise PortfolioMiniError("selection row binding drifted")
        if row["image_path"] in paths:
            raise PortfolioMiniError("selection image_path is duplicated")
        paths.add(row["image_path"])
        counts[pool] += 1
    if counts != Counter(POOL_QUOTAS):
        raise PortfolioMiniError("selection pool counts drifted")
    return value


def _assignments_for_selection(
    selections: list[dict[str, Any]],
    catalog: AssetCatalog,
    *,
    selection_sha256: str,
) -> tuple[CapabilityAssignment, ...]:
    taxonomy = load_default_taxonomy_registry()
    task_spec = load_mvp_task_specification_v1()
    assignments: list[CapabilityAssignment] = []
    extra_exact = (
        ("divergent_rec", "product.style_recommendation"),
        ("encyclopedia", "knowledge.visual_encyclopedia"),
    )
    for row in selections:
        resolution = catalog.resolve_path(row["image_path"])
        asset = resolution.asset
        if (
            asset.sha256 != row["image_sha256"]
            or asset.source_dataset != row["source_dataset"]
            or asset.source_revision != row["source_revision"]
            or asset.source_record_id != row["source_record_id"]
            or asset.transform_policy_version
            != row["transform_policy_version"]
            or asset.product_id != row["product_id"]
            or asset.license_id != row["license_id"]
            or asset.cloud_upload_allowed != row["cloud_upload_allowed"]
            or asset.public_demo_allowed != row["public_demo_allowed"]
        ):
            raise PortfolioMiniError(
                f"selection/catalog binding drifted: {row['image_path']}"
            )
        bindings = [
            (row["canonical_intent"], row["canonical_capability"]),
        ]
        if row["pool"] == "exact":
            bindings.extend(extra_exact)
        for intent, capability_id in bindings:
            capability = taxonomy.capabilities_by_id[capability_id]
            task = task_spec.capabilities_by_id[capability_id]
            identity = {
                "asset_id": asset.asset_id,
                "capability": capability_id,
                "image_path": row["image_path"],
                "intent": intent,
                "policy": ASSIGNMENT_POLICY_VERSION,
                "selection_sha256": selection_sha256,
            }
            assignments.append(
                CapabilityAssignment(
                    assignment_id=(
                        "portfolio-mini.assignment."
                        + sha256_bytes(canonical_json_bytes(identity))
                    ),
                    asset_id=asset.asset_id,
                    image_path=row["image_path"],
                    canonical_intent=intent,
                    canonical_capability=capability_id,
                    acceptable_capabilities=(capability_id,),
                    requires_card=capability.requires_card,
                    allowed_tools=task.allowed_tools,
                    taxonomy_version=TAXONOMY_VERSION,
                    taxonomy_sha256=DEFAULT_TAXONOMY_SHA256,
                    task_spec_version=PHASE3_TASK_SPEC_VERSION,
                    task_spec_sha256=PHASE3_TASK_SPEC_SHA256,
                    source_ref="project-choice:portfolio-mini-assets-v1",
                    source_artifact_sha256=selection_sha256,
                    rationale="Predeclared portfolio asset-to-capability mapping.",
                )
            )
    assignments.sort(
        key=lambda item: (
            item.image_path,
            item.canonical_intent,
            item.canonical_capability,
        )
    )
    if len(assignments) != 254:
        raise PortfolioMiniError("Portfolio mini assignment count is not 254")
    return tuple(assignments)


def _stable_artifact(
    path: Path,
    *,
    source_id: str,
    artifact_format: str,
    rows: int,
) -> _SourceArtifact:
    content = read_stable_regular_file(
        path,
        label=f"{source_id} artifact",
        max_bytes=_MAX_METADATA_BYTES,
    )
    return _SourceArtifact(
        source_id=source_id,
        artifact_format=artifact_format,
        content_sha256=sha256_bytes(content),
        bytes=len(content),
        rows=rows,
    )


def _unique_dict_rows(
    rows: list[Any],
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise PortfolioMiniError(f"{label} contains a malformed row")
        identity = row[key]
        if identity in result:
            raise PortfolioMiniError(f"{label} contains duplicate {key}")
        result[identity] = row
    return result


def _resolve_source_image(root: Path, local_path: str) -> Path:
    root = root.absolute()
    _require_real_directory(root, "source asset root")
    normalized = _canonical_relative_path(local_path)
    current = root
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise PortfolioMiniError(f"source image is unavailable: {local_path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise PortfolioMiniError(f"source path contains a symlink: {local_path}")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PortfolioMiniError(f"source path parent is not a directory: {local_path}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise PortfolioMiniError(f"source image is not a regular file: {local_path}")
    if current.suffix.lower() not in _IMAGE_SUFFIXES:
        raise PortfolioMiniError(f"unsupported source image suffix: {local_path}")
    return current


def _canonical_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != unicodedata.normalize("NFC", value)
        or "\\" in value
    ):
        raise PortfolioMiniError("source local_path must be canonical POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PortfolioMiniError("source local_path must be a safe relative path")
    if path.as_posix() != value:
        raise PortfolioMiniError("source local_path is not normalized")
    return value


def _read_image(path: Path, label: str) -> bytes:
    try:
        content = read_stable_regular_file(
            path,
            label=label,
            max_bytes=_MAX_IMAGE_BYTES,
        )
    except ArtifactFormatError as error:
        raise PortfolioMiniError(str(error)) from error
    if not content:
        raise PortfolioMiniError(f"{label} is empty")
    return content


def _ensure_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _require_real_directory(path, "Portfolio mini output root")


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PortfolioMiniError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PortfolioMiniError(f"{label} must be a real directory")


def _publish_idempotent_file(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        try:
            existing = read_stable_regular_file(
                path,
                label=f"existing artifact {path.name}",
                max_bytes=_MAX_METADATA_BYTES,
            )
        except ArtifactFormatError as error:
            raise FileExistsError(f"existing artifact is unsafe: {path}") from error
        if existing != content:
            raise FileExistsError(
                f"existing artifact conflicts with deterministic output: {path}"
            )
        return
    atomic_create_file(path, content)


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PortfolioMiniError(f"{label} must be lowercase hexadecimal")
    return value


__all__ = [
    "ASSIGNMENT_POLICY_VERSION",
    "DATASET_ASSETS_FILE",
    "POOL_BINDINGS",
    "POOL_QUOTAS",
    "PortfolioMiniAssignmentResult",
    "PortfolioMiniBuildResult",
    "PortfolioMiniError",
    "SELECTION_MANIFEST_FILE",
    "SELECTION_POLICY_VERSION",
    "assemble_portfolio_mini",
    "build_portfolio_mini_capability_assignments",
    "load_portfolio_mini_drafts",
]
