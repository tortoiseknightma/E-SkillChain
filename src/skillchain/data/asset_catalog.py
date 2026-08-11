"""Deterministic, content-verified asset catalog and leakage components.

The catalog is the trust boundary between heterogeneous Phase 1 adapters and
corpus planning.  It recomputes fingerprints from final published bytes, joins
typed provenance/product/lineage relations, and freezes the resulting
connected components in create-only artifacts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Iterable, Literal

import imagehash
from PIL import Image, ImageOps
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.schemas import DatasetAsset, Intent

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
LeakageKeyKind = Literal[
    "content_sha256",
    "source_record",
    "product",
    "near_duplicate_cluster",
    "derivation_parent",
]

CATALOG_POLICY_VERSION = "dataset-asset-catalog-v1"
LEAKAGE_POLICY_VERSION = "dataset-asset-components-v1"
_ASSET_FILE = "assets.jsonl"
_COMPONENT_FILE = "components.jsonl"
_MANIFEST_FILE = "manifest.json"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class CatalogError(ValueError):
    pass


class CatalogIntegrityError(CatalogError):
    pass


class DuplicateAssetIdError(CatalogError):
    pass


class DuplicateLocalPathError(CatalogError):
    pass


class AssetNotCatalogedError(CatalogError):
    pass


class AmbiguousAssetPathError(CatalogError):
    pass


class AssetReferenceMismatchError(CatalogError):
    pass


class IncompleteLeakageEvidenceError(CatalogError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _nonblank(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} 不得为空")
    return value


class DatasetAssetDraft(StrictModel):
    """Adapter-owned provenance before final-byte fingerprints are computed."""

    schema_version: Literal[1] = 1
    source_dataset: str
    source_revision: str
    source_record_id: str
    transform_policy_version: str = "identity-v1"
    local_path: str
    product_id: str | None = None
    derivation_parent_asset_ids: list[str] = Field(default_factory=list)
    derivation_parent_asset_id: str | None = None
    license_id: str
    source_url: str | None = None
    attribution: str | None = None
    cloud_upload_allowed: bool | None = None
    public_demo_allowed: bool = False

    @field_validator(
        "source_dataset",
        "source_revision",
        "source_record_id",
        "transform_policy_version",
        "local_path",
        "license_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "product_id",
        "derivation_parent_asset_id",
        "source_url",
        "attribution",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("derivation_parent_asset_ids")
    @classmethod
    def validate_parent_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_nonblank(item, "derivation_parent_asset_ids") for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("derivation_parent_asset_ids 不得重复")
        return sorted(cleaned)

    @model_validator(mode="after")
    def normalize_parent_alias(self):
        if (
            self.derivation_parent_asset_id is not None
            and self.derivation_parent_asset_id in self.derivation_parent_asset_ids
        ):
            raise ValueError("singular derivation parent 不得在列表中重复")
        return self


class NearDuplicatePolicy(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = "phash64-exif-rgb-singlelink-v1"
    algorithm: Literal["imagehash.phash"] = "imagehash.phash"
    hash_size: Literal[8] = 8
    max_phash_hamming_distance: int = Field(default=4, ge=0, le=16)
    normalization: Literal["exif-transpose-rgb-v1"] = "exif-transpose-rgb-v1"
    linkage: Literal["connected-components"] = "connected-components"
    pillow_version: str = Field(default_factory=lambda: package_version("pillow"))
    imagehash_version: str = Field(default_factory=lambda: package_version("imagehash"))
    scipy_version: str = Field(default_factory=lambda: package_version("scipy"))

    @field_validator(
        "policy_version",
        "pillow_version",
        "imagehash_version",
        "scipy_version",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @property
    def max_hamming_distance(self) -> int:
        return self.max_phash_hamming_distance


DEFAULT_NEAR_DUPLICATE_POLICY = NearDuplicatePolicy()


class LeakageKey(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: LeakageKeyKind
    namespace: str
    value: str

    @field_validator("namespace", "value")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.kind, self.namespace, self.value)


class LeakageComponent(StrictModel):
    schema_version: Literal[1] = 1
    component_id: str
    asset_ids: tuple[str, ...]
    linkage_keys: tuple[LeakageKey, ...]
    key_counts: dict[LeakageKeyKind, int]

    @model_validator(mode="after")
    def validate_component(self):
        if not self.component_id.startswith("leakage.assetcc.v1."):
            raise ValueError("component_id policy prefix 非法")
        if not self.asset_ids or tuple(sorted(self.asset_ids)) != self.asset_ids:
            raise ValueError("asset_ids 必须非空且排序")
        if len(self.asset_ids) != len(set(self.asset_ids)):
            raise ValueError("asset_ids 不得重复")
        if (
            tuple(sorted(self.linkage_keys, key=lambda key: key.sort_key))
            != self.linkage_keys
        ):
            raise ValueError("linkage_keys 必须按 typed key 排序")
        return self


class CatalogFile(StrictModel):
    path: str
    count: int = Field(ge=0)
    bytes: int = Field(ge=0)
    sha256: Sha256


class SourceSnapshot(StrictModel):
    source_dataset: str
    source_revision: str
    asset_count: int = Field(gt=0)


class AssetCatalogManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_asset_schema_version: Literal[2] = 2
    catalog_policy_version: Literal["dataset-asset-catalog-v1"] = CATALOG_POLICY_VERSION
    leakage_policy_version: Literal["dataset-asset-components-v1"] = (
        LEAKAGE_POLICY_VERSION
    )
    near_duplicate_policy: NearDuplicatePolicy
    coverage_roots: tuple[str, ...] = Field(min_length=1)
    sources: tuple[SourceSnapshot, ...]
    assets: CatalogFile
    components: CatalogFile
    asset_count: int = Field(gt=0)
    component_count: int = Field(gt=0)
    near_duplicate_cluster_count: int = Field(gt=0)
    relation_counts: dict[str, int]
    component_size_histogram: dict[int, int]
    unknown_cloud_permission_count: int = Field(ge=0)
    public_demo_allowed_count: int = Field(ge=0)
    catalog_sha256: Sha256

    @field_validator("coverage_roots")
    @classmethod
    def validate_coverage_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(_canonical_coverage_root(root) for root in value)
        if canonical != tuple(sorted(set(canonical))):
            raise ValueError("coverage_roots 必须 canonical、排序且不重复")
        return canonical


@dataclass(frozen=True)
class AssetResolution:
    asset: DatasetAsset
    leakage_group_id: str

    def __getattr__(self, name: str):
        return getattr(self.asset, name)


class QueryAssetReference(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    intent: Intent
    asset_id: str

    @field_validator("query_id", "asset_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class GalleryAssetReference(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    image_path: str

    @field_validator("asset_id", "image_path")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class QueryGalleryLeakageReport(StrictModel):
    query_count: int
    gallery_count: int
    exact_or_near_duplicate_violations: dict[str, list[str]]
    missing_exact_match_positive: list[str]
    missing_divergent_product_identity: list[str]
    divergent_product_exclusions: dict[str, list[str]]
    divergent_unknown_product_exclusions: dict[str, list[str]]

    @property
    def violation_count(self) -> int:
        return (
            len(self.exact_or_near_duplicate_violations)
            + len(self.missing_exact_match_positive)
            + len(self.missing_divergent_product_identity)
            + len(self.divergent_product_exclusions)
            + len(self.divergent_unknown_product_exclusions)
        )


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


@dataclass(frozen=True)
class AssetCatalog:
    root: Path
    asset_root: Path
    manifest: AssetCatalogManifest
    assets: tuple[DatasetAsset, ...]
    components: tuple[LeakageComponent, ...]
    _by_asset_id: dict[str, AssetResolution]
    _by_path: dict[str, AssetResolution]
    _members_by_component: dict[str, tuple[DatasetAsset, ...]]
    _files_verified: bool

    @property
    def catalog_sha256(self) -> str:
        return self.manifest.catalog_sha256

    @property
    def leakage_policy_version(self) -> str:
        return self.manifest.leakage_policy_version

    def require_verified_files(self) -> None:
        """Reject metadata-only catalogs at every formal boundary."""

        if not self._files_verified:
            raise CatalogIntegrityError(
                "formal operation requires load_asset_catalog(..., verify_files=True)"
            )

    def verify_asset_ids(self, asset_ids: Iterable[str]) -> None:
        """Recheck only referenced bytes on a metadata-validated catalog."""

        unique = {
            asset_id: self.resolve_asset_id(asset_id).asset
            for asset_id in set(asset_ids)
        }
        _validate_asset_files(unique.values(), self.asset_root)

    def resolve_path(self, local_path: str | Path) -> AssetResolution:
        key = _canonical_local_path(local_path)
        try:
            return self._by_path[key]
        except KeyError:
            raise AssetNotCatalogedError(f"asset path 未登记: {key}") from None

    def resolve_asset_id(self, asset_id: str) -> AssetResolution:
        try:
            return self._by_asset_id[asset_id]
        except KeyError:
            raise AssetNotCatalogedError(f"asset_id 未登记: {asset_id}") from None

    def component_for_asset(self, asset_id: str) -> str:
        return self.resolve_asset_id(asset_id).leakage_group_id

    def component_members(self, component_id: str) -> tuple[DatasetAsset, ...]:
        try:
            return self._members_by_component[component_id]
        except KeyError:
            raise AssetNotCatalogedError(
                f"leakage component 未登记: {component_id}"
            ) from None

    def verify_reference(
        self,
        asset_id: str,
        image_path: str | Path,
        leakage_group_id: str | None = None,
    ) -> AssetResolution:
        by_id = self.resolve_asset_id(asset_id)
        by_path = self.resolve_path(image_path)
        if by_id.asset.asset_id != by_path.asset.asset_id:
            raise AssetReferenceMismatchError(
                "asset_id 与 image_path 指向不同 catalog asset"
            )
        if leakage_group_id is not None and by_id.leakage_group_id != leakage_group_id:
            raise AssetReferenceMismatchError("leakage_group_id 与 catalog 不一致")
        return by_id


def inventory_dataset_asset(
    draft: DatasetAssetDraft,
    asset_root: str | Path,
) -> DatasetAsset:
    """Compute identity and fingerprints from the final published image bytes."""

    asset_root = Path(asset_root).resolve()
    local_path = _canonical_local_path(draft.local_path)
    path = _resolve_asset_file(asset_root, local_path)
    sha256, phash = _fingerprint_file(path)
    asset_id = _expected_asset_id(
        source_dataset=draft.source_dataset,
        source_record_id=draft.source_record_id,
        transform_policy_version=draft.transform_policy_version,
        sha256=sha256,
    )
    parent_ids = list(draft.derivation_parent_asset_ids)
    if draft.derivation_parent_asset_id is not None:
        parent_ids.append(draft.derivation_parent_asset_id)
    parent_ids.sort()
    return DatasetAsset(
        schema_version=2,
        asset_id=asset_id,
        source_dataset=draft.source_dataset,
        source_revision=draft.source_revision,
        source_record_id=draft.source_record_id,
        transform_policy_version=draft.transform_policy_version,
        local_path=local_path,
        sha256=sha256,
        phash=phash,
        near_duplicate_cluster_id=None,
        product_id=draft.product_id,
        derivation_parent_asset_ids=parent_ids,
        license_id=draft.license_id,
        source_url=draft.source_url,
        attribution=draft.attribution,
        cloud_upload_allowed=draft.cloud_upload_allowed,
        public_demo_allowed=draft.public_demo_allowed,
    )


def publish_asset_catalog(
    assets: Iterable[DatasetAsset],
    output_dir: str | Path,
    asset_root: str | Path,
    policy: NearDuplicatePolicy = DEFAULT_NEAR_DUPLICATE_POLICY,
    *,
    coverage_roots: Iterable[str | Path],
) -> Path:
    """Verify, build and create an immutable catalog directory."""

    output_dir = Path(output_dir)
    asset_root = Path(asset_root).resolve()
    coverage = tuple(
        sorted({_canonical_coverage_root(path) for path in coverage_roots})
    )
    if not coverage:
        raise IncompleteLeakageEvidenceError(
            "formal asset catalog 必须声明至少一个 coverage root"
        )
    normalized_assets, components, relation_counts = _build_catalog_records(
        tuple(assets), policy
    )
    _validate_asset_files(normalized_assets, asset_root)
    _validate_coverage(normalized_assets, asset_root, coverage)
    assets_bytes = _canonical_jsonl_bytes(normalized_assets)
    components_bytes = _canonical_jsonl_bytes(components)
    manifest = _build_manifest(
        normalized_assets,
        components,
        assets_bytes,
        components_bytes,
        relation_counts,
        policy,
        coverage,
    )
    manifest_bytes = _canonical_json_bytes(manifest.model_dump(mode="json"))

    if output_dir.exists():
        loaded = load_asset_catalog(output_dir, asset_root, verify_files=True)
        if loaded.manifest == manifest:
            return output_dir
        raise FileExistsError("asset catalog 目标已存在且内容不同，拒绝覆盖")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        _exclusive_write(staging / _ASSET_FILE, assets_bytes)
        _exclusive_write(staging / _COMPONENT_FILE, components_bytes)
        _exclusive_write(staging / _MANIFEST_FILE, manifest_bytes)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def load_asset_catalog(
    catalog_dir: str | Path,
    asset_root: str | Path,
    *,
    verify_files: bool = True,
) -> AssetCatalog:
    catalog_dir = Path(catalog_dir)
    asset_root = Path(asset_root).resolve()
    try:
        manifest_bytes = (catalog_dir / _MANIFEST_FILE).read_bytes()
        assets_bytes = (catalog_dir / _ASSET_FILE).read_bytes()
        components_bytes = (catalog_dir / _COMPONENT_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise CatalogIntegrityError(f"asset catalog 文件缺失: {exc.filename}") from None
    try:
        manifest = AssetCatalogManifest.model_validate_json(manifest_bytes)
        assets = tuple(_parse_jsonl(assets_bytes, DatasetAsset, "assets.jsonl"))
        components = tuple(
            _parse_jsonl(components_bytes, LeakageComponent, "components.jsonl")
        )
    except (ValidationError, ValueError) as exc:
        raise CatalogIntegrityError(f"asset catalog schema 校验失败: {exc}") from exc

    if manifest_bytes != _canonical_json_bytes(manifest.model_dump(mode="json")):
        raise CatalogIntegrityError("manifest.json 不是 canonical JSON")
    _verify_manifest_self_hash(manifest)
    _verify_descriptor(manifest.assets, assets_bytes, len(assets), _ASSET_FILE)
    _verify_descriptor(
        manifest.components, components_bytes, len(components), _COMPONENT_FILE
    )
    rebuilt_assets, rebuilt_components, relation_counts = _build_catalog_records(
        assets, manifest.near_duplicate_policy
    )
    if _canonical_jsonl_bytes(rebuilt_assets) != assets_bytes:
        raise CatalogIntegrityError("assets.jsonl 与重算 near-duplicate cluster 不一致")
    if _canonical_jsonl_bytes(rebuilt_components) != components_bytes:
        raise CatalogIntegrityError("components.jsonl 与 typed leakage keys 不一致")
    if relation_counts != manifest.relation_counts:
        raise CatalogIntegrityError("manifest relation_counts 与重算结果不一致")
    rebuilt_manifest = _build_manifest(
        rebuilt_assets,
        rebuilt_components,
        assets_bytes,
        components_bytes,
        relation_counts,
        manifest.near_duplicate_policy,
        manifest.coverage_roots,
    )
    if rebuilt_manifest != manifest:
        raise CatalogIntegrityError("manifest 与 catalog 记录的重算结果不一致")
    if verify_files:
        _validate_asset_files(assets, asset_root)
        _validate_coverage(assets, asset_root, manifest.coverage_roots)

    by_component: dict[str, str] = {}
    members_by_component: dict[str, tuple[DatasetAsset, ...]] = {}
    assets_by_id = {asset.asset_id: asset for asset in assets}
    for component in components:
        members = tuple(assets_by_id[asset_id] for asset_id in component.asset_ids)
        members_by_component[component.component_id] = members
        for asset_id in component.asset_ids:
            if asset_id in by_component:
                raise CatalogIntegrityError(f"asset 出现在多个 component: {asset_id}")
            by_component[asset_id] = component.component_id
    if set(by_component) != set(assets_by_id):
        raise CatalogIntegrityError("component 未完整覆盖 assets")

    by_asset_id = {
        asset_id: AssetResolution(asset, by_component[asset_id])
        for asset_id, asset in assets_by_id.items()
    }
    by_path: dict[str, AssetResolution] = {}
    for resolution in by_asset_id.values():
        canonical = _canonical_local_path(resolution.asset.local_path)
        if canonical in by_path:
            raise AmbiguousAssetPathError(
                f"canonical local_path 多义: {resolution.asset.local_path}"
            )
        by_path[canonical] = resolution
    return AssetCatalog(
        root=catalog_dir,
        asset_root=asset_root,
        manifest=manifest,
        assets=assets,
        components=components,
        _by_asset_id=by_asset_id,
        _by_path=by_path,
        _members_by_component=members_by_component,
        _files_verified=verify_files,
    )


def audit_query_gallery_eligibility(
    query_refs: Iterable[QueryAssetReference],
    gallery_asset_ids: Iterable[str],
    catalog: AssetCatalog,
) -> QueryGalleryLeakageReport:
    catalog.require_verified_files()
    query_refs = tuple(query_refs)
    if len({ref.query_id for ref in query_refs}) != len(query_refs):
        raise CatalogError("query_id 不得重复")
    gallery_asset_ids = tuple(gallery_asset_ids)
    if len(set(gallery_asset_ids)) != len(gallery_asset_ids):
        raise CatalogError("gallery_asset_ids 不得重复")
    catalog.verify_asset_ids(
        [*(ref.asset_id for ref in query_refs), *gallery_asset_ids]
    )
    gallery = [
        catalog.resolve_asset_id(asset_id).asset for asset_id in gallery_asset_ids
    ]
    forbidden_group_by_asset = _query_gallery_forbidden_groups(catalog.assets)
    gallery_by_forbidden_group: dict[str, list[str]] = defaultdict(list)
    gallery_by_product: dict[tuple[str, str], list[DatasetAsset]] = defaultdict(list)
    unknown_product_gallery: list[str] = []
    for asset in gallery:
        gallery_by_forbidden_group[forbidden_group_by_asset[asset.asset_id]].append(
            asset.asset_id
        )
        if asset.product_id is not None:
            gallery_by_product[(asset.source_dataset, asset.product_id)].append(asset)
        else:
            unknown_product_gallery.append(asset.asset_id)
    for asset_ids in gallery_by_forbidden_group.values():
        asset_ids.sort()

    duplicate_violations: dict[str, list[str]] = {}
    missing_positive: list[str] = []
    missing_divergent_identity: list[str] = []
    divergent_exclusions: dict[str, list[str]] = {}
    divergent_unknown_exclusions: dict[str, list[str]] = {}
    for ref in query_refs:
        query = catalog.resolve_asset_id(ref.asset_id).asset
        forbidden = gallery_by_forbidden_group.get(
            forbidden_group_by_asset[query.asset_id],
            [],
        )
        if forbidden:
            duplicate_violations[ref.query_id] = forbidden

        product_gallery = (
            []
            if query.product_id is None
            else gallery_by_product.get((query.source_dataset, query.product_id), [])
        )
        if ref.intent == "exact_match":
            eligible = [
                asset for asset in product_gallery if asset.asset_id not in forbidden
            ]
            if not eligible:
                missing_positive.append(ref.query_id)
        elif ref.intent == "divergent_rec":
            if query.product_id is None:
                missing_divergent_identity.append(ref.query_id)
            else:
                if product_gallery:
                    divergent_exclusions[ref.query_id] = sorted(
                        asset.asset_id for asset in product_gallery
                    )
                if unknown_product_gallery:
                    divergent_unknown_exclusions[ref.query_id] = sorted(
                        unknown_product_gallery
                    )
    return QueryGalleryLeakageReport(
        query_count=len(query_refs),
        gallery_count=len(gallery),
        exact_or_near_duplicate_violations=duplicate_violations,
        missing_exact_match_positive=sorted(missing_positive),
        missing_divergent_product_identity=sorted(missing_divergent_identity),
        divergent_product_exclusions=divergent_exclusions,
        divergent_unknown_product_exclusions=divergent_unknown_exclusions,
    )


def assert_query_gallery_eligible(report: QueryGalleryLeakageReport) -> None:
    if report.violation_count:
        raise CatalogError(
            "query/gallery eligibility 失败: "
            f"exact_or_near={len(report.exact_or_near_duplicate_violations)} "
            f"missing_exact_positive={len(report.missing_exact_match_positive)} "
            "missing_divergent_product_identity="
            f"{len(report.missing_divergent_product_identity)} "
            f"divergent_same_product={len(report.divergent_product_exclusions)} "
            "divergent_unknown_product="
            f"{len(report.divergent_unknown_product_exclusions)}"
        )


def _build_catalog_records(
    raw_assets: tuple[DatasetAsset, ...],
    policy: NearDuplicatePolicy,
) -> tuple[tuple[DatasetAsset, ...], tuple[LeakageComponent, ...], dict[str, int]]:
    assets = tuple(
        sorted(
            (
                asset.model_copy(
                    update={
                        "derivation_parent_asset_ids": sorted(
                            asset.derivation_parent_asset_ids
                        )
                    }
                )
                for asset in raw_assets
            ),
            key=lambda asset: asset.asset_id,
        )
    )
    _validate_asset_uniqueness(assets)
    _validate_asset_identities(assets)
    if not assets:
        raise IncompleteLeakageEvidenceError("asset catalog 不得为空")
    if any(asset.phash is None for asset in assets):
        raise IncompleteLeakageEvidenceError(
            "formal asset catalog 要求每个 asset 都有 phash"
        )

    near_union = _UnionFind(asset.asset_id for asset in assets)
    by_phash: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        assert asset.phash is not None
        by_phash[asset.phash].append(asset.asset_id)
    phash_exact_relations = 0
    for asset_ids in by_phash.values():
        ordered = sorted(asset_ids)
        for asset_id in ordered[1:]:
            near_union.union(ordered[0], asset_id)
            phash_exact_relations += 1
    phash_near_relations = 0
    for left_hash, right_hash in _near_phash_pairs(
        tuple(sorted(by_phash)), policy.max_phash_hamming_distance
    ):
        near_union.union(by_phash[left_hash][0], by_phash[right_hash][0])
        phash_near_relations += 1

    near_members: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        near_members[near_union.find(asset.asset_id)].append(asset.asset_id)
    near_id_by_asset: dict[str, str] = {}
    policy_payload = policy.model_dump(mode="json")
    for member_ids in near_members.values():
        ordered = tuple(sorted(member_ids))
        digest = hashlib.sha256(
            _canonical_json_bytes({"policy": policy_payload, "asset_ids": ordered})
        ).hexdigest()
        cluster_id = f"ndc.v1.{digest}"
        near_id_by_asset.update({asset_id: cluster_id for asset_id in ordered})

    normalized_assets: list[DatasetAsset] = []
    for asset in assets:
        expected = near_id_by_asset[asset.asset_id]
        if (
            asset.near_duplicate_cluster_id is not None
            and asset.near_duplicate_cluster_id != expected
        ):
            raise CatalogIntegrityError(
                f"stale near_duplicate_cluster_id: {asset.asset_id}"
            )
        normalized_assets.append(
            asset.model_copy(update={"near_duplicate_cluster_id": expected})
        )
    assets = tuple(normalized_assets)

    leakage_union = _UnionFind(asset.asset_id for asset in assets)
    keys_by_asset: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    relation_counts: Counter[str] = Counter(
        {
            "phash_exact": phash_exact_relations,
            "phash_near": phash_near_relations,
        }
    )

    def connect_typed(kind: LeakageKeyKind, namespace_value_pairs) -> None:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for asset, namespace, value in namespace_value_pairs:
            if value is None:
                continue
            key = (namespace, value)
            grouped[key].append(asset.asset_id)
            keys_by_asset[asset.asset_id].add((kind, namespace, value))
        for (namespace, value), asset_ids in sorted(grouped.items()):
            ordered = sorted(asset_ids)
            for asset_id in ordered[1:]:
                leakage_union.union(ordered[0], asset_id)
                relation_counts[kind] += 1

    connect_typed(
        "content_sha256",
        ((asset, "sha256-v1", asset.sha256) for asset in assets),
    )
    connect_typed(
        "source_record",
        ((asset, asset.source_dataset, asset.source_record_id) for asset in assets),
    )
    connect_typed(
        "product",
        ((asset, asset.source_dataset, asset.product_id) for asset in assets),
    )
    connect_typed(
        "near_duplicate_cluster",
        (
            (
                asset,
                policy.policy_version,
                asset.near_duplicate_cluster_id,
            )
            for asset in assets
        ),
    )

    by_id = {asset.asset_id: asset for asset in assets}
    for asset in assets:
        for parent_id in sorted(asset.derivation_parent_asset_ids):
            if parent_id not in by_id:
                raise IncompleteLeakageEvidenceError(
                    f"derivation parent 未登记: {asset.asset_id} -> {parent_id}"
                )
            edge_id = hashlib.sha256(
                "\n".join(sorted((asset.asset_id, parent_id))).encode("utf-8")
            ).hexdigest()
            key = ("derivation_parent", "asset-lineage-v1", edge_id)
            keys_by_asset[asset.asset_id].add(key)
            keys_by_asset[parent_id].add(key)
            leakage_union.union(asset.asset_id, parent_id)
            relation_counts["derivation_parent"] += 1

    component_members: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        component_members[leakage_union.find(asset.asset_id)].append(asset.asset_id)
    components: list[LeakageComponent] = []
    for member_ids in component_members.values():
        ordered_ids = tuple(sorted(member_ids))
        component_id = (
            "leakage.assetcc.v1."
            + hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "policy": LEAKAGE_POLICY_VERSION,
                        "near_duplicate_policy": policy_payload,
                        "asset_ids": ordered_ids,
                    }
                )
            ).hexdigest()
        )
        raw_keys = set().union(*(keys_by_asset[asset_id] for asset_id in ordered_ids))
        keys = tuple(
            LeakageKey(kind=kind, namespace=namespace, value=value)
            for kind, namespace, value in sorted(raw_keys)
        )
        key_counts: Counter[str] = Counter(key.kind for key in keys)
        components.append(
            LeakageComponent(
                component_id=component_id,
                asset_ids=ordered_ids,
                linkage_keys=keys,
                key_counts=dict(sorted(key_counts.items())),
            )
        )
    return (
        assets,
        tuple(sorted(components, key=lambda item: item.component_id)),
        dict(sorted(relation_counts.items())),
    )


def _build_manifest(
    assets: tuple[DatasetAsset, ...],
    components: tuple[LeakageComponent, ...],
    assets_bytes: bytes,
    components_bytes: bytes,
    relation_counts: dict[str, int],
    policy: NearDuplicatePolicy,
    coverage_roots: tuple[str, ...],
) -> AssetCatalogManifest:
    source_counts = Counter(
        (asset.source_dataset, asset.source_revision) for asset in assets
    )
    sources = tuple(
        SourceSnapshot(
            source_dataset=dataset,
            source_revision=revision,
            asset_count=count,
        )
        for (dataset, revision), count in sorted(source_counts.items())
    )
    histogram = dict(
        sorted(Counter(len(item.asset_ids) for item in components).items())
    )
    unsigned = {
        "schema_version": 1,
        "dataset_asset_schema_version": 2,
        "catalog_policy_version": CATALOG_POLICY_VERSION,
        "leakage_policy_version": LEAKAGE_POLICY_VERSION,
        "near_duplicate_policy": policy.model_dump(mode="json"),
        "coverage_roots": coverage_roots,
        "sources": [source.model_dump(mode="json") for source in sources],
        "assets": _descriptor(_ASSET_FILE, assets_bytes, len(assets)),
        "components": _descriptor(_COMPONENT_FILE, components_bytes, len(components)),
        "asset_count": len(assets),
        "component_count": len(components),
        "near_duplicate_cluster_count": len(
            {asset.near_duplicate_cluster_id for asset in assets}
        ),
        "relation_counts": relation_counts,
        "component_size_histogram": histogram,
        "unknown_cloud_permission_count": sum(
            asset.cloud_upload_allowed is None for asset in assets
        ),
        "public_demo_allowed_count": sum(asset.public_demo_allowed for asset in assets),
    }
    # Normalize through the persisted model before hashing.  In particular,
    # JSON turns the integer keys of component_size_histogram into strings; a
    # direct hash of ``unsigned`` can therefore order 10 numerically here but
    # lexicographically after reload and report a false self-hash mismatch.
    provisional = AssetCatalogManifest(
        **unsigned,
        catalog_sha256="0" * 64,
    )
    normalized_unsigned = provisional.model_dump(
        mode="json",
        exclude={"catalog_sha256"},
    )
    return AssetCatalogManifest(
        **normalized_unsigned,
        catalog_sha256=hashlib.sha256(
            _canonical_json_bytes(normalized_unsigned)
        ).hexdigest(),
    )


def _verify_manifest_self_hash(manifest: AssetCatalogManifest) -> None:
    unsigned = manifest.model_dump(mode="json", exclude={"catalog_sha256"})
    actual = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if actual != manifest.catalog_sha256:
        raise CatalogIntegrityError("asset catalog manifest self hash 不一致")


def _descriptor(path: str, content: bytes, count: int) -> dict[str, object]:
    return {
        "path": path,
        "count": count,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _verify_descriptor(
    descriptor: CatalogFile, content: bytes, count: int, expected_path: str
) -> None:
    if (
        descriptor.path != expected_path
        or descriptor.count != count
        or descriptor.bytes != len(content)
        or descriptor.sha256 != hashlib.sha256(content).hexdigest()
    ):
        raise CatalogIntegrityError(
            f"catalog artifact descriptor 不一致: {expected_path}"
        )


def _validate_asset_uniqueness(assets: tuple[DatasetAsset, ...]) -> None:
    ids = [asset.asset_id for asset in assets]
    if len(ids) != len(set(ids)):
        raise DuplicateAssetIdError("asset_id 重复")
    canonical_paths: dict[str, str] = {}
    for asset in assets:
        normalized = _canonical_local_path(asset.local_path)
        if asset.local_path != normalized:
            raise CatalogIntegrityError(
                f"local_path 不是 canonical NFC/POSIX 路径: {asset.local_path}"
            )
        canonical = _canonical_path_key(asset.local_path)
        previous = canonical_paths.get(canonical)
        if previous is not None:
            raise DuplicateLocalPathError(
                f"local_path Unicode/casefold 冲突: {previous} / {asset.local_path}"
            )
        canonical_paths[canonical] = asset.local_path


def _validate_asset_identities(assets: tuple[DatasetAsset, ...]) -> None:
    for asset in assets:
        expected = _expected_asset_id(
            source_dataset=asset.source_dataset,
            source_record_id=asset.source_record_id,
            transform_policy_version=asset.transform_policy_version,
            sha256=asset.sha256,
        )
        if asset.asset_id != expected:
            raise CatalogIntegrityError(
                f"asset_id 与 SHA/来源可重算身份不一致: {asset.asset_id}"
            )


def _query_gallery_forbidden_groups(
    assets: Iterable[DatasetAsset],
) -> dict[str, str]:
    """Close every non-product relation forbidden across query/gallery.

    Product identity is intentionally excluded: Exact Match needs a distinct
    view of the same product in the gallery, while Divergent Recommendation
    handles same-product candidates through a separate exclusion rule.
    """

    assets = tuple(assets)
    union = _UnionFind(asset.asset_id for asset in assets)
    known = {asset.asset_id for asset in assets}

    def connect(values: Iterable[tuple[object, str]]) -> None:
        grouped: dict[object, list[str]] = defaultdict(list)
        for key, asset_id in values:
            if key is not None:
                grouped[key].append(asset_id)
        for asset_ids in grouped.values():
            ordered = sorted(asset_ids)
            for asset_id in ordered[1:]:
                union.union(ordered[0], asset_id)

    connect((asset.sha256, asset.asset_id) for asset in assets)
    connect(
        (
            (asset.source_dataset, asset.source_record_id),
            asset.asset_id,
        )
        for asset in assets
    )
    connect((asset.near_duplicate_cluster_id, asset.asset_id) for asset in assets)
    for asset in assets:
        for parent_id in asset.derivation_parent_asset_ids:
            if parent_id not in known:
                raise IncompleteLeakageEvidenceError(
                    f"derivation parent 未登记: {asset.asset_id} -> {parent_id}"
                )
            union.union(asset.asset_id, parent_id)
    return {asset.asset_id: union.find(asset.asset_id) for asset in assets}


def _validate_asset_files(assets: Iterable[DatasetAsset], asset_root: Path) -> None:
    for asset in assets:
        path = _resolve_asset_file(asset_root, asset.local_path)
        sha256, phash = _fingerprint_file(path)
        if sha256 != asset.sha256:
            raise CatalogIntegrityError(
                f"asset final-byte SHA 不一致: {asset.asset_id}"
            )
        if phash != asset.phash:
            raise CatalogIntegrityError(f"asset final pHash 不一致: {asset.asset_id}")


def _validate_coverage(
    assets: Iterable[DatasetAsset], asset_root: Path, coverage_roots: Iterable[str]
) -> None:
    assets = tuple(assets)
    coverage_roots = tuple(coverage_roots)
    if not coverage_roots:
        return

    registered_paths = {asset.local_path for asset in assets}
    uncovered = sorted(
        local_path
        for local_path in registered_paths
        if not any(
            root == "." or local_path == root or local_path.startswith(f"{root}/")
            for root in coverage_roots
        )
    )
    if uncovered:
        raise IncompleteLeakageEvidenceError(
            f"已登记 asset 不在任何 coverage root 内: {uncovered[:5]}"
        )

    discovered_by_key: dict[str, str] = {}
    discovered_paths: set[str] = set()
    unsupported_images: list[str] = []
    for relative_root in coverage_roots:
        directory = (
            asset_root
            if relative_root == "."
            else _resolve_under_root(asset_root, relative_root, require_file=False)
        )
        if not directory.is_dir():
            raise CatalogIntegrityError(f"coverage root 不存在: {relative_root}")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            raw_local_path = path.relative_to(asset_root).as_posix()
            local_path = _canonical_local_path(raw_local_path)
            if raw_local_path != local_path:
                raise IncompleteLeakageEvidenceError(
                    f"coverage root 含非 canonical Unicode/POSIX 路径: {raw_local_path}"
                )
            _resolve_asset_file(asset_root, raw_local_path)
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                if _is_decodable_image(path):
                    unsupported_images.append(local_path)
                continue
            key = _canonical_path_key(local_path)
            previous = discovered_by_key.get(key)
            if previous is not None and previous != raw_local_path:
                raise IncompleteLeakageEvidenceError(
                    "coverage root 含 Unicode/casefold 冲突图片: "
                    f"{previous} / {raw_local_path}"
                )
            discovered_by_key[key] = raw_local_path
            discovered_paths.add(local_path)

    if unsupported_images:
        raise IncompleteLeakageEvidenceError(
            "coverage root 含策略未支持但可解码的图片格式: "
            f"{sorted(set(unsupported_images))[:5]}"
        )
    missing = sorted(discovered_paths - registered_paths)
    if missing:
        raise IncompleteLeakageEvidenceError(
            f"coverage root 存在未登记图片: {missing[:5]}"
        )


def _is_decodable_image(path: Path) -> bool:
    try:
        with Image.open(path) as opened:
            opened.verify()
        return True
    except (OSError, ValueError):
        return False


def _near_phash_pairs(
    hashes: tuple[str, ...], threshold: int
) -> tuple[tuple[str, str], ...]:
    if len(hashes) < 2:
        return ()
    band_count = threshold + 1
    base_width, remainder = divmod(64, band_count)
    bands: list[tuple[int, int]] = []
    offset = 0
    for index in range(band_count):
        width = base_width + (1 if index < remainder else 0)
        bands.append((offset, width))
        offset += width
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for value in hashes:
        number = int(value, 16)
        for index, (start, width) in enumerate(bands):
            mask = (1 << width) - 1
            buckets[(index, (number >> start) & mask)].append(value)
    candidates: set[tuple[str, str]] = set()
    for values in buckets.values():
        ordered = sorted(values)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                candidates.add((left, right))
    return tuple(
        pair
        for pair in sorted(candidates)
        if _phash_distance(pair[0], pair[1]) <= threshold
    )


def _phash_distance(left: str | None, right: str | None) -> int:
    if left is None or right is None:
        return 65
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _canonical_local_path(path: str | Path) -> str:
    value = unicodedata.normalize("NFC", str(path).replace("\\", "/")).strip()
    candidate = Path(value)
    if not value or candidate.is_absolute() or value.startswith("/"):
        raise ValueError("local_path 必须是非空相对路径")
    if ".." in candidate.parts:
        raise ValueError("local_path 不得包含 ..")
    normalized = candidate.as_posix()
    if normalized in {".", ""}:
        raise ValueError("local_path 不得为空")
    return normalized


def _canonical_coverage_root(path: str | Path) -> str:
    value = unicodedata.normalize("NFC", str(path).replace("\\", "/")).strip()
    if value in {"", "."}:
        if value == ".":
            return value
        raise ValueError("coverage root 不得为空")
    return _canonical_local_path(value).rstrip("/")


def _canonical_path_key(path: str | Path) -> str:
    return _canonical_local_path(path).casefold()


def _resolve_asset_file(asset_root: Path, local_path: str | Path) -> Path:
    path = _resolve_under_root(asset_root, local_path, require_file=True)
    if not path.is_file():
        raise CatalogIntegrityError(f"asset 文件不存在: {local_path}")
    return path


def _resolve_under_root(
    asset_root: Path, local_path: str | Path, *, require_file: bool
) -> Path:
    normalized = _canonical_local_path(local_path)
    candidate = asset_root / normalized
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        kind = "文件" if require_file else "路径"
        raise CatalogIntegrityError(f"asset {kind}不存在: {normalized}") from None
    try:
        resolved.relative_to(asset_root)
    except ValueError:
        raise CatalogIntegrityError(
            f"asset path 通过 symlink 越界: {normalized}"
        ) from None
    return resolved


def _expected_asset_id(
    *,
    source_dataset: str,
    source_record_id: str,
    transform_policy_version: str,
    sha256: str,
) -> str:
    identity_payload = {
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "transform_policy_version": transform_policy_version,
        "sha256": sha256,
    }
    return (
        "asset.v2."
        + hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()
    )


def _fingerprint_file(path: Path) -> tuple[str, str]:
    """Compute byte SHA and decoded pHash from one immutable byte snapshot."""

    try:
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as opened:
            opened.load()
            # Avoid two full-size copies for the common RGB/no-EXIF case.  A
            # complete Core verification decodes every asset, and Pillow's
            # native allocator otherwise retains several gigabytes per scan.
            ImageOps.exif_transpose(opened, in_place=True)
            if opened.mode == "RGB":
                phash = str(imagehash.phash(opened, hash_size=8))
            else:
                with opened.convert("RGB") as normalized:
                    phash = str(imagehash.phash(normalized, hash_size=8))
        return hashlib.sha256(content).hexdigest(), phash
    except (MemoryError, OSError, ValueError) as exc:
        raise CatalogIntegrityError(f"asset 无法解码计算 pHash: {path}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(values: Iterable[BaseModel]) -> bytes:
    return b"".join(
        _canonical_json_bytes(value.model_dump(mode="json")) for value in values
    )


def _parse_jsonl(content: bytes, model_type, label: str) -> list:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CatalogIntegrityError(f"{label} 不是 UTF-8") from exc
    values = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CatalogIntegrityError(f"{label} 第 {line_number} 行为空")
        values.append(model_type.model_validate_json(line))
    return values


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as target:
        target.write(content)


__all__ = [
    "ASSET_GROUPING_POLICY_VERSION",
    "AssetCatalog",
    "AssetCatalogManifest",
    "AssetNotCatalogedError",
    "AssetReferenceMismatchError",
    "AssetResolution",
    "CATALOG_POLICY_VERSION",
    "CatalogError",
    "CatalogIntegrityError",
    "DatasetAssetDraft",
    "DEFAULT_NEAR_DUPLICATE_POLICY",
    "DuplicateAssetIdError",
    "DuplicateLocalPathError",
    "GalleryAssetReference",
    "IncompleteLeakageEvidenceError",
    "LEAKAGE_POLICY_VERSION",
    "LeakageComponent",
    "LeakageKey",
    "NearDuplicatePolicy",
    "QueryAssetReference",
    "QueryGalleryLeakageReport",
    "assert_query_gallery_eligible",
    "audit_query_gallery_eligibility",
    "inventory_dataset_asset",
    "load_asset_catalog",
    "publish_asset_catalog",
]

# Backward-readable semantic alias used by planning manifests.
ASSET_GROUPING_POLICY_VERSION = LEAKAGE_POLICY_VERSION
