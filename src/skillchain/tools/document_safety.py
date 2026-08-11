"""Create-only document-safety approvals bound to a verified asset catalog.

The catalog is the formal bridge between an external privacy review and the
``document_ocr`` tool.  It authorizes a reviewed asset only for an explicit
query/asset pair, binds the approval to the final image bytes, and requires a
redacted image to name one of its direct catalog lineage parents.

The canonical ``approvals.jsonl`` bytes are also the review ledger.  Their
SHA-256 must be pinned outside this bundle and supplied to every publish/load;
deriving that expected value from an untrusted manifest would remove the trust
boundary and permit coordinated re-hashing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import stat
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.document_ocr import (
    DocumentSafetyApproval,
    document_safety_approval_digest,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

DOCUMENT_SAFETY_CATALOG_SCHEMA_VERSION = 1
DOCUMENT_SAFETY_CATALOG_POLICY_VERSION = "document-safety-catalog-v1"

_MANIFEST_FILE = "manifest.json"
_APPROVALS_FILE = "approvals.jsonl"
_ASSET_CATALOG_FILES = ("assets.jsonl", "components.jsonl", "manifest.json")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_APPROVALS_BYTES = 64 * 1024 * 1024

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DocumentSafetyCatalogError(ValueError):
    """The safety catalog or its authoritative asset binding is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DocumentSafetyRecord(_StrictFrozenModel):
    """One query-scoped authorization for an already self-hashed approval."""

    schema_version: Literal[1] = DOCUMENT_SAFETY_CATALOG_SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    approval: DocumentSafetyApproval

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("query_id must be a non-blank canonical string")
        if any(ord(character) < 32 for character in value):
            raise ValueError("query_id must not contain control characters")
        return value

    @property
    def asset_id(self) -> str:
        return self.approval.asset_id

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.query_id, self.asset_id)


class ApprovalArtifactDescriptor(_StrictFrozenModel):
    path: Literal["approvals.jsonl"] = _APPROVALS_FILE
    count: int = Field(gt=0)
    bytes: int = Field(gt=0)
    sha256: Sha256


class AssetCatalogArtifactBinding(_StrictFrozenModel):
    path: Literal["assets.jsonl", "components.jsonl", "manifest.json"]
    bytes: int = Field(gt=0)
    sha256: Sha256


class DocumentSafetyCatalogManifest(_StrictFrozenModel):
    schema_version: Literal[1] = DOCUMENT_SAFETY_CATALOG_SCHEMA_VERSION
    policy_version: Literal["document-safety-catalog-v1"] = (
        DOCUMENT_SAFETY_CATALOG_POLICY_VERSION
    )
    asset_catalog_sha256: Sha256
    asset_catalog_policy_version: str = Field(min_length=1)
    leakage_policy_version: str = Field(min_length=1)
    asset_catalog_artifacts: tuple[AssetCatalogArtifactBinding, ...] = Field(
        min_length=3,
        max_length=3,
    )
    asset_catalog_bundle_sha256: Sha256
    review_ledger_sha256: Sha256
    approvals: ApprovalArtifactDescriptor
    approval_count: int = Field(gt=0)
    query_count: int = Field(gt=0)
    approved_no_pii_count: int = Field(ge=0)
    approved_redacted_count: int = Field(ge=0)
    approval_policy_versions: tuple[str, ...] = Field(min_length=1)
    catalog_sha256: Sha256

    @field_validator("asset_catalog_policy_version", "leakage_policy_version")
    @classmethod
    def validate_catalog_versions(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError(
                "catalog policy versions must be canonical non-blank strings"
            )
        return value

    @field_validator("approval_policy_versions")
    @classmethod
    def validate_approval_policy_versions(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("approval policy versions must be canonical strings")
        if value != tuple(sorted(set(value))):
            raise ValueError("approval policy versions must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        artifact_paths = tuple(item.path for item in self.asset_catalog_artifacts)
        if artifact_paths != _ASSET_CATALOG_FILES:
            raise ValueError(
                "asset catalog artifact bindings must be unique and sorted"
            )
        if self.approvals.count != self.approval_count:
            raise ValueError("approval descriptor count mismatch")
        if self.review_ledger_sha256 != self.approvals.sha256:
            raise ValueError("review ledger hash must bind the exact approval bytes")
        if self.query_count > self.approval_count:
            raise ValueError("query_count cannot exceed approval_count")
        if (
            self.approved_no_pii_count + self.approved_redacted_count
            != self.approval_count
        ):
            raise ValueError("approval decision counts do not sum to approval_count")
        if self.asset_catalog_bundle_sha256 != _asset_catalog_bundle_digest(
            self.asset_catalog_artifacts
        ):
            raise ValueError("asset catalog artifact bundle hash mismatch")
        if self.catalog_sha256 != document_safety_catalog_digest(self):
            raise ValueError("document safety catalog self hash mismatch")
        return self


@dataclass(frozen=True)
class _AssetCatalogSnapshot:
    contents: tuple[tuple[str, bytes], ...]
    artifacts: tuple[AssetCatalogArtifactBinding, ...]
    verified_catalog: AssetCatalog = field(compare=False, repr=False)


@dataclass(frozen=True)
class _SafetyCatalogSnapshot:
    manifest_bytes: bytes
    approvals_bytes: bytes


@dataclass(frozen=True)
class DocumentSafetyCatalog:
    """Verified immutable approvals with a fail-closed registry resolver."""

    root: Path
    manifest: DocumentSafetyCatalogManifest
    approvals: tuple[DocumentSafetyRecord, ...]
    manifest_bytes: bytes
    approvals_bytes: bytes
    _by_query_asset: Mapping[tuple[str, str], DocumentSafetyApproval]

    @property
    def catalog_sha256(self) -> str:
        return self.manifest.catalog_sha256

    @property
    def formal_runtime_binding_sha256(self) -> str:
        """Bind the live OCR approval resolver to its reviewed catalog bytes."""

        return sha256_bytes(
            canonical_json_bytes(
                {
                    "asset_catalog_sha256": self.manifest.asset_catalog_sha256,
                    "catalog_sha256": self.catalog_sha256,
                    "policy_version": "document-safety-resolver-runtime-v1",
                    "review_ledger_sha256": self.manifest.review_ledger_sha256,
                }
            )
        )

    def approval_for(
        self, query_id: str, asset_id: str
    ) -> DocumentSafetyApproval | None:
        """Return only an approval explicitly authorized for this exact pair."""

        if not isinstance(query_id, str) or not isinstance(asset_id, str):
            return None
        return self._by_query_asset.get((query_id, asset_id))


def build_document_safety_record(
    query_id: str,
    approval: DocumentSafetyApproval,
) -> DocumentSafetyRecord:
    """Construct the strict query-scoped record used by the publisher."""

    return DocumentSafetyRecord(query_id=query_id, approval=approval)


def publish_document_safety_catalog(
    approvals: Iterable[DocumentSafetyRecord],
    output_dir: str | Path,
    asset_catalog: AssetCatalog,
    *,
    expected_review_ledger_sha256: str,
) -> DocumentSafetyCatalog:
    """Publish against an externally pinned review-ledger digest, create-only."""

    output_dir = Path(output_dir)
    if _lexists(output_dir):
        raise FileExistsError(f"document safety catalog already exists: {output_dir}")
    expected_review_ledger_sha256 = _validate_expected_review_ledger_sha256(
        expected_review_ledger_sha256
    )

    before = _snapshot_verified_asset_catalog(asset_catalog)
    records = _normalize_records(tuple(approvals), require_sorted=False)
    _verify_records_against_asset_catalog(records, before.verified_catalog)
    approvals_bytes = canonical_jsonl_bytes(
        tuple(record.model_dump(mode="json") for record in records)
    )
    if sha256_bytes(approvals_bytes) != expected_review_ledger_sha256:
        raise DocumentSafetyCatalogError(
            "generated approvals do not match the pinned external review ledger"
        )
    manifest = _build_manifest(
        records,
        approvals_bytes,
        before,
        before.verified_catalog,
        expected_review_ledger_sha256,
    )
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))

    staging = _create_staging_directory(output_dir)
    try:
        _write_staging_bundle(staging, approvals_bytes, manifest_bytes)
        _load_safety_bundle(
            staging,
            before.verified_catalog,
            expected_asset_snapshot=before,
            expected_review_ledger_sha256=expected_review_ledger_sha256,
        )

        after = _snapshot_verified_asset_catalog(asset_catalog)
        if after != before:
            raise DocumentSafetyCatalogError(
                "asset catalog bytes changed while building the safety catalog"
            )
        _verify_records_against_asset_catalog(records, after.verified_catalog)
        _publish_create_only(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return load_document_safety_catalog(
        output_dir,
        asset_catalog,
        expected_review_ledger_sha256=expected_review_ledger_sha256,
    )


def build_document_safety_catalog(
    approvals: Iterable[DocumentSafetyRecord],
    output_dir: str | Path,
    asset_catalog: AssetCatalog,
    *,
    expected_review_ledger_sha256: str,
) -> DocumentSafetyCatalog:
    """Alias with build-style naming used by other tool-layer artifacts."""

    return publish_document_safety_catalog(
        approvals,
        output_dir,
        asset_catalog,
        expected_review_ledger_sha256=expected_review_ledger_sha256,
    )


def load_document_safety_catalog(
    root: str | Path,
    asset_catalog: AssetCatalog,
    *,
    expected_review_ledger_sha256: str,
) -> DocumentSafetyCatalog:
    """Load canonical approvals using a digest sourced outside the bundle."""

    expected_review_ledger_sha256 = _validate_expected_review_ledger_sha256(
        expected_review_ledger_sha256
    )
    before_catalog = _snapshot_verified_asset_catalog(asset_catalog)
    loaded = _load_safety_bundle(
        Path(root),
        before_catalog.verified_catalog,
        expected_asset_snapshot=before_catalog,
        expected_review_ledger_sha256=expected_review_ledger_sha256,
    )
    after_catalog = _snapshot_verified_asset_catalog(asset_catalog)
    if after_catalog != before_catalog:
        raise DocumentSafetyCatalogError(
            "asset catalog bytes changed while loading the safety catalog"
        )

    after_bundle = _read_safety_bundle(Path(root))
    if (
        after_bundle.manifest_bytes != loaded.manifest_bytes
        or after_bundle.approvals_bytes != loaded.approvals_bytes
    ):
        raise DocumentSafetyCatalogError(
            "document safety catalog changed while it was being loaded"
        )
    _verify_records_against_asset_catalog(
        loaded.approvals,
        after_catalog.verified_catalog,
    )
    return loaded


def document_safety_catalog_digest(
    manifest: DocumentSafetyCatalogManifest | dict[str, Any],
) -> str:
    """Hash every manifest field except its own digest."""

    if isinstance(manifest, BaseModel):
        unsigned = manifest.model_dump(mode="json", exclude={"catalog_sha256"})
    else:
        unsigned = dict(manifest)
        unsigned.pop("catalog_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def document_safety_review_ledger_digest(
    approvals: Iterable[DocumentSafetyRecord],
) -> str:
    """Return the digest that a trusted review step must pin externally."""

    records = _normalize_records(tuple(approvals), require_sorted=False)
    content = canonical_jsonl_bytes(
        tuple(record.model_dump(mode="json") for record in records)
    )
    return sha256_bytes(content)


def _normalize_records(
    values: tuple[DocumentSafetyRecord, ...],
    *,
    require_sorted: bool,
) -> tuple[DocumentSafetyRecord, ...]:
    if not values:
        raise DocumentSafetyCatalogError(
            "at least one document safety approval is required"
        )
    if any(not isinstance(value, DocumentSafetyRecord) for value in values):
        raise DocumentSafetyCatalogError(
            "approvals must contain strict DocumentSafetyRecord instances"
        )
    validated = tuple(
        _model_from_json_value(
            DocumentSafetyRecord,
            value.model_dump(mode="json"),
            f"document safety record {ordinal}",
        )
        for ordinal, value in enumerate(values, start=1)
    )
    keys = tuple(value.sort_key for value in validated)
    if len(keys) != len(set(keys)):
        raise DocumentSafetyCatalogError(
            "document safety query/asset pairs must be unique"
        )
    ordered = tuple(sorted(validated, key=lambda value: value.sort_key))
    if require_sorted and tuple(value.sort_key for value in ordered) != keys:
        raise DocumentSafetyCatalogError(
            "document safety approvals must use canonical query/asset order"
        )
    return ordered


def _verify_records_against_asset_catalog(
    records: tuple[DocumentSafetyRecord, ...],
    asset_catalog: AssetCatalog,
) -> None:
    referenced_ids: set[str] = set()
    for record in records:
        if record.approval.approval_sha256 != document_safety_approval_digest(
            record.approval
        ):
            raise DocumentSafetyCatalogError(
                "document safety approval self hash mismatch"
            )
        referenced_ids.add(record.asset_id)
        try:
            approved_asset = asset_catalog.resolve_asset_id(record.asset_id).asset
        except Exception as error:
            raise DocumentSafetyCatalogError(
                f"approval references an uncataloged asset: {record.asset_id}"
            ) from error
        if approved_asset.sha256 != record.approval.image_sha256:
            raise DocumentSafetyCatalogError(
                f"approval image hash does not match asset catalog: {record.asset_id}"
            )
        _require_regular_catalog_asset(asset_catalog, approved_asset.local_path)

        if record.approval.decision != "approved_redacted":
            continue
        parent_hash = record.approval.redaction_parent_sha256
        direct_parent_hashes: set[str] = set()
        for parent_id in approved_asset.derivation_parent_asset_ids:
            try:
                parent = asset_catalog.resolve_asset_id(parent_id).asset
            except Exception as error:
                raise DocumentSafetyCatalogError(
                    f"redacted asset has an uncataloged direct parent: {parent_id}"
                ) from error
            referenced_ids.add(parent_id)
            direct_parent_hashes.add(parent.sha256)
            _require_regular_catalog_asset(asset_catalog, parent.local_path)
        if parent_hash not in direct_parent_hashes:
            raise DocumentSafetyCatalogError(
                "redacted approval parent hash must match a direct catalog lineage parent"
            )

    try:
        asset_catalog.verify_asset_ids(referenced_ids)
    except Exception as error:
        raise DocumentSafetyCatalogError(
            "approved image or redaction-parent bytes no longer match the asset catalog"
        ) from error


def _build_manifest(
    records: tuple[DocumentSafetyRecord, ...],
    approvals_bytes: bytes,
    asset_snapshot: _AssetCatalogSnapshot,
    asset_catalog: AssetCatalog,
    review_ledger_sha256: str,
) -> DocumentSafetyCatalogManifest:
    decisions = [record.approval.decision for record in records]
    unsigned: dict[str, Any] = {
        "schema_version": DOCUMENT_SAFETY_CATALOG_SCHEMA_VERSION,
        "policy_version": DOCUMENT_SAFETY_CATALOG_POLICY_VERSION,
        "asset_catalog_sha256": asset_catalog.catalog_sha256,
        "asset_catalog_policy_version": asset_catalog.manifest.catalog_policy_version,
        "leakage_policy_version": asset_catalog.leakage_policy_version,
        "asset_catalog_artifacts": [
            item.model_dump(mode="json") for item in asset_snapshot.artifacts
        ],
        "asset_catalog_bundle_sha256": _asset_catalog_bundle_digest(
            asset_snapshot.artifacts
        ),
        "review_ledger_sha256": review_ledger_sha256,
        "approvals": {
            "path": _APPROVALS_FILE,
            "count": len(records),
            "bytes": len(approvals_bytes),
            "sha256": sha256_bytes(approvals_bytes),
        },
        "approval_count": len(records),
        "query_count": len({record.query_id for record in records}),
        "approved_no_pii_count": decisions.count("approved_no_pii"),
        "approved_redacted_count": decisions.count("approved_redacted"),
        "approval_policy_versions": sorted(
            {record.approval.policy_version for record in records}
        ),
    }
    payload = {
        **unsigned,
        "catalog_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    return _model_from_json_value(
        DocumentSafetyCatalogManifest,
        payload,
        "document safety manifest",
    )


def _load_safety_bundle(
    root: Path,
    asset_catalog: AssetCatalog,
    *,
    expected_asset_snapshot: _AssetCatalogSnapshot,
    expected_review_ledger_sha256: str,
) -> DocumentSafetyCatalog:
    snapshot = _read_safety_bundle(root)
    try:
        manifest_value = parse_canonical_json(
            snapshot.manifest_bytes,
            label=_MANIFEST_FILE,
        )
        if not isinstance(manifest_value, dict):
            raise DocumentSafetyCatalogError("manifest.json must contain one object")
        manifest = _model_from_json_value(
            DocumentSafetyCatalogManifest,
            manifest_value,
            _MANIFEST_FILE,
        )
        if snapshot.manifest_bytes != canonical_json_bytes(
            manifest.model_dump(mode="json")
        ):
            raise DocumentSafetyCatalogError("manifest.json must be canonical JSON")
        if manifest.review_ledger_sha256 != expected_review_ledger_sha256:
            raise DocumentSafetyCatalogError(
                "document safety catalog does not match the pinned review ledger"
            )

        row_values = parse_canonical_jsonl(
            snapshot.approvals_bytes,
            label=_APPROVALS_FILE,
        )
        records = tuple(
            _model_from_json_value(
                DocumentSafetyRecord,
                value,
                f"{_APPROVALS_FILE} line {line_number}",
            )
            for line_number, value in enumerate(row_values, start=1)
        )
    except ArtifactFormatError as error:
        raise DocumentSafetyCatalogError(str(error)) from error
    except (ValidationError, DocumentSafetyCatalogError):
        raise
    except ValueError as error:
        raise DocumentSafetyCatalogError(
            "document safety catalog schema validation failed"
        ) from error

    records = _normalize_records(records, require_sorted=True)
    canonical_records = canonical_jsonl_bytes(
        tuple(record.model_dump(mode="json") for record in records)
    )
    if snapshot.approvals_bytes != canonical_records:
        raise DocumentSafetyCatalogError("approvals.jsonl must be canonical JSONL")
    if (
        manifest.approvals.path != _APPROVALS_FILE
        or manifest.approvals.count != len(records)
        or manifest.approvals.bytes != len(snapshot.approvals_bytes)
        or manifest.approvals.sha256 != sha256_bytes(snapshot.approvals_bytes)
    ):
        raise DocumentSafetyCatalogError("approvals.jsonl descriptor mismatch")

    if tuple(manifest.asset_catalog_artifacts) != expected_asset_snapshot.artifacts:
        raise DocumentSafetyCatalogError(
            "safety manifest does not bind the current asset catalog artifact bytes"
        )
    if manifest.asset_catalog_sha256 != asset_catalog.catalog_sha256:
        raise DocumentSafetyCatalogError("asset catalog self-hash binding mismatch")
    if (
        manifest.asset_catalog_policy_version
        != asset_catalog.manifest.catalog_policy_version
        or manifest.leakage_policy_version != asset_catalog.leakage_policy_version
    ):
        raise DocumentSafetyCatalogError("asset catalog policy binding mismatch")

    _verify_records_against_asset_catalog(records, asset_catalog)
    rebuilt_manifest = _build_manifest(
        records,
        snapshot.approvals_bytes,
        expected_asset_snapshot,
        asset_catalog,
        expected_review_ledger_sha256,
    )
    if rebuilt_manifest != manifest:
        raise DocumentSafetyCatalogError(
            "document safety manifest does not match rebuilt catalog contents"
        )
    return DocumentSafetyCatalog(
        root=root,
        manifest=manifest,
        approvals=records,
        manifest_bytes=snapshot.manifest_bytes,
        approvals_bytes=snapshot.approvals_bytes,
        _by_query_asset=MappingProxyType(
            {record.sort_key: record.approval for record in records}
        ),
    )


def _read_safety_bundle(root: Path) -> _SafetyCatalogSnapshot:
    _require_real_directory(root, "document safety catalog")
    try:
        names = {item.name for item in root.iterdir()}
    except OSError as error:
        raise DocumentSafetyCatalogError(
            "document safety catalog directory cannot be enumerated"
        ) from error
    if names != {_MANIFEST_FILE, _APPROVALS_FILE}:
        raise DocumentSafetyCatalogError(
            "document safety catalog artifact set is invalid"
        )
    try:
        manifest_bytes = read_stable_regular_file(
            root / _MANIFEST_FILE,
            label=_MANIFEST_FILE,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        approvals_bytes = read_stable_regular_file(
            root / _APPROVALS_FILE,
            label=_APPROVALS_FILE,
            max_bytes=_MAX_APPROVALS_BYTES,
        )
    except ArtifactFormatError as error:
        raise DocumentSafetyCatalogError(str(error)) from error
    return _SafetyCatalogSnapshot(
        manifest_bytes=manifest_bytes,
        approvals_bytes=approvals_bytes,
    )


def _snapshot_verified_asset_catalog(
    asset_catalog: AssetCatalog,
) -> _AssetCatalogSnapshot:
    if not isinstance(asset_catalog, AssetCatalog):
        raise DocumentSafetyCatalogError("asset_catalog must be an AssetCatalog")
    try:
        asset_catalog.require_verified_files()
    except Exception as error:
        raise DocumentSafetyCatalogError(
            "document safety requires an AssetCatalog loaded with verify_files=True"
        ) from error

    root = Path(asset_catalog.root)
    _require_real_directory(root, "asset catalog")
    try:
        names = {item.name for item in root.iterdir()}
    except OSError as error:
        raise DocumentSafetyCatalogError(
            "asset catalog directory cannot be enumerated"
        ) from error
    if names != set(_ASSET_CATALOG_FILES):
        raise DocumentSafetyCatalogError("asset catalog artifact set is invalid")

    contents: list[tuple[str, bytes]] = []
    artifacts: list[AssetCatalogArtifactBinding] = []
    try:
        for name in _ASSET_CATALOG_FILES:
            content = read_stable_regular_file(
                root / name,
                label=f"asset catalog {name}",
                max_bytes=_MAX_APPROVALS_BYTES,
            )
            contents.append((name, content))
            artifacts.append(
                AssetCatalogArtifactBinding(
                    path=name,
                    bytes=len(content),
                    sha256=sha256_bytes(content),
                )
            )
    except (ArtifactFormatError, ValidationError) as error:
        raise DocumentSafetyCatalogError(
            "asset catalog artifacts must be stable regular files"
        ) from error

    try:
        current = load_asset_catalog(
            root,
            asset_catalog.asset_root,
            verify_files=True,
        )
    except Exception as error:
        raise DocumentSafetyCatalogError(
            "asset catalog no longer passes full verification"
        ) from error
    if (
        current.manifest != asset_catalog.manifest
        or current.assets != asset_catalog.assets
        or current.components != asset_catalog.components
        or current.catalog_sha256 != asset_catalog.catalog_sha256
    ):
        raise DocumentSafetyCatalogError(
            "live asset catalog differs from the verified catalog object"
        )
    return _AssetCatalogSnapshot(
        contents=tuple(contents),
        artifacts=tuple(artifacts),
        verified_catalog=current,
    )


def _asset_catalog_bundle_digest(
    artifacts: tuple[AssetCatalogArtifactBinding, ...],
) -> str:
    payload = [artifact.model_dump(mode="json") for artifact in artifacts]
    return sha256_bytes(canonical_json_bytes(payload))


def _require_regular_catalog_asset(
    asset_catalog: AssetCatalog,
    local_path: str,
) -> None:
    current = Path(asset_catalog.asset_root)
    for part in Path(local_path.replace("\\", "/")).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise DocumentSafetyCatalogError(
                f"catalog asset cannot be inspected: {local_path}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise DocumentSafetyCatalogError(
                f"catalog asset path must not traverse a symlink: {local_path}"
            )
    if not stat.S_ISREG(metadata.st_mode):
        raise DocumentSafetyCatalogError(
            f"catalog asset must be a regular file: {local_path}"
        )
    try:
        current.resolve(strict=True).relative_to(asset_catalog.asset_root)
    except (OSError, ValueError) as error:
        raise DocumentSafetyCatalogError(
            f"catalog asset escapes its authoritative root: {local_path}"
        ) from error


def _model_from_json_value(model_type, value: object, label: str):
    try:
        return model_type.model_validate_json(canonical_json_bytes(value))
    except (ValidationError, ValueError, TypeError) as error:
        raise DocumentSafetyCatalogError(f"{label} is invalid") from error


def _validate_expected_review_ledger_sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DocumentSafetyCatalogError(
            "expected_review_ledger_sha256 must be a lowercase SHA-256 digest"
        )
    return value


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DocumentSafetyCatalogError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DocumentSafetyCatalogError(f"{label} must be a non-symlink directory")


def _create_staging_directory(destination: Path) -> Path:
    if _lexists(destination):
        raise FileExistsError(f"document safety catalog already exists: {destination}")
    return new_staging_directory(destination)


def _write_staging_bundle(
    staging: Path,
    approvals_bytes: bytes,
    manifest_bytes: bytes,
) -> None:
    _exclusive_write(staging / _APPROVALS_FILE, approvals_bytes)
    _exclusive_write(staging / _MANIFEST_FILE, manifest_bytes)


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_create_only(staging: Path, destination: Path) -> None:
    if _lexists(destination):
        raise FileExistsError(f"document safety catalog already exists: {destination}")
    try:
        atomic_publish_new_directory(staging, destination)
    except FileExistsError:
        raise FileExistsError(
            f"document safety catalog already exists: {destination}"
        ) from None


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = [
    "DOCUMENT_SAFETY_CATALOG_POLICY_VERSION",
    "DOCUMENT_SAFETY_CATALOG_SCHEMA_VERSION",
    "ApprovalArtifactDescriptor",
    "AssetCatalogArtifactBinding",
    "DocumentSafetyCatalog",
    "DocumentSafetyCatalogError",
    "DocumentSafetyCatalogManifest",
    "DocumentSafetyRecord",
    "build_document_safety_catalog",
    "build_document_safety_record",
    "document_safety_catalog_digest",
    "document_safety_review_ledger_digest",
    "load_document_safety_catalog",
    "publish_document_safety_catalog",
]
