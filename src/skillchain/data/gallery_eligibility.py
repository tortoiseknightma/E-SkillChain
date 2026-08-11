"""Canonical query/gallery eligibility manifests and verification.

The manifest is a content-addressed proof over one exact Query artifact, one
exact gallery artifact, and one verified asset catalog.  Consumers must call
``verify_gallery_eligibility`` with those artifacts; trusting the JSON fields
alone is deliberately unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Annotated, Any, Literal, Sequence

from pydantic import (
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import (
    AssetCatalog,
    GalleryAssetReference,
    QueryAssetReference,
    QueryGalleryLeakageReport,
    assert_query_gallery_eligible,
    audit_query_gallery_eligibility,
)
from skillchain.schemas import DatasetAsset, Query, StrictModel
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ELIGIBILITY_POLICY_VERSION = "query-gallery-eligibility-v1"


class GalleryEligibilityError(ValueError):
    """An eligibility artifact or its transitive evidence is invalid."""


class GalleryEligibilityViolation(GalleryEligibilityError):
    """The recomputed query/gallery audit contains a blocking violation."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("query/gallery eligibility audit failed")
        self.payload = payload


class GalleryEligibilityManifest(StrictModel):
    """A successful, canonical eligibility proof.

    ``eligible`` and ``status`` are literals so a blocked diagnostic can never
    be mistaken for a consumable manifest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    eligibility_policy_version: Literal["query-gallery-eligibility-v1"] = (
        ELIGIBILITY_POLICY_VERSION
    )
    catalog_policy_version: str
    catalog_sha256: Sha256
    leakage_policy_version: str
    command: Literal["audit-gallery"] = "audit-gallery"
    eligible: Literal[True] = True
    query_artifact_sha256: Sha256
    gallery_artifact_sha256: Sha256
    report: QueryGalleryLeakageReport
    status: Literal["ok"] = "ok"
    eligibility_sha256: Sha256

    @field_validator("catalog_policy_version", "leakage_policy_version")
    @classmethod
    def validate_policy_version(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @model_validator(mode="after")
    def validate_success_report(self):
        if self.report.violation_count:
            raise ValueError(
                "successful eligibility manifest must not contain violations"
            )
        return self


@dataclass(frozen=True)
class VerifiedGalleryEligibility:
    """Eligibility evidence reloaded and recomputed from current artifacts."""

    manifest: GalleryEligibilityManifest
    manifest_bytes: bytes
    query_bytes: bytes
    queries: tuple[Query, ...]
    gallery_bytes: bytes
    gallery_references: tuple[GalleryAssetReference, ...]
    gallery_assets: tuple[DatasetAsset, ...]
    catalog: AssetCatalog


def load_query_artifact(
    path: str | Path,
    catalog: AssetCatalog,
) -> tuple[bytes, tuple[Query, ...]]:
    """Load canonical schema-v2 Query JSONL and verify catalog references."""

    _require_verified_catalog(catalog)
    content = _read_artifact(path, "queries")
    rows = _load_jsonl_objects(content, "queries")
    queries: list[Query] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            queries.append(Query.model_validate(row))
        except ValidationError as exc:
            raise GalleryEligibilityError(
                f"queries line {line_number} violates schema-v2 Query: {exc}"
            ) from exc
    if len({query.query_id for query in queries}) != len(queries):
        raise GalleryEligibilityError("queries query_id must be unique")
    canonical = canonical_jsonl_bytes(queries)
    if content != canonical:
        raise GalleryEligibilityError("queries must be canonical schema-v2 Query JSONL")
    for query in queries:
        catalog.verify_reference(
            query.asset_id,
            query.image_path,
            leakage_group_id=query.leakage_group_id,
        )
    return content, tuple(queries)


def load_gallery_artifact(
    path: str | Path,
    catalog: AssetCatalog,
) -> tuple[bytes, tuple[GalleryAssetReference, ...]]:
    """Load canonical gallery JSONL and verify every catalog reference."""

    _require_verified_catalog(catalog)
    content = _read_artifact(path, "gallery assets")
    rows = _load_jsonl_objects(content, "gallery assets")
    references: list[GalleryAssetReference] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            references.append(GalleryAssetReference.model_validate(row))
        except ValidationError as exc:
            raise GalleryEligibilityError(
                f"gallery assets line {line_number} is invalid: {exc}"
            ) from exc
    asset_ids = [reference.asset_id for reference in references]
    if len(asset_ids) != len(set(asset_ids)):
        raise GalleryEligibilityError("gallery asset_id must be unique")
    image_paths = [reference.image_path for reference in references]
    if len(image_paths) != len(set(image_paths)):
        raise GalleryEligibilityError("gallery image_path must be unique")
    if content != canonical_jsonl_bytes(references):
        raise GalleryEligibilityError("gallery assets must be canonical JSONL")
    for reference in references:
        catalog.verify_reference(reference.asset_id, reference.image_path)
    return content, tuple(references)


def build_gallery_eligibility_manifest(
    query_path: str | Path,
    gallery_path: str | Path,
    catalog: AssetCatalog,
) -> GalleryEligibilityManifest:
    """Recompute the audit and build a strong success-only manifest."""

    query_bytes, queries = load_query_artifact(query_path, catalog)
    gallery_bytes, gallery_references = load_gallery_artifact(gallery_path, catalog)
    report = _audit(queries, gallery_references, catalog)
    common = _common_payload(query_bytes, gallery_bytes, report, catalog)
    if report.violation_count:
        blocked = {**common, "eligible": False, "status": "blocked"}
        blocked["eligibility_sha256"] = sha256_bytes(canonical_json_bytes(blocked))
        try:
            assert_query_gallery_eligible(report)
        except ValueError as exc:
            blocked["message"] = str(exc)
        raise GalleryEligibilityViolation(blocked)

    unsigned = {**common, "eligible": True, "status": "ok"}
    return GalleryEligibilityManifest.model_validate(
        {
            **unsigned,
            "eligibility_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )


def verify_gallery_eligibility(
    manifest_path: str | Path,
    query_path: str | Path,
    gallery_path: str | Path,
    catalog: AssetCatalog,
) -> VerifiedGalleryEligibility:
    """Strictly verify and recompute a published eligibility manifest."""

    _require_verified_catalog(catalog)
    manifest = load_gallery_eligibility_manifest(manifest_path)
    manifest_bytes = canonical_json_bytes(manifest)

    if manifest.catalog_sha256 != catalog.catalog_sha256:
        raise GalleryEligibilityError(
            "eligibility manifest catalog_sha256 does not match current catalog"
        )
    if manifest.catalog_policy_version != catalog.manifest.catalog_policy_version:
        raise GalleryEligibilityError(
            "eligibility manifest catalog_policy_version does not match current catalog"
        )
    if manifest.leakage_policy_version != catalog.leakage_policy_version:
        raise GalleryEligibilityError(
            "eligibility manifest leakage_policy_version does not match current catalog"
        )

    query_bytes, queries = load_query_artifact(query_path, catalog)
    gallery_bytes, gallery_references = load_gallery_artifact(gallery_path, catalog)
    if sha256_bytes(query_bytes) != manifest.query_artifact_sha256:
        raise GalleryEligibilityError(
            "eligibility manifest query_artifact_sha256 does not match current queries"
        )
    if sha256_bytes(gallery_bytes) != manifest.gallery_artifact_sha256:
        raise GalleryEligibilityError(
            "eligibility manifest gallery_artifact_sha256 does not match current gallery"
        )

    recomputed_report = _audit(queries, gallery_references, catalog)
    if recomputed_report.violation_count:
        raise GalleryEligibilityError(
            "current query/gallery artifacts are no longer eligible"
        )
    if recomputed_report.model_dump(mode="json") != manifest.report.model_dump(
        mode="json"
    ):
        raise GalleryEligibilityError(
            "eligibility manifest report does not match the recomputed audit"
        )
    if manifest.report.violation_count:
        raise GalleryEligibilityError(
            "successful eligibility manifest contains blocking violations"
        )

    gallery_assets = tuple(
        catalog.resolve_asset_id(reference.asset_id).asset
        for reference in gallery_references
    )
    return VerifiedGalleryEligibility(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        query_bytes=query_bytes,
        queries=queries,
        gallery_bytes=gallery_bytes,
        gallery_references=gallery_references,
        gallery_assets=gallery_assets,
        catalog=catalog,
    )


def load_gallery_eligibility_manifest(
    path: str | Path,
) -> GalleryEligibilityManifest:
    """Load only canonical manifest structure and its content self-hash.

    This intentionally does not establish catalog or artifact eligibility;
    consumers needing that guarantee must call ``verify_gallery_eligibility``.
    """

    manifest_bytes = _read_artifact(path, "eligibility manifest")
    raw_manifest = _load_json_object(manifest_bytes, "eligibility manifest")
    try:
        manifest = GalleryEligibilityManifest.model_validate(raw_manifest)
    except ValidationError as exc:
        raise GalleryEligibilityError(
            f"eligibility manifest violates schema: {exc}"
        ) from exc
    if manifest_bytes != canonical_json_bytes(manifest):
        raise GalleryEligibilityError("eligibility manifest must be canonical JSON")
    _verify_manifest_self_hash(manifest)
    return manifest


def _audit(
    queries: Sequence[Query],
    gallery_references: Sequence[GalleryAssetReference],
    catalog: AssetCatalog,
) -> QueryGalleryLeakageReport:
    query_references = tuple(
        QueryAssetReference(
            query_id=query.query_id,
            intent=query.canonical_intent,
            asset_id=query.asset_id,
        )
        for query in queries
    )
    return audit_query_gallery_eligibility(
        query_references,
        (reference.asset_id for reference in gallery_references),
        catalog,
    )


def _common_payload(
    query_bytes: bytes,
    gallery_bytes: bytes,
    report: QueryGalleryLeakageReport,
    catalog: AssetCatalog,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "catalog_policy_version": catalog.manifest.catalog_policy_version,
        "catalog_sha256": catalog.catalog_sha256,
        "leakage_policy_version": catalog.leakage_policy_version,
        "command": "audit-gallery",
        "query_artifact_sha256": sha256_bytes(query_bytes),
        "gallery_artifact_sha256": sha256_bytes(gallery_bytes),
        "report": report.model_dump(mode="json"),
    }


def _verify_manifest_self_hash(manifest: GalleryEligibilityManifest) -> None:
    unsigned = manifest.model_dump(mode="json", exclude={"eligibility_sha256"})
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if manifest.eligibility_sha256 != expected:
        raise GalleryEligibilityError("eligibility manifest self hash mismatch")


def _require_verified_catalog(catalog: AssetCatalog) -> None:
    try:
        catalog.require_verified_files()
    except ValueError as exc:
        raise GalleryEligibilityError(str(exc)) from exc


def _read_artifact(path: str | Path, label: str) -> bytes:
    path = Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise GalleryEligibilityError(f"{label} does not exist: {path}") from None
    except OSError as exc:
        raise GalleryEligibilityError(f"unable to inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise GalleryEligibilityError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise GalleryEligibilityError(f"{label} must be a regular file: {path}")
    try:
        with path.open("rb") as source:
            opened_metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise GalleryEligibilityError(
                    f"{label} must remain a regular file while reading: {path}"
                )
            if (metadata.st_dev, metadata.st_ino) != (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
            ):
                raise GalleryEligibilityError(
                    f"{label} changed before it could be read: {path}"
                )
            return source.read()
    except OSError as exc:
        raise GalleryEligibilityError(f"unable to read {label}: {path}") from exc


def _load_jsonl_objects(content: bytes, label: str) -> tuple[dict[str, Any], ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GalleryEligibilityError(f"{label} must be UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise GalleryEligibilityError(f"{label} JSONL must contain at least one row")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GalleryEligibilityError(f"{label} line {line_number} is blank")
        row = _decode_json(line, f"{label} line {line_number}")
        if not isinstance(row, dict):
            raise GalleryEligibilityError(
                f"{label} line {line_number} must be an object"
            )
        rows.append(row)
    return tuple(rows)


def _load_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GalleryEligibilityError(f"{label} must be UTF-8") from exc
    value = _decode_json(text, label)
    if not isinstance(value, dict):
        raise GalleryEligibilityError(f"{label} root must be an object")
    return value


def _decode_json(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise GalleryEligibilityError(f"{label} is invalid JSON: {exc}") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GalleryEligibilityError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


__all__ = [
    "ELIGIBILITY_POLICY_VERSION",
    "GalleryEligibilityError",
    "GalleryEligibilityManifest",
    "GalleryEligibilityViolation",
    "VerifiedGalleryEligibility",
    "build_gallery_eligibility_manifest",
    "load_gallery_artifact",
    "load_gallery_eligibility_manifest",
    "load_query_artifact",
    "verify_gallery_eligibility",
]
