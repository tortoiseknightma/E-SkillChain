"""Owner-authorized remote-processing overlays for Portfolio catalogs.

The accepted query corpus and its source adapters are immutable historical
artifacts.  A later owner decision may authorize remote model inference for the
exact selected assets without rewriting those artifacts.  This module verifies
that decision against either the legacy mini selection manifest or the Core v9
all-catalog-assets manifest and the byte-verified base catalog, then publishes a
create-only catalog whose only asset-field change is
``cloud_upload_allowed=True``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import (
    AssetCatalog,
    DatasetAsset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MAX_AUTHORIZATION_BYTES = 2 * 1024 * 1024
_MAX_SELECTION_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_DATASET_ASSETS_BYTES = 64 * 1024 * 1024
_MAX_PLAN_BYTES = 64 * 1024 * 1024
_MAX_QUERY_BYTES = 64 * 1024 * 1024
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_VERIFIED_RUNTIME_TOKEN = object()

PortfolioProcessor = Literal[
    "aifast-gemini-feedback",
    "aifast-gemini-judge",
    "dashscope-kimi-feedback",
    "dashscope-kimi-judge",
    "dashscope-qwen-assistant",
]
PortfolioRemoteProcessingScope = Literal[
    "dev_mini-selected-query-assets",
    "core-v9-all-catalog-assets",
]
_CORE_V9_PROCESSORS_V1: tuple[PortfolioProcessor, ...] = (
    "dashscope-kimi-feedback",
    "dashscope-kimi-judge",
    "dashscope-qwen-assistant",
)
_CORE_V9_PROCESSORS_V2: tuple[PortfolioProcessor, ...] = (
    "aifast-gemini-judge",
    "dashscope-kimi-feedback",
    "dashscope-qwen-assistant",
)


class PortfolioRemoteProcessingError(ValueError):
    """The authorization or the catalog transition is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PortfolioSourceRemoteDecision(_StrictFrozenModel):
    source_id: str
    source_revision: str = Field(min_length=1)
    # The mini manifest exposes one source artifact digest per selected row.
    # Core v9 instead binds source provenance through the immutable selection
    # manifest, inventory/source-policy digests, DatasetAssetDraft rows, and
    # catalog file digests, so a per-source digest is not available there.
    source_artifact_sha256: Sha256 | None = None
    asset_count: int = Field(gt=0)
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID.fullmatch(value):
            raise ValueError("source_id must be canonical")
        return value

    @field_validator("source_revision")
    @classmethod
    def validate_source_revision(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("source_revision must be canonical")
        return value


class PortfolioRemoteProcessingAuthorization(_StrictFrozenModel):
    schema_version: Literal[1, 2] = 1
    authorization_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    status: Literal["owner-approved"] = "owner-approved"
    track: Literal["portfolio"] = "portfolio"
    scope: PortfolioRemoteProcessingScope = "dev_mini-selected-query-assets"
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    owner_statement: str = Field(min_length=1)
    base_selection_manifest_sha256: Sha256
    base_dataset_assets_sha256: Sha256
    base_catalog_sha256: Sha256
    base_catalog_manifest_file_sha256: Sha256
    query_artifact_sha256: Sha256
    plan_sha256: Sha256
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    processor_scope: tuple[PortfolioProcessor, ...]
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    source_decisions: tuple[PortfolioSourceRemoteDecision, ...]
    risk_boundary: str = Field(min_length=1)
    # Core-v9-only bindings.  They remain optional at the field level so
    # historical mini authorization files continue to validate byte-for-byte;
    # the scope-aware model validator below makes the complete set mandatory
    # for Core v9.
    kind: Literal["portfolio-core-remote-processing-owner-authorization"] | None = None
    authorized_asset_count: int | None = Field(default=None, gt=0)
    base_catalog_asset_file_sha256: Sha256 | None = None
    base_catalog_components_file_sha256: Sha256 | None = None
    base_catalog_permission_counts: dict[str, int] | None = None
    query_count: int | None = Field(default=None, gt=0)
    query_referenced_unique_asset_count: int | None = Field(default=None, gt=0)
    formal_eligible: Literal[False] | None = None
    runtime_overlay_status: (
        Literal["pending-create-only-publication-and-preflight"] | None
    ) = None

    @field_validator("processor_scope", "source_decisions", mode="before")
    @classmethod
    def coerce_json_arrays(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("reviewer_id", "owner_statement", "risk_boundary")
    @classmethod
    def validate_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("authorization text fields must be canonical")
        return value

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        if self.processor_scope != tuple(sorted(set(self.processor_scope))):
            raise ValueError("processor_scope must be sorted and unique")
        if not self.processor_scope:
            raise ValueError("processor_scope must not be empty")
        source_ids = tuple(item.source_id for item in self.source_decisions)
        if not source_ids or source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("source_decisions must be non-empty, sorted, and unique")
        if self.scope == "dev_mini-selected-query-assets":
            if any(
                item.source_artifact_sha256 is None for item in self.source_decisions
            ):
                raise ValueError("mini source_decisions require source_artifact_sha256")
            return self

        required_core_values = {
            "kind": self.kind,
            "authorized_asset_count": self.authorized_asset_count,
            "base_catalog_asset_file_sha256": (self.base_catalog_asset_file_sha256),
            "base_catalog_components_file_sha256": (
                self.base_catalog_components_file_sha256
            ),
            "base_catalog_permission_counts": self.base_catalog_permission_counts,
            "query_count": self.query_count,
            "query_referenced_unique_asset_count": (
                self.query_referenced_unique_asset_count
            ),
            "formal_eligible": self.formal_eligible,
            "runtime_overlay_status": self.runtime_overlay_status,
        }
        missing = sorted(
            name for name, value in required_core_values.items() if value is None
        )
        if missing:
            raise ValueError(
                "Core v9 authorization lacks required bindings: " + ", ".join(missing)
            )
        expected_processors = (
            _CORE_V9_PROCESSORS_V1
            if self.schema_version == 1
            else _CORE_V9_PROCESSORS_V2
        )
        if self.processor_scope != expected_processors:
            raise ValueError(
                "Core v9 processor_scope differs from the schema-versioned "
                "owner-approved processor set"
            )
        assert self.authorized_asset_count is not None
        if sum(item.asset_count for item in self.source_decisions) != (
            self.authorized_asset_count
        ):
            raise ValueError(
                "Core v9 authorized asset count differs from source decisions"
            )
        assert self.base_catalog_permission_counts is not None
        if (
            set(self.base_catalog_permission_counts)
            != {
                "false",
                "true",
                "unknown",
            }
            or any(
                type(value) is not int or value < 0
                for value in self.base_catalog_permission_counts.values()
            )
            or sum(self.base_catalog_permission_counts.values())
            != self.authorized_asset_count
        ):
            raise ValueError("Core v9 base catalog permission counts are invalid")
        assert self.query_count is not None
        assert self.query_referenced_unique_asset_count is not None
        if self.query_referenced_unique_asset_count > self.query_count:
            raise ValueError("Core v9 unique query asset count exceeds query count")
        return self


class PortfolioRemoteProcessingReceipt(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-remote-processing-catalog-receipt"] = (
        "portfolio-remote-processing-catalog-receipt"
    )
    policy_version: Literal["portfolio-remote-processing-overlay-v1"] = (
        "portfolio-remote-processing-overlay-v1"
    )
    authorization_id: str
    authorization_file_sha256: Sha256
    base_selection_manifest_sha256: Sha256
    base_dataset_assets_sha256: Sha256
    base_catalog_sha256: Sha256
    base_catalog_manifest_file_sha256: Sha256
    output_catalog_sha256: Sha256
    output_catalog_manifest_file_sha256: Sha256
    asset_count: int = Field(gt=0)
    source_asset_counts: dict[str, int]
    before_permission_counts: dict[str, int]
    after_permission_counts: dict[str, int]
    asset_id_set_sha256: Sha256
    image_binding_set_sha256: Sha256
    base_components_sha256: Sha256
    output_components_sha256: Sha256
    components_unchanged: Literal[True] = True
    public_demo_allowed_count: Literal[0] = 0
    redistribution_allowed: Literal[False] = False
    formal_eligible: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("receipt self hash mismatch")
        required_count_keys = {"false", "true", "unknown"}
        if (
            set(self.before_permission_counts) != required_count_keys
            or any(value < 0 for value in self.before_permission_counts.values())
            or sum(self.before_permission_counts.values()) != self.asset_count
        ):
            raise ValueError("base Portfolio permission counts are invalid")
        if self.after_permission_counts != {
            "false": 0,
            "true": self.asset_count,
            "unknown": 0,
        }:
            raise ValueError("output Portfolio permissions are incomplete")
        if self.base_components_sha256 != self.output_components_sha256:
            raise ValueError("leakage components changed during permission overlay")
        return self


@dataclass(frozen=True)
class PortfolioRemoteProcessingResult:
    output_catalog: Path
    output_catalog_sha256: str
    receipt_path: Path
    receipt_sha256: str
    authorization_file_sha256: str
    asset_count: int


@dataclass(frozen=True)
class PortfolioQueryBindingResult:
    query_count: int
    unique_asset_count: int
    catalog_sha256: str
    query_artifact_sha256: str


@dataclass(frozen=True)
class VerifiedPortfolioRemoteProcessingRuntime:
    """Freshly verified owner authority and runtime catalog binding."""

    authorization: PortfolioRemoteProcessingAuthorization
    receipt: PortfolioRemoteProcessingReceipt
    processor: PortfolioProcessor
    catalog: AssetCatalog = field(repr=False, compare=False)
    authorization_file_sha256: str
    receipt_file_sha256: str
    dataset_assets_sha256: str
    plan_sha256: str
    query_artifact_sha256: str
    _verification_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _SelectionBindings:
    rows: tuple[dict[str, object], ...]
    dataset_assets_sha256: str
    dataset_asset_count: int
    manifest_kind: Literal["mini", "core-v9"]
    dataset_assets_path: str | None = None
    dataset_asset_drafts: tuple[dict[str, object], ...] | None = None


def load_portfolio_remote_processing_authorization(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[PortfolioRemoteProcessingAuthorization, bytes]:
    expected_sha256 = _require_sha256(expected_sha256, "authorization SHA-256")
    content = read_stable_regular_file(
        path,
        label="Portfolio remote-processing authorization",
        max_bytes=_MAX_AUTHORIZATION_BYTES,
    )
    if sha256_bytes(content) != expected_sha256:
        raise PortfolioRemoteProcessingError("authorization SHA-256 mismatch")
    try:
        parse_canonical_json(
            content,
            label="Portfolio remote-processing authorization",
        )
        authorization = PortfolioRemoteProcessingAuthorization.model_validate_json(
            content,
            strict=True,
        )
    except (ArtifactFormatError, ValueError) as error:
        raise PortfolioRemoteProcessingError(str(error)) from error
    return authorization, content


def publish_remote_authorized_portfolio_catalog(
    *,
    authorization_file: str | Path,
    expected_authorization_sha256: str,
    selection_manifest: str | Path,
    base_catalog: str | Path,
    asset_root: str | Path,
    output_catalog: str | Path,
    receipt_output: str | Path,
) -> PortfolioRemoteProcessingResult:
    """Publish a v2 runtime catalog after exact owner authorization.

    The base catalog, selection manifest, and selected image files are all
    reverified.  No query, image, adapter, source-review record, or public-demo
    permission is modified.
    """

    authorization, authorization_content = (
        load_portfolio_remote_processing_authorization(
            authorization_file,
            expected_sha256=expected_authorization_sha256,
        )
    )
    selection_content = read_stable_regular_file(
        selection_manifest,
        label="Portfolio mini selection manifest",
        max_bytes=_MAX_SELECTION_BYTES,
    )
    selection_sha256 = sha256_bytes(selection_content)
    if selection_sha256 != authorization.base_selection_manifest_sha256:
        raise PortfolioRemoteProcessingError("selection manifest SHA-256 mismatch")
    try:
        raw_selection = parse_canonical_json(
            selection_content,
            label="Portfolio mini selection manifest",
        )
    except ArtifactFormatError as error:
        raise PortfolioRemoteProcessingError(str(error)) from error
    selection = _selection_bindings(raw_selection)
    _verify_selection_scope(authorization, selection)
    if selection.dataset_assets_sha256 != authorization.base_dataset_assets_sha256:
        raise PortfolioRemoteProcessingError("dataset-assets SHA-256 mismatch")

    base_manifest_content = read_stable_regular_file(
        Path(base_catalog) / "manifest.json",
        label="base AssetCatalog manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if (
        sha256_bytes(base_manifest_content)
        != authorization.base_catalog_manifest_file_sha256
    ):
        raise PortfolioRemoteProcessingError(
            "base catalog manifest file SHA-256 mismatch"
        )
    catalog = load_asset_catalog(base_catalog, asset_root, verify_files=True)
    catalog.require_verified_files()
    if catalog.catalog_sha256 != authorization.base_catalog_sha256:
        raise PortfolioRemoteProcessingError("base catalog SHA-256 mismatch")
    _verify_authorized_catalog(authorization, catalog)

    source_counts = _verify_authorized_selection(
        authorization=authorization,
        selections=selection.rows,
        catalog=catalog,
    )
    before_counts = _permission_counts(catalog.assets)
    upgraded_assets = tuple(
        asset.model_copy(update={"cloud_upload_allowed": True})
        for asset in catalog.assets
    )
    if any(asset.public_demo_allowed for asset in upgraded_assets):
        raise PortfolioRemoteProcessingError(
            "remote-processing authorization must not enable public demo"
        )

    output_path = publish_asset_catalog(
        upgraded_assets,
        output_catalog,
        asset_root,
        policy=catalog.manifest.near_duplicate_policy,
        coverage_roots=catalog.manifest.coverage_roots,
    )
    output = load_asset_catalog(output_path, asset_root, verify_files=True)
    output.require_verified_files()
    _verify_permission_only_transition(catalog, output)

    output_manifest_content = read_stable_regular_file(
        output_path / "manifest.json",
        label="output AssetCatalog manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    unsigned_receipt = {
        "schema_version": 1,
        "kind": "portfolio-remote-processing-catalog-receipt",
        "policy_version": "portfolio-remote-processing-overlay-v1",
        "authorization_id": authorization.authorization_id,
        "authorization_file_sha256": sha256_bytes(authorization_content),
        "base_selection_manifest_sha256": selection_sha256,
        "base_dataset_assets_sha256": selection.dataset_assets_sha256,
        "base_catalog_sha256": catalog.catalog_sha256,
        "base_catalog_manifest_file_sha256": sha256_bytes(base_manifest_content),
        "output_catalog_sha256": output.catalog_sha256,
        "output_catalog_manifest_file_sha256": sha256_bytes(output_manifest_content),
        "asset_count": len(output.assets),
        "source_asset_counts": dict(sorted(source_counts.items())),
        "before_permission_counts": before_counts,
        "after_permission_counts": _permission_counts(output.assets),
        "asset_id_set_sha256": _asset_id_set_sha256(output.assets),
        "image_binding_set_sha256": _image_binding_set_sha256(output.assets),
        "base_components_sha256": catalog.manifest.components.sha256,
        "output_components_sha256": output.manifest.components.sha256,
        "components_unchanged": True,
        "public_demo_allowed_count": output.manifest.public_demo_allowed_count,
        "redistribution_allowed": False,
        "formal_eligible": False,
    }
    receipt = PortfolioRemoteProcessingReceipt.model_validate(
        {
            **unsigned_receipt,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(unsigned_receipt)),
        },
        strict=True,
    )
    receipt_path = atomic_create_file(
        receipt_output,
        canonical_json_bytes(receipt.model_dump(mode="json")),
    )
    return PortfolioRemoteProcessingResult(
        output_catalog=output_path,
        output_catalog_sha256=output.catalog_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt.receipt_sha256,
        authorization_file_sha256=sha256_bytes(authorization_content),
        asset_count=len(output.assets),
    )


def load_portfolio_remote_processing_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
) -> PortfolioRemoteProcessingReceipt:
    content = read_stable_regular_file(
        path,
        label="Portfolio remote-processing receipt",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if sha256_bytes(content) != _require_sha256(
        expected_sha256,
        "receipt file SHA-256",
    ):
        raise PortfolioRemoteProcessingError("receipt file SHA-256 mismatch")
    try:
        parse_canonical_json(
            content,
            label="Portfolio remote-processing receipt",
        )
        return PortfolioRemoteProcessingReceipt.model_validate_json(
            content,
            strict=True,
        )
    except (ArtifactFormatError, ValueError) as error:
        raise PortfolioRemoteProcessingError(str(error)) from error


def verify_remote_authorized_query_bindings(
    *,
    catalog_dir: str | Path,
    asset_root: str | Path,
    queries_path: str | Path,
    expected_query_artifact_sha256: str | None = None,
    _verified_catalog: AssetCatalog | None = None,
) -> PortfolioQueryBindingResult:
    """Verify every accepted query against a cloud-enabled catalog."""

    if _verified_catalog is None:
        catalog = load_asset_catalog(catalog_dir, asset_root, verify_files=True)
    else:
        if type(_verified_catalog) is not AssetCatalog or (
            _verified_catalog.root.resolve() != Path(catalog_dir).resolve()
            or _verified_catalog.asset_root != Path(asset_root).resolve()
        ):
            raise PortfolioRemoteProcessingError(
                "preverified query catalog differs from runtime paths"
            )
        catalog = _verified_catalog
    catalog.require_verified_files()
    query_content = read_stable_regular_file(
        queries_path,
        label="accepted Portfolio query artifact",
        max_bytes=_MAX_QUERY_BYTES,
    )
    query_artifact_sha256 = sha256_bytes(query_content)
    if (
        expected_query_artifact_sha256 is not None
        and query_artifact_sha256
        != _require_sha256(
            expected_query_artifact_sha256,
            "expected query artifact SHA-256",
        )
    ):
        raise PortfolioRemoteProcessingError("query artifact SHA-256 mismatch")
    try:
        rows = parse_canonical_jsonl(
            query_content,
            label="accepted Portfolio query artifact",
        )
    except ArtifactFormatError as error:
        raise PortfolioRemoteProcessingError(str(error)) from error
    if not rows:
        raise PortfolioRemoteProcessingError("accepted query artifact is empty")
    asset_ids: set[str] = set()
    query_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PortfolioRemoteProcessingError("query row must be an object")
        query_id = row.get("query_id")
        asset_id = row.get("asset_id")
        image_path = row.get("image_path")
        if not all(
            isinstance(value, str) and value
            for value in (
                query_id,
                asset_id,
                image_path,
            )
        ):
            raise PortfolioRemoteProcessingError(
                "query row lacks query_id, asset_id, or image_path"
            )
        if query_id in query_ids:
            raise PortfolioRemoteProcessingError("query ID is duplicated")
        query_ids.add(query_id)
        resolution = catalog.verify_reference(asset_id, image_path)
        if resolution.asset.cloud_upload_allowed is not True:
            raise PortfolioRemoteProcessingError(
                f"query asset is not cloud-enabled: {query_id}"
            )
        asset_ids.add(asset_id)
    return PortfolioQueryBindingResult(
        query_count=len(rows),
        unique_asset_count=len(asset_ids),
        catalog_sha256=catalog.catalog_sha256,
        query_artifact_sha256=query_artifact_sha256,
    )


def verify_portfolio_remote_processing_runtime(
    *,
    authorization_file: str | Path,
    expected_authorization_sha256: str,
    receipt_file: str | Path,
    expected_receipt_file_sha256: str,
    selection_manifest: str | Path,
    dataset_assets: str | Path,
    base_catalog: str | Path,
    output_catalog: str | Path,
    asset_root: str | Path,
    plan_file: str | Path,
    queries_file: str | Path,
    processor: PortfolioProcessor,
    _verified_catalogs: tuple[AssetCatalog, AssetCatalog] | None = None,
) -> VerifiedPortfolioRemoteProcessingRuntime:
    """Freshly verify the complete permission chain before remote execution."""

    authorization, authorization_content = (
        load_portfolio_remote_processing_authorization(
            authorization_file,
            expected_sha256=expected_authorization_sha256,
        )
    )
    if processor not in authorization.processor_scope:
        raise PortfolioRemoteProcessingError(
            f"processor is outside the owner-authorized scope: {processor}"
        )
    authorization_file_sha256 = sha256_bytes(authorization_content)

    dataset_assets_content = read_stable_regular_file(
        dataset_assets,
        label="Portfolio mini dataset assets",
        max_bytes=_MAX_DATASET_ASSETS_BYTES,
    )
    dataset_assets_sha256 = sha256_bytes(dataset_assets_content)
    if dataset_assets_sha256 != authorization.base_dataset_assets_sha256:
        raise PortfolioRemoteProcessingError("dataset-assets file SHA-256 mismatch")
    try:
        dataset_asset_rows = parse_canonical_jsonl(
            dataset_assets_content,
            label="Portfolio mini dataset assets",
        )
    except ArtifactFormatError as error:
        raise PortfolioRemoteProcessingError(str(error)) from error

    selection_content = read_stable_regular_file(
        selection_manifest,
        label="Portfolio mini selection manifest",
        max_bytes=_MAX_SELECTION_BYTES,
    )
    selection_sha256 = sha256_bytes(selection_content)
    if selection_sha256 != authorization.base_selection_manifest_sha256:
        raise PortfolioRemoteProcessingError("selection manifest SHA-256 mismatch")
    try:
        raw_selection = parse_canonical_json(
            selection_content,
            label="Portfolio mini selection manifest",
        )
    except ArtifactFormatError as error:
        raise PortfolioRemoteProcessingError(str(error)) from error
    if not isinstance(raw_selection, dict):
        raise PortfolioRemoteProcessingError("selection manifest must be an object")
    selection = _selection_bindings(raw_selection)
    _verify_selection_scope(authorization, selection)
    if selection.dataset_assets_sha256 != dataset_assets_sha256:
        raise PortfolioRemoteProcessingError(
            "selection manifest does not bind the supplied dataset-assets file"
        )
    if len(dataset_asset_rows) != selection.dataset_asset_count or len(
        dataset_asset_rows
    ) != len(selection.rows):
        raise PortfolioRemoteProcessingError(
            "dataset-assets descriptor or row count differs from selection"
        )
    if selection.manifest_kind == "mini":
        if selection.dataset_assets_path != Path(dataset_assets).name:
            raise PortfolioRemoteProcessingError(
                "dataset-assets descriptor or row count differs from selection"
            )
    elif selection.dataset_asset_drafts != tuple(dataset_asset_rows):
        raise PortfolioRemoteProcessingError(
            "Core v9 dataset-assets rows differ from selection drafts"
        )

    base_manifest_content = read_stable_regular_file(
        Path(base_catalog) / "manifest.json",
        label="base AssetCatalog manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    base_manifest_file_sha256 = sha256_bytes(base_manifest_content)
    if base_manifest_file_sha256 != authorization.base_catalog_manifest_file_sha256:
        raise PortfolioRemoteProcessingError(
            "base catalog manifest file SHA-256 mismatch"
        )
    if _verified_catalogs is None:
        base = load_asset_catalog(base_catalog, asset_root, verify_files=True)
        output = load_asset_catalog(output_catalog, asset_root, verify_files=True)
    else:
        if (
            not isinstance(_verified_catalogs, tuple)
            or len(_verified_catalogs) != 2
            or any(type(item) is not AssetCatalog for item in _verified_catalogs)
        ):
            raise PortfolioRemoteProcessingError("preverified catalog pair is invalid")
        base, output = _verified_catalogs
        expected_asset_root = Path(asset_root).resolve()
        if (
            base.root.resolve() != Path(base_catalog).resolve()
            or output.root.resolve() != Path(output_catalog).resolve()
            or base.asset_root != expected_asset_root
            or output.asset_root != expected_asset_root
        ):
            raise PortfolioRemoteProcessingError(
                "preverified catalog pair differs from runtime paths"
            )
    base.require_verified_files()
    output.require_verified_files()
    if base.catalog_sha256 != authorization.base_catalog_sha256:
        raise PortfolioRemoteProcessingError("base catalog SHA-256 mismatch")
    _verify_authorized_catalog(authorization, base)
    source_counts = _verify_authorized_selection(
        authorization=authorization,
        selections=selection.rows,
        catalog=base,
    )

    _verify_permission_only_transition(base, output)
    output_manifest_content = read_stable_regular_file(
        Path(output_catalog) / "manifest.json",
        label="output AssetCatalog manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )

    receipt = load_portfolio_remote_processing_receipt(
        receipt_file,
        expected_sha256=expected_receipt_file_sha256,
    )
    expected_receipt_values = {
        "authorization_id": authorization.authorization_id,
        "authorization_file_sha256": authorization_file_sha256,
        "base_selection_manifest_sha256": selection_sha256,
        "base_dataset_assets_sha256": dataset_assets_sha256,
        "base_catalog_sha256": base.catalog_sha256,
        "base_catalog_manifest_file_sha256": base_manifest_file_sha256,
        "output_catalog_sha256": output.catalog_sha256,
        "output_catalog_manifest_file_sha256": sha256_bytes(output_manifest_content),
        "asset_count": len(output.assets),
        "source_asset_counts": dict(sorted(source_counts.items())),
        "before_permission_counts": _permission_counts(base.assets),
        "after_permission_counts": _permission_counts(output.assets),
        "asset_id_set_sha256": _asset_id_set_sha256(output.assets),
        "image_binding_set_sha256": _image_binding_set_sha256(output.assets),
        "base_components_sha256": base.manifest.components.sha256,
        "output_components_sha256": output.manifest.components.sha256,
        "public_demo_allowed_count": output.manifest.public_demo_allowed_count,
    }
    for name, expected in expected_receipt_values.items():
        if getattr(receipt, name) != expected:
            raise PortfolioRemoteProcessingError(
                f"runtime differs from permission receipt: {name}"
            )

    expected_plan_scope = (
        "core" if authorization.scope == "core-v9-all-catalog-assets" else "dev_mini"
    )
    plan_content = read_stable_regular_file(
        plan_file,
        label=f"Portfolio {expected_plan_scope} plan",
        max_bytes=_MAX_PLAN_BYTES,
    )
    plan_sha256 = sha256_bytes(plan_content)
    if plan_sha256 != authorization.plan_sha256:
        raise PortfolioRemoteProcessingError("plan SHA-256 mismatch")
    try:
        raw_plan = parse_canonical_json(
            plan_content,
            label=f"Portfolio {expected_plan_scope} plan",
        )
    except ArtifactFormatError as error:
        raise PortfolioRemoteProcessingError(str(error)) from error
    if (
        not isinstance(raw_plan, dict)
        or raw_plan.get("scope") != expected_plan_scope
        or raw_plan.get("asset_catalog_sha256") != base.catalog_sha256
        or not isinstance(raw_plan.get("queries"), list)
    ):
        raise PortfolioRemoteProcessingError(
            f"plan does not bind the historical {expected_plan_scope} catalog"
        )

    query_bindings = verify_remote_authorized_query_bindings(
        catalog_dir=output_catalog,
        asset_root=asset_root,
        queries_path=queries_file,
        expected_query_artifact_sha256=authorization.query_artifact_sha256,
        _verified_catalog=output,
    )
    if len(raw_plan["queries"]) != query_bindings.query_count:
        raise PortfolioRemoteProcessingError(
            "plan/query count differs during runtime preflight"
        )
    if authorization.scope == "core-v9-all-catalog-assets":
        if authorization.query_count != query_bindings.query_count:
            raise PortfolioRemoteProcessingError(
                "Core v9 query count differs from owner authorization"
            )
        if (
            authorization.query_referenced_unique_asset_count
            != query_bindings.unique_asset_count
        ):
            raise PortfolioRemoteProcessingError(
                "Core v9 unique query asset count differs from owner authorization"
            )
    return VerifiedPortfolioRemoteProcessingRuntime(
        authorization=authorization,
        receipt=receipt,
        processor=processor,
        catalog=output,
        authorization_file_sha256=authorization_file_sha256,
        receipt_file_sha256=_require_sha256(
            expected_receipt_file_sha256,
            "receipt file SHA-256",
        ),
        dataset_assets_sha256=dataset_assets_sha256,
        plan_sha256=plan_sha256,
        query_artifact_sha256=query_bindings.query_artifact_sha256,
        _verification_token=_VERIFIED_RUNTIME_TOKEN,
    )


def require_verified_portfolio_remote_processing_runtime(
    runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    processor: PortfolioProcessor,
    catalog_sha256: str,
) -> VerifiedPortfolioRemoteProcessingRuntime:
    """Reject forged or mismatched Portfolio runtime preflight handles."""

    if (
        type(runtime) is not VerifiedPortfolioRemoteProcessingRuntime
        or runtime._verification_token is not _VERIFIED_RUNTIME_TOKEN
    ):
        raise PortfolioRemoteProcessingError(
            "remote execution requires a verified Portfolio runtime preflight"
        )
    if runtime.processor != processor:
        raise PortfolioRemoteProcessingError("runtime processor scope mismatch")
    if runtime.catalog.catalog_sha256 != _require_sha256(
        catalog_sha256,
        "runtime catalog SHA-256",
    ):
        raise PortfolioRemoteProcessingError("runtime catalog SHA-256 mismatch")
    return runtime


def _selection_bindings(
    value: object,
) -> _SelectionBindings:
    if not isinstance(value, dict):
        raise PortfolioRemoteProcessingError("selection manifest must be an object")
    selections = value.get("selections")
    dataset_assets = value.get("dataset_assets")
    if not isinstance(selections, list) or not selections:
        raise PortfolioRemoteProcessingError(
            "selection manifest lacks selections or dataset-assets binding"
        )
    if isinstance(dataset_assets, dict):
        required = {
            "cloud_upload_allowed",
            "image_path",
            "image_sha256",
            "public_demo_allowed",
            "source_artifact_sha256",
            "source_dataset",
            "source_revision",
        }
        normalized: list[dict[str, object]] = []
        for row in selections:
            if not isinstance(row, dict) or not required.issubset(row):
                raise PortfolioRemoteProcessingError(
                    "selection row schema is incomplete"
                )
            _require_sha256(row["image_sha256"], "selection image SHA-256")
            _require_sha256(
                row["source_artifact_sha256"],
                "selection source artifact SHA-256",
            )
            normalized.append(row)
        declared_rows = dataset_assets.get("rows")
        if type(declared_rows) is not int or declared_rows != len(normalized):
            raise PortfolioRemoteProcessingError(
                "dataset-assets descriptor or row count differs from selection"
            )
        dataset_assets_path = dataset_assets.get("path")
        if not isinstance(dataset_assets_path, str) or not dataset_assets_path:
            raise PortfolioRemoteProcessingError(
                "selection manifest lacks dataset-assets descriptor"
            )
        return _SelectionBindings(
            rows=tuple(normalized),
            dataset_assets_sha256=_require_sha256(
                dataset_assets.get("sha256"),
                "dataset-assets SHA-256",
            ),
            dataset_asset_count=len(normalized),
            manifest_kind="mini",
            dataset_assets_path=dataset_assets_path,
        )

    dataset_assets_sha256 = value.get("dataset_assets_sha256")
    dataset_asset_count = value.get("dataset_asset_count")
    if (
        not isinstance(dataset_assets_sha256, str)
        or type(dataset_asset_count) is not int
        or dataset_asset_count <= 0
        or dataset_asset_count != len(selections)
    ):
        raise PortfolioRemoteProcessingError(
            "Core v9 selection manifest lacks dataset-assets binding"
        )
    normalized = []
    drafts: list[dict[str, object]] = []
    required_core = {
        "candidate_sha256",
        "destination_path",
        "draft",
        "expected_sha256",
        "source_id",
    }
    required_draft = {
        "cloud_upload_allowed",
        "local_path",
        "public_demo_allowed",
        "source_dataset",
        "source_revision",
    }
    for row in selections:
        if not isinstance(row, dict) or not required_core.issubset(row):
            raise PortfolioRemoteProcessingError(
                "Core v9 selection row schema is incomplete"
            )
        draft = row["draft"]
        if not isinstance(draft, dict) or not required_draft.issubset(draft):
            raise PortfolioRemoteProcessingError(
                "Core v9 selection draft schema is incomplete"
            )
        destination_path = row["destination_path"]
        source_id = row["source_id"]
        if (
            not isinstance(destination_path, str)
            or not destination_path
            or destination_path != draft["local_path"]
            or not isinstance(source_id, str)
            or not source_id
            or source_id != draft["source_dataset"]
        ):
            raise PortfolioRemoteProcessingError(
                "Core v9 selection row/draft binding differs"
            )
        image_sha256 = _require_sha256(
            row["expected_sha256"],
            "Core v9 expected image SHA-256",
        )
        _require_sha256(
            row["candidate_sha256"],
            "Core v9 candidate SHA-256",
        )
        normalized.append(
            {
                "cloud_upload_allowed": draft["cloud_upload_allowed"],
                "image_path": destination_path,
                "image_sha256": image_sha256,
                "public_demo_allowed": draft["public_demo_allowed"],
                "source_artifact_sha256": None,
                "source_dataset": source_id,
                "source_revision": draft["source_revision"],
            }
        )
        drafts.append(draft)
    return _SelectionBindings(
        rows=tuple(normalized),
        dataset_assets_sha256=_require_sha256(
            dataset_assets_sha256,
            "dataset-assets SHA-256",
        ),
        dataset_asset_count=dataset_asset_count,
        manifest_kind="core-v9",
        dataset_asset_drafts=tuple(drafts),
    )


def _verify_selection_scope(
    authorization: PortfolioRemoteProcessingAuthorization,
    selection: _SelectionBindings,
) -> None:
    expected_kind = (
        "core-v9" if authorization.scope == "core-v9-all-catalog-assets" else "mini"
    )
    if selection.manifest_kind != expected_kind:
        raise PortfolioRemoteProcessingError(
            "selection manifest schema differs from authorization scope"
        )


def _verify_authorized_catalog(
    authorization: PortfolioRemoteProcessingAuthorization,
    catalog: AssetCatalog,
) -> None:
    if authorization.scope != "core-v9-all-catalog-assets":
        return
    if authorization.authorized_asset_count != len(catalog.assets):
        raise PortfolioRemoteProcessingError(
            "Core v9 catalog asset count differs from owner authorization"
        )
    if authorization.base_catalog_asset_file_sha256 != catalog.manifest.assets.sha256:
        raise PortfolioRemoteProcessingError(
            "Core v9 catalog asset file differs from owner authorization"
        )
    if (
        authorization.base_catalog_components_file_sha256
        != catalog.manifest.components.sha256
    ):
        raise PortfolioRemoteProcessingError(
            "Core v9 catalog components file differs from owner authorization"
        )
    if authorization.base_catalog_permission_counts != _permission_counts(
        catalog.assets
    ):
        raise PortfolioRemoteProcessingError(
            "Core v9 base permissions differ from owner authorization"
        )
    if catalog.manifest.public_demo_allowed_count != 0:
        raise PortfolioRemoteProcessingError(
            "Core v9 owner authorization requires public_demo_allowed=0"
        )


def _verify_authorized_selection(
    *,
    authorization: PortfolioRemoteProcessingAuthorization,
    selections: tuple[dict[str, object], ...],
    catalog: AssetCatalog,
) -> Counter[str]:
    if len(selections) != len(catalog.assets):
        raise PortfolioRemoteProcessingError(
            "selection and catalog asset counts differ"
        )
    decisions = {item.source_id: item for item in authorization.source_decisions}
    source_counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    for row in selections:
        path = row["image_path"]
        source_id = row["source_dataset"]
        if not isinstance(path, str) or not isinstance(source_id, str):
            raise PortfolioRemoteProcessingError("selection binding fields are invalid")
        if path in seen_paths:
            raise PortfolioRemoteProcessingError("selection image path is duplicated")
        seen_paths.add(path)
        decision = decisions.get(source_id)
        if decision is None:
            raise PortfolioRemoteProcessingError(
                f"selection source is not owner-authorized: {source_id}"
            )
        if (
            row["source_revision"] != decision.source_revision
            or (
                decision.source_artifact_sha256 is not None
                and row["source_artifact_sha256"] != decision.source_artifact_sha256
            )
            or row["public_demo_allowed"] is not False
        ):
            raise PortfolioRemoteProcessingError(
                f"selection source binding differs from authorization: {source_id}"
            )
        resolution = catalog.resolve_path(path)
        asset = resolution.asset
        if (
            asset.source_dataset != source_id
            or asset.source_revision != decision.source_revision
            or asset.sha256 != row["image_sha256"]
            or asset.cloud_upload_allowed != row["cloud_upload_allowed"]
            or asset.public_demo_allowed is not False
        ):
            raise PortfolioRemoteProcessingError(
                f"selection/catalog binding drifted: {path}"
            )
        source_counts[source_id] += 1
    if set(seen_paths) != {asset.local_path for asset in catalog.assets}:
        raise PortfolioRemoteProcessingError(
            "selection does not exactly cover the base catalog"
        )
    expected_counts = {
        item.source_id: item.asset_count for item in authorization.source_decisions
    }
    if dict(source_counts) != expected_counts:
        raise PortfolioRemoteProcessingError(
            "authorized source asset counts differ from selection"
        )
    return source_counts


def _verify_permission_only_transition(
    base: AssetCatalog,
    output: AssetCatalog,
) -> None:
    if len(base.assets) != len(output.assets):
        raise PortfolioRemoteProcessingError("output catalog asset count changed")
    base_by_id = {item.asset_id: item for item in base.assets}
    output_by_id = {item.asset_id: item for item in output.assets}
    if set(base_by_id) != set(output_by_id):
        raise PortfolioRemoteProcessingError("asset identities changed")
    for asset_id, before in base_by_id.items():
        after = output_by_id[asset_id]
        before_payload = before.model_dump(
            mode="json",
            exclude={"cloud_upload_allowed"},
        )
        after_payload = after.model_dump(
            mode="json",
            exclude={"cloud_upload_allowed"},
        )
        if before_payload != after_payload:
            raise PortfolioRemoteProcessingError(
                f"non-permission asset field changed: {asset_id}"
            )
        if after.cloud_upload_allowed is not True:
            raise PortfolioRemoteProcessingError(
                f"output asset is not cloud-enabled: {asset_id}"
            )
    if base.components != output.components:
        raise PortfolioRemoteProcessingError("leakage components changed")
    if output.manifest.unknown_cloud_permission_count != 0:
        raise PortfolioRemoteProcessingError(
            "output catalog retains unknown cloud permissions"
        )
    if output.manifest.public_demo_allowed_count != 0:
        raise PortfolioRemoteProcessingError(
            "output catalog unexpectedly permits public demo"
        )


def _permission_counts(assets: tuple[DatasetAsset, ...]) -> dict[str, int]:
    counts = Counter(
        "unknown"
        if item.cloud_upload_allowed is None
        else "true"
        if item.cloud_upload_allowed
        else "false"
        for item in assets
    )
    return {
        "false": counts["false"],
        "true": counts["true"],
        "unknown": counts["unknown"],
    }


def _asset_id_set_sha256(assets: tuple[DatasetAsset, ...]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(item.asset_id for item in assets)))


def _image_binding_set_sha256(assets: tuple[DatasetAsset, ...]) -> str:
    bindings = sorted(
        (
            {
                "asset_id": item.asset_id,
                "local_path": item.local_path,
                "sha256": item.sha256,
            }
            for item in assets
        ),
        key=lambda value: value["asset_id"],
    )
    return sha256_bytes(canonical_json_bytes(bindings))


def _self_hash(model: BaseModel, field_name: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(model.model_dump(mode="json", exclude={field_name}))
    )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PortfolioRemoteProcessingError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


__all__ = [
    "PortfolioRemoteProcessingAuthorization",
    "PortfolioRemoteProcessingError",
    "PortfolioRemoteProcessingReceipt",
    "PortfolioRemoteProcessingResult",
    "PortfolioQueryBindingResult",
    "PortfolioProcessor",
    "PortfolioSourceRemoteDecision",
    "VerifiedPortfolioRemoteProcessingRuntime",
    "load_portfolio_remote_processing_authorization",
    "load_portfolio_remote_processing_receipt",
    "publish_remote_authorized_portfolio_catalog",
    "require_verified_portfolio_remote_processing_runtime",
    "verify_portfolio_remote_processing_runtime",
    "verify_remote_authorized_query_bindings",
]
