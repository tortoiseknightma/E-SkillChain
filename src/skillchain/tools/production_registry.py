"""Production assembly for an externally locked, authority-issued tool registry.

The in-process registry authority in :mod:`skillchain.tools.registry` proves that
the canonical reviewed service graph created one registry instance.  It does not
by itself prove that the paths and digests supplied by an operator were approved
outside that process.  This module closes that second boundary with two files:

* an artifact lock pins every data/model input needed to assemble the services;
* a runtime lock pins one exact authority-issued registry-generation snapshot.

Both files must themselves be supplied with an expected SHA-256 from an external
trust root.  A freshly calculated digest can be emitted as a candidate, but it
cannot authorize its own use.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.data.kb_catalog import load_kb_catalog
from skillchain.tools.document_ocr import (
    DocumentOCRService,
    RapidOCROnnxBackend,
)
from skillchain.tools.document_safety import (
    DocumentSafetyCatalog,
    load_document_safety_catalog,
)
from skillchain.tools.embedding import (
    DashScopeEmbeddingClient,
    FormalEmbeddingBackend,
    OpenCLIPEmbeddingBackend,
)
from skillchain.tools.kb_index import load_kb_bundle
from skillchain.tools.kb_lookup import KBLookupService
from skillchain.tools.model_artifacts import load_model_artifact_manifest
from skillchain.tools.multi_product import MultiProductSearchService
from skillchain.tools.object_detect import (
    ObjectDetectionService,
    UltralyticsDetectorBackend,
)
from skillchain.tools.product_index import ProductIndex
from skillchain.tools.product_search import ProductSearchService
from skillchain.tools.registry import (
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    FormalRegistryRuntimeHandle,
    FormalRegistryRuntimeSnapshot,
    MVPToolServices,
    RegistryError,
    ToolRegistry,
    build_mvp_registry,
    build_mvp_registry_spec,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_LOCK_MAX_BYTES = 128 * 1024
_RELATIVE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class FormalRegistryConfigurationError(ValueError):
    """A production registry lock or deployment path is not trustworthy."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LockedPath(_StrictFrozenModel):
    """One canonical path below the separately supplied artifact root."""

    path: str
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class FormalRegistryArtifactLock(_StrictFrozenModel):
    """External identities needed before constructing the real services."""

    schema_version: Literal[2] = 2
    policy_version: Literal["formal-registry-artifacts-v2"] = (
        "formal-registry-artifacts-v2"
    )
    tool_spec_registry_sha256: Sha256
    embedding_provider: Literal["dashscope", "open_clip"]
    embedding_model: str
    embedding_manifest: LockedPath | None = None
    product_index: LockedPath
    product_query_artifact: LockedPath
    kb_catalog: LockedPath
    kb_index: LockedPath
    detector_manifest: LockedPath
    ocr_manifest: LockedPath
    asset_catalog_path: str
    asset_root_path: str
    asset_catalog_sha256: Sha256
    document_safety_catalog_path: str
    document_safety_catalog_sha256: Sha256
    document_safety_review_ledger_sha256: Sha256

    @field_validator(
        "asset_catalog_path",
        "asset_root_path",
        "document_safety_catalog_path",
    )
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_embedding_track(self) -> Self:
        if self.embedding_provider == "dashscope":
            if self.embedding_model != "qwen3-vl-embedding":
                raise ValueError("DashScope embedding model is not canonical")
            if self.embedding_manifest is not None:
                raise ValueError(
                    "DashScope embedding must not carry a local model manifest"
                )
        else:
            if self.embedding_model != (
                "open-clip:xlm-roberta-large-ViT-H-14:frozen_laion5b_s13b_b90k"
            ):
                raise ValueError("OpenCLIP embedding model is not canonical")
            if self.embedding_manifest is None:
                raise ValueError("OpenCLIP embedding requires a locked model manifest")
        return self


class FormalRegistryRuntimeLock(_StrictFrozenModel):
    """Exact snapshot produced by the canonical in-process registry authority."""

    schema_version: Literal[1, 2] = 1
    policy_version: Literal[
        "formal-registry-runtime-v1",
        "formal-registry-runtime-v2",
    ] = "formal-registry-runtime-v1"
    artifact_lock_file_sha256: Sha256
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    tool_runtime_bindings: tuple["RuntimeBindingLock", ...] = Field(
        min_length=7,
        max_length=8,
    )
    evidence_runtime_bindings: tuple["EvidenceRuntimeBindingLock", ...] = Field(
        min_length=7,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_tool_sets(self) -> Self:
        expected_policy = f"formal-registry-runtime-v{self.schema_version}"
        if self.policy_version != expected_policy:
            raise ValueError("runtime lock policy does not match its schema generation")
        expected_names = (
            MVP_TOOL_NAMES if self.schema_version == 1 else MVP_TOOL_NAMES_V2
        )
        expected = tuple(sorted(expected_names))
        tool_names = tuple(item.name for item in self.tool_runtime_bindings)
        evidence_names = tuple(item.name for item in self.evidence_runtime_bindings)
        if tool_names != expected:
            raise ValueError(
                "runtime lock tool set does not match its schema generation"
            )
        if evidence_names != expected:
            raise ValueError(
                "runtime lock evidence set does not match its schema generation"
            )
        evidence = {item.name: item.sha256 for item in self.evidence_runtime_bindings}
        populated_evidence = {"object_detect", "document_ocr"}
        if self.schema_version == 2:
            populated_evidence.add("multi_product_search")
        for name in expected:
            if (name in populated_evidence) != (evidence[name] is not None):
                raise ValueError(
                    "runtime evidence population does not match its schema generation"
                )
        return self

    @classmethod
    def from_snapshot(
        cls,
        snapshot: FormalRegistryRuntimeSnapshot,
        *,
        artifact_lock_file_sha256: str,
    ) -> "FormalRegistryRuntimeLock":
        tool_names = frozenset(name for name, _digest in snapshot.tool_runtime_bindings)
        evidence_names = frozenset(
            name for name, _digest in snapshot.evidence_runtime_bindings
        )
        if tool_names == MVP_TOOL_NAMES and evidence_names == MVP_TOOL_NAMES:
            schema_version = 1
        elif tool_names == MVP_TOOL_NAMES_V2 and evidence_names == MVP_TOOL_NAMES_V2:
            schema_version = 2
        else:
            raise ValueError(
                "formal runtime snapshot is not a supported registry generation"
            )
        return cls.model_validate(
            {
                "schema_version": schema_version,
                "policy_version": f"formal-registry-runtime-v{schema_version}",
                "artifact_lock_file_sha256": artifact_lock_file_sha256,
                "registry_sha256": snapshot.registry_sha256,
                "registry_runtime_sha256": snapshot.registry_runtime_sha256,
                "tool_runtime_bindings": tuple(
                    RuntimeBindingLock(name=name, sha256=digest)
                    for name, digest in snapshot.tool_runtime_bindings
                ),
                "evidence_runtime_bindings": tuple(
                    EvidenceRuntimeBindingLock(name=name, sha256=digest)
                    for name, digest in snapshot.evidence_runtime_bindings
                ),
            },
            strict=True,
        )

    def snapshot(self) -> FormalRegistryRuntimeSnapshot:
        return FormalRegistryRuntimeSnapshot(
            registry_sha256=self.registry_sha256,
            registry_runtime_sha256=self.registry_runtime_sha256,
            tool_runtime_bindings=tuple(
                (item.name, item.sha256) for item in self.tool_runtime_bindings
            ),
            evidence_runtime_bindings=tuple(
                (item.name, item.sha256) for item in self.evidence_runtime_bindings
            ),
        )


class RuntimeBindingLock(_StrictFrozenModel):
    name: str = Field(min_length=1)
    sha256: Sha256


class EvidenceRuntimeBindingLock(_StrictFrozenModel):
    name: str = Field(min_length=1)
    sha256: Sha256 | None


@dataclass(frozen=True)
class AuthorityIssuedFormalRegistry:
    """A live registry plus the authority capability and externally locked state."""

    registry: ToolRegistry
    handle: FormalRegistryRuntimeHandle
    snapshot: FormalRegistryRuntimeSnapshot
    asset_catalog: AssetCatalog
    document_safety_catalog: DocumentSafetyCatalog
    product_search: ProductSearchService
    object_detection: ObjectDetectionService
    document_ocr: DocumentOCRService
    multi_product: MultiProductSearchService
    artifact_lock_file_sha256: str
    runtime_lock_file_sha256: str | None


@dataclass(frozen=True)
class _LoadedFormalServices:
    asset_catalog: AssetCatalog
    document_safety_catalog: DocumentSafetyCatalog
    product_search: ProductSearchService
    kb_lookup: KBLookupService
    object_detection: ObjectDetectionService
    document_ocr: DocumentOCRService


def load_formal_registry_artifact_lock(
    path: str | Path,
    *,
    expected_lock_file_sha256: str,
) -> tuple[FormalRegistryArtifactLock, str]:
    """Load a canonical artifact lock only through an external file digest."""

    return _load_lock(
        path,
        FormalRegistryArtifactLock,
        expected_lock_file_sha256=expected_lock_file_sha256,
        label="formal registry artifact lock",
    )


def load_formal_registry_runtime_lock(
    path: str | Path,
    *,
    expected_lock_file_sha256: str,
) -> tuple[FormalRegistryRuntimeLock, str]:
    """Load a canonical runtime lock only through an external file digest."""

    return _load_lock(
        path,
        FormalRegistryRuntimeLock,
        expected_lock_file_sha256=expected_lock_file_sha256,
        label="formal registry runtime lock",
    )


def build_formal_registry_runtime_candidate(
    artifact_lock_path: str | Path,
    artifact_root: str | Path,
    *,
    expected_artifact_lock_file_sha256: str,
    api_key: str | None = None,
) -> FormalRegistryRuntimeLock:
    """Assemble a candidate lock without exposing its unapproved live registry.

    This performs the paid live embedding canary and eagerly loads the local
    detector/OCR runtimes.  Authority issuance is needed to capture the
    candidate snapshot, but that live registry is deliberately discarded.  A
    usable registry can only be returned later by
    :func:`load_authority_issued_formal_registry` with an externally pinned
    runtime lock.
    """

    artifact_lock, artifact_lock_sha256 = load_formal_registry_artifact_lock(
        artifact_lock_path,
        expected_lock_file_sha256=expected_artifact_lock_file_sha256,
    )
    issued = _assemble_authority_issued_registry(
        artifact_lock,
        artifact_root,
        artifact_lock_file_sha256=artifact_lock_sha256,
        api_key=api_key,
        runtime_lock_file_sha256=None,
    )
    candidate = FormalRegistryRuntimeLock.from_snapshot(
        issued.snapshot,
        artifact_lock_file_sha256=artifact_lock_sha256,
    )
    return candidate


def publish_formal_registry_runtime_candidate(
    artifact_lock_path: str | Path,
    artifact_root: str | Path,
    destination: str | Path,
    *,
    expected_artifact_lock_file_sha256: str,
    api_key: str | None = None,
) -> FormalRegistryRuntimeLock:
    """Create, never replace, a candidate runtime lock for external approval."""

    candidate = build_formal_registry_runtime_candidate(
        artifact_lock_path,
        artifact_root,
        expected_artifact_lock_file_sha256=expected_artifact_lock_file_sha256,
        api_key=api_key,
    )
    _atomic_create_file(
        destination,
        canonical_json_bytes(candidate.model_dump(mode="json")),
    )
    return candidate


def load_authority_issued_formal_registry(
    artifact_lock_path: str | Path,
    runtime_lock_path: str | Path,
    artifact_root: str | Path,
    *,
    expected_artifact_lock_file_sha256: str,
    expected_runtime_lock_file_sha256: str,
    api_key: str | None = None,
) -> AuthorityIssuedFormalRegistry:
    """Rebuild and verify one production-eligible registry generation."""

    artifact_lock, artifact_lock_sha256 = load_formal_registry_artifact_lock(
        artifact_lock_path,
        expected_lock_file_sha256=expected_artifact_lock_file_sha256,
    )
    runtime_lock, runtime_lock_sha256 = load_formal_registry_runtime_lock(
        runtime_lock_path,
        expected_lock_file_sha256=expected_runtime_lock_file_sha256,
    )
    if runtime_lock.artifact_lock_file_sha256 != artifact_lock_sha256:
        raise FormalRegistryConfigurationError(
            "runtime lock does not bind the supplied artifact lock"
        )
    issued = _assemble_authority_issued_registry(
        artifact_lock,
        artifact_root,
        artifact_lock_file_sha256=artifact_lock_sha256,
        api_key=api_key,
        runtime_lock_file_sha256=runtime_lock_sha256,
    )
    expected_snapshot = runtime_lock.snapshot()
    if issued.snapshot != expected_snapshot:
        raise FormalRegistryConfigurationError(
            "authority-issued registry does not match the external runtime lock"
        )
    issued.handle.verify(expected_snapshot)
    return issued


def build_formal_runtime_context_from_env():
    """Runtime factory for ``formal_evaluation_cli`` without hard-coded paths."""

    from skillchain.tools.formal_evaluation_cli import FormalRuntimeContext

    required = {
        "artifact_lock_path": "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK",
        "runtime_lock_path": "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK",
        "artifact_root": "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_ROOT",
        "artifact_lock_sha256": "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK_SHA256",
        "runtime_lock_sha256": "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK_SHA256",
    }
    values: dict[str, str] = {}
    missing: list[str] = []
    for name, variable in required.items():
        value = os.environ.get(variable)
        if value is None or not value.strip():
            missing.append(variable)
        else:
            values[name] = value
    if missing:
        raise FormalRegistryConfigurationError(
            "formal registry environment is incomplete: " + ", ".join(sorted(missing))
        )
    issued = load_authority_issued_formal_registry(
        values["artifact_lock_path"],
        values["runtime_lock_path"],
        values["artifact_root"],
        expected_artifact_lock_file_sha256=values["artifact_lock_sha256"],
        expected_runtime_lock_file_sha256=values["runtime_lock_sha256"],
    )
    registry_schema_version = issued.registry.manifest.schema_version
    if registry_schema_version == 1:
        multi_product_executor = issued.multi_product
    elif registry_schema_version == 2:
        # Registry v2 owns multi-product execution through its canonical
        # ``multi_product_search`` ToolSpec.  Supplying the legacy out-of-band
        # executor would create a second runtime authority and is rejected by
        # the formal evaluator as an unused multi-product runtime.
        multi_product_executor = None
    else:  # pragma: no cover - guarded by the authority-issued loader.
        raise FormalRegistryConfigurationError(
            "formal runtime context received an unsupported registry generation"
        )
    return FormalRuntimeContext(
        catalog=issued.asset_catalog,
        registry=issued.registry,
        document_safety_catalog=issued.document_safety_catalog,
        multi_product_executor=multi_product_executor,
    )


def _assemble_authority_issued_registry(
    artifact_lock: FormalRegistryArtifactLock,
    artifact_root: str | Path,
    *,
    artifact_lock_file_sha256: str,
    api_key: str | None,
    runtime_lock_file_sha256: str | None,
) -> AuthorityIssuedFormalRegistry:
    root = _validated_artifact_root(artifact_root)
    services = _load_locked_services(artifact_lock, root, api_key=api_key)
    legacy_registry_sha256 = build_mvp_registry_spec().registry_sha256
    v2_registry_sha256 = build_mvp_registry_spec(
        include_multi_product=True
    ).registry_sha256
    if artifact_lock.tool_spec_registry_sha256 == legacy_registry_sha256:
        include_multi_product = False
    elif artifact_lock.tool_spec_registry_sha256 == v2_registry_sha256:
        include_multi_product = True
    else:
        raise FormalRegistryConfigurationError(
            "locked tool specs are not a supported registry generation"
        )
    registry = build_mvp_registry(
        MVPToolServices(
            product_search=services.product_search,
            kb_lookup=services.kb_lookup,
            object_detection=services.object_detection,
            document_ocr=services.document_ocr,
            safety_approval_for=services.document_safety_catalog.approval_for,
        ),
        include_multi_product=include_multi_product,
    )
    if registry.registry_sha256 != artifact_lock.tool_spec_registry_sha256:
        raise FormalRegistryConfigurationError(
            "live tool specs do not match the artifact lock"
        )
    try:
        handle = registry.require_formal_runtime()
        snapshot = handle.snapshot()
    except RegistryError as error:
        raise FormalRegistryConfigurationError(
            "canonical registry authority refused the live service graph"
        ) from error
    if snapshot.registry_sha256 != artifact_lock.tool_spec_registry_sha256:
        raise FormalRegistryConfigurationError(
            "authority snapshot changed the locked registry identity"
        )
    multi_product = MultiProductSearchService(
        services.object_detection,
        services.product_search,
    )
    return AuthorityIssuedFormalRegistry(
        registry=registry,
        handle=handle,
        snapshot=snapshot,
        asset_catalog=services.asset_catalog,
        document_safety_catalog=services.document_safety_catalog,
        product_search=services.product_search,
        object_detection=services.object_detection,
        document_ocr=services.document_ocr,
        multi_product=multi_product,
        artifact_lock_file_sha256=artifact_lock_file_sha256,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
    )


def _load_locked_services(
    lock: FormalRegistryArtifactLock,
    root: Path,
    *,
    api_key: str | None,
) -> _LoadedFormalServices:
    asset_root = _locked_path(root, lock.asset_root_path)
    asset_catalog = load_asset_catalog(
        _locked_path(root, lock.asset_catalog_path),
        asset_root,
        verify_files=True,
    )
    if asset_catalog.catalog_sha256 != lock.asset_catalog_sha256:
        raise FormalRegistryConfigurationError(
            "asset catalog does not match the external artifact lock"
        )

    query_path = _locked_file(root, lock.product_query_artifact)
    product_index = ProductIndex.load(
        _locked_path(root, lock.product_index.path),
        expected_manifest_sha256=lock.product_index.sha256,
    )
    eligibility = product_index.manifest.get("query_gallery_eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("asset_catalog_sha256") != asset_catalog.catalog_sha256
    ):
        raise FormalRegistryConfigurationError(
            "product index and runtime AssetCatalog are not the same locked catalog"
        )
    if lock.embedding_provider == "dashscope":
        client = DashScopeEmbeddingClient(api_key=api_key)
    else:
        assert lock.embedding_manifest is not None
        embedding_artifact = load_model_artifact_manifest(
            _locked_path(root, lock.embedding_manifest.path),
            expected_kind="multimodal_embedding",
            expected_manifest_sha256=lock.embedding_manifest.sha256,
            verify_files=True,
        )
        client = OpenCLIPEmbeddingBackend(embedding_artifact)
        if client.model != lock.embedding_model:
            raise FormalRegistryConfigurationError(
                "local embedding model does not match the artifact lock"
            )
    embedding = FormalEmbeddingBackend(
        client,
        canary_text=product_index.canary_text,
        canary_vector=product_index.canary_vector,
    )
    embedding.verify_canary(force=True)
    product_search = ProductSearchService(
        product_index,
        embedding,
        query_artifact=query_path,
    )

    kb_catalog = load_kb_catalog(
        _locked_path(root, lock.kb_catalog.path),
        expected_catalog_sha256=lock.kb_catalog.sha256,
    )
    kb_bundle = load_kb_bundle(
        _locked_path(root, lock.kb_index.path),
        catalog=kb_catalog,
        expected_bundle_sha256=lock.kb_index.sha256,
    )
    kb_lookup = KBLookupService(kb_bundle)

    detector_artifact = load_model_artifact_manifest(
        _locked_path(root, lock.detector_manifest.path),
        expected_kind="object_detector",
        expected_manifest_sha256=lock.detector_manifest.sha256,
        verify_files=True,
    )
    detector_backend = UltralyticsDetectorBackend(detector_artifact)
    # Loading, not merely hashing, catches missing packages and unreadable model
    # formats before an authority handle can be issued.
    detector_backend._load_model()
    object_detection = ObjectDetectionService(detector_artifact, detector_backend)

    ocr_artifact = load_model_artifact_manifest(
        _locked_path(root, lock.ocr_manifest.path),
        expected_kind="document_ocr",
        expected_manifest_sha256=lock.ocr_manifest.sha256,
        verify_files=True,
    )
    ocr_backend = RapidOCROnnxBackend(ocr_artifact)
    document_ocr = DocumentOCRService(ocr_artifact, ocr_backend)
    ocr_backend._load_engine(document_ocr.config)

    safety_catalog = load_document_safety_catalog(
        _locked_path(root, lock.document_safety_catalog_path),
        asset_catalog,
        expected_review_ledger_sha256=(lock.document_safety_review_ledger_sha256),
    )
    if safety_catalog.catalog_sha256 != lock.document_safety_catalog_sha256:
        raise FormalRegistryConfigurationError(
            "document safety catalog does not match the external artifact lock"
        )
    return _LoadedFormalServices(
        asset_catalog=asset_catalog,
        document_safety_catalog=safety_catalog,
        product_search=product_search,
        kb_lookup=kb_lookup,
        object_detection=object_detection,
        document_ocr=document_ocr,
    )


def _locked_file(root: Path, locked: LockedPath) -> Path:
    path = _locked_path(root, locked.path)
    content = read_stable_regular_file(
        path,
        label=f"locked artifact {locked.path}",
    )
    if sha256_bytes(content) != locked.sha256:
        raise FormalRegistryConfigurationError(
            f"locked artifact digest mismatch: {locked.path}"
        )
    return path


def _locked_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise FormalRegistryConfigurationError(
                f"locked artifact path is unavailable: {relative}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalRegistryConfigurationError(
                f"locked artifact path contains a symlink: {relative}"
            )
    return current


def _validated_artifact_root(value: str | Path) -> Path:
    root = Path(value).absolute()
    try:
        metadata = root.lstat()
    except OSError as error:
        raise FormalRegistryConfigurationError(
            "formal registry artifact root is unavailable"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FormalRegistryConfigurationError(
            "formal registry artifact root must be a non-symlink directory"
        )
    return root


def _validate_relative_path(value: str) -> str:
    if not value or value != value.strip() or "\\" in value:
        raise ValueError("artifact path must be canonical POSIX relative syntax")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError("artifact path must stay below the artifact root")
    if path.as_posix() != value or any(
        _RELATIVE_COMPONENT.fullmatch(part) is None or part.endswith((".", " "))
        for part in path.parts
    ):
        raise ValueError("artifact path contains an unsafe component")
    return value


def _load_lock(
    path: str | Path,
    model_type,
    *,
    expected_lock_file_sha256: str,
    label: str,
):
    if re.fullmatch(r"[0-9a-f]{64}", expected_lock_file_sha256) is None:
        raise FormalRegistryConfigurationError(
            f"{label} expected SHA-256 must be 64 lowercase hex characters"
        )
    try:
        content = read_stable_regular_file(
            path,
            label=label,
            max_bytes=_LOCK_MAX_BYTES,
        )
        if sha256_bytes(content) != expected_lock_file_sha256:
            raise FormalRegistryConfigurationError(
                f"{label} does not match the external expected SHA-256"
            )
        value = parse_canonical_json(content, label=label)
        if not isinstance(value, dict):
            raise FormalRegistryConfigurationError(f"{label} must contain an object")
        parsed = model_type.model_validate_json(content, strict=True)
    except FormalRegistryConfigurationError:
        raise
    except (ArtifactFormatError, ValueError) as error:
        raise FormalRegistryConfigurationError(f"{label} is invalid") from error
    return parsed, sha256_bytes(content)


def _atomic_create_file(path: str | Path, content: bytes) -> Path:
    """Create one file atomically without ever replacing an existing target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"target already exists; refusing overwrite: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
            temporary = Path(target.name)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"target already exists; refusing overwrite: {path}")
        # os.link gives create-only publication semantics on the same volume.
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "AuthorityIssuedFormalRegistry",
    "FormalRegistryArtifactLock",
    "FormalRegistryConfigurationError",
    "FormalRegistryRuntimeLock",
    "LockedPath",
    "build_formal_registry_runtime_candidate",
    "build_formal_runtime_context_from_env",
    "load_authority_issued_formal_registry",
    "load_formal_registry_artifact_lock",
    "load_formal_registry_runtime_lock",
    "publish_formal_registry_runtime_candidate",
]
