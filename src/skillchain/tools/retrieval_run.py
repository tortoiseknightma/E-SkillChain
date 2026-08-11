"""Create-only, canonical evidence bundles for formal retrieval runs.

The low-level search services accept filesystem paths for convenient local use.
Formal experiments must not: an audited query collection is not proof that the
path supplied to one particular call came from that collection.  This module
therefore resolves every image from an authoritative schema-v2 ``Query`` row
and records one immutable call envelope per query.

This is deliberately an evidence boundary, not an evaluator.  Downstream
evaluation code can require :class:`VerifiedRetrievalRun` and thereby refuse
bare hit lists or arbitrary prediction files.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Annotated, Any, Literal, Protocol

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
from skillchain.data.gallery_eligibility import load_query_artifact
from skillchain.schemas import Query
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.tools.contracts import (
    JSONValue,
    ProductSearchTrace,
    RetrievalArtifactBinding,
    validate_json_value,
)
from skillchain.tools.registry import ToolExecutionContext, ToolRegistry

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

RETRIEVAL_RUN_POLICY_VERSION = "retrieval-run-v2"
_SPEC_FILE = "spec.json"
_EXECUTIONS_FILE = "query-executions.jsonl"
_CALLS_FILE = "tool-calls.jsonl"
_MANIFEST_FILE = "manifest.json"
_ARTIFACT_FILES = (_SPEC_FILE, _EXECUTIONS_FILE, _CALLS_FILE)
_CATALOG_FILES = ("manifest.json", "assets.jsonl", "components.jsonl")
_VERIFIED_MARKER = object()
_PUBLIC_ERROR_MESSAGES = {
    "context_violation": "tool context validation failed",
    "handler_failed": "tool execution failed",
    "invalid_arguments": "tool arguments are invalid",
    "invalid_output": "tool output is invalid",
    "safety_gate_failed": "tool safety approval is unavailable",
    "size_limit": "tool payload exceeds the size limit",
    "tool_execution_failed": "tool execution failed",
    "unknown_tool": "unknown tool",
}


class RetrievalRunError(ValueError):
    """A retrieval run or one of its transitive artifacts is invalid."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RunArtifactDescriptor(_FrozenStrictModel):
    path: str
    bytes: int = Field(ge=0)
    sha256: Sha256
    rows: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if value not in _ARTIFACT_FILES:
            raise ValueError("run artifact path is not allowed")
        return value


class RetrievalRunSpec(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["retrieval-run-spec"] = "retrieval-run-spec"
    policy_version: Literal["retrieval-run-v2"] = RETRIEVAL_RUN_POLICY_VERSION
    assurance: Literal["registry-verified", "provisional-executor"]
    run_id: str
    query_artifact_sha256: Sha256
    query_count: int = Field(gt=0)
    query_order_sha256: Sha256
    asset_catalog_sha256: Sha256
    artifact_binding: RetrievalArtifactBinding
    tool_name: str
    tool_spec_sha256: Sha256
    executor_id: str
    image_input_policy: Literal["authoritative-query-asset-v1"] = (
        "authoritative-query-asset-v1"
    )
    spec_sha256: Sha256

    @field_validator("run_id", "tool_name", "executor_id")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @model_validator(mode="after")
    def validate_binding(self):
        if self.artifact_binding.mode != "verified":
            raise ValueError("formal retrieval runs require a verified binding")
        if self.artifact_binding.query_artifact_sha256 != self.query_artifact_sha256:
            raise ValueError("binding/query artifact sha256 mismatch")
        if self.artifact_binding.asset_catalog_sha256 != self.asset_catalog_sha256:
            raise ValueError("binding/catalog sha256 mismatch")
        return self


class RetrievalToolCall(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    call_id: str
    query_id: str
    ordinal: int = Field(ge=0)
    tool_name: str
    tool_spec_sha256: Sha256
    arguments: dict[str, JSONValue]
    arguments_sha256: Sha256
    status: Literal["ok", "error"]
    output: JSONValue
    output_sha256: Sha256
    artifact_binding: RetrievalArtifactBinding
    call_sha256: Sha256

    @field_validator("call_id", "query_id", "tool_name")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @model_validator(mode="after")
    def validate_binding(self):
        if self.artifact_binding.mode != "verified":
            raise ValueError("formal retrieval calls require a verified binding")
        return self


class RetrievalQueryExecution(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    query_id: str
    query_ordinal: int = Field(ge=0)
    query_row_sha256: Sha256
    asset_id: str
    asset_sha256: Sha256
    call_ids: list[str]
    status: Literal["ok", "error"]
    execution_sha256: Sha256

    @field_validator("query_id", "asset_id")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("call_ids")
    @classmethod
    def validate_call_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("query execution must reference at least one call")
        if any(not call_id.strip() for call_id in value):
            raise ValueError("call_ids must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("call_ids must be unique")
        return value


class RetrievalRunManifest(_FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["retrieval-run"] = "retrieval-run"
    policy_version: Literal["retrieval-run-v2"] = RETRIEVAL_RUN_POLICY_VERSION
    run_id: str
    status: Literal["complete"] = "complete"
    spec_sha256: Sha256
    query_artifact_sha256: Sha256
    query_count: int = Field(gt=0)
    call_count: int = Field(gt=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    artifact_binding: RetrievalArtifactBinding
    artifacts: dict[str, RunArtifactDescriptor]
    manifest_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run_id must not be blank")
        return value

    @model_validator(mode="after")
    def validate_counts_and_artifacts(self):
        if self.artifact_binding.mode != "verified":
            raise ValueError("formal retrieval manifests require a verified binding")
        if self.success_count + self.error_count != self.query_count:
            raise ValueError("success/error counts must cover every query")
        if set(self.artifacts) != set(_ARTIFACT_FILES):
            raise ValueError("retrieval manifest artifact set is invalid")
        for name, descriptor in self.artifacts.items():
            if descriptor.path != name:
                raise ValueError("retrieval manifest artifact path mismatch")
        return self


@dataclass(frozen=True)
class RetrievalRequest:
    """One authoritative invocation supplied to a retrieval executor.

    ``image_path`` is resolved solely from ``query_id -> Query.asset_id ->
    AssetCatalog``.  No public run-builder argument can override it.  The
    persisted arguments intentionally contain logical identity rather than a
    machine-specific filesystem path.
    """

    query_id: str
    query_ordinal: int
    asset_id: str
    image_path: Path
    image_sha256: str
    cloud_upload_allowed: bool | None
    text: str
    tool_name: str
    artifact_binding: RetrievalArtifactBinding
    arguments: Mapping[str, JSONValue]


class RetrievalExecutor(Protocol):
    def __call__(self, request: RetrievalRequest) -> object:
        """Execute one retrieval call and return a JSON hit list."""


@dataclass(frozen=True)
class RegistryRetrievalExecutor:
    """Bind product retrieval to one live registry and runtime identity."""

    registry: ToolRegistry
    tool_name: Literal[
        "image_product_search",
        "text_product_search",
        "style_similar_search",
    ]
    product_search_service: object

    def __post_init__(self) -> None:
        from skillchain.tools.product_search import ProductSearchService

        if not isinstance(self.product_search_service, ProductSearchService):
            raise TypeError("product_search_service must be a ProductSearchService")
        if self.tool_name not in {spec.name for spec in self.registry.specs()}:
            raise RetrievalRunError("retrieval tool is absent from the registry")
        service_binding = self.runtime_binding_sha256
        registry_binding = self.registry.runtime_binding_sha256(self.tool_name)
        if registry_binding != service_binding:
            raise RetrievalRunError(
                "registry handler is not bound to the supplied product runtime"
            )

    @property
    def runtime_binding_sha256(self) -> str:
        try:
            binding = getattr(
                self.product_search_service, "formal_runtime_binding_sha256"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RetrievalRunError("product runtime is not formal-ready") from exc
        return _require_sha256(binding, "runtime_binding_sha256")

    @property
    def tool_spec_sha256(self) -> str:
        return next(
            spec.spec_sha256
            for spec in self.registry.specs()
            if spec.name == self.tool_name
        )

    @property
    def executor_id(self) -> str:
        identity = {
            "policy_version": "registry-retrieval-executor-v1",
            "registry_sha256": self.registry.registry_sha256,
            "registry_runtime_sha256": self.registry.registry_runtime_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "tool_name": self.tool_name,
            "tool_spec_sha256": self.tool_spec_sha256,
        }
        return "registry.v1." + sha256_bytes(canonical_json_bytes(identity))

    def __call__(self, request: RetrievalRequest) -> object:
        if request.tool_name != self.tool_name:
            raise RetrievalRunError("retrieval request tool does not match executor")
        if (
            self.registry.runtime_binding_sha256(self.tool_name)
            != self.runtime_binding_sha256
        ):
            raise RetrievalRunError(
                "product runtime identity changed before invocation"
            )

        def resolve_asset(asset_id: str) -> Path:
            if asset_id != request.asset_id:
                raise RetrievalRunError("registry requested a non-authoritative asset")
            return request.image_path

        context = ToolExecutionContext(
            query_id=request.query_id,
            query_asset_id=request.asset_id,
            query_text=request.text,
            resolve_asset=resolve_asset,
            query_cloud_upload_allowed=request.cloud_upload_allowed,
        )
        arguments: dict[str, object]
        if self.tool_name == "text_product_search":
            arguments = {"query": request.text}
        else:
            arguments = {"asset_id": request.asset_id}
        invocation = self.registry.invoke(self.tool_name, arguments, context)
        if (
            self.registry.runtime_binding_sha256(self.tool_name)
            != self.runtime_binding_sha256
        ):
            raise RetrievalRunError(
                "product runtime identity changed during invocation"
            )
        if invocation.spec_sha256 != self.tool_spec_sha256:
            raise RetrievalRunError("registry tool spec changed during retrieval")
        return invocation.output


@dataclass(frozen=True)
class VerifiedRetrievalRun:
    """A run whose bundle and current transitive inputs were fully verified."""

    root: Path
    manifest: RetrievalRunManifest
    spec: RetrievalRunSpec
    queries: tuple[Query, ...]
    executions: tuple[RetrievalQueryExecution, ...]
    calls: tuple[RetrievalToolCall, ...]
    _marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ProvisionalRetrievalRun:
    """Auditable bundle built by a generic executor, never accepted by evaluators."""

    root: Path
    manifest: RetrievalRunManifest
    spec: RetrievalRunSpec
    queries: tuple[Query, ...]
    executions: tuple[RetrievalQueryExecution, ...]
    calls: tuple[RetrievalToolCall, ...]


@dataclass(frozen=True)
class _ExternalSnapshot:
    query_bytes: bytes
    catalog_files: tuple[tuple[str, str], ...]
    query_assets: tuple[tuple[str, str], ...]


def _create_retrieval_run(
    query_artifact_path: str | Path,
    output_dir: str | Path,
    catalog: AssetCatalog,
    artifact_binding: RetrievalArtifactBinding,
    executor: RetrievalExecutor | Callable[[RetrievalRequest], object],
    *,
    run_id: str,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
    formal_registry: bool,
) -> VerifiedRetrievalRun | ProvisionalRetrievalRun:
    """Execute every authoritative query and atomically publish a new bundle.

    Ordinary executor exceptions are retained as hashed error results and do
    not remove a query from the run.  Contract violations (including a hit
    with the wrong binding) abort publication.
    """

    output_dir = Path(output_dir)
    query_artifact_path = Path(query_artifact_path)
    tool_spec_sha256 = _require_sha256(tool_spec_sha256, "tool_spec_sha256")
    run_id = _require_nonblank(run_id, "run_id")
    tool_name = _require_nonblank(tool_name, "tool_name")
    executor_id = _require_nonblank(executor_id, "executor_id")
    if not callable(executor):
        raise TypeError("executor must be callable")
    _require_verified_binding(artifact_binding)

    current_catalog, before = _capture_external_snapshot(query_artifact_path, catalog)
    query_bytes, queries = load_query_artifact(query_artifact_path, current_catalog)
    if not queries:
        raise RetrievalRunError("formal retrieval run requires at least one query")
    _validate_binding_inputs(artifact_binding, query_bytes, current_catalog)
    spec = _build_spec(
        run_id=run_id,
        query_bytes=query_bytes,
        queries=queries,
        catalog=current_catalog,
        artifact_binding=artifact_binding,
        tool_name=tool_name,
        tool_spec_sha256=tool_spec_sha256,
        executor_id=executor_id,
        assurance=("registry-verified" if formal_registry else "provisional-executor"),
    )

    staging = new_staging_directory(output_dir)
    try:
        calls: list[RetrievalToolCall] = []
        executions: list[RetrievalQueryExecution] = []
        for query_ordinal, query in enumerate(queries):
            asset = current_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            ).asset
            image_path = _resolved_asset_path(current_catalog, asset.local_path)
            _verify_asset_snapshot(image_path, asset.sha256, asset.asset_id)
            arguments = _authoritative_arguments(query, asset.sha256)
            arguments_sha256 = sha256_bytes(canonical_json_bytes(arguments))
            call_id = _call_id(
                spec.spec_sha256,
                query.query_id,
                0,
                tool_name,
                arguments_sha256,
            )
            request = RetrievalRequest(
                query_id=query.query_id,
                query_ordinal=query_ordinal,
                asset_id=query.asset_id,
                image_path=image_path,
                image_sha256=asset.sha256,
                cloud_upload_allowed=asset.cloud_upload_allowed,
                text=query.text,
                tool_name=tool_name,
                artifact_binding=artifact_binding,
                arguments=arguments,
            )
            try:
                raw_output = executor(request)
            except Exception as exc:  # noqa: BLE001 - errors are run evidence
                status: Literal["ok", "error"] = "error"
                stable_code = getattr(exc, "code", None)
                code = (
                    stable_code
                    if isinstance(stable_code, str)
                    and stable_code in _PUBLIC_ERROR_MESSAGES
                    else "tool_execution_failed"
                )
                output: JSONValue = {
                    "error": {
                        "code": code,
                        "message": _PUBLIC_ERROR_MESSAGES[code],
                    }
                }
            else:
                status = "ok"
                output = _validated_success_output(
                    raw_output,
                    artifact_binding,
                    query=query,
                    tool_name=tool_name,
                    expected_image_sha256=asset.sha256,
                    require_trace=formal_registry,
                )
            _verify_asset_snapshot(image_path, asset.sha256, asset.asset_id)

            call = _build_call(
                call_id=call_id,
                query_id=query.query_id,
                ordinal=0,
                tool_name=tool_name,
                tool_spec_sha256=tool_spec_sha256,
                arguments=arguments,
                arguments_sha256=arguments_sha256,
                status=status,
                output=output,
                artifact_binding=artifact_binding,
            )
            calls.append(call)
            executions.append(
                _build_execution(
                    query=query,
                    query_ordinal=query_ordinal,
                    asset_sha256=asset.sha256,
                    call_ids=[call.call_id],
                    status=status,
                )
            )

        _assert_external_snapshot(query_artifact_path, catalog, before)
        spec_bytes = canonical_json_bytes(spec)
        execution_bytes = canonical_jsonl_bytes(executions)
        call_bytes = canonical_jsonl_bytes(calls)
        artifact_bytes = {
            _SPEC_FILE: spec_bytes,
            _EXECUTIONS_FILE: execution_bytes,
            _CALLS_FILE: call_bytes,
        }
        for name, content in artifact_bytes.items():
            atomic_create_file(staging / name, content)
        manifest = _build_manifest(
            spec=spec,
            executions=executions,
            calls=calls,
            artifact_bytes=artifact_bytes,
        )
        atomic_create_file(staging / _MANIFEST_FILE, canonical_json_bytes(manifest))
        _load_retrieval_run(
            staging,
            query_artifact_path,
            catalog,
            artifact_binding=artifact_binding,
            tool_name=tool_name,
            tool_spec_sha256=tool_spec_sha256,
            executor_id=executor_id,
            verified=formal_registry,
        )
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return _load_retrieval_run(
        output_dir,
        query_artifact_path,
        catalog,
        artifact_binding=artifact_binding,
        tool_name=tool_name,
        tool_spec_sha256=tool_spec_sha256,
        executor_id=executor_id,
        verified=formal_registry,
    )


def create_provisional_retrieval_run(
    query_artifact_path: str | Path,
    output_dir: str | Path,
    catalog: AssetCatalog,
    artifact_binding: RetrievalArtifactBinding,
    executor: RetrievalExecutor | Callable[[RetrievalRequest], object],
    *,
    run_id: str,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
) -> ProvisionalRetrievalRun:
    """Build generic call evidence explicitly marked ineligible for evaluation."""
    result = _create_retrieval_run(
        query_artifact_path,
        output_dir,
        catalog,
        artifact_binding,
        executor,
        run_id=run_id,
        tool_name=tool_name,
        tool_spec_sha256=tool_spec_sha256,
        executor_id=executor_id,
        formal_registry=False,
    )
    if not isinstance(result, ProvisionalRetrievalRun):
        raise AssertionError("generic retrieval builder returned a formal run")
    return result


def create_registry_retrieval_run(
    query_artifact_path: str | Path,
    output_dir: str | Path,
    catalog: AssetCatalog,
    artifact_binding: RetrievalArtifactBinding,
    executor: RegistryRetrievalExecutor,
    *,
    run_id: str,
) -> VerifiedRetrievalRun:
    """Create a run without allowing caller-supplied registry identities."""

    if not isinstance(executor, RegistryRetrievalExecutor):
        raise TypeError("executor must be a RegistryRetrievalExecutor")
    result = _create_retrieval_run(
        query_artifact_path,
        output_dir,
        catalog,
        artifact_binding,
        executor,
        run_id=run_id,
        tool_name=executor.tool_name,
        tool_spec_sha256=executor.tool_spec_sha256,
        executor_id=executor.executor_id,
        formal_registry=True,
    )
    if not isinstance(result, VerifiedRetrievalRun):
        raise AssertionError("registry retrieval builder returned a provisional run")
    return result


def _load_retrieval_run(
    run_dir: str | Path,
    query_artifact_path: str | Path,
    catalog: AssetCatalog,
    *,
    artifact_binding: RetrievalArtifactBinding,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
    verified: bool,
) -> VerifiedRetrievalRun | ProvisionalRetrievalRun:
    """Load and reverify a run against its current authoritative inputs."""

    run_dir = Path(run_dir)
    query_artifact_path = Path(query_artifact_path)
    tool_name = _require_nonblank(tool_name, "tool_name")
    tool_spec_sha256 = _require_sha256(tool_spec_sha256, "tool_spec_sha256")
    executor_id = _require_nonblank(executor_id, "executor_id")
    _require_verified_binding(artifact_binding)

    current_catalog, external_before = _capture_external_snapshot(
        query_artifact_path, catalog
    )
    query_bytes, queries = load_query_artifact(query_artifact_path, current_catalog)
    if not queries:
        raise RetrievalRunError("formal retrieval run requires at least one query")
    _validate_binding_inputs(artifact_binding, query_bytes, current_catalog)

    bundle_before = {
        name: _read_regular(run_dir / name, f"retrieval run {name}")
        for name in (*_ARTIFACT_FILES, _MANIFEST_FILE)
    }
    manifest = _load_model(
        bundle_before[_MANIFEST_FILE], RetrievalRunManifest, _MANIFEST_FILE
    )
    spec = _load_model(bundle_before[_SPEC_FILE], RetrievalRunSpec, _SPEC_FILE)
    executions = _load_jsonl_models(
        bundle_before[_EXECUTIONS_FILE],
        RetrievalQueryExecution,
        _EXECUTIONS_FILE,
    )
    calls = _load_jsonl_models(
        bundle_before[_CALLS_FILE], RetrievalToolCall, _CALLS_FILE
    )

    _verify_self_hash(spec, "spec_sha256", _SPEC_FILE)
    _verify_self_hash(manifest, "manifest_sha256", _MANIFEST_FILE)
    for execution in executions:
        _verify_self_hash(
            execution, "execution_sha256", f"execution {execution.query_id}"
        )
    for call in calls:
        _verify_self_hash(call, "call_sha256", f"call {call.call_id}")
    _verify_bundle_descriptors(manifest, bundle_before, executions, calls)
    _verify_run_contract(
        manifest=manifest,
        spec=spec,
        queries=queries,
        executions=executions,
        calls=calls,
        catalog=current_catalog,
        query_bytes=query_bytes,
        artifact_binding=artifact_binding,
        tool_name=tool_name,
        tool_spec_sha256=tool_spec_sha256,
        executor_id=executor_id,
        require_trace=verified,
    )

    bundle_after = {
        name: _read_regular(run_dir / name, f"retrieval run {name}")
        for name in (*_ARTIFACT_FILES, _MANIFEST_FILE)
    }
    if bundle_after != bundle_before:
        raise RetrievalRunError("retrieval run bundle changed during verification")
    _assert_external_snapshot(query_artifact_path, catalog, external_before)

    result_type = VerifiedRetrievalRun if verified else ProvisionalRetrievalRun
    result_kwargs = dict(
        root=run_dir,
        manifest=manifest,
        spec=spec,
        queries=queries,
        executions=executions,
        calls=calls,
    )
    if verified:
        result_kwargs["_marker"] = _VERIFIED_MARKER
    return result_type(**result_kwargs)


def load_verified_retrieval_run(
    run_dir: str | Path,
    query_artifact_path: str | Path,
    catalog: AssetCatalog,
    *,
    artifact_binding: RetrievalArtifactBinding,
    executor: RegistryRetrievalExecutor,
) -> VerifiedRetrievalRun:
    """Reverify formal evidence against the same live registry-backed runtime."""
    if not isinstance(executor, RegistryRetrievalExecutor):
        raise TypeError("executor must be a RegistryRetrievalExecutor")
    result = _load_retrieval_run(
        run_dir,
        query_artifact_path,
        catalog,
        artifact_binding=artifact_binding,
        tool_name=executor.tool_name,
        tool_spec_sha256=executor.tool_spec_sha256,
        executor_id=executor.executor_id,
        verified=True,
    )
    if not isinstance(result, VerifiedRetrievalRun):
        raise AssertionError("formal loader returned a provisional run")
    return result


def load_provisional_retrieval_run(
    run_dir: str | Path,
    query_artifact_path: str | Path,
    catalog: AssetCatalog,
    *,
    artifact_binding: RetrievalArtifactBinding,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
) -> ProvisionalRetrievalRun:
    """Load generic evidence without granting the evaluator marker."""
    result = _load_retrieval_run(
        run_dir,
        query_artifact_path,
        catalog,
        artifact_binding=artifact_binding,
        tool_name=tool_name,
        tool_spec_sha256=tool_spec_sha256,
        executor_id=executor_id,
        verified=False,
    )
    if not isinstance(result, ProvisionalRetrievalRun):
        raise AssertionError("provisional loader returned a formal run")
    return result


def require_verified_retrieval_run(value: object) -> VerifiedRetrievalRun:
    """Typed evaluator gate; bare hits and prediction paths are rejected."""

    if (
        not isinstance(value, VerifiedRetrievalRun)
        or value._marker is not _VERIFIED_MARKER
    ):
        raise TypeError(
            "evaluator input must be a VerifiedRetrievalRun returned by the verifier"
        )
    return value


def _build_spec(
    *,
    run_id: str,
    query_bytes: bytes,
    queries: Sequence[Query],
    catalog: AssetCatalog,
    artifact_binding: RetrievalArtifactBinding,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
    assurance: Literal["registry-verified", "provisional-executor"],
) -> RetrievalRunSpec:
    unsigned = {
        "schema_version": 1,
        "kind": "retrieval-run-spec",
        "policy_version": RETRIEVAL_RUN_POLICY_VERSION,
        "assurance": assurance,
        "run_id": run_id,
        "query_artifact_sha256": sha256_bytes(query_bytes),
        "query_count": len(queries),
        "query_order_sha256": _query_order_sha256(queries),
        "asset_catalog_sha256": catalog.catalog_sha256,
        "artifact_binding": artifact_binding.model_dump(mode="json"),
        "tool_name": tool_name,
        "tool_spec_sha256": tool_spec_sha256,
        "executor_id": executor_id,
        "image_input_policy": "authoritative-query-asset-v1",
    }
    return RetrievalRunSpec.model_validate(
        {**unsigned, "spec_sha256": sha256_bytes(canonical_json_bytes(unsigned))}
    )


def _build_call(
    *,
    call_id: str,
    query_id: str,
    ordinal: int,
    tool_name: str,
    tool_spec_sha256: str,
    arguments: dict[str, JSONValue],
    arguments_sha256: str,
    status: Literal["ok", "error"],
    output: JSONValue,
    artifact_binding: RetrievalArtifactBinding,
) -> RetrievalToolCall:
    unsigned = {
        "schema_version": 1,
        "call_id": call_id,
        "query_id": query_id,
        "ordinal": ordinal,
        "tool_name": tool_name,
        "tool_spec_sha256": tool_spec_sha256,
        "arguments": arguments,
        "arguments_sha256": arguments_sha256,
        "status": status,
        "output": output,
        "output_sha256": sha256_bytes(canonical_json_bytes(output)),
        "artifact_binding": artifact_binding.model_dump(mode="json"),
    }
    return RetrievalToolCall.model_validate(
        {**unsigned, "call_sha256": sha256_bytes(canonical_json_bytes(unsigned))}
    )


def _build_execution(
    *,
    query: Query,
    query_ordinal: int,
    asset_sha256: str,
    call_ids: Sequence[str],
    status: Literal["ok", "error"],
) -> RetrievalQueryExecution:
    unsigned = {
        "schema_version": 1,
        "query_id": query.query_id,
        "query_ordinal": query_ordinal,
        "query_row_sha256": sha256_bytes(canonical_json_bytes(query)),
        "asset_id": query.asset_id,
        "asset_sha256": asset_sha256,
        "call_ids": list(call_ids),
        "status": status,
    }
    return RetrievalQueryExecution.model_validate(
        {
            **unsigned,
            "execution_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )


def _build_manifest(
    *,
    spec: RetrievalRunSpec,
    executions: Sequence[RetrievalQueryExecution],
    calls: Sequence[RetrievalToolCall],
    artifact_bytes: Mapping[str, bytes],
) -> RetrievalRunManifest:
    success_count = sum(execution.status == "ok" for execution in executions)
    artifacts = {
        name: RunArtifactDescriptor(
            path=name,
            bytes=len(content),
            sha256=sha256_bytes(content),
            rows=(
                1
                if name == _SPEC_FILE
                else len(executions)
                if name == _EXECUTIONS_FILE
                else len(calls)
            ),
        ).model_dump(mode="json")
        for name, content in artifact_bytes.items()
    }
    unsigned = {
        "schema_version": 1,
        "kind": "retrieval-run",
        "policy_version": RETRIEVAL_RUN_POLICY_VERSION,
        "run_id": spec.run_id,
        "status": "complete",
        "spec_sha256": spec.spec_sha256,
        "query_artifact_sha256": spec.query_artifact_sha256,
        "query_count": len(executions),
        "call_count": len(calls),
        "success_count": success_count,
        "error_count": len(executions) - success_count,
        "artifact_binding": spec.artifact_binding.model_dump(mode="json"),
        "artifacts": artifacts,
    }
    return RetrievalRunManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
    )


def _verify_run_contract(
    *,
    manifest: RetrievalRunManifest,
    spec: RetrievalRunSpec,
    queries: Sequence[Query],
    executions: Sequence[RetrievalQueryExecution],
    calls: Sequence[RetrievalToolCall],
    catalog: AssetCatalog,
    query_bytes: bytes,
    artifact_binding: RetrievalArtifactBinding,
    tool_name: str,
    tool_spec_sha256: str,
    executor_id: str,
    require_trace: bool,
) -> None:
    query_sha256 = sha256_bytes(query_bytes)
    expected_spec_values = {
        "assurance": ("registry-verified" if require_trace else "provisional-executor"),
        "query_artifact_sha256": query_sha256,
        "query_count": len(queries),
        "query_order_sha256": _query_order_sha256(queries),
        "asset_catalog_sha256": catalog.catalog_sha256,
        "artifact_binding": artifact_binding,
        "tool_name": tool_name,
        "tool_spec_sha256": tool_spec_sha256,
        "executor_id": executor_id,
    }
    for name, expected in expected_spec_values.items():
        if getattr(spec, name) != expected:
            raise RetrievalRunError(f"spec {name} does not match authoritative input")

    expected_manifest_values = {
        "run_id": spec.run_id,
        "spec_sha256": spec.spec_sha256,
        "query_artifact_sha256": query_sha256,
        "query_count": len(queries),
        "call_count": len(calls),
        "artifact_binding": artifact_binding,
    }
    for name, expected in expected_manifest_values.items():
        if getattr(manifest, name) != expected:
            raise RetrievalRunError(f"manifest {name} does not match verified run")

    expected_query_ids = [query.query_id for query in queries]
    actual_query_ids = [execution.query_id for execution in executions]
    if actual_query_ids != expected_query_ids:
        raise RetrievalRunError(
            "query executions must cover the complete authoritative query order"
        )
    if len(actual_query_ids) != len(set(actual_query_ids)):
        raise RetrievalRunError("query executions contain duplicate query_id")

    query_by_id = {query.query_id: query for query in queries}
    calls_by_query: dict[str, list[RetrievalToolCall]] = defaultdict(list)
    call_ids: set[str] = set()
    for call in calls:
        try:
            validate_json_value(call.arguments)
            validate_json_value(call.output)
        except (TypeError, ValueError) as exc:
            raise RetrievalRunError(
                f"call contains a non-JSON value: {call.call_id}"
            ) from exc
        if call.call_id in call_ids:
            raise RetrievalRunError(f"duplicate retrieval call_id: {call.call_id}")
        call_ids.add(call.call_id)
        if call.query_id not in query_by_id:
            raise RetrievalRunError(f"orphan retrieval call: {call.call_id}")
        calls_by_query[call.query_id].append(call)

    success_count = 0
    for query_ordinal, (query, execution) in enumerate(
        zip(queries, executions, strict=True)
    ):
        if execution.query_ordinal != query_ordinal:
            raise RetrievalRunError("query execution ordinal mismatch")
        resolution = catalog.verify_reference(
            query.asset_id,
            query.image_path,
            leakage_group_id=query.leakage_group_id,
        )
        expected_execution = _build_execution(
            query=query,
            query_ordinal=query_ordinal,
            asset_sha256=resolution.asset.sha256,
            call_ids=execution.call_ids,
            status=execution.status,
        )
        if execution != expected_execution:
            raise RetrievalRunError(
                f"query execution identity mismatch: {query.query_id}"
            )

        query_calls = calls_by_query.get(query.query_id, [])
        ordinals = [call.ordinal for call in query_calls]
        if ordinals != list(range(len(query_calls))):
            raise RetrievalRunError(
                f"retrieval call ordinals are not contiguous: {query.query_id}"
            )
        if [call.call_id for call in query_calls] != execution.call_ids:
            raise RetrievalRunError(
                f"query execution call references mismatch: {query.query_id}"
            )
        if len(query_calls) != 1:
            raise RetrievalRunError(
                f"formal retrieval run requires one call per query: {query.query_id}"
            )

        call = query_calls[0]
        expected_arguments = _authoritative_arguments(query, resolution.asset.sha256)
        expected_arguments_sha256 = sha256_bytes(
            canonical_json_bytes(expected_arguments)
        )
        if call.arguments != expected_arguments:
            raise RetrievalRunError(
                f"call arguments are not authoritative: {query.query_id}"
            )
        if call.arguments_sha256 != expected_arguments_sha256:
            raise RetrievalRunError(f"call arguments hash mismatch: {call.call_id}")
        if call.tool_name != tool_name or call.tool_spec_sha256 != tool_spec_sha256:
            raise RetrievalRunError(f"call tool identity mismatch: {call.call_id}")
        if call.artifact_binding != artifact_binding:
            raise RetrievalRunError(f"call artifact binding mismatch: {call.call_id}")
        expected_call_id = _call_id(
            spec.spec_sha256,
            query.query_id,
            call.ordinal,
            call.tool_name,
            call.arguments_sha256,
        )
        if call.call_id != expected_call_id:
            raise RetrievalRunError(f"call id mismatch: {call.call_id}")
        if call.output_sha256 != sha256_bytes(canonical_json_bytes(call.output)):
            raise RetrievalRunError(f"call output hash mismatch: {call.call_id}")
        if call.status == "ok":
            _validated_success_output(
                call.output,
                artifact_binding,
                query=query,
                tool_name=tool_name,
                expected_image_sha256=resolution.asset.sha256,
                require_trace=require_trace,
            )
            success_count += 1
        else:
            _validate_error_output(call.output, call.call_id)
        expected_status = call.status
        if execution.status != expected_status:
            raise RetrievalRunError(f"query/call status mismatch: {query.query_id}")

    if set(calls_by_query) != set(expected_query_ids):
        raise RetrievalRunError("retrieval calls do not cover every query")
    if manifest.success_count != success_count:
        raise RetrievalRunError("manifest success_count mismatch")
    if manifest.error_count != len(queries) - success_count:
        raise RetrievalRunError("manifest error_count mismatch")


def _verify_bundle_descriptors(
    manifest: RetrievalRunManifest,
    bundle: Mapping[str, bytes],
    executions: Sequence[RetrievalQueryExecution],
    calls: Sequence[RetrievalToolCall],
) -> None:
    expected_rows = {
        _SPEC_FILE: 1,
        _EXECUTIONS_FILE: len(executions),
        _CALLS_FILE: len(calls),
    }
    for name in _ARTIFACT_FILES:
        content = bundle[name]
        descriptor = manifest.artifacts[name]
        if descriptor.bytes != len(content):
            raise RetrievalRunError(f"{name}: byte count mismatch")
        if descriptor.sha256 != sha256_bytes(content):
            raise RetrievalRunError(f"{name}: sha256 mismatch")
        if descriptor.rows != expected_rows[name]:
            raise RetrievalRunError(f"{name}: row count mismatch")


def _capture_external_snapshot(
    query_artifact_path: Path,
    catalog: AssetCatalog,
) -> tuple[AssetCatalog, _ExternalSnapshot]:
    try:
        catalog.require_verified_files()
        current = load_asset_catalog(
            catalog.root,
            catalog.asset_root,
            verify_files=True,
        )
    except ValueError as exc:
        raise RetrievalRunError(str(exc)) from exc
    if current.catalog_sha256 != catalog.catalog_sha256:
        raise RetrievalRunError("asset catalog changed since it was loaded")
    catalog_files = tuple(
        (
            name,
            sha256_bytes(_read_regular(current.root / name, f"asset catalog {name}")),
        )
        for name in _CATALOG_FILES
    )
    query_bytes = _read_regular(query_artifact_path, "query artifact")
    loaded_bytes, queries = load_query_artifact(query_artifact_path, current)
    if query_bytes != loaded_bytes:
        raise RetrievalRunError("query artifact changed while it was loaded")
    query_assets: list[tuple[str, str]] = []
    for asset_id in sorted({query.asset_id for query in queries}):
        asset = current.resolve_asset_id(asset_id).asset
        path = _resolved_asset_path(current, asset.local_path)
        content_sha256 = sha256_bytes(
            _read_regular(path, f"query asset {asset.asset_id}")
        )
        if content_sha256 != asset.sha256:
            raise RetrievalRunError(f"query asset sha256 mismatch: {asset.asset_id}")
        query_assets.append((asset.asset_id, content_sha256))
    return current, _ExternalSnapshot(
        query_bytes=query_bytes,
        catalog_files=catalog_files,
        query_assets=tuple(query_assets),
    )


def _assert_external_snapshot(
    query_artifact_path: Path,
    catalog: AssetCatalog,
    expected: _ExternalSnapshot,
) -> None:
    _, actual = _capture_external_snapshot(query_artifact_path, catalog)
    if actual != expected:
        raise RetrievalRunError("authoritative retrieval inputs changed during run")


def _validate_binding_inputs(
    artifact_binding: RetrievalArtifactBinding,
    query_bytes: bytes,
    catalog: AssetCatalog,
) -> None:
    if artifact_binding.query_artifact_sha256 != sha256_bytes(query_bytes):
        raise RetrievalRunError(
            "retrieval binding does not match the authoritative query artifact"
        )
    if artifact_binding.asset_catalog_sha256 != catalog.catalog_sha256:
        raise RetrievalRunError(
            "retrieval binding does not match the authoritative asset catalog"
        )


def _authoritative_arguments(query: Query, asset_sha256: str) -> dict[str, JSONValue]:
    return {
        "query_id": query.query_id,
        "query_asset": {
            "asset_id": query.asset_id,
            "sha256": asset_sha256,
        },
        "query_text": query.text,
    }


def _validated_success_output(
    value: object,
    artifact_binding: RetrievalArtifactBinding,
    *,
    query: Query | None = None,
    tool_name: str | None = None,
    expected_image_sha256: str | None = None,
    require_trace: bool = False,
) -> JSONValue:
    try:
        output = validate_json_value(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalRunError(f"retrieval output is not strict JSON: {exc}") from exc
    if isinstance(output, dict):
        try:
            trace = ProductSearchTrace.model_validate(output, strict=True)
        except ValidationError as exc:
            raise RetrievalRunError("product retrieval trace is invalid") from exc
        if trace.artifact_binding != artifact_binding:
            raise RetrievalRunError("product retrieval trace binding mismatch")
        if tool_name is not None and trace.tool_name != tool_name:
            raise RetrievalRunError("product retrieval trace tool mismatch")
        if query is not None:
            if trace.input_kind == "image" and trace.query_asset_id != query.asset_id:
                raise RetrievalRunError("product retrieval trace asset mismatch")
            if (
                trace.input_kind == "image"
                and expected_image_sha256 is not None
                and trace.query_input_sha256 != expected_image_sha256
            ):
                raise RetrievalRunError("product retrieval trace input hash mismatch")
            if trace.input_kind == "text" and trace.query_text != query.text:
                raise RetrievalRunError("product retrieval trace text mismatch")
            if trace.input_kind == "text" and trace.query_input_sha256 != sha256_bytes(
                query.text.encode("utf-8")
            ):
                raise RetrievalRunError("product retrieval trace input hash mismatch")
        return output
    if require_trace:
        raise RetrievalRunError("formal product retrieval output must be a trace")
    if not isinstance(output, list):
        raise RetrievalRunError(
            "successful retrieval output must be a hit list or trace"
        )
    for index, item in enumerate(output):
        if not isinstance(item, dict):
            raise RetrievalRunError(f"retrieval hit {index} must be an object")
        if "artifact_binding" not in item:
            raise RetrievalRunError(
                f"retrieval hit {index} is missing artifact_binding"
            )
        try:
            hit_binding = RetrievalArtifactBinding.model_validate(
                item["artifact_binding"], strict=True
            )
        except ValidationError as exc:
            raise RetrievalRunError(
                f"retrieval hit {index} has an invalid artifact_binding"
            ) from exc
        if hit_binding != artifact_binding:
            raise RetrievalRunError(f"retrieval hit {index} artifact_binding mismatch")
    return output


def _validate_error_output(value: JSONValue, call_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"error"}:
        raise RetrievalRunError(f"error call has invalid output envelope: {call_id}")
    error = value["error"]
    if not isinstance(error, dict) or set(error) != {"code", "message"}:
        raise RetrievalRunError(f"error call has invalid error payload: {call_id}")
    if any(not isinstance(error[key], str) or not error[key].strip() for key in error):
        raise RetrievalRunError(f"error call has blank error metadata: {call_id}")
    code = error["code"]
    if code not in _PUBLIC_ERROR_MESSAGES:
        raise RetrievalRunError(f"error call has unknown public code: {call_id}")
    if error["message"] != _PUBLIC_ERROR_MESSAGES[code]:
        raise RetrievalRunError(f"error call has noncanonical message: {call_id}")


def _query_order_sha256(queries: Sequence[Query]) -> str:
    rows = (
        {
            "ordinal": ordinal,
            "query_id": query.query_id,
            "query_row_sha256": sha256_bytes(canonical_json_bytes(query)),
        }
        for ordinal, query in enumerate(queries)
    )
    return sha256_bytes(canonical_jsonl_bytes(rows))


def _call_id(
    spec_sha256: str,
    query_id: str,
    ordinal: int,
    tool_name: str,
    arguments_sha256: str,
) -> str:
    payload = {
        "arguments_sha256": arguments_sha256,
        "ordinal": ordinal,
        "query_id": query_id,
        "spec_sha256": spec_sha256,
        "tool_name": tool_name,
    }
    return "call.v1." + sha256_bytes(canonical_json_bytes(payload))


def _resolved_asset_path(catalog: AssetCatalog, local_path: str) -> Path:
    candidate = catalog.asset_root.joinpath(*local_path.split("/"))
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(catalog.asset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise RetrievalRunError(f"query asset path is invalid: {local_path}") from exc
    return resolved


def _verify_asset_snapshot(path: Path, expected_sha256: str, asset_id: str) -> None:
    if sha256_bytes(_read_regular(path, f"query asset {asset_id}")) != expected_sha256:
        raise RetrievalRunError(f"query asset changed during call: {asset_id}")


def _load_model(content: bytes, model_type, label: str):
    raw = _load_json_object(content, label)
    try:
        model = model_type.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise RetrievalRunError(f"{label}: schema validation failed: {exc}") from exc
    if content != canonical_json_bytes(model):
        raise RetrievalRunError(f"{label}: must be canonical JSON")
    return model


def _load_jsonl_models(content: bytes, model_type, label: str):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetrievalRunError(f"{label}: must be UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise RetrievalRunError(f"{label}: must contain at least one row")
    models = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RetrievalRunError(f"{label}: line {line_number} is blank")
        raw = _decode_json(line, f"{label} line {line_number}")
        if not isinstance(raw, dict):
            raise RetrievalRunError(f"{label}: line {line_number} must be an object")
        try:
            models.append(model_type.model_validate(raw, strict=True))
        except ValidationError as exc:
            raise RetrievalRunError(
                f"{label}: line {line_number} schema validation failed: {exc}"
            ) from exc
    result = tuple(models)
    if content != canonical_jsonl_bytes(result):
        raise RetrievalRunError(f"{label}: must be canonical JSONL")
    return result


def _load_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetrievalRunError(f"{label}: must be UTF-8") from exc
    value = _decode_json(text, label)
    if not isinstance(value, dict):
        raise RetrievalRunError(f"{label}: root must be an object")
    return value


def _decode_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RetrievalRunError(f"{label}: invalid JSON: {exc}") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RetrievalRunError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise RetrievalRunError(f"JSON contains a non-finite number: {value}")


def _read_regular(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise RetrievalRunError(f"{label} does not exist: {path}") from None
    except OSError as exc:
        raise RetrievalRunError(f"unable to inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RetrievalRunError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RetrievalRunError(f"{label} must be a regular file: {path}")
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise RetrievalRunError(
                    f"{label} must remain a regular file while reading: {path}"
                )
            if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
                raise RetrievalRunError(f"{label} changed before reading: {path}")
            return source.read()
    except OSError as exc:
        raise RetrievalRunError(f"unable to read {label}: {path}") from exc


def _verify_self_hash(model: BaseModel, field_name: str, label: str) -> None:
    payload = model.model_dump(mode="json", exclude={field_name})
    expected = sha256_bytes(canonical_json_bytes(payload))
    if getattr(model, field_name) != expected:
        raise RetrievalRunError(f"{label}: self hash mismatch")


def _require_verified_binding(binding: RetrievalArtifactBinding) -> None:
    if not isinstance(binding, RetrievalArtifactBinding):
        raise TypeError("artifact_binding must be a RetrievalArtifactBinding")
    if binding.mode != "verified":
        raise RetrievalRunError("formal retrieval run rejects provisional bindings")


def _require_nonblank(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be blank")
    return value


def _require_sha256(value: str, label: str) -> str:
    value = _require_nonblank(value, label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


publish_provisional_retrieval_run = create_provisional_retrieval_run


__all__ = [
    "RETRIEVAL_RUN_POLICY_VERSION",
    "RegistryRetrievalExecutor",
    "ProvisionalRetrievalRun",
    "RetrievalExecutor",
    "RetrievalQueryExecution",
    "RetrievalRequest",
    "RetrievalRunError",
    "RetrievalRunManifest",
    "RetrievalRunSpec",
    "RetrievalToolCall",
    "RunArtifactDescriptor",
    "VerifiedRetrievalRun",
    "create_provisional_retrieval_run",
    "create_registry_retrieval_run",
    "load_verified_retrieval_run",
    "load_provisional_retrieval_run",
    "publish_provisional_retrieval_run",
    "require_verified_retrieval_run",
]
