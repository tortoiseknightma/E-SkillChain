"""Versioned, strict, and deterministic registries for the MVP tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol
from weakref import WeakKeyDictionary, ref

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

MVP_TOOL_NAMES = frozenset(
    {
        "image_product_search",
        "text_product_search",
        "style_similar_search",
        "encyclopedia_lookup",
        "recipe_lookup",
        "object_detect",
        "document_ocr",
    }
)
MVP_TOOL_NAMES_V2 = frozenset({*MVP_TOOL_NAMES, "multi_product_search"})
_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024

InputBinding = Literal[
    "query_asset",
    "query_asset_and_text_exact",
    "query_text_exact",
    "query_derived_text",
]
NetworkPolicy = Literal["offline", "runtime_bound_embedding"]
OutputTrust = Literal[
    "retrieval_evidence",
    "model_prediction",
    "untrusted_document_text",
    "mixed_prediction_and_retrieval_evidence",
]


class RegistryError(ValueError):
    """Registry construction or persisted-spec validation failed."""


class ToolCallError(ValueError):
    """Stable public tool failure which never exposes a local path or traceback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ToolSpec(_StrictFrozenModel):
    """Serializable identity of one externally callable tool."""

    schema_version: Literal[1] = 1
    name: str
    tool_version: str
    description: str
    input_binding: InputBinding
    network_policy: NetworkPolicy
    output_trust: OutputTrust
    input_json_schema: dict[str, Any]
    output_json_schema: dict[str, Any]
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("name", "tool_version", "description")
    @classmethod
    def require_nonblank(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @model_validator(mode="after")
    def verify_self_hash(self) -> "ToolSpec":
        expected = _digest_without_field(self.model_dump(mode="json"), "spec_sha256")
        if self.spec_sha256 != expected:
            raise ValueError("spec_sha256 mismatch")
        return self


class RegistryManifest(_StrictFrozenModel):
    """Canonical, content-addressed snapshot of one registry generation."""

    schema_version: Literal[1, 2] = 1
    tool_count: Literal[7, 8] = 7
    tools: tuple[ToolSpec, ...]
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_registry(self) -> "RegistryManifest":
        names = [tool.name for tool in self.tools]
        if names != sorted(names):
            raise ValueError("registry tools must be sorted by name")
        expected = MVP_TOOL_NAMES if self.schema_version == 1 else MVP_TOOL_NAMES_V2
        if set(names) != expected or self.tool_count != len(expected):
            raise ValueError("registry tool set does not match its schema generation")
        expected = _digest_without_field(
            self.model_dump(mode="json"), "registry_sha256"
        )
        if self.registry_sha256 != expected:
            raise ValueError("registry_sha256 mismatch")
        return self


class AssetToolInput(_StrictFrozenModel):
    asset_id: str = Field(min_length=1)


class StyleSimilarSearchInput(_StrictFrozenModel):
    """Private runner-bound input for query-conditioned Style retrieval."""

    asset_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=4096)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class TextProductSearchInput(_StrictFrozenModel):
    query: str = Field(min_length=1, max_length=4096)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class EncyclopediaLookupInput(_StrictFrozenModel):
    entity: str = Field(min_length=1, max_length=512)

    @field_validator("entity")
    @classmethod
    def reject_blank_entity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity must not be blank")
        return value


class RecipeLookupInput(_StrictFrozenModel):
    dish: str = Field(min_length=1, max_length=512)

    @field_validator("dish")
    @classmethod
    def reject_blank_dish(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dish must not be blank")
        return value


@dataclass(frozen=True)
class ToolExecutionContext:
    """Opaque authoritative query view supplied by a formal runner."""

    query_id: str
    query_asset_id: str
    query_text: str
    resolve_asset: Callable[[str], Path]
    query_cloud_upload_allowed: bool | None = None

    def asset_path(self, asset_id: str) -> Path:
        if asset_id != self.query_asset_id:
            raise ToolCallError(
                "context_violation",
                "asset_id is not the authoritative asset for this query",
            )
        return Path(self.resolve_asset(asset_id))

    def remote_asset_path(self, asset_id: str) -> Path:
        """Resolve an image only when its catalog permission explicitly allows upload."""

        if self.query_cloud_upload_allowed is not True:
            raise ToolCallError(
                "permission_denied",
                "remote image processing is not permitted for this asset",
            )
        return self.asset_path(asset_id)


ToolHandler = Callable[[BaseModel, ToolExecutionContext], object]
RuntimeBindingResolver = Callable[[], str]
FormalEvidenceValidator = Callable[[object], None]

_CANONICAL_MVP_BUILDER_TOKEN = object()
_FORMAL_RUNTIME_HANDLE_MARKER = object()


class SafetyApprovalResolver(Protocol):
    def __call__(self, query_id: str, asset_id: str) -> object: ...


@dataclass(frozen=True)
class ToolDefinition:
    """Non-serializable handler plus all inputs needed to derive a ToolSpec."""

    name: str
    tool_version: str
    description: str
    input_model: type[BaseModel]
    output_type: Any
    input_binding: InputBinding
    network_policy: NetworkPolicy
    output_trust: OutputTrust
    handler: ToolHandler
    runtime_binding_sha256: str | None = None
    runtime_binding_resolver: RuntimeBindingResolver | None = None
    evidence_runtime_binding_sha256: str | None = None
    evidence_runtime_binding_resolver: RuntimeBindingResolver | None = None
    formal_evidence_validator: FormalEvidenceValidator | None = None


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    input_model: type[BaseModel]
    output_adapter: TypeAdapter[Any]
    handler: ToolHandler
    runtime_binding_sha256: str
    runtime_binding_explicit: bool
    runtime_binding_resolver: RuntimeBindingResolver | None
    evidence_runtime_binding_sha256: str | None
    evidence_runtime_binding_resolver: RuntimeBindingResolver | None
    formal_evidence_validator: FormalEvidenceValidator | None
    handler_implementation_sha256: str


@dataclass(frozen=True)
class ToolInvocationResult:
    """Validated public output plus hashes required by a private call trace."""

    tool_name: str
    spec_sha256: str
    arguments: dict[str, Any]
    arguments_bytes: bytes
    arguments_sha256: str
    output: object
    output_bytes: bytes
    output_sha256: str


@dataclass(frozen=True)
class FormalRegistryRuntimeSnapshot:
    registry_sha256: str
    registry_runtime_sha256: str
    tool_runtime_bindings: tuple[tuple[str, str], ...]
    evidence_runtime_bindings: tuple[tuple[str, str | None], ...]

    def tool_runtime_sha256(self, name: str) -> str:
        try:
            return dict(self.tool_runtime_bindings)[name]
        except KeyError as error:
            raise RegistryError("unknown tool in formal runtime snapshot") from error

    def evidence_runtime_sha256(self, name: str) -> str | None:
        try:
            return dict(self.evidence_runtime_bindings)[name]
        except KeyError as error:
            raise RegistryError("unknown tool in formal runtime snapshot") from error


@dataclass(frozen=True)
class MVPToolServices:
    """Explicit runtimes used to wire either canonical registry generation."""

    product_search: object
    kb_lookup: object
    object_detection: object
    document_ocr: object
    safety_approval_for: SafetyApprovalResolver
    # Portfolio diagnostics may supply a public-data composition that is not
    # eligible for the Formal authority.  Formal callers leave this unset and
    # continue to receive the reviewed MultiProductSearchService.
    multi_product_search: object | None = None


class ToolRegistry:
    """Dispatch one complete, versioned tool set with deterministic bytes."""

    __slots__ = ("_manifest", "_tools", "__weakref__")

    def __init__(
        self,
        definitions: list[ToolDefinition] | tuple[ToolDefinition, ...],
    ):
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise RegistryError("tool names must be unique")
        actual = set(names)
        if actual == MVP_TOOL_NAMES:
            pass
        elif actual == MVP_TOOL_NAMES_V2:
            pass
        else:
            missing = sorted(MVP_TOOL_NAMES - actual)
            extra = sorted(actual - MVP_TOOL_NAMES_V2)
            raise RegistryError(
                "registry must contain exactly seven legacy or eight v2 MVP tools; "
                f"missing_legacy={missing}, unsupported={extra}"
            )

        registered: dict[str, RegisteredTool] = {}
        for definition in definitions:
            input_adapter = TypeAdapter(definition.input_model)
            output_adapter = TypeAdapter(definition.output_type)
            input_schema = input_adapter.json_schema()
            output_schema = output_adapter.json_schema()
            _assert_recursively_strict_schema(input_schema, f"{definition.name} input")
            _assert_recursively_strict_schema(
                output_schema, f"{definition.name} output"
            )
            spec_payload = {
                "schema_version": 1,
                "name": definition.name,
                "tool_version": definition.tool_version,
                "description": definition.description,
                "input_binding": definition.input_binding,
                "network_policy": definition.network_policy,
                "output_trust": definition.output_trust,
                "input_json_schema": input_schema,
                "output_json_schema": output_schema,
            }
            spec = ToolSpec.model_validate(
                {
                    **spec_payload,
                    "spec_sha256": sha256_bytes(canonical_json_bytes(spec_payload)),
                }
            )
            registered[definition.name] = RegisteredTool(
                spec=spec,
                input_model=definition.input_model,
                output_adapter=output_adapter,
                handler=definition.handler,
                runtime_binding_sha256=_runtime_handler_binding(definition),
                runtime_binding_explicit=(
                    definition.runtime_binding_sha256 is not None
                ),
                runtime_binding_resolver=definition.runtime_binding_resolver,
                evidence_runtime_binding_sha256=(
                    _optional_runtime_binding(
                        definition.evidence_runtime_binding_sha256,
                        f"{definition.name} evidence runtime",
                    )
                ),
                evidence_runtime_binding_resolver=(
                    definition.evidence_runtime_binding_resolver
                ),
                formal_evidence_validator=definition.formal_evidence_validator,
                handler_implementation_sha256=(
                    _handler_implementation_binding(definition.handler)
                ),
            )
        self._tools = registered
        self._manifest = _build_manifest(tuple(self.specs()))

    @property
    def registry_sha256(self) -> str:
        return self._manifest.registry_sha256

    @property
    def manifest(self) -> RegistryManifest:
        return self._manifest

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name].spec for name in sorted(self._tools))

    def runtime_binding_sha256(self, name: str) -> str:
        """Return the explicit live handler/runtime binding for one tool."""
        registered = self._tools.get(name)
        if registered is None:
            raise RegistryError("unknown tool runtime binding")
        return _resolve_registered_runtime(registered, evidence=False)

    def evidence_runtime_binding_sha256(self, name: str) -> str | None:
        registered = self._tools.get(name)
        if registered is None:
            raise RegistryError("unknown tool evidence runtime binding")
        return _resolve_registered_runtime(registered, evidence=True)

    def verify_formal_evidence(
        self,
        name: str,
        evidence: object,
        *,
        recheck_runtime: bool = True,
    ) -> None:
        self._assert_formal_runtime_structure()
        registered = self._tools.get(name)
        if registered is None:
            raise RegistryError("unknown formal evidence tool")
        validator = registered.formal_evidence_validator
        if validator is None:
            raise RegistryError(f"formal evidence validator is unavailable: {name}")
        before = (
            (
                self.runtime_binding_sha256(name),
                self.evidence_runtime_binding_sha256(name),
            )
            if recheck_runtime
            else None
        )
        try:
            validator(evidence)
        except RegistryError:
            raise
        except Exception as error:
            raise RegistryError(f"formal evidence mismatch: {name}") from error
        after = (
            (
                self.runtime_binding_sha256(name),
                self.evidence_runtime_binding_sha256(name),
            )
            if recheck_runtime
            else None
        )
        if after != before:
            raise RegistryError(
                f"tool runtime changed while verifying evidence: {name}"
            )

    @property
    def formal_runtime_ready(self) -> bool:
        try:
            formal_registry_runtime_snapshot(self)
        except Exception:
            return False
        return True

    def require_formal_runtime(self) -> "FormalRegistryRuntimeHandle":
        """Return the controlled handle after resolving every live binding once."""

        handle = require_formal_registry_runtime(self)
        handle.snapshot()
        return handle

    def formal_runtime_snapshot(self) -> FormalRegistryRuntimeSnapshot:
        """Resolve every formal runtime once into an immutable comparison token."""

        return formal_registry_runtime_snapshot(self)

    def verify_formal_runtime_snapshot(
        self, snapshot: FormalRegistryRuntimeSnapshot
    ) -> None:
        """Fail closed if any formal runtime changed since ``snapshot``."""

        require_formal_registry_runtime(self).verify(snapshot)

    def _assert_formal_runtime_structure(self) -> None:
        require_formal_registry_runtime(self)._validate_registry(self)

    @property
    def registry_runtime_sha256(self) -> str:
        if _formal_runtime_is_issued(self):
            return formal_registry_runtime_snapshot(self).registry_runtime_sha256
        tool_bindings = tuple(
            (name, self.runtime_binding_sha256(name)) for name in sorted(self._tools)
        )
        evidence_bindings = tuple(
            (name, self.evidence_runtime_binding_sha256(name))
            for name in sorted(self._tools)
        )
        return self._registry_runtime_digest(tool_bindings, evidence_bindings)

    def _registry_runtime_digest(
        self,
        tool_bindings: tuple[tuple[str, str], ...],
        evidence_bindings: tuple[tuple[str, str | None], ...],
    ) -> str:
        generation = self._manifest.schema_version
        payload = {
            "policy_version": f"registry-runtime-v{generation}",
            "registry_sha256": self.registry_sha256,
            "tools": dict(tool_bindings),
            "evidence_runtime": dict(evidence_bindings),
            "handler_implementation": {
                name: self._tools[name].handler_implementation_sha256
                for name in sorted(self._tools)
            },
            "reviewed_runtime_implementation_sha256": (
                _REVIEWED_RUNTIME_IMPLEMENTATION_SHA256
            ),
            "runtime_binding_mode": {
                name: (
                    "explicit"
                    if self._tools[name].runtime_binding_explicit
                    else "derived-diagnostic"
                )
                for name in sorted(self._tools)
            },
            "registry_builder": (
                f"canonical-mvp-v{generation}"
                if _formal_runtime_is_issued(self)
                else "generic-diagnostic"
            ),
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> ToolInvocationResult:
        registered = self._tools.get(name)
        if registered is None:
            raise ToolCallError("unknown_tool", "unknown tool")
        if not isinstance(arguments, Mapping):
            raise ToolCallError("invalid_arguments", "tool arguments must be an object")
        raw_arguments = dict(arguments)
        try:
            argument_bytes = canonical_json_bytes(raw_arguments)
        except (ArtifactFormatError, TypeError, ValueError) as error:
            raise ToolCallError(
                "invalid_arguments", "tool arguments are not JSON"
            ) from error
        if len(argument_bytes) > _MAX_ARGUMENT_BYTES:
            raise ToolCallError("size_limit", "tool arguments exceed the byte limit")
        try:
            validated_input = registered.input_model.model_validate(
                raw_arguments, strict=True
            )
        except ValidationError as error:
            raise ToolCallError(
                "invalid_arguments", "tool arguments violate schema"
            ) from error

        dumped_arguments = validated_input.model_dump(mode="json")
        self._enforce_context_binding(registered.spec, dumped_arguments, context)
        argument_bytes = canonical_json_bytes(dumped_arguments)
        formally_attested = _formal_runtime_is_issued(self)
        if formally_attested:
            self._assert_formal_runtime_structure()
        runtime_before = (
            self.runtime_binding_sha256(name) if formally_attested else None
        )
        try:
            raw_output = registered.handler(validated_input, context)
        except ToolCallError:
            raise
        except Exception as error:
            raise ToolCallError("handler_failed", "tool execution failed") from error
        try:
            validated_output = registered.output_adapter.validate_python(
                raw_output, strict=True
            )
            dumped_output = registered.output_adapter.dump_python(
                validated_output, mode="json"
            )
            output_bytes = canonical_json_bytes(dumped_output)
        except (ValidationError, ArtifactFormatError, TypeError, ValueError) as error:
            raise ToolCallError(
                "invalid_output", "tool output violates schema"
            ) from error
        if len(output_bytes) > _MAX_OUTPUT_BYTES:
            raise ToolCallError("size_limit", "tool output exceeds the byte limit")
        if runtime_before is not None:
            try:
                self._assert_formal_runtime_structure()
                runtime_after = self.runtime_binding_sha256(name)
            except RegistryError as error:
                raise ToolCallError(
                    "runtime_changed", "tool runtime changed during invocation"
                ) from error
            if runtime_after != runtime_before:
                raise ToolCallError(
                    "runtime_changed", "tool runtime changed during invocation"
                )
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256=registered.spec.spec_sha256,
            arguments=dumped_arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=dumped_output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> object:
        return self.invoke(name, arguments, context).output

    def call_tool_json(
        self,
        name: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> str:
        return self.invoke(name, arguments, context).output_bytes.decode("utf-8")

    @staticmethod
    def _enforce_context_binding(
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> None:
        if spec.input_binding in {
            "query_asset",
            "query_asset_and_text_exact",
        }:
            asset_id = arguments.get("asset_id")
            if asset_id != context.query_asset_id:
                raise ToolCallError(
                    "context_violation",
                    "asset_id is not the authoritative asset for this query",
                )
            if (
                spec.input_binding == "query_asset_and_text_exact"
                and arguments.get("query") != context.query_text
            ):
                raise ToolCallError(
                    "context_violation",
                    "query text is not the authoritative text for this query",
                )
        elif spec.input_binding == "query_text_exact":
            if arguments.get("query") != context.query_text:
                raise ToolCallError(
                    "context_violation",
                    "query text is not the authoritative text for this query",
                )


class DiagnosticToolRegistry(ToolRegistry):
    """Explicitly provisional registry which can never receive formal authority."""

    __slots__ = ()


class _FormalToolRegistry(ToolRegistry):
    """Registry type constructed only after the reviewed service graph is verified."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _FormalRegistryAttestation:
    nonce: object
    registry_sha256: str
    manifest_identity: int
    manifest_bytes: bytes
    tools_identity: int
    tool_entries: tuple[tuple[str, RegisteredTool], ...]
    runtime_guard: Callable[[], None]


@dataclass(frozen=True, slots=True)
class FormalRegistryRuntimeHandle:
    """Opaque capability for one authority-issued formal registry instance."""

    _registry_ref: Any
    _nonce: object
    _marker: object

    def _validate_registry(self, registry: ToolRegistry) -> None:
        if self._marker is not _FORMAL_RUNTIME_HANDLE_MARKER:
            raise RegistryError(
                "formal registry runtime handle is not authority-issued"
            )
        if self._registry_ref() is not registry:
            raise RegistryError(
                "formal registry runtime handle belongs to another registry"
            )
        _FORMAL_RUNTIME_AUTHORITY.validate(registry, handle=self)

    def snapshot(self) -> FormalRegistryRuntimeSnapshot:
        registry = self._registry_ref()
        if registry is None:
            raise RegistryError("formal registry runtime is no longer available")
        return _capture_formal_registry_runtime(registry, self)

    def verify(self, snapshot: FormalRegistryRuntimeSnapshot) -> None:
        if not isinstance(snapshot, FormalRegistryRuntimeSnapshot):
            raise TypeError("formal runtime snapshot required")
        if self.snapshot() != snapshot:
            raise RegistryError("formal registry runtime changed during verification")


class _FormalRuntimeAuthority:
    """Keep formal grants outside writable registry instance state."""

    __slots__ = ("_lock", "_records")

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: WeakKeyDictionary[ToolRegistry, _FormalRegistryAttestation] = (
            WeakKeyDictionary()
        )

    def issue(
        self,
        registry: ToolRegistry,
        runtime_guard: Callable[[], None],
        *,
        builder_token: object,
    ) -> FormalRegistryRuntimeHandle:
        if builder_token is not _CANONICAL_MVP_BUILDER_TOKEN:
            raise RegistryError("formal registry authority rejected the issuer")
        if type(registry) is not _FormalToolRegistry:
            raise RegistryError("formal authority requires the canonical registry type")
        tools = object.__getattribute__(registry, "_tools")
        manifest = object.__getattribute__(registry, "_manifest")
        attestation = _FormalRegistryAttestation(
            nonce=object(),
            registry_sha256=manifest.registry_sha256,
            manifest_identity=id(manifest),
            manifest_bytes=canonical_json_bytes(manifest.model_dump(mode="json")),
            tools_identity=id(tools),
            tool_entries=tuple(sorted(tools.items())),
            runtime_guard=runtime_guard,
        )
        with self._lock:
            if registry in self._records:
                raise RegistryError("formal registry authority refuses re-issuance")
            self._records[registry] = attestation
        handle = FormalRegistryRuntimeHandle(
            _registry_ref=ref(registry),
            _nonce=attestation.nonce,
            _marker=_FORMAL_RUNTIME_HANDLE_MARKER,
        )
        try:
            handle.snapshot()
        except Exception:
            with self._lock:
                self._records.pop(registry, None)
            raise
        return handle

    def is_issued(self, registry: ToolRegistry) -> bool:
        if type(registry) is not _FormalToolRegistry:
            return False
        with self._lock:
            return registry in self._records

    def handle(self, registry: ToolRegistry) -> FormalRegistryRuntimeHandle:
        attestation = self.validate(registry)
        return FormalRegistryRuntimeHandle(
            _registry_ref=ref(registry),
            _nonce=attestation.nonce,
            _marker=_FORMAL_RUNTIME_HANDLE_MARKER,
        )

    def validate(
        self,
        registry: ToolRegistry,
        *,
        handle: FormalRegistryRuntimeHandle | None = None,
    ) -> _FormalRegistryAttestation:
        if type(registry) is not _FormalToolRegistry:
            raise RegistryError(
                "formal registry requires reviewed concrete services from "
                "build_mvp_registry"
            )
        with self._lock:
            attestation = self._records.get(registry)
        if attestation is None:
            raise RegistryError("formal registry has no authority-issued attestation")
        if handle is not None and handle._nonce is not attestation.nonce:
            raise RegistryError("formal registry runtime handle attestation mismatch")
        tools = object.__getattribute__(registry, "_tools")
        manifest = object.__getattribute__(registry, "_manifest")
        if (
            id(tools) != attestation.tools_identity
            or id(manifest) != attestation.manifest_identity
            or tuple(sorted(tools.items())) != attestation.tool_entries
            or manifest.registry_sha256 != attestation.registry_sha256
            or canonical_json_bytes(manifest.model_dump(mode="json"))
            != attestation.manifest_bytes
        ):
            raise RegistryError("formal registry structure changed after attestation")
        if not _reviewed_runtime_members_are_unchanged():
            raise RegistryError("reviewed formal runtime class implementation changed")
        missing = _missing_formal_runtime_bindings(tools)
        if missing:
            raise RegistryError(
                "formal registry requires explicit runtime bindings for every tool: "
                + ", ".join(missing)
            )
        attestation.runtime_guard()
        return attestation


_FORMAL_RUNTIME_AUTHORITY = _FormalRuntimeAuthority()


def _formal_runtime_is_issued(registry: ToolRegistry) -> bool:
    return _FORMAL_RUNTIME_AUTHORITY.is_issued(registry)


def require_formal_registry_runtime(
    registry: ToolRegistry,
) -> FormalRegistryRuntimeHandle:
    """Return only the authority-held capability for a canonical registry."""

    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    return _FORMAL_RUNTIME_AUTHORITY.handle(registry)


def formal_registry_runtime_snapshot(
    registry: ToolRegistry,
) -> FormalRegistryRuntimeSnapshot:
    """Capture a canonical registry through the authority-held capability."""

    return require_formal_registry_runtime(registry).snapshot()


def _capture_formal_registry_runtime(
    registry: ToolRegistry,
    handle: FormalRegistryRuntimeHandle,
) -> FormalRegistryRuntimeSnapshot:
    handle._validate_registry(registry)
    tools = object.__getattribute__(registry, "_tools")
    tool_bindings = tuple(
        (name, _resolve_registered_runtime(tools[name], evidence=False))
        for name in sorted(tools)
    )
    evidence_bindings = tuple(
        (name, _resolve_registered_runtime(tools[name], evidence=True))
        for name in sorted(tools)
    )
    handle._validate_registry(registry)
    manifest = object.__getattribute__(registry, "_manifest")
    return FormalRegistryRuntimeSnapshot(
        registry_sha256=manifest.registry_sha256,
        registry_runtime_sha256=registry._registry_runtime_digest(
            tool_bindings, evidence_bindings
        ),
        tool_runtime_bindings=tool_bindings,
        evidence_runtime_bindings=evidence_bindings,
    )


def _missing_formal_runtime_bindings(
    tools: Mapping[str, RegisteredTool],
) -> list[str]:
    return sorted(
        name
        for name, tool in tools.items()
        if not tool.runtime_binding_explicit
        or tool.runtime_binding_resolver is None
        or tool.formal_evidence_validator is None
        or (
            name in {"object_detect", "document_ocr", "multi_product_search"}
            and (
                tool.evidence_runtime_binding_sha256 is None
                or tool.evidence_runtime_binding_resolver is None
            )
        )
    )


def publish_registry_manifest(
    registry: ToolRegistry,
    path: str | Path,
) -> Path:
    """Create, never replace, one canonical registry snapshot."""

    return atomic_create_file(
        path,
        canonical_json_bytes(registry.manifest.model_dump(mode="json")),
    )


def load_registry_manifest(
    path: str | Path,
    *,
    expected_registry: ToolRegistry | None = None,
) -> RegistryManifest:
    """Load a canonical registry snapshot and optionally bind it to live handlers."""

    content = read_stable_regular_file(path, label="tool registry manifest")
    raw = parse_canonical_json(content, label="tool registry manifest")
    if not isinstance(raw, dict):
        raise RegistryError("tool registry manifest must be an object")
    try:
        manifest_input = dict(raw)
        tools = manifest_input.get("tools")
        if not isinstance(tools, list):
            raise ValueError("tools must be an array")
        manifest_input["tools"] = tuple(
            ToolSpec.model_validate(tool, strict=True) for tool in tools
        )
        manifest = RegistryManifest.model_validate(manifest_input, strict=True)
    except ValidationError as error:
        raise RegistryError("tool registry manifest violates schema") from error
    if expected_registry is not None and manifest != expected_registry.manifest:
        raise RegistryError("tool registry manifest does not match live registry")
    return manifest


def build_mvp_registry(
    services: MVPToolServices,
    *,
    include_multi_product: bool = False,
    _allow_unconfigured: bool = False,
) -> ToolRegistry:
    """Wire one stable generation; only reviewed concrete services become formal."""

    from skillchain.tools.contracts import ProductSearchTrace
    from skillchain.tools.document_ocr import (
        DocumentOCRResult,
        DocumentSafetyApproval,
    )
    from skillchain.tools.kb_lookup import KBHit
    from skillchain.tools.multi_product import (
        MultiProductResult,
        MultiProductSearchService,
    )
    from skillchain.tools.object_detect import (
        ObjectDetectionResult,
    )

    product_service = services.product_search
    kb_service = services.kb_lookup
    detection_service = services.object_detection
    ocr_service = services.document_ocr
    supplied_multi_product = services.multi_product_search
    multi_product_service = (
        None
        if not include_multi_product
        else supplied_multi_product
        if supplied_multi_product is not None
        else None
        if _allow_unconfigured
        else MultiProductSearchService(detection_service, product_service)
    )
    safety_owner = getattr(services.safety_approval_for, "__self__", None)
    safety_function = getattr(services.safety_approval_for, "__func__", None)

    def embedding_backend_is_trusted() -> bool:
        wrapper = getattr(product_service, "backend", None)
        upstream = getattr(wrapper, "_backend", None)
        if type(wrapper) is not _REVIEWED_RUNTIME_TYPES["FormalEmbeddingBackend"]:
            return False
        if type(upstream) is _REVIEWED_RUNTIME_TYPES["DashScopeEmbeddingClient"]:
            members = (
                "_embed_batched",
                "_image_bytes_content",
                "_image_content",
                "_parse_response",
                "_request_embeddings",
                "_retry_delay",
                "_text_content",
                "embed_images",
                "embed_image_bytes",
                "embed_texts",
                "execution_location",
                "formal_runtime_binding_sha256",
                "model",
            )
        elif type(upstream) is _REVIEWED_RUNTIME_TYPES["OpenCLIPEmbeddingBackend"]:
            members = (
                "_load_config",
                "_load_runtime",
                "_runtime",
                "_to_numpy",
                "embed_images",
                "embed_image_bytes",
                "embed_texts",
                "execution_location",
                "formal_runtime_binding_sha256",
                "model",
            )
        else:
            return False
        return _has_no_instance_overrides(upstream, members)

    def formal_service_types_are_trusted() -> bool:
        base_trusted = (
            _reviewed_runtime_members_are_unchanged()
            and type(product_service) is _REVIEWED_RUNTIME_TYPES["ProductSearchService"]
            and type(getattr(product_service, "index", None))
            is _REVIEWED_RUNTIME_TYPES["ProductIndex"]
            and _has_no_instance_overrides(
                product_service.index,
                (
                    "canary_text",
                    "canary_vector",
                    "require_formal_verified",
                    "retrieval_binding",
                ),
            )
            and type(getattr(product_service, "backend", None))
            is _REVIEWED_RUNTIME_TYPES["FormalEmbeddingBackend"]
            and embedding_backend_is_trusted()
            and _has_no_instance_overrides(
                product_service,
                (
                    "execution_location",
                    "formal_runtime_binding_sha256",
                    "trace_image_product_search",
                    "trace_text_product_search",
                    "trace_similar_styles",
                ),
            )
            and _has_no_instance_overrides(
                product_service.backend,
                (
                    "_require_verified",
                    "canary_text",
                    "canary_vector",
                    "embed_images",
                    "embed_image_bytes",
                    "embed_texts",
                    "execution_location",
                    "formal_ready",
                    "runtime_binding_sha256",
                    "verify_canary",
                ),
            )
            and type(kb_service) is _REVIEWED_RUNTIME_TYPES["KBLookupService"]
            and _has_no_instance_overrides(
                kb_service,
                (
                    "artifact_binding_for",
                    "encyclopedia_lookup",
                    "formal_runtime_binding_sha256",
                    "recipe_lookup",
                ),
            )
            and type(detection_service)
            is _REVIEWED_RUNTIME_TYPES["ObjectDetectionService"]
            and type(getattr(detection_service, "backend", None))
            is _REVIEWED_RUNTIME_TYPES["UltralyticsDetectorBackend"]
            and _has_no_instance_overrides(
                detection_service, ("detect", "formal_runtime_binding_sha256")
            )
            and _has_no_instance_overrides(
                detection_service.backend,
                (
                    "_load_model",
                    "_model",
                    "detect",
                    "formal_runtime_binding_sha256",
                ),
            )
            and type(ocr_service) is _REVIEWED_RUNTIME_TYPES["DocumentOCRService"]
            and type(getattr(ocr_service, "backend", None))
            is _REVIEWED_RUNTIME_TYPES["RapidOCROnnxBackend"]
            and _has_no_instance_overrides(
                ocr_service, ("formal_runtime_binding_sha256", "ocr")
            )
            and _has_no_instance_overrides(
                ocr_service.backend,
                (
                    "_engine",
                    "_load_engine",
                    "formal_runtime_binding_sha256",
                    "recognize",
                ),
            )
            and getattr(ocr_service, "field_extractor", object()) is None
            and type(safety_owner) is _REVIEWED_RUNTIME_TYPES["DocumentSafetyCatalog"]
            and _has_no_instance_overrides(
                safety_owner,
                (
                    "approval_for",
                    "catalog_sha256",
                    "formal_runtime_binding_sha256",
                ),
            )
            and safety_function
            is _REVIEWED_RUNTIME_TYPES["DocumentSafetyCatalog"].approval_for
        )
        if not base_trusted:
            return False
        if not include_multi_product:
            return True
        return type(multi_product_service) is _REVIEWED_RUNTIME_TYPES[
            "MultiProductSearchService"
        ] and _has_no_instance_overrides(
            multi_product_service,
            ("execution_location", "formal_runtime_binding_sha256", "search"),
        )

    def require_trusted_formal_services() -> None:
        if not formal_service_types_are_trusted():
            raise RegistryError(
                "formal registry service graph changed or is not a reviewed runtime"
            )

    product_runtime = _service_runtime_binding(
        product_service, allow_unconfigured=_allow_unconfigured
    )
    kb_runtime = _service_runtime_binding(
        kb_service, allow_unconfigured=_allow_unconfigured
    )
    detection_runtime = _service_runtime_binding(
        detection_service, allow_unconfigured=_allow_unconfigured
    )
    ocr_runtime = _service_runtime_binding(
        ocr_service, allow_unconfigured=_allow_unconfigured
    )
    multi_product_runtime = (
        _service_runtime_binding(
            multi_product_service, allow_unconfigured=_allow_unconfigured
        )
        if include_multi_product
        else None
    )
    safety_runtime = _service_runtime_binding(
        services.safety_approval_for, allow_unconfigured=_allow_unconfigured
    )
    product_runtime_resolver = _service_runtime_resolver(
        product_service, allow_unconfigured=_allow_unconfigured
    )
    kb_runtime_resolver = _service_runtime_resolver(
        kb_service, allow_unconfigured=_allow_unconfigured
    )
    detection_runtime_resolver = _service_runtime_resolver(
        detection_service, allow_unconfigured=_allow_unconfigured
    )
    ocr_runtime_resolver = _service_runtime_resolver(
        ocr_service, allow_unconfigured=_allow_unconfigured
    )
    multi_product_runtime_resolver = (
        _service_runtime_resolver(
            multi_product_service, allow_unconfigured=_allow_unconfigured
        )
        if include_multi_product
        else None
    )
    safety_runtime_resolver = _service_runtime_resolver(
        services.safety_approval_for, allow_unconfigured=_allow_unconfigured
    )
    combined_ocr_runtime = (
        None
        if ocr_runtime is None or safety_runtime is None
        else sha256_bytes(
            canonical_json_bytes(
                {
                    "document_ocr_runtime_sha256": ocr_runtime,
                    "policy_version": "document-ocr-safety-composite-v1",
                    "safety_approval_runtime_sha256": safety_runtime,
                }
            )
        )
    )

    def combined_ocr_runtime_resolver() -> str:
        if ocr_runtime_resolver is None or safety_runtime_resolver is None:
            raise RegistryError("formal OCR runtime resolver is unavailable")
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "document_ocr_runtime_sha256": ocr_runtime_resolver(),
                    "policy_version": "document-ocr-safety-composite-v1",
                    "safety_approval_runtime_sha256": safety_runtime_resolver(),
                }
            )
        )

    detection_evidence_runtime = _service_evidence_runtime_binding(
        detection_service,
        allow_unconfigured=_allow_unconfigured,
    )
    ocr_evidence_runtime = _service_evidence_runtime_binding(
        ocr_service,
        allow_unconfigured=_allow_unconfigured,
    )
    detection_evidence_resolver = _service_evidence_runtime_resolver(
        detection_service,
        allow_unconfigured=_allow_unconfigured,
    )
    ocr_evidence_resolver = _service_evidence_runtime_resolver(
        ocr_service,
        allow_unconfigured=_allow_unconfigured,
    )

    def image_product_handler(value: BaseModel, context: ToolExecutionContext):
        asset_id = value.model_dump()["asset_id"]
        return product_service.trace_image_product_search(
            product_image_path(context, asset_id), asset_id=asset_id
        )

    def text_product_handler(value: BaseModel, _context: ToolExecutionContext):
        return product_service.trace_text_product_search(value.model_dump()["query"])

    def style_handler(value: BaseModel, context: ToolExecutionContext):
        arguments = value.model_dump()
        asset_id = arguments["asset_id"]
        query = arguments["query"]
        if getattr(product_service, "query_conditioned_style", False) is True:
            return product_service.trace_similar_styles(
                product_image_path(context, asset_id),
                asset_id=asset_id,
                query_text=query,
            )
        return product_service.trace_similar_styles(
            product_image_path(context, asset_id), asset_id=asset_id
        )

    def product_image_path(context: ToolExecutionContext, asset_id: str) -> Path:
        location = getattr(product_service, "execution_location", None)
        if location == "local":
            return context.asset_path(asset_id)
        if location == "remote":
            return context.remote_asset_path(asset_id)
        raise ToolCallError(
            "runtime_unconfigured",
            "product embedding execution location is unavailable",
        )

    def encyclopedia_handler(value: BaseModel, _context: ToolExecutionContext):
        return kb_service.encyclopedia_lookup(value.model_dump()["entity"])

    def recipe_handler(value: BaseModel, _context: ToolExecutionContext):
        return kb_service.recipe_lookup(value.model_dump()["dish"])

    def detection_handler(value: BaseModel, context: ToolExecutionContext):
        asset_id = value.model_dump()["asset_id"]
        return detection_service.detect(
            context.asset_path(asset_id),
            asset_id=asset_id,
        )

    def multi_product_handler(value: BaseModel, context: ToolExecutionContext):
        if multi_product_service is None:
            raise ToolCallError(
                "runtime_unconfigured",
                "multi-product runtime is not configured",
            )
        asset_id = value.model_dump()["asset_id"]
        return multi_product_service.search(
            product_image_path(context, asset_id),
            asset_id=asset_id,
        )

    def ocr_handler(value: BaseModel, context: ToolExecutionContext):
        asset_id = value.model_dump()["asset_id"]
        approval = services.safety_approval_for(context.query_id, asset_id)
        if not isinstance(approval, DocumentSafetyApproval):
            raise ToolCallError(
                "safety_gate_failed",
                "document safety approval is unavailable",
            )
        return ocr_service.ocr(
            context.asset_path(asset_id),
            asset_id=asset_id,
            safety_approval=approval,
        )

    def product_evidence_validator(value: object) -> None:
        if not isinstance(value, ProductSearchTrace):
            raise RegistryError("product evidence has the wrong type")
        expected = getattr(product_service, "artifact_binding", None)
        if expected is None or value.artifact_binding != expected:
            raise RegistryError("product evidence artifact binding mismatch")
        if any(hit.artifact_binding != expected for hit in value.hits):
            raise RegistryError("product hit artifact binding mismatch")

    def kb_evidence_validator(value: object, *, kind: str) -> None:
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, KBHit) for item in value
        ):
            raise RegistryError("KB evidence has the wrong type")
        expected = kb_service.artifact_binding_for(kind)
        if any(item.artifact_binding != expected for item in value):
            raise RegistryError("KB evidence artifact binding mismatch")

    def detection_evidence_validator(value: object) -> None:
        if not isinstance(value, ObjectDetectionResult):
            raise RegistryError("detection evidence has the wrong type")
        expected = _service_evidence_runtime_binding(
            detection_service, allow_unconfigured=False
        )
        if _model_runtime_digest(value.runtime_binding) != expected:
            raise RegistryError("detection evidence runtime mismatch")

    def multi_product_evidence_validator(value: object) -> None:
        if not isinstance(value, MultiProductResult):
            raise RegistryError("multi-product evidence has the wrong type")
        expected_product = getattr(product_service, "artifact_binding", None)
        if expected_product is None or value.artifact_binding != expected_product:
            raise RegistryError("multi-product retrieval binding mismatch")
        expected_detector = _service_evidence_runtime_binding(
            detection_service, allow_unconfigured=False
        )
        if _model_runtime_digest(value.detection_runtime_binding) != expected_detector:
            raise RegistryError("multi-product detector runtime mismatch")

    def ocr_evidence_validator(value: object) -> None:
        if not isinstance(value, DocumentOCRResult):
            raise RegistryError("OCR evidence has the wrong type")
        expected = _service_evidence_runtime_binding(
            ocr_service, allow_unconfigured=False
        )
        if _model_runtime_digest(value.runtime_binding) != expected:
            raise RegistryError("OCR evidence runtime mismatch")

    definitions: tuple[ToolDefinition, ...] = (
        ToolDefinition(
            name="image_product_search",
            tool_version="2.1.0",
            description="Search the audited product gallery using the query image.",
            input_model=AssetToolInput,
            output_type=ProductSearchTrace,
            input_binding="query_asset",
            network_policy="runtime_bound_embedding",
            output_trust="retrieval_evidence",
            handler=image_product_handler,
            runtime_binding_sha256=product_runtime,
            runtime_binding_resolver=product_runtime_resolver,
            formal_evidence_validator=product_evidence_validator,
        ),
        ToolDefinition(
            name="text_product_search",
            tool_version="2.1.0",
            description="Search the audited product gallery using exact query text.",
            input_model=TextProductSearchInput,
            output_type=ProductSearchTrace,
            input_binding="query_text_exact",
            network_policy="runtime_bound_embedding",
            output_trust="retrieval_evidence",
            handler=text_product_handler,
            runtime_binding_sha256=product_runtime,
            runtime_binding_resolver=product_runtime_resolver,
            formal_evidence_validator=product_evidence_validator,
        ),
        ToolDefinition(
            name="style_similar_search",
            tool_version="2.3.0",
            description=(
                "Return evidence-backed same-category alternatives or curated "
                "cross-category coordination candidates."
            ),
            input_model=StyleSimilarSearchInput,
            output_type=ProductSearchTrace,
            input_binding="query_asset_and_text_exact",
            network_policy="runtime_bound_embedding",
            output_trust="retrieval_evidence",
            handler=style_handler,
            runtime_binding_sha256=product_runtime,
            runtime_binding_resolver=product_runtime_resolver,
            formal_evidence_validator=product_evidence_validator,
        ),
        ToolDefinition(
            name="encyclopedia_lookup",
            tool_version="1.0.0",
            description="Retrieve source-grounded encyclopedia evidence.",
            input_model=EncyclopediaLookupInput,
            output_type=list[KBHit],
            input_binding="query_derived_text",
            network_policy="offline",
            output_trust="retrieval_evidence",
            handler=encyclopedia_handler,
            runtime_binding_sha256=kb_runtime,
            runtime_binding_resolver=kb_runtime_resolver,
            formal_evidence_validator=lambda value: kb_evidence_validator(
                value, kind="encyclopedia"
            ),
        ),
        ToolDefinition(
            name="recipe_lookup",
            tool_version="1.0.0",
            description="Retrieve source-grounded recipe evidence.",
            input_model=RecipeLookupInput,
            output_type=list[KBHit],
            input_binding="query_derived_text",
            network_policy="offline",
            output_trust="retrieval_evidence",
            handler=recipe_handler,
            runtime_binding_sha256=kb_runtime,
            runtime_binding_resolver=kb_runtime_resolver,
            formal_evidence_validator=lambda value: kb_evidence_validator(
                value, kind="recipe"
            ),
        ),
        ToolDefinition(
            name="object_detect",
            tool_version="1.0.0",
            description="Detect normalized objects in the authoritative query image.",
            input_model=AssetToolInput,
            output_type=ObjectDetectionResult,
            input_binding="query_asset",
            network_policy="offline",
            output_trust="model_prediction",
            handler=detection_handler,
            runtime_binding_sha256=detection_runtime,
            runtime_binding_resolver=detection_runtime_resolver,
            evidence_runtime_binding_sha256=detection_evidence_runtime,
            evidence_runtime_binding_resolver=detection_evidence_resolver,
            formal_evidence_validator=detection_evidence_validator,
        ),
        ToolDefinition(
            name="document_ocr",
            tool_version="1.0.0",
            description="Transcribe an approved document image with line evidence.",
            input_model=AssetToolInput,
            output_type=DocumentOCRResult,
            input_binding="query_asset",
            network_policy="offline",
            output_trust="untrusted_document_text",
            handler=ocr_handler,
            runtime_binding_sha256=combined_ocr_runtime,
            runtime_binding_resolver=(
                None if _allow_unconfigured else combined_ocr_runtime_resolver
            ),
            evidence_runtime_binding_sha256=ocr_evidence_runtime,
            evidence_runtime_binding_resolver=ocr_evidence_resolver,
            formal_evidence_validator=ocr_evidence_validator,
        ),
    )
    if include_multi_product:
        definitions = (
            *definitions,
            ToolDefinition(
                name="multi_product_search",
                tool_version="1.0.0",
                description=(
                    "Detect visible items in the authoritative query image and "
                    "retrieve product evidence for each private canonical crop."
                ),
                input_model=AssetToolInput,
                output_type=MultiProductResult,
                input_binding="query_asset",
                network_policy="runtime_bound_embedding",
                output_trust="mixed_prediction_and_retrieval_evidence",
                handler=multi_product_handler,
                runtime_binding_sha256=multi_product_runtime,
                runtime_binding_resolver=multi_product_runtime_resolver,
                evidence_runtime_binding_sha256=multi_product_runtime,
                evidence_runtime_binding_resolver=multi_product_runtime_resolver,
                formal_evidence_validator=multi_product_evidence_validator,
            ),
        )
    trusted = not _allow_unconfigured and formal_service_types_are_trusted()
    registry: ToolRegistry
    if trusted:
        registry = _FormalToolRegistry(definitions)
        _FORMAL_RUNTIME_AUTHORITY.issue(
            registry,
            require_trusted_formal_services,
            builder_token=_CANONICAL_MVP_BUILDER_TOKEN,
        )
    else:
        registry = DiagnosticToolRegistry(definitions)
    return registry


def build_mvp_registry_spec(*, include_multi_product: bool = False) -> ToolRegistry:
    """Build introspection-only specs without loading indexes or model runtimes."""

    class _UnavailableService:
        def __getattr__(self, _name: str):
            def unavailable(*_args, **_kwargs):
                raise ToolCallError(
                    "runtime_unconfigured",
                    "tool runtime is not configured",
                )

            return unavailable

    unavailable = _UnavailableService()
    return build_mvp_registry(
        MVPToolServices(
            product_search=unavailable,
            kb_lookup=unavailable,
            object_detection=unavailable,
            document_ocr=unavailable,
            safety_approval_for=lambda _query_id, _asset_id: None,
        ),
        include_multi_product=include_multi_product,
        _allow_unconfigured=True,
    )


def _service_runtime_binding(
    service: object, *, allow_unconfigured: bool
) -> str | None:
    try:
        binding = getattr(service, "formal_runtime_binding_sha256")
    except (AttributeError, TypeError, ValueError):
        binding = None
    if binding is None:
        # Bound resolver methods carry the reviewed catalog/service on
        # ``__self__``; use that explicit identity rather than deriving one
        # from Python bytecode.
        try:
            binding = getattr(
                getattr(service, "__self__"),
                "formal_runtime_binding_sha256",
            )
        except (AttributeError, TypeError, ValueError):
            binding = None
    if (
        isinstance(binding, str)
        and len(binding) == 64
        and all(character in "0123456789abcdef" for character in binding)
    ):
        return binding
    if allow_unconfigured:
        return None
    raise RegistryError(
        "formal MVP registry service lacks formal_runtime_binding_sha256: "
        f"{type(service).__module__}.{type(service).__qualname__}"
    )


def _has_no_instance_overrides(service: object, names: tuple[str, ...]) -> bool:
    """Reject per-instance callables that bypass the reviewed concrete class."""

    try:
        state = vars(service)
    except TypeError:
        return False
    return all(name not in state for name in names)


def _service_runtime_resolver(
    service: object, *, allow_unconfigured: bool
) -> RuntimeBindingResolver | None:
    if allow_unconfigured:
        return None

    def resolve() -> str:
        value = _service_runtime_binding(service, allow_unconfigured=False)
        assert value is not None
        return value

    # Fail at construction as well as at every later formal check.
    resolve()
    return resolve


def _model_runtime_digest(value: object) -> str:
    if not isinstance(value, BaseModel):
        raise RegistryError("evidence runtime binding must be a typed model")
    return sha256_bytes(canonical_json_bytes(value.model_dump(mode="json")))


def _service_evidence_runtime_binding(
    service: object, *, allow_unconfigured: bool
) -> str | None:
    try:
        artifact = getattr(service, "artifact")
        binding = getattr(artifact, "runtime_binding")
        value = _model_runtime_digest(binding)
    except (AttributeError, TypeError, ValueError, RegistryError):
        if allow_unconfigured:
            return None
        raise RegistryError(
            "formal model service lacks a typed artifact runtime binding"
        ) from None
    return value


def _service_evidence_runtime_resolver(
    service: object, *, allow_unconfigured: bool
) -> RuntimeBindingResolver | None:
    if allow_unconfigured:
        return None

    def resolve() -> str:
        value = _service_evidence_runtime_binding(service, allow_unconfigured=False)
        assert value is not None
        return value

    resolve()
    return resolve


def _optional_runtime_binding(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RegistryError(f"{label} must be a lowercase sha256")
    return value


def _resolve_registered_runtime(
    registered: RegisteredTool, *, evidence: bool
) -> str | None:
    expected = (
        registered.evidence_runtime_binding_sha256
        if evidence
        else registered.runtime_binding_sha256
    )
    resolver = (
        registered.evidence_runtime_binding_resolver
        if evidence
        else registered.runtime_binding_resolver
    )
    if resolver is None:
        return expected
    try:
        current = resolver()
    except Exception as error:
        raise RegistryError("live runtime binding cannot be resolved") from error
    current = _optional_runtime_binding(current, "live runtime binding")
    if current != expected:
        raise RegistryError("live runtime binding changed after registry construction")
    return current


def _handler_implementation_binding(handler: ToolHandler) -> str:
    code = getattr(handler, "__code__", None)
    payload = {
        "code_sha256": sha256_bytes(code.co_code) if code is not None else None,
        "constants": repr(code.co_consts) if code is not None else None,
        "handler": (
            f"{getattr(handler, '__module__', '')}."
            f"{getattr(handler, '__qualname__', '')}"
        ),
        "policy_version": "handler-implementation-v1",
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _runtime_handler_binding(definition: ToolDefinition) -> str:
    explicit = definition.runtime_binding_sha256
    if explicit is not None:
        if len(explicit) != 64 or any(
            character not in "0123456789abcdef" for character in explicit
        ):
            raise RegistryError("runtime_binding_sha256 must be a lowercase sha256")
        return explicit
    code = getattr(definition.handler, "__code__", None)
    payload = {
        "code_sha256": sha256_bytes(code.co_code) if code is not None else None,
        "handler": (
            f"{getattr(definition.handler, '__module__', '')}."
            f"{getattr(definition.handler, '__qualname__', '')}"
        ),
        "policy_version": "derived-handler-runtime-v1",
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _build_manifest(specs: tuple[ToolSpec, ...]) -> RegistryManifest:
    ordered = tuple(sorted(specs, key=lambda spec: spec.name))
    names = {item.name for item in ordered}
    schema_version = 1 if names == MVP_TOOL_NAMES else 2
    tool_count = len(ordered)
    payload = {
        "schema_version": schema_version,
        "tool_count": tool_count,
        "tools": [spec.model_dump(mode="json") for spec in ordered],
    }
    return RegistryManifest.model_validate(
        {
            "schema_version": schema_version,
            "tool_count": tool_count,
            "tools": ordered,
            "registry_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


def _digest_without_field(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _assert_recursively_strict_schema(schema: dict[str, Any], label: str) -> None:
    """Require every typed object node to reject unknown fields."""

    stack: list[object] = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if (
                node.get("type") == "object"
                and node.get("additionalProperties") is not False
            ):
                raise RegistryError(f"{label} contains a non-strict object schema")
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _capture_reviewed_runtime_contract() -> tuple[
    dict[str, type[Any]], tuple[tuple[type[Any], str, object], ...]
]:
    """Capture reviewed class identities before callers can replace implementations."""

    from skillchain.tools.document_ocr import (
        DocumentOCRService,
        RapidOCROnnxBackend,
    )
    from skillchain.tools.document_safety import DocumentSafetyCatalog
    from skillchain.tools.embedding import (
        DashScopeEmbeddingClient,
        FormalEmbeddingBackend,
        OpenCLIPEmbeddingBackend,
    )
    from skillchain.tools.kb_lookup import KBLookupService
    from skillchain.tools.multi_product import MultiProductSearchService
    from skillchain.tools.object_detect import (
        ObjectDetectionService,
        UltralyticsDetectorBackend,
    )
    from skillchain.tools.product_index import ProductIndex
    from skillchain.tools.product_search import ProductSearchService

    types = {
        cls.__name__: cls
        for cls in (
            DashScopeEmbeddingClient,
            DocumentOCRService,
            DocumentSafetyCatalog,
            FormalEmbeddingBackend,
            KBLookupService,
            MultiProductSearchService,
            ObjectDetectionService,
            OpenCLIPEmbeddingBackend,
            ProductIndex,
            ProductSearchService,
            RapidOCROnnxBackend,
            UltralyticsDetectorBackend,
        )
    }
    reviewed_classes = (ToolRegistry, *types.values())
    members: list[tuple[type[Any], str, object]] = []
    for cls in reviewed_classes:
        for name, value in vars(cls).items():
            if name.startswith("__"):
                continue
            if callable(value) or isinstance(
                value, (classmethod, property, staticmethod)
            ):
                members.append((cls, name, inspect.getattr_static(cls, name)))
    return types, tuple(members)


def _reviewed_runtime_members_are_unchanged() -> bool:
    for cls, name, expected in _REVIEWED_RUNTIME_MEMBERS:
        try:
            current = inspect.getattr_static(cls, name)
        except AttributeError:
            return False
        if current is not expected:
            return False
    return True


def _reviewed_member_source_sha256(value: object) -> str:
    """Bind formal runtime identity to reviewed method/property source text."""

    if isinstance(value, (classmethod, staticmethod)):
        callables = (value.__func__,)
    elif isinstance(value, property):
        callables = tuple(
            member
            for member in (value.fget, value.fset, value.fdel)
            if member is not None
        )
    else:
        callables = (value,)
    try:
        sources = tuple(inspect.getsource(member) for member in callables)
    except (OSError, TypeError) as error:
        raise RegistryError(
            "reviewed formal runtime source cannot be fingerprinted"
        ) from error
    return sha256_bytes(canonical_json_bytes({"sources": list(sources)}))


_REVIEWED_RUNTIME_TYPES, _REVIEWED_RUNTIME_MEMBERS = (
    _capture_reviewed_runtime_contract()
)
_REVIEWED_RUNTIME_IMPLEMENTATION_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "members": [
                {
                    "class": f"{cls.__module__}.{cls.__qualname__}",
                    "member": name,
                    "source_sha256": _reviewed_member_source_sha256(value),
                }
                for cls, name, value in _REVIEWED_RUNTIME_MEMBERS
            ],
            "policy_version": "reviewed-runtime-implementation-v1",
        }
    )
)


__all__ = [
    "AssetToolInput",
    "DiagnosticToolRegistry",
    "EncyclopediaLookupInput",
    "FormalRegistryRuntimeHandle",
    "FormalRegistryRuntimeSnapshot",
    "MVP_TOOL_NAMES",
    "MVP_TOOL_NAMES_V2",
    "MVPToolServices",
    "RecipeLookupInput",
    "RegisteredTool",
    "RegistryError",
    "RegistryManifest",
    "StyleSimilarSearchInput",
    "TextProductSearchInput",
    "ToolCallError",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolInvocationResult",
    "ToolRegistry",
    "ToolSpec",
    "build_mvp_registry",
    "build_mvp_registry_spec",
    "formal_registry_runtime_snapshot",
    "load_registry_manifest",
    "publish_registry_manifest",
    "require_formal_registry_runtime",
]
