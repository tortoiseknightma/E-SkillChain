"""Prospective Codex-CLI static-authoring contracts.

This module deliberately does not extend the API-provider authoring contract.
Codex CLI does not expose a provider request ID, served revision, provider-side
attempt count, or a per-call price receipt.  Its evidence is therefore tracked
as a separately named, platform-mediated session.
"""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import sys
from typing import Annotated, Literal, Self

from packaging.requirements import InvalidRequirement, Requirement
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.static_authoring import (
    AuthoringContentOutputContract,
    AuthoringContentPayload,
    AuthoringDraftBundle,
    AuthoringInput,
    CanonicalSpecification,
    CompilerIdentity,
    PromptIdentity,
    PublicSourceMaterial,
    ReferenceSkillBundle,
    build_authoring_content_output_contract,
    build_authoring_draft_bundle,
    normalize_authoring_content_payload,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CODEX_AUTHOR_RUN_ID = "llm-static-codex-primary-20260724-high-v2"
CODEX_EXECUTION_TYPE = "codex_mediated_static_author_v1"
CODEX_EVIDENCE_TIER = "platform-mediated_non-provider-attested"
CODEX_ENV_ALLOWLIST = (
    "ALL_PROXY",
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
CODEX_TRUSTED_SOURCE_FILES = (
    "scripts/run_codex_authoring.py",
    "src/skillchain/__init__.py",
    "src/skillchain/codex_authoring.py",
    "src/skillchain/config.py",
    "src/skillchain/llm.py",
    "src/skillchain/schemas.py",
    "src/skillchain/static_authoring.py",
    "src/skillchain/synthesis/__init__.py",
    "src/skillchain/synthesis/store.py",
    "src/skillchain/task_spec.py",
    "src/skillchain/taxonomy.py",
    "src/skillchain/tools/__init__.py",
    "src/skillchain/tools/registry.py",
    "src/skillchain/tools/serialization.py",
)
CODEX_RUNTIME_DISTRIBUTIONS = (
    "anthropic",
    "openai",
    "packaging",
    "pydantic",
    "python-dotenv",
)
CODEX_COMMAND_SHAPE = (
    "codex.exe",
    "exec",
    "--model",
    "gpt-5.6-sol",
    "--sandbox",
    "read-only",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--strict-config",
    "--skip-git-repo-check",
    "--cd",
    "<external-empty-scratch>",
    "--output-schema",
    "<frozen-schema>",
    "--json",
    "--output-last-message",
    "<create-only-final>",
    "--config",
    'model_reasoning_effort="high"',
    "-",
)

_STDIN_PREFIX = (
    "Act only as the static Skill author described by the frozen request below.\n"
    "Do not inspect the filesystem, call tools, search the web, ask questions, "
    "or add commentary.\n"
    "Use only bytes in <authoring_request>. Return exactly one JSON object that "
    "matches output_contract; an invalid final response permanently fails this "
    "one-session authorization.\n"
    "<authoring_request>\n"
).encode("utf-8")
_STDIN_SUFFIX = b"\n</authoring_request>\n"


class CodexAuthoringContractError(ValueError):
    """A Codex authoring artifact failed closed."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _digest_without(model: BaseModel, field_name: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(payload))


class CodexModelCatalogEvidence(_StrictFrozenModel):
    requested_model: Literal["gpt-5.6-sol"]
    supported_in_api: Literal[True]
    supported_reasoning_efforts: tuple[
        Literal["low", "medium", "high", "xhigh", "max", "ultra"], ...
    ]
    visibility: Literal["list"]

    @field_validator("supported_reasoning_efforts", mode="before")
    @classmethod
    def coerce_efforts(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_efforts(self) -> Self:
        if self.supported_reasoning_efforts != tuple(
            sorted(set(self.supported_reasoning_efforts))
        ):
            raise ValueError("Codex reasoning efforts must be sorted and unique")
        if "high" not in self.supported_reasoning_efforts:
            raise ValueError("Codex catalog does not expose high reasoning")
        return self


class CodexModelAccessEvidence(_StrictFrozenModel):
    schema_version: Literal[2]
    checked_on: str
    method: str
    cli_version: Literal["0.145.0"]
    binary: dict[str, object]
    auth: dict[str, object]
    model_catalog: CodexModelCatalogEvidence
    connectivity: dict[str, object]
    evidence_limitations: tuple[str, ...]

    @field_validator("evidence_limitations", mode="before")
    @classmethod
    def coerce_limitations(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.binary != {
            "bytes": 359245096,
            "discovery": "first codex.exe on PATH",
            "sha256": (
                "83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c"
            ),
        }:
            raise ValueError("Codex binary evidence drifted")
        if self.auth != {
            "configured": True,
            "mode": "chatgpt",
            "secret_material_recorded": False,
        }:
            raise ValueError("Codex auth evidence drifted")
        expected_connectivity = {
            "chatgpt_http_required_probe": "reachable_http_403",
            "inference_request_performed": False,
            "inference_success_verified": False,
            "overall_doctor_status": "ok",
            "responses_websocket_handshake": "succeeded",
        }
        if self.connectivity != expected_connectivity:
            raise ValueError("Codex connectivity evidence drifted")
        return self


class VerifiedCodexModelAccessEvidence:
    """Opaque-enough immutable evidence handle used by the packet builder."""

    __slots__ = ("value", "file_sha256", "path")

    def __init__(
        self,
        *,
        value: CodexModelAccessEvidence,
        file_sha256: str,
        path: Path,
    ) -> None:
        self.value = value
        self.file_sha256 = file_sha256
        self.path = path


def load_codex_model_access_evidence(
    path: str | Path, *, expected_file_sha256: str
) -> VerifiedCodexModelAccessEvidence:
    try:
        content = read_stable_regular_file(path, label="Codex model-access evidence")
        if sha256_bytes(content) != expected_file_sha256:
            raise CodexAuthoringContractError(
                "Codex model-access evidence digest mismatch"
            )
        raw = parse_canonical_json(content, label="Codex model-access evidence")
        if not isinstance(raw, dict):
            raise CodexAuthoringContractError(
                "Codex model-access evidence must contain an object"
            )
        value = CodexModelAccessEvidence.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise CodexAuthoringContractError(
            "Codex model-access evidence is invalid"
        ) from error
    return VerifiedCodexModelAccessEvidence(
        value=value,
        file_sha256=expected_file_sha256,
        path=Path(path).resolve(strict=True),
    )


class CodexAuthorModelIdentity(_StrictFrozenModel):
    execution_surface: Literal["codex_cli"] = "codex_cli"
    requested_model: Literal["gpt-5.6-sol"]
    reasoning_effort: Literal["high"]
    served_model: None = None
    served_revision: None = None
    provider_request_id: None = None
    identity_evidence: Literal["requested_and_catalog_visible_only"] = (
        "requested_and_catalog_visible_only"
    )


class CodexSessionBudget(_StrictFrozenModel):
    max_exec_sessions: Literal[1] = 1
    max_followup_sessions: Literal[0] = 0
    max_repository_retries: Literal[0] = 0
    max_repairs: Literal[0] = 0
    max_fallbacks: Literal[0] = 0
    max_visible_tool_activities: Literal[0] = 0
    max_accepted_final_outputs: Literal[1] = 1
    timeout_seconds: Literal[600] = 600
    max_final_output_bytes: Literal[65536] = 65536
    max_event_log_bytes: Literal[16777216] = 16777216
    max_stderr_bytes: Literal[1048576] = 1048576
    max_human_review_minutes: Literal[30] = 30
    comparability_input_token_target: Literal[30000] = 30000
    comparability_output_token_target: Literal[6000] = 6000
    comparability_total_token_target: Literal[36000] = 36000
    comparability_cost_ceiling_microusd: Literal[3000] = 3000
    token_and_cost_enforcement: Literal["unavailable_on_codex_cli"] = (
        "unavailable_on_codex_cli"
    )


class CodexAuthoringInput(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    status: Literal["frozen_candidate_not_authorized"] = (
        "frozen_candidate_not_authorized"
    )
    consumers: tuple[Literal["llm_static", "s1"], ...]
    execution_type: Literal["codex_mediated_static_author_v1"]
    evidence_tier: Literal["platform-mediated_non-provider-attested"]
    taxonomy: CanonicalSpecification
    task_specification: CanonicalSpecification
    tool_registry: CanonicalSpecification
    tool_registry_runtime_binding: Literal["deferred_until_bank_compile"]
    tool_registry_runtime_sha256: None
    prompt: PromptIdentity
    compiler: CompilerIdentity
    model: CodexAuthorModelIdentity
    session_budget: CodexSessionBudget
    public_sources: tuple[PublicSourceMaterial, ...]
    reference_skill_bundle: ReferenceSkillBundle | None
    semantic_source_packet_file_sha256: Sha256
    model_access_evidence_file_sha256: Sha256
    input_isolation: Literal["behaviorally_constrained_not_mechanically_proven"]
    formal_provider_call_eligible: Literal[False] = False
    input_sha256: Sha256

    @field_validator("consumers", "public_sources", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if self.consumers != ("llm_static", "s1"):
            raise ValueError("Codex authoring consumers drifted")
        if self.compiler.compiler_version != "4.0.0":
            raise ValueError("Codex authoring requires compiler v4")
        if self.public_sources or self.reference_skill_bundle is not None:
            raise ValueError("the approved Codex candidate freezes empty external input")
        if self.input_sha256 != _digest_without(self, "input_sha256"):
            raise ValueError("Codex authoring input digest mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class CodexAuthoringOutputContract(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    mode: Literal["codex_cli_output_schema_single_final"] = (
        "codex_cli_output_schema_single_final"
    )
    authoring_input_sha256: Sha256
    schema_name: Literal["skillchain_authoring_content_v2"]
    json_schema_canonical_json: str
    json_schema_sha256: Sha256
    semantic_schema_sha256: Sha256
    projection_policy: Literal[
        "openai_structured_outputs_subset_v1_local_semantic_revalidation"
    ]
    cli_schema_enforcement: Literal["requested_not_provider_attested"] = (
        "requested_not_provider_attested"
    )
    final_message_policy: Literal["one_json_object_no_commentary"]
    visible_tool_activity_allowed: Literal[False] = False
    runner_strict_validation: Literal[True] = True
    contract_sha256: Sha256

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        content = self.json_schema_canonical_json.encode("utf-8")
        try:
            raw = parse_canonical_json(content, label="Codex output JSON Schema")
        except ArtifactFormatError as error:
            raise ValueError("Codex output JSON Schema is not canonical") from error
        if (
            not isinstance(raw, dict)
            or raw.get("type") != "object"
            or sha256_bytes(content) != self.json_schema_sha256
            or self.contract_sha256 != _digest_without(self, "contract_sha256")
        ):
            raise ValueError("Codex output contract drifted")
        validate_codex_cli_output_schema(raw)
        return self


class CanonicalCodexAuthoringRequest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    authoring_input: CodexAuthoringInput
    output_contract: CodexAuthoringOutputContract
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if (
            self.output_contract.authoring_input_sha256
            != self.authoring_input.input_sha256
            or self.request_sha256 != _digest_without(self, "request_sha256")
        ):
            raise ValueError("Codex authoring request drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_codex_authoring_input(
    *,
    semantic_source: AuthoringInput,
    semantic_source_packet_file_sha256: str,
    model_access_evidence: VerifiedCodexModelAccessEvidence,
) -> CodexAuthoringInput:
    if (
        semantic_source.schema_version != 4
        or semantic_source.compiler.compiler_version != "4.0.0"
        or semantic_source.tool_registry_runtime_binding
        != "deferred_until_bank_compile"
        or semantic_source.public_sources
        or semantic_source.reference_skill_bundle is not None
    ):
        raise CodexAuthoringContractError(
            "semantic source is not the empty-source author-content v2 contract"
        )
    if (
        config.AUTHOR_PROVIDER,
        config.AUTHOR_MODEL,
        config.AUTHOR_REASONING_EFFORT,
    ) != ("codex_internal", "gpt-5.6-sol", "high"):
        raise CodexAuthoringContractError("configured Codex Author selection drifted")
    payload = {
        "schema_version": 1,
        "status": "frozen_candidate_not_authorized",
        "consumers": ["llm_static", "s1"],
        "execution_type": CODEX_EXECUTION_TYPE,
        "evidence_tier": CODEX_EVIDENCE_TIER,
        "taxonomy": semantic_source.taxonomy.model_dump(mode="json"),
        "task_specification": semantic_source.task_specification.model_dump(mode="json"),
        "tool_registry": semantic_source.tool_registry.model_dump(mode="json"),
        "tool_registry_runtime_binding": "deferred_until_bank_compile",
        "tool_registry_runtime_sha256": None,
        "prompt": semantic_source.prompt.model_dump(mode="json"),
        "compiler": semantic_source.compiler.model_dump(mode="json"),
        "model": CodexAuthorModelIdentity(
            requested_model="gpt-5.6-sol",
            reasoning_effort="high",
        ).model_dump(mode="json"),
        "session_budget": CodexSessionBudget().model_dump(mode="json"),
        "public_sources": [],
        "reference_skill_bundle": None,
        "semantic_source_packet_file_sha256": (
            semantic_source_packet_file_sha256
        ),
        "model_access_evidence_file_sha256": model_access_evidence.file_sha256,
        "input_isolation": "behaviorally_constrained_not_mechanically_proven",
        "formal_provider_call_eligible": False,
    }
    return CodexAuthoringInput.model_validate(
        {**payload, "input_sha256": sha256_bytes(canonical_json_bytes(payload))},
        strict=True,
    )


def build_codex_output_contract(
    *,
    authoring_input: CodexAuthoringInput,
    semantic_source: AuthoringInput,
) -> CodexAuthoringOutputContract:
    semantic_contract: AuthoringContentOutputContract = (
        build_authoring_content_output_contract(semantic_source)
    )
    semantic_schema_text = semantic_contract.json_schema_canonical_json
    try:
        semantic_schema = parse_canonical_json(
            semantic_schema_text.encode("utf-8"),
            label="semantic authoring output JSON Schema",
        )
    except ArtifactFormatError as error:
        raise CodexAuthoringContractError(
            "semantic authoring output JSON Schema is invalid"
        ) from error
    if not isinstance(semantic_schema, dict):
        raise CodexAuthoringContractError(
            "semantic authoring output JSON Schema must contain an object"
        )
    projected_schema = project_codex_cli_output_schema(semantic_schema)
    validate_codex_cli_output_schema(projected_schema)
    schema_text = canonical_json_bytes(projected_schema).decode("utf-8")
    payload = {
        "schema_version": 2,
        "mode": "codex_cli_output_schema_single_final",
        "authoring_input_sha256": authoring_input.input_sha256,
        "schema_name": "skillchain_authoring_content_v2",
        "json_schema_canonical_json": schema_text,
        "json_schema_sha256": sha256_bytes(schema_text.encode("utf-8")),
        "semantic_schema_sha256": sha256_bytes(
            semantic_schema_text.encode("utf-8")
        ),
        "projection_policy": (
            "openai_structured_outputs_subset_v1_local_semantic_revalidation"
        ),
        "cli_schema_enforcement": "requested_not_provider_attested",
        "final_message_policy": "one_json_object_no_commentary",
        "visible_tool_activity_allowed": False,
        "runner_strict_validation": True,
    }
    return CodexAuthoringOutputContract.model_validate(
        {**payload, "contract_sha256": sha256_bytes(canonical_json_bytes(payload))},
        strict=True,
    )


def project_codex_cli_output_schema(
    semantic_schema: dict[str, object],
) -> dict[str, object]:
    """Project the strict local schema onto the CLI-supported JSON subset.

    Representation constraints omitted here remain mandatory in
    ``normalize_codex_authoring_output`` and the trusted compiler.  This
    projection only prevents an interface-level schema rejection before the
    single authorized model session can start.
    """

    def project(value: object) -> object:
        if isinstance(value, list):
            return [project(item) for item in value]
        if not isinstance(value, dict):
            return value
        result: dict[str, object] = {}
        for key, child in value.items():
            if key in {"$schema", "title", "uniqueItems", "minLength"}:
                continue
            if key == "oneOf":
                if "anyOf" in value:
                    raise CodexAuthoringContractError(
                        "semantic schema mixes oneOf and anyOf"
                    )
                result["anyOf"] = project(child)
                continue
            if key == "const":
                result["enum"] = [project(child)]
                if "type" not in value:
                    inferred_type = (
                        "boolean"
                        if isinstance(child, bool)
                        else "integer"
                        if isinstance(child, int)
                        else "number"
                        if isinstance(child, float)
                        else "string"
                        if isinstance(child, str)
                        else None
                    )
                    if inferred_type is None:
                        raise CodexAuthoringContractError(
                            "Codex schema cannot infer the type of a const"
                        )
                    result["type"] = inferred_type
                continue
            result[key] = project(child)
        return result

    projected = project(semantic_schema)
    if not isinstance(projected, dict):
        raise CodexAuthoringContractError("projected Codex schema is not an object")
    return projected


def validate_codex_cli_output_schema(schema: dict[str, object]) -> None:
    """Fail closed on keywords outside the frozen Structured Outputs subset."""

    allowed = {
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "items",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "type",
    }

    def visit(node: object, path: str, depth: int) -> tuple[int, int]:
        if not isinstance(node, dict):
            raise CodexAuthoringContractError(
                f"Codex output schema node is not an object: {path}"
            )
        unknown = sorted(set(node) - allowed)
        if unknown:
            raise CodexAuthoringContractError(
                f"unsupported Codex output schema keywords at {path}: {unknown}"
            )
        if depth > 10:
            raise CodexAuthoringContractError(
                "Codex output schema exceeds the frozen nesting limit"
            )
        property_count = 0
        string_budget = 0
        node_type = node.get("type")
        if node_type not in {
            "array",
            "boolean",
            "integer",
            "number",
            "object",
            "string",
        } and "anyOf" not in node:
            raise CodexAuthoringContractError(
                f"Codex schema node lacks a supported type: {path}"
            )
        properties = node.get("properties")
        if node_type == "object":
            if not isinstance(properties, dict):
                raise CodexAuthoringContractError(
                    f"Codex object schema lacks properties: {path}"
                )
            required = node.get("required")
            if (
                not isinstance(required, list)
                or set(required) != set(properties)
                or len(required) != len(properties)
                or node.get("additionalProperties") is not False
            ):
                raise CodexAuthoringContractError(
                    f"Codex object schema is not closed and fully required: {path}"
                )
            property_count += len(properties)
            string_budget += sum(len(str(name)) for name in properties)
            for name, child in properties.items():
                child_properties, child_strings = visit(
                    child, f"{path}.properties[{name!r}]", depth + 1
                )
                property_count += child_properties
                string_budget += child_strings
        elif properties is not None or "required" in node or "additionalProperties" in node:
            raise CodexAuthoringContractError(
                f"non-object Codex schema has object keywords: {path}"
            )
        if node_type == "array":
            if "items" not in node:
                raise CodexAuthoringContractError(
                    f"Codex array schema lacks items: {path}"
                )
            child_properties, child_strings = visit(
                node["items"], f"{path}.items", depth + 1
            )
            property_count += child_properties
            string_budget += child_strings
        elif "items" in node or "minItems" in node or "maxItems" in node:
            raise CodexAuthoringContractError(
                f"non-array Codex schema has array keywords: {path}"
            )
        if "anyOf" in node:
            branches = node["anyOf"]
            if not isinstance(branches, list) or not branches:
                raise CodexAuthoringContractError(
                    f"Codex anyOf is empty or invalid: {path}"
                )
            for index, branch in enumerate(branches):
                child_properties, child_strings = visit(
                    branch, f"{path}.anyOf[{index}]", depth + 1
                )
                property_count += child_properties
                string_budget += child_strings
        enum = node.get("enum")
        if enum is not None:
            if not isinstance(enum, list) or not enum:
                raise CodexAuthoringContractError(
                    f"Codex enum is empty or invalid: {path}"
                )
            string_budget += sum(
                len(item) for item in enum if isinstance(item, str)
            )
        return property_count, string_budget

    if schema.get("type") != "object" or "anyOf" in schema:
        raise CodexAuthoringContractError(
            "Codex output schema root must be an object without anyOf"
        )
    properties, strings = visit(schema, "$", 1)
    if properties > 5000 or strings > 120000:
        raise CodexAuthoringContractError(
            "Codex output schema exceeds the frozen size limits"
        )


def build_codex_runtime_dependency_snapshot(
    repository_root: str | Path,
) -> dict[str, object]:
    """Return exact local source, interpreter, lock, and package identities."""

    root = Path(repository_root).resolve(strict=True)

    def binding(relative: str) -> dict[str, object]:
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise CodexAuthoringContractError(
                f"trusted runtime source is not a regular repository file: {relative}"
            )
        content = read_stable_regular_file(path, label=f"runtime dependency {relative}")
        return {
            "file": relative,
            "bytes": len(content),
            "file_sha256": sha256_bytes(content),
        }

    def executable(path_value: str) -> dict[str, object]:
        path = Path(path_value).resolve(strict=True)
        before = path.stat()
        if not path.is_file() or path.is_symlink():
            raise CodexAuthoringContractError(
                "Python runtime executable is not a regular file"
            )
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            content = handle.read()
            after_read = os.fstat(handle.fileno())
        after = path.stat()
        snapshots = (opened, after_read, after)
        if any(
            (
                snapshot.st_size,
                snapshot.st_mtime_ns,
                getattr(snapshot, "st_ino", 0),
            )
            != (
                before.st_size,
                before.st_mtime_ns,
                getattr(before, "st_ino", 0),
            )
            for snapshot in snapshots
        ):
            raise CodexAuthoringContractError(
                "Python runtime executable changed during hashing"
            )
        return {
            "path": str(path),
            "bytes": len(content),
            "file_sha256": sha256_bytes(content),
        }

    def distribution_snapshot(name: str) -> dict[str, object]:
        distribution = metadata.distribution(name)
        entries = distribution.files
        if entries is None:
            raise CodexAuthoringContractError(
                f"runtime distribution has no installed-file manifest: {name}"
            )
        files: list[dict[str, object]] = []
        excluded_external_entrypoints: list[str] = []
        for entry in sorted(entries, key=lambda item: item.as_posix()):
            if ".." in entry.parts:
                excluded_external_entrypoints.append(entry.as_posix())
                continue
            path = Path(distribution.locate_file(entry)).resolve(strict=True)
            if not path.is_file() or path.is_symlink():
                raise CodexAuthoringContractError(
                    f"runtime distribution file is unsafe or missing: {name}:{entry}"
                )
            content = read_stable_regular_file(
                path,
                label=f"runtime distribution {name}:{entry.as_posix()}",
            )
            files.append(
                {
                    "file": entry.as_posix(),
                    "bytes": len(content),
                    "file_sha256": sha256_bytes(content),
                }
            )
        payload: dict[str, object] = {
            "requested_name": name,
            "installed_name": distribution.metadata["Name"],
            "version": distribution.version,
            "files": files,
            "excluded_external_entrypoints": excluded_external_entrypoints,
        }
        return {
            **payload,
            "tree_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }

    def distribution_closure() -> tuple[str, ...]:
        pending = list(CODEX_RUNTIME_DISTRIBUTIONS)
        resolved: dict[str, str] = {}

        def normalized(name: str) -> str:
            return name.casefold().replace("_", "-").replace(".", "-")

        while pending:
            requested = pending.pop()
            key = normalized(requested)
            if key in resolved:
                continue
            distribution = metadata.distribution(requested)
            installed_name = distribution.metadata["Name"]
            if not isinstance(installed_name, str) or not installed_name:
                raise CodexAuthoringContractError(
                    f"runtime distribution has no installed name: {requested}"
                )
            resolved[key] = installed_name
            for raw_requirement in distribution.requires or ():
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement as error:
                    raise CodexAuthoringContractError(
                        "runtime distribution requirement is invalid: "
                        f"{installed_name}:{raw_requirement}"
                    ) from error
                if requirement.marker is not None and not requirement.marker.evaluate(
                    {"extra": ""}
                ):
                    continue
                pending.append(requirement.name)
        return tuple(resolved[key] for key in sorted(resolved))

    distribution_names = distribution_closure()

    return {
        "schema_version": 2,
        "artifact_kind": "codex_author_runtime_dependency_snapshot",
        "trusted_sources": [
            binding(relative) for relative in CODEX_TRUSTED_SOURCE_FILES
        ],
        "repository_locks": [
            binding("pyproject.toml"),
            binding("uv.lock"),
        ],
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": sys.version,
            "executable": executable(sys.executable),
            "base_executable": executable(
                str(getattr(sys, "_base_executable", sys.executable))
            ),
        },
        "distribution_roots": list(CODEX_RUNTIME_DISTRIBUTIONS),
        "distributions": [
            distribution_snapshot(name) for name in distribution_names
        ],
    }


def build_codex_authoring_request(
    *,
    authoring_input: CodexAuthoringInput,
    output_contract: CodexAuthoringOutputContract,
) -> CanonicalCodexAuthoringRequest:
    payload = {
        "schema_version": 1,
        "authoring_input": authoring_input.model_dump(mode="json"),
        "output_contract": output_contract.model_dump(mode="json"),
    }
    return CanonicalCodexAuthoringRequest.model_validate(
        {**payload, "request_sha256": sha256_bytes(canonical_json_bytes(payload))},
        strict=True,
    )


def render_codex_authoring_stdin(request: CanonicalCodexAuthoringRequest) -> bytes:
    request = CanonicalCodexAuthoringRequest.model_validate(
        request.model_dump(mode="python"), strict=True
    )
    return _STDIN_PREFIX + request.canonical_bytes() + _STDIN_SUFFIX


def load_codex_authoring_input(
    path: str | Path, *, expected_file_sha256: str
) -> CodexAuthoringInput:
    try:
        content = read_stable_regular_file(path, label="Codex authoring input")
        if sha256_bytes(content) != expected_file_sha256:
            raise CodexAuthoringContractError("Codex authoring input digest mismatch")
        raw = parse_canonical_json(content, label="Codex authoring input")
        if not isinstance(raw, dict):
            raise CodexAuthoringContractError(
                "Codex authoring input must contain an object"
            )
        return CodexAuthoringInput.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise CodexAuthoringContractError("Codex authoring input is invalid") from error


def parse_codex_authoring_request(
    content: bytes,
) -> CanonicalCodexAuthoringRequest:
    try:
        raw = parse_canonical_json(content, label="Codex canonical authoring request")
        if not isinstance(raw, dict):
            raise CodexAuthoringContractError(
                "Codex canonical authoring request must contain an object"
            )
        return CanonicalCodexAuthoringRequest.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise CodexAuthoringContractError(
            "Codex canonical authoring request is invalid"
        ) from error


def normalize_codex_authoring_output(
    raw_final: bytes,
    *,
    authoring_input: CodexAuthoringInput,
    semantic_source: AuthoringInput,
) -> AuthoringDraftBundle:
    """Strictly validate semantic content, then bind it to the Codex input."""

    try:
        raw = parse_strict_json(raw_final, label="Codex author final output")
        if not isinstance(raw, dict):
            raise CodexAuthoringContractError(
                "Codex author final output must contain an object"
            )
        content = AuthoringContentPayload.model_validate(raw, strict=True)
        semantic_bundle = normalize_authoring_content_payload(
            content,
            authoring_input=semantic_source,
        )
    except (ArtifactFormatError, ValidationError) as error:
        raise CodexAuthoringContractError(
            "Codex author final output is not valid author content"
        ) from error
    return build_authoring_draft_bundle(
        authoring_input_sha256=authoring_input.input_sha256,
        drafts=semantic_bundle.drafts,
    )


__all__ = [
    "CODEX_AUTHOR_RUN_ID",
    "CODEX_COMMAND_SHAPE",
    "CODEX_ENV_ALLOWLIST",
    "CODEX_EVIDENCE_TIER",
    "CODEX_EXECUTION_TYPE",
    "CODEX_RUNTIME_DISTRIBUTIONS",
    "CODEX_TRUSTED_SOURCE_FILES",
    "CanonicalCodexAuthoringRequest",
    "CodexAuthoringContractError",
    "CodexAuthoringInput",
    "CodexAuthoringOutputContract",
    "CodexSessionBudget",
    "build_codex_authoring_input",
    "build_codex_authoring_request",
    "build_codex_output_contract",
    "build_codex_runtime_dependency_snapshot",
    "load_codex_authoring_input",
    "load_codex_model_access_evidence",
    "normalize_codex_authoring_output",
    "parse_codex_authoring_request",
    "project_codex_cli_output_schema",
    "render_codex_authoring_stdin",
    "validate_codex_cli_output_schema",
]
