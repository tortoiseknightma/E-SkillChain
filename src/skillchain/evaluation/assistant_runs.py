"""Frozen five-configuration Assistant runs with create-only evidence bundles.

This module is intentionally an integrity boundary, not a claim that a real
model/evaluator has been calibrated.  A matrix plan can only be used after its
canonical bytes have been checked against an externally supplied digest and a
live, explicitly generation-bound seven- or eight-tool registry.  Every run
bundle retains one row per planned query, including runtime failures, and can
only acquire a verified handle after deep revalidation against that plan and
registry.

``SpecBaseline`` is represented in the plan as a diagnostic treatment but is
structurally excluded from the five formal configurations accepted by the run
builder.
"""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from skillchain.evaluation.packets import (
    AssistantResult,
    AssistantRunConfig,
    AssistantToolTrace,
    RubricSnapshot,
    VisibleCard,
    VisibleToolEvidence,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    PublicScorerCallEvidenceV2,
)
from skillchain.llm import LLMUsage
from skillchain.schemas import Query
from skillchain.synthesis.splitting import (
    FrozenSplitManifest,
    assert_no_group_leakage,
    build_atomic_groups,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.contracts import JSONValue, validate_json_value
from skillchain.tools.registry import (
    DiagnosticToolRegistry,
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    ToolRegistry,
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

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AssetToken = Annotated[str, StringConstraints(pattern=r"^asset-token-[0-9a-f]{32}$")]

ASSISTANT_MATRIX_POLICY_VERSION = "assistant-matrix-v2"
ASSISTANT_RUN_POLICY_VERSION = "assistant-run-bundle-v1"
ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION = "runner-owned-assistant-v3"
ASSISTANT_ROUTE_CALL_EVIDENCE_POLICY_VERSION = "assistant-route-call-evidence-v2"
ASSISTANT_ROUTE_RESPONSE_PREFIX_MAX_BYTES = 8_192
PHASE4_SELECTION_POLICY_VERSION = "phase4-input-selection-v1"
FORMAL_INELIGIBLE_REASON = "pending-real-model-and-judge-calibration"

MAIN_CONFIG_ORDER: tuple[AssistantRunConfig, ...] = (
    "noskill",
    "llm_static",
    "s1",
    "s1s2",
    "full",
)

_PLAN_MARKER = object()
_RUN_MARKER = object()
_FIVE_RUN_MARKER = object()
_PHASE4_INPUTS_MARKER = object()

_SPEC_FILE = "run-spec.json"
_REQUESTS_FILE = "requests.jsonl"
_RESULTS_FILE = "results.jsonl"
_MANIFEST_FILE = "manifest.json"
_ARTIFACT_FILES = (_SPEC_FILE, _REQUESTS_FILE, _RESULTS_FILE)
_BUNDLE_FILES = (*_ARTIFACT_FILES, _MANIFEST_FILE)

_PUBLIC_INPUT_FORBIDDEN_KEYS = frozenset(
    {
        "acceptable_capabilities",
        "bank",
        "bank_sha256",
        "canonical_capability",
        "config",
        "ground_truth",
        "gt_capability",
        "judge_output",
        "judge_scores",
        "route_trace",
        "skill_description",
        "skill_body",
        "skill_slug",
        "source_dataset",
        "split",
        "split_name",
    }
)


class AssistantRunError(ValueError):
    """A plan, backend response, or persisted run violates the frozen contract."""


class AssistantBackendContractError(AssistantRunError):
    """A backend claimed execution metadata that differs from the run lock."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _hash_payload(payload: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(payload)))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _model_payload(model: BaseModel, *, exclude: set[str] | None = None) -> dict:
    return model.model_dump(mode="json", exclude=exclude or set())


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(_model_payload(model, exclude={field_name}))


def _assert_no_forbidden_keys(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            if normalized in _PUBLIC_INPUT_FORBIDDEN_KEYS:
                raise ValueError(f"{label} contains forbidden field: {key}")
            _assert_no_forbidden_keys(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item, label=label)


class BackboneLock(_StrictFrozenModel):
    provider: str
    model: str
    endpoint: str
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int | None = None
    system_prompt_sha256: Sha256
    identity_sha256: Sha256

    @field_validator("provider", "model", "endpoint")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("https://") or "@" in value.split("/", 3)[2]:
            raise ValueError("backbone endpoint must be credential-free HTTPS")
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.identity_sha256 != _self_hash(self, "identity_sha256"):
            raise ValueError("backbone identity self hash mismatch")
        return self


class InferenceBudget(_StrictFrozenModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_turns: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)
    budget_sha256: Sha256

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.budget_sha256 != _self_hash(self, "budget_sha256"):
            raise ValueError("inference budget self hash mismatch")
        return self


class ToolRuntimeBinding(_StrictFrozenModel):
    tool_name: str
    tool_spec_sha256: Sha256
    runtime_binding_sha256: Sha256

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if value not in MVP_TOOL_NAMES_V2:
            raise ValueError("unknown tool in registry runtime lock")
        return value


class RegistryRuntimeLock(_StrictFrozenModel):
    """Legacy seven-tool lock; its serialized bytes remain unchanged."""

    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    tools: tuple[ToolRuntimeBinding, ...]
    lock_sha256: Sha256

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        names = tuple(item.tool_name for item in self.tools)
        if names != tuple(sorted(MVP_TOOL_NAMES)):
            raise ValueError("registry lock must contain exactly seven sorted tools")
        if self.lock_sha256 != _self_hash(self, "lock_sha256"):
            raise ValueError("registry runtime lock self hash mismatch")
        return self


class RegistryRuntimeLockV2(_StrictFrozenModel):
    """Explicit eight-tool generation containing the composite ToolSpec."""

    schema_version: Literal[2] = 2
    policy_version: Literal["assistant-registry-runtime-v2"] = (
        "assistant-registry-runtime-v2"
    )
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    tools: tuple[ToolRuntimeBinding, ...]
    lock_sha256: Sha256

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        names = tuple(item.tool_name for item in self.tools)
        if names != tuple(sorted(MVP_TOOL_NAMES_V2)):
            raise ValueError("registry-v2 lock must contain exactly eight sorted tools")
        if self.lock_sha256 != _self_hash(self, "lock_sha256"):
            raise ValueError("registry-v2 runtime lock self hash mismatch")
        return self


AssistantRegistryRuntimeLock = RegistryRuntimeLock | RegistryRuntimeLockV2


def _opaque_asset_token(*, query_id: str, asset_id: str) -> str:
    """Return the model-visible handle for one authoritative query asset.

    The token intentionally commits to the query/asset pair without exposing the
    catalog asset identifier, its source prefix, or a local path.  It is stable
    when the same catalog asset is moved beneath a different asset-root.
    """

    digest = sha256_bytes(
        canonical_json_bytes({"query_id": query_id, "asset_id": asset_id})
    )
    return f"asset-token-{digest[:32]}"


class AssistantAssetBinding(_StrictFrozenModel):
    """Internal, hash-bound asset authority for a model-visible query token."""

    schema_version: Literal[1] = 1
    asset_id: str
    image_path: str
    leakage_group_id: str
    asset_token: AssetToken
    binding_sha256: Sha256

    @field_validator("asset_id", "image_path", "leakage_group_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.binding_sha256 != _self_hash(self, "binding_sha256"):
            raise ValueError("Assistant asset binding self hash mismatch")
        return self


class AssistantQueryInput(_StrictFrozenModel):
    query_id: str
    public_input_json: str
    public_input_sha256: Sha256
    # Absent only for immutable r1/legacy matrix artifacts.  New projections
    # always carry this internal binding while keeping it out of the model
    # message itself.
    asset_binding: AssistantAssetBinding | None = None
    query_sha256: Sha256

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler):
        """Do not alter legacy canonical bytes by serialising a new null field."""

        payload = handler(self)
        if self.asset_binding is None:
            payload.pop("asset_binding", None)
        return payload

    @model_validator(mode="after")
    def validate_public_input(self) -> Self:
        content = self.public_input_json.encode("utf-8")
        try:
            parsed = parse_canonical_json(content, label=f"query {self.query_id}")
        except ArtifactFormatError as error:
            raise ValueError("public_input_json must be canonical JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("public Assistant input must be a JSON object")
        _assert_no_forbidden_keys(parsed, label=f"query {self.query_id}")
        if self.asset_binding is not None:
            if set(parsed) != {"asset_id", "text", "turns"}:
                raise ValueError(
                    "bound Assistant public input must contain only asset_id, text, turns"
                )
            token = parsed.get("asset_id")
            if token != self.asset_binding.asset_token:
                raise ValueError(
                    "public Assistant asset token differs from hidden binding"
                )
            expected_token = _opaque_asset_token(
                query_id=self.query_id, asset_id=self.asset_binding.asset_id
            )
            if self.asset_binding.asset_token != expected_token:
                raise ValueError("hidden Assistant binding has an invalid asset token")
        if self.public_input_sha256 != sha256_bytes(content):
            raise ValueError("public Assistant input hash mismatch")
        # Older r1 records did not carry an internal binding.  Their original
        # self-hash must remain valid so they remain readable byte-for-byte.
        expected_query_sha256 = (
            _self_hash(self, "query_sha256")
            if self.asset_binding is not None
            else _hash_payload(
                {
                    "query_id": self.query_id,
                    "public_input_json": self.public_input_json,
                    "public_input_sha256": self.public_input_sha256,
                }
            )
        )
        if self.query_sha256 != expected_query_sha256:
            raise ValueError("Assistant query self hash mismatch")
        return self


class JudgeAuditSelectionEntry(_StrictFrozenModel):
    query_id: str
    config: AssistantRunConfig
    evaluation_id: Sha256

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")


class Phase4InputSelectionManifest(_StrictFrozenModel):
    """Externally locked coordination record for one Phase 4 matrix.

    This artifact binds the selected query order, frozen split, final rubric,
    and judge-audit sample before an :class:`AssistantMatrixPlan` can be built.
    It remains diagnostic-only until a later protocol supplies real-model and
    evaluator calibration evidence.
    """

    schema_version: Literal[1] = 1
    kind: Literal["phase4-input-selection"] = "phase4-input-selection"
    policy_version: Literal["phase4-input-selection-v1"] = (
        PHASE4_SELECTION_POLICY_VERSION
    )
    matrix_run_id: str
    query_artifact_sha256: Sha256
    split_manifest_sha256: Sha256
    final_rubric_file_sha256: Sha256
    final_rubric_content_sha256: Sha256
    query_ids: tuple[str, ...]
    judge_audit_entries: tuple[JudgeAuditSelectionEntry, ...]
    judge_audit_evaluation_ids_sha256: Sha256
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    selection_sha256: Sha256

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "matrix_run_id")

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.query_ids:
            raise ValueError("Phase 4 selection must contain at least one query")
        if any(not item or item != item.strip() for item in self.query_ids):
            raise ValueError(
                "Phase 4 selection query IDs must be non-blank and trimmed"
            )
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("Phase 4 selection query IDs must be unique")
        if not self.judge_audit_entries:
            raise ValueError("judge audit selection must contain at least one entry")
        evaluation_ids = tuple(item.evaluation_id for item in self.judge_audit_entries)
        if evaluation_ids != tuple(sorted(evaluation_ids)) or len(
            evaluation_ids
        ) != len(set(evaluation_ids)):
            raise ValueError(
                "judge audit selection entries must have unique sorted evaluation IDs"
            )
        instances = tuple(
            (item.query_id, item.config) for item in self.judge_audit_entries
        )
        if len(instances) != len(set(instances)):
            raise ValueError("judge audit selection instances must be unique")
        if any(
            item.query_id not in self.query_ids for item in self.judge_audit_entries
        ):
            raise ValueError("judge audit selection references an unselected query")
        if self.judge_audit_evaluation_ids_sha256 != _hash_payload(
            list(evaluation_ids)
        ):
            raise ValueError("judge audit selection evaluation-ID hash mismatch")
        if self.selection_sha256 != _self_hash(self, "selection_sha256"):
            raise ValueError("Phase 4 input selection self hash mismatch")
        return self


class MatrixTreatment(_StrictFrozenModel):
    config: AssistantRunConfig
    bank_sha256: Sha256 | None
    router_stage: Literal["disabled", "llm_static", "s1", "s2"]
    body_stage: Literal["disabled", "llm_static", "s1", "s3"]

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        expected = {
            "noskill": (None, "disabled", "disabled"),
            "llm_static": ("required", "llm_static", "llm_static"),
            "s1": ("required", "s1", "s1"),
            "s1s2": ("required", "s2", "s1"),
            "full": ("required", "s2", "s3"),
        }[self.config]
        expected_bank, expected_router, expected_body = expected
        if (self.bank_sha256 is None) != (expected_bank is None):
            raise ValueError("treatment Bank presence does not match configuration")
        if (self.router_stage, self.body_stage) != (
            expected_router,
            expected_body,
        ):
            raise ValueError("treatment stages do not match configuration")
        return self


class DiagnosticSpecBaseline(_StrictFrozenModel):
    name: Literal["spec_baseline"] = "spec_baseline"
    bank_sha256: Sha256
    purpose: Literal["diagnostic-only"] = "diagnostic-only"
    formal_eligible: Literal[False] = False


class AssistantMatrixPlan(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["assistant-matrix-plan"] = "assistant-matrix-plan"
    policy_version: Literal["assistant-matrix-v2"] = ASSISTANT_MATRIX_POLICY_VERSION
    matrix_run_id: str
    query_artifact_sha256: Sha256
    split_manifest_sha256: Sha256
    backbone: BackboneLock
    budget: InferenceBudget
    registry: AssistantRegistryRuntimeLock
    queries: tuple[AssistantQueryInput, ...]
    query_order_sha256: Sha256
    query_set_sha256: Sha256
    judge_audit_selection_manifest_sha256: Sha256
    judge_audit_evaluation_ids: tuple[Sha256, ...]
    judge_audit_evaluation_ids_sha256: Sha256
    final_rubric_sha256: Sha256
    treatments: tuple[MatrixTreatment, ...]
    diagnostics: tuple[DiagnosticSpecBaseline]
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    plan_sha256: Sha256

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "matrix_run_id")

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if not self.queries:
            raise ValueError("Assistant matrix requires at least one query")
        query_ids = tuple(item.query_id for item in self.queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Assistant matrix query IDs must be unique")
        expected_order_hash = _hash_payload(list(query_ids))
        if self.query_order_sha256 != expected_order_hash:
            raise ValueError("Assistant matrix query-order hash mismatch")
        expected_set_hash = _hash_payload(
            [item.model_dump(mode="json") for item in self.queries]
        )
        if self.query_set_sha256 != expected_set_hash:
            raise ValueError("Assistant matrix query-set hash mismatch")
        if (
            not self.judge_audit_evaluation_ids
            or self.judge_audit_evaluation_ids
            != tuple(sorted(self.judge_audit_evaluation_ids))
            or len(self.judge_audit_evaluation_ids)
            != len(set(self.judge_audit_evaluation_ids))
        ):
            raise ValueError(
                "judge audit evaluation IDs must be non-empty, unique, sorted"
            )
        if self.judge_audit_evaluation_ids_sha256 != _hash_payload(
            list(self.judge_audit_evaluation_ids)
        ):
            raise ValueError("judge audit evaluation-ID commitment mismatch")
        if tuple(item.config for item in self.treatments) != MAIN_CONFIG_ORDER:
            raise ValueError(
                "Assistant matrix must contain exactly the five main configurations "
                "in canonical order"
            )
        if len(self.diagnostics) != 1 or self.diagnostics[0].name != "spec_baseline":
            raise ValueError("SpecBaseline must be present exactly once as diagnostic")
        if self.plan_sha256 != _self_hash(self, "plan_sha256"):
            raise ValueError("Assistant matrix plan self hash mismatch")
        return self


@dataclass(frozen=True)
class AssistantQueryDefinition:
    """Convenience input converted to immutable canonical JSON in the plan."""

    query_id: str
    public_input: Mapping[str, JSONValue]


@dataclass(frozen=True)
class CreatedAssistantMatrixPlan:
    path: Path
    plan: AssistantMatrixPlan
    file_sha256: str


@dataclass(frozen=True)
class VerifiedAssistantMatrixPlan:
    """Integrity-verified plan; this does not grant formal-result eligibility."""

    path: Path
    plan: AssistantMatrixPlan
    expected_file_sha256: str
    _content: bytes = field(repr=False)
    _marker: object = field(repr=False)


@dataclass(frozen=True)
class VerifiedPhase4Inputs:
    """Four externally locked inputs, cross-checked and held as typed values."""

    query_path: Path
    split_manifest_path: Path
    rubric_path: Path
    selection_path: Path
    expected_query_sha256: str
    expected_split_manifest_sha256: str
    expected_rubric_file_sha256: str
    expected_selection_file_sha256: str
    queries: tuple[Query, ...]
    selected_queries: tuple[Query, ...]
    assistant_queries: tuple[AssistantQueryInput, ...]
    split_manifest: FrozenSplitManifest
    rubric: RubricSnapshot
    selection: Phase4InputSelectionManifest
    _snapshots: tuple[tuple[Path, bytes], ...] = field(repr=False)
    _marker: object = field(repr=False)


def make_backbone_lock(
    *,
    provider: str,
    model: str,
    endpoint: str,
    temperature: float,
    top_p: float,
    seed: int | None,
    system_prompt_sha256: str,
) -> BackboneLock:
    payload = {
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "system_prompt_sha256": system_prompt_sha256,
    }
    return BackboneLock.model_validate(
        {**payload, "identity_sha256": _hash_payload(payload)}, strict=True
    )


def make_inference_budget(
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    max_tool_calls: int,
    max_turns: int,
    timeout_ms: int,
) -> InferenceBudget:
    payload = {
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "max_tool_calls": max_tool_calls,
        "max_turns": max_turns,
        "timeout_ms": timeout_ms,
    }
    return InferenceBudget.model_validate(
        {**payload, "budget_sha256": _hash_payload(payload)}, strict=True
    )


def make_phase4_input_selection_manifest(
    *,
    matrix_run_id: str,
    query_artifact_sha256: str,
    split_manifest_sha256: str,
    final_rubric_file_sha256: str,
    final_rubric_content_sha256: str,
    query_ids: Sequence[str],
    judge_audit_entries: Sequence[JudgeAuditSelectionEntry],
) -> Phase4InputSelectionManifest:
    """Build the typed coordination artifact that must be locked externally."""

    entries = tuple(judge_audit_entries)
    if any(type(item) is not JudgeAuditSelectionEntry for item in entries):
        raise TypeError(
            "judge_audit_entries must contain JudgeAuditSelectionEntry values"
        )
    evaluation_ids = [item.evaluation_id for item in entries]
    unsigned = {
        "schema_version": 1,
        "kind": "phase4-input-selection",
        "policy_version": PHASE4_SELECTION_POLICY_VERSION,
        "matrix_run_id": matrix_run_id,
        "query_artifact_sha256": query_artifact_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "final_rubric_file_sha256": final_rubric_file_sha256,
        "final_rubric_content_sha256": final_rubric_content_sha256,
        "query_ids": tuple(query_ids),
        "judge_audit_entries": entries,
        "judge_audit_evaluation_ids_sha256": _hash_payload(evaluation_ids),
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    return Phase4InputSelectionManifest.model_validate(
        {**unsigned, "selection_sha256": _hash_payload(unsigned)}, strict=True
    )


def load_verified_phase4_inputs(
    *,
    query_path: str | Path,
    expected_query_sha256: str,
    split_manifest_path: str | Path,
    expected_split_manifest_sha256: str,
    rubric_path: str | Path,
    expected_rubric_file_sha256: str,
    selection_path: str | Path,
    expected_selection_file_sha256: str,
) -> VerifiedPhase4Inputs:
    """Load, cross-bind, and externally authenticate all matrix inputs.

    The query artifact is the canonical full split assignment.  The selection
    manifest chooses the exact ordered subset exposed to the Assistant and
    cross-references the other three externally locked files.
    """

    expected = (
        _require_sha256(expected_query_sha256, "expected query artifact sha256"),
        _require_sha256(
            expected_split_manifest_sha256,
            "expected split manifest sha256",
        ),
        _require_sha256(
            expected_rubric_file_sha256,
            "expected final rubric file sha256",
        ),
        _require_sha256(
            expected_selection_file_sha256,
            "expected Phase 4 selection file sha256",
        ),
    )
    paths = tuple(
        Path(value).absolute()
        for value in (
            query_path,
            split_manifest_path,
            rubric_path,
            selection_path,
        )
    )
    if len(set(paths)) != len(paths):
        raise AssistantRunError("Phase 4 input roles must use four distinct files")
    contents = _read_phase4_input_files(paths)
    for label, content, digest in zip(
        ("query artifact", "split manifest", "final rubric", "input selection"),
        contents,
        expected,
        strict=True,
    ):
        if sha256_bytes(content) != digest:
            raise AssistantRunError(f"Phase 4 {label} external digest mismatch")
    parsed = _parse_and_validate_phase4_inputs(contents, expected)
    if _read_phase4_input_files(paths) != contents:
        raise AssistantRunError("Phase 4 inputs changed during verification")
    return VerifiedPhase4Inputs(
        query_path=paths[0],
        split_manifest_path=paths[1],
        rubric_path=paths[2],
        selection_path=paths[3],
        expected_query_sha256=expected[0],
        expected_split_manifest_sha256=expected[1],
        expected_rubric_file_sha256=expected[2],
        expected_selection_file_sha256=expected[3],
        queries=parsed[0],
        selected_queries=parsed[1],
        assistant_queries=parsed[2],
        split_manifest=parsed[3],
        rubric=parsed[4],
        selection=parsed[5],
        _snapshots=tuple(zip(paths, contents, strict=True)),
        _marker=_PHASE4_INPUTS_MARKER,
    )


def require_verified_phase4_inputs(value: object) -> VerifiedPhase4Inputs:
    """Deeply revalidate a typed handle and all four on-disk snapshots."""

    if (
        not isinstance(value, VerifiedPhase4Inputs)
        or value._marker is not _PHASE4_INPUTS_MARKER
    ):
        raise TypeError(
            "Phase 4 matrix inputs require the external-digest typed loader"
        )
    paths = (
        value.query_path,
        value.split_manifest_path,
        value.rubric_path,
        value.selection_path,
    )
    contents = _read_phase4_input_files(paths)
    if tuple(zip(paths, contents, strict=True)) != value._snapshots:
        raise AssistantRunError("verified Phase 4 inputs changed on disk")
    expected = (
        value.expected_query_sha256,
        value.expected_split_manifest_sha256,
        value.expected_rubric_file_sha256,
        value.expected_selection_file_sha256,
    )
    if any(
        sha256_bytes(content) != digest
        for content, digest in zip(contents, expected, strict=True)
    ):
        raise AssistantRunError("verified Phase 4 input digest changed")
    parsed = _parse_and_validate_phase4_inputs(contents, expected)
    held = (
        value.queries,
        value.selected_queries,
        value.assistant_queries,
        value.split_manifest,
        value.rubric,
        value.selection,
    )
    if parsed != held:
        raise AssistantRunError("verified Phase 4 typed inputs were mutated")
    return value


def _read_phase4_input_files(paths: Sequence[Path]) -> tuple[bytes, ...]:
    labels = ("query artifact", "split manifest", "final rubric", "input selection")
    content: list[bytes] = []
    for path, label in zip(paths, labels, strict=True):
        try:
            content.append(read_stable_regular_file(path, label=f"Phase 4 {label}"))
        except ArtifactFormatError as error:
            raise AssistantRunError(str(error)) from error
    return tuple(content)


def _parse_and_validate_phase4_inputs(
    contents: tuple[bytes, ...],
    expected: tuple[str, ...],
) -> tuple[
    tuple[Query, ...],
    tuple[Query, ...],
    tuple[AssistantQueryInput, ...],
    FrozenSplitManifest,
    RubricSnapshot,
    Phase4InputSelectionManifest,
]:
    query_bytes, split_bytes, rubric_bytes, selection_bytes = contents
    queries = _load_jsonl_models(query_bytes, Query, "Phase 4 query artifact")
    query_ids = tuple(item.query_id for item in queries)
    if query_ids != tuple(sorted(query_ids)) or len(query_ids) != len(set(query_ids)):
        raise AssistantRunError(
            "Phase 4 query artifact must have unique rows sorted by query_id"
        )
    if query_bytes != canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in queries)
    ):
        raise AssistantRunError("Phase 4 query artifact must be canonical typed JSONL")

    split_manifest = _load_model(
        split_bytes,
        FrozenSplitManifest,
        "Phase 4 split manifest",
    )
    if split_manifest.assignment_sha256 != expected[0]:
        raise AssistantRunError(
            "Phase 4 split manifest does not bind the query artifact"
        )
    split_counts = Counter(item.split for item in queries)
    if split_counts != Counter(split_manifest.split_sizes):
        raise AssistantRunError("Phase 4 split manifest counts do not match queries")
    if {item.taxonomy_version for item in queries} != {split_manifest.taxonomy_version}:
        raise AssistantRunError("Phase 4 split taxonomy does not match queries")
    try:
        assert_no_group_leakage(list(queries))
    except ValueError as error:
        raise AssistantRunError(
            "Phase 4 query artifact violates group split"
        ) from error
    if len(build_atomic_groups(list(queries))) != split_manifest.component_count:
        raise AssistantRunError("Phase 4 split component count does not match queries")
    frozen = tuple(item for item in queries if item.split == "test_frozen")
    frozen_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in frozen)
    )
    if (
        len(frozen) != split_manifest.count
        or sha256_bytes(frozen_bytes) != split_manifest.sha256
    ):
        raise AssistantRunError("Phase 4 frozen-test commitment does not match queries")
    if Counter(item.canonical_intent for item in frozen) != Counter(
        split_manifest.intent_counts
    ):
        raise AssistantRunError("Phase 4 frozen-test intent counts do not match")
    if Counter(
        item.canonical_capability
        for item in frozen
        if item.canonical_capability is not None
    ) != Counter(split_manifest.capability_counts):
        raise AssistantRunError("Phase 4 frozen-test capability counts do not match")
    if sum(item.is_boundary for item in frozen) != split_manifest.boundary_count:
        raise AssistantRunError("Phase 4 frozen-test boundary count does not match")

    rubric = _load_model(rubric_bytes, RubricSnapshot, "Phase 4 final rubric")
    selection = _load_model(
        selection_bytes,
        Phase4InputSelectionManifest,
        "Phase 4 input selection",
    )
    if (
        selection.query_artifact_sha256,
        selection.split_manifest_sha256,
        selection.final_rubric_file_sha256,
        selection.final_rubric_content_sha256,
    ) != (expected[0], expected[1], expected[2], rubric.content_sha256):
        raise AssistantRunError(
            "Phase 4 selection does not bind the query, split, and rubric files"
        )
    by_id = {item.query_id: item for item in queries}
    try:
        selected = tuple(by_id[query_id] for query_id in selection.query_ids)
    except KeyError as error:
        raise AssistantRunError(
            "Phase 4 selection references a query absent from the locked artifact"
        ) from error
    assistant_queries = tuple(build_assistant_query_input(item) for item in selected)
    return queries, selected, assistant_queries, split_manifest, rubric, selection


def build_assistant_query_input(query: Query) -> AssistantQueryInput:
    """Project one verified query into the only Assistant-visible input shape."""

    if type(query) is not Query:
        raise TypeError("Assistant public input requires exactly a schema-v2 Query")
    asset_token = _opaque_asset_token(query_id=query.query_id, asset_id=query.asset_id)
    binding_unsigned = {
        "schema_version": 1,
        "asset_id": query.asset_id,
        "image_path": query.image_path,
        "leakage_group_id": query.leakage_group_id,
        "asset_token": asset_token,
    }
    asset_binding = AssistantAssetBinding.model_validate(
        {
            **binding_unsigned,
            "binding_sha256": _hash_payload(binding_unsigned),
        },
        strict=True,
    )
    public_value = validate_json_value(
        {
            "asset_id": asset_token,
            "text": query.text,
            "turns": [item.model_dump(mode="json") for item in query.turns],
        }
    )
    if not isinstance(public_value, dict):  # pragma: no cover - fixed projection
        raise AssertionError("Assistant public projection must be an object")
    _assert_no_forbidden_keys(public_value, label=f"query {query.query_id}")
    public_bytes = canonical_json_bytes(public_value)
    unsigned = {
        "query_id": query.query_id,
        "public_input_json": public_bytes.decode("utf-8"),
        "public_input_sha256": sha256_bytes(public_bytes),
        "asset_binding": asset_binding,
    }
    return AssistantQueryInput.model_validate(
        {**unsigned, "query_sha256": _hash_payload(unsigned)}, strict=True
    )


def build_legacy_assistant_query_input(query: Query) -> AssistantQueryInput:
    """Rebuild the r1 public projection for an already-frozen launch.

    The v33 launch instances were frozen before opaque asset bindings were
    introduced.  This compatibility path is intentionally explicit and is
    only used by a recovery execution whose launch plan carries those legacy
    public-input hashes; new launches must use ``build_assistant_query_input``.
    """

    if type(query) is not Query:
        raise TypeError("Assistant public input requires exactly a schema-v2 Query")
    public_value = validate_json_value(
        {
            "asset_id": query.asset_id,
            "image_path": query.image_path,
            "text": query.text,
            "turns": [item.model_dump(mode="json") for item in query.turns],
        }
    )
    if not isinstance(public_value, dict):  # pragma: no cover - fixed projection
        raise AssertionError("Assistant public projection must be an object")
    _assert_no_forbidden_keys(public_value, label=f"query {query.query_id}")
    public_bytes = canonical_json_bytes(public_value)
    unsigned = {
        "query_id": query.query_id,
        "public_input_json": public_bytes.decode("utf-8"),
        "public_input_sha256": sha256_bytes(public_bytes),
    }
    return AssistantQueryInput.model_validate(
        {**unsigned, "query_sha256": _hash_payload(unsigned)}, strict=True
    )


def _registry_lock(registry: ToolRegistry) -> AssistantRegistryRuntimeLock:
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    if type(registry) is not DiagnosticToolRegistry:
        registry.require_formal_runtime()
    runtime_before = registry.registry_runtime_sha256
    specs = {item.name: item for item in registry.specs()}
    generation = registry.manifest.schema_version
    expected_names = MVP_TOOL_NAMES if generation == 1 else MVP_TOOL_NAMES_V2
    if set(specs) != expected_names:
        raise AssistantRunError(
            "tool registry manifest does not match its declared generation"
        )
    tools = tuple(
        ToolRuntimeBinding(
            tool_name=name,
            tool_spec_sha256=specs[name].spec_sha256,
            runtime_binding_sha256=registry.runtime_binding_sha256(name),
        )
        for name in sorted(expected_names)
    )
    runtime_after = registry.registry_runtime_sha256
    if runtime_after != runtime_before:
        raise AssistantRunError(
            "tool registry runtime changed while the matrix lock was captured"
        )
    payload: dict[str, object] = {
        "registry_sha256": registry.registry_sha256,
        "registry_runtime_sha256": runtime_before,
        "tools": tuple(item.model_dump(mode="json") for item in tools),
    }
    lock_type: type[RegistryRuntimeLock] | type[RegistryRuntimeLockV2]
    if generation == 1:
        lock_type = RegistryRuntimeLock
    else:
        lock_type = RegistryRuntimeLockV2
        payload = {
            "schema_version": 2,
            "policy_version": "assistant-registry-runtime-v2",
            **payload,
        }
    return lock_type.model_validate(
        {**payload, "lock_sha256": _hash_payload(payload)}, strict=True
    )


def build_assistant_matrix_plan(
    *,
    inputs: VerifiedPhase4Inputs,
    backbone: BackboneLock,
    budget: InferenceBudget,
    registry: ToolRegistry,
    bank_sha256_by_config: Mapping[str, str],
    spec_baseline_bank_sha256: str,
) -> AssistantMatrixPlan:
    """Build the single lock shared by all five treatments.

    Config-specific records contain only the intended treatment identity.  A
    caller therefore has no field through which it can vary backbone, registry,
    query order, or inference budget by configuration.
    """

    inputs = require_verified_phase4_inputs(inputs)
    if type(backbone) is not BackboneLock:
        raise TypeError("backbone must be a BackboneLock")
    if type(budget) is not InferenceBudget:
        raise TypeError("budget must be an InferenceBudget")
    required_banks = {"llm_static", "s1", "s1s2", "full"}
    if set(bank_sha256_by_config) != required_banks:
        raise ValueError(
            "bank_sha256_by_config must contain exactly llm_static, s1, s1s2, full"
        )
    query_models = inputs.assistant_queries
    query_ids = [item.query_id for item in query_models]
    audit_evaluation_ids = tuple(
        item.evaluation_id for item in inputs.selection.judge_audit_entries
    )
    treatments = (
        MatrixTreatment(
            config="noskill",
            bank_sha256=None,
            router_stage="disabled",
            body_stage="disabled",
        ),
        MatrixTreatment(
            config="llm_static",
            bank_sha256=bank_sha256_by_config["llm_static"],
            router_stage="llm_static",
            body_stage="llm_static",
        ),
        MatrixTreatment(
            config="s1",
            bank_sha256=bank_sha256_by_config["s1"],
            router_stage="s1",
            body_stage="s1",
        ),
        MatrixTreatment(
            config="s1s2",
            bank_sha256=bank_sha256_by_config["s1s2"],
            router_stage="s2",
            body_stage="s1",
        ),
        MatrixTreatment(
            config="full",
            bank_sha256=bank_sha256_by_config["full"],
            router_stage="s2",
            body_stage="s3",
        ),
    )
    registry_lock = _registry_lock(registry)
    diagnostic = DiagnosticSpecBaseline(bank_sha256=spec_baseline_bank_sha256)
    unsigned = {
        "schema_version": 1,
        "kind": "assistant-matrix-plan",
        "policy_version": ASSISTANT_MATRIX_POLICY_VERSION,
        "matrix_run_id": inputs.selection.matrix_run_id,
        "query_artifact_sha256": inputs.expected_query_sha256,
        "split_manifest_sha256": inputs.expected_split_manifest_sha256,
        "backbone": backbone,
        "budget": budget,
        "registry": registry_lock,
        "queries": tuple(query_models),
        "query_order_sha256": _hash_payload(query_ids),
        "query_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in query_models]
        ),
        "judge_audit_selection_manifest_sha256": (
            inputs.expected_selection_file_sha256
        ),
        "judge_audit_evaluation_ids": audit_evaluation_ids,
        "judge_audit_evaluation_ids_sha256": _hash_payload(list(audit_evaluation_ids)),
        "final_rubric_sha256": inputs.rubric.content_sha256,
        "treatments": treatments,
        "diagnostics": (diagnostic,),
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    plan = AssistantMatrixPlan.model_validate(
        {**unsigned, "plan_sha256": _hash_payload(unsigned)}, strict=True
    )
    require_verified_phase4_inputs(inputs)
    return plan


def create_assistant_matrix_plan_file(
    path: str | Path, plan: AssistantMatrixPlan
) -> CreatedAssistantMatrixPlan:
    if type(plan) is not AssistantMatrixPlan:
        raise TypeError("plan must be an AssistantMatrixPlan")
    content = canonical_json_bytes(plan.model_dump(mode="json"))
    path = atomic_create_file(path, content)
    return CreatedAssistantMatrixPlan(
        path=path,
        plan=plan,
        file_sha256=sha256_bytes(content),
    )


def load_verified_assistant_matrix_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
    registry: ToolRegistry,
) -> VerifiedAssistantMatrixPlan:
    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "expected plan file sha256"
    )
    path = Path(path)
    try:
        content = read_stable_regular_file(path, label="Assistant matrix plan")
    except ArtifactFormatError as error:
        raise AssistantRunError(str(error)) from error
    if sha256_bytes(content) != expected_file_sha256:
        raise AssistantRunError("Assistant matrix plan external digest mismatch")
    plan = _load_model(content, AssistantMatrixPlan, "Assistant matrix plan")
    _verify_registry_lock(plan.registry, registry)
    try:
        after = read_stable_regular_file(path, label="Assistant matrix plan")
    except ArtifactFormatError as error:
        raise AssistantRunError(str(error)) from error
    if after != content:
        raise AssistantRunError("Assistant matrix plan changed during verification")
    return VerifiedAssistantMatrixPlan(
        path=path,
        plan=plan,
        expected_file_sha256=expected_file_sha256,
        _content=content,
        _marker=_PLAN_MARKER,
    )


def require_verified_assistant_matrix_plan(
    value: object,
    *,
    registry: ToolRegistry,
) -> VerifiedAssistantMatrixPlan:
    if (
        not isinstance(value, VerifiedAssistantMatrixPlan)
        or value._marker is not _PLAN_MARKER
    ):
        raise TypeError(
            "Assistant runs require a plan returned by the external-digest verifier"
        )
    try:
        current = read_stable_regular_file(value.path, label="Assistant matrix plan")
    except ArtifactFormatError as error:
        raise AssistantRunError(str(error)) from error
    if current != value._content or sha256_bytes(current) != value.expected_file_sha256:
        raise AssistantRunError("verified Assistant matrix plan changed on disk")
    reparsed = _load_model(current, AssistantMatrixPlan, "Assistant matrix plan")
    if reparsed != value.plan:
        raise AssistantRunError("verified Assistant matrix plan handle was mutated")
    _verify_registry_lock(reparsed.registry, registry)
    return value


class AssistantRequestSnapshot(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    matrix_run_id: str
    config: AssistantRunConfig
    query_ordinal: int = Field(ge=0)
    query: AssistantQueryInput
    treatment: MatrixTreatment
    backbone: BackboneLock
    budget: InferenceBudget
    registry: AssistantRegistryRuntimeLock
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.treatment.config != self.config:
            raise ValueError("Assistant request treatment/config mismatch")
        if self.request_sha256 != _self_hash(self, "request_sha256"):
            raise ValueError("Assistant request self hash mismatch")
        return self


AssistantRouteFailureSubtype = Literal[
    "invalid_route_json",
    "length",
    "out_of_enum",
    "response_empty_text",
    "response_finish_reason",
    "response_tool_calls",
    "response_contract",
    "route_budget",
]


class AssistantRouteFailureShape(_StrictFrozenModel):
    """Safe route diagnostics that never retain response text or arguments."""

    schema_version: Literal[1] = 1
    transport: Literal["json_object"] = "json_object"
    finish_reason: str
    response_text_bytes: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    payload_status: Literal[
        "not_examined",
        "empty",
        "invalid_json",
        "non_object",
        "unexpected_keys",
        "schema_invalid",
        "out_of_enum",
    ]
    normalized_response_sha256: Sha256

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str) -> str:
        return _nonblank(value, "finish_reason")


class AssistantBackendResponse(_StrictFrozenModel):
    """Backend result with explicit echoes of every locked runtime invariant."""

    schema_version: Literal[1, 2] = 1
    request_sha256: Sha256
    backbone_provider: str
    backbone_model: str
    backbone_endpoint: str
    backbone_identity_sha256: Sha256
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    budget_sha256: Sha256
    response_text: str
    visible_cards: tuple[VisibleCard, ...] = ()
    visible_tool_evidence: tuple[VisibleToolEvidence, ...] = ()
    tool_trace: tuple[AssistantToolTrace, ...] = ()
    selected_capability: str | None = None
    skill_slug: str | None = None
    route_trace_sha256: Sha256 | None = None
    backbone_request_id: str | None = None
    usage: LLMUsage
    turn_count: int = Field(ge=1)
    latency_ms: int = Field(ge=0)
    error_code: str | None = None
    provider_exception_type: (
        Literal[
            "APIConnectionError",
            "RateLimitError",
            "LLMTimeoutError",
            "APIStatusError408",
            "APIStatusError429",
            "APIStatusError5xx",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    forfeited_reservation_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    budget_forfeit_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    route_failure_subtype: AssistantRouteFailureSubtype | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    route_failure_shape: AssistantRouteFailureShape | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("backbone_provider", "backbone_model", "backbone_endpoint")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "selected_capability",
        "skill_slug",
        "backbone_request_id",
        "error_code",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_response_text(self) -> Self:
        if self.error_code is None and not self.response_text.strip():
            raise ValueError("successful backend response requires visible text")
        if self.provider_exception_type is not None and not (
            self.error_code is not None
            and self.error_code.startswith("provider_pre_response_")
        ):
            raise ValueError(
                "provider exception diagnostics require a pre-response error"
            )
        forfeit_binding = (
            self.forfeited_reservation_sha256,
            self.budget_forfeit_sha256,
        )
        if any(value is None for value in forfeit_binding) != all(
            value is None for value in forfeit_binding
        ):
            raise ValueError("Assistant budget-forfeit binding must be complete")
        provider_pre_response = self.error_code is not None and (
            self.error_code.startswith("provider_pre_response_")
        )
        if self.schema_version == 1:
            if any(value is not None for value in forfeit_binding):
                raise ValueError("legacy Assistant response has budget-forfeit fields")
        elif provider_pre_response != all(
            value is not None for value in forfeit_binding
        ):
            raise ValueError(
                "active Assistant pre-response failure must bind its budget forfeit"
            )
        routing = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if any(value is not None for value in routing) and not all(
            value is not None for value in routing
        ):
            raise ValueError("backend routing identity must be complete or absent")
        route_failure = (self.route_failure_subtype, self.route_failure_shape)
        if any(value is None for value in route_failure) != all(
            value is None for value in route_failure
        ):
            raise ValueError("route failure subtype and shape must be paired")
        if self.route_failure_subtype is not None:
            expected_error = (
                "route_length"
                if self.route_failure_subtype == "length"
                else "route_contract_error"
            )
            if self.error_code != expected_error or any(
                value is not None for value in routing
            ):
                raise ValueError(
                    "route failure diagnostics differ from the coarse error/routing"
                )
        return self


class AssistantModelCallReceipt(_StrictFrozenModel):
    call_index: int = Field(ge=1)
    provider: str
    endpoint: str
    requested_model: str
    response_model: str
    provider_request_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    finish_reason: str
    latency_ms: int = Field(ge=0)
    response_sha256: Sha256

    @field_validator(
        "provider",
        "endpoint",
        "requested_model",
        "response_model",
        "provider_request_id",
        "finish_reason",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)


class AssistantRouteCallEvidence(_StrictFrozenModel):
    """Bounded, self-authenticating evidence for one captured route response.

    The exact UTF-8 response prefix is base64 encoded so truncation cannot
    corrupt a multibyte character.  The full response is always bound by its
    byte count and SHA-256.  This runner-private record deliberately contains
    no configuration label, local path, source row, template identity, or test
    metadata and is never projected into an Assistant or Judge prompt.
    """

    schema_version: Literal[1] = 1
    policy_version: Literal[
        "assistant-route-call-evidence-v1",
        "assistant-route-call-evidence-v2",
    ] = ASSISTANT_ROUTE_CALL_EVIDENCE_POLICY_VERSION
    attempt_index: int = Field(ge=1, le=2)
    request_variant: Literal["initial", "fixed_repair"] | None = None
    repair_of_wire_request_sha256: Sha256 | None = None
    ignored_response_keys: tuple[
        Literal["asset_id", "description", "text", "turns"], ...
    ] = ()
    wire_request_sha256: Sha256
    route_schema_sha256: Sha256
    response_sha256: Sha256
    response_text_sha256: Sha256
    response_text_bytes: int = Field(ge=0)
    response_text_prefix_bytes: int = Field(
        ge=0, le=ASSISTANT_ROUTE_RESPONSE_PREFIX_MAX_BYTES
    )
    response_text_prefix_base64: str
    response_text_truncated: bool
    failure_reason: AssistantRouteFailureSubtype | None = None
    payload_status: (
        Literal[
            "not_examined",
            "empty",
            "invalid_json",
            "non_object",
            "unexpected_keys",
            "schema_invalid",
            "out_of_enum",
        ]
        | None
    ) = None
    call_receipt: AssistantModelCallReceipt
    evidence_sha256: Sha256

    @field_validator("ignored_response_keys", mode="before")
    @classmethod
    def coerce_ignored_response_keys(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler):
        """Preserve the exact bytes and self hashes of immutable v1 evidence."""

        payload = handler(self)
        if self.policy_version == "assistant-route-call-evidence-v1":
            payload.pop("request_variant", None)
            payload.pop("repair_of_wire_request_sha256", None)
            payload.pop("ignored_response_keys", None)
        return payload

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.policy_version == "assistant-route-call-evidence-v1":
            if (
                self.request_variant is not None
                or self.repair_of_wire_request_sha256 is not None
                or self.ignored_response_keys
            ):
                raise ValueError("legacy route evidence cannot claim v2 semantics")
        else:
            if self.request_variant is None:
                raise ValueError("active route evidence requires a request variant")
            if self.request_variant == "fixed_repair":
                if (
                    self.attempt_index != 2
                    or self.repair_of_wire_request_sha256 is None
                    or self.repair_of_wire_request_sha256 == self.wire_request_sha256
                ):
                    raise ValueError(
                        "fixed route repair must bind a different initial wire"
                    )
            elif self.repair_of_wire_request_sha256 is not None:
                raise ValueError("initial route evidence cannot bind a repair wire")
            if self.ignored_response_keys != tuple(
                sorted(set(self.ignored_response_keys))
            ):
                raise ValueError(
                    "ignored route response keys must be sorted and unique"
                )
        try:
            prefix = base64.b64decode(
                self.response_text_prefix_base64.encode("ascii"),
                validate=True,
            )
        except (UnicodeEncodeError, ValueError) as error:
            raise ValueError(
                "route response prefix must be canonical base64"
            ) from error
        if base64.b64encode(prefix).decode("ascii") != self.response_text_prefix_base64:
            raise ValueError("route response prefix base64 is not canonical")
        if len(prefix) != self.response_text_prefix_bytes:
            raise ValueError("route response prefix byte count mismatch")
        if self.response_text_truncated != (
            self.response_text_bytes > self.response_text_prefix_bytes
        ):
            raise ValueError("route response truncation flag differs from byte counts")
        if self.response_text_truncated:
            if (
                self.response_text_prefix_bytes
                != ASSISTANT_ROUTE_RESPONSE_PREFIX_MAX_BYTES
            ):
                raise ValueError(
                    "truncated route response must retain the full prefix cap"
                )
        elif (
            self.response_text_bytes != self.response_text_prefix_bytes
            or sha256_bytes(prefix) != self.response_text_sha256
        ):
            raise ValueError("complete route response text binding mismatch")
        if (self.failure_reason is None) != (self.payload_status is None):
            raise ValueError("route failure reason and payload status must be paired")
        if self.failure_reason is not None and self.ignored_response_keys:
            raise ValueError("failed route responses cannot claim ignored keys")
        if (
            self.response_sha256 != self.call_receipt.response_sha256
            or self.call_receipt.finish_reason
            != self.call_receipt.finish_reason.strip()
        ):
            raise ValueError("route response evidence differs from its call receipt")
        if self.evidence_sha256 != _self_hash(self, "evidence_sha256"):
            raise ValueError("route-call evidence self hash mismatch")
        return self


def make_assistant_route_call_evidence(
    *,
    attempt_index: int,
    request_variant: Literal["initial", "fixed_repair"] = "initial",
    repair_of_wire_request_sha256: str | None = None,
    ignored_response_keys: tuple[
        Literal["asset_id", "description", "text", "turns"], ...
    ] = (),
    wire_request_sha256: str,
    route_schema_sha256: str,
    response_text: str,
    call_receipt: AssistantModelCallReceipt,
    failure_reason: AssistantRouteFailureSubtype | None,
    payload_status: str | None,
) -> AssistantRouteCallEvidence:
    """Create bounded route response evidence without retaining local identity."""

    response_bytes = response_text.encode("utf-8")
    prefix = response_bytes[:ASSISTANT_ROUTE_RESPONSE_PREFIX_MAX_BYTES]
    unsigned = {
        "schema_version": 1,
        "policy_version": ASSISTANT_ROUTE_CALL_EVIDENCE_POLICY_VERSION,
        "attempt_index": attempt_index,
        "request_variant": request_variant,
        "repair_of_wire_request_sha256": repair_of_wire_request_sha256,
        "ignored_response_keys": tuple(sorted(set(ignored_response_keys))),
        "wire_request_sha256": wire_request_sha256,
        "route_schema_sha256": route_schema_sha256,
        "response_sha256": call_receipt.response_sha256,
        "response_text_sha256": sha256_bytes(response_bytes),
        "response_text_bytes": len(response_bytes),
        "response_text_prefix_bytes": len(prefix),
        "response_text_prefix_base64": base64.b64encode(prefix).decode("ascii"),
        "response_text_truncated": len(prefix) != len(response_bytes),
        "failure_reason": failure_reason,
        "payload_status": payload_status,
        "call_receipt": call_receipt,
    }
    return AssistantRouteCallEvidence.model_validate(
        {**unsigned, "evidence_sha256": _hash_payload(unsigned)}, strict=True
    )


class AssistantSharedRouteReference(_StrictFrozenModel):
    """Non-billable reference to one separately persisted shared route call."""

    artifact_sha256: Sha256
    route_call_response_sha256: Sha256
    reserved_usage: LLMUsage
    reserved_turns: Literal[1] = 1


class AssistantRouteAttempt(_StrictFrozenModel):
    """Internal record of the routing decision, retained across later errors.

    The public Assistant result may be an error row, but an already validated
    route remains essential evidence for attribution: a correct route followed
    by a failed tool call is a tool failure, not an unobservable routing
    failure.  This receipt is intentionally kept only in runner-owned evidence
    and is never copied into a Final Judge packet.
    """

    schema_version: Literal[1] = 1
    status: Literal["not_applicable", "selected", "failed"]
    selected_capability: str | None = None
    skill_slug: str | None = None
    route_trace_sha256: Sha256 | None = None
    error_code: Literal["timeout", "runtime_error"] | None = None
    route_attempt_sha256: Sha256

    @field_validator("selected_capability", "skill_slug", "error_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        routing = (
            self.selected_capability,
            self.skill_slug,
            self.route_trace_sha256,
        )
        if self.status == "selected":
            if not all(value is not None for value in routing) or self.error_code:
                raise ValueError(
                    "selected route attempt must retain full route identity"
                )
        elif self.status == "failed":
            if any(value is not None for value in routing) or self.error_code is None:
                raise ValueError("failed route attempt must retain only its error")
        elif any(value is not None for value in (*routing, self.error_code)):
            raise ValueError("not-applicable route attempt must not retain route data")
        if self.route_attempt_sha256 != _self_hash(self, "route_attempt_sha256"):
            raise ValueError("route attempt self hash mismatch")
        return self


def make_assistant_route_attempt(
    *,
    status: Literal["not_applicable", "selected", "failed"],
    selected_capability: str | None = None,
    skill_slug: str | None = None,
    route_trace_sha256: str | None = None,
    error_code: Literal["timeout", "runtime_error"] | None = None,
) -> AssistantRouteAttempt:
    """Create one self-authenticating route-attempt receipt."""

    unsigned = {
        "schema_version": 1,
        "status": status,
        "selected_capability": selected_capability,
        "skill_slug": skill_slug,
        "route_trace_sha256": route_trace_sha256,
        "error_code": error_code,
    }
    return AssistantRouteAttempt.model_validate(
        {**unsigned, "route_attempt_sha256": _hash_payload(unsigned)}, strict=True
    )


class AssistantExecutionReceipt(_StrictFrozenModel):
    """Runner-owned provenance, never accepted from a diagnostic backend."""

    schema_version: Literal[1] = 1
    policy_version: Literal[
        "runner-owned-assistant-v1",
        "runner-owned-assistant-v2",
        "runner-owned-assistant-v3",
    ] = ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION
    request_sha256: Sha256
    asset_catalog_sha256: Sha256 | None
    query_asset_id: str | None
    query_asset_sha256: Sha256 | None
    shared_route_reference: AssistantSharedRouteReference | None = None
    model_calls: tuple[AssistantModelCallReceipt, ...]
    tool_trace: tuple[AssistantToolTrace, ...]
    # ``None`` preserves the canonical bytes of immutable pre-route-attempt
    # runner bundles.  New runner-owned receipts must carry an attempt.
    route_attempt: AssistantRouteAttempt | None = None
    route_call_evidence: AssistantRouteCallEvidence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    aggregate_usage: LLMUsage
    runner_latency_ms: int = Field(ge=0)
    outcome: Literal["success", "timeout", "runtime_error"]
    response_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("model_calls", "tool_trace", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler):
        """Keep legacy receipt hashes stable when route_attempt is absent."""

        payload = handler(self)
        if self.route_attempt is None:
            payload.pop("route_attempt", None)
        if self.route_call_evidence is None:
            payload.pop("route_call_evidence", None)
        return payload

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if tuple(item.call_index for item in self.model_calls) != tuple(
            range(1, len(self.model_calls) + 1)
        ):
            raise ValueError("runner model-call indices must be contiguous")
        if self.aggregate_usage != LLMUsage(
            input_tokens=sum(item.input_tokens for item in self.model_calls),
            output_tokens=sum(item.output_tokens for item in self.model_calls),
        ):
            raise ValueError("runner aggregate usage differs from model-call receipts")
        if self.outcome == "success" and not self.model_calls:
            raise ValueError("successful runner receipt requires a real model call")
        asset_identity = (
            self.asset_catalog_sha256,
            self.query_asset_id,
            self.query_asset_sha256,
        )
        if self.outcome == "success" and not all(asset_identity):
            raise ValueError("successful runner receipt requires asset provenance")
        if any(item is None for item in asset_identity) != all(
            item is None for item in asset_identity
        ):
            raise ValueError("runner asset provenance must be complete or absent")
        if self.route_attempt is not None and self.route_attempt.status == "selected":
            if not self.model_calls:
                raise ValueError("selected route attempt requires a model-call receipt")
        route_evidence = self.route_call_evidence
        if route_evidence is not None:
            if (
                self.shared_route_reference is not None
                or not self.model_calls
                or route_evidence.call_receipt != self.model_calls[0]
            ):
                raise ValueError(
                    "route-call evidence differs from the execution receipt"
                )
        if (
            self.policy_version == ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION
            and self.route_attempt is not None
            and self.route_attempt.status != "not_applicable"
            and self.shared_route_reference is None
            and self.model_calls
            and route_evidence is None
        ):
            raise ValueError("active captured route call requires bounded evidence")
        if (
            self.policy_version == ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION
            and route_evidence is not None
            and route_evidence.policy_version
            != ASSISTANT_ROUTE_CALL_EVIDENCE_POLICY_VERSION
        ):
            raise ValueError("active runner receipt requires active route evidence")
        receipt_payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.shared_route_reference is None:
            receipt_payload.pop("shared_route_reference")
        if self.receipt_sha256 != _hash_payload(receipt_payload):
            raise ValueError("runner receipt self hash mismatch")
        return self


_RUNNER_EXECUTION_MARKER = object()


@dataclass(frozen=True)
class RunnerOwnedAssistantExecution:
    response: AssistantBackendResponse
    receipt: AssistantExecutionReceipt
    scorer_calls: tuple[PublicScorerCallEvidenceV2, ...] = ()
    scorer_capture_policy_version: (
        Literal["portfolio-gcs-scorer-evidence-v2"] | None
    ) = None
    _marker: object = field(
        default=_RUNNER_EXECUTION_MARKER,
        repr=False,
        compare=False,
    )


def _issue_runner_owned_assistant_execution(
    response: AssistantBackendResponse,
    receipt: AssistantExecutionReceipt,
    *,
    scorer_calls: Sequence[PublicScorerCallEvidenceV2] = (),
    scorer_capture_policy_version: (
        Literal["portfolio-gcs-scorer-evidence-v2"] | None
    ) = None,
) -> RunnerOwnedAssistantExecution:
    """Internal capability used by the reviewed production runner module."""

    if type(response) is not AssistantBackendResponse:
        raise TypeError("runner response must be exactly AssistantBackendResponse")
    if type(receipt) is not AssistantExecutionReceipt:
        raise TypeError("runner receipt must be exactly AssistantExecutionReceipt")
    captured = tuple(scorer_calls)
    if any(type(item) is not PublicScorerCallEvidenceV2 for item in captured):
        raise TypeError("runner scorer calls must be validated v2 evidence")
    if scorer_capture_policy_version is None and captured:
        raise TypeError("runner scorer calls require an explicit capture policy")
    if scorer_capture_policy_version not in {
        None,
        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    }:
        raise TypeError("runner scorer capture policy is unsupported")
    return RunnerOwnedAssistantExecution(
        response=response,
        receipt=receipt,
        scorer_calls=captured,
        scorer_capture_policy_version=scorer_capture_policy_version,
        _marker=_RUNNER_EXECUTION_MARKER,
    )


class AssistantRunResultRow(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    query_ordinal: int = Field(ge=0)
    request_sha256: Sha256
    turn_count: int = Field(ge=1)
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"]
    execution_receipt: AssistantExecutionReceipt | None
    result: AssistantResult
    row_sha256: Sha256

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if (self.execution_receipt is None) != (
            self.execution_provenance == "backend-reported-v1"
        ):
            raise ValueError("Assistant row provenance/receipt mismatch")
        if self.row_sha256 != _self_hash(self, "row_sha256"):
            raise ValueError("Assistant result row self hash mismatch")
        return self


class RunArtifactDescriptor(_StrictFrozenModel):
    path: Literal["run-spec.json", "requests.jsonl", "results.jsonl"]
    bytes: int = Field(ge=0)
    sha256: Sha256
    rows: int = Field(ge=0)


class AssistantRunSpec(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["assistant-run-spec"] = "assistant-run-spec"
    policy_version: Literal["assistant-run-bundle-v1"] = ASSISTANT_RUN_POLICY_VERSION
    matrix_run_id: str
    config: AssistantRunConfig
    matrix_plan_sha256: Sha256
    query_count: int = Field(gt=0)
    query_order_sha256: Sha256
    backbone_identity_sha256: Sha256
    budget_sha256: Sha256
    registry_sha256: Sha256
    registry_runtime_sha256: Sha256
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"]
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    spec_sha256: Sha256

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _nonblank(value, "matrix_run_id")

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.spec_sha256 != _self_hash(self, "spec_sha256"):
            raise ValueError("Assistant run spec self hash mismatch")
        return self


class AssistantRunManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["assistant-run-bundle"] = "assistant-run-bundle"
    policy_version: Literal["assistant-run-bundle-v1"] = ASSISTANT_RUN_POLICY_VERSION
    matrix_run_id: str
    config: AssistantRunConfig
    status: Literal["complete"] = "complete"
    matrix_plan_sha256: Sha256
    spec_sha256: Sha256
    query_count: int = Field(gt=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    artifacts: dict[str, RunArtifactDescriptor]
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"]
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.success_count + self.error_count != self.query_count:
            raise ValueError("Assistant success/error counts must cover denominator")
        if set(self.artifacts) != set(_ARTIFACT_FILES):
            raise ValueError("Assistant run manifest artifact set is invalid")
        if any(name != item.path for name, item in self.artifacts.items()):
            raise ValueError("Assistant run artifact descriptor path mismatch")
        if self.manifest_sha256 != _self_hash(self, "manifest_sha256"):
            raise ValueError("Assistant run manifest self hash mismatch")
        return self


class AssistantBackend(Protocol):
    def execute(self, request: AssistantRequestSnapshot) -> AssistantBackendResponse:
        """Execute exactly one frozen request and return audited runtime identity."""


@dataclass(frozen=True)
class CreatedAssistantRunBundle:
    root: Path
    config: AssistantRunConfig
    external_manifest_sha256: str


@dataclass(frozen=True)
class VerifiedAssistantRunBundle:
    """Deep-verified evidence; still formally ineligible pending calibration."""

    root: Path
    manifest: AssistantRunManifest
    spec: AssistantRunSpec
    requests: tuple[AssistantRequestSnapshot, ...]
    rows: tuple[AssistantRunResultRow, ...]
    external_manifest_sha256: str
    _snapshot: tuple[tuple[str, bytes], ...] = field(repr=False)
    _marker: object = field(repr=False)


@dataclass(frozen=True)
class VerifiedFiveConfigRuns:
    plan_sha256: str
    matrix_run_id: str
    query_artifact_sha256: str
    split_manifest_sha256: str
    query_order_sha256: str
    query_set_sha256: str
    runs: tuple[VerifiedAssistantRunBundle, ...]
    judge_audit_selection_manifest_sha256: str
    judge_audit_evaluation_ids: tuple[str, ...]
    judge_audit_evaluation_ids_sha256: str
    final_rubric_sha256: str
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: str = FORMAL_INELIGIBLE_REASON
    _marker: object = field(default=None, repr=False)


def _treatment_for(plan: AssistantMatrixPlan, config: str) -> MatrixTreatment:
    if config == "spec_baseline":
        raise AssistantRunError(
            "SpecBaseline is diagnostic-only and cannot enter the five-config run"
        )
    for treatment in plan.treatments:
        if treatment.config == config:
            return treatment
    raise AssistantRunError("unknown Assistant configuration")


def _request_for(
    plan: AssistantMatrixPlan,
    treatment: MatrixTreatment,
    query: AssistantQueryInput,
    query_ordinal: int,
) -> AssistantRequestSnapshot:
    unsigned = {
        "schema_version": 1,
        "matrix_run_id": plan.matrix_run_id,
        "config": treatment.config,
        "query_ordinal": query_ordinal,
        "query": query,
        "treatment": treatment,
        "backbone": plan.backbone,
        "budget": plan.budget,
        "registry": plan.registry,
    }
    return AssistantRequestSnapshot.model_validate(
        {**unsigned, "request_sha256": _hash_payload(unsigned)}, strict=True
    )


def _validate_backend_response(
    response: AssistantBackendResponse,
    request: AssistantRequestSnapshot,
) -> None:
    expected = (
        request.request_sha256,
        request.backbone.provider,
        request.backbone.model,
        request.backbone.endpoint,
        request.backbone.identity_sha256,
        request.registry.registry_sha256,
        request.registry.registry_runtime_sha256,
        request.budget.budget_sha256,
    )
    actual = (
        response.request_sha256,
        response.backbone_provider,
        response.backbone_model,
        response.backbone_endpoint,
        response.backbone_identity_sha256,
        response.registry_sha256,
        response.registry_runtime_sha256,
        response.budget_sha256,
    )
    if actual != expected:
        raise AssistantBackendContractError(
            "backend runtime identity differs from the frozen matrix request"
        )
    if response.error_code is None and response.backbone_request_id is None:
        raise AssistantBackendContractError(
            "successful backend response omitted provider request identity"
        )
    if response.usage.input_tokens > request.budget.max_input_tokens:
        raise AssistantBackendContractError(
            "backend exceeded frozen input-token budget"
        )
    if response.usage.output_tokens > request.budget.max_output_tokens:
        raise AssistantBackendContractError(
            "backend exceeded frozen output-token budget"
        )
    if len(response.tool_trace) > request.budget.max_tool_calls:
        raise AssistantBackendContractError("backend exceeded frozen tool-call budget")
    if response.turn_count > request.budget.max_turns:
        raise AssistantBackendContractError("backend exceeded frozen turn budget")
    if (
        response.latency_ms > request.budget.timeout_ms
        and response.error_code != "timeout"
    ):
        raise AssistantBackendContractError(
            "backend exceeded timeout without retaining a timeout error"
        )
    runtime_by_tool = {
        item.tool_name: item.runtime_binding_sha256 for item in request.registry.tools
    }
    if any(
        trace.runtime_binding_sha256 != runtime_by_tool[trace.tool_name]
        for trace in response.tool_trace
    ):
        raise AssistantBackendContractError(
            "tool trace runtime differs from frozen registry"
        )
    routing = (
        response.selected_capability,
        response.skill_slug,
        response.route_trace_sha256,
    )
    if request.config == "noskill" and any(value is not None for value in routing):
        raise AssistantBackendContractError("NoSkill backend returned Skill routing")
    if (
        request.config != "noskill"
        and response.error_code is None
        and not all(value is not None for value in routing)
    ):
        raise AssistantBackendContractError(
            "successful Skill backend omitted routing identity"
        )


def _exception_response(
    request: AssistantRequestSnapshot, *, latency_ms: int
) -> AssistantBackendResponse:
    return AssistantBackendResponse(
        request_sha256=request.request_sha256,
        backbone_provider=request.backbone.provider,
        backbone_model=request.backbone.model,
        backbone_endpoint=request.backbone.endpoint,
        backbone_identity_sha256=request.backbone.identity_sha256,
        registry_sha256=request.registry.registry_sha256,
        registry_runtime_sha256=request.registry.registry_runtime_sha256,
        budget_sha256=request.budget.budget_sha256,
        response_text="",
        usage=LLMUsage(input_tokens=0, output_tokens=0),
        turn_count=1,
        latency_ms=latency_ms,
        error_code=(
            "timeout" if latency_ms > request.budget.timeout_ms else "runtime_error"
        ),
    )


def _result_row(
    *,
    plan: AssistantMatrixPlan,
    request: AssistantRequestSnapshot,
    response: AssistantBackendResponse,
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"],
    execution_receipt: AssistantExecutionReceipt | None,
) -> AssistantRunResultRow:
    routing_present = response.selected_capability is not None
    result = AssistantResult(
        run_id=plan.matrix_run_id,
        query_id=request.query.query_id,
        config=request.config,
        response_text=response.response_text,
        visible_cards=response.visible_cards,
        visible_tool_evidence=response.visible_tool_evidence,
        tool_trace=response.tool_trace,
        selected_capability=response.selected_capability,
        skill_slug=response.skill_slug,
        bank_sha256=(request.treatment.bank_sha256 if routing_present else None),
        route_trace_sha256=response.route_trace_sha256,
        query_artifact_sha256=plan.query_artifact_sha256,
        split_manifest_sha256=plan.split_manifest_sha256,
        registry_sha256=plan.registry.registry_sha256,
        registry_runtime_sha256=plan.registry.registry_runtime_sha256,
        backbone_provider=plan.backbone.provider,
        backbone_model=plan.backbone.model,
        backbone_request_id=response.backbone_request_id,
        usage=response.usage,
        latency_ms=response.latency_ms,
        error_code=response.error_code,
    )
    unsigned = {
        "schema_version": 1,
        "query_ordinal": request.query_ordinal,
        "request_sha256": request.request_sha256,
        "turn_count": response.turn_count,
        "execution_provenance": execution_provenance,
        "execution_receipt": execution_receipt,
        "result": result,
    }
    return AssistantRunResultRow.model_validate(
        {**unsigned, "row_sha256": _hash_payload(unsigned)}, strict=True
    )


def _run_spec(
    plan: AssistantMatrixPlan,
    config: AssistantRunConfig,
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"],
) -> AssistantRunSpec:
    unsigned = {
        "schema_version": 1,
        "kind": "assistant-run-spec",
        "policy_version": ASSISTANT_RUN_POLICY_VERSION,
        "matrix_run_id": plan.matrix_run_id,
        "config": config,
        "matrix_plan_sha256": plan.plan_sha256,
        "query_count": len(plan.queries),
        "query_order_sha256": plan.query_order_sha256,
        "backbone_identity_sha256": plan.backbone.identity_sha256,
        "budget_sha256": plan.budget.budget_sha256,
        "registry_sha256": plan.registry.registry_sha256,
        "registry_runtime_sha256": plan.registry.registry_runtime_sha256,
        "execution_provenance": execution_provenance,
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    return AssistantRunSpec.model_validate(
        {**unsigned, "spec_sha256": _hash_payload(unsigned)}, strict=True
    )


def _exception_receipt(
    request: AssistantRequestSnapshot,
    response: AssistantBackendResponse,
) -> AssistantExecutionReceipt:
    outcome = "timeout" if response.error_code == "timeout" else "runtime_error"
    route_attempt = make_assistant_route_attempt(
        status="not_applicable" if request.config == "noskill" else "failed",
        error_code=None if request.config == "noskill" else outcome,
    )
    unsigned = {
        "schema_version": 1,
        "policy_version": ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
        "request_sha256": request.request_sha256,
        "asset_catalog_sha256": None,
        "query_asset_id": None,
        "query_asset_sha256": None,
        "model_calls": (),
        "tool_trace": (),
        "route_attempt": route_attempt,
        "aggregate_usage": response.usage,
        "runner_latency_ms": response.latency_ms,
        "outcome": outcome,
        "response_sha256": _hash_payload(response),
    }
    return AssistantExecutionReceipt.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )


def _validate_runner_execution(
    execution: RunnerOwnedAssistantExecution,
    request: AssistantRequestSnapshot,
) -> None:
    if (
        type(execution) is not RunnerOwnedAssistantExecution
        or execution._marker is not _RUNNER_EXECUTION_MARKER
    ):
        raise AssistantBackendContractError(
            "production runner returned an unauthorised execution wrapper"
        )
    response = execution.response
    receipt = execution.receipt
    successful_trace = tuple(
        (
            item.call_index,
            item.tool_name,
            item.arguments_sha256,
            item.result_sha256,
        )
        for item in response.tool_trace
        if item.status == "success"
    )
    scorer_trace = tuple(
        (
            item.call_index,
            item.tool_name,
            item.arguments_sha256,
            item.result_sha256,
        )
        for item in execution.scorer_calls
    )
    if (
        receipt.request_sha256 != request.request_sha256
        or receipt.response_sha256 != _hash_payload(response)
        or receipt.aggregate_usage != response.usage
        or receipt.runner_latency_ms != response.latency_ms
        or receipt.tool_trace != response.tool_trace
        or (receipt.outcome == "success") != (response.error_code is None)
        or (
            execution.scorer_capture_policy_version
            == GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
            and scorer_trace != successful_trace
        )
        or (execution.scorer_capture_policy_version is None and execution.scorer_calls)
    ):
        raise AssistantBackendContractError(
            "runner-owned receipt differs from the captured execution"
        )
    route_attempt = receipt.route_attempt
    if route_attempt is None:
        raise AssistantBackendContractError(
            "runner-owned receipt omitted the route-attempt record"
        )
    response_routing = (
        response.selected_capability,
        response.skill_slug,
        response.route_trace_sha256,
    )
    if request.config == "noskill":
        if route_attempt.status != "not_applicable" or any(
            value is not None for value in response_routing
        ):
            raise AssistantBackendContractError(
                "NoSkill runner receipt retained an invalid route attempt"
            )
    elif route_attempt.status == "selected":
        expected_routing = (
            route_attempt.selected_capability,
            route_attempt.skill_slug,
            route_attempt.route_trace_sha256,
        )
        if response_routing != expected_routing:
            raise AssistantBackendContractError(
                "runner route attempt differs from the captured response"
            )
    elif route_attempt.status == "failed":
        if any(value is not None for value in response_routing):
            raise AssistantBackendContractError(
                "failed runner route attempt retained a route identity"
            )
    else:  # pragma: no cover - Literal is validated above
        raise AssistantBackendContractError("runner route attempt status is invalid")
    try:
        public = parse_canonical_json(
            request.query.public_input_json.encode("utf-8"),
            label="runner-owned request public input",
        )
    except ArtifactFormatError as error:
        raise AssistantBackendContractError(
            "runner-owned request public input is invalid"
        ) from error
    if not isinstance(public, dict):
        raise AssistantBackendContractError(
            "runner-owned request public input is invalid"
        )
    expected_asset_id = (
        request.query.asset_binding.asset_id
        if request.query.asset_binding is not None
        else public.get("asset_id")
    )
    if (
        not isinstance(expected_asset_id, str)
        or receipt.query_asset_id != expected_asset_id
    ):
        raise AssistantBackendContractError(
            "runner asset receipt differs from the frozen authoritative binding"
        )
    for call in receipt.model_calls:
        if (
            call.provider != request.backbone.provider
            or call.endpoint != request.backbone.endpoint
            or call.requested_model != request.backbone.model
            or call.response_model != request.backbone.model
        ):
            raise AssistantBackendContractError(
                "runner model-call receipt differs from the backbone lock"
            )


def create_assistant_run_bundle(
    verified_plan: VerifiedAssistantMatrixPlan,
    *,
    config: str,
    backend: AssistantBackend,
    registry: ToolRegistry,
    output_dir: str | Path,
) -> CreatedAssistantRunBundle:
    """Diagnostic entry point accepting backend-reported execution metadata."""

    return _create_assistant_run_bundle(
        verified_plan,
        config=config,
        backend=backend,
        runner=None,
        registry=registry,
        output_dir=output_dir,
        execution_provenance="backend-reported-v1",
    )


def create_runner_owned_assistant_run_bundle(
    verified_plan: VerifiedAssistantMatrixPlan,
    *,
    config: str,
    runner: object,
    registry: ToolRegistry,
    output_dir: str | Path,
) -> CreatedAssistantRunBundle:
    """Production entry point accepting only the repository's reviewed runner."""

    from skillchain.runners.assistant import require_production_assistant_runner

    runner = require_production_assistant_runner(runner)
    if object.__getattribute__(runner, "_registry") is not registry:
        raise AssistantRunError("production runner registry object differs from caller")
    return _create_assistant_run_bundle(
        verified_plan,
        config=config,
        backend=None,
        runner=runner,
        registry=registry,
        output_dir=output_dir,
        execution_provenance="runner-owned-v1",
    )


def _create_assistant_run_bundle(
    verified_plan: VerifiedAssistantMatrixPlan,
    *,
    config: str,
    backend: AssistantBackend | None,
    runner: object | None,
    registry: ToolRegistry,
    output_dir: str | Path,
    execution_provenance: Literal["backend-reported-v1", "runner-owned-v1"],
) -> CreatedAssistantRunBundle:
    """Run one treatment and atomically publish an immutable evidence bundle."""

    verified_plan = require_verified_assistant_matrix_plan(
        verified_plan, registry=registry
    )
    plan = verified_plan.plan
    treatment = _treatment_for(plan, config)
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"Assistant run destination already exists; refusing overwrite: {output_dir}"
        )
    staging = new_staging_directory(output_dir)
    try:
        requests: list[AssistantRequestSnapshot] = []
        rows: list[AssistantRunResultRow] = []
        for ordinal, query in enumerate(plan.queries):
            _verify_registry_lock(plan.registry, registry)
            request = _request_for(plan, treatment, query, ordinal)
            requests.append(request)
            started = time.perf_counter_ns()
            execution_receipt: AssistantExecutionReceipt | None = None
            try:
                if execution_provenance == "runner-owned-v1":
                    execution = runner.execute(request.model_copy(deep=True))
                    response = execution.response
                    execution_receipt = execution.receipt
                else:
                    response = backend.execute(request.model_copy(deep=True))
            except Exception:
                elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
                response = _exception_response(request, latency_ms=elapsed_ms)
                if execution_provenance == "runner-owned-v1":
                    execution_receipt = _exception_receipt(request, response)
            else:
                if execution_provenance == "runner-owned-v1":
                    _validate_runner_execution(execution, request)
            if type(response) is not AssistantBackendResponse:
                raise AssistantBackendContractError(
                    "backend must return exactly AssistantBackendResponse"
                )
            _validate_backend_response(response, request)
            _verify_registry_lock(plan.registry, registry)
            rows.append(
                _result_row(
                    plan=plan,
                    request=request,
                    response=response,
                    execution_provenance=execution_provenance,
                    execution_receipt=execution_receipt,
                )
            )

        spec = _run_spec(plan, treatment.config, execution_provenance)
        artifact_bytes = {
            _SPEC_FILE: canonical_json_bytes(spec.model_dump(mode="json")),
            _REQUESTS_FILE: canonical_jsonl_bytes(
                tuple(item.model_dump(mode="json") for item in requests)
            ),
            _RESULTS_FILE: canonical_jsonl_bytes(
                tuple(item.model_dump(mode="json") for item in rows)
            ),
        }
        for name, content in artifact_bytes.items():
            atomic_create_file(staging / name, content)
        descriptors = {
            name: RunArtifactDescriptor(
                path=name,
                bytes=len(content),
                sha256=sha256_bytes(content),
                rows=(
                    1
                    if name == _SPEC_FILE
                    else len(requests)
                    if name == _REQUESTS_FILE
                    else len(rows)
                ),
            )
            for name, content in artifact_bytes.items()
        }
        success_count = sum(row.result.error_code is None for row in rows)
        unsigned_manifest = {
            "schema_version": 1,
            "kind": "assistant-run-bundle",
            "policy_version": ASSISTANT_RUN_POLICY_VERSION,
            "matrix_run_id": plan.matrix_run_id,
            "config": treatment.config,
            "status": "complete",
            "matrix_plan_sha256": plan.plan_sha256,
            "spec_sha256": spec.spec_sha256,
            "query_count": len(rows),
            "success_count": success_count,
            "error_count": len(rows) - success_count,
            "artifacts": {
                name: item.model_dump(mode="json") for name, item in descriptors.items()
            },
            "execution_provenance": execution_provenance,
            "formal_eligible": False,
            "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
        }
        manifest = AssistantRunManifest.model_validate(
            {
                **unsigned_manifest,
                "manifest_sha256": _hash_payload(unsigned_manifest),
            },
            strict=True,
        )
        manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
        atomic_create_file(staging / _MANIFEST_FILE, manifest_bytes)
        require_verified_assistant_matrix_plan(verified_plan, registry=registry)
        atomic_publish_new_directory(staging, output_dir)
        return CreatedAssistantRunBundle(
            root=output_dir,
            config=treatment.config,
            external_manifest_sha256=sha256_bytes(manifest_bytes),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_verified_assistant_run_bundle(
    run_dir: str | Path,
    verified_plan: VerifiedAssistantMatrixPlan,
    *,
    expected_manifest_sha256: str,
    registry: ToolRegistry,
) -> VerifiedAssistantRunBundle:
    """Deeply revalidate a bundle against external plan and manifest digests."""

    verified_plan = require_verified_assistant_matrix_plan(
        verified_plan, registry=registry
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected run manifest sha256"
    )
    root = Path(run_dir)
    _require_exact_bundle_directory(root)
    before = _read_bundle(root)
    if sha256_bytes(before[_MANIFEST_FILE]) != expected_manifest_sha256:
        raise AssistantRunError("Assistant run external manifest digest mismatch")
    manifest = _load_model(before[_MANIFEST_FILE], AssistantRunManifest, _MANIFEST_FILE)
    spec = _load_model(before[_SPEC_FILE], AssistantRunSpec, _SPEC_FILE)
    requests = _load_jsonl_models(
        before[_REQUESTS_FILE], AssistantRequestSnapshot, _REQUESTS_FILE
    )
    rows = _load_jsonl_models(
        before[_RESULTS_FILE], AssistantRunResultRow, _RESULTS_FILE
    )
    _verify_run_bundle(
        verified_plan.plan,
        manifest=manifest,
        spec=spec,
        requests=requests,
        rows=rows,
        bundle_bytes=before,
    )
    _require_exact_bundle_directory(root)
    after = _read_bundle(root)
    if after != before:
        raise AssistantRunError("Assistant run bundle changed during verification")
    require_verified_assistant_matrix_plan(verified_plan, registry=registry)
    return VerifiedAssistantRunBundle(
        root=root,
        manifest=manifest,
        spec=spec,
        requests=requests,
        rows=rows,
        external_manifest_sha256=expected_manifest_sha256,
        _snapshot=tuple((name, before[name]) for name in _BUNDLE_FILES),
        _marker=_RUN_MARKER,
    )


def require_verified_assistant_run_bundle(
    value: object,
) -> VerifiedAssistantRunBundle:
    if (
        not isinstance(value, VerifiedAssistantRunBundle)
        or value._marker is not _RUN_MARKER
    ):
        raise TypeError("run must be returned by load_verified_assistant_run_bundle")
    _require_exact_bundle_directory(value.root)
    current = _read_bundle(value.root)
    _require_exact_bundle_directory(value.root)
    if tuple((name, current[name]) for name in _BUNDLE_FILES) != value._snapshot:
        raise AssistantRunError("verified Assistant run bundle changed on disk")
    reparsed = (
        _load_model(current[_MANIFEST_FILE], AssistantRunManifest, _MANIFEST_FILE),
        _load_model(current[_SPEC_FILE], AssistantRunSpec, _SPEC_FILE),
        _load_jsonl_models(
            current[_REQUESTS_FILE], AssistantRequestSnapshot, _REQUESTS_FILE
        ),
        _load_jsonl_models(
            current[_RESULTS_FILE], AssistantRunResultRow, _RESULTS_FILE
        ),
    )
    if reparsed != (value.manifest, value.spec, value.requests, value.rows):
        raise AssistantRunError("verified Assistant run handle was mutated")
    return value


def load_verified_five_config_runs(
    verified_plan: VerifiedAssistantMatrixPlan,
    *,
    bundle_locks: Mapping[str, tuple[str | Path, str]],
    registry: ToolRegistry,
) -> VerifiedFiveConfigRuns:
    """Verify exactly one externally locked run for every main treatment."""

    verified_plan = require_verified_assistant_matrix_plan(
        verified_plan, registry=registry
    )
    if set(bundle_locks) != set(MAIN_CONFIG_ORDER):
        raise AssistantRunError(
            "five-run gate requires exactly noskill, llm_static, s1, s1s2, full"
        )
    runs = tuple(
        load_verified_assistant_run_bundle(
            bundle_locks[config][0],
            verified_plan,
            expected_manifest_sha256=bundle_locks[config][1],
            registry=registry,
        )
        for config in MAIN_CONFIG_ORDER
    )
    if tuple(run.manifest.config for run in runs) != MAIN_CONFIG_ORDER:
        raise AssistantRunError("five-run gate configuration order mismatch")
    for run in runs:
        require_verified_assistant_run_bundle(run)
    return VerifiedFiveConfigRuns(
        plan_sha256=verified_plan.plan.plan_sha256,
        matrix_run_id=verified_plan.plan.matrix_run_id,
        query_artifact_sha256=verified_plan.plan.query_artifact_sha256,
        split_manifest_sha256=verified_plan.plan.split_manifest_sha256,
        query_order_sha256=verified_plan.plan.query_order_sha256,
        query_set_sha256=verified_plan.plan.query_set_sha256,
        runs=runs,
        judge_audit_selection_manifest_sha256=(
            verified_plan.plan.judge_audit_selection_manifest_sha256
        ),
        judge_audit_evaluation_ids=(verified_plan.plan.judge_audit_evaluation_ids),
        judge_audit_evaluation_ids_sha256=(
            verified_plan.plan.judge_audit_evaluation_ids_sha256
        ),
        final_rubric_sha256=verified_plan.plan.final_rubric_sha256,
        _marker=_FIVE_RUN_MARKER,
    )


def require_verified_five_config_runs(value: object) -> VerifiedFiveConfigRuns:
    if (
        not isinstance(value, VerifiedFiveConfigRuns)
        or value._marker is not _FIVE_RUN_MARKER
    ):
        raise TypeError(
            "judge audit requires five externally verified Assistant run bundles"
        )
    if tuple(run.manifest.config for run in value.runs) != MAIN_CONFIG_ORDER:
        raise AssistantRunError("verified five-run set was mutated or is incomplete")
    if value.judge_audit_evaluation_ids_sha256 != _hash_payload(
        list(value.judge_audit_evaluation_ids)
    ):
        raise AssistantRunError("verified five-run audit commitment is invalid")
    if value.query_order_sha256 != _hash_payload(
        [request.query.query_id for request in value.runs[0].requests]
    ) or value.query_set_sha256 != _hash_payload(
        [request.query.model_dump(mode="json") for request in value.runs[0].requests]
    ):
        raise AssistantRunError("verified five-run query commitment is invalid")
    if any(
        require_verified_assistant_run_bundle(run).manifest.matrix_plan_sha256
        != value.plan_sha256
        for run in value.runs
    ):
        raise AssistantRunError("verified five-run set mixes matrix plans")
    return value


def _verify_run_bundle(
    plan: AssistantMatrixPlan,
    *,
    manifest: AssistantRunManifest,
    spec: AssistantRunSpec,
    requests: tuple[AssistantRequestSnapshot, ...],
    rows: tuple[AssistantRunResultRow, ...],
    bundle_bytes: dict[str, bytes],
) -> None:
    treatment = _treatment_for(plan, manifest.config)
    expected_common = (
        plan.matrix_run_id,
        treatment.config,
        plan.plan_sha256,
        len(plan.queries),
    )
    if (
        manifest.matrix_run_id,
        manifest.config,
        manifest.matrix_plan_sha256,
        manifest.query_count,
    ) != expected_common:
        raise AssistantRunError("Assistant manifest differs from frozen matrix plan")
    if (
        spec.matrix_run_id,
        spec.config,
        spec.matrix_plan_sha256,
        spec.query_count,
    ) != expected_common:
        raise AssistantRunError("Assistant run spec differs from frozen matrix plan")
    if manifest.spec_sha256 != spec.spec_sha256:
        raise AssistantRunError("Assistant manifest/spec hash mismatch")
    if manifest.execution_provenance != spec.execution_provenance:
        raise AssistantRunError("Assistant manifest/spec provenance mismatch")
    if (
        spec.query_order_sha256,
        spec.backbone_identity_sha256,
        spec.budget_sha256,
        spec.registry_sha256,
        spec.registry_runtime_sha256,
    ) != (
        plan.query_order_sha256,
        plan.backbone.identity_sha256,
        plan.budget.budget_sha256,
        plan.registry.registry_sha256,
        plan.registry.registry_runtime_sha256,
    ):
        raise AssistantRunError("Assistant run invariant lock mismatch")
    if len(requests) != len(plan.queries) or len(rows) != len(plan.queries):
        raise AssistantRunError(
            "Assistant run must retain exactly one request/result per query"
        )
    expected_requests = tuple(
        _request_for(plan, treatment, query, ordinal)
        for ordinal, query in enumerate(plan.queries)
    )
    if requests != expected_requests:
        raise AssistantRunError(
            "Assistant request rows differ from frozen query order or config"
        )
    for ordinal, (request, row) in enumerate(zip(requests, rows, strict=True)):
        if row.query_ordinal != ordinal or row.request_sha256 != request.request_sha256:
            raise AssistantRunError("Assistant result/request row alignment mismatch")
        _verify_result_against_request(plan, request, row)
        if row.execution_provenance != manifest.execution_provenance:
            raise AssistantRunError("Assistant row provenance differs from run spec")
    if tuple(row.result.query_id for row in rows) != tuple(
        query.query_id for query in plan.queries
    ):
        raise AssistantRunError("Assistant result query coverage/order mismatch")
    success_count = sum(row.result.error_code is None for row in rows)
    if (manifest.success_count, manifest.error_count) != (
        success_count,
        len(rows) - success_count,
    ):
        raise AssistantRunError("Assistant result denominator counts are invalid")
    expected_rows = {
        _SPEC_FILE: 1,
        _REQUESTS_FILE: len(requests),
        _RESULTS_FILE: len(rows),
    }
    for name in _ARTIFACT_FILES:
        descriptor = manifest.artifacts[name]
        content = bundle_bytes[name]
        if (
            descriptor.bytes != len(content)
            or descriptor.sha256 != sha256_bytes(content)
            or descriptor.rows != expected_rows[name]
        ):
            raise AssistantRunError(f"Assistant artifact descriptor mismatch: {name}")


def _verify_result_against_request(
    plan: AssistantMatrixPlan,
    request: AssistantRequestSnapshot,
    row: AssistantRunResultRow,
) -> None:
    result = row.result
    if (
        result.run_id,
        result.query_id,
        result.config,
        result.query_artifact_sha256,
        result.split_manifest_sha256,
        result.registry_sha256,
        result.registry_runtime_sha256,
        result.backbone_provider,
        result.backbone_model,
    ) != (
        plan.matrix_run_id,
        request.query.query_id,
        request.config,
        plan.query_artifact_sha256,
        plan.split_manifest_sha256,
        plan.registry.registry_sha256,
        plan.registry.registry_runtime_sha256,
        plan.backbone.provider,
        plan.backbone.model,
    ):
        raise AssistantRunError("Assistant result identity differs from request lock")
    if row.turn_count > plan.budget.max_turns:
        raise AssistantRunError("Assistant result exceeds frozen turn budget")
    if result.usage.input_tokens > plan.budget.max_input_tokens:
        raise AssistantRunError("Assistant result exceeds frozen input-token budget")
    if result.usage.output_tokens > plan.budget.max_output_tokens:
        raise AssistantRunError("Assistant result exceeds frozen output-token budget")
    if len(result.tool_trace) > plan.budget.max_tool_calls:
        raise AssistantRunError("Assistant result exceeds frozen tool-call budget")
    runtime_by_tool = {
        item.tool_name: item.runtime_binding_sha256 for item in plan.registry.tools
    }
    if any(
        trace.runtime_binding_sha256 != runtime_by_tool[trace.tool_name]
        for trace in result.tool_trace
    ):
        raise AssistantRunError("Assistant result tool runtime lock mismatch")
    routing_present = result.selected_capability is not None
    expected_bank = request.treatment.bank_sha256 if routing_present else None
    if result.bank_sha256 != expected_bank:
        raise AssistantRunError("Assistant result Bank differs from treatment lock")
    if row.execution_receipt is not None:
        receipt = row.execution_receipt
        public = parse_canonical_json(
            request.query.public_input_json.encode("utf-8"),
            label="saved runner-owned public input",
        )
        expected_asset_id = (
            request.query.asset_binding.asset_id
            if request.query.asset_binding is not None
            else public.get("asset_id")
            if isinstance(public, dict)
            else None
        )
        if (
            not isinstance(public, dict)
            or receipt.query_asset_id != expected_asset_id
            or receipt.request_sha256 != request.request_sha256
            or receipt.aggregate_usage != result.usage
            or receipt.runner_latency_ms != result.latency_ms
            or receipt.tool_trace != result.tool_trace
            or (receipt.outcome == "success") != (result.error_code is None)
        ):
            raise AssistantRunError("Assistant runner receipt/result mismatch")
        if receipt.outcome == "success" and result.backbone_request_id not in {
            item.provider_request_id for item in receipt.model_calls
        }:
            raise AssistantRunError(
                "Assistant result request ID is absent from runner receipt"
            )
        # Old, externally locked runner bundles predate route-attempt receipts
        # and remain readable.  New receipts are verified against the saved
        # result so a coordinated rehash cannot relabel a tool failure.
        if receipt.route_attempt is not None:
            attempt = receipt.route_attempt
            result_routing = (
                result.selected_capability,
                result.skill_slug,
                result.route_trace_sha256,
            )
            if request.config == "noskill":
                if attempt.status != "not_applicable" or any(
                    value is not None for value in result_routing
                ):
                    raise AssistantRunError("NoSkill route-attempt receipt is invalid")
            elif attempt.status == "selected":
                if result_routing != (
                    attempt.selected_capability,
                    attempt.skill_slug,
                    attempt.route_trace_sha256,
                ):
                    raise AssistantRunError(
                        "Assistant result route differs from route-attempt receipt"
                    )
            elif attempt.status == "failed":
                if any(value is not None for value in result_routing):
                    raise AssistantRunError(
                        "failed route-attempt receipt retained route identity"
                    )
            else:  # pragma: no cover - Literal is validated above
                raise AssistantRunError("route-attempt receipt status is invalid")


def _verify_registry_lock(
    lock: AssistantRegistryRuntimeLock, registry: ToolRegistry
) -> None:
    current = _registry_lock(registry)
    if current != lock:
        raise AssistantRunError("live tool registry differs from Assistant matrix lock")


def _require_exact_bundle_directory(root: Path) -> None:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise AssistantRunError(
            "Assistant run directory cannot be inspected"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AssistantRunError("Assistant run root must be a real directory")
    if hasattr(root, "is_junction") and root.is_junction():
        raise AssistantRunError("Assistant run root must not be a junction")
    try:
        entries = {item.name for item in root.iterdir()}
    except OSError as error:
        raise AssistantRunError("Assistant run directory cannot be listed") from error
    if entries != set(_BUNDLE_FILES):
        raise AssistantRunError("Assistant run directory file set is invalid")


def _read_bundle(root: Path) -> dict[str, bytes]:
    content: dict[str, bytes] = {}
    for name in _BUNDLE_FILES:
        try:
            content[name] = read_stable_regular_file(
                root / name, label=f"Assistant run {name}"
            )
        except ArtifactFormatError as error:
            raise AssistantRunError(str(error)) from error
    return content


def _load_model(content: bytes, model_type, label: str):
    try:
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise AssistantRunError(str(error)) from error
    if not isinstance(raw, dict):
        raise AssistantRunError(f"{label}: root must be an object")
    try:
        return model_type.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise AssistantRunError(
            f"{label}: schema validation failed: {error}"
        ) from error


def _load_jsonl_models(content: bytes, model_type, label: str) -> tuple:
    try:
        parsed = parse_canonical_jsonl(content, label=label)
    except ArtifactFormatError as error:
        raise AssistantRunError(str(error)) from error
    if not parsed:
        raise AssistantRunError(f"{label}: must contain at least one row")
    models = []
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        try:
            models.append(model_type.model_validate_json(line, strict=True))
        except ValidationError as error:
            raise AssistantRunError(
                f"{label} line {line_number}: schema validation failed: {error}"
            ) from error
    return tuple(models)


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if (
        len(value) != 64
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "ASSISTANT_MATRIX_POLICY_VERSION",
    "ASSISTANT_RUN_POLICY_VERSION",
    "ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION",
    "ASSISTANT_ROUTE_CALL_EVIDENCE_POLICY_VERSION",
    "ASSISTANT_ROUTE_RESPONSE_PREFIX_MAX_BYTES",
    "PHASE4_SELECTION_POLICY_VERSION",
    "FORMAL_INELIGIBLE_REASON",
    "MAIN_CONFIG_ORDER",
    "AssistantBackend",
    "AssistantBackendContractError",
    "AssistantBackendResponse",
    "AssistantRegistryRuntimeLock",
    "AssistantExecutionReceipt",
    "AssistantModelCallReceipt",
    "AssistantMatrixPlan",
    "AssistantQueryDefinition",
    "AssistantQueryInput",
    "AssistantRequestSnapshot",
    "AssistantRouteCallEvidence",
    "AssistantSharedRouteReference",
    "AssistantRunError",
    "AssistantRunManifest",
    "AssistantRunResultRow",
    "AssistantRunSpec",
    "AssistantRouteAttempt",
    "BackboneLock",
    "CreatedAssistantMatrixPlan",
    "CreatedAssistantRunBundle",
    "DiagnosticSpecBaseline",
    "InferenceBudget",
    "JudgeAuditSelectionEntry",
    "MatrixTreatment",
    "Phase4InputSelectionManifest",
    "RegistryRuntimeLockV2",
    "RegistryRuntimeLock",
    "RunnerOwnedAssistantExecution",
    "ToolRuntimeBinding",
    "VerifiedAssistantMatrixPlan",
    "VerifiedAssistantRunBundle",
    "VerifiedFiveConfigRuns",
    "VerifiedPhase4Inputs",
    "build_assistant_matrix_plan",
    "build_assistant_query_input",
    "build_legacy_assistant_query_input",
    "create_assistant_matrix_plan_file",
    "create_assistant_run_bundle",
    "create_runner_owned_assistant_run_bundle",
    "load_verified_assistant_matrix_plan",
    "load_verified_assistant_run_bundle",
    "load_verified_five_config_runs",
    "load_verified_phase4_inputs",
    "make_backbone_lock",
    "make_assistant_route_attempt",
    "make_assistant_route_call_evidence",
    "make_inference_budget",
    "make_phase4_input_selection_manifest",
    "require_verified_assistant_matrix_plan",
    "require_verified_assistant_run_bundle",
    "require_verified_five_config_runs",
    "require_verified_phase4_inputs",
]
