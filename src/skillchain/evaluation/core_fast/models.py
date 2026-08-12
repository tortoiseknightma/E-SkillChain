from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain import config


CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)
CONFIGS = ("noskill", "llm_static", "s1", "s1s2", "full")
EVOLUTION_STAGES = ("s1", "s2", "full")
SPLIT_COUNTS = {
    "dev_mini": 200,
    "opt_pool": 800,
    "val": 200,
    "test_frozen": 300,
}

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CallRole = Literal["feedback", "creator", "assistant", "judge", "route_only"]
CallStatus = Literal[
    "success",
    "provider_error",
    "empty_response",
    "schema_error",
    "interrupted_unknown",
    "budget_exceeded",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ModelRole(FrozenStrictModel):
    provider: str
    requested_model: str
    moving_alias: bool = False
    reasoning_effort: str | None = None
    credential_env: str | None = None
    estimated_call_cost_cny: float = Field(ge=0)

    @field_validator("provider", "requested_model")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model role strings must be non-blank and trimmed")
        return value


class CorePaths(FrozenStrictModel):
    queries: str
    static_bank: str
    opt_static_results: str
    opt_route_attribution: str
    opt_fold_mapping: str
    val_gate_assignments: str
    s1_authoring_input: str
    asset_catalog_dir: str | None = None
    asset_root: str | None = None
    selection_manifest: str | None = None
    dataset_assets: str | None = None
    runtime_catalog_assets: str | None = None
    rpc_scenes: str | None = None
    inaturalist_manifest: str | None = None
    recipe_evidence: str | None = None
    fashioniq_captions: tuple[str, ...] = ()
    style_coordination_graph: str | None = None
    final_rubric: str | None = None


class FixedSample(FrozenStrictModel):
    query_id: str
    capability: str
    role: Literal["failure", "anchor", "smoke", "body_failure", "body_anchor"]

    @model_validator(mode="after")
    def valid_capability(self) -> Self:
        if self.capability not in CAPABILITIES:
            raise ValueError(f"unknown capability: {self.capability}")
        return self


class FixedSamples(FrozenStrictModel):
    canary12: tuple[FixedSample, ...]
    dev_smoke24: tuple[FixedSample, ...]
    body48: tuple[FixedSample, ...]

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        expected = ((self.canary12, 12), (self.dev_smoke24, 24), (self.body48, 48))
        for rows, count in expected:
            ids = tuple(row.query_id for row in rows)
            if len(rows) != count or len(ids) != len(set(ids)):
                raise ValueError(
                    f"fixed sample must contain exactly {count} unique IDs"
                )
        for capability in CAPABILITIES:
            canary = [row.role for row in self.canary12 if row.capability == capability]
            smoke = [row for row in self.dev_smoke24 if row.capability == capability]
            body = [row for row in self.body48 if row.capability == capability]
            if (
                len(canary) != 2
                or "failure" not in canary
                or any(role not in {"failure", "anchor"} for role in canary)
            ):
                raise ValueError(
                    f"canary12 role composition is invalid for {capability}"
                )
            if len(smoke) != 4 or any(row.role != "smoke" for row in smoke):
                raise ValueError(f"dev_smoke24 must contain four rows for {capability}")
            body_roles = sorted(row.role for row in body)
            if (
                len(body_roles) != 8
                or "body_failure" not in body_roles
                or any(
                    role not in {"body_failure", "body_anchor"} for role in body_roles
                )
            ):
                raise ValueError(f"body48 role composition is invalid for {capability}")
        return self


class GateRules(FrozenStrictModel):
    s1_replay_macro_delta_pp_min: Literal[0.0] = 0.0
    s1_system_macro_delta_pp_min: Literal[2.0] = 2.0
    s1_bootstrap_ci95_lower_pp_min: Literal[0.0] = 0.0
    s1_hard_error_delta_pp_max: Literal[1.0] = 1.0
    s1_max_capability_drop_pp: Literal[3.0] = 3.0
    s2_broken_penalty: Literal[2.5] = 2.5
    s3_judge_subset_per_affected_capability: Literal[4] = 4
    s3_judge_subset_max: Literal[24] = 24


class S1Settings(FrozenStrictModel):
    round_id: str = Field(default="r1", pattern=r"^r[1-9][0-9]*$")
    feedback_mode: Literal["fresh-per-round"] = "fresh-per-round"
    feedback_total_count: int = Field(default=48, ge=1, le=60)
    feedback_canary_count: int = Field(default=6, ge=1, le=60)
    feedback_selection_policy: Literal["discovery-stratified-v1"] = (
        "discovery-stratified-v1"
    )
    feedback_allocation: Literal["target-focused", "balanced-six-capability"] = (
        "balanced-six-capability"
    )
    target_capabilities: tuple[str, ...] = ()
    proposal_mode: Literal["sparse-parent-patch-v1"] = "sparse-parent-patch-v1"
    max_patched_capabilities: int = Field(default=3, ge=1, le=3)
    protected_capabilities: tuple[str, ...] = ("knowledge.visual_encyclopedia",)
    creator_directives: tuple[str, ...] = ()
    required_patch_phrases: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_protected_capabilities(self) -> Self:
        protected = self.protected_capabilities
        if protected != tuple(sorted(set(protected))):
            raise ValueError("S1 protected capabilities must be sorted and unique")
        if not set(protected) <= set(CAPABILITIES):
            raise ValueError("S1 protected capabilities contain an unknown capability")
        if "knowledge.visual_encyclopedia" not in protected:
            raise ValueError("Core Fast S1 must protect Encyclopedia after R0")
        if len(protected) == len(CAPABILITIES):
            raise ValueError("S1 must leave at least one capability patchable")
        if self.creator_directives != tuple(
            dict.fromkeys(self.creator_directives)
        ) or any(not item or item != item.strip() for item in self.creator_directives):
            raise ValueError("S1 Creator directives must be trimmed and unique")
        for capability, phrases in self.required_patch_phrases.items():
            if capability not in CAPABILITIES or capability in protected:
                raise ValueError(
                    "required S1 patch phrases must target an unprotected capability"
                )
            if phrases != tuple(sorted(set(phrases))) or any(
                not phrase or phrase != phrase.strip() for phrase in phrases
            ):
                raise ValueError(
                    "required S1 patch phrases must be trimmed, sorted, and unique"
                )
        if self.feedback_canary_count > self.feedback_total_count:
            raise ValueError("Feedback canary count cannot exceed total count")
        targets = self.target_capabilities
        if not targets:
            targets = tuple(sorted(set(CAPABILITIES) - set(protected)))
            object.__setattr__(self, "target_capabilities", targets)
        if targets != tuple(sorted(set(targets))) or not targets:
            raise ValueError(
                "S1 target capabilities must be sorted, unique, and nonempty"
            )
        if not set(targets) <= set(CAPABILITIES) - set(protected):
            raise ValueError("S1 target capabilities must be patchable capabilities")
        if self.feedback_allocation == "target-focused" and (
            len(targets) != 1 or self.max_patched_capabilities != 1
        ):
            raise ValueError(
                "target-focused Feedback requires one target and one patched capability"
            )
        if self.max_patched_capabilities > len(targets):
            raise ValueError(
                "S1 cannot patch more capabilities than its frozen targets"
            )
        return self


class Concurrency(FrozenStrictModel):
    assistant: int = Field(default=config.ASSISTANT_VALIDATED_CONCURRENCY, ge=1)
    assistant_requests_per_second: float = Field(
        default=config.ASSISTANT_REQUESTS_PER_SECOND,
        gt=0,
    )
    # This is a provider-capacity ceiling, not the number of calls that every
    # stage must create.  S1 currently has a fixed 12-row Feedback batch, so
    # its effective concurrency is min(feedback, 12).
    feedback: int = Field(default=config.FEEDBACK_JUDGE_VALIDATED_CONCURRENCY, ge=1)
    feedback_requests_per_second: float = Field(
        default=config.FEEDBACK_JUDGE_REQUESTS_PER_SECOND,
        gt=0,
    )
    final_judge: Literal[4] = 4

    @model_validator(mode="after")
    def validate_measured_capacity_profiles(self) -> Self:
        assistant_expected = (
            config.ASSISTANT_VALIDATED_CONCURRENCY,
            config.ASSISTANT_REQUESTS_PER_SECOND,
        )
        if (self.assistant, self.assistant_requests_per_second) != assistant_expected:
            raise ValueError(
                "Core Fast Assistant capacity must match the measured profile "
                f"{assistant_expected[0]} inflight/"
                f"{assistant_expected[1]:g} requests per second"
            )
        feedback_expected = (
            config.FEEDBACK_JUDGE_VALIDATED_CONCURRENCY,
            config.FEEDBACK_JUDGE_REQUESTS_PER_SECOND,
        )
        if (self.feedback, self.feedback_requests_per_second) != feedback_expected:
            raise ValueError(
                "Core Fast Qwen3.8 Feedback capacity must match the measured "
                f"profile {feedback_expected[0]} inflight/"
                f"{feedback_expected[1]:g} requests per second"
            )
        return self


class Limits(FrozenStrictModel):
    max_feedback_calls: int = Field(default=48, ge=1, le=60)
    max_creator_calls: Literal[3] = 3
    external_cost_cny: Literal[250.0] = 250.0
    final_judge_format_retries: Literal[1] = 1


class CommandSet(FrozenStrictModel):
    feedback: tuple[str, ...] = ()
    creator: tuple[str, ...] = ()
    assistant: tuple[str, ...] = ()
    judge: tuple[str, ...] = ()
    route_only: tuple[str, ...] = ()

    def for_role(self, role: CallRole) -> tuple[str, ...]:
        return getattr(self, role)


class RuntimeSettings(FrozenStrictModel):
    adapter: Literal["command", "python"] = "command"
    python_factory: str | None = None
    commands: CommandSet = CommandSet()
    command_timeout_seconds: int = Field(default=1800, ge=1)

    @model_validator(mode="after")
    def validate_adapter(self) -> Self:
        if self.adapter == "python" and not self.python_factory:
            raise ValueError("python adapter requires python_factory=module:function")
        return self


class CoreFastSpec(FrozenStrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["core-experiment-fast-v1"] = "core-experiment-fast-v1"
    experiment_id: str
    paths: CorePaths
    asset_catalog_sha256: Sha256 | None = None
    static_bank_file_sha256: Sha256 | None = None
    opt_static_results_sha256: Sha256
    opt_fold_mapping_sha256: Sha256
    val_gate_assignments_sha256: Sha256
    s1_authoring_input_file_sha256: Sha256
    split_counts: dict[str, int]
    capabilities: tuple[str, ...]
    configs: tuple[str, ...]
    fixed_samples: FixedSamples
    models: dict[CallRole, ModelRole]
    gates: GateRules = GateRules()
    s1_settings: S1Settings = S1Settings()
    concurrency: Concurrency = Concurrency()
    limits: Limits = Limits()
    runtime: RuntimeSettings = RuntimeSettings()
    bootstrap_replicates: Literal[10_000] = 10_000
    bootstrap_seed: Literal[2026080601] = 2026080601
    disclosures: tuple[str, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.split_counts != SPLIT_COUNTS:
            raise ValueError("Core Fast Path split counts must be 200/800/200/300")
        if self.capabilities != CAPABILITIES:
            raise ValueError("Core Fast Path capability order is fixed")
        if self.configs != CONFIGS:
            raise ValueError("Core Fast Path five-config order is fixed")
        if set(self.models) != {
            "feedback",
            "creator",
            "assistant",
            "judge",
            "route_only",
        }:
            raise ValueError("all five model/execution roles must be configured")
        if self.limits.max_feedback_calls != self.s1_settings.feedback_total_count:
            raise ValueError(
                "Feedback call limit must equal the pre-frozen S1 Feedback count"
            )
        feedback = self.models["feedback"]
        if (
            feedback.requested_model.casefold() != "qwen3.8-max"
            or not feedback.moving_alias
        ):
            raise ValueError("Feedback must disclose Qwen3.8-Max as a moving alias")
        creator = self.models["creator"]
        if (creator.requested_model, creator.reasoning_effort) != (
            "gpt-5.6-sol",
            "high",
        ):
            raise ValueError("Creator/optimizers must use gpt-5.6-sol/high")
        return self

    def resolved_path(self, field_name: str, *, base_dir: Path) -> Path:
        value = getattr(self.paths, field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"path {field_name} is not configured")
        expanded = os.path.expandvars(value)
        path = Path(expanded)
        return path if path.is_absolute() else (base_dir / path).resolve()


class CallIntent(FrozenStrictModel):
    schema_version: Literal[1] = 1
    call_id: str
    role: CallRole
    purpose: str
    requested_model: str
    payload: dict[str, object]


class CallResult(FrozenStrictModel):
    schema_version: Literal[1] = 1
    call_id: str
    role: CallRole
    status: CallStatus
    requested_model: str
    returned_model: str | None = None
    raw_output: str | None = None
    schema_valid: bool = False
    output: dict[str, object] | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_cny: float = Field(default=0.0, ge=0)
    cost_basis: Literal[
        "provider_reported",
        "token_pricing",
        "planning_estimate",
        "unavailable",
    ] = "unavailable"
    latency_ms: int = Field(default=0, ge=0)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_cost_basis(self) -> Self:
        if self.cost_cny > 0 and self.cost_basis == "unavailable":
            raise ValueError("positive cost requires an explicit cost basis")
        return self


class ToolTraceItem(FrozenStrictModel):
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    status: Literal["success", "error"]
    result_sha256: str | None = None
    error_code: str | None = None


class AssistantObservation(FrozenStrictModel):
    schema_version: Literal[1] = 1
    query_id: str
    response_text: str
    selected_capability: str | None
    route_trace_key: str | None
    tool_trace_key: str
    tool_trace: tuple[ToolTraceItem, ...] = ()
    replay_context: dict[str, object] = Field(default_factory=dict)
    answer_mode: Literal["supported", "fallback", "unresolved"]
    oracle_available: bool = True
    gcs_components: dict[str, bool]
    gcs_reason_codes: tuple[str, ...] = ()
    gcs_score: float = Field(ge=0.0, le=1.0)
    hard_error: bool
    card_violation: bool = False
    evidence_violation: bool = False
    tool_violation: bool = False
    source: str | None = None
    repair: str | None = None
    style_submode: str | None = None

    @field_validator("tool_trace", "gcs_reason_codes", mode="before")
    @classmethod
    def coerce_tool_trace(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_gcs(self) -> Self:
        expected = {
            "route_acceptable",
            "no_hard_error",
            "tool_contract_pass",
            "evidence_grounded",
            "output_contract_pass",
        }
        if set(self.gcs_components) != expected:
            raise ValueError("GCS must contain the five fixed components")
        # GCS is the existing conjunctive project metric: a query succeeds
        # only when all five components pass.  The component mean is useful
        # diagnostically but must never be substituted for GCS at a Gate.
        calculated = float(all(self.gcs_components.values()))
        if abs(calculated - self.gcs_score) > 1e-9:
            raise ValueError("gcs_score must equal the five-component conjunction")
        if self.gcs_reason_codes != tuple(sorted(set(self.gcs_reason_codes))):
            raise ValueError("GCS reason codes must be sorted and unique")
        return self


class JudgeObservation(FrozenStrictModel):
    schema_version: Literal[1] = 1
    query_id: str
    j_project: float = Field(ge=0.0, le=100.0)
    dimensions: dict[str, float]


class StageDecision(FrozenStrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["s1", "s2", "full"]
    accepted: bool
    alias_of: Literal["llm_static", "s1", "s1s2"] | None
    parent_bank: str
    candidate_bank: str | None
    selected_bank: str
    reasons: tuple[str, ...]
    metrics: dict[str, object]

    @field_validator("reasons", mode="before")
    @classmethod
    def coerce_reasons(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def load_core_fast_spec(path: str | Path) -> CoreFastSpec:
    source = Path(path)
    try:
        return CoreFastSpec.model_validate_json(source.read_bytes(), strict=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load Core Fast Path spec: {source}") from error


__all__ = [
    "AssistantObservation",
    "CAPABILITIES",
    "CONFIGS",
    "CallIntent",
    "CallResult",
    "CallRole",
    "CoreFastSpec",
    "EVOLUTION_STAGES",
    "JudgeObservation",
    "SPLIT_COUNTS",
    "StageDecision",
    "load_core_fast_spec",
]
