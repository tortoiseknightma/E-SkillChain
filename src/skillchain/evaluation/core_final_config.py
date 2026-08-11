"""Zero-call configuration framework for the final Portfolio core evaluation.

The framework separates the scientific contract that must not drift after the
``dev_mini`` diagnostic from operational decisions that are intentionally left
open until that diagnostic is complete.  It never authorizes or starts a model
run.  A compiled artifact can become ``ready_to_freeze`` only after every
required input is content-addressed and every refinement slot has a recorded
decision artifact.
"""

from __future__ import annotations

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

from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

POLICY_VERSION = "portfolio-core-final-evaluation-framework-v1"
MAIN_CONFIG_ORDER = ("noskill", "llm_static", "s1", "s1s2", "full")
CAPABILITY_ORDER = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)

FrameworkStatus = Literal[
    "draft_waiting_for_dev_mini_diagnostics",
    "draft_pending_freeze_inputs",
    "ready_to_freeze",
]
BindingStatus = Literal["pending", "verified"]
ResolutionStatus = Literal["pending", "resolved"]


class CoreFinalFrameworkError(ValueError):
    """The core-final framework or one of its evidence bindings is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _self_hash(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude={"framework_sha256"})
    return sha256_bytes(canonical_json_bytes(payload))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _portable_relative_path(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError(f"{label} must be a normalized repository-relative path")
    return candidate.as_posix()


class CoreEvaluationScope(_StrictFrozenModel):
    profile: Literal["core"] = "core"
    query_count: Literal[1500] = 1500
    config_count: Literal[5] = 5
    canonical_instance_count: Literal[7500] = 7500
    batch_size: Literal[25] = 25
    batch_count: Literal[60] = 60
    config_shard_count: Literal[300] = 300
    capabilities: tuple[str, ...] = CAPABILITY_ORDER
    config_order: tuple[str, ...] = MAIN_CONFIG_ORDER

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.capabilities != CAPABILITY_ORDER:
            raise ValueError("the six-capability core scope is fixed")
        if self.config_order != MAIN_CONFIG_ORDER:
            raise ValueError("the canonical five-config order is fixed")
        if self.canonical_instance_count != self.query_count * self.config_count:
            raise ValueError(
                "canonical instance count must equal query_count * config_count"
            )
        if self.batch_count * self.batch_size != self.query_count:
            raise ValueError("batch geometry must cover each core query exactly once")
        if self.config_shard_count != self.batch_count * self.config_count:
            raise ValueError("config shard count must equal batch_count * config_count")
        return self


class FixedScientificContract(_StrictFrozenModel):
    research_target: Literal["paper-mechanism-oriented-final-core-evaluation"] = (
        "paper-mechanism-oriented-final-core-evaluation"
    )
    track: Literal["portfolio"] = "portfolio"
    same_query_across_all_configs: Literal[True] = True
    paired_input_order: Literal[True] = True
    fixed_full_denominator: Literal[True] = True
    identical_user_input_and_tool_surface: Literal[True] = True
    identical_assistant_action_budget: Literal[True] = True
    shared_route_artifact_for_s1s2_and_full: Literal[True] = True
    one_canonical_run_per_config: Literal[True] = True
    one_sealed_test_schedule: Literal[True] = True
    no_intermediate_score_exposure: Literal[True] = True
    no_result_dependent_retry_or_selection: Literal[True] = True
    challenge_final_reported_separately: Literal[True] = True
    not_a_production_traffic_claim: Literal[True] = True
    frozen_primary_comparisons: tuple[str, ...] = (
        "s1_vs_llm_static",
        "s1s2_vs_s1",
        "full_vs_s1s2",
        "full_vs_llm_static",
    )
    fields_forbidden_to_refine: tuple[str, ...] = (
        "scope.capabilities",
        "scope.config_order",
        "fixed_contract.frozen_primary_comparisons",
        "metrics.primary_quality_metric",
        "metrics.primary_routing_metric",
        "metrics.group_resampling_unit",
        "fixed_contract.fixed_full_denominator",
        "fixed_contract.one_sealed_test_schedule",
    )

    @model_validator(mode="after")
    def validate_policy_lists(self) -> Self:
        expected_comparisons = (
            "s1_vs_llm_static",
            "s1s2_vs_s1",
            "full_vs_s1s2",
            "full_vs_llm_static",
        )
        expected_forbidden = (
            "scope.capabilities",
            "scope.config_order",
            "fixed_contract.frozen_primary_comparisons",
            "metrics.primary_quality_metric",
            "metrics.primary_routing_metric",
            "metrics.group_resampling_unit",
            "fixed_contract.fixed_full_denominator",
            "fixed_contract.one_sealed_test_schedule",
        )
        if self.frozen_primary_comparisons != expected_comparisons:
            raise ValueError("primary comparisons are fixed by policy")
        if self.fields_forbidden_to_refine != expected_forbidden:
            raise ValueError("forbidden refinement fields are fixed by policy")
        return self


class CoreMetricContract(_StrictFrozenModel):
    primary_quality_metric: Literal["j_project_0_100"] = "j_project_0_100"
    quality_dimensions: tuple[str, ...] = (
        "task_completion_0_10",
        "commercial_card_compliance_0_10_when_required",
        "conversation_quality_0_20",
        "capability_alignment_0_10",
    )
    quality_formula: Literal[
        "100*(TCR+CCC*requires_card+CQ+CA)/(40+10*requires_card)"
    ] = "100*(TCR+CCC*requires_card+CQ+CA)/(40+10*requires_card)"
    paper_table2_avg_name_prohibited: Literal[True] = True
    primary_routing_metric: Literal["canonical_capability_macro_f1"] = (
        "canonical_capability_macro_f1"
    )
    noskill_routing_value: Literal["not_applicable"] = "not_applicable"
    confidence_interval: Literal["paired_cluster_bootstrap_95pct"] = (
        "paired_cluster_bootstrap_95pct"
    )
    bootstrap_replicates: Literal[10000] = 10000
    group_resampling_unit: Literal["leakage_connected_component"] = (
        "leakage_connected_component"
    )
    routing_significance_test: Literal["paired_mcnemar"] = "paired_mcnemar"
    report_per_capability_and_confusion_matrix: Literal[True] = True

    @model_validator(mode="after")
    def validate_quality_dimensions(self) -> Self:
        if self.quality_dimensions != (
            "task_completion_0_10",
            "commercial_card_compliance_0_10_when_required",
            "conversation_quality_0_20",
            "capability_alignment_0_10",
        ):
            raise ValueError("quality dimensions are fixed by policy")
        return self


class CoreEligibilityContract(_StrictFrozenModel):
    exact_match_required_for_primary_capability_claims: Literal[True] = True
    minimum_opt_per_capability: Literal[60] = 60
    minimum_val_per_capability: Literal[30] = 30
    minimum_test_per_capability: Literal[30] = 30
    val_route_gate_count: Literal[80] = 80
    val_body_gate_count: Literal[80] = 80
    val_shadow_count: Literal[40] = 40
    ineligible_capabilities_reported_as_exploratory: Literal[True] = True


class CoreToolEvaluationContract(_StrictFrozenModel):
    canonical_tool_count: Literal[7] = 7
    gold_cases_per_tool_range: tuple[Literal[30], Literal[50]] = (30, 50)
    all_canonical_tools_covered: Literal[True] = True
    ranking_detection_and_ocr_in_tool_test_frozen: Literal[True] = True
    multi_product_end_to_end_gold_required: Literal[True] = True
    fake_fixture_cannot_pass_core_gate: Literal[True] = True


class EvaluatorReliabilityContract(_StrictFrozenModel):
    judge_human_audit_sample_range: tuple[Literal[50], Literal[100]] = (50, 100)
    grouped_samples_separate_from_scored_core: Literal[True] = True
    blinded_human_scores: Literal[True] = True
    agreement_and_failure_modes_reported: Literal[True] = True
    thresholds_frozen_before_core_unseal: Literal[True] = True


class HumanEvaluationContract(_StrictFrozenModel):
    method: Literal["blind_side_by_side"] = "blind_side_by_side"
    core_sample_size_range: tuple[Literal[100], Literal[150]] = (100, 150)
    provisional_sample_size: Literal[125] = 125
    contrasts: tuple[str, ...] = (
        "full_vs_llm_static",
        "full_vs_s1s2",
    )
    randomized_left_right: Literal[True] = True
    sampling_seed_frozen_before_unseal: Literal[True] = True
    online_ab_test_in_scope: Literal[False] = False

    @model_validator(mode="after")
    def validate_contrasts(self) -> Self:
        if self.contrasts != ("full_vs_llm_static", "full_vs_s1s2"):
            raise ValueError("human-evaluation contrasts are fixed by policy")
        return self


class ProvisionalOperationalBaseline(_StrictFrozenModel):
    """Starting point only; decision artifacts may replace every value here."""

    disposition: Literal["provisional_not_authorized"] = "provisional_not_authorized"
    assistant_concurrency: Literal[2] = 2
    final_judge_concurrency: Literal[8] = 8
    assistant_requests_per_minute: Literal[40] = 40
    assistant_rate_limit_policy: Literal["smooth_start_v1"] = "smooth_start_v1"
    assistant_minimum_start_interval_seconds: Literal[1.5] = 1.5
    max_active_waves: Literal[10] = 10
    canonical_repetitions: Literal[1] = 1
    optional_repetition_candidate_configs: tuple[str, ...] = (
        "noskill",
        "s1s2",
        "full",
    )
    optional_repetition_count_if_adopted: Literal[2] = 2
    phase_budget_candidate_cny: Literal[1100] = 1100
    feedback_model_calls_per_canonical_instance: Literal[0] = 0

    @model_validator(mode="after")
    def validate_optional_repetition_subset(self) -> Self:
        if self.optional_repetition_candidate_configs != (
            "noskill",
            "s1s2",
            "full",
        ):
            raise ValueError("optional repetition candidate configs are fixed")
        return self


class ArtifactBinding(_StrictFrozenModel):
    role: str
    category: Literal["dataset", "runtime", "evidence"]
    required_for_freeze: bool
    description: str
    status: BindingStatus = "pending"
    artifact_root_id: str | None = None
    path: str | None = None
    file_sha256: Sha256 | None = None
    byte_count: int | None = Field(default=None, ge=0)

    @field_validator("role", "description")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("binding text fields must be non-blank and trimmed")
        return value

    @field_validator("artifact_root_id")
    @classmethod
    def validate_artifact_root_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip() or not value.replace("-", "").isalnum():
            raise ValueError("artifact_root_id must be a trimmed alphanumeric slug")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return None if value is None else _portable_relative_path(value, "binding path")

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        details = (
            self.artifact_root_id,
            self.path,
            self.file_sha256,
            self.byte_count,
        )
        if self.status == "pending" and any(item is not None for item in details):
            raise ValueError("pending binding cannot contain verified file details")
        if self.status == "verified" and any(item is None for item in details):
            raise ValueError("verified binding requires path, hash, and byte count")
        return self


class RefinementSlot(_StrictFrozenModel):
    slot_id: str
    purpose: str
    allowed_fields: tuple[str, ...]
    required_evidence_roles: tuple[str, ...]
    status: ResolutionStatus = "pending"
    decision_artifact_root_id: str | None = None
    decision_artifact_path: str | None = None
    decision_artifact_sha256: Sha256 | None = None

    @field_validator("slot_id", "purpose")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("refinement text fields must be non-blank and trimmed")
        return value

    @field_validator("decision_artifact_path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _portable_relative_path(value, "decision artifact path")
        )

    @field_validator("decision_artifact_root_id")
    @classmethod
    def validate_root_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip() or not value.replace("-", "").isalnum():
            raise ValueError(
                "decision_artifact_root_id must be a trimmed alphanumeric slug"
            )
        return value

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        details = (
            self.decision_artifact_root_id,
            self.decision_artifact_path,
            self.decision_artifact_sha256,
        )
        if self.status == "pending" and any(item is not None for item in details):
            raise ValueError("pending refinement cannot contain a decision artifact")
        if self.status == "resolved" and any(item is None for item in details):
            raise ValueError(
                "resolved refinement requires a decision artifact and hash"
            )
        if not self.allowed_fields:
            raise ValueError("refinement slot must list at least one allowed field")
        return self


class CoreFinalEvaluationFramework(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["core-final-evaluation-framework"] = "core-final-evaluation-framework"
    policy_version: Literal[POLICY_VERSION] = POLICY_VERSION
    status: FrameworkStatus
    execution_authorized: Literal[False] = False
    model_calls_performed: Literal[0] = 0
    scope: CoreEvaluationScope
    fixed_contract: FixedScientificContract
    metrics: CoreMetricContract
    eligibility: CoreEligibilityContract
    tool_evaluation: CoreToolEvaluationContract
    evaluator_reliability: EvaluatorReliabilityContract
    human_evaluation: HumanEvaluationContract
    provisional_operational_baseline: ProvisionalOperationalBaseline
    artifact_bindings: tuple[ArtifactBinding, ...]
    refinement_slots: tuple[RefinementSlot, ...]
    blockers: tuple[str, ...]
    framework_sha256: Sha256

    @model_validator(mode="after")
    def validate_framework(self) -> Self:
        binding_policy = tuple(
            (
                item.role,
                item.category,
                item.required_for_freeze,
                item.description,
            )
            for item in self.artifact_bindings
        )
        if binding_policy != _BINDING_DEFINITIONS:
            raise ValueError("artifact binding definitions or order differ from policy")
        binding_roles = tuple(item.role for item in self.artifact_bindings)
        slot_policy = tuple(
            (
                item.slot_id,
                item.purpose,
                item.allowed_fields,
                item.required_evidence_roles,
            )
            for item in self.refinement_slots
        )
        if slot_policy != _REFINEMENT_DEFINITIONS:
            raise ValueError("refinement definitions or order differ from policy")
        evidence_roles = set(binding_roles)
        for slot in self.refinement_slots:
            missing = set(slot.required_evidence_roles) - evidence_roles
            if missing:
                raise ValueError(
                    f"refinement slot {slot.slot_id} references unknown evidence roles"
                )
        expected_blockers = derive_blockers(
            self.artifact_bindings, self.refinement_slots
        )
        if self.blockers != expected_blockers:
            raise ValueError("framework blocker list is stale or forged")
        expected_status = derive_status(expected_blockers)
        if self.status != expected_status:
            raise ValueError("framework status does not match its blockers")
        if self.framework_sha256 != _self_hash(self):
            raise ValueError("framework self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


_BINDING_DEFINITIONS: tuple[tuple[str, str, bool, str], ...] = (
    ("core_query_plan", "dataset", True, "Frozen 1,500-query core plan."),
    (
        "core_query_plan_manifest",
        "dataset",
        True,
        "Core plan count and digest manifest.",
    ),
    ("core_queries", "dataset", True, "Materialized core query JSONL."),
    (
        "core_corpus_owner_audit_approval",
        "evidence",
        True,
        "Terminal owner approval for the immutable 200-query core corpus audit.",
    ),
    ("core_capability_assignments", "dataset", True, "Frozen capability assignments."),
    ("core_asset_catalog_manifest", "dataset", True, "Core asset catalog manifest."),
    ("core_asset_catalog_assets", "dataset", True, "Core asset rows."),
    ("core_asset_catalog_components", "dataset", True, "Core leakage components."),
    (
        "core_remote_processing_authorization",
        "dataset",
        True,
        "Owner upload authority.",
    ),
    (
        "core_remote_processing_receipt",
        "dataset",
        True,
        "Verified remote-processing preflight.",
    ),
    (
        "core_leakage_and_exclusion_lock",
        "dataset",
        True,
        "Frozen exclusion and leakage lock.",
    ),
    ("test_frozen_seal", "dataset", True, "Single-use frozen-test seal."),
    (
        "challenge_final_seal",
        "dataset",
        False,
        "Optional separately reported challenge seal.",
    ),
    (
        "final_runtime_lock",
        "runtime",
        True,
        "Final Assistant, Feedback, and Judge runtime identities.",
    ),
    (
        "treatment_chain_manifest",
        "runtime",
        True,
        "Accepted S1/S2/S3 treatment lineage.",
    ),
    ("bank_llm_static", "runtime", True, "Frozen LLMStaticSkill Bank."),
    ("bank_s1", "runtime", True, "Frozen accepted S1 Bank."),
    ("bank_s1s2", "runtime", True, "Frozen accepted S1+S2 Bank."),
    ("bank_full", "runtime", True, "Frozen accepted Full Bank."),
    (
        "judge_rubric",
        "runtime",
        True,
        "Frozen project-quality rubric and parser contract.",
    ),
    ("model_role_selection", "runtime", True, "Frozen role-to-model disclosure."),
    (
        "tool_registry_runtime_lock",
        "runtime",
        True,
        "Frozen tool surface and implementation identity.",
    ),
    (
        "environment_lock",
        "runtime",
        True,
        "Code and environment identity used for the run.",
    ),
    (
        "dev_mini_200x5_completion",
        "evidence",
        True,
        "Complete 200x5 diagnostic receipt.",
    ),
    (
        "dev_mini_diagnostic_decision",
        "evidence",
        True,
        "Recorded diagnosis and adopted patch set.",
    ),
    ("judge_human_audit", "evidence", True, "Separate Judge-human calibration audit."),
    ("core_tool_gold_gate", "evidence", True, "Core tool gold-set gate result."),
    (
        "core_budget_authority",
        "evidence",
        True,
        "Owner-approved core phase budget cap.",
    ),
    (
        "core_schedule_stress",
        "evidence",
        True,
        "Zero-provider 7,500-instance schedule stress receipt.",
    ),
    (
        "core_parallel_smoke",
        "evidence",
        True,
        "Small real-provider concurrency smoke receipt.",
    ),
    ("human_sbs_plan", "evidence", True, "Frozen blind side-by-side sampling plan."),
    (
        "owner_freeze_approval",
        "evidence",
        True,
        "Explicit approval to freeze, not to execute.",
    ),
)

_REFINEMENT_DEFINITIONS: tuple[
    tuple[str, str, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "diagnostic_runtime_patch_set",
        "Adopt only symmetric runtime or parser corrections justified by the mini diagnosis.",
        (
            "runtime.assistant_output_limits",
            "runtime.assistant_action_budget",
            "runtime.final_judge_contract",
            "runtime.final_judge_retry_policy",
            "runtime.evidence_projection",
        ),
        (
            "dev_mini_200x5_completion",
            "dev_mini_diagnostic_decision",
            "judge_human_audit",
        ),
    ),
    (
        "parallel_execution_profile",
        "Choose concurrency, rate limits, wave limits, and recovery policy.",
        (
            "execution.assistant_concurrency",
            "execution.judge_concurrency",
            "execution.rate_limits",
            "execution.max_active_waves",
            "execution.circuit_breakers",
        ),
        ("dev_mini_200x5_completion", "core_schedule_stress", "core_parallel_smoke"),
    ),
    (
        "phase_budget_cap",
        "Set the core monetary and retry ceiling before execution authority is requested.",
        ("execution.phase_budget_cny", "execution.retry_budget"),
        ("dev_mini_200x5_completion", "core_budget_authority"),
    ),
    (
        "judge_and_tool_gate_thresholds",
        "Freeze Judge-human and tool correctness thresholds from independent audits.",
        ("gates.judge_human", "gates.tool_gold", "gates.parser_failure"),
        ("judge_human_audit", "core_tool_gold_gate"),
    ),
    (
        "optional_repetitions",
        "Decide before unsealing whether the preregistered subset receives two repetitions.",
        ("analysis.optional_repetitions", "analysis.repeated_configs"),
        ("dev_mini_200x5_completion", "core_budget_authority"),
    ),
    (
        "analysis_seeds_and_min_n",
        "Freeze bootstrap, sampling, and capability eligibility decisions before scores exist.",
        (
            "analysis.bootstrap_seed",
            "analysis.human_sample_seed",
            "analysis.capability_eligibility",
            "analysis.minimum_reportable_n",
        ),
        ("core_query_plan_manifest", "human_sbs_plan"),
    ),
)


def derive_blockers(
    bindings: tuple[ArtifactBinding | dict[str, object], ...],
    slots: tuple[RefinementSlot | dict[str, object], ...],
) -> tuple[str, ...]:
    def field(item: BaseModel | dict[str, object], name: str) -> object:
        return getattr(item, name) if isinstance(item, BaseModel) else item[name]

    blockers = [
        f"binding:{field(item, 'role')}"
        for item in bindings
        if field(item, "required_for_freeze") is True
        and field(item, "status") != "verified"
    ]
    blockers.extend(
        f"refinement:{field(item, 'slot_id')}"
        for item in slots
        if field(item, "status") != "resolved"
    )
    return tuple(blockers)


def derive_status(blockers: tuple[str, ...]) -> FrameworkStatus:
    if "binding:dev_mini_200x5_completion" in blockers:
        return "draft_waiting_for_dev_mini_diagnostics"
    return "draft_pending_freeze_inputs" if blockers else "ready_to_freeze"


def _finalize_framework(payload: dict[str, object]) -> CoreFinalEvaluationFramework:
    payload["blockers"] = derive_blockers(
        tuple(payload["artifact_bindings"]),  # type: ignore[arg-type]
        tuple(payload["refinement_slots"]),  # type: ignore[arg-type]
    )
    payload["status"] = derive_status(payload["blockers"])  # type: ignore[arg-type]
    json_payload = _jsonable(payload)
    assert isinstance(json_payload, dict)
    json_payload["framework_sha256"] = sha256_bytes(canonical_json_bytes(json_payload))
    return CoreFinalEvaluationFramework.model_validate_json(
        canonical_json_bytes(json_payload)
    )


def build_core_final_evaluation_framework() -> CoreFinalEvaluationFramework:
    """Build the default all-pending, zero-call core-final framework."""

    bindings = tuple(
        ArtifactBinding(
            role=role,
            category=category,  # type: ignore[arg-type]
            required_for_freeze=required,
            description=description,
        )
        for role, category, required, description in _BINDING_DEFINITIONS
    )
    slots = tuple(
        RefinementSlot(
            slot_id=slot_id,
            purpose=purpose,
            allowed_fields=allowed_fields,
            required_evidence_roles=evidence_roles,
        )
        for slot_id, purpose, allowed_fields, evidence_roles in _REFINEMENT_DEFINITIONS
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "core-final-evaluation-framework",
        "policy_version": POLICY_VERSION,
        "execution_authorized": False,
        "model_calls_performed": 0,
        "scope": CoreEvaluationScope(),
        "fixed_contract": FixedScientificContract(),
        "metrics": CoreMetricContract(),
        "eligibility": CoreEligibilityContract(),
        "tool_evaluation": CoreToolEvaluationContract(),
        "evaluator_reliability": EvaluatorReliabilityContract(),
        "human_evaluation": HumanEvaluationContract(),
        "provisional_operational_baseline": ProvisionalOperationalBaseline(),
        "artifact_bindings": bindings,
        "refinement_slots": slots,
    }
    return _finalize_framework(payload)


def _repo_relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CoreFinalFrameworkError(
            f"bound artifact must be a regular file below repository root: {path}"
        ) from exc
    return _portable_relative_path(relative.as_posix(), "bound artifact path")


def bind_framework_artifact(
    framework: CoreFinalEvaluationFramework,
    *,
    role: str,
    path: str | Path,
    expected_sha256: str,
    repository_root: str | Path,
    artifact_root_id: str = "repository",
) -> CoreFinalEvaluationFramework:
    """Return a new framework with one stable, hash-checked artifact binding."""

    matches = [item for item in framework.artifact_bindings if item.role == role]
    if not matches:
        raise CoreFinalFrameworkError(f"unknown artifact role: {role}")
    if matches[0].status == "verified":
        raise CoreFinalFrameworkError(f"artifact role is already verified: {role}")
    content = read_stable_regular_file(path, label=f"core binding {role}")
    actual = sha256_bytes(content)
    if actual != expected_sha256:
        raise CoreFinalFrameworkError(
            f"artifact digest mismatch for {role}: expected {expected_sha256}, got {actual}"
        )
    relative = _repo_relative_path(Path(path), Path(repository_root))
    replacement = matches[0].model_copy(
        update={
            "status": "verified",
            "artifact_root_id": artifact_root_id,
            "path": relative,
            "file_sha256": actual,
            "byte_count": len(content),
        }
    )
    payload = framework.model_dump(mode="python", exclude={"framework_sha256"})
    payload["artifact_bindings"] = tuple(
        replacement if item.role == role else item
        for item in framework.artifact_bindings
    )
    return _finalize_framework(payload)


def resolve_framework_slot(
    framework: CoreFinalEvaluationFramework,
    *,
    slot_id: str,
    decision_artifact_path: str | Path,
    expected_sha256: str,
    repository_root: str | Path,
    artifact_root_id: str = "repository",
) -> CoreFinalEvaluationFramework:
    """Resolve one refinement slot with a content-addressed decision artifact."""

    matches = [item for item in framework.refinement_slots if item.slot_id == slot_id]
    if not matches:
        raise CoreFinalFrameworkError(f"unknown refinement slot: {slot_id}")
    if matches[0].status == "resolved":
        raise CoreFinalFrameworkError(f"refinement slot is already resolved: {slot_id}")
    missing_evidence = [
        role
        for role in matches[0].required_evidence_roles
        if next(
            item for item in framework.artifact_bindings if item.role == role
        ).status
        != "verified"
    ]
    if missing_evidence:
        raise CoreFinalFrameworkError(
            f"cannot resolve {slot_id}; required evidence is pending: "
            + ", ".join(missing_evidence)
        )
    content = read_stable_regular_file(
        decision_artifact_path, label=f"core refinement {slot_id}"
    )
    actual = sha256_bytes(content)
    if actual != expected_sha256:
        raise CoreFinalFrameworkError(
            f"decision digest mismatch for {slot_id}: expected {expected_sha256}, got {actual}"
        )
    relative = _repo_relative_path(Path(decision_artifact_path), Path(repository_root))
    replacement = matches[0].model_copy(
        update={
            "status": "resolved",
            "decision_artifact_root_id": artifact_root_id,
            "decision_artifact_path": relative,
            "decision_artifact_sha256": actual,
        }
    )
    payload = framework.model_dump(mode="python", exclude={"framework_sha256"})
    payload["refinement_slots"] = tuple(
        replacement if item.slot_id == slot_id else item
        for item in framework.refinement_slots
    )
    return _finalize_framework(payload)


def create_core_final_framework_artifact(
    framework: CoreFinalEvaluationFramework, output: str | Path
) -> Path:
    """Create one canonical framework artifact without overwriting anything."""

    return atomic_create_file(output, framework.canonical_bytes())


def load_core_final_framework_artifact(
    path: str | Path,
) -> CoreFinalEvaluationFramework:
    """Load one canonical, self-hashed framework artifact."""

    content = read_stable_regular_file(
        path, label="core-final framework", max_bytes=2 * 1024 * 1024
    )
    payload = parse_canonical_json(content, label="core-final framework")
    if not isinstance(payload, dict):
        raise CoreFinalFrameworkError("core-final framework must be a JSON object")
    try:
        return CoreFinalEvaluationFramework.model_validate_json(content)
    except ValueError as exc:
        raise CoreFinalFrameworkError("invalid core-final framework") from exc
