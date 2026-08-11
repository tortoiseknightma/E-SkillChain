"""Evaluator identity binding and the sealed ``judge_audit`` release gate.

The packet schemas live in :mod:`skillchain.evaluation.packets`.  This module
adds the runtime boundary that those schemas deliberately do not own:

* final and feedback packets have independent, exact-type serializers;
* prompt snapshots bind distinct cache namespaces and typed evaluator/model
  identities;
* a sealed audit payload is not read or decrypted until all five Assistant run
  bundles have independently passed external-digest verification.

All artifacts remain explicitly formal-ineligible until real evaluator
calibration has been run and recorded by a later protocol revision.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Annotated, Literal, Protocol, Self

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
from skillchain.evaluation.assistant_runs import (
    FORMAL_INELIGIBLE_REASON,
    VerifiedAssistantMatrixPlan,
    VerifiedFiveConfigRuns,
    VerifiedPhase4Inputs,
    require_verified_assistant_matrix_plan,
    require_verified_five_config_runs,
    require_verified_phase4_inputs,
)
from skillchain.evaluation.packets import (
    FEEDBACK_CACHE_NAMESPACE,
    FINAL_CACHE_NAMESPACE,
    EvaluationImage,
    EvaluatorPromptSnapshot,
    FeedbackPacket,
    FeedbackPacketV3,
    FinalEvaluationPacket,
    PromptMessage,
    build_feedback_evaluator_prompt,
    build_feedback_evaluator_prompt_v6,
    build_final_evaluator_prompt,
    build_final_evaluation_packet,
    derive_blinded_evaluation_id,
)
from skillchain.data.asset_catalog import AssetCatalog
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)
from skillchain.tools.registry import ToolRegistry

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

EVALUATOR_ISOLATION_POLICY_VERSION = "evaluator-isolation-v3"
JUDGE_AUDIT_SEAL_POLICY_VERSION = "judge-audit-seal-v1"

_FINAL_PACKET_FIELDS = frozenset(
    {
        "cache_namespace",
        "card_requirement",
        "cards",
        "evaluation_id",
        "image",
        "packet_kind",
        "packet_sha256",
        "response_text",
        "rubric",
        "schema_version",
        "tool_evidence",
        "turns",
    }
)
_FEEDBACK_PACKET_FIELDS = frozenset(
    {
        "acceptable_capabilities",
        "cache_namespace",
        "cards",
        "canonical_capability",
        "image",
        "packet_kind",
        "packet_sha256",
        "query_id",
        "response_text",
        "rubric",
        "schema_version",
        "tool_evidence",
        "tool_trace",
        "turns",
    }
)
_FEEDBACK_PACKET_V3_FIELDS = frozenset(
    {
        "acceptable_capabilities",
        "cache_namespace",
        "cards",
        "canonical_capability",
        "gcs_contract",
        "gcs_diagnostics",
        "image",
        "packet_kind",
        "packet_sha256",
        "query_id",
        "response_text",
        "rubric",
        "schema_version",
        "tool_evidence",
        "tool_trace",
        "turns",
    }
)
_FINAL_FORBIDDEN_KEYS = frozenset(
    {
        "acceptable_capabilities",
        "bank",
        "bank_sha256",
        "config",
        "description",
        "evolution_diagnostic",
        "ground_truth",
        "gt_capability",
        "judge_model",
        "judge_output",
        "judge_scores",
        "route_trace",
        "route_trace_sha256",
        "skill_slug",
        "source",
        "source_dataset",
        "split",
        "split_manifest_sha256",
    }
)
_FINAL_FORBIDDEN_VALUE_NAMES = frozenset(
    {
        "bank",
        "bank_sha256",
        "config",
        "ground_truth",
        "gt_capability",
        "judge_output",
        "judge_scores",
        "route_trace",
        "route_trace_sha256",
        "skill_slug",
        "source_dataset",
        "split",
        "split_manifest_sha256",
    }
)
_FINAL_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:config(?:uration)?|bank_sha256|skill_slug|ground_truth|"
    r"gt_capability|judge_output|judge_scores|split(?:_name)?|"
    r"route_trace(?:_sha256)?)\b\s*(?::|=|is\b)",
    re.IGNORECASE,
)
_FINAL_TREATMENT_NAME = re.compile(
    r"\b(?:no[ _-]?skill|llm[ _-]?static(?:[ _-]?skill)?|"
    r"spec[ _-]?baseline)\b",
    re.IGNORECASE,
)
_FINAL_TREATMENT_LABEL = re.compile(
    r"\b(?:full|s1|s2|s3|s1\+s2|s1s2)\b\s+"
    r"(?:config(?:uration)?|treatment)\b",
    re.IGNORECASE,
)


class EvaluatorIsolationError(ValueError):
    """A packet, prompt identity, or sealed audit violates isolation."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _hash_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(value)))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field_name}))


class FeedbackEvaluatorIdentity(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    role: Literal["development-feedback"] = "development-feedback"
    cache_namespace: Literal["feedback-evaluator-v2"] = FEEDBACK_CACHE_NAMESPACE
    provider: str
    model: str
    model_family: str
    endpoint: str
    identity_sha256: Sha256

    @field_validator("provider", "model", "model_family", "endpoint")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("https://") or "@" in value.split("/", 3)[2]:
            raise ValueError(
                "feedback evaluator endpoint must be credential-free HTTPS"
            )
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.identity_sha256 != _self_hash(self, "identity_sha256"):
            raise ValueError("feedback evaluator identity hash mismatch")
        return self


class FinalEvaluatorIdentity(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    role: Literal["blinded-final-judge"] = "blinded-final-judge"
    cache_namespace: Literal["final-evaluator-v2"] = FINAL_CACHE_NAMESPACE
    provider: str
    model: str
    model_family: str
    endpoint: str
    identity_sha256: Sha256

    @field_validator("provider", "model", "model_family", "endpoint")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("https://") or "@" in value.split("/", 3)[2]:
            raise ValueError("final evaluator endpoint must be credential-free HTTPS")
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.identity_sha256 != _self_hash(self, "identity_sha256"):
            raise ValueError("final evaluator identity hash mismatch")
        return self


class EvaluatorIsolationLock(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["evaluator-isolation-lock"] = "evaluator-isolation-lock"
    policy_version: Literal["evaluator-isolation-v3"] = (
        EVALUATOR_ISOLATION_POLICY_VERSION
    )
    feedback: FeedbackEvaluatorIdentity
    final: FinalEvaluatorIdentity
    shared_model_runtime: Literal[False] = False
    independence_limitation: Literal[
        "cross-provider-isolation-with-third-party-gateway-identity-risk"
    ] = "cross-provider-isolation-with-third-party-gateway-identity-risk"
    calibration_status: Literal["not-run"] = "not-run"
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    lock_sha256: Sha256

    @model_validator(mode="after")
    def validate_separation(self) -> Self:
        feedback_runtime = (
            self.feedback.provider,
            self.feedback.model,
            self.feedback.endpoint,
        )
        final_runtime = (
            self.final.provider,
            self.final.model,
            self.final.endpoint,
        )
        if feedback_runtime == final_runtime:
            raise ValueError(
                "evaluator-isolation-v3 requires different provider runtimes"
            )
        if self.feedback.model_family.casefold() == self.final.model_family.casefold():
            raise ValueError("evaluator-isolation-v3 requires different model families")
        if self.feedback.cache_namespace == self.final.cache_namespace:
            raise ValueError("feedback and final cache namespaces must differ")
        if self.feedback.identity_sha256 == self.final.identity_sha256:
            raise ValueError("feedback and final role identities must remain distinct")
        if self.lock_sha256 != _self_hash(self, "lock_sha256"):
            raise ValueError("evaluator isolation lock self hash mismatch")
        return self


class FeedbackEvaluatorPromptSnapshot(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    packet_kind: Literal["feedback"] = "feedback"
    cache_namespace: Literal["feedback-evaluator-v2"] = FEEDBACK_CACHE_NAMESPACE
    evaluator: FeedbackEvaluatorIdentity
    evaluator_isolation_sha256: Sha256
    packet_sha256: Sha256
    image: EvaluationImage
    messages: tuple[PromptMessage, PromptMessage]
    base_prompt_sha256: Sha256
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    prompt_sha256: Sha256

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.prompt_sha256 != _self_hash(self, "prompt_sha256"):
            raise ValueError("feedback prompt snapshot self hash mismatch")
        return self


class FinalEvaluatorPromptSnapshot(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    packet_kind: Literal["final"] = "final"
    cache_namespace: Literal["final-evaluator-v2"] = FINAL_CACHE_NAMESPACE
    evaluator: FinalEvaluatorIdentity
    evaluator_isolation_sha256: Sha256
    packet_sha256: Sha256
    image: EvaluationImage
    messages: tuple[PromptMessage, PromptMessage]
    base_prompt_sha256: Sha256
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    prompt_sha256: Sha256

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.prompt_sha256 != _self_hash(self, "prompt_sha256"):
            raise ValueError("final prompt snapshot self hash mismatch")
        return self


def make_feedback_evaluator_identity(
    *, provider: str, model: str, model_family: str, endpoint: str
) -> FeedbackEvaluatorIdentity:
    unsigned = {
        "schema_version": 1,
        "role": "development-feedback",
        "cache_namespace": FEEDBACK_CACHE_NAMESPACE,
        "provider": provider,
        "model": model,
        "model_family": model_family,
        "endpoint": endpoint,
    }
    return FeedbackEvaluatorIdentity.model_validate(
        {**unsigned, "identity_sha256": _hash_payload(unsigned)}, strict=True
    )


def make_final_evaluator_identity(
    *, provider: str, model: str, model_family: str, endpoint: str
) -> FinalEvaluatorIdentity:
    unsigned = {
        "schema_version": 1,
        "role": "blinded-final-judge",
        "cache_namespace": FINAL_CACHE_NAMESPACE,
        "provider": provider,
        "model": model,
        "model_family": model_family,
        "endpoint": endpoint,
    }
    return FinalEvaluatorIdentity.model_validate(
        {**unsigned, "identity_sha256": _hash_payload(unsigned)}, strict=True
    )


def make_evaluator_isolation_lock(
    feedback: FeedbackEvaluatorIdentity,
    final: FinalEvaluatorIdentity,
) -> EvaluatorIsolationLock:
    if type(feedback) is not FeedbackEvaluatorIdentity:
        raise TypeError("feedback identity must be FeedbackEvaluatorIdentity")
    if type(final) is not FinalEvaluatorIdentity:
        raise TypeError("final identity must be FinalEvaluatorIdentity")
    unsigned = {
        "schema_version": 1,
        "kind": "evaluator-isolation-lock",
        "policy_version": EVALUATOR_ISOLATION_POLICY_VERSION,
        "feedback": feedback,
        "final": final,
        "shared_model_runtime": False,
        "independence_limitation": (
            "cross-provider-isolation-with-third-party-gateway-identity-risk"
        ),
        "calibration_status": "not-run",
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    return EvaluatorIsolationLock.model_validate(
        {**unsigned, "lock_sha256": _hash_payload(unsigned)}, strict=True
    )


def make_active_portfolio_evaluator_isolation_lock() -> EvaluatorIsolationLock:
    """Bind active roles without retaining a stale pre-swap family label."""

    feedback = make_feedback_evaluator_identity(
        provider=config.FEEDBACK_JUDGE_PROVIDER,
        model=config.FEEDBACK_JUDGE_MODEL,
        model_family=config.FEEDBACK_JUDGE_MODEL,
        endpoint=config.PROVIDER_ENDPOINTS[config.FEEDBACK_JUDGE_PROVIDER],
    )
    final = make_final_evaluator_identity(
        provider=config.PORTFOLIO_JUDGE_PROVIDER,
        model=config.PORTFOLIO_JUDGE_MODEL,
        model_family=config.PORTFOLIO_JUDGE_MODEL,
        endpoint=config.PROVIDER_ENDPOINTS[config.PORTFOLIO_JUDGE_PROVIDER],
    )
    return make_evaluator_isolation_lock(feedback, final)


def serialize_final_evaluation_packet(packet: FinalEvaluationPacket) -> bytes:
    """Serialize only the exact final allowlist type and recursively audit keys."""

    packet = _validated_final_packet(packet)
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    parsed = _parse_object(content, "final evaluation packet")
    if set(parsed) != _FINAL_PACKET_FIELDS:
        raise EvaluatorIsolationError("final packet top-level allowlist mismatch")
    _assert_final_keys_clean(parsed, label="final packet")
    _assert_final_values_clean(parsed, label="final packet")
    return content


def serialize_feedback_packet(packet: FeedbackPacket) -> bytes:
    """Serialize only the distinct development-feedback packet type."""

    packet = _validated_feedback_packet(packet)
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    parsed = _parse_object(content, "feedback packet")
    if set(parsed) != _FEEDBACK_PACKET_FIELDS:
        raise EvaluatorIsolationError("feedback packet top-level allowlist mismatch")
    return content


def serialize_feedback_packet_v3(packet: FeedbackPacketV3) -> bytes:
    """Serialize only the forward GCS-diagnostic Feedback packet type."""

    packet = _validated_feedback_packet_v3(packet)
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    parsed = _parse_object(content, "feedback packet v3")
    if set(parsed) != _FEEDBACK_PACKET_V3_FIELDS:
        raise EvaluatorIsolationError("feedback packet v3 top-level allowlist mismatch")
    return content


def build_bound_feedback_prompt(
    packet: FeedbackPacket,
    evaluator_isolation: EvaluatorIsolationLock,
) -> FeedbackEvaluatorPromptSnapshot:
    evaluator_isolation = _validated_isolation_lock(evaluator_isolation)
    packet = _validated_feedback_packet(packet)
    serialize_feedback_packet(packet)
    base = build_feedback_evaluator_prompt(packet)
    return _feedback_snapshot(
        base,
        evaluator_isolation.feedback,
        evaluator_isolation.lock_sha256,
    )


def build_bound_feedback_prompt_v6(
    packet: FeedbackPacketV3,
    evaluator_isolation: EvaluatorIsolationLock,
) -> FeedbackEvaluatorPromptSnapshot:
    """Bind the forward GCS-aware Feedback prompt to evaluator isolation."""

    evaluator_isolation = _validated_isolation_lock(evaluator_isolation)
    packet = _validated_feedback_packet_v3(packet)
    serialize_feedback_packet_v3(packet)
    base = build_feedback_evaluator_prompt_v6(packet)
    return _feedback_snapshot(
        base,
        evaluator_isolation.feedback,
        evaluator_isolation.lock_sha256,
    )


def build_bound_final_prompt(
    packet: FinalEvaluationPacket,
    evaluator_isolation: EvaluatorIsolationLock,
) -> FinalEvaluatorPromptSnapshot:
    evaluator_isolation = _validated_isolation_lock(evaluator_isolation)
    packet = _validated_final_packet(packet)
    serialize_final_evaluation_packet(packet)
    base = build_final_evaluator_prompt(packet)
    visible_payload = _parse_object(
        base.messages[1].content.encode("utf-8") + b"\n",
        "final model-visible prompt payload",
    )
    _assert_final_keys_clean(visible_payload, label="final model-visible prompt")
    _assert_final_values_clean(visible_payload, label="final model-visible prompt")
    return _final_snapshot(
        base,
        evaluator_isolation.final,
        evaluator_isolation.lock_sha256,
    )


def build_verified_final_evaluation_packet(
    inputs: VerifiedPhase4Inputs,
    plan: VerifiedAssistantMatrixPlan,
    runs: VerifiedFiveConfigRuns,
    *,
    config: str,
    query_ordinal: int,
    registry: ToolRegistry,
    asset_catalog: AssetCatalog,
    blinding_key: bytes,
) -> FinalEvaluationPacket:
    """Build a final packet without accepting a caller-paired Query/result.

    Query, rubric, result row, and their ordering are selected only from typed,
    externally locked handles.  All handles are deeply revalidated before and
    after packet construction so an on-disk replacement cannot race the join.
    """

    inputs = require_verified_phase4_inputs(inputs)
    plan = require_verified_assistant_matrix_plan(plan, registry=registry)
    runs = require_verified_five_config_runs(runs)
    matrix = plan.plan
    plan_commitments = (
        matrix.plan_sha256,
        matrix.matrix_run_id,
        matrix.query_artifact_sha256,
        matrix.split_manifest_sha256,
        matrix.query_order_sha256,
        matrix.query_set_sha256,
        matrix.judge_audit_selection_manifest_sha256,
        matrix.judge_audit_evaluation_ids,
        matrix.judge_audit_evaluation_ids_sha256,
        matrix.final_rubric_sha256,
    )
    run_commitments = (
        runs.plan_sha256,
        runs.matrix_run_id,
        runs.query_artifact_sha256,
        runs.split_manifest_sha256,
        runs.query_order_sha256,
        runs.query_set_sha256,
        runs.judge_audit_selection_manifest_sha256,
        runs.judge_audit_evaluation_ids,
        runs.judge_audit_evaluation_ids_sha256,
        runs.final_rubric_sha256,
    )
    if run_commitments != plan_commitments:
        raise EvaluatorIsolationError("verified runs do not belong to the matrix plan")
    selection_ids = tuple(
        item.evaluation_id for item in inputs.selection.judge_audit_entries
    )
    input_commitments = (
        inputs.selection.matrix_run_id,
        inputs.expected_query_sha256,
        inputs.expected_split_manifest_sha256,
        inputs.expected_selection_file_sha256,
        selection_ids,
        inputs.selection.judge_audit_evaluation_ids_sha256,
        inputs.rubric.content_sha256,
    )
    if (
        input_commitments
        != (
            matrix.matrix_run_id,
            matrix.query_artifact_sha256,
            matrix.split_manifest_sha256,
            matrix.judge_audit_selection_manifest_sha256,
            matrix.judge_audit_evaluation_ids,
            matrix.judge_audit_evaluation_ids_sha256,
            matrix.final_rubric_sha256,
        )
        or inputs.assistant_queries != matrix.queries
    ):
        raise EvaluatorIsolationError(
            "verified query/rubric/selection inputs do not belong to the matrix plan"
        )
    for entry in inputs.selection.judge_audit_entries:
        expected_id = derive_blinded_evaluation_id(
            blinding_key=blinding_key,
            run_id=matrix.matrix_run_id,
            query_id=entry.query_id,
            config=entry.config,
        )
        if expected_id != entry.evaluation_id:
            raise EvaluatorIsolationError(
                "judge audit selection does not match the packet blinding key"
            )
    if type(query_ordinal) is not int or not 0 <= query_ordinal < len(matrix.queries):
        raise IndexError("query_ordinal is outside the verified matrix query order")
    try:
        run = next(item for item in runs.runs if item.manifest.config == config)
    except StopIteration:
        raise ValueError("config is not one of the five verified matrix runs") from None
    query = inputs.selected_queries[query_ordinal]
    request = run.requests[query_ordinal]
    row = run.rows[query_ordinal]
    if (
        request.query != matrix.queries[query_ordinal]
        or request.query.query_id != query.query_id
        or row.query_ordinal != query_ordinal
        or row.request_sha256 != request.request_sha256
        or row.result.query_id != query.query_id
        or row.result.config != config
    ):
        raise EvaluatorIsolationError(
            "verified query, request, and result row are not the same matrix instance"
        )
    packet = build_final_evaluation_packet(
        query,
        row.result,
        asset_catalog=asset_catalog,
        rubric=inputs.rubric,
        blinding_key=blinding_key,
    )
    require_verified_phase4_inputs(inputs)
    require_verified_assistant_matrix_plan(plan, registry=registry)
    require_verified_five_config_runs(runs)
    return packet


def _feedback_snapshot(
    base: EvaluatorPromptSnapshot,
    evaluator: FeedbackEvaluatorIdentity,
    evaluator_isolation_sha256: str,
) -> FeedbackEvaluatorPromptSnapshot:
    unsigned = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": FEEDBACK_CACHE_NAMESPACE,
        "evaluator": evaluator,
        "evaluator_isolation_sha256": evaluator_isolation_sha256,
        "packet_sha256": base.packet_sha256,
        "image": base.image,
        "messages": base.messages,
        "base_prompt_sha256": base.prompt_sha256,
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    return FeedbackEvaluatorPromptSnapshot.model_validate(
        {**unsigned, "prompt_sha256": _hash_payload(unsigned)}, strict=True
    )


def _final_snapshot(
    base: EvaluatorPromptSnapshot,
    evaluator: FinalEvaluatorIdentity,
    evaluator_isolation_sha256: str,
) -> FinalEvaluatorPromptSnapshot:
    unsigned = {
        "schema_version": 2,
        "packet_kind": "final",
        "cache_namespace": FINAL_CACHE_NAMESPACE,
        "evaluator": evaluator,
        "evaluator_isolation_sha256": evaluator_isolation_sha256,
        "packet_sha256": base.packet_sha256,
        "image": base.image,
        "messages": base.messages,
        "base_prompt_sha256": base.prompt_sha256,
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    return FinalEvaluatorPromptSnapshot.model_validate(
        {**unsigned, "prompt_sha256": _hash_payload(unsigned)}, strict=True
    )


class JudgeAuditDimension(_StrictFrozenModel):
    dimension: Literal["TCR", "CCC", "CQ", "CA"]
    human_score: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def validate_maximum(self) -> Self:
        maximum = 20 if self.dimension == "CQ" else 10
        if self.human_score > maximum:
            raise ValueError("judge audit dimension score exceeds rubric maximum")
        return self


class JudgeAuditReferenceRow(_StrictFrozenModel):
    evaluation_id: Sha256
    reviewer_id_sha256: Sha256
    requires_card: bool
    dimensions: tuple[JudgeAuditDimension, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        names = tuple(item.dimension for item in self.dimensions)
        expected = {"CA", "CQ", "TCR"}
        if self.requires_card:
            expected.add("CCC")
        if (
            set(names) != expected
            or names != tuple(sorted(names))
            or len(names) != len(set(names))
        ):
            raise ValueError("judge audit dimensions must be unique and sorted")
        return self


class JudgeAuditDataset(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["judge-audit-reference"] = "judge-audit-reference"
    audit_id: str
    rubric_sha256: Sha256
    rows: tuple[JudgeAuditReferenceRow, ...]
    dataset_sha256: Sha256

    @field_validator("audit_id")
    @classmethod
    def validate_audit_id(cls, value: str) -> str:
        return _nonblank(value, "audit_id")

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if not self.rows:
            raise ValueError("judge audit dataset must contain at least one row")
        ids = tuple(item.evaluation_id for item in self.rows)
        if len(ids) != len(set(ids)):
            raise ValueError("judge audit evaluation IDs must be unique")
        if self.dataset_sha256 != _self_hash(self, "dataset_sha256"):
            raise ValueError("judge audit dataset self hash mismatch")
        return self


class SealedJudgeAuditEnvelope(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["sealed-judge-audit"] = "sealed-judge-audit"
    policy_version: Literal["judge-audit-seal-v1"] = JUDGE_AUDIT_SEAL_POLICY_VERSION
    audit_id: str
    seal_scheme: str
    ciphertext_base64: str
    ciphertext_sha256: Sha256
    plaintext_sha256: Sha256
    query_count: int = Field(gt=0)
    matrix_plan_sha256: Sha256
    selection_manifest_sha256: Sha256
    evaluation_ids_sha256: Sha256
    final_rubric_sha256: Sha256
    evaluator_isolation_sha256: Sha256
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal["pending-real-model-and-judge-calibration"] = (
        FORMAL_INELIGIBLE_REASON
    )
    envelope_sha256: Sha256

    @field_validator("audit_id", "seal_scheme")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        try:
            ciphertext = base64.b64decode(self.ciphertext_base64, validate=True)
        except ValueError as error:
            raise ValueError(
                "judge audit ciphertext must use canonical base64"
            ) from error
        if (
            not ciphertext
            or base64.b64encode(ciphertext).decode("ascii") != self.ciphertext_base64
        ):
            raise ValueError(
                "judge audit ciphertext must be non-empty canonical base64"
            )
        if sha256_bytes(ciphertext) != self.ciphertext_sha256:
            raise ValueError("judge audit ciphertext hash mismatch")
        if self.envelope_sha256 != _self_hash(self, "envelope_sha256"):
            raise ValueError("judge audit envelope self hash mismatch")
        return self


class JudgeAuditDecryptor(Protocol):
    def __call__(self, ciphertext: bytes, *, seal_scheme: str, audit_id: str) -> bytes:
        """Decrypt externally sealed bytes after the five-run gate opens."""


@dataclass(frozen=True)
class CreatedSealedJudgeAudit:
    path: Path
    external_file_sha256: str


@dataclass(frozen=True)
class OpenedJudgeAudit:
    envelope: SealedJudgeAuditEnvelope
    dataset: JudgeAuditDataset
    run_plan_sha256: str
    evaluator_isolation_sha256: str
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: str = FORMAL_INELIGIBLE_REASON
    _sealed_bytes: bytes = field(repr=False, default=b"")


def build_judge_audit_dataset(
    *,
    audit_id: str,
    rubric_sha256: str,
    rows: tuple[JudgeAuditReferenceRow, ...],
) -> JudgeAuditDataset:
    unsigned = {
        "schema_version": 1,
        "kind": "judge-audit-reference",
        "audit_id": audit_id,
        "rubric_sha256": rubric_sha256,
        "rows": rows,
    }
    return JudgeAuditDataset.model_validate(
        {**unsigned, "dataset_sha256": _hash_payload(unsigned)}, strict=True
    )


def create_sealed_judge_audit(
    path: str | Path,
    *,
    audit_id: str,
    seal_scheme: str,
    ciphertext: bytes,
    plaintext_sha256: str,
    query_count: int,
    matrix_plan_sha256: str,
    selection_manifest_sha256: str,
    evaluation_ids_sha256: str,
    final_rubric_sha256: str,
    evaluator_isolation: EvaluatorIsolationLock,
) -> CreatedSealedJudgeAudit:
    """Persist opaque, externally encrypted audit bytes before model runs.

    Cryptography is deliberately delegated to the named external seal scheme;
    this project does not invent an ad-hoc cipher.  The plaintext commitment is
    checked only after the five-run release gate.
    """

    evaluator_isolation = _validated_isolation_lock(evaluator_isolation)
    if not isinstance(ciphertext, bytes) or not ciphertext:
        raise ValueError("sealed judge audit ciphertext must be non-empty bytes")
    unsigned = {
        "schema_version": 1,
        "kind": "sealed-judge-audit",
        "policy_version": JUDGE_AUDIT_SEAL_POLICY_VERSION,
        "audit_id": _nonblank(audit_id, "audit_id"),
        "seal_scheme": _nonblank(seal_scheme, "seal_scheme"),
        "ciphertext_base64": base64.b64encode(ciphertext).decode("ascii"),
        "ciphertext_sha256": sha256_bytes(ciphertext),
        "plaintext_sha256": plaintext_sha256,
        "query_count": query_count,
        "matrix_plan_sha256": matrix_plan_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "evaluation_ids_sha256": evaluation_ids_sha256,
        "final_rubric_sha256": final_rubric_sha256,
        "evaluator_isolation_sha256": evaluator_isolation.lock_sha256,
        "formal_eligible": False,
        "formal_ineligible_reason": FORMAL_INELIGIBLE_REASON,
    }
    envelope = SealedJudgeAuditEnvelope.model_validate(
        {**unsigned, "envelope_sha256": _hash_payload(unsigned)}, strict=True
    )
    content = canonical_json_bytes(envelope.model_dump(mode="json"))
    created = atomic_create_file(path, content)
    return CreatedSealedJudgeAudit(
        path=created,
        external_file_sha256=sha256_bytes(content),
    )


def open_sealed_judge_audit(
    path: str | Path,
    verified_runs: VerifiedFiveConfigRuns,
    *,
    expected_file_sha256: str,
    evaluator_isolation: EvaluatorIsolationLock,
    decryptor: JudgeAuditDecryptor,
) -> OpenedJudgeAudit:
    """Read/decrypt the audit only after all five run handles pass the gate."""

    # This check intentionally precedes even filesystem inspection of ``path``.
    verified_runs = require_verified_five_config_runs(verified_runs)
    evaluator_isolation = _validated_isolation_lock(evaluator_isolation)
    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "expected sealed-audit file sha256"
    )
    path = Path(path)
    try:
        content = read_stable_regular_file(path, label="sealed judge audit")
    except ArtifactFormatError as error:
        raise EvaluatorIsolationError(str(error)) from error
    if sha256_bytes(content) != expected_file_sha256:
        raise EvaluatorIsolationError("sealed judge audit external digest mismatch")
    envelope = _load_model(content, SealedJudgeAuditEnvelope, "sealed judge audit")
    if envelope.matrix_plan_sha256 != verified_runs.plan_sha256:
        raise EvaluatorIsolationError("sealed audit belongs to another matrix plan")
    if (
        envelope.selection_manifest_sha256
        != verified_runs.judge_audit_selection_manifest_sha256
        or envelope.evaluation_ids_sha256
        != verified_runs.judge_audit_evaluation_ids_sha256
        or envelope.final_rubric_sha256 != verified_runs.final_rubric_sha256
    ):
        raise EvaluatorIsolationError(
            "sealed audit selection or rubric commitment mismatch"
        )
    if envelope.evaluator_isolation_sha256 != evaluator_isolation.lock_sha256:
        raise EvaluatorIsolationError("sealed audit evaluator lock mismatch")
    ciphertext = base64.b64decode(envelope.ciphertext_base64, validate=True)
    plaintext = decryptor(
        ciphertext,
        seal_scheme=envelope.seal_scheme,
        audit_id=envelope.audit_id,
    )
    if not isinstance(plaintext, bytes):
        raise TypeError("judge audit decryptor must return bytes")
    if sha256_bytes(plaintext) != envelope.plaintext_sha256:
        raise EvaluatorIsolationError("judge audit plaintext commitment mismatch")
    dataset = _load_model(plaintext, JudgeAuditDataset, "judge audit plaintext")
    if dataset.audit_id != envelope.audit_id:
        raise EvaluatorIsolationError("judge audit identity mismatch after unsealing")
    if len(dataset.rows) != envelope.query_count:
        raise EvaluatorIsolationError("judge audit row count mismatch after unsealing")
    if envelope.query_count != len(verified_runs.judge_audit_evaluation_ids):
        raise EvaluatorIsolationError("judge audit selection count mismatch")
    if tuple(sorted(row.evaluation_id for row in dataset.rows)) != (
        verified_runs.judge_audit_evaluation_ids
    ):
        raise EvaluatorIsolationError("judge audit evaluation-ID set mismatch")
    if dataset.rubric_sha256 != verified_runs.final_rubric_sha256:
        raise EvaluatorIsolationError("judge audit rubric mismatch")
    try:
        after = read_stable_regular_file(path, label="sealed judge audit")
    except ArtifactFormatError as error:
        raise EvaluatorIsolationError(str(error)) from error
    if after != content:
        raise EvaluatorIsolationError("sealed judge audit changed while opening")
    require_verified_five_config_runs(verified_runs)
    return OpenedJudgeAudit(
        envelope=envelope,
        dataset=dataset,
        run_plan_sha256=verified_runs.plan_sha256,
        evaluator_isolation_sha256=evaluator_isolation.lock_sha256,
        _sealed_bytes=content,
    )


def _assert_final_keys_clean(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            if normalized in _FINAL_FORBIDDEN_KEYS or normalized.startswith(
                "canonical_"
            ):
                raise EvaluatorIsolationError(
                    f"{label} contains forbidden final-evaluator field: {key}"
                )
            _assert_final_keys_clean(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_final_keys_clean(item, label=label)


def _assert_final_values_clean(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_final_values_clean(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_final_values_clean(item, label=label)
    elif isinstance(value, str):
        normalized = value.strip().casefold().replace("-", "_")
        if (
            normalized in _FINAL_FORBIDDEN_VALUE_NAMES
            or _FINAL_SECRET_ASSIGNMENT.search(value)
            or _FINAL_TREATMENT_NAME.search(value)
            # Treatment labels are forbidden only when the complete public
            # value is itself a hidden label.  Product titles and evidence may
            # legitimately contain phrases such as "FULL TREATMENT".
            or _FINAL_TREATMENT_LABEL.fullmatch(value.strip())
        ):
            raise EvaluatorIsolationError(f"{label} contains treatment/secret text")


def _validated_final_packet(packet: object) -> FinalEvaluationPacket:
    if type(packet) is not FinalEvaluationPacket:
        raise TypeError("final serializer accepts only FinalEvaluationPacket")
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    return FinalEvaluationPacket.model_validate_json(content, strict=True)


def _validated_feedback_packet(packet: object) -> FeedbackPacket:
    if type(packet) is not FeedbackPacket:
        raise TypeError("feedback serializer accepts only FeedbackPacket")
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    return FeedbackPacket.model_validate_json(content, strict=True)


def _validated_feedback_packet_v3(packet: object) -> FeedbackPacketV3:
    if type(packet) is not FeedbackPacketV3:
        raise TypeError("feedback v3 serializer accepts only FeedbackPacketV3")
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    return FeedbackPacketV3.model_validate_json(content, strict=True)


def _validated_isolation_lock(value: object) -> EvaluatorIsolationLock:
    if type(value) is not EvaluatorIsolationLock:
        raise TypeError("evaluator isolation requires EvaluatorIsolationLock")
    content = canonical_json_bytes(value.model_dump(mode="json"))
    return EvaluatorIsolationLock.model_validate_json(content, strict=True)


def _parse_object(content: bytes, label: str) -> dict:
    try:
        parsed = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise EvaluatorIsolationError(str(error)) from error
    if not isinstance(parsed, dict):
        raise EvaluatorIsolationError(f"{label} must be a JSON object")
    return parsed


def _load_model(content: bytes, model_type, label: str):
    _parse_object(content, label)
    try:
        return model_type.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise EvaluatorIsolationError(
            f"{label}: schema validation failed: {error}"
        ) from error


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
    "EVALUATOR_ISOLATION_POLICY_VERSION",
    "JUDGE_AUDIT_SEAL_POLICY_VERSION",
    "CreatedSealedJudgeAudit",
    "EvaluatorIsolationError",
    "EvaluatorIsolationLock",
    "FeedbackEvaluatorIdentity",
    "FeedbackEvaluatorPromptSnapshot",
    "FinalEvaluatorIdentity",
    "FinalEvaluatorPromptSnapshot",
    "JudgeAuditDataset",
    "JudgeAuditDecryptor",
    "JudgeAuditDimension",
    "JudgeAuditReferenceRow",
    "OpenedJudgeAudit",
    "SealedJudgeAuditEnvelope",
    "build_bound_feedback_prompt",
    "build_bound_feedback_prompt_v6",
    "build_bound_final_prompt",
    "build_judge_audit_dataset",
    "build_verified_final_evaluation_packet",
    "create_sealed_judge_audit",
    "make_evaluator_isolation_lock",
    "make_active_portfolio_evaluator_isolation_lock",
    "make_feedback_evaluator_identity",
    "make_final_evaluator_identity",
    "open_sealed_judge_audit",
    "serialize_feedback_packet",
    "serialize_feedback_packet_v3",
    "serialize_final_evaluation_packet",
]
