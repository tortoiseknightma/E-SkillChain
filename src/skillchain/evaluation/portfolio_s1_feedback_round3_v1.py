"""Fresh fixed-240 S1 Feedback Round3 contracts.

Round3 is the third and final algorithm iteration.  It reuses only the frozen
SelectionV2/order, FeedbackPacketV3 source bindings, remote asset authority,
and cumulative cost commitments.  It imports zero historical Feedback outputs.
Every selected identity receives a fresh Qwen3.8-Max JSON-object primary call;
at most three ordered same-entry parser/length retries are globally claimable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
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

from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
)
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    VisualFeedbackOutput,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import _require_policy_labeled_suggestions
from skillchain.evaluation.packets import FeedbackPacketV3
from skillchain.evaluation.portfolio_gcs import GCS_CAPABILITY_ORDER, GCS_V2_POLICY_SHA256
from skillchain.evaluation.portfolio_s1_feedback import (
    FEEDBACK_PHASE_COUNTS_V2,
    FeedbackGCSContractProjectionV5,
    PortfolioS1FeedbackActionableSuggestionV5,
    PortfolioS1FeedbackBundleEntryV5,
    PortfolioS1FeedbackBundleV5,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackGCSContractSetV5,
    PortfolioS1FeedbackModelEntryV5,
    PortfolioS1FeedbackModelProjectionV5,
    PortfolioS1FeedbackRepresentativeExampleV5,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    _CREATOR_RUNTIME_HANDLE_RE,
    _feedback_v5_coverage,
    _hash_payload,
    _iter_string_values,
    _model_hash,
    _nested_json_model,
    _validate_model_projection_privacy,
    parse_policy_labeled_feedback_suggestion_v1,
    require_verified_static_feedback_source_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    SelectedFeedbackRemoteBindingV2,
    SelectedFeedbackRemoteRuntimeError,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    PARENT_SELECTION_FILE_SHA256,
    PARENT_SELECTION_SHA256,
    ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
    PortfolioS1FeedbackRecoveryAuthorizationV1,
    PortfolioS1FeedbackRecoveryControlV1,
    PortfolioS1FeedbackRecoveryLedgerV1,
    PortfolioS1FeedbackRecoveryRunV1,
    PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1,
    RecoveryFeedbackEvaluationResultV1,
    VerifiedPortfolioS1FeedbackParentEvidenceV1,
    build_portfolio_s1_feedback_recovery_control_v1,
    build_portfolio_s1_feedback_recovery_run_v1,
    build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1,
    load_feedback_recovery_ledger_v1,
    load_parent_selected_feedback_remote_runtime_v1,
    load_portfolio_s1_feedback_recovery_authorization_v1,
    load_portfolio_s1_feedback_recovery_control_v1,
    load_portfolio_s1_feedback_recovery_run_v1,
    load_portfolio_s1_feedback_parent_evidence_receipt_v1,
    load_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1,
    load_verified_parent_feedback_evidence_v1,
    redact_recovery_result_for_creator_privacy_v1,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6,
    QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13,
    QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3,
    QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3,
    Qwen38FeedbackModelSourceLockV3,
    Qwen38FeedbackPricingLockV6,
    Qwen38FeedbackRoleSelectionV13,
    load_qwen38_feedback_model_source_lock_v3,
    load_qwen38_feedback_pricing_lock_v6,
    load_qwen38_feedback_role_selection_v13,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import VerifiedStaticGCSCorpus
from skillchain.evaluation.visual_runtime import (
    SelectedFeedbackImageBinding,
    VerifiedSelectedFeedbackRemoteRuntime,
    _make_verified_selected_feedback_remote_runtime,
)
from skillchain.llm import LLMUsage
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes, sha256_bytes
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Round3Status = Literal["parsed", "parse_error", "provider_error", "timeout"]

ROUND3_PHASE_COUNTS = (12, 60, 120, 240)
ROUND3_PROVIDER_CALL_CEILING = 243
ROUND3_GLOBAL_RETRY_CEILING = 3
ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY = "22.764432000000"
ROUND3_PER_CALL_RESERVATION_CNY = "0.461544000000"
ROUND3_NEW_MAXIMUM_RESERVATION_CNY = "112.155192000000"
ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY = "134.919624000000"
ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY = "135.000000000000"
ROUND3_OWNER_BUDGET_CEILING_CNY = "150.000000000000"
ROUND3_PARENT_RECOVERY_RUN_SHA256 = (
    "e4308369746eca137d41c40efc4617f983f7b6463ca2daa1b1f77cd8b57452de"
)
ROUND3_PARENT_RECOVERY_ARTIFACT_SET_SHA256 = (
    "15e015d61caec024fbc6380e9a04390c8486e129fe53d296fd2a2be9b887d048"
)

ROUND3_PREDECESSOR_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-predecessors-v1"
ROUND3_AUTHORIZATION_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-authorization-v1"
ROUND3_CONTROL_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-control-v1"
ROUND3_LAUNCH_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-launch-v1"
ROUND3_RETRY_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-global-retry-v1"
ROUND3_RESERVATION_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-reservation-v1"
ROUND3_BOUND_POLICY_VERSION_V1 = "portfolio-s1-bound-feedback-round3-v1"
ROUND3_RUN_POLICY_VERSION_V1 = "portfolio-s1-feedback-round3-run-v1"
ROUND3_BUNDLE_POLICY_VERSION_V10 = "portfolio-s1-feedback-bundle-v10"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def round3_retry_policy_v1() -> dict[str, object]:
    return {
        "policy_version": ROUND3_RETRY_POLICY_VERSION_V1,
        "scope": "fresh-fixed-selection240-same-order",
        "provider_call_ceiling": 243,
        "normal_calls": 240,
        "global_retry_ceiling": 3,
        "max_lifetime_attempts_per_entry": 2,
        "attempt_transport_policy_version": ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
        "attempt_transport_policy_sha256": ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
        "retry_eligibility": [
            "strict-parser-failure-without-policy-label-only-failure",
            "length-parser-failure",
            "reasoning-present-empty-assistant-content",
        ],
        "terminal_without_retry": [
            "provider-error",
            "timeout",
            "refusal",
            "tool-call",
            "input-echo",
            "privacy-failure",
            "orphan",
            "fourth-eligible-failure",
        ],
        "persistence_order": [
            "claim-create-only-if-needed",
            "reservation-create-only",
            "provider-invoke-once",
            "settlement-create-only",
        ],
    }


ROUND3_RETRY_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(round3_retry_policy_v1())
)


class PortfolioS1FeedbackRound3PredecessorReceiptV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-predecessors"] = (
        "portfolio-s1-feedback-round3-predecessors"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-predecessors-v1"] = (
        ROUND3_PREDECESSOR_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    base_parent_evidence_sha256: Sha256
    base_parent_run_sha256: Sha256
    recovery_authorization_file_sha256: Sha256
    recovery_control_file_sha256: Sha256
    recovery_launch_file_sha256: Sha256
    recovery_launch_sha256: Sha256
    recovery_parent_evidence_file_sha256: Sha256
    recovery_run_file_sha256: Sha256
    recovery_run_sha256: Literal[ROUND3_PARENT_RECOVERY_RUN_SHA256] = (
        ROUND3_PARENT_RECOVERY_RUN_SHA256
    )
    recovery_artifact_set_sha256: Literal[
        ROUND3_PARENT_RECOVERY_ARTIFACT_SET_SHA256
    ] = ROUND3_PARENT_RECOVERY_ARTIFACT_SET_SHA256
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    selected_count: Literal[240] = 240
    historical_feedback_outputs_imported: Literal[0] = 0
    historical_roots_read_only: Literal[True] = True
    recovery_top_inventory_sha256: Sha256
    recovery_claim_inventory_sha256: Sha256
    recovery_reservation_inventory_sha256: Sha256
    recovery_bound_inventory_sha256: Sha256
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.receipt_sha256 != _model_hash(self, "receipt_sha256"):
            raise ValueError("Round3 predecessor receipt hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3PredecessorsV1:
    base: VerifiedPortfolioS1FeedbackParentEvidenceV1
    recovery_root: Path
    recovery_authorization: PortfolioS1FeedbackRecoveryAuthorizationV1
    recovery_control: PortfolioS1FeedbackRecoveryControlV1
    recovery_launch: PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1
    recovery_run: PortfolioS1FeedbackRecoveryRunV1
    recovery_ledger: PortfolioS1FeedbackRecoveryLedgerV1
    receipt: PortfolioS1FeedbackRound3PredecessorReceiptV1


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3GovernanceV1:
    repository_root: Path
    source_lock: Qwen38FeedbackModelSourceLockV3
    pricing_lock: Qwen38FeedbackPricingLockV6
    role_selection: Qwen38FeedbackRoleSelectionV13
    _marker: object | None = field(default=None, repr=False, compare=False)


_VERIFIED_ROUND3_GOVERNANCE_TOKEN = object()


def load_verified_round3_governance_v1(
    repository_root: str | Path,
) -> VerifiedPortfolioS1FeedbackRound3GovernanceV1:
    """Load the exact checked-in V3/V6/V13 live-call authority triad."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or any(
        value == "0" * 64
        for value in (
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3,
            QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3,
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6,
            QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6,
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13,
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13,
        )
    ):
        raise PortfolioS1FeedbackError("Round3 governance constants are not frozen")
    source = load_qwen38_feedback_model_source_lock_v3(
        root / QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3,
        expected_file_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3,
    )
    pricing = load_qwen38_feedback_pricing_lock_v6(
        root / QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6,
        expected_file_sha256=QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6,
    )
    role = load_qwen38_feedback_role_selection_v13(
        root / QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13,
        expected_file_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13,
    )
    if (
        type(source) is not Qwen38FeedbackModelSourceLockV3
        or type(pricing) is not Qwen38FeedbackPricingLockV6
        or type(role) is not Qwen38FeedbackRoleSelectionV13
        or source.source_lock_sha256 != QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
        or pricing.pricing_lock_sha256 != QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
        or role.selection_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
        or source.requested_response_format != "json_object"
        or not source.old_remote_auth_does_not_authorize_round3_transport
        or not source.live_call_authority
        or not pricing.live_provider_calls_authorized
        or role.feedback_evaluator.get("requested_response_format") != "json_object"
        or role.feedback_evaluator.get("live_call_authority") is not True
        or role.feedback_evaluator.get(
            "old_remote_auth_does_not_authorize_round3_transport"
        )
        is not True
    ):
        raise PortfolioS1FeedbackError("Round3 governance triad drifted")
    return VerifiedPortfolioS1FeedbackRound3GovernanceV1(
        repository_root=root,
        source_lock=source,
        pricing_lock=pricing,
        role_selection=role,
        _marker=_VERIFIED_ROUND3_GOVERNANCE_TOKEN,
    )


def load_verified_round3_predecessors_v1(
    base_parent_root: str | Path,
    recovery_root: str | Path,
) -> VerifiedPortfolioS1FeedbackRound3PredecessorsV1:
    """Bind prior cost and fixed selection while importing zero prior outputs."""

    base = load_verified_parent_feedback_evidence_v1(base_parent_root)
    supplied = Path(recovery_root)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackError("Round3 predecessor root cannot be a symlink")
    root = supplied.resolve(strict=True)
    expected_files = {
        "authorization-recovery-v1.json",
        "control-recovery-v1.json",
        "launch-lock-recovery-v1.json",
        "parent-evidence-v1.json",
        "run-recovery-v1.json",
    }
    expected_dirs = {
        "bound-feedback-recovery-v1",
        "provider-attempts-recovery-v1",
        "recovery-claims-v1",
    }
    children = tuple(root.iterdir())
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in children if item.is_file()} != expected_files
        or {item.name for item in children if item.is_dir()} != expected_dirs
        or any(item.is_symlink() for item in children)
    ):
        raise PortfolioS1FeedbackError("Round3 predecessor inventory drifted")
    parent_evidence = load_portfolio_s1_feedback_parent_evidence_receipt_v1(
        root / "parent-evidence-v1.json"
    )
    if parent_evidence != base.receipt:
        raise PortfolioS1FeedbackError("Round3 predecessor base evidence drifted")
    authorization = load_portfolio_s1_feedback_recovery_authorization_v1(
        root / "authorization-recovery-v1.json"
    )
    control = load_portfolio_s1_feedback_recovery_control_v1(
        root / "control-recovery-v1.json"
    )
    launch = load_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
        root / "launch-lock-recovery-v1.json"
    )
    run = load_portfolio_s1_feedback_recovery_run_v1(root / "run-recovery-v1.json")
    if (
        control != build_portfolio_s1_feedback_recovery_control_v1(base, authorization)
        or launch
        != build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
            base, authorization, control, run_id=launch.run_id
        )
    ):
        raise PortfolioS1FeedbackError("Round3 predecessor control drifted")
    ledger = load_feedback_recovery_ledger_v1(
        root, parent=base, authorization=authorization, control=control
    )
    if run != build_portfolio_s1_feedback_recovery_run_v1(
        base, authorization, control, ledger, terminal_reason=run.terminal_reason
    ):
        raise PortfolioS1FeedbackError("Round3 predecessor run differs from ledger")
    if (
        run.status != "stopped_nonparsed"
        or run.parsed_total != 154
        or run.attempted_total != 155
        or run.cumulative_provider_call_count != 160
        or run.recovery_claim_count != 3
        or run.orphan_count != 0
        or run.run_sha256 != ROUND3_PARENT_RECOVERY_RUN_SHA256
        or run.artifact_set_sha256 != ROUND3_PARENT_RECOVERY_ARTIFACT_SET_SHA256
        or run.cumulative_actual_cost_cny != ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    ):
        raise PortfolioS1FeedbackError("Round3 predecessor terminal facts drifted")
    def inventory_sha256(directory: Path) -> str:
        members = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
        if any(
            not item.is_file() or item.is_symlink() or item.suffix != ".json"
            for item in members
        ):
            raise PortfolioS1FeedbackError("Round3 predecessor directory drifted")
        return _hash_payload(
            [
                {
                    "name": item.name,
                    "file_sha256": sha256_bytes(
                        read_stable_regular_file(
                            item,
                            label=f"Round3 predecessor {item.name}",
                            max_bytes=16 * 1024 * 1024,
                        )
                    ),
                }
                for item in members
            ]
        )

    top_inventory_sha256 = _hash_payload(
        [
            {
                "name": name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        root / name,
                        label=f"Round3 predecessor {name}",
                        max_bytes=128 * 1024 * 1024,
                    )
                ),
            }
            for name in sorted(expected_files)
        ]
    )
    draft = PortfolioS1FeedbackRound3PredecessorReceiptV1.model_construct(
        base_parent_evidence_sha256=base.receipt.evidence_sha256,
        base_parent_run_sha256=base.run.run_sha256,
        recovery_authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        recovery_control_file_sha256=sha256_bytes(control.canonical_bytes()),
        recovery_launch_file_sha256=sha256_bytes(launch.canonical_bytes()),
        recovery_launch_sha256=launch.launch_lock_sha256,
        recovery_parent_evidence_file_sha256=sha256_bytes(parent_evidence.canonical_bytes()),
        recovery_run_file_sha256=sha256_bytes(run.canonical_bytes()),
        recovery_top_inventory_sha256=top_inventory_sha256,
        recovery_claim_inventory_sha256=inventory_sha256(root / "recovery-claims-v1"),
        recovery_reservation_inventory_sha256=inventory_sha256(
            root / "provider-attempts-recovery-v1"
        ),
        recovery_bound_inventory_sha256=inventory_sha256(
            root / "bound-feedback-recovery-v1"
        ),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1FeedbackRound3PredecessorReceiptV1.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )
    return VerifiedPortfolioS1FeedbackRound3PredecessorsV1(
        base=base,
        recovery_root=root,
        recovery_authorization=authorization,
        recovery_control=control,
        recovery_launch=launch,
        recovery_run=run,
        recovery_ledger=ledger,
        receipt=receipt,
    )


class PortfolioS1FeedbackRound3AuthorizationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-authorization"] = (
        "portfolio-s1-feedback-round3-authorization"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-authorization-v1"] = (
        ROUND3_AUTHORIZATION_POLICY_VERSION_V1
    )
    status: Literal["owner-approved"] = "owner-approved"
    authorization_id: str
    reviewer_id: str
    reviewed_at: str
    predecessor_receipt_sha256: Sha256
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    selected_entry_sha256s: tuple[Sha256, ...]
    selected_query_ids: tuple[str, ...]
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = ROUND3_PHASE_COUNTS
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1] = (
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    )
    response_format: Literal["json_object"] = "json_object"
    thinking_budget: Literal[2048] = 2048
    max_completion_tokens: Literal[6144] = 6144
    timeout_seconds: Literal[600] = 600
    provider_internal_max_attempts: Literal[1] = 1
    global_retry_ceiling: Literal[3] = 3
    provider_call_ceiling: Literal[243] = 243
    parent_cumulative_actual_cost_cny: Literal[
        ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    new_maximum_reservation_cny: Literal[
        ROUND3_NEW_MAXIMUM_RESERVATION_CNY
    ] = ROUND3_NEW_MAXIMUM_RESERVATION_CNY
    cumulative_maximum_reservation_cny: Literal[
        ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    ] = ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    cumulative_technical_hard_cap_cny: Literal[
        ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    ] = ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    owner_budget_ceiling_cny: Literal[ROUND3_OWNER_BUDGET_CEILING_CNY] = (
        ROUND3_OWNER_BUDGET_CEILING_CNY
    )
    model_source_lock_file: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3
    )
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    )
    pricing_lock_file: Literal[QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6] = (
        QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    )
    role_selection_file: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
    )
    live_call_authority: Literal[True] = True
    membership_ancestry_role: Literal[
        "image_source_membership_only_not_call_authorization"
    ] = "image_source_membership_only_not_call_authorization"
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    membership_authorization_id: str
    membership_authorization_file_sha256: Sha256
    membership_receipt_file_sha256: Sha256
    membership_receipt_sha256: Sha256
    membership_catalog_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator(
        "selected_entry_sha256s", "selected_query_ids", "phase_counts", mode="before"
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("reviewed_at")
    @classmethod
    def _reviewed_at(cls, value: str) -> str:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("Round3 reviewed_at is not canonical ISO-8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond:
            raise ValueError("Round3 reviewed_at needs timezone and second precision")
        canonical = parsed.isoformat(timespec="seconds")
        if parsed.utcoffset().total_seconds() == 0:
            canonical = canonical.removesuffix("+00:00") + "Z"
        if canonical != value:
            raise ValueError("Round3 reviewed_at is not canonical ISO-8601")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if (
            len(self.selected_entry_sha256s) != 240
            or len(set(self.selected_entry_sha256s)) != 240
            or len(self.selected_query_ids) != 240
            or len(set(self.selected_query_ids)) != 240
            or self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or Decimal(self.parent_cumulative_actual_cost_cny)
            + Decimal(self.new_maximum_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or Decimal(self.cumulative_technical_hard_cap_cny)
            > Decimal(self.owner_budget_ceiling_cny)
            or self.authorization_sha256 != _model_hash(self, "authorization_sha256")
        ):
            raise ValueError("Round3 authorization drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_authorization_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: str,
) -> PortfolioS1FeedbackRound3AuthorizationV1:
    if (
        type(governance) is not VerifiedPortfolioS1FeedbackRound3GovernanceV1
        or governance._marker is not _VERIFIED_ROUND3_GOVERNANCE_TOKEN
    ):
        raise PortfolioS1FeedbackError("Round3 governance is not verified")
    selection = predecessors.base.selection
    parent = predecessors.base
    draft = PortfolioS1FeedbackRound3AuthorizationV1.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        predecessor_receipt_sha256=predecessors.receipt.receipt_sha256,
        selected_entry_sha256s=tuple(item.entry_sha256 for item in selection.entries),
        selected_query_ids=tuple(item.query_id for item in selection.entries),
        membership_authorization_id=parent.authorization.authorization_id,
        membership_authorization_file_sha256=sha256_bytes(
            parent.authorization.canonical_bytes()
        ),
        membership_receipt_file_sha256=sha256_bytes(
            parent.remote_receipt.canonical_bytes()
        ),
        membership_receipt_sha256=parent.remote_receipt.receipt_sha256,
        membership_catalog_sha256=parent.authorization.parent_remote_catalog_sha256,
        authorization_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackRound3AuthorizationV1.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-remote-runtime"] = (
        "portfolio-s1-feedback-round3-remote-runtime"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-remote-runtime-v1"
    ] = "portfolio-s1-feedback-round3-remote-runtime-v1"
    processor: Literal["dashscope-qwen38-feedback"] = "dashscope-qwen38-feedback"
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    authorization_id: str
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    corpus_sha256: Sha256
    catalog_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
    )
    live_transport_policy_sha256: Literal[
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    membership_ancestry_role: Literal[
        "image_source_membership_only_not_call_authorization"
    ] = "image_source_membership_only_not_call_authorization"
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    membership_authorization_id: str
    membership_authorization_file_sha256: Sha256
    membership_receipt_file_sha256: Sha256
    membership_receipt_sha256: Sha256
    selected_count: Literal[240] = 240
    selected_bindings: tuple[SelectedFeedbackRemoteBindingV2, ...]
    selected_binding_set_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("selected_bindings", mode="before")
    @classmethod
    def _bindings_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if (
            tuple(item.selection_ordinal for item in self.selected_bindings)
            != tuple(range(1, 241))
            or len({item.selection_entry_sha256 for item in self.selected_bindings})
            != 240
            or len({item.query_id for item in self.selected_bindings}) != 240
            or len({item.asset_id for item in self.selected_bindings}) != 240
            or self.selected_binding_set_sha256
            != _hash_payload(
                [item.model_dump(mode="json") for item in self.selected_bindings]
            )
            or self.receipt_sha256 != _model_hash(self, "receipt_sha256")
        ):
            raise ValueError("Round3 remote runtime receipt drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def prepare_round3_remote_runtime_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    corpus: VerifiedStaticGCSCorpus,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create/resume the fresh authority while using V7 only for image membership."""

    if authorization != build_round3_authorization_v1(
        predecessors,
        governance,
        authorization_id=authorization.authorization_id,
        reviewer_id=authorization.reviewer_id,
        reviewed_at=authorization.reviewed_at,
    ):
        raise SelectedFeedbackRemoteRuntimeError("Round3 authorization drifted")
    selection = predecessors.base.selection
    if (
        corpus.corpus_sha256 != selection.corpus_sha256
        or len(verified_sources) != 240
        or tuple(item.selection_entry_sha256 for item in verified_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "Round3 remote sources differ from fixed selected240"
        )
    membership_runtime = load_parent_selected_feedback_remote_runtime_v1(
        predecessors.base,
        parent_remote_runtime,
        verified_sources=verified_sources,
    )
    if (
        authorization.membership_authorization_id,
        authorization.membership_authorization_file_sha256,
        authorization.membership_receipt_file_sha256,
        authorization.membership_receipt_sha256,
        authorization.membership_catalog_sha256,
    ) != (
        membership_runtime.authorization.authorization_id,
        membership_runtime.authorization_file_sha256,
        membership_runtime.receipt_file_sha256,
        membership_runtime.receipt.receipt_sha256,
        membership_runtime.catalog.catalog_sha256,
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "Round3 membership ancestry drifted"
        )
    receipt_bindings: list[SelectedFeedbackRemoteBindingV2] = []
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, verified_sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, selection, predecessors.base.control, entry
        )
        resolution = membership_runtime.catalog.verify_reference(
            entry.asset_id,
            source.row.query.image_path,
            entry.leakage_group_id,
        )
        if (
            resolution.asset.asset_id != entry.asset_id
            or resolution.asset.sha256 != entry.image_sha256
            or resolution.asset.cloud_upload_allowed is not True
        ):
            raise SelectedFeedbackRemoteRuntimeError(
                "Round3 remote image membership drifted"
            )
        receipt_bindings.append(
            SelectedFeedbackRemoteBindingV2(
                selection_ordinal=entry.selection_ordinal,
                selection_entry_sha256=entry.entry_sha256,
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
        image_bindings.append(
            SelectedFeedbackImageBinding(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
        )
    draft = PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1.model_construct(
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        corpus_sha256=corpus.corpus_sha256,
        catalog_sha256=membership_runtime.catalog.catalog_sha256,
        membership_authorization_id=authorization.membership_authorization_id,
        membership_authorization_file_sha256=(
            authorization.membership_authorization_file_sha256
        ),
        membership_receipt_file_sha256=authorization.membership_receipt_file_sha256,
        membership_receipt_sha256=authorization.membership_receipt_sha256,
        selected_bindings=tuple(receipt_bindings),
        selected_binding_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in receipt_bindings]
        ),
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    expected = PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target, label="Round3 remote receipt", max_bytes=8 * 1024 * 1024
        )
        receipt = (
            PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1.model_validate_json(
                content, strict=True
            )
        )
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "Round3 remote receipt resume conflict"
            )
    else:
        if target.is_symlink():
            raise SelectedFeedbackRemoteRuntimeError(
                "Round3 remote receipt target is a symlink"
            )
        atomic_create_file(target, expected.canonical_bytes())
        receipt = expected
        content = expected.canonical_bytes()
    return _make_verified_selected_feedback_remote_runtime(
        authorization=authorization,
        receipt=receipt,
        catalog=membership_runtime.catalog,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        receipt_file_sha256=sha256_bytes(content),
        selected_bindings=tuple(image_bindings),
        processor="dashscope-qwen38-feedback",
        expected_binding_count=240,
    )


class PortfolioS1FeedbackRound3ControlV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-control"] = (
        "portfolio-s1-feedback-round3-control"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-control-v1"] = (
        ROUND3_CONTROL_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    source_control_sha256: Sha256
    source_control_file_sha256: Sha256
    predecessor_receipt_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
    )
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    remote_catalog_sha256: Sha256
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = ROUND3_PHASE_COUNTS
    retry_policy_sha256: Literal[ROUND3_RETRY_POLICY_SHA256_V1] = (
        ROUND3_RETRY_POLICY_SHA256_V1
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1] = (
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    )
    response_format: Literal["json_object"] = "json_object"
    cumulative_maximum_reservation_cny: Literal[
        ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    ] = ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    cumulative_technical_hard_cap_cny: Literal[
        ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    ] = ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    feedback_concurrency: Literal[2] = 2
    max_active_calls: Literal[2] = 2
    historical_feedback_outputs_imported: Literal[0] = 0
    stop_on_nonparsed_or_orphan: Literal[True] = True
    require_parsed240_for_bundle: Literal[True] = True
    control_sha256: Sha256

    @field_validator("phase_counts", mode="before")
    @classmethod
    def _phases(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or self.control_sha256 != _model_hash(self, "control_sha256")
        ):
            raise ValueError("Round3 control drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_control_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
) -> PortfolioS1FeedbackRound3ControlV1:
    expected_authorization = build_round3_authorization_v1(
        predecessors,
        governance,
        authorization_id=authorization.authorization_id,
        reviewer_id=authorization.reviewer_id,
        reviewed_at=authorization.reviewed_at,
    )
    if (
        authorization != expected_authorization
        or remote_receipt.authorization_id != authorization.authorization_id
        or remote_receipt.authorization_sha256 != authorization.authorization_sha256
        or remote_receipt.authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or remote_receipt.selection_sha256
        != predecessors.base.selection.selection_sha256
        or remote_receipt.membership_authorization_id
        != authorization.membership_authorization_id
        or remote_receipt.membership_receipt_sha256
        != authorization.membership_receipt_sha256
        or remote_receipt.catalog_sha256 != authorization.membership_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("Round3 authorization/predecessor drifted")
    draft = PortfolioS1FeedbackRound3ControlV1.model_construct(
        source_control_sha256=predecessors.base.control.control_sha256,
        source_control_file_sha256=sha256_bytes(predecessors.base.control.canonical_bytes()),
        predecessor_receipt_sha256=predecessors.receipt.receipt_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        remote_receipt_file_sha256=sha256_bytes(remote_receipt.canonical_bytes()),
        remote_receipt_sha256=remote_receipt.receipt_sha256,
        remote_catalog_sha256=remote_receipt.catalog_sha256,
        control_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"control_sha256"})
    return PortfolioS1FeedbackRound3ControlV1.model_validate(
        {**unsigned, "control_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1FeedbackRound3LaunchV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-launch"] = (
        "portfolio-s1-feedback-round3-launch"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-launch-v1"] = (
        ROUND3_LAUNCH_POLICY_VERSION_V1
    )
    run_id: str
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    predecessor_receipt_sha256: Sha256
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
    )
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    provider_call_ceiling: Literal[243] = 243
    global_retry_ceiling: Literal[3] = 3
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    transport_policy_sha256: Literal[ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1] = (
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    )
    response_format: Literal["json_object"] = "json_object"
    cumulative_maximum_reservation_cny: Literal[
        ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    ] = ROUND3_CUMULATIVE_MAXIMUM_RESERVATION_CNY
    cumulative_technical_hard_cap_cny: Literal[
        ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    ] = ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    launch_sha256: Sha256

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if (
            Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or self.launch_sha256 != _model_hash(self, "launch_sha256")
        ):
            raise ValueError("Round3 launch hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_launch_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    *,
    run_id: str,
) -> PortfolioS1FeedbackRound3LaunchV1:
    if control != build_round3_control_v1(
        predecessors, governance, authorization, remote_receipt
    ):
        raise PortfolioS1FeedbackError("Round3 launch control drifted")
    draft = PortfolioS1FeedbackRound3LaunchV1.model_construct(
        run_id=run_id,
        predecessor_receipt_sha256=predecessors.receipt.receipt_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        control_file_sha256=sha256_bytes(control.canonical_bytes()),
        remote_receipt_file_sha256=sha256_bytes(remote_receipt.canonical_bytes()),
        remote_receipt_sha256=remote_receipt.receipt_sha256,
        launch_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_sha256"})
    return PortfolioS1FeedbackRound3LaunchV1.model_validate(
        {**unsigned, "launch_sha256": _hash_payload(unsigned)}, strict=True
    )


def is_round3_retry_eligible_v1(result: RecoveryFeedbackEvaluationResultV1) -> bool:
    """Admit only safe JSON parser/length/empty-content failures."""

    if (
        type(result) is not RecoveryFeedbackEvaluationResultV1
        or result.wire_kind != "round3_primary_json_object_v1"
        or result.status != "parse_error"
        or result.error_code != "invalid_feedback_json"
        or result.request_id is None
        or result.usage is None
        or result.response_redaction_reason is not None
        or result.refusal_present is not False
        or result.tool_calls
        or result.raw_response_text is None
        or result.finish_reason not in {"stop", "length"}
    ):
        return False
    if result.finish_reason == "length":
        return True
    if not result.raw_response_text:
        return result.reasoning_present is True and result.raw_response_bytes == 0
    try:
        parsed = parse_visual_feedback_output_v3(result.raw_response_text)
    except EvaluatorOutputParseError:
        return True
    try:
        _require_policy_labeled_suggestions(parsed)
    except EvaluatorOutputParseError:
        # A policy-label-only failure is semantic, not transport/schema retryable.
        return False
    return False


class Round3GlobalRetryClaimV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-global-retry-claim"] = (
        "portfolio-s1-feedback-round3-global-retry-claim"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-global-retry-v1"
    ] = ROUND3_RETRY_POLICY_VERSION_V1
    policy_sha256: Literal[ROUND3_RETRY_POLICY_SHA256_V1] = (
        ROUND3_RETRY_POLICY_SHA256_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    control_sha256: Sha256
    claim_ordinal: Literal[1, 2, 3]
    previous_claim_sha256: Sha256 | None = None
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    first_global_call_ordinal: int = Field(ge=1, le=242)
    first_artifact_sha256: Sha256
    first_feedback_result_sha256: Sha256
    trigger_finish_reason: Literal["stop", "length"]
    trigger_raw_response_bytes: int = Field(ge=0)
    claim_sha256: Sha256

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        if (self.claim_ordinal == 1) != (self.previous_claim_sha256 is None):
            raise ValueError("Round3 retry claim chain is not contiguous")
        if self.claim_sha256 != _model_hash(self, "claim_sha256"):
            raise ValueError("Round3 retry claim hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Round3FeedbackCallReservationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-call-reservation"] = (
        "portfolio-s1-feedback-round3-call-reservation"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-reservation-v1"] = (
        ROUND3_RESERVATION_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    source_control_sha256: Sha256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=243)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    wire_kind: Literal["round3_primary_json_object_v1"] = (
        "round3_primary_json_object_v1"
    )
    provider_internal_max_attempts: Literal[1] = 1
    reservation_cny: Literal[ROUND3_PER_CALL_RESERVATION_CNY] = (
        ROUND3_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        ancestry = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
        )
        if self.attempt_index == 1 and any(item is not None for item in ancestry):
            raise ValueError("Round3 first reservation carries retry ancestry")
        if self.attempt_index == 2 and any(item is None for item in ancestry):
            raise ValueError("Round3 retry reservation lacks claim ancestry")
        if self.reservation_sha256 != _model_hash(self, "reservation_sha256"):
            raise ValueError("Round3 reservation hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class BoundRound3FeedbackArtifactV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-bound-feedback-round3"] = (
        "portfolio-s1-bound-feedback-round3"
    )
    policy_version: Literal["portfolio-s1-bound-feedback-round3-v1"] = (
        ROUND3_BOUND_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    source_control_sha256: Sha256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=243)
    reservation_sha256: Sha256
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    feedback_packet: FeedbackPacketV3
    feedback_result: RecoveryFeedbackEvaluationResultV1
    status: Round3Status
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, RecoveryFeedbackEvaluationResultV1)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        ancestry = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
        )
        if (
            self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.wire_kind != "round3_primary_json_object_v1"
            or self.feedback_result.status != self.status
            or (self.attempt_index == 1 and any(item is not None for item in ancestry))
            or (self.attempt_index == 2 and any(item is None for item in ancestry))
            or self.artifact_sha256 != _model_hash(self, "artifact_sha256")
        ):
            raise ValueError("Round3 bound artifact drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_round3_claim_set_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3ControlV1,
    claims: tuple[Round3GlobalRetryClaimV1, ...],
    artifacts: tuple[BoundRound3FeedbackArtifactV1, ...],
) -> None:
    if len(claims) > ROUND3_GLOBAL_RETRY_CEILING:
        raise PortfolioS1FeedbackError("Round3 exceeds three global retry claims")
    first_by_sha = {
        item.artifact_sha256: item for item in artifacts if item.attempt_index == 1
    }
    prior: Round3GlobalRetryClaimV1 | None = None
    used_entries: set[str] = set()
    for ordinal, claim in enumerate(claims, 1):
        first = first_by_sha.get(claim.first_artifact_sha256)
        entry = selection.entries[claim.selection_ordinal - 1]
        eligible_unclaimed = sorted(
            (
                item
                for item in artifacts
                if item.attempt_index == 1
                and item.status != "parsed"
                and item.selection_entry_sha256 not in used_entries
                and is_round3_retry_eligible_v1(item.feedback_result)
            ),
            key=lambda item: item.global_call_ordinal,
        )
        if (
            type(claim) is not Round3GlobalRetryClaimV1
            or claim.claim_ordinal != ordinal
            or claim.previous_claim_sha256
            != (None if prior is None else prior.claim_sha256)
            or claim.selection_sha256 != selection.selection_sha256
            or claim.control_sha256 != control.control_sha256
            or claim.selection_entry_sha256 != entry.entry_sha256
            or claim.query_id != entry.query_id
            or claim.selection_entry_sha256 in used_entries
            or first is None
            or first.selection_entry_sha256 != claim.selection_entry_sha256
            or first.global_call_ordinal != claim.first_global_call_ordinal
            or first.feedback_result.result_sha256
            != claim.first_feedback_result_sha256
            or first.feedback_result.finish_reason != claim.trigger_finish_reason
            or first.feedback_result.raw_response_bytes
            != claim.trigger_raw_response_bytes
            or not is_round3_retry_eligible_v1(first.feedback_result)
            or not eligible_unclaimed
            or eligible_unclaimed[0] != first
        ):
            raise PortfolioS1FeedbackError("Round3 retry claim set drifted")
        used_entries.add(claim.selection_entry_sha256)
        prior = claim


def build_round3_retry_claim_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3ControlV1,
    first_artifact: BoundRound3FeedbackArtifactV1,
    *,
    existing_claims: tuple[Round3GlobalRetryClaimV1, ...],
    existing_artifacts: tuple[BoundRound3FeedbackArtifactV1, ...],
) -> Round3GlobalRetryClaimV1:
    validate_round3_claim_set_v1(selection, control, existing_claims, existing_artifacts)
    if (
        len(existing_claims) >= 3
        or first_artifact.attempt_index != 1
        or first_artifact not in existing_artifacts
        or not is_round3_retry_eligible_v1(first_artifact.feedback_result)
        or any(
            item.selection_entry_sha256 == first_artifact.selection_entry_sha256
            for item in existing_claims
        )
    ):
        raise PortfolioS1FeedbackError("Round3 retry claim is not eligible")
    eligible_unclaimed = sorted(
        (
            item
            for item in existing_artifacts
            if item.attempt_index == 1
            and item.status != "parsed"
            and is_round3_retry_eligible_v1(item.feedback_result)
            and not any(
                claim.selection_entry_sha256 == item.selection_entry_sha256
                for claim in existing_claims
            )
        ),
        key=lambda item: item.global_call_ordinal,
    )
    if not eligible_unclaimed or eligible_unclaimed[0] != first_artifact:
        raise PortfolioS1FeedbackError("Round3 claim skipped earlier eligible failure")
    entry = selection.entries[first_artifact.selection_ordinal - 1]
    draft = Round3GlobalRetryClaimV1.model_construct(
        control_sha256=control.control_sha256,
        claim_ordinal=len(existing_claims) + 1,
        previous_claim_sha256=(
            None if not existing_claims else existing_claims[-1].claim_sha256
        ),
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        first_global_call_ordinal=first_artifact.global_call_ordinal,
        first_artifact_sha256=first_artifact.artifact_sha256,
        first_feedback_result_sha256=first_artifact.feedback_result.result_sha256,
        trigger_finish_reason=first_artifact.feedback_result.finish_reason,
        trigger_raw_response_bytes=first_artifact.feedback_result.raw_response_bytes,
        claim_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"claim_sha256"})
    claim = Round3GlobalRetryClaimV1.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(unsigned)}, strict=True
    )
    validate_round3_claim_set_v1(
        selection, control, (*existing_claims, claim), existing_artifacts
    )
    return claim


def build_round3_reservation_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    source: VerifiedStaticFeedbackSourceV2,
    *,
    attempt_index: Literal[1, 2],
    global_call_ordinal: int,
    first_artifact: BoundRound3FeedbackArtifactV1 | None = None,
    retry_claim: Round3GlobalRetryClaimV1 | None = None,
) -> Round3FeedbackCallReservationV1:
    selection = predecessors.base.selection
    entry = next(
        item for item in selection.entries if item.entry_sha256 == source.selection_entry_sha256
    )
    require_verified_static_feedback_source_v2(
        source, selection, predecessors.base.control, entry
    )
    if control != build_round3_control_v1(
        predecessors, governance, authorization, remote_receipt
    ):
        raise PortfolioS1FeedbackError("Round3 reservation control drifted")
    if attempt_index == 1:
        if first_artifact is not None or retry_claim is not None:
            raise PortfolioS1FeedbackError("Round3 first attempt has retry ancestry")
    elif (
        first_artifact is None
        or retry_claim is None
        or first_artifact.attempt_index != 1
        or first_artifact.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.first_artifact_sha256 != first_artifact.artifact_sha256
        or retry_claim.first_feedback_result_sha256
        != first_artifact.feedback_result.result_sha256
        or global_call_ordinal <= first_artifact.global_call_ordinal
        or not is_round3_retry_eligible_v1(first_artifact.feedback_result)
    ):
        raise PortfolioS1FeedbackError("Round3 retry reservation ancestry drifted")
    if global_call_ordinal > ROUND3_PROVIDER_CALL_CEILING:
        raise PortfolioS1FeedbackError("Round3 provider call ceiling exceeded")
    draft = Round3FeedbackCallReservationV1.model_construct(
        source_control_sha256=predecessors.base.control.control_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        packet_sha256=source.packet.packet_sha256,
        checkpoint_file_sha256=source.row.checkpoint_file_sha256,
        checkpoint_row_sha256=source.row.checkpoint_row_sha256,
        sidecar_sha256=source.row.sidecar.evidence_sha256,
        attempt_index=attempt_index,
        global_call_ordinal=global_call_ordinal,
        previous_artifact_sha256=(
            None if first_artifact is None else first_artifact.artifact_sha256
        ),
        retry_claim_sha256=None if retry_claim is None else retry_claim.claim_sha256,
        retry_claim_ordinal=None if retry_claim is None else retry_claim.claim_ordinal,
        reservation_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"reservation_sha256"})
    return Round3FeedbackCallReservationV1.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(unsigned)}, strict=True
    )


def build_bound_round3_artifact_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    source: VerifiedStaticFeedbackSourceV2,
    result: RecoveryFeedbackEvaluationResultV1,
    *,
    reservation: Round3FeedbackCallReservationV1,
    first_artifact: BoundRound3FeedbackArtifactV1 | None = None,
    retry_claim: Round3GlobalRetryClaimV1 | None = None,
) -> BoundRound3FeedbackArtifactV1:
    expected = build_round3_reservation_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        source,
        attempt_index=reservation.attempt_index,
        global_call_ordinal=reservation.global_call_ordinal,
        first_artifact=first_artifact,
        retry_claim=retry_claim,
    )
    result = redact_recovery_result_for_creator_privacy_v1(
        result,
        private_query_ids=tuple(
            item.query_id for item in predecessors.base.selection.entries
        ),
    )
    if (
        reservation != expected
        or type(result) is not RecoveryFeedbackEvaluationResultV1
        or result.wire_kind != "round3_primary_json_object_v1"
        or result.query_id != reservation.query_id
        or result.packet_sha256 != source.packet.packet_sha256
        or result.image_sha256 != source.packet.image.sha256
        or result.remote_authorization_id != authorization.authorization_id
        or result.remote_authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or result.remote_receipt_file_sha256 != control.remote_receipt_file_sha256
        or result.remote_receipt_sha256 != control.remote_receipt_sha256
        or result.asset_catalog_sha256 != control.remote_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("Round3 provider result source drifted")
    draft = BoundRound3FeedbackArtifactV1.model_construct(
        source_control_sha256=predecessors.base.control.control_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=reservation.selection_ordinal,
        selection_entry_sha256=reservation.selection_entry_sha256,
        query_id=reservation.query_id,
        attempt_index=reservation.attempt_index,
        global_call_ordinal=reservation.global_call_ordinal,
        reservation_sha256=reservation.reservation_sha256,
        previous_artifact_sha256=reservation.previous_artifact_sha256,
        retry_claim_sha256=reservation.retry_claim_sha256,
        retry_claim_ordinal=reservation.retry_claim_ordinal,
        feedback_packet=source.packet,
        feedback_result=result,
        status=result.status,
        artifact_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"artifact_sha256"})
    return BoundRound3FeedbackArtifactV1.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(unsigned)}, strict=True
    )


def _write_model(path: str | Path, model: BaseModel) -> Path:
    return atomic_create_file(path, canonical_json_bytes(model.model_dump(mode="json")))


def _load_model(
    path: str | Path,
    *,
    model_type: type[BaseModel],
    label: str,
    expected_file_sha256: str | None = None,
    max_bytes: int = 128 * 1024 * 1024,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    if expected_file_sha256 is not None and sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1FeedbackError(f"{label} file hash mismatch")
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical")
    return model


def write_round3_predecessor_receipt_v1(
    path: str | Path, receipt: PortfolioS1FeedbackRound3PredecessorReceiptV1
) -> Path:
    return _write_model(path, receipt)


def write_round3_authorization_v1(
    path: str | Path, authorization: PortfolioS1FeedbackRound3AuthorizationV1
) -> Path:
    return _write_model(path, authorization)


def load_round3_authorization_v1(
    path: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
) -> PortfolioS1FeedbackRound3AuthorizationV1:
    model = _load_model(
        path,
        model_type=PortfolioS1FeedbackRound3AuthorizationV1,
        label="Round3 authorization",
        max_bytes=4 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3AuthorizationV1)
    expected = build_round3_authorization_v1(
        predecessors,
        governance,
        authorization_id=model.authorization_id,
        reviewer_id=model.reviewer_id,
        reviewed_at=model.reviewed_at,
    )
    if model != expected:
        raise PortfolioS1FeedbackError("Round3 authorization differs from authority")
    return model


def load_round3_remote_receipt_v1(
    path: str | Path,
) -> PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1:
    model = _load_model(
        path,
        model_type=PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
        label="Round3 remote receipt",
        max_bytes=8 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1)
    return model


def write_round3_control_v1(
    path: str | Path, control: PortfolioS1FeedbackRound3ControlV1
) -> Path:
    return _write_model(path, control)


def load_round3_control_v1(
    path: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
) -> PortfolioS1FeedbackRound3ControlV1:
    model = _load_model(
        path,
        model_type=PortfolioS1FeedbackRound3ControlV1,
        label="Round3 control",
        max_bytes=4 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3ControlV1)
    if model != build_round3_control_v1(
        predecessors, governance, authorization, remote_receipt
    ):
        raise PortfolioS1FeedbackError("Round3 control differs from authority")
    return model


def write_round3_launch_v1(
    path: str | Path, launch: PortfolioS1FeedbackRound3LaunchV1
) -> Path:
    return _write_model(path, launch)


def load_round3_launch_v1(
    path: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
) -> PortfolioS1FeedbackRound3LaunchV1:
    model = _load_model(
        path,
        model_type=PortfolioS1FeedbackRound3LaunchV1,
        label="Round3 launch",
        max_bytes=4 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3LaunchV1)
    if model != build_round3_launch_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        run_id=model.run_id,
    ):
        raise PortfolioS1FeedbackError("Round3 launch differs from authority")
    return model


def round3_attempt_filename_v1(
    global_call_ordinal: int, selection_entry_sha256: str, attempt_index: int
) -> str:
    if global_call_ordinal not in range(1, 244) or attempt_index not in {1, 2}:
        raise PortfolioS1FeedbackError("Round3 attempt filename identity is invalid")
    return (
        f"{global_call_ordinal:04d}-{selection_entry_sha256[:16]}-"
        f"attempt-{attempt_index}.json"
    )


def round3_claim_filename_v1(claim_ordinal: int) -> str:
    if claim_ordinal not in {1, 2, 3}:
        raise PortfolioS1FeedbackError("Round3 claim ordinal is invalid")
    return f"claim-{claim_ordinal:02d}.json"


def write_round3_claim_v1(path: str | Path, claim: Round3GlobalRetryClaimV1) -> Path:
    if Path(path).name != round3_claim_filename_v1(claim.claim_ordinal):
        raise PortfolioS1FeedbackError("Round3 claim filename drifted")
    return _write_model(path, claim)


def load_round3_claim_v1(path: str | Path) -> Round3GlobalRetryClaimV1:
    model = _load_model(
        path,
        model_type=Round3GlobalRetryClaimV1,
        label="Round3 retry claim",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3GlobalRetryClaimV1)
    if Path(path).name != round3_claim_filename_v1(model.claim_ordinal):
        raise PortfolioS1FeedbackError("Round3 claim filename drifted")
    return model


def write_round3_reservation_v1(
    path: str | Path, reservation: Round3FeedbackCallReservationV1
) -> Path:
    if Path(path).name != round3_attempt_filename_v1(
        reservation.global_call_ordinal,
        reservation.selection_entry_sha256,
        reservation.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 reservation filename drifted")
    return _write_model(path, reservation)


def load_round3_reservation_v1(path: str | Path) -> Round3FeedbackCallReservationV1:
    model = _load_model(
        path,
        model_type=Round3FeedbackCallReservationV1,
        label="Round3 reservation",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3FeedbackCallReservationV1)
    if Path(path).name != round3_attempt_filename_v1(
        model.global_call_ordinal, model.selection_entry_sha256, model.attempt_index
    ):
        raise PortfolioS1FeedbackError("Round3 reservation filename drifted")
    return model


def write_bound_round3_artifact_v1(
    path: str | Path, artifact: BoundRound3FeedbackArtifactV1
) -> Path:
    if Path(path).name != round3_attempt_filename_v1(
        artifact.global_call_ordinal,
        artifact.selection_entry_sha256,
        artifact.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 artifact filename drifted")
    return _write_model(path, artifact)


def load_bound_round3_artifact_v1(path: str | Path) -> BoundRound3FeedbackArtifactV1:
    model = _load_model(
        path,
        model_type=BoundRound3FeedbackArtifactV1,
        label="Round3 bound artifact",
        max_bytes=16 * 1024 * 1024,
    )
    assert isinstance(model, BoundRound3FeedbackArtifactV1)
    if Path(path).name != round3_attempt_filename_v1(
        model.global_call_ordinal, model.selection_entry_sha256, model.attempt_index
    ):
        raise PortfolioS1FeedbackError("Round3 artifact filename drifted")
    return model


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3LedgerV1:
    claims: tuple[Round3GlobalRetryClaimV1, ...]
    reservations: tuple[Round3FeedbackCallReservationV1, ...]
    artifacts: tuple[BoundRound3FeedbackArtifactV1, ...]
    orphaned_reservations: tuple[Round3FeedbackCallReservationV1, ...]
    pending_claims: tuple[Round3GlobalRetryClaimV1, ...]


def _safe_inventory(directory: Path, *, label: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} directory is unsafe")
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(not item.is_file() or item.is_symlink() or item.suffix != ".json" for item in paths):
        raise PortfolioS1FeedbackError(f"{label} inventory is unsafe")
    return paths


def validate_round3_ledger_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> None:
    selection = predecessors.base.selection
    if (
        len(sources) != 240
        or tuple(item.selection_entry_sha256 for item in sources)
        != tuple(item.entry_sha256 for item in selection.entries)
    ):
        raise PortfolioS1FeedbackError("Round3 source set differs from selected240")
    source_by_entry: dict[str, VerifiedStaticFeedbackSourceV2] = {}
    for entry, source in zip(selection.entries, sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, selection, predecessors.base.control, entry
        )
        source_by_entry[entry.entry_sha256] = source
    if (
        control
        != build_round3_control_v1(
            predecessors, governance, authorization, remote_receipt
        )
        or tuple(item.global_call_ordinal for item in ledger.reservations)
        != tuple(range(1, len(ledger.reservations) + 1))
        or len(ledger.reservations) > ROUND3_PROVIDER_CALL_CEILING
        or len({item.reservation_sha256 for item in ledger.reservations})
        != len(ledger.reservations)
        or len({item.artifact_sha256 for item in ledger.artifacts})
        != len(ledger.artifacts)
    ):
        raise PortfolioS1FeedbackError("Round3 ledger root identity drifted")
    validate_round3_claim_set_v1(selection, control, ledger.claims, ledger.artifacts)
    reservation_by_sha = {item.reservation_sha256: item for item in ledger.reservations}
    artifact_by_reservation: dict[str, BoundRound3FeedbackArtifactV1] = {}
    artifact_by_key: dict[tuple[str, int], BoundRound3FeedbackArtifactV1] = {}
    for artifact in ledger.artifacts:
        reservation = reservation_by_sha.get(artifact.reservation_sha256)
        key = (artifact.selection_entry_sha256, artifact.attempt_index)
        if (
            reservation is None
            or artifact.reservation_sha256 in artifact_by_reservation
            or key in artifact_by_key
            or artifact.selection_ordinal != reservation.selection_ordinal
            or artifact.selection_entry_sha256 != reservation.selection_entry_sha256
            or artifact.global_call_ordinal != reservation.global_call_ordinal
            or artifact.control_sha256 != control.control_sha256
            or artifact.authorization_sha256 != authorization.authorization_sha256
        ):
            raise PortfolioS1FeedbackError("Round3 artifact/reservation binding drifted")
        artifact_by_reservation[artifact.reservation_sha256] = artifact
        artifact_by_key[key] = artifact
    expected_orphans = tuple(
        item
        for item in ledger.reservations
        if item.reservation_sha256 not in artifact_by_reservation
    )
    if ledger.orphaned_reservations != expected_orphans or len(expected_orphans) > 2:
        raise PortfolioS1FeedbackError("Round3 orphan ledger drifted")
    if expected_orphans:
        final_reservation = ledger.reservations[-1]
        allowed_tail = (final_reservation,)
        if len(ledger.reservations) >= 2:
            previous = ledger.reservations[-2]
            if (
                previous.attempt_index == 1
                and final_reservation.attempt_index == 1
                and previous.selection_ordinal + 1
                == final_reservation.selection_ordinal
            ):
                allowed_tail = (previous, final_reservation)
        if any(item not in allowed_tail for item in expected_orphans):
            raise PortfolioS1FeedbackError(
                "Round3 orphans are outside the final reserved wave"
            )
    first_reservations = tuple(item for item in ledger.reservations if item.attempt_index == 1)
    if tuple(item.selection_ordinal for item in first_reservations) != tuple(
        range(1, len(first_reservations) + 1)
    ):
        raise PortfolioS1FeedbackError("Round3 first attempts changed fixed order")
    claim_by_sha = {item.claim_sha256: item for item in ledger.claims}
    first_artifact_by_entry = {
        item.selection_entry_sha256: item
        for item in ledger.artifacts
        if item.attempt_index == 1
    }
    used_claims: list[str] = []
    for reservation in ledger.reservations:
        entry = selection.entries[reservation.selection_ordinal - 1]
        verified_source = source_by_entry.get(reservation.selection_entry_sha256)
        first_artifact = (
            first_artifact_by_entry.get(reservation.selection_entry_sha256)
            if reservation.attempt_index == 2
            else None
        )
        claim = (
            claim_by_sha.get(reservation.retry_claim_sha256 or "")
            if reservation.attempt_index == 2
            else None
        )
        if verified_source is None:
            raise PortfolioS1FeedbackError("Round3 reservation source is unknown")
        if reservation.attempt_index == 2:
            if (
                claim is None
                or claim.selection_entry_sha256 != entry.entry_sha256
                or first_artifact is None
                or reservation.previous_artifact_sha256
                != first_artifact.artifact_sha256
                or reservation.retry_claim_ordinal != claim.claim_ordinal
            ):
                raise PortfolioS1FeedbackError("Round3 retry reservation claim drifted")
            used_claims.append(claim.claim_sha256)
        expected_reservation = build_round3_reservation_v1(
            predecessors,
            governance,
            authorization,
            control,
            remote_receipt,
            verified_source,
            attempt_index=reservation.attempt_index,
            global_call_ordinal=reservation.global_call_ordinal,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if reservation != expected_reservation:
            raise PortfolioS1FeedbackError(
                "Round3 reservation differs from verified source"
            )
    for artifact in ledger.artifacts:
        reservation = reservation_by_sha[artifact.reservation_sha256]
        verified_source = source_by_entry[artifact.selection_entry_sha256]
        first_artifact = (
            first_artifact_by_entry.get(artifact.selection_entry_sha256)
            if artifact.attempt_index == 2
            else None
        )
        claim = (
            claim_by_sha.get(artifact.retry_claim_sha256 or "")
            if artifact.attempt_index == 2
            else None
        )
        expected_artifact = build_bound_round3_artifact_v1(
            predecessors,
            governance,
            authorization,
            control,
            remote_receipt,
            verified_source,
            artifact.feedback_result,
            reservation=reservation,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if artifact != expected_artifact:
            raise PortfolioS1FeedbackError(
                "Round3 artifact differs from verified source/result"
            )
    claim_sha256s = [item.claim_sha256 for item in ledger.claims]
    if used_claims != claim_sha256s[: len(used_claims)]:
        raise PortfolioS1FeedbackError("Round3 retry claims were consumed out of order")
    expected_pending = ledger.claims[len(used_claims) :]
    if ledger.pending_claims != expected_pending or len(expected_pending) > 1:
        raise PortfolioS1FeedbackError("Round3 pending claim drifted")


def load_round3_ledger_v1(
    output_root: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> PortfolioS1FeedbackRound3LedgerV1:
    root = Path(output_root)
    claims = tuple(
        load_round3_claim_v1(path)
        for path in _safe_inventory(root / "global-retry-claims-round3-v1", label="Round3 claims")
    )
    reservations = tuple(
        sorted(
            (
                load_round3_reservation_v1(path)
                for path in _safe_inventory(
                    root / "provider-attempts-round3-v1", label="Round3 reservations"
                )
            ),
            key=lambda item: item.global_call_ordinal,
        )
    )
    artifacts = tuple(
        sorted(
            (
                load_bound_round3_artifact_v1(path)
                for path in _safe_inventory(
                    root / "bound-feedback-round3-v1", label="Round3 artifacts"
                )
            ),
            key=lambda item: item.global_call_ordinal,
        )
    )
    settled = {item.reservation_sha256 for item in artifacts}
    used_claim_count = sum(item.attempt_index == 2 for item in reservations)
    ledger = PortfolioS1FeedbackRound3LedgerV1(
        claims=claims,
        reservations=reservations,
        artifacts=artifacts,
        orphaned_reservations=tuple(
            item for item in reservations if item.reservation_sha256 not in settled
        ),
        pending_claims=claims[used_claim_count:],
    )
    validate_round3_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    return ledger


class Round3NextStepV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal[
        "create_claim",
        "reserve_retry",
        "reserve_primary_wave",
        "phase_complete",
        "finalize_complete",
        "terminal_nonparsed",
        "terminal_orphan",
        "terminal_budget",
    ]
    selection_ordinals: tuple[int, ...] = ()
    attempt_index: Literal[1, 2] | None = None
    claim_ordinal: Literal[1, 2, 3] | None = None
    parsed_count: int = Field(ge=0, le=240)
    provider_calls_reserved: int = Field(ge=0, le=243)
    terminal_error_code: str | None = None

    @field_validator("selection_ordinals", mode="before")
    @classmethod
    def _ordinals(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def next_round3_step_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    phase_count: Literal[12, 60, 120, 240],
) -> Round3NextStepV1:
    validate_round3_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    if phase_count not in ROUND3_PHASE_COUNTS:
        raise PortfolioS1FeedbackError("Round3 phase boundary is invalid")
    artifacts_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item for item in ledger.artifacts
    }
    final_by_ordinal: dict[int, BoundRound3FeedbackArtifactV1] = {}
    for artifact in ledger.artifacts:
        prior = final_by_ordinal.get(artifact.selection_ordinal)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_ordinal[artifact.selection_ordinal] = artifact
    parsed_count = sum(item.status == "parsed" for item in final_by_ordinal.values())
    common = {
        "parsed_count": parsed_count,
        "provider_calls_reserved": len(ledger.reservations),
    }
    if ledger.orphaned_reservations:
        return Round3NextStepV1(
            kind="terminal_orphan",
            selection_ordinals=tuple(
                item.selection_ordinal for item in ledger.orphaned_reservations
            ),
            terminal_error_code="orphan",
            **common,
        )
    retry_failures = sorted(
        (
            item
            for item in ledger.artifacts
            if item.attempt_index == 2 and item.status != "parsed"
        ),
        key=lambda item: item.global_call_ordinal,
    )
    if retry_failures:
        failed = retry_failures[0]
        return Round3NextStepV1(
            kind="terminal_nonparsed",
            selection_ordinals=(failed.selection_ordinal,),
            attempt_index=2,
            terminal_error_code=failed.feedback_result.error_code,
            **common,
        )
    first_failures = sorted(
        (
            item
            for item in ledger.artifacts
            if item.attempt_index == 1
            and item.status != "parsed"
            and (item.selection_entry_sha256, 2) not in artifacts_by_key
        ),
        key=lambda item: item.global_call_ordinal,
    )
    if first_failures:
        noneligible = tuple(
            item
            for item in first_failures
            if not is_round3_retry_eligible_v1(item.feedback_result)
        )
        if noneligible:
            failed = noneligible[0]
            return Round3NextStepV1(
                kind="terminal_nonparsed",
                selection_ordinals=(failed.selection_ordinal,),
                attempt_index=1,
                terminal_error_code=failed.feedback_result.error_code,
                **common,
            )
        claimed_entries = {item.selection_entry_sha256 for item in ledger.claims}
        unclaimed = tuple(
            item
            for item in first_failures
            if item.selection_entry_sha256 not in claimed_entries
        )
        if len(unclaimed) > ROUND3_GLOBAL_RETRY_CEILING - len(ledger.claims):
            return Round3NextStepV1(
                kind="terminal_nonparsed",
                selection_ordinals=tuple(item.selection_ordinal for item in unclaimed),
                attempt_index=1,
                terminal_error_code="global_retry_ceiling_exceeded",
                **common,
            )
        first = first_failures[0]
        claim = next(
            (
                item
                for item in ledger.claims
                if item.selection_entry_sha256 == first.selection_entry_sha256
            ),
            None,
        )
        if claim is None:
            if len(ledger.reservations) >= ROUND3_PROVIDER_CALL_CEILING:
                return Round3NextStepV1(
                    kind="terminal_budget",
                    selection_ordinals=(first.selection_ordinal,),
                    attempt_index=2,
                    terminal_error_code="provider_call_ceiling_exceeded",
                    **common,
                )
            return Round3NextStepV1(
                kind="create_claim",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=len(ledger.claims) + 1,  # type: ignore[arg-type]
                **common,
            )
        if len(ledger.reservations) >= ROUND3_PROVIDER_CALL_CEILING:
            return Round3NextStepV1(
                kind="terminal_budget",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=claim.claim_ordinal,
                terminal_error_code="provider_call_ceiling_exceeded",
                **common,
            )
        return Round3NextStepV1(
            kind="reserve_retry",
            selection_ordinals=(first.selection_ordinal,),
            attempt_index=2,
            claim_ordinal=claim.claim_ordinal,
            **common,
        )
    final_phase_ordinals = tuple(
        ordinal
        for ordinal in range(1, phase_count + 1)
        if ordinal in final_by_ordinal and final_by_ordinal[ordinal].status == "parsed"
    )
    if len(final_phase_ordinals) == phase_count:
        return Round3NextStepV1(
            kind="finalize_complete" if phase_count == 240 else "phase_complete",
            **common,
        )
    pending = tuple(
        ordinal for ordinal in range(1, phase_count + 1) if ordinal not in final_by_ordinal
    )
    if not pending:
        raise PortfolioS1FeedbackError("Round3 phase has unresolved nonparsed output")
    wave = pending[:2]
    if len(ledger.reservations) + len(wave) > ROUND3_PROVIDER_CALL_CEILING:
        return Round3NextStepV1(
            kind="terminal_budget",
            selection_ordinals=wave,
            attempt_index=1,
            terminal_error_code="provider_call_ceiling_exceeded",
            **common,
        )
    return Round3NextStepV1(
        kind="reserve_primary_wave",
        selection_ordinals=wave,
        attempt_index=1,
        **common,
    )


class Round3RunAttemptV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=243)
    reservation_sha256: Sha256
    artifact_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    error_code: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    raw_response_bytes: int | None = Field(default=None, ge=0)
    tool_call_count: int | None = Field(default=None, ge=0)
    refusal_present: bool | None = None
    response_redaction_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    actual_cost_cny: str | None = None

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        usage_known = self.input_tokens is not None or self.output_tokens is not None
        if usage_known:
            if self.input_tokens is None or self.output_tokens is None:
                raise ValueError("Round3 run attempt has partial usage")
            expected_cost = (
                Decimal(self.input_tokens) * Decimal(12)
                + Decimal(self.output_tokens) * Decimal(36)
            ) / Decimal(1_000_000)
            expected_cost_text = format(
                expected_cost.quantize(Decimal("0.000000000001")), "f"
            )
            if self.actual_cost_cny != expected_cost_text:
                raise ValueError("Round3 run attempt cost differs from exact usage")
        elif self.actual_cost_cny is not None:
            raise ValueError("Round3 run attempt cost exists without usage")
        if self.status == "orphan":
            if self.artifact_sha256 is not None or any(
                item is not None
                for item in (
                    self.request_id,
                    self.finish_reason,
                    self.raw_response_bytes,
                    self.tool_call_count,
                    self.refusal_present,
                    self.input_tokens,
                    self.output_tokens,
                    self.actual_cost_cny,
                )
            ):
                raise ValueError("Round3 run orphan contains response data")
        elif self.artifact_sha256 is None:
            raise ValueError("Round3 settled run attempt lacks artifact")
        if self.status == "parsed" and self.error_code is not None:
            raise ValueError("Round3 parsed attempt claims error")
        if self.status not in {"parsed", "orphan"} and self.error_code is None:
            raise ValueError("Round3 failed attempt lacks error code")
        if (self.attempt_index == 2) != (self.retry_claim_ordinal is not None):
            raise ValueError("Round3 run retry claim identity drifted")
        return self


class PortfolioS1FeedbackRound3RunV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-run"] = (
        "portfolio-s1-feedback-round3-run"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-run-v1"] = (
        ROUND3_RUN_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    predecessor_receipt_sha256: Sha256
    authorization_sha256: Sha256
    control_sha256: Sha256
    expected_count: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = ROUND3_PHASE_COUNTS
    attempted_count: int = Field(ge=1, le=240)
    parsed_count: int = Field(ge=0, le=240)
    error_count: int = Field(ge=0, le=240)
    orphan_count: int = Field(ge=0, le=2)
    provider_calls_reserved: int = Field(ge=1, le=243)
    retry_count: int = Field(ge=0, le=3)
    retry_claim_sha256s: tuple[Sha256, ...]
    terminal_phase_count: Literal[12, 60, 120, 240]
    status: Literal[
        "stopped_nonparsed", "stopped_orphan", "stopped_budget", "completed"
    ]
    terminal_reason: (
        Literal[
            "usage_limit_exceeded",
            "accountable_cost_exceeded",
            "provider_call_ceiling_exceeded",
        ]
        | None
    ) = None
    historical_feedback_outputs_imported: Literal[0] = 0
    usage_known_count: int = Field(ge=0, le=243)
    usage_unknown_count: int = Field(ge=0, le=243)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY
    new_actual_cost_cny: str
    cumulative_actual_cost_cny: str
    cumulative_accountable_cost_cny: str
    artifacts: tuple[Round3RunAttemptV1, ...]
    artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator(
        "phase_counts", "retry_claim_sha256s", "artifacts", mode="before"
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        final = {item.selection_entry_sha256: item for item in self.artifacts}
        statuses = Counter(item.status for item in final.values())
        new_actual = sum(
            (Decimal(item.actual_cost_cny) for item in self.artifacts if item.actual_cost_cny),
            Decimal("0"),
        )
        cumulative_actual = Decimal(ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY) + new_actual
        accountable = cumulative_actual + Decimal(ROUND3_PER_CALL_RESERVATION_CNY) * sum(
            item.input_tokens is None for item in self.artifacts
        )
        accountable_breach = accountable > Decimal(
            ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
        )
        usage_breach = any(
            item.input_tokens is not None
            and (
                item.input_tokens > 20_000
                or (item.output_tokens or 0) > 6_154
            )
            for item in self.artifacts
        )

        def money(value: Decimal) -> str:
            return format(value.quantize(Decimal("0.000000000001")), "f")

        if (
            self.phase_counts != FEEDBACK_PHASE_COUNTS_V2
            or tuple(item.global_call_ordinal for item in self.artifacts)
            != tuple(range(1, len(self.artifacts) + 1))
            or self.provider_calls_reserved != len(self.artifacts)
            or self.attempted_count != len(final)
            or self.parsed_count != statuses["parsed"]
            or self.error_count != self.attempted_count - self.parsed_count
            or self.orphan_count != sum(item.status == "orphan" for item in self.artifacts)
            or self.retry_count != sum(item.attempt_index == 2 for item in self.artifacts)
            or self.retry_count != len(self.retry_claim_sha256s)
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count
            != self.provider_calls_reserved - self.usage_known_count
            or self.input_tokens != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens != sum(item.output_tokens or 0 for item in self.artifacts)
            or self.new_actual_cost_cny != money(new_actual)
            or self.cumulative_actual_cost_cny != money(cumulative_actual)
            or self.cumulative_accountable_cost_cny != money(accountable)
            or self.artifact_set_sha256
            != _hash_payload([item.model_dump(mode="json") for item in self.artifacts])
        ):
            raise ValueError("Round3 run counts, costs, or hashes drifted")
        if self.status == "completed":
            if (
                self.terminal_reason is not None
                or accountable_breach
                or usage_breach
                or self.attempted_count != 240
                or self.parsed_count != 240
                or self.orphan_count
                or tuple(sorted(item.selection_ordinal for item in final.values()))
                != tuple(range(1, 241))
            ):
                raise ValueError("completed Round3 run is not fresh parsed240")
        elif self.status == "stopped_orphan":
            if not self.orphan_count or self.terminal_reason not in {
                None,
                "usage_limit_exceeded",
                "accountable_cost_exceeded",
                "provider_call_ceiling_exceeded",
            }:
                raise ValueError("Round3 orphan status drifted")
        elif self.status == "stopped_budget":
            if self.terminal_reason is None or self.orphan_count:
                raise ValueError("Round3 budget status drifted")
        elif (
            self.error_count < 1
            or self.orphan_count
            or self.terminal_reason is not None
            or accountable_breach
            or usage_breach
        ):
            raise ValueError("Round3 nonparsed status drifted")
        if self.status in {"stopped_budget", "stopped_orphan"}:
            if self.terminal_reason == "usage_limit_exceeded" and not usage_breach:
                raise ValueError("Round3 usage-limit reason lacks evidence")
            if (
                self.terminal_reason == "accountable_cost_exceeded"
                and not accountable_breach
            ):
                raise ValueError("Round3 accountable-cost reason lacks evidence")
            if (
                self.terminal_reason == "provider_call_ceiling_exceeded"
                and self.provider_calls_reserved != ROUND3_PROVIDER_CALL_CEILING
            ):
                raise ValueError("Round3 call-ceiling reason lacks evidence")
            if usage_breach and self.terminal_reason != "usage_limit_exceeded":
                raise ValueError("Round3 usage-limit evidence was not reported")
            if (
                accountable_breach
                and not usage_breach
                and self.terminal_reason != "accountable_cost_exceeded"
            ):
                raise ValueError("Round3 accountable-cost evidence was not reported")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Round3 run hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _actual_cost(usage: LLMUsage | None) -> str | None:
    if usage is None:
        return None
    value = (
        Decimal(usage.input_tokens) * Decimal(12)
        + Decimal(usage.output_tokens) * Decimal(36)
    ) / Decimal(1_000_000)
    return format(value.quantize(Decimal("0.000000000001")), "f")


def build_round3_run_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    terminal_reason: (
        Literal[
            "usage_limit_exceeded",
            "accountable_cost_exceeded",
            "provider_call_ceiling_exceeded",
        ]
        | None
    ) = None,
) -> PortfolioS1FeedbackRound3RunV1:
    validate_round3_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    artifact_by_reservation = {
        item.reservation_sha256: item for item in ledger.artifacts
    }
    rows: list[Round3RunAttemptV1] = []
    for reservation in ledger.reservations:
        artifact = artifact_by_reservation.get(reservation.reservation_sha256)
        if artifact is None:
            rows.append(
                Round3RunAttemptV1(
                    selection_ordinal=reservation.selection_ordinal,
                    selection_entry_sha256=reservation.selection_entry_sha256,
                    attempt_index=reservation.attempt_index,
                    global_call_ordinal=reservation.global_call_ordinal,
                    reservation_sha256=reservation.reservation_sha256,
                    retry_claim_ordinal=reservation.retry_claim_ordinal,
                    status="orphan",
                )
            )
            continue
        result = artifact.feedback_result
        usage = result.usage
        rows.append(
            Round3RunAttemptV1(
                selection_ordinal=artifact.selection_ordinal,
                selection_entry_sha256=artifact.selection_entry_sha256,
                attempt_index=artifact.attempt_index,
                global_call_ordinal=artifact.global_call_ordinal,
                reservation_sha256=artifact.reservation_sha256,
                artifact_sha256=artifact.artifact_sha256,
                retry_claim_ordinal=artifact.retry_claim_ordinal,
                status=artifact.status,
                error_code=result.error_code,
                request_id=result.request_id,
                finish_reason=result.finish_reason,
                raw_response_bytes=result.raw_response_bytes,
                tool_call_count=result.tool_call_count,
                refusal_present=result.refusal_present,
                response_redaction_reason=result.response_redaction_reason,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                reasoning_tokens=result.reasoning_tokens,
                reasoning_bytes=result.reasoning_bytes,
                actual_cost_cny=_actual_cost(usage),
            )
        )
    if not rows:
        raise PortfolioS1FeedbackError("Round3 run has no provider attempts")
    final = {item.selection_entry_sha256: item for item in rows}
    parsed = sum(item.status == "parsed" for item in final.values())
    usage_limit = any(
        (item.input_tokens or 0) > 20_000 or (item.output_tokens or 0) > 6_154
        for item in rows
        if item.input_tokens is not None
    )
    new_actual = sum(
        (Decimal(item.actual_cost_cny) for item in rows if item.actual_cost_cny),
        Decimal("0"),
    )
    cumulative_actual = Decimal(ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY) + new_actual
    unknown = sum(item.input_tokens is None for item in rows)
    accountable = cumulative_actual + Decimal(ROUND3_PER_CALL_RESERVATION_CNY) * unknown
    if usage_limit:
        terminal_reason = "usage_limit_exceeded"
    elif accountable > Decimal(ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY):
        terminal_reason = "accountable_cost_exceeded"
    if ledger.orphaned_reservations:
        status = "stopped_orphan"
    elif terminal_reason is not None:
        status = "stopped_budget"
    elif len(final) == 240 and parsed == 240:
        status = "completed"
    elif any(item.status != "parsed" for item in final.values()):
        status = "stopped_nonparsed"
    else:
        raise PortfolioS1FeedbackError("Round3 nonterminal prefix cannot publish a run")
    max_selection = max(item.selection_ordinal for item in rows)
    terminal_phase = next(item for item in ROUND3_PHASE_COUNTS if max_selection <= item)

    def money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.000000000001")), "f")

    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-run",
        "policy_version": ROUND3_RUN_POLICY_VERSION_V1,
        "selection_sha256": predecessors.base.selection.selection_sha256,
        "predecessor_receipt_sha256": predecessors.receipt.receipt_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "control_sha256": control.control_sha256,
        "expected_count": 240,
        "phase_counts": ROUND3_PHASE_COUNTS,
        "attempted_count": len(final),
        "parsed_count": parsed,
        "error_count": len(final) - parsed,
        "orphan_count": len(ledger.orphaned_reservations),
        "provider_calls_reserved": len(rows),
        "retry_count": len(ledger.claims),
        "retry_claim_sha256s": tuple(item.claim_sha256 for item in ledger.claims),
        "terminal_phase_count": terminal_phase,
        "status": status,
        "terminal_reason": terminal_reason,
        "historical_feedback_outputs_imported": 0,
        "usage_known_count": len(rows) - unknown,
        "usage_unknown_count": unknown,
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "prior_cumulative_actual_cost_cny": ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY,
        "new_actual_cost_cny": money(new_actual),
        "cumulative_actual_cost_cny": money(cumulative_actual),
        "cumulative_accountable_cost_cny": money(accountable),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRound3RunV1.model_validate(
        {**unsigned, "run_sha256": _hash_payload(unsigned)}, strict=True
    )


class Round3BundleProvenanceV10(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=240)
    origin: Literal["round3_fresh_qwen_json_object"] = (
        "round3_fresh_qwen_json_object"
    )
    bound_artifact_sha256: Sha256
    feedback_result_sha256: Sha256
    final_attempt_index: Literal[1, 2]
    final_global_call_ordinal: int = Field(ge=1, le=243)
    historical_feedback_output_imported: Literal[False] = False


class PortfolioS1FeedbackBundleV10(PortfolioS1FeedbackBundleV5):
    schema_version: Literal[10] = 10
    policy_version: Literal["portfolio-s1-feedback-bundle-v10"] = (
        ROUND3_BUNDLE_POLICY_VERSION_V10
    )
    provider_call_count: Literal[240, 241, 242, 243]
    predecessor_receipt_sha256: Sha256
    predecessor_receipt_file_sha256: Sha256
    round3_run_sha256: Sha256
    round3_run_file_sha256: Sha256
    round3_artifact_set_sha256: Sha256
    round3_authorization_sha256: Sha256
    round3_control_sha256: Sha256
    transport_policy_sha256: Literal[ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1] = (
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    )
    retry_claim_count: Literal[0, 1, 2, 3]
    retry_claim_sha256s: tuple[Sha256, ...]
    fresh_output_count: Literal[240] = 240
    historical_feedback_outputs_imported: Literal[0] = 0
    entry_provenance: tuple[Round3BundleProvenanceV10, ...]

    @field_validator("retry_claim_sha256s", "entry_provenance", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_round3(self) -> Self:
        if (
            self.run_sha256 != self.round3_run_sha256
            or self.run_file_sha256 != self.round3_run_file_sha256
            or self.authorization_sha256 != self.round3_authorization_sha256
            or self.control_sha256 != self.round3_control_sha256
            or self.retry_claim_count != len(self.retry_claim_sha256s)
            or len(self.entry_provenance) != 240
            or tuple(item.selection_ordinal for item in self.entry_provenance)
            != tuple(range(1, 241))
        ):
            raise ValueError("BundleV10 Round3 lineage drifted")
        for entry, provenance in zip(self.entries, self.entry_provenance, strict=True):
            if (
                entry.bound_artifact_sha256 != provenance.bound_artifact_sha256
                or entry.feedback_result_sha256 != provenance.feedback_result_sha256
            ):
                raise ValueError("BundleV10 provenance hashes drifted")
        return self


def build_portfolio_s1_feedback_bundle_v10(
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV1,
    control: PortfolioS1FeedbackRound3ControlV1,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    run: PortfolioS1FeedbackRound3RunV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> PortfolioS1FeedbackBundleV10:
    validate_round3_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    if run != build_round3_run_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    ):
        raise PortfolioS1FeedbackError("BundleV10 run differs from exact ledger")
    if run.status != "completed":
        raise PortfolioS1FeedbackError("BundleV10 requires fresh parsed240")
    final_by_ordinal: dict[int, BoundRound3FeedbackArtifactV1] = {}
    for artifact in ledger.artifacts:
        prior = final_by_ordinal.get(artifact.selection_ordinal)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_ordinal[artifact.selection_ordinal] = artifact
    if set(final_by_ordinal) != set(range(1, 241)) or any(
        item.status != "parsed" for item in final_by_ordinal.values()
    ):
        raise PortfolioS1FeedbackError("BundleV10 final output set is not parsed240")

    selection = predecessors.base.selection
    selected_query_ids = tuple(item.query_id for item in selection.entries)
    private_entries: list[PortfolioS1FeedbackBundleEntryV5] = []
    model_entries: list[PortfolioS1FeedbackModelEntryV5] = []
    representatives: list[PortfolioS1FeedbackRepresentativeExampleV5] = []
    contracts: dict[str, FeedbackGCSContractProjectionV5] = {}
    provenance: list[Round3BundleProvenanceV10] = []
    artifact_hashes: list[dict[str, object]] = []
    for selected in selection.entries:
        artifact = final_by_ordinal[selected.selection_ordinal]
        packet = artifact.feedback_packet
        result = artifact.feedback_result
        feedback = result.parsed_feedback
        if (
            type(packet) is not FeedbackPacketV3
            or type(feedback) is not VisualFeedbackOutput
            or artifact.status != "parsed"
            or artifact.selection_sha256 != selection.selection_sha256
            or artifact.source_control_sha256
            != predecessors.base.control.control_sha256
            or artifact.control_sha256 != control.control_sha256
            or artifact.query_id != selected.query_id
            or result.model != "qwen3.8-max"
            or result.wire_kind != "round3_primary_json_object_v1"
            or result.packet_sha256 != packet.packet_sha256
            or packet.query_id != selected.query_id
            or packet.gcs_diagnostics.answer_mode != selected.answer_mode
            or packet.gcs_diagnostics.reason_codes != selected.reason_codes
        ):
            raise PortfolioS1FeedbackError("BundleV10 artifact binding drifted")
        assert feedback is not None
        _validate_model_projection_privacy(
            feedback.model_dump(mode="json"), private_query_ids=selected_query_ids
        )
        if any(
            _CREATOR_RUNTIME_HANDLE_RE.search(text)
            for text in _iter_string_values(feedback.model_dump(mode="json"))
        ):
            raise PortfolioS1FeedbackError("BundleV10 contains runtime metadata")
        labeled = tuple(
            parse_policy_labeled_feedback_suggestion_v1(item)
            for item in feedback.skill_suggestions
        )
        if len(set((item.disposition, item.text) for item in labeled)) != len(labeled):
            raise PortfolioS1FeedbackError("BundleV10 suggestions repeat")
        diagnostic = VisualFeedbackOutput(
            schema_version=1,
            summary=feedback.summary,
            rule_violations=feedback.rule_violations,
            ideal_response_gaps=feedback.ideal_response_gaps,
            skill_suggestions=(),
        )
        model_entries.append(
            PortfolioS1FeedbackModelEntryV5(
                selection_ordinal=selected.selection_ordinal,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                answer_mode=selected.answer_mode,
                gcs=selected.gcs,
                gcs_components=packet.gcs_diagnostics.components,
                reason_codes=selected.reason_codes,
                diagnostic_feedback=diagnostic,
                labeled_suggestions=labeled,
                actionable_suggestions=tuple(
                    item.text for item in labeled if item.disposition == "policy_compatible"
                ),
            )
        )
        packet_contract = packet.gcs_contract
        projected = FeedbackGCSContractProjectionV5(
            capability=selected.capability,
            required_sections=packet_contract.required_sections,
            fallback_markers=packet_contract.fallback_markers,
            preferred_fallback_marker=packet_contract.preferred_fallback_marker,
            card_requirement=packet_contract.card_requirement,
            legal_tool_sequences=packet_contract.legal_tool_sequences,
        )
        if contracts.setdefault(selected.capability, projected) != projected:
            raise PortfolioS1FeedbackError("BundleV10 GCS contract drifted")
        private_entries.append(
            PortfolioS1FeedbackBundleEntryV5(
                selection_ordinal=selected.selection_ordinal,
                query_id=selected.query_id,
                capability=selected.capability,
                role=selected.role,
                primary_cluster=selected.primary_cluster,
                atomic_component_id=selected.atomic_component_id,
                packet_sha256=packet.packet_sha256,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=result.result_sha256,
            )
        )
        artifact_hashes.append(
            {
                "selection_ordinal": selected.selection_ordinal,
                "bound_artifact_sha256": artifact.artifact_sha256,
                "feedback_result_sha256": result.result_sha256,
            }
        )
        provenance.append(
            Round3BundleProvenanceV10(
                selection_ordinal=selected.selection_ordinal,
                bound_artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=result.result_sha256,
                final_attempt_index=artifact.attempt_index,
                final_global_call_ordinal=artifact.global_call_ordinal,
            )
        )
        if selected.selection_ordinal <= 12:
            representatives.append(
                PortfolioS1FeedbackRepresentativeExampleV5(
                    selection_ordinal=selected.selection_ordinal,
                    capability=selected.capability,
                    role=selected.role,
                    primary_cluster=selected.primary_cluster,
                    gcs_diagnostics=packet.gcs_diagnostics,
                    turns=packet.turns,
                    response_text=packet.response_text,
                    cards=packet.cards,
                    tool_evidence=packet.tool_evidence,
                )
            )
    projection = PortfolioS1FeedbackModelProjectionV5(
        coverage=_feedback_v5_coverage(selection),
        gcs_contract=PortfolioS1FeedbackGCSContractSetV5(
            capability_contracts=tuple(contracts[item] for item in GCS_CAPABILITY_ORDER)
        ),
        feedback_entries=tuple(model_entries),
        representative_examples=tuple(representatives),
        actionable_suggestions=tuple(
            PortfolioS1FeedbackActionableSuggestionV5(
                selection_ordinal=item.selection_ordinal,
                capability=item.capability,
                text=text,
            )
            for item in model_entries
            for text in item.actionable_suggestions
        ),
        actionable_suggestion_count=sum(
            len(item.actionable_suggestions) for item in model_entries
        ),
    )
    _validate_model_projection_privacy(projection, private_query_ids=selected_query_ids)
    run_file_sha256 = sha256_bytes(run.canonical_bytes())
    unsigned = {
        "schema_version": 10,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": ROUND3_BUNDLE_POLICY_VERSION_V10,
        "status": "complete_policy_filtered_feedback",
        "selection_sha256": selection.selection_sha256,
        "selection_file_sha256": PARENT_SELECTION_FILE_SHA256,
        "control_sha256": control.control_sha256,
        "control_file_sha256": sha256_bytes(control.canonical_bytes()),
        "run_sha256": run.run_sha256,
        "run_file_sha256": run_file_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "authorization_file_sha256": sha256_bytes(authorization.canonical_bytes()),
        "corpus_sha256": selection.corpus_sha256,
        "parent_static_bank_sha256": selection.parent_static_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 240,
        "attempted_count": 240,
        "parsed_count": 240,
        "provider_call_count": run.provider_calls_reserved,
        "selected_query_ids": selected_query_ids,
        "entries": tuple(private_entries),
        "bound_artifact_set_sha256": _hash_payload(artifact_hashes),
        "model_projection": projection,
        "predecessor_receipt_sha256": predecessors.receipt.receipt_sha256,
        "predecessor_receipt_file_sha256": sha256_bytes(
            predecessors.receipt.canonical_bytes()
        ),
        "round3_run_sha256": run.run_sha256,
        "round3_run_file_sha256": run_file_sha256,
        "round3_artifact_set_sha256": run.artifact_set_sha256,
        "round3_authorization_sha256": authorization.authorization_sha256,
        "round3_control_sha256": control.control_sha256,
        "transport_policy_sha256": ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
        "retry_claim_count": len(ledger.claims),
        "retry_claim_sha256s": tuple(item.claim_sha256 for item in ledger.claims),
        "fresh_output_count": 240,
        "historical_feedback_outputs_imported": 0,
        "entry_provenance": tuple(provenance),
    }
    return PortfolioS1FeedbackBundleV10.model_validate(
        {**unsigned, "bundle_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_round3_run_v1(path: str | Path, run: PortfolioS1FeedbackRound3RunV1) -> Path:
    return _write_model(path, run)


def load_round3_run_v1(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> PortfolioS1FeedbackRound3RunV1:
    return _load_model(
        path,
        model_type=PortfolioS1FeedbackRound3RunV1,
        label="Round3 run",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


def write_portfolio_s1_feedback_bundle_v10(
    path: str | Path, bundle: PortfolioS1FeedbackBundleV10
) -> Path:
    return _write_model(path, bundle)


def load_portfolio_s1_feedback_bundle_v10(
    path: str | Path, *, expected_file_sha256: str
) -> PortfolioS1FeedbackBundleV10:
    return _load_model(
        path,
        model_type=PortfolioS1FeedbackBundleV10,
        label="Portfolio S1 Feedback BundleV10",
        expected_file_sha256=expected_file_sha256,
    )  # type: ignore[return-value]


__all__ = [
    "ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY",
    "ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY",
    "ROUND3_PER_CALL_RESERVATION_CNY",
    "ROUND3_PHASE_COUNTS",
    "BoundRound3FeedbackArtifactV1",
    "PortfolioS1FeedbackBundleV10",
    "PortfolioS1FeedbackRound3AuthorizationV1",
    "PortfolioS1FeedbackRound3ControlV1",
    "PortfolioS1FeedbackRound3LaunchV1",
    "PortfolioS1FeedbackRound3LedgerV1",
    "PortfolioS1FeedbackRound3PredecessorReceiptV1",
    "PortfolioS1FeedbackRound3RunV1",
    "PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1",
    "Round3FeedbackCallReservationV1",
    "Round3GlobalRetryClaimV1",
    "Round3NextStepV1",
    "VerifiedPortfolioS1FeedbackRound3GovernanceV1",
    "VerifiedPortfolioS1FeedbackRound3PredecessorsV1",
    "build_bound_round3_artifact_v1",
    "build_portfolio_s1_feedback_bundle_v10",
    "build_round3_authorization_v1",
    "build_round3_control_v1",
    "build_round3_launch_v1",
    "build_round3_reservation_v1",
    "build_round3_retry_claim_v1",
    "build_round3_run_v1",
    "is_round3_retry_eligible_v1",
    "load_bound_round3_artifact_v1",
    "load_portfolio_s1_feedback_bundle_v10",
    "load_round3_ledger_v1",
    "load_round3_authorization_v1",
    "load_round3_control_v1",
    "load_round3_launch_v1",
    "load_round3_remote_receipt_v1",
    "load_round3_reservation_v1",
    "load_round3_run_v1",
    "load_verified_round3_governance_v1",
    "load_verified_round3_predecessors_v1",
    "next_round3_step_v1",
    "prepare_round3_remote_runtime_v1",
    "round3_attempt_filename_v1",
    "round3_claim_filename_v1",
    "write_bound_round3_artifact_v1",
    "write_portfolio_s1_feedback_bundle_v10",
    "write_round3_authorization_v1",
    "write_round3_claim_v1",
    "write_round3_control_v1",
    "write_round3_launch_v1",
    "write_round3_predecessor_receipt_v1",
    "write_round3_reservation_v1",
    "write_round3_run_v1",
]
