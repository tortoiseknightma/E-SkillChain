"""Forward-only strict-schema overlay for the stopped Round3 object canary.

This module reuses the frozen SelectionV2, source membership, phase ordering,
and failure semantics from :mod:`portfolio_s1_feedback_round3_v1`.  It imports
no Feedback output from the terminated JSON-object canary; that root is bound
only as immutable lineage and cumulative-cost evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    _require_policy_labeled_suggestions,
)
from skillchain.evaluation.packets import FeedbackPacketV3
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    _hash_payload,
    _model_hash,
    _nested_json_model,
    require_verified_static_feedback_source_v2,
)

from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    PARENT_SELECTION_FILE_SHA256,
    PARENT_SELECTION_SHA256,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
    RecoveryFeedbackEvaluationResultV2,
    load_parent_selected_feedback_remote_runtime_v1,
    redact_recovery_result_for_creator_privacy_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    SelectedFeedbackRemoteBindingV2,
    SelectedFeedbackRemoteRuntimeError,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_v1 import (
    PortfolioS1FeedbackRound3AuthorizationV1,
    PortfolioS1FeedbackRound3ControlV1,
    PortfolioS1FeedbackRound3LaunchV1,
    PortfolioS1FeedbackRound3RunV1,
    PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    Round3FeedbackCallReservationV1,
    Round3GlobalRetryClaimV1,
    Round3RunAttemptV1,
    VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    load_round3_run_v1,
    load_verified_round3_predecessors_v1,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7,
    QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14,
    QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4,
    QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4,
    Qwen38FeedbackModelSourceLockV4,
    Qwen38FeedbackPricingLockV7,
    Qwen38FeedbackRoleSelectionV14,
    load_qwen38_feedback_model_source_lock_v4,
    load_qwen38_feedback_pricing_lock_v7,
    load_qwen38_feedback_role_selection_v14,
    require_qwen38_feedback_pre_call_budget_v4,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import VerifiedStaticGCSCorpus
from skillchain.evaluation.visual_runtime import (
    SelectedFeedbackImageBinding,
    VerifiedSelectedFeedbackRemoteRuntime,
    _make_verified_selected_feedback_remote_runtime,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


ROUND3_SCHEMA_PHASE_COUNTS = (12, 60, 120, 240)
ROUND3_SCHEMA_PROVIDER_CALL_CEILING = 243
ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING = 15
ROUND3_SCHEMA_GLOBAL_RETRY_CEILING = 3
ROUND3_SCHEMA_LIVE_PHASE_CEILING: Literal[12] = 12
ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY = "24.319500000000"
ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY = "0.461544000000"
ROUND3_SCHEMA_LIVE_MAXIMUM_RESERVATION_CNY = "6.923160000000"
ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY = "10.000000000000"
ROUND3_SCHEMA_FUTURE_MAXIMUM_RESERVATION_CNY = "112.155192000000"
ROUND3_SCHEMA_FUTURE_CUMULATIVE_MAXIMUM_CNY = "136.474692000000"
ROUND3_SCHEMA_FUTURE_TECHNICAL_HARD_CAP_CNY = "137.000000000000"

ROUND3_SCHEMA_STOPPED_OBJECT_RUN_FILE_SHA256 = (
    "5f6673045f5a307e3fe311bb0f722ff86822a976664021b77c22b6c5e856d94d"
)
ROUND3_SCHEMA_STOPPED_OBJECT_RUN_SHA256 = (
    "1da0e7bd42b9af090e6151b402fb45670fa7a81f81104ab75145ccb301110d2a"
)
ROUND3_SCHEMA_STOPPED_OBJECT_ARTIFACT_SET_SHA256 = (
    "914dec9a53ee31287ebf01bd6f916156341aa0ab015a3580bf0d5865f685c86d"
)
ROUND3_SCHEMA_STOPPED_OBJECT_TOP_INVENTORY_SHA256 = (
    "963152c9d459dee4cff198da999b966de3ff56ea2d97f9aea7349819b6e75c34"
)
ROUND3_SCHEMA_STOPPED_OBJECT_CLAIM_INVENTORY_SHA256 = (
    "c36630b89e04acbe18a3de4594ef99e182f6d20b747d25cb0a5b429ee7f6f8a4"
)
ROUND3_SCHEMA_STOPPED_OBJECT_RESERVATION_INVENTORY_SHA256 = (
    "cfb2b9d4e13c51b7debc01b6a7ae8694dc0513e9352d40123773b3e72092c184"
)
ROUND3_SCHEMA_STOPPED_OBJECT_BOUND_INVENTORY_SHA256 = (
    "b52601fe1d96d1c1ed02d454008dc93ace2549fa8ad002ea4a757e303194826b"
)

ROUND3_SCHEMA_PREDECESSOR_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-predecessors-v2"
)
ROUND3_SCHEMA_AUTHORIZATION_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-authorization-v2"
)
ROUND3_SCHEMA_REMOTE_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-remote-runtime-v2"
)
ROUND3_SCHEMA_CONTROL_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-control-v2"
)
ROUND3_SCHEMA_LAUNCH_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-launch-v2"
)
ROUND3_SCHEMA_RESERVATION_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-reservation-v2"
)
ROUND3_SCHEMA_BOUND_POLICY_VERSION_V2 = (
    "portfolio-s1-bound-feedback-round3-schema-v1"
)
ROUND3_SCHEMA_RUN_POLICY_VERSION_V2 = "portfolio-s1-feedback-round3-run-v2"
ROUND3_SCHEMA_BUNDLE_POLICY_VERSION_V11 = "portfolio-s1-feedback-bundle-v11"

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Round3SchemaStatus = Literal["parsed", "parse_error", "provider_error", "timeout"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

ROUND3_SCHEMA_RETRY_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-global-retry-v2"
)


def round3_schema_retry_policy_v2() -> dict[str, object]:
    """Return the unchanged retry semantics under the new schema wire."""

    return {
        "policy_version": ROUND3_SCHEMA_RETRY_POLICY_VERSION_V2,
        "scope": "fresh-fixed-selection240-same-order",
        "provider_call_ceiling": 243,
        "live_canary_provider_call_ceiling": 15,
        "normal_calls": 240,
        "live_canary_normal_calls": 12,
        "global_retry_ceiling": 3,
        "max_lifetime_attempts_per_entry": 2,
        "attempt_transport_policy_version": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
        ),
        "attempt_transport_policy_sha256": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        ),
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
        "phase60_requires_new_owner_approval": True,
    }


ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(round3_schema_retry_policy_v2())
)


class PortfolioS1FeedbackRound3SchemaPredecessorReceiptV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-round3-predecessors"] = (
        "portfolio-s1-feedback-round3-predecessors"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-predecessors-v2"
    ] = ROUND3_SCHEMA_PREDECESSOR_POLICY_VERSION_V2
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    prior_predecessor_receipt_sha256: Sha256
    stopped_object_authorization_file_sha256: Sha256
    stopped_object_control_file_sha256: Sha256
    stopped_object_launch_file_sha256: Sha256
    stopped_object_run_file_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_RUN_FILE_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_RUN_FILE_SHA256
    stopped_object_run_sha256: Literal[ROUND3_SCHEMA_STOPPED_OBJECT_RUN_SHA256] = (
        ROUND3_SCHEMA_STOPPED_OBJECT_RUN_SHA256
    )
    stopped_object_artifact_set_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_ARTIFACT_SET_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_ARTIFACT_SET_SHA256
    stopped_object_top_inventory_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_TOP_INVENTORY_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_TOP_INVENTORY_SHA256
    stopped_object_claim_inventory_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_CLAIM_INVENTORY_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_CLAIM_INVENTORY_SHA256
    stopped_object_reservation_inventory_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_RESERVATION_INVENTORY_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_RESERVATION_INVENTORY_SHA256
    stopped_object_bound_inventory_sha256: Literal[
        ROUND3_SCHEMA_STOPPED_OBJECT_BOUND_INVENTORY_SHA256
    ] = ROUND3_SCHEMA_STOPPED_OBJECT_BOUND_INVENTORY_SHA256
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    selected_count: Literal[240] = 240
    historical_feedback_outputs_imported: Literal[0] = 0
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    historical_roots_read_only: Literal[True] = True
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.receipt_sha256 != _model_hash(self, "receipt_sha256"):
            raise ValueError("Round3 schema predecessor receipt hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1:
    prior: VerifiedPortfolioS1FeedbackRound3PredecessorsV1
    stopped_object_root: Path
    stopped_object_authorization: PortfolioS1FeedbackRound3AuthorizationV1
    stopped_object_control: PortfolioS1FeedbackRound3ControlV1
    stopped_object_launch: PortfolioS1FeedbackRound3LaunchV1
    stopped_object_run: PortfolioS1FeedbackRound3RunV1
    receipt: PortfolioS1FeedbackRound3SchemaPredecessorReceiptV2


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1:
    repository_root: Path
    source_lock: Qwen38FeedbackModelSourceLockV4
    pricing_lock: Qwen38FeedbackPricingLockV7
    role_selection: Qwen38FeedbackRoleSelectionV14
    _marker: object | None = field(default=None, repr=False, compare=False)


_VERIFIED_SCHEMA_GOVERNANCE_TOKEN = object()


def _inventory_sha256(directory: Path, *, label: str) -> str:
    members = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in members
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory drifted")
    return _hash_payload(
        [
            {
                "name": item.name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        item,
                        label=f"{label} {item.name}",
                        max_bytes=16 * 1024 * 1024,
                    )
                ),
            }
            for item in members
        ]
    )


def load_verified_round3_schema_governance_v1(
    repository_root: str | Path,
) -> VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1:
    root = Path(repository_root).resolve(strict=True)
    source = load_qwen38_feedback_model_source_lock_v4(
        root / QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4,
        expected_file_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4,
    )
    pricing = load_qwen38_feedback_pricing_lock_v7(
        root / QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7,
        expected_file_sha256=QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7,
    )
    role = load_qwen38_feedback_role_selection_v14(
        root / QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14,
        expected_file_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14,
    )
    if (
        type(source) is not Qwen38FeedbackModelSourceLockV4
        or type(pricing) is not Qwen38FeedbackPricingLockV7
        or type(role) is not Qwen38FeedbackRoleSelectionV14
        or source.source_lock_sha256 != QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
        or pricing.pricing_lock_sha256 != QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
        or role.selection_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14
        or source.requested_response_format != "json_schema"
        or source.requested_json_schema_sha256
        != VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
        or not source.requested_json_schema_strict
        or pricing.outer_orchestration_policy_sha256
        != ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2
        or pricing.provider_call_ceiling
        != ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING
        or role.feedback_evaluator.get("requested_response_format") != "json_schema"
        or role.feedback_evaluator.get("phase60_requires_new_owner_approval")
        is not True
    ):
        raise PortfolioS1FeedbackError("Round3 schema governance triad drifted")
    return VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1(
        repository_root=root,
        source_lock=source,
        pricing_lock=pricing,
        role_selection=role,
        _marker=_VERIFIED_SCHEMA_GOVERNANCE_TOKEN,
    )


def load_verified_round3_schema_predecessors_v1(
    base_parent_root: str | Path,
    recovery_root: str | Path,
    stopped_object_root: str | Path,
) -> VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1:
    """Deep-bind the stopped object root without importing any of its outputs."""

    prior = load_verified_round3_predecessors_v1(base_parent_root, recovery_root)
    supplied = Path(stopped_object_root)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackError("stopped Round3 object root cannot be a symlink")
    root = supplied.resolve(strict=True)
    expected_files = {
        "authorization-round3-v1.json",
        "control-round3-v1.json",
        "launch-round3-v1.json",
        "predecessors-round3-v1.json",
        "remote-runtime-receipt-round3-v1.json",
        "run-round3-v1.json",
        "selection-v2.json",
    }
    expected_dirs = {
        "bound-feedback-round3-v1",
        "global-retry-claims-round3-v1",
        "provider-attempts-round3-v1",
    }
    children = tuple(root.iterdir())
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in children if item.is_file()} != expected_files
        or {item.name for item in children if item.is_dir()} != expected_dirs
        or any(item.is_symlink() for item in children)
    ):
        raise PortfolioS1FeedbackError("stopped Round3 object inventory drifted")
    authorization = _load_schema_model(
        root / "authorization-round3-v1.json",
        model_type=PortfolioS1FeedbackRound3AuthorizationV1,
        label="stopped Round3 object authorization",
    )
    control = _load_schema_model(
        root / "control-round3-v1.json",
        model_type=PortfolioS1FeedbackRound3ControlV1,
        label="stopped Round3 object control",
    )
    launch = _load_schema_model(
        root / "launch-round3-v1.json",
        model_type=PortfolioS1FeedbackRound3LaunchV1,
        label="stopped Round3 object launch",
    )
    assert isinstance(authorization, PortfolioS1FeedbackRound3AuthorizationV1)
    assert isinstance(control, PortfolioS1FeedbackRound3ControlV1)
    assert isinstance(launch, PortfolioS1FeedbackRound3LaunchV1)
    run = load_round3_run_v1(
        root / "run-round3-v1.json",
        expected_file_sha256=ROUND3_SCHEMA_STOPPED_OBJECT_RUN_FILE_SHA256,
    )
    if (
        run.status != "stopped_nonparsed"
        or run.parsed_count != 10
        or run.attempted_count != 12
        or run.provider_calls_reserved != 14
        or run.retry_count != 2
        or run.orphan_count != 0
        or run.run_sha256 != ROUND3_SCHEMA_STOPPED_OBJECT_RUN_SHA256
        or run.artifact_set_sha256
        != ROUND3_SCHEMA_STOPPED_OBJECT_ARTIFACT_SET_SHA256
        or run.cumulative_actual_cost_cny
        != ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ):
        raise PortfolioS1FeedbackError("stopped Round3 object terminal facts drifted")
    top_inventory = _hash_payload(
        [
            {
                "name": name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        root / name,
                        label=f"stopped Round3 object {name}",
                        max_bytes=128 * 1024 * 1024,
                    )
                ),
            }
            for name in sorted(expected_files)
        ]
    )
    claim_inventory = _inventory_sha256(
        root / "global-retry-claims-round3-v1", label="stopped Round3 claims"
    )
    reservation_inventory = _inventory_sha256(
        root / "provider-attempts-round3-v1",
        label="stopped Round3 reservations",
    )
    bound_inventory = _inventory_sha256(
        root / "bound-feedback-round3-v1", label="stopped Round3 artifacts"
    )
    if (
        top_inventory != ROUND3_SCHEMA_STOPPED_OBJECT_TOP_INVENTORY_SHA256
        or claim_inventory
        != ROUND3_SCHEMA_STOPPED_OBJECT_CLAIM_INVENTORY_SHA256
        or reservation_inventory
        != ROUND3_SCHEMA_STOPPED_OBJECT_RESERVATION_INVENTORY_SHA256
        or bound_inventory != ROUND3_SCHEMA_STOPPED_OBJECT_BOUND_INVENTORY_SHA256
    ):
        raise PortfolioS1FeedbackError("stopped Round3 object bytes drifted")
    draft = PortfolioS1FeedbackRound3SchemaPredecessorReceiptV2.model_construct(
        prior_predecessor_receipt_sha256=prior.receipt.receipt_sha256,
        stopped_object_authorization_file_sha256=sha256_bytes(
            authorization.canonical_bytes()
        ),
        stopped_object_control_file_sha256=sha256_bytes(control.canonical_bytes()),
        stopped_object_launch_file_sha256=sha256_bytes(launch.canonical_bytes()),
        stopped_object_top_inventory_sha256=top_inventory,
        stopped_object_claim_inventory_sha256=claim_inventory,
        stopped_object_reservation_inventory_sha256=reservation_inventory,
        stopped_object_bound_inventory_sha256=bound_inventory,
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1FeedbackRound3SchemaPredecessorReceiptV2.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )
    return VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1(
        prior=prior,
        stopped_object_root=root,
        stopped_object_authorization=authorization,
        stopped_object_control=control,
        stopped_object_launch=launch,
        stopped_object_run=run,
        receipt=receipt,
    )


class PortfolioS1FeedbackRound3AuthorizationV2(
    PortfolioS1FeedbackRound3AuthorizationV1
):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-authorization-v2"
    ] = ROUND3_SCHEMA_AUTHORIZATION_POLICY_VERSION_V2
    authorized_selection_ordinals: tuple[int, ...] = tuple(range(1, 13))
    live_phase_ceiling: Literal[12] = 12
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1] = (
        VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
    )
    requested_json_schema_strict: Literal[True] = True
    provider_call_ceiling: Literal[15] = 15
    parent_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    new_maximum_reservation_cny: Literal[
        ROUND3_SCHEMA_LIVE_MAXIMUM_RESERVATION_CNY
    ] = ROUND3_SCHEMA_LIVE_MAXIMUM_RESERVATION_CNY
    cumulative_maximum_reservation_cny: Literal["31.242660000000"] = (
        "31.242660000000"
    )
    cumulative_technical_hard_cap_cny: Literal["34.319500000000"] = (
        "34.319500000000"
    )
    owner_budget_ceiling_cny: Literal[ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY] = (
        ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY
    )
    future_provider_call_ceiling: Literal[243] = 243
    future_cumulative_maximum_reservation_cny: Literal[
        ROUND3_SCHEMA_FUTURE_CUMULATIVE_MAXIMUM_CNY
    ] = ROUND3_SCHEMA_FUTURE_CUMULATIVE_MAXIMUM_CNY
    future_cumulative_technical_hard_cap_cny: Literal[
        ROUND3_SCHEMA_FUTURE_TECHNICAL_HARD_CAP_CNY
    ] = ROUND3_SCHEMA_FUTURE_TECHNICAL_HARD_CAP_CNY
    future_full_run_authorization_status: Literal["not_granted"] = "not_granted"
    phase60_requires_new_owner_approval: Literal[True] = True
    model_source_lock_file: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
    )
    pricing_lock_file: Literal[QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7] = (
        QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
    )
    role_selection_file: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14
    )
    terminated_json_object_canary_outputs_imported: Literal[0] = 0

    @field_validator("authorized_selection_ordinals", mode="before")
    @classmethod
    def _authorized_ordinals_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if (
            self.authorized_selection_ordinals != tuple(range(1, 13))
            or len(self.selected_entry_sha256s) != 240
            or len(set(self.selected_entry_sha256s)) != 240
            or len(self.selected_query_ids) != 240
            or len(set(self.selected_query_ids)) != 240
            or self.phase_counts != ROUND3_SCHEMA_PHASE_COUNTS
            or Decimal(self.parent_cumulative_actual_cost_cny)
            + Decimal(self.new_maximum_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.new_maximum_reservation_cny)
            >= Decimal(self.owner_budget_ceiling_cny)
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or self.future_full_run_authorization_status != "not_granted"
            or self.authorization_sha256 != _model_hash(self, "authorization_sha256")
        ):
            raise ValueError("Round3 schema authorization drifted")
        return self


def build_round3_schema_authorization_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: str,
) -> PortfolioS1FeedbackRound3AuthorizationV2:
    if (
        type(governance) is not VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1
        or governance._marker is not _VERIFIED_SCHEMA_GOVERNANCE_TOKEN
    ):
        raise PortfolioS1FeedbackError("Round3 schema governance is not verified")
    parent = predecessors.prior.base
    selection = parent.selection
    draft = PortfolioS1FeedbackRound3AuthorizationV2.model_construct(
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
    return PortfolioS1FeedbackRound3AuthorizationV2.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2(
    PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1
):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-remote-runtime-v2"
    ] = ROUND3_SCHEMA_REMOTE_POLICY_VERSION_V2
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14
    )
    live_transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    live_authorized_selection_ordinals: tuple[int, ...] = tuple(range(1, 13))

    @field_validator("live_authorized_selection_ordinals", mode="before")
    @classmethod
    def _live_ordinals_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if (
            self.live_authorized_selection_ordinals != tuple(range(1, 13))
            or tuple(item.selection_ordinal for item in self.selected_bindings)
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
            raise ValueError("Round3 schema remote receipt drifted")
        return self


def prepare_round3_schema_remote_runtime_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    corpus: VerifiedStaticGCSCorpus,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    receipt_path: str | Path,
    verified_sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedSelectedFeedbackRemoteRuntime:
    """Create/resume schema authority while reusing only image membership."""

    if authorization != build_round3_schema_authorization_v2(
        predecessors,
        governance,
        authorization_id=authorization.authorization_id,
        reviewer_id=authorization.reviewer_id,
        reviewed_at=authorization.reviewed_at,
    ):
        raise SelectedFeedbackRemoteRuntimeError("Round3 schema authorization drifted")
    parent = predecessors.prior.base
    selection = parent.selection
    if (
        corpus.corpus_sha256 != selection.corpus_sha256
        or len(verified_sources) != 240
        or tuple(item.selection_entry_sha256 for item in verified_sources)
        != tuple(item.entry_sha256 for item in selection.entries)
    ):
        raise SelectedFeedbackRemoteRuntimeError(
            "Round3 schema sources differ from fixed selected240"
        )
    membership_runtime = load_parent_selected_feedback_remote_runtime_v1(
        parent,
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
            "Round3 schema membership ancestry drifted"
        )
    receipt_bindings: list[SelectedFeedbackRemoteBindingV2] = []
    image_bindings: list[SelectedFeedbackImageBinding] = []
    for entry, source in zip(selection.entries, verified_sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, selection, parent.control, entry
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
                "Round3 schema remote image membership drifted"
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
    draft = PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2.model_construct(
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
    expected = PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2.model_validate(
        {**unsigned, "receipt_sha256": _hash_payload(unsigned)}, strict=True
    )
    target = Path(receipt_path)
    if target.exists():
        content = read_stable_regular_file(
            target, label="Round3 schema remote receipt", max_bytes=8 * 1024 * 1024
        )
        receipt = PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2.model_validate_json(
            content, strict=True
        )
        if receipt.canonical_bytes() != content or receipt != expected:
            raise SelectedFeedbackRemoteRuntimeError(
                "Round3 schema remote receipt resume conflict"
            )
    else:
        if target.is_symlink():
            raise SelectedFeedbackRemoteRuntimeError(
                "Round3 schema remote receipt target is a symlink"
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


class PortfolioS1FeedbackRound3ControlV2(PortfolioS1FeedbackRound3ControlV1):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-control-v2"
    ] = ROUND3_SCHEMA_CONTROL_POLICY_VERSION_V2
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14
    )
    retry_policy_sha256: Literal[ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2] = (
        ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1] = (
        VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
    )
    requested_json_schema_strict: Literal[True] = True
    cumulative_maximum_reservation_cny: Literal["31.242660000000"] = (
        "31.242660000000"
    )
    cumulative_technical_hard_cap_cny: Literal["34.319500000000"] = (
        "34.319500000000"
    )
    live_phase_ceiling: Literal[12] = 12
    provider_call_ceiling: Literal[15] = 15
    phase60_requires_new_owner_approval: Literal[True] = True
    terminated_json_object_canary_outputs_imported: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if (
            self.phase_counts != ROUND3_SCHEMA_PHASE_COUNTS
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or self.provider_call_ceiling != 15
            or self.control_sha256 != _model_hash(self, "control_sha256")
        ):
            raise ValueError("Round3 schema control drifted")
        return self


def build_round3_schema_control_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
) -> PortfolioS1FeedbackRound3ControlV2:
    expected_authorization = build_round3_schema_authorization_v2(
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
        != predecessors.prior.base.selection.selection_sha256
        or remote_receipt.membership_authorization_id
        != authorization.membership_authorization_id
        or remote_receipt.membership_receipt_sha256
        != authorization.membership_receipt_sha256
        or remote_receipt.catalog_sha256 != authorization.membership_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("Round3 schema authorization drifted")
    draft = PortfolioS1FeedbackRound3ControlV2.model_construct(
        source_control_sha256=predecessors.prior.base.control.control_sha256,
        source_control_file_sha256=sha256_bytes(
            predecessors.prior.base.control.canonical_bytes()
        ),
        predecessor_receipt_sha256=predecessors.receipt.receipt_sha256,
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        remote_receipt_file_sha256=sha256_bytes(remote_receipt.canonical_bytes()),
        remote_receipt_sha256=remote_receipt.receipt_sha256,
        remote_catalog_sha256=remote_receipt.catalog_sha256,
        control_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"control_sha256"})
    return PortfolioS1FeedbackRound3ControlV2.model_validate(
        {**unsigned, "control_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1FeedbackRound3LaunchV2(PortfolioS1FeedbackRound3LaunchV1):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-launch-v2"
    ] = ROUND3_SCHEMA_LAUNCH_POLICY_VERSION_V2
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
    )
    pricing_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    ] = QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14
    )
    provider_call_ceiling: Literal[15] = 15
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1] = (
        VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
    )
    requested_json_schema_strict: Literal[True] = True
    cumulative_maximum_reservation_cny: Literal["31.242660000000"] = (
        "31.242660000000"
    )
    cumulative_technical_hard_cap_cny: Literal["34.319500000000"] = (
        "34.319500000000"
    )
    live_phase_ceiling: Literal[12] = 12
    phase60_requires_new_owner_approval: Literal[True] = True

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if (
            Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.cumulative_technical_hard_cap_cny)
            or self.provider_call_ceiling != 15
            or self.launch_sha256 != _model_hash(self, "launch_sha256")
        ):
            raise ValueError("Round3 schema launch drifted")
        return self


def build_round3_schema_launch_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    *,
    run_id: str,
) -> PortfolioS1FeedbackRound3LaunchV2:
    if control != build_round3_schema_control_v2(
        predecessors, governance, authorization, remote_receipt
    ):
        raise PortfolioS1FeedbackError("Round3 schema launch control drifted")
    draft = PortfolioS1FeedbackRound3LaunchV2.model_construct(
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
    return PortfolioS1FeedbackRound3LaunchV2.model_validate(
        {**unsigned, "launch_sha256": _hash_payload(unsigned)}, strict=True
    )


def is_round3_schema_retry_eligible_v1(
    result: RecoveryFeedbackEvaluationResultV2,
) -> bool:
    if (
        type(result) is not RecoveryFeedbackEvaluationResultV2
        or result.wire_kind != "round3_primary_json_schema_v1"
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
        return False
    return False


class Round3SchemaGlobalRetryClaimV2(Round3GlobalRetryClaimV1):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-global-retry-v2"
    ] = ROUND3_SCHEMA_RETRY_POLICY_VERSION_V2
    policy_sha256: Literal[ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2] = (
        ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2
    )


class Round3SchemaFeedbackCallReservationV2(Round3FeedbackCallReservationV1):
    schema_version: Literal[2] = 2
    policy_version: Literal[
        "portfolio-s1-feedback-round3-reservation-v2"
    ] = ROUND3_SCHEMA_RESERVATION_POLICY_VERSION_V2
    global_call_ordinal: int = Field(ge=1, le=15)
    wire_kind: Literal["round3_primary_json_schema_v1"] = (
        "round3_primary_json_schema_v1"
    )


class BoundRound3SchemaFeedbackArtifactV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-bound-feedback-round3"] = (
        "portfolio-s1-bound-feedback-round3"
    )
    policy_version: Literal[
        "portfolio-s1-bound-feedback-round3-schema-v1"
    ] = ROUND3_SCHEMA_BOUND_POLICY_VERSION_V2
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    source_control_sha256: Sha256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=1, le=240)
    selection_entry_sha256: Sha256
    query_id: str
    attempt_index: Literal[1, 2]
    global_call_ordinal: int = Field(ge=1, le=15)
    reservation_sha256: Sha256
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: Literal[1, 2, 3] | None = None
    feedback_packet: FeedbackPacketV3
    feedback_result: RecoveryFeedbackEvaluationResultV2
    status: Round3SchemaStatus
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, RecoveryFeedbackEvaluationResultV2)

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
            or self.feedback_result.wire_kind != "round3_primary_json_schema_v1"
            or self.feedback_result.status != self.status
            or (self.attempt_index == 1 and any(item is not None for item in ancestry))
            or (self.attempt_index == 2 and any(item is None for item in ancestry))
            or self.artifact_sha256 != _model_hash(self, "artifact_sha256")
        ):
            raise ValueError("Round3 schema bound artifact drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_round3_schema_claim_set_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    claims: tuple[Round3SchemaGlobalRetryClaimV2, ...],
    artifacts: tuple[BoundRound3SchemaFeedbackArtifactV2, ...],
) -> None:
    if len(claims) > ROUND3_SCHEMA_GLOBAL_RETRY_CEILING:
        raise PortfolioS1FeedbackError("Round3 schema exceeds three retry claims")
    first_by_sha = {
        item.artifact_sha256: item for item in artifacts if item.attempt_index == 1
    }
    prior: Round3SchemaGlobalRetryClaimV2 | None = None
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
                and is_round3_schema_retry_eligible_v1(item.feedback_result)
            ),
            key=lambda item: item.global_call_ordinal,
        )
        if (
            type(claim) is not Round3SchemaGlobalRetryClaimV2
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
            or not is_round3_schema_retry_eligible_v1(first.feedback_result)
            or not eligible_unclaimed
            or eligible_unclaimed[0] != first
        ):
            raise PortfolioS1FeedbackError("Round3 schema retry claim set drifted")
        used_entries.add(claim.selection_entry_sha256)
        prior = claim


def build_round3_schema_retry_claim_v2(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    first_artifact: BoundRound3SchemaFeedbackArtifactV2,
    *,
    existing_claims: tuple[Round3SchemaGlobalRetryClaimV2, ...],
    existing_artifacts: tuple[BoundRound3SchemaFeedbackArtifactV2, ...],
) -> Round3SchemaGlobalRetryClaimV2:
    validate_round3_schema_claim_set_v1(
        selection, control, existing_claims, existing_artifacts
    )
    if (
        len(existing_claims) >= 3
        or first_artifact.attempt_index != 1
        or first_artifact not in existing_artifacts
        or not is_round3_schema_retry_eligible_v1(first_artifact.feedback_result)
        or any(
            item.selection_entry_sha256 == first_artifact.selection_entry_sha256
            for item in existing_claims
        )
    ):
        raise PortfolioS1FeedbackError("Round3 schema retry claim is not eligible")
    eligible_unclaimed = sorted(
        (
            item
            for item in existing_artifacts
            if item.attempt_index == 1
            and item.status != "parsed"
            and is_round3_schema_retry_eligible_v1(item.feedback_result)
            and not any(
                claim.selection_entry_sha256 == item.selection_entry_sha256
                for claim in existing_claims
            )
        ),
        key=lambda item: item.global_call_ordinal,
    )
    if not eligible_unclaimed or eligible_unclaimed[0] != first_artifact:
        raise PortfolioS1FeedbackError("Round3 schema skipped an eligible failure")
    entry = selection.entries[first_artifact.selection_ordinal - 1]
    draft = Round3SchemaGlobalRetryClaimV2.model_construct(
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
    claim = Round3SchemaGlobalRetryClaimV2.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(unsigned)}, strict=True
    )
    validate_round3_schema_claim_set_v1(
        selection, control, (*existing_claims, claim), existing_artifacts
    )
    return claim


def build_round3_schema_reservation_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    source: VerifiedStaticFeedbackSourceV2,
    *,
    attempt_index: Literal[1, 2],
    global_call_ordinal: int,
    first_artifact: BoundRound3SchemaFeedbackArtifactV2 | None = None,
    retry_claim: Round3SchemaGlobalRetryClaimV2 | None = None,
) -> Round3SchemaFeedbackCallReservationV2:
    parent = predecessors.prior.base
    selection = parent.selection
    entry = next(
        item
        for item in selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    require_verified_static_feedback_source_v2(source, selection, parent.control, entry)
    if control != build_round3_schema_control_v2(
        predecessors, governance, authorization, remote_receipt
    ):
        raise PortfolioS1FeedbackError("Round3 schema reservation control drifted")
    if attempt_index == 1:
        if first_artifact is not None or retry_claim is not None:
            raise PortfolioS1FeedbackError(
                "Round3 schema first attempt has retry ancestry"
            )
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
        or not is_round3_schema_retry_eligible_v1(first_artifact.feedback_result)
    ):
        raise PortfolioS1FeedbackError("Round3 schema retry ancestry drifted")
    if global_call_ordinal > ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING:
        raise PortfolioS1FeedbackError("Round3 schema live call ceiling exceeded")
    if entry.selection_ordinal > ROUND3_SCHEMA_LIVE_PHASE_CEILING:
        raise PortfolioS1FeedbackError("Round3 schema phase60 is not authorized")
    draft = Round3SchemaFeedbackCallReservationV2.model_construct(
        source_control_sha256=parent.control.control_sha256,
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
    return Round3SchemaFeedbackCallReservationV2.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(unsigned)}, strict=True
    )


def build_bound_round3_schema_artifact_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    source: VerifiedStaticFeedbackSourceV2,
    result: RecoveryFeedbackEvaluationResultV2,
    *,
    reservation: Round3SchemaFeedbackCallReservationV2,
    first_artifact: BoundRound3SchemaFeedbackArtifactV2 | None = None,
    retry_claim: Round3SchemaGlobalRetryClaimV2 | None = None,
) -> BoundRound3SchemaFeedbackArtifactV2:
    expected = build_round3_schema_reservation_v2(
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
    redacted = redact_recovery_result_for_creator_privacy_v1(
        result,
        private_query_ids=tuple(
            item.query_id for item in predecessors.prior.base.selection.entries
        ),
    )
    if (
        reservation != expected
        or type(redacted) is not RecoveryFeedbackEvaluationResultV2
        or redacted.wire_kind != "round3_primary_json_schema_v1"
        or redacted.query_id != reservation.query_id
        or redacted.packet_sha256 != source.packet.packet_sha256
        or redacted.image_sha256 != source.packet.image.sha256
        or redacted.remote_authorization_id != authorization.authorization_id
        or redacted.remote_authorization_file_sha256
        != sha256_bytes(authorization.canonical_bytes())
        or redacted.remote_receipt_file_sha256
        != control.remote_receipt_file_sha256
        or redacted.remote_receipt_sha256 != control.remote_receipt_sha256
        or redacted.asset_catalog_sha256 != control.remote_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("Round3 schema provider result drifted")
    draft = BoundRound3SchemaFeedbackArtifactV2.model_construct(
        source_control_sha256=predecessors.prior.base.control.control_sha256,
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
        feedback_result=redacted,
        status=redacted.status,
        artifact_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"artifact_sha256"})
    return BoundRound3SchemaFeedbackArtifactV2.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(unsigned)}, strict=True
    )


def _write_schema_model(path: str | Path, model: BaseModel) -> Path:
    target = Path(path)
    atomic_create_file(target, canonical_json_bytes(model.model_dump(mode="json")))
    return target


def _load_schema_model(
    path: str | Path,
    *,
    model_type: type[BaseModel],
    label: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical")
    return model


def round3_schema_attempt_filename_v1(
    global_call_ordinal: int,
    selection_entry_sha256: str,
    attempt_index: int,
) -> str:
    if global_call_ordinal not in range(1, 16) or attempt_index not in {1, 2}:
        raise PortfolioS1FeedbackError("Round3 schema attempt identity is invalid")
    return (
        f"{global_call_ordinal:04d}-{selection_entry_sha256[:16]}-"
        f"attempt-{attempt_index}.json"
    )


def round3_schema_claim_filename_v1(claim_ordinal: int) -> str:
    if claim_ordinal not in {1, 2, 3}:
        raise PortfolioS1FeedbackError("Round3 schema claim ordinal is invalid")
    return f"claim-{claim_ordinal:02d}.json"


def write_round3_schema_claim_v2(
    path: str | Path, claim: Round3SchemaGlobalRetryClaimV2
) -> Path:
    if Path(path).name != round3_schema_claim_filename_v1(claim.claim_ordinal):
        raise PortfolioS1FeedbackError("Round3 schema claim filename drifted")
    return _write_schema_model(path, claim)


def load_round3_schema_claim_v2(path: str | Path) -> Round3SchemaGlobalRetryClaimV2:
    model = _load_schema_model(
        path,
        model_type=Round3SchemaGlobalRetryClaimV2,
        label="Round3 schema retry claim",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3SchemaGlobalRetryClaimV2)
    if Path(path).name != round3_schema_claim_filename_v1(model.claim_ordinal):
        raise PortfolioS1FeedbackError("Round3 schema claim filename drifted")
    return model


def write_round3_schema_reservation_v2(
    path: str | Path, reservation: Round3SchemaFeedbackCallReservationV2
) -> Path:
    if Path(path).name != round3_schema_attempt_filename_v1(
        reservation.global_call_ordinal,
        reservation.selection_entry_sha256,
        reservation.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 schema reservation filename drifted")
    return _write_schema_model(path, reservation)


def load_round3_schema_reservation_v2(
    path: str | Path,
) -> Round3SchemaFeedbackCallReservationV2:
    model = _load_schema_model(
        path,
        model_type=Round3SchemaFeedbackCallReservationV2,
        label="Round3 schema reservation",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3SchemaFeedbackCallReservationV2)
    if Path(path).name != round3_schema_attempt_filename_v1(
        model.global_call_ordinal,
        model.selection_entry_sha256,
        model.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 schema reservation filename drifted")
    return model


def write_bound_round3_schema_artifact_v2(
    path: str | Path, artifact: BoundRound3SchemaFeedbackArtifactV2
) -> Path:
    if Path(path).name != round3_schema_attempt_filename_v1(
        artifact.global_call_ordinal,
        artifact.selection_entry_sha256,
        artifact.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 schema artifact filename drifted")
    return _write_schema_model(path, artifact)


def load_bound_round3_schema_artifact_v2(
    path: str | Path,
) -> BoundRound3SchemaFeedbackArtifactV2:
    model = _load_schema_model(
        path,
        model_type=BoundRound3SchemaFeedbackArtifactV2,
        label="Round3 schema bound artifact",
    )
    assert isinstance(model, BoundRound3SchemaFeedbackArtifactV2)
    if Path(path).name != round3_schema_attempt_filename_v1(
        model.global_call_ordinal,
        model.selection_entry_sha256,
        model.attempt_index,
    ):
        raise PortfolioS1FeedbackError("Round3 schema artifact filename drifted")
    return model


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3SchemaLedgerV2:
    claims: tuple[Round3SchemaGlobalRetryClaimV2, ...]
    reservations: tuple[Round3SchemaFeedbackCallReservationV2, ...]
    artifacts: tuple[BoundRound3SchemaFeedbackArtifactV2, ...]
    orphaned_reservations: tuple[Round3SchemaFeedbackCallReservationV2, ...]
    pending_claims: tuple[Round3SchemaGlobalRetryClaimV2, ...]


def _safe_schema_inventory(directory: Path, *, label: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} directory is unsafe")
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in paths
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory is unsafe")
    return paths


def validate_round3_schema_ledger_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> None:
    selection = predecessors.prior.base.selection
    if (
        len(sources) != 240
        or tuple(item.selection_entry_sha256 for item in sources)
        != tuple(item.entry_sha256 for item in selection.entries)
        or control
        != build_round3_schema_control_v2(
            predecessors, governance, authorization, remote_receipt
        )
        or tuple(item.global_call_ordinal for item in ledger.reservations)
        != tuple(range(1, len(ledger.reservations) + 1))
        or len(ledger.reservations) > ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING
        or len({item.reservation_sha256 for item in ledger.reservations})
        != len(ledger.reservations)
        or len({item.artifact_sha256 for item in ledger.artifacts})
        != len(ledger.artifacts)
    ):
        raise PortfolioS1FeedbackError("Round3 schema ledger root identity drifted")
    source_by_entry: dict[str, VerifiedStaticFeedbackSourceV2] = {}
    for entry, source in zip(selection.entries, sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, selection, predecessors.prior.base.control, entry
        )
        source_by_entry[entry.entry_sha256] = source
    validate_round3_schema_claim_set_v1(
        selection, control, ledger.claims, ledger.artifacts
    )
    reservation_by_sha = {
        item.reservation_sha256: item for item in ledger.reservations
    }
    artifact_by_reservation: dict[str, BoundRound3SchemaFeedbackArtifactV2] = {}
    artifact_by_key: dict[
        tuple[str, int], BoundRound3SchemaFeedbackArtifactV2
    ] = {}
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
            raise PortfolioS1FeedbackError(
                "Round3 schema artifact/reservation binding drifted"
            )
        artifact_by_reservation[artifact.reservation_sha256] = artifact
        artifact_by_key[key] = artifact
    expected_orphans = tuple(
        item
        for item in ledger.reservations
        if item.reservation_sha256 not in artifact_by_reservation
    )
    if ledger.orphaned_reservations != expected_orphans or len(expected_orphans) > 2:
        raise PortfolioS1FeedbackError("Round3 schema orphan ledger drifted")
    if expected_orphans:
        final_reservation = ledger.reservations[-1]
        allowed_wave = (final_reservation,)
        if len(ledger.reservations) >= 2:
            previous = ledger.reservations[-2]
            if (
                previous.attempt_index == 1
                and final_reservation.attempt_index == 1
                and previous.selection_ordinal + 1
                == final_reservation.selection_ordinal
            ):
                allowed_wave = (previous, final_reservation)
        if any(item not in allowed_wave for item in expected_orphans):
            raise PortfolioS1FeedbackError(
                "Round3 schema orphan is outside the final reserved wave"
            )
    first_reservations = tuple(
        item for item in ledger.reservations if item.attempt_index == 1
    )
    if tuple(item.selection_ordinal for item in first_reservations) != tuple(
        range(1, len(first_reservations) + 1)
    ):
        raise PortfolioS1FeedbackError(
            "Round3 schema first attempts changed fixed order"
        )
    claim_by_sha = {item.claim_sha256: item for item in ledger.claims}
    first_artifact_by_entry = {
        item.selection_entry_sha256: item
        for item in ledger.artifacts
        if item.attempt_index == 1
    }
    used_claims: list[str] = []
    for reservation in ledger.reservations:
        entry = selection.entries[reservation.selection_ordinal - 1]
        source = source_by_entry.get(reservation.selection_entry_sha256)
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
        if source is None:
            raise PortfolioS1FeedbackError("Round3 schema reservation source unknown")
        if reservation.attempt_index == 2:
            if (
                claim is None
                or first_artifact is None
                or claim.selection_entry_sha256 != entry.entry_sha256
                or reservation.previous_artifact_sha256
                != first_artifact.artifact_sha256
                or reservation.retry_claim_ordinal != claim.claim_ordinal
            ):
                raise PortfolioS1FeedbackError(
                    "Round3 schema retry reservation claim drifted"
                )
            used_claims.append(claim.claim_sha256)
        expected = build_round3_schema_reservation_v2(
            predecessors,
            governance,
            authorization,
            control,
            remote_receipt,
            source,
            attempt_index=reservation.attempt_index,
            global_call_ordinal=reservation.global_call_ordinal,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if reservation != expected:
            raise PortfolioS1FeedbackError(
                "Round3 schema reservation differs from verified source"
            )
    for artifact in ledger.artifacts:
        reservation = reservation_by_sha[artifact.reservation_sha256]
        source = source_by_entry[artifact.selection_entry_sha256]
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
        expected = build_bound_round3_schema_artifact_v2(
            predecessors,
            governance,
            authorization,
            control,
            remote_receipt,
            source,
            artifact.feedback_result,
            reservation=reservation,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if artifact != expected:
            raise PortfolioS1FeedbackError(
                "Round3 schema artifact differs from verified source/result"
            )
    claim_sha256s = [item.claim_sha256 for item in ledger.claims]
    if used_claims != claim_sha256s[: len(used_claims)]:
        raise PortfolioS1FeedbackError(
            "Round3 schema retry claims consumed out of order"
        )
    expected_pending = ledger.claims[len(used_claims) :]
    if ledger.pending_claims != expected_pending or len(expected_pending) > 1:
        raise PortfolioS1FeedbackError("Round3 schema pending claim drifted")


def load_round3_schema_ledger_v1(
    output_root: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> PortfolioS1FeedbackRound3SchemaLedgerV2:
    root = Path(output_root)
    claims = tuple(
        load_round3_schema_claim_v2(path)
        for path in _safe_schema_inventory(
            root / "global-retry-claims-round3-schema-v1",
            label="Round3 schema claims",
        )
    )
    reservations = tuple(
        sorted(
            (
                load_round3_schema_reservation_v2(path)
                for path in _safe_schema_inventory(
                    root / "provider-attempts-round3-schema-v1",
                    label="Round3 schema reservations",
                )
            ),
            key=lambda item: item.global_call_ordinal,
        )
    )
    artifacts = tuple(
        sorted(
            (
                load_bound_round3_schema_artifact_v2(path)
                for path in _safe_schema_inventory(
                    root / "bound-feedback-round3-schema-v1",
                    label="Round3 schema artifacts",
                )
            ),
            key=lambda item: item.global_call_ordinal,
        )
    )
    settled = {item.reservation_sha256 for item in artifacts}
    used_claim_count = sum(item.attempt_index == 2 for item in reservations)
    ledger = PortfolioS1FeedbackRound3SchemaLedgerV2(
        claims=claims,
        reservations=reservations,
        artifacts=artifacts,
        orphaned_reservations=tuple(
            item for item in reservations if item.reservation_sha256 not in settled
        ),
        pending_claims=claims[used_claim_count:],
    )
    validate_round3_schema_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    return ledger


class Round3SchemaNextStepV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal[
        "create_claim",
        "reserve_retry",
        "reserve_primary_wave",
        "phase_complete",
        "terminal_nonparsed",
        "terminal_orphan",
        "terminal_budget",
    ]
    selection_ordinals: tuple[int, ...] = ()
    attempt_index: Literal[1, 2] | None = None
    claim_ordinal: Literal[1, 2, 3] | None = None
    parsed_count: int = Field(ge=0, le=12)
    provider_calls_reserved: int = Field(ge=0, le=15)
    terminal_error_code: str | None = None

    @field_validator("selection_ordinals", mode="before")
    @classmethod
    def _ordinals(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def next_round3_schema_step_v1(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    phase_count: int = 12,
) -> Round3SchemaNextStepV2:
    validate_round3_schema_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    if phase_count != ROUND3_SCHEMA_LIVE_PHASE_CEILING:
        raise PortfolioS1FeedbackError(
            "Round3 schema phase60 requires new owner approval"
        )
    artifacts_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item
        for item in ledger.artifacts
    }
    final_by_ordinal: dict[int, BoundRound3SchemaFeedbackArtifactV2] = {}
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
        return Round3SchemaNextStepV2(
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
        return Round3SchemaNextStepV2(
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
            if not is_round3_schema_retry_eligible_v1(item.feedback_result)
        )
        if noneligible:
            failed = noneligible[0]
            return Round3SchemaNextStepV2(
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
        if len(unclaimed) > ROUND3_SCHEMA_GLOBAL_RETRY_CEILING - len(ledger.claims):
            return Round3SchemaNextStepV2(
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
            if len(ledger.reservations) >= ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING:
                return Round3SchemaNextStepV2(
                    kind="terminal_budget",
                    selection_ordinals=(first.selection_ordinal,),
                    attempt_index=2,
                    terminal_error_code="provider_call_ceiling_exceeded",
                    **common,
                )
            return Round3SchemaNextStepV2(
                kind="create_claim",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=len(ledger.claims) + 1,  # type: ignore[arg-type]
                **common,
            )
        if len(ledger.reservations) >= ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING:
            return Round3SchemaNextStepV2(
                kind="terminal_budget",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=claim.claim_ordinal,
                terminal_error_code="provider_call_ceiling_exceeded",
                **common,
            )
        return Round3SchemaNextStepV2(
            kind="reserve_retry",
            selection_ordinals=(first.selection_ordinal,),
            attempt_index=2,
            claim_ordinal=claim.claim_ordinal,
            **common,
        )
    if all(
        ordinal in final_by_ordinal and final_by_ordinal[ordinal].status == "parsed"
        for ordinal in range(1, 13)
    ):
        return Round3SchemaNextStepV2(kind="phase_complete", **common)
    pending = tuple(
        ordinal for ordinal in range(1, 13) if ordinal not in final_by_ordinal
    )
    if not pending:
        raise PortfolioS1FeedbackError(
            "Round3 schema phase has unresolved nonparsed output"
        )
    wave = pending[:2]
    if len(ledger.reservations) + len(wave) > ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING:
        return Round3SchemaNextStepV2(
            kind="terminal_budget",
            selection_ordinals=wave,
            attempt_index=1,
            terminal_error_code="provider_call_ceiling_exceeded",
            **common,
        )
    return Round3SchemaNextStepV2(
        kind="reserve_primary_wave",
        selection_ordinals=wave,
        attempt_index=1,
        **common,
    )


class PortfolioS1FeedbackRound3SchemaRunV2(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-feedback-round3-schema-run"] = (
        "portfolio-s1-feedback-round3-schema-run"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-run-v2"] = (
        ROUND3_SCHEMA_RUN_POLICY_VERSION_V2
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    predecessor_receipt_sha256: Sha256
    authorization_sha256: Sha256
    control_sha256: Sha256
    launch_sha256: Sha256
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    retry_policy_sha256: Literal[ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2] = (
        ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2
    )
    expected_canary_count: Literal[12] = 12
    selected_universe_count: Literal[240] = 240
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = (
        ROUND3_SCHEMA_PHASE_COUNTS
    )
    attempted_count: int = Field(ge=1, le=12)
    parsed_count: int = Field(ge=0, le=12)
    error_count: int = Field(ge=0, le=12)
    orphan_count: int = Field(ge=0, le=2)
    provider_calls_reserved: int = Field(ge=1, le=15)
    retry_count: int = Field(ge=0, le=3)
    retry_claim_sha256s: tuple[Sha256, ...]
    terminal_phase_count: Literal[12] = 12
    status: Literal[
        "stopped_nonparsed",
        "stopped_orphan",
        "stopped_budget",
        "completed_canary",
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
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    phase60_requires_new_owner_approval: Literal[True] = True
    bundle_v11_policy_version_reserved: Literal[
        "portfolio-s1-feedback-bundle-v11"
    ] = ROUND3_SCHEMA_BUNDLE_POLICY_VERSION_V11
    bundle_publishable_from_canary: Literal[False] = False
    usage_known_count: int = Field(ge=0, le=15)
    usage_unknown_count: int = Field(ge=0, le=15)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ] = ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    new_actual_cost_cny: str
    fresh_accountable_cost_cny: str
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
            (
                Decimal(item.actual_cost_cny)
                for item in self.artifacts
                if item.actual_cost_cny is not None
            ),
            Decimal("0"),
        )
        unknown = sum(item.input_tokens is None for item in self.artifacts)
        fresh_accountable = new_actual + Decimal(
            ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY
        ) * unknown
        cumulative_actual = Decimal(
            ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
        ) + new_actual
        cumulative_accountable = Decimal(
            ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
        ) + fresh_accountable
        usage_breach = any(
            item.input_tokens is not None
            and (
                item.input_tokens > 20_000 or (item.output_tokens or 0) > 6_154
            )
            for item in self.artifacts
        )
        accountable_breach = fresh_accountable > Decimal(
            ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY
        )

        def money(value: Decimal) -> str:
            return format(value.quantize(Decimal("0.000000000001")), "f")

        if (
            self.phase_counts != ROUND3_SCHEMA_PHASE_COUNTS
            or tuple(item.global_call_ordinal for item in self.artifacts)
            != tuple(range(1, len(self.artifacts) + 1))
            or self.provider_calls_reserved != len(self.artifacts)
            or self.attempted_count != len(final)
            or self.parsed_count != statuses["parsed"]
            or self.error_count != self.attempted_count - self.parsed_count
            or self.orphan_count
            != sum(item.status == "orphan" for item in self.artifacts)
            or self.retry_count
            != sum(item.attempt_index == 2 for item in self.artifacts)
            or self.retry_count != len(self.retry_claim_sha256s)
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count
            != self.provider_calls_reserved - self.usage_known_count
            or self.input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
            or self.new_actual_cost_cny != money(new_actual)
            or self.fresh_accountable_cost_cny != money(fresh_accountable)
            or self.cumulative_actual_cost_cny != money(cumulative_actual)
            or self.cumulative_accountable_cost_cny
            != money(cumulative_accountable)
            or self.artifact_set_sha256
            != _hash_payload(
                [item.model_dump(mode="json") for item in self.artifacts]
            )
        ):
            raise ValueError("Round3 schema run counts, costs, or hashes drifted")
        if self.status == "completed_canary":
            if (
                self.terminal_reason is not None
                or usage_breach
                or accountable_breach
                or self.attempted_count != 12
                or self.parsed_count != 12
                or self.orphan_count
                or tuple(sorted(item.selection_ordinal for item in final.values()))
                != tuple(range(1, 13))
            ):
                raise ValueError("completed Round3 schema canary is not parsed12")
        elif self.status == "stopped_orphan":
            if not self.orphan_count:
                raise ValueError("Round3 schema orphan status drifted")
        elif self.status == "stopped_budget":
            if self.terminal_reason is None or self.orphan_count:
                raise ValueError("Round3 schema budget status drifted")
        elif (
            self.error_count < 1
            or self.orphan_count
            or self.terminal_reason is not None
            or usage_breach
            or accountable_breach
        ):
            raise ValueError("Round3 schema nonparsed status drifted")
        if self.status in {"stopped_budget", "stopped_orphan"}:
            if self.terminal_reason == "usage_limit_exceeded" and not usage_breach:
                raise ValueError("Round3 schema usage reason lacks evidence")
            if (
                self.terminal_reason == "accountable_cost_exceeded"
                and not accountable_breach
            ):
                raise ValueError("Round3 schema cost reason lacks evidence")
            if (
                self.terminal_reason == "provider_call_ceiling_exceeded"
                and self.provider_calls_reserved
                != ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING
            ):
                raise ValueError("Round3 schema call reason lacks evidence")
            if usage_breach and self.terminal_reason != "usage_limit_exceeded":
                raise ValueError("Round3 schema usage evidence was not reported")
            if (
                accountable_breach
                and not usage_breach
                and self.terminal_reason != "accountable_cost_exceeded"
            ):
                raise ValueError("Round3 schema cost evidence was not reported")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("Round3 schema run hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _round3_schema_actual_cost(result: RecoveryFeedbackEvaluationResultV2) -> str | None:
    usage = result.usage
    if usage is None:
        return None
    value = (
        Decimal(usage.input_tokens) * Decimal(12)
        + Decimal(usage.output_tokens) * Decimal(36)
    ) / Decimal(1_000_000)
    return format(value.quantize(Decimal("0.000000000001")), "f")


def build_round3_schema_run_v2(
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    authorization: PortfolioS1FeedbackRound3AuthorizationV2,
    control: PortfolioS1FeedbackRound3ControlV2,
    launch: PortfolioS1FeedbackRound3LaunchV2,
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
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
) -> PortfolioS1FeedbackRound3SchemaRunV2:
    validate_round3_schema_ledger_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        ledger,
        sources,
    )
    if launch != build_round3_schema_launch_v2(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        run_id=launch.run_id,
    ):
        raise PortfolioS1FeedbackError("Round3 schema run launch drifted")
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
                actual_cost_cny=_round3_schema_actual_cost(result),
            )
        )
    if not rows:
        raise PortfolioS1FeedbackError("Round3 schema run has no provider attempts")
    final = {item.selection_entry_sha256: item for item in rows}
    parsed = sum(item.status == "parsed" for item in final.values())
    usage_limit = any(
        item.input_tokens is not None
        and (item.input_tokens > 20_000 or (item.output_tokens or 0) > 6_154)
        for item in rows
    )
    new_actual = sum(
        (
            Decimal(item.actual_cost_cny)
            for item in rows
            if item.actual_cost_cny is not None
        ),
        Decimal("0"),
    )
    unknown = sum(item.input_tokens is None for item in rows)
    fresh_accountable = new_actual + Decimal(
        ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY
    ) * unknown
    cumulative_actual = Decimal(
        ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ) + new_actual
    cumulative_accountable = Decimal(
        ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
    ) + fresh_accountable
    if usage_limit:
        terminal_reason = "usage_limit_exceeded"
    elif fresh_accountable > Decimal(ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY):
        terminal_reason = "accountable_cost_exceeded"
    if ledger.orphaned_reservations:
        status = "stopped_orphan"
    elif terminal_reason is not None:
        status = "stopped_budget"
    elif len(final) == 12 and parsed == 12:
        status = "completed_canary"
    elif any(item.status != "parsed" for item in final.values()):
        status = "stopped_nonparsed"
    else:
        raise PortfolioS1FeedbackError(
            "Round3 schema nonterminal prefix cannot publish a run"
        )

    def money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.000000000001")), "f")

    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-s1-feedback-round3-schema-run",
        "policy_version": ROUND3_SCHEMA_RUN_POLICY_VERSION_V2,
        "selection_sha256": predecessors.prior.base.selection.selection_sha256,
        "predecessor_receipt_sha256": predecessors.receipt.receipt_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "control_sha256": control.control_sha256,
        "launch_sha256": launch.launch_sha256,
        "transport_policy_sha256": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        ),
        "retry_policy_sha256": ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2,
        "expected_canary_count": 12,
        "selected_universe_count": 240,
        "phase_counts": ROUND3_SCHEMA_PHASE_COUNTS,
        "attempted_count": len(final),
        "parsed_count": parsed,
        "error_count": len(final) - parsed,
        "orphan_count": len(ledger.orphaned_reservations),
        "provider_calls_reserved": len(rows),
        "retry_count": sum(item.attempt_index == 2 for item in rows),
        "retry_claim_sha256s": tuple(item.claim_sha256 for item in ledger.claims),
        "terminal_phase_count": 12,
        "status": status,
        "terminal_reason": terminal_reason,
        "historical_feedback_outputs_imported": 0,
        "terminated_json_object_canary_outputs_imported": 0,
        "phase60_requires_new_owner_approval": True,
        "bundle_v11_policy_version_reserved": ROUND3_SCHEMA_BUNDLE_POLICY_VERSION_V11,
        "bundle_publishable_from_canary": False,
        "usage_known_count": len(rows) - unknown,
        "usage_unknown_count": unknown,
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "prior_cumulative_actual_cost_cny": ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY,
        "new_actual_cost_cny": money(new_actual),
        "fresh_accountable_cost_cny": money(fresh_accountable),
        "cumulative_actual_cost_cny": money(cumulative_actual),
        "cumulative_accountable_cost_cny": money(cumulative_accountable),
        "artifacts": tuple(rows),
        "artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRound3SchemaRunV2.model_validate(
        {**unsigned, "run_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1FeedbackBundleV11Identity(_StrictFrozenModel):
    schema_version: Literal[11] = 11
    kind: Literal["portfolio-s1-feedback-bundle-identity"] = (
        "portfolio-s1-feedback-bundle-identity"
    )
    policy_version: Literal["portfolio-s1-feedback-bundle-v11"] = (
        ROUND3_SCHEMA_BUNDLE_POLICY_VERSION_V11
    )
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    required_run_policy_version: Literal[
        "portfolio-s1-feedback-round3-run-v2"
    ] = ROUND3_SCHEMA_RUN_POLICY_VERSION_V2
    required_fresh_parsed_count: Literal[240] = 240
    live_canary_publishable: Literal[False] = False
    phase60_requires_new_owner_approval: Literal[True] = True
    identity_sha256: Sha256

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if self.identity_sha256 != _model_hash(self, "identity_sha256"):
            raise ValueError("BundleV11 reserved identity hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s1_feedback_bundle_v11_identity_v1(
) -> PortfolioS1FeedbackBundleV11Identity:
    draft = PortfolioS1FeedbackBundleV11Identity.model_construct(
        identity_sha256="0" * 64
    )
    unsigned = draft.model_dump(mode="json", exclude={"identity_sha256"})
    return PortfolioS1FeedbackBundleV11Identity.model_validate(
        {**unsigned, "identity_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_round3_schema_run_v2(
    path: str | Path, run: PortfolioS1FeedbackRound3SchemaRunV2
) -> Path:
    return _write_schema_model(path, run)


def load_round3_schema_run_v2(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaRunV2:
    model = _load_schema_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaRunV2,
        label="Round3 schema run",
        max_bytes=128 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaRunV2)
    return model


def require_round3_schema_pre_reservation_budget_v1(
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
) -> str:
    """Recheck the V4 authority immediately before every reservation write."""

    actual = Decimal(ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY)
    unknown = len(ledger.orphaned_reservations)
    for artifact in ledger.artifacts:
        usage = artifact.feedback_result.usage
        if usage is None:
            unknown += 1
        else:
            actual += (
                Decimal(usage.input_tokens) * Decimal(12)
                + Decimal(usage.output_tokens) * Decimal(36)
            ) / Decimal(1_000_000)
    committed = actual + Decimal(ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY) * unknown
    return require_qwen38_feedback_pre_call_budget_v4(
        estimated_input_tokens_including_images=20_000,
        provider_calls_already_reserved=len(ledger.reservations),
        committed_cumulative_cost_cny=format(
            committed.quantize(Decimal("0.000000000001")), "f"
        ),
    )


__all__ = [
    "ROUND3_SCHEMA_FUTURE_CUMULATIVE_MAXIMUM_CNY",
    "ROUND3_SCHEMA_FUTURE_MAXIMUM_RESERVATION_CNY",
    "ROUND3_SCHEMA_FUTURE_TECHNICAL_HARD_CAP_CNY",
    "ROUND3_SCHEMA_GLOBAL_RETRY_CEILING",
    "ROUND3_SCHEMA_LIVE_MAXIMUM_RESERVATION_CNY",
    "ROUND3_SCHEMA_LIVE_PHASE_CEILING",
    "ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY",
    "ROUND3_SCHEMA_LIVE_PROVIDER_CALL_CEILING",
    "ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY",
    "ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY",
    "ROUND3_SCHEMA_PHASE_COUNTS",
    "ROUND3_SCHEMA_PROVIDER_CALL_CEILING",
    "ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2",
    "ROUND3_SCHEMA_RETRY_POLICY_VERSION_V2",
    "round3_schema_retry_policy_v2",
]
