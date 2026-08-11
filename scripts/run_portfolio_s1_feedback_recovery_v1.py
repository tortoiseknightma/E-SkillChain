"""Forward-only recovery runner for the terminal Qwen3.8 Feedback v3 root.

The recovery root imports only SHA-bound parent evidence.  It is a new serial
algorithm/transport identity and never resumes or rewrites the terminal parent
run.  ``prepare`` and ``dry-run`` perform no provider calls; ``execute``
advances maximum-selection-ordinal phases 73 -> 120 -> 240, whose required
combined parsed totals are 74 -> 120 -> 240, using create-only claims,
reservations, and settlements.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import sys
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain import config
from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackError,
    VerifiedStaticFeedbackSourceV2,
    build_verified_static_feedback_sources_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    BoundFeedbackRecoveryArtifactV1,
    FeedbackRecoveryNextStepV1,
    PortfolioS1FeedbackRecoveryAuthorizationV1,
    PortfolioS1FeedbackRecoveryControlV1,
    PortfolioS1FeedbackRecoveryLedgerV1,
    PortfolioS1FeedbackRecoveryRunV1,
    PortfolioS1FeedbackBundleV9,
    PortfolioS1FeedbackParentEvidenceReceiptV1,
    PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1,
    RecoveryFeedbackEvaluationResultV1,
    RecoveryWireKind,
    VerifiedPortfolioS1FeedbackParentEvidenceV1,
    build_bound_feedback_recovery_artifact_v1,
    build_feedback_recovery_call_reservation_v1,
    build_feedback_recovery_global_claim_v1,
    build_portfolio_s1_feedback_bundle_v9,
    build_portfolio_s1_feedback_recovery_authorization_v1,
    build_portfolio_s1_feedback_recovery_control_v1,
    build_portfolio_s1_feedback_recovery_run_v1,
    build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1,
    feedback_recovery_attempt_filename_v1,
    feedback_recovery_claim_filename_v1,
    load_feedback_recovery_ledger_v1,
    load_portfolio_s1_feedback_bundle_v9,
    load_portfolio_s1_feedback_recovery_run_v1,
    load_selected_qwen38_feedback_remote_runtime_v7,
    load_verified_parent_feedback_evidence_v1,
    next_feedback_recovery_step_v1,
    redact_recovery_result_for_creator_privacy_v1,
    run_visual_feedback_recovery_v1,
    write_bound_feedback_recovery_artifact_v1,
    write_feedback_recovery_call_reservation_v1,
    write_feedback_recovery_global_claim_v1,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    load_verified_static_gcs_corpus,
)
from skillchain.evaluation.visual_runtime import (
    VerifiedSelectedFeedbackRemoteRuntime,
)
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes
from skillchain.tools.serialization import read_stable_regular_file, sha256_bytes

from scripts.run_portfolio_s1_feedback import _exclusive_execute_writer_lock


RECOVERY_AUTHORIZATION_FILE = "authorization-recovery-v1.json"
RECOVERY_CONTROL_FILE = "control-recovery-v1.json"
RECOVERY_LAUNCH_FILE = "launch-lock-recovery-v1.json"
RECOVERY_PARENT_EVIDENCE_FILE = "parent-evidence-v1.json"
RECOVERY_RUN_FILE = "run-recovery-v1.json"
RECOVERY_BUNDLE_FILE = "bundle-v9.json"
RECOVERY_CLAIM_DIR = "recovery-claims-v1"
RECOVERY_ATTEMPT_DIR = "provider-attempts-recovery-v1"
RECOVERY_BOUND_DIR = "bound-feedback-recovery-v1"
RECOVERY_REMOTE_RECEIPT_FILE = "remote-runtime-receipt-v7.json"
RECOVERY_PHASE_MAX_SELECTION_ORDINALS = (73, 120, 240)
RECOVERY_PARENT_ACTUAL_COST_CNY = Decimal("10.747284000000")
RECOVERY_PER_CALL_RESERVATION_CNY = Decimal("0.461544000000")
RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY = Decimal("89.000000000000")
RECOVERY_INPUT_TOKEN_LIMIT = 20_000
RECOVERY_OUTPUT_TOKEN_LIMIT = 6_154
_EXECUTE_WRITER_LOCK_FILE = "execute-writer.lock"

_EXPECTED_TOP_LEVEL = frozenset(
    {
        RECOVERY_AUTHORIZATION_FILE,
        RECOVERY_CONTROL_FILE,
        RECOVERY_LAUNCH_FILE,
        RECOVERY_PARENT_EVIDENCE_FILE,
        RECOVERY_RUN_FILE,
        RECOVERY_BUNDLE_FILE,
        RECOVERY_CLAIM_DIR,
        RECOVERY_ATTEMPT_DIR,
        RECOVERY_BOUND_DIR,
        _EXECUTE_WRITER_LOCK_FILE,
    }
)


class PortfolioS1FeedbackRecoveryCLIError(RuntimeError):
    """The derived recovery CLI failed closed."""


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackRecoveryV1:
    output_dir: Path
    parent: VerifiedPortfolioS1FeedbackParentEvidenceV1
    parent_evidence: PortfolioS1FeedbackParentEvidenceReceiptV1
    authorization: PortfolioS1FeedbackRecoveryAuthorizationV1
    control: PortfolioS1FeedbackRecoveryControlV1
    launch_lock: PortfolioS1Qwen38FeedbackRecoveryLaunchLockV1
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime


@dataclass(frozen=True)
class PortfolioS1FeedbackRecoveryExecutionV1:
    step: FeedbackRecoveryNextStepV1
    ledger: PortfolioS1FeedbackRecoveryLedgerV1
    run: PortfolioS1FeedbackRecoveryRunV1 | None = None
    bundle: PortfolioS1FeedbackBundleV9 | None = None


RecoveryRunnerV1 = Callable[
    [VerifiedStaticFeedbackSourceV2, RecoveryWireKind],
    RecoveryFeedbackEvaluationResultV1,
]


def _parse_reviewed_at_v1(value: str) -> datetime:
    """Parse one canonical, second-precision, timezone-aware ISO timestamp."""

    if type(value) is not str or not value:
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery reviewed_at must be canonical timezone-aware ISO-8601"
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery reviewed_at must be canonical timezone-aware ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond:
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery reviewed_at must be canonical timezone-aware ISO-8601"
        )
    canonical = parsed.isoformat(timespec="seconds")
    if parsed.utcoffset().total_seconds() == 0:
        canonical = canonical.removesuffix("+00:00") + "Z"
    if canonical != value:
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery reviewed_at must be canonical timezone-aware ISO-8601"
        )
    return parsed


def _publish_or_resume(path: Path, content: bytes, *, label: str) -> None:
    """Create a canonical file once, or accept only the exact existing bytes."""

    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PortfolioS1FeedbackRecoveryCLIError(
                f"{label} resume target is not a safe regular file"
            )
        existing = read_stable_regular_file(
            path,
            label=label,
            max_bytes=128 * 1024 * 1024,
        )
        if existing != content:
            raise PortfolioS1FeedbackRecoveryCLIError(
                f"{label} exact-resume bytes differ"
            )
        return
    atomic_create_file(path, content)


def _prepare_output_root(output_dir: Path) -> Path:
    """Create or strictly resume only the recovery-owned filesystem shape."""

    output = output_dir.absolute()
    if os.path.lexists(output):
        if output.is_symlink() or not output.is_dir():
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery output is not a safe directory"
            )
        unexpected = {item.name for item in output.iterdir()} - _EXPECTED_TOP_LEVEL
        if unexpected:
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery output contains unexpected entries: "
                + ", ".join(sorted(unexpected))
            )
    else:
        output.mkdir(parents=True)
    for name in (RECOVERY_CLAIM_DIR, RECOVERY_ATTEMPT_DIR, RECOVERY_BOUND_DIR):
        directory = output / name
        if os.path.lexists(directory):
            if directory.is_symlink() or not directory.is_dir():
                raise PortfolioS1FeedbackRecoveryCLIError(
                    f"Feedback recovery {name} is not a safe directory"
                )
        else:
            directory.mkdir()
    return output


def prepare_recovery_run_v1(
    arguments: argparse.Namespace,
) -> PreparedPortfolioS1FeedbackRecoveryV1:
    """Build or exact-resume the zero-provider recovery authorization chain."""

    parent = load_verified_parent_feedback_evidence_v1(arguments.parent_root)
    output = arguments.output_dir.absolute()
    output_resolved = output.resolve(strict=False)
    parent_resolved = parent.root.resolve(strict=True)
    if (
        output_resolved == parent_resolved
        or output_resolved.is_relative_to(parent_resolved)
        or parent_resolved.is_relative_to(output_resolved)
    ):
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery output must not alias or overlap its immutable parent root"
        )
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=arguments.expected_execution_control_sha256,
        artifact_repository_root=arguments.artifact_repository_root,
    )
    sources = build_verified_static_feedback_sources_v2(
        corpus,
        parent.selection,
        parent.control,
    )
    if len(sources) != 240 or tuple(
        item.selection_entry_sha256 for item in sources
    ) != tuple(item.entry_sha256 for item in parent.selection.entries):
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery sources differ from frozen selected240"
        )
    parent_remote_runtime = corpus.core_inputs.runtime_for("dashscope-qwen-assistant")
    remote_runtime = load_selected_qwen38_feedback_remote_runtime_v7(
        parent,
        parent_remote_runtime,
        verified_sources=sources,
    )
    authorization = build_portfolio_s1_feedback_recovery_authorization_v1(
        parent,
        authorization_id=arguments.authorization_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=_parse_reviewed_at_v1(arguments.reviewed_at),
    )
    control = build_portfolio_s1_feedback_recovery_control_v1(parent, authorization)
    launch = build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
        parent,
        authorization,
        control,
        run_id=arguments.run_id,
    )
    output = _prepare_output_root(output)
    _publish_or_resume(
        output / RECOVERY_PARENT_EVIDENCE_FILE,
        parent.receipt.canonical_bytes(),
        label="Feedback recovery parent evidence",
    )
    _publish_or_resume(
        output / RECOVERY_AUTHORIZATION_FILE,
        authorization.canonical_bytes(),
        label="Feedback recovery authorization",
    )
    _publish_or_resume(
        output / RECOVERY_CONTROL_FILE,
        control.canonical_bytes(),
        label="Feedback recovery control",
    )
    _publish_or_resume(
        output / RECOVERY_LAUNCH_FILE,
        launch.canonical_bytes(),
        label="Feedback recovery launch lock",
    )
    return PreparedPortfolioS1FeedbackRecoveryV1(
        output_dir=output,
        parent=parent,
        parent_evidence=parent.receipt,
        authorization=authorization,
        control=control,
        launch_lock=launch,
        sources=sources,
        remote_runtime=remote_runtime,
    )


def _require_qwen_execute_environment_v1() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Qwen Feedback credential is absent; no recovery attempt was reserved"
        )


def _final_artifact_for_ordinal(
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    selection_ordinal: int,
) -> BoundFeedbackRecoveryArtifactV1 | None:
    matches = tuple(
        item for item in ledger.artifacts if item.selection_ordinal == selection_ordinal
    )
    if not matches:
        return None
    return max(matches, key=lambda item: item.lifetime_attempt_index)


def _load_ledger(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
) -> PortfolioS1FeedbackRecoveryLedgerV1:
    ledger = load_feedback_recovery_ledger_v1(
        prepared.output_dir,
        parent=prepared.parent,
        authorization=prepared.authorization,
        control=prepared.control,
    )
    run_path = prepared.output_dir / RECOVERY_RUN_FILE
    bundle_path = prepared.output_dir / RECOVERY_BUNDLE_FILE
    run_exists = os.path.lexists(run_path)
    bundle_exists = os.path.lexists(bundle_path)
    if bundle_exists and not run_exists:
        raise PortfolioS1FeedbackRecoveryCLIError(
            "Feedback recovery BundleV9 exists without its run"
        )
    run = None
    if run_exists:
        if run_path.is_symlink() or not run_path.is_file():
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery run is not a safe regular file"
            )
        run_content = read_stable_regular_file(
            run_path,
            label="Feedback recovery run",
            max_bytes=128 * 1024 * 1024,
        )
        run = load_portfolio_s1_feedback_recovery_run_v1(run_path)
        expected_run = build_portfolio_s1_feedback_recovery_run_v1(
            prepared.parent,
            prepared.authorization,
            prepared.control,
            ledger,
            terminal_reason=run.terminal_reason,
        )
        if run != expected_run or run_content != expected_run.canonical_bytes():
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery run differs from its exact ledger"
            )
    if bundle_exists:
        if bundle_path.is_symlink() or not bundle_path.is_file() or run is None:
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery BundleV9 is not a safe terminal file"
            )
        bundle_content = read_stable_regular_file(
            bundle_path,
            label="Feedback recovery BundleV9",
            max_bytes=128 * 1024 * 1024,
        )
        bundle = load_portfolio_s1_feedback_bundle_v9(
            bundle_path,
            expected_file_sha256=sha256_bytes(bundle_content),
        )
        expected_bundle = build_portfolio_s1_feedback_bundle_v9(
            prepared.parent,
            prepared.authorization,
            prepared.control,
            ledger,
            run,
        )
        if (
            bundle != expected_bundle
            or bundle_content != expected_bundle.canonical_bytes()
        ):
            raise PortfolioS1FeedbackRecoveryCLIError(
                "Feedback recovery BundleV9 differs from its exact ledger"
            )
    return ledger


def _post_settlement_budget_terminal_reason(
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
) -> Literal["usage_limit_exceeded", "accountable_cost_exceeded"] | None:
    actual = RECOVERY_PARENT_ACTUAL_COST_CNY
    unknown_usage_count = 0
    for artifact in ledger.artifacts:
        usage = artifact.feedback_result.usage
        if usage is None:
            unknown_usage_count += 1
            continue
        if (
            usage.input_tokens > RECOVERY_INPUT_TOKEN_LIMIT
            or usage.output_tokens > RECOVERY_OUTPUT_TOKEN_LIMIT
        ):
            return "usage_limit_exceeded"
        actual += (
            Decimal(usage.input_tokens) * Decimal(12)
            + Decimal(usage.output_tokens) * Decimal(36)
        ) / Decimal(1_000_000)
    accountable = actual + (
        Decimal(unknown_usage_count) * RECOVERY_PER_CALL_RESERVATION_CNY
    )
    if accountable > RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY:
        return "accountable_cost_exceeded"
    return None


def _publish_terminal_run(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    step: FeedbackRecoveryNextStepV1,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None = None,
) -> PortfolioS1FeedbackRecoveryExecutionV1:
    run = build_portfolio_s1_feedback_recovery_run_v1(
        prepared.parent,
        prepared.authorization,
        prepared.control,
        ledger,
        terminal_reason=terminal_reason,
    )
    _publish_or_resume(
        prepared.output_dir / RECOVERY_RUN_FILE,
        run.canonical_bytes(),
        label="Feedback recovery terminal run",
    )
    return PortfolioS1FeedbackRecoveryExecutionV1(
        step=step,
        ledger=ledger,
        run=run,
    )


def _finalize_completed_recovery(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
    ledger: PortfolioS1FeedbackRecoveryLedgerV1,
    step: FeedbackRecoveryNextStepV1,
) -> PortfolioS1FeedbackRecoveryExecutionV1:
    run = build_portfolio_s1_feedback_recovery_run_v1(
        prepared.parent,
        prepared.authorization,
        prepared.control,
        ledger,
    )
    bundle = build_portfolio_s1_feedback_bundle_v9(
        prepared.parent,
        prepared.authorization,
        prepared.control,
        ledger,
        run,
    )
    _publish_or_resume(
        prepared.output_dir / RECOVERY_RUN_FILE,
        run.canonical_bytes(),
        label="Feedback recovery completed run",
    )
    _publish_or_resume(
        prepared.output_dir / RECOVERY_BUNDLE_FILE,
        bundle.canonical_bytes(),
        label="Feedback recovery BundleV9",
    )
    return PortfolioS1FeedbackRecoveryExecutionV1(
        step=step,
        ledger=ledger,
        run=run,
        bundle=bundle,
    )


def execute_recovery_run_v1(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
    *,
    recovery_runner: RecoveryRunnerV1,
    stop_after_count: Literal[73, 120, 240] | None = None,
) -> PortfolioS1FeedbackRecoveryExecutionV1:
    """Advance the create-only recovery ledger through one fixed boundary."""

    with _exclusive_execute_writer_lock(prepared.output_dir):
        while True:
            ledger = _load_ledger(prepared)
            budget_reason = _post_settlement_budget_terminal_reason(ledger)
            if budget_reason is not None:
                step = FeedbackRecoveryNextStepV1(
                    kind="terminal_budget",
                    parsed_total=73
                    + sum(
                        item.status == "parsed"
                        for item in {
                            artifact.selection_ordinal: artifact
                            for artifact in ledger.artifacts
                        }.values()
                    ),
                    terminal_error_code=budget_reason,
                )
                return _publish_terminal_run(
                    prepared,
                    ledger,
                    step,
                    terminal_reason=budget_reason,
                )
            step = next_feedback_recovery_step_v1(
                prepared.parent,
                prepared.authorization,
                prepared.control,
                ledger,
                stop_after_max_selection_ordinal=stop_after_count,
            )
            if step.kind == "create_claim":
                trigger = (
                    None
                    if step.claim_ordinal == 1
                    else _final_artifact_for_ordinal(
                        ledger, step.selection_ordinal or 0
                    )
                )
                claim = build_feedback_recovery_global_claim_v1(
                    prepared.parent,
                    prepared.control,
                    existing_claims=ledger.claims,
                    existing_artifacts=ledger.artifacts,
                    trigger_artifact=trigger,
                )
                write_feedback_recovery_global_claim_v1(
                    prepared.output_dir
                    / RECOVERY_CLAIM_DIR
                    / feedback_recovery_claim_filename_v1(claim.claim_ordinal),
                    claim,
                )
                continue
            if step.kind in {"reserve_claimed_call", "reserve_primary_call"}:
                if (
                    step.selection_ordinal is None
                    or step.lifetime_attempt_index is None
                    or step.new_call_ordinal is None
                    or step.wire_kind is None
                ):
                    raise PortfolioS1FeedbackRecoveryCLIError(
                        "Feedback recovery reserve step is incomplete"
                    )
                source = prepared.sources[step.selection_ordinal - 1]
                claim = (
                    ledger.pending_claims[0]
                    if step.kind == "reserve_claimed_call"
                    else None
                )
                previous = (
                    None
                    if step.selection_ordinal == 73
                    else _final_artifact_for_ordinal(ledger, step.selection_ordinal)
                )
                reservation = build_feedback_recovery_call_reservation_v1(
                    prepared.parent,
                    prepared.authorization,
                    prepared.control,
                    source,
                    new_call_ordinal=step.new_call_ordinal,
                    lifetime_attempt_index=step.lifetime_attempt_index,
                    claim=claim,
                    previous_artifact=previous,
                    existing_claims=ledger.claims,
                    existing_artifacts=ledger.artifacts,
                )
                attempt_name = feedback_recovery_attempt_filename_v1(reservation)
                write_feedback_recovery_call_reservation_v1(
                    prepared.output_dir / RECOVERY_ATTEMPT_DIR / attempt_name,
                    reservation,
                )
                result = recovery_runner(source, step.wire_kind)
                result = redact_recovery_result_for_creator_privacy_v1(
                    result,
                    private_query_ids=tuple(
                        item.query_id for item in prepared.parent.selection.entries
                    ),
                )
                artifact = build_bound_feedback_recovery_artifact_v1(
                    prepared.parent,
                    prepared.authorization,
                    prepared.control,
                    source,
                    result,
                    reservation=reservation,
                    claim=claim,
                    previous_artifact=previous,
                    existing_claims=ledger.claims,
                    existing_artifacts=ledger.artifacts,
                )
                write_bound_feedback_recovery_artifact_v1(
                    prepared.output_dir / RECOVERY_BOUND_DIR / attempt_name,
                    artifact,
                )
                continue
            if step.kind == "phase_complete":
                return PortfolioS1FeedbackRecoveryExecutionV1(
                    step=step,
                    ledger=ledger,
                )
            if step.kind == "finalize_complete":
                return _finalize_completed_recovery(prepared, ledger, step)
            if step.kind in {"terminal_nonparsed", "terminal_orphan"}:
                return _publish_terminal_run(prepared, ledger, step)
            if step.kind == "terminal_budget":
                reason = step.terminal_error_code
                if reason not in {
                    "usage_limit_exceeded",
                    "accountable_cost_exceeded",
                    "provider_call_ceiling_exceeded",
                }:
                    raise PortfolioS1FeedbackRecoveryCLIError(
                        "Feedback recovery budget step lacks a valid reason"
                    )
                return _publish_terminal_run(
                    prepared,
                    ledger,
                    step,
                    terminal_reason=reason,
                )
            raise PortfolioS1FeedbackRecoveryCLIError(
                f"unsupported Feedback recovery step: {step.kind}"
            )


def _live_recovery_runner_v1(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
) -> RecoveryRunnerV1:
    isolation = make_active_portfolio_evaluator_isolation_lock()

    def invoke(
        source: VerifiedStaticFeedbackSourceV2,
        wire_kind: RecoveryWireKind,
    ) -> RecoveryFeedbackEvaluationResultV1:
        return run_visual_feedback_recovery_v1(
            source.packet,
            isolation,
            remote_runtime=prepared.remote_runtime,
            wire_kind=wire_kind,
            record_usage=True,
        )

    return invoke


def execute_live_recovery_run_v1(
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
    *,
    stop_after_count: Literal[73, 120, 240] | None = None,
) -> PortfolioS1FeedbackRecoveryExecutionV1:
    _require_qwen_execute_environment_v1()
    return execute_recovery_run_v1(
        prepared,
        recovery_runner=_live_recovery_runner_v1(prepared),
        stop_after_count=stop_after_count,
    )


def _summary_v1(
    mode: Literal["prepare", "dry-run", "execute"],
    prepared: PreparedPortfolioS1FeedbackRecoveryV1,
    outcome: PortfolioS1FeedbackRecoveryExecutionV1 | None,
    *,
    stop_after_count: int | None,
) -> dict[str, object]:
    ledger = _load_ledger(prepared) if outcome is None else outcome.ledger
    final_by_ordinal = {item.selection_ordinal: item for item in ledger.artifacts}
    parsed_total = 73 + sum(
        item.status == "parsed" for item in final_by_ordinal.values()
    )
    run = None if outcome is None else outcome.run
    bundle = None if outcome is None else outcome.bundle
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-recovery-v1-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_dry_run": 0 if mode != "execute" else None,
        "parent_run_sha256": prepared.parent.run.run_sha256,
        "parent_evidence_sha256": prepared.parent_evidence.evidence_sha256,
        "authorization_sha256": prepared.authorization.authorization_sha256,
        "control_sha256": prepared.control.control_sha256,
        "launch_lock_sha256": prepared.launch_lock.launch_lock_sha256,
        "imported_parsed_count": 73,
        "unresolved_count": 167,
        "combined_parsed_count": parsed_total,
        "phase_max_selection_ordinals": list(RECOVERY_PHASE_MAX_SELECTION_ORDINALS),
        "phase_expected_parsed_totals": [74, 120, 240],
        "new_provider_call_ceiling": 169,
        "provider_internal_max_attempts": 1,
        "execution_concurrency": 1,
        "max_active_provider_calls": 1,
        "parent_execution_mode": "immutable_import_not_resume",
        "not_parent_resume": True,
        "algorithm_identity": "new_derived_recovery_algorithm_v1",
        "transport_identity": "new_json_object_recovery_transport_v8",
        "owner_authorized_budget_ceiling_cny": "150.000000000000",
        "technical_cumulative_hard_cap_cny": "89.000000000000",
        "stop_after_count": stop_after_count,
        "next_or_terminal_step": None if outcome is None else outcome.step.kind,
        "new_provider_calls_reserved": len(ledger.reservations),
        "new_provider_calls_settled": len(ledger.artifacts),
        "recovery_claim_count": len(ledger.claims),
        "new_usage_known_count": sum(
            item.feedback_result.usage is not None for item in ledger.artifacts
        ),
        "new_usage_unknown_or_orphan_count": sum(
            item.feedback_result.usage is None for item in ledger.artifacts
        )
        + len(ledger.orphaned_reservations),
        "run_status": None if run is None else run.status,
        "new_actual_cost_cny": None if run is None else run.new_actual_cost_cny,
        "cumulative_actual_cost_cny": (
            None if run is None else run.cumulative_actual_cost_cny
        ),
        "cumulative_accountable_cost_cny": (
            None if run is None else run.cumulative_accountable_cost_cny
        ),
        "run_sha256": None if run is None else run.run_sha256,
        "bundle_sha256": None if bundle is None else bundle.bundle_sha256,
    }


def build_parser_v1() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "dry-run", "execute"), required=True
    )
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stop-after-count",
        type=int,
        choices=RECOVERY_PHASE_MAX_SELECTION_ORDINALS,
        help="execute/resume through one fixed maximum selection ordinal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser_v1().parse_args(argv)
    try:
        if arguments.stop_after_count is not None and arguments.mode != "execute":
            raise PortfolioS1FeedbackRecoveryCLIError(
                "--stop-after-count is only valid with --mode execute"
            )
        prepared = prepare_recovery_run_v1(arguments)
        outcome = None
        if arguments.mode == "dry-run":
            ledger = _load_ledger(prepared)
            step = next_feedback_recovery_step_v1(
                prepared.parent,
                prepared.authorization,
                prepared.control,
                ledger,
            )
            outcome = PortfolioS1FeedbackRecoveryExecutionV1(
                step=step,
                ledger=ledger,
            )
        if arguments.mode == "execute":
            outcome = execute_live_recovery_run_v1(
                prepared,
                stop_after_count=arguments.stop_after_count,
            )
        print(
            canonical_json_bytes(
                _summary_v1(
                    arguments.mode,
                    prepared,
                    outcome,
                    stop_after_count=arguments.stop_after_count,
                )
            ).decode("utf-8"),
            end="",
        )
        return (
            2
            if outcome is not None
            and outcome.step.kind
            in {"terminal_nonparsed", "terminal_orphan", "terminal_budget"}
            else 0
        )
    except (
        PortfolioS1FeedbackError,
        PortfolioS1FeedbackRecoveryCLIError,
        RuntimeError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


__all__ = [
    "PreparedPortfolioS1FeedbackRecoveryV1",
    "PortfolioS1FeedbackRecoveryExecutionV1",
    "PortfolioS1FeedbackRecoveryCLIError",
    "build_parser_v1",
    "execute_live_recovery_run_v1",
    "execute_recovery_run_v1",
    "main",
    "prepare_recovery_run_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
