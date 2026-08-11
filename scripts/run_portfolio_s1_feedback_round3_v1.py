"""Run the wholly fresh fixed-240 Qwen3.8-Max Feedback Round3.

``prepare`` and ``dry-run`` are zero-provider operations.  ``execute`` uses
create-only provider identities, two-call primary waves, three global
same-entry retry claims, and the frozen 12 -> 60 -> 120 -> 240 boundaries.
Historical Feedback outputs are never imported.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
import sys
from typing import Literal, cast

from dotenv import load_dotenv

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
    RecoveryFeedbackEvaluationResultV1,
    run_visual_feedback_round3_primary_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_v1 import (
    ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY,
    ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY,
    ROUND3_PER_CALL_RESERVATION_CNY,
    ROUND3_PHASE_COUNTS,
    BoundRound3FeedbackArtifactV1,
    PortfolioS1FeedbackBundleV10,
    PortfolioS1FeedbackRound3AuthorizationV1,
    PortfolioS1FeedbackRound3ControlV1,
    PortfolioS1FeedbackRound3LaunchV1,
    PortfolioS1FeedbackRound3LedgerV1,
    PortfolioS1FeedbackRound3RunV1,
    PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
    Round3FeedbackCallReservationV1,
    VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    VerifiedPortfolioS1FeedbackRound3PredecessorsV1,
    build_bound_round3_artifact_v1,
    build_portfolio_s1_feedback_bundle_v10,
    build_round3_authorization_v1,
    build_round3_control_v1,
    build_round3_launch_v1,
    build_round3_reservation_v1,
    build_round3_retry_claim_v1,
    build_round3_run_v1,
    load_portfolio_s1_feedback_bundle_v10,
    load_round3_ledger_v1,
    load_round3_run_v1,
    load_verified_round3_governance_v1,
    load_verified_round3_predecessors_v1,
    next_round3_step_v1,
    prepare_round3_remote_runtime_v1,
    round3_attempt_filename_v1,
    round3_claim_filename_v1,
    write_bound_round3_artifact_v1,
    write_round3_claim_v1,
    write_round3_reservation_v1,
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


ROUND3_PREDECESSOR_FILE = "predecessors-round3-v1.json"
ROUND3_SELECTION_FILE = "selection-v2.json"
ROUND3_AUTHORIZATION_FILE = "authorization-round3-v1.json"
ROUND3_REMOTE_RECEIPT_FILE = "remote-runtime-receipt-round3-v1.json"
ROUND3_CONTROL_FILE = "control-round3-v1.json"
ROUND3_LAUNCH_FILE = "launch-round3-v1.json"
ROUND3_RUN_FILE = "run-round3-v1.json"
ROUND3_BUNDLE_FILE = "bundle-v10.json"
ROUND3_CLAIM_DIR = "global-retry-claims-round3-v1"
ROUND3_ATTEMPT_DIR = "provider-attempts-round3-v1"
ROUND3_BOUND_DIR = "bound-feedback-round3-v1"
_EXECUTE_WRITER_LOCK_FILE = "execute-writer.lock"
_INPUT_TOKEN_LIMIT = 20_000
_OUTPUT_TOKEN_LIMIT = 6_154
_DEFAULT_DASHSCOPE_ENV_FILE = Path(r"D:\athena\ECommerceSkillChain\.env")

_EXPECTED_TOP_LEVEL = frozenset(
    {
        ROUND3_PREDECESSOR_FILE,
        ROUND3_SELECTION_FILE,
        ROUND3_AUTHORIZATION_FILE,
        ROUND3_REMOTE_RECEIPT_FILE,
        ROUND3_CONTROL_FILE,
        ROUND3_LAUNCH_FILE,
        ROUND3_RUN_FILE,
        ROUND3_BUNDLE_FILE,
        ROUND3_CLAIM_DIR,
        ROUND3_ATTEMPT_DIR,
        ROUND3_BOUND_DIR,
        _EXECUTE_WRITER_LOCK_FILE,
    }
)


class PortfolioS1FeedbackRound3CLIError(RuntimeError):
    """The Round3 runner failed closed."""


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackRound3V1:
    output_dir: Path
    predecessors: VerifiedPortfolioS1FeedbackRound3PredecessorsV1
    governance: VerifiedPortfolioS1FeedbackRound3GovernanceV1
    authorization: PortfolioS1FeedbackRound3AuthorizationV1
    control: PortfolioS1FeedbackRound3ControlV1
    launch: PortfolioS1FeedbackRound3LaunchV1
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3ExecutionV1:
    ledger: PortfolioS1FeedbackRound3LedgerV1
    phase_count: int
    run: PortfolioS1FeedbackRound3RunV1 | None = None
    bundle: PortfolioS1FeedbackBundleV10 | None = None


Round3RunnerV1 = Callable[
    [VerifiedStaticFeedbackSourceV2], RecoveryFeedbackEvaluationResultV1
]


def _publish_or_resume(path: Path, content: bytes, *, label: str) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PortfolioS1FeedbackRound3CLIError(
                f"{label} resume target is not a safe regular file"
            )
        if read_stable_regular_file(path, label=label, max_bytes=128 * 1024 * 1024) != content:
            raise PortfolioS1FeedbackRound3CLIError(f"{label} exact-resume bytes differ")
        return
    atomic_create_file(path, content)


def _prepare_output_root(output_dir: Path) -> Path:
    output = output_dir.absolute()
    if os.path.lexists(output):
        if output.is_symlink() or not output.is_dir():
            raise PortfolioS1FeedbackRound3CLIError(
                "Round3 output is not a safe directory"
            )
        unexpected = {item.name for item in output.iterdir()} - _EXPECTED_TOP_LEVEL
        if unexpected:
            raise PortfolioS1FeedbackRound3CLIError(
                "Round3 output contains unexpected entries: "
                + ", ".join(sorted(unexpected))
            )
    else:
        output.mkdir(parents=True)
    for name in (ROUND3_CLAIM_DIR, ROUND3_ATTEMPT_DIR, ROUND3_BOUND_DIR):
        directory = output / name
        if os.path.lexists(directory):
            if directory.is_symlink() or not directory.is_dir():
                raise PortfolioS1FeedbackRound3CLIError(
                    f"Round3 {name} is not a safe directory"
                )
        else:
            directory.mkdir()
    return output


def _reject_root_overlap(output: Path, *immutable_roots: Path) -> None:
    candidate = output.resolve(strict=False)
    for immutable in immutable_roots:
        frozen = immutable.resolve(strict=True)
        if (
            candidate == frozen
            or candidate.is_relative_to(frozen)
            or frozen.is_relative_to(candidate)
        ):
            raise PortfolioS1FeedbackRound3CLIError(
                "Round3 output overlaps immutable predecessor/source evidence"
            )


def prepare_round3_run_v1(
    arguments: argparse.Namespace,
) -> PreparedPortfolioS1FeedbackRound3V1:
    """Reload every authority and create/resume the zero-provider launch chain."""

    predecessors = load_verified_round3_predecessors_v1(
        arguments.base_parent_root, arguments.recovery_root
    )
    governance = load_verified_round3_governance_v1(arguments.repository_root)
    output_candidate = arguments.output_dir.absolute()
    _reject_root_overlap(
        output_candidate,
        predecessors.base.root,
        predecessors.recovery_root,
        Path(arguments.execution_root),
    )
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=arguments.expected_execution_control_sha256,
        artifact_repository_root=arguments.artifact_repository_root,
    )
    sources = build_verified_static_feedback_sources_v2(
        corpus, predecessors.base.selection, predecessors.base.control
    )
    if (
        len(sources) != 240
        or tuple(item.selection_entry_sha256 for item in sources)
        != tuple(item.entry_sha256 for item in predecessors.base.selection.entries)
    ):
        raise PortfolioS1FeedbackRound3CLIError(
            "Round3 sources differ from frozen selected240"
        )
    authorization = build_round3_authorization_v1(
        predecessors,
        governance,
        authorization_id=arguments.authorization_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=arguments.reviewed_at,
    )
    output = _prepare_output_root(output_candidate)
    _publish_or_resume(
        output / ROUND3_PREDECESSOR_FILE,
        predecessors.receipt.canonical_bytes(),
        label="Round3 predecessor receipt",
    )
    _publish_or_resume(
        output / ROUND3_SELECTION_FILE,
        predecessors.base.selection.canonical_bytes(),
        label="Round3 selection",
    )
    _publish_or_resume(
        output / ROUND3_AUTHORIZATION_FILE,
        authorization.canonical_bytes(),
        label="Round3 authorization",
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for(
        "dashscope-qwen-assistant"
    )
    remote_runtime = prepare_round3_remote_runtime_v1(
        predecessors,
        governance,
        corpus,
        authorization,
        parent_remote_runtime,
        receipt_path=output / ROUND3_REMOTE_RECEIPT_FILE,
        verified_sources=sources,
    )
    remote_receipt = cast(
        PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV1,
        remote_runtime.receipt,
    )
    control = build_round3_control_v1(
        predecessors, governance, authorization, remote_receipt
    )
    launch = build_round3_launch_v1(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        run_id=arguments.run_id,
    )
    _publish_or_resume(
        output / ROUND3_CONTROL_FILE,
        control.canonical_bytes(),
        label="Round3 control",
    )
    _publish_or_resume(
        output / ROUND3_LAUNCH_FILE,
        launch.canonical_bytes(),
        label="Round3 launch",
    )
    prepared = PreparedPortfolioS1FeedbackRound3V1(
        output_dir=output,
        predecessors=predecessors,
        governance=governance,
        authorization=authorization,
        control=control,
        launch=launch,
        remote_receipt=remote_receipt,
        sources=sources,
        remote_runtime=remote_runtime,
    )
    _load_ledger(prepared)
    return prepared


def _load_ledger(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
) -> PortfolioS1FeedbackRound3LedgerV1:
    ledger = load_round3_ledger_v1(
        prepared.output_dir,
        predecessors=prepared.predecessors,
        governance=prepared.governance,
        authorization=prepared.authorization,
        control=prepared.control,
        remote_receipt=prepared.remote_receipt,
        sources=prepared.sources,
    )
    run_path = prepared.output_dir / ROUND3_RUN_FILE
    bundle_path = prepared.output_dir / ROUND3_BUNDLE_FILE
    run_exists = os.path.lexists(run_path)
    bundle_exists = os.path.lexists(bundle_path)
    if bundle_exists and not run_exists:
        raise PortfolioS1FeedbackRound3CLIError("BundleV10 exists without Round3 run")
    if run_exists:
        run = load_round3_run_v1(run_path)
        expected_run = build_round3_run_v1(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.remote_receipt,
            ledger,
            prepared.sources,
            terminal_reason=run.terminal_reason,
        )
        if run != expected_run or run.canonical_bytes() != read_stable_regular_file(
            run_path, label="Round3 run", max_bytes=128 * 1024 * 1024
        ):
            raise PortfolioS1FeedbackRound3CLIError(
                "Round3 run differs from exact ledger"
            )
        if bundle_exists:
            bundle_content = read_stable_regular_file(
                bundle_path, label="Round3 BundleV10", max_bytes=128 * 1024 * 1024
            )
            bundle = load_portfolio_s1_feedback_bundle_v10(
                bundle_path, expected_file_sha256=sha256_bytes(bundle_content)
            )
            expected_bundle = build_portfolio_s1_feedback_bundle_v10(
                prepared.predecessors,
                prepared.governance,
                prepared.authorization,
                prepared.control,
                prepared.remote_receipt,
                ledger,
                run,
                prepared.sources,
            )
            if bundle != expected_bundle or bundle.canonical_bytes() != bundle_content:
                raise PortfolioS1FeedbackRound3CLIError(
                    "BundleV10 differs from exact fresh Round3 ledger"
                )
        # A completed run without BundleV10 is the sole recoverable terminal
        # prefix: a crash may occur between the two create-only publications.
        # ``execute`` deterministically derives the missing bundle with zero
        # provider calls.
    return ledger


def _require_qwen_execute_environment() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackRound3CLIError(
            "Qwen Feedback credential is absent; no Round3 attempt was reserved"
        )


def _load_owner_dashscope_environment(path: Path) -> None:
    """Load the owner-designated newest DashScope key before client creation."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackRound3CLIError(
            "DashScope env file cannot be a symlink"
        )
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or not load_dotenv(resolved, override=True):
        raise PortfolioS1FeedbackRound3CLIError(
            "DashScope env file is missing or contains no environment values"
        )
    _require_qwen_execute_environment()


def _live_runner(prepared: PreparedPortfolioS1FeedbackRound3V1) -> Round3RunnerV1:
    isolation = make_active_portfolio_evaluator_isolation_lock()

    def invoke(source: VerifiedStaticFeedbackSourceV2) -> RecoveryFeedbackEvaluationResultV1:
        return run_visual_feedback_round3_primary_v1(
            source.packet,
            isolation,
            remote_runtime=prepared.remote_runtime,
            record_usage=True,
        )

    return invoke


def _source(prepared: PreparedPortfolioS1FeedbackRound3V1, ordinal: int) -> VerifiedStaticFeedbackSourceV2:
    return prepared.sources[ordinal - 1]


def _final_artifact(
    ledger: PortfolioS1FeedbackRound3LedgerV1, ordinal: int
) -> BoundRound3FeedbackArtifactV1 | None:
    matches = tuple(item for item in ledger.artifacts if item.selection_ordinal == ordinal)
    return max(matches, key=lambda item: item.attempt_index) if matches else None


def _accountable_cost(
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    *,
    additional_reservations: int = 0,
) -> Decimal:
    actual = Decimal(ROUND3_PARENT_CUMULATIVE_ACTUAL_CNY)
    usage_unknown = len(ledger.orphaned_reservations) + additional_reservations
    for artifact in ledger.artifacts:
        usage = artifact.feedback_result.usage
        if usage is None:
            usage_unknown += 1
        else:
            actual += (
                Decimal(usage.input_tokens) * Decimal(12)
                + Decimal(usage.output_tokens) * Decimal(36)
            ) / Decimal(1_000_000)
    return actual + Decimal(ROUND3_PER_CALL_RESERVATION_CNY) * usage_unknown


def _budget_terminal_reason(
    ledger: PortfolioS1FeedbackRound3LedgerV1,
) -> Literal["usage_limit_exceeded", "accountable_cost_exceeded"] | None:
    if any(
        item.feedback_result.usage is not None
        and (
            item.feedback_result.usage.input_tokens > _INPUT_TOKEN_LIMIT
            or item.feedback_result.usage.output_tokens > _OUTPUT_TOKEN_LIMIT
        )
        for item in ledger.artifacts
    ):
        return "usage_limit_exceeded"
    if _accountable_cost(ledger) > Decimal(ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY):
        return "accountable_cost_exceeded"
    return None


def _publish_terminal(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None = None,
) -> tuple[PortfolioS1FeedbackRound3RunV1, PortfolioS1FeedbackBundleV10 | None]:
    run = build_round3_run_v1(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        ledger,
        prepared.sources,
        terminal_reason=terminal_reason,
    )
    _publish_or_resume(
        prepared.output_dir / ROUND3_RUN_FILE,
        run.canonical_bytes(),
        label="Round3 run",
    )
    bundle = None
    if run.status == "completed":
        bundle = build_portfolio_s1_feedback_bundle_v10(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.remote_receipt,
            ledger,
            run,
            prepared.sources,
        )
        _publish_or_resume(
            prepared.output_dir / ROUND3_BUNDLE_FILE,
            bundle.canonical_bytes(),
            label="Round3 BundleV10",
        )
    return run, bundle


def _reserve(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    *,
    selection_ordinal: int,
    attempt_index: Literal[1, 2],
) -> Round3FeedbackCallReservationV1:
    source = _source(prepared, selection_ordinal)
    first = _final_artifact(ledger, selection_ordinal) if attempt_index == 2 else None
    claim = (
        next(
            item
            for item in ledger.claims
            if item.selection_entry_sha256 == source.selection_entry_sha256
        )
        if attempt_index == 2
        else None
    )
    reservation = build_round3_reservation_v1(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        source,
        attempt_index=attempt_index,
        global_call_ordinal=len(ledger.reservations) + 1,
        first_artifact=first,
        retry_claim=claim,
    )
    path = prepared.output_dir / ROUND3_ATTEMPT_DIR / round3_attempt_filename_v1(
        reservation.global_call_ordinal,
        reservation.selection_entry_sha256,
        reservation.attempt_index,
    )
    write_round3_reservation_v1(path, reservation)
    return reservation


def _settle(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    ledger: PortfolioS1FeedbackRound3LedgerV1,
    reservation: Round3FeedbackCallReservationV1,
    result: RecoveryFeedbackEvaluationResultV1,
) -> BoundRound3FeedbackArtifactV1:
    source = _source(prepared, reservation.selection_ordinal)
    first = (
        _final_artifact(ledger, reservation.selection_ordinal)
        if reservation.attempt_index == 2
        else None
    )
    claim = (
        next(
            item
            for item in ledger.claims
            if item.claim_sha256 == reservation.retry_claim_sha256
        )
        if reservation.attempt_index == 2
        else None
    )
    artifact = build_bound_round3_artifact_v1(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        source,
        result,
        reservation=reservation,
        first_artifact=first,
        retry_claim=claim,
    )
    path = prepared.output_dir / ROUND3_BOUND_DIR / round3_attempt_filename_v1(
        artifact.global_call_ordinal,
        artifact.selection_entry_sha256,
        artifact.attempt_index,
    )
    write_bound_round3_artifact_v1(path, artifact)
    return artifact


def _invoke_primary_wave(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    ledger_before_reservation: PortfolioS1FeedbackRound3LedgerV1,
    reservations: tuple[Round3FeedbackCallReservationV1, ...],
    runner: Round3RunnerV1,
) -> None:
    futures: dict[int, Future[RecoveryFeedbackEvaluationResultV1]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for reservation in reservations:
            futures[reservation.global_call_ordinal] = executor.submit(
                runner, _source(prepared, reservation.selection_ordinal)
            )
    results: dict[int, RecoveryFeedbackEvaluationResultV1] = {}
    for ordinal, future in futures.items():
        try:
            result = future.result()
        except Exception:
            continue
        if type(result) is RecoveryFeedbackEvaluationResultV1:
            results[ordinal] = result
    # Settle in deterministic call-ordinal order only after the whole wave has
    # returned, so terminal precedence sees both same-wave responses.
    for reservation in reservations:
        result = results.get(reservation.global_call_ordinal)
        if result is None:
            continue
        try:
            _settle(prepared, ledger_before_reservation, reservation, result)
        except (PortfolioS1FeedbackError, OSError, ValueError):
            # The paid identity remains an explicit orphan.  The other wave
            # member is still settled before the terminal run is published.
            continue


def execute_round3_run_v1(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    *,
    runner: Round3RunnerV1,
    stop_after_count: Literal[12, 60, 120, 240] = 240,
) -> PortfolioS1FeedbackRound3ExecutionV1:
    """Execute/resume through the requested frozen parsed-count boundary."""

    phases = tuple(item for item in ROUND3_PHASE_COUNTS if item <= stop_after_count)
    with _exclusive_execute_writer_lock(prepared.output_dir):
        for phase_count in phases:
            while True:
                ledger = _load_ledger(prepared)
                run_path = prepared.output_dir / ROUND3_RUN_FILE
                if os.path.lexists(run_path):
                    run = load_round3_run_v1(run_path)
                    bundle = None
                    bundle_path = prepared.output_dir / ROUND3_BUNDLE_FILE
                    if os.path.lexists(bundle_path):
                        content = read_stable_regular_file(
                            bundle_path,
                            label="Round3 BundleV10",
                            max_bytes=128 * 1024 * 1024,
                        )
                        bundle = load_portfolio_s1_feedback_bundle_v10(
                            bundle_path, expected_file_sha256=sha256_bytes(content)
                        )
                    elif run.status == "completed":
                        bundle = build_portfolio_s1_feedback_bundle_v10(
                            prepared.predecessors,
                            prepared.governance,
                            prepared.authorization,
                            prepared.control,
                            prepared.remote_receipt,
                            ledger,
                            run,
                            prepared.sources,
                        )
                        _publish_or_resume(
                            bundle_path,
                            bundle.canonical_bytes(),
                            label="Round3 BundleV10",
                        )
                    return PortfolioS1FeedbackRound3ExecutionV1(
                        ledger=ledger, phase_count=phase_count, run=run, bundle=bundle
                    )
                budget_reason = _budget_terminal_reason(ledger)
                if budget_reason is not None and ledger.reservations:
                    run, bundle = _publish_terminal(
                        prepared, ledger, terminal_reason=budget_reason
                    )
                    return PortfolioS1FeedbackRound3ExecutionV1(
                        ledger=ledger, phase_count=phase_count, run=run, bundle=bundle
                    )
                step = next_round3_step_v1(
                    prepared.predecessors,
                    prepared.governance,
                    prepared.authorization,
                    prepared.control,
                    prepared.remote_receipt,
                    ledger,
                    prepared.sources,
                    phase_count=cast(Literal[12, 60, 120, 240], phase_count),
                )
                if step.kind in {"phase_complete", "finalize_complete"}:
                    if phase_count == 240:
                        run, bundle = _publish_terminal(prepared, ledger)
                        return PortfolioS1FeedbackRound3ExecutionV1(
                            ledger=ledger,
                            phase_count=phase_count,
                            run=run,
                            bundle=bundle,
                        )
                    break
                if step.kind.startswith("terminal_"):
                    reason = (
                        "provider_call_ceiling_exceeded"
                        if step.terminal_error_code == "provider_call_ceiling_exceeded"
                        else None
                    )
                    run, bundle = _publish_terminal(prepared, ledger, terminal_reason=reason)
                    return PortfolioS1FeedbackRound3ExecutionV1(
                        ledger=ledger, phase_count=phase_count, run=run, bundle=bundle
                    )
                if step.kind == "create_claim":
                    first = _final_artifact(ledger, step.selection_ordinals[0])
                    if first is None:
                        raise PortfolioS1FeedbackRound3CLIError(
                            "Round3 retry claim lacks first artifact"
                        )
                    claim = build_round3_retry_claim_v1(
                        prepared.predecessors.base.selection,
                        prepared.control,
                        first,
                        existing_claims=ledger.claims,
                        existing_artifacts=ledger.artifacts,
                    )
                    write_round3_claim_v1(
                        prepared.output_dir
                        / ROUND3_CLAIM_DIR
                        / round3_claim_filename_v1(claim.claim_ordinal),
                        claim,
                    )
                    continue
                if step.kind == "reserve_retry":
                    if _accountable_cost(ledger, additional_reservations=1) > Decimal(
                        ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
                    ):
                        run, bundle = _publish_terminal(
                            prepared, ledger, terminal_reason="accountable_cost_exceeded"
                        )
                        return PortfolioS1FeedbackRound3ExecutionV1(
                            ledger=ledger,
                            phase_count=phase_count,
                            run=run,
                            bundle=bundle,
                        )
                    reservation = _reserve(
                        prepared,
                        ledger,
                        selection_ordinal=step.selection_ordinals[0],
                        attempt_index=2,
                    )
                    try:
                        result = runner(_source(prepared, reservation.selection_ordinal))
                        if type(result) is RecoveryFeedbackEvaluationResultV1:
                            _settle(prepared, ledger, reservation, result)
                    except Exception:
                        pass
                    continue
                if step.kind != "reserve_primary_wave":
                    raise PortfolioS1FeedbackRound3CLIError(
                        f"unsupported Round3 step {step.kind}"
                    )
                wave_size = len(step.selection_ordinals)
                if _accountable_cost(
                    ledger, additional_reservations=wave_size
                ) > Decimal(ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY):
                    if not ledger.reservations:
                        raise PortfolioS1FeedbackRound3CLIError(
                            "Round3 frozen reservation cannot fund its first wave"
                        )
                    run, bundle = _publish_terminal(
                        prepared, ledger, terminal_reason="accountable_cost_exceeded"
                    )
                    return PortfolioS1FeedbackRound3ExecutionV1(
                        ledger=ledger, phase_count=phase_count, run=run, bundle=bundle
                    )
                reservations: list[Round3FeedbackCallReservationV1] = []
                rolling = ledger
                for ordinal in step.selection_ordinals:
                    reservation = _reserve(
                        prepared,
                        rolling,
                        selection_ordinal=ordinal,
                        attempt_index=1,
                    )
                    reservations.append(reservation)
                    rolling = _load_ledger(prepared)
                    if rolling.orphaned_reservations:
                        # The wave is not invoked until all of its create-only
                        # reservations exist; these are expected interim orphans.
                        continue
                _invoke_primary_wave(prepared, ledger, tuple(reservations), runner)
        ledger = _load_ledger(prepared)
        return PortfolioS1FeedbackRound3ExecutionV1(
            ledger=ledger, phase_count=stop_after_count
        )


def execute_live_round3_run_v1(
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    *,
    stop_after_count: Literal[12, 60, 120, 240] = 240,
) -> PortfolioS1FeedbackRound3ExecutionV1:
    _require_qwen_execute_environment()
    return execute_round3_run_v1(
        prepared,
        runner=_live_runner(prepared),
        stop_after_count=stop_after_count,
    )


def _summary(
    mode: str,
    prepared: PreparedPortfolioS1FeedbackRound3V1,
    outcome: PortfolioS1FeedbackRound3ExecutionV1 | None,
) -> dict[str, object]:
    ledger = _load_ledger(prepared)
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_preflight": 0 if mode != "execute" else None,
        "selected_count": 240,
        "historical_feedback_outputs_imported": 0,
        "phase_counts": list(ROUND3_PHASE_COUNTS),
        "feedback_concurrency": 2,
        "global_retry_ceiling": 3,
        "provider_call_ceiling": 243,
        "provider_calls_reserved": len(ledger.reservations),
        "settled_count": len(ledger.artifacts),
        "orphan_count": len(ledger.orphaned_reservations),
        "retry_claim_count": len(ledger.claims),
        "predecessor_receipt_sha256": prepared.predecessors.receipt.receipt_sha256,
        "authorization_sha256": prepared.authorization.authorization_sha256,
        "control_sha256": prepared.control.control_sha256,
        "launch_sha256": prepared.launch.launch_sha256,
        "remote_receipt_sha256": prepared.remote_receipt.receipt_sha256,
        "run_sha256": None if outcome is None or outcome.run is None else outcome.run.run_sha256,
        "bundle_sha256": (
            None if outcome is None or outcome.bundle is None else outcome.bundle.bundle_sha256
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "dry-run", "execute"), required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--base-parent-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dashscope-env-file",
        type=Path,
        default=_DEFAULT_DASHSCOPE_ENV_FILE,
        help="owner-designated .env loaded with override before any DashScope call",
    )
    parser.add_argument("--stop-after-count", type=int, choices=ROUND3_PHASE_COUNTS, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "execute":
            _load_owner_dashscope_environment(arguments.dashscope_env_file)
        prepared = prepare_round3_run_v1(arguments)
        outcome = None
        if arguments.mode == "dry-run":
            _load_ledger(prepared)
        elif arguments.mode == "execute":
            outcome = execute_live_round3_run_v1(
                prepared,
                stop_after_count=cast(
                    Literal[12, 60, 120, 240], arguments.stop_after_count
                ),
            )
        print(
            canonical_json_bytes(_summary(arguments.mode, prepared, outcome)).decode(
                "utf-8"
            ),
            end="",
        )
        return 0
    except (PortfolioS1FeedbackError, PortfolioS1FeedbackRound3CLIError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


__all__ = [
    "PortfolioS1FeedbackRound3CLIError",
    "PortfolioS1FeedbackRound3ExecutionV1",
    "PreparedPortfolioS1FeedbackRound3V1",
    "build_parser",
    "execute_live_round3_run_v1",
    "execute_round3_run_v1",
    "main",
    "prepare_round3_run_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
