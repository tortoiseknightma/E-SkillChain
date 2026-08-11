"""Run the forward-only Qwen3.8-Max strict-JSON-Schema canary.

``prepare`` and ``dry-run`` are zero-provider operations. ``execute`` is
limited to the fresh fixed ordinals 1..12 plus at most three same-entry
parser/length retries (15 provider identities total). The stopped JSON-object
canary is immutable cost/lineage evidence only; none of its Feedback outputs
is imported. Phase 60 requires a new owner approval and a new forward version.
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
    RecoveryFeedbackEvaluationResultV2,
    run_visual_feedback_round3_schema_primary_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_v1 import (
    ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY,
    ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY,
    ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY,
    BoundRound3SchemaFeedbackArtifactV2,
    PortfolioS1FeedbackBundleV11Identity,
    PortfolioS1FeedbackRound3AuthorizationV2,
    PortfolioS1FeedbackRound3ControlV2,
    PortfolioS1FeedbackRound3LaunchV2,
    PortfolioS1FeedbackRound3SchemaLedgerV2,
    PortfolioS1FeedbackRound3SchemaRunV2,
    PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    Round3SchemaFeedbackCallReservationV2,
    VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    build_bound_round3_schema_artifact_v2,
    build_portfolio_s1_feedback_bundle_v11_identity_v1,
    build_round3_schema_authorization_v2,
    build_round3_schema_control_v2,
    build_round3_schema_launch_v2,
    build_round3_schema_reservation_v2,
    build_round3_schema_retry_claim_v2,
    build_round3_schema_run_v2,
    load_round3_schema_ledger_v1,
    load_round3_schema_run_v2,
    load_verified_round3_schema_governance_v1,
    load_verified_round3_schema_predecessors_v1,
    next_round3_schema_step_v1,
    prepare_round3_schema_remote_runtime_v1,
    require_round3_schema_pre_reservation_budget_v1,
    round3_schema_attempt_filename_v1,
    round3_schema_claim_filename_v1,
    write_bound_round3_schema_artifact_v2,
    write_round3_schema_claim_v2,
    write_round3_schema_reservation_v2,
    write_round3_schema_run_v2,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    load_verified_static_gcs_corpus,
)
from skillchain.evaluation.visual_runtime import (
    VerifiedSelectedFeedbackRemoteRuntime,
)
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes
from skillchain.tools.serialization import read_stable_regular_file

from scripts.run_portfolio_s1_feedback import _exclusive_execute_writer_lock


SCHEMA_PREDECESSOR_FILE = "predecessors-round3-schema-v2.json"
SCHEMA_SELECTION_FILE = "selection-v2.json"
SCHEMA_AUTHORIZATION_FILE = "authorization-round3-schema-v2.json"
SCHEMA_REMOTE_RECEIPT_FILE = "remote-runtime-receipt-round3-schema-v2.json"
SCHEMA_CONTROL_FILE = "control-round3-schema-v2.json"
SCHEMA_LAUNCH_FILE = "launch-round3-schema-v2.json"
SCHEMA_RUN_FILE = "run-round3-schema-v2.json"
SCHEMA_BUNDLE_IDENTITY_FILE = "bundle-v11-identity.json"
SCHEMA_CLAIM_DIR = "global-retry-claims-round3-schema-v1"
SCHEMA_ATTEMPT_DIR = "provider-attempts-round3-schema-v1"
SCHEMA_BOUND_DIR = "bound-feedback-round3-schema-v1"
_EXECUTE_WRITER_LOCK_FILE = "execute-writer.lock"
_DEFAULT_DASHSCOPE_ENV_FILE = Path(r"D:\athena\ECommerceSkillChain\.env")

_EXPECTED_TOP_LEVEL = frozenset(
    {
        SCHEMA_PREDECESSOR_FILE,
        SCHEMA_SELECTION_FILE,
        SCHEMA_AUTHORIZATION_FILE,
        SCHEMA_REMOTE_RECEIPT_FILE,
        SCHEMA_CONTROL_FILE,
        SCHEMA_LAUNCH_FILE,
        SCHEMA_RUN_FILE,
        SCHEMA_BUNDLE_IDENTITY_FILE,
        SCHEMA_CLAIM_DIR,
        SCHEMA_ATTEMPT_DIR,
        SCHEMA_BOUND_DIR,
        _EXECUTE_WRITER_LOCK_FILE,
    }
)


class PortfolioS1FeedbackRound3SchemaCLIError(RuntimeError):
    """The strict-schema canary failed closed."""


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackRound3SchemaV1:
    output_dir: Path
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1
    authorization: PortfolioS1FeedbackRound3AuthorizationV2
    control: PortfolioS1FeedbackRound3ControlV2
    launch: PortfolioS1FeedbackRound3LaunchV2
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2
    bundle_identity: PortfolioS1FeedbackBundleV11Identity
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3SchemaExecutionV1:
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2
    run: PortfolioS1FeedbackRound3SchemaRunV2 | None = None


Round3SchemaRunnerV1 = Callable[
    [VerifiedStaticFeedbackSourceV2], RecoveryFeedbackEvaluationResultV2
]


def _publish_or_resume(path: Path, content: bytes, *, label: str) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                f"{label} resume target is not a safe regular file"
            )
        if (
            read_stable_regular_file(path, label=label, max_bytes=128 * 1024 * 1024)
            != content
        ):
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                f"{label} exact-resume bytes differ"
            )
        return
    atomic_create_file(path, content)


def _prepare_output_root(output_dir: Path) -> Path:
    output = output_dir.absolute()
    if os.path.lexists(output):
        if output.is_symlink() or not output.is_dir():
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                "Round3 schema output is not a safe directory"
            )
        unexpected = {item.name for item in output.iterdir()} - _EXPECTED_TOP_LEVEL
        if unexpected:
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                "Round3 schema output contains unexpected entries: "
                + ", ".join(sorted(unexpected))
            )
    else:
        output.mkdir(parents=True)
    for name in (SCHEMA_CLAIM_DIR, SCHEMA_ATTEMPT_DIR, SCHEMA_BOUND_DIR):
        directory = output / name
        if os.path.lexists(directory):
            if directory.is_symlink() or not directory.is_dir():
                raise PortfolioS1FeedbackRound3SchemaCLIError(
                    f"Round3 schema {name} is not a safe directory"
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
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                "Round3 schema output overlaps immutable evidence"
            )


def prepare_round3_schema_canary_v1(
    arguments: argparse.Namespace,
) -> PreparedPortfolioS1FeedbackRound3SchemaV1:
    """Create/resume the complete zero-provider schema-canary authority chain."""

    predecessors = load_verified_round3_schema_predecessors_v1(
        arguments.base_parent_root,
        arguments.recovery_root,
        arguments.stopped_object_root,
    )
    governance = load_verified_round3_schema_governance_v1(arguments.repository_root)
    output_candidate = arguments.output_dir.absolute()
    _reject_root_overlap(
        output_candidate,
        predecessors.prior.base.root,
        predecessors.prior.recovery_root,
        predecessors.stopped_object_root,
        Path(arguments.execution_root),
    )
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=arguments.expected_execution_control_sha256,
        artifact_repository_root=arguments.artifact_repository_root,
    )
    sources = build_verified_static_feedback_sources_v2(
        corpus,
        predecessors.prior.base.selection,
        predecessors.prior.base.control,
    )
    if len(sources) != 240 or tuple(
        item.selection_entry_sha256 for item in sources
    ) != tuple(item.entry_sha256 for item in predecessors.prior.base.selection.entries):
        raise PortfolioS1FeedbackRound3SchemaCLIError(
            "Round3 schema sources differ from fixed selected240"
        )
    authorization = build_round3_schema_authorization_v2(
        predecessors,
        governance,
        authorization_id=arguments.authorization_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=arguments.reviewed_at,
    )
    output = _prepare_output_root(output_candidate)
    _publish_or_resume(
        output / SCHEMA_PREDECESSOR_FILE,
        predecessors.receipt.canonical_bytes(),
        label="Round3 schema predecessor receipt",
    )
    _publish_or_resume(
        output / SCHEMA_SELECTION_FILE,
        predecessors.prior.base.selection.canonical_bytes(),
        label="Round3 schema selection",
    )
    _publish_or_resume(
        output / SCHEMA_AUTHORIZATION_FILE,
        authorization.canonical_bytes(),
        label="Round3 schema authorization",
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for("dashscope-qwen-assistant")
    remote_runtime = prepare_round3_schema_remote_runtime_v1(
        predecessors,
        governance,
        corpus,
        authorization,
        parent_remote_runtime,
        receipt_path=output / SCHEMA_REMOTE_RECEIPT_FILE,
        verified_sources=sources,
    )
    remote_receipt = cast(
        PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
        remote_runtime.receipt,
    )
    control = build_round3_schema_control_v2(
        predecessors, governance, authorization, remote_receipt
    )
    launch = build_round3_schema_launch_v2(
        predecessors,
        governance,
        authorization,
        control,
        remote_receipt,
        run_id=arguments.run_id,
    )
    bundle_identity = build_portfolio_s1_feedback_bundle_v11_identity_v1()
    _publish_or_resume(
        output / SCHEMA_CONTROL_FILE,
        control.canonical_bytes(),
        label="Round3 schema control",
    )
    _publish_or_resume(
        output / SCHEMA_LAUNCH_FILE,
        launch.canonical_bytes(),
        label="Round3 schema launch",
    )
    _publish_or_resume(
        output / SCHEMA_BUNDLE_IDENTITY_FILE,
        bundle_identity.canonical_bytes(),
        label="BundleV11 reserved identity",
    )
    prepared = PreparedPortfolioS1FeedbackRound3SchemaV1(
        output_dir=output,
        predecessors=predecessors,
        governance=governance,
        authorization=authorization,
        control=control,
        launch=launch,
        remote_receipt=remote_receipt,
        bundle_identity=bundle_identity,
        sources=sources,
        remote_runtime=remote_runtime,
    )
    _load_ledger(prepared)
    return prepared


def _load_ledger(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
) -> PortfolioS1FeedbackRound3SchemaLedgerV2:
    ledger = load_round3_schema_ledger_v1(
        prepared.output_dir,
        predecessors=prepared.predecessors,
        governance=prepared.governance,
        authorization=prepared.authorization,
        control=prepared.control,
        remote_receipt=prepared.remote_receipt,
        sources=prepared.sources,
    )
    run_path = prepared.output_dir / SCHEMA_RUN_FILE
    if os.path.lexists(run_path):
        run = load_round3_schema_run_v2(run_path)
        expected = build_round3_schema_run_v2(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.launch,
            prepared.remote_receipt,
            ledger,
            prepared.sources,
            terminal_reason=run.terminal_reason,
        )
        if run != expected or run.canonical_bytes() != read_stable_regular_file(
            run_path,
            label="Round3 schema run",
            max_bytes=128 * 1024 * 1024,
        ):
            raise PortfolioS1FeedbackRound3SchemaCLIError(
                "Round3 schema run differs from exact ledger"
            )
    return ledger


def _require_qwen_execute_environment() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackRound3SchemaCLIError(
            "Qwen credential is absent; no schema attempt was reserved"
        )


def _load_owner_dashscope_environment(path: Path) -> None:
    supplied = Path(path)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackRound3SchemaCLIError(
            "DashScope env file cannot be a symlink"
        )
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or not load_dotenv(resolved, override=True):
        raise PortfolioS1FeedbackRound3SchemaCLIError(
            "DashScope env file is missing or contains no environment values"
        )
    _require_qwen_execute_environment()


def _live_runner(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
) -> Round3SchemaRunnerV1:
    isolation = make_active_portfolio_evaluator_isolation_lock()

    def invoke(
        source: VerifiedStaticFeedbackSourceV2,
    ) -> RecoveryFeedbackEvaluationResultV2:
        return run_visual_feedback_round3_schema_primary_v1(
            source.packet,
            isolation,
            remote_runtime=prepared.remote_runtime,
            record_usage=True,
        )

    return invoke


def _source(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1, ordinal: int
) -> VerifiedStaticFeedbackSourceV2:
    return prepared.sources[ordinal - 1]


def _final_artifact(
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2, ordinal: int
) -> BoundRound3SchemaFeedbackArtifactV2 | None:
    matches = tuple(
        item for item in ledger.artifacts if item.selection_ordinal == ordinal
    )
    return max(matches, key=lambda item: item.attempt_index) if matches else None


def _fresh_accountable_cost(
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
) -> Decimal:
    actual = Decimal("0")
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
    return actual + Decimal(ROUND3_SCHEMA_PER_CALL_RESERVATION_CNY) * unknown


def _budget_terminal_reason(
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
) -> Literal["usage_limit_exceeded", "accountable_cost_exceeded"] | None:
    if any(
        item.feedback_result.usage is not None
        and (
            item.feedback_result.usage.input_tokens > 20_000
            or item.feedback_result.usage.output_tokens > 6_154
        )
        for item in ledger.artifacts
    ):
        return "usage_limit_exceeded"
    if _fresh_accountable_cost(ledger) > Decimal(ROUND3_SCHEMA_LIVE_PHASE_HARD_CAP_CNY):
        return "accountable_cost_exceeded"
    return None


def _publish_terminal(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None = None,
) -> PortfolioS1FeedbackRound3SchemaRunV2:
    run = build_round3_schema_run_v2(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.launch,
        prepared.remote_receipt,
        ledger,
        prepared.sources,
        terminal_reason=terminal_reason,
    )
    write_round3_schema_run_v2(prepared.output_dir / SCHEMA_RUN_FILE, run)
    return run


def _reserve(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
    *,
    selection_ordinal: int,
    attempt_index: Literal[1, 2],
) -> Round3SchemaFeedbackCallReservationV2:
    # This authority check is deliberately adjacent to the create-only write.
    require_round3_schema_pre_reservation_budget_v1(ledger)
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
    reservation = build_round3_schema_reservation_v2(
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
    write_round3_schema_reservation_v2(
        prepared.output_dir
        / SCHEMA_ATTEMPT_DIR
        / round3_schema_attempt_filename_v1(
            reservation.global_call_ordinal,
            reservation.selection_entry_sha256,
            reservation.attempt_index,
        ),
        reservation,
    )
    return reservation


def _settle(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2,
    reservation: Round3SchemaFeedbackCallReservationV2,
    result: RecoveryFeedbackEvaluationResultV2,
) -> BoundRound3SchemaFeedbackArtifactV2:
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
    artifact = build_bound_round3_schema_artifact_v2(
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
    write_bound_round3_schema_artifact_v2(
        prepared.output_dir
        / SCHEMA_BOUND_DIR
        / round3_schema_attempt_filename_v1(
            artifact.global_call_ordinal,
            artifact.selection_entry_sha256,
            artifact.attempt_index,
        ),
        artifact,
    )
    return artifact


def _invoke_primary_wave(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    ledger_before_reservation: PortfolioS1FeedbackRound3SchemaLedgerV2,
    reservations: tuple[Round3SchemaFeedbackCallReservationV2, ...],
    runner: Round3SchemaRunnerV1,
) -> None:
    futures: dict[int, Future[RecoveryFeedbackEvaluationResultV2]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for reservation in reservations:
            futures[reservation.global_call_ordinal] = executor.submit(
                runner, _source(prepared, reservation.selection_ordinal)
            )
    results: dict[int, RecoveryFeedbackEvaluationResultV2] = {}
    for ordinal, future in futures.items():
        try:
            result = future.result()
        except Exception:
            continue
        if type(result) is RecoveryFeedbackEvaluationResultV2:
            results[ordinal] = result
    for reservation in reservations:
        result = results.get(reservation.global_call_ordinal)
        if result is None:
            continue
        try:
            # Both reservations are create-only writes completed before either
            # provider future is joined.  Rebuild the durable ledger for each
            # settlement so the successful peer in a partially failed wave is
            # bound against the actual reservation prefix, not the stale
            # pre-wave snapshot.  This also preserves an already-published
            # sibling settlement when the other call becomes an orphan.
            current_ledger = _load_ledger(prepared)
            _settle(prepared, current_ledger, reservation, result)
        except (PortfolioS1FeedbackError, OSError, ValueError):
            continue


def execute_round3_schema_canary_v1(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    *,
    runner: Round3SchemaRunnerV1,
) -> PortfolioS1FeedbackRound3SchemaExecutionV1:
    """Execute/resume only the currently authorized fresh canary12."""

    with _exclusive_execute_writer_lock(prepared.output_dir):
        while True:
            ledger = _load_ledger(prepared)
            run_path = prepared.output_dir / SCHEMA_RUN_FILE
            if os.path.lexists(run_path):
                return PortfolioS1FeedbackRound3SchemaExecutionV1(
                    ledger=ledger, run=load_round3_schema_run_v2(run_path)
                )
            budget_reason = _budget_terminal_reason(ledger)
            if budget_reason is not None and ledger.reservations:
                return PortfolioS1FeedbackRound3SchemaExecutionV1(
                    ledger=ledger,
                    run=_publish_terminal(
                        prepared, ledger, terminal_reason=budget_reason
                    ),
                )
            step = next_round3_schema_step_v1(
                prepared.predecessors,
                prepared.governance,
                prepared.authorization,
                prepared.control,
                prepared.remote_receipt,
                ledger,
                prepared.sources,
                phase_count=12,
            )
            if step.kind == "phase_complete":
                return PortfolioS1FeedbackRound3SchemaExecutionV1(
                    ledger=ledger, run=_publish_terminal(prepared, ledger)
                )
            if step.kind.startswith("terminal_"):
                reason = (
                    "provider_call_ceiling_exceeded"
                    if step.terminal_error_code == "provider_call_ceiling_exceeded"
                    else None
                )
                return PortfolioS1FeedbackRound3SchemaExecutionV1(
                    ledger=ledger,
                    run=_publish_terminal(prepared, ledger, terminal_reason=reason),
                )
            if step.kind == "create_claim":
                first = _final_artifact(ledger, step.selection_ordinals[0])
                if first is None:
                    raise PortfolioS1FeedbackRound3SchemaCLIError(
                        "Round3 schema retry claim lacks first artifact"
                    )
                claim = build_round3_schema_retry_claim_v2(
                    prepared.predecessors.prior.base.selection,
                    prepared.control,
                    first,
                    existing_claims=ledger.claims,
                    existing_artifacts=ledger.artifacts,
                )
                write_round3_schema_claim_v2(
                    prepared.output_dir
                    / SCHEMA_CLAIM_DIR
                    / round3_schema_claim_filename_v1(claim.claim_ordinal),
                    claim,
                )
                continue
            if step.kind == "reserve_retry":
                reservation = _reserve(
                    prepared,
                    ledger,
                    selection_ordinal=step.selection_ordinals[0],
                    attempt_index=2,
                )
                try:
                    result = runner(_source(prepared, reservation.selection_ordinal))
                    if type(result) is RecoveryFeedbackEvaluationResultV2:
                        _settle(prepared, ledger, reservation, result)
                except Exception:
                    pass
                continue
            if step.kind != "reserve_primary_wave":
                raise PortfolioS1FeedbackRound3SchemaCLIError(
                    f"unsupported Round3 schema step {step.kind}"
                )
            reservations: list[Round3SchemaFeedbackCallReservationV2] = []
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
            _invoke_primary_wave(prepared, ledger, tuple(reservations), runner)


def execute_live_round3_schema_canary_v1(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
) -> PortfolioS1FeedbackRound3SchemaExecutionV1:
    _require_qwen_execute_environment()
    return execute_round3_schema_canary_v1(prepared, runner=_live_runner(prepared))


def _summary(
    mode: str,
    prepared: PreparedPortfolioS1FeedbackRound3SchemaV1,
    outcome: PortfolioS1FeedbackRound3SchemaExecutionV1 | None,
) -> dict[str, object]:
    ledger = _load_ledger(prepared)
    run = None if outcome is None else outcome.run
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-schema-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_preflight": 0 if mode != "execute" else None,
        "selected_universe_count": 240,
        "live_authorized_count": 12,
        "historical_feedback_outputs_imported": 0,
        "terminated_json_object_canary_outputs_imported": 0,
        "response_format": "json_schema",
        "feedback_concurrency": 2,
        "global_retry_ceiling": 3,
        "provider_call_ceiling": 15,
        "fresh_stage_hard_cap_cny": "10.000000000000",
        "prior_cumulative_actual_cost_cny": (
            ROUND3_SCHEMA_PARENT_CUMULATIVE_ACTUAL_CNY
        ),
        "phase60_requires_new_owner_approval": True,
        "provider_calls_reserved": len(ledger.reservations),
        "settled_count": len(ledger.artifacts),
        "orphan_count": len(ledger.orphaned_reservations),
        "retry_claim_count": len(ledger.claims),
        "predecessor_receipt_sha256": prepared.predecessors.receipt.receipt_sha256,
        "authorization_sha256": prepared.authorization.authorization_sha256,
        "control_sha256": prepared.control.control_sha256,
        "launch_sha256": prepared.launch.launch_sha256,
        "remote_receipt_sha256": prepared.remote_receipt.receipt_sha256,
        "bundle_v11_identity_sha256": prepared.bundle_identity.identity_sha256,
        "bundle_v11_published": False,
        "run_sha256": None if run is None else run.run_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "dry-run", "execute"), required=True
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--base-parent-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--stopped-object-root", type=Path, required=True)
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
        help="owner-designated .env loaded with override before any live setup",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        # Load/fail closed before prepare can construct a client or reserve a call.
        if arguments.mode == "execute":
            _load_owner_dashscope_environment(arguments.dashscope_env_file)
        prepared = prepare_round3_schema_canary_v1(arguments)
        outcome = None
        if arguments.mode == "dry-run":
            _load_ledger(prepared)
        elif arguments.mode == "execute":
            outcome = execute_live_round3_schema_canary_v1(prepared)
        print(
            canonical_json_bytes(_summary(arguments.mode, prepared, outcome)).decode(
                "utf-8"
            ),
            end="",
        )
        return 0
    except (
        PortfolioS1FeedbackError,
        PortfolioS1FeedbackRound3SchemaCLIError,
        OSError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


__all__ = [
    "PortfolioS1FeedbackRound3SchemaCLIError",
    "PortfolioS1FeedbackRound3SchemaExecutionV1",
    "PreparedPortfolioS1FeedbackRound3SchemaV1",
    "build_parser",
    "execute_live_round3_schema_canary_v1",
    "execute_round3_schema_canary_v1",
    "main",
    "prepare_round3_schema_canary_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
