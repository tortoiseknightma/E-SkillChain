"""Run the forward-only Qwen3.8-Max strict-JSON-Schema phase60.

The phase imports the frozen twelve parsed canary artifacts only as a verified
prefix.  It adds first attempts for fixed selection ordinals 13..60 and uses a
new, separately approved pool of at most twelve same-entry retries.  ``prepare``
and ``dry-run`` are zero-provider operations.  No mode can publish BundleV11 or
start the S1 Creator, and phase120 always requires a new forward authority.
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
from typing import Literal

# Importing ``skillchain.config`` normally performs an implicit dotenv search.
# Keep all module/bootstrap imports credential-blind; the owner-designated file
# is loaded explicitly only after the live authority gate in ``main``.
_PRIOR_PYTHON_DOTENV_DISABLED = os.environ.get("PYTHON_DOTENV_DISABLED")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

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
    load_parent_selected_feedback_remote_runtime_v1,
    run_visual_feedback_round3_schema_primary_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_phase60_v1 import (
    ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR,
    ROUND3_SCHEMA_PHASE60_BOUND_DIR,
    ROUND3_SCHEMA_PHASE60_CLAIM_DIR,
    ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY,
    ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING,
    ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING,
    ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY,
    ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY,
    BoundRound3SchemaPhase60FeedbackArtifactV1,
    PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    PortfolioS1FeedbackRound3SchemaPhase60LaunchV1,
    PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    PortfolioS1FeedbackRound3SchemaPhase60RunV1,
    Round3SchemaPhase60CallReservationV1,
    VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    build_bound_round3_schema_phase60_artifact_v1,
    build_round3_schema_phase60_authorization_v1,
    build_round3_schema_phase60_control_v1,
    build_round3_schema_phase60_launch_v1,
    build_round3_schema_phase60_owner_approval_v1,
    build_round3_schema_phase60_reservation_v1,
    build_round3_schema_phase60_retry_claim_v1,
    build_round3_schema_phase60_run_v1,
    load_round3_schema_phase60_ledger_v1,
    load_round3_schema_phase60_run_v1,
    load_verified_round3_schema_canary_prefix_v1,
    load_verified_round3_schema_phase60_governance_v1,
    next_round3_schema_phase60_step_v1,
    require_round3_schema_phase60_live_governance_v1,
    require_round3_schema_phase60_pre_reservation_budget_v1,
    round3_schema_phase60_attempt_filename_v1,
    round3_schema_phase60_claim_filename_v1,
    round3_schema_phase60_fresh_accountable_cost_v1,
    write_bound_round3_schema_phase60_artifact_v1,
    write_round3_schema_phase60_claim_v1,
    write_round3_schema_phase60_reservation_v1,
    write_round3_schema_phase60_run_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_v1 import (
    load_verified_round3_schema_governance_v1,
    load_verified_round3_schema_predecessors_v1,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    VerifiedStaticGCSCorpus,
    load_verified_static_gcs_corpus,
)
from skillchain.evaluation.visual_runtime import (
    VerifiedSelectedFeedbackRemoteRuntime,
)
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes
from skillchain.tools.serialization import read_stable_regular_file
from scripts.run_portfolio_s1_feedback import _exclusive_execute_writer_lock

if _PRIOR_PYTHON_DOTENV_DISABLED is None:
    os.environ.pop("PYTHON_DOTENV_DISABLED", None)
else:
    os.environ["PYTHON_DOTENV_DISABLED"] = _PRIOR_PYTHON_DOTENV_DISABLED


PHASE60_CANARY_MANIFEST_FILE = "canary12-result-manifest-v1.json"
PHASE60_SELECTION_FILE = "selection-v2.json"
PHASE60_OWNER_APPROVAL_FILE = "owner-approval-round3-schema-phase60-v1.json"
PHASE60_AUTHORIZATION_FILE = "authorization-round3-schema-phase60-v1.json"
PHASE60_CONTROL_FILE = "control-round3-schema-phase60-v1.json"
PHASE60_LAUNCH_FILE = "launch-round3-schema-phase60-v1.json"
PHASE60_RUN_FILE = "run-round3-schema-phase60-v1.json"
_EXECUTE_WRITER_LOCK_FILE = "execute-writer.lock"
_DEFAULT_DASHSCOPE_ENV_FILE = Path(r"D:\athena\ECommerceSkillChain\.env")

_EXPECTED_TOP_LEVEL = frozenset(
    {
        PHASE60_CANARY_MANIFEST_FILE,
        PHASE60_SELECTION_FILE,
        PHASE60_OWNER_APPROVAL_FILE,
        PHASE60_AUTHORIZATION_FILE,
        PHASE60_CONTROL_FILE,
        PHASE60_LAUNCH_FILE,
        PHASE60_RUN_FILE,
        ROUND3_SCHEMA_PHASE60_CLAIM_DIR,
        ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR,
        ROUND3_SCHEMA_PHASE60_BOUND_DIR,
        _EXECUTE_WRITER_LOCK_FILE,
    }
)


class PortfolioS1FeedbackRound3SchemaPhase60CLIError(RuntimeError):
    """The forward-only strict-schema phase60 CLI failed closed."""


@dataclass(frozen=True)
class PreflightPortfolioS1FeedbackRound3SchemaPhase60V1:
    output_candidate: Path
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1
    corpus: VerifiedStaticGCSCorpus
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackRound3SchemaPhase60V1:
    output_dir: Path
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1
    launch: PortfolioS1FeedbackRound3SchemaPhase60LaunchV1
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1:
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1
    run: PortfolioS1FeedbackRound3SchemaPhase60RunV1 | None = None


Phase60RunnerV1 = Callable[
    [VerifiedStaticFeedbackSourceV2], RecoveryFeedbackEvaluationResultV2
]


def _publish_or_resume(path: Path, content: bytes, *, label: str) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                f"{label} resume target is not a safe regular file"
            )
        if (
            read_stable_regular_file(path, label=label, max_bytes=256 * 1024 * 1024)
            != content
        ):
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                f"{label} exact-resume bytes differ"
            )
        return
    atomic_create_file(path, content)


def _prepare_output_root(output_dir: Path) -> Path:
    output = output_dir.absolute()
    if os.path.lexists(output):
        if output.is_symlink() or not output.is_dir():
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 output is not a safe directory"
            )
        unexpected = {item.name for item in output.iterdir()} - _EXPECTED_TOP_LEVEL
        if unexpected:
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 output contains unexpected entries: "
                + ", ".join(sorted(unexpected))
            )
    else:
        output.mkdir(parents=True)
    for name in (
        ROUND3_SCHEMA_PHASE60_CLAIM_DIR,
        ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR,
        ROUND3_SCHEMA_PHASE60_BOUND_DIR,
    ):
        directory = output / name
        if os.path.lexists(directory):
            if directory.is_symlink() or not directory.is_dir():
                raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                    f"phase60 {name} is not a safe directory"
                )
        else:
            directory.mkdir()
    return output


def _reject_root_overlap(output: Path, *immutable_roots: Path) -> None:
    candidate = output.resolve(strict=False)
    for immutable_root in immutable_roots:
        supplied = Path(immutable_root)
        if supplied.is_symlink():
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 immutable evidence root cannot be a symlink"
            )
        frozen = supplied.resolve(strict=True)
        if (
            candidate == frozen
            or candidate.is_relative_to(frozen)
            or frozen.is_relative_to(candidate)
        ):
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 output overlaps immutable canary/base/recovery/object/static evidence"
            )


def preflight_round3_schema_phase60_v1(
    arguments: argparse.Namespace,
) -> PreflightPortfolioS1FeedbackRound3SchemaPhase60V1:
    """Deep-verify pending phase60 inputs without creating any owned state."""

    governance = load_verified_round3_schema_phase60_governance_v1(
        arguments.repository_root
    )
    predecessors = load_verified_round3_schema_predecessors_v1(
        arguments.base_parent_root,
        arguments.recovery_root,
        arguments.stopped_object_root,
    )
    canary_governance = load_verified_round3_schema_governance_v1(
        arguments.repository_root
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
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "phase60 sources differ from fixed selected240"
        )
    prefix = load_verified_round3_schema_canary_prefix_v1(
        arguments.repository_root,
        arguments.canary_root,
        predecessors=predecessors,
        governance=canary_governance,
        sources=sources,
    )
    output_candidate = arguments.output_dir.absolute()
    _reject_root_overlap(
        output_candidate,
        prefix.canary_root,
        predecessors.prior.base.root,
        predecessors.prior.recovery_root,
        predecessors.stopped_object_root,
        Path(arguments.execution_root),
        Path(arguments.artifact_repository_root),
    )
    return PreflightPortfolioS1FeedbackRound3SchemaPhase60V1(
        output_candidate=output_candidate,
        prefix=prefix,
        governance=governance,
        corpus=corpus,
        sources=sources,
    )


def prepare_round3_schema_phase60_v1(
    arguments: argparse.Namespace,
) -> PreparedPortfolioS1FeedbackRound3SchemaPhase60V1:
    """Build/exact-resume the complete zero-provider phase60 authority chain."""

    checked = preflight_round3_schema_phase60_v1(arguments)
    prefix = checked.prefix
    governance = checked.governance
    corpus = checked.corpus
    sources = checked.sources
    # Pending V5/V8/V15 must stop before an output root, remote runtime,
    # provider client, reservation, or claim is constructed.
    require_round3_schema_phase60_live_governance_v1(governance)

    approval = build_round3_schema_phase60_owner_approval_v1(
        prefix,
        governance,
        approval_id=arguments.approval_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=arguments.reviewed_at,
    )
    authorization = build_round3_schema_phase60_authorization_v1(
        prefix,
        governance,
        approval,
        sources,
        authorization_id=arguments.authorization_id,
    )
    control = build_round3_schema_phase60_control_v1(
        prefix, governance, approval, authorization, sources
    )
    launch = build_round3_schema_phase60_launch_v1(
        prefix,
        governance,
        approval,
        authorization,
        control,
        sources,
        run_id=arguments.run_id,
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for("dashscope-qwen-assistant")
    remote_runtime = load_parent_selected_feedback_remote_runtime_v1(
        prefix.predecessors.prior.base,
        parent_remote_runtime,
        verified_sources=sources,
    )

    output = _prepare_output_root(checked.output_candidate)
    _publish_or_resume(
        output / PHASE60_CANARY_MANIFEST_FILE,
        prefix.manifest.canonical_bytes(),
        label="phase60 frozen canary manifest",
    )
    _publish_or_resume(
        output / PHASE60_SELECTION_FILE,
        prefix.selection.canonical_bytes(),
        label="phase60 fixed selection",
    )
    _publish_or_resume(
        output / PHASE60_OWNER_APPROVAL_FILE,
        approval.canonical_bytes(),
        label="phase60 owner approval",
    )
    _publish_or_resume(
        output / PHASE60_AUTHORIZATION_FILE,
        authorization.canonical_bytes(),
        label="phase60 authorization",
    )
    _publish_or_resume(
        output / PHASE60_CONTROL_FILE,
        control.canonical_bytes(),
        label="phase60 control",
    )
    _publish_or_resume(
        output / PHASE60_LAUNCH_FILE,
        launch.canonical_bytes(),
        label="phase60 launch",
    )
    prepared = PreparedPortfolioS1FeedbackRound3SchemaPhase60V1(
        output_dir=output,
        prefix=prefix,
        governance=governance,
        approval=approval,
        authorization=authorization,
        control=control,
        launch=launch,
        sources=sources,
        remote_runtime=remote_runtime,
    )
    _load_ledger(prepared)
    return prepared


def _load_ledger(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
) -> PortfolioS1FeedbackRound3SchemaPhase60LedgerV1:
    ledger = load_round3_schema_phase60_ledger_v1(
        prepared.output_dir,
        prefix=prepared.prefix,
        governance=prepared.governance,
        approval=prepared.approval,
        authorization=prepared.authorization,
        control=prepared.control,
        sources=prepared.sources,
    )
    run_path = prepared.output_dir / PHASE60_RUN_FILE
    if os.path.lexists(run_path):
        if run_path.is_symlink() or not run_path.is_file():
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 run is not a safe regular file"
            )
        run = load_round3_schema_phase60_run_v1(run_path)
        expected = build_round3_schema_phase60_run_v1(
            prepared.prefix,
            prepared.governance,
            prepared.approval,
            prepared.authorization,
            prepared.control,
            prepared.launch,
            ledger,
            prepared.sources,
            terminal_reason=run.terminal_reason,
        )
        if run != expected or run.canonical_bytes() != read_stable_regular_file(
            run_path, label="phase60 run", max_bytes=256 * 1024 * 1024
        ):
            raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                "phase60 run differs from exact durable ledger"
            )
    return ledger


def _require_qwen_execute_environment() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "Qwen credential is absent; no phase60 attempt was reserved"
        )


def _load_owner_dashscope_environment(path: Path) -> None:
    supplied = Path(path)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "DashScope env file cannot be a symlink"
        )
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or not load_dotenv(resolved, override=True):
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "DashScope env file is missing or contains no environment values"
        )
    _require_qwen_execute_environment()


def _live_runner(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
) -> Phase60RunnerV1:
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
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1, ordinal: int
) -> VerifiedStaticFeedbackSourceV2:
    if ordinal not in range(13, 61):
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "phase60 source ordinal is outside 13..60"
        )
    return prepared.sources[ordinal - 1]


def _final_artifact(
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    ordinal: int,
) -> BoundRound3SchemaPhase60FeedbackArtifactV1 | None:
    matches = tuple(
        item for item in ledger.artifacts if item.selection_ordinal == ordinal
    )
    return max(matches, key=lambda item: item.attempt_index) if matches else None


def _budget_terminal_reason(
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
) -> (
    Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None
):
    if any(
        item.feedback_result.usage is not None
        and (
            item.feedback_result.usage.input_tokens > 20_000
            or item.feedback_result.usage.output_tokens > 6_154
        )
        for item in ledger.artifacts
    ):
        return "usage_limit_exceeded"
    if Decimal(round3_schema_phase60_fresh_accountable_cost_v1(ledger)) > Decimal(
        ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY
    ):
        return "accountable_cost_exceeded"
    if len(ledger.reservations) >= ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING:
        final_by_ordinal: dict[int, BoundRound3SchemaPhase60FeedbackArtifactV1] = {}
        for artifact in ledger.artifacts:
            prior = final_by_ordinal.get(artifact.selection_ordinal)
            if prior is None or artifact.attempt_index > prior.attempt_index:
                final_by_ordinal[artifact.selection_ordinal] = artifact
        if len(final_by_ordinal) < 48 or any(
            item.status != "parsed" for item in final_by_ordinal.values()
        ):
            return "provider_call_ceiling_exceeded"
    return None


def _publish_terminal(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None = None,
) -> PortfolioS1FeedbackRound3SchemaPhase60RunV1:
    run = build_round3_schema_phase60_run_v1(
        prepared.prefix,
        prepared.governance,
        prepared.approval,
        prepared.authorization,
        prepared.control,
        prepared.launch,
        ledger,
        prepared.sources,
        terminal_reason=terminal_reason,
    )
    write_round3_schema_phase60_run_v1(prepared.output_dir / PHASE60_RUN_FILE, run)
    return run


def _reserve(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    *,
    selection_ordinal: int,
    attempt_index: Literal[1, 2],
) -> Round3SchemaPhase60CallReservationV1:
    # Keep this budget check adjacent to the create-only reservation write.
    require_round3_schema_phase60_pre_reservation_budget_v1(ledger)
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
    reservation = build_round3_schema_phase60_reservation_v1(
        prepared.prefix,
        prepared.governance,
        prepared.approval,
        prepared.authorization,
        prepared.control,
        prepared.sources,
        source,
        attempt_index=attempt_index,
        phase60_call_ordinal=len(ledger.reservations) + 1,
        first_artifact=first,
        retry_claim=claim,
    )
    write_round3_schema_phase60_reservation_v1(
        prepared.output_dir
        / ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR
        / round3_schema_phase60_attempt_filename_v1(
            reservation.phase60_call_ordinal,
            reservation.selection_entry_sha256,
            reservation.attempt_index,
        ),
        reservation,
    )
    return reservation


def _settle(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    reservation: Round3SchemaPhase60CallReservationV1,
    result: RecoveryFeedbackEvaluationResultV2,
) -> BoundRound3SchemaPhase60FeedbackArtifactV1:
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
    artifact = build_bound_round3_schema_phase60_artifact_v1(
        prepared.prefix,
        prepared.governance,
        prepared.approval,
        prepared.authorization,
        prepared.control,
        prepared.sources,
        source,
        result,
        reservation=reservation,
        first_artifact=first,
        retry_claim=claim,
    )
    write_bound_round3_schema_phase60_artifact_v1(
        prepared.output_dir
        / ROUND3_SCHEMA_PHASE60_BOUND_DIR
        / round3_schema_phase60_attempt_filename_v1(
            artifact.phase60_call_ordinal,
            artifact.selection_entry_sha256,
            artifact.attempt_index,
        ),
        artifact,
    )
    return artifact


def _invoke_primary_wave(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    reservations: tuple[Round3SchemaPhase60CallReservationV1, ...],
    runner: Phase60RunnerV1,
) -> None:
    futures: dict[int, Future[RecoveryFeedbackEvaluationResultV2]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for reservation in reservations:
            futures[reservation.phase60_call_ordinal] = executor.submit(
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
        result = results.get(reservation.phase60_call_ordinal)
        if result is None:
            continue
        try:
            # Rebuild the durable two-reservation prefix before each settlement.
            _settle(prepared, _load_ledger(prepared), reservation, result)
        except (PortfolioS1FeedbackError, OSError, ValueError):
            continue


def execute_round3_schema_phase60_v1(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    *,
    runner: Phase60RunnerV1,
) -> PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1:
    """Execute/resume only phase60; every crash becomes a durable orphan stop."""

    with _exclusive_execute_writer_lock(prepared.output_dir):
        while True:
            ledger = _load_ledger(prepared)
            run_path = prepared.output_dir / PHASE60_RUN_FILE
            if os.path.lexists(run_path):
                return PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1(
                    ledger=ledger,
                    run=load_round3_schema_phase60_run_v1(run_path),
                )
            budget_reason = _budget_terminal_reason(ledger)
            if budget_reason is not None and ledger.reservations:
                return PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1(
                    ledger=ledger,
                    run=_publish_terminal(
                        prepared, ledger, terminal_reason=budget_reason
                    ),
                )
            step = next_round3_schema_phase60_step_v1(
                prepared.prefix,
                prepared.governance,
                prepared.approval,
                prepared.authorization,
                prepared.control,
                ledger,
                prepared.sources,
                phase_target_count=60,
            )
            if step.kind == "phase_complete":
                return PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1(
                    ledger=ledger, run=_publish_terminal(prepared, ledger)
                )
            if step.kind.startswith("terminal_"):
                reason = (
                    step.terminal_error_code
                    if step.kind == "terminal_budget"
                    and step.terminal_error_code
                    in {
                        "usage_limit_exceeded",
                        "accountable_cost_exceeded",
                        "provider_call_ceiling_exceeded",
                    }
                    else None
                )
                return PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1(
                    ledger=ledger,
                    run=_publish_terminal(
                        prepared,
                        ledger,
                        terminal_reason=reason,  # type: ignore[arg-type]
                    ),
                )
            if step.kind == "create_claim":
                first = _final_artifact(ledger, step.selection_ordinals[0])
                if first is None:
                    raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                        "phase60 retry claim lacks its first artifact"
                    )
                claim = build_round3_schema_phase60_retry_claim_v1(
                    prepared.prefix.selection,
                    prepared.control,
                    first,
                    existing_claims=ledger.claims,
                    existing_artifacts=ledger.artifacts,
                )
                write_round3_schema_phase60_claim_v1(
                    prepared.output_dir
                    / ROUND3_SCHEMA_PHASE60_CLAIM_DIR
                    / round3_schema_phase60_claim_filename_v1(claim.claim_ordinal),
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
                        _settle(prepared, _load_ledger(prepared), reservation, result)
                except Exception:
                    pass
                continue
            if step.kind != "reserve_primary_wave":
                raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
                    f"unsupported phase60 step {step.kind}"
                )
            reservations: list[Round3SchemaPhase60CallReservationV1] = []
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
            _invoke_primary_wave(prepared, tuple(reservations), runner)


def execute_live_round3_schema_phase60_v1(
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
) -> PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1:
    _require_qwen_execute_environment()
    return execute_round3_schema_phase60_v1(prepared, runner=_live_runner(prepared))


def _summary(
    mode: str,
    prepared: PreparedPortfolioS1FeedbackRound3SchemaPhase60V1,
    outcome: PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1 | None,
) -> dict[str, object]:
    ledger = _load_ledger(prepared)
    run = None if outcome is None else outcome.run
    final_by_ordinal: dict[int, BoundRound3SchemaPhase60FeedbackArtifactV1] = {}
    for artifact in ledger.artifacts:
        prior = final_by_ordinal.get(artifact.selection_ordinal)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_ordinal[artifact.selection_ordinal] = artifact
    new_parsed = sum(item.status == "parsed" for item in final_by_ordinal.values())
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-schema-phase60-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_preflight": 0 if mode != "execute" else None,
        "selected_universe_count": 240,
        "imported_canary_parsed_count": 12,
        "new_phase60_selection_count": 48,
        "combined_parsed_count": 12 + new_parsed,
        "response_format": "json_schema",
        "feedback_concurrency": 2,
        "new_global_retry_ceiling": ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING,
        "new_provider_call_ceiling": (ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING),
        "per_call_reservation_cny": ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY,
        "prior_cumulative_actual_cost_cny": (
            ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
        ),
        "provider_calls_reserved": len(ledger.reservations),
        "settled_count": len(ledger.artifacts),
        "orphan_count": len(ledger.orphaned_reservations),
        "retry_claim_count": len(ledger.claims),
        "canary_manifest_sha256": prepared.prefix.manifest.manifest_sha256,
        "selection_sha256": prepared.prefix.selection.selection_sha256,
        "approval_sha256": prepared.approval.approval_sha256,
        "authorization_sha256": prepared.authorization.authorization_sha256,
        "control_sha256": prepared.control.control_sha256,
        "launch_sha256": prepared.launch.launch_sha256,
        "phase120_requires_new_owner_approval": True,
        "bundle_v11_published": False,
        "s1_creator_started": False,
        "run_sha256": None if run is None else run.run_sha256,
    }


def _preflight_summary(
    checked: PreflightPortfolioS1FeedbackRound3SchemaPhase60V1,
) -> dict[str, object]:
    try:
        require_round3_schema_phase60_live_governance_v1(checked.governance)
    except PortfolioS1FeedbackError:
        live_authority = False
    else:
        live_authority = True
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-schema-phase60-preflight-summary",
        "mode": "preflight",
        "status": "ready_for_prepare" if live_authority else "approval_required",
        "provider_calls_performed_by_preflight": 0,
        "output_root_created_by_preflight": False,
        "provider_client_constructed": False,
        "claims_created": 0,
        "reservations_created": 0,
        "selected_universe_count": len(checked.sources),
        "imported_canary_parsed_count": len(checked.prefix.final_artifacts),
        "new_phase60_selection_count": 48,
        "response_format": "json_schema",
        "canary_manifest_sha256": checked.prefix.manifest.manifest_sha256,
        "canary_run_sha256": checked.prefix.run.run_sha256,
        "selection_sha256": checked.prefix.selection.selection_sha256,
        "source_lock_sha256": checked.governance.source_lock.source_lock_sha256,
        "pricing_lock_sha256": checked.governance.pricing_lock.pricing_lock_sha256,
        "role_selection_sha256": (checked.governance.role_selection.selection_sha256),
        "separate_budget_and_retry_approval_required": not live_authority,
        "phase120_requires_new_owner_approval": True,
        "bundle_v11_published": False,
        "s1_creator_started": False,
    }


def _require_authority_cli_arguments(arguments: argparse.Namespace) -> None:
    required = (
        "approval_id",
        "authorization_id",
        "reviewer_id",
        "reviewed_at",
        "run_id",
    )
    missing = tuple(
        name.replace("_", "-")
        for name in required
        if not isinstance(getattr(arguments, name, None), str)
        or not getattr(arguments, name).strip()
    )
    if missing:
        raise PortfolioS1FeedbackRound3SchemaPhase60CLIError(
            "phase60 authority arguments are required outside preflight: "
            + ", ".join(missing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "prepare", "dry-run", "execute"),
        required=True,
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--base-parent-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--stopped-object-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approval-id")
    parser.add_argument("--authorization-id")
    parser.add_argument("--reviewer-id")
    parser.add_argument("--reviewed-at")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--dashscope-env-file",
        type=Path,
        default=_DEFAULT_DASHSCOPE_ENV_FILE,
        help="owner-designated .env loaded with override before live execution",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "preflight":
            checked = preflight_round3_schema_phase60_v1(arguments)
            print(
                canonical_json_bytes(_preflight_summary(checked)).decode("utf-8"),
                end="",
            )
            return 0
        _require_authority_cli_arguments(arguments)
        # Preparation verifies the V5/V8/V15 live grant before any client or
        # reservation.  The credential override happens only after that gate.
        prepared = prepare_round3_schema_phase60_v1(arguments)
        outcome = None
        if arguments.mode == "dry-run":
            _load_ledger(prepared)
        elif arguments.mode == "execute":
            _load_owner_dashscope_environment(arguments.dashscope_env_file)
            outcome = execute_live_round3_schema_phase60_v1(prepared)
        print(
            canonical_json_bytes(_summary(arguments.mode, prepared, outcome)).decode(
                "utf-8"
            ),
            end="",
        )
        return 0
    except (
        PortfolioS1FeedbackError,
        PortfolioS1FeedbackRound3SchemaPhase60CLIError,
        OSError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


__all__ = [
    "PHASE60_AUTHORIZATION_FILE",
    "PHASE60_CANARY_MANIFEST_FILE",
    "PHASE60_CONTROL_FILE",
    "PHASE60_LAUNCH_FILE",
    "PHASE60_OWNER_APPROVAL_FILE",
    "PHASE60_RUN_FILE",
    "PHASE60_SELECTION_FILE",
    "PortfolioS1FeedbackRound3SchemaPhase60CLIError",
    "PortfolioS1FeedbackRound3SchemaPhase60ExecutionV1",
    "PreflightPortfolioS1FeedbackRound3SchemaPhase60V1",
    "PreparedPortfolioS1FeedbackRound3SchemaPhase60V1",
    "build_parser",
    "execute_live_round3_schema_phase60_v1",
    "execute_round3_schema_phase60_v1",
    "main",
    "preflight_round3_schema_phase60_v1",
    "prepare_round3_schema_phase60_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
