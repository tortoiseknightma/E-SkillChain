"""Fresh-v3 live-authorized Qwen3.8 Feedback runner.

This entry point rejects historical Feedback roots and preserves each provider
attempt as a create-only reservation plus settlement.  The checked-in
governance locks bind the owner-authorized CNY150 ceiling separately from the
compiler/runtime CNY113 technical hard stop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Callable, Literal

from skillchain import config
from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    redact_feedback_result_for_creator_privacy,
    run_visual_feedback,
)
from skillchain.evaluation.packets import RubricSnapshot
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    build_portfolio_s1_feedback_selection_v2,
    build_verified_static_feedback_sources_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_retry_v3 import (
    BoundFeedbackArtifactV5,
    FeedbackCallReservationV5,
    FeedbackGlobalRetryClaimV2,
    FRESH_V3_OWNER_BUDGET_STATEMENT,
    PortfolioS1FeedbackAuthorizationV7,
    PortfolioS1FeedbackControlV12,
    PortfolioS1FeedbackBundleV8,
    PortfolioS1FeedbackRunV5,
    PortfolioS1Qwen38FeedbackLaunchLockV5,
    build_bound_feedback_artifact_v5,
    build_feedback_call_reservation_v5,
    build_feedback_global_retry_claim_v2,
    build_portfolio_s1_feedback_bundle_v8,
    build_portfolio_s1_feedback_control_v12,
    build_portfolio_s1_feedback_run_v5,
    build_qwen38_feedback_launch_lock_v5,
    build_selected_qwen38_feedback_authorization_v7,
    is_qwen38_schema_or_length_retry_eligible,
    load_bound_feedback_artifact_v5,
    load_feedback_call_reservation_v5,
    load_feedback_global_retry_claim_v2,
    prepare_selected_qwen38_feedback_remote_runtime_v7,
    require_feedback_v12_creator_projection_privacy,
    validate_portfolio_s1_feedback_control_v12,
    validate_selected_qwen38_feedback_authorization_v7,
    validate_feedback_global_retry_claim_set_v2,
    write_bound_feedback_artifact_v5,
    write_feedback_call_reservation_v5,
    write_feedback_global_retry_claim_v2,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS,
    QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS,
    QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS,
    QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2,
    QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2,
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2,
    QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2,
    PortfolioS1QwenFeedbackGovernanceError,
    Qwen38FeedbackModelSourceLockV2,
    Qwen38FeedbackPricingLockV5,
    Qwen38FeedbackRoleSelectionV12,
    load_qwen38_feedback_model_source_lock_v2,
    load_qwen38_feedback_pricing_lock_v5,
    load_qwen38_feedback_role_selection_v12,
    require_qwen38_feedback_pre_call_budget_v3,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    load_verified_static_gcs_corpus,
)
from skillchain.evaluation.visual_runtime import VerifiedSelectedFeedbackRemoteRuntime
from skillchain.synthesis.store import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_portfolio_s1_feedback import (  # noqa: E402
    SELECTION_SEED,
    _exclusive_execute_writer_lock,
    _load_discovery600,
    _publish_or_resume,
    _rubric,
    _validate_input_reservation_bounds,
)


class PortfolioS1FeedbackV3RunError(RuntimeError):
    """The fresh-v3 runner failed closed."""


class PortfolioS1FeedbackV3BudgetError(PortfolioS1FeedbackV3RunError):
    """A settled fresh-v3 attempt exceeded its frozen budget contract."""

    def __init__(
        self,
        message: str,
        *,
        terminal_reason: Literal[
            "usage_limit_exceeded", "accountable_cost_exceeded"
        ],
    ) -> None:
        super().__init__(message)
        self.terminal_reason = terminal_reason


def _parse_reviewed_at_v3(value: str) -> datetime:
    """Parse one canonical second-precision ISO-8601 timestamp with timezone."""

    if type(value) is not str or not value:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 reviewed_at must be canonical timezone-aware ISO-8601"
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 reviewed_at must be canonical timezone-aware ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 reviewed_at must be canonical timezone-aware ISO-8601"
        )
    canonical = parsed.isoformat(timespec="seconds")
    if parsed.utcoffset().total_seconds() == 0:
        canonical = canonical.removesuffix("+00:00") + "Z"
    if value != canonical:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 reviewed_at must be canonical timezone-aware ISO-8601"
        )
    return parsed


@dataclass(frozen=True)
class FeedbackBudgetSnapshotV5:
    provider_calls_reserved: int
    usage_known_count: int
    usage_unknown_count: int
    input_tokens: int
    output_tokens: int
    settled_actual_cost_cny: Decimal
    unresolved_reserve_cny: Decimal
    accountable_cost_cny: Decimal


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackV3Run:
    output_dir: Path
    selection: PortfolioS1FeedbackSelectionV2
    authorization: PortfolioS1FeedbackAuthorizationV7
    control: PortfolioS1FeedbackControlV12
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime
    launch_lock: PortfolioS1Qwen38FeedbackLaunchLockV5
    model_source_lock: Qwen38FeedbackModelSourceLockV2
    pricing_lock: Qwen38FeedbackPricingLockV5
    role_selection: Qwen38FeedbackRoleSelectionV12


FeedbackRunner = Callable[..., FeedbackEvaluationResult]
RunSourceV5 = Callable[..., BoundFeedbackArtifactV5]
_LEDGER_LOCK = Lock()
_ACTIVE_ATTEMPTS: set[tuple[str, str, int]] = set()


def _active_key(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    entry_sha256: str,
    attempt_index: int,
) -> tuple[str, str, int]:
    return (str(prepared.output_dir.resolve()), entry_sha256, attempt_index)


def _reservation_path_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> Path:
    return (
        prepared.output_dir
        / "provider-attempts-v3"
        / (
            f"{source.row.query_ordinal:04d}-"
            f"{source.selection_entry_sha256[:16]}-attempt-{attempt_index}.json"
        )
    )


def _artifact_path_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> Path:
    return (
        prepared.output_dir
        / "bound-feedback-v3"
        / (
            f"{source.row.query_ordinal:04d}-"
            f"{source.selection_entry_sha256[:16]}-attempt-{attempt_index}.json"
        )
    )


def _claim_path_v2(
    prepared: PreparedPortfolioS1FeedbackV3Run, claim_ordinal: int
) -> Path:
    return (
        prepared.output_dir
        / "global-retry-claims-v2"
        / f"claim-{claim_ordinal:02d}.json"
    )


def _reject_historical_output_dir(output_dir: Path) -> None:
    legacy_paths = (
        "provider-attempts",
        "provider-attempts-v2",
        "bound-feedback",
        "bound-feedback-v2",
        "global-retry-claim.json",
        "selection.json",
        "selected-assets-authorization.json",
        "control.json",
        "launch-lock.json",
        "remote-runtime-receipt.json",
        "portfolio-s1-feedback-run.json",
        "portfolio-s1-feedback-bundle.json",
    )
    for relative in legacy_paths:
        path = output_dir / relative
        if path.is_file() or (path.is_dir() and any(path.iterdir())):
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 rejects a historical Feedback run root before provider"
            )


def _reject_historical_roots(prepared: PreparedPortfolioS1FeedbackV3Run) -> None:
    _reject_historical_output_dir(prepared.output_dir)


def _load_claims_v2(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    first_artifacts: tuple[BoundFeedbackArtifactV5, ...],
) -> tuple[FeedbackGlobalRetryClaimV2, ...]:
    root = prepared.output_dir / "global-retry-claims-v2"
    if not root.exists():
        return ()
    if not root.is_dir():
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 retry claim directory inventory drifted"
        )
    allowed = {f"claim-{item:02d}.json" for item in range(1, 4)}
    unexpected = tuple(
        path for path in root.iterdir() if not path.is_file() or path.name not in allowed
    )
    if unexpected:
        raise PortfolioS1FeedbackV3RunError("fresh-v3 retry claim file set drifted")
    claims: list[FeedbackGlobalRetryClaimV2] = []
    missing_seen = False
    for ordinal in range(1, 4):
        path = _claim_path_v2(prepared, ordinal)
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 retry claims are not contiguous"
            )
        claims.append(load_feedback_global_retry_claim_v2(path))
    try:
        validate_feedback_global_retry_claim_set_v2(
            prepared.selection,
            prepared.control,
            tuple(claims),
            first_artifacts,
        )
    except PortfolioS1FeedbackError as error:
        raise PortfolioS1FeedbackV3RunError(str(error)) from error
    return tuple(claims)


def _attempt_state_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
) -> tuple[
    tuple[BoundFeedbackArtifactV5, ...],
    tuple[FeedbackCallReservationV5, ...],
    tuple[FeedbackGlobalRetryClaimV2, ...],
]:
    _reject_historical_roots(prepared)
    artifacts: dict[tuple[str, int], BoundFeedbackArtifactV5] = {}
    reservations: dict[tuple[str, int], FeedbackCallReservationV5] = {}
    entry_by_sha = {item.entry_sha256: item for item in prepared.selection.entries}
    expected_reservation_names = {
        _reservation_path_v5(prepared, source, attempt).name
        for source in prepared.sources
        for attempt in (1, 2)
    }
    expected_artifact_names = {
        _artifact_path_v5(prepared, source, attempt).name
        for source in prepared.sources
        for attempt in (1, 2)
    }
    for root, expected, label in (
        (prepared.output_dir / "provider-attempts-v3", expected_reservation_names, "reservation"),
        (prepared.output_dir / "bound-feedback-v3", expected_artifact_names, "settlement"),
    ):
        if root.exists() and (
            not root.is_dir()
            or any(
                not item.is_file() or item.name not in expected
                for item in root.iterdir()
            )
        ):
            raise PortfolioS1FeedbackV3RunError(
                f"fresh-v3 {label} directory inventory drifted"
            )
    # First attempts must be reconstructed before claims can be verified.
    for source in prepared.sources:
        key = (source.selection_entry_sha256, 1)
        reservation_path = _reservation_path_v5(prepared, source, 1)
        artifact_path = _artifact_path_v5(prepared, source, 1)
        if artifact_path.exists() and not reservation_path.exists():
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 settlement exists without reservation"
            )
        if not reservation_path.exists():
            continue
        reservation = load_feedback_call_reservation_v5(reservation_path)
        entry = entry_by_sha[source.selection_entry_sha256]
        expected = build_feedback_call_reservation_v5(
            prepared.selection,
            prepared.control,
            entry,
            source,
            attempt_index=1,
            global_call_ordinal=reservation.global_call_ordinal,
        )
        if reservation != expected:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 first reservation resume conflict"
            )
        reservations[key] = reservation
        if artifact_path.exists():
            artifact = load_bound_feedback_artifact_v5(artifact_path)
            rebuilt = build_bound_feedback_artifact_v5(
                prepared.selection,
                prepared.control,
                source,
                artifact.feedback_result,
                reservation=reservation,
            )
            if artifact != rebuilt:
                raise PortfolioS1FeedbackV3RunError(
                    "fresh-v3 first settlement resume conflict"
                )
            artifacts[key] = artifact
    firsts = tuple(item for item in artifacts.values() if item.attempt_index == 1)
    claims = _load_claims_v2(prepared, firsts)
    claim_by_entry = {item.selection_entry_sha256: item for item in claims}
    for source in prepared.sources:
        key = (source.selection_entry_sha256, 2)
        reservation_path = _reservation_path_v5(prepared, source, 2)
        artifact_path = _artifact_path_v5(prepared, source, 2)
        if artifact_path.exists() and not reservation_path.exists():
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 retry settlement exists without reservation"
            )
        if not reservation_path.exists():
            continue
        first = artifacts.get((source.selection_entry_sha256, 1))
        claim = claim_by_entry.get(source.selection_entry_sha256)
        if first is None or claim is None:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 retry reservation lacks first settlement or claim"
            )
        reservation = load_feedback_call_reservation_v5(reservation_path)
        entry = entry_by_sha[source.selection_entry_sha256]
        expected = build_feedback_call_reservation_v5(
            prepared.selection,
            prepared.control,
            entry,
            source,
            attempt_index=2,
            global_call_ordinal=reservation.global_call_ordinal,
            first_artifact=first,
            retry_claim=claim,
        )
        if reservation != expected:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 retry reservation resume conflict"
            )
        reservations[key] = reservation
        if artifact_path.exists():
            artifact = load_bound_feedback_artifact_v5(artifact_path)
            rebuilt = build_bound_feedback_artifact_v5(
                prepared.selection,
                prepared.control,
                source,
                artifact.feedback_result,
                reservation=reservation,
                first_artifact=first,
                retry_claim=claim,
            )
            if artifact != rebuilt:
                raise PortfolioS1FeedbackV3RunError(
                    "fresh-v3 retry settlement resume conflict"
                )
            artifacts[key] = artifact
    ordinals = sorted(item.global_call_ordinal for item in reservations.values())
    if ordinals != list(range(1, len(reservations) + 1)) or len(reservations) > 243:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 global call ordinals are not unique and contiguous"
        )
    orphans = tuple(
        item for key, item in reservations.items() if key not in artifacts
    )
    return tuple(artifacts.values()), orphans, claims


def _budget_snapshot_v5(
    settled: tuple[BoundFeedbackArtifactV5, ...],
    orphans: tuple[FeedbackCallReservationV5, ...],
) -> FeedbackBudgetSnapshotV5:
    """Rebuild actual-plus-unresolved cost from every create-only reservation."""

    known = input_tokens = output_tokens = 0
    actual = Decimal("0")
    for artifact in settled:
        usage = artifact.feedback_result.usage
        if usage is None:
            continue
        if (
            usage.input_tokens > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
            or usage.output_tokens > QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
        ):
            raise PortfolioS1FeedbackV3BudgetError(
                "fresh-v3 provider usage exceeded its per-call reservation",
                terminal_reason="usage_limit_exceeded",
            )
        known += 1
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        actual += (
            Decimal(usage.input_tokens)
            * Decimal(QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS)
            + Decimal(usage.output_tokens)
            * Decimal(QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS)
        ) / Decimal(1_000_000)
    unknown = len(settled) - known + len(orphans)
    unresolved = Decimal(unknown) * Decimal(
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2
    )
    accountable = actual + unresolved
    if accountable > Decimal(QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2):
        raise PortfolioS1FeedbackV3BudgetError(
            "fresh-v3 accountable cost exceeded the CNY113 hard cap",
            terminal_reason="accountable_cost_exceeded",
        )
    return FeedbackBudgetSnapshotV5(
        provider_calls_reserved=len(settled) + len(orphans),
        usage_known_count=known,
        usage_unknown_count=unknown,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        settled_actual_cost_cny=actual,
        unresolved_reserve_cny=unresolved,
        accountable_cost_cny=accountable,
    )


def _claim_retry_v2(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    first: BoundFeedbackArtifactV5,
    artifacts: dict[tuple[str, int], BoundFeedbackArtifactV5],
) -> FeedbackGlobalRetryClaimV2:
    with _LEDGER_LOCK:
        firsts = tuple(item for item in artifacts.values() if item.attempt_index == 1)
        claims = _load_claims_v2(prepared, firsts)
        existing = next(
            (
                item
                for item in claims
                if item.selection_entry_sha256 == first.selection_entry_sha256
            ),
            None,
        )
        if existing is not None:
            return existing
        if len(claims) >= 3:
            raise PortfolioS1FeedbackV3RunError(
                "fourth eligible Feedback result is terminal without provider call"
            )
        entry = prepared.selection.entries[first.selection_ordinal - 1] if hasattr(first, "selection_ordinal") else next(
            item
            for item in prepared.selection.entries
            if item.entry_sha256 == first.selection_entry_sha256
        )
        claim = build_feedback_global_retry_claim_v2(
            prepared.selection,
            prepared.control,
            entry,
            first,
            existing_claims=claims,
            existing_first_artifacts=firsts,
        )
        try:
            write_feedback_global_retry_claim_v2(
                _claim_path_v2(prepared, claim.claim_ordinal), claim
            )
        except FileExistsError as error:
            raise PortfolioS1FeedbackV3RunError(
                "concurrent fresh-v3 retry claim detected"
            ) from error
        return claim


def _run_source_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    source: VerifiedStaticFeedbackSourceV2,
    feedback_runner: FeedbackRunner,
    *,
    attempt_index: Literal[1, 2],
    first_artifact: BoundFeedbackArtifactV5 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV2 | None = None,
    ) -> BoundFeedbackArtifactV5:
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    with _LEDGER_LOCK:
        _require_qwen_execute_environment_v3()
        settled, orphans, _claims = _attempt_state_v5(prepared)
        _budget_snapshot_v5(settled, orphans)
        foreign_orphans = tuple(
            item
            for item in orphans
            if _active_key(
                prepared, item.selection_entry_sha256, item.attempt_index
            )
            not in _ACTIVE_ATTEMPTS
        )
        if foreign_orphans:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 has an orphan reservation; provider retry forbidden"
            )
        if any(
            item.selection_entry_sha256 == source.selection_entry_sha256
            and item.attempt_index == attempt_index
            for item in settled
        ):
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 attempt is already settled; duplicate call forbidden"
            )
        provider_calls = len(settled) + len(orphans)
        committed = Decimal(provider_calls) * Decimal(
            QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2
        )
        reservation_cost = require_qwen38_feedback_pre_call_budget_v3(
            estimated_input_tokens_including_images=20_000,
            provider_calls_already_reserved=provider_calls,
            committed_cost_cny=format(committed, "f"),
        )
        if reservation_cost != QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2:
            raise PortfolioS1FeedbackV3RunError("fresh-v3 reservation cost drifted")
        reservation = build_feedback_call_reservation_v5(
            prepared.selection,
            prepared.control,
            entry,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=provider_calls + 1,
            first_artifact=first_artifact,
            retry_claim=retry_claim,
        )
        try:
            write_feedback_call_reservation_v5(
                _reservation_path_v5(prepared, source, attempt_index), reservation
            )
        except FileExistsError as error:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 reservation already exists; replacement forbidden"
            ) from error
        _ACTIVE_ATTEMPTS.add(
            _active_key(prepared, source.selection_entry_sha256, attempt_index)
        )
    try:
        result = feedback_runner(source, attempt_index)
    except BaseException:
        with _LEDGER_LOCK:
            _ACTIVE_ATTEMPTS.discard(
                _active_key(prepared, source.selection_entry_sha256, attempt_index)
            )
        raise
    try:
        if result.status == "parsed":
            try:
                require_feedback_v12_creator_projection_privacy(
                    result,
                    private_query_ids=tuple(
                        item.query_id for item in prepared.selection.entries
                    ),
                )
            except PortfolioS1FeedbackError:
                result = redact_feedback_result_for_creator_privacy(result)
        artifact = build_bound_feedback_artifact_v5(
            prepared.selection,
            prepared.control,
            source,
            result,
            reservation=reservation,
            first_artifact=first_artifact,
            retry_claim=retry_claim,
        )
    except BaseException:
        with _LEDGER_LOCK:
            _ACTIVE_ATTEMPTS.discard(
                _active_key(prepared, source.selection_entry_sha256, attempt_index)
            )
        raise
    try:
        write_bound_feedback_artifact_v5(
            _artifact_path_v5(prepared, source, attempt_index), artifact
        )
        with _LEDGER_LOCK:
            all_settled, all_orphans, _claims = _attempt_state_v5(prepared)
            _budget_snapshot_v5(all_settled, all_orphans)
    except FileExistsError as error:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 settlement already exists; replacement forbidden"
        ) from error
    finally:
        with _LEDGER_LOCK:
            _ACTIVE_ATTEMPTS.discard(
                _active_key(prepared, source.selection_entry_sha256, attempt_index)
            )
    return artifact


def _execute_phase_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    artifacts: dict[tuple[str, int], BoundFeedbackArtifactV5],
    feedback_runner: FeedbackRunner,
    *,
    run_source: RunSourceV5 = _run_source_v5,
) -> None:
    """Execute one cumulative phase with crash-prioritized ordered retries."""

    _reject_historical_roots(prepared)
    if any(type(item) is not VerifiedStaticFeedbackSourceV2 for item in sources):
        raise PortfolioS1FeedbackV3RunError("fresh-v3 phase source type drifted")

    def process_settled_failures() -> None:
        failed_retries = sorted(
            (
                item
                for (_entry_sha, attempt), item in artifacts.items()
                if attempt == 2 and item.status != "parsed"
            ),
            key=lambda item: item.global_call_ordinal,
        )
        if failed_retries:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 retry attempt failed; run stopped"
            )
        first_failures = sorted(
            (
                item
                for (entry_sha, attempt), item in artifacts.items()
                if attempt == 1
                and (entry_sha, 2) not in artifacts
                and item.status != "parsed"
            ),
            key=lambda item: item.global_call_ordinal,
        )
        if not first_failures:
            return
        noneligible = [
            item
            for item in first_failures
            if not is_qwen38_schema_or_length_retry_eligible(item.feedback_result)
        ]
        if noneligible:
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 stopped on a nonretryable failure before any retry"
            )
        firsts = tuple(item for item in artifacts.values() if item.attempt_index == 1)
        claims = _load_claims_v2(prepared, firsts)
        claimed_entries = {item.selection_entry_sha256 for item in claims}
        unclaimed = [
            item
            for item in first_failures
            if item.selection_entry_sha256 not in claimed_entries
        ]
        if len(unclaimed) > 3 - len(claims):
            raise PortfolioS1FeedbackV3RunError(
                "fourth eligible Feedback result is terminal without provider call"
            )
        for first in first_failures:
            claim = next(
                (
                    item
                    for item in _load_claims_v2(prepared, firsts)
                    if item.selection_entry_sha256 == first.selection_entry_sha256
                ),
                None,
            )
            if claim is None:
                claim = _claim_retry_v2(prepared, first, artifacts)
            source = next(
                item
                for item in sources
                if item.selection_entry_sha256 == first.selection_entry_sha256
            )
            retry = run_source(
                prepared,
                source,
                feedback_runner,
                attempt_index=2,
                first_artifact=first,
                retry_claim=claim,
            )
            artifacts[(retry.selection_entry_sha256, 2)] = retry
            if retry.status != "parsed":
                raise PortfolioS1FeedbackV3RunError(
                    "fresh-v3 retry attempt failed; run stopped"
                )

    # Crash recovery: pending claims/unclaimed eligible settlements always win
    # over every new normal reservation.
    process_settled_failures()
    pending = tuple(
        source
        for source in sources
        if (source.selection_entry_sha256, 1) not in artifacts
    )
    for offset in range(0, len(pending), 2):
        wave = pending[offset : offset + 2]
        completed: list[BoundFeedbackArtifactV5] = []
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(
                    run_source,
                    prepared,
                    source,
                    feedback_runner,
                    attempt_index=1,
                )
                for source in wave
            )
            for future in futures:
                try:
                    completed.append(future.result())
                except Exception as error:  # noqa: PERF203
                    failures.append(error)
        for artifact in completed:
            artifacts[(artifact.selection_entry_sha256, 1)] = artifact
        if failures:
            _settled, orphans, _claims = _attempt_state_v5(prepared)
            if orphans:
                raise PortfolioS1FeedbackV3RunError(
                    "fresh-v3 wave left an orphan; retry forbidden"
                ) from failures[0]
            raise failures[0]
        # Inspect the entire wave before creating any claim.  A provider,
        # privacy, echo, or other noneligible failure has strict precedence.
        nonparsed = [item for item in completed if item.status != "parsed"]
        if any(
            not is_qwen38_schema_or_length_retry_eligible(item.feedback_result)
            for item in nonparsed
        ):
            raise PortfolioS1FeedbackV3RunError(
                "fresh-v3 stopped on same-wave nonretryable failure"
            )
        process_settled_failures()


def _require_live_governance_locks_v3(
    source: Qwen38FeedbackModelSourceLockV2,
    pricing: Qwen38FeedbackPricingLockV5,
    role: Qwen38FeedbackRoleSelectionV12,
) -> None:
    """Require the re-signed CNY150/technical-CNY113 governance triad."""

    if (
        type(source) is not Qwen38FeedbackModelSourceLockV2
        or type(pricing) is not Qwen38FeedbackPricingLockV5
        or type(role) is not Qwen38FeedbackRoleSelectionV12
        or source.source_lock_sha256 != QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2
        or pricing.pricing_lock_sha256
        != QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5
        or role.selection_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12
        or pricing.live_provider_calls_authorized is not True
        or role.feedback_evaluator.get("live_provider_calls_authorized") is not True
        or pricing.owner_budget_authorized_cap_cny
        != "150.000000000000"
        or pricing.technical_phase_hard_cap_cny != "113.000000000000"
        or role.feedback_evaluator.get("owner_authorized_budget_ceiling_cny")
        != "150.000000000000"
        or role.feedback_evaluator.get("technical_phase_hard_cap_cny")
        != "113.000000000000"
    ):
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 live governance lacks exact CNY150 owner authority with the CNY113 technical stop"
        )


def _require_live_authority(
    prepared: PreparedPortfolioS1FeedbackV3Run,
) -> None:
    """Reject stale or pending authority before any provider action."""

    _require_live_governance_locks_v3(
        prepared.model_source_lock,
        prepared.pricing_lock,
        prepared.role_selection,
    )
    if (
        type(prepared.authorization) is not PortfolioS1FeedbackAuthorizationV7
        or type(prepared.control) is not PortfolioS1FeedbackControlV12
        or type(prepared.launch_lock)
        is not PortfolioS1Qwen38FeedbackLaunchLockV5
        or prepared.authorization.owner_authorized_budget_ceiling_cny
        != "150.000000000000"
        or prepared.authorization.approved_technical_phase_hard_cap_cny
        != "113.000000000000"
        or prepared.authorization.owner_approved_phase_hard_cap_cny
        != "113.000000000000"
    ):
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 execution locks lack exact CNY150 authority with the CNY113 technical stop"
        )
    validate_selected_qwen38_feedback_authorization_v7(
        prepared.authorization, prepared.selection
    )
    validate_portfolio_s1_feedback_control_v12(
        prepared.control, prepared.selection, prepared.authorization
    )
    expected_launch = build_qwen38_feedback_launch_lock_v5(
        run_id=prepared.launch_lock.run_id,
        selection=prepared.selection,
        authorization=prepared.authorization,
        control=prepared.control,
    )
    if prepared.launch_lock != expected_launch:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 launch lock differs from the frozen execution identity"
        )


def execute_run_v3(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    *,
    feedback_runner: FeedbackRunner,
    stop_after_count: Literal[12, 60, 120, 240] | None = None,
) -> tuple[PortfolioS1FeedbackRunV5, PortfolioS1FeedbackBundleV8] | None:
    """Execute the frozen 12, 60, 120, 240 sequence under one writer lock."""

    if stop_after_count not in {None, 12, 60, 120, 240}:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 stop count must be one frozen phase boundary"
        )

    _reject_historical_roots(prepared)
    _require_live_authority(prepared)
    with _exclusive_execute_writer_lock(prepared.output_dir):
        return _execute_run_v3_locked(
            prepared,
            feedback_runner=feedback_runner,
            stop_after_count=stop_after_count,
        )


def _publish_terminal_run_v5(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    *,
    terminal_reason: Literal[
        "usage_limit_exceeded", "accountable_cost_exceeded"
    ] | None = None,
) -> PortfolioS1FeedbackRunV5 | None:
    settled, orphans, claims = _attempt_state_v5(prepared)
    if not settled and not orphans:
        return None
    if terminal_reason is None:
        try:
            _budget_snapshot_v5(settled, orphans)
        except PortfolioS1FeedbackV3BudgetError as error:
            terminal_reason = error.terminal_reason
    run = build_portfolio_s1_feedback_run_v5(
        prepared.selection,
        prepared.control,
        settled,
        retry_claims=claims,
        orphaned_attempts=orphans,
        terminal_reason=terminal_reason,
    )
    _publish_or_resume(
        prepared.output_dir / "run-v5.json",
        run.canonical_bytes(),
        "fresh-v3 terminal run",
    )
    return run


def _resume_final_publication_v3(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    settled: tuple[BoundFeedbackArtifactV5, ...],
    orphans: tuple[FeedbackCallReservationV5, ...],
    claims: tuple[FeedbackGlobalRetryClaimV2, ...],
) -> tuple[PortfolioS1FeedbackRunV5, PortfolioS1FeedbackBundleV8] | None:
    """Finalize or reject an already published run before any new provider call."""

    run_path = prepared.output_dir / "run-v5.json"
    bundle_path = prepared.output_dir / "bundle-v8.json"
    if bundle_path.exists() and not run_path.exists():
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 bundle exists without its run"
        )
    if not run_path.exists():
        return None
    terminal_reason = None
    try:
        _budget_snapshot_v5(settled, orphans)
    except PortfolioS1FeedbackV3BudgetError as error:
        terminal_reason = error.terminal_reason
    run = build_portfolio_s1_feedback_run_v5(
        prepared.selection,
        prepared.control,
        settled,
        retry_claims=claims,
        orphaned_attempts=orphans,
        terminal_reason=terminal_reason,
    )
    _publish_or_resume(run_path, run.canonical_bytes(), "fresh-v3 resumed run")
    if run.status != "completed":
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 terminal run is immutable and cannot resume provider calls"
        )
    bundle = build_portfolio_s1_feedback_bundle_v8(
        prepared.selection,
        prepared.control,
        prepared.authorization,
        settled,
        run,
        retry_claims=claims,
    )
    _publish_or_resume(
        bundle_path,
        bundle.canonical_bytes(),
        "fresh-v3 resumed bundle",
    )
    return run, bundle


def _execute_run_v3_locked(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    *,
    feedback_runner: FeedbackRunner,
    stop_after_count: Literal[12, 60, 120, 240] | None = None,
) -> tuple[PortfolioS1FeedbackRunV5, PortfolioS1FeedbackBundleV8] | None:
    settled, orphans, claims = _attempt_state_v5(prepared)
    resumed = _resume_final_publication_v3(
        prepared, settled, orphans, claims
    )
    if resumed is not None:
        return resumed
    if orphans:
        _publish_terminal_run_v5(prepared)
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 has an orphan reservation and cannot call provider"
        )
    try:
        _budget_snapshot_v5(settled, orphans)
    except PortfolioS1FeedbackV3BudgetError as error:
        _publish_terminal_run_v5(
            prepared, terminal_reason=error.terminal_reason
        )
        raise
    artifacts = {
        (item.selection_entry_sha256, item.attempt_index): item for item in settled
    }
    try:
        for count in (12, 60, 120, 240):
            _execute_phase_v5(
                prepared,
                prepared.sources[:count],
                artifacts,
                feedback_runner,
            )
            phase_entry_sha256s = {
                item.entry_sha256 for item in prepared.selection.entries[:count]
            }
            if any(
                (entry_sha, 1) not in artifacts
                or artifacts.get((entry_sha, 2), artifacts[(entry_sha, 1)]).status
                != "parsed"
                for entry_sha in phase_entry_sha256s
            ):
                raise PortfolioS1FeedbackV3RunError(
                    f"fresh-v3 phase did not complete parsed{count}"
                )
            if stop_after_count == count and count < 240:
                return None
    except PortfolioS1FeedbackV3BudgetError as error:
        _publish_terminal_run_v5(
            prepared, terminal_reason=error.terminal_reason
        )
        raise
    except BaseException:
        _publish_terminal_run_v5(prepared)
        raise
    settled, orphans, claims = _attempt_state_v5(prepared)
    _budget_snapshot_v5(settled, orphans)
    run = build_portfolio_s1_feedback_run_v5(
        prepared.selection,
        prepared.control,
        settled,
        retry_claims=claims,
        orphaned_attempts=orphans,
    )
    if run.status != "completed":
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 cannot publish a noncompleted Feedback bundle"
        )
    bundle = build_portfolio_s1_feedback_bundle_v8(
        prepared.selection,
        prepared.control,
        prepared.authorization,
        settled,
        run,
        retry_claims=claims,
    )
    _publish_or_resume(
        prepared.output_dir / "run-v5.json",
        run.canonical_bytes(),
        "fresh-v3 completed run",
    )
    _publish_or_resume(
        prepared.output_dir / "bundle-v8.json",
        bundle.canonical_bytes(),
        "fresh-v3 completed bundle",
    )
    return run, bundle


def prepare_run_v3(
    arguments: argparse.Namespace,
) -> PreparedPortfolioS1FeedbackV3Run:
    """Build or exact-resume the fresh-v3 pre-provider artifact chain."""

    expected_governance = (
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2,
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12,
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12,
    )
    supplied_governance = (
        arguments.expected_model_source_lock_sha256,
        arguments.expected_pricing_lock_sha256,
        arguments.expected_role_selection_file_sha256,
        arguments.expected_role_selection_sha256,
    )
    if supplied_governance != expected_governance:
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 requires source-v2, pricing-v5, and role-v12 exact identities"
        )
    if arguments.owner_authorized_budget_ceiling_cny != "150.000000000000":
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 owner-authorized budget ceiling must be exactly CNY150"
        )
    if arguments.approved_technical_phase_hard_cap_cny != "113.000000000000":
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 technical execution hard cap must remain exactly CNY113"
        )
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=arguments.expected_execution_control_sha256,
        artifact_repository_root=arguments.artifact_repository_root,
    )
    discovery_ids, discovery_sha256, mapping_sha256 = _load_discovery600(
        arguments.fold_manifest,
        arguments.fold_mapping,
        expected_manifest_sha256=arguments.expected_fold_manifest_sha256,
        expected_mapping_sha256=arguments.expected_fold_mapping_sha256,
    )
    selection = build_portfolio_s1_feedback_selection_v2(
        corpus,
        discovery_ids,
        discovery_query_ids_sha256=discovery_sha256,
        fold_mapping_sha256=mapping_sha256,
        seed=SELECTION_SEED,
    )
    source_lock = load_qwen38_feedback_model_source_lock_v2(
        arguments.model_source_lock,
        expected_file_sha256=arguments.expected_model_source_lock_sha256,
    )
    pricing_lock = load_qwen38_feedback_pricing_lock_v5(
        arguments.pricing_lock,
        expected_file_sha256=arguments.expected_pricing_lock_sha256,
    )
    role_selection = load_qwen38_feedback_role_selection_v12(
        arguments.role_selection_file,
        expected_file_sha256=arguments.expected_role_selection_file_sha256,
    )
    if (
        source_lock.source_lock_sha256 != QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2
        or pricing_lock.pricing_lock_sha256
        != QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5
        or role_selection.selection_sha256
        != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12
        or source_lock.endpoint != config.PROVIDER_ENDPOINTS["qwen"]
    ):
        raise PortfolioS1FeedbackV3RunError(
            "fresh-v3 loaded governance or provider endpoint drifted"
        )
    _require_live_governance_locks_v3(
        source_lock, pricing_lock, role_selection
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for(
        "dashscope-qwen-assistant"
    )
    authorization = build_selected_qwen38_feedback_authorization_v7(
        selection,
        parent_remote_runtime,
        authorization_id=arguments.authorization_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=_parse_reviewed_at_v3(arguments.reviewed_at),
        owner_authorized_budget_ceiling_cny=(
            arguments.owner_authorized_budget_ceiling_cny
        ),
        approved_technical_phase_hard_cap_cny=(
            arguments.approved_technical_phase_hard_cap_cny
        ),
        model_source_lock_file_sha256=(
            arguments.expected_model_source_lock_sha256
        ),
        model_source_lock_sha256=source_lock.source_lock_sha256,
        pricing_lock_file_sha256=arguments.expected_pricing_lock_sha256,
        pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
        role_selection_file_sha256=(
            arguments.expected_role_selection_file_sha256
        ),
        role_selection_sha256=role_selection.selection_sha256,
    )
    rubric: RubricSnapshot = _rubric(
        arguments.rubric_file,
        expected_sha256=arguments.expected_rubric_sha256,
        rubric_id=arguments.rubric_id,
        rubric_version=arguments.rubric_version,
    )
    control = build_portfolio_s1_feedback_control_v12(
        selection, authorization, rubric=rubric
    )
    output = arguments.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    _reject_historical_output_dir(output)
    _publish_or_resume(
        output / "selection-v2.json",
        selection.canonical_bytes(),
        "fresh-v3 selection",
    )
    _publish_or_resume(
        output / "authorization-v7.json",
        authorization.canonical_bytes(),
        "fresh-v3 authorization",
    )
    _publish_or_resume(
        output / "control-v12.json",
        control.canonical_bytes(),
        "fresh-v3 control",
    )
    launch_lock = build_qwen38_feedback_launch_lock_v5(
        run_id=arguments.run_id,
        selection=selection,
        authorization=authorization,
        control=control,
    )
    _publish_or_resume(
        output / "launch-lock-v5.json",
        launch_lock.canonical_bytes(),
        "fresh-v3 launch lock",
    )
    sources = build_verified_static_feedback_sources_v2(
        corpus, selection, control
    )
    _validate_input_reservation_bounds(sources)
    remote_runtime = prepare_selected_qwen38_feedback_remote_runtime_v7(
        corpus,
        selection,
        authorization,
        control,
        parent_remote_runtime,
        receipt_path=output / "remote-runtime-receipt-v7.json",
        verified_sources=sources,
    )
    return PreparedPortfolioS1FeedbackV3Run(
        output_dir=output,
        selection=selection,
        authorization=authorization,
        control=control,
        sources=sources,
        remote_runtime=remote_runtime,
        launch_lock=launch_lock,
        model_source_lock=source_lock,
        pricing_lock=pricing_lock,
        role_selection=role_selection,
    )


def _require_qwen_execute_environment_v3() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackV3RunError(
            "Qwen Feedback credential is absent; no fresh-v3 attempt was reserved"
        )


def _live_feedback_runner_v3(
    prepared: PreparedPortfolioS1FeedbackV3Run,
) -> FeedbackRunner:
    isolation = make_active_portfolio_evaluator_isolation_lock()

    def invoke(
        source: VerifiedStaticFeedbackSourceV2, _attempt_index: int
    ) -> FeedbackEvaluationResult:
        return run_visual_feedback(
            source.packet,
            isolation,
            remote_runtime=prepared.remote_runtime,
            max_tokens=None,
            max_completion_tokens=6144,
            timeout_seconds=600,
            record_usage=True,
        )

    return invoke


def execute_live_run_v3(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    *,
    stop_after_count: Literal[12, 60, 120, 240] | None = None,
) -> tuple[PortfolioS1FeedbackRunV5, PortfolioS1FeedbackBundleV8] | None:
    """Production provider entry point; fake-client tests use ``execute_run_v3``."""

    return execute_run_v3(
        prepared,
        feedback_runner=_live_feedback_runner_v3(prepared),
        stop_after_count=stop_after_count,
    )


def _summary_v3(
    mode: Literal["prepare", "dry-run", "execute"],
    prepared: PreparedPortfolioS1FeedbackV3Run,
    outcome: tuple[PortfolioS1FeedbackRunV5, PortfolioS1FeedbackBundleV8]
    | None,
    *,
    stop_after_count: int | None,
) -> dict[str, object]:
    settled, orphans, claims = _attempt_state_v5(prepared)
    budget = _budget_snapshot_v5(settled, orphans)
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-v3-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_dry_run": 0 if mode != "execute" else None,
        "selected_count": 240,
        "phase_counts": [12, 60, 120, 240],
        "feedback_concurrency": 2,
        "internal_attempts_per_provider_call": 1,
        "max_attempts_per_retried_query": 2,
        "global_retry_token_count": 3,
        "global_retry_claimed_count": len(claims),
        "provider_call_ceiling": 243,
        "provider_attempt_reservation_count": budget.provider_calls_reserved,
        "retry_attempt_reservation_count": sum(
            item.attempt_index == 2 for item in (*settled, *orphans)
        ),
        "usage_known_settlement_count": budget.usage_known_count,
        "usage_unknown_or_unresolved_count": budget.usage_unknown_count,
        "input_tokens": budget.input_tokens,
        "output_tokens_including_reasoning": budget.output_tokens,
        "settled_actual_cost_cny": format(
            budget.settled_actual_cost_cny.quantize(Decimal("0.000000000001")),
            "f",
        ),
        "unknown_or_unresolved_reserve_cny": format(
            budget.unresolved_reserve_cny.quantize(Decimal("0.000000000001")),
            "f",
        ),
        "accountable_cost_cny": format(
            budget.accountable_cost_cny.quantize(Decimal("0.000000000001")),
            "f",
        ),
        "technical_phase_hard_cap_cny": (
            QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2
        ),
        "owner_authorized_budget_ceiling_cny": "150.000000000000",
        "owner_budget_authority_statement": FRESH_V3_OWNER_BUDGET_STATEMENT,
        "stop_after_count": stop_after_count,
        "resumable_settlement_count": len(settled),
        "selection_sha256": prepared.selection.selection_sha256,
        "control_sha256": prepared.control.control_sha256,
        "run_sha256": None if outcome is None else outcome[0].run_sha256,
        "bundle_sha256": None if outcome is None else outcome[1].bundle_sha256,
    }


def build_parser_v3() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "dry-run", "execute"), required=True
    )
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--expected-fold-manifest-sha256", required=True)
    parser.add_argument("--fold-mapping", type=Path, required=True)
    parser.add_argument("--expected-fold-mapping-sha256", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--expected-rubric-sha256", required=True)
    parser.add_argument("--rubric-id", required=True)
    parser.add_argument("--rubric-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--owner-authorized-budget-ceiling-cny", required=True)
    parser.add_argument("--approved-technical-phase-hard-cap-cny", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-source-lock", type=Path, required=True)
    parser.add_argument("--expected-model-source-lock-sha256", required=True)
    parser.add_argument("--pricing-lock", type=Path, required=True)
    parser.add_argument("--expected-pricing-lock-sha256", required=True)
    parser.add_argument("--role-selection-file", type=Path, required=True)
    parser.add_argument("--expected-role-selection-file-sha256", required=True)
    parser.add_argument("--expected-role-selection-sha256", required=True)
    parser.add_argument(
        "--stop-after-count",
        type=int,
        choices=(12, 60, 120, 240),
        help="execute/resume through one frozen cumulative phase boundary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser_v3().parse_args(argv)
    try:
        if arguments.stop_after_count is not None and arguments.mode != "execute":
            raise PortfolioS1FeedbackV3RunError(
                "--stop-after-count is only valid with --mode execute"
            )
        prepared = prepare_run_v3(arguments)
        outcome = None
        if arguments.mode == "dry-run":
            _attempt_state_v5(prepared)
        elif arguments.mode == "execute":
            outcome = execute_live_run_v3(
                prepared, stop_after_count=arguments.stop_after_count
            )
        print(
            canonical_json_bytes(
                _summary_v3(
                    arguments.mode,
                    prepared,
                    outcome,
                    stop_after_count=arguments.stop_after_count,
                )
            ).decode("utf-8"),
            end="",
        )
        return 0
    except (
        PortfolioS1FeedbackError,
        PortfolioS1FeedbackV3RunError,
        PortfolioS1QwenFeedbackGovernanceError,
        RuntimeError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2


__all__ = [
    "FeedbackBudgetSnapshotV5",
    "PreparedPortfolioS1FeedbackV3Run",
    "PortfolioS1FeedbackV3BudgetError",
    "PortfolioS1FeedbackV3RunError",
    "_artifact_path_v5",
    "_attempt_state_v5",
    "_budget_snapshot_v5",
    "_claim_retry_v2",
    "_execute_phase_v5",
    "_reject_historical_roots",
    "_reservation_path_v5",
    "_run_source_v5",
    "_require_live_governance_locks_v3",
    "_require_live_authority",
    "build_parser_v3",
    "execute_live_run_v3",
    "execute_run_v3",
    "main",
    "prepare_run_v3",
    "_execute_run_v3_locked",
    "_publish_terminal_run_v5",
    "_parse_reviewed_at_v3",
]


if __name__ == "__main__":
    raise SystemExit(main())
