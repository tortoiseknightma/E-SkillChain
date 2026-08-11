"""Prepare, dry-run, or execute the selected48 Portfolio S1 Feedback run.

No command in this script mutates the historical Core DashScope owner
authorization.  A new exact selected48 authorization is required explicitly.
Execution is fixed to a six-capability canary followed by the remaining 42
rows, with concurrency two, one attempt per row, and create-only resume.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Callable, Iterator, Literal

from skillchain import config
from skillchain.evaluation.evaluator_isolation import (
    EvaluatorIsolationLock,
    build_bound_feedback_prompt,
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    run_visual_feedback,
    visual_feedback_response_format_v1,
)
from skillchain.evaluation.packets import RubricSnapshot
from skillchain.evaluation.portfolio_s1_feedback import (
    BoundFeedbackArtifact,
    FeedbackCallReservationV1,
    PortfolioS1FeedbackAuthorizationV2,
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackBundleV4,
    PortfolioS1FeedbackControlV1,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackRunV1,
    PortfolioS1FeedbackSelectionV1,
    VerifiedStaticFeedbackSource,
    build_bound_feedback_artifact,
    build_feedback_call_reservation,
    build_portfolio_s1_feedback_bundle,
    build_portfolio_s1_feedback_bundle_v4,
    build_portfolio_s1_feedback_control,
    build_portfolio_s1_feedback_run,
    build_portfolio_s1_feedback_selection,
    build_verified_static_feedback_sources,
    load_portfolio_s1_feedback_bundle,
    load_portfolio_s1_feedback_bundle_v4,
    load_feedback_call_reservation,
    load_portfolio_s1_feedback_run,
    resume_bound_feedback_artifact,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    prepare_selected_qwen_feedback_remote_runtime,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackGovernanceError,
    PortfolioS1QwenFeedbackAuthorizationV3,
    PortfolioS1QwenFeedbackLaunchLockV1,
    QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS,
    QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS,
    QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS,
    QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS,
    QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY,
    QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY,
    Qwen37FeedbackModelSourceLockV1,
    Qwen37FeedbackPricingLockV1,
    build_qwen37_feedback_launch_lock,
    build_selected_qwen_feedback_authorization,
    load_qwen37_feedback_model_source_lock,
    load_qwen37_feedback_pricing_lock,
    require_qwen37_feedback_pre_call_budget,
)
from skillchain.evaluation.portfolio_static_gcs_corpus import (
    VerifiedStaticGCSCorpus,
    load_verified_static_gcs_corpus,
)
from skillchain.evaluation.visual_runtime import (
    VerifiedSelectedFeedbackRemoteRuntime,
)
from skillchain.synthesis.portfolio_core_selection import (
    CreatorSelectionEntry,
    canonical_creator_selection_index_bytes,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import (
    parse_canonical_jsonl,
    read_stable_regular_file,
)


SELECTION_SEED = 20260808


class PortfolioS1FeedbackRunError(RuntimeError):
    """Preparation, resume, or execution failed closed."""


@dataclass(frozen=True)
class PreparedPortfolioS1FeedbackRun:
    output_dir: Path
    corpus: VerifiedStaticGCSCorpus
    selection: PortfolioS1FeedbackSelectionV1
    authorization: (
        PortfolioS1FeedbackAuthorizationV2 | PortfolioS1QwenFeedbackAuthorizationV3
    )
    control: PortfolioS1FeedbackControlV1
    sources: tuple[VerifiedStaticFeedbackSource, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime
    launch_lock: PortfolioS1QwenFeedbackLaunchLockV1 | None = None
    model_source_lock: Qwen37FeedbackModelSourceLockV1 | None = None
    pricing_lock: Qwen37FeedbackPricingLockV1 | None = None


FeedbackRunner = Callable[..., FeedbackEvaluationResult]
_BUDGET_RESERVATION_LOCK = Lock()
_EXECUTE_WRITER_LOCK_FILE = "execute-writer.lock"
_CNY_SCALE = Decimal("0.000000000001")
# Qwen3.7 defaults to at most 2,621,440 pixels when high-resolution mode is
# omitted.  The documented formula is max_pixels / (32x32) + 2, yielding
# 2,562 image tokens.
_QWEN37_DEFAULT_IMAGE_TOKEN_UPPER_BOUND = 2_562
_QWEN37_MESSAGE_FRAMING_TOKEN_UPPER_BOUND = 512


@dataclass(frozen=True)
class _FeedbackBudgetSnapshot:
    provider_calls_reserved: int
    settled_usage_known: int
    usage_unknown_or_unresolved: int
    input_tokens: int
    output_tokens: int
    settled_actual_cost_cny: Decimal
    unknown_or_unresolved_reserve_cny: Decimal
    accountable_cost_cny: Decimal


def _cny(value: Decimal) -> str:
    return format(value.quantize(_CNY_SCALE), "f")


def _estimated_input_tokens(
    source: VerifiedStaticFeedbackSource,
    evaluator_isolation: EvaluatorIsolationLock,
) -> int:
    """Return a deterministic conservative bound for the exact request wire."""

    bound = build_bound_feedback_prompt(source.packet, evaluator_isolation)
    visible_text_bytes = sum(
        len(message.content.encode("utf-8")) for message in bound.messages
    )
    response_schema_bytes = len(
        canonical_json_bytes(
            visual_feedback_response_format_v1().model_dump(mode="json", by_alias=True)
        )
    )
    # UTF-8 byte count upper-bounds text tokens for the provider tokenizer;
    # include the full response schema and a fixed message-framing allowance.
    return (
        visible_text_bytes
        + response_schema_bytes
        + _QWEN37_DEFAULT_IMAGE_TOKEN_UPPER_BOUND
        + _QWEN37_MESSAGE_FRAMING_TOKEN_UPPER_BOUND
    )


def _validate_input_reservation_bounds(
    sources: tuple[VerifiedStaticFeedbackSource, ...],
) -> None:
    isolation = make_active_portfolio_evaluator_isolation_lock()
    estimates = tuple(_estimated_input_tokens(source, isolation) for source in sources)
    if not estimates or max(estimates) > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS:
        raise PortfolioS1FeedbackRunError(
            "Qwen Feedback request exceeds the frozen input-token reservation"
        )


def _require_frozen_qwen_endpoint(
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
) -> None:
    if config.PROVIDER_ENDPOINTS["qwen"] != model_source_lock.endpoint:
        raise PortfolioS1FeedbackRunError(
            "runtime Qwen endpoint differs from the frozen model source lock"
        )


def _require_qwen_execute_environment() -> None:
    key_name = config.PROVIDER_API_KEY_ENV["qwen"]
    if not os.environ.get(key_name, "").strip():
        raise PortfolioS1FeedbackRunError(
            "Qwen Feedback API credential is absent; no provider attempt was reserved"
        )


@contextmanager
def _exclusive_execute_writer_lock(output_dir: Path) -> Iterator[None]:
    """Permit only one provider-writing process for a frozen run identity.

    A normally exiting process removes its lock.  A killed process leaves the
    create-only lock behind so an operator must inspect the recorded PID before
    any resume; the immutable per-call reservations remain the exactly-once
    authority either way.
    """

    path = output_dir / _EXECUTE_WRITER_LOCK_FILE
    owner = canonical_json_bytes(
        {"owner_pid": os.getpid(), "owner_token": os.urandom(16).hex()}
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise PortfolioS1FeedbackRunError(
            "Feedback execute writer lock already exists; inspect the recorded "
            "owner before any explicit recovery"
        ) from error
    except OSError as error:
        raise PortfolioS1FeedbackRunError(
            "cannot acquire Feedback execute writer lock"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(owner)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        observed = read_stable_regular_file(
            path, label="Feedback execute writer lock", max_bytes=1024
        )
        if observed != owner:
            raise PortfolioS1FeedbackRunError(
                "Feedback execute writer lock ownership changed; preserving it"
            )
        try:
            path.unlink()
        except OSError as error:
            raise PortfolioS1FeedbackRunError(
                "cannot release Feedback execute writer lock; preserving it"
            ) from error


def _require_file_sha(path: Path, expected: str, label: str) -> bytes:
    content = read_stable_regular_file(path, label=label, max_bytes=16 * 1024 * 1024)
    if sha256_bytes(content) != expected:
        raise PortfolioS1FeedbackRunError(f"{label} SHA-256 mismatch")
    return content


def _load_creator_entries(
    path: Path, expected_sha256: str
) -> tuple[CreatorSelectionEntry, ...]:
    content = _require_file_sha(path, expected_sha256, "Creator240 index")
    try:
        rows = parse_canonical_jsonl(content, label="Creator240 index")
        entries = tuple(
            CreatorSelectionEntry.model_validate(row, strict=True) for row in rows
        )
    except ValueError as error:
        raise PortfolioS1FeedbackRunError("Creator240 index is invalid") from error
    if (
        len(entries) != 240
        or canonical_creator_selection_index_bytes(entries) != content
        or tuple(item.selection_rank for item in entries) != tuple(range(1, 241))
    ):
        raise PortfolioS1FeedbackRunError(
            "Creator240 index is not the exact canonical 240-row contract"
        )
    return entries


def _rubric(
    path: Path,
    *,
    expected_sha256: str,
    rubric_id: str,
    rubric_version: str,
) -> RubricSnapshot:
    content = _require_file_sha(path, expected_sha256, "Feedback rubric")
    try:
        text = content.decode("utf-8")
    except UnicodeError as error:
        raise PortfolioS1FeedbackRunError("Feedback rubric is not UTF-8") from error
    if text.endswith("\n") and not text.endswith("\n\n"):
        text = text[:-1]
    if not text or text != text.strip():
        raise PortfolioS1FeedbackRunError(
            "Feedback rubric must be nonblank canonical trimmed UTF-8"
        )
    return RubricSnapshot(
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        content=text,
        content_sha256=sha256_bytes(text.encode("utf-8")),
    )


def _reviewed_at(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise PortfolioS1FeedbackRunError("reviewed_at is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PortfolioS1FeedbackRunError("reviewed_at requires a timezone")
    return parsed


def _load_qwen_role_selection(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_selection_sha256: str,
) -> tuple[str, str]:
    content = _require_file_sha(path, expected_file_sha256, "model role selection")
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PortfolioS1FeedbackRunError(
            "model role selection is invalid JSON"
        ) from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise PortfolioS1FeedbackRunError("model role selection is not canonical JSON")
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }
    feedback = payload.get("feedback_evaluator")
    if (
        payload.get("schema_version") != 8
        or payload.get("selection_sha256") != expected_selection_sha256
        or sha256_bytes(canonical_json_bytes(unsigned)) != expected_selection_sha256
        or not isinstance(feedback, dict)
        or feedback.get("provider") != "qwen"
        or feedback.get("model") != "qwen3.7-plus-2026-05-26"
        or feedback.get("processor") != "dashscope-qwen37-feedback"
        or feedback.get("requested_response_format") != "json_schema"
        or feedback.get("requested_json_schema_strict") is not True
        or feedback.get("enable_thinking") is not True
        or feedback.get("thinking_budget") != 2048
        or feedback.get("max_tokens") is not None
        or feedback.get("max_completion_tokens") != 4096
        or feedback.get("timeout_seconds") != 600
    ):
        raise PortfolioS1FeedbackRunError(
            "model role selection differs from active Qwen3.7 Feedback"
        )
    return expected_file_sha256, expected_selection_sha256


def _publish_or_resume(path: Path, content: bytes, label: str) -> None:
    if path.exists():
        observed = read_stable_regular_file(
            path, label=label, max_bytes=32 * 1024 * 1024
        )
        if observed != content:
            raise PortfolioS1FeedbackRunError(f"{label} resume conflict")
        return
    atomic_create_file(path, content)


def prepare_run(arguments: argparse.Namespace) -> PreparedPortfolioS1FeedbackRun:
    corpus = load_verified_static_gcs_corpus(
        arguments.execution_root,
        expected_control_file_sha256=arguments.expected_execution_control_sha256,
        artifact_repository_root=arguments.artifact_repository_root,
    )
    creator_entries = _load_creator_entries(
        arguments.creator_selection_index,
        arguments.expected_creator_selection_sha256,
    )
    selection = build_portfolio_s1_feedback_selection(
        corpus,
        creator_entries,
        creator_selection_file_sha256=(arguments.expected_creator_selection_sha256),
        creator_selection_sha256=arguments.expected_creator_selection_sha256,
        seed=SELECTION_SEED,
    )
    model_source_lock = load_qwen37_feedback_model_source_lock(
        arguments.model_source_lock,
        expected_file_sha256=arguments.expected_model_source_lock_sha256,
    )
    _require_frozen_qwen_endpoint(model_source_lock)
    pricing_lock = load_qwen37_feedback_pricing_lock(
        arguments.pricing_lock,
        expected_file_sha256=arguments.expected_pricing_lock_sha256,
    )
    role_selection_file_sha256, role_selection_sha256 = _load_qwen_role_selection(
        arguments.role_selection_file,
        expected_file_sha256=arguments.expected_role_selection_file_sha256,
        expected_selection_sha256=arguments.expected_role_selection_sha256,
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for("dashscope-qwen-assistant")
    authorization = build_selected_qwen_feedback_authorization(
        selection,
        parent_remote_runtime,
        authorization_id=arguments.authorization_id,
        reviewer_id=arguments.reviewer_id,
        reviewed_at=_reviewed_at(arguments.reviewed_at),
        owner_statement=arguments.owner_statement,
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=(arguments.expected_model_source_lock_sha256),
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=arguments.expected_pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
    )
    control = build_portfolio_s1_feedback_control(
        selection,
        authorization,
        rubric=_rubric(
            arguments.rubric_file,
            expected_sha256=arguments.expected_rubric_sha256,
            rubric_id=arguments.rubric_id,
            rubric_version=arguments.rubric_version,
        ),
        feedback_concurrency=2,
    )
    output = arguments.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    _publish_or_resume(
        output / "selection.json", selection.canonical_bytes(), "Feedback selection"
    )
    _publish_or_resume(
        output / "selected-assets-authorization.json",
        authorization.canonical_bytes(),
        "selected Feedback authorization",
    )
    _publish_or_resume(
        output / "control.json", control.canonical_bytes(), "Feedback control"
    )
    launch_lock = build_qwen37_feedback_launch_lock(
        run_id=arguments.run_id,
        selection_sha256=selection.selection_sha256,
        corpus_sha256=selection.corpus_sha256,
        authorization=authorization,
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=arguments.expected_model_source_lock_sha256,
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=arguments.expected_pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        control_file_sha256=sha256_bytes(control.canonical_bytes()),
        control_sha256=control.control_sha256,
    )
    _publish_or_resume(
        output / "launch-lock.json",
        launch_lock.canonical_bytes(),
        "Qwen Feedback launch lock",
    )
    sources = build_verified_static_feedback_sources(corpus, selection, control)
    # Prove every frozen request fits the priced input reservation before
    # publishing a remote receipt or permitting any provider attempt.
    _validate_input_reservation_bounds(sources)
    remote_runtime = prepare_selected_qwen_feedback_remote_runtime(
        corpus,
        selection,
        authorization,
        control,
        parent_remote_runtime,
        receipt_path=output / "remote-runtime-receipt.json",
        verified_sources=sources,
    )
    return PreparedPortfolioS1FeedbackRun(
        output_dir=output,
        corpus=corpus,
        selection=selection,
        authorization=authorization,
        control=control,
        sources=sources,
        remote_runtime=remote_runtime,
        launch_lock=launch_lock,
        model_source_lock=model_source_lock,
        pricing_lock=pricing_lock,
    )


def _artifact_path(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSource,
) -> Path:
    return (
        prepared.output_dir
        / "bound-feedback"
        / (f"{source.row.query_ordinal:04d}-{source.selection_entry_sha256[:16]}.json")
    )


def _reservation_path(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSource,
) -> Path:
    return (
        prepared.output_dir
        / "provider-attempts"
        / (f"{source.row.query_ordinal:04d}-{source.selection_entry_sha256[:16]}.json")
    )


def _load_reservation(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSource,
) -> FeedbackCallReservationV1 | None:
    path = _reservation_path(prepared, source)
    if not path.exists():
        return None
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    reservation = load_feedback_call_reservation(
        path,
        expected_selection_sha256=prepared.selection.selection_sha256,
        expected_control_sha256=prepared.control.control_sha256,
        expected_entry_sha256=entry.entry_sha256,
    )
    expected = build_feedback_call_reservation(
        prepared.selection,
        prepared.control,
        entry,
        verified_source=source,
    )
    if reservation != expected:
        raise PortfolioS1FeedbackRunError("Feedback call reservation source conflict")
    return reservation


def _budget_snapshot(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> _FeedbackBudgetSnapshot:
    """Rebuild accountable Qwen cost from create-only attempts and settlements."""

    reserved = 0
    known = 0
    unknown = 0
    input_tokens = 0
    output_tokens = 0
    settled_actual = Decimal(0)
    unknown_reserve = Decimal(0)
    fixed_reserve = Decimal(QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY)
    per_million = Decimal(1_000_000)
    input_rate = Decimal(QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS)
    output_rate = Decimal(QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS)

    for entry, source in zip(prepared.selection.entries, prepared.sources, strict=True):
        reservation = _load_reservation(prepared, source)
        if reservation is None:
            continue
        reserved += 1
        artifact = resume_bound_feedback_artifact(
            _artifact_path(prepared, source),
            selection=prepared.selection,
            control=prepared.control,
            entry=entry,
            verified_source=source,
            reservation=reservation,
        )
        usage = None if artifact is None else artifact.feedback_result.usage
        if usage is None:
            unknown += 1
            unknown_reserve += fixed_reserve
            continue
        if (
            usage.input_tokens > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
            or usage.output_tokens > QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
        ):
            raise PortfolioS1FeedbackRunError(
                "Qwen Feedback provider usage exceeded the frozen reservation"
            )
        known += 1
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        settled_actual += (
            Decimal(usage.input_tokens) * input_rate
            + Decimal(usage.output_tokens) * output_rate
        ) / per_million

    accountable = settled_actual + unknown_reserve
    if accountable > Decimal(QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY):
        raise PortfolioS1FeedbackRunError(
            "Qwen Feedback accountable cost exceeds the frozen phase hard cap"
        )
    return _FeedbackBudgetSnapshot(
        provider_calls_reserved=reserved,
        settled_usage_known=known,
        usage_unknown_or_unresolved=unknown,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        settled_actual_cost_cny=settled_actual,
        unknown_or_unresolved_reserve_cny=unknown_reserve,
        accountable_cost_cny=accountable,
    )


def _verify_attempt_file_set(prepared: PreparedPortfolioS1FeedbackRun) -> None:
    expected_reservations = {
        _reservation_path(prepared, source).name for source in prepared.sources
    }
    expected_results = {
        _artifact_path(prepared, source).name for source in prepared.sources
    }
    for directory, expected, label in (
        (
            prepared.output_dir / "provider-attempts",
            expected_reservations,
            "Feedback reservation",
        ),
        (
            prepared.output_dir / "bound-feedback",
            expected_results,
            "Feedback settlement",
        ),
    ):
        if not directory.exists():
            continue
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if (
            not actual <= expected
            or len(actual) > prepared.control.provider_call_ceiling
        ):
            raise PortfolioS1FeedbackRunError(
                f"{label} file set exceeds the frozen selected48 ceiling"
            )


def _resume_all_before_provider(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> dict[str, BoundFeedbackArtifact]:
    _verify_attempt_file_set(prepared)
    resumed: dict[str, BoundFeedbackArtifact] = {}
    for entry, source in zip(prepared.selection.entries, prepared.sources, strict=True):
        reservation = _load_reservation(prepared, source)
        artifact_exists = _artifact_path(prepared, source).exists()
        if artifact_exists and reservation is None:
            raise PortfolioS1FeedbackRunError(
                "Feedback settlement exists without its provider reservation"
            )
        if reservation is None:
            continue
        artifact = resume_bound_feedback_artifact(
            _artifact_path(prepared, source),
            selection=prepared.selection,
            control=prepared.control,
            entry=entry,
            verified_source=source,
            reservation=reservation,
        )
        if artifact is None:
            raise PortfolioS1FeedbackRunError(
                "orphan Feedback provider reservation forbids a replacement call"
            )
        resumed[entry.entry_sha256] = artifact
    run_path = prepared.output_dir / "run.json"
    if run_path.exists():
        run = load_portfolio_s1_feedback_run(
            run_path,
            expected_file_sha256=sha256_bytes(
                read_stable_regular_file(
                    run_path, label="Feedback run", max_bytes=4 * 1024 * 1024
                )
            ),
        )
        expected = build_portfolio_s1_feedback_run(
            prepared.selection,
            prepared.control,
            tuple(
                resumed[item.entry_sha256]
                for item in prepared.selection.entries
                if item.entry_sha256 in resumed
            ),
        )
        if run != expected:
            raise PortfolioS1FeedbackRunError("Feedback run resume conflict")
        if run.status == "stopped_nonparsed":
            raise PortfolioS1FeedbackRunError(
                "Feedback run previously stopped on a nonparsed fixed sample"
            )
    nonparsed = [item for item in resumed.values() if item.status != "parsed"]
    if nonparsed:
        if not run_path.exists():
            _publish_terminal_run(prepared, resumed)
        raise PortfolioS1FeedbackRunError(
            "existing nonparsed Feedback artifact stops the fixed run"
        )
    _budget_snapshot(prepared)
    return resumed


def _publish_terminal_run(
    prepared: PreparedPortfolioS1FeedbackRun,
    artifacts: dict[str, BoundFeedbackArtifact],
) -> PortfolioS1FeedbackRunV1:
    ordered = tuple(
        artifacts[item.entry_sha256]
        for item in prepared.selection.entries
        if item.entry_sha256 in artifacts
    )
    run = build_portfolio_s1_feedback_run(prepared.selection, prepared.control, ordered)
    _publish_or_resume(
        prepared.output_dir / "run.json", run.canonical_bytes(), "Feedback run"
    )
    return run


def _run_source(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSource,
    feedback_runner: FeedbackRunner,
) -> BoundFeedbackArtifact:
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    reservation = build_feedback_call_reservation(
        prepared.selection,
        prepared.control,
        entry,
        verified_source=source,
    )
    evaluator_isolation = make_active_portfolio_evaluator_isolation_lock()
    estimated_input_tokens = _estimated_input_tokens(source, evaluator_isolation)
    # Two workers may enter together.  Serialize the read-verify-reserve step so
    # both the call ceiling and the phase cap are enforced on persisted state.
    with _BUDGET_RESERVATION_LOCK:
        budget = _budget_snapshot(prepared)
        reserved_cost = require_qwen37_feedback_pre_call_budget(
            estimated_input_tokens_including_images=estimated_input_tokens,
            provider_calls_already_reserved=budget.provider_calls_reserved,
            committed_cost_cny=_cny(budget.accountable_cost_cny),
        )
        if reserved_cost != QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY:
            raise PortfolioS1FeedbackRunError(
                "Qwen Feedback pre-call reservation amount drifted"
            )
        try:
            atomic_create_file(
                _reservation_path(prepared, source), reservation.canonical_bytes()
            )
        except FileExistsError as error:
            raise PortfolioS1FeedbackRunError(
                "Feedback provider call reservation already exists; refusing "
                "a concurrent or replacement call"
            ) from error
    result = feedback_runner(
        source.packet,
        evaluator_isolation,
        remote_runtime=prepared.remote_runtime,
        max_tokens=prepared.control.max_tokens,
        max_completion_tokens=prepared.control.max_completion_tokens,
        timeout_seconds=prepared.control.requested_timeout_seconds or 600,
        record_usage=True,
    )
    if (
        result.asset_catalog_sha256 != prepared.remote_runtime.catalog.catalog_sha256
        or result.remote_authorization_id != prepared.authorization.authorization_id
        or result.remote_authorization_file_sha256
        != prepared.control.authorization_file_sha256
        or result.remote_receipt_file_sha256
        != prepared.remote_runtime.receipt_file_sha256
        or result.remote_receipt_sha256
        != prepared.remote_runtime.receipt.receipt_sha256
    ):
        raise PortfolioS1FeedbackRunError(
            "Feedback provider result differs from the selected48 remote runtime"
        )
    artifact = build_bound_feedback_artifact(
        prepared.selection,
        prepared.control,
        entry,
        source.packet,
        result,
        verified_source=source,
        reservation=reservation,
    )
    _publish_or_resume(
        _artifact_path(prepared, source),
        artifact.canonical_bytes(),
        "bound Feedback artifact",
    )
    # The bound artifact is the immutable settlement.  Rebuild the budget now
    # so an over-ceiling provider usage cannot be followed by another call.
    _budget_snapshot(prepared)
    return artifact


def _execute_phase(
    prepared: PreparedPortfolioS1FeedbackRun,
    sources: tuple[VerifiedStaticFeedbackSource, ...],
    artifacts: dict[str, BoundFeedbackArtifact],
    feedback_runner: FeedbackRunner,
) -> None:
    pending = tuple(
        item for item in sources if item.selection_entry_sha256 not in artifacts
    )
    for offset in range(0, len(pending), 2):
        wave = pending[offset : offset + 2]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(_run_source, prepared, source, feedback_runner)
                for source in wave
            )
            completed = tuple(future.result() for future in futures)
        for artifact in completed:
            artifacts[artifact.selection_entry_sha256] = artifact
        _budget_snapshot(prepared)
        if any(item.status != "parsed" for item in completed):
            _publish_terminal_run(prepared, artifacts)
            raise PortfolioS1FeedbackRunError(
                "Feedback stopped on a nonparsed fixed sample; no replacement allowed"
            )


def _execute_run_locked(
    prepared: PreparedPortfolioS1FeedbackRun,
    *,
    feedback_runner: FeedbackRunner = run_visual_feedback,
    stop_after_canary: bool = False,
) -> PortfolioS1FeedbackBundleV1 | PortfolioS1FeedbackBundleV4 | None:
    if type(prepared.authorization) is PortfolioS1QwenFeedbackAuthorizationV3:
        _require_qwen_execute_environment()
    artifacts = _resume_all_before_provider(prepared)
    by_hash = {item.selection_entry_sha256: item for item in prepared.sources}
    canary = tuple(by_hash[item] for item in prepared.selection.canary_entry_sha256s)
    remaining = tuple(
        by_hash[item] for item in prepared.selection.remaining_entry_sha256s
    )
    _execute_phase(prepared, canary, artifacts, feedback_runner)
    if any(
        item.entry_sha256 not in artifacts
        for item in prepared.selection.entries
        if item.entry_sha256 in prepared.selection.canary_entry_sha256s
    ):
        raise PortfolioS1FeedbackRunError("Feedback canary did not complete parsed6")
    if stop_after_canary:
        return None
    _execute_phase(prepared, remaining, artifacts, feedback_runner)
    ordered = tuple(artifacts[item.entry_sha256] for item in prepared.selection.entries)
    run = _publish_terminal_run(prepared, artifacts)
    if type(prepared.authorization) is PortfolioS1QwenFeedbackAuthorizationV3:
        bundle = build_portfolio_s1_feedback_bundle_v4(
            prepared.selection,
            prepared.control,
            prepared.authorization,
            ordered,
            run,
            corpus=prepared.corpus,
        )
    else:
        bundle = build_portfolio_s1_feedback_bundle(
            prepared.selection,
            prepared.control,
            prepared.authorization,
            ordered,
            run,
        )
    bundle_path = prepared.output_dir / "portfolio-s1-feedback-bundle.json"
    _publish_or_resume(bundle_path, bundle.canonical_bytes(), "S1 Feedback bundle")
    bundle_file_sha256 = sha256_bytes(
        read_stable_regular_file(
            bundle_path, label="S1 Feedback bundle", max_bytes=32 * 1024 * 1024
        )
    )
    loaded = (
        load_portfolio_s1_feedback_bundle_v4(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
        if type(bundle) is PortfolioS1FeedbackBundleV4
        else load_portfolio_s1_feedback_bundle(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    )
    if loaded != bundle:
        raise PortfolioS1FeedbackRunError("S1 Feedback bundle resume conflict")
    return bundle


def execute_run(
    prepared: PreparedPortfolioS1FeedbackRun,
    *,
    feedback_runner: FeedbackRunner = run_visual_feedback,
    stop_after_canary: bool = False,
) -> PortfolioS1FeedbackBundleV1 | PortfolioS1FeedbackBundleV4 | None:
    with _exclusive_execute_writer_lock(prepared.output_dir):
        return _execute_run_locked(
            prepared,
            feedback_runner=feedback_runner,
            stop_after_canary=stop_after_canary,
        )


def _summary(
    mode: Literal["prepare", "dry-run", "execute"],
    prepared: PreparedPortfolioS1FeedbackRun,
    bundle: PortfolioS1FeedbackBundleV1 | PortfolioS1FeedbackBundleV4 | None = None,
    *,
    stop_after_canary: bool = False,
) -> dict[str, object]:
    existing = sum(_artifact_path(prepared, item).exists() for item in prepared.sources)
    reserved = sum(
        _reservation_path(prepared, item).exists() for item in prepared.sources
    )
    budget = _budget_snapshot(prepared)
    hard_cap = Decimal(QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY)
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_dry_run": 0 if mode != "execute" else None,
        "selected_count": 48,
        "canary_count": 6,
        "remaining_count": 42,
        "feedback_concurrency": 2,
        "max_attempts": 1,
        "provider_call_ceiling": prepared.control.provider_call_ceiling,
        "provider": prepared.control.provider,
        "model": prepared.control.model,
        "processor": prepared.control.processor,
        "provider_attempt_reservation_count": reserved,
        "usage_known_settlement_count": budget.settled_usage_known,
        "usage_unknown_or_unresolved_count": (budget.usage_unknown_or_unresolved),
        "input_tokens": budget.input_tokens,
        "output_tokens_including_reasoning": budget.output_tokens,
        "settled_actual_cost_cny": _cny(budget.settled_actual_cost_cny),
        "unknown_or_unresolved_reserve_cny": _cny(
            budget.unknown_or_unresolved_reserve_cny
        ),
        "accountable_cost_cny": _cny(budget.accountable_cost_cny),
        "phase_hard_cap_cny": _cny(hard_cap),
        "phase_cap_remaining_cny": _cny(hard_cap - budget.accountable_cost_cny),
        "stop_after_canary": stop_after_canary,
        "resumable_artifact_count": existing,
        "selection_sha256": prepared.selection.selection_sha256,
        "control_sha256": prepared.control.control_sha256,
        "bundle_sha256": None if bundle is None else bundle.bundle_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "dry-run", "execute"), required=True
    )
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--expected-execution-control-sha256", required=True)
    parser.add_argument("--artifact-repository-root", type=Path, required=True)
    parser.add_argument("--creator-selection-index", type=Path, required=True)
    parser.add_argument("--expected-creator-selection-sha256", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--expected-rubric-sha256", required=True)
    parser.add_argument("--rubric-id", required=True)
    parser.add_argument("--rubric-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--owner-statement", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-source-lock", type=Path, required=True)
    parser.add_argument("--expected-model-source-lock-sha256", required=True)
    parser.add_argument("--pricing-lock", type=Path, required=True)
    parser.add_argument("--expected-pricing-lock-sha256", required=True)
    parser.add_argument("--role-selection-file", type=Path, required=True)
    parser.add_argument("--expected-role-selection-file-sha256", required=True)
    parser.add_argument("--expected-role-selection-sha256", required=True)
    parser.add_argument(
        "--stop-after-canary",
        action="store_true",
        help=(
            "execute and verify only the fixed six-capability canary; a later "
            "execute command without this flag resumes those exact bytes before "
            "running the remaining 42"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.stop_after_canary and arguments.mode != "execute":
            raise PortfolioS1FeedbackRunError(
                "--stop-after-canary is only valid with --mode execute"
            )
        prepared = prepare_run(arguments)
        if arguments.mode == "dry-run":
            _resume_all_before_provider(prepared)
        bundle = (
            execute_run(
                prepared,
                stop_after_canary=arguments.stop_after_canary,
            )
            if arguments.mode == "execute"
            else None
        )
        print(
            canonical_json_bytes(
                _summary(
                    arguments.mode,
                    prepared,
                    bundle,
                    stop_after_canary=arguments.stop_after_canary,
                )
            ).decode("utf-8"),
            end="",
        )
        return 0
    except (
        PortfolioS1FeedbackError,
        PortfolioS1FeedbackRunError,
        PortfolioS1QwenFeedbackGovernanceError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
