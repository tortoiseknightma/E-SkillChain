"""Prepare, dry-run, or execute a frozen Portfolio S1 Feedback run.

No command in this script mutates the historical Core DashScope owner
authorization.  Historical selected48 and forward Discovery240 identities are
kept distinct.  Round 2 executes fixed cumulative 12/60/120/240 phases with
concurrency two, one attempt per row, and create-only resume.
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
    build_bound_feedback_prompt_v6,
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    redact_feedback_result_for_creator_privacy,
    run_visual_feedback,
    visual_feedback_response_format_v1,
)
from skillchain.evaluation.packets import RubricSnapshot
from skillchain.evaluation.portfolio_s1_feedback import (
    BoundFeedbackArtifact,
    BoundFeedbackArtifactV2,
    BoundFeedbackArtifactV3,
    BoundFeedbackArtifactV4,
    FeedbackCallReservationV1,
    FeedbackCallReservationV2,
    FeedbackCallReservationV3,
    FeedbackCallReservationV4,
    FeedbackGlobalRetryClaimV1,
    PortfolioS1FeedbackAuthorizationV4,
    PortfolioS1FeedbackAuthorizationV5,
    PortfolioS1FeedbackAuthorizationV6,
    PortfolioS1FeedbackAuthorizationV2,
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackBundleV4,
    PortfolioS1FeedbackBundleV5,
    PortfolioS1FeedbackBundleV6,
    PortfolioS1FeedbackBundleV7,
    PortfolioS1FeedbackControlV1,
    PortfolioS1FeedbackControlV9,
    PortfolioS1FeedbackControlV10,
    PortfolioS1FeedbackControlV11,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackRunV1,
    PortfolioS1FeedbackRunV2,
    PortfolioS1FeedbackRunV3,
    PortfolioS1FeedbackRunV4,
    PortfolioS1FeedbackSelectionV1,
    PortfolioS1FeedbackSelectionV2,
    PortfolioS1QwenFeedbackLaunchLockV2,
    PortfolioS1Qwen38FeedbackLaunchLockV3,
    PortfolioS1Qwen38FeedbackLaunchLockV4,
    VerifiedStaticFeedbackSource,
    VerifiedStaticFeedbackSourceV2,
    build_bound_feedback_artifact,
    build_bound_feedback_artifact_v3,
    build_bound_feedback_artifact_v4,
    build_feedback_call_reservation,
    build_feedback_call_reservation_v3,
    build_feedback_call_reservation_v4,
    build_feedback_global_retry_claim_v1,
    build_portfolio_s1_feedback_bundle,
    build_portfolio_s1_feedback_bundle_v4,
    build_portfolio_s1_feedback_bundle_v5,
    build_portfolio_s1_feedback_bundle_v6,
    build_portfolio_s1_feedback_bundle_v7,
    build_portfolio_s1_feedback_control,
    build_portfolio_s1_feedback_control_v11,
    build_portfolio_s1_feedback_run,
    build_portfolio_s1_feedback_run_v3,
    build_portfolio_s1_feedback_run_v4,
    build_portfolio_s1_feedback_selection,
    build_portfolio_s1_feedback_selection_v2,
    build_qwen38_feedback_launch_lock_v4,
    build_selected_qwen38_feedback_authorization_v6,
    build_verified_static_feedback_sources,
    build_verified_static_feedback_sources_v2,
    load_bound_feedback_artifact_v3,
    load_bound_feedback_artifact_v4,
    load_feedback_call_reservation_v3,
    load_feedback_call_reservation_v4,
    load_feedback_global_retry_claim_v1,
    load_portfolio_s1_feedback_bundle,
    load_portfolio_s1_feedback_bundle_v4,
    load_portfolio_s1_feedback_bundle_v5,
    load_portfolio_s1_feedback_bundle_v6,
    load_portfolio_s1_feedback_bundle_v7,
    load_feedback_call_reservation,
    load_portfolio_s1_feedback_run,
    load_portfolio_s1_feedback_run_v3,
    load_portfolio_s1_feedback_run_v4,
    is_qwen38_strict_schema_retry_eligible,
    resume_bound_feedback_artifact,
    resume_bound_feedback_artifact_v3,
    require_feedback_v11_creator_projection_privacy,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    prepare_selected_qwen_feedback_remote_runtime,
    prepare_selected_qwen38_feedback_remote_runtime_v6,
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
    Qwen37FeedbackPricingLockV2,
    QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS,
    QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS,
    QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS,
    QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS,
    QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY,
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11,
    QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY,
    Qwen38FeedbackModelSourceLockV1,
    Qwen38FeedbackPricingLockV3,
    Qwen38FeedbackPricingLockV4,
    build_qwen37_feedback_launch_lock,
    build_selected_qwen_feedback_authorization,
    load_qwen37_feedback_model_source_lock,
    load_qwen37_feedback_pricing_lock,
    load_qwen38_feedback_model_source_lock,
    load_qwen38_feedback_pricing_lock_v4,
    require_qwen37_feedback_pre_call_budget,
    require_qwen38_feedback_pre_call_budget,
    require_qwen38_feedback_pre_call_budget_v2,
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
from skillchain.synthesis.portfolio_opt_folds import (
    OptFoldAssignment,
    OptFoldManifest,
    canonical_opt_fold_mapping_bytes,
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
    selection: PortfolioS1FeedbackSelectionV1 | PortfolioS1FeedbackSelectionV2
    authorization: (
        PortfolioS1FeedbackAuthorizationV2
        | PortfolioS1QwenFeedbackAuthorizationV3
        | PortfolioS1FeedbackAuthorizationV4
        | PortfolioS1FeedbackAuthorizationV5
        | PortfolioS1FeedbackAuthorizationV6
    )
    control: (
        PortfolioS1FeedbackControlV1
        | PortfolioS1FeedbackControlV9
        | PortfolioS1FeedbackControlV10
        | PortfolioS1FeedbackControlV11
    )
    sources: tuple[VerifiedStaticFeedbackSource | VerifiedStaticFeedbackSourceV2, ...]
    remote_runtime: VerifiedSelectedFeedbackRemoteRuntime
    launch_lock: (
        PortfolioS1QwenFeedbackLaunchLockV1
        | PortfolioS1QwenFeedbackLaunchLockV2
        | PortfolioS1Qwen38FeedbackLaunchLockV3
        | PortfolioS1Qwen38FeedbackLaunchLockV4
        | None
    ) = None
    model_source_lock: (
        Qwen37FeedbackModelSourceLockV1 | Qwen38FeedbackModelSourceLockV1 | None
    ) = None
    pricing_lock: (
        Qwen37FeedbackPricingLockV1
        | Qwen37FeedbackPricingLockV2
        | Qwen38FeedbackPricingLockV3
        | Qwen38FeedbackPricingLockV4
        | None
    ) = None


FeedbackSource = VerifiedStaticFeedbackSource | VerifiedStaticFeedbackSourceV2
FeedbackArtifact = (
    BoundFeedbackArtifact
    | BoundFeedbackArtifactV2
    | BoundFeedbackArtifactV3
    | BoundFeedbackArtifactV4
)
FeedbackReservation = (
    FeedbackCallReservationV1
    | FeedbackCallReservationV2
    | FeedbackCallReservationV3
    | FeedbackCallReservationV4
)
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
    source: FeedbackSource,
    evaluator_isolation: EvaluatorIsolationLock,
) -> int:
    """Return a deterministic conservative bound for the exact request wire."""

    bound = (
        build_bound_feedback_prompt_v6(source.packet, evaluator_isolation)
        if type(source) is VerifiedStaticFeedbackSourceV2
        else build_bound_feedback_prompt(source.packet, evaluator_isolation)
    )
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
    sources: tuple[FeedbackSource, ...],
) -> None:
    isolation = make_active_portfolio_evaluator_isolation_lock()
    estimates = tuple(_estimated_input_tokens(source, isolation) for source in sources)
    if not estimates or max(estimates) > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS:
        raise PortfolioS1FeedbackRunError(
            "Qwen Feedback request exceeds the frozen input-token reservation"
        )


def _require_frozen_qwen_endpoint(
    model_source_lock: Qwen37FeedbackModelSourceLockV1
    | Qwen38FeedbackModelSourceLockV1,
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


def _load_discovery600(
    manifest_path: Path,
    mapping_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_mapping_sha256: str,
) -> tuple[tuple[str, ...], str, str]:
    manifest_content = _require_file_sha(
        manifest_path, expected_manifest_sha256, "Discovery600 fold manifest"
    )
    mapping_content = _require_file_sha(
        mapping_path, expected_mapping_sha256, "Discovery600 fold mapping"
    )
    try:
        manifest = OptFoldManifest.model_validate_json(manifest_content, strict=True)
        assignments = tuple(
            OptFoldAssignment.model_validate(item, strict=True)
            for item in parse_canonical_jsonl(
                mapping_content, label="Discovery600 fold mapping"
            )
        )
    except ValueError as error:
        raise PortfolioS1FeedbackRunError(
            "Discovery600 fold artifacts are invalid"
        ) from error
    discovery = tuple(
        sorted(item.query_id for item in assignments if item.role == "discovery")
    )
    replay = tuple(
        sorted(item.query_id for item in assignments if item.role == "replay")
    )
    if (
        canonical_json_bytes(manifest.model_dump(mode="json")) != manifest_content
        or canonical_opt_fold_mapping_bytes(assignments) != mapping_content
        or manifest.mapping_sha256 != expected_mapping_sha256
        or len(assignments) != 800
        or len({item.query_id for item in assignments}) != 800
        or len(discovery) != 600
        or len(replay) != 200
        or set(discovery) & set(replay)
        or manifest.discovery_query_ids_sha256
        != sha256_bytes(canonical_json_bytes(list(discovery)))
        or manifest.replay_query_ids_sha256
        != sha256_bytes(canonical_json_bytes(list(replay)))
    ):
        raise PortfolioS1FeedbackRunError(
            "Discovery600 fold artifacts drifted from the frozen 600/200 split"
        )
    return discovery, manifest.discovery_query_ids_sha256, expected_mapping_sha256


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
    round2: bool = False,
    round2_retry_required: bool = False,
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
    if round2_retry_required and (
        not round2
        or payload.get("schema_version") != 11
        or expected_file_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11
        or expected_selection_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11
    ):
        raise PortfolioS1FeedbackRunError(
            "active Discovery240 rejects historical role selection before provider"
        )
    round2_retry = round2 and payload.get("schema_version") == 11
    expected = (
        {
            "schema_version": 11,
            "model": "qwen3.8-max",
            "processor": "dashscope-qwen38-feedback",
            "cache_namespace": "feedback-evaluator-v11",
            "authorization_scope": (
                "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-same-entry-invalid-json-retry"
            ),
            "prompt_policy_version": "visual-feedback-gcs-policy-labels-prompt-v6",
            "provider_call_ceiling": 241,
            "cap_field": "technical_phase_hard_cap_cny",
            "phase_hard_cap_cny": "94.000000000000",
            "worst_case_reservation_cny": "93.463656000000",
            "pricing_lock_file": "specs/authoring/price-qwen3.8-max-feedback-v4.json",
        }
        if round2_retry
        else {
            "schema_version": 10,
            "model": "qwen3.8-max",
            "processor": "dashscope-qwen38-feedback",
            "cache_namespace": "feedback-evaluator-v11",
            "authorization_scope": (
                "core-opt800-s1-feedback-discovery-selected-240-assets"
            ),
            "prompt_policy_version": "visual-feedback-gcs-policy-labels-prompt-v6",
            "provider_call_ceiling": 240,
            "cap_field": "technical_phase_hard_cap_cny",
            "phase_hard_cap_cny": "94.000000000000",
            "worst_case_reservation_cny": "93.075840000000",
            "pricing_lock_file": ("specs/authoring/price-qwen3.8-max-feedback-v3.json"),
        }
        if round2
        else {
            "schema_version": 8,
            "model": "qwen3.7-plus-2026-05-26",
            "processor": "dashscope-qwen37-feedback",
            "cache_namespace": "feedback-evaluator-v9",
            "authorization_scope": "core-opt800-s1-feedback-selected-48-assets",
            "prompt_policy_version": ("visual-feedback-response-schema-v1-prompt-v5"),
            "provider_call_ceiling": 48,
            "cap_field": "phase_hard_cap_cny",
            "phase_hard_cap_cny": "4.000000000000",
            "worst_case_reservation_cny": "3.496704000000",
            "pricing_lock_file": (
                "specs/authoring/price-qwen3.7-plus-2026-05-26-feedback-v1.json"
            ),
        }
    )
    if (
        payload.get("schema_version") != expected["schema_version"]
        or payload.get("selection_sha256") != expected_selection_sha256
        or sha256_bytes(canonical_json_bytes(unsigned)) != expected_selection_sha256
        or not isinstance(feedback, dict)
        or feedback.get("provider") != "qwen"
        or feedback.get("model") != expected["model"]
        or feedback.get("processor") != expected["processor"]
        or feedback.get("cache_namespace") != expected["cache_namespace"]
        or feedback.get("authorization_scope") != expected["authorization_scope"]
        or feedback.get("prompt_policy_version") != expected["prompt_policy_version"]
        or feedback.get("provider_call_ceiling") != expected["provider_call_ceiling"]
        or feedback.get(expected["cap_field"]) != expected["phase_hard_cap_cny"]
        or feedback.get("worst_case_reservation_cny")
        != expected["worst_case_reservation_cny"]
        or feedback.get("pricing_lock_file") != expected["pricing_lock_file"]
        or feedback.get("requested_response_format") != "json_schema"
        or feedback.get("requested_json_schema_strict") is not True
        or feedback.get("enable_thinking") is not True
        or feedback.get("thinking_budget") != 2048
        or feedback.get("max_tokens") is not None
        or feedback.get("max_completion_tokens") != 4096
        or feedback.get("timeout_seconds") != 600
        or (
            round2_retry
            and (
                feedback.get("live_provider_calls_authorized") is not True
                or feedback.get("budget_authorization_status")
                != "owner_approved_for_exact_discovery_selected240_plus_one_global_invalid_json_retry"
                or feedback.get("retry_policy")
                != "one_global_same_entry_strict_schema_retry_v1"
                or feedback.get("global_retry_token_count") != 1
                or feedback.get("max_attempts_per_retried_query") != 2
                or feedback.get("outer_orchestration_policy_sha256")
                != "6ba3799fce29820466446c6ec0ee98312c6e889fc17c255b25453b9f70694995"
                or feedback.get("technical_phase_hard_cap_cny") != "94.000000000000"
            )
        )
        or (
            round2
            and not round2_retry
            and (
                feedback.get("live_provider_calls_authorized") is not True
                or feedback.get("budget_authorization_status")
                != "owner_approved_for_exact_discovery_selected240"
                or feedback.get("technical_phase_hard_cap_cny") != "94.000000000000"
            )
        )
    ):
        raise PortfolioS1FeedbackRunError(
            "model role selection differs from the selected Qwen Feedback identity"
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
    round2 = getattr(arguments, "selection_profile", "selected48") == "discovery240"
    if round2:
        if any(
            value is None
            for value in (
                arguments.fold_manifest,
                arguments.expected_fold_manifest_sha256,
                arguments.fold_mapping,
                arguments.expected_fold_mapping_sha256,
            )
        ):
            raise PortfolioS1FeedbackRunError(
                "Discovery240 requires the frozen fold manifest and mapping"
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
    else:
        if (
            arguments.creator_selection_index is None
            or arguments.expected_creator_selection_sha256 is None
        ):
            raise PortfolioS1FeedbackRunError(
                "selected48 requires the frozen Creator240 index"
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
    model_source_lock = (
        load_qwen38_feedback_model_source_lock(
            arguments.model_source_lock,
            expected_file_sha256=arguments.expected_model_source_lock_sha256,
        )
        if round2
        else load_qwen37_feedback_model_source_lock(
            arguments.model_source_lock,
            expected_file_sha256=arguments.expected_model_source_lock_sha256,
        )
    )
    _require_frozen_qwen_endpoint(model_source_lock)
    pricing_lock = (
        load_qwen38_feedback_pricing_lock_v4(
            arguments.pricing_lock,
            expected_file_sha256=arguments.expected_pricing_lock_sha256,
        )
        if round2
        else load_qwen37_feedback_pricing_lock(
            arguments.pricing_lock,
            expected_file_sha256=arguments.expected_pricing_lock_sha256,
        )
    )
    if round2 and (
        arguments.expected_pricing_lock_sha256
        != QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4
        or arguments.expected_role_selection_file_sha256
        != QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11
        or arguments.expected_role_selection_sha256
        != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11
    ):
        raise PortfolioS1FeedbackRunError(
            "active Discovery240 requires fresh pricing-v4 and role-v11 identity"
        )
    role_selection_file_sha256, role_selection_sha256 = _load_qwen_role_selection(
        arguments.role_selection_file,
        expected_file_sha256=arguments.expected_role_selection_file_sha256,
        expected_selection_sha256=arguments.expected_role_selection_sha256,
        round2=round2,
        round2_retry_required=round2,
    )
    parent_remote_runtime = corpus.core_inputs.runtime_for("dashscope-qwen-assistant")
    rubric = _rubric(
        arguments.rubric_file,
        expected_sha256=arguments.expected_rubric_sha256,
        rubric_id=arguments.rubric_id,
        rubric_version=arguments.rubric_version,
    )
    if round2:
        assert type(selection) is PortfolioS1FeedbackSelectionV2
        authorization = build_selected_qwen38_feedback_authorization_v6(
            selection,
            parent_remote_runtime,
            authorization_id=arguments.authorization_id,
            reviewer_id=arguments.reviewer_id,
            reviewed_at=_reviewed_at(arguments.reviewed_at),
            owner_statement=arguments.owner_statement,
            owner_approved_phase_hard_cap_cny=(
                getattr(arguments, "approved_phase_hard_cap_cny", None)
            ),
            model_source_lock_file_sha256=(arguments.expected_model_source_lock_sha256),
            model_source_lock_sha256=model_source_lock.source_lock_sha256,
            pricing_lock_file_sha256=arguments.expected_pricing_lock_sha256,
            pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
            role_selection_file_sha256=role_selection_file_sha256,
            role_selection_sha256=role_selection_sha256,
        )
        control = build_portfolio_s1_feedback_control_v11(
            selection, authorization, rubric=rubric
        )
    else:
        assert type(selection) is PortfolioS1FeedbackSelectionV1
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
            rubric=rubric,
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
    launch_lock = (
        build_qwen38_feedback_launch_lock_v4(
            run_id=arguments.run_id,
            selection=selection,
            authorization=authorization,
            control=control,
        )
        if round2
        else build_qwen37_feedback_launch_lock(
            run_id=arguments.run_id,
            selection_sha256=selection.selection_sha256,
            corpus_sha256=selection.corpus_sha256,
            authorization=authorization,
            model_source_lock=model_source_lock,
            model_source_lock_file_sha256=(arguments.expected_model_source_lock_sha256),
            pricing_lock=pricing_lock,
            pricing_lock_file_sha256=arguments.expected_pricing_lock_sha256,
            role_selection_file_sha256=role_selection_file_sha256,
            role_selection_sha256=role_selection_sha256,
            control_file_sha256=sha256_bytes(control.canonical_bytes()),
            control_sha256=control.control_sha256,
        )
    )
    _publish_or_resume(
        output / "launch-lock.json",
        launch_lock.canonical_bytes(),
        "Qwen Feedback launch lock",
    )
    sources = (
        build_verified_static_feedback_sources_v2(corpus, selection, control)
        if round2
        else build_verified_static_feedback_sources(corpus, selection, control)
    )
    # Prove every frozen request fits the priced input reservation before
    # publishing a remote receipt or permitting any provider attempt.
    _validate_input_reservation_bounds(sources)
    remote_runtime = (
        prepare_selected_qwen38_feedback_remote_runtime_v6(
            corpus,
            selection,
            authorization,
            control,
            parent_remote_runtime,
            receipt_path=output / "remote-runtime-receipt.json",
            verified_sources=sources,
        )
        if round2
        else prepare_selected_qwen_feedback_remote_runtime(
            corpus,
            selection,
            authorization,
            control,
            parent_remote_runtime,
            receipt_path=output / "remote-runtime-receipt.json",
            verified_sources=sources,
        )
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
    source: FeedbackSource,
) -> Path:
    return (
        prepared.output_dir
        / "bound-feedback"
        / (f"{source.row.query_ordinal:04d}-{source.selection_entry_sha256[:16]}.json")
    )


def _reservation_path(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: FeedbackSource,
) -> Path:
    return (
        prepared.output_dir
        / "provider-attempts"
        / (f"{source.row.query_ordinal:04d}-{source.selection_entry_sha256[:16]}.json")
    )


def _artifact_path_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> Path:
    return (
        prepared.output_dir
        / "bound-feedback-v2"
        / (
            f"{source.row.query_ordinal:04d}-"
            f"{source.selection_entry_sha256[:16]}-attempt-{attempt_index}.json"
        )
    )


def _reservation_path_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> Path:
    return (
        prepared.output_dir
        / "provider-attempts-v2"
        / (
            f"{source.row.query_ordinal:04d}-"
            f"{source.selection_entry_sha256[:16]}-attempt-{attempt_index}.json"
        )
    )


def _retry_claim_path(prepared: PreparedPortfolioS1FeedbackRun) -> Path:
    return prepared.output_dir / "global-retry-claim.json"


def _load_reservation(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: FeedbackSource,
) -> FeedbackReservation | None:
    path = _reservation_path(prepared, source)
    if not path.exists():
        return None
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        assert type(source) is VerifiedStaticFeedbackSourceV2
        reservation = load_feedback_call_reservation_v3(
            path,
            expected_selection_sha256=prepared.selection.selection_sha256,
            expected_control_sha256=prepared.control.control_sha256,
            expected_entry_sha256=entry.entry_sha256,
        )
        expected = build_feedback_call_reservation_v3(
            prepared.selection,
            prepared.control,
            entry,
            verified_source=source,
        )
    else:
        assert type(prepared.control) is PortfolioS1FeedbackControlV1
        assert type(source) is VerifiedStaticFeedbackSource
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


def _resume_artifact(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: FeedbackSource,
    entry,
    reservation: FeedbackReservation,
) -> FeedbackArtifact | None:
    path = _artifact_path(prepared, source)
    if not path.exists():
        return None
    if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        assert type(source) is VerifiedStaticFeedbackSourceV2
        assert type(reservation) is FeedbackCallReservationV3
        artifact = load_bound_feedback_artifact_v3(
            path,
            expected_selection_sha256=prepared.selection.selection_sha256,
            expected_control_sha256=prepared.control.control_sha256,
            expected_entry_sha256=entry.entry_sha256,
        )
        return resume_bound_feedback_artifact_v3(
            artifact,
            prepared.selection,
            prepared.control,
            entry,
            verified_source=source,
            reservation=reservation,
        )
    assert type(prepared.control) is PortfolioS1FeedbackControlV1
    assert type(source) is VerifiedStaticFeedbackSource
    assert type(reservation) is FeedbackCallReservationV1
    return resume_bound_feedback_artifact(
        path,
        selection=prepared.selection,
        control=prepared.control,
        entry=entry,
        verified_source=source,
        reservation=reservation,
    )


def _load_reservation_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> FeedbackCallReservationV4 | None:
    path = _reservation_path_v4(prepared, source, attempt_index)
    if not path.exists():
        return None
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    return load_feedback_call_reservation_v4(
        path,
        expected_selection_sha256=prepared.selection.selection_sha256,
        expected_control_sha256=prepared.control.control_sha256,
        expected_entry_sha256=entry.entry_sha256,
        expected_attempt_index=attempt_index,
    )


def _load_artifact_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSourceV2,
    attempt_index: Literal[1, 2],
) -> BoundFeedbackArtifactV4 | None:
    path = _artifact_path_v4(prepared, source, attempt_index)
    if not path.exists():
        return None
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    artifact = load_bound_feedback_artifact_v4(
        path,
        expected_selection_sha256=prepared.selection.selection_sha256,
        expected_control_sha256=prepared.control.control_sha256,
        expected_entry_sha256=entry.entry_sha256,
        expected_attempt_index=attempt_index,
    )
    reservation = _load_reservation_v4(prepared, source, attempt_index)
    if reservation is None:
        raise PortfolioS1FeedbackRunError(
            "Feedback v2 settlement exists without its reservation"
        )
    first = None if attempt_index == 1 else _load_artifact_v4(prepared, source, 1)
    claim = None if attempt_index == 1 else _load_retry_claim_v1(prepared)
    if attempt_index == 2 and (first is None or claim is None):
        raise PortfolioS1FeedbackRunError(
            "Feedback v2 retry settlement lacks first artifact or claim"
        )
    expected = build_bound_feedback_artifact_v4(
        prepared.selection,
        prepared.control,
        entry,
        source.packet,
        artifact.feedback_result,
        verified_source=source,
        reservation=reservation,
        first_artifact=first,
        retry_claim=claim,
    )
    if artifact != expected:
        raise PortfolioS1FeedbackRunError("Feedback v2 settlement source conflict")
    return artifact


def _load_retry_claim_v1(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> FeedbackGlobalRetryClaimV1 | None:
    path = _retry_claim_path(prepared)
    if not path.exists():
        return None
    claim = load_feedback_global_retry_claim_v1(path)
    if (
        type(prepared.control) is not PortfolioS1FeedbackControlV11
        or claim.selection_sha256 != prepared.selection.selection_sha256
        or claim.control_sha256 != prepared.control.control_sha256
    ):
        raise PortfolioS1FeedbackRunError("Feedback global retry claim root conflict")
    return claim


def _budget_snapshot(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> _FeedbackBudgetSnapshot:
    """Rebuild accountable Qwen cost from create-only attempts and settlements."""

    if type(prepared.control) is PortfolioS1FeedbackControlV11:
        return _budget_snapshot_v4(prepared)

    reserved = 0
    known = 0
    unknown = 0
    input_tokens = 0
    output_tokens = 0
    settled_actual = Decimal(0)
    unknown_reserve = Decimal(0)
    round2 = type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    fixed_reserve = Decimal(
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
        if round2
        else QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    per_million = Decimal(1_000_000)
    input_rate = Decimal(
        QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
        if round2
        else QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_rate = Decimal(
        QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
        if round2
        else QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )

    for entry, source in zip(prepared.selection.entries, prepared.sources, strict=True):
        reservation = _load_reservation(prepared, source)
        if reservation is None:
            continue
        reserved += 1
        artifact = _resume_artifact(prepared, source, entry, reservation)
        usage = None if artifact is None else artifact.feedback_result.usage
        if usage is None:
            unknown += 1
            unknown_reserve += fixed_reserve
            continue
        if usage.input_tokens > (
            QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
            if round2
            else QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
        ) or usage.output_tokens > (
            QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS
            if round2
            else QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
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
    hard_cap = Decimal(
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
        if round2
        else QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    )
    if accountable > hard_cap:
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


def _budget_snapshot_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> _FeedbackBudgetSnapshot:
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    reserved: list[FeedbackCallReservationV4] = []
    known = input_tokens = output_tokens = 0
    unknown = 0
    settled_actual = Decimal(0)
    unknown_reserve = Decimal(0)
    for source in prepared.sources:
        assert type(source) is VerifiedStaticFeedbackSourceV2
        for attempt in (1, 2):
            reservation = _load_reservation_v4(prepared, source, attempt)
            artifact = _load_artifact_v4(prepared, source, attempt)
            if artifact is not None and reservation is None:
                raise PortfolioS1FeedbackRunError(
                    "Feedback v2 settlement exists without its reservation"
                )
            if reservation is None:
                continue
            reserved.append(reservation)
            if artifact is None or artifact.feedback_result.usage is None:
                unknown += 1
                unknown_reserve += Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY)
                continue
            usage = artifact.feedback_result.usage
            if (
                usage.input_tokens > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
                or usage.output_tokens > QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS
            ):
                raise PortfolioS1FeedbackRunError(
                    "Qwen3.8 Feedback v2 usage exceeded reservation"
                )
            known += 1
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            settled_actual += (
                Decimal(usage.input_tokens)
                * Decimal(QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS)
                + Decimal(usage.output_tokens)
                * Decimal(QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS)
            ) / Decimal(1_000_000)
    ordinals = sorted(item.global_call_ordinal for item in reserved)
    if (
        len(reserved) > 241
        or sum(item.attempt_index == 2 for item in reserved) > 1
        or ordinals != list(range(1, len(reserved) + 1))
    ):
        raise PortfolioS1FeedbackRunError(
            "Qwen3.8 Feedback v2 provider reservation sequence drifted"
        )
    accountable = settled_actual + unknown_reserve
    if accountable > Decimal(QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY):
        raise PortfolioS1FeedbackRunError(
            "Qwen3.8 Feedback v2 accountable cost exceeds CNY94"
        )
    return _FeedbackBudgetSnapshot(
        provider_calls_reserved=len(reserved),
        settled_usage_known=known,
        usage_unknown_or_unresolved=unknown,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        settled_actual_cost_cny=settled_actual,
        unknown_or_unresolved_reserve_cny=unknown_reserve,
        accountable_cost_cny=accountable,
    )


def _verify_attempt_file_set(prepared: PreparedPortfolioS1FeedbackRun) -> None:
    if type(prepared.control) is PortfolioS1FeedbackControlV11:
        for legacy_name in ("provider-attempts", "bound-feedback"):
            legacy = prepared.output_dir / legacy_name
            if legacy.exists() and any(path.is_file() for path in legacy.iterdir()):
                raise PortfolioS1FeedbackRunError(
                    "fresh Feedback v2 root contains legacy provider artifacts"
                )
        expected_reservations = {
            _reservation_path_v4(prepared, source, attempt).name
            for source in prepared.sources
            for attempt in (1, 2)
        }
        expected_results = {
            _artifact_path_v4(prepared, source, attempt).name
            for source in prepared.sources
            for attempt in (1, 2)
        }
        for directory, expected, label in (
            (
                prepared.output_dir / "provider-attempts-v2",
                expected_reservations,
                "Feedback v2 reservation",
            ),
            (
                prepared.output_dir / "bound-feedback-v2",
                expected_results,
                "Feedback v2 settlement",
            ),
        ):
            if not directory.exists():
                continue
            actual = {path.name for path in directory.iterdir() if path.is_file()}
            if not actual <= expected or len(actual) > 241:
                raise PortfolioS1FeedbackRunError(
                    f"{label} file set exceeds the frozen 241-call ceiling"
                )
        return
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
) -> dict[str, FeedbackArtifact]:
    if type(prepared.control) is PortfolioS1FeedbackControlV11:
        return _resume_all_before_provider_v4(prepared)
    _verify_attempt_file_set(prepared)
    resumed: dict[str, FeedbackArtifact] = {}
    orphaned_entry_sha256s: list[str] = []
    for entry, source in zip(prepared.selection.entries, prepared.sources, strict=True):
        reservation = _load_reservation(prepared, source)
        artifact_exists = _artifact_path(prepared, source).exists()
        if artifact_exists and reservation is None:
            raise PortfolioS1FeedbackRunError(
                "Feedback settlement exists without its provider reservation"
            )
        if reservation is None:
            continue
        artifact = _resume_artifact(prepared, source, entry, reservation)
        if artifact is None:
            if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
                orphaned_entry_sha256s.append(entry.entry_sha256)
                continue
            raise PortfolioS1FeedbackRunError(
                "orphan Feedback provider reservation forbids a replacement call"
            )
        resumed[entry.entry_sha256] = artifact
    run_path = prepared.output_dir / "run.json"
    if run_path.exists():
        run_file_sha256 = sha256_bytes(
            read_stable_regular_file(
                run_path, label="Feedback run", max_bytes=16 * 1024 * 1024
            )
        )
        ordered_resumed = tuple(
            resumed[item.entry_sha256]
            for item in prepared.selection.entries
            if item.entry_sha256 in resumed
        )
        if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
            assert type(prepared.control) is PortfolioS1FeedbackControlV10
            run = load_portfolio_s1_feedback_run_v3(
                run_path, expected_file_sha256=run_file_sha256
            )
            expected = build_portfolio_s1_feedback_run_v3(
                prepared.selection,
                prepared.control,
                ordered_resumed,
                orphaned_entry_sha256s=tuple(orphaned_entry_sha256s),
            )
        else:
            assert type(prepared.control) is PortfolioS1FeedbackControlV1
            run = load_portfolio_s1_feedback_run(
                run_path, expected_file_sha256=run_file_sha256
            )
            expected = build_portfolio_s1_feedback_run(
                prepared.selection, prepared.control, ordered_resumed
            )
        if run != expected:
            raise PortfolioS1FeedbackRunError("Feedback run resume conflict")
        if run.status in {"stopped_nonparsed", "stopped_orphan"}:
            raise PortfolioS1FeedbackRunError(
                "Feedback run previously stopped on a terminal fixed sample"
            )
    if orphaned_entry_sha256s:
        if not run_path.exists():
            _publish_terminal_run(
                prepared,
                resumed,
                orphaned_entry_sha256s=tuple(orphaned_entry_sha256s),
            )
        raise PortfolioS1FeedbackRunError(
            "orphan Feedback provider reservation was published as stopped_orphan; "
            "a replacement call is forbidden"
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


def _attempt_state_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> tuple[
    tuple[BoundFeedbackArtifactV4, ...],
    tuple[FeedbackCallReservationV4, ...],
    dict[str, BoundFeedbackArtifactV4],
]:
    artifacts: list[BoundFeedbackArtifactV4] = []
    orphans: list[FeedbackCallReservationV4] = []
    latest: dict[str, BoundFeedbackArtifactV4] = {}
    for source in prepared.sources:
        assert type(source) is VerifiedStaticFeedbackSourceV2
        for attempt in (1, 2):
            reservation = _load_reservation_v4(prepared, source, attempt)
            artifact = _load_artifact_v4(prepared, source, attempt)
            if artifact is not None and reservation is None:
                raise PortfolioS1FeedbackRunError(
                    "Feedback v2 settlement exists without reservation"
                )
            if reservation is not None and artifact is None:
                orphans.append(reservation)
                continue
            if artifact is not None:
                if artifact.reservation_sha256 != reservation.reservation_sha256:
                    raise PortfolioS1FeedbackRunError(
                        "Feedback v2 settlement/reservation conflict"
                    )
                artifacts.append(artifact)
                latest[source.selection_entry_sha256] = artifact
    artifacts.sort(key=lambda item: item.global_call_ordinal)
    orphans.sort(key=lambda item: item.global_call_ordinal)
    return tuple(artifacts), tuple(orphans), latest


def _resume_all_before_provider_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
) -> dict[str, FeedbackArtifact]:
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    _verify_attempt_file_set(prepared)
    artifacts, orphans, latest = _attempt_state_v4(prepared)
    claim = _load_retry_claim_v1(prepared)
    if claim is not None:
        entry = next(
            (
                item
                for item in prepared.selection.entries
                if item.entry_sha256 == claim.selection_entry_sha256
            ),
            None,
        )
        first = next(
            (
                item
                for item in artifacts
                if item.selection_entry_sha256 == claim.selection_entry_sha256
                and item.attempt_index == 1
            ),
            None,
        )
        if (
            entry is None
            or first is None
            or claim
            != build_feedback_global_retry_claim_v1(
                prepared.selection, prepared.control, entry, first
            )
        ):
            raise PortfolioS1FeedbackRunError(
                "Feedback v2 global retry claim resume conflict"
            )
    run_path = prepared.output_dir / "run.json"
    if orphans:
        if not run_path.exists():
            _publish_terminal_run(prepared, latest)
        raise PortfolioS1FeedbackRunError(
            "orphan Feedback v2 reservation published stopped_orphan; no retry"
        )
    terminal_nonparsed = [
        item
        for item in latest.values()
        if item.status != "parsed"
        and not (
            item.attempt_index == 1
            and is_qwen38_strict_schema_retry_eligible(item.feedback_result)
            and (
                claim is None
                or claim.selection_entry_sha256 == item.selection_entry_sha256
            )
        )
    ]
    if terminal_nonparsed:
        if not run_path.exists():
            _publish_terminal_run(prepared, latest)
        raise PortfolioS1FeedbackRunError(
            "existing nonretryable Feedback v2 settlement stops the run"
        )
    if run_path.exists():
        content = read_stable_regular_file(
            run_path, label="Feedback run v4", max_bytes=16 * 1024 * 1024
        )
        run = load_portfolio_s1_feedback_run_v4(
            run_path, expected_file_sha256=sha256_bytes(content)
        )
        expected = build_portfolio_s1_feedback_run_v4(
            prepared.selection,
            prepared.control,
            artifacts,
            retry_claim=claim,
        )
        if run != expected:
            raise PortfolioS1FeedbackRunError("Feedback run v4 resume conflict")
        if run.status != "completed":
            raise PortfolioS1FeedbackRunError(
                "Feedback run v4 previously stopped terminally"
            )
    _budget_snapshot(prepared)
    return latest


def _publish_terminal_run(
    prepared: PreparedPortfolioS1FeedbackRun,
    artifacts: dict[str, FeedbackArtifact],
    *,
    orphaned_entry_sha256s: tuple[str, ...] = (),
) -> PortfolioS1FeedbackRunV1 | PortfolioS1FeedbackRunV3 | PortfolioS1FeedbackRunV4:
    ordered = tuple(
        artifacts[item.entry_sha256]
        for item in prepared.selection.entries
        if item.entry_sha256 in artifacts
    )
    if type(prepared.control) is PortfolioS1FeedbackControlV11:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
        all_artifacts, orphans, _latest = _attempt_state_v4(prepared)
        claim = _load_retry_claim_v1(prepared)
        run = build_portfolio_s1_feedback_run_v4(
            prepared.selection,
            prepared.control,
            all_artifacts,
            retry_claim=claim,
            orphaned_attempts=orphans,
        )
    elif type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        run = build_portfolio_s1_feedback_run_v3(
            prepared.selection,
            prepared.control,
            ordered,
            orphaned_entry_sha256s=orphaned_entry_sha256s,
        )
    else:
        if orphaned_entry_sha256s:
            raise PortfolioS1FeedbackRunError(
                "historical Feedback run cannot publish a forward orphan status"
            )
        assert type(prepared.control) is PortfolioS1FeedbackControlV1
        run = build_portfolio_s1_feedback_run(
            prepared.selection, prepared.control, ordered
        )
    _publish_or_resume(
        prepared.output_dir / "run.json", run.canonical_bytes(), "Feedback run"
    )
    return run


def _run_source(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: FeedbackSource,
    feedback_runner: FeedbackRunner,
) -> FeedbackArtifact:
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    round2 = type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    if round2:
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        assert type(source) is VerifiedStaticFeedbackSourceV2
        reservation = build_feedback_call_reservation_v3(
            prepared.selection,
            prepared.control,
            entry,
            verified_source=source,
        )
    else:
        assert type(prepared.control) is PortfolioS1FeedbackControlV1
        assert type(source) is VerifiedStaticFeedbackSource
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
        reserved_cost = (
            require_qwen38_feedback_pre_call_budget(
                estimated_input_tokens_including_images=estimated_input_tokens,
                provider_calls_already_reserved=budget.provider_calls_reserved,
                committed_cost_cny=_cny(budget.accountable_cost_cny),
            )
            if round2
            else require_qwen37_feedback_pre_call_budget(
                estimated_input_tokens_including_images=estimated_input_tokens,
                provider_calls_already_reserved=budget.provider_calls_reserved,
                committed_cost_cny=_cny(budget.accountable_cost_cny),
            )
        )
        expected_reservation_cny = (
            QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
            if round2
            else QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
        )
        if reserved_cost != expected_reservation_cny:
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
        max_tokens=None if round2 else prepared.control.max_tokens,
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
    if round2 and result.status == "parsed":
        try:
            require_feedback_v11_creator_projection_privacy(
                result,
                private_query_ids=tuple(
                    item.query_id for item in prepared.selection.entries
                ),
            )
        except PortfolioS1FeedbackError:
            # The provider call has already happened, so fail as an immutable,
            # accountable nonparsed settlement instead of raising through the
            # reservation/settlement crash window.
            result = redact_feedback_result_for_creator_privacy(result)
    if round2:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        assert type(source) is VerifiedStaticFeedbackSourceV2
        assert type(reservation) is FeedbackCallReservationV3
        artifact = build_bound_feedback_artifact_v3(
            prepared.selection,
            prepared.control,
            entry,
            source.packet,
            result,
            verified_source=source,
            reservation=reservation,
        )
    else:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV1
        assert type(prepared.control) is PortfolioS1FeedbackControlV1
        assert type(source) is VerifiedStaticFeedbackSource
        assert type(reservation) is FeedbackCallReservationV1
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


def _run_source_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    source: VerifiedStaticFeedbackSourceV2,
    feedback_runner: FeedbackRunner,
    *,
    attempt_index: Literal[1, 2],
    first_artifact: BoundFeedbackArtifactV4 | None = None,
    retry_claim: FeedbackGlobalRetryClaimV1 | None = None,
) -> BoundFeedbackArtifactV4:
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == source.selection_entry_sha256
    )
    evaluator_isolation = make_active_portfolio_evaluator_isolation_lock()
    estimated_input_tokens = _estimated_input_tokens(source, evaluator_isolation)
    with _BUDGET_RESERVATION_LOCK:
        budget = _budget_snapshot(prepared)
        reserved_cost = require_qwen38_feedback_pre_call_budget_v2(
            estimated_input_tokens_including_images=estimated_input_tokens,
            provider_calls_already_reserved=budget.provider_calls_reserved,
            committed_cost_cny=_cny(budget.accountable_cost_cny),
        )
        if reserved_cost != QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY:
            raise PortfolioS1FeedbackRunError(
                "Qwen3.8 Feedback v2 pre-call reservation amount drifted"
            )
        reservation = build_feedback_call_reservation_v4(
            prepared.selection,
            prepared.control,
            entry,
            verified_source=source,
            attempt_index=attempt_index,
            global_call_ordinal=budget.provider_calls_reserved + 1,
            first_artifact=first_artifact,
            retry_claim=retry_claim,
        )
        try:
            atomic_create_file(
                _reservation_path_v4(prepared, source, attempt_index),
                reservation.canonical_bytes(),
            )
        except FileExistsError as error:
            raise PortfolioS1FeedbackRunError(
                "Feedback v2 reservation already exists; replacement forbidden"
            ) from error
    result = feedback_runner(
        source.packet,
        evaluator_isolation,
        remote_runtime=prepared.remote_runtime,
        max_tokens=None,
        max_completion_tokens=prepared.control.max_completion_tokens,
        timeout_seconds=prepared.control.requested_timeout_seconds,
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
            "Feedback v2 provider result differs from frozen remote runtime"
        )
    if result.status == "parsed":
        try:
            require_feedback_v11_creator_projection_privacy(
                result,
                private_query_ids=tuple(
                    item.query_id for item in prepared.selection.entries
                ),
            )
        except PortfolioS1FeedbackError:
            result = redact_feedback_result_for_creator_privacy(result)
    artifact = build_bound_feedback_artifact_v4(
        prepared.selection,
        prepared.control,
        entry,
        source.packet,
        result,
        verified_source=source,
        reservation=reservation,
        first_artifact=first_artifact,
        retry_claim=retry_claim,
    )
    _publish_or_resume(
        _artifact_path_v4(prepared, source, attempt_index),
        artifact.canonical_bytes(),
        "bound Feedback v2 artifact",
    )
    _budget_snapshot(prepared)
    return artifact


def _execute_phase(
    prepared: PreparedPortfolioS1FeedbackRun,
    sources: tuple[FeedbackSource, ...],
    artifacts: dict[str, FeedbackArtifact],
    feedback_runner: FeedbackRunner,
) -> None:
    if type(prepared.control) is PortfolioS1FeedbackControlV11:
        _execute_phase_v4(prepared, sources, artifacts, feedback_runner)
        return
    pending = tuple(
        item for item in sources if item.selection_entry_sha256 not in artifacts
    )
    for offset in range(0, len(pending), 2):
        wave = pending[offset : offset + 2]
        completed: list[FeedbackArtifact] = []
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(_run_source, prepared, source, feedback_runner)
                for source in wave
            )
            for future in futures:
                try:
                    completed.append(future.result())
                except Exception as error:  # noqa: PERF203 - must settle the whole wave
                    failures.append(error)
        for artifact in completed:
            artifacts[artifact.selection_entry_sha256] = artifact
        if failures:
            orphaned: list[str] = []
            if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
                for source in wave:
                    entry = next(
                        item
                        for item in prepared.selection.entries
                        if item.entry_sha256 == source.selection_entry_sha256
                    )
                    reservation = _load_reservation(prepared, source)
                    if reservation is None:
                        continue
                    artifact = _resume_artifact(prepared, source, entry, reservation)
                    if artifact is None:
                        orphaned.append(entry.entry_sha256)
                    else:
                        artifacts[entry.entry_sha256] = artifact
                if orphaned:
                    _publish_terminal_run(
                        prepared,
                        artifacts,
                        orphaned_entry_sha256s=tuple(orphaned),
                    )
                    raise PortfolioS1FeedbackRunError(
                        "Feedback provider wave failed after reservation; "
                        "stopped_orphan was published and no replacement is allowed"
                    ) from failures[0]
            raise failures[0]
        _budget_snapshot(prepared)
        if any(item.status != "parsed" for item in completed):
            _publish_terminal_run(prepared, artifacts)
            raise PortfolioS1FeedbackRunError(
                "Feedback stopped on a nonparsed fixed sample; no replacement allowed"
            )


def _claim_retry_v1(
    prepared: PreparedPortfolioS1FeedbackRun,
    artifact: BoundFeedbackArtifactV4,
) -> FeedbackGlobalRetryClaimV1:
    assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    assert type(prepared.control) is PortfolioS1FeedbackControlV11
    entry = next(
        item
        for item in prepared.selection.entries
        if item.entry_sha256 == artifact.selection_entry_sha256
    )
    expected = build_feedback_global_retry_claim_v1(
        prepared.selection, prepared.control, entry, artifact
    )
    path = _retry_claim_path(prepared)
    if path.exists():
        observed = _load_retry_claim_v1(prepared)
        if observed != expected:
            raise PortfolioS1FeedbackRunError(
                "global Feedback retry token is already claimed by another entry"
            )
        return expected
    try:
        atomic_create_file(path, expected.canonical_bytes())
    except FileExistsError as error:
        raise PortfolioS1FeedbackRunError(
            "concurrent Feedback retry claim detected"
        ) from error
    return expected


def _execute_phase_v4(
    prepared: PreparedPortfolioS1FeedbackRun,
    sources: tuple[FeedbackSource, ...],
    artifacts: dict[str, FeedbackArtifact],
    feedback_runner: FeedbackRunner,
) -> None:
    typed_sources = tuple(
        source for source in sources if type(source) is VerifiedStaticFeedbackSourceV2
    )
    if len(typed_sources) != len(sources):
        raise PortfolioS1FeedbackRunError("Feedback v2 phase source type drifted")
    # Crash recovery is ordered before every new first-attempt reservation.
    # Once an eligible first settlement exists, the next and only permissible
    # provider action is its same-entry second attempt.
    existing_eligible = sorted(
        (
            item
            for item in artifacts.values()
            if type(item) is BoundFeedbackArtifactV4
            and item.attempt_index == 1
            and is_qwen38_strict_schema_retry_eligible(item.feedback_result)
        ),
        key=lambda item: item.global_call_ordinal,
    )
    existing_claim = _load_retry_claim_v1(prepared)
    if len(existing_eligible) > 1 or (
        existing_eligible
        and existing_claim is not None
        and existing_claim.selection_entry_sha256
        != existing_eligible[0].selection_entry_sha256
    ):
        _publish_terminal_run(prepared, artifacts)
        raise PortfolioS1FeedbackRunError(
            "multiple settled schema failures cannot consume the one retry"
        )
    if existing_eligible:
        first = existing_eligible[0]
        claim = existing_claim or _claim_retry_v1(prepared, first)
        source = next(
            item
            for item in typed_sources
            if item.selection_entry_sha256 == first.selection_entry_sha256
        )
        retry = _run_source_v4(
            prepared,
            source,
            feedback_runner,
            attempt_index=2,
            first_artifact=first,
            retry_claim=claim,
        )
        artifacts[retry.selection_entry_sha256] = retry
        if retry.status != "parsed":
            _publish_terminal_run(prepared, artifacts)
            raise PortfolioS1FeedbackRunError(
                "the crash-recovered Feedback schema retry failed; run stopped"
            )
    pending = tuple(
        source
        for source in typed_sources
        if source.selection_entry_sha256 not in artifacts
    )
    for offset in range(0, len(pending), 2):
        wave = pending[offset : offset + 2]
        completed: list[BoundFeedbackArtifactV4] = []
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(
                    _run_source_v4,
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
            artifacts[artifact.selection_entry_sha256] = artifact
        if failures:
            _all, orphans, latest = _attempt_state_v4(prepared)
            artifacts.update(latest)
            if orphans:
                _publish_terminal_run(prepared, artifacts)
                raise PortfolioS1FeedbackRunError(
                    "Feedback v2 provider wave left an orphan; no retry allowed"
                ) from failures[0]
            raise failures[0]
        nonparsed = [item for item in completed if item.status != "parsed"]
        noneligible = [
            item
            for item in nonparsed
            if not is_qwen38_strict_schema_retry_eligible(item.feedback_result)
        ]
        if noneligible:
            _publish_terminal_run(prepared, artifacts)
            raise PortfolioS1FeedbackRunError(
                "Feedback v2 stopped on a nonretryable fixed-sample failure"
            )
        eligible = sorted(
            (item for item in nonparsed if item not in noneligible),
            key=lambda item: item.global_call_ordinal,
        )
        claim = _load_retry_claim_v1(prepared)
        if len(eligible) > 1 or (
            eligible
            and claim is not None
            and claim.selection_entry_sha256 != eligible[0].selection_entry_sha256
        ):
            _publish_terminal_run(prepared, artifacts)
            raise PortfolioS1FeedbackRunError(
                "second strict-schema Feedback failure stops the fresh run"
            )
        if eligible:
            first = eligible[0]
            claim = _claim_retry_v1(prepared, first)
            source = next(
                item
                for item in typed_sources
                if item.selection_entry_sha256 == first.selection_entry_sha256
            )
            retry = _run_source_v4(
                prepared,
                source,
                feedback_runner,
                attempt_index=2,
                first_artifact=first,
                retry_claim=claim,
            )
            artifacts[retry.selection_entry_sha256] = retry
            if retry.status != "parsed":
                _publish_terminal_run(prepared, artifacts)
                raise PortfolioS1FeedbackRunError(
                    "the sole Feedback schema retry failed; run stopped"
                )
        _budget_snapshot(prepared)

    # Resume path: a crash may leave one eligible first settlement plus its
    # create-only claim but no second reservation.  Claim ownership makes this
    # the only safe provider call that may still be created.
    claim = _load_retry_claim_v1(prepared)
    eligible_existing = sorted(
        (
            item
            for item in artifacts.values()
            if type(item) is BoundFeedbackArtifactV4
            and item.attempt_index == 1
            and is_qwen38_strict_schema_retry_eligible(item.feedback_result)
        ),
        key=lambda item: item.global_call_ordinal,
    )
    pending_retry = None
    if claim is None and len(eligible_existing) == 1:
        pending_retry = eligible_existing[0]
        claim = _claim_retry_v1(prepared, pending_retry)
    elif claim is not None:
        pending_retry = next(
            (
                item
                for item in eligible_existing
                if item.selection_entry_sha256 == claim.selection_entry_sha256
            ),
            None,
        )
    if len(eligible_existing) > 1 or (eligible_existing and pending_retry is None):
        _publish_terminal_run(prepared, artifacts)
        raise PortfolioS1FeedbackRunError(
            "additional strict-schema Feedback failure cannot be retried"
        )
    if pending_retry is not None:
        assert claim is not None
        current = pending_retry
        if type(current) is BoundFeedbackArtifactV4:
            source = next(
                item
                for item in typed_sources
                if item.selection_entry_sha256 == current.selection_entry_sha256
            )
            retry = _run_source_v4(
                prepared,
                source,
                feedback_runner,
                attempt_index=2,
                first_artifact=current,
                retry_claim=claim,
            )
            artifacts[retry.selection_entry_sha256] = retry
            if retry.status != "parsed":
                _publish_terminal_run(prepared, artifacts)
                raise PortfolioS1FeedbackRunError(
                    "the resumed Feedback schema retry failed; run stopped"
                )


def _execute_run_locked(
    prepared: PreparedPortfolioS1FeedbackRun,
    *,
    feedback_runner: FeedbackRunner = run_visual_feedback,
    stop_after_canary: bool = False,
    stop_after_count: int | None = None,
) -> (
    PortfolioS1FeedbackBundleV1
    | PortfolioS1FeedbackBundleV4
    | PortfolioS1FeedbackBundleV5
    | PortfolioS1FeedbackBundleV6
    | PortfolioS1FeedbackBundleV7
    | None
):
    if type(prepared.authorization) in {
        PortfolioS1QwenFeedbackAuthorizationV3,
        PortfolioS1FeedbackAuthorizationV4,
        PortfolioS1FeedbackAuthorizationV5,
        PortfolioS1FeedbackAuthorizationV6,
    }:
        _require_qwen_execute_environment()
    artifacts = _resume_all_before_provider(prepared)
    by_hash = {item.selection_entry_sha256: item for item in prepared.sources}
    if type(prepared.selection) is PortfolioS1FeedbackSelectionV2:
        target = 12 if stop_after_canary else stop_after_count
        for phase_count in prepared.selection.phase_counts:
            phase_sources = tuple(
                by_hash[item.entry_sha256]
                for item in prepared.selection.entries[:phase_count]
            )
            _execute_phase(prepared, phase_sources, artifacts, feedback_runner)
            if any(
                item.entry_sha256 not in artifacts
                or artifacts[item.entry_sha256].status != "parsed"
                for item in prepared.selection.entries[:phase_count]
            ):
                raise PortfolioS1FeedbackRunError(
                    f"Feedback phase did not complete parsed{phase_count}"
                )
            if target == phase_count:
                return None
    else:
        canary = tuple(
            by_hash[item] for item in prepared.selection.canary_entry_sha256s
        )
        remaining = tuple(
            by_hash[item] for item in prepared.selection.remaining_entry_sha256s
        )
        _execute_phase(prepared, canary, artifacts, feedback_runner)
        if any(
            item.entry_sha256 not in artifacts
            for item in prepared.selection.entries
            if item.entry_sha256 in prepared.selection.canary_entry_sha256s
        ):
            raise PortfolioS1FeedbackRunError(
                "Feedback canary did not complete parsed6"
            )
        if stop_after_canary:
            return None
        _execute_phase(prepared, remaining, artifacts, feedback_runner)
    ordered = tuple(artifacts[item.entry_sha256] for item in prepared.selection.entries)
    run = _publish_terminal_run(prepared, artifacts)
    if type(prepared.authorization) is PortfolioS1FeedbackAuthorizationV6:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
        assert type(prepared.control) is PortfolioS1FeedbackControlV11
        assert type(run) is PortfolioS1FeedbackRunV4
        all_artifacts, orphans, _latest = _attempt_state_v4(prepared)
        if orphans:
            raise PortfolioS1FeedbackRunError(
                "completed Feedback v2 run unexpectedly contains an orphan"
            )
        bundle = build_portfolio_s1_feedback_bundle_v7(
            prepared.selection,
            prepared.control,
            prepared.authorization,
            all_artifacts,
            run,
            retry_claim=_load_retry_claim_v1(prepared),
        )
    elif type(prepared.authorization) is PortfolioS1FeedbackAuthorizationV5:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
        assert type(prepared.control) is PortfolioS1FeedbackControlV10
        assert type(run) is PortfolioS1FeedbackRunV3
        bundle = build_portfolio_s1_feedback_bundle_v6(
            prepared.selection,
            prepared.control,
            prepared.authorization,
            ordered,
            run,
        )
    elif type(prepared.authorization) is PortfolioS1FeedbackAuthorizationV4:
        assert type(prepared.selection) is PortfolioS1FeedbackSelectionV2
        assert type(prepared.control) is PortfolioS1FeedbackControlV9
        assert type(run) is PortfolioS1FeedbackRunV2
        bundle = build_portfolio_s1_feedback_bundle_v5(
            prepared.selection,
            prepared.control,
            prepared.authorization,
            ordered,
            run,
        )
    elif type(prepared.authorization) is PortfolioS1QwenFeedbackAuthorizationV3:
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
    if type(bundle) is PortfolioS1FeedbackBundleV7:
        loaded = load_portfolio_s1_feedback_bundle_v7(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    elif type(bundle) is PortfolioS1FeedbackBundleV6:
        loaded = load_portfolio_s1_feedback_bundle_v6(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    elif type(bundle) is PortfolioS1FeedbackBundleV5:
        loaded = load_portfolio_s1_feedback_bundle_v5(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    elif type(bundle) is PortfolioS1FeedbackBundleV4:
        loaded = load_portfolio_s1_feedback_bundle_v4(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    else:
        loaded = load_portfolio_s1_feedback_bundle(
            bundle_path, expected_file_sha256=bundle_file_sha256
        )
    if loaded != bundle:
        raise PortfolioS1FeedbackRunError("S1 Feedback bundle resume conflict")
    return bundle


def execute_run(
    prepared: PreparedPortfolioS1FeedbackRun,
    *,
    feedback_runner: FeedbackRunner = run_visual_feedback,
    stop_after_canary: bool = False,
    stop_after_count: int | None = None,
) -> (
    PortfolioS1FeedbackBundleV1
    | PortfolioS1FeedbackBundleV4
    | PortfolioS1FeedbackBundleV5
    | PortfolioS1FeedbackBundleV6
    | PortfolioS1FeedbackBundleV7
    | None
):
    with _exclusive_execute_writer_lock(prepared.output_dir):
        return _execute_run_locked(
            prepared,
            feedback_runner=feedback_runner,
            stop_after_canary=stop_after_canary,
            stop_after_count=stop_after_count,
        )


def _summary(
    mode: Literal["prepare", "dry-run", "execute"],
    prepared: PreparedPortfolioS1FeedbackRun,
    bundle: (
        PortfolioS1FeedbackBundleV1
        | PortfolioS1FeedbackBundleV4
        | PortfolioS1FeedbackBundleV5
        | PortfolioS1FeedbackBundleV6
        | PortfolioS1FeedbackBundleV7
        | None
    ) = None,
    *,
    stop_after_canary: bool = False,
    stop_after_count: int | None = None,
) -> dict[str, object]:
    fresh_retry = type(prepared.control) is PortfolioS1FeedbackControlV11
    existing = (
        sum(
            _artifact_path_v4(prepared, item, attempt).exists()
            for item in prepared.sources
            for attempt in (1, 2)
        )
        if fresh_retry
        else sum(_artifact_path(prepared, item).exists() for item in prepared.sources)
    )
    budget = _budget_snapshot(prepared)
    reserved = budget.provider_calls_reserved
    round2 = type(prepared.selection) is PortfolioS1FeedbackSelectionV2
    hard_cap = Decimal(
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
        if round2
        else QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    )
    return {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-cli-summary",
        "mode": mode,
        "provider_calls_performed_by_dry_run": 0 if mode != "execute" else None,
        "selection_profile": "discovery240" if round2 else "selected48",
        "selected_count": len(prepared.selection.entries),
        "phase_counts": list(prepared.selection.phase_counts) if round2 else [6, 48],
        "canary_count": 12 if round2 else 6,
        "remaining_count": 228 if round2 else 42,
        "feedback_concurrency": 2,
        "max_attempts": 1,
        "max_attempts_per_retried_query": 2 if fresh_retry else 1,
        "global_retry_token_count": 1 if fresh_retry else 0,
        "global_retry_claimed": (
            _retry_claim_path(prepared).exists() if fresh_retry else False
        ),
        "retry_attempt_reservation_count": (
            sum(
                _reservation_path_v4(prepared, item, 2).exists()
                for item in prepared.sources
            )
            if fresh_retry
            else 0
        ),
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
        "stop_after_count": stop_after_count,
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
    parser.add_argument(
        "--selection-profile",
        choices=("selected48", "discovery240"),
        default="selected48",
    )
    parser.add_argument("--creator-selection-index", type=Path)
    parser.add_argument("--expected-creator-selection-sha256")
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--expected-fold-manifest-sha256")
    parser.add_argument("--fold-mapping", type=Path)
    parser.add_argument("--expected-fold-mapping-sha256")
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--expected-rubric-sha256", required=True)
    parser.add_argument("--rubric-id", required=True)
    parser.add_argument("--rubric-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--owner-statement", required=True)
    parser.add_argument(
        "--approved-phase-hard-cap-cny",
        help=(
            "explicit owner-approved phase cap; discovery240 Qwen3.8-Max "
            "requires exactly 94.000000000000"
        ),
    )
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
            "execute and verify only the first frozen phase (12 for active "
            "Discovery240); a later execute resumes those exact bytes"
        ),
    )
    parser.add_argument(
        "--stop-after-count",
        type=int,
        choices=(12, 60, 120),
        help=(
            "for discovery240 execute/resume only through the fixed cumulative "
            "12, 60, or 120 boundary; omit to complete 240"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if (
            arguments.mode == "execute"
            and arguments.selection_profile != "discovery240"
        ):
            raise PortfolioS1FeedbackRunError(
                "live Feedback execution is permanently restricted to Qwen3.8-Max "
                "Discovery240; selected48 is historical prepare/dry-run only"
            )
        if arguments.stop_after_canary and arguments.mode != "execute":
            raise PortfolioS1FeedbackRunError(
                "--stop-after-canary is only valid with --mode execute"
            )
        if arguments.stop_after_count is not None and (
            arguments.mode != "execute" or arguments.selection_profile != "discovery240"
        ):
            raise PortfolioS1FeedbackRunError(
                "--stop-after-count is only valid for discovery240 execute"
            )
        if arguments.stop_after_canary and arguments.stop_after_count is not None:
            raise PortfolioS1FeedbackRunError(
                "choose only one Feedback phase stop option"
            )
        if arguments.selection_profile == "discovery240" and (
            arguments.approved_phase_hard_cap_cny
            != QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
        ):
            raise PortfolioS1FeedbackRunError(
                "discovery240 Qwen3.8-Max requires the explicit owner-approved CNY94 cap"
            )
        if arguments.selection_profile != "discovery240" and (
            arguments.approved_phase_hard_cap_cny is not None
        ):
            raise PortfolioS1FeedbackRunError(
                "the Qwen3.8-Max approved phase cap is only valid for discovery240"
            )
        prepared = prepare_run(arguments)
        if arguments.mode == "dry-run":
            _resume_all_before_provider(prepared)
        bundle = (
            execute_run(
                prepared,
                stop_after_canary=arguments.stop_after_canary,
                stop_after_count=arguments.stop_after_count,
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
                    stop_after_count=arguments.stop_after_count,
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
