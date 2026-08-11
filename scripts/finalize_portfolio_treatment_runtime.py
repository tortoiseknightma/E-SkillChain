"""Finalize one matrix-ready Portfolio runtime from real treatment evidence.

This command performs no model calls.  It consumes externally SHA-bound
LLMStatic/S1/S2/S3 model artifacts and three stage gates, replays every Bank
construction, applies each accept/rollback decision, and publishes a
self-contained create-only runtime.  A successful publish is accepted only
after :func:`load_verified_portfolio_treatment_runtime` re-opens the result.

Historical deterministic scaffold directories are never accepted as model
stage inputs and are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import Any, Callable, Literal, Mapping

from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_evolution_inputs import (  # noqa: E402
    PortfolioEvolutionInputPacket,
)
from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from scripts.run_codex_authoring import _event_audit  # noqa: E402
from scripts.run_portfolio_evolution_model import (  # noqa: E402
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    INVOCATION_POLICY_VERSION,
    S3_IMPLEMENTATION_VERSION,
    S3_INVOCATION_POLICY_VERSION,
    S3_TEXTOPT_COMPILER_SNAPSHOT_FILE,
    EvolutionInvocationReceipt,
    _actionable_mutation_scope,
    _parse_and_compile_s3_text_patch,
    _parse_mutation_changes,
)
from skillchain.codex_authoring import normalize_codex_authoring_output  # noqa: E402
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    CARD_REQUIREMENT_GUARD_POLICY_SHA256,
    CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
    FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS,
    FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    FINAL_JUDGE_RETRY_POLICY_SHA256,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_THINKING_BUDGET,
    FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
)
from skillchain.evaluation.portfolio_attribution import (  # noqa: E402
    PortfolioAttributionError,
    PortfolioParentAttributionPacket,
    verify_portfolio_parent_attribution_gate_binding,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    VerifiedPortfolioTreatmentRuntime,
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_s3_textopt import (  # noqa: E402
    PortfolioS3RejectedEditBuffer,
    parse_portfolio_s3_rejected_edit_buffer,
    parse_portfolio_s3_text_patch_artifact,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
    PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
    PortfolioArtifactBinding,
    PortfolioBankBinding,
    PortfolioModelInvocationReceipt,
    PortfolioStageGateReport,
    PortfolioStageGateResultSet,
    PortfolioStageMutation,
    PortfolioTreatmentChainManifest,
    PortfolioTreatmentConfig,
    PortfolioTreatmentError,
    PortfolioTreatmentReceipt,
    apply_portfolio_stage_mutation,
    build_portfolio_execution_artifact_aliases,
    build_portfolio_model_invocation_receipt,
    compile_portfolio_s1_creator_bank,
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
    verify_portfolio_stage_gate_evidence,
)
from skillchain.runners.assistant import (  # noqa: E402
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
    evolution_bank_boundary_violations,
)
from skillchain.static_authoring import (  # noqa: E402
    AuthoringDraftBundle,
    StaticBankArtifact,
    build_authoring_draft_bundle,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
    PortfolioToolRuntime,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_ORDER: tuple[PortfolioTreatmentConfig, ...] = (
    "llm_static",
    "s1",
    "s1s2",
    "full",
)
_STAGE_CONFIG = {
    "s1": "s1",
    "s2": "s1s2",
    "s3": "full",
}
_STAGE_NAME = {
    "s1": "s1_creator",
    "s2": "s2_route_optimizer",
    "s3": "s3_body_refiner",
}
_INPUT_KIND = {
    "s1": "trajectory_bundle",
    "s2": "route_examples",
    "s3": "body_attribution",
}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_EVENT_BYTES = 16 * 1024 * 1024
_FINALIZER_PATH = Path(__file__).resolve()


class PortfolioTreatmentFinalizationError(ValueError):
    """An externally bound finalization input failed closed."""


def _require_current_mutation_implementation(
    source_key: Literal["s1", "s2", "s3"],
    detailed: EvolutionInvocationReceipt,
) -> None:
    """Keep a SHA-bound real S1 compatible; require current S2/S3 algorithms."""

    expected = {
        "s2": (
            IMPLEMENTATION_VERSION,
            INVOCATION_POLICY_VERSION,
            "new_persistent",
        ),
        "s3": (
            S3_IMPLEMENTATION_VERSION,
            S3_INVOCATION_POLICY_VERSION,
            "ephemeral",
        ),
    }.get(source_key)
    if expected is not None and (
        detailed.implementation_id != IMPLEMENTATION_ID
        or detailed.implementation_version != expected[0]
        or detailed.policy_version != expected[1]
        or detailed.session_mode != expected[2]
    ):
        raise PortfolioTreatmentFinalizationError(
            f"{source_key} does not use its required stage-specific implementation"
        )


@dataclass(frozen=True)
class StageSource:
    key: Literal["s1", "s2", "s3"]
    root: Path
    expected_invocation_receipt_file_sha256: str
    implementation_source_path: Path | None = None
    expected_implementation_source_file_sha256: str | None = None


@dataclass(frozen=True)
class GateSource:
    key: Literal["s1", "s2", "s3"]
    report_path: Path
    expected_report_file_sha256: str
    parent_results_path: Path
    candidate_results_path: Path


@dataclass(frozen=True)
class FinalizationInputs:
    output_dir: Path
    tool_runtime_source_root: Path
    expected_tool_runtime_source_lock_file_sha256: str
    semantic_authoring_input_path: Path
    expected_semantic_authoring_input_file_sha256: str
    codex_authoring_input_path: Path
    expected_codex_authoring_input_file_sha256: str
    static_authoring_root: Path
    expected_static_invocation_receipt_file_sha256: str
    static_human_review_path: Path
    expected_static_human_review_file_sha256: str
    stages: tuple[StageSource, StageSource, StageSource]
    gates: tuple[GateSource, GateSource, GateSource]
    optimization_batch_id: str = "dev-mini-001-r3"


@dataclass(frozen=True)
class FinalizedRuntime:
    root: Path
    runtime_lock_file_sha256: str
    runtime_lock_sha256: str
    treatment_chain_manifest_file_sha256: str
    treatment_chain_sha256: str
    verified: VerifiedPortfolioTreatmentRuntime


@dataclass(frozen=True)
class _ToolSource:
    runtime: PortfolioToolRuntime
    source_lock: dict[str, Any]
    source_lock_file_sha256: str
    source_lock_sha256: str
    recipe_bytes: bytes
    recipe_rows: int


@dataclass(frozen=True)
class _StaticSource:
    bank: StaticBankArtifact
    invocation: PortfolioModelInvocationReceipt
    implementation_file_sha256: str
    semantic_bytes: bytes
    codex_bytes: bytes
    draft_bytes: bytes
    review_bytes: bytes
    prompt_bytes: bytes
    event_bytes: bytes
    stderr_bytes: bytes
    raw_bytes: bytes
    detailed_receipt_bytes: bytes
    common_authoring_input_sha256: str


@dataclass(frozen=True)
class _EvolutionSource:
    key: Literal["s1", "s2", "s3"]
    candidate: StaticBankArtifact
    mutation: PortfolioStageMutation | None
    invocation: PortfolioModelInvocationReceipt
    detailed: EvolutionInvocationReceipt
    stage_input: PortfolioEvolutionInputPacket | PortfolioParentAttributionPacket
    stage_input_bytes: bytes
    rejected_edit_buffer_bytes: bytes | None
    implementation_source_bytes: bytes
    source_files: Mapping[str, bytes]


@dataclass(frozen=True)
class _GateEvidence:
    key: Literal["s1", "s2", "s3"]
    report: PortfolioStageGateReport
    parent_results: PortfolioStageGateResultSet
    candidate_results: PortfolioStageGateResultSet
    report_bytes: bytes
    parent_results_bytes: bytes
    candidate_results_bytes: bytes


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise PortfolioTreatmentFinalizationError(f"{label} is not a SHA-256")
    return value


def _read_file(
    path: str | Path,
    *,
    label: str,
    expected_file_sha256: str | None = None,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    resolved = Path(path).absolute()
    try:
        content = read_stable_regular_file(
            resolved,
            label=label,
            max_bytes=max_bytes,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioTreatmentFinalizationError(
            f"{label} cannot be read safely"
        ) from error
    if expected_file_sha256 is not None:
        expected = _require_sha256(expected_file_sha256, f"{label} expected digest")
        if sha256_bytes(content) != expected:
            raise PortfolioTreatmentFinalizationError(f"{label} file SHA-256 mismatch")
    return content


def _canonical_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise PortfolioTreatmentFinalizationError(
            f"{label} is not canonical JSON"
        ) from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise PortfolioTreatmentFinalizationError(
            f"{label} must be one canonical JSON object"
        )
    return raw


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if (
        not isinstance(observed, str)
        or not _SHA256_RE.fullmatch(observed)
        or observed != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise PortfolioTreatmentFinalizationError(f"{label} {field} mismatch")
    return observed


def _relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PortfolioTreatmentFinalizationError(
            f"{label} must be a normalized relative POSIX path"
        )
    return value


def _write_unique(root: Path, relative: str, content: bytes) -> Path:
    normalized = _relative_path(relative, label="runtime output path")
    path = root / Path(normalized)
    if os.path.lexists(path):
        if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
            return path
        raise PortfolioTreatmentFinalizationError(
            f"runtime output path collision: {normalized}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _runtime_sources(recipe_evidence: Path) -> PortfolioRuntimeSources:
    clean = REPOSITORY_ROOT / "data" / "clean"
    return PortfolioRuntimeSources(
        selection_manifest=clean / "query_images" / "selection-manifest.json",
        dataset_assets=clean / "query_images" / "dataset-assets.jsonl",
        runtime_catalog_assets=(
            clean / "portfolio-mini-asset-catalog-v3" / "assets.jsonl"
        ),
        rpc_scenes=clean / "rpc-multi-product-query-v1" / "scenes.jsonl",
        inaturalist_manifest=(
            clean
            / "portfolio-source-pools"
            / "encyclopedia-inaturalist"
            / "manifest.jsonl"
        ),
        recipe_evidence=recipe_evidence,
    )


def _recipe_rows(content: bytes) -> int:
    if not content or not content.endswith(b"\n"):
        raise PortfolioTreatmentFinalizationError(
            "tool runtime recipe evidence must be non-empty JSONL"
        )
    count = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line:
            continue
        raw = _canonical_object(
            line + b"\n",
            label=f"recipe evidence row {line_number}",
        )
        if not raw:
            raise PortfolioTreatmentFinalizationError(
                "recipe evidence contains an empty object"
            )
        count += 1
    if count == 0:
        raise PortfolioTreatmentFinalizationError(
            "tool runtime recipe evidence contains no rows"
        )
    return count


def _load_tool_source(inputs: FinalizationInputs) -> _ToolSource:
    source_root = inputs.tool_runtime_source_root.absolute()
    if not source_root.is_dir():
        raise PortfolioTreatmentFinalizationError(
            "tool runtime source root is not a directory"
        )
    lock_bytes = _read_file(
        source_root / "runtime-lock.json",
        label="tool runtime source lock",
        expected_file_sha256=(inputs.expected_tool_runtime_source_lock_file_sha256),
    )
    lock = _canonical_object(lock_bytes, label="tool runtime source lock")
    lock_sha256 = _validate_self_hash(
        lock,
        field="runtime_lock_sha256",
        label="tool runtime source lock",
    )
    if (
        lock.get("kind") != "portfolio-assistant-runtime-lock"
        or lock.get("track") != "portfolio"
        or lock.get("formal_eligible") is not False
        or lock.get("policy_version") != PORTFOLIO_TOOL_RUNTIME_POLICY
    ):
        raise PortfolioTreatmentFinalizationError(
            "tool runtime source lock is not a Portfolio public-data runtime"
        )
    prompt_path = source_root / "system-prompt.txt"
    try:
        with prompt_path.open("r", encoding="utf-8", newline=None) as stream:
            normalized_prompt = stream.read()
    except (OSError, UnicodeDecodeError) as error:
        raise PortfolioTreatmentFinalizationError(
            "tool runtime system prompt is not readable UTF-8"
        ) from error
    if normalized_prompt != PORTFOLIO_SYSTEM_PROMPT or lock.get(
        "system_prompt_sha256"
    ) != sha256_bytes(normalized_prompt.encode("utf-8")):
        raise PortfolioTreatmentFinalizationError(
            "tool runtime source prompt differs from the frozen prompt"
        )
    recipe_path = source_root / "recipe-evidence.jsonl"
    recipe_bytes = _read_file(
        recipe_path,
        label="tool runtime recipe evidence",
    )
    rows = _recipe_rows(recipe_bytes)
    runtime = build_portfolio_tool_runtime(_runtime_sources(recipe_path))
    if (
        runtime.registry.registry_sha256 != lock.get("tool_registry_sha256")
        or runtime.registry.registry_runtime_sha256
        != lock.get("tool_registry_runtime_sha256")
        or runtime.index.runtime_data_sha256 != lock.get("runtime_data_sha256")
        or list(runtime.index.source_sha256s) != lock.get("source_sha256s")
    ):
        raise PortfolioTreatmentFinalizationError(
            "rebuilt public-data tool runtime differs from its source lock"
        )
    return _ToolSource(
        runtime=runtime,
        source_lock=lock,
        source_lock_file_sha256=sha256_bytes(lock_bytes),
        source_lock_sha256=lock_sha256,
        recipe_bytes=recipe_bytes,
        recipe_rows=rows,
    )


def _validate_static_review(
    review_bytes: bytes,
    *,
    invocation_receipt_file_sha256: str,
    pre_review_file_sha256: str,
    pre_review: AuthoringDraftBundle,
    codex_input_sha256: str,
) -> str:
    review = _canonical_object(review_bytes, label="LLMStatic human review")
    review_sha256 = _validate_self_hash(
        review,
        field="receipt_sha256",
        label="LLMStatic human review",
    )
    human = review.get("human_review")
    if (
        review.get("artifact_kind") != "codex_authoring_human_review"
        or review.get("decision") != "accepted_unchanged"
        or review.get("canonical_invocation_receipt_file_sha256")
        != invocation_receipt_file_sha256
        or review.get("pre_review_file_sha256") != pre_review_file_sha256
        or review.get("pre_review_bundle_sha256") != pre_review.bundle_sha256
        or review.get("authoring_input_sha256") != codex_input_sha256
        or not isinstance(human, dict)
        or human.get("changed") is not False
        or human.get("post_review_sha256") != pre_review_file_sha256
        or human.get("post_review_canonical_json")
        != pre_review.canonical_bytes().decode("utf-8")
    ):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic human review does not accept the exact pre-review draft"
        )
    return review_sha256


def _load_static_source(
    inputs: FinalizationInputs,
    *,
    tool_runtime_sha256: str,
) -> _StaticSource:
    root = inputs.static_authoring_root.absolute()
    if not root.is_dir():
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic authoring root is not a directory"
        )
    detailed_bytes = _read_file(
        root / "invocation-receipt.json",
        label="LLMStatic detailed invocation receipt",
        expected_file_sha256=(inputs.expected_static_invocation_receipt_file_sha256),
    )
    detailed = _canonical_object(
        detailed_bytes,
        label="LLMStatic detailed invocation receipt",
    )
    _validate_self_hash(
        detailed,
        field="receipt_payload_sha256",
        label="LLMStatic detailed invocation receipt",
    )
    bundle_files = detailed.get("canonical_bundle_files")
    if not isinstance(bundle_files, dict):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic invocation receipt lacks its canonical bundle"
        )

    def bundle_file(name: str, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
        expected = bundle_files.get(name)
        if not isinstance(expected, str):
            raise PortfolioTreatmentFinalizationError(
                f"LLMStatic invocation receipt lacks {name}"
            )
        return _read_file(
            root / name,
            label=f"LLMStatic {name}",
            expected_file_sha256=expected,
            max_bytes=max_bytes,
        )

    prompt_bytes = bundle_file("authoring-stdin.txt")
    event_bytes = bundle_file("codex-events.jsonl", max_bytes=_MAX_EVENT_BYTES)
    stderr_bytes = bundle_file("codex-stderr.bin")
    raw_bytes = bundle_file("author-content.raw.json")
    draft_bytes = bundle_file("pre-review-draft.json")

    if (
        detailed.get("status") != "codex_session_completed_draft_ready_for_review"
        or detailed.get("execution_type") != "codex_mediated_static_author_v1"
        or detailed.get("exit_code") != 0
        or detailed.get("authorization_consumed") is not True
        or detailed.get("visible_agent_message_count") != 1
        or detailed.get("visible_tool_activity") is not False
        or detailed.get("repair_performed") is not False
        or detailed.get("fallback_performed") is not False
        or detailed.get("followup_performed") is not False
        or detailed.get("repository_retry_performed") is not False
        or detailed.get("raw_final_sha256") != sha256_bytes(raw_bytes)
        or detailed.get("event_log_sha256") != sha256_bytes(event_bytes)
        or detailed.get("stderr_sha256") != sha256_bytes(stderr_bytes)
        or detailed.get("stdin_request_file_sha256") != sha256_bytes(prompt_bytes)
    ):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic invocation was not one accepted clean model turn"
        )
    try:
        audit = _event_audit(event_bytes)
        audit.require_formal_success()
    except Exception as error:
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic Codex event evidence is invalid"
        ) from error
    if (
        audit.thread_id != detailed.get("thread_id")
        or audit.input_tokens != detailed.get("input_tokens")
        or audit.output_tokens != detailed.get("output_tokens")
        or not audit.agent_messages
        or raw_bytes
        not in {
            audit.agent_messages[0].encode("utf-8"),
            audit.agent_messages[0].encode("utf-8") + b"\n",
        }
    ):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic event audit differs from its terminal artifacts"
        )

    rebind = load_verified_codex_draft_rebind(
        codex_input_path=inputs.codex_authoring_input_path,
        expected_codex_input_file_sha256=(
            inputs.expected_codex_authoring_input_file_sha256
        ),
        semantic_input_path=inputs.semantic_authoring_input_path,
        expected_semantic_input_file_sha256=(
            inputs.expected_semantic_authoring_input_file_sha256
        ),
        draft_path=root / "pre-review-draft.json",
        expected_draft_file_sha256=sha256_bytes(draft_bytes),
    )
    if (
        detailed.get("authoring_input_file_sha256")
        != inputs.expected_codex_authoring_input_file_sha256
    ):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic invocation used another Codex authoring packet"
        )
    review_bytes = _read_file(
        inputs.static_human_review_path,
        label="LLMStatic human review",
        expected_file_sha256=inputs.expected_static_human_review_file_sha256,
    )
    _validate_static_review(
        review_bytes,
        invocation_receipt_file_sha256=sha256_bytes(detailed_bytes),
        pre_review_file_sha256=sha256_bytes(draft_bytes),
        pre_review=rebind.source_draft,
        codex_input_sha256=rebind.codex_input.input_sha256,
    )
    bank = compile_verified_codex_llm_static_bank(
        rebind,
        tool_registry_runtime_sha256=tool_runtime_sha256,
    )
    requested_model = detailed.get("requested_model")
    reasoning_effort = detailed.get("reasoning_effort")
    if (
        not isinstance(requested_model, str)
        or reasoning_effort not in {"low", "medium", "high", "xhigh", "max", "ultra"}
        or audit.thread_id is None
        or audit.input_tokens is None
        or audit.output_tokens is None
        or audit.input_tokens <= 0
        or audit.output_tokens <= 0
    ):
        raise PortfolioTreatmentFinalizationError(
            "LLMStatic invocation lacks usable model identity or token evidence"
        )
    invocation = build_portfolio_model_invocation_receipt(
        stage="static_author",
        requested_model=requested_model,
        effort=reasoning_effort,
        thread_id=audit.thread_id,
        prompt_sha256=sha256_bytes(prompt_bytes),
        output_sha256=sha256_bytes(raw_bytes),
        event_log_sha256=sha256_bytes(event_bytes),
        stderr_sha256=sha256_bytes(stderr_bytes),
        input_tokens=audit.input_tokens,
        output_tokens=audit.output_tokens,
    )
    semantic_bytes = _read_file(
        inputs.semantic_authoring_input_path,
        label="semantic authoring input final snapshot",
        expected_file_sha256=inputs.expected_semantic_authoring_input_file_sha256,
    )
    codex_bytes = _read_file(
        inputs.codex_authoring_input_path,
        label="Codex authoring input final snapshot",
        expected_file_sha256=inputs.expected_codex_authoring_input_file_sha256,
    )
    return _StaticSource(
        bank=bank,
        invocation=invocation,
        implementation_file_sha256=sha256_bytes(detailed_bytes),
        semantic_bytes=semantic_bytes,
        codex_bytes=codex_bytes,
        draft_bytes=draft_bytes,
        review_bytes=review_bytes,
        prompt_bytes=prompt_bytes,
        event_bytes=event_bytes,
        stderr_bytes=stderr_bytes,
        raw_bytes=raw_bytes,
        detailed_receipt_bytes=detailed_bytes,
        common_authoring_input_sha256=rebind.semantic_input.input_sha256,
    )


def _load_model[T](
    content: bytes,
    model_type: type[T],
    *,
    label: str,
) -> T:
    _canonical_object(content, label=label)
    try:
        value = model_type.model_validate_json(  # type: ignore[attr-defined]
            content,
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentFinalizationError(f"{label} is invalid") from error
    canonical = getattr(value, "canonical_bytes", None)
    expected = (
        canonical()
        if canonical is not None
        else canonical_json_bytes(value.model_dump(mode="json"))
    )
    if expected != content:
        raise PortfolioTreatmentFinalizationError(f"{label} bytes are not canonical")
    return value


def _safe_stage_filename(value: str) -> str:
    if (
        not value
        or value != Path(value).name
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise PortfolioTreatmentFinalizationError(
            "evolution receipt output name is not a safe basename"
        )
    return value


def _load_evolution_source(
    source: StageSource,
    *,
    parent_bank: StaticBankArtifact | None,
    prior_stage: _EvolutionSource | None,
    prior_gate: PortfolioStageGateReport | None,
    semantic_bytes: bytes,
    codex_bytes: bytes,
    common_authoring_input_sha256: str,
    tool_registry_runtime_sha256: str,
) -> _EvolutionSource:
    root = source.root.absolute()
    if not root.is_dir():
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} model stage root is not a directory"
        )
    detailed_bytes = _read_file(
        root / "invocation-receipt.json",
        label=f"{source.key} detailed invocation receipt",
        expected_file_sha256=source.expected_invocation_receipt_file_sha256,
    )
    detailed = _load_model(
        detailed_bytes,
        EvolutionInvocationReceipt,
        label=f"{source.key} detailed invocation receipt",
    )
    expected_stage = _STAGE_NAME[source.key]
    if detailed.status != "completed" or detailed.stage != expected_stage:
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} is not a completed invocation of the live evolution runner"
        )
    _require_current_mutation_implementation(source.key, detailed)
    implementation_path = (
        source.implementation_source_path
        if source.implementation_source_path is not None
        else REPOSITORY_ROOT / "scripts" / "run_portfolio_evolution_model.py"
    )
    implementation_expected = (
        source.expected_implementation_source_file_sha256
        if source.expected_implementation_source_file_sha256 is not None
        else detailed.implementation_file_sha256
    )
    implementation_source_bytes = _read_file(
        implementation_path,
        label=f"{source.key} evolution implementation source",
        expected_file_sha256=implementation_expected,
    )
    if sha256_bytes(implementation_source_bytes) != detailed.implementation_file_sha256:
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} implementation source differs from its invocation"
        )

    expected_outputs = {
        _safe_stage_filename(binding.file): binding.file_sha256
        for binding in detailed.output_files
    }
    expected_source_names = set(expected_outputs) | {
        "invocation-receipt.json",
        "model-invocation-receipt.json",
    }
    actual_source_names: set[str] = set()
    for item in root.iterdir():
        metadata = item.lstat()
        reparse = int(getattr(metadata, "st_file_attributes", 0)) & int(
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        if stat.S_ISLNK(metadata.st_mode) or reparse:
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} model stage contains a symlink/reparse point"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} model stage contains a non-file entry"
            )
        actual_source_names.add(item.name)
    if actual_source_names != expected_source_names:
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} model stage file set differs from its receipt"
        )
    source_files: dict[str, bytes] = {
        "invocation-receipt.json": detailed_bytes,
    }
    for name, expected in expected_outputs.items():
        source_files[name] = _read_file(
            root / name,
            label=f"{source.key} model output {name}",
            expected_file_sha256=expected,
            max_bytes=(
                _MAX_EVENT_BYTES if name == "codex-events.jsonl" else _MAX_JSON_BYTES
            ),
        )
    if source.key == "s3":
        compiler_source = source_files.get(S3_TEXTOPT_COMPILER_SNAPSHOT_FILE)
        if (
            compiler_source is None
            or detailed.s3_textopt_compiler_file_sha256 != sha256_bytes(compiler_source)
        ):
            raise PortfolioTreatmentFinalizationError(
                "S3 TextOpt compiler snapshot differs from its receipt"
            )
    core_bytes = _read_file(
        root / "model-invocation-receipt.json",
        label=f"{source.key} core invocation receipt",
    )
    core = _load_model(
        core_bytes,
        PortfolioModelInvocationReceipt,
        label=f"{source.key} core invocation receipt",
    )
    source_files["model-invocation-receipt.json"] = core_bytes
    if (
        core.stage != expected_stage
        or core.requested_model != detailed.requested_model
        or core.effort != detailed.reasoning_effort
        or core.thread_id != detailed.thread_id
        or core.input_tokens != detailed.input_tokens
        or core.output_tokens != detailed.output_tokens
        or core.prompt_sha256 != sha256_bytes(source_files["prompt.txt"])
        or core.output_sha256 != sha256_bytes(source_files["raw-model-output.json"])
        or core.event_log_sha256 != sha256_bytes(source_files["codex-events.jsonl"])
        or core.stderr_sha256 != sha256_bytes(source_files["codex-stderr.bin"])
    ):
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} core and detailed invocation evidence differ"
        )

    input_by_role = {item.role: item for item in detailed.input_files}
    if source.key == "s1":
        expected_roles = {
            "stage_input",
            "semantic_authoring_input",
            "codex_authoring_input",
        }
    elif source.key == "s3":
        expected_roles = {
            "stage_input",
            "parent_bank",
            "prior_stage_gate",
        }
        if any(item.role == "rejected_edit_buffer" for item in detailed.input_files):
            expected_roles.add("rejected_edit_buffer")
    elif detailed.session_mode == "resume":
        expected_roles = {
            "stage_input",
            "parent_bank",
            "prior_invocation_receipt",
            "prior_stage_gate",
        }
    else:
        expected_roles = {"stage_input", "parent_bank"}
    if set(input_by_role) != expected_roles:
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} invocation input role set is invalid"
        )
    input_contents: dict[str, bytes] = {}
    for role, binding in input_by_role.items():
        content = _read_file(
            binding.path,
            label=f"{source.key} invocation input {role}",
            expected_file_sha256=binding.file_sha256,
        )
        if binding.content_sha256 != sha256_bytes(content):
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} invocation input content digest differs"
            )
        input_contents[role] = content
    rejected_edit_buffer: PortfolioS3RejectedEditBuffer | None = None
    rejected_edit_buffer_bytes = input_contents.get("rejected_edit_buffer")
    if rejected_edit_buffer_bytes is not None:
        try:
            rejected_edit_buffer = parse_portfolio_s3_rejected_edit_buffer(
                rejected_edit_buffer_bytes
            )
        except PortfolioTreatmentError as error:
            raise PortfolioTreatmentFinalizationError(
                "S3 rejected-edit buffer is invalid"
            ) from error
    if source.key == "s1":
        if (
            input_contents["semantic_authoring_input"] != semantic_bytes
            or input_contents["codex_authoring_input"] != codex_bytes
            or parent_bank is not None
        ):
            raise PortfolioTreatmentFinalizationError(
                "S1 invocation used another common authoring contract"
            )
    elif (
        parent_bank is None
        or input_contents["parent_bank"] != parent_bank.canonical_bytes()
    ):
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} invocation did not consume its gated parent Bank"
        )

    if source.key == "s3" and detailed.session_mode == "ephemeral":
        if prior_stage is None or prior_gate is None:
            raise PortfolioTreatmentFinalizationError(
                "S3 ephemeral lineage lacks its immediately prior S2 stage/gate"
            )
        bound_gate = _load_model(
            input_contents["prior_stage_gate"],
            PortfolioStageGateReport,
            label="S3 bound selected-parent S2 gate report",
        )
        selected_parent_sha256 = (
            bound_gate.candidate_bank_sha256
            if bound_gate.decision == "accepted"
            else bound_gate.parent_bank_sha256
        )
        if (
            prior_stage.key != "s2"
            or prior_stage.detailed.implementation_id != IMPLEMENTATION_ID
            or prior_stage.detailed.implementation_version != IMPLEMENTATION_VERSION
            or prior_stage.detailed.policy_version != INVOCATION_POLICY_VERSION
            or prior_stage.detailed.session_mode != "new_persistent"
            or prior_stage.detailed.session_turn_index != 1
            or input_contents["prior_stage_gate"] != prior_gate.canonical_bytes()
            or bound_gate != prior_gate
            or bound_gate.schema_version != 4
            or bound_gate.stage != "s2_route_optimizer"
            or bound_gate.config != "s1s2"
            or bound_gate.candidate_bank_sha256 != prior_stage.candidate.bank_sha256
            or detailed.session_turn_index != 1
            or detailed.prior_invocation_receipt_file_sha256 is not None
            or detailed.prior_gate_report_file_sha256
            != sha256_bytes(input_contents["prior_stage_gate"])
            or detailed.thread_id == prior_stage.detailed.thread_id
            or parent_bank is None
            or parent_bank.bank_sha256 != selected_parent_sha256
        ):
            raise PortfolioTreatmentFinalizationError(
                "S3 ephemeral lineage differs from S2, its selected gate, or parent"
            )
    elif detailed.session_mode == "resume":
        if source.key != "s3" or prior_stage is None or prior_gate is None:
            raise PortfolioTreatmentFinalizationError(
                "only S3 may resume the immediately prior S2 session"
            )
        prior_detailed = _load_model(
            input_contents["prior_invocation_receipt"],
            EvolutionInvocationReceipt,
            label="S3 bound prior S2 invocation receipt",
        )
        bound_gate = _load_model(
            input_contents["prior_stage_gate"],
            PortfolioStageGateReport,
            label="S3 bound prior S2 gate report",
        )
        selected_parent_sha256 = (
            bound_gate.candidate_bank_sha256
            if bound_gate.decision == "accepted"
            else bound_gate.parent_bank_sha256
        )
        if (
            prior_stage.key != "s2"
            or input_contents["prior_invocation_receipt"]
            != prior_stage.detailed.canonical_bytes()
            or input_contents["prior_stage_gate"] != prior_gate.canonical_bytes()
            or prior_detailed != prior_stage.detailed
            or bound_gate != prior_gate
            or prior_detailed.session_mode != "new_persistent"
            or prior_detailed.session_turn_index != 1
            or detailed.session_turn_index != 2
            or prior_detailed.thread_id != detailed.thread_id
            or detailed.resume_thread_id != prior_detailed.thread_id
            or prior_detailed.session_scratch_path != detailed.session_scratch_path
            or prior_detailed.session_scratch_device != detailed.session_scratch_device
            or prior_detailed.session_scratch_inode != detailed.session_scratch_inode
            or prior_detailed.requested_model != detailed.requested_model
            or prior_detailed.reasoning_effort != detailed.reasoning_effort
            or prior_detailed.policy_version != detailed.policy_version
            or prior_detailed.implementation_id != detailed.implementation_id
            or prior_detailed.implementation_version != detailed.implementation_version
            or prior_detailed.implementation_file_sha256
            != detailed.implementation_file_sha256
            or prior_detailed.codex_executable_sha256
            != detailed.codex_executable_sha256
            or detailed.prior_invocation_receipt_file_sha256
            != sha256_bytes(input_contents["prior_invocation_receipt"])
            or detailed.prior_gate_report_file_sha256
            != sha256_bytes(input_contents["prior_stage_gate"])
            or parent_bank is None
            or parent_bank.bank_sha256 != selected_parent_sha256
        ):
            raise PortfolioTreatmentFinalizationError(
                "S3 resumed-session lineage differs from S2, its gate, or parent"
            )
    elif source.key == "s3":
        raise PortfolioTreatmentFinalizationError(
            "S3 must use the v1.5 gate-bound ephemeral clean turn"
        )

    if source.key == "s1":
        packet = _load_model(
            input_contents["stage_input"],
            PortfolioEvolutionInputPacket,
            label="s1 evolution input packet",
        )
        if packet.manifest.stage != "s1":
            raise PortfolioTreatmentFinalizationError(
                "S1 evolution input packet stage mismatch"
            )
    else:
        packet = _load_model(
            input_contents["stage_input"],
            PortfolioParentAttributionPacket,
            label=f"{source.key} current-parent attribution packet",
        )
        if (
            parent_bank is None
            or packet.stage != expected_stage
            or packet.source_bank_sha256 != parent_bank.bank_sha256
        ):
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} attribution is not bound to its gated parent Bank"
            )
    candidate = _load_model(
        source_files["candidate-bank.json"],
        StaticBankArtifact,
        label=f"{source.key} candidate Bank",
    )
    if (
        candidate.bank_sha256 != detailed.candidate_bank_sha256
        or candidate.tool_registry_runtime_sha256 != tool_registry_runtime_sha256
    ):
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} candidate Bank identity differs from its receipt/runtime"
        )

    mutation: PortfolioStageMutation | None = None
    raw_bytes = source_files["raw-model-output.json"]
    if source.key == "s1":
        normalized = _load_model(
            source_files["normalized-draft.json"],
            AuthoringDraftBundle,
            label="S1 normalized draft",
        )
        semantic_raw = _canonical_object(
            semantic_bytes,
            label="S1 semantic authoring input",
        )
        codex_raw = _canonical_object(codex_bytes, label="S1 Codex authoring input")
        from skillchain.codex_authoring import CodexAuthoringInput
        from skillchain.static_authoring import AuthoringInput

        semantic = AuthoringInput.model_validate(semantic_raw, strict=True)
        codex_input = CodexAuthoringInput.model_validate(codex_raw, strict=True)
        try:
            codex_bound = normalize_codex_authoring_output(
                raw_bytes,
                authoring_input=codex_input,
                semantic_source=semantic,
            )
            replayed_draft = build_authoring_draft_bundle(
                authoring_input_sha256=common_authoring_input_sha256,
                drafts=codex_bound.drafts,
            )
            replayed_bank = compile_portfolio_s1_creator_bank(
                semantic,
                replayed_draft,
                tool_registry_runtime_sha256=tool_registry_runtime_sha256,
            )
        except Exception as error:
            raise PortfolioTreatmentFinalizationError(
                "S1 raw model output cannot replay through the trusted compiler"
            ) from error
        if replayed_draft != normalized or replayed_bank != candidate:
            raise PortfolioTreatmentFinalizationError(
                "S1 normalized draft or candidate differs from replay"
            )
    else:
        if parent_bank is None:  # pragma: no cover - guarded above
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} parent Bank is absent"
            )
        mutation = _load_model(
            source_files["stage-mutation.json"],
            PortfolioStageMutation,
            label=f"{source.key} stage mutation",
        )
        actionable_capability_ids, optimization_scope = _actionable_mutation_scope(
            expected_stage,
            packet,
            parent_bank=parent_bank,
        )
        if source.key == "s3":
            try:
                normalized_patch = parse_portfolio_s3_text_patch_artifact(
                    source_files["s3-text-patch.json"]
                )
            except PortfolioTreatmentError as error:
                raise PortfolioTreatmentFinalizationError(
                    "S3 normalized TextOpt patch is invalid"
                ) from error
            try:
                replayed_patch, replayed_change = _parse_and_compile_s3_text_patch(
                    raw_bytes,
                    parent_bank=parent_bank,
                    actionable_capability_ids=actionable_capability_ids,
                    optimization_scope=optimization_scope,
                    rejected_edit_buffer=rejected_edit_buffer,
                )
            except Exception as error:
                raise PortfolioTreatmentFinalizationError(
                    "S3 raw proposal cannot replay through TextOpt"
                ) from error
            if replayed_patch != normalized_patch:
                raise PortfolioTreatmentFinalizationError(
                    "S3 normalized TextOpt patch differs from raw replay"
                )
            parsed_changes = (replayed_change,)
        else:
            parsed_changes = _parse_mutation_changes(
                raw_bytes,
                stage=expected_stage,
                parent_bank=parent_bank,
                actionable_capability_ids=actionable_capability_ids,
                optimization_scope=optimization_scope,
            )
        expected_input_digest = sha256_bytes(input_contents["stage_input"])
        mutation_inputs = {
            binding.artifact_kind: binding for binding in mutation.input_artifacts
        }
        expected_mutation_input_kinds = {_INPUT_KIND[source.key]}
        if rejected_edit_buffer_bytes is not None:
            expected_mutation_input_kinds.add("rejected_edit_buffer")
        stage_input_binding = mutation_inputs.get(_INPUT_KIND[source.key])
        rejected_buffer_binding = mutation_inputs.get("rejected_edit_buffer")
        mutation_inputs_valid = (
            len(mutation_inputs) == len(mutation.input_artifacts)
            and set(mutation_inputs) == expected_mutation_input_kinds
            and stage_input_binding is not None
            and stage_input_binding.artifact_file_sha256 == expected_input_digest
            and stage_input_binding.artifact_content_sha256 == expected_input_digest
            and (
                rejected_edit_buffer_bytes is None
                or (
                    rejected_buffer_binding is not None
                    and rejected_buffer_binding.artifact_file_sha256
                    == sha256_bytes(rejected_edit_buffer_bytes)
                    and rejected_buffer_binding.artifact_content_sha256
                    == sha256_bytes(rejected_edit_buffer_bytes)
                )
            )
        )
        if (
            tuple(mutation.changes) != parsed_changes
            or mutation.parent_bank_sha256 != parent_bank.bank_sha256
            or mutation.mutation_sha256 != detailed.mutation_sha256
            or mutation.implementation_id != detailed.implementation_id
            or mutation.implementation_version != detailed.implementation_version
            or mutation.implementation_file_sha256
            != detailed.implementation_file_sha256
            or not mutation_inputs_valid
            or apply_portfolio_stage_mutation(parent_bank, mutation) != candidate
        ):
            raise PortfolioTreatmentFinalizationError(
                f"{source.key} raw mutation or candidate differs from replay"
            )
    return _EvolutionSource(
        key=source.key,
        candidate=candidate,
        mutation=mutation,
        invocation=core,
        detailed=detailed,
        stage_input=packet,
        stage_input_bytes=input_contents["stage_input"],
        rejected_edit_buffer_bytes=rejected_edit_buffer_bytes,
        implementation_source_bytes=implementation_source_bytes,
        source_files=source_files,
    )


def _load_gate(source: GateSource) -> _GateEvidence:
    report_bytes = _read_file(
        source.report_path,
        label=f"{source.key} gate report",
        expected_file_sha256=source.expected_report_file_sha256,
    )
    report = _load_model(
        report_bytes,
        PortfolioStageGateReport,
        label=f"{source.key} gate report",
    )
    if report.config != _STAGE_CONFIG[source.key]:
        raise PortfolioTreatmentFinalizationError(f"{source.key} gate config mismatch")
    parent = _read_file(
        source.parent_results_path,
        label=f"{source.key} parent gate results",
        expected_file_sha256=report.parent_result_file_sha256,
    )
    candidate = _read_file(
        source.candidate_results_path,
        label=f"{source.key} candidate gate results",
        expected_file_sha256=report.candidate_result_file_sha256,
    )
    parent_results = _load_model(
        parent,
        PortfolioStageGateResultSet,
        label=f"{source.key} parent gate results",
    )
    candidate_results = _load_model(
        candidate,
        PortfolioStageGateResultSet,
        label=f"{source.key} candidate gate results",
    )
    try:
        verify_portfolio_stage_gate_evidence(
            report,
            parent_results,
            candidate_results,
        )
    except PortfolioTreatmentError as error:
        raise PortfolioTreatmentFinalizationError(
            f"{source.key} gate result projection mismatch"
        ) from error
    return _GateEvidence(
        key=source.key,
        report=report,
        parent_results=parent_results,
        candidate_results=candidate_results,
        report_bytes=report_bytes,
        parent_results_bytes=parent,
        candidate_results_bytes=candidate,
    )


def _derive_split(
    queries: tuple[object, ...],
    *,
    optimization_batch_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(queries) != 200:
        raise PortfolioTreatmentFinalizationError(
            "Portfolio finalization requires exactly 200 verified queries"
        )
    optimization: list[object] = []
    evaluation: list[object] = []
    for query in queries:
        if getattr(query, "synthesis_batch_id", None) == optimization_batch_id:
            optimization.append(query)
        else:
            evaluation.append(query)
    if len(optimization) != 25 or len(evaluation) != 175:
        raise PortfolioTreatmentFinalizationError(
            "Portfolio optimization/evaluation split must be exactly 25/175"
        )

    def values(rows: list[object], field: str) -> tuple[str, ...]:
        observed = tuple(sorted(str(getattr(item, field)) for item in rows))
        if len(observed) != len(set(observed)):
            raise PortfolioTreatmentFinalizationError(
                f"Portfolio split contains duplicate {field}"
            )
        return observed

    opt_ids = values(optimization, "query_id")
    eval_ids = values(evaluation, "query_id")
    opt_groups = tuple(
        sorted({str(getattr(item, "leakage_group_id")) for item in optimization})
    )
    eval_groups = tuple(
        sorted({str(getattr(item, "leakage_group_id")) for item in evaluation})
    )
    if set(opt_ids) & set(eval_ids) or set(opt_groups) & set(eval_groups):
        raise PortfolioTreatmentFinalizationError(
            "Portfolio optimization/evaluation split leaks queries or asset groups"
        )
    return opt_ids, eval_ids, opt_groups, eval_groups


def _artifact_binding(
    kind: str,
    relative: str,
    content: bytes,
) -> PortfolioArtifactBinding:
    digest = sha256_bytes(content)
    return PortfolioArtifactBinding(
        artifact_kind=kind,
        artifact_file=relative,
        artifact_file_sha256=digest,
        artifact_content_sha256=digest,
    )


def _treatment_receipt(
    *,
    config: PortfolioTreatmentConfig,
    common_authoring_input_sha256: str,
    implementation_id: str,
    implementation_version: str,
    implementation_file_sha256: str,
    input_artifacts: tuple[PortfolioArtifactBinding, ...],
    parent: StaticBankArtifact | None,
    mutation: PortfolioStageMutation | None,
    mutation_file: str | None,
    candidate: StaticBankArtifact,
    candidate_file: str,
    output: StaticBankArtifact,
    invocation: PortfolioModelInvocationReceipt,
    invocation_file: str,
    raw_file: str,
    raw_bytes: bytes,
    gate_file: str,
    gate_sha256: str,
    gate_bytes: bytes,
    decision: Literal["accepted", "rolled_back"],
) -> PortfolioTreatmentReceipt:
    stage = {
        "llm_static": "static_author",
        "s1": "s1_creator",
        "s1s2": "s2_route_optimizer",
        "full": "s3_body_refiner",
    }[config]
    generation = {
        "llm_static": "codex_static_author",
        "s1": "codex_s1_creator",
        "s1s2": "llm_route_optimizer",
        "full": "llm_body_refiner",
    }[config]
    edit_scope = {
        "llm_static": "create_bank",
        "s1": "create_bank",
        "s1s2": "description_only",
        "full": "body_only",
    }[config]
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-treatment-receipt",
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "config": config,
        "stage": stage,
        "generation_kind": generation,
        "edit_scope": edit_scope,
        "common_authoring_input_sha256": common_authoring_input_sha256,
        "implementation_id": implementation_id,
        "implementation_version": implementation_version,
        "implementation_file_sha256": implementation_file_sha256,
        "input_artifacts": [item.model_dump(mode="json") for item in input_artifacts],
        "parent_bank_sha256": parent.bank_sha256 if parent else None,
        "mutation_sha256": mutation.mutation_sha256 if mutation else None,
        "mutation_file": mutation_file,
        "mutation_file_sha256": (
            sha256_bytes(mutation.canonical_bytes()) if mutation else None
        ),
        "candidate_bank_sha256": candidate.bank_sha256,
        "candidate_bank_file": candidate_file,
        "candidate_bank_file_sha256": sha256_bytes(candidate.canonical_bytes()),
        "output_bank_sha256": output.bank_sha256,
        "output_bank_file_sha256": sha256_bytes(output.canonical_bytes()),
        "model_call_count": 1,
        "invocation_receipt_file": invocation_file,
        "invocation_receipt_file_sha256": sha256_bytes(invocation.canonical_bytes()),
        "raw_model_output_file": raw_file,
        "raw_model_output_file_sha256": sha256_bytes(raw_bytes),
        "gate_report_file": gate_file,
        "gate_report_sha256": gate_sha256,
        "gate_report_file_sha256": sha256_bytes(gate_bytes),
        "decision": decision,
    }
    try:
        return PortfolioTreatmentReceipt.model_validate(
            {
                **payload,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentFinalizationError(
            f"{config} treatment receipt is invalid"
        ) from error


def _chain_manifest(
    *,
    records: tuple[PortfolioTreatmentReceipt, ...],
    outputs: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
    optimization_query_ids: tuple[str, ...],
    evaluation_query_ids: tuple[str, ...],
    optimization_leakage_group_ids: tuple[str, ...],
    evaluation_leakage_group_ids: tuple[str, ...],
) -> PortfolioTreatmentChainManifest:
    banks = tuple(
        PortfolioBankBinding(
            config=config,
            bank_file=f"bank-{config}.json",
            bank_sha256=outputs[config].bank_sha256,
            bank_file_sha256=sha256_bytes(outputs[config].canonical_bytes()),
        )
        for config in _CONFIG_ORDER
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-treatment-chain-manifest",
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "status": "ready_for_matrix",
        "track": "portfolio",
        "formal_eligible": False,
        "optimization_query_ids": list(optimization_query_ids),
        "evaluation_query_ids": list(evaluation_query_ids),
        "optimization_leakage_group_ids": list(optimization_leakage_group_ids),
        "evaluation_leakage_group_ids": list(evaluation_leakage_group_ids),
        "records": [item.model_dump(mode="json") for item in records],
        "banks": [item.model_dump(mode="json") for item in banks],
    }
    try:
        return PortfolioTreatmentChainManifest.model_validate(
            {
                **payload,
                "chain_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentFinalizationError(
            "treatment chain manifest is invalid"
        ) from error


def _copy_static_files(staging: Path, source: _StaticSource) -> None:
    files = {
        "inputs/static/semantic-authoring-input.json": source.semantic_bytes,
        "inputs/static/codex-authoring-input.json": source.codex_bytes,
        "inputs/static/pre-review-draft.json": source.draft_bytes,
        "inputs/static/human-review.json": source.review_bytes,
        "stages/llm_static/prompt.txt": source.prompt_bytes,
        "stages/llm_static/codex-events.jsonl": source.event_bytes,
        "stages/llm_static/codex-stderr.bin": source.stderr_bytes,
        "stages/llm_static/raw-model-output.json": source.raw_bytes,
        "stages/llm_static/invocation-receipt.json": (source.detailed_receipt_bytes),
        "stages/llm_static/model-invocation-receipt.json": (
            source.invocation.canonical_bytes()
        ),
        "stages/llm_static/candidate-bank.json": source.bank.canonical_bytes(),
    }
    for relative, content in files.items():
        _write_unique(staging, relative, content)


def _copy_evolution_files(
    staging: Path,
    source: _EvolutionSource,
) -> tuple[str, str, str, str | None]:
    config = _STAGE_CONFIG[source.key]
    stage_dir = f"stages/{config}"
    for name, content in source.source_files.items():
        _write_unique(staging, f"{stage_dir}/{name}", content)
    _write_unique(
        staging,
        f"{stage_dir}/implementation-source.py",
        source.implementation_source_bytes,
    )
    stage_input_file = f"inputs/{config}/packet.json"
    _write_unique(staging, stage_input_file, source.stage_input_bytes)
    if source.rejected_edit_buffer_bytes is not None:
        _write_unique(
            staging,
            f"inputs/{config}/rejected-edit-buffer.json",
            source.rejected_edit_buffer_bytes,
        )
    if source.mutation is not None:
        content_by_kind = {
            _INPUT_KIND[source.key]: source.stage_input_bytes,
        }
        if source.rejected_edit_buffer_bytes is not None:
            content_by_kind["rejected_edit_buffer"] = source.rejected_edit_buffer_bytes
        for binding in source.mutation.input_artifacts:
            content = content_by_kind.get(binding.artifact_kind)
            if (
                content is None
                or binding.artifact_file_sha256 != sha256_bytes(content)
                or binding.artifact_content_sha256 != sha256_bytes(content)
            ):
                raise PortfolioTreatmentFinalizationError(
                    f"{source.key} mutation binds another compiler input"
                )
            _write_unique(
                staging,
                f"{stage_dir}/{binding.artifact_file}",
                content,
            )
    mutation_file = (
        f"{stage_dir}/stage-mutation.json" if source.mutation is not None else None
    )
    return (
        f"{stage_dir}/candidate-bank.json",
        f"{stage_dir}/model-invocation-receipt.json",
        f"{stage_dir}/raw-model-output.json",
        mutation_file,
    )


def _copy_gate_files(staging: Path, gate: _GateEvidence) -> str:
    config = _STAGE_CONFIG[gate.key]
    report_file = f"gates/{config}/gate-report.json"
    _write_unique(staging, report_file, gate.report_bytes)
    _write_unique(
        staging,
        gate.report.parent_result_file,
        gate.parent_results_bytes,
    )
    _write_unique(
        staging,
        gate.report.candidate_result_file,
        gate.candidate_results_bytes,
    )
    return report_file


def _runtime_lock(
    *,
    tool: _ToolSource,
    outputs: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
    records: tuple[PortfolioTreatmentReceipt, ...],
    manifest: PortfolioTreatmentChainManifest,
    manifest_file_sha256: str,
    iteration_report_sha256: str,
) -> dict[str, Any]:
    registry = tool.runtime.registry
    execution_artifact_aliases = build_portfolio_execution_artifact_aliases(manifest)
    tool_bindings = [
        {
            "tool_name": spec.name,
            "tool_spec_sha256": spec.spec_sha256,
            "runtime_binding_sha256": registry.runtime_binding_sha256(spec.name),
        }
        for spec in registry.specs()
    ]
    code_files = {
        "config_file_sha256": REPOSITORY_ROOT / "src" / "skillchain" / "config.py",
        "runner_file_sha256": (
            REPOSITORY_ROOT / "src" / "skillchain" / "runners" / "assistant.py"
        ),
        "llm_adapter_file_sha256": REPOSITORY_ROOT / "src" / "skillchain" / "llm.py",
        "evaluator_outputs_file_sha256": (
            REPOSITORY_ROOT
            / "src"
            / "skillchain"
            / "evaluation"
            / "evaluator_outputs.py"
        ),
        "final_runtime_file_sha256": (
            REPOSITORY_ROOT / "src" / "skillchain" / "evaluation" / "final_runtime.py"
        ),
        "packets_file_sha256": (
            REPOSITORY_ROOT / "src" / "skillchain" / "evaluation" / "packets.py"
        ),
        "evaluator_isolation_file_sha256": (
            REPOSITORY_ROOT
            / "src"
            / "skillchain"
            / "evaluation"
            / "evaluator_isolation.py"
        ),
        "portfolio_tool_runtime_file_sha256": (
            REPOSITORY_ROOT / "src" / "skillchain" / "tools" / "portfolio_runtime.py"
        ),
        "tool_registry_file_sha256": (
            REPOSITORY_ROOT / "src" / "skillchain" / "tools" / "registry.py"
        ),
        "shard_runner_file_sha256": (
            REPOSITORY_ROOT / "scripts" / "run_portfolio_shard.py"
        ),
        "portfolio_execution_file_sha256": (
            REPOSITORY_ROOT
            / "src"
            / "skillchain"
            / "evaluation"
            / "portfolio_execution.py"
        ),
        "treatment_finalizer_file_sha256": _FINALIZER_PATH,
    }
    payload = {
        "schema_version": 2,
        "kind": "portfolio-assistant-runtime-lock",
        "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
        "track": "portfolio",
        "formal_eligible": False,
        "formal_ineligible_reason": (
            "public-dataset-metadata-adapters-are-not-formal-tool-runtimes"
        ),
        "stage": "real_treatment_chain_finalization",
        "tool_runtime_source_lock_file_sha256": tool.source_lock_file_sha256,
        "tool_runtime_source_lock_sha256": tool.source_lock_sha256,
        "compatible_frozen_noskill_runtime_lock_sha256": tool.source_lock_sha256,
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        "runtime_data_sha256": tool.runtime.index.runtime_data_sha256,
        "source_sha256s": list(tool.runtime.index.source_sha256s),
        "tool_bindings": tool_bindings,
        "recipe_evidence_rows": tool.recipe_rows,
        "system_prompt_sha256": sha256_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")),
        "iteration_report_sha256": iteration_report_sha256,
        "bank_sha256s": {
            config: outputs[config].bank_sha256 for config in _CONFIG_ORDER
        },
        "bank_source_file_sha256s": {
            config: sha256_bytes(outputs[config].canonical_bytes())
            for config in _CONFIG_ORDER
        },
        "bank_policy": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "official_matrix_eligible": True,
        "treatment_chain_status": "ready_for_matrix",
        "treatment_chain_manifest_file": "treatment-chain-manifest.json",
        "treatment_chain_manifest_file_sha256": manifest_file_sha256,
        "treatment_chain_sha256": manifest.chain_sha256,
        "treatment_record_sha256s": {
            item.config: item.receipt_sha256 for item in records
        },
        "execution_artifact_alias_policy_version": (
            PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
        ),
        "execution_artifact_aliases": [
            item.model_dump(mode="json") for item in execution_artifact_aliases
        ],
        "execution_artifact_alias_provider_model_call_count": 0,
        "shared_stage2_route_policy_version": SHARED_STAGE2_ROUTE_POLICY_VERSION,
        "shared_stage2_route_schema_sha256": sha256_bytes(
            canonical_json_bytes(SharedStage2RouteArtifact.model_json_schema())
        ),
        "portfolio_router_contract_version": PORTFOLIO_ROUTER_CONTRACT_VERSION,
        "portfolio_router_contract_sha256": PORTFOLIO_ROUTER_CONTRACT_SHA256,
        "portfolio_router_request_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        ),
        "portfolio_router_pricing_reservation_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
        ),
        "portfolio_failure_policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
        "portfolio_circuit_breaker_threshold": PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
        "portfolio_max_retryable_attempts_per_query": (
            PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
        ),
        "noskill_execution_policy_version": NOSKILL_EXECUTION_POLICY_VERSION,
        "noskill_execution_contract_sha256": NOSKILL_EXECUTION_CONTRACT_SHA256,
        **{name: _file_sha(path) for name, path in code_files.items()},
        "final_judge_parser_policy_version": (FINAL_JUDGE_PARSER_POLICY_VERSION_V4),
        "final_judge_parser_policy_sha256": (FINAL_JUDGE_PARSER_POLICY_SHA256_V4),
        "final_judge_result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
        "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
        "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
        "final_judge_thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
        "final_judge_transport_policy_version": FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
        "final_judge_transport_policy_sha256": FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
        "final_judge_requested_response_format": "json_object",
        "final_judge_max_billable_input_tokens": (
            FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
        ),
        "final_judge_max_billable_output_tokens": (
            FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        ),
        "final_judge_provider_input_token_reserve": (
            FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
        ),
        "final_judge_provider_output_token_reserve": (
            FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
        ),
        "final_judge_provider_pricing_status": PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
        "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
        "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
        "provider_pricing_contract_version": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
        ),
        "provider_pricing_contract_sha256": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ),
        "card_requirement_guard_policy_version": (
            CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (CARD_REQUIREMENT_GUARD_POLICY_SHA256),
        "model_calls_performed": 0,
        "treatment_model_calls_bound": 4,
    }
    return {
        **payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def finalize_portfolio_treatment_runtime(
    inputs: FinalizationInputs,
    *,
    portfolio_inputs_loader: Callable[[], object] = _load_active_inputs,
) -> FinalizedRuntime:
    """Replay, package, publish, and reload one real treatment runtime."""

    if os.path.lexists(inputs.output_dir):
        raise FileExistsError(f"create-only output already exists: {inputs.output_dir}")
    if not inputs.optimization_batch_id.strip():
        raise PortfolioTreatmentFinalizationError(
            "optimization_batch_id must be non-blank"
        )
    stage_by_key = {item.key: item for item in inputs.stages}
    gate_by_key = {item.key: item for item in inputs.gates}
    if set(stage_by_key) != {"s1", "s2", "s3"} or set(gate_by_key) != {
        "s1",
        "s2",
        "s3",
    }:
        raise PortfolioTreatmentFinalizationError(
            "finalization requires exactly one S1/S2/S3 stage and gate"
        )

    tool = _load_tool_source(inputs)
    static = _load_static_source(
        inputs,
        tool_runtime_sha256=tool.runtime.registry.registry_runtime_sha256,
    )
    gates = {key: _load_gate(gate_by_key[key]) for key in ("s1", "s2", "s3")}
    evolution: dict[str, _EvolutionSource] = {}
    outputs: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {
        "llm_static": static.bank,
    }
    parent = static.bank
    for key in ("s1", "s2", "s3"):
        stage = _load_evolution_source(
            stage_by_key[key],
            parent_bank=None if key == "s1" else parent,
            prior_stage=evolution.get("s2") if key == "s3" else None,
            prior_gate=gates["s2"].report if key == "s3" else None,
            semantic_bytes=static.semantic_bytes,
            codex_bytes=static.codex_bytes,
            common_authoring_input_sha256=(static.common_authoring_input_sha256),
            tool_registry_runtime_sha256=(
                tool.runtime.registry.registry_runtime_sha256
            ),
        )
        gate = gates[key].report
        config = _STAGE_CONFIG[key]
        if (
            gate.parent_bank_sha256 != parent.bank_sha256
            or gate.candidate_bank_sha256 != stage.candidate.bank_sha256
        ):
            raise PortfolioTreatmentFinalizationError(
                f"{key} gate does not bind its exact parent/candidate Banks"
            )
        if key != "s1":
            if not isinstance(
                stage.stage_input,
                PortfolioParentAttributionPacket,
            ):  # pragma: no cover - guarded by the stage loader
                raise PortfolioTreatmentFinalizationError(
                    f"{key} lacks typed current-parent attribution"
                )
            try:
                verify_portfolio_parent_attribution_gate_binding(
                    stage.stage_input,
                    gates[key].parent_results,
                )
            except PortfolioAttributionError as error:
                raise PortfolioTreatmentFinalizationError(
                    f"{key} attribution is not the gate-parent smoke run"
                ) from error
        output = stage.candidate if gate.decision == "accepted" else parent
        outputs[config] = output
        parent = output
        evolution[key] = stage
    violations = evolution_bank_boundary_violations(outputs)
    if violations:
        raise PortfolioTreatmentFinalizationError(
            "final output Bank stage boundary violation: " + "; ".join(violations)
        )

    loaded_inputs = portfolio_inputs_loader()
    queries = getattr(loaded_inputs, "queries", None)
    if not isinstance(queries, tuple):
        raise PortfolioTreatmentFinalizationError(
            "Portfolio input loader returned no frozen query tuple"
        )
    split = _derive_split(
        queries,
        optimization_batch_id=inputs.optimization_batch_id,
    )
    optimization_ids = split[0]
    for key, stage in evolution.items():
        if key == "s1":
            if not isinstance(stage.stage_input, PortfolioEvolutionInputPacket):
                raise PortfolioTreatmentFinalizationError(
                    "S1 lacks its complete optimization input packet"
                )
            if (
                stage.stage_input.manifest.accepted_batch_id
                != inputs.optimization_batch_id
                or stage.stage_input.manifest.query_ids != optimization_ids
            ):
                raise PortfolioTreatmentFinalizationError(
                    "S1 input does not cover the frozen optimization split"
                )
        else:
            if not isinstance(
                stage.stage_input,
                PortfolioParentAttributionPacket,
            ):
                raise PortfolioTreatmentFinalizationError(
                    f"{key} lacks current-parent attribution"
                )
            if (
                stage.stage_input.optimization_query_ids_sha256
                != sha256_bytes(canonical_json_bytes(list(optimization_ids)))
                or not set(stage.stage_input.query_ids).issubset(optimization_ids)
                or stage.stage_input.query_ids != gates[key].report.evaluation_query_ids
            ):
                raise PortfolioTreatmentFinalizationError(
                    f"{key} attribution/gate differs from the optimization split"
                )

    staging: Path | None = None
    try:
        staging = new_staging_directory(inputs.output_dir)
        _write_unique(staging, "recipe-evidence.jsonl", tool.recipe_bytes)
        _write_unique(
            staging,
            "system-prompt.txt",
            PORTFOLIO_SYSTEM_PROMPT.encode("utf-8"),
        )
        _copy_static_files(staging, static)
        gate_files = {
            key: _copy_gate_files(staging, gate) for key, gate in gates.items()
        }
        evolution_files = {
            key: _copy_evolution_files(staging, stage)
            for key, stage in evolution.items()
        }
        for config, bank in outputs.items():
            _write_unique(staging, f"bank-{config}.json", bank.canonical_bytes())

        static_gate_file = "inputs/static/human-review.json"
        static_inputs = (
            _artifact_binding(
                "semantic_authoring_input",
                "inputs/static/semantic-authoring-input.json",
                static.semantic_bytes,
            ),
            _artifact_binding(
                "codex_authoring_input",
                "inputs/static/codex-authoring-input.json",
                static.codex_bytes,
            ),
            _artifact_binding(
                "pre_review_draft",
                "inputs/static/pre-review-draft.json",
                static.draft_bytes,
            ),
            _artifact_binding(
                "human_review",
                static_gate_file,
                static.review_bytes,
            ),
        )
        records: list[PortfolioTreatmentReceipt] = [
            _treatment_receipt(
                config="llm_static",
                common_authoring_input_sha256=(static.common_authoring_input_sha256),
                implementation_id="codex-mediated-static-author-v5",
                implementation_version="5.0.0",
                implementation_file_sha256=(static.implementation_file_sha256),
                input_artifacts=static_inputs,
                parent=None,
                mutation=None,
                mutation_file=None,
                candidate=static.bank,
                candidate_file="stages/llm_static/candidate-bank.json",
                output=static.bank,
                invocation=static.invocation,
                invocation_file=("stages/llm_static/model-invocation-receipt.json"),
                raw_file="stages/llm_static/raw-model-output.json",
                raw_bytes=static.raw_bytes,
                gate_file=static_gate_file,
                gate_sha256=sha256_bytes(static.review_bytes),
                gate_bytes=static.review_bytes,
                decision="accepted",
            )
        ]
        parent = static.bank
        for key in ("s1", "s2", "s3"):
            stage = evolution[key]
            gate = gates[key]
            config = _STAGE_CONFIG[key]
            candidate_file, invocation_file, raw_file, mutation_file = evolution_files[
                key
            ]
            stage_input_file = f"inputs/{config}/packet.json"
            stage_input_binding = _artifact_binding(
                _INPUT_KIND[key],
                stage_input_file,
                stage.stage_input_bytes,
            )
            stage_dir = f"stages/{config}"
            input_artifact_list = [
                stage_input_binding,
                _artifact_binding(
                    "detailed_invocation_receipt",
                    f"{stage_dir}/invocation-receipt.json",
                    stage.detailed.canonical_bytes(),
                ),
            ]
            if key == "s1":
                input_artifact_list.extend(
                    (
                        _artifact_binding(
                            "creator_packet",
                            stage_input_file,
                            stage.stage_input_bytes,
                        ),
                        _artifact_binding(
                            "engineer_review",
                            gate_files[key],
                            gate.report_bytes,
                        ),
                    )
                )
            elif key == "s3":
                input_artifact_list.append(
                    _artifact_binding(
                        "prior_stage_gate",
                        gate_files["s2"],
                        gates["s2"].report_bytes,
                    )
                )
                if stage.rejected_edit_buffer_bytes is not None:
                    input_artifact_list.append(
                        _artifact_binding(
                            "rejected_edit_buffer",
                            f"inputs/{config}/rejected-edit-buffer.json",
                            stage.rejected_edit_buffer_bytes,
                        )
                    )
            input_artifacts = tuple(input_artifact_list)
            output = outputs[config]
            record = _treatment_receipt(
                config=config,
                common_authoring_input_sha256=(static.common_authoring_input_sha256),
                implementation_id=stage.detailed.implementation_id,
                implementation_version=stage.detailed.implementation_version,
                implementation_file_sha256=(stage.detailed.implementation_file_sha256),
                input_artifacts=input_artifacts,
                parent=parent,
                mutation=stage.mutation,
                mutation_file=mutation_file,
                candidate=stage.candidate,
                candidate_file=candidate_file,
                output=output,
                invocation=stage.invocation,
                invocation_file=invocation_file,
                raw_file=raw_file,
                raw_bytes=stage.source_files["raw-model-output.json"],
                gate_file=gate_files[key],
                gate_sha256=gate.report.gate_report_sha256,
                gate_bytes=gate.report_bytes,
                decision=gate.report.decision,
            )
            records.append(record)
            parent = output
        records_tuple = tuple(records)
        if len(records_tuple) != 4:  # pragma: no cover - fixed loop guard
            raise PortfolioTreatmentFinalizationError(
                "treatment record count is not four"
            )
        typed_records = (
            records_tuple[0],
            records_tuple[1],
            records_tuple[2],
            records_tuple[3],
        )
        for record in typed_records:
            _write_unique(
                staging,
                f"treatments/{record.config}-receipt.json",
                record.canonical_bytes(),
            )
        manifest = _chain_manifest(
            records=typed_records,
            outputs=outputs,
            optimization_query_ids=split[0],
            evaluation_query_ids=split[1],
            optimization_leakage_group_ids=split[2],
            evaluation_leakage_group_ids=split[3],
        )
        manifest_bytes = manifest.canonical_bytes()
        _write_unique(
            staging,
            "treatment-chain-manifest.json",
            manifest_bytes,
        )
        iteration_payload = {
            "schema_version": 1,
            "kind": "portfolio-real-treatment-iteration-report",
            "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
            "status": "ready_for_matrix",
            "track": "portfolio",
            "formal_eligible": False,
            "optimization_query_count": len(split[0]),
            "evaluation_query_count": len(split[1]),
            "treatment_chain_sha256": manifest.chain_sha256,
            "stage_decisions": {
                key: gates[key].report.decision for key in ("s1", "s2", "s3")
            },
            "model_calls_bound": {
                "llm_static": 1,
                "s1_creator": 1,
                "s2_route_optimizer": 1,
                "s3_body_refiner": 1,
                "total": 4,
            },
            "finalizer_model_calls_performed": 0,
        }
        iteration_report = {
            **iteration_payload,
            "report_sha256": sha256_bytes(canonical_json_bytes(iteration_payload)),
        }
        _write_unique(
            staging,
            "iteration-report.json",
            canonical_json_bytes(iteration_report),
        )
        lock = _runtime_lock(
            tool=tool,
            outputs=outputs,
            records=typed_records,
            manifest=manifest,
            manifest_file_sha256=sha256_bytes(manifest_bytes),
            iteration_report_sha256=iteration_report["report_sha256"],
        )
        lock_bytes = canonical_json_bytes(lock)
        _write_unique(staging, "runtime-lock.json", lock_bytes)
        lock_file_sha256 = sha256_bytes(lock_bytes)

        # The disk loader must be able to replay the staged bytes before they
        # become the only published copy.
        load_verified_portfolio_treatment_runtime(
            staging,
            expected_runtime_lock_file_sha256=lock_file_sha256,
        )
        summary_payload = {
            "schema_version": 1,
            "kind": "portfolio-real-treatment-runtime-summary",
            "output_dir": str(inputs.output_dir.absolute()),
            "formal_eligible": False,
            "official_matrix_eligible": True,
            "model_calls_performed": 0,
            "treatment_model_calls_bound": 4,
            "runtime_lock_file_sha256": lock_file_sha256,
            "runtime_lock_sha256": lock["runtime_lock_sha256"],
            "treatment_chain_manifest_file_sha256": sha256_bytes(manifest_bytes),
            "treatment_chain_sha256": manifest.chain_sha256,
            "bank_sha256s": lock["bank_sha256s"],
        }
        summary = {
            **summary_payload,
            "summary_sha256": sha256_bytes(canonical_json_bytes(summary_payload)),
        }
        _write_unique(staging, "summary.json", canonical_json_bytes(summary))
        atomic_publish_new_directory(staging, inputs.output_dir)
        staging = None
        verified = load_verified_portfolio_treatment_runtime(
            inputs.output_dir,
            expected_runtime_lock_file_sha256=lock_file_sha256,
        )
        return FinalizedRuntime(
            root=inputs.output_dir.absolute(),
            runtime_lock_file_sha256=lock_file_sha256,
            runtime_lock_sha256=lock["runtime_lock_sha256"],
            treatment_chain_manifest_file_sha256=sha256_bytes(manifest_bytes),
            treatment_chain_sha256=manifest.chain_sha256,
            verified=verified,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tool-runtime-source-root", type=Path, required=True)
    parser.add_argument(
        "--tool-runtime-source-lock-file-sha256",
        required=True,
    )
    parser.add_argument("--semantic-authoring-input", type=Path, required=True)
    parser.add_argument(
        "--semantic-authoring-input-file-sha256",
        required=True,
    )
    parser.add_argument("--codex-authoring-input", type=Path, required=True)
    parser.add_argument("--codex-authoring-input-file-sha256", required=True)
    parser.add_argument("--static-authoring-root", type=Path, required=True)
    parser.add_argument(
        "--static-invocation-receipt-file-sha256",
        required=True,
    )
    parser.add_argument("--static-human-review", type=Path, required=True)
    parser.add_argument("--static-human-review-file-sha256", required=True)
    for key in ("s1", "s2", "s3"):
        parser.add_argument(f"--{key}-stage-root", type=Path, required=True)
        parser.add_argument(
            f"--{key}-invocation-receipt-file-sha256",
            required=True,
        )
        parser.add_argument(
            f"--{key}-implementation-source",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{key}-implementation-source-file-sha256",
            required=True,
        )
        parser.add_argument(f"--{key}-gate-report", type=Path, required=True)
        parser.add_argument(
            f"--{key}-gate-report-file-sha256",
            required=True,
        )
        parser.add_argument(
            f"--{key}-parent-gate-results",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{key}-candidate-gate-results",
            type=Path,
            required=True,
        )
    parser.add_argument(
        "--optimization-batch-id",
        default="dev-mini-001-r3",
    )
    return parser


def _inputs_from_args(args: argparse.Namespace) -> FinalizationInputs:
    stages = tuple(
        StageSource(
            key=key,
            root=getattr(args, f"{key}_stage_root"),
            expected_invocation_receipt_file_sha256=getattr(
                args,
                f"{key}_invocation_receipt_file_sha256",
            ),
            implementation_source_path=getattr(
                args,
                f"{key}_implementation_source",
            ),
            expected_implementation_source_file_sha256=getattr(
                args,
                f"{key}_implementation_source_file_sha256",
            ),
        )
        for key in ("s1", "s2", "s3")
    )
    gates = tuple(
        GateSource(
            key=key,
            report_path=getattr(args, f"{key}_gate_report"),
            expected_report_file_sha256=getattr(
                args,
                f"{key}_gate_report_file_sha256",
            ),
            parent_results_path=getattr(
                args,
                f"{key}_parent_gate_results",
            ),
            candidate_results_path=getattr(
                args,
                f"{key}_candidate_gate_results",
            ),
        )
        for key in ("s1", "s2", "s3")
    )
    return FinalizationInputs(
        output_dir=args.output_dir,
        tool_runtime_source_root=args.tool_runtime_source_root,
        expected_tool_runtime_source_lock_file_sha256=(
            args.tool_runtime_source_lock_file_sha256
        ),
        semantic_authoring_input_path=args.semantic_authoring_input,
        expected_semantic_authoring_input_file_sha256=(
            args.semantic_authoring_input_file_sha256
        ),
        codex_authoring_input_path=args.codex_authoring_input,
        expected_codex_authoring_input_file_sha256=(
            args.codex_authoring_input_file_sha256
        ),
        static_authoring_root=args.static_authoring_root,
        expected_static_invocation_receipt_file_sha256=(
            args.static_invocation_receipt_file_sha256
        ),
        static_human_review_path=args.static_human_review,
        expected_static_human_review_file_sha256=(args.static_human_review_file_sha256),
        stages=stages,  # type: ignore[arg-type]
        gates=gates,  # type: ignore[arg-type]
        optimization_batch_id=args.optimization_batch_id,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = finalize_portfolio_treatment_runtime(_inputs_from_args(args))
    except (
        ArtifactFormatError,
        FileExistsError,
        OSError,
        PortfolioTreatmentError,
        PortfolioTreatmentFinalizationError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"finalize-portfolio-treatment-runtime: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model_calls_performed": 0,
                "official_matrix_eligible": True,
                "output_dir": str(result.root),
                "runtime_lock_file_sha256": (result.runtime_lock_file_sha256),
                "runtime_lock_sha256": result.runtime_lock_sha256,
                "treatment_chain_manifest_file_sha256": (
                    result.treatment_chain_manifest_file_sha256
                ),
                "treatment_chain_sha256": result.treatment_chain_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
