"""Create the zero-call execution control package for the Portfolio matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import secrets
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_portfolio_assistant_smoke import _hash, _model  # noqa: E402
from scripts.run_portfolio_shard import _assistant_row  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
)
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
    load_final_judge_evaluation_result,
)
from skillchain.evaluation.packets import RubricSnapshot  # noqa: E402
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    load_portfolio_launch_package,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_CNY_QUANTUM,
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
    STATIC_OPT_BANK_FILE,
    STATIC_OPT_REFRESH_RECEIPT_FILE,
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
)
from skillchain.runners.assistant import (  # noqa: E402
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    noskill_execution_contract_payload,
)
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_json,
    read_stable_regular_file,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SKILLED_CONFIGS = {"llm_static", "s1", "s1s2", "full"}
_ALLOWED_PARTIAL_LAUNCH_BLOCKERS = ("dashscope_budget_approval_required",)
_ACTIVE_FINAL_RESULT_LOCK = {
    "final_judge_result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
    "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
    "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
    "final_judge_thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
    "final_judge_max_billable_input_tokens": (FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS),
    "final_judge_max_billable_output_tokens": (FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS),
    "final_judge_provider_input_token_reserve": (
        FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
    ),
    "final_judge_provider_output_token_reserve": (
        FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
    ),
    "final_judge_transport_policy_version": FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
    "final_judge_transport_policy_sha256": FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    "final_judge_requested_response_format": "json_object",
    "final_judge_provider_pricing_status": PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    "card_requirement_guard_policy_version": CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    "card_requirement_guard_policy_sha256": CARD_REQUIREMENT_GUARD_POLICY_SHA256,
}
_ACTIVE_BUDGET_CONTRACT = {
    "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
    "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
    "provider_pricing_contract_version": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION),
    "provider_pricing_contract_sha256": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256),
}
_ACTIVE_EVALUATOR_SOURCE_LOCK = {
    "config_file_sha256": sha256_bytes(
        (SOURCE_ROOT / "skillchain" / "config.py").read_bytes()
    ),
    "packets_file_sha256": sha256_bytes(
        (SOURCE_ROOT / "skillchain" / "evaluation" / "packets.py").read_bytes()
    ),
    "evaluator_isolation_file_sha256": sha256_bytes(
        (
            SOURCE_ROOT / "skillchain" / "evaluation" / "evaluator_isolation.py"
        ).read_bytes()
    ),
    "portfolio_tool_runtime_file_sha256": sha256_bytes(
        (SOURCE_ROOT / "skillchain" / "tools" / "portfolio_runtime.py").read_bytes()
    ),
    "tool_registry_file_sha256": sha256_bytes(
        (SOURCE_ROOT / "skillchain" / "tools" / "registry.py").read_bytes()
    ),
}


def _final_judge_comparison_identity(final: object) -> dict[str, object]:
    """Project the complete Judge request/billing identity used for comparison."""

    identity = {
        "provider": getattr(final, "provider"),
        "model": getattr(final, "model"),
        "endpoint": getattr(final, "endpoint"),
        "max_tokens": getattr(final, "max_tokens"),
        "max_attempts": getattr(final, "max_attempts"),
        "retry_policy_version": getattr(final, "retry_policy_version"),
        "retry_policy_sha256": getattr(final, "retry_policy_sha256"),
        "thinking_budget": getattr(final, "thinking_budget"),
        "max_billable_input_tokens": getattr(final, "max_billable_input_tokens"),
        "max_billable_output_tokens": getattr(final, "max_billable_output_tokens"),
    }
    if getattr(final, "schema_version", 0) >= 10:
        identity.update(
            {
                "transport_policy_version": getattr(final, "transport_policy_version"),
                "transport_policy_sha256": getattr(final, "transport_policy_sha256"),
                "requested_response_format": getattr(
                    final, "requested_response_format"
                ),
            }
        )
    return identity


def _require_active_final_result_runtime(
    runtime_lock: dict,
    *,
    label: str,
) -> None:
    if any(
        runtime_lock.get(field) != expected
        for field, expected in _ACTIVE_FINAL_RESULT_LOCK.items()
    ):
        raise ValueError(
            f"{label} does not bind the active final-Judge schema/cache and "
            "card-requirement guard"
        )


def _require_active_budget_runtime(runtime_lock: dict, *, label: str) -> None:
    if any(
        runtime_lock.get(field) != expected
        for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
    ):
        raise ValueError(f"{label} does not bind the active hard-budget contract")


def _cny_argument(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
        normalized = parsed.quantize(PORTFOLIO_BUDGET_CNY_QUANTUM)
    except (InvalidOperation, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "CNY value must be a finite decimal"
        ) from error
    if not parsed.is_finite() or parsed != normalized or parsed < 0:
        raise argparse.ArgumentTypeError(
            "CNY value must be nonnegative with at most 12 decimal places"
        )
    return normalized


def _cny_text(value: Decimal) -> str:
    return format(value, ".12f")


_HARD_LEDGER_COST_FIELDS = (
    "budget_ledger_settled_actual_cost_cny",
    "budget_ledger_unresolved_reserved_cost_cny",
    "budget_ledger_accountable_cost_cny",
)
_HARD_LEDGER_FORFEITED_COST_FIELD = "budget_ledger_forfeited_reserved_cost_cny"
_HARD_LEDGER_IDENTITY_FIELDS = (
    "budget_authority_sha256",
    "budget_ledger_last_event_index",
    "budget_ledger_last_event_sha256",
)


def _persisted_ledger_cny(payload: dict, field: str, *, label: str) -> Decimal:
    raw = payload.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{label} {field} must be a 12-place CNY string")
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"{label} {field} is not decimal CNY") from error
    if not value.is_finite() or value < 0 or raw != _cny_text(value):
        raise ValueError(f"{label} {field} is not canonical CNY")
    return value


def _numeric_cny(payload: dict, field: str, *, label: str) -> Decimal:
    raw = payload.get(field)
    if type(raw) not in {int, float}:
        raise ValueError(f"{label} {field} must be numeric CNY")
    try:
        value = Decimal(str(raw)).quantize(PORTFOLIO_BUDGET_CNY_QUANTUM)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} {field} is not decimal CNY") from error
    if not value.is_finite() or value < 0:
        raise ValueError(f"{label} {field} is not non-negative CNY")
    return value


def _validate_source_cost_accounting(
    *,
    audit: dict,
    summary: dict,
    checkpoint_local_cost_cny: Decimal,
) -> Decimal:
    """Return the source execution-root settled cost after scope-aware checks."""

    hard_fields = (
        _HARD_LEDGER_COST_FIELDS
        + (_HARD_LEDGER_FORFEITED_COST_FIELD,)
        + _HARD_LEDGER_IDENTITY_FIELDS
    )
    has_hard_ledger = any(field in audit or field in summary for field in hard_fields)
    if not has_hard_ledger:
        local_float = float(checkpoint_local_cost_cny)
        if (
            abs(float(audit.get("dashscope_observed_cost_cny", -1.0)) - local_float)
            > 1e-9
            or abs(
                float(summary.get("observed_dashscope_cost_cny", -1.0)) - local_float
            )
            > 1e-9
        ):
            raise ValueError("source shard observed cost differs from checkpoints")
        return checkpoint_local_cost_cny

    required_fields = _HARD_LEDGER_COST_FIELDS + _HARD_LEDGER_IDENTITY_FIELDS
    missing = [f"audit.{field}" for field in required_fields if field not in audit] + [
        f"summary.{field}" for field in required_fields if field not in summary
    ]
    forfeit_field_present = (
        _HARD_LEDGER_FORFEITED_COST_FIELD in audit
        or _HARD_LEDGER_FORFEITED_COST_FIELD in summary
    )
    if forfeit_field_present:
        missing.extend(
            f"{label}.{_HARD_LEDGER_FORFEITED_COST_FIELD}"
            for label, payload in (("audit", audit), ("summary", summary))
            if _HARD_LEDGER_FORFEITED_COST_FIELD not in payload
        )
    if missing:
        raise ValueError(
            "source hard-ledger checkpoints are incomplete: " + ",".join(missing)
        )
    for field in _HARD_LEDGER_IDENTITY_FIELDS:
        if audit[field] != summary[field]:
            raise ValueError(
                f"source hard-ledger audit/summary identity differs: {field}"
            )
    authority_sha256 = audit["budget_authority_sha256"]
    last_event_index = audit["budget_ledger_last_event_index"]
    last_event_sha256 = audit["budget_ledger_last_event_sha256"]
    if (
        not isinstance(authority_sha256, str)
        or _SHA256.fullmatch(authority_sha256) is None
        or type(last_event_index) is not int
        or last_event_index < 0
        or (last_event_index == 0 and last_event_sha256 is not None)
        or (
            last_event_index > 0
            and (
                not isinstance(last_event_sha256, str)
                or _SHA256.fullmatch(last_event_sha256) is None
            )
        )
    ):
        raise ValueError("source hard-ledger identity is invalid")

    audit_costs = {
        field: _persisted_ledger_cny(audit, field, label="source audit")
        for field in _HARD_LEDGER_COST_FIELDS
    }
    summary_costs = {
        field: _persisted_ledger_cny(summary, field, label="source summary")
        for field in _HARD_LEDGER_COST_FIELDS
    }
    if audit_costs != summary_costs:
        raise ValueError("source hard-ledger audit/summary costs differ")
    forfeited = Decimal("0")
    if forfeit_field_present:
        audit_forfeited = _persisted_ledger_cny(
            audit,
            _HARD_LEDGER_FORFEITED_COST_FIELD,
            label="source audit",
        )
        summary_forfeited = _persisted_ledger_cny(
            summary,
            _HARD_LEDGER_FORFEITED_COST_FIELD,
            label="source summary",
        )
        if audit_forfeited != summary_forfeited:
            raise ValueError("source hard-ledger audit/summary costs differ")
        forfeited = audit_forfeited
    settled = audit_costs["budget_ledger_settled_actual_cost_cny"]
    unresolved = audit_costs["budget_ledger_unresolved_reserved_cost_cny"]
    accountable = audit_costs["budget_ledger_accountable_cost_cny"]
    audit_prior = _numeric_cny(
        audit,
        "prior_dashscope_observed_cost_cny",
        label="source audit",
    )
    summary_prior = _numeric_cny(
        summary,
        "prior_dashscope_observed_cost_cny",
        label="source summary",
    )
    if audit_prior != summary_prior:
        raise ValueError("source hard-ledger audit/summary prior cost differs")
    if accountable != audit_prior + settled + forfeited + unresolved:
        raise ValueError("source hard-ledger accountable cost is inconsistent")

    for payload, field, label in (
        (audit, "dashscope_observed_cost_cny", "source audit"),
        (summary, "observed_dashscope_cost_cny", "source summary"),
    ):
        if _numeric_cny(payload, field, label=label) != settled:
            raise ValueError(
                "source execution-root observed cost differs from ledger settled cost"
            )
    cumulative = audit_prior + settled
    for payload, field, label in (
        (audit, "cumulative_dashscope_observed_cost_cny", "source audit"),
        (summary, "cumulative_dashscope_cost_cny", "source summary"),
    ):
        if _numeric_cny(payload, field, label=label) != cumulative:
            raise ValueError(
                "source phase-cumulative cost differs from prior plus ledger settled"
            )
    if checkpoint_local_cost_cny > settled:
        raise ValueError(
            "source checkpoint-local cost exceeds execution-root ledger settled cost"
        )
    if (
        "shard_local_observed_cost_cny" in audit
        and _numeric_cny(
            audit,
            "shard_local_observed_cost_cny",
            label="source audit",
        )
        != checkpoint_local_cost_cny
    ):
        raise ValueError(
            "source audit shard-local cost differs from imported checkpoints"
        )
    return settled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-scope",
        choices=(
            "partial_shard_repair",
            "core_canary",
            "full_matrix",
            "static_opt_rollout",
        ),
        default="partial_shard_repair",
        help=(
            "Prepare either the historical four-skilled-shard repair flow or "
            "the first split-frozen 25-query Core batch across all five logical "
            "configs, or a fresh complete launch matrix."
        ),
    )
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--launch-plan-file-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-lock-file-sha256", required=True)
    parser.add_argument("--rubric", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--approved-dashscope-budget-cny", type=_cny_argument, required=True
    )
    parser.add_argument(
        "--phase-cumulative-cap-cny",
        type=_cny_argument,
        default=None,
        help=(
            "Cumulative DashScope ceiling for this partial repair phase. It "
            "must be above carried prior cost and no higher than the launch's "
            "operator-approved total cap."
        ),
    )
    parser.add_argument(
        "--authorized-shard-id",
        action="append",
        default=[],
        help="One target skilled shard authorized in this execution root.",
    )
    parser.add_argument("--source-execution-root", type=Path)
    parser.add_argument("--source-shard-id")
    parser.add_argument(
        "--expected-source-shard-audit-sha256",
        default=None,
        help="Expected self SHA-256 from the frozen source shard audit.",
    )
    parser.add_argument(
        "--prior-dashscope-observed-cost-cny",
        type=_cny_argument,
        default=Decimal("0.000000000000"),
        help="Already-spent cost carried into this fresh execution root.",
    )
    parser.add_argument(
        "--legacy-launch-compat",
        action="store_true",
        help=(
            "Use the legacy public query projection when rebinding an existing "
            "launch plan whose frozen query hashes predate the current projector."
        ),
    )
    return parser


def _canonical_object(path: Path, *, label: str) -> tuple[bytes, dict]:
    content = read_stable_regular_file(path, label=label, max_bytes=64 * 1024 * 1024)
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return content, value


def _require_self_hash(value: dict, field: str, *, label: str) -> str:
    supplied = value.get(field)
    if not isinstance(supplied, str) or not _SHA256.fullmatch(supplied):
        raise ValueError(f"{label} lacks a valid {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if supplied != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{label} self hash mismatch")
    return supplied


def _validate_fixed_zero(path: Path, *, query_id: str, error_code: str) -> None:
    _, value = _canonical_object(path, label=f"fixed-zero result {query_id}")
    supplied = _require_self_hash(
        value, "result_sha256", label=f"fixed-zero result {query_id}"
    )
    del supplied
    if (
        value.get("kind") != "portfolio-final-fixed-zero"
        or value.get("query_id") != query_id
        or value.get("assistant_error_code") != error_code
        or value.get("j_project") != 0.0
    ):
        raise ValueError(
            f"fixed-zero result differs from Assistant failure: {query_id}"
        )


def _artifact_set(
    shard_root: Path, query_ids: tuple[str, ...]
) -> tuple[dict[str, str], str]:
    relative_paths = ["shard-audit.json", "shard-summary.json"]
    relative_paths.extend(f"assistant/{query_id}.json" for query_id in query_ids)
    relative_paths.extend(f"final/{query_id}.json" for query_id in query_ids)
    if len(relative_paths) != 52 or len(set(relative_paths)) != 52:
        raise ValueError("frozen NoSkill artifact layout is not exactly 52 files")
    actual = {
        path.relative_to(shard_root).as_posix()
        for path in shard_root.rglob("*")
        if path.is_file()
    }
    if actual != set(relative_paths):
        raise ValueError("frozen NoSkill shard contains missing or extra files")
    file_sha256s = {
        relative: sha256_bytes(
            read_stable_regular_file(
                shard_root / relative,
                label=f"frozen NoSkill artifact {relative}",
                max_bytes=64 * 1024 * 1024,
            )
        )
        for relative in sorted(relative_paths)
    }
    return file_sha256s, sha256_bytes(canonical_json_bytes(file_sha256s))


def _validate_source_checkpoints(
    *,
    source_root: Path,
    source_shard,
    members: tuple,
    source_runtime_lock: dict,
    summary: dict,
    audit: dict,
) -> tuple[dict, Decimal, Decimal]:
    _require_active_final_result_runtime(
        source_runtime_lock,
        label="source runtime lock",
    )
    assistant_error_ids: list[str] = []
    assistant_input_tokens = 0
    assistant_output_tokens = 0
    assistant_model_calls = 0
    tool_call_count = 0
    fixed_zero_count = 0
    judge_input_tokens = 0
    judge_output_tokens = 0
    judge_model_calls = 0
    judge_captured_response_count = 0
    judge_retried_row_count = 0
    judge_initial_empty_response_count = 0
    judge_initial_invalid_json_response_count = 0
    judge_reasoning_present_response_count = 0
    judge_reasoning_tokens_reported_total = 0
    judge_reasoning_tokens_unavailable_response_count = 0
    judge_reasoning_bytes_total = 0
    judge_response_receipts: list[dict[str, object]] = []
    judge_status_counts: Counter[str] = Counter()
    judge_shape_counts: Counter[str] = Counter()
    judge_card_guard_adjusted_count = 0
    judged_scores: list[float] = []
    scored_scores: list[float] = []
    all_scores: list[float] = []
    assistant_contract: dict | None = None
    final_contract: dict | None = None

    for member in members:
        assistant_path = source_root / member.assistant_output_relpath
        _, assistant_raw = _canonical_object(
            assistant_path,
            label=f"source Assistant checkpoint {member.query_id}",
        )
        response, receipt = _assistant_row(assistant_path)
        request = AssistantRequestSnapshot.model_validate_json(
            canonical_json_bytes(assistant_raw.get("request")),
            strict=True,
        )
        if (
            assistant_raw.get("instance_sha256") != member.instance_sha256
            or assistant_raw.get("query_ordinal") != member.query_ordinal
            or request.query.query_id != member.query_id
            or request.config != "noskill"
            or receipt.request_sha256 != request.request_sha256
            or receipt.response_sha256 != _hash(_model(response))
            or receipt.aggregate_usage != response.usage
            or receipt.tool_trace != response.tool_trace
            or request.treatment.bank_sha256 is not None
            or request.treatment.router_stage != "disabled"
            or request.treatment.body_stage != "disabled"
            or response.selected_capability is not None
            or response.skill_slug is not None
            or response.route_trace_sha256 is not None
        ):
            raise ValueError(
                f"source Assistant checkpoint differs from launch: {member.query_id}"
            )
        current_assistant_contract = {
            "backbone": request.backbone.model_dump(mode="json"),
            "budget": request.budget.model_dump(mode="json"),
            "registry": request.registry.model_dump(mode="json"),
        }
        if assistant_contract is None:
            assistant_contract = current_assistant_contract
        elif assistant_contract != current_assistant_contract:
            raise ValueError("source NoSkill Assistant request contract drifted")
        assistant_input_tokens += response.usage.input_tokens
        assistant_output_tokens += response.usage.output_tokens
        assistant_model_calls += len(receipt.model_calls)
        tool_call_count += len(response.tool_trace)

        final_path = source_root / member.final_output_relpath
        if response.error_code is not None:
            assistant_error_ids.append(member.query_id)
            fixed_zero_count += 1
            _validate_fixed_zero(
                final_path,
                query_id=member.query_id,
                error_code=response.error_code,
            )
            all_scores.append(0.0)
            continue

        final = load_final_judge_evaluation_result(final_path)
        if (
            final.schema_version
            != source_runtime_lock["final_judge_result_schema_version"]
            or final.cache_namespace
            != source_runtime_lock["final_judge_cache_namespace"]
            or final.parser_policy_version
            != source_runtime_lock.get("final_judge_parser_policy_version")
            or final.parser_policy_sha256
            != source_runtime_lock.get("final_judge_parser_policy_sha256")
            or final.card_requirement_guard_policy_version
            != source_runtime_lock["card_requirement_guard_policy_version"]
            or final.card_requirement_guard_policy_sha256
            != source_runtime_lock["card_requirement_guard_policy_sha256"]
            or final.visible_card_count != len(response.visible_cards)
            or final.max_attempts != source_runtime_lock["final_judge_max_attempts"]
            or final.retry_policy_version
            != source_runtime_lock["final_judge_retry_policy_version"]
            or final.retry_policy_sha256
            != source_runtime_lock["final_judge_retry_policy_sha256"]
            or final.thinking_budget
            != source_runtime_lock["final_judge_thinking_budget"]
            or final.max_billable_input_tokens
            != source_runtime_lock["final_judge_max_billable_input_tokens"]
            or final.max_billable_output_tokens
            != source_runtime_lock["final_judge_max_billable_output_tokens"]
        ):
            raise ValueError(
                "source final checkpoint parser/guard differs from runtime: "
                f"{member.query_id}"
            )
        if final.card_requirement_guard_adjusted:
            judge_card_guard_adjusted_count += 1
        current_final_contract = _final_judge_comparison_identity(final)
        if final_contract is None:
            final_contract = current_final_contract
        elif final_contract != current_final_contract:
            raise ValueError("source NoSkill final-Judge request contract drifted")
        judge_status_counts[final.outcome.status] += 1
        if final.raw_dimensions_shape is not None:
            judge_shape_counts[final.raw_dimensions_shape] += 1
        aggregate_usage = final.aggregate_usage
        judge_input_tokens += aggregate_usage.input_tokens
        judge_output_tokens += aggregate_usage.output_tokens
        judge_model_calls += final.attempts
        judge_captured_response_count += final.captured_response_count
        if final.attempts == 2:
            judge_retried_row_count += 1
        initial = final.initial_empty_response
        if initial is not None:
            if initial.retry_reason == "empty_final_response":
                judge_initial_empty_response_count += 1
            else:
                judge_initial_invalid_json_response_count += 1
        for reasoning_present, reasoning_tokens, reasoning_bytes in tuple(
            item
            for item in (
                (
                    initial.reasoning_present,
                    initial.reasoning_tokens,
                    initial.reasoning_bytes,
                )
                if initial is not None
                else None,
                (
                    final.reasoning_present,
                    final.reasoning_tokens,
                    final.reasoning_bytes,
                )
                if final.request_id is not None
                else None,
            )
            if item is not None
        ):
            if reasoning_present:
                judge_reasoning_present_response_count += 1
            if reasoning_tokens is None:
                judge_reasoning_tokens_unavailable_response_count += 1
            else:
                judge_reasoning_tokens_reported_total += reasoning_tokens
            judge_reasoning_bytes_total += reasoning_bytes
        judge_response_receipts.append(
            {
                "query_id": member.query_id,
                "attempts": final.attempts,
                "initial_empty_response": (
                    initial is not None
                    and initial.retry_reason == "empty_final_response"
                ),
                "initial_retry_reason": (
                    None if initial is None else initial.retry_reason
                ),
                "initial_reasoning_present": (
                    None if initial is None else initial.reasoning_present
                ),
                "initial_reasoning_tokens": (
                    None if initial is None else initial.reasoning_tokens
                ),
                "initial_reasoning_bytes": (
                    None if initial is None else initial.reasoning_bytes
                ),
                "initial_reasoning_sha256": (
                    None if initial is None else initial.reasoning_sha256
                ),
                "terminal_response_captured": final.request_id is not None,
                "terminal_reasoning_present": final.reasoning_present,
                "terminal_reasoning_tokens": final.reasoning_tokens,
                "terminal_reasoning_bytes": final.reasoning_bytes,
                "terminal_reasoning_sha256": final.reasoning_sha256,
            }
        )
        score = final.outcome.scores.j_project
        judged_scores.append(score)
        if final.outcome.status == "scored":
            scored_scores.append(score)
        all_scores.append(score)

    checkpoint_local_cost = (
        Decimal(assistant_input_tokens) * Decimal("0.15")
        + Decimal(assistant_output_tokens) * Decimal("1.5")
        + Decimal(judge_input_tokens) * Decimal("6.5")
        + Decimal(judge_output_tokens) * Decimal("27.0")
    ) / Decimal(1_000_000)
    checkpoint_local_cost = checkpoint_local_cost.quantize(PORTFOLIO_BUDGET_CNY_QUANTUM)
    expected = {
        "assistant_error_count": len(assistant_error_ids),
        "assistant_error_ids": assistant_error_ids,
        "assistant_input_tokens": assistant_input_tokens,
        "assistant_model_calls": assistant_model_calls,
        "assistant_output_tokens": assistant_output_tokens,
        "assistant_success_count": len(members) - len(assistant_error_ids),
        "fixed_zero_count": fixed_zero_count,
        "judge_input_tokens": judge_input_tokens,
        "judge_invoked_count": len(judged_scores),
        "judge_model_calls": judge_model_calls,
        "judge_output_tokens": judge_output_tokens,
        "judge_captured_response_count": judge_captured_response_count,
        "judge_retried_row_count": judge_retried_row_count,
        "judge_initial_empty_response_count": judge_initial_empty_response_count,
        "judge_initial_invalid_json_response_count": (
            judge_initial_invalid_json_response_count
        ),
        "judge_reasoning_present_response_count": (
            judge_reasoning_present_response_count
        ),
        "judge_reasoning_tokens_reported_total": (
            judge_reasoning_tokens_reported_total
        ),
        "judge_reasoning_tokens_unavailable_response_count": (
            judge_reasoning_tokens_unavailable_response_count
        ),
        "judge_reasoning_bytes_total": judge_reasoning_bytes_total,
        "judge_response_receipts": judge_response_receipts,
        "judge_raw_dimensions_shape_counts": dict(sorted(judge_shape_counts.items())),
        "judge_card_requirement_guard_adjusted_count": (
            judge_card_guard_adjusted_count
        ),
        "judge_status_counts": dict(sorted(judge_status_counts.items())),
        "max_j_project": max(all_scores) if all_scores else 0.0,
        "mean_j_project_all_rows": sum(all_scores) / len(all_scores),
        "mean_j_project_judged_rows": sum(judged_scores) / len(judged_scores),
        "mean_j_project_scored_rows": sum(scored_scores) / len(scored_scores),
        "min_j_project": min(all_scores) if all_scores else 0.0,
        "tool_call_count": tool_call_count,
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(f"source shard audit aggregate mismatch: {field}")
    execution_root_settled_cost = _validate_source_cost_accounting(
        audit=audit,
        summary=summary,
        checkpoint_local_cost_cny=checkpoint_local_cost,
    )
    for field, expected_value in _ACTIVE_FINAL_RESULT_LOCK.items():
        if audit.get(field) != expected_value:
            raise ValueError(
                f"source shard audit final-result contract mismatch: {field}"
            )
    if assistant_contract is None or final_contract is None:
        raise ValueError("source frozen NoSkill lacks a comparison request contract")
    return (
        {
            "assistant": assistant_contract,
            "final_judge": final_contract,
        },
        checkpoint_local_cost,
        execution_root_settled_cost,
    )


def _launch_endpoint(plan, check_id: str) -> str:
    matches = tuple(
        item.detail for item in plan.preflight_checks if item.check_id == check_id
    )
    if len(matches) != 1 or not matches[0].startswith("https://"):
        raise ValueError(f"launch lacks one verified {check_id} endpoint")
    return matches[0]


def _comparison_launch_projection(plan) -> dict:
    return {
        "schema_version": 1,
        "field_source": "portfolio-launch-plan-v1",
        "corpus": {
            "portfolio_plan_sha256": plan.portfolio_plan_sha256,
            "portfolio_plan_manifest_file_sha256": (
                plan.portfolio_plan_manifest_file_sha256
            ),
            "accepted_ledger_sha256": plan.accepted_ledger_sha256,
            "query_artifact_sha256": plan.query_artifact_sha256,
            "capability_assignments_sha256": plan.capability_assignments_sha256,
            "seed_set_sha256": plan.seed_set_sha256,
        },
        "catalog_and_image_manifests": {
            "base_catalog_sha256": plan.base_catalog_sha256,
            "runtime_catalog_sha256": plan.runtime_catalog_sha256,
            "authorization_file_sha256": plan.authorization_file_sha256,
            "receipt_file_sha256": plan.receipt_file_sha256,
            "receipt_sha256": plan.receipt_sha256,
            "remote_runtime_binding_sha256s": list(plan.remote_runtime_binding_sha256s),
        },
        "models": {
            "assistant": {
                "provider": plan.assistant_provider,
                "model": plan.assistant_model,
                "endpoint": _launch_endpoint(plan, "endpoint.dashscope"),
            },
            "feedback": {
                "provider": plan.feedback_provider,
                "model": plan.feedback_model,
                "endpoint": _launch_endpoint(
                    plan,
                    (
                        "endpoint.kimi_dashscope"
                        if plan.feedback_provider == "kimi"
                        else "endpoint.aifast"
                    ),
                ),
            },
            "final_judge": {
                "provider": plan.final_provider,
                "model": plan.final_model,
                "endpoint": _launch_endpoint(
                    plan,
                    (
                        "endpoint.aifast"
                        if plan.final_provider == "gemini"
                        else "endpoint.kimi_dashscope"
                    ),
                ),
            },
        },
    }


def _comparison_contract(
    *,
    source_launch,
    target_launch,
    source_runtime_lock: dict,
    target_runtime_lock: dict,
    source_request_contract: dict,
    rubric_file_sha256: str,
    rubric_content_sha256: str,
    source_control: dict,
) -> tuple[dict, str]:
    _require_active_final_result_runtime(
        source_runtime_lock,
        label="source runtime lock",
    )
    _require_active_final_result_runtime(
        target_runtime_lock,
        label="target runtime lock",
    )
    source_projection = _comparison_launch_projection(source_launch.plan)
    target_projection = _comparison_launch_projection(target_launch.plan)
    if source_projection != target_projection:
        raise ValueError("source/target launch comparison projection drifted")
    if (
        source_control.get("rubric_file_sha256") != rubric_file_sha256
        or source_control.get("rubric_content_sha256") != rubric_content_sha256
    ):
        raise ValueError("source/target comparison rubric drifted")

    assistant = source_request_contract["assistant"]
    backbone = assistant["backbone"]
    budget = assistant["budget"]
    registry = assistant["registry"]
    expected_backbone = {
        "provider": target_launch.plan.assistant_provider,
        "model": target_launch.plan.assistant_model,
        "endpoint": _launch_endpoint(target_launch.plan, "endpoint.dashscope"),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": target_runtime_lock.get("system_prompt_sha256"),
    }
    if any(backbone.get(key) != value for key, value in expected_backbone.items()):
        raise ValueError("source/target Assistant sampling identity drifted")
    expected_budget = {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "max_tool_calls": 3,
        "max_turns": 5,
        "timeout_ms": 180000,
    }
    if any(budget.get(key) != value for key, value in expected_budget.items()):
        raise ValueError("source/target Assistant budget contract drifted")
    if (
        registry.get("registry_sha256")
        != source_runtime_lock.get("tool_registry_sha256")
        or registry.get("registry_runtime_sha256")
        != source_runtime_lock.get("tool_registry_runtime_sha256")
        or source_runtime_lock.get("tool_registry_sha256")
        != target_runtime_lock.get("tool_registry_sha256")
        or source_runtime_lock.get("tool_registry_runtime_sha256")
        != target_runtime_lock.get("tool_registry_runtime_sha256")
    ):
        raise ValueError("source/target capability registry drifted")
    final_contract = source_request_contract["final_judge"]
    target_final = target_projection["models"]["final_judge"]
    if any(
        final_contract.get(field) != target_final[field]
        for field in ("provider", "model", "endpoint")
    ) or (
        final_contract.get("max_tokens") != 2048
        or final_contract.get("max_attempts") != FINAL_JUDGE_MAX_ATTEMPTS
        or final_contract.get("retry_policy_version")
        != FINAL_JUDGE_RETRY_POLICY_VERSION
        or final_contract.get("retry_policy_sha256") != FINAL_JUDGE_RETRY_POLICY_SHA256
        or final_contract.get("thinking_budget") != FINAL_JUDGE_THINKING_BUDGET
        or final_contract.get("max_billable_input_tokens")
        != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
        or final_contract.get("max_billable_output_tokens")
        != FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        or final_contract.get("transport_policy_version")
        != FINAL_JUDGE_TRANSPORT_POLICY_VERSION
        or final_contract.get("transport_policy_sha256")
        != FINAL_JUDGE_TRANSPORT_POLICY_SHA256
        or final_contract.get("requested_response_format") != "json_object"
    ):
        raise ValueError("source/target final-Judge identity drifted")
    if (
        source_runtime_lock.get("final_judge_parser_policy_version")
        != target_runtime_lock.get("final_judge_parser_policy_version")
        or source_runtime_lock.get("final_judge_parser_policy_sha256")
        != target_runtime_lock.get("final_judge_parser_policy_sha256")
        or any(
            source_runtime_lock.get(field) != target_runtime_lock.get(field)
            for field in _ACTIVE_FINAL_RESULT_LOCK
        )
        or any(
            lock.get("noskill_execution_contract_sha256")
            != NOSKILL_EXECUTION_CONTRACT_SHA256
            or lock.get("noskill_execution_policy_version")
            != NOSKILL_EXECUTION_POLICY_VERSION
            for lock in (source_runtime_lock, target_runtime_lock)
        )
    ):
        raise ValueError("source/target parser or NoSkill contract drifted")

    payload = {
        "schema_version": 1,
        "kind": "portfolio-external-frozen-noskill-comparison-contract",
        "field_sources": {
            "corpus_catalog_models_endpoints": "source-and-target-launch-plan-v1",
            "assistant_sampling_budget_registry": (
                "source-noskill-checkpoints-and-target-runner-constants-v1"
            ),
            "rubric": "source-and-target-execution-control",
            "parser": "source-and-target-runtime-lock",
            "noskill_execution": NOSKILL_EXECUTION_POLICY_VERSION,
        },
        "launch_projection": target_projection,
        "assistant_sampling": {
            key: backbone[key]
            for key in (
                "provider",
                "model",
                "endpoint",
                "temperature",
                "top_p",
                "seed",
                "system_prompt_sha256",
            )
        },
        "assistant_budget_contract": {
            **expected_budget,
            "budget_sha256": budget["budget_sha256"],
        },
        "capability_registry": {
            "registry_sha256": registry["registry_sha256"],
            "registry_runtime_sha256": registry["registry_runtime_sha256"],
            "registry_lock_sha256": registry["lock_sha256"],
        },
        "feedback_identity": target_projection["models"]["feedback"],
        "final_judge_identity": final_contract,
        "rubric": {
            "file_sha256": rubric_file_sha256,
            "content_sha256": rubric_content_sha256,
        },
        "parser": {
            "result_schema_version": target_runtime_lock[
                "final_judge_result_schema_version"
            ],
            "cache_namespace": target_runtime_lock["final_judge_cache_namespace"],
            "policy_version": target_runtime_lock["final_judge_parser_policy_version"],
            "policy_sha256": target_runtime_lock["final_judge_parser_policy_sha256"],
            "card_requirement_guard_policy_version": target_runtime_lock[
                "card_requirement_guard_policy_version"
            ],
            "card_requirement_guard_policy_sha256": target_runtime_lock[
                "card_requirement_guard_policy_sha256"
            ],
            "max_attempts": target_runtime_lock["final_judge_max_attempts"],
            "retry_policy_version": target_runtime_lock[
                "final_judge_retry_policy_version"
            ],
            "retry_policy_sha256": target_runtime_lock[
                "final_judge_retry_policy_sha256"
            ],
        },
        "noskill_execution_contract": {
            **noskill_execution_contract_payload(),
            "contract_sha256": NOSKILL_EXECUTION_CONTRACT_SHA256,
            "policy_version": NOSKILL_EXECUTION_POLICY_VERSION,
        },
        "excluded_as_treatment_or_run_specific": [
            "matrix_run_id",
            "launch_plan_sha256",
            "runtime_lock_sha256",
            "skill_bank_sha256s",
        ],
    }
    return payload, sha256_bytes(canonical_json_bytes(payload))


def _external_frozen_noskill_binding(
    *,
    source_execution_root: Path,
    source_shard_id: str,
    expected_audit_sha256: str,
    target_launch,
    target_runtime_lock: dict,
    rubric_file_sha256: str,
    rubric_content_sha256: str,
) -> tuple[dict, bytes]:
    if not _SHA256.fullmatch(expected_audit_sha256):
        raise ValueError("expected source shard audit SHA-256 is invalid")
    control_bytes, source_control = _canonical_object(
        source_execution_root / "execution-control.json",
        label="source execution control",
    )
    source_control_sha256 = _require_self_hash(
        source_control,
        "control_sha256",
        label="source execution control",
    )
    source_launch = load_portfolio_launch_package(
        source_control["launch_root"],
        expected_plan_file_sha256=source_control["launch_plan_file_sha256"],
        _allow_legacy_budget_contract=True,
    )
    source_runtime_root = Path(source_control["runtime_root"])
    source_runtime_bytes, source_runtime_lock = _canonical_object(
        source_runtime_root / "runtime-lock.json",
        label="source runtime lock",
    )
    if sha256_bytes(source_runtime_bytes) != source_control.get(
        "runtime_lock_file_sha256"
    ):
        raise ValueError("source runtime lock file digest mismatch")
    source_runtime_sha256 = _require_self_hash(
        source_runtime_lock,
        "runtime_lock_sha256",
        label="source runtime lock",
    )
    if (
        source_control.get("runtime_lock_sha256") != source_runtime_sha256
        or source_control.get("launch_plan_sha256")
        != source_launch.plan.launch_plan_sha256
        or target_runtime_lock.get("compatible_frozen_noskill_runtime_lock_sha256")
        != source_runtime_sha256
    ):
        raise ValueError("source/target frozen NoSkill runtime binding mismatch")
    source_key = read_stable_regular_file(
        source_execution_root / "blinding-key.bin",
        label="source execution blinding key",
        max_bytes=32,
    )
    if len(source_key) != 32 or sha256_bytes(source_key) != source_control.get(
        "blinding_key_sha256"
    ):
        raise ValueError("source execution blinding key mismatch")

    try:
        source_shard = next(
            item
            for item in source_launch.plan.shards
            if item.shard_id == source_shard_id
        )
    except StopIteration as error:
        raise ValueError("source frozen shard is absent from source launch") from error
    if source_shard.config != "noskill" or source_shard.query_count != 25:
        raise ValueError("source frozen shard must be one complete NoSkill shard")
    source_members = tuple(
        item for item in source_launch.instances if item.shard_id == source_shard_id
    )
    if len(source_members) != 25:
        raise ValueError("source frozen NoSkill membership is incomplete")

    target_candidates = tuple(
        item
        for item in target_launch.plan.shards
        if item.config == "noskill"
        and item.accepted_batch_id == source_shard.accepted_batch_id
    )
    if len(target_candidates) != 1:
        raise ValueError("target launch lacks one matching NoSkill comparison shard")
    target_shard = target_candidates[0]
    target_members = tuple(
        item
        for item in target_launch.instances
        if item.shard_id == target_shard.shard_id
    )
    compatibility_fields = (
        "query_id",
        "query_sha256",
        "public_input_sha256",
        "asset_id",
        "image_path",
        "image_sha256",
    )
    for source_member, target_member in zip(
        source_members, target_members, strict=True
    ):
        if any(
            getattr(source_member, field) != getattr(target_member, field)
            for field in compatibility_fields
        ):
            raise ValueError(
                f"target launch query/public/image drift: {source_member.query_id}"
            )

    source_shard_root = source_execution_root / source_shard.output_relpath
    summary_bytes, summary = _canonical_object(
        source_shard_root / "shard-summary.json",
        label="source frozen shard summary",
    )
    summary_sha256 = _require_self_hash(
        summary,
        "summary_sha256",
        label="source frozen shard summary",
    )
    audit_bytes, audit = _canonical_object(
        source_shard_root / "shard-audit.json",
        label="source frozen shard audit",
    )
    audit_sha256 = _require_self_hash(
        audit,
        "audit_sha256",
        label="source frozen shard audit",
    )
    if audit_sha256 != expected_audit_sha256:
        raise ValueError("source frozen shard audit differs from expected self digest")
    if (
        summary.get("kind") != "portfolio-shard-summary"
        or summary.get("status") != "complete"
        or summary.get("config") != "noskill"
        or summary.get("shard_id") != source_shard_id
        or summary.get("query_count") != 25
        or audit.get("kind") != "portfolio-shard-audit"
        or audit.get("formal_eligible") is not False
        or audit.get("config") != "noskill"
        or audit.get("shard_id") != source_shard_id
        or audit.get("query_count") != 25
        or audit.get("shard_summary_sha256") != summary_sha256
        or audit.get("runtime_lock_sha256") != source_runtime_sha256
        or audit.get("launch_plan_sha256") != source_launch.plan.launch_plan_sha256
    ):
        raise ValueError("source frozen shard summary/audit binding mismatch")
    (
        source_request_contract,
        checkpoint_local_cost,
        execution_root_settled_cost,
    ) = _validate_source_checkpoints(
        source_root=source_execution_root,
        source_shard=source_shard,
        members=source_members,
        source_runtime_lock=source_runtime_lock,
        summary=summary,
        audit=audit,
    )
    comparison_payload, comparison_sha256 = _comparison_contract(
        source_launch=source_launch,
        target_launch=target_launch,
        source_runtime_lock=source_runtime_lock,
        target_runtime_lock=target_runtime_lock,
        source_request_contract=source_request_contract,
        rubric_file_sha256=rubric_file_sha256,
        rubric_content_sha256=rubric_content_sha256,
        source_control=source_control,
    )
    artifact_file_sha256s, artifact_set_sha256 = _artifact_set(
        source_shard_root,
        source_shard.query_ids,
    )
    compatibility_payload = [
        {field: getattr(item, field) for field in compatibility_fields}
        for item in target_members
    ]
    binding_payload = {
        "source_execution_root": source_execution_root.as_posix(),
        "source_execution_control_file_sha256": sha256_bytes(control_bytes),
        "source_execution_control_sha256": source_control_sha256,
        "source_launch_root": source_control["launch_root"],
        "source_launch_plan_file_sha256": source_control["launch_plan_file_sha256"],
        "source_launch_plan_sha256": source_launch.plan.launch_plan_sha256,
        "source_runtime_root": source_control["runtime_root"],
        "source_runtime_lock_file_sha256": sha256_bytes(source_runtime_bytes),
        "source_runtime_lock_sha256": source_runtime_sha256,
        "source_shard_id": source_shard_id,
        "source_shard_sha256": source_shard.shard_sha256,
        "source_shard_output_relpath": source_shard.output_relpath,
        "source_shard_summary_file_sha256": sha256_bytes(summary_bytes),
        "source_shard_summary_sha256": summary_sha256,
        "source_shard_audit_file_sha256": sha256_bytes(audit_bytes),
        "source_shard_audit_sha256": audit_sha256,
        "target_shard_id": target_shard.shard_id,
        "target_shard_sha256": target_shard.shard_sha256,
        "accepted_batch_id": source_shard.accepted_batch_id,
        "query_count": 25,
        "assistant_checkpoint_count": 25,
        "final_checkpoint_count": 25,
        "checkpoint_local_dashscope_observed_cost_cny": _cny_text(
            checkpoint_local_cost
        ),
        "source_execution_root_settled_actual_cost_cny": _cny_text(
            execution_root_settled_cost
        ),
        "artifact_file_count": 52,
        "artifact_file_sha256s": artifact_file_sha256s,
        "artifact_set_sha256": artifact_set_sha256,
        "query_public_image_compatibility_sha256": sha256_bytes(
            canonical_json_bytes(compatibility_payload)
        ),
        "blinding_key_sha256": sha256_bytes(source_key),
        "comparison_contract_payload": comparison_payload,
        "comparison_contract_sha256": comparison_sha256,
    }
    return {
        **binding_payload,
        "binding_sha256": sha256_bytes(canonical_json_bytes(binding_payload)),
    }, source_key


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        print("prepare-portfolio-execution: output directory exists", file=sys.stderr)
        return 2
    try:
        launch = load_portfolio_launch_package(
            args.launch_root,
            expected_plan_file_sha256=args.launch_plan_file_sha256,
        )
        partial_repair = args.execution_scope == "partial_shard_repair"
        core_canary = args.execution_scope == "core_canary"
        static_opt = args.execution_scope == "static_opt_rollout"
        if (
            not partial_repair
            and not core_canary
            and not static_opt
            and launch.plan.kind == "portfolio-core-split-x5-launch-plan"
        ):
            raise ValueError(
                "complete Core matrix execution is not authorized; Gate 0 permits "
                "only the frozen 25-query Core canary"
            )
        if partial_repair:
            if launch.plan.blockers not in (
                (),
                _ALLOWED_PARTIAL_LAUNCH_BLOCKERS,
            ):
                raise ValueError(
                    "launch has blockers beyond partial-repair budget approval"
                )
            if not launch.plan.execution_ready and (
                launch.plan.blockers != _ALLOWED_PARTIAL_LAUNCH_BLOCKERS
            ):
                raise ValueError("launch plan is not ready for partial shard repair")
        elif launch.plan.blockers or not launch.plan.execution_ready:
            scope_label = (
                "Core canary"
                if core_canary
                else "Static opt rollout"
                if static_opt
                else "full matrix"
            )
            raise ValueError(f"{scope_label} requires a blocker-free ready launch")
        launch_budget = launch.plan.budget
        if (
            launch_budget.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION
            or launch_budget.policy_sha256 != PORTFOLIO_BUDGET_POLICY_SHA256
            or launch_budget.provider_pricing_contract_version
            != PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            or launch_budget.provider_pricing_contract_sha256
            != PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
            or launch_budget.over_budget_policy
            != "reserve_before_each_provider_call_halt_before_call"
        ):
            raise ValueError("execution launch lacks the active hard-budget contract")
        launch_approved = Decimal(
            str(launch_budget.operator_approved_dashscope_budget_cny)
        ).quantize(PORTFOLIO_BUDGET_CNY_QUANTUM)
        if args.approved_dashscope_budget_cny != launch_approved:
            raise ValueError(
                "execution budget differs from the launch's operator-approved cap"
            )
        if (
            launch.state.status != "not_started"
            or launch.state.model_calls_performed != 0
        ):
            raise ValueError("target launch must be fresh and not started")
        if (
            args.prior_dashscope_observed_cost_cny < 0
            or args.prior_dashscope_observed_cost_cny
            >= args.approved_dashscope_budget_cny
        ):
            raise ValueError(
                "prior observed cost must be non-negative and below the approved budget"
            )
        phase_cap = (
            args.approved_dashscope_budget_cny
            if args.phase_cumulative_cap_cny is None
            else args.phase_cumulative_cap_cny
        )
        if (
            phase_cap <= args.prior_dashscope_observed_cost_cny
            or phase_cap > args.approved_dashscope_budget_cny
        ):
            raise ValueError(
                "phase cumulative cap must exceed prior cost and stay within "
                "the launch operator-approved total cap"
            )
        if core_canary and args.approved_dashscope_budget_cny > Decimal("100"):
            raise ValueError(
                "Gate 0 Core canary exceeds the separately approved CNY 100 cap"
            )
        shard_by_id = {item.shard_id: item for item in launch.plan.shards}
        if partial_repair:
            if (
                len(args.authorized_shard_id) != 4
                or len(set(args.authorized_shard_id)) != 4
            ):
                raise ValueError(
                    "partial shard repair requires exactly four unique shard IDs"
                )
            try:
                authorized_shards = tuple(
                    sorted(
                        (shard_by_id[item] for item in args.authorized_shard_id),
                        key=lambda item: item.shard_ordinal,
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "authorized shard is absent from target launch"
                ) from error
            if (
                {item.config for item in authorized_shards} != _SKILLED_CONFIGS
                or len({item.accepted_batch_id for item in authorized_shards}) != 1
                or any(item.query_count != 25 for item in authorized_shards)
            ):
                raise ValueError(
                    "authorized shards must be the four skilled configs of one batch"
                )
            if (
                args.source_execution_root is None
                or args.source_shard_id is None
                or args.expected_source_shard_audit_sha256 is None
            ):
                raise ValueError(
                    "partial shard repair requires the frozen NoSkill source"
                )
        elif core_canary:
            if (
                launch.plan.kind != "portfolio-core-split-x5-launch-plan"
                or launch.plan.dataset_profile != "core"
                or launch.plan.selected_splits != ("dev_mini",)
            ):
                raise ValueError(
                    "Core canary requires a Core launch restricted to dev_mini"
                )
            if (
                args.authorized_shard_id
                or args.source_execution_root is not None
                or args.source_shard_id is not None
                or args.expected_source_shard_audit_sha256 is not None
            ):
                raise ValueError(
                    "Core canary selects its first frozen batch and cannot import "
                    "repair evidence or accept an ad-hoc shard subset"
                )
            first_batch_id = min(
                launch.plan.shards,
                key=lambda item: item.shard_ordinal,
            ).accepted_batch_id
            authorized_shards = tuple(
                sorted(
                    (
                        item
                        for item in launch.plan.shards
                        if item.accepted_batch_id == first_batch_id
                    ),
                    key=lambda item: item.shard_ordinal,
                )
            )
            if (
                len(authorized_shards) != 5
                or {item.config for item in authorized_shards}
                != {"noskill", *_SKILLED_CONFIGS}
                or any(item.query_count != 25 for item in authorized_shards)
                or len({item.query_ids for item in authorized_shards}) != 1
            ):
                raise ValueError(
                    "Core canary first batch does not cover five logical configs"
                )
        elif static_opt:
            if (
                launch.plan.kind != "portfolio-core-static-opt-800x1-launch-plan"
                or launch.plan.execution_mode != "static_opt_rollout"
                or launch.plan.dataset_profile != "core"
                or launch.plan.selected_splits != ("opt_pool",)
                or launch.plan.config_order != ("llm_static",)
                or launch.plan.query_count != 800
                or launch.plan.instance_count != 800
                or launch.plan.shard_count != 32
                or launch.plan.calls.final_judge_instance_count != 0
                or launch.plan.calls.final_judge_call_count != 0
                or args.legacy_launch_compat
            ):
                raise ValueError(
                    "static opt rollout requires the exact 800x1 Assistant-only launch"
                )
            if (
                args.authorized_shard_id
                or args.source_execution_root is not None
                or args.source_shard_id is not None
                or args.expected_source_shard_audit_sha256 is not None
            ):
                raise ValueError(
                    "static opt rollout cannot import evidence or select ad-hoc shards"
                )
            authorized_shards = tuple(
                sorted(launch.plan.shards, key=lambda item: item.shard_ordinal)
            )
            if (
                len(authorized_shards) != 32
                or any(
                    item.config != "llm_static" or item.query_count != 25
                    for item in authorized_shards
                )
                or len(
                    {
                        query_id
                        for item in authorized_shards
                        for query_id in item.query_ids
                    }
                )
                != 800
            ):
                raise ValueError("static opt launch membership is incomplete")
        else:
            if (
                args.authorized_shard_id
                or args.source_execution_root is not None
                or args.source_shard_id is not None
                or args.expected_source_shard_audit_sha256 is not None
            ):
                raise ValueError(
                    "full matrix must use a fresh execution root and cannot import "
                    "partial-repair evidence or an explicit shard subset"
                )
            authorized_shards = tuple(
                sorted(launch.plan.shards, key=lambda item: item.shard_ordinal)
            )
            if (
                len(authorized_shards) != launch.plan.shard_count
                or any(item.query_count != 25 for item in authorized_shards)
                or set(shard_by_id) != {item.shard_id for item in authorized_shards}
            ):
                raise ValueError("full matrix launch membership is incomplete")
        runtime_lock_bytes = (args.runtime_root / "runtime-lock.json").read_bytes()
        if sha256_bytes(runtime_lock_bytes) != args.runtime_lock_file_sha256:
            raise ValueError("runtime lock file digest mismatch")
        runtime_lock = parse_canonical_json(
            runtime_lock_bytes,
            label="Portfolio runtime lock",
        )
        if not isinstance(runtime_lock, dict):
            raise ValueError("Portfolio runtime lock must contain an object")
        target_runtime_sha256 = _require_self_hash(
            runtime_lock,
            "runtime_lock_sha256",
            label="Portfolio runtime lock",
        )
        if runtime_lock.get("formal_eligible") is not False:
            raise ValueError("execution runtime must remain Portfolio-only")
        if target_runtime_sha256 != runtime_lock.get("runtime_lock_sha256"):
            raise ValueError("Portfolio runtime lock content digest mismatch")
        target_runtime_bindings = tuple(
            item
            for item in launch.plan.artifacts
            if item.artifact_id == "assistant_runtime_lock"
        )
        if (
            len(target_runtime_bindings) != 1
            or target_runtime_bindings[0].status != "verified"
            or target_runtime_bindings[0].file_sha256 != args.runtime_lock_file_sha256
            or target_runtime_bindings[0].content_sha256 != target_runtime_sha256
        ):
            raise ValueError("target launch does not bind the supplied runtime lock")
        treatment_runtime = None
        static_runtime = None
        execution_artifact_aliases: list[dict] = []
        rubric = None
        rubric_file_sha256 = None
        if static_opt:
            if args.rubric is not None:
                raise ValueError(
                    "static opt rollout must not bind the legacy Final-Judge rubric"
                )
            static_runtime = load_verified_portfolio_static_opt_runtime(
                args.runtime_root,
                expected_runtime_lock_file_sha256=args.runtime_lock_file_sha256,
            )
            if dict(static_runtime.runtime_lock) != runtime_lock:
                raise ValueError(
                    "verified Static opt runtime differs from the execution lock"
                )
            _require_active_budget_runtime(runtime_lock, label="execution runtime")
            if (
                runtime_lock.get("execution_artifact_aliases") != []
                or runtime_lock.get(
                    "execution_artifact_alias_provider_model_call_count"
                )
                != 0
            ):
                raise ValueError("Static opt runtime must contain no artifact aliases")
            launch_artifacts = {
                item.artifact_id: item for item in launch.plan.artifacts
            }
            if set(launch_artifacts) != {
                "assistant_runtime_lock",
                "static_contract_refresh_receipt",
                "bank.llm_static",
            }:
                raise ValueError(
                    "Static opt launch must bind only its runtime, refresh receipt, "
                    "and llm_static Bank"
                )
            receipt_binding = launch_artifacts["static_contract_refresh_receipt"]
            bank_binding = launch_artifacts["bank.llm_static"]
            if (
                receipt_binding.status != "verified"
                or receipt_binding.file_sha256
                != static_runtime.refresh_receipt_file_sha256
                or receipt_binding.content_sha256
                != static_runtime.refresh_receipt["receipt_sha256"]
                or Path(receipt_binding.path or "").resolve(strict=True)
                != (args.runtime_root / STATIC_OPT_REFRESH_RECEIPT_FILE).resolve(
                    strict=True
                )
                or bank_binding.status != "verified"
                or bank_binding.file_sha256 != static_runtime.bank_file_sha256
                or bank_binding.content_sha256 != static_runtime.bank.bank_sha256
                or Path(bank_binding.path or "").resolve(strict=True)
                != (args.runtime_root / STATIC_OPT_BANK_FILE).resolve(strict=True)
            ):
                raise ValueError(
                    "Static opt launch artifact bindings differ from its runtime"
                )
        else:
            if args.rubric is None:
                raise ValueError("non-Static execution requires the frozen rubric")
            treatment_runtime = load_verified_portfolio_treatment_runtime(
                args.runtime_root,
                expected_runtime_lock_file_sha256=args.runtime_lock_file_sha256,
            )
            if dict(treatment_runtime.runtime_lock) != runtime_lock:
                raise ValueError(
                    "verified treatment runtime differs from the execution lock"
                )
            execution_artifact_aliases = [
                item.model_dump(mode="json")
                for item in treatment_runtime.chain.execution_artifact_aliases
            ]
            if (
                runtime_lock.get("execution_artifact_aliases")
                != execution_artifact_aliases
                or runtime_lock.get(
                    "execution_artifact_alias_provider_model_call_count"
                )
                != 0
            ):
                raise ValueError(
                    "runtime execution-artifact aliases differ from the treatment chain"
                )
            target_chain_bindings = tuple(
                item
                for item in launch.plan.artifacts
                if item.artifact_id == "treatment_chain_manifest"
            )
            if (
                len(target_chain_bindings) != 1
                or target_chain_bindings[0].status != "verified"
                or target_chain_bindings[0].file_sha256
                != treatment_runtime.manifest_file_sha256
                or target_chain_bindings[0].content_sha256
                != treatment_runtime.chain.manifest.chain_sha256
            ):
                raise ValueError(
                    "target launch does not bind the verified treatment chain"
                )
            evaluator_root = SOURCE_ROOT / "skillchain" / "evaluation"
            if (
                runtime_lock.get("evaluator_outputs_file_sha256")
                != sha256_bytes((evaluator_root / "evaluator_outputs.py").read_bytes())
                or runtime_lock.get("final_runtime_file_sha256")
                != sha256_bytes((evaluator_root / "final_runtime.py").read_bytes())
                or runtime_lock.get("final_judge_parser_policy_version")
                != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
                or runtime_lock.get("final_judge_parser_policy_sha256")
                != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
            ):
                raise ValueError(
                    "execution runtime does not bind the active final-Judge parser"
                )
            _require_active_final_result_runtime(
                runtime_lock,
                label="execution runtime",
            )
            rubric_bytes = args.rubric.read_bytes()
            rubric = RubricSnapshot.model_validate_json(rubric_bytes, strict=True)
            rubric_file_sha256 = sha256_bytes(rubric_bytes)
            if rubric_file_sha256 != PORTFOLIO_FINAL_RUBRIC_FILE_SHA256:
                raise ValueError(
                    "execution rubric differs from the frozen Portfolio rubric"
                )
        external_frozen_shard: dict | None = None
        if partial_repair:
            assert args.source_execution_root is not None
            assert args.source_shard_id is not None
            assert args.expected_source_shard_audit_sha256 is not None
            assert rubric_file_sha256 is not None
            assert rubric is not None
            external_frozen_shard, key = _external_frozen_noskill_binding(
                source_execution_root=args.source_execution_root,
                source_shard_id=args.source_shard_id,
                expected_audit_sha256=(args.expected_source_shard_audit_sha256),
                target_launch=launch,
                target_runtime_lock=runtime_lock,
                rubric_file_sha256=rubric_file_sha256,
                rubric_content_sha256=rubric.content_sha256,
            )
            authorized_batch_id = authorized_shards[0].accepted_batch_id
            if external_frozen_shard["accepted_batch_id"] != authorized_batch_id:
                raise ValueError(
                    "external frozen NoSkill and authorized skilled shards "
                    "differ by batch"
                )
        else:
            key = secrets.token_bytes(32)
            authorized_batch_id = (
                authorized_shards[0].accepted_batch_id if core_canary else None
            )
        incremental_authorized = phase_cap - args.prior_dashscope_observed_cost_cny
        control_payload = {
            "schema_version": 4,
            "kind": "portfolio-matrix-execution-control",
            "track": "portfolio",
            "formal_eligible": False,
            "status": "prepared_not_started",
            "execution_scope": args.execution_scope,
            "matrix_run_id": launch.plan.matrix_run_id,
            "launch_root": args.launch_root.as_posix(),
            "launch_plan_file_sha256": args.launch_plan_file_sha256,
            "launch_plan_sha256": launch.plan.launch_plan_sha256,
            "runtime_root": args.runtime_root.as_posix(),
            "runtime_lock_file_sha256": args.runtime_lock_file_sha256,
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "execution_artifact_aliases": execution_artifact_aliases,
            "execution_artifact_alias_provider_model_call_count": 0,
            "blinding_key_sha256": sha256_bytes(key),
            "approved_dashscope_budget_cny": _cny_text(
                args.approved_dashscope_budget_cny
            ),
            "phase_cumulative_cap_cny": _cny_text(phase_cap),
            "incremental_authorized_dashscope_budget_cny": _cny_text(
                incremental_authorized
            ),
            "prior_dashscope_observed_cost_cny": (
                _cny_text(args.prior_dashscope_observed_cost_cny)
            ),
            "budget_ledger_relpath": "budget-ledger",
            "budget_authority_relpath": "budget-ledger/budget-authority.json",
            "budget_authority_creation_policy": "create_only_on_first_execute",
            **_ACTIVE_BUDGET_CONTRACT,
            "authorized_batch_id": authorized_batch_id,
            "authorized_shard_ids": [item.shard_id for item in authorized_shards],
            "authorized_shards": [
                {
                    "shard_id": item.shard_id,
                    "shard_ordinal": item.shard_ordinal,
                    "shard_sha256": item.shard_sha256,
                    "config": item.config,
                    "accepted_batch_id": item.accepted_batch_id,
                    "query_count": item.query_count,
                }
                for item in authorized_shards
            ],
            "local_authorized_shard_count": len(authorized_shards),
            "local_authorized_instance_count": sum(
                item.query_count for item in authorized_shards
            ),
            "external_frozen_shard_count": (
                1 if external_frozen_shard is not None else 0
            ),
            "comparison_shard_count": (
                len(authorized_shards) + 1
                if external_frozen_shard is not None
                else len(authorized_shards)
            ),
            "shard_count": launch.plan.shard_count,
            "instance_count": launch.plan.instance_count,
            "launch_shard_count": launch.plan.shard_count,
            "launch_instance_count": launch.plan.instance_count,
            "legacy_launch_compat": bool(args.legacy_launch_compat),
            "model_calls_performed": 0,
        }
        if static_opt:
            assert static_runtime is not None
            control_payload.update(
                {
                    "evaluation_stages": ["assistant", "gcs_v2"],
                    "assistant_checkpoint_schema_version": 2,
                    "gcs_policy_version": GCS_V2_POLICY_VERSION,
                    "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
                    "gcs_scorer_evidence_policy_version": (
                        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
                    ),
                    "static_bank_file_sha256": static_runtime.bank_file_sha256,
                    "static_bank_sha256": static_runtime.bank.bank_sha256,
                    "static_contract_refresh_receipt_file_sha256": (
                        static_runtime.refresh_receipt_file_sha256
                    ),
                    "static_contract_refresh_receipt_sha256": (
                        static_runtime.refresh_receipt["receipt_sha256"]
                    ),
                    "semantic_authoring_input_file_sha256": (
                        static_runtime.semantic_authoring_input_file_sha256
                    ),
                    "semantic_authoring_input_sha256": (
                        static_runtime.semantic_authoring_input.input_sha256
                    ),
                    "core_runtime_sources_receipt_file_sha256": (
                        static_runtime.core_source_receipt_file_sha256
                    ),
                    "core_runtime_sources_receipt_sha256": (
                        static_runtime.core_source_receipt["receipt_sha256"]
                    ),
                    "runtime_data_sha256": runtime_lock["runtime_data_sha256"],
                    "task_spec_version": runtime_lock["task_spec_version"],
                    "task_spec_sha256": runtime_lock["task_spec_sha256"],
                    "task_spec_file_sha256": runtime_lock["task_spec_file_sha256"],
                    "pairwise_judge_enabled": False,
                    "legacy_final_judge_enabled": False,
                    "analyzer_provider_call_count": 0,
                }
            )
        else:
            assert treatment_runtime is not None
            assert args.rubric is not None
            assert rubric_file_sha256 is not None
            assert rubric is not None
            control_payload.update(
                {
                    "treatment_chain_manifest_file_sha256": (
                        treatment_runtime.manifest_file_sha256
                    ),
                    "treatment_chain_sha256": (
                        treatment_runtime.chain.manifest.chain_sha256
                    ),
                    "treatment_record_sha256s": {
                        item.config: item.receipt_sha256
                        for item in treatment_runtime.chain.manifest.records
                    },
                    "execution_artifact_alias_policy_version": runtime_lock[
                        "execution_artifact_alias_policy_version"
                    ],
                    "rubric_path": args.rubric.as_posix(),
                    "rubric_file_sha256": rubric_file_sha256,
                    "rubric_content_sha256": rubric.content_sha256,
                }
            )
        if external_frozen_shard is not None:
            control_payload["external_frozen_shard"] = external_frozen_shard
        else:
            control_payload["authorized_batch_ids"] = sorted(
                {item.accepted_batch_id for item in authorized_shards}
            )
        control = {
            **control_payload,
            "control_sha256": sha256_bytes(canonical_json_bytes(control_payload)),
        }
        if static_opt:
            assert static_runtime is not None
            validate_portfolio_static_opt_execution_control(control, static_runtime)
        args.output_dir.mkdir(parents=True)
        atomic_create_file(args.output_dir / "blinding-key.bin", key)
        atomic_create_file(
            args.output_dir / "execution-control.json",
            canonical_json_bytes(control),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"prepare-portfolio-execution: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "control_sha256": control["control_sha256"],
                "external_frozen_shard_audit_sha256": (
                    external_frozen_shard["source_shard_audit_sha256"]
                    if external_frozen_shard is not None
                    else None
                ),
                "incremental_authorized_dashscope_budget_cny": (
                    _cny_text(incremental_authorized)
                ),
                "local_authorized_shard_count": len(authorized_shards),
                "model_calls_performed": 0,
                "output_dir": str(args.output_dir),
                "status": control["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
