"""Deep-validate one completed Portfolio shard and update its launch state."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path, PurePosixPath
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.run_portfolio_assistant_smoke import _hash, _model  # noqa: E402
from scripts.run_portfolio_shard import (  # noqa: E402
    _assistant_row,
    _assistant_result,
    _load_control,
    _verify_scorer_sidecar_binding,
)
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
    build_assistant_query_input,
    build_legacy_assistant_query_input,
)
from skillchain.evaluation.evaluator_outputs import (  # noqa: E402
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
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
    FINAL_JUDGE_RETRY_POLICY_SHA256_V2,
    FINAL_JUDGE_RETRY_POLICY_SHA256_V3,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_RETRY_POLICY_VERSION_V2,
    FINAL_JUDGE_RETRY_POLICY_VERSION_V3,
    FINAL_JUDGE_THINKING_BUDGET,
    FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
    load_final_judge_evaluation_result,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    PortfolioLaunchState,
    load_portfolio_launch_package,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PortfolioAttemptReceipt,
    PortfolioBudgetLedgerState,
    load_portfolio_budget_ledger,
    portfolio_attempt_provider_call_count,
    portfolio_budget_settled_cost_cny,
)
from skillchain.evaluation.portfolio_gcs import (  # noqa: E402
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_gcs_evidence import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.runners.assistant import (  # noqa: E402
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_replace_file,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARED_ROUTE_CONFIGS = frozenset({"s1s2", "full"})
_QUERY_ARTIFACT_PATH = REPOSITORY_ROOT / "data" / "queries" / "queries.jsonl"
_ACTIVE_FINAL_RESULT_LOCK = {
    "final_judge_result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
    "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
    "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
    "final_judge_thinking_budget": FINAL_JUDGE_THINKING_BUDGET,
    "final_judge_transport_policy_version": FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
    "final_judge_transport_policy_sha256": FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    "final_judge_requested_response_format": "json_object",
    "final_judge_max_billable_input_tokens": (FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS),
    "final_judge_max_billable_output_tokens": (FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS),
    "final_judge_provider_input_token_reserve": (
        FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
    ),
    "final_judge_provider_output_token_reserve": (
        FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
    ),
    "final_judge_provider_pricing_status": PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    "card_requirement_guard_policy_version": CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    "card_requirement_guard_policy_sha256": CARD_REQUIREMENT_GUARD_POLICY_SHA256,
}
_HISTORICAL_FINAL_RESULT_V6_LOCK = {
    "final_judge_result_schema_version": 6,
    "final_judge_cache_namespace": "final-evaluator-v7",
    "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
    "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION_V2,
    "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256_V2,
    "final_judge_thinking_budget": 6_144,
    "final_judge_max_billable_input_tokens": 32_768,
    "final_judge_max_billable_output_tokens": 8_192,
    "final_judge_provider_input_token_reserve": 229_376,
    "final_judge_provider_output_token_reserve": 8_192,
    "card_requirement_guard_policy_version": CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    "card_requirement_guard_policy_sha256": CARD_REQUIREMENT_GUARD_POLICY_SHA256,
}
_HISTORICAL_FINAL_RESULT_V7_LOCK = {
    **_HISTORICAL_FINAL_RESULT_V6_LOCK,
    "final_judge_result_schema_version": 7,
    "final_judge_cache_namespace": "final-evaluator-v8",
    "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION_V3,
    "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256_V3,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument(
        "--artifact-alias-source-shard-id",
        help=(
            "Finalize this logical shard by reusing the exact source shard "
            "artifacts; performs zero provider calls."
        ),
    )
    parser.add_argument(
        "--legacy-launch-compat",
        action="store_true",
        help="Validate against the legacy public projection frozen by the launch.",
    )
    return parser


def _load_canonical_object(path: Path, *, label: str) -> dict:
    value = parse_canonical_json(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _require_self_hash(value: dict, field: str, *, label: str) -> str:
    supplied = value.get(field)
    if not isinstance(supplied, str) or not _SHA256.fullmatch(supplied):
        raise ValueError(f"{label} lacks a valid {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if supplied != _hash(unsigned):
        raise ValueError(f"{label} self hash mismatch")
    return supplied


def _control_cny(control: dict, field: str) -> Decimal:
    value = control.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d{12}", value):
        raise ValueError(f"execution control {field} must be a 12-place CNY string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:  # pragma: no cover - regex excludes it
        raise ValueError(f"execution control {field} is not decimal") from error
    return parsed


def _budget_ledger_root(execution_root: Path, control: dict) -> Path:
    relative = control.get("budget_ledger_relpath")
    authority_relative = control.get("budget_authority_relpath")
    if (
        relative != "budget-ledger"
        or authority_relative != "budget-ledger/budget-authority.json"
        or control.get("budget_authority_creation_policy")
        != "create_only_on_first_execute"
    ):
        raise ValueError("execution control budget-ledger layout drifted")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError("execution control budget-ledger path is unsafe")
    return execution_root.joinpath(*parsed.parts)


def _validate_budget_authority(
    *,
    control: dict,
    launch: object,
    budget_ledger: PortfolioBudgetLedgerState,
) -> None:
    """Bind cumulative prior/cap fields to the launch and immutable ledger."""

    approved = _control_cny(control, "approved_dashscope_budget_cny")
    phase_cap = _control_cny(control, "phase_cumulative_cap_cny")
    incremental = _control_cny(control, "incremental_authorized_dashscope_budget_cny")
    prior = _control_cny(control, "prior_dashscope_observed_cost_cny")
    launch_approved = Decimal(
        str(launch.plan.budget.operator_approved_dashscope_budget_cny)
    )
    if (
        approved != launch_approved
        or phase_cap > approved
        or prior >= phase_cap
        or incremental != phase_cap - prior
    ):
        raise ValueError("execution cumulative budget fields are inconsistent")
    if (
        budget_ledger.authority.matrix_run_id != launch.plan.matrix_run_id
        or budget_ledger.authority.phase_cap_cny != phase_cap
        or budget_ledger.authority.prior_observed_cost_cny != prior
        or budget_ledger.authority.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION
    ):
        raise ValueError("create-only budget authority differs from control")
    if budget_ledger.accountable_cost_cny > phase_cap:
        raise ValueError("execution phase exceeded its cumulative cap")


def _ledger_event_sha256(event: object) -> str:
    kind = getattr(event, "kind", None)
    if kind == "portfolio-budget-reservation":
        value = getattr(event, "reservation_sha256", None)
    elif kind == "portfolio-budget-settlement":
        value = getattr(event, "settlement_sha256", None)
    elif kind == "portfolio-budget-forfeit":
        value = getattr(event, "forfeit_sha256", None)
    else:
        raise ValueError("Portfolio budget ledger contains an unknown event kind")
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("Portfolio budget ledger event lacks a valid self hash")
    return value


def _ledger_prefix_snapshot(
    budget_ledger: PortfolioBudgetLedgerState,
    *,
    last_event_index: int,
) -> dict[str, object]:
    """Replay an already validated ledger state through one immutable prefix."""

    if type(last_event_index) is not int or not (
        0 <= last_event_index <= budget_ledger.last_event_index
    ):
        raise ValueError("shard summary budget event index is outside the ledger")
    events = sorted(
        (
            *budget_ledger.reservations,
            *budget_ledger.settlements,
            *getattr(budget_ledger, "forfeits", ()),
        ),
        key=lambda item: item.ledger_event_index,
    )
    if len(events) != budget_ledger.last_event_index or any(
        event.ledger_event_index != expected
        for expected, event in enumerate(events, start=1)
    ):
        raise ValueError("validated budget ledger event inventory is inconsistent")
    for event in events:
        _ledger_event_sha256(event)
    prefix = tuple(events[:last_event_index])
    reservations = {
        event.reservation_sha256: event
        for event in prefix
        if event.kind == "portfolio-budget-reservation"
    }
    settlements = {
        event.reservation_sha256: event
        for event in prefix
        if event.kind == "portfolio-budget-settlement"
    }
    forfeits = {
        event.reservation_sha256: event
        for event in prefix
        if event.kind == "portfolio-budget-forfeit"
    }
    if (
        not set(settlements) <= set(reservations)
        or not set(forfeits) <= set(reservations)
        or set(settlements) & set(forfeits)
    ):
        raise ValueError("shard summary budget prefix has invalid terminal events")
    settled = sum(
        (event.actual_cost_cny for event in settlements.values()),
        start=Decimal("0.000000000000"),
    )
    forfeited = sum(
        (event.forfeited_cost_cny for event in forfeits.values()),
        start=Decimal("0.000000000000"),
    )
    unresolved = sum(
        (
            event.reserved_cost_cny
            for digest, event in reservations.items()
            if digest not in settlements and digest not in forfeits
        ),
        start=Decimal("0.000000000000"),
    )
    accountable = (
        budget_ledger.authority.prior_observed_cost_cny
        + settled
        + forfeited
        + unresolved
    )
    return {
        "budget_authority_sha256": budget_ledger.authority.authority_sha256,
        "budget_ledger_last_event_index": last_event_index,
        "budget_ledger_last_event_sha256": (
            None if not prefix else _ledger_event_sha256(prefix[-1])
        ),
        "budget_ledger_settled_actual_cost_cny": format(settled, ".12f"),
        "budget_ledger_forfeited_reserved_cost_cny": format(forfeited, ".12f"),
        "budget_ledger_unresolved_reserved_cost_cny": format(unresolved, ".12f"),
        "budget_ledger_accountable_cost_cny": format(accountable, ".12f"),
        "observed_dashscope_cost_cny": float(settled),
        "prior_dashscope_observed_cost_cny": float(
            budget_ledger.authority.prior_observed_cost_cny
        ),
        "cumulative_dashscope_cost_cny": float(
            budget_ledger.authority.prior_observed_cost_cny + settled
        ),
    }


def _summary_number_matches(value: object, expected: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        observed = Decimal(str(value))
        target = Decimal(str(expected))
    except InvalidOperation:
        return False
    return observed.is_finite() and target.is_finite() and observed == target


def _prepare_current_shard_summary(
    source_summary: dict,
    *,
    shard_id: str,
    config: str,
    query_count: int,
    budget_ledger: PortfolioBudgetLedgerState,
) -> tuple[dict, bool]:
    """Validate a create-only summary prefix and project it to the ledger tail.

    Concurrent workers legitimately finish against different append-only ledger
    prefixes.  Only the global ledger snapshot fields may advance here; the
    shard identity, completion claim, elapsed time, and every other field remain
    byte-for-value bound by the source summary's self hash.
    """

    supplied = source_summary.get("summary_sha256")
    unsigned = dict(source_summary)
    unsigned.pop("summary_sha256", None)
    if (
        supplied != _hash(unsigned)
        or source_summary.get("status") != "complete"
        or source_summary.get("query_count") != query_count
        or source_summary.get("shard_id") != shard_id
        or source_summary.get("config") != config
    ):
        raise ValueError("shard summary is not a complete self-consistent result")

    prefix = _ledger_prefix_snapshot(
        budget_ledger,
        last_event_index=source_summary.get("budget_ledger_last_event_index"),
    )
    decimal_fields = {
        "observed_dashscope_cost_cny",
        "prior_dashscope_observed_cost_cny",
        "cumulative_dashscope_cost_cny",
    }
    for field, expected in prefix.items():
        matches = (
            _summary_number_matches(source_summary.get(field), expected)
            if field in decimal_fields
            else source_summary.get(field) == expected
        )
        if not matches:
            raise ValueError(
                f"shard summary is not an authentic budget-ledger prefix: {field}"
            )

    current = _ledger_prefix_snapshot(
        budget_ledger,
        last_event_index=budget_ledger.last_event_index,
    )
    refreshed_unsigned = dict(unsigned)
    refreshed_unsigned.update(current)
    refreshed = {
        **refreshed_unsigned,
        "summary_sha256": _hash(refreshed_unsigned),
    }
    return refreshed, refreshed != source_summary


def _budget_ledger_tail_identity(
    budget_ledger: PortfolioBudgetLedgerState,
) -> tuple[object, ...]:
    return (
        budget_ledger.authority.authority_sha256,
        budget_ledger.last_event_index,
        budget_ledger.last_event_sha256,
        budget_ledger.settled_actual_cost_cny,
        getattr(
            budget_ledger,
            "forfeited_reserved_cost_cny",
            Decimal("0.000000000000"),
        ),
        budget_ledger.unresolved_reserved_cost_cny,
        budget_ledger.accountable_cost_cny,
    )


def _locked_final_result_contract(runtime_lock: dict) -> tuple[int, str]:
    """Return the receipt schema/namespace, rejecting partial new contracts."""

    schema_version = runtime_lock.get("final_judge_result_schema_version")
    cache_namespace = runtime_lock.get("final_judge_cache_namespace")
    if schema_version == 4 or cache_namespace == "final-evaluator-v5":
        if (
            schema_version != 4
            or cache_namespace != "final-evaluator-v5"
            or runtime_lock.get("card_requirement_guard_policy_version")
            != CARD_REQUIREMENT_GUARD_POLICY_VERSION
            or runtime_lock.get("card_requirement_guard_policy_sha256")
            != CARD_REQUIREMENT_GUARD_POLICY_SHA256
            or runtime_lock.get("final_judge_parser_policy_version")
            != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
            or runtime_lock.get("final_judge_parser_policy_sha256")
            != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
        ):
            raise ValueError("runtime lock has a partial historical schema-v4 contract")
        return 4, "final-evaluator-v5"

    if schema_version == 6 or cache_namespace == "final-evaluator-v7":
        expected_historical = (
            _HISTORICAL_FINAL_RESULT_V6_LOCK
            | _ACTIVE_EVALUATOR_SOURCE_LOCK
            | _ACTIVE_BUDGET_CONTRACT
        )
        if any(
            runtime_lock.get(field) != expected
            for field, expected in expected_historical.items()
        ) or (
            runtime_lock.get("final_judge_parser_policy_version")
            != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
            or runtime_lock.get("final_judge_parser_policy_sha256")
            != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
        ):
            raise ValueError("runtime lock has a partial historical schema-v6 contract")
        return 6, "final-evaluator-v7"

    if schema_version == 7 or cache_namespace == "final-evaluator-v8":
        expected_historical = (
            _HISTORICAL_FINAL_RESULT_V7_LOCK
            | _ACTIVE_EVALUATOR_SOURCE_LOCK
            | _ACTIVE_BUDGET_CONTRACT
        )
        if any(
            runtime_lock.get(field) != expected
            for field, expected in expected_historical.items()
        ) or (
            runtime_lock.get("final_judge_parser_policy_version")
            != FINAL_JUDGE_PARSER_POLICY_VERSION_V3
            or runtime_lock.get("final_judge_parser_policy_sha256")
            != FINAL_JUDGE_PARSER_POLICY_SHA256_V3
        ):
            raise ValueError("runtime lock has a partial historical schema-v7 contract")
        return 7, "final-evaluator-v8"

    has_new_contract_field = any(
        field in runtime_lock
        for field in (
            *_ACTIVE_FINAL_RESULT_LOCK,
            *_ACTIVE_EVALUATOR_SOURCE_LOCK,
            *_ACTIVE_BUDGET_CONTRACT,
        )
    )
    if has_new_contract_field:
        if any(
            runtime_lock.get(field) != expected
            for field, expected in (
                _ACTIVE_FINAL_RESULT_LOCK
                | _ACTIVE_EVALUATOR_SOURCE_LOCK
                | _ACTIVE_BUDGET_CONTRACT
            ).items()
        ):
            raise ValueError(
                "runtime lock has a partial or inactive final-Judge card-guard/"
                "evaluator-source/hard-budget contract"
            )
        if (
            runtime_lock.get("final_judge_parser_policy_version")
            != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
            or runtime_lock.get("final_judge_parser_policy_sha256")
            != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
        ):
            raise ValueError("current final-Judge runtime does not bind parser v4")
        return FINAL_JUDGE_RESULT_SCHEMA_VERSION, FINAL_JUDGE_CACHE_NAMESPACE

    parser_version = runtime_lock.get("final_judge_parser_policy_version")
    parser_sha256 = runtime_lock.get("final_judge_parser_policy_sha256")
    if (
        parser_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V3
        and parser_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V3
    ):
        return 3, "final-evaluator-v4"
    if (
        parser_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V2
        and parser_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V2
    ):
        return 2, "final-evaluator-v3"
    if parser_version is None and parser_sha256 is None:
        return 1, "final-evaluator-v2"
    raise ValueError("runtime lock has an unknown final-Judge parser contract")


def _should_record_judge_response_receipt(parser_schema_version: int) -> bool:
    """Return whether the final-result schema carries retry/reasoning receipts."""

    return parser_schema_version in {5, 6, 7, 8, 9, 10}


def _require_local_finalize_authorized(control: dict, shard_id: str) -> None:
    """Reject local mutation outside the execution control's exact scope."""

    frozen = control.get("external_frozen_shard")
    if isinstance(frozen, dict) and shard_id == frozen.get("target_shard_id"):
        raise ValueError("external frozen shard must not be finalized locally")
    authorized = tuple(control.get("authorized_shard_ids", ()))
    if authorized and shard_id not in authorized:
        raise ValueError("selected shard is outside the execution authorization")


def _summarize_tool_errors(rows: list[tuple[str, tuple]]) -> dict[str, object]:
    """Project recovered tool failures without hiding them as terminal errors."""

    error_ids: list[str] = []
    by_code: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    error_count = 0
    for query_id, trace in rows:
        query_has_error = False
        for item in trace:
            if item.status == "success":
                continue
            if item.error_code is None:
                raise ValueError("failed tool trace lacks an error code")
            query_has_error = True
            error_count += 1
            by_code[item.error_code] += 1
            by_tool[item.tool_name] += 1
        if query_has_error:
            error_ids.append(query_id)
    return {
        "tool_error_count": error_count,
        "tool_error_query_count": len(error_ids),
        "tool_error_ids": error_ids,
        "tool_error_counts_by_code": dict(sorted(by_code.items())),
        "tool_error_counts_by_tool": dict(sorted(by_tool.items())),
    }


def _summarize_card_policy(
    rows: list[tuple[str, bool, int]],
) -> dict[str, object]:
    """Report the required/forbidden visible-card contract independently of J."""

    required_missing: list[str] = []
    forbidden_present: list[str] = []
    for query_id, requires_card, card_count in rows:
        if card_count < 0:
            raise ValueError("visible card count cannot be negative")
        if requires_card and card_count == 0:
            required_missing.append(query_id)
        elif not requires_card and card_count != 0:
            forbidden_present.append(query_id)
    violation_ids = required_missing + forbidden_present
    return {
        "card_policy_compliant_count": len(rows) - len(violation_ids),
        "card_policy_violation_count": len(violation_ids),
        "card_policy_violation_ids": violation_ids,
        "required_missing_card_ids": required_missing,
        "forbidden_card_ids": forbidden_present,
    }


def _load_launch_card_requirements(
    launch,
    *,
    query_artifact_path: Path | None = None,
    legacy_launch_compat: bool = False,
) -> dict[str, bool]:
    """Replay the frozen private Query binding without loading all active inputs."""

    core_input_files = getattr(launch.plan, "core_input_files", None)
    if core_input_files is None:
        resolved_query_artifact_path = (
            _QUERY_ARTIFACT_PATH
            if query_artifact_path is None
            else Path(query_artifact_path)
        )
    else:
        bound_query_artifact_path = Path(core_input_files.query_artifact_path)
        if (
            core_input_files.expected_query_artifact_sha256
            != launch.plan.query_artifact_sha256
        ):
            raise ValueError("Core query artifact binding differs from launch plan")
        if (
            query_artifact_path is not None
            and Path(query_artifact_path) != bound_query_artifact_path
        ):
            raise ValueError("Core query artifact path differs from launch binding")
        resolved_query_artifact_path = bound_query_artifact_path

    content = read_stable_regular_file(
        resolved_query_artifact_path,
        label="Portfolio query artifact",
        max_bytes=64 * 1024 * 1024,
    )
    if sha256_bytes(content) != launch.plan.query_artifact_sha256:
        raise ValueError("Portfolio query artifact SHA-256 differs from launch plan")
    rows = parse_canonical_jsonl(content, label="Portfolio query artifact")
    queries: dict[str, Query] = {}
    for line_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Portfolio query artifact line {line_number} is not an object"
            )
        query = Query.model_validate(raw, strict=True)
        if query.query_id in queries:
            raise ValueError(f"duplicate Portfolio query_id: {query.query_id}")
        if type(query.requires_card) is not bool:
            raise ValueError(f"query lacks a frozen card requirement: {query.query_id}")
        queries[query.query_id] = query

    launch_query_ids = {item.query_id for item in launch.instances}
    if core_input_files is None:
        if launch_query_ids != set(queries):
            raise ValueError("Portfolio query artifact coverage differs from launch")
    else:
        source_query_count = getattr(launch.plan, "source_query_count", None)
        launch_query_count = getattr(launch.plan, "query_count", None)
        if (
            type(source_query_count) is not int
            or source_query_count <= 0
            or len(queries) != source_query_count
        ):
            raise ValueError("Core query artifact count differs from launch source")
        if (
            type(launch_query_count) is not int
            or launch_query_count <= 0
            or len(launch_query_ids) != launch_query_count
        ):
            raise ValueError("Core selected query count differs from launch")
        if not launch_query_ids <= set(queries):
            raise ValueError("Core query artifact lacks selected launch queries")
        selected_source_ids = {
            query_id
            for query_id, query in queries.items()
            if query.split in launch.plan.selected_splits
        }
        if launch_query_ids != selected_source_ids:
            raise ValueError("Core selected split coverage differs from launch")

    public_inputs = {}
    for member in launch.instances:
        query = queries[member.query_id]
        public = public_inputs.get(member.query_id)
        if public is None:
            public = (
                build_legacy_assistant_query_input(query)
                if legacy_launch_compat
                else build_assistant_query_input(query)
            )
            public_inputs[member.query_id] = public
        common_binding_drifted = (
            query.query_id != member.query_id
            or public.query_sha256 != member.query_sha256
            or public.public_input_sha256 != member.public_input_sha256
        )
        if core_input_files is None:
            profile_binding_drifted = (
                query.asset_id != member.asset_id
                or query.image_path != member.image_path
                or query.synthesis_batch_id != member.accepted_batch_id
            )
        else:
            asset_binding = public.asset_binding
            profile_binding_drifted = (
                query.split not in launch.plan.selected_splits
                or query.generator_batch_id != member.accepted_batch_id
                or asset_binding is None
                or asset_binding.asset_token != member.asset_token
                or asset_binding.binding_sha256 != member.asset_binding_sha256
            )
        if common_binding_drifted or profile_binding_drifted:
            raise ValueError(
                f"Portfolio query binding differs from launch: {member.query_id}"
            )
    return {
        query_id: queries[query_id].requires_card
        for query_id in sorted(launch_query_ids)
    }


def _artifact_set_sha256s(
    shard_root: Path,
    *,
    expected: dict[str, str],
) -> str:
    if len(expected) != 52 or any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        for name, digest in expected.items()
    ):
        raise ValueError("external frozen shard artifact set is malformed")
    actual_names = {
        path.relative_to(shard_root).as_posix()
        for path in shard_root.rglob("*")
        if path.is_file()
    }
    if actual_names != set(expected):
        raise ValueError("external frozen shard contains missing or extra files")
    for relative, expected_sha256 in expected.items():
        content = read_stable_regular_file(
            shard_root / relative,
            label=f"external frozen artifact {relative}",
            max_bytes=64 * 1024 * 1024,
        )
        if sha256_bytes(content) != expected_sha256:
            raise ValueError(
                f"external frozen shard artifact digest mismatch: {relative}"
            )
    return _hash(dict(sorted(expected.items())))


def _validate_external_frozen_target(
    *,
    control: dict,
    launch,
    runtime_lock: dict,
    execution_root: Path,
) -> str | None:
    """Re-verify the immutable external NoSkill evidence before state credit.

    The preparation command deeply validates every checkpoint.  Finalization
    repeats all immutable bindings and the exact 52-file digest set, so a
    later source mutation cannot be silently credited in the target launch.
    """

    descriptor = control.get("external_frozen_shard")
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("external_frozen_shard must be one object")
    _require_self_hash(
        descriptor,
        "binding_sha256",
        label="external frozen shard binding",
    )

    source_root = Path(str(descriptor.get("source_execution_root", "")))
    control_path = source_root / "execution-control.json"
    control_bytes = read_stable_regular_file(
        control_path,
        label="external source execution control",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(control_bytes) != descriptor.get(
        "source_execution_control_file_sha256"
    ):
        raise ValueError("external source execution-control file digest mismatch")
    source_control = _load_canonical_object(
        control_path,
        label="external source execution control",
    )
    source_control_sha256 = _require_self_hash(
        source_control,
        "control_sha256",
        label="external source execution control",
    )
    if source_control_sha256 != descriptor.get("source_execution_control_sha256"):
        raise ValueError("external source execution-control self digest mismatch")

    source_launch = load_portfolio_launch_package(
        source_control["launch_root"],
        expected_plan_file_sha256=descriptor["source_launch_plan_file_sha256"],
    )
    if (
        source_control.get("launch_plan_file_sha256")
        != descriptor.get("source_launch_plan_file_sha256")
        or Path(source_control["launch_root"]).resolve()
        != Path(str(descriptor.get("source_launch_root", ""))).resolve()
        or source_launch.plan.launch_plan_sha256
        != descriptor.get("source_launch_plan_sha256")
    ):
        raise ValueError("external source launch binding mismatch")

    source_runtime_root = Path(source_control["runtime_root"])
    source_runtime_path = source_runtime_root / "runtime-lock.json"
    source_runtime_bytes = read_stable_regular_file(
        source_runtime_path,
        label="external source runtime lock",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(source_runtime_bytes) != descriptor.get(
        "source_runtime_lock_file_sha256"
    ):
        raise ValueError("external source runtime-lock file digest mismatch")
    source_runtime = _load_canonical_object(
        source_runtime_path,
        label="external source runtime lock",
    )
    source_runtime_sha256 = _require_self_hash(
        source_runtime,
        "runtime_lock_sha256",
        label="external source runtime lock",
    )
    if (
        source_runtime_sha256 != descriptor.get("source_runtime_lock_sha256")
        or source_runtime_root.resolve()
        != Path(str(descriptor.get("source_runtime_root", ""))).resolve()
        or source_control.get("runtime_lock_sha256") != source_runtime_sha256
        or runtime_lock.get("compatible_frozen_noskill_runtime_lock_sha256")
        != source_runtime_sha256
    ):
        raise ValueError("external source/target runtime compatibility mismatch")
    source_key = read_stable_regular_file(
        source_root / "blinding-key.bin",
        label="external source blinding key",
        max_bytes=32,
    )
    if (
        len(source_key) != 32
        or sha256_bytes(source_key) != source_control.get("blinding_key_sha256")
        or sha256_bytes(source_key) != descriptor.get("blinding_key_sha256")
    ):
        raise ValueError("external source blinding key mismatch")

    try:
        source_shard = next(
            item
            for item in source_launch.plan.shards
            if item.shard_id == descriptor.get("source_shard_id")
        )
        target_shard = next(
            item
            for item in launch.plan.shards
            if item.shard_id == descriptor.get("target_shard_id")
        )
    except StopIteration as error:
        raise ValueError("external frozen source or target shard is absent") from error
    if (
        source_shard.config != "noskill"
        or target_shard.config != "noskill"
        or source_shard.shard_sha256 != descriptor.get("source_shard_sha256")
        or target_shard.shard_sha256 != descriptor.get("target_shard_sha256")
        or source_shard.accepted_batch_id != target_shard.accepted_batch_id
        or source_shard.accepted_batch_id != descriptor.get("accepted_batch_id")
        or source_shard.query_count != 25
        or target_shard.query_count != 25
    ):
        raise ValueError("external frozen source/target shard binding mismatch")

    source_members = tuple(
        item
        for item in source_launch.instances
        if item.shard_id == source_shard.shard_id
    )
    target_members = tuple(
        item for item in launch.instances if item.shard_id == target_shard.shard_id
    )
    compatibility_fields = (
        "query_id",
        "query_sha256",
        "public_input_sha256",
        "asset_id",
        "image_path",
        "image_sha256",
    )
    if len(source_members) != 25 or len(target_members) != 25:
        raise ValueError("external frozen source/target membership is incomplete")
    for source_member, target_member in zip(
        source_members, target_members, strict=True
    ):
        if any(
            getattr(source_member, field) != getattr(target_member, field)
            for field in compatibility_fields
        ):
            raise ValueError(
                f"external frozen query/public/image drift: {source_member.query_id}"
            )
    compatibility_payload = [
        {field: getattr(item, field) for field in compatibility_fields}
        for item in target_members
    ]
    if _hash(compatibility_payload) != descriptor.get(
        "query_public_image_compatibility_sha256"
    ):
        raise ValueError("external frozen compatibility projection mismatch")

    source_shard_root = source_root / source_shard.output_relpath
    summary_path = source_shard_root / "shard-summary.json"
    summary_bytes = read_stable_regular_file(
        summary_path,
        label="external frozen shard summary",
        max_bytes=4 * 1024 * 1024,
    )
    source_summary = _load_canonical_object(
        summary_path,
        label="external frozen shard summary",
    )
    source_summary_sha256 = _require_self_hash(
        source_summary,
        "summary_sha256",
        label="external frozen shard summary",
    )
    audit_path = source_shard_root / "shard-audit.json"
    audit_bytes = read_stable_regular_file(
        audit_path,
        label="external frozen shard audit",
        max_bytes=4 * 1024 * 1024,
    )
    source_audit = _load_canonical_object(
        audit_path,
        label="external frozen shard audit",
    )
    source_audit_sha256 = _require_self_hash(
        source_audit,
        "audit_sha256",
        label="external frozen shard audit",
    )
    if (
        sha256_bytes(summary_bytes)
        != descriptor.get("source_shard_summary_file_sha256")
        or source_summary_sha256 != descriptor.get("source_shard_summary_sha256")
        or sha256_bytes(audit_bytes) != descriptor.get("source_shard_audit_file_sha256")
        or source_audit_sha256 != descriptor.get("source_shard_audit_sha256")
        or source_summary.get("status") != "complete"
        or source_summary.get("shard_id") != source_shard.shard_id
        or source_summary.get("config") != "noskill"
        or source_audit.get("shard_summary_sha256") != source_summary_sha256
        or source_audit.get("runtime_lock_sha256") != source_runtime_sha256
        or source_audit.get("launch_plan_sha256")
        != source_launch.plan.launch_plan_sha256
    ):
        raise ValueError("external frozen summary/audit binding mismatch")

    expected_artifacts = descriptor.get("artifact_file_sha256s")
    if not isinstance(expected_artifacts, dict):
        raise ValueError("external frozen artifact digest map is absent")
    observed_set_sha256 = _artifact_set_sha256s(
        source_shard_root,
        expected=expected_artifacts,
    )
    if descriptor.get(
        "artifact_file_count"
    ) != 52 or observed_set_sha256 != descriptor.get("artifact_set_sha256"):
        raise ValueError("external frozen artifact set digest mismatch")
    comparison_payload = descriptor.get("comparison_contract_payload")
    if (
        not isinstance(comparison_payload, dict)
        or _hash(comparison_payload) != descriptor.get("comparison_contract_sha256")
        or descriptor.get("blinding_key_sha256") != control.get("blinding_key_sha256")
        or execution_root.resolve() == source_root.resolve()
    ):
        raise ValueError("external frozen comparison contract binding mismatch")
    return target_shard.shard_id


def _validate_turn_accounting(response, receipt) -> tuple[int, int, int]:
    """Return shared reserved turns/input/output after exact turn validation."""

    shared = receipt.shared_route_reference
    reserved_turns = 0 if shared is None else shared.reserved_turns
    expected_turns = max(1, len(receipt.model_calls) + reserved_turns)
    if response.turn_count != expected_turns:
        raise ValueError("Assistant turn count differs from actual plus reserved calls")
    if shared is None:
        return 0, 0, 0
    return (
        reserved_turns,
        shared.reserved_usage.input_tokens,
        shared.reserved_usage.output_tokens,
    )


def _shared_route_owner_id(launch, shard) -> str | None:
    if shard.config not in _SHARED_ROUTE_CONFIGS:
        return None
    candidates = tuple(
        item
        for item in launch.plan.shards
        if item.accepted_batch_id == shard.accepted_batch_id
        and item.config in _SHARED_ROUTE_CONFIGS
    )
    if len(candidates) != 2:
        raise ValueError("batch lacks the paired shared-route configurations")
    return min(candidates, key=lambda item: item.shard_ordinal).shard_id


def _launch_completion_status(plan, completed_shard_ids: tuple[str, ...]) -> str:
    all_shard_ids = {item.shard_id for item in plan.shards}
    return (
        "completed"
        if len(completed_shard_ids) == plan.shard_count
        and set(completed_shard_ids) == all_shard_ids
        else "running"
    )


def _usage_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    input_rate: float,
    output_rate: float,
) -> float:
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _attempt_cost_and_calls(shard_root: Path, *, shard_id: str) -> tuple[float, int]:
    observed_cost = 0.0
    model_calls = 0
    for path in sorted((shard_root / "attempt-receipts").glob("*.json")):
        raw = _load_canonical_object(
            path,
            label=f"Portfolio attempt receipt {path.name}",
        )
        receipt = PortfolioAttemptReceipt.model_validate(raw, strict=True)
        if receipt.shard_id != shard_id:
            raise ValueError("Portfolio attempt receipt differs from shard")
        input_rate, output_rate = (
            (6.5, 27.0) if receipt.failure_stage == "final_judge" else (0.15, 1.5)
        )
        observed_cost += _usage_cost(
            receipt.captured_input_tokens,
            receipt.captured_output_tokens,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        model_calls += portfolio_attempt_provider_call_count(receipt)
    return observed_cost, model_calls


def _execution_model_calls(root: Path) -> int:
    """Count only calls made in this execution root, never external history."""

    total = 0
    for path in root.glob("shards/*/assistant/*.json"):
        _, receipt = _assistant_row(path)
        total += len(receipt.model_calls)
    for path in root.glob("shards/*/final/*.json"):
        raw = _load_canonical_object(path, label=f"final result {path.name}")
        if raw.get("kind") == "portfolio-final-fixed-zero":
            continue
        # Legacy v6 persisted two pre-response provider errors as Judge
        # results with no request ID.  They are still real attempted calls;
        # the repaired runner records the equivalent event in an attempt
        # receipt instead of writing a final result.
        result = load_final_judge_evaluation_result(path)
        total += result.attempts
    shared_route_artifact_sha256s: set[str] = set()
    for path in root.glob("shared-routes/*/*.json"):
        route = SharedStage2RouteArtifact.model_validate_json(
            path.read_bytes(),
            strict=True,
        )
        if route.policy_version != SHARED_STAGE2_ROUTE_POLICY_VERSION:
            raise ValueError("shared route uses an inactive policy version")
        shared_route_artifact_sha256s.add(route.artifact_sha256)
    total += len(shared_route_artifact_sha256s)
    for shard_root in root.glob("shards/*"):
        if not shard_root.is_dir():
            continue
        _, attempt_calls = _attempt_cost_and_calls(
            shard_root,
            shard_id=shard_root.name,
        )
        total += attempt_calls
    return total


def _finalize_static_opt_shard(
    *,
    execution_root: Path,
    control: dict,
    launch: object,
    shard: object,
    members: tuple,
    budget_ledger: PortfolioBudgetLedgerState,
) -> tuple[dict, PortfolioLaunchState, bool]:
    """Commit one Assistant+GCS-v2 shard without reading a Judge artifact."""

    if (
        control.get("execution_scope") != "static_opt_rollout"
        or control.get("evaluation_stages") != ["assistant", "gcs_v2"]
        or control.get("assistant_checkpoint_schema_version") != 2
        or control.get("gcs_policy_version") != GCS_V2_POLICY_VERSION
        or control.get("gcs_policy_sha256") != GCS_V2_POLICY_SHA256
        or control.get("gcs_scorer_evidence_policy_version")
        != GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
        or control.get("pairwise_judge_enabled") is not False
        or control.get("legacy_final_judge_enabled") is not False
        or control.get("analyzer_provider_call_count") != 0
        or launch.plan.kind != "portfolio-core-static-opt-800x1-launch-plan"
        or launch.plan.execution_mode != "static_opt_rollout"
        or launch.plan.selected_splits != ("opt_pool",)
        or launch.plan.config_order != ("llm_static",)
        or shard.config != "llm_static"
        or len(members) != 25
    ):
        raise ValueError("static opt shard does not bind the exact GCS v2 scope")
    shard_root = execution_root / shard.output_relpath
    final_root = shard_root / "final"
    if final_root.exists() and any(final_root.iterdir()):
        raise ValueError("static opt shard contains a forbidden Final Judge artifact")
    if any(execution_root.glob("shards/*/final/*.json")):
        raise ValueError(
            "static opt execution contains a forbidden Final Judge artifact"
        )
    if any(path.is_file() for path in execution_root.rglob("*pairwise*")):
        raise ValueError("static opt shard contains a forbidden Pairwise artifact")

    summary_path = shard_root / "shard-summary.json"
    source_summary_bytes = read_stable_regular_file(
        summary_path,
        label=f"static opt shard summary {shard.shard_id}",
    )
    source_summary = parse_canonical_json(
        source_summary_bytes,
        label=f"static opt shard summary {shard.shard_id}",
    )
    if not isinstance(source_summary, dict):
        raise ValueError("static opt shard summary must contain an object")
    summary, summary_refresh_required = _prepare_current_shard_summary(
        source_summary,
        shard_id=shard.shard_id,
        config=shard.config,
        query_count=len(members),
        budget_ledger=budget_ledger,
    )
    assistant_error_ids: list[str] = []
    assistant_model_calls = 0
    assistant_input_tokens = 0
    assistant_output_tokens = 0
    tool_call_count = 0
    tool_trace_rows: list[tuple[str, tuple]] = []
    sidecar_hashes: list[str] = []
    for member in members:
        assistant_path = execution_root / member.assistant_output_relpath
        raw = _load_canonical_object(
            assistant_path,
            label=f"schema-v2 Assistant checkpoint {member.query_id}",
        )
        if raw.get("schema_version") != 2:
            raise ValueError("static opt shard contains a legacy Assistant checkpoint")
        response, receipt, evidence = _assistant_row(
            assistant_path,
            require_schema_v2=True,
            include_scorer_evidence=True,
        )
        if evidence is None:  # pragma: no cover - loader enforces this
            raise AssertionError("schema-v2 checkpoint lost scorer evidence")
        request = AssistantRequestSnapshot.model_validate_json(
            canonical_json_bytes(raw.get("request")),
            strict=True,
        )
        result = _assistant_result(launch, request, response)
        _verify_scorer_sidecar_binding(
            evidence,
            launch=launch,
            member=member,
            request=request,
            result=result,
            receipt=receipt,
        )
        if (
            raw.get("instance_sha256") != member.instance_sha256
            or raw.get("query_ordinal") != member.query_ordinal
            or request.query.query_id != member.query_id
            or request.config != "llm_static"
            or receipt.request_sha256 != request.request_sha256
            or receipt.aggregate_usage != response.usage
            or receipt.tool_trace != response.tool_trace
        ):
            raise ValueError(
                f"static opt Assistant checkpoint differs from launch: {member.query_id}"
            )
        if response.error_code is not None:
            assistant_error_ids.append(member.query_id)
        assistant_model_calls += len(receipt.model_calls)
        assistant_input_tokens += response.usage.input_tokens
        assistant_output_tokens += response.usage.output_tokens
        tool_call_count += len(response.tool_trace)
        tool_trace_rows.append((member.query_id, response.tool_trace))
        sidecar_hashes.append(evidence.evidence_sha256)

    if len(set(sidecar_hashes)) != len(members):
        raise ValueError("static opt shard contains duplicate scorer sidecars")
    tool_error_summary = _summarize_tool_errors(tool_trace_rows)
    audit_payload = {
        "schema_version": 1,
        "kind": "portfolio-static-opt-shard-audit",
        "formal_eligible": False,
        "execution_scope": "static_opt_rollout",
        "shard_id": shard.shard_id,
        "config": "llm_static",
        "query_count": len(members),
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": control["gcs_policy_version"],
        "gcs_policy_sha256": control["gcs_policy_sha256"],
        "gcs_scorer_evidence_policy_version": control[
            "gcs_scorer_evidence_policy_version"
        ],
        "scorer_sidecar_count": len(sidecar_hashes),
        "scorer_sidecar_sha256s": sidecar_hashes,
        "assistant_success_count": len(members) - len(assistant_error_ids),
        "assistant_error_count": len(assistant_error_ids),
        "assistant_error_ids": assistant_error_ids,
        "assistant_model_calls": assistant_model_calls,
        "assistant_input_tokens": assistant_input_tokens,
        "assistant_output_tokens": assistant_output_tokens,
        "tool_call_count": tool_call_count,
        **tool_error_summary,
        "final_judge_invoked_count": 0,
        "pairwise_judge_invoked_count": 0,
        "final_artifact_count": 0,
        "analyzer_provider_call_count": 0,
        "runtime_lock_sha256": control["runtime_lock_sha256"],
        "launch_plan_sha256": control["launch_plan_sha256"],
        "shard_summary_sha256": summary["summary_sha256"],
        **_ACTIVE_BUDGET_CONTRACT,
        **_ledger_prefix_snapshot(
            budget_ledger,
            last_event_index=budget_ledger.last_event_index,
        ),
    }
    audit = {**audit_payload, "audit_sha256": _hash(audit_payload)}
    audit_bytes = canonical_json_bytes(audit)
    audit_path = shard_root / "shard-audit.json"
    latest_budget_ledger = load_portfolio_budget_ledger(
        _budget_ledger_root(execution_root, control)
    )
    if _budget_ledger_tail_identity(
        latest_budget_ledger
    ) != _budget_ledger_tail_identity(budget_ledger):
        raise ValueError("budget ledger changed during static shard finalization")
    if read_stable_regular_file(summary_path, label="static opt shard summary") != (
        source_summary_bytes
    ):
        raise ValueError("static opt shard summary changed during finalization")
    if summary_refresh_required:
        if audit_path.exists():
            raise ValueError("stale static summary already has a bound audit")
        atomic_replace_file(summary_path, canonical_json_bytes(summary))
    if audit_path.exists():
        if audit_path.read_bytes() != audit_bytes:
            raise ValueError("existing static shard audit differs from checkpoints")
    else:
        atomic_create_file(audit_path, audit_bytes)

    completed = set(launch.state.completed_shard_ids)
    completed.add(shard.shard_id)
    ordered_completed = tuple(
        item.shard_id for item in launch.plan.shards if item.shard_id in completed
    )
    state = PortfolioLaunchState(
        matrix_run_id=launch.plan.matrix_run_id,
        launch_plan_sha256=launch.plan.launch_plan_sha256,
        status=_launch_completion_status(launch.plan, ordered_completed),
        completed_shard_ids=ordered_completed,
        failed_shard_ids=launch.state.failed_shard_ids,
        model_calls_performed=_execution_model_calls(execution_root),
        dashscope_observed_cost_cny=float(
            portfolio_budget_settled_cost_cny(budget_ledger)
        ),
        aifast_observed_cost_cny=launch.state.aifast_observed_cost_cny,
    )
    atomic_replace_file(
        launch.root / "run-state.json",
        canonical_json_bytes(state.model_dump(mode="json")),
    )
    return audit, state, summary_refresh_required


def _validate_fixed_zero(
    path: Path,
    *,
    query_id: str,
    assistant_error_code: str,
) -> None:
    data = _load_canonical_object(path, label=f"fixed-zero result {query_id}")
    supplied = data.get("result_sha256")
    unsigned = dict(data)
    unsigned.pop("result_sha256", None)
    if (
        supplied != _hash(unsigned)
        or data.get("kind") != "portfolio-final-fixed-zero"
        or data.get("query_id") != query_id
        or data.get("assistant_error_code") != assistant_error_code
        or data.get("j_project") != 0.0
    ):
        raise ValueError(
            f"fixed-zero result differs from Assistant failure: {query_id}"
        )


def _validated_fixed_zero_error(
    path: Path,
    *,
    query_id: str,
    response_error_code: str | None,
) -> str | None:
    """Return the only terminal error represented by a valid fixed-zero row."""

    if response_error_code is not None:
        _validate_fixed_zero(
            path,
            query_id=query_id,
            assistant_error_code=response_error_code,
        )
        return response_error_code
    data = _load_canonical_object(path, label=f"final result {query_id}")
    if data.get("kind") != "portfolio-final-fixed-zero":
        return None
    packet_error_code = data.get("assistant_error_code")
    if packet_error_code != "hidden_evaluation_identity":
        raise ValueError(
            f"successful Assistant has an unsupported fixed-zero result: {query_id}"
        )
    _validate_fixed_zero(
        path,
        query_id=query_id,
        assistant_error_code=packet_error_code,
    )
    return packet_error_code


def _execution_artifact_alias(
    control: dict,
    runtime_lock: dict,
    *,
    target_config: str,
    source_config: str,
) -> dict:
    aliases = control.get("execution_artifact_aliases")
    if not isinstance(aliases, list) or any(
        not isinstance(item, dict) for item in aliases
    ):
        raise ValueError("execution control lacks typed artifact aliases")
    matches = tuple(
        item for item in aliases if item.get("target_config") == target_config
    )
    if len(matches) != 1:
        raise ValueError("target shard lacks one exact execution artifact alias")
    alias = matches[0]
    if (
        runtime_lock.get("execution_artifact_aliases") != aliases
        or runtime_lock.get("execution_artifact_alias_policy_version")
        != control.get("execution_artifact_alias_policy_version")
        or alias.get("source_config") != source_config
        or alias.get("provider_model_call_count") != 0
        or alias.get("reuse_scope") != "assistant_and_evaluator_query_artifacts"
        or alias.get("source_bank_sha256") != alias.get("target_bank_sha256")
        or alias.get("source_bank_file_sha256") != alias.get("target_bank_file_sha256")
    ):
        raise ValueError("execution artifact alias differs from the runtime lock")
    return alias


def _finalize_execution_artifact_alias(
    *,
    execution_root: Path,
    control: dict,
    runtime_lock: dict,
    launch: object,
    budget_ledger: PortfolioBudgetLedgerState,
    target_shard: object,
    source_shard_id: str,
) -> tuple[dict, PortfolioLaunchState]:
    source_shard = next(
        item for item in launch.plan.shards if item.shard_id == source_shard_id
    )
    alias = _execution_artifact_alias(
        control,
        runtime_lock,
        target_config=str(target_shard.config),
        source_config=str(source_shard.config),
    )
    if (
        source_shard.accepted_batch_id != target_shard.accepted_batch_id
        or source_shard.query_ids != target_shard.query_ids
        or source_shard.shard_id not in launch.state.completed_shard_ids
        or target_shard.shard_id in launch.state.completed_shard_ids
    ):
        raise ValueError(
            "artifact alias source is not the completed paired source shard"
        )

    source_root = execution_root / source_shard.output_relpath
    summary_path = source_root / "shard-summary.json"
    audit_path = source_root / "shard-audit.json"
    summary_bytes = read_stable_regular_file(
        summary_path,
        label=f"artifact alias source summary {source_shard.shard_id}",
    )
    audit_bytes = read_stable_regular_file(
        audit_path,
        label=f"artifact alias source audit {source_shard.shard_id}",
    )
    summary = parse_canonical_json(summary_bytes, label="artifact alias source summary")
    audit = parse_canonical_json(audit_bytes, label="artifact alias source audit")
    if not isinstance(summary, dict) or not isinstance(audit, dict):
        raise ValueError("artifact alias source summary/audit is not an object")
    source_summary_sha256 = _require_self_hash(
        summary,
        "summary_sha256",
        label="artifact alias source summary",
    )
    source_audit_sha256 = _require_self_hash(
        audit,
        "audit_sha256",
        label="artifact alias source audit",
    )
    if (
        summary.get("status") != "complete"
        or summary.get("shard_id") != source_shard.shard_id
        or summary.get("config") != source_shard.config
        or audit.get("kind") != "portfolio-shard-audit"
        or audit.get("shard_id") != source_shard.shard_id
        or audit.get("config") != source_shard.config
        or audit.get("shard_summary_sha256") != source_summary_sha256
        or audit.get("runtime_lock_sha256") != control["runtime_lock_sha256"]
        or audit.get("launch_plan_sha256") != control["launch_plan_sha256"]
    ):
        raise ValueError("artifact alias source is not a verified complete shard")

    calls_before = _execution_model_calls(execution_root)
    ledger_before = _budget_ledger_tail_identity(budget_ledger)
    payload = {
        "schema_version": 1,
        "kind": "portfolio-shard-artifact-alias",
        "policy_version": control["execution_artifact_alias_policy_version"],
        "matrix_run_id": launch.plan.matrix_run_id,
        "accepted_batch_id": target_shard.accepted_batch_id,
        "target_shard_id": target_shard.shard_id,
        "target_config": target_shard.config,
        "source_shard_id": source_shard.shard_id,
        "source_config": source_shard.config,
        "query_ids": list(target_shard.query_ids),
        "treatment_alias_sha256": alias["alias_sha256"],
        "source_bank_sha256": alias["source_bank_sha256"],
        "target_bank_sha256": alias["target_bank_sha256"],
        "source_shard_summary_file_sha256": sha256_bytes(summary_bytes),
        "source_shard_summary_sha256": source_summary_sha256,
        "source_shard_audit_file_sha256": sha256_bytes(audit_bytes),
        "source_shard_audit_sha256": source_audit_sha256,
        "reuse_scope": "assistant_and_evaluator_query_artifacts",
        "provider_model_call_count": 0,
        "rejected_candidate_use": "diagnostic_only",
        "runtime_lock_sha256": control["runtime_lock_sha256"],
        "launch_plan_sha256": control["launch_plan_sha256"],
    }
    receipt = {**payload, "alias_receipt_sha256": _hash(payload)}
    receipt_bytes = canonical_json_bytes(receipt)
    target_root = execution_root / target_shard.output_relpath
    target_root.parent.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(exist_ok=True)
    if target_root.is_symlink() or any(
        item.name != "artifact-alias.json" for item in target_root.iterdir()
    ):
        raise ValueError("artifact alias target directory contains unexpected files")
    receipt_path = target_root / "artifact-alias.json"
    if receipt_path.exists():
        if (
            read_stable_regular_file(
                receipt_path,
                label=f"artifact alias receipt {target_shard.shard_id}",
            )
            != receipt_bytes
        ):
            raise ValueError("existing artifact alias receipt differs")
    else:
        atomic_create_file(receipt_path, receipt_bytes)

    latest_ledger = load_portfolio_budget_ledger(
        _budget_ledger_root(execution_root, control)
    )
    calls_after = _execution_model_calls(execution_root)
    if (
        calls_after != calls_before
        or _budget_ledger_tail_identity(latest_ledger) != ledger_before
    ):
        raise ValueError("artifact alias finalization changed provider-call evidence")
    completed = set(launch.state.completed_shard_ids)
    completed.add(target_shard.shard_id)
    ordered_completed = tuple(
        item.shard_id for item in launch.plan.shards if item.shard_id in completed
    )
    state = PortfolioLaunchState(
        matrix_run_id=launch.plan.matrix_run_id,
        launch_plan_sha256=launch.plan.launch_plan_sha256,
        status=_launch_completion_status(launch.plan, ordered_completed),
        completed_shard_ids=ordered_completed,
        failed_shard_ids=launch.state.failed_shard_ids,
        model_calls_performed=calls_after,
        dashscope_observed_cost_cny=float(
            portfolio_budget_settled_cost_cny(latest_ledger)
        ),
        aifast_observed_cost_cny=launch.state.aifast_observed_cost_cny,
    )
    atomic_replace_file(
        launch.root / "run-state.json",
        canonical_json_bytes(state.model_dump(mode="json")),
    )
    return receipt, state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        control = _load_control(args.execution_root)
        runtime_lock = _load_canonical_object(
            Path(control["runtime_root"]) / "runtime-lock.json",
            label="Portfolio runtime lock",
        )
        launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        static_opt = control.get("execution_scope") == "static_opt_rollout"
        if static_opt:
            verified_static_runtime = load_verified_portfolio_static_opt_runtime(
                control["runtime_root"],
                expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
            )
            if dict(verified_static_runtime.runtime_lock) != runtime_lock:
                raise ValueError(
                    "finalizer runtime differs from verified Static opt runtime"
                )
            validate_portfolio_static_opt_execution_control(
                control,
                verified_static_runtime,
            )
            locked_parser_version = None
            locked_parser_sha256 = None
            parser_schema_version = None
            final_cache_namespace = None
        else:
            locked_parser_version = runtime_lock.get(
                "final_judge_parser_policy_version"
            )
            locked_parser_sha256 = runtime_lock.get("final_judge_parser_policy_sha256")
            parser_schema_version, final_cache_namespace = (
                _locked_final_result_contract(runtime_lock)
            )
        if any(
            control.get(field) != expected or runtime_lock.get(field) != expected
            for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
        ):
            raise ValueError("execution/runtime hard-budget contract drifted")
        if (
            launch.plan.budget.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION
            or launch.plan.budget.policy_sha256 != PORTFOLIO_BUDGET_POLICY_SHA256
            or launch.plan.budget.provider_pricing_contract_version
            != PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            or launch.plan.budget.provider_pricing_contract_sha256
            != PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ):
            raise ValueError("launch hard-budget contract drifted")
        budget_ledger = load_portfolio_budget_ledger(
            _budget_ledger_root(args.execution_root, control)
        )
        _validate_budget_authority(
            control=control,
            launch=launch,
            budget_ledger=budget_ledger,
        )
        shard = next(
            item for item in launch.plan.shards if item.shard_id == args.shard_id
        )
        _require_local_finalize_authorized(control, shard.shard_id)
        members = tuple(
            item for item in launch.instances if item.shard_id == shard.shard_id
        )
        if len(members) != 25:
            raise ValueError("selected shard does not contain exactly 25 instances")
        if static_opt:
            if args.artifact_alias_source_shard_id is not None:
                raise ValueError("static opt rollout forbids artifact aliases")
            if args.legacy_launch_compat:
                raise ValueError(
                    "static opt rollout forbids legacy launch compatibility"
                )
            audit, state, summary_refresh_required = _finalize_static_opt_shard(
                execution_root=args.execution_root,
                control=control,
                launch=launch,
                shard=shard,
                members=members,
                budget_ledger=budget_ledger,
            )
            verified_launch = load_portfolio_launch_package(
                control["launch_root"],
                expected_plan_file_sha256=control["launch_plan_file_sha256"],
            )
            if verified_launch.state != state:
                raise ValueError("static opt launch state failed verification")
            print(
                json.dumps(
                    {
                        "audit_sha256": audit["audit_sha256"],
                        "assistant_error_count": audit["assistant_error_count"],
                        "final_judge_invoked_count": 0,
                        "model_calls_performed": state.model_calls_performed,
                        "pairwise_judge_invoked_count": 0,
                        "scorer_sidecar_count": audit["scorer_sidecar_count"],
                        "shard_id": shard.shard_id,
                        "status": state.status,
                        "summary_refreshed": summary_refresh_required,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        card_requirements = _load_launch_card_requirements(
            launch,
            legacy_launch_compat=args.legacy_launch_compat,
        )
        external_frozen_target_id = _validate_external_frozen_target(
            control=control,
            launch=launch,
            runtime_lock=runtime_lock,
            execution_root=args.execution_root,
        )
        if args.artifact_alias_source_shard_id is not None:
            receipt, state = _finalize_execution_artifact_alias(
                execution_root=args.execution_root,
                control=control,
                runtime_lock=runtime_lock,
                launch=launch,
                budget_ledger=budget_ledger,
                target_shard=shard,
                source_shard_id=args.artifact_alias_source_shard_id,
            )
            verified_launch = load_portfolio_launch_package(
                control["launch_root"],
                expected_plan_file_sha256=control["launch_plan_file_sha256"],
            )
            if verified_launch.state != state:
                raise ValueError("artifact alias launch state failed verification")
            print(
                json.dumps(
                    {
                        "alias_receipt_sha256": receipt["alias_receipt_sha256"],
                        "launch_status": state.status,
                        "model_calls_performed": state.model_calls_performed,
                        "source_shard_id": receipt["source_shard_id"],
                        "status": "artifact_alias_committed",
                        "target_shard_id": receipt["target_shard_id"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if any(
            runtime_lock.get(field) != expected
            for field, expected in {
                "shared_stage2_route_policy_version": (
                    SHARED_STAGE2_ROUTE_POLICY_VERSION
                ),
                "portfolio_router_contract_version": (
                    PORTFOLIO_ROUTER_CONTRACT_VERSION
                ),
                "portfolio_router_contract_sha256": (PORTFOLIO_ROUTER_CONTRACT_SHA256),
                "portfolio_router_request_max_output_tokens": (
                    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
                ),
                "portfolio_router_pricing_reservation_max_output_tokens": (
                    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
                ),
                "portfolio_failure_policy_version": (PORTFOLIO_FAILURE_POLICY_VERSION),
            }.items()
        ):
            raise ValueError("runtime lock does not bind the active Router contract")
        shared_route_enabled = True

        summary_path = args.execution_root / shard.output_relpath / "shard-summary.json"
        source_summary_bytes = read_stable_regular_file(
            summary_path,
            label=f"shard summary {shard.shard_id}",
        )
        source_summary = parse_canonical_json(
            source_summary_bytes,
            label=f"shard summary {shard.shard_id}",
        )
        if not isinstance(source_summary, dict):
            raise ValueError("shard summary must contain an object")
        summary, summary_refresh_required = _prepare_current_shard_summary(
            source_summary,
            shard_id=shard.shard_id,
            config=shard.config,
            query_count=len(members),
            budget_ledger=budget_ledger,
        )
        supplied_summary_sha256 = summary["summary_sha256"]
        refreshed_summary_bytes = canonical_json_bytes(summary)
        audit_path = args.execution_root / shard.output_relpath / "shard-audit.json"
        if summary_refresh_required and audit_path.exists():
            raise ValueError(
                "stale shard summary already has a bound audit; refusing refresh"
            )

        assistant_error_ids: list[str] = []
        assistant_input_tokens = 0
        assistant_output_tokens = 0
        assistant_model_calls = 0
        assistant_effective_turns = 0
        shared_route_reference_count = 0
        shared_route_reserved_turns = 0
        shared_route_reserved_input_tokens = 0
        shared_route_reserved_output_tokens = 0
        shared_route_artifacts: dict[str, SharedStage2RouteArtifact] = {}
        tool_call_count = 0
        tool_trace_rows: list[tuple[str, tuple]] = []
        card_policy_rows: list[tuple[str, bool, int]] = []
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

        for member in members:
            assistant_path = args.execution_root / member.assistant_output_relpath
            assistant_raw = _load_canonical_object(
                assistant_path,
                label=f"Assistant checkpoint {member.query_id}",
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
                or request.config != shard.config
                or receipt.request_sha256 != request.request_sha256
                or receipt.response_sha256 != _hash(_model(response))
                or receipt.aggregate_usage != response.usage
                or receipt.tool_trace != response.tool_trace
            ):
                raise ValueError(
                    f"Assistant checkpoint differs from launch member: {member.query_id}"
                )
            (
                member_reserved_turns,
                member_reserved_input_tokens,
                member_reserved_output_tokens,
            ) = _validate_turn_accounting(response, receipt)
            if shared_route_enabled and shard.config in _SHARED_ROUTE_CONFIGS:
                reference = receipt.shared_route_reference
                if reference is None:
                    raise ValueError(
                        f"shared-route config lacks route reference: {member.query_id}"
                    )
                route_path = (
                    args.execution_root
                    / "shared-routes"
                    / shard.accepted_batch_id
                    / f"{member.query_id}.json"
                )
                route = SharedStage2RouteArtifact.model_validate_json(
                    route_path.read_bytes(),
                    strict=True,
                )
                if (
                    route.policy_version != SHARED_STAGE2_ROUTE_POLICY_VERSION
                    or route.matrix_run_id != launch.plan.matrix_run_id
                    or route.query_id != member.query_id
                    or route.query_ordinal != member.query_ordinal
                    or route.artifact_sha256 != reference.artifact_sha256
                    or route.route_call.response_sha256
                    != reference.route_call_response_sha256
                    or route.route_call.input_tokens
                    != reference.reserved_usage.input_tokens
                    or route.route_call.output_tokens
                    != reference.reserved_usage.output_tokens
                ):
                    raise ValueError(
                        f"shared route differs from Assistant reference: "
                        f"{member.query_id}"
                    )
                shared_route_artifacts[route.artifact_sha256] = route
            elif receipt.shared_route_reference is not None:
                raise ValueError(
                    f"non-shared config carries route reference: {member.query_id}"
                )
            assistant_input_tokens += response.usage.input_tokens
            assistant_output_tokens += response.usage.output_tokens
            assistant_model_calls += len(receipt.model_calls)
            assistant_effective_turns += response.turn_count
            if receipt.shared_route_reference is not None:
                shared_route_reference_count += 1
                shared_route_reserved_turns += member_reserved_turns
                shared_route_reserved_input_tokens += member_reserved_input_tokens
                shared_route_reserved_output_tokens += member_reserved_output_tokens
            tool_call_count += len(response.tool_trace)
            tool_trace_rows.append((member.query_id, response.tool_trace))
            card_policy_rows.append(
                (
                    member.query_id,
                    card_requirements[member.query_id],
                    len(response.visible_cards),
                )
            )

            final_path = args.execution_root / member.final_output_relpath
            fixed_zero_error = _validated_fixed_zero_error(
                final_path,
                query_id=member.query_id,
                response_error_code=response.error_code,
            )
            if fixed_zero_error is not None:
                assistant_error_ids.append(member.query_id)
                fixed_zero_count += 1
                all_scores.append(0.0)
                continue

            final = load_final_judge_evaluation_result(final_path)
            if (
                final.schema_version != parser_schema_version
                or final.cache_namespace != final_cache_namespace
                or (
                    parser_schema_version > 1
                    and (
                        final.parser_policy_version != locked_parser_version
                        or final.parser_policy_sha256 != locked_parser_sha256
                    )
                )
                or (
                    parser_schema_version in {4, 5, 6, 7, 8, 9, 10}
                    and (
                        final.card_requirement_guard_policy_version
                        != runtime_lock["card_requirement_guard_policy_version"]
                        or final.card_requirement_guard_policy_sha256
                        != runtime_lock["card_requirement_guard_policy_sha256"]
                        or final.visible_card_count != len(response.visible_cards)
                    )
                )
                or (
                    parser_schema_version in {5, 6, 7, 8, 9, 10}
                    and (
                        final.max_attempts != runtime_lock["final_judge_max_attempts"]
                        or final.retry_policy_version
                        != runtime_lock["final_judge_retry_policy_version"]
                        or final.retry_policy_sha256
                        != runtime_lock["final_judge_retry_policy_sha256"]
                    )
                )
                or (
                    parser_schema_version in {6, 7, 8, 9, 10}
                    and (
                        final.thinking_budget
                        != runtime_lock["final_judge_thinking_budget"]
                        or final.max_billable_input_tokens
                        != runtime_lock["final_judge_max_billable_input_tokens"]
                        or final.max_billable_output_tokens
                        != runtime_lock["final_judge_max_billable_output_tokens"]
                    )
                )
                or (
                    parser_schema_version == 10
                    and (
                        final.transport_policy_version
                        != runtime_lock["final_judge_transport_policy_version"]
                        or final.transport_policy_sha256
                        != runtime_lock["final_judge_transport_policy_sha256"]
                        or final.requested_response_format
                        != runtime_lock["final_judge_requested_response_format"]
                    )
                )
            ):
                raise ValueError(
                    f"final result parser/guard policy differs from runtime: "
                    f"{member.query_id}"
                )
            if final.card_requirement_guard_adjusted:
                judge_card_guard_adjusted_count += 1
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
            reasoning_receipts = tuple(
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
            )
            for (
                reasoning_present,
                reasoning_tokens,
                reasoning_bytes,
            ) in reasoning_receipts:
                if reasoning_present:
                    judge_reasoning_present_response_count += 1
                if reasoning_tokens is None:
                    judge_reasoning_tokens_unavailable_response_count += 1
                else:
                    judge_reasoning_tokens_reported_total += reasoning_tokens
                judge_reasoning_bytes_total += reasoning_bytes
            if _should_record_judge_response_receipt(parser_schema_version):
                judge_response_receipts.append(
                    {
                        "query_id": member.query_id,
                        "attempts": final.attempts,
                        "initial_empty_response": (
                            initial is not None
                            and initial.retry_reason == "empty_final_response"
                        ),
                        **(
                            {
                                "initial_retry_reason": (
                                    None if initial is None else initial.retry_reason
                                )
                            }
                            if parser_schema_version in {7, 8, 9, 10}
                            else {}
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

        execution_root_observed_cost = float(
            portfolio_budget_settled_cost_cny(budget_ledger)
        )
        summary_execution_root_observed_cost = float(
            summary.get("observed_dashscope_cost_cny", -1.0)
        )
        expected_settled_cost = format(budget_ledger.settled_actual_cost_cny, ".12f")
        if (
            summary.get("budget_authority_sha256")
            != budget_ledger.authority.authority_sha256
            or summary.get("budget_ledger_last_event_index")
            != budget_ledger.last_event_index
            or summary.get("budget_ledger_last_event_sha256")
            != budget_ledger.last_event_sha256
            or summary.get("budget_ledger_settled_actual_cost_cny")
            != expected_settled_cost
            or summary.get("budget_ledger_forfeited_reserved_cost_cny")
            != format(
                getattr(
                    budget_ledger,
                    "forfeited_reserved_cost_cny",
                    Decimal("0.000000000000"),
                ),
                ".12f",
            )
            or summary.get("budget_ledger_unresolved_reserved_cost_cny")
            != format(budget_ledger.unresolved_reserved_cost_cny, ".12f")
            or summary.get("budget_ledger_accountable_cost_cny")
            != format(budget_ledger.accountable_cost_cny, ".12f")
        ):
            raise ValueError("shard summary differs from the verified budget ledger")
        prior_cost = float(budget_ledger.authority.prior_observed_cost_cny)
        shard_filter = frozenset({shard.shard_id})
        assistant_cost = float(
            portfolio_budget_settled_cost_cny(
                budget_ledger,
                shard_ids=shard_filter,
                stages=frozenset({"assistant_route", "assistant_action"}),
            )
        )
        judge_cost = float(
            portfolio_budget_settled_cost_cny(
                budget_ledger,
                shard_ids=shard_filter,
                stages=frozenset({"final_judge"}),
            )
        )
        attempt_cost_estimate, retryable_attempt_model_calls = _attempt_cost_and_calls(
            args.execution_root / shard.output_relpath,
            shard_id=shard.shard_id,
        )
        shared_route_owner_id = (
            _shared_route_owner_id(launch, shard) if shared_route_enabled else None
        )
        owns_shared_routes = shared_route_owner_id == shard.shard_id
        shared_route_cost = float(
            portfolio_budget_settled_cost_cny(
                budget_ledger,
                shard_ids=shard_filter,
                stages=frozenset({"shared_route"}),
            )
        )
        shard_local_observed_cost = float(
            portfolio_budget_settled_cost_cny(
                budget_ledger,
                shard_ids=shard_filter,
            )
        )
        if (
            summary_execution_root_observed_cost < 0
            or abs(summary_execution_root_observed_cost - execution_root_observed_cost)
            > 1e-9
            or shard_local_observed_cost > summary_execution_root_observed_cost + 1e-9
        ):
            raise ValueError(
                "shard summary root-cumulative cost differs from verified usage"
            )
        if (
            abs(
                float(
                    summary.get(
                        "prior_dashscope_observed_cost_cny",
                        -1.0,
                    )
                )
                - prior_cost
            )
            > 1e-9
            or abs(
                float(
                    summary.get(
                        "cumulative_dashscope_cost_cny",
                        -1.0,
                    )
                )
                - (prior_cost + summary_execution_root_observed_cost)
            )
            > 1e-9
        ):
            raise ValueError(
                "shard summary prior/phase-cumulative cost is inconsistent"
            )
        tool_error_summary = _summarize_tool_errors(tool_trace_rows)
        card_policy_summary = _summarize_card_policy(card_policy_rows)
        audit_payload = {
            "schema_version": parser_schema_version,
            "kind": "portfolio-shard-audit",
            "formal_eligible": False,
            "shard_id": shard.shard_id,
            "config": shard.config,
            "query_count": len(members),
            "assistant_success_count": len(members) - len(assistant_error_ids),
            "assistant_error_count": len(assistant_error_ids),
            "assistant_error_ids": assistant_error_ids,
            "fixed_zero_count": fixed_zero_count,
            "judge_invoked_count": len(judged_scores),
            "judge_status_counts": dict(sorted(judge_status_counts.items())),
            "tool_call_count": tool_call_count,
            **tool_error_summary,
            **card_policy_summary,
            "assistant_model_calls": assistant_model_calls,
            "assistant_input_tokens": assistant_input_tokens,
            "assistant_output_tokens": assistant_output_tokens,
            "judge_input_tokens": judge_input_tokens,
            "judge_output_tokens": judge_output_tokens,
            "mean_j_project_all_rows": (
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            ),
            "mean_j_project_judged_rows": (
                sum(judged_scores) / len(judged_scores) if judged_scores else 0.0
            ),
            "min_j_project": min(all_scores) if all_scores else 0.0,
            "max_j_project": max(all_scores) if all_scores else 0.0,
            "dashscope_observed_cost_cny": (summary_execution_root_observed_cost),
            "prior_dashscope_observed_cost_cny": prior_cost,
            "cumulative_dashscope_observed_cost_cny": (
                prior_cost + summary_execution_root_observed_cost
            ),
            "shard_summary_sha256": supplied_summary_sha256,
            "runtime_lock_sha256": control["runtime_lock_sha256"],
            "launch_plan_sha256": control["launch_plan_sha256"],
            **_ACTIVE_BUDGET_CONTRACT,
            "budget_authority_sha256": (budget_ledger.authority.authority_sha256),
            "budget_ledger_last_event_index": budget_ledger.last_event_index,
            "budget_ledger_last_event_sha256": budget_ledger.last_event_sha256,
            "budget_ledger_settled_actual_cost_cny": format(
                budget_ledger.settled_actual_cost_cny, ".12f"
            ),
            "budget_ledger_forfeited_reserved_cost_cny": format(
                getattr(
                    budget_ledger,
                    "forfeited_reserved_cost_cny",
                    Decimal("0.000000000000"),
                ),
                ".12f",
            ),
            "budget_ledger_unresolved_reserved_cost_cny": format(
                budget_ledger.unresolved_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_accountable_cost_cny": format(
                budget_ledger.accountable_cost_cny, ".12f"
            ),
        }
        if control.get("execution_scope") == "partial_shard_repair":
            audit_payload.update(
                {
                    "execution_root_observed_cost_cny": (
                        summary_execution_root_observed_cost
                    ),
                    "shard_local_observed_cost_cny": (shard_local_observed_cost),
                    "shard_assistant_observed_cost_cny": assistant_cost,
                    "shard_final_judge_observed_cost_cny": judge_cost,
                    "shard_retryable_attempt_artifact_estimate_cost_cny": (
                        attempt_cost_estimate
                    ),
                    "shard_shared_route_observed_cost_cny": (shared_route_cost),
                    "phase_cumulative_observed_cost_cny": (
                        prior_cost + summary_execution_root_observed_cost
                    ),
                    "phase_cumulative_cap_cny": control["phase_cumulative_cap_cny"],
                    "retryable_attempt_model_calls": (retryable_attempt_model_calls),
                    "assistant_effective_turns": assistant_effective_turns,
                    "shared_route_reference_count": (shared_route_reference_count),
                    "shared_route_reserved_turns": (shared_route_reserved_turns),
                    "shared_route_reserved_input_tokens": (
                        shared_route_reserved_input_tokens
                    ),
                    "shared_route_reserved_output_tokens": (
                        shared_route_reserved_output_tokens
                    ),
                    "shared_route_model_calls_owned": (
                        len(shared_route_artifacts) if owns_shared_routes else 0
                    ),
                    "shared_route_cost_owner_shard_id": (shared_route_owner_id),
                }
            )
        if parser_schema_version > 1:
            audit_payload.update(
                {
                    "final_judge_parser_policy_version": (locked_parser_version),
                    "final_judge_parser_policy_sha256": (locked_parser_sha256),
                    "judge_raw_dimensions_shape_counts": dict(
                        sorted(judge_shape_counts.items())
                    ),
                    "mean_j_project_scored_rows": (
                        sum(scored_scores) / len(scored_scores)
                        if scored_scores
                        else 0.0
                    ),
                }
            )
        if parser_schema_version in {4, 5, 6, 7, 8, 9, 10}:
            audit_payload.update(
                {
                    "final_judge_result_schema_version": parser_schema_version,
                    "final_judge_cache_namespace": final_cache_namespace,
                    "card_requirement_guard_policy_version": runtime_lock[
                        "card_requirement_guard_policy_version"
                    ],
                    "card_requirement_guard_policy_sha256": runtime_lock[
                        "card_requirement_guard_policy_sha256"
                    ],
                    "judge_card_requirement_guard_adjusted_count": (
                        judge_card_guard_adjusted_count
                    ),
                }
            )
        if parser_schema_version in {5, 6, 7, 8, 9, 10}:
            audit_payload.update(
                {
                    "final_judge_max_attempts": runtime_lock[
                        "final_judge_max_attempts"
                    ],
                    "final_judge_retry_policy_version": runtime_lock[
                        "final_judge_retry_policy_version"
                    ],
                    "final_judge_retry_policy_sha256": runtime_lock[
                        "final_judge_retry_policy_sha256"
                    ],
                    "judge_model_calls": judge_model_calls,
                    "judge_captured_response_count": (judge_captured_response_count),
                    "judge_retried_row_count": judge_retried_row_count,
                    "judge_initial_empty_response_count": (
                        judge_initial_empty_response_count
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
                }
            )
        if parser_schema_version in {7, 8, 9, 10}:
            audit_payload["judge_initial_invalid_json_response_count"] = (
                judge_initial_invalid_json_response_count
            )
        if parser_schema_version in {6, 7, 8, 9, 10}:
            audit_payload.update(
                {
                    "final_judge_thinking_budget": runtime_lock[
                        "final_judge_thinking_budget"
                    ],
                    "final_judge_max_billable_input_tokens": runtime_lock[
                        "final_judge_max_billable_input_tokens"
                    ],
                    "final_judge_max_billable_output_tokens": runtime_lock[
                        "final_judge_max_billable_output_tokens"
                    ],
                    "final_judge_provider_input_token_reserve": runtime_lock[
                        "final_judge_provider_input_token_reserve"
                    ],
                    "final_judge_provider_output_token_reserve": runtime_lock[
                        "final_judge_provider_output_token_reserve"
                    ],
                }
            )
        if parser_schema_version == 10:
            audit_payload.update(
                {
                    "final_judge_transport_policy_version": runtime_lock[
                        "final_judge_transport_policy_version"
                    ],
                    "final_judge_transport_policy_sha256": runtime_lock[
                        "final_judge_transport_policy_sha256"
                    ],
                    "final_judge_requested_response_format": runtime_lock[
                        "final_judge_requested_response_format"
                    ],
                    "final_judge_provider_pricing_status": runtime_lock[
                        "final_judge_provider_pricing_status"
                    ],
                }
            )
        audit = {
            **audit_payload,
            "audit_sha256": _hash(audit_payload),
        }
        expected_audit_bytes = canonical_json_bytes(audit)
        latest_budget_ledger = load_portfolio_budget_ledger(
            _budget_ledger_root(args.execution_root, control)
        )
        if _budget_ledger_tail_identity(
            latest_budget_ledger
        ) != _budget_ledger_tail_identity(budget_ledger):
            raise ValueError("budget ledger changed during shard finalization")
        if (
            read_stable_regular_file(
                summary_path,
                label=f"shard summary {shard.shard_id}",
            )
            != source_summary_bytes
        ):
            raise ValueError("shard summary changed during finalization")
        if summary_refresh_required:
            atomic_replace_file(summary_path, refreshed_summary_bytes)
            if (
                read_stable_regular_file(
                    summary_path,
                    label=f"refreshed shard summary {shard.shard_id}",
                )
                != refreshed_summary_bytes
            ):
                raise ValueError("refreshed shard summary failed verification")
        if audit_path.exists():
            if audit_path.read_bytes() != expected_audit_bytes:
                raise ValueError(
                    "existing shard audit differs from verified checkpoints"
                )
        else:
            atomic_create_file(audit_path, expected_audit_bytes)

        partial_repair = control.get("execution_scope") == "partial_shard_repair"
        if partial_repair:
            authorized = set(control.get("authorized_shard_ids", ()))
            launch_shards_by_id = {item.shard_id: item for item in launch.plan.shards}
            local_completed: set[str] = set()
            for path in args.execution_root.glob("shards/*/shard-audit.json"):
                item = _load_canonical_object(
                    path,
                    label=f"shard audit {path.parent.name}",
                )
                supplied = _require_self_hash(
                    item,
                    "audit_sha256",
                    label=f"shard audit {path.parent.name}",
                )
                del supplied
                local_shard_id = item.get("shard_id")
                local_plan_shard = launch_shards_by_id.get(local_shard_id)
                if (
                    local_shard_id != path.parent.name
                    or local_shard_id not in authorized
                    or local_shard_id == external_frozen_target_id
                    or local_plan_shard is None
                    or item.get("kind") != "portfolio-shard-audit"
                    or item.get("config") != local_plan_shard.config
                    or item.get("query_count") != local_plan_shard.query_count
                    or item.get("runtime_lock_sha256") != control["runtime_lock_sha256"]
                    or item.get("launch_plan_sha256") != launch.plan.launch_plan_sha256
                ):
                    raise ValueError(
                        "local shard audit is outside the partial execution scope"
                    )
                local_completed.add(local_shard_id)
            if shard.shard_id not in local_completed:
                raise ValueError("new local shard audit was not discovered")
            completed = set(local_completed)
            if external_frozen_target_id is None:
                raise ValueError("partial shard repair lacks verified external NoSkill")
            completed.add(external_frozen_target_id)
            if not set(launch.state.completed_shard_ids).issubset(completed):
                raise ValueError(
                    "launch state claims shards without verified local/external evidence"
                )
        else:
            completed = set(launch.state.completed_shard_ids)
            completed.add(shard.shard_id)
        ordered_completed = tuple(
            item.shard_id for item in launch.plan.shards if item.shard_id in completed
        )
        total_calls = _execution_model_calls(args.execution_root)
        state = PortfolioLaunchState(
            matrix_run_id=launch.plan.matrix_run_id,
            launch_plan_sha256=launch.plan.launch_plan_sha256,
            status=_launch_completion_status(
                launch.plan,
                ordered_completed,
            ),
            completed_shard_ids=ordered_completed,
            failed_shard_ids=launch.state.failed_shard_ids,
            model_calls_performed=total_calls,
            dashscope_observed_cost_cny=execution_root_observed_cost,
            aifast_observed_cost_cny=launch.state.aifast_observed_cost_cny,
        )
        atomic_replace_file(
            launch.root / "run-state.json",
            canonical_json_bytes(state.model_dump(mode="json")),
        )
        verified_launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        if verified_launch.state != state:
            raise ValueError("updated launch state failed round-trip verification")

    except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
        print(f"finalize-portfolio-shard: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "audit_sha256": audit["audit_sha256"],
                "assistant_error_count": audit["assistant_error_count"],
                "cumulative_dashscope_observed_cost_cny": audit[
                    "cumulative_dashscope_observed_cost_cny"
                ],
                "judge_status_counts": audit["judge_status_counts"],
                "mean_j_project_all_rows": audit["mean_j_project_all_rows"],
                "model_calls_performed": state.model_calls_performed,
                "shard_id": shard.shard_id,
                "summary_refreshed": summary_refresh_required,
                "status": state.status,
                "tool_error_count": audit["tool_error_count"],
                "card_policy_violation_count": audit["card_policy_violation_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
