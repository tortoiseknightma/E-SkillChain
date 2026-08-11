"""Read-only cross-config audit for one 25-query Portfolio batch.

The command never invokes a model and never writes into the execution,
launch, or runtime roots.  Its only output is one canonical, content-addressed
JSON value on stdout.  Incomplete 25 x 5 input fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from scripts.prepare_portfolio_execution import (  # noqa: E402
    _final_judge_comparison_identity,
)
from scripts.run_portfolio_assistant_smoke import _hash, _model  # noqa: E402
from scripts.run_portfolio_shard import (  # noqa: E402
    _assistant_result,
    _assistant_row,
    _load_control,
)
from skillchain import config as skillchain_config  # noqa: E402
from skillchain.data.asset_catalog import load_asset_catalog  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
    _validate_backend_response,
)
from skillchain.evaluation.evaluator_isolation import (  # noqa: E402
    build_bound_final_prompt,
    make_active_portfolio_evaluator_isolation_lock,
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
from skillchain.evaluation.packets import (  # noqa: E402
    RubricSnapshot,
    build_final_evaluation_packet,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    MAIN_CONFIG_ORDER,
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PortfolioAttemptReceipt,
    load_portfolio_budget_ledger,
    portfolio_attempt_provider_call_count,
    portfolio_budget_settled_cost_cny,
)
from skillchain.runners.assistant import (  # noqa: E402
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
    evolution_bank_boundary_violations,
    noskill_execution_contract_payload,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.planning import CapabilityAssignment  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


_EXPECTED_CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)
_EXPECTED_ASSISTANT_PROVIDER = "qwen"
_EXPECTED_ASSISTANT_MODEL = "qwen3-vl-flash-2026-01-22"
_EXPECTED_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_EXPECTED_BUDGET = {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096,
    "max_tool_calls": 3,
    "max_turns": 5,
    "timeout_ms": 180000,
}
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
_ACTIVE_BUDGET_CONTRACT = {
    "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
    "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
    "provider_pricing_contract_version": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION),
    "provider_pricing_contract_sha256": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256),
}
_EXPECTED_JUDGE_PROVIDER = "gemini"
_EXPECTED_JUDGE_MODEL = "gemini-3.6-flash"
_EXPECTED_JUDGE_ENDPOINT = skillchain_config.PROVIDER_ENDPOINTS[
    _EXPECTED_JUDGE_PROVIDER
]
_EXPECTED_JUDGE_MAX_TOKENS = 2048
_OBSERVED_SOURCE_FILES = {
    "src/skillchain/config.py": SOURCE_ROOT / "skillchain" / "config.py",
    "src/skillchain/evaluation/evaluator_isolation.py": (
        SOURCE_ROOT / "skillchain" / "evaluation" / "evaluator_isolation.py"
    ),
    "src/skillchain/evaluation/packets.py": (
        SOURCE_ROOT / "skillchain" / "evaluation" / "packets.py"
    ),
    "src/skillchain/runners/assistant.py": (
        SOURCE_ROOT / "skillchain" / "runners" / "assistant.py"
    ),
    "src/skillchain/tools/portfolio_runtime.py": (
        SOURCE_ROOT / "skillchain" / "tools" / "portfolio_runtime.py"
    ),
    "src/skillchain/tools/registry.py": (
        SOURCE_ROOT / "skillchain" / "tools" / "registry.py"
    ),
}
_ACTIVE_EVALUATOR_SOURCE_LOCK = {
    "config_file_sha256": sha256_bytes(
        _OBSERVED_SOURCE_FILES["src/skillchain/config.py"].read_bytes()
    ),
    "packets_file_sha256": sha256_bytes(
        _OBSERVED_SOURCE_FILES["src/skillchain/evaluation/packets.py"].read_bytes()
    ),
    "evaluator_isolation_file_sha256": sha256_bytes(
        _OBSERVED_SOURCE_FILES[
            "src/skillchain/evaluation/evaluator_isolation.py"
        ].read_bytes()
    ),
    "portfolio_tool_runtime_file_sha256": sha256_bytes(
        _OBSERVED_SOURCE_FILES["src/skillchain/tools/portfolio_runtime.py"].read_bytes()
    ),
    "tool_registry_file_sha256": sha256_bytes(
        _OBSERVED_SOURCE_FILES["src/skillchain/tools/registry.py"].read_bytes()
    ),
}


def _require_active_final_result_runtime(
    runtime_lock: Mapping[str, object],
    *,
    label: str,
) -> None:
    if any(
        runtime_lock.get(field) != expected
        for field, expected in _ACTIVE_FINAL_RESULT_LOCK.items()
    ):
        raise ValueError(
            f"{label} does not bind the active final-Judge result, retry, "
            "and card-requirement-guard contracts"
        )
    if any(
        runtime_lock.get(field) != expected
        for field, expected in _ACTIVE_EVALUATOR_SOURCE_LOCK.items()
    ):
        raise ValueError(
            f"{label} does not bind the active final-evaluation config, packet, "
            "and evaluator-isolation sources"
        )
    if any(
        runtime_lock.get(field) != expected
        for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
    ):
        raise ValueError(f"{label} does not bind the active hard-budget contract")


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _control_cny(control: Mapping[str, object], field: str) -> Decimal:
    value = control.get(field)
    if not isinstance(value, str) or re.fullmatch(r"\d+\.\d{12}", value) is None:
        raise ValueError(f"execution control {field} must be a 12-place CNY string")
    try:
        return Decimal(value)
    except InvalidOperation as error:  # pragma: no cover - regex excludes it
        raise ValueError(f"execution control {field} is not decimal") from error


def _budget_ledger_root(
    execution_root: Path,
    control: Mapping[str, object],
) -> Path:
    relative = control.get("budget_ledger_relpath")
    if (
        relative != "budget-ledger"
        or control.get("budget_authority_relpath")
        != "budget-ledger/budget-authority.json"
        or control.get("budget_authority_creation_policy")
        != "create_only_on_first_execute"
    ):
        raise ValueError("execution control budget-ledger layout drifted")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError("execution control budget-ledger path is unsafe")
    return execution_root.joinpath(*parsed.parts)


@dataclass(frozen=True)
class _RowObservation:
    query_id: str
    config: str
    canonical_capability: str
    card_requirement: str
    visible_card_count: int
    card_policy_compliant: bool
    assistant_error_code: str | None
    receipt_outcome: str
    model_call_count: int
    finish_reasons: tuple[str, ...]
    turn_count: int
    tool_trace: tuple[tuple[str, str, str | None], ...]
    selected_capability: str | None
    skill_slug: str | None
    route_trace_sha256: str | None
    route_call_response_sha256: str | None
    shared_route_artifact_sha256: str | None
    shared_route_reserved_input_tokens: int | None
    shared_route_reserved_output_tokens: int | None
    shared_route_reserved_turns: int
    shared_route_status: str | None
    route_acceptable: bool | None
    assistant_input_tokens: int
    assistant_output_tokens: int
    judge_status: str
    judge_error_code: str | None
    judge_shape: str | None
    judge_finish_reason: str | None
    judge_raw_response_bytes: int | None
    judge_tool_call_count: int | None
    judge_input_tokens: int
    judge_output_tokens: int
    judge_attempts: int
    judge_initial_empty_response: bool
    judge_terminal_response_captured: bool
    judge_initial_reasoning_present: bool | None
    judge_initial_reasoning_tokens: int | None
    judge_initial_reasoning_bytes: int | None
    judge_initial_reasoning_sha256: str | None
    judge_terminal_reasoning_present: bool | None
    judge_terminal_reasoning_tokens: int | None
    judge_terminal_reasoning_bytes: int | None
    judge_terminal_reasoning_sha256: str | None
    j_project: float
    judge_initial_retry_reason: str | None = None


@dataclass(frozen=True)
class _ShardArtifactSource:
    """Physical source for one logical launch shard."""

    execution_root: Path
    launch: object
    shard: object
    members: tuple[object, ...]
    control: Mapping[str, object]
    runtime_root: Path
    runtime_raw: Mapping[str, object]
    source_request_contract: Mapping[str, object] | None = None
    external: bool = False
    artifact_alias: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _LaunchProfileInputs:
    """Private inputs and local artifact roots reconstructed from one launch."""

    profile: str
    queries: tuple[object, ...]
    capability_assignments_path: Path
    catalog_dir: Path
    asset_root: Path
    selected_splits: tuple[str, ...]
    split_by_query: Mapping[str, str]
    population_split_counts: Mapping[str, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--batch-id")
    selection.add_argument(
        "--shard-id",
        action="append",
        dest="shard_ids",
        help="Repeat exactly five times to audit an explicit 1x5 shard set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optionally create this canonical JSON audit file. "
            "Existing paths are never overwritten."
        ),
    )
    parser.add_argument(
        "--expected-noskill-audit-sha256",
        help=(
            "Optional previously observed NoSkill shard-audit digest. "
            "A match is freeze evidence, not proof of wall-clock order."
        ),
    )
    return parser


def _resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_launch_profile_inputs(launch: object) -> _LaunchProfileInputs:
    """Reconstruct the private input side selected by the loaded launch plan.

    Historical ``dev_mini`` keeps its accepted compatibility loader.  Core is
    reconstructed solely from the typed, externally hashed paths embedded in
    the launch plan, so an auditor cannot silently fall back to mini data.
    """

    plan = getattr(launch, "plan")
    plan_profile = getattr(plan, "dataset_profile", None)
    plan_kind = str(getattr(plan, "kind", ""))
    launch_query_ids = {
        str(query_id)
        for shard in getattr(plan, "shards")
        for query_id in getattr(shard, "query_ids")
    }
    if plan_profile == "core":
        if plan_kind != "portfolio-core-split-x5-launch-plan":
            raise ValueError("Core dataset profile has a non-Core launch kind")
        verified = reconstruct_verified_portfolio_core_inputs(plan)
        selected_splits = tuple(str(item) for item in plan.selected_splits)
        split_by_query = {
            str(query.query_id): str(query.split) for query in verified.queries
        }
        selected_queries = tuple(
            query for query in verified.queries if str(query.split) in selected_splits
        )
        if (
            len(selected_queries) != int(plan.query_count)
            or {str(query.query_id) for query in selected_queries} != launch_query_ids
        ):
            raise ValueError("Core private input selection differs from launch shards")
        population_split_counts = Counter(
            str(query.split) for query in verified.queries
        )
        files = verified.files
        return _LaunchProfileInputs(
            profile="core",
            queries=tuple(verified.queries),
            capability_assignments_path=files.capability_assignments_path,
            catalog_dir=(files.runtime_catalog_dir or files.base_catalog_dir),
            asset_root=files.asset_root,
            selected_splits=selected_splits,
            split_by_query=split_by_query,
            population_split_counts=dict(population_split_counts),
        )

    if (
        plan_profile is not None
        or plan_kind == "portfolio-core-split-x5-launch-plan"
        or tuple(getattr(plan, "selected_splits", ()))
    ):
        raise ValueError("launch profile fields mix Core and dev_mini identities")
    active = _load_active_inputs()
    if (
        active.expected_query_artifact_sha256 != plan.query_artifact_sha256
        or active.expected_capability_assignments_sha256
        != plan.capability_assignments_sha256
        or {str(query.query_id) for query in active.queries} != launch_query_ids
    ):
        raise ValueError("active dev_mini inputs differ from launch plan")
    remote = active.remote_files
    return _LaunchProfileInputs(
        profile="dev_mini",
        queries=tuple(active.queries),
        capability_assignments_path=Path(active.capability_assignments_path),
        catalog_dir=Path(remote.output_catalog_dir),
        asset_root=Path(remote.asset_root),
        selected_splits=(),
        split_by_query={},
        population_split_counts={},
    )


def _require_launch_profile_inputs(
    launch: object,
    value: _LaunchProfileInputs,
) -> _LaunchProfileInputs:
    """Cheaply bind an already deep-verified profile handle to this launch."""

    if type(value) is not _LaunchProfileInputs:
        raise TypeError("cached launch profile inputs have an invalid type")
    plan = getattr(launch, "plan")
    expected_profile = str(getattr(plan, "dataset_profile", None) or "dev_mini")
    launch_query_ids = {
        str(query_id)
        for shard in getattr(plan, "shards")
        for query_id in getattr(shard, "query_ids")
    }
    private_by_id = {str(query.query_id): query for query in value.queries}
    selected_query_ids = (
        {
            query_id
            for query_id, split in value.split_by_query.items()
            if split in value.selected_splits
        }
        if value.profile == "core"
        else set(private_by_id)
    )
    if (
        value.profile != expected_profile
        or selected_query_ids != launch_query_ids
        or set(private_by_id) != {str(query.query_id) for query in value.queries}
        or len(private_by_id) != len(value.queries)
        or (
            value.profile == "core"
            and value.selected_splits
            != tuple(str(item) for item in getattr(plan, "selected_splits"))
        )
    ):
        raise ValueError("cached private inputs differ from launch profile")
    return value


def _load_canonical_object(path: Path, *, label: str) -> tuple[dict, bytes]:
    content = read_stable_regular_file(path, label=label)
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value, content


def _content_address(payload: Mapping[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop("audit_sha256", None)
    return {**unsigned, "audit_sha256": _hash(unsigned)}


def _observed_source_file_sha256s() -> dict[str, str]:
    """Hash live evaluator-input sources without claiming launch-lock binding."""

    return {
        label: sha256_bytes(
            read_stable_regular_file(path, label=f"observed source {label}")
        )
        for label, path in sorted(_OBSERVED_SOURCE_FILES.items())
    }


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _launch_endpoint(plan: object, check_id: str) -> str:
    matches = tuple(
        item.detail
        for item in getattr(plan, "preflight_checks")
        if item.check_id == check_id
    )
    if len(matches) != 1 or not matches[0].startswith("https://"):
        raise ValueError(f"launch lacks one verified {check_id} endpoint")
    return matches[0]


def _comparison_launch_projection(plan: object) -> dict[str, object]:
    feedback_endpoint_check = (
        "endpoint.kimi_dashscope"
        if getattr(plan, "feedback_provider") == "kimi"
        else "endpoint.aifast"
    )
    final_endpoint_check = (
        "endpoint.aifast"
        if getattr(plan, "final_provider") == "gemini"
        else "endpoint.kimi_dashscope"
    )
    return {
        "schema_version": 1,
        "field_source": "portfolio-launch-plan-v1",
        "corpus": {
            "portfolio_plan_sha256": getattr(plan, "portfolio_plan_sha256"),
            "portfolio_plan_manifest_file_sha256": getattr(
                plan, "portfolio_plan_manifest_file_sha256"
            ),
            "accepted_ledger_sha256": getattr(plan, "accepted_ledger_sha256"),
            "query_artifact_sha256": getattr(plan, "query_artifact_sha256"),
            "capability_assignments_sha256": getattr(
                plan, "capability_assignments_sha256"
            ),
            "seed_set_sha256": getattr(plan, "seed_set_sha256"),
        },
        "catalog_and_image_manifests": {
            "base_catalog_sha256": getattr(plan, "base_catalog_sha256"),
            "runtime_catalog_sha256": getattr(plan, "runtime_catalog_sha256"),
            "authorization_file_sha256": getattr(plan, "authorization_file_sha256"),
            "receipt_file_sha256": getattr(plan, "receipt_file_sha256"),
            "receipt_sha256": getattr(plan, "receipt_sha256"),
            "remote_runtime_binding_sha256s": list(
                getattr(plan, "remote_runtime_binding_sha256s")
            ),
        },
        "models": {
            "assistant": {
                "provider": getattr(plan, "assistant_provider"),
                "model": getattr(plan, "assistant_model"),
                "endpoint": _launch_endpoint(plan, "endpoint.dashscope"),
            },
            "feedback": {
                "provider": getattr(plan, "feedback_provider"),
                "model": getattr(plan, "feedback_model"),
                "endpoint": _launch_endpoint(plan, feedback_endpoint_check),
            },
            "final_judge": {
                "provider": getattr(plan, "final_provider"),
                "model": getattr(plan, "final_model"),
                "endpoint": _launch_endpoint(plan, final_endpoint_check),
            },
        },
    }


def _source_request_contract(
    *,
    source: _ShardArtifactSource,
) -> dict[str, object]:
    _require_active_final_result_runtime(
        source.runtime_raw,
        label="external source runtime lock",
    )
    assistant_contract: dict[str, object] | None = None
    final_contract: dict[str, object] | None = None
    for member in source.members:
        assistant_path = source.execution_root / str(
            getattr(member, "assistant_output_relpath")
        )
        assistant_raw, _ = _load_canonical_object(
            assistant_path,
            label=f"external source Assistant {getattr(member, 'query_id')}",
        )
        request = AssistantRequestSnapshot.model_validate_json(
            canonical_json_bytes(assistant_raw.get("request")),
            strict=True,
        )
        current_assistant = {
            "backbone": request.backbone.model_dump(mode="json"),
            "budget": request.budget.model_dump(mode="json"),
            "registry": request.registry.model_dump(mode="json"),
        }
        if assistant_contract is None:
            assistant_contract = current_assistant
        elif assistant_contract != current_assistant:
            raise ValueError("external source NoSkill request contract drifted")
        response, _ = _assistant_row(assistant_path)
        if response.error_code is not None:
            continue
        final = load_final_judge_evaluation_result(
            source.execution_root / str(getattr(member, "final_output_relpath"))
        )
        if (
            final.schema_version
            != source.runtime_raw["final_judge_result_schema_version"]
            or final.cache_namespace
            != source.runtime_raw["final_judge_cache_namespace"]
            or final.parser_policy_version
            != source.runtime_raw.get("final_judge_parser_policy_version")
            or final.parser_policy_sha256
            != source.runtime_raw.get("final_judge_parser_policy_sha256")
            or final.card_requirement_guard_policy_version
            != source.runtime_raw["card_requirement_guard_policy_version"]
            or final.card_requirement_guard_policy_sha256
            != source.runtime_raw["card_requirement_guard_policy_sha256"]
            or final.visible_card_count != len(response.visible_cards)
            or final.max_attempts != source.runtime_raw["final_judge_max_attempts"]
            or final.retry_policy_version
            != source.runtime_raw["final_judge_retry_policy_version"]
            or final.retry_policy_sha256
            != source.runtime_raw["final_judge_retry_policy_sha256"]
            or final.transport_policy_version
            != source.runtime_raw["final_judge_transport_policy_version"]
            or final.transport_policy_sha256
            != source.runtime_raw["final_judge_transport_policy_sha256"]
            or final.requested_response_format
            != source.runtime_raw["final_judge_requested_response_format"]
        ):
            raise ValueError(
                "external source final-Judge parser/card-guard contract drifted"
            )
        current_final = {
            **_final_judge_comparison_identity(final),
            "transport_policy_version": final.transport_policy_version,
            "transport_policy_sha256": final.transport_policy_sha256,
            "requested_response_format": final.requested_response_format,
        }
        if final_contract is None:
            final_contract = current_final
        elif final_contract != current_final:
            raise ValueError("external source final-Judge contract drifted")
    if assistant_contract is None or final_contract is None:
        raise ValueError("external source lacks comparison request contracts")
    return {
        "assistant": assistant_contract,
        "final_judge": final_contract,
    }


def _comparison_contract_payload(
    *,
    launch: object,
    control: Mapping[str, object],
    runtime_raw: Mapping[str, object],
    source_request_contract: Mapping[str, object],
    require_noskill_runtime_binding: bool = False,
) -> dict[str, object]:
    _require_active_final_result_runtime(
        runtime_raw,
        label="comparison runtime lock",
    )
    plan = getattr(launch, "plan")
    assistant = source_request_contract["assistant"]
    if not isinstance(assistant, dict):
        raise ValueError("source Assistant comparison contract is invalid")
    backbone = assistant["backbone"]
    budget = assistant["budget"]
    registry = assistant["registry"]
    if not all(isinstance(item, dict) for item in (backbone, budget, registry)):
        raise ValueError("source Assistant comparison contract is invalid")
    final_contract = source_request_contract["final_judge"]
    if not isinstance(final_contract, dict):
        raise ValueError("source final-Judge comparison contract is invalid")
    expected_backbone = {
        "provider": getattr(plan, "assistant_provider"),
        "model": getattr(plan, "assistant_model"),
        "endpoint": _launch_endpoint(plan, "endpoint.dashscope"),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": runtime_raw.get("system_prompt_sha256"),
    }
    if any(backbone.get(key) != value for key, value in expected_backbone.items()):
        raise ValueError("NoSkill Assistant sampling identity is incompatible")
    if any(budget.get(key) != value for key, value in _EXPECTED_BUDGET.items()):
        raise ValueError("NoSkill Assistant budget contract is incompatible")
    if registry.get("registry_sha256") != runtime_raw.get(
        "tool_registry_sha256"
    ) or registry.get("registry_runtime_sha256") != runtime_raw.get(
        "tool_registry_runtime_sha256"
    ):
        raise ValueError("NoSkill capability registry is incompatible")
    expected_final = _comparison_launch_projection(plan)["models"]["final_judge"]
    if (
        any(
            final_contract.get(field) != expected_final[field]
            for field in ("provider", "model", "endpoint")
        )
        or final_contract.get("max_tokens") != _EXPECTED_JUDGE_MAX_TOKENS
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
        raise ValueError("NoSkill final-Judge contract is incompatible")
    if (
        runtime_raw.get("final_judge_parser_policy_version")
        != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
        or runtime_raw.get("final_judge_parser_policy_sha256")
        != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
    ):
        raise ValueError("NoSkill final-Judge parser is incompatible")
    if require_noskill_runtime_binding and (
        runtime_raw.get("noskill_execution_contract_sha256")
        != NOSKILL_EXECUTION_CONTRACT_SHA256
        or runtime_raw.get("noskill_execution_policy_version")
        != NOSKILL_EXECUTION_POLICY_VERSION
    ):
        raise ValueError("runtime lacks the active NoSkill contract binding")
    return {
        "schema_version": 1,
        "kind": "portfolio-external-frozen-noskill-comparison-contract",
        "field_sources": {
            "corpus_catalog_models_endpoints": ("source-and-target-launch-plan-v1"),
            "assistant_sampling_budget_registry": (
                "source-noskill-checkpoints-and-target-runner-constants-v1"
            ),
            "rubric": "source-and-target-execution-control",
            "parser": "source-and-target-runtime-lock",
            "noskill_execution": NOSKILL_EXECUTION_POLICY_VERSION,
        },
        "launch_projection": _comparison_launch_projection(plan),
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
            **_EXPECTED_BUDGET,
            "budget_sha256": budget["budget_sha256"],
        },
        "capability_registry": {
            "registry_sha256": registry["registry_sha256"],
            "registry_runtime_sha256": registry["registry_runtime_sha256"],
            "registry_lock_sha256": registry["lock_sha256"],
        },
        "feedback_identity": _comparison_launch_projection(plan)["models"]["feedback"],
        "final_judge_identity": final_contract,
        "rubric": {
            "file_sha256": control.get("rubric_file_sha256"),
            "content_sha256": control.get("rubric_content_sha256"),
        },
        "parser": {
            "result_schema_version": runtime_raw.get(
                "final_judge_result_schema_version"
            ),
            "cache_namespace": runtime_raw.get("final_judge_cache_namespace"),
            "policy_version": runtime_raw.get("final_judge_parser_policy_version"),
            "policy_sha256": runtime_raw.get("final_judge_parser_policy_sha256"),
            "card_requirement_guard_policy_version": runtime_raw.get(
                "card_requirement_guard_policy_version"
            ),
            "card_requirement_guard_policy_sha256": runtime_raw.get(
                "card_requirement_guard_policy_sha256"
            ),
            "max_attempts": runtime_raw.get("final_judge_max_attempts"),
            "retry_policy_version": runtime_raw.get("final_judge_retry_policy_version"),
            "retry_policy_sha256": runtime_raw.get("final_judge_retry_policy_sha256"),
            "transport_policy_version": runtime_raw.get(
                "final_judge_transport_policy_version"
            ),
            "transport_policy_sha256": runtime_raw.get(
                "final_judge_transport_policy_sha256"
            ),
            "requested_response_format": runtime_raw.get(
                "final_judge_requested_response_format"
            ),
            "provider_pricing_status": runtime_raw.get(
                "final_judge_provider_pricing_status"
            ),
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


def _shard_artifact_inventory(
    execution_root: Path,
    *,
    shard: object,
    members: Sequence[object],
) -> dict[str, str]:
    shard_root = execution_root / str(getattr(shard, "output_relpath"))
    query_ids = tuple(str(getattr(member, "query_id")) for member in members)
    relative_paths = ["shard-audit.json", "shard-summary.json"]
    relative_paths.extend(f"assistant/{query_id}.json" for query_id in query_ids)
    relative_paths.extend(f"final/{query_id}.json" for query_id in query_ids)
    actual = {
        path.relative_to(shard_root).as_posix()
        for path in shard_root.rglob("*")
        if path.is_file()
    }
    if len(relative_paths) != 52 or actual != set(relative_paths):
        raise ValueError(
            "external frozen NoSkill artifact layout is not exactly 52 files"
        )
    return {
        relative: sha256_bytes(
            read_stable_regular_file(
                shard_root / relative,
                label=f"external frozen artifact {relative}",
                max_bytes=64 * 1024 * 1024,
            )
        )
        for relative in sorted(relative_paths)
    }


def _member_compatibility_violations(
    *,
    target_members: Sequence[object],
    source_members: Sequence[object],
) -> list[str]:
    violations: list[str] = []
    source_by_query = {
        str(getattr(member, "query_id")): member for member in source_members
    }
    if len(source_by_query) != len(source_members):
        return ["external NoSkill source contains duplicate query IDs"]
    fields = (
        "query_ordinal",
        "query_sha256",
        "public_input_sha256",
        "asset_id",
        "image_path",
        "image_sha256",
    )
    for target in target_members:
        query_id = str(getattr(target, "query_id"))
        source = source_by_query.get(query_id)
        if source is None:
            violations.append(f"external NoSkill source lacks target query: {query_id}")
            continue
        drifted = [
            field
            for field in fields
            if getattr(target, field) != getattr(source, field)
        ]
        if drifted:
            violations.append(
                f"external NoSkill input drift for {query_id}: " + ",".join(drifted)
            )
    if len(target_members) != len(source_members):
        violations.append("external NoSkill source query count differs from target")
    return violations


def _load_external_frozen_source(
    *,
    control: Mapping[str, object],
    target_launch: object,
    target_shards: Sequence[object],
    target_members: Sequence[object],
) -> tuple[_ShardArtifactSource | None, dict[str, object] | None, list[str]]:
    descriptor = control.get("external_frozen_shard")
    if descriptor is None:
        return None, None, []
    if not isinstance(descriptor, dict):
        raise ValueError("external_frozen_shard must be one object")
    required = {
        "source_execution_root",
        "source_execution_control_file_sha256",
        "source_execution_control_sha256",
        "source_launch_root",
        "source_launch_plan_file_sha256",
        "source_launch_plan_sha256",
        "source_runtime_root",
        "source_runtime_lock_file_sha256",
        "source_runtime_lock_sha256",
        "source_shard_id",
        "source_shard_sha256",
        "source_shard_output_relpath",
        "source_shard_summary_file_sha256",
        "source_shard_summary_sha256",
        "source_shard_audit_file_sha256",
        "source_shard_audit_sha256",
        "target_shard_id",
        "target_shard_sha256",
        "accepted_batch_id",
        "query_count",
        "assistant_checkpoint_count",
        "final_checkpoint_count",
        "artifact_file_count",
        "artifact_file_sha256s",
        "artifact_set_sha256",
        "query_public_image_compatibility_sha256",
        "blinding_key_sha256",
        "comparison_contract_payload",
        "comparison_contract_sha256",
        "binding_sha256",
    }
    missing = sorted(required - set(descriptor))
    if missing:
        raise ValueError(
            "external_frozen_shard lacks required fields: " + ",".join(missing)
        )
    for field in required:
        if field.endswith("_sha256"):
            _require_sha256(descriptor.get(field), label=f"external {field}")
    if (
        descriptor.get("query_count") != 25
        or descriptor.get("assistant_checkpoint_count") != 25
        or descriptor.get("final_checkpoint_count") != 25
        or descriptor.get("artifact_file_count") != 52
    ):
        raise ValueError("external frozen shard policy or artifact count drifted")
    unsigned_binding = dict(descriptor)
    supplied_binding_sha256 = unsigned_binding.pop("binding_sha256", None)
    if supplied_binding_sha256 != sha256_bytes(canonical_json_bytes(unsigned_binding)):
        raise ValueError("external frozen shard binding self hash mismatch")

    target_noskill = next(
        (item for item in target_shards if getattr(item, "config") == "noskill"),
        None,
    )
    if target_noskill is None:
        raise ValueError("target launch lacks NoSkill shard")
    if (
        descriptor.get("target_shard_id") != getattr(target_noskill, "shard_id")
        or descriptor.get("target_shard_sha256")
        != getattr(target_noskill, "shard_sha256")
        or descriptor.get("accepted_batch_id")
        != getattr(target_noskill, "accepted_batch_id")
    ):
        raise ValueError("external frozen shard target binding differs")

    source_execution_root = _resolve_repository_path(
        str(descriptor["source_execution_root"])
    )
    source_control_path = source_execution_root / "execution-control.json"
    source_control, source_control_bytes = _load_canonical_object(
        source_control_path, label="external source execution control"
    )
    if (
        sha256_bytes(source_control_bytes)
        != descriptor["source_execution_control_file_sha256"]
        or source_control.get("control_sha256")
        != descriptor["source_execution_control_sha256"]
    ):
        raise ValueError("external source execution control digest mismatch")
    unsigned_source_control = dict(source_control)
    supplied_source_control_sha = unsigned_source_control.pop("control_sha256", None)
    if supplied_source_control_sha != _hash(unsigned_source_control):
        raise ValueError("external source execution control self hash mismatch")
    source_key = read_stable_regular_file(
        source_execution_root / "blinding-key.bin",
        label="external source blinding key",
        max_bytes=32,
    )
    if len(source_key) != 32 or sha256_bytes(source_key) != source_control.get(
        "blinding_key_sha256"
    ):
        raise ValueError("external source execution blinding key mismatch")

    source_launch_root = _resolve_repository_path(str(descriptor["source_launch_root"]))
    if _resolve_repository_path(str(source_control.get("launch_root"))) != (
        source_launch_root
    ):
        raise ValueError("external source launch root differs from source control")
    source_launch_plan_bytes = read_stable_regular_file(
        source_launch_root / "launch-plan.json",
        label="external source launch plan",
    )
    if (
        sha256_bytes(source_launch_plan_bytes)
        != descriptor["source_launch_plan_file_sha256"]
        or source_control.get("launch_plan_file_sha256")
        != descriptor["source_launch_plan_file_sha256"]
    ):
        raise ValueError("external source launch-plan file digest mismatch")
    source_launch = load_portfolio_launch_package(
        source_launch_root,
        expected_plan_file_sha256=str(descriptor["source_launch_plan_file_sha256"]),
    )
    if (
        source_launch.plan.launch_plan_sha256 != descriptor["source_launch_plan_sha256"]
        or source_control.get("launch_plan_sha256")
        != descriptor["source_launch_plan_sha256"]
    ):
        raise ValueError("external source launch content binding differs")

    source_shard = next(
        (
            item
            for item in source_launch.plan.shards
            if item.shard_id == descriptor["source_shard_id"]
        ),
        None,
    )
    if (
        source_shard is None
        or source_shard.config != "noskill"
        or source_shard.shard_sha256 != descriptor["source_shard_sha256"]
        or source_shard.output_relpath != descriptor["source_shard_output_relpath"]
    ):
        raise ValueError("external source NoSkill shard binding differs")
    source_members = tuple(
        item
        for item in source_launch.instances
        if item.shard_id == source_shard.shard_id
    )

    source_runtime_root = _resolve_repository_path(
        str(source_control.get("runtime_root"))
    )
    if source_runtime_root != _resolve_repository_path(
        str(descriptor["source_runtime_root"])
    ):
        raise ValueError("external source runtime root differs from source control")
    source_runtime_raw, source_runtime_bytes = _load_canonical_object(
        source_runtime_root / "runtime-lock.json",
        label="external source runtime lock",
    )
    if (
        sha256_bytes(source_runtime_bytes)
        != descriptor["source_runtime_lock_file_sha256"]
        or source_control.get("runtime_lock_file_sha256")
        != descriptor["source_runtime_lock_file_sha256"]
        or source_runtime_raw.get("runtime_lock_sha256")
        != descriptor["source_runtime_lock_sha256"]
        or source_control.get("runtime_lock_sha256")
        != descriptor["source_runtime_lock_sha256"]
    ):
        raise ValueError("external source runtime-lock binding differs")
    unsigned_source_runtime = dict(source_runtime_raw)
    supplied_source_runtime_sha = unsigned_source_runtime.pop(
        "runtime_lock_sha256", None
    )
    if supplied_source_runtime_sha != _hash(unsigned_source_runtime):
        raise ValueError("external source runtime-lock self hash mismatch")

    source_shard_root = source_execution_root / source_shard.output_relpath
    source_summary, source_summary_bytes = _load_canonical_object(
        source_shard_root / "shard-summary.json",
        label="external source NoSkill summary",
    )
    source_audit, source_audit_bytes = _load_canonical_object(
        source_shard_root / "shard-audit.json",
        label="external source NoSkill audit",
    )
    if (
        sha256_bytes(source_summary_bytes)
        != descriptor["source_shard_summary_file_sha256"]
        or source_summary.get("summary_sha256")
        != descriptor["source_shard_summary_sha256"]
        or sha256_bytes(source_audit_bytes)
        != descriptor["source_shard_audit_file_sha256"]
        or source_audit.get("audit_sha256") != descriptor["source_shard_audit_sha256"]
    ):
        raise ValueError("external source NoSkill summary/audit digest mismatch")

    inventory = _shard_artifact_inventory(
        source_execution_root,
        shard=source_shard,
        members=source_members,
    )
    inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
    if (
        len(inventory) != descriptor["artifact_file_count"]
        or inventory != descriptor["artifact_file_sha256s"]
        or inventory_sha256 != descriptor["artifact_set_sha256"]
    ):
        raise ValueError("external source NoSkill artifact set differs")
    compatibility_payload = [
        {
            field: getattr(item, field)
            for field in (
                "query_id",
                "query_sha256",
                "public_input_sha256",
                "asset_id",
                "image_path",
                "image_sha256",
            )
        }
        for item in target_members
    ]
    if sha256_bytes(canonical_json_bytes(compatibility_payload)) != descriptor.get(
        "query_public_image_compatibility_sha256"
    ):
        raise ValueError("external NoSkill compatibility digest differs")
    if sha256_bytes(source_key) != descriptor.get("blinding_key_sha256"):
        raise ValueError("external NoSkill blinding-key binding differs")

    source_value = _ShardArtifactSource(
        execution_root=source_execution_root,
        launch=source_launch,
        shard=source_shard,
        members=source_members,
        control=source_control,
        runtime_root=source_runtime_root,
        runtime_raw=source_runtime_raw,
        external=True,
    )
    source_contract = _source_request_contract(source=source_value)
    comparison_contract = descriptor.get("comparison_contract_payload")
    if not isinstance(comparison_contract, dict):
        raise ValueError("external comparison contract must be one object")
    if (
        sha256_bytes(canonical_json_bytes(comparison_contract))
        != descriptor["comparison_contract_sha256"]
    ):
        raise ValueError("external comparison contract self hash mismatch")
    source_projection = _comparison_contract_payload(
        launch=source_launch,
        control=source_control,
        runtime_raw=source_runtime_raw,
        source_request_contract=source_contract,
        require_noskill_runtime_binding=True,
    )
    if comparison_contract != source_projection:
        raise ValueError("external comparison contract differs from source projection")

    compatibility_violations = _member_compatibility_violations(
        target_members=target_members,
        source_members=source_members,
    )
    evidence = {
        "execution_mode": "external_frozen_reference",
        "target_shard_id": descriptor["target_shard_id"],
        "source_execution_root": str(descriptor["source_execution_root"]),
        "source_execution_control_file_sha256": descriptor[
            "source_execution_control_file_sha256"
        ],
        "source_execution_control_sha256": descriptor[
            "source_execution_control_sha256"
        ],
        "source_launch_plan_file_sha256": descriptor["source_launch_plan_file_sha256"],
        "source_launch_plan_sha256": descriptor["source_launch_plan_sha256"],
        "source_runtime_lock_file_sha256": descriptor[
            "source_runtime_lock_file_sha256"
        ],
        "source_runtime_lock_sha256": descriptor["source_runtime_lock_sha256"],
        "source_shard_id": source_shard.shard_id,
        "source_artifact_count": len(inventory),
        "source_artifact_set_sha256": inventory_sha256,
        "source_audit_sha256": source_audit["audit_sha256"],
        "source_observed_cost_cny": float(source_audit["dashscope_observed_cost_cny"]),
        "comparison_contract_sha256": descriptor["comparison_contract_sha256"],
        "binding_sha256": descriptor["binding_sha256"],
    }
    return (
        _ShardArtifactSource(
            execution_root=source_value.execution_root,
            launch=source_value.launch,
            shard=source_value.shard,
            members=source_value.members,
            control=source_value.control,
            runtime_root=source_value.runtime_root,
            runtime_raw=source_value.runtime_raw,
            source_request_contract=source_contract,
            external=True,
        ),
        evidence,
        compatibility_violations,
    )


def _load_internal_artifact_alias_sources(
    *,
    control: Mapping[str, object],
    launch: object,
    execution_root: Path,
    runtime_root: Path,
    shards: Sequence[object],
    members_by_config: Mapping[str, Sequence[object]],
) -> tuple[dict[str, _ShardArtifactSource], list[dict[str, object]]]:
    """Resolve local zero-call rollback aliases to their physical source shards."""

    aliases = control.get("execution_artifact_aliases", [])
    if not isinstance(aliases, list) or any(
        not isinstance(item, dict) for item in aliases
    ):
        raise ValueError("execution artifact aliases must be a list of objects")
    if not aliases:
        return {}, []
    runtime_raw, _ = _load_canonical_object(
        runtime_root / "runtime-lock.json",
        label="Portfolio runtime lock for artifact aliases",
    )
    if (
        runtime_raw.get("execution_artifact_aliases") != aliases
        or runtime_raw.get("execution_artifact_alias_policy_version")
        != control.get("execution_artifact_alias_policy_version")
        or runtime_raw.get("execution_artifact_alias_provider_model_call_count") != 0
        or control.get("execution_artifact_alias_provider_model_call_count") != 0
    ):
        raise ValueError("execution artifact aliases differ from runtime/control")

    shards_by_config = {str(getattr(item, "config")): item for item in shards}
    sources: dict[str, _ShardArtifactSource] = {}
    evidence: list[dict[str, object]] = []
    for alias in aliases:
        target_config = alias.get("target_config")
        source_config = alias.get("source_config")
        if target_config not in shards_by_config:
            continue
        target = shards_by_config[target_config]
        source = shards_by_config.get(str(source_config))
        unsigned_alias = dict(alias)
        supplied_alias_sha256 = unsigned_alias.pop("alias_sha256", None)
        if (
            source is None
            or supplied_alias_sha256 != _hash(unsigned_alias)
            or alias.get("provider_model_call_count") != 0
            or alias.get("reuse_scope") != "assistant_and_evaluator_query_artifacts"
            or alias.get("rejected_candidate_use") != "diagnostic_only"
            or alias.get("source_bank_sha256") != alias.get("target_bank_sha256")
            or alias.get("source_bank_file_sha256")
            != alias.get("target_bank_file_sha256")
            or getattr(source, "accepted_batch_id")
            != getattr(target, "accepted_batch_id")
            or getattr(source, "query_ids") != getattr(target, "query_ids")
            or getattr(source, "shard_id")
            not in getattr(launch, "state").completed_shard_ids
            or getattr(target, "shard_id")
            not in getattr(launch, "state").completed_shard_ids
        ):
            raise ValueError("execution artifact alias is not a completed exact reuse")

        target_root = execution_root / str(getattr(target, "output_relpath"))
        entries = tuple(target_root.iterdir()) if target_root.is_dir() else ()
        if target_root.is_symlink() or {item.name for item in entries} != {
            "artifact-alias.json"
        }:
            raise ValueError("artifact alias target contains unexpected files")
        receipt, receipt_bytes = _load_canonical_object(
            target_root / "artifact-alias.json",
            label=f"artifact alias receipt {target_config}",
        )
        unsigned_receipt = dict(receipt)
        supplied_receipt_sha256 = unsigned_receipt.pop("alias_receipt_sha256", None)
        source_root = execution_root / str(getattr(source, "output_relpath"))
        summary, summary_bytes = _load_canonical_object(
            source_root / "shard-summary.json",
            label=f"artifact alias source summary {source_config}",
        )
        audit, audit_bytes = _load_canonical_object(
            source_root / "shard-audit.json",
            label=f"artifact alias source audit {source_config}",
        )
        if (
            supplied_receipt_sha256 != _hash(unsigned_receipt)
            or receipt.get("kind") != "portfolio-shard-artifact-alias"
            or receipt.get("target_shard_id") != getattr(target, "shard_id")
            or receipt.get("target_config") != target_config
            or receipt.get("source_shard_id") != getattr(source, "shard_id")
            or receipt.get("source_config") != source_config
            or receipt.get("query_ids") != list(getattr(target, "query_ids"))
            or receipt.get("treatment_alias_sha256") != supplied_alias_sha256
            or receipt.get("provider_model_call_count") != 0
            or receipt.get("runtime_lock_sha256") != control.get("runtime_lock_sha256")
            or receipt.get("launch_plan_sha256") != control.get("launch_plan_sha256")
            or receipt.get("source_shard_summary_file_sha256")
            != sha256_bytes(summary_bytes)
            or receipt.get("source_shard_summary_sha256")
            != summary.get("summary_sha256")
            or receipt.get("source_shard_audit_file_sha256")
            != sha256_bytes(audit_bytes)
            or receipt.get("source_shard_audit_sha256") != audit.get("audit_sha256")
        ):
            raise ValueError("artifact alias receipt differs from its source")

        source_members = tuple(members_by_config[str(source_config)])
        if tuple(getattr(item, "query_id") for item in source_members) != tuple(
            getattr(item, "query_id") for item in members_by_config[str(target_config)]
        ):
            raise ValueError("artifact alias source query membership differs")
        sources[str(target_config)] = _ShardArtifactSource(
            execution_root=execution_root,
            launch=launch,
            shard=source,
            members=source_members,
            control=control,
            runtime_root=runtime_root,
            runtime_raw=runtime_raw,
            external=False,
            artifact_alias=receipt,
        )
        evidence.append(
            {
                "target_config": target_config,
                "target_shard_id": getattr(target, "shard_id"),
                "source_config": source_config,
                "source_shard_id": getattr(source, "shard_id"),
                "alias_sha256": supplied_alias_sha256,
                "alias_receipt_sha256": supplied_receipt_sha256,
                "alias_receipt_file_sha256": sha256_bytes(receipt_bytes),
                "provider_model_call_count": 0,
            }
        )
    return sources, evidence


def _failure_audit(
    *,
    status: str,
    requested_batch_id: str | None,
    blockers: Sequence[str],
) -> dict[str, object]:
    return _content_address(
        {
            "schema_version": 1,
            "kind": "portfolio-batch-cross-config-audit",
            "track": "portfolio",
            "formal_eligible": False,
            "status": status,
            "batch_id": requested_batch_id,
            "query_count": 0,
            "row_count": 0,
            "config_order": list(MAIN_CONFIG_ORDER),
            "blockers": sorted(set(blockers)),
            "review_flags": [],
        }
    )


def _select_batch_shards(
    shards: Sequence[object],
    *,
    batch_id: str | None,
    shard_ids: Sequence[str] | None,
) -> tuple[str, tuple[object, ...], tuple[object, ...]]:
    by_id = {str(getattr(item, "shard_id")): item for item in shards}
    if shard_ids:
        if len(shard_ids) != 5 or len(set(shard_ids)) != 5:
            raise ValueError(
                "explicit shard selection requires exactly five unique IDs"
            )
        try:
            selected = tuple(by_id[item] for item in shard_ids)
        except KeyError as error:
            raise ValueError(
                f"selected shard is absent from launch: {error.args[0]}"
            ) from error
        batch_ids = {str(getattr(item, "accepted_batch_id")) for item in selected}
        if len(batch_ids) != 1:
            raise ValueError("explicit shards do not belong to one accepted batch")
        selected_batch = batch_ids.pop()
        if batch_id is not None and batch_id != selected_batch:
            raise ValueError("batch ID differs from explicit shard selection")
    elif batch_id is not None:
        selected_batch = batch_id
        selected = tuple(
            item
            for item in shards
            if str(getattr(item, "accepted_batch_id")) == batch_id
        )
    else:
        if len(shards) < 5:
            raise ValueError("launch does not contain a first five-shard batch")
        selected = tuple(shards[:5])
        batch_ids = {str(getattr(item, "accepted_batch_id")) for item in selected}
        if len(batch_ids) != 1:
            raise ValueError("the first five launch shards do not form one batch")
        selected_batch = batch_ids.pop()

    launch_sequence = tuple(
        item
        for item in shards
        if str(getattr(item, "accepted_batch_id")) == selected_batch
    )
    selected_ids = {str(getattr(item, "shard_id")) for item in selected}
    launch_sequence_ids = {str(getattr(item, "shard_id")) for item in launch_sequence}
    positions = [
        index
        for index, item in enumerate(shards)
        if str(getattr(item, "accepted_batch_id")) == selected_batch
    ]
    if (
        len(launch_sequence) != 5
        or selected_ids != launch_sequence_ids
        or positions != list(range(positions[0], positions[0] + 5))
    ):
        raise ValueError(
            "selected batch is not the exact contiguous five-shard launch sequence"
        )

    by_config = {str(getattr(item, "config")): item for item in selected}
    if len(selected) != 5 or set(by_config) != set(MAIN_CONFIG_ORDER):
        raise ValueError(
            "selected batch does not contain exactly the five main configs"
        )
    ordered = tuple(by_config[config] for config in MAIN_CONFIG_ORDER)
    if any(int(getattr(item, "query_count")) != 25 for item in ordered):
        raise ValueError("selected batch contains a non-25-query shard")
    return selected_batch, ordered, launch_sequence


def _paired_member_violations(
    members_by_config: Mapping[str, Sequence[object]],
) -> list[str]:
    violations: list[str] = []
    expected_ids = tuple(
        str(getattr(item, "query_id")) for item in members_by_config["noskill"]
    )
    for config in MAIN_CONFIG_ORDER:
        ids = tuple(
            str(getattr(item, "query_id")) for item in members_by_config[config]
        )
        if ids != expected_ids:
            violations.append(f"paired query order differs for {config}")
    fields = (
        "query_ordinal",
        "query_sha256",
        "public_input_sha256",
        "asset_id",
        "image_path",
        "asset_token",
        "asset_binding_sha256",
        "image_sha256",
        "accepted_batch_id",
    )
    for index, query_id in enumerate(expected_ids):
        reference = members_by_config["noskill"][index]
        for config in MAIN_CONFIG_ORDER[1:]:
            candidate = members_by_config[config][index]
            drifted = [
                field
                for field in fields
                if getattr(reference, field, None) != getattr(candidate, field, None)
            ]
            if drifted:
                violations.append(
                    f"paired input drift for {query_id}/{config}: {','.join(drifted)}"
                )
    return violations


def _load_assignments(
    path: Path,
    expected_file_sha256: str,
) -> dict[tuple[str, str, str], CapabilityAssignment]:
    content = read_stable_regular_file(
        path,
        label="Portfolio capability assignments",
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise ValueError("capability-assignment file digest mismatch")
    rows = parse_canonical_jsonl(content, label="Portfolio capability assignments")
    assignments = [
        CapabilityAssignment.model_validate(item, strict=True) for item in rows
    ]
    return _index_assignments(assignments)


def _index_assignments(
    assignments: Sequence[CapabilityAssignment],
) -> dict[tuple[str, str, str], CapabilityAssignment]:
    by_key = {
        (item.asset_id, item.image_path, item.canonical_intent): item
        for item in assignments
    }
    if len(by_key) != len(assignments):
        raise ValueError("capability assignments contain duplicate composite keys")
    if len({item.assignment_id for item in assignments}) != len(assignments):
        raise ValueError("capability assignments contain duplicate assignment IDs")
    return by_key


def _compare_operator_maps(
    operators_by_config: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[dict[str, list[str]], list[str]]:
    violations: list[str] = []
    reference = operators_by_config.get("llm_static", {})
    for config in MAIN_CONFIG_ORDER[2:]:
        if operators_by_config.get(config, {}) != reference:
            violations.append(
                f"{config} Bank operator sets differ from llm_static by capability"
            )
    return (
        {
            capability: list(operators)
            for capability, operators in sorted(reference.items())
        },
        violations,
    )


def _stage_boundary_check(
    banks: Mapping[str, StaticBankArtifact],
) -> dict[str, object]:
    violations = list(evolution_bank_boundary_violations(banks))
    return {
        "policy_version": "s2-description-s3-body-only-v1",
        "violation_count": len(violations),
        "violations": violations,
    }


def _load_banks(
    runtime_root: Path,
    runtime_lock: Mapping[str, object],
) -> tuple[
    dict[str, StaticBankArtifact],
    dict[str, dict[str, object]],
    dict[str, list[str]],
]:
    locked = runtime_lock.get("bank_sha256s")
    if not isinstance(locked, dict):
        raise ValueError("runtime lock has no Bank identity mapping")
    banks: dict[str, StaticBankArtifact] = {}
    contexts: dict[str, dict[str, object]] = {}
    violations: list[str] = []
    operators_by_config: dict[str, dict[str, tuple[str, ...]]] = {}

    for config in MAIN_CONFIG_ORDER[1:]:
        path = runtime_root / f"bank-{config}.json"
        bank = StaticBankArtifact.model_validate_json(
            read_stable_regular_file(path, label=f"{config} Bank"),
            strict=True,
        )
        banks[config] = bank
        if bank.bank_sha256 != locked.get(config):
            violations.append(f"{config} Bank hash differs from runtime lock")
        if bank.tool_registry_sha256 != runtime_lock.get(
            "tool_registry_sha256"
        ) or bank.tool_registry_runtime_sha256 != runtime_lock.get(
            "tool_registry_runtime_sha256"
        ):
            violations.append(f"{config} Bank registry binding differs")

        by_slug = {item.slug: item for item in bank.skills}
        by_capability = {item.capability_id: item for item in bank.skills}
        if tuple(sorted(by_capability)) != _EXPECTED_CAPABILITIES:
            violations.append(
                f"{config} Bank does not cover the exact six capabilities"
            )
        contexts[config] = {
            "by_slug": by_slug,
            "by_capability": by_capability,
        }
        operators = {
            capability: tuple(skill.operators)
            for capability, skill in by_capability.items()
        }
        operators_by_config[config] = operators

    operator_matrix, operator_violations = _compare_operator_maps(operators_by_config)
    violations.extend(operator_violations)
    stage_boundary_check = _stage_boundary_check(banks)
    stage_boundary_violations = list(stage_boundary_check["violations"])
    violations.extend(stage_boundary_violations)
    return (
        banks,
        contexts,
        {
            "violations": sorted(set(violations)),
            "operator_matrix": operator_matrix,
            "stage_boundary_check": stage_boundary_check,
        },
    )


def _tool_gate_violations(
    *,
    query_id: str,
    config: str,
    response: object,
    assignment: CapabilityAssignment | None,
    bank_context: Mapping[str, object] | None,
) -> list[str]:
    violations: list[str] = []
    selected_capability = getattr(response, "selected_capability")
    skill_slug = getattr(response, "skill_slug")
    error_code = getattr(response, "error_code")
    tool_trace = getattr(response, "tool_trace")
    if config == "noskill":
        if selected_capability is not None or skill_slug is not None:
            violations.append(f"NoSkill retained Skill identity: {query_id}")
        return violations
    if error_code is not None:
        return violations
    assert bank_context is not None
    by_slug = bank_context["by_slug"]
    skill = by_slug.get(skill_slug)
    if skill is None or skill.capability_id != selected_capability:
        violations.append(f"route skill/capability mismatch: {query_id}/{config}")
        return violations
    allowed = set(skill.operators)
    used = {item.tool_name for item in tool_trace}
    if not used <= allowed:
        violations.append(f"tool trace exceeds selected operators: {query_id}/{config}")
    if (
        assignment is not None
        and skill.capability_id == assignment.canonical_capability
        and tuple(skill.operators) != tuple(assignment.allowed_tools)
    ):
        violations.append(f"Bank operators differ from assignment: {query_id}/{config}")
    return violations


def _completion_snapshot(
    execution_root: Path,
    shards: Sequence[object],
    members_by_config: Mapping[str, Sequence[object]],
    *,
    sources_by_config: Mapping[str, _ShardArtifactSource] | None = None,
) -> tuple[dict[str, object], list[str]]:
    snapshot: dict[str, object] = {}
    blockers: list[str] = []
    for shard in shards:
        config = str(getattr(shard, "config"))
        source = None if sources_by_config is None else sources_by_config.get(config)
        members = members_by_config[config] if source is None else source.members
        physical_root = execution_root if source is None else source.execution_root
        physical_shard = shard if source is None else source.shard
        expected_assistant = {
            Path(str(getattr(item, "assistant_output_relpath"))).name
            for item in members
        }
        expected_final = {
            Path(str(getattr(item, "final_output_relpath"))).name for item in members
        }
        shard_root = physical_root / str(getattr(physical_shard, "output_relpath"))
        assistant_dir = shard_root / "assistant"
        final_dir = shard_root / "final"
        actual_assistant = (
            {path.name for path in assistant_dir.glob("*.json")}
            if assistant_dir.is_dir()
            else set()
        )
        actual_final = (
            {path.name for path in final_dir.glob("*.json")}
            if final_dir.is_dir()
            else set()
        )
        summary_exists = (shard_root / "shard-summary.json").is_file()
        audit_exists = (shard_root / "shard-audit.json").is_file()
        snapshot[config] = {
            "shard_id": str(getattr(shard, "shard_id")),
            "artifact_source": (
                "external_frozen_reference"
                if source is not None and source.external
                else "accepted_parent_artifact_alias"
                if source is not None and source.artifact_alias is not None
                else "local_execution"
            ),
            "physical_shard_id": str(getattr(physical_shard, "shard_id")),
            "assistant_checkpoint_count": len(actual_assistant),
            "final_checkpoint_count": len(actual_final),
            "summary_present": summary_exists,
            "audit_present": audit_exists,
            "audit_sha256": None,
        }
        if actual_assistant != expected_assistant:
            blockers.append(f"{config} Assistant checkpoint set is incomplete or extra")
        if actual_final != expected_final:
            blockers.append(f"{config} Final checkpoint set is incomplete or extra")
        if not summary_exists:
            blockers.append(f"{config} shard summary is absent")
        if not audit_exists:
            blockers.append(f"{config} shard audit is absent")
    return snapshot, blockers


def _validate_summary(
    path: Path,
    *,
    shard_id: str,
    config: str,
) -> str:
    raw, _ = _load_canonical_object(path, label=f"{config} shard summary")
    supplied = raw.get("summary_sha256")
    unsigned = dict(raw)
    unsigned.pop("summary_sha256", None)
    if (
        supplied != _hash(unsigned)
        or raw.get("status") != "complete"
        or raw.get("shard_id") != shard_id
        or raw.get("config") != config
        or raw.get("query_count") != 25
    ):
        raise ValueError(f"{config} shard summary is inconsistent")
    return str(supplied)


def _validate_shard_audit(
    path: Path,
    *,
    shard_id: str,
    config: str,
    runtime_lock_sha256: str,
    launch_plan_sha256: str,
    shard_summary_sha256: str,
    runtime_lock: Mapping[str, object] | None = None,
) -> str:
    raw, _ = _load_canonical_object(path, label=f"{config} shard audit")
    supplied = raw.get("audit_sha256")
    unsigned = dict(raw)
    unsigned.pop("audit_sha256", None)
    result_contract_invalid = False
    if runtime_lock is not None:
        result_schema_version = runtime_lock.get("final_judge_result_schema_version")
        retry_count_fields = (
            "judge_model_calls",
            "judge_captured_response_count",
            "judge_retried_row_count",
            "judge_initial_empty_response_count",
            "judge_reasoning_present_response_count",
            "judge_reasoning_tokens_reported_total",
            "judge_reasoning_tokens_unavailable_response_count",
            "judge_reasoning_bytes_total",
        ) + (
            ("judge_initial_invalid_json_response_count",)
            if result_schema_version in {7, 8, 9, 10}
            else ()
        )
        reasoning_receipt_fields = {
            "query_id",
            "attempts",
            "initial_empty_response",
            "initial_reasoning_present",
            "initial_reasoning_tokens",
            "initial_reasoning_bytes",
            "initial_reasoning_sha256",
            "terminal_response_captured",
            "terminal_reasoning_present",
            "terminal_reasoning_tokens",
            "terminal_reasoning_bytes",
            "terminal_reasoning_sha256",
        }
        if result_schema_version in {7, 8, 9, 10}:
            reasoning_receipt_fields.add("initial_retry_reason")
        response_receipts = raw.get("judge_response_receipts")
        receipts_invalid = (
            not isinstance(response_receipts, list)
            or len(response_receipts) != raw.get("judge_invoked_count")
            or any(
                not isinstance(item, dict) or set(item) != reasoning_receipt_fields
                for item in (
                    response_receipts if isinstance(response_receipts, list) else ()
                )
            )
        )
        result_contract_invalid = (
            raw.get("schema_version") != result_schema_version
            or any(
                raw.get(field) != runtime_lock.get(field)
                for field in _ACTIVE_FINAL_RESULT_LOCK
            )
            or not isinstance(
                raw.get("judge_card_requirement_guard_adjusted_count"), int
            )
            or any(type(raw.get(field)) is not int for field in retry_count_fields)
            or receipts_invalid
        )
    if (
        supplied != _hash(unsigned)
        or raw.get("kind") != "portfolio-shard-audit"
        or raw.get("shard_id") != shard_id
        or raw.get("config") != config
        or raw.get("query_count") != 25
        or raw.get("runtime_lock_sha256") != runtime_lock_sha256
        or raw.get("launch_plan_sha256") != launch_plan_sha256
        or raw.get("shard_summary_sha256") != shard_summary_sha256
        or result_contract_invalid
    ):
        raise ValueError(f"{config} shard audit is inconsistent")
    return str(supplied)


def _noskill_freeze_evidence(
    *,
    observed_sha256: str,
    expected_sha256: str | None,
) -> tuple[dict[str, object], list[str]]:
    if expected_sha256 is not None and not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(
            "expected NoSkill shard-audit digest must be lowercase SHA-256"
        )
    matches = None if expected_sha256 is None else observed_sha256 == expected_sha256
    violations = (
        []
        if matches is not False
        else ["NoSkill shard audit differs from the previously observed freeze digest"]
    )
    return (
        {
            "expected_audit_sha256": expected_sha256,
            "observed_audit_sha256": observed_sha256,
            "matches": matches,
            "evidence_scope": "digest_match_only_not_wall_clock_order_proof",
        },
        violations,
    )


def _validate_fixed_zero(
    path: Path,
    *,
    query_id: str,
    error_code: str,
) -> str:
    raw, content = _load_canonical_object(path, label=f"fixed-zero {query_id}")
    supplied = raw.get("result_sha256")
    unsigned = dict(raw)
    unsigned.pop("result_sha256", None)
    if (
        supplied != _hash(unsigned)
        or raw.get("kind") != "portfolio-final-fixed-zero"
        or raw.get("query_id") != query_id
        or raw.get("assistant_error_code") != error_code
        or raw.get("j_project") != 0.0
    ):
        raise ValueError(f"fixed-zero result differs from Assistant error: {query_id}")
    return sha256_bytes(content)


def _summarize_observations(
    rows: Sequence[_RowObservation],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    summaries: dict[str, object] = {}
    by_query: dict[str, dict[str, _RowObservation]] = defaultdict(dict)
    for row in rows:
        by_query[row.query_id][row.config] = row

    for config in MAIN_CONFIG_ORDER:
        selected = [item for item in rows if item.config == config]
        assistant_errors = Counter(
            item.assistant_error_code
            for item in selected
            if item.assistant_error_code is not None
        )
        judge_statuses = Counter(item.judge_status for item in selected)
        shapes = Counter(
            item.judge_shape for item in selected if item.judge_shape is not None
        )
        tool_errors = [
            (item.query_id, tool_name, error_code)
            for item in selected
            for tool_name, status, error_code in item.tool_trace
            if status != "success"
        ]
        tool_error_ids = list(dict.fromkeys(item[0] for item in tool_errors))
        tool_error_codes = Counter(
            error_code if error_code is not None else "missing_error_code"
            for _, _, error_code in tool_errors
        )
        tool_error_tools = Counter(tool_name for _, tool_name, _ in tool_errors)
        successful_routes = [
            item
            for item in selected
            if item.config != "noskill"
            and item.assistant_error_code is None
            and item.route_acceptable is not None
        ]
        summaries[config] = {
            "row_count": len(selected),
            "assistant_success_count": sum(
                item.assistant_error_code is None for item in selected
            ),
            "assistant_error_counts": dict(sorted(assistant_errors.items())),
            "assistant_model_calls": sum(item.model_call_count for item in selected),
            "assistant_tool_calls": sum(len(item.tool_trace) for item in selected),
            "assistant_tool_error_count": len(tool_errors),
            "assistant_tool_error_query_count": len(tool_error_ids),
            "assistant_tool_error_ids": tool_error_ids,
            "assistant_tool_error_counts_by_code": dict(
                sorted(tool_error_codes.items())
            ),
            "assistant_tool_error_counts_by_tool": dict(
                sorted(tool_error_tools.items())
            ),
            "card_policy_compliant_count": sum(
                item.card_policy_compliant for item in selected
            ),
            "card_policy_violation_count": sum(
                not item.card_policy_compliant for item in selected
            ),
            "card_policy_violation_ids": [
                item.query_id for item in selected if not item.card_policy_compliant
            ],
            "required_missing_card_ids": [
                item.query_id
                for item in selected
                if item.card_requirement == "required" and item.visible_card_count == 0
            ],
            "forbidden_card_ids": [
                item.query_id
                for item in selected
                if item.card_requirement == "forbidden" and item.visible_card_count != 0
            ],
            "assistant_input_tokens": sum(
                item.assistant_input_tokens for item in selected
            ),
            "assistant_output_tokens": sum(
                item.assistant_output_tokens for item in selected
            ),
            "successful_route_count": len(successful_routes),
            "acceptable_route_count": sum(
                item.route_acceptable is True for item in successful_routes
            ),
            "unacceptable_route_count": sum(
                item.route_acceptable is False for item in successful_routes
            ),
            "judge_status_counts": dict(sorted(judge_statuses.items())),
            "judge_shape_counts": dict(sorted(shapes.items())),
            "judge_input_tokens": sum(item.judge_input_tokens for item in selected),
            "judge_output_tokens": sum(item.judge_output_tokens for item in selected),
            "judge_model_calls": sum(item.judge_attempts for item in selected),
            "judge_captured_response_count": sum(
                int(item.judge_initial_retry_reason is not None)
                + int(item.judge_terminal_response_captured)
                for item in selected
            ),
            "judge_retried_row_count": sum(
                item.judge_attempts == 2 for item in selected
            ),
            "judge_initial_empty_response_count": sum(
                item.judge_initial_empty_response for item in selected
            ),
            "judge_initial_invalid_json_response_count": sum(
                item.judge_initial_retry_reason == "invalid_judge_json"
                for item in selected
            ),
            "judge_reasoning_present_response_count": sum(
                item.judge_initial_reasoning_present is True for item in selected
            )
            + sum(item.judge_terminal_reasoning_present is True for item in selected),
            "judge_reasoning_tokens_reported_total": sum(
                0
                if item.judge_initial_reasoning_tokens is None
                else item.judge_initial_reasoning_tokens
                for item in selected
            )
            + sum(
                0
                if item.judge_terminal_reasoning_tokens is None
                else item.judge_terminal_reasoning_tokens
                for item in selected
            ),
            "judge_reasoning_tokens_unavailable_response_count": sum(
                item.judge_initial_retry_reason is not None
                and item.judge_initial_reasoning_tokens is None
                for item in selected
            )
            + sum(
                item.judge_terminal_response_captured
                and item.judge_terminal_reasoning_tokens is None
                for item in selected
            ),
            "judge_reasoning_bytes_total": sum(
                0
                if item.judge_initial_reasoning_bytes is None
                else item.judge_initial_reasoning_bytes
                for item in selected
            )
            + sum(
                0
                if item.judge_terminal_reasoning_bytes is None
                else item.judge_terminal_reasoning_bytes
                for item in selected
            ),
            "mean_j_project_all_rows": (
                sum(item.j_project for item in selected) / len(selected)
                if selected
                else 0.0
            ),
        }

    skilled_errors: list[dict[str, object]] = []
    for query_id in sorted(by_query):
        paired = by_query[query_id]
        noskill = paired.get("noskill")
        for config in MAIN_CONFIG_ORDER[1:]:
            row = paired.get(config)
            if row is None or row.assistant_error_code is None:
                continue
            skilled_errors.append(
                {
                    "query_id": query_id,
                    "config": config,
                    "canonical_capability": row.canonical_capability,
                    "error_code": row.assistant_error_code,
                    "receipt_outcome": row.receipt_outcome,
                    "noskill_error_code": (
                        None if noskill is None else noskill.assistant_error_code
                    ),
                    "skilled_only_vs_noskill": (
                        noskill is not None and noskill.assistant_error_code is None
                    ),
                    "model_call_count": row.model_call_count,
                    "finish_reasons": list(row.finish_reasons),
                    "turn_count": row.turn_count,
                    "tool_trace": [
                        {
                            "tool_name": tool_name,
                            "status": status,
                            "error_code": error_code,
                        }
                        for tool_name, status, error_code in row.tool_trace
                    ],
                }
            )

    judge_anomalies = [
        {
            "query_id": row.query_id,
            "config": row.config,
            "status": row.judge_status,
            "error_code": row.judge_error_code,
            "finish_reason": row.judge_finish_reason,
            "raw_response_bytes": row.judge_raw_response_bytes,
            "tool_call_count": row.judge_tool_call_count,
            "attempts": row.judge_attempts,
            "initial_empty_response": row.judge_initial_empty_response,
            "initial_retry_reason": row.judge_initial_retry_reason,
            "terminal_response_captured": row.judge_terminal_response_captured,
            "initial_reasoning_present": row.judge_initial_reasoning_present,
            "initial_reasoning_tokens": row.judge_initial_reasoning_tokens,
            "initial_reasoning_bytes": row.judge_initial_reasoning_bytes,
            "initial_reasoning_sha256": row.judge_initial_reasoning_sha256,
            "terminal_reasoning_present": row.judge_terminal_reasoning_present,
            "terminal_reasoning_tokens": row.judge_terminal_reasoning_tokens,
            "terminal_reasoning_bytes": row.judge_terminal_reasoning_bytes,
            "terminal_reasoning_sha256": row.judge_terminal_reasoning_sha256,
            "j_project": row.j_project,
        }
        for row in sorted(rows, key=lambda item: (item.query_id, item.config))
        if row.judge_status not in {"scored", "not_invoked_assistant_error"}
    ]
    route_performance_observations = [
        {
            "query_id": row.query_id,
            "config": row.config,
            "canonical_capability": row.canonical_capability,
            "selected_capability": row.selected_capability,
            "skill_slug": row.skill_slug,
            "observation": "selected_capability_not_acceptable",
        }
        for row in sorted(rows, key=lambda item: (item.query_id, item.config))
        if row.route_acceptable is False
    ]
    judge_response_receipts = [
        {
            "query_id": row.query_id,
            "config": row.config,
            "attempts": row.judge_attempts,
            "initial_empty_response": row.judge_initial_empty_response,
            "initial_retry_reason": row.judge_initial_retry_reason,
            "terminal_response_captured": row.judge_terminal_response_captured,
            "initial_reasoning_present": row.judge_initial_reasoning_present,
            "initial_reasoning_tokens": row.judge_initial_reasoning_tokens,
            "initial_reasoning_bytes": row.judge_initial_reasoning_bytes,
            "initial_reasoning_sha256": row.judge_initial_reasoning_sha256,
            "terminal_reasoning_present": row.judge_terminal_reasoning_present,
            "terminal_reasoning_tokens": row.judge_terminal_reasoning_tokens,
            "terminal_reasoning_bytes": row.judge_terminal_reasoning_bytes,
            "terminal_reasoning_sha256": row.judge_terminal_reasoning_sha256,
        }
        for row in sorted(rows, key=lambda item: (item.query_id, item.config))
        if row.judge_attempts > 0
    ]
    return (
        summaries,
        skilled_errors,
        judge_anomalies,
        route_performance_observations,
        judge_response_receipts,
    )


def _route_comparison_state(row: _RowObservation) -> str:
    if row.assistant_error_code is None:
        return "routed_success"
    if row.model_call_count == 0:
        return "assistant_error_before_route_response"
    if row.tool_trace or row.model_call_count > 1:
        return "assistant_error_after_route_identity_cleared"
    return "assistant_error_after_route_response_without_validated_decision"


def _expected_turn_count(receipt: object) -> int:
    shared = getattr(receipt, "shared_route_reference", None)
    reserved_turns = 0 if shared is None else int(shared.reserved_turns)
    return max(1, len(getattr(receipt, "model_calls")) + reserved_turns)


def _legacy_s1s2_full_route_reuse_check(
    rows: Sequence[_RowObservation],
    *,
    runner_file_sha256: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    by_query: dict[str, dict[str, _RowObservation]] = defaultdict(dict)
    for row in rows:
        if row.config in {"s1s2", "full"}:
            by_query[row.query_id][row.config] = row

    details: list[dict[str, object]] = []
    violations = [
        "S1+S2 and Full do not persist a shared Stage2 route decision identity"
    ]
    semantic_match_count = 0
    semantic_mismatch_count = 0
    unavailable_count = 0
    separate_router_call_count = 0

    for query_id in sorted(by_query):
        paired = by_query[query_id]
        if set(paired) != {"s1s2", "full"}:
            violations.append(
                f"S1+S2/Full route pair is absent from audit rows: {query_id}"
            )
            continue
        s1s2 = paired["s1s2"]
        full = paired["full"]
        s1s2_state = _route_comparison_state(s1s2)
        full_state = _route_comparison_state(full)
        if s1s2_state == full_state == "routed_success":
            if s1s2.selected_capability == full.selected_capability:
                semantic_status = "matched"
                semantic_match_count += 1
            else:
                semantic_status = "mismatched"
                semantic_mismatch_count += 1
                violations.append(
                    f"S1+S2/Full semantic route decision differs: {query_id}"
                )
        else:
            semantic_status = "unverifiable_due_assistant_error"
            unavailable_count += 1
            violations.append(
                f"S1+S2/Full route decision is unavailable for comparison: {query_id}"
            )

        separate_calls = (
            s1s2.route_call_response_sha256 is not None
            and full.route_call_response_sha256 is not None
            and s1s2.route_call_response_sha256 != full.route_call_response_sha256
        )
        if separate_calls:
            separate_router_call_count += 1
        details.append(
            {
                "query_id": query_id,
                "s1s2": {
                    "state": s1s2_state,
                    "assistant_error_code": s1s2.assistant_error_code,
                    "selected_capability": s1s2.selected_capability,
                    "route_trace_sha256": s1s2.route_trace_sha256,
                    "route_call_response_sha256": (s1s2.route_call_response_sha256),
                },
                "full": {
                    "state": full_state,
                    "assistant_error_code": full.assistant_error_code,
                    "selected_capability": full.selected_capability,
                    "route_trace_sha256": full.route_trace_sha256,
                    "route_call_response_sha256": (full.route_call_response_sha256),
                },
                "semantic_decision_status": semantic_status,
                "separate_router_calls_observed": separate_calls,
                "shared_stage2_decision_identity": None,
            }
        )

    if separate_router_call_count:
        violations.append(
            "S1+S2 and Full independently invoked the router instead of reusing "
            "one Stage2 decision"
        )
    return (
        {
            "required_policy": "shared-stage2-route-decision-per-query",
            "runner_policy_evidence": {
                "runner_file_sha256": runner_file_sha256,
                "policy": "each-skilled-config-invokes-router-before-action",
                "first_model_call_interpretation": "route_call",
            },
            "shared_decision_identity_persisted": False,
            "query_count": len(details),
            "semantic_match_count": semantic_match_count,
            "semantic_mismatch_count": semantic_mismatch_count,
            "unverifiable_due_assistant_error_count": unavailable_count,
            "separate_router_call_count": separate_router_call_count,
            "rows": details,
        },
        violations,
    )


def _s1s2_full_route_reuse_check(
    rows: Sequence[_RowObservation],
    *,
    runner_file_sha256: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    paired_rows = [row for row in rows if row.config in {"s1s2", "full"}]
    if not any(row.shared_route_artifact_sha256 is not None for row in paired_rows):
        return _legacy_s1s2_full_route_reuse_check(
            rows, runner_file_sha256=runner_file_sha256
        )

    by_query: dict[str, dict[str, _RowObservation]] = defaultdict(dict)
    for row in paired_rows:
        by_query[row.query_id][row.config] = row
    details: list[dict[str, object]] = []
    violations: list[str] = []
    matched_count = 0
    terminal_count = 0
    mismatch_count = 0

    for query_id in sorted(by_query):
        pair = by_query[query_id]
        if set(pair) != {"s1s2", "full"}:
            violations.append(
                f"S1+S2/Full route pair is absent from audit rows: {query_id}"
            )
            continue
        s1s2 = pair["s1s2"]
        full = pair["full"]
        shared_identity = (
            s1s2.shared_route_artifact_sha256,
            s1s2.route_call_response_sha256,
            s1s2.shared_route_reserved_input_tokens,
            s1s2.shared_route_reserved_output_tokens,
            s1s2.shared_route_reserved_turns,
        )
        full_identity = (
            full.shared_route_artifact_sha256,
            full.route_call_response_sha256,
            full.shared_route_reserved_input_tokens,
            full.shared_route_reserved_output_tokens,
            full.shared_route_reserved_turns,
        )
        identity_matches = (
            all(value is not None for value in shared_identity[:4])
            and shared_identity == full_identity
        )
        status_matches = (
            s1s2.shared_route_status is not None
            and s1s2.shared_route_status == full.shared_route_status
        )
        route_trace_matches = s1s2.shared_route_status == "terminal_route_error" or (
            s1s2.route_trace_sha256 == s1s2.shared_route_artifact_sha256
            and full.route_trace_sha256 == full.shared_route_artifact_sha256
        )
        semantic_matches = s1s2.selected_capability == full.selected_capability
        terminal_fixed_zero = (
            status_matches
            and s1s2.shared_route_status == "terminal_route_error"
            and s1s2.assistant_error_code is not None
            and full.assistant_error_code is not None
            and s1s2.model_call_count == 0
            and full.model_call_count == 0
            and not s1s2.tool_trace
            and not full.tool_trace
            and s1s2.selected_capability is None
            and full.selected_capability is None
            and s1s2.route_trace_sha256 is None
            and full.route_trace_sha256 is None
            and s1s2.turn_count == s1s2.shared_route_reserved_turns
            and full.turn_count == full.shared_route_reserved_turns
        )
        routed = (
            status_matches
            and s1s2.shared_route_status == "selected"
            and route_trace_matches
            and semantic_matches
            and s1s2.selected_capability is not None
        )
        if identity_matches and terminal_fixed_zero:
            row_status = "matched_terminal_fixed_zero"
            matched_count += 1
            terminal_count += 1
        elif identity_matches and routed:
            row_status = "matched_selected_route"
            matched_count += 1
        else:
            row_status = "mismatched"
            mismatch_count += 1
            if not identity_matches:
                violations.append(
                    f"S1+S2/Full shared route identity differs: {query_id}"
                )
            if not status_matches:
                violations.append(f"S1+S2/Full shared route status differs: {query_id}")
            if not route_trace_matches:
                violations.append(
                    f"S1+S2/Full route trace differs from shared artifact: {query_id}"
                )
            if not semantic_matches:
                violations.append(
                    f"S1+S2/Full semantic route decision differs: {query_id}"
                )
            if (
                status_matches
                and s1s2.shared_route_status == "terminal_route_error"
                and not terminal_fixed_zero
            ):
                violations.append(
                    "terminal shared route did not produce paired fixed zeros: "
                    f"{query_id}"
                )
        details.append(
            {
                "query_id": query_id,
                "status": row_status,
                "shared_route_artifact_sha256": (s1s2.shared_route_artifact_sha256),
                "route_call_response_sha256": (s1s2.route_call_response_sha256),
                "reserved_usage": {
                    "input_tokens": s1s2.shared_route_reserved_input_tokens,
                    "output_tokens": s1s2.shared_route_reserved_output_tokens,
                },
                "reserved_turns": s1s2.shared_route_reserved_turns,
                "route_status": s1s2.shared_route_status,
                "identity_matches": identity_matches,
            }
        )

    return (
        {
            "required_policy": "shared-stage2-route-decision-per-query",
            "runner_policy_evidence": {
                "runner_file_sha256": runner_file_sha256,
                "policy": "one-persisted-route-artifact-referenced-by-both-configs",
                "first_model_call_interpretation": "action_call_with_reserved_route",
            },
            "shared_decision_identity_persisted": mismatch_count == 0,
            "query_count": len(details),
            "matched_query_count": matched_count,
            "terminal_shared_route_fixed_zero_count": terminal_count,
            "mismatch_count": mismatch_count,
            "rows": details,
        },
        violations,
    )


def _selected_batch_cost(
    summaries: Mapping[str, object],
    *,
    qwen_input_rate: float,
    qwen_output_rate: float,
    judge_input_rate: float,
    judge_output_rate: float,
    shared_route_input_tokens: int = 0,
    shared_route_output_tokens: int = 0,
    assistant_attempt_input_tokens: int = 0,
    assistant_attempt_output_tokens: int = 0,
    judge_attempt_input_tokens: int = 0,
    judge_attempt_output_tokens: int = 0,
) -> float:
    assistant_input = sum(
        int(summary["assistant_input_tokens"]) for summary in summaries.values()
    )
    assistant_output = sum(
        int(summary["assistant_output_tokens"]) for summary in summaries.values()
    )
    judge_input = sum(
        int(summary["judge_input_tokens"]) for summary in summaries.values()
    )
    judge_output = sum(
        int(summary["judge_output_tokens"]) for summary in summaries.values()
    )
    return (
        (assistant_input + shared_route_input_tokens + assistant_attempt_input_tokens)
        * qwen_input_rate
        + (
            assistant_output
            + shared_route_output_tokens
            + assistant_attempt_output_tokens
        )
        * qwen_output_rate
        + (judge_input + judge_attempt_input_tokens) * judge_input_rate
        + (judge_output + judge_attempt_output_tokens) * judge_output_rate
    ) / 1_000_000


def _selected_attempt_usage(
    execution_root: Path,
    shards: Sequence[object],
    *,
    sources_by_config: Mapping[str, _ShardArtifactSource],
) -> tuple[dict[str, int], list[dict[str, str]]]:
    totals = {
        "receipt_count": 0,
        "model_call_count": 0,
        "assistant_input_tokens": 0,
        "assistant_output_tokens": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
    }
    artifacts: list[dict[str, str]] = []
    for shard in shards:
        config = str(getattr(shard, "config"))
        source = sources_by_config.get(config)
        physical_root = execution_root if source is None else source.execution_root
        physical_shard = shard if source is None else source.shard
        shard_id = str(getattr(physical_shard, "shard_id"))
        physical_config = str(getattr(physical_shard, "config"))
        receipt_root = (
            physical_root
            / str(getattr(physical_shard, "output_relpath"))
            / "attempt-receipts"
        )
        for path in sorted(receipt_root.glob("*.json")):
            content = read_stable_regular_file(
                path,
                label=f"Portfolio attempt receipt {path.name}",
            )
            receipt = PortfolioAttemptReceipt.model_validate_json(
                content,
                strict=True,
            )
            if receipt.shard_id != shard_id or receipt.config != physical_config:
                raise ValueError("Portfolio attempt receipt differs from shard")
            prefix = "judge" if receipt.failure_stage == "final_judge" else "assistant"
            totals["receipt_count"] += 1
            totals["model_call_count"] += portfolio_attempt_provider_call_count(receipt)
            totals[f"{prefix}_input_tokens"] += receipt.captured_input_tokens
            totals[f"{prefix}_output_tokens"] += receipt.captured_output_tokens
            artifacts.append(
                {
                    "kind": "attempt_receipt",
                    "query_id": receipt.query_id,
                    "config": config,
                    "file_sha256": sha256_bytes(content),
                }
            )
    return totals, artifacts


def audit_portfolio_batch(
    execution_root: Path,
    *,
    batch_id: str | None = None,
    shard_ids: Sequence[str] | None = None,
    expected_noskill_audit_sha256: str | None = None,
    _profile_inputs: _LaunchProfileInputs | None = None,
) -> dict[str, object]:
    execution_root = execution_root.absolute()
    control = _load_control(execution_root)
    launch_root = _resolve_repository_path(str(control["launch_root"]))
    runtime_root = _resolve_repository_path(str(control["runtime_root"]))
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=str(control["launch_plan_file_sha256"]),
    )
    selected_batch, shards, launch_sequence = _select_batch_shards(
        launch.plan.shards,
        batch_id=batch_id,
        shard_ids=shard_ids,
    )
    dataset_profile = str(getattr(launch.plan, "dataset_profile", None) or "dev_mini")
    selected_query_count = int(getattr(shards[0], "query_count"))
    members_by_config = {
        str(getattr(shard, "config")): tuple(
            item
            for item in launch.instances
            if item.shard_id == getattr(shard, "shard_id")
        )
        for shard in shards
    }
    external_source, external_evidence, external_compatibility_violations = (
        _load_external_frozen_source(
            control=control,
            target_launch=launch,
            target_shards=shards,
            target_members=members_by_config["noskill"],
        )
    )
    internal_sources, internal_alias_evidence = _load_internal_artifact_alias_sources(
        control=control,
        launch=launch,
        execution_root=execution_root,
        runtime_root=runtime_root,
        shards=shards,
        members_by_config=members_by_config,
    )
    sources_by_config = dict(internal_sources)
    if external_source is not None:
        sources_by_config["noskill"] = external_source
    paired_violations = _paired_member_violations(members_by_config)
    paired_violations.extend(external_compatibility_violations)
    observed_source_file_sha256s = _observed_source_file_sha256s()
    completion, incomplete = _completion_snapshot(
        execution_root,
        shards,
        members_by_config,
        sources_by_config=sources_by_config,
    )
    if incomplete:
        return _content_address(
            {
                "schema_version": 1,
                "kind": "portfolio-batch-cross-config-audit",
                "track": "portfolio",
                "formal_eligible": False,
                "status": "incomplete_fail_closed",
                "batch_id": selected_batch,
                "query_count": selected_query_count,
                "row_count": sum(
                    min(
                        int(item["assistant_checkpoint_count"]),
                        int(item["final_checkpoint_count"]),
                    )
                    for item in completion.values()
                ),
                "config_order": list(MAIN_CONFIG_ORDER),
                **(
                    {
                        "dataset_profile": "core",
                        "selected_splits": list(launch.plan.selected_splits),
                        "batch_split": None,
                    }
                    if dataset_profile == "core"
                    else {}
                ),
                "launch_shard_sequence": [
                    {
                        "shard_id": str(getattr(shard, "shard_id")),
                        "config": str(getattr(shard, "config")),
                    }
                    for shard in launch_sequence
                ],
                "shard_ids": [str(getattr(shard, "shard_id")) for shard in shards],
                "completion": completion,
                "bindings": {
                    "runtime_lock_sha256": control.get("runtime_lock_sha256"),
                    "launch_plan_sha256": control.get("launch_plan_sha256"),
                    "observed_source_file_sha256s": (observed_source_file_sha256s),
                    "external_frozen_shard": external_evidence,
                    "internal_artifact_aliases": internal_alias_evidence,
                },
                "blockers": sorted(set(incomplete + paired_violations)),
                "review_flags": [],
                "model_calls_performed": 0,
            }
        )

    violations = list(paired_violations)
    artifact_hashes: list[dict[str, str]] = []
    artifact_hashes.extend(
        {
            "kind": "artifact_alias",
            "query_id": str(item["target_shard_id"]),
            "config": str(item["target_config"]),
            "file_sha256": str(item["alias_receipt_file_sha256"]),
        }
        for item in internal_alias_evidence
    )

    runtime_raw, runtime_content = _load_canonical_object(
        runtime_root / "runtime-lock.json",
        label="Portfolio runtime lock",
    )
    if sha256_bytes(runtime_content) != control["runtime_lock_file_sha256"]:
        violations.append("runtime-lock external file hash differs from control")
    unsigned_runtime = dict(runtime_raw)
    supplied_runtime_sha = unsigned_runtime.pop("runtime_lock_sha256", None)
    if (
        supplied_runtime_sha != _hash(unsigned_runtime)
        or supplied_runtime_sha != control["runtime_lock_sha256"]
    ):
        violations.append("runtime-lock content identity differs from control")
    if (
        runtime_raw.get("final_judge_parser_policy_version")
        != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
        or runtime_raw.get("final_judge_parser_policy_sha256")
        != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
    ):
        violations.append("runtime lock does not bind the v4 Judge parser")
    if any(
        runtime_raw.get(field) != expected
        for field, expected in _ACTIVE_FINAL_RESULT_LOCK.items()
    ):
        violations.append(
            "runtime lock does not bind the active final-Judge result, retry, "
            "and card-requirement-guard contracts"
        )
    if any(
        runtime_raw.get(field) != expected
        for field, expected in _ACTIVE_EVALUATOR_SOURCE_LOCK.items()
    ):
        violations.append(
            "runtime lock does not bind the active final-evaluation config, "
            "packet, and evaluator-isolation sources"
        )
    if any(
        runtime_raw.get(field) != expected or control.get(field) != expected
        for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
    ):
        violations.append("runtime/control hard-budget contract drifted")
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
        violations.append("launch hard-budget contract drifted")
    budget_ledger = load_portfolio_budget_ledger(
        _budget_ledger_root(execution_root, control)
    )
    if (
        budget_ledger.authority.matrix_run_id != launch.plan.matrix_run_id
        or budget_ledger.authority.phase_cap_cny
        != _control_cny(control, "phase_cumulative_cap_cny")
        or budget_ledger.authority.prior_observed_cost_cny
        != _control_cny(control, "prior_dashscope_observed_cost_cny")
        or budget_ledger.authority.policy_version != PORTFOLIO_BUDGET_POLICY_VERSION
    ):
        violations.append("create-only budget authority differs from control")
    router_runtime_contract = {
        "shared_stage2_route_policy_version": SHARED_STAGE2_ROUTE_POLICY_VERSION,
        "portfolio_router_contract_version": PORTFOLIO_ROUTER_CONTRACT_VERSION,
        "portfolio_router_contract_sha256": PORTFOLIO_ROUTER_CONTRACT_SHA256,
        "portfolio_router_request_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        ),
        "portfolio_router_pricing_reservation_max_output_tokens": (
            PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
        ),
        "portfolio_failure_policy_version": PORTFOLIO_FAILURE_POLICY_VERSION,
    }
    if any(
        runtime_raw.get(field) != expected
        for field, expected in router_runtime_contract.items()
    ):
        violations.append("runtime lock does not bind the active Router contract")
    if observed_source_file_sha256s[
        "src/skillchain/runners/assistant.py"
    ] != runtime_raw.get("runner_file_sha256"):
        violations.append("live Assistant runner differs from the runtime lock")
    if external_source is not None:
        expected_local_shard_ids = {
            str(getattr(item, "shard_id"))
            for item in shards
            if str(getattr(item, "config")) != "noskill"
        }
        authorized_shard_ids = set(control.get("authorized_shard_ids", ()))
        if (
            control.get("execution_scope") != "partial_shard_repair"
            or control.get("authorized_batch_id") != selected_batch
            or authorized_shard_ids != expected_local_shard_ids
            or control.get("local_authorized_shard_count") != 4
            or control.get("external_frozen_shard_count") != 1
            or control.get("comparison_shard_count") != 5
        ):
            violations.append(
                "external NoSkill execution scope differs from the selected 1x5"
            )
        if runtime_raw.get(
            "compatible_frozen_noskill_runtime_lock_sha256"
        ) != external_source.runtime_raw.get("runtime_lock_sha256"):
            violations.append(
                "target runtime does not bind the external frozen NoSkill runtime"
            )
        target_projection = _comparison_contract_payload(
            launch=launch,
            control=control,
            runtime_raw=runtime_raw,
            source_request_contract=(external_source.source_request_contract or {}),
            require_noskill_runtime_binding=True,
        )
        descriptor = control["external_frozen_shard"]
        if not isinstance(descriptor, dict):
            raise ValueError("external_frozen_shard must be one object")
        if target_projection != descriptor.get(
            "comparison_contract_payload"
        ) or sha256_bytes(canonical_json_bytes(target_projection)) != descriptor.get(
            "comparison_contract_sha256"
        ):
            violations.append(
                "external NoSkill comparison contract differs from target projection"
            )

    banks, bank_contexts, bank_audit = _load_banks(runtime_root, runtime_raw)
    del banks
    violations.extend(bank_audit["violations"])
    profile_inputs = (
        _load_launch_profile_inputs(launch)
        if _profile_inputs is None
        else _require_launch_profile_inputs(launch, _profile_inputs)
    )
    private_by_id = {item.query_id: item for item in profile_inputs.queries}
    assignments = _load_assignments(
        profile_inputs.capability_assignments_path,
        launch.plan.capability_assignments_sha256,
    )
    selected_query_ids = {
        str(getattr(member, "query_id")) for member in members_by_config["noskill"]
    }
    selected_batch_splits = {
        profile_inputs.split_by_query[query_id]
        for query_id in selected_query_ids
        if query_id in profile_inputs.split_by_query
    }
    if profile_inputs.profile == "core" and len(selected_batch_splits) != 1:
        violations.append("Core selected batch is not split-atomic")
    selected_batch_split = (
        next(iter(selected_batch_splits)) if len(selected_batch_splits) == 1 else None
    )
    for member in members_by_config["noskill"]:
        query_id = str(getattr(member, "query_id"))
        private = private_by_id.get(query_id)
        if private is None:
            violations.append(f"missing active query input: {query_id}")
            continue
        assignment = assignments.get(
            (
                str(private.asset_id),
                str(private.image_path),
                private.canonical_intent,
            )
        )
        if assignment is None:
            violations.append(f"missing capability assignment: {query_id}")
            continue
        for config in MAIN_CONFIG_ORDER[1:]:
            skill = bank_contexts[config]["by_capability"].get(
                assignment.canonical_capability
            )
            if skill is None or tuple(skill.operators) != tuple(
                assignment.allowed_tools
            ):
                violations.append(
                    f"{config} Bank operators differ from assignment for "
                    f"{assignment.canonical_capability}"
                )

    rubric_path = _resolve_repository_path(str(control["rubric_path"]))
    rubric_content = read_stable_regular_file(rubric_path, label="Portfolio rubric")
    if sha256_bytes(rubric_content) != control["rubric_file_sha256"]:
        violations.append("rubric file hash differs from execution control")
    rubric = RubricSnapshot.model_validate_json(rubric_content, strict=True)
    if rubric.content_sha256 != control["rubric_content_sha256"]:
        violations.append("rubric content hash differs from execution control")

    catalog = load_asset_catalog(
        profile_inputs.catalog_dir,
        profile_inputs.asset_root,
        verify_files=True,
    )
    isolation = make_active_portfolio_evaluator_isolation_lock()
    blinding_key = read_stable_regular_file(
        execution_root / "blinding-key.bin",
        label="Portfolio blinding key",
        max_bytes=32,
    )
    blinding_keys = {config: blinding_key for config in MAIN_CONFIG_ORDER}
    if external_source is not None:
        blinding_keys["noskill"] = read_stable_regular_file(
            external_source.execution_root / "blinding-key.bin",
            label="external frozen NoSkill blinding key",
            max_bytes=32,
        )

    observations: list[_RowObservation] = []
    backbone_hashes: set[str] = set()
    budget_hashes: set[str] = set()
    registry_lock_hashes: set[str] = set()
    summary_sha256s: dict[str, str] = {}
    shard_audit_sha256s: dict[str, str] = {}
    shared_route_artifacts: dict[str, SharedStage2RouteArtifact] = {}

    for shard in shards:
        config = str(getattr(shard, "config"))
        source = sources_by_config.get(config)
        effective_execution_root = (
            execution_root if source is None else source.execution_root
        )
        effective_launch = launch if source is None else source.launch
        effective_control = control if source is None else source.control
        effective_runtime_raw = runtime_raw if source is None else source.runtime_raw
        effective_shard = shard if source is None else source.shard
        effective_members = (
            members_by_config[config] if source is None else source.members
        )
        physical_config = (
            config if source is None else str(getattr(source.shard, "config"))
        )
        shard_id = str(getattr(effective_shard, "shard_id"))
        shard_root = effective_execution_root / str(
            getattr(effective_shard, "output_relpath")
        )
        summary_sha256s[config] = _validate_summary(
            shard_root / "shard-summary.json",
            shard_id=shard_id,
            config=physical_config,
        )
        shard_audit_sha256s[config] = _validate_shard_audit(
            shard_root / "shard-audit.json",
            shard_id=shard_id,
            config=physical_config,
            runtime_lock_sha256=str(effective_control["runtime_lock_sha256"]),
            launch_plan_sha256=str(effective_control["launch_plan_sha256"]),
            shard_summary_sha256=summary_sha256s[config],
            runtime_lock=effective_runtime_raw,
        )
        completion[config]["audit_sha256"] = shard_audit_sha256s[config]
        for member in effective_members:
            query_id = str(getattr(member, "query_id"))
            assistant_path = effective_execution_root / str(
                getattr(member, "assistant_output_relpath")
            )
            assistant_raw, assistant_content = _load_canonical_object(
                assistant_path,
                label=f"Assistant checkpoint {query_id}/{config}",
            )
            supplied_row_sha = assistant_raw.get("row_sha256")
            unsigned_row = dict(assistant_raw)
            unsigned_row.pop("row_sha256", None)
            if supplied_row_sha != _hash(unsigned_row):
                violations.append(f"Assistant row hash mismatch: {query_id}/{config}")
            response, receipt = _assistant_row(assistant_path)
            request = AssistantRequestSnapshot.model_validate_json(
                canonical_json_bytes(assistant_raw.get("request")),
                strict=True,
            )
            private = private_by_id.get(query_id)
            expected_asset_id = (
                None if private is None else str(getattr(private, "asset_id"))
            )
            asset_binding = request.query.asset_binding
            member_asset_token = getattr(member, "asset_token", None)
            if (
                (member_asset_token is not None and asset_binding is None)
                or private is not None
                and asset_binding is not None
                and (
                    asset_binding.asset_id != expected_asset_id
                    or asset_binding.image_path != str(getattr(private, "image_path"))
                    or asset_binding.leakage_group_id
                    != str(getattr(private, "leakage_group_id"))
                    or (
                        member_asset_token is not None
                        and (
                            asset_binding.asset_token != member_asset_token
                            or asset_binding.binding_sha256
                            != getattr(member, "asset_binding_sha256", None)
                        )
                    )
                )
            ):
                violations.append(
                    f"Assistant hidden asset binding mismatch: {query_id}/{config}"
                )
            try:
                _validate_backend_response(response, request)
            except (TypeError, ValueError) as error:
                violations.append(
                    f"Assistant response contract mismatch {query_id}/{config}: {error}"
                )
            expected_receipt_outcome = (
                "success"
                if response.error_code is None
                else (
                    "timeout" if response.error_code == "timeout" else "runtime_error"
                )
            )
            if (
                assistant_raw.get("instance_sha256")
                != getattr(member, "instance_sha256")
                or assistant_raw.get("query_ordinal")
                != getattr(member, "query_ordinal")
                or request.query.query_id != query_id
                or request.query.query_sha256 != getattr(member, "query_sha256")
                or request.query.public_input_sha256
                != getattr(member, "public_input_sha256")
                or request.config != physical_config
                or receipt.request_sha256 != request.request_sha256
                or receipt.response_sha256 != _hash(_model(response))
                or receipt.aggregate_usage != response.usage
                or receipt.tool_trace != response.tool_trace
                or receipt.outcome != expected_receipt_outcome
                or receipt.query_asset_id != expected_asset_id
                or receipt.query_asset_sha256 != getattr(member, "image_sha256")
                or receipt.asset_catalog_sha256
                != effective_launch.plan.runtime_catalog_sha256
            ):
                violations.append(
                    f"Assistant/launch/receipt binding mismatch: {query_id}/{config}"
                )

            backbone_hashes.add(request.backbone.identity_sha256)
            budget_hashes.add(request.budget.budget_sha256)
            registry_lock_hashes.add(request.registry.lock_sha256)
            if (
                request.backbone.provider != _EXPECTED_ASSISTANT_PROVIDER
                or request.backbone.model != _EXPECTED_ASSISTANT_MODEL
                or request.backbone.endpoint != _EXPECTED_ENDPOINT
                or request.backbone.temperature != 0.0
                or request.backbone.top_p != 1.0
                or request.backbone.seed is not None
                or request.backbone.system_prompt_sha256
                != effective_runtime_raw.get("system_prompt_sha256")
                or request.backbone.model != effective_launch.plan.assistant_model
                or request.backbone.provider != effective_launch.plan.assistant_provider
            ):
                violations.append(f"Assistant model lock drift: {query_id}/{config}")
            if {
                key: getattr(request.budget, key) for key in _EXPECTED_BUDGET
            } != _EXPECTED_BUDGET:
                violations.append(f"Assistant budget drift: {query_id}/{config}")
            if request.registry.registry_sha256 != effective_runtime_raw.get(
                "tool_registry_sha256"
            ) or request.registry.registry_runtime_sha256 != effective_runtime_raw.get(
                "tool_registry_runtime_sha256"
            ):
                violations.append(f"Assistant registry drift: {query_id}/{config}")
            for call in receipt.model_calls:
                if (
                    call.provider != _EXPECTED_ASSISTANT_PROVIDER
                    or call.endpoint != _EXPECTED_ENDPOINT
                    or call.requested_model != _EXPECTED_ASSISTANT_MODEL
                    or call.response_model != _EXPECTED_ASSISTANT_MODEL
                ):
                    violations.append(
                        f"Assistant call identity drift: {query_id}/{config}"
                    )
            reserved_turns = (
                0
                if receipt.shared_route_reference is None
                else receipt.shared_route_reference.reserved_turns
            )
            expected_turn_count = _expected_turn_count(receipt)
            if response.turn_count != expected_turn_count:
                violations.append(
                    f"Assistant turn/call count mismatch: {query_id}/{config}"
                )
            shared_route_status: str | None = None
            if receipt.shared_route_reference is not None:
                if config not in {"s1s2", "full"} or (
                    source is not None and source.artifact_alias is None
                ):
                    violations.append(
                        f"unexpected shared route reference: {query_id}/{config}"
                    )
                route_path = (
                    execution_root
                    / "shared-routes"
                    / str(getattr(shard, "accepted_batch_id"))
                    / f"{query_id}.json"
                )
                try:
                    route_artifact = shared_route_artifacts.get(
                        query_id
                    ) or SharedStage2RouteArtifact.model_validate_json(
                        read_stable_regular_file(
                            route_path,
                            label=f"shared Stage2 route {query_id}",
                        ),
                        strict=True,
                    )
                    shared_route_artifacts[query_id] = route_artifact
                    shared_route_status = route_artifact.status
                    route_reference = receipt.shared_route_reference
                    if (
                        route_artifact.policy_version
                        != SHARED_STAGE2_ROUTE_POLICY_VERSION
                        or route_reference.artifact_sha256
                        != route_artifact.artifact_sha256
                        or route_reference.route_call_response_sha256
                        != route_artifact.route_call.response_sha256
                        or route_reference.reserved_usage.input_tokens
                        != route_artifact.route_call.input_tokens
                        or route_reference.reserved_usage.output_tokens
                        != route_artifact.route_call.output_tokens
                    ):
                        violations.append(
                            f"shared route artifact/reference mismatch: "
                            f"{query_id}/{config}"
                        )
                except (OSError, TypeError, ValueError) as error:
                    violations.append(
                        f"shared route artifact is invalid: {query_id}/{config}: "
                        f"{error}"
                    )
            elif config in {"s1s2", "full"} and external_source is not None:
                violations.append(
                    f"shared route reference is absent: {query_id}/{config}"
                )

            private_requires_card = None if private is None else private.requires_card
            assignment = (
                None
                if private is None
                else assignments.get(
                    (
                        str(private.asset_id),
                        str(private.image_path),
                        private.canonical_intent,
                    )
                )
            )
            if assignment is None:
                violations.append(f"missing capability assignment: {query_id}/{config}")
                canonical_capability = "unknown"
            else:
                canonical_capability = assignment.canonical_capability
                if private is None or assignment.image_path != private.image_path:
                    violations.append(
                        f"capability assignment image differs: {query_id}/{config}"
                    )

            if config == "noskill":
                if request.treatment.bank_sha256 is not None:
                    violations.append(f"NoSkill retained Bank identity: {query_id}")
            else:
                locked_bank_sha = runtime_raw["bank_sha256s"].get(config)
                if request.treatment.bank_sha256 != locked_bank_sha:
                    violations.append(
                        f"treatment Bank differs from runtime: {query_id}/{config}"
                    )
            violations.extend(
                _tool_gate_violations(
                    query_id=query_id,
                    config=config,
                    response=response,
                    assignment=assignment,
                    bank_context=(
                        None if config == "noskill" else bank_contexts[config]
                    ),
                )
            )

            runtime_by_tool = {
                item.tool_name: item.runtime_binding_sha256
                for item in request.registry.tools
            }
            if any(
                runtime_by_tool.get(trace.tool_name) != trace.runtime_binding_sha256
                for trace in response.tool_trace
            ):
                violations.append(f"tool runtime binding mismatch: {query_id}/{config}")

            final_path = effective_execution_root / str(
                getattr(member, "final_output_relpath")
            )
            judge_status = "not_invoked_assistant_error"
            judge_error_code: str | None = None
            judge_shape: str | None = None
            judge_finish_reason: str | None = None
            judge_raw_response_bytes: int | None = None
            judge_tool_call_count: int | None = None
            judge_input_tokens = 0
            judge_output_tokens = 0
            judge_attempts = 0
            judge_initial_empty_response = False
            judge_terminal_response_captured = False
            judge_initial_reasoning_present: bool | None = None
            judge_initial_reasoning_tokens: int | None = None
            judge_initial_reasoning_bytes: int | None = None
            judge_initial_reasoning_sha256: str | None = None
            judge_terminal_reasoning_present: bool | None = None
            judge_terminal_reasoning_tokens: int | None = None
            judge_terminal_reasoning_bytes: int | None = None
            judge_terminal_reasoning_sha256: str | None = None
            j_project = 0.0
            assistant_error_code = response.error_code
            if response.error_code is not None:
                final_file_sha = _validate_fixed_zero(
                    final_path,
                    query_id=query_id,
                    error_code=response.error_code,
                )
            else:
                final_raw, final_content = _load_canonical_object(
                    final_path,
                    label=f"Final checkpoint {query_id}/{config}",
                )
                if final_raw.get("kind") == "portfolio-final-fixed-zero":
                    # A fail-closed packet sanitizer can produce a typed
                    # fixed-zero after the Assistant response itself was
                    # already provider-successful.  Validate that artifact
                    # directly and carry its error into the observation rather
                    # than trying to parse it as a visual Judge receipt.
                    assistant_error_code = str(final_raw.get("assistant_error_code"))
                    final_file_sha = _validate_fixed_zero(
                        final_path,
                        query_id=query_id,
                        error_code=assistant_error_code,
                    )
                else:
                    final = load_final_judge_evaluation_result(final_path)
                    final_file_sha = sha256_bytes(final_content)
                    result = _assistant_result(effective_launch, request, response)
                    packet = build_final_evaluation_packet(
                        private_by_id[query_id],
                        result,
                        asset_catalog=catalog,
                        rubric=rubric,
                        blinding_key=blinding_keys[config],
                    )
                    bound_prompt = build_bound_final_prompt(packet, isolation)
                    if (
                        final.schema_version
                        != effective_runtime_raw.get(
                            "final_judge_result_schema_version"
                        )
                        or final.cache_namespace
                        != effective_runtime_raw.get("final_judge_cache_namespace")
                        or final.parser_policy_version
                        != effective_runtime_raw.get(
                            "final_judge_parser_policy_version"
                        )
                        or final.parser_policy_sha256
                        != effective_runtime_raw.get("final_judge_parser_policy_sha256")
                        or final.card_requirement_guard_policy_version
                        != effective_runtime_raw.get(
                            "card_requirement_guard_policy_version"
                        )
                        or final.card_requirement_guard_policy_sha256
                        != effective_runtime_raw.get(
                            "card_requirement_guard_policy_sha256"
                        )
                        or final.visible_card_count != len(response.visible_cards)
                        or final.provider != _EXPECTED_JUDGE_PROVIDER
                        or final.model != _EXPECTED_JUDGE_MODEL
                        or final.endpoint != _EXPECTED_JUDGE_ENDPOINT
                        or final.max_tokens != _EXPECTED_JUDGE_MAX_TOKENS
                        or final.attempts not in {1, 2}
                        or final.max_attempts != FINAL_JUDGE_MAX_ATTEMPTS
                        or final.retry_policy_version
                        != FINAL_JUDGE_RETRY_POLICY_VERSION
                        or final.retry_policy_sha256 != FINAL_JUDGE_RETRY_POLICY_SHA256
                        or final.thinking_budget != FINAL_JUDGE_THINKING_BUDGET
                        or final.max_billable_input_tokens
                        != FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
                        or final.max_billable_output_tokens
                        != FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
                        or final.transport_policy_version
                        != FINAL_JUDGE_TRANSPORT_POLICY_VERSION
                        or final.transport_policy_sha256
                        != FINAL_JUDGE_TRANSPORT_POLICY_SHA256
                        or final.requested_response_format != "json_object"
                        or final.packet_sha256 != packet.packet_sha256
                        or final.evaluation_id != packet.evaluation_id
                        or final.image_sha256 != packet.image.sha256
                        or final.prompt_sha256 != bound_prompt.prompt_sha256
                        or final.asset_catalog_sha256
                        != effective_launch.plan.runtime_catalog_sha256
                        or final.outcome.prompt_sha256 != final.prompt_sha256
                        or final.outcome.judge_provider != _EXPECTED_JUDGE_PROVIDER
                        or final.outcome.judge_model != _EXPECTED_JUDGE_MODEL
                    ):
                        violations.append(
                            f"Assistant/Final/Judge binding mismatch: {query_id}/{config}"
                        )
                    judge_status = final.outcome.status
                    judge_error_code = final.outcome.error_code
                    judge_shape = final.raw_dimensions_shape
                    judge_finish_reason = final.finish_reason
                    judge_raw_response_bytes = final.raw_response_bytes
                    judge_tool_call_count = final.tool_call_count
                    aggregate_usage = final.aggregate_usage
                    judge_input_tokens = aggregate_usage.input_tokens
                    judge_output_tokens = aggregate_usage.output_tokens
                    judge_attempts = final.attempts
                    initial = final.initial_empty_response
                    judge_initial_retry_reason = (
                        None if initial is None else initial.retry_reason
                    )
                    judge_initial_empty_response = (
                        judge_initial_retry_reason == "empty_final_response"
                    )
                    judge_terminal_response_captured = final.request_id is not None
                    judge_initial_reasoning_present = (
                        None if initial is None else initial.reasoning_present
                    )
                    judge_initial_reasoning_tokens = (
                        None if initial is None else initial.reasoning_tokens
                    )
                    judge_initial_reasoning_bytes = (
                        None if initial is None else initial.reasoning_bytes
                    )
                    judge_initial_reasoning_sha256 = (
                        None if initial is None else initial.reasoning_sha256
                    )
                    judge_terminal_reasoning_present = final.reasoning_present
                    judge_terminal_reasoning_tokens = final.reasoning_tokens
                    judge_terminal_reasoning_bytes = final.reasoning_bytes
                    judge_terminal_reasoning_sha256 = final.reasoning_sha256
                    j_project = final.outcome.scores.j_project
                    if final.outcome.status == "scored" and (
                        final.finish_reason != "stop"
                        or final.tool_call_count != 0
                        or final.raw_dimensions_shape is None
                    ):
                        violations.append(
                            f"scored Judge result has invalid stop state: {query_id}/{config}"
                        )

            artifact_hashes.extend(
                (
                    {
                        "kind": "assistant",
                        "query_id": query_id,
                        "config": config,
                        "file_sha256": sha256_bytes(assistant_content),
                    },
                    {
                        "kind": "final",
                        "query_id": query_id,
                        "config": config,
                        "file_sha256": final_file_sha,
                    },
                )
            )
            observations.append(
                _RowObservation(
                    query_id=query_id,
                    config=config,
                    canonical_capability=canonical_capability,
                    card_requirement=(
                        "required"
                        if private_requires_card is True
                        else "forbidden"
                        if private_requires_card is False
                        else "unknown"
                    ),
                    visible_card_count=len(response.visible_cards),
                    card_policy_compliant=(
                        private_requires_card is True and bool(response.visible_cards)
                    )
                    or (private_requires_card is False and not response.visible_cards),
                    assistant_error_code=assistant_error_code,
                    receipt_outcome=receipt.outcome,
                    model_call_count=len(receipt.model_calls),
                    finish_reasons=tuple(
                        item.finish_reason for item in receipt.model_calls
                    ),
                    turn_count=response.turn_count,
                    tool_trace=tuple(
                        (item.tool_name, item.status, item.error_code)
                        for item in response.tool_trace
                    ),
                    selected_capability=response.selected_capability,
                    skill_slug=response.skill_slug,
                    route_trace_sha256=response.route_trace_sha256,
                    route_call_response_sha256=(
                        receipt.shared_route_reference.route_call_response_sha256
                        if receipt.shared_route_reference is not None
                        else receipt.model_calls[0].response_sha256
                        if config != "noskill" and receipt.model_calls
                        else None
                    ),
                    shared_route_artifact_sha256=(
                        None
                        if receipt.shared_route_reference is None
                        else receipt.shared_route_reference.artifact_sha256
                    ),
                    shared_route_reserved_input_tokens=(
                        None
                        if receipt.shared_route_reference is None
                        else receipt.shared_route_reference.reserved_usage.input_tokens
                    ),
                    shared_route_reserved_output_tokens=(
                        None
                        if receipt.shared_route_reference is None
                        else receipt.shared_route_reference.reserved_usage.output_tokens
                    ),
                    shared_route_reserved_turns=reserved_turns,
                    shared_route_status=shared_route_status,
                    route_acceptable=(
                        None
                        if (
                            config == "noskill"
                            or response.error_code is not None
                            or assignment is None
                        )
                        else response.selected_capability
                        in assignment.acceptable_capabilities
                    ),
                    assistant_input_tokens=response.usage.input_tokens,
                    assistant_output_tokens=response.usage.output_tokens,
                    judge_status=judge_status,
                    judge_error_code=judge_error_code,
                    judge_shape=judge_shape,
                    judge_finish_reason=judge_finish_reason,
                    judge_raw_response_bytes=judge_raw_response_bytes,
                    judge_tool_call_count=judge_tool_call_count,
                    judge_input_tokens=judge_input_tokens,
                    judge_output_tokens=judge_output_tokens,
                    judge_attempts=judge_attempts,
                    judge_initial_empty_response=(judge_initial_empty_response),
                    judge_terminal_response_captured=(judge_terminal_response_captured),
                    judge_initial_reasoning_present=(judge_initial_reasoning_present),
                    judge_initial_reasoning_tokens=(judge_initial_reasoning_tokens),
                    judge_initial_reasoning_bytes=(judge_initial_reasoning_bytes),
                    judge_initial_reasoning_sha256=(judge_initial_reasoning_sha256),
                    judge_terminal_reasoning_present=(judge_terminal_reasoning_present),
                    judge_terminal_reasoning_tokens=(judge_terminal_reasoning_tokens),
                    judge_terminal_reasoning_bytes=(judge_terminal_reasoning_bytes),
                    judge_terminal_reasoning_sha256=(judge_terminal_reasoning_sha256),
                    j_project=j_project,
                    judge_initial_retry_reason=judge_initial_retry_reason,
                )
            )

    for query_id, route_artifact in sorted(shared_route_artifacts.items()):
        route_path = (
            execution_root / "shared-routes" / selected_batch / f"{query_id}.json"
        )
        artifact_hashes.append(
            {
                "kind": "shared_route",
                "query_id": query_id,
                "config": "s1s2+full",
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        route_path,
                        label=f"shared Stage2 route artifact {query_id}",
                    )
                ),
            }
        )

    if len(backbone_hashes) != 1:
        violations.append("Assistant backbone identity differs across selected rows")
    if len(budget_hashes) != 1:
        violations.append("Assistant inference budget differs across selected rows")
    if len(registry_lock_hashes) != 1:
        violations.append("Assistant registry lock differs across selected rows")

    noskill_freeze_evidence, noskill_freeze_violations = _noskill_freeze_evidence(
        observed_sha256=shard_audit_sha256s["noskill"],
        expected_sha256=expected_noskill_audit_sha256,
    )
    violations.extend(noskill_freeze_violations)

    (
        summaries,
        skilled_errors,
        judge_anomalies,
        route_performance_observations,
        judge_response_receipts,
    ) = _summarize_observations(observations)
    route_reuse_check, route_reuse_violations = _s1s2_full_route_reuse_check(
        observations,
        runner_file_sha256=runtime_raw.get("runner_file_sha256"),
    )
    violations.extend(route_reuse_violations)
    attempt_usage, attempt_artifacts = _selected_attempt_usage(
        execution_root,
        shards,
        sources_by_config=sources_by_config,
    )
    artifact_hashes.extend(attempt_artifacts)
    review_flags = [
        f"skilled Assistant errors require classification: {len(skilled_errors)}"
        for _ in [0]
        if skilled_errors
    ]
    if judge_anomalies:
        review_flags.append(
            f"non-scored Judge rows require classification: {len(judge_anomalies)}"
        )
    all_assistant_errors = sum(
        sum(summary["assistant_error_counts"].values())
        for summary in summaries.values()
    )
    if all_assistant_errors:
        review_flags.append(
            f"Assistant errors remain fixed-zero rows: {all_assistant_errors}"
        )
    all_tool_errors = sum(
        int(summary["assistant_tool_error_count"]) for summary in summaries.values()
    )
    if all_tool_errors:
        review_flags.append(
            f"recovered Assistant tool errors require classification: {all_tool_errors}"
        )
    all_card_policy_violations = sum(
        int(summary["card_policy_violation_count"]) for summary in summaries.values()
    )
    if all_card_policy_violations:
        review_flags.append(
            "visible-card contract violations require classification: "
            f"{all_card_policy_violations}"
        )
    if attempt_usage["receipt_count"]:
        review_flags.append(
            "retryable attempt receipts require classification: "
            f"{attempt_usage['receipt_count']}"
        )

    local_selected_shard_ids = frozenset(
        str(getattr(shard, "shard_id"))
        for shard in shards
        if str(getattr(shard, "config")) not in sources_by_config
    )
    selected_cost_decimal = portfolio_budget_settled_cost_cny(
        budget_ledger,
        shard_ids=local_selected_shard_ids,
    )
    if external_source is not None:
        external_ledger = load_portfolio_budget_ledger(
            _budget_ledger_root(
                external_source.execution_root,
                external_source.control,
            )
        )
        selected_cost_decimal += portfolio_budget_settled_cost_cny(
            external_ledger,
            shard_ids=frozenset({str(getattr(external_source.shard, "shard_id"))}),
        )
    selected_cost = float(selected_cost_decimal)
    root_cost = float(portfolio_budget_settled_cost_cny(budget_ledger))
    prior_cost = float(budget_ledger.authority.prior_observed_cost_cny)
    cumulative_cost = prior_cost + root_cost
    accountable_cost = float(budget_ledger.accountable_cost_cny)
    approved_cost = float(control["approved_dashscope_budget_cny"])
    if cumulative_cost > approved_cost + 1e-9:
        violations.append("cumulative DashScope observed cost exceeds approval")
    if accountable_cost > approved_cost + 1e-9:
        violations.append("cumulative DashScope accountable cost exceeds approval")

    status = (
        "failed"
        if violations
        else ("review_required" if review_flags else "passed_clean")
    )
    payload = {
        "schema_version": 1,
        "kind": "portfolio-batch-cross-config-audit",
        "track": "portfolio",
        "formal_eligible": False,
        "status": status,
        "batch_id": selected_batch,
        "query_count": selected_query_count,
        "row_count": len(observations),
        "config_order": list(MAIN_CONFIG_ORDER),
        **(
            {
                "dataset_profile": "core",
                "selected_splits": list(profile_inputs.selected_splits),
                "batch_split": selected_batch_split,
            }
            if profile_inputs.profile == "core"
            else {}
        ),
        "launch_shard_sequence": [
            {
                "shard_id": str(getattr(shard, "shard_id")),
                "config": str(getattr(shard, "config")),
            }
            for shard in launch_sequence
        ],
        "shard_ids": [str(getattr(shard, "shard_id")) for shard in shards],
        "completion": completion,
        "bindings": {
            "matrix_run_id": launch.plan.matrix_run_id,
            "runtime_lock_sha256": control["runtime_lock_sha256"],
            "launch_plan_sha256": control["launch_plan_sha256"],
            "rubric_file_sha256": control["rubric_file_sha256"],
            "rubric_content_sha256": control["rubric_content_sha256"],
            "assistant_backbone_identity_sha256": next(iter(backbone_hashes), None),
            "assistant_budget_sha256": next(iter(budget_hashes), None),
            "assistant_registry_lock_sha256": next(iter(registry_lock_hashes), None),
            "judge_parser_policy_version": (FINAL_JUDGE_PARSER_POLICY_VERSION_V4),
            "judge_parser_policy_sha256": (FINAL_JUDGE_PARSER_POLICY_SHA256_V4),
            "observed_source_file_sha256s": observed_source_file_sha256s,
            "summary_sha256s": summary_sha256s,
            "shard_audit_sha256s": shard_audit_sha256s,
            "noskill_freeze_evidence": noskill_freeze_evidence,
            "external_frozen_shard": external_evidence,
            "internal_artifact_aliases": internal_alias_evidence,
            **_ACTIVE_BUDGET_CONTRACT,
            "budget_authority_sha256": (budget_ledger.authority.authority_sha256),
            "budget_ledger_last_event_index": budget_ledger.last_event_index,
            "budget_ledger_last_event_sha256": budget_ledger.last_event_sha256,
        },
        "paired_input_check": {
            "paired_query_count": selected_query_count,
            "violation_count": len(paired_violations),
        },
        "bank_operator_check": {
            "capabilities": list(_EXPECTED_CAPABILITIES),
            "operator_matrix": bank_audit["operator_matrix"],
        },
        "bank_stage_boundary_check": bank_audit["stage_boundary_check"],
        "config_summary": summaries,
        "route_performance_observations": route_performance_observations,
        "s1s2_full_route_reuse_check": route_reuse_check,
        "skilled_runtime_errors": skilled_errors,
        "judge_anomalies": judge_anomalies,
        "judge_response_receipts": judge_response_receipts,
        "cost": {
            "currency": "CNY",
            "selected_batch_observed_cost_cny": selected_cost,
            "selected_batch_attempt_receipt_count": attempt_usage["receipt_count"],
            "selected_batch_attempt_model_call_count": attempt_usage[
                "model_call_count"
            ],
            "selected_batch_assistant_attempt_input_tokens": attempt_usage[
                "assistant_input_tokens"
            ],
            "selected_batch_assistant_attempt_output_tokens": attempt_usage[
                "assistant_output_tokens"
            ],
            "selected_batch_judge_attempt_input_tokens": attempt_usage[
                "judge_input_tokens"
            ],
            "selected_batch_judge_attempt_output_tokens": attempt_usage[
                "judge_output_tokens"
            ],
            "execution_root_observed_cost_cny": root_cost,
            "prior_dashscope_observed_cost_cny": prior_cost,
            "cumulative_dashscope_observed_cost_cny": cumulative_cost,
            "cumulative_dashscope_accountable_cost_cny": accountable_cost,
            "approved_dashscope_budget_cny": approved_cost,
            "remaining_approved_budget_cny": approved_cost - accountable_cost,
            "budget_ledger_settled_actual_cost_cny": format(
                budget_ledger.settled_actual_cost_cny, ".12f"
            ),
            "budget_ledger_forfeit_count": len(budget_ledger.forfeits),
            "budget_ledger_forfeited_reserved_cost_cny": format(
                budget_ledger.forfeited_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_unresolved_reserved_cost_cny": format(
                budget_ledger.unresolved_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_accountable_cost_cny": format(
                budget_ledger.accountable_cost_cny, ".12f"
            ),
            "budget_ledger_remaining_phase_cap_cny": format(
                budget_ledger.remaining_cost_cny, ".12f"
            ),
        },
        "artifact_set_sha256": sha256_bytes(canonical_json_bytes(artifact_hashes)),
        "blockers": sorted(set(violations)),
        "review_flags": sorted(set(review_flags)),
        "model_calls_performed": 0,
    }
    return _content_address(payload)


def _exit_code(audit: Mapping[str, object]) -> int:
    status = audit.get("status")
    if status == "passed_clean":
        return 0
    if status == "review_required":
        return 3
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_portfolio_batch(
            args.execution_root,
            batch_id=args.batch_id,
            shard_ids=args.shard_ids,
            expected_noskill_audit_sha256=(args.expected_noskill_audit_sha256),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        audit = _failure_audit(
            status="failed",
            requested_batch_id=args.batch_id,
            blockers=[str(error)],
        )
    audit_bytes = canonical_json_bytes(audit)
    if args.output is not None:
        try:
            atomic_create_file(args.output, audit_bytes)
        except OSError as error:
            print(f"portfolio-batch-audit: {error}", file=sys.stderr)
            return 2
    sys.stdout.buffer.write(audit_bytes)
    return _exit_code(audit)


if __name__ == "__main__":
    raise SystemExit(main())
