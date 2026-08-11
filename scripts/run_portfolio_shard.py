"""Dry-run or execute one resumable 25-query Portfolio Assistant+Judge shard."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
import platform
from pathlib import Path
import sys
import time
from collections.abc import Callable
from typing import Literal, Mapping

# The OpenAI SDK computes a client platform header lazily.  On this Windows
# host ``platform.platform()`` can enter a slow/unresponsive WMI query for the
# CPU caption, which makes every parallel worker vulnerable to a startup
# timeout before the provider request is even issued.  The SDK only needs the
# OS family for this header; keep the deterministic Windows identity while
# avoiding the WMI probe in each worker process.
platform.platform = lambda *args, **kwargs: "Windows"  # type: ignore[assignment]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_launch import _load_active_inputs  # noqa: E402
from scripts.run_portfolio_assistant_smoke import (  # noqa: E402
    _hash,
    _model,
    _self_model,
    _treatment,
)
from skillchain import config  # noqa: E402
from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantRequestSnapshot,
    BackboneLock,
    InferenceBudget,
    _registry_lock,
    build_legacy_assistant_query_input,
)
from skillchain.evaluation.evaluator_isolation import (  # noqa: E402
    EvaluatorIsolationError,
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
    FinalJudgeBudgetContext,
    FinalJudgeEvaluationResult,
    load_final_judge_evaluation_result,
    run_visual_final_judge,
    write_final_judge_evaluation_result,
)
from skillchain.evaluation.packets import (  # noqa: E402
    AssistantResult,
    FinalEvaluationPacket,
    HiddenEvaluationIdentityError,
    RubricSnapshot,
    build_final_evaluation_packet,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_core_runtime_sources import (  # noqa: E402
    load_verified_portfolio_core_runtime_sources,
)
from skillchain.evaluation.portfolio_core_inputs import (  # noqa: E402
    VerifiedPortfolioCoreInputs,
)
from skillchain.evaluation.portfolio_parallel import (  # noqa: E402
    PortfolioAggregateProviderGate,
    load_parallel_profile,
    validate_parallel_profile_sources,
)
from skillchain.evaluation.portfolio_gcs import (  # noqa: E402
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_gcs_evidence import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION,
    PublicScorerEvidenceV2,
    expected_multi_items_from_call_v2,
    make_public_scorer_evidence_v2,
    require_public_scorer_evidence_v2,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD,
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_FAILURE_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE,
    PortfolioAttemptReceipt,
    PortfolioAttemptLimitError,
    PortfolioBudgetDuplicateCallError,
    PortfolioBudgetExceededError,
    PortfolioBudgetOrphanedCallError,
    ProviderPreResponseCircuitBreaker,
    ValidatedRouteIdentity,
    create_retryable_attempt_receipt,
    forfeit_portfolio_provider_call,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
    load_query_attempt_receipts,
    open_portfolio_budget_ledger_session,
    portfolio_budget_settled_cost_cny,
    require_retryable_attempt_available,
)
from skillchain.runners.assistant import (  # noqa: E402
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    AssistantProviderPreResponseError,
    PortfolioAssistantBudgetContext,
    PortfolioAssistantRunner,
    PortfolioStaticOptAssistantRunner,
    SharedStage2RouteArtifact,
    require_portfolio_assistant_runner,
    require_portfolio_static_opt_assistant_runner,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.task_spec import load_mvp_task_specification_v1  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)


_ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES = frozenset(
    {"invalid_json", "non_object", "unexpected_keys", "schema_invalid"}
)
_ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES = frozenset(
    {"invalid_route_json", "length", "response_empty_text"}
)
_SHARED_ROUTE_OWNER_CONFIG = "s1s2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform real Assistant and final-Judge calls; default is zero-call dry-run.",
    )
    parser.add_argument(
        "--parallel-profile",
        type=Path,
        help="Optional immutable aggregate provider-concurrency profile.",
    )
    parser.add_argument(
        "--legacy-launch-compat",
        action="store_true",
        help="Rebuild the legacy public query projection bound by an older launch.",
    )
    return parser


@contextmanager
def _provider_slot(
    gate: PortfolioAggregateProviderGate | None,
    stage: Literal["assistant", "final_judge"],
    *,
    label: str,
):
    if gate is None:
        yield
        return
    with gate.acquire(stage, label=label):
        yield


def _load_control(root: Path) -> dict:
    data = json.loads((root / "execution-control.json").read_bytes())
    supplied = data.get("control_sha256")
    unsigned = dict(data)
    unsigned.pop("control_sha256", None)
    if supplied != _hash(unsigned):
        raise ValueError("execution control self hash mismatch")
    key = (root / "blinding-key.bin").read_bytes()
    if len(key) != 32 or sha256_bytes(key) != data.get("blinding_key_sha256"):
        raise ValueError("execution blinding key mismatch")
    return data


_ACTIVE_FINAL_RESULT_CONTRACT = {
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
_ACTIVE_EVALUATOR_SOURCE_CONTRACT = {
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
_ACTIVE_GCS_SOURCE_CONTRACT = {
    "assistant_checkpoint_schema_version": 2,
    "gcs_policy_version": GCS_V2_POLICY_VERSION,
    "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
    "gcs_scorer_evidence_policy_version": (GCS_SCORER_EVIDENCE_V2_POLICY_VERSION),
    "gcs_scorer_evidence_schema_version": (GCS_SCORER_EVIDENCE_V2_SCHEMA_VERSION),
    "portfolio_gcs_file_sha256": sha256_bytes(
        (SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_gcs.py").read_bytes()
    ),
    "portfolio_gcs_evidence_file_sha256": sha256_bytes(
        (
            SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_gcs_evidence.py"
        ).read_bytes()
    ),
    "task_spec_file_sha256": sha256_bytes(
        (
            REPOSITORY_ROOT / "specs" / "task_specs" / "ecommerce-task-spec-v1.json"
        ).read_bytes()
    ),
    "task_spec_sha256": load_mvp_task_specification_v1().task_spec_sha256,
}
_ACTIVE_BUDGET_CONTRACT = {
    "portfolio_budget_policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
    "portfolio_budget_policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
    "provider_pricing_contract_version": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION),
    "provider_pricing_contract_sha256": (PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256),
}


def _require_active_final_result_runtime(lock: Mapping[str, object]) -> None:
    if any(
        lock.get(field) != expected
        for field, expected in _ACTIVE_FINAL_RESULT_CONTRACT.items()
    ):
        raise ValueError(
            "runtime lock does not bind the active final-Judge result and "
            "card-requirement guard contract"
        )
    if any(
        lock.get(field) != expected
        for field, expected in _ACTIVE_EVALUATOR_SOURCE_CONTRACT.items()
    ):
        raise ValueError(
            "runtime lock does not bind the active final-evaluation config, "
            "packet, and evaluator-isolation sources"
        )
    if any(
        lock.get(field) != expected
        for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
    ):
        raise ValueError("runtime lock does not bind the active hard-budget contract")


def _require_active_gcs_v2_runtime(lock: Mapping[str, object]) -> None:
    if any(
        lock.get(field) != expected
        for field, expected in _ACTIVE_GCS_SOURCE_CONTRACT.items()
    ):
        raise ValueError("runtime lock does not bind the active GCS v2 contract")


def _require_final_checkpoint_contract(
    result: FinalJudgeEvaluationResult,
    *,
    runtime_lock: Mapping[str, object],
    visible_card_count: int,
) -> None:
    _require_active_final_result_runtime(runtime_lock)
    if (
        result.schema_version != runtime_lock["final_judge_result_schema_version"]
        or result.cache_namespace != runtime_lock["final_judge_cache_namespace"]
        or result.card_requirement_guard_policy_version
        != runtime_lock["card_requirement_guard_policy_version"]
        or result.card_requirement_guard_policy_sha256
        != runtime_lock["card_requirement_guard_policy_sha256"]
        or result.max_attempts != runtime_lock["final_judge_max_attempts"]
        or result.retry_policy_version
        != runtime_lock["final_judge_retry_policy_version"]
        or result.retry_policy_sha256 != runtime_lock["final_judge_retry_policy_sha256"]
        or result.thinking_budget != runtime_lock["final_judge_thinking_budget"]
        or result.max_billable_input_tokens
        != runtime_lock["final_judge_max_billable_input_tokens"]
        or result.max_billable_output_tokens
        != runtime_lock["final_judge_max_billable_output_tokens"]
        or result.transport_policy_version
        != runtime_lock["final_judge_transport_policy_version"]
        or result.transport_policy_sha256
        != runtime_lock["final_judge_transport_policy_sha256"]
        or result.requested_response_format
        != runtime_lock["final_judge_requested_response_format"]
        or result.visible_card_count != visible_card_count
    ):
        raise ValueError(
            "final checkpoint uses a different result or card-requirement "
            "guard contract"
        )


def _inputs_for_launch(launch_plan):
    if getattr(launch_plan, "kind", None) in {
        "portfolio-core-split-x5-launch-plan",
        "portfolio-core-static-opt-800x1-launch-plan",
    }:
        return reconstruct_verified_portfolio_core_inputs(launch_plan)
    return _load_active_inputs()


def _build_runner(
    runtime_root: Path,
    runtime_lock_file_sha256: str,
    *,
    inputs,
    static_opt: bool = False,
    qwen_call_start_waiter: Callable[[str], float] | None = None,
):
    if static_opt:
        static_runtime = load_verified_portfolio_static_opt_runtime(
            runtime_root,
            expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
        )
        lock = dict(static_runtime.runtime_lock)
        banks = {"llm_static": static_runtime.bank}
        treatment_runtime = None
    else:
        treatment_runtime = load_verified_portfolio_treatment_runtime(
            runtime_root,
            expected_runtime_lock_file_sha256=runtime_lock_file_sha256,
        )
        lock = dict(treatment_runtime.runtime_lock)
        banks = dict(treatment_runtime.chain.output_banks)
    core = type(inputs) is VerifiedPortfolioCoreInputs
    if core:
        source_dir = lock.get("core_runtime_sources_dir")
        source_receipt_sha256 = lock.get("core_runtime_sources_receipt_file_sha256")
        if source_dir != "core-runtime-sources" or not isinstance(
            source_receipt_sha256, str
        ):
            raise ValueError(
                "Core runtime lock lacks its materialized tool-source receipt"
            )
        verified_sources = load_verified_portfolio_core_runtime_sources(
            inputs,
            output_dir=runtime_root / source_dir,
            expected_receipt_file_sha256=source_receipt_sha256,
        )
        sources = verified_sources.sources
        catalog_root = inputs.files.runtime_catalog_dir
        if catalog_root is None:
            raise ValueError("Core execution requires its permission overlay catalog")
        catalog_asset_root = inputs.files.asset_root
    else:
        clean = REPOSITORY_ROOT / "data" / "clean"
        sources = PortfolioRuntimeSources(
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
            recipe_evidence=runtime_root / "recipe-evidence.jsonl",
        )
        catalog_root = clean / "portfolio-mini-asset-catalog-v3"
        catalog_asset_root = clean
    runtime = build_portfolio_tool_runtime(sources)
    if static_opt:
        if any(
            lock.get(field) != expected
            for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
        ):
            raise ValueError(
                "Static opt runtime does not bind the active hard-budget contract"
            )
        _require_active_gcs_v2_runtime(lock)
    else:
        evaluator_root = SOURCE_ROOT / "skillchain" / "evaluation"
        if (
            lock.get("evaluator_outputs_file_sha256")
            != sha256_bytes((evaluator_root / "evaluator_outputs.py").read_bytes())
            or lock.get("final_runtime_file_sha256")
            != sha256_bytes((evaluator_root / "final_runtime.py").read_bytes())
            or lock.get("final_judge_parser_policy_version")
            != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
            or lock.get("final_judge_parser_policy_sha256")
            != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
        ):
            raise ValueError("runtime lock differs from the active final-Judge parser")
        _require_active_final_result_runtime(lock)
    if lock.get("schema_version") == 2:
        execution_policy_checks = {
            "runner_file_sha256": sha256_bytes(
                (SOURCE_ROOT / "skillchain" / "runners" / "assistant.py").read_bytes()
            ),
            "shard_runner_file_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "portfolio_execution_file_sha256": sha256_bytes(
                (
                    SOURCE_ROOT / "skillchain" / "evaluation" / "portfolio_execution.py"
                ).read_bytes()
            ),
            "shared_stage2_route_policy_version": (SHARED_STAGE2_ROUTE_POLICY_VERSION),
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
            "portfolio_circuit_breaker_threshold": (
                PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
            ),
            "portfolio_max_retryable_attempts_per_query": (
                PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
            ),
            "noskill_execution_policy_version": (NOSKILL_EXECUTION_POLICY_VERSION),
            "noskill_execution_contract_sha256": (NOSKILL_EXECUTION_CONTRACT_SHA256),
        }
        if any(
            lock.get(field) != expected
            for field, expected in execution_policy_checks.items()
        ):
            raise ValueError(
                "runtime lock differs from the active shared-route or "
                "failure-containment policy"
            )
    catalog = (
        inputs.runtime_asset_catalog()
        if core
        else load_asset_catalog(
            catalog_root,
            catalog_asset_root,
            verify_files=True,
        )
    )
    if static_opt:
        runner = require_portfolio_static_opt_assistant_runner(
            PortfolioStaticOptAssistantRunner(
                registry=runtime.registry,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT,
                bank=banks["llm_static"],
                asset_catalog=catalog,
                runtime_lock=lock,
                runtime_lock_file_sha256=runtime_lock_file_sha256,
                qwen_call_start_waiter=qwen_call_start_waiter,
            )
        )
    else:
        runner = require_portfolio_assistant_runner(
            PortfolioAssistantRunner(
                registry=runtime.registry,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT,
                banks=banks,
                asset_catalog=catalog,
                runtime_lock=lock,
                runtime_lock_file_sha256=runtime_lock_file_sha256,
                qwen_call_start_waiter=qwen_call_start_waiter,
            )
        )
    return runtime, banks, catalog, runner


def _request(
    *,
    launch,
    shard,
    member,
    query,
    treatment,
    backbone,
    budget,
    registry_lock,
) -> AssistantRequestSnapshot:
    payload = {
        "schema_version": 1,
        "matrix_run_id": launch.plan.matrix_run_id,
        "config": shard.config,
        "query_ordinal": member.query_ordinal,
        "query": query,
        "treatment": treatment,
        "backbone": backbone,
        "budget": budget,
        "registry": registry_lock,
    }
    hash_payload = {
        key: _model(value) if hasattr(value, "model_dump") else value
        for key, value in payload.items()
    }
    return AssistantRequestSnapshot.model_validate(
        {**payload, "request_sha256": _hash(hash_payload)},
        strict=True,
    )


def _assistant_row(
    path: Path,
    *,
    require_schema_v2: bool = False,
    include_scorer_evidence: bool = False,
) -> (
    tuple[AssistantBackendResponse, AssistantExecutionReceipt]
    | tuple[
        AssistantBackendResponse,
        AssistantExecutionReceipt,
        PublicScorerEvidenceV2 | None,
    ]
):
    data = json.loads(path.read_bytes())
    supplied = data.get("row_sha256")
    unsigned = dict(data)
    unsigned.pop("row_sha256", None)
    if supplied != _hash(unsigned):
        raise ValueError(f"Assistant checkpoint hash mismatch: {path}")
    schema_version = data.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"Assistant checkpoint schema is unsupported: {path}")
    if require_schema_v2 and schema_version != 2:
        raise ValueError(
            f"GCS v2 execution rejects legacy Assistant checkpoint: {path}"
        )
    sidecar = data.get("public_scorer_evidence")
    if schema_version == 2 and sidecar is None:
        raise ValueError(
            f"schema-v2 Assistant checkpoint lacks scorer evidence: {path}"
        )
    if schema_version == 1 and sidecar is not None:
        raise ValueError(
            f"legacy Assistant checkpoint carries v2 scorer evidence: {path}"
        )
    response = AssistantBackendResponse.model_validate_json(
        canonical_json_bytes(data["response"]),
        strict=True,
    )
    receipt = AssistantExecutionReceipt.model_validate_json(
        canonical_json_bytes(data["receipt"]),
        strict=True,
    )
    if receipt.response_sha256 != _hash(_model(response)):
        raise ValueError(f"Assistant checkpoint receipt/response mismatch: {path}")
    evidence = (
        require_public_scorer_evidence_v2(sidecar) if sidecar is not None else None
    )
    if include_scorer_evidence:
        return response, receipt, evidence
    return response, receipt


def _requires_gcs_v2_checkpoint(control: Mapping[str, object]) -> bool:
    required = control.get("execution_scope") == "static_opt_rollout"
    declarations = (
        control.get("assistant_checkpoint_schema_version"),
        control.get("gcs_policy_version"),
        control.get("gcs_scorer_evidence_policy_version"),
    )
    if required and declarations != (
        2,
        GCS_V2_POLICY_VERSION,
        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    ):
        raise ValueError(
            "static opt rollout lacks the exact GCS v2 checkpoint contract"
        )
    if not required and any(item is not None for item in declarations):
        if declarations != (
            2,
            GCS_V2_POLICY_VERSION,
            GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        ):
            raise ValueError("execution control carries a partial GCS v2 contract")
        required = True
    return required


def _verify_scorer_sidecar_binding(
    evidence: PublicScorerEvidenceV2,
    *,
    launch,
    member,
    request: AssistantRequestSnapshot,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
) -> None:
    successful_trace = tuple(
        (item.call_index, item.tool_name, item.arguments_sha256, item.result_sha256)
        for item in result.tool_trace
        if item.status == "success"
    )
    scorer_trace = tuple(
        (item.call_index, item.tool_name, item.arguments_sha256, item.result_sha256)
        for item in evidence.calls
    )
    expected = (
        evidence.matrix_run_id == launch.plan.matrix_run_id,
        evidence.instance_id == member.instance_sha256,
        evidence.request_sha256 == request.request_sha256,
        evidence.query_id == member.query_id,
        evidence.config == member.config,
        evidence.query_artifact_sha256 == launch.plan.query_artifact_sha256,
        evidence.assistant_result_sha256 == _hash(_model(result)),
        evidence.assistant_receipt_sha256 == receipt.receipt_sha256,
        scorer_trace == successful_trace,
    )
    if not all(expected):
        raise ValueError("GCS v2 scorer sidecar differs from its frozen execution")


def _assistant_result(launch, request, response) -> AssistantResult:
    routed = response.selected_capability is not None
    # Runner-private pre-response diagnostics retain their exact stage in the
    # schema-v2 checkpoint and attempt receipt.  ``AssistantResult`` exposes
    # only the frozen public error vocabulary used by GCS and packet builders.
    # Map without mutating the runner-owned response bound by its receipt.
    public_error_code = (
        "runtime_error"
        if response.error_code is not None
        and response.error_code.startswith("provider_pre_response_")
        else response.error_code
    )
    return AssistantResult(
        run_id=launch.plan.matrix_run_id,
        query_id=request.query.query_id,
        config=request.config,
        response_text=response.response_text,
        visible_cards=response.visible_cards,
        visible_tool_evidence=response.visible_tool_evidence,
        tool_trace=response.tool_trace,
        selected_capability=response.selected_capability,
        skill_slug=response.skill_slug,
        bank_sha256=request.treatment.bank_sha256 if routed else None,
        route_trace_sha256=response.route_trace_sha256,
        query_artifact_sha256=launch.plan.query_artifact_sha256,
        split_manifest_sha256=launch.plan.portfolio_plan_sha256,
        registry_sha256=response.registry_sha256,
        registry_runtime_sha256=response.registry_runtime_sha256,
        backbone_provider=response.backbone_provider,
        backbone_model=response.backbone_model,
        backbone_request_id=response.backbone_request_id,
        usage=response.usage,
        latency_ms=response.latency_ms,
        error_code=public_error_code,
    )


def _build_final_packet_fail_closed(
    query: Query,
    result: AssistantResult,
    *,
    asset_catalog: AssetCatalog,
    rubric: RubricSnapshot,
    blinding_key: bytes,
) -> tuple[FinalEvaluationPacket | None, str | None]:
    """Map a public identity leak to a terminal zero without calling the Judge."""

    try:
        packet = build_final_evaluation_packet(
            query,
            result,
            asset_catalog=asset_catalog,
            rubric=rubric,
            blinding_key=blinding_key,
        )
    except HiddenEvaluationIdentityError:
        return None, "hidden_evaluation_identity"
    except EvaluatorIsolationError:
        # Product evidence is model-visible and may legitimately contain words
        # such as ``FULL TREATMENT``.  If the isolation sanitizer rejects that
        # evidence, preserve a typed terminal zero rather than aborting the
        # shard without an Assistant error checkpoint.
        return None, "final_packet_isolation_error"
    return packet, None


def _final_fixed_zero(query_id: str, assistant_error_code: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "portfolio-final-fixed-zero",
        "query_id": query_id,
        "assistant_error_code": assistant_error_code,
        "failure_class": "terminal_task_failure",
        "retryable": False,
        "score_disposition": "fixed_zero",
        "j_project": 0.0,
    }


def _existing_fixed_zero_error(path: Path, *, query_id: str) -> str | None:
    data = json.loads(path.read_bytes())
    if data.get("kind") != "portfolio-final-fixed-zero":
        return None
    assistant_error_code = data.get("assistant_error_code")
    if not isinstance(assistant_error_code, str):
        raise ValueError(f"fixed-zero result lacks an error code: {query_id}")
    expected = _final_fixed_zero(query_id, assistant_error_code)
    expected_with_hash = {**expected, "result_sha256": _hash(expected)}
    if data != expected_with_hash:
        raise ValueError(f"fixed-zero result is not canonical and valid: {query_id}")
    return assistant_error_code


def _observed_cost(root: Path) -> float:
    ledger_root = root / "budget-ledger"
    if (ledger_root / "budget-authority.json").is_file():
        return float(
            portfolio_budget_settled_cost_cny(load_portfolio_budget_ledger(ledger_root))
        )

    # Compatibility only for historical execution roots that predate the
    # create-only hard-budget ledger.
    total = 0.0
    for path in root.glob("shared-routes/*/*.json"):
        data = json.loads(path.read_bytes())
        usage = data.get("route_call", {})
        total += (
            int(usage.get("input_tokens", 0)) * 0.15
            + int(usage.get("output_tokens", 0)) * 1.5
        ) / 1_000_000
    for path in root.glob("shards/*/assistant/*.json"):
        response = json.loads(path.read_bytes()).get("response", {})
        usage = response.get("usage", {})
        total += (
            int(usage.get("input_tokens", 0)) * 0.15
            + int(usage.get("output_tokens", 0)) * 1.5
        ) / 1_000_000
    for path in root.glob("shards/*/final/*.json"):
        data = json.loads(path.read_bytes())
        if data.get("kind") == "portfolio-final-fixed-zero":
            continue
        result = load_final_judge_evaluation_result(path)
        usage = result.aggregate_usage
        total += (usage.input_tokens * 6.5 + usage.output_tokens * 27.0) / 1_000_000
    for path in root.glob("shards/*/attempt-receipts/*.json"):
        receipt = PortfolioAttemptReceipt.model_validate_json(
            path.read_bytes(),
            strict=True,
        )
        input_rate, output_rate = (
            (6.5, 27.0) if receipt.failure_stage == "final_judge" else (0.15, 1.5)
        )
        total += (
            receipt.captured_input_tokens * input_rate
            + receipt.captured_output_tokens * output_rate
        ) / 1_000_000
    return total


def _cumulative_observed_cost(control: dict, root: Path) -> float:
    return float(control.get("prior_dashscope_observed_cost_cny", 0.0)) + (
        _observed_cost(root)
    )


def _load_or_initialize_budget_ledger(
    *,
    execution_root: Path,
    control: Mapping[str, object],
    matrix_run_id: str,
):
    """Create the immutable authority on first execute, or validate a resume."""

    if control.get("schema_version") != 4 or any(
        control.get(field) != expected
        for field, expected in _ACTIVE_BUDGET_CONTRACT.items()
    ):
        raise ValueError("execution control hard-budget contract drifted")
    if (
        control.get("budget_ledger_relpath") != "budget-ledger"
        or control.get("budget_authority_relpath")
        != "budget-ledger/budget-authority.json"
        or control.get("budget_authority_creation_policy")
        != "create_only_on_first_execute"
    ):
        raise ValueError("execution control budget-ledger layout drifted")
    ledger_root = execution_root / "budget-ledger"
    cny_values: dict[str, Decimal] = {}
    for field in (
        "approved_dashscope_budget_cny",
        "phase_cumulative_cap_cny",
        "incremental_authorized_dashscope_budget_cny",
        "prior_dashscope_observed_cost_cny",
    ):
        raw = control.get(field)
        if not isinstance(raw, str):
            raise ValueError(f"execution control {field} must be a CNY string")
        try:
            value = Decimal(raw)
        except InvalidOperation as error:
            raise ValueError(
                f"execution control {field} must be a decimal CNY string"
            ) from error
        if not value.is_finite() or raw != format(value, ".12f") or value < 0:
            raise ValueError(
                f"execution control {field} must be a non-negative 12-place CNY string"
            )
        cny_values[field] = value
    phase_cap = cny_values["phase_cumulative_cap_cny"]
    prior_cost = cny_values["prior_dashscope_observed_cost_cny"]
    if (
        phase_cap > cny_values["approved_dashscope_budget_cny"]
        or cny_values["incremental_authorized_dashscope_budget_cny"]
        != phase_cap - prior_cost
        or prior_cost >= phase_cap
    ):
        raise ValueError("execution control CNY authority is inconsistent")
    authority_path = ledger_root / "budget-authority.json"
    if not authority_path.exists():
        try:
            initialize_portfolio_budget_ledger(
                ledger_root,
                matrix_run_id=matrix_run_id,
                phase_cap_cny=phase_cap,
                prior_observed_cost_cny=prior_cost,
            )
        except FileExistsError:
            # Another shard may have atomically published the create-only
            # authority after our existence check.  The unconditional load and
            # exact control checks below decide whether that winner is valid.
            pass
    ledger = load_portfolio_budget_ledger(ledger_root)
    if (
        ledger.authority.matrix_run_id != matrix_run_id
        or ledger.authority.phase_cap_cny != phase_cap
        or ledger.authority.prior_observed_cost_cny != prior_cost
    ):
        raise ValueError("create-only budget authority differs from control")
    return ledger_root, open_portfolio_budget_ledger_session(ledger_root)


def _shared_route_path(root: Path, shard, member) -> Path:
    return root / "shared-routes" / shard.accepted_batch_id / f"{member.query_id}.json"


def _is_retryable_route_contract_failure(
    response: AssistantBackendResponse,
) -> bool:
    """Return whether one captured route response is safe to correct once.

    Syntax/shape failures, truncation, and empty output qualify for one fixed
    repair prompt. Semantic route failures, tool calls, finish-reason drift,
    response-contract drift, and budget failures remain terminal and fail closed.
    """

    if response.error_code not in {"route_contract_error", "route_length"}:
        return False
    return _is_retryable_route_failure_shape(
        response.route_failure_subtype,
        response.route_failure_shape,
    )


def _is_retryable_route_failure_shape(
    failure_subtype: str | None,
    failure_shape: object | None,
) -> bool:
    if failure_subtype not in _ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES:
        return False
    payload_status = getattr(failure_shape, "payload_status", None)
    if failure_subtype == "invalid_route_json":
        return payload_status in _ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES
    if failure_subtype == "length":
        return payload_status == "not_examined"
    return payload_status == "empty"


def _route_contract_attempt_failure_subtype(
    failure_subtype: str | None,
    failure_shape: object | None,
) -> str:
    """Map one eligible captured failure to its stable v3 receipt subtype."""

    if not _is_retryable_route_failure_shape(failure_subtype, failure_shape):
        raise ValueError("route failure is not eligible for fixed repair")
    if failure_subtype == "length":
        return "route_contract_length"
    if failure_subtype == "response_empty_text":
        return "route_contract_empty"
    return f"route_contract_{getattr(failure_shape, 'payload_status')}"


def _is_retryable_shared_route_contract_failure(
    artifact: SharedStage2RouteArtifact,
) -> bool:
    if artifact.status != "terminal_route_error":
        return False
    return _is_retryable_route_failure_shape(
        artifact.failure_subtype,
        artifact.failure_shape,
    )


def _route_contract_retry_available(
    shard_root: Path,
    *,
    query_ordinal: int,
    query_id: str,
    instance_sha256: str,
) -> bool:
    """Bound correction by the existing two-attempt query ceiling."""

    receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=query_ordinal,
        query_id=query_id,
        instance_sha256=instance_sha256,
    )
    return len(receipts) + 1 < PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY


def _repair_of_route_wire_sha256(
    receipts: tuple[PortfolioAttemptReceipt, ...],
    *,
    request_sha256: str,
) -> str | None:
    """Return the captured first Route wire bound by one fixed repair."""

    if not receipts:
        return None
    latest = receipts[-1]
    if latest.request_sha256 != request_sha256:
        raise ValueError("route retry receipt differs from the frozen request")
    if not latest.failure_subtype.startswith("route_contract_"):
        return None
    evidence = latest.route_call_evidence
    if evidence is None:
        # Immutable v1 receipts remain replayable.  New v2 format-retry
        # receipts cannot reach this branch because their schema requires the
        # exact wire request evidence.
        return None
    return evidence.wire_request_sha256


def _validated_route_identity(
    request: AssistantRequestSnapshot,
    response: AssistantBackendResponse,
) -> ValidatedRouteIdentity | None:
    values = (
        response.selected_capability,
        response.skill_slug,
        response.route_trace_sha256,
        request.treatment.bank_sha256,
    )
    if all(value is not None for value in values):
        return ValidatedRouteIdentity(
            selected_capability=response.selected_capability,
            skill_slug=response.skill_slug,
            route_trace_sha256=response.route_trace_sha256,
            bank_sha256=request.treatment.bank_sha256,
        )
    return None


def _record_orphaned_provider_call_if_present(
    *,
    ledger_root: Path,
    shard_root: Path,
    breaker: ProviderPreResponseCircuitBreaker,
    matrix_run_id: str,
    shard_id: str,
    config_name: str,
    instance_sha256: str,
    query_id: str,
    query_ordinal: int,
    request_sha256: str,
    attempt_index: int,
    stages: frozenset[str],
) -> PortfolioAttemptReceipt | None:
    """Convert a crash-window ledger call into a retryable non-score receipt."""

    ledger = load_portfolio_budget_ledger(ledger_root)
    matching = tuple(
        reservation
        for reservation in ledger.reservations
        if (
            reservation.identity.matrix_run_id == matrix_run_id
            and reservation.identity.shard_id == shard_id
            and reservation.identity.config == config_name
            and reservation.identity.query_id == query_id
            and reservation.identity.instance_sha256 == instance_sha256
            and reservation.identity.request_sha256 == request_sha256
            and reservation.identity.attempt_index == attempt_index
            and reservation.identity.stage in stages
        )
    )
    if not matching:
        return None
    matching_sha256s = {item.reservation_sha256 for item in matching}
    settlements = tuple(
        item
        for item in ledger.settlements
        if item.reservation_sha256 in matching_sha256s
    )
    forfeits = tuple(
        item for item in ledger.forfeits if item.reservation_sha256 in matching_sha256s
    )
    unresolved = tuple(
        item
        for item in ledger.unresolved_reservations
        if item.reservation_sha256 in matching_sha256s
    )
    if len(unresolved) > 1 or len(forfeits) > 1:
        raise ValueError("orphaned provider-call scope is not uniquely terminalizable")
    if unresolved:
        appended, _ = forfeit_portfolio_provider_call(
            ledger_root,
            reservation_sha256=unresolved[0].reservation_sha256,
            reason="orphan_recovered_after_owner_exit",
        )
        forfeits = (*forfeits, appended)
    if len(matching) != len(settlements) + len(forfeits):
        raise ValueError("orphaned provider calls lack unique terminal outcomes")
    forfeit = forfeits[0] if forfeits else None
    if forfeit is not None:
        last = next(
            item
            for item in matching
            if item.reservation_sha256 == forfeit.reservation_sha256
        )
    else:
        last = max(matching, key=lambda item: item.ledger_event_index)
    receipt, _ = create_retryable_attempt_receipt(
        shard_root,
        breaker=breaker,
        matrix_run_id=matrix_run_id,
        shard_id=shard_id,
        config=config_name,
        instance_sha256=instance_sha256,
        query_id=query_id,
        query_ordinal=query_ordinal,
        request_sha256=request_sha256,
        failure_stage=last.identity.stage,
        failure_subtype="orphaned_provider_call",
        circuit_id=f"orphaned:{last.identity.provider}:{last.identity.model}",
        captured_provider_response_count=len(settlements),
        captured_input_tokens=sum(item.actual_input_tokens for item in settlements),
        captured_output_tokens=sum(item.actual_output_tokens for item in settlements),
        forfeited_reservation_sha256=(
            None if forfeit is None else forfeit.reservation_sha256
        ),
        budget_forfeit_sha256=(None if forfeit is None else forfeit.forfeit_sha256),
        exception_type=PortfolioBudgetOrphanedCallError.__qualname__,
    )
    if receipt.attempt_index != attempt_index:
        raise ValueError("orphaned provider call attempt identity drifted")
    return receipt


def _recoverable_stop(
    *,
    shard_id: str,
    query_id: str,
    reason: str,
    circuit_id: str | None = None,
    recovery_policy: str = "resume_earliest_incomplete_frozen_member",
) -> int:
    print(
        json.dumps(
            {
                "status": "recoverable_incomplete",
                "shard_id": shard_id,
                "query_id": query_id,
                "reason": reason,
                "circuit_id": circuit_id,
                "score_disposition": "not_scored_no_fixed_zero",
                "resume_policy": recovery_policy,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    try:
        control = _load_control(args.execution_root)
        gcs_v2_checkpoint = _requires_gcs_v2_checkpoint(control)
        static_opt = control.get("execution_scope") == "static_opt_rollout"
        launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        parallel_profile = None
        provider_gate = None
        if args.parallel_profile is not None:
            parallel_profile = load_parallel_profile(args.parallel_profile)
            validate_parallel_profile_sources(
                parallel_profile,
                repository_root=REPOSITORY_ROOT,
            )
            if (
                parallel_profile.matrix_run_id != launch.plan.matrix_run_id
                or parallel_profile.launch_plan_sha256 != launch.plan.launch_plan_sha256
                or parallel_profile.execution_scope != control.get("execution_scope")
            ):
                raise ValueError(
                    "parallel profile does not bind this launch and execution scope"
                )
            provider_gate = PortfolioAggregateProviderGate(
                args.execution_root / parallel_profile.gate_db_relpath,
                assistant_concurrency=parallel_profile.assistant_concurrency,
                final_judge_concurrency=parallel_profile.final_judge_concurrency,
                qwen_requests_per_minute_cap=(
                    parallel_profile.qwen_requests_per_minute_cap
                ),
                qwen_rate_limit_policy=parallel_profile.qwen_rate_limit_policy,
            )
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
            raise ValueError("launch hard-budget contract drifted")
        shard = next(
            item for item in launch.plan.shards if item.shard_id == args.shard_id
        )
        authorized_shard_ids = tuple(control.get("authorized_shard_ids", ()))
        if authorized_shard_ids and shard.shard_id not in authorized_shard_ids:
            raise ValueError("selected shard is outside the execution authorization")
        frozen = control.get("external_frozen_shard")
        if isinstance(frozen, dict) and shard.shard_id == frozen.get("target_shard_id"):
            raise ValueError("external frozen shard must not be executed locally")
        control_aliases = control.get("execution_artifact_aliases", [])
        if not isinstance(control_aliases, list) or any(
            not isinstance(item, dict) for item in control_aliases
        ):
            raise ValueError("execution artifact aliases must be a list of objects")
        target_aliases = tuple(
            item
            for item in control_aliases
            if item.get("target_config") == shard.config
        )
        if len(target_aliases) > 1:
            raise ValueError("shard config has ambiguous execution artifact aliases")
        if args.execute and target_aliases:
            alias = target_aliases[0]
            if (
                alias.get("provider_model_call_count") != 0
                or alias.get("reuse_scope") != "assistant_and_evaluator_query_artifacts"
            ):
                raise ValueError("execution artifact alias contract is invalid")
            raise ValueError(
                "execution artifact alias must reuse its source shard without "
                "a local provider call"
            )
        members = tuple(
            item for item in launch.instances if item.shard_id == shard.shard_id
        )
        if len(members) != 25:
            raise ValueError("selected shard does not contain exactly 25 instances")
        runtime_root = Path(control["runtime_root"])
        if static_opt:
            verified_static_runtime = load_verified_portfolio_static_opt_runtime(
                runtime_root,
                expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
            )
            runtime_lock = dict(verified_static_runtime.runtime_lock)
            validate_portfolio_static_opt_execution_control(
                control,
                verified_static_runtime,
            )
        else:
            verified_treatment_runtime = load_verified_portfolio_treatment_runtime(
                runtime_root,
                expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
            )
            if (
                control.get("treatment_chain_manifest_file_sha256")
                != verified_treatment_runtime.manifest_file_sha256
                or control.get("treatment_chain_sha256")
                != verified_treatment_runtime.chain.manifest.chain_sha256
                or control.get("treatment_record_sha256s")
                != {
                    item.config: item.receipt_sha256
                    for item in verified_treatment_runtime.chain.manifest.records
                }
                or control_aliases
                != [
                    item.model_dump(mode="json")
                    for item in getattr(
                        verified_treatment_runtime.chain,
                        "execution_artifact_aliases",
                        (),
                    )
                ]
            ):
                raise ValueError(
                    "execution control differs from the verified treatment chain"
                )
            runtime_lock = dict(verified_treatment_runtime.runtime_lock)
            _require_active_final_result_runtime(runtime_lock)
        if gcs_v2_checkpoint:
            _require_active_gcs_v2_runtime(runtime_lock)
        inputs = _inputs_for_launch(launch.plan)
        if args.legacy_launch_compat and launch.plan.kind in {
            "portfolio-core-split-x5-launch-plan",
            "portfolio-core-static-opt-800x1-launch-plan",
        }:
            raise ValueError("Core launch cannot use the legacy public projection")
        runtime, banks, catalog, runner = _build_runner(
            runtime_root,
            control["runtime_lock_file_sha256"],
            inputs=inputs,
            static_opt=static_opt,
            qwen_call_start_waiter=(
                None
                if provider_gate is None
                else provider_gate.wait_for_qwen_call_start
            ),
        )
        rubric = None
        if not static_opt:
            rubric_bytes = Path(control["rubric_path"]).read_bytes()
            if sha256_bytes(rubric_bytes) != control["rubric_file_sha256"]:
                raise ValueError("rubric file digest mismatch")
            rubric = RubricSnapshot.model_validate_json(rubric_bytes, strict=True)
        if args.legacy_launch_compat:
            public_by_id = {
                item.query_id: build_legacy_assistant_query_input(item)
                for item in inputs.queries
            }
        else:
            public_by_id = {item.query_id: item for item in inputs.assistant_queries}
        private_by_id = {item.query_id: item for item in inputs.queries}
        judge_runtime = (
            None if static_opt else inputs.runtime_for("aifast-gemini-judge")
        )
        isolation = (
            None if static_opt else make_active_portfolio_evaluator_isolation_lock()
        )
        backbone_payload = {
            "provider": "qwen",
            "model": "qwen3-vl-flash-2026-01-22",
            "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": None,
            "system_prompt_sha256": sha256_bytes(
                PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")
            ),
        }
        backbone = _self_model(BackboneLock, backbone_payload, "identity_sha256")
        budget_payload = {
            "max_input_tokens": 32768,
            "max_output_tokens": 4096,
            "max_tool_calls": 3,
            "max_turns": 5,
            "timeout_ms": 180000,
        }
        budget = _self_model(InferenceBudget, budget_payload, "budget_sha256")
        registry_lock = _registry_lock(runtime.registry)
        bank_sha = (
            None if shard.config == "noskill" else banks[shard.config].bank_sha256
        )
        treatment = _treatment(shard.config, bank_sha)
        requests = [
            _request(
                launch=launch,
                shard=shard,
                member=member,
                query=public_by_id[member.query_id],
                treatment=treatment,
                backbone=backbone,
                budget=budget,
                registry_lock=registry_lock,
            )
            for member in members
        ]
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "dry_run_passed",
                        "model_calls_performed": 0,
                        "shard_id": shard.shard_id,
                        "config": shard.config,
                        "query_count": len(requests),
                        "runtime_lock_sha256": control["runtime_lock_sha256"],
                        "gcs_policy_sha256": (
                            runtime_lock["gcs_policy_sha256"] if static_opt else None
                        ),
                        "rubric_content_sha256": (
                            None if rubric is None else rubric.content_sha256
                        ),
                        "estimated_call_range": [
                            len(requests) * 2,
                            len(requests) * 6,
                        ],
                        "retryable_infrastructure_policy": {
                            "circuit_breaker_threshold": (
                                PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
                            ),
                            "max_attempts_per_query": (
                                PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
                            ),
                            "automatic_same_process_retry": False,
                            "score_disposition": "not_scored_no_fixed_zero",
                            "resume_policy": (
                                "resume_earliest_incomplete_frozen_member"
                            ),
                            "route_contract_retry": {
                                "retry_count": 1,
                                "eligible_payload_statuses": sorted(
                                    _ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES
                                ),
                                "eligible_failure_subtypes": sorted(
                                    _ROUTE_CONTRACT_RETRYABLE_FAILURE_SUBTYPES
                                ),
                                "wire_policy": "distinct_fixed_repair_prompt",
                                "response_prefix_recovery": False,
                                "shared_route_owner_config": (
                                    _SHARED_ROUTE_OWNER_CONFIG
                                ),
                                "exhausted_disposition": ("terminal_fixed_zero"),
                            },
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        ledger_root, budget_session = _load_or_initialize_budget_ledger(
            execution_root=args.execution_root,
            control=control,
            matrix_run_id=launch.plan.matrix_run_id,
        )
        if budget_session.state.remaining_cost_cny <= Decimal("0"):
            return _recoverable_stop(
                shard_id=shard.shard_id,
                query_id=members[0].query_id,
                reason="budget_cap_reached_before_provider_call",
                recovery_policy="new_authorized_execution_phase_required",
            )
        key = (args.execution_root / "blinding-key.bin").read_bytes()
        breaker = ProviderPreResponseCircuitBreaker()
        shard_root = args.execution_root / shard.output_relpath
        completed = 0
        recoverable_incomplete = 0
        for member, request in zip(members, requests, strict=True):
            assistant_path = args.execution_root / member.assistant_output_relpath
            final_path = args.execution_root / member.final_output_relpath
            assistant_path.parent.mkdir(parents=True, exist_ok=True)
            if not static_opt:
                final_path.parent.mkdir(parents=True, exist_ok=True)
            # These are also needed when resuming from an existing Assistant
            # checkpoint whose final artifact is still missing.
            prebuilt_packet: FinalEvaluationPacket | None = None
            packet_error_code: str | None = None
            if assistant_path.exists():
                if gcs_v2_checkpoint:
                    response, receipt, scorer_evidence = _assistant_row(
                        assistant_path,
                        require_schema_v2=True,
                        include_scorer_evidence=True,
                    )
                    if scorer_evidence is None:  # pragma: no cover - loader enforces
                        raise AssertionError("GCS v2 checkpoint lost scorer evidence")
                    _verify_scorer_sidecar_binding(
                        scorer_evidence,
                        launch=launch,
                        member=member,
                        request=request,
                        result=_assistant_result(launch, request, response),
                        receipt=receipt,
                    )
                else:
                    response, receipt = _assistant_row(assistant_path)
            else:
                prior_attempt_receipts = load_query_attempt_receipts(
                    shard_root,
                    query_ordinal=member.query_ordinal,
                    query_id=member.query_id,
                    instance_sha256=member.instance_sha256,
                )
                try:
                    assistant_attempt_index = require_retryable_attempt_available(
                        shard_root,
                        query_ordinal=member.query_ordinal,
                        query_id=member.query_id,
                        instance_sha256=member.instance_sha256,
                    )
                except PortfolioAttemptLimitError:
                    return _recoverable_stop(
                        shard_id=shard.shard_id,
                        query_id=member.query_id,
                        reason="query_retryable_attempt_limit_reached",
                    )
                assistant_budget_context = PortfolioAssistantBudgetContext(
                    ledger_root=ledger_root,
                    shard_id=shard.shard_id,
                    instance_sha256=member.instance_sha256,
                    attempt_index=assistant_attempt_index,
                    repair_of_route_wire_request_sha256=(
                        _repair_of_route_wire_sha256(
                            prior_attempt_receipts,
                            request_sha256=request.request_sha256,
                        )
                    ),
                )
                shared_route = None
                if shard.config in {"s1s2", "full"}:
                    shared_route_path = _shared_route_path(
                        args.execution_root, shard, member
                    )
                    if shared_route_path.exists():
                        shared_route = SharedStage2RouteArtifact.model_validate_json(
                            shared_route_path.read_bytes(),
                            strict=True,
                        )
                    elif shard.config != _SHARED_ROUTE_OWNER_CONFIG:
                        return _recoverable_stop(
                            shard_id=shard.shard_id,
                            query_id=member.query_id,
                            reason="shared_route_owner_incomplete",
                            recovery_policy=(
                                "run_s1s2_shared_route_owner_then_resume_current_member"
                            ),
                        )
                    else:
                        try:
                            with _provider_slot(
                                provider_gate,
                                "assistant",
                                label=f"{shard.shard_id}:{member.query_id}:shared-route",
                            ):
                                shared_route = runner.prepare_shared_stage2_route(
                                    request,
                                    budget_context=assistant_budget_context,
                                )
                        except PortfolioBudgetExceededError:
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="budget_cap_reached_before_provider_call",
                                recovery_policy=(
                                    "new_authorized_execution_phase_required"
                                ),
                            )
                        except PortfolioBudgetDuplicateCallError:
                            orphaned = _record_orphaned_provider_call_if_present(
                                ledger_root=ledger_root,
                                shard_root=shard_root,
                                breaker=breaker,
                                matrix_run_id=launch.plan.matrix_run_id,
                                shard_id=shard.shard_id,
                                config_name=shard.config,
                                instance_sha256=member.instance_sha256,
                                query_id=member.query_id,
                                query_ordinal=member.query_ordinal,
                                request_sha256=request.request_sha256,
                                attempt_index=assistant_attempt_index,
                                stages=frozenset({"shared_route"}),
                            )
                            if orphaned is None:
                                raise
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="orphaned_provider_call_recorded",
                                circuit_id=orphaned.circuit_id,
                            )
                        except AssistantProviderPreResponseError as error:
                            receipt_record, _ = create_retryable_attempt_receipt(
                                shard_root,
                                breaker=breaker,
                                matrix_run_id=launch.plan.matrix_run_id,
                                shard_id=shard.shard_id,
                                config=shard.config,
                                instance_sha256=member.instance_sha256,
                                query_id=member.query_id,
                                query_ordinal=member.query_ordinal,
                                request_sha256=request.request_sha256,
                                failure_stage="shared_route",
                                failure_subtype="provider_pre_response",
                                circuit_id=(
                                    f"assistant:{request.backbone.provider}:"
                                    f"{request.backbone.model}"
                                ),
                                forfeited_reservation_sha256=(
                                    error.forfeited_reservation_sha256
                                ),
                                budget_forfeit_sha256=error.budget_forfeit_sha256,
                                exception_type=error.provider_exception_type,
                            )
                            recoverable_incomplete += 1
                            if receipt_record.circuit_open:
                                return _recoverable_stop(
                                    shard_id=shard.shard_id,
                                    query_id=member.query_id,
                                    reason="provider_pre_response_circuit_open",
                                    circuit_id=receipt_record.circuit_id,
                                )
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="retryable_provider_failure_recorded",
                                circuit_id=receipt_record.circuit_id,
                            )
                        breaker.record_valid_response(
                            f"assistant:{request.backbone.provider}:"
                            f"{request.backbone.model}"
                        )
                        if _is_retryable_shared_route_contract_failure(
                            shared_route
                        ) and _route_contract_retry_available(
                            shard_root,
                            query_ordinal=member.query_ordinal,
                            query_id=member.query_id,
                            instance_sha256=member.instance_sha256,
                        ):
                            route_call = shared_route.route_call
                            receipt_record, _ = create_retryable_attempt_receipt(
                                shard_root,
                                breaker=breaker,
                                matrix_run_id=launch.plan.matrix_run_id,
                                shard_id=shard.shard_id,
                                config=shard.config,
                                instance_sha256=member.instance_sha256,
                                query_id=member.query_id,
                                query_ordinal=member.query_ordinal,
                                request_sha256=request.request_sha256,
                                failure_stage="shared_route",
                                failure_subtype=(
                                    _route_contract_attempt_failure_subtype(
                                        shared_route.failure_subtype,
                                        shared_route.failure_shape,
                                    )
                                ),
                                circuit_id=(
                                    "assistant_route_contract:"
                                    f"{request.backbone.provider}:"
                                    f"{request.backbone.model}"
                                ),
                                captured_provider_response_count=1,
                                captured_input_tokens=route_call.input_tokens,
                                captured_output_tokens=route_call.output_tokens,
                                route_call_evidence=(shared_route.route_call_evidence),
                            )
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason=("retryable_route_contract_failure_recorded"),
                                circuit_id=receipt_record.circuit_id,
                            )
                        shared_route_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_create_file(
                            shared_route_path,
                            canonical_json_bytes(shared_route.model_dump(mode="json")),
                        )
                execution_stages = (
                    frozenset({"assistant_action"})
                    if shard.config in {"noskill", "s1s2", "full"}
                    else frozenset({"assistant_route", "assistant_action"})
                )
                try:
                    with _provider_slot(
                        provider_gate,
                        "assistant",
                        label=f"{shard.shard_id}:{member.query_id}:assistant",
                    ):
                        execution = runner.execute(
                            request,
                            shared_stage2_route=shared_route,
                            budget_context=assistant_budget_context,
                            scorer_query=(
                                private_by_id[member.query_id]
                                if gcs_v2_checkpoint
                                else None
                            ),
                        )
                except PortfolioBudgetExceededError:
                    return _recoverable_stop(
                        shard_id=shard.shard_id,
                        query_id=member.query_id,
                        reason="budget_cap_reached_before_provider_call",
                        recovery_policy="new_authorized_execution_phase_required",
                    )
                except PortfolioBudgetDuplicateCallError:
                    orphaned = _record_orphaned_provider_call_if_present(
                        ledger_root=ledger_root,
                        shard_root=shard_root,
                        breaker=breaker,
                        matrix_run_id=launch.plan.matrix_run_id,
                        shard_id=shard.shard_id,
                        config_name=shard.config,
                        instance_sha256=member.instance_sha256,
                        query_id=member.query_id,
                        query_ordinal=member.query_ordinal,
                        request_sha256=request.request_sha256,
                        attempt_index=assistant_attempt_index,
                        stages=execution_stages,
                    )
                    if orphaned is None:
                        raise
                    return _recoverable_stop(
                        shard_id=shard.shard_id,
                        query_id=member.query_id,
                        reason="orphaned_provider_call_recorded",
                        circuit_id=orphaned.circuit_id,
                    )
                response, receipt = execution.response, execution.receipt
                if response.error_code is not None and response.error_code.startswith(
                    "provider_pre_response_"
                ):
                    terminalize_static_gcs_retry = (
                        static_opt
                        and gcs_v2_checkpoint
                        and assistant_attempt_index
                        == PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
                    )
                    failure_stage = (
                        "assistant_route"
                        if response.error_code == "provider_pre_response_route"
                        else "assistant_action"
                    )
                    receipt_record, _ = create_retryable_attempt_receipt(
                        shard_root,
                        breaker=breaker,
                        matrix_run_id=launch.plan.matrix_run_id,
                        shard_id=shard.shard_id,
                        config=shard.config,
                        instance_sha256=member.instance_sha256,
                        query_id=member.query_id,
                        query_ordinal=member.query_ordinal,
                        request_sha256=request.request_sha256,
                        failure_stage=failure_stage,
                        failure_subtype=response.error_code,
                        circuit_id=(
                            f"assistant:{request.backbone.provider}:"
                            f"{request.backbone.model}"
                        ),
                        # On terminal exhaustion the immutable Assistant
                        # checkpoint below owns all captured responses and
                        # usage.  The attempt receipt owns only the uncaptured
                        # pre-response invocation/forfeit, avoiding duplicate
                        # execution-call accounting.
                        captured_provider_response_count=(
                            0
                            if terminalize_static_gcs_retry
                            else len(receipt.model_calls)
                        ),
                        captured_input_tokens=(
                            0
                            if terminalize_static_gcs_retry
                            else receipt.aggregate_usage.input_tokens
                        ),
                        captured_output_tokens=(
                            0
                            if terminalize_static_gcs_retry
                            else receipt.aggregate_usage.output_tokens
                        ),
                        assistant_response_sha256=receipt.response_sha256,
                        assistant_receipt_sha256=receipt.receipt_sha256,
                        forfeited_reservation_sha256=(
                            response.forfeited_reservation_sha256
                        ),
                        budget_forfeit_sha256=response.budget_forfeit_sha256,
                        exception_type=getattr(
                            response, "provider_exception_type", None
                        ),
                        validated_route_identity=_validated_route_identity(
                            request, response
                        ),
                    )
                    if receipt_record.attempt_index != assistant_attempt_index:
                        raise ValueError(
                            "Assistant retry receipt differs from the in-memory "
                            "execution attempt"
                        )
                    if not terminalize_static_gcs_retry:
                        recoverable_incomplete += 1
                        if receipt_record.circuit_open:
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="provider_pre_response_circuit_open",
                                circuit_id=receipt_record.circuit_id,
                            )
                        return _recoverable_stop(
                            shard_id=shard.shard_id,
                            query_id=member.query_id,
                            reason="retryable_provider_failure_recorded",
                            circuit_id=receipt_record.circuit_id,
                        )
                    # Static opt has no downstream Judge that can terminalize an
                    # exhausted Assistant failure.  On the second and final
                    # retryable attempt, retain the just-produced response,
                    # receipt, and scorer capture below and publish one normal
                    # schema-v2 checkpoint.  Its hard error deterministically
                    # makes GCS zero.  A historical pair of attempt receipts is
                    # deliberately not sufficient to reconstruct this row.
                if receipt.model_calls:
                    breaker.record_valid_response(
                        f"assistant:{request.backbone.provider}:"
                        f"{request.backbone.model}"
                    )
                retryable_route_contract_failure = shard.config not in {
                    "s1s2",
                    "full",
                } and _is_retryable_route_contract_failure(response)
                if retryable_route_contract_failure:
                    route_retry_available = _route_contract_retry_available(
                        shard_root,
                        query_ordinal=member.query_ordinal,
                        query_id=member.query_id,
                        instance_sha256=member.instance_sha256,
                    )
                    terminalize_static_gcs_retry = (
                        static_opt
                        and gcs_v2_checkpoint
                        and assistant_attempt_index
                        == PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
                    )
                else:
                    route_retry_available = False
                    terminalize_static_gcs_retry = False
                if route_retry_available or terminalize_static_gcs_retry:
                    # The active attempt-v4 schema intentionally reserves the
                    # ``route_contract_*`` family for an initial response whose
                    # exact wire evidence authorizes a repair.  The exhausted
                    # fixed-repair response is already bound in the Assistant
                    # receipt that will be checkpointed below, so its attempt
                    # receipt uses a distinct terminal subtype and does not
                    # duplicate that evidence.
                    exhausted_route_failure = (
                        terminalize_static_gcs_retry and not route_retry_available
                    )
                    receipt_record, _ = create_retryable_attempt_receipt(
                        shard_root,
                        breaker=breaker,
                        matrix_run_id=launch.plan.matrix_run_id,
                        shard_id=shard.shard_id,
                        config=shard.config,
                        instance_sha256=member.instance_sha256,
                        query_id=member.query_id,
                        query_ordinal=member.query_ordinal,
                        request_sha256=request.request_sha256,
                        failure_stage="assistant_route",
                        failure_subtype=(
                            "retryable_route_contract_exhausted"
                            if exhausted_route_failure
                            else _route_contract_attempt_failure_subtype(
                                response.route_failure_subtype,
                                response.route_failure_shape,
                            )
                        ),
                        circuit_id=(
                            "assistant_route_contract:"
                            f"{request.backbone.provider}:"
                            f"{request.backbone.model}"
                        ),
                        # The terminal schema-v2 checkpoint, rather than this
                        # exhaustion marker, owns the captured fixed-repair
                        # response and its usage.
                        captured_provider_response_count=(
                            0
                            if terminalize_static_gcs_retry
                            else len(receipt.model_calls)
                        ),
                        captured_input_tokens=(
                            0
                            if terminalize_static_gcs_retry
                            else receipt.aggregate_usage.input_tokens
                        ),
                        captured_output_tokens=(
                            0
                            if terminalize_static_gcs_retry
                            else receipt.aggregate_usage.output_tokens
                        ),
                        assistant_response_sha256=receipt.response_sha256,
                        assistant_receipt_sha256=receipt.receipt_sha256,
                        route_call_evidence=(
                            None
                            if exhausted_route_failure
                            else receipt.route_call_evidence
                        ),
                    )
                    if receipt_record.attempt_index != assistant_attempt_index:
                        raise ValueError(
                            "Assistant route retry receipt differs from the "
                            "in-memory execution attempt"
                        )
                    if not terminalize_static_gcs_retry:
                        return _recoverable_stop(
                            shard_id=shard.shard_id,
                            query_id=member.query_id,
                            reason="retryable_route_contract_failure_recorded",
                            circuit_id=receipt_record.circuit_id,
                        )
                assistant_result = (
                    _assistant_result(launch, request, response)
                    if gcs_v2_checkpoint
                    else None
                )
                scorer_evidence = None
                if gcs_v2_checkpoint:
                    if (
                        execution.scorer_capture_policy_version
                        != GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
                    ):
                        raise ValueError(
                            "Assistant execution omitted required GCS v2 capture"
                        )
                    multi_item_sets = tuple(
                        expected
                        for call in execution.scorer_calls
                        if (expected := expected_multi_items_from_call_v2(call))
                        is not None
                    )
                    if len(multi_item_sets) > 1:
                        raise ValueError(
                            "GCS v2 terminal contains multiple multi-product calls"
                        )
                    scorer_evidence = make_public_scorer_evidence_v2(
                        matrix_run_id=launch.plan.matrix_run_id,
                        instance_id=member.instance_sha256,
                        request_sha256=request.request_sha256,
                        query_id=member.query_id,
                        config=member.config,
                        query_artifact_sha256=launch.plan.query_artifact_sha256,
                        assistant_result_sha256=_hash(_model(assistant_result)),
                        assistant_receipt_sha256=receipt.receipt_sha256,
                        calls=execution.scorer_calls,
                        expected_multi_items=(
                            multi_item_sets[0] if multi_item_sets else None
                        ),
                    )
                    _verify_scorer_sidecar_binding(
                        scorer_evidence,
                        launch=launch,
                        member=member,
                        request=request,
                        result=assistant_result,
                        receipt=receipt,
                    )
                if (
                    response.error_code is None
                    and not final_path.exists()
                    and control.get("execution_scope") != "static_opt_rollout"
                ):
                    prebuilt_packet, packet_error_code = (
                        _build_final_packet_fail_closed(
                            private_by_id[member.query_id],
                            (
                                assistant_result
                                if assistant_result is not None
                                else _assistant_result(launch, request, response)
                            ),
                            asset_catalog=catalog,
                            rubric=rubric,
                            blinding_key=key,
                        )
                    )
                    if packet_error_code is not None:
                        # The runner-owned receipt already binds the immutable
                        # Assistant response.  Reclassifying that response here
                        # would leave ``receipt.response_sha256`` describing
                        # different bytes.  Packet isolation is a trusted
                        # runtime contract, not a model outcome, so fail before
                        # publishing either checkpoint.
                        raise ValueError(
                            "final packet isolation failed after the immutable "
                            f"Assistant receipt was issued: {packet_error_code}"
                        )
                if (
                    type(response) is AssistantBackendResponse
                    and type(receipt) is AssistantExecutionReceipt
                    and receipt.response_sha256 != _hash(_model(response))
                ):
                    raise ValueError(
                        "Assistant response differs from its runner-owned receipt"
                    )
                row_payload = {
                    "schema_version": 2 if gcs_v2_checkpoint else 1,
                    "kind": "portfolio-assistant-checkpoint",
                    "instance_sha256": member.instance_sha256,
                    "query_ordinal": member.query_ordinal,
                    "request": _model(request),
                    "response": _model(response),
                    "receipt": _model(receipt),
                }
                if scorer_evidence is not None:
                    row_payload["public_scorer_evidence"] = _model(scorer_evidence)
                atomic_create_file(
                    assistant_path,
                    canonical_json_bytes(
                        {**row_payload, "row_sha256": _hash(row_payload)}
                    ),
                )
            if control.get("execution_scope") == "static_opt_rollout":
                # Static discovery is intentionally Assistant+GCS only.  The
                # immutable schema-v2 checkpoint above is the terminal row;
                # neither the legacy visual Final Judge nor Pairwise is part
                # of this execution scope.
                completed += 1
                budget_ledger = budget_session.state
                cost = float(portfolio_budget_settled_cost_cny(budget_ledger))
                cumulative_cost = float(
                    budget_ledger.authority.prior_observed_cost_cny
                    + budget_ledger.settled_actual_cost_cny
                )
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "query_id": member.query_id,
                            "observed_dashscope_cost_cny": round(cost, 8),
                            "cumulative_dashscope_cost_cny": round(cumulative_cost, 8),
                            "budget_accountable_cost_cny": format(
                                budget_ledger.accountable_cost_cny, ".12f"
                            ),
                            "budget_remaining_cost_cny": format(
                                budget_ledger.remaining_cost_cny, ".12f"
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            # Packet isolation has already succeeded before an Assistant row
            # reaches this legacy Final-Judge path.  A sanitizer failure is a
            # trusted contract error and publishes neither checkpoint.
            if response.error_code is None or packet_error_code is not None:
                if final_path.exists():
                    existing_fixed_zero = _existing_fixed_zero_error(
                        final_path,
                        query_id=member.query_id,
                    )
                    if existing_fixed_zero is not None and (
                        existing_fixed_zero != "hidden_evaluation_identity"
                    ):
                        raise ValueError(
                            "successful Assistant has an unsupported fixed-zero "
                            f"checkpoint: {member.query_id}"
                        )
                    if existing_fixed_zero is None:
                        loaded_final = load_final_judge_evaluation_result(final_path)
                        if (
                            loaded_final.parser_policy_version
                            != FINAL_JUDGE_PARSER_POLICY_VERSION_V4
                            or loaded_final.parser_policy_sha256
                            != FINAL_JUDGE_PARSER_POLICY_SHA256_V4
                        ):
                            raise ValueError(
                                "final checkpoint uses a different parser policy"
                            )
                        _require_final_checkpoint_contract(
                            loaded_final,
                            runtime_lock=runtime_lock,
                            visible_card_count=len(response.visible_cards),
                        )
                else:
                    try:
                        judge_attempt_index = require_retryable_attempt_available(
                            shard_root,
                            query_ordinal=member.query_ordinal,
                            query_id=member.query_id,
                            instance_sha256=member.instance_sha256,
                        )
                    except PortfolioAttemptLimitError:
                        return _recoverable_stop(
                            shard_id=shard.shard_id,
                            query_id=member.query_id,
                            reason="query_retryable_attempt_limit_reached",
                        )
                    if prebuilt_packet is None and packet_error_code is None:
                        packet, packet_error_code = _build_final_packet_fail_closed(
                            private_by_id[member.query_id],
                            _assistant_result(launch, request, response),
                            asset_catalog=catalog,
                            rubric=rubric,
                            blinding_key=key,
                        )
                    else:
                        packet = prebuilt_packet
                    if packet_error_code is not None:
                        skipped = _final_fixed_zero(
                            member.query_id,
                            packet_error_code,
                        )
                        atomic_create_file(
                            final_path,
                            canonical_json_bytes(
                                {**skipped, "result_sha256": _hash(skipped)}
                            ),
                        )
                    else:
                        if packet is None:
                            raise AssertionError(
                                "successful final packet construction returned no packet"
                            )
                        try:
                            with _provider_slot(
                                provider_gate,
                                "final_judge",
                                label=f"{shard.shard_id}:{member.query_id}:judge",
                            ):
                                judge = run_visual_final_judge(
                                    packet,
                                    isolation,
                                    remote_runtime=judge_runtime,
                                    budget_context=FinalJudgeBudgetContext(
                                        ledger_root=ledger_root,
                                        matrix_run_id=launch.plan.matrix_run_id,
                                        shard_id=shard.shard_id,
                                        config=shard.config,
                                        query_id=member.query_id,
                                        instance_sha256=member.instance_sha256,
                                        request_sha256=request.request_sha256,
                                        attempt_index=judge_attempt_index,
                                    ),
                                    max_tokens=2048,
                                    record_usage=True,
                                )
                        except PortfolioBudgetExceededError:
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="budget_cap_reached_before_provider_call",
                                recovery_policy=(
                                    "new_authorized_execution_phase_required"
                                ),
                            )
                        except PortfolioBudgetDuplicateCallError:
                            orphaned = _record_orphaned_provider_call_if_present(
                                ledger_root=ledger_root,
                                shard_root=shard_root,
                                breaker=breaker,
                                matrix_run_id=launch.plan.matrix_run_id,
                                shard_id=shard.shard_id,
                                config_name=shard.config,
                                instance_sha256=member.instance_sha256,
                                query_id=member.query_id,
                                query_ordinal=member.query_ordinal,
                                request_sha256=request.request_sha256,
                                attempt_index=judge_attempt_index,
                                stages=frozenset({"final_judge"}),
                            )
                            if orphaned is None:
                                raise
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="orphaned_provider_call_recorded",
                                circuit_id=orphaned.circuit_id,
                            )
                        _require_final_checkpoint_contract(
                            judge,
                            runtime_lock=runtime_lock,
                            visible_card_count=len(response.visible_cards),
                        )
                        terminal_provider_pre_response = (
                            judge.outcome.status in {"provider_error", "timeout"}
                            and judge.request_id is None
                        )
                        if terminal_provider_pre_response:
                            captured_usage = judge.aggregate_usage
                            receipt_record, _ = create_retryable_attempt_receipt(
                                shard_root,
                                breaker=breaker,
                                matrix_run_id=launch.plan.matrix_run_id,
                                shard_id=shard.shard_id,
                                config=shard.config,
                                instance_sha256=member.instance_sha256,
                                query_id=member.query_id,
                                query_ordinal=member.query_ordinal,
                                request_sha256=request.request_sha256,
                                failure_stage="final_judge",
                                failure_subtype=(
                                    f"provider_pre_response_{judge.outcome.status}"
                                ),
                                circuit_id=(
                                    f"final_judge:{judge.provider}:{judge.model}"
                                ),
                                captured_provider_response_count=(
                                    judge.captured_response_count
                                ),
                                captured_input_tokens=captured_usage.input_tokens,
                                captured_output_tokens=captured_usage.output_tokens,
                                judge_result_sha256=judge.result_sha256,
                                forfeited_reservation_sha256=(
                                    judge.forfeited_reservation_sha256
                                ),
                                budget_forfeit_sha256=(judge.budget_forfeit_sha256),
                                validated_route_identity=_validated_route_identity(
                                    request, response
                                ),
                            )
                            recoverable_incomplete += 1
                            if receipt_record.circuit_open:
                                return _recoverable_stop(
                                    shard_id=shard.shard_id,
                                    query_id=member.query_id,
                                    reason="provider_pre_response_circuit_open",
                                    circuit_id=receipt_record.circuit_id,
                                )
                            return _recoverable_stop(
                                shard_id=shard.shard_id,
                                query_id=member.query_id,
                                reason="retryable_provider_failure_recorded",
                                circuit_id=receipt_record.circuit_id,
                            )
                        breaker.record_valid_response(
                            f"final_judge:{judge.provider}:{judge.model}"
                        )
                        write_final_judge_evaluation_result(final_path, judge)
            else:
                skipped = _final_fixed_zero(member.query_id, response.error_code)
                if not final_path.exists():
                    atomic_create_file(
                        final_path,
                        canonical_json_bytes(
                            {**skipped, "result_sha256": _hash(skipped)}
                        ),
                    )
            completed += 1
            budget_ledger = budget_session.state
            cost = float(portfolio_budget_settled_cost_cny(budget_ledger))
            cumulative_cost = float(
                budget_ledger.authority.prior_observed_cost_cny
                + budget_ledger.settled_actual_cost_cny
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "query_id": member.query_id,
                        "observed_dashscope_cost_cny": round(cost, 8),
                        "cumulative_dashscope_cost_cny": round(cumulative_cost, 8),
                        "budget_accountable_cost_cny": format(
                            budget_ledger.accountable_cost_cny, ".12f"
                        ),
                        "budget_remaining_cost_cny": format(
                            budget_ledger.remaining_cost_cny, ".12f"
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        if recoverable_incomplete:
            return _recoverable_stop(
                shard_id=shard.shard_id,
                query_id=next(
                    member.query_id
                    for member in members
                    if not (args.execution_root / member.final_output_relpath).exists()
                ),
                reason="retryable_infrastructure_failures_remain",
            )
        budget_ledger = budget_session.state
        observed_cost = float(portfolio_budget_settled_cost_cny(budget_ledger))
        cumulative_observed_cost = float(
            budget_ledger.authority.prior_observed_cost_cny
            + budget_ledger.settled_actual_cost_cny
        )
        summary_payload = {
            "schema_version": 1,
            "kind": "portfolio-shard-summary",
            "shard_id": shard.shard_id,
            "config": shard.config,
            "query_count": len(members),
            "status": "complete",
            "observed_dashscope_cost_cny": observed_cost,
            "prior_dashscope_observed_cost_cny": float(
                control.get("prior_dashscope_observed_cost_cny", 0.0)
            ),
            "cumulative_dashscope_cost_cny": cumulative_observed_cost,
            "budget_authority_sha256": budget_ledger.authority.authority_sha256,
            "budget_ledger_last_event_index": budget_ledger.last_event_index,
            "budget_ledger_last_event_sha256": budget_ledger.last_event_sha256,
            "budget_ledger_settled_actual_cost_cny": format(
                budget_ledger.settled_actual_cost_cny, ".12f"
            ),
            "budget_ledger_forfeited_reserved_cost_cny": format(
                budget_ledger.forfeited_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_unresolved_reserved_cost_cny": format(
                budget_ledger.unresolved_reserved_cost_cny, ".12f"
            ),
            "budget_ledger_accountable_cost_cny": format(
                budget_ledger.accountable_cost_cny, ".12f"
            ),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        summary_path = args.execution_root / shard.output_relpath / "shard-summary.json"
        if not summary_path.exists():
            atomic_create_file(
                summary_path,
                canonical_json_bytes(
                    {**summary_payload, "summary_sha256": _hash(summary_payload)}
                ),
            )
        print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
        print(f"portfolio-shard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
