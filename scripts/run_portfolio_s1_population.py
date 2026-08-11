"""Dry-run or execute the independent S1 replay/body-Gate population.

This is deliberately sequential.  The persisted call-level gate still caps
Assistant work at the globally approved value (two), while one process uses a
single permit.  ``--stop-after-shards 1`` is the paid canary; rerunning the
same command resumes from byte-verified schema-v2 checkpoints before making
any provider call.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import sys


platform.platform = lambda *args, **kwargs: "Windows"  # type: ignore[assignment]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import run_portfolio_shard as established  # noqa: E402
from skillchain import config  # noqa: E402
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    BackboneLock,
    InferenceBudget,
    MatrixTreatment,
    _registry_lock,
)
from skillchain.evaluation.portfolio_core_runtime_sources import (  # noqa: E402
    load_verified_portfolio_core_runtime_sources,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY,
    PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE,
    PortfolioAttemptLimitError,
    PortfolioBudgetDuplicateCallError,
    PortfolioBudgetExceededError,
    ProviderPreResponseCircuitBreaker,
    create_retryable_attempt_receipt,
    load_portfolio_budget_ledger,
    load_query_attempt_receipts,
    portfolio_budget_settled_cost_cny,
    require_retryable_attempt_available,
)
from skillchain.evaluation.portfolio_gcs_evidence import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    expected_multi_items_from_call_v2,
    make_public_scorer_evidence_v2,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_parallel import (  # noqa: E402
    PortfolioAggregateProviderGate,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (  # noqa: E402
    PortfolioS1ExperimentAssistantRunner,
    S1_EXPERIMENT_CONTROL_KIND,
    load_verified_portfolio_s1_experiment_launch,
    load_verified_portfolio_s1_experiment_runtime,
    require_portfolio_s1_experiment_assistant_runner,
    validate_portfolio_s1_experiment_execution_control,
)
from skillchain.runners.assistant import PortfolioAssistantBudgetContext  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.portfolio_runtime import (  # noqa: E402
    PORTFOLIO_SYSTEM_PROMPT,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _model(value):
    return value.model_dump(mode="json")


def _self_model(model_type, payload: dict, field: str):
    return model_type.model_validate(
        {**payload, field: _hash(payload)}, strict=True
    )


def _treatment(config_name: str, bank_sha256: str | None) -> MatrixTreatment:
    stages = {
        "llm_static": ("llm_static", "llm_static"),
        "s1": ("s1", "s1"),
    }
    try:
        router, body = stages[config_name]
    except KeyError as error:
        raise ValueError("S1 population only supports llm_static and s1") from error
    return MatrixTreatment(
        config=config_name,
        bank_sha256=bank_sha256,
        router_stage=router,
        body_stage=body,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-after-shards",
        type=int,
        help="Stop after this many newly completed 25-query shards.",
    )
    return parser


def _canonical_control(root: Path) -> dict:
    path = root / "execution-control.json"
    try:
        content = read_stable_regular_file(path, label="S1 execution control")
        raw = parse_canonical_json(content, label="S1 execution control")
    except (OSError, ArtifactFormatError) as error:
        raise ValueError("S1 execution control cannot be read") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise ValueError("S1 execution control is not canonical")
    supplied = raw.get("control_sha256")
    unsigned = dict(raw)
    unsigned.pop("control_sha256", None)
    if (
        raw.get("schema_version") != 4
        or raw.get("kind") != S1_EXPERIMENT_CONTROL_KIND
        or supplied != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise ValueError("S1 execution control identity drifted")
    return raw


def _launch_facade(launch):
    return SimpleNamespace(plan=SimpleNamespace(**dict(launch.plan)))


def _load_inputs(control: dict, plan: dict):
    source_root = control.get("core_input_binding_launch_root")
    source_sha = control.get("core_input_binding_launch_plan_file_sha256")
    if not isinstance(source_root, str) or not isinstance(source_sha, str):
        raise ValueError("S1 execution lacks its verified Core input binding")
    source = load_portfolio_launch_package(
        source_root, expected_plan_file_sha256=source_sha
    )
    inputs = reconstruct_verified_portfolio_core_inputs(source.plan)
    if (
        inputs.expected_plan_sha256 != plan["portfolio_plan_sha256"]
        or inputs.expected_query_artifact_sha256 != plan["query_artifact_sha256"]
        or inputs.expected_capability_assignments_sha256
        != plan["capability_assignments_sha256"]
        or inputs.expected_output_catalog_sha256 != plan["runtime_catalog_sha256"]
    ):
        raise ValueError("S1 launch differs from the reconstructed Core inputs")
    return inputs


def _build_runner(runtime, inputs, *, call_waiter=None):
    lock = runtime.runtime_lock
    sources = load_verified_portfolio_core_runtime_sources(
        inputs,
        output_dir=runtime.root / "core-runtime-sources",
        expected_receipt_file_sha256=lock[
            "core_runtime_sources_receipt_file_sha256"
        ],
    )
    tool_runtime = build_portfolio_tool_runtime(sources.sources)
    catalog = inputs.runtime_asset_catalog()
    runner = require_portfolio_s1_experiment_assistant_runner(
        PortfolioS1ExperimentAssistantRunner(
            registry=tool_runtime.registry,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            banks=runtime.banks,
            asset_catalog=catalog,
            runtime_lock=lock,
            runtime_lock_file_sha256=runtime.runtime_lock_file_sha256,
            qwen_call_start_waiter=call_waiter,
        )
    )
    return tool_runtime, runner


def _request_context(tool_runtime):
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
    return backbone, budget, _registry_lock(tool_runtime.registry)


def _build_requests(launch, inputs, shard, members, context):
    facade = _launch_facade(launch)
    public_by_id = {item.query_id: item for item in inputs.assistant_queries}
    backbone, budget, registry_lock = context
    treatment = _treatment(
        shard.config,
        launch.plan["bank_sha256s"][shard.config],
    )
    return tuple(
        established._request(
            launch=facade,
            shard=shard,
            member=member,
            query=public_by_id[member.query_id],
            treatment=treatment,
            backbone=backbone,
            budget=budget,
            registry_lock=registry_lock,
        )
        for member in members
    )


def _verify_checkpoint(path, *, facade, member, request):
    try:
        raw = parse_canonical_json(
            read_stable_regular_file(path, label="S1 Assistant checkpoint"),
            label="S1 Assistant checkpoint",
        )
    except (OSError, ArtifactFormatError) as error:
        raise ValueError(f"Assistant checkpoint cannot be read: {path}") from error
    if not isinstance(raw, dict) or raw.get("request") != _model(request):
        raise ValueError(f"Assistant checkpoint request drifted: {path}")
    response, receipt, evidence = established._assistant_row(
        path, require_schema_v2=True, include_scorer_evidence=True
    )
    if evidence is None:
        raise AssertionError("schema-v2 loader lost scorer evidence")
    result = established._assistant_result(facade, request, response)
    established._verify_scorer_sidecar_binding(
        evidence,
        launch=facade,
        member=member,
        request=request,
        result=result,
        receipt=receipt,
    )
    return response, receipt, evidence


def _audit_shard(execution_root: Path, launch, shard, members, requests) -> dict:
    facade = _launch_facade(launch)
    row_bindings = []
    model_calls = 0
    for member, request in zip(members, requests, strict=True):
        path = execution_root / member.assistant_output_relpath
        _response, receipt, evidence = _verify_checkpoint(
            path, facade=facade, member=member, request=request
        )
        model_calls += len(receipt.model_calls)
        row_bindings.append(
            {
                "query_id": member.query_id,
                "instance_sha256": member.instance_sha256,
                "checkpoint_file_sha256": sha256_bytes(path.read_bytes()),
                "checkpoint_row_sha256": json.loads(path.read_bytes())[
                    "row_sha256"
                ],
                "scorer_evidence_sha256": evidence.evidence_sha256,
            }
        )
        if (execution_root / member.final_output_relpath).exists():
            raise ValueError("GCS-only S1 execution produced a forbidden Final row")
    payload = {
        "schema_version": 1,
        "kind": "portfolio-s1-gcs-shard-audit",
        "matrix_run_id": launch.plan["matrix_run_id"],
        "execution_scope": launch.plan["execution_mode"],
        "shard_id": shard.shard_id,
        "shard_sha256": shard.shard_sha256,
        "config": shard.config,
        "accepted_batch_id": shard.accepted_batch_id,
        "assistant_checkpoint_schema_version": 2,
        "query_count": 25,
        "assistant_checkpoint_count": 25,
        "public_scorer_evidence_count": 25,
        "final_judge_checkpoint_count": 0,
        "pairwise_checkpoint_count": 0,
        "model_calls_performed": model_calls,
        "rows": row_bindings,
    }
    audit = {
        **payload,
        "audit_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    path = execution_root / shard.output_relpath / "shard-audit.json"
    content = canonical_json_bytes(audit)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("existing S1 shard audit conflicts with checkpoints")
    else:
        atomic_create_file(path, content)
    return audit


def _recoverable(reason: str, *, shard_id: str, query_id: str) -> int:
    print(
        json.dumps(
            {
                "status": "recoverable_incomplete",
                "reason": reason,
                "shard_id": shard_id,
                "query_id": query_id,
                "resume_policy": "rerun_same_frozen_population",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE


def _execute_member(
    *,
    execution_root,
    launch,
    shard,
    member,
    request,
    private_query,
    runner,
    gate,
    ledger_root,
    breaker,
) -> int | None:
    facade = _launch_facade(launch)
    checkpoint = execution_root / member.assistant_output_relpath
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        _verify_checkpoint(
            checkpoint, facade=facade, member=member, request=request
        )
        return None
    shard_root = execution_root / shard.output_relpath
    prior = load_query_attempt_receipts(
        shard_root,
        query_ordinal=member.query_ordinal,
        query_id=member.query_id,
        instance_sha256=member.instance_sha256,
    )
    try:
        attempt = require_retryable_attempt_available(
            shard_root,
            query_ordinal=member.query_ordinal,
            query_id=member.query_id,
            instance_sha256=member.instance_sha256,
        )
    except PortfolioAttemptLimitError:
        return _recoverable(
            "query_retryable_attempt_limit_reached",
            shard_id=shard.shard_id,
            query_id=member.query_id,
        )
    budget_context = PortfolioAssistantBudgetContext(
        ledger_root=ledger_root,
        shard_id=shard.shard_id,
        instance_sha256=member.instance_sha256,
        attempt_index=attempt,
        repair_of_route_wire_request_sha256=(
            established._repair_of_route_wire_sha256(
                prior, request_sha256=request.request_sha256
            )
        ),
    )
    try:
        with gate.acquire(
            "assistant", label=f"{shard.shard_id}:{member.query_id}:assistant"
        ):
            execution = runner.execute(
                request,
                budget_context=budget_context,
                scorer_query=private_query,
            )
    except PortfolioBudgetExceededError:
        return _recoverable(
            "budget_cap_reached_before_provider_call",
            shard_id=shard.shard_id,
            query_id=member.query_id,
        )
    except PortfolioBudgetDuplicateCallError:
        orphaned = established._record_orphaned_provider_call_if_present(
            ledger_root=ledger_root,
            shard_root=shard_root,
            breaker=breaker,
            matrix_run_id=launch.plan["matrix_run_id"],
            shard_id=shard.shard_id,
            config_name=shard.config,
            instance_sha256=member.instance_sha256,
            query_id=member.query_id,
            query_ordinal=member.query_ordinal,
            request_sha256=request.request_sha256,
            attempt_index=attempt,
            stages=frozenset({"assistant_route", "assistant_action"}),
        )
        if orphaned is None:
            raise
        return _recoverable(
            "orphaned_provider_call_recorded",
            shard_id=shard.shard_id,
            query_id=member.query_id,
        )
    response, receipt = execution.response, execution.receipt
    terminal_attempt = attempt == PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
    retryable_pre_response = response.error_code is not None and (
        response.error_code.startswith("provider_pre_response_")
    )
    retryable_route = established._is_retryable_route_contract_failure(response)
    if retryable_pre_response or retryable_route:
        if retryable_pre_response:
            failure_stage = (
                "assistant_route"
                if response.error_code == "provider_pre_response_route"
                else "assistant_action"
            )
            failure_subtype = (
                "retryable_provider_pre_response_exhausted"
                if terminal_attempt
                else response.error_code
            )
            route_evidence = None
        else:
            failure_stage = "assistant_route"
            failure_subtype = (
                "retryable_route_contract_exhausted"
                if terminal_attempt
                else established._route_contract_attempt_failure_subtype(
                    response.route_failure_subtype, response.route_failure_shape
                )
            )
            route_evidence = None if terminal_attempt else receipt.route_call_evidence
        receipt_record, _ = create_retryable_attempt_receipt(
            shard_root,
            breaker=breaker,
            matrix_run_id=launch.plan["matrix_run_id"],
            shard_id=shard.shard_id,
            config=shard.config,
            instance_sha256=member.instance_sha256,
            query_id=member.query_id,
            query_ordinal=member.query_ordinal,
            request_sha256=request.request_sha256,
            failure_stage=failure_stage,
            failure_subtype=failure_subtype,
            circuit_id=(
                f"assistant:{request.backbone.provider}:{request.backbone.model}"
            ),
            captured_provider_response_count=(
                0 if terminal_attempt else len(receipt.model_calls)
            ),
            captured_input_tokens=(
                0 if terminal_attempt else receipt.aggregate_usage.input_tokens
            ),
            captured_output_tokens=(
                0 if terminal_attempt else receipt.aggregate_usage.output_tokens
            ),
            assistant_response_sha256=receipt.response_sha256,
            assistant_receipt_sha256=receipt.receipt_sha256,
            forfeited_reservation_sha256=(
                response.forfeited_reservation_sha256
                if retryable_pre_response
                else None
            ),
            budget_forfeit_sha256=(
                response.budget_forfeit_sha256 if retryable_pre_response else None
            ),
            exception_type=(
                response.provider_exception_type if retryable_pre_response else None
            ),
            route_call_evidence=route_evidence,
            validated_route_identity=established._validated_route_identity(
                request, response
            ),
        )
        if receipt_record.attempt_index != attempt:
            raise ValueError("S1 retry receipt attempt identity drifted")
        if not terminal_attempt:
            return _recoverable(
                "retryable_assistant_failure_recorded",
                shard_id=shard.shard_id,
                query_id=member.query_id,
            )
    if receipt.model_calls:
        breaker.record_valid_response(
            f"assistant:{request.backbone.provider}:{request.backbone.model}"
        )
    result = established._assistant_result(facade, request, response)
    if (
        execution.scorer_capture_policy_version
        != GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
    ):
        raise ValueError("S1 Assistant omitted the required scorer capture")
    multi_sets = tuple(
        expected
        for call in execution.scorer_calls
        if (expected := expected_multi_items_from_call_v2(call)) is not None
    )
    if len(multi_sets) > 1:
        raise ValueError("S1 Assistant captured multiple multi-item sets")
    evidence = make_public_scorer_evidence_v2(
        matrix_run_id=launch.plan["matrix_run_id"],
        instance_id=member.instance_sha256,
        request_sha256=request.request_sha256,
        query_id=member.query_id,
        config=member.config,
        query_artifact_sha256=launch.plan["query_artifact_sha256"],
        assistant_result_sha256=_hash(_model(result)),
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=execution.scorer_calls,
        expected_multi_items=multi_sets[0] if multi_sets else None,
    )
    established._verify_scorer_sidecar_binding(
        evidence,
        launch=facade,
        member=member,
        request=request,
        result=result,
        receipt=receipt,
    )
    if (
        type(response) is not AssistantBackendResponse
        or type(receipt) is not AssistantExecutionReceipt
        or receipt.response_sha256 != _hash(_model(response))
    ):
        raise ValueError("S1 response differs from its runner-owned receipt")
    row_payload = {
        "schema_version": 2,
        "kind": "portfolio-assistant-checkpoint",
        "instance_sha256": member.instance_sha256,
        "query_ordinal": member.query_ordinal,
        "request": _model(request),
        "response": _model(response),
        "receipt": _model(receipt),
        "public_scorer_evidence": _model(evidence),
    }
    atomic_create_file(
        checkpoint,
        canonical_json_bytes(
            {**row_payload, "row_sha256": _hash(row_payload)}
        ),
    )
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stop_after_shards is not None and args.stop_after_shards < 1:
        print("run-portfolio-s1-population: stop-after-shards must be positive", file=sys.stderr)
        return 2
    try:
        control = _canonical_control(args.execution_root)
        launch = load_verified_portfolio_s1_experiment_launch(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        runtime = load_verified_portfolio_s1_experiment_runtime(
            control["runtime_root"],
            expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
        )
        validate_portfolio_s1_experiment_execution_control(control, launch, runtime)
        inputs = _load_inputs(control, dict(launch.plan))
        gate = None
        if args.execute:
            gate = PortfolioAggregateProviderGate(
                args.execution_root / "provider-gate.sqlite",
                assistant_concurrency=int(control["assistant_concurrency"]),
                final_judge_concurrency=0,
                qwen_requests_per_minute_cap=int(
                    control["qwen_requests_per_minute_cap"]
                ),
                qwen_rate_limit_policy=control["qwen_rate_limit_policy"],
            )
        tool_runtime, runner = _build_runner(
            runtime,
            inputs,
            call_waiter=(None if gate is None else gate.wait_for_qwen_call_start),
        )
        context = _request_context(tool_runtime)
        private_by_id = {item.query_id: item for item in inputs.queries}
        if args.execute:
            ledger_root, session = established._load_or_initialize_budget_ledger(
                execution_root=args.execution_root,
                control=control,
                matrix_run_id=launch.plan["matrix_run_id"],
            )
            if session.state.remaining_cost_cny <= Decimal("0"):
                return _recoverable(
                    "budget_cap_reached_before_provider_call",
                    shard_id=launch.shards[0].shard_id,
                    query_id=launch.shards[0].query_ids[0],
                )
        else:
            ledger_root = args.execution_root / "budget-ledger"
        breaker = ProviderPreResponseCircuitBreaker()
        newly_completed = 0
        verified_complete = 0
        considered = 0
        for shard in launch.shards:
            members = tuple(
                item for item in launch.instances if item.shard_id == shard.shard_id
            )
            requests = _build_requests(launch, inputs, shard, members, context)
            complete_before = all(
                (args.execution_root / item.assistant_output_relpath).is_file()
                for item in members
            )
            if complete_before:
                _audit_shard(
                    args.execution_root, launch, shard, members, requests
                )
                verified_complete += 1
                continue
            if args.stop_after_shards is not None and considered >= args.stop_after_shards:
                break
            considered += 1
            if args.dry_run:
                # Request construction plus runtime/input/selection deep loading is
                # the complete zero-provider preflight. Existing partial rows are
                # byte-verified before reporting readiness.
                facade = _launch_facade(launch)
                for member, request in zip(members, requests, strict=True):
                    path = args.execution_root / member.assistant_output_relpath
                    if path.exists():
                        _verify_checkpoint(
                            path, facade=facade, member=member, request=request
                        )
                continue
            assert gate is not None
            for member, request in zip(members, requests, strict=True):
                result = _execute_member(
                    execution_root=args.execution_root,
                    launch=launch,
                    shard=shard,
                    member=member,
                    request=request,
                    private_query=private_by_id[member.query_id],
                    runner=runner,
                    gate=gate,
                    ledger_root=ledger_root,
                    breaker=breaker,
                )
                if result is not None:
                    return result
            _audit_shard(args.execution_root, launch, shard, members, requests)
            newly_completed += 1
            if (
                args.stop_after_shards is not None
                and newly_completed >= args.stop_after_shards
            ):
                break
        if args.dry_run:
            result = {
                "status": "dry_run_passed",
                "model_calls_performed": 0,
                "execution_scope": launch.plan["execution_mode"],
                "authorized_shard_count": len(launch.shards),
                "considered_incomplete_shard_count": considered,
                "verified_complete_shard_count": verified_complete,
                "stop_after_shards": args.stop_after_shards,
                "assistant_concurrency_cap": control["assistant_concurrency"],
                "effective_process_concurrency": 1,
                "final_judge_enabled": False,
                "pairwise_judge_enabled": False,
            }
        else:
            ledger = load_portfolio_budget_ledger(ledger_root)
            total_complete = sum(
                all(
                    (args.execution_root / member.assistant_output_relpath).is_file()
                    for member in launch.instances
                    if member.shard_id == shard.shard_id
                )
                for shard in launch.shards
            )
            result = {
                "status": (
                    "completed" if total_complete == len(launch.shards) else "stopped"
                ),
                "execution_scope": launch.plan["execution_mode"],
                "newly_completed_shard_count": newly_completed,
                "completed_shard_count": total_complete,
                "authorized_shard_count": len(launch.shards),
                "assistant_checkpoint_count": sum(
                    1
                    for member in launch.instances
                    if (args.execution_root / member.assistant_output_relpath).is_file()
                ),
                "observed_dashscope_cost_cny": format(
                    portfolio_budget_settled_cost_cny(ledger), ".12f"
                ),
                "budget_unresolved_reservation_count": len(
                    ledger.unresolved_reservations
                ),
                "stop_after_shards": args.stop_after_shards,
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"run-portfolio-s1-population: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
