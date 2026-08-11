from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_portfolio_matrix as matrix
from scripts import run_portfolio_shard as shard_runner
from scripts.run_portfolio_matrix import (
    _finalize_parallel_stage,
    _ordered_authorized_shards,
    validate_runtime_binding,
)
from skillchain.evaluation.assistant_runs import (
    ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantModelCallReceipt,
    AssistantRouteFailureShape,
    make_assistant_route_call_evidence,
    make_assistant_route_attempt,
)
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE,
    PortfolioBudgetDuplicateCallError,
    PortfolioBudgetExceededError,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
    load_query_attempt_receipts,
    make_portfolio_budget_call_identity,
    portfolio_attempt_provider_call_count,
    reserve_portfolio_provider_call,
    settle_portfolio_provider_call_success,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    PublicScorerEvidenceIntegrityError,
)
from skillchain.llm import LLMUsage
from skillchain.runners.assistant import PortfolioAssistantBudgetContext
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _route_call_evidence(
    *,
    input_tokens: int,
    output_tokens: int,
    payload_status: str,
    response_text: str,
    attempt_index: int = 1,
    wire_request_sha256: str = "d" * 64,
    repair_of_wire_request_sha256: str | None = None,
):
    call = AssistantModelCallReceipt(
        call_index=1,
        provider="qwen",
        endpoint="https://fixture.invalid/v1",
        requested_model="qwen3-vl-flash-2026-01-22",
        response_model="qwen3-vl-flash-2026-01-22",
        provider_request_id="fixture-route-call",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason="stop",
        latency_ms=1,
        response_sha256="c" * 64,
    )
    return make_assistant_route_call_evidence(
        attempt_index=attempt_index,
        request_variant=(
            "fixed_repair" if repair_of_wire_request_sha256 is not None else "initial"
        ),
        repair_of_wire_request_sha256=repair_of_wire_request_sha256,
        wire_request_sha256=wire_request_sha256,
        route_schema_sha256="e" * 64,
        response_text=response_text,
        call_receipt=call,
        failure_reason="invalid_route_json",
        payload_status=payload_status,
    )


def test_parallel_stage_commits_successful_siblings_before_returning_failure(
    monkeypatch, tmp_path
) -> None:
    import scripts.run_portfolio_matrix as matrix

    stage = (
        SimpleNamespace(shard_id="shard-ok"),
        SimpleNamespace(shard_id="shard-failed"),
    )
    launch = SimpleNamespace(
        state=SimpleNamespace(completed_shard_ids=frozenset()),
    )
    refreshed = SimpleNamespace(
        state=SimpleNamespace(completed_shard_ids=frozenset({"shard-ok"})),
    )
    commands = []
    monkeypatch.setattr(
        matrix,
        "_run",
        lambda command: commands.append(command) or 0,
    )
    monkeypatch.setattr(
        matrix,
        "load_portfolio_launch_package",
        lambda *_args, **_kwargs: refreshed,
    )
    result_launch, error = _finalize_parallel_stage(
        args=SimpleNamespace(execute=True, execution_root=tmp_path),
        control={
            "launch_root": "launch",
            "launch_plan_file_sha256": "a" * 64,
        },
        launch=launch,
        stage=stage,
        results={"shard-ok": 0, "shard-failed": 17},
    )

    assert result_launch is refreshed
    assert error == 17
    assert len(commands) == 1
    assert "shard-ok" in commands[0]
    assert "shard-failed" not in commands[0]


def test_runtime_binding_validation_checks_locked_source_hashes(tmp_path) -> None:
    from scripts import run_portfolio_matrix as matrix

    source_files = {
        "runner_file_sha256": matrix.REPOSITORY_ROOT
        / "src"
        / "skillchain"
        / "runners"
        / "assistant.py",
        "shard_runner_file_sha256": matrix.REPOSITORY_ROOT
        / "scripts"
        / "run_portfolio_shard.py",
        "portfolio_execution_file_sha256": matrix.REPOSITORY_ROOT
        / "src"
        / "skillchain"
        / "evaluation"
        / "portfolio_execution.py",
        "llm_adapter_file_sha256": matrix.REPOSITORY_ROOT
        / "src"
        / "skillchain"
        / "llm.py",
    }
    lock = {key: sha256_bytes(path.read_bytes()) for key, path in source_files.items()}
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    control = {
        "runtime_root": str(tmp_path),
        "runtime_lock_file_sha256": sha256_bytes(lock_path.read_bytes()),
    }
    validate_runtime_binding(control)
    lock["runner_file_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    control["runtime_lock_file_sha256"] = sha256_bytes(lock_path.read_bytes())
    with pytest.raises(ValueError, match="runtime binding drifted"):
        validate_runtime_binding(control)


def _launch(count: int = 40):
    shards = tuple(
        SimpleNamespace(shard_id=f"shard-{index:02d}", shard_ordinal=index)
        for index in range(count)
    )
    return SimpleNamespace(plan=SimpleNamespace(shards=shards, shard_count=count))


def test_full_matrix_authorization_preserves_all_launch_shards() -> None:
    launch = _launch()
    control = {
        "execution_scope": "full_matrix",
        "authorized_shard_ids": [item.shard_id for item in launch.plan.shards],
        "external_frozen_shard_count": 0,
    }

    assert _ordered_authorized_shards(control, launch) == launch.plan.shards


def test_full_matrix_rejects_subset_or_reordered_authorization() -> None:
    launch = _launch()
    authorized = [item.shard_id for item in launch.plan.shards]
    authorized[0], authorized[1] = authorized[1], authorized[0]

    with pytest.raises(ValueError, match="exact launch"):
        _ordered_authorized_shards(
            {
                "execution_scope": "full_matrix",
                "authorized_shard_ids": authorized,
                "external_frozen_shard_count": 0,
            },
            launch,
        )


def _full_matrix_budget_control() -> dict[str, object]:
    return {
        "schema_version": 4,
        "execution_scope": "full_matrix",
        "approved_dashscope_budget_cny": "200.000000000000",
        "phase_cumulative_cap_cny": "200.000000000000",
        "incremental_authorized_dashscope_budget_cny": "174.570408300000",
        "prior_dashscope_observed_cost_cny": "25.429591700000",
        "budget_ledger_relpath": "budget-ledger",
        "budget_authority_relpath": "budget-ledger/budget-authority.json",
        "budget_authority_creation_policy": "create_only_on_first_execute",
        **shard_runner._ACTIVE_BUDGET_CONTRACT,
    }


def test_full_matrix_budget_authority_carries_prior(tmp_path) -> None:
    control = _full_matrix_budget_control()

    ledger_root, session = shard_runner._load_or_initialize_budget_ledger(
        execution_root=tmp_path,
        control=control,
        matrix_run_id="matrix-with-prior",
    )

    assert ledger_root == tmp_path / "budget-ledger"
    assert session.state.authority.phase_cap_cny == Decimal("200.000000000000")
    assert session.state.authority.prior_observed_cost_cny == Decimal("25.429591700000")
    assert session.state.remaining_cost_cny == Decimal("174.570408300000")


def test_budget_authority_concurrent_identical_winner_is_loaded(
    monkeypatch,
    tmp_path,
) -> None:
    original_initialize = shard_runner.initialize_portfolio_budget_ledger

    def initialize_after_concurrent_winner(ledger_root, **kwargs):
        original_initialize(ledger_root, **kwargs)
        original_initialize(ledger_root, **kwargs)

    monkeypatch.setattr(
        shard_runner,
        "initialize_portfolio_budget_ledger",
        initialize_after_concurrent_winner,
    )

    ledger_root, session = shard_runner._load_or_initialize_budget_ledger(
        execution_root=tmp_path,
        control=_full_matrix_budget_control(),
        matrix_run_id="matrix-with-prior",
    )

    assert ledger_root == tmp_path / "budget-ledger"
    assert session.state.authority.matrix_run_id == "matrix-with-prior"
    assert session.state.authority.phase_cap_cny == Decimal("200.000000000000")
    assert session.state.authority.prior_observed_cost_cny == Decimal("25.429591700000")
    assert session.state.last_event_index == 0
    assert session.state.reservations == ()


def test_budget_authority_concurrent_drifted_winner_fails_closed(
    monkeypatch,
    tmp_path,
) -> None:
    original_initialize = shard_runner.initialize_portfolio_budget_ledger
    winner_bytes: dict[str, bytes] = {}

    def initialize_after_concurrent_drifted_winner(ledger_root, **kwargs):
        _, authority_path = original_initialize(
            ledger_root,
            matrix_run_id="competing-matrix",
            phase_cap_cny=kwargs["phase_cap_cny"],
            prior_observed_cost_cny=kwargs["prior_observed_cost_cny"],
        )
        winner_bytes["authority"] = authority_path.read_bytes()
        original_initialize(ledger_root, **kwargs)

    monkeypatch.setattr(
        shard_runner,
        "initialize_portfolio_budget_ledger",
        initialize_after_concurrent_drifted_winner,
    )

    with pytest.raises(
        ValueError,
        match="create-only budget authority differs from control",
    ):
        shard_runner._load_or_initialize_budget_ledger(
            execution_root=tmp_path,
            control=_full_matrix_budget_control(),
            matrix_run_id="matrix-with-prior",
        )

    authority_path = tmp_path / "budget-ledger" / "budget-authority.json"
    assert authority_path.read_bytes() == winner_bytes["authority"]
    assert (
        load_portfolio_budget_ledger(tmp_path / "budget-ledger").authority.matrix_run_id
        == "competing-matrix"
    )


def test_partial_repair_still_requires_four_local_shards() -> None:
    launch = _launch(5)
    control = {
        "execution_scope": "partial_shard_repair",
        "authorized_shard_ids": [item.shard_id for item in launch.plan.shards[1:]],
        "external_frozen_shard_count": 1,
    }

    assert _ordered_authorized_shards(control, launch) == launch.plan.shards[1:]


def test_core_canary_requires_one_dev_mini_batch_across_five_configs() -> None:
    query_ids = tuple(f"r3-core-{index:04d}" for index in range(25))
    configs = ("noskill", "llm_static", "s1", "s1s2", "full")
    shards = tuple(
        SimpleNamespace(
            shard_id=f"core-canary-{config}",
            shard_ordinal=index,
            accepted_batch_id="core-r2-batch-001",
            config=config,
            query_ids=query_ids,
        )
        for index, config in enumerate(configs)
    )
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            kind="portfolio-core-split-x5-launch-plan",
            selected_splits=("dev_mini",),
            shards=shards,
            shard_count=len(shards),
        )
    )
    control = {
        "execution_scope": "core_canary",
        "authorized_shard_ids": [item.shard_id for item in shards],
        "external_frozen_shard_count": 0,
    }

    assert _ordered_authorized_shards(control, launch) == shards

    control["authorized_shard_ids"] = control["authorized_shard_ids"][:-1]
    with pytest.raises(ValueError, match="25-query x5"):
        _ordered_authorized_shards(control, launch)


def test_static_opt_authorization_is_exact_800x1_without_final_judge() -> None:
    shards = []
    for shard_index in range(32):
        query_ids = tuple(
            f"core-opt-{shard_index * 25 + offset:04d}" for offset in range(25)
        )
        shards.append(
            SimpleNamespace(
                shard_id=f"static-opt-{shard_index:02d}",
                shard_ordinal=shard_index,
                accepted_batch_id=f"core-opt-batch-{shard_index:03d}",
                config="llm_static",
                query_count=25,
                query_ids=query_ids,
            )
        )
    shards = tuple(shards)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            kind="portfolio-core-static-opt-800x1-launch-plan",
            execution_mode="static_opt_rollout",
            selected_splits=("opt_pool",),
            config_order=("llm_static",),
            query_count=800,
            instance_count=800,
            shard_count=32,
            shards=shards,
        )
    )
    control = {
        "execution_scope": "static_opt_rollout",
        "authorized_shard_ids": [item.shard_id for item in shards],
        "external_frozen_shard_count": 0,
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": "portfolio-grounded-contract-success-v2",
        "gcs_scorer_evidence_policy_version": ("portfolio-gcs-scorer-evidence-v2"),
    }

    assert _ordered_authorized_shards(control, launch) == shards

    launch.plan.selected_splits = ("test_frozen",)
    with pytest.raises(ValueError, match="exact opt_pool"):
        _ordered_authorized_shards(control, launch)


def test_static_opt_parallel_scheduler_spans_generator_batches(
    monkeypatch, tmp_path: Path
) -> None:
    shards = tuple(
        SimpleNamespace(
            shard_id=f"static-{index:02d}",
            accepted_batch_id=f"batch-{index:02d}",
        )
        for index in range(10)
    )
    launch = SimpleNamespace(
        state=SimpleNamespace(
            completed_shard_ids=(),
            status="not_started",
            model_calls_performed=0,
        )
    )
    stages: list[tuple[str, ...]] = []

    def run_stage(**kwargs):
        stage = kwargs["shards"]
        stages.append(tuple(item.shard_id for item in stage))
        return {item.shard_id: 0 for item in stage}

    monkeypatch.setattr(matrix, "_run_parallel_stage", run_stage)
    monkeypatch.setattr(
        matrix,
        "_finalize_parallel_stage",
        lambda **kwargs: (kwargs["launch"], None),
    )
    monkeypatch.setattr(
        matrix, "load_portfolio_launch_package", lambda *_a, **_k: launch
    )
    profile = SimpleNamespace(worker_count=4, profile_sha256="a" * 64)
    args = SimpleNamespace(
        execution_root=tmp_path,
        parallel_profile=tmp_path / "parallel.json",
        execute=False,
        stop_after_shards=None,
    )

    assert (
        matrix._run_parallel_static_opt(
            args=args,
            control={
                "launch_root": "launch",
                "launch_plan_file_sha256": "b" * 64,
            },
            launch=launch,
            shards=shards,
            profile=profile,
        )
        == 0
    )
    assert [len(stage) for stage in stages] == [4, 4, 2]
    assert len({item for stage in stages for item in stage}) == 10


def test_static_opt_runner_loader_never_opens_treatment_chain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bank = SimpleNamespace(bank_sha256="b" * 64)
    lock = {
        **shard_runner._ACTIVE_BUDGET_CONTRACT,
        **shard_runner._ACTIVE_GCS_SOURCE_CONTRACT,
        "schema_version": 1,
    }
    verified = SimpleNamespace(runtime_lock=lock, bank=bank)
    runtime = SimpleNamespace(registry=object())
    catalog = object()
    captured = {}

    monkeypatch.setattr(
        shard_runner,
        "load_verified_portfolio_treatment_runtime",
        lambda *_a, **_k: pytest.fail("Static opt must not load a treatment chain"),
    )
    monkeypatch.setattr(
        shard_runner,
        "load_verified_portfolio_static_opt_runtime",
        lambda *_a, **_k: verified,
    )
    monkeypatch.setattr(
        shard_runner,
        "build_portfolio_tool_runtime",
        lambda *_a, **_k: runtime,
    )
    monkeypatch.setattr(
        shard_runner,
        "load_asset_catalog",
        lambda *_a, **_k: catalog,
    )
    monkeypatch.setattr(
        shard_runner,
        "PortfolioStaticOptAssistantRunner",
        lambda **kwargs: captured.update(kwargs) or "static-runner",
    )
    monkeypatch.setattr(
        shard_runner,
        "require_portfolio_static_opt_assistant_runner",
        lambda value: value,
    )

    def call_start_waiter(_label):
        return 0.0

    built_runtime, banks, built_catalog, runner = shard_runner._build_runner(
        tmp_path,
        "f" * 64,
        inputs=object(),
        static_opt=True,
        qwen_call_start_waiter=call_start_waiter,
    )

    assert built_runtime is runtime
    assert banks == {"llm_static": bank}
    assert built_catalog is catalog
    assert runner == "static-runner"
    assert captured["qwen_call_start_waiter"] is call_start_waiter
    assert captured["bank"] is bank


def test_shard_loader_reconstructs_core_inputs_from_launch_binding(monkeypatch) -> None:
    plan = SimpleNamespace(kind="portfolio-core-split-x5-launch-plan")
    verified = object()
    monkeypatch.setattr(
        shard_runner,
        "reconstruct_verified_portfolio_core_inputs",
        lambda value: verified if value is plan else None,
    )
    monkeypatch.setattr(
        shard_runner,
        "_load_active_inputs",
        lambda: pytest.fail("Core must not fall back to dev_mini inputs"),
    )

    assert shard_runner._inputs_for_launch(plan) is verified


def _install_execute_fixture(monkeypatch, tmp_path, *, runner, config="noskill"):
    shard_id = f"00-fixture-{config}"
    matrix_run_id = "portfolio-budget-fixture"
    shard = SimpleNamespace(
        shard_id=shard_id,
        config=config,
        accepted_batch_id="dev-mini-001-r3",
        output_relpath=f"shards/00-fixture-{config}",
    )
    members = tuple(
        SimpleNamespace(
            shard_id=shard_id,
            query_id=f"dm-{index + 1:03d}",
            query_ordinal=index,
            config=config,
            instance_sha256=f"{index + 1:064x}",
            assistant_output_relpath=(
                f"shards/00-fixture-{config}/assistant/dm-{index + 1:03d}.json"
            ),
            final_output_relpath=(
                f"shards/00-fixture-{config}/final/dm-{index + 1:03d}.json"
            ),
        )
        for index in range(25)
    )
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id=matrix_run_id,
            query_artifact_sha256="9" * 64,
            portfolio_plan_sha256="8" * 64,
            shards=(shard,),
            budget=SimpleNamespace(
                policy_version=shard_runner.PORTFOLIO_BUDGET_POLICY_VERSION,
                policy_sha256=shard_runner.PORTFOLIO_BUDGET_POLICY_SHA256,
                provider_pricing_contract_version=(
                    shard_runner.PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
                ),
                provider_pricing_contract_sha256=(
                    shard_runner.PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
                ),
                over_budget_policy=(
                    "reserve_before_each_provider_call_halt_before_call"
                ),
            ),
        ),
        instances=members,
    )
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_bytes(b"{}")
    control = {
        "schema_version": 4,
        "execution_scope": "partial_shard_repair",
        "launch_root": "fixture-launch",
        "launch_plan_file_sha256": "a" * 64,
        "authorized_shard_ids": [shard_id],
        "runtime_root": "fixture-runtime",
        "runtime_lock_file_sha256": "b" * 64,
        "runtime_lock_sha256": "c" * 64,
        "treatment_chain_manifest_file_sha256": "d" * 64,
        "treatment_chain_sha256": "e" * 64,
        "treatment_record_sha256s": {},
        "rubric_path": str(rubric_path),
        "rubric_file_sha256": sha256_bytes(rubric_path.read_bytes()),
        "phase_cumulative_cap_cny": "10.000000000000",
        "incremental_authorized_dashscope_budget_cny": "8.750000000000",
        "prior_dashscope_observed_cost_cny": "1.250000000000",
        "approved_dashscope_budget_cny": "10.000000000000",
        "portfolio_budget_policy_version": (
            shard_runner.PORTFOLIO_BUDGET_POLICY_VERSION
        ),
        "portfolio_budget_policy_sha256": (shard_runner.PORTFOLIO_BUDGET_POLICY_SHA256),
        "provider_pricing_contract_version": (
            shard_runner.PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
        ),
        "provider_pricing_contract_sha256": (
            shard_runner.PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ),
        "budget_ledger_relpath": "budget-ledger",
        "budget_authority_relpath": "budget-ledger/budget-authority.json",
        "budget_authority_creation_policy": "create_only_on_first_execute",
    }
    verified_runtime = SimpleNamespace(
        manifest_file_sha256="d" * 64,
        runtime_lock={},
        chain=SimpleNamespace(
            manifest=SimpleNamespace(chain_sha256="e" * 64, records=())
        ),
    )
    public = tuple(SimpleNamespace(query_id=item.query_id) for item in members)
    private = tuple(SimpleNamespace(query_id=item.query_id) for item in members)
    inputs = SimpleNamespace(
        assistant_queries=public,
        queries=private,
        runtime_for=lambda _runtime_id: object(),
    )
    requests = {
        item.query_id: SimpleNamespace(
            query=SimpleNamespace(query_id=item.query_id),
            config=config,
            treatment=SimpleNamespace(
                bank_sha256=None if config == "noskill" else "f" * 64
            ),
            backbone=SimpleNamespace(provider="qwen", model="qwen3-vl-flash"),
            request_sha256=f"{index + 101:064x}",
        )
        for index, item in enumerate(members)
    }

    monkeypatch.setattr(shard_runner, "_load_control", lambda _root: control)
    monkeypatch.setattr(
        shard_runner,
        "load_portfolio_launch_package",
        lambda *_args, **_kwargs: launch,
    )
    monkeypatch.setattr(
        shard_runner,
        "load_verified_portfolio_treatment_runtime",
        lambda *_args, **_kwargs: verified_runtime,
    )
    monkeypatch.setattr(
        shard_runner, "_require_active_final_result_runtime", lambda _lock: None
    )
    monkeypatch.setattr(
        shard_runner,
        "_build_runner",
        lambda *_args, **_kwargs: (
            SimpleNamespace(registry=object()),
            (
                {}
                if config == "noskill"
                else {config: SimpleNamespace(bank_sha256="f" * 64)}
            ),
            object(),
            runner,
        ),
    )
    monkeypatch.setattr(
        shard_runner,
        "RubricSnapshot",
        SimpleNamespace(model_validate_json=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(shard_runner, "_load_active_inputs", lambda: inputs)
    monkeypatch.setattr(shard_runner, "_registry_lock", lambda _registry: object())
    monkeypatch.setattr(
        shard_runner,
        "_request",
        lambda **kwargs: requests[kwargs["member"].query_id],
    )
    monkeypatch.setattr(
        shard_runner,
        "_model",
        lambda value: {"fixture_type": type(value).__name__},
    )
    monkeypatch.setattr(
        shard_runner, "make_active_portfolio_evaluator_isolation_lock", object
    )
    (tmp_path / "blinding-key.bin").write_bytes(b"k" * 32)
    return control, launch, shard, members, requests


def _install_static_execute_fixture(monkeypatch, tmp_path, *, runner):
    active_model_projection = shard_runner._model
    control, launch, shard, members, requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=runner,
        config="llm_static",
    )
    control.update(
        {
            "execution_scope": "static_opt_rollout",
            "assistant_checkpoint_schema_version": 2,
            "gcs_policy_version": shard_runner.GCS_V2_POLICY_VERSION,
            "gcs_scorer_evidence_policy_version": (
                shard_runner.GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
            ),
        }
    )
    verified_runtime = SimpleNamespace(
        runtime_lock={
            **shard_runner._ACTIVE_BUDGET_CONTRACT,
            **shard_runner._ACTIVE_GCS_SOURCE_CONTRACT,
        }
    )
    monkeypatch.setattr(
        shard_runner,
        "load_verified_portfolio_static_opt_runtime",
        lambda *_args, **_kwargs: verified_runtime,
    )
    monkeypatch.setattr(
        shard_runner,
        "validate_portfolio_static_opt_execution_control",
        lambda *_args, **_kwargs: None,
    )

    def fixture_model(value):
        if hasattr(value, "model_dump"):
            return active_model_projection(value)
        return {"fixture_type": type(value).__name__}

    monkeypatch.setattr(shard_runner, "_model", fixture_model)
    return control, launch, shard, members, requests


def _provider_pre_response_execution(request, *, attempt_index: int):
    usage = LLMUsage(input_tokens=0, output_tokens=0)
    forfeited_reservation_sha256 = f"{attempt_index + 10:064x}"
    budget_forfeit_sha256 = f"{attempt_index + 20:064x}"
    response = AssistantBackendResponse(
        schema_version=2,
        request_sha256=request.request_sha256,
        backbone_provider=request.backbone.provider,
        backbone_model=request.backbone.model,
        backbone_endpoint="https://fixture.invalid/v1",
        backbone_identity_sha256="1" * 64,
        registry_sha256="2" * 64,
        registry_runtime_sha256="3" * 64,
        budget_sha256="4" * 64,
        response_text="",
        usage=usage,
        turn_count=1,
        latency_ms=1,
        error_code="provider_pre_response_route",
        provider_exception_type="RateLimitError",
        forfeited_reservation_sha256=forfeited_reservation_sha256,
        budget_forfeit_sha256=budget_forfeit_sha256,
    )
    route_attempt = make_assistant_route_attempt(
        status="failed",
        error_code="runtime_error",
    )
    receipt_payload = {
        "schema_version": 1,
        "policy_version": ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
        "request_sha256": request.request_sha256,
        "asset_catalog_sha256": "5" * 64,
        "query_asset_id": f"asset-{request.query.query_id}",
        "query_asset_sha256": "6" * 64,
        "model_calls": [],
        "tool_trace": [],
        "route_attempt": route_attempt.model_dump(mode="json"),
        "aggregate_usage": usage.model_dump(mode="json"),
        "runner_latency_ms": 1,
        "outcome": "runtime_error",
        "response_sha256": sha256_bytes(
            canonical_json_bytes(response.model_dump(mode="json"))
        ),
    }
    receipt = AssistantExecutionReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        },
        strict=True,
    )
    return SimpleNamespace(
        response=response,
        receipt=receipt,
        scorer_calls=(),
        scorer_capture_policy_version=(
            shard_runner.GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
        ),
    )


def _route_contract_execution(
    request,
    *,
    attempt_index: int,
    repair_of_wire_request_sha256: str | None,
):
    response_text = '{"selected_capability":"x","extra":true}'
    route_call_evidence = _route_call_evidence(
        input_tokens=31,
        output_tokens=9,
        payload_status="unexpected_keys",
        response_text=response_text,
        attempt_index=attempt_index,
        wire_request_sha256=("d" * 64 if attempt_index == 1 else "f" * 64),
        repair_of_wire_request_sha256=repair_of_wire_request_sha256,
    )
    call = route_call_evidence.call_receipt
    usage = LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens)
    failure_shape = AssistantRouteFailureShape(
        finish_reason="stop",
        response_text_bytes=len(response_text.encode("utf-8")),
        tool_call_count=0,
        payload_status="unexpected_keys",
        normalized_response_sha256="7" * 64,
    )
    response = AssistantBackendResponse(
        schema_version=2,
        request_sha256=request.request_sha256,
        backbone_provider=request.backbone.provider,
        backbone_model=request.backbone.model,
        backbone_endpoint="https://fixture.invalid/v1",
        backbone_identity_sha256="1" * 64,
        registry_sha256="2" * 64,
        registry_runtime_sha256="3" * 64,
        budget_sha256="4" * 64,
        response_text="",
        usage=usage,
        turn_count=1,
        latency_ms=1,
        error_code="route_contract_error",
        route_failure_subtype="invalid_route_json",
        route_failure_shape=failure_shape,
    )
    route_attempt = make_assistant_route_attempt(
        status="failed",
        error_code="runtime_error",
    )
    receipt_payload = {
        "schema_version": 1,
        "policy_version": ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
        "request_sha256": request.request_sha256,
        "asset_catalog_sha256": "5" * 64,
        "query_asset_id": f"asset-{request.query.query_id}",
        "query_asset_sha256": "6" * 64,
        "model_calls": [call.model_dump(mode="json")],
        "tool_trace": [],
        "route_attempt": route_attempt.model_dump(mode="json"),
        "route_call_evidence": route_call_evidence.model_dump(mode="json"),
        "aggregate_usage": usage.model_dump(mode="json"),
        "runner_latency_ms": 1,
        "outcome": "runtime_error",
        "response_sha256": sha256_bytes(
            canonical_json_bytes(response.model_dump(mode="json"))
        ),
    }
    receipt = AssistantExecutionReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        },
        strict=True,
    )
    return SimpleNamespace(
        response=response,
        receipt=receipt,
        scorer_calls=(),
        scorer_capture_policy_version=(
            shard_runner.GCS_SCORER_EVIDENCE_V2_POLICY_VERSION
        ),
    )


@pytest.mark.parametrize(
    ("payload_status", "expected"),
    [
        ("invalid_json", True),
        ("non_object", True),
        ("unexpected_keys", True),
        ("schema_invalid", True),
        ("out_of_enum", False),
        ("empty", False),
        ("not_examined", False),
    ],
)
def test_route_contract_retry_accepts_only_correctable_json_shape_failures(
    payload_status,
    expected,
) -> None:
    response = SimpleNamespace(
        error_code="route_contract_error",
        route_failure_subtype="invalid_route_json",
        route_failure_shape=SimpleNamespace(payload_status=payload_status),
    )

    assert shard_runner._is_retryable_route_contract_failure(response) is expected


@pytest.mark.parametrize(
    ("error_code", "failure_subtype", "payload_status", "expected", "receipt_subtype"),
    [
        ("route_length", "length", "not_examined", True, "route_contract_length"),
        (
            "route_contract_error",
            "response_empty_text",
            "empty",
            True,
            "route_contract_empty",
        ),
        (
            "route_contract_error",
            "out_of_enum",
            "out_of_enum",
            False,
            None,
        ),
        (
            "route_contract_error",
            "response_tool_calls",
            "not_examined",
            False,
            None,
        ),
    ],
)
def test_route_contract_retry_handles_only_fixed_repair_failure_classes(
    error_code,
    failure_subtype,
    payload_status,
    expected,
    receipt_subtype,
) -> None:
    shape = SimpleNamespace(payload_status=payload_status)
    response = SimpleNamespace(
        error_code=error_code,
        route_failure_subtype=failure_subtype,
        route_failure_shape=shape,
    )

    assert shard_runner._is_retryable_route_contract_failure(response) is expected
    if expected:
        assert (
            shard_runner._route_contract_attempt_failure_subtype(
                failure_subtype,
                shape,
            )
            == receipt_subtype
        )
    else:
        with pytest.raises(ValueError, match="not eligible"):
            shard_runner._route_contract_attempt_failure_subtype(
                failure_subtype,
                shape,
            )


def test_route_contract_failure_gets_one_receipted_retry_then_fixed_zero(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    response = SimpleNamespace(
        error_code="route_contract_error",
        route_failure_subtype="invalid_route_json",
        route_failure_shape=SimpleNamespace(payload_status="unexpected_keys"),
        selected_capability=None,
        skill_slug=None,
        route_trace_sha256=None,
        visible_cards=(),
    )
    usage = SimpleNamespace(input_tokens=31, output_tokens=9)
    route_call_evidence = _route_call_evidence(
        input_tokens=31,
        output_tokens=9,
        payload_status="unexpected_keys",
        response_text='{"selected_capability":"x","extra":true}',
    )
    receipt = SimpleNamespace(
        model_calls=(object(),),
        aggregate_usage=usage,
        response_sha256="a" * 64,
        receipt_sha256="b" * 64,
        route_call_evidence=route_call_evidence,
    )

    contexts = []

    class ContractFailingRunner:
        def execute(self, *_args, **kwargs):
            contexts.append(kwargs["budget_context"])
            return SimpleNamespace(response=response, receipt=receipt)

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=ContractFailingRunner(),
        config="s1",
    )

    first_exit = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert first_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    first = members[0]
    shard_root = tmp_path / shard.output_relpath
    attempt_receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert len(attempt_receipts) == 1
    assert attempt_receipts[0].failure_stage == "assistant_route"
    assert attempt_receipts[0].failure_subtype == "route_contract_unexpected_keys"
    assert attempt_receipts[0].captured_provider_response_count == 1
    assert attempt_receipts[0].captured_input_tokens == 31
    assert attempt_receipts[0].captured_output_tokens == 9
    assert attempt_receipts[0].assistant_response_sha256 == "a" * 64
    assert attempt_receipts[0].assistant_receipt_sha256 == "b" * 64
    assert contexts[0].repair_of_route_wire_request_sha256 is None
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()
    assert '"reason": "retryable_route_contract_failure_recorded"' in (
        capsys.readouterr().out
    )

    second_exit = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert second_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert (tmp_path / first.assistant_output_relpath).exists()
    fixed = json.loads((tmp_path / first.final_output_relpath).read_bytes())
    assert fixed["assistant_error_code"] == "route_contract_error"
    assert fixed["score_disposition"] == "fixed_zero"
    assert contexts[1].repair_of_route_wire_request_sha256 == (
        route_call_evidence.wire_request_sha256
    )
    assert (
        len(
            load_query_attempt_receipts(
                shard_root,
                query_ordinal=first.query_ordinal,
                query_id=first.query_id,
                instance_sha256=first.instance_sha256,
            )
        )
        == 1
    )


def test_static_gcs_second_provider_failure_is_receipted_then_terminalized(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    calls: list[tuple[str, int]] = []

    class RetryExhaustingRunner:
        def execute(self, request, **kwargs):
            context = kwargs["budget_context"]
            calls.append((request.query.query_id, context.attempt_index))
            if request.query.query_id == "dm-001":
                return _provider_pre_response_execution(
                    request,
                    attempt_index=context.attempt_index,
                )
            raise PortfolioBudgetExceededError("stop after proving continuation")

    _control, launch, shard, members, requests = _install_static_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=RetryExhaustingRunner(),
    )
    monkeypatch.setattr(
        shard_runner,
        "run_visual_final_judge",
        lambda *_args, **_kwargs: pytest.fail("Static opt must not call Final Judge"),
    )

    first_exit = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    first = members[0]
    shard_root = tmp_path / shard.output_relpath
    assistant_path = tmp_path / first.assistant_output_relpath
    assert first_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    first_receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert len(first_receipts) == 1
    assert first_receipts[0].policy_version == "portfolio-shard-attempt-v4"
    assert first_receipts[0].attempt_index == 1
    assert not assistant_path.exists()
    assert not list(shard_root.glob("final/*.json"))
    assert '"reason": "retryable_provider_failure_recorded"' in (
        capsys.readouterr().out
    )

    second_exit = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert second_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert calls == [("dm-001", 1), ("dm-001", 2), ("dm-002", 1)]
    receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert [item.attempt_index for item in receipts] == [1, 2]
    assert all(item.policy_version == "portfolio-shard-attempt-v4" for item in receipts)
    assert receipts[-1].circuit_open is True
    assistant_rows = list(shard_root.glob("assistant/*.json"))
    assert assistant_rows == [assistant_path]
    response, receipt, sidecar = shard_runner._assistant_row(
        assistant_path,
        require_schema_v2=True,
        include_scorer_evidence=True,
    )
    assert response.error_code == "provider_pre_response_route"
    assert receipt.outcome == "runtime_error"
    assert sidecar is not None
    assert sidecar.calls == ()
    terminal_result = shard_runner._assistant_result(
        launch,
        requests[first.query_id],
        response,
    )
    assert terminal_result.error_code == "runtime_error"
    assert sidecar.assistant_result_sha256 == shard_runner._hash(
        shard_runner._model(terminal_result)
    )
    assert not list(shard_root.glob("final/*.json"))
    assert not list(tmp_path.rglob("*pairwise*"))
    output = capsys.readouterr().out
    assert '"completed": 1' in output
    assert '"query_id": "dm-002"' in output


def test_legacy_provider_retry_exhaustion_remains_recoverable_without_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    class LegacyRetryExhaustingRunner:
        def execute(self, request, **kwargs):
            return _provider_pre_response_execution(
                request,
                attempt_index=kwargs["budget_context"].attempt_index,
            )

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=LegacyRetryExhaustingRunner(),
    )
    argv = [
        "--execution-root",
        str(tmp_path),
        "--shard-id",
        shard.shard_id,
        "--execute",
    ]

    assert shard_runner.main(argv) == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert shard_runner.main(argv) == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE

    first = members[0]
    receipts = load_query_attempt_receipts(
        tmp_path / shard.output_relpath,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert [item.attempt_index for item in receipts] == [1, 2]
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()


def test_static_gcs_second_route_failure_retains_live_repair_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    contexts: list[tuple[str, int, str | None]] = []

    class RouteRetryExhaustingRunner:
        def execute(self, request, **kwargs):
            context = kwargs["budget_context"]
            contexts.append(
                (
                    request.query.query_id,
                    context.attempt_index,
                    context.repair_of_route_wire_request_sha256,
                )
            )
            if request.query.query_id == "dm-001":
                return _route_contract_execution(
                    request,
                    attempt_index=context.attempt_index,
                    repair_of_wire_request_sha256=(
                        context.repair_of_route_wire_request_sha256
                    ),
                )
            raise PortfolioBudgetExceededError("stop after proving continuation")

    _control, _launch, shard, members, _requests = _install_static_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=RouteRetryExhaustingRunner(),
    )
    argv = [
        "--execution-root",
        str(tmp_path),
        "--shard-id",
        shard.shard_id,
        "--execute",
    ]

    assert shard_runner.main(argv) == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert shard_runner.main(argv) == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE

    first = members[0]
    initial_wire_sha256 = "d" * 64
    assert contexts == [
        ("dm-001", 1, None),
        ("dm-001", 2, initial_wire_sha256),
        ("dm-002", 1, None),
    ]
    receipts = load_query_attempt_receipts(
        tmp_path / shard.output_relpath,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert [item.attempt_index for item in receipts] == [1, 2]
    assert receipts[0].failure_subtype == "route_contract_unexpected_keys"
    assert receipts[0].route_call_evidence is not None
    assert receipts[1].failure_subtype == "retryable_route_contract_exhausted"
    assert receipts[1].route_call_evidence is None
    assert receipts[1].captured_provider_response_count == 0
    assert receipts[1].captured_input_tokens == 0
    assert receipts[1].captured_output_tokens == 0
    assistant_path = tmp_path / first.assistant_output_relpath
    response, receipt, sidecar = shard_runner._assistant_row(
        assistant_path,
        require_schema_v2=True,
        include_scorer_evidence=True,
    )
    assert response.error_code == "route_contract_error"
    assert receipt.route_call_evidence is not None
    assert receipt.route_call_evidence.attempt_index == 2
    assert receipt.route_call_evidence.request_variant == "fixed_repair"
    assert len(receipt.model_calls) == 1
    assert (
        sum(portfolio_attempt_provider_call_count(item) for item in receipts)
        + len(receipt.model_calls)
        == 2
    )
    assert (
        receipt.route_call_evidence.repair_of_wire_request_sha256 == initial_wire_sha256
    )
    assert sidecar is not None
    assert sidecar.calls == ()
    assert not list((tmp_path / shard.output_relpath).glob("final/*.json"))


def test_static_gcs_retry_exhaustion_scorer_contract_failure_remains_fatal(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class InvalidSecondCaptureRunner:
        def execute(self, request, **kwargs):
            execution = _provider_pre_response_execution(
                request,
                attempt_index=kwargs["budget_context"].attempt_index,
            )
            if kwargs["budget_context"].attempt_index == 2:
                execution.scorer_capture_policy_version = "drifted-policy"
            return execution

    _control, _launch, shard, members, _requests = _install_static_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=InvalidSecondCaptureRunner(),
    )
    argv = [
        "--execution-root",
        str(tmp_path),
        "--shard-id",
        shard.shard_id,
        "--execute",
    ]

    assert shard_runner.main(argv) == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert shard_runner.main(argv) == 2

    first = members[0]
    receipts = load_query_attempt_receipts(
        tmp_path / shard.output_relpath,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert [item.attempt_index for item in receipts] == [1, 2]
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()
    assert "omitted required GCS v2 capture" in capsys.readouterr().err


def test_full_waits_for_s1s2_shared_route_owner_without_a_provider_call(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    calls = {"prepare": 0, "execute": 0}

    class MustNotRunBeforeOwner:
        def prepare_shared_stage2_route(self, *_args, **_kwargs):
            calls["prepare"] += 1
            raise AssertionError("Full must not create a shared route")

        def execute(self, *_args, **_kwargs):
            calls["execute"] += 1
            raise AssertionError("Full must wait for the shared route owner")

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=MustNotRunBeforeOwner(),
        config="full",
    )

    exit_code = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert calls == {"prepare": 0, "execute": 0}
    assert not list((tmp_path / "shared-routes").rglob("*.json"))
    assert not list((tmp_path / shard.output_relpath).rglob("*.json"))
    assert '"reason": "shared_route_owner_incomplete"' in capsys.readouterr().out


def test_s1s2_retryable_shared_route_failure_is_receipted_not_published(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    artifact = SimpleNamespace(
        status="terminal_route_error",
        failure_subtype="invalid_route_json",
        failure_shape=SimpleNamespace(payload_status="invalid_json"),
        route_call=SimpleNamespace(input_tokens=27, output_tokens=6),
        route_call_evidence=_route_call_evidence(
            input_tokens=27,
            output_tokens=6,
            payload_status="invalid_json",
            response_text="{not-json",
        ),
    )
    calls = {"prepare": 0, "execute": 0}

    class SharedRouteContractFailingRunner:
        def prepare_shared_stage2_route(self, *_args, **_kwargs):
            calls["prepare"] += 1
            return artifact

        def execute(self, *_args, **_kwargs):
            calls["execute"] += 1
            raise AssertionError("retryable shared route must stop before action")

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=SharedRouteContractFailingRunner(),
        config="s1s2",
    )

    exit_code = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert calls == {"prepare": 1, "execute": 0}
    assert not list((tmp_path / "shared-routes").rglob("*.json"))
    first = members[0]
    receipts = load_query_attempt_receipts(
        tmp_path / shard.output_relpath,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert len(receipts) == 1
    assert receipts[0].failure_stage == "shared_route"
    assert receipts[0].failure_subtype == "route_contract_invalid_json"
    assert receipts[0].captured_provider_response_count == 1
    assert receipts[0].captured_input_tokens == 27
    assert receipts[0].captured_output_tokens == 6
    assert '"reason": "retryable_route_contract_failure_recorded"' in (
        capsys.readouterr().out
    )


def test_first_execute_creates_budget_authority_and_passes_assistant_context(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    observed = {}

    class RejectingRunner:
        def execute(self, request, **kwargs):
            observed["request"] = request
            observed["kwargs"] = kwargs
            observed["ledger"] = load_portfolio_budget_ledger(
                kwargs["budget_context"].ledger_root
            )
            raise PortfolioBudgetExceededError("fixture hard cap")

    control, launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=RejectingRunner(),
    )

    exit_code = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    context = observed["kwargs"]["budget_context"]
    assert type(context) is PortfolioAssistantBudgetContext
    assert context.ledger_root == tmp_path / "budget-ledger"
    assert context.shard_id == shard.shard_id
    assert context.instance_sha256 == members[0].instance_sha256
    assert context.attempt_index == 1
    ledger = observed["ledger"]
    assert ledger.authority.matrix_run_id == launch.plan.matrix_run_id
    assert ledger.authority.phase_cap_cny == Decimal(
        control["phase_cumulative_cap_cny"]
    )
    assert ledger.authority.prior_observed_cost_cny == Decimal(
        control["prior_dashscope_observed_cost_cny"]
    )
    assert ledger.reservations == ()
    assert not (tmp_path / members[0].assistant_output_relpath).exists()
    assert not (tmp_path / members[0].final_output_relpath).exists()
    output = capsys.readouterr().out
    assert '"reason": "budget_cap_reached_before_provider_call"' in output
    assert '"resume_policy": "new_authorized_execution_phase_required"' in output
    assert '"score_disposition": "not_scored_no_fixed_zero"' in output


def test_scorer_integrity_failure_publishes_no_assistant_checkpoint(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class IntegrityFailingRunner:
        def execute(self, *_args, **_kwargs):
            raise PublicScorerEvidenceIntegrityError("fixture scorer integrity failure")

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=IntegrityFailingRunner(),
    )

    assert (
        shard_runner.main(
            [
                "--execution-root",
                str(tmp_path),
                "--shard-id",
                shard.shard_id,
                "--execute",
            ]
        )
        == 2
    )

    assert not (tmp_path / members[0].assistant_output_relpath).exists()
    assert not (tmp_path / members[0].final_output_relpath).exists()
    assert not (tmp_path / shard.output_relpath / "shard-summary.json").exists()
    assert "fixture scorer integrity failure" in capsys.readouterr().err


def test_judge_receives_budget_context_and_overage_writes_no_fixed_zero(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    observed = {}
    response = SimpleNamespace(error_code=None, visible_cards=())
    receipt = SimpleNamespace(model_calls=())

    class SuccessfulAssistantRunner:
        def execute(self, request, **kwargs):
            observed["assistant_context"] = kwargs["budget_context"]
            return SimpleNamespace(response=response, receipt=receipt)

    _control, launch, shard, members, requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=SuccessfulAssistantRunner(),
    )
    monkeypatch.setattr(shard_runner, "_assistant_result", lambda *_args: object())
    monkeypatch.setattr(
        shard_runner,
        "_build_final_packet_fail_closed",
        lambda *_args, **_kwargs: (object(), None),
    )

    def reject_judge_before_provider(*_args, **kwargs):
        observed["judge_context"] = kwargs["budget_context"]
        raise PortfolioBudgetExceededError("fixture Judge hard cap")

    monkeypatch.setattr(
        shard_runner, "run_visual_final_judge", reject_judge_before_provider
    )

    exit_code = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assistant_context = observed["assistant_context"]
    judge_context = observed["judge_context"]
    assert judge_context.ledger_root == assistant_context.ledger_root
    assert judge_context.matrix_run_id == launch.plan.matrix_run_id
    assert judge_context.shard_id == shard.shard_id
    assert judge_context.config == "noskill"
    assert judge_context.query_id == members[0].query_id
    assert judge_context.instance_sha256 == members[0].instance_sha256
    assert judge_context.request_sha256 == requests[members[0].query_id].request_sha256
    assert judge_context.attempt_index == 1
    assert (tmp_path / members[0].assistant_output_relpath).exists()
    final_path = tmp_path / members[0].final_output_relpath
    assert not final_path.exists()
    assert not list(tmp_path.rglob("*fixed-zero*"))
    output = capsys.readouterr().out
    assert '"reason": "budget_cap_reached_before_provider_call"' in output
    assert '"resume_policy": "new_authorized_execution_phase_required"' in output
    assert '"score_disposition": "not_scored_no_fixed_zero"' in output


def test_packet_isolation_failure_does_not_mutate_or_publish_assistant(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    response = SimpleNamespace(error_code=None, visible_cards=())
    receipt = SimpleNamespace(model_calls=())

    class SuccessfulAssistantRunner:
        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(response=response, receipt=receipt)

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=SuccessfulAssistantRunner(),
    )
    monkeypatch.setattr(shard_runner, "_assistant_result", lambda *_args: object())
    monkeypatch.setattr(
        shard_runner,
        "_build_final_packet_fail_closed",
        lambda *_args, **_kwargs: (None, "hidden_evaluation_identity"),
    )
    monkeypatch.setattr(
        shard_runner,
        "run_visual_final_judge",
        lambda *_args, **_kwargs: pytest.fail("Judge must not run"),
    )

    exit_code = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )

    assert exit_code == 2
    first = members[0]
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()
    assert response.error_code is None
    assert "immutable Assistant receipt was issued" in capsys.readouterr().err


def test_restart_forfeits_unsettled_orphan_before_writing_attempt_receipt(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class ProviderMustNotRun:
        def execute(self, *_args, **_kwargs):
            raise PortfolioBudgetDuplicateCallError(
                "provider-call identity already has an immutable reservation"
            )

    control, launch, shard, members, requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=ProviderMustNotRun(),
    )
    monkeypatch.setattr(
        shard_runner,
        "run_visual_final_judge",
        lambda *_a, **_k: pytest.fail("orphan must stop before final Judge"),
    )
    ledger_root = tmp_path / "budget-ledger"
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=launch.plan.matrix_run_id,
        phase_cap_cny=Decimal(control["phase_cumulative_cap_cny"]),
        prior_observed_cost_cny=Decimal(control["prior_dashscope_observed_cost_cny"]),
    )
    first = members[0]
    identity = make_portfolio_budget_call_identity(
        matrix_run_id=launch.plan.matrix_run_id,
        shard_id=shard.shard_id,
        config=shard.config,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
        request_sha256=requests[first.query_id].request_sha256,
        wire_request_sha256="6" * 64,
        stage="assistant_action",
        attempt_index=1,
        call_index=1,
    )
    reservation, _ = reserve_portfolio_provider_call(
        ledger_root,
        identity=identity,
    )
    before = load_portfolio_budget_ledger(ledger_root)

    exit_code = shard_runner.main(
        [
            "--execution-root",
            str(tmp_path),
            "--shard-id",
            shard.shard_id,
            "--execute",
        ]
    )

    assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    after = load_portfolio_budget_ledger(ledger_root)
    assert after.settlements == ()
    assert len(after.forfeits) == 1
    assert after.unresolved_reservations == ()
    assert after.accountable_cost_cny == before.accountable_cost_cny
    forfeit = after.forfeits[0]
    assert forfeit.reservation_sha256 == reservation.reservation_sha256
    receipt = load_query_attempt_receipts(
        tmp_path / shard.output_relpath,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )[0]
    assert receipt.failure_subtype == "orphaned_provider_call"
    assert receipt.captured_provider_response_count == 0
    assert receipt.forfeited_reservation_sha256 == reservation.reservation_sha256
    assert receipt.budget_forfeit_sha256 == forfeit.forfeit_sha256
    assert '"reason": "orphaned_provider_call_recorded"' in capsys.readouterr().out


def test_restart_records_settled_orphan_without_replaying_provider_call(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    provider_calls = {"assistant": 0, "judge": 0}

    class ProviderMustNotRun:
        def execute(self, *_args, **_kwargs):
            raise PortfolioBudgetDuplicateCallError(
                "provider-call identity already has an immutable reservation"
            )

    control, launch, shard, members, requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=ProviderMustNotRun(),
    )

    def judge_must_not_run(*_args, **_kwargs):
        provider_calls["judge"] += 1
        raise AssertionError("Assistant orphan must stop before final Judge")

    monkeypatch.setattr(shard_runner, "run_visual_final_judge", judge_must_not_run)
    ledger_root = tmp_path / "budget-ledger"
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=launch.plan.matrix_run_id,
        phase_cap_cny=Decimal(control["phase_cumulative_cap_cny"]),
        prior_observed_cost_cny=Decimal(control["prior_dashscope_observed_cost_cny"]),
    )
    first = members[0]
    request_sha256 = requests[first.query_id].request_sha256

    def seed_settled_orphan(
        *,
        attempt_index: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        identity = make_portfolio_budget_call_identity(
            matrix_run_id=launch.plan.matrix_run_id,
            shard_id=shard.shard_id,
            config=shard.config,
            query_id=first.query_id,
            instance_sha256=first.instance_sha256,
            request_sha256=request_sha256,
            wire_request_sha256=f"{attempt_index + 5:x}" * 64,
            stage="assistant_action",
            attempt_index=attempt_index,
            call_index=1,
        )
        reservation, _ = reserve_portfolio_provider_call(
            ledger_root,
            identity=identity,
        )
        settle_portfolio_provider_call_success(
            ledger_root,
            reservation_sha256=reservation.reservation_sha256,
            actual_input_tokens=input_tokens,
            actual_output_tokens=output_tokens,
            provider_request_id=f"req-orphan-attempt-{attempt_index}",
            response_sha256=f"{attempt_index + 7:x}" * 64,
        )

    seed_settled_orphan(attempt_index=1, input_tokens=12, output_tokens=3)

    first_exit = shard_runner.main(
        [
            "--execution-root",
            str(tmp_path),
            "--shard-id",
            shard.shard_id,
            "--execute",
        ]
    )

    assert first_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    shard_root = tmp_path / shard.output_relpath
    receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert len(receipts) == 1
    first_receipt = receipts[0]
    assert first_receipt.attempt_index == 1
    assert first_receipt.failure_stage == "assistant_action"
    assert first_receipt.failure_subtype == "orphaned_provider_call"
    assert first_receipt.retryable is True
    assert first_receipt.score_disposition == "not_scored_no_fixed_zero"
    assert first_receipt.captured_provider_response_count == 1
    assert first_receipt.captured_input_tokens == 12
    assert first_receipt.captured_output_tokens == 3
    assert first_receipt.exception_type == "PortfolioBudgetOrphanedCallError"
    assert first_receipt.circuit_open is False
    assert provider_calls == {"assistant": 0, "judge": 0}
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()
    first_output = capsys.readouterr().out
    assert '"status": "recoverable_incomplete"' in first_output
    assert '"reason": "orphaned_provider_call_recorded"' in first_output
    assert '"score_disposition": "not_scored_no_fixed_zero"' in first_output

    seed_settled_orphan(attempt_index=2, input_tokens=14, output_tokens=4)

    second_exit = shard_runner.main(
        [
            "--execution-root",
            str(tmp_path),
            "--shard-id",
            shard.shard_id,
            "--execute",
        ]
    )

    assert second_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    receipts = load_query_attempt_receipts(
        shard_root,
        query_ordinal=first.query_ordinal,
        query_id=first.query_id,
        instance_sha256=first.instance_sha256,
    )
    assert [receipt.attempt_index for receipt in receipts] == [1, 2]
    assert all(
        receipt.failure_subtype == "orphaned_provider_call"
        and receipt.retryable is True
        and receipt.score_disposition == "not_scored_no_fixed_zero"
        for receipt in receipts
    )
    assert receipts[1].captured_provider_response_count == 1
    assert receipts[1].captured_input_tokens == 14
    assert receipts[1].captured_output_tokens == 4
    assert receipts[1].circuit_open is True
    assert provider_calls == {"assistant": 0, "judge": 0}
    assert not (tmp_path / first.assistant_output_relpath).exists()
    assert not (tmp_path / first.final_output_relpath).exists()
    assert not list(tmp_path.rglob("*fixed-zero*"))
    ledger = load_portfolio_budget_ledger(ledger_root)
    assert len(ledger.reservations) == len(ledger.settlements) == 2
    assert ledger.unresolved_reservations == ()
    second_output = capsys.readouterr().out
    assert '"status": "recoverable_incomplete"' in second_output
    assert '"reason": "orphaned_provider_call_recorded"' in second_output
    assert '"score_disposition": "not_scored_no_fixed_zero"' in second_output


@pytest.mark.parametrize("terminal_status", ["provider_error", "timeout"])
def test_judge_retry_terminal_pre_response_failure_remains_recoverable(
    monkeypatch,
    tmp_path,
    capsys,
    terminal_status,
) -> None:
    usage = SimpleNamespace(input_tokens=37, output_tokens=11)
    response = SimpleNamespace(
        error_code=None,
        visible_cards=(),
        selected_capability=None,
        skill_slug=None,
        route_trace_sha256=None,
    )
    assistant_receipt = SimpleNamespace(model_calls=())

    class SuccessfulAssistant:
        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(response=response, receipt=assistant_receipt)

    _control, _launch, shard, members, _requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=SuccessfulAssistant(),
    )
    monkeypatch.setattr(
        shard_runner,
        "_assistant_row",
        lambda _path: (response, assistant_receipt),
    )
    monkeypatch.setattr(shard_runner, "_assistant_result", lambda *_args: object())
    monkeypatch.setattr(
        shard_runner,
        "_build_final_packet_fail_closed",
        lambda *_args, **_kwargs: (object(), None),
    )
    monkeypatch.setattr(
        shard_runner,
        "_require_final_checkpoint_contract",
        lambda *_args, **_kwargs: None,
    )
    judge_calls = 0

    def retry_then_fail_before_terminal_response(*_args, **_kwargs):
        nonlocal judge_calls
        judge_calls += 1
        return SimpleNamespace(
            outcome=SimpleNamespace(status=terminal_status),
            request_id=None,
            captured_response_count=1,
            aggregate_usage=usage,
            result_sha256=f"{judge_calls:x}" * 64,
            provider="kimi",
            model="kimi-k2.6",
            forfeited_reservation_sha256="e" * 64,
            budget_forfeit_sha256="f" * 64,
        )

    monkeypatch.setattr(
        shard_runner,
        "run_visual_final_judge",
        retry_then_fail_before_terminal_response,
    )

    first = members[0]
    shard_root = tmp_path / shard.output_relpath
    for expected_attempt in (1, 2):
        exit_code = shard_runner.main(
            [
                "--execution-root",
                str(tmp_path),
                "--shard-id",
                shard.shard_id,
                "--execute",
            ]
        )
        assert exit_code == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
        attempts = load_query_attempt_receipts(
            shard_root,
            query_ordinal=first.query_ordinal,
            query_id=first.query_id,
            instance_sha256=first.instance_sha256,
        )
        assert len(attempts) == expected_attempt
        receipt = attempts[-1]
        assert receipt.attempt_index == expected_attempt
        assert receipt.failure_stage == "final_judge"
        assert receipt.failure_subtype == f"provider_pre_response_{terminal_status}"
        assert receipt.captured_provider_response_count == 1
        assert receipt.captured_input_tokens == usage.input_tokens
        assert receipt.captured_output_tokens == usage.output_tokens
        assert receipt.judge_result_sha256 == f"{expected_attempt:x}" * 64
        assert receipt.score_disposition == "not_scored_no_fixed_zero"
        assert not (tmp_path / first.final_output_relpath).exists()

    third_exit = shard_runner.main(
        ["--execution-root", str(tmp_path), "--shard-id", shard.shard_id, "--execute"]
    )
    assert third_exit == PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    assert judge_calls == 2
    assert not (tmp_path / first.final_output_relpath).exists()
    assert '"reason": "query_retryable_attempt_limit_reached"' in (
        capsys.readouterr().out
    )


def test_two_shards_resume_missing_member_and_rerun_without_duplicate_calls(
    monkeypatch,
    tmp_path,
) -> None:
    executor_calls: list[str] = []
    provider_calls: list[str] = []
    terminal_response = SimpleNamespace(
        error_code="fixture_terminal_error",
        route_failure_subtype=None,
        route_failure_shape=None,
        selected_capability=None,
        skill_slug=None,
        route_trace_sha256=None,
        visible_cards=(),
    )
    terminal_receipt = SimpleNamespace(model_calls=(object(),))

    class CountingZeroProviderRunner:
        def execute(self, request, **_kwargs):
            query_id = request.query.query_id
            executor_calls.append(query_id)
            # This counter represents the one provider response captured by
            # the fake execution.  No provider adapter is installed or called.
            provider_calls.append(query_id)
            return SimpleNamespace(
                response=terminal_response,
                receipt=terminal_receipt,
            )

    control, launch, first_shard, first_members, requests = _install_execute_fixture(
        monkeypatch,
        tmp_path,
        runner=CountingZeroProviderRunner(),
    )
    second_shard_id = "01-fixture-noskill"
    second_shard = SimpleNamespace(
        shard_id=second_shard_id,
        config="noskill",
        accepted_batch_id="dev-mini-002-r3",
        output_relpath=f"shards/{second_shard_id}",
    )
    second_members = tuple(
        SimpleNamespace(
            shard_id=second_shard_id,
            query_id=f"dm-{index + 26:03d}",
            query_ordinal=index + 25,
            instance_sha256=f"{index + 26:064x}",
            assistant_output_relpath=(
                f"shards/{second_shard_id}/assistant/dm-{index + 26:03d}.json"
            ),
            final_output_relpath=(
                f"shards/{second_shard_id}/final/dm-{index + 26:03d}.json"
            ),
        )
        for index in range(25)
    )
    for index, member in enumerate(second_members):
        requests[member.query_id] = SimpleNamespace(
            query=SimpleNamespace(query_id=member.query_id),
            config="noskill",
            treatment=SimpleNamespace(bank_sha256=None),
            backbone=SimpleNamespace(provider="qwen", model="qwen3-vl-flash"),
            request_sha256=f"{index + 1001:064x}",
        )
    all_shards = (first_shard, second_shard)
    all_members = first_members + second_members
    launch.plan.shards = all_shards
    launch.instances = all_members
    control["authorized_shard_ids"] = [item.shard_id for item in all_shards]
    inputs = SimpleNamespace(
        assistant_queries=tuple(
            SimpleNamespace(query_id=item.query_id) for item in all_members
        ),
        queries=tuple(SimpleNamespace(query_id=item.query_id) for item in all_members),
        runtime_for=lambda _runtime_id: object(),
    )
    monkeypatch.setattr(shard_runner, "_load_active_inputs", lambda: inputs)
    monkeypatch.setattr(
        shard_runner,
        "_request",
        lambda **kwargs: requests[kwargs["member"].query_id],
    )

    def load_fixture_assistant_checkpoint(path):
        payload = json.loads(path.read_bytes())
        supplied = payload.pop("row_sha256")
        assert supplied == shard_runner._hash(payload)
        return terminal_response, terminal_receipt

    monkeypatch.setattr(
        shard_runner,
        "_assistant_row",
        load_fixture_assistant_checkpoint,
    )

    def final_judge_must_not_run(*_args, **_kwargs):
        raise AssertionError("terminal fixture response must not call Final Judge")

    monkeypatch.setattr(
        shard_runner,
        "run_visual_final_judge",
        final_judge_must_not_run,
    )

    def execute(shard) -> None:
        assert (
            shard_runner.main(
                [
                    "--execution-root",
                    str(tmp_path),
                    "--shard-id",
                    shard.shard_id,
                    "--execute",
                ]
            )
            == 0
        )

    def member_hashes(members) -> dict[str, str]:
        relative_paths = {
            relative_path
            for member in members
            for relative_path in (
                member.assistant_output_relpath,
                member.final_output_relpath,
            )
        }
        return {
            relative_path: sha256_bytes((tmp_path / relative_path).read_bytes())
            for relative_path in sorted(relative_paths)
        }

    # First write commits one complete shard; the absent second shard models
    # an interruption between frozen shard boundaries.
    execute(first_shard)
    first_hashes = member_hashes(first_members)
    first_call_snapshot = (tuple(executor_calls), tuple(provider_calls))
    assert first_call_snapshot == (
        tuple(item.query_id for item in first_members),
        tuple(item.query_id for item in first_members),
    )
    assert not (tmp_path / second_shard.output_relpath).exists()

    # Resume may revisit the committed shard, but it must only execute the
    # previously absent shard and must preserve all completed member bytes.
    execute(first_shard)
    assert (tuple(executor_calls), tuple(provider_calls)) == first_call_snapshot
    assert member_hashes(first_members) == first_hashes
    execute(second_shard)
    complete_hashes = member_hashes(all_members)
    expected_query_ids = sorted(item.query_id for item in all_members)
    assert sorted(executor_calls) == expected_query_ids
    assert sorted(provider_calls) == expected_query_ids
    assert member_hashes(first_members) == first_hashes

    expected_member_paths = {
        relative_path
        for shard in all_shards
        for member in all_members
        if member.shard_id == shard.shard_id
        for relative_path in (
            member.assistant_output_relpath,
            member.final_output_relpath,
        )
    }
    actual_member_paths = {
        path.relative_to(tmp_path).as_posix()
        for directory in ("assistant", "final")
        for path in (tmp_path / "shards").glob(f"*/{directory}/*.json")
    }
    assert actual_member_paths == expected_member_paths
    for shard in all_shards:
        assert shard.output_relpath == f"shards/{shard.shard_id}"
        assert (tmp_path / shard.output_relpath).relative_to(
            tmp_path
        ).as_posix() == shard.output_relpath
        for member in (item for item in all_members if item.shard_id == shard.shard_id):
            assert member.assistant_output_relpath == (
                f"{shard.output_relpath}/assistant/{member.query_id}.json"
            )
            assert member.final_output_relpath == (
                f"{shard.output_relpath}/final/{member.query_id}.json"
            )

    # Simulate an interruption after an Assistant checkpoint was committed but
    # before its final member and shard summary were durably visible.  Resume
    # reconstructs the deterministic terminal final without another executor
    # or provider call, and produces the exact original bytes.
    missing_final = tmp_path / second_members[0].final_output_relpath
    missing_summary = tmp_path / second_shard.output_relpath / "shard-summary.json"
    missing_final.unlink()
    missing_summary.unlink()
    assert not missing_final.exists()
    assert not missing_summary.exists()
    call_snapshot = (tuple(executor_calls), tuple(provider_calls))

    execute(second_shard)

    assert (tuple(executor_calls), tuple(provider_calls)) == call_snapshot
    assert missing_final.exists()
    assert missing_summary.exists()
    assert member_hashes(all_members) == complete_hashes

    # A further whole-set rerun is a pure verification/no-op for member data.
    execute(first_shard)
    execute(second_shard)
    assert (tuple(executor_calls), tuple(provider_calls)) == call_snapshot
    assert member_hashes(all_members) == complete_hashes
    assert sorted(executor_calls) == expected_query_ids
    assert sorted(provider_calls) == expected_query_ids
