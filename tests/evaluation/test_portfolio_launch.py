from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts import run_portfolio_shard as shard_runner
from skillchain.evaluation.assistant_runs import MAIN_CONFIG_ORDER
from skillchain.evaluation.final_runtime import FINAL_JUDGE_RESULT_SCHEMA_VERSION
from skillchain.evaluation.packets import HiddenEvaluationIdentityError
from skillchain.evaluation.portfolio_launch import (
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_BUDGET_POLICY_SHA256_V1,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION,
    PortfolioBudgetPolicy,
    PortfolioCallPlan,
    PortfolioLaunchState,
    PortfolioRatePolicy,
    _balanced_config_order,
    _runtime_execution_contract_success_detail,
)
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_BUDGET_POLICY_VERSION,
)


def test_balanced_launch_schedule_rotates_all_five_configs() -> None:
    observed = tuple(_balanced_config_order(index) for index in range(5))

    assert observed[0] == MAIN_CONFIG_ORDER
    assert all(set(order) == set(MAIN_CONFIG_ORDER) for order in observed)
    assert tuple(order[0] for order in observed) == MAIN_CONFIG_ORDER


def test_runtime_execution_preflight_detail_tracks_active_judge_schema() -> None:
    detail = _runtime_execution_contract_success_detail()

    assert f"Judge schema {FINAL_JUDGE_RESULT_SCHEMA_VERSION}" in detail
    assert "Judge schema 6" not in detail


def test_call_plan_reserves_one_empty_response_retry_per_judge_row() -> None:
    calls = PortfolioCallPlan(
        assistant_call_floor=1600,
        assistant_call_ceiling=4800,
        final_judge_empty_response_retry_call_ceiling=1000,
        total_call_floor=2600,
        total_call_ceiling=6800,
        note="actual Judge accounting uses persisted attempts",
    )

    assert calls.final_judge_call_count == 1000
    assert calls.final_judge_empty_response_retry_call_ceiling == 1000
    assert calls.total_call_ceiling == 6800


def test_provider_retry_attempt_is_separate_from_judge_semantic_retry() -> None:
    rate_policy = PortfolioRatePolicy(
        checkpoint_policy="after_every_25_query_shard"
    )
    calls = PortfolioCallPlan(
        assistant_call_floor=1600,
        assistant_call_ceiling=4800,
        final_judge_empty_response_retry_call_ceiling=1000,
        total_call_floor=2600,
        total_call_ceiling=6800,
        note="actual Judge accounting uses persisted attempts",
    )

    assert rate_policy.provider_retry_attempts == 1
    assert calls.final_judge_call_count == 1000
    assert calls.final_judge_empty_response_retry_call_ceiling == 1000


def test_call_plan_rejects_old_ceiling_when_retry_reserve_is_active() -> None:
    with pytest.raises(ValidationError, match="bounds are inconsistent"):
        PortfolioCallPlan(
            assistant_call_floor=1600,
            assistant_call_ceiling=4800,
            final_judge_empty_response_retry_call_ceiling=1000,
            total_call_floor=2600,
            total_call_ceiling=5800,
            note="invalid retry reserve",
        )


def test_call_plan_still_loads_historical_pre_retry_shape() -> None:
    calls = PortfolioCallPlan(
        assistant_call_floor=1600,
        assistant_call_ceiling=4800,
        total_call_floor=2600,
        total_call_ceiling=5800,
        note="historical plan",
    )

    assert calls.final_judge_empty_response_retry_call_ceiling is None
    assert "final_judge_empty_response_retry_call_ceiling" not in calls.model_dump(
        mode="json"
    )


def _budget_policy(*, context: dict | None = None, **overrides) -> PortfolioBudgetPolicy:
    payload = {
        "currency": "CNY",
        "autonomous_dashscope_budget_cny": 10.0,
        "operator_approved_dashscope_budget_cny": 182.0,
        "planning_ceiling_cny": 136.0,
        "qwen_input_cny_per_million_tokens": 0.15,
        "qwen_output_cny_per_million_tokens": 1.5,
        "kimi_input_cny_per_million_tokens": 6.5,
        "kimi_output_cny_per_million_tokens": 27.0,
        "pricing_basis": "test",
        "aifast_cost_status": "gateway_price_not_locked",
        "checkpoint_policy": "after_every_25_query_shard",
        "over_budget_policy": "halt_before_next_shard",
        **overrides,
    }
    return PortfolioBudgetPolicy.model_validate(payload, strict=True, context=context)


def test_launch_budget_keeps_historical_v1_readable() -> None:
    historical = _budget_policy()

    assert historical.policy_version is None
    assert historical.over_budget_policy == "halt_before_next_shard"


def test_launch_budget_binds_per_provider_reserve_contract() -> None:
    active = _budget_policy(
        policy_version=PORTFOLIO_BUDGET_POLICY_VERSION,
        policy_sha256=PORTFOLIO_BUDGET_POLICY_SHA256,
        provider_pricing_contract_version=(
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
        ),
        provider_pricing_contract_sha256=(
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
        ),
        over_budget_policy="reserve_before_each_provider_call_halt_before_call",
    )

    assert active.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256
    with pytest.raises(ValidationError, match="budget contract drifted"):
        _budget_policy(
            policy_version=PORTFOLIO_BUDGET_POLICY_VERSION,
            policy_sha256="0" * 64,
            provider_pricing_contract_version=(
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_VERSION
            ),
            provider_pricing_contract_sha256=(
                PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
            ),
            over_budget_policy=(
                "reserve_before_each_provider_call_halt_before_call"
            ),
        )


def test_launch_budget_reads_v23_mixed_contract_only_in_legacy_context() -> None:
    historical_v23 = {
        "policy_version": "portfolio-call-hard-cap-v1",
        "policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256_V1,
        "provider_pricing_contract_version": "portfolio-provider-pricing-contract-v2",
        "provider_pricing_contract_sha256": (
            PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256_V2
        ),
        "over_budget_policy": "reserve_before_each_provider_call_halt_before_call",
    }

    with pytest.raises(ValidationError, match="budget contract drifted"):
        _budget_policy(**historical_v23)

    loaded = _budget_policy(
        context={"allow_legacy_budget_contract": True},
        **historical_v23,
    )
    assert loaded.policy_sha256 == PORTFOLIO_BUDGET_POLICY_SHA256_V1

    with pytest.raises(ValidationError, match="budget contract drifted"):
        _budget_policy(
            context={"allow_legacy_budget_contract": True},
            **{
                **historical_v23,
                "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION,
                "policy_sha256": PORTFOLIO_BUDGET_POLICY_SHA256,
            },
        )


def test_launch_state_accepts_only_empty_not_started_state() -> None:
    state = PortfolioLaunchState(
        matrix_run_id="portfolio-dev-mini-200x5-v1",
        launch_plan_sha256="a" * 64,
        status="not_started",
        completed_shard_ids=(),
        failed_shard_ids=(),
        model_calls_performed=0,
    )

    assert state.status == "not_started"


def test_launch_state_rejects_progress_under_not_started_status() -> None:
    with pytest.raises(ValidationError, match="must be empty"):
        PortfolioLaunchState(
            matrix_run_id="portfolio-dev-mini-200x5-v1",
            launch_plan_sha256="a" * 64,
            status="not_started",
            completed_shard_ids=("00-batch-noskill",),
            failed_shard_ids=(),
            model_calls_performed=1,
        )


def test_launch_state_rejects_overlapping_shard_sets() -> None:
    with pytest.raises(ValidationError, match="unique and disjoint"):
        PortfolioLaunchState(
            matrix_run_id="portfolio-dev-mini-200x5-v1",
            launch_plan_sha256="a" * 64,
            status="failed",
            completed_shard_ids=("00-batch-noskill",),
            failed_shard_ids=("00-batch-noskill",),
            model_calls_performed=1,
        )


def test_fresh_execution_root_carries_prior_approved_cost(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(shard_runner, "_observed_cost", lambda _root: 0.25)

    assert shard_runner._cumulative_observed_cost(
        {"prior_dashscope_observed_cost_cny": 2.5348485},
        tmp_path,
    ) == pytest.approx(2.7848485)


def test_shard_maps_hidden_public_identity_to_terminal_fixed_zero(
    monkeypatch,
) -> None:
    def reject_hidden_identity(*_args, **_kwargs):
        raise HiddenEvaluationIdentityError("private value reached public output")

    monkeypatch.setattr(
        shard_runner,
        "build_final_evaluation_packet",
        reject_hidden_identity,
    )

    packet, error_code = shard_runner._build_final_packet_fail_closed(
        object(),
        object(),
        asset_catalog=object(),
        rubric=object(),
        blinding_key=b"x" * 32,
    )

    assert packet is None
    assert error_code == "hidden_evaluation_identity"
    assert shard_runner._final_fixed_zero("dm-001", error_code) == {
        "schema_version": 1,
        "kind": "portfolio-final-fixed-zero",
        "query_id": "dm-001",
        "assistant_error_code": "hidden_evaluation_identity",
        "failure_class": "terminal_task_failure",
        "retryable": False,
        "score_disposition": "fixed_zero",
        "j_project": 0.0,
    }


def test_shard_resume_accepts_only_canonical_hidden_identity_fixed_zero(
    tmp_path,
) -> None:
    final_path = tmp_path / "dm-001.json"
    payload = shard_runner._final_fixed_zero(
        "dm-001",
        "hidden_evaluation_identity",
    )
    final_path.write_bytes(
        shard_runner.canonical_json_bytes(
            {**payload, "result_sha256": shard_runner._hash(payload)}
        )
    )

    assert (
        shard_runner._existing_fixed_zero_error(
            final_path,
            query_id="dm-001",
        )
        == "hidden_evaluation_identity"
    )
