from scripts import rebind_portfolio_core_treatment_runtime as rebind_script
from scripts import run_portfolio_shard as shard_runner
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)


def test_core_runtime_rebind_covers_every_active_lock_contract() -> None:
    contract = rebind_script._active_code_contract()
    expected = {
        **shard_runner._ACTIVE_FINAL_RESULT_CONTRACT,
        **shard_runner._ACTIVE_EVALUATOR_SOURCE_CONTRACT,
        **shard_runner._ACTIVE_BUDGET_CONTRACT,
        "final_judge_parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "final_judge_parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    }

    assert all(contract.get(field) == value for field, value in expected.items())
    assert contract["shared_stage2_route_policy_version"] == (
        shard_runner.SHARED_STAGE2_ROUTE_POLICY_VERSION
    )
    assert contract["portfolio_router_contract_version"] == (
        shard_runner.PORTFOLIO_ROUTER_CONTRACT_VERSION
    )
    assert contract["portfolio_router_contract_sha256"] == (
        shard_runner.PORTFOLIO_ROUTER_CONTRACT_SHA256
    )
    assert contract["portfolio_router_request_max_output_tokens"] == 64
    assert contract["portfolio_router_pricing_reservation_max_output_tokens"] == 512
    assert contract["noskill_execution_policy_version"] == (
        shard_runner.NOSKILL_EXECUTION_POLICY_VERSION
    )
    assert contract["portfolio_failure_policy_version"] == (
        shard_runner.PORTFOLIO_FAILURE_POLICY_VERSION
    )
