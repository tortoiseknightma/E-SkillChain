from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_portfolio_batch
from scripts import finalize_portfolio_shard
from scripts import prepare_portfolio_execution
from scripts import run_portfolio_shard
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.final_runtime import (
    CARD_REQUIREMENT_GUARD_POLICY_SHA256,
    CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    FINAL_JUDGE_RETRY_POLICY_SHA256,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
)
from skillchain.tools.serialization import sha256_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPOSITORY_ROOT / "src" / "skillchain" / "evaluation"


def _active_lock() -> dict[str, object]:
    return {
        **prepare_portfolio_execution._ACTIVE_FINAL_RESULT_LOCK,
        **prepare_portfolio_execution._ACTIVE_BUDGET_CONTRACT,
        "final_judge_parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "final_judge_parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
        "final_judge_result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        "final_judge_cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "final_judge_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
        "final_judge_retry_policy_version": FINAL_JUDGE_RETRY_POLICY_VERSION,
        "final_judge_retry_policy_sha256": FINAL_JUDGE_RETRY_POLICY_SHA256,
        "card_requirement_guard_policy_version": (
            CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (
            CARD_REQUIREMENT_GUARD_POLICY_SHA256
        ),
        "config_file_sha256": sha256_bytes(
            (REPOSITORY_ROOT / "src" / "skillchain" / "config.py").read_bytes()
        ),
        "packets_file_sha256": sha256_bytes(
            (EVALUATION_ROOT / "packets.py").read_bytes()
        ),
        "evaluator_isolation_file_sha256": sha256_bytes(
            (EVALUATION_ROOT / "evaluator_isolation.py").read_bytes()
        ),
        "portfolio_tool_runtime_file_sha256": sha256_bytes(
            (
                REPOSITORY_ROOT
                / "src"
                / "skillchain"
                / "tools"
                / "portfolio_runtime.py"
            ).read_bytes()
        ),
        "tool_registry_file_sha256": sha256_bytes(
            (
                REPOSITORY_ROOT
                / "src"
                / "skillchain"
                / "tools"
                / "registry.py"
            ).read_bytes()
        ),
    }


def test_active_runtime_guard_identity_is_shared_by_execution_surfaces() -> None:
    lock = _active_lock()

    run_portfolio_shard._require_active_final_result_runtime(lock)
    prepare_portfolio_execution._require_active_final_result_runtime(
        lock,
        label="test runtime",
    )
    audit_portfolio_batch._require_active_final_result_runtime(
        lock,
        label="test runtime",
    )
    assert finalize_portfolio_shard._locked_final_result_contract(lock) == (
        FINAL_JUDGE_RESULT_SCHEMA_VERSION,
        FINAL_JUDGE_CACHE_NAMESPACE,
    )


@pytest.mark.parametrize(
    "missing_field",
    [
        "final_judge_result_schema_version",
        "final_judge_cache_namespace",
        "final_judge_max_attempts",
        "final_judge_retry_policy_version",
        "final_judge_retry_policy_sha256",
        "final_judge_thinking_budget",
        "final_judge_max_billable_input_tokens",
        "final_judge_max_billable_output_tokens",
        "final_judge_provider_input_token_reserve",
        "final_judge_provider_output_token_reserve",
        "card_requirement_guard_policy_version",
        "card_requirement_guard_policy_sha256",
        "portfolio_budget_policy_version",
        "portfolio_budget_policy_sha256",
        "provider_pricing_contract_version",
        "provider_pricing_contract_sha256",
        "config_file_sha256",
        "packets_file_sha256",
        "evaluator_isolation_file_sha256",
        "portfolio_tool_runtime_file_sha256",
        "tool_registry_file_sha256",
    ],
)
def test_new_runtime_rejects_every_partial_card_guard_binding(
    missing_field: str,
) -> None:
    lock = _active_lock()
    lock.pop(missing_field)

    with pytest.raises(ValueError, match="does not bind"):
        run_portfolio_shard._require_active_final_result_runtime(lock)
    with pytest.raises(ValueError, match="does not bind"):
        prepare_portfolio_execution._require_active_final_result_runtime(
            lock,
            label="test runtime",
        )
    with pytest.raises(ValueError, match="does not bind"):
        audit_portfolio_batch._require_active_final_result_runtime(
            lock,
            label="test runtime",
        )
    with pytest.raises(ValueError, match="partial or inactive"):
        finalize_portfolio_shard._locked_final_result_contract(lock)


def test_shard_finalizer_retains_read_only_schema_1_to_3_derivation() -> None:
    assert finalize_portfolio_shard._locked_final_result_contract({}) == (
        1,
        "final-evaluator-v2",
    )
    assert finalize_portfolio_shard._locked_final_result_contract(
        {
            "final_judge_parser_policy_version": (
                FINAL_JUDGE_PARSER_POLICY_VERSION_V2
            ),
            "final_judge_parser_policy_sha256": (
                FINAL_JUDGE_PARSER_POLICY_SHA256_V2
            ),
        }
    ) == (2, "final-evaluator-v3")
    assert finalize_portfolio_shard._locked_final_result_contract(
        {
            "final_judge_parser_policy_version": (
                FINAL_JUDGE_PARSER_POLICY_VERSION_V3
            ),
            "final_judge_parser_policy_sha256": (
                FINAL_JUDGE_PARSER_POLICY_SHA256_V3
            ),
        }
    ) == (3, "final-evaluator-v4")


def test_shard_finalizer_retains_read_only_schema_4_contract() -> None:
    lock = {
        "final_judge_parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
        "final_judge_parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
        "final_judge_result_schema_version": 4,
        "final_judge_cache_namespace": "final-evaluator-v5",
        "card_requirement_guard_policy_version": (
            CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (
            CARD_REQUIREMENT_GUARD_POLICY_SHA256
        ),
    }

    assert finalize_portfolio_shard._locked_final_result_contract(lock) == (
        4,
        "final-evaluator-v5",
    )


def test_shard_finalizer_retains_read_only_schema_7_v3_contract() -> None:
    lock = _active_lock()
    lock.update(
        {
            "final_judge_parser_policy_version": (
                FINAL_JUDGE_PARSER_POLICY_VERSION_V3
            ),
            "final_judge_parser_policy_sha256": (
                FINAL_JUDGE_PARSER_POLICY_SHA256_V3
            ),
            "final_judge_result_schema_version": 7,
            "final_judge_cache_namespace": "final-evaluator-v8",
        }
    )

    assert finalize_portfolio_shard._locked_final_result_contract(lock) == (
        7,
        "final-evaluator-v8",
    )


def test_shard_finalizer_rejects_unknown_historical_parser_pair() -> None:
    with pytest.raises(ValueError, match="unknown final-Judge parser contract"):
        finalize_portfolio_shard._locked_final_result_contract(
            {
                "final_judge_parser_policy_version": (
                    FINAL_JUDGE_PARSER_POLICY_VERSION_V3
                ),
                "final_judge_parser_policy_sha256": "0" * 64,
            }
        )
