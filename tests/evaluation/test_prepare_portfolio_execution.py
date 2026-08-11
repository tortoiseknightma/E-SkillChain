from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_portfolio_execution as preparation
from scripts.prepare_portfolio_execution import (
    _external_frozen_noskill_binding,
    _validate_source_cost_accounting,
    build_parser,
)
from skillchain.evaluation.portfolio_launch import (
    load_portfolio_launch_package,
)
from skillchain.runners.assistant import (
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXECUTION_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-dev-mini-200x5-execution-v6"
)
TARGET_LAUNCH_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-dev-mini-200x5-launch-v12"
)
SOURCE_RUNTIME_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-public-data-runtime-v9"
)
SOURCE_SHARD_ID = "00-dev-mini-001-r3-00-noskill"
SOURCE_AUDIT_SHA256 = "1bae36e1bab96e13645e1e5f78b296a15002d0fb91c050ef3c4a9f96a2cd3047"
TARGET_LAUNCH_PLAN_FILE_SHA256 = (
    "9c1f64b30e7a677d384be4f81ee4c19bede5eb4101baa5bf92c93a6147ac3b86"
)
HARD_LEDGER_SOURCE_EXECUTION_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-dev-mini-200x5-execution-v15"
)
HARD_LEDGER_TARGET_LAUNCH_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-dev-mini-200x5-launch-v23"
)
HARD_LEDGER_SOURCE_RUNTIME_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-public-data-runtime-v21"
)
HARD_LEDGER_SOURCE_AUDIT_SHA256 = (
    "c08f38a84ea0f1bd1591fe66266d16b29cf8d0d3d4844a5dfd70fd504fd8d032"
)
HARD_LEDGER_TARGET_LAUNCH_PLAN_FILE_SHA256 = (
    "dea2785e8e0f3853198b1ed531f334a8efa33266b17ad2502ef251559b86c695"
)


def test_final_judge_comparison_identity_includes_billing_budget() -> None:
    final = SimpleNamespace(
        provider="kimi",
        model="kimi-k2.6",
        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=2048,
        max_attempts=2,
        retry_policy_version="retry-v3",
        retry_policy_sha256="a" * 64,
        thinking_budget=6144,
        max_billable_input_tokens=32768,
        max_billable_output_tokens=8192,
    )

    assert preparation._final_judge_comparison_identity(final) == {
        "provider": "kimi",
        "model": "kimi-k2.6",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 2048,
        "max_attempts": 2,
        "retry_policy_version": "retry-v3",
        "retry_policy_sha256": "a" * 64,
        "thinking_budget": 6144,
        "max_billable_input_tokens": 32768,
        "max_billable_output_tokens": 8192,
    }


def test_schema10_final_judge_identity_includes_json_transport() -> None:
    final = SimpleNamespace(
        schema_version=10,
        provider="gemini",
        model="gemini-3.6-flash",
        endpoint="https://example.aifast.net/v1",
        max_tokens=2048,
        max_attempts=2,
        retry_policy_version=preparation.FINAL_JUDGE_RETRY_POLICY_VERSION,
        retry_policy_sha256=preparation.FINAL_JUDGE_RETRY_POLICY_SHA256,
        thinking_budget=preparation.FINAL_JUDGE_THINKING_BUDGET,
        max_billable_input_tokens=preparation.FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
        max_billable_output_tokens=(preparation.FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS),
        transport_policy_version=preparation.FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
        transport_policy_sha256=preparation.FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
        requested_response_format="json_object",
    )

    identity = preparation._final_judge_comparison_identity(final)

    assert identity["transport_policy_version"] == (
        preparation.FINAL_JUDGE_TRANSPORT_POLICY_VERSION
    )
    assert identity["transport_policy_sha256"] == (
        preparation.FINAL_JUDGE_TRANSPORT_POLICY_SHA256
    )
    assert identity["requested_response_format"] == "json_object"


def _target_runtime_lock() -> dict:
    source = json.loads((SOURCE_RUNTIME_ROOT / "runtime-lock.json").read_bytes())
    payload = {
        key: value for key, value in source.items() if key != "runtime_lock_sha256"
    }
    payload.update(
        {
            "compatible_frozen_noskill_runtime_lock_sha256": source[
                "runtime_lock_sha256"
            ],
            "noskill_execution_contract_sha256": (NOSKILL_EXECUTION_CONTRACT_SHA256),
        }
    )
    return {
        **payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _hard_ledger_target_runtime_lock() -> dict:
    source = json.loads(
        (HARD_LEDGER_SOURCE_RUNTIME_ROOT / "runtime-lock.json").read_bytes()
    )
    payload = {
        key: value for key, value in source.items() if key != "runtime_lock_sha256"
    }
    payload["compatible_frozen_noskill_runtime_lock_sha256"] = source[
        "runtime_lock_sha256"
    ]
    return {
        **payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _launch():
    return load_portfolio_launch_package(
        TARGET_LAUNCH_ROOT,
        expected_plan_file_sha256=TARGET_LAUNCH_PLAN_FILE_SHA256,
    )


def _hard_ledger_launch():
    return load_portfolio_launch_package(
        HARD_LEDGER_TARGET_LAUNCH_ROOT,
        expected_plan_file_sha256=HARD_LEDGER_TARGET_LAUNCH_PLAN_FILE_SHA256,
        _allow_legacy_budget_contract=True,
    )


def _hard_ledger_cost_payloads() -> tuple[dict, dict]:
    shared = {
        "budget_authority_sha256": "a" * 64,
        "budget_ledger_last_event_index": 723,
        "budget_ledger_last_event_sha256": "b" * 64,
        "budget_ledger_settled_actual_cost_cny": "11.723577450000",
        "budget_ledger_unresolved_reserved_cost_cny": "3.962470400000",
        "budget_ledger_accountable_cost_cny": "15.686047850000",
        "prior_dashscope_observed_cost_cny": 0.0,
    }
    audit = {
        **shared,
        "dashscope_observed_cost_cny": 11.72357745,
        "cumulative_dashscope_observed_cost_cny": 11.72357745,
    }
    summary = {
        **shared,
        "observed_dashscope_cost_cny": 11.72357745,
        "cumulative_dashscope_cost_cny": 11.72357745,
    }
    return audit, summary


def test_partial_repair_cli_locks_four_shards_and_phase_cap() -> None:
    arguments = build_parser().parse_args(
        [
            "--launch-root",
            "launch",
            "--launch-plan-file-sha256",
            "a" * 64,
            "--runtime-root",
            "runtime",
            "--runtime-lock-file-sha256",
            "b" * 64,
            "--rubric",
            "rubric.json",
            "--output-dir",
            "execution",
            "--approved-dashscope-budget-cny",
            "51",
            "--prior-dashscope-observed-cost-cny",
            "21.197859",
            "--phase-cumulative-cap-cny",
            "38",
            "--source-execution-root",
            "source-execution",
            "--source-shard-id",
            SOURCE_SHARD_ID,
            "--expected-source-shard-audit-sha256",
            SOURCE_AUDIT_SHA256,
            "--authorized-shard-id",
            "01-batch-llm-static",
            "--authorized-shard-id",
            "02-batch-s1",
            "--authorized-shard-id",
            "03-batch-s1s2",
            "--authorized-shard-id",
            "04-batch-full",
        ]
    )

    assert arguments.phase_cumulative_cap_cny == Decimal("38.000000000000")
    assert arguments.authorized_shard_id == [
        "01-batch-llm-static",
        "02-batch-s1",
        "03-batch-s1s2",
        "04-batch-full",
    ]


def test_full_matrix_cli_accepts_carried_prior_without_frozen_shards() -> None:
    arguments = build_parser().parse_args(
        [
            "--execution-scope",
            "full_matrix",
            "--launch-root",
            "launch",
            "--launch-plan-file-sha256",
            "a" * 64,
            "--runtime-root",
            "runtime",
            "--runtime-lock-file-sha256",
            "b" * 64,
            "--rubric",
            "rubric.json",
            "--output-dir",
            "execution",
            "--approved-dashscope-budget-cny",
            "200",
            "--phase-cumulative-cap-cny",
            "200",
            "--prior-dashscope-observed-cost-cny",
            "25.4295917",
        ]
    )

    assert arguments.execution_scope == "full_matrix"
    assert arguments.phase_cumulative_cap_cny == Decimal("200.000000000000")
    assert arguments.prior_dashscope_observed_cost_cny == Decimal("25.429591700000")
    assert arguments.authorized_shard_id == []
    assert arguments.source_execution_root is None
    assert arguments.source_shard_id is None
    assert arguments.expected_source_shard_audit_sha256 is None


def test_core_canary_cli_has_no_ad_hoc_shard_or_source_inputs() -> None:
    arguments = build_parser().parse_args(
        [
            "--execution-scope",
            "core_canary",
            "--launch-root",
            "launch",
            "--launch-plan-file-sha256",
            "a" * 64,
            "--runtime-root",
            "runtime",
            "--runtime-lock-file-sha256",
            "b" * 64,
            "--rubric",
            "rubric.json",
            "--output-dir",
            "execution",
            "--approved-dashscope-budget-cny",
            "100",
            "--phase-cumulative-cap-cny",
            "30",
        ]
    )

    assert arguments.execution_scope == "core_canary"
    assert arguments.phase_cumulative_cap_cny == Decimal("30.000000000000")
    assert arguments.authorized_shard_id == []
    assert arguments.source_execution_root is None
    assert arguments.source_shard_id is None


def test_static_opt_cli_selects_dedicated_scope_without_ad_hoc_inputs() -> None:
    arguments = preparation.build_parser().parse_args(
        [
            "--execution-scope",
            "static_opt_rollout",
            "--launch-root",
            "launch",
            "--launch-plan-file-sha256",
            "a" * 64,
            "--runtime-root",
            "runtime",
            "--runtime-lock-file-sha256",
            "b" * 64,
            "--output-dir",
            "execution",
            "--approved-dashscope-budget-cny",
            "200",
        ]
    )

    assert arguments.execution_scope == "static_opt_rollout"
    assert arguments.authorized_shard_id == []
    assert arguments.source_execution_root is None
    assert arguments.source_shard_id is None
    assert arguments.expected_source_shard_audit_sha256 is None
    assert arguments.rubric is None


def test_static_opt_budget_contract_does_not_require_final_judge_sources() -> None:
    runtime_lock = dict(preparation._ACTIVE_BUDGET_CONTRACT)

    preparation._require_active_budget_runtime(
        runtime_lock,
        label="Static opt runtime",
    )

    runtime_lock.pop("portfolio_budget_policy_sha256")
    with pytest.raises(ValueError, match="active hard-budget contract"):
        preparation._require_active_budget_runtime(
            runtime_lock,
            label="Static opt runtime",
        )


def test_source_cost_accounting_keeps_legacy_shard_local_semantics() -> None:
    local = Decimal("2.762144250000")
    assert (
        _validate_source_cost_accounting(
            audit={"dashscope_observed_cost_cny": 2.7621442500000004},
            summary={"observed_dashscope_cost_cny": 2.7621442500000004},
            checkpoint_local_cost_cny=local,
        )
        == local
    )

    with pytest.raises(ValueError, match="observed cost differs from checkpoints"):
        _validate_source_cost_accounting(
            audit={"dashscope_observed_cost_cny": 11.72357745},
            summary={"observed_dashscope_cost_cny": 11.72357745},
            checkpoint_local_cost_cny=local,
        )


def test_source_cost_accounting_accepts_hard_ledger_root_scope() -> None:
    audit, summary = _hard_ledger_cost_payloads()

    assert _validate_source_cost_accounting(
        audit=audit,
        summary=summary,
        checkpoint_local_cost_cny=Decimal("2.959384900000"),
    ) == Decimal("11.723577450000")


def test_hard_ledger_prior_remains_independent_of_shard_local_cost() -> None:
    audit, summary = _hard_ledger_cost_payloads()
    for payload in (audit, summary):
        payload["prior_dashscope_observed_cost_cny"] = 5.0
        payload["budget_ledger_accountable_cost_cny"] = "20.686047850000"
    audit["cumulative_dashscope_observed_cost_cny"] = 16.72357745
    summary["cumulative_dashscope_cost_cny"] = 16.72357745

    assert _validate_source_cost_accounting(
        audit=audit,
        summary=summary,
        checkpoint_local_cost_cny=Decimal("2.959384900000"),
    ) == Decimal("11.723577450000")


def test_source_cost_accounting_includes_forfeited_full_reserves() -> None:
    audit, summary = _hard_ledger_cost_payloads()
    for payload in (audit, summary):
        payload["budget_ledger_forfeited_reserved_cost_cny"] = "2.000000000000"
        payload["budget_ledger_accountable_cost_cny"] = "17.686047850000"

    assert _validate_source_cost_accounting(
        audit=audit,
        summary=summary,
        checkpoint_local_cost_cny=Decimal("2.959384900000"),
    ) == Decimal("11.723577450000")


def test_source_cost_accounting_rejects_one_sided_forfeited_cost() -> None:
    audit, summary = _hard_ledger_cost_payloads()
    audit["budget_ledger_forfeited_reserved_cost_cny"] = "2.000000000000"

    with pytest.raises(ValueError, match="checkpoints are incomplete"):
        _validate_source_cost_accounting(
            audit=audit,
            summary=summary,
            checkpoint_local_cost_cny=Decimal("2.959384900000"),
        )


def test_source_cost_accounting_rejects_hard_ledger_scope_drift() -> None:
    audit, summary = _hard_ledger_cost_payloads()
    audit["dashscope_observed_cost_cny"] = 2.9593849

    with pytest.raises(
        ValueError,
        match="execution-root observed cost differs from ledger settled cost",
    ):
        _validate_source_cost_accounting(
            audit=audit,
            summary=summary,
            checkpoint_local_cost_cny=Decimal("2.959384900000"),
        )


def test_source_cost_accounting_rejects_local_above_root() -> None:
    audit, summary = _hard_ledger_cost_payloads()

    with pytest.raises(ValueError, match="checkpoint-local cost exceeds"):
        _validate_source_cost_accounting(
            audit=audit,
            summary=summary,
            checkpoint_local_cost_cny=Decimal("11.723577450001"),
        )


def test_external_frozen_hard_ledger_records_both_cost_scopes() -> None:
    binding, _ = _external_frozen_noskill_binding(
        source_execution_root=HARD_LEDGER_SOURCE_EXECUTION_ROOT,
        source_shard_id=SOURCE_SHARD_ID,
        expected_audit_sha256=HARD_LEDGER_SOURCE_AUDIT_SHA256,
        target_launch=_hard_ledger_launch(),
        target_runtime_lock=_hard_ledger_target_runtime_lock(),
        rubric_file_sha256=(
            "3270d3f12ca31a24c78e250437a2f159ac660f6df1fd9f215796720b99636e62"
        ),
        rubric_content_sha256=(
            "39cf1f7caf0b45fe4099adbf999ddd5ed42e4a88a1ba2ef9b35f2d1b748fc01f"
        ),
    )

    assert binding["checkpoint_local_dashscope_observed_cost_cny"] == ("2.959384900000")
    assert binding["source_execution_root_settled_actual_cost_cny"] == (
        "11.723577450000"
    )


def test_external_frozen_schema3_noskill_is_not_reused_by_schema4_chain() -> None:
    with pytest.raises(ValueError, match="card-requirement guard"):
        _external_frozen_noskill_binding(
            source_execution_root=SOURCE_EXECUTION_ROOT,
            source_shard_id=SOURCE_SHARD_ID,
            expected_audit_sha256=SOURCE_AUDIT_SHA256,
            target_launch=_launch(),
            target_runtime_lock=_target_runtime_lock(),
            rubric_file_sha256=(
                "3270d3f12ca31a24c78e250437a2f159ac660f6df1fd9f215796720b99636e62"
            ),
            rubric_content_sha256=(
                "39cf1f7caf0b45fe4099adbf999ddd5ed42e4a88a1ba2ef9b35f2d1b748fc01f"
            ),
        )


def test_external_frozen_noskill_rejects_wrong_expected_audit() -> None:
    with pytest.raises(ValueError, match="expected self digest"):
        _external_frozen_noskill_binding(
            source_execution_root=SOURCE_EXECUTION_ROOT,
            source_shard_id=SOURCE_SHARD_ID,
            expected_audit_sha256="a" * 64,
            target_launch=_launch(),
            target_runtime_lock=_target_runtime_lock(),
            rubric_file_sha256=(
                "3270d3f12ca31a24c78e250437a2f159ac660f6df1fd9f215796720b99636e62"
            ),
            rubric_content_sha256=(
                "39cf1f7caf0b45fe4099adbf999ddd5ed42e4a88a1ba2ef9b35f2d1b748fc01f"
            ),
        )


def test_external_frozen_noskill_rejects_target_public_input_drift() -> None:
    launch = _launch()
    target_index = next(
        index
        for index, item in enumerate(launch.instances)
        if item.shard_id == SOURCE_SHARD_ID
    )
    drifted_instances = list(launch.instances)
    drifted_instances[target_index] = drifted_instances[target_index].model_copy(
        update={"public_input_sha256": "f" * 64}
    )
    drifted_launch = replace(launch, instances=tuple(drifted_instances))

    with pytest.raises(ValueError, match="query/public/image drift"):
        _external_frozen_noskill_binding(
            source_execution_root=SOURCE_EXECUTION_ROOT,
            source_shard_id=SOURCE_SHARD_ID,
            expected_audit_sha256=SOURCE_AUDIT_SHA256,
            target_launch=drifted_launch,
            target_runtime_lock=_target_runtime_lock(),
            rubric_file_sha256=(
                "3270d3f12ca31a24c78e250437a2f159ac660f6df1fd9f215796720b99636e62"
            ),
            rubric_content_sha256=(
                "39cf1f7caf0b45fe4099adbf999ddd5ed42e4a88a1ba2ef9b35f2d1b748fc01f"
            ),
        )


def _active_comparison_lock() -> dict[str, object]:
    return {
        **preparation._ACTIVE_FINAL_RESULT_LOCK,
        **preparation._ACTIVE_BUDGET_CONTRACT,
        **preparation._ACTIVE_EVALUATOR_SOURCE_LOCK,
        "final_judge_parser_policy_version": (
            preparation.FINAL_JUDGE_PARSER_POLICY_VERSION_V4
        ),
        "final_judge_parser_policy_sha256": (
            preparation.FINAL_JUDGE_PARSER_POLICY_SHA256_V4
        ),
        "system_prompt_sha256": "a" * 64,
        "tool_registry_sha256": "b" * 64,
        "tool_registry_runtime_sha256": "c" * 64,
        "noskill_execution_contract_sha256": NOSKILL_EXECUTION_CONTRACT_SHA256,
        "noskill_execution_policy_version": NOSKILL_EXECUTION_POLICY_VERSION,
    }


def test_comparison_requires_current_noskill_contract_on_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(
        assistant_provider="qwen",
        assistant_model="qwen3-vl-flash-2026-01-22",
    )
    launch = SimpleNamespace(plan=plan)
    projection = {
        "models": {
            "final_judge": {
                "provider": "kimi",
                "model": "kimi-k2.6",
                "endpoint": "https://judge.example/v1",
            },
            "feedback": {
                "provider": "gemini",
                "model": "gemini-test",
                "endpoint": "https://feedback.example/v1",
            },
        }
    }
    monkeypatch.setattr(
        preparation,
        "_comparison_launch_projection",
        lambda _plan: projection,
    )
    endpoints = {
        "endpoint.dashscope": "https://assistant.example/v1",
        "endpoint.aifast": "https://feedback.example/v1",
        "endpoint.kimi_dashscope": "https://judge.example/v1",
    }
    monkeypatch.setattr(
        preparation,
        "_launch_endpoint",
        lambda _plan, check_id: endpoints[check_id],
    )
    request_contract = {
        "assistant": {
            "backbone": {
                "provider": "qwen",
                "model": "qwen3-vl-flash-2026-01-22",
                "endpoint": "https://assistant.example/v1",
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": None,
                "system_prompt_sha256": "a" * 64,
            },
            "budget": {
                "max_input_tokens": 32768,
                "max_output_tokens": 4096,
                "max_tool_calls": 3,
                "max_turns": 5,
                "timeout_ms": 180000,
                "budget_sha256": "d" * 64,
            },
            "registry": {
                "registry_sha256": "b" * 64,
                "registry_runtime_sha256": "c" * 64,
                "lock_sha256": "e" * 64,
            },
        },
        "final_judge": {
            **projection["models"]["final_judge"],
            "max_tokens": 2048,
            "max_attempts": preparation.FINAL_JUDGE_MAX_ATTEMPTS,
            "retry_policy_version": preparation.FINAL_JUDGE_RETRY_POLICY_VERSION,
            "retry_policy_sha256": preparation.FINAL_JUDGE_RETRY_POLICY_SHA256,
            "thinking_budget": preparation.FINAL_JUDGE_THINKING_BUDGET,
            "max_billable_input_tokens": (
                preparation.FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
            ),
            "max_billable_output_tokens": (
                preparation.FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
            ),
            "transport_policy_version": (
                preparation.FINAL_JUDGE_TRANSPORT_POLICY_VERSION
            ),
            "transport_policy_sha256": (
                preparation.FINAL_JUDGE_TRANSPORT_POLICY_SHA256
            ),
            "requested_response_format": "json_object",
        },
    }
    source_lock = _active_comparison_lock()
    target_lock = _active_comparison_lock()
    payload, _ = preparation._comparison_contract(
        source_launch=launch,
        target_launch=launch,
        source_runtime_lock=source_lock,
        target_runtime_lock=target_lock,
        source_request_contract=request_contract,
        rubric_file_sha256="f" * 64,
        rubric_content_sha256="1" * 64,
        source_control={
            "rubric_file_sha256": "f" * 64,
            "rubric_content_sha256": "1" * 64,
        },
    )
    assert payload["field_sources"]["noskill_execution"] == (
        NOSKILL_EXECUTION_POLICY_VERSION
    )

    source_lock["noskill_execution_policy_version"] = (
        "noskill-native-function-calling-v1"
    )
    with pytest.raises(ValueError, match="NoSkill contract drifted"):
        preparation._comparison_contract(
            source_launch=launch,
            target_launch=launch,
            source_runtime_lock=source_lock,
            target_runtime_lock=target_lock,
            source_request_contract=request_contract,
            rubric_file_sha256="f" * 64,
            rubric_content_sha256="1" * 64,
            source_control={
                "rubric_file_sha256": "f" * 64,
                "rubric_content_sha256": "1" * 64,
            },
        )
