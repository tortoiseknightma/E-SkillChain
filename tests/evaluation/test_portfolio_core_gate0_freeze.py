from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

from skillchain import config
from skillchain.evaluation import portfolio_execution
from skillchain.evaluation.final_runtime import (
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_ANSWER_MAX_TOKENS,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
)
from skillchain.runners.assistant import (
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    SharedStage2RouteArtifact,
    assistant_router_contract_payload,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "specs" / "evaluation" / "portfolio-core-gate0-freeze-v1.json"
)
CONTRACT_V2_PATH = (
    ROOT / "specs" / "evaluation" / "portfolio-core-gate0-freeze-v2.json"
)
CONTRACT_V3_PATH = (
    ROOT / "specs" / "evaluation" / "portfolio-core-gate0-freeze-v3.json"
)
ROLE_SELECTION_PATH = ROOT / "specs" / "authoring" / "model-role-selection-v4.json"
SHARD_RUNNER_PATH = ROOT / "scripts" / "run_portfolio_shard.py"


def _load_json(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    raw = read_stable_regular_file(path, label=label)
    parsed = parse_canonical_json(raw, label=label)
    assert isinstance(parsed, dict)
    return raw, parsed


def _literal_dict_assignment(path: Path, name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, dict)
        matches.append(value)
    assert len(matches) == 1
    return matches[0]


def _literal_frozenset_assignment(path: Path, name: str) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[frozenset[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Name)
        assert node.value.func.id == "frozenset"
        assert len(node.value.args) == 1
        value = ast.literal_eval(node.value.args[0])
        assert isinstance(value, set)
        matches.append(frozenset(value))
    assert len(matches) == 1
    return matches[0]


def test_gate0_freeze_is_canonical_self_hashed_and_non_authorizing() -> None:
    raw, contract = _load_json(CONTRACT_PATH, label="Core Gate0 freeze")
    assert raw == canonical_json_bytes(contract)

    unsigned = dict(contract)
    supplied_hash = unsigned.pop("contract_sha256")
    assert supplied_hash == sha256_bytes(canonical_json_bytes(unsigned))
    assert supplied_hash == (
        "72a0e51ba592fef82180856326bf815778c647cafca6b7bd506f4e8516cde385"
    )
    assert contract["policy_version"] == "portfolio-core-gate0-freeze-v1"
    assert contract["track"] == "portfolio"
    assert contract["formal_eligible"] is False
    assert contract["core_execution_authorized"] is False
    assert contract["model_calls_performed"] == 0
    assert contract["status"] == "frozen_contract_only_not_execution_authority"


def test_gate0_v2_freezes_active_router_v6_and_distinct_fixed_repair() -> None:
    raw, contract = _load_json(CONTRACT_V2_PATH, label="Core Gate0 freeze v2")
    assert raw == canonical_json_bytes(contract)

    unsigned = dict(contract)
    supplied_hash = unsigned.pop("contract_sha256")
    assert supplied_hash == sha256_bytes(canonical_json_bytes(unsigned))
    assert supplied_hash == (
        "447538c944dbbccd8c435dc2944a77e33b60746da5945f51fcfc35f0e7417230"
    )
    assert contract["policy_version"] == "portfolio-core-gate0-freeze-v2"
    assert contract["core_execution_authorized"] is False
    assert contract["model_calls_performed"] == 0

    runtime = contract["runtime"]
    models = contract["models_and_limits"]
    assert isinstance(runtime, dict)
    assert isinstance(models, dict)
    router = runtime["router_contract"]
    assistant = models["assistant"]
    assert isinstance(router, dict)
    assert isinstance(assistant, dict)
    assert router["shared_stage2_route_policy_version"] == (
        SHARED_STAGE2_ROUTE_POLICY_VERSION
    )
    assert SHARED_STAGE2_ROUTE_POLICY_VERSION == "shared-stage2-route-v6"
    assert router["shared_stage2_route_schema_sha256"] == sha256_bytes(
        canonical_json_bytes(SharedStage2RouteArtifact.model_json_schema())
    )
    assert router["shared_stage2_route_schema_sha256"] == (
        "9459c916a493e0bfb461400db23e2eca2d148b98acecefe42631763492450e3f"
    )
    assert router["core_route_output_json_schema_sha256"] == (
        "ee6f3c28d3d97fc43b0cbeb7ae9947529614ddd82c59b7f47ff32137ec3ebb48"
    )
    assert router["portfolio_router_contract_version"] == (
        PORTFOLIO_ROUTER_CONTRACT_VERSION
    )
    assert PORTFOLIO_ROUTER_CONTRACT_VERSION == "portfolio-assistant-router-v6"
    assert router["portfolio_router_contract_sha256"] == (
        PORTFOLIO_ROUTER_CONTRACT_SHA256
    )
    assert PORTFOLIO_ROUTER_CONTRACT_SHA256 == (
        "dd6b73405e0507605234e3cdf3621c9361bd2105bb9e49f46c5bdf3b81d1b355"
    )
    assert router["portfolio_router_contract_payload"] == (
        assistant_router_contract_payload()
    )
    payload = router["portfolio_router_contract_payload"]
    assert payload["route_user_input_fields"] == ["turns"]
    assert payload["route_image_attachment"] is False
    assert payload["parser"] == "full_strict_json_no_prefix_recovery"
    assert payload["ignored_top_level_response_keys"] == [
        "asset_id",
        "description",
        "text",
        "turns",
    ]
    assert payload["unknown_top_level_response_key_policy"] == "reject"
    assert payload["repair_wire_policy"] == (
        "fixed_prompt_must_differ_from_initial"
    )
    assert payload["repair_previous_response_input"] == "forbidden"

    assert router["portfolio_failure_policy_version"] == (
        portfolio_execution.PORTFOLIO_FAILURE_POLICY_VERSION_V3
    )
    assert portfolio_execution.PORTFOLIO_FAILURE_POLICY_VERSION_V3 == (
        "portfolio-shard-attempt-v3"
    )
    assert router["portfolio_router_request_max_output_tokens"] == (
        portfolio_execution.PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
    )
    assert portfolio_execution.PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS == 64
    assert router["portfolio_router_pricing_reservation_max_output_tokens"] == (
        portfolio_execution.PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
    )
    assert portfolio_execution.PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS == 512
    assert assistant["route_request_max_output_tokens"] == 64
    assert assistant["route_pricing_reservation_max_output_tokens"] == 512
    assert "route_max_output_tokens" not in assistant

    route_retry = runtime["retries"]["route_format_retry"]
    assert route_retry["eligible_failures"] == [
        "length",
        "empty",
        "invalid_json",
        "non_object",
        "unexpected_keys",
        "schema_invalid",
    ]
    assert route_retry["eligible_failure_subtypes"] == [
        "invalid_route_json",
        "length",
        "response_empty_text",
    ]
    assert route_retry["eligible_payload_statuses"] == [
        "invalid_json",
        "non_object",
        "schema_invalid",
        "unexpected_keys",
    ]
    assert route_retry["eligible_failure_shapes"] == [
        {
            "attempt_failure_subtype": "route_contract_length",
            "failure_subtype": "length",
            "payload_status": "not_examined",
        },
        {
            "attempt_failure_subtype": "route_contract_empty",
            "failure_subtype": "response_empty_text",
            "payload_status": "empty",
        },
        {
            "attempt_failure_subtype": "route_contract_invalid_json",
            "failure_subtype": "invalid_route_json",
            "payload_status": "invalid_json",
        },
        {
            "attempt_failure_subtype": "route_contract_non_object",
            "failure_subtype": "invalid_route_json",
            "payload_status": "non_object",
        },
        {
            "attempt_failure_subtype": "route_contract_unexpected_keys",
            "failure_subtype": "invalid_route_json",
            "payload_status": "unexpected_keys",
        },
        {
            "attempt_failure_subtype": "route_contract_schema_invalid",
            "failure_subtype": "invalid_route_json",
            "payload_status": "schema_invalid",
        },
    ]
    assert router["eligible_repair_failure_shapes"] == route_retry[
        "eligible_failure_shapes"
    ]
    assert router["non_repairable_failure_shapes"] == [
        {"failure_subtype": "out_of_enum", "payload_status": "out_of_enum"},
        {
            "failure_subtype": "response_tool_calls",
            "payload_status": "not_examined",
        },
        {
            "failure_subtype": "response_finish_reason",
            "payload_status": "not_examined",
        },
        {
            "failure_subtype": "response_contract",
            "payload_status": "not_examined",
        },
        {"failure_subtype": "route_budget", "payload_status": "not_examined"},
    ]
    assert route_retry["wire_policy"] == "fixed_prompt_must_differ_from_initial"
    assert route_retry["repair_request_variant"] == "fixed_repair"
    assert route_retry["response_prefix_recovery"] is False
    assert router["response_prefix_recovery"] is False
    assert "route_empty" not in runtime["retries"]["non_retryable"]


def test_gate0_v3_freezes_forfeit_terminal_accounting_and_safe_canary_parallelism() -> None:
    raw, contract = _load_json(CONTRACT_V3_PATH, label="Core Gate0 freeze v3")
    assert raw == canonical_json_bytes(contract)

    unsigned = dict(contract)
    supplied_hash = unsigned.pop("contract_sha256")
    assert supplied_hash == sha256_bytes(canonical_json_bytes(unsigned))
    assert supplied_hash == (
        "52ba0ce63487e8bba16921079e3aeea98f809531129024013c909adf98f262cd"
    )
    assert sha256_bytes(raw) == (
        "379c0716f2af8f08d24287a0c329a2195b2679cd94dea45b8c3983eac4a34bc4"
    )
    assert contract["policy_version"] == "portfolio-core-gate0-freeze-v3"
    assert contract["core_execution_authorized"] is False
    assert contract["model_calls_performed"] == 0

    budget = contract["budget"]
    assert budget["policy_version"] == (
        portfolio_execution.PORTFOLIO_BUDGET_POLICY_VERSION
    )
    assert budget["policy_version"] == "portfolio-call-hard-cap-v2"
    assert budget["accounting_policy"] == (
        "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
    )
    assert budget["exception_policy"] == (
        "append_full_reserve_forfeit_without_captured_response"
    )
    assert budget["terminal_outcome_policy"] == (
        "exactly_one_of_settlement_or_forfeit"
    )
    assert budget["forfeit_disposition"] == (
        "charged_full_reserve_unknown_actual_usage"
    )

    runtime = contract["runtime"]
    concurrency = runtime["concurrency"]
    assert concurrency["qwen_inflight"] == 1
    assert concurrency["kimi_inflight"] == 2
    assert concurrency["qwen_requests_per_minute"] == 20
    assert concurrency["qwen_minimum_start_interval_seconds"] == "3.000000"

    router = runtime["router_contract"]
    assert router["portfolio_failure_policy_version"] == (
        portfolio_execution.PORTFOLIO_FAILURE_POLICY_VERSION
    )
    assert portfolio_execution.PORTFOLIO_FAILURE_POLICY_VERSION == (
        "portfolio-shard-attempt-v4"
    )
    assert router["portfolio_router_contract_version"] == (
        PORTFOLIO_ROUTER_CONTRACT_VERSION
    )
    assert router["portfolio_router_contract_sha256"] == (
        PORTFOLIO_ROUTER_CONTRACT_SHA256
    )

    recovery = runtime["pause_resume"]
    assert recovery["provider_exception_terminal_event"] == (
        "append_create_only_full_reserve_forfeit_before_attempt_receipt"
    )
    assert recovery["dead_owner_orphan_recovery"] == (
        "append_one_full_reserve_forfeit_after_confirmed_owner_exit"
    )
    assert recovery["unresolved_provider_reservation_policy"] == (
        "forfeit_full_reserve_after_failure_or_confirmed_dead_owner_never_release_unknown_cost"
    )

    judge = runtime["final_judge_contract"]
    assert judge == {
        "cache_namespace": FINAL_JUDGE_CACHE_NAMESPACE,
        "forfeit_binding_fields": [
            "forfeited_reservation_sha256",
            "budget_forfeit_sha256",
        ],
        "provider_pre_response_terminal_event": "full_reserve_forfeit",
        "result_schema_version": FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    }
    assert FINAL_JUDGE_RESULT_SCHEMA_VERSION == 9
    assert FINAL_JUDGE_CACHE_NAMESPACE == "final-evaluator-v10"


def test_gate0_models_prices_and_token_envelopes_bind_active_runtime() -> None:
    _, contract = _load_json(CONTRACT_PATH, label="Core Gate0 freeze")
    models = contract["models_and_limits"]
    pricing = contract["pricing"]
    assert isinstance(models, dict)
    assert isinstance(pricing, dict)

    assistant = models["assistant"]
    assert isinstance(assistant, dict)
    assert assistant["provider"] == config.ASSISTANT_PROVIDER == "qwen"
    assert (
        assistant["model"]
        == config.LEGACY_PORTFOLIO_ASSISTANT_MODEL
        == portfolio_execution.PORTFOLIO_QWEN_MODEL
        == "qwen3-vl-flash-2026-01-22"
    )
    assert assistant["thinking"] is False
    assert assistant["provider_seed"] is None
    assert assistant["provider_max_input_tokens"] == (
        portfolio_execution.PORTFOLIO_QWEN_PROVIDER_MAX_INPUT_TOKENS
    )
    assert assistant["route_max_output_tokens"] == (
        portfolio_execution.PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
    )

    active_assistant_budget = _literal_dict_assignment(
        SHARD_RUNNER_PATH, "budget_payload"
    )
    assert active_assistant_budget == {
        "max_input_tokens": assistant["aggregate_max_input_tokens"],
        "max_output_tokens": assistant["aggregate_max_output_tokens"],
        "max_tool_calls": assistant["max_tool_calls"],
        "max_turns": assistant["request_max_turns_including_route"],
        "timeout_ms": 180000,
    }
    assert assistant["max_action_model_calls"] == assistant["max_tool_calls"] + 1
    assert assistant["max_tool_calls_per_turn"] == 1
    assert assistant["parallel_tool_calls"] is False

    _, role_selection = _load_json(ROLE_SELECTION_PATH, label="Core role selection")
    role_binding = models["role_selection_binding"]
    assert isinstance(role_binding, dict)
    role_raw = ROLE_SELECTION_PATH.read_bytes()
    assert sha256_bytes(role_raw) == role_binding["file_sha256"]
    assert role_selection["selection_sha256"] == role_binding["selection_sha256"]
    assert role_selection["assistant"]["model"] == assistant["model"]
    assert role_selection["feedback_evaluator"]["model"] == "kimi-k2.6"
    assert role_selection["offline_judge"]["model"] == "kimi-k2.6"

    for role in ("feedback_evaluator", "legacy_final_judge", "blind_pairwise_judge"):
        role_contract = models[role]
        assert isinstance(role_contract, dict)
        assert role_contract["provider"] == "kimi"
        assert (
            role_contract["model"]
            == config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL
            == portfolio_execution.PORTFOLIO_KIMI_MODEL
            == "kimi-k2.6"
        )
        assert role_contract["provider_seed"] is None

    for role in ("legacy_final_judge", "blind_pairwise_judge"):
        judge = models[role]
        assert isinstance(judge, dict)
        assert judge["answer_max_tokens"] == FINAL_JUDGE_ANSWER_MAX_TOKENS
        # Gate0 is an immutable legacy Kimi contract; it must not follow the
        # active Final Judge role used by new Core Fast runs.
        assert judge["thinking_budget"] == 6144
        assert judge["max_billable_input_tokens"] == 32768
        assert judge["max_billable_output_tokens"] == 8192
        assert judge["provider_input_token_reserve"] == 229376
        assert judge["provider_output_token_limit"] == 16384

    qwen_price = pricing["qwen"]
    kimi_price = pricing["kimi"]
    assert isinstance(qwen_price, dict)
    assert isinstance(kimi_price, dict)
    runtime_tiers = getattr(portfolio_execution, "_QWEN_SETTLEMENT_TIERS")
    expected_tiers = [
        {
            "max_input_tokens_inclusive": maximum,
            "input_cny_per_million_tokens": format(input_rate, "f"),
            "output_cny_per_million_tokens": format(output_rate, "f"),
        }
        for maximum, input_rate, output_rate in runtime_tiers
    ]
    assert qwen_price["settlement_input_length_tiers"] == expected_tiers
    assert qwen_price["reservation_input_cny_per_million_tokens"] == format(
        portfolio_execution.PORTFOLIO_QWEN_RESERVE_INPUT_CNY_PER_MILLION, "f"
    )
    assert qwen_price["reservation_output_cny_per_million_tokens"] == format(
        portfolio_execution.PORTFOLIO_QWEN_RESERVE_OUTPUT_CNY_PER_MILLION, "f"
    )
    assert kimi_price["input_cny_per_million_tokens"] == format(
        portfolio_execution.PORTFOLIO_KIMI_INPUT_CNY_PER_MILLION, "f"
    )
    assert kimi_price["output_cny_per_million_tokens"] == format(
        portfolio_execution.PORTFOLIO_KIMI_OUTPUT_CNY_PER_MILLION, "f"
    )
    assert pricing["source_url"] == (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )


def test_gate0_core_sharding_concurrency_retry_and_budget_are_exact() -> None:
    _, contract = _load_json(CONTRACT_PATH, label="Core Gate0 freeze")
    scope = contract["scope"]
    runtime = contract["runtime"]
    budget = contract["budget"]
    assert isinstance(scope, dict)
    assert isinstance(runtime, dict)
    assert isinstance(budget, dict)

    assert scope["dataset_query_count"] == 1500
    assert scope["canonical_full_core_row_count"] == 7500
    assert scope["splits"] == {
        "dev_mini": 200,
        "optimization_pool": 800,
        "test": 300,
        "validation": 200,
    }
    assert scope["config_order"] == [
        "noskill",
        "llm_static",
        "s1",
        "s1s2",
        "full",
    ]
    assert len(scope["capabilities"]) == 6

    sharding = runtime["sharding"]
    concurrency = runtime["concurrency"]
    recovery = runtime["pause_resume"]
    retries = runtime["retries"]
    assert isinstance(sharding, dict)
    assert isinstance(concurrency, dict)
    assert isinstance(recovery, dict)
    assert isinstance(retries, dict)
    assert sharding == {
        "full_core_config_shards": 300,
        "full_core_query_shards": 60,
        "queries_per_shard": 25,
        "single_wave_canary_before_parallel_run": True,
    }
    assert concurrency["qwen_inflight"] == 2
    assert concurrency["kimi_inflight"] == 8
    assert concurrency["max_active_waves"] == 10
    assert concurrency["score_dependent_change_forbidden"] is True
    assert recovery["circuit_breaker_consecutive_retryable_failures"] == (
        portfolio_execution.PORTFOLIO_CIRCUIT_BREAKER_THRESHOLD
    )
    assert recovery["max_retryable_attempts_per_query"] == (
        portfolio_execution.PORTFOLIO_MAX_RETRYABLE_ATTEMPTS_PER_QUERY
    )
    assert recovery["recoverable_stop_exit_code"] == (
        portfolio_execution.PORTFOLIO_RECOVERABLE_STOP_EXIT_CODE
    )
    assert recovery["automatic_same_process_infrastructure_retry"] is False

    route_retry = retries["route_format_retry"]
    final_retry = retries["final_judge_format_retry"]
    pairwise_retry = retries["blind_pairwise_format_retry"]
    assert isinstance(route_retry, dict)
    assert isinstance(final_retry, dict)
    assert isinstance(pairwise_retry, dict)
    active_route_statuses = _literal_frozenset_assignment(
        SHARD_RUNNER_PATH, "_ROUTE_CONTRACT_RETRYABLE_PAYLOAD_STATUSES"
    )
    assert route_retry["eligible_payload_statuses"] == sorted(active_route_statuses)
    assert route_retry["retry_count"] == 1
    assert route_retry["max_attempts"] == 2
    assert final_retry["retry_count"] == 1
    assert final_retry["max_attempts"] == FINAL_JUDGE_MAX_ATTEMPTS == 2
    assert FINAL_JUDGE_RETRY_POLICY_VERSION == (
        "empty-or-invalid-final-response-single-retry-v3"
    )
    assert pairwise_retry["retry_count"] == 1
    assert pairwise_retry["max_attempts"] == 2

    assert budget["currency"] == "CNY"
    assert Decimal(budget["target_cap_cny"]) == Decimal("180")
    assert Decimal(budget["hard_cap_cny"]) == Decimal("200")
    assert Decimal(budget["target_cap_cny"]) < Decimal(budget["hard_cap_cny"])
    assert budget["authority_required_before_first_provider_call"] is True


def test_gate0_gcs_pairwise_seeds_and_single_unseal_event_are_frozen() -> None:
    _, contract = _load_json(CONTRACT_PATH, label="Core Gate0 freeze")
    evaluation = contract["evaluation"]
    reproducibility = contract["reproducibility"]
    unseal = contract["test_unseal"]
    assert isinstance(evaluation, dict)
    assert isinstance(reproducibility, dict)
    assert isinstance(unseal, dict)

    gcs = evaluation["grounded_contract_success"]
    assert isinstance(gcs, dict)
    assert gcs["formula"] == (
        "int(route_acceptable AND no_hard_error AND tool_contract_pass AND "
        "evidence_grounded AND output_contract_pass)"
    )
    assert set(gcs["components"]) == {
        "route_acceptable",
        "no_hard_error",
        "tool_contract_pass",
        "evidence_grounded",
        "output_contract_pass",
    }
    assert gcs["missing_invalid_or_unresolved_component_value"] == 0
    assert gcs["fixed_full_denominator"] is True
    assert gcs["capability_macro_formula"] == (
        "unweighted_mean_of_six_capability_gcs_means"
    )
    system_gate = gcs["system_gain_threshold_final_minus_static"]
    semantic_gate = gcs["robust_semantic_quality_threshold"]
    assert isinstance(system_gate, dict)
    assert isinstance(semantic_gate, dict)
    assert system_gate == {
        "capability_macro_gcs_delta_pp_min": "2.000000",
        "cluster_bootstrap_ci95_lower_pp_min": "0.000000",
        "hard_error_delta_pp_max": "1.000000",
        "per_capability_gcs_delta_pp_min": "-3.000000",
    }
    assert semantic_gate["human_tie_adjusted_preference_point_min"] == "0.550000"
    assert (
        semantic_gate[
            "human_tie_adjusted_preference_ci95_lower_strictly_greater_than"
        ]
        == "0.500000"
    )

    pairwise = evaluation["blind_pairwise"]
    assert isinstance(pairwise, dict)
    assert pairwise["decision_values"] == ["A", "B", "tie", "invalid"]
    assert pairwise["smoke_schedule"] == {
        "pairs": 20,
        "total_calls": 40,
        "use_abba": True,
    }
    assert pairwise["calibration_schedule"] == {
        "dev_pairs": 60,
        "orientations_per_pair": 2,
        "total_calls": 120,
    }
    assert pairwise["test_orientation"] == {
        "ab_count": 150,
        "ba_count": 150,
        "design": "fixed_seed_balanced_single_orientation_per_pair",
        "pairs": 300,
        "total_calls": 300,
    }
    assert pairwise["optional_second_test_batch_authorized"] is False
    assert pairwise["tie_adjusted_final_preference_formula"] == (
        "(final_wins + 0.5 * ties) / (final_wins + static_wins + ties)"
    )

    assert reproducibility["bootstrap"] == {
        "confidence_level": "0.950000",
        "replicates": 10000,
        "resampling_unit": "leakage_connected_component",
    }
    assert reproducibility["seeds"] == {
        "bootstrap_seed": 2026080601,
        "hidden_repeat_seed": 2026080605,
        "human_sample_seed": 2026080604,
        "pairwise_orientation_seed": 2026080603,
        "pairwise_sample_seed": 2026080602,
        "test_schedule_seed": 2026080607,
        "validation_partition_seed": 2026080606,
    }
    assert reproducibility["seeds_frozen_before_validation_or_test_scores"] is True

    assert unseal["state"] == "sealed"
    assert unseal["current_event_receipt"] is None
    assert unseal["single_use_create_only"] is True
    assert unseal["maximum_event_count"] == 1
    assert len(unseal["prerequisite_ids"]) == 10
    assert unseal["schedule"] == {
        "canonical_config_count": 5,
        "canonical_rows": 1500,
        "confirmatory_repeat_configs": ["llm_static", "frozen_final"],
        "confirmatory_repeat_rows": 600,
        "query_count": 300,
        "total_rows": 2100,
    }
    assert unseal["scores_hidden_until_all_scheduled_rows_terminal"] is True
    assert contract["core_execution_authorized"] is False
