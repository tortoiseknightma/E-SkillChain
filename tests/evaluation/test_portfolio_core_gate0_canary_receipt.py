from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import build_portfolio_core_gate0_canary_receipt as gate0
from skillchain.evaluation.portfolio_launch import MAIN_CONFIG_ORDER
from skillchain.evaluation.portfolio_parallel import (
    build_parallel_profile,
    write_parallel_profile,
)
from skillchain.evaluation.portfolio_execution import (
    ProviderPreResponseCircuitBreaker,
    create_retryable_attempt_receipt,
    forfeit_portfolio_provider_call,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
    make_portfolio_budget_call_identity,
    reserve_portfolio_provider_call,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_CAPABILITY_QUERY_COUNTS = dict(
    zip(gate0.CAPABILITIES, (5, 4, 4, 4, 4, 4), strict=True)
)
_RUN_MATRIX_SHA256 = sha256_bytes(
    (gate0.REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py").read_bytes()
)
_RUNTIME_ROUTER_FIELDS = (
    "shared_stage2_route_policy_version",
    "shared_stage2_route_schema_sha256",
    "portfolio_router_contract_version",
    "portfolio_router_contract_sha256",
    "portfolio_failure_policy_version",
    "portfolio_router_request_max_output_tokens",
    "portfolio_router_pricing_reservation_max_output_tokens",
)


def _gate0_freeze_binding(version: int) -> dict[str, object]:
    path = (
        gate0.REPOSITORY_ROOT
        / "specs"
        / "evaluation"
        / f"portfolio-core-gate0-freeze-v{version}.json"
    )
    content = path.read_bytes()
    return gate0._freeze_binding(path, sha256_bytes(content))


def _write_runtime_lock(
    root: Path,
    *,
    freeze_version: int = 3,
    router_overrides: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root.mkdir()
    router = (
        gate0._active_router_freeze_contract()
        if freeze_version == 3
        else gate0._legacy_v2_router_freeze_contract()
    )
    values = {field: router[field] for field in _RUNTIME_ROUTER_FIELDS}
    values.update(router_overrides or {})
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-assistant-runtime-lock",
        "treatment_chain_sha256": "c" * 64,
        **values,
    }
    if freeze_version == 3:
        unsigned.update(
            {
                "portfolio_budget_policy_version": (
                    "portfolio-call-hard-cap-v2"
                ),
                "final_judge_result_schema_version": 9,
                "final_judge_cache_namespace": "final-evaluator-v10",
            }
        )
    lock = {
        **unsigned,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    content = canonical_json_bytes(lock)
    (root / "runtime-lock.json").write_bytes(content)
    control = {
        "runtime_lock_file_sha256": sha256_bytes(content),
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "treatment_chain_sha256": "c" * 64,
    }
    return lock, control


def _analysis(
    *,
    assistant_error: tuple[str, str] | None = None,
    judge_error: tuple[str, str] | None = None,
) -> dict[str, object]:
    config_summary = []
    capability_summary = []
    for config in MAIN_CONFIG_ORDER:
        assistant_total = int(assistant_error is not None and assistant_error[0] == config)
        judge_total = int(judge_error is not None and judge_error[0] == config)
        config_summary.append(
            {
                "scope": "dev_mini",
                "config": config,
                "query_count": 25,
                "assistant_hard_error_count": assistant_total,
                "evaluator_anomaly_count": judge_total,
            }
        )
        for capability, count in _CAPABILITY_QUERY_COUNTS.items():
            assistant_count = int(
                assistant_error == (config, capability)
            )
            judge_count = int(judge_error == (config, capability))
            capability_summary.append(
                {
                    "scope": "dev_mini",
                    "config": config,
                    "canonical_capability": capability,
                    "query_count": count,
                    "assistant_hard_error_count": assistant_count,
                    "evaluator_anomaly_count": judge_count,
                    "judge_non_scored_count": judge_count,
                }
            )
    return {
        "kind": "portfolio-matrix-analysis",
        "dataset_profile": "core",
        "row_count": 125,
        "selected_query_count": 25,
        "config_order": list(MAIN_CONFIG_ORDER),
        "capabilities": list(gate0.CAPABILITIES),
        "config_summary": config_summary,
        "capability_summary": capability_summary,
        "model_calls_performed": 0,
    }


def _invariants(
    *,
    hard_error_passed: bool = True,
    execution_root: Path | None = None,
    parallel_profile: Path | None = None,
    freeze_version: int = 3,
) -> dict[str, object]:
    execution = (execution_root or Path.cwd()).resolve()
    profile = (parallel_profile or Path.cwd() / "parallel-profile.json").resolve()
    freeze = _gate0_freeze_binding(freeze_version)
    router = (
        gate0._active_router_freeze_contract()
        if freeze_version == 3
        else gate0._legacy_v2_router_freeze_contract()
    )
    rerun_argv = (
        str(Path(sys.executable).resolve()),
        str(
            (
                gate0.REPOSITORY_ROOT
                / "scripts"
                / "run_portfolio_matrix.py"
            ).resolve()
        ),
        "--execution-root",
        str(execution),
        "--execute",
        "--parallel-profile",
        str(profile),
    )
    return {
        "freeze": freeze,
        "execution": {"control_sha256": "b" * 64},
        "launch": {"plan_sha256": "c" * 64},
        "runtime": {
            "runtime_lock_sha256": "d" * 64,
            "router_contract": {
                **{
                    field: router[field]
                    for field in _RUNTIME_ROUTER_FIELDS
                },
                "exact_match_to_freeze": True,
            },
            **(
                {
                    "budget_forfeit_contract": {
                        "policy_version": "portfolio-call-hard-cap-v2",
                        "exact_match_to_freeze": True,
                    },
                    "final_judge_contract": {
                        "result_schema_version": 9,
                        "cache_namespace": "final-evaluator-v10",
                        "exact_match_to_freeze": True,
                    },
                }
                if freeze_version == 3
                else {}
            ),
        },
        "parallel_profile": {
            "path": str(profile),
            "file_sha256": "9" * 64,
            "profile_sha256": "8" * 64,
            "scheduler_file_sha256": _RUN_MATRIX_SHA256,
        },
        "rerun_command": {
            "working_directory": str(gate0.REPOSITORY_ROOT.resolve()),
            "argv": list(rerun_argv),
            "argv_sha256": sha256_bytes(canonical_json_bytes(list(rerun_argv))),
            "run_matrix_file_sha256": _RUN_MATRIX_SHA256,
        },
        "canary_geometry": {"batch_id": "core-dev-mini-001"},
        "full_alias": {
            "source_config": "s1s2",
            "target_config": "full",
            "provider_model_call_count": 0,
        },
        "audit": {"audit_sha256": "e" * 64},
        "analysis": {"analysis_sha256": "f" * 64},
        "hard_error_gate": {"passed": hard_error_passed},
        "provider_ledger": {
            **(
                {
                    "policy_version": "portfolio-call-hard-cap-v2",
                    "accounting_policy": (
                        "settled_actual_plus_forfeited_full_reserve_plus_"
                        "unresolved_reserve"
                    ),
                    "exception_policy": (
                        "append_full_reserve_forfeit_without_captured_response"
                    ),
                    "terminal_outcome_policy": (
                        "exactly_one_of_settlement_or_forfeit"
                    ),
                }
                if freeze_version == 3
                else {}
            ),
            "reservation_count": 100,
            "settlement_count": 100,
            **(
                {
                    "forfeit_count": 0,
                    "terminal_outcome_count": 100,
                    "unresolved_reservation_count": 0,
                    "launch_state_model_call_count": 100,
                    "forfeited_reserved_cost_cny": "0.000000000000",
                    "forfeit_counts_by_config": {},
                    "forfeit_counts_by_stage": {},
                    "forfeit_counts_by_reason": {},
                    "forfeit_attempt_receipt_binding": {
                        "attempt_policy_version": "portfolio-shard-attempt-v4",
                        "binding_fields": [
                            "forfeited_reservation_sha256",
                            "budget_forfeit_sha256",
                        ],
                        "bound_forfeit_count": 0,
                        "binding_set_sha256": sha256_bytes(
                            canonical_json_bytes([])
                        ),
                        "bindings": [],
                        "exact_bidirectional_binding": True,
                    },
                }
                if freeze_version == 3
                else {}
            ),
            "last_event_index": 200,
            "provider_call_identity_set_sha256": "1" * 64,
            "ledger_inventory": {
                "file_count": 201,
                "artifact_set_sha256": "7" * 64,
            },
        },
        "checkpoint_terminal_inventory": {
            "file_count": 210,
            "artifact_set_sha256": "2" * 64,
        },
    }


def _successful_rerun(
    snapshot: dict[str, object],
    snapshot_file_sha256: str,
    invariants: dict[str, object],
) -> dict[str, object]:
    return gate0._rerun_payload(
        snapshot=snapshot,
        snapshot_file_sha256=snapshot_file_sha256,
        pre_run_invariants=invariants,
        argv=invariants["rerun_command"]["argv"],
        started_at_utc="2026-08-07T00:00:00.000000Z",
        finished_at_utc="2026-08-07T00:00:00.100000Z",
        exit_code=0,
        stdout=b"completed\n",
        stderr=b"",
    )


def _forfeited_ledger_fixture(
    tmp_path: Path,
    *,
    forfeit_reason: str = "provider_call_ended_without_captured_response",
    failure_subtype: str = "provider_pre_response_provider_error",
    exception_type: str = "RateLimitError",
):
    execution_root = tmp_path / "execution"
    ledger_root = execution_root / "budget-ledger"
    shard = SimpleNamespace(
        shard_id="core-s1",
        config="s1",
        output_relpath="shards/core-s1",
        query_ids=("query-001",),
    )
    shard_root = execution_root / shard.output_relpath
    shard_root.mkdir(parents=True)
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id="core-matrix",
        phase_cap_cny=Decimal("30"),
        prior_observed_cost_cny=Decimal("0"),
    )
    identity = make_portfolio_budget_call_identity(
        matrix_run_id="core-matrix",
        shard_id=shard.shard_id,
        config=shard.config,
        query_id="query-001",
        instance_sha256="a" * 64,
        request_sha256="b" * 64,
        wire_request_sha256="c" * 64,
        stage="assistant_action",
        attempt_index=1,
        call_index=1,
    )
    reservation, _ = reserve_portfolio_provider_call(
        ledger_root,
        identity=identity,
    )
    forfeit, _ = forfeit_portfolio_provider_call(
        ledger_root,
        reservation_sha256=reservation.reservation_sha256,
        reason=forfeit_reason,
    )
    receipt, receipt_path = create_retryable_attempt_receipt(
        shard_root,
        breaker=ProviderPreResponseCircuitBreaker(),
        matrix_run_id="core-matrix",
        shard_id=shard.shard_id,
        config=shard.config,
        instance_sha256="a" * 64,
        query_id="query-001",
        query_ordinal=0,
        request_sha256="b" * 64,
        failure_stage="assistant_action",
        failure_subtype=failure_subtype,
        circuit_id="qwen:assistant_action",
        forfeited_reservation_sha256=reservation.reservation_sha256,
        budget_forfeit_sha256=forfeit.forfeit_sha256,
        exception_type=exception_type,
    )
    return execution_root, shard, reservation, forfeit, receipt, receipt_path


def test_hard_error_gate_passes_only_when_both_roles_are_zero() -> None:
    passed = gate0.build_hard_error_gate(_analysis())

    assert passed["passed"] is True
    assert passed["overall"]["assistant"]["rate"] == "0.000000000000"
    assert passed["overall"]["judge"]["rate"] == "0.000000000000"
    assert set(passed["by_capability"]) == set(gate0.CAPABILITIES)
    assert passed["all_six_capabilities_covered"] is True
    assert passed["any_single_error_fails_this_batch"] is True

    assistant_failed = gate0.build_hard_error_gate(
        _analysis(
            assistant_error=("noskill", "knowledge.visual_encyclopedia")
        )
    )
    judge_failed = gate0.build_hard_error_gate(
        _analysis(judge_error=("full", "utility.recipe_guidance"))
    )

    assert assistant_failed["passed"] is False
    assert assistant_failed["overall"]["assistant"] == {
        "hard_error_count": 1,
        "denominator": 125,
        "rate": "0.008000000000",
        "limit_exclusive": "0.005",
        "passed": False,
    }
    assert (
        assistant_failed["by_capability"]["knowledge.visual_encyclopedia"]
        ["assistant"]["passed"]
        is False
    )
    assert judge_failed["passed"] is False
    assert judge_failed["overall"]["judge"]["hard_error_count"] == 1


def test_hard_error_gate_rejects_missing_capability_or_judge_alias() -> None:
    missing = _analysis()
    missing["capability_summary"] = missing["capability_summary"][:-1]
    try:
        gate0.build_hard_error_gate(missing)
    except gate0.Gate0CanaryEvidenceError as error:
        assert "six capabilities" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("missing capability row was accepted")

    mismatched = _analysis()
    mismatched["capability_summary"][0]["judge_non_scored_count"] = 1
    try:
        gate0.build_hard_error_gate(mismatched)
    except gate0.Gate0CanaryEvidenceError as error:
        assert "Judge hard-error aliases" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("mismatched Judge aliases were accepted")


def test_parallel_profile_is_strictly_bound_to_gate0_freeze(
    tmp_path: Path,
) -> None:
    freeze_path = (
        gate0.REPOSITORY_ROOT
        / "specs"
        / "evaluation"
        / "portfolio-core-gate0-freeze-v1.json"
    )
    freeze_content = freeze_path.read_bytes()
    freeze = gate0._freeze_binding(freeze_path, sha256_bytes(freeze_content))
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id="core-canary",
            launch_plan_sha256="a" * 64,
        )
    )
    profile_path = tmp_path / "parallel.json"
    write_parallel_profile(
        profile_path,
        build_parallel_profile(
            matrix_run_id="core-canary",
            launch_plan_sha256="a" * 64,
            execution_scope="core_canary",
            assistant_concurrency=2,
            final_judge_concurrency=8,
            qwen_requests_per_minute_cap=40,
        ),
    )
    profile_content = profile_path.read_bytes()

    binding = gate0._parallel_profile_binding(
        profile_path,
        expected_file_sha256=sha256_bytes(profile_content),
        launch=launch,
        freeze=freeze,
    )

    assert binding["assistant_concurrency"] == 2
    assert binding["final_judge_concurrency"] == 8
    assert binding["qwen_requests_per_minute_cap"] == 40
    assert binding["scheduler_file_sha256"] == _RUN_MATRIX_SHA256


def test_parallel_profile_binds_v3_reduced_provider_concurrency(
    tmp_path: Path,
) -> None:
    freeze = _gate0_freeze_binding(3)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id="core-canary-v3",
            launch_plan_sha256="a" * 64,
        )
    )
    profile_path = tmp_path / "parallel-v3.json"
    write_parallel_profile(
        profile_path,
        build_parallel_profile(
            matrix_run_id="core-canary-v3",
            launch_plan_sha256="a" * 64,
            execution_scope="core_canary",
            assistant_concurrency=1,
            final_judge_concurrency=2,
            qwen_requests_per_minute_cap=20,
        ),
    )
    content = profile_path.read_bytes()

    binding = gate0._parallel_profile_binding(
        profile_path,
        expected_file_sha256=sha256_bytes(content),
        launch=launch,
        freeze=freeze,
    )

    assert binding["assistant_concurrency"] == 1
    assert binding["final_judge_concurrency"] == 2
    assert binding["qwen_requests_per_minute_cap"] == 20


def test_freeze_binding_supports_v1_v2_and_active_forfeit_v3() -> None:
    legacy = _gate0_freeze_binding(1)
    previous = _gate0_freeze_binding(2)
    current = _gate0_freeze_binding(3)

    assert legacy["policy_version"] == "portfolio-core-gate0-freeze-v1"
    assert "router_contract" not in legacy
    assert previous["policy_version"] == "portfolio-core-gate0-freeze-v2"
    assert previous["router_contract"] == (
        gate0._legacy_v2_router_freeze_contract()
    )
    assert current["policy_version"] == "portfolio-core-gate0-freeze-v3"
    assert current["router_contract"] == gate0._active_router_freeze_contract()
    assert current["route_format_retry"] == gate0._active_route_retry_freeze()
    assert current["budget_forfeit_contract"] == gate0._V3_BUDGET_FREEZE
    assert current["final_judge_contract"] == gate0._V3_FINAL_JUDGE_FREEZE
    assert current["assistant_concurrency"] == 1
    assert current["final_judge_concurrency"] == 2
    assert current["qwen_requests_per_minute_cap"] == 20


def test_v2_runtime_lock_binding_proves_exact_router_contract(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    _lock, control = _write_runtime_lock(runtime_root, freeze_version=2)
    freeze = _gate0_freeze_binding(2)

    binding = gate0._runtime_lock_binding(
        runtime_root=runtime_root,
        control=control,
        freeze=freeze,
    )

    assert binding["runtime_lock_sha256"] == control["runtime_lock_sha256"]
    assert binding["router_contract"] == {
        **{
            field: freeze["router_contract"][field]
            for field in _RUNTIME_ROUTER_FIELDS
        },
        "exact_match_to_freeze": True,
    }


def test_v3_runtime_lock_binds_attempt_budget_and_final_judge(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    _lock, control = _write_runtime_lock(runtime_root, freeze_version=3)
    freeze = _gate0_freeze_binding(3)

    binding = gate0._runtime_lock_binding(
        runtime_root=runtime_root,
        control=control,
        freeze=freeze,
    )

    assert binding["router_contract"]["portfolio_failure_policy_version"] == (
        "portfolio-shard-attempt-v4"
    )
    assert binding["budget_forfeit_contract"] == {
        "policy_version": "portfolio-call-hard-cap-v2",
        "exact_match_to_freeze": True,
    }
    assert binding["final_judge_contract"] == {
        "result_schema_version": 9,
        "cache_namespace": "final-evaluator-v10",
        "exact_match_to_freeze": True,
    }


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("portfolio_budget_policy_version", "portfolio-call-hard-cap-v1"),
        ("final_judge_result_schema_version", 8),
        ("final_judge_cache_namespace", "final-evaluator-v9"),
    ],
)
def test_v3_runtime_lock_rejects_budget_or_judge_drift(
    tmp_path: Path,
    field: str,
    drifted: object,
) -> None:
    runtime_root = tmp_path / "runtime"
    lock, _control = _write_runtime_lock(runtime_root, freeze_version=3)
    lock[field] = drifted
    lock.pop("runtime_lock_sha256")
    lock["runtime_lock_sha256"] = sha256_bytes(canonical_json_bytes(lock))
    content = canonical_json_bytes(lock)
    (runtime_root / "runtime-lock.json").write_bytes(content)
    control = {
        "runtime_lock_file_sha256": sha256_bytes(content),
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "treatment_chain_sha256": "c" * 64,
    }

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="budget or final-Judge contract differs",
    ):
        gate0._runtime_lock_binding(
            runtime_root=runtime_root,
            control=control,
            freeze=_gate0_freeze_binding(3),
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("shared_stage2_route_policy_version", "shared-stage2-route-v5"),
        ("shared_stage2_route_schema_sha256", "1" * 64),
        ("portfolio_router_contract_version", "portfolio-assistant-router-v5"),
        ("portfolio_router_contract_sha256", "0" * 64),
        ("portfolio_failure_policy_version", "portfolio-shard-attempt-v2"),
        ("portfolio_router_request_max_output_tokens", 512),
        ("portfolio_router_pricing_reservation_max_output_tokens", 64),
    ],
)
def test_v2_runtime_lock_rejects_each_router_binding_drift(
    tmp_path: Path,
    field: str,
    drifted: object,
) -> None:
    runtime_root = tmp_path / "runtime"
    _lock, control = _write_runtime_lock(
        runtime_root,
        freeze_version=2,
        router_overrides={field: drifted},
    )

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="Router contract differs",
    ):
        gate0._runtime_lock_binding(
            runtime_root=runtime_root,
            control=control,
            freeze=_gate0_freeze_binding(2),
        )


def test_evidence_v2_rejects_v1_freeze_or_incomplete_runtime_proof() -> None:
    legacy_freeze = _invariants(freeze_version=2)
    legacy_freeze["freeze"] = _gate0_freeze_binding(1)
    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="evidence-v2 requires Gate0 freeze-v2",
    ):
        gate0._require_v2_evidence_bindings(legacy_freeze)

    incomplete_runtime = _invariants(freeze_version=2)
    incomplete_runtime["runtime"] = {
        **incomplete_runtime["runtime"],
        "router_contract": {
            "exact_match_to_freeze": True,
        },
    }
    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="proof is incomplete or drifted",
    ):
        gate0._require_v2_evidence_bindings(incomplete_runtime)


def test_evidence_v3_is_default_while_v1_and_v2_snapshots_remain_readable(
    tmp_path: Path,
) -> None:
    invariants = _invariants()
    current = gate0._snapshot_payload(invariants)
    assert gate0.POLICY_VERSION == "portfolio-core-gate0-canary-evidence-v3"
    assert current["policy_version"] == gate0.POLICY_VERSION

    for version, policy, legacy_invariants in (
        (1, gate0.LEGACY_POLICY_VERSION, invariants),
        (2, gate0.PREVIOUS_POLICY_VERSION, _invariants(freeze_version=2)),
    ):
        legacy_unsigned = {
            **current,
            "policy_version": policy,
            "invariants": legacy_invariants,
        }
        legacy_unsigned.pop("snapshot_sha256")
        legacy = {
            **legacy_unsigned,
            "snapshot_sha256": gate0._hash_payload(legacy_unsigned),
        }
        path = tmp_path / f"legacy-v{version}-snapshot.json"
        content = canonical_json_bytes(legacy)
        path.write_bytes(content)

        loaded, loaded_content = gate0._load_snapshot(path, sha256_bytes(content))
        assert loaded["policy_version"] == policy
        assert loaded_content == content


@pytest.mark.parametrize(
    ("freeze_version", "evidence_policy"),
    [
        (1, gate0.LEGACY_POLICY_VERSION),
        (2, gate0.PREVIOUS_POLICY_VERSION),
    ],
)
def test_v1_and_v2_evidence_remain_rerun_verifiable(
    freeze_version: int,
    evidence_policy: str,
) -> None:
    invariants = _invariants(freeze_version=freeze_version)
    snapshot = gate0._content_address(
        {
            "schema_version": 1,
            "kind": gate0.SNAPSHOT_KIND,
            "policy_version": evidence_policy,
            "track": "portfolio",
            "formal_eligible": False,
            "phase": "pre_rerun",
            "status": "captured",
            "invariants": invariants,
            "hard_error_gate_passed": True,
            "model_calls_performed": 0,
        },
        field="snapshot_sha256",
    )
    snapshot_file_sha = sha256_bytes(canonical_json_bytes(snapshot))
    rerun = _successful_rerun(snapshot, snapshot_file_sha, invariants)
    receipt = gate0.build_final_receipt(
        snapshot=snapshot,
        snapshot_file_sha256=snapshot_file_sha,
        rerun_receipt=rerun,
        rerun_receipt_file_sha256=sha256_bytes(canonical_json_bytes(rerun)),
        current_invariants=invariants,
    )

    assert receipt["policy_version"] == evidence_policy
    assert receipt["gate0_canary_passed"] is True
    assert receipt["rerun"]["forfeit_count_delta"] == 0


def test_full_alias_binding_requires_physical_s1s2_zero_call_source() -> None:
    source = SimpleNamespace(config="s1s2", shard_id="core-s1s2")
    target = SimpleNamespace(config="full", shard_id="core-full")
    alias = {
        "alias_sha256": "a" * 64,
        "target_config": "full",
        "source_config": "s1s2",
        "stage_decision": "rolled_back",
        "provider_model_call_count": 0,
        "source_bank_sha256": "b" * 64,
        "target_bank_sha256": "b" * 64,
        "source_bank_file_sha256": "c" * 64,
        "target_bank_file_sha256": "c" * 64,
    }
    control = {
        "execution_artifact_aliases": [alias],
        "execution_artifact_alias_policy_version": (
            "portfolio-execution-artifact-alias-v1"
        ),
        "execution_artifact_alias_provider_model_call_count": 0,
    }
    audit = {
        "completion": {
            "full": {
                "artifact_source": "accepted_parent_artifact_alias",
                "physical_shard_id": "core-s1s2",
                "assistant_checkpoint_count": 25,
                "final_checkpoint_count": 25,
            }
        },
        "bindings": {
            "internal_artifact_aliases": [
                {
                    "source_config": "s1s2",
                    "target_config": "full",
                    "provider_model_call_count": 0,
                    "alias_receipt_sha256": "d" * 64,
                }
            ]
        },
    }

    binding = gate0._full_alias_binding(
        control=control,
        audit=audit,
        selected_shards=(source, target),
    )

    assert binding["source_shard_id"] == "core-s1s2"
    assert binding["target_shard_id"] == "core-full"
    assert binding["provider_model_call_count"] == 0


def test_ledger_binding_rejects_missing_control_matrix_run_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ledger = SimpleNamespace(
        authority=SimpleNamespace(
            matrix_run_id="core-matrix",
            phase_cap_cny=Decimal("30.000000000000"),
            authority_sha256="a" * 64,
            policy_version="portfolio-call-hard-cap-v2",
            accounting_policy=(
                "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
            ),
            exception_policy=(
                "append_full_reserve_forfeit_without_captured_response"
            ),
        ),
        unresolved_reservations=(),
        reservations=(),
        settlements=(),
        forfeits=(),
        last_event_index=0,
        last_event_sha256=None,
        settled_actual_cost_cny=Decimal("0.000000000000"),
        forfeited_reserved_cost_cny=Decimal("0.000000000000"),
        accountable_cost_cny=Decimal("0.000000000000"),
    )
    monkeypatch.setattr(gate0, "load_portfolio_budget_ledger", lambda _root: ledger)
    monkeypatch.setattr(
        gate0,
        "_stable_directory_inventory",
        lambda *_args, **_kwargs: {
            "file_count": 1,
            "files": [],
            "artifact_set_sha256": "b" * 64,
        },
    )

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="differs from authority",
    ):
        gate0._ledger_binding(
            execution_root=tmp_path,
            control={
                "budget_ledger_relpath": "budget-ledger",
                "phase_cumulative_cap_cny": "30.000000000000",
            },
            selected_shards=(SimpleNamespace(shard_id="s1"),),
            expected_model_call_count=1,
        )

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="differs from authority",
    ):
        gate0._ledger_binding(
            execution_root=tmp_path,
            control={
                "matrix_run_id": "core-matrix",
                "budget_ledger_relpath": "budget-ledger",
                "phase_cumulative_cap_cny": "30.000000000000",
            },
            selected_shards=(SimpleNamespace(shard_id="s1"),),
            expected_model_call_count=1,
        )


def test_ledger_binding_counts_forfeit_and_proves_attempt_v4_pair(
    tmp_path: Path,
) -> None:
    execution_root, shard, reservation, forfeit, receipt, _ = (
        _forfeited_ledger_fixture(tmp_path)
    )

    binding = gate0._ledger_binding(
        execution_root=execution_root,
        control={
            "matrix_run_id": "core-matrix",
            "budget_ledger_relpath": "budget-ledger",
            "phase_cumulative_cap_cny": "30.000000000000",
        },
        selected_shards=(shard,),
        expected_model_call_count=1,
    )

    assert binding["reservation_count"] == 1
    assert binding["settlement_count"] == 0
    assert binding["forfeit_count"] == 1
    assert binding["terminal_outcome_count"] == 1
    assert binding["unresolved_reservation_count"] == 0
    assert binding["forfeited_reserved_cost_cny"] == format(
        reservation.reserved_cost_cny,
        "f",
    )
    assert binding["forfeit_counts_by_config"] == {"s1": 1}
    assert binding["forfeit_counts_by_stage"] == {"assistant_action": 1}
    assert binding["forfeit_counts_by_reason"] == {
        "provider_call_ended_without_captured_response": 1
    }
    proof = binding["forfeit_attempt_receipt_binding"]
    assert proof["bound_forfeit_count"] == 1
    assert proof["exact_bidirectional_binding"] is True
    assert proof["bindings"] == [
        {
            "reservation_sha256": reservation.reservation_sha256,
            "forfeit_sha256": forfeit.forfeit_sha256,
            "attempt_receipt_sha256": receipt.receipt_sha256,
            "config": "s1",
            "stage": "assistant_action",
            "reason": "provider_call_ended_without_captured_response",
        }
    ]


def test_forfeit_binding_rejects_receipt_pointing_to_unknown_forfeit(
    tmp_path: Path,
) -> None:
    execution_root, shard, _reservation, _forfeit, _receipt, receipt_path = (
        _forfeited_ledger_fixture(tmp_path)
    )
    raw = json.loads(receipt_path.read_bytes())
    raw["budget_forfeit_sha256"] = "d" * 64
    raw.pop("receipt_sha256")
    raw["receipt_sha256"] = sha256_bytes(canonical_json_bytes(raw))
    receipt_path.write_bytes(canonical_json_bytes(raw))

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="unknown reservation or forfeit",
    ):
        gate0._forfeit_attempt_receipt_binding(
            execution_root=execution_root,
            selected_shards=(shard,),
            ledger=load_portfolio_budget_ledger(
                execution_root / "budget-ledger"
            ),
        )


def test_forfeit_binding_accepts_orphan_receipt_for_existing_generic_forfeit(
    tmp_path: Path,
) -> None:
    execution_root, shard, reservation, forfeit, receipt, _ = (
        _forfeited_ledger_fixture(
            tmp_path,
            failure_subtype="orphaned_provider_call",
            exception_type="PortfolioBudgetOrphanedCallError",
        )
    )

    proof = gate0._forfeit_attempt_receipt_binding(
        execution_root=execution_root,
        selected_shards=(shard,),
        ledger=load_portfolio_budget_ledger(execution_root / "budget-ledger"),
    )

    assert proof["bound_forfeit_count"] == 1
    assert proof["bindings"] == [
        {
            "reservation_sha256": reservation.reservation_sha256,
            "forfeit_sha256": forfeit.forfeit_sha256,
            "attempt_receipt_sha256": receipt.receipt_sha256,
            "config": "s1",
            "stage": "assistant_action",
            "reason": "provider_call_ended_without_captured_response",
        }
    ]


def test_forfeit_binding_rejects_provider_pre_response_receipt_for_orphan_reason(
    tmp_path: Path,
) -> None:
    execution_root, shard, *_ = _forfeited_ledger_fixture(
        tmp_path,
        forfeit_reason="orphan_recovered_after_owner_exit",
    )

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="attempt receipt and budget forfeit call identity differ",
    ):
        gate0._forfeit_attempt_receipt_binding(
            execution_root=execution_root,
            selected_shards=(shard,),
            ledger=load_portfolio_budget_ledger(
                execution_root / "budget-ledger"
            ),
        )


def test_external_output_check_normalizes_parent_traversal(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    disguised_internal = tmp_path / "other" / ".." / "execution" / "receipt.json"

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="outside execution",
    ):
        gate0._require_external_output(
            disguised_internal,
            scoped_roots=(execution_root,),
        )


def test_final_receipt_proves_zero_delta_and_fails_any_drift() -> None:
    invariants = _invariants()
    snapshot = gate0._snapshot_payload(invariants)
    snapshot_file_sha = sha256_bytes(canonical_json_bytes(snapshot))
    rerun = _successful_rerun(snapshot, snapshot_file_sha, invariants)
    rerun_file_sha = sha256_bytes(canonical_json_bytes(rerun))

    passed = gate0.build_final_receipt(
        snapshot=snapshot,
        snapshot_file_sha256=snapshot_file_sha,
        rerun_receipt=rerun,
        rerun_receipt_file_sha256=rerun_file_sha,
        current_invariants=invariants,
    )

    assert passed["status"] == "passed"
    assert passed["gate0_canary_passed"] is True
    assert passed["policy_version"] == "portfolio-core-gate0-canary-evidence-v3"
    assert passed["bindings"]["runtime"]["router_contract"][
        "exact_match_to_freeze"
    ] is True
    assert passed["rerun"]["provider_call_count_delta"] == 0
    assert passed["rerun"]["settlement_count_delta"] == 0
    assert passed["rerun"]["forfeit_count_delta"] == 0
    assert passed["rerun"]["terminal_outcome_count_delta"] == 0
    assert passed["rerun"]["ledger_event_count_delta"] == 0
    assert passed["rerun"]["ledger_inventory_unchanged"] is True
    assert passed["rerun"]["zero_incremental_provider_calls_proved"] is True
    assert (
        passed["rerun"]["checkpoint_and_terminal_artifact_bytes_unchanged"]
        is True
    )

    ledger_drift = _invariants()
    ledger_drift["provider_ledger"] = {
        **ledger_drift["provider_ledger"],
        "reservation_count": 101,
        "settlement_count": 101,
        "terminal_outcome_count": 101,
        "launch_state_model_call_count": 101,
        "last_event_index": 202,
        "provider_call_identity_set_sha256": "3" * 64,
        "ledger_inventory": {
            "file_count": 203,
            "artifact_set_sha256": "4" * 64,
        },
    }
    failed = gate0.build_final_receipt(
        snapshot=snapshot,
        snapshot_file_sha256=snapshot_file_sha,
        rerun_receipt=rerun,
        rerun_receipt_file_sha256=rerun_file_sha,
        current_invariants=ledger_drift,
    )

    assert failed["status"] == "failed_rerun_drift"
    assert failed["gate0_canary_passed"] is False
    assert failed["rerun"]["provider_call_count_delta"] == 1
    assert failed["rerun"]["ledger_event_count_delta"] == 2
    assert "provider_ledger" in failed["rerun"]["changed_invariant_sections"]


def test_final_receipt_explicitly_rejects_incremental_forfeit() -> None:
    invariants = _invariants()
    snapshot = gate0._snapshot_payload(invariants)
    snapshot_file_sha = sha256_bytes(canonical_json_bytes(snapshot))
    rerun = _successful_rerun(snapshot, snapshot_file_sha, invariants)
    rerun_file_sha = sha256_bytes(canonical_json_bytes(rerun))
    drifted = _invariants()
    new_forfeit_bindings = [
        {
            "reservation_sha256": "4" * 64,
            "forfeit_sha256": "5" * 64,
            "attempt_receipt_sha256": "6" * 64,
            "config": "s1",
            "stage": "assistant_action",
            "reason": "provider_call_ended_without_captured_response",
        }
    ]
    drifted["provider_ledger"] = {
        **drifted["provider_ledger"],
        "reservation_count": 101,
        "forfeit_count": 1,
        "terminal_outcome_count": 101,
        "launch_state_model_call_count": 101,
        "forfeited_reserved_cost_cny": "0.179404800000",
        "forfeit_counts_by_config": {"s1": 1},
        "forfeit_counts_by_stage": {"assistant_action": 1},
        "forfeit_counts_by_reason": {
            "provider_call_ended_without_captured_response": 1
        },
        "last_event_index": 202,
        "provider_call_identity_set_sha256": "3" * 64,
        "forfeit_attempt_receipt_binding": {
            "attempt_policy_version": "portfolio-shard-attempt-v4",
            "binding_fields": [
                "forfeited_reservation_sha256",
                "budget_forfeit_sha256",
            ],
            "bound_forfeit_count": 1,
            "binding_set_sha256": sha256_bytes(
                canonical_json_bytes(new_forfeit_bindings)
            ),
            "bindings": new_forfeit_bindings,
            "exact_bidirectional_binding": True,
        },
        "ledger_inventory": {
            "file_count": 203,
            "artifact_set_sha256": "4" * 64,
        },
    }

    failed = gate0.build_final_receipt(
        snapshot=snapshot,
        snapshot_file_sha256=snapshot_file_sha,
        rerun_receipt=rerun,
        rerun_receipt_file_sha256=rerun_file_sha,
        current_invariants=drifted,
    )

    assert failed["status"] == "failed_rerun_drift"
    assert failed["rerun"]["reservation_count_delta"] == 1
    assert failed["rerun"]["settlement_count_delta"] == 0
    assert failed["rerun"]["forfeit_count_delta"] == 1
    assert failed["rerun"]["terminal_outcome_count_delta"] == 1
    assert failed["rerun"]["ledger_event_count_delta"] == 2
    assert failed["rerun"]["ledger_inventory_unchanged"] is False
    assert failed["rerun"]["zero_incremental_provider_calls_proved"] is False


def test_failed_frozen_command_receipt_cannot_be_verified(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    parallel_profile = tmp_path / "parallel.json"
    parallel_profile.write_bytes(b"{}")
    invariants = _invariants(
        execution_root=execution_root,
        parallel_profile=parallel_profile,
    )
    snapshot = gate0._snapshot_payload(invariants)
    snapshot_content = canonical_json_bytes(snapshot)
    failed_rerun = gate0._rerun_payload(
        snapshot=snapshot,
        snapshot_file_sha256=sha256_bytes(snapshot_content),
        pre_run_invariants=invariants,
        argv=invariants["rerun_command"]["argv"],
        started_at_utc="2026-08-07T00:00:00.000000Z",
        finished_at_utc="2026-08-07T00:00:00.100000Z",
        exit_code=5,
        stdout=b"",
        stderr=b"recoverable stop",
    )
    receipt_path = tmp_path / "failed-rerun.json"
    receipt_content = canonical_json_bytes(failed_rerun)
    receipt_path.write_bytes(receipt_content)

    with pytest.raises(
        gate0.Gate0CanaryEvidenceError,
        match="invalid or command-drifted",
    ):
        gate0._load_rerun_receipt(
            receipt_path,
            sha256_bytes(receipt_content),
            snapshot=snapshot,
            snapshot_file_sha256=sha256_bytes(snapshot_content),
            execution_root=execution_root,
            parallel_profile_path=parallel_profile,
        )


def test_cli_requires_create_only_frozen_rerun_before_verify(
    monkeypatch,
    tmp_path: Path,
) -> None:
    roots = tuple(tmp_path / name for name in ("execution", "launch", "runtime"))
    for root in roots:
        root.mkdir()
    parallel_profile = tmp_path / "parallel-profile.json"
    parallel_profile.write_bytes(canonical_json_bytes({"fixture": True}))
    invariants = _invariants(
        execution_root=roots[0],
        parallel_profile=parallel_profile,
    )
    monkeypatch.setattr(
        gate0,
        "_collect_canary_state",
        lambda **_kwargs: (invariants, roots),
    )
    child_calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        child_calls.append(tuple(argv))
        assert kwargs["cwd"] == gate0.REPOSITORY_ROOT.resolve()
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout=b"complete\n", stderr=b"")

    monkeypatch.setattr(gate0.subprocess, "run", fake_run)
    output = tmp_path / "evidence" / "snapshot.json"
    common = [
        "--execution-root",
        str(roots[0]),
        "--freeze",
        str(tmp_path / "freeze.json"),
        "--freeze-file-sha256",
        "a" * 64,
        "--parallel-profile",
        str(parallel_profile),
        "--parallel-profile-file-sha256",
        sha256_bytes(parallel_profile.read_bytes()),
    ]

    assert gate0.main(["snapshot", *common, "--output", str(output)]) == 0
    snapshot_content = output.read_bytes()
    snapshot = json.loads(snapshot_content)
    assert snapshot["kind"] == gate0.SNAPSHOT_KIND
    assert snapshot["model_calls_performed"] == 0
    assert gate0.main(["snapshot", *common, "--output", str(output)]) == 2
    assert output.read_bytes() == snapshot_content

    rerun_output = tmp_path / "evidence" / "rerun.json"
    rerun_args = [
        "rerun",
        *common,
        "--output",
        str(rerun_output),
        "--snapshot",
        str(output),
        "--snapshot-file-sha256",
        sha256_bytes(snapshot_content),
    ]
    assert gate0.main(rerun_args) == 0
    assert len(child_calls) == 1
    assert child_calls[0] == tuple(invariants["rerun_command"]["argv"])
    rerun_content = rerun_output.read_bytes()
    rerun = json.loads(rerun_content)
    assert rerun["kind"] == gate0.RERUN_KIND
    assert rerun["status"] == "completed_success"
    assert rerun["command"]["exit_code"] == 0
    assert "DASHSCOPE_API_KEY" not in rerun_content.decode("utf-8")
    assert gate0.main(rerun_args) == 2
    assert len(child_calls) == 1

    final_output = tmp_path / "evidence" / "final.json"
    verify_args = [
        "verify",
        *common,
        "--output",
        str(final_output),
        "--snapshot",
        str(output),
        "--snapshot-file-sha256",
        sha256_bytes(snapshot_content),
        "--rerun-receipt",
        str(rerun_output),
        "--rerun-receipt-file-sha256",
        sha256_bytes(rerun_content),
    ]
    assert gate0.main(verify_args) == 0
    final = json.loads(final_output.read_bytes())
    assert final["kind"] == gate0.RECEIPT_KIND
    assert final["gate0_canary_passed"] is True
    assert final["pre_rerun_snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert final["rerun"]["frozen_command_invocation_proved"] is True


def test_verify_parser_cannot_omit_rerun_receipt() -> None:
    with pytest.raises(SystemExit):
        gate0.build_parser().parse_args(
            [
                "verify",
                "--execution-root",
                "execution",
                "--freeze",
                "freeze.json",
                "--freeze-file-sha256",
                "a" * 64,
                "--parallel-profile",
                "parallel.json",
                "--parallel-profile-file-sha256",
                "b" * 64,
                "--output",
                "final.json",
                "--snapshot",
                "snapshot.json",
                "--snapshot-file-sha256",
                "c" * 64,
            ]
        )
