from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.prepare_core_final_evaluation import main as prepare_main
from skillchain.evaluation.core_final_config import (
    CAPABILITY_ORDER,
    MAIN_CONFIG_ORDER,
    CoreFinalEvaluationFramework,
    CoreFinalFrameworkError,
    bind_framework_artifact,
    build_core_final_evaluation_framework,
    create_core_final_framework_artifact,
    load_core_final_framework_artifact,
    resolve_framework_slot,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _artifact(tmp_path, name: str, content: bytes = b"evidence\n"):
    path = tmp_path / name
    path.write_bytes(content)
    return path, sha256_bytes(content)


def test_default_framework_fixes_core_geometry_and_performs_zero_calls() -> None:
    framework = build_core_final_evaluation_framework()

    assert framework.scope.query_count == 1500
    assert framework.scope.canonical_instance_count == 7500
    assert framework.scope.batch_count == 60
    assert framework.scope.config_shard_count == 300
    assert framework.scope.capabilities == CAPABILITY_ORDER
    assert framework.scope.config_order == MAIN_CONFIG_ORDER
    assert framework.execution_authorized is False
    assert framework.model_calls_performed == 0
    assert framework.provisional_operational_baseline.assistant_concurrency == 2
    assert (
        framework.provisional_operational_baseline.assistant_rate_limit_policy
        == "smooth_start_v1"
    )
    assert (
        framework.provisional_operational_baseline.assistant_minimum_start_interval_seconds
        == 1.5
    )
    assert framework.provisional_operational_baseline.final_judge_concurrency == 8
    assert framework.provisional_operational_baseline.phase_budget_candidate_cny == 1100
    assert framework.tool_evaluation.canonical_tool_count == 7
    assert framework.tool_evaluation.gold_cases_per_tool_range == (30, 50)
    assert framework.evaluator_reliability.judge_human_audit_sample_range == (
        50,
        100,
    )
    assert framework.status == "draft_waiting_for_dev_mini_diagnostics"
    assert "binding:dev_mini_200x5_completion" in framework.blockers
    assert "refinement:parallel_execution_profile" in framework.blockers
    assert all("challenge_final_seal" not in blocker for blocker in framework.blockers)


def test_framework_rejects_fixed_config_drift_and_self_hash_tampering() -> None:
    framework = build_core_final_evaluation_framework()
    payload = framework.model_dump(mode="json")
    drifted = deepcopy(payload)
    drifted["scope"]["config_order"] = list(reversed(MAIN_CONFIG_ORDER))
    with pytest.raises(ValidationError, match="five-config order"):
        CoreFinalEvaluationFramework.model_validate_json(canonical_json_bytes(drifted))

    tampered = deepcopy(payload)
    tampered["human_evaluation"]["provisional_sample_size"] = 100
    with pytest.raises(ValidationError):
        CoreFinalEvaluationFramework.model_validate_json(canonical_json_bytes(tampered))

    policy_drift = deepcopy(payload)
    policy_drift["artifact_bindings"][0]["required_for_freeze"] = False
    unsigned = {
        key: value for key, value in policy_drift.items() if key != "framework_sha256"
    }
    policy_drift["framework_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    with pytest.raises(ValidationError, match="binding definitions"):
        CoreFinalEvaluationFramework.model_validate_json(
            canonical_json_bytes(policy_drift)
        )


def test_binding_is_hash_checked_and_changes_only_its_blocker(tmp_path) -> None:
    framework = build_core_final_evaluation_framework()
    path, digest = _artifact(tmp_path, "completion.json")

    with pytest.raises(CoreFinalFrameworkError, match="digest mismatch"):
        bind_framework_artifact(
            framework,
            role="dev_mini_200x5_completion",
            path=path,
            expected_sha256="0" * 64,
            repository_root=tmp_path,
        )

    bound = bind_framework_artifact(
        framework,
        role="dev_mini_200x5_completion",
        path=path,
        expected_sha256=digest,
        repository_root=tmp_path,
    )
    binding = next(
        item
        for item in bound.artifact_bindings
        if item.role == "dev_mini_200x5_completion"
    )
    assert binding.status == "verified"
    assert binding.artifact_root_id == "repository"
    assert binding.path == "completion.json"
    assert "binding:dev_mini_200x5_completion" not in bound.blockers
    assert len(bound.blockers) == len(framework.blockers) - 1
    assert bound.status == "draft_pending_freeze_inputs"
    assert bound.framework_sha256 != framework.framework_sha256


def test_refinement_requires_bound_evidence_and_records_decision(tmp_path) -> None:
    framework = build_core_final_evaluation_framework()
    decision, decision_sha = _artifact(tmp_path, "runtime-decision.json", b"decision\n")
    with pytest.raises(CoreFinalFrameworkError, match="required evidence is pending"):
        resolve_framework_slot(
            framework,
            slot_id="diagnostic_runtime_patch_set",
            decision_artifact_path=decision,
            expected_sha256=decision_sha,
            repository_root=tmp_path,
        )

    for role in (
        "dev_mini_200x5_completion",
        "dev_mini_diagnostic_decision",
        "judge_human_audit",
    ):
        path, digest = _artifact(tmp_path, f"{role}.json", role.encode())
        framework = bind_framework_artifact(
            framework,
            role=role,
            path=path,
            expected_sha256=digest,
            repository_root=tmp_path,
        )

    resolved = resolve_framework_slot(
        framework,
        slot_id="diagnostic_runtime_patch_set",
        decision_artifact_path=decision,
        expected_sha256=decision_sha,
        repository_root=tmp_path,
    )
    slot = next(
        item
        for item in resolved.refinement_slots
        if item.slot_id == "diagnostic_runtime_patch_set"
    )
    assert slot.status == "resolved"
    assert slot.decision_artifact_root_id == "repository"
    assert slot.decision_artifact_path == "runtime-decision.json"
    assert "refinement:diagnostic_runtime_patch_set" not in resolved.blockers


def test_framework_artifact_is_canonical_loadable_and_create_only(tmp_path) -> None:
    framework = build_core_final_evaluation_framework()
    output = tmp_path / "core-final-framework.json"
    create_core_final_framework_artifact(framework, output)

    loaded = load_core_final_framework_artifact(output)
    assert loaded == framework
    assert output.read_bytes() == framework.canonical_bytes()
    with pytest.raises(FileExistsError):
        create_core_final_framework_artifact(framework, output)


def test_cli_can_emit_draft_but_readiness_check_writes_nothing(tmp_path) -> None:
    draft = tmp_path / "draft.json"
    assert prepare_main(["--output", str(draft)]) == 0
    assert load_core_final_framework_artifact(draft).model_calls_performed == 0

    blocked = tmp_path / "blocked.json"
    assert prepare_main(["--output", str(blocked), "--require-ready-to-freeze"]) == 3
    assert not blocked.exists()


def test_tracked_candidate_binds_verified_core_data() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    tracked = load_core_final_framework_artifact(
        repository_root
        / "specs"
        / "evaluation"
        / "core-final-evaluation-framework-v1.json"
    )
    verified = {
        item.role: item
        for item in tracked.artifact_bindings
        if item.status == "verified"
    }
    assert set(verified) == {
        "core_query_plan",
        "core_query_plan_manifest",
        "core_queries",
        "core_corpus_owner_audit_approval",
        "core_capability_assignments",
        "core_asset_catalog_manifest",
        "core_asset_catalog_assets",
        "core_asset_catalog_components",
    }
    assert {item.artifact_root_id for item in verified.values()} == {"skillchain-data"}
    assert tracked.status == "draft_waiting_for_dev_mini_diagnostics"
    assert len(tracked.blockers) == 29
    assert tracked.model_calls_performed == 0
