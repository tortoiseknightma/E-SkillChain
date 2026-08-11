from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from scripts import prepare_portfolio_launch as prepare_launch_script

from skillchain.evaluation.assistant_runs import build_assistant_query_input
from skillchain.evaluation.portfolio_core_inputs import (
    CORE_BATCH_SIZE,
    CORE_QUERY_COUNT,
    CORE_SPLIT_COUNTS,
    CORE_SPLIT_ORDER,
    PortfolioCoreBatch,
    PortfolioCoreInputFiles,
    VerifiedPortfolioCoreInputs,
    _assert_opaque_public_projection,
)
from skillchain.evaluation.portfolio_inputs import (
    PortfolioQueryAssetBinding,
    PortfolioRemoteProcessingFiles,
)
from skillchain.evaluation.portfolio_inputs import (
    ACTIVE_PORTFOLIO_PROCESSOR_ORDER,
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
)
from skillchain.evaluation.portfolio_launch import (
    PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256,
    PORTFOLIO_CORE_ROLE_SELECTION_SHA256,
    PORTFOLIO_LAUNCH_POLICY_VERSION,
    PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
    PORTFOLIO_ROLE_SELECTION_SHA256,
    PortfolioCoreInputFilesBinding,
    PortfolioArtifactBinding,
    PortfolioLaunchArtifactInputs,
    PortfolioLaunchError,
    PortfolioLaunchInstance,
    PortfolioLaunchPlan,
    _PLAN_NORMALIZATION_MARKER,
    _build_core_input_files_binding,
    _build_instances_and_shards,
    _environment_checks,
    _verify_role_selection,
    build_portfolio_launch_plan,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
    PortfolioTreatmentChainManifest,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import sha256_bytes
from skillchain.evaluation import portfolio_launch as launch_module
from skillchain.schemas import ConversationTurn, LabelDecision, Query


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def synthetic_core_inputs() -> VerifiedPortfolioCoreInputs:
    queries: list[Query] = []
    assistant_queries = []
    assets = []
    batches: list[PortfolioCoreBatch] = []
    query_ordinal = 0
    batch_ordinal = 0
    for split in CORE_SPLIT_ORDER:
        for _ in range(CORE_SPLIT_COUNTS[split] // CORE_BATCH_SIZE):
            batch_ordinal += 1
            batch_id = f"core-batch-{batch_ordinal:03d}"
            query_ids: list[str] = []
            for _position in range(CORE_BATCH_SIZE):
                query_ordinal += 1
                query_id = f"core-query-{query_ordinal:04d}"
                asset_id = f"private-source-record-{query_ordinal:04d}"
                image_path = f"private/catalog/{query_ordinal:04d}.jpg"
                capability = "utility.document_reading"
                text = f"Public request {query_ordinal}"
                query = Query(
                    schema_version=2,
                    taxonomy_version="taxonomy-test",
                    task_spec_version="task-test",
                    query_id=query_id,
                    asset_id=asset_id,
                    image_path=image_path,
                    leakage_group_id=f"private-leakage-{query_ordinal:04d}",
                    template_family="private-template-family",
                    generator_batch_id=batch_id,
                    text=text,
                    turns=[ConversationTurn(role="user", content=text)],
                    canonical_intent="utility",
                    canonical_capability=capability,
                    acceptable_capabilities=[capability],
                    requires_card=False,
                    split=split,
                    label_provenance=[
                        LabelDecision(
                            decision_type="constructed",
                            annotator_kind="planner",
                            annotator_id="private-label-source",
                            canonical_intent="utility",
                            canonical_capability=capability,
                            acceptable_capabilities=[capability],
                        )
                    ],
                )
                projected = build_assistant_query_input(query)
                _assert_opaque_public_projection(query, projected)
                queries.append(query)
                assistant_queries.append(projected)
                assets.append(
                    PortfolioQueryAssetBinding(
                        query_id=query_id,
                        asset_id=asset_id,
                        image_path=image_path,
                        image_sha256=f"{query_ordinal:064x}",
                    )
                )
                query_ids.append(query_id)
            batches.append(
                PortfolioCoreBatch(
                    batch_id=batch_id,
                    split=split,
                    query_ids=tuple(query_ids),
                )
            )
    # The scheduler only consumes the typed query/projection/asset/batch members.
    # Full artifact lineage is exercised by the loader and the formal dry-run.
    return VerifiedPortfolioCoreInputs(
        files=None,  # type: ignore[arg-type]
        plan=None,  # type: ignore[arg-type]
        queries=tuple(queries),
        assistant_queries=tuple(assistant_queries),
        query_assets=tuple(assets),
        batches=tuple(batches),
        remote_runtimes=(),
        _snapshots=(),
        _marker=object(),
    )


def test_core_scheduler_builds_full_1500x5_split_atomic_matrix_without_private_dto(
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    instances, shards = _build_instances_and_shards(
        synthetic_core_inputs,
        matrix_run_id="portfolio-core-test-1500x5",
        selected_splits=CORE_SPLIT_ORDER,
    )

    assert len(instances) == CORE_QUERY_COUNT * 5 == 7500
    assert len(shards) == (CORE_QUERY_COUNT // CORE_BATCH_SIZE) * 5 == 300
    assert all(shard.query_count == CORE_BATCH_SIZE for shard in shards)
    assert len({(item.query_id, item.config) for item in instances}) == 7500
    for instance in instances:
        public = instance.model_dump(mode="json")
        assert "asset_id" not in public
        assert "image_path" not in public
        assert set(public).isdisjoint(
            {
                "source",
                "source_record_id",
                "template_family",
                "canonical_intent",
                "canonical_capability",
                "acceptable_capabilities",
                "split",
                "label_provenance",
            }
        )
        assert public["asset_token"].startswith("asset-token-")
        assert len(public["asset_binding_sha256"]) == 64

    first_query = synthetic_core_inputs.queries[0]
    first_public = json.loads(
        synthetic_core_inputs.assistant_queries[0].public_input_json
    )
    assert set(first_public) == {"asset_id", "text", "turns"}
    assert first_query.asset_id not in first_public.values()
    assert first_query.image_path not in first_public.values()


def test_static_opt_scheduler_is_exactly_800x1_and_32_atomic_shards(
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    instances, shards = _build_instances_and_shards(
        synthetic_core_inputs,
        matrix_run_id="portfolio-core-static-opt-test",
        selected_splits=("opt_pool",),
        config_order=("llm_static",),
    )

    assert len(instances) == 800
    assert len(shards) == 32
    assert {item.config for item in instances} == {"llm_static"}
    assert {item.config for item in shards} == {"llm_static"}
    assert all(item.query_count == 25 for item in shards)
    assert len({item.query_id for item in instances}) == 800
    assert tuple(item.shard_ordinal for item in shards) == tuple(range(32))


def test_core_launch_instance_rejects_mixed_opaque_and_private_asset_binding(
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    instances, _ = _build_instances_and_shards(
        synthetic_core_inputs,
        matrix_run_id="portfolio-core-test-binding",
        selected_splits=("dev_mini",),
    )
    payload = instances[0].model_dump(mode="json")
    payload.update({"asset_id": "private-id", "image_path": "private/path.jpg"})

    with pytest.raises(ValidationError, match="exactly one"):
        PortfolioLaunchInstance.model_validate(payload, strict=True)


def test_processor_orders_preserve_history_and_bind_role_swap() -> None:
    assert ACTIVE_PORTFOLIO_PROCESSOR_ORDER == (
        "dashscope-qwen-assistant",
        "dashscope-kimi-feedback",
        "aifast-gemini-judge",
    )
    assert CORE_PORTFOLIO_PROCESSOR_ORDER == (
        "dashscope-qwen-assistant",
        "dashscope-kimi-feedback",
        "dashscope-kimi-judge",
    )
    assert ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER == (
        "dashscope-qwen-assistant",
        "dashscope-kimi-feedback",
        "aifast-gemini-judge",
    )


def test_prepare_launch_cli_exposes_static_opt_mode() -> None:
    arguments = prepare_launch_script.build_parser().parse_args(
        [
            "--output-dir",
            "launch",
            "--profile",
            "core",
            "--execution-mode",
            "static_opt_rollout",
            "--split",
            "opt_pool",
        ]
    )
    assert arguments.execution_mode == "static_opt_rollout"
    assert arguments.split == ["opt_pool"]
    geometry_artifacts = prepare_launch_script._artifact_inputs(arguments)
    assert geometry_artifacts.bank_paths == {}
    assert geometry_artifacts.assistant_runtime_lock_path is None


def test_static_opt_launch_cli_binds_only_static_bank_and_refresh_receipt() -> None:
    arguments = prepare_launch_script.build_parser().parse_args(
        [
            "--output-dir",
            "launch",
            "--profile",
            "core",
            "--execution-mode",
            "static_opt_rollout",
            "--split",
            "opt_pool",
            "--assistant-runtime-lock",
            "runtime/runtime-lock.json",
            "--assistant-runtime-lock-sha256",
            "a" * 64,
            "--static-contract-refresh-receipt",
            "runtime/static-contract-refresh-receipt.json",
            "--static-contract-refresh-receipt-sha256",
            "b" * 64,
            "--bank",
            f"llm_static=runtime/bank-llm_static.json@{'c' * 64}",
        ]
    )

    artifacts = prepare_launch_script._artifact_inputs(arguments)

    assert artifacts.treatment_chain_manifest_path is None
    assert artifacts.static_contract_refresh_receipt_file_sha256 == "b" * 64
    assert set(artifacts.bank_paths or {}) == {"llm_static"}


def test_static_opt_launch_cli_rejects_treatment_chain() -> None:
    arguments = prepare_launch_script.build_parser().parse_args(
        [
            "--output-dir",
            "launch",
            "--profile",
            "core",
            "--execution-mode",
            "static_opt_rollout",
            "--split",
            "opt_pool",
            "--treatment-chain-manifest",
            "runtime/treatment-chain-manifest.json",
            "--treatment-chain-manifest-sha256",
            "a" * 64,
            "--bank",
            f"llm_static=runtime/bank-llm_static.json@{'b' * 64}",
        ]
    )

    with pytest.raises(ValueError, match="forbids a treatment-chain"):
        prepare_launch_script._artifact_inputs(arguments)


def test_core_keeps_v4_while_active_dev_mini_binds_v7_and_loads_v6() -> None:
    v4_file, v4_selection = _verify_role_selection(
        REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v4.json",
        expected_file_sha256=PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256,
        core=True,
    )
    v7_file, v7_selection = _verify_role_selection(
        REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v7.json",
        expected_file_sha256=PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
    )
    legacy_v6_file, legacy_v6_selection = _verify_role_selection(
        REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v6.json",
        expected_file_sha256=(
            "86d5b7763dfee3087d8a9df05659387ca1033093fc021d806643864080e770f9"
        ),
    )

    assert (v4_file, v4_selection) == (
        PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256,
        PORTFOLIO_CORE_ROLE_SELECTION_SHA256,
    )
    assert (v7_file, v7_selection) == (
        PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
        PORTFOLIO_ROLE_SELECTION_SHA256,
    )
    assert (legacy_v6_file, legacy_v6_selection) == (
        "86d5b7763dfee3087d8a9df05659387ca1033093fc021d806643864080e770f9",
        "940f886e40fd0438197384b4de9042efe4f620d8b7a8557060e36f407b0294a0",
    )
    with pytest.raises(ValueError, match="content drifted"):
        _verify_role_selection(
            REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v7.json",
            expected_file_sha256=PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
            core=True,
        )


def test_core_environment_preflight_does_not_require_gemini_or_aifast(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks, blockers = _environment_checks(core=True)
    check_ids = {item.check_id for item in checks}

    assert "env.DASHSCOPE_API_KEY" in check_ids
    assert "endpoint.dashscope" in check_ids
    assert "endpoint.kimi_dashscope" in check_ids
    assert "env.GEMINI_API_KEY" not in check_ids
    assert "endpoint.aifast" not in check_ids
    assert "missing_env:GEMINI_API_KEY" not in blockers


def test_role_swapped_core_environment_requires_gemini_and_aifast(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks, blockers = _environment_checks(core=True, require_gemini=True)
    check_ids = {item.check_id for item in checks}

    assert "env.GEMINI_API_KEY" in check_ids
    assert "endpoint.aifast" in check_ids
    assert "endpoint.kimi_dashscope" in check_ids
    assert "missing_env:GEMINI_API_KEY" in blockers


def _synthetic_core_files(tmp_path: Path) -> PortfolioCoreInputFiles:
    return PortfolioCoreInputFiles(
        plan_path=tmp_path / "plan.json",
        expected_plan_sha256="1" * 64,
        pre_generation_manifest_path=tmp_path / "pre-generation.json",
        expected_pre_generation_manifest_file_sha256="2" * 64,
        split_assignment_path=tmp_path / "final-splits.jsonl",
        expected_split_assignment_sha256="3" * 64,
        query_artifact_path=tmp_path / "queries.jsonl",
        expected_query_artifact_sha256="4" * 64,
        materialization_manifest_path=tmp_path / "repair-manifest.json",
        expected_materialization_manifest_file_sha256="5" * 64,
        capability_assignments_path=tmp_path / "capabilities.jsonl",
        expected_capability_assignments_sha256="6" * 64,
        base_catalog_dir=tmp_path / "base-catalog",
        expected_base_catalog_sha256="7" * 64,
        asset_root=tmp_path / "assets",
    )


def _synthetic_core_plan(
    monkeypatch,
    tmp_path: Path,
    inputs: VerifiedPortfolioCoreInputs,
) -> tuple[PortfolioLaunchPlan, VerifiedPortfolioCoreInputs]:
    bound_inputs = replace(inputs, files=_synthetic_core_files(tmp_path))
    monkeypatch.setattr(
        launch_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )
    plan, _ = build_portfolio_launch_plan(
        bound_inputs,
        matrix_run_id="portfolio-core-binding-test",
        role_selection_path=(
            REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v4.json"
        ),
        expected_role_selection_file_sha256=(PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256),
        operator_approved_dashscope_budget_cny=200.0,
        planning_ceiling_cny=180.0,
        available_disk_bytes=3 * 1024 * 1024 * 1024,
        selected_splits=("dev_mini",),
    )
    return plan, bound_inputs


def test_core_full_matrix_uses_forward_v7_role_identity(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    bound_inputs = replace(
        synthetic_core_inputs,
        files=_synthetic_core_files(tmp_path),
    )
    monkeypatch.setattr(
        launch_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )

    plan, _ = build_portfolio_launch_plan(
        bound_inputs,
        matrix_run_id="portfolio-core-v7-role-swap-test",
        role_selection_path=(
            REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v7.json"
        ),
        expected_role_selection_file_sha256=PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
        operator_approved_dashscope_budget_cny=200.0,
        planning_ceiling_cny=180.0,
        available_disk_bytes=3 * 1024 * 1024 * 1024,
        selected_splits=("dev_mini",),
    )

    assert plan.role_selection_sha256 == PORTFOLIO_ROLE_SELECTION_SHA256
    assert plan.active_processor_order == ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
    assert (plan.feedback_provider, plan.feedback_model) == ("kimi", "kimi-k2.6")
    assert (plan.final_provider, plan.final_model) == (
        "gemini",
        "gemini-3.6-flash",
    )
    assert "aifast_gemini_judge_pricing_and_ceiling_required" in plan.blockers


def test_static_opt_launch_identity_disables_every_final_judge_call(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    bound_inputs = replace(
        synthetic_core_inputs,
        files=_synthetic_core_files(tmp_path),
    )
    monkeypatch.setattr(
        launch_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )

    plan, instances = build_portfolio_launch_plan(
        bound_inputs,
        matrix_run_id="portfolio-core-static-opt-plan-test",
        role_selection_path=(
            REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v4.json"
        ),
        expected_role_selection_file_sha256=(PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256),
        operator_approved_dashscope_budget_cny=200.0,
        planning_ceiling_cny=180.0,
        available_disk_bytes=3 * 1024 * 1024 * 1024,
        selected_splits=("opt_pool",),
        execution_mode="static_opt_rollout",
    )

    assert plan.kind == "portfolio-core-static-opt-800x1-launch-plan"
    assert plan.execution_mode == "static_opt_rollout"
    assert plan.selected_splits == ("opt_pool",)
    assert plan.config_order == ("llm_static",)
    assert (plan.query_count, plan.instance_count, plan.shard_count) == (800, 800, 32)
    assert plan.rate.assistant_concurrency == 2
    assert plan.rate.final_judge_concurrency == 0
    assert plan.calls.final_judge_instance_count == 0
    assert plan.calls.final_judge_call_count == 0
    assert plan.calls.final_judge_empty_response_retry_call_ceiling == 0
    assert plan.calls.total_call_floor == plan.calls.assistant_call_floor
    assert len(instances.splitlines()) == 800


def test_static_opt_launch_rejects_any_split_other_than_opt_pool(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    bound_inputs = replace(
        synthetic_core_inputs,
        files=_synthetic_core_files(tmp_path),
    )
    monkeypatch.setattr(
        launch_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )

    with pytest.raises(PortfolioLaunchError, match="requires Core selected_splits"):
        build_portfolio_launch_plan(
            bound_inputs,
            matrix_run_id="portfolio-core-static-wrong-split",
            role_selection_path=(
                REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v4.json"
            ),
            expected_role_selection_file_sha256=(
                PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256
            ),
            selected_splits=("dev_mini",),
            execution_mode="static_opt_rollout",
        )


def test_core_ready_launch_uses_verified_runtime_rebind_projection(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    config_order = ("llm_static", "s1", "s1s2", "full")
    registry_sha256 = "a" * 64
    registry_runtime_sha256 = "b" * 64
    banks = {
        config: StaticBankArtifact.model_construct(
            tool_registry_sha256=registry_sha256,
            tool_registry_runtime_sha256=registry_runtime_sha256,
            bank_sha256=f"{index:x}" * 64,
        )
        for index, config in enumerate(config_order, 1)
    }
    bank_file_sha256s = {}
    for config, bank in banks.items():
        path = runtime_root / f"bank-{config}.json"
        path.write_bytes(bank.canonical_bytes())
        bank_file_sha256s[config] = sha256_bytes(bank.canonical_bytes())
    runtime_lock_path = runtime_root / "runtime-lock.json"
    treatment_path = runtime_root / "treatment-chain-manifest.json"
    runtime_lock_path.write_bytes(b"{}\n")
    treatment_path.write_bytes(b"{}\n")

    query_ids = tuple(item.query_id for item in synthetic_core_inputs.queries[:200])
    records = tuple(
        SimpleNamespace(config=config, receipt_sha256=f"{index + 4:x}" * 64)
        for index, config in enumerate(config_order)
    )
    source_bank_bindings = tuple(
        SimpleNamespace(
            config=config,
            bank_sha256="f" * 64,
            bank_file_sha256="e" * 64,
        )
        for config in config_order
    )
    manifest = PortfolioTreatmentChainManifest.model_construct(
        status="ready_for_matrix",
        optimization_query_ids=query_ids[:25],
        evaluation_query_ids=query_ids[25:],
        records=records,
        banks=source_bank_bindings,
        chain_sha256="c" * 64,
    )
    output_bindings = tuple(
        SimpleNamespace(
            config=config,
            role="output",
            rebound_bank_file=f"bank-{config}.json",
        )
        for config in config_order
    )
    fake_chain = SimpleNamespace(
        manifest=manifest,
        output_banks=banks,
        compatibility_rebind=SimpleNamespace(bank_bindings=output_bindings),
    )
    runtime = {
        "runtime_lock_sha256": "d" * 64,
        "runtime_compatibility_rebind_file": "compatibility-rebind.json",
        "tool_registry_sha256": registry_sha256,
        "tool_registry_runtime_sha256": registry_runtime_sha256,
        "bank_sha256s": {config: bank.bank_sha256 for config, bank in banks.items()},
        "bank_policy": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "official_matrix_eligible": True,
        "treatment_chain_status": "ready_for_matrix",
        "treatment_chain_manifest_file_sha256": "e" * 64,
        "treatment_chain_sha256": manifest.chain_sha256,
        "treatment_record_sha256s": {
            item.config: item.receipt_sha256 for item in records
        },
    }
    fake_verified_runtime = SimpleNamespace(
        runtime_lock=runtime,
        chain=fake_chain,
    )
    from skillchain.evaluation import portfolio_treatment_io

    monkeypatch.setattr(
        portfolio_treatment_io,
        "load_verified_portfolio_treatment_runtime",
        lambda *_args, **_kwargs: fake_verified_runtime,
    )

    def fake_artifact_binding(*, artifact_id, path, expected_file_sha256, kind):
        del kind
        if artifact_id == "assistant_runtime_lock":
            value = runtime
            content_sha256 = runtime["runtime_lock_sha256"]
        elif artifact_id == "treatment_chain_manifest":
            value = manifest
            content_sha256 = manifest.chain_sha256
        else:
            config = artifact_id.removeprefix("bank.")
            value = banks[config]
            content_sha256 = value.bank_sha256
        return (
            PortfolioArtifactBinding(
                artifact_id=artifact_id,
                path=path.as_posix(),
                file_sha256=expected_file_sha256,
                content_sha256=content_sha256,
                status="verified",
                detail="test verified",
            ),
            value,
        )

    monkeypatch.setattr(launch_module, "_artifact_binding", fake_artifact_binding)
    monkeypatch.setattr(
        launch_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )
    bound_inputs = replace(
        synthetic_core_inputs,
        files=_synthetic_core_files(tmp_path),
    )
    artifacts = PortfolioLaunchArtifactInputs(
        assistant_runtime_lock_path=runtime_lock_path,
        assistant_runtime_lock_file_sha256="d" * 64,
        treatment_chain_manifest_path=treatment_path,
        treatment_chain_manifest_file_sha256="e" * 64,
        bank_paths={
            config: runtime_root / f"bank-{config}.json" for config in config_order
        },
        bank_file_sha256s=bank_file_sha256s,
    )

    plan, _ = build_portfolio_launch_plan(
        bound_inputs,
        matrix_run_id="portfolio-core-rebind-ready-test",
        role_selection_path=(
            REPOSITORY_ROOT / "specs" / "authoring" / "model-role-selection-v4.json"
        ),
        expected_role_selection_file_sha256=(PORTFOLIO_CORE_ROLE_SELECTION_FILE_SHA256),
        artifacts=artifacts,
        operator_approved_dashscope_budget_cny=200.0,
        planning_ceiling_cny=180.0,
        available_disk_bytes=3 * 1024 * 1024 * 1024,
        selected_splits=("dev_mini",),
    )

    checks = {item.check_id: item.status for item in plan.preflight_checks}
    assert checks["binding.runtime_compatibility_rebind"] == "passed"
    assert checks["binding.treatment_chain_semantics"] == "passed"
    assert "runtime_compatibility_rebind_invalid" not in plan.blockers
    assert "treatment_chain_semantics_invalid" not in plan.blockers


def test_core_plan_carries_reconstructable_typed_input_recipe(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    plan, expected = _synthetic_core_plan(monkeypatch, tmp_path, synthetic_core_inputs)
    captured: list[PortfolioCoreInputFiles] = []

    def fake_load(files: PortfolioCoreInputFiles) -> VerifiedPortfolioCoreInputs:
        captured.append(files)
        return expected

    monkeypatch.setattr(launch_module, "load_verified_portfolio_core_inputs", fake_load)
    rebuilt = reconstruct_verified_portfolio_core_inputs(plan)

    assert rebuilt is expected
    assert captured == [expected.files]
    assert plan.core_input_files is not None
    assert plan.core_input_files.plan_path == expected.files.plan_path.as_posix()
    assert plan.core_input_files.remote_files is None
    first_public = json.loads(rebuilt.assistant_queries[0].public_input_json)
    assert set(first_public) == {"asset_id", "text", "turns"}
    assert rebuilt.queries[0].asset_id not in first_public.values()
    assert rebuilt.queries[0].image_path not in first_public.values()
    assert plan.rate.assistant_concurrency == 2


def test_core_input_recipe_automatically_binds_remote_permission_files(
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    base_files = _synthetic_core_files(tmp_path)
    remote_files = PortfolioRemoteProcessingFiles(
        authorization_file=tmp_path / "authorization.json",
        expected_authorization_file_sha256="a" * 64,
        receipt_file=tmp_path / "receipt.json",
        expected_receipt_file_sha256="b" * 64,
        selection_manifest=tmp_path / "selection.json",
        dataset_assets=tmp_path / "dataset-assets.jsonl",
        base_catalog_dir=base_files.base_catalog_dir,
        output_catalog_dir=tmp_path / "runtime-catalog",
        asset_root=base_files.asset_root,
        processor_order=CORE_PORTFOLIO_PROCESSOR_ORDER,
    )
    files = replace(
        base_files,
        runtime_catalog_dir=remote_files.output_catalog_dir,
        expected_runtime_catalog_sha256="e" * 64,
        remote_files=remote_files,
    )
    receipt = SimpleNamespace(
        receipt_sha256="c" * 64,
        base_selection_manifest_sha256="d" * 64,
        base_catalog_sha256=files.expected_base_catalog_sha256,
    )
    authorization = SimpleNamespace(authorization_id="core-owner-auth-test")
    catalog = SimpleNamespace(catalog_sha256="e" * 64)
    runtimes = tuple(
        SimpleNamespace(
            processor=processor,
            authorization=authorization,
            authorization_file_sha256="a" * 64,
            receipt_file_sha256="b" * 64,
            receipt=receipt,
            dataset_assets_sha256="f" * 64,
            catalog=catalog,
        )
        for processor in CORE_PORTFOLIO_PROCESSOR_ORDER
    )
    inputs = replace(
        synthetic_core_inputs,
        files=files,
        remote_runtimes=runtimes,
    )

    binding = _build_core_input_files_binding(inputs)

    assert binding.remote_files is not None
    assert binding.remote_files.authorization_file == (
        remote_files.authorization_file.as_posix()
    )
    assert binding.remote_files.expected_receipt_sha256 == "c" * 64
    assert binding.remote_files.expected_selection_manifest_sha256 == "d" * 64
    assert binding.remote_files.expected_dataset_assets_sha256 == "f" * 64
    assert binding.to_input_files() == files


def test_core_input_recipe_rejects_relative_path(tmp_path: Path) -> None:
    payload = {
        "plan_path": "relative/plan.json",
        "expected_plan_sha256": "1" * 64,
        "pre_generation_manifest_path": (tmp_path / "pre.json").as_posix(),
        "expected_pre_generation_manifest_file_sha256": "2" * 64,
        "split_assignment_path": (tmp_path / "split.jsonl").as_posix(),
        "expected_split_assignment_sha256": "3" * 64,
        "query_artifact_path": (tmp_path / "queries.jsonl").as_posix(),
        "expected_query_artifact_sha256": "4" * 64,
        "materialization_manifest_path": (tmp_path / "repair.json").as_posix(),
        "expected_materialization_manifest_file_sha256": "5" * 64,
        "capability_assignments_path": (tmp_path / "caps.jsonl").as_posix(),
        "expected_capability_assignments_sha256": "6" * 64,
        "base_catalog_dir": (tmp_path / "catalog").as_posix(),
        "expected_base_catalog_sha256": "7" * 64,
        "asset_root": (tmp_path / "assets").as_posix(),
    }

    with pytest.raises(ValidationError, match="normalized absolute"):
        PortfolioCoreInputFilesBinding.model_validate(payload, strict=True)


def test_core_plan_rejects_input_hash_or_profile_drift(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    plan, _ = _synthetic_core_plan(monkeypatch, tmp_path, synthetic_core_inputs)
    payload = plan.model_dump(mode="python")
    hash_drift = dict(payload)
    hash_drift["core_input_files"] = {
        **hash_drift["core_input_files"],
        "expected_query_artifact_sha256": "9" * 64,
    }
    with pytest.raises(ValidationError, match="differs from plan hashes"):
        PortfolioLaunchPlan.model_validate(
            hash_drift,
            strict=True,
            context=_PLAN_NORMALIZATION_MARKER,
        )

    profile_drift = {**payload, "dataset_profile": None}
    with pytest.raises(ValidationError, match="identity or split selection drifted"):
        PortfolioLaunchPlan.model_validate(
            profile_drift,
            strict=True,
            context=_PLAN_NORMALIZATION_MARKER,
        )

    missing_binding = {**payload, "core_input_files": None}
    with pytest.raises(ValidationError, match="identity or split selection drifted"):
        PortfolioLaunchPlan.model_validate(
            missing_binding,
            strict=True,
            context=_PLAN_NORMALIZATION_MARKER,
        )


def test_dev_mini_plan_must_omit_core_input_recipe(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    core_plan, _ = _synthetic_core_plan(monkeypatch, tmp_path, synthetic_core_inputs)
    payload = core_plan.model_dump(mode="python")
    payload.update(
        {
            "kind": "portfolio-dev-mini-200x5-launch-plan",
            "policy_version": PORTFOLIO_LAUNCH_POLICY_VERSION,
            "dataset_profile": None,
            "source_query_count": None,
            "selected_splits": (),
            "split_assignment_sha256": None,
            "materialization_manifest_file_sha256": None,
            "core_input_files": None,
            "accepted_ledger_sha256": "a" * 64,
            "seed_set_sha256": "b" * 64,
            "active_processor_order": ACTIVE_PORTFOLIO_PROCESSOR_ORDER,
            "remote_runtime_binding_sha256s": (
                "c" * 64,
                "d" * 64,
                "e" * 64,
            ),
            "authorization_file_sha256": "f" * 64,
            "receipt_file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "role_selection_file_sha256": PORTFOLIO_ROLE_SELECTION_FILE_SHA256,
            "role_selection_sha256": PORTFOLIO_ROLE_SELECTION_SHA256,
            "feedback_provider": "kimi",
            "feedback_model": "kimi-k2.6",
            "final_provider": "gemini",
            "final_model": "gemini-3.6-flash",
            "blockers": ("test_blocker",),
            "scheduling_policy": "balanced_cyclic_config_order_by_accepted_batch",
        }
    )
    dev_plan = PortfolioLaunchPlan.model_validate(
        payload,
        strict=True,
        context=_PLAN_NORMALIZATION_MARKER,
    )

    assert dev_plan.core_input_files is None
    assert "core_input_files" not in dev_plan.model_dump(mode="json")

    payload["core_input_files"] = core_plan.core_input_files.model_dump(mode="python")
    with pytest.raises(ValidationError, match="historical dev_mini"):
        PortfolioLaunchPlan.model_validate(
            payload,
            strict=True,
            context=_PLAN_NORMALIZATION_MARKER,
        )


def test_core_reconstruction_fails_closed_on_bound_path_tamper(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    plan, _ = _synthetic_core_plan(monkeypatch, tmp_path, synthetic_core_inputs)
    payload = plan.model_dump(mode="python")
    payload["core_input_files"] = {
        **payload["core_input_files"],
        "plan_path": (tmp_path / "moved-or-missing-plan.json").as_posix(),
    }
    tampered = PortfolioLaunchPlan.model_validate(
        payload,
        strict=True,
        context=_PLAN_NORMALIZATION_MARKER,
    )

    with pytest.raises(PortfolioLaunchError, match="could not be reconstructed"):
        reconstruct_verified_portfolio_core_inputs(tampered)


def test_prepare_core_launch_reuses_external_typed_input_recipe(
    monkeypatch,
    tmp_path: Path,
    synthetic_core_inputs: VerifiedPortfolioCoreInputs,
) -> None:
    source_plan = object()
    source_package = SimpleNamespace(plan=source_plan)
    source_root = tmp_path / "input-binding-launch"
    plan_file_sha256 = "a" * 64
    observed: dict[str, object] = {}

    def fake_load(
        root: Path,
        *,
        expected_plan_file_sha256: str,
        _allow_legacy_budget_contract: bool = False,
    ):
        observed["root"] = root
        observed["plan_file_sha256"] = expected_plan_file_sha256
        observed["allow_legacy_budget_contract"] = _allow_legacy_budget_contract
        return source_package

    def fake_reconstruct(plan: object):
        observed["plan"] = plan
        return synthetic_core_inputs

    monkeypatch.setattr(
        prepare_launch_script,
        "load_portfolio_launch_package",
        fake_load,
    )
    monkeypatch.setattr(
        prepare_launch_script,
        "reconstruct_verified_portfolio_core_inputs",
        fake_reconstruct,
    )
    arguments = prepare_launch_script.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--profile",
            "core",
            "--core-input-binding-launch-root",
            str(source_root),
            "--core-input-binding-launch-plan-file-sha256",
            plan_file_sha256,
        ]
    )

    assert prepare_launch_script._load_core_inputs(arguments) is synthetic_core_inputs
    assert observed == {
        "root": source_root,
        "plan_file_sha256": plan_file_sha256,
        "allow_legacy_budget_contract": True,
        "plan": source_plan,
    }


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (
            ["--core-input-binding-launch-root", "bound-launch"],
            "must be supplied together",
        ),
        (
            [
                "--core-input-binding-launch-root",
                "bound-launch",
                "--core-input-binding-launch-plan-file-sha256",
                "b" * 64,
                "--core-asset-root",
                "asset-root",
            ],
            "cannot be mixed with raw Core inputs",
        ),
    ],
)
def test_prepare_core_launch_rejects_ambiguous_input_recipe(
    tmp_path: Path,
    extra_args: list[str],
    message: str,
) -> None:
    arguments = prepare_launch_script.build_parser().parse_args(
        ["--output-dir", str(tmp_path / "output"), "--profile", "core", *extra_args]
    )

    with pytest.raises(ValueError, match=message):
        prepare_launch_script._load_core_inputs(arguments)
