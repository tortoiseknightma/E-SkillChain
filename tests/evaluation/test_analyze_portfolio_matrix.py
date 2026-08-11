from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_portfolio_matrix as analysis
from skillchain.evaluation.portfolio_launch import MAIN_CONFIG_ORDER
from skillchain.synthesis.portfolio_core_r3_overlay import (
    R3PublishedFile,
    R3RepairManifest,
    build_r3_repair_plan,
)


def _row(
    *,
    query_ordinal: int,
    config: str,
    capability: str = "product.exact_match",
    selected_capability: str | None = "product.exact_match",
    j_project: float = 50.0,
    leakage_group_id: str | None = None,
) -> analysis.MatrixRow:
    skilled = config != "noskill"
    return analysis.MatrixRow(
        query_ordinal=query_ordinal,
        query_id=f"q-{query_ordinal:03d}",
        batch_id="batch-001",
        partition="optimization25",
        leakage_group_id=leakage_group_id or f"cluster-{query_ordinal // 2}",
        canonical_capability=capability,
        config=config,
        bank_sha256=(
            f"{MAIN_CONFIG_ORDER.index(config) + 1:064x}" if skilled else None
        ),
        j_project=j_project,
        selected_capability=selected_capability,
        route_correct=selected_capability == capability,
        route_acceptable=selected_capability == capability,
        structural_adherence=1.0 if skilled else None,
        gate_aligned_adherence=0.75 if skilled else None,
        assistant_hard_error=False,
        evaluator_anomaly=False,
        assistant_status="success",
        assistant_error_code=None,
        assistant_receipt_outcome="success",
        assistant_model_call_count=1,
        assistant_finish_reasons=("stop",),
        tool_call_count=1,
        tool_error_count=0,
        tool_error_codes=(),
        tool_statuses=("image_search:success:",),
        tool_recovery_status="none",
        card_requirement="required",
        visible_card_count=1,
        card_policy_compliant=True,
        card_guard_adjusted=False,
        judge_status="scored",
        judge_error_code=None,
        judge_provider="kimi",
        judge_model="kimi-k2.6",
        judge_finish_reason="stop",
        attempt_receipt_count=0,
        attempt_failure_stages=(),
        attempt_failure_subtypes=(),
        attempt_recovery_status="none",
        attempt_circuit_open_count=0,
        attempt_captured_provider_response_count=0,
        attempt_captured_input_tokens=0,
        attempt_captured_output_tokens=0,
    )


def _rectangle(query_count: int = 3) -> list[analysis.MatrixRow]:
    return [
        _row(query_ordinal=index, config=config)
        for index in range(query_count)
        for config in MAIN_CONFIG_ORDER
    ]


def test_macro_f1_counts_missing_prediction_as_wrong_across_six_classes() -> None:
    rows = [
        _row(
            query_ordinal=index,
            config="s1",
            capability=capability,
            selected_capability=(capability if index != 5 else None),
        )
        for index, capability in enumerate(analysis.CAPABILITIES)
    ]

    result = analysis._macro_f1(rows)

    assert result["denominator"] == 6
    assert result["missing_or_invalid_prediction_count"] == 1
    assert result["macro_f1"] == pytest.approx(5 / 6)
    assert result["classes"][analysis.CAPABILITIES[-1]]["fn"] == 1
    assert result["classes"][analysis.CAPABILITIES[-1]]["f1"] == 0.0


def test_json_projection_converts_nested_row_tuples_for_canonical_hashing() -> None:
    projected = analysis._json_projection(
        [
            {
                "finish_reasons": ("stop",),
                "nested": ({"codes": ("context_violation",)},),
            }
        ]
    )

    assert projected == [
        {
            "finish_reasons": ["stop"],
            "nested": [{"codes": ["context_violation"]}],
        }
    ]
    analysis.canonical_json_bytes(projected)
    assert analysis._json_cell(("stop", "length")) == '["stop","length"]'


def test_config_summary_keeps_error_zero_in_full_j_denominator() -> None:
    rows = _rectangle(2)
    error_index = next(
        index
        for index, row in enumerate(rows)
        if row.config == "s1" and row.query_id == "q-001"
    )
    rows[error_index] = replace(
        rows[error_index],
        j_project=0.0,
        selected_capability=None,
        route_correct=False,
        route_acceptable=False,
        structural_adherence=0.0,
        gate_aligned_adherence=0.0,
        assistant_hard_error=True,
        assistant_status="error",
        assistant_error_code="runtime_error",
        assistant_receipt_outcome="runtime_error",
        judge_status="not_invoked_assistant_error",
        judge_provider=None,
        judge_model=None,
        judge_finish_reason=None,
    )

    summary = next(
        item
        for item in analysis._config_summary(rows, "synthetic")
        if item["config"] == "s1"
    )

    assert summary["j_denominator_all_rows"] == 2
    assert summary["mean_j_project_all_rows"] == 25.0
    assert summary["route_missing_or_invalid_prediction_count"] == 1
    assert summary["hard_error_count"] == 1
    assert summary["assistant_hard_error_count"] == 1
    assert summary["evaluator_anomaly_count"] == 0
    assert summary["structural_adherence_mean"] == 0.5
    assert summary["gate_aligned_adherence_mean"] == 0.375


def test_noskill_routing_is_not_applicable_but_skilled_missing_is_wrong() -> None:
    rows = _rectangle(1)
    rows = [
        replace(
            row,
            selected_capability=None,
            route_correct=False,
            route_acceptable=False,
        )
        if row.config in {"noskill", "s1"}
        else row
        for row in rows
    ]

    summaries = analysis._config_summary(rows, "synthetic")
    noskill = next(item for item in summaries if item["config"] == "noskill")
    skilled = next(item for item in summaries if item["config"] == "s1")

    assert noskill["routing_applicability"] == "not_applicable"
    assert noskill["route_accuracy_missing_is_wrong"] is None
    assert noskill["route_macro_f1_6class_missing_is_wrong"] is None
    assert noskill["route_classes"] is None
    assert skilled["routing_applicability"] == "applicable"
    assert skilled["route_accuracy_missing_is_wrong"] == 0.0
    assert skilled["route_missing_or_invalid_prediction_count"] == 1

    capability = analysis._capability_summary(rows, "synthetic")
    noskill_capability = next(
        item for item in capability if item["config"] == "noskill"
    )
    assert noskill_capability["routing_applicability"] == "not_applicable"
    assert noskill_capability["route_class_f1"] is None


def test_evaluator_anomaly_is_not_an_assistant_hard_error() -> None:
    rows = _rectangle(1)
    index = next(index for index, row in enumerate(rows) if row.config == "full")
    rows[index] = replace(
        rows[index],
        j_project=0.0,
        evaluator_anomaly=True,
        judge_status="provider_error",
        judge_error_code="provider_error",
    )

    summary = next(
        item
        for item in analysis._config_summary(rows, "synthetic")
        if item["config"] == "full"
    )
    assert summary["hard_error_count"] == 0
    assert summary["assistant_hard_error_count"] == 0
    assert summary["evaluator_anomaly_count"] == 1


def test_paired_summary_rejects_independent_sampling_for_same_bank() -> None:
    rows = _rectangle(3)
    scores = {
        ("q-000", "s1"): 40.0,
        ("q-000", "s1s2"): 50.0,
        ("q-001", "s1"): 50.0,
        ("q-001", "s1s2"): 50.0,
        ("q-002", "s1"): 60.0,
        ("q-002", "s1s2"): 55.0,
    }
    rows = [
        replace(row, j_project=scores.get((row.query_id, row.config), row.j_project))
        for row in rows
    ]
    banks = {
        "noskill": None,
        "llm_static": "1" * 64,
        "s1": "2" * 64,
        "s1s2": "2" * 64,
        "full": "2" * 64,
    }

    with pytest.raises(ValueError, match="independently sampled scores"):
        analysis._paired_summaries(rows, scope="synthetic", bank_sha256s=banks)


def test_paired_summary_reports_artifact_alias_as_strict_zero_and_all_ties() -> None:
    rows = _rectangle(3)
    banks = {
        "noskill": None,
        "llm_static": "1" * 64,
        "s1": "2" * 64,
        "s1s2": "3" * 64,
        "full": "3" * 64,
    }

    first = analysis._paired_summaries(rows, scope="synthetic", bank_sha256s=banks)
    second = analysis._paired_summaries(rows, scope="synthetic", bank_sha256s=banks)
    contrast = next(
        item
        for item in first
        if item["baseline_config"] == "s1s2" and item["treatment_config"] == "full"
    )

    assert first == second
    assert contrast["paired_mean_delta_j_project"] == 0.0
    assert contrast["paired_median_delta_j_project"] == 0.0
    assert (contrast["win_count"], contrast["tie_count"], contrast["loss_count"]) == (
        0,
        3,
        0,
    )
    assert contrast["cluster_count"] == 2
    assert contrast["mean_delta_ci95_low"] == 0.0
    assert contrast["mean_delta_ci95_high"] == 0.0
    assert contrast["same_output_bank_no_op_contrast"] is True
    assert contrast["treatment_attribution_allowed"] is False


def test_partition_contract_rejects_query_or_cluster_overlap() -> None:
    optimization = tuple(f"q-{index:03d}" for index in range(25))
    evaluation = tuple(f"q-{index:03d}" for index in range(25, 200))
    analysis._validate_partition_contract(
        optimization_query_ids=optimization,
        evaluation_query_ids=evaluation,
        optimization_leakage_group_ids=("opt",),
        evaluation_leakage_group_ids=("eval",),
    )

    with pytest.raises(ValueError, match="25/175"):
        analysis._validate_partition_contract(
            optimization_query_ids=optimization,
            evaluation_query_ids=(optimization[0], *evaluation[1:]),
            optimization_leakage_group_ids=("opt",),
            evaluation_leakage_group_ids=("eval",),
        )
    with pytest.raises(ValueError, match="leakage"):
        analysis._validate_partition_contract(
            optimization_query_ids=optimization,
            evaluation_query_ids=evaluation,
            optimization_leakage_group_ids=("shared",),
            evaluation_leakage_group_ids=("shared",),
        )


def test_complete_selection_uses_launch_geometry_instead_of_200x5_constants() -> None:
    shards = tuple(
        SimpleNamespace(
            accepted_batch_id=batch_id,
            config=config,
            query_ids=tuple(f"{batch_id}-q-{index}" for index in range(3)),
            query_count=3,
        )
        for batch_id in ("batch-a", "batch-b")
        for config in MAIN_CONFIG_ORDER
    )
    plan = SimpleNamespace(
        shards=shards,
        query_count=6,
        instance_count=30,
        shard_count=10,
    )

    assert analysis._select_batch_ids(plan, None) == ("batch-a", "batch-b")


def test_static_gcs_v2_contract_is_exact_and_not_a_five_config_alias() -> None:
    control = {
        "execution_scope": "static_opt_rollout",
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": "portfolio-grounded-contract-success-v2",
        "gcs_scorer_evidence_policy_version": ("portfolio-gcs-scorer-evidence-v2"),
        "execution_artifact_aliases": [],
    }
    plan = SimpleNamespace(
        kind="portfolio-core-static-opt-800x1-launch-plan",
        dataset_profile="core",
        execution_mode="static_opt_rollout",
        config_order=("llm_static",),
        selected_splits=("opt_pool",),
        query_count=800,
        instance_count=800,
        shard_count=32,
    )

    analysis._require_static_gcs_v2_contract(control, plan)

    with pytest.raises(ValueError, match="exact GCS v2"):
        analysis._require_static_gcs_v2_contract(
            {**control, "assistant_checkpoint_schema_version": 1}, plan
        )
    with pytest.raises(ValueError, match="exact GCS v2"):
        analysis._require_static_gcs_v2_contract(
            control,
            SimpleNamespace(**{**vars(plan), "config_order": MAIN_CONFIG_ORDER}),
        )


def test_static_gcs_batch_selection_requires_exact_32_by_25_geometry() -> None:
    shards = tuple(
        SimpleNamespace(
            accepted_batch_id=f"batch-{batch:02d}",
            shard_id=f"shard-{batch:02d}",
            config="llm_static",
            query_count=25,
            query_ids=tuple(f"q-{batch * 25 + ordinal:04d}" for ordinal in range(25)),
        )
        for batch in range(32)
    )
    plan = SimpleNamespace(
        shards=shards,
        query_count=800,
        instance_count=800,
        shard_count=32,
    )

    assert len(analysis._select_static_gcs_batch_ids(plan, None)) == 32
    assert analysis._select_static_gcs_batch_ids(plan, "batch-07") == ("batch-07",)

    drifted = list(shards)
    drifted[0] = SimpleNamespace(**{**vars(shards[0]), "config": "full"})
    with pytest.raises(ValueError, match="one 25-query shard"):
        analysis._select_static_gcs_batch_ids(
            SimpleNamespace(**{**vars(plan), "shards": tuple(drifted)}), None
        )


def test_static_gcs_pairwise_and_gate_are_typed_not_applicable() -> None:
    first = analysis._static_not_applicable_artifacts(
        scope="opt_pool",
        population_mapping_sha256="a" * 64,
        query_count=800,
    )
    second = analysis._static_not_applicable_artifacts(
        scope="opt_pool",
        population_mapping_sha256="a" * 64,
        query_count=800,
    )

    assert first == second
    bootstrap, gate = first
    assert bootstrap["status"] == "not_applicable"
    assert bootstrap["reason_code"] == "single_config_static_opt_rollout"
    assert bootstrap["available_configs"] == ["llm_static"]
    assert gate["status"] == "not_applicable"
    assert gate["treatment_config"] is None
    assert gate["required_treatment_config"] == "full"
    assert bootstrap["model_calls_performed"] == gate["model_calls_performed"] == 0


def test_static_gcs_strata_are_typed_and_never_inferred_from_response() -> None:
    style_query = SimpleNamespace(
        query_id="q-style",
        canonical_capability="product.style_recommendation",
        text="请围绕这件衣服搭配鞋和包",
        is_boundary=True,
    )
    exact_query = SimpleNamespace(
        query_id="q-exact",
        canonical_capability="product.exact_match",
        text="请找同款",
        is_boundary=False,
    )

    style = analysis._static_gcs_strata_identity(
        style_query,
        source_dataset="fashioniq",
        repair_query_ids=frozenset({"q-style"}),
    )
    exact = analysis._static_gcs_strata_identity(
        exact_query,
        source_dataset="abo",
        repair_query_ids=frozenset({"q-style"}),
    )

    assert tuple(style) == analysis._STATIC_GCS_STRATUM_ORDER
    assert style["capability"] == {
        "status": "available",
        "value": "product.style_recommendation",
    }
    assert style["source"] == {"status": "available", "value": "fashioniq"}
    assert style["repair"]["value"] == "r3_language_repaired"
    assert style["boundary"]["value"] == "boundary"
    assert style["style_submode"]["value"] == "cross_category_coordination"
    assert exact["repair"]["value"] == "r3_carry_forward"
    assert exact["style_submode"] == {"status": "not_applicable", "value": None}


def test_static_gcs_strata_summary_has_a_fixed_denominator_per_dimension() -> None:
    def score(
        query_id: str,
        *,
        gcs: int,
        style_support_status: str | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            query_id=query_id,
            gcs=gcs,
            hard_error=0,
            oracle_available=True,
            semantic_claim_support_resolved=bool(gcs),
            style_support_status=style_support_status,
            route_acceptable=1,
            no_hard_error=1,
            tool_contract_pass=gcs,
            evidence_grounded=gcs,
            output_contract_pass=gcs,
        )

    strata = {
        "q-exact": {
            "capability": {
                "status": "available",
                "value": "product.exact_match",
            },
            "source": {"status": "available", "value": "abo"},
            "repair": {"status": "available", "value": "r3_carry_forward"},
            "boundary": {"status": "available", "value": "non_boundary"},
            "style_submode": {"status": "not_applicable", "value": None},
        },
        "q-style": {
            "capability": {
                "status": "available",
                "value": "product.style_recommendation",
            },
            "source": {"status": "available", "value": "fashioniq"},
            "repair": {
                "status": "available",
                "value": "r3_language_repaired",
            },
            "boundary": {"status": "available", "value": "boundary"},
            "style_submode": {
                "status": "available",
                "value": "cross_category_coordination",
            },
        },
    }

    rows = analysis._static_gcs_strata_summary_rows(
        (
            score("q-exact", gcs=1, style_support_status=None),
            score("q-style", gcs=0, style_support_status="unsupported"),
        ),
        strata_by_query=strata,
        scope="opt_pool:batch:batch-01",
    )

    assert rows == analysis._static_gcs_strata_summary_rows(
        (
            score("q-exact", gcs=1, style_support_status=None),
            score("q-style", gcs=0, style_support_status="unsupported"),
        ),
        strata_by_query=strata,
        scope="opt_pool:batch:batch-01",
    )
    for stratum in analysis._STATIC_GCS_STRATUM_ORDER:
        assert sum(row["query_count"] for row in rows if row["stratum"] == stratum) == 2
    style_row = next(
        row
        for row in rows
        if row["stratum"] == "style_submode"
        and row["value"] == "cross_category_coordination"
    )
    assert style_row["query_count"] == 1
    assert style_row["gcs_rate"] == 0.0
    assert style_row["style_unsupported_count"] == 1
    not_applicable = next(
        row
        for row in rows
        if row["stratum"] == "style_submode" and row["value_status"] == "not_applicable"
    )
    assert not_applicable["value"] is None
    assert not_applicable["query_count"] == 1


def test_static_gcs_repair_membership_comes_from_hash_bound_r3_plan(
    tmp_path: Path,
) -> None:
    repair_plan = build_r3_repair_plan(
        tuple(
            (
                f"repair-{batch:02d}",
                tuple(f"q-{batch * 25 + ordinal:04d}" for ordinal in range(25)),
            )
            for batch in range(10)
        )
    )
    repair_plan_bytes = repair_plan.canonical_bytes()
    repair_plan_path = tmp_path / "repair-plan.json"
    repair_plan_path.write_bytes(repair_plan_bytes)
    descriptor = R3PublishedFile(
        relative_path="repair-plan.json",
        sha256=analysis.sha256_bytes(repair_plan_bytes),
        bytes=len(repair_plan_bytes),
    )
    manifest = R3RepairManifest.create(
        run_id="static-gcs-test",
        source_r2_sha256="1" * 64,
        repair_plan_sha256=repair_plan.repair_plan_sha256,
        queries_sha256="2" * 64,
        parent_batch_index_sha256="3" * 64,
        repair_batch_index_sha256="4" * 64,
        language_validation_sha256="5" * 64,
        repair_ledger_sha256="6" * 64,
        repair_checkpoint_sha256="7" * 64,
        file_count=1,
        files=(descriptor,),
    )
    manifest_path = tmp_path / "repair-manifest.json"
    manifest_bytes = manifest.canonical_bytes()
    manifest_path.write_bytes(manifest_bytes)
    verified = SimpleNamespace(
        files=SimpleNamespace(
            materialization_manifest_path=manifest_path,
            expected_materialization_manifest_file_sha256=(
                analysis.sha256_bytes(manifest_bytes)
            ),
        )
    )

    query_ids, binding = analysis._load_static_gcs_repair_query_ids(verified)

    assert len(query_ids) == 250
    assert "q-0000" in query_ids and "q-0249" in query_ids
    assert binding["repair_plan_sha256"] == repair_plan.repair_plan_sha256
    assert binding["repair_query_count"] == 250
    repair_plan_path.write_bytes(repair_plan_bytes + b"\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        analysis._load_static_gcs_repair_query_ids(verified)


def test_static_checkpoint_request_uses_json_aware_strict_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class RequestStub:
        @classmethod
        def model_validate_json(cls, content: bytes, *, strict: bool):
            captured["content"] = content
            captured["strict"] = strict
            return "parsed-request"

    monkeypatch.setattr(analysis, "AssistantRequestSnapshot", RequestStub)
    wire = {"registry": {"tools": [{"tool_name": "document_ocr"}]}}

    parsed = analysis._assistant_request_from_checkpoint(wire)

    assert parsed == "parsed-request"
    assert captured == {
        "content": analysis.canonical_json_bytes(wire),
        "strict": True,
    }


def test_static_gcs_recovers_taskspec_from_verified_semantic_input() -> None:
    task_spec = analysis.load_mvp_task_specification_v1()
    task_bytes = analysis.canonical_json_bytes(task_spec.model_dump(mode="json"))
    task_file_sha256 = analysis.sha256_bytes(
        analysis.MVP_TASK_SPEC_V1_PATH.read_bytes()
    )
    semantic_file_sha256 = "1" * 64
    semantic_input_sha256 = "2" * 64
    refresh_file_sha256 = "3" * 64
    refresh_sha256 = "4" * 64
    core_file_sha256 = "5" * 64
    core_sha256 = "6" * 64
    runtime_data_sha256 = "7" * 64
    semantic = SimpleNamespace(
        input_sha256=semantic_input_sha256,
        task_specification=SimpleNamespace(
            content=task_bytes,
            version=task_spec.task_spec_version,
            identity_sha256=task_spec.task_spec_sha256,
        ),
    )
    lock = {
        "task_spec_version": task_spec.task_spec_version,
        "task_spec_sha256": task_spec.task_spec_sha256,
        "task_spec_file_sha256": task_file_sha256,
        "semantic_authoring_input_file_sha256": semantic_file_sha256,
        "semantic_authoring_input_sha256": semantic_input_sha256,
        "static_contract_refresh_receipt_file_sha256": refresh_file_sha256,
        "static_contract_refresh_receipt_sha256": refresh_sha256,
        "core_runtime_sources_receipt_file_sha256": core_file_sha256,
        "core_runtime_sources_receipt_sha256": core_sha256,
        "runtime_data_sha256": runtime_data_sha256,
    }
    runtime = SimpleNamespace(
        runtime_lock=lock,
        semantic_authoring_input=semantic,
        semantic_authoring_input_file_sha256=semantic_file_sha256,
        refresh_receipt_file_sha256=refresh_file_sha256,
        refresh_receipt={
            "receipt_sha256": refresh_sha256,
            "new_semantic_authoring_input_file_sha256": semantic_file_sha256,
            "new_semantic_authoring_input_sha256": semantic_input_sha256,
        },
        core_source_receipt_file_sha256=core_file_sha256,
        core_source_receipt={
            "receipt_sha256": core_sha256,
            "runtime_data_sha256": runtime_data_sha256,
        },
    )

    recovered, binding = analysis._load_static_runtime_task_specification(runtime)

    assert recovered == task_spec
    assert binding["semantic_authoring_input_sha256"] == semantic_input_sha256
    assert binding["static_contract_refresh_receipt_sha256"] == refresh_sha256
    with pytest.raises(ValueError, match="binding drifted"):
        analysis._load_static_runtime_task_specification(
            SimpleNamespace(
                **{
                    **vars(runtime),
                    "runtime_lock": {**lock, "task_spec_sha256": "f" * 64},
                }
            )
        )


def test_core_partition_layout_and_scopes_retain_frozen_split_names() -> None:
    query_splits = {
        "core-dev-1": "dev_mini",
        "core-dev-2": "dev_mini",
        "core-val-1": "val",
    }
    profile_inputs = SimpleNamespace(
        profile="core",
        selected_splits=("dev_mini", "val"),
        population_split_counts={"dev_mini": 2, "val": 1},
        split_by_query=query_splits,
    )
    plan = SimpleNamespace(
        selected_splits=("dev_mini", "val"),
        query_count=3,
        shards=(SimpleNamespace(query_ids=tuple(query_splits)),),
    )
    layout = analysis._build_partition_layout(
        plan=plan,
        manifest=SimpleNamespace(),
        profile_inputs=profile_inputs,
    )
    rows = _rectangle(3)
    split_by_ordinal = ("dev_mini", "dev_mini", "val")
    rows = [replace(row, partition=split_by_ordinal[row.query_ordinal]) for row in rows]

    scopes = analysis._scope_rows(
        rows,
        full_matrix=True,
        batch_id=None,
        layout=layout,
    )
    coverage = analysis._partition_coverage(rows, layout=layout)

    assert tuple(scopes) == ("dev_mini", "val", "all_selected")
    assert len(scopes["dev_mini"]) == 10
    assert len(scopes["val"]) == 5
    assert len(scopes["all_selected"]) == 15
    assert coverage == {
        "dev_mini": {
            "selected_query_count": 2,
            "expected_query_count": 2,
            "population_complete": True,
        },
        "val": {
            "selected_query_count": 1,
            "expected_query_count": 1,
            "population_complete": True,
        },
        "all_selected": {
            "selected_query_count": 3,
            "expected_query_count": 3,
            "population_complete": True,
        },
    }
    assert layout.primary_effect_scope == "not_selected"
    assert not ({"optimization25", "evaluation175", "all200"} & set(scopes))


def test_core_one_batch_canary_reports_real_split_as_incomplete_population() -> None:
    rows = [replace(row, partition="dev_mini") for row in _rectangle(3)]
    layout = analysis._PartitionLayout(
        profile="core",
        partition_order=("dev_mini",),
        partition_by_query={row.query_id: "dev_mini" for row in rows},
        expected_query_counts={"dev_mini": 200},
        aggregate_name="all_selected",
        primary_effect_scope="not_selected",
        use_policy={"primary_effect_scope": "not_selected"},
    )

    scopes = analysis._scope_rows(
        rows,
        full_matrix=False,
        batch_id="core-batch-001",
        layout=layout,
    )
    coverage = analysis._partition_coverage(rows, layout=layout)

    assert tuple(scopes) == ("dev_mini",)
    assert coverage["dev_mini"] == {
        "selected_query_count": 3,
        "expected_query_count": 200,
        "population_complete": False,
    }
    assert coverage["all_selected"]["population_complete"] is False


def test_audit_gate_accepts_review_flags_but_rejects_any_blocker() -> None:
    clean = {
        "batch_id": "batch-001",
        "status": "review_required",
        "query_count": 25,
        "row_count": 125,
        "blockers": [],
        "review_flags": ["recovered tool error"],
        "model_calls_performed": 0,
    }
    analysis._require_usable_audit(clean, batch_id="batch-001")

    analysis._require_usable_audit(
        {**clean, "query_count": 3, "row_count": 15},
        batch_id="batch-001",
    )
    core = {
        **clean,
        "dataset_profile": "core",
        "batch_split": "dev_mini",
    }
    analysis._require_usable_audit(
        core,
        batch_id="batch-001",
        expected_profile="core",
        expected_split="dev_mini",
    )
    with pytest.raises(ValueError, match="failed closed"):
        analysis._require_usable_audit(
            {**core, "batch_split": "optimization25"},
            batch_id="batch-001",
            expected_profile="core",
            expected_split="dev_mini",
        )

    with pytest.raises(ValueError, match="failed closed"):
        analysis._require_usable_audit(
            {**clean, "status": "failed", "blockers": ["drift"]},
            batch_id="batch-001",
        )


def test_adherence_contract_uses_frozen_gate_parent_banks() -> None:
    static = SimpleNamespace(bank_sha256="1" * 64)
    s1 = SimpleNamespace(bank_sha256="2" * 64)
    s1s2 = SimpleNamespace(bank_sha256="3" * 64)
    full = SimpleNamespace(bank_sha256="4" * 64)
    chain = SimpleNamespace(
        output_banks={
            "llm_static": static,
            "s1": s1,
            "s1s2": s1s2,
            "full": full,
        },
        gate_reports={
            "s1": SimpleNamespace(adherence_contract_bank_sha256=static.bank_sha256),
            "s1s2": SimpleNamespace(adherence_contract_bank_sha256=s1.bank_sha256),
            "full": SimpleNamespace(adherence_contract_bank_sha256=s1s2.bank_sha256),
        },
    )

    contracts = analysis._adherence_contract_banks(chain)

    assert contracts == {
        "llm_static": static,
        "s1": static,
        "s1s2": s1,
        "full": s1s2,
    }


def test_adherence_contract_translates_verified_runtime_rebind_lineage() -> None:
    source_banks = {
        "llm_static": SimpleNamespace(bank_sha256="1" * 64),
        "s1": SimpleNamespace(bank_sha256="2" * 64),
        "s1s2": SimpleNamespace(bank_sha256="3" * 64),
        "full": SimpleNamespace(bank_sha256="4" * 64),
    }
    rebound_banks = {
        "llm_static": SimpleNamespace(bank_sha256="a" * 64),
        "s1": SimpleNamespace(bank_sha256="b" * 64),
        "s1s2": SimpleNamespace(bank_sha256="c" * 64),
        "full": SimpleNamespace(bank_sha256="d" * 64),
    }
    gate_reports = {
        "s1": SimpleNamespace(
            adherence_contract_bank_sha256=source_banks["llm_static"].bank_sha256
        ),
        "s1s2": SimpleNamespace(
            adherence_contract_bank_sha256=source_banks["s1"].bank_sha256
        ),
        "full": SimpleNamespace(
            adherence_contract_bank_sha256=source_banks["s1s2"].bank_sha256
        ),
    }
    source_chain = SimpleNamespace(
        output_banks=source_banks,
        gate_reports=gate_reports,
    )
    bindings = tuple(
        SimpleNamespace(
            config=config,
            role="output",
            source_bank_sha256=source_banks[config].bank_sha256,
            rebound_bank_sha256=rebound_banks[config].bank_sha256,
        )
        for config in source_banks
    )
    chain = SimpleNamespace(
        output_banks=rebound_banks,
        gate_reports=gate_reports,
        source_chain=source_chain,
        compatibility_rebind=SimpleNamespace(bank_bindings=bindings),
    )

    contracts = analysis._adherence_contract_banks(chain)

    assert contracts == {
        "llm_static": rebound_banks["llm_static"],
        "s1": rebound_banks["llm_static"],
        "s1s2": rebound_banks["s1"],
        "full": rebound_banks["s1s2"],
    }


def test_adherence_contract_rejects_tampered_runtime_rebind_lineage() -> None:
    source_bank = SimpleNamespace(bank_sha256="1" * 64)
    rebound_bank = SimpleNamespace(bank_sha256="a" * 64)
    gate_report = SimpleNamespace(adherence_contract_bank_sha256="1" * 64)
    source_chain = SimpleNamespace(
        output_banks={
            "llm_static": source_bank,
            "s1": source_bank,
            "s1s2": source_bank,
            "full": source_bank,
        },
        gate_reports={config: gate_report for config in ("s1", "s1s2", "full")},
    )
    chain = SimpleNamespace(
        output_banks={
            "llm_static": rebound_bank,
            "s1": rebound_bank,
            "s1s2": rebound_bank,
            "full": rebound_bank,
        },
        gate_reports=source_chain.gate_reports,
        source_chain=source_chain,
        compatibility_rebind=SimpleNamespace(
            bank_bindings=(
                SimpleNamespace(
                    config="llm_static",
                    role="output",
                    source_bank_sha256="f" * 64,
                    rebound_bank_sha256=rebound_bank.bank_sha256,
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="s1 gate adherence contract Bank"):
        analysis._adherence_contract_banks(chain)


def test_output_guard_refuses_execution_launch_and_runtime_descendants(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "execution"
    launch = tmp_path / "launch"
    runtime = tmp_path / "runtime"
    for root in (execution, launch, runtime):
        root.mkdir()

    for output in (
        execution / "analysis",
        launch / "analysis",
        runtime / "analysis",
    ):
        with pytest.raises(ValueError, match="outside"):
            analysis._require_external_output(
                output,
                execution_root=execution,
                launch_root=launch,
                runtime_root=runtime,
            )

    valid = tmp_path / "analysis"
    analysis._require_external_output(
        valid,
        execution_root=execution,
        launch_root=launch,
        runtime_root=runtime,
    )
    valid.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        analysis._require_external_output(
            valid,
            execution_root=execution,
            launch_root=launch,
            runtime_root=runtime,
        )


def test_external_analysis_source_uses_strict_auditor_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_shard = SimpleNamespace(
        shard_id="00-batch-001-noskill",
        config="noskill",
        accepted_batch_id="batch-001",
    )
    target_member = SimpleNamespace(query_id="q-001", shard_id=target_shard.shard_id)
    launch = SimpleNamespace(
        plan=SimpleNamespace(shards=(target_shard,)),
        instances=(target_member,),
    )
    source_member = SimpleNamespace(query_id="q-001")
    source = SimpleNamespace(
        execution_root=tmp_path / "source-execution",
        shard=SimpleNamespace(
            shard_id="00-source-noskill",
            config="noskill",
            output_relpath="shards/00-source-noskill",
        ),
        members=(source_member,),
        external=True,
    )
    evidence = {
        "target_shard_id": target_shard.shard_id,
        "source_execution_root": "source-execution",
        "source_shard_id": "00-source-noskill",
        "source_artifact_set_sha256": "a" * 64,
    }
    observed: dict[str, object] = {}

    def fake_load_external_frozen_source(**kwargs: object):
        observed.update(kwargs)
        return source, evidence, []

    monkeypatch.setattr(
        analysis,
        "_load_external_frozen_source",
        fake_load_external_frozen_source,
    )

    sources, resolved_evidence = analysis._external_analysis_sources(
        control={
            "external_frozen_shard": {
                "accepted_batch_id": "batch-001",
            }
        },
        launch=launch,
        selected_batch_ids=("batch-001",),
    )

    assert sources == {target_shard.shard_id: source}
    assert resolved_evidence == evidence
    assert observed["target_members"] == (target_member,)
    root, physical_shard, members = analysis._artifact_projection(
        execution_root=tmp_path / "repair-execution",
        target_shard=target_shard,
        target_members=(target_member,),
        external_sources_by_shard=sources,
    )
    assert root == source.execution_root
    assert physical_shard == source.shard
    assert members == {"q-001": source_member}


def test_artifact_projection_reads_full_rollback_alias_from_parent_shard(
    tmp_path: Path,
) -> None:
    target = SimpleNamespace(shard_id="core-full", config="full")
    target_members = (
        SimpleNamespace(query_id="core-q-1"),
        SimpleNamespace(query_id="core-q-2"),
    )
    source_shard = SimpleNamespace(
        shard_id="core-s1s2",
        config="s1s2",
        output_relpath="shards/core-s1s2",
    )
    source_members = (
        SimpleNamespace(query_id="core-q-2"),
        SimpleNamespace(query_id="core-q-1"),
    )
    source = SimpleNamespace(
        execution_root=tmp_path,
        shard=source_shard,
        members=source_members,
        external=False,
        artifact_alias={"provider_model_call_count": 0},
    )

    root, physical_shard, by_query = analysis._artifact_projection(
        execution_root=tmp_path / "logical",
        target_shard=target,
        target_members=target_members,
        external_sources_by_shard={target.shard_id: source},
    )

    assert root == tmp_path
    assert physical_shard is source_shard
    assert tuple(sorted(by_query)) == ("core-q-1", "core-q-2")


def test_external_analysis_requires_one_matching_batch() -> None:
    launch = SimpleNamespace(plan=SimpleNamespace(shards=()), instances=())

    with pytest.raises(ValueError, match="one explicit matching batch"):
        analysis._external_analysis_sources(
            control={
                "external_frozen_shard": {
                    "accepted_batch_id": "batch-001",
                }
            },
            launch=launch,
            selected_batch_ids=("batch-001", "batch-002"),
        )


def test_external_analysis_rejects_auditor_compatibility_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_shard = SimpleNamespace(
        shard_id="00-batch-001-noskill",
        config="noskill",
        accepted_batch_id="batch-001",
    )
    launch = SimpleNamespace(
        plan=SimpleNamespace(shards=(target_shard,)),
        instances=(),
    )
    monkeypatch.setattr(
        analysis,
        "_load_external_frozen_source",
        lambda **_kwargs: (
            SimpleNamespace(),
            {"target_shard_id": target_shard.shard_id},
            ["query drift"],
        ),
    )

    with pytest.raises(ValueError, match="query drift"):
        analysis._external_analysis_sources(
            control={
                "external_frozen_shard": {
                    "accepted_batch_id": "batch-001",
                }
            },
            launch=launch,
            selected_batch_ids=("batch-001",),
        )


def test_external_report_discloses_direct_source_without_copying() -> None:
    coverage = {
        name: {
            "selected_query_count": 25 if name != "evaluation175" else 0,
            "expected_query_count": expected,
            "population_complete": name == "optimization25",
        }
        for name, expected in (
            ("optimization25", 25),
            ("evaluation175", 175),
            ("all200", 200),
        )
    }

    report = analysis._report(
        selection="batch:batch-001",
        partition_coverage=coverage,
        config_summaries=(),
        paired_summaries=(),
        no_op_contrasts=(),
        audit_records=(),
        external_frozen_shard={
            "source_execution_root": "source-execution",
            "source_shard_id": "00-source-noskill",
            "source_artifact_set_sha256": "a" * 64,
        },
    ).decode("utf-8")

    assert "read directly" in report
    assert "no artifacts were copied" in report
    assert "source-execution" in report
