from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_portfolio_batch as batch_audit
from skillchain.evaluation.assistant_runs import (
    AssistantModelCallReceipt,
    make_assistant_route_call_evidence,
)
from skillchain.evaluation.portfolio_execution import (
    ProviderPreResponseCircuitBreaker,
    create_retryable_attempt_receipt,
)
from skillchain.evaluation.portfolio_launch import MAIN_CONFIG_ORDER
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _route_evidence():
    call = AssistantModelCallReceipt(
        call_index=1,
        provider="qwen",
        endpoint="https://fixture.invalid/v1",
        requested_model="model",
        response_model="model",
        provider_request_id="fixture-route",
        input_tokens=10,
        output_tokens=2,
        finish_reason="stop",
        latency_ms=1,
        response_sha256="b" * 64,
    )
    return make_assistant_route_call_evidence(
        attempt_index=1,
        wire_request_sha256="c" * 64,
        route_schema_sha256="d" * 64,
        response_text="{not-json",
        call_receipt=call,
        failure_reason="invalid_route_json",
        payload_status="invalid_json",
    )


@dataclass(frozen=True)
class _Shard:
    shard_id: str
    accepted_batch_id: str
    config: str
    query_count: int = 25
    output_relpath: str = ""


@dataclass(frozen=True)
class _Member:
    query_id: str
    query_ordinal: int
    query_sha256: str
    public_input_sha256: str
    asset_id: str
    image_path: str
    image_sha256: str
    accepted_batch_id: str
    shard_id: str
    assistant_output_relpath: str
    final_output_relpath: str


def _shards() -> tuple[_Shard, ...]:
    return tuple(
        _Shard(
            shard_id=f"{index:02d}-batch-001-{config}",
            accepted_batch_id="batch-001",
            config=config,
            output_relpath=f"shards/{index:02d}-batch-001-{config}",
        )
        for index, config in enumerate(MAIN_CONFIG_ORDER)
    )


def _members(shards: tuple[_Shard, ...]) -> tuple[_Member, ...]:
    rows: list[_Member] = []
    for shard in shards:
        for index in range(25):
            query_id = f"q-{index:03d}"
            rows.append(
                _Member(
                    query_id=query_id,
                    query_ordinal=index,
                    query_sha256=f"{index:064x}",
                    public_input_sha256=f"{index + 100:064x}",
                    asset_id=f"asset-{index}",
                    image_path=f"images/{index}.jpg",
                    image_sha256=f"{index + 200:064x}",
                    accepted_batch_id="batch-001",
                    shard_id=shard.shard_id,
                    assistant_output_relpath=(
                        f"{shard.output_relpath}/assistant/{query_id}.json"
                    ),
                    final_output_relpath=(
                        f"{shard.output_relpath}/final/{query_id}.json"
                    ),
                )
            )
    return tuple(rows)


def test_selected_attempt_usage_counts_only_uncaptured_pre_response_call(
    tmp_path: Path,
) -> None:
    shard = _Shard(
        shard_id="00-batch-001-s1",
        accepted_batch_id="batch-001",
        config="s1",
        output_relpath="shards/00-batch-001-s1",
    )
    shard_root = tmp_path / shard.output_relpath
    breaker = ProviderPreResponseCircuitBreaker()
    common = {
        "shard_root": shard_root,
        "breaker": breaker,
        "matrix_run_id": "portfolio-mini",
        "shard_id": shard.shard_id,
        "config": shard.config,
        "request_sha256": "a" * 64,
    }
    create_retryable_attempt_receipt(
        **common,
        instance_sha256="1" * 64,
        query_id="query-provider-pre-response",
        query_ordinal=0,
        failure_stage="shared_route",
        failure_subtype="provider_pre_response",
        circuit_id="assistant:qwen:model",
        forfeited_reservation_sha256="4" * 64,
        budget_forfeit_sha256="5" * 64,
        exception_type="AssistantProviderPreResponseError",
    )
    create_retryable_attempt_receipt(
        **common,
        instance_sha256="2" * 64,
        query_id="query-route-contract",
        query_ordinal=1,
        failure_stage="assistant_route",
        failure_subtype="route_contract_invalid_json",
        circuit_id="assistant_route_contract:qwen:model",
        captured_provider_response_count=1,
        captured_input_tokens=10,
        captured_output_tokens=2,
        route_call_evidence=_route_evidence(),
    )
    create_retryable_attempt_receipt(
        **common,
        instance_sha256="3" * 64,
        query_id="query-settled-orphan",
        query_ordinal=2,
        failure_stage="assistant_action",
        failure_subtype="orphaned_provider_call",
        circuit_id="orphaned:qwen:model",
        captured_provider_response_count=1,
        captured_input_tokens=12,
        captured_output_tokens=3,
        exception_type="PortfolioBudgetOrphanedCallError",
    )

    usage, artifacts = batch_audit._selected_attempt_usage(
        tmp_path,
        (shard,),
        sources_by_config={},
    )

    assert usage == {
        "receipt_count": 3,
        "model_call_count": 3,
        "assistant_input_tokens": 22,
        "assistant_output_tokens": 5,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
    }
    assert len(artifacts) == 3
    assert batch_audit._selected_batch_cost(
        {},
        qwen_input_rate=0.15,
        qwen_output_rate=1.5,
        judge_input_rate=6.5,
        judge_output_rate=27.0,
        assistant_attempt_input_tokens=usage["assistant_input_tokens"],
        assistant_attempt_output_tokens=usage["assistant_output_tokens"],
    ) == pytest.approx(((22 * 0.15) + (5 * 1.5)) / 1_000_000)


def _observation(
    *,
    query_id: str,
    config: str,
) -> batch_audit._RowObservation:
    return batch_audit._RowObservation(
        query_id=query_id,
        config=config,
        canonical_capability="product.exact_match",
        card_requirement="required",
        visible_card_count=1,
        card_policy_compliant=True,
        assistant_error_code=None,
        receipt_outcome="success",
        model_call_count=2 if config != "noskill" else 1,
        finish_reasons=("stop",),
        turn_count=1,
        tool_trace=(),
        selected_capability=("product.exact_match" if config != "noskill" else None),
        skill_slug=f"{config}-exact" if config != "noskill" else None,
        route_trace_sha256="a" * 64 if config != "noskill" else None,
        route_call_response_sha256=(
            f"{MAIN_CONFIG_ORDER.index(config) + 1:064x}"
            if config != "noskill"
            else None
        ),
        shared_route_artifact_sha256=None,
        shared_route_reserved_input_tokens=None,
        shared_route_reserved_output_tokens=None,
        shared_route_reserved_turns=0,
        shared_route_status=None,
        route_acceptable=True if config != "noskill" else None,
        assistant_input_tokens=100,
        assistant_output_tokens=20,
        judge_status="scored",
        judge_error_code=None,
        judge_shape="assessment_array",
        judge_finish_reason="stop",
        judge_raw_response_bytes=100,
        judge_tool_call_count=0,
        judge_input_tokens=50,
        judge_output_tokens=10,
        judge_attempts=1,
        judge_initial_empty_response=False,
        judge_terminal_response_captured=True,
        judge_initial_reasoning_present=None,
        judge_initial_reasoning_tokens=None,
        judge_initial_reasoning_bytes=None,
        judge_initial_reasoning_sha256=None,
        judge_terminal_reasoning_present=False,
        judge_terminal_reasoning_tokens=None,
        judge_terminal_reasoning_bytes=0,
        judge_terminal_reasoning_sha256=None,
        j_project=80.0,
    )


def test_content_address_is_stable_and_excludes_its_own_field() -> None:
    audit = batch_audit._content_address({"kind": "test-audit", "values": [1, 2, 3]})
    unsigned = dict(audit)
    supplied = unsigned.pop("audit_sha256")

    assert supplied == sha256_bytes(canonical_json_bytes(unsigned))
    assert batch_audit._content_address(audit) == audit


def test_core_profile_inputs_reconstruct_from_launch_without_dev_mini_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queries = tuple(
        SimpleNamespace(query_id=f"core-{index:02d}", split="dev_mini")
        for index in range(25)
    )
    files = SimpleNamespace(
        capability_assignments_path=tmp_path / "core-assignments.jsonl",
        runtime_catalog_dir=tmp_path / "runtime-catalog",
        base_catalog_dir=tmp_path / "base-catalog",
        asset_root=tmp_path / "assets",
    )
    plan = SimpleNamespace(
        dataset_profile="core",
        kind="portfolio-core-split-x5-launch-plan",
        selected_splits=("dev_mini",),
        query_count=25,
        shards=(SimpleNamespace(query_ids=tuple(item.query_id for item in queries)),),
    )
    monkeypatch.setattr(
        batch_audit,
        "reconstruct_verified_portfolio_core_inputs",
        lambda _plan: SimpleNamespace(queries=queries, files=files),
    )
    monkeypatch.setattr(
        batch_audit,
        "_load_active_inputs",
        lambda: (_ for _ in ()).throw(
            AssertionError("Core must not load dev_mini inputs")
        ),
    )

    resolved = batch_audit._load_launch_profile_inputs(SimpleNamespace(plan=plan))

    assert resolved.profile == "core"
    assert resolved.queries == queries
    assert resolved.selected_splits == ("dev_mini",)
    assert resolved.split_by_query["core-00"] == "dev_mini"
    assert resolved.population_split_counts == {"dev_mini": 25}
    assert resolved.capability_assignments_path == files.capability_assignments_path
    assert resolved.catalog_dir == files.runtime_catalog_dir
    assert resolved.asset_root == files.asset_root
    assert (
        batch_audit._require_launch_profile_inputs(SimpleNamespace(plan=plan), resolved)
        is resolved
    )
    with pytest.raises(ValueError, match="differ from launch"):
        batch_audit._require_launch_profile_inputs(
            SimpleNamespace(
                plan=SimpleNamespace(
                    **{
                        **vars(plan),
                        "shards": (SimpleNamespace(query_ids=("other-query",)),),
                    }
                )
            ),
            resolved,
        )


def test_assignment_index_allows_one_asset_across_distinct_intents() -> None:
    exact = SimpleNamespace(
        assignment_id="assignment-exact",
        asset_id="asset-shared",
        image_path="images/shared.jpg",
        canonical_intent="exact_match",
    )
    recommendation = SimpleNamespace(
        assignment_id="assignment-recommendation",
        asset_id="asset-shared",
        image_path="images/shared.jpg",
        canonical_intent="divergent_rec",
    )

    indexed = batch_audit._index_assignments((exact, recommendation))

    assert indexed[("asset-shared", "images/shared.jpg", "exact_match")] is exact
    assert (
        indexed[("asset-shared", "images/shared.jpg", "divergent_rec")]
        is recommendation
    )


def test_assignment_index_rejects_duplicate_composite_key() -> None:
    first = SimpleNamespace(
        assignment_id="assignment-one",
        asset_id="asset-shared",
        image_path="images/shared.jpg",
        canonical_intent="exact_match",
    )
    duplicate = SimpleNamespace(
        assignment_id="assignment-two",
        asset_id="asset-shared",
        image_path="images/shared.jpg",
        canonical_intent="exact_match",
    )

    with pytest.raises(ValueError, match="duplicate composite keys"):
        batch_audit._index_assignments((first, duplicate))


def test_observed_evaluator_input_sources_are_hashed_without_lock_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packets = tmp_path / "packets.py"
    isolation = tmp_path / "evaluator_isolation.py"
    config = tmp_path / "config.py"
    packets.write_bytes(b"packets-v1\n")
    isolation.write_bytes(b"isolation-v1\n")
    config.write_bytes(b"config-v1\n")
    monkeypatch.setattr(
        batch_audit,
        "_OBSERVED_SOURCE_FILES",
        {
            "src/skillchain/evaluation/packets.py": packets,
            "src/skillchain/evaluation/evaluator_isolation.py": isolation,
            "src/skillchain/config.py": config,
        },
    )

    observed = batch_audit._observed_source_file_sha256s()

    assert observed == {
        "src/skillchain/config.py": sha256_bytes(b"config-v1\n"),
        "src/skillchain/evaluation/evaluator_isolation.py": sha256_bytes(
            b"isolation-v1\n"
        ),
        "src/skillchain/evaluation/packets.py": sha256_bytes(b"packets-v1\n"),
    }


def test_selects_one_complete_five_config_batch_in_canonical_order() -> None:
    shards = tuple(reversed(_shards()))

    batch_id, selected, launch_sequence = batch_audit._select_batch_shards(
        shards,
        batch_id="batch-001",
        shard_ids=None,
    )

    assert batch_id == "batch-001"
    assert tuple(item.config for item in selected) == MAIN_CONFIG_ORDER
    assert tuple(item.config for item in launch_sequence) == tuple(
        reversed(MAIN_CONFIG_ORDER)
    )


def test_core_canary_batch_selection_does_not_require_other_batches_complete() -> None:
    canary = _shards()
    later = tuple(
        replace(
            shard,
            shard_id=shard.shard_id.replace("batch-001", "batch-002"),
            accepted_batch_id="batch-002",
            output_relpath=shard.output_relpath.replace("batch-001", "batch-002"),
        )
        for shard in canary
    )

    batch_id, selected, launch_sequence = batch_audit._select_batch_shards(
        (*canary, *later),
        batch_id="batch-001",
        shard_ids=None,
    )

    assert batch_id == "batch-001"
    assert selected == canary
    assert launch_sequence == canary


def test_explicit_shard_selection_fails_closed_when_not_one_by_five() -> None:
    shards = _shards()

    with pytest.raises(ValueError, match="exactly five"):
        batch_audit._select_batch_shards(
            shards,
            batch_id=None,
            shard_ids=[item.shard_id for item in shards[:4]],
        )


def test_batch_selection_rejects_noncontiguous_launch_sequence() -> None:
    shards = list(_shards())
    shards.insert(
        2,
        _Shard(
            shard_id="99-batch-002-noskill",
            accepted_batch_id="batch-002",
            config="noskill",
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact contiguous five-shard launch sequence",
    ):
        batch_audit._select_batch_shards(
            shards,
            batch_id="batch-001",
            shard_ids=None,
        )


def test_completion_snapshot_treats_full_rollback_alias_as_logically_complete(
    tmp_path: Path,
) -> None:
    source_shard = _Shard(
        shard_id="000-core-batch-s1s2",
        accepted_batch_id="core-batch",
        config="s1s2",
        output_relpath="shards/000-core-batch-s1s2",
    )
    target_shard = _Shard(
        shard_id="001-core-batch-full",
        accepted_batch_id="core-batch",
        config="full",
        output_relpath="shards/001-core-batch-full",
    )
    source_members = tuple(
        SimpleNamespace(
            query_id=f"core-{index:02d}",
            assistant_output_relpath=(
                f"{source_shard.output_relpath}/assistant/core-{index:02d}.json"
            ),
            final_output_relpath=(
                f"{source_shard.output_relpath}/final/core-{index:02d}.json"
            ),
        )
        for index in range(25)
    )
    target_members = tuple(
        SimpleNamespace(
            query_id=item.query_id,
            assistant_output_relpath=(
                f"{target_shard.output_relpath}/assistant/{item.query_id}.json"
            ),
            final_output_relpath=(
                f"{target_shard.output_relpath}/final/{item.query_id}.json"
            ),
        )
        for item in source_members
    )
    source_root = tmp_path / source_shard.output_relpath
    for directory in (source_root / "assistant", source_root / "final"):
        directory.mkdir(parents=True, exist_ok=True)
        for member in source_members:
            (directory / f"{member.query_id}.json").write_bytes(b"{}\n")
    (source_root / "shard-summary.json").write_bytes(b"{}\n")
    (source_root / "shard-audit.json").write_bytes(b"{}\n")
    alias_source = batch_audit._ShardArtifactSource(
        execution_root=tmp_path,
        launch=SimpleNamespace(),
        shard=source_shard,
        members=source_members,
        control={},
        runtime_root=tmp_path,
        runtime_raw={},
        artifact_alias={"provider_model_call_count": 0},
    )

    snapshot, blockers = batch_audit._completion_snapshot(
        tmp_path,
        (target_shard,),
        {"full": target_members},
        sources_by_config={"full": alias_source},
    )

    assert blockers == []
    assert snapshot["full"]["artifact_source"] == "accepted_parent_artifact_alias"
    assert snapshot["full"]["physical_shard_id"] == source_shard.shard_id
    assert snapshot["full"]["assistant_checkpoint_count"] == 25
    assert snapshot["full"]["final_checkpoint_count"] == 25


def test_internal_full_alias_resolves_with_zero_provider_calls(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    source = SimpleNamespace(
        shard_id="000-core-batch-s1s2",
        accepted_batch_id="core-batch",
        config="s1s2",
        query_ids=("core-q-1", "core-q-2"),
        output_relpath="shards/000-core-batch-s1s2",
    )
    target = SimpleNamespace(
        shard_id="001-core-batch-full",
        accepted_batch_id="core-batch",
        config="full",
        query_ids=source.query_ids,
        output_relpath="shards/001-core-batch-full",
    )
    common_bank_sha = "a" * 64
    common_bank_file_sha = "b" * 64
    alias_unsigned = {
        "target_config": "full",
        "source_config": "s1s2",
        "provider_model_call_count": 0,
        "reuse_scope": "assistant_and_evaluator_query_artifacts",
        "rejected_candidate_use": "diagnostic_only",
        "source_bank_sha256": common_bank_sha,
        "target_bank_sha256": common_bank_sha,
        "source_bank_file_sha256": common_bank_file_sha,
        "target_bank_file_sha256": common_bank_file_sha,
    }
    alias = {**alias_unsigned, "alias_sha256": batch_audit._hash(alias_unsigned)}
    policy = "portfolio-execution-artifact-alias-v1"
    control = {
        "execution_artifact_aliases": [alias],
        "execution_artifact_alias_policy_version": policy,
        "execution_artifact_alias_provider_model_call_count": 0,
        "runtime_lock_sha256": "c" * 64,
        "launch_plan_sha256": "d" * 64,
    }
    (runtime_root / "runtime-lock.json").write_bytes(
        canonical_json_bytes(
            {
                "execution_artifact_aliases": [alias],
                "execution_artifact_alias_policy_version": policy,
                "execution_artifact_alias_provider_model_call_count": 0,
            }
        )
    )
    source_root = tmp_path / source.output_relpath
    source_root.mkdir(parents=True)
    summary = {"summary_sha256": "e" * 64}
    audit = {"audit_sha256": "f" * 64}
    summary_bytes = canonical_json_bytes(summary)
    audit_bytes = canonical_json_bytes(audit)
    (source_root / "shard-summary.json").write_bytes(summary_bytes)
    (source_root / "shard-audit.json").write_bytes(audit_bytes)
    target_root = tmp_path / target.output_relpath
    target_root.mkdir(parents=True)
    receipt_unsigned = {
        "kind": "portfolio-shard-artifact-alias",
        "target_shard_id": target.shard_id,
        "target_config": target.config,
        "source_shard_id": source.shard_id,
        "source_config": source.config,
        "query_ids": list(target.query_ids),
        "treatment_alias_sha256": alias["alias_sha256"],
        "provider_model_call_count": 0,
        "runtime_lock_sha256": control["runtime_lock_sha256"],
        "launch_plan_sha256": control["launch_plan_sha256"],
        "source_shard_summary_file_sha256": sha256_bytes(summary_bytes),
        "source_shard_summary_sha256": summary["summary_sha256"],
        "source_shard_audit_file_sha256": sha256_bytes(audit_bytes),
        "source_shard_audit_sha256": audit["audit_sha256"],
    }
    receipt = {
        **receipt_unsigned,
        "alias_receipt_sha256": batch_audit._hash(receipt_unsigned),
    }
    (target_root / "artifact-alias.json").write_bytes(canonical_json_bytes(receipt))
    source_members = tuple(
        SimpleNamespace(query_id=query_id) for query_id in source.query_ids
    )
    target_members = tuple(
        SimpleNamespace(query_id=query_id) for query_id in target.query_ids
    )
    launch = SimpleNamespace(
        state=SimpleNamespace(
            completed_shard_ids=frozenset({source.shard_id, target.shard_id})
        )
    )

    sources, evidence = batch_audit._load_internal_artifact_alias_sources(
        control=control,
        launch=launch,
        execution_root=tmp_path,
        runtime_root=runtime_root,
        shards=(source, target),
        members_by_config={"s1s2": source_members, "full": target_members},
    )

    assert set(sources) == {"full"}
    assert sources["full"].shard is source
    assert sources["full"].artifact_alias == receipt
    assert evidence[0]["provider_model_call_count"] == 0


def _boundary_bank(
    *,
    description: str,
    body: str,
    operators: tuple[str, ...] = ("document_ocr",),
) -> SimpleNamespace:
    return SimpleNamespace(
        skills=(
            SimpleNamespace(
                slug=f"lineage-{sha256_bytes((description + body).encode())[:8]}",
                version=1,
                capability_id="utility.document_reading",
                description=description,
                body=body,
                static_refs=(),
                operators=operators,
                parent_skill_sha256=None,
                skill_sha256="a" * 64,
            ),
        )
    )


def test_stage_boundary_audit_reports_s2_and_s3_drift() -> None:
    s1_body = "# Body\n\nRead the document.\n"
    s2_body = "# Body\n\nS2 illegally changed execution.\n"
    check = batch_audit._stage_boundary_check(
        {
            "s1": _boundary_bank(
                description="Initial route.",
                body=s1_body,
            ),
            "s1s2": _boundary_bank(
                description="Optimized route.",
                body=s2_body,
            ),
            "full": _boundary_bank(
                description="S3 illegally changed routing.",
                body="# Body\n\nRefined response procedure.\n",
            ),
        }
    )

    assert check == {
        "policy_version": "s2-description-s3-body-only-v1",
        "violation_count": 2,
        "violations": [
            "S2 may only change Description; utility.document_reading changed body",
            "S3 may only change Body; utility.document_reading changed description",
        ],
    }


def test_paired_input_check_detects_one_config_drift() -> None:
    shards = _shards()
    members = _members(shards)
    by_config = {
        config: tuple(
            item
            for item in members
            if item.shard_id
            == next(shard.shard_id for shard in shards if shard.config == config)
        )
        for config in MAIN_CONFIG_ORDER
    }
    drifted = list(by_config["s1"])
    drifted[3] = replace(drifted[3], image_sha256="f" * 64)
    by_config["s1"] = tuple(drifted)

    violations = batch_audit._paired_member_violations(by_config)

    assert violations == ["paired input drift for q-003/s1: image_sha256"]


def test_operator_map_comparison_rejects_skilled_access_drift() -> None:
    reference = {
        "product.exact_match": (
            "image_product_search",
            "text_product_search",
        )
    }
    maps = {config: dict(reference) for config in MAIN_CONFIG_ORDER[1:]}
    maps["full"] = {"product.exact_match": ("image_product_search",)}

    matrix, violations = batch_audit._compare_operator_maps(maps)

    assert matrix["product.exact_match"] == [
        "image_product_search",
        "text_product_search",
    ]
    assert violations == [
        "full Bank operator sets differ from llm_static by capability"
    ]


def test_tool_gate_reports_out_of_bank_operator_but_not_route_quality() -> None:
    skill = SimpleNamespace(
        slug="s1-exact",
        capability_id="product.exact_match",
        operators=("image_product_search",),
    )
    response = SimpleNamespace(
        selected_capability="product.exact_match",
        skill_slug="s1-exact",
        error_code=None,
        tool_trace=(SimpleNamespace(tool_name="text_product_search"),),
    )
    assignment = SimpleNamespace(
        canonical_capability="product.style_recommendation",
        acceptable_capabilities=("product.style_recommendation",),
        allowed_tools=("style_similar_search",),
    )

    violations = batch_audit._tool_gate_violations(
        query_id="q-001",
        config="s1",
        response=response,
        assignment=assignment,
        bank_context={"by_slug": {"s1-exact": skill}},
    )

    assert violations == ["tool trace exceeds selected operators: q-001/s1"]


def test_summary_surfaces_skilled_only_error_and_judge_anomaly() -> None:
    rows = [
        _observation(query_id=f"q-{index:03d}", config=config)
        for config in MAIN_CONFIG_ORDER
        for index in range(25)
    ]
    s1_index = next(
        index
        for index, row in enumerate(rows)
        if row.query_id == "q-002" and row.config == "s1"
    )
    rows[s1_index] = replace(
        rows[s1_index],
        assistant_error_code="runtime_error",
        receipt_outcome="runtime_error",
        judge_status="not_invoked_assistant_error",
        judge_shape=None,
        j_project=0.0,
    )
    llm_static_index = next(
        index
        for index, row in enumerate(rows)
        if row.query_id == "q-003" and row.config == "llm_static"
    )
    rows[llm_static_index] = replace(
        rows[llm_static_index],
        selected_capability="product.multi_search",
        skill_slug="llm_static-multi",
        route_acceptable=False,
        tool_trace=(
            ("text_product_search", "error", "context_violation"),
            ("image_product_search", "success", None),
        ),
        visible_card_count=0,
        card_policy_compliant=False,
    )
    full_index = next(
        index
        for index, row in enumerate(rows)
        if row.query_id == "q-004" and row.config == "full"
    )
    rows[full_index] = replace(
        rows[full_index],
        judge_status="parse_error",
        judge_error_code="invalid_judge_json",
        judge_shape=None,
        judge_raw_response_bytes=0,
        judge_attempts=2,
        judge_initial_empty_response=True,
        judge_initial_retry_reason="empty_final_response",
        judge_initial_reasoning_present=True,
        judge_initial_reasoning_tokens=4,
        judge_initial_reasoning_bytes=8,
        judge_initial_reasoning_sha256="f" * 64,
        j_project=0.0,
    )

    (
        summaries,
        skilled_errors,
        judge_anomalies,
        route_performance_observations,
        judge_response_receipts,
    ) = batch_audit._summarize_observations(rows)

    assert summaries["s1"]["assistant_error_counts"] == {"runtime_error": 1}
    assert summaries["llm_static"]["successful_route_count"] == 25
    assert summaries["llm_static"]["acceptable_route_count"] == 24
    assert summaries["llm_static"]["unacceptable_route_count"] == 1
    assert summaries["llm_static"]["assistant_tool_calls"] == 2
    assert summaries["llm_static"]["assistant_tool_error_count"] == 1
    assert summaries["llm_static"]["assistant_tool_error_query_count"] == 1
    assert summaries["llm_static"]["assistant_tool_error_ids"] == ["q-003"]
    assert summaries["llm_static"]["assistant_tool_error_counts_by_code"] == {
        "context_violation": 1
    }
    assert summaries["llm_static"]["assistant_tool_error_counts_by_tool"] == {
        "text_product_search": 1
    }
    assert summaries["llm_static"]["card_policy_compliant_count"] == 24
    assert summaries["llm_static"]["card_policy_violation_count"] == 1
    assert summaries["llm_static"]["card_policy_violation_ids"] == ["q-003"]
    assert summaries["llm_static"]["required_missing_card_ids"] == ["q-003"]
    assert summaries["llm_static"]["forbidden_card_ids"] == []
    assert skilled_errors[0]["query_id"] == "q-002"
    assert skilled_errors[0]["skilled_only_vs_noskill"] is True
    assert route_performance_observations == [
        {
            "query_id": "q-003",
            "config": "llm_static",
            "canonical_capability": "product.exact_match",
            "selected_capability": "product.multi_search",
            "skill_slug": "llm_static-multi",
            "observation": "selected_capability_not_acceptable",
        }
    ]
    assert judge_anomalies == [
        {
            "query_id": "q-004",
            "config": "full",
            "status": "parse_error",
            "error_code": "invalid_judge_json",
            "finish_reason": "stop",
            "raw_response_bytes": 0,
            "tool_call_count": 0,
            "attempts": 2,
            "initial_empty_response": True,
            "initial_retry_reason": "empty_final_response",
            "terminal_response_captured": True,
            "initial_reasoning_present": True,
            "initial_reasoning_tokens": 4,
            "initial_reasoning_bytes": 8,
            "initial_reasoning_sha256": "f" * 64,
            "terminal_reasoning_present": False,
            "terminal_reasoning_tokens": None,
            "terminal_reasoning_bytes": 0,
            "terminal_reasoning_sha256": None,
            "j_project": 0.0,
        }
    ]
    assert len(judge_response_receipts) == 125
    assert summaries["full"]["judge_model_calls"] == 26
    assert summaries["full"]["judge_retried_row_count"] == 1
    assert summaries["full"]["judge_initial_empty_response_count"] == 1
    assert summaries["full"]["judge_reasoning_present_response_count"] == 1
    assert summaries["full"]["judge_reasoning_tokens_reported_total"] == 4
    assert summaries["full"]["judge_reasoning_bytes_total"] == 8


def test_shard_audit_validation_checks_self_and_execution_bindings(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 3,
        "kind": "portfolio-shard-audit",
        "formal_eligible": False,
        "shard_id": "00-batch-001-noskill",
        "config": "noskill",
        "query_count": 25,
        "runtime_lock_sha256": "a" * 64,
        "launch_plan_sha256": "b" * 64,
        "shard_summary_sha256": "c" * 64,
    }
    audit = {**payload, "audit_sha256": batch_audit._hash(payload)}
    path = tmp_path / "shard-audit.json"
    path.write_bytes(canonical_json_bytes(audit))

    assert (
        batch_audit._validate_shard_audit(
            path,
            shard_id="00-batch-001-noskill",
            config="noskill",
            runtime_lock_sha256="a" * 64,
            launch_plan_sha256="b" * 64,
            shard_summary_sha256="c" * 64,
        )
        == audit["audit_sha256"]
    )
    with pytest.raises(ValueError, match="shard audit is inconsistent"):
        batch_audit._validate_shard_audit(
            path,
            shard_id="00-batch-001-noskill",
            config="s1",
            runtime_lock_sha256="a" * 64,
            launch_plan_sha256="b" * 64,
            shard_summary_sha256="c" * 64,
        )


def test_current_shard_audit_rejects_reasoning_text_fields(
    tmp_path: Path,
) -> None:
    runtime_lock = {
        **batch_audit._ACTIVE_FINAL_RESULT_LOCK,
        **batch_audit._ACTIVE_EVALUATOR_SOURCE_LOCK,
    }
    receipt = {
        "query_id": "q-001",
        "attempts": 1,
        "initial_empty_response": False,
        "initial_retry_reason": None,
        "initial_reasoning_present": None,
        "initial_reasoning_tokens": None,
        "initial_reasoning_bytes": None,
        "initial_reasoning_sha256": None,
        "terminal_response_captured": True,
        "terminal_reasoning_present": True,
        "terminal_reasoning_tokens": 3,
        "terminal_reasoning_bytes": 7,
        "terminal_reasoning_sha256": "d" * 64,
    }
    payload = {
        "schema_version": runtime_lock["final_judge_result_schema_version"],
        "kind": "portfolio-shard-audit",
        "formal_eligible": False,
        "shard_id": "00-batch-001-noskill",
        "config": "noskill",
        "query_count": 25,
        "runtime_lock_sha256": "a" * 64,
        "launch_plan_sha256": "b" * 64,
        "shard_summary_sha256": "c" * 64,
        "judge_invoked_count": 1,
        "judge_card_requirement_guard_adjusted_count": 0,
        "judge_model_calls": 1,
        "judge_captured_response_count": 1,
        "judge_retried_row_count": 0,
        "judge_initial_empty_response_count": 0,
        "judge_initial_invalid_json_response_count": 0,
        "judge_reasoning_present_response_count": 1,
        "judge_reasoning_tokens_reported_total": 3,
        "judge_reasoning_tokens_unavailable_response_count": 0,
        "judge_reasoning_bytes_total": 7,
        "judge_response_receipts": [receipt],
        **batch_audit._ACTIVE_FINAL_RESULT_LOCK,
    }
    path = tmp_path / "current-shard-audit.json"
    valid = {**payload, "audit_sha256": batch_audit._hash(payload)}
    path.write_bytes(canonical_json_bytes(valid))

    assert (
        batch_audit._validate_shard_audit(
            path,
            shard_id="00-batch-001-noskill",
            config="noskill",
            runtime_lock_sha256="a" * 64,
            launch_plan_sha256="b" * 64,
            shard_summary_sha256="c" * 64,
            runtime_lock=runtime_lock,
        )
        == valid["audit_sha256"]
    )

    unsafe_payload = {
        **payload,
        "judge_response_receipts": [
            {**receipt, "reasoning_content": "must never be persisted"}
        ],
    }
    path.write_bytes(
        canonical_json_bytes(
            {
                **unsafe_payload,
                "audit_sha256": batch_audit._hash(unsafe_payload),
            }
        )
    )
    with pytest.raises(ValueError, match="shard audit is inconsistent"):
        batch_audit._validate_shard_audit(
            path,
            shard_id="00-batch-001-noskill",
            config="noskill",
            runtime_lock_sha256="a" * 64,
            launch_plan_sha256="b" * 64,
            shard_summary_sha256="c" * 64,
            runtime_lock=runtime_lock,
        )


def test_noskill_freeze_evidence_is_digest_only_and_mismatch_blocks() -> None:
    observed = "a" * 64

    evidence, violations = batch_audit._noskill_freeze_evidence(
        observed_sha256=observed,
        expected_sha256=observed,
    )

    assert evidence == {
        "expected_audit_sha256": observed,
        "observed_audit_sha256": observed,
        "matches": True,
        "evidence_scope": "digest_match_only_not_wall_clock_order_proof",
    }
    assert violations == []

    _, violations = batch_audit._noskill_freeze_evidence(
        observed_sha256=observed,
        expected_sha256="b" * 64,
    )
    assert violations == [
        "NoSkill shard audit differs from the previously observed freeze digest"
    ]


def test_s1s2_full_route_reuse_check_blocks_independent_and_missing_routes() -> None:
    rows = [
        _observation(query_id=f"q-{index:03d}", config=config)
        for config in MAIN_CONFIG_ORDER
        for index in range(3)
    ]
    route_error_index = next(
        index
        for index, row in enumerate(rows)
        if row.query_id == "q-001" and row.config == "s1s2"
    )
    rows[route_error_index] = replace(
        rows[route_error_index],
        assistant_error_code="runtime_error",
        receipt_outcome="runtime_error",
        selected_capability=None,
        skill_slug=None,
        route_trace_sha256=None,
        route_acceptable=None,
        judge_status="not_invoked_assistant_error",
        model_call_count=1,
    )
    provider_error_index = next(
        index
        for index, row in enumerate(rows)
        if row.query_id == "q-002" and row.config == "s1s2"
    )
    rows[provider_error_index] = replace(
        rows[provider_error_index],
        assistant_error_code="runtime_error",
        receipt_outcome="runtime_error",
        selected_capability=None,
        skill_slug=None,
        route_trace_sha256=None,
        route_call_response_sha256=None,
        route_acceptable=None,
        judge_status="not_invoked_assistant_error",
        model_call_count=0,
    )

    check, violations = batch_audit._s1s2_full_route_reuse_check(rows)

    assert check["query_count"] == 3
    assert check["semantic_match_count"] == 1
    assert check["semantic_mismatch_count"] == 0
    assert check["unverifiable_due_assistant_error_count"] == 2
    assert check["separate_router_call_count"] == 2
    assert check["rows"][1]["s1s2"]["state"] == (
        "assistant_error_after_route_response_without_validated_decision"
    )
    assert check["rows"][2]["s1s2"]["state"] == (
        "assistant_error_before_route_response"
    )
    assert (
        "S1+S2 and Full do not persist a shared Stage2 route decision identity"
        in violations
    )
    assert (
        "S1+S2 and Full independently invoked the router instead of reusing "
        "one Stage2 decision"
    ) in violations
    assert (
        "S1+S2/Full route decision is unavailable for comparison: q-001" in violations
    )


def test_s1s2_full_route_reuse_check_blocks_semantic_mismatch() -> None:
    rows = [
        _observation(query_id="q-001", config=config) for config in MAIN_CONFIG_ORDER
    ]
    full_index = next(index for index, row in enumerate(rows) if row.config == "full")
    rows[full_index] = replace(
        rows[full_index],
        selected_capability="product.multi_search",
        skill_slug="full-multi",
        route_acceptable=False,
    )

    check, violations = batch_audit._s1s2_full_route_reuse_check(rows)

    assert check["semantic_match_count"] == 0
    assert check["semantic_mismatch_count"] == 1
    assert "S1+S2/Full semantic route decision differs: q-001" in violations


def test_shared_route_reuse_accepts_selected_and_terminal_fixed_zero() -> None:
    selected = []
    terminal = []
    for config in ("s1s2", "full"):
        selected.append(
            replace(
                _observation(query_id="q-selected", config=config),
                model_call_count=1,
                turn_count=2,
                route_trace_sha256="c" * 64,
                route_call_response_sha256="d" * 64,
                shared_route_artifact_sha256="c" * 64,
                shared_route_reserved_input_tokens=17,
                shared_route_reserved_output_tokens=5,
                shared_route_reserved_turns=1,
                shared_route_status="selected",
            )
        )
        terminal.append(
            replace(
                _observation(query_id="q-terminal", config=config),
                assistant_error_code="runtime_error",
                receipt_outcome="runtime_error",
                model_call_count=0,
                turn_count=1,
                selected_capability=None,
                skill_slug=None,
                route_trace_sha256=None,
                route_call_response_sha256="e" * 64,
                shared_route_artifact_sha256="f" * 64,
                shared_route_reserved_input_tokens=13,
                shared_route_reserved_output_tokens=4,
                shared_route_reserved_turns=1,
                shared_route_status="terminal_route_error",
                route_acceptable=None,
                judge_status="not_invoked_assistant_error",
                judge_shape=None,
                judge_finish_reason=None,
                judge_raw_response_bytes=None,
                judge_tool_call_count=None,
                judge_input_tokens=0,
                judge_output_tokens=0,
                j_project=0.0,
            )
        )

    check, violations = batch_audit._s1s2_full_route_reuse_check(
        [*selected, *terminal],
        runner_file_sha256="a" * 64,
    )

    assert violations == []
    assert check["shared_decision_identity_persisted"] is True
    assert check["matched_query_count"] == 2
    assert check["terminal_shared_route_fixed_zero_count"] == 1
    assert {item["status"] for item in check["rows"]} == {
        "matched_selected_route",
        "matched_terminal_fixed_zero",
    }


def test_shared_route_reuse_blocks_reserved_usage_drift() -> None:
    s1s2 = replace(
        _observation(query_id="q-001", config="s1s2"),
        route_trace_sha256="c" * 64,
        route_call_response_sha256="d" * 64,
        shared_route_artifact_sha256="c" * 64,
        shared_route_reserved_input_tokens=17,
        shared_route_reserved_output_tokens=5,
        shared_route_reserved_turns=1,
        shared_route_status="selected",
    )
    full = replace(
        _observation(query_id="q-001", config="full"),
        route_trace_sha256="c" * 64,
        route_call_response_sha256="d" * 64,
        shared_route_artifact_sha256="c" * 64,
        shared_route_reserved_input_tokens=18,
        shared_route_reserved_output_tokens=5,
        shared_route_reserved_turns=1,
        shared_route_status="selected",
    )

    check, violations = batch_audit._s1s2_full_route_reuse_check([s1s2, full])

    assert check["mismatch_count"] == 1
    assert "S1+S2/Full shared route identity differs: q-001" in violations


def test_expected_turn_count_includes_reserved_shared_route_turn() -> None:
    without_shared = SimpleNamespace(
        model_calls=(object(), object()),
        shared_route_reference=None,
    )
    with_shared = SimpleNamespace(
        model_calls=(object(), object()),
        shared_route_reference=SimpleNamespace(reserved_turns=1),
    )
    terminal = SimpleNamespace(
        model_calls=(),
        shared_route_reference=SimpleNamespace(reserved_turns=1),
    )

    assert batch_audit._expected_turn_count(without_shared) == 2
    assert batch_audit._expected_turn_count(with_shared) == 3
    assert batch_audit._expected_turn_count(terminal) == 1


def test_external_member_compatibility_excludes_run_specific_fields() -> None:
    target = SimpleNamespace(
        query_id="q-001",
        query_ordinal=0,
        query_sha256="a" * 64,
        public_input_sha256="b" * 64,
        asset_id="asset-1",
        image_path="images/1.jpg",
        image_sha256="c" * 64,
        matrix_run_id="target",
        instance_sha256="d" * 64,
    )
    source = SimpleNamespace(
        **{
            **vars(target),
            "matrix_run_id": "source",
            "instance_sha256": "e" * 64,
        }
    )

    assert (
        batch_audit._member_compatibility_violations(
            target_members=(target,),
            source_members=(source,),
        )
        == []
    )

    drifted = SimpleNamespace(**{**vars(source), "image_sha256": "f" * 64})
    assert batch_audit._member_compatibility_violations(
        target_members=(target,),
        source_members=(drifted,),
    ) == ["external NoSkill input drift for q-001: image_sha256"]


def test_external_descriptor_without_full_comparison_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="lacks required fields"):
        batch_audit._load_external_frozen_source(
            control={"external_frozen_shard": {"config": "noskill"}},
            target_launch=SimpleNamespace(),
            target_shards=(),
            target_members=(),
        )


def test_comparison_projection_requires_complete_judge_billing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    judge_endpoint = "https://www.aifast.club/v1"
    plan = SimpleNamespace(
        assistant_provider="qwen",
        assistant_model="qwen3-vl-flash-2026-01-22",
    )
    launch = SimpleNamespace(plan=plan)
    projection = {
        "models": {
            "final_judge": {
                "provider": "gemini",
                "model": "gemini-3.6-flash",
                "endpoint": judge_endpoint,
            },
            "feedback": {
                "provider": "kimi",
                "model": "kimi-k2.6",
                "endpoint": endpoint,
            },
        }
    }
    monkeypatch.setattr(
        batch_audit,
        "_require_active_final_result_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        batch_audit,
        "_comparison_launch_projection",
        lambda _plan: projection,
    )
    monkeypatch.setattr(batch_audit, "_launch_endpoint", lambda *_args: endpoint)
    runtime = {
        "system_prompt_sha256": "1" * 64,
        "tool_registry_sha256": "2" * 64,
        "tool_registry_runtime_sha256": "3" * 64,
        "final_judge_parser_policy_version": (
            batch_audit.FINAL_JUDGE_PARSER_POLICY_VERSION_V4
        ),
        "final_judge_parser_policy_sha256": (
            batch_audit.FINAL_JUDGE_PARSER_POLICY_SHA256_V4
        ),
        "final_judge_result_schema_version": (
            batch_audit.FINAL_JUDGE_RESULT_SCHEMA_VERSION
        ),
        "final_judge_cache_namespace": batch_audit.FINAL_JUDGE_CACHE_NAMESPACE,
        "card_requirement_guard_policy_version": (
            batch_audit.CARD_REQUIREMENT_GUARD_POLICY_VERSION
        ),
        "card_requirement_guard_policy_sha256": (
            batch_audit.CARD_REQUIREMENT_GUARD_POLICY_SHA256
        ),
        "final_judge_max_attempts": batch_audit.FINAL_JUDGE_MAX_ATTEMPTS,
        "final_judge_retry_policy_version": (
            batch_audit.FINAL_JUDGE_RETRY_POLICY_VERSION
        ),
        "final_judge_retry_policy_sha256": (
            batch_audit.FINAL_JUDGE_RETRY_POLICY_SHA256
        ),
        "final_judge_transport_policy_version": (
            batch_audit.FINAL_JUDGE_TRANSPORT_POLICY_VERSION
        ),
        "final_judge_transport_policy_sha256": (
            batch_audit.FINAL_JUDGE_TRANSPORT_POLICY_SHA256
        ),
        "final_judge_requested_response_format": "json_object",
        "final_judge_provider_pricing_status": (
            batch_audit.PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS
        ),
        "noskill_execution_contract_sha256": (
            batch_audit.NOSKILL_EXECUTION_CONTRACT_SHA256
        ),
        "noskill_execution_policy_version": (
            batch_audit.NOSKILL_EXECUTION_POLICY_VERSION
        ),
    }
    final_identity = {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "endpoint": judge_endpoint,
        "max_tokens": 2048,
        "max_attempts": batch_audit.FINAL_JUDGE_MAX_ATTEMPTS,
        "retry_policy_version": batch_audit.FINAL_JUDGE_RETRY_POLICY_VERSION,
        "retry_policy_sha256": batch_audit.FINAL_JUDGE_RETRY_POLICY_SHA256,
        "thinking_budget": batch_audit.FINAL_JUDGE_THINKING_BUDGET,
        "max_billable_input_tokens": (
            batch_audit.FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
        ),
        "max_billable_output_tokens": (
            batch_audit.FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
        ),
        "transport_policy_version": batch_audit.FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
        "transport_policy_sha256": batch_audit.FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
        "requested_response_format": "json_object",
    }
    request_contract = {
        "assistant": {
            "backbone": {
                "provider": "qwen",
                "model": "qwen3-vl-flash-2026-01-22",
                "endpoint": endpoint,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": None,
                "system_prompt_sha256": "1" * 64,
            },
            "budget": {**batch_audit._EXPECTED_BUDGET, "budget_sha256": "4" * 64},
            "registry": {
                "registry_sha256": "2" * 64,
                "registry_runtime_sha256": "3" * 64,
                "lock_sha256": "5" * 64,
            },
        },
        "final_judge": final_identity,
    }

    payload = batch_audit._comparison_contract_payload(
        launch=launch,
        control={"rubric_file_sha256": "6" * 64, "rubric_content_sha256": "7" * 64},
        runtime_raw=runtime,
        source_request_contract=request_contract,
        require_noskill_runtime_binding=True,
    )

    assert payload["final_judge_identity"] == final_identity
    incomplete = {
        **request_contract,
        "final_judge": {
            key: value
            for key, value in final_identity.items()
            if key != "transport_policy_sha256"
        },
    }
    with pytest.raises(ValueError, match="final-Judge contract is incompatible"):
        batch_audit._comparison_contract_payload(
            launch=launch,
            control={
                "rubric_file_sha256": "6" * 64,
                "rubric_content_sha256": "7" * 64,
            },
            runtime_raw=runtime,
            source_request_contract=incomplete,
            require_noskill_runtime_binding=True,
        )


def test_external_artifact_inventory_is_exactly_52_files(tmp_path: Path) -> None:
    shard = SimpleNamespace(output_relpath="shards/source-noskill")
    members = tuple(SimpleNamespace(query_id=f"q-{index:03d}") for index in range(25))
    shard_root = tmp_path / shard.output_relpath
    (shard_root / "assistant").mkdir(parents=True)
    (shard_root / "final").mkdir()
    (shard_root / "shard-audit.json").write_bytes(b"audit")
    (shard_root / "shard-summary.json").write_bytes(b"summary")
    for member in members:
        (shard_root / "assistant" / f"{member.query_id}.json").write_bytes(
            member.query_id.encode()
        )
        (shard_root / "final" / f"{member.query_id}.json").write_bytes(
            f"final-{member.query_id}".encode()
        )

    inventory = batch_audit._shard_artifact_inventory(
        tmp_path,
        shard=shard,
        members=members,
    )

    assert len(inventory) == 52
    assert list(inventory) == sorted(inventory)
    assert sha256_bytes(canonical_json_bytes(inventory))
    (shard_root / "unexpected.json").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exactly 52 files"):
        batch_audit._shard_artifact_inventory(
            tmp_path,
            shard=shard,
            members=members,
        )


def test_incomplete_batch_emits_hashed_fail_closed_audit_without_deep_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = _shards()
    members = _members(shards)
    package = SimpleNamespace(
        plan=SimpleNamespace(shards=shards),
        instances=members,
    )
    monkeypatch.setattr(
        batch_audit,
        "_load_control",
        lambda _root: {
            "launch_root": "unused-launch",
            "launch_plan_file_sha256": "a" * 64,
            "runtime_root": "unused-runtime",
        },
    )
    monkeypatch.setattr(
        batch_audit,
        "load_portfolio_launch_package",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(
        batch_audit,
        "_load_active_inputs",
        lambda: (_ for _ in ()).throw(
            AssertionError("incomplete audit must not deep-load inputs")
        ),
    )

    audit = batch_audit.audit_portfolio_batch(
        tmp_path,
        batch_id="batch-001",
    )

    assert audit["status"] == "incomplete_fail_closed"
    assert audit["row_count"] == 0
    assert audit["model_calls_performed"] == 0
    assert set(audit["bindings"]["observed_source_file_sha256s"]) == {
        "src/skillchain/config.py",
        "src/skillchain/evaluation/evaluator_isolation.py",
        "src/skillchain/evaluation/packets.py",
        "src/skillchain/runners/assistant.py",
        "src/skillchain/tools/portfolio_runtime.py",
        "src/skillchain/tools/registry.py",
    }
    unsigned = dict(audit)
    supplied = unsigned.pop("audit_sha256")
    assert supplied == sha256_bytes(canonical_json_bytes(unsigned))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("passed_clean", 0),
        ("review_required", 3),
        ("failed", 2),
        ("incomplete_fail_closed", 2),
    ],
)
def test_exit_code_is_fail_closed(status: str, expected: int) -> None:
    assert batch_audit._exit_code({"status": status}) == expected


def test_main_prints_only_canonical_content_addressed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    expected = batch_audit._failure_audit(
        status="review_required",
        requested_batch_id="batch-001",
        blockers=[],
    )
    monkeypatch.setattr(
        batch_audit,
        "audit_portfolio_batch",
        lambda *_args, **_kwargs: expected,
    )

    exit_code = batch_audit.main(
        [
            "--execution-root",
            str(tmp_path),
            "--batch-id",
            "batch-001",
        ]
    )
    stdout, stderr = capfd.readouterr()

    assert exit_code == 3
    assert stderr == ""
    assert json.loads(stdout) == expected
    assert stdout.encode("utf-8") == canonical_json_bytes(expected)


def test_main_create_only_output_matches_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    expected = batch_audit._failure_audit(
        status="passed_clean",
        requested_batch_id="batch-001",
        blockers=[],
    )
    monkeypatch.setattr(
        batch_audit,
        "audit_portfolio_batch",
        lambda *_args, **_kwargs: expected,
    )
    output = tmp_path / "audit.json"

    exit_code = batch_audit.main(
        [
            "--execution-root",
            str(tmp_path),
            "--batch-id",
            "batch-001",
            "--output",
            str(output),
        ]
    )
    stdout, stderr = capfd.readouterr()

    assert exit_code == 0
    assert stderr == ""
    assert output.read_bytes() == stdout.encode("utf-8")
    assert output.read_bytes() == canonical_json_bytes(expected)


def test_main_output_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    expected = batch_audit._failure_audit(
        status="passed_clean",
        requested_batch_id="batch-001",
        blockers=[],
    )
    monkeypatch.setattr(
        batch_audit,
        "audit_portfolio_batch",
        lambda *_args, **_kwargs: expected,
    )
    output = tmp_path / "audit.json"
    original = b"do-not-overwrite\n"
    output.write_bytes(original)

    exit_code = batch_audit.main(
        [
            "--execution-root",
            str(tmp_path),
            "--batch-id",
            "batch-001",
            "--output",
            str(output),
        ]
    )
    stdout, stderr = capfd.readouterr()

    assert exit_code == 2
    assert stdout == ""
    assert "portfolio-batch-audit:" in stderr
    assert output.read_bytes() == original
