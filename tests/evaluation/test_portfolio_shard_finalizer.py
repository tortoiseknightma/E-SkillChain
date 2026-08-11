from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finalize_portfolio_shard as finalizer
from skillchain.evaluation.assistant_runs import (
    AssistantModelCallReceipt,
    make_assistant_route_call_evidence,
)
from skillchain.evaluation.portfolio_execution import (
    ProviderPreResponseCircuitBreaker,
    create_retryable_attempt_receipt,
)
from skillchain.evaluation.portfolio_gcs import GCS_V2_POLICY_SHA256
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.tools.serialization import canonical_jsonl_bytes, sha256_bytes


def _card_query(
    query_id: str,
    *,
    requires_card: bool,
    split: str = "dev_mini",
) -> Query:
    text = f"Public request {query_id}"
    capability = "utility.document_reading"
    return Query(
        schema_version=2,
        taxonomy_version="taxonomy-test",
        task_spec_version="task-test",
        query_id=query_id,
        asset_id=f"asset-{query_id}",
        image_path=f"query_images/{query_id}.jpg",
        leakage_group_id=f"leakage-{query_id}",
        template_family="portfolio-finalizer-test",
        generator_batch_id="batch-1",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent="utility",
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        requires_card=requires_card,
        split=split,
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="portfolio-finalizer-test",
                canonical_intent="utility",
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _launch_member(query: Query, *, core: bool = False) -> SimpleNamespace:
    public = finalizer.build_assistant_query_input(query)
    assert public.asset_binding is not None
    return SimpleNamespace(
        query_id=query.query_id,
        query_sha256=public.query_sha256,
        public_input_sha256=public.public_input_sha256,
        asset_id=None if core else query.asset_id,
        image_path=None if core else query.image_path,
        accepted_batch_id=(
            query.generator_batch_id if core else query.synthesis_batch_id
        ),
        asset_token=public.asset_binding.asset_token,
        asset_binding_sha256=public.asset_binding.binding_sha256,
    )


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


def _budget_state_for_summary_refresh() -> SimpleNamespace:
    reservation_one = SimpleNamespace(
        kind="portfolio-budget-reservation",
        ledger_event_index=1,
        reservation_sha256="a" * 64,
        reserved_cost_cny=Decimal("1.000000000000"),
    )
    settlement_one = SimpleNamespace(
        kind="portfolio-budget-settlement",
        ledger_event_index=2,
        reservation_sha256=reservation_one.reservation_sha256,
        settlement_sha256="b" * 64,
        actual_cost_cny=Decimal("0.100000000000"),
    )
    reservation_two = SimpleNamespace(
        kind="portfolio-budget-reservation",
        ledger_event_index=3,
        reservation_sha256="c" * 64,
        reserved_cost_cny=Decimal("1.000000000000"),
    )
    settlement_two = SimpleNamespace(
        kind="portfolio-budget-settlement",
        ledger_event_index=4,
        reservation_sha256=reservation_two.reservation_sha256,
        settlement_sha256="d" * 64,
        actual_cost_cny=Decimal("0.200000000000"),
    )
    return SimpleNamespace(
        authority=SimpleNamespace(
            authority_sha256="e" * 64,
            prior_observed_cost_cny=Decimal("2.000000000000"),
        ),
        reservations=(reservation_one, reservation_two),
        settlements=(settlement_one, settlement_two),
        settled_actual_cost_cny=Decimal("0.300000000000"),
        unresolved_reserved_cost_cny=Decimal("0.000000000000"),
        accountable_cost_cny=Decimal("2.300000000000"),
        last_event_index=4,
        last_event_sha256=settlement_two.settlement_sha256,
    )


def _stale_summary(state: SimpleNamespace) -> dict[str, object]:
    prefix = finalizer._ledger_prefix_snapshot(state, last_event_index=2)
    payload = {
        "schema_version": 1,
        "kind": "portfolio-shard-summary",
        "shard_id": "00-shard",
        "config": "noskill",
        "query_count": 25,
        "status": "complete",
        **prefix,
        "elapsed_seconds": 12.5,
    }
    return {**payload, "summary_sha256": finalizer._hash(payload)}


def test_stale_concurrent_summary_refreshes_only_global_ledger_snapshot() -> None:
    state = _budget_state_for_summary_refresh()
    source = _stale_summary(state)
    original = dict(source)

    refreshed, changed = finalizer._prepare_current_shard_summary(
        source,
        shard_id="00-shard",
        config="noskill",
        query_count=25,
        budget_ledger=state,
    )

    assert changed is True
    assert refreshed["budget_ledger_last_event_index"] == 4
    assert refreshed["budget_ledger_last_event_sha256"] == "d" * 64
    assert refreshed["budget_ledger_settled_actual_cost_cny"] == "0.300000000000"
    assert refreshed["budget_ledger_accountable_cost_cny"] == "2.300000000000"
    assert refreshed["observed_dashscope_cost_cny"] == 0.3
    assert refreshed["elapsed_seconds"] == source["elapsed_seconds"]
    assert source == original
    assert refreshed["summary_sha256"] == finalizer._hash(
        {key: value for key, value in refreshed.items() if key != "summary_sha256"}
    )


def test_current_concurrent_summary_is_byte_value_idempotent() -> None:
    state = _budget_state_for_summary_refresh()
    stale = _stale_summary(state)
    current, _ = finalizer._prepare_current_shard_summary(
        stale,
        shard_id="00-shard",
        config="noskill",
        query_count=25,
        budget_ledger=state,
    )

    replayed, changed = finalizer._prepare_current_shard_summary(
        current,
        shard_id="00-shard",
        config="noskill",
        query_count=25,
        budget_ledger=state,
    )

    assert changed is False
    assert replayed == current
    assert finalizer.canonical_json_bytes(replayed) == finalizer.canonical_json_bytes(
        current
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "budget_ledger_last_event_sha256",
            "f" * 64,
            "authentic budget-ledger prefix",
        ),
        (
            "budget_ledger_settled_actual_cost_cny",
            "0.200000000000",
            "authentic budget-ledger prefix",
        ),
        ("budget_ledger_last_event_index", 5, "outside the ledger"),
    ),
)
def test_summary_refresh_rejects_a_self_hashed_non_prefix_snapshot(
    field: str,
    value: object,
    message: str,
) -> None:
    state = _budget_state_for_summary_refresh()
    source = _stale_summary(state)
    source[field] = value
    unsigned = {key: value for key, value in source.items() if key != "summary_sha256"}
    source["summary_sha256"] = finalizer._hash(unsigned)

    with pytest.raises(ValueError, match=message):
        finalizer._prepare_current_shard_summary(
            source,
            shard_id="00-shard",
            config="noskill",
            query_count=25,
            budget_ledger=state,
        )


@pytest.mark.parametrize(
    ("parser_schema_version", "expected"),
    ((4, False), (5, True), (6, True)),
)
def test_judge_response_receipts_cover_schema_6(
    parser_schema_version: int,
    expected: bool,
) -> None:
    assert (
        finalizer._should_record_judge_response_receipt(parser_schema_version)
        is expected
    )


def test_hidden_identity_fixed_zero_is_valid_after_successful_assistant(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "dm-001.json"
    payload = {
        "schema_version": 1,
        "kind": "portfolio-final-fixed-zero",
        "query_id": "dm-001",
        "assistant_error_code": "hidden_evaluation_identity",
        "failure_class": "terminal_task_failure",
        "retryable": False,
        "score_disposition": "fixed_zero",
        "j_project": 0.0,
    }
    final_path.write_bytes(
        finalizer.canonical_json_bytes(
            {**payload, "result_sha256": finalizer._hash(payload)}
        )
    )

    assert (
        finalizer._validated_fixed_zero_error(
            final_path,
            query_id="dm-001",
            response_error_code=None,
        )
        == "hidden_evaluation_identity"
    )


def test_finalize_scope_rejects_external_and_unauthorized_shards() -> None:
    control = {
        "authorized_shard_ids": ["01-skilled"],
        "external_frozen_shard": {"target_shard_id": "00-noskill"},
    }

    with pytest.raises(ValueError, match="must not be finalized locally"):
        finalizer._require_local_finalize_authorized(control, "00-noskill")
    with pytest.raises(ValueError, match="outside the execution authorization"):
        finalizer._require_local_finalize_authorized(control, "02-other")

    finalizer._require_local_finalize_authorized(control, "01-skilled")
    # An old execution control has no partial-scope allowlist.
    finalizer._require_local_finalize_authorized({}, "00-legacy")


def _cumulative_budget_fixture(*, accountable: str = "25.429591700000"):
    control = {
        "approved_dashscope_budget_cny": "200.000000000000",
        "phase_cumulative_cap_cny": "200.000000000000",
        "incremental_authorized_dashscope_budget_cny": "174.570408300000",
        "prior_dashscope_observed_cost_cny": "25.429591700000",
    }
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id="matrix-with-prior",
            budget=SimpleNamespace(operator_approved_dashscope_budget_cny=200.0),
        )
    )
    ledger = SimpleNamespace(
        authority=SimpleNamespace(
            matrix_run_id="matrix-with-prior",
            phase_cap_cny=Decimal("200.000000000000"),
            prior_observed_cost_cny=Decimal("25.429591700000"),
            policy_version=finalizer.PORTFOLIO_BUDGET_POLICY_VERSION,
        ),
        accountable_cost_cny=Decimal(accountable),
    )
    return control, launch, ledger


def test_finalizer_accepts_full_matrix_cumulative_prior() -> None:
    control, launch, ledger = _cumulative_budget_fixture()

    finalizer._validate_budget_authority(
        control=control,
        launch=launch,
        budget_ledger=ledger,
    )


def test_finalizer_rejects_over_cap_for_every_execution_scope() -> None:
    control, launch, ledger = _cumulative_budget_fixture(accountable="200.000000000001")
    control["execution_scope"] = "full_matrix"

    with pytest.raises(ValueError, match="execution phase exceeded"):
        finalizer._validate_budget_authority(
            control=control,
            launch=launch,
            budget_ledger=ledger,
        )


def test_turn_accounting_distinguishes_actual_and_reserved_calls() -> None:
    usage = SimpleNamespace(input_tokens=17, output_tokens=3)
    shared = SimpleNamespace(reserved_turns=1, reserved_usage=usage)
    receipt = SimpleNamespace(model_calls=(object(),), shared_route_reference=shared)
    response = SimpleNamespace(turn_count=2)

    assert finalizer._validate_turn_accounting(response, receipt) == (1, 17, 3)

    receipt_without_shared = SimpleNamespace(
        model_calls=(object(), object()),
        shared_route_reference=None,
    )
    assert finalizer._validate_turn_accounting(
        SimpleNamespace(turn_count=2),
        receipt_without_shared,
    ) == (0, 0, 0)
    with pytest.raises(ValueError, match="actual plus reserved"):
        finalizer._validate_turn_accounting(
            SimpleNamespace(turn_count=1),
            receipt,
        )


def test_attempt_accounting_adds_only_uncaptured_pre_response_call(
    tmp_path: Path,
) -> None:
    shard_id = "00-batch-001-s1"
    shard_root = tmp_path / shard_id
    breaker = ProviderPreResponseCircuitBreaker()
    common = {
        "shard_root": shard_root,
        "breaker": breaker,
        "matrix_run_id": "portfolio-mini",
        "shard_id": shard_id,
        "config": "s1",
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
        forfeited_reservation_sha256="e" * 64,
        budget_forfeit_sha256="f" * 64,
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

    observed_cost, model_calls = finalizer._attempt_cost_and_calls(
        shard_root,
        shard_id=shard_id,
    )

    assert model_calls == 3
    assert observed_cost == pytest.approx(
        ((10 + 12) * 0.15 + (2 + 3) * 1.5) / 1_000_000
    )


def test_tool_error_summary_retains_recovered_error_identity() -> None:
    success = SimpleNamespace(
        status="success",
        error_code=None,
        tool_name="image_product_search",
    )
    context_error = SimpleNamespace(
        status="error",
        error_code="context_violation",
        tool_name="text_product_search",
    )
    timeout = SimpleNamespace(
        status="timeout",
        error_code="timeout",
        tool_name="document_ocr",
    )

    assert finalizer._summarize_tool_errors(
        [
            ("dm-001", (success, context_error)),
            ("dm-002", (success,)),
            ("dm-003", (timeout,)),
        ]
    ) == {
        "tool_error_count": 2,
        "tool_error_query_count": 2,
        "tool_error_ids": ["dm-001", "dm-003"],
        "tool_error_counts_by_code": {"context_violation": 1, "timeout": 1},
        "tool_error_counts_by_tool": {
            "document_ocr": 1,
            "text_product_search": 1,
        },
    }


def test_card_policy_summary_keeps_required_and_forbidden_violations_separate() -> None:
    assert finalizer._summarize_card_policy(
        [
            ("dm-001", True, 1),
            ("dm-002", True, 0),
            ("dm-003", False, 1),
            ("dm-004", False, 0),
        ]
    ) == {
        "card_policy_compliant_count": 2,
        "card_policy_violation_count": 2,
        "card_policy_violation_ids": ["dm-002", "dm-003"],
        "required_missing_card_ids": ["dm-002"],
        "forbidden_card_ids": ["dm-003"],
    }


def test_card_requirements_replay_frozen_query_and_launch_bindings(
    tmp_path: Path,
) -> None:
    query = _card_query("dm-001", requires_card=True)
    one_query = canonical_jsonl_bytes([query.model_dump(mode="json")])
    artifact = tmp_path / "queries.jsonl"
    artifact.write_bytes(one_query)
    member = _launch_member(query)
    launch = SimpleNamespace(
        plan=SimpleNamespace(query_artifact_sha256=sha256_bytes(one_query)),
        instances=(member,),
    )

    assert finalizer._load_launch_card_requirements(
        launch,
        query_artifact_path=artifact,
    ) == {query.query_id: query.requires_card}

    launch.instances = (
        SimpleNamespace(**{**vars(member), "accepted_batch_id": "wrong-batch"}),
    )
    with pytest.raises(ValueError, match="query binding differs from launch"):
        finalizer._load_launch_card_requirements(
            launch,
            query_artifact_path=artifact,
        )


def test_card_requirements_core_uses_bound_full_source_subset(
    tmp_path: Path,
) -> None:
    selected = _card_query("dm-001", requires_card=True)
    unselected = _card_query(
        "core-201",
        requires_card=False,
        split="opt_pool",
    )
    content = canonical_jsonl_bytes(
        [
            selected.model_dump(mode="json"),
            unselected.model_dump(mode="json"),
        ]
    )
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=2,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(_launch_member(selected, core=True),),
    )

    assert finalizer._load_launch_card_requirements(launch) == {"dm-001": True}


def test_card_requirements_core_rejects_source_count_drift(tmp_path: Path) -> None:
    query = _card_query("dm-001", requires_card=True)
    content = canonical_jsonl_bytes([query.model_dump(mode="json")])
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=2,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(_launch_member(query, core=True),),
    )

    with pytest.raises(ValueError, match="count differs from launch source"):
        finalizer._load_launch_card_requirements(launch)


def test_card_requirements_core_rejects_missing_selected_query(
    tmp_path: Path,
) -> None:
    source_query = _card_query("dm-001", requires_card=True)
    selected_query = _card_query("dm-002", requires_card=False)
    content = canonical_jsonl_bytes([source_query.model_dump(mode="json")])
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=1,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(_launch_member(selected_query, core=True),),
    )

    with pytest.raises(ValueError, match="lacks selected launch queries"):
        finalizer._load_launch_card_requirements(launch)


def test_card_requirements_non_core_still_rejects_extra_source_row(
    tmp_path: Path,
) -> None:
    selected = _card_query("dm-001", requires_card=True)
    extra = _card_query("dm-002", requires_card=False)
    content = canonical_jsonl_bytes(
        [selected.model_dump(mode="json"), extra.model_dump(mode="json")]
    )
    artifact = tmp_path / "queries.jsonl"
    artifact.write_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(query_artifact_sha256=sha256_bytes(content)),
        instances=(_launch_member(selected),),
    )

    with pytest.raises(ValueError, match="coverage differs from launch"):
        finalizer._load_launch_card_requirements(
            launch,
            query_artifact_path=artifact,
        )


def test_card_requirements_core_rejects_extra_selected_split_row(
    tmp_path: Path,
) -> None:
    selected = _card_query("dm-001", requires_card=True)
    omitted = _card_query("dm-002", requires_card=False)
    content = canonical_jsonl_bytes(
        [selected.model_dump(mode="json"), omitted.model_dump(mode="json")]
    )
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=2,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(_launch_member(selected, core=True),),
    )

    with pytest.raises(ValueError, match="selected split coverage differs"):
        finalizer._load_launch_card_requirements(launch)


def test_card_requirements_core_rejects_wrong_generator_batch(
    tmp_path: Path,
) -> None:
    query = _card_query("dm-001", requires_card=True)
    content = canonical_jsonl_bytes([query.model_dump(mode="json")])
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    member = _launch_member(query, core=True)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=1,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(
            SimpleNamespace(**{**vars(member), "accepted_batch_id": "wrong-batch"}),
        ),
    )

    with pytest.raises(ValueError, match="query binding differs from launch"):
        finalizer._load_launch_card_requirements(launch)


def test_card_requirements_core_rejects_binding_sha_drift(tmp_path: Path) -> None:
    query = _card_query("dm-001", requires_card=True)
    content = canonical_jsonl_bytes([query.model_dump(mode="json")])
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256="f" * 64,
            ),
            query_artifact_sha256=digest,
            source_query_count=1,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(_launch_member(query, core=True),),
    )

    with pytest.raises(ValueError, match="binding differs from launch plan"):
        finalizer._load_launch_card_requirements(launch)


@pytest.mark.parametrize("field", ["asset_token", "asset_binding_sha256"])
def test_card_requirements_core_rejects_opaque_asset_binding_drift(
    tmp_path: Path,
    field: str,
) -> None:
    query = _card_query("dm-001", requires_card=True)
    content = canonical_jsonl_bytes([query.model_dump(mode="json")])
    artifact = tmp_path / "core-queries.jsonl"
    artifact.write_bytes(content)
    digest = sha256_bytes(content)
    member = _launch_member(query, core=True)
    tampered = SimpleNamespace(
        **{
            **vars(member),
            field: ("asset-token-wrong" if field == "asset_token" else "f" * 64),
        }
    )
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            core_input_files=SimpleNamespace(
                query_artifact_path=artifact.absolute().as_posix(),
                expected_query_artifact_sha256=digest,
            ),
            query_artifact_sha256=digest,
            source_query_count=1,
            query_count=1,
            selected_splits=("dev_mini",),
        ),
        instances=(tampered,),
    )

    with pytest.raises(ValueError, match="query binding differs from launch"):
        finalizer._load_launch_card_requirements(launch)


def test_shared_route_cost_owner_is_deterministic_across_pair() -> None:
    s1s2 = SimpleNamespace(
        shard_id="03-s1s2",
        shard_ordinal=3,
        accepted_batch_id="batch-1",
        config="s1s2",
    )
    full = SimpleNamespace(
        shard_id="04-full",
        shard_ordinal=4,
        accepted_batch_id="batch-1",
        config="full",
    )
    launch = SimpleNamespace(plan=SimpleNamespace(shards=(full, s1s2)))

    assert finalizer._shared_route_owner_id(launch, s1s2) == "03-s1s2"
    assert finalizer._shared_route_owner_id(launch, full) == "03-s1s2"
    assert (
        finalizer._shared_route_owner_id(
            launch,
            SimpleNamespace(config="noskill"),
        )
        is None
    )


def test_external_plus_four_local_shards_keeps_full_launch_running() -> None:
    shards = tuple(
        SimpleNamespace(shard_id=f"{index:02d}-shard") for index in range(40)
    )
    plan = SimpleNamespace(shards=shards, shard_count=40)

    assert (
        finalizer._launch_completion_status(
            plan,
            tuple(item.shard_id for item in shards[:5]),
        )
        == "running"
    )
    assert (
        finalizer._launch_completion_status(
            plan,
            tuple(item.shard_id for item in shards),
        )
        == "completed"
    )


def test_external_artifact_set_requires_exact_52_files(tmp_path: Path) -> None:
    expected: dict[str, str] = {}
    for index in range(52):
        relative = f"artifact-{index:02d}.json"
        content = f'{{"index":{index}}}'.encode()
        path = tmp_path / relative
        path.write_bytes(content)
        expected[relative] = sha256_bytes(content)

    observed = finalizer._artifact_set_sha256s(tmp_path, expected=expected)
    assert observed == finalizer._hash(dict(sorted(expected.items())))

    (tmp_path / "artifact-03.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        finalizer._artifact_set_sha256s(tmp_path, expected=expected)


def test_execution_model_calls_are_local_and_shared_routes_are_unique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant = tmp_path / "shards" / "01-skilled" / "assistant" / "q.json"
    final = tmp_path / "shards" / "01-skilled" / "final" / "q.json"
    route_a = tmp_path / "shared-routes" / "batch" / "q-a.json"
    route_b = tmp_path / "shared-routes" / "batch" / "q-b.json"
    for path in (assistant, final, route_a, route_b):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")

    monkeypatch.setattr(
        finalizer,
        "_assistant_row",
        lambda _path: (
            None,
            SimpleNamespace(model_calls=(object(), object())),
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_load_canonical_object",
        lambda _path, *, label: {},
    )
    monkeypatch.setattr(
        finalizer,
        "load_final_judge_evaluation_result",
        lambda _path: SimpleNamespace(request_id="judge-request", attempts=2),
    )
    monkeypatch.setattr(
        finalizer.SharedStage2RouteArtifact,
        "model_validate_json",
        lambda _content, strict: SimpleNamespace(
            artifact_sha256="a" * 64,
            policy_version=finalizer.SHARED_STAGE2_ROUTE_POLICY_VERSION,
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_attempt_cost_and_calls",
        lambda _root, *, shard_id: (0.0, 3),
    )

    # 2 Assistant + 2 Judge attempts + 1 unique shared route + 3 failed/captured
    # attempt calls. External source artifacts are outside tmp_path and absent.
    assert finalizer._execution_model_calls(tmp_path) == 8


def test_execution_model_calls_rejects_legacy_shared_route_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = tmp_path / "shared-routes" / "batch" / "q.json"
    route.parent.mkdir(parents=True)
    route.write_bytes(b"{}")
    monkeypatch.setattr(
        finalizer.SharedStage2RouteArtifact,
        "model_validate_json",
        lambda _content, strict: SimpleNamespace(
            artifact_sha256="a" * 64,
            policy_version="shared-stage2-route-v1",
        ),
    )

    with pytest.raises(ValueError, match="inactive policy"):
        finalizer._execution_model_calls(tmp_path)


def test_static_opt_finalizer_rejects_any_final_judge_artifact(tmp_path: Path) -> None:
    shard = SimpleNamespace(
        shard_id="static-00",
        config="llm_static",
        output_relpath="shards/static-00",
    )
    members = tuple(SimpleNamespace() for _ in range(25))
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            kind="portfolio-core-static-opt-800x1-launch-plan",
            execution_mode="static_opt_rollout",
            selected_splits=("opt_pool",),
            config_order=("llm_static",),
        )
    )
    control = {
        "execution_scope": "static_opt_rollout",
        "evaluation_stages": ["assistant", "gcs_v2"],
        "assistant_checkpoint_schema_version": 2,
        "gcs_policy_version": "portfolio-grounded-contract-success-v2",
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "gcs_scorer_evidence_policy_version": ("portfolio-gcs-scorer-evidence-v2"),
        "pairwise_judge_enabled": False,
        "legacy_final_judge_enabled": False,
        "analyzer_provider_call_count": 0,
    }
    forbidden = tmp_path / shard.output_relpath / "final" / "q.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"{}")

    with pytest.raises(ValueError, match="forbidden Final Judge artifact"):
        finalizer._finalize_static_opt_shard(
            execution_root=tmp_path,
            control=control,
            launch=launch,
            shard=shard,
            members=members,
            budget_ledger=SimpleNamespace(),
        )
