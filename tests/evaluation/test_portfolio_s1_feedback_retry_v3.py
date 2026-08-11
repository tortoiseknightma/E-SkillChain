from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal

import pytest

from scripts.run_portfolio_s1_feedback_v3 import (
    PreparedPortfolioS1FeedbackV3Run,
    PortfolioS1FeedbackV3BudgetError,
    PortfolioS1FeedbackV3RunError,
    _execute_phase_v5,
    _execute_run_v3_locked,
    _claim_retry_v2,
    _parse_reviewed_at_v3,
    _publish_terminal_run_v5,
    _require_live_governance_locks_v3,
    _require_live_authority,
    _reject_historical_roots,
    build_parser_v3,
    execute_run_v3,
    prepare_run_v3,
    main,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
    run_visual_feedback,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
)
from skillchain.evaluation.portfolio_s1_feedback_retry_v3 import (
    BoundFeedbackArtifactV5,
    FeedbackCallReservationV5,
    FRESH_V3_OWNER_BUDGET_STATEMENT,
    PortfolioS1FeedbackAuthorizationV7,
    PortfolioS1FeedbackControlV12,
    PortfolioS1Qwen38FeedbackLaunchLockV5,
    QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2,
    build_selected_qwen38_feedback_authorization_v7,
    build_feedback_global_retry_claim_v2,
    build_portfolio_s1_feedback_run_v5,
    is_qwen38_schema_or_length_retry_eligible,
    load_selected_qwen38_feedback_authorization_v7,
    qwen38_feedback_global_retry_policy_v2,
    validate_feedback_global_retry_claim_set_v2,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2,
    QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2 as GOVERNANCE_RETRY_SHA,
    QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7 as GOVERNANCE_TRANSPORT_SHA,
    PortfolioS1QwenFeedbackGovernanceError,
    SelectedQwenFeedbackAssetV1,
    load_qwen38_feedback_model_source_lock_v2,
    load_qwen38_feedback_pricing_lock_v5,
    load_qwen38_feedback_role_selection_v12,
    require_qwen38_feedback_pre_call_budget_v3,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from skillchain.llm import LLMUsage


def _result(
    *,
    status: str = "parsed",
    error_code: str | None = None,
    raw: str = "{}",
    redaction: str | None = None,
    finish_reason: str = "stop",
    cache_namespace: str = "feedback-evaluator-v12",
    schema_version: int = 6,
    usage: LLMUsage | None = None,
) -> FeedbackEvaluationResult:
    return FeedbackEvaluationResult.model_construct(
        schema_version=schema_version,
        cache_namespace=cache_namespace,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
        max_completion_tokens=6144,
        status=status,
        error_code=error_code,
        raw_response_text=raw,
        response_redaction_reason=redaction,
        finish_reason=finish_reason,
        tool_calls=(),
        parsed_feedback=None if status != "parsed" else SimpleNamespace(),
        usage=usage,
        request_id="request-1",
        query_id="query",
        packet_sha256="1" * 64,
        prompt_sha256="2" * 64,
        image_sha256="3" * 64,
        wire_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        remote_authorization_id="remote-auth",
        remote_authorization_file_sha256="6" * 64,
        remote_receipt_file_sha256="7" * 64,
        remote_receipt_sha256="8" * 64,
        provider="qwen",
        model="qwen3.8-max",
        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
        result_sha256="8" * 64,
    )


def _invalid_schema_result(*, finish_reason: str = "stop") -> FeedbackEvaluationResult:
    return _result(
        status="parse_error",
        error_code="invalid_feedback_json",
        finish_reason=finish_reason,
        raw='{"schema_version":1,"summary":"x","rule_violations":["bad"],'
        '"ideal_response_gaps":[],"skill_suggestions":[]}',
    )


def _prepared(tmp_path: Path, count: int) -> PreparedPortfolioS1FeedbackV3Run:
    entries = tuple(
        SimpleNamespace(
            entry_sha256=f"{ordinal:064x}",
            selection_ordinal=ordinal,
            query_id=f"query-{ordinal}",
        )
        for ordinal in range(1, count + 1)
    )
    selection = PortfolioS1FeedbackSelectionV2.model_construct(
        selection_sha256="b" * 64,
        corpus_sha256="d" * 64,
        entries=entries,
        phase_counts=(12, 60, 120, 240),
    )
    control = PortfolioS1FeedbackControlV12.model_construct(
        control_sha256="c" * 64,
        selection_sha256=selection.selection_sha256,
        provider_call_ceiling=243,
    )
    sources = tuple(
        VerifiedStaticFeedbackSourceV2(
            corpus_sha256="d" * 64,
            selection_sha256=selection.selection_sha256,
            control_sha256=control.control_sha256,
            selection_entry_sha256=entry.entry_sha256,
            corpus=SimpleNamespace(),
            row=SimpleNamespace(query_ordinal=entry.selection_ordinal),
            asset_catalog=SimpleNamespace(),
            packet=SimpleNamespace(query_id=entry.query_id),
            _marker=object(),
        )
        for entry in entries
    )
    output = tmp_path / "fresh-v3"
    output.mkdir()
    return PreparedPortfolioS1FeedbackV3Run(
        output_dir=output,
        selection=selection,
        authorization=PortfolioS1FeedbackAuthorizationV7.model_construct(
            authorization_id="auth-v3",
            owner_approved_phase_hard_cap_cny="113.000000000000",
        ),
        control=control,
        sources=sources,
        remote_runtime=SimpleNamespace(),
        launch_lock=SimpleNamespace(),
        model_source_lock=SimpleNamespace(),
        pricing_lock=SimpleNamespace(),
        role_selection=SimpleNamespace(),
    )


def _artifact(
    prepared: PreparedPortfolioS1FeedbackV3Run,
    source: VerifiedStaticFeedbackSourceV2,
    *,
    attempt_index: int,
    global_call_ordinal: int,
    result: FeedbackEvaluationResult,
) -> BoundFeedbackArtifactV5:
    return BoundFeedbackArtifactV5.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=source.selection_entry_sha256,
        query_id=source.packet.query_id,
        attempt_index=attempt_index,
        global_call_ordinal=global_call_ordinal,
        status=result.status,
        feedback_result=result,
        reservation_sha256=f"{global_call_ordinal + 2000:064x}",
        retry_claim_ordinal=None,
        artifact_sha256=f"{global_call_ordinal + 1000:064x}",
    )


def test_policy_producer_matches_final_governance_and_transport() -> None:
    assert QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2 == GOVERNANCE_RETRY_SHA
    assert sha256_bytes(
        canonical_json_bytes(qwen38_feedback_global_retry_policy_v2())
    ) == GOVERNANCE_RETRY_SHA
    assert VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7 == GOVERNANCE_TRANSPORT_SHA


def test_forward_budget_authority_distinguishes_cny150_from_technical_cny113(
) -> None:
    models = (
        PortfolioS1FeedbackAuthorizationV7.model_construct(
            owner_authorized_budget_ceiling_cny="150.000000000000",
            approved_technical_phase_hard_cap_cny="113.000000000000",
            owner_approved_phase_hard_cap_cny="113.000000000000",
            phase_hard_cap_cny="113.000000000000",
            owner_statement=FRESH_V3_OWNER_BUDGET_STATEMENT,
        ),
        PortfolioS1FeedbackControlV12.model_construct(
            owner_authorized_budget_ceiling_cny="150.000000000000",
            approved_technical_phase_hard_cap_cny="113.000000000000",
        ),
        PortfolioS1Qwen38FeedbackLaunchLockV5.model_construct(
            owner_authorized_budget_ceiling_cny="150.000000000000",
            approved_technical_phase_hard_cap_cny="113.000000000000",
            owner_approved_phase_hard_cap_cny="113.000000000000",
            phase_hard_cap_cny="113.000000000000",
        ),
    )
    for model in models:
        assert model.owner_authorized_budget_ceiling_cny == "150.000000000000"
        assert model.approved_technical_phase_hard_cap_cny == "113.000000000000"
    assert models[0].owner_statement == FRESH_V3_OWNER_BUDGET_STATEMENT
    option_strings = {
        option
        for action in build_parser_v3()._actions
        for option in action.option_strings
    }
    assert "--owner-authorized-budget-ceiling-cny" in option_strings
    assert "--approved-technical-phase-hard-cap-cny" in option_strings
    assert "--approved-phase-hard-cap-cny" not in option_strings
    assert "--owner-statement" not in option_strings


@pytest.mark.parametrize(
    ("timestamp", "expected_offset_seconds"),
    (
        ("2026-08-10T14:30:44Z", 0),
        ("2026-08-10T22:30:44+08:00", 8 * 60 * 60),
    ),
)
def test_authorization_v7_builder_preserves_aware_datetime_and_strict_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timestamp: str,
    expected_offset_seconds: int,
) -> None:
    reviewed_at = _parse_reviewed_at_v3(timestamp)
    assets = tuple(
        SelectedQwenFeedbackAssetV1(
            query_id=f"query-{ordinal:03d}",
            asset_id=f"asset-{ordinal:03d}",
            image_sha256=f"{ordinal:064x}",
        )
        for ordinal in range(240)
    )
    selection = PortfolioS1FeedbackSelectionV2.model_construct(
        selection_sha256="b" * 64
    )
    parent = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="parent-auth"),
        authorization_file_sha256="1" * 64,
        receipt_file_sha256="2" * 64,
        receipt=SimpleNamespace(receipt_sha256="3" * 64),
        catalog=SimpleNamespace(catalog_sha256="4" * 64),
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback_retry_v3."
        "require_verified_portfolio_remote_processing_runtime",
        lambda runtime, **_kwargs: runtime,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback_retry_v3."
        "_selected_qwen_feedback_assets_v4",
        lambda _selection: assets,
    )

    authorization = build_selected_qwen38_feedback_authorization_v7(
        selection,
        parent,
        authorization_id="fresh-v3-auth",
        reviewer_id="owner",
        reviewed_at=reviewed_at,
        owner_authorized_budget_ceiling_cny="150.000000000000",
        approved_technical_phase_hard_cap_cny="113.000000000000",
        model_source_lock_file_sha256=(
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2
        ),
        model_source_lock_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2,
        pricing_lock_file_sha256=QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
        pricing_lock_sha256=QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5,
        role_selection_file_sha256=(
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12
        ),
        role_selection_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12,
    )
    assert type(authorization.reviewed_at) is datetime
    assert authorization.reviewed_at.utcoffset() is not None
    assert (
        authorization.reviewed_at.utcoffset().total_seconds()
        == expected_offset_seconds
    )

    content = authorization.canonical_bytes()
    path = tmp_path / "authorization-v7.json"
    path.write_bytes(content)
    loaded = load_selected_qwen38_feedback_authorization_v7(
        path, expected_file_sha256=sha256_bytes(content)
    )
    assert loaded == authorization
    assert type(loaded.reviewed_at) is datetime


@pytest.mark.parametrize(
    "timestamp",
    (
        "2026-08-10T22:30:44",
        "2026-08-10 22:30:44+08:00",
        "2026-08-10T22:30:44.000000+08:00",
        "2026-08-10T14:30:44+00:00",
    ),
)
def test_reviewed_at_v3_rejects_naive_or_noncanonical_values(
    timestamp: str,
) -> None:
    with pytest.raises(
        PortfolioS1FeedbackV3RunError, match="canonical timezone-aware"
    ):
        _parse_reviewed_at_v3(timestamp)


def test_live_governance_triad_binds_cny150_owner_and_cny113_technical() -> None:
    source = load_qwen38_feedback_model_source_lock_v2(
        Path("specs/authoring/qwen3.8-max-feedback-source-lock-v2.json"),
        expected_file_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2,
    )
    pricing = load_qwen38_feedback_pricing_lock_v5(
        Path("specs/authoring/price-qwen3.8-max-feedback-v5.json"),
        expected_file_sha256=QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
    )
    role = load_qwen38_feedback_role_selection_v12(
        Path("specs/authoring/model-role-selection-v12.json"),
        expected_file_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12,
    )
    _require_live_governance_locks_v3(source, pricing, role)
    assert pricing.owner_budget_authorized_cap_cny == "150.000000000000"
    assert pricing.technical_phase_hard_cap_cny == "113.000000000000"
    assert role.feedback_evaluator["owner_authorized_budget_ceiling_cny"] == (
        "150.000000000000"
    )
    assert role.feedback_evaluator["technical_phase_hard_cap_cny"] == (
        "113.000000000000"
    )


def test_eligibility_is_fresh_schema_or_length_only() -> None:
    assert is_qwen38_schema_or_length_retry_eligible(_invalid_schema_result())
    assert is_qwen38_schema_or_length_retry_eligible(
        _invalid_schema_result(finish_reason="length")
    )
    valid_parser_text = canonical_json_bytes(
        {
            "schema_version": 1,
            "summary": "x",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": ["policy label missing"],
        }
    ).decode()
    assert not is_qwen38_schema_or_length_retry_eligible(
        _result(
            status="parse_error",
            error_code="invalid_feedback_json",
            raw=valid_parser_text,
        )
    )
    assert not is_qwen38_schema_or_length_retry_eligible(
        _result(
            status="parse_error",
            error_code="invalid_feedback_json",
            finish_reason="length",
            cache_namespace="feedback-evaluator-v11",
            schema_version=5,
        )
    )


def test_three_global_retries_succeed_in_first_call_order(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 6)
    calls: list[tuple[str, int]] = []
    ordinal = 0

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        nonlocal ordinal
        ordinal += 1
        calls.append((source.packet.query_id, attempt_index))
        failed_first = int(source.packet.query_id.split("-")[1]) <= 3
        result = (
            _invalid_schema_result(finish_reason="length")
            if failed_first and attempt_index == 1
            else _result()
        )
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=ordinal,
            result=result,
        )

    artifacts: dict[tuple[str, int], BoundFeedbackArtifactV5] = {}
    _execute_phase_v5(
        prepared,
        prepared.sources,
        artifacts,
        lambda *_a, **_k: None,
        run_source=run_source,
    )
    assert len(calls) == 9
    assert [item for item in calls if item[1] == 2] == [
        ("query-1", 2),
        ("query-2", 2),
        ("query-3", 2),
    ]
    assert tuple(
        path.name
        for path in sorted((prepared.output_dir / "global-retry-claims-v2").iterdir())
    ) == ("claim-01.json", "claim-02.json", "claim-03.json")


def test_normal_240_uses_no_retry_or_claim(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 240)
    calls = 0

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        nonlocal calls
        calls += 1
        assert attempt_index == 1
        return _artifact(
            prepared,
            source,
            attempt_index=1,
            global_call_ordinal=calls,
            result=_result(),
        )

    artifacts: dict[tuple[str, int], BoundFeedbackArtifactV5] = {}
    _execute_phase_v5(
        prepared,
        prepared.sources,
        artifacts,
        lambda *_a, **_k: None,
        run_source=run_source,
    )
    assert calls == 240
    assert len(artifacts) == 240
    assert not (prepared.output_dir / "global-retry-claims-v2").exists()


def test_completed_run_v5_records_parsed240_and_no_terminal_reason(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path, 240)
    artifacts = tuple(
        _artifact(
            prepared,
            source,
            attempt_index=1,
            global_call_ordinal=ordinal,
            result=_result(),
        )
        for ordinal, source in enumerate(prepared.sources, start=1)
    )
    run = build_portfolio_s1_feedback_run_v5(
        prepared.selection, prepared.control, artifacts
    )
    assert run.status == "completed"
    assert run.terminal_reason is None
    assert run.attempted_count == 240
    assert run.parsed_count == 240
    assert run.provider_calls_reserved == 240


def test_fourth_eligible_is_terminal_without_retry_call(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 5)
    calls: list[tuple[str, int]] = []
    ordinal = 0

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        nonlocal ordinal
        ordinal += 1
        calls.append((source.packet.query_id, attempt_index))
        number = int(source.packet.query_id.split("-")[1])
        result = (
            _invalid_schema_result()
            if number <= 4 and attempt_index == 1
            else _result()
        )
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=ordinal,
            result=result,
        )

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="fourth eligible"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {},
            lambda *_a, **_k: None,
            run_source=run_source,
        )
    assert ("query-4", 2) not in calls
    # The fourth failure shares a wave with the third; the whole oversubscribed
    # wave fails closed before consuming its remaining single claim.
    assert sum(attempt == 2 for _query, attempt in calls) == 2


def test_same_wave_noneligible_precedes_eligible_claim(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 2)
    calls: list[tuple[str, int]] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append((source.packet.query_id, attempt_index))
        result = (
            _invalid_schema_result()
            if source.packet.query_id == "query-1"
            else _result(status="provider_error", error_code="provider_error", raw="")
        )
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls),
            result=result,
        )

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="same-wave"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {},
            lambda *_a, **_k: None,
            run_source=run_source,
        )
    assert all(attempt == 1 for _query, attempt in calls)
    assert not (prepared.output_dir / "global-retry-claims-v2").exists()


@pytest.mark.parametrize(
    ("status", "error_code", "redaction"),
    [
        ("provider_error", "provider_error", None),
        ("parse_error", "creator_projection_privacy", "creator_projection_privacy"),
        ("parse_error", "input_image_echo", "input_image_echo"),
    ],
)
def test_provider_privacy_and_echo_never_retry(
    tmp_path: Path,
    status: str,
    error_code: str,
    redaction: str | None,
) -> None:
    prepared = _prepared(tmp_path, 1)
    calls: list[int] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=1,
            result=_result(
                status=status,
                error_code=error_code,
                raw="redacted",
                redaction=redaction,
            ),
        )

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="nonretryable"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {},
            lambda *_a, **_k: None,
            run_source=run_source,
        )
    assert calls == [1]


def test_retry_failure_is_terminal_at_two_attempts(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 1)
    calls: list[int] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls),
            result=_invalid_schema_result(),
        )

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="retry attempt failed"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {},
            lambda *_a, **_k: None,
            run_source=run_source,
        )
    assert calls == [1, 2]


def test_orphaned_reservation_never_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 1)
    calls: list[int] = []

    def crash(_prepared, _source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        raise RuntimeError("post-reservation crash")

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((), (SimpleNamespace(),), ()),
    )
    with pytest.raises(PortfolioS1FeedbackV3RunError, match="orphan"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {},
            lambda *_a, **_k: None,
            run_source=crash,
        )
    assert calls == [1]


def test_crash_settlement_is_retried_before_new_first(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 2)
    first = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_invalid_schema_result(finish_reason="length"),
    )
    calls: list[tuple[str, int]] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append((source.packet.query_id, attempt_index))
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls) + 1,
            result=_result(),
        )

    artifacts = {(first.selection_entry_sha256, 1): first}
    _execute_phase_v5(
        prepared,
        prepared.sources,
        artifacts,
        lambda *_a, **_k: None,
        run_source=run_source,
    )
    assert calls == [("query-1", 2), ("query-2", 1)]


def test_crash_after_claim_resumes_same_retry_before_new_first(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 2)
    first = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_invalid_schema_result(),
    )
    artifacts = {(first.selection_entry_sha256, 1): first}
    claim = _claim_retry_v2(prepared, first, artifacts)
    assert claim.selection_entry_sha256 == first.selection_entry_sha256
    calls: list[tuple[str, int]] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append((source.packet.query_id, attempt_index))
        return _artifact(
            prepared,
            source,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls) + 1,
            result=_result(),
        )

    _execute_phase_v5(
        prepared,
        prepared.sources,
        artifacts,
        lambda *_a, **_k: None,
        run_source=run_source,
    )
    assert calls == [("query-1", 2), ("query-2", 1)]


def test_claim_set_rejects_self_consistent_wrong_selection_entry(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path, 2)
    first = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_invalid_schema_result(),
    )
    claim = build_feedback_global_retry_claim_v2(
        prepared.selection,
        prepared.control,
        prepared.selection.entries[0],
        first,
        existing_claims=(),
        existing_first_artifacts=(first,),
    )
    tampered = claim.model_copy(
        update={
            "selection_entry_sha256": prepared.selection.entries[1].entry_sha256,
            "query_id": prepared.selection.entries[1].query_id,
        }
    )
    with pytest.raises(PortfolioS1FeedbackError, match="claim set drifted"):
        validate_feedback_global_retry_claim_set_v2(
            prepared.selection,
            prepared.control,
            (tampered,),
            (first,),
        )


def test_late_third_claim_accepts_first_global_ordinal_242(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 3)
    firsts = tuple(
        _artifact(
            prepared,
            source,
            attempt_index=1,
            global_call_ordinal=ordinal,
            result=_invalid_schema_result(),
        )
        for source, ordinal in zip(prepared.sources, (1, 2, 242), strict=True)
    )
    claims = ()
    for entry, first in zip(prepared.selection.entries, firsts, strict=True):
        claim = build_feedback_global_retry_claim_v2(
            prepared.selection,
            prepared.control,
            entry,
            first,
            existing_claims=claims,
            existing_first_artifacts=firsts,
        )
        claims = (*claims, claim)
    assert claims[-1].first_global_call_ordinal == 242


def test_fresh_v3_rejects_historical_root_before_provider(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 1)
    legacy = prepared.output_dir / "provider-attempts-v2"
    legacy.mkdir()
    (legacy / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PortfolioS1FeedbackV3RunError, match="historical"):
        _reject_historical_roots(prepared)


def test_fresh_v3_rejects_historical_top_level_run_marker(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 1)
    (prepared.output_dir / "portfolio-s1-feedback-run.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(PortfolioS1FeedbackV3RunError, match="historical"):
        _reject_historical_roots(prepared)


def test_checked_in_pending_authority_stops_before_provider(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 1)
    with pytest.raises(PortfolioS1FeedbackV3RunError, match="lacks exact CNY150"):
        _require_live_authority(prepared)


def test_243_call_and_cny113_technical_envelope() -> None:
    assert require_qwen38_feedback_pre_call_budget_v3(
        estimated_input_tokens_including_images=20_000,
        provider_calls_already_reserved=242,
        committed_cost_cny=format(Decimal("0.461544") * 242, "f"),
    ) == "0.461544000000"
    assert Decimal("0.461544") * 243 == Decimal("112.155192")
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="exhausted"):
        require_qwen38_feedback_pre_call_budget_v3(
            estimated_input_tokens_including_images=20_000,
            provider_calls_already_reserved=243,
            committed_cost_cny="112.155192",
        )


def test_runtime_invalid_token_identity_fails_before_image_load() -> None:
    packet = SimpleNamespace()
    with pytest.raises((TypeError, ValueError)):
        run_visual_feedback(  # type: ignore[arg-type]
            packet,
            SimpleNamespace(),
            remote_runtime=SimpleNamespace(),
            max_completion_tokens=5000,
        )


def test_execute_writer_lock_conflict_is_zero_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path, 1)
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._require_live_authority",
        lambda _prepared: None,
    )
    (prepared.output_dir / "execute-writer.lock").write_text(
        "occupied", encoding="utf-8"
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _result()

    with pytest.raises(RuntimeError, match="writer lock already exists"):
        execute_run_v3(prepared, feedback_runner=provider)
    assert calls == 0


@pytest.mark.parametrize(
    ("directory", "name"),
    [
        ("provider-attempts-v3", "unexpected.json"),
        ("bound-feedback-v3", "unexpected.json"),
        ("global-retry-claims-v2", "claim-04.json"),
    ],
)
def test_strict_v3_inventory_drift_is_zero_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    directory: str,
    name: str,
) -> None:
    prepared = _prepared(tmp_path, 1)
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._require_live_authority",
        lambda _prepared: None,
    )
    root = prepared.output_dir / directory
    root.mkdir()
    (root / name).write_text("{}", encoding="utf-8")
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _result()

    with pytest.raises(
        PortfolioS1FeedbackV3RunError, match="inventory|file set"
    ):
        execute_run_v3(prepared, feedback_runner=provider)
    assert calls == 0


def test_resumed_failed_retry_is_terminal_before_new_first(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, 2)
    first = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_invalid_schema_result(),
    )
    retry = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=2,
        global_call_ordinal=2,
        result=_invalid_schema_result(),
    )
    calls = 0

    def run_source(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="retry attempt failed"):
        _execute_phase_v5(
            prepared,
            prepared.sources,
            {
                (first.selection_entry_sha256, 1): first,
                (retry.selection_entry_sha256, 2): retry,
            },
            lambda *_a, **_k: None,
            run_source=run_source,
        )
    assert calls == 0


def test_terminal_run_preserves_all_attempt_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    first = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_result(usage=LLMUsage(input_tokens=100, output_tokens=20)),
    )
    failed = _artifact(
        prepared,
        prepared.sources[1],
        attempt_index=1,
        global_call_ordinal=2,
        result=_result(
            status="provider_error",
            error_code="provider_error",
            raw="",
            usage=LLMUsage(input_tokens=50, output_tokens=10),
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((first, failed), (), ()),
    )
    run = _publish_terminal_run_v5(prepared)
    assert run is not None
    assert run.status == "stopped_nonparsed"
    assert run.provider_calls_reserved == 2
    assert run.usage_known_count == 2
    assert run.input_tokens == 150
    assert run.output_tokens == 30
    assert run.actual_cost_cny == "0.002880000000"
    assert len(run.artifacts) == 2


def test_initial_orphan_publishes_terminal_run_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    source = prepared.sources[0]
    orphan = FeedbackCallReservationV5.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=source.selection_entry_sha256,
        selection_ordinal=1,
        query_id=source.packet.query_id,
        attempt_index=1,
        global_call_ordinal=1,
        previous_artifact_sha256=None,
        retry_claim_sha256=None,
        retry_claim_ordinal=None,
        reservation_sha256="9" * 64,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((), (orphan,), ()),
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="orphan"):
        _execute_run_v3_locked(prepared, feedback_runner=provider)
    terminal = json.loads((prepared.output_dir / "run-v5.json").read_bytes())
    assert terminal["status"] == "stopped_orphan"
    assert terminal["orphan_count"] == 1
    assert calls == 0


def test_budget_breach_and_sibling_orphan_publish_combined_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    over_usage = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_result(usage=LLMUsage(input_tokens=20_001, output_tokens=0)),
    )
    orphan = FeedbackCallReservationV5.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=prepared.sources[1].selection_entry_sha256,
        selection_ordinal=2,
        query_id=prepared.sources[1].packet.query_id,
        attempt_index=1,
        global_call_ordinal=2,
        previous_artifact_sha256=None,
        retry_claim_sha256=None,
        retry_claim_ordinal=None,
        reservation_sha256="9" * 64,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((over_usage,), (orphan,), ()),
    )
    run = _publish_terminal_run_v5(prepared)
    assert run is not None
    assert run.status == "stopped_orphan"
    assert run.terminal_reason == "usage_limit_exceeded"
    assert run.provider_calls_reserved == 2
    assert run.orphan_count == 1
    assert len(run.artifacts) == 2


def test_noneligible_failure_publishes_terminal_run_before_new_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    failed = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_result(
            status="provider_error", error_code="provider_error", raw=""
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((failed,), (), ()),
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    with pytest.raises(PortfolioS1FeedbackV3RunError, match="nonretryable"):
        _execute_run_v3_locked(prepared, feedback_runner=provider)
    terminal = json.loads((prepared.output_dir / "run-v5.json").read_bytes())
    assert terminal["status"] == "stopped_nonparsed"
    assert terminal["error_count"] == 1
    assert calls == 0


def test_run_written_bundle_missing_resumes_without_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 240)
    artifacts = tuple(
        _artifact(
            prepared,
            source,
            attempt_index=1,
            global_call_ordinal=ordinal,
            result=_result(),
        )
        for ordinal, source in enumerate(prepared.sources, start=1)
    )
    run_bytes = b'{"completed":"run"}'
    bundle_bytes = b'{"completed":"bundle"}'
    fake_run = SimpleNamespace(
        status="completed",
        run_sha256="a" * 64,
        canonical_bytes=lambda: run_bytes,
    )
    fake_bundle = SimpleNamespace(
        bundle_sha256="b" * 64,
        canonical_bytes=lambda: bundle_bytes,
    )
    (prepared.output_dir / "run-v5.json").write_bytes(run_bytes)
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: (artifacts, (), ()),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.build_portfolio_s1_feedback_run_v5",
        lambda *_args, **_kwargs: fake_run,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.build_portfolio_s1_feedback_bundle_v8",
        lambda *_args, **_kwargs: fake_bundle,
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    outcome = _execute_run_v3_locked(prepared, feedback_runner=provider)
    assert outcome == (fake_run, fake_bundle)
    assert (prepared.output_dir / "bundle-v8.json").read_bytes() == bundle_bytes
    assert calls == 0


@pytest.mark.parametrize(
    "usage",
    [
        LLMUsage(input_tokens=20_001, output_tokens=0),
        LLMUsage(input_tokens=0, output_tokens=6_155),
    ],
)
def test_over_usage_publishes_budget_terminal_and_calls_no_next_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    usage: LLMUsage,
) -> None:
    prepared = _prepared(tmp_path, 2)
    bad = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_result(usage=usage),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((bad,), (), ()),
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    with pytest.raises(PortfolioS1FeedbackV3BudgetError, match="usage exceeded"):
        _execute_run_v3_locked(prepared, feedback_runner=provider)
    terminal = json.loads((prepared.output_dir / "run-v5.json").read_bytes())
    assert terminal["status"] == "stopped_budget"
    assert terminal["terminal_reason"] == "usage_limit_exceeded"
    assert calls == 0


def test_accountable_cap_breach_publishes_terminal_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    artifact = _artifact(
        prepared,
        prepared.sources[0],
        attempt_index=1,
        global_call_ordinal=1,
        result=_result(usage=LLMUsage(input_tokens=1, output_tokens=1)),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((artifact,), (), ()),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2",
        "0.000001000000",
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    with pytest.raises(PortfolioS1FeedbackV3BudgetError, match="hard cap"):
        _execute_run_v3_locked(prepared, feedback_runner=provider)
    terminal = json.loads((prepared.output_dir / "run-v5.json").read_bytes())
    assert terminal["terminal_reason"] == "accountable_cost_exceeded"
    assert calls == 0


@pytest.mark.parametrize(
    ("target", "expected_phases"),
    [(12, [12]), (60, [12, 60]), (120, [12, 60, 120])],
)
def test_phase_stops_are_exact_and_do_not_publish_terminal_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: int,
    expected_phases: list[int],
) -> None:
    prepared = _prepared(tmp_path, 240)
    observed: list[int] = []
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._attempt_state_v5",
        lambda _prepared: ((), (), ()),
    )

    def phase(_prepared, sources, artifacts, _runner):
        observed.append(len(sources))
        for ordinal, source in enumerate(sources, start=1):
            artifacts.setdefault(
                (source.selection_entry_sha256, 1),
                _artifact(
                    prepared,
                    source,
                    attempt_index=1,
                    global_call_ordinal=ordinal,
                    result=_result(),
                ),
            )

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._execute_phase_v5", phase
    )
    assert (
        _execute_run_v3_locked(
            prepared,
            feedback_runner=lambda *_a, **_k: pytest.fail("provider called"),
            stop_after_count=target,  # type: ignore[arg-type]
        )
        is None
    )
    assert observed == expected_phases
    assert not (prepared.output_dir / "run-v5.json").exists()


def test_actual_argparse_prepare_parses_canonical_datetime_and_exact_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "fresh-v3-prepare"
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("fixture", encoding="utf-8")
    parent_runtime = SimpleNamespace()
    corpus = SimpleNamespace(
        core_inputs=SimpleNamespace(
            runtime_for=lambda processor: (
                parent_runtime
                if processor == "dashscope-qwen-assistant"
                else pytest.fail("unexpected processor")
            )
        )
    )
    selection = SimpleNamespace(
        selection_sha256="a" * 64,
        canonical_bytes=lambda: b'{"kind":"selection-v2"}\n',
    )
    source_lock = SimpleNamespace(
        source_lock_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2,
        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    pricing_lock = SimpleNamespace(
        pricing_lock_sha256=QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5
    )
    role_selection = SimpleNamespace(
        selection_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12
    )
    observed_reviewed_at: list[datetime] = []

    def build_authorization(_selection, _parent, **kwargs):
        observed_reviewed_at.append(kwargs["reviewed_at"])
        return SimpleNamespace(
            canonical_bytes=lambda: b'{"kind":"authorization-v7"}\n'
        )

    control = SimpleNamespace(
        control_sha256="c" * 64,
        canonical_bytes=lambda: b'{"kind":"control-v12"}\n',
    )
    launch = SimpleNamespace(
        canonical_bytes=lambda: b'{"kind":"launch-lock-v5"}\n'
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.load_verified_static_gcs_corpus",
        lambda *_args, **_kwargs: corpus,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._load_discovery600",
        lambda *_args, **_kwargs: (("query",), "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "build_portfolio_s1_feedback_selection_v2",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "load_qwen38_feedback_model_source_lock_v2",
        lambda *_args, **_kwargs: source_lock,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "load_qwen38_feedback_pricing_lock_v5",
        lambda *_args, **_kwargs: pricing_lock,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "load_qwen38_feedback_role_selection_v12",
        lambda *_args, **_kwargs: role_selection,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._require_live_governance_locks_v3",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "build_selected_qwen38_feedback_authorization_v7",
        build_authorization,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._rubric",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "build_portfolio_s1_feedback_control_v12",
        lambda *_args, **_kwargs: control,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "build_qwen38_feedback_launch_lock_v5",
        lambda **_kwargs: launch,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "build_verified_static_feedback_sources_v2",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._validate_input_reservation_bounds",
        lambda _sources: None,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3."
        "prepare_selected_qwen38_feedback_remote_runtime_v7",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    arguments = build_parser_v3().parse_args(
        [
            "--mode",
            "prepare",
            "--execution-root",
            str(placeholder),
            "--expected-execution-control-sha256",
            "x",
            "--artifact-repository-root",
            str(placeholder),
            "--fold-manifest",
            str(placeholder),
            "--expected-fold-manifest-sha256",
            "x",
            "--fold-mapping",
            str(placeholder),
            "--expected-fold-mapping-sha256",
            "x",
            "--rubric-file",
            str(placeholder),
            "--expected-rubric-sha256",
            "x",
            "--rubric-id",
            "rubric",
            "--rubric-version",
            "v1",
            "--output-dir",
            str(output),
            "--authorization-id",
            "auth",
            "--reviewer-id",
            "reviewer",
            "--reviewed-at",
            "2026-08-10T22:30:44+08:00",
            "--owner-authorized-budget-ceiling-cny",
            "150.000000000000",
            "--approved-technical-phase-hard-cap-cny",
            "113.000000000000",
            "--run-id",
            "run",
            "--model-source-lock",
            str(placeholder),
            "--expected-model-source-lock-sha256",
            QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2,
            "--pricing-lock",
            str(placeholder),
            "--expected-pricing-lock-sha256",
            QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5,
            "--role-selection-file",
            str(placeholder),
            "--expected-role-selection-file-sha256",
            QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12,
            "--expected-role-selection-sha256",
            QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12,
        ]
    )

    first = prepare_run_v3(arguments)
    assert first.output_dir == output.absolute()
    assert observed_reviewed_at[0].utcoffset() is not None
    assert observed_reviewed_at[0].utcoffset().total_seconds() == 8 * 60 * 60
    (output / "launch-lock-v5.json").unlink()
    resumed = prepare_run_v3(arguments)
    assert resumed.output_dir == first.output_dir
    assert len(observed_reviewed_at) == 2
    assert observed_reviewed_at[1] == observed_reviewed_at[0]
    assert (output / "launch-lock-v5.json").read_bytes() == (
        b'{"kind":"launch-lock-v5"}\n'
    )


def test_cli_execute_wires_fresh_live_runner_and_phase_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared = _prepared(tmp_path, 1)
    observed: list[int | None] = []
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.prepare_run_v3",
        lambda _arguments: prepared,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3.execute_live_run_v3",
        lambda _prepared, *, stop_after_count: observed.append(stop_after_count),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback_v3._summary_v3",
        lambda *_args, **_kwargs: {"ok": True},
    )
    required_paths = tmp_path / "placeholder"
    args = [
        "--mode",
        "execute",
        "--execution-root",
        str(required_paths),
        "--expected-execution-control-sha256",
        "x",
        "--artifact-repository-root",
        str(required_paths),
        "--fold-manifest",
        str(required_paths),
        "--expected-fold-manifest-sha256",
        "x",
        "--fold-mapping",
        str(required_paths),
        "--expected-fold-mapping-sha256",
        "x",
        "--rubric-file",
        str(required_paths),
        "--expected-rubric-sha256",
        "x",
        "--rubric-id",
        "rubric",
        "--rubric-version",
        "v1",
        "--output-dir",
        str(tmp_path / "output"),
        "--authorization-id",
        "auth",
        "--reviewer-id",
        "reviewer",
        "--reviewed-at",
        "2026-08-10T00:00:00Z",
        "--owner-authorized-budget-ceiling-cny",
        "150.000000000000",
        "--approved-technical-phase-hard-cap-cny",
        "113.000000000000",
        "--run-id",
        "run",
        "--model-source-lock",
        str(required_paths),
        "--expected-model-source-lock-sha256",
        "x",
        "--pricing-lock",
        str(required_paths),
        "--expected-pricing-lock-sha256",
        "x",
        "--role-selection-file",
        str(required_paths),
        "--expected-role-selection-file-sha256",
        "x",
        "--expected-role-selection-sha256",
        "x",
        "--stop-after-count",
        "120",
    ]
    assert main(args) == 0
    assert observed == [120]
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert build_parser_v3().parse_args(args).stop_after_count == 120
