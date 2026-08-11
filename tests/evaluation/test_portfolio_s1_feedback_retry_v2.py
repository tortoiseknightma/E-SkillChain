from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_portfolio_s1_feedback import (
    PreparedPortfolioS1FeedbackRun,
    PortfolioS1FeedbackRunError,
    _claim_retry_v1,
    _execute_phase_v4,
    _execute_run_locked,
    _load_qwen_role_selection,
)
from skillchain.evaluation.feedback_runtime import FeedbackEvaluationResult
from skillchain.evaluation.portfolio_s1_feedback import (
    BoundFeedbackArtifactV4,
    FeedbackCallReservationV4,
    PortfolioS1FeedbackAuthorizationV6,
    PortfolioS1FeedbackControlV11,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1,
    VerifiedStaticFeedbackSourceV2,
    build_bound_feedback_artifact_v4,
    is_qwen38_strict_schema_retry_eligible,
)
from skillchain.synthesis.store import canonical_json_bytes


ROLE_V10_FILE_SHA256 = (
    "39e8d099f0b303725ee0be59f758560627ecbc3ce156524da4f2a2d3d20f3501"
)
ROLE_V10_SHA256 = "be5f7f7ee1f30a81d7d2a8423384adb41d74e4673000901e883af704e9eead16"
ROLE_V11_FILE_SHA256 = (
    "048362dd770c626ac4c2293b1d8a28cb4fcb2a3a240c4f79dca745e4dbb5c50c"
)
ROLE_V11_SHA256 = "8f9252b9dc4ddf6f3d6569f783e150f55f7f6cd422b77126e73b5c38ba99e432"


def _result(
    *,
    status: str = "parsed",
    error_code: str | None = None,
    raw: str = "{}",
    redaction: str | None = None,
    finish_reason: str = "stop",
    tool_calls: tuple[object, ...] = (),
    wire_sha256: str = "a" * 64,
) -> FeedbackEvaluationResult:
    return FeedbackEvaluationResult.model_construct(
        status=status,
        error_code=error_code,
        raw_response_text=raw,
        response_redaction_reason=redaction,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        parsed_feedback=None if status != "parsed" else SimpleNamespace(),
        request_id="request-1",
        packet_sha256="1" * 64,
        image_sha256="2" * 64,
        prompt_sha256="3" * 64,
        wire_sha256=wire_sha256,
        asset_catalog_sha256="4" * 64,
        remote_authorization_id="auth-v2",
        remote_authorization_file_sha256="5" * 64,
        remote_receipt_file_sha256="6" * 64,
        remote_receipt_sha256="7" * 64,
        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
        result_sha256="8" * 64,
        usage=None,
    )


def _invalid_schema_result() -> FeedbackEvaluationResult:
    return _result(
        status="parse_error",
        error_code="invalid_feedback_json",
        raw='{"schema_version":1,"summary":"x","rule_violations":["bad"],'
        '"ideal_response_gaps":[],"skill_suggestions":[]}',
    )


def _prepared(tmp_path: Path, count: int) -> PreparedPortfolioS1FeedbackRun:
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
        entries=entries,
        phase_counts=(12, 60, 120, 240),
    )
    control = PortfolioS1FeedbackControlV11.model_construct(
        control_sha256="c" * 64,
        selection_sha256=selection.selection_sha256,
        provider_call_ceiling=241,
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
    output = tmp_path / "fresh-v2"
    output.mkdir()
    return PreparedPortfolioS1FeedbackRun(
        output_dir=output,
        corpus=SimpleNamespace(),
        selection=selection,
        authorization=PortfolioS1FeedbackAuthorizationV6.model_construct(
            authorization_id="auth-v2"
        ),
        control=control,
        sources=sources,
        remote_runtime=SimpleNamespace(),
    )


def _install_phase_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._budget_snapshot",
        lambda _prepared: SimpleNamespace(
            provider_calls_reserved=0,
            accountable_cost_cny=Decimal("0"),
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._publish_terminal_run",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )


def test_retry_policy_and_role_v11_are_exact_and_v10_is_not_active() -> None:
    assert QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1 == (
        "6ba3799fce29820466446c6ec0ee98312c6e889fc17c255b25453b9f70694995"
    )
    assert _load_qwen_role_selection(
        Path("specs/authoring/model-role-selection-v11.json"),
        expected_file_sha256=ROLE_V11_FILE_SHA256,
        expected_selection_sha256=ROLE_V11_SHA256,
        round2=True,
        round2_retry_required=True,
    ) == (ROLE_V11_FILE_SHA256, ROLE_V11_SHA256)
    with pytest.raises(PortfolioS1FeedbackRunError, match="rejects historical"):
        _load_qwen_role_selection(
            Path("specs/authoring/model-role-selection-v10.json"),
            expected_file_sha256=ROLE_V10_FILE_SHA256,
            expected_selection_sha256=ROLE_V10_SHA256,
            round2=True,
            round2_retry_required=True,
        )


def test_retry_eligibility_requires_parser_v3_itself_to_fail() -> None:
    assert is_qwen38_strict_schema_retry_eligible(_invalid_schema_result())
    parser_valid_but_policy_label_invalid = canonical_json_bytes(
        {
            "schema_version": 1,
            "summary": "x",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": ["missing policy disposition"],
        }
    ).decode()
    result = _result(
        status="parse_error",
        error_code="invalid_feedback_json",
        raw=parser_valid_but_policy_label_invalid,
    )
    assert not is_qwen38_strict_schema_retry_eligible(result)
    assert not is_qwen38_strict_schema_retry_eligible(
        _result(status="provider_error", error_code="provider_error", raw="")
    )
    assert not is_qwen38_strict_schema_retry_eligible(
        _result(
            status="parse_error",
            error_code="creator_projection_privacy",
            raw="redacted",
            redaction="creator_projection_privacy",
        )
    )


def test_bound_retry_rejects_same_entry_wire_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_result = _invalid_schema_result()
    first = BoundFeedbackArtifactV4.model_construct(
        feedback_result=first_result,
        artifact_sha256="9" * 64,
    )
    reservation = FeedbackCallReservationV4.model_construct(
        attempt_index=2,
        global_call_ordinal=2,
        previous_artifact_sha256=first.artifact_sha256,
        retry_claim_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback.build_feedback_call_reservation_v4",
        lambda *_args, **_kwargs: reservation,
    )
    packet = SimpleNamespace()
    with pytest.raises(PortfolioS1FeedbackError, match="source drifted"):
        build_bound_feedback_artifact_v4(
            SimpleNamespace(),
            SimpleNamespace(
                authorization_id="auth-v2", authorization_file_sha256="5" * 64
            ),
            SimpleNamespace(image_sha256="2" * 64),
            packet,
            _result(wire_sha256="f" * 64),
            verified_source=SimpleNamespace(packet=packet),
            reservation=reservation,
            first_artifact=first,
            retry_claim=SimpleNamespace(),
        )


def test_fresh_v2_normal_240_uses_zero_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 240)
    _install_phase_fakes(monkeypatch)
    calls: list[tuple[str, int]] = []
    ordinal = 0

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        nonlocal ordinal
        ordinal += 1
        calls.append((source.packet.query_id, attempt_index))
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=ordinal,
            status="parsed",
            feedback_result=_result(),
            artifact_sha256=f"{ordinal + 1000:064x}",
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    artifacts: dict[str, object] = {}
    _execute_phase_v4(prepared, prepared.sources, artifacts, lambda *_a, **_k: None)
    assert len(calls) == 240
    assert all(attempt == 1 for _query, attempt in calls)
    assert len(artifacts) == 240
    assert not (prepared.output_dir / "global-retry-claim.json").exists()


def test_one_global_schema_retry_succeeds_on_same_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 3)
    _install_phase_fakes(monkeypatch)
    calls: list[tuple[str, int]] = []
    ordinal = 0

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        nonlocal ordinal
        ordinal += 1
        calls.append((source.packet.query_id, attempt_index))
        result = (
            _invalid_schema_result()
            if source.packet.query_id == "query-1" and attempt_index == 1
            else _result()
        )
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=ordinal,
            status=result.status,
            feedback_result=result,
            artifact_sha256=f"{ordinal + 2000:064x}",
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    artifacts: dict[str, object] = {}
    _execute_phase_v4(prepared, prepared.sources, artifacts, lambda *_a, **_k: None)
    assert calls.count(("query-1", 1)) == 1
    assert calls.count(("query-1", 2)) == 1
    assert len(calls) == 4
    assert artifacts[prepared.selection.entries[0].entry_sha256].status == "parsed"


@pytest.mark.parametrize("retry_status", ["parse_error", "provider_error"])
def test_retry_failure_is_terminal_and_never_gets_third_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retry_status: str
) -> None:
    prepared = _prepared(tmp_path, 1)
    _install_phase_fakes(monkeypatch)
    calls: list[int] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        result = (
            _invalid_schema_result()
            if attempt_index == 1 or retry_status == "parse_error"
            else _result(status="provider_error", error_code="provider_error", raw="")
        )
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls),
            status=result.status,
            feedback_result=result,
            artifact_sha256=f"{len(calls) + 3000:064x}",
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    with pytest.raises(PortfolioS1FeedbackRunError, match="retry failed"):
        _execute_phase_v4(prepared, prepared.sources, {}, lambda *_a, **_k: None)
    assert calls == [1, 2]


def test_two_schema_failures_stop_without_spending_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    _install_phase_fakes(monkeypatch)
    calls: list[int] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls),
            status="parse_error",
            feedback_result=_invalid_schema_result(),
            artifact_sha256=f"{len(calls) + 4000:064x}",
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    with pytest.raises(PortfolioS1FeedbackRunError, match="second strict-schema"):
        _execute_phase_v4(prepared, prepared.sources, {}, lambda *_a, **_k: None)
    assert calls == [1, 1]
    assert not (prepared.output_dir / "global-retry-claim.json").exists()


@pytest.mark.parametrize(
    ("status", "error_code", "redaction"),
    [
        ("provider_error", "provider_error", None),
        ("parse_error", "creator_projection_privacy", "creator_projection_privacy"),
        ("parse_error", "input_image_echo", "input_image_echo"),
    ],
)
def test_provider_privacy_and_input_echo_never_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    error_code: str,
    redaction: str | None,
) -> None:
    prepared = _prepared(tmp_path, 1)
    _install_phase_fakes(monkeypatch)
    calls: list[int] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        result = _result(
            status=status, error_code=error_code, raw="redacted", redaction=redaction
        )
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=1,
            status=result.status,
            feedback_result=result,
            artifact_sha256="e" * 64,
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    with pytest.raises(PortfolioS1FeedbackRunError, match="nonretryable"):
        _execute_phase_v4(prepared, prepared.sources, {}, lambda *_a, **_k: None)
    assert calls == [1]


def test_crash_after_claim_before_second_reservation_resumes_only_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 1)
    _install_phase_fakes(monkeypatch)
    source = prepared.sources[0]
    first = BoundFeedbackArtifactV4.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=source.selection_entry_sha256,
        query_id=source.packet.query_id,
        attempt_index=1,
        global_call_ordinal=1,
        status="parse_error",
        feedback_result=_invalid_schema_result(),
        artifact_sha256="d" * 64,
    )
    _claim_retry_v1(prepared, first)
    calls: list[int] = []

    def run_source(_prepared, _source, _runner, *, attempt_index, **_kwargs):
        calls.append(attempt_index)
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=2,
            global_call_ordinal=2,
            status="parsed",
            feedback_result=_result(),
            artifact_sha256="c" * 64,
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    artifacts: dict[str, object] = {source.selection_entry_sha256: first}
    _execute_phase_v4(prepared, prepared.sources, artifacts, lambda *_a, **_k: None)
    assert calls == [2]
    assert artifacts[source.selection_entry_sha256].attempt_index == 2


def test_crash_after_first_settlement_before_claim_retries_before_new_first_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    _install_phase_fakes(monkeypatch)
    failed_source = prepared.sources[0]
    first = BoundFeedbackArtifactV4.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=failed_source.selection_entry_sha256,
        query_id=failed_source.packet.query_id,
        attempt_index=1,
        global_call_ordinal=1,
        status="parse_error",
        feedback_result=_invalid_schema_result(),
        artifact_sha256="b" * 64,
    )
    calls: list[tuple[str, int]] = []

    def run_source(_prepared, source, _runner, *, attempt_index, **_kwargs):
        calls.append((source.packet.query_id, attempt_index))
        return BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=attempt_index,
            global_call_ordinal=len(calls) + 1,
            status="parsed",
            feedback_result=_result(),
            artifact_sha256=f"{len(calls) + 5000:064x}",
        )

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", run_source)
    artifacts: dict[str, object] = {failed_source.selection_entry_sha256: first}
    _execute_phase_v4(prepared, prepared.sources, artifacts, lambda *_a, **_k: None)
    assert calls == [("query-1", 2), ("query-2", 1)]


def test_resume_with_two_eligible_first_settlements_stops_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    _install_phase_fakes(monkeypatch)
    artifacts = {
        source.selection_entry_sha256: BoundFeedbackArtifactV4.model_construct(
            selection_sha256=prepared.selection.selection_sha256,
            control_sha256=prepared.control.control_sha256,
            selection_entry_sha256=source.selection_entry_sha256,
            query_id=source.packet.query_id,
            attempt_index=1,
            global_call_ordinal=index,
            status="parse_error",
            feedback_result=_invalid_schema_result(),
            artifact_sha256=f"{index + 6000:064x}",
        )
        for index, source in enumerate(prepared.sources, 1)
    }
    provider_calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        pytest.fail("multiple settled failures reached provider")

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", forbidden)
    with pytest.raises(PortfolioS1FeedbackRunError, match="multiple settled"):
        _execute_phase_v4(prepared, prepared.sources, artifacts, lambda *_a, **_k: None)
    assert provider_calls == 0


def test_resume_claim_conflict_stops_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 2)
    _install_phase_fakes(monkeypatch)
    source = prepared.sources[0]
    first = BoundFeedbackArtifactV4.model_construct(
        selection_sha256=prepared.selection.selection_sha256,
        control_sha256=prepared.control.control_sha256,
        selection_entry_sha256=source.selection_entry_sha256,
        query_id=source.packet.query_id,
        attempt_index=1,
        global_call_ordinal=1,
        status="parse_error",
        feedback_result=_invalid_schema_result(),
        artifact_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._load_retry_claim_v1",
        lambda _prepared: SimpleNamespace(
            selection_entry_sha256=prepared.sources[1].selection_entry_sha256
        ),
    )
    provider_calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        pytest.fail("conflicting retry claim reached provider")

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", forbidden)
    with pytest.raises(PortfolioS1FeedbackRunError, match="multiple settled"):
        _execute_phase_v4(
            prepared,
            prepared.sources,
            {source.selection_entry_sha256: first},
            lambda *_a, **_k: None,
        )
    assert provider_calls == 0


def test_orphaned_reservation_never_reaches_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 1)
    _install_phase_fakes(monkeypatch)
    calls: list[int] = []
    orphan = FeedbackCallReservationV4.model_construct(
        selection_entry_sha256=prepared.sources[0].selection_entry_sha256,
        attempt_index=1,
        global_call_ordinal=1,
    )

    def crash(*_args, attempt_index, **_kwargs):
        calls.append(attempt_index)
        raise RuntimeError("post-reservation crash")

    monkeypatch.setattr("scripts.run_portfolio_s1_feedback._run_source_v4", crash)
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._attempt_state_v4",
        lambda _prepared: ((), (orphan,), {}),
    )
    with pytest.raises(PortfolioS1FeedbackRunError, match="orphan"):
        _execute_phase_v4(prepared, prepared.sources, {}, lambda *_a, **_k: None)
    assert calls == [1]


def test_fresh_v2_rejects_legacy_attempt_root_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepared(tmp_path, 1)
    legacy = prepared.output_dir / "provider-attempts"
    legacy.mkdir()
    (legacy / "historical-v1.json").write_text("{}", encoding="utf-8")
    provider_calls = 0

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        pytest.fail("legacy root reached provider")

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._require_qwen_execute_environment",
        lambda: None,
    )
    with pytest.raises(PortfolioS1FeedbackRunError, match="legacy provider"):
        _execute_run_locked(prepared, feedback_runner=forbidden_provider)
    assert provider_calls == 0
