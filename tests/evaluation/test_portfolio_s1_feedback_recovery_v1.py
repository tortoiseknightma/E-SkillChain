from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil

import pytest

from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.portfolio_s1_feedback import PortfolioS1FeedbackError
from skillchain.evaluation import portfolio_s1_feedback_recovery_v1 as recovery
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    BoundFeedbackRecoveryArtifactV1,
    PortfolioS1FeedbackRecoveryRunAttemptV1,
    PortfolioS1FeedbackRecoveryRunV1,
    PortfolioS1FeedbackRecoveryLedgerV1,
    RecoveryFeedbackEvaluationResultV1,
    build_feedback_recovery_global_claim_v1,
    build_portfolio_s1_feedback_recovery_authorization_v1,
    build_portfolio_s1_feedback_recovery_control_v1,
    build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1,
    load_verified_parent_feedback_evidence_v1,
    recovery_trigger_kind,
    run_visual_feedback_recovery_v1,
    validate_feedback_recovery_claim_set_v1,
)
from skillchain.llm import LLMUsage


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PARENT_ROOT = (
    _REPOSITORY_ROOT
    / "runs"
    / "portfolio"
    / "core-s1"
    / "s1-feedback-round2-qwen38-v3"
)


@pytest.fixture(scope="module")
def parent():
    return load_verified_parent_feedback_evidence_v1(_PARENT_ROOT)


@pytest.fixture(scope="module")
def authority(parent):
    authorization = build_portfolio_s1_feedback_recovery_authorization_v1(
        parent,
        authorization_id="portfolio-s1-feedback-recovery-test-v1",
        reviewer_id="codex-test",
        reviewed_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    control = build_portfolio_s1_feedback_recovery_control_v1(
        parent, authorization
    )
    launch = build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1(
        parent,
        authorization,
        control,
        run_id="portfolio-s1-feedback-recovery-test-run-v1",
    )
    return authorization, control, launch


def _trigger_result(
    *,
    raw: str,
    finish_reason: str = "stop",
    reasoning_present: bool = False,
) -> RecoveryFeedbackEvaluationResultV1:
    return RecoveryFeedbackEvaluationResultV1.model_construct(
        status="parse_error",
        error_code="invalid_feedback_json",
        request_id="request-test",
        usage=LLMUsage(input_tokens=1, output_tokens=1),
        response_redaction_reason=None,
        refusal_present=False,
        tool_calls=(),
        raw_response_text=raw,
        finish_reason=finish_reason,
        reasoning_present=reasoning_present,
        result_sha256="a" * 64,
    )


def _fake_artifact(
    parent,
    *,
    selection_ordinal: int,
    new_call_ordinal: int,
    lifetime_attempt_index: int,
    result: RecoveryFeedbackEvaluationResultV1,
    status: str = "parse_error",
) -> BoundFeedbackRecoveryArtifactV1:
    entry = parent.selection.entries[selection_ordinal - 1]
    return BoundFeedbackRecoveryArtifactV1.model_construct(
        selection_ordinal=selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        new_call_ordinal=new_call_ordinal,
        lifetime_attempt_index=lifetime_attempt_index,
        status=status,
        feedback_result=result,
        artifact_sha256=f"{new_call_ordinal:064x}",
    )


def _run_attempt(
    *,
    ordinal: int,
    call: int,
    status: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> PortfolioS1FeedbackRecoveryRunAttemptV1:
    usage = (
        None
        if input_tokens is None or output_tokens is None
        else LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    )
    has_response = status not in {"provider_error", "timeout", "orphan"}
    return PortfolioS1FeedbackRecoveryRunAttemptV1(
        selection_ordinal=ordinal,
        selection_entry_sha256=f"{ordinal:064x}",
        new_call_ordinal=call,
        new_attempt_index=1,
        lifetime_attempt_index=3 if ordinal == 73 else 1,
        wire_kind="recovery_v8_stage1" if ordinal == 73 else "primary_v7",
        reservation_sha256=f"{call + 100:064x}",
        artifact_sha256=None if status == "orphan" else f"{call + 200:064x}",
        recovery_claim_ordinal=1 if ordinal == 73 else None,
        status=status,
        error_code=None if status in {"parsed", "orphan"} else status,
        request_id="request" if has_response else None,
        finish_reason="stop" if has_response else None,
        wire_sha256=None if status == "orphan" else f"{call + 300:064x}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=0 if has_response else None,
        reasoning_bytes=0 if has_response else None,
        refusal_present=False if has_response else None,
        refusal_bytes=0 if has_response else None,
        actual_cost_cny=(
            None if usage is None else recovery._recovery_actual_cost(usage)
        ),
    )


def _validated_run(
    rows: tuple[PortfolioS1FeedbackRecoveryRunAttemptV1, ...],
    *,
    status: str,
    terminal_reason: str | None,
) -> PortfolioS1FeedbackRecoveryRunV1:
    final = {item.selection_entry_sha256: item for item in rows}
    parsed = sum(item.status == "parsed" for item in final.values())
    new_actual = sum(
        (Decimal(item.actual_cost_cny) for item in rows if item.actual_cost_cny),
        Decimal("0"),
    )
    cumulative_actual = Decimal(recovery.PARENT_ACTUAL_COST_CNY) + new_actual
    unknown = sum(item.input_tokens is None for item in rows)
    accountable = cumulative_actual + Decimal(
        recovery.RECOVERY_PER_CALL_RESERVATION_CNY
    ) * unknown
    def money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.000000000001")), "f")
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-recovery-run",
        "policy_version": recovery.RECOVERY_RUN_POLICY_VERSION_V1,
        "selection_sha256": recovery.PARENT_SELECTION_SHA256,
        "authorization_sha256": "a" * 64,
        "control_sha256": "b" * 64,
        "parent_run_file_sha256": recovery.PARENT_RUN_FILE_SHA256,
        "parent_run_sha256": recovery.PARENT_RUN_SHA256,
        "parent_evidence_sha256": "c" * 64,
        "expected_count": 240,
        "imported_count": 73,
        "unresolved_count": 167,
        "phase_max_selection_ordinals": (
            recovery.RECOVERY_PHASE_MAX_SELECTION_ORDINALS
        ),
        "phase_expected_parsed_totals": (
            recovery.RECOVERY_PHASE_EXPECTED_PARSED_TOTALS
        ),
        "attempted_total": 73 + len(final),
        "parsed_total": 73 + parsed,
        "new_attempted_count": len(final),
        "new_parsed_count": parsed,
        "new_error_count": len(final) - parsed,
        "new_provider_call_count": len(rows),
        "cumulative_provider_call_count": 76 + len(rows),
        "recovery_claim_count": 1,
        "recovery_claim_sha256s": ("d" * 64,),
        "orphan_count": sum(item.status == "orphan" for item in rows),
        "terminal_phase_max_selection_ordinal": 120
        if max(item.selection_ordinal for item in rows) > 73
        else 73,
        "status": status,
        "terminal_reason": terminal_reason,
        "new_usage_known_count": len(rows) - unknown,
        "new_usage_unknown_count": unknown,
        "parent_input_tokens": 10,
        "parent_output_tokens": 20,
        "new_input_tokens": sum(item.input_tokens or 0 for item in rows),
        "new_output_tokens": sum(item.output_tokens or 0 for item in rows),
        "cumulative_input_tokens": 10 + sum(item.input_tokens or 0 for item in rows),
        "cumulative_output_tokens": 20 + sum(item.output_tokens or 0 for item in rows),
        "parent_actual_cost_cny": recovery.PARENT_ACTUAL_COST_CNY,
        "new_actual_cost_cny": money(new_actual),
        "cumulative_actual_cost_cny": money(cumulative_actual),
        "cumulative_accountable_cost_cny": money(accountable),
        "artifacts": rows,
        "artifact_set_sha256": recovery._hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRecoveryRunV1.model_validate(
        {**unsigned, "run_sha256": recovery._hash_payload(unsigned)}, strict=True
    )


def test_parent_loader_imports_exact73_and_separates_phase_semantics(parent) -> None:
    assert parent.run.run_sha256 == recovery.PARENT_RUN_SHA256
    assert len(parent.imported_artifacts) == 73
    assert parent.receipt.imported_selection_ordinals == tuple((*range(1, 73), 74))
    assert recovery.RECOVERY_PHASE_MAX_SELECTION_ORDINALS == (73, 120, 240)
    assert recovery.RECOVERY_PHASE_EXPECTED_PARSED_TOTALS == (74, 120, 240)


def test_parent_loader_rejects_filename_substitution(tmp_path: Path) -> None:
    copied = tmp_path / "parent"
    shutil.copytree(_PARENT_ROOT, copied)
    member = next((copied / "provider-attempts-v3").glob("*.json"))
    member.rename(member.with_name("renamed.json"))
    with pytest.raises(PortfolioS1FeedbackError, match="filenames or bytes drifted"):
        load_verified_parent_feedback_evidence_v1(copied)


def test_recovery_authority_is_serial_and_forward_only(authority) -> None:
    authorization, control, launch = authority
    for item in (authorization, control, launch):
        assert item.execution_concurrency == 1
        assert item.max_active_calls == 1
        assert item.not_parent_resume is True
    assert authorization.phase_expected_parsed_totals == (74, 120, 240)


def test_trigger_policy_distinguishes_transport_from_policy_label_failures() -> None:
    valid_but_unlabeled = json.dumps(
        {
            "schema_version": 1,
            "summary": "valid",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": ["missing disposition"],
        },
        separators=(",", ":"),
    )
    assert recovery_trigger_kind(
        _trigger_result(raw=valid_but_unlabeled)
    ) is None
    assert recovery_trigger_kind(
        _trigger_result(raw=valid_but_unlabeled, finish_reason="length")
    ) == "length_parser_failure"
    assert recovery_trigger_kind(
        _trigger_result(raw="", reasoning_present=True)
    ) == "reasoning_empty"
    assert recovery_trigger_kind(
        _trigger_result(raw="", reasoning_present=False)
    ) is None


def test_invalid_wire_and_wrong_isolation_fail_before_image_or_provider(
    parent, monkeypatch
) -> None:
    class WireKindSubclass(str):
        pass

    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("image/provider path must remain untouched")

    monkeypatch.setattr(recovery, "load_verified_evaluator_image", forbidden)
    monkeypatch.setattr(recovery.llm, "chat", forbidden)
    packet = parent.terminal_ordinal73_artifacts[-1].feedback_packet
    isolation = make_active_portfolio_evaluator_isolation_lock()
    with pytest.raises(PortfolioS1FeedbackError, match="wire kind"):
        run_visual_feedback_recovery_v1(
            packet,
            isolation,
            remote_runtime=object(),  # type: ignore[arg-type]
            wire_kind="forged",  # type: ignore[arg-type]
        )
    with pytest.raises(PortfolioS1FeedbackError, match="wire kind"):
        run_visual_feedback_recovery_v1(
            packet,
            isolation,
            remote_runtime=object(),  # type: ignore[arg-type]
            wire_kind=WireKindSubclass("primary_v7"),  # type: ignore[arg-type]
        )
    wrong = isolation.model_copy(update={"shared_model_runtime": True})
    with pytest.raises(PortfolioS1FeedbackError, match="isolated role"):
        run_visual_feedback_recovery_v1(
            packet,
            wrong,
            remote_runtime=object(),  # type: ignore[arg-type]
            wire_kind="primary_v7",
        )
    assert touched is False


def test_claim_builder_and_validator_require_earliest_unclaimed_settlement(
    parent, authority
) -> None:
    _authorization, control, _launch = authority
    claim1 = build_feedback_recovery_global_claim_v1(parent, control)
    parsed73 = _fake_artifact(
        parent,
        selection_ordinal=73,
        new_call_ordinal=1,
        lifetime_attempt_index=3,
        result=RecoveryFeedbackEvaluationResultV1.model_construct(status="parsed"),
        status="parsed",
    )
    first = _fake_artifact(
        parent,
        selection_ordinal=75,
        new_call_ordinal=2,
        lifetime_attempt_index=1,
        result=_trigger_result(raw="{"),
    )
    later = _fake_artifact(
        parent,
        selection_ordinal=76,
        new_call_ordinal=3,
        lifetime_attempt_index=1,
        result=_trigger_result(raw="{"),
    )
    artifacts = (parsed73, first, later)
    with pytest.raises(PortfolioS1FeedbackError, match="earliest eligible"):
        build_feedback_recovery_global_claim_v1(
            parent,
            control,
            existing_claims=(claim1,),
            existing_artifacts=artifacts,
            trigger_artifact=later,
        )
    claim2 = build_feedback_recovery_global_claim_v1(
        parent,
        control,
        existing_claims=(claim1,),
        existing_artifacts=artifacts,
        trigger_artifact=first,
    )
    validate_feedback_recovery_claim_set_v1(
        parent, control, (claim1, claim2), artifacts
    )
    unsigned = claim2.model_dump(mode="json", exclude={"claim_sha256"})
    unsigned["trigger_finish_reason"] = "length"
    forged_finish = type(claim2).model_validate(
        {
            **unsigned,
            "claim_sha256": recovery._hash_payload(unsigned),
        },
        strict=True,
    )
    with pytest.raises(PortfolioS1FeedbackError, match="trigger ancestry"):
        validate_feedback_recovery_claim_set_v1(
            parent, control, (claim1, forged_finish), artifacts
        )
    forged = claim2.model_copy(
        update={"trigger_artifact_sha256": later.artifact_sha256}
    )
    with pytest.raises(PortfolioS1FeedbackError, match="claim chain drifted"):
        validate_feedback_recovery_claim_set_v1(
            parent, control, (claim1, forged), artifacts
        )


def test_ledger_rejects_rehashed_retry_ancestry_not_matching_claim(
    parent, authority
) -> None:
    authorization, control, _launch = authority
    claim1 = build_feedback_recovery_global_claim_v1(parent, control)
    entry = parent.selection.entries[72]
    forged_previous = "f" * 64
    reservation = recovery.FeedbackRecoveryCallReservationV1.model_construct(
        selection_sha256=parent.selection.selection_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=73,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        new_call_ordinal=1,
        new_attempt_index=1,
        lifetime_attempt_index=3,
        wire_kind="recovery_v8_stage1",
        previous_artifact_sha256=forged_previous,
        recovery_claim_ordinal=1,
        recovery_claim_sha256=claim1.claim_sha256,
        retry_trigger_kind=claim1.trigger_kind,
        reservation_sha256="e" * 64,
    )
    artifact = BoundFeedbackRecoveryArtifactV1.model_construct(
        selection_sha256=parent.selection.selection_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=73,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        new_call_ordinal=1,
        new_attempt_index=1,
        lifetime_attempt_index=3,
        wire_kind="recovery_v8_stage1",
        reservation_sha256=reservation.reservation_sha256,
        previous_artifact_sha256=forged_previous,
        recovery_claim_ordinal=1,
        recovery_claim_sha256=claim1.claim_sha256,
        status="parsed",
        artifact_sha256="d" * 64,
    )
    ledger = PortfolioS1FeedbackRecoveryLedgerV1(
        claims=(claim1,),
        reservations=(reservation,),
        artifacts=(artifact,),
        orphaned_reservations=(),
        pending_claims=(),
    )
    with pytest.raises(PortfolioS1FeedbackError, match="reservation/claim ancestry"):
        recovery._validate_feedback_recovery_ledger_v1(
            parent, authorization, control, ledger
        )


def test_provider_error_attempt_is_publishable_and_keeps_unknown_reserve() -> None:
    row = _run_attempt(
        ordinal=73,
        call=1,
        status="provider_error",
        input_tokens=None,
        output_tokens=None,
    )
    run = _validated_run((row,), status="stopped_nonparsed", terminal_reason=None)
    assert run.new_usage_unknown_count == 1
    assert run.cumulative_accountable_cost_cny == "11.208828000000"


def test_next_call_budget_guard_applies_before_any_claim_or_reservation() -> None:
    ledger = PortfolioS1FeedbackRecoveryLedgerV1(
        claims=(),
        reservations=tuple(object() for _ in range(169)),  # type: ignore[arg-type]
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(),
    )
    assert (
        recovery._next_feedback_recovery_call_budget_error(ledger)
        == "provider_call_ceiling_exceeded"
    )
    assert Decimal(recovery.RECOVERY_CUMULATIVE_MAXIMUM_CNY) < Decimal(
        recovery.RECOVERY_TECHNICAL_CUMULATIVE_HARD_CAP_CNY
    )


def test_orphan_remains_terminal_when_usage_breach_also_exceeds_cny89() -> None:
    response = _run_attempt(
        ordinal=73,
        call=1,
        status="parse_error",
        input_tokens=7_000_000,
        output_tokens=0,
    )
    orphan = _run_attempt(
        ordinal=75,
        call=2,
        status="orphan",
        input_tokens=None,
        output_tokens=None,
    )
    run = _validated_run(
        (response, orphan),
        status="stopped_orphan",
        terminal_reason="usage_limit_exceeded",
    )
    assert Decimal(run.cumulative_accountable_cost_cny) > Decimal("89")
    assert run.status == "stopped_orphan"
    assert run.terminal_reason == "usage_limit_exceeded"


def test_run_rejects_self_consistent_cost_tamper() -> None:
    row = _run_attempt(
        ordinal=73,
        call=1,
        status="parse_error",
        input_tokens=1,
        output_tokens=1,
    )
    payload = row.model_dump(mode="python")
    payload["actual_cost_cny"] = "0.000000000000"
    with pytest.raises(ValueError, match="differs from token usage"):
        PortfolioS1FeedbackRecoveryRunAttemptV1.model_validate(
            payload, strict=True
        )

    valid = _validated_run(
        (
            _run_attempt(
                ordinal=73,
                call=1,
                status="provider_error",
                input_tokens=None,
                output_tokens=None,
            ),
        ),
        status="stopped_nonparsed",
        terminal_reason=None,
    )
    run_payload = valid.model_dump(mode="python")
    run_payload["cumulative_accountable_cost_cny"] = "10.747284000000"
    unsigned = {
        key: value for key, value in run_payload.items() if key != "run_sha256"
    }
    run_payload["run_sha256"] = recovery._hash_payload(unsigned)
    with pytest.raises(ValueError, match="counts drifted"):
        PortfolioS1FeedbackRecoveryRunV1.model_validate(
            run_payload, strict=True
        )
