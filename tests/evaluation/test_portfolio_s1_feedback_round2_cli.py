from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_portfolio_s1_feedback import (
    PreparedPortfolioS1FeedbackRun,
    PortfolioS1FeedbackRunError,
    _execute_phase,
    _execute_run_locked,
    _load_qwen_role_selection,
    _resume_all_before_provider,
)
from skillchain import config
from skillchain.evaluation.evaluator_outputs import parse_visual_feedback_output_v3
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
    _make_result,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    FeedbackCallReservationV2,
    FeedbackCallReservationV3,
    PortfolioS1FeedbackAuthorizationV5,
    PortfolioS1FeedbackControlV10,
    PortfolioS1FeedbackRunV3,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    load_portfolio_s1_feedback_run_v3,
    write_feedback_call_reservation_v2,
)
from skillchain.evaluation.packets import (
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.llm import LLMUsage
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


ROLE_V10_FILE_SHA256 = (
    "39e8d099f0b303725ee0be59f758560627ecbc3ce156524da4f2a2d3d20f3501"
)
ROLE_V10_SHA256 = "be5f7f7ee1f30a81d7d2a8423384adb41d74e4673000901e883af704e9eead16"


def _round2_result(
    query_id: str,
    *,
    summary: str,
    prepared: PreparedPortfolioS1FeedbackRun,
):
    raw = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": summary,
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [
                    "[policy_compatible] Preserve the exact fallback marker."
                ],
            }
        )
        .decode("utf-8")
        .strip()
    )
    return _make_result(
        schema_version=5,
        cache_namespace="feedback-evaluator-v11",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
        requested_response_format="json_schema",
        requested_json_schema_sha256=VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
        requested_thinking=True,
        requested_thinking_budget=2048,
        requested_timeout_seconds=600,
        requested_temperature=None,
        requested_top_p=None,
        query_id=query_id,
        packet_sha256="7" * 64,
        prompt_sha256="8" * 64,
        image_sha256="9" * 64,
        wire_sha256="a" * 64,
        asset_catalog_sha256=prepared.remote_runtime.catalog.catalog_sha256,
        remote_authorization_id=prepared.authorization.authorization_id,
        remote_authorization_file_sha256=(prepared.control.authorization_file_sha256),
        remote_receipt_file_sha256=prepared.remote_runtime.receipt_file_sha256,
        remote_receipt_sha256=prepared.remote_runtime.receipt.receipt_sha256,
        provider="qwen",
        model="qwen3.8-max",
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        max_tokens=None,
        max_completion_tokens=4096,
        status="parsed",
        request_id=f"request-{query_id}",
        raw_response_text=raw,
        raw_response_sha256=sha256_bytes(raw.encode("utf-8")),
        raw_response_bytes=len(raw.encode("utf-8")),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parse_visual_feedback_output_v3(raw),
        usage=LLMUsage(input_tokens=20, output_tokens=8),
        finish_reason="stop",
        latency_ms=3,
        reasoning_present=False,
        reasoning_tokens=None,
        reasoning_bytes=0,
        reasoning_sha256=None,
    )


def _round2_prepared(tmp_path: Path, count: int) -> PreparedPortfolioS1FeedbackRun:
    entries = tuple(
        SimpleNamespace(
            entry_sha256=f"{ordinal:064x}",
            selection_ordinal=ordinal,
            query_id=f"private-query-{ordinal}",
        )
        for ordinal in range(1, count + 1)
    )
    selection = PortfolioS1FeedbackSelectionV2.model_construct(
        selection_sha256="b" * 64,
        entries=entries,
        phase_counts=(12, 60, 120, 240),
    )
    control = PortfolioS1FeedbackControlV10.model_construct(
        control_sha256="c" * 64,
        selection_sha256=selection.selection_sha256,
        authorization_file_sha256="d" * 64,
        provider_call_ceiling=240,
        max_completion_tokens=4096,
        requested_timeout_seconds=600,
    )
    authorization = PortfolioS1FeedbackAuthorizationV5.model_construct(
        authorization_id="round2-test-authorization"
    )
    sources = tuple(
        VerifiedStaticFeedbackSourceV2(
            corpus_sha256="e" * 64,
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
    output_dir = tmp_path / "round2"
    output_dir.mkdir()
    return PreparedPortfolioS1FeedbackRun(
        output_dir=output_dir,
        corpus=SimpleNamespace(),
        selection=selection,
        authorization=authorization,
        control=control,
        sources=sources,
        remote_runtime=SimpleNamespace(
            catalog=SimpleNamespace(catalog_sha256="f" * 64),
            receipt_file_sha256="1" * 64,
            receipt=SimpleNamespace(receipt_sha256="2" * 64),
        ),
    )


class _FakeBoundArtifact:
    def __init__(self, entry_sha256: str, result) -> None:
        self.selection_entry_sha256 = entry_sha256
        self.status = result.status
        self.feedback_result = result

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "selection_entry_sha256": self.selection_entry_sha256,
                "status": self.status,
                "result_sha256": self.feedback_result.result_sha256,
            }
        )


def _install_round2_execution_fakes(monkeypatch, terminal_runs: list[object]) -> None:
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.make_active_portfolio_evaluator_isolation_lock",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._estimated_input_tokens", lambda *_: 100
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._budget_snapshot",
        lambda _prepared: SimpleNamespace(
            provider_calls_reserved=0,
            accountable_cost_cny=Decimal("0"),
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.require_qwen38_feedback_pre_call_budget",
        lambda **_kwargs: "0.387816000000",
    )

    def reservation(_selection, _control, entry, **_kwargs):
        return FeedbackCallReservationV3.model_construct(
            selection_entry_sha256=entry.entry_sha256,
            selection_ordinal=entry.selection_ordinal,
            query_id=entry.query_id,
            reservation_sha256=f"{entry.selection_ordinal + 1000:064x}",
        )

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_feedback_call_reservation_v3",
        reservation,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_bound_feedback_artifact_v3",
        lambda _selection, _control, entry, _packet, result, **_kwargs: (
            _FakeBoundArtifact(entry.entry_sha256, result)
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._publish_terminal_run",
        lambda _prepared, artifacts, **kwargs: terminal_runs.append(
            (tuple(artifacts.values()), kwargs)
        ),
    )


def test_round2_role_selection_is_exact_qwen38_selected240() -> None:
    assert _load_qwen_role_selection(
        Path("specs/authoring/model-role-selection-v10.json"),
        expected_file_sha256=ROLE_V10_FILE_SHA256,
        expected_selection_sha256=ROLE_V10_SHA256,
        round2=True,
    ) == (ROLE_V10_FILE_SHA256, ROLE_V10_SHA256)


def test_round2_execute_stops_at_each_cumulative_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    entries = tuple(
        SimpleNamespace(entry_sha256=f"{ordinal:064x}") for ordinal in range(1, 241)
    )
    selection = PortfolioS1FeedbackSelectionV2.model_construct(
        entries=entries,
        phase_counts=(12, 60, 120, 240),
    )
    control = PortfolioS1FeedbackControlV10.model_construct()
    authorization = PortfolioS1FeedbackAuthorizationV5.model_construct()
    sources = tuple(
        SimpleNamespace(selection_entry_sha256=item.entry_sha256) for item in entries
    )
    prepared = PreparedPortfolioS1FeedbackRun(
        output_dir=tmp_path,
        corpus=SimpleNamespace(),
        selection=selection,
        authorization=authorization,
        control=control,
        sources=sources,
        remote_runtime=SimpleNamespace(),
    )
    observed_counts: list[int] = []

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._require_qwen_execute_environment",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._resume_all_before_provider",
        lambda _prepared: {},
    )

    def fake_execute_phase(_prepared, phase_sources, artifacts, _runner) -> None:
        observed_counts.append(len(phase_sources))
        for source in phase_sources:
            artifacts[source.selection_entry_sha256] = SimpleNamespace(
                selection_entry_sha256=source.selection_entry_sha256,
                status="parsed",
            )

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._execute_phase", fake_execute_phase
    )

    assert _execute_run_locked(prepared, stop_after_count=120) is None
    assert observed_counts == [12, 60, 120]


def test_round2_first_privacy_failure_is_one_terminal_call_without_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared = _round2_prepared(tmp_path, 1)
    terminal_runs: list[object] = []
    _install_round2_execution_fakes(monkeypatch, terminal_runs)
    calls: list[str] = []

    def privacy_feedback(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        return _round2_result(
            packet.query_id,
            summary=f"Leaked private identity {packet.query_id}",
            prepared=prepared,
        )

    artifacts: dict[str, object] = {}
    with pytest.raises(PortfolioS1FeedbackRunError, match="nonparsed"):
        _execute_phase(
            prepared,
            (prepared.sources[0],),
            artifacts,
            privacy_feedback,
        )

    assert calls == ["private-query-1"]
    assert len(terminal_runs) == 1
    terminal_artifact = next(iter(artifacts.values()))
    assert terminal_artifact.status == "parse_error"
    assert terminal_artifact.feedback_result.error_code == (
        "creator_projection_privacy"
    )
    assert terminal_artifact.feedback_result.raw_response_text is None
    _execute_phase(
        prepared,
        (prepared.sources[0],),
        artifacts,
        lambda *_args, **_kwargs: pytest.fail("terminal entry was retried"),
    )
    assert calls == ["private-query-1"]


def test_round2_privacy_failure_stops_after_at_most_one_concurrent_wave(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared = _round2_prepared(tmp_path, 3)
    terminal_runs: list[object] = []
    _install_round2_execution_fakes(monkeypatch, terminal_runs)
    calls: list[str] = []

    def mixed_feedback(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        summary = (
            f"Leaked private identity {packet.query_id}"
            if packet.query_id == "private-query-1"
            else "Safe policy-compatible diagnostic."
        )
        return _round2_result(packet.query_id, summary=summary, prepared=prepared)

    with pytest.raises(PortfolioS1FeedbackRunError, match="nonparsed"):
        _execute_phase(
            prepared,
            prepared.sources,
            {},
            mixed_feedback,
        )

    assert set(calls) == {"private-query-1", "private-query-2"}
    assert "private-query-3" not in calls
    assert len(terminal_runs) == 1


def test_round2_post_provider_identity_failure_publishes_orphan_in_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared = _round2_prepared(tmp_path, 3)
    terminal_runs: list[object] = []
    _install_round2_execution_fakes(monkeypatch, terminal_runs)
    calls: list[str] = []
    reservations: dict[str, FeedbackCallReservationV3] = {}

    def reservation(_selection, _control, entry, **_kwargs):
        value = FeedbackCallReservationV3.model_construct(
            selection_entry_sha256=entry.entry_sha256,
            selection_ordinal=entry.selection_ordinal,
            query_id=entry.query_id,
            reservation_sha256=f"{entry.selection_ordinal + 2000:064x}",
        )
        reservations[entry.entry_sha256] = value
        return value

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_feedback_call_reservation_v3",
        reservation,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._load_reservation",
        lambda _prepared, source: reservations.get(source.selection_entry_sha256),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._resume_artifact",
        lambda *_args, **_kwargs: None,
    )

    def identity_drift(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        result = _round2_result(
            packet.query_id,
            summary="Safe policy-compatible diagnostic.",
            prepared=prepared,
        )
        return result.model_copy(update={"asset_catalog_sha256": "0" * 64})

    with pytest.raises(PortfolioS1FeedbackRunError, match="stopped_orphan"):
        _execute_phase(prepared, prepared.sources, {}, identity_drift)

    assert set(calls) == {"private-query-1", "private-query-2"}
    assert "private-query-3" not in calls
    assert len(terminal_runs) == 1
    _artifacts, kwargs = terminal_runs[0]
    assert set(kwargs["orphaned_entry_sha256s"]) == {
        prepared.selection.entries[0].entry_sha256,
        prepared.selection.entries[1].entry_sha256,
    }


def test_round2_orphan_restart_publishes_rebuildable_run_without_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared = _round2_prepared(tmp_path, 2)
    orphan_entry = prepared.selection.entries[1]
    reservation = FeedbackCallReservationV3.model_construct(
        selection_entry_sha256=orphan_entry.entry_sha256,
        selection_ordinal=orphan_entry.selection_ordinal,
        query_id=orphan_entry.query_id,
        reservation_sha256="3" * 64,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._verify_attempt_file_set", lambda _: None
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._load_reservation",
        lambda _prepared, source: (
            reservation
            if source.selection_entry_sha256 == orphan_entry.entry_sha256
            else None
        ),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._resume_artifact",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PortfolioS1FeedbackRunError, match="stopped_orphan"):
        _resume_all_before_provider(prepared)

    run_path = prepared.output_dir / "run.json"
    run = load_portfolio_s1_feedback_run_v3(
        run_path,
        expected_file_sha256=sha256_bytes(run_path.read_bytes()),
    )
    assert isinstance(run, PortfolioS1FeedbackRunV3)
    assert run.status == "stopped_orphan"
    assert run.provider_calls_reserved == 1
    assert run.artifacts[0].selection_ordinal == 2
    assert run.artifacts[0].selection_entry_sha256 == orphan_entry.entry_sha256
    assert run.artifacts[0].status == "orphan"

    provider_calls = 0

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        pytest.fail("orphan restart reached provider")

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._require_qwen_execute_environment",
        lambda: None,
    )
    with pytest.raises(PortfolioS1FeedbackRunError, match="terminal"):
        _execute_run_locked(prepared, feedback_runner=forbidden_provider)
    assert provider_calls == 0


def test_qwen37_twelve_call_root_is_rejected_before_qwen38_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _round2_prepared(tmp_path, 12)
    attempts = prepared.output_dir / "provider-attempts"
    attempts.mkdir()
    for entry, source in zip(prepared.selection.entries, prepared.sources, strict=True):
        unsigned = {
            "schema_version": 2,
            "kind": "portfolio-s1-feedback-call-reservation",
            "policy_version": "portfolio-s1-feedback-call-reservation-v4",
            "selection_sha256": prepared.selection.selection_sha256,
            "control_sha256": prepared.control.control_sha256,
            "corpus_sha256": "e" * 64,
            "selection_entry_sha256": entry.entry_sha256,
            "selection_ordinal": entry.selection_ordinal,
            "query_id": entry.query_id,
            "packet_sha256": "7" * 64,
            "checkpoint_file_sha256": "8" * 64,
            "checkpoint_row_sha256": "9" * 64,
            "sidecar_sha256": "a" * 64,
            "provider": "qwen",
            "model": "qwen3.7-plus-2026-05-26",
            "cache_namespace": "feedback-evaluator-v10",
            "max_tokens": None,
            "max_completion_tokens": 4096,
            "max_attempts": 1,
            "reservation_cny": "0.072848000000",
        }
        reservation = FeedbackCallReservationV2.model_validate(
            {
                **unsigned,
                "reservation_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            },
            strict=True,
        )
        write_feedback_call_reservation_v2(
            attempts
            / (f"{source.row.query_ordinal:04d}-{entry.entry_sha256[:16]}.json"),
            reservation,
        )

    provider_calls = 0

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        pytest.fail("legacy Qwen3.7 root reached the Qwen3.8 provider")

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._require_qwen_execute_environment",
        lambda: None,
    )
    with pytest.raises(ValueError, match="schema_version"):
        _execute_run_locked(prepared, feedback_runner=forbidden_provider)
    assert provider_calls == 0
