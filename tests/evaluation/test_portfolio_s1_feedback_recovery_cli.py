from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_portfolio_s1_feedback_recovery_v1 as recovery_cli
from skillchain.tools.serialization import canonical_json_bytes


class _FakeCanonical:
    def __init__(self, payload: dict[str, object]) -> None:
        self._content = canonical_json_bytes(payload)
        for key, value in payload.items():
            setattr(self, key, value)

    def canonical_bytes(self) -> bytes:
        return self._content


def _arguments(tmp_path: Path) -> Namespace:
    parent = tmp_path / "parent"
    parent.mkdir()
    execution = tmp_path / "execution"
    execution.mkdir()
    artifact_repository = tmp_path / "artifacts"
    artifact_repository.mkdir()
    return Namespace(
        parent_root=parent,
        execution_root=execution,
        expected_execution_control_sha256="a" * 64,
        artifact_repository_root=artifact_repository,
        output_dir=tmp_path / "recovery",
        authorization_id="recovery-auth-v1",
        reviewer_id="owner",
        reviewed_at="2026-08-11T10:00:00+08:00",
        run_id="recovery-run-v1",
    )


def test_parse_reviewed_at_requires_canonical_timezone() -> None:
    parsed = recovery_cli._parse_reviewed_at_v1("2026-08-11T02:00:00Z")
    assert parsed == datetime(2026, 8, 11, 2, 0, tzinfo=timezone.utc)
    for invalid in (
        "2026-08-11T02:00:00",
        "2026-08-11T02:00:00.001Z",
        "2026-08-11T02:00:00+00:00",
    ):
        with pytest.raises(
            recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
            match="canonical timezone-aware",
        ):
            recovery_cli._parse_reviewed_at_v1(invalid)


def test_cli_direct_help_bootstraps_repository_imports(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(recovery_cli.__file__)), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Forward-only recovery runner" in completed.stdout


def test_prepare_recovery_builds_exact_zero_call_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    entries = tuple(
        SimpleNamespace(entry_sha256=f"{index:064x}") for index in range(1, 241)
    )
    parent_receipt = _FakeCanonical(
        {"kind": "parent-evidence", "evidence_sha256": "b" * 64}
    )
    parent = SimpleNamespace(
        root=arguments.parent_root.resolve(),
        selection=SimpleNamespace(entries=entries),
        authorization=object(),
        control=object(),
        run=SimpleNamespace(run_sha256="c" * 64),
        receipt=parent_receipt,
    )
    corpus = SimpleNamespace(
        core_inputs=SimpleNamespace(
            runtime_for=lambda processor: (
                processor,
                "verified-parent-remote-runtime",
            )
        )
    )
    sources = tuple(
        SimpleNamespace(selection_entry_sha256=item.entry_sha256)
        for item in entries
    )
    authorization = _FakeCanonical(
        {"kind": "recovery-authorization", "authorization_sha256": "d" * 64}
    )
    control = _FakeCanonical(
        {"kind": "recovery-control", "control_sha256": "e" * 64}
    )
    launch = _FakeCanonical(
        {"kind": "recovery-launch", "launch_lock_sha256": "f" * 64}
    )

    monkeypatch.setattr(
        recovery_cli,
        "load_verified_parent_feedback_evidence_v1",
        lambda root: parent,
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_verified_static_gcs_corpus",
        lambda *args, **kwargs: corpus,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_verified_static_feedback_sources_v2",
        lambda *args: sources,
    )
    remote_calls: list[dict[str, object]] = []

    def _load_remote(*args, **kwargs):
        remote_calls.append(kwargs)
        assert args[-1] == (
            "dashscope-qwen-assistant",
            "verified-parent-remote-runtime",
        )
        assert kwargs["verified_sources"] is sources
        return "verified-recovery-remote-runtime"

    monkeypatch.setattr(
        recovery_cli,
        "load_selected_qwen38_feedback_remote_runtime_v7",
        _load_remote,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_recovery_authorization_v1",
        lambda parent_arg, **kwargs: authorization,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_recovery_control_v1",
        lambda parent_arg, authorization_arg: control,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_qwen38_feedback_recovery_launch_lock_v1",
        lambda *args, **kwargs: launch,
    )

    prepared = recovery_cli.prepare_recovery_run_v1(arguments)
    assert prepared.parent is parent
    assert prepared.sources is sources
    assert prepared.remote_runtime == "verified-recovery-remote-runtime"
    assert len(remote_calls) == 1
    assert (arguments.output_dir / recovery_cli.RECOVERY_PARENT_EVIDENCE_FILE).read_bytes() == (
        parent_receipt.canonical_bytes()
    )
    assert (arguments.output_dir / recovery_cli.RECOVERY_AUTHORIZATION_FILE).read_bytes() == (
        authorization.canonical_bytes()
    )
    assert (arguments.output_dir / recovery_cli.RECOVERY_CONTROL_FILE).read_bytes() == (
        control.canonical_bytes()
    )
    assert (arguments.output_dir / recovery_cli.RECOVERY_LAUNCH_FILE).read_bytes() == (
        launch.canonical_bytes()
    )
    for directory in (
        recovery_cli.RECOVERY_CLAIM_DIR,
        recovery_cli.RECOVERY_ATTEMPT_DIR,
        recovery_cli.RECOVERY_BOUND_DIR,
    ):
        assert (arguments.output_dir / directory).is_dir()

    overlapping = Namespace(**vars(arguments))
    overlapping.output_dir = arguments.parent_root / "forbidden-child"
    with pytest.raises(
        recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
        match="must not alias or overlap",
    ):
        recovery_cli.prepare_recovery_run_v1(overlapping)

    # Exact resume is permitted; a conflicting byte is not.
    recovery_cli.prepare_recovery_run_v1(arguments)
    (arguments.output_dir / recovery_cli.RECOVERY_CONTROL_FILE).write_bytes(b"{}\n")
    with pytest.raises(
        recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
        match="exact-resume bytes differ",
    ):
        recovery_cli.prepare_recovery_run_v1(arguments)


def test_cli_rejects_phase_stop_outside_execute(capsys: pytest.CaptureFixture[str]) -> None:
    result = recovery_cli.main(
        [
            "--mode",
            "dry-run",
            "--parent-root",
            "parent",
            "--execution-root",
            "execution",
            "--expected-execution-control-sha256",
            "a" * 64,
            "--artifact-repository-root",
            "artifacts",
            "--output-dir",
            "recovery",
            "--authorization-id",
            "authorization",
            "--reviewer-id",
            "owner",
            "--reviewed-at",
            "2026-08-11T02:00:00Z",
            "--run-id",
            "run",
            "--stop-after-count",
            "73",
        ]
    )
    assert result == 2
    assert "only valid with --mode execute" in capsys.readouterr().err


def test_execute_dispatches_claim_reservation_single_call_and_phase_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    sources = tuple(SimpleNamespace(packet=object()) for _ in range(240))
    prepared = SimpleNamespace(
        output_dir=output,
        parent=SimpleNamespace(
            selection=SimpleNamespace(
                entries=tuple(
                    SimpleNamespace(query_id=f"q-{index}")
                    for index in range(1, 241)
                )
            )
        ),
        authorization=object(),
        control=object(),
        sources=sources,
    )
    empty = SimpleNamespace(
        claims=(),
        reservations=(),
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(),
    )
    claim = SimpleNamespace(claim_ordinal=1, claim_sha256="a" * 64)
    claimed = SimpleNamespace(
        claims=(claim,),
        reservations=(),
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(claim,),
    )
    artifact = SimpleNamespace(
        selection_ordinal=73,
        lifetime_attempt_index=3,
        status="parsed",
        feedback_result=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1)),
    )
    settled = SimpleNamespace(
        claims=(claim,),
        reservations=(object(),),
        artifacts=(artifact,),
        orphaned_reservations=(),
        pending_claims=(),
    )
    ledgers = iter((empty, claimed, settled))
    monkeypatch.setattr(recovery_cli, "_load_ledger", lambda prepared_arg: next(ledgers))
    monkeypatch.setattr(
        recovery_cli,
        "_exclusive_execute_writer_lock",
        lambda output_arg: nullcontext(),
    )
    steps = iter(
        (
            SimpleNamespace(
                kind="create_claim",
                claim_ordinal=1,
                selection_ordinal=73,
            ),
            SimpleNamespace(
                kind="reserve_claimed_call",
                claim_ordinal=1,
                selection_ordinal=73,
                lifetime_attempt_index=3,
                new_call_ordinal=1,
                wire_kind="recovery_v8_stage1",
            ),
            SimpleNamespace(kind="phase_complete", parsed_total=74),
        )
    )
    monkeypatch.setattr(
        recovery_cli,
        "next_feedback_recovery_step_v1",
        lambda *args, **kwargs: next(steps),
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_feedback_recovery_global_claim_v1",
        lambda *args, **kwargs: claim,
    )
    writes: list[str] = []
    monkeypatch.setattr(
        recovery_cli,
        "write_feedback_recovery_global_claim_v1",
        lambda path, value: writes.append("claim"),
    )
    reservation = SimpleNamespace(
        new_call_ordinal=1,
        selection_entry_sha256="b" * 64,
        new_attempt_index=1,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_feedback_recovery_call_reservation_v1",
        lambda *args, **kwargs: reservation,
    )
    monkeypatch.setattr(
        recovery_cli,
        "feedback_recovery_attempt_filename_v1",
        lambda value: "0001-b-attempt-1.json",
    )
    monkeypatch.setattr(
        recovery_cli,
        "write_feedback_recovery_call_reservation_v1",
        lambda path, value: writes.append("reservation"),
    )
    result = object()
    monkeypatch.setattr(
        recovery_cli,
        "redact_recovery_result_for_creator_privacy_v1",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_bound_feedback_recovery_artifact_v1",
        lambda *args, **kwargs: artifact,
    )
    monkeypatch.setattr(
        recovery_cli,
        "write_bound_feedback_recovery_artifact_v1",
        lambda path, value: writes.append("artifact"),
    )
    calls: list[tuple[object, str]] = []

    def _run(source, wire_kind):
        calls.append((source, wire_kind))
        return result

    outcome = recovery_cli.execute_recovery_run_v1(
        prepared,
        recovery_runner=_run,
        stop_after_count=73,
    )
    assert outcome.step.kind == "phase_complete"
    assert outcome.step.parsed_total == 74
    assert writes == ["claim", "reservation", "artifact"]
    assert calls == [(sources[72], "recovery_v8_stage1")]


def test_phase73_summary_reports_combined_parsed74() -> None:
    artifact = SimpleNamespace(
        selection_ordinal=73,
        status="parsed",
        feedback_result=SimpleNamespace(usage=SimpleNamespace()),
    )
    ledger = SimpleNamespace(
        reservations=(object(),),
        artifacts=(artifact,),
        orphaned_reservations=(),
        claims=(object(),),
    )
    prepared = SimpleNamespace(
        parent=SimpleNamespace(run=SimpleNamespace(run_sha256="a" * 64)),
        parent_evidence=SimpleNamespace(evidence_sha256="b" * 64),
        authorization=SimpleNamespace(authorization_sha256="c" * 64),
        control=SimpleNamespace(control_sha256="d" * 64),
        launch_lock=SimpleNamespace(launch_lock_sha256="e" * 64),
    )
    outcome = SimpleNamespace(
        step=SimpleNamespace(kind="phase_complete"),
        ledger=ledger,
        run=None,
        bundle=None,
    )
    summary = recovery_cli._summary_v1(
        "execute", prepared, outcome, stop_after_count=73
    )
    assert summary["combined_parsed_count"] == 74
    assert summary["phase_expected_parsed_totals"] == [74, 120, 240]
    assert summary["execution_concurrency"] == 1
    assert summary["not_parent_resume"] is True
    assert summary["new_provider_calls_reserved"] == 1
    assert summary["next_or_terminal_step"] == "phase_complete"


def test_run_written_bundle_missing_exactly_finalizes_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    run = _FakeCanonical({"terminal_reason": None})
    (output / recovery_cli.RECOVERY_RUN_FILE).write_bytes(run.canonical_bytes())
    ledger = SimpleNamespace(
        claims=(),
        reservations=(),
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(),
    )
    prepared = SimpleNamespace(
        output_dir=output,
        parent=object(),
        authorization=object(),
        control=object(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_feedback_recovery_ledger_v1",
        lambda *args, **kwargs: ledger,
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_portfolio_s1_feedback_recovery_run_v1",
        lambda path: run,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_recovery_run_v1",
        lambda *args, **kwargs: run,
    )
    monkeypatch.setattr(
        recovery_cli,
        "next_feedback_recovery_step_v1",
        lambda *args, **kwargs: SimpleNamespace(
            kind="finalize_complete", parsed_total=240
        ),
    )
    monkeypatch.setattr(
        recovery_cli,
        "_post_settlement_budget_terminal_reason",
        lambda value: None,
    )
    monkeypatch.setattr(
        recovery_cli,
        "_exclusive_execute_writer_lock",
        lambda output_arg: nullcontext(),
    )
    expected = SimpleNamespace(
        step=SimpleNamespace(kind="finalize_complete"),
        ledger=ledger,
        run=run,
        bundle=object(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "_finalize_completed_recovery",
        lambda *args: expected,
    )
    provider_calls: list[object] = []
    outcome = recovery_cli.execute_recovery_run_v1(
        prepared,
        recovery_runner=lambda *args: provider_calls.append(args),
    )
    assert outcome is expected
    assert provider_calls == []


def test_conflicting_run_fails_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    (output / recovery_cli.RECOVERY_RUN_FILE).write_bytes(b"conflict\n")
    ledger = SimpleNamespace(
        claims=(),
        reservations=(),
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(),
    )
    observed = _FakeCanonical({"terminal_reason": None})
    prepared = SimpleNamespace(
        output_dir=output,
        parent=object(),
        authorization=object(),
        control=object(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_feedback_recovery_ledger_v1",
        lambda *args, **kwargs: ledger,
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_portfolio_s1_feedback_recovery_run_v1",
        lambda path: observed,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_recovery_run_v1",
        lambda *args, **kwargs: observed,
    )
    monkeypatch.setattr(
        recovery_cli,
        "_exclusive_execute_writer_lock",
        lambda output_arg: nullcontext(),
    )
    provider_calls: list[object] = []
    with pytest.raises(
        recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
        match="differs from its exact ledger",
    ):
        recovery_cli.execute_recovery_run_v1(
            prepared,
            recovery_runner=lambda *args: provider_calls.append(args),
        )
    assert provider_calls == []


def test_bundle_without_run_fails_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    (output / recovery_cli.RECOVERY_BUNDLE_FILE).write_bytes(b"bundle\n")
    prepared = SimpleNamespace(
        output_dir=output,
        parent=object(),
        authorization=object(),
        control=object(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_feedback_recovery_ledger_v1",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "_exclusive_execute_writer_lock",
        lambda output_arg: nullcontext(),
    )
    provider_calls: list[object] = []
    with pytest.raises(
        recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
        match="exists without its run",
    ):
        recovery_cli.execute_recovery_run_v1(
            prepared,
            recovery_runner=lambda *args: provider_calls.append(args),
        )
    assert provider_calls == []


def test_conflicting_bundle_bytes_fail_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    run = _FakeCanonical({"terminal_reason": None})
    bundle = _FakeCanonical({"bundle_sha256": "b" * 64})
    (output / recovery_cli.RECOVERY_RUN_FILE).write_bytes(run.canonical_bytes())
    (output / recovery_cli.RECOVERY_BUNDLE_FILE).write_bytes(b"conflict\n")
    ledger = SimpleNamespace(
        claims=(),
        reservations=(),
        artifacts=(),
        orphaned_reservations=(),
        pending_claims=(),
    )
    prepared = SimpleNamespace(
        output_dir=output,
        parent=object(),
        authorization=object(),
        control=object(),
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_feedback_recovery_ledger_v1",
        lambda *args, **kwargs: ledger,
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_portfolio_s1_feedback_recovery_run_v1",
        lambda path: run,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_recovery_run_v1",
        lambda *args, **kwargs: run,
    )
    monkeypatch.setattr(
        recovery_cli,
        "load_portfolio_s1_feedback_bundle_v9",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        recovery_cli,
        "build_portfolio_s1_feedback_bundle_v9",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        recovery_cli,
        "_exclusive_execute_writer_lock",
        lambda output_arg: nullcontext(),
    )
    provider_calls: list[object] = []
    with pytest.raises(
        recovery_cli.PortfolioS1FeedbackRecoveryCLIError,
        match="BundleV9 differs from its exact ledger",
    ):
        recovery_cli.execute_recovery_run_v1(
            prepared,
            recovery_runner=lambda *args: provider_calls.append(args),
        )
    assert provider_calls == []
