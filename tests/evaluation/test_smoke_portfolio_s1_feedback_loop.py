from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.smoke_portfolio_s1_feedback_loop import (
    REPLAY_NOT_RUN,
    PortfolioS1ZeroProviderSmokeError,
    _FakeCodexProcess,
    _FakeFeedbackRunner,
    _NetworkDenied,
    run_smoke_twice,
)
from skillchain.evaluation.evaluator_outputs import parse_visual_feedback_output
from skillchain.evaluation.feedback_runtime import (
    load_feedback_evaluation_result,
    write_feedback_evaluation_result,
)
from skillchain.evaluation.packets import EvaluationImage


def test_deterministic_feedback_substitute_is_strict_and_zero_usage(
    tmp_path: Path,
) -> None:
    packet = SimpleNamespace(
        query_id="smoke-query",
        packet_sha256="1" * 64,
        image=EvaluationImage(mime_type="image/jpeg", sha256="2" * 64),
    )
    remote = SimpleNamespace(
        catalog=SimpleNamespace(catalog_sha256="3" * 64),
        authorization=SimpleNamespace(authorization_id="local-smoke-auth"),
        authorization_file_sha256="4" * 64,
        receipt_file_sha256="5" * 64,
        receipt=SimpleNamespace(receipt_sha256="6" * 64),
    )
    runner = _FakeFeedbackRunner()

    first = runner(
        packet,
        object(),
        remote_runtime=remote,
        max_tokens=None,
        max_completion_tokens=4096,
        timeout_seconds=600,
        record_usage=True,
    )
    second = runner(
        packet,
        object(),
        remote_runtime=remote,
        max_tokens=None,
        max_completion_tokens=4096,
        timeout_seconds=600,
        record_usage=True,
    )

    assert first == second
    assert first.schema_version == 3
    assert first.cache_namespace == "feedback-evaluator-v9"
    assert first.provider == "qwen"
    assert first.model == "qwen3.7-plus-2026-05-26"
    assert first.requested_response_format == "json_schema"
    assert first.requested_thinking is True
    assert first.requested_thinking_budget == 2048
    assert first.requested_timeout_seconds == 600
    assert first.max_tokens is None
    assert first.max_completion_tokens == 4096
    assert first.status == "parsed"
    assert first.usage is not None and first.usage.total_tokens == 0
    assert parse_visual_feedback_output(first.raw_response_text or "") == (
        first.parsed_feedback
    )
    receipt = write_feedback_evaluation_result(tmp_path / "feedback.json", first)
    assert load_feedback_evaluation_result(
        receipt, expected_result_sha256=first.result_sha256
    ) == first
    assert runner.calls == 2


def test_fake_codex_is_exactly_once_and_emits_clean_turn(tmp_path: Path) -> None:
    response = b'{"schema_version":2,"drafts":[]}'
    process = _FakeCodexProcess(response)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    final = tmp_path / "final.json"
    command = (
        "codex.exe",
        "exec",
        "--output-last-message",
        str(final),
    )

    result = process(
        command,
        stdin=b"request",
        cwd=scratch,
        environment={},
        timeout_seconds=600,
    )

    assert result.returncode == 0
    assert final.read_bytes() == response
    assert process.calls == 1
    with pytest.raises(PortfolioS1ZeroProviderSmokeError, match="more than once"):
        process(
            command,
            stdin=b"request",
            cwd=scratch,
            environment={},
            timeout_seconds=600,
        )


def test_network_guard_fails_closed_without_real_connection() -> None:
    guard = _NetworkDenied()
    with (
        guard,
        pytest.raises(PortfolioS1ZeroProviderSmokeError, match="network connection"),
    ):
        import socket

        socket.create_connection(("example.invalid", 443))
    assert guard.attempts == 1


def test_twice_report_requires_identical_canonical_invariants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def fake_once(_args, root: Path):
        calls.append(root)
        runtime_digit = "7" if root.name == "run-1" else "a"
        runtime_file_digit = "8" if root.name == "run-1" else "b"
        return {
            "selection_sha256": "1" * 64,
            "control_sha256": "2" * 64,
            "feedback_bundle_sha256": "3" * 64,
            "feedback_model_projection_sha256": "4" * 64,
            "creator_candidate_bank_sha256": "5" * 64,
            "creator_candidate_bank_file_sha256": "6" * 64,
            "two_bank_runtime_lock_sha256": runtime_digit * 64,
            "two_bank_runtime_lock_file_sha256": runtime_file_digit * 64,
            "parent_static_bank_sha256": "9" * 64,
            "style_tool_version": "2.3.0",
            "selected_rows": 48,
            "local_feedback_evaluations": 48,
            "local_fake_codex_invocations": 1,
            "external_feedback_provider_calls": 0,
            "external_codex_provider_calls": 0,
            "external_assistant_provider_calls": 0,
            "replay_status": REPLAY_NOT_RUN,
            "body_gate_status": REPLAY_NOT_RUN,
        }

    monkeypatch.setattr(
        "scripts.smoke_portfolio_s1_feedback_loop.run_smoke_once", fake_once
    )
    output = tmp_path / "smoke"
    report = run_smoke_twice(argparse.Namespace(output_root=output))

    assert calls == [output / "run-1", output / "run-2"]
    assert report["status"] == "passed"
    assert report["external_provider_call_count"] == 0
    assert report["canonical_invariants_equal"] is True
    assert report["path_bound_runtime_identities_equal"] is False
    assert report["replay_status"] == REPLAY_NOT_RUN
    assert (output / "smoke-report.json").is_file()
