from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from threading import Lock
import time

import pytest

from scripts import run_portfolio_s1_feedback_round3_v1 as cli
from skillchain import config
from skillchain.evaluation.packets import (
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.evaluation.portfolio_s1_feedback import PortfolioS1FeedbackError
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    EMPTY_SHA256,
    ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
    _make_recovery_result,
)
from skillchain.evaluation import portfolio_s1_feedback_round3_v1 as round3
from skillchain.evaluation.portfolio_s1_feedback_round3_v1 import (
    PortfolioS1FeedbackRound3AuthorizationV1,
    VerifiedPortfolioS1FeedbackRound3GovernanceV1,
    build_round3_authorization_v1,
)
from skillchain.llm import LLMUsage
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


_BASE_PARENT = Path("runs/portfolio/core-s1/s1-feedback-round2-qwen38-v3")
_RECOVERY_PARENT = Path("runs/portfolio/core-s1/s1-feedback-round2-qwen38-recovery-v1")
_ARTIFACT_ROOT = Path(r"C:\Users\torto\.codex\worktrees\b5cd\ECommerceSkillChain")
_EXECUTION_ROOT = _ARTIFACT_ROOT / Path(
    "runs/portfolio/core-static-opt/static-opt-execution-v6"
)
_EXECUTION_CONTROL_FILE_SHA256 = (
    "dcd94c985f6c76f4d5858fa29c223fcc2318ea3284609f309e1105e52800931e"
)


@pytest.fixture(scope="session")
def prepared_session(tmp_path_factory: pytest.TempPathFactory):
    required_roots = (
        _BASE_PARENT,
        _RECOVERY_PARENT,
        _ARTIFACT_ROOT,
        _EXECUTION_ROOT,
    )
    if any(not root.exists() for root in required_roots):
        pytest.skip("frozen Round3 evidence roots are not mounted")
    output = tmp_path_factory.mktemp("round3-authority")
    arguments = argparse.Namespace(
        repository_root=Path.cwd(),
        base_parent_root=_BASE_PARENT,
        recovery_root=_RECOVERY_PARENT,
        execution_root=_EXECUTION_ROOT,
        expected_execution_control_sha256=_EXECUTION_CONTROL_FILE_SHA256,
        artifact_repository_root=_ARTIFACT_ROOT,
        output_dir=output,
        authorization_id="portfolio-s1-qwen38-round3-focused-v1",
        reviewer_id="owner",
        reviewed_at="2026-08-11T00:00:00Z",
        run_id="portfolio-s1-qwen38-round3-focused-v1",
    )
    return cli.prepare_round3_run_v1(arguments)


@pytest.fixture(autouse=True)
def fast_verified_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        round3,
        "require_verified_static_feedback_source_v2",
        lambda source, _selection, _control, _entry: source,
    )


def _fresh_prepared(prepared_session, root: Path):
    root.mkdir()
    for name in (cli.ROUND3_CLAIM_DIR, cli.ROUND3_ATTEMPT_DIR, cli.ROUND3_BOUND_DIR):
        (root / name).mkdir()
    return replace(prepared_session, output_dir=root)


def _template_feedback(prepared):
    template = next(
        item.feedback_result
        for item in prepared.predecessors.base.artifacts
        if item.status == "parsed"
    )
    assert template.parsed_feedback is not None
    return template.parsed_feedback


def _result(
    prepared,
    source,
    *,
    status: str = "parsed",
    input_tokens: int = 100,
    output_tokens: int = 100,
):
    common = {
        "wire_kind": "round3_primary_json_object_v1",
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
        "transport_policy_sha256": ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
        "requested_response_format": "json_object",
        "requested_json_schema_sha256": None,
        "requested_stream": False,
        "query_id": source.packet.query_id,
        "packet_sha256": source.packet.packet_sha256,
        "prompt_sha256": "1" * 64,
        "image_sha256": source.packet.image.sha256,
        "wire_sha256": "2" * 64,
        "asset_catalog_sha256": prepared.control.remote_catalog_sha256,
        "remote_authorization_id": prepared.authorization.authorization_id,
        "remote_authorization_file_sha256": sha256_bytes(
            prepared.authorization.canonical_bytes()
        ),
        "remote_receipt_file_sha256": prepared.control.remote_receipt_file_sha256,
        "remote_receipt_sha256": prepared.control.remote_receipt_sha256,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
    }
    if status == "provider_error":
        return _make_recovery_result(
            **common, status="provider_error", error_code="provider_error"
        )
    feedback = _template_feedback(prepared)
    raw = (
        canonical_json_bytes(feedback.model_dump(mode="json")).decode("utf-8")
        if status == "parsed"
        else "{"
    )
    receipt = {
        "request_id": f"req-{source.packet.query_id}",
        "raw_response_text": raw,
        "raw_response_sha256": sha256_bytes(raw.encode("utf-8")),
        "raw_response_bytes": len(raw.encode("utf-8")),
        "tool_calls": (),
        "tool_call_count": 0,
        "usage": LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        "finish_reason": "stop",
        "latency_ms": 1,
        "reasoning_present": False,
        "reasoning_tokens": 0,
        "reasoning_bytes": 0,
        "reasoning_sha256": None,
        "refusal_present": False,
        "refusal_bytes": 0,
        "refusal_sha256": EMPTY_SHA256,
    }
    if status == "parsed":
        return _make_recovery_result(
            **common,
            **receipt,
            status="parsed",
            parsed_feedback=feedback,
        )
    return _make_recovery_result(
        **common,
        **receipt,
        status="parse_error",
        error_code="invalid_feedback_json",
    )


def test_governance_is_live_authority_and_history_is_not_imported(prepared_session):
    prepared = prepared_session
    assert prepared.authorization.live_call_authority is True
    assert prepared.authorization.old_remote_auth_does_not_authorize_round3_transport
    assert prepared.authorization.membership_ancestry_role.endswith(
        "not_call_authorization"
    )
    assert prepared.control.historical_feedback_outputs_imported == 0
    assert prepared.predecessors.receipt.historical_feedback_outputs_imported == 0
    forged = VerifiedPortfolioS1FeedbackRound3GovernanceV1(
        repository_root=prepared.governance.repository_root,
        source_lock=prepared.governance.source_lock,
        pricing_lock=prepared.governance.pricing_lock,
        role_selection=prepared.governance.role_selection,
    )
    with pytest.raises(PortfolioS1FeedbackError, match="not verified"):
        build_round3_authorization_v1(
            prepared.predecessors,
            forged,
            authorization_id="forged",
            reviewer_id="owner",
            reviewed_at="2026-08-11T00:00:00Z",
        )
    with pytest.raises(ValueError, match="second precision"):
        PortfolioS1FeedbackRound3AuthorizationV1.model_validate(
            {
                **prepared.authorization.model_dump(mode="json"),
                "reviewed_at": "2026-08-11T00:00:00.1Z",
            },
            strict=True,
        )


def test_normal240_bundle_provenance_and_missing_bundle_repair(
    prepared_session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prepared = _fresh_prepared(prepared_session, tmp_path / "normal240")
    reservations = []
    artifacts = []
    for ordinal, source in enumerate(prepared.sources, 1):
        reservation = round3.build_round3_reservation_v1(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.remote_receipt,
            source,
            attempt_index=1,
            global_call_ordinal=ordinal,
        )
        artifact = round3.build_bound_round3_artifact_v1(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.remote_receipt,
            source,
            _result(prepared, source),
            reservation=reservation,
        )
        reservations.append(reservation)
        artifacts.append(artifact)
    ledger = round3.PortfolioS1FeedbackRound3LedgerV1(
        claims=(),
        reservations=tuple(reservations),
        artifacts=tuple(artifacts),
        orphaned_reservations=(),
        pending_claims=(),
    )
    run = round3.build_round3_run_v1(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        ledger,
        prepared.sources,
    )
    bundle = round3.build_portfolio_s1_feedback_bundle_v10(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        ledger,
        run,
        prepared.sources,
    )
    assert run.status == "completed" and run.provider_calls_reserved == 240
    assert bundle.provider_call_count == 240
    assert bundle.round3_artifact_set_sha256 == run.artifact_set_sha256
    assert len(bundle.entry_provenance) == 240
    assert {item.origin for item in bundle.entry_provenance} == {
        "round3_fresh_qwen_json_object"
    }
    cli._publish_or_resume(
        prepared.output_dir / cli.ROUND3_RUN_FILE,
        run.canonical_bytes(),
        label="test completed Round3 run",
    )
    monkeypatch.setattr(cli, "_load_ledger", lambda _prepared: ledger)
    calls = 0

    def forbidden(_source):
        nonlocal calls
        calls += 1
        raise AssertionError("bundle repair must not call provider")

    repaired = cli.execute_round3_run_v1(prepared, runner=forbidden)
    assert calls == 0
    assert repaired.bundle == bundle


def test_primary_phase_uses_two_call_waves(prepared_session, tmp_path: Path):
    prepared = _fresh_prepared(prepared_session, tmp_path / "concurrency2")
    active = 0
    maximum = 0
    lock = Lock()

    def runner(source):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.003)
        result = _result(prepared, source)
        with lock:
            active -= 1
        return result

    outcome = cli.execute_round3_run_v1(prepared, runner=runner, stop_after_count=12)
    assert outcome.run is None
    assert len(outcome.ledger.artifacts) == 12
    assert maximum == 2


def test_retry_claim_and_three_claim_ceiling(prepared_session, tmp_path: Path):
    prepared = _fresh_prepared(prepared_session, tmp_path / "retry-ceiling")
    calls: Counter[str] = Counter()
    fail_ordinals = {1, 3, 5, 7}
    ordinal_by_query = {
        item.query_id: item.selection_ordinal
        for item in prepared.predecessors.base.selection.entries
    }

    def runner(source):
        calls[source.packet.query_id] += 1
        ordinal = ordinal_by_query[source.packet.query_id]
        if ordinal in fail_ordinals and calls[source.packet.query_id] == 1:
            return _result(prepared, source, status="parse_error")
        return _result(prepared, source)

    outcome = cli.execute_round3_run_v1(prepared, runner=runner, stop_after_count=12)
    assert outcome.run is not None and outcome.run.status == "stopped_nonparsed"
    assert outcome.run.retry_count == 3
    assert len(outcome.ledger.claims) == 3
    assert outcome.run.provider_calls_reserved <= 243
    assert calls[prepared.sources[0].packet.query_id] == 2
    assert calls[prepared.sources[6].packet.query_id] == 1


def test_same_wave_first_settlement_failure_keeps_second_and_stops_orphan(
    prepared_session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prepared = _fresh_prepared(prepared_session, tmp_path / "partial-orphan")
    original = cli.write_bound_round3_artifact_v1

    def fail_first(path, artifact):
        if artifact.global_call_ordinal == 1:
            raise OSError("injected first settlement failure")
        return original(path, artifact)

    monkeypatch.setattr(cli, "write_bound_round3_artifact_v1", fail_first)
    calls = 0

    def runner(source):
        nonlocal calls
        calls += 1
        return _result(prepared, source)

    outcome = cli.execute_round3_run_v1(prepared, runner=runner, stop_after_count=12)
    assert calls == 2
    assert outcome.run is not None and outcome.run.status == "stopped_orphan"
    assert tuple(
        item.global_call_ordinal for item in outcome.ledger.orphaned_reservations
    ) == (1,)
    assert tuple(item.global_call_ordinal for item in outcome.ledger.artifacts) == (2,)


def test_usage_breach_is_published_as_budget_terminal(prepared_session, tmp_path: Path):
    prepared = _fresh_prepared(prepared_session, tmp_path / "usage")

    def runner(source):
        return _result(
            prepared,
            source,
            input_tokens=20_001 if source is prepared.sources[0] else 100,
        )

    outcome = cli.execute_round3_run_v1(prepared, runner=runner, stop_after_count=12)
    assert outcome.run is not None and outcome.run.status == "stopped_budget"
    assert outcome.run.terminal_reason == "usage_limit_exceeded"
    assert outcome.run.provider_calls_reserved == 2


def test_provider_error_beats_retry_in_same_wave(prepared_session, tmp_path: Path):
    prepared = _fresh_prepared(prepared_session, tmp_path / "same-wave-terminal")

    def runner(source):
        if source is prepared.sources[0]:
            return _result(prepared, source, status="parse_error")
        return _result(prepared, source, status="provider_error")

    outcome = cli.execute_round3_run_v1(prepared, runner=runner, stop_after_count=12)
    assert outcome.run is not None and outcome.run.status == "stopped_nonparsed"
    assert outcome.run.provider_calls_reserved == 2
    assert outcome.run.retry_count == 0
    assert outcome.ledger.claims == ()


def test_zero_provider_dry_run_root_has_no_attempts(prepared_session):
    prepared = prepared_session
    ledger = cli._load_ledger(prepared)
    assert ledger.reservations == ()
    assert ledger.artifacts == ()
    assert not (prepared.output_dir / cli.ROUND3_RUN_FILE).exists()
    payload = json.loads(canonical_json_bytes(cli._summary("dry-run", prepared, None)))
    assert payload["provider_calls_performed_by_preflight"] == 0
    assert payload["historical_feedback_outputs_imported"] == 0


def test_owner_env_overrides_stale_key_and_only_execute_loads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=fresh-test-key\n", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "stale-test-key")
    cli._load_owner_dashscope_environment(env_file)
    assert config.os.environ["DASHSCOPE_API_KEY"] == "fresh-test-key"

    base_argv = [
        "--base-parent-root",
        "unused-base",
        "--recovery-root",
        "unused-recovery",
        "--execution-root",
        "unused-execution",
        "--expected-execution-control-sha256",
        "0" * 64,
        "--artifact-repository-root",
        "unused-repository",
        "--output-dir",
        "unused-output",
        "--authorization-id",
        "unused-auth",
        "--reviewer-id",
        "owner",
        "--reviewed-at",
        "2026-08-11T00:00:00Z",
        "--run-id",
        "unused-run",
    ]
    loaded = 0

    def record_load(_path):
        nonlocal loaded
        loaded += 1
        raise cli.PortfolioS1FeedbackRound3CLIError("stop-before-prepare")

    def stop_prepare(_arguments):
        raise cli.PortfolioS1FeedbackRound3CLIError("prepare-called")

    monkeypatch.setattr(cli, "_load_owner_dashscope_environment", record_load)
    monkeypatch.setattr(cli, "prepare_round3_run_v1", stop_prepare)
    assert cli.main(["--mode", "prepare", *base_argv]) == 2
    assert cli.main(["--mode", "dry-run", *base_argv]) == 2
    assert loaded == 0
    assert cli.main(["--mode", "execute", *base_argv]) == 2
    assert loaded == 1
