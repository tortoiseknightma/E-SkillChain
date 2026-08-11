from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import Lock

import pytest

from scripts import run_portfolio_s1_feedback_round3_schema_v1 as cli
from skillchain import config
from skillchain.evaluation import portfolio_s1_feedback_recovery_v1 as recovery
from skillchain.evaluation import portfolio_s1_feedback_round3_schema_v1 as schema
from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    visual_feedback_response_format_v1,
)
from skillchain.evaluation.packets import (
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    EMPTY_SHA256,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
    ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
    RecoveryFeedbackEvaluationResultV1,
    RecoveryFeedbackEvaluationResultV2,
    _make_recovery_result,
)
from skillchain.llm import LLMResponse, LLMUsage
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BASE_PARENT = _REPOSITORY_ROOT / Path(
    "runs/portfolio/core-s1/s1-feedback-round2-qwen38-v3"
)
_RECOVERY_PARENT = _REPOSITORY_ROOT / Path(
    "runs/portfolio/core-s1/s1-feedback-round2-qwen38-recovery-v1"
)
_STOPPED_OBJECT = _REPOSITORY_ROOT / Path(
    "runs/portfolio/core-s1/s1-feedback-round3-qwen38-v1"
)
_ARTIFACT_ROOT = Path(r"C:\Users\torto\.codex\worktrees\b5cd\ECommerceSkillChain")
_EXECUTION_ROOT = _ARTIFACT_ROOT / Path(
    "runs/portfolio/core-static-opt/static-opt-execution-v6"
)
_EXECUTION_CONTROL_FILE_SHA256 = (
    "dcd94c985f6c76f4d5858fa29c223fcc2318ea3284609f309e1105e52800931e"
)


@pytest.fixture(scope="session")
def prepared_session(tmp_path_factory: pytest.TempPathFactory):
    if not _EXECUTION_ROOT.exists():
        pytest.skip("frozen opt800 source root is not mounted")
    output = tmp_path_factory.mktemp("round3-schema-authority")
    arguments = argparse.Namespace(
        repository_root=_REPOSITORY_ROOT,
        base_parent_root=_BASE_PARENT,
        recovery_root=_RECOVERY_PARENT,
        stopped_object_root=_STOPPED_OBJECT,
        execution_root=_EXECUTION_ROOT,
        expected_execution_control_sha256=_EXECUTION_CONTROL_FILE_SHA256,
        artifact_repository_root=_ARTIFACT_ROOT,
        output_dir=output,
        authorization_id="portfolio-s1-qwen38-round3-schema-focused-v1",
        reviewer_id="codex-test",
        reviewed_at="2026-08-11T00:00:00Z",
        run_id="portfolio-s1-qwen38-round3-schema-focused-v1",
    )
    return cli.prepare_round3_schema_canary_v1(arguments)


@pytest.fixture(autouse=True)
def fast_verified_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        schema,
        "require_verified_static_feedback_source_v2",
        lambda source, _selection, _control, _entry: source,
    )


def _fresh_prepared(prepared_session, root: Path):
    root.mkdir()
    for name in (cli.SCHEMA_CLAIM_DIR, cli.SCHEMA_ATTEMPT_DIR, cli.SCHEMA_BOUND_DIR):
        (root / name).mkdir()
    return replace(prepared_session, output_dir=root)


def _template_feedback(prepared):
    template = next(
        item.feedback_result
        for item in prepared.predecessors.prior.base.artifacts
        if item.status == "parsed"
    )
    assert template.parsed_feedback is not None
    return template.parsed_feedback


def _result(
    prepared,
    source,
    *,
    status: str = "parsed",
    wire_kind: str = "round3_primary_json_schema_v1",
    input_tokens: int = 100,
    output_tokens: int = 100,
):
    schema_wire = wire_kind == "round3_primary_json_schema_v1"
    common = {
        "wire_kind": wire_kind,
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
            if schema_wire
            else ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
        ),
        "transport_policy_sha256": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
            if schema_wire
            else ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
        ),
        "requested_response_format": "json_schema" if schema_wire else "json_object",
        "requested_json_schema_sha256": (
            VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1 if schema_wire else None
        ),
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
    feedback = _template_feedback(prepared)
    raw = (
        canonical_json_bytes(feedback.model_dump(mode="json")).decode("utf-8")
        if status == "parsed"
        else "{"
    )
    receipt = {
        "request_id": f"req-{source.packet.query_id}-{status}",
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
            **common, **receipt, status="parsed", parsed_feedback=feedback
        )
    return _make_recovery_result(
        **common,
        **receipt,
        status="parse_error",
        error_code="invalid_feedback_json",
    )


def test_prepare_is_zero_call_and_imports_no_historical_feedback(prepared_session):
    prepared = prepared_session
    assert prepared.authorization.authorized_selection_ordinals == tuple(range(1, 13))
    assert prepared.authorization.provider_call_ceiling == 15
    assert prepared.authorization.response_format == "json_schema"
    assert prepared.predecessors.receipt.historical_feedback_outputs_imported == 0
    assert (
        prepared.predecessors.receipt.terminated_json_object_canary_outputs_imported
        == 0
    )
    assert prepared.bundle_identity.live_canary_publishable is False
    assert not tuple((prepared.output_dir / cli.SCHEMA_ATTEMPT_DIR).iterdir())
    assert not tuple((prepared.output_dir / cli.SCHEMA_BOUND_DIR).iterdir())
    assert not (prepared.output_dir / cli.SCHEMA_RUN_FILE).exists()


def test_round3_schema_runner_passes_exact_openai_compatible_payload(
    prepared_session, monkeypatch: pytest.MonkeyPatch
):
    source = prepared_session.sources[0]
    feedback = _template_feedback(prepared_session)
    raw = canonical_json_bytes(feedback.model_dump(mode="json")).decode("utf-8")
    calls: list[tuple[str, list[dict[str, object]], dict[str, object]]] = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model="qwen3.8-max",
            response_model="qwen3.8-max",
            request_id="req-round3-schema-wire",
            text=raw,
            usage=LLMUsage(input_tokens=100, output_tokens=100),
            finish_reason="stop",
            latency_ms=1,
        )

    monkeypatch.setattr(recovery.llm, "chat", fake_chat)
    result = recovery.run_visual_feedback_round3_schema_primary_v1(
        source.packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=prepared_session.remote_runtime,
        record_usage=True,
    )

    assert type(result) is RecoveryFeedbackEvaluationResultV2
    assert result.status == "parsed"
    provider, _messages, kwargs = calls[0]
    assert provider == "qwen"
    response_format = kwargs["response_format"]
    assert response_format == visual_feedback_response_format_v1()
    sdk_payload = response_format.model_dump(mode="json", by_alias=True)
    assert sdk_payload["type"] == "json_schema"
    assert "strict" not in sdk_payload
    assert sdk_payload["json_schema"]["name"] == "visual_feedback_output_v1"
    assert sdk_payload["json_schema"]["strict"] is True
    assert sdk_payload["json_schema"]["schema"]["additionalProperties"] is False
    assert kwargs["json_mode"] is False
    assert kwargs["qwen_feedback_recovery_json_object"] is False
    assert kwargs["max_attempts"] == 1
    assert kwargs["thinking_budget"] == 2048
    assert kwargs["max_completion_tokens"] == 6144


def test_mixed_json_object_result_is_rejected(prepared_session, tmp_path: Path):
    prepared = _fresh_prepared(prepared_session, tmp_path / "mixed-wire")
    source = prepared.sources[0]
    reservation = schema.build_round3_schema_reservation_v2(
        prepared.predecessors,
        prepared.governance,
        prepared.authorization,
        prepared.control,
        prepared.remote_receipt,
        source,
        attempt_index=1,
        global_call_ordinal=1,
    )
    old_result = _result(prepared, source, wire_kind="round3_primary_json_object_v1")
    assert type(old_result) is RecoveryFeedbackEvaluationResultV1
    with pytest.raises(
        schema.PortfolioS1FeedbackError, match="provider result drifted"
    ):
        schema.build_bound_round3_schema_artifact_v2(
            prepared.predecessors,
            prepared.governance,
            prepared.authorization,
            prepared.control,
            prepared.remote_receipt,
            source,
            old_result,  # type: ignore[arg-type]
            reservation=reservation,
        )


def test_same_wave_partial_failure_publishes_orphan_without_recall(
    prepared_session, tmp_path: Path
):
    prepared = _fresh_prepared(prepared_session, tmp_path / "partial-orphan")
    calls: Counter[str] = Counter()

    def runner(source):
        calls[source.packet.query_id] += 1
        if source.selection_entry_sha256 == prepared.sources[0].selection_entry_sha256:
            raise RuntimeError("simulated paid-call settlement loss")
        return _result(prepared, source)

    outcome = cli.execute_round3_schema_canary_v1(prepared, runner=runner)

    assert outcome.run is not None
    assert outcome.run.status == "stopped_orphan"
    assert outcome.run.provider_calls_reserved == 2
    assert outcome.run.orphan_count == 1
    assert len(outcome.ledger.artifacts) == 1
    assert calls == Counter(
        {
            prepared.sources[0].packet.query_id: 1,
            prepared.sources[1].packet.query_id: 1,
        }
    )


def test_three_global_retries_complete_canary_at_15_call_ceiling(
    prepared_session, tmp_path: Path
):
    prepared = _fresh_prepared(prepared_session, tmp_path / "retry-ceiling")
    calls: Counter[str] = Counter()
    lock = Lock()
    retry_ordinals = {1, 3, 5}
    retry_entry_sha256s = {
        prepared.sources[ordinal - 1].selection_entry_sha256
        for ordinal in retry_ordinals
    }

    def runner(source):
        with lock:
            calls[source.packet.query_id] += 1
            call_index = calls[source.packet.query_id]
        if source.selection_entry_sha256 in retry_entry_sha256s and call_index == 1:
            return _result(prepared, source, status="parse_error")
        return _result(prepared, source)

    outcome = cli.execute_round3_schema_canary_v1(prepared, runner=runner)

    assert outcome.run is not None
    assert outcome.run.status == "completed_canary"
    assert outcome.run.parsed_count == 12
    assert outcome.run.provider_calls_reserved == 15
    assert outcome.run.retry_count == 3
    assert len(outcome.ledger.claims) == 3
    with pytest.raises(ValueError, match="15-call provider ceiling"):
        schema.require_round3_schema_pre_reservation_budget_v1(outcome.ledger)


def test_phase60_is_rejected_before_any_new_reservation(prepared_session):
    ledger = cli._load_ledger(prepared_session)
    with pytest.raises(schema.PortfolioS1FeedbackError, match="phase60"):
        schema.next_round3_schema_step_v1(
            prepared_session.predecessors,
            prepared_session.governance,
            prepared_session.authorization,
            prepared_session.control,
            prepared_session.remote_receipt,
            ledger,
            prepared_session.sources,
            phase_count=60,
        )
