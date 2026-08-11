from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_portfolio_s1_feedback_round3_schema_v1 as canary_cli
from skillchain import config
from skillchain.evaluation import (
    portfolio_s1_feedback_round3_schema_phase60_v1 as phase60,
)
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
)
from skillchain.evaluation.packets import (
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    EMPTY_SHA256,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
    RecoveryFeedbackEvaluationResultV2,
    _make_recovery_result,
)
from skillchain.llm import LLMUsage
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANARY_RELATIVE_ROOT = Path(
    "runs/portfolio/core-s1/s1-feedback-round3-qwen38-schema-v1"
)
_BASE_PARENT_RELATIVE_ROOT = Path("runs/portfolio/core-s1/s1-feedback-round2-qwen38-v3")
_RECOVERY_PARENT_RELATIVE_ROOT = Path(
    "runs/portfolio/core-s1/s1-feedback-round2-qwen38-recovery-v1"
)
_STOPPED_OBJECT_RELATIVE_ROOT = Path(
    "runs/portfolio/core-s1/s1-feedback-round3-qwen38-v1"
)
_EXECUTION_RELATIVE_ROOT = Path(
    "runs/portfolio/core-static-opt/static-opt-execution-v6"
)
_EXECUTION_CONTROL_FILE_SHA256 = (
    "dcd94c985f6c76f4d5858fa29c223fcc2318ea3284609f309e1105e52800931e"
)


def test_phase60_cli_bootstrap_keeps_dotenv_disabled_during_imports(
    tmp_path: Path,
) -> None:
    script = (
        _REPOSITORY_ROOT
        / "scripts"
        / "run_portfolio_s1_feedback_round3_schema_phase60_v1.py"
    )
    probe = """
import os
from pathlib import Path
import runpy
import sys
import dotenv

states = []

def guarded_load_dotenv(*args, **kwargs):
    state = os.environ.get("PYTHON_DOTENV_DISABLED")
    states.append(state)
    if state != "1":
        raise RuntimeError("dotenv import was not disabled")
    return False

dotenv.load_dotenv = guarded_load_dotenv
sys.argv = [str(Path(sys.argv[1])), "--help"]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as error:
    if error.code != 0:
        raise
print("DOTENV_IMPORT_STATES=" + ",".join(states))
"""
    environment = dict(os.environ)
    environment.pop("DASHSCOPE_API_KEY", None)
    environment.pop("PYTHON_DOTENV_DISABLED", None)
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DOTENV_IMPORT_STATES=1" in completed.stdout


def _candidate_repositories() -> tuple[Path, ...]:
    candidates = [_REPOSITORY_ROOT]
    worktrees = Path.home() / ".codex" / "worktrees"
    if worktrees.is_dir():
        candidates.extend(
            item / "ECommerceSkillChain"
            for item in sorted(worktrees.iterdir(), key=lambda path: path.name)
            if item.is_dir()
        )
    return tuple(candidates)


def _repository_containing(relative_path: Path) -> Path | None:
    return next(
        (
            candidate
            for candidate in _candidate_repositories()
            if (candidate / relative_path).exists()
        ),
        None,
    )


@pytest.fixture(scope="session")
def phase60_authority(tmp_path_factory: pytest.TempPathFactory):
    evidence_repository = _repository_containing(_CANARY_RELATIVE_ROOT)
    artifact_repository = _repository_containing(_EXECUTION_RELATIVE_ROOT)
    if evidence_repository is None or artifact_repository is None:
        pytest.skip("frozen phase60 evidence roots are not mounted")
    required = (
        evidence_repository / _BASE_PARENT_RELATIVE_ROOT,
        evidence_repository / _RECOVERY_PARENT_RELATIVE_ROOT,
        evidence_repository / _STOPPED_OBJECT_RELATIVE_ROOT,
        evidence_repository / _CANARY_RELATIVE_ROOT,
        artifact_repository / _EXECUTION_RELATIVE_ROOT,
    )
    if any(not path.exists() for path in required):
        pytest.skip("frozen phase60 predecessor roots are incomplete")

    canary_prepare_output = tmp_path_factory.mktemp("phase60-canary-authority")
    prepared = canary_cli.prepare_round3_schema_canary_v1(
        argparse.Namespace(
            repository_root=_REPOSITORY_ROOT,
            base_parent_root=required[0],
            recovery_root=required[1],
            stopped_object_root=required[2],
            execution_root=required[4],
            expected_execution_control_sha256=_EXECUTION_CONTROL_FILE_SHA256,
            artifact_repository_root=artifact_repository,
            output_dir=canary_prepare_output,
            authorization_id="phase60-focused-canary-authority-v1",
            reviewer_id="codex-test",
            reviewed_at="2026-08-11T00:00:00Z",
            run_id="phase60-focused-canary-authority-v1",
        )
    )
    prefix = phase60.load_verified_round3_schema_canary_prefix_v1(
        _REPOSITORY_ROOT,
        required[3],
        predecessors=prepared.predecessors,
        governance=prepared.governance,
        sources=prepared.sources,
    )
    pending_governance = phase60.load_verified_round3_schema_phase60_governance_v1(
        _REPOSITORY_ROOT
    )

    # The tracked V5/V8/V15 triad is intentionally pending and cannot be made
    # live by a production parser.  These unvalidated copies exist only inside
    # this test module so the already-gated state machine can be exercised.
    source_lock = pending_governance.source_lock.model_copy(
        update={
            "live_call_authority": True,
            "live_provider_calls_authorized": True,
            "owner_phase60_budget_authorization_status": "granted",
            "owner_phase60_retry_authorization_status": "granted",
        }
    )
    pricing_lock = pending_governance.pricing_lock.model_copy(
        update={
            "fresh_run_and_retry_scope_owner_approved": True,
            "live_provider_calls_authorized": True,
            "owner_budget_authorization_status": (
                "granted_phase60_budget_and_retry_approval"
            ),
            "owner_phase60_retry_authorization_status": "granted",
        }
    )
    feedback_evaluator = dict(pending_governance.role_selection.feedback_evaluator)
    feedback_evaluator.update(
        {
            "live_call_authority": True,
            "live_provider_calls_authorized": True,
            "owner_phase60_budget_authorization_status": "granted",
            "owner_phase60_retry_authorization_status": "granted",
        }
    )
    role_selection = pending_governance.role_selection.model_copy(
        update={"feedback_evaluator": feedback_evaluator}
    )
    live_governance = replace(
        pending_governance,
        source_lock=source_lock,
        pricing_lock=pricing_lock,
        role_selection=role_selection,
    )
    approval = phase60.build_round3_schema_phase60_owner_approval_v1(
        prefix,
        live_governance,
        approval_id="phase60-focused-test-only-owner-grant-v1",
        reviewer_id="codex-test",
        reviewed_at="2026-08-11T00:00:00+00:00",
    )
    authorization = phase60.build_round3_schema_phase60_authorization_v1(
        prefix,
        live_governance,
        approval,
        prepared.sources,
        authorization_id="phase60-focused-test-only-authority-v1",
    )
    control = phase60.build_round3_schema_phase60_control_v1(
        prefix,
        live_governance,
        approval,
        authorization,
        prepared.sources,
    )
    launch = phase60.build_round3_schema_phase60_launch_v1(
        prefix,
        live_governance,
        approval,
        authorization,
        control,
        prepared.sources,
        run_id="phase60-focused-test-only-run-v1",
    )
    return SimpleNamespace(
        evidence_repository=evidence_repository,
        canary_root=required[3],
        prepared=prepared,
        prefix=prefix,
        pending_governance=pending_governance,
        governance=live_governance,
        approval=approval,
        authorization=authorization,
        control=control,
        launch=launch,
        sources=prepared.sources,
    )


@pytest.fixture(autouse=True)
def fast_verified_phase60_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        phase60,
        "require_verified_static_feedback_source_v2",
        lambda source, _selection, _control, _entry: source,
    )


def _ledger(*, claims=(), reservations=(), artifacts=()):
    settled = {item.reservation_sha256 for item in artifacts}
    used_claim_count = sum(item.attempt_index == 2 for item in reservations)
    return phase60.PortfolioS1FeedbackRound3SchemaPhase60LedgerV1(
        claims=tuple(claims),
        reservations=tuple(reservations),
        artifacts=tuple(artifacts),
        orphaned_reservations=tuple(
            item for item in reservations if item.reservation_sha256 not in settled
        ),
        pending_claims=tuple(claims)[used_claim_count:],
    )


def _template_feedback(authority):
    return next(
        item.feedback_result.parsed_feedback
        for item in authority.prefix.predecessors.prior.base.artifacts
        if item.status == "parsed" and item.feedback_result.parsed_feedback is not None
    )


def _result(
    authority,
    source,
    *,
    status: str = "parsed",
    input_tokens: int = 100,
    output_tokens: int = 100,
):
    feedback = _template_feedback(authority)
    raw = (
        canonical_json_bytes(feedback.model_dump(mode="json")).decode("utf-8")
        if status == "parsed"
        else "{"
    )
    common = {
        "wire_kind": "round3_primary_json_schema_v1",
        "prompt_policy_version": VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        "prompt_policy_sha256": VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        "transport_policy_version": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
        ),
        "transport_policy_sha256": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        ),
        "requested_response_format": "json_schema",
        "requested_json_schema_sha256": VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
        "query_id": source.packet.query_id,
        "packet_sha256": source.packet.packet_sha256,
        "prompt_sha256": "1" * 64,
        "image_sha256": source.packet.image.sha256,
        "wire_sha256": "2" * 64,
        "asset_catalog_sha256": authority.authorization.membership_catalog_sha256,
        "remote_authorization_id": (
            authority.authorization.membership_authorization_id
        ),
        "remote_authorization_file_sha256": (
            authority.authorization.membership_authorization_file_sha256
        ),
        "remote_receipt_file_sha256": (
            authority.authorization.membership_receipt_file_sha256
        ),
        "remote_receipt_sha256": authority.authorization.membership_receipt_sha256,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "request_id": f"req-phase60-{source.packet.query_id}-{status}",
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
        result = _make_recovery_result(
            **common,
            status="parsed",
            parsed_feedback=feedback,
        )
    else:
        result = _make_recovery_result(
            **common,
            status="parse_error",
            error_code="invalid_feedback_json",
        )
    assert type(result) is RecoveryFeedbackEvaluationResultV2
    return result


def _append_primary(
    authority,
    ledger,
    selection_ordinal: int,
    *,
    status: str = "parsed",
    input_tokens: int = 100,
    output_tokens: int = 100,
):
    source = authority.sources[selection_ordinal - 1]
    call_ordinal = len(ledger.reservations) + 1
    reservation = phase60.build_round3_schema_phase60_reservation_v1(
        authority.prefix,
        authority.governance,
        authority.approval,
        authority.authorization,
        authority.control,
        authority.sources,
        source,
        attempt_index=1,
        phase60_call_ordinal=call_ordinal,
    )
    artifact = phase60.build_bound_round3_schema_phase60_artifact_v1(
        authority.prefix,
        authority.governance,
        authority.approval,
        authority.authorization,
        authority.control,
        authority.sources,
        source,
        _result(
            authority,
            source,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        reservation=reservation,
    )
    return _ledger(
        claims=ledger.claims,
        reservations=(*ledger.reservations, reservation),
        artifacts=(*ledger.artifacts, artifact),
    )


def _append_retry(authority, ledger, selection_ordinal: int, *, status="parsed"):
    first = next(
        item
        for item in ledger.artifacts
        if item.selection_ordinal == selection_ordinal and item.attempt_index == 1
    )
    claim = phase60.build_round3_schema_phase60_retry_claim_v1(
        authority.prefix.selection,
        authority.control,
        first,
        existing_claims=ledger.claims,
        existing_artifacts=ledger.artifacts,
    )
    source = authority.sources[selection_ordinal - 1]
    reservation = phase60.build_round3_schema_phase60_reservation_v1(
        authority.prefix,
        authority.governance,
        authority.approval,
        authority.authorization,
        authority.control,
        authority.sources,
        source,
        attempt_index=2,
        phase60_call_ordinal=len(ledger.reservations) + 1,
        first_artifact=first,
        retry_claim=claim,
    )
    artifact = phase60.build_bound_round3_schema_phase60_artifact_v1(
        authority.prefix,
        authority.governance,
        authority.approval,
        authority.authorization,
        authority.control,
        authority.sources,
        source,
        _result(authority, source, status=status),
        reservation=reservation,
        first_artifact=first,
        retry_claim=claim,
    )
    return _ledger(
        claims=(*ledger.claims, claim),
        reservations=(*ledger.reservations, reservation),
        artifacts=(*ledger.artifacts, artifact),
    )


def _build_no_retry_phase60(authority):
    ledger = _ledger()
    for ordinal in phase60.ROUND3_SCHEMA_PHASE60_NEW_ORDINALS:
        ledger = _append_primary(authority, ledger, ordinal)
    return ledger


def _build_twelve_retry_phase60(authority):
    ledger = _ledger()
    for ordinal in range(13, 25):
        ledger = _append_primary(authority, ledger, ordinal, status="parse_error")
        ledger = _append_retry(authority, ledger, ordinal)
    for ordinal in range(25, 61):
        ledger = _append_primary(authority, ledger, ordinal)
    return ledger


def test_tracked_canary_manifest_and_external_prefix_are_exact12(
    phase60_authority,
):
    manifest_path = (
        _REPOSITORY_ROOT
        / phase60.ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1
    )
    content = manifest_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        phase60.ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    )
    manifest = phase60.PortfolioS1FeedbackRound3SchemaCanary12ResultManifestV1.model_validate_json(
        content, strict=True
    )
    assert manifest.canonical_bytes() == content
    assert manifest.top_file_inventory_sha256 == (
        "441a01195ccfabb8c33ce65965a4d76a3fa0dee6d42739ca02e9aadcbc0d4d7b"
    )
    assert manifest.claim_file_inventory_sha256 == (
        "47e185840a7b07802af8d0ece23ca5e1e425e1a5a2f9f2816b8cbe71b254a43f"
    )
    assert manifest.reservation_file_inventory_sha256 == (
        "3793f10a0c1c2c1cb7b682955543f5eb9d024474428d2335a551abe16e1516cd"
    )
    assert manifest.bound_file_inventory_sha256 == (
        "468eee5d58e54b65663e8ed0f41a6c4b6ecae0c6df38bc41e09cf850c4a9e287"
    )
    prefix = phase60_authority.prefix
    assert tuple(item.selection_ordinal for item in prefix.final_artifacts) == tuple(
        range(1, 13)
    )
    assert all(item.status == "parsed" for item in prefix.final_artifacts)
    assert tuple(item.selection_entry_sha256 for item in prefix.final_artifacts) == (
        tuple(item.entry_sha256 for item in prefix.selection.entries[:12])
    )
    assert tuple(item.attempt_index for item in prefix.final_artifacts) == (
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        2,
        1,
    )
    assert len(prefix.ledger.claims) == 3
    assert prefix.run.provider_calls_reserved == 15
    assert prefix.run.run_sha256 == (
        "2d40e9ee4d9cc0d3d92585f1126ddcb1362ee7ef32c35f60c86593d668c0cd72"
    )


def test_manifest_and_canary_byte_drift_fail_closed(phase60_authority, tmp_path: Path):
    manifest_source = (
        _REPOSITORY_ROOT
        / phase60.ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1
    )
    fake_repository = tmp_path / "manifest-drift-repository"
    fake_manifest = (
        fake_repository
        / phase60.ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1
    )
    fake_manifest.parent.mkdir(parents=True)
    fake_manifest.write_bytes(manifest_source.read_bytes() + b" ")
    with pytest.raises(
        phase60.PortfolioS1FeedbackError, match="manifest file hash drifted"
    ):
        phase60.load_verified_round3_schema_canary_prefix_v1(
            fake_repository,
            phase60_authority.canary_root,
            predecessors=phase60_authority.prepared.predecessors,
            governance=phase60_authority.prepared.governance,
            sources=phase60_authority.sources,
        )

    canary_copy = tmp_path / "canary-byte-drift"
    shutil.copytree(phase60_authority.canary_root, canary_copy)
    run_path = canary_copy / "run-round3-schema-v2.json"
    run_path.write_bytes(run_path.read_bytes() + b" ")
    with pytest.raises(phase60.PortfolioS1FeedbackError):
        phase60.load_verified_round3_schema_canary_prefix_v1(
            _REPOSITORY_ROOT,
            canary_copy,
            predecessors=phase60_authority.prepared.predecessors,
            governance=phase60_authority.prepared.governance,
            sources=phase60_authority.sources,
        )


def test_pending_governance_blocks_phase60_before_any_reservation(
    phase60_authority,
):
    pending = phase60_authority.pending_governance
    assert pending.source_lock.live_call_authority is False
    assert pending.pricing_lock.owner_budget_authorized_cap_cny == "0.000000000000"
    assert pending.role_selection.feedback_evaluator["creator_authorized"] is False
    with pytest.raises(
        phase60.PortfolioS1FeedbackError,
        match="pending separate owner budget and retry approval",
    ):
        phase60.require_round3_schema_phase60_live_governance_v1(pending)
    with pytest.raises(
        phase60.PortfolioS1FeedbackError,
        match="pending separate owner budget and retry approval",
    ):
        phase60.build_round3_schema_phase60_owner_approval_v1(
            phase60_authority.prefix,
            pending,
            approval_id="must-not-exist",
            reviewer_id="codex-test",
            reviewed_at="2026-08-11T00:00:00+00:00",
        )
    assert _ledger().reservations == ()


def test_phase60_completes_with_exact_48_new_first_attempts_and_no_publish(
    phase60_authority,
):
    ledger = _build_no_retry_phase60(phase60_authority)
    step = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    assert step.kind == "phase_complete"
    assert step.combined_parsed_count == 60
    assert tuple(item.selection_ordinal for item in ledger.reservations) == tuple(
        range(13, 61)
    )
    assert tuple(item.phase60_call_ordinal for item in ledger.reservations) == tuple(
        range(1, 49)
    )
    assert tuple(
        item.cumulative_global_call_ordinal for item in ledger.reservations
    ) == tuple(range(16, 64))
    run = phase60.build_round3_schema_phase60_run_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.launch,
        ledger,
        phase60_authority.sources,
    )
    assert run.status == "completed_phase60"
    assert run.new_provider_calls_reserved == 48
    assert run.cumulative_provider_calls_reserved == 63
    assert run.new_retry_count == 0
    assert run.canary_provider_attempts_replayed == 0
    assert run.bundle_v11_publishable_from_phase60 is False
    assert run.s1_creator_start_authorized is False
    assert run.phase120_requires_new_owner_approval is True


def test_phase60_twelve_new_retries_are_independent_and_exhaust_exact_ceiling(
    phase60_authority,
):
    ledger = _build_twelve_retry_phase60(phase60_authority)
    assert len(ledger.claims) == 12
    assert len(ledger.reservations) == 60
    assert tuple(item.claim_ordinal for item in ledger.claims) == tuple(range(1, 13))
    assert not (
        {item.claim_sha256 for item in ledger.claims}
        & {item.claim_sha256 for item in phase60_authority.prefix.ledger.claims}
    )
    step = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    assert step.kind == "phase_complete"
    assert step.combined_parsed_count == 60
    run = phase60.build_round3_schema_phase60_run_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.launch,
        ledger,
        phase60_authority.sources,
    )
    assert run.status == "completed_phase60"
    assert run.new_retry_count == 12
    assert run.cumulative_retry_count == 15
    assert run.new_provider_calls_reserved == 60
    assert run.cumulative_provider_calls_reserved == 75
    assert run.bundle_v11_publishable_from_phase60 is False
    assert run.s1_creator_start_authorized is False
    with pytest.raises(phase60.PortfolioS1FeedbackError, match="provider call ceiling"):
        phase60.require_round3_schema_phase60_pre_reservation_budget_v1(ledger)


def test_thirteenth_new_eligible_failure_is_terminal_without_extra_call(
    phase60_authority,
):
    ledger = _ledger()
    for ordinal in range(13, 25):
        ledger = _append_primary(
            phase60_authority, ledger, ordinal, status="parse_error"
        )
        ledger = _append_retry(phase60_authority, ledger, ordinal)
    ledger = _append_primary(phase60_authority, ledger, 25, status="parse_error")
    assert len(ledger.reservations) == 25
    step = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    assert step.kind == "terminal_nonparsed"
    assert step.selection_ordinals == (25,)
    assert step.terminal_error_code == "global_retry_ceiling_exceeded"
    assert step.new_provider_calls_reserved == 25
    assert step.next_phase60_call_ordinal is None
    run = phase60.build_round3_schema_phase60_run_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.launch,
        ledger,
        phase60_authority.sources,
    )
    assert run.status == "stopped_nonparsed"
    assert run.new_provider_calls_reserved == 25
    assert run.bundle_v11_publishable_from_phase60 is False


def test_reserved_but_unsettled_call_is_terminal_orphan_and_never_recalled(
    phase60_authority,
):
    source = phase60_authority.sources[12]
    reservation = phase60.build_round3_schema_phase60_reservation_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.sources,
        source,
        attempt_index=1,
        phase60_call_ordinal=1,
    )
    ledger = _ledger(reservations=(reservation,))
    first = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    resumed = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    assert first == resumed
    assert first.kind == "terminal_orphan"
    assert first.selection_ordinals == (13,)
    assert first.next_phase60_call_ordinal is None
    run = phase60.build_round3_schema_phase60_run_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.launch,
        ledger,
        phase60_authority.sources,
    )
    assert run.status == "stopped_orphan"
    assert run.orphan_count == 1
    assert run.fresh_accountable_cost_cny == (
        phase60.ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    )


def test_usage_budget_breach_stops_before_next_reservation(
    phase60_authority,
):
    ledger = _append_primary(
        phase60_authority,
        _ledger(),
        13,
        input_tokens=20_001,
        output_tokens=100,
    )
    step = phase60.next_round3_schema_phase60_step_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        ledger,
        phase60_authority.sources,
    )
    assert step.kind == "terminal_budget"
    assert step.terminal_error_code == "usage_limit_exceeded"
    assert step.new_provider_calls_reserved == 1
    assert step.next_phase60_call_ordinal is None
    with pytest.raises(phase60.PortfolioS1FeedbackError, match="usage ceiling"):
        phase60.require_round3_schema_phase60_pre_reservation_budget_v1(ledger)
    run = phase60.build_round3_schema_phase60_run_v1(
        phase60_authority.prefix,
        phase60_authority.governance,
        phase60_authority.approval,
        phase60_authority.authorization,
        phase60_authority.control,
        phase60_authority.launch,
        ledger,
        phase60_authority.sources,
    )
    assert run.status == "stopped_budget"
    assert run.terminal_reason == "usage_limit_exceeded"
    assert run.bundle_v11_publishable_from_phase60 is False
    assert run.s1_creator_start_authorized is False
