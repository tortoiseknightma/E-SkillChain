from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import scripts.approve_codex_authoring_v4 as approval_v4
import scripts.build_codex_authoring_approval_package_v4 as builder_v4
import scripts.run_codex_authoring as runner_v2
import scripts.run_codex_authoring_v4 as runner_v4
from skillchain.codex_authoring_v4 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CreateOnceConflictError,
    classified_atomic_create,
)
from skillchain.static_authoring import build_spec_draft_bundle
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    ROOT / "specs" / "authoring" / "authoring-freeze-lock-codex-high-v4.json"
)


def _signed(payload: dict[str, object], field: str) -> bytes:
    return canonical_json_bytes(
        {**payload, field: sha256_bytes(canonical_json_bytes(payload))}
    )


def _guard_and_claim(
    root: Path,
) -> tuple[Path, bytes, Path, bytes, Path]:
    authoring = root / "specs" / "authoring"
    authoring.mkdir(parents=True)
    output = root / "runs" / "formal-authoring" / CODEX_AUTHOR_RUN_ID
    guard_path = authoring / f"{CODEX_AUTHOR_RUN_ID}-terminal-guard.json"
    claim_path = authoring / f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
    guard_payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_conditional_terminal_guard",
        "status": "inactive_until_exact_claim_commits",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "activation_claim_file": claim_path.relative_to(root).as_posix(),
        "canonical_output_directory": output.relative_to(root).as_posix(),
        "canonical_receipt_file": (
            output / "invocation-receipt.json"
        ).relative_to(root).as_posix(),
        "active_status": "authorization_consumed_result_rejected",
        "active_formal_codex_session_eligible": False,
        "active_process_launch_state": "unknown",
        "activation_requires_claim_to_bind_this_guard_sha256": True,
        "canonical_override_requires_full_bundle_validation": True,
        "model_retry_fallback_repair_followup_authorized": False,
        "storage_disaster_guarantee": "not_claimed",
    }
    guard_bytes = _signed(guard_payload, "terminal_guard_payload_sha256")
    request_bytes = b'{"request":"frozen"}\n'
    stdin_bytes = b"frozen stdin\n"
    schema_bytes = b'{"type":"object"}\n'
    claim_payload = {
        "schema_version": 3,
        "status": "consumed_before_codex_process_launch",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "claimant_nonce": "a" * 64,
        "freeze_lock_file_sha256": "1" * 64,
        "freeze_payload_sha256": "2" * 64,
        "owner_approval_file": (
            authoring / f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"
        ).relative_to(root).as_posix(),
        "owner_approval_file_sha256": "3" * 64,
        "owner_approval_payload_sha256": "4" * 64,
        "terminal_guard_file": guard_path.relative_to(root).as_posix(),
        "terminal_guard_file_sha256": sha256_bytes(guard_bytes),
        "v2_retirement_claim_file_sha256": "5" * 64,
        "authoring_input_file_sha256": "6" * 64,
        "canonical_request_file_sha256": sha256_bytes(request_bytes),
        "stdin_request_file_sha256": sha256_bytes(stdin_bytes),
        "output_schema_file_sha256": sha256_bytes(schema_bytes),
        "materialized_output_schema_sha256": sha256_bytes(schema_bytes),
        "runtime_lock_file_sha256": "7" * 64,
        "runtime_payload_sha256": "8" * 64,
        "source_manifest_file_sha256": "9" * 64,
        "binary_evidence_file_sha256": "a" * 64,
        "prompt_isolation_evidence_file_sha256": "b" * 64,
        "environment_policy_sha256": "c" * 64,
        "environment_instance_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "max_codex_exec_sessions": 1,
        "retry_fallback_repair_followup_policy": "forbidden",
        "required_output_directory": output.relative_to(root).as_posix(),
        "required_receipt_file": (
            output / "invocation-receipt.json"
        ).relative_to(root).as_posix(),
    }
    claim_bytes = _signed(claim_payload, "claim_payload_sha256")
    return guard_path, guard_bytes, claim_path, claim_bytes, output


def test_classified_create_recovers_link_success_then_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "attempt-claim.json"
    original_unlink = Path.unlink

    def injected_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{destination.name}."):
            raise PermissionError("injected post-link cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", injected_unlink)
    content = b'{"claimant_nonce":"unique-a"}\n'
    result = classified_atomic_create(
        destination,
        content,
        label="fault-injected claim",
    )
    assert destination.read_bytes() == content
    assert result.committed is True
    assert result.cleanup_error is not None
    assert runner_v4.claim_commit_allows_process_launch(result) is False


def test_classified_create_recovers_baseexception_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "attempt-claim.json"
    original_link = os.link

    def link_then_interrupt(source: object, target: object) -> None:
        original_link(source, target)
        raise KeyboardInterrupt("injected after link commit")

    monkeypatch.setattr(os, "link", link_then_interrupt)
    content = b'{"claimant_nonce":"unique-b"}\n'
    result = classified_atomic_create(
        destination,
        content,
        label="fault-injected claim",
    )
    assert result.committed is True
    assert result.recovered_after_exception is True
    assert destination.read_bytes() == content
    assert runner_v4.claim_commit_allows_process_launch(result) is False


def test_classified_create_link_failure_does_not_consume_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "attempt-claim.json"

    def fail_link(source: object, target: object) -> None:
        raise OSError("injected before commit")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected before commit"):
        classified_atomic_create(
            destination,
            b'{"claimant_nonce":"unique-c"}\n',
            label="fault-injected claim",
        )
    assert not os.path.lexists(destination)


def test_two_nonce_claimants_allow_only_one_launch(tmp_path: Path) -> None:
    destination = tmp_path / "attempt-claim.json"
    launches = 0
    launch_lock = threading.Lock()

    def compete(nonce: str) -> bool:
        nonlocal launches
        try:
            result = classified_atomic_create(
                destination,
                canonical_json_bytes({"claimant_nonce": nonce}),
                label="concurrent claim",
            )
        except CreateOnceConflictError:
            return False
        if runner_v4.claim_commit_allows_process_launch(result):
            with launch_lock:
                launches += 1
            return True
        return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = list(pool.map(compete, ("a" * 64, "b" * 64)))
    assert winners.count(True) == 1
    assert launches == 1
    assert json.loads(destination.read_text(encoding="utf-8"))[
        "claimant_nonce"
    ] in {"a" * 64, "b" * 64}


class _FakeProcess:
    def __init__(self, *, wait_fails: bool = False) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.killed = False
        self.waited = False
        self.wait_fails = wait_fails

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1

    def wait(self, timeout: int | None = None) -> int:
        self.waited = True
        if self.wait_fails:
            raise TimeoutError("injected wait failure")
        self.returncode = 1 if self.returncode is None else self.returncode
        return self.returncode


class _SetRaisesEvent:
    def set(self) -> None:
        raise KeyboardInterrupt("injected immediately after Popen")


def test_supervisor_reaps_process_when_spawn_marker_raises_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(runner_v4.subprocess, "Popen", lambda *a, **k: process)
    result = runner_v4._run_capped_process_v4(
        ["fake-codex"],
        cwd=tmp_path,
        environment={},
        stdin=b"request",
        timeout_seconds=1,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=_SetRaisesEvent(),  # type: ignore[arg-type]
    )
    assert result.process_created is True
    assert result.process_reaped is True
    assert process.killed is True
    assert process.waited is True
    assert "KeyboardInterrupt" in (result.supervision_error or "")


def test_supervisor_reaps_process_when_thread_construction_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(runner_v4.subprocess, "Popen", lambda *a, **k: process)

    def interrupted_thread(*args: object, **kwargs: object) -> None:
        raise MemoryError("injected thread construction failure")

    monkeypatch.setattr(runner_v4.threading, "Thread", interrupted_thread)
    result = runner_v4._run_capped_process_v4(
        ["fake-codex"],
        cwd=tmp_path,
        environment={},
        stdin=b"request",
        timeout_seconds=1,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=threading.Event(),
    )
    assert result.process_created is True
    assert result.process_reaped is True
    assert process.killed is True
    assert "MemoryError" in (result.supervision_error or "")


def test_unreaped_process_can_never_form_a_canonical_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(wait_fails=True)
    monkeypatch.setattr(runner_v4.subprocess, "Popen", lambda *a, **k: process)
    result = runner_v4._run_capped_process_v4(
        ["fake-codex"],
        cwd=tmp_path,
        environment={},
        stdin=b"request",
        timeout_seconds=1,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=threading.Event(),
    )
    assert result.process_created is True
    assert result.process_reaped is False
    assert (
        runner_v4.terminal_process_state_is_stable(
            {
                "process_launch_attempted": True,
                "process_launched": result.process_created,
                "process_reaped": result.process_reaped,
                "pipe_threads_terminated": result.pipe_threads_terminated,
                "process_result_committed": True,
            }
        )
        is False
    )


def test_partial_process_result_copy_can_never_form_canonical_terminal_state() -> None:
    assert (
        runner_v4.terminal_process_state_is_stable(
            {
                "process_launch_attempted": True,
                "process_launched": True,
                "process_reaped": True,
                "pipe_threads_terminated": True,
                "process_result_committed": False,
            }
        )
        is False
    )


def test_guard_is_inactive_before_claim_and_active_after_exact_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    monkeypatch.setattr(runner_v4, "ROOT", tmp_path)
    arguments = {
        "guard_path": guard_path,
        "guard_file_sha256": sha256_bytes(guard_bytes),
        "claim_path": claim_path,
        "output_dir": output,
    }
    assert (
        runner_v4.resolve_terminal_state(**arguments)
        == "terminal_guard_inactive_no_claim"
    )
    claim_path.write_bytes(claim_bytes)
    claim = json.loads(claim_bytes)
    monkeypatch.setattr(
        runner_v4,
        "_load_terminal_validation_context",
        lambda **kwargs: (claim, {}, {}, object(), object()),
    )
    assert (
        runner_v4.resolve_terminal_state(**arguments)
        == "terminal_guard_active_authorization_consumed_result_rejected"
    )


def _clean_events(message: str = "{}") -> bytes:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "reasoning-1", "type": "reasoning", "text": ""},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": message,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
            },
        },
    ]
    return b"".join(
        json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
        for event in events
    )


def _write_valid_bundle(
    *,
    root: Path,
    output: Path,
    guard_path: Path,
    guard_bytes: bytes,
    claim_bytes: bytes,
    event_bytes: bytes | None = None,
    raw_bytes: bytes = b"{}",
    draft_bytes: bytes = b'{"compiled":"draft"}\n',
    request_bytes: bytes = b'{"request":"frozen"}\n',
    stdin_bytes: bytes = b"frozen stdin\n",
    schema_bytes: bytes = b'{"type":"object"}\n',
) -> dict[str, bytes]:
    claim = json.loads(claim_bytes)
    if event_bytes is None:
        event_bytes = _clean_events(raw_bytes.decode("utf-8").removesuffix("\n"))
    audit = runner_v2._event_audit(event_bytes)
    evidence = {
        "schema_version": 4,
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "status": "codex_session_completed_draft_ready_for_review",
        "formal_codex_session_eligible": True,
        "authorization_consumed": True,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "repository_retry_performed": False,
        "followup_performed": False,
        "repair_performed": False,
        "fallback_performed": False,
        "process_launch_attempted": True,
        "process_launched": True,
        "process_reaped": True,
        "pipe_threads_terminated": True,
        "process_result_committed": True,
        "exit_code": 0,
        "timed_out": False,
        "stdout_limit_exceeded": False,
        "stderr_limit_exceeded": False,
        "stdin_completed": True,
        "process_supervision_error": None,
        "thread_id": audit.thread_id,
        "input_tokens": audit.input_tokens,
        "cached_input_tokens": audit.cached_input_tokens,
        "output_tokens": audit.output_tokens,
        "event_types": list(audit.event_types),
        "visible_tool_activity": audit.visible_tool_activity,
        "visible_agent_message_count": len(audit.agent_messages),
        "disallowed_item_types": list(audit.disallowed_item_types),
        "attempt_claim_file_sha256": sha256_bytes(claim_bytes),
        "attempt_claim_payload_sha256": claim["claim_payload_sha256"],
        "terminal_guard_file": guard_path.relative_to(root).as_posix(),
        "terminal_guard_file_sha256": sha256_bytes(guard_bytes),
        "freeze_lock_file_sha256": claim["freeze_lock_file_sha256"],
        "freeze_payload_sha256": claim["freeze_payload_sha256"],
        "owner_approval_file_sha256": claim["owner_approval_file_sha256"],
        "owner_approval_payload_sha256": claim[
            "owner_approval_payload_sha256"
        ],
        "authoring_input_file_sha256": claim["authoring_input_file_sha256"],
        "canonical_request_file_sha256": sha256_bytes(request_bytes),
        "stdin_request_file_sha256": sha256_bytes(stdin_bytes),
        "output_schema_file_sha256": sha256_bytes(schema_bytes),
        "runtime_lock_file_sha256": claim["runtime_lock_file_sha256"],
        "runtime_payload_sha256": claim["runtime_payload_sha256"],
        "source_manifest_file_sha256": claim["source_manifest_file_sha256"],
        "binary_evidence_file_sha256": claim[
            "binary_evidence_file_sha256"
        ],
        "prompt_isolation_evidence_file_sha256": claim[
            "prompt_isolation_evidence_file_sha256"
        ],
        "environment_policy_sha256": claim["environment_policy_sha256"],
        "environment_instance_sha256": claim["environment_instance_sha256"],
        "command_sha256": claim["command_sha256"],
        "event_log_sha256": sha256_bytes(event_bytes),
        "stderr_sha256": sha256_bytes(b""),
        "raw_final_sha256": sha256_bytes(raw_bytes),
        "pre_review_draft_sha256": sha256_bytes(draft_bytes),
    }
    files = {
        "codex-events.jsonl": event_bytes,
        "codex-stderr.bin": b"",
        "authoring-request.json": request_bytes,
        "authoring-stdin.txt": stdin_bytes,
        "authoring-output-schema.json": schema_bytes,
        "session-evidence.json": canonical_json_bytes(evidence),
        "author-content.raw.json": raw_bytes,
        "pre-review-draft.json": draft_bytes,
    }
    receipt = {
        **evidence,
        "artifact_kind": "codex_authoring_canonical_terminal_receipt",
        "canonical_bundle_complete": True,
        "canonical_bundle_files": {
            name: sha256_bytes(content) for name, content in sorted(files.items())
        },
        "terminal_guard_overridden_by_this_valid_bundle": True,
    }
    files["invocation-receipt.json"] = _signed(
        receipt,
        "receipt_payload_sha256",
    )
    output.mkdir(parents=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    return files


def _resign_bundle(output: Path, evidence: dict[str, object]) -> None:
    evidence_bytes = canonical_json_bytes(evidence)
    (output / "session-evidence.json").write_bytes(evidence_bytes)
    files = {
        child.name: child.read_bytes()
        for child in output.iterdir()
        if child.name != "invocation-receipt.json"
    }
    receipt = {
        **evidence,
        "artifact_kind": "codex_authoring_canonical_terminal_receipt",
        "canonical_bundle_complete": True,
        "canonical_bundle_files": {
            name: sha256_bytes(content) for name, content in sorted(files.items())
        },
        "terminal_guard_overridden_by_this_valid_bundle": True,
    }
    (output / "invocation-receipt.json").write_bytes(
        _signed(receipt, "receipt_payload_sha256")
    )


def _patch_valid_terminal_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    claim_bytes: bytes,
    draft_bytes: bytes,
) -> None:
    claim = json.loads(claim_bytes)
    monkeypatch.setattr(runner_v4, "ROOT", root)
    monkeypatch.setattr(
        runner_v4,
        "_load_terminal_validation_context",
        lambda **kwargs: (claim, {}, {}, object(), object()),
    )

    class _Compiled:
        def canonical_bytes(self) -> bytes:
            return draft_bytes

    monkeypatch.setattr(
        runner_v4,
        "normalize_codex_authoring_output",
        lambda *args, **kwargs: _Compiled(),
    )


def test_valid_canonical_bundle_overrides_guard_and_tamper_reactivates_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    files = _write_valid_bundle(
        root=tmp_path,
        output=output,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )
    arguments = {
        "guard_path": guard_path,
        "guard_file_sha256": sha256_bytes(guard_bytes),
        "claim_path": claim_path,
        "output_dir": output,
    }
    assert (
        runner_v4.resolve_terminal_state(**arguments)
        == "canonical_terminal_receipt_valid"
    )
    (output / "authoring-request.json").write_bytes(b"tampered\n")
    assert (
        runner_v4.resolve_terminal_state(**arguments)
        == "terminal_guard_active_authorization_consumed_result_rejected"
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "receipt-self-hash"))
def test_canonical_override_rejects_incomplete_extra_or_rewritten_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    files = _write_valid_bundle(
        root=tmp_path,
        output=output,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )
    if mutation == "missing":
        (output / "authoring-stdin.txt").unlink()
        evidence = json.loads(
            (output / "session-evidence.json").read_text(encoding="utf-8")
        )
        _resign_bundle(output, evidence)
    elif mutation == "extra":
        (output / "unexpected.bin").write_bytes(b"unexpected")
        evidence = json.loads(
            (output / "session-evidence.json").read_text(encoding="utf-8")
        )
        _resign_bundle(output, evidence)
    else:
        receipt = bytearray((output / "invocation-receipt.json").read_bytes())
        receipt[-2] = ord(" ")
        (output / "invocation-receipt.json").write_bytes(bytes(receipt))
    assert (
        runner_v4.resolve_terminal_state(
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
            claim_path=claim_path,
            output_dir=output,
        )
        == "terminal_guard_active_authorization_consumed_result_rejected"
    )


def test_canonical_override_rejects_semantically_invalid_event_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    files = _write_valid_bundle(
        root=tmp_path,
        output=output,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )
    invalid_events = b'{"type":"thread.started","thread_id":"thread-1"}\n'
    (output / "codex-events.jsonl").write_bytes(invalid_events)
    audit = runner_v2._event_audit(invalid_events)
    evidence = json.loads(
        (output / "session-evidence.json").read_text(encoding="utf-8")
    )
    evidence.update(
        {
            "event_log_sha256": sha256_bytes(invalid_events),
            "thread_id": audit.thread_id,
            "input_tokens": audit.input_tokens,
            "cached_input_tokens": audit.cached_input_tokens,
            "output_tokens": audit.output_tokens,
            "event_types": list(audit.event_types),
            "visible_tool_activity": audit.visible_tool_activity,
            "visible_agent_message_count": len(audit.agent_messages),
            "disallowed_item_types": list(audit.disallowed_item_types),
        }
    )
    _resign_bundle(output, evidence)
    assert (
        runner_v4.resolve_terminal_state(
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
            claim_path=claim_path,
            output_dir=output,
        )
        == "terminal_guard_active_authorization_consumed_result_rejected"
    )


def test_failure_status_must_match_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    files = _write_valid_bundle(
        root=tmp_path,
        output=output,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )
    (output / "pre-review-draft.json").unlink()
    evidence = json.loads(
        (output / "session-evidence.json").read_text(encoding="utf-8")
    )
    evidence["status"] = "codex_process_timed_out"
    evidence["formal_codex_session_eligible"] = False
    evidence["pre_review_draft_sha256"] = None
    evidence["timed_out"] = False
    _resign_bundle(output, evidence)
    assert (
        runner_v4.resolve_terminal_state(
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
            claim_path=claim_path,
            output_dir=output,
        )
        == "terminal_guard_active_authorization_consumed_result_rejected"
    )


def test_foreign_or_malformed_claim_never_activates_a_trusted_terminal_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, _claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(
        _signed(
            {
                "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
                "run_id": CODEX_AUTHOR_RUN_ID,
                "claimant_nonce": "f" * 64,
            },
            "claim_payload_sha256",
        )
    )
    monkeypatch.setattr(runner_v4, "ROOT", tmp_path)
    assert (
        runner_v4.resolve_terminal_state(
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
            claim_path=claim_path,
            output_dir=output,
        )
        == "terminal_state_unverifiable_authorization_not_reusable"
    )


def test_rename_postcommit_exception_is_classified_by_exact_bundle_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    staging = output.parent / ".postcommit-staging"
    files = _write_valid_bundle(
        root=tmp_path,
        output=staging,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )

    def commit_then_raise(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, destination)
        raise PermissionError("injected cleanup failure after rename")

    monkeypatch.setattr(
        runner_v4,
        "atomic_publish_new_directory",
        commit_then_raise,
    )
    runner_v4._publish_and_classify_bundle(
        staging,
        output,
        files,
        claim_path=claim_path,
        guard_path=guard_path,
        guard_file_sha256=sha256_bytes(guard_bytes),
    )
    assert output.is_dir()
    assert not staging.exists()


def test_publish_precommit_failure_or_third_party_output_never_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_path, guard_bytes, claim_path, claim_bytes, output = _guard_and_claim(
        tmp_path
    )
    guard_path.write_bytes(guard_bytes)
    claim_path.write_bytes(claim_bytes)
    staging = output.parent / ".precommit-staging"
    files = _write_valid_bundle(
        root=tmp_path,
        output=staging,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
    )
    _patch_valid_terminal_context(
        monkeypatch,
        root=tmp_path,
        claim_bytes=claim_bytes,
        draft_bytes=files["pre-review-draft.json"],
    )

    def fail_before_commit(source: Path, destination: Path) -> None:
        raise PermissionError("injected failure before rename")

    monkeypatch.setattr(
        runner_v4,
        "atomic_publish_new_directory",
        fail_before_commit,
    )
    with pytest.raises(PermissionError, match="before rename"):
        runner_v4._publish_and_classify_bundle(
            staging,
            output,
            files,
            claim_path=claim_path,
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
        )
    assert not output.exists()
    assert not staging.exists()

    staging = output.parent / ".third-party-staging"
    staging.mkdir(parents=True)
    output.mkdir(parents=True)
    (output / "third-party.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(PermissionError, match="before rename"):
        runner_v4._publish_and_classify_bundle(
            staging,
            output,
            files,
            claim_path=claim_path,
            guard_path=guard_path,
            guard_file_sha256=sha256_bytes(guard_bytes),
        )
    assert (output / "third-party.txt").read_text(encoding="utf-8") == "occupied"
    assert not staging.exists()


def test_approval_incident_validation_fails_before_retirement_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoring = tmp_path / "specs" / "authoring"
    authoring.mkdir(parents=True)
    v2_path = authoring / "v2-incident.json"
    v3_path = authoring / "v3-incident.json"
    v2_bytes = _signed(
        {
            "claim_created": False,
            "codex_exec_process_spawned": False,
            "inference_requested": False,
            "authorization_consumed_at_incident": False,
        },
        "incident_payload_sha256",
    )
    v3_bytes = _signed(
        {
            "status": "superseded_before_owner_approval_and_inference",
            "owner_approval_created": False,
            "claim_created": False,
            "codex_exec_process_spawned": False,
            "inference_requested": False,
        },
        "incident_payload_sha256",
    )
    v2_path.write_bytes(v2_bytes)
    v3_path.write_bytes(v3_bytes)
    freeze = {
        "bindings": {
            "v2_preflight_incident": {
                "file": v2_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_bytes(v2_bytes),
            },
            "v3_preapproval_audit_incident": {
                "file": v3_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_bytes(v3_bytes),
            },
        }
    }
    monkeypatch.setattr(approval_v4, "ROOT", tmp_path)
    monkeypatch.setattr(approval_v4, "V2_PREFLIGHT_INCIDENT_PATH", v2_path)
    monkeypatch.setattr(approval_v4, "V3_AUDIT_INCIDENT_PATH", v3_path)
    approval_v4._validate_incidents_before_retirement(freeze)
    v2_path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="file SHA-256 drifted"):
        approval_v4._validate_incidents_before_retirement(freeze)


def test_approval_create_recovers_exact_postlink_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "owner-approval.json"
    content = b'{"approval":"exact"}\n'

    def commit_then_raise(path: Path, value: bytes) -> None:
        path.write_bytes(value)
        raise PermissionError("injected cleanup failure after commit")

    monkeypatch.setattr(approval_v4, "atomic_create_file", commit_then_raise)
    warning = approval_v4._create_or_verify(
        destination,
        content,
        "fault-injected owner approval",
    )
    assert destination.read_bytes() == content
    assert warning is not None
    assert "PermissionError" in warning


def test_historical_freeze_check_covers_unreused_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder_v4, "ROOT", tmp_path)
    bound = tmp_path / "specs" / "history.json"
    bound.parent.mkdir(parents=True)
    bound.write_bytes(b'{"history":"exact"}\n')
    freeze_path = tmp_path / "specs" / "freeze.json"
    payload = {
        "bindings": {
            "unreused_history": {
                "file": bound.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_bytes(bound.read_bytes()),
            }
        }
    }
    freeze_bytes = _signed(payload, "freeze_payload_sha256")
    freeze_path.write_bytes(freeze_bytes)
    freeze = json.loads(freeze_bytes)
    builder_v4._verify_historical_freeze(
        freeze_path,
        expected_file_sha256=sha256_bytes(freeze_bytes),
        expected_payload_sha256=freeze["freeze_payload_sha256"],
        label="test history",
    )
    bound.write_bytes(b'{"history":"drifted"}\n')
    with pytest.raises(ValueError, match="bound history drifted"):
        builder_v4._verify_historical_freeze(
            freeze_path,
            expected_file_sha256=sha256_bytes(freeze_bytes),
            expected_payload_sha256=freeze["freeze_payload_sha256"],
            label="test history",
        )


def _materialize_terminal_context_candidate(
    destination_root: Path,
) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    """Copy the frozen candidate and its live runtime closure into a temp root."""

    generated: dict[str, bytes] = {}
    if FREEZE_PATH.is_file():
        freeze_bytes = FREEZE_PATH.read_bytes()
    else:
        expected = builder_v4._expected_files()
        generated = {
            path.relative_to(ROOT).as_posix(): content
            for path, content in expected.items()
        }
        freeze_bytes = generated[FREEZE_PATH.relative_to(ROOT).as_posix()]
    freeze = json.loads(freeze_bytes)

    def candidate_bytes(relative: str) -> bytes:
        content = generated.get(relative)
        if content is not None:
            return content
        return (ROOT / relative).read_bytes()

    copied: dict[str, bytes] = {}

    def materialize(relative: str) -> bytes:
        content = candidate_bytes(relative)
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied[relative] = content
        return content

    freeze_relative = FREEZE_PATH.relative_to(ROOT).as_posix()
    freeze_target = destination_root / freeze_relative
    freeze_target.parent.mkdir(parents=True, exist_ok=True)
    freeze_target.write_bytes(freeze_bytes)
    copied[freeze_relative] = freeze_bytes

    terminal_bindings = (
        "authoring_input",
        "semantic_source",
        "canonical_request",
        "stdin_request",
        "output_schema",
        "runtime_lock",
        "source_manifest",
        "binary_binding_evidence",
        "prompt_isolation_evidence",
        "v2_preflight_incident",
        "v3_preapproval_audit_incident",
    )
    bindings = freeze["bindings"]
    for name in terminal_bindings:
        materialize(bindings[name]["file"])

    source_manifest = json.loads(
        copied[bindings["source_manifest"]["file"]]
    )
    for section in ("trusted_sources", "repository_locks"):
        for item in source_manifest[section]:
            materialize(item["file"])
    return freeze, freeze_bytes, copied


def test_real_authority_chain_and_trusted_compiler_accept_legal_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real guard/claim/freeze/approval chain and compiler."""

    freeze, freeze_bytes, copied = _materialize_terminal_context_candidate(
        tmp_path
    )
    bindings = freeze["bindings"]
    paths = freeze["authorization"]["paths"]

    def bound_bytes(name: str) -> bytes:
        return copied[bindings[name]["file"]]

    def bound_sha(name: str) -> str:
        return bindings[name]["file_sha256"]

    guard_path = tmp_path / paths["terminal_guard_file"]
    guard_bytes = builder_v4.planned_terminal_guard_bytes()
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    guard_path.write_bytes(guard_bytes)
    guard_sha = sha256_bytes(guard_bytes)
    guard = json.loads(guard_bytes)

    retirement_bytes = builder_v4.planned_v2_retirement_claim_bytes()
    retirement = json.loads(retirement_bytes)
    retirement_relative = freeze["prior_authorization_retirement"][
        "retirement_claim_file"
    ]
    retirement_path = tmp_path / retirement_relative
    retirement_path.parent.mkdir(parents=True, exist_ok=True)
    retirement_path.write_bytes(retirement_bytes)
    retirement_sha = sha256_bytes(retirement_bytes)

    v2_incident = json.loads(bound_bytes("v2_preflight_incident"))
    v3_incident = json.loads(bound_bytes("v3_preapproval_audit_incident"))
    freeze_sha = sha256_bytes(freeze_bytes)
    freeze_relative = FREEZE_PATH.relative_to(ROOT).as_posix()
    approval_payload = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": freeze_relative,
        "freeze_lock_file_sha256": freeze_sha,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "accepted_evidence_tier": "platform-mediated_non-provider-attested",
        "one_means_one_codex_exec_session": True,
        "backend_attempt_count_unobservable": True,
        "input_isolation_not_mechanically_proven": True,
        "invalid_output_permanently_consumes_authorization": True,
        "s1_must_use_same_surface_model_effort_and_budget": True,
        "deterministic_environment_v3_accepted": True,
        "absolute_binary_path_binding_accepted": True,
        "transactional_terminal_guard_v4_accepted": True,
        "v2_authorization_retired_without_inference": True,
        "v2_retirement_claim_file": retirement_relative,
        "v2_retirement_claim_file_sha256": retirement_sha,
        "v2_retirement_claim_payload_sha256": retirement[
            "retirement_payload_sha256"
        ],
        "v2_preflight_incident_file_sha256": bound_sha(
            "v2_preflight_incident"
        ),
        "v2_preflight_incident_payload_sha256": v2_incident[
            "incident_payload_sha256"
        ],
        "v3_preapproval_audit_incident_file_sha256": bound_sha(
            "v3_preapproval_audit_incident"
        ),
        "v3_preapproval_audit_incident_payload_sha256": v3_incident[
            "incident_payload_sha256"
        ],
        "terminal_guard_file": paths["terminal_guard_file"],
        "terminal_guard_file_sha256": guard_sha,
        "terminal_guard_payload_sha256": guard[
            "terminal_guard_payload_sha256"
        ],
    }
    approval_bytes = _signed(approval_payload, "approval_payload_sha256")
    approval = json.loads(approval_bytes)
    approval_path = tmp_path / paths["approval_record_file"]
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    approval_path.write_bytes(approval_bytes)

    runtime = json.loads(bound_bytes("runtime_lock"))
    output = tmp_path / paths["output_directory"]
    claim_path = tmp_path / paths["claim_file"]
    claim_payload = {
        "schema_version": 3,
        "status": "consumed_before_codex_process_launch",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "claimant_nonce": "a" * 64,
        "freeze_lock_file_sha256": freeze_sha,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "owner_approval_file": paths["approval_record_file"],
        "owner_approval_file_sha256": sha256_bytes(approval_bytes),
        "owner_approval_payload_sha256": approval["approval_payload_sha256"],
        "terminal_guard_file": paths["terminal_guard_file"],
        "terminal_guard_file_sha256": guard_sha,
        "v2_retirement_claim_file_sha256": retirement_sha,
        "authoring_input_file_sha256": bound_sha("authoring_input"),
        "canonical_request_file_sha256": bound_sha("canonical_request"),
        "stdin_request_file_sha256": bound_sha("stdin_request"),
        "output_schema_file_sha256": bound_sha("output_schema"),
        "materialized_output_schema_sha256": bound_sha("output_schema"),
        "runtime_lock_file_sha256": bound_sha("runtime_lock"),
        "runtime_payload_sha256": runtime["runtime_payload_sha256"],
        "source_manifest_file_sha256": bound_sha("source_manifest"),
        "binary_evidence_file_sha256": bound_sha("binary_binding_evidence"),
        "prompt_isolation_evidence_file_sha256": bound_sha(
            "prompt_isolation_evidence"
        ),
        "environment_policy_sha256": sha256_bytes(
            canonical_json_bytes(runtime["environment_policy"])
        ),
        "environment_instance_sha256": "f" * 64,
        "command_sha256": sha256_bytes(
            canonical_json_bytes(list(runner_v4.CODEX_COMMAND_SHAPE))
        ),
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "max_codex_exec_sessions": 1,
        "retry_fallback_repair_followup_policy": "forbidden",
        "required_output_directory": paths["output_directory"],
        "required_receipt_file": paths["receipt_file"],
    }
    claim_bytes = _signed(claim_payload, "claim_payload_sha256")
    claim_path.write_bytes(claim_bytes)

    authoring_input = runner_v4.load_codex_authoring_input(
        tmp_path / bindings["authoring_input"]["file"],
        expected_file_sha256=bound_sha("authoring_input"),
    )
    semantic_source = runner_v4.load_authoring_packet(
        tmp_path / bindings["semantic_source"]["file"],
        expected_file_sha256=bound_sha("semantic_source"),
    )
    spec_bundle = build_spec_draft_bundle(semantic_source)
    raw_bytes = canonical_json_bytes(
        {
            "schema_version": 2,
            "drafts": [
                {
                    "capability_id": draft.capability_id,
                    "objective": draft.objective,
                    "steps": [
                        {
                            "instruction": step.instruction,
                            "tool_name": step.tool_name,
                            "success_rule_ids": list(step.success_rule_ids),
                        }
                        for step in draft.steps
                    ],
                    "fallback_instruction": draft.fallback_instruction,
                    "citation_source_ids": list(draft.citation_source_ids),
                }
                for draft in spec_bundle.drafts
            ],
        }
    )
    compiled = runner_v4.normalize_codex_authoring_output(
        raw_bytes,
        authoring_input=authoring_input,
        semantic_source=semantic_source,
    )
    event_bytes = _clean_events(raw_bytes.decode("utf-8").removesuffix("\n"))
    _write_valid_bundle(
        root=tmp_path,
        output=output,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
        claim_bytes=claim_bytes,
        event_bytes=event_bytes,
        raw_bytes=raw_bytes,
        draft_bytes=compiled.canonical_bytes(),
        request_bytes=bound_bytes("canonical_request"),
        stdin_bytes=bound_bytes("stdin_request"),
        schema_bytes=bound_bytes("output_schema"),
    )

    monkeypatch.setattr(runner_v4, "ROOT", tmp_path)
    monkeypatch.setattr(runner_v4, "FREEZE_PATH", tmp_path / freeze_relative)
    assert (
        runner_v4.resolve_terminal_state(
            guard_path=guard_path,
            guard_file_sha256=guard_sha,
            claim_path=claim_path,
            output_dir=output,
        )
        == "canonical_terminal_receipt_valid"
    )


def _skip_if_v4_authority_phase_started() -> None:
    occupied = [
        path
        for path in (
            builder_v4.APPROVAL_PATH,
            builder_v4.TERMINAL_GUARD_PATH,
            builder_v4.CLAIM_PATH,
            builder_v4.OUTPUT_DIR,
            builder_v4.RECEIPT_PATH,
            builder_v4.V2_RETIREMENT_CLAIM_PATH,
        )
        if os.path.lexists(path)
    ]
    if occupied:
        pytest.skip(
            "preapproval-only candidate gate; v4 authority phase has started"
        )


def test_v4_expected_candidate_can_be_built_without_authority_side_effects() -> None:
    _skip_if_v4_authority_phase_started()
    expected = builder_v4._expected_files()
    assert builder_v4.FREEZE_PATH in expected
    freeze = json.loads(expected[builder_v4.FREEZE_PATH])
    assert freeze["candidate_id"] == CODEX_AUTHOR_CANDIDATE_ID
    assert freeze["invocation_authorized"] is False
    assert freeze["terminal_guard"]["expected_file_sha256"] == sha256_bytes(
        builder_v4.planned_terminal_guard_bytes()
    )
    assert not any(
        os.path.lexists(path)
        for path in (
            builder_v4.APPROVAL_PATH,
            builder_v4.TERMINAL_GUARD_PATH,
            builder_v4.CLAIM_PATH,
            builder_v4.OUTPUT_DIR,
            builder_v4.RECEIPT_PATH,
            builder_v4.V2_RETIREMENT_CLAIM_PATH,
        )
    )


def test_v4_candidate_package_is_exact_and_authority_paths_are_absent() -> None:
    _skip_if_v4_authority_phase_started()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_codex_authoring_approval_package_v4.py",
            "--check-candidate",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["candidate_id"] == CODEX_AUTHOR_CANDIDATE_ID
    assert report["run_id"] == CODEX_AUTHOR_RUN_ID
    assert report["invocation_authorized"] is False
    assert report["terminal_guard_created"] is False
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    paths = freeze["authorization"]["paths"]
    assert not any(
        os.path.lexists(ROOT / paths[name])
        for name in (
            "approval_record_file",
            "terminal_guard_file",
            "claim_file",
            "output_directory",
            "receipt_file",
        )
    )
