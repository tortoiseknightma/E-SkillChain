"""Consume one owner-approved Codex CLI static-authoring session.

The runner is intentionally separate from the API-provider authoring runner.
It creates the attempt claim before process launch and never retries, repairs,
falls back, resumes, or follows up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any

import skillchain.codex_authoring as codex_authoring_module
from skillchain.codex_authoring import (
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CODEX_ENV_ALLOWLIST,
    CodexAuthoringContractError,
    build_codex_runtime_dependency_snapshot,
    load_codex_authoring_input,
    normalize_codex_authoring_output,
    parse_codex_authoring_request,
    render_codex_authoring_stdin,
    validate_codex_cli_output_schema,
)
from skillchain.static_authoring import load_authoring_packet
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_MODULE_PATH = Path(codex_authoring_module.__file__).resolve(strict=True)
FREEZE_PATH = (
    ROOT / "specs" / "authoring" / "authoring-freeze-lock-codex-high-v2.json"
)

_SAFE_ENV_NAMES = frozenset(CODEX_ENV_ALLOWLIST)
_ALLOWED_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
        "turn.failed",
        "error",
    }
)
_ALLOWED_NON_TOOL_ITEM_TYPES = frozenset({"agent_message", "reasoning"})


def _sha_file(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CodexAuthoringContractError(f"hash target is not a regular file: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after_read = os.fstat(handle.fileno())
    after = path.lstat()
    snapshots = (before, opened, after_read, after)
    if any(
        (
            item.st_size,
            item.st_mtime_ns,
            getattr(item, "st_ino", 0),
        )
        != (
            before.st_size,
            before.st_mtime_ns,
            getattr(before, "st_ino", 0),
        )
        for item in snapshots[1:]
    ):
        raise CodexAuthoringContractError(f"hash target changed during read: {path}")
    return digest


def _load_object(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    try:
        content = read_stable_regular_file(path, label=label)
        if sha256_bytes(content) != expected_sha256:
            raise CodexAuthoringContractError(f"{label} digest mismatch")
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise CodexAuthoringContractError(f"{label} is not canonical JSON") from error
    if not isinstance(raw, dict):
        raise CodexAuthoringContractError(f"{label} must contain an object")
    return raw


def _self_hash(raw: dict[str, Any], field: str, label: str) -> None:
    unsigned = dict(raw)
    observed = unsigned.pop(field, None)
    if observed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise CodexAuthoringContractError(f"{label} self digest mismatch")


def _repo_path(reference: object, label: str, *, must_exist: bool) -> Path:
    if (
        not isinstance(reference, str)
        or not reference
        or reference != reference.strip()
    ):
        raise CodexAuthoringContractError(f"{label} path is invalid")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise CodexAuthoringContractError(f"{label} path escapes the repository")
    resolved = (ROOT / relative).resolve()
    if ROOT.resolve() not in resolved.parents:
        raise CodexAuthoringContractError(f"{label} path escapes the repository")
    if must_exist and not resolved.is_file():
        raise CodexAuthoringContractError(f"{label} file is missing")
    return resolved


def _binding(
    freeze: dict[str, Any], name: str
) -> tuple[Path, str]:
    bindings = freeze.get("bindings")
    item = bindings.get(name) if isinstance(bindings, dict) else None
    if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
        raise CodexAuthoringContractError(f"{name} binding is invalid")
    expected = item.get("file_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CodexAuthoringContractError(f"{name} digest is invalid")
    path = _repo_path(item.get("file"), name, must_exist=True)
    if _sha_file(path) != expected:
        raise CodexAuthoringContractError(f"{name} binding drifted")
    return path, expected


def _runtime_binding(
    runtime: dict[str, Any], name: str
) -> tuple[Path, str]:
    item = runtime.get(name)
    if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
        raise CodexAuthoringContractError(f"runtime {name} binding is invalid")
    expected = item.get("file_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CodexAuthoringContractError(f"runtime {name} digest is invalid")
    path = _repo_path(item.get("file"), f"runtime {name}", must_exist=True)
    if _sha_file(path) != expected:
        raise CodexAuthoringContractError(f"runtime {name} binding drifted")
    return path, expected


def _validate_frozen_source_bindings(
    freeze_bindings: dict[str, tuple[Path, str]],
    runtime: dict[str, Any],
) -> None:
    freeze_runner = freeze_bindings["runner"]
    freeze_contract = freeze_bindings["contract_module"]
    freeze_manifest = freeze_bindings["source_manifest"]
    freeze_prompt_isolation = freeze_bindings["prompt_isolation_evidence"]
    runtime_runner = _runtime_binding(runtime, "runner")
    runtime_contract = _runtime_binding(runtime, "contract_module")
    runtime_manifest = _runtime_binding(runtime, "source_manifest")
    runtime_prompt_isolation = _runtime_binding(
        runtime,
        "prompt_isolation_evidence",
    )
    if (
        freeze_runner != runtime_runner
        or freeze_contract != runtime_contract
        or freeze_manifest != runtime_manifest
        or freeze_prompt_isolation != runtime_prompt_isolation
        or freeze_runner[0] != RUNNER_PATH
        or freeze_contract[0] != CONTRACT_MODULE_PATH
        or CONTRACT_MODULE_PATH
        != (ROOT / "src" / "skillchain" / "codex_authoring.py").resolve(strict=True)
    ):
        raise CodexAuthoringContractError(
            "freeze/runtime source bindings are not the canonical runner and contract"
        )


def _safe_environment(
    runtime: dict[str, Any],
) -> tuple[dict[str, str], str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES
    }
    runtime_policy = runtime.get("environment_policy")
    if not isinstance(runtime_policy, dict):
        raise CodexAuthoringContractError("Codex runtime lacks environment policy")
    commitments = {
        name: sha256_bytes(value.encode("utf-8"))
        for name, value in sorted(environment.items())
    }
    if (
        runtime_policy.get("allowed_names") != list(CODEX_ENV_ALLOWLIST)
        or runtime_policy.get("present_value_sha256") != commitments
        or runtime_policy.get("all_other_environment_variables_removed") is not True
    ):
        raise CodexAuthoringContractError(
            "live Codex environment differs from the frozen value commitments"
        )
    policy = {
        "policy_version": "codex-author-env-allowlist-v1",
        "allowed_names": sorted(_SAFE_ENV_NAMES),
        "present_value_sha256": commitments,
        "credential_values_recorded": False,
    }
    return environment, sha256_bytes(canonical_json_bytes(policy))


def _resolve_codex(runtime: dict[str, Any]) -> Path:
    discovered = shutil.which("codex")
    if not discovered:
        raise CodexAuthoringContractError("Codex CLI is not on PATH")
    path = Path(discovered).resolve(strict=True)
    binary = runtime.get("binary")
    if not isinstance(binary, dict):
        raise CodexAuthoringContractError("Codex runtime lacks binary identity")
    if (
        path.name.casefold() != "codex.exe"
        or path.stat().st_size != binary.get("bytes")
        or _sha_file(path) != binary.get("sha256")
    ):
        raise CodexAuthoringContractError("live Codex CLI differs from runtime lock")
    return path


def _read_bound_bytes(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    if sha256_bytes(content) != expected_sha256:
        raise CodexAuthoringContractError(f"{label} binding drifted")
    return content


def _validate_dependency_snapshot(
    manifest: dict[str, Any],
) -> None:
    _self_hash(
        manifest,
        "source_manifest_payload_sha256",
        "Codex source manifest",
    )
    unsigned = dict(manifest)
    unsigned.pop("source_manifest_payload_sha256", None)
    live = build_codex_runtime_dependency_snapshot(ROOT)
    if canonical_json_bytes(unsigned) != canonical_json_bytes(live):
        raise CodexAuthoringContractError(
            "live Python/source/dependency closure differs from the frozen manifest"
        )


@dataclass(frozen=True)
class CodexEventAudit:
    event_types: tuple[str, ...]
    thread_id: str | None
    thread_started_count: int
    turn_started_count: int
    turn_completed_count: int
    failure_event_count: int
    agent_messages: tuple[str, ...]
    disallowed_item_types: tuple[str, ...]
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None

    @property
    def visible_tool_activity(self) -> bool:
        return bool(self.disallowed_item_types)

    def require_formal_success(self) -> None:
        if (
            self.thread_started_count != 1
            or self.turn_started_count != 1
            or self.turn_completed_count != 1
            or self.failure_event_count
            or len(self.agent_messages) != 1
            or self.disallowed_item_types
            or self.thread_id is None
            or self.input_tokens is None
            or self.cached_input_tokens is None
            or self.output_tokens is None
        ):
            raise CodexAuthoringContractError(
                "Codex JSONL does not describe exactly one clean completed turn"
            )


def _event_audit(content: bytes) -> CodexEventAudit:
    """Parse the frozen CLI JSONL shape with a fail-closed item allowlist."""

    event_types: list[str] = []
    thread_ids: list[str] = []
    thread_started_count = 0
    turn_started_count = 0
    turn_completed_count = 0
    failure_event_count = 0
    agent_messages: list[str] = []
    disallowed_item_types: list[str] = []
    usage: tuple[int, int, int] | None = None
    state = "expect_thread"
    item_states: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise CodexAuthoringContractError(
                f"Codex JSONL contains a blank line at {line_number}"
            )
        try:
            event = parse_strict_json(
                line,
                label=f"Codex JSONL event line {line_number}",
            )
        except ArtifactFormatError as error:
            raise CodexAuthoringContractError(
                f"Codex stdout line {line_number} is not strict JSON"
            ) from error
        if not isinstance(event, dict):
            raise CodexAuthoringContractError("Codex event must contain an object")
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _ALLOWED_EVENT_TYPES:
            raise CodexAuthoringContractError(
                f"Codex emitted an unknown event type: {event_type!r}"
            )
        event_types.append(event_type)
        if event_type == "thread.started":
            if state != "expect_thread":
                raise CodexAuthoringContractError(
                    "Codex thread.started is duplicated or out of order"
                )
            thread_started_count += 1
            thread_id = event.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id.strip():
                raise CodexAuthoringContractError(
                    "Codex thread.started lacks a thread_id"
                )
            thread_ids.append(thread_id)
            state = "expect_turn"
        elif event_type == "turn.started":
            if state != "expect_turn":
                raise CodexAuthoringContractError(
                    "Codex turn.started is duplicated or out of order"
                )
            turn_started_count += 1
            state = "in_turn"
        elif event_type == "turn.completed":
            if state != "in_turn":
                raise CodexAuthoringContractError(
                    "Codex turn.completed is duplicated or out of order"
                )
            turn_completed_count += 1
            raw_usage = event.get("usage")
            if not isinstance(raw_usage, dict):
                raise CodexAuthoringContractError(
                    "Codex turn.completed lacks usage"
                )
            values = tuple(
                raw_usage.get(name)
                for name in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                )
            )
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise CodexAuthoringContractError(
                    "Codex turn.completed usage is invalid"
                )
            usage = values  # type: ignore[assignment]
            state = "terminal"
        elif event_type == "turn.failed":
            if state != "in_turn":
                raise CodexAuthoringContractError(
                    "Codex turn.failed is duplicated or out of order"
                )
            failure_event_count += 1
            state = "terminal"
        elif event_type == "error":
            if state == "terminal":
                raise CodexAuthoringContractError(
                    "Codex error event appeared after the terminal event"
                )
            failure_event_count += 1

        if event_type.startswith("item."):
            if state != "in_turn":
                raise CodexAuthoringContractError(
                    f"Codex {event_type} appeared outside the active turn"
                )
            item = event.get("item")
            if not isinstance(item, dict):
                raise CodexAuthoringContractError(
                    f"Codex {event_type} lacks an item object"
                )
            item_type = item.get("type")
            if not isinstance(item_type, str):
                raise CodexAuthoringContractError("Codex item lacks a type")
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise CodexAuthoringContractError("Codex item lacks an id")
            prior = item_states.get(item_id)
            if event_type == "item.started":
                if prior is not None:
                    raise CodexAuthoringContractError(
                        "Codex item.started reused an item id"
                    )
                item_states[item_id] = (item_type, "started")
            elif event_type == "item.updated":
                if (
                    prior is None
                    or prior[0] != item_type
                    or prior[1] not in {"started", "updated"}
                ):
                    raise CodexAuthoringContractError(
                        "Codex item.updated has no matching live item"
                    )
                item_states[item_id] = (item_type, "updated")
            else:
                if prior is not None and (
                    prior[0] != item_type or prior[1] == "completed"
                ):
                    raise CodexAuthoringContractError(
                        "Codex item.completed conflicts with its item lifecycle"
                    )
                item_states[item_id] = (item_type, "completed")
            if item_type not in _ALLOWED_NON_TOOL_ITEM_TYPES:
                disallowed_item_types.append(item_type)
            if event_type == "item.completed" and item_type == "agent_message":
                message = item.get("text")
                if not isinstance(message, str):
                    raise CodexAuthoringContractError(
                        "completed Codex agent_message lacks text"
                    )
                agent_messages.append(message)
        event_thread_id = event.get("thread_id")
        if event_type != "thread.started" and event_thread_id is not None:
            if (
                not isinstance(event_thread_id, str)
                or not thread_ids
                or event_thread_id != thread_ids[0]
            ):
                raise CodexAuthoringContractError(
                    "Codex event thread_id differs from thread.started"
                )
    if len(set(thread_ids)) > 1:
        raise CodexAuthoringContractError("Codex emitted multiple thread IDs")
    incomplete_items = sorted(
        item_id
        for item_id, (_, item_state) in item_states.items()
        if item_state != "completed"
    )
    if incomplete_items:
        raise CodexAuthoringContractError(
            "Codex event log ended with incomplete items: "
            + ", ".join(incomplete_items)
        )
    return CodexEventAudit(
        event_types=tuple(event_types),
        thread_id=thread_ids[0] if thread_ids else None,
        thread_started_count=thread_started_count,
        turn_started_count=turn_started_count,
        turn_completed_count=turn_completed_count,
        failure_event_count=failure_event_count,
        agent_messages=tuple(agent_messages),
        disallowed_item_types=tuple(disallowed_item_types),
        input_tokens=usage[0] if usage else None,
        cached_input_tokens=usage[1] if usage else None,
        output_tokens=usage[2] if usage else None,
    )


@dataclass(frozen=True)
class CappedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_limit_exceeded: bool
    stderr_limit_exceeded: bool
    stdin_completed: bool
    supervision_error: str | None


def _run_capped_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdin: bytes,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
    spawned: threading.Event,
) -> CappedProcessResult:
    """Run one process with bounded concurrent stdin/stdout/stderr supervision."""

    deadline = time.monotonic() + timeout_seconds
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    stdin_completed = threading.Event()
    supervision_failed = threading.Event()
    supervision_errors: list[str] = []
    supervision_error_lock = threading.Lock()

    def record_supervision_error(stage: str, error: BaseException) -> None:
        with supervision_error_lock:
            supervision_errors.append(
                f"{stage}: {type(error).__name__}: {error}"
            )
        supervision_failed.set()

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        bufsize=0,
    )
    spawned.set()
    if process.stdout is None or process.stderr is None or process.stdin is None:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise CodexAuthoringContractError("Codex process pipes are unavailable")

    def drain(
        stream: Any,
        destination: bytearray,
        limit: int,
        overflow: threading.Event,
        stage: str,
    ) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                remaining = max(0, limit - len(destination))
                destination.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
        except Exception as error:
            record_supervision_error(stage, error)

    def feed_stdin() -> None:
        try:
            view = memoryview(stdin)
            written = 0
            while written < len(view):
                count = process.stdin.write(view[written:])
                if count is None or count <= 0:
                    raise OSError(
                        f"short stdin write stopped after {written} bytes"
                    )
                written += count
            process.stdin.flush()
            if written != len(stdin):
                raise OSError(
                    f"short stdin write: expected {len(stdin)}, wrote {written}"
                )
            stdin_completed.set()
        except BrokenPipeError as error:
            record_supervision_error("stdin", error)
        except Exception as error:
            record_supervision_error("stdin", error)
        finally:
            try:
                process.stdin.close()
            except Exception as error:
                record_supervision_error("stdin_close", error)

    timed_out = False
    returncode: int | None = None
    threads: dict[str, threading.Thread] = {}
    try:
        threads = {
            "stdout": threading.Thread(
                target=drain,
                args=(
                    process.stdout,
                    stdout,
                    stdout_limit,
                    stdout_overflow,
                    "stdout",
                ),
                daemon=True,
            ),
            "stderr": threading.Thread(
                target=drain,
                args=(
                    process.stderr,
                    stderr,
                    stderr_limit,
                    stderr_overflow,
                    "stderr",
                ),
                daemon=True,
            ),
            "stdin": threading.Thread(
                target=feed_stdin,
                daemon=True,
            ),
        }
        for name in ("stdout", "stderr", "stdin"):
            threads[name].start()
        while process.poll() is None:
            if stdout_overflow.is_set() or stderr_overflow.is_set():
                process.kill()
                break
            if supervision_failed.is_set():
                process.kill()
                break
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(0.02)
        returncode = process.wait(timeout=5)
    except Exception as error:
        record_supervision_error("process_supervision", error)
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except Exception as error:
                record_supervision_error("process_kill", error)
        try:
            returncode = process.wait(timeout=5)
        except Exception as error:
            record_supervision_error("process_final_wait", error)
        for thread in threads.values():
            if thread.ident is not None:
                thread.join(timeout=5)
        for name, stage, stream in (
            ("stdin", "stdin_final_close", process.stdin),
            ("stdout", "stdout_close", process.stdout),
            ("stderr", "stderr_close", process.stderr),
        ):
            thread = threads.get(name)
            if thread is not None and thread.is_alive():
                continue
            try:
                stream.close()
            except Exception as error:
                record_supervision_error(stage, error)
    live_threads = [
        name
        for name, thread in threads.items()
        if thread.is_alive()
    ]
    if live_threads:
        with supervision_error_lock:
            supervision_errors.append(
                "pipe threads did not terminate: " + ", ".join(live_threads)
            )
    if returncode is None:
        returncode = process.returncode if process.returncode is not None else -1
    return CappedProcessResult(
        returncode=returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        timed_out=timed_out,
        stdout_limit_exceeded=stdout_overflow.is_set(),
        stderr_limit_exceeded=stderr_overflow.is_set(),
        stdin_completed=stdin_completed.is_set(),
        supervision_error="; ".join(supervision_errors)
        if supervision_errors
        else None,
    )


def _new_external_scratch_root() -> tuple[Path, Path, Path]:
    temp_root = Path(tempfile.mkdtemp(prefix="skillchain-codex-author-")).resolve(
        strict=True
    )
    repository = ROOT.resolve(strict=True)
    if temp_root == repository or repository in temp_root.parents:
        shutil.rmtree(temp_root)
        raise CodexAuthoringContractError(
            "Codex system-temp root unexpectedly resides in the repository"
        )
    scratch = temp_root / "scratch"
    control = temp_root / "control"
    scratch.mkdir()
    control.mkdir()
    if any(scratch.iterdir()):
        shutil.rmtree(temp_root)
        raise CodexAuthoringContractError("Codex external scratch is not empty")
    return temp_root, scratch, control


def _approval(
    approval_path: Path,
    approval_file_sha256: str,
    freeze: dict[str, Any],
    freeze_file_sha256: str,
) -> dict[str, Any]:
    raw = _load_object(approval_path, approval_file_sha256, "owner approval")
    _self_hash(raw, "approval_payload_sha256", "owner approval")
    required = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze.get("freeze_payload_sha256"),
        "candidate_id": freeze.get("candidate_id"),
        "run_id": CODEX_AUTHOR_RUN_ID,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "accepted_evidence_tier": "platform-mediated_non-provider-attested",
        "one_means_one_codex_exec_session": True,
        "backend_attempt_count_unobservable": True,
        "input_isolation_not_mechanically_proven": True,
        "invalid_output_permanently_consumes_authorization": True,
        "s1_must_use_same_surface_model_effort_and_budget": True,
    }
    if set(raw) != set(required) | {"approval_payload_sha256"}:
        raise CodexAuthoringContractError(
            "owner approval contains missing or unexpected fields"
        )
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CodexAuthoringContractError(
                f"owner approval does not accept frozen field: {key}"
            )
    return raw


def _command(
    executable: Path,
    *,
    scratch: Path,
    schema: Path,
    final_message: Path,
) -> list[str]:
    return [
        str(executable),
        "exec",
        "--model",
        "gpt-5.6-sol",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--cd",
        str(scratch),
        "--output-schema",
        str(schema),
        "--json",
        "--output-last-message",
        str(final_message),
        "--config",
        'model_reasoning_effort="high"',
        "-",
    ]


def _normalized_command(
    command: list[str],
    *,
    executable: Path,
    scratch: Path,
    schema: Path,
    final_message: Path,
) -> list[str]:
    return [
        "codex.exe"
        if value == str(executable)
        else "<external-empty-scratch>"
        if value == str(scratch)
        else "<frozen-schema>"
        if value == str(schema)
        else "<create-only-final>"
        if value == str(final_message)
        else value
        for value in command
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock-file-sha256", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--approval-record-file-sha256", required=True)
    args = parser.parse_args()

    freeze_file_sha256 = args.freeze_lock_file_sha256
    freeze = _load_object(FREEZE_PATH, freeze_file_sha256, "Codex freeze lock")
    _self_hash(freeze, "freeze_payload_sha256", "Codex freeze lock")
    if (
        freeze.get("status") != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id")
        != "authoring-codex-high-20260724-v2"
        or freeze.get("invocation_authorized") is not False
    ):
        raise CodexAuthoringContractError("Codex freeze lock is not an approvable candidate")

    required_binding_names = (
        "model_role_selection",
        "protocol",
        "model_access_evidence",
        "prompt_isolation_evidence",
        "semantic_source",
        "authoring_input",
        "output_schema",
        "canonical_request",
        "stdin_request",
        "runtime_lock",
        "runner",
        "contract_module",
        "builder",
        "approval_builder",
        "source_manifest",
    )
    freeze_bindings = {
        name: _binding(freeze, name) for name in required_binding_names
    }
    packet_path, packet_sha256 = freeze_bindings["authoring_input"]
    semantic_path, semantic_sha256 = freeze_bindings["semantic_source"]
    request_path, request_sha256 = freeze_bindings["canonical_request"]
    stdin_path, stdin_sha256 = freeze_bindings["stdin_request"]
    schema_path, schema_sha256 = freeze_bindings["output_schema"]
    runtime_path, runtime_file_sha256 = freeze_bindings["runtime_lock"]
    runtime = _load_object(
        runtime_path,
        runtime_file_sha256,
        "Codex runtime lock",
    )
    _self_hash(runtime, "runtime_payload_sha256", "Codex runtime lock")
    if runtime.get("status") != "frozen":
        raise CodexAuthoringContractError("Codex runtime lock is not frozen")
    _validate_frozen_source_bindings(freeze_bindings, runtime)
    source_manifest_path, source_manifest_file_sha256 = freeze_bindings[
        "source_manifest"
    ]
    source_manifest = _load_object(
        source_manifest_path,
        source_manifest_file_sha256,
        "Codex source manifest",
    )
    _validate_dependency_snapshot(source_manifest)
    prompt_isolation_path, prompt_isolation_file_sha256 = freeze_bindings[
        "prompt_isolation_evidence"
    ]
    prompt_isolation = _load_object(
        prompt_isolation_path,
        prompt_isolation_file_sha256,
        "Codex prompt-isolation evidence",
    )
    _self_hash(
        prompt_isolation,
        "evidence_payload_sha256",
        "Codex prompt-isolation evidence",
    )
    prompt_probe = prompt_isolation.get("probe")
    if (
        prompt_isolation.get("status")
        != "verified_no_project_instruction_markers_from_external_working_directory"
        or not isinstance(prompt_probe, dict)
        or prompt_probe.get("inference_requested") is not False
        or prompt_probe.get("working_directory_policy")
        != "fresh_system_temp_directory_outside_repository"
        or prompt_probe.get("project_markers")
        != {
            "ARIS Skill Scope": False,
            "Interview-oriented project memory": False,
        }
    ):
        raise CodexAuthoringContractError(
            "Codex prompt-isolation evidence is not an accepted negative probe"
        )

    authoring_input = load_codex_authoring_input(
        packet_path,
        expected_file_sha256=packet_sha256,
    )
    semantic_source = load_authoring_packet(
        semantic_path,
        expected_file_sha256=semantic_sha256,
    )
    request_bytes = _read_bound_bytes(
        request_path,
        request_sha256,
        "Codex canonical request",
    )
    stdin_bytes = _read_bound_bytes(
        stdin_path,
        stdin_sha256,
        "Codex stdin request",
    )
    schema_bytes = _read_bound_bytes(
        schema_path,
        schema_sha256,
        "Codex output schema",
    )
    request = parse_codex_authoring_request(request_bytes)
    try:
        raw_schema = parse_canonical_json(
            schema_bytes,
            label="Codex output schema",
        )
    except ArtifactFormatError as error:
        raise CodexAuthoringContractError(
            "Codex output schema is not canonical JSON"
        ) from error
    if not isinstance(raw_schema, dict):
        raise CodexAuthoringContractError(
            "Codex output schema must contain an object"
        )
    validate_codex_cli_output_schema(raw_schema)
    if (
        authoring_input.model.requested_model != "gpt-5.6-sol"
        or authoring_input.model.reasoning_effort != "high"
        or authoring_input.session_budget.max_exec_sessions != 1
        or request.authoring_input != authoring_input
        or request.output_contract.json_schema_canonical_json.encode("utf-8")
        != schema_bytes
        or render_codex_authoring_stdin(request) != stdin_bytes
    ):
        raise CodexAuthoringContractError(
            "Codex packet/request/schema/stdin or budget drifted"
        )

    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    if not isinstance(paths, dict):
        raise CodexAuthoringContractError("Codex authorization paths are missing")
    if paths.get("run_id") != CODEX_AUTHOR_RUN_ID:
        raise CodexAuthoringContractError("Codex authorization run ID drifted")
    expected_approval_path = _repo_path(
        paths.get("approval_record_file"),
        "frozen owner approval",
        must_exist=True,
    )
    approval_path = _repo_path(
        Path(args.approval_record).as_posix(),
        "owner approval",
        must_exist=True,
    )
    if approval_path != expected_approval_path:
        raise CodexAuthoringContractError(
            "owner approval path differs from the frozen authorization path"
        )
    approval = _approval(
        approval_path,
        args.approval_record_file_sha256,
        freeze,
        freeze_file_sha256,
    )
    output_dir = _repo_path(
        paths.get("output_directory"), "output directory", must_exist=False
    )
    claim_path = _repo_path(paths.get("claim_file"), "claim", must_exist=False)
    receipt_path = _repo_path(paths.get("receipt_file"), "receipt", must_exist=False)
    if receipt_path != output_dir / "invocation-receipt.json":
        raise CodexAuthoringContractError(
            "frozen receipt must be atomically published inside the output directory"
        )
    if any(os.path.lexists(path) for path in (output_dir, claim_path, receipt_path)):
        raise FileExistsError(
            "Codex authorization has already been consumed or its output path exists"
        )

    executable = _resolve_codex(runtime)
    safe_env, env_policy_sha256 = _safe_environment(runtime)
    staging = new_staging_directory(output_dir)
    temp_root: Path | None = None
    try:
        temp_root, scratch, control = _new_external_scratch_root()
        materialized_schema_path = control / "output-schema.json"
        atomic_create_file(materialized_schema_path, schema_bytes)
        final_message_path = staging / "author-content.raw.json"
        command = _command(
            executable,
            scratch=scratch,
            schema=materialized_schema_path,
            final_message=final_message_path,
        )
        command_shape = runtime.get("command_shape")
        normalized_command = _normalized_command(
            command,
            executable=executable,
            scratch=scratch,
            schema=materialized_schema_path,
            final_message=final_message_path,
        )
        if (
            command_shape != list(CODEX_COMMAND_SHAPE)
            or normalized_command != list(CODEX_COMMAND_SHAPE)
            or any(scratch.iterdir())
        ):
            raise CodexAuthoringContractError("Codex runtime command shape drifted")
    except Exception:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    claim_payload = {
        "schema_version": 1,
        "status": "consumed_before_codex_process_launch",
        "run_id": CODEX_AUTHOR_RUN_ID,
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "owner_approval_file": approval_path.relative_to(ROOT).as_posix(),
        "owner_approval_file_sha256": args.approval_record_file_sha256,
        "owner_approval_payload_sha256": approval["approval_payload_sha256"],
        "authoring_input_file_sha256": packet_sha256,
        "canonical_request_file_sha256": request_sha256,
        "stdin_request_file_sha256": stdin_sha256,
        "output_schema_file_sha256": schema_sha256,
        "materialized_output_schema_sha256": sha256_bytes(schema_bytes),
        "runtime_lock_file_sha256": runtime_file_sha256,
        "runtime_payload_sha256": runtime["runtime_payload_sha256"],
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "prompt_isolation_evidence_file_sha256": (
            prompt_isolation_file_sha256
        ),
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "max_codex_exec_sessions": 1,
        "retry_fallback_repair_followup_policy": "forbidden",
        "required_output_directory": output_dir.relative_to(ROOT).as_posix(),
        "required_receipt_file": receipt_path.relative_to(ROOT).as_posix(),
    }
    claim_payload_sha256 = sha256_bytes(canonical_json_bytes(claim_payload))
    claim_bytes = canonical_json_bytes(
        {
            **claim_payload,
            "claim_payload_sha256": claim_payload_sha256,
        }
    )
    try:
        atomic_create_file(claim_path, claim_bytes)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    started = time.perf_counter_ns()
    exit_code: int | None = None
    stdout = b""
    stderr = b""
    status = "codex_process_failed"
    error_type: str | None = None
    error_message: str | None = None
    draft_sha256: str | None = None
    raw_final_sha256: str | None = None
    audit: CodexEventAudit | None = None
    event_audit_error: str | None = None
    timed_out = False
    stdout_limit_exceeded = False
    stderr_limit_exceeded = False
    process_launched = False
    stdin_completed = False
    supervision_error: str | None = None
    failure_stage: str | None = "process_spawn"
    scratch_cleanup_succeeded = False
    scratch_cleanup_error: str | None = None
    process_spawned = threading.Event()
    try:
        completed = _run_capped_process(
            command,
            cwd=scratch,
            environment=safe_env,
            stdin=stdin_bytes,
            timeout_seconds=authoring_input.session_budget.timeout_seconds,
            stdout_limit=authoring_input.session_budget.max_event_log_bytes,
            stderr_limit=authoring_input.session_budget.max_stderr_bytes,
            spawned=process_spawned,
        )
        process_launched = process_spawned.is_set()
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = completed.timed_out
        stdout_limit_exceeded = completed.stdout_limit_exceeded
        stderr_limit_exceeded = completed.stderr_limit_exceeded
        stdin_completed = completed.stdin_completed
        supervision_error = completed.supervision_error
        failure_stage = "process_result"
        try:
            audit = _event_audit(stdout)
        except Exception as event_error:
            event_audit_error = (
                f"{type(event_error).__name__}: {event_error}"
            )
        if timed_out:
            raise CodexAuthoringContractError("Codex author process timed out")
        if stdout_limit_exceeded or stderr_limit_exceeded:
            raise CodexAuthoringContractError(
                "Codex author process exceeded a frozen output byte limit"
            )
        if exit_code != 0:
            raise CodexAuthoringContractError(
                f"Codex author process exited with {exit_code}"
            )
        if supervision_error is not None or not stdin_completed:
            raise CodexAuthoringContractError(
                "Codex process supervision failed: "
                + (supervision_error or "stdin did not complete")
            )
        failure_stage = "event_audit"
        if event_audit_error is not None or audit is None:
            raise CodexAuthoringContractError(
                f"Codex event audit failed: {event_audit_error}"
            )
        audit.require_formal_success()
        failure_stage = "output_capture"
        if not final_message_path.is_file() or final_message_path.is_symlink():
            raise CodexAuthoringContractError("Codex final message is missing")
        raw_final = read_stable_regular_file(
            final_message_path,
            label="Codex final message",
            max_bytes=authoring_input.session_budget.max_final_output_bytes,
        )
        raw_final_sha256 = sha256_bytes(raw_final)
        agent_message_bytes = audit.agent_messages[0].encode("utf-8")
        if raw_final not in {
            agent_message_bytes,
            agent_message_bytes + b"\n",
        }:
            raise CodexAuthoringContractError(
                "Codex JSONL final message differs from output-last-message"
            )
        failure_stage = "trusted_compiler"
        draft = normalize_codex_authoring_output(
            raw_final,
            authoring_input=authoring_input,
            semantic_source=semantic_source,
        )
        draft_bytes = draft.canonical_bytes()
        draft_sha256 = sha256_bytes(draft_bytes)
        atomic_create_file(staging / "pre-review-draft.json", draft_bytes)
        status = "codex_session_completed_draft_ready_for_review"
        failure_stage = None
    except Exception as error:  # claim is already consumed; preserve every failure.
        process_launched = process_spawned.is_set()
        error_type = type(error).__name__
        error_message = str(error)
        if not process_launched:
            status = "codex_process_launch_failed"
        elif timed_out:
            status = "codex_process_timed_out"
        elif stdout_limit_exceeded or stderr_limit_exceeded:
            status = "codex_process_output_limit_exceeded"
        elif exit_code not in (None, 0):
            status = "codex_process_nonzero_exit"
        elif supervision_error is not None or not stdin_completed:
            status = "codex_process_supervision_failed"
        elif audit is None:
            status = "codex_session_event_audit_rejected"
        elif audit.turn_completed_count != 1:
            status = "codex_session_incomplete"
        else:
            status = "codex_session_completed_draft_rejected"
    finally:
        try:
            shutil.rmtree(temp_root)
            scratch_cleanup_succeeded = True
        except Exception as cleanup_error:
            scratch_cleanup_error = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
        atomic_create_file(staging / "codex-events.jsonl", stdout)
        atomic_create_file(staging / "codex-stderr.bin", stderr)
        atomic_create_file(staging / "authoring-request.json", request_bytes)
        atomic_create_file(staging / "authoring-stdin.txt", stdin_bytes)
        atomic_create_file(staging / "authoring-output-schema.json", schema_bytes)
        evidence = {
            "schema_version": 2,
            "run_id": CODEX_AUTHOR_RUN_ID,
            "status": status,
            "execution_type": "codex_mediated_static_author_v1",
            "evidence_tier": "platform-mediated_non-provider-attested",
            "formal_codex_session_eligible": status
            == "codex_session_completed_draft_ready_for_review",
            "formal_provider_call_eligible": False,
            "authorization_consumed": True,
            "requested_model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "served_model": None,
            "served_revision": None,
            "provider_request_id": None,
            "backend_attempt_count": None,
            "backend_attempt_count_evidence": "unobservable",
            "thread_id": audit.thread_id if audit else None,
            "input_tokens": audit.input_tokens if audit else None,
            "cached_input_tokens": (
                audit.cached_input_tokens if audit else None
            ),
            "output_tokens": audit.output_tokens if audit else None,
            "cost_microusd": None,
            "token_evidence": (
                "platform_reported_non_provider_attested"
                if audit and audit.input_tokens is not None
                else "unavailable"
            ),
            "cost_evidence": "unavailable_on_subscription_cli",
            "repository_retry_performed": False,
            "followup_performed": False,
            "repair_performed": False,
            "fallback_performed": False,
            "visible_tool_activity": (
                audit.visible_tool_activity if audit else None
            ),
            "visible_agent_message_count": (
                len(audit.agent_messages) if audit else None
            ),
            "disallowed_item_types": (
                list(audit.disallowed_item_types) if audit else []
            ),
            "input_isolation": (
                "behaviorally_constrained_not_mechanically_proven"
            ),
            "repository_working_directory_excluded": True,
            "external_scratch_was_empty_before_launch": True,
            "process_launched": process_launched,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_limit_exceeded": stdout_limit_exceeded,
            "stderr_limit_exceeded": stderr_limit_exceeded,
            "stdin_completed": stdin_completed,
            "process_supervision_error": supervision_error,
            "elapsed_ms": elapsed_ms,
            "event_types": list(audit.event_types) if audit else [],
            "event_audit_error": event_audit_error,
            "event_log_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
            "raw_final_sha256": raw_final_sha256,
            "pre_review_draft_sha256": draft_sha256,
            "error_type": error_type,
            "error_message": error_message,
            "failure_stage": failure_stage,
            "scratch_cleanup_succeeded": scratch_cleanup_succeeded,
            "scratch_cleanup_error": scratch_cleanup_error,
            "freeze_lock_file_sha256": freeze_file_sha256,
            "freeze_payload_sha256": freeze["freeze_payload_sha256"],
            "owner_approval_file": approval_path.relative_to(ROOT).as_posix(),
            "owner_approval_file_sha256": args.approval_record_file_sha256,
            "owner_approval_payload_sha256": approval[
                "approval_payload_sha256"
            ],
            "attempt_claim_file_sha256": sha256_bytes(claim_bytes),
            "attempt_claim_payload_sha256": claim_payload_sha256,
            "authoring_input_file_sha256": packet_sha256,
            "canonical_request_file_sha256": request_sha256,
            "stdin_request_file_sha256": stdin_sha256,
            "output_schema_file_sha256": schema_sha256,
            "runtime_lock_file_sha256": runtime_file_sha256,
            "runtime_payload_sha256": runtime["runtime_payload_sha256"],
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "prompt_isolation_evidence_file_sha256": (
                prompt_isolation_file_sha256
            ),
            "environment_policy_sha256": env_policy_sha256,
            "command_sha256": sha256_bytes(
                canonical_json_bytes(normalized_command)
            ),
        }
        evidence_bytes = canonical_json_bytes(
            {
                **evidence,
                "receipt_payload_sha256": sha256_bytes(
                    canonical_json_bytes(evidence)
                ),
            }
        )
        atomic_create_file(staging / "invocation-receipt.json", evidence_bytes)
        atomic_publish_new_directory(staging, output_dir)

    print(
        json.dumps(
            {
                "status": status,
                "run_id": CODEX_AUTHOR_RUN_ID,
                "authorization_consumed": True,
                "receipt": receipt_path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "codex_session_completed_draft_ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
